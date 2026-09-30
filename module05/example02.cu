// =====================================================================
// Module 5 / Example 2 : "AoS vs SoA -- and the condition under which
//                         AoS is actually fine"
//
// The folklore is "AoS bad, SoA good". The truth is sharper, and this
// example measures both halves of it.
//
//   PART 1 -- touch ONE field of the struct.
//       AoS: a warp reading p[i].x issues 32 addresses 24 B apart. They
//       span 768 B = 24 sectors, of which it uses 128 B. 16.7% efficient.
//       SoA: 4 sectors, 100%. Expect a large gap.
//
//   PART 2 -- touch EVERY field of the struct.
//       AoS: the warp's six loads together cover the same 768 B / 24
//       sectors, and now every byte is used. DRAM traffic is identical
//       to SoA. The remaining gap is instruction/request overhead and
//       partial-sector writes, not wasted bandwidth. Expect a SMALL gap.
//
//   PART 3 -- SoA + float4, on the Part-2 workload: same sectors, a
//       quarter of the load instructions (one LDG.E.128 per 4 elements).
//
// Everything is validated against a CPU reference.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN:    .\example02.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK(x) do {                                                      \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                  \
                cudaGetErrorName(e_), __FILE__, __LINE__,                  \
                cudaGetErrorString(e_));                                   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

static const double PEAK_GBS = 432.0;
static const long long N  = 24LL * 1024 * 1024;  // 25,165,824 particles.
// Sized so that ONE SoA field (96 MB) already exceeds the 48 MB L2. At
// N = 8M a single field is 32 MB, fits in L2, and the SoA kernel reports
// >1300 GB/s -- three times DRAM peak. That number is real; it is just
// not a DRAM measurement. Always check your buffer against L2 first.
static const float     DT = 0.125f;

struct Particle { float x, y, z, vx, vy, vz; };   // 24 B

// ---- PART 1 kernels: touch one field only (x *= 2) ------------------
__global__ void scale_x_aos(Particle* p, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i].x *= 2.0f;        // stride 24 B -> 24 sectors/warp
}
__global__ void scale_x_soa(float* __restrict__ x, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.0f;          // stride 4 B -> 4 sectors/warp
}

// ---- PART 2 kernels: touch every field (pos += vel*dt) --------------
__global__ void update_aos(Particle* p, long long n, float dt)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle q = p[i];                // compiler emits 24 B of loads
    q.x += q.vx * dt; q.y += q.vy * dt; q.z += q.vz * dt;
    p[i] = q;
}
__global__ void update_soa(float* __restrict__ x, float* __restrict__ y,
                           float* __restrict__ z,
                           const float* __restrict__ vx,
                           const float* __restrict__ vy,
                           const float* __restrict__ vz,
                           long long n, float dt)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

// ---- PART 3: SoA + 128-bit loads ------------------------------------
// Thread i owns particles 4i..4i+3. The reinterpret_cast is legal only
// because cudaMalloc returns >=256 B-aligned pointers and we offset by 0.
__global__ void update_soa_vec4(float* __restrict__ x, float* __restrict__ y,
                                float* __restrict__ z,
                                const float* __restrict__ vx,
                                const float* __restrict__ vy,
                                const float* __restrict__ vz,
                                long long nVec, float dt)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nVec) return;

    float4*       x4  = reinterpret_cast<float4*>(x);
    float4*       y4  = reinterpret_cast<float4*>(y);
    float4*       z4  = reinterpret_cast<float4*>(z);
    const float4* vx4 = reinterpret_cast<const float4*>(vx);
    const float4* vy4 = reinterpret_cast<const float4*>(vy);
    const float4* vz4 = reinterpret_cast<const float4*>(vz);

    float4 a, b;
    a = x4[i]; b = vx4[i];
    a.x += b.x*dt; a.y += b.y*dt; a.z += b.z*dt; a.w += b.w*dt; x4[i] = a;
    a = y4[i]; b = vy4[i];
    a.x += b.x*dt; a.y += b.y*dt; a.z += b.z*dt; a.w += b.w*dt; y4[i] = a;
    a = z4[i]; b = vz4[i];
    a.x += b.x*dt; a.y += b.y*dt; a.z += b.z*dt; a.w += b.w*dt; z4[i] = a;
}

// ---------------------------------------------------------------------
static float initv(long long i, int c) { return (float)(((i*7 + c*13) % 101) - 50) * 0.01f; }

// Duration-based warm-up. A fixed iteration count is not enough: any
// host-side gap (a 570 MB gather loop, say) lets the memory controller drop
// to a lower P-state, and the next kernel then measures 2-3x slow. Keep
// launching until at least WARM_MS of GPU time has passed, then time.
template <typename Launch>
static double time_ms(Launch launch)
{
    const float WARM_MS = 300.0f;
    cudaEvent_t b, e;
    CHECK(cudaEventCreate(&b)); CHECK(cudaEventCreate(&e));
    float warm = 0.0f;
    while (warm < WARM_MS) {
        CHECK(cudaEventRecord(b));
        for (int i = 0; i < 5; ++i) launch();
        CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
        CHECK(cudaGetLastError());
        float dt = 0; CHECK(cudaEventElapsedTime(&dt, b, e));
        warm += dt;
    }
    const int IT = 25;
    CHECK(cudaEventRecord(b));
    for (int i = 0; i < IT; ++i) launch();
    CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
    CHECK(cudaGetLastError());
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, b, e));
    CHECK(cudaEventDestroy(b)); CHECK(cudaEventDestroy(e));
    return ms / IT;
}

static void check(const float* got, const float* ref, long long n, const char* tag)
{
    long long bad = 0;
    for (long long i = 0; i < n; ++i)
        if (fabs(got[i] - ref[i]) > 1e-5f * fmaxf(1.0f, fabsf(ref[i]))) ++bad;
    printf("      %-22s %s (%lld mismatches)\n", tag, bad ? "FAIL" : "PASS", bad);
}

int main(void)
{
    CHECK(cudaSetDevice(0));
    const long long aosBytes = N * (long long)sizeof(Particle);
    printf("N = %lld particles. AoS array = %.0f MB, one SoA field = %.0f MB.\n",
           N, aosBytes/(1024.0*1024.0), N*4.0/(1024.0*1024.0));
    printf("L2 is 48 MB, so every buffer here is comfortably DRAM-resident.\n\n");

    Particle* h_p   = (Particle*)malloc(aosBytes);
    float*    h_out = (float*)malloc(N * sizeof(float));
    float*    h_ref = (float*)malloc(N * sizeof(float));
    float*    h_tmp = (float*)malloc(N * sizeof(float));
    for (long long i = 0; i < N; ++i) {
        h_p[i].x  = initv(i,0); h_p[i].y  = initv(i,1); h_p[i].z  = initv(i,2);
        h_p[i].vx = initv(i,3); h_p[i].vy = initv(i,4); h_p[i].vz = initv(i,5);
    }

    const int TPB = 256;
    const int GRID_N   = (int)((N + TPB - 1) / TPB);
    const long long nVec = N / 4;                 // N divisible by 4 here
    const int GRID_V   = (int)((nVec + TPB - 1) / TPB);

    Particle* d_p;  CHECK(cudaMalloc(&d_p,  aosBytes));
    Particle* d_p0; CHECK(cudaMalloc(&d_p0, aosBytes));
    float *d_x0,*d_y0,*d_z0;
    CHECK(cudaMalloc(&d_x0, N*sizeof(float)));
    CHECK(cudaMalloc(&d_y0, N*sizeof(float)));
    CHECK(cudaMalloc(&d_z0, N*sizeof(float)));
    float *d_x,*d_y,*d_z,*d_vx,*d_vy,*d_vz;
    CHECK(cudaMalloc(&d_x , N*sizeof(float))); CHECK(cudaMalloc(&d_y , N*sizeof(float)));
    CHECK(cudaMalloc(&d_z , N*sizeof(float))); CHECK(cudaMalloc(&d_vx, N*sizeof(float)));
    CHECK(cudaMalloc(&d_vy, N*sizeof(float))); CHECK(cudaMalloc(&d_vz, N*sizeof(float)));

    #define FILL_SOA(dst, field) do {                                          \
        for (long long i_ = 0; i_ < N; ++i_) h_tmp[i_] = h_p[i_].field;        \
        CHECK(cudaMemcpy(dst, h_tmp, N*sizeof(float), cudaMemcpyHostToDevice));\
    } while (0)

    FILL_SOA(d_vx, vx); FILL_SOA(d_vy, vy); FILL_SOA(d_vz, vz);
    FILL_SOA(d_x0, x);  FILL_SOA(d_y0, y);  FILL_SOA(d_z0, z);
    CHECK(cudaMemcpy(d_p0, h_p, aosBytes, cudaMemcpyHostToDevice));

    // Restoring state must be a device-to-device copy. Doing it from the
    // host stalls the GPU long enough to drop its clocks, and the next
    // measurement is then a measurement of the power manager.
    #define RESET_AOS() CHECK(cudaMemcpy(d_p, d_p0, aosBytes, cudaMemcpyDeviceToDevice))
    #define RESET_SOA() do {                                                 \
        CHECK(cudaMemcpy(d_x, d_x0, N*sizeof(float), cudaMemcpyDeviceToDevice)); \
        CHECK(cudaMemcpy(d_y, d_y0, N*sizeof(float), cudaMemcpyDeviceToDevice)); \
        CHECK(cudaMemcpy(d_z, d_z0, N*sizeof(float), cudaMemcpyDeviceToDevice)); \
    } while (0)

    RESET_AOS();

    // =================== PART 1: one field =========================
    printf("=== PART 1: touch ONE field (x *= 2). Useful bytes = 8/particle ===\n");
    printf("  %-18s %9s %10s %9s %9s\n", "version", "ms", "effGB/s", "%ofpeak", "model");
    {
        const long long useful = 8LL * N;
        for (long long i = 0; i < N; ++i) h_ref[i] = h_p[i].x * 2.0f;

        RESET_AOS();
        double ms = time_ms([&]{ scale_x_aos<<<GRID_N, TPB>>>(d_p, N); });
        RESET_AOS();
        scale_x_aos<<<GRID_N, TPB>>>(d_p, N);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        Particle* hb = (Particle*)malloc(aosBytes);
        CHECK(cudaMemcpy(hb, d_p, aosBytes, cudaMemcpyDeviceToHost));
        for (long long i = 0; i < N; ++i) h_out[i] = hb[i].x;
        free(hb);
        double g = useful/(ms*1e-3)/1e9;
        printf("  %-18s %9.3f %10.1f %8.1f%% %8.1f%%\n", "AoS  p[i].x", ms, g,
               100*g/PEAK_GBS, 100.0*128.0/(24.0*32.0));
        check(h_out, h_ref, N, "AoS one-field");

        RESET_SOA();
        ms = time_ms([&]{ scale_x_soa<<<GRID_N, TPB>>>(d_x, N); });
        RESET_SOA();
        scale_x_soa<<<GRID_N, TPB>>>(d_x, N);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out, d_x, N*sizeof(float), cudaMemcpyDeviceToHost));
        g = useful/(ms*1e-3)/1e9;
        printf("  %-18s %9.3f %10.1f %8.1f%% %8.1f%%\n", "SoA  x[i]", ms, g,
               100*g/PEAK_GBS, 100.0);
        check(h_out, h_ref, N, "SoA one-field");
    }

    // =================== PART 2/3: every field =====================
    printf("\n=== PART 2/3: touch EVERY field (pos += vel*dt). Useful = 36 B/particle ===\n");
    printf("  %-18s %9s %10s %9s\n", "version", "ms", "effGB/s", "%ofpeak");
    {
        const long long useful = 36LL * N;
        for (long long i = 0; i < N; ++i) h_ref[i] = h_p[i].x + h_p[i].vx * DT;

        RESET_AOS();
        double ms = time_ms([&]{ update_aos<<<GRID_N, TPB>>>(d_p, N, DT); });
        RESET_AOS();
        update_aos<<<GRID_N, TPB>>>(d_p, N, DT);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        Particle* hb = (Particle*)malloc(aosBytes);
        CHECK(cudaMemcpy(hb, d_p, aosBytes, cudaMemcpyDeviceToHost));
        for (long long i = 0; i < N; ++i) h_out[i] = hb[i].x;
        free(hb);
        double g = useful/(ms*1e-3)/1e9;
        printf("  %-18s %9.3f %10.1f %8.1f%%\n", "v1 AoS", ms, g, 100*g/PEAK_GBS);
        check(h_out, h_ref, N, "v1 AoS");

        RESET_SOA();
        ms = time_ms([&]{ update_soa<<<GRID_N,TPB>>>(d_x,d_y,d_z,d_vx,d_vy,d_vz,N,DT); });
        RESET_SOA();
        update_soa<<<GRID_N,TPB>>>(d_x,d_y,d_z,d_vx,d_vy,d_vz,N,DT);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out, d_x, N*sizeof(float), cudaMemcpyDeviceToHost));
        g = useful/(ms*1e-3)/1e9;
        printf("  %-18s %9.3f %10.1f %8.1f%%\n", "v2 SoA", ms, g, 100*g/PEAK_GBS);
        check(h_out, h_ref, N, "v2 SoA");

        RESET_SOA();
        ms = time_ms([&]{ update_soa_vec4<<<GRID_V,TPB>>>(d_x,d_y,d_z,d_vx,d_vy,d_vz,nVec,DT); });
        RESET_SOA();
        update_soa_vec4<<<GRID_V,TPB>>>(d_x,d_y,d_z,d_vx,d_vy,d_vz,nVec,DT);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out, d_x, N*sizeof(float), cudaMemcpyDeviceToHost));
        g = useful/(ms*1e-3)/1e9;
        printf("  %-18s %9.3f %10.1f %8.1f%%\n", "v3 SoA+float4", ms, g, 100*g/PEAK_GBS);
        check(h_out, h_ref, N, "v3 SoA+float4");
    }

    printf("\n  Read the two tables together. AoS is not slow because it is a\n"
           "  struct; it is slow when a warp uses a small FRACTION of each\n"
           "  sector it forces DRAM to deliver. Touch all 24 B of every\n"
           "  Particle and the same layout moves the same bytes as SoA.\n");

    free(h_tmp); free(h_p); free(h_out); free(h_ref);
    CHECK(cudaFree(d_p));  CHECK(cudaFree(d_p0));
    CHECK(cudaFree(d_x0)); CHECK(cudaFree(d_y0)); CHECK(cudaFree(d_z0));
    CHECK(cudaFree(d_x));  CHECK(cudaFree(d_y));  CHECK(cudaFree(d_z));
    CHECK(cudaFree(d_vx)); CHECK(cudaFree(d_vy)); CHECK(cudaFree(d_vz));
    CHECK(cudaDeviceReset());
    return 0;
}
