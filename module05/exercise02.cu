// =====================================================================
// Module 5 / Exercise 2 : "AoS -> SoA -> 128-bit, with a tail that
//                          refuses to divide"
//
// GOAL
//   A correct but slow particle integrator, v1, is given to you in AoS
//   layout. Produce v2 (SoA) and v3 (SoA + 128-bit loads). The harness
//   times all three over the same 858 MB of useful traffic and validates
//   every one against a CPU reference.
//
//   N = 25,000,003. That is deliberate: N % 4 == 3, so the vectorised
//   version cannot simply process N/4 float4s and stop. Getting the last
//   three particles right is worth as many marks as the fast path.
//
// BEFORE YOU START, on paper:
//   1. For v1, warp 0 executes `Particle q = p[i]`. Which byte offsets
//      does lane 0 touch? Lane 1? How many distinct 32 B sectors does the
//      whole warp's 24 B-per-lane read cover, and what fraction of those
//      bytes does the kernel use? Now answer the same question for a
//      kernel that reads ONLY p[i].x. The two answers are very different
//      and that difference is the entire AoS-vs-SoA argument.
//   2. Predict the v1:v2:v3 bandwidth ratios. Commit to numbers.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
// RUN:    .\exercise02.exe
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

// 16,000,003 is odd and 16,000,003 % 4 == 3. One SoA field is 61 MB,
// past the 48 MB L2, and the vectorised kernel is forced to deal with a
// 3-element tail.
static const long long N  = 16000003LL;
static const float     DT = 0.125f;

struct Particle { float x, y, z, vx, vy, vz; };   // 24 B

// A plain 1-read-1-write stream. This is the fastest thing this machine
// can do with this much data, and we measure it in the same run as the
// three versions under test. Reporting "% of measured stream" instead of
// only "% of 432 GB/s nominal" makes the table immune to the clock and
// power-cap drift a laptop GPU exhibits under sustained load.
__global__ void stream_ref(const float* __restrict__ in, float* __restrict__ out,
                           long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

// ============================ v1 : AoS ===============================
__global__ void update_aos(Particle* p, long long n, float dt)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle q = p[i];
    q.x += q.vx * dt; q.y += q.vy * dt; q.z += q.vz * dt;
    p[i] = q;
}

// ============================ v2 : SoA ===============================
// TODO 2: Same physics as update_aos, over six separate contiguous
//         arrays. Write it so that the 32 lanes of a warp, on every one
//         of the six memory instructions, present 32 addresses that span
//         exactly 128 contiguous bytes.
//
//         (The `__restrict__` on the parameters is already there and it
//         matters: it tells the compiler the six arrays do not alias, so
//         it may hoist all the loads ahead of all the stores. Without it
//         the compiler must assume x and vx might overlap.)
__global__ void update_soa(float* __restrict__ x, float* __restrict__ y,
                           float* __restrict__ z,
                           const float* __restrict__ vx,
                           const float* __restrict__ vy,
                           const float* __restrict__ vz,
                           long long n, float dt)
{
    (void)x; (void)y; (void)z; (void)vx; (void)vy; (void)vz; (void)n; (void)dt;
    // YOUR CODE HERE (TODO 2)
}

// ====================== v3 : SoA + 128-bit ===========================
// The grid for this kernel is sized over nVec, NOT over n. See main().
//
// TODO 3: Fast path. For i < nVec, process four consecutive particles per
//         thread using 128-bit (float4) loads and stores.
//
//         Two things to convince yourself of before writing it:
//           - why is reinterpret_cast<float4*>(x) legal here at all?
//             What would have to be true of `x` for it to be illegal,
//             and what does cudaMalloc guarantee?
//           - the warp still covers 128 B per lane-group of 8. How many
//             32 B sectors does one float4 instruction from one warp
//             touch, and how does that compare with the four scalar
//             instructions it replaces? If the sector count is the same,
//             what exactly are you buying?
//
// TODO 4: The tail. n is NOT a multiple of 4. Elements 4*nVec .. n-1 must
//         still be updated, exactly once each, by this same launch.
//
//         Be careful where you put it. The obvious placements are wrong
//         in ways the validator will catch: the tail report at the end of
//         the run counts mismatches in the last few elements separately
//         from the rest, so a tail bug shows up as a small nonzero number
//         rather than as a general failure.
__global__ void update_soa_vec4(float* __restrict__ x, float* __restrict__ y,
                                float* __restrict__ z,
                                const float* __restrict__ vx,
                                const float* __restrict__ vy,
                                const float* __restrict__ vz,
                                long long n, long long nVec, float dt)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    (void)i; (void)x; (void)y; (void)z; (void)vx; (void)vy; (void)vz;
    (void)n; (void)nVec; (void)dt;

    // YOUR CODE HERE (TODO 3)

    // YOUR CODE HERE (TODO 4)
}

// ---------------------------------------------------------------------
static float initv(long long i, int c) { return (float)(((i*7 + c*13) % 101) - 50) * 0.01f; }

// Timing with a *duration-based* warm-up. A fixed iteration count is not
// enough on a laptop GPU: any host-side gap lets the memory controller drop
// to a lower P-state, and the next kernel then measures 2-3x slow. We keep
// launching until at least WARM_MS of GPU time has elapsed, and only then
// start the clock.
template <typename L>
static double time_ms(L launch)
{
    const float WARM_MS = 150.0f;
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

    const int IT = 20;
    CHECK(cudaEventRecord(b));
    for (int i = 0; i < IT; ++i) launch();
    CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
    CHECK(cudaGetLastError());
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, b, e));
    CHECK(cudaEventDestroy(b)); CHECK(cudaEventDestroy(e));
    return ms / IT;
}

static long long mism(const float* got, const float* ref, long long n)
{
    long long bad = 0;
    for (long long i = 0; i < n; ++i)
        if (fabs(got[i] - ref[i]) > 1e-5f * fmaxf(1.0f, fabsf(ref[i]))) ++bad;
    return bad;
}

int main(void)
{
    CHECK(cudaSetDevice(0));

    const long long aosBytes = N * (long long)sizeof(Particle);
    const long long fldBytes = N * (long long)sizeof(float);
    const long long useful   = 36LL * N;     // read 6 floats, write 3

    printf("N = %lld  (N %% 4 = %lld)\n", N, N % 4);
    printf("AoS array %.0f MB, one SoA field %.0f MB, L2 = 48 MB\n",
           aosBytes/1048576.0, fldBytes/1048576.0);
    printf("useful traffic / launch = %.0f MB\n\n", useful/1048576.0);

    Particle* h_p   = (Particle*)malloc(aosBytes);
    float*    h_tmp = (float*)malloc(fldBytes);
    float*    h_got = (float*)malloc(fldBytes);
    float*    h_rx  = (float*)malloc(fldBytes);
    float*    h_ry  = (float*)malloc(fldBytes);
    float*    h_rz  = (float*)malloc(fldBytes);
    if (!h_p || !h_tmp || !h_got || !h_rx || !h_ry || !h_rz) {
        printf("host allocation failed\n"); return 1;
    }
    for (long long i = 0; i < N; ++i) {
        h_p[i].x  = initv(i,0); h_p[i].y  = initv(i,1); h_p[i].z  = initv(i,2);
        h_p[i].vx = initv(i,3); h_p[i].vy = initv(i,4); h_p[i].vz = initv(i,5);
    }
    // CPU reference
    for (long long i = 0; i < N; ++i) {
        h_rx[i] = h_p[i].x + h_p[i].vx * DT;
        h_ry[i] = h_p[i].y + h_p[i].vy * DT;
        h_rz[i] = h_p[i].z + h_p[i].vz * DT;
    }

    Particle* d_p; CHECK(cudaMalloc(&d_p, aosBytes));
    float *d_x,*d_y,*d_z,*d_vx,*d_vy,*d_vz;
    CHECK(cudaMalloc(&d_x , fldBytes)); CHECK(cudaMalloc(&d_y , fldBytes));
    CHECK(cudaMalloc(&d_z , fldBytes)); CHECK(cudaMalloc(&d_vx, fldBytes));
    CHECK(cudaMalloc(&d_vy, fldBytes)); CHECK(cudaMalloc(&d_vz, fldBytes));

    // Pristine device-side copies of the initial state. Restoring from these
    // is a device-to-device copy; restoring from the host would stall the GPU
    // long enough to drop its clocks and ruin the next measurement.
    Particle* d_p0; CHECK(cudaMalloc(&d_p0, aosBytes));
    float *d_x0,*d_y0,*d_z0;
    CHECK(cudaMalloc(&d_x0, fldBytes)); CHECK(cudaMalloc(&d_y0, fldBytes));
    CHECK(cudaMalloc(&d_z0, fldBytes));

    // ---- TODO 1: the layout transformation ----
    // UPLOAD_FIELD(dst, field) must leave the device array `dst` holding
    // h_p[0].field, h_p[1].field, ... h_p[N-1].field, contiguously.
    // h_tmp is a host scratch buffer of N floats, already allocated.
    //
    // This is the price of SoA and you should be able to state it: how
    // many bytes does one invocation read on the host, and how many does
    // it send over PCIe? Multiply by six. Then ask how many timesteps the
    // SoA kernel has to run before that price is repaid. The solution
    // notes do this arithmetic; do it yourself first.
    #define UPLOAD_FIELD(dst, field) do {                                  \
        /* YOUR CODE HERE (TODO 1) */                                      \
        (void)(dst); CHECK(cudaMemset((dst), 0, fldBytes));                \
    } while (0)

    UPLOAD_FIELD(d_vx, vx); UPLOAD_FIELD(d_vy, vy); UPLOAD_FIELD(d_vz, vz);
    UPLOAD_FIELD(d_x0, x);  UPLOAD_FIELD(d_y0, y);  UPLOAD_FIELD(d_z0, z);
    CHECK(cudaMemcpy(d_p0, h_p, aosBytes, cudaMemcpyHostToDevice));

    // From here on, "reset the inputs" means these two macros only.
    #define RESET_AOS()  CHECK(cudaMemcpy(d_p, d_p0, aosBytes, cudaMemcpyDeviceToDevice))
    #define RESET_SOA()  do {                                                     \
        CHECK(cudaMemcpy(d_x, d_x0, fldBytes, cudaMemcpyDeviceToDevice));         \
        CHECK(cudaMemcpy(d_y, d_y0, fldBytes, cudaMemcpyDeviceToDevice));         \
        CHECK(cudaMemcpy(d_z, d_z0, fldBytes, cudaMemcpyDeviceToDevice));         \
    } while (0)

    const int TPB   = 256;
    const int GRID  = (int)((N + TPB - 1) / TPB);
    // The number of whole float4 groups in N. Get this wrong by one and
    // you will either skip four particles or run off the end of the array.
    const long long nVec = 0;   // YOUR CODE HERE (TODO 3)
    const int GRID_V = (int)((nVec + TPB - 1) / TPB);

    printf("nVec = %lld, 4*nVec = %lld, tail = %lld element(s)\n\n",
           nVec, 4*nVec, N - 4*nVec);
    if (nVec <= 0) {
        printf("Set nVec (TODO 3) to continue.\n");
        CHECK(cudaDeviceReset());
        return 0;
    }

    RESET_AOS();

    double streamGBs = 1.0;   // filled in below, inside the timing loop

    printf("  %-18s %9s %10s %9s %10s   %s\n",
           "version", "ms", "GB/s", "%ofpeak", "%ofstream", "validation");

    // Two passes, second one reported. The first settles the memory P-state
    // and the SM clock; on a laptop GPU whichever version is measured first
    // otherwise reads 20-40%% slow, which would invert the ranking.
    for (int pass = 0; pass < 2; ++pass) {

        // The machine's streaming ceiling, measured in the same loop and at
        // the same thermal state as the three versions under test. Reporting
        // "% of measured stream" as well as "% of 432 GB/s nominal" makes the
        // ranking immune to the clock and power-cap drift a laptop GPU shows
        // under sustained load.
        {
            double ms = time_ms([&]{ stream_ref<<<GRID,TPB>>>(d_vx, d_x, N); });
            streamGBs = (8.0 * (double)N) / (ms * 1e-3) / 1e9;
        }
        RESET_SOA();

    // ---------------- v1 AoS ----------------
    {
        double ms = time_ms([&]{ update_aos<<<GRID, TPB>>>(d_p, N, DT); });
        RESET_AOS();
        update_aos<<<GRID, TPB>>>(d_p, N, DT);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        Particle* hb = (Particle*)malloc(aosBytes);
        CHECK(cudaMemcpy(hb, d_p, aosBytes, cudaMemcpyDeviceToHost));
        long long bad = 0;
        for (long long i = 0; i < N; ++i) h_got[i] = hb[i].x; bad += mism(h_got, h_rx, N);
        for (long long i = 0; i < N; ++i) h_got[i] = hb[i].y; bad += mism(h_got, h_ry, N);
        for (long long i = 0; i < N; ++i) h_got[i] = hb[i].z; bad += mism(h_got, h_rz, N);
        free(hb);
        double g = useful/(ms*1e-3)/1e9;
        if (pass == 1)
        printf("  %-18s %9.3f %10.1f %8.1f%% %9.1f%%   %s (%lld)\n", "v1 AoS", ms, g,
               100*g/PEAK_GBS, 100*g/streamGBs, bad ? "FAIL" : "PASS", bad);
    }

    // ---------------- v2 SoA ----------------
    {
        double ms = time_ms([&]{
            update_soa<<<GRID, TPB>>>(d_x,d_y,d_z,d_vx,d_vy,d_vz,N,DT); });
        RESET_SOA();
        update_soa<<<GRID, TPB>>>(d_x,d_y,d_z,d_vx,d_vy,d_vz,N,DT);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        long long bad = 0;
        CHECK(cudaMemcpy(h_got, d_x, fldBytes, cudaMemcpyDeviceToHost)); bad += mism(h_got,h_rx,N);
        CHECK(cudaMemcpy(h_got, d_y, fldBytes, cudaMemcpyDeviceToHost)); bad += mism(h_got,h_ry,N);
        CHECK(cudaMemcpy(h_got, d_z, fldBytes, cudaMemcpyDeviceToHost)); bad += mism(h_got,h_rz,N);
        double g = useful/(ms*1e-3)/1e9;
        if (pass == 1)
        printf("  %-18s %9.3f %10.1f %8.1f%% %9.1f%%   %s (%lld)\n", "v2 SoA", ms, g,
               100*g/PEAK_GBS, 100*g/streamGBs, bad ? "FAIL" : "PASS", bad);
    }

    // ---------------- v3 SoA + float4 ----------------
    {
        double ms = time_ms([&]{
            update_soa_vec4<<<GRID_V, TPB>>>(d_x,d_y,d_z,d_vx,d_vy,d_vz,N,nVec,DT); });
        RESET_SOA();
        update_soa_vec4<<<GRID_V, TPB>>>(d_x,d_y,d_z,d_vx,d_vy,d_vz,N,nVec,DT);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        long long bad = 0, tailBad = 0;
        CHECK(cudaMemcpy(h_got, d_x, fldBytes, cudaMemcpyDeviceToHost));
        bad += mism(h_got,h_rx,N); tailBad += mism(h_got+4*nVec, h_rx+4*nVec, N-4*nVec);
        CHECK(cudaMemcpy(h_got, d_y, fldBytes, cudaMemcpyDeviceToHost));
        bad += mism(h_got,h_ry,N); tailBad += mism(h_got+4*nVec, h_ry+4*nVec, N-4*nVec);
        CHECK(cudaMemcpy(h_got, d_z, fldBytes, cudaMemcpyDeviceToHost));
        bad += mism(h_got,h_rz,N); tailBad += mism(h_got+4*nVec, h_rz+4*nVec, N-4*nVec);
        double g = useful/(ms*1e-3)/1e9;
        if (pass == 1)
        if (pass == 1)
        printf("  %-18s %9.3f %10.1f %8.1f%% %9.1f%%   %s (%lld, of which %lld in the tail)\n",
               "v3 SoA+float4", ms, g, 100*g/PEAK_GBS, 100*g/streamGBs,
               bad ? "FAIL" : "PASS", bad, tailBad);
    }

    }   // end pass loop

    printf("\n  measured streaming ceiling (1 read + 1 write) : %.1f GB/s"
           " = %.0f%% of the 432 GB/s nominal peak\n",
           streamGBs, 100.0 * streamGBs / PEAK_GBS);
    printf("  If that is well under ~85%%, this GPU is in a reduced memory\n"
           "  P-state or is power-capped, and every absolute number above is\n"
           "  scaled down with it. The %%ofstream column is not.\n"
           "  Check: nvidia-smi --query-gpu=clocks.mem,"
           "clocks_throttle_reasons.active --format=csv\n\n");

    free(h_p); free(h_tmp); free(h_got); free(h_rx); free(h_ry); free(h_rz);
    CHECK(cudaFree(d_p));  CHECK(cudaFree(d_p0));
    CHECK(cudaFree(d_x0)); CHECK(cudaFree(d_y0)); CHECK(cudaFree(d_z0));
    CHECK(cudaFree(d_x));  CHECK(cudaFree(d_y));  CHECK(cudaFree(d_z));
    CHECK(cudaFree(d_vx)); CHECK(cudaFree(d_vy)); CHECK(cudaFree(d_vz));
    CHECK(cudaDeviceReset());
    return 0;
}
