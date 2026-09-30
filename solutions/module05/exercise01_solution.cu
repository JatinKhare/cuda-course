// =====================================================================
// Module 5 / Exercise 1 -- SOLUTION
//   "Count the sectors before you run the kernel"
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
// RUN:    .\exercise01_solution.exe
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

static const double PEAK_GBS   = 432.0;
static const int    SECTOR_B   = 32;
static const int    WARP       = 32;

// ================= PART A : the sector model =========================

enum Pattern { P_CONTIG, P_OFFSET, P_STRIDE, P_AOS_X, P_NUM };

// Element index touched by lane `lane` of warp 0.
static long long elem_index(Pattern p, int lane, long long param)
{
    switch (p) {
        case P_CONTIG: return lane;
        case P_OFFSET: return lane + param;          // param = k, in floats
        case P_STRIDE: return (long long)lane * param;
        case P_AOS_X:  return (long long)lane * 6;   // .x of Particle[24 B]
        default:       return lane;
    }
}

// --- TODO 1 (solved) -------------------------------------------------
static unsigned long long sector_of(uintptr_t addr)
{
    return (unsigned long long)(addr / SECTOR_B);   // == addr >> 5
}

// --- TODO 2 (solved) -------------------------------------------------
static int warp0_distinct_sectors(uintptr_t base, Pattern p, long long param,
                                  unsigned long long* out, int outCap)
{
    unsigned long long seen[64];
    int n = 0;
    for (int lane = 0; lane < WARP; ++lane) {
        uintptr_t a = base + (uintptr_t)(elem_index(p, lane, param) * sizeof(float));
        unsigned long long s = sector_of(a);
        bool dup = false;
        for (int k = 0; k < n; ++k) if (seen[k] == s) { dup = true; break; }
        if (!dup) { if (n < 64) seen[n] = s; ++n; }
    }
    for (int k = 0; k < n && k < outCap; ++k) out[k] = seen[k];
    return n;
}

// --- TODO 3 (solved) -------------------------------------------------
static double efficiency(long long requestedBytes, int distinctSectors)
{
    return (double)requestedBytes / (double)(SECTOR_B * distinctSectors);
}

// ================= PART B : the kernels ==============================

__global__ void k_pattern(float* a, long long nThreads, int pattern, long long param)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nThreads) return;
    long long i;
    switch (pattern) {
        case P_CONTIG: i = t;              break;
        case P_OFFSET: i = t + param;      break;
        case P_STRIDE: i = t * param;      break;
        default:       i = t * 6;          break;   // P_AOS_X
    }
    a[i] = a[i] * 2.0f + 1.0f;
}

// float4 version, used only to probe alignment (TODO 4).
__global__ void k_vec4(float* a, long long nVec)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nVec) return;
    float4* a4 = reinterpret_cast<float4*>(a);
    float4 v = a4[t];
    v.x = v.x*2.0f+1.0f; v.y = v.y*2.0f+1.0f;
    v.z = v.z*2.0f+1.0f; v.w = v.w*2.0f+1.0f;
    a4[t] = v;
}

// ---------------------------------------------------------------------
template <typename L>
static double time_ms(L launch)
{
    cudaEvent_t b, e;
    CHECK(cudaEventCreate(&b)); CHECK(cudaEventCreate(&e));
    launch(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    const int IT = 25;
    CHECK(cudaEventRecord(b));
    for (int i = 0; i < IT; ++i) launch();
    CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
    CHECK(cudaGetLastError());
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, b, e));
    CHECK(cudaEventDestroy(b)); CHECK(cudaEventDestroy(e));
    return ms / IT;
}

int main(void)
{
    CHECK(cudaSetDevice(0));

    const long long N = 64LL * 1024 * 1024;          // 256 MB >> 48 MB L2
    const long long bytes = N * sizeof(float);
    float* d = nullptr;
    CHECK(cudaMalloc(&d, bytes));
    CHECK(cudaMemset(d, 0, bytes));
    const uintptr_t base = (uintptr_t)d;

    printf("base = %p  (base %% 256 = %llu)\n\n",
           (void*)d, (unsigned long long)(base % 256));

    // -------------------- PART A --------------------
    struct Case { Pattern p; long long param; const char* label; };
    const Case cases[] = {
        { P_CONTIG, 0, "contiguous"            },
        { P_OFFSET, 1, "offset k=1 float"      },
        { P_OFFSET, 8, "offset k=8 floats"     },
        { P_STRIDE, 2, "stride s=2"            },
        { P_STRIDE, 8, "stride s=8"            },
        { P_STRIDE,32, "stride s=32"           },
        { P_AOS_X,  0, "AoS .x (stride 6)"     },
    };
    const int NCASES = (int)(sizeof(cases)/sizeof(cases[0]));

    printf("=== PART A: warp 0's footprint (computed by your TODOs) ===\n");
    printf("  %-22s %10s %10s %10s %12s\n",
           "pattern", "req.B", "sectors", "movedB", "efficiency");

    double modelEff[16];
    for (int c = 0; c < NCASES; ++c) {
        unsigned long long sec[64];
        int n = warp0_distinct_sectors(base, cases[c].p, cases[c].param, sec, 64);
        long long req = (long long)WARP * (long long)sizeof(float);
        double eff = efficiency(req, n);
        modelEff[c] = eff;
        printf("  %-22s %10lld %10d %10lld %11.1f%%\n",
               cases[c].label, req, n, (long long)n * SECTOR_B, 100.0 * eff);

        // Arithmetic self-consistency (this must hold for ANY correct answer).
        if (n < 1 || n > 32) { printf("  [FAIL] sector count out of range\n"); return 1; }
        if (fabs(eff * SECTOR_B * n - (double)req) > 1e-9) {
            printf("  [FAIL] efficiency formula inconsistent\n"); return 1;
        }
    }
    printf("  [OK] sector counts in [1,32] and efficiency formula self-consistent\n\n");

    // -------------------- TODO 4: alignment --------------------
    // float4 loads require the ADDRESS to be 16 B aligned. base is 256 B
    // aligned, so base + 4*k is 16 B aligned iff k % 4 == 0. The smallest
    // strictly positive such k is 4.
    const int k_align = 4;

    printf("=== TODO 4: float4 from base + k floats, k = %d ===\n", k_align);
    if (k_align < 0) { printf("Set TODO 4 first.\n"); return 0; }
    {
        long long nVec = (N - 64) / 4;
        k_vec4<<<(int)((nVec + 255) / 256), 256>>>(d + k_align, nVec);
        cudaError_t launchErr = cudaGetLastError();
        cudaError_t runErr    = cudaDeviceSynchronize();
        if (launchErr != cudaSuccess || runErr != cudaSuccess) {
            printf("  float4 load at +%d floats FAILED: %s / %s\n", k_align,
                   cudaGetErrorName(launchErr), cudaGetErrorName(runErr));
            printf("  (cudaErrorMisalignedAddress means k*4 is not a multiple of 16.)\n");
            return 1;
        }
        printf("  float4 load at +%d floats succeeded -> address is 16 B aligned.\n\n",
               k_align);
    }
    CHECK(cudaMemset(d, 0, bytes));

    // -------------------- PART B: measure --------------------
    const int TPB = 256;

    printf("=== PART B: measured (25 iters, 256 MB buffer) ===\n");
    printf("  %-22s %9s %10s %9s %11s %12s\n",
           "pattern", "ms", "effGB/s", "%ofpeak", "modelEff", "impliedDRAM");

    double measGBs[16];
    // Two passes; we report the second. The first pass finishes settling
    // the memory P-state. Without it, whichever case is measured FIRST is
    // penalised, which silently corrupts every ratio in the table below.
    for (int pass = 0; pass < 2; ++pass)
    for (int c = 0; c < NCASES; ++c) {
        long long span = 1;
        if (cases[c].p == P_STRIDE) span = cases[c].param;
        if (cases[c].p == P_AOS_X)  span = 6;
        long long nt = (N - 64) / span;

        double ms = time_ms([&]{
            k_pattern<<<(int)((nt + TPB - 1)/TPB), TPB>>>(
                d, nt, (int)cases[c].p, cases[c].param);
        });
        double g = (8.0 * (double)nt) / (ms * 1e-3) / 1e9;
        measGBs[c] = g;
        if (pass == 1)
        printf("  %-22s %9.3f %10.1f %8.1f%% %10.1f%% %12.1f\n",
               cases[c].label, ms, g, 100.0*g/PEAK_GBS,
               100.0*modelEff[c], g / modelEff[c]);
    }

    // -------------------- correctness --------------------
    // A plain contiguous pass, validated element-by-element against a CPU
    // reference, so the file has a real PASS/FAIL and not just timings.
    {
        const long long M = 1 << 20;
        float* h = (float*)malloc(M * sizeof(float));
        float* r = (float*)malloc(M * sizeof(float));
        for (long long i = 0; i < M; ++i) h[i] = (float)((i * 37) % 1000) * 0.001f;
        for (long long i = 0; i < M; ++i) r[i] = h[i] * 2.0f + 1.0f;
        CHECK(cudaMemcpy(d, h, M * sizeof(float), cudaMemcpyHostToDevice));
        k_pattern<<<(int)((M + TPB - 1)/TPB), TPB>>>(d, M, P_CONTIG, 0);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h, d, M * sizeof(float), cudaMemcpyDeviceToHost));
        long long bad = 0;
        for (long long i = 0; i < M; ++i)
            if (fabs(h[i] - r[i]) > 1e-5f * fmaxf(1.0f, fabsf(r[i]))) ++bad;
        printf("\n  kernel numerics: %s (%lld mismatches)\n", bad ? "FAIL" : "PASS", bad);
        free(h); free(r);
    }

    // -------------------- the check that matters --------------------
    // Relative bandwidth should track relative model efficiency, using
    // contiguous as the reference point.
    printf("\n=== Predicted vs measured, normalised to 'contiguous' ===\n");
    printf("  %-22s %14s %14s %8s\n", "pattern", "model ratio", "measured ratio", "err");
    int off = 0;
    for (int c = 0; c < NCASES; ++c) {
        double mr = modelEff[c] / modelEff[0];
        double xr = measGBs[c]  / measGBs[0];
        double err = 100.0 * (xr - mr) / mr;
        if (fabs(err) > 15.0) ++off;
        printf("  %-22s %14.3f %14.3f %7.1f%%\n", cases[c].label, mr, xr, err);
    }
    printf("  patterns where the sector model mispredicts by >15%%: %d\n", off);
    printf("  (A nonzero count is not a bug. Work out which one, and why.)\n");

    CHECK(cudaFree(d));
    CHECK(cudaDeviceReset());
    return 0;
}
