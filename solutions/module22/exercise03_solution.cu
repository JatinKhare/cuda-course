// =============================================================================
// Module 22 / Exercise 3 — SOLUTION — the critical path of a 4-kernel pipeline.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
// RUN  : exercise03_solution.exe
//
// PROFILE:
//   NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
//   "$NSYS" profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
//        -o ex03sol --stats=true --force-overwrite=true ./exercise03_solution.exe
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#define CHECK(call) do {                                                       \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        printf("CUDA error %s (%s) at %s:%d\n", cudaGetErrorName(_e),          \
               cudaGetErrorString(_e), __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

struct NvtxRange {
    explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
    ~NvtxRange()                         { nvtxRangePop(); }
    NvtxRange(const NvtxRange &)            = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
};
#define NVTX_CAT2(a, b) a##b
#define NVTX_CAT(a, b)  NVTX_CAT2(a, b)
#define NVTX_RANGE(name) NvtxRange NVTX_CAT(_nvtxScope, __LINE__)(name)

#define N       (1 << 16)       // 65536 elements = 256 KB, L2-resident
#define BLOCK      256
#define STEPS      400

// Four pipeline stages. Stage 0 carries ~4x the arithmetic of the others, so
// "optimize the biggest kernel" is the tempting answer.
#define R0         330
#define R1          85
#define R2          85
#define R3          85

__device__ __forceinline__ float chain(float x, int rep, float k)
{
    #pragma unroll 4
    for (int r = 0; r < rep; ++r) x = fmaf(x, k, 1.0f);
    return x;
}

// Four distinct __global__ functions, not one parameterised kernel, so that
// cuda_gpu_kern_sum reports four separate rows and you can see the shape of
// the pipeline in the profile.
#define STAGE_KERNEL(NAME)                                                     \
__global__ void NAME(const float *__restrict__ in, float *__restrict__ out,    \
                     int n, int rep, float k)                                  \
{                                                                              \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i < n) out[i] = chain(in[i], rep, k);                                  \
}
STAGE_KERNEL(stage0)
STAGE_KERNEL(stage1)
STAGE_KERNEL(stage2)
STAGE_KERNEL(stage3)

// The fused pipeline: identical arithmetic in identical order, one launch.
// The result is bit-identical because every intermediate was already a float
// when it went to global memory, so keeping it in a register rounds the same.
__global__ void fused(const float *__restrict__ in, float *__restrict__ out,
                      int n, int r0, int r1, int r2, int r3,
                      float k0, float k1, float k2, float k3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    x = chain(x, r0, k0);
    x = chain(x, r1, k1);
    x = chain(x, r2, k2);
    x = chain(x, r3, k3);
    out[i] = x;
}

// Used only to measure the launch floor L.
__global__ void nullKernel(float *a) { if (threadIdx.x == 1023) a[0] = 1.0f; }

static const float K0 = 1.00003f, K1 = 1.00005f, K2 = 1.00007f, K3 = 1.00011f;
static int gBlocks = (N + BLOCK - 1) / BLOCK;

// -----------------------------------------------------------------------------
// The per-launch host cost L. Enqueue a trivial kernel back-to-back with no
// synchronization and divide: the device cannot be the limit, so what is left
// is the host's cost of asking.
// -----------------------------------------------------------------------------
static double measureLaunchFloorUs(float *scratch, cudaEvent_t e0, cudaEvent_t e1)
{
    const int IT = 4000;
    double best = 1e30;
    for (int s = 0; s < 4; ++s) {
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < IT; ++i) nullKernel<<<gBlocks, BLOCK>>>(scratch);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        double us = 1000.0 * (double)ms / (double)IT;
        if (us < best) best = us;
    }
    return best;
}

// -----------------------------------------------------------------------------
// "Solo duration" of one stage, measured the only way a program can measure
// itself: a back-to-back async loop. READ THE OUTPUT OF THIS FUNCTION
// SCEPTICALLY. It is an upper bound, and for a kernel shorter than L it is a
// measurement of L and nothing else.
// -----------------------------------------------------------------------------
typedef void (*StageFn)(const float *, float *, int, int, float);

static double soloUs(StageFn fn, const float *in, float *out, int rep, float k,
                     cudaEvent_t e0, cudaEvent_t e1)
{
    const int IT = 2000;
    double best = 1e30;
    for (int s = 0; s < 4; ++s) {
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < IT; ++i) fn<<<gBlocks, BLOCK>>>(in, out, N, rep, k);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        double us = 1000.0 * (double)ms / (double)IT;
        if (us < best) best = us;
    }
    return best;
}

// =============================================================================
// The pipeline: four launches per step. The `s*` arguments scale each stage's
// rep count, so (0.5,1,1,1) is "make the longest kernel twice as fast" and
// (0.5,0.5,0.5,0.5) is "make every kernel twice as fast".
// =============================================================================
static double runStaged(const float *in, float *t0, float *t1, float *t2,
                        float *out, double s0, double s1, double s2, double s3,
                        const char *name, cudaEvent_t e0, cudaEvent_t e1)
{
    NvtxRange _r(name);
    int r0 = (int)(R0 * s0), r1 = (int)(R1 * s1);
    int r2 = (int)(R2 * s2), r3 = (int)(R3 * s3);
    CHECK(cudaEventRecord(e0));
    for (int s = 0; s < STEPS; ++s) {
        stage0<<<gBlocks, BLOCK>>>(in, t0,  N, r0, K0);
        stage1<<<gBlocks, BLOCK>>>(t0, t1,  N, r1, K1);
        stage2<<<gBlocks, BLOCK>>>(t1, t2,  N, r2, K2);
        stage3<<<gBlocks, BLOCK>>>(t2, out, N, r3, K3);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return 1000.0 * (double)ms / (double)STEPS;     // us per step
}

// =============================================================================
// TODO 4 (DESIGN) — runFused. One launch per step, bit-identical output.
// =============================================================================
static double runFused(const float *in, float *out,
                       cudaEvent_t e0, cudaEvent_t e1)
{
    NVTX_RANGE("fused");
    CHECK(cudaEventRecord(e0));
    for (int s = 0; s < STEPS; ++s) {
        fused<<<gBlocks, BLOCK>>>(in, out, N, R0, R1, R2, R3, K0, K1, K2, K3);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return 1000.0 * (double)ms / (double)STEPS;
}

// =============================================================================
// TODO 1, 2, 3, 5 — the predictions and the profile numbers.
//
// TODO 1  PRED_LAUNCH_US  — the per-launch host cost on this machine, in us.
//         Example 2 measured a 13.9 us floor in one session; a dedicated probe
//         measured 5.8-8.2 us in others, flat from 1 to 1024 blocks. Anything
//         in that band is the right order of magnitude.
//
// TODO 2  PRED_BEST       — 3, fuse.
//
// TODO 3  PRED_FUSED_BUCKET — 3 (1.8x .. 3.5x). Measured 2.14-2.36x.
//
// TODO 5  The capture range brackets the BASELINE staged configuration only,
//         so the launch count is exactly 4 * STEPS = 1600 and
//         cuda_gpu_kern_sum describes one configuration. Captured on the
//         RTX 3500 Ada with Nsight Systems 2025.6.3; tables in the solution md.
// =============================================================================
#define PRED_LAUNCH_US      10.0
#define PRED_BEST              3
#define PRED_FUSED_BUCKET      3
#define NS_LAUNCH_COUNT     1600
#define NS_KERNEL_NS     4969221.0

// -----------------------------------------------------------------------------
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    nvtxNameOsThreadA(0, "pipeline-main");

    printf("Module 22 / Exercise 3 — the critical path of a 4-kernel pipeline\n");
    printf("N = %d, %d steps, 4 launches per step (staged) / 1 (fused)\n\n",
           N, STEPS);

    float *hIn = (float *)malloc(sizeof(float) * N);
    float *hA  = (float *)malloc(sizeof(float) * N);
    float *hB  = (float *)malloc(sizeof(float) * N);
    srand(20260202);
    for (int i = 0; i < N; ++i)
        hIn[i] = 0.25f + 0.5f * ((float)rand() / (float)RAND_MAX);

    float *dIn, *t0, *t1, *t2, *dOut;
    CHECK(cudaMalloc(&dIn,  sizeof(float) * N));
    CHECK(cudaMalloc(&t0,   sizeof(float) * N));
    CHECK(cudaMalloc(&t1,   sizeof(float) * N));
    CHECK(cudaMalloc(&t2,   sizeof(float) * N));
    CHECK(cudaMalloc(&dOut, sizeof(float) * N));
    CHECK(cudaMemcpy(dIn, hIn, sizeof(float) * N, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    // Compute warm-up: this pipeline is L2-resident and FFMA-bound, so the SM
    // clock is the resource that has to be ramped (spec S12.4 corollary).
    {
        cudaEvent_t w0, w1;
        CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        CHECK(cudaEventRecord(w0));
        float el = 0.0f;
        do {
            for (int i = 0; i < 400; ++i) stage0<<<gBlocks, BLOCK>>>(dIn, t0, N, R0, K0);
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            CHECK(cudaEventElapsedTime(&el, w0, w1));
        } while (el < 1500.0f);
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    }

    double L  = measureLaunchFloorUs(t0, e0, e1);
    double d0 = soloUs(stage0, dIn, t0, R0, K0, e0, e1);
    double d1 = soloUs(stage1, dIn, t0, R1, K1, e0, e1);
    double d2 = soloUs(stage2, dIn, t0, R2, K2, e0, e1);
    double d3 = soloUs(stage3, dIn, t0, R3, K3, e0, e1);

    // All competing configurations timed back-to-back with nothing printed in
    // between (spec S12.1), baseline placed first and last so drift is visible.
    // The capture range brackets only the first baseline, so the profile
    // describes exactly one configuration and the launch count is known.
    CHECK(cudaProfilerStart());
    double base = runStaged(dIn, t0, t1, t2, dOut, 1.0, 1.0, 1.0, 1.0,
                            "staged-baseline", e0, e1);
    CHECK(cudaProfilerStop());

    double halfBig = runStaged(dIn, t0, t1, t2, dOut, 0.5, 1.0, 1.0, 1.0,
                               "staged-half-longest", e0, e1);
    double halfAll = runStaged(dIn, t0, t1, t2, dOut, 0.5, 0.5, 0.5, 0.5,
                               "staged-half-all", e0, e1);
    double fusedUs = runFused(dIn, dOut, e0, e1);
    double base2   = runStaged(dIn, t0, t1, t2, dOut, 1.0, 1.0, 1.0, 1.0,
                               "staged-baseline-2", e0, e1);
    if (base2 < base) base = base2;

    // Validation pass, after all timing (spec S12.2).
    runStaged(dIn, t0, t1, t2, dOut, 1.0, 1.0, 1.0, 1.0, "verify-staged", e0, e1);
    CHECK(cudaMemcpy(hA, dOut, sizeof(float) * N, cudaMemcpyDeviceToHost));
    runFused(dIn, dOut, e0, e1);
    CHECK(cudaMemcpy(hB, dOut, sizeof(float) * N, cudaMemcpyDeviceToHost));

    printf("measured launch floor L                 : %7.2f us / launch\n", L);
    printf("event-measured stage times (rep %4d/%d/%d/%d):\n", R0, R1, R2, R3);
    printf("                                          %7.2f %7.2f %7.2f %7.2f us\n",
           d0, d1, d2, d3);
    printf("  ^ stage 0 carries 3.9x the arithmetic of the others. If these four\n");
    printf("    numbers do not reflect that, the instrument is reporting its own\n");
    printf("    floor, not the kernels. Only nsys can see through that.\n");

    printf("\n%-24s %12s %12s\n", "configuration", "us / step", "vs base");
    printf("%-24s %12.2f %12s\n",   "staged baseline",      base,    "1.00x");
    printf("%-24s %12.2f %11.2fx\n", "halve longest kernel", halfBig, base / halfBig);
    printf("%-24s %12.2f %11.2fx\n", "halve all four",       halfAll, base / halfAll);
    printf("%-24s %12.2f %11.2fx\n", "fuse into one launch", fusedUs, base / fusedUs);

    // ---- scoring -----------------------------------------------------------
    int score = 0;

    int diff = 0;
    for (int i = 0; i < N; ++i) if (hA[i] != hB[i]) ++diff;
    bool exactOk = (diff == 0);
    printf("\n[%s] 1. fused output is bit-identical to staged (%d/%d differ)\n",
           exactOk ? "x" : " ", diff, N);
    score += exactOk;

    bool fuseWins = (base / fusedUs) >= 1.20;
    printf("[%s] 2. fusion beats the baseline by >= 1.20x (got %.2fx)\n",
           fuseWins ? "x" : " ", base / fusedUs);
    score += fuseWins;

    bool lOk = (PRED_LAUNCH_US > 0.0) && (PRED_LAUNCH_US > 0.5 * L)
                                      && (PRED_LAUNCH_US < 2.0 * L);
    printf("[%s] 3. PRED_LAUNCH_US within 2x of measured (%.1f vs %.2f us)\n",
           lOk ? "x" : " ", (double)PRED_LAUNCH_US, L);
    score += lOk;

    // 1 = halve the longest kernel, 2 = halve all four, 3 = fuse.
    int bestMeasured = 1; double bestUs = halfBig;
    if (halfAll < bestUs) { bestMeasured = 2; bestUs = halfAll; }
    if (fusedUs < bestUs) { bestMeasured = 3; bestUs = fusedUs; }
    bool bestOk = (PRED_BEST == bestMeasured);
    printf("[%s] 4. PRED_BEST matches the measurement (you said %d, winner %d)\n",
           bestOk ? "x" : " ", (int)PRED_BEST, bestMeasured);
    score += bestOk;

    // Buckets placed in empty parts of the measured distribution (spec S12.5d):
    // repeated runs put fusion at 2.1-2.6x, so both edges have >= 0.3x margin.
    double fr = base / fusedUs;
    int bucket = fr < 1.2 ? 1 : (fr < 1.8 ? 2 : (fr < 3.5 ? 3 : 4));
    bool bucketOk = (PRED_FUSED_BUCKET == bucket);
    printf("[%s] 5. PRED_FUSED_BUCKET correct (you said %d, measured %.2fx = bucket %d)\n",
           bucketOk ? "x" : " ", (int)PRED_FUSED_BUCKET, fr, bucket);
    score += bucketOk;

    int wantLaunches = 4 * STEPS;
    bool cntOk = (NS_LAUNCH_COUNT == wantLaunches);
    printf("[%s] 6. nsys cudaLaunchKernel count exact (you said %d)\n",
           cntOk ? "x" : " ", (int)NS_LAUNCH_COUNT);
    score += cntOk;

    // The punchline, as two inequalities that no in-program instrument could
    // have produced. Both are ratios inside the capture, so they survive this
    // laptop's clock swings (spec S12.5c).
    double kms       = NS_KERNEL_NS / 1e6;
    double stagedMs  = base * STEPS / 1000.0;
    double perLaunch = (NS_KERNEL_NS > 0.0) ? NS_KERNEL_NS / 1e3 / (double)wantLaunches : 0.0;
    double busy      = 100.0 * kms / stagedMs;
    bool kerOk = (NS_KERNEL_NS > 0.0) && perLaunch < L && busy > 5.0 && busy < 60.0;
    printf("[%s] 7. nsys kernel time: %.2f us/launch vs %.2f us launch floor;\n",
           kerOk ? "x" : " ", perLaunch, L);
    printf("       GPU busy %.1f%% of the staged run (%.2f ms of %.2f ms)\n",
           busy, kms, stagedMs);
    score += kerOk;

    if (kerOk) {
        printf("\n    The average kernel in this pipeline is SHORTER than the call\n");
        printf("    that launches it. Halving any kernel -- or all of them --\n");
        printf("    cannot touch the %.0f%% of the time the GPU spends waiting.\n",
               100.0 - busy);
        printf("    Fewer, larger launches is the only lever: fusion here\n");
        printf("    (Module 11), or CUDA graphs (Module 28) when the kernels\n");
        printf("    cannot be fused.\n");
    }

    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(dIn)); CHECK(cudaFree(t0)); CHECK(cudaFree(t1));
    CHECK(cudaFree(t2)); CHECK(cudaFree(dOut));
    free(hIn); free(hA); free(hB);
    CHECK(cudaDeviceReset());

    printf("\nSCORE: %d/7\n", score);
    printf("OVERALL: %s\n", score == 7 ? "PASS" : "FAIL");
    return score == 7 ? 0 : 1;
}
