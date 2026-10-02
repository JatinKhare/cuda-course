// =============================================================================
// Module 22 / Exercise 2 — SOLUTION — find the gap.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
// RUN  : exercise02_solution.exe
//
// PROFILE:
//   NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
//   "$NSYS" profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
//        -o ex02sol --stats=true --force-overwrite=true ./exercise02_solution.exe
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

#define N        (1 << 20)      // 1 Mi samples = 4 MB
#define TAPS       33
#define BLOCK     256
#define ITERS     300

// TODO 1 — the NVTX scope guard, identical in shape to Exercise 1's.
struct NvtxRange {
    explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
    ~NvtxRange()                         { nvtxRangePop(); }
    NvtxRange(const NvtxRange &)            = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
};
#define NVTX_CAT2(a, b) a##b
#define NVTX_CAT(a, b)  NVTX_CAT2(a, b)
#define NVTX_RANGE(name) NvtxRange NVTX_CAT(_nvtxScope, __LINE__)(name)

__constant__ float cCoef[TAPS];

__global__ void fir(const float *__restrict__ in, float *__restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    #pragma unroll
    for (int t = 0; t < TAPS; ++t) {
        int j = i + t - TAPS / 2;
        j = j < 0 ? 0 : (j >= n ? n - 1 : j);
        acc = fmaf(cCoef[t], in[j], acc);
    }
    out[i] = acc;
}

__global__ void rescale(float *__restrict__ a, int n, float s)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= s;
}

static void buildCoefficients(float *c)
{
    double sum = 0.0;
    for (int t = 0; t < TAPS; ++t) {
        double x = (double)(t - TAPS / 2) / 6.0;
        double w = exp(-0.5 * x * x);
        c[t] = (float)w;
        sum += w;
    }
    for (int t = 0; t < TAPS; ++t) c[t] = (float)(c[t] / sum);
}

// -----------------------------------------------------------------------------
// Three host-side helpers. Each is three lines long and each looks harmless.
// -----------------------------------------------------------------------------

// Cause 1. "Make sure the device has the input." Correct, idempotent, and
// 4 MB of PCIe traffic per call for data that has not changed since the
// program started.
static void stageInput(float *dIn, const float *hIn)
{
    CHECK(cudaMemcpy(dIn, hIn, sizeof(float) * N, cudaMemcpyHostToDevice));
}

// Cause 2. "Read the result so we can report its peak." Also correct. Also a
// full 4 MB device-to-host transfer plus a 1 Mi-element host scan, per
// iteration, for a number the program only prints once.
static float readPeak(float *hStage, const float *dOut)
{
    CHECK(cudaMemcpy(hStage, dOut, sizeof(float) * N, cudaMemcpyDeviceToHost));
    float m = 0.0f;
    for (int i = 0; i < N; ++i) { float v = fabsf(hStage[i]); if (v > m) m = v; }
    return m;
}

// Cause 3. "Progress bookkeeping." The counter is the only thing the call site
// can see. The cudaDeviceSynchronize() is the only thing that costs.
static long long gProgressCalls = 0;
static void recordProgress(int iter, const float *dOut)
{
    ++gProgressCalls;
    if (iter < 0) printf("%p\n", (const void *)dOut);   // never true
    CHECK(cudaDeviceSynchronize());
}

static int gBlocks = (N + BLOCK - 1) / BLOCK;

// Deterministic per-iteration gain, so the 300 iterations are not redundant
// work and the final buffer depends on the last one.
static float gainOf(int it) { return 0.5f + 0.001f * (float)it; }

// =============================================================================
// runSlow — the program being diagnosed. Not modified.
// =============================================================================
static double runSlow(float *dIn, float *dOut, const float *hIn, float *hStage,
                      float *peakOut)
{
    NVTX_RANGE("slow");
    cudaEvent_t w0, w1;
    CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
    CHECK(cudaEventRecord(w0));

    float peak = 0.0f;
    for (int it = 0; it < ITERS; ++it) {
        NVTX_RANGE("iter");
        { NVTX_RANGE("stage-input"); stageInput(dIn, hIn); }
        { NVTX_RANGE("fir");     fir<<<gBlocks, BLOCK>>>(dIn, dOut, N);
                                 CHECK(cudaGetLastError()); }
        { NVTX_RANGE("rescale"); rescale<<<gBlocks, BLOCK>>>(dOut, N, gainOf(it));
                                 CHECK(cudaGetLastError()); }
        { NVTX_RANGE("telemetry"); peak = readPeak(hStage, dOut); }
        { NVTX_RANGE("progress");  recordProgress(it, dOut); }
    }

    CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
    float ms; CHECK(cudaEventElapsedTime(&ms, w0, w1));
    *peakOut = peak;
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    return ms;
}

// =============================================================================
// TODO 3, 4, 5 — runFast.
// =============================================================================
static double runFast(float *dIn, float *dOut, const float *hIn, float *hStage,
                      float *peakOut)
{
    NVTX_RANGE("fast");
    cudaEvent_t w0, w1;
    CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));

    // FIX 3: the input is loop-invariant. One upload, before the timed region.
    // What this removes is 299 x 4 MB of PCIe traffic AND 299 host/device
    // round trips: cudaMemcpy from PAGEABLE host memory is synchronous with
    // respect to the host, so each call also drained the queue.
    { NVTX_RANGE("stage-once"); stageInput(dIn, hIn); }

    CHECK(cudaEventRecord(w0));
    for (int it = 0; it < ITERS; ++it) {
        NVTX_RANGE("iter");
        { NVTX_RANGE("fir");     fir<<<gBlocks, BLOCK>>>(dIn, dOut, N);
                                 CHECK(cudaGetLastError()); }
        { NVTX_RANGE("rescale"); rescale<<<gBlocks, BLOCK>>>(dOut, N, gainOf(it));
                                 CHECK(cudaGetLastError()); }

        // FIX 5: the progress bookkeeping still happens, exactly ITERS times.
        // Only the cudaDeviceSynchronize() buried inside recordProgress() is
        // gone. That call is invisible at the call site -- `recordProgress(it)`
        // reads as pure host accounting -- and it is the one remaining reason
        // the host could not run ahead after FIX 3 and FIX 4.
        ++gProgressCalls;
    }
    CHECK(cudaEventRecord(w1));
    CHECK(cudaEventSynchronize(w1));
    float ms; CHECK(cudaEventElapsedTime(&ms, w0, w1));

    // FIX 4: the telemetry the program actually consumes is the LAST peak.
    // One readback, after the loop, outside the timed region.
    *peakOut = readPeak(hStage, dOut);

    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    return ms;
}

// =============================================================================
// TODO 2 — the diagnosis, in numbers read off `nsys stats`.
//
// Captured on the RTX 3500 Ada with Nsight Systems 2025.6.3. See the solution
// md for the full tables.
// =============================================================================
#define DIAG_TOP_API     2            // 2 = cudaMemcpy
#define DIAG_H2D_MB   1258.291        // MB of Host-to-Device traffic in the capture
#define DIAG_KERNEL_NS  16653926.0    // fir + rescale Total Time (ns)

// -----------------------------------------------------------------------------
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    nvtxNameOsThreadA(0, "fir-main");

    printf("Module 22 / Exercise 2 — find the gap\n");
    printf("N = %d samples, %d taps, %d iterations, %d launches per version\n\n",
           N, TAPS, ITERS, ITERS * 2);

    float *hIn    = (float *)malloc(sizeof(float) * N);
    float *hStage = (float *)malloc(sizeof(float) * N);
    float *r1     = (float *)malloc(sizeof(float) * N);
    float *r2     = (float *)malloc(sizeof(float) * N);
    srand(20251111);
    for (int i = 0; i < N; ++i)
        hIn[i] = sinf(0.001f * (float)i) + 0.01f * ((float)rand() / (float)RAND_MAX - 0.5f);

    float *dIn = nullptr, *dOut = nullptr;
    CHECK(cudaMalloc(&dIn,  sizeof(float) * N));
    CHECK(cudaMalloc(&dOut, sizeof(float) * N));

    // Warm-up, deliberately outside the capture range.
    {
        float coef[TAPS];
        buildCoefficients(coef);
        CHECK(cudaMemcpyToSymbol(cCoef, coef, sizeof(float) * TAPS));
        CHECK(cudaMemcpy(dIn, hIn, sizeof(float) * N, cudaMemcpyHostToDevice));
        for (int i = 0; i < 1500; ++i) fir<<<gBlocks, BLOCK>>>(dIn, dOut, N);
        CHECK(cudaDeviceSynchronize());
    }

    float peakS = 0.0f, peakF = 0.0f;
    long long callsAfterSlow = 0;

    CHECK(cudaProfilerStart());
    double wallS = runSlow(dIn, dOut, hIn, hStage, &peakS);
    CHECK(cudaProfilerStop());
    CHECK(cudaMemcpy(r1, dOut, sizeof(float) * N, cudaMemcpyDeviceToHost));
    callsAfterSlow = gProgressCalls;

    double wallF = runFast(dIn, dOut, hIn, hStage, &peakF);
    if (wallF > 0.0) CHECK(cudaMemcpy(r2, dOut, sizeof(float) * N, cudaMemcpyDeviceToHost));

    if (wallF < 0.0) {
        printf("Set TODO 3/4/5 first.\n");
        CHECK(cudaFree(dIn)); CHECK(cudaFree(dOut));
        free(hIn); free(hStage); free(r1); free(r2);
        CHECK(cudaDeviceReset());
        return 0;
    }

    printf("%-10s %12s %14s\n", "version", "wall (ms)", "us / iter");
    printf("%-10s %12.3f %14.1f\n", "slow", wallS, 1000.0 * wallS / ITERS);
    printf("%-10s %12.3f %14.1f\n", "fast", wallF, 1000.0 * wallF / ITERS);
    printf("speedup  : %.2fx\n\n", wallS / wallF);

    int score = 0;

    int bad = 0;
    for (int i = 0; i < N; ++i) if (r1[i] != r2[i]) ++bad;
    bool outOk = (bad == 0);
    printf("[%s] 1. filtered signal matches bit-for-bit (%d/%d differ)\n",
           outOk ? "x" : " ", bad, N);
    score += outOk;

    bool peakOk = (peakS == peakF);
    printf("[%s] 2. reported peak matches   (%.7f vs %.7f)\n",
           peakOk ? "x" : " ", (double)peakS, (double)peakF);
    score += peakOk;

    bool telOk = (callsAfterSlow == ITERS) && (gProgressCalls == 2LL * ITERS);
    printf("[%s] 3. progress hook still called %d times in runFast (%lld total)\n",
           telOk ? "x" : " ", ITERS, gProgressCalls);
    score += telOk;

    bool fast = (wallS / wallF) >= 6.0;
    printf("[%s] 4. speedup >= 6.0x         (got %.2fx)\n",
           fast ? "x" : " ", wallS / wallF);
    score += fast;

    bool given = (DIAG_TOP_API != 0) && (DIAG_H2D_MB > 0.0) && (DIAG_KERNEL_NS > 0.0);
    if (!given) {
        printf("[ ] 5-7. TODO 2 not filled in -- profile runSlow with nsys.\n");
    } else {
        bool topOk = (DIAG_TOP_API == 2);
        printf("[%s] 5. top cuda_api_sum row identified\n", topOk ? "x" : " ");

        // 300 uploads of 4 MB. nsys reports MB in SI units (1e6), to one
        // decimal. Accept 5% either side.
        double wantMB = (double)ITERS * (double)N * 4.0 / 1.0e6;
        bool h2dOk = fabs(DIAG_H2D_MB - wantMB) <= 0.05 * wantMB;
        printf("[%s] 6. H2D traffic in capture  (you said %.1f MB, expected %.1f)\n",
               h2dOk ? "x" : " ", DIAG_H2D_MB, wantMB);

        // The kernel total must be positive, must be far under runSlow's wall
        // time (that is the entire point), and must be close to runFast's wall
        // time, because runFast is GPU-bound and its wall time IS the kernels.
        // The window is wide enough for profiling overhead and clock variation,
        // far too narrow to hit by guessing.
        double kms = DIAG_KERNEL_NS / 1e6;
        bool kerOk = kms > 0.60 * wallF && kms < 1.30 * wallF && kms < 0.5 * wallS;
        printf("[%s] 7. kernel time plausible   (%.3f ms; runFast wall %.3f ms)\n",
               kerOk ? "x" : " ", kms, wallF);
        score += topOk + h2dOk + kerOk;

        if (kerOk) {
            printf("\n    GPU busy, runSlow = %.1f%%   (%.3f ms of %.3f ms)\n",
                   100.0 * kms / wallS, kms, wallS);
            printf("    GPU busy, runFast = %.1f%%   (%.3f ms of %.3f ms)\n",
                   100.0 * kms / wallF, kms, wallF);
            printf("    (runFast can read slightly over 100%%: the kernel total\n"
                   "     comes from a PROFILED capture and the wall time does not.\n"
                   "     It means runFast is GPU-bound, which is the goal.)\n");
            printf("    The kernels never changed. Only the host did.\n");
        }
    }

    CHECK(cudaFree(dIn)); CHECK(cudaFree(dOut));
    free(hIn); free(hStage); free(r1); free(r2);
    CHECK(cudaDeviceReset());

    printf("\nSCORE: %d/7\n", score);
    printf("OVERALL: %s\n", score == 7 ? "PASS" : "FAIL");
    return score == 7 ? 0 : 1;
}
