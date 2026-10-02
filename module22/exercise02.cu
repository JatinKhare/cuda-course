// =============================================================================
// Module 22 / Exercise 2 — find the gap.
//
// SYMPTOM (this is all you are told):
//
//   `runSlow()` applies a 33-tap FIR filter to a 1 Mi-sample signal 300 times,
//   with a different output gain each time, and reports the peak magnitude of
//   the last result.
//
//   The kernels are fine. Somebody has already checked them: they are
//   coalesced, their launch configuration is sensible, and Nsight Compute
//   would have nothing interesting to say about them.
//
//   The program nevertheless takes more than an order of magnitude longer
//   than the sum of its kernel durations. The GPU is idle for almost all of
//   the run.
//
//   Nothing in the kernels is wrong. Everything that is wrong is on the host,
//   and none of it is visible by reading any single line in isolation --
//   every suspect call is correct, idempotent, and three lines long.
//
// YOUR JOB: instrument it, profile it, say precisely what is wrong, fix it.
//
// There are THREE independent causes. Two of them are named for you in the
// TODOs. The third is not: you have to find it. It does not look like a
// performance bug when you read it, and -- read the next sentence twice -- it
// barely shows up in `cuda_api_sum` for runSlow either, because something
// else is already paying its bill. You will only see it after you fix the
// other two.
//
// WHAT TO FILL IN
//   TODO 1  NVTX instrumentation good enough to localize the cost
//   TODO 2  your diagnosis, as numbers the harness can check      (needs nsys)
//   TODO 3  eliminate the redundant host->device traffic
//   TODO 4  eliminate the per-iteration host/device round trip
//   TODO 5  find and eliminate the third cause                      (DESIGN)
//
// SCORING: 7 points. OVERALL: PASS requires all seven.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
//        (nvtx3 ships with CUDA 13.2 and is header-only: no -l flag.)
// RUN  : exercise02.exe
//
// PROFILE (you need this for TODO 2):
//   NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
//   "$NSYS" profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
//        -o ex02 --stats=true --force-overwrite=true ./exercise02.exe
//
//   The capture range is already placed around runSlow only, so every table
//   you read describes the program you are diagnosing and nothing else.
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

// =============================================================================
// TODO 1 — the NVTX scope guard.
//
// Build a type whose constructor opens a named NVTX range and whose destructor
// closes it, plus a macro NVTX_RANGE("name") that declares one. Two things are
// easy to get wrong: two NVTX_RANGE uses in the same scope must not collide,
// and the type must not be copyable (a copy pops the range twice, and an
// unbalanced push/pop stack silently reparents every range after it, which is
// worse than no instrumentation because the timeline still looks plausible).
//
// The C API is nvtxRangePushA(const char *) / nvtxRangePop(void).
//
// Then use it. Where you put the ranges is the judgement call: a range per
// iteration over 300 iterations is useful; a range per kernel launch over 600
// launches starts to cost real time inside the thing you are measuring. Name
// them so that `nsys stats --report nvtx_sum` alone tells a reader who has
// never seen this file which phase is expensive.
// =============================================================================
// TODO 1: YOUR CODE HERE  (define struct NvtxRange and #define NVTX_RANGE)


__constant__ float cCoef[TAPS];

// -----------------------------------------------------------------------------
// The two kernels. Do not modify them. The exercise is entirely host-side and
// the harness checks that both versions produce bit-identical output.
// -----------------------------------------------------------------------------
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
// Three host-side helpers. Read all three. Each is correct.
// -----------------------------------------------------------------------------
static void stageInput(float *dIn, const float *hIn)
{
    CHECK(cudaMemcpy(dIn, hIn, sizeof(float) * N, cudaMemcpyHostToDevice));
}

static float readPeak(float *hStage, const float *dOut)
{
    CHECK(cudaMemcpy(hStage, dOut, sizeof(float) * N, cudaMemcpyDeviceToHost));
    float m = 0.0f;
    for (int i = 0; i < N; ++i) { float v = fabsf(hStage[i]); if (v > m) m = v; }
    return m;
}

static long long gProgressCalls = 0;
static void recordProgress(int iter, const float *dOut)
{
    ++gProgressCalls;
    if (iter < 0) printf("%p\n", (const void *)dOut);   // never true
    CHECK(cudaDeviceSynchronize());
}

static int gBlocks = (N + BLOCK - 1) / BLOCK;

// Deterministic per-iteration gain. The 300 iterations are not redundant work:
// the final buffer depends on the last gain, so you cannot skip any of them.
static float gainOf(int it) { return 0.5f + 0.001f * (float)it; }

// =============================================================================
// runSlow — do not modify. This is the program you are diagnosing.
// =============================================================================
static double runSlow(float *dIn, float *dOut, const float *hIn, float *hStage,
                      float *peakOut)
{
    cudaEvent_t w0, w1;
    CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
    CHECK(cudaEventRecord(w0));

    float peak = 0.0f;
    for (int it = 0; it < ITERS; ++it) {
        stageInput(dIn, hIn);
        fir<<<gBlocks, BLOCK>>>(dIn, dOut, N);
        CHECK(cudaGetLastError());
        rescale<<<gBlocks, BLOCK>>>(dOut, N, gainOf(it));
        CHECK(cudaGetLastError());
        peak = readPeak(hStage, dOut);
        recordProgress(it, dOut);
    }

    CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
    float ms; CHECK(cudaEventElapsedTime(&ms, w0, w1));
    *peakOut = peak;
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    return ms;
}

// =============================================================================
// TODO 3, 4, 5 — runFast.
//
// Same 600 launches, same arithmetic, same final buffer contents, same
// reported peak. The signature is fixed; everything about WHEN the host talks
// to the device is yours.
//
//   TODO 3: one of the three helpers moves the same 4 MB across PCIe on every
//           iteration for data that has not changed since the program started.
//           Make it happen once. Note which cuda_api_sum row this removes and
//           by how much -- and note that the bytes are only half the story.
//
//   TODO 4: another helper forces the host to wait for the device on every
//           iteration to produce a value the program consumes exactly once.
//           Move it.
//
//   TODO 5 (DESIGN): after 3 and 4, profile again. There is still a gap on
//           every iteration. Find the third cause and make sure runFast does
//           not pay it.
//
//           Constraint, and it is the point: whatever the third helper is
//           BOOKKEEPING must still happen. The harness checks that the
//           progress counter advances exactly ITERS times inside runFast, so
//           deleting the call is not a fix -- it is a change of behaviour. You
//           have to separate the part of that helper that is doing work from
//           the part that is costing you the GPU.
//
//           Do not edit runSlow(), stageInput(), readPeak() or
//           recordProgress(); the harness needs the slow version to stay slow
//           to have something to compare against.
//
// The harness requires: bit-identical output, the same reported peak, the
// progress counter at ITERS, and a wall-time speedup of at least 6.0x.
// =============================================================================
static double runFast(float *dIn, float *dOut, const float *hIn, float *hStage,
                      float *peakOut)
{
    cudaEvent_t w0, w1;
    CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));

    // TODO 3/4/5: YOUR CODE HERE
    //
    // Record w0, run the ITERS loop, record w1, synchronize, and leave the
    // peak of the final buffer in *peakOut.
    (void)dIn; (void)dOut; (void)hIn; (void)hStage;

    // ---- leave the code below this line alone --------------------------------
    if (gProgressCalls <= (long long)ITERS) {   // TODO 3/4/5 not attempted
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
        *peakOut = 0.0f;
        return -1.0;
    }
    float ms; CHECK(cudaEventElapsedTime(&ms, w0, w1));
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    return ms;
}

// =============================================================================
// TODO 2 — your diagnosis, in numbers.
//
// Profile runSlow with the command in the header and fill these in.
//
//   DIAG_TOP_API   Which row of cuda_api_sum has the largest "Total Time (ns)",
//                  ignoring the cudaProfilerStart row (that row is the
//                  duration of the capture range itself, not a cost)?
//                      1 = cudaLaunchKernel
//                      2 = cudaMemcpy
//                      3 = cudaDeviceSynchronize
//                      4 = cudaMemcpyToSymbol
//                      5 = cudaMalloc
//
//   DIAG_H2D_MB    The "Total (MB)" of [CUDA memcpy Host-to-Device] in
//                  cuda_gpu_mem_size_sum, for the capture range.
//
//   DIAG_KERNEL_NS The summed "Total Time (ns)" of the two kernel rows in
//                  cuda_gpu_kern_sum, for the capture range.
//
// This last one is not optional decoration: the harness has no other way to
// know how long the GPU actually executed, so it uses your number to print the
// busy fraction of both versions. Leave them at 0 to skip; the harness will
// withhold the points rather than passing you quietly.
// =============================================================================
#define DIAG_TOP_API     0      // TODO 2a: YOUR ANSWER HERE (1-5)
#define DIAG_H2D_MB      0.0    // TODO 2b: YOUR ANSWER HERE (MB)
#define DIAG_KERNEL_NS   0.0    // TODO 2c: YOUR ANSWER HERE (ns)

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

    // Warm-up: context creation, module load, clocks. Deliberately outside the
    // capture range -- the first cudaMalloc alone can cost 100 ms.
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

        // nsys reports MB in SI units (1e6 bytes), to three decimals. Accept 5%
        // either side.
        double wantMB = (double)ITERS * (double)N * 4.0 / 1.0e6;
        bool h2dOk = fabs(DIAG_H2D_MB - wantMB) <= 0.05 * wantMB;
        printf("[%s] 6. H2D traffic in capture  (you said %.1f MB)\n",
               h2dOk ? "x" : " ", DIAG_H2D_MB);

        // The kernel total must be far under runSlow's wall time -- that is the
        // entire finding -- and close to runFast's wall time, because a fixed
        // runFast is GPU-bound and its wall time IS the kernels. The window is
        // wide enough for profiling overhead and clock variation, and far too
        // narrow to hit by guessing.
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
