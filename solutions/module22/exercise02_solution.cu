// =============================================================================
// Module 22 / Exercise 2 — find the gap.
//
// SYMPTOM (this is all you are told):
//
//   `runSlow()` applies a 33-tap FIR filter to a 4 Mi-sample signal, 200 times.
//   The kernels are fine. Somebody has already checked them: they are
//   coalesced, they hit a reasonable fraction of the bandwidth ceiling, and
//   Nsight Compute would have nothing interesting to say about them.
//
//   The program nevertheless takes between 3x and 6x longer than the sum of
//   its kernel durations. The GPU is idle for most of the run.
//
//   Nothing in the kernels is wrong. Everything that is wrong is on the host,
//   and none of it is visible by reading any single line in isolation.
//
// YOUR JOB: instrument it, profile it, say precisely what is wrong, fix it.
//
// There are THREE independent causes. Two of them are named for you in the
// TODOs. The third is not: you have to find it on the timeline. It is a real
// pattern that gets into real codebases for a real reason, and it does not
// look like a performance bug when you read it.
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
// RUN  : exercise02.exe
//
// PROFILE:
//   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
//        -o ex02 --stats=true --force-overwrite=true exercise02.exe
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

#define N        (1 << 22)      // 4 Mi samples = 16 MB
#define TAPS       33
#define BLOCK     256
#define ITERS     200

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

// -----------------------------------------------------------------------------
// Host-side helpers. Read these; one of them is not what it looks like.
// -----------------------------------------------------------------------------

// Builds the filter taps. Pure host arithmetic, deterministic, no state.
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

// A progress/telemetry hook. The kind of thing that gets added during a
// debugging session and then never removed, because it "only prints".
static long long gSampleCount = 0;
static void recordProgress(int iter, const float *dSignal)
{
    // Keep a running count so the call cannot be optimized away.
    gSampleCount += (long long)N;
    if (iter < 0) {                       // never true; keeps dSignal live
        printf("%p\n", (const void *)dSignal);
    }
    // Make the telemetry timestamp line up with the device's view of progress.
    CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
struct GpuTimeline {
    cudaEvent_t *beg, *end; int cap, n;
    void init(int c) {
        cap = c; n = 0;
        beg = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * cap);
        end = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * cap);
        for (int i = 0; i < cap; ++i) { CHECK(cudaEventCreate(&beg[i])); CHECK(cudaEventCreate(&end[i])); }
    }
    void open()  { if (n < cap) CHECK(cudaEventRecord(beg[n])); }
    void close() { if (n < cap) { CHECK(cudaEventRecord(end[n])); ++n; } }
    double totalMs() const {
        double s = 0.0;
        for (int i = 0; i < n; ++i) { float ms; cudaEventElapsedTime(&ms, beg[i], end[i]); s += ms; }
        return s;
    }
    void destroy() {
        for (int i = 0; i < cap; ++i) { cudaEventDestroy(beg[i]); cudaEventDestroy(end[i]); }
        free(beg); free(end);
    }
};

static int gBlocks = (N + BLOCK - 1) / BLOCK;

// =============================================================================
// runSlow — do not modify. This is the program you are diagnosing.
// =============================================================================
static double runSlow(float *in, float *out, float *dNorm, double *gpuMsOut,
                      float *normOut)
{
    GpuTimeline tl; tl.init(ITERS * 2);
    cudaEvent_t w0, w1;
    CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
    CHECK(cudaEventRecord(w0));

    float coef[TAPS];
    float hNorm = 1.0f;

    NVTX_RANGE("slow");
    for (int it = 0; it < ITERS; ++it) {
        NVTX_RANGE("iter");
        { NVTX_RANGE("coef");  buildCoefficients(coef);
                               CHECK(cudaMemcpyToSymbol(cCoef, coef, sizeof(float) * TAPS)); }

        { NVTX_RANGE("fir");
          tl.open(); fir<<<gBlocks, BLOCK>>>(in, out, N); tl.close();
          CHECK(cudaGetLastError()); }

        { NVTX_RANGE("rescale");
          tl.open(); rescale<<<gBlocks, BLOCK>>>(out, N, 1.0f); tl.close();
          CHECK(cudaGetLastError()); }

        { NVTX_RANGE("readback");
          CHECK(cudaMemcpy(&hNorm, dNorm, sizeof(float), cudaMemcpyDeviceToHost)); }

        { NVTX_RANGE("telemetry"); recordProgress(it, out); }

        float *t = in; in = out; out = t;        // ping-pong
    }

    CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
    float ms; CHECK(cudaEventElapsedTime(&ms, w0, w1));
    *gpuMsOut = tl.totalMs();
    *normOut  = hNorm;
    tl.destroy();
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    return ms;
}

// =============================================================================
// TODO 3, 4, 5 — runFast.
//
// Same 400 launches, same ping-pong, same final contents, same final hNorm.
// The signature and the ping-pong are fixed; everything about when the host
// talks to the device is yours.
//
//   TODO 3: the coefficient table is rebuilt and re-uploaded 200 times and is
//           bit-identical every time. Upload it once. Note which cuda_api_sum
//           row this removes and by how much -- it is not the row most people
//           predict, because 132 bytes is nothing and the cost is not bytes.
//
//   TODO 4: the four-byte readback of dNorm forces the host to wait for the
//           device every iteration. The program uses hNorm only after the loop
//           ends. Move it.
//
//   TODO 5 (DESIGN): after 3 and 4, profile again. There is still a gap on
//           every iteration and the GPU is still not saturated. Find the third
//           cause and make sure runFast does not pay it.
//
//           Constraint, and it is the whole point: the telemetry that
//           runSlow collects must still be collected. The harness checks that
//           gSampleCount ends at exactly 2 * ITERS * N, so you cannot simply
//           delete the progress hook -- 200 iterations of bookkeeping still
//           have to happen. You have to separate the part of that hook that
//           is doing work from the part that is costing you the GPU.
//
//           Do not edit runSlow() or recordProgress(); the harness needs the
//           slow version to stay slow to have something to compare against.
//
// The harness requires the output to match runSlow's elementwise, the same
// final hNorm, 400 launches, the telemetry count, and a speedup of >= 2.5x.
// =============================================================================
static double runFast(float *in, float *out, float *dNorm, double *gpuMsOut,
                      float *normOut)
{
    GpuTimeline tl; tl.init(ITERS * 2);
    cudaEvent_t w0, w1;
    CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));

    float hNorm = 1.0f;

    NVTX_RANGE("fast");

    // FIX 3: the coefficients are a loop invariant. Build and upload once.
    // What this removes is not 200 x 132 bytes of traffic -- that is nothing.
    // cudaMemcpyToSymbol with a pageable host source is SYNCHRONOUS with
    // respect to the host: it drains the queue before it copies. So each one
    // was a full host/device round trip disguised as a 132-byte upload.
    {
        NVTX_RANGE("coef-once");
        float coef[TAPS];
        buildCoefficients(coef);
        CHECK(cudaMemcpyToSymbol(cCoef, coef, sizeof(float) * TAPS));
    }

    CHECK(cudaEventRecord(w0));
    for (int it = 0; it < ITERS; ++it) {
        NVTX_RANGE("iter");
        { NVTX_RANGE("fir");
          tl.open(); fir<<<gBlocks, BLOCK>>>(in, out, N); tl.close();
          CHECK(cudaGetLastError()); }

        { NVTX_RANGE("rescale");
          tl.open(); rescale<<<gBlocks, BLOCK>>>(out, N, 1.0f); tl.close();
          CHECK(cudaGetLastError()); }

        // FIX 4: no per-iteration readback. hNorm is only read after the loop.

        // FIX 5: the telemetry still happens -- gSampleCount is maintained
        // exactly as before -- but without recordProgress's hidden
        // cudaDeviceSynchronize(). That one line was the dominant cost, and it
        // is invisible at the call site: `recordProgress(it, out)` looks like
        // pure host bookkeeping. On the timeline it is the top row of
        // cuda_api_sum.
        gSampleCount += (long long)N;

        float *t = in; in = out; out = t;        // ping-pong
    }
    CHECK(cudaEventRecord(w1));
    CHECK(cudaEventSynchronize(w1));

    // The one readback the program actually needed.
    CHECK(cudaMemcpy(&hNorm, dNorm, sizeof(float), cudaMemcpyDeviceToHost));

    // ---- leave the code below this line alone --------------------------------
    if (tl.n == 0) {
        tl.destroy();
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
        *gpuMsOut = 0.0; *normOut = 0.0;
        return -1.0;
    }
    float ms; CHECK(cudaEventElapsedTime(&ms, w0, w1));
    *gpuMsOut = tl.totalMs();
    *normOut  = hNorm;
    tl.destroy();
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    return ms;
}

// =============================================================================
// TODO 2 — your diagnosis, in numbers.
//
// Profile runSlow (the capture range is already placed) and fill these in.
//
//   DIAG_TOP_API   Which row of cuda_api_sum has the largest "Total Time (ns)",
//                  ignoring the cudaProfilerStart row (that row is just the
//                  duration of the capture range itself, not a cost)?
//                      1 = cudaLaunchKernel
//                      2 = cudaMemcpy
//                      3 = cudaDeviceSynchronize
//                      4 = cudaMemcpyToSymbol
//                      5 = cudaMalloc
//
//   DIAG_H2D_COUNT How many [CUDA memcpy Host-to-Device] operations does
//                  cuda_gpu_mem_time_sum report for the capture range?
//
//   DIAG_KERNEL_NS The summed "Total Time (ns)" of the two kernel rows in
//                  cuda_gpu_kern_sum, for the capture range (runSlow only --
//                  the capture range closes before runFast).
//
// Leave them at 0 to skip; the harness will withhold the points.
// =============================================================================
#define DIAG_TOP_API     0      // TODO 2a: YOUR ANSWER HERE (1-5)
#define DIAG_H2D_COUNT   0      // TODO 2b: YOUR ANSWER HERE
#define DIAG_KERNEL_NS   0.0    // TODO 2c: YOUR ANSWER HERE

// -----------------------------------------------------------------------------
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Module 22 / Exercise 2 — find the gap\n");
    printf("N = %d samples, %d taps, %d iterations, %d launches\n\n",
           N, TAPS, ITERS, ITERS * 2);

    float *h  = (float *)malloc(sizeof(float) * N);
    float *r1 = (float *)malloc(sizeof(float) * N);
    float *r2 = (float *)malloc(sizeof(float) * N);
    srand(20251111);
    for (int i = 0; i < N; ++i)
        h[i] = sinf(0.001f * (float)i) + 0.01f * ((float)rand() / (float)RAND_MAX - 0.5f);

    float *a = nullptr, *b = nullptr, *dNorm = nullptr;
    CHECK(cudaMalloc(&a, sizeof(float) * N));
    CHECK(cudaMalloc(&b, sizeof(float) * N));
    CHECK(cudaMalloc(&dNorm, sizeof(float)));
    float one = 1.0f;
    CHECK(cudaMemcpy(dNorm, &one, sizeof(float), cudaMemcpyHostToDevice));

    // Warm-up, kept outside the capture range.
    {
        float coef[TAPS]; buildCoefficients(coef);
        CHECK(cudaMemcpyToSymbol(cCoef, coef, sizeof(float) * TAPS));
        CHECK(cudaMemcpy(a, h, sizeof(float) * N, cudaMemcpyHostToDevice));
        for (int i = 0; i < 150; ++i) fir<<<gBlocks, BLOCK>>>(a, b, N);
        CHECK(cudaDeviceSynchronize());
    }

    double wallS = 0.0, gpuS = 0.0, wallF = 0.0, gpuF = 0.0;
    float normS = 0.0f, normF = 0.0f;

    CHECK(cudaMemcpy(a, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    CHECK(cudaProfilerStart());
    wallS = runSlow(a, b, dNorm, &gpuS, &normS);
    CHECK(cudaProfilerStop());
    // ITERS is even, so the ping-pong leaves the result in `a`.
    CHECK(cudaMemcpy(r1, a, sizeof(float) * N, cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(a, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    wallF = runFast(a, b, dNorm, &gpuF, &normF);
    if (wallF > 0.0) CHECK(cudaMemcpy(r2, a, sizeof(float) * N, cudaMemcpyDeviceToHost));

    if (wallF < 0.0) {
        printf("Set TODO 3/4/5 first.\n");
        CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(dNorm));
        free(h); free(r1); free(r2); CHECK(cudaDeviceReset());
        return 0;
    }

    printf("%-10s %12s %14s %12s\n", "version", "wall (ms)", "GPU<=(ms)", "busy<= %");
    printf("%-10s %12.3f %14.3f %11.1f%%\n", "slow", wallS, gpuS, 100.0 * gpuS / wallS);
    printf("%-10s %12.3f %14.3f %11.1f%%\n", "fast", wallF, gpuF, 100.0 * gpuF / wallF);
    printf("speedup  : %.2fx\n\n", wallS / wallF);

    int score = 0;

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (fabsf(r1[i] - r2[i]) > 1e-5f * fmaxf(1.0f, fabsf(r1[i]))) ++bad;
    bool outOk = (bad == 0);
    printf("[%s] 1. filtered signal matches (%d/%d samples differ)\n", outOk ? "x" : " ", bad, N);
    score += outOk;

    bool normOk = fabsf(normS - normF) <= 1e-6f * fmaxf(1.0f, fabsf(normS));
    printf("[%s] 2. final hNorm matches     (%.6f vs %.6f)\n",
           normOk ? "x" : " ", (double)normS, (double)normF);
    score += normOk;

    bool fast = (wallS / wallF) >= 2.5;
    printf("[%s] 3. speedup >= 2.5x         (got %.2fx)\n", fast ? "x" : " ", wallS / wallF);
    score += fast;

    bool busyOk = (100.0 * gpuF / wallF) >= 85.0;
    printf("[%s] 4. fast version >= 85%% busy (got %.1f%%)\n",
           busyOk ? "x" : " ", 100.0 * gpuF / wallF);
    score += busyOk;

    // The telemetry must still have been collected -- deleting the progress
    // hook is not a fix, it is a change of behaviour.
    long long wantSamples = 2LL * (long long)ITERS * (long long)N;
    bool telOk = (gSampleCount == wantSamples);
    printf("[%s] 4b. telemetry still collected (%lld of %lld samples)\n",
           telOk ? "x" : " ", gSampleCount, wantSamples);
    if (!telOk) score -= 1;         // a missing hook invalidates the comparison

    // TODO 2 cross-checks.
    bool given = (DIAG_TOP_API != 0) && (DIAG_H2D_COUNT != 0) && (DIAG_KERNEL_NS > 0.0);
    if (!given) printf("[ ] 5-7. TODO 2 not filled in -- profile runSlow.\n");
    else {
        bool topOk = (DIAG_TOP_API == 3);
        printf("[%s] 5. top cuda_api_sum row identified\n", topOk ? "x" : " ");
        // One cudaMemcpyToSymbol per iteration, and nothing else inside the
        // capture range moves host memory to the device.
        bool h2dOk = (DIAG_H2D_COUNT == ITERS);
        printf("[%s] 6. H2D operation count   (you said %d)\n", h2dOk ? "x" : " ", DIAG_H2D_COUNT);
        double kms = DIAG_KERNEL_NS / 1e6;
        bool kerOk = kms > 0.30 * gpuS && kms < gpuS;
        printf("[%s] 7. kernel time plausible (%.3f ms; event bound %.3f ms)\n",
               kerOk ? "x" : " ", kms, gpuS);
        score += topOk + h2dOk + kerOk;
        if (kerOk)
            printf("\n    true GPU busy in runSlow = %.1f%%  (event bound said %.1f%%)\n",
                   100.0 * kms / wallS, 100.0 * gpuS / wallS);
    }

    CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(dNorm));
    free(h); free(r1); free(r2);
    CHECK(cudaDeviceReset());

    printf("\nSCORE: %d/7\n", score);
    printf("OVERALL: %s\n", score == 7 ? "PASS" : "FAIL");
    return score == 7 ? 0 : 1;
}
