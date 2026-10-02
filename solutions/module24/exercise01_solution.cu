/* =====================================================================
 * Module 24 / Exercise 01 SOLUTION — Build the copy/compute overlap pipeline
 *
 * GOAL
 *   A 48 MB host-resident buffer must be sent to the GPU, transformed by
 *   a compute-bound kernel, and brought back.  Written the obvious way
 *   that is three serial phases.  Your job is to turn it into a pipeline
 *   whose transfers overlap its computation, and to predict how much that
 *   can possibly be worth on THIS GPU before you measure it.
 *
 *   The harness measures the three phase times and prints them, together
 *   with this device's asyncEngineCount, BEFORE it looks at your
 *   prediction.  Run it once with the TODOs blank to collect those
 *   numbers; they are the input to TODO 5.
 *
 * BUILD
 *   nvcc -arch=sm_89 -O3 -std=c++17 -o exercise01.exe exercise01.cu
 * RUN
 *   exercise01.exe
 *
 * WHAT IS SCORED (10 points; OVERALL: PASS requires all of them)
 *   2  chunkOf() partitions [0,n) exactly once, including a ragged tail
 *   2  the pipeline produces the correct answer from pinned host memory
 *   2  pipelineBound() agrees with the reference on five synthetic inputs
 *   2  the measured speedup reaches at least 70% of your own bound
 *   2  PRED_BUCKET matches the bucket the measurement lands in
 *
 * NOTES
 *   - Pinned host memory is supplied for you.  cudaMemcpyAsync only
 *     overlaps out of pinned memory; Module 26 owns the mechanism.  The
 *     harness also runs your pipeline on a pageable buffer and prints the
 *     result for comparison -- that run is not scored.
 *   - Module 25 covers events properly.  You do not need any event here.
 *   - Do not change anything outside a "YOUR CODE HERE" region.
 * ===================================================================== */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            printf("CUDA error %s:%d '%s': %s (%s)\n", __FILE__, __LINE__,     \
                   #call, cudaGetErrorName(err_), cudaGetErrorString(err_));   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ===================================================================
 * TODO 5a — PREDICTION.  Set this to 1, 2, 3 or 4 before the harness
 * will run anything you wrote.  Which bucket will the measured
 * speedup (serial time / your pipeline time) fall into?
 *
 *     1 : below 1.10x      (essentially no overlap)
 *     2 : 1.10x .. 1.30x
 *     3 : 1.30x .. 2.00x
 *     4 : above 2.00x
 *
 * Derive it from the phase times and asyncEngineCount the harness
 * prints.  Guessing "as big as possible" is a specific, wrong answer.
 * =================================================================== */
static const int PRED_BUCKET = 3;   /* SOLUTION */

/* ---- tuning constants you own ------------------------------------- */
static const int N_STREAMS = 4;
static const int N_CHUNKS  = 16;

/* ---- the payload (do not modify) ---------------------------------- */
__global__ void condition(const float* __restrict__ in, float* __restrict__ out,
                          int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    for (int k = 0; k < iters; ++k) v = fmaf(v, 1.000001f, 1.0e-7f);
    out[i] = v;
}
__global__ void streamWarm(const float* __restrict__ a, float* __restrict__ b, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) b[i] = a[i] * 1.000001f;
}
__global__ void computeWarm(float* sink, int iters)
{
    float a = (float)(threadIdx.x + 1), b = 1.0000001f;
    for (int k = 0; k < iters; ++k) a = fmaf(a, b, 1.0e-6f);
    if (a == 12345.678f) sink[0] = a;
}

/* ===================================================================
 * TODO 1 — chunk arithmetic.
 *
 * Fill in the element offset and length of chunk k when [0, n) is split
 * into nChunks pieces.  n is NOT a multiple of nChunks and must not be
 * rounded.  Every element must belong to exactly one chunk; no chunk may
 * reach past n; a chunk may legitimately be empty if nChunks > n.
 *
 * The harness verifies this directly, for several (n, nChunks) pairs
 * including degenerate ones, before it runs anything on the GPU.
 * =================================================================== */
static void chunkOf(int k, int n, int nChunks, int* off, int* len)
{
    int c = (n + nChunks - 1) / nChunks;     /* round UP, never down */
    int o = k * c;
    if (o > n) o = n;
    int l = c;
    if (o + l > n) l = n - o;                /* ragged last chunk */
    if (l < 0) l = 0;
    *off = o;
    *len = l;
}

/* ===================================================================
 * TODO 2 — stream assignment.
 *
 * Return the index of the stream that should carry chunk k, given
 * nStreams streams.  Think about what property the assignment must have
 * for two chunks that are in flight at the same time, and what it costs
 * you if two consecutive chunks land in the same stream.
 * =================================================================== */
static int streamFor(int k, int nStreams)
{
    return k % nStreams;      /* round robin: consecutive chunks differ */
}

/* ===================================================================
 * TODO 5b — the achievable bound.
 *
 * Given the three measured phase times for the WHOLE buffer and the
 * number of copy engines the device reports, return the largest speedup
 * a chunked pipeline could possibly have over running the three phases
 * back to back.  Assume chunks are small enough that ramp-up and drain
 * are negligible, and that the kernel and the copy engine(s) are the only
 * resources.
 *
 * This must be correct for copyEngines == 1 AND for copyEngines >= 2;
 * the harness probes both.  Returning the same expression for both is a
 * specific, wrong answer.
 * =================================================================== */
static double pipelineBound(double h2dMs, double kernelMs, double d2hMs, int copyEngines)
{
    double serial = h2dMs + kernelMs + d2hMs;
    /* With two or more copy engines the two directions run concurrently, so
     * the copy resource is busy for max(H, D).  With one engine it serves both
     * directions and is busy for H + D. */
    double copyBusy = (copyEngines >= 2) ? ((h2dMs > d2hMs) ? h2dMs : d2hMs)
                                         : (h2dMs + d2hMs);
    double critical = (copyBusy > kernelMs) ? copyBusy : kernelMs;
    return serial / critical;
}

/* ---- globals the pipeline needs (set up by main) ------------------- */
static int           g_n, g_iters;
static float        *g_dIn, *g_dOut;
static cudaStream_t  g_s[8];

/* ===================================================================
 * TODO 3 — the pipeline itself.  THIS IS THE EXERCISE.
 *
 * Issue the work for all nChunks chunks so that transfers overlap
 * computation.  You have N_STREAMS streams in g_s[], the device buffers
 * g_dIn / g_dOut, and the host buffers hIn / hOut (both are the SAME
 * length as the device buffers, so chunk k lives at the same offset in
 * all four).
 *
 * Requirements, in order of importance:
 *
 *   (a) Chunk k's kernel must not start before chunk k's input has
 *       arrived, and chunk k's result must not be copied out before its
 *       kernel has finished.  You are not allowed to use events for
 *       this -- there is a cheaper mechanism.
 *
 *   (b) THE COPY ENGINE MUST NEVER BE LEFT HOLDING A TRANSFER THAT IS
 *       WAITING ON A KERNEL WHILE ANOTHER TRANSFER IS READY TO RUN.
 *       Read the asyncEngineCount line the harness prints before you
 *       decide what this costs you.  The most natural way to write this
 *       loop violates (b) and measures SLOWER than the serial version.
 *
 *   (c) Nothing in this function may synchronize.  No cudaDeviceSynchronize,
 *       no cudaStreamSynchronize, no blocking cudaMemcpy, no cudaMalloc,
 *       no cudaFree, no allocation of any kind.  Every one of those is a
 *       synchronization point -- some of them silently.
 *
 * Use cudaMemcpyAsync(dst, src, bytes, kind, stream) and the four-argument
 * launch form kernel<<<grid, block, 0, stream>>>(...).
 * =================================================================== */
static void readerPipeline(const float* hIn, float* hOut, int nChunks)
{
    /* Deferred-D2H software pipeline.  Chunk k's result copy is issued one
     * loop iteration AFTER its kernel, so the single copy engine always has a
     * transfer at the head of its queue that is ready to run. */
    for (int k = 0; k < nChunks; ++k) {
        int off, len;
        chunkOf(k, g_n, nChunks, &off, &len);
        if (len <= 0) continue;
        cudaStream_t st = g_s[streamFor(k, N_STREAMS)];

        CHECK(cudaMemcpyAsync(g_dIn + off, hIn + off, (size_t)len * sizeof(float),
                              cudaMemcpyHostToDevice, st));
        condition<<<(len + 255) / 256, 256, 0, st>>>(g_dIn + off, g_dOut + off, len, g_iters);
        CHECK(cudaGetLastError());

        if (k >= 1) {
            int o2, l2;
            chunkOf(k - 1, g_n, nChunks, &o2, &l2);
            if (l2 > 0)
                CHECK(cudaMemcpyAsync(hOut + o2, g_dOut + o2, (size_t)l2 * sizeof(float),
                                      cudaMemcpyDeviceToHost,
                                      g_s[streamFor(k - 1, N_STREAMS)]));
        }
    }
    /* the deferred copy for the final chunk */
    int oL, lL;
    chunkOf(nChunks - 1, g_n, nChunks, &oL, &lL);
    if (lL > 0)
        CHECK(cudaMemcpyAsync(hOut + oL, g_dOut + oL, (size_t)lL * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              g_s[streamFor(nChunks - 1, N_STREAMS)]));
}

/* ===================================================================
 * TODO 4 — the wait.
 *
 * Block the host until every chunk's result is in hOut, and no longer.
 * This runs after readerPipeline() returns and is inside the timed
 * region.  There is more than one correct answer; prefer the one that
 * expresses what you actually need.
 * =================================================================== */
static void waitForPipeline(void)
{
    /* Wait for the streams this pipeline used, and nothing else.
     * cudaDeviceSynchronize() would also be correct and would also wait for
     * work that has nothing to do with us. */
    for (int i = 0; i < N_STREAMS; ++i) CHECK(cudaStreamSynchronize(g_s[i]));
}

/* ------------------------------------------------------------------ */
/* Harness below this line.  Do not modify.                            */
/* ------------------------------------------------------------------ */

static int peek(const int* p) { const volatile int* q = p; return *q; }

static uint64_t fnv1a(uint64_t h, long long v)
{
    unsigned char* b = (unsigned char*)&v;
    for (int i = 0; i < 8; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static void serialRun(const float* hIn, float* hOut)
{
    CHECK(cudaMemcpy(g_dIn, hIn, (size_t)g_n * 4, cudaMemcpyHostToDevice));
    condition<<<(g_n + 255) / 256, 256>>>(g_dIn, g_dOut, g_n, g_iters);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(hOut, g_dOut, (size_t)g_n * 4, cudaMemcpyDeviceToHost));
}

static void warmUp(float* dA, float* dB, size_t nWarm, float* sink)
{
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count() < 1500) {
        streamWarm<<<960, 256>>>(dA, dB, nWarm);
        CHECK(cudaDeviceSynchronize());
    }
    t0 = clk::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count() < 500) {
        computeWarm<<<960, 256>>>(sink, 20000);
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaGetLastError());
}

/* structural test of chunkOf: every element covered exactly once */
static bool checkChunkOf(void)
{
    struct { int n, c; } cases[6] = {
        {12582917, 16}, {12582917, 7}, {1000, 1}, {1000, 1000}, {5, 8}, {17, 5}
    };
    for (int t = 0; t < 6; ++t) {
        int n = cases[t].n, nc = cases[t].c;
        long long covered = 0;
        int prevEnd = 0;
        bool ok = true;
        for (int k = 0; k < nc; ++k) {
            int off = -1, len = -1;
            chunkOf(k, n, nc, &off, &len);
            if (len < 0 || off < 0 || off + len > n) { ok = false; break; }
            if (len > 0 && off != prevEnd)           { ok = false; break; }
            if (len > 0) prevEnd = off + len;
            covered += len;
        }
        if (!ok || covered != n) {
            printf("  chunkOf FAILED for n=%d nChunks=%d (covered %lld)\n", n, nc, covered);
            return false;
        }
    }
    return true;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    const int    N     = 12582917;
    const size_t bytes = (size_t)N * sizeof(float);
    g_n     = N;
    g_iters = 3800;

    float *hPin = nullptr, *hPinOut = nullptr, *hPag = nullptr, *hPagOut = nullptr;
    CHECK(cudaMallocHost(&hPin,    bytes));
    CHECK(cudaMallocHost(&hPinOut, bytes));
    hPag    = (float*)malloc(bytes);
    hPagOut = (float*)malloc(bytes);
    if (!hPag || !hPagOut) { printf("host allocation failed\n"); return 1; }
    CHECK(cudaMalloc(&g_dIn,  bytes));
    CHECK(cudaMalloc(&g_dOut, bytes));
    for (int i = 0; i < N; ++i) hPin[i] = hPag[i] = 0.5f + (float)(i % 1021) * 1.0e-4f;

    const size_t nWarm = (size_t)64 * 1024 * 1024;
    float *wA, *wB, *sink;
    CHECK(cudaMalloc(&wA, nWarm * 4));
    CHECK(cudaMalloc(&wB, nWarm * 4));
    CHECK(cudaMalloc(&sink, 4));
    CHECK(cudaMemset(wA, 0, nWarm * 4));
    for (int i = 0; i < N_STREAMS; ++i) CHECK(cudaStreamCreate(&g_s[i]));
    cudaEvent_t evA, evB;
    CHECK(cudaEventCreate(&evA));
    CHECK(cudaEventCreate(&evB));

    printf("device                 : %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("asyncEngineCount       : %d\n", prop.asyncEngineCount);
    printf("concurrentKernels      : %d\n", prop.concurrentKernels);
    printf("problem                : N = %d floats = %.1f MB, %d FFMA/element\n",
           N, bytes / 1048576.0, g_iters);
    printf("pipeline               : %d chunks over %d streams\n\n", N_CHUNKS, N_STREAMS);

    printf("warming up (1500 ms stream + 500 ms compute) ...\n");
    warmUp(wA, wB, nWarm, sink);
    printf("done.\n\n");

    /* ---- ONE rotated sweep over every timed configuration ---------
     * Spec 12.1 / 12.9: the three phase times, the serial baseline and
     * the reader's pipeline are all timed inside the SAME rotated sweep.
     * Every ratio this harness scores -- including the bound, which is a
     * function of the phase times -- is therefore computed between
     * samples taken at one operating point, which is what spec 12.5b
     * says removes the need for a separate operating-point guard.
     *   cfg 0  H2D alone      cfg 3  serial, three phases back to back
     *   cfg 1  kernel alone   cfg 4  your pipeline, pinned
     *   cfg 2  D2H alone      cfg 5  your pipeline, pageable
     * SWEEPS == NCFG so every configuration leads once (spec 12.9).   */
    const int NCFG = 6;
    float  best[6];
    float  h2d = 0, ker = 0, d2h = 0;
    double sp = 0.0, spPag = 0.0, myBound = 0.0;
    for (int i = 0; i < NCFG; ++i) best[i] = 1e30f;
    for (int sweep = 0; sweep < NCFG; ++sweep) {
        for (int q = 0; q < NCFG; ++q) {
            int c = (q + sweep) % NCFG;
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(evA, 0));
            switch (c) {
                case 0:
                    CHECK(cudaMemcpy(g_dIn, hPin, bytes, cudaMemcpyHostToDevice));
                    break;
                case 1:
                    condition<<<(N + 255) / 256, 256>>>(g_dIn, g_dOut, N, g_iters);
                    CHECK(cudaGetLastError());
                    break;
                case 2:
                    CHECK(cudaMemcpy(hPinOut, g_dOut, bytes, cudaMemcpyDeviceToHost));
                    break;
                case 3:
                    serialRun(hPin, hPinOut);
                    break;
                case 4:
                    readerPipeline(hPin, hPinOut, N_CHUNKS);
                    waitForPipeline();
                    break;
                default:
                    readerPipeline(hPag, hPagOut, N_CHUNKS);
                    waitForPipeline();
                    break;
            }
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(evB, 0));
            CHECK(cudaEventSynchronize(evB));
            float m; CHECK(cudaEventElapsedTime(&m, evA, evB));
            if (m < best[c]) best[c] = m;
        }
    }
    h2d = best[0]; ker = best[1]; d2h = best[2];

    printf("=== measured phase times (pinned, whole buffer) ===============\n");
    printf("  H2D    %8.3f ms  (%.1f GB/s)\n", h2d, bytes / h2d / 1e6);
    printf("  kernel %8.3f ms\n", ker);
    printf("  D2H    %8.3f ms  (%.1f GB/s)\n", d2h, bytes / d2h / 1e6);
    printf("  sum    %8.3f ms\n", h2d + ker + d2h);
    printf("  serial, three phases back to back: %8.3f ms\n\n", best[3]);

    if (peek(&PRED_BUCKET) < 1 || peek(&PRED_BUCKET) > 4) {
        printf("  NOTE: with the TODOs still blank, two of the six timed\n");
        printf("  configurations do nothing at all, which lets the clock sag\n");
        printf("  between the others.  Use the numbers above to decide an ORDER\n");
        printf("  OF MAGNITUDE; the scored run measures them again with your\n");
        printf("  pipeline in place.\n\n");
        printf("Set TODO 5a (PRED_BUCKET) first.\n");
        goto cleanup;
    }

    {
        int score = 0;

        printf("=== TODO 1: chunkOf() structural test =========================\n");
        bool chunkOk = checkChunkOf();
        printf("  %s\n", chunkOk ? "ok (+2)" : "FAILED (+0)");
        if (chunkOk) score += 2;
        printf("  TODO 2 diagnostic -- your chunk -> stream map:");
        for (int k = 0; k < (N_CHUNKS < 12 ? N_CHUNKS : 12); ++k)
            printf(" %d", streamFor(k, N_STREAMS));
        printf(" ...\n\n");

        printf("=== TODO 5b: pipelineBound() reference check ==================\n");
        {
            const double probes[5][4] = {
                {4.0, 6.0, 4.0, 1.0}, {4.0, 6.0, 4.0, 2.0}, {10.0, 1.0, 10.0, 1.0},
                {1.0, 10.0, 1.0, 1.0}, {2.0, 3.0, 7.0, 2.0}
            };
            uint64_t h = 1469598103934665603ULL;
            for (int i = 0; i < 5; ++i) {
                double v = pipelineBound(probes[i][0], probes[i][1], probes[i][2],
                                         (int)probes[i][3]);
                printf("  bound(H=%.1f K=%.1f D=%.1f engines=%d) = %.4f\n",
                       probes[i][0], probes[i][1], probes[i][2], (int)probes[i][3], v);
                h = fnv1a(h, (long long)llround(v * 1.0e4));
            }
            bool bok = (h == 0xd7dde7f86c547c0bULL);
            printf("  %s\n\n", bok ? "matches the reference (+2)" : "does NOT match the reference (+0)");
            if (bok) score += 2;
        }

        myBound = pipelineBound(h2d, ker, d2h, prop.asyncEngineCount);
        printf("  your bound for the measured phases: %.3fx\n\n", myBound);

        printf("=== timing (one rotated sweep of 6 configurations, min of 6) ===\n");
        sp    = best[3] / best[4];
        spPag = best[3] / best[5];
        printf("  serial                      %9.3f ms\n", best[3]);
        printf("  your pipeline, pinned       %9.3f ms   %.2fx\n", best[4], sp);
        printf("  your pipeline, pageable     %9.3f ms   %.2fx   (not scored)\n", best[5], spPag);
        printf("\n");

        /* ---- correctness pass (separate from timing) --------------- */
        printf("=== correctness ===============================================\n");
        for (int i = 0; i < N; ++i) hPinOut[i] = -1.0f;
        readerPipeline(hPin, hPinOut, N_CHUNKS);
        waitForPipeline();
        CHECK(cudaDeviceSynchronize());
        int bad = 0; double worst = 0.0; int untouched = 0;
        for (int i = 0; i < N; i += 4093) {
            if (hPinOut[i] == -1.0f) ++untouched;
            float v = hPin[i];
            for (int k = 0; k < g_iters; ++k) v = fmaf(v, 1.000001f, 1.0e-7f);
            double e = fabs((double)v - (double)hPinOut[i]);
            if (!(e <= 1e-5 * fmax(1.0, fabs((double)v)))) ++bad;
            if (e > worst) worst = e;
        }
        /* the very last element is the one a truncating chunk plan drops */
        float vLast = hPin[N - 1];
        for (int k = 0; k < g_iters; ++k) vLast = fmaf(vLast, 1.000001f, 1.0e-7f);
        bool lastOk = fabs((double)vLast - (double)hPinOut[N - 1])
                      <= 1e-5 * fmax(1.0, fabs((double)vLast));
        printf("  sampled %d elements: %d never written, %d wrong, worst %.3e\n",
               (N + 4092) / 4093, untouched, bad, worst);
        printf("  last element (index %d): %s\n", N - 1, lastOk ? "ok" : "WRONG/unwritten");
        bool correct = (bad == 0) && (untouched == 0) && lastOk;
        printf("  %s\n\n", correct ? "ok (+2)" : "FAILED (+0)");
        if (correct) score += 2;

        /* ---- performance gate ------------------------------------- */
        printf("=== performance ===============================================\n");
        bool perfOk = correct && (myBound > 1.0) && (sp >= 0.70 * myBound);
        printf("  measured %.3fx against your bound %.3fx = %.0f%% of bound\n",
               sp, myBound, (myBound > 0.0) ? 100.0 * sp / myBound : 0.0);
        printf("  %s\n\n", perfOk ? "ok (+2)" : "FAILED -- need >= 70% of your own bound (+0)");
        if (perfOk) score += 2;

        /* ---- prediction ------------------------------------------- */
        int actualBucket = (sp < 1.10) ? 1 : (sp < 1.30) ? 2 : (sp < 2.00) ? 3 : 4;
        bool predOk = (peek(&PRED_BUCKET) == actualBucket);
        printf("=== prediction ================================================\n");
        printf("  you said bucket %d, the measurement landed in bucket %d\n",
               peek(&PRED_BUCKET), actualBucket);
        printf("  %s\n\n", predOk ? "ok (+2)" : "FAILED (+0)");
        if (predOk) score += 2;

        printf("SCORE: %d/10\n", score);
        printf("OVERALL: %s\n", score == 10 ? "PASS" : "FAIL");
    }

cleanup:
    for (int i = 0; i < N_STREAMS; ++i) CHECK(cudaStreamDestroy(g_s[i]));
    CHECK(cudaEventDestroy(evA)); CHECK(cudaEventDestroy(evB));
    CHECK(cudaFree(wA)); CHECK(cudaFree(wB)); CHECK(cudaFree(sink));
    CHECK(cudaFree(g_dIn)); CHECK(cudaFree(g_dOut));
    CHECK(cudaFreeHost(hPin)); CHECK(cudaFreeHost(hPinOut));
    free(hPag); free(hPagOut);
    CHECK(cudaDeviceReset());
    return 0;
}
