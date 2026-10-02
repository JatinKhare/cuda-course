/* =====================================================================
 * Module 24 / Exercise 02 — The concurrency that isn't there (debugging)
 *
 * SYMPTOM
 *   `pipeline()` below chunks a 48 MB transform across four streams, the
 *   way a copy/compute overlap pipeline is supposed to be written.  It
 *   produces the correct answer.  It should run noticeably faster than
 *   the serial version that does the same work in three phases.
 *
 *   It does not.  It lands far short of what the phase times permit, and
 *   with some chunk counts it is slower than the serial version.  There is
 *   no error, no warning, and no wrong number anywhere.  compute-sanitizer
 *   is clean.
 *
 *   There are THREE independent causes.  Find them, fix them, and leave
 *   what the program computes unchanged.
 *
 * TOOLS
 *   The program builds its own diagnosis out of CUDA events.  The
 *   OP_BEGIN / OP_END macros inside pipeline() record a start and an end
 *   timestamp around each operation it issues; the harness replays them
 *   into a printed timeline.  TODO 4 asks you to reduce that timeline to
 *   one number that says whether anything overlapped.
 *   **Leave the OP_BEGIN / OP_END calls bracketing the operations they
 *   bracket now.**  They compile to nothing while the harness is not
 *   tracing, and they are your only instrument.
 *   Module 25 covers events properly; here they are stopwatches.
 *   Nsight Systems shows all of this in one capture; the command is in
 *   the solution notes.  Module 22 owns that tool.
 *
 * BUILD
 *   nvcc -arch=sm_89 -O3 -std=c++17 -o exercise02.exe exercise02.cu
 *   (and, for TODO 5a, also:)
 *   nvcc -arch=sm_89 -O3 -std=c++17 --default-stream per-thread -o exercise02_pt.exe exercise02.cu
 * RUN
 *   exercise02.exe
 *
 * WHAT IS SCORED (10 points; OVERALL: PASS requires all of them)
 *   3  DIAG_A/B/C name the three real causes (any order)
 *   2  overlapFactor() agrees with the reference on a synthetic timeline
 *   3  the repaired pipeline is still correct AND reaches >= 1.35x serial
 *   1  PRED_PT_CURED: how many causes --default-stream per-thread removes
 *   1  PRED_BUCKET: the bucket the repaired speedup lands in
 * ===================================================================== */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
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
 * TODO 1 — DIAGNOSIS.  Pick the THREE statements that are true of this
 * program and that explain the missing concurrency.  Any order.
 *
 *   1  The kernel's grid is too small to fill the GPU, so chunks from
 *      different streams serialize on the SMs.
 *   2  An operation inside the loop is issued to the legacy default
 *      stream, which synchronizes with every other blocking stream.
 *   3  Chunks are assigned to streams round-robin; that mapping puts
 *      work that must be ordered into different streams.
 *   4  A device allocation inside the loop forces a device-wide implicit
 *      synchronization.
 *   5  A host buffer that a copy touches is pageable, so that copy cannot
 *      overlap with anything.
 *   6  There are four streams and fewer copy engines than that; a stream
 *      count above the copy-engine count serializes the pipeline.
 *   7  cudaMemcpyAsync is only asynchronous for transfers smaller than
 *      64 KB, and these chunks are larger.
 *   8  Every stream's kernel writes the same device buffer, so the driver
 *      inserts a dependency to prevent a data race.
 * =================================================================== */
static const int DIAG_A = 0;   /* YOUR CODE HERE */
static const int DIAG_B = 0;   /* YOUR CODE HERE */
static const int DIAG_C = 0;   /* YOUR CODE HERE */

/* ===================================================================
 * TODO 5 — TWO PREDICTIONS.
 *
 * (a) PRED_PT_CURED: of the three causes you named, how many are
 *     REMOVED AS CAUSES if the program is compiled with
 *     --default-stream per-thread and nothing else is changed?
 *     Answer 0, 1, 2 or 3.
 *
 *     Build the second binary named at the top of this file and run it.
 *     WARNING: the elapsed time is not the evidence you want.  Three
 *     independent causes mask each other -- removing one of them can
 *     leave the measured time exactly where it was, because either of
 *     the other two alone is enough to serialize the whole pipeline.
 *     Answer the question about causes, not about milliseconds.
 *
 * (b) PRED_BUCKET: once all three are repaired, which bucket does the
 *     speedup (serial / repaired) land in?
 *        1 : below 1.20x    2 : 1.20 .. 1.35x
 *        3 : 1.35 .. 2.00x  4 : above 2.00x
 * =================================================================== */
static const int PRED_PT_CURED = -1;   /* YOUR CODE HERE (0..3) */
static const int PRED_BUCKET   = 0;    /* YOUR CODE HERE (1..4) */

static const int N_STREAMS = 4;
static const int N_CHUNKS  = 16;
static const int BLOCK     = 256;
#define CLIP_T 1.0e6f

/* ---- payload (do not modify) -------------------------------------- */
__global__ void condition(const float* __restrict__ in, float* __restrict__ out,
                          int n, int iters, int* clip, float* blockFirst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    for (int k = 0; k < iters; ++k) v = fmaf(v, 1.000001f, 1.0e-7f);
    out[i] = v;
    if (v > CLIP_T) atomicAdd(clip, 1);
    if (threadIdx.x == 0) blockFirst[blockIdx.x] = v;
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

/* ---- state --------------------------------------------------------- */
static int           g_n, g_iters;
static float        *g_dIn, *g_dOut;
static int          *g_dClip;         /* one counter per chunk            */
static float        *g_dScratch;      /* per-block scratch, laid out as
                                       * N_CHUNKS slices of g_stride floats;
                                       * slice k starts at
                                       * g_dScratch + k*g_stride           */
static int           g_stride;        /* floats per chunk slice           */
static cudaStream_t  g_s[8];
static float        *g_hIn;           /* pinned, allocated by the harness */
static float        *g_hOut;          /* TODO 3 owns this one             */

/* ---- the instrument (harness-owned; do not move the call sites) ----- */
#define TRACE_MAX 64
static int          g_trace = 0;
static int          g_nOps  = 0;
static cudaEvent_t  g_sEv[TRACE_MAX], g_eEv[TRACE_MAX];
static const char*  g_tag[TRACE_MAX];

#define OP_BEGIN(st, name)                                                     \
    do { if (g_trace && g_nOps < TRACE_MAX) {                                  \
             g_tag[g_nOps] = (name);                                           \
             CHECK(cudaEventRecord(g_sEv[g_nOps], (st)));                      \
         } } while (0)
#define OP_END(st)                                                             \
    do { if (g_trace && g_nOps < TRACE_MAX) {                                  \
             CHECK(cudaEventRecord(g_eEv[g_nOps], (st)));                      \
             ++g_nOps;                                                         \
         } } while (0)

static void chunkOf(int k, int n, int nChunks, int* off, int* len)
{
    int c = (n + nChunks - 1) / nChunks;
    int o = k * c; if (o > n) o = n;
    int l = c; if (o + l > n) l = n - o; if (l < 0) l = 0;
    *off = o; *len = l;
}

/* ===================================================================
 * TODO 3 — the host output buffer.
 *
 * allocHostOut() produces the buffer that every device-to-host copy in
 * the pipeline writes into; freeHostOut() releases it.  Make them right.
 * (The input buffer, allocated by the harness in main(), is already
 *  right -- read it if you need a model.)
 * =================================================================== */
static float* allocHostOut(size_t bytes)
{
    return (float*)malloc(bytes);          /* YOUR CODE HERE */
}
static void freeHostOut(float* p)
{
    free(p);                               /* YOUR CODE HERE */
}

/* ===================================================================
 * TODO 2 — the pipeline.
 *
 * Repair it.  Constraints, all of them checked:
 *   - every chunk is still transformed into g_hOut;
 *   - g_dClip[k] is still set to zero before chunk k's kernel runs, and
 *     the kernel still counts into it;
 *   - the kernel still writes one float per block into THIS CHUNK'S slice
 *     of the scratch buffer -- slice k begins at g_dScratch + k*g_stride
 *     and is g_stride floats long, which is >= any chunk's block count;
 *   - the OP_BEGIN / OP_END pairs still bracket the same operations;
 *   - nothing in this function may synchronize before the final wait.
 * =================================================================== */
static void pipeline(int nChunks)
{
    for (int k = 0; k < nChunks; ++k) {
        int off, len;
        chunkOf(k, g_n, nChunks, &off, &len);
        if (len <= 0) continue;
        cudaStream_t st = g_s[k % N_STREAMS];
        int blocks = (len + BLOCK - 1) / BLOCK;

        /* this chunk's clip counter starts at zero */
        CHECK(cudaMemsetAsync(g_dClip + k, 0, sizeof(int)));

        /* a per-block scratch exactly as long as this chunk's grid.  Chunk
         * lengths differ (the last one is ragged), so it is sized per chunk. */
        float* scratch = nullptr;
        CHECK(cudaMalloc(&scratch, (size_t)blocks * sizeof(float)));

        OP_BEGIN(st, "H2D");
        CHECK(cudaMemcpyAsync(g_dIn + off, g_hIn + off, (size_t)len * sizeof(float),
                              cudaMemcpyHostToDevice, st));
        OP_END(st);

        OP_BEGIN(st, "ker");
        condition<<<blocks, BLOCK, 0, st>>>(g_dIn + off, g_dOut + off, len, g_iters,
                                            g_dClip + k, scratch);
        CHECK(cudaGetLastError());
        OP_END(st);

        /* the scratch is only used by this chunk's kernel, so release it as
         * soon as the kernel has been issued */
        CHECK(cudaFree(scratch));

        if (k >= 1) {
            int o2, l2;
            chunkOf(k - 1, g_n, nChunks, &o2, &l2);
            if (l2 > 0) {
                cudaStream_t sp = g_s[(k - 1) % N_STREAMS];
                OP_BEGIN(sp, "D2H");
                CHECK(cudaMemcpyAsync(g_hOut + o2, g_dOut + o2, (size_t)l2 * sizeof(float),
                                      cudaMemcpyDeviceToHost, sp));
                OP_END(sp);
            }
        }
    }
    int oL, lL;
    chunkOf(nChunks - 1, g_n, nChunks, &oL, &lL);
    if (lL > 0) {
        cudaStream_t sp = g_s[(nChunks - 1) % N_STREAMS];
        OP_BEGIN(sp, "D2H");
        CHECK(cudaMemcpyAsync(g_hOut + oL, g_dOut + oL, (size_t)lL * sizeof(float),
                              cudaMemcpyDeviceToHost, sp));
        OP_END(sp);
    }
    for (int i = 0; i < N_STREAMS; ++i) CHECK(cudaStreamSynchronize(g_s[i]));
}

/* ===================================================================
 * TODO 4 — the instrument.
 *
 * The harness gives you, for every operation the pipeline issued, the
 * GPU time at which it started and the time at which it finished, both
 * in milliseconds relative to one reference point.  Reduce that timeline
 * to ONE number that answers "did anything overlap?":
 *
 *     1.0  nothing ever ran at the same time as anything else
 *     2.0  on average two operations were in flight at once
 *
 * `nOps` operations, start[i] <= end[i], unsorted, from several streams.
 * The number must not change if the program idles before or after the
 * pipeline.  Use the span from the earliest start to the latest end as
 * the denominator, so that a gap inside the pipeline counts against it.
 * Return 0.0 if nOps <= 0 or the span is zero.
 * =================================================================== */
static double overlapFactor(const double* start, const double* end, int nOps)
{
    /* YOUR CODE HERE */
    (void)start; (void)end; (void)nOps;
    return 0.0;
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

static void serialRun(void)
{
    int blocks = (g_n + BLOCK - 1) / BLOCK;
    CHECK(cudaMemcpy(g_dIn, g_hIn, (size_t)g_n * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(g_dClip, 0, sizeof(int)));
    condition<<<blocks, BLOCK>>>(g_dIn, g_dOut, g_n, g_iters, g_dClip, g_dScratch);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(g_hOut, g_dOut, (size_t)g_n * 4, cudaMemcpyDeviceToHost));
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

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    const int    N     = 12582917;
    const size_t bytes = (size_t)N * sizeof(float);
    g_n = N; g_iters = 3800;

    CHECK(cudaMallocHost(&g_hIn, bytes));          /* pinned: the model */
    g_hOut = allocHostOut(bytes);
    if (!g_hOut) { printf("host allocation failed\n"); return 1; }
    CHECK(cudaMalloc(&g_dIn,  bytes));
    CHECK(cudaMalloc(&g_dOut, bytes));
    CHECK(cudaMalloc(&g_dClip, (size_t)N_CHUNKS * sizeof(int)));
    {   /* one scratch slice per chunk, long enough for the largest chunk's
         * grid and also for the whole-buffer grid the serial version uses */
        int cMax = (N + N_CHUNKS - 1) / N_CHUNKS;
        g_stride = (cMax + BLOCK - 1) / BLOCK;
        size_t slots = (size_t)N_CHUNKS * g_stride;
        size_t full  = (size_t)((N + BLOCK - 1) / BLOCK);
        if (full > slots) slots = full;
        CHECK(cudaMalloc(&g_dScratch, slots * sizeof(float)));
    }
    for (int i = 0; i < N; ++i) g_hIn[i] = 0.5f + (float)(i % 1021) * 1.0e-4f;

    const size_t nWarm = (size_t)64 * 1024 * 1024;
    float *wA, *wB, *sink;
    CHECK(cudaMalloc(&wA, nWarm * 4));
    CHECK(cudaMalloc(&wB, nWarm * 4));
    CHECK(cudaMalloc(&sink, 4));
    CHECK(cudaMemset(wA, 0, nWarm * 4));
    for (int i = 0; i < N_STREAMS; ++i) CHECK(cudaStreamCreate(&g_s[i]));

    cudaEvent_t evA, evB, t0;
    CHECK(cudaEventCreate(&evA));
    CHECK(cudaEventCreate(&evB));
    CHECK(cudaEventCreate(&t0));
    for (int i = 0; i < TRACE_MAX; ++i) {
        CHECK(cudaEventCreate(&g_sEv[i]));
        CHECK(cudaEventCreate(&g_eEv[i]));
    }
    double stA[TRACE_MAX], enA[TRACE_MAX];

    float  serialMs = 1e30f, pipeMs = 1e30f;
    int    nOps = 0;
    double sp = 0.0;

    printf("device            : %s, asyncEngineCount = %d, %d SMs\n",
           prop.name, prop.asyncEngineCount, prop.multiProcessorCount);
    printf("problem           : N = %d floats = %.1f MB, %d FFMA/element\n",
           N, bytes / 1048576.0, g_iters);
    printf("pipeline          : %d chunks over %d streams\n", N_CHUNKS, N_STREAMS);
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
    printf("default stream    : PER-THREAD (built with --default-stream per-thread)\n\n");
#else
    printf("default stream    : LEGACY (nvcc default)\n\n");
#endif
    printf("warming up (1500 ms stream + 500 ms compute) ...\n");
    warmUp(wA, wB, nWarm, sink);
    printf("done.\n\n");

    /* ---- the symptom.  Serial and pipeline are timed against each
     * other inside one rotated sweep, so the ratio this harness scores
     * is taken at a single operating point (spec 12.1/12.5b/12.9). ---- */
    for (int sweep = 0; sweep < 3; ++sweep)
        for (int q = 0; q < 2; ++q) {
            int c = (q + sweep) % 2;
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(evA, 0));
            if (c == 0) serialRun(); else pipeline(N_CHUNKS);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(evB, 0));
            CHECK(cudaEventSynchronize(evB));
            float m; CHECK(cudaEventElapsedTime(&m, evA, evB));
            if (c == 0) { if (m < serialMs) serialMs = m; }
            else        { if (m < pipeMs)   pipeMs   = m; }
        }
    sp = serialMs / pipeMs;
    printf("=== the symptom ================================================\n");
    printf("  serial, three phases, one stream : %8.3f ms\n", serialMs);
    printf("  'pipelined', %d streams           : %8.3f ms   %.2fx\n",
           N_STREAMS, pipeMs, sp);
    printf("\n");

    /* ---- the event timeline, from the pipeline as it stands -------- */
    printf("=== the event timeline (first 12 operations) ===================\n");
    {
        g_trace = 1; g_nOps = 0;
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(t0, 0));
        pipeline(N_CHUNKS);
        CHECK(cudaDeviceSynchronize());
        g_trace = 0;
        nOps = g_nOps;
        for (int i = 0; i < nOps; ++i) {
            float a, b;
            CHECK(cudaEventElapsedTime(&a, t0, g_sEv[i]));
            CHECK(cudaEventElapsedTime(&b, t0, g_eEv[i]));
            stA[i] = a; enA[i] = b;
        }
        /* one column per span/48, so the whole run always fits the width */
        double span = 0.0;
        for (int i = 0; i < nOps; ++i) if (enA[i] > span) span = enA[i];
        double col = (span > 0.0) ? span / 48.0 : 1.0;
        printf("  %-3s %-4s %9s %9s  timeline (1 column = %.3f ms)\n",
               "op", "kind", "start", "end", col);
        for (int i = 0; i < nOps && i < 12; ++i) {
            printf("  %-3d %-4s %9.3f %9.3f  ", i, g_tag[i], stA[i], enA[i]);
            int a = (int)(stA[i] / col), b = (int)(enA[i] / col);
            if (a < 0) a = 0;
            if (b < a) b = a;
            for (int c = 0; c < a && c < 48; ++c) putchar('.');
            for (int c = a; c <= b && c < 48; ++c) putchar('#');
            putchar('\n');
        }
        printf("  ... %d operations in total.  If the bars never share a column,\n", nOps);
        printf("  nothing overlapped.\n\n");
    }

    if (peek(&DIAG_A) == 0 || peek(&DIAG_B) == 0 || peek(&DIAG_C) == 0 ||
        peek(&PRED_PT_CURED) < 0 || peek(&PRED_BUCKET) == 0) {
        printf("Set TODO 1 (DIAG_A/B/C) and TODO 5 (both predictions) first.\n");
        goto cleanup;
    }

    {
        int score = 0;

        printf("=== TODO 1: diagnosis ==========================================\n");
        {
            int d[3] = {peek(&DIAG_A), peek(&DIAG_B), peek(&DIAG_C)};
            for (int i = 0; i < 3; ++i)
                for (int j = i + 1; j < 3; ++j)
                    if (d[j] < d[i]) { int t = d[i]; d[i] = d[j]; d[j] = t; }
            uint64_t h = 1469598103934665603ULL;
            for (int i = 0; i < 3; ++i) h = fnv1a(h, d[i]);
            bool ok = (h == 0x2c15fe3416432c40ULL);
            printf("  you named %d, %d, %d : %s\n", d[0], d[1], d[2],
                   ok ? "correct (+3)" : "not the three causes (+0)");
            if (ok) score += 3;
        }
        printf("\n");

        printf("=== TODO 4: overlapFactor() ====================================\n");
        {
            const double ss[4] = {0.0, 1.0, 2.0, 4.0};
            const double ee[4] = {3.0, 5.0, 6.0, 6.0};
            double got  = overlapFactor(ss, ee, 4);
            double want = (3.0 + 4.0 + 4.0 + 2.0) / 6.0;
            bool ok = fabs(got - want) <= 1e-6 * fmax(1.0, fabs(want));
            printf("  synthetic timeline -> you %.6f, reference %.6f : %s\n",
                   got, want, ok ? "ok (+2)" : "wrong (+0)");
            if (ok) score += 2;
            printf("  overlapFactor on the traced run above : %.3f\n",
                   overlapFactor(stA, enA, nOps));
            printf("  (3 operations per chunk share one copy engine and one GPU;\n");
            printf("   a repaired pipeline on this device reports well above 1.0.)\n");
        }
        printf("\n");

        printf("=== TODO 2/3: the repair =======================================\n");
        printf("  serial   %8.3f ms\n", serialMs);
        printf("  pipeline %8.3f ms   %.2fx\n", pipeMs, sp);

        /* correctness, in a separate pass.  The clip counters are poisoned
         * first, so a pipeline that simply dropped the memset fails. */
        for (int i = 0; i < N; ++i) g_hOut[i] = -1.0f;
        CHECK(cudaMemset(g_dClip, 0xFF, (size_t)N_CHUNKS * sizeof(int)));
        pipeline(N_CHUNKS);
        CHECK(cudaDeviceSynchronize());
        int bad = 0, untouched = 0;
        for (int i = 0; i < N; i += 4093) {
            if (g_hOut[i] == -1.0f) ++untouched;
            float v = g_hIn[i];
            for (int k = 0; k < g_iters; ++k) v = fmaf(v, 1.000001f, 1.0e-7f);
            if (fabs((double)v - (double)g_hOut[i]) > 1e-5 * fmax(1.0, fabs((double)v))) ++bad;
        }
        int hClip[64];
        CHECK(cudaMemcpy(hClip, g_dClip, (size_t)N_CHUNKS * sizeof(int),
                         cudaMemcpyDeviceToHost));
        bool clipOk = true;
        for (int k = 0; k < N_CHUNKS; ++k) if (hClip[k] != 0) clipOk = false;
        /* every chunk must have written ITS OWN scratch slice.  This check is
         * order independent: no chunk may land in another chunk's slice. */
        size_t slots = (size_t)N_CHUNKS * g_stride;
        float* hbf = (float*)malloc(slots * sizeof(float));
        CHECK(cudaMemcpy(hbf, g_dScratch, slots * sizeof(float), cudaMemcpyDeviceToHost));
        bool scratchOk = true;
        for (int kc = 0; kc < N_CHUNKS; ++kc) {
            int oc, lc; chunkOf(kc, N, N_CHUNKS, &oc, &lc);
            int nb = (lc + BLOCK - 1) / BLOCK;
            for (int b = 0; b < nb; ++b) {
                float v = g_hIn[oc + b * BLOCK];
                for (int k = 0; k < g_iters; ++k) v = fmaf(v, 1.000001f, 1.0e-7f);
                double got = hbf[(size_t)kc * g_stride + b];
                if (fabs((double)v - got) > 1e-5 * fmax(1.0, fabs((double)v)))
                    scratchOk = false;
            }
        }
        free(hbf);
        bool fixOk = (bad == 0) && (untouched == 0) && clipOk && scratchOk && (sp >= 1.35);
        printf("  %d wrong, %d unwritten, clip counters %s, per-block scratch %s\n",
               bad, untouched, clipOk ? "ok" : "BROKEN", scratchOk ? "ok" : "BROKEN");
        printf("  %s\n\n", fixOk ? "ok (+3)"
                                 : "FAILED -- need correct, behaviour preserved, and >= 1.35x (+0)");
        if (fixOk) score += 3;

        printf("=== TODO 5: the two predictions ================================\n");
        {
            uint64_t h = fnv1a(1469598103934665603ULL, peek(&PRED_PT_CURED));
            bool a = (h == 0x29034675a49f07c2ULL);
            printf("  (a) per-thread default stream removes %d of the three : %s\n",
                   peek(&PRED_PT_CURED), a ? "ok (+1)" : "wrong (+0)");
            if (a) score += 1;
            int actual = (sp < 1.20) ? 1 : (sp < 1.35) ? 2 : (sp < 2.00) ? 3 : 4;
            bool b = (peek(&PRED_BUCKET) == actual);
            printf("  (b) bucket: you said %d, measurement is in %d : %s\n",
                   peek(&PRED_BUCKET), actual, b ? "ok (+1)" : "wrong (+0)");
            if (b) score += 1;
        }

        printf("\nSCORE: %d/10\n", score);
        printf("OVERALL: %s\n", score == 10 ? "PASS" : "FAIL");
    }

cleanup:
    for (int i = 0; i < TRACE_MAX; ++i) {
        CHECK(cudaEventDestroy(g_sEv[i]));
        CHECK(cudaEventDestroy(g_eEv[i]));
    }
    for (int i = 0; i < N_STREAMS; ++i) CHECK(cudaStreamDestroy(g_s[i]));
    CHECK(cudaEventDestroy(evA)); CHECK(cudaEventDestroy(evB)); CHECK(cudaEventDestroy(t0));
    CHECK(cudaFree(wA)); CHECK(cudaFree(wB)); CHECK(cudaFree(sink));
    CHECK(cudaFree(g_dIn)); CHECK(cudaFree(g_dOut));
    CHECK(cudaFree(g_dClip)); CHECK(cudaFree(g_dScratch));
    CHECK(cudaFreeHost(g_hIn));
    freeHostOut(g_hOut);
    CHECK(cudaDeviceReset());
    return 0;
}
