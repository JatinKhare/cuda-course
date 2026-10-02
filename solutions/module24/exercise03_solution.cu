/* =====================================================================
 * Module 24 / Exercise 03 SOLUTION — Express a dependency graph, then
 *                     find out whether the hardware cares
 *
 * GOAL
 *   Nine operations have to run.  Their dependencies form a DAG, not a
 *   line:
 *
 *       n0 H2D(x) --> n1 preA1(x) --> n2 preA2(x) --+
 *                                                   +--> n6 join(x,y->z)
 *       n3 H2D(y) --> n4 preB1(y) --> n5 preB2(y) --+          |
 *                                                              v
 *                                             n8 D2H(p) <-- n7 reduce(z->p)
 *
 *   Written as one stream that is nine operations long.  Your job is to
 *   express the DAG itself -- with the FEWEST streams and the FEWEST
 *   events that still expose every independent pair -- and then to find
 *   out what that buys.
 *
 *   The harness runs the whole thing TWICE, at two problem sizes:
 *     SMALL  each kernel launches 16 blocks   (0.4 blocks per SM)
 *     LARGE  each kernel launches 640 blocks  (16 blocks per SM)
 *   The arithmetic per element is identical.  Only the grid changes.
 *
 * BUILD
 *   nvcc -arch=sm_89 -O3 -std=c++17 -o exercise03.exe exercise03.cu
 * RUN
 *   exercise03.exe
 *
 * WHAT IS SCORED (10 points; OVERALL: PASS requires all of them)
 *   2  MIN_STREAMS and MIN_EVENTS are the true minima for this DAG
 *   2  criticalPath() agrees with the reference on four synthetic inputs
 *   2  the reduction is correct at BOTH sizes
 *   2  the SMALL speedup reaches >= 70% of the measured critical-path bound
 *   1  PRED_SMALL matches the bucket the SMALL measurement lands in
 *   1  PRED_LARGE matches the bucket the LARGE measurement lands in
 *
 * NOTES
 *   - Module 25 covers events properly.  You need cudaEventRecord(event,
 *     stream) and cudaStreamWaitEvent(stream, event, 0) here purely as the
 *     mechanism for a cross-stream edge; nothing is being timed with them.
 *     Create them with cudaEventDisableTiming -- the harness already does.
 *   - All host buffers are pinned.  Module 26 owns pinned memory.
 *   - Do not change anything outside a "YOUR CODE HERE" region.
 * ===================================================================== */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
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
 * TODO 1 — THE DESIGN QUESTION.
 *
 * How many streams, and how many events, does this DAG need?
 *
 *   MIN_STREAMS : the smallest number of streams in which every pair of
 *                 operations that CAN run at the same time actually is
 *                 allowed to.  (One stream always works and exposes
 *                 nothing.  Nine streams work too and are not minimal.)
 *   MIN_EVENTS  : with that many streams, the smallest number of events
 *                 needed to enforce every dependency edge in the picture.
 *                 Count edges you get for free before you count events.
 *
 * The harness hands issueDag() exactly MIN_STREAMS streams and
 * MIN_EVENTS events -- no more.  Claim too few and you will not be able
 * to build a correct graph; claim too many and this TODO scores zero.
 * =================================================================== */
static const int MIN_STREAMS = 2;   /* the DAG is two chains wide        */
static const int MIN_EVENTS  = 1;   /* one cross-stream edge to enforce  */

/* ===================================================================
 * TODO 5 — TWO PREDICTIONS, before you run anything.
 *
 * Speedup = (one stream, all nine in order) / (your DAG).
 * Which bucket does each size land in?
 *
 *     1 : below 1.15x        2 : 1.15x .. 1.40x
 *     3 : 1.40x .. 2.00x     4 : above 2.00x
 *
 * The harness prints the nine measured node durations and your own
 * critical-path bound BEFORE it scores these.  Run it once with the
 * TODOs blank to collect them.  The two answers are not the same
 * number, and the reason is in Module 1, not in this module.
 * =================================================================== */
static const int PRED_SMALL = 3;    /* SOLUTION */
static const int PRED_LARGE = 1;    /* SOLUTION */

/* ---- fixed problem parameters -------------------------------------- */
static const int BLOCK        = 128;
static const int SMALL_BLOCKS = 16;
static const int LARGE_BLOCKS = 640;
static const int ITERS        = 500000;   /* dependent FFMAs per element  */

/* ---- payload kernels (do not modify) ------------------------------- */

/* One chain step, in place.  Dependent FFMAs, so the duration is a
 * function of ITERS and of how many warps are competing for a scheduler --
 * not of the data. */
__global__ void chainStep(float* p, int n, int iters, float a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = p[i];
    for (int k = 0; k < iters; ++k) v = fmaf(v, a, 1.0e-7f);
    p[i] = v;
}

/* The join: reads both chains, writes z.  Half the arithmetic. */
__global__ void joinKernel(const float* __restrict__ x, const float* __restrict__ y,
                           float* __restrict__ z, int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i] + y[i];
    for (int k = 0; k < iters; ++k) v = fmaf(v, 1.0000005f, 1.0e-7f);
    z[i] = v;
}

/* Fixed-order per-block tree reduction -> one partial per block. */
__global__ void reduceKernel(const float* __restrict__ z, int n, float* part)
{
    __shared__ float s[BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (i < n) ? z[i] : 0.0f;
    __syncthreads();
    for (int o = BLOCK / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = s[0];
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

/* ---- the chain constants, so host and device agree ----------------- */
static const float A1 = 1.0000010f, A2 = 1.0000013f;
static const float B1 = 1.0000020f, B2 = 1.0000021f;

/* ---- everything the DAG touches ------------------------------------ */
struct Bufs {
    float* hx;      /* pinned host input x                               */
    float* hy;      /* pinned host input y                               */
    float* hPart;   /* pinned host output, one float per block           */
    float* dx;
    float* dy;
    float* dz;
    float* dPart;
    int    n;       /* elements                                          */
    int    blocks;  /* grid for every kernel; also the length of dPart   */
};

/* ===================================================================
 * TODO 2 — the critical path.
 *
 * d[0..8] are the measured durations, in ms, of the nine nodes drawn at
 * the top of this file, in that numbering.  Return the length of the
 * longest path through the DAG -- the time the graph takes if every
 * operation that may run concurrently does, and nothing costs anything
 * to schedule.
 *
 * The harness checks this against a reference on four synthetic duration
 * vectors before it uses your answer for anything.
 * =================================================================== */
static double criticalPath(const double* d)
{
    /* The two preprocessing chains are the only parallelism in the graph.
     * Everything from the join onwards is a line, so it is pure addition. */
    double chainA = d[0] + d[1] + d[2];
    double chainB = d[3] + d[4] + d[5];
    double head   = (chainA > chainB) ? chainA : chainB;
    return head + d[6] + d[7] + d[8];
}

/* ===================================================================
 * TODO 3 — BUILD THE GRAPH.  This is the exercise.
 *
 * Issue all nine operations into the `nStreams` streams you were given,
 * using the `nEvents` events you were given, so that:
 *
 *   (a) every edge in the picture is enforced;
 *   (b) nothing that the picture leaves unordered is ordered by you;
 *   (c) the function does not block -- no synchronize of any kind, no
 *       allocation, no blocking cudaMemcpy;
 *   (d) when it returns, waiting on the streams is enough to know the
 *       whole graph is done.
 *
 * The nine operations, with the calls you need:
 *   n0  cudaMemcpyAsync(B.dx, B.hx, bytes, cudaMemcpyHostToDevice, s)
 *   n1  chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dx, B.n, ITERS, A1)
 *   n2  chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dx, B.n, ITERS, A2)
 *   n3  cudaMemcpyAsync(B.dy, B.hy, bytes, cudaMemcpyHostToDevice, s)
 *   n4  chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dy, B.n, ITERS, B1)
 *   n5  chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dy, B.n, ITERS, B2)
 *   n6  joinKernel<<<B.blocks, BLOCK, 0, s>>>(B.dx, B.dy, B.dz, B.n, ITERS/2)
 *   n7  reduceKernel<<<B.blocks, BLOCK, 0, s>>>(B.dz, B.n, B.dPart)
 *   n8  cudaMemcpyAsync(B.hPart, B.dPart, B.blocks*4, cudaMemcpyDeviceToHost, s)
 * with bytes = (size_t)B.n * sizeof(float).
 *
 * One trap worth naming, because it costs an afternoon: an event is a
 * marker in a stream, not a handle on an operation.  cudaEventRecord
 * captures the work that has ALREADY been issued into that stream, and
 * cudaStreamWaitEvent makes the waiting stream wait for exactly that
 * much.  Record too early and the wait is satisfied immediately and
 * enforces nothing -- and the program still prints the right answer most
 * of the time.
 * =================================================================== */
static void issueDag(const Bufs& B, cudaStream_t* s, int nStreams,
                     cudaEvent_t* ev, int nEvents)
{
    (void)nStreams; (void)nEvents;
    const size_t bytes = (size_t)B.n * sizeof(float);

    /* Chain B, start to finish, in stream 1.  Same-stream ordering gives
     * the two edges n3->n4 and n4->n5 for free: a stream is an ordered
     * queue, and that is the whole reason the minimum event count is one
     * and not six. */
    CHECK(cudaMemcpyAsync(B.dy, B.hy, bytes, cudaMemcpyHostToDevice, s[1]));
    chainStep<<<B.blocks, BLOCK, 0, s[1]>>>(B.dy, B.n, ITERS, B1);
    chainStep<<<B.blocks, BLOCK, 0, s[1]>>>(B.dy, B.n, ITERS, B2);

    /* The marker goes in AFTER the last node of chain B has been issued.
     * cudaEventRecord captures the stream's contents at the moment it is
     * called -- not at the moment the event is waited on. */
    CHECK(cudaEventRecord(ev[0], s[1]));

    /* Chain A, start to finish, in stream 0.  Nothing orders it against
     * chain B, which is exactly what the picture asks for. */
    CHECK(cudaMemcpyAsync(B.dx, B.hx, bytes, cudaMemcpyHostToDevice, s[0]));
    chainStep<<<B.blocks, BLOCK, 0, s[0]>>>(B.dx, B.n, ITERS, A1);
    chainStep<<<B.blocks, BLOCK, 0, s[0]>>>(B.dx, B.n, ITERS, A2);

    /* The one edge a stream cannot supply: n5 -> n6, across streams. */
    CHECK(cudaStreamWaitEvent(s[0], ev[0], 0));

    joinKernel<<<B.blocks, BLOCK, 0, s[0]>>>(B.dx, B.dy, B.dz, B.n, ITERS / 2);
    reduceKernel<<<B.blocks, BLOCK, 0, s[0]>>>(B.dz, B.n, B.dPart);
    CHECK(cudaMemcpyAsync(B.hPart, B.dPart, (size_t)B.blocks * sizeof(float),
                          cudaMemcpyDeviceToHost, s[0]));
    CHECK(cudaGetLastError());
}

/* ===================================================================
 * TODO 4 — the wait.  Block until the whole graph has finished and
 * B.hPart holds the result.
 * =================================================================== */
static void waitForDag(cudaStream_t* s, int nStreams)
{
    /* Stream 0 carries the terminal node, and stream 0 already waits on
     * stream 1 through the event, so synchronizing stream 0 alone is
     * sufficient.  Synchronizing both is cheap and says what is meant. */
    for (int i = 0; i < nStreams; ++i) CHECK(cudaStreamSynchronize(s[i]));
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

/* the reference serialization: all nine, one stream, topological order */
static void serialDag(const Bufs& B, cudaStream_t s)
{
    size_t bytes = (size_t)B.n * sizeof(float);
    CHECK(cudaMemcpyAsync(B.dx, B.hx, bytes, cudaMemcpyHostToDevice, s));
    chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dx, B.n, ITERS, A1);
    chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dx, B.n, ITERS, A2);
    CHECK(cudaMemcpyAsync(B.dy, B.hy, bytes, cudaMemcpyHostToDevice, s));
    chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dy, B.n, ITERS, B1);
    chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dy, B.n, ITERS, B2);
    joinKernel<<<B.blocks, BLOCK, 0, s>>>(B.dx, B.dy, B.dz, B.n, ITERS / 2);
    reduceKernel<<<B.blocks, BLOCK, 0, s>>>(B.dz, B.n, B.dPart);
    CHECK(cudaMemcpyAsync(B.hPart, B.dPart, (size_t)B.blocks * sizeof(float),
                          cudaMemcpyDeviceToHost, s));
    CHECK(cudaGetLastError());
}

/* one node, alone, for the duration vector */
static void runNode(const Bufs& B, int node, cudaStream_t s)
{
    size_t bytes = (size_t)B.n * sizeof(float);
    switch (node) {
        case 0: CHECK(cudaMemcpyAsync(B.dx, B.hx, bytes, cudaMemcpyHostToDevice, s)); break;
        case 1: chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dx, B.n, ITERS, A1); break;
        case 2: chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dx, B.n, ITERS, A2); break;
        case 3: CHECK(cudaMemcpyAsync(B.dy, B.hy, bytes, cudaMemcpyHostToDevice, s)); break;
        case 4: chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dy, B.n, ITERS, B1); break;
        case 5: chainStep<<<B.blocks, BLOCK, 0, s>>>(B.dy, B.n, ITERS, B2); break;
        case 6: joinKernel<<<B.blocks, BLOCK, 0, s>>>(B.dx, B.dy, B.dz, B.n, ITERS / 2); break;
        case 7: reduceKernel<<<B.blocks, BLOCK, 0, s>>>(B.dz, B.n, B.dPart); break;
        default: CHECK(cudaMemcpyAsync(B.hPart, B.dPart, (size_t)B.blocks * sizeof(float),
                                       cudaMemcpyDeviceToHost, s)); break;
    }
    CHECK(cudaGetLastError());
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

/* host reference for one element, in the same order the GPU uses */
static double refElement(float x0, float y0)
{
    float x = x0, y = y0;
    for (int k = 0; k < ITERS; ++k) x = fmaf(x, A1, 1.0e-7f);
    for (int k = 0; k < ITERS; ++k) x = fmaf(x, A2, 1.0e-7f);
    for (int k = 0; k < ITERS; ++k) y = fmaf(y, B1, 1.0e-7f);
    for (int k = 0; k < ITERS; ++k) y = fmaf(y, B2, 1.0e-7f);
    float v = x + y;
    for (int k = 0; k < ITERS / 2; ++k) v = fmaf(v, 1.0000005f, 1.0e-7f);
    return (double)v;
}

struct Result { double serialMs, dagMs, bound, sp; int bucket; double d[9]; };

/* time every configuration for one size inside ONE rotated sweep:
 *   cfg 0..8  node i alone      cfg 9  serial      cfg 10  the reader's DAG
 * Every ratio scored below is computed between samples taken inside this
 * one sweep, which is what spec 12.5b says removes the need for a separate
 * operating-point guard.  SWEEPS == NCFG so each configuration leads once. */
static Result measure(const Bufs& B, cudaStream_t* s, int nStreams,
                      cudaEvent_t* ev, int nEvents, cudaEvent_t a, cudaEvent_t b)
{
    const int NCFG = 11;
    float best[11];
    for (int i = 0; i < NCFG; ++i) best[i] = 1e30f;
    for (int sweep = 0; sweep < NCFG; ++sweep) {
        for (int q = 0; q < NCFG; ++q) {
            int c = (q + sweep) % NCFG;
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(a, 0));
            if (c < 9)       runNode(B, c, s[0]);
            else if (c == 9) serialDag(B, s[0]);
            else           { issueDag(B, s, nStreams, ev, nEvents);
                             waitForDag(s, nStreams); }
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(b, 0));
            CHECK(cudaEventSynchronize(b));
            float m; CHECK(cudaEventElapsedTime(&m, a, b));
            if (m < best[c]) best[c] = m;
        }
    }
    Result r;
    for (int i = 0; i < 9; ++i) r.d[i] = best[i];
    r.serialMs = best[9];
    r.dagMs    = best[10];
    r.bound    = criticalPath(r.d);
    r.bound    = (r.bound > 0.0) ? (r.serialMs / r.bound) : 0.0;
    r.sp       = (r.dagMs > 0.0) ? (r.serialMs / r.dagMs) : 0.0;
    r.bucket   = (r.sp < 1.15) ? 1 : (r.sp < 1.40) ? 2 : (r.sp < 2.00) ? 3 : 4;
    return r;
}

/* The reader's DAG must produce, element for element, what the serial
 * version produces -- same kernels, same per-element order, so the two
 * partial arrays must agree.  A missing or mis-timed cross-stream edge
 * shows up here as a difference. */
static bool checkAgainstSerial(const Bufs& B, cudaStream_t* st, int nStreams,
                               cudaEvent_t* ev, int nEvents)
{
    float* ref = (float*)malloc((size_t)B.blocks * sizeof(float));
    if (!ref) return false;

    for (int i = 0; i < B.blocks; ++i) B.hPart[i] = -1.0f;
    serialDag(B, st[0]);
    CHECK(cudaStreamSynchronize(st[0]));
    for (int i = 0; i < B.blocks; ++i) ref[i] = B.hPart[i];

    for (int i = 0; i < B.blocks; ++i) B.hPart[i] = -1.0f;
    issueDag(B, st, nStreams, ev, nEvents);
    waitForDag(st, nStreams);
    CHECK(cudaDeviceSynchronize());

    int bad = 0, untouched = 0;
    double worst = 0.0, sum = 0.0, refSum = 0.0;
    for (int i = 0; i < B.blocks; ++i) {
        if (B.hPart[i] == -1.0f) ++untouched;
        double e = fabs((double)B.hPart[i] - (double)ref[i]);
        if (e > worst) worst = e;
        if (!(e <= 1e-6 * fmax(1.0, fabs((double)ref[i])))) ++bad;
        sum    += (double)B.hPart[i];
        refSum += (double)ref[i];
    }
    free(ref);
    printf("    %d of %d partials differ from the serial result (worst %.3e), "
           "%d never written\n", bad, B.blocks, worst, untouched);
    printf("    total: your DAG %.6e, serial %.6e\n", sum, refSum);
    return (bad == 0) && (untouched == 0);
}

/* An independent anchor: recompute one block's worth of elements on the
 * host, in the same order, and compare against that block's partial.  This
 * is what rules out "both versions are wrong in the same way". */
static bool checkAnchor(const Bufs& B)
{
    double want = 0.0;
    for (int i = 0; i < BLOCK && i < B.n; ++i) want += refElement(B.hx[i], B.hy[i]);
    double got = (double)B.hPart[0];
    bool ok = fabs(got - want) <= 1e-4 * fmax(1.0, fabs(want));
    printf("    host anchor on block 0: GPU %.6f, host %.6f, %s\n",
           got, want, ok ? "ok" : "MISMATCH");
    return ok;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("device        : %s (%d SMs), asyncEngineCount = %d\n",
           prop.name, prop.multiProcessorCount, prop.asyncEngineCount);
    printf("DAG           : 9 nodes, two chains of 3 joining into a reduce\n");
    printf("SMALL         : %d blocks x %d threads = %.2f blocks/SM\n",
           SMALL_BLOCKS, BLOCK, (double)SMALL_BLOCKS / prop.multiProcessorCount);
    printf("LARGE         : %d blocks x %d threads = %.2f blocks/SM\n\n",
           LARGE_BLOCKS, BLOCK, (double)LARGE_BLOCKS / prop.multiProcessorCount);

    int ms = peek(&MIN_STREAMS), me = peek(&MIN_EVENTS);
    bool haveDesign = (ms >= 1 && ms <= 8 && me >= 0 && me <= 8);

    const int nMax = LARGE_BLOCKS * BLOCK;
    Bufs S, L;
    CHECK(cudaMallocHost(&S.hx, (size_t)nMax * 4));
    CHECK(cudaMallocHost(&S.hy, (size_t)nMax * 4));
    CHECK(cudaMallocHost(&S.hPart, (size_t)LARGE_BLOCKS * 4));
    CHECK(cudaMalloc(&S.dx, (size_t)nMax * 4));
    CHECK(cudaMalloc(&S.dy, (size_t)nMax * 4));
    CHECK(cudaMalloc(&S.dz, (size_t)nMax * 4));
    CHECK(cudaMalloc(&S.dPart, (size_t)LARGE_BLOCKS * 4));
    L = S;
    S.n = SMALL_BLOCKS * BLOCK; S.blocks = SMALL_BLOCKS;
    L.n = LARGE_BLOCKS * BLOCK; L.blocks = LARGE_BLOCKS;
    for (int i = 0; i < nMax; ++i) {
        S.hx[i] = 0.25f + (float)(i % 1021) * 1.0e-5f;
        S.hy[i] = 0.50f + (float)(i % 997)  * 1.0e-5f;
    }

    const size_t nWarm = (size_t)64 * 1024 * 1024;
    float *wA, *wB, *sink;
    CHECK(cudaMalloc(&wA, nWarm * 4));
    CHECK(cudaMalloc(&wB, nWarm * 4));
    CHECK(cudaMalloc(&sink, 4));
    CHECK(cudaMemset(wA, 0, nWarm * 4));

    cudaStream_t st[8];
    cudaEvent_t  ev[8];
    for (int i = 0; i < 8; ++i) {
        CHECK(cudaStreamCreate(&st[i]));
        CHECK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming));
    }
    cudaEvent_t ta, tb;
    CHECK(cudaEventCreate(&ta));
    CHECK(cudaEventCreate(&tb));

    printf("warming up (1500 ms stream + 500 ms compute) ...\n");
    warmUp(wA, wB, nWarm, sink);
    printf("done.\n\n");

    if (!haveDesign) {
        printf("Set TODO 1 (MIN_STREAMS, MIN_EVENTS) first.\n");
        goto cleanup;
    }
    {
        int score = 0;
        const char* nodeName[9] = {"H2D x", "preA1", "preA2", "H2D y",
                                   "preB1", "preB2", "join ", "reduce", "D2H p"};

        printf("=== TODO 1: the shape of the graph =============================\n");
        {
            uint64_t h = fnv1a(fnv1a(1469598103934665603ULL, ms), me);
            bool ok = (h == 0xe6273486e4d7b1a0ULL);
            printf("  you claim %d stream(s) and %d event(s) : %s\n", ms, me,
                   ok ? "minimal (+2)" : "not the minimum (+0)");
            if (ok) score += 2;
        }
        printf("\n=== TODO 2: criticalPath() reference check =====================\n");
        {
            const double P[4][9] = {
                {1, 2, 3, 4, 5, 6, 7, 8, 9},
                {5, 5, 5, 1, 1, 1, 2, 2, 2},
                {0.5, 0.25, 0.125, 0.5, 0.25, 0.125, 1.0, 0.0, 0.75},
                {3.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.5}
            };
            uint64_t h = 1469598103934665603ULL;
            for (int i = 0; i < 4; ++i) {
                double v = criticalPath(P[i]);
                printf("  probe %d -> %.4f\n", i, v);
                h = fnv1a(h, (long long)llround(v * 1.0e4));
            }
            bool ok = (h == 0x26904b84a740f459ULL);
            printf("  %s\n", ok ? "matches the reference (+2)" : "does NOT match (+0)");
            if (ok) score += 2;
        }

        printf("\n=== timing =====================================================\n");
        Result rs = measure(S, st, ms, ev, me, ta, tb);
        Result rl = measure(L, st, ms, ev, me, ta, tb);

        for (int pass = 0; pass < 2; ++pass) {
            const Result& r = pass ? rl : rs;
            printf("  --- %s (%d blocks) ---\n", pass ? "LARGE" : "SMALL",
                   pass ? LARGE_BLOCKS : SMALL_BLOCKS);
            printf("    node durations (ms):");
            for (int i = 0; i < 9; ++i) printf(" %s=%.3f", nodeName[i], r.d[i]);
            printf("\n");
            printf("    serial %8.3f ms | your DAG %8.3f ms | %.3fx\n",
                   r.serialMs, r.dagMs, r.sp);
            printf("    your critical-path bound %.3fx -> you reached %.0f%% of it\n",
                   r.bound, (r.bound > 0.0) ? 100.0 * r.sp / r.bound : 0.0);
        }

        printf("\n=== correctness ================================================\n");
        bool corr = true;
        for (int pass = 0; pass < 2; ++pass) {
            const Bufs& B = pass ? L : S;
            printf("  --- %s ---\n", pass ? "LARGE" : "SMALL");
            corr = checkAgainstSerial(B, st, ms, ev, me) && corr;
        }
        printf("  --- host anchor (SMALL, block 0) ---\n");
        corr = checkAnchor(S) && corr;
        printf("  %s\n", corr ? "ok (+2)" : "FAILED (+0)");
        if (corr) score += 2;

        printf("\n=== performance ================================================\n");
        {
            bool ok = corr && (rs.bound > 1.0) && (rs.sp >= 0.70 * rs.bound);
            printf("  SMALL: %.3fx against a bound of %.3fx : %s\n", rs.sp, rs.bound,
                   ok ? "ok (+2)" : "FAILED -- need at least 70 percent of it (+0)");
            if (ok) score += 2;
        }

        printf("\n=== predictions ================================================\n");
        {
            bool a = (peek(&PRED_SMALL) == rs.bucket);
            bool b = (peek(&PRED_LARGE) == rl.bucket);
            printf("  SMALL: you said %d, measured bucket %d : %s\n",
                   peek(&PRED_SMALL), rs.bucket, a ? "ok (+1)" : "wrong (+0)");
            printf("  LARGE: you said %d, measured bucket %d : %s\n",
                   peek(&PRED_LARGE), rl.bucket, b ? "ok (+1)" : "wrong (+0)");
            if (a) score += 1;
            if (b) score += 1;
        }

        printf("\nSCORE: %d/10\n", score);
        printf("OVERALL: %s\n", score == 10 ? "PASS" : "FAIL");
    }

cleanup:
    for (int i = 0; i < 8; ++i) {
        CHECK(cudaStreamDestroy(st[i]));
        CHECK(cudaEventDestroy(ev[i]));
    }
    CHECK(cudaEventDestroy(ta)); CHECK(cudaEventDestroy(tb));
    CHECK(cudaFree(wA)); CHECK(cudaFree(wB)); CHECK(cudaFree(sink));
    CHECK(cudaFree(S.dx)); CHECK(cudaFree(S.dy)); CHECK(cudaFree(S.dz));
    CHECK(cudaFree(S.dPart));
    CHECK(cudaFreeHost(S.hx)); CHECK(cudaFreeHost(S.hy)); CHECK(cudaFreeHost(S.hPart));
    CHECK(cudaDeviceReset());
    return 0;
}
