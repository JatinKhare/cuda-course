/* =====================================================================
 * Module 24 / Example 02 — The copy/compute overlap pipeline
 *
 * GOAL
 *   A. Isolate pinned vs pageable: does cudaMemcpyAsync overlap with a
 *      kernel at all?  (Two streams, one copy, one kernel, nothing else.)
 *   B. Derive the pipeline bound from the phase times and this GPU's
 *      asyncEngineCount, then measure three issue orders against it:
 *        order 1  per chunk: H2D(k), K(k), D2H(k)        -- the obvious one
 *        order 2  per chunk: H2D(k), K(k), D2H(k-1)      -- deferred D2H
 *        order 3  two phases: all H2D+K, then all D2H
 *   C. Sweep the chunk count and show the ramp/drain cost and the
 *      per-chunk launch overhead at the two ends.
 *   D. A three-node dependency graph expressed with events
 *      (cudaEventRecord + cudaStreamWaitEvent).
 *
 * BUILD
 *   nvcc -arch=sm_89 -O3 -std=c++17 -o example02.exe example02.cu
 * RUN
 *   example02.exe
 *
 * Module 25 covers events properly (timing and cross-stream dependencies);
 * they appear here because a DAG cannot be expressed without them.
 * Module 26 owns pinned memory; part A measures why this module needs it.
 * Module 28 owns CUDA graphs, which is what you reach for when the host-side
 * issue cost in part C starts to dominate.
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

/* ---- payload ----------------------------------------------------- */
__global__ void condition(const float* __restrict__ in, float* __restrict__ out,
                          int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    for (int k = 0; k < iters; ++k) v = fmaf(v, 1.000001f, 1.0e-7f);
    out[i] = v;
}
__global__ void scaleKernel(float* p, int n, float a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] *= a;
}
__global__ void addKernel(const float* __restrict__ x, const float* __restrict__ y,
                          float* __restrict__ z, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
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

/* ---- globals used by the pipeline variants ------------------------ */
static int           g_n, g_iters, g_nStreams;
static float        *g_hPinIn, *g_hPinOut, *g_hPagIn, *g_hPagOut, *g_dIn, *g_dOut;
static cudaStream_t  g_s[16];

static void chunkOf(int k, int n, int nChunks, int* off, int* len)
{
    int c = (n + nChunks - 1) / nChunks;
    *off = k * c;
    *len = (*off + c <= n) ? c : (n - *off);
    if (*len < 0) *len = 0;
}

enum Order { ORDER_RR = 0, ORDER_DEFER = 1, ORDER_TWOPHASE = 2 };

static void pipeline(int nChunks, Order order, bool pinned)
{
    float* hIn  = pinned ? g_hPinIn  : g_hPagIn;
    float* hOut = pinned ? g_hPinOut : g_hPagOut;

    if (order == ORDER_TWOPHASE) {
        for (int k = 0; k < nChunks; ++k) {
            int o, l; chunkOf(k, g_n, nChunks, &o, &l); if (l <= 0) break;
            cudaStream_t st = g_s[k % g_nStreams];
            CHECK(cudaMemcpyAsync(g_dIn + o, hIn + o, (size_t)l * 4, cudaMemcpyHostToDevice, st));
            condition<<<(l + 255) / 256, 256, 0, st>>>(g_dIn + o, g_dOut + o, l, g_iters);
            CHECK(cudaGetLastError());
        }
        for (int k = 0; k < nChunks; ++k) {
            int o, l; chunkOf(k, g_n, nChunks, &o, &l); if (l <= 0) break;
            CHECK(cudaMemcpyAsync(hOut + o, g_dOut + o, (size_t)l * 4,
                                  cudaMemcpyDeviceToHost, g_s[k % g_nStreams]));
        }
        return;
    }

    for (int k = 0; k < nChunks; ++k) {
        int o, l; chunkOf(k, g_n, nChunks, &o, &l); if (l <= 0) break;
        cudaStream_t st = g_s[k % g_nStreams];
        CHECK(cudaMemcpyAsync(g_dIn + o, hIn + o, (size_t)l * 4, cudaMemcpyHostToDevice, st));
        condition<<<(l + 255) / 256, 256, 0, st>>>(g_dIn + o, g_dOut + o, l, g_iters);
        CHECK(cudaGetLastError());
        if (order == ORDER_RR) {
            CHECK(cudaMemcpyAsync(hOut + o, g_dOut + o, (size_t)l * 4,
                                  cudaMemcpyDeviceToHost, st));
        } else if (k >= 1) {
            int o2, l2; chunkOf(k - 1, g_n, nChunks, &o2, &l2);
            if (l2 > 0)
                CHECK(cudaMemcpyAsync(hOut + o2, g_dOut + o2, (size_t)l2 * 4,
                                      cudaMemcpyDeviceToHost, g_s[(k - 1) % g_nStreams]));
        }
    }
    if (order == ORDER_DEFER) {
        int o2, l2; chunkOf(nChunks - 1, g_n, nChunks, &o2, &l2);
        if (l2 > 0)
            CHECK(cudaMemcpyAsync(hOut + o2, g_dOut + o2, (size_t)l2 * 4,
                                  cudaMemcpyDeviceToHost, g_s[(nChunks - 1) % g_nStreams]));
    }
}

static void serialRun(bool pinned)
{
    float* hIn  = pinned ? g_hPinIn  : g_hPagIn;
    float* hOut = pinned ? g_hPinOut : g_hPagOut;
    CHECK(cudaMemcpy(g_dIn, hIn, (size_t)g_n * 4, cudaMemcpyHostToDevice));
    condition<<<(g_n + 255) / 256, 256>>>(g_dIn, g_dOut, g_n, g_iters);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(hOut, g_dOut, (size_t)g_n * 4, cudaMemcpyDeviceToHost));
}

static float timeRegion(void (*body)(void*), void* arg, cudaEvent_t a, cudaEvent_t b)
{
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a, 0));
    body(arg);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(b, 0));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

struct Cfg { int nChunks; Order order; bool pinned; bool serial; };
static void runCfg(void* v)
{
    Cfg* c = (Cfg*)v;
    if (c->serial) serialRun(c->pinned); else pipeline(c->nChunks, c->order, c->pinned);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    const int    N     = 12582917;
    const size_t bytes = (size_t)N * sizeof(float);
    g_n        = N;
    g_iters    = 3800;
    g_nStreams = 4;

    CHECK(cudaMallocHost(&g_hPinIn,  bytes));
    CHECK(cudaMallocHost(&g_hPinOut, bytes));
    g_hPagIn  = (float*)malloc(bytes);
    g_hPagOut = (float*)malloc(bytes);
    CHECK(cudaMalloc(&g_dIn,  bytes));
    CHECK(cudaMalloc(&g_dOut, bytes));
    if (!g_hPagIn || !g_hPagOut) { printf("host alloc failed\n"); return 1; }
    for (int i = 0; i < N; ++i) g_hPinIn[i] = g_hPagIn[i] = 0.5f + (float)(i % 1021) * 1.0e-4f;

    const size_t nWarm = (size_t)64 * 1024 * 1024;
    float *wA, *wB, *sink;
    CHECK(cudaMalloc(&wA, nWarm * 4));
    CHECK(cudaMalloc(&wB, nWarm * 4));
    CHECK(cudaMalloc(&sink, 4));
    CHECK(cudaMemset(wA, 0, nWarm * 4));

    for (int i = 0; i < g_nStreams; ++i) CHECK(cudaStreamCreate(&g_s[i]));
    cudaEvent_t evA, evB;
    CHECK(cudaEventCreate(&evA));
    CHECK(cudaEventCreate(&evB));

    printf("device %s, asyncEngineCount = %d, %d SMs\n",
           prop.name, prop.asyncEngineCount, prop.multiProcessorCount);
    printf("N = %d floats = %.1f MB, %d FFMA/element\n\n", N, bytes / 1048576.0, g_iters);
    printf("warming up (1500 ms stream + 500 ms compute) ...\n");
    warmUp(wA, wB, nWarm, sink);
    printf("done.\n\n");

    /* =========== A. does cudaMemcpyAsync overlap at all? =========== */
    printf("=== A. Pinned vs pageable, isolated ============================\n");
    printf("One copy in stream 0, one independent kernel in stream 1, nothing else.\n");
    printf("All nine configurations are timed back to back in one rotated sweep\n");
    printf("(spec 12.1/12.9), so the kernel's own duration cannot drift between\n");
    printf("the 'alone' and the 'both' column.\n");
    printf("overlap = (copy_alone + kernel_alone) / (both).  1.00 = no overlap.\n\n");
    {
        const int kblocks = (N + 255) / 256;
        /* cfg 0        : kernel alone
         * cfg 1..4     : copy alone      (H2Dpin, H2Dpag, D2Hpin, D2Hpag)
         * cfg 5..8     : copy + kernel   (same order)                        */
        const int NA = 9;
        float bestA[NA];
        for (int i = 0; i < NA; ++i) bestA[i] = 1e30f;
        float* hbuf[4] = {g_hPinIn, g_hPagIn, g_hPinOut, g_hPagOut};
        bool   isH2D[4] = {true, true, false, false};

        for (int sweep = 0; sweep < NA; ++sweep) {
            for (int q = 0; q < NA; ++q) {
                int c = (q + sweep) % NA;
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventRecord(evA, 0));
                if (c == 0) {
                    condition<<<kblocks, 256, 0, g_s[1]>>>(g_dIn, g_dOut, N, g_iters);
                    CHECK(cudaGetLastError());
                } else {
                    int j = (c - 1) % 4;
                    if (isH2D[j])
                        CHECK(cudaMemcpyAsync(g_dIn, hbuf[j], bytes, cudaMemcpyHostToDevice, g_s[0]));
                    else
                        CHECK(cudaMemcpyAsync(hbuf[j], g_dIn, bytes, cudaMemcpyDeviceToHost, g_s[0]));
                    if (c >= 5) {
                        condition<<<kblocks, 256, 0, g_s[1]>>>(g_dIn, g_dOut, N, g_iters);
                        CHECK(cudaGetLastError());
                    }
                }
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventRecord(evB, 0));
                CHECK(cudaEventSynchronize(evB));
                float m; CHECK(cudaEventElapsedTime(&m, evA, evB));
                if (m < bestA[c]) bestA[c] = m;
            }
        }
        float k = bestA[0];
        printf("  kernel alone: %.3f ms\n\n", k);
        printf("  %-16s %9s %8s %9s %9s %9s\n",
               "case", "copy ms", "GB/s", "both ms", "overlap", "ideal");
        const char* nm[4] = {"H2D pinned", "H2D pageable", "D2H pinned", "D2H pageable"};
        for (int j = 0; j < 4; ++j) {
            float c = bestA[1 + j], b = bestA[5 + j];
            printf("  %-16s %9.3f %8.1f %9.3f %8.3fx %8.3fx\n",
                   nm[j], c, bytes / c / 1e6, b, (c + k) / b, (c + k) / fmaxf(c, k));
        }
        double rawRatio = 0.5 * ((double)bestA[2] / bestA[1] + (double)bestA[4] / bestA[3]);
        printf("\n  Pinned memory overlaps; pageable memory does not.  The reason is not\n");
        printf("  bandwidth: pinned is only %.2fx faster raw here (Module 4 measured\n", rawRatio);
        printf("  1.04-1.15x on this Windows/WDDM driver).  THE OVERLAP IS THE PAYOFF,\n");
        printf("  NOT THE RAW COPY SPEED.  A pageable cudaMemcpyAsync must stage through\n");
        printf("  a driver-owned pinned bounce buffer, and the staging is done by the\n");
        printf("  CALLING HOST THREAD before the call returns: the call is asynchronous\n");
        printf("  in name only, and nothing after it in program order can start early.\n");
        printf("  Module 26 owns the mechanism and the page-pinning cost.\n\n");
    }

    /* =========== B. the bound, and three issue orders ============== */
    printf("=== B. The pipeline bound and three issue orders ===============\n");
    float h2d = 0, ker = 0, d2h = 0;
    {
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(evA, 0));
        CHECK(cudaMemcpy(g_dIn, g_hPinIn, bytes, cudaMemcpyHostToDevice));
        CHECK(cudaEventRecord(evB, 0)); CHECK(cudaEventSynchronize(evB));
        CHECK(cudaEventElapsedTime(&h2d, evA, evB));
        CHECK(cudaEventRecord(evA, 0));
        condition<<<(N + 255) / 256, 256>>>(g_dIn, g_dOut, N, g_iters);
        CHECK(cudaEventRecord(evB, 0)); CHECK(cudaEventSynchronize(evB));
        CHECK(cudaEventElapsedTime(&ker, evA, evB));
        CHECK(cudaEventRecord(evA, 0));
        CHECK(cudaMemcpy(g_hPinOut, g_dOut, bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaEventRecord(evB, 0)); CHECK(cudaEventSynchronize(evB));
        CHECK(cudaEventElapsedTime(&d2h, evA, evB));
        CHECK(cudaGetLastError());
    }
    double copyBusy  = (prop.asyncEngineCount >= 2) ? fmax(h2d, d2h) : (h2d + d2h);
    double bound2way = (h2d + ker + d2h) / fmax(copyBusy, (double)ker);
    double bound3way = (h2d + ker + d2h) / fmax(fmax((double)h2d, (double)d2h), (double)ker);
    printf("  H2D %.3f ms | kernel %.3f ms | D2H %.3f ms | serial sum %.3f ms\n",
           h2d, ker, d2h, h2d + ker + d2h);
    printf("  copy engines = %d  ->  copy engine is busy %.3f ms total\n",
           prop.asyncEngineCount, copyBusy);
    printf("  achievable bound  = (H+K+D)/max(copyBusy, K) = %.3fx\n", bound2way);
    printf("  if there were 2+ copy engines it would be  = %.3fx  (NOT available here)\n\n",
           bound3way);

    const int NCHUNKS = 16;
    const int NCFG = 4;
    const char* name[NCFG] = {
        "serial (blocking copies, one stream)",
        "order 1: per chunk H2D,K,D2H",
        "order 2: per chunk H2D,K,D2H(k-1)",
        "order 3: two phases, all H2D+K then all D2H"
    };
    Cfg cfgs[NCFG] = {
        {NCHUNKS, ORDER_RR,       true, true },
        {NCHUNKS, ORDER_RR,       true, false},
        {NCHUNKS, ORDER_DEFER,    true, false},
        {NCHUNKS, ORDER_TWOPHASE, true, false}
    };
    float best[NCFG];
    for (int i = 0; i < NCFG; ++i) best[i] = 1e30f;
    for (int sweep = 0; sweep < NCFG; ++sweep)
        for (int q = 0; q < NCFG; ++q) {
            int c = (q + sweep) % NCFG;
            float m = timeRegion(runCfg, &cfgs[c], evA, evB);
            if (m < best[c]) best[c] = m;
        }
    printf("  %-45s %9s %8s %9s\n", "configuration", "ms", "x serial", "% of bound");
    for (int i = 0; i < NCFG; ++i) {
        if (i == 0) printf("  %-45s %9.3f %8.2f %9s\n", name[i], best[i], 1.0, "--");
        else        printf("  %-45s %9.3f %8.2f %8.0f%%\n", name[i], best[i],
                           best[0] / best[i], 100.0 * (best[0] / best[i]) / bound2way);
    }
    printf("\n  Order 1 is the obvious code and it does not overlap at all.\n");
    printf("  With ONE copy engine the engine takes transfers in issue order.\n");
    printf("  D2H(k) is issued immediately after K(k), so it sits at the head of\n");
    printf("  the engine's queue waiting for a kernel, and H2D(k+1) -- which is\n");
    printf("  ready to run -- is stuck behind it.  Head-of-line blocking turns the\n");
    printf("  whole pipeline back into H2D, K, D2H, H2D, K, D2H, ...\n");
    printf("  Orders 2 and 3 both keep a ready transfer at the head of the queue.\n");
    printf("  This defect does not exist on a GPU with two copy engines, which is\n");
    printf("  why it is absent from most tutorials.\n\n");

    /* =========== C. chunk-count sweep ============================== */
    printf("=== C. How many chunks? =======================================\n");
    printf("  Order 2, pinned, 4 streams.  Ramp+drain costs about (H+D)/nChunks;\n");
    printf("  host-side issue cost grows linearly in nChunks.\n\n");
    printf("  %7s %10s %10s %10s\n", "chunks", "ms", "x serial", "% of bound");
    {
        const int NS = 7;
        int chunkList[NS] = {1, 2, 4, 8, 16, 32, 64};
        float bestC[NS];
        for (int i = 0; i < NS; ++i) bestC[i] = 1e30f;
        Cfg cc[NS];
        for (int i = 0; i < NS; ++i) cc[i] = Cfg{chunkList[i], ORDER_DEFER, true, false};
        for (int sweep = 0; sweep < NS; ++sweep)
            for (int q = 0; q < NS; ++q) {
                int c = (q + sweep) % NS;
                float m = timeRegion(runCfg, &cc[c], evA, evB);
                if (m < bestC[c]) bestC[c] = m;
            }
        for (int i = 0; i < NS; ++i)
            printf("  %7d %10.3f %10.2f %9.0f%%\n", chunkList[i], bestC[i],
                   best[0] / bestC[i], 100.0 * (best[0] / bestC[i]) / bound2way);
        printf("\n  nChunks = 1 cannot overlap by construction: there is nothing to\n");
        printf("  overlap with.  The curve is broad -- anything from 4 to 32 is within\n");
        printf("  a few percent -- and turns down again when the per-chunk host cost\n");
        printf("  (3-20 us per enqueue, Module 2) stops being amortized.\n\n");
    }

    /* =========== D. a dependency graph with events ================= */
    printf("=== D. A dependency graph: two chains, a join, a scale =========\n");
    printf("    s0:  H2D(x) -> scale(x)  --+\n");
    printf("    s1:  H2D(y) -> scale(y)  --+--> add(x,y->z) -> D2H(z)\n");
    printf("  The join must wait for BOTH chains.  Same-stream ordering supplies\n");
    printf("  one edge for free; the other needs an event.  Module 25 covers\n");
    printf("  events properly -- here they are only the dependency mechanism.\n\n");
    {
        const int   M = 1 << 22;
        const size_t mb = (size_t)M * 4;
        float *hx, *hy, *hz, *dx, *dy, *dz;
        CHECK(cudaMallocHost(&hx, mb)); CHECK(cudaMallocHost(&hy, mb));
        CHECK(cudaMallocHost(&hz, mb));
        CHECK(cudaMalloc(&dx, mb)); CHECK(cudaMalloc(&dy, mb)); CHECK(cudaMalloc(&dz, mb));
        for (int i = 0; i < M; ++i) { hx[i] = (float)(i % 97); hy[i] = (float)(i % 89); }

        cudaEvent_t depY;
        CHECK(cudaEventCreateWithFlags(&depY, cudaEventDisableTiming));
        int blocks = (M + 255) / 256;

        auto runDag = [&](bool useEvent) {
            CHECK(cudaMemcpyAsync(dx, hx, mb, cudaMemcpyHostToDevice, g_s[0]));
            scaleKernel<<<blocks, 256, 0, g_s[0]>>>(dx, M, 2.0f);
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpyAsync(dy, hy, mb, cudaMemcpyHostToDevice, g_s[1]));
            scaleKernel<<<blocks, 256, 0, g_s[1]>>>(dy, M, 3.0f);
            CHECK(cudaGetLastError());
            if (useEvent) {
                CHECK(cudaEventRecord(depY, g_s[1]));
                CHECK(cudaStreamWaitEvent(g_s[0], depY, 0));
            }
            addKernel<<<blocks, 256, 0, g_s[0]>>>(dx, dy, dz, M);
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpyAsync(hz, dz, mb, cudaMemcpyDeviceToHost, g_s[0]));
            CHECK(cudaStreamSynchronize(g_s[0]));
        };

        /* correct version */
        for (int i = 0; i < M; ++i) hz[i] = -1.0f;
        runDag(true);
        CHECK(cudaDeviceSynchronize());
        int bad = 0;
        for (int i = 0; i < M; i += 1021) {
            float want = 2.0f * (float)(i % 97) + 3.0f * (float)(i % 89);
            if (fabsf(hz[i] - want) > 1e-5f * fmaxf(1.0f, fabsf(want))) ++bad;
        }
        printf("  with cudaStreamWaitEvent : mismatches = %d\n", bad);

        /* the same graph with the event removed: the join may read dy before
         * stream 1's scale has run.  Whether it is actually wrong depends on
         * timing, which is exactly the point. */
        for (int i = 0; i < M; ++i) hz[i] = -1.0f;
        CHECK(cudaMemset(dy, 0, mb));
        runDag(false);
        CHECK(cudaDeviceSynchronize());
        int bad2 = 0;
        for (int i = 0; i < M; i += 1021) {
            float want = 2.0f * (float)(i % 97) + 3.0f * (float)(i % 89);
            if (fabsf(hz[i] - want) > 1e-5f * fmaxf(1.0f, fabsf(want))) ++bad2;
        }
        printf("  without it               : mismatches = %d   <-- a race, not a hang;\n", bad2);
        printf("    its visibility depends on scheduling, which is why you cannot\n");
        printf("    test a missing cross-stream dependency into existence.\n\n");

        CHECK(cudaEventDestroy(depY));
        CHECK(cudaFree(dx)); CHECK(cudaFree(dy)); CHECK(cudaFree(dz));
        CHECK(cudaFreeHost(hx)); CHECK(cudaFreeHost(hy)); CHECK(cudaFreeHost(hz));
    }

    /* =========== validation ======================================== */
    printf("=== Validation ================================================\n");
    for (int i = 0; i < N; ++i) g_hPinOut[i] = -1.0f;
    pipeline(NCHUNKS, ORDER_DEFER, true);
    for (int i = 0; i < g_nStreams; ++i) CHECK(cudaStreamSynchronize(g_s[i]));
    int bad = 0; double worst = 0.0;
    for (int i = 0; i < N; i += 7919) {
        float v = g_hPinIn[i];
        for (int k = 0; k < g_iters; ++k) v = fmaf(v, 1.000001f, 1.0e-7f);
        double e = fabs((double)v - (double)g_hPinOut[i]);
        if (e > 1e-5 * fmax(1.0, fabs((double)v))) ++bad;
        if (e > worst) worst = e;
    }
    printf("  worst abs error %.3e over %d samples, mismatches %d\n",
           worst, (N + 7918) / 7919, bad);

    bool pass = (bad == 0)
             && ((double)(best[0] / best[2]) >= 0.80 * bound2way)   /* deferred order works */
             && (best[0] / best[1] < 1.20f);                        /* naive order does not */
    printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");

    for (int i = 0; i < g_nStreams; ++i) CHECK(cudaStreamDestroy(g_s[i]));
    CHECK(cudaEventDestroy(evA)); CHECK(cudaEventDestroy(evB));
    CHECK(cudaFree(wA)); CHECK(cudaFree(wB)); CHECK(cudaFree(sink));
    CHECK(cudaFree(g_dIn)); CHECK(cudaFree(g_dOut));
    CHECK(cudaFreeHost(g_hPinIn)); CHECK(cudaFreeHost(g_hPinOut));
    free(g_hPagIn); free(g_hPagOut);
    CHECK(cudaDeviceReset());
    return pass ? 0 : 1;
}
