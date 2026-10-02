/* =====================================================================
 * Module 24 / Example 01 — What a stream is, and what the default stream does
 *
 * GOAL
 *   Establish the entire stream model by measurement:
 *     A. The machine's stream-relevant capabilities (asyncEngineCount,
 *        concurrentKernels, the stream priority range).
 *     B. Ordering: operations in one stream are ordered; operations in
 *        different streams are not. Concurrency happens only if there is
 *        hardware left over.
 *     C. The legacy default stream destroys concurrency, and the three
 *        cures (cudaStreamNonBlocking, --default-stream per-thread,
 *        don't use stream 0).
 *     D. Implicit synchronization: cudaMalloc / cudaFree / a blocking
 *        pageable cudaMemcpy serialize streams without being told to.
 *     E. cudaStreamQuery (non-blocking test) and cudaLaunchHostFunc.
 *
 * BUILD (legacy default stream — the default)
 *   nvcc -arch=sm_89 -O3 -std=c++17 -o example01.exe example01.cu
 * BUILD (per-thread default stream — run BOTH and compare part C)
 *   nvcc -arch=sm_89 -O3 -std=c++17 --default-stream per-thread -o example01_pt.exe example01.cu
 * RUN
 *   example01.exe
 *
 * Module 25 covers events properly; this file uses cudaEventRecord only as
 * a timestamp and cudaEventQuery/cudaStreamQuery as a completion test.
 * Module 26 owns pinned memory; cudaMallocHost appears here because
 * cudaMemcpyAsync only overlaps from pinned memory (measured in example02).
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

#define CHECK_KERNEL()                                                         \
    do {                                                                       \
        CHECK(cudaGetLastError());                                             \
        CHECK(cudaDeviceSynchronize());                                        \
    } while (0)

/* ------------------------------------------------------------------ */
/* Payload kernels                                                     */
/* ------------------------------------------------------------------ */

/* Compute-bound, no memory traffic beyond one load and one store, so the
 * duration is a clean function of `iters` and of how much of the machine
 * the launch occupies. */
__global__ void condition(const float* __restrict__ in, float* __restrict__ out,
                          int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    for (int k = 0; k < iters; ++k) v = fmaf(v, 1.000001f, 1.0e-7f);
    out[i] = v;
}

__global__ void touch(float* p)
{
    if (threadIdx.x == 0) p[0] = p[0] + 0.0f;
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
    if (a == 12345.678f) sink[0] = a;   /* never true; defeats dead-code removal */
}

/* ------------------------------------------------------------------ */
/* Spec section 12 rule 4: 1500 ms streaming warm-up, then 500 ms compute. */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/* Part C/D: one pipeline, several deliberate perturbations.           */
/* ------------------------------------------------------------------ */

enum Poison {
    POISON_NONE = 0,       /* the pipeline as it should be written            */
    POISON_DEFAULT_STREAM, /* one tiny kernel launched with no stream argument*/
    POISON_ALLOC_IN_LOOP,  /* cudaMalloc/cudaFree of scratch, per chunk       */
    POISON_BLOCKING_COPY   /* a 4-byte blocking D2H into a pageable host int  */
};

struct Pipe {
    float*        hIn;
    float*        hOut;
    float*        dIn;
    float*        dOut;
    int           n;
    int           nChunks;
    int           nStreams;
    cudaStream_t* s;
    int           iters;
};

static void chunkOf(int k, int n, int nChunks, int* off, int* len)
{
    int c = (n + nChunks - 1) / nChunks;
    *off = k * c;
    *len = (*off + c <= n) ? c : (n - *off);
    if (*len < 0) *len = 0;
}

/* The deferred-D2H issue order: chunk k's device-to-host copy is issued one
 * iteration AFTER its kernel, so the single copy engine never has a transfer
 * at the head of its queue that is waiting on a kernel.  example02 measures
 * why this matters. */
static void runPipeline(const Pipe& p, Poison poison)
{
    for (int k = 0; k < p.nChunks; ++k) {
        int off, len;
        chunkOf(k, p.n, p.nChunks, &off, &len);
        if (len <= 0) break;
        cudaStream_t st = p.s[k % p.nStreams];

        CHECK(cudaMemcpyAsync(p.dIn + off, p.hIn + off, (size_t)len * sizeof(float),
                              cudaMemcpyHostToDevice, st));
        condition<<<(len + 255) / 256, 256, 0, st>>>(p.dIn + off, p.dOut + off, len, p.iters);
        CHECK(cudaGetLastError());

        if (poison == POISON_DEFAULT_STREAM) {
            touch<<<1, 32>>>(p.dOut);           /* <<< no stream argument >>> */
            CHECK(cudaGetLastError());
        } else if (poison == POISON_ALLOC_IN_LOOP) {
            void* scratch = nullptr;
            CHECK(cudaMalloc(&scratch, 1024));
            CHECK(cudaFree(scratch));
        } else if (poison == POISON_BLOCKING_COPY) {
            int probe = 0;
            CHECK(cudaMemcpy(&probe, p.dOut, sizeof(int), cudaMemcpyDeviceToHost));
        }

        if (k >= 1) {
            int o2, l2;
            chunkOf(k - 1, p.n, p.nChunks, &o2, &l2);
            if (l2 > 0)
                CHECK(cudaMemcpyAsync(p.hOut + o2, p.dOut + o2, (size_t)l2 * sizeof(float),
                                      cudaMemcpyDeviceToHost, p.s[(k - 1) % p.nStreams]));
        }
    }
    int o2, l2;
    chunkOf(p.nChunks - 1, p.n, p.nChunks, &o2, &l2);
    if (l2 > 0)
        CHECK(cudaMemcpyAsync(p.hOut + o2, p.dOut + o2, (size_t)l2 * sizeof(float),
                              cudaMemcpyDeviceToHost, p.s[(p.nChunks - 1) % p.nStreams]));
}

static void runSerial(const Pipe& p)
{
    CHECK(cudaMemcpy(p.dIn, p.hIn, (size_t)p.n * sizeof(float), cudaMemcpyHostToDevice));
    condition<<<(p.n + 255) / 256, 256>>>(p.dIn, p.dOut, p.n, p.iters);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(p.hOut, p.dOut, (size_t)p.n * sizeof(float), cudaMemcpyDeviceToHost));
}

/* Time one configuration.  Every timed region is bracketed by events recorded
 * in the legacy default stream, which is exactly the right instrument here:
 * a legacy-default-stream event record is a device-wide barrier, so it
 * measures the whole pipeline including its tail. */
static float timeOnce(const Pipe& p, Poison poison, bool serial,
                      cudaEvent_t a, cudaEvent_t b)
{
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a, 0));
    if (serial) runSerial(p); else runPipeline(p, poison);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(b, 0));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    /* ============================================================== */
    printf("=== A. What this GPU can and cannot overlap ==================\n");
    printf("device                  : %s (cc %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("asyncEngineCount        : %d\n", prop.asyncEngineCount);
    printf("concurrentKernels       : %d\n", prop.concurrentKernels);
    int loPri = 0, hiPri = 0;
    CHECK(cudaDeviceGetStreamPriorityRange(&loPri, &hiPri));
    printf("stream priority range   : least=%d  greatest=%d  (%d levels)\n",
           loPri, hiPri, loPri - hiPri + 1);
    printf("\n");
    printf("asyncEngineCount is the number of DMA ('copy') engines that can run\n");
    printf("a cudaMemcpyAsync concurrently with kernel execution.\n");
    if (prop.asyncEngineCount >= 2) {
        printf("  >= 2  ->  H2D and D2H can run AT THE SAME TIME as each other and\n");
        printf("            as a kernel: genuine three-way overlap is possible.\n");
    } else {
        printf("  == 1  ->  ONE engine serves BOTH directions.  Copies overlap with\n");
        printf("            KERNELS but never with each other.  The best a pipeline\n");
        printf("            can do is max(H2D + D2H, kernel), NOT max(H2D, kernel, D2H).\n");
        printf("            Every bound in this module is computed from that fact.\n");
    }
    printf("\n");

    /* ============================================================== */
    /* Allocations                                                     */
    /* ============================================================== */
    const int    N        = 12582917;                 /* prime-ish: forces a ragged last chunk */
    const size_t bytes    = (size_t)N * sizeof(float);
    const int    ITERS    = 3800;
    const int    NSTREAMS = 4;
    const int    NCHUNKS  = 16;

    float *hIn = nullptr, *hOut = nullptr, *dIn = nullptr, *dOut = nullptr;
    CHECK(cudaMallocHost(&hIn,  bytes));    /* pinned: see example02 for why */
    CHECK(cudaMallocHost(&hOut, bytes));
    CHECK(cudaMalloc(&dIn,  bytes));
    CHECK(cudaMalloc(&dOut, bytes));
    for (int i = 0; i < N; ++i) hIn[i] = 0.5f + (float)(i % 1021) * 1.0e-4f;

    /* warm-up buffers (>= 4x L2 so the streaming warm-up really ramps DRAM) */
    const size_t nWarm = (size_t)64 * 1024 * 1024;   /* 256 MB each */
    float *wA = nullptr, *wB = nullptr, *sink = nullptr;
    CHECK(cudaMalloc(&wA, nWarm * sizeof(float)));
    CHECK(cudaMalloc(&wB, nWarm * sizeof(float)));
    CHECK(cudaMalloc(&sink, sizeof(float)));
    CHECK(cudaMemset(wA, 0, nWarm * sizeof(float)));

    printf("warming up (1500 ms stream + 500 ms compute, spec 12.4) ...\n");
    warmUp(wA, wB, nWarm, sink);
    printf("done.\n\n");

    cudaStream_t s[8];
    for (int i = 0; i < NSTREAMS; ++i) CHECK(cudaStreamCreate(&s[i]));
    cudaEvent_t evA, evB;
    CHECK(cudaEventCreate(&evA));
    CHECK(cudaEventCreate(&evB));

    /* ============================================================== */
    printf("=== B. Ordering is per stream; concurrency needs free hardware ===\n");
    printf("Two INDEPENDENT kernels, one per stream.  The grid of each is swept.\n");
    printf("ratio = (both, two streams) / (one alone).  1.00 = perfect concurrency,\n");
    printf("2.00 = no concurrency at all.\n\n");
    printf("  blocks  blk/SM   one(ms)   two(ms)   ratio   verdict\n");
    {
        const int thr = 128, it = 2000000;
        float* scratch = dOut;   /* reuse; the two kernels write disjoint halves */
        for (int blocks : {1, 10, 20, 40, 80, 240}) {
            int n = blocks * thr;
            float t1 = 1e30f, t2 = 1e30f;
            for (int r = 0; r < 3; ++r) {
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventRecord(evA, 0));
                condition<<<blocks, thr, 0, s[0]>>>(dIn, scratch, n, it);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventRecord(evB, 0));
                CHECK(cudaEventSynchronize(evB));
                float m; CHECK(cudaEventElapsedTime(&m, evA, evB));
                if (m < t1) t1 = m;

                CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventRecord(evA, 0));
                condition<<<blocks, thr, 0, s[0]>>>(dIn, scratch, n, it);
                condition<<<blocks, thr, 0, s[1]>>>(dIn + n, scratch + n, n, it);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventRecord(evB, 0));
                CHECK(cudaEventSynchronize(evB));
                CHECK(cudaEventElapsedTime(&m, evA, evB));
                if (m < t2) t2 = m;
            }
            double perSM = (double)blocks / prop.multiProcessorCount;
            const char* verdict = (t2 / t1 < 1.10) ? "free"
                                : (t2 / t1 < 1.50) ? "partial" : "no room";
            printf("  %6d  %6.2f  %8.3f  %8.3f  %6.3f   %s\n",
                   blocks, perSM, t1, t2, t2 / t1, verdict);
        }
    }
    printf("\n  A kernel that already fills the machine gains nothing from a second\n");
    printf("  stream.  Module 1's wave arithmetic is the predictor: at 40 blocks\n");
    printf("  this kernel is one block per SM and a second copy is free; at 240\n");
    printf("  blocks it is 6 blocks per SM and the two runs simply queue up.\n\n");

    /* ============================================================== */
    printf("=== C/D. The default stream, and other silent concurrency killers ===\n");
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
    printf("THIS BINARY WAS BUILT WITH --default-stream per-thread.\n");
#else
    printf("THIS BINARY USES THE LEGACY DEFAULT STREAM (the nvcc default).\n");
#endif
    printf("Pipeline: N = %d floats (%.1f MB), %d chunks over %d streams, %d FFMA/elem.\n\n",
           N, bytes / 1048576.0, NCHUNKS, NSTREAMS, ITERS);

    Pipe p{hIn, hOut, dIn, dOut, N, NCHUNKS, NSTREAMS, s, ITERS};

    /* Phase times, measured once, so the reader can compute the bound. */
    float h2d = 0, ker = 0, d2h = 0;
    {
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(evA, 0));
        CHECK(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));
        CHECK(cudaEventRecord(evB, 0));
        CHECK(cudaEventSynchronize(evB));
        CHECK(cudaEventElapsedTime(&h2d, evA, evB));

        CHECK(cudaEventRecord(evA, 0));
        condition<<<(N + 255) / 256, 256>>>(dIn, dOut, N, ITERS);
        CHECK(cudaEventRecord(evB, 0));
        CHECK(cudaEventSynchronize(evB));
        CHECK(cudaEventElapsedTime(&ker, evA, evB));

        CHECK(cudaEventRecord(evA, 0));
        CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaEventRecord(evB, 0));
        CHECK(cudaEventSynchronize(evB));
        CHECK(cudaEventElapsedTime(&d2h, evA, evB));
        CHECK_KERNEL();
    }
    double copyTotal = (prop.asyncEngineCount >= 2) ? fmax(h2d, d2h) : (h2d + d2h);
    double bound     = (h2d + ker + d2h) / fmax(copyTotal, (double)ker);
    printf("phases: H2D %.3f ms (%.1f GB/s) | kernel %.3f ms | D2H %.3f ms (%.1f GB/s)\n",
           h2d, bytes / h2d / 1e6, ker, d2h, bytes / d2h / 1e6);
    printf("with %d copy engine(s) the pipeline bound is (H+K+D)/max(%s, K) = %.3fx\n\n",
           prop.asyncEngineCount, (prop.asyncEngineCount >= 2) ? "max(H,D)" : "H+D", bound);

    /* Rotated sweep over the five configurations (spec 12.1, 12.9). */
    const int NCFG = 5;
    const char* cfgName[NCFG] = {
        "serial (one stream, blocking copies)",
        "pipeline, clean",
        "pipeline + one kernel on the DEFAULT stream",
        "pipeline + cudaMalloc/cudaFree inside the loop",
        "pipeline + a 4-byte blocking cudaMemcpy inside the loop"
    };
    float best[NCFG];
    for (int i = 0; i < NCFG; ++i) best[i] = 1e30f;
    const int NSWEEP = 5;                       /* SWEEPS >= NCFG */
    for (int sweep = 0; sweep < NSWEEP; ++sweep) {
        for (int q = 0; q < NCFG; ++q) {
            int c = (q + sweep) % NCFG;
            float ms;
            switch (c) {
                case 0:  ms = timeOnce(p, POISON_NONE, true,  evA, evB); break;
                case 1:  ms = timeOnce(p, POISON_NONE, false, evA, evB); break;
                case 2:  ms = timeOnce(p, POISON_DEFAULT_STREAM, false, evA, evB); break;
                case 3:  ms = timeOnce(p, POISON_ALLOC_IN_LOOP,  false, evA, evB); break;
                default: ms = timeOnce(p, POISON_BLOCKING_COPY,  false, evA, evB); break;
            }
            if (ms < best[c]) best[c] = ms;
        }
    }
    printf("  %-56s %9s %8s\n", "configuration", "ms", "x serial");
    for (int i = 0; i < NCFG; ++i)
        printf("  %-56s %9.3f %8.2f\n", cfgName[i], best[i], best[0] / best[i]);
    printf("\n");
    printf("  The clean pipeline reaches %.0f%% of its %.2fx bound.\n",
           100.0 * (best[0] / best[1]) / bound, bound);
    printf("  One kernel issued with no stream argument costs %.2fx of the win.\n",
           (best[0] / best[1]) / (best[0] / best[2]));
    printf("  None of the three broken versions is WRONG.  All three produce the\n");
    printf("  correct answer.  They are merely serial, and nothing in the source\n");
    printf("  says so.\n\n");

    /* The three cures, measured. */
    printf("Cure 1: create the streams with cudaStreamNonBlocking.\n");
    {
        cudaStream_t nb[8];
        for (int i = 0; i < NSTREAMS; ++i)
            CHECK(cudaStreamCreateWithFlags(&nb[i], cudaStreamNonBlocking));
        Pipe pnb = p;
        pnb.s = nb;
        float clean = 1e30f, poisoned = 1e30f;
        for (int r = 0; r < 3; ++r) {
            float a1 = timeOnce(pnb, POISON_NONE, false, evA, evB);
            float a2 = timeOnce(pnb, POISON_DEFAULT_STREAM, false, evA, evB);
            if (a1 < clean) clean = a1;
            if (a2 < poisoned) poisoned = a2;
        }
        printf("   clean %.3f ms (%.2fx)   with the default-stream kernel %.3f ms (%.2fx)\n",
               clean, best[0] / clean, poisoned, best[0] / poisoned);
        printf("   A cudaStreamNonBlocking stream does not synchronize with the\n");
        printf("   legacy default stream, so the poison is harmless.\n");
        for (int i = 0; i < NSTREAMS; ++i) CHECK(cudaStreamDestroy(nb[i]));
    }
    printf("\nCure 2: compile with --default-stream per-thread.  Rebuild this file\n");
    printf("   with that flag and compare row 3 of the table above.\n");
    printf("Cure 3: pass an explicit stream to EVERY asynchronous call.  Note that\n");
    printf("   cure 2 does NOT rescue row 4: cudaMalloc is a device-wide implicit\n");
    printf("   synchronization no matter which default stream you selected.\n\n");

    /* ============================================================== */
    printf("=== E. cudaStreamQuery and cudaLaunchHostFunc ================\n");
    {
        CHECK(cudaDeviceSynchronize());
        printf("  query on an idle stream          : %s\n",
               cudaGetErrorName(cudaStreamQuery(s[0])));
        condition<<<(N + 255) / 256, 256, 0, s[0]>>>(dIn, dOut, N, ITERS);
        CHECK(cudaGetLastError());
        printf("  query immediately after a launch : %s\n",
               cudaGetErrorName(cudaStreamQuery(s[0])));
        int spins = 0;
        while (cudaStreamQuery(s[0]) == cudaErrorNotReady) ++spins;
        printf("  host spun %d times before it reported cudaSuccess\n", spins);
        printf("  (cudaStreamQuery does not clear the last-error slot and is not\n");
        printf("   a synchronization point; it is a test, and the host may do\n");
        printf("   useful work between tests.)\n");

        static int callbackHits = 0;
        callbackHits = 0;
        condition<<<(N + 255) / 256, 256, 0, s[1]>>>(dIn, dOut, N, ITERS);
        CHECK(cudaGetLastError());
        CHECK(cudaLaunchHostFunc(s[1],
            [](void* ud) { ++*(int*)ud; }, &callbackHits));
        printf("  callbackHits right after enqueue : %d\n", callbackHits);
        CHECK(cudaStreamSynchronize(s[1]));
        printf("  callbackHits after stream sync   : %d\n", callbackHits);
        printf("  A host function runs on a driver thread when the stream reaches\n");
        printf("  it.  You MUST NOT call any CUDA API inside it -- not even\n");
        printf("  cudaGetLastError.  Doing so deadlocks or returns undefined\n");
        printf("  results, because the callback runs inside the driver's own\n");
        printf("  progress machinery.\n\n");
    }

    /* ============================================================== */
    /* Validation pass (spec: timing first, validation second)         */
    /* ============================================================== */
    printf("=== Validation ================================================\n");
    for (int i = 0; i < N; ++i) hOut[i] = -1.0f;
    runPipeline(p, POISON_NONE);
    for (int i = 0; i < NSTREAMS; ++i) CHECK(cudaStreamSynchronize(s[i]));
    CHECK(cudaDeviceSynchronize());

    int    bad    = 0;
    double worst  = 0.0;
    for (int i = 0; i < N; i += 7919) {
        float v = hIn[i];
        for (int k = 0; k < ITERS; ++k) v = fmaf(v, 1.000001f, 1.0e-7f);
        double e = fabs((double)v - (double)hOut[i]);
        double tol = 1e-5 * fmax(1.0, fabs((double)v));
        if (!(e <= tol)) { ++bad; }
        if (e > worst) worst = e;
    }
    printf("  sampled %d elements, worst abs error %.3e, mismatches %d\n",
           (N + 7918) / 7919, worst, bad);

    bool pass = (bad == 0) && ((double)(best[0] / best[1]) >= 0.85 * bound);
    printf("\n  overlap achieved : %.2fx (bound %.2fx)\n", best[0] / best[1], bound);
    printf("OVERALL: %s\n", pass ? "PASS" : "FAIL");

    for (int i = 0; i < NSTREAMS; ++i) CHECK(cudaStreamDestroy(s[i]));
    CHECK(cudaEventDestroy(evA));
    CHECK(cudaEventDestroy(evB));
    CHECK(cudaFree(wA)); CHECK(cudaFree(wB)); CHECK(cudaFree(sink));
    CHECK(cudaFree(dIn)); CHECK(cudaFree(dOut));
    CHECK(cudaFreeHost(hIn)); CHECK(cudaFreeHost(hOut));
    CHECK(cudaDeviceReset());
    return pass ? 0 : 1;
}
