// =============================================================================
// Module 14 / Exercise 2 — 65,536 bins (design)
//
// THE PROBLEM
//   Histogram 2^26 32-bit keys into 65,536 bins. The bin array alone is
//   262,144 bytes. The largest shared-memory allocation a single block can hold
//   on this GPU is 101,376 bytes, opt-in, and that is for ONE resident block per
//   SM. So the move you have used for the last two hundred lines of this module
//   -- put a private copy of the histogram in shared memory -- is not available.
//
//   `hist_naive` (given) does one global atomic per element. Beat it.
//
//   You get up to 64 MB of device scratch memory (`d_scratch`), any number of
//   kernel launches, and any launch configuration. You must produce the exact
//   histogram.
//
//   The harness runs TWO input distributions and gates on BOTH:
//     flat  keys uniform over all 65,536 bins
//     hot   75% of the keys inside 32 of the 65,536 bins
//   and requires >= 5.00x on `hot` while not regressing below 0.90x on `flat`.
//   Both halves of that gate are load-bearing. The obvious adaptation of the
//   shared-memory technique to this bin count passes neither.
//
// BUILD
//   nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
// RUN
//   .\exercise02.exe
//
//   compute-sanitizer --tool memcheck .\exercise02.exe
//   (expect a benign "Resetting device while there are still other users"
//    warning from cudaDeviceReset(); it is not a memory error)
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            printf("CUDA error %s at %s:%d -> %s\n", #call, __FILE__,          \
                   __LINE__, cudaGetErrorString(_e));                          \
            exit(1);                                                           \
        }                                                                      \
    } while (0)
#define CHECK_KERNEL()                                                         \
    do { CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize()); } while (0)

static const size_t N      = 67108864;    // 2^26 keys x 4 B = 256 MiB
static const int    NBINS  = 65536;       // 262144 B of bins: 2.6x the 99 KB max
static const int    BLK    = 256;
static const size_t SCRATCH_BYTES = 64u << 20;   // 64 MB of device scratch
static const double PEAK   = 432.0;
static const char*  DN[2]  = { "flat", "hot" };

__device__ __host__ __forceinline__
unsigned int binOf(unsigned int frac, int nBins)
{
    return (unsigned int)(((unsigned long long)frac * (unsigned long long)nBins) >> 30);
}

// ==================================================================== TODO 1
// PREDICT BEFORE YOU BUILD. Two numbers: the speedup your finished version will
// have over `hist_naive` on the `flat` input and on the `hot` input. Octave
// scored (within a factor of two).
//
// Work them out separately. `hist_naive` on `flat` and `hist_naive` on `hot`
// execute exactly the same number of atomic instructions on exactly the same
// 65,536 addresses; only the distribution over those addresses differs. Decide
// what that is worth, and what is therefore available to be recovered in each
// column. One of your two numbers should be much less exciting than the other,
// and if it is not, re-read Module 10's contention table.
//
// YOUR CODE HERE
static double PRED[2] = { 0.0, 0.0 };   // { flat, hot }

// ================================================================ the floor
__global__ void k_ceiling(const uint4* __restrict__ in, size_t n4, unsigned int* sink)
{
    unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        uint4 v = in[i];
        a0 += v.x; a1 += v.y; a2 += v.z; a3 += v.w;
    }
    unsigned int s = a0 + a1 + a2 + a3;
    if (s == 0xFFFFFFFFu) sink[threadIdx.x & 31] = s;
}

// ========================================================= given: the baseline
__global__ void hist_naive(const uint4* __restrict__ in, size_t n4,
                           unsigned int* __restrict__ hist, int nBins)
{
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        uint4 v = in[i];
        atomicAdd(&hist[binOf(v.x, nBins)], 1u);
        atomicAdd(&hist[binOf(v.y, nBins)], 1u);
        atomicAdd(&hist[binOf(v.z, nBins)], 1u);
        atomicAdd(&hist[binOf(v.w, nBins)], 1u);
    }
}

// ==================================================================== TODO 2
// DESIGN + IMPLEMENT: the accumulation pass.
//
// The requirement, and nothing more: *reduce the number of concurrent atomic
// operations contending for the same address, without changing the result, and
// without assuming the histogram fits anywhere it does not fit.*
//
// Two independent levers exist and they attack different things. One of them
// reduces contention *between blocks*; the other reduces it *inside a warp*.
// They compose. Neither of them is free, and at least one of them has a
// parameter with a wrong answer on each side (Module 10 measured what happens
// when a contention fix is applied to something that was not contended; this
// module's Example 2 measured what happens when the auxiliary footprint stops
// fitting in the 48 MB L2).
//
// `part` points at the scratch buffer your planFast() asked for; `gmask` is
// whatever you want it to be.
//
// YOUR CODE HERE
__global__ void fast_accum(const uint4* __restrict__ in, size_t n4,
                           unsigned int* __restrict__ part, int nBins, int gmask)
{
    (void)in; (void)n4; (void)part; (void)nBins; (void)gmask;
}

// ==================================================================== TODO 4
// Whatever TODO 2 accumulated into `part`, turn it into the final `hist`.
// State the traffic this pass costs; the harness prints your plan's numbers
// next to N so you can check it is small.
//
// YOUR CODE HERE
__global__ void fast_fold(const unsigned int* __restrict__ part,
                          unsigned int* __restrict__ hist, int nBins, int G)
{
    (void)part; (void)hist; (void)nBins; (void)G;
}

// ==================================================================== TODO 3
struct Plan { int grid, block, copies; size_t scratchBytes; };

// DESIGN. Return the launch shape, the number of private copies (if your
// answer has copies), and how much of the 64 MB scratch budget you want.
// The harness rejects a plan that asks for more than the budget, and prints
// the footprint as a percentage of the 48 MB L2 -- that percentage is the
// number to think hardest about.
//
// Returning zeros stops the program with a message.
//
// YOUR CODE HERE
static Plan planFast(int nBins, size_t n, int nSM, int blocksPerSM)
{
    (void)nBins; (void)n; (void)nSM; (void)blocksPerSM;
    Plan p; p.grid = 0; p.block = 0; p.copies = 0; p.scratchBytes = 0;
    return p;
}

// Orchestrate. You may launch as many kernels as you like here, in any order,
// and you may clear `scratch` however you like -- but everything you do in this
// function is inside the timed region, so count it.
//
// YOUR CODE HERE
static void runFast(const Plan& p, const uint4* in4, size_t n4,
                    unsigned int* hist, unsigned int* scratch, int nBins)
{
    (void)p; (void)in4; (void)n4; (void)hist; (void)scratch; (void)nBins;
}

// =============================================================== input data
static unsigned int xs32(unsigned int* st)
{
    unsigned int x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

// dist 0 "flat": uniform over all 65536 bins.
// dist 1 "hot" : 75% of the mass inside 32 bins, the rest spread uniformly.
static void gen(int dist, unsigned int* h, size_t n)
{
    unsigned int st = (dist == 0) ? 0x5EEDu : 0xBADCAFEu;
    const unsigned int span = 1073741824u / (unsigned)NBINS;
    for (size_t i = 0; i < n; ++i) {
        unsigned int r = xs32(&st);
        double u = (double)(xs32(&st) >> 8) * (1.0 / 16777216.0);
        if (dist == 1 && (r & 3u) != 3u) {
            unsigned int hot = (r >> 2) & 31u;
            h[i] = hot * span + (unsigned int)(u * (double)span);
        } else {
            unsigned int f = (unsigned int)(u * 1073741824.0);
            h[i] = f > 1073741823u ? 1073741823u : f;
        }
    }
}

static void warmup(const uint4* d4, unsigned int* sink, int grid)
{
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    float acc = 0.0f;
    while (acc < 1500.0f) {
        CHECK(cudaEventRecord(a));
        for (int i = 0; i < 8; ++i) k_ceiling<<<grid, BLK>>>(d4, N / 4, sink);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        float ms; CHECK(cudaEventElapsedTime(&ms, a, b)); acc += ms;
    }
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
}

static int octave(double got, double pred)
{
    if (pred <= 0.0) return 0;
    double r = got / pred;
    return (r >= 0.5 && r <= 2.0);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("=== Module 14 / Exercise 2 — %d bins ===\n", NBINS);
    printf("N = %zu keys = %.0f MiB;  the histogram alone is %d B, and the\n"
           "largest shared-memory allocation this GPU permits is 101376 B.\n",
           N, N * 4.0 / 1048576.0, NBINS * 4);

    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    const int nSM = prop.multiProcessorCount;

    int occ = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (void*)hist_naive, BLK, 0));
    Plan plan = planFast(NBINS, N, nSM, occ);
    if (plan.grid <= 0 || plan.block <= 0 || plan.scratchBytes == 0) {
        printf("Set TODO 3 (planFast) first.\n"); return 0;
    }
    if (plan.scratchBytes > SCRATCH_BYTES) {
        printf("plan asks for %zu B of scratch; the budget is %zu B.\n",
               plan.scratchBytes, SCRATCH_BYTES); return 1;
    }
    if (PRED[0] <= 0.0 || PRED[1] <= 0.0) { printf("Set TODO 1 first.\n"); return 0; }

    printf("\nyour plan: grid %d x %d, %d copies, %zu B scratch (%.2f MB, %.0f%% of L2)\n",
           plan.grid, plan.block, plan.copies, plan.scratchBytes,
           plan.scratchBytes / 1048576.0, 100.0 * plan.scratchBytes / 50331648.0);
    printf("cost model: %.3g global atomics from the accumulate pass (upper bound),\n"
           "            %.3g B read + %.3g B written by the fold pass,\n"
           "            against %.3g global atomics for the naive kernel.\n",
           (double)N, (double)plan.scratchBytes, (double)NBINS * 4, (double)N);

    // -------------------------------------------------------------- memory
    unsigned int* h_in = (unsigned int*)malloc(N * sizeof(unsigned int));
    unsigned long long* h_ref = (unsigned long long*)malloc((size_t)NBINS * sizeof(unsigned long long));
    unsigned int* h_out = (unsigned int*)malloc((size_t)NBINS * sizeof(unsigned int));
    if (!h_in || !h_ref || !h_out) { printf("host alloc failed\n"); return 1; }

    unsigned int* d_in[2];
    for (int d = 0; d < 2; ++d) {
        CHECK(cudaMalloc(&d_in[d], N * sizeof(unsigned int)));
        gen(d, h_in, N);
        CHECK(cudaMemcpy(d_in[d], h_in, N * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    unsigned int *d_hist, *d_scratch, *d_sink;
    CHECK(cudaMalloc(&d_hist, (size_t)NBINS * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_scratch, SCRATCH_BYTES));
    CHECK(cudaMalloc(&d_sink, 32 * sizeof(unsigned int)));
    CHECK(cudaMemset(d_sink, 0, 32 * sizeof(unsigned int)));

    printf("\nwarm-up (1500 ms)...\n");
    warmup((const uint4*)d_in[0], d_sink, nSM * occ);

    // -------------------------------------------------------------- timing
    cudaEvent_t evA, evB; CHECK(cudaEventCreate(&evA)); CHECK(cudaEventCreate(&evB));
    const int NC = 6;          // {ceiling, naive, fast} x {flat, hot}
    double best[NC]; int iters[NC];
    for (int c = 0; c < NC; ++c) {
        int k = c / 2, d = c % 2;
        CHECK(cudaEventRecord(evA));
        if (k == 0) k_ceiling<<<nSM * occ, BLK>>>((const uint4*)d_in[d], N / 4, d_sink);
        else if (k == 1) hist_naive<<<nSM * occ, BLK>>>((const uint4*)d_in[d], N / 4, d_hist, NBINS);
        else runFast(plan, (const uint4*)d_in[d], N / 4, d_hist, d_scratch, NBINS);
        CHECK(cudaEventRecord(evB)); CHECK(cudaEventSynchronize(evB));
        float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
        int it = (int)(10.0f / (ms > 0.0001f ? ms : 0.0001f));
        iters[c] = it < 3 ? 3 : (it > 60 ? 60 : it);
        best[c] = 1e30;
    }
    CHECK_KERNEL();
    for (int sw = 0; sw < NC; ++sw)
        for (int q = 0; q < NC; ++q) {
            int c = (q + sw) % NC, k = c / 2, d = c % 2;
            CHECK(cudaEventRecord(evA));
            for (int it = 0; it < iters[c]; ++it) {
                if (k == 0) k_ceiling<<<nSM * occ, BLK>>>((const uint4*)d_in[d], N / 4, d_sink);
                else if (k == 1) hist_naive<<<nSM * occ, BLK>>>((const uint4*)d_in[d], N / 4, d_hist, NBINS);
                else runFast(plan, (const uint4*)d_in[d], N / 4, d_hist, d_scratch, NBINS);
            }
            CHECK(cudaEventRecord(evB)); CHECK(cudaEventSynchronize(evB));
            float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
            double per = (double)ms / iters[c];
            if (per < best[c]) best[c] = per;
        }
    CHECK_KERNEL();

    printf("\n--- timing (min of %d rotated sweeps, ms) ---\n", NC);
    printf("  %-24s %10s %10s\n", "", DN[0], DN[1]);
    printf("  %-24s %10.4f %10.4f\n", "ceiling (stream only)", best[0], best[1]);
    printf("  %-24s %10.4f %10.4f\n", "hist_naive", best[2], best[3]);
    printf("  %-24s %10.4f %10.4f\n", "your fast version", best[4], best[5]);
    printf("  %-24s %9.2fx %9.2fx\n", "speedup", best[2] / best[4], best[3] / best[5]);
    printf("  %-24s %9.0f%% %9.0f%%\n", "% of ceiling (fast)",
           100.0 * best[0] / best[4], 100.0 * best[1] / best[5]);
    printf("  %-24s %9.0f%% %9.0f%%\n", "% of ceiling (naive)",
           100.0 * best[0] / best[2], 100.0 * best[1] / best[3]);
    printf("\n  ceiling %.4f ms = %.1f GB/s = %.0f%% of %.0f GB/s peak\n",
           best[0], N * 4.0 / (best[0] * 1e6), 100.0 * N * 4.0 / (best[0] * 1e6) / PEAK, PEAK);

    // ---------------------------------------------------------- validation
    printf("\n--- validation (separate pass) ---\n");
    int score = 0, maxScore = 0;
    int exact = 1;
    for (int d = 0; d < 2; ++d) {
        gen(d, h_in, N);
        for (int b = 0; b < NBINS; ++b) h_ref[b] = 0ull;
        for (size_t i = 0; i < N; ++i) h_ref[binOf(h_in[i], NBINS)]++;

        CHECK(cudaMemset(d_hist, 0, (size_t)NBINS * sizeof(unsigned int)));
        runFast(plan, (const uint4*)d_in[d], N / 4, d_hist, d_scratch, NBINS);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(h_out, d_hist, (size_t)NBINS * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));
        int bad = 0;
        unsigned long long tot = 0ull;
        for (int b = 0; b < NBINS; ++b) {
            tot += h_out[b];
            if ((unsigned long long)h_out[b] != h_ref[b]) bad++;
        }
        printf("  [%s] %-5s : %d/%d bins wrong, total %llu (expected %zu)\n",
               bad ? "FAIL" : "PASS", DN[d], bad, NBINS,
               (unsigned long long)tot, N);
        if (bad) exact = 0;
    }
    score += exact; maxScore += 1;

    double sFlat = best[2] / best[4], sHot = best[3] / best[5];
    int gHot = (sHot >= 5.0), gFlat = (sFlat >= 0.90);
    printf("  [%s] hot  distribution at least 5.00x (got %.2fx)\n", gHot ? "PASS" : "FAIL", sHot);
    printf("  [%s] flat distribution no worse than 0.90x (got %.2fx)\n", gFlat ? "PASS" : "FAIL", sFlat);
    score += gHot + gFlat; maxScore += 2;

    printf("  TODO 1 predictions (octave-scored):\n");
    for (int d = 0; d < 2; ++d) {
        double got = (d == 0) ? sFlat : sHot;
        int ok = octave(got, PRED[d]);
        printf("    [%s] %-5s predicted %.2fx, measured %.2fx\n",
               ok ? "PASS" : "FAIL", DN[d], PRED[d], got);
        score += ok; maxScore += 1;
    }
    printf("\n  score: %d/%d\n", score, maxScore);

    CHECK(cudaEventDestroy(evA)); CHECK(cudaEventDestroy(evB));
    for (int d = 0; d < 2; ++d) CHECK(cudaFree(d_in[d]));
    CHECK(cudaFree(d_hist)); CHECK(cudaFree(d_scratch)); CHECK(cudaFree(d_sink));
    free(h_in); free(h_ref); free(h_out);
    CHECK(cudaDeviceReset());

    printf("OVERALL: %s\n", score == maxScore ? "PASS" : "FAIL");
    return score == maxScore ? 0 : 1;
}
