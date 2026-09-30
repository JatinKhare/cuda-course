// =============================================================================
// Module 14 / Exercise 1 — the privatized histogram ladder
//
// GOAL
//   Build the four rungs above a global-atomic 256-bin byte histogram and find
//   out which rung is worth what on which input distribution. The harness times
//   six kernels x three distributions in one rotated sweep and validates every
//   one of them against an exact CPU reference.
//
//     v0  hist_global   one global atomic per element            (given)
//     v1  hist_shared   shared privatization, grid = ceil(N/BLK) (you write it)
//     v2  hist_shared   the same kernel, YOUR grid               (you choose it)
//     v3  hist_repl     R replicas of the histogram in shared    (you write it)
//     v4  hist_vec      v3 with uchar4 loads                     (you write it)
//
//   NOTE: the block is 128 threads and there are 256 bins. That is not an
//   accident; it is the first thing you have to get right.
//
// BUILD
//   nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN
//   .\exercise01.exe
//
// Useful while debugging (the bugs this exercise invites are exactly the ones
// these tools were built for):
//   compute-sanitizer --tool racecheck   .\exercise01.exe
//   compute-sanitizer --tool memcheck    .\exercise01.exe
//   compute-sanitizer --tool initcheck --initcheck-address-space shared .\exercise01.exe
//   (memcheck prints a benign "Resetting device while there are still other
//    users" warning on any program that calls cudaDeviceReset(). Ignore it.)
//
// TODO 1, TODO 3 and TODO 4's chooseR() must be filled before the program will
// run at all; everything else reports FAIL rather than crashing.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
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

static const size_t N     = 201326592;   // 192 MiB of bytes, 4.0x the 48 MB L2
static const int    NBINS = 256;
static const int    BLK   = 128;         // NOTE: nBins > blockDim.x
static const int    NDIST = 3;
static const double PEAK  = 432.0;
static const char*  DIST_NAME[NDIST] = { "uniform", "same-bin", "clustered" };

// ==================================================================== TODO 1
// PREDICT, IN WRITING, BEFORE YOU BUILD ANYTHING.
//
// For each of the three input distributions below, predict the speedup of the
// finished kernel (v4: coarsened, privatized, replicated, vectorized) over the
// global-atomic baseline v0. Scored to within a factor of two.
//
//   uniform    every one of the 256 bins equally likely
//   same-bin   every single element lands in bin 37
//   clustered  runs of 8192 identical values, so a warp's 32 lanes always
//              share one bin
//
// Before you guess: work out what v0 costs on each (Module 10's contention
// economics is the whole input to this) and what v4 costs on each. One of those
// two numbers should be the same in all three columns, and you should be able
// to say why before you measure it.
//
// YOUR CODE HERE
static double PRED_SPEEDUP[NDIST] = { 0.0, 0.0, 0.0 };

// ============================================================ given baseline
__global__ void hist_global(const unsigned char* __restrict__ in, size_t n,
                            unsigned int* __restrict__ hist)
{
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&hist[in[i]], 1u);
}

__global__ void k_ceiling(const uchar4* __restrict__ in, size_t n4,
                          unsigned int* __restrict__ sink)
{
    unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        uchar4 v = in[i];
        a0 += v.x; a1 += v.y; a2 += v.z; a3 += v.w;
    }
    unsigned int s = a0 + a1 + a2 + a3;
    if (s == 0xFFFFFFFFu) sink[threadIdx.x & 31] = s;
}

// ==================================================================== TODO 2
__global__ void hist_shared(const unsigned char* __restrict__ in, size_t n,
                            unsigned int* __restrict__ hist, int nBins)
{
    extern __shared__ unsigned int s[];

    // TODO 2 — the whole privatized histogram, in four moves. The dynamic
    // shared allocation is exactly nBins * sizeof(unsigned int) bytes.
    //
    //   (a) every one of the nBins private bins must hold 0 before any thread
    //       adds to it. Every bin. Read the block size again.
    //   (b) whatever guarantee (a) needs in order to be true for the OTHER
    //       threads of this block, provide it.
    //   (c) accumulate. This kernel must work for ANY grid size, including one
    //       far smaller than n / blockDim.x, so a thread handles many elements.
    //       The private bins are shared between the threads of this block, so
    //       the update still has to be indivisible.
    //   (d) fold the private copy into hist[]. There is a guarantee needed
    //       before this can start, and it is not the same guarantee as (b).
    //       Bins that stayed at zero need not be flushed at all.
    //
    // YOUR CODE HERE
    (void)s; (void)in; (void)n; (void)hist; (void)nBins;
}

// ==================================================================== TODO 4
__global__ void hist_repl(const unsigned char* __restrict__ in, size_t n,
                          unsigned int* __restrict__ hist, int nBins, int rmask)
{
    extern __shared__ unsigned int s[];

    // TODO 4 (part 1) — R = rmask+1 independent copies of the histogram, so
    // that threads which would otherwise collide on one bin collide on
    // different copies. The dynamic shared allocation is
    // R * nBins * sizeof(unsigned int) bytes.
    //
    //   (a) zero all R copies, pick this thread's copy, accumulate into it.
    //       WHICH threads share a copy is a decision, not a detail: Ada's
    //       shared-memory unit already merges the lanes of a single warp that
    //       target one address into one operation, so splitting a warp across
    //       copies buys nothing and costs something. Also decide how the copies
    //       are laid out relative to each other; one of the two obvious layouts
    //       turns a conflict-free access pattern into a banked one (Module 7).
    //   (b) fold the R copies per bin, then flush.
    //
    // YOUR CODE HERE
    (void)s; (void)in; (void)n; (void)hist; (void)nBins; (void)rmask;
}

// ==================================================================== TODO 5
__global__ void hist_vec(const uchar4* __restrict__ in4, size_t n4,
                         unsigned int* __restrict__ hist, int nBins, int rmask)
{
    extern __shared__ unsigned int s[];

    // TODO 5 — the same kernel as TODO 4, except that the input side reads
    // uchar4 instead of unsigned char. n4 = N/4 and N is a multiple of 4, so
    // there is no tail to handle. Four bins come out of every load.
    //
    // Before you write it: this changes nothing whatsoever about the number of
    // atomic operations executed. Commit to whether it can possibly matter, and
    // by how much, then look at what it measures.
    //
    // YOUR CODE HERE
    (void)s; (void)in4; (void)n4; (void)hist; (void)nBins; (void)rmask;
}

// ==================================================================== TODO 3
// DESIGN. Return the number of blocks to launch for v2, v3 and v4.
//
// v1 uses grid = ceil(N / blockDim) — one element per thread — and the harness
// prints what that costs in flush atomics. Your grid has to satisfy two
// requirements that pull in opposite directions:
//
//   * grid * nBins must be small compared with N, or the flush is a second
//     histogram's worth of global atomics (Module 10's traffic arithmetic);
//   * the machine must not be left idle.
//
// `blocksPerSM` is what cudaOccupancyMaxActiveBlocksPerMultiprocessor returned
// for this kernel at its actual shared-memory footprint. Returning 0 makes the
// program stop and tell you to fill this in.
//
// YOUR CODE HERE
static int chooseGrid(int nSM, int blocksPerSM, size_t n, int nBins)
{
    (void)nSM; (void)blocksPerSM; (void)n; (void)nBins;
    return 0;
}

// TODO 4 (part 2) — DESIGN. How many replicas?
// Derive the shared-memory footprint R * nBins * 4 B, work out what that does
// to the number of resident blocks per SM, and decide. The harness prints the
// occupancy it gets for your answer. Returning 0 stops the program.
//
// YOUR CODE HERE
static int chooseR(int nBins, int blockDim)
{
    (void)nBins; (void)blockDim;
    return 0;
}

// =============================================================== input data
static unsigned int xs32(unsigned int* st)
{
    unsigned int x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

static void gen(int dist, unsigned char* h, size_t n, unsigned long long* ref)
{
    for (int b = 0; b < NBINS; ++b) ref[b] = 0ull;
    unsigned int st = 0xC0FFEE11u + 7919u * (unsigned)dist;
    for (size_t i = 0; i < n; ++i) {
        unsigned char v;
        if (dist == 0)      v = (unsigned char)(xs32(&st) >> 24);
        else if (dist == 1) v = (unsigned char)37;
        else                v = (unsigned char)((i >> 13) & 255);
        h[i] = v;
        ref[v]++;
    }
}

// ================================================================== harness
enum { K_CEIL = 0, K_GLOBAL, K_NAIVEGRID, K_COARSE, K_REPL, K_VEC, NKER };
static const char* KNAME[NKER] = {
    "ceiling (stream only)",
    "v0 global atomic",
    "v1 hist_shared, grid=ceil(N/BLK)",
    "v2 hist_shared, your grid",
    "v3 hist_repl, your grid and R",
    "v4 hist_vec (uchar4)"
};

struct Geom { int grid[NKER], smem[NKER], R; };

static void launch(int k, const Geom& g, const unsigned char* d_in,
                   unsigned int* d_hist, unsigned int* d_sink)
{
    switch (k) {
    case K_CEIL:      k_ceiling<<<g.grid[k], BLK>>>((const uchar4*)d_in, N / 4, d_sink); break;
    case K_GLOBAL:    hist_global<<<g.grid[k], BLK>>>(d_in, N, d_hist); break;
    case K_NAIVEGRID: hist_shared<<<g.grid[k], BLK, g.smem[k]>>>(d_in, N, d_hist, NBINS); break;
    case K_COARSE:    hist_shared<<<g.grid[k], BLK, g.smem[k]>>>(d_in, N, d_hist, NBINS); break;
    case K_REPL:      hist_repl<<<g.grid[k], BLK, g.smem[k]>>>(d_in, N, d_hist, NBINS, g.R - 1); break;
    default:          hist_vec<<<g.grid[k], BLK, g.smem[k]>>>((const uchar4*)d_in, N / 4,
                                                              d_hist, NBINS, g.R - 1); break;
    }
}

static void warmup(const unsigned char* d, unsigned int* sink, int grid)
{
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    float acc = 0.0f;
    while (acc < 1500.0f) {
        CHECK(cudaEventRecord(a));
        for (int i = 0; i < 8; ++i) k_ceiling<<<grid, BLK>>>((const uchar4*)d, N / 4, sink);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        float ms; CHECK(cudaEventElapsedTime(&ms, a, b)); acc += ms;
    }
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
}

static int octave(double got, double pred)
{
    if (pred <= 0.0) return 0;
    double r = got / pred;
    return (r >= 0.5 && r <= 2.0) ? 1 : 0;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("=== Module 14 / Exercise 1 — the privatized histogram ladder ===\n");
    printf("N = %zu bytes (%.0f MiB), %d bins, block = %d threads (%d warps)\n",
           N, N / 1048576.0, NBINS, BLK, BLK / 32);

    if (chooseR(NBINS, BLK) < 1 || chooseGrid(1, 1, N, NBINS) < 1) {
        printf("Set TODO 3 (chooseGrid) and TODO 4 (chooseR) first.\n"); return 0;
    }
    if (PRED_SPEEDUP[0] <= 0.0 || PRED_SPEEDUP[1] <= 0.0 || PRED_SPEEDUP[2] <= 0.0) {
        printf("Set TODO 1 (the three predictions) first.\n"); return 0;
    }

    unsigned char* h_in = (unsigned char*)malloc(N);
    unsigned long long* h_ref = (unsigned long long*)malloc((size_t)NDIST * NBINS * sizeof(unsigned long long));
    if (!h_in || !h_ref) { printf("host alloc failed\n"); return 1; }

    unsigned char* d_in[NDIST];
    for (int d = 0; d < NDIST; ++d) {
        CHECK(cudaMalloc(&d_in[d], N));
        gen(d, h_in, N, h_ref + (size_t)d * NBINS);
        CHECK(cudaMemcpy(d_in[d], h_in, N, cudaMemcpyHostToDevice));
    }
    unsigned int *d_hist, *d_sink;
    CHECK(cudaMalloc(&d_hist, NBINS * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_sink, 32 * sizeof(unsigned int)));
    CHECK(cudaMemset(d_sink, 0, 32 * sizeof(unsigned int)));

    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    const int nSM = prop.multiProcessorCount;

    Geom g;
    g.R = chooseR(NBINS, BLK);
    if (g.R < 1 || g.R > BLK / 32) { printf("Set TODO 4 (chooseR) first.\n"); return 0; }
    if (PRED_SPEEDUP[0] <= 0.0 || PRED_SPEEDUP[1] <= 0.0 || PRED_SPEEDUP[2] <= 0.0) {
        printf("Set TODO 1 (the three predictions) first.\n"); return 0;
    }

    int occ1 = 0, occR = 0, occ0 = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ1, (void*)hist_shared, BLK, NBINS * 4));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occR, (void*)hist_repl, BLK, g.R * NBINS * 4));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ0, (void*)hist_global, BLK, 0));

    int gCoarse = chooseGrid(nSM, occ1, N, NBINS);
    int gRepl   = chooseGrid(nSM, occR, N, NBINS);
    if (gCoarse <= 0 || gRepl <= 0) { printf("Set TODO 3 (chooseGrid) first.\n"); return 0; }

    g.grid[K_CEIL] = nSM * occ0;            g.smem[K_CEIL] = 0;
    g.grid[K_GLOBAL] = nSM * occ0;          g.smem[K_GLOBAL] = 0;
    g.grid[K_NAIVEGRID] = (int)((N + BLK - 1) / BLK); g.smem[K_NAIVEGRID] = NBINS * 4;
    g.grid[K_COARSE] = gCoarse;             g.smem[K_COARSE] = NBINS * 4;
    g.grid[K_REPL] = gRepl;                 g.smem[K_REPL] = g.R * NBINS * 4;
    g.grid[K_VEC] = gRepl;                  g.smem[K_VEC] = g.R * NBINS * 4;

    printf("\nyour choices: R = %d (%d B shared, %d blocks/SM), grid = %d\n",
           g.R, g.R * NBINS * 4, occR, gRepl);
    printf("flush global atomics: v1 %.3g  v2 %.3g  v3/v4 %.3g   (N = %.3g)\n",
           (double)g.grid[K_NAIVEGRID] * NBINS, (double)gCoarse * NBINS,
           (double)gRepl * NBINS, (double)N);

    printf("\nwarm-up (1500 ms)...\n");
    warmup(d_in[0], d_sink, g.grid[K_CEIL]);

    cudaEvent_t evA, evB; CHECK(cudaEventCreate(&evA)); CHECK(cudaEventCreate(&evB));
    const int NC = NKER * NDIST;
    double best[NKER * NDIST]; int iters[NKER * NDIST];
    for (int c = 0; c < NC; ++c) {
        int d = c / NKER, k = c % NKER;
        CHECK(cudaMemset(d_hist, 0, NBINS * sizeof(unsigned int)));
        CHECK(cudaEventRecord(evA));
        launch(k, g, d_in[d], d_hist, d_sink);
        CHECK(cudaEventRecord(evB)); CHECK(cudaEventSynchronize(evB));
        float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
        int it = (int)(10.0f / (ms > 0.0001f ? ms : 0.0001f));
        iters[c] = it < 3 ? 3 : (it > 100 ? 100 : it);
        best[c] = 1e30;
    }
    CHECK_KERNEL();

    for (int sw = 0; sw < NC; ++sw)
        for (int q = 0; q < NC; ++q) {
            int c = (q + sw) % NC, d = c / NKER, k = c % NKER;
            CHECK(cudaEventRecord(evA));
            for (int it = 0; it < iters[c]; ++it) launch(k, g, d_in[d], d_hist, d_sink);
            CHECK(cudaEventRecord(evB)); CHECK(cudaEventSynchronize(evB));
            float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
            double per = (double)ms / iters[c];
            if (per < best[c]) best[c] = per;
        }
    CHECK_KERNEL();

    printf("\n--- timing (min of %d rotated sweeps, ms) ---\n", NC);
    printf("  %-34s", "kernel");
    for (int d = 0; d < NDIST; ++d) printf("%12s", DIST_NAME[d]);
    printf("\n");
    for (int k = 0; k < NKER; ++k) {
        printf("  %-34s", KNAME[k]);
        for (int d = 0; d < NDIST; ++d) printf("%12.4f", best[d * NKER + k]);
        printf("\n");
    }
    // The ceiling kernel does not depend on the input, so the honest min-of-N
    // for it is the minimum over all three of its slots in the sweep.
    double ceilMs = best[K_CEIL];
    for (int d = 1; d < NDIST; ++d)
        if (best[d * NKER + K_CEIL] < ceilMs) ceilMs = best[d * NKER + K_CEIL];

    printf("\n  speedup over v0, and %% of the measured ceiling:\n");
    for (int k = K_NAIVEGRID; k < NKER; ++k) {
        printf("  %-34s", KNAME[k]);
        for (int d = 0; d < NDIST; ++d)
            printf("  %8.2fx/%3.0f%%", best[d * NKER + K_GLOBAL] / best[d * NKER + k],
                   100.0 * ceilMs / best[d * NKER + k]);
        printf("\n");
    }
    printf("\n  ceiling: %.4f ms = %.1f GB/s = %.1f%% of %.0f GB/s peak\n",
           ceilMs, (double)N / (ceilMs * 1e6),
           100.0 * (double)N / (ceilMs * 1e6) / PEAK, PEAK);
    printf("  (on a thermally settled machine every v4 column reaches 99-100%% of this;\n"
           "   the scored gate is the v4/v2 ratio, which is stable across thermal states.)\n");

    // ------------------------------------------------- validation (pass 2)
    printf("\n--- validation ---\n");
    int correct = 0;
    for (int k = K_NAIVEGRID; k < NKER; ++k) {
        int bad = 0;
        for (int d = 0; d < NDIST; ++d) {
            CHECK(cudaMemset(d_hist, 0, NBINS * sizeof(unsigned int)));
            launch(k, g, d_in[d], d_hist, d_sink);
            CHECK_KERNEL();
            unsigned int h[NBINS];
            CHECK(cudaMemcpy(h, d_hist, sizeof(h), cudaMemcpyDeviceToHost));
            for (int b = 0; b < NBINS; ++b)
                if ((unsigned long long)h[b] != h_ref[(size_t)d * NBINS + b]) bad++;
        }
        printf("  [%s] %-34s exact on all %d distributions%s\n",
               bad ? "FAIL" : "PASS", KNAME[k], NDIST, bad ? " (WRONG)" : "");
        if (!bad) correct++;
    }

    int score = correct, maxScore = 4;
    // Both perf gates are RATIOS between kernels timed in the same rotated
    // sweep, which spec section 12 rule 5 names as the stable quantity here.
    int gate1 = 1;
    double worstVec = 1e30;
    for (int d = 0; d < NDIST; ++d) {
        double r = best[d * NKER + K_COARSE] / best[d * NKER + K_VEC];
        if (r < worstVec) worstVec = r;
        if (r < 1.50) gate1 = 0;
    }
    printf("  [%s] v4 at least 1.50x v2 on every distribution (worst %.2fx)\n",
           gate1 ? "PASS" : "FAIL", worstVec);
    score += gate1; maxScore += 1;

    double sSame = best[1 * NKER + K_GLOBAL] / best[1 * NKER + K_COARSE];
    int gate2 = (sSame >= 20.0);
    printf("  [%s] v2 at least 20x v0 on same-bin (got %.1fx)\n", gate2 ? "PASS" : "FAIL", sSame);
    score += gate2; maxScore += 1;

    printf("  TODO 1 predictions (octave-scored, v4 vs v0):\n");
    for (int d = 0; d < NDIST; ++d) {
        double got = best[d * NKER + K_GLOBAL] / best[d * NKER + K_VEC];
        int ok = octave(got, PRED_SPEEDUP[d]);
        printf("    [%s] %-10s predicted %7.1fx, measured %7.1fx\n",
               ok ? "PASS" : "FAIL", DIST_NAME[d], PRED_SPEEDUP[d], got);
        score += ok; maxScore += 1;
    }

    printf("\n  score: %d/%d\n", score, maxScore);

    CHECK(cudaEventDestroy(evA)); CHECK(cudaEventDestroy(evB));
    for (int d = 0; d < NDIST; ++d) CHECK(cudaFree(d_in[d]));
    CHECK(cudaFree(d_hist)); CHECK(cudaFree(d_sink));
    free(h_in); free(h_ref);
    CHECK(cudaDeviceReset());

    printf("OVERALL: %s\n", score == maxScore ? "PASS" : "FAIL");
    return score == maxScore ? 0 : 1;
}
