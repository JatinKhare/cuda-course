// =====================================================================
// Module 10 / Exercise 2 : SOLUTION -- "Make the contention go away"
//
// THE WORKLOAD -- a weighted category tally (a scatter-add, not a
// histogram and not a reduction; Modules 12 and 14 own those).
//
//   For n samples, each carrying a category cat[i] in [0, ncat) and an
//   integer weight w[i], compute
//
//       total[c] = sum of w[i] over all i with cat[i] == c        (exact)
//
//   `tally_naive` does the obvious thing: one global atomicAdd per
//   sample. It is CORRECT. It is also, on the contended inputs, more
//   than an order of magnitude off the achievable rate, and the reason
//   is entirely visible in Module 10's contention curve.
//
// YOUR JOB
//   Write `tally_fast`. It must produce bit-identical results to the CPU
//   reference in EVERY scenario the harness runs, and it must be fast in
//   the contended ones without being slow in the uncontended ones.
//
//   The harness runs four scenarios, which is the whole point:
//
//     ncat = 16,   clustered : few addresses, and neighbouring lanes of a
//                              warp usually target the SAME one
//     ncat = 16,   shuffled  : the same few addresses, but lanes of a
//                              warp rarely collide with each other
//     ncat = 4096, clustered : many addresses, lanes still collide
//     ncat = 4096, shuffled  : many addresses, almost no collisions
//
//   `ncat` is a RUNTIME argument, not a compile-time constant. Several
//   plausible strategies win one column and lose another. Read the two
//   axes -- "how many distinct addresses" and "how many lanes of one warp
//   hit the same address" -- as independent, because they are.
//
// CONSTRAINTS
//   - No more than 48 KB of shared memory per block.
//   - Results must be exact integers, identical every run.
//   - You may change nothing above the "YOUR CODE" banner.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
// RUN:   .\exercise02_solution.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(x) do {                                                      \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                  \
                cudaGetErrorName(e_), __FILE__, __LINE__,                  \
                cudaGetErrorString(e_));                                   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

static const int N          = 8000000;
static const int NCAT_SMALL = 16;
static const int NCAT_BIG   = 4096;
static const int SWEEPS     = 4;
static const int ITERS      = 20;
static const int SMEM_LIMIT = 48 * 1024;

// ---------------------------------------------------------------------
// The baseline. Do not modify. One global atomic per sample.
// ---------------------------------------------------------------------
__global__ void tally_naive(const unsigned int* __restrict__ cat,
                            const unsigned int* __restrict__ w,
                            int n, int ncat,
                            unsigned long long* __restrict__ total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    (void)ncat;
    atomicAdd(&total[cat[i]], (unsigned long long)w[i]);
}

// =====================================================================
//                            YOUR CODE
// =====================================================================

// ---------------------------------------------------------------------
// TODO 1: Choose the launch configuration for `tally_fast`.
//
//   Set *grid, *block and *smemBytes. All three may depend on n and on
//   ncat -- the harness calls this once per scenario and honours whatever
//   you return, including 0 dynamic shared bytes.
//
//   Leave *grid at 0 and the program stops with a message.
//
//   Two things to decide, not one:
//     - how many threads, and therefore whether a thread handles one
//       sample or several (Module 3's grid-stride loop is available);
//     - how much per-block state you are willing to pay for, given that
//       *smemBytes must be <= 48 KB for ANY ncat the harness passes, and
//       given that every byte of it has to be initialised and drained by
//       the block that owns it.
// ---------------------------------------------------------------------
static void choose_launch(int n, int ncat, int* grid, int* block, size_t* smemBytes)
{
    (void)n; (void)ncat;
    const int blk = 256;
    *block = blk;

    // A private bin is 32 bits, not 64. Per-block partial sums cannot
    // overflow: max weight is 1000, so a block would need 4.29e6 samples
    // to wrap, and no block ever sees that many. Halving the bin width
    // halves the shared-memory footprint and buys a cheaper shared atomic.
    const size_t priv = (size_t)ncat * sizeof(unsigned int);

    if (priv <= (size_t)SMEM_LIMIT) {
        *smemBytes = priv;
        // Two competing limits on the grid:
        //   - work:    ~8 samples per thread keeps the grid-stride loop
        //              long enough to amortise init + flush.
        //   - traffic: the flush costs grid*ncat global atomics. Cap the
        //              grid so that stays under n/16.
        long long byWork    = ((long long)n + (long long)blk * 8 - 1)
                              / ((long long)blk * 8);
        long long byTraffic = (long long)n / (16LL * (long long)ncat);
        long long g = byWork < byTraffic ? byWork : byTraffic;
        if (g < 80) g = 80;                 // at least 2 blocks per SM
        if (g > 65535) g = 65535;
        *grid = (int)g;
    } else {
        // ncat too large to privatise at all: fall back to one atomic per
        // sample, which is fine because ncat that large means almost no
        // contention anyway.
        *smemBytes = 0;
        *grid = (int)(((long long)n + blk - 1) / blk);
    }
}

// ---------------------------------------------------------------------
// TODO 2: Write the kernel. THIS IS THE DESIGN TODO.
//
//   Requirement, stated semantically on purpose:
//
//     Reduce the number of concurrent global atomic operations that
//     target the SAME address, without changing the result.
//
//   That is the only thing that has to improve. How you achieve it is
//   yours to choose, and the four scenarios are chosen so that no single
//   one-line answer wins all four. Whatever you pick, the kernel must
//   still be exact and must still work for ncat = 4096.
//
//   Available to you from earlier modules: shared memory (M6), bank
//   behaviour (M7), warp intrinsics and the active mask (M8), barriers
//   and the rule that every thread of the block must reach them (M9),
//   and everything in this module.
//
//   The kernel is launched as
//       tally_fast<<<grid, block, smemBytes>>>(cat, w, n, ncat, total);
//   with `total` already zeroed.
// ---------------------------------------------------------------------
__global__ void tally_fast(const unsigned int* __restrict__ cat,
                           const unsigned int* __restrict__ w,
                           int n, int ncat,
                           unsigned long long* __restrict__ total)
{
    extern __shared__ unsigned int s[];

    const int stride = gridDim.x * blockDim.x;
    const int i0     = blockIdx.x * blockDim.x + threadIdx.x;

    if ((size_t)ncat * sizeof(unsigned int) > (size_t)SMEM_LIMIT) {
        // No private copy available. Straight to global.
        for (int i = i0; i < n; i += stride)
            atomicAdd(&total[cat[i]], (unsigned long long)w[i]);
        return;
    }

    // ---- 1. zero the block's private copy ----
    for (int t = threadIdx.x; t < ncat; t += blockDim.x) s[t] = 0u;
    __syncthreads();      // nobody may add before every bin is zero

    // ---- 2. accumulate privately; contention now lives at the SM ----
    for (int i = i0; i < n; i += stride)
        atomicAdd(&s[cat[i]], w[i]);
    __syncthreads();      // every add complete before anyone reads

    // ---- 3. flush: exactly ncat global atomics per block, and fewer if
    //         some bins were never touched ----
    for (int t = threadIdx.x; t < ncat; t += blockDim.x)
        if (s[t] != 0u) atomicAdd(&total[t], (unsigned long long)s[t]);
}

// ---------------------------------------------------------------------
// TODO 3: A traffic model for your design.
//
//   Return the number of GLOBAL atomic instructions your kernel executes
//   for the given n, ncat, grid and block. Count instructions, not bytes
//   and not lanes -- if you elect one lane of a warp to perform an atomic
//   on behalf of 32 lanes, that is one instruction.
//
//   The naive kernel's answer is exactly n. The harness prints yours
//   beside it. It does not verify it; it is your prediction, and if the
//   measured ratio does not resemble the ratio of these two numbers, the
//   interesting work of this exercise is explaining why.
// ---------------------------------------------------------------------
static double predicted_global_atomics(int n, int ncat, int grid, int block)
{
    (void)block;
    if ((size_t)ncat * sizeof(unsigned int) > (size_t)SMEM_LIMIT)
        return (double)n;                       // fallback path
    return (double)grid * (double)ncat;         // one flush per bin per block
}

// ---------------------------------------------------------------------
// TODO 4: Commit to two numbers BEFORE you build.
//
//   PRED_RATIO_16   : tally_naive time / tally_fast time, expected, for
//                     the (ncat = 16, clustered) scenario
//   PRED_RATIO_4096 : the same ratio for (ncat = 4096, clustered)
//
//   Use the contention curve from the lesson. Scoring is by octave: you
//   are right if your prediction is within a factor of 2 of the measured
//   ratio. The two will not be remotely similar. The second one is very
//   sensitive to a design decision that barely affects the first; working
//   out which decision, and in which direction, is most of the thinking
//   in this exercise.
//
//   Set either to 0.0 and the program stops.
// ---------------------------------------------------------------------
static const double PRED_RATIO_16   = 12.0;  // YOUR CODE HERE (TODO 4)
static const double PRED_RATIO_4096 = 3.0;   // YOUR CODE HERE (TODO 4)

// =====================================================================
//                          END OF YOUR CODE
// =====================================================================

struct Scenario {
    const char* name;
    int         ncat;
    int         shuffled;
    unsigned int* d_cat;
    unsigned long long* ref;
    double      tNaive, tFast;
};

static unsigned int lcg(unsigned int& s) { s = s * 1664525u + 1013904223u; return s; }

int main(void)
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("n = %d samples, weighted tally, exact unsigned 64-bit sums\n\n", N);

    {
        int g = 0, b = 0; size_t sm = 0;
        choose_launch(N, NCAT_SMALL, &g, &b, &sm);
        if (g <= 0 || b <= 0) { printf("Set TODO 1 first.\n"); return 0; }
        if (PRED_RATIO_16 <= 0.0 || PRED_RATIO_4096 <= 0.0) {
            printf("Set TODO 4 first.\n"); return 0;
        }
    }

    // ---------------- deterministic input ----------------
    unsigned int* h_w   = (unsigned int*)malloc((size_t)N * 4);
    unsigned int* h_c16c = (unsigned int*)malloc((size_t)N * 4);  // 16, clustered
    unsigned int* h_c16s = (unsigned int*)malloc((size_t)N * 4);  // 16, shuffled
    unsigned int* h_c4kc = (unsigned int*)malloc((size_t)N * 4);  // 4096, clustered
    unsigned int* h_c4ks = (unsigned int*)malloc((size_t)N * 4);  // 4096, shuffled

    {
        unsigned int s = 987654321u;
        for (int i = 0; i < N; ++i) h_w[i] = (lcg(s) >> 20) % 1000u + 1u;

        // Skewed: category 0 takes about half of everything, then a
        // geometric tail. Clustered: the category changes only every
        // `runlen` samples, so lanes of a warp usually agree.
        s = 13572468u;
        int runlen = 0; unsigned int cur16 = 0, cur4k = 0;
        for (int i = 0; i < N; ++i) {
            if (runlen == 0) {
                runlen = 1 + (int)((lcg(s) >> 16) % 96u);
                unsigned int r = lcg(s) >> 8;
                unsigned int k = 0;
                while (k < 15u && (r & 1u)) { ++k; r >>= 1; }   // geometric
                cur16 = k;
                cur4k = (k * 251u + ((lcg(s) >> 9) % 256u)) % (unsigned)NCAT_BIG;
            }
            h_c16c[i] = cur16;
            h_c4kc[i] = cur4k;
            --runlen;
        }
        // Shuffled: same multiset, deterministic Fisher-Yates.
        memcpy(h_c16s, h_c16c, (size_t)N * 4);
        memcpy(h_c4ks, h_c4kc, (size_t)N * 4);
        s = 24681357u;
        for (int i = N - 1; i > 0; --i) {
            int j = (int)(((unsigned long long)lcg(s) * (unsigned long long)(i + 1)) >> 32);
            unsigned int t = h_c16s[i]; h_c16s[i] = h_c16s[j]; h_c16s[j] = t;
        }
        s = 97531864u;
        for (int i = N - 1; i > 0; --i) {
            int j = (int)(((unsigned long long)lcg(s) * (unsigned long long)(i + 1)) >> 32);
            unsigned int t = h_c4ks[i]; h_c4ks[i] = h_c4ks[j]; h_c4ks[j] = t;
        }
    }

    // ---------------- CPU references ----------------
    unsigned long long* ref16c = (unsigned long long*)calloc(NCAT_SMALL, 8);
    unsigned long long* ref16s = (unsigned long long*)calloc(NCAT_SMALL, 8);
    unsigned long long* ref4kc = (unsigned long long*)calloc(NCAT_BIG, 8);
    unsigned long long* ref4ks = (unsigned long long*)calloc(NCAT_BIG, 8);
    for (int i = 0; i < N; ++i) {
        ref16c[h_c16c[i]] += h_w[i];
        ref16s[h_c16s[i]] += h_w[i];
        ref4kc[h_c4kc[i]] += h_w[i];
        ref4ks[h_c4ks[i]] += h_w[i];
    }
    {
        double busiest = 0.0, tot = 0.0;
        for (int c = 0; c < NCAT_SMALL; ++c) { tot += (double)ref16c[c];
            if ((double)ref16c[c] > busiest) busiest = (double)ref16c[c]; }
        printf("skew: with ncat=16 the busiest category carries %.1f%% of the weight\n\n",
               100.0 * busiest / tot);
    }

    // ---------------- device buffers ----------------
    unsigned int *d_w, *d16c, *d16s, *d4kc, *d4ks;
    unsigned long long* d_tot;
    CHECK(cudaMalloc(&d_w,   (size_t)N * 4));
    CHECK(cudaMalloc(&d16c,  (size_t)N * 4));
    CHECK(cudaMalloc(&d16s,  (size_t)N * 4));
    CHECK(cudaMalloc(&d4kc,  (size_t)N * 4));
    CHECK(cudaMalloc(&d4ks,  (size_t)N * 4));
    CHECK(cudaMalloc(&d_tot, (size_t)NCAT_BIG * 8));
    CHECK(cudaMemcpy(d_w,  h_w,   (size_t)N * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d16c, h_c16c, (size_t)N * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d16s, h_c16s, (size_t)N * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d4kc, h_c4kc, (size_t)N * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d4ks, h_c4ks, (size_t)N * 4, cudaMemcpyHostToDevice));

    Scenario sc[4] = {
        { "ncat=16   clustered", NCAT_SMALL, 0, d16c, ref16c, 1e30, 1e30 },
        { "ncat=16   shuffled ", NCAT_SMALL, 1, d16s, ref16s, 1e30, 1e30 },
        { "ncat=4096 clustered", NCAT_BIG,   0, d4kc, ref4kc, 1e30, 1e30 },
        { "ncat=4096 shuffled ", NCAT_BIG,   1, d4ks, ref4ks, 1e30, 1e30 },
    };

    // ---- validate the launch configs before we run anything ----
    int   fg[4], fb[4]; size_t fs[4];
    for (int k = 0; k < 4; ++k) {
        choose_launch(N, sc[k].ncat, &fg[k], &fb[k], &fs[k]);
        if (fg[k] <= 0 || fb[k] <= 0 || fb[k] > 1024) {
            printf("TODO 1 returned an illegal config for %s: grid=%d block=%d\n",
                   sc[k].name, fg[k], fb[k]);
            return 1;
        }
        if (fs[k] > (size_t)SMEM_LIMIT) {
            printf("TODO 1 asks for %zu B of shared memory for %s; the limit is %d B.\n",
                   fs[k], sc[k].name, SMEM_LIMIT);
            return 1;
        }
    }

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    // duration-based clock warm-up
    for (int i = 0; i < 40; ++i)
        tally_naive<<<(N + 255) / 256, 256>>>(d16c, d_w, N, NCAT_SMALL, d_tot);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

#define TIME(store, launch)                                                    \
    do {                                                                       \
        launch; CHECK(cudaDeviceSynchronize());                                \
        CHECK(cudaEventRecord(e0));                                            \
        for (int r_ = 0; r_ < ITERS; ++r_) { launch; }                         \
        CHECK(cudaEventRecord(e1));                                            \
        CHECK(cudaEventSynchronize(e1));                                       \
        float ms_ = 0.f; CHECK(cudaEventElapsedTime(&ms_, e0, e1));            \
        if (ms_ / ITERS < (store)) (store) = ms_ / ITERS;                      \
    } while (0)

    // ---- all eight configurations, back to back, min of SWEEPS ----
    for (int sweep = 0; sweep < SWEEPS; ++sweep)
        for (int k = 0; k < 4; ++k) {
            TIME(sc[k].tNaive,
                 (tally_naive<<<(N + 255) / 256, 256>>>(sc[k].d_cat, d_w, N,
                                                        sc[k].ncat, d_tot)));
            TIME(sc[k].tFast,
                 (tally_fast<<<fg[k], fb[k], fs[k]>>>(sc[k].d_cat, d_w, N,
                                                      sc[k].ncat, d_tot)));
        }
    CHECK(cudaGetLastError());
#undef TIME

    // ---------------- validation, second pass ----------------
    printf("=== correctness ===\n");
    unsigned long long* h_tot = (unsigned long long*)malloc((size_t)NCAT_BIG * 8);
    int correct[4], naiveOk[4];
    for (int k = 0; k < 4; ++k) {
        CHECK(cudaMemset(d_tot, 0, (size_t)sc[k].ncat * 8));
        tally_naive<<<(N + 255) / 256, 256>>>(sc[k].d_cat, d_w, N, sc[k].ncat, d_tot);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_tot, d_tot, (size_t)sc[k].ncat * 8, cudaMemcpyDeviceToHost));
        naiveOk[k] = (memcmp(h_tot, sc[k].ref, (size_t)sc[k].ncat * 8) == 0);

        CHECK(cudaMemset(d_tot, 0, (size_t)sc[k].ncat * 8));
        tally_fast<<<fg[k], fb[k], fs[k]>>>(sc[k].d_cat, d_w, N, sc[k].ncat, d_tot);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_tot, d_tot, (size_t)sc[k].ncat * 8, cudaMemcpyDeviceToHost));
        correct[k] = (memcmp(h_tot, sc[k].ref, (size_t)sc[k].ncat * 8) == 0);

        int firstBad = -1;
        for (int c = 0; c < sc[k].ncat && firstBad < 0; ++c)
            if (h_tot[c] != sc[k].ref[c]) firstBad = c;
        printf("  %-20s naive %-4s  fast %-4s", sc[k].name,
               naiveOk[k] ? "PASS" : "FAIL", correct[k] ? "PASS" : "FAIL");
        if (!correct[k])
            printf("   first mismatch: cat %d, got %llu, want %llu",
                   firstBad, h_tot[firstBad], sc[k].ref[firstBad]);
        printf("\n");
    }
    printf("\n");

    // ---------------- results ----------------
    printf("=== timing (min of %d sweeps x %d iters, all configs back to back) ===\n",
           SWEEPS, ITERS);
    // Compulsory traffic per launch: cat[] and w[] are each 4 B x n, read
    // once. The output is ncat x 8 B and negligible. 432 GB/s is peak.
    const double BYTES = (double)N * 8.0;
    printf("  %-20s %10s %10s %9s %9s %8s %14s\n",
           "scenario", "naive ms", "fast ms", "speedup", "fast GB/s", "%peak",
           "your model");
    for (int k = 0; k < 4; ++k) {
        double model = predicted_global_atomics(N, sc[k].ncat, fg[k], fb[k]);
        const double gbs = BYTES / (sc[k].tFast * 1e-3) / 1e9;
        printf("  %-20s %10.4f %10.4f %8.2fx %9.1f %7.1f%% ", sc[k].name,
               sc[k].tNaive, sc[k].tFast, sc[k].tNaive / sc[k].tFast,
               gbs, 100.0 * gbs / 432.0);
        if (model >= 0.0) printf("%14.0f\n", model);
        else              printf("%14s\n", "(TODO 3)");
    }
    printf("  (naive global atomic instructions per launch: %d in every row)\n", N);
    printf("\n  grid/block/smem chosen: ");
    for (int k = 0; k < 4; ++k) printf("[%d,%d,%zuB] ", fg[k], fb[k], fs[k]);
    printf("\n\n");

    // ---------------- scoring ----------------
    const double r16   = sc[0].tNaive / sc[0].tFast;
    const double r4096 = sc[2].tNaive / sc[2].tFast;
    printf("=== TODO 4 prediction (within a factor of 2 counts as right) ===\n");
    const int p16ok   = (PRED_RATIO_16   <= 2.0 * r16   && PRED_RATIO_16   >= 0.5 * r16);
    const int p4096ok = (PRED_RATIO_4096 <= 2.0 * r4096 && PRED_RATIO_4096 >= 0.5 * r4096);
    printf("  ncat=16   clustered: predicted %6.2fx  measured %6.2fx  [%s]\n",
           PRED_RATIO_16, r16, p16ok ? "ok" : "no");
    printf("  ncat=4096 clustered: predicted %6.2fx  measured %6.2fx  [%s]\n\n",
           PRED_RATIO_4096, r4096, p4096ok ? "ok" : "no");

    int score = 0, maxScore = 9;
    for (int k = 0; k < 4; ++k) { score += correct[k]; }
    printf("=== scoring ===\n");
    printf("  [%s] correct in all four scenarios (%d/4)\n",
           (correct[0] && correct[1] && correct[2] && correct[3]) ? "PASS" : "FAIL",
           correct[0] + correct[1] + correct[2] + correct[3]);
    const int f16c = (r16 >= 4.0);
    const int f16s = (sc[1].tNaive / sc[1].tFast >= 2.0);
    printf("  [%s] ncat=16 clustered at least 4.00x  (got %.2fx)\n",
           f16c ? "PASS" : "FAIL", r16);
    printf("  [%s] ncat=16 shuffled  at least 2.00x  (got %.2fx)\n",
           f16s ? "PASS" : "FAIL", sc[1].tNaive / sc[1].tFast);
    score += f16c + f16s;
    const double r4096s = sc[3].tNaive / sc[3].tFast;
    const int noReg = (r4096 >= 2.50) && (r4096s >= 2.50);
    printf("  [%s] ncat=4096 at least 2.50x on BOTH orders (got %.2fx, %.2fx)\n",
           noReg ? "PASS" : "FAIL", r4096, r4096s);
    score += noReg;
    printf("  [%s] TODO 4 predictions (%d/2)\n", (p16ok && p4096ok) ? "PASS" : "FAIL",
           p16ok + p4096ok);
    score += p16ok + p4096ok;

    const int pass = (score == maxScore);
    printf("\n  score: %d/%d\n", score, maxScore);
    printf("OVERALL: %s\n", pass ? "PASS" : "FAIL");

    free(h_w); free(h_c16c); free(h_c16s); free(h_c4kc); free(h_c4ks);
    free(ref16c); free(ref16s); free(ref4kc); free(ref4ks); free(h_tot);
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_w)); CHECK(cudaFree(d16c)); CHECK(cudaFree(d16s));
    CHECK(cudaFree(d4kc)); CHECK(cudaFree(d4ks)); CHECK(cudaFree(d_tot));
    CHECK(cudaDeviceReset());
    return pass ? 0 : 1;
}
