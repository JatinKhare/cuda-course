// =====================================================================
// Module 7 / Exercise 3 : "Which of these is the disaster?"
//
// GOAL
//   Two kernels. One of them reads shared memory at an index that comes
//   out of a data array, which is the pattern everybody flags on sight.
//   The other reads its own private slice of a shared scratch buffer at
//   `scratch[tid][k]`, which is the pattern nobody flags at all.
//
//   Exactly one of those instincts is right. Decide which BEFORE you run
//   anything, and write down why.
//
//   -- Kernel A ------------------------------------------------------
//       v = sA[ (idx[tid] & 31) * SCALE + <row offset> ];
//     `idx` is an ordinary device array. Two instantiations are timed,
//     SCALE = 1 and SCALE = 32, each against different `idx` contents:
//       A1  SCALE=1,  idx = a seeded pseudo-random array
//       A2  SCALE=1,  idx = your adversarial array (TODO 4)
//       A3  SCALE=32, idx = warp-uniform values
//       A4  SCALE=32, idx = your adversarial array (TODO 4)
//
//   -- Kernel B ------------------------------------------------------
//     Every thread owns SCRATCH consecutive floats of a shared scratch
//     buffer and sweeps them:
//       B1  the layout as written below
//       B2  the layout your TODO 3 produces, same number of bytes
//
// PREDICT BEFORE YOU RUN, in writing
//   1. Rank all six configurations from fastest to slowest.
//   2. For A1 and A2: does the CONTENT of `idx` change the cost? Give a
//      reason that does not depend on what the content happens to be.
//   3. For B1: `scratch[tid][k]` is per-thread private storage with no
//      sharing between threads at all. Say out loud why that makes no
//      difference whatsoever to the bank arithmetic.
//
// WHAT THE PROGRAM CHECKS
//   - both kernels against a CPU reference (PASS/FAIL),
//   - your claim about the worst case kernel A can reach at SCALE = 1,
//     against a brute-force search over index contents,
//   - that your adversarial array really is adversarial at SCALE = 32,
//   - that your TODO 3 layout uses exactly as many bytes as B1,
//   - your six predicted ratios.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN:    .\exercise03.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <ctime>
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

#define WARP       32
#define BANKS      32
#define THREADS_A 256
#define THREADS_B 128
#define SCRATCH    32          // floats of private scratch per thread
#define SA_FLOATS 2048         // kernel A's shared array
#define REPS       96
#define BLOCKS    640

__constant__ float W[32];

// =============== the degree calculator (Exercise 1) ===================
static int degree_of(const int* elemIdx, int n)
{
    int  words[BANKS][WARP]; int nw[BANKS];
    for (int b = 0; b < BANKS; ++b) nw[b] = 0;
    for (int i = 0; i < n; ++i) {
        const int w = elemIdx[i];
        const int b = ((unsigned)w) % BANKS;
        bool seen = false;
        for (int k = 0; k < nw[b]; ++k) if (words[b][k] == w) { seen = true; break; }
        if (!seen) words[b][nw[b]++] = w;
    }
    int mx = 0;
    for (int b = 0; b < BANKS; ++b) if (nw[b] > mx) mx = nw[b];
    return mx;
}

// ---------------------------------------------------------------------
// TODO 2: Over EVERY possible set of values the 32 lanes of a warp could
//         read out of `idx`, what is the largest conflict degree kernel A
//         can be made to exhibit when SCALE == 1?
//
//         Do not guess from the shape of the expression. Work out what
//         the mask `& 31` does to the set of banks the warp can reach,
//         and what it simultaneously does to the set of WORDS.
//
//         The harness brute-forces the same question and prints what it
//         found. Set this to 0 and it will tell you to fill it in.
// ---------------------------------------------------------------------
static const int WORST_A1_DEGREE = 1;    // TODO 2 (solved)

// ---------------------------------------------------------------------
// TODO 4: Fill hIdxAdv[0..31] with the values that MAXIMISE the conflict
//         degree of kernel A when SCALE == 32. Lane L of every warp reads
//         hIdxAdv[L]. Values are masked with & 31 inside the kernel, so
//         only the low 5 bits matter.
//
//         The harness checks the degree your array achieves at SCALE=32
//         and, separately, at SCALE=1.
// ---------------------------------------------------------------------
static void make_adversarial(int* hIdxAdv /* 32 entries */)
{
    // TODO 4 (solved): any 32 values that are pairwise distinct mod 32.
    for (int lane = 0; lane < WARP; ++lane)
        hIdxAdv[lane] = lane;
}

// ---------------------------------------------------------------------
// TODO 3: Kernel B's scratch layout.
//
//         B1 stores thread `t`'s k-th scratch float at  t * SCRATCH + k.
//         Produce a different location for the same (t, k) such that:
//           (a) the buffer still holds exactly THREADS_B * SCRATCH
//               floats -- you may not add a single word of padding;
//           (b) for a fixed k, the 32 lanes of a warp reach 32 distinct
//               banks;
//           (c) the map (t, k) -> location is injective.
//         `nthreads` is THREADS_B.
// ---------------------------------------------------------------------
__host__ __device__ __forceinline__ int scratch_index_v2(int t, int k, int nthreads)
{
    // TODO 3 (solved): transpose the scratch buffer. Same bytes, and now
    // the warp's 32 lanes are 32 adjacent words, i.e. 32 distinct banks.
    return k * nthreads + t;
}

__host__ __device__ __forceinline__ int scratch_index_v1(int t, int k, int nthreads)
{
    (void)nthreads;
    return t * SCRATCH + k;
}

// ---------------------------------------------------------------------
// TODO 1: Your six predicted relative costs, from your written ranking.
//         A1..A4 are relative to A1. B1 and B2 are relative to B2.
// ---------------------------------------------------------------------
// TODO 1 (solved). The two that catch people are A3 (a warp-uniform
// index at stride 32 is a BROADCAST, not a 32-way conflict) and B1
// (per-thread private scratch is laid out at stride SCRATCH = 32 floats,
// which puts all 32 lanes in one bank).
// ---------------------------------------------------------------------
static int predictedDegree[6] = {
     1,   // A1  SCALE=1,  random idx      : low 5 bits pick the bank AND the word
     1,   // A2  SCALE=1,  adversarial idx : same argument, no data can break it
     1,   // A3  SCALE=32, warp-uniform    : one word, 32 readers -> broadcast
    32,   // A4  SCALE=32, adversarial     : 32 distinct words in bank 0
    32,   // B1  scratch[t * SCRATCH + k]  : stride 32 floats -> bank 0 for all
     1    // B2  your layout               : lanes are adjacent words
};

// ============================ kernels =================================

template <int SCALE>
__global__ void kernelA(const int* __restrict__ idx, float* __restrict__ out)
{
    __shared__ float sA[SA_FLOATS];
    const int t = threadIdx.x;
    for (int i = t; i < SA_FLOATS; i += blockDim.x) sA[i] = (float)((i * 3 + 1) & 63);
    __syncthreads();            // barrier; Module 9 makes this precise.

    const int base = (idx[t] & 31) * SCALE;
    float acc = 0.f;
    #pragma unroll 1
    for (int j = 0; j < REPS; ++j) {
        // + j*32 floats = + 128 B: a different word, the same bank.
        acc += sA[(base + j * 32     ) & (SA_FLOATS - 1)] * W[ j       & 31];
        acc += sA[(base + j * 32 + 32) & (SA_FLOATS - 1)] * W[(j +  8) & 31];
        acc += sA[(base + j * 32 + 64) & (SA_FLOATS - 1)] * W[(j + 16) & 31];
        acc += sA[(base + j * 32 + 96) & (SA_FLOATS - 1)] * W[(j + 24) & 31];
    }
    out[blockIdx.x * blockDim.x + t] = acc;
}

template <int VER>
__global__ void kernelB(float* __restrict__ out)
{
    __shared__ float scratch[THREADS_B * SCRATCH];
    const int t = threadIdx.x;

    #pragma unroll 1
    for (int k = 0; k < SCRATCH; ++k) {
        const int loc = (VER == 1) ? scratch_index_v1(t, k, THREADS_B)
                                   : scratch_index_v2(t, k, THREADS_B);
        scratch[loc] = (float)((t * 7 + k * 3) & 63);
    }
    __syncthreads();

    float acc = 0.f;
    #pragma unroll 1
    for (int j = 0; j < REPS; ++j)
        #pragma unroll 1
        for (int k = 0; k < SCRATCH; k += 4) {
            const int k0 = (k + j) & (SCRATCH - 1);
            const int k1 = (k + j + 1) & (SCRATCH - 1);
            const int k2 = (k + j + 2) & (SCRATCH - 1);
            const int k3 = (k + j + 3) & (SCRATCH - 1);
            const int l0 = (VER == 1) ? scratch_index_v1(t, k0, THREADS_B) : scratch_index_v2(t, k0, THREADS_B);
            const int l1 = (VER == 1) ? scratch_index_v1(t, k1, THREADS_B) : scratch_index_v2(t, k1, THREADS_B);
            const int l2 = (VER == 1) ? scratch_index_v1(t, k2, THREADS_B) : scratch_index_v2(t, k2, THREADS_B);
            const int l3 = (VER == 1) ? scratch_index_v1(t, k3, THREADS_B) : scratch_index_v2(t, k3, THREADS_B);
            acc += scratch[l0] * W[k0] + scratch[l1] * W[k1]
                 + scratch[l2] * W[k2] + scratch[l3] * W[k3];
        }
    out[blockIdx.x * blockDim.x + t] = acc;
}

// ========================== CPU reference =============================
static void refA_exact(const int* hIdx, const float* w, int scale, float* out)
{
    for (int t = 0; t < THREADS_A; ++t) {
        const int base = (hIdx[t] & 31) * scale;
        double acc = 0.0;
        for (int j = 0; j < REPS; ++j) {
            const int wi[4] = { j & 31, (j + 8) & 31, (j + 16) & 31, (j + 24) & 31 };
            for (int q = 0; q < 4; ++q) {
                const int i = (base + j * 32 + q * 32) & (SA_FLOATS - 1);
                acc += (double)(float)((i * 3 + 1) & 63) * (double)w[wi[q]];
            }
        }
        out[t] = (float)acc;
    }
}

static void refB(const float* w, float* out)
{
    for (int t = 0; t < THREADS_B; ++t) {
        double acc = 0.0;
        for (int j = 0; j < REPS; ++j)
            for (int k = 0; k < SCRATCH; k += 4)
                for (int q = 0; q < 4; ++q) {
                    const int kk = (k + j + q) & (SCRATCH - 1);
                    acc += (double)(float)((t * 7 + kk * 3) & 63) * (double)w[kk];
                }
        out[t] = (float)acc;
    }
}

// =============================== main =================================
static const char* CNAME[6] = {
    "A1 scale1 random", "A2 scale1 advers", "A3 scale32 unif",
    "A4 scale32 adver", "B1 scratch[t][k]", "B2 your layout  "
};
static const int CREF[6] = { 0, 0, 0, 0, 5, 5 };

int main()
{
    printf("Module 7 / Exercise 3 - one of these looks bad and is not\n\n");

    if (WORST_A1_DEGREE <= 0) { printf("Set TODO 2 first.\n"); return 0; }

    int hIdxAdv[WARP];
    make_adversarial(hIdxAdv);
    { int all0 = 1; for (int i = 1; i < WARP; ++i) if (hIdxAdv[i] != hIdxAdv[0]) all0 = 0;
      if (all0) { printf("Set TODO 4 first.\n"); return 0; } }

    if (scratch_index_v2(1, 0, THREADS_B) == scratch_index_v1(1, 0, THREADS_B) &&
        scratch_index_v2(1, 1, THREADS_B) == scratch_index_v1(1, 1, THREADS_B)) {
        printf("Set TODO 3 first.\n"); return 0;
    }
    for (int i = 0; i < 6; ++i) if (predictedDegree[i] <= 0) { printf("Set TODO 1 first.\n"); return 0; }

    // ---- TODO 2: brute-force the claim -------------------------------
    int foundMax = 0;
    {
        srand(12345);
        int probe[WARP];
        for (int trial = 0; trial < 200000; ++trial) {
            for (int l = 0; l < WARP; ++l) {
                int v;
                if      (trial == 0) v = l;             // all distinct
                else if (trial == 1) v = 0;             // all equal
                else if (trial == 2) v = (l & 1);       // two values
                else                 v = rand();
                probe[l] = (v & 31) * 1;                // SCALE == 1
            }
            const int d = degree_of(probe, WARP);
            if (d > foundMax) foundMax = d;
        }
    }
    const bool t2ok = (WORST_A1_DEGREE == foundMax);
    printf("TODO 2: you claimed the worst reachable degree at SCALE=1 is %d.\n",
           WORST_A1_DEGREE);
    printf("        200000 candidate index sets, including all-distinct and\n");
    printf("        all-equal, produced a maximum of %d. %s\n",
           foundMax, t2ok ? "ok" : "MISSED");

    // ---- TODO 4: is it actually adversarial? --------------------------
    int p32[WARP], p1[WARP];
    for (int l = 0; l < WARP; ++l) { p32[l] = (hIdxAdv[l] & 31) * 32; p1[l] = (hIdxAdv[l] & 31); }
    const int dAdv32 = degree_of(p32, WARP);
    const int dAdv1  = degree_of(p1,  WARP);
    const bool t4ok = (dAdv32 == 32);
    printf("TODO 4: your array gives degree %d at SCALE=32 (%s) and degree %d\n",
           dAdv32, t4ok ? "ok" : "not maximal", dAdv1);
    printf("        at SCALE=1. Those two numbers are the exercise.\n");

    // ---- TODO 3: structural ------------------------------------------
    int t3fail = 0;
    {
        static char used[THREADS_B * SCRATCH];
        for (int i = 0; i < THREADS_B * SCRATCH; ++i) used[i] = 0;
        for (int t = 0; t < THREADS_B && !t3fail; ++t)
            for (int k = 0; k < SCRATCH; ++k) {
                const int loc = scratch_index_v2(t, k, THREADS_B);
                if (loc < 0 || loc >= THREADS_B * SCRATCH) { t3fail = 1; break; }
                if (used[loc]) { t3fail = 2; break; }
                used[loc] = 1;
            }
        for (int k = 0; k < SCRATCH && !t3fail; ++k) {
            int idxs[WARP];
            for (int l = 0; l < WARP; ++l) idxs[l] = scratch_index_v2(l, k, THREADS_B);
            if (degree_of(idxs, WARP) != 1) t3fail = 3;
        }
    }
    const char* T3MSG[4] = { "ok", "out of range", "not injective", "still conflicted" };
    printf("TODO 3: layout test %s, size %d floats (B1 uses %d)\n\n",
           T3MSG[t3fail], THREADS_B * SCRATCH, THREADS_B * SCRATCH);

    // ---- data ----------------------------------------------------------
    float hW[32];
    for (int i = 0; i < 32; ++i) hW[i] = (float)((i % 7) - 3) * 0.25f;
    CHECK(cudaMemcpyToSymbol(W, hW, sizeof(hW)));

    int hRand[THREADS_A], hUnif[THREADS_A], hAdv[THREADS_A];
    srand(20240607);
    for (int t = 0; t < THREADS_A; ++t) {
        hRand[t] = rand();
        hUnif[t] = (t / WARP) * 7;              // warp-uniform
        hAdv[t]  = hIdxAdv[t & 31];
    }
    int *dRand = NULL, *dUnif = NULL, *dAdv = NULL;
    CHECK(cudaMalloc(&dRand, sizeof(hRand))); CHECK(cudaMemcpy(dRand, hRand, sizeof(hRand), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&dUnif, sizeof(hUnif))); CHECK(cudaMemcpy(dUnif, hUnif, sizeof(hUnif), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&dAdv,  sizeof(hAdv )));  CHECK(cudaMemcpy(dAdv,  hAdv,  sizeof(hAdv ), cudaMemcpyHostToDevice));

    const size_t NA = (size_t)BLOCKS * THREADS_A;
    const size_t NB = (size_t)BLOCKS * THREADS_B;
    float* dOut = NULL; CHECK(cudaMalloc(&dOut, (NA > NB ? NA : NB) * sizeof(float)));
    float* hOut = (float*)malloc((NA > NB ? NA : NB) * sizeof(float));

    // ---- time all six back to back --------------------------------------
    double best[6]; for (int i = 0; i < 6; ++i) best[i] = 1e30;
    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

#define FIRE(i) do {                                                            \
        if (i == 0) kernelA<1 ><<<BLOCKS,THREADS_A>>>(dRand, dOut);             \
        if (i == 1) kernelA<1 ><<<BLOCKS,THREADS_A>>>(dAdv,  dOut);             \
        if (i == 2) kernelA<32><<<BLOCKS,THREADS_A>>>(dUnif, dOut);             \
        if (i == 3) kernelA<32><<<BLOCKS,THREADS_A>>>(dAdv,  dOut);             \
        if (i == 4) kernelB<1 ><<<BLOCKS,THREADS_B>>>(dOut);                    \
        if (i == 5) kernelB<2 ><<<BLOCKS,THREADS_B>>>(dOut);                    \
    } while (0)

    clock_t w0 = clock();
    while ((double)(clock() - w0) / CLOCKS_PER_SEC < 4.0) {
        for (int i = 0; i < 6; ++i) FIRE(i);
        CHECK(cudaDeviceSynchronize());
    }
    for (int sweep = 0; sweep < 4; ++sweep)
        for (int q = 0; q < 6; ++q) {
            const int i = (q + sweep) % 6;       // rotate order: no config is
            // always measured right after the clock dip that follows a sync.
            FIRE(i); CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(e0));
            for (int r = 0; r < 20; ++r) FIRE(i);
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= 20.f; if (ms < best[i]) best[i] = ms;
        }
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

    // ---- validate, second pass -------------------------------------------
    int bad = 0;
    {
        float ref[THREADS_A];
        const int* srcs[4] = { hRand, hAdv, hUnif, hAdv };
        const int  scal[4] = { 1, 1, 32, 32 };
        for (int i = 0; i < 4; ++i) {
            CHECK(cudaMemset(dOut, 0, NA * sizeof(float)));
            FIRE(i); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(hOut, dOut, NA * sizeof(float), cudaMemcpyDeviceToHost));
            refA_exact(srcs[i], hW, scal[i], ref);
            for (size_t n = 0; n < NA; ++n)
                if (fabsf(hOut[n] - ref[n % THREADS_A]) >
                    1e-5f * fmaxf(1.0f, fabsf(ref[n % THREADS_A]))) { bad++; break; }
        }
        float refb[THREADS_B]; refB(hW, refb);
        for (int i = 4; i < 6; ++i) {
            CHECK(cudaMemset(dOut, 0, NB * sizeof(float)));
            FIRE(i); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(hOut, dOut, NB * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t n = 0; n < NB; ++n)
                if (fabsf(hOut[n] - refb[n % THREADS_B]) >
                    1e-5f * fmaxf(1.0f, fabsf(refb[n % THREADS_B]))) { bad++; break; }
        }
    }
    printf("Numerical check vs CPU reference: %s\n\n", bad ? "FAIL" : "PASS");

    // ---- score --------------------------------------------------------
    // The harness derives the true degrees from the same index
    // expressions the kernels use.
    int trueDeg[6];
    {
        int b[WARP];
        for (int l = 0; l < WARP; ++l) b[l] = (hRand[l] & 31);
        trueDeg[0] = degree_of(b, WARP);
        for (int l = 0; l < WARP; ++l) b[l] = (hAdv[l] & 31);
        trueDeg[1] = degree_of(b, WARP);
        for (int l = 0; l < WARP; ++l) b[l] = (hUnif[l] & 31) * 32;
        trueDeg[2] = degree_of(b, WARP);
        for (int l = 0; l < WARP; ++l) b[l] = (hAdv[l] & 31) * 32;
        trueDeg[3] = degree_of(b, WARP);
        for (int l = 0; l < WARP; ++l) b[l] = scratch_index_v1(l, 0, THREADS_B);
        trueDeg[4] = degree_of(b, WARP);
        for (int l = 0; l < WARP; ++l) b[l] = scratch_index_v2(l, 0, THREADS_B);
        trueDeg[5] = degree_of(b, WARP);
    }

    printf("%-18s %7s %7s %10s %10s   %s\n",
           "config", "yours", "actual", "ms", "measured", "verdict");
    printf("---------------------------------------------------------------------------\n");
    int good = 0;
    for (int i = 0; i < 6; ++i) {
        const bool ok = (predictedDegree[i] == trueDeg[i]);
        if (ok) good++;
        printf("%-18s %7d %7d %10.4f %9.2fx   %s\n",
               CNAME[i], predictedDegree[i], trueDeg[i],
               best[i], best[i] / best[CREF[i]], ok ? "ok" : "MISSED");
    }
    printf("\nDegrees correct: %d/6\n", good);
    printf("\nA degree of D costs at most D/2 times a conflict-free access,\n");
    printf("so the two 32-way rows could have been 16x. A4 measured %.1fx,\n",
           best[3] / best[0]);
    printf("B1 measured %.1fx. The rest of each loop -- the FFMAs, the loop\n",
           best[4] / best[5]);
    printf("control, the store -- does not slow down when the banks do.\n");

    const bool pass = !bad && t2ok && t4ok && !t3fail && good == 6;
    printf("\nTODO 1: %d/6   TODO 2: %s   TODO 3: %s   TODO 4: %s   numerics: %s\n",
           good, t2ok ? "ok" : "MISSED", t3fail ? "FAIL" : "ok",
           t4ok ? "ok" : "MISSED", bad ? "FAIL" : "PASS");
    printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");

    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(dRand)); CHECK(cudaFree(dUnif)); CHECK(cudaFree(dAdv)); CHECK(cudaFree(dOut));
    free(hOut);
    CHECK(cudaDeviceReset());
    return 0;
}
