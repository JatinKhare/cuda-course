// =====================================================================
// Module 7 / Exercise 1 : "Work out the bank map before you run it"
//
// GOAL
//   Turn shared-memory conflict analysis into arithmetic you perform,
//   the way Module 5 turned coalescing into sector counting. Before you
//   touch this file, fill the table below ON PAPER for warp 0 (lanes
//   0..31). Only then write the code that reproduces it.
//
//   Shared memory on this GPU: 32 banks, 4 B wide, striped so that the
//   4 B word at byte address A lives in bank (A / 4) % 32. A bank can
//   return one 4 B word per cycle per request. Assume the arrays below
//   start at bank 0.
//
//     #  expression      | lane->bank for lanes 0..31 | max multiplicity
//                        | (write the first 8)        | of DISTINCT words
//                        |                            | in one bank
//     --+----------------+----------------------------+------------------
//     0 | s [tid]        |                            |
//     1 | s [2*tid]      |                            |
//     2 | s [8*tid]      |                            |
//     3 | s [32*tid]     |                            |
//     4 | s [tid/2]      |                            |
//     5 | s [31-tid]     |                            |
//     6 | dd[tid]        |                            |
//     7 | dd[2*tid]      |                            |
//
//   s  is `__shared__ float  s [...]`   (4 B elements)
//   dd is `__shared__ double dd[...]`   (8 B elements)
//
//   Two lanes that read the SAME 4 B word are a broadcast: the bank
//   drives one word onto the crossbar and every requesting lane takes a
//   copy, in one cycle. Only DISTINCT words in the same bank serialize.
//   This is the same broadcast-vs-serialize split Module 4 measured for
//   constant memory, in a different memory.
//
//   Then, from your multiplicities, predict the RELATIVE cost of each
//   pattern. Rows 0-5 are relative to row 0; rows 6-7 are relative to
//   row 6. Write the eight numbers down before you edit anything. You
//   will type them into TODO 3 and the harness will score them.
//
// WHAT THE PROGRAM CHECKS
//   - your bank function has the structural properties a 32 x 4 B banked
//     memory must have,
//   - your conflict-degree function is internally consistent with the
//     set of distinct words the warp actually touches,
//   - the kernel results against a CPU reference (PASS/FAIL),
//   - your eight predicted ratios against the measured ratios (30% band).
//
//   At least one of your eight predictions will be wrong. Finding out
//   which one, and why, is the exercise.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN:    .\exercise01.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
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

static const int BANKS  = 32;
static const int BANK_W = 4;      // bytes per bank
static const int WARP   = 32;

// ================= the eight patterns ================================
enum { N_PAT = 8 };
static const char* PAT_NAME[N_PAT] = {
    "s[tid]", "s[2*tid]", "s[8*tid]", "s[32*tid]",
    "s[tid/2]", "s[31-tid]", "dd[tid]", "dd[2*tid]"
};
static const int PAT_ELEM_BYTES[N_PAT] = { 4, 4, 4, 4, 4, 4, 8, 8 };
static const int PAT_REF[N_PAT]        = { 0, 0, 0, 0, 0, 0, 6, 6 };

__host__ __device__ static inline int pat_index(int p, int lane)
{
    switch (p) {
        case 0: return lane;
        case 1: return 2 * lane;
        case 2: return 8 * lane;
        case 3: return 32 * lane;
        case 4: return lane / 2;
        case 5: return 31 - lane;
        case 6: return lane;
        default: return 2 * lane;
    }
}

// ---------------------------------------------------------------------
// TODO 1: Map a byte address to the id of the bank that holds it.
//         Two 4 B words live in the same bank exactly when this function
//         returns the same value for both. Return a value in [0, 32).
//         Return -1 only for the unimplemented stub.
// ---------------------------------------------------------------------
static int bank_of(uintptr_t byte_addr)
{
    return (int)((byte_addr / BANK_W) % BANKS);          // TODO 1 (solved)
}

// ---------------------------------------------------------------------
// TODO 2: Given `n` lanes whose element indices are elemIdx[0..n-1] and
//         an element size of `elemBytes` bytes, return the number of
//         cycles the bank array needs to satisfy this group: the largest
//         number of DISTINCT 4 B words that any single bank must supply.
//
//         Remember:
//           - an element of `elemBytes` bytes occupies elemBytes/4
//             consecutive words, each in its own bank;
//           - lanes asking for the same word do not add a cycle.
//
//         Return 0 only for the unimplemented stub.
// ---------------------------------------------------------------------
static int degree_in_group(const int* elemIdx, int n, int elemBytes)
{
    // TODO 2 (solved): distinct words per bank, then the maximum.
    if (n <= 0) return 0;
    long long words[BANKS][WARP * 4];
    int       nw[BANKS];
    for (int b = 0; b < BANKS; ++b) nw[b] = 0;

    for (int i = 0; i < n; ++i) {
        const uintptr_t a0 = (uintptr_t)elemIdx[i] * (uintptr_t)elemBytes;
        for (int off = 0; off < elemBytes; off += BANK_W) {
            const uintptr_t a = a0 + off;
            const int       b = bank_of(a);
            const long long w = (long long)(a / BANK_W);
            bool seen = false;
            for (int k = 0; k < nw[b]; ++k) if (words[b][k] == w) { seen = true; break; }
            if (!seen) words[b][nw[b]++] = w;
        }
    }
    int mx = 0;
    for (int b = 0; b < BANKS; ++b) if (nw[b] > mx) mx = nw[b];
    return mx;
}

// ---------------------------------------------------------------------
// TODO 4: A bank array 32 x 4 B wide can deliver at most 128 B per
//         cycle. A warp of 32 lanes each asking for `elemBytes` bytes
//         asks for 32*elemBytes bytes. When that exceeds 128 B the
//         hardware cannot serve the warp as one group no matter how the
//         addresses fall; it splits the warp into equal, CONTIGUOUS lane
//         groups small enough to fit, and resolves conflicts inside each
//         group independently.
//
//         Return the total number of cycles for pattern `p`: split the
//         warp accordingly, call degree_in_group on each group, and sum.
//         For 4 B elements there is exactly one group of 32 lanes.
//
//         Return 0 only for the unimplemented stub.
// ---------------------------------------------------------------------
static int cycles_for_pattern(int p)
{
    // TODO 4 (solved): split into contiguous groups small enough that the
    // group's total request fits in the 128 B the bank array delivers per
    // cycle, then sum the per-group cost.
    const int eb     = PAT_ELEM_BYTES[p];
    int       phases = eb / BANK_W; if (phases < 1) phases = 1;
    const int lanes  = WARP / phases;

    int total = 0;
    for (int ph = 0; ph < phases; ++ph) {
        int idx[WARP];
        for (int i = 0; i < lanes; ++i) idx[i] = pat_index(p, ph * lanes + i);
        total += degree_in_group(idx, lanes, eb);
    }
    return total;
}

// ---------------------------------------------------------------------
// TODO 3: Your eight predicted RELATIVE costs, from your paper table.
//         Index i is the measured time of pattern i divided by the
//         measured time of pattern PAT_REF[i]. So predicted[0] and
//         predicted[6] are 1.0 by definition. Leave the array at zero
//         and the harness will tell you to fill it in.
// ---------------------------------------------------------------------
// TODO 3 (solved). These are cycles(p) / cycles(ref), with every cycle
// count floored at 2, because the measured floor for ANY 32-lane shared
// access on sm_89 is two pipeline cycles. See the solution notes.
static double predicted[N_PAT] = {
    1.00,   // 0  s[tid]     : 1 cycle  -> floored to 2 -> 2/2
    1.00,   // 1  s[2*tid]   : 2 cycles -> 2/2. Two-way conflicts are FREE.
    4.00,   // 2  s[8*tid]   : 8 cycles -> 8/2
   16.00,   // 3  s[32*tid]  : 32 cycles -> 32/2
    1.00,   // 4  s[tid/2]   : broadcast pairs, 1 cycle -> floored to 2
    1.00,   // 5  s[31-tid]  : a permutation of s[tid]; order is invisible
    1.00,   // 6  dd[tid]    : 2 phases x 1 cycle = 2, the double reference
    2.00    // 7  dd[2*tid]  : 2 phases x 2 cycles = 4 -> 4/2
};

// ========================== the kernels ==============================
//
// Each loop iteration advances the index by a multiple of 128 B, which
// changes the word but NOT the bank. So every iteration has exactly the
// conflict degree of the pattern, and this is a clean throughput test.
// Four accumulators keep the arithmetic dependency chain off the
// critical path. All values are small integers held exactly in fp32 and
// fp64, so the CPU reference can be compared without tolerance slack.

#define ITERS   512
#define BLOCKS  320
#define THREADS 256
#define TNF     4096      // floats  (16 KB)
#define TND     2048      // doubles (16 KB)

template <int P>
__global__ void kF(float* out)
{
    __shared__ float s[TNF];
    const int t = threadIdx.x, lane = t & 31;
    for (int i = t; i < TNF; i += blockDim.x) s[i] = (float)(i & 255);
    __syncthreads();                       // barrier; Module 9 makes this precise.

    const int idx = pat_index(P, lane) & (TNF - 1);
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    #pragma unroll 4
    for (int it = 0; it < ITERS; ++it) {
        const int o = (it * 32) & (TNF - 1);
        a0 += s[(idx + o      ) & (TNF - 1)];
        a1 += s[(idx + o +  32) & (TNF - 1)];
        a2 += s[(idx + o +  64) & (TNF - 1)];
        a3 += s[(idx + o +  96) & (TNF - 1)];
    }
    out[blockIdx.x * blockDim.x + t] = a0 + a1 + a2 + a3;
}

template <int P>
__global__ void kD(float* out)
{
    __shared__ double dd[TND];
    const int t = threadIdx.x, lane = t & 31;
    for (int i = t; i < TND; i += blockDim.x) dd[i] = (double)(i & 255);
    __syncthreads();

    // The accumulation is an integer XOR of the loaded bit patterns, not
    // a double add. Ada runs FP64 at 1/64 the FP32 rate, so `a += dd[i]`
    // would measure the FP64 pipe and hide the banks completely. The
    // loads are still genuine 8 B shared loads, which is what we time.
    const int idx = pat_index(P, lane) & (TND - 1);
    long long a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    #pragma unroll 4
    for (int it = 0; it < ITERS; ++it) {
        const int o = (it * 16) & (TND - 1);   // 16 doubles = 128 B
        a0 ^= __double_as_longlong(dd[(idx + o     ) & (TND - 1)]);
        a1 ^= __double_as_longlong(dd[(idx + o + 16) & (TND - 1)]);
        a2 ^= __double_as_longlong(dd[(idx + o + 32) & (TND - 1)]);
        a3 ^= __double_as_longlong(dd[(idx + o + 48) & (TND - 1)]);
    }
    out[blockIdx.x * blockDim.x + t] = (float)(unsigned)((a0 ^ a1 ^ a2 ^ a3) & 0xFFFFFFLL);
}

typedef void (*Launch)(float*);
static void L0(float* o){ kF<0><<<BLOCKS,THREADS>>>(o); }
static void L1(float* o){ kF<1><<<BLOCKS,THREADS>>>(o); }
static void L2(float* o){ kF<2><<<BLOCKS,THREADS>>>(o); }
static void L3(float* o){ kF<3><<<BLOCKS,THREADS>>>(o); }
static void L4(float* o){ kF<4><<<BLOCKS,THREADS>>>(o); }
static void L5(float* o){ kF<5><<<BLOCKS,THREADS>>>(o); }
static void L6(float* o){ kD<6><<<BLOCKS,THREADS>>>(o); }
static void L7(float* o){ kD<7><<<BLOCKS,THREADS>>>(o); }

// ======================= CPU reference ===============================
// A thread's result depends only on its lane, so 32 values per pattern
// describe every one of the BLOCKS*THREADS outputs.
static long long bits_of(double v) { long long b; memcpy(&b, &v, sizeof(b)); return b; }

static void cpu_reference(int p, double* ref32)
{
    const bool wide = (PAT_ELEM_BYTES[p] == 8);
    const int  TN   = wide ? TND : TNF;
    const int  step = wide ? 16  : 32;
    for (int lane = 0; lane < WARP; ++lane) {
        const int idx = pat_index(p, lane) & (TN - 1);
        double    acc  = 0.0;
        long long bacc = 0;
        for (int it = 0; it < ITERS; ++it) {
            const int o = (it * step) & (TN - 1);
            for (int k = 0; k < 4; ++k) {
                const int j = (idx + o + k * step) & (TN - 1);
                if (wide) bacc ^= bits_of((double)(j & 255));
                else      acc  += (double)(j & 255);
            }
        }
        ref32[lane] = wide ? (double)(unsigned)(bacc & 0xFFFFFFLL) : acc;
    }
}

// ============================= main ==================================
int main()
{
    printf("Module 7 / Exercise 1 - bank maps by hand, then measured\n\n");

    if (bank_of(0) < 0) { printf("Set TODO 1 first.\n"); return 0; }

    // ---- structural tests on TODO 1 ---------------------------------
    int fail1 = 0;
    {
        int seen[BANKS] = {0};
        for (int w = 0; w < BANKS; ++w) {
            int b = bank_of((uintptr_t)w * BANK_W);
            if (b < 0 || b >= BANKS) { fail1 = 1; break; }
            seen[b]++;
        }
        for (int b = 0; b < BANKS && !fail1; ++b) if (seen[b] != 1) fail1 = 1;   // a bijection
        if (!fail1)
            for (int w = 0; w < 4 * BANKS; ++w)                                  // period 128 B
                if (bank_of((uintptr_t)w * BANK_W) !=
                    bank_of((uintptr_t)(w + BANKS) * BANK_W)) { fail1 = 1; break; }
        if (!fail1)
            for (int a = 0; a < 16; ++a)                                         // sub-word bytes
                if (bank_of((uintptr_t)a) != bank_of((uintptr_t)(a / 4) * 4)) { fail1 = 1; break; }
    }
    printf("TODO 1 structural test : %s\n", fail1 ? "FAIL" : "ok");

    if (degree_in_group(NULL, 0, 4) == 0) {
        int probe[WARP]; for (int i = 0; i < WARP; ++i) probe[i] = i;
        if (degree_in_group(probe, WARP, 4) == 0) { printf("Set TODO 2 first.\n"); return 0; }
    }

    // ---- TODO 2 consistency ------------------------------------------
    // Sum over banks of distinct words must equal the total number of
    // distinct words the warp touches, and the max over banks can be no
    // smaller than that total divided by 32. The harness computes the
    // total without reference to banking.
    int fail2 = 0;
    for (int p = 0; p < N_PAT; ++p) {
        int idx[WARP];
        for (int lane = 0; lane < WARP; ++lane) idx[lane] = pat_index(p, lane);
        const int eb = PAT_ELEM_BYTES[p];
        long long words[WARP * 4]; int nwords = 0;
        for (int lane = 0; lane < WARP; ++lane)
            for (int off = 0; off < eb; off += BANK_W) {
                long long w = ((long long)idx[lane] * eb + off) / BANK_W;
                bool seen = false;
                for (int k = 0; k < nwords; ++k) if (words[k] == w) { seen = true; break; }
                if (!seen) words[nwords++] = w;
            }
        int d = degree_in_group(idx, WARP, eb);
        int lower = (nwords + BANKS - 1) / BANKS;
        if (d < lower || d > nwords) fail2 = 1;
    }
    printf("TODO 2 consistency test: %s\n", fail2 ? "FAIL" : "ok");

    if (cycles_for_pattern(0) == 0) { printf("Set TODO 4 first.\n"); return 0; }

    int fail3 = 0;
    for (int i = 0; i < N_PAT; ++i) if (predicted[i] <= 0.0) fail3 = 1;
    if (fail3) { printf("Set TODO 3 first (all eight entries must be > 0).\n"); return 0; }

    // ---- the table your arithmetic produced --------------------------
    int cyc[N_PAT];
    printf("\n%-11s %6s %8s %9s\n", "pattern", "bytes", "cycles", "predicted");
    printf("-------------------------------------------\n");
    for (int p = 0; p < N_PAT; ++p) {
        cyc[p] = cycles_for_pattern(p);
        printf("%-11s %6d %8d %8.2fx\n", PAT_NAME[p], PAT_ELEM_BYTES[p], cyc[p], predicted[p]);
    }

    // ---- measure -----------------------------------------------------
    const size_t NOUT = (size_t)BLOCKS * THREADS;
    float* dOut = NULL; float* hOut = (float*)malloc(NOUT * sizeof(float));
    CHECK(cudaMalloc(&dOut, NOUT * sizeof(float)));

    Launch L[N_PAT] = { L0, L1, L2, L3, L4, L5, L6, L7 };
    double best[N_PAT];
    for (int p = 0; p < N_PAT; ++p) best[p] = 1e30;

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    // Duration-based clock warm-up. This laptop GPU idles near 0.5 GHz.
    clock_t w0 = clock();
    while ((double)(clock() - w0) / CLOCKS_PER_SEC < 4.0) {
        for (int p = 0; p < N_PAT; ++p) L[p](dOut);
        CHECK(cudaDeviceSynchronize());
    }

    // All configurations timed back to back; min over 4 sweeps.
    for (int sweep = 0; sweep < 4; ++sweep) {
        for (int q = 0; q < N_PAT; ++q) {
            const int p = (q + sweep) % N_PAT;   // rotate order: no config is
            // always measured right after the clock dip that follows a sync.
            L[p](dOut); CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(e0));
            for (int i = 0; i < 20; ++i) L[p](dOut);
            CHECK(cudaEventRecord(e1));
            CHECK(cudaEventSynchronize(e1));
            float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= 20.f;
            if (ms < best[p]) best[p] = ms;
        }
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // ---- validate, in a second pass ----------------------------------
    int numFail = 0;
    for (int p = 0; p < N_PAT; ++p) {
        L[p](dOut);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hOut, dOut, NOUT * sizeof(float), cudaMemcpyDeviceToHost));
        double ref[WARP]; cpu_reference(p, ref);
        for (size_t i = 0; i < NOUT; ++i) {
            double r = ref[i & 31];
            if (fabs((double)hOut[i] - r) > 1e-5 * fmax(1.0, fabs(r))) { numFail++; break; }
        }
    }
    printf("\nNumerical check vs CPU reference: %s\n", numFail ? "FAIL" : "PASS");

    // ---- score -------------------------------------------------------
    printf("\n%-11s %8s %10s %11s %11s   %s\n",
           "pattern", "cycles", "ms", "measured", "predicted", "verdict");
    printf("--------------------------------------------------------------------------\n");
    int good = 0;
    for (int p = 0; p < N_PAT; ++p) {
        double meas = best[p] / best[PAT_REF[p]];
        double err  = fabs(meas - predicted[p]) / meas;
        bool ok = (err <= 0.30);
        if (ok) good++;
        printf("%-11s %8d %10.4f %10.2fx %10.2fx   %s\n",
               PAT_NAME[p], cyc[p], best[p], meas, predicted[p], ok ? "ok" : "MISSED");
    }
    printf("\nPredictions within 30%%: %d/%d\n", good, N_PAT);
    if (good < N_PAT) {
        printf("\nThe MISSED rows are the exercise. The hardware is not wrong.\n");
        printf("For each one, ask: how many cycles did the bank array actually\n");
        printf("need, and what did it charge you for the cheapest possible\n");
        printf("access of the same width? The harness will not tell you.\n");
    }

    bool pass = !fail1 && !fail2 && !numFail && good >= 6;
    printf("\nTODO 1: %s   TODO 2: %s   TODO 4: %s   numerics: %s   predictions: %d/8\n",
           fail1 ? "FAIL" : "ok", fail2 ? "FAIL" : "ok",
           (cyc[0] > 0) ? "ok" : "FAIL", numFail ? "FAIL" : "PASS", good);
    printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");

    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(dOut)); free(hOut);
    CHECK(cudaDeviceReset());
    return 0;
}
