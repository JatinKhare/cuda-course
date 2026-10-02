// =============================================================================
// Module 19 / Exercise 1 — SOLUTION — occupancy by hand vs the API.
//
// GOAL : Implement the placement arithmetic the hardware performs when it
//        decides how many blocks of your kernel fit on one SM, and make it
//        agree with cudaOccupancyMaxActiveBlocksPerMultiprocessor on every
//        kernel in the table -- not on most of them.
//
//        Three scoring groups:
//          (1) 18 real kernels, your blocks/SM vs the CUDA occupancy API;
//          (2) your BINDING-LIMITER answer on those same 18 kernels;
//          (3) 10 hypothetical (registers, shared bytes, threads) triples,
//              scored against FNV-1a hashes so the answers are not in this file;
//          (4) a round-trip consistency test on TODO 5.
//
//        A model that is right on "most" kernels is the normal outcome of
//        guessing the arithmetic. Several rows here exist specifically to
//        separate a model that is nearly right from one that is right.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN  : exercise01.exe
// See also: nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise01.exe exercise01.cu
//           to see the register and shared-memory numbers the program reads
//           back at runtime with cudaFuncGetAttributes.
//
// What you may assume you already know:
//   Module 1  — the SM has 4 processing blocks; each owns a 16384-register
//               slice of the 65536-register file and one warp scheduler; a
//               block is placed whole, never migrates, and is gated on
//               thread/warp slots, block slots, registers and shared memory.
//   Module 6  — per-block shared memory carries a 1024 B driver reserve and is
//               allocated in units of 128 B; 102400 B per SM.
//   Module 8  — a 100-thread block is FOUR warps, and the 28 threads that do
//               not exist still occupy their slots.
//   Module 18 — registers are allocated in granules of 8 per thread, per warp.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                       \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        printf("CUDA error %s (%s) at %s:%d\n", cudaGetErrorName(_e),          \
               cudaGetErrorString(_e), __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// ---- sm_89 placement budget. These are the only constants you need. ---------
#define REGS_PER_SM       65536
#define REG_SLICES        4
#define REGS_PER_SLICE    16384
#define REG_GRAN          8
#define WARPS_PER_SM      48
#define MAX_BLOCKS_PER_SM 24
#define SMEM_PER_SM       102400
#define SMEM_RESERVE      1024
#define SMEM_GRAN         128

static int ceilDiv(int a, int b) { return (a + b - 1) / b; }
static int roundUp(int a, int g) { return ceilDiv(a, g) * g; }

// Limiter codes, used by TODO 3 and by the scoring.
#define LIM_REGS   0
#define LIM_SMEM   1
#define LIM_WARPS  2
#define LIM_BLOCKS 3

// =============================================================================
// TODO 1: how many blocks fit on one SM if ONLY the register file mattered?
//
//   `regsPerThread` is what `cudaFuncGetAttributes().numRegs` reports and what
//   `-Xptxas -v` prints as "Used N registers".
//
//   Two facts have to be in this expression, and a model with only one of them
//   is right on most kernels and wrong on some. One of the two is in Module 18;
//   the other is in Module 1's description of what an SM is made of. If you
//   write a single division by 65536 you have the second one wrong.
//
//   Return the number of whole blocks, which may legitimately exceed 24 here --
//   TODO 3 takes the minimum.
// =============================================================================
static int blocksByRegisters(int regsPerThread, int threads)
{
    if (regsPerThread <= 0 || threads <= 0) return 0;
    const int warpsPerBlock = ceilDiv(threads, 32);
    // granule: 8 registers per thread, allocated per warp -> 256 regs/warp
    const int regsPerWarp   = roundUp(regsPerThread, REG_GRAN) * 32;
    // a warp lives entirely inside ONE of the four 16384-register slices
    const int warpsPerSlice = REGS_PER_SLICE / regsPerWarp;
    return (REG_SLICES * warpsPerSlice) / warpsPerBlock;
}

// =============================================================================
// TODO 2: the other two per-resource limits.
//
//   (a) blocksBySharedMemory: `smemBytes` is the per-block request, static plus
//       dynamic. Module 6 measured two quantisations that both have to be here;
//       without them 16384 B gives 6 blocks instead of the 5 the hardware
//       allows.
//
//   (b) blocksByWarpSlots: the SM has 48 warp slots. Careful -- this is NOT
//       1536 / threads. Module 8 explains why those two differ, and one row of
//       the table below is chosen so that they differ.
// =============================================================================
static int blocksBySharedMemory(int smemBytes)
{
    if (smemBytes < 0) return 0;
    return SMEM_PER_SM / roundUp(smemBytes + SMEM_RESERVE, SMEM_GRAN);
}

static int blocksByWarpSlots(int threads)
{
    if (threads <= 0) return 0;
    return WARPS_PER_SM / ceilDiv(threads, 32);
}

// =============================================================================
// TODO 3: combine. Fill `cand[4]` in the order
//            { registers, shared, warp slots, block slots }
//         return the number of blocks the hardware will place, and write the
//         index of the BINDING resource into *limiter. If several resources tie
//         for the minimum, report the FIRST of them in the order above.
// =============================================================================
static int blocksPerSM(int regsPerThread, int smemBytes, int threads,
                       int *limiter, int cand[4])
{
    cand[0] = blocksByRegisters(regsPerThread, threads);
    cand[1] = blocksBySharedMemory(smemBytes);
    cand[2] = blocksByWarpSlots(threads);
    cand[3] = MAX_BLOCKS_PER_SM;
    int best = cand[0], which = LIM_REGS;
    for (int i = 1; i < 4; ++i) if (cand[i] < best) { best = cand[i]; which = i; }
    *limiter = which;
    return best;
}

// =============================================================================
// TODO 4: occupancy, as a percentage.
//
//   Occupancy is resident WARPS over the maximum resident warps, not blocks
//   over blocks and not threads over threads. Two rows of the table check that
//   you did not write `blocks * threads / 1536`: one uses 100-thread blocks and
//   one uses 1024-thread blocks, and those two expressions disagree on both.
// =============================================================================
static double occupancyPercent(int blocks, int threads)
{
    if (blocks <= 0 || threads <= 0) return 0.0;
    int warps = blocks * ceilDiv(threads, 32);
    if (warps > WARPS_PER_SM) warps = WARPS_PER_SM;
    return 100.0 * (double)warps / (double)WARPS_PER_SM;
}

// =============================================================================
// TODO 5 (DESIGN): invert the arithmetic.
//
//   You are about to add work to a kernel and you want to know how much
//   register pressure you can afford. Return the LARGEST per-thread register
//   count at which a block of `threads` threads still achieves `targetBlocks`
//   blocks per SM, considering registers only; return 0 if `targetBlocks` is
//   unreachable at that block size even at the minimum useful register count.
//
//   This is the number you actually use when tuning -- "I have 4 blocks/SM and
//   I would like a fifth; what is my budget?" -- and it is not obtained by
//   dividing anything by anything, because of the two quantisations in TODO 1.
//   Think about what the answer must look like before you write the loop.
//
//   The harness scores this by round-trip: for each (targetBlocks, threads) it
//   checks that your answer R satisfies
//        blocksByRegisters(R, threads)     >= targetBlocks
//        blocksByRegisters(R + 1, threads) <  targetBlocks
//   (a correct answer is the largest R with the first property, so R+1 must
//   fail), and that R is in [1, 255].
// =============================================================================
static int maxRegistersFor(int targetBlocks, int threads)
{
    if (targetBlocks <= 0 || threads <= 0) return 0;
    // The function R -> blocksByRegisters(R, threads) is non-increasing in R
    // and is a staircase (two quantisations), so just walk down from 255 and
    // take the first R that still reaches the target. 255 steps, done once.
    for (int R = 255; R >= 1; --R)
        if (blocksByRegisters(R, threads) >= targetBlocks) return R;
    return 0;
}

// =============================================================================
// The kernel family. `W` private state values live across the whole time loop,
// so W is a handle on the register count; `SFLOATS` of static shared memory is
// a handle on the shared footprint. These are the shape of any sequential-state
// kernel (IIR filter bank, RNN step, per-thread integrator): one sample in, W
// recursive stages, state carried in registers.
// =============================================================================
#define WMAX 160
__constant__ float cA[WMAX];
__constant__ float cB[WMAX];

template<int W, int T, int SFLOATS>
__global__ __launch_bounds__(T) void stateKernel(const float * __restrict__ x,
                                                 float * __restrict__ y,
                                                 int nsig, int L)
{
    __shared__ float sh[SFLOATS > 0 ? SFLOATS : 1];
    const int sig = blockIdx.x * T + threadIdx.x;
    if (SFLOATS > 0) {
        for (int k = threadIdx.x; k < SFLOATS; k += T) sh[k] = (float)k;
        __syncthreads();
    }
    if (sig >= nsig) return;
    float st[W];
    #pragma unroll
    for (int i = 0; i < W; ++i) st[i] = 0.0f;
    for (int t = 0; t < L; ++t) {
        float v = x[(size_t)t * nsig + sig];
        #pragma unroll
        for (int i = 0; i < W; ++i) {
            v     = fmaf(cA[i], v, st[i]);
            st[i] = fmaf(cB[i], v, 0.5f * st[i]);
        }
        if (SFLOATS > 0) v += sh[sig % (SFLOATS > 0 ? SFLOATS : 1)] * 0.0f;
        y[(size_t)t * nsig + sig] = v;
    }
}

// ------------------------------------------------------------- answer hashing
static unsigned fnv1a(const char *s)
{
    unsigned h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h;
}
static int hashCheck(int idx, int answer, unsigned expected)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "M19E1|%d|%d", idx, answer);
    return fnv1a(buf) == expected;
}

// ------------------------------------------------------- CPU reference kernel
static void cpuReference(const float *x, float *y, int nsig, int L, int sig,
                         const float *hA, const float *hB, int W)
{
    float st[WMAX];
    for (int i = 0; i < W; ++i) st[i] = 0.0f;
    for (int t = 0; t < L; ++t) {
        float v = x[(size_t)t * nsig + sig];
        for (int i = 0; i < W; ++i) {
            v     = fmaf(hA[i], v, st[i]);
            st[i] = fmaf(hB[i], v, 0.5f * st[i]);
        }
        y[t] = v;
    }
}

typedef struct { const char *name; const void *fn; int threads; int dynSmem; } Entry;

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // ceilDiv and roundUp are provided for you to use in the TODOs; these two
    // casts keep the shipped file warning-clean before you have used them.
    (void)&ceilDiv; (void)&roundUp;

    // --- graceful exit with the TODOs unfilled -------------------------------
    {
        int cand[4], lim;
        if (blocksByRegisters(40, 256) == 0) { printf("Set TODO 1 first.\n"); return 0; }
        if (blocksBySharedMemory(0) == 0)    { printf("Set TODO 2 first.\n"); return 0; }
        if (blocksByWarpSlots(256) == 0)     { printf("Set TODO 2 first.\n"); return 0; }
        if (blocksPerSM(40, 0, 256, &lim, cand) == 0) { printf("Set TODO 3 first.\n"); return 0; }
        if (occupancyPercent(6, 256) == 0.0) { printf("Set TODO 4 first.\n"); return 0; }
        if (maxRegistersFor(4, 256) == 0)    { printf("Set TODO 5 first.\n"); return 0; }
    }

    printf("=== Module 19 / Exercise 1 - occupancy by hand ===\n\n");

    const Entry rows[] = {
        { "state<16>  256 thr",      (const void*)stateKernel<16,256,0>,   256, 0 },
        { "state<32>  256 thr",      (const void*)stateKernel<32,256,0>,   256, 0 },
        { "state<48>  256 thr",      (const void*)stateKernel<48,256,0>,   256, 0 },
        { "state<64>  256 thr",      (const void*)stateKernel<64,256,0>,   256, 0 },
        { "state<96>  256 thr",      (const void*)stateKernel<96,256,0>,   256, 0 },
        { "state<48>  128 thr",      (const void*)stateKernel<48,128,0>,   128, 0 },
        { "state<96>  128 thr",      (const void*)stateKernel<96,128,0>,   128, 0 },
        { "state<4>   256 thr",      (const void*)stateKernel<4,256,0>,    256, 0 },
        { "state<4>   512 thr",      (const void*)stateKernel<4,512,0>,    512, 0 },
        { "state<4>  1024 thr",      (const void*)stateKernel<4,1024,0>,  1024, 0 },
        { "state<4>   100 thr",      (const void*)stateKernel<4,100,0>,    100, 0 },
        { "state<4>    32 thr",      (const void*)stateKernel<4,32,0>,      32, 0 },
        { "state<4>  256 +16384B",   (const void*)stateKernel<4,256,4096>, 256, 0 },
        { "state<4>  256 +25600B",   (const void*)stateKernel<4,256,6400>, 256, 0 },
        { "state<4>  256 +49152B",   (const void*)stateKernel<4,256,12288>,256, 0 },
        { "state<4>  256 dyn 16384B",(const void*)stateKernel<4,256,0>,    256, 16384 },
        { "state<30>  64 thr",       (const void*)stateKernel<30,64,0>,     64, 0 },
        { "state<72>  96 thr",       (const void*)stateKernel<72,96,0>,     96, 0 },
    };
    const int NROWS = (int)(sizeof(rows) / sizeof(rows[0]));
    static const char *limName[4] = { "registers", "shared", "warp slots", "block slots" };

    printf("-- part 1: 18 real kernels --------------------------------------------\n");
    printf(" %-24s %4s %4s %6s | %3s %3s %3s %3s | %4s %-11s %4s %7s\n",
           "kernel", "thr", "reg", "smemB", "reg", "shr", "wrp", "blk",
           "you", "limiter", "API", "occ");

    int blocksOk = 0;
    for (int i = 0; i < NROWS; ++i) {
        cudaFuncAttributes at; CHECK(cudaFuncGetAttributes(&at, rows[i].fn));
        int api = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api, rows[i].fn,
                                                            rows[i].threads, rows[i].dynSmem));
        int smem = (int)at.sharedSizeBytes + rows[i].dynSmem;
        int cand[4], lim = -1;
        int mine = blocksPerSM(at.numRegs, smem, rows[i].threads, &lim, cand);
        int good = (mine == api);
        if (good) blocksOk++;
        printf(" %-24s %4d %4d %6d | %3d %3d %3d %3d | %4d %-11s %4d %6.1f%%%s\n",
               rows[i].name, rows[i].threads, at.numRegs, smem,
               cand[0], cand[1], cand[2], cand[3], mine,
               (lim >= 0 && lim < 4) ? limName[lim] : "?", api,
               occupancyPercent(api, rows[i].threads),
               good ? "" : "   <== WRONG");
    }
    printf("\n  blocks/SM correct on %d/%d kernels\n", blocksOk, NROWS);
    printf("  (the limiter column is printed here but scored in part 2, where the\n"
           "   inputs are literals and the answer cannot drift with the compiler.)\n");

    // ---------------------------------------------------------------- part 2
    printf("\n-- part 2: 10 hypothetical kernels (hashed answers) --------------------\n");
    printf("  These register counts do not all occur naturally in this file's kernel\n"
           "  family, which is exactly why they are here: two of them separate the\n"
           "  right register model from the two wrong ones.\n\n");
    struct { int regs, smem, threads; } hyp[10] = {
        {  84,     0,  256 },
        {  41,     0,  256 },
        {  32,     0,  100 },
        {  24,     0, 1024 },
        {  16,     0,   32 },
        {  40, 12288,  256 },
        {  40, 16384,  256 },
        {  64, 49152,  256 },
        {  98,     0,   64 },
        {  78,     0,  160 },
    };
    static const unsigned HYP_HASH[10] = {
        0x05d7770eu, 0x656eac6cu, 0x22579b61u, 0xc0efd026u, 0xabdb03cau,
        0xb64c20f9u, 0x23cfe4a9u, 0x9353ab7fu, 0x1518a778u, 0x8fb00757u
    };
    int hypOk = 0;
    char limSeq[128], occSeq[256];
    int lp = snprintf(limSeq, sizeof(limSeq), "M19E1L");
    int op = snprintf(occSeq, sizeof(occSeq), "M19E1O");
    printf(" %4s %7s %5s | %4s %4s %4s %4s | %4s %-11s %7s %s\n",
           "regs", "smemB", "thr", "reg", "shr", "wrp", "blk", "you", "limiter", "occ", "");
    for (int i = 0; i < 10; ++i) {
        int cand[4], lim = -1;
        int mine = blocksPerSM(hyp[i].regs, hyp[i].smem, hyp[i].threads, &lim, cand);
        int ok = hashCheck(i, mine, HYP_HASH[i]);
        if (ok) hypOk++;
        double occ = occupancyPercent(mine, hyp[i].threads);
        lp += snprintf(limSeq + lp, sizeof(limSeq) - lp, "|%d", lim);
        op += snprintf(occSeq + op, sizeof(occSeq) - op, "|%d", (int)(occ * 10.0 + 0.5));
        printf(" %4d %7d %5d | %4d %4d %4d %4d | %4d %-11s %6.1f%% %s\n",
               hyp[i].regs, hyp[i].smem, hyp[i].threads,
               cand[0], cand[1], cand[2], cand[3], mine,
               (lim >= 0 && lim < 4) ? limName[lim] : "?", occ, ok ? "ok" : "WRONG");
    }
    const int limSeqOk = (fnv1a(limSeq) == 0x8ad18415u);
    const int occSeqOk = (fnv1a(occSeq) == 0xfbe1da0au);
    printf("\n  %d/10 block counts correct; limiter sequence %s; occupancy sequence %s\n",
           hypOk, limSeqOk ? "ok" : "WRONG", occSeqOk ? "ok" : "WRONG");

    // ---------------------------------------------------------------- part 3
    printf("\n-- part 3: TODO 5, the inverted arithmetic -----------------------------\n");
    printf(" %7s %8s | %9s %s\n", "blocks", "threads", "max regs", "round trip");
    struct { int b, t; } inv[8] = {
        { 2, 256 }, { 3, 256 }, { 4, 256 }, { 6, 256 },
        { 4, 128 }, { 8, 128 }, { 2, 512 }, { 12, 64 }
    };
    int invOk = 0;
    for (int i = 0; i < 8; ++i) {
        int R = maxRegistersFor(inv[i].b, inv[i].t);
        int up = (R >= 1 && R <= 255) ? blocksByRegisters(R, inv[i].t) : -1;
        int dn = (R >= 1 && R < 255)  ? blocksByRegisters(R + 1, inv[i].t) : 0;
        int ok = (R >= 1 && R <= 255 && up >= inv[i].b && (R == 255 || dn < inv[i].b));
        if (ok) invOk++;
        printf(" %7d %8d | %9d %s (blocks at R = %d, at R+1 = %d)\n",
               inv[i].b, inv[i].t, R, ok ? "ok   " : "WRONG", up, dn);
    }
    printf("\n  %d/8 round trips consistent\n", invOk);

    // ------------------------------------------------- numerical sanity check
    printf("\n-- the kernels really compute something (untimed) ----------------------\n");
    int numOk = 0;
    {
        const int W = 16, T = 256, nsig = T * 8, L = 64;
        float hA[WMAX], hB[WMAX];
        for (int i = 0; i < WMAX; ++i) {
            hA[i] = 0.980f + 0.0010f * (float)(i % 7);
            hB[i] = 0.0040f - 0.00010f * (float)(i % 11);
        }
        CHECK(cudaMemcpyToSymbol(cA, hA, sizeof(hA)));
        CHECK(cudaMemcpyToSymbol(cB, hB, sizeof(hB)));
        size_t n = (size_t)nsig * L;
        float *hx = (float*)malloc(n * 4), *hy = (float*)malloc(n * 4);
        unsigned s = 1u;
        for (size_t i = 0; i < n; ++i) {
            s = s * 1664525u + 1013904223u;
            hx[i] = (float)((s >> 9) & 0xFFFFu) / 65536.0f - 0.5f;
        }
        float *dx, *dy;
        CHECK(cudaMalloc(&dx, n * 4)); CHECK(cudaMalloc(&dy, n * 4));
        CHECK(cudaMemcpy(dx, hx, n * 4, cudaMemcpyHostToDevice));
        stateKernel<W,T,0><<<nsig / T, T>>>(dx, dy, nsig, L);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hy, dy, n * 4, cudaMemcpyDeviceToHost));
        float ref[64]; double worst = 0.0;
        for (int sig = 0; sig < nsig; sig += 37) {
            cpuReference(hx, ref, nsig, L, sig, hA, hB, W);
            for (int t = 0; t < L; ++t) {
                double e = fabs((double)hy[(size_t)t * nsig + sig] - (double)ref[t]);
                double sc = fabs((double)ref[t]) + 1e-6;
                if (e / sc > worst) worst = e / sc;
            }
        }
        numOk = (worst <= 1e-4);
        printf("  worst relative error: %.3g  (%s)\n", worst, numOk ? "ok" : "FAIL");
        free(hx); free(hy); CHECK(cudaFree(dx)); CHECK(cudaFree(dy));
    }

    // ---------------------------------------------------------------- scoring
    int score = 0;
    score += (blocksOk == NROWS) ? 3 : 0;
    score += (hypOk   == 10)    ? 2 : 0;
    score += limSeqOk           ? 2 : 0;
    score += occSeqOk           ? 1 : 0;
    score += (invOk   == 8)     ? 2 : 0;
    printf("\n-- scoring ------------------------------------------------------------\n");
    printf("  blocks/SM on real kernels   %2d/%2d  -> %d/3\n", blocksOk, NROWS, (blocksOk == NROWS) ? 3 : 0);
    printf("  hypothetical block counts   %2d/10  -> %d/2\n", hypOk, (hypOk == 10) ? 2 : 0);
    printf("  binding limiter (TODO 3)       %-4s-> %d/2\n", limSeqOk ? "ok" : "no", limSeqOk ? 2 : 0);
    printf("  occupancy %% (TODO 4)           %-4s-> %d/1\n", occSeqOk ? "ok" : "no", occSeqOk ? 1 : 0);
    printf("  TODO 5 round trips          %2d/ 8  -> %d/2\n", invOk, (invOk == 8) ? 2 : 0);
    printf("\n  SCORE: %d/10\n", score);
    printf("\nOVERALL: %s\n", (score == 10 && numOk) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (score == 10 && numOk) ? 0 : 1;
}
