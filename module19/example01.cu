// =============================================================================
// Module 19 / Example 1 — the four placement limiters, computed by hand and
//                         checked against the CUDA occupancy API.
//
// GOAL : Everything in this file is *static* resource arithmetic. No kernel is
//        timed. Four parts:
//
//   A  The four limiters on sm_89, applied to 24 real kernels. For each one the
//      program prints the four candidate block counts, the minimum, which
//      resource bound it, and what
//      cudaOccupancyMaxActiveBlocksPerMultiprocessor says. They agree 24/24.
//
//   B  Why the register term is NOT 65536 / (regs * threads). Module 18
//      established the 8-register-per-thread granule; that is necessary and not
//      sufficient. The register file is four 16384-register slices (Module 1),
//      one per processing block, and a warp lives entirely inside one slice.
//      This part prints the kernels where the aggregate model gives the wrong
//      answer and the slice model gives the right one.
//
//   C  The register/occupancy exchange rate: how many resident warps each
//      register you give up actually buys, as a table and as a derivative.
//      The answer collapses quadratically, which is the single most useful
//      fact in this module.
//
//   D  The API surface: cudaFuncGetAttributes, cudaOccupancyMaxPotentialBlockSize
//      and what __launch_bounds__(T, B) costs you, with the register cap it
//      forces on ptxas computed in closed form.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN  : example01.exe
// See also: nvcc -arch=sm_89 -O3 -Xptxas -v -o example01.exe example01.cu
//           (the "Used N registers" lines are what part A reads at runtime)
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

// ---- Ada / sm_89 placement constants (spec §1) ------------------------------
#define SM_COUNT          40
#define REGS_PER_SM       65536     // 4 slices x 16384
#define REG_SLICES        4         // one per processing block (Module 1)
#define REGS_PER_SLICE    16384
#define REG_GRAN          8         // registers per thread, allocated per warp
#define WARPS_PER_SM      48        // warp slots; 48 * 32 = 1536 threads
#define MAX_BLOCKS_PER_SM 24
#define SMEM_PER_SM       102400
#define SMEM_RESERVE      1024      // per-block driver reserve (Module 6)
#define SMEM_GRAN         128       // allocation granularity (Module 6)

static int ceilDiv(int a, int b) { return (a + b - 1) / b; }
static int roundUp(int a, int g) { return ceilDiv(a, g) * g; }
static int roundDown(int a, int g) { return (a / g) * g; }

// -----------------------------------------------------------------------------
// The four limiters. Returns blocks per SM and, through `limiter`, the name of
// the resource that bound it. `cand` receives the four candidate values in the
// order { registers, shared, warp slots, block slots }.
// -----------------------------------------------------------------------------
static int blocksPerSM(int regsPerThread, int smemBytes, int threads,
                       const char **limiter, int cand[4])
{
    const int warpsPerBlock = ceilDiv(threads, 32);

    // 1. REGISTERS. A warp's allocation is roundUp(regs,8)*32, and it must fit
    //    entirely in ONE of the four 16384-register slices.
    const int regsPerWarp   = roundUp(regsPerThread, REG_GRAN) * 32;
    const int warpsPerSlice = regsPerWarp ? (REGS_PER_SLICE / regsPerWarp) : WARPS_PER_SM;
    cand[0] = (REG_SLICES * warpsPerSlice) / warpsPerBlock;

    // 2. SHARED MEMORY. Per-block request + 1024 B driver reserve, rounded up
    //    to 128 B, divided into 102400 B.
    cand[1] = SMEM_PER_SM / roundUp(smemBytes + SMEM_RESERVE, SMEM_GRAN);

    // 3. WARP SLOTS. 48 per SM. A 100-thread block consumes FOUR slots, not
    //    3.125 — partial warps occupy whole slots (Module 8).
    cand[2] = WARPS_PER_SM / warpsPerBlock;

    // 4. BLOCK SLOTS. A hard architectural cap.
    cand[3] = MAX_BLOCKS_PER_SM;

    static const char *names[4] = { "registers", "shared", "warp slots", "block slots" };
    int best = cand[0], which = 0;
    for (int i = 1; i < 4; ++i) if (cand[i] < best) { best = cand[i]; which = i; }
    *limiter = names[which];
    return best;
}

// Module 18's model: one 65536-register pool, 8-register granule, no slices.
static int blocksPerSM_aggregate(int regsPerThread, int smemBytes, int threads)
{
    const int w = ceilDiv(threads, 32);
    int byReg = REGS_PER_SM / (roundUp(regsPerThread, REG_GRAN) * 32 * w);
    int bySm  = SMEM_PER_SM / roundUp(smemBytes + SMEM_RESERVE, SMEM_GRAN);
    int byWrp = WARPS_PER_SM / w;
    int m = byReg;
    if (bySm < m) m = bySm;
    if (byWrp < m) m = byWrp;
    if (MAX_BLOCKS_PER_SM < m) m = MAX_BLOCKS_PER_SM;
    return m;
}

static double occupancyOf(int blocks, int threads)
{
    int w = blocks * ceilDiv(threads, 32);
    if (w > WARPS_PER_SM) w = WARPS_PER_SM;
    return (double)w / WARPS_PER_SM;
}

// =============================================================================
// The kernel family. `W` private state values live across the whole time loop,
// so W is a direct handle on the register count; `S` floats of static shared
// memory give a handle on the shared-memory footprint. Nothing here is timed —
// the kernels exist so that the compiler produces real resource numbers for
// cudaFuncGetAttributes to report.
//
// This is the shape of any sequential-state kernel: an IIR filter bank, an RNN
// step, a per-thread integrator. One input sample per time step, W recursive
// stages, state carried in registers.
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
        if (SFLOATS > 0) v += sh[sig % (SFLOATS > 0 ? SFLOATS : 1)] * 0.0f;  // keeps sh[] alive
        y[(size_t)t * nsig + sig] = v;
    }
}

// Same body, but asking ptxas for a minimum number of resident blocks. The
// second argument of __launch_bounds__ is a hard register cap in disguise.
template<int W, int T, int MINB>
__global__ __launch_bounds__(T, MINB) void stateKernelMin(const float * __restrict__ x,
                                                          float * __restrict__ y,
                                                          int nsig, int L)
{
    const int sig = blockIdx.x * T + threadIdx.x;
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
        y[(size_t)t * nsig + sig] = v;
    }
}

// -----------------------------------------------------------------------------
typedef struct { const char *name; const void *fn; int threads; int dynSmem; } Entry;

static int reportRow(const Entry *e, int *agreed, int *aggWrong)
{
    cudaFuncAttributes at;
    CHECK(cudaFuncGetAttributes(&at, e->fn));
    int api = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api, e->fn, e->threads, e->dynSmem));

    const int smem = (int)at.sharedSizeBytes + e->dynSmem;
    const char *lim = "?";
    int cand[4];
    int mine = blocksPerSM(at.numRegs, smem, e->threads, &lim, cand);
    int agg  = blocksPerSM_aggregate(at.numRegs, smem, e->threads);

    if (mine == api) (*agreed)++;
    if (agg != api) (*aggWrong)++;

    printf(" %-26s %4d %4d %6d | %3d %3d %3d %3d | %3d %-12s %3d  %5.1f%%%s\n",
           e->name, e->threads, at.numRegs, smem,
           cand[0], cand[1], cand[2], cand[3],
           mine, lim, api, 100.0 * occupancyOf(api, e->threads),
           (mine == api) ? "" : "   <== HAND COUNT DISAGREES");
    return agg;
}

// -----------------------------------------------------------------------------
// Part C: the exchange rate. With `threads` per block, how many warps/SM does a
// kernel using R registers per thread get?
// -----------------------------------------------------------------------------
static int warpsForRegs(int R, int threads)
{
    const char *lim; int cand[4];
    int b = blocksPerSM(R, 0, threads, &lim, cand);
    int w = b * ceilDiv(threads, 32);
    return w > WARPS_PER_SM ? WARPS_PER_SM : w;
}

// The register cap __launch_bounds__(T, B) forces on ptxas, in closed form.
static int launchBoundsRegCap(int threads, int minBlocks)
{
    int cap = REGS_PER_SM / (threads * minBlocks);
    cap = roundDown(cap, REG_GRAN);
    if (cap > 255) cap = 255;
    return cap;
}

// -----------------------------------------------------------------------------
static void cpuReference(const float *x, float *y, int nsig, int L, int sig,
                         const float *hA, const float *hB, int W)
{
    float *st = (float*)malloc(sizeof(float) * (size_t)W);
    for (int i = 0; i < W; ++i) st[i] = 0.0f;
    for (int t = 0; t < L; ++t) {
        float v = x[(size_t)t * nsig + sig];
        for (int i = 0; i < W; ++i) {
            v     = fmaf(hA[i], v, st[i]);
            st[i] = fmaf(hB[i], v, 0.5f * st[i]);
        }
        y[t] = v;
    }
    free(st);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("=== Module 19 / Example 1 - the four placement limiters ===\n\n");
    printf("sm_89 placement budget, per SM:\n");
    printf("  warp slots          %d   (= %d threads; a partial warp still takes a slot)\n",
           WARPS_PER_SM, WARPS_PER_SM * 32);
    printf("  block slots         %d\n", MAX_BLOCKS_PER_SM);
    printf("  32-bit registers    %d  = %d slices x %d, granule %d regs/thread/warp\n",
           REGS_PER_SM, REG_SLICES, REGS_PER_SLICE, REG_GRAN);
    printf("  shared memory       %d B, +%d B driver reserve per block, %d B granularity\n\n",
           SMEM_PER_SM, SMEM_RESERVE, SMEM_GRAN);

    // ---------------------------------------------------------------- Part A
    printf("-- A. four candidates, their minimum, and the API ---------------------\n");
    printf(" %-26s %4s %4s %6s | %3s %3s %3s %3s | %3s %-12s %3s  %6s\n",
           "kernel", "thr", "reg", "smemB", "reg", "shr", "wrp", "blk",
           "min", "limiter", "API", "occ");

    const Entry rows[] = {
        // registers bind
        { "state<16> 256 thr",       (const void*)stateKernel<16,256,0>,   256, 0 },
        { "state<32> 256 thr",       (const void*)stateKernel<32,256,0>,   256, 0 },
        { "state<48> 256 thr",       (const void*)stateKernel<48,256,0>,   256, 0 },
        { "state<64> 256 thr",       (const void*)stateKernel<64,256,0>,   256, 0 },
        { "state<96> 256 thr",       (const void*)stateKernel<96,256,0>,   256, 0 },
        { "state<48> 128 thr",       (const void*)stateKernel<48,128,0>,   128, 0 },
        { "state<96> 128 thr",       (const void*)stateKernel<96,128,0>,   128, 0 },
        // warp slots bind
        { "state<4>  256 thr",       (const void*)stateKernel<4,256,0>,    256, 0 },
        { "state<4>  512 thr",       (const void*)stateKernel<4,512,0>,    512, 0 },
        { "state<4> 1024 thr",       (const void*)stateKernel<4,1024,0>,  1024, 0 },
        { "state<4>  100 thr",       (const void*)stateKernel<4,100,0>,    100, 0 },
        { "state<4>  160 thr",       (const void*)stateKernel<4,160,0>,    160, 0 },
        // block slots bind
        { "state<4>   32 thr",       (const void*)stateKernel<4,32,0>,      32, 0 },
        { "state<4>   64 thr",       (const void*)stateKernel<4,64,0>,      64, 0 },
        // shared memory binds (static)
        { "state<4> 256 +4096B",     (const void*)stateKernel<4,256,1024>, 256, 0 },
        { "state<4> 256 +16384B",    (const void*)stateKernel<4,256,4096>, 256, 0 },
        { "state<4> 256 +25600B",    (const void*)stateKernel<4,256,6400>, 256, 0 },
        { "state<4> 256 +49152B",    (const void*)stateKernel<4,256,12288>,256, 0 },
        // shared memory binds (dynamic, third launch parameter)
        { "state<4> 256 dyn 12288B", (const void*)stateKernel<4,256,0>,    256, 12288 },
        { "state<4> 256 dyn 16384B", (const void*)stateKernel<4,256,0>,    256, 16384 },
        { "state<4> 256 dyn 32768B", (const void*)stateKernel<4,256,0>,    256, 32768 },
        // the slice-model cases: registers bind and the aggregate model is wrong
        { "state<30>  64 thr",       (const void*)stateKernel<30,64,0>,     64, 0 },
        { "state<72>  96 thr",       (const void*)stateKernel<72,96,0>,     96, 0 },
        { "state<64> 160 thr",       (const void*)stateKernel<64,160,0>,   160, 0 },
    };
    const int NROWS = (int)(sizeof(rows) / sizeof(rows[0]));

    int agreed = 0, aggWrong = 0;
    for (int i = 0; i < NROWS; ++i) (void)reportRow(&rows[i], &agreed, &aggWrong);
    printf("\n  hand count agrees with cudaOccupancyMaxActiveBlocksPerMultiprocessor"
           " on %d/%d kernels\n", agreed, NROWS);
    printf("  (ties are reported as the first binding resource in the order\n"
           "   registers, shared, warp slots, block slots.)\n");
    printf("  Not a limiter, a hard error: a per-block shared request above\n"
           "  49152 B is rejected at launch unless you opt in with\n"
           "  cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, n)\n"
           "  (Module 6). That gate returns 0 blocks, not fewer blocks.\n");

    // ---------------------------------------------------------------- Part B
    printf("\n-- B. why the register term is not 65536/(regs*threads) ---------------\n");
    printf("  Module 18 established the 8-register granule. That is necessary and\n"
           "  NOT sufficient: the register file is %d slices of %d registers\n"
           "  (Module 1's processing blocks) and a warp lives wholly inside one.\n\n",
           REG_SLICES, REGS_PER_SLICE);
    printf(" %-26s %4s %4s | %8s %8s %4s\n",
           "kernel", "thr", "reg", "slice", "aggregate", "API");
    for (int i = 0; i < NROWS; ++i) {
        cudaFuncAttributes at; CHECK(cudaFuncGetAttributes(&at, rows[i].fn));
        int api = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api, rows[i].fn,
                                                            rows[i].threads, rows[i].dynSmem));
        int smem = (int)at.sharedSizeBytes + rows[i].dynSmem;
        const char *lim; int cand[4];
        int mine = blocksPerSM(at.numRegs, smem, rows[i].threads, &lim, cand);
        int agg  = blocksPerSM_aggregate(at.numRegs, smem, rows[i].threads);
        if (agg != mine)
            printf(" %-26s %4d %4d | %8d %8d %4d   aggregate is WRONG\n",
                   rows[i].name, rows[i].threads, at.numRegs, mine, agg, api);
    }
    printf("\n  Worked example, state<30> at 64 threads (2 warps):\n");
    {
        cudaFuncAttributes at; CHECK(cudaFuncGetAttributes(&at, (const void*)stateKernel<30,64,0>));
        int rpw = roundUp(at.numRegs, REG_GRAN) * 32;
        printf("    %d regs/thread -> roundUp(%d,%d) = %d -> %d regs per warp\n",
               at.numRegs, at.numRegs, REG_GRAN, roundUp(at.numRegs, REG_GRAN), rpw);
        printf("    aggregate : %d / %d                = %d warps -> %d blocks of 2 warps\n",
               REGS_PER_SM, rpw, REGS_PER_SM / rpw, (REGS_PER_SM / rpw) / 2);
        printf("    slices    : 4 * floor(%d / %d) = 4 * %d = %d warps -> %d blocks\n",
               REGS_PER_SLICE, rpw, REGS_PER_SLICE / rpw,
               4 * (REGS_PER_SLICE / rpw), (4 * (REGS_PER_SLICE / rpw)) / 2);
        printf("    the 2 leftover registers per slice cannot be pooled across slices.\n");
    }

    // ---------------------------------------------------------------- Part C
    printf("\n-- C. the register / occupancy exchange rate --------------------------\n");
    printf("  resident warps as a function of registers per thread.\n");
    printf("  'gain' is the warps bought by surrendering the NEXT granule of 8.\n\n");
    printf(" %5s | %6s %5s | %6s %5s | %6s %5s\n",
           "regs", "128thr", "gain", "256thr", "gain", "512thr", "gain");
    for (int R = 24; R <= 168; R += 8) {
        int w128 = warpsForRegs(R, 128),  n128 = warpsForRegs(R - 8, 128);
        int w256 = warpsForRegs(R, 256),  n256 = warpsForRegs(R - 8, 256);
        int w512 = warpsForRegs(R, 512),  n512 = warpsForRegs(R - 8, 512);
        printf(" %5d | %6d %5d | %6d %5d | %6d %5d\n",
               R, w128, n128 - w128, w256, n256 - w256, w512, n512 - w512);
    }
    printf("\n  Reading the table: the exchange rate is not linear. Ignoring block\n"
           "  quantisation, resident warps ~ 4*floor(16384 / (32*roundUp(R,8))), so\n"
           "  d(warps)/d(R) ~ -2048/R^2. Every granule you surrender buys FOUR\n"
           "  TIMES less than the one before it did at half the register count.\n"
           "  At 40 registers a granule is worth several warps; at 120 it is worth\n"
           "  one; past 168 it is often worth nothing at all.\n");

    printf("\n  Block size changes the quantisation loss at a FIXED register count:\n");
    for (int R = 88; R <= 104; R += 8) {
        printf("    %3d regs/thread: 32thr %2d  64thr %2d  128thr %2d  256thr %2d  512thr %2d warps\n",
               R, warpsForRegs(R,32), warpsForRegs(R,64), warpsForRegs(R,128),
               warpsForRegs(R,256), warpsForRegs(R,512));
    }
    printf("    Same registers, different occupancy: a big block must place all of\n"
           "    its warps at once, so it rounds down harder. A zero in the table is\n"
           "    not an occupancy of zero -- it is a launch that cannot be placed at\n"
           "    all and fails with cudaErrorLaunchOutOfResources.\n");

    // ---------------------------------------------------------------- Part D
    printf("\n-- D. the API surface -------------------------------------------------\n");
    {
        cudaFuncAttributes at;
        CHECK(cudaFuncGetAttributes(&at, (const void*)stateKernel<64,256,0>));
        printf("  cudaFuncGetAttributes(state<64>, 256 thr):\n");
        printf("    numRegs           %d\n", at.numRegs);
        printf("    localSizeBytes    %d   (this is the SPILL figure at runtime)\n",
               (int)at.localSizeBytes);
        printf("    sharedSizeBytes   %d   (static only; dynamic is invisible here)\n",
               (int)at.sharedSizeBytes);
        printf("    maxThreadsPerBlock %d  (what __launch_bounds__'s 1st arg set)\n",
               at.maxThreadsPerBlock);

        int minGrid = 0, blockSize = 0;
        CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
                                                 stateKernel<64,256,0>, 0, 0));
        printf("\n  cudaOccupancyMaxPotentialBlockSize -> block %d, minGrid %d\n",
               blockSize, minGrid);
        printf("    It returns the block size that MAXIMISES OCCUPANCY, and it is\n"
               "    honest about that: the word 'performance' does not appear in its\n"
               "    contract. Module 18 measured those two diverging by 18.7x on one\n"
               "    GEMM source. Treat it as a starting point, never as an answer.\n");
    }
    printf("\n  what __launch_bounds__(T, B) costs you: it caps ptxas at\n"
           "    roundDown(65536 / (T*B), 8) registers per thread\n");
    printf(" %10s %10s %10s %10s %10s\n", "threads", "minBlocks", "cap(pred)", "regs(real)", "spillB");
    {
        struct { const char *n; const void *fn; int t; int b; } lb[] = {
            { "state<96>", (const void*)stateKernelMin<96,128,1>,  128, 1  },
            { "state<96>", (const void*)stateKernelMin<96,128,3>,  128, 3  },
            { "state<96>", (const void*)stateKernelMin<96,128,4>,  128, 4  },
            { "state<96>", (const void*)stateKernelMin<96,128,5>,  128, 5  },
            { "state<96>", (const void*)stateKernelMin<96,128,6>,  128, 6  },
            { "state<96>", (const void*)stateKernelMin<96,128,8>,  128, 8  },
            { "state<96>", (const void*)stateKernelMin<96,128,12>, 128, 12 },
        };
        for (int i = 0; i < 7; ++i) {
            cudaFuncAttributes a; CHECK(cudaFuncGetAttributes(&a, lb[i].fn));
            printf(" %10d %10d %10d %10d %10d%s\n", lb[i].t, lb[i].b,
                   launchBoundsRegCap(lb[i].t, lb[i].b), a.numRegs, (int)a.localSizeBytes,
                   (a.localSizeBytes > 0) ? "   <- ptxas had to spill" : "");
        }
    }
    printf("\n  The first row does not bind (the cap is above what the kernel wanted),\n"
           "  which is why asking for 1 block changes nothing. Every later row is\n"
           "  ptxas being told to fit in less, and from the 5th it pays in local\n"
           "  memory -- which Module 4 established is DRAM. Exercise 2 measures\n"
           "  what each of those rows is worth.\n");

    // ------------------------------------------------- correctness (untimed)
    printf("\n-- correctness check (the kernel family really computes something) -----\n");
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
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hy, dy, n * 4, cudaMemcpyDeviceToHost));

        float *ref = (float*)malloc(sizeof(float) * (size_t)L);
        double worst = 0.0;
        for (int sig = 0; sig < nsig; sig += 37) {
            cpuReference(hx, ref, nsig, L, sig, hA, hB, W);
            for (int t = 0; t < L; ++t) {
                double e = fabs((double)hy[(size_t)t * nsig + sig] - (double)ref[t]);
                double sc = fabs((double)ref[t]) + 1e-6;
                if (e / sc > worst) worst = e / sc;
            }
        }
        int ok = (worst <= 1e-4);
        printf("  worst relative error over %d sampled signals: %.3g  (%s)\n",
               nsig / 37, worst, ok ? "ok" : "TOO LARGE");
        free(ref); free(hx); free(hy);
        CHECK(cudaFree(dx)); CHECK(cudaFree(dy));

        printf("\nOVERALL: %s\n", (ok && agreed == NROWS) ? "PASS" : "FAIL");
        CHECK(cudaDeviceReset());
        return (ok && agreed == NROWS) ? 0 : 1;
    }
}
