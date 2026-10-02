// =============================================================================
// Module 23 / Exercise 2 - derive the counters yourself.
//
// GOAL : `ncu` cannot run on this machine (ERR_NVGPUCTRPERM). So build the
//        counters. You will write kernels that compute, about themselves, four
//        of the quantities Nsight Compute would report:
//
//          sectors per request    l1tex__t_sectors_... / l1tex__t_requests_...
//          bank conflict degree   l1tex__data_bank_conflicts_...
//          achieved occupancy     sm__warps_active.avg.pct_of_peak_sustained_*
//          lane efficiency        smsp__thread_inst_executed_per_inst_executed
//
//        and then write CLOSED-FORM predictions for the first two and check the
//        measurement against the formula, row by row.
//
//        This is harder than reading the tool and it teaches the metrics far
//        better. A counter you had to build is a counter you cannot misread.
//
// WHAT TO FILL IN
//   TODO 1  distinctInWarp() and sectorsThisRequest()
//   TODO 2  conflictDegree() -- the obvious version is wrong on two of the
//           eight patterns, and the program prints both answers side by side
//   TODO 3  reduceOccupancy() -- two reductions differing only by a denominator
//   TODO 4  recordIssue() -- lane efficiency; the wrong reporter lane counts
//           nothing at all and still prints a number
//   TODO 5  predictSectorsStride(), predictDegreeStride(), predictDegreeDivide()
//           -- all CLOSED FORM. No enumeration, no loop over 32 lanes.
//
// Every row is scored against your formula. OVERALL: PASS requires all of them.
//
// BUILD: nvcc -arch=sm_89 -O3 -lineinfo -o exercise02.exe exercise02.cu
// RUN  : exercise02.exe
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

#define FULL      0xffffffffu
#define SECTOR_B  32
#define BANKS     32
#define NSM       40
#define WARP_SLOTS_PER_SM 48

#define NGLOBAL   ((size_t)1 << 22)        // 4 Mi floats of source data
#define SH_WORDS  2048                     // shared probe array, floats

// =============================================================================
// TODO 1 — distinct-key count across a warp, and sectors per request.
// =============================================================================
__device__ __forceinline__ unsigned distinctInWarp(unsigned key)
{
    // TODO 1a: return the number of DISTINCT values of `key` across the 32 lanes
    //          of this (fully active) warp. Every lane must return the same
    //          answer. One warp intrinsic does the hard part in one instruction
    //          -- find it rather than looping 32 times.
    // YOUR CODE HERE
    (void)key;
    return 0u;
}

__device__ __forceinline__ unsigned sectorsThisRequest(const void *addr)
{
    // TODO 1b: the number of 32-byte sectors one warp instruction touches, given
    //          each lane's byte address. This is exactly
    //            l1tex__t_sectors_... / l1tex__t_requests_...
    //          and exactly Module 5's hand procedure. One line, on top of 1a.
    // YOUR CODE HERE
    (void)addr;
    return 0u;
}

// =============================================================================
// TODO 2 — shared-memory conflict degree, counting DISTINCT WORDS per bank.
// =============================================================================
__device__ __forceinline__ unsigned conflictDegree(unsigned wordIndex)
{
    // TODO 2: the shared-memory conflict degree of this warp's access -- what
    //         l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum divided by
    //         the request count reports.
    //
    //         bank = (byte address / 4) % 32, so bank = wordIndex % 32.
    //
    //         The degree is the maximum, over the 32 banks, of the number of
    //         ??? that bank must supply. Getting the ??? right IS the exercise:
    //         the obvious choice disagrees with the hardware on two of the
    //         eight patterns tested, and the program prints both answers side
    //         by side so you can see which two and why.
    //
    //         Every lane must return the same value.
    // YOUR CODE HERE
    (void)wordIndex;
    return 0u;
}

// Same thing done the WRONG way: bucket lanes, not distinct words.
__device__ __forceinline__ unsigned conflictDegreeByLane(unsigned wordIndex)
{
    unsigned bank = wordIndex & (BANKS - 1);
    unsigned deg  = 0;
    #pragma unroll
    for (int b = 0; b < BANKS; ++b) {
        unsigned c = (unsigned)__popc(__ballot_sync(FULL, bank == (unsigned)b));
        if (c > deg) deg = c;
    }
    return deg;
}

// =============================================================================
// Address generators. `kind` selects the source expression under study.
//   0..5 : in[base + stride*i]              (global)
//   6..  : s[expr(tid)]                     (shared)
// =============================================================================
struct Pattern { const char *expr; int stride; int base; };

__constant__ Pattern dPatG[8];

__global__ void kSectors(const float * __restrict__ in, unsigned *out, int np)
{
    int lane = (int)(threadIdx.x & 31u);
    for (int p = 0; p < np; ++p) {
        int s = dPatG[p].stride, b = dPatG[p].base;
        const float *a = in + (size_t)b + (size_t)s * lane;
        unsigned n = sectorsThisRequest(a);
        // Keep the load alive so the address arithmetic is real, not folded.
        float v = *a;
        if (lane == 0 && v == 1e30f) out[np + p] = 1u;
        if (lane == 0) out[p] = n;
    }
}

__global__ void kBanks(unsigned *outDeg, unsigned *outLane, float *sink, int np)
{
    __shared__ float s[SH_WORDS];
    for (int i = (int)threadIdx.x; i < SH_WORDS; i += (int)blockDim.x)
        s[i] = (float)(i & 255);
    __syncthreads();

    int lane = (int)(threadIdx.x & 31u);
    float acc = 0.0f;
    for (int p = 0; p < np; ++p) {
        unsigned w;
        switch (p) {
            case 0: w = (unsigned)(lane);      break;   // s[tid]
            case 1: w = (unsigned)(2 * lane);  break;   // s[2*tid]
            case 2: w = (unsigned)(3 * lane);  break;   // s[3*tid]
            case 3: w = (unsigned)(4 * lane);  break;   // s[4*tid]
            case 4: w = (unsigned)(8 * lane);  break;   // s[8*tid]
            case 5: w = (unsigned)(32 * lane); break;   // s[32*tid]
            case 6: w = (unsigned)(lane / 2);  break;   // s[tid/2]
            default:w = 0u;                    break;   // s[0]
        }
        w &= (SH_WORDS - 1);
        unsigned d  = conflictDegree(w);
        unsigned dl = conflictDegreeByLane(w);
        acc += s[w];
        if (lane == 0) { outDeg[p] = d; outLane[p] = dl; }
    }
    if (acc == 1e30f) sink[0] = acc;
}

// =============================================================================
// TODO 3 — achieved occupancy, both denominators, Module 19's instrument.
// =============================================================================
__device__ __forceinline__ unsigned smid(void)
{ unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned long long clk(void)
{ unsigned long long r; asm volatile("mov.u64 %0, %%clock64;" : "=l"(r) :: "memory"); return r; }

__global__ void kOccProbe(const int * __restrict__ work, float *sink,
                          unsigned long long *resident,
                          unsigned long long *smStart,
                          unsigned long long *smEnd)
{
    unsigned long long t0 = clk();
    unsigned sm = smid();

    float a[4]; const float b = 1.0000001f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = (float)(threadIdx.x + i);
    int W = work[blockIdx.x];
    #pragma unroll 8
    for (int t = 0; t < W; ++t) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = fmaf(a[i], b, 1.0f);
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) s += a[i];
    if (s == 1e30f) sink[0] = s;

    unsigned long long t1 = clk();
    if ((threadIdx.x & 31u) == 0u) {
        atomicAdd(&resident[sm], t1 - t0);
        atomicMin(&smStart[sm], t0);
        atomicMax(&smEnd[sm],   t1);
    }
}

// TODO 3 — the two reductions.  `warpsPerSlot` = WARP_SLOTS_PER_SM.
// occ_active  : each SM's warp-cycles over ITS OWN busy span, averaged over
//               the SMs that ran anything.
// occ_elapsed : each SM's warp-cycles over the WHOLE kernel's span, averaged
//               over all NSM SMs (idle SMs contribute zero, not nothing).
// The whole-kernel span may only be the maximum PER-SM span: %clock64 counters
// are per-SM and are not mutually synchronised (M19 measured 298 M cycles of
// offset inside one launch), so min/max over raw timestamps across SMs is
// meaningless.
static void reduceOccupancy(const unsigned long long *res,
                            const unsigned long long *st,
                            const unsigned long long *en,
                            double *occActive, double *occElapsed)
{
    // TODO 3: two occupancy numbers out of the same three per-SM arrays.
    //
    //   res[i] = total warp-cycles of residency accumulated on SM i
    //   st[i]  = earliest clock64() any warp on SM i observed  (~0ull if none)
    //   en[i]  = latest   clock64() any warp on SM i observed  (0 if none)
    //
    //   occActive  -> sm__warps_active.avg.pct_of_peak_sustained_ACTIVE,
    //                 i.e. ncu's "Achieved Occupancy". Each SM's warp-cycles
    //                 over THAT SM's OWN busy span, averaged over the SMs that
    //                 ran something.
    //   occElapsed -> sm__warps_active.avg.pct_of_peak_sustained_ELAPSED.
    //                 Each SM's warp-cycles over the WHOLE KERNEL's span,
    //                 averaged over all NSM SMs, so an idle SM contributes a
    //                 zero rather than being left out of the average.
    //
    //   Both are percentages; the per-SM denominator is
    //   WARP_SLOTS_PER_SM * (the relevant span).
    //
    //   ONE TRAP, and it is why this is a TODO: the %clock64 counters are
    //   PER-SM and are NOT mutually synchronised. Module 19 measured up to
    //   298 million cycles of offset inside a single launch, so the kernel's
    //   span may NOT be max(en) - min(st) taken across SMs. What is the only
    //   defensible estimate of the kernel span from these arrays?
    //
    // YOUR CODE HERE
    (void)res; (void)st; (void)en;
    *occActive = 0.0; *occElapsed = 0.0;
}

// =============================================================================
// TODO 4 — lane efficiency.
//   smsp__thread_inst_executed_per_inst_executed.ratio
//     = sum over executed instructions of (active lanes) / (instructions)
// Exactly one lane per warp must do the accumulating, and it has to be a lane
// that is guaranteed active -- the LOWEST SET BIT of the active mask.
// =============================================================================
__device__ __forceinline__ void recordIssue(unsigned long long *c)
{
    // TODO 4: accumulate the two counters whose quotient is
    //           smsp__thread_inst_executed_per_inst_executed.ratio
    //         for the instruction stream at THIS point in the kernel:
    //           c[0] += (threads executing here, this issue)
    //           c[1] += 1                   (one warp instruction issued)
    //
    //         Exactly ONE lane per warp may do the accumulating, and choosing
    //         the wrong one fails silently: pick a lane that is not active in
    //         this region and both counters stay at zero while the program
    //         still prints a ratio. Which lane is guaranteed to be active?
    // YOUR CODE HERE
    (void)c;
}

__global__ void kDiverge(const int * __restrict__ flag, float *out,
                         unsigned long long *c, int n, int mode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // mode 0: 2-way, lane-varying      mode 1: warp-uniform
    // mode 2: 1 lane in 32 takes the costly path
    int p;
    if      (mode == 0) p = (i & 1);
    else if (mode == 1) p = ((i >> 5) & 1);
    else                p = ((i & 31) == 7);
    float v = (float)flag[i];
    if (p) { v = v * 1.25f + 1.0f; recordIssue(c); }
    else   { v = v * 0.75f - 1.0f; if (mode != 2) recordIssue(c); }
    out[i] = v;
}

// =============================================================================
// TODO 5 — the closed-form predictions. No lane enumeration allowed.
// =============================================================================

// Sectors touched by one warp executing `in[baseFloats + strideFloats*lane]`,
// for a 4-byte element and a 32 B-aligned base pointer.
//
// Two regimes. Once the byte stride reaches the sector size every lane owns a
// distinct sector and the answer saturates at 32. Below that the lanes cover a
// contiguous byte RANGE, and the count is the number of sector boundaries that
// range crosses, plus one -- which is where the misalignment term lives.
static int predictSectorsStride(int strideFloats, int baseFloats)
{
    // TODO 5a: CLOSED FORM. Sectors touched by one warp executing
    //            in[baseFloats + strideFloats*lane]
    //          for a 4-byte element and a 32 B-aligned base pointer.
    //          No loop over lanes. There are two regimes; find where they meet,
    //          and note that `baseFloats` can push the covered byte range
    //          across one extra sector boundary.
    // YOUR CODE HERE
    (void)strideFloats; (void)baseFloats;
    return 0;
}

static int predictDegreeStride(int strideWords)
{
    // TODO 5b: CLOSED FORM. Conflict degree of `s[strideWords * tid]` for a
    //          4-byte access across 32 banks. No loop over lanes.
    //          Which lanes land in the same bank, and are the WORDS they ask
    //          that bank for the same or different? The answer is a standard
    //          number-theoretic function of strideWords and 32.
    // YOUR CODE HERE
    (void)strideWords;
    return 0;
}

static int predictDegreeDivide(int k)
{
    // TODO 5c: CLOSED FORM. Conflict degree of `s[tid / k]`, 1 <= k <= 32.
    //          This is NOT of the form s[stride*tid], so 5b does not apply.
    //          How many DISTINCT words does the warp ask for, and how are they
    //          spread over the banks? The answer does not depend on k, and the
    //          by-lane column of the printed table will tell you whether you
    //          have it right.
    // YOUR CODE HERE
    (void)k;
    return 0;
}

// =============================================================================
// Harness
// =============================================================================
static const Pattern hPatG[8] = {
    { "in[i]",            1,  0 },
    { "in[i + 1]",        1,  1 },
    { "in[2*i]",          2,  0 },
    { "in[4*i]",          4,  0 },
    { "in[8*i]",          8,  0 },
    { "in[16*i]",        16,  0 },
    { "in[0]",            0,  0 },
    { "in[2*i + 1]",      2,  1 },
};
static const char *shExpr[8] = {
    "s[tid]", "s[2*tid]", "s[3*tid]", "s[4*tid]",
    "s[8*tid]", "s[32*tid]", "s[tid/2]", "s[0]"
};
static const int shStride[8] = { 1, 2, 3, 4, 8, 32, -2, 0 };   // -k means tid/k

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 23 / Exercise 2 - derive the counters yourself ===\n");
    printf("ncu status: ERR_NVGPUCTRPERM. Every quantity below is one the tool\n"
           "would report; here the kernel computes it about itself.\n\n");

    int score = 0, total = 0;

    // Shipped with the TODOs unfilled every predictor returns 0. Say so and
    // stop, rather than printing a table of zeros that looks like a result.
    if (predictSectorsStride(1, 0) == 0 && predictDegreeStride(1) == 0) {
        printf("Set TODO 5 first (and TODOs 1-4).\n");
        return 0;
    }

    // ------------------------------------------------------------ sectors
    unsigned *dOut; CHECK(cudaMalloc(&dOut, 32 * sizeof(unsigned)));
    float *dIn;     CHECK(cudaMalloc(&dIn, NGLOBAL * sizeof(float)));
    CHECK(cudaMemset(dIn, 0, NGLOBAL * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(dPatG, hPatG, sizeof(hPatG)));
    CHECK(cudaMemset(dOut, 0, 32 * sizeof(unsigned)));
    kSectors<<<1, 32>>>(dIn, dOut, 8);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    unsigned hOut[32]; CHECK(cudaMemcpy(hOut, dOut, 32 * sizeof(unsigned), cudaMemcpyDeviceToHost));

    printf("-- A. sectors per request ------------------------------------------\n");
    printf("   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum\n");
    printf("   / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum\n\n");
    printf("   %-14s %10s %10s %12s %10s\n",
           "expression", "measured", "predicted", "bytes/sector", "used/moved");
    for (int p = 0; p < 8; ++p) {
        int pred = predictSectorsStride(hPatG[p].stride, hPatG[p].base);
        double moved = 32.0 * (double)hOut[p];
        double bps   = 128.0 / (double)hOut[p];
        int good = ((int)hOut[p] == pred);
        // A broadcast asks for 4 distinct bytes, not 128, so the "used/moved"
        // column is not defined for it. Printing 400% would be a lie of the
        // same shape as calling a broadcast 12.5% efficient.
        if (hPatG[p].stride == 0)
            printf("   %-14s %10u %10d %12.2f %10s%s\n",
                   hPatG[p].expr, hOut[p], pred, bps, "broadcast",
                   good ? "" : "   <-- MISMATCH");
        else
            printf("   %-14s %10u %10d %12.2f %9.1f%%%s\n",
                   hPatG[p].expr, hOut[p], pred, bps, 100.0 * 128.0 / moved,
                   good ? "" : "   <-- MISMATCH");
        score += good; ++total;
    }
    printf("\n   `bytes/sector` is ncu's\n"
           "   smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio;\n"
           "   32.00 is perfect, 4.00 means seven eighths of every sector moved\n"
           "   for this instruction is discarded. `in[i+1]` is the row to stare\n"
           "   at: four bytes of misalignment costs a QUARTER more sectors (5\n"
           "   instead of 4), on every load, forever. Module 5 predicted 80%%\n"
           "   efficiency for it and measured 97.9%%, because the neighbouring\n"
           "   warp's boundary sector was already in cache -- the instruction\n"
           "   counter and the traffic counter disagree, and both are right.\n\n");

    // ------------------------------------------------------------ bank conflicts
    unsigned *dDeg, *dLane; float *dSink;
    CHECK(cudaMalloc(&dDeg,  32 * sizeof(unsigned)));
    CHECK(cudaMalloc(&dLane, 32 * sizeof(unsigned)));
    CHECK(cudaMalloc(&dSink, 16));
    kBanks<<<1, 32>>>(dDeg, dLane, dSink, 8);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    unsigned hDeg[32], hLane[32];
    CHECK(cudaMemcpy(hDeg,  dDeg,  32 * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hLane, dLane, 32 * sizeof(unsigned), cudaMemcpyDeviceToHost));

    printf("-- B. shared-memory bank conflicts ---------------------------------\n");
    printf("   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum\n");
    printf("   l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum\n\n");
    printf("   %-12s %9s %10s %12s %11s %10s\n",
           "expression", "degree", "predicted", "wavefronts", "conflicts", "by-lane");
    for (int p = 0; p < 8; ++p) {
        int pred = (shStride[p] < 0) ? predictDegreeDivide(-shStride[p])
                                     : predictDegreeStride(shStride[p]);
        int good = ((int)hDeg[p] == pred);
        printf("   %-12s %9u %10d %12u %11u %10u%s\n",
               shExpr[p], hDeg[p], pred, hDeg[p], hDeg[p] - 1u, hLane[p],
               good ? "" : "   <-- MISMATCH");
        score += good; ++total;
    }
    printf("\n   The last column is the same kernel counting LANES per bank\n"
           "   instead of distinct words. It disagrees on exactly the two rows\n"
           "   where the hardware broadcasts -- `s[tid/2]` and `s[0]` -- and in\n"
           "   both cases it invents a conflict that costs nothing. The counter\n"
           "   counts words. Note also that the Ada cost law is max(2, D) for a\n"
           "   4 B access (M7), so `s[2*tid]` reports one conflict per request\n"
           "   and costs zero cycles; on LDS.128 the floor of 2 is gone and the\n"
           "   cost really is proportional to D (M18).\n\n");

    // ------------------------------------------------------------ occupancy
    printf("-- C. achieved occupancy, both denominators ------------------------\n");
    printf("   sm__warps_active.avg.pct_of_peak_sustained_active   (\"Achieved\")\n"
           "   sm__warps_active.avg.pct_of_peak_sustained_elapsed\n\n");

    int blocksPerSM = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kOccProbe, 256, 0));
    const int oneWave = blocksPerSM * NSM;
    printf("   256 threads/block -> %d blocks/SM -> theoretical %.1f%%, 1 wave = %d blocks\n\n",
           blocksPerSM, 100.0 * blocksPerSM * 8 / WARP_SLOTS_PER_SM, oneWave);

    unsigned long long *dRes, *dS, *dE;
    CHECK(cudaMalloc(&dRes, NSM * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&dS,   NSM * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&dE,   NSM * sizeof(unsigned long long)));
    int *dWork; CHECK(cudaMalloc(&dWork, (oneWave + 1) * sizeof(int)));
    int *hWork = (int*)malloc((oneWave + 1) * sizeof(int));
    unsigned long long hRes[NSM], hS[NSM], hE[NSM], big[NSM];

    struct { const char *name; int grid; int imbalanced; } ocfg[3] = {
        { "uniform, exactly 1 wave", oneWave,     0 },
        { "uniform, 1 wave + 1 blk", oneWave + 1, 0 },
        { "1..8x imbalance, 1 wave", oneWave,     1 },
    };
    printf("   %-26s %7s %12s %12s %9s\n",
           "configuration", "blocks", "occ_active", "occ_elapsed", "ms");
    double occA[3], occE[3];
    for (int c = 0; c < 3; ++c) {
        for (int b = 0; b < ocfg[c].grid; ++b)
            hWork[b] = ocfg[c].imbalanced ? (2667 * (1 + (b % 8))) : 12000;
        CHECK(cudaMemcpy(dWork, hWork, ocfg[c].grid * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dRes, 0, NSM * sizeof(unsigned long long)));
        CHECK(cudaMemset(dE,   0, NSM * sizeof(unsigned long long)));
        for (int i = 0; i < NSM; ++i) big[i] = ~0ull;
        CHECK(cudaMemcpy(dS, big, NSM * sizeof(unsigned long long), cudaMemcpyHostToDevice));

        cudaEvent_t a, b2; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b2));
        CHECK(cudaEventRecord(a));
        kOccProbe<<<ocfg[c].grid, 256>>>(dWork, dSink, dRes, dS, dE);
        CHECK(cudaEventRecord(b2)); CHECK(cudaEventSynchronize(b2));
        float ms; CHECK(cudaEventElapsedTime(&ms, a, b2));
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b2));
        CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(hRes, dRes, NSM * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hS,   dS,   NSM * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hE,   dE,   NSM * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        reduceOccupancy(hRes, hS, hE, &occA[c], &occE[c]);
        printf("   %-26s %7d %11.1f%% %11.1f%% %8.3f\n",
               ocfg[c].name, ocfg[c].grid, occA[c], occE[c], ms);
    }
    // The structural facts M19 established, now re-derived here.
    int c1 = (fabs(occA[1] - occA[0]) < fabs(occE[1] - occE[0]));   // tail: elapsed moves more
    int c2 = (occE[2] < occE[0] - 5.0);                             // imbalance crushes elapsed
    int c3 = (occA[2] > occA[0]);                                   // ...and does not lower active
    printf("\n   [%s] a tail moves occ_elapsed more than occ_active\n", c1 ? "x" : " ");
    printf("   [%s] an imbalance drops occ_elapsed by more than 5 points\n", c2 ? "x" : " ");
    printf("   [%s] the same imbalance does NOT drop occ_active\n", c3 ? "x" : " ");
    score += c1 + c2 + c3; total += 3;
    printf("\n   Only occ_active is in ncu's Occupancy section; the _elapsed form\n"
           "   exists but you must ask for it with --metrics. The tutorial metric\n"
           "   is the blind one.\n\n");

    // ------------------------------------------------------------ lane efficiency
    printf("-- D. lane efficiency ----------------------------------------------\n");
    printf("   smsp__thread_inst_executed_per_inst_executed.ratio\n\n");
    {
        const int n = 1 << 20;
        int *dFlag; float *dOutF; unsigned long long *dC;
        CHECK(cudaMalloc(&dFlag, n * sizeof(int)));
        CHECK(cudaMalloc(&dOutF, n * sizeof(float)));
        CHECK(cudaMalloc(&dC,    2 * sizeof(unsigned long long)));
        int   *hFlag = (int*)malloc(n * sizeof(int));
        float *hOutF = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; ++i) hFlag[i] = (int)((i * 1103515245u + 12345u) & 255u);
        CHECK(cudaMemcpy(dFlag, hFlag, n * sizeof(int), cudaMemcpyHostToDevice));

        const char *mname[3] = { "p = i & 1        (2-way)",
                                 "p = (i>>5) & 1   (uniform)",
                                 "p = (i&31)==7    (1 of 32)" };
        const double want[3] = { 16.0, 32.0, 1.0 };
        printf("   %-28s %14s %12s %10s\n",
               "predicate", "thr_inst/inst", "predicted", "lane eff");
        for (int mode = 0; mode < 3; ++mode) {
            CHECK(cudaMemset(dC, 0, 2 * sizeof(unsigned long long)));
            kDiverge<<<(n + 255) / 256, 256>>>(dFlag, dOutF, dC, n, mode);
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            unsigned long long hC[2];
            CHECK(cudaMemcpy(hC, dC, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            double r = (double)hC[0] / (double)hC[1];
            int good = (fabs(r - want[mode]) < 0.01);
            printf("   %-28s %14.2f %12.2f %9.1f%%%s\n",
                   mname[mode], r, want[mode], 100.0 * r / 32.0,
                   good ? "" : "   <-- MISMATCH");
            score += good; ++total;
        }
        // numerical validation, separate pass, on the last mode run
        CHECK(cudaMemcpy(hOutF, dOutF, n * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            int p = ((i & 31) == 7);
            float v = (float)hFlag[i];
            float ref = p ? (v * 1.25f + 1.0f) : (v * 0.75f - 1.0f);
            if (fabsf(hOutF[i] - ref) > 1e-5f * fmaxf(1.0f, fabsf(ref))) ++bad;
        }
        printf("   numerical check of kDiverge: %d mismatches\n", bad);
        int good = (bad == 0); score += good; ++total;

        printf("\n   Mode 0 issues BOTH bodies with 16 lanes on in each, so the\n"
               "   average over issued instructions is (16+16)/2 = 16.00 and the\n"
               "   lane efficiency is 50%%. That is real divergence cost.\n"
               "   Mode 2 counts only the rare side, where 1 lane in 32 is on, and\n"
               "   reports 1.00 -- 3.1%% lane efficiency for a kernel that is\n"
               "   almost entirely fine. The ratio is an average over ISSUED\n"
               "   INSTRUCTIONS, not over time, so a cheap rare branch sinks it\n"
               "   while costing almost nothing. Never read lane efficiency\n"
               "   without the instruction count it is averaged over.\n"
               "   What this ratio cannot see ALONE is predication. ncu gives you\n"
               "   a companion metric for exactly that:\n"
               "     smsp__thread_inst_executed_pred_on_per_inst_executed.ratio\n"
               "   (\"Avg. Not Predicated Off Threads Per Warp\"). A predicated-off\n"
               "   lane is ACTIVE but not predicated-on, so it is counted by the\n"
               "   first metric and not by the second; the GAP between the two is\n"
               "   exactly the predication, while a real branch lowers BOTH.\n"
               "   Module 8 proved __activemask() alone cannot make that\n"
               "   distinction. It cannot -- but this PAIR of counters can.\n");

        free(hFlag); free(hOutF);
        CHECK(cudaFree(dFlag)); CHECK(cudaFree(dOutF)); CHECK(cudaFree(dC));
    }

    free(hWork);
    CHECK(cudaFree(dRes)); CHECK(cudaFree(dS)); CHECK(cudaFree(dE)); CHECK(cudaFree(dWork));
    CHECK(cudaFree(dDeg)); CHECK(cudaFree(dLane)); CHECK(cudaFree(dSink));
    CHECK(cudaFree(dOut)); CHECK(cudaFree(dIn));

    printf("\nSCORE: %d/%d\n", score, total);
    printf("OVERALL: %s\n", (score == total) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (score == total) ? 0 : 1;
}
