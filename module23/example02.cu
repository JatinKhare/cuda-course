// =============================================================================
// Module 23 / Example 2 — reconstructing the three sections you read FIRST:
//                         Speed of Light, Occupancy, Warp State Statistics.
//
// `ncu` fails with ERR_NVGPUCTRPERM on this machine, so this program builds the
// sections rather than reading them. Every metric name printed is the real one.
//
//   A  SPEED OF LIGHT
//        sm__throughput.avg.pct_of_peak_sustained_elapsed      ("Compute (SM)")
//        gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed ("Memory")
//      Four kernels, one per quadrant of the (compute, memory) plane. This is
//      the section that decides which OTHER section you open.
//
//   B  OCCUPANCY
//        sm__warps_active.avg.pct_of_peak_sustained_active   <- "Achieved"
//        sm__warps_active.avg.pct_of_peak_sustained_elapsed
//      Module 19's instrument (clock64 + %smid), showing the measured fact
//      that the first denominator is structurally blind to tails and imbalance.
//
//   C  WARP STATE STATISTICS
//        smsp__thread_inst_executed_per_inst_executed.ratio  <- lane efficiency
//        smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio
//      The first is Module 8's lane efficiency, which M8 computed by hand; it
//      is reconstructed here exactly. The stall-reason family is given as a
//      table with the "does occupancy help?" column from Module 20.
//
// BUILD: nvcc -arch=sm_89 -O3 -lineinfo -o example02.exe example02.cu
// RUN  : example02.exe
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

#define NSM                 40
#define WARP_SLOTS_PER_SM   48
#define DRAM_PIN_PEAK_GBS  432.0

#define NFLOAT   ((size_t)64*1024*1024)        // 256 MB, 5.3x the 48 MB L2
#define CHASE_N  ((size_t)64*1024*1024)
#define CHASE_P  2097169u                      // prime stride, full cycle on 2^26
#define CHASE_HOPS 2048

// =============================================================================
// Part A kernels — one per Speed-of-Light quadrant.
// =============================================================================

// (1) memory high, compute ~0.
__global__ void kStream(const float4 * __restrict__ in, float *sink, size_t n4)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float4 a = make_float4(0.f,0.f,0.f,0.f);
    for (; i < n4; i += gridDim.x * (size_t)blockDim.x) {
        float4 v = in[i];
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    if (a.x == 1e30f) sink[0] = a.x + a.y + a.z + a.w;
}

// (2) compute high, memory ~0.  Eight independent chains (M20: L/T = 3.8, so
//     four in flight per scheduler already saturates), outer loop unrolled 8
//     deep so loop overhead is not the thing being measured (M21 rule).
#define CHAINS 8
__global__ void kFfma(float *sink, int iters)
{
    float a[CHAINS]; const float b = 1.0000001f;
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) a[i] = (float)(threadIdx.x + i);
    #pragma unroll 8
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int i = 0; i < CHAINS; ++i) a[i] = fmaf(a[i], b, 1.0f);
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) s += a[i];
    if (s == 1e30f) sink[0] = s;
}

// (3) BOTH low: two shared-memory operand loads per FMA, which is Module 17's
//     tiled GEMM inner loop with the GEMM removed. Capped by the on-chip
//     operand-fetch ceiling at 7-13% of the FP32 plateau no matter what.
#define SH_N 2048
__global__ void kOperandBound(float *sink, int iters)
{
    __shared__ float s[SH_N];
    for (int i = threadIdx.x; i < SH_N; i += blockDim.x) s[i] = (float)(i & 255);
    __syncthreads();
    float acc = 0.0f;
    int base = (int)threadIdx.x;
    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            float x = s[(base + u*33)      & (SH_N-1)];
            float y = s[(base + u*33 + 97) & (SH_N-1)];
            acc = fmaf(x, y, acc);
        }
        base += 1;                       // defeat loop-invariant hoisting
    }
    if (acc == 1e30f) sink[0] = acc + s[0];
}

// (4) NOTHING saturated: a dependent pointer walk. One outstanding load per
//     thread, so Little's Law, not the roofline, is the governing equation.
__global__ void kChase(const unsigned * __restrict__ idx, unsigned *sink, int hops)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned p = (tid * 7919u) & (unsigned)(CHASE_N - 1);
    for (int h = 0; h < hops; ++h) p = idx[p];
    if (p == 0xffffffffu) sink[0] = p;
}
__global__ void kChaseInit(unsigned *idx, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    for (; i < n; i += gridDim.x * (size_t)blockDim.x)
        idx[i] = (unsigned)((i + CHASE_P) & (n - 1));
}

// =============================================================================
// Part B — Module 19's achieved-occupancy instrument.
//
// Per warp: timestamp entry and exit with clock64(), find the SM with %smid,
// and reduce three quantities per SM. Every quantity has to be per-SM because
// the per-SM cycle counters are NOT synchronised (M19 measured 298 M cycles of
// offset inside one launch), so a global min/max over raw timestamps is
// meaningless.
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
    if ((threadIdx.x & 31u) == 0u) {               // one reporter per warp
        atomicAdd(&resident[sm], t1 - t0);
        atomicMin(&smStart[sm], t0);
        atomicMax(&smEnd[sm],   t1);
    }
}

// =============================================================================
// Part C — lane efficiency, reconstructed exactly.
//
//   smsp__thread_inst_executed_per_inst_executed.ratio
//     = (threads that executed an instruction) / (instructions issued)
//   lane efficiency = that / 32.
//
// Module 8 computed this by hand for its divergent kernels (47.7% and 56%).
// Here the kernel counts it for itself with __popc(__activemask()).
// =============================================================================
__device__ __forceinline__ void record(unsigned long long *c)
{
    unsigned m = __activemask();
    int lead = __ffs(m) - 1;
    if ((int)(threadIdx.x & 31u) == lead) {
        atomicAdd(&c[0], (unsigned long long)__popc(m));   // thread_inst_executed
        atomicAdd(&c[1], 1ull);                            // inst_executed
    }
}
__global__ void kDivergent(const int * __restrict__ flag, float *out,
                           unsigned long long *c, int n, int mode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // mode 0: predicate varies per lane   -> 2-way divergence
    // mode 1: predicate is warp-uniform   -> no divergence
    int p = (mode == 0) ? (i & 1) : ((i >> 5) & 1);
    float v = (float)flag[i];
    if (p) { v = v * 1.25f + 1.0f; record(c); }
    else   { v = v * 0.75f - 1.0f; record(c); }
    out[i] = v;
}

// =============================================================================
// Harness
// =============================================================================
static float4 *gBig4; static float *gBig; static float *gSink;
static unsigned *gIdx, *gUSink;

static double timeLaunch(void (*f)(int), int arg, int iters)
{
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) f(arg);
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms / iters;
}
static void lStream (int a) { (void)a; kStream<<<640,256>>>(gBig4, gSink, NFLOAT/4); }
static void lFfma   (int a) { kFfma<<<480,128>>>(gSink, a); }
static void lOperand(int a) { kOperandBound<<<480,128>>>(gSink, a); }
static void lChase  (int a) { kChase<<<1,32>>>(gIdx, gUSink, a); }   // ONE warp, by design

static double minOf(void (*f)(int), int arg, int sweeps)
{
    double t = timeLaunch(f, arg, 1);
    int n = (int)(10.0/(t > 1e-3 ? t : 1e-3)); if (n < 3) n = 3; if (n > 100) n = 100;
    double best = 1e30;
    for (int s = 0; s < sweeps; ++s) { double x = timeLaunch(f, arg, n); if (x < best) best = x; }
    return best;
}
static void warmFor(float ms, void (*f)(int), int arg)
{
    cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
    float el = 0.f; CHECK(cudaEventRecord(w0));
    while (el < ms) { f(arg);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el, w0, w1)); }
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 23 / Example 2 - Speed of Light, Occupancy, Warp State ===\n");
    printf("ncu status here: ERR_NVGPUCTRPERM. Sections reconstructed, metric\n"
           "names real.\n\n");
    int ok = 1;

    CHECK(cudaMalloc(&gBig,  NFLOAT*sizeof(float)));
    CHECK(cudaMalloc(&gIdx,  CHASE_N*sizeof(unsigned)));
    CHECK(cudaMalloc(&gSink, 16));
    CHECK(cudaMalloc(&gUSink, 16));
    gBig4 = (float4*)gBig;
    CHECK(cudaMemset(gBig, 1, NFLOAT*sizeof(float)));
    kChaseInit<<<640,256>>>(gIdx, CHASE_N);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

    // ---- group 1: off-chip axis, 1500 ms streaming warm-up ------------------
    printf("-- warming 1500 ms streaming ---------------------------------------\n");
    warmFor(1500.f, lStream, 0);
    double tStream = minOf(lStream, 0, 4);
    double tChase  = minOf(lChase, CHASE_HOPS, 4);

    // ---- group 2: on-chip axis, 500 ms compute warm-up ----------------------
    printf("-- warming 500 ms compute ------------------------------------------\n");
    warmFor(500.f, lFfma, 2000);
    const int FFMA_IT = 8192, OPER_IT = 4096;
    double tFfma    = minOf(lFfma,    FFMA_IT, 3);
    double tOperand = minOf(lOperand, OPER_IT, 3);
    CHECK(cudaGetLastError());

    // ---- ceilings measured in this same program -----------------------------
    const double dramCeil = (double)(NFLOAT*sizeof(float)) / (tStream*1e-3) / 1e9;
    const double fp32Ceil = 480.0*128.0*(double)FFMA_IT*CHAINS*2.0 / (tFfma*1e-3) / 1e9;

    // ---- per-kernel achieved rates -----------------------------------------
    struct Row { const char *name; double dramGBs, gflops; const char *note; } row[4];
    row[0].name="kStream (640,256)";
    row[0].dramGBs = (double)(NFLOAT*sizeof(float))/(tStream*1e-3)/1e9;
    row[0].gflops  = (double)NFLOAT/(tStream*1e-3)/1e9;   // one FADD per float
    row[0].note    = "memory bound";
    row[1].name="kFfma (480,128)";
    row[1].dramGBs = 0.0;
    row[1].gflops  = fp32Ceil;
    row[1].note    = "compute bound";
    row[2].name="kOperandBound (480,128)";
    row[2].dramGBs = 0.0;
    row[2].gflops  = 480.0*128.0*(double)OPER_IT*16.0*2.0/(tOperand*1e-3)/1e9;
    row[2].note    = "on-chip operand fetch";
    row[3].name="kChase (1,32)";
    // one 32 B sector per hop, 32 threads -> 1.0 kB in flight (Module 21)
    row[3].dramGBs = 32.0*(double)CHASE_HOPS*32.0/(tChase*1e-3)/1e9;
    row[3].gflops  = 0.0;
    row[3].note    = "LATENCY - nothing saturated";

    printf("\n-- A. Speed of Light -----------------------------------------------\n");
    printf("   ceilings measured in this run: DRAM %.1f GB/s, FP32 %.0f GFLOP/s\n\n",
           dramCeil, fp32Ceil);
    printf("   %-24s %10s %10s   %s\n", "kernel", "Compute%", "Memory%", "verdict");
    for (int i = 0; i < 4; ++i)
        printf("   %-24s %9.1f%% %9.1f%%   %s\n", row[i].name,
               100.0*row[i].gflops/fp32Ceil, 100.0*row[i].dramGBs/DRAM_PIN_PEAK_GBS,
               row[i].note);
    printf("\n   The decision tree the real section exists to feed:\n"
           "     Memory high, Compute low   -> MemoryWorkloadAnalysis\n"
           "     Compute high, Memory low   -> instruction mix; you are near done\n"
           "     BOTH mid                   -> the binding ceiling is on chip,\n"
           "                                   not at the pins (Module 21)\n"
           "     BOTH low                   -> nothing is saturated. STOP reading\n"
           "                                   the roofline; open WarpStateStatistics\n"
           "                                   and apply Little's Law (Module 20).\n"
           "   Row 4 is the row everybody misdiagnoses. `ncu` will label a\n"
           "   dependent pointer walk with a Memory Throughput near 1%% and a\n"
           "   reader who has only learned \"higher is better\" concludes \"not\n"
           "   memory bound, so compute bound\". It is bound by neither.\n");

    // ------------------------------------------------------------------ Part B
    printf("\n-- B. Occupancy ----------------------------------------------------\n");
    printf("   sm__warps_active.avg.pct_of_peak_sustained_active   (\"Achieved\")\n"
           "   sm__warps_active.avg.pct_of_peak_sustained_elapsed\n\n");

    int blocksPerSM = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kOccProbe, 256, 0));
    const int warpsPerBlock = 256/32;
    const double theoretical = 100.0 * blocksPerSM * warpsPerBlock / WARP_SLOTS_PER_SM;
    const int oneWave = blocksPerSM * NSM;
    printf("   256 threads/block -> %d blocks/SM -> theoretical occupancy %.1f%%\n",
           blocksPerSM, theoretical);
    printf("   one wave = %d blocks\n\n", oneWave);

    unsigned long long *dRes, *dS, *dE;
    CHECK(cudaMalloc(&dRes, NSM*sizeof(unsigned long long)));
    CHECK(cudaMalloc(&dS,   NSM*sizeof(unsigned long long)));
    CHECK(cudaMalloc(&dE,   NSM*sizeof(unsigned long long)));
    int *dWork; CHECK(cudaMalloc(&dWork, (oneWave+1)*sizeof(int)));
    int *hWork = (int*)malloc((oneWave+1)*sizeof(int));
    unsigned long long hRes[NSM], hS[NSM], hE[NSM];

    struct { const char *name; int grid; int imbalanced; } ocfg[3] = {
        { "uniform, exactly 1 wave", oneWave,     0 },
        { "uniform, 1 wave + 1 blk", oneWave + 1, 0 },
        { "1..8x imbalance, 1 wave", oneWave,     1 },
    };
    printf("   %-26s %7s %12s %12s %10s\n",
           "configuration", "blocks", "occ_active", "occ_elapsed", "ms");
    double occAct[3], occEla[3];
    for (int c = 0; c < 3; ++c) {
        for (int b = 0; b < ocfg[c].grid; ++b)
            hWork[b] = ocfg[c].imbalanced ? (2667 * (1 + (b % 8))) : 12000;
        CHECK(cudaMemcpy(dWork, hWork, ocfg[c].grid*sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dRes, 0, NSM*sizeof(unsigned long long)));
        CHECK(cudaMemset(dE,   0, NSM*sizeof(unsigned long long)));
        { unsigned long long big[NSM]; for (int i = 0; i < NSM; ++i) big[i] = ~0ull;
          CHECK(cudaMemcpy(dS, big, NSM*sizeof(unsigned long long), cudaMemcpyHostToDevice)); }

        cudaEvent_t a,b2; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b2));
        CHECK(cudaEventRecord(a));
        kOccProbe<<<ocfg[c].grid,256>>>(dWork, gSink, dRes, dS, dE);
        CHECK(cudaEventRecord(b2)); CHECK(cudaEventSynchronize(b2));
        float ms; CHECK(cudaEventElapsedTime(&ms, a, b2));
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b2));
        CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(hRes, dRes, NSM*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hS,   dS,   NSM*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hE,   dE,   NSM*sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        unsigned long long maxSpan = 0;
        for (int i = 0; i < NSM; ++i)
            if (hE[i] > hS[i]) { unsigned long long sp = hE[i]-hS[i]; if (sp > maxSpan) maxSpan = sp; }
        double sumA = 0.0, sumE = 0.0; int used = 0;
        for (int i = 0; i < NSM; ++i) {
            if (hE[i] <= hS[i]) continue;
            double span = (double)(hE[i]-hS[i]);
            sumA += (double)hRes[i] / (WARP_SLOTS_PER_SM * span);
            sumE += (double)hRes[i] / (WARP_SLOTS_PER_SM * (double)maxSpan);
            ++used;
        }
        occAct[c] = used ? 100.0*sumA/used : 0.0;
        occEla[c] = used ? 100.0*sumE/NSM  : 0.0;
        printf("   %-26s %7d %11.1f%% %11.1f%% %9.3f\n",
               ocfg[c].name, ocfg[c].grid, occAct[c], occEla[c], ms);
    }
    printf("\n   Read the two columns against each other. Adding ONE block to a\n"
           "   full wave barely moves occ_active and visibly drops occ_elapsed:\n"
           "   the tail is a second wave in which 39 of 40 SMs are idle, and the\n"
           "   per-SM-busy denominator simply does not count idle SMs. The 1..8x\n"
           "   imbalance is worse: occ_active moves by a couple of points in\n"
           "   either direction while occ_elapsed collapses, because an SM\n"
           "   that finishes early stops contributing to its OWN denominator\n"
           "   but still counts as idle machine in the whole-kernel one.\n"
           "   Module 19 measured a 1..8x imbalance moving occ_active UP by\n"
           "   3.4 points while the kernel got 53%% slower.\n"
           "   The metric every tutorial quotes is `..._pct_of_peak_sustained_\n"
           "   ACTIVE`. It is the one that cannot see either effect.\n"
           "   Second fact, also Module 19's: a BARRIER or a DRAM stall does not\n"
           "   lower achieved occupancy at all. A stalled warp is still resident.\n"
           "   That is how 100%% occupancy and 6%% of peak coexist.\n");

    // ------------------------------------------------------------------ Part C
    printf("\n-- C. Warp State Statistics ----------------------------------------\n");
    printf("   smsp__thread_inst_executed_per_inst_executed.ratio\n\n");
    {
        const int n = 1 << 20;
        int *dFlag; float *dOut; unsigned long long *dC;
        CHECK(cudaMalloc(&dFlag, n*sizeof(int)));
        CHECK(cudaMalloc(&dOut,  n*sizeof(float)));
        CHECK(cudaMalloc(&dC,    2*sizeof(unsigned long long)));
        int *hFlag = (int*)malloc(n*sizeof(int));
        float *hOut = (float*)malloc(n*sizeof(float));
        for (int i = 0; i < n; ++i) hFlag[i] = (int)((i*1103515245u + 12345u) & 255u);
        CHECK(cudaMemcpy(dFlag, hFlag, n*sizeof(int), cudaMemcpyHostToDevice));

        printf("   %-34s %14s %12s\n", "kernel", "thr_inst/inst", "lane eff");
        double le[2];
        for (int mode = 0; mode < 2; ++mode) {
            CHECK(cudaMemset(dC, 0, 2*sizeof(unsigned long long)));
            kDivergent<<<(n+255)/256,256>>>(dFlag, dOut, dC, n, mode);
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            unsigned long long hC[2];
            CHECK(cudaMemcpy(hC, dC, 2*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            double r = (double)hC[0]/(double)hC[1];
            le[mode] = 100.0*r/32.0;
            printf("   %-34s %14.2f %11.1f%%\n",
                   mode == 0 ? "predicate (i & 1), lane-varying"
                             : "predicate (i>>5) & 1, warp-uniform", r, le[mode]);
            // validate against the hand-computed value
            double want = (mode == 0) ? 16.0 : 32.0;
            if (fabs(r - want) > 0.01) { printf("      <-- expected %.0f\n", want); ok = 0; }
        }
        // numerical validation of the kernel itself, separate pass
        CHECK(cudaMemcpy(hOut, dOut, n*sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            int p = ((i >> 5) & 1);
            float v = (float)hFlag[i];
            float ref = p ? (v*1.25f + 1.0f) : (v*0.75f - 1.0f);
            if (fabsf(hOut[i]-ref) > 1e-5f*fmaxf(1.0f, fabsf(ref))) ++bad;
        }
        printf("   numerical check of kDivergent (mode 1): %d mismatches\n", bad);
        if (bad) ok = 0;
        printf("\n   Module 8 computed exactly this ratio by hand (47.7%% for its\n"
               "   control-flow shape, 56%% for its naive work mapping). The\n"
               "   counter is not telling you something new; it is telling you\n"
               "   the thing you already know how to derive, for a kernel too\n"
               "   large to derive it for.\n"
               "   Caveat, from Module 8: this ratio CANNOT distinguish real\n"
               "   divergence from predication. A predicated-off lane is absent\n"
               "   from the active mask exactly as a not-taken lane is. Only the\n"
               "   SASS distinguishes them, and the fix is different in each case.\n");
        free(hFlag); free(hOut);
        CHECK(cudaFree(dFlag)); CHECK(cudaFree(dOut)); CHECK(cudaFree(dC));
    }

    printf("\n   The stall-reason family (Module 20's taxonomy, with the metric\n"
           "   names). `ncu --section WarpStateStatistics` prints these as\n"
           "   average cycles a warp spent stalled per issued instruction:\n\n");
    printf("   %-22s %-44s %s\n", "ncu reason", "what the warp is waiting for", "occupancy helps?");
    struct { const char *r; const char *w; const char *o; } st[8] = {
      {"long_scoreboard",  "a global/local load (L2 or DRAM)",            "yes"},
      {"short_scoreboard", "a shared-memory load, or MUFU",               "yes"},
      {"barrier",          "the slowest warp of its block",               "no - rebalance"},
      {"wait",             "a fixed-latency dependency (e.g. FFMA, 4 cyc)","yes, or ILP"},
      {"imc_miss",         "an immediate-constant cache miss",            "no"},
      {"no_instruction",   "the instruction cache",                       "no - shrink loop"},
      {"math_pipe_throttle","the pipe is full of other warps' work",      "NO - already full"},
      {"not_selected",     "another warp was chosen this cycle",          "NO - too MUCH"},
    };
    for (int i = 0; i < 8; ++i) printf("   %-22s %-44s %s\n", st[i].r, st[i].w, st[i].o);
    printf("\n   metric form: smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio\n"
           "   companion : smsp__warps_eligible.avg.per_cycle_active\n"
           "   The last two rows are the ones that invert the usual advice.\n"
           "   `not_selected` dominating means the schedulers are saturated and\n"
           "   MORE warps would make it worse; the fix is fewer instructions.\n"
           "   And `smsp__warps_eligible` near 0 with occupancy near 100%% is\n"
           "   precisely Module 19's point that resident is not eligible.\n");

    free(hWork);
    CHECK(cudaFree(dRes)); CHECK(cudaFree(dS)); CHECK(cudaFree(dE)); CHECK(cudaFree(dWork));
    CHECK(cudaFree(gBig)); CHECK(cudaFree(gIdx)); CHECK(cudaFree(gSink)); CHECK(cudaFree(gUSink));

    // Spec SS12 rules 5b and 13. The UPPER bounds are hardware: 432.0 GB/s at
    // the pins and 31,795 GFLOP/s at 3105 MHz (nvidia-smi clocks.max.sm). A
    // measurement above either one means the arithmetic is wrong, and that is a
    // FAIL. A measurement far below is a statement about this laptop's power
    // state -- this GPU was observed pinned in P8 for minutes at a time -- so it
    // warns instead. Parts B and C do not depend on the clock.
    int impossible = (dramCeil > DRAM_PIN_PEAK_GBS || fp32Ceil >= 31795.0);
    int healthy    = (dramCeil > 150.0 && fp32Ceil > 6000.0);
    printf("\n   sanity: DRAM %.1f GB/s, FP32 %.0f GFLOP/s : %s\n",
           dramCeil, fp32Ceil,
           impossible ? "ABOVE A HARDWARE BOUND - the probe is broken"
                      : (healthy ? "healthy operating point"
                                 : "LOW - GPU is power/thermally capped"));
    if (!healthy)
        printf("   WARNING: section A's ceilings were measured from a capped\n"
               "            operating point and are not this GPU's real ones.\n");
    if (impossible) ok = 0;
    printf("\nOVERALL: %s\n", ok ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return ok ? 0 : 1;
}
