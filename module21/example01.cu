// =============================================================================
// Module 21 / Example 1 — building the roofline for THIS GPU.
//
// GOAL : Measure every ceiling the model needs, in one program, and assemble
//        them into a hierarchical roofline that can be read off the screen.
//
//   A  Five measured ceilings, each checked against a hardware bound before it
//      is believed (spec SS12 rule 13):
//        - DRAM read bandwidth           (buffer >> the 48 MB L2)
//        - L2 read bandwidth             (working set inside the 48 MB L2)
//        - shared-memory read bandwidth, scalar LDS and vector LDS.128
//        - FP32 FFMA throughput
//        - instruction ISSUE rate, in warp-instructions per second
//   B  The ridge point of each memory ceiling against the compute ceiling.
//   C  An ASCII log-log roofline, with the ceilings drawn and known kernels
//      placed on it.
//   D  The same shared-memory probe with the loop-invariant address restored --
//      the Module 17 bug that reported four times the physical maximum. The
//      sanity check catches it. This is the part of the program that matters
//      most: a microbenchmark you have not bounded is a number, not a ceiling.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN  : example01.exe
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

// ---------------------------------------------------------------------------
// Hardware constants (spec SS1 and SS12). These produce the BOUNDS. Every
// measurement below is checked against the bound derived from them before it
// is allowed to become a ceiling.
// ---------------------------------------------------------------------------
#define SM_COUNT            40
#define SCHEDULERS_PER_SM    4          // 4 processing blocks, 1 scheduler each
#define FP32_LANES_PER_SM  128          // 4 x 32
#define SMEM_BYTES_PER_CYCLE_PER_SM 128 // 32 banks x 4 B
#define L2_BYTES      (48u*1024u*1024u)
#define DRAM_PIN_PEAK_GBS   432.0       // 192-bit @ 9.001 GHz, DDR
// The device's own maximum SM clock, from
//   nvidia-smi --query-gpu=clocks.max.sm --format=csv   ->  3105 MHz
// This is the only clock figure that makes a legitimate BOUND: nothing the
// SM does can be faster than its fastest possible clock. cudaDevAttrClockRate
// reports 1.545 GHz, which is neither the boost clock nor an upper bound, and
// spec SS12 rule 6 forbids using it for a percentage.
#define SM_CLOCK_MAX_GHZ    3.105
#define SM_CLOCK_SEEN_GHZ   2.04        // highest SM clock ever observed here

// Derived hardware bounds. Nothing measured may exceed these.
#define BOUND_DRAM_GBS      DRAM_PIN_PEAK_GBS
#define BOUND_SMEM_GBS      (SM_COUNT * (double)SMEM_BYTES_PER_CYCLE_PER_SM * SM_CLOCK_MAX_GHZ)
#define BOUND_FP32_GFLOPS   (SM_COUNT * (double)FP32_LANES_PER_SM * 2.0 * SM_CLOCK_MAX_GHZ)
#define BOUND_ISSUE_GIPS    (SM_COUNT * (double)SCHEDULERS_PER_SM * SM_CLOCK_MAX_GHZ)

// ---------------------------------------------------------------------------
// Probe kernels
// ---------------------------------------------------------------------------

// Streaming read. float4 so the loop is not instruction-bound; grid-stride so
// the grid size is a free parameter; the store is unreachable but the compiler
// cannot prove it, so nothing is eliminated.
__global__ void streamRead(const float4 * __restrict__ src, float *sink, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    for (; i < n; i += gridDim.x * (size_t)blockDim.x) {
        float4 v = src[i];
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    if (a.x == 1e30f) sink[0] = a.x + a.y + a.z + a.w;
}

// Same kernel, run REPS times over a small buffer. The buffer is the variable:
// 256 MB measures DRAM, 24 MB measures L2.
__global__ void streamReadRep(const float4 * __restrict__ src, float *sink,
                              size_t n, int reps)
{
    const size_t stride = gridDim.x * (size_t)blockDim.x;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int r = 0; r < reps; ++r) {
        for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n; i += stride) {
            float4 v = src[i];
            a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
        }
    }
    if (a.x == 1e30f) sink[0] = a.x + a.y + a.z + a.w;
}

// ---- shared-memory read bandwidth ----------------------------------------
// SPROBE_W reads per inner step. Stride 33 floats keeps the 32 lanes of a warp
// in 32 distinct banks (gcd(33,32) = 1, Module 7) AND keeps consecutive reads
// non-adjacent, so ptxas cannot contract them into LDS.128.
//
// `base += 1` at the bottom of the outer loop is the entire reason this number
// is honest. Without it every address is loop-invariant, the compiler hoists
// all SPROBE_W loads out of the timing loop, and the kernel reports a number
// that cannot physically happen. Part D runs exactly that version.
#define SPROBE_N 2048                   // 8 KB of shared memory
#define SPROBE_W 32
__global__ void sharedScalarProbe(float *sink, int iters)
{
    __shared__ float s[SPROBE_N];
    for (int i = threadIdx.x; i < SPROBE_N; i += blockDim.x) s[i] = (float)(i & 255);
    __syncthreads();

    float acc = 0.0f;
    int base = (int)threadIdx.x;
    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int u = 0; u < SPROBE_W; ++u) acc += s[(base + u*33) & (SPROBE_N-1)];
        base += 1;
    }
    if (acc == 1e30f) sink[0] = acc;
}

// The broken twin: identical except `base` never changes.
__global__ void sharedScalarProbeHoisted(float *sink, int iters)
{
    __shared__ float s[SPROBE_N];
    for (int i = threadIdx.x; i < SPROBE_N; i += blockDim.x) s[i] = (float)(i & 255);
    __syncthreads();

    float acc = 0.0f;
    const int base = (int)threadIdx.x;
    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int u = 0; u < SPROBE_W; ++u) acc += s[(base + u*33) & (SPROBE_N-1)];
    }
    if (acc == 1e30f) sink[0] = acc;
}

// Vector form. 16 B per lane; the access is phase-split into 4 phases of 8
// lanes, and 8 lanes x 16 B = 128 B is exactly one cycle of the bank array.
#define VPROBE_N 512                    // 512 float4 = 8 KB
#define VPROBE_W 16
__global__ void sharedVecProbe(float *sink, int iters)
{
    __shared__ float4 s[VPROBE_N];
    for (int i = threadIdx.x; i < VPROBE_N; i += blockDim.x)
        s[i] = make_float4((float)i, 1.f, 2.f, 3.f);
    __syncthreads();

    float acc = 0.0f;
    int base = (int)threadIdx.x;
    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int u = 0; u < VPROBE_W; ++u) {
            float4 v = s[(base + u*32) & (VPROBE_N-1)];
            acc += v.x + v.y + v.z + v.w;
        }
        base += 1;
    }
    if (acc == 1e30f) sink[0] = acc;
}

// ---- FP32 and issue ------------------------------------------------------
// Eight independent chains: enough instruction-level parallelism that the
// 4-cycle FFMA latency is fully hidden by one warp, so the measurement is a
// throughput measurement and not a latency measurement.
#define CHAINS 8
__global__ void ffmaProbe(float *sink, int iters)
{
    float a[CHAINS];
    const float b = 1.0000001f;
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) a[i] = (float)(threadIdx.x + i);
    // Unrolled 8 deep so the loop counter and branch are ~4% of the issued
    // instructions; otherwise the overhead caps the measurement around 73% of
    // the bound and you report a loop, not a pipeline.
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

// Same shape, but each chain step is FFMA followed by FADD. Under IEEE rules
// (a*b + 1) + b may not be re-associated into a*b + (1+b), so nvcc must emit
// both instructions. 2 instructions, 3 FLOPs. If this kernel issues at the
// same warp-instruction rate as ffmaProbe, the issue rate is the ceiling and
// the FLOP count is just how much work you chose to put in each slot.
__global__ void mixedProbe(float *sink, int iters)
{
    float a[CHAINS];
    const float b = 1.0000001f;
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) a[i] = (float)(threadIdx.x + i);
    #pragma unroll 8
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int i = 0; i < CHAINS; ++i) a[i] = fmaf(a[i], b, 1.0f) + b;
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) s += a[i];
    if (s == 1e30f) sink[0] = s;
}

// ---------------------------------------------------------------------------
// Timing helpers (spec SS12)
// ---------------------------------------------------------------------------
static double timeLaunch(void (*launch)(int), int inner, int iters)
{
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch(inner);
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms / iters;
}

// Globals the launch thunks read.
static float4 *g_big, *g_small;
static float  *g_sink;
static size_t  g_bigN, g_smallN;

static void lDram (int inner) { (void)inner; streamRead<<<640,256>>>(g_big, g_sink, g_bigN); }
static void lL2   (int reps)  { streamReadRep<<<640,256>>>(g_small, g_sink, g_smallN, reps); }
static void lSmem (int it)    { sharedScalarProbe<<<480,128>>>(g_sink, it); }
static void lSmemH(int it)    { sharedScalarProbeHoisted<<<480,128>>>(g_sink, it); }
static void lVmem (int it)    { sharedVecProbe<<<480,128>>>(g_sink, it); }
static void lFfma (int it)    { ffmaProbe<<<480,128>>>(g_sink, it); }
static void lMixed(int it)    { mixedProbe<<<480,128>>>(g_sink, it); }

// ---------------------------------------------------------------------------
// The roofline model itself
// ---------------------------------------------------------------------------
typedef struct {
    const char *name;
    double bwGBs;        // ceiling of this level, GB/s
    double ridge;        // FLOP/byte at which it meets the compute plateau
} Level;

// Attainable GFLOP/s for a kernel of arithmetic intensity `ai` measured at a
// level whose bandwidth is `bwGBs`, against a compute plateau `peakGF`.
static double attainable(double ai, double bwGBs, double peakGF)
{
    double mem = ai * bwGBs;            // GB/s * FLOP/byte = GFLOP/s
    return (mem < peakGF) ? mem : peakGF;
}

// ---------------------------------------------------------------------------
// ASCII log-log roofline
// ---------------------------------------------------------------------------
#define PLOT_W 76
#define PLOT_H 22
static void plotRoofline(const Level *lv, int nlv, double peakGF,
                         const char *marks, const double *markAI,
                         const double *markGF, int nmark)
{
    static char grid[PLOT_H][PLOT_W+1];
    const double x0 = log10(1.0/32.0), x1 = log10(1024.0);
    const double y0 = log10(1.0),      y1 = log10(32768.0);
    for (int r = 0; r < PLOT_H; ++r) { for (int c = 0; c < PLOT_W; ++c) grid[r][c]=' '; grid[r][PLOT_W]='\0'; }

    const char *sym = "=-.";
    for (int L = 0; L < nlv; ++L) {
        for (int c = 0; c < PLOT_W; ++c) {
            double ai = pow(10.0, x0 + (x1-x0)*c/(PLOT_W-1.0));
            double g  = attainable(ai, lv[L].bwGBs, peakGF);
            int r = (int)lround((PLOT_H-1) * (1.0 - (log10(g)-y0)/(y1-y0)));
            if (r < 0) r = 0; if (r >= PLOT_H) continue;
            if (grid[r][c] == ' ') grid[r][c] = sym[L % 3];
        }
    }
    for (int m = 0; m < nmark; ++m) {
        int c = (int)lround((PLOT_W-1) * (log10(markAI[m])-x0)/(x1-x0));
        int r = (int)lround((PLOT_H-1) * (1.0 - (log10(markGF[m])-y0)/(y1-y0)));
        if (c < 0) c = 0; if (c >= PLOT_W) c = PLOT_W-1;
        if (r < 0) r = 0; if (r >= PLOT_H) r = PLOT_H-1;
        grid[r][c] = marks[m];
    }
    printf("  GFLOP/s\n");
    for (int r = 0; r < PLOT_H; ++r) {
        double g = pow(10.0, y1 - (y1-y0)*r/(PLOT_H-1.0));
        if (r % 3 == 0) printf(" %8.0f |%s\n", g, grid[r]);
        else            printf("          |%s\n", grid[r]);
    }
    printf("          +");
    for (int c = 0; c < PLOT_W; ++c) printf("-");
    printf("\n           0.03      0.25         2          16        128      1024\n");
    printf("                     arithmetic intensity, FLOP per byte AT THAT LEVEL\n");
}

// ---------------------------------------------------------------------------
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("=== Module 21 / Example 1 - the roofline for this GPU ===\n\n");
    printf("Hardware bounds, derived from the spec table, never measured:\n");
    printf("  DRAM pin peak                 %10.1f GB/s\n", BOUND_DRAM_GBS);
    printf("  shared array, 40 SM x 128 B/cy %9.1f GB/s   (at the %.3f GHz max SM clock)\n",
           BOUND_SMEM_GBS, SM_CLOCK_MAX_GHZ);
    printf("  FP32, 40 SM x 128 lanes x 2    %9.1f GFLOP/s\n", BOUND_FP32_GFLOPS);
    printf("  issue, 40 SM x 4 schedulers    %9.1f G warp-instr/s\n\n", BOUND_ISSUE_GIPS);

    // ---- allocation -------------------------------------------------------
    g_bigN   = (size_t)256*1024*1024 / sizeof(float4);   // 256 MB, 5.3x L2
    g_smallN = (size_t) 24*1024*1024 / sizeof(float4);   //  24 MB, half of L2
    CHECK(cudaMalloc(&g_big,   g_bigN  *sizeof(float4)));
    CHECK(cudaMalloc(&g_small, g_smallN*sizeof(float4)));
    CHECK(cudaMalloc(&g_sink,  sizeof(float)*4));
    CHECK(cudaMemset(g_big,   1, g_bigN  *sizeof(float4)));
    CHECK(cudaMemset(g_small, 1, g_smallN*sizeof(float4)));

    // ---- the two warm-ups and the two timing groups ------------------------
    // Spec SS12 rule 1 says to time COMPETING configurations back to back. The
    // DRAM ceiling and the FP32 ceiling are not competitors -- they are two
    // different axes of one model -- and they need different warm-ups, because
    // the power manager trades away whatever the running kernel is not using
    // (Module 16 watched the SM clock drop to 285 MHz under a pure read).
    // So: 1500 ms streaming, measure the off-chip group; then 500 ms compute,
    // measure the on-chip group. Within each group the configurations ARE
    // competitors and are rotated back to back with SWEEPS >= NCFG.
    enum { NMEMCFG = 2, NCOMPCFG = 5 };
    // indices into the merged best[]/inner[] arrays built after both sweeps
    enum { C_DRAM=0, C_L2, C_SMEM, C_SMEMH, C_VMEM, C_FFMA, C_MIX };
    void (*mlaunch[NMEMCFG])(int)  = { lDram, lL2 };
    int    minner [NMEMCFG]        = { 0,     8   };
    void (*claunch[NCOMPCFG])(int) = { lSmem, lSmemH, lVmem, lFfma, lMixed };
    int    cinner [NCOMPCFG]       = { 4096,  4096,   4096,  8192,  8192   };
    int    miters[NMEMCFG], citers[NCOMPCFG];
    double mbest[NMEMCFG],  cbest[NCOMPCFG];

    printf("-- warming: 1500 ms streaming, then the off-chip group ----------------\n");
    {
        cudaEvent_t w0, w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        float el = 0.f; CHECK(cudaEventRecord(w0));
        while (el < 1500.f) { streamRead<<<640,256>>>(g_big, g_sink, g_bigN);
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            CHECK(cudaEventElapsedTime(&el, w0, w1)); }
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    }
    for (int i = 0; i < NMEMCFG; ++i) {
        double t = timeLaunch(mlaunch[i], minner[i], 1);
        int n = (int)(10.0 / (t > 1e-4 ? t : 1e-4));
        if (n < 3) n = 3; if (n > 200) n = 200;
        miters[i] = n; mbest[i] = 1e30;
    }
    for (int s = 0; s < 4; ++s)
        for (int q = 0; q < NMEMCFG; ++q) {
            int p = (q + s) % NMEMCFG;
            double t = timeLaunch(mlaunch[p], minner[p], miters[p]);
            if (t < mbest[p]) mbest[p] = t;
        }
    CHECK(cudaGetLastError());

    printf("-- warming: 500 ms compute, then the on-chip group --------------------\n");
    {
        cudaEvent_t w0, w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        float el = 0.f; CHECK(cudaEventRecord(w0));
        while (el < 500.f) { ffmaProbe<<<480,128>>>(g_sink, 2000);
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            CHECK(cudaEventElapsedTime(&el, w0, w1)); }
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    }
    for (int i = 0; i < NCOMPCFG; ++i) {
        double t = timeLaunch(claunch[i], cinner[i], 1);
        int n = (int)(10.0 / (t > 1e-4 ? t : 1e-4));
        if (n < 3) n = 3; if (n > 200) n = 200;
        citers[i] = n; cbest[i] = 1e30;
    }
    for (int s = 0; s < NCOMPCFG; ++s)
        for (int q = 0; q < NCOMPCFG; ++q) {
            int p = (q + s) % NCOMPCFG;
            double t = timeLaunch(claunch[p], cinner[p], citers[p]);
            if (t < cbest[p]) cbest[p] = t;
        }
    CHECK(cudaGetLastError());

    // Merge so the arithmetic below reads the way the model does.
    double best[7] = { mbest[0], mbest[1], cbest[0], cbest[1], cbest[2],
                       cbest[3], cbest[4] };
    int    inner[7]= { minner[0], minner[1], cinner[0], cinner[1], cinner[2],
                       cinner[3], cinner[4] };

    const double dramGBs = (double)(g_bigN*sizeof(float4)) / (best[C_DRAM]*1e-3) / 1e9;
    const double l2GBs   = (double)(g_smallN*sizeof(float4)) * inner[C_L2]
                           / (best[C_L2]*1e-3) / 1e9;
    const double smemB   = 480.0*128.0*(double)inner[C_SMEM]*SPROBE_W*4.0;
    const double smemGBs = smemB / (best[C_SMEM]*1e-3) / 1e9;
    const double smemHGBs= smemB / (best[C_SMEMH]*1e-3) / 1e9;
    const double vmemB   = 480.0*128.0*(double)inner[C_VMEM]*VPROBE_W*16.0;
    const double vmemGBs = vmemB / (best[C_VMEM]*1e-3) / 1e9;
    const double ffmaFl  = 480.0*128.0*(double)inner[C_FFMA]*CHAINS*2.0;
    const double ffmaGF  = ffmaFl / (best[C_FFMA]*1e-3) / 1e9;
    const double mixFl   = 480.0*128.0*(double)inner[C_MIX]*CHAINS*3.0;
    const double mixGF   = mixFl / (best[C_MIX]*1e-3) / 1e9;
    // Warp-instructions per second. Counted from the SASS loop body, not from
    // the source: `cuobjdump -sass` shows 68 instructions per 64 FFMAs in
    // ffmaProbe and 132 per 64 FFMA + 64 FADD in mixedProbe, i.e. the loop
    // counter and branch are 6% and 3% of what is issued. Counting only the
    // arithmetic would understate the issue rate by exactly that much.
    const double ffmaGIPS = 480.0*128.0/32.0*(double)inner[C_FFMA]*CHAINS*(68.0/64.0)
                            / (best[C_FFMA]*1e-3) / 1e9;
    const double mixGIPS  = 480.0*128.0/32.0*(double)inner[C_MIX ]*CHAINS*(132.0/64.0)
                            / (best[C_MIX ]*1e-3) / 1e9;
    const double clockGHz = ffmaGF / (SM_COUNT*(double)FP32_LANES_PER_SM*2.0);

    printf("\n-- A. measured ceilings, each against its hardware bound --------------\n");
    printf("  %-34s %12s %12s %8s\n", "probe", "measured", "bound", "frac");
    printf("  %-34s %10.1f GB/s %10.1f  %7.1f%%  %s\n", "DRAM read (256 MB buffer)",
           dramGBs, BOUND_DRAM_GBS, 100.0*dramGBs/BOUND_DRAM_GBS,
           dramGBs <= BOUND_DRAM_GBS ? "ok" : "IMPOSSIBLE");
    printf("  %-34s %10.1f GB/s %10s  %7s   %s\n", "L2 read (24 MB working set)",
           l2GBs, "-", "-", l2GBs > dramGBs ? "ok (> DRAM, as it must be)" : "SUSPECT");
    printf("  %-34s %10.1f GB/s %10.1f  %7.1f%%  %s\n", "shared read, scalar LDS",
           smemGBs, BOUND_SMEM_GBS, 100.0*smemGBs/BOUND_SMEM_GBS,
           smemGBs <= BOUND_SMEM_GBS ? "ok" : "IMPOSSIBLE");
    printf("  %-34s %10.1f GB/s %10.1f  %7.1f%%  %s\n", "shared read, LDS.128",
           vmemGBs, BOUND_SMEM_GBS, 100.0*vmemGBs/BOUND_SMEM_GBS,
           vmemGBs <= BOUND_SMEM_GBS ? "ok" : "IMPOSSIBLE");
    printf("  %-34s %8.1f GFLOP/s %10.1f  %7.1f%%  %s\n", "FP32 FFMA",
           ffmaGF, BOUND_FP32_GFLOPS, 100.0*ffmaGF/BOUND_FP32_GFLOPS,
           ffmaGF <= BOUND_FP32_GFLOPS ? "ok" : "IMPOSSIBLE");
    printf("  %-34s %6.1f Gwarpinst/s %10.1f  %7.1f%%  %s\n", "issue rate, from the FFMA probe",
           ffmaGIPS, BOUND_ISSUE_GIPS, 100.0*ffmaGIPS/BOUND_ISSUE_GIPS,
           ffmaGIPS <= BOUND_ISSUE_GIPS ? "ok" : "IMPOSSIBLE");
    printf("  %-34s %6.1f Gwarpinst/s %10.1f  %7.1f%%  %s\n", "issue rate, from the mixed probe",
           mixGIPS, BOUND_ISSUE_GIPS, 100.0*mixGIPS/BOUND_ISSUE_GIPS,
           mixGIPS <= BOUND_ISSUE_GIPS ? "ok" : "IMPOSSIBLE");
    printf("\n  Implied SM clock from the FFMA ceiling: %.3f GHz.\n", clockGHz);
    printf("  The bounds above use the device's MAXIMUM %.3f GHz, because that is the\n"
           "  only clock figure that is a bound. It is deliberately loose. At the\n"
           "  clock this run actually ran at, the bank array could deliver at most\n"
           "  %.0f GB/s and the LDS.128 probe measured %.0f -- %s.\n"
           "  Module 17 saw the same: the shared-memory kernel clocks higher than the\n"
           "  FFMA kernel, so the RATIO between the scalar and vector rows (%.2f, and\n"
           "  Module 7's two-cycle floor predicts 1.9) is the trustworthy quantity,\n"
           "  not the B/cycle/SM the absolute figure implies.\n",
           SM_CLOCK_MAX_GHZ,
           SM_COUNT*(double)SMEM_BYTES_PER_CYCLE_PER_SM*clockGHz, vmemGBs,
           vmemGBs <= SM_COUNT*(double)SMEM_BYTES_PER_CYCLE_PER_SM*clockGHz
             ? "inside it" : "ABOVE it, which is the clock talking",
           vmemGBs/smemGBs);
    printf("  (cudaDevAttrClockRate would say 1.545 and is not usable for this --\n"
           "   the percentage it produces exceeds 100%%. Spec SS12 rule 6.)\n");
    printf("\n  The two issue numbers agree to %.1f%%, and the mixed kernel delivers\n",
           100.0*fabs(ffmaGIPS-mixGIPS)/ffmaGIPS);
    printf("  %.0f GFLOP/s against the FFMA kernel's %.0f -- a ratio of %.2f. The SASS\n",
           mixGF, ffmaGF, mixGF/ffmaGF);
    printf("  loop bodies are 132 instructions per 192 FLOPs and 68 per 128, so the\n"
           "  predicted ratio is (192/132)/(128/68) = %.2f. The SLOT is the ceiling;\n"
           "  how much arithmetic you put in it is a property of your code.\n",
           (192.0/132.0)/(128.0/68.0));
    printf("\n  Note what this does to the phrase 'FP32 peak'. The FFMA probe issues 68\n"
           "  instructions per 64 FFMAs, so the FP32 lanes are idle 6%% of cycles even\n"
           "  in the best kernel anyone can write for them. The %.0f GFLOP/s above is\n"
           "  94%% of a true lane peak of %.0f at the implied %.3f GHz. There is no\n"
           "  compute ceiling that is not also an issue ceiling.\n",
           ffmaGF, ffmaGF*68.0/64.0, clockGHz*68.0/64.0);

    // ---- B. ridge points ---------------------------------------------------
    Level lv[3] = {
        { "DRAM",          dramGBs, ffmaGF/dramGBs },
        { "L2",            l2GBs,   ffmaGF/l2GBs   },
        { "shared, scalar",smemGBs, ffmaGF/smemGBs },
    };
    printf("\n-- B. ridge points ----------------------------------------------------\n");
    printf("  ridge = compute ceiling / memory ceiling = the arithmetic intensity at\n"
           "  which a kernel stops being memory bound at that level.\n\n");
    printf("  %-18s %12s %14s\n", "level", "GB/s", "ridge FLOP/B");
    for (int i = 0; i < 3; ++i)
        printf("  %-18s %12.1f %14.2f\n", lv[i].name, lv[i].bwGBs, lv[i].ridge);
    printf("  %-18s %12.1f %14.2f\n", "shared, LDS.128", vmemGBs, ffmaGF/vmemGBs);
    printf("\n  The DRAM ridge is the classical 'machine balance'. The shared ridge is\n"
           "  the one that decides whether a tiled GEMM can work: a kernel that reads\n"
           "  two shared floats per FMA offers 2 FLOP / 8 B = 0.25 FLOP/byte, which is\n"
           "  %.0fx below the shared ridge. Hence %.1f%% of the compute ceiling, before\n",
           lv[2].ridge/0.25, 100.0*0.25/lv[2].ridge);
    printf("  anything about DRAM is considered.\n");

    // ---- C. the plot -------------------------------------------------------
    printf("\n-- C. the hierarchical roofline ---------------------------------------\n");
    printf("  '=' DRAM ceiling   '-' L2 ceiling   '.' shared (scalar LDS) ceiling\n");
    printf("  markers: S stream/SAXPY  R reduction  N naive GEMM  T tiled GEMM\n"
           "           G register-tiled GEMM  B cuBLAS\n\n");
    {
        // Known measurements from Modules 11-18, placed at the AI of the level
        // that binds them. These are recorded numbers, not measured here.
        const char  mk[]   = "SRNTGB";
        const double mAI[] = { 0.1667, 0.25, 0.25, 0.25,  1.333, 2.0 };
        const double mGF[] = { 68.0,   102.0, 1300.0, 1710.0, 7800.0, 8900.0 };
        plotRoofline(lv, 3, ffmaGF, mk, mAI, mGF, 6);
    }

    // ---- D. the probe that lies --------------------------------------------
    printf("\n-- D. the same shared probe with the address hoisted out of the loop ---\n");
    printf("  honest probe   : %9.1f GB/s   (%.1f%% of the %0.f GB/s bound)\n",
           smemGBs, 100.0*smemGBs/BOUND_SMEM_GBS, BOUND_SMEM_GBS);
    printf("  hoisted probe  : %9.1f GB/s   (%.1f%% of the same bound)  %s\n",
           smemHGBs, 100.0*smemHGBs/BOUND_SMEM_GBS,
           smemHGBs > BOUND_SMEM_GBS ? "<-- REJECTED, physically impossible"
                                     : "<-- (did not exceed the bound this run)");
    printf("\n  Both kernels have identical source except for one `base += 1`.\n");
    printf("  cuobjdump -sass, loop body between the back edge and its branch:\n"
           "    honest  : 133 instructions, of which 32 are LDS\n"
           "    hoisted :  35 instructions, of which  0 are LDS\n"
           "  The hoisted kernel's %d loads sit ABOVE the loop. It is timing 32\n"
           "  FADDs on a loop-invariant constant and calling it shared bandwidth.\n"
           "  Measured ratio this run: %.1fx.\n", SPROBE_W, smemHGBs/smemGBs);
    printf("  Spec SS12 rule 13: a microbenchmark is a number until you have bounded\n"
           "  it. Module 17's first shared probe reported 40 TB/s -- 4x this bound --\n"
           "  and the hardware was not the thing that was wrong.\n");

    // ---- validation pass (untimed) -----------------------------------------
    // Every ceiling must be positive, finite, and inside its bound. The L2
    // figure must exceed the DRAM figure or the working set was not resident.
    printf("\n-- validation (second, untimed pass) -----------------------------------\n");
    int ok = 1;
    struct { const char *n; double v, b; } chk[5] = {
        { "DRAM",      dramGBs, BOUND_DRAM_GBS },
        { "shared",    smemGBs, BOUND_SMEM_GBS },
        { "shared.128",vmemGBs, BOUND_SMEM_GBS },
        { "FP32",      ffmaGF,  BOUND_FP32_GFLOPS },
        { "issue",     ffmaGIPS,BOUND_ISSUE_GIPS },
    };
    for (int i = 0; i < 5; ++i) {
        int good = (chk[i].v > 0.0) && (chk[i].v < chk[i].b) && isfinite(chk[i].v);
        printf("  %-12s %10.1f  <= bound %10.1f   %s\n",
               chk[i].n, chk[i].v, chk[i].b, good ? "ok" : "FAIL");
        if (!good) ok = 0;
    }
    { int good = l2GBs > dramGBs;
      printf("  %-12s %10.1f  >  DRAM    %10.1f   %s\n", "L2", l2GBs, dramGBs,
             good ? "ok" : "FAIL (working set not L2-resident?)");
      if (!good) ok = 0; }
    { int good = (vmemGBs > smemGBs);
      printf("  %-12s LDS.128 %6.2fx scalar LDS              %s\n", "vector",
             vmemGBs/smemGBs, good ? "ok" : "FAIL");
      if (!good) ok = 0; }

    CHECK(cudaFree(g_big)); CHECK(cudaFree(g_small)); CHECK(cudaFree(g_sink));
    printf("\nOVERALL: %s\n", ok ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return ok ? 0 : 1;
}
