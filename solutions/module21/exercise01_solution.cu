// =============================================================================
// Module 21 / Exercise 1 — SOLUTION — build the roofline for this GPU.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
// RUN  : exercise01_solution.exe
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

// ---- hardware constants, from the spec table and from nvidia-smi -----------
#define SM_COUNT              40
#define SCHEDULERS_PER_SM      4
#define FP32_LANES_PER_SM    128
#define SMEM_BANKS            32
#define SMEM_BANK_BYTES        4
#define DRAM_PIN_PEAK_GBS  432.0
// nvidia-smi --query-gpu=clocks.max.sm --format=csv  ->  3105 MHz
#define SM_CLOCK_MAX_GHZ   3.105

#define LEVEL_DRAM   0
#define LEVEL_SHARED 1
#define LEVEL_FP32   2
#define LEVEL_ISSUE  3

#define BIGN  ((size_t)256*1024*1024/sizeof(float4))   // 256 MB, 5.3x the L2
#define SPROBE_N 2048
#define SPROBE_W 32
#define CHAINS 8

// =============================================================================
// TODO 1 — the DRAM streaming-read probe.
// =============================================================================
__global__ void dramProbe(const float4 * __restrict__ src, float *sink, size_t n)
{
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    float4 a = make_float4(0.f,0.f,0.f,0.f);
    for (; i < n; i += gridDim.x*(size_t)blockDim.x) {
        float4 v = src[i];
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    // The store is unreachable, but the compiler cannot prove it, so neither
    // the loads nor the adds can be deleted.
    if (a.x == 1e30f) sink[0] = a.x + a.y + a.z + a.w;
}

// =============================================================================
// TODO 2 — the shared-memory read probe.
//   * stride 33 floats: gcd(33,32) = 1, so the 32 lanes of a warp land in 32
//     distinct banks -> conflict-free (Module 7);
//   * 33 is also not 1, so consecutive reads are not adjacent and ptxas cannot
//     contract them into LDS.128 -- this probe measures scalar LDS;
//   * `base += 1` makes every address depend on the loop index. Without it the
//     whole sum is loop-invariant, ptxas hoists all 32 LDS above the loop, and
//     the kernel reports several times the bank array's physical maximum.
// =============================================================================
__global__ void sharedProbe(float *sink, int iters)
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

// Provided, already written: the same probe with the address hoisted. Your
// bound from TODO 4 has to reject it.
__global__ void sharedProbeBroken(float *sink, int iters)
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

// =============================================================================
// TODO 3 — the FP32 throughput probe. Eight independent chains hide the FFMA
// latency with ILP alone, so one warp can saturate; `#pragma unroll 8` on the
// outer loop makes the loop counter and branch 4 of every 68 issued
// instructions instead of 4 of every 12.
// =============================================================================
__global__ void ffmaProbe(float *sink, int iters)
{
    float a[CHAINS];
    const float b = 1.0000001f;
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

// =============================================================================
// TODO 4 — the hardware bounds. Return a negative number for an unknown level.
//   DRAM   : the pin rate. 192 bits x 9.001 GHz x 2 (DDR) / 8 = 432.0 GB/s.
//   SHARED : the bank array delivers 32 banks x 4 B = 128 B per cycle per SM.
//   FP32   : 128 lanes x 2 FLOP (an FMA) per cycle per SM.
//   ISSUE  : 4 schedulers x 1 warp-instruction per cycle per SM.
// The clock used must be the device MAXIMUM, not the clock you think it is
// running at and not cudaDevAttrClockRate -- a bound has to hold for every run.
// =============================================================================
static double hardwareBound(int level)
{
    const double f = SM_CLOCK_MAX_GHZ;
    switch (level) {
        case LEVEL_DRAM:   return DRAM_PIN_PEAK_GBS;
        case LEVEL_SHARED: return SM_COUNT * (double)SMEM_BANKS * SMEM_BANK_BYTES * f;
        case LEVEL_FP32:   return SM_COUNT * (double)FP32_LANES_PER_SM * 2.0 * f;
        case LEVEL_ISSUE:  return SM_COUNT * (double)SCHEDULERS_PER_SM * f;
        default:           return -1.0;
    }
}

// =============================================================================
// TODO 5 — the roofline itself.
// =============================================================================
static double rooflineGFLOPs(double ai, double bwGBs, double peakGF)
{
    double mem = ai * bwGBs;                 // FLOP/byte x GB/s = GFLOP/s
    return (mem < peakGF) ? mem : peakGF;
}
static double ridgeFlopPerByte(double peakGF, double bwGBs)
{
    return peakGF / bwGBs;
}

// =============================================================================
// Harness below this line. You do not need to change it.
// =============================================================================
static unsigned fnvNums(const double *v, int n, double scale)
{ char buf[64]; unsigned h = 2166136261u;
  for (int i = 0; i < n; ++i) { snprintf(buf, sizeof buf, "%lld", (long long)llround(v[i]*scale));
      for (char *p = buf; *p; ++p) { h ^= (unsigned char)*p; h *= 16777619u; } }
  return h; }

static double timeIt(void (*f)(int), int arg, int iters)
{
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) f(arg);
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms/iters;
}
static float4 *gBig; static float *gSink;
static void lDram  (int a) { (void)a; dramProbe<<<640,256>>>(gBig, gSink, BIGN); }
static void lSmem  (int a) { sharedProbe<<<480,128>>>(gSink, a); }
static void lSmemB (int a) { sharedProbeBroken<<<480,128>>>(gSink, a); }
static void lFfma  (int a) { ffmaProbe<<<480,128>>>(gSink, a); }

#define PLOT_W 72
#define PLOT_H 20
static void plotRoofline(const double *bw, const char **nm, int nlv, double peak)
{
    static char grid[PLOT_H][PLOT_W+1];
    const double x0 = log10(1.0/32.0), x1 = log10(1024.0);
    const double y0 = log10(1.0),      y1 = log10(32768.0);
    for (int r = 0; r < PLOT_H; ++r) { for (int c = 0; c < PLOT_W; ++c) grid[r][c]=' ';
                                       grid[r][PLOT_W]='\0'; }
    const char *sym = "=-.";
    for (int L = 0; L < nlv; ++L)
        for (int c = 0; c < PLOT_W; ++c) {
            double ai = pow(10.0, x0 + (x1-x0)*c/(PLOT_W-1.0));
            double g  = rooflineGFLOPs(ai, bw[L], peak);
            if (g <= 0) continue;
            int r = (int)lround((PLOT_H-1)*(1.0 - (log10(g)-y0)/(y1-y0)));
            if (r < 0) r = 0; if (r >= PLOT_H) continue;
            if (grid[r][c] == ' ') grid[r][c] = sym[L%3];
        }
    printf("  GFLOP/s\n");
    for (int r = 0; r < PLOT_H; ++r) {
        double g = pow(10.0, y1 - (y1-y0)*r/(PLOT_H-1.0));
        if (r % 3 == 0) printf(" %8.0f |%s\n", g, grid[r]);
        else            printf("          |%s\n", grid[r]);
    }
    printf("          +");
    for (int c = 0; c < PLOT_W; ++c) printf("-");
    printf("\n           0.03     0.25        2         16       128     1024  FLOP/B\n");
    for (int L = 0; L < nlv; ++L)
        printf("   '%c' %-16s %9.1f GB/s   ridge %7.2f FLOP/byte\n",
               sym[L%3], nm[L], bw[L], ridgeFlopPerByte(peak, bw[L]));
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 21 / Exercise 1 - build the roofline ===\n\n");

    if (hardwareBound(LEVEL_DRAM) < 0.0 || hardwareBound(LEVEL_SHARED) < 0.0 ||
        hardwareBound(LEVEL_FP32) < 0.0 || hardwareBound(LEVEL_ISSUE) < 0.0) {
        printf("Set TODO 4 first.\n"); return 0;
    }
    if (rooflineGFLOPs(1.0, 1.0, 1.0) < 0.0 || ridgeFlopPerByte(1.0, 1.0) < 0.0) {
        printf("Set TODO 5 first.\n"); return 0;
    }

    CHECK(cudaMalloc(&gBig, BIGN*sizeof(float4)));
    CHECK(cudaMalloc(&gSink, 4*sizeof(float)));
    CHECK(cudaMemset(gBig, 1, BIGN*sizeof(float4)));

    // --- warm-up: 1500 ms streaming, then the off-chip measurement; then
    //     500 ms compute, then the on-chip measurements. Spec SS12 rule 4 and
    //     its corollary.
    printf("-- warming 1500 ms streaming ----------------------------------------\n");
    { cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0; CHECK(cudaEventRecord(w0));
      while (el < 1500.f) { lDram(0);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); }
    CHECK(cudaGetLastError());
    double bD = 1e30;
    { double t = timeIt(lDram,0,1); int n = (int)(10.0/(t>1e-3?t:1e-3));
      if (n<3) n=3; if (n>200) n=200;
      for (int s = 0; s < 4; ++s) { double x = timeIt(lDram,0,n); if (x<bD) bD=x; } }

    printf("-- warming 500 ms compute -------------------------------------------\n");
    { cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0; CHECK(cudaEventRecord(w0));
      while (el < 500.f) { lFfma(2000);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); }
    CHECK(cudaGetLastError());

    // Three on-chip configurations, rotated, SWEEPS >= NCFG.
    void (*cl[3])(int) = { lSmem, lSmemB, lFfma };
    int   ca[3]        = { 4096,  4096,   8192  };
    int   ci[3];       double cb[3];
    for (int i = 0; i < 3; ++i) {
        double t = timeIt(cl[i], ca[i], 1);
        int n = (int)(10.0/(t>1e-3?t:1e-3)); if (n<3) n=3; if (n>200) n=200;
        ci[i]=n; cb[i]=1e30;
    }
    CHECK(cudaGetLastError());
    for (int s = 0; s < 4; ++s)
        for (int q = 0; q < 3; ++q) {
            int p = (q+s)%3; double t = timeIt(cl[p], ca[p], ci[p]);
            if (t < cb[p]) cb[p] = t;
        }
    CHECK(cudaGetLastError());

    const double smemBytes = 480.0*128.0*(double)ca[0]*SPROBE_W*4.0;
    const double dramGBs = (double)(BIGN*sizeof(float4))/(bD*1e-3)/1e9;
    const double smemGBs = smemBytes/(cb[0]*1e-3)/1e9;
    const double brokGBs = smemBytes/(cb[1]*1e-3)/1e9;
    const double ffmaGF  = 480.0*128.0*(double)ca[2]*CHAINS*2.0/(cb[2]*1e-3)/1e9;
    // Warp-instructions per second, counting the FFMAs only: the loop
    // overhead your implementation emits is not counted, so this row is a
    // LOWER bound on the issue rate. Example 1 does the exact version from
    // a SASS instruction census.
    const double issueGI = ffmaGF/64.0;

    printf("\n-- measured, each against your TODO 4 bound --------------------------\n");
    printf("  %-26s %12s %12s %8s %s\n","probe","measured","bound","frac","verdict");
    struct { const char *n; double v, b; const char *u; } row[5] = {
        { "DRAM read",          dramGBs, hardwareBound(LEVEL_DRAM),   "GB/s"     },
        { "shared read (yours)",smemGBs, hardwareBound(LEVEL_SHARED), "GB/s"     },
        { "shared read (broken)",brokGBs,hardwareBound(LEVEL_SHARED), "GB/s"     },
        { "FP32 FFMA",          ffmaGF,  hardwareBound(LEVEL_FP32),   "GFLOP/s"  },
        { "issue rate (lower bd)",issueGI,hardwareBound(LEVEL_ISSUE),  "Ginstr/s" },
    };
    for (int i = 0; i < 5; ++i)
        printf("  %-26s %10.1f %-9s %10.1f %7.1f%%  %s\n", row[i].n, row[i].v, row[i].u,
               row[i].b, 100.0*row[i].v/row[i].b,
               row[i].v <= row[i].b ? "accepted" : "REJECTED - impossible");

    printf("\n-- your roofline ----------------------------------------------------\n");
    { const double bw[3] = { dramGBs, 1305.0, smemGBs };
      const char *nm[3] = { "DRAM", "L2 (recorded)", "shared, scalar" };
      plotRoofline(bw, nm, 3, ffmaGF); }
    printf("\n  machine balance (DRAM ridge) = %.2f FLOP/byte\n",
           ridgeFlopPerByte(ffmaGF, dramGBs));
    printf("  a kernel reading 2 shared floats per FMA offers 0.25 FLOP/byte and is\n"
           "  therefore capped at %.1f%% of the compute ceiling.\n",
           100.0*rooflineGFLOPs(0.25, smemGBs, ffmaGF)/ffmaGF);

    // ----------------------------------------------------------------- scoring
    int score = 0;
    printf("\n-- scoring -----------------------------------------------------------\n");

    int okD = (dramGBs > 200.0 && dramGBs <= hardwareBound(LEVEL_DRAM));
    printf("  [%s] 1. DRAM probe in 200..432 GB/s and inside its bound  (%.1f)\n",
           okD?"x":" ", dramGBs); score += okD;

    int okS = (smemGBs > 2000.0 && smemGBs < 12000.0 && smemGBs <= hardwareBound(LEVEL_SHARED));
    printf("  [%s] 2. shared probe in 2000..12000 GB/s and inside its bound (%.1f)\n",
           okS?"x":" ", smemGBs); score += okS;

    int okF = (ffmaGF > 8000.0 && ffmaGF <= hardwareBound(LEVEL_FP32));
    printf("  [%s] 3. FFMA probe above 8000 GFLOP/s and inside its bound (%.1f)\n",
           okF?"x":" ", ffmaGF); score += okF;

    double bnds[4] = { hardwareBound(LEVEL_DRAM), hardwareBound(LEVEL_SHARED),
                       hardwareBound(LEVEL_FP32), hardwareBound(LEVEL_ISSUE) };
    unsigned hB = fnvNums(bnds, 4, 100.0);
    int okB = (hB == 4013188270u);
    printf("  [%s] 4. hardwareBound() matches the reference\n", okB?"x":" ");
    score += okB;

    double probe[6] = {
        rooflineGFLOPs(0.25, 400.0, 18000.0), rooflineGFLOPs(64.0, 400.0, 18000.0),
        rooflineGFLOPs(45.0, 400.0, 18000.0), ridgeFlopPerByte(18000.0, 400.0),
        ridgeFlopPerByte(18000.0, 5400.0),    rooflineGFLOPs(1.333, 5400.0, 18000.0) };
    unsigned hR = fnvNums(probe, 6, 1000.0);
    int okR = (hR == 445527222u);
    printf("  [%s] 5. rooflineGFLOPs()/ridgeFlopPerByte() match the reference\n",
           okR?"x":" ");
    score += okR;

    int okX = (brokGBs > hardwareBound(LEVEL_SHARED));
    printf("  [%s] 6. your bound REJECTS the hoisted probe (%.1f vs %.1f)\n",
           okX?"x":" ", brokGBs, hardwareBound(LEVEL_SHARED));
    score += okX;

    CHECK(cudaFree(gBig)); CHECK(cudaFree(gSink));
    printf("\nSCORE: %d/6\n", score);
    printf("OVERALL: %s\n", score == 6 ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return score == 6 ? 0 : 1;
}
