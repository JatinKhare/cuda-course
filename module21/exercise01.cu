// =============================================================================
// Module 21 / Exercise 1 — build the roofline for this GPU.
//
// GOAL : Measure every ceiling the roofline needs -- DRAM, shared memory, FP32
//        and the instruction issue rate -- assemble them into a model, and
//        make the model refuse a measurement that cannot physically be true.
//
//        The last part is the point of the exercise. Module 17's first
//        shared-memory probe reported 40 TB/s, four times the bank array's
//        theoretical maximum, because the compiler had hoisted the loop-
//        invariant address out of the timing loop. A number that exceeds a
//        hardware bound is not a discovery; it is a broken benchmark. This
//        file ships the broken probe alongside yours and requires your bound
//        to throw it out.
//
// WHAT TO FILL IN
//   TODO 1  the DRAM streaming-read probe kernel
//   TODO 2  the shared-memory read probe kernel
//   TODO 3  the FP32 throughput probe kernel
//   TODO 4  hardwareBound() -- the four bounds, derived, not measured  (DESIGN)
//   TODO 5  rooflineGFLOPs() and ridgeFlopPerByte()
//
// SCORING: 6 points. OVERALL: PASS requires all six.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN  : exercise01.exe
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
// Read all `n` float4 of `src` exactly once and make sure nothing you do can
// be deleted. `src` is 256 MB, five times the 48 MB L2, so this measures DRAM
// and not cache. It is launched as <<<640,256>>>, which is far fewer threads
// than elements, so the traversal is yours to choose.
__global__ void dramProbe(const float4 * __restrict__ src, float *sink, size_t n)
{
    // TODO 1: read every element of src once, accumulate, and write the result
    //         to sink[0] under a condition the compiler cannot evaluate. If
    //         the accumulator is dead, the loads are dead and you will measure
    //         an empty loop.
    // YOUR CODE HERE
    (void)src; (void)sink; (void)n;
}

// =============================================================================
// TODO 2 — the shared-memory read probe.
//
// The harness computes the bandwidth as
//     blocks x threads x iters x SPROBE_W x 4 bytes / elapsed,
// so your inner loop must perform exactly SPROBE_W scalar shared reads per
// value of `t`, and every one of them must actually execute.
//
// Three separate things can make this number a lie. Defeat all three:
//   (a) a bank conflict would make the probe measure a conflict, not the array
//       (Module 7: bank = (addr/4) % 32);
//   (b) if consecutive reads are adjacent in memory, ptxas contracts four of
//       them into one LDS.128 and you are no longer measuring scalar LDS
//       (Module 7 rule, Module 18 confirmation);
//   (c) if the addresses do not depend on `t`, every load is loop-invariant
//       and is hoisted out of the timing loop entirely.
// Check the result against your TODO 4 bound before you believe it.
// =============================================================================
__global__ void sharedProbe(float *sink, int iters)
{
    __shared__ float s[SPROBE_N];
    for (int i = threadIdx.x; i < SPROBE_N; i += blockDim.x) s[i] = (float)(i & 255);
    __syncthreads();
    float acc = 0.0f;
    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        // TODO 2: SPROBE_W scalar shared-memory reads per iteration of t.
        // YOUR CODE HERE
    }
    if (acc == 1e30f) sink[0] = acc + s[0];
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
// TODO 3 — the FP32 throughput probe.
//
// The harness counts  blocks x threads x iters x CHAINS x 2  FLOPs, so each
// iteration of `t` must perform exactly CHAINS fused multiply-adds per thread.
//
// Two things stand between you and the ceiling. An FFMA has a multi-cycle
// latency, so a single dependent chain measures latency rather than
// throughput; and the loop counter, compare and branch are issued instructions
// too, competing for the same slots as the arithmetic. A loop body of
// 8 FFMAs + 4 overhead instructions caps the measurement at 8/12 of the real
// ceiling, which looks like a hardware fact and is not one.
// =============================================================================
__global__ void ffmaProbe(float *sink, int iters)
{
    float a[CHAINS];
    const float b = 1.0000001f;
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) a[i] = (float)(threadIdx.x + i);

    // TODO 3: CHAINS fused multiply-adds per iteration of t, arranged so the
    //         measurement is a throughput measurement and the loop overhead is
    //         a small fraction of what is issued.
    // YOUR CODE HERE
    (void)iters; (void)b;

    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) s += a[i];
    if (s == 1e30f) sink[0] = s;
}

// =============================================================================
// TODO 4 — DESIGN. The hardware bounds.
//
// For each level return the largest value the silicon could possibly produce,
// in GB/s for the two memory levels, GFLOP/s for FP32 and G warp-instructions
// per second for issue. Everything you need is in the constants above.
//
// These are BOUNDS, not estimates and not measurements. Two consequences you
// have to think about rather than look up:
//   * which of the clock figures available to you may appear in a bound, and
//     why the other ones may not. cudaDevAttrClockRate reports 1.545 GHz on
//     this part while the SM runs anywhere from 0.49 to over 2 GHz;
//   * a bound that is too tight rejects correct measurements, and a bound that
//     is too loose accepts the broken probe this file ships. Check 6 scores
//     exactly that: your bound has to reject sharedProbeBroken.
//
// Return a negative number for an unrecognised level (the harness uses that to
// detect that this TODO has not been filled in).
// =============================================================================
static double hardwareBound(int level)
{
    // TODO 4: return the bound for LEVEL_DRAM / LEVEL_SHARED / LEVEL_FP32 /
    //         LEVEL_ISSUE, and a negative number otherwise.
    // YOUR CODE HERE
    (void)level;
    return -1.0;
}

// =============================================================================
// TODO 5 — the roofline itself.
//
// rooflineGFLOPs : the attainable GFLOP/s for a kernel whose arithmetic
//                  intensity, measured at the level whose bandwidth is bwGBs,
//                  is `ai` FLOP per byte, against a compute plateau peakGF.
//                  Mind the units: GB/s times FLOP/byte is already GFLOP/s.
// ridgeFlopPerByte : the arithmetic intensity at which that level stops being
//                  the binding constraint.
//
// Both must return a negative number only when they have not been written; the
// harness uses that to detect an unfilled TODO.
// =============================================================================
static double rooflineGFLOPs(double ai, double bwGBs, double peakGF)
{
    // TODO 5a: the attainable rate.
    // YOUR CODE HERE
    (void)ai; (void)bwGBs; (void)peakGF;
    return -1.0;
}
static double ridgeFlopPerByte(double peakGF, double bwGBs)
{
    // TODO 5b: the ridge point.
    // YOUR CODE HERE
    (void)peakGF; (void)bwGBs;
    return -1.0;
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
