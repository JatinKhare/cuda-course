// =============================================================================
// Module 19 / Exercise 2 — SOLUTION — the register/occupancy exchange rate,
//                          measured, and the operating point chosen.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
// RUN  : exercise02_solution.exe
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                       \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        printf("CUDA error %s (%s) at %s:%d\n", cudaGetErrorName(_e),          \
               cudaGetErrorString(_e), __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define REGS_PER_SM       65536
#define REG_SLICES        4
#define REGS_PER_SLICE    16384
#define REG_GRAN          8
#define WARPS_PER_SM      48
#define MAX_BLOCKS_PER_SM 24
#define SMEM_PER_SM       102400
#define SMEM_RESERVE      1024
#define SMEM_GRAN         128
#define SM_COUNT          40

#define TPB    128          // threads per block, fixed for the whole sweep
#define WSTATE 96           // recursive stages held in registers
#define RPASS  2            // passes over the stage cascade per input sample
#define NCFG   9
#define NSWEEP 9            // >= NCFG so every configuration leads once (12.9)

static int ceilDiv(int a, int b) { return (a + b - 1) / b; }
static int roundUp(int a, int g) { return ceilDiv(a, g) * g; }
static int roundDown(int a, int g) { return (a / g) * g; }

// Given (Module 19 Exercise 1): resident warps for a register count.
static int residentWarps(int regsPerThread, int threads)
{
    if (regsPerThread <= 0 || threads <= 0) return 0;
    const int wpb   = ceilDiv(threads, 32);
    const int rpw   = roundUp(regsPerThread, REG_GRAN) * 32;
    const int slice = REGS_PER_SLICE / rpw;
    int blocks = (REG_SLICES * slice) / wpb;
    int byWarp = WARPS_PER_SM / wpb;
    if (byWarp < blocks) blocks = byWarp;
    if (MAX_BLOCKS_PER_SM < blocks) blocks = MAX_BLOCKS_PER_SM;
    int w = blocks * wpb;
    return w > WARPS_PER_SM ? WARPS_PER_SM : w;
}

// ============================ TODO 1 ========================================
static int registerCapFor(int threads, int minBlocks)
{
    if (threads <= 0 || minBlocks <= 0) return 0;
    int cap = REGS_PER_SM / (threads * minBlocks);
    cap = roundDown(cap, REG_GRAN);
    if (cap > 255) cap = 255;
    if (cap < 8)   cap = 8;
    return cap;
}

// ============================ TODO 2 ========================================
static int capBindsAt(int threads, int unconstrainedRegs)
{
    if (threads <= 0 || unconstrainedRegs <= 0) return 0;
    for (int b = 1; b <= MAX_BLOCKS_PER_SM; ++b)
        if (registerCapFor(threads, b) < unconstrainedRegs) return b;
    return 0;
}

// ============================ TODO 3, 4, 5 ==================================
#define PRED_MAXOCC_BUCKET  4     // 100%-occupancy build is 5-20x slower
#define PRED_UNCONSTRAINED  2     // no: the compiler's choice leaves warps on the table
#define CHOSEN_MINB         4
#define CHOSEN_RULE         2     // highest occupancy among the spill-free builds

// =============================================================================
// The kernel. WSTATE recursive stages, state live in registers across the whole
// time loop; one input sample and one output sample per step, so the register
// pressure is a design parameter and the dependence chain is long.
// =============================================================================
#define WMAX 160
__constant__ float cA[WMAX];
__constant__ float cB[WMAX];

template<int W, int R, int T, int MINB>
__global__ __launch_bounds__(T, MINB) void cascade(const float * __restrict__ x,
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
        for (int r = 0; r < R; ++r) {
            #pragma unroll
            for (int i = 0; i < W; ++i) {
                v     = fmaf(cA[i], v, st[i]);
                st[i] = fmaf(cB[i], v, 0.5f * st[i]);
            }
        }
        y[(size_t)t * nsig + sig] = v;
    }
}

__global__ void warmStream(const float4 * __restrict__ s, float *o, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float4 a = make_float4(0, 0, 0, 0);
    for (; i < n; i += gridDim.x * (size_t)blockDim.x) {
        float4 v = s[i]; a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    if (a.x == 1e30f) o[0] = a.x + a.y + a.z + a.w;
}
__global__ void ffmaCeiling(float *o, int iters)
{
    float a[8], b = 1.0000001f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = (float)(threadIdx.x + i);
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = fmaf(a[i], b, 1.0f);
    }
    float s = 0; for (int i = 0; i < 8; ++i) s += a[i];
    if (s == 1e30f) o[0] = s;
}

// ------------------------------------------------------------------- harness
static int   gNsig, gL;
static const float *gX; static float *gY;
template<int MB> static void runCfg(void)
{
    cascade<WSTATE,RPASS,TPB,MB><<<(gNsig + TPB - 1) / TPB, TPB>>>(gX, gY, gNsig, gL);
}
static double timeOne(void (*f)(void), int iters)
{
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) f();
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms / iters;
}
static void warmUp(float *sink, float4 *stream, size_t nb)
{
    cudaEvent_t w0, w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
    float el = 0; CHECK(cudaEventRecord(w0));
    while (el < 1500.0f) { warmStream<<<320,256>>>(stream, sink, nb);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el, w0, w1)); }
    el = 0; CHECK(cudaEventRecord(w0));
    while (el < 500.0f) { ffmaCeiling<<<480,128>>>(sink, 2000);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el, w0, w1)); }
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
}
static double probeCeiling(float *sink)
{
    const int blocks = 480, thr = 128, iters = 4000;
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
        CHECK(cudaEventRecord(a));
        ffmaCeiling<<<blocks,thr>>>(sink, iters);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
    }
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    double flops = 2.0 * blocks * thr * 8.0 * iters;
    return flops / (best * 1e-3) / 1e9;
}

static void cpuReference(const float *x, float *y, int nsig, int L, int sig,
                         const float *hA, const float *hB)
{
    float st[WSTATE];
    for (int i = 0; i < WSTATE; ++i) st[i] = 0.0f;
    for (int t = 0; t < L; ++t) {
        float v = x[(size_t)t * nsig + sig];
        for (int r = 0; r < RPASS; ++r)
            for (int i = 0; i < WSTATE; ++i) {
                v     = fmaf(hA[i], v, st[i]);
                st[i] = fmaf(hB[i], v, 0.5f * st[i]);
            }
        y[t] = v;
    }
}

typedef struct { int minb; void (*run)(void); const void *fn; } Cfg;

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 19 / Exercise 2 - SOLUTION - the exchange rate ===\n\n");

    const int nsig = SM_COUNT * 1536, L = 256;
    gNsig = nsig; gL = L;
    const size_t n = (size_t)nsig * L;

    float hA[WMAX], hB[WMAX];
    for (int i = 0; i < WMAX; ++i) {
        hA[i] = 0.980f + 0.0010f * (float)(i % 7);
        hB[i] = 0.0040f - 0.00010f * (float)(i % 11);
    }
    CHECK(cudaMemcpyToSymbol(cA, hA, sizeof(hA)));
    CHECK(cudaMemcpyToSymbol(cB, hB, sizeof(hB)));

    float *hx = (float*)malloc(n * 4), *hy = (float*)malloc(n * 4);
    unsigned s = 11u;
    for (size_t i = 0; i < n; ++i) { s = s * 1664525u + 1013904223u;
        hx[i] = (float)((s >> 9) & 0xFFFFu) / 65536.0f - 0.5f; }
    float *dx, *dy;
    CHECK(cudaMalloc(&dx, n * 4)); CHECK(cudaMalloc(&dy, n * 4));
    CHECK(cudaMemcpy(dx, hx, n * 4, cudaMemcpyHostToDevice));
    gX = dx; gY = dy;

    Cfg cfg[NCFG] = {
        {  1, runCfg<1>,  (const void*)cascade<WSTATE,RPASS,TPB,1>  },
        {  2, runCfg<2>,  (const void*)cascade<WSTATE,RPASS,TPB,2>  },
        {  3, runCfg<3>,  (const void*)cascade<WSTATE,RPASS,TPB,3>  },
        {  4, runCfg<4>,  (const void*)cascade<WSTATE,RPASS,TPB,4>  },
        {  5, runCfg<5>,  (const void*)cascade<WSTATE,RPASS,TPB,5>  },
        {  6, runCfg<6>,  (const void*)cascade<WSTATE,RPASS,TPB,6>  },
        {  8, runCfg<8>,  (const void*)cascade<WSTATE,RPASS,TPB,8>  },
        { 10, runCfg<10>, (const void*)cascade<WSTATE,RPASS,TPB,10> },
        { 12, runCfg<12>, (const void*)cascade<WSTATE,RPASS,TPB,12> },
    };

    // resources first, before anything is timed
    cudaFuncAttributes at[NCFG]; int apiBlk[NCFG];
    for (int i = 0; i < NCFG; ++i) {
        CHECK(cudaFuncGetAttributes(&at[i], cfg[i].fn));
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&apiBlk[i], cfg[i].fn, TPB, 0));
    }
    const int unconstrainedRegs = at[0].numRegs;

    printf("-- warming up: 1500 ms streaming, then 500 ms compute ------------------\n");
    float *sink; float4 *stream;
    size_t nb = (size_t)256 * 1024 * 1024 / 16;
    CHECK(cudaMalloc(&stream, nb * 16)); CHECK(cudaMemset(stream, 1, nb * 16));
    CHECK(cudaMalloc(&sink, 4));
    warmUp(sink, stream, nb);

    // operating-point guard (spec 12.5b): this program SCORES predictions, so it
    // must refuse to score from a power-capped operating point.
    double ceil0 = probeCeiling(sink);
    for (int tries = 0; tries < 5 && ceil0 < 12000.0; ++tries) {
        printf("  FFMA ceiling probe %.0f GFLOP/s is low; idling 10 s and re-warming\n", ceil0);
        std::this_thread::sleep_for(std::chrono::seconds(10));
        warmUp(sink, stream, nb);
        ceil0 = probeCeiling(sink);
    }
    printf("  FFMA ceiling probe: %.0f GFLOP/s%s\n", ceil0,
           (ceil0 < 12000.0) ? "  *** LOW - results below may not be this GPU's ***" : "");

    int iters[NCFG];
    for (int i = 0; i < NCFG; ++i) {
        double t = timeOne(cfg[i].run, 1);
        int k = (int)(10.0 / (t > 0 ? t : 0.01));
        if (k < 3) k = 3; if (k > 64) k = 64;
        iters[i] = k;
    }
    CHECK(cudaGetLastError());

    double best[NCFG];
    for (int i = 0; i < NCFG; ++i) best[i] = 1e30;
    for (int sw = 0; sw < NSWEEP; ++sw)
        for (int q = 0; q < NCFG; ++q) {
            int p = (q + sw) % NCFG;
            double t = timeOne(cfg[p].run, iters[p]);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    // ------------------------------------------------------------------ table
    printf("\n-- the sweep: same source, nine register budgets -----------------------\n");
    printf(" %10s %6s %7s %7s %5s %7s %8s %9s %8s\n",
           "__lb__(B)", "regs", "yourCap", "spillB", "blk", "warps", "occ", "ms", "Gelem/s");
    double bestG = 0.0; int bestIdx = 0;
    double gel[NCFG];
    for (int i = 0; i < NCFG; ++i) {
        gel[i] = (double)n / (best[i] * 1e-3) / 1e9;
        if (gel[i] > bestG) { bestG = gel[i]; bestIdx = i; }
    }
    for (int i = 0; i < NCFG; ++i) {
        int warps = apiBlk[i] * (TPB / 32);
        printf(" %10d %6d %7d %7d %5d %7d %7.1f%% %9.4f %8.3f%s\n",
               cfg[i].minb, at[i].numRegs, registerCapFor(TPB, cfg[i].minb),
               (int)at[i].localSizeBytes, apiBlk[i], warps,
               100.0 * warps / WARPS_PER_SM, best[i], gel[i],
               (i == bestIdx) ? "   <- fastest" : "");
    }

    printf("\n  the exchange rate, read off the two columns you did not measure:\n");
    printf(" %8s %10s %12s\n", "regs", "warps(pred)", "warps gained");
    for (int i = 1; i < NCFG; ++i) {
        int w0 = residentWarps(at[i-1].numRegs, TPB);
        int w1 = residentWarps(at[i].numRegs,   TPB);
        int dR = at[i-1].numRegs - at[i].numRegs;
        printf(" %4d->%-4d %10d %8d warps for %d registers\n",
               at[i-1].numRegs, at[i].numRegs, w1, w1 - w0, dR);
    }

    // ------------------------------------------------------------------ score
    const double maxOccRatio = gel[bestIdx] / gel[NCFG-1];
    const double unconRatio  = gel[bestIdx] / gel[0];
    int predBucket = (maxOccRatio < 1.2) ? 1 : (maxOccRatio < 2.0) ? 2 :
                     (maxOccRatio < 5.0) ? 3 : (maxOccRatio < 20.0) ? 4 : 5;
    int predUncon  = (unconRatio < 1.03) ? 1 : (at[0].localSizeBytes > 0 ? 3 : 2);

    // which minBlocks does each candidate rule select, applied to the measurement?
    int ruleSel[7]; for (int r = 0; r < 7; ++r) ruleSel[r] = -1;
    {
        int bOcc = 0, bOccNoSpill = -1, bLowOcc = 0, bSpillUnder100 = -1;
        for (int i = 0; i < NCFG; ++i) {
            if (apiBlk[i] > apiBlk[bOcc]) bOcc = i;
            if (apiBlk[i] < apiBlk[bLowOcc]) bLowOcc = i;
            if (at[i].localSizeBytes == 0 &&
                (bOccNoSpill < 0 || apiBlk[i] > apiBlk[bOccNoSpill])) bOccNoSpill = i;
            if (at[i].localSizeBytes < 100 &&
                (bSpillUnder100 < 0 || apiBlk[i] > apiBlk[bSpillUnder100])) bSpillUnder100 = i;
        }
        ruleSel[1] = cfg[bOcc].minb;             // highest occupancy
        ruleSel[2] = cfg[bOccNoSpill].minb;      // highest occupancy, no spill
        ruleSel[3] = cfg[0].minb;                // whatever the compiler chose
        ruleSel[4] = cfg[bSpillUnder100].minb;   // highest occupancy, spill < 100 B
        ruleSel[5] = cfg[bLowOcc].minb;          // lowest occupancy
        ruleSel[6] = cfg[NCFG/2].minb;           // the middle of the sweep
    }

    double chosenG = 0.0; int chosenRegs = 0;
    for (int i = 0; i < NCFG; ++i)
        if (cfg[i].minb == CHOSEN_MINB) { chosenG = gel[i]; chosenRegs = at[i].numRegs; }
    // Within 20% of the fastest row measured in THIS run (the top four rows are
    // within 1.0-1.37x of each other and their ordering is not stable run to
    // run), and internally consistent with TODO 4: if you said the compiler's
    // unconstrained choice is not the best, you may not then ship it.
    const int selfConsistent = !(PRED_UNCONSTRAINED != 1 && chosenRegs == unconstrainedRegs);
    const int chosenOk = (chosenG >= 0.80 * bestG) && selfConsistent;
    const int ruleOk = (CHOSEN_RULE >= 1 && CHOSEN_RULE <= 6 &&
                        ruleSel[CHOSEN_RULE] == CHOSEN_MINB && chosenOk);

    int capOk = 1;
    for (int i = 0; i < NCFG; ++i)
        if (at[i].localSizeBytes > 0 && registerCapFor(TPB, cfg[i].minb) != at[i].numRegs) capOk = 0;
    int measuredSpillAt = 0, measuredCapAt = 0;
    for (int i = 0; i < NCFG; ++i)
        if (at[i].localSizeBytes > 0) { measuredSpillAt = cfg[i].minb; break; }
    for (int i = 0; i < NCFG; ++i)
        if (at[i].numRegs < unconstrainedRegs) { measuredCapAt = cfg[i].minb; break; }
    const int capBindOk = (capBindsAt(TPB, unconstrainedRegs) == measuredCapAt);

    printf("\n-- scoring ------------------------------------------------------------\n");
    printf("  unconstrained build uses %d registers; the cap first BINDS at\n"
           "  __launch_bounds__(%d, %d); you said %d  -> %s\n",
           unconstrainedRegs, TPB, measuredCapAt, capBindsAt(TPB, unconstrainedRegs),
           capBindOk ? "correct" : "WRONG");
    int lastNoSpillRegs = unconstrainedRegs;
    for (int i = 0; i < NCFG; ++i)
        if (at[i].localSizeBytes == 0) lastNoSpillRegs = at[i].numRegs;
    printf("  ...and ptxas does not SPILL until (%d, %d). Everything between those\n"
           "  two bounds is rescheduling: %d registers per thread given up for free.\n",
           TPB, measuredSpillAt, unconstrainedRegs - lastNoSpillRegs);
    printf("  register cap formula matches ptxas on every spilling row: %s\n",
           capOk ? "yes" : "NO");
    printf("  fastest build is __launch_bounds__(%d, %d) at %.3f Gelem/s\n",
           TPB, cfg[bestIdx].minb, bestG);
    printf("  100%%-occupancy build is %.2fx slower  -> bucket %d, you said %d  %s\n",
           maxOccRatio, predBucket, PRED_MAXOCC_BUCKET,
           (predBucket == PRED_MAXOCC_BUCKET) ? "correct" : "WRONG");
    printf("  compiler's unconstrained build is %.2fx off the best -> answer %d,"
           " you said %d  %s\n", unconRatio, predUncon, PRED_UNCONSTRAINED,
           (predUncon == PRED_UNCONSTRAINED) ? "correct" : "WRONG");
    printf("  your operating point minBlocks=%d: %.3f Gelem/s = %.1f%% of best  %s\n",
           CHOSEN_MINB, chosenG, 100.0 * chosenG / bestG, chosenOk ? "ok" : "TOO SLOW");
    printf("  your rule %d selects minBlocks=%d from the measured table  %s\n",
           CHOSEN_RULE, (CHOSEN_RULE >= 1 && CHOSEN_RULE <= 6) ? ruleSel[CHOSEN_RULE] : -1,
           ruleOk ? "ok" : "does not select your own (or the best) operating point");

    int score = 0;
    score += capOk ? 2 : 0;
    score += capBindOk ? 1 : 0;
    score += (predBucket == PRED_MAXOCC_BUCKET) ? 2 : 0;
    score += (predUncon == PRED_UNCONSTRAINED) ? 1 : 0;
    score += chosenOk ? 2 : 0;
    score += ruleOk ? 2 : 0;
    printf("\n  SCORE: %d/10\n", score);

    // ------------------------------------------- validation, separate pass
    printf("\n-- validation (second, untimed pass) -----------------------------------\n");
    int numOk = 1;
    double worst = 0.0;
    float *ref = (float*)malloc(sizeof(float) * (size_t)L);
    for (int i = 0; i < NCFG && numOk; ++i) {
        CHECK(cudaMemset(dy, 0, n * 4));
        cfg[i].run();
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hy, dy, n * 4, cudaMemcpyDeviceToHost));
        for (int sig = 0; sig < nsig; sig += 1021) {
            cpuReference(hx, ref, nsig, L, sig, hA, hB);
            for (int t = 0; t < L; ++t) {
                double e = fabs((double)hy[(size_t)t * nsig + sig] - (double)ref[t]);
                double sc = fabs((double)ref[t]) + 1e-6;
                if (e / sc > worst) worst = e / sc;
            }
        }
        if (worst > 1e-4) { numOk = 0; printf("  config minb=%d mismatches\n", cfg[i].minb); }
    }
    printf("  worst relative error over all 9 builds: %.3g  (%s)\n",
           worst, numOk ? "ok" : "FAIL");

    free(ref); free(hx); free(hy);
    CHECK(cudaFree(dx)); CHECK(cudaFree(dy));
    CHECK(cudaFree(sink)); CHECK(cudaFree(stream));
    printf("\nOVERALL: %s\n", (score == 10 && numOk) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (score == 10 && numOk) ? 0 : 1;
}
