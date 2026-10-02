// =============================================================================
// Module 20 / Exercise 1 - the ILP / occupancy exchange rate.
//
// GOAL : Build a kernel whose per-thread instruction-level parallelism and
//        whose occupancy can be dialled independently, then measure the 2-D
//        surface they span and show that it is really a function of ONE
//        variable.
//
// Before you build, write down:
//   P1  the dependent-FFMA latency of this GPU, in cycles          (TODO 3)
//   P2  what ILP 1 -> 8 is worth at the LOWEST occupancy swept     (TODO 4)
//   P3  what ILP 1 -> 8 is worth at the HIGHEST occupancy swept    (TODO 4)
//   P4  a rule for how many resident warps a given ILP replaces    (TODO 5)
//
// The program scores all four against its own measurements, out of 10.
//
// WARNING, and it is the hard part of this exercise: the compiler is allowed
// to merge, reorder, hoist and delete your "independent" chains, and if it
// does the experiment measures nothing.  Part A is an automatic check that
// they survived; `cuobjdump -sass` is how you find out WHY when it fails.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN  : exercise01.exe
// SASS : nvcc -arch=sm_89 -O3 -cubin -o exercise01.cubin exercise01.cu
//        cuobjdump -sass exercise01.cubin
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

#define SM_COUNT  40
#define BODY      256          // FFMAs per outer iteration, identical for all C
#define FB        1.0000001f
#define FC        1.0e-6f

// =============================== TODO 1 =====================================
// Build C independent dependent-chains of fused multiply-adds.
//
// Requirements - all four matter, three of them are checked automatically:
//   (a) chainBody() must execute EXACTLY `BODY` FFMAs per call, for every C.
//       Not BODY*C, not C.  If the FFMA count per outer-loop iteration varies
//       with C then the loop's own IADD3/ISETP/BRA are a different fraction of
//       the instruction stream in each configuration, and the table measures
//       the compiler's unroll heuristic instead of ILP.
//   (b) the C chains must be mutually independent: chain i's next value may
//       depend only on chain i's previous value.
//   (c) each chain must be a true recurrence, so the compiler cannot shorten
//       it.
//   (d) `a` must stay in registers.  Anything that forces a runtime index into
//       it puts the array in local memory, which Module 4 established is DRAM.
//
// Use FB and FC as the multiplier and the addend.
// =============================================================================
template<int C>
__device__ __forceinline__ void chainSeed(float (&a)[C], unsigned tid)
{
    // YOUR CODE HERE
    (void)a; (void)tid;
}

template<int C>
__device__ __forceinline__ void chainBody(float (&a)[C])
{
    // YOUR CODE HERE
    (void)a;
}
// ============================= end TODO 1 ====================================

template<int C>
__global__ void ilpKernel(float *out, long long *cyc, int iters)
{
    float a[C];
    chainSeed<C>(a, blockIdx.x * blockDim.x + threadIdx.x);
    long long t0 = clock64();
    for (int t = 0; t < iters; ++t) chainBody<C>(a);
    long long t1 = clock64();
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < C; ++i) s += a[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    if (blockIdx.x == 0 && threadIdx.x == 0) cyc[0] = t1 - t0;
}

static float chainHost(int C, int tid, int iters)
{
    float a[32];
    for (int i = 0; i < C; ++i) a[i] = (float)(tid + i + 1) * 1.0e-3f;
    for (int t = 0; t < iters; ++t)
        for (int u = 0; u < BODY / C; ++u)
            for (int i = 0; i < C; ++i) a[i] = fmaf(a[i], FB, FC);
    float s = 0.0f;
    for (int i = 0; i < C; ++i) s += a[i];
    return s;
}

// =============================== TODO 3 =====================================
// P1.  The dependent-FFMA latency of this GPU, in whole cycles.  Scored to
// within +/- 1 against Part A's measurement.  Leave it at 0 and the program
// stops with a message instead of scoring you.
#define PRED_LATENCY_CYCLES   0          // YOUR CODE HERE

// =============================== TODO 4 =====================================
// P2 / P3.  What going from ILP 1 to ILP 8 is worth, at the lowest and at the
// highest occupancy in the sweep.  P2 is scored to within a factor 1.33,
// P3 to within +/- 0.25 absolute.
#define PRED_ILP_GAIN_LOW     0.0        // YOUR CODE HERE
#define PRED_ILP_GAIN_HIGH    0.0        // YOUR CODE HERE

// =============================== TODO 5 =====================================
// P4.  The model.  Return the minimum number of resident warps per SCHEDULER
// needed to saturate the FP32 issue port when each thread carries C
// independent chains.  Derive it from PRED_LATENCY_CYCLES and the issue rate;
// do not fit it to the table, which is printed after this is called.  Scored
// against the measured 85%-of-peak crossing on at least 4 of the 5 ILP levels,
// to within a factor of two.
static int predNeededWarps(int C)
{
    // YOUR CODE HERE
    (void)C;
    return 1;
}
// ============================= end TODOs =====================================

__global__ void warmStream(const float4 *__restrict__ s, float *o, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float4 acc = make_float4(0, 0, 0, 0);
    for (; i < n; i += gridDim.x * (size_t)blockDim.x) {
        float4 v = s[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1e30f) o[0] = acc.x + acc.y + acc.z + acc.w;
}

// ------------------------------------------------------------------ harness --
static float *d_out; static long long *d_cyc;
static int g_blocks = 1, g_iters = 100, g_threads = 128;

// =============================== TODO 2 =====================================
// Set g_threads (threads per block) and g_blocks (blocks per SM) so that the
// launch `<<<g_blocks * 40, g_threads>>>` places exactly `warpsPerScheduler`
// resident warps on every one of the SM's four warp schedulers, in exactly one
// wave, with every SM holding the same number of blocks.
//
// Two facts you need, and one assumption to avoid:
//   - an Ada SM has four warp schedulers and a block's warps are handed to
//     them in linear warp order;
//   - the grid must be a whole number of waves or you are measuring a tail
//     (Module 1);
//   - do NOT assume that any (threads, blocks) pair with the right product
//     works.  A block whose warp count is not a multiple of four loads the
//     four schedulers unequally, and the busiest one sets the time.
static void setOccupancy(int warpsPerScheduler)
{
    // YOUR CODE HERE
    (void)warpsPerScheduler;
    g_threads = 0;
    g_blocks  = 0;
}
// ============================= end TODO 2 ====================================

typedef void (*launch_t)(void);
template<int C> static void launchGrid(void)
{ ilpKernel<C><<<g_blocks*SM_COUNT, g_threads>>>(d_out, d_cyc, g_iters); }
template<int C> static void launchOneWarp(void)
{ ilpKernel<C><<<1, 32>>>(d_out, d_cyc, g_iters); }

static double timeOne(launch_t f, int it)
{
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < it; ++i) f();
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms / it;
}

#define NW 5
#define NI 5
#define NCFG (NW*NI)

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 20 / Exercise 1 - the ILP/occupancy exchange rate ===\n\n");

    const int wps[NW] = { 1, 2, 4, 8, 12 };
    const int ilp[NI] = { 1, 2, 4, 8, 16 };

    setOccupancy(1);
    volatile int    _b  = g_blocks, _t = g_threads, _p1 = PRED_LATENCY_CYCLES;
    volatile double _p2 = PRED_ILP_GAIN_LOW, _p3 = PRED_ILP_GAIN_HIGH;
    if (_b <= 0 || _t <= 0)       { printf("Set TODO 2 first.\n"); return 0; }
    if (_p1 <= 0)                 { printf("Set TODO 3 first.\n"); return 0; }
    if (_p2 <= 0.0 || _p3 <= 0.0) { printf("Set TODO 4 first.\n"); return 0; }

    CHECK(cudaMalloc(&d_out, (size_t)12*SM_COUNT*128*sizeof(float)));
    CHECK(cudaMalloc(&d_cyc, sizeof(long long)));

    printf("-- warming up: 1500 ms streaming, then 500 ms compute ----------------\n");
    {
        size_t nb = (size_t)256*1024*1024/16;
        float4 *ds; CHECK(cudaMalloc(&ds, nb*16)); CHECK(cudaMemset(ds, 1, nb*16));
        cudaEvent_t w0, w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        float el = 0; CHECK(cudaEventRecord(w0));
        while (el < 1500.0f) { warmStream<<<320,256>>>(ds, d_out, nb);
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            CHECK(cudaEventElapsedTime(&el, w0, w1)); }
        setOccupancy(12); g_iters = 200;
        el = 0; CHECK(cudaEventRecord(w0));
        while (el < 500.0f) { launchGrid<8>();
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            CHECK(cudaEventElapsedTime(&el, w0, w1)); }
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); CHECK(cudaFree(ds));
    }
    CHECK(cudaGetLastError());

    int score = 0, maxScore = 10;

    // ======================================================================= A
    printf("\n-- A. one warp: is the ILP real? --------------------------------------\n");
    const int latC[6] = { 1, 2, 3, 4, 6, 8 };
    double cyc[6];
    {
        g_iters = 4000;
        launch_t fs[6] = { launchOneWarp<1>, launchOneWarp<2>, launchOneWarp<3>,
                           launchOneWarp<4>, launchOneWarp<6>, launchOneWarp<8> };
        printf("   %5s %14s %12s %12s\n", "ILP", "cycles", "cyc/FFMA", "vs ILP=1");
        for (int i = 0; i < 6; ++i) {
            long long best = (1LL<<62);
            for (int r = 0; r < 5; ++r) {
                fs[i](); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
                long long h; CHECK(cudaMemcpy(&h, d_cyc, 8, cudaMemcpyDeviceToHost));
                if (h < best) best = h;
            }
            cyc[i] = (double)best / ((double)g_iters * BODY);
            printf("   %5d %14lld %12.3f %11.2fx\n", latC[i], best, cyc[i], cyc[0]/cyc[i]);
        }
    }
    if (cyc[0] < 0.05) {
        printf("\n   TODO 1 looks empty: the timed loop costs %.4f cycles per\n"
               "   nominal FFMA, i.e. there is nothing in it.  Set TODO 1 first.\n",
               cyc[0]);
        CHECK(cudaFree(d_out)); CHECK(cudaFree(d_cyc));
        CHECK(cudaDeviceReset());
        return 0;
    }
    const double Lmeas = cyc[0], Tmeas = cyc[5];
    const double needed = Lmeas / Tmeas;
    printf("\n   measured dependent-FFMA latency   L = %.2f cycles\n", Lmeas);
    printf("   measured saturated issue interval T = %.2f cycles\n", Tmeas);
    printf("   Little's Law concurrency L/T        = %.2f independent FFMAs\n", needed);

    int chainsReal = (cyc[3] < 0.55*cyc[0]);
    printf("\n   [%s] the chains are independent in the generated code\n",
           chainsReal ? " OK " : "FAIL");
    if (!chainsReal)
        printf("        ILP=4 is not at least 1.8x faster than ILP=1, so the four\n"
               "        chains were merged, hoisted or eliminated.  Run cuobjdump\n"
               "        -sass and count the FFMAs between the two loop branches.\n");
    else score += 2;

    int latOk = (fabs((double)PRED_LATENCY_CYCLES - Lmeas) <= 1.0);
    printf("   [%s] predicted latency %d cycles vs measured %.2f\n",
           latOk ? " OK " : "FAIL", PRED_LATENCY_CYCLES, Lmeas);
    if (latOk) score += 2;

    // ======================================================================= B
    printf("\n-- B. the 2-D table: %d configurations, one rotated sweep -------------\n",
           NCFG);
    launch_t cfg[NCFG];
    {
        launch_t byIlp[NI] = { launchGrid<1>, launchGrid<2>, launchGrid<4>,
                               launchGrid<8>, launchGrid<16> };
        for (int w = 0; w < NW; ++w)
            for (int i = 0; i < NI; ++i) cfg[w*NI+i] = byIlp[i];
    }
    int rowIters[NW];
    for (int w = 0; w < NW; ++w) rowIters[w] = 6000 / wps[w];

    double best[NCFG];
    for (int i = 0; i < NCFG; ++i) best[i] = 1e30;
    for (int s = 0; s < NCFG; ++s)
        for (int q = 0; q < NCFG; ++q) {
            int p = (q + s) % NCFG;
            setOccupancy(wps[p / NI]);
            g_iters = rowIters[p / NI];
            double t = timeOne(cfg[p], 3);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    double gf[NCFG], peak = 0.0;
    for (int w = 0; w < NW; ++w)
        for (int i = 0; i < NI; ++i) {
            int p = w*NI+i;
            double flops = (double)wps[w]*SM_COUNT*128.0*(double)rowIters[w]*BODY*2.0;
            gf[p] = flops / (best[p]*1e-3) / 1e9;
            if (gf[p] > peak) peak = gf[p];
        }

    printf("   GFLOP/s\n   %-9s", "warps/sch");
    for (int i = 0; i < NI; ++i) printf("%9s%-2d", "ILP=", ilp[i]);
    printf("%12s\n", "ILP 1->8");
    for (int w = 0; w < NW; ++w) {
        printf("   %-9d", wps[w]);
        for (int i = 0; i < NI; ++i) printf("%11.0f", gf[w*NI+i]);
        printf("%11.2fx\n", gf[w*NI+3] / gf[w*NI]);
    }
    printf("\n   %% of best cell (%.0f GFLOP/s)\n   %-9s", peak, "warps/sch");
    for (int i = 0; i < NI; ++i) printf("%9s%-2d", "ILP=", ilp[i]);
    printf("\n");
    for (int w = 0; w < NW; ++w) {
        printf("   %-9d", wps[w]);
        for (int i = 0; i < NI; ++i) printf("%10.0f%%", 100.0*gf[w*NI+i]/peak);
        printf("\n");
    }

    double gainLow  = gf[0*NI+3] / gf[0*NI];
    double gainHigh = gf[(NW-1)*NI+3] / gf[(NW-1)*NI];
    printf("\n   ILP 1->8 at %2d warp/sched : predicted %.2fx, measured %.2fx\n",
           wps[0], (double)PRED_ILP_GAIN_LOW, gainLow);
    printf("   ILP 1->8 at %2d warps/sched: predicted %.2fx, measured %.2fx\n",
           wps[NW-1], (double)PRED_ILP_GAIN_HIGH, gainHigh);
    int pLow  = (gainLow  >= 0.75*PRED_ILP_GAIN_LOW  && gainLow  <= 1.33*PRED_ILP_GAIN_LOW);
    int pHigh = (fabs(gainHigh - PRED_ILP_GAIN_HIGH) <= 0.25);
    printf("   [%s] low-occupancy prediction   [%s] high-occupancy prediction\n",
           pLow ? " OK " : "FAIL", pHigh ? " OK " : "FAIL");
    if (pLow)  score += 1;
    if (pHigh) score += 1;

    // ======================================================================= C
    printf("\n-- C. the exchange rate ----------------------------------------------\n");
    printf("   %6s %18s %18s %10s\n", "ILP", "predicted warps", "measured warps", "verdict");
    int modelHits = 0;
    for (int i = 0; i < NI; ++i) {
        int pred = predNeededWarps(ilp[i]);
        int meas = -1;
        for (int w = 0; w < NW; ++w)
            if (gf[w*NI+i] >= 0.85*peak) { meas = wps[w]; break; }
        int hit = (meas > 0 && pred >= meas/2 && pred <= meas*2);
        if (hit) ++modelHits;
        printf("   %6d %18d %18d %10s\n", ilp[i], pred, meas, hit ? "ok" : "miss");
    }
    printf("\n   model within a factor of two on %d of %d rows\n", modelHits, NI);
    if (modelHits >= 4) score += 2;

    printf("\n   The invariant: ILP x warps-per-scheduler is the quantity the\n"
           "   scheduler sees.  %d is the number it needs.  Everything above that\n"
           "   number is inventory against non-uniform behaviour, not throughput.\n",
           PRED_LATENCY_CYCLES);

    // ------------------------------------------------- validation, 2nd pass --
    printf("\n-- validation (second, untimed pass) ----------------------------------\n");
    int ok = 1;
    {
        const int VIT = 4;
        setOccupancy(2); g_iters = VIT;
        int nThreads = g_blocks*SM_COUNT*g_threads;
        float *h = (float*)malloc((size_t)nThreads*sizeof(float));
        launch_t fs[NI] = { launchGrid<1>, launchGrid<2>, launchGrid<4>,
                            launchGrid<8>, launchGrid<16> };
        for (int i = 0; i < NI; ++i) {
            CHECK(cudaMemset(d_out, 0, (size_t)nThreads*sizeof(float)));
            fs[i](); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h, d_out, (size_t)nThreads*sizeof(float), cudaMemcpyDeviceToHost));
            double worst = 0.0;
            for (int t = 0; t < 128; ++t) {
                float ref = chainHost(ilp[i], t, VIT);
                double e = fabs((double)h[t]-(double)ref)/fmax(1.0, fabs((double)ref));
                if (e > worst) worst = e;
            }
            int zero = 0; for (int t = 0; t < nThreads; ++t) if (h[t] == 0.0f) zero++;
            printf("   ILP=%-3d worst rel err %.3e, unwritten %d  [%s]\n",
                   ilp[i], worst, zero, (worst <= 1e-5 && zero == 0) ? "OK" : "FAIL");
            if (!(worst <= 1e-5 && zero == 0)) ok = 0;
        }
        free(h);
    }
    if (ok) score += 2;

    CHECK(cudaFree(d_out)); CHECK(cudaFree(d_cyc));
    printf("\nSCORE: %d/%d\n", score, maxScore);
    printf("OVERALL: %s\n", (score == maxScore && ok) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (score == maxScore && ok) ? 0 : 1;
}
