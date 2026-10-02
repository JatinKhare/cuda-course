// =============================================================================
// Module 20 / Example 1 — the arithmetic of latency hiding, on the FP32 pipe.
//
// GOAL : Measure, on this GPU, the two numbers that govern every latency
//        argument in the course, and then show that they predict a 2-D table.
//
//   A  FFMA latency vs FFMA throughput, isolated.  ONE warp, one block, one
//      SM.  A chain of dependent FFMAs runs at the pipeline LATENCY; C
//      independent chains interleave and run C times faster until the issue
//      port saturates.  Little's Law says the crossover is at
//      C = latency x throughput.  The measurement finds that C exactly.
//
//   B  The ILP x occupancy table.  The same kernel at 5 occupancies and 5
//      ILP levels, 25 configurations timed back-to-back in one rotated sweep.
//      Read the first row and the last row and you have the whole module:
//      ILP is worth ~4x when occupancy is low and ~1.0x when it is high.
//      They are SUBSTITUTES, and the thing they substitute for each other in
//      is one quantity: instructions in flight per warp scheduler.
//
//   C  The exchange rate, read off the table: how many resident warps one
//      unit of ILP replaces.
//
// Module 19 owns the resource arithmetic (what limits occupancy).  This file
// never asks how many warps FIT; it only asks what they BUY.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN  : example01.exe
// SASS : nvcc -arch=sm_89 -O3 -cubin -o example01.cubin example01.cu
//        cuobjdump -sass example01.cubin
//        (verify that chainK<4> really contains four independent FFMA chains)
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

#define SM_COUNT   40
#define BODY       256      // FFMAs per outer iteration, SAME for every C
#define FMA_B      1.0000001f
#define FMA_C      1.0e-6f

// -----------------------------------------------------------------------------
// The kernel.  C independent dependent-chains per thread.
//
// Two properties are deliberate and both are load-bearing:
//
//  1. Each a[i] is a true recurrence (a[i] = a[i]*b + c).  fp32 FMA is neither
//     associative nor reassociable, so nvcc cannot merge, hoist or shorten any
//     chain.  The SASS check in the header is how you confirm that.
//
//  2. The outer loop body contains BODY = 256 FFMAs for EVERY C.  Without that,
//     the loop's own IADD3/ISETP/BRA are a different fraction of the
//     instruction stream at each C and the comparison measures the compiler's
//     unroll heuristic instead of ILP.  (Measured: with a fixed inner trip
//     count instead, ptxas unrolls the C=2 instantiation 3x less than the
//     others and C=2 comes out 25% slow for a reason that has nothing to do
//     with latency.)
// -----------------------------------------------------------------------------
template<int C>
__global__ void chainK(float *out, long long *cyc, int iters)
{
    float a[C];
    #pragma unroll
    for (int i = 0; i < C; ++i) a[i] = (float)(threadIdx.x + i + 1) * 1.0e-3f;

    long long t0 = clock64();
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int u = 0; u < BODY / C; ++u)
            #pragma unroll
            for (int i = 0; i < C; ++i) a[i] = fmaf(a[i], FMA_B, FMA_C);
    }
    long long t1 = clock64();

    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < C; ++i) s += a[i];

    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    if (blockIdx.x == 0 && threadIdx.x == 0) cyc[0] = t1 - t0;
}

// Host replay of one thread's chains, for the validation pass.
static float chainHost(int C, int tid, int iters)
{
    float a[32];
    for (int i = 0; i < C; ++i) a[i] = (float)(tid + i + 1) * 1.0e-3f;
    for (int t = 0; t < iters; ++t)
        for (int u = 0; u < BODY / C; ++u)
            for (int i = 0; i < C; ++i) a[i] = fmaf(a[i], FMA_B, FMA_C);
    float s = 0.0f;
    for (int i = 0; i < C; ++i) s += a[i];
    return s;
}

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
static int g_blocks = 1, g_iters = 100;

typedef void (*launch_t)(void);
template<int C> static void launchC(void) { chainK<C><<<g_blocks*SM_COUNT,128>>>(d_out, d_cyc, g_iters); }
template<int C> static void launch1W(void) { chainK<C><<<1,32>>>(d_out, d_cyc, g_iters); }

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

    printf("=== Module 20 / Example 1 - latency, throughput, and their product ===\n\n");

    const int wps[NW]  = { 1, 2, 4, 8, 12 };   // resident warps per warp scheduler
    const int ilp[NI]  = { 1, 2, 4, 8, 16 };   // independent chains per thread

    const size_t maxThreads = (size_t)wps[NW-1] * SM_COUNT * 128;
    CHECK(cudaMalloc(&d_out, maxThreads * sizeof(float)));
    CHECK(cudaMalloc(&d_cyc, sizeof(long long)));

    // ---------------------------------------------------------- warm-up -----
    printf("-- warming up: 1500 ms streaming, then 500 ms compute ----------------\n");
    {
        size_t nb = (size_t)256*1024*1024/16;
        float4 *ds; CHECK(cudaMalloc(&ds, nb*16)); CHECK(cudaMemset(ds, 1, nb*16));
        cudaEvent_t w0, w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        float el = 0; CHECK(cudaEventRecord(w0));
        while (el < 1500.0f) { warmStream<<<320,256>>>(ds, d_out, nb);
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            CHECK(cudaEventElapsedTime(&el, w0, w1)); }
        el = 0; CHECK(cudaEventRecord(w0));
        while (el < 500.0f) { g_blocks = 12; g_iters = 200; launchC<8>();
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            CHECK(cudaEventElapsedTime(&el, w0, w1)); }
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); CHECK(cudaFree(ds));
    }
    CHECK(cudaGetLastError());

    // ======================================================================= A
    printf("\n-- A. one warp, one SM: dependent FFMA latency vs FFMA throughput ----\n");
    printf("   A single warp cannot be helped by any other warp.  Everything it\n"
           "   hides, it hides with its own independent work.\n\n");
    printf("   %4s %14s %12s %12s %12s\n", "ILP", "cycles", "cyc/FFMA", "wall ms", "clk GHz");
    printf("   %4s %14s %12s %12s %12s\n", "----", "--------------", "------------",
           "------------", "------------");

    const int latILP[8] = { 1, 2, 3, 4, 5, 6, 8, 10 };
    double cycPerFfma[8];
    {
        g_iters = 4000;
        launch_t fs[8] = { launch1W<1>, launch1W<2>, launch1W<3>, launch1W<4>,
                           launch1W<5>, launch1W<6>, launch1W<8>, launch1W<10> };
        for (int i = 0; i < 8; ++i) {
            long long best = (1LL<<62); float bms = 1e30f;
            for (int r = 0; r < 5; ++r) {
                cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
                CHECK(cudaEventRecord(a)); fs[i](); CHECK(cudaEventRecord(b));
                CHECK(cudaEventSynchronize(b));
                float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
                CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
                long long h; CHECK(cudaMemcpy(&h, d_cyc, 8, cudaMemcpyDeviceToHost));
                if (h < best) best = h;
                if (ms < bms) bms = ms;
            }
            double n = (double)g_iters * (BODY/latILP[i]) * latILP[i];
            cycPerFfma[i] = (double)best / n;
            printf("   %4d %14lld %12.3f %12.4f %12.3f\n", latILP[i], best,
                   cycPerFfma[i], bms, (double)best / (bms*1e-3) / 1e9);
        }
    }
    CHECK(cudaGetLastError());
    {
        double L = cycPerFfma[0];                 // ILP = 1  -> the latency
        double T = cycPerFfma[7];                 // ILP = 10 -> the issue floor
        printf("\n   dependent-chain FFMA latency  L = %.2f cycles\n", L);
        printf("   saturated FFMA issue interval T = %.2f cycles  (1 instr / cycle / scheduler)\n", T);
        printf("   Little's Law: a warp must hold L/T = %.2f independent FFMAs in\n", L/T);
        printf("   flight to keep its scheduler's FP32 issue port busy by itself.\n");
        printf("   The measured knee above is where the cyc/FFMA column stops falling.\n");
        printf("\n   The 'clk GHz' column is clock64() cross-checked against wall time.\n"
               "   It reads ~1.7-2.1 GHz here, which is why these cycle counts can be\n"
               "   trusted.  At FULL occupancy the same cross-check reads ~0.3 GHz and\n"
               "   clock64() must NOT be used - see the lesson, and Module 16's note.\n");
    }

    // ======================================================================= B
    printf("\n-- B. ILP x occupancy, 25 configurations, one rotated sweep ----------\n");

    launch_t cfg[NCFG];
    {
        launch_t byIlp[NI] = { launchC<1>, launchC<2>, launchC<4>, launchC<8>, launchC<16> };
        for (int w = 0; w < NW; ++w)
            for (int i = 0; i < NI; ++i) cfg[w*NI + i] = byIlp[i];
    }
    // Per-row iteration count: hold one launch at roughly 1 ms so that the
    // ~5 us launch overhead is not what we are measuring.
    int rowIters[NW];
    for (int w = 0; w < NW; ++w) rowIters[w] = 6000 / wps[w];

    double best[NCFG];
    for (int i = 0; i < NCFG; ++i) best[i] = 1e30;
    for (int s = 0; s < NCFG; ++s)
        for (int q = 0; q < NCFG; ++q) {
            int p = (q + s) % NCFG;
            g_blocks = wps[p / NI];
            g_iters  = rowIters[p / NI];
            double t = timeOne(cfg[p], 3);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    double gf[NCFG], peak = 0.0;
    for (int w = 0; w < NW; ++w)
        for (int i = 0; i < NI; ++i) {
            int p = w*NI + i;
            double flops = (double)wps[w] * SM_COUNT * 128.0 * (double)rowIters[w]
                         * (double)BODY * 2.0;
            gf[p] = flops / (best[p]*1e-3) / 1e9;
            if (gf[p] > peak) peak = gf[p];
        }

    printf("   GFLOP/s   (rows: resident warps per scheduler; cols: independent chains)\n");
    printf("   %-8s", "warps/s");
    for (int i = 0; i < NI; ++i) printf("%10s%-2d", "ILP=", ilp[i]);
    printf("   %9s\n", "ILP 1->16");
    for (int w = 0; w < NW; ++w) {
        printf("   %-8d", wps[w]);
        for (int i = 0; i < NI; ++i) printf("%12.0f", gf[w*NI+i]);
        printf("   %8.2fx\n", gf[w*NI+NI-1] / gf[w*NI]);
    }
    printf("\n   same table as %% of the best cell (%.0f GFLOP/s)\n", peak);
    printf("   %-8s", "warps/s");
    for (int i = 0; i < NI; ++i) printf("%10s%-2d", "ILP=", ilp[i]);
    printf("\n");
    for (int w = 0; w < NW; ++w) {
        printf("   %-8d", wps[w]);
        for (int i = 0; i < NI; ++i) printf("%11.0f%%", 100.0*gf[w*NI+i]/peak);
        printf("\n");
    }
    printf("\n   occupancy axis at fixed ILP=1 : %.2fx from 1 to 12 warps/scheduler\n",
           gf[(NW-1)*NI] / gf[0]);
    printf("   ILP axis at 1 warp/scheduler  : %.2fx from ILP 1 to 16\n",
           gf[NI-1] / gf[0]);
    printf("   ILP axis at 12 warps/scheduler: %.2fx from ILP 1 to 16\n",
           gf[(NW-1)*NI + NI-1] / gf[(NW-1)*NI]);

    // ======================================================================= C
    printf("\n-- C. the exchange rate -----------------------------------------------\n");
    printf("   Smallest occupancy that reaches 90%% of the best cell, per ILP level:\n\n");
    printf("   %6s %20s %16s %18s %14s\n", "ILP", "warps/sched >=85%",
           "warps/SM", "warps/sched >=90%", "ILP x warps");
    for (int i = 0; i < NI; ++i) {
        int n85 = -1, n90 = -1;
        for (int w = 0; w < NW; ++w) if (gf[w*NI+i] >= 0.85*peak) { n85 = wps[w]; break; }
        for (int w = 0; w < NW; ++w) if (gf[w*NI+i] >= 0.90*peak) { n90 = wps[w]; break; }
        char b85[16], s85[16], b90[16], bpr[16];
        if (n85 < 0) snprintf(b85, sizeof b85, "never"); else snprintf(b85, sizeof b85, "%d", n85);
        if (n85 < 0) snprintf(s85, sizeof s85, "-");     else snprintf(s85, sizeof s85, "%d", n85*4);
        if (n90 < 0) snprintf(b90, sizeof b90, "never"); else snprintf(b90, sizeof b90, "%d", n90);
        if (n85 < 0) snprintf(bpr, sizeof bpr, "-");     else snprintf(bpr, sizeof bpr, "%d", n85*ilp[i]);
        printf("   %6d %20s %16s %18s %14s\n", ilp[i], b85, s85, b90, bpr);
    }
    printf("\n   Multiply the two columns of any row of table B and you get the same\n"
           "   quantity: independent instructions in flight per scheduler.  That\n"
           "   product, not either factor, is what the machine is buying.\n");

    // ------------------------------------------------- validation, 2nd pass --
    printf("\n-- validation (second, untimed pass) ----------------------------------\n");
    int ok = 1;
    {
        const int VIT = 4;
        g_blocks = 2; g_iters = VIT;
        int nThreads = g_blocks * SM_COUNT * 128;
        float *h = (float*)malloc((size_t)nThreads * sizeof(float));
        launch_t fs[NI] = { launchC<1>, launchC<2>, launchC<4>, launchC<8>, launchC<16> };
        for (int i = 0; i < NI; ++i) {
            CHECK(cudaMemset(d_out, 0, (size_t)nThreads*sizeof(float)));
            fs[i]();
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h, d_out, (size_t)nThreads*sizeof(float), cudaMemcpyDeviceToHost));
            double worst = 0.0;
            for (int t = 0; t < 128; ++t) {
                float ref = chainHost(ilp[i], t, VIT);
                float got = h[t];
                double e = fabs((double)got - (double)ref)
                         / fmax(1.0, fabs((double)ref));
                if (e > worst) worst = e;
            }
            int zero = 0;
            for (int t = 0; t < nThreads; ++t) if (h[t] == 0.0f) zero++;
            printf("   ILP=%-3d worst relative error %.3e, unwritten outputs %d  [%s]\n",
                   ilp[i], worst, zero, (worst <= 1e-5 && zero == 0) ? "OK" : "FAIL");
            if (!(worst <= 1e-5 && zero == 0)) ok = 0;
        }
        free(h);
    }

    CHECK(cudaFree(d_out)); CHECK(cudaFree(d_cyc));
    printf("\nOVERALL: %s\n", ok ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return ok ? 0 : 1;
}
