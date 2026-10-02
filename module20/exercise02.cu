// =============================================================================
// Module 20 / Exercise 2 - four slow kernels, four different reasons.
//
// GOAL : Low throughput is a symptom, not a diagnosis.  Four kernels here are
//        all far from their respective ceilings, for four different reasons,
//        and each one is cured by a different lever.  Apply the wrong lever
//        and nothing happens - the program measures that too.
//
// You may change any kernel BODY.  You may not change any launch
// configuration: each kernel's grid is a budget imposed by its caller, and
// kernel C's thread count is fixed by the problem itself.
//
// The program prints two pieces of evidence before it scores you:
//   1. each original kernel's time against occupancy (or, for C, against
//      block shape at a constant thread count);
//   2. where each kernel sits relative to a hardware bound.
// That is the same evidence a profiler would give you.  Module 23 names the
// counters; `ncu` cannot be run on this machine, so the harness constructs
// the evidence directly.
//
// Commit to P1-P5 before you build.  Scored out of 10.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
// RUN  : exercise02.exe
// SASS : nvcc -arch=sm_89 -O3 -cubin -o exercise02.cubin exercise02.cu
//        cuobjdump -sass exercise02.cubin
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <thread>
#include <chrono>
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
#define THR        128
#define NELEM      (1u << 24)          // 16.7 M floats = 64 MB per array
#define KCHAIN     262144              // length of kernel C's accumulator chain
#define CSTEP      16                  // FFMAs between two loop branches, both versions
#define NC_ELEM    5120u               // kernel C: the problem has only this many items
#define KTERMS     64                  // divides per element in kernel D
#define PEAK_GBS   432.0

// Budgets.  Each kernel is launched at the configuration its caller imposes;
// you may change the kernel body, never the launch.
#define BLOCKS_A   1                   // blocks per SM
#define BLOCKS_B   8
#define BLOCKS_C   1
#define BLOCKS_D   12

// =============================================================================
//  A. y[i] = 2*x[i] + y0[i], grid-stride, one block per SM.
// =============================================================================
__global__ void kA_slow(const float *__restrict__ x, const float *__restrict__ y,
                        float *o, unsigned n)
{
    unsigned stride = gridDim.x * blockDim.x;
    #pragma unroll 1
    for (unsigned i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride)
        o[i] = fmaf(2.0f, x[i], y[i]);
}

// --------------------------------- TODO 2 ------------------------------------
// Write a faster kA that computes exactly the same thing, at the same launch
// configuration (BLOCKS_A blocks per SM).  Every element of o must be written
// exactly once; the validator checks for unwritten elements, and n is not a
// multiple of anything convenient once you start processing several elements
// per iteration.
//
// Gate: >= 1.70x against kA_slow.
__global__ void kA_fast(const float *__restrict__ x, const float *__restrict__ y,
                        float *o, unsigned n)
{
    // YOUR CODE HERE
    (void)x; (void)y; (void)o; (void)n;
}
// ----------------------------- end TODO 2 ------------------------------------

// =============================================================================
//  B. o[i] = x[i] + y[i], full occupancy, vectorised, already at the ceiling.
// =============================================================================
__global__ void kB(const float4 *__restrict__ x, const float4 *__restrict__ y,
                   float4 *o, unsigned n4)
{
    unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x*blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 a = x[i], b = y[i];
        o[i] = make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
    }
}

// =============================================================================
//  C. s = sum over k of x[i] * c_k, accumulated in one float.  One block/SM.
// =============================================================================
__global__ void kC_slow(const float *__restrict__ x, float *o, int K, unsigned n)
{
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i], w = fmaf(x[i], 0.5f, 1.0e-7f);
    float s = 0.0f;
    for (int t = 0; t < K / CSTEP; ++t) {               // one serial chain
        #pragma unroll
        for (int u = 0; u < CSTEP; ++u) s = fmaf(v, w, s);
    }
    o[i] = s;
}

// --------------------------------- TODO 3 ------------------------------------
// Write a faster kC.  The thread count is fixed by the problem (NC_ELEM
// items), so you cannot add warps.  Keep CSTEP fused multiply-adds between
// consecutive loop branches so that the comparison is not contaminated by a
// different amount of loop overhead.
//
// Your result will not be bit-identical to kC_slow's and that is expected:
// say in your own notes why, and check that the validator's tolerance is the
// right one for that kind of difference.
//
// Gate: >= 2.50x against kC_slow.
__global__ void kC_fast(const float *__restrict__ x, float *o, int K, unsigned n)
{
    // YOUR CODE HERE
    (void)x; (void)o; (void)K; (void)n;
}
// ----------------------------- end TODO 3 ------------------------------------

// =============================================================================
//  D. o[i] = sum_{k=1..K} k / (x[i] + k).  Full occupancy.
// =============================================================================
__global__ void kD_slow(const float *__restrict__ x, float *o, unsigned n, int K)
{
    unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = x[i], s = 0.0f;
        #pragma unroll 8
        for (int k = 1; k <= K; ++k) s += (float)k / (v + (float)k);
        o[i] = s;
    }
}

// --------------------------------- TODO 4 ------------------------------------
// Write a faster kD at the same launch configuration.  The validator accepts
// a relative error up to 1e-5, which is looser than kD_slow achieves; read
// that as permission, and work out what it is permission for.  Count the SASS
// instructions in the inner loop before and after, with
// `cuobjdump -sass`; the ratio you measure should be close to the ratio of
// those counts, and if it is not, your diagnosis is wrong.
//
// Gate: >= 1.40x against kD_slow.
__global__ void kD_fast(const float *__restrict__ x, float *o, unsigned n, int K)
{
    // YOUR CODE HERE
    (void)x; (void)o; (void)n; (void)K;
}
// ----------------------------- end TODO 4 ------------------------------------

// =============================== TODO 1 ======================================
// P1.  Classify each of the four kernels.  Each code is used exactly once.
//   1 = long-scoreboard: warps waiting on global memory, too few requests in
//       flight to cover the latency
//   2 = bandwidth-bound: the DRAM bus is already saturated; concurrency is
//       not the problem and cannot be the fix
//   3 = execution dependency: a serial arithmetic chain running at the
//       instruction latency instead of its throughput
//   4 = not-selected / issue-limited: more eligible warps than there are
//       issue slots; the kernel is paying for instructions, not for stalls
// Leave any entry at 0 and the program stops instead of scoring you.
static const int diagnosis[4] = { 0, 0, 0, 0 };     // YOUR CODE HERE
// =============================== TODO 5 ======================================
// P2-P5.  The speedup you expect from the best available fix, per kernel.
// Scored to within a factor 1.5 on at least 3 of the 4.  One of these four
// numbers is not a guess at all; you should be able to derive it from the two
// numbers Example 1 measures.
static const double predSpeedup[4] = { 0.0, 0.0, 0.0, 0.0 };   // YOUR CODE HERE
// ============================= end TODOs =====================================

__global__ void warmStream(const float4 *__restrict__ s, float *o, size_t n)
{
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    float4 acc = make_float4(0,0,0,0);
    for (; i < n; i += gridDim.x*(size_t)blockDim.x) {
        float4 v = s[i]; acc.x+=v.x; acc.y+=v.y; acc.z+=v.z; acc.w+=v.w; }
    if (acc.x == 1e30f) o[0] = acc.x;
}
__global__ void warmFfma(float *o, int iters)
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
__global__ void fillPattern(float *x, unsigned n, unsigned salt)
{
    for (unsigned i = blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += gridDim.x*blockDim.x) {
        unsigned h = (i ^ salt) * 2654435761u;
        x[i] = 1.0f + (float)(h >> 20) * (1.0f/4096.0f);   // in [1, 5)
    }
}

// ------------------------------------------------------------------ harness --
static float *dx, *dy, *doo;
static int g_blocks = 1;
typedef void (*launch_t)(void);
static void lA_slow(void){ kA_slow<<<g_blocks*SM_COUNT,THR>>>(dx,dy,doo,NELEM); }
static void lA_fast(void){ kA_fast<<<g_blocks*SM_COUNT,THR>>>(dx,dy,doo,NELEM); }
static void lB     (void){ kB     <<<g_blocks*SM_COUNT,THR>>>((const float4*)dx,
                                    (const float4*)dy,(float4*)doo,NELEM/4); }
static int g_cthr = 32;        // kernel C: block SHAPE varies, thread count does not
static void lC_slow(void){ kC_slow<<<NC_ELEM/g_cthr,g_cthr>>>(dx,doo,KCHAIN,NC_ELEM); }
static void lC_fast(void){ kC_fast<<<NC_ELEM/g_cthr,g_cthr>>>(dx,doo,KCHAIN,NC_ELEM); }
static void lD_slow(void){ kD_slow<<<g_blocks*SM_COUNT,THR>>>(dx,doo,NELEM,KTERMS); }
static void lD_fast(void){ kD_fast<<<g_blocks*SM_COUNT,THR>>>(dx,doo,NELEM,KTERMS); }

static double timeOne(launch_t f, int it)
{
    cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < it; ++i) f();
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms,a,b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms / it;
}

static const char *diagName(int d)
{
    switch (d) {
    case 1: return "long-scoreboard (global)";
    case 2: return "bandwidth-bound";
    case 3: return "execution dependency";
    case 4: return "not-selected / issue";
    default: return "unset";
    }
}

#define NSW 4
#define NEV 16          /* 4 kernels x 4 occupancies */

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 20 / Exercise 2 - same symptom, four different causes ===\n\n");

    {
        volatile int d0 = diagnosis[0], d1 = diagnosis[1];
        volatile int d2 = diagnosis[2], d3 = diagnosis[3];
        volatile double p0 = predSpeedup[0], p1 = predSpeedup[1];
        volatile double p2 = predSpeedup[2], p3 = predSpeedup[3];
        if (d0 <= 0 || d1 <= 0 || d2 <= 0 || d3 <= 0) {
            printf("Set TODO 1 first.\n"); return 0;
        }
        if (p0 <= 0.0 || p1 <= 0.0 || p2 <= 0.0 || p3 <= 0.0) {
            printf("Set TODO 5 first.\n"); return 0;
        }
    }

    const size_t nb = (size_t)NELEM * sizeof(float);
    CHECK(cudaMalloc(&dx, nb)); CHECK(cudaMalloc(&dy, nb)); CHECK(cudaMalloc(&doo, nb));
    fillPattern<<<1024,256>>>(dx, NELEM, 12345u);
    fillPattern<<<1024,256>>>(dy, NELEM, 67890u);
    CHECK(cudaDeviceSynchronize());

    // ---- warm-up, with an operating-point guard (spec 12.4 + 12.5b) --------
    // After a few minutes of back-to-back benchmarking this part pins its
    // memory clock at 6001 MHz instead of 9001 and every bandwidth figure
    // below drops by ~20%.  The speedup gates here are ratios and survive
    // that, but kernel B's "is it already at the bus?" evidence does not.
    // Probe the ceiling with kernel B itself; if it is low, idle and re-warm.
    printf("-- warming up: 1500 ms streaming, then 500 ms compute ----------------\n");
    {
        const double CEIL_FLOOR_GBPS = 330.0;   // healthy ~380, power-capped ~300
        cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        for (int attempt = 0; ; ++attempt) {
            float el = 0; CHECK(cudaEventRecord(w0));
            while (el < 1500.0f) { warmStream<<<320,256>>>((const float4*)dx, doo, NELEM/4);
                CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
                CHECK(cudaEventElapsedTime(&el,w0,w1)); }
            el = 0; CHECK(cudaEventRecord(w0));
            while (el < 500.0f) { warmFfma<<<480,128>>>(doo, 2000);
                CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
                CHECK(cudaEventElapsedTime(&el,w0,w1)); }
            CHECK(cudaGetLastError());

            g_blocks = BLOCKS_B;
            CHECK(cudaEventRecord(w0));
            for (int k = 0; k < 8; ++k) lB();
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            float pms = 0; CHECK(cudaEventElapsedTime(&pms, w0, w1));
            double g = 3.0*(double)NELEM*4.0 / (((double)pms/8.0)*1e-3) / 1e9;
            if (g >= CEIL_FLOOR_GBPS) {
                printf("   warm-up complete: the ceiling probe reads %.1f GB/s\n", g);
                break;
            }
            if (attempt >= 5) {
                printf("   WARNING: the ceiling probe still reads only %.1f GB/s after\n"
                       "   %d cool-downs.  This part is power limited right now; the\n"
                       "   ratios below are real but the %%-of-peak evidence is not\n"
                       "   this GPU's healthy figure.  Let it idle and re-run.\n",
                       g, attempt);
                break;
            }
            printf("   ceiling probe reads only %.1f GB/s (healthy is >= %.0f): this\n"
                   "   part is power or thermally limited.  Idling 10 s and warming\n"
                   "   again (%d/5).\n", g, CEIL_FLOOR_GBPS, attempt + 1);
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    }
    CHECK(cudaGetLastError());

    // ------------------------------------------------- evidence: occupancy ---
    // All 16 points timed back to back in one rotated sweep.
    const int sweepBlocks[NSW] = { 1, 2, 4, 12 };
    const int sweepCthr  [NSW] = { 32, 64, 128, 256 };   // kernel C only
    launch_t orig[4] = { lA_slow, lB, lC_slow, lD_slow };
    double ev[NEV];
    for (int i = 0; i < NEV; ++i) ev[i] = 1e30;
    for (int s = 0; s < NEV; ++s)
        for (int q = 0; q < NEV; ++q) {
            int p = (q + s) % NEV, k = p / NSW, b = p % NSW;
            g_blocks = sweepBlocks[b];
            g_cthr   = sweepCthr[b];
            double t = timeOne(orig[k], 3);
            if (t < ev[p]) ev[p] = t;
        }
    CHECK(cudaGetLastError());

    printf("\n-- evidence 1: does more occupancy help? (ms, min of %d sweeps) ------\n", NEV);
    printf("   %-10s", "kernel");
    for (int b = 0; b < NSW; ++b) printf("%10d", sweepBlocks[b]);
    printf("%12s%12s\n", "1->12", "budget");
    const char *kn[4] = { "A", "B", "C", "D" };
    const int budget[4] = { BLOCKS_A, BLOCKS_B, BLOCKS_C, BLOCKS_D };
    for (int k = 0; k < 4; ++k) {
        printf("   %-10s", kn[k]);
        for (int b = 0; b < NSW; ++b) printf("%10.3f", ev[k*NSW+b]);
        printf("%11.2fx%12d\n", ev[k*NSW]/ev[k*NSW+NSW-1], budget[k]);
    }
    printf("   (A, B, D: columns are blocks per SM, one-wave launches.\n"
           "    C: the problem has only %u items, so its thread count CANNOT be\n"
           "    changed; its columns are 32/64/128/256 threads per block at a\n"
           "    constant %u threads - four block shapes, one occupancy.)\n",
           NC_ELEM, NC_ELEM);

    printf("\n-- evidence 2: where is each kernel against a hardware bound? --------\n");
    {
        double gbsA = 3.0*NELEM*4.0 / (ev[0*NSW+3]*1e-3) / 1e9;
        double gbsB = 3.0*NELEM*4.0 / (ev[1*NSW+3]*1e-3) / 1e9;
        printf("   A at 12 blk/SM : %6.1f GB/s = %4.1f%% of %.0f GB/s pin peak (3N traffic)\n",
               gbsA, 100.0*gbsA/PEAK_GBS, PEAK_GBS);
        printf("   B at 12 blk/SM : %6.1f GB/s = %4.1f%% of %.0f GB/s pin peak (3N traffic)\n",
               gbsB, 100.0*gbsB/PEAK_GBS, PEAK_GBS);
        printf("   C: %d FFMAs x %u threads -> %.0f GFLOP/s, against the ~20000\n"
               "      GFLOP/s FP32 ceiling Example 1 measures\n",
               KCHAIN, NC_ELEM,
               2.0*KCHAIN*(double)NC_ELEM/(ev[2*NSW+0]*1e-3)/1e9);
        printf("   D: traffic is 2N = %.1f GB/s - nowhere near the bus\n",
               2.0*NELEM*4.0/(ev[3*NSW+3]*1e-3)/1e9);
    }

    // -------------------------------------------------- the reader's fixes ---
    launch_t fixed[4] = { lA_fast, lB, lC_fast, lD_fast };
    double tOrig[4], tFix[4];
    for (int k = 0; k < 4; ++k) { tOrig[k] = 1e30; tFix[k] = 1e30; }
    for (int s = 0; s < 8; ++s)
        for (int q = 0; q < 8; ++q) {
            int p = (q + s) % 8, k = p >> 1;
            g_blocks = budget[k];
            double t = timeOne((p & 1) ? fixed[k] : orig[k], 3);
            if (p & 1) { if (t < tFix[k])  tFix[k]  = t; }
            else       { if (t < tOrig[k]) tOrig[k] = t; }
        }
    CHECK(cudaGetLastError());

    printf("\n-- the fixes, at each kernel's own budget -----------------------------\n");
    printf("   %-4s %-26s %10s %10s %10s %10s\n",
           "k", "your diagnosis", "orig ms", "fixed ms", "speedup", "predicted");
    for (int k = 0; k < 4; ++k)
        printf("   %-4s %-26s %10.3f %10.3f %9.2fx %9.2fx\n",
               kn[k], diagName(diagnosis[k]), tOrig[k], tFix[k],
               tOrig[k]/tFix[k], predSpeedup[k]);

    // ----------------------------------------------------------- validation --
    printf("\n-- validation (second, untimed pass) ----------------------------------\n");
    int ok = 1;
    {
        float *hx = (float*)malloc(nb), *hy = (float*)malloc(nb), *ho = (float*)malloc(nb);
        CHECK(cudaMemcpy(hx, dx, nb, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hy, dy, nb, cudaMemcpyDeviceToHost));

        // A
        g_blocks = BLOCKS_A; CHECK(cudaMemset(doo, 0, nb)); lA_fast();
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(ho, doo, nb, cudaMemcpyDeviceToHost));
        { double worst = 0; unsigned bad = 0;
          for (unsigned i = 0; i < NELEM; i += 9973u) {
              double ref = 2.0*(double)hx[i] + (double)hy[i];
              double e = fabs((double)ho[i]-ref)/fmax(1.0,fabs(ref));
              if (e > worst) worst = e; }
          for (unsigned i = 0; i < NELEM; i += 7919u) if (ho[i] == 0.0f) ++bad;
          printf("   A  worst rel err %.3e, unwritten %u  [%s]\n", worst, bad,
                 (worst <= 1e-6 && bad == 0) ? "OK" : "FAIL");
          if (!(worst <= 1e-6 && bad == 0)) ok = 0; }

        // B
        g_blocks = BLOCKS_B; CHECK(cudaMemset(doo, 0, nb)); lB();
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(ho, doo, nb, cudaMemcpyDeviceToHost));
        { double worst = 0;
          for (unsigned i = 0; i < NELEM; i += 9973u) {
              double ref = (double)hx[i] + (double)hy[i];
              double e = fabs((double)ho[i]-ref)/fmax(1.0,fabs(ref));
              if (e > worst) worst = e; }
          printf("   B  worst rel err %.3e  [%s]\n", worst, worst <= 1e-6 ? "OK" : "FAIL");
          if (worst > 1e-6) ok = 0; }

        // C : reassociated sum.  Reference in double; tolerance scaled by the
        //     accumulated magnitude S, not by |result| (Module 16's rule).
        g_blocks = BLOCKS_C; g_cthr = 32; CHECK(cudaMemset(doo, 0, nb)); lC_fast();
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(ho, doo, nb, cudaMemcpyDeviceToHost));
        { double worstRatio = 0; const double u = 5.96e-8;
          double gammaK = (double)KCHAIN*u / (1.0 - (double)KCHAIN*u);
          for (unsigned i = 0; i < NC_ELEM; i += 37u) {
              double w    = (double)fmaf(hx[i], 0.5f, 1.0e-7f);
              double term = (double)hx[i] * w;
              double ref  = term * (double)KCHAIN;
              double S    = fabs(term) * (double)KCHAIN;
              double err  = fabs((double)ho[i] - ref);
              double ratio = err / (gammaK * S);
              if (ratio > worstRatio) worstRatio = ratio; }
          printf("   C  worst err/(gamma_K*S) %.3f  [%s]\n", worstRatio,
                 worstRatio <= 1.0 ? "OK" : "FAIL");
          if (worstRatio > 1.0) ok = 0; }

        // D : __fdividef is ~2 ulp; compare against the double reference with a
        //     tolerance that admits that, and confirm the SLOW kernel is tighter.
        for (int which = 0; which < 2; ++which) {
            g_blocks = BLOCKS_D; CHECK(cudaMemset(doo, 0, nb));
            if (which) lD_fast(); else lD_slow();
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(ho, doo, nb, cudaMemcpyDeviceToHost));
            double worst = 0;
            for (unsigned i = 0; i < NELEM; i += 9973u) {
                double ref = 0.0;
                for (int k = 1; k <= KTERMS; ++k) ref += (double)k/((double)hx[i]+(double)k);
                double e = fabs((double)ho[i]-ref)/fmax(1.0,fabs(ref));
                if (e > worst) worst = e; }
            printf("   D%s worst rel err %.3e  [%s]\n", which ? "' " : "  ", worst,
                   worst <= 1e-5 ? "OK" : "FAIL");
            if (worst > 1e-5) ok = 0;
        }
        free(hx); free(hy); free(ho);
    }

    // ---------------------------------------------------------------- score --
    printf("\n-- scoring ------------------------------------------------------------\n");
    int score = 0;
    const int truth[4] = { 1, 2, 3, 4 };
    int nDiag = 0;
    for (int k = 0; k < 4; ++k) if (diagnosis[k] == truth[k]) ++nDiag;
    printf("   diagnoses correct            : %d/4\n", nDiag);
    if (nDiag == 4) score += 4;

    int nGate = 0;
    const double gate[4] = { 1.7, 0.0, 2.5, 1.4 };   // required speedup; B has none
    for (int k = 0; k < 4; ++k) {
        double sp = tOrig[k]/tFix[k];
        int pass = (k == 1) ? (sp > 0.90 && sp < 1.10) : (sp >= gate[k]);
        if (pass) ++nGate;
        printf("   kernel %s fix %-14s: %.2fx  (gate %s)  [%s]\n", kn[k],
               (k==1) ? "(must be 1.0x)" : "speedup", sp,
               (k==1) ? "0.90-1.10" : (k==0?">=1.70":(k==2?">=2.50":">=1.40")),
               pass ? "OK" : "FAIL");
    }
    if (nGate == 4) score += 3;

    int nPred = 0;
    for (int k = 0; k < 4; ++k) {
        double sp = tOrig[k]/tFix[k];
        int pass = (sp >= predSpeedup[k]/1.5 && sp <= predSpeedup[k]*1.5);
        if (pass) ++nPred;
    }
    printf("   speedup predictions within 1.5x: %d/4\n", nPred);
    if (nPred >= 3) score += 3;

    CHECK(cudaFree(dx)); CHECK(cudaFree(dy)); CHECK(cudaFree(doo));
    printf("\nSCORE: %d/10\n", score);
    printf("OVERALL: %s\n", (score == 10 && ok) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (score == 10 && ok) ? 0 : 1;
}
