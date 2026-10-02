// =============================================================================
// Module 20 / Exercise 2 — SOLUTION
// Four slow kernels, four different reasons, four different fixes.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
// RUN  : exercise02_solution.exe
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

// --------------------------------- TODO 2 (solved) ---------------------------
// Diagnosis: long-scoreboard.  At one block per SM there are 4 warps per
// scheduler; each has exactly ONE global load outstanding at a time because the
// `#pragma unroll 1` forbids the compiler from starting iteration i+1's loads
// before iteration i's store.  Example 2 measured the requirement: ~4
// outstanding 128 B warp-loads per scheduler.  We have 1.
// Fix: raise memory-level parallelism, not occupancy (occupancy is the budget).
// Issue 8 independent loads, then consume them.
#define AMLP 8
__global__ void kA_fast(const float *__restrict__ x, const float *__restrict__ y,
                        float *o, unsigned n)
{
    unsigned stride = gridDim.x * blockDim.x;
    unsigned base   = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned i = base;
    for (; i + (AMLP-1)*stride < n; i += AMLP*stride) {
        float xv[AMLP], yv[AMLP];
        #pragma unroll
        for (int c = 0; c < AMLP; ++c) { xv[c] = x[i + c*stride]; yv[c] = y[i + c*stride]; }
        #pragma unroll
        for (int c = 0; c < AMLP; ++c) o[i + c*stride] = fmaf(2.0f, xv[c], yv[c]);
    }
    for (; i < n; i += stride) o[i] = fmaf(2.0f, x[i], y[i]);   // tail, exactly once
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

// --------------------------------- TODO 3 (solved) ---------------------------
// Diagnosis: execution dependency.  Every FFMA reads the accumulator the
// previous FFMA wrote, so the kernel runs at the FFMA LATENCY (4 cycles),
// not at its throughput (1 cycle).  No amount of occupancy is available
// (the budget is one block per SM) and more outstanding loads would do
// nothing: there is exactly one load in the whole kernel.
// Fix: four partial accumulators.  That is a reassociation of a floating-point
// sum, so the answer changes in the last bits - which is why the validator
// uses a reference computed in double and a tolerance scaled by the sum of
// the magnitudes, not by the result (Module 16's rule).
__global__ void kC_fast(const float *__restrict__ x, float *o, int K, unsigned n)
{
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i], w = fmaf(x[i], 0.5f, 1.0e-7f);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int t = 0; t < K / CSTEP; ++t) {               // four chains, same
        #pragma unroll                                  // CSTEP FFMAs per branch
        for (int u = 0; u < CSTEP/4; ++u) {
            s0 = fmaf(v, w, s0);
            s1 = fmaf(v, w, s1);
            s2 = fmaf(v, w, s2);
            s3 = fmaf(v, w, s3);
        }
    }
    o[i] = (s0 + s1) + (s2 + s3);
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

// --------------------------------- TODO 4 (solved) ---------------------------
// Diagnosis: not-selected / issue-limited.  At 12 warps per scheduler there
// are three times as many eligible warps as the scheduler can ever issue, and
// the K independent divisions already give enormous ILP.  The kernel is doing
// 64 IEEE-correct divisions per element, and an IEEE fp32 divide is not one
// instruction: nvcc emits MUFU.RCP plus several FFMA/FADD plus an FCHK
// range check.  Nothing about warps or chains can help; the only lever left
// is to issue fewer instructions for the same answer.
// Fix: __fdividef - one MUFU.RCP and one FMUL, ~2 ulp, valid for |y| < 2^126.
__global__ void kD_fast(const float *__restrict__ x, float *o, unsigned n, int K)
{
    unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = x[i], s = 0.0f;
        #pragma unroll 8
        for (int k = 1; k <= K; ++k) s += __fdividef((float)k, v + (float)k);
        o[i] = s;
    }
}
// ----------------------------- end TODO 4 ------------------------------------

// =============================== TODO 1 (solved) =============================
// 1 = long-scoreboard: waiting on global memory, too few requests in flight
// 2 = bandwidth-bound: the DRAM bus is already saturated
// 3 = execution dependency: a serial chain running at instruction latency
// 4 = not-selected / issue-limited: more eligible warps than issue slots
static const int diagnosis[4] = { 1, 2, 3, 4 };
// =============================== TODO 5 (solved) =============================
// Predicted speedup of the best available fix, per kernel.
//   A : one outstanding load per warp -> eight.  The budget gives 4 warps per
//       scheduler, so the kernel already has 4 warp-loads in flight; the fix
//       takes it past Example 2's knee of 4.  Expect ~2x, not 8x.
//   B : nothing helps.  1.0x.
//   C : one warp per scheduler and one chain = 1 of the 4 instructions in
//       flight the FP32 pipe needs.  Four accumulators supply exactly 4.
//       Expect close to the full 4x; the loop's own three instructions per
//       CSTEP FFMAs eat a little of it, so ~3.2x.
//   D : the divide shrinks from ~10 instructions to 2, but the surrounding
//       FADD/I2F work does not shrink: expect ~2x, not 5x.
static const double predSpeedup[4] = { 2.0, 1.0, 3.2, 2.0 };
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
    printf("=== Module 20 / Exercise 2 (solution) - same symptom, four causes ===\n\n");

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
