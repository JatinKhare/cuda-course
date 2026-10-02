// =============================================================================
// Module 20 / Exercise 3 — SOLUTION
// Little's Law, quantitatively: compute the required concurrency, compute what
// the kernel actually supplies, reconcile, then close the gap.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
// RUN  : exercise03_solution.exe
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

#define SM_COUNT     40
#define THREADS      128            // fixed by the caller
#define BLOCKS_PER_SM 1             // fixed by the caller
#define STRIDE       ((unsigned)(BLOCKS_PER_SM*SM_COUNT*THREADS))   // 5120
#define NELEM        (STRIDE * 32u * 800u)     // 131,072,000 floats = 524 MB
#define PEAK_GBS     432.0
#define CLK_GHZ      1.90           // recovered at low occupancy by Example 1

// -----------------------------------------------------------------------------
// The kernel you are given.  One scalar load in flight per thread, ever.
// `#pragma unroll 1` is what pins it there: without it nvcc would software
// pipeline the loop and start the next load early.
// -----------------------------------------------------------------------------
__global__ void kBaseline(const float *__restrict__ x, float *part, unsigned n)
{
    const unsigned stride = gridDim.x * blockDim.x;
    const unsigned base   = blockIdx.x*blockDim.x + threadIdx.x;
    float s = 0.0f;
    #pragma unroll 1
    for (unsigned i = base; i < n; i += stride) s += x[i];
    part[base] = s;
}

// =============================== TODO 4 (solved) =============================
// Same launch, same traffic, same arithmetic.  The only thing that changes is
// how many bytes this thread has asked for and not yet received.
//
// Two multipliers are available and they compose:
//   - MLP: issue MLP independent loads before consuming any of them;
//   - width: a float4 load occupies ONE outstanding-request slot and carries
//     four times the payload (Module 5 / Module 11).
// 4 float4 loads in flight per thread = 4 slots x 16 B x 32 lanes = 2048 B per
// warp, against the baseline's 128 B.  16x the concurrency from the same 5120
// threads.
#define FIX_MLP            4
#define FIX_BYTES_PER_LANE 16
__global__ void kFixed(const float *__restrict__ x, float *part, unsigned n)
{
    const unsigned stride4 = gridDim.x * blockDim.x;
    const unsigned base4   = blockIdx.x*blockDim.x + threadIdx.x;
    const float4 *x4 = (const float4*)x;
    const unsigned n4 = n >> 2;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (unsigned i = base4; i + (FIX_MLP-1)*stride4 < n4; i += FIX_MLP*stride4) {
        float4 v[FIX_MLP];
        #pragma unroll
        for (int c = 0; c < FIX_MLP; ++c) v[c] = x4[i + c*stride4];
        #pragma unroll
        for (int c = 0; c < FIX_MLP; ++c) {
            acc.x += v[c].x; acc.y += v[c].y; acc.z += v[c].z; acc.w += v[c].w;
        }
    }
    part[base4] = (acc.x + acc.y) + (acc.z + acc.w);
}
// ============================= end TODO 4 ====================================

// =============================== TODO 1 (solved) =============================
// Bytes this launch has outstanding at any instant, device-wide, when every
// thread holds `loadsPerThread` requests of `bytesPerLane` bytes per lane.
// Concurrency is counted in BYTES because that is the unit the DRAM controller
// is rate-limited in.  Threads, not warps: a warp's coalesced load is just 32
// lanes' worth of bytes and the arithmetic is the same either way.
static double suppliedConcurrencyBytes(int threadsPerBlock, int blocksPerSM,
                                       int loadsPerThread, int bytesPerLane)
{
    return (double)threadsPerBlock * blocksPerSM * SM_COUNT
         * loadsPerThread * bytesPerLane;
}
// =============================== TODO 2 (solved) =============================
// Little's Law inverted.  concurrency = throughput x latency, so
// latency = concurrency / throughput.  Q in bytes, B in GB/s (1e9 B/s),
// answer in nanoseconds.
static double effectiveLatencyNs(double Qbytes, double B_GBs)
{
    return Qbytes / (B_GBs * 1e9) * 1e9;
}
// =============================== TODO 3 (solved) =============================
// Little's Law forward: how many bytes must be in flight to run at B_GBs
// when each request takes L_ns to come back.
static double requiredConcurrencyBytes(double B_GBs, double L_ns)
{
    return B_GBs * 1e9 * (L_ns * 1e-9);
}
// =============================== TODO 5 (solved) =============================
// Predicted bandwidth of the fixed kernel, in GB/s.  Once the supplied
// concurrency exceeds the requirement the kernel is no longer latency-bound,
// so the prediction is simply the measured streaming ceiling.  The harness
// measures that ceiling and prints it; ~410 GB/s on this part (Module 12).
#define PRED_FIXED_GBS  410.0
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
// The ceiling probe: full occupancy, wide loads, plenty of MLP.
__global__ void kCeiling(const float4 *__restrict__ x, float *part, unsigned n4)
{
    const unsigned stride = gridDim.x*blockDim.x;
    float4 acc = make_float4(0,0,0,0);
    for (unsigned i = blockIdx.x*blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = x[i]; acc.x+=v.x; acc.y+=v.y; acc.z+=v.z; acc.w+=v.w; }
    part[blockIdx.x*blockDim.x + threadIdx.x] = (acc.x+acc.y)+(acc.z+acc.w);
}
__global__ void fillPattern(float *x, unsigned n)
{
    for (unsigned i = blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += gridDim.x*blockDim.x)
        x[i] = 1.0f + (float)((i * 2654435761u) >> 21) * (1.0f/2048.0f); // [1,3)
}

static float *dx, *dpart;
typedef void (*launch_t)(void);
static void lBase (void){ kBaseline<<<BLOCKS_PER_SM*SM_COUNT, THREADS>>>(dx, dpart, NELEM); }
static void lFixed(void){ kFixed   <<<BLOCKS_PER_SM*SM_COUNT, THREADS>>>(dx, dpart, NELEM); }
static void lCeil (void){ kCeiling <<<12*SM_COUNT, THREADS>>>((const float4*)dx, dpart, NELEM/4); }

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

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 20 / Exercise 3 (solution) - Little's Law, quantitative ===\n\n");

    const size_t nb = (size_t)NELEM * sizeof(float);
    CHECK(cudaMalloc(&dx, nb));
    CHECK(cudaMalloc(&dpart, (size_t)12*SM_COUNT*THREADS*sizeof(float)));
    fillPattern<<<1024,256>>>(dx, NELEM);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // ---- warm-up, with an operating-point guard (spec 12.4 + 12.5b) --------
    // 1500 ms streaming ramps the memory P-state, 500 ms compute pulls the SM
    // clock back up.  That is necessary and not sufficient: after a few minutes
    // of back-to-back benchmarking this part pins its memory clock at 6001 MHz
    // instead of 9001 and the whole machine runs at ~320 GB/s instead of ~410.
    // TODO 5 is scored against an ABSOLUTE bandwidth, so that state would fail
    // a correct answer.  Probe the ceiling; if it is low, idle and re-warm.
    printf("-- warming up: 1500 ms streaming, then 500 ms compute ----------------\n");
    {
        const double CEIL_FLOOR_GBPS = 370.0;   // healthy ~410, power-capped ~320
        cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        for (int attempt = 0; ; ++attempt) {
            float el = 0; CHECK(cudaEventRecord(w0));
            while (el < 1500.0f) { warmStream<<<320,256>>>((const float4*)dx, dpart, NELEM/4);
                CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
                CHECK(cudaEventElapsedTime(&el,w0,w1)); }
            el = 0; CHECK(cudaEventRecord(w0));
            while (el < 500.0f) { warmFfma<<<480,128>>>(dpart, 2000);
                CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
                CHECK(cudaEventElapsedTime(&el,w0,w1)); }
            CHECK(cudaGetLastError());

            CHECK(cudaEventRecord(w0));
            for (int k = 0; k < 4; ++k) lCeil();
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            float pms = 0; CHECK(cudaEventElapsedTime(&pms, w0, w1));
            double g = (double)nb / (((double)pms/4.0)*1e-3) / 1e9;
            if (g >= CEIL_FLOOR_GBPS) {
                printf("   warm-up complete: the ceiling probe reads %.1f GB/s\n", g);
                break;
            }
            if (attempt >= 5) {
                printf("   WARNING: the ceiling probe still reads only %.1f GB/s after\n"
                       "   %d cool-downs.  This part is power limited right now.  The\n"
                       "   RATIOS below are real but the absolute GB/s are not this\n"
                       "   GPU's healthy figures, and TODO 5 is scored in absolute\n"
                       "   GB/s -- let it idle for a minute and re-run.\n", g, attempt);
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

    // --------------------------------------------------- time all three ------
    launch_t cfg[3] = { lBase, lFixed, lCeil };
    double best[3] = { 1e30, 1e30, 1e30 };
    for (int s = 0; s < 4; ++s)
        for (int q = 0; q < 3; ++q) {
            int p = (q + s) % 3;
            double t = timeOne(cfg[p], 3);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    const double bytes = (double)nb;
    const double gbsBase  = bytes/(best[0]*1e-3)/1e9;
    const double gbsFixed = bytes/(best[1]*1e-3)/1e9;
    const double gbsCeil  = bytes/(best[2]*1e-3)/1e9;

    printf("\n-- measured ----------------------------------------------------------\n");
    printf("   buffer                     %8.1f MB  (%.1fx the 48 MB L2)\n",
           bytes/1048576.0, bytes/48.0/1048576.0);
    printf("   baseline  <<<%d,%d>>>      %8.4f ms  %7.1f GB/s  %5.1f%% of pin peak\n",
           BLOCKS_PER_SM*SM_COUNT, THREADS, best[0], gbsBase, 100.0*gbsBase/PEAK_GBS);
    printf("   your fix  <<<%d,%d>>>      %8.4f ms  %7.1f GB/s  %5.1f%% of pin peak\n",
           BLOCKS_PER_SM*SM_COUNT, THREADS, best[1], gbsFixed, 100.0*gbsFixed/PEAK_GBS);
    printf("   ceiling probe (full occ)   %8.4f ms  %7.1f GB/s  %5.1f%% of pin peak\n",
           best[2], gbsCeil, 100.0*gbsCeil/PEAK_GBS);

    // ------------------------------------------------------ the arithmetic ---
    printf("\n-- the arithmetic ----------------------------------------------------\n");
    const double Q0   = suppliedConcurrencyBytes(THREADS, BLOCKS_PER_SM, 1, 4);
    const double Lns  = effectiveLatencyNs(Q0, gbsBase);
    const double Qreq = requiredConcurrencyBytes(gbsCeil, Lns);

    const double Q0ref   = (double)THREADS*BLOCKS_PER_SM*SM_COUNT*1*4;
    const double Lref    = Q0ref/(gbsBase*1e9)*1e9;
    const double Qreqref = gbsCeil*1e9*Lref*1e-9;

    printf("   supplied concurrency Q0    %10.0f B = %6.1f KB   [%s]\n",
           Q0, Q0/1024.0, fabs(Q0-Q0ref) <= 1.0 ? " OK " : "FAIL");
    printf("     = %d threads x 1 load x 4 B/lane\n", THREADS*BLOCKS_PER_SM*SM_COUNT);
    printf("   effective latency L        %10.1f ns = %5.0f cycles at %.2f GHz   [%s]\n",
           Lns, Lns*CLK_GHZ, CLK_GHZ, fabs(Lns-Lref) <= 0.01*Lref ? " OK " : "FAIL");
    printf("   required concurrency Q*    %10.0f B = %6.1f KB   [%s]\n",
           Qreq, Qreq/1024.0, fabs(Qreq-Qreqref) <= 0.01*Qreqref ? " OK " : "FAIL");
    printf("   shortfall Q*/Q0            %10.2fx\n", Qreq/Q0);
    printf("   predicted baseline BW = Q0/L = %7.1f GB/s   (measured %7.1f)\n",
           Q0/(Lns*1e-9)/1e9, gbsBase);

    printf("\n   Sanity-check against Module 4's 575-cycle dependent-load figure:\n"
           "   %.0f cycles here versus 575 there.  They are different numbers\n"
           "   because they are different experiments - see the lesson.\n", Lns*CLK_GHZ);

    const double Qfix = suppliedConcurrencyBytes(THREADS, BLOCKS_PER_SM,
                                                FIX_MLP, FIX_BYTES_PER_LANE);
    printf("\n   your fixed kernel supplies %10.0f B = %6.1f KB = %.2fx Q*\n",
           Qfix, Qfix/1024.0, Qfix/Qreq);

    // ----------------------------------------------------------- validation --
    printf("\n-- validation (second, untimed pass) ----------------------------------\n");
    int ok = 1;
    {
        double refSum = 0.0;
        for (unsigned i = 0; i < NELEM; ++i)
            refSum += (double)(1.0f + (float)((i * 2654435761u) >> 21) * (1.0f/2048.0f));
        const double u = 5.96e-8;
        const double K = (double)NELEM / (double)STRIDE;    // terms per thread
        const double gammaK = K*u/(1.0 - K*u);

        float *hp = (float*)malloc((size_t)STRIDE*sizeof(float));
        const char *nm[2] = { "baseline", "fixed   " };
        launch_t fs[2] = { lBase, lFixed };
        for (int k = 0; k < 2; ++k) {
            CHECK(cudaMemset(dpart, 0, (size_t)STRIDE*sizeof(float)));
            fs[k](); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(hp, dpart, (size_t)STRIDE*sizeof(float), cudaMemcpyDeviceToHost));
            double tot = 0.0; unsigned zero = 0;
            for (unsigned t = 0; t < STRIDE; ++t) { tot += (double)hp[t];
                                                    if (hp[t] == 0.0f) ++zero; }
            double err = fabs(tot - refSum);
            double ratio = err / (gammaK * refSum);
            printf("   %s sum %.6e vs %.6e, err/(gamma_K*S) %.3f, unwritten %u  [%s]\n",
                   nm[k], tot, refSum, ratio, zero,
                   (ratio <= 1.0 && zero == 0) ? "OK" : "FAIL");
            if (!(ratio <= 1.0 && zero == 0)) ok = 0;
        }
        free(hp);
    }

    // ---------------------------------------------------------------- score --
    printf("\n-- scoring ------------------------------------------------------------\n");
    int score = 0;
    int a1 = (fabs(Q0-Q0ref) <= 1.0);
    int a2 = (fabs(Lns-Lref) <= 0.01*Lref);
    int a3 = (fabs(Qreq-Qreqref) <= 0.01*Qreqref);
    printf("   TODO 1 supplied concurrency    [%s]\n", a1 ? " OK " : "FAIL");
    printf("   TODO 2 effective latency       [%s]\n", a2 ? " OK " : "FAIL");
    printf("   TODO 3 required concurrency    [%s]\n", a3 ? " OK " : "FAIL");
    score += (a1?2:0) + (a2?2:0) + (a3?2:0);

    int gate = (gbsFixed >= 0.95*gbsCeil);
    printf("   TODO 4 fixed kernel %.1f GB/s = %.1f%% of the measured ceiling"
           " (need 95%%)  [%s]\n", gbsFixed, 100.0*gbsFixed/gbsCeil, gate ? " OK " : "FAIL");
    if (gate) score += 2;

    int pred = (fabs(PRED_FIXED_GBS - gbsFixed) <= 0.12*gbsFixed);
    printf("   TODO 5 predicted %.1f GB/s vs measured %.1f (within 12%%)  [%s]\n",
           (double)PRED_FIXED_GBS, gbsFixed, pred ? " OK " : "FAIL");
    if (pred) score += 2;

    CHECK(cudaFree(dx)); CHECK(cudaFree(dpart));
    printf("\nSCORE: %d/10\n", score);
    printf("OVERALL: %s\n", (score == 10 && ok) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (score == 10 && ok) ? 0 : 1;
}
