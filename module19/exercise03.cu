// =============================================================================
// Module 19 / Exercise 3 — theoretical vs achieved occupancy.
//
// GOAL : Exercise 1 computed what the hardware is WILLING to place. This file
//        is about what it actually SUSTAINED, which is a different number and
//        is usually the one that matters.
//
//        `ncu` cannot run on this machine (ERR_NVGPUCTRPERM), so you are going
//        to build the counter yourself out of clock64() and %smid, and you are
//        going to discover that "achieved occupancy" has TWO defensible
//        definitions that differ by exactly the effect you are hunting.
//
//        The kernel is a grid-stride loop over `nch` chunks of work, so the
//        grid size is a free parameter: every launch below computes the same
//        answer, and the validation pass checks that. Some per-chunk costs are
//        uniform and some vary by up to 8x, which is what a ragged real
//        workload looks like.
//
//        Five TODOs: the instrument, the two denominators, a predictive model,
//        a launch-sizing rule you design, and two predictions.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN  : exercise03.exe
//
// The run takes a couple of minutes: a 2 s warm-up, an operating-point guard
// (spec 12.5b), 14 timed configurations at 5 repetitions each, and a separate
// untimed validation pass.
//
// Assumed: Module 1 (waves, tail effect, the block placement gate), Module 3
// (grid-stride loops), Module 8 (warps), Module 10 (atomics -- used here only
// as a counter), and Module 19 Exercise 1 (the four limiters).
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

#define SM_COUNT     40
#define WARPS_PER_SM 48
#define TPB          256
#define NREP         5
// Operating-point guard (spec 12.5b). The balanced two-wave reference launch
// measures 1.24-1.26 ms on a healthy machine and 3.13 ms on a contended one.
#define REF_MS_HEALTHY 1.80

__device__ __forceinline__ unsigned long long clk(void)
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}
__device__ __forceinline__ unsigned smId(void)
{
    unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r;
}

typedef struct {
    unsigned long long *lo;    // per SM: earliest warp entry  (init ~0ull)
    unsigned long long *hi;    // per SM: latest warp exit     (init 0)
    unsigned long long *cyc;   // per SM: sum of warp lifetimes(init 0)
} Prof;

// =============================================================================
// TODO 1: the instrument.
//
//   Residency is a per-WARP property, so exactly one lane of each warp should
//   report -- 32 reporters per warp would multiply every number by 32 and cost
//   32x the atomic traffic. `clk()` and `smId()` are given.
//
//   Between them, profIn and profOut must leave, for each SM:
//     p.lo[sm]  = the earliest entry timestamp of any warp on that SM
//     p.hi[sm]  = the latest exit timestamp of any warp on that SM
//     p.cyc[sm] = the sum over warps of (exit - entry)
//   The host initialises lo to ~0ull and hi/cyc to 0 before every launch.
//
//   Three things to get right. Which atomic belongs on which array. That `sm`
//   and `t0` must be captured in profIn and carried to profOut by the caller
//   (they are passed by reference and by value respectively). And that
//   clock64() counters are NOT comparable between SMs -- every quantity here
//   is per SM for that reason, and nothing in this file ever subtracts a
//   timestamp taken on one SM from one taken on another.
// =============================================================================
__device__ __forceinline__ void profIn(Prof p, unsigned long long &t0, unsigned &sm)
{
    // YOUR CODE HERE
    (void)p; (void)t0; (void)sm;
}
__device__ __forceinline__ void profOut(Prof p, unsigned long long t0, unsigned sm)
{
    // YOUR CODE HERE
    (void)p; (void)t0; (void)sm;
}

// Host-side readback of the three per-SM arrays.
typedef struct { double sumCyc, sumSpan, maxSpan; int smsUsed; } ProfData;

// =============================================================================
// TODO 2: the two denominators. Both return a FRACTION in [0,1].
//
//   The numerator is the same for both: the total warp-cycles of residency the
//   instrument counted. The denominator is where the two definitions part.
//
//   occActive  — divide by the warp-cycles the SMs COULD have held while they
//                were busy. `d->sumSpan` is the sum over SMs of (hi - lo).
//                This is what Nsight Compute calls Achieved Occupancy
//                (sm__warps_active.avg.pct_of_peak_sustained_ACTIVE).
//
//   occElapsed — divide by the warp-cycles the WHOLE MACHINE could have held
//                for the whole kernel, idle SMs included. You are given the
//                kernel's wall time in ms and the recovered SM clock in Hz;
//                `cudaDevAttrClockRate` is not the real clock (spec 12.6) and
//                is not available to you here.
//                (..._pct_of_peak_sustained_ELAPSED.)
//
//   Write them both before you look at the table. One of them is blind to the
//   effect the rest of this exercise is about.
// =============================================================================
static double occActive(const ProfData *d)
{
    if (d->sumSpan <= 0.0) return 0.0;
    // YOUR CODE HERE
    return 0.0;
}
static double occElapsed(const ProfData *d, double kernelMs, double clockHz)
{
    if (kernelMs <= 0.0 || clockHz <= 0.0) return 0.0;
    // YOUR CODE HERE
    return 0.0;
}

// =============================================================================
// TODO 3: the cheapest useful model of achieved occupancy.
//
//   Before any profiler, there is an argument that costs nothing: a grid of
//   `grid` blocks cannot keep more warp slots occupied than it has blocks to
//   put in them. Return the fraction of the THEORETICAL occupancy that the
//   grid size alone permits -- a number in [0,1] that multiplies theoretical
//   occupancy to give a bound on achieved occupancy.
//
//   `blocksPerSM` is what cudaOccupancyMaxActiveBlocksPerMultiprocessor
//   returned; `nsm` is the SM count. Module 1 gave you the vocabulary.
//
//   The harness checks that your expression is a genuine UPPER BOUND on every
//   row of the sweep (it is never exceeded) and that on the under-filled rows
//   it is tight enough to be worth having (within 40%). It will not be exact
//   above one wave, and part of this exercise is seeing by how much and asking
//   where the rest went.
// =============================================================================
static double predictAchievedFraction(int grid, int blocksPerSM, int nsm)
{
    if (grid <= 0 || blocksPerSM <= 0 || nsm <= 0) return 0.0;
    // YOUR CODE HERE
    return 0.0;
}

// =============================================================================
// TODO 4 (DESIGN): size the launch.
//
//   The shipped baseline launches a quarter of a wave and leaves most of the
//   machine empty. Replace that decision with a rule. Given the number of work
//   chunks and the machine's shape, return the grid you would actually launch,
//   and set *cannotFill to 1 if and only if no grid can fill the machine for
//   this problem -- an honest "this launch is too small" is a legitimate and
//   necessary answer, and the harness tests a problem where it is the only
//   correct one.
//
//   Your rule is evaluated by MEASUREMENT on two differently sized problems
//   with a ragged cost distribution. For each it must
//     - return a grid in [1, nch],
//     - beat the quarter-wave baseline by at least 1.25x in wall time,
//     - and reach at least 50% on the elapsed-denominator occupancy.
//
//   Two things to think through, and they pull in opposite directions. Fewer
//   blocks than one wave is obviously wrong. But more blocks than one wave is
//   what lets the work distributor refill an SM whose block finished early,
//   and with a ragged cost distribution that refilling is the only load
//   balancing you get for free -- so "exactly one wave", the answer Modules 1
//   and 3 would suggest, is not automatically the right answer here. Decide
//   what you believe before you run it.
// =============================================================================
static int chooseGrid(int nch, int blocksPerSM, int nsm, int *cannotFill)
{
    *cannotFill = 0;
    if (nch <= 0 || blocksPerSM <= 0 || nsm <= 0) return 0;
    // YOUR CODE HERE
    return 0;
}

// =============================================================================
// TODO 5 (PREDICTIONS). Commit both before you build.
//
//   P1: the harness runs one grid of exactly one wave twice -- once with a
//       uniform per-chunk cost, once with the cost varying 1..8x. Total work
//       and mean cost are identical. Which occupancy number falls?
//         1 = occ_active falls, occ_elapsed does not
//         2 = occ_elapsed falls, occ_active does not
//         3 = both fall by about the same amount
//
//   P2: your TODO 4 fix changes the launch, not the kernel. What does it do to
//       the THEORETICAL occupancy of this kernel?
//         1 = raises it   2 = lowers it   3 = leaves it unchanged
// =============================================================================
#define PRED_DENOMINATOR 0
#define PRED_THEORETICAL 0

// ------------------------------------------------------------------- payload
__host__ __device__ __forceinline__ int chunkIters(int ch, int base, int imbalanced)
{
    if (!imbalanced) return base * 4;
    unsigned h = (unsigned)ch * 2654435761u;
    h ^= h >> 13; h *= 2246822519u; h ^= h >> 16;
    return base * (1 + (int)(h & 7u));
}
__host__ __device__ __forceinline__ float workFn(float seed, int iters)
{
    float a = seed, b = seed * 0.5f + 1.0f, c = seed * 0.25f + 2.0f, d = seed * 0.125f + 3.0f;
    for (int i = 0; i < iters; ++i) {
        a = fmaf(a, 0.9999f, 0.0001f); b = fmaf(b, 0.9998f, 0.0002f);
        c = fmaf(c, 0.9997f, 0.0003f); d = fmaf(d, 0.9996f, 0.0004f);
    }
    return a + b + c + d;
}
__global__ __launch_bounds__(TPB) void shard(const float * __restrict__ in,
                                             float * __restrict__ out,
                                             int nch, int base, int imbalanced, Prof p)
{
    unsigned long long t0 = 0; unsigned sm = 0;
    profIn(p, t0, sm);
    for (int ch = blockIdx.x; ch < nch; ch += gridDim.x) {
        int j = ch * TPB + threadIdx.x;
        out[j] = workFn(in[j], chunkIters(ch, base, imbalanced));
    }
    profOut(p, t0, sm);
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
static Prof gp;
static void profReset(void)
{
    unsigned long long big = ~0ull, zero = 0ull;
    for (int i = 0; i < SM_COUNT; ++i) {
        CHECK(cudaMemcpy(gp.lo + i,  &big,  8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gp.hi + i,  &zero, 8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gp.cyc + i, &zero, 8, cudaMemcpyHostToDevice));
    }
}
static ProfData profRead(void)
{
    unsigned long long lo[SM_COUNT], hi[SM_COUNT], cy[SM_COUNT];
    CHECK(cudaMemcpy(lo, gp.lo,  8 * SM_COUNT, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hi, gp.hi,  8 * SM_COUNT, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cy, gp.cyc, 8 * SM_COUNT, cudaMemcpyDeviceToHost));
    ProfData d; d.sumCyc = d.sumSpan = d.maxSpan = 0.0; d.smsUsed = 0;
    for (int i = 0; i < SM_COUNT; ++i) {
        if (hi[i] == 0) continue;
        double sp = (double)(hi[i] - lo[i]);
        d.sumCyc += (double)cy[i]; d.sumSpan += sp;
        if (sp > d.maxSpan) d.maxSpan = sp;
        d.smsUsed++;
    }
    return d;
}
static const float *gIn; static float *gOut;
static int gNch, gBase, gImbal, gGrid;
static double runOnce(ProfData *out)
{
    double bestMs = 1e30; ProfData bestD;
    bestD.sumCyc = bestD.sumSpan = bestD.maxSpan = 0.0; bestD.smsUsed = 0;
    for (int r = 0; r < NREP; ++r) {
        profReset();
        cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        CHECK(cudaEventRecord(a));
        shard<<<gGrid, TPB>>>(gIn, gOut, gNch, gBase, gImbal, gp);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
        CHECK(cudaGetLastError());
        if (ms < bestMs) { bestMs = ms; bestD = profRead(); }
    }
    *out = bestD;
    return bestMs;
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
        CHECK(cudaEventRecord(a)); ffmaCeiling<<<blocks,thr>>>(sink, iters);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
    }
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return 2.0 * blocks * thr * 8.0 * iters / (best * 1e-3) / 1e9;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    {
        ProfData z; z.sumCyc = 1.0; z.sumSpan = 1.0; z.maxSpan = 1.0; z.smsUsed = 1;
        int cf0 = 0;
        // TODO 1 lives in device code and cannot be checked from the host; if you
        // leave it blank the calibration below simply reports 0.0%.
        if (occActive(&z) == 0.0)                   { printf("Set TODO 1 and TODO 2 first.\n"); return 0; }
        if (occElapsed(&z, 1.0, 1.0) == 0.0)        { printf("Set TODO 1 and TODO 2 first.\n"); return 0; }
        if (predictAchievedFraction(100, 6, 40) == 0.0) { printf("Set TODO 3 first.\n"); return 0; }
        if (chooseGrid(480, 6, 40, &cf0) == 0)      { printf("Set TODO 4 first.\n"); return 0; }
        if (PRED_DENOMINATOR == 0 || PRED_THEORETICAL == 0) { printf("Set TODO 5 first.\n"); return 0; }
    }

    printf("=== Module 19 / Exercise 3 - theoretical vs achieved occupancy ===\n\n");

    CHECK(cudaMalloc(&gp.lo,  8 * SM_COUNT));
    CHECK(cudaMalloc(&gp.hi,  8 * SM_COUNT));
    CHECK(cudaMalloc(&gp.cyc, 8 * SM_COUNT));

    int blkPerSM = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkPerSM, (const void*)shard, TPB, 0));
    cudaFuncAttributes at; CHECK(cudaFuncGetAttributes(&at, (const void*)shard));
    const int wave = blkPerSM * SM_COUNT;
    const double theoretical = (double)blkPerSM * (TPB / 32) / WARPS_PER_SM;
    printf("shard(): %d registers, %d B shared, %d threads -> %d blocks/SM,\n"
           "         THEORETICAL occupancy %.1f%%, one wave = %d blocks\n\n",
           at.numRegs, (int)at.sharedSizeBytes, TPB, blkPerSM, 100.0 * theoretical, wave);

    const int nchMain = 2 * wave, nchBig = 8 * wave, nchSmall = wave / 3;
    const int base = 6000;
    const size_t n = (size_t)nchBig * TPB;
    float *hin = (float*)malloc(n * 4), *hout = (float*)malloc(n * 4);
    unsigned s = 5u;
    for (size_t i = 0; i < n; ++i) { s = s * 1664525u + 1013904223u;
        hin[i] = (float)((s >> 9) & 0xFFFFu) / 65536.0f - 0.5f; }
    float *din, *dout;
    CHECK(cudaMalloc(&din, n * 4)); CHECK(cudaMalloc(&dout, n * 4));
    CHECK(cudaMemcpy(din, hin, n * 4, cudaMemcpyHostToDevice));
    gIn = din; gOut = dout;

    printf("-- warming up: 1500 ms streaming, then 500 ms compute ------------------\n");
    float *sink; float4 *stream; size_t nb = (size_t)256 * 1024 * 1024 / 16;
    CHECK(cudaMalloc(&stream, nb * 16)); CHECK(cudaMemset(stream, 1, nb * 16));
    CHECK(cudaMalloc(&sink, 4));
    warmUp(sink, stream, nb);
    double ceil0 = probeCeiling(sink);
    for (int tries = 0; tries < 5 && ceil0 < 12000.0; ++tries) {
        printf("  FFMA ceiling probe %.0f GFLOP/s is low; idling 10 s and re-warming\n", ceil0);
        std::this_thread::sleep_for(std::chrono::seconds(10));
        warmUp(sink, stream, nb); ceil0 = probeCeiling(sink);
    }
    printf("  FFMA ceiling probe: %.0f GFLOP/s%s\n", ceil0,
           (ceil0 < 12000.0) ? "  *** LOW - this operating point is power capped ***" : "");

    // ------------------------------------------------- clock + instrument check
    // Reference run: uniform cost, two full waves, the most balanced launch the
    // harness can make. Its slowest SM is busy for essentially the whole kernel,
    // which is what makes it usable as a clock reference -- and as the
    // operating-point guard (spec 12.5b). The FFMA ceiling probe above is NOT
    // sufficient on its own: it runs 15360 threads and draws little power, so it
    // reports a healthy 17000 GFLOP/s on a machine where this full-machine
    // kernel is running 2.5x slow because another process has the GPU.
    // Measured: 1.24-1.26 ms healthy, 3.13 ms contended.
    ProfData dRef; gNch = nchMain; gBase = base; gImbal = 0; gGrid = 2 * wave;
    double msRef = runOnce(&dRef);
    for (int tries = 0; tries < 5 && msRef > REF_MS_HEALTHY; ++tries) {
        printf("  reference launch took %.4f ms (healthy is under %.2f); idling 10 s\n",
               msRef, REF_MS_HEALTHY);
        std::this_thread::sleep_for(std::chrono::seconds(10));
        warmUp(sink, stream, nb);
        msRef = runOnce(&dRef);
    }
    if (msRef > REF_MS_HEALTHY)
        printf("  *** reference launch still %.4f ms: this operating point is not\n"
               "  *** this GPU's, and the gates below may fail on correct answers.\n", msRef);
    const double clockHz = dRef.maxSpan / (msRef * 1e-3);
    // Spec 12.13: sanity-check a clock64()-derived figure against a hardware
    // bound. nvidia-smi --query-gpu=clocks.max.sm reports 3105 MHz on this part.
    if (clockHz > 3.105e9 || clockHz < 0.3e9)
        printf("  *** recovered clock %.3f GHz is outside [0.30, 3.105] GHz --\n"
               "  *** the instrument is wrong, not the hardware.\n", clockHz / 1e9);

    // Calibration: one block per SM. Resident warps are then TPB/32 out of 48 by
    // construction, so occ_active has a known answer and a mis-wired instrument
    // cannot pass.
    ProfData dCal; gGrid = SM_COUNT; gNch = SM_COUNT;
    double msCal = runOnce(&dCal);
    const double calTarget = (double)(TPB / 32) / WARPS_PER_SM;
    const double calA = occActive(&dCal);
    const double calE = occElapsed(&dCal, msCal, clockHz);
    const int calOk  = (calA >= 0.75 * calTarget && calA <= 1.05 * calTarget && dCal.smsUsed == SM_COUNT);
    const int calEOk = (calE >= 0.70 * calA && calE <= 1.30 * calA);
    printf("\n-- instrument calibration: one block per SM ---------------------------\n");
    printf("  SMs reporting %d/%d, expected occ_active %.1f%%, measured %.1f%%  %s\n",
           dCal.smsUsed, SM_COUNT, 100.0 * calTarget, 100.0 * calA, calOk ? "ok" : "WRONG");
    printf("  occ_elapsed %.1f%% (should track occ_active on a single wave)  %s\n",
           100.0 * calE, calEOk ? "ok" : "WRONG");
    printf("  recovered SM clock %.3f GHz\n", clockHz / 1e9);

    // ---------------------------------------------------------------- the sweep
    const int grids[6] = { wave/4, wave/2, wave, wave + 1, (3*wave)/2, 2*wave };
    double msU[6], msI[6]; ProfData dU[6], dI[6];
    gNch = nchMain;
    for (int k = 0; k < 6; ++k) { gImbal = 0; gGrid = grids[k]; msU[k] = runOnce(&dU[k]); }
    for (int k = 0; k < 6; ++k) { gImbal = 1; gGrid = grids[k]; msI[k] = runOnce(&dI[k]); }

    printf("\n-- the sweep: %d chunks of work, six grids, two cost profiles ---------\n", nchMain);
    printf(" %7s %7s | %9s %10s %11s | %9s %10s %11s | %9s\n",
           "grid", "waves", "ms(unif)", "act(unif)", "elap(unif)",
           "ms(imb)", "act(imb)", "elap(imb)", "model");
    int modelOk = 0, modelTested = 0;
    for (int k = 0; k < 6; ++k) {
        double aU = occActive(&dU[k]), eU = occElapsed(&dU[k], msU[k], clockHz);
        double aI = occActive(&dI[k]), eI = occElapsed(&dI[k], msI[k], clockHz);
        double pred = predictAchievedFraction(grids[k], blkPerSM, SM_COUNT) * theoretical;
        // The wave fraction is an UPPER BOUND on achieved occupancy: it must
        // never be exceeded, and on an under-filled grid it must be tight
        // enough to be worth having.
        modelTested++;
        int rowOk = (aU <= 1.05 * pred) && (k >= 2 || aU >= 0.60 * pred);
        if (rowOk) modelOk++;
        printf(" %7d %7.2f | %9.4f %9.1f%% %10.1f%% | %9.4f %9.1f%% %10.1f%% | %8.1f%%\n",
               grids[k], (double)grids[k] / wave, msU[k], 100*aU, 100*eU,
               msI[k], 100*aI, 100*eI, 100*pred);
    }
    printf("\n  wave model (TODO 3) bounds occ_active correctly on %d/%d grids\n",
           modelOk, modelTested);

    // -------------------------------------------------------- TODO 4 evaluation
    printf("\n-- TODO 4: your grid choice, measured ---------------------------------\n");
    int cf = 0;
    const int gSmall = chooseGrid(nchSmall, blkPerSM, SM_COUNT, &cf);
    const int smallOk = (gSmall == nchSmall && cf == 1);
    printf("  nch = %d (less than one wave): you chose grid %d, cannotFill %d  %s\n",
           nchSmall, gSmall, cf, smallOk ? "ok" : "WRONG");

    int cf2 = 0, cf3 = 0;
    const int gMain = chooseGrid(nchMain, blkPerSM, SM_COUNT, &cf2);
    const int gBig  = chooseGrid(nchBig,  blkPerSM, SM_COUNT, &cf3);
    ProfData dMain, dBig;
    gNch = nchMain; gImbal = 1; gGrid = gMain; double msMain = runOnce(&dMain);
    gNch = nchBig;  gImbal = 1; gGrid = gBig;  double msBig  = runOnce(&dBig);
    // the shipped-bad baselines for the same two workloads
    ProfData dBadM, dBadB;
    gNch = nchMain; gImbal = 1; gGrid = wave/4; double msBadM = runOnce(&dBadM);
    gNch = nchBig;  gImbal = 1; gGrid = wave/4; double msBadB = runOnce(&dBadB);

    double eMain = occElapsed(&dMain, msMain, clockHz);
    double eBig  = occElapsed(&dBig,  msBig,  clockHz);
    int mainOk = (cf2 == 0 && gMain >= 1 && gMain <= nchMain &&
                  msBadM / msMain >= 1.25 && eMain >= 0.50);
    int bigOk  = (cf3 == 0 && gBig  >= 1 && gBig  <= nchBig  &&
                  msBadB / msBig  >= 1.25 && eBig  >= 0.50);
    printf("  nch = %d: grid %d -> %.4f ms (baseline %.4f, %.2fx), occ_elapsed %.1f%%  %s\n",
           nchMain, gMain, msMain, msBadM, msBadM / msMain, 100 * eMain, mainOk ? "ok" : "WRONG");
    printf("  nch = %d: grid %d -> %.4f ms (baseline %.4f, %.2fx), occ_elapsed %.1f%%  %s\n",
           nchBig, gBig, msBig, msBadB, msBadB / msBig, 100 * eBig, bigOk ? "ok" : "WRONG");

    // ---------------------------------------------------------- TODO 5 scoring
    // Which denominator moved when only the cost profile changed, at a fixed grid?
    const int kk = 2;                                  // exactly one wave
    double dropAct = occActive(&dU[kk]) - occActive(&dI[kk]);
    double dropEla = occElapsed(&dU[kk], msU[kk], clockHz) - occElapsed(&dI[kk], msI[kk], clockHz);
    int trueDen = (dropEla > dropAct + 0.05) ? 2 : (dropAct > dropEla + 0.05) ? 1 : 3;
    int blkAfter = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blkAfter, (const void*)shard, TPB, 0));
    int trueTheo = (blkAfter > blkPerSM) ? 1 : (blkAfter < blkPerSM) ? 2 : 3;

    printf("\n-- TODO 5 ------------------------------------------------------------\n");
    printf("  at a fixed grid of one wave, going uniform -> imbalanced moved\n"
           "  occ_active by %+.1f points and occ_elapsed by %+.1f points\n",
           -100 * dropAct, -100 * dropEla);
    printf("  P1 denominator : truth %d, you said %d  %s\n", trueDen, PRED_DENOMINATOR,
           (trueDen == PRED_DENOMINATOR) ? "correct" : "WRONG");
    printf("  P2 theoretical : blocks/SM before %d, after %d -> truth %d, you said %d  %s\n",
           blkPerSM, blkAfter, trueTheo, PRED_THEORETICAL,
           (trueTheo == PRED_THEORETICAL) ? "correct" : "WRONG");

    // ---------------------------------------------------------------- scoring
    int score = 0;
    score += calOk ? 2 : 0;
    score += calEOk ? 1 : 0;
    score += (modelOk == modelTested) ? 2 : 0;
    score += smallOk ? 1 : 0;
    score += (mainOk && bigOk) ? 2 : 0;
    score += (trueDen == PRED_DENOMINATOR) ? 1 : 0;
    score += (trueTheo == PRED_THEORETICAL) ? 1 : 0;
    printf("\n-- scoring ------------------------------------------------------------\n");
    printf("  instrument (occ_active on a known launch)   -> %d/2\n", calOk ? 2 : 0);
    printf("  occ_elapsed consistent on a single wave     -> %d/1\n", calEOk ? 1 : 0);
    printf("  wave model on the under-filled grids        -> %d/2\n", (modelOk == modelTested) ? 2 : 0);
    printf("  chooseGrid on an undersized problem         -> %d/1\n", smallOk ? 1 : 0);
    printf("  chooseGrid gates on two real problems       -> %d/2\n", (mainOk && bigOk) ? 2 : 0);
    printf("  P1 which denominator                        -> %d/1\n", (trueDen == PRED_DENOMINATOR) ? 1 : 0);
    printf("  P2 effect on theoretical occupancy          -> %d/1\n", (trueTheo == PRED_THEORETICAL) ? 1 : 0);
    printf("\n  SCORE: %d/10\n", score);

    // -------------------------------------------- validation, separate pass
    printf("\n-- validation (second, untimed pass) -----------------------------------\n");
    int numOk = 1;
    const int vg[4] = { wave/4, wave, 2*wave, gMain };
    for (int v = 0; v < 4 && numOk; ++v) {
        CHECK(cudaMemset(dout, 0, n * 4));
        profReset();
        shard<<<vg[v], TPB>>>(din, dout, nchMain, base, 1, gp);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hout, dout, n * 4, cudaMemcpyDeviceToHost));
        for (int ch = 0; ch < nchMain; ch += 83) {
            int j = ch * TPB + (ch % TPB);
            float ref = workFn(hin[j], chunkIters(ch, base, 1));
            double e = fabs((double)hout[j] - (double)ref);
            double sc = fabs((double)ref) + 1e-6;
            if (e / sc > 1e-5) { numOk = 0;
                printf("  MISMATCH grid %d chunk %d: got %.8g want %.8g\n", vg[v], ch, hout[j], ref);
                break; }
        }
    }
    printf("  every grid produces the same answer: %s\n", numOk ? "yes" : "NO");

    free(hin); free(hout);
    CHECK(cudaFree(din)); CHECK(cudaFree(dout));
    CHECK(cudaFree(sink)); CHECK(cudaFree(stream));
    CHECK(cudaFree(gp.lo)); CHECK(cudaFree(gp.hi)); CHECK(cudaFree(gp.cyc));
    printf("\nOVERALL: %s\n", (score == 10 && numOk) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (score == 10 && numOk) ? 0 : 1;
}
