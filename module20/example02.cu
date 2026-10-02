// =============================================================================
// Module 20 / Example 2 — Little's Law at the memory system.
//
// GOAL : Do the one calculation that explains the design of the whole machine,
//        then measure it.
//
//   A  The arithmetic.  432 GB/s of DRAM with a ~575-cycle dependent-load
//      latency (Module 4) requires a specific number of bytes to be in flight
//      at all times.  Compute it.  Then compute what 1536 resident threads per
//      SM can actually hold outstanding.  The two numbers are the reason the
//      SM has 48 warp slots and the reason "just add warps" usually works.
//
//   B  The MLP x occupancy table.  A streaming read kernel with C independent
//      hoisted float4 loads per thread, at 5 occupancies.  The table does not
//      have two independent axes: every cell with the same PRODUCT
//      (warps/scheduler x C) lands on the same bandwidth.  That is Little's
//      Law printed as a 2-D array.
//
//   C  How many loads can ONE warp have outstanding?  A single warp, C
//      independent L2-resident loads per step, steps serialised by a data
//      dependence.  cycles/step = latency + C * (marginal cost).  The fit
//      gives both the latency and the point past which another outstanding
//      load stops buying anything.
//
// Module 11 measured the 1-block-vs-8-block version of B for saxpy and got
// 2.1x / 1.00x.  This file generalises that one contrast into the surface it
// is a slice of.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN  : example02.exe
// SASS : nvcc -arch=sm_89 -O3 -cubin -o example02.cubin example02.cu
//        cuobjdump -sass example02.cubin
//        (count the LDG.E.128 between the two IADD3s of the loop: there must
//         be C of them back to back, not C separated by their consumers)
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

#define SM_COUNT        40
#define THREADS_PER_SM  1536
#define PIN_PEAK_GBS    432.0       // 192-bit GDDR6 @ 9.001 GHz, a BOUND
#define DRAM_LAT_CYC    575.0       // Module 4, dependent-load latency

// n4 is divisible by C*stride for every (C, blocksPerSM) pair used below:
//   stride = blocksPerSM * 40 * 128  (max 12*5120 = 61440), C max 32
//   lcm requirement 32*61440 = 1,966,080 ; 1,966,080 * 17 = 33,423,360 float4
#define N4   (1966080u * 17u)       // 33,423,360 float4 = 534.8 MB, > 11x L2

// -----------------------------------------------------------------------------
// Streaming read with C independent loads in flight per thread.
// The C loads are issued back to back BEFORE any of them is consumed; that is
// the whole point and it is the one thing you must check in the SASS.
// -----------------------------------------------------------------------------
template<int C>
__global__ void readMLP(const float4 *__restrict__ x, float *out, unsigned n4)
{
    const unsigned stride = gridDim.x * blockDim.x;
    const unsigned base   = blockIdx.x * blockDim.x + threadIdx.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

    for (unsigned i = base; i + (C-1)*stride < n4; i += C*stride) {
        float4 v[C];
        #pragma unroll
        for (int c = 0; c < C; ++c) v[c] = x[i + c*stride];     // issue all C
        #pragma unroll
        for (int c = 0; c < C; ++c) {                            // then consume
            acc.x += v[c].x; acc.y += v[c].y; acc.z += v[c].z; acc.w += v[c].w;
        }
    }
    out[base] = acc.x + acc.y + acc.z + acc.w;
}

// -----------------------------------------------------------------------------
// One warp, C independent L2-resident loads per step.  pos depends on the
// previous step's data, so steps cannot overlap and the measurement is
// (latency of one step) rather than (steady-state throughput).
// 32 MB working set: far bigger than the 128 KB L1, comfortably inside the
// 48 MB L2, so every load is an L2 hit and DRAM bandwidth never enters.
// -----------------------------------------------------------------------------
template<int C>
__global__ void warpMLP(const float *__restrict__ x, float *o, long long *cyc,
                        int steps, unsigned mask)
{
    unsigned pos = 12345u;
    float acc = 0.f;
    long long t0 = clock64();
    for (int s = 0; s < steps; ++s) {
        float v[C];
        #pragma unroll
        for (int c = 0; c < C; ++c)
            v[c] = x[(((pos + (unsigned)c * 2654435761u) & mask) & ~31u) + threadIdx.x];
        float t = 0.f;
        #pragma unroll
        for (int c = 0; c < C; ++c) t += v[c];
        acc += t;
        pos = 1103515245u * pos + 12345u + (unsigned)(t == 1e30f);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) cyc[0] = t1 - t0;
    o[threadIdx.x] = acc;
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
__global__ void fillPattern(float *x, size_t n)
{
    for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n;
         i += gridDim.x*(size_t)blockDim.x)
        x[i] = (float)((i % 7u) + 1u);
}

// ------------------------------------------------------------------ harness --
static const float4 *g_x; static float *g_out;
static int g_blocks = 1;
typedef void (*launch_t)(void);
template<int C> static void launchMLP(void)
{ readMLP<C><<<g_blocks*SM_COUNT, 128>>>(g_x, g_out, N4); }

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
#define NC 6
#define NCFG (NW*NC)

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 20 / Example 2 - Little's Law at the memory system ===\n\n");

    const int wps[NW] = { 1, 2, 4, 8, 12 };
    const int mlp[NC] = { 1, 2, 4, 8, 16, 32 };

    const size_t bytes = (size_t)N4 * 16;
    float4 *dx; float *dout, *dsmall; long long *dcyc;
    CHECK(cudaMalloc(&dx, bytes));
    CHECK(cudaMalloc(&dout, (size_t)12*SM_COUNT*128*sizeof(float)));
    CHECK(cudaMalloc(&dsmall, 32*sizeof(float)));
    CHECK(cudaMalloc(&dcyc, sizeof(long long)));
    fillPattern<<<1024,256>>>((float*)dx, (size_t)N4*4);
    CHECK(cudaDeviceSynchronize());
    g_x = dx; g_out = dout;

    printf("-- warming up: 1500 ms streaming, then 500 ms compute ----------------\n");
    {
        cudaEvent_t w0, w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        float el = 0; CHECK(cudaEventRecord(w0));
        while (el < 1500.0f) { warmStream<<<320,256>>>(dx, dout, N4);
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            CHECK(cudaEventElapsedTime(&el, w0, w1)); }
        el = 0; CHECK(cudaEventRecord(w0));
        while (el < 500.0f) { warmFfma<<<480,128>>>(dout, 2000);
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            CHECK(cudaEventElapsedTime(&el, w0, w1)); }
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    }
    CHECK(cudaGetLastError());

    // ======================================================================= B
    //  (measured first; section A quotes the measured ceiling)
    double best[NCFG];
    for (int i = 0; i < NCFG; ++i) best[i] = 1e30;
    {
        launch_t byC[NC] = { launchMLP<1>, launchMLP<2>, launchMLP<4>,
                             launchMLP<8>, launchMLP<16>, launchMLP<32> };
        for (int s = 0; s < NCFG; ++s)
            for (int q = 0; q < NCFG; ++q) {
                int p = (q + s) % NCFG;
                g_blocks = wps[p / NC];
                double t = timeOne(byC[p % NC], 3);
                if (t < best[p]) best[p] = t;
            }
    }
    CHECK(cudaGetLastError());

    double gbs[NCFG], ceiling = 0.0;
    for (int i = 0; i < NCFG; ++i) {
        gbs[i] = (double)bytes / (best[i]*1e-3) / 1e9;
        if (gbs[i] > ceiling) ceiling = gbs[i];
    }

    // ======================================================================= A
    printf("\n-- A. how much has to be in flight ------------------------------------\n");
    {
        // Recover a clock from the FFMA pipe: Module 16 showed clock64() lies at
        // full occupancy and cudaDevAttrClockRate lies always.  Use ~1.9 GHz,
        // the value Example 1 recovers at low occupancy, as the stated figure.
        const double clkGHz   = 1.90;
        const double latNs    = DRAM_LAT_CYC / clkGHz;
        const double inFlight = ceiling * 1e9 * latNs * 1e-9;   // bytes
        printf("   measured streaming read ceiling      %10.1f GB/s  (%.1f%% of %.0f pin peak)\n",
               ceiling, 100.0*ceiling/PIN_PEAK_GBS, PIN_PEAK_GBS);
        printf("   Module 4 dependent-load latency      %10.1f cycles = %.0f ns at %.2f GHz\n",
               DRAM_LAT_CYC, latNs, clkGHz);
        printf("   Little's Law  bytes in flight        %10.0f B  = %.0f KB\n",
               inFlight, inFlight/1024.0);
        printf("                 32 B sectors in flight %10.0f\n", inFlight/32.0);
        printf("                 per SM                 %10.1f sectors\n",
               inFlight/32.0/SM_COUNT);
        printf("                 128 B warp-loads/SM    %10.1f\n",
               inFlight/128.0/SM_COUNT);
        printf("\n   Now the supply side.  One SM holds %d threads = %d warps.\n",
               THREADS_PER_SM, THREADS_PER_SM/32);
        printf("   If every resident warp has exactly ONE coalesced 128 B load\n");
        printf("   outstanding, that is %d warp-loads = %d sectors per SM.\n",
               THREADS_PER_SM/32, (THREADS_PER_SM/32)*4);
        printf("   Demand %.1f, supply %d.  Ratio %.2f.\n",
               inFlight/128.0/SM_COUNT, THREADS_PER_SM/32,
               (double)(THREADS_PER_SM/32) / (inFlight/128.0/SM_COUNT));
        printf("\n   Read that ratio as the design statement it is: a full SM with\n"
               "   ONE outstanding load per warp is within a small factor of what\n"
               "   the DRAM needs.  That is why 48 warp slots exist, and it is why\n"
               "   a kernel at 25%% occupancy with no ILP cannot saturate the bus.\n"
               "   Per thread the demand is only %.1f bytes - LESS than one float -\n"
               "   so the quantity that matters is never 'loads per thread' alone.\n",
               inFlight/(SM_COUNT*(double)THREADS_PER_SM));
    }

    // -------------------------------------------------------------- print B --
    printf("\n-- B. MLP x occupancy: GB/s ------------------------------------------\n");
    printf("   buffer %.1f MB = %.1fx L2; every cell reads all of it exactly once\n\n",
           bytes/1048576.0, bytes/48.0/1048576.0);
    printf("   %-8s", "warps/s");
    for (int c = 0; c < NC; ++c) printf("%8s%-2d", "C=", mlp[c]);
    printf("\n");
    for (int w = 0; w < NW; ++w) {
        printf("   %-8d", wps[w]);
        for (int c = 0; c < NC; ++c) printf("%10.1f", gbs[w*NC+c]);
        printf("\n");
    }
    printf("\n   the same cells, indexed by the PRODUCT  (warps/sched x C)\n");
    printf("   %10s %10s %10s %10s\n", "product", "cells", "mean GB/s", "spread");
    {
        for (int prod = 1; prod <= 384; prod *= 2) {
            double sum = 0, lo = 1e30, hi = 0; int n = 0;
            for (int w = 0; w < NW; ++w)
                for (int c = 0; c < NC; ++c)
                    if (wps[w]*mlp[c] == prod) {
                        double v = gbs[w*NC+c]; sum += v; ++n;
                        if (v < lo) lo = v; if (v > hi) hi = v;
                    }
            if (n == 0) continue;
            printf("   %10d %10d %10.1f %9.1f%%\n", prod, n, sum/n,
                   n > 1 ? 100.0*(hi-lo)/ (sum/n) : 0.0);
        }
    }
    printf("\n   Cells with the same product agree far better than cells in the same\n"
           "   row or the same column.  Occupancy and MLP are not two knobs; they\n"
           "   are two ways of turning one knob.\n");
    {
        int kneeProd = -1;
        for (int prod = 1; prod <= 384 && kneeProd < 0; prod *= 2)
            for (int w = 0; w < NW; ++w)
                for (int c = 0; c < NC; ++c)
                    if (wps[w]*mlp[c] == prod && gbs[w*NC+c] >= 0.97*ceiling) {
                        kneeProd = prod; break;
                    }
        printf("\n   first product reaching 97%% of the ceiling: %d\n", kneeProd);
        printf("   = %d outstanding 128 B warp-loads per scheduler\n", kneeProd);
        printf("   = %d per SM = %d B in flight per SM = %.0f KB device-wide\n",
               kneeProd*4, kneeProd*4*128, kneeProd*4*128*SM_COUNT/1024.0);
        printf("   Compare with section A's demand figure.  The measured number is\n"
               "   SMALLER, and the reason is not that the latency is shorter: it is\n"
               "   that C counts loads in the SOURCE.  Nothing stops the compiler\n"
               "   from issuing the next loop iteration's loads before this one's\n"
               "   adds retire, so the real in-flight count is larger than C and\n"
               "   this knee is a LOWER BOUND on concurrency.  Exercise 3 pins a\n"
               "   loop to genuinely one load in flight with `#pragma unroll 1` and\n"
               "   recovers 569 cycles, i.e. Module 4's 575 almost exactly.\n");
    }

    // ======================================================================= C
    printf("\n-- C. how many loads can ONE warp have outstanding? ------------------\n");
    printf("   single warp, 32 MB L2-resident working set, steps serialised\n\n");
    printf("   %4s %12s %12s %12s %12s\n", "C", "cycles", "cyc/step", "cyc/load", "marginal");
    {
        const unsigned n32 = (32u*1024u*1024u)/4u;
        float *dx32; CHECK(cudaMalloc(&dx32, (size_t)n32*4));
        CHECK(cudaMemset(dx32, 0, (size_t)n32*4));
        const unsigned mask = n32 - 1u;
        const int STEPS = 1000;
        const int cs[9] = { 1, 2, 3, 4, 6, 8, 12, 16, 32 };
        double cps[9];
        for (int i = 0; i < 9; ++i) {
            long long bestc = (1LL<<62);
            for (int r = 0; r < 4; ++r) {
                switch (cs[i]) {
                case 1:  warpMLP<1 ><<<1,32>>>(dx32,dsmall,dcyc,STEPS,mask); break;
                case 2:  warpMLP<2 ><<<1,32>>>(dx32,dsmall,dcyc,STEPS,mask); break;
                case 3:  warpMLP<3 ><<<1,32>>>(dx32,dsmall,dcyc,STEPS,mask); break;
                case 4:  warpMLP<4 ><<<1,32>>>(dx32,dsmall,dcyc,STEPS,mask); break;
                case 6:  warpMLP<6 ><<<1,32>>>(dx32,dsmall,dcyc,STEPS,mask); break;
                case 8:  warpMLP<8 ><<<1,32>>>(dx32,dsmall,dcyc,STEPS,mask); break;
                case 12: warpMLP<12><<<1,32>>>(dx32,dsmall,dcyc,STEPS,mask); break;
                case 16: warpMLP<16><<<1,32>>>(dx32,dsmall,dcyc,STEPS,mask); break;
                default: warpMLP<32><<<1,32>>>(dx32,dsmall,dcyc,STEPS,mask); break;
                }
                CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
                long long h; CHECK(cudaMemcpy(&h, dcyc, 8, cudaMemcpyDeviceToHost));
                if (h < bestc) bestc = h;
            }
            cps[i] = (double)bestc / STEPS;
            printf("   %4d %12lld %12.1f %12.1f", cs[i], bestc, cps[i], cps[i]/cs[i]);
            if (i) printf("%12.1f\n", (cps[i]-cps[i-1])/(cs[i]-cs[i-1]));
            else   printf("%12s\n", "-");
        }
        // least-squares fit over C >= 4
        double sx=0, sy=0, sxx=0, sxy=0; int n=0;
        for (int i = 3; i < 9; ++i) { double x=cs[i], y=cps[i];
            sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; ++n; }
        double slope = (n*sxy - sx*sy) / (n*sxx - sx*sx);
        double icept = (sy - slope*sx) / n;
        printf("\n   fit over C >= 4 :  cycles/step = %.1f + %.2f * C\n", icept, slope);
        printf("   intercept = the latency the warp cannot avoid paying once (L2)\n");
        printf("   slope     = what each ADDITIONAL outstanding load costs in issue\n"
                "               and transport.  More loads keep helping only while\n"
                "               slope * C is small next to the intercept; past\n"
                "               C ~ %.0f the kernel is no longer latency-bound at\n"
                "               all and extra MLP buys nothing.\n", icept/slope);
        CHECK(cudaFree(dx32));
    }

    // ------------------------------------------------- validation, 2nd pass --
    printf("\n-- validation (second, untimed pass) ----------------------------------\n");
    int ok = 1;
    {
        g_blocks = 4;
        const unsigned stride = (unsigned)g_blocks*SM_COUNT*128;
        float *h = (float*)malloc((size_t)stride*sizeof(float));
        launch_t byC[NC] = { launchMLP<1>, launchMLP<2>, launchMLP<4>,
                             launchMLP<8>, launchMLP<16>, launchMLP<32> };
        for (int c = 0; c < NC; ++c) {
            CHECK(cudaMemset(dout, 0, (size_t)stride*sizeof(float)));
            byC[c]();
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h, dout, (size_t)stride*sizeof(float), cudaMemcpyDeviceToHost));
            // every thread touches exactly N4/stride float4 = 4*N4/stride floats,
            // whose values are (index % 7) + 1 with index = 4*(float4 index) + lane
            double worst = 0.0;
            for (unsigned t = 0; t < stride; t += 97) {
                double ref = 0.0;
                for (unsigned i = t; i + (mlp[c]-1)*stride < N4; i += mlp[c]*stride)
                    for (int k = 0; k < mlp[c]; ++k) {
                        unsigned f4 = i + k*stride;
                        for (int l = 0; l < 4; ++l) ref += (double)(((4u*f4 + l) % 7u) + 1u);
                    }
                double e = fabs((double)h[t] - ref) / fmax(1.0, ref);
                if (e > worst) worst = e;
            }
            int zero = 0;
            for (unsigned t = 0; t < stride; ++t) if (h[t] == 0.0f) zero++;
            printf("   C=%-3d worst relative error %.3e, unwritten outputs %u  [%s]\n",
                   mlp[c], worst, zero, (worst <= 1e-5 && zero == 0) ? "OK" : "FAIL");
            if (!(worst <= 1e-5 && zero == 0)) ok = 0;
        }
        free(h);
    }

    CHECK(cudaFree(dx)); CHECK(cudaFree(dout));
    CHECK(cudaFree(dsmall)); CHECK(cudaFree(dcyc));
    printf("\nOVERALL: %s\n", ok ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return ok ? 0 : 1;
}
