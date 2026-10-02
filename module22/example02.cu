// =============================================================================
// Module 22 / Example 2 — what a gap costs: launch overhead and the sync tax.
//
// GOAL : Put a number on the two quantities a timeline makes visible but does
//        not compute for you:
//
//          L = the per-launch cost paid on the HOST, which sets a floor under
//              the iteration time no matter how short the kernel is;
//          S = the extra cost of forcing the host to wait for the device
//              every iteration instead of running ahead.
//
//        The method needs no profiler. Sweep the kernel's duration D over two
//        decades and time a loop of back-to-back launches. When D >> L the
//        loop costs D per iteration. When D << L it costs L. So the curve
//        FLATTENS, and the height of the floor is L measured directly.
//
//        Module 11 found that a fused chain beat its 2.00x traffic prediction
//        at 3.31x on 4 MB arrays and attributed the excess to launch overhead
//        of roughly 10 us. This example measures that constant head-on.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN  : example02.exe
//
// PROFILE:
//   nsys profile --trace=cuda -o ex02 --stats=true --force-overwrite=true example02.exe
//
//   Then compare this program's D column against cuda_kern_exec_sum's KAvg,
//   and this program's L against its AAvg (the cudaLaunchKernel API duration).
//   They are measuring the same two things from opposite sides.
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

#define NELEM   (1 << 14)      // 16384 floats = 64 KB, comfortably L2-resident
#define BLOCK      256
#define SWEEPS       8         // >= NCFG, so rotation is fair (spec S12.9)
#define TARGET_US 15000.0      // ~15 ms per timed segment (spec S12.12)
#define PILOT_IT   200
#define MIN_IT     200
#define MAX_IT    4000

// Eight kernel durations spanning ~1 us to ~170 us. `work` is an inner trip
// count; the kernel is deliberately compute-bound and L2-resident so that its
// duration is set by arithmetic, not by the memory system or the clock state.
static const int kWork[] = { 1, 4, 16, 64, 256, 1024, 4096, 16384 };
#define NCFG ((int)(sizeof(kWork) / sizeof(kWork[0])))

// Spec S12.11/S12.14: the accumulator must be live or the loop is deleted, and
// the inner loop must carry enough independent FFMAs that loop overhead is not
// what we are timing. Four chains, unrolled by 4.
__global__ void spin(float *a, int n, int work)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x0 = a[i], x1 = x0 + 1.0f, x2 = x0 + 2.0f, x3 = x0 + 3.0f;
    const float c = 1.0000001f;
    #pragma unroll 4
    for (int t = 0; t < work; ++t) {
        x0 = fmaf(x0, c, 1.0f); x1 = fmaf(x1, c, 1.0f);
        x2 = fmaf(x2, c, 1.0f); x3 = fmaf(x3, c, 1.0f);
    }
    float r = x0 + x1 + x2 + x3;
    if (r == -1.0f) a[i] = r;          // never true; keeps the chain live
}

// -----------------------------------------------------------------------------
// Two timing modes for the same loop.
//
//  ASYNC: enqueue ITERS launches, then synchronize once. The host runs ahead
//         and the device sees a continuous queue. Per-iteration cost = max(D, L).
//
//  SYNC : enqueue one launch, then block until it finishes, ITERS times. The
//         queue is empty at the top of every iteration, so the device waits for
//         the host's launch call before it can start. Per-iteration cost
//         = D + L + S.
// -----------------------------------------------------------------------------
enum Mode { MODE_ASYNC, MODE_SYNC };

static double timeLoop(Mode mode, float *d, int work, int iters,
                       cudaEvent_t e0, cudaEvent_t e1)
{
    int blocks = (NELEM + BLOCK - 1) / BLOCK;
    CHECK(cudaEventRecord(e0));
    if (mode == MODE_ASYNC) {
        for (int i = 0; i < iters; ++i) spin<<<blocks, BLOCK>>>(d, NELEM, work);
    } else {
        for (int i = 0; i < iters; ++i) {
            spin<<<blocks, BLOCK>>>(d, NELEM, work);
            CHECK(cudaDeviceSynchronize());
        }
    }
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return (double)ms * 1000.0 / (double)iters;      // microseconds per iteration
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Module 22 / Example 2 — launch overhead and the sync tax\n");
    printf("%d elements, %d sweeps, segment length auto-scaled to ~%.0f ms\n\n",
           NELEM, SWEEPS, TARGET_US / 1000.0);

    float *d = nullptr;
    CHECK(cudaMalloc(&d, sizeof(float) * NELEM));
    CHECK(cudaMemset(d, 0, sizeof(float) * NELEM));
    int blocks = (NELEM + BLOCK - 1) / BLOCK;

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    // Compute warm-up (spec S12.4 corollary: match the warm-up to the resource).
    // This kernel is L2-resident and FFMA-bound, so the SM clock is what has to
    // be ramped; a streaming warm-up would be the wrong one.
    {
        cudaEvent_t w0, w1;
        CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        CHECK(cudaEventRecord(w0));
        float el = 0.0f;
        do {
            for (int i = 0; i < 200; ++i) spin<<<blocks, BLOCK>>>(d, NELEM, 2048);
            CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
            CHECK(cudaEventElapsedTime(&el, w0, w1));
        } while (el < 1500.0f);
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    }

    // Spec S12.12: choose the iteration count per configuration so every timed
    // segment lasts ~15 ms. A fixed count would give the 1-work configuration a
    // 1.2 ms segment, short enough for the clock to sag between segments.
    int iters[NCFG];
    for (int c = 0; c < NCFG; ++c) {
        double us = timeLoop(MODE_ASYNC, d, kWork[c], PILOT_IT, e0, e1);
        double n  = TARGET_US / fmax(us, 0.1);
        iters[c]  = (int)fmin((double)MAX_IT, fmax((double)MIN_IT, n));
    }

    double bestAsync[NCFG], bestSync[NCFG];
    for (int c = 0; c < NCFG; ++c) { bestAsync[c] = 1e30; bestSync[c] = 1e30; }

    // Spec S12.1/S12.9: all configurations timed back-to-back in one loop, with
    // the starting configuration rotated so none of them always goes first.
    for (int s = 0; s < SWEEPS; ++s) {
        for (int q = 0; q < NCFG; ++q) {
            int c = (q + s) % NCFG;
            double a = timeLoop(MODE_ASYNC, d, kWork[c], iters[c], e0, e1);
            double b = timeLoop(MODE_SYNC,  d, kWork[c], iters[c], e0, e1);
            if (a < bestAsync[c]) bestAsync[c] = a;
            if (b < bestSync[c])  bestSync[c]  = b;
        }
    }

    // The launch floor L is the asymptote of the async curve as D -> 0. The
    // smallest configuration is the closest approach to it we can measure.
    double L = bestAsync[0];

    printf("%8s %12s %12s %12s %12s %10s\n",
           "work", "async us", "sync us", "sync tax", "D est us", "busy %");
    printf("%8s %12s %12s %12s %12s %10s\n",
           "-----", "--------", "-------", "--------", "--------", "------");
    for (int c = 0; c < NCFG; ++c) {
        // For configurations well above the floor, the async cost IS the kernel
        // duration. Below the floor the kernel duration is hidden and we can
        // only bound it. Report the estimate and flag which regime we are in.
        double Dest  = bestAsync[c];
        bool  hidden = bestAsync[c] < 1.30 * L;
        double busy  = 100.0 * Dest / bestSync[c];
        printf("%8d %12.3f %12.3f %12.3f %12s %9.1f%%\n",
               kWork[c], bestAsync[c], bestSync[c], bestSync[c] - bestAsync[c],
               hidden ? "<= floor" : "= async", busy);
    }

    printf("\nlaunch floor L (async cost of the shortest kernel) : %.3f us\n", L);
    double taxSum = 0.0;
    for (int c = 0; c < NCFG; ++c) taxSum += bestSync[c] - bestAsync[c];
    double S = taxSum / NCFG;
    printf("mean sync tax S (sync - async, across the sweep)   : %.3f us\n", S);
    printf("break-even kernel duration (D = L)                 : %.3f us\n", L);

    printf("\nHow to read this\n");
    printf("  * The async column FLATTENS at the bottom of the table. Below\n");
    printf("    about %.0f us of kernel work the loop is not executing your\n", L);
    printf("    kernel, it is executing cudaLaunchKernel. The GPU is idle for\n");
    printf("    the remainder and no kernel optimisation can recover it.\n");
    printf("  * The sync tax is roughly constant and independent of D. It is\n");
    printf("    the cost of emptying the queue -- the host cannot enqueue\n");
    printf("    iteration k+1 while it is blocked waiting for iteration k.\n");
    printf("  * The busy column is what Nsight Systems shows you as a timeline\n");
    printf("    full of gaps. Fixing it is a host-side problem: fewer, larger\n");
    printf("    launches (kernel fusion, Module 11), or removing the per-launch\n");
    printf("    cost entirely (CUDA graphs, Module 28).\n");

    // ---- validation --------------------------------------------------------
    // Three structural facts this example asserts. All are properties of the
    // shape of the data, not of any particular absolute timing, so they survive
    // the clock variation documented in spec S12.
    bool floorFlat   = bestAsync[1] < 1.40 * bestAsync[0];        // curve is flat at the bottom
    bool curveRises  = bestAsync[NCFG - 1] > 10.0 * bestAsync[0]; // and rises at the top
    bool taxPositive = S > 0.5;                                   // sync is never free
    printf("\nflat at small D (async[1] < 1.40 x async[0]) : %s\n", floorFlat   ? "yes" : "no");
    printf("rises at large D (async[last] > 10 x async[0]): %s\n", curveRises  ? "yes" : "no");
    printf("sync tax is positive                         : %s\n", taxPositive ? "yes" : "no");

    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d));
    CHECK(cudaDeviceReset());

    bool ok = floorFlat && curveRises && taxPositive;
    printf("\nOVERALL: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
