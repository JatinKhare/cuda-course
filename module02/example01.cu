// =====================================================================
// Module 2 / Example 1 : "The launch is asynchronous"
//
// GOAL
//   Demonstrate, with measurements rather than assertions:
//     (1) what a __global__ function is and how it is launched,
//     (2) that the launch statement RETURNS TO THE HOST before the
//         kernel has run -- and how few microseconds that takes,
//     (3) that device-side printf output is buffered and only appears
//         at a synchronization point,
//     (4) that cudaGetLastError() catches launch-CONFIGURATION errors
//         at the launch site, and that such an error is NON-STICKY:
//         the context survives and later work still succeeds.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN:    .\example01.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// The two macros this course uses everywhere.
//
// CHECK(expr)        wraps a CUDA runtime call that RETURNS a cudaError_t.
// CHECK_KERNEL()     is used immediately after a <<<>>> launch. It does
//                    TWO things, and both are necessary:
//                      cudaGetLastError()      -> launch-config errors
//                      cudaDeviceSynchronize() -> execution errors
// ---------------------------------------------------------------------
#define CHECK(x) do {                                                      \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                  \
                cudaGetErrorName(e_), __FILE__, __LINE__,                  \
                cudaGetErrorString(e_));                                   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

#define CHECK_KERNEL() do {                                                \
    CHECK(cudaGetLastError());                                             \
    CHECK(cudaDeviceSynchronize());                                        \
} while (0)

// ---------------------------------------------------------------------
// __global__ : compiled for the DEVICE, callable from the HOST, launched
// with <<<>>>. Must return void -- there is no value for the launch
// statement to hand back, because the launch returns before the function
// has executed.
// ---------------------------------------------------------------------
__global__ void hello(void)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("    [device] block %d thread %d (global id %d)\n",
           blockIdx.x, threadIdx.x, gid);
}

// ---------------------------------------------------------------------
// A kernel with a tunable amount of work, so the host has something to
// race against. The dependent FMA chain cannot be optimized away because
// the result is stored.
// ---------------------------------------------------------------------
__global__ void busy(float* out, int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 1.0f + 1e-7f * (float)i;
    for (int k = 0; k < iters; ++k)
        acc = fmaf(acc, 1.0000001f, 1e-7f);
    out[i] = acc;
}

int main(void)
{
    CHECK(cudaSetDevice(0));
    CHECK(cudaFree(0));   // force context creation now, so the first real
                          // launch below is not charged for it

    // =================================================================
    // Part 1 -- device printf and ordering
    // =================================================================
    printf("Part 1: device printf\n");
    printf("  [host] BEFORE launch\n");
    fflush(stdout);

    hello<<<2, 4>>>();
    CHECK(cudaGetLastError());          // launch-config check only; no sync yet

    printf("  [host] AFTER launch statement, BEFORE cudaDeviceSynchronize\n");
    fflush(stdout);

    CHECK(cudaDeviceSynchronize());     // device printf FIFO is flushed here

    printf("  [host] AFTER cudaDeviceSynchronize\n\n");
    fflush(stdout);

    // =================================================================
    // Part 2 -- how long does the launch statement itself take?
    // =================================================================
    printf("Part 2: launch overhead vs kernel duration\n");

    const int n = 1 << 20;
    float* d_out = nullptr;
    CHECK(cudaMalloc(&d_out, (size_t)n * sizeof(float)));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    // Warm up: the very first launch of a kernel pays for module load.
    busy<<<blocks, threads>>>(d_out, n, 1);
    CHECK_KERNEL();

    cudaEvent_t evStart, evStop;
    CHECK(cudaEventCreate(&evStart));
    CHECK(cudaEventCreate(&evStop));

    const int kIters = 20000;   // enough to make the kernel take ~ms

    // (a) Host-side wall clock around the LAUNCH STATEMENT ONLY. This is
    // the one legitimate use of a CPU clock near an async launch: we are
    // timing the CPU-side call, not the GPU-side work.
    //
    // Each launch here is preceded by a full sync, so it is the first
    // launch after an idle queue -- on Windows (WDDM) that forces the
    // driver to build and submit a fresh command buffer, which is the
    // expensive case.
    double launchUs = 0.0;
    const int kReps = 20;
    for (int r = 0; r < kReps; ++r) {
        CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        busy<<<blocks, threads>>>(d_out, n, kIters);
        auto t1 = std::chrono::high_resolution_clock::now();
        launchUs += std::chrono::duration<double, std::micro>(t1 - t0).count();
        CHECK_KERNEL();
    }
    launchUs /= kReps;

    // (b) Back-to-back launches into an already-busy queue. The GPU is
    // still chewing on launch r-1 while the host enqueues launch r, so
    // this measures the pure enqueue cost.
    CHECK(cudaDeviceSynchronize());
    auto b0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < kReps; ++r)
        busy<<<blocks, threads>>>(d_out, n, kIters);
    auto b1 = std::chrono::high_resolution_clock::now();
    double enqueueUs =
        std::chrono::duration<double, std::micro>(b1 - b0).count() / kReps;
    CHECK_KERNEL();

    // GPU-side duration of the same kernel, measured with events.
    CHECK(cudaEventRecord(evStart));
    for (int r = 0; r < kReps; ++r)
        busy<<<blocks, threads>>>(d_out, n, kIters);
    CHECK(cudaEventRecord(evStop));
    CHECK(cudaEventSynchronize(evStop));
    float totalMs = 0.0f;
    CHECK(cudaEventElapsedTime(&totalMs, evStart, evStop));
    CHECK(cudaGetLastError());
    double kernelMs = totalMs / kReps;

    printf("  launch after an idle queue      : %8.2f us (CPU time)\n", launchUs);
    printf("  launch into a busy queue        : %8.2f us (CPU time)\n", enqueueUs);
    printf("  kernel actually ran for         : %8.2f us (GPU time)\n", kernelMs * 1000.0);
    printf("  work / enqueue ratio            : %8.1fx\n\n", (kernelMs * 1000.0) / enqueueUs);

    // =================================================================
    // Part 3 -- a launch-configuration error is caught at the launch,
    //           and it is NON-STICKY.
    // =================================================================
    printf("Part 3: an illegal launch configuration\n");

    const int tooMany = 1025;     // hardware maximum on sm_89 is 1024
    busy<<<1, tooMany>>>(d_out, n, 1);

    cudaError_t launchErr = cudaGetLastError();   // reads AND clears
    printf("  cudaGetLastError() after <<<1,%d>>> : %s (%s)\n",
           tooMany, cudaGetErrorName(launchErr), cudaGetErrorString(launchErr));
    printf("  cudaGetLastError() a second time    : %s  <- cleared by the first read\n",
           cudaGetErrorName(cudaGetLastError()));

    // The context is still healthy: this launch succeeds.
    busy<<<blocks, threads>>>(d_out, n, 1);
    CHECK_KERNEL();
    printf("  a legal launch after the failed one : OK (error was NON-STICKY)\n\n");

    // =================================================================
    // Part 4 -- host and device pointers are not interchangeable
    // =================================================================
    float probe = -1.0f;
    CHECK(cudaMemcpy(&probe, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Part 4: d_out is a DEVICE pointer\n");
    printf("  d_out               = %p  (an address in the GPU address space)\n", (void*)d_out);
    printf("  d_out[0] via memcpy = %.6f\n", probe);
    printf("  *d_out on the host  = segmentation fault -- never do it.\n");

    CHECK(cudaEventDestroy(evStart));
    CHECK(cudaEventDestroy(evStop));
    CHECK(cudaFree(d_out));
    CHECK(cudaDeviceReset());
    return 0;
}
