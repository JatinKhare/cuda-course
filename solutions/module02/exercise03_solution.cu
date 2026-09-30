// =====================================================================
// Module 2 / Exercise 3 SOLUTION : "Predict the behavior of an asynchronous launch"
//
// This is a PREDICT-THE-BEHAVIOR exercise. Write your three predictions
// into the TODOs below BEFORE you compile. Then run it once and see how
// you did. Changing a prediction after running defeats the exercise.
//
// GOAL
//   Commit to an answer about (a) the interleaving of host and device
//   output around an asynchronous launch, (b) how expensive the launch
//   statement itself is, and (c) whether a result is safe to read back
//   without an explicit synchronization.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
// RUN:    .\exercise03_solution.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

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

// =====================================================================
// TODO 1: Phase A prints four kinds of line:
//
//     H1  "[host] before launch"
//     H2  "[host] after the launch statement"
//     D   eight "[device] ..." lines, one per thread
//     H3  "[host] after cudaDeviceSynchronize"
//
//   The kernel deliberately spins for several milliseconds, far longer
//   than the host needs to reach H2.
//
//   Which ordering do you expect on stdout?
//     1 : H1, H2, D..., H3
//     2 : H1, D..., H2, H3
//     3 : H1, H2, H3, D...
//     4 : varies from run to run; no fixed answer
//
//   Set the value. Then, separately, predict whether the eight D lines
//   will appear in increasing global-thread-id order, and why.
// =====================================================================
static int predicted_phaseA_order = 1;   // TODO 1 (solved)

// =====================================================================
// TODO 2: Phase B times the launch STATEMENT itself on the host clock
//   (not the kernel), for a kernel that takes milliseconds of GPU time,
//   launched back-to-back into an already-busy queue.
//
//   Which bucket does the per-launch host cost fall into?
//     1 : under 1 us
//     2 : roughly 1-20 us
//     3 : roughly 100-1000 us
//     4 : about the same as the kernel's own duration
// =====================================================================
static int predicted_launch_cost_bucket = 2;  // TODO 2 (solved)

// =====================================================================
// TODO 3: Phase C launches the compute kernel and then immediately calls
//
//     cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
//
//   with NO cudaDeviceSynchronize() and no other synchronization of any
//   kind in between.
//
//   Does h_out receive the kernel's results, or does it receive whatever
//   was in d_out before the kernel ran?
//     true  : h_out is correct
//     false : h_out is stale / garbage; this is a race
//
//   Justify your answer from the semantics of the call, not from what
//   feels safe.
// =====================================================================
static bool predicted_memcpy_sees_results = true;   // TODO 3 (solved)

// ---------------------------------------------------------------------
__global__ void announce_and_spin(float* sink, int spinIters)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("  [device] global id %d (block %d, thread %d)\n",
           gid, blockIdx.x, threadIdx.x);

    float acc = 1.0f + 1e-7f * (float)gid;
    for (int k = 0; k < spinIters; ++k)
        acc = fmaf(acc, 1.0000001f, 1e-7f);
    sink[gid] = acc;
}

// Same spin, no printf -- used for the timing phases.
__global__ void spin_only(float* sink, int spinIters)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 1.0f + 1e-7f * (float)gid;
    for (int k = 0; k < spinIters; ++k)
        acc = fmaf(acc, 1.0000001f, 1e-7f);
    sink[gid] = acc;
}

__host__ __device__ __forceinline__ float transform(float x)
{
    return fmaf(x, 3.0f, -1.0f);
}

__global__ void compute(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = transform(in[i]);
}

int main(void)
{
    if (predicted_phaseA_order < 1 || predicted_phaseA_order > 4 ||
        predicted_launch_cost_bucket < 1 || predicted_launch_cost_bucket > 4) {
        printf("Fill in TODO 1 and TODO 2 (and TODO 3) with your predictions first.\n");
        return 0;
    }

    CHECK(cudaSetDevice(0));
    CHECK(cudaFree(0));                 // create the context up front

    // =================================================================
    // Phase A -- ordering
    // =================================================================
    float* d_sink = nullptr;
    CHECK(cudaMalloc((void**)&d_sink, 40 * 32 * sizeof(float)));

    printf("--- Phase A ---\n");
    printf("  [host] before launch\n");
    fflush(stdout);

    announce_and_spin<<<2, 4>>>(d_sink, 4000000);
    CHECK(cudaGetLastError());

    printf("  [host] after the launch statement\n");
    fflush(stdout);

    CHECK(cudaDeviceSynchronize());

    printf("  [host] after cudaDeviceSynchronize\n");
    printf("  (TODO 1: compare the ordering above with your prediction, %d)\n\n",
           predicted_phaseA_order);
    fflush(stdout);

    // =================================================================
    // Phase B -- cost of the launch statement
    // =================================================================
    const int n = 1 << 20;
    const size_t bytes = (size_t)n * sizeof(float);
    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc((void**)&d_in,  bytes));
    CHECK(cudaMalloc((void**)&d_out, bytes));

    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    if (!h_in || !h_out || !h_ref) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (int i = 0; i < n; ++i)
        h_in[i] = -1.0f + 2.0f * ((float)(i % 2048) / 2047.0f);
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    const int spin    = 200000;

    spin_only<<<1, 1>>>(d_sink, 1);              // warm-up (module load)
    CHECK_KERNEL();

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));

    const int reps = 20;
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r)
        compute<<<blocks, threads>>>(d_in, d_out, n);   // trivially cheap
    auto t1 = std::chrono::high_resolution_clock::now();
    CHECK(cudaEventRecord(e1));
    CHECK_KERNEL();

    double hostUsPerLaunch =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;

    // And a long kernel, so you can see the host is not waiting for it.
    CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r)
        spin_only<<<40, 32>>>(d_sink, spin);
    auto t3 = std::chrono::high_resolution_clock::now();
    double hostUsPerLongLaunch =
        std::chrono::duration<double, std::micro>(t3 - t2).count() / reps;
    CHECK(cudaEventRecord(e0));
    CHECK(cudaEventSynchronize(e0));
    CHECK_KERNEL();

    // GPU time of the long kernel, measured properly with events.
    CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r)
        spin_only<<<40, 32>>>(d_sink, spin);
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CHECK(cudaGetLastError());
    double gpuUsPerLongKernel = (double)ms * 1000.0 / reps;

    int actual_bucket = hostUsPerLongLaunch < 1.0    ? 1
                      : hostUsPerLongLaunch < 20.0   ? 2
                      : hostUsPerLongLaunch < 1000.0 ? 3 : 4;
    if (hostUsPerLongLaunch > 0.5 * gpuUsPerLongKernel) actual_bucket = 4;

    printf("--- Phase B ---\n");
    printf("  host cost/launch, short kernel : %8.2f us\n", hostUsPerLaunch);
    printf("  host cost/launch, long kernel  : %8.2f us\n", hostUsPerLongLaunch);
    printf("  GPU duration of the long kernel: %8.2f us\n", gpuUsPerLongKernel);
    printf("  bucket observed = %d, you predicted %d -> %s\n\n",
           actual_bucket, predicted_launch_cost_bucket,
           actual_bucket == predicted_launch_cost_bucket ? "MATCH" : "MISMATCH");

    // =================================================================
    // Phase C -- memcpy with no explicit synchronization
    // =================================================================
    CHECK(cudaMemset(d_out, 0, bytes));
    CHECK(cudaDeviceSynchronize());

    compute<<<blocks, threads>>>(d_in, d_out, n);
    CHECK(cudaGetLastError());
    // NO cudaDeviceSynchronize() here. On purpose.
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i)
        h_ref[i] = transform(h_in[i]);

    int bad = 0;
    for (int i = 0; i < n; ++i) {
        double d   = fabs((double)h_out[i] - (double)h_ref[i]);
        double tol = 1e-5 * fmax(1.0, fabs((double)h_ref[i]));
        if (d > tol) ++bad;
    }
    bool actual_memcpy_sees_results = (bad == 0);

    printf("--- Phase C ---\n");
    printf("  mismatching elements after an unsynchronized D2H copy : %d / %d\n",
           bad, n);
    printf("  h_out correct = %s, you predicted %s -> %s\n\n",
           actual_memcpy_sees_results ? "true" : "false",
           predicted_memcpy_sees_results ? "true" : "false",
           actual_memcpy_sees_results == predicted_memcpy_sees_results
               ? "MATCH" : "MISMATCH");

    int matched = (actual_bucket == predicted_launch_cost_bucket)
                + (actual_memcpy_sees_results == predicted_memcpy_sees_results);
    printf("%s -- %d of the 2 machine-checkable predictions matched.\n",
           matched == 2 ? "PASS" : "FAIL", matched);
    printf("(TODO 1 is checked by eye against the Phase A output above.)\n");

    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_sink));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);
    CHECK(cudaDeviceReset());
    return matched == 2 ? 0 : 1;
}
