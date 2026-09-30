// =====================================================================
// Module 2 / Exercise 2 SOLUTION : "The error is not where it is reported"
//
// This is a DEBUGGING exercise. The program below is broken. This header
// states the SYMPTOM only. Diagnose it yourself before opening the
// solution notes.
//
// WHAT IT IS SUPPOSED TO DO
//   out[i] = clamp(a * in[i] + b, lo, hi) for n elements, validated
//   against a CPU reference.
//
// SYMPTOM (what you actually get)
//   1. It compiles warning-clean.
//   2. The error check immediately after the kernel launch reports
//      SUCCESS.
//   3. The first call that reports anything is the cudaMemcpy that
//      brings the results back:
//
//        CUDA error cudaErrorIllegalAddress ...
//          -> an illegal memory access was encountered
//
//      ...but that memcpy's arguments are plainly correct: right
//      pointers, right byte count, right direction, and the host buffer
//      is big enough.
//   4. Every CUDA call after it reports the SAME error -- including
//      cudaFree, which cannot possibly be dereferencing anything.
//   5. Delete the kernel launch and all the errors vanish (the program
//      then merely prints FAIL).
//
// YOUR JOB
//   TODO 1 (solved) -- a synchronizing check was added after the launch.
//   TODO 2 (solved) -- the wrapper is now given an ELEMENT COUNT, not a
//                      byte count. See exercise02_solution.md.
//
// Tools worth reaching for:
//   nvcc -arch=sm_89 -O3 -lineinfo -o exercise02_solution.exe exercise02_solution.cu
//   compute-sanitizer --tool memcheck .\exercise02_solution.exe
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
// RUN:    .\exercise02_solution.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// This CHECK reports and CONTINUES rather than exiting, so that you can
// watch what happens to the calls that come after a failure. Do not
// "fix" that: the cascade it exposes is evidence.
static int g_errors = 0;
#define CHECK(x) do {                                                      \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) {                                               \
        ++g_errors;                                                        \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                  \
                cudaGetErrorName(e_), __FILE__, __LINE__,                  \
                cudaGetErrorString(e_));                                   \
    }                                                                      \
} while (0)

__host__ __device__ __forceinline__ float clamp_f(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void saxpy_clamp(const float* in, float* out, int n,
                            float a, float b, float lo, float hi)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = clamp_f(fmaf(a, in[i], b), lo, hi);
}

// ---------------------------------------------------------------------
// Convenience wrapper. The parameter is now unambiguously an ELEMENT
// COUNT, which is what both the grid size and the kernel bound need.
// ---------------------------------------------------------------------
static void launch_saxpy_clamp(const float* d_in, float* d_out, int nElems,
                               float a, float b, float lo, float hi)
{
    const int threads = 256;
    const int blocks  = (nElems + threads - 1) / threads;

    saxpy_clamp<<<blocks, threads>>>(d_in, d_out, nElems, a, b, lo, hi);

    CHECK(cudaGetLastError());        // TODO 1: launch-configuration errors
    CHECK(cudaDeviceSynchronize());   // TODO 1: errors raised during execution
}

int main(void)
{
    const int    n     = 1000003;
    const size_t bytes = (size_t)n * sizeof(float);
    const float  a = 1.75f, b = -0.5f, lo = -2.0f, hi = 2.0f;

    CHECK(cudaSetDevice(0));

    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    if (!h_in || !h_out || !h_ref) { fprintf(stderr, "host alloc failed\n"); return 1; }

    for (int i = 0; i < n; ++i)
        h_in[i] = -2.0f + 4.0f * ((float)(i % 4096) / 4095.0f);

    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc((void**)&d_in,  bytes));
    CHECK(cudaMalloc((void**)&d_out, bytes));

    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    launch_saxpy_clamp(d_in, d_out, n, a, b, lo, hi);   // TODO 2: n, not bytes

    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i)
        h_ref[i] = clamp_f(fmaf(a, h_in[i], b), lo, hi);

    int bad = 0;
    for (int i = 0; i < n; ++i) {
        double d   = fabs((double)h_out[i] - (double)h_ref[i]);
        double tol = 1e-5 * fmax(1.0, fabs((double)h_ref[i]));
        if (d > tol) ++bad;
    }
    printf("%s (%d mismatching elements of %d, %d CUDA errors reported)\n",
           (bad == 0 && g_errors == 0) ? "PASS" : "FAIL", bad, n, g_errors);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);
    CHECK(cudaDeviceReset());
    return (bad == 0 && g_errors == 0) ? 0 : 1;
}
