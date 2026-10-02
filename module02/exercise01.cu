// =====================================================================
// Module 2 / Exercise 1 : "One definition, two compilers"
//
// GOAL
//   Evaluate a degree-5 polynomial elementwise on the GPU using Horner's
//   rule, validate against a CPU reference that calls THE SAME source
//   function, and get the execution-space qualifiers right.
//
//     p(x) = 2x^5 - 3x^4 + 0.5x^3 + x^2 - 4x + 7
//
//   Horner form:
//     p(x) = ((((2x - 3)x + 0.5)x + 1)x - 4)x + 7
//
//   There are three TODOs. One of them has an obvious answer that
//   compiles for the GPU and then breaks the build for the CPU. Read the
//   whole file before you start typing.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN:    .\exercise01.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
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
    CHECK(cudaGetLastError());        /* launch-configuration errors */    \
    CHECK(cudaDeviceSynchronize());   /* errors raised during execution */ \
} while (0)

// ---------------------------------------------------------------------
// TODO 1: Give horner5 the execution-space qualifier(s) it needs.
//
//   Before you choose, find EVERY call site of horner5 in this file and
//   note which side of the host/device line each one is on. The qualifier
//   set must make all of them legal in a single nvcc invocation.
//
//   Replace the placeholder below. Do not change the body.
// ---------------------------------------------------------------------
__host__ __device__ __forceinline__ float horner5(float x)   // YOUR CODE HERE (TODO 1)
{
    float p = 2.0f;
    p = p * x - 3.0f;
    p = p * x + 0.5f;
    p = p * x + 1.0f;
    p = p * x - 4.0f;
    p = p * x + 7.0f;
    return p;
}

// ---------------------------------------------------------------------
// TODO 2: Write the kernel body.
//
//   Requirements:
//     * one element per thread, using the 1-D global index
//       blockIdx.x * blockDim.x + threadIdx.x  (Module 3 owns indexing in
//       depth; simple 1-D is all you need here),
//     * out[i] must receive horner5(x[i]),
//     * n is NOT a multiple of the block size, so some threads in the
//       final block have no element. Those threads must not touch either
//       array -- not the store, and not the load. A guard that only
//       protects the store is still an out-of-range read, and
//       compute-sanitizer will say so.
// ---------------------------------------------------------------------
__global__ void eval_poly(const float* x, float* out, int n)
{
    // YOUR CODE HERE (TODO 2)
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if(id<=1000003)
        out[id] = horner5(x[id]);
}

int main(void)
{
    const int n = 1000003;                       // prime-ish, on purpose
    const size_t bytes = (size_t)n * sizeof(float);

    CHECK(cudaSetDevice(0));

    float* h_x   = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    if (!h_x || !h_out || !h_ref) { fprintf(stderr, "host alloc failed\n"); return 1; }

    // Deterministic, index-derived input in [-1.5, 1.5].
    for (int i = 0; i < n; ++i)
        h_x[i] = -1.5f + 3.0f * ((float)(i % 8192) / 8191.0f);

    float *d_x = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc((void**)&d_x,   bytes));
    CHECK(cudaMalloc((void**)&d_out, bytes));
    CHECK(cudaMemset(d_out, 0, bytes));          // so an empty kernel gives FAIL,
                                                 // not garbage
    CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;

    // -----------------------------------------------------------------
    // TODO 3: Compute the number of blocks.
    //
    //   Every one of the n elements must be covered by exactly one
    //   thread, and you may not launch a block that has no work at all.
    // -----------------------------------------------------------------
    int blocks = n/threads + 1;          // YOUR CODE HERE (TODO 3)

    if (blocks <= 0) { printf("Set TODO 3 (blocks) first.\n"); return 0; }

    printf("n = %d, threads/block = %d, blocks = %d  -> %lld threads launched\n",
           n, threads, blocks, (long long)blocks * threads);

    eval_poly<<<blocks, threads>>>(d_x, d_out, n);
    CHECK_KERNEL();

    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // ---- CPU reference: same function, host compilation --------------
    for (int i = 0; i < n; ++i)
        h_ref[i] = horner5(h_x[i]);

    int bad = 0; double worst = 0.0; int worstIdx = -1;
    for (int i = 0; i < n; ++i) {
        double d   = fabs((double)h_out[i] - (double)h_ref[i]);
        double tol = 1e-5 * fmax(1.0, fabs((double)h_ref[i]));
        if (d > worst) { worst = d; worstIdx = i; }
        if (d > tol) ++bad;
    }

    printf("worst |gpu - cpu| = %.3e at i = %d (gpu %.7f, cpu %.7f)\n",
           worst, worstIdx,
           worstIdx >= 0 ? h_out[worstIdx] : 0.0f,
           worstIdx >= 0 ? h_ref[worstIdx] : 0.0f);
    printf("%s (%d mismatching elements of %d)\n",
           bad == 0 ? "PASS" : "FAIL", bad, n);

    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_out));
    free(h_x); free(h_out); free(h_ref);
    CHECK(cudaDeviceReset());
    return bad == 0 ? 0 : 1;
}
