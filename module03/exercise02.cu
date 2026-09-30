// =====================================================================
// Module 3 / Exercise 2 : "Port a triple-nested loop"  (CPU -> GPU)
//
// GOAL
//   You are given a correct C++ loop nest over a batched 2D tensor and
//   nothing else. You design the entire parallel decomposition:
//     - what exactly one thread computes,
//     - the block shape,
//     - the grid shape,
//     - the index flattening that turns (batch, row, col) into an offset.
//
//   The tensor is `in[B][R][C]` stored contiguously in row-major order:
//   the element (b, r, c) lives at offset ((b * R) + r) * C + c. C is the
//   fastest-varying axis, then R, then B. B = 37, R = 1013, C = 577 --
//   no dimension is a multiple of 32 and none is a power of two.
//
//   The computation (see cpuReference below for the authority):
//
//       out[b][r][c] = 0.50 * in[b][r][c]
//                    + 0.25 * in[b][r][(c+1) % C]
//                    + 0.25 * in[(b+1) % B][r][c]
//
//   REQUIREMENT: your launch must be genuinely three-dimensional --
//   the harness rejects any configuration with gridDim.z * blockDim.z
//   equal to 1. Decide which problem axis deserves z and why.
//
//   The harness also reports achieved bandwidth. Two decompositions can
//   both be correct and differ by several times in speed; which axis you
//   hand to threadIdx.x decides that.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
// RUN:    .\exercise02.exe
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

static const int    B = 37;      // batches
static const int    R = 1013;    // rows per batch
static const int    C = 577;     // columns per row  (fastest axis)
static const double PEAK_BW = 432.0;

// ---------------------------------------------------------------------
// The authority. Do not change this function; your kernel must reproduce
// it exactly (to fp32 tolerance).
// ---------------------------------------------------------------------
static void cpuReference(const float* in, float* out, int b_, int r_, int c_)
{
    for (int b = 0; b < b_; ++b) {
        for (int r = 0; r < r_; ++r) {
            for (int c = 0; c < c_; ++c) {
                int cNext = (c + 1) % c_;
                int bNext = (b + 1) % b_;
                out[((size_t)b * r_ + r) * c_ + c] =
                      0.50f * in[((size_t)b     * r_ + r) * c_ + c]
                    + 0.25f * in[((size_t)b     * r_ + r) * c_ + cNext]
                    + 0.25f * in[((size_t)bNext * r_ + r) * c_ + c];
            }
        }
    }
}

// ---------------------------------------------------------------------
// TODO 1: Write the kernel.
//
//   The signature below is a suggestion, not a constraint -- change it if
//   your decomposition needs something else, and update the launch to
//   match. What is fixed is the contract: after the launch, every one of
//   B*R*C output elements must hold the value cpuReference would write.
//
//   Things worth deciding on purpose rather than by habit:
//     - which of the three problem axes threadIdx.x should walk, given
//       that lanes 0..31 of a warp differ only in the linearized thread
//       index and x varies fastest;
//     - whether any thread should handle more than one element;
//     - where the bounds guard goes and which axes it must cover.
// ---------------------------------------------------------------------
__global__ void batched_blend(const float* __restrict__ in,
                              float* __restrict__ out,
                              int b_, int r_, int c_)
{
    (void)in; (void)out; (void)b_; (void)r_; (void)c_;
    // YOUR CODE HERE (TODO 1)
}

// ---------------------------------------------------------------------
static int compare(const float* got, const float* ref, size_t n)
{
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < n; ++i) {
        float g = got[i], e = ref[i];
        if (!(fabsf(g - e) <= 1e-5f * fmaxf(1.0f, fabsf(e)))) {
            if (bad == 0) first = i;
            ++bad;
        }
    }
    if (bad == 0) { printf("  VALIDATION        PASS  (%zu elements)\n", n); return 0; }
    size_t c = first % C, r = (first / C) % R, b = first / ((size_t)R * C);
    printf("  VALIDATION        FAIL  %zu / %zu wrong; first at (b=%zu, r=%zu, c=%zu): got %.6f, want %.6f\n",
           bad, n, b, r, c, got[first], ref[first]);
    return 1;
}

int main(void)
{
    CHECK(cudaSetDevice(0));

    const size_t n     = (size_t)B * R * C;
    const size_t bytes = n * sizeof(float);

    printf("=== Module 3 / Exercise 2 : batched blend, B=%d R=%d C=%d (%zu elements, %.1f MB) ===\n",
           B, R, C, n, bytes / 1.0e6);

    float* h_in  = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_in || !h_ref || !h_out) { printf("host alloc failed\n"); return 1; }

    // Deterministic, index-derived, and distinct along all three axes so
    // that a wrong flattening cannot accidentally agree.
    for (int b = 0; b < B; ++b)
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                h_in[((size_t)b * R + r) * C + c] =
                      0.031f * (float)b
                    + 0.007f * (float)(r % 251)
                    + 0.013f * (float)(c % 199)
                    + 0.0001f * (float)((b * 7919 + r * 104729 + c * 15485863) % 1000);

    cpuReference(h_in, h_ref, B, R, C);

    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0, bytes));

    // -----------------------------------------------------------------
    // TODO 2: the block shape. Remember: blockDim.x * blockDim.y *
    //         blockDim.z <= 1024, and the block is chopped into warps in
    //         linearized order, x fastest.
    // -----------------------------------------------------------------
    dim3 block(0, 0, 0);   // YOUR CODE HERE (TODO 2)

    // -----------------------------------------------------------------
    // TODO 3: the grid shape. It must cover every element of the tensor.
    //         Note the hardware limits: gridDim.x <= 2^31-1, but
    //         gridDim.y and gridDim.z <= 65535.
    // -----------------------------------------------------------------
    dim3 grid(0, 0, 0);    // YOUR CODE HERE (TODO 3)

    if (block.x == 0 || block.y == 0 || block.z == 0 ||
        grid.x  == 0 || grid.y  == 0 || grid.z  == 0) {
        printf("\nSet TODO 2 (block) and TODO 3 (grid) first; a dim3 component is still 0.\n");
        CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
        free(h_in); free(h_ref); free(h_out);
        CHECK(cudaDeviceReset());
        return 0;
    }

    unsigned tpb = block.x * block.y * block.z;
    long long launched = (long long)grid.x * grid.y * grid.z * tpb;
    printf("\nlaunch <<< (%u,%u,%u), (%u,%u,%u) >>>  = %u threads/block, %lld threads total\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z, tpb, launched);
    printf("  threads per element = %.3f\n", (double)launched / (double)n);

    int fails = 0;
    if (tpb > 1024) { printf("  [REQUIREMENT FAIL] block has %u threads, max is 1024\n", tpb); ++fails; }
    if (grid.z * block.z <= 1) {
        printf("  [REQUIREMENT FAIL] the launch must use the z axis (gridDim.z * blockDim.z > 1)\n");
        ++fails;
    }
    if (grid.y > 65535u || grid.z > 65535u) {
        printf("  [REQUIREMENT FAIL] gridDim.y/z exceed the 65535 hardware limit\n"); ++fails;
    }

    batched_blend<<<grid, block>>>(d_in, d_out, B, R, C);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    fails += compare(h_out, h_ref, n);

    // Timing: warm-up + 20 iterations.
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    batched_blend<<<grid, block>>>(d_in, d_out, B, R, C);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(t0));
    for (int it = 0; it < 20; ++it)
        batched_blend<<<grid, block>>>(d_in, d_out, B, R, C);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    CHECK(cudaGetLastError());
    float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, t0, t1)); ms /= 20.f;
    CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));

    // Compulsory traffic: the input array must cross the bus at least
    // once and the output once -> 8 B per element. The kernel issues
    // three loads per element; how much of the extra two the caches
    // absorb is exactly what the number below measures.
    double gbs = 8.0 * (double)n / (ms * 1.0e-3) / 1.0e9;
    printf("  TIME              %8.4f ms   %7.1f GB/s effective   %5.1f%% of peak\n",
           ms, gbs, 100.0 * gbs / PEAK_BW);

    printf("\nOVERALL: %s\n", fails == 0 ? "PASS" : "FAIL");

    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    free(h_in); free(h_ref); free(h_out);
    CHECK(cudaDeviceReset());
    return fails == 0 ? 0 : 1;
}
