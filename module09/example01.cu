// =====================================================================
// Module 9 / Example 1 : What __syncthreads() actually guarantees
//
// GOAL
//   Separate the two guarantees a block barrier makes, and show that
//   each one is load-bearing on its own:
//
//     G1 (execution barrier) no thread of the block advances past the
//        barrier until every non-exited thread of the block reaches it.
//     G2 (memory fence, block scope) every shared- and global-memory
//        write issued by a thread of the block before the barrier is
//        visible to every thread of the block after the barrier.
//
//   Part A  : a shared-memory rotate that needs both. Run it with no
//             barrier, with `volatile` instead of a barrier, and with
//             the barrier.
//   Part B  : __syncthreads_count() driving a block-wide convergence
//             loop -- a barrier inside a loop whose trip count is
//             block-uniform *by construction*.
//   Part C  : __syncwarp() -- a warp-scope barrier, and an honest look
//             at a kernel that passes without one and is still wrong.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN:    .\example01.exe
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
    CHECK(cudaGetLastError());                                             \
    CHECK(cudaDeviceSynchronize());                                        \
} while (0)

static const int TPB   = 256;   // threads per block  (8 warps)
static const int SHIFT = 96;    // deliberately crosses warp boundaries
static const int NBLK  = 4096;

// ---------------------------------------------------------------------
// PART A -- one dataflow, three synchronization stories
// ---------------------------------------------------------------------

// A1: no synchronization at all. Thread t reads s[(t+96)&255], a slot
//     written by a thread in a *different warp*. Nothing makes that
//     warp have run yet (G1 missing) and nothing makes its store
//     visible if it has (G2 missing).
__global__ void rotate_nobarrier(const float* __restrict__ in,
                                 float* __restrict__ out)
{
    __shared__ float s[TPB];
    const int t    = threadIdx.x;
    const int base = blockIdx.x * TPB;
    s[t] = in[base + t];
    out[base + t] = s[(t + SHIFT) & (TPB - 1)];
}

// A2: `volatile` instead of a barrier. volatile forbids the compiler
//     from keeping s[t] in a register and from reordering the volatile
//     accesses with respect to each other. It does NOT make warp 0 wait
//     for warp 3, and it does not publish anything. It fixes neither
//     G1 nor G2.
__global__ void rotate_volatile(const float* __restrict__ in,
                                float* __restrict__ out)
{
    volatile __shared__ float s[TPB];
    const int t    = threadIdx.x;
    const int base = blockIdx.x * TPB;
    s[t] = in[base + t];
    out[base + t] = s[(t + SHIFT) & (TPB - 1)];
}

// A3: the barrier. Block-uniform control flow, so it is well defined.
__global__ void rotate_barrier(const float* __restrict__ in,
                               float* __restrict__ out)
{
    __shared__ float s[TPB];
    const int t    = threadIdx.x;
    const int base = blockIdx.x * TPB;
    s[t] = in[base + t];
    __syncthreads();                       // G1 + G2
    out[base + t] = s[(t + SHIFT) & (TPB - 1)];
}

// ---------------------------------------------------------------------
// PART B -- __syncthreads_count(): barrier + block-wide predicate sum
// ---------------------------------------------------------------------
// Each thread owns one cell. A cell "settles" when it reaches its
// target. Every sweep an unsettled cell takes one step and also reads
// its left neighbour, so the sweep genuinely needs G1+G2. The loop must
// run until *no* cell in the block changed. The exit test is therefore
// a block-wide reduction -- and __syncthreads_count() gives us the
// barrier and the reduction in one instruction, returning the same
// value to every thread, which is exactly what makes the loop condition
// block-uniform and hence the in-loop barrier well defined.
__global__ void settle_count(const int* __restrict__ target,
                             int* __restrict__ sweeps_out,
                             int* __restrict__ val_out)
{
    __shared__ int s[TPB];
    const int t    = threadIdx.x;
    const int base = blockIdx.x * TPB;

    s[t] = 0;
    __syncthreads();

    const int tgt = target[base + t];
    int sweeps = 0;
    int active;
    do {
        const int left    = s[(t + TPB - 1) & (TPB - 1)];    // needs G1+G2
        const int changed = (s[t] < tgt) ? 1 : 0;
        __syncthreads();                    // all reads done before writes
        if (changed) s[t] += 1 + (left & 0); // (left & 0) keeps the read live
        active = __syncthreads_count(changed);   // barrier + block-wide sum
        ++sweeps;
    } while (active > 0);                   // block-uniform: `active` is
                                            // identical in every thread

    val_out[base + t] = s[t];
    if (t == 0) sweeps_out[blockIdx.x] = sweeps;
}

// ---------------------------------------------------------------------
// PART C -- __syncwarp(): warp-scope barrier
// ---------------------------------------------------------------------
// Only 32 threads are involved and they are all in one warp. Pre-Volta
// this needed no synchronization at all: a warp had one program counter,
// so lane 5's store had provably retired before lane 4's load issued.
// Independent thread scheduling (Module 8) removed that guarantee: lanes
// carry their own PCs and may be scheduled as separate sub-groups. The
// warp-scope barrier is __syncwarp(mask).
__global__ void warp_rotate_nosync(const float* __restrict__ in,
                                   float* __restrict__ out)
{
    __shared__ float s[32];
    const int t = threadIdx.x;
    s[t] = in[t];
    /* nothing here -- legacy "warp-synchronous" reasoning */
    out[t] = s[(t + 1) & 31];
}

__global__ void warp_rotate_syncwarp(const float* __restrict__ in,
                                     float* __restrict__ out)
{
    __shared__ float s[32];
    const int t = threadIdx.x;
    s[t] = in[t];
    __syncwarp();                           // mask defaults to 0xffffffff
    out[t] = s[(t + 1) & 31];
}

// ---------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------
static int count_mismatch(const float* got, const float* ref, int n)
{
    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (fabsf(got[i] - ref[i]) > 1e-5f * fmaxf(1.0f, fabsf(ref[i]))) ++bad;
    return bad;
}

typedef void (*rot_k)(const float*, float*);

int main(void)
{
    const int N = NBLK * TPB;

    float *h_in  = (float*)malloc((size_t)N * sizeof(float));
    float *h_out = (float*)malloc((size_t)N * sizeof(float));
    float *h_ref = (float*)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)((unsigned)(i * 1664525u + 1013904223u) % 977u) * 0.001f;
    for (int b = 0; b < NBLK; ++b)
        for (int t = 0; t < TPB; ++t)
            h_ref[b * TPB + t] = h_in[b * TPB + ((t + SHIFT) & (TPB - 1))];

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)N * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    printf("=== PART A: shared-memory rotate, %d blocks x %d threads, shift %d ===\n",
           NBLK, TPB, SHIFT);
    printf("%-28s %14s  %s\n", "variant", "wrong elems", "guarantees supplied");

    const char* aname[3] = { "no barrier", "volatile __shared__", "__syncthreads()" };
    const char* aguar[3] = { "none", "none", "G1 + G2" };
    rot_k       akern[3] = { rotate_nobarrier, rotate_volatile, rotate_barrier };
    for (int v = 0; v < 3; ++v) {
        CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
        akern[v]<<<NBLK, TPB>>>(d_in, d_out);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
        printf("%-28s %14d  %s\n", aname[v], count_mismatch(h_out, h_ref, N), aguar[v]);
    }
    printf("\n  `volatile` changes what the COMPILER may do. It changes nothing\n"
           "  about what the SCHEDULER may do. It is not synchronization.\n");

    // ---------------- PART B ----------------
    printf("\n=== PART B: __syncthreads_count() convergence loop ===\n");
    int *h_tgt = (int*)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; ++i) h_tgt[i] = (int)((unsigned)(i * 2654435761u) % 37u);
    int *d_tgt, *d_sw, *d_val;
    CHECK(cudaMalloc(&d_tgt, (size_t)N * sizeof(int)));
    CHECK(cudaMalloc(&d_val, (size_t)N * sizeof(int)));
    CHECK(cudaMalloc(&d_sw,  (size_t)NBLK * sizeof(int)));
    CHECK(cudaMemcpy(d_tgt, h_tgt, (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    settle_count<<<NBLK, TPB>>>(d_tgt, d_sw, d_val);
    CHECK_KERNEL();
    int *h_sw  = (int*)malloc((size_t)NBLK * sizeof(int));
    int *h_val = (int*)malloc((size_t)N * sizeof(int));
    CHECK(cudaMemcpy(h_sw,  d_sw,  (size_t)NBLK * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_val, d_val, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
    int bad = 0, mx = 0, mn = 1 << 30;
    for (int i = 0; i < N; ++i) if (h_val[i] != h_tgt[i]) ++bad;
    for (int b = 0; b < NBLK; ++b) {
        if (h_sw[b] > mx) mx = h_sw[b];
        if (h_sw[b] < mn) mn = h_sw[b];
    }
    printf("  settled values wrong: %d\n", bad);
    printf("  sweeps per block: min %d, max %d  (= 1 + max target in the block)\n", mn, mx);
    printf("  The loop condition is `active > 0`. Every thread gets the SAME\n"
           "  `active`, because __syncthreads_count() returns the block-wide sum\n"
           "  to all of them. That is what makes the in-loop barrier legal.\n");

    // ---------------- PART C ----------------
    printf("\n=== PART C: one warp, shared-memory rotate by 1 ===\n");
    float h_w[32], h_wr[32];
    for (int i = 0; i < 32; ++i) h_w[i] = 1.0f + (float)i;
    for (int i = 0; i < 32; ++i) h_wr[i] = h_w[(i + 1) & 31];
    float *d_w, *d_wo;
    CHECK(cudaMalloc(&d_w,  32 * sizeof(float)));
    CHECK(cudaMalloc(&d_wo, 32 * sizeof(float)));
    CHECK(cudaMemcpy(d_w, h_w, 32 * sizeof(float), cudaMemcpyHostToDevice));
    const char* cn[2] = { "no __syncwarp   (UNDEFINED)", "__syncwarp()    (correct)" };
    for (int v = 0; v < 2; ++v) {
        CHECK(cudaMemset(d_wo, 0, 32 * sizeof(float)));
        if (v == 0) warp_rotate_nosync  <<<1, 32>>>(d_w, d_wo);
        else        warp_rotate_syncwarp<<<1, 32>>>(d_w, d_wo);
        CHECK_KERNEL();
        float got[32];
        CHECK(cudaMemcpy(got, d_wo, 32 * sizeof(float), cudaMemcpyDeviceToHost));
        printf("  %-30s wrong elems: %d\n", cn[v], count_mismatch(got, h_wr, 32));
    }
    printf("  If the first line printed 0, that is a measurement, not a proof.\n"
           "  See Exercise 1 on why `passed` and `correct` are different claims.\n");

    free(h_in); free(h_out); free(h_ref); free(h_tgt); free(h_sw); free(h_val);
    CHECK(cudaFree(d_in));  CHECK(cudaFree(d_out)); CHECK(cudaFree(d_tgt));
    CHECK(cudaFree(d_val)); CHECK(cudaFree(d_sw));
    CHECK(cudaFree(d_w));   CHECK(cudaFree(d_wo));
    CHECK(cudaDeviceReset());
    return 0;
}
