// =====================================================================
// Module 4 / Example 2 : "Where the bytes actually come from"
//
// GOAL
//   Two measurements that change how you benchmark for the rest of the
//   course.
//
//   Part A -- L2 is 48 MB. If your benchmark buffers fit in it, you are
//             measuring L2, not DRAM, and you will report a bandwidth
//             ABOVE the 432 GB/s the DRAM interface can physically
//             deliver. Sweep the working-set size and watch the number
//             fall off a cliff exactly where the working set stops
//             fitting in L2.
//
//   Part B -- Host memory. A pageable host buffer cannot be the target
//             of a DMA engine, because the OS may move or evict its
//             physical pages at any moment. The driver therefore copies
//             it into a small pinned staging ("bounce") buffer first.
//             Pinned memory (cudaMallocHost) skips that copy.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN:    .\example02.exe
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

// ---------------------------------------------------------------------
// A pure streaming copy. float4 (16 B per thread) keeps each warp's
// request at 512 B = 4 full 128 B cache lines, so we are not limited by
// the number of memory instructions in flight. Grid-stride loop so the
// grid size is independent of n.
// ---------------------------------------------------------------------
__global__ void stream_copy(const float4* __restrict__ in,
                            float4* __restrict__ out, size_t n4)
{
    size_t i      = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) out[i] = in[i];
}

int main(void)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    int l2 = 0;
    CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));

    const double PEAK_GBS = 432.0;
    printf("=== %s : L2 = %.0f MB, DRAM peak = %.0f GB/s ===\n",
           p.name, l2 / (1024.0 * 1024.0), PEAK_GBS);

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    // -----------------------------------------------------------------
    // Clock warm-up. A laptop GPU idles at a low SM/memory clock; the
    // first ~100 ms of work is measured at the wrong frequency and the
    // first few sizes of the sweep come out nonsensically slow. Burn
    // some work first. Do this in every benchmark you ever write.
    // -----------------------------------------------------------------
    {
        size_t wb = 64u << 20;
        float4 *a = nullptr, *b = nullptr;
        CHECK(cudaMalloc(&a, wb)); CHECK(cudaMalloc(&b, wb));
        CHECK(cudaMemset(a, 1, wb));
        for (int i = 0; i < 300; ++i)
            stream_copy<<<p.multiProcessorCount * 16, 256>>>(a, b, wb / sizeof(float4));
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(a)); CHECK(cudaFree(b));
    }

    // =================================================================
    // Part A -- working-set sweep
    // =================================================================
    // The copy touches TWO buffers, so the working set is 2 * bytes.
    // The L2 boundary is crossed when 2*bytes exceeds 48 MB, i.e. at
    // bytes ~= 24 MB.
    const size_t mb[] = { 1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512 };
    const int    nsz  = (int)(sizeof(mb) / sizeof(mb[0]));
    const int    ITERS = 50;

    printf("\n--- Part A: streaming copy, buffer size sweep ---\n");
    printf("%8s %12s %10s %10s %8s  %s\n",
           "MB each", "workset MB", "ms", "GB/s", "%peak", "verdict");

    int pass = 1;
    double bw_small = 0.0, bw_large = 0.0;

    for (int k = 0; k < nsz; ++k) {
        size_t bytes = mb[k] << 20;
        size_t n4    = bytes / sizeof(float4);

        float4 *d_in = nullptr, *d_out = nullptr;
        CHECK(cudaMalloc(&d_in,  bytes));
        CHECK(cudaMalloc(&d_out, bytes));
        CHECK(cudaMemset(d_in, 0x3f, bytes));
        CHECK(cudaMemset(d_out, 0, bytes));

        // Fixed grid: 16 blocks per SM of 256 threads. Enough warps to
        // saturate the memory system without depending on n.
        int threads = 256;
        int blocks  = p.multiProcessorCount * 16;

        stream_copy<<<blocks, threads>>>(d_in, d_out, n4);   // warm-up
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < ITERS; ++i)
            stream_copy<<<blocks, threads>>>(d_in, d_out, n4);
        CHECK(cudaEventRecord(ev1));
        CHECK(cudaEventSynchronize(ev1));
        CHECK(cudaGetLastError());

        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        ms /= ITERS;

        double gb  = 2.0 * (double)bytes / 1e9;     // one read + one write
        double bws = gb / (ms / 1e3);
        double ws  = 2.0 * mb[k];

        const char* verdict = (bws > PEAK_GBS) ? "IMPOSSIBLE from DRAM -> L2 hit"
                            : (ws <= 48.0)     ? "partly L2-resident"
                                               : "real DRAM";
        printf("%8zu %12.0f %10.4f %10.1f %7.1f%%  %s\n",
               mb[k], ws, ms, bws, 100.0 * bws / PEAK_GBS, verdict);

        if (mb[k] == 8)   bw_small = bws;
        if (mb[k] == 256) bw_large = bws;

        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
    }

    // Validation: the small (L2-resident) case must beat the large
    // (DRAM) case by a wide margin, and the large case must be below
    // the physical DRAM peak.
    if (!(bw_small > 1.5 * bw_large)) {
        printf("  [FAIL] L2-resident case did not clearly beat the DRAM case\n");
        pass = 0;
    }
    if (!(bw_large < PEAK_GBS)) {
        printf("  [FAIL] DRAM case exceeded the physical peak -- impossible\n");
        pass = 0;
    }

    // =================================================================
    // Part B -- pageable vs pinned host memory
    // =================================================================
    const size_t HB = 128u << 20;            // 128 MB
    const int    HITER = 20;

    void *h_pageable = malloc(HB);
    void *h_pinned   = nullptr;
    void *d_buf      = nullptr;
    CHECK(cudaMallocHost(&h_pinned, HB));    // page-locked, DMA-able
    CHECK(cudaMalloc(&d_buf, HB));

    // Deterministic fill so both copies move identical bytes.
    unsigned char* q = (unsigned char*)h_pageable;
    unsigned char* r = (unsigned char*)h_pinned;
    for (size_t i = 0; i < HB; ++i) { q[i] = (unsigned char)(i & 0xff); r[i] = q[i]; }

    float ms = 0.0f;
    double bw_pageable, bw_pinned;

    for (int i = 0; i < 3; ++i)                                        // warm-up
        CHECK(cudaMemcpy(d_buf, h_pageable, HB, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < HITER; ++i)
        CHECK(cudaMemcpy(d_buf, h_pageable, HB, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(ev1));
    CHECK(cudaEventSynchronize(ev1));
    CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    bw_pageable = (double)HB * HITER / 1e9 / (ms / 1e3);

    for (int i = 0; i < 3; ++i)                                        // warm-up
        CHECK(cudaMemcpy(d_buf, h_pinned, HB, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < HITER; ++i)
        CHECK(cudaMemcpy(d_buf, h_pinned, HB, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(ev1));
    CHECK(cudaEventSynchronize(ev1));
    CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    bw_pinned = (double)HB * HITER / 1e9 / (ms / 1e3);

    printf("\n--- Part B: host-to-device copy, %zu MB ---\n", HB >> 20);
    printf("  pageable (malloc)      : %6.2f GB/s\n", bw_pageable);
    printf("  pinned   (cudaMallocHost): %6.2f GB/s\n", bw_pinned);
    printf("  speedup                : %.2fx\n", bw_pinned / bw_pageable);

    if (!(bw_pinned > 0.90 * bw_pageable)) {
        printf("  [FAIL] pinned copy was not faster than pageable\n");
        pass = 0;
    }

    // Correctness: the device buffer must equal what we sent.
    unsigned char* h_back = (unsigned char*)malloc(HB);
    CHECK(cudaMemcpy(h_back, d_buf, HB, cudaMemcpyDeviceToHost));
    size_t mismatch = 0;
    for (size_t i = 0; i < HB; ++i) if (h_back[i] != (unsigned char)(i & 0xff)) ++mismatch;
    if (mismatch) { printf("  [FAIL] %zu byte mismatches after round trip\n", mismatch); pass = 0; }

    printf("\n%s\n", pass ? "PASS" : "FAIL");

    free(h_back);
    free(h_pageable);
    CHECK(cudaFreeHost(h_pinned));
    CHECK(cudaFree(d_buf));
    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaEventDestroy(ev1));
    CHECK(cudaDeviceReset());
    return pass ? 0 : 1;
}
