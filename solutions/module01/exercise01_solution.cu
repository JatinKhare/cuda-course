// =====================================================================
// SOLUTION -- Module 1 / Exercise 1
// BUILD:  nvcc -arch=sm_89 -o exercise01_solution.exe exercise01_solution.cu
// RUN:    .\exercise01_solution.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
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

__device__ __forceinline__ unsigned int smid()
{
    unsigned int r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}

// TODO 4 solved: one designated thread stores. threadIdx.x == 0 is the
// conventional choice. All threads of a block are resident on the same
// SM for the block's entire lifetime (a block never migrates), so any
// thread would report the same %smid -- but having 128 threads write the
// same address is a wasteful (though benign) write conflict, and it is
// the habit that turns into a real race the moment the value differs.
__global__ void report_sm(unsigned int* out)
{
    if (threadIdx.x == 0)
        out[blockIdx.x] = smid();
}

int main(void)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));

    int memClockKHz = 0, busWidthBits = 0, l2Bytes = 0;
    CHECK(cudaDeviceGetAttribute(&memClockKHz,  cudaDevAttrMemoryClockRate,      dev));
    CHECK(cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, dev));
    CHECK(cudaDeviceGetAttribute(&l2Bytes,      cudaDevAttrL2CacheSize,          dev));

    printf("=== Device %d: %s (sm_%d%d) ===\n", dev, p.name, p.major, p.minor);
    printf("  SMs                        : %d\n",        p.multiProcessorCount);
    printf("  Warp size                  : %d\n",        p.warpSize);
    printf("  Max threads / block        : %d\n",        p.maxThreadsPerBlock);
    printf("  Max threads / SM           : %d\n",        p.maxThreadsPerMultiProcessor);
    printf("  Max blocks  / SM           : %d\n",        p.maxBlocksPerMultiProcessor);
    printf("  32-bit registers / SM      : %d\n",        p.regsPerMultiprocessor);
    printf("  Shared mem / block (max)   : %zu B\n",     p.sharedMemPerBlock);
    printf("  Shared mem / SM            : %zu B\n",     p.sharedMemPerMultiprocessor);
    printf("  L2 cache                   : %d B\n",      l2Bytes);
    printf("  Global memory              : %.2f GiB\n",  p.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("  Memory clock               : %d kHz\n",    memClockKHz);
    printf("  Memory bus width           : %d bits\n",   busWidthBits);

    // TODO 1 solved.
    //   bytes/transfer = busWidthBits / 8
    //   transfers/s    = 2 * clock   (GDDR is double-data-rate: one
    //                                 transfer on each clock edge)
    //   CUDA reports the *command* clock in kHz, not the effective
    //   data rate, hence the explicit factor of 2.
    double peakBW_GBs = 2.0 * (double)memClockKHz * 1.0e3
                      * (busWidthBits / 8.0) / 1.0e9;

    // TODO 2 solved.
    int maxWarpsPerSM        = p.maxThreadsPerMultiProcessor / p.warpSize;
    int maxWarpsDevice       = maxWarpsPerSM * p.multiProcessorCount;
    long long threadsInFlight = (long long)p.maxThreadsPerMultiProcessor
                              * p.multiProcessorCount;

    printf("\n--- Derived ---\n");
    printf("  Peak BW                    : %.1f GB/s\n", peakBW_GBs);
    printf("  Max resident warps / SM    : %d\n",        maxWarpsPerSM);
    printf("  Max resident warps / GPU   : %d\n",        maxWarpsDevice);
    printf("  Max threads in flight      : %lld\n",      threadsInFlight);

    if (maxWarpsPerSM * p.warpSize != p.maxThreadsPerMultiProcessor) {
        printf("  [FAIL] warps/SM inconsistent with threads/SM\n"); return 1;
    }
    if (threadsInFlight != (long long)maxWarpsDevice * p.warpSize) {
        printf("  [FAIL] threadsInFlight inconsistent with warps/GPU\n"); return 1;
    }
    printf("  [OK] derived residency numbers are self-consistent\n");

    // TODO 3 solved: 2 blocks per SM.
    int threadsPerBlock = 128;
    int nBlocks = 2 * p.multiProcessorCount;

    unsigned int *d_sm = nullptr;
    unsigned int *h_sm = (unsigned int*)malloc(nBlocks * sizeof(unsigned int));
    CHECK(cudaMalloc(&d_sm, nBlocks * sizeof(unsigned int)));
    CHECK(cudaMemset(d_sm, 0xFF, nBlocks * sizeof(unsigned int)));

    report_sm<<<nBlocks, threadsPerBlock>>>(d_sm);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_sm, d_sm, nBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    int* hist = (int*)calloc(p.multiProcessorCount, sizeof(int));
    int distinct = 0, bad = 0;
    for (int b = 0; b < nBlocks; ++b) {
        unsigned int s = h_sm[b];
        if (s >= (unsigned)p.multiProcessorCount) { bad++; continue; }
        if (hist[s]++ == 0) distinct++;
    }

    printf("\n--- Block -> SM mapping (%d blocks x %d threads) ---\n", nBlocks, threadsPerBlock);
    for (int s = 0; s < p.multiProcessorCount; ++s)
        printf("  SM %2d : %d block(s)\n", s, hist[s]);
    printf("  distinct SMs used = %d / %d, unwritten/invalid entries = %d\n",
           distinct, p.multiProcessorCount, bad);

    free(hist); free(h_sm);
    CHECK(cudaFree(d_sm));
    CHECK(cudaDeviceReset());
    return 0;
}
