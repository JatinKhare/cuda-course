// =====================================================================
// Module 1 / Exercise 1 : "Hardware census + block-to-SM residency"
//
// GOAL
//   Turn the abstract numbers the driver reports into a concrete mental
//   model of the machine you are about to program, and then *observe*
//   where blocks physically land.
//
//   Part A: derive hardware capacities from device attributes.
//   Part B: launch a kernel that reports which SM each block ran on,
//           and check your prediction about the distribution.
//
// BUILD:  nvcc -arch=sm_89 -o exercise01.exe exercise01.cu
// RUN:    .\exercise01.exe
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

// ---------------------------------------------------------------------
// Reads the hardware SM identifier of the SM this thread is executing on.
// %smid is a "special register" -- real silicon state, not a CUDA
// abstraction. There is no C++ API for it; you must drop to PTX.
// ---------------------------------------------------------------------
__device__ __forceinline__ unsigned int smid()
{
    unsigned int r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}

// ---------------------------------------------------------------------
// Each block writes the SM it was resident on into out[blockIdx.x].
//
// TODO 4: Exactly ONE thread per block should perform the store.
//         Which one, and how do you express that? (Think about what
//         would go wrong if all 128 threads stored.)
//         Also: is the value of smid() guaranteed to be identical for
//         every thread in the block? Justify your answer in your reply.
// ---------------------------------------------------------------------
__global__ void report_sm(unsigned int* out)
{
    //int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
    //if(thread_id%128 == 0)
        //out[thread_id/128] = smid();
    // YOUR CODE HERE (TODO 4)
    if(threadIdx.x ==0)
        out[blockIdx.x] = smid();
}

int main(void)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));

    // Clock / bus width are queried via attributes (the cudaDeviceProp
    // fields for these are deprecated in recent CUDA versions).
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

    // -----------------------------------------------------------------
    // TODO 1: Theoretical peak global-memory bandwidth, in GB/s
    //         (use 1 GB = 1e9 bytes, the convention NVIDIA specs use).
    //         Careful: GDDR6/GDDR6X is double-data-rate. Decide whether
    //         the reported clock already accounts for that, and say why
    //         in your answer. Your number should land near the published
    //         spec for this GPU -- if it is off by exactly 2x, you know
    //         which way to go.
    // -----------------------------------------------------------------
    double peakBW_GBs = 2*(busWidthBits * memClockKHz/(1000000 * 8));   // YOUR CODE HERE (TODO 1)

    // -----------------------------------------------------------------
    // TODO 2: Fill in the three residency numbers below.
    //   maxWarpsPerSM   : how many warps can be *resident* on one SM
    //                     (resident = context allocated, not necessarily
    //                     issuing this cycle)
    //   maxWarpsDevice  : ... across the whole GPU
    //   threadsInFlight : total threads that can be simultaneously
    //                     resident on the whole GPU
    // Derive them from the queried fields above -- do not hard-code.
    // -----------------------------------------------------------------
    int maxWarpsPerSM   = p.maxThreadsPerMultiProcessor/p.warpSize;        // YOUR CODE HERE (TODO 2)
    int maxWarpsDevice  = maxWarpsPerSM*p.multiProcessorCount;        // YOUR CODE HERE (TODO 2)
    long long threadsInFlight = p.maxThreadsPerMultiProcessor*p.multiProcessorCount;  // YOUR CODE HERE (TODO 2)

    printf("\n--- Derived ---\n");
    printf("  Peak BW                    : %.1f GB/s\n", peakBW_GBs);
    printf("  Max resident warps / SM    : %d\n",        maxWarpsPerSM);
    printf("  Max resident warps / GPU   : %d\n",        maxWarpsDevice);
    printf("  Max threads in flight      : %lld\n",      threadsInFlight);

    // Sanity checks (these must pass for your TODO 2 to be self-consistent)
    if (maxWarpsPerSM * p.warpSize != p.maxThreadsPerMultiProcessor) {
        printf("  [FAIL] warps/SM inconsistent with threads/SM\n"); return 1;
    }
    if (threadsInFlight != (long long)maxWarpsDevice * p.warpSize) {
        printf("  [FAIL] threadsInFlight inconsistent with warps/GPU\n"); return 1;
    }
    printf("  [OK] derived residency numbers are self-consistent\n");

    // -----------------------------------------------------------------
    // Part B -- where do blocks actually run?
    //
    // TODO 3: Choose the launch configuration.
    //   Launch exactly 2 blocks per SM, with 128 threads per block.
    //   nBlocks must be computed from the device properties.
    // -----------------------------------------------------------------
    int threadsPerBlock = 128;
    int nBlocks = p.multiProcessorCount*2;   // YOUR CODE HERE (TODO 3)

    if (nBlocks <= 0) { printf("\nSet TODO 3 (nBlocks) to continue.\n"); return 0; }

    unsigned int *d_sm = nullptr;
    unsigned int *h_sm = (unsigned int*)malloc(nBlocks * sizeof(unsigned int));
    CHECK(cudaMalloc(&d_sm, nBlocks * sizeof(unsigned int)));
    CHECK(cudaMemset(d_sm, 0xFF, nBlocks * sizeof(unsigned int)));

    report_sm<<<nBlocks, threadsPerBlock>>>(d_sm);
    CHECK(cudaGetLastError());        // catches launch-configuration errors
    CHECK(cudaDeviceSynchronize());   // catches errors raised during execution

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
