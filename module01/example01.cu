

// =====================================================================
// Module 1 / Example 1 : "Latency hiding is a concurrency budget"
//
// GOAL
//   Show, by measurement, that a GPU does not make an individual memory
//   access faster -- it makes many of them overlap. We hold the work per
//   thread constant and vary only how many warps are resident per SM,
//   then report achieved dependent-load throughput.
//
//   Each thread walks a pointer chain:  idx = next[idx], 128 times.
//   Every step depends on the previous one, so a single thread has
//   exactly ONE memory request outstanding at a time and can do nothing
//   but wait. The only source of memory-level parallelism available to
//   the machine is *other warps*.
//
//   The chase table is 32 MB and the successor function is a full-period
//   LCG permutation, so consecutive steps land at effectively random
//   addresses: every step is a fresh 32-byte sector, served by L2 (the
//   table fits inside the 48 MB L2, which keeps this a *latency*
//   experiment rather than a DRAM-bandwidth experiment).
//
//   Little's Law says the achievable rate is
//       requests in flight / latency per request.
//   One warp per SM gives 40 SMs * 32 lanes = 1280 requests in flight.
//   Multiply the warps, multiply the requests in flight -- until some
//   other resource (L2 sector throughput) becomes the binding limit and
//   the curve flattens. Watch where that happens.
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

// ---- Problem constants ----------------------------------------------
static const unsigned int LOG2_N = 23;                 // 2^23 entries
static const unsigned int N      = 1u << LOG2_N;       // 8,388,608
static const unsigned int MASK   = N - 1u;
static const int          STEPS  = 256;                // chain length per thread
static const int          ITERS  = 30;                 // timed iterations
static const double       PEAK_GBS = 432.0;            // RTX 3500 Ada Laptop

// Full-period LCG permutation on Z_2^26:  a = 1 (mod 4), c odd.
__host__ __device__ __forceinline__
unsigned int successor(unsigned int x)
{
    return (unsigned int)((1103515245ull * (unsigned long long)x + 12345ull)
                          & (unsigned long long)MASK);
}

// Deterministic, well-spread starting point for thread `tid`.
__host__ __device__ __forceinline__
unsigned int start_index(unsigned int tid)
{
    return (unsigned int)(((unsigned long long)tid * 2654435761ull)
                          & (unsigned long long)MASK);
}

// ---------------------------------------------------------------------
// One dependent load per iteration. No ILP, no unrolling benefit: the
// address of step s+1 is the *value* returned by step s.
// ---------------------------------------------------------------------
__global__ void chase(const unsigned int* __restrict__ next,
                      unsigned int* __restrict__ out,
                      int steps)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = start_index(tid);
    for (int s = 0; s < steps; ++s)
        idx = next[idx];
    out[tid] = idx;
}

int main(void)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    const int nSMs = p.multiProcessorCount;

    printf("=== %s : %d SMs, %d threads/SM (%d warps), %d blocks/SM ===\n",
           p.name, nSMs, p.maxThreadsPerMultiProcessor,
           p.maxThreadsPerMultiProcessor / p.warpSize,
           p.maxBlocksPerMultiProcessor);
    printf("Chase table: %u entries (%.0f MB), chain length %d, %d timed iters\n\n",
           N, N * 4.0 / (1024.0 * 1024.0), STEPS, ITERS);

    // ---- build the successor table on the host ----------------------
    unsigned int* h_next = (unsigned int*)malloc((size_t)N * sizeof(unsigned int));
    if (!h_next) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (unsigned int i = 0; i < N; ++i) h_next[i] = successor(i);

    unsigned int* d_next = nullptr;
    CHECK(cudaMalloc(&d_next, (size_t)N * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_next, h_next, (size_t)N * sizeof(unsigned int),
                     cudaMemcpyHostToDevice));

    // blockDim = 32 => one warp per block => warpsPerSM == blocksPerSM.
    // This lets us dial resident warps per SM directly, and the grid is
    // always exactly one wave (blocksPerSM * nSMs blocks).
    const int warpsPerSM[] = { 1, 2, 3, 4, 5, 6, 8, 12, 16, 24 };
    const int nCfg = (int)(sizeof(warpsPerSM) / sizeof(warpsPerSM[0]));

    const int maxThreads = 24 * 32 * nSMs;
    unsigned int* d_out = nullptr;
    CHECK(cudaMalloc(&d_out, (size_t)maxThreads * sizeof(unsigned int)));
    unsigned int* h_out = (unsigned int*)malloc((size_t)maxThreads * sizeof(unsigned int));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    printf("%6s %7s %8s %12s %9s %9s %9s %8s %8s %6s\n",
           "warps", "blocks", "threads", "loads", "ms", "Gload/s", "GB/s*",
           "%peak", "ns/step", "vs 1w");
    printf("------ ------- -------- ------------ --------- --------- --------- -------- -------- ------\n");

    double base_rate = 0.0;
    double best_rate = 0.0;
    int    allPass   = 1;

    for (int c = 0; c < nCfg; ++c) {
        const int W        = warpsPerSM[c];
        const int nBlocks  = W * nSMs;           // blockDim = 32 => W warps/SM
        const int nThreads = nBlocks * 32;

        // warm-up (also forces module load and first-touch of d_next)
        chase<<<nBlocks, 32>>>(d_next, d_out, STEPS);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(t0));
        for (int it = 0; it < ITERS; ++it)
            chase<<<nBlocks, 32>>>(d_next, d_out, STEPS);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaGetLastError());
        CHECK(cudaEventSynchronize(t1));

        float ms_total = 0.0f;
        CHECK(cudaEventElapsedTime(&ms_total, t0, t1));
        const double ms = ms_total / ITERS;

        const double loads = (double)nThreads * (double)STEPS;
        const double rate  = loads / (ms * 1.0e-3) / 1.0e9;        // Gload/s
        // Each dependent load lands in its own 32 B sector (random address),
        // so DRAM traffic is 32 B per load, not 4 B.
        const double gbs   = loads * 32.0 / (ms * 1.0e-3) / 1.0e9;
        const double nsper = ms * 1.0e6 / (double)STEPS;           // ns per chain step

        // ---- validate against a CPU chase of the same chains --------
        CHECK(cudaMemcpy(h_out, d_out, (size_t)nThreads * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));
        int mism = 0;
        for (int t = 0; t < nThreads; ++t) {
            unsigned int idx = start_index((unsigned int)t);
            for (int s = 0; s < STEPS; ++s) idx = h_next[idx];
            if (h_out[t] != idx) {
                if (mism < 5)
                    printf("  [MISMATCH] thread %d: gpu=%u cpu=%u\n", t, h_out[t], idx);
                ++mism;
            }
        }
        if (mism) allPass = 0;

        if (c == 0) base_rate = rate;

        printf("%6d %7d %8d %12.0f %9.3f %9.2f %9.1f %7.1f%% %8.2f %5.2fx%s\n",
               W, nBlocks, nThreads, loads, ms, rate, gbs,
               100.0 * gbs / PEAK_GBS, nsper, rate / base_rate,
               mism ? "  <-- FAIL" : "");

        if (rate > best_rate) best_rate = rate;
    }

    printf("\n*GB/s counts one 32 B sector per dependent load: the addresses are\n");
    printf(" random, so the 4 B actually consumed costs a whole sector. %%peak is\n");
    printf(" measured against 432 GB/s of DRAM bandwidth; values above 100%% are\n");
    printf(" legitimate here -- the 32 MB table lives inside the 48 MB L2, and L2\n");
    printf(" bandwidth is several times DRAM bandwidth.\n");
    printf("\nThroughput gain from 1 warp/SM to the best configuration: %.2fx\n",
           base_rate > 0.0 ? best_rate / base_rate : 0.0);
    printf("Note the ns/step column: per-step latency never gets *shorter*. Past\n");
    printf("the knee it gets longer, because requests now queue. Only aggregate\n");
    printf("throughput improves -- that is the entire GPU bargain.\n");

    // ---- self-validation -------------------------------------------
    //  1. every configuration must reproduce the CPU chase exactly
    //  2. adding warps must actually buy throughput
    int ok = allPass;
    if (!allPass) printf("\n[FAIL] GPU chase disagreed with CPU reference\n");
    if (!(best_rate >= 2.0 * base_rate)) {
        printf("\n[FAIL] expected >=2x throughput gain from added warps\n");
        ok = 0;
    }
    printf("\n%s\n", ok ? "PASS" : "FAIL");

    free(h_next); free(h_out);
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    CHECK(cudaFree(d_next));
    CHECK(cudaFree(d_out));
    CHECK(cudaDeviceReset());
    return ok ? 0 : 1;
}
