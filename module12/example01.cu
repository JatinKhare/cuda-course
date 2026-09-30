// ============================================================================
// Module 12 / example01.cu -- The reduction ladder: six versions, one problem
//
// GOAL : Sum 2^26 floats. Six kernels compute the same sum. Each one repairs a
//        specific, nameable defect of the one before it. The harness times all
//        six back to back and reports each as a percentage of the measured
//        streaming ceiling, because reduction is a bandwidth problem and the
//        only honest scoreboard is "how much of the bus did you use".
//
// BUILD: nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN  : .\example01.exe
//
// SASS : nvcc -arch=sm_89 -O3 -c -o example01.o example01.cu
//        cuobjdump -sass example01.o > example01.sass
//        (look for SHFL.DOWN in reduce5/reduce6 and for the absence of a
//         BAR.SYNC between the five shuffles)
//
// Timing follows AUTHORING_SPEC section 12: duration-based clock warm-up,
// iteration count auto-scaled so every timed segment is ~10 ms, all six
// configurations timed back to back in one loop, sweep order ROTATED, min of
// 4 sweeps, validation in a separate second pass.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(_e), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_KERNEL()                                                         \
    do {                                                                       \
        CHECK(cudaGetLastError());                                             \
        CHECK(cudaDeviceSynchronize());                                        \
    } while (0)

static const unsigned BS = 256;               // threads per block, all versions
static const long long N  = 1LL << 26;        // 67,108,864 floats = 256 MiB
static const int NCFG     = 7;   // six ladder versions + the streaming ceiling
static const int NSWEEP   = 4;

// ---------------------------------------------------------------------------
// Warp-level tail. Module 30 owns __shfl_*_sync as a topic; Module 12 uses it
// because the algorithm demands it. The mask names the 32 lanes that must
// participate; on sm_70+ that mask is how you CREATE the convergence the
// hardware no longer gives you for free (Module 8).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warpReduceSum(float v)
{
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;   // only lane 0 holds the full sum
}

// ---------------------------------------------------------------------------
// v1 -- interleaved addressing with a modulo test. Maximally divergent.
// ---------------------------------------------------------------------------
__global__ void reduce1(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    unsigned tid = threadIdx.x;
    long long i  = (long long)blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// v2 -- same tree, contiguous-thread indexing. Divergence mostly gone; the
//       stride-2s access into shared memory introduces bank conflicts.
// ---------------------------------------------------------------------------
__global__ void reduce2(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    unsigned tid = threadIdx.x;
    long long i  = (long long)blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        unsigned idx = 2 * s * tid;
        if (idx < blockDim.x) sdata[idx] += sdata[idx + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// v3 -- sequential addressing. Conflict-free and non-divergent, but half the
//       threads are idle on the very first iteration and it only gets worse.
// ---------------------------------------------------------------------------
__global__ void reduce3(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    unsigned tid = threadIdx.x;
    long long i  = (long long)blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// v4 -- first add during the global load. Half the blocks, each thread loads
//       and adds two elements before the tree starts.
// ---------------------------------------------------------------------------
__global__ void reduce4(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    unsigned tid = threadIdx.x;
    long long i  = (long long)blockIdx.x * (blockDim.x * 2) + tid;
    float v = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    sdata[tid] = v;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// v5 -- warp-level tail. The shared-memory tree stops when the survivors fit
//       in one warp; the last five steps are register-to-register shuffles.
//
//       The legacy version of this tail used `volatile __shared__` and no
//       synchronization at all. That is BROKEN on sm_70+ (Modules 8 and 9);
//       example02.cu part F shows it failing. Never write it.
// ---------------------------------------------------------------------------
__global__ void reduce5(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    unsigned tid = threadIdx.x;
    long long i  = (long long)blockIdx.x * (blockDim.x * 2) + tid;
    float v = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    sdata[tid] = v;
    __syncthreads();

    // Stop at s == 32: below that, every surviving thread is in warp 0.
    for (unsigned s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        float w = warpReduceSum(sdata[tid]);
        if (tid == 0) out[blockIdx.x] = w;
    }
}

// ---------------------------------------------------------------------------
// v6 -- grid-stride accumulation into a register, then a fully unrolled tree
//       whose bounds are compile-time constants.
//
//       The grid is sized to the machine, not to the data. Each thread reads
//       many elements into one register before any shared memory is touched,
//       so the shared-memory tree and the barriers run ONCE per block instead
//       of once per 512 elements.
// ---------------------------------------------------------------------------
template <unsigned BLOCK>
__global__ void reduce6(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BLOCK];
    unsigned tid   = threadIdx.x;
    long long i    = (long long)blockIdx.x * (BLOCK * 2) + tid;
    long long step = (long long)BLOCK * 2 * gridDim.x;

    float sum = 0.0f;
    while (i + BLOCK < n) {              // two independent loads per iteration
        sum += in[i] + in[i + BLOCK];
        i   += step;
    }
    while (i < n) { sum += in[i]; i += step; }   // ragged tail

    sdata[tid] = sum;
    __syncthreads();

    if (BLOCK >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (BLOCK >=  512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (BLOCK >=  256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (BLOCK >=  128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }

    if (tid < 32) {
        float w = sdata[tid];
        if (BLOCK >= 64) w += sdata[tid + 32];
        w = warpReduceSum(w);
        if (tid == 0) out[blockIdx.x] = w;
    }
}

// ---------------------------------------------------------------------------
// Finalization for every version: one block, grid-stride, fixed order.
// Deterministic by construction -- example02.cu part D measures that.
// ---------------------------------------------------------------------------
__global__ void finalizeKernel(const float* __restrict__ partial, float* out, long long m)
{
    __shared__ float sdata[1024];
    unsigned tid = threadIdx.x;
    float sum = 0.0f;
    for (long long i = tid; i < m; i += blockDim.x) sum += partial[i];
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float w = warpReduceSum(sdata[tid]);
        if (tid == 0) *out = w;
    }
}

// ---------------------------------------------------------------------------
// A pure streaming kernel. This is the ceiling: it reads every byte the
// reduction reads and writes almost nothing, but does no tree at all.
// Nothing above it can be faster; anything well below it is wasting the bus.
// ---------------------------------------------------------------------------
__global__ void streamCeiling(const float* __restrict__ in, float* out, long long n)
{
    long long i    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)blockDim.x * gridDim.x;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (; i + 3 * step < n; i += 4 * step) {
        acc0 += in[i];
        acc1 += in[i + step];
        acc2 += in[i + 2 * step];
        acc3 += in[i + 3 * step];
    }
    for (; i < n; i += step) acc0 += in[i];
    float s = (acc0 + acc1) + (acc2 + acc3);
    if (s == 1.2345e-30f) out[blockIdx.x] = s;   // never true; keeps the loads live
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
struct Cfg {
    const char* name;
    const char* fixes;
    int   blocks;
    
};

static void launchVersion(int v, const float* d_in, float* d_partial,
                          long long n, int blocks, int gridV6)
{
    switch (v) {
        case 0: reduce1<<<blocks, BS>>>(d_in, d_partial, n); break;
        case 1: reduce2<<<blocks, BS>>>(d_in, d_partial, n); break;
        case 2: reduce3<<<blocks, BS>>>(d_in, d_partial, n); break;
        case 3: reduce4<<<blocks, BS>>>(d_in, d_partial, n); break;
        case 4: reduce5<<<blocks, BS>>>(d_in, d_partial, n); break;
        case 5: reduce6<BS><<<gridV6, BS>>>(d_in, d_partial, n); break;
        case 6: streamCeiling<<<gridV6, BS>>>(d_in, d_partial, n); break;
        default: break;
    }
}

int main(void)
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int smCount = 0;
    CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));

    printf("Module 12 example 01 -- the reduction ladder\n");
    printf("GPU: %s, %d SMs, CC %d.%d\n", prop.name, smCount, prop.major, prop.minor);
    printf("N = %lld floats = %.1f MiB (L2 is 48 MB, so this is DRAM traffic)\n\n",
           N, (double)N * 4.0 / (1024.0 * 1024.0));

    // ---- deterministic, index-derived host data ---------------------------
    float* h_in = (float*)malloc((size_t)N * sizeof(float));
    if (!h_in) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (long long i = 0; i < N; ++i) {
        unsigned h = (unsigned)i * 2654435761u;
        h ^= h >> 15;
        h_in[i] = (float)(h & 0xFFFFu) * (1.0f / 65536.0f);   // [0,1)
    }
    double ref = 0.0;                     // double-precision reference
    for (long long i = 0; i < N; ++i) ref += (double)h_in[i];

    float *d_in = nullptr, *d_partial = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in, (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_partial, (size_t)(N / BS + 64) * sizeof(float)));
    CHECK(cudaMalloc(&d_out, sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // ---- grid for v6: one occupancy-limited wave over the machine ---------
    int blocksPerSM = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, reduce6<BS>, BS, 0));
    int gridV6 = blocksPerSM * smCount;

    Cfg cfg[NCFG] = {
        {"v1 interleaved, tid mod 2s ", "-- baseline", (int)(N / BS)},
        {"v2 contiguous index 2*s*tid ", "divergence ", (int)(N / BS)},
        {"v3 sequential addressing    ", "bank confl.", (int)(N / BS)},
        {"v4 first add during load    ", "idle threads", (int)(N / (BS * 2))},
        {"v5 warp tail, __shfl_down   ", "tail barriers", (int)(N / (BS * 2))},
        {"v6 grid-stride + unrolled   ", "block count ", gridV6},
        {"   streaming ceiling (no tree)", "n/a        ", gridV6},
    };
    printf("blocks/SM for v6 (occupancy API): %d  ->  grid = %d blocks\n",
           blocksPerSM, gridV6);
    printf("v1-v3 launch %d blocks; v4-v5 launch %d blocks.\n\n",
           (int)(N / BS), (int)(N / (BS * 2)));

    cudaEvent_t evStart, evStop;
    CHECK(cudaEventCreate(&evStart));
    CHECK(cudaEventCreate(&evStop));

    // ---- clock warm-up: duration based, not iteration based ---------------
    {
        cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        float acc = 0.0f;
        CHECK(cudaEventRecord(a));
        while (acc < 1500.0f) {
            for (int k = 0; k < 20; ++k)
                reduce6<BS><<<gridV6, BS>>>(d_in, d_partial, N);
            CHECK(cudaEventRecord(b));
            CHECK(cudaEventSynchronize(b));
            CHECK(cudaEventElapsedTime(&acc, a, b));
        }
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
        CHECK_KERNEL();
    }

    // ---- auto-scale iteration count to ~10 ms per timed segment -----------
    int iters[NCFG];
    for (int v = 0; v < NCFG; ++v) {
        launchVersion(v, d_in, d_partial, N, cfg[v].blocks, gridV6);
        CHECK_KERNEL();
        CHECK(cudaEventRecord(evStart));
        for (int k = 0; k < 5; ++k)
            launchVersion(v, d_in, d_partial, N, cfg[v].blocks, gridV6);
        CHECK(cudaEventRecord(evStop));
        CHECK(cudaEventSynchronize(evStop));
        float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        double per = ms / 5.0;
        int it = (int)(10.0 / (per > 1e-6 ? per : 1e-6));
        if (it < 20) it = 20;
        if (it > 400) it = 400;
        iters[v] = it;
    }

    // ---- TIMING PASS: all six back to back, rotated order, min of NSWEEP --
    double best[NCFG];
    for (int v = 0; v < NCFG; ++v) best[v] = 1e30;

    for (int sweep = 0; sweep < NSWEEP; ++sweep) {
        for (int q = 0; q < NCFG; ++q) {
            int v = (q + sweep) % NCFG;              // spec section 12 rule 9
            CHECK(cudaEventRecord(evStart));
            for (int k = 0; k < iters[v]; ++k)
                launchVersion(v, d_in, d_partial, N, cfg[v].blocks, gridV6);
            CHECK(cudaEventRecord(evStop));
            CHECK(cudaEventSynchronize(evStop));
            float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
            double per = ms / iters[v];
            if (per < best[v]) best[v] = per;
        }
    }
    CHECK_KERNEL();

    // The ceiling is measured INSIDE the same rotated sweep as the ladder, so
    // it carries exactly the same clock history as every version it is the
    // denominator for.
    double ceilMs  = best[NCFG - 1];
    double ceilGBs = (double)N * 4.0 / (ceilMs * 1.0e-3) / 1.0e9;

    // ---- PART B: the finalization pass nobody times ------------------------
    // Pass 1 leaves `blocks` partial sums. Somebody has to add those up too.
    // v1..v3 leave 262144 of them; v6 leaves 240. Time the second kernel for
    // each distinct partial count and add it to the per-version total.
    double finMs[NCFG];
    {
        long long ms_of[NCFG];
        for (int v = 0; v < NCFG; ++v) ms_of[v] = (v >= 5) ? gridV6 : cfg[v].blocks;
        for (int v = 0; v < NCFG - 1; ++v) {
            double b = 1e30;
            finalizeKernel<<<1, 1024>>>(d_partial, d_out, ms_of[v]);
            CHECK_KERNEL();
            for (int sweep = 0; sweep < NSWEEP; ++sweep) {
                CHECK(cudaEventRecord(evStart));
                for (int k = 0; k < 50; ++k)
                    finalizeKernel<<<1, 1024>>>(d_partial, d_out, ms_of[v]);
                CHECK(cudaEventRecord(evStop));
                CHECK(cudaEventSynchronize(evStop));
                float t = 0.0f; CHECK(cudaEventElapsedTime(&t, evStart, evStop));
                if (t / 50.0 < b) b = t / 50.0;
            }
            finMs[v] = b;
        }
    }

    // ---- PART C: the same ladder on data that fits in L2 -------------------
    // 4 MiB instead of 256 MiB. Nothing here is a DRAM bandwidth number; the
    // point is that per-block overhead, which is invisible at 256 MiB, is the
    // whole story when the data is small.
    const long long NS = 1LL << 20;
    double smallBest[NCFG];
    {
        for (int v = 0; v < NCFG; ++v) smallBest[v] = 1e30;
        int sBlocks[NCFG];
        for (int v = 0; v < NCFG; ++v)
            sBlocks[v] = (v <= 2) ? (int)(NS / BS)
                       : (v <= 4) ? (int)(NS / (BS * 2)) : gridV6;
        for (int v = 0; v < NCFG - 1; ++v) {
            launchVersion(v, d_in, d_partial, NS, sBlocks[v], gridV6);
        }
        CHECK_KERNEL();
        for (int sweep = 0; sweep < NSWEEP; ++sweep) {
            for (int q = 0; q < NCFG - 1; ++q) {
                int v = (q + sweep) % (NCFG - 1);
                CHECK(cudaEventRecord(evStart));
                for (int k = 0; k < 100; ++k)
                    launchVersion(v, d_in, d_partial, NS, sBlocks[v], gridV6);
                CHECK(cudaEventRecord(evStop));
                CHECK(cudaEventSynchronize(evStop));
                float t = 0.0f; CHECK(cudaEventElapsedTime(&t, evStart, evStop));
                if (t / 100.0 < smallBest[v]) smallBest[v] = t / 100.0;
            }
        }
        CHECK_KERNEL();
    }

    // ---- VALIDATION PASS (separate, after all timing) ---------------------
    int pass = 1;
    float got[NCFG];
    for (int v = 0; v < NCFG - 1; ++v) {
        launchVersion(v, d_in, d_partial, N, cfg[v].blocks, gridV6);
        CHECK_KERNEL();
        long long m = (v == 5) ? gridV6 : cfg[v].blocks;
        finalizeKernel<<<1, 1024>>>(d_partial, d_out, m);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(&got[v], d_out, sizeof(float), cudaMemcpyDeviceToHost));
        double rel = fabs((double)got[v] - ref) / fabs(ref);
        if (!(rel <= 1e-5)) pass = 0;
    }

    // ---- report -----------------------------------------------------------
    printf("Measured streaming ceiling (read-only, no tree): %.4f ms = %.1f GB/s"
           " (%.1f%% of the 432.0 GB/s pin peak)\n\n",
           ceilMs, ceilGBs, 100.0 * ceilGBs / 432.0);

    printf("%-30s %8s %9s %9s %9s %9s\n",
           "version", "ms", "GB/s", "%ceiling", "%peak", "vs v1");
    printf("---------------------------------------------------------------------------------\n");
    for (int v = 0; v < NCFG; ++v) {
        double gbs = (double)N * 4.0 / (best[v] * 1.0e-3) / 1.0e9;
        printf("%-30s %8.4f %9.1f %8.1f%% %8.1f%% %8.2fx\n",
               cfg[v].name, best[v], gbs,
               100.0 * gbs / ceilGBs, 100.0 * gbs / 432.0,
               best[0] / best[v]);
    }
    printf("\niterations per timed segment: ");
    for (int v = 0; v < NCFG; ++v) printf("%d ", iters[v]);
    printf("\n\n");

    printf("Part B -- the finalization pass, and the end-to-end total:\n");
    printf("%-30s %9s %9s %9s %9s\n",
           "version", "partials", "pass1 ms", "pass2 ms", "total ms");
    printf("---------------------------------------------------------------------------------\n");
    for (int v = 0; v < NCFG - 1; ++v) {
        long long m = (v == 5) ? gridV6 : cfg[v].blocks;
        printf("%-30s %9lld %9.4f %9.4f %9.4f\n",
               cfg[v].name, m, best[v], finMs[v], best[v] + finMs[v]);
    }

    printf("\nPart C -- the same six kernels on 4 MiB (L2-resident; these are\n"
           "          NOT DRAM bandwidth numbers, per spec section 12 rule 7):\n");
    printf("%-30s %9s %9s\n", "version", "ms", "vs v6");
    printf("---------------------------------------------------\n");
    for (int v = 0; v < NCFG - 1; ++v)
        printf("%-30s %9.5f %8.2fx\n", cfg[v].name, smallBest[v],
               smallBest[v] / smallBest[NCFG - 2]);
    printf("\n");

    printf("validation (double reference = %.6f):\n", ref);
    for (int v = 0; v < NCFG - 1; ++v) {
        double rel = fabs((double)got[v] - ref) / fabs(ref);
        printf("  %-30s got %.6f  rel.err %.3e  %s\n",
               cfg[v].name, (double)got[v], rel, (rel <= 1e-5) ? "ok" : "BAD");
    }

    printf("\nEach step repairs the defect named in the previous row:\n");
    for (int v = 1; v < NCFG - 1; ++v)
        printf("  v%d -> v%d removes: %s\n", v, v + 1, cfg[v].fixes);

    CHECK(cudaEventDestroy(evStart));
    CHECK(cudaEventDestroy(evStop));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_partial));
    CHECK(cudaFree(d_out));
    free(h_in);
    CHECK(cudaDeviceReset());

    printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
