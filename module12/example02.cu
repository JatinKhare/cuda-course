// ============================================================================
// Module 12 / example02.cu -- finalization, determinism, accuracy, primitives
//
// GOAL : Everything that happens AFTER a block has its partial sum.
//        A : the three multi-block finalization strategies, timed
//        B : determinism -- which of them is bit-reproducible, and across what
//        C : floating-point accuracy -- why the tree is MORE accurate than the
//            sequential loop it replaces
//        D : warp primitives -- __shfl_down_sync vs __reduce_add_sync vs
//            cg::reduce, and what the SASS shows
//        E : CUB BlockReduce / DeviceReduce, what you would actually ship
//        F : the legacy `volatile __shared__` warp tail, as a bug
//
// BUILD: nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler /Zc:preprocessor ^
//             -o example02.exe example02.cu
//        (the extra flags are for <cub/cub.cuh>; Module 9 used the same pair
//         for <cuda/atomic>. Everything except part E builds without them.)
// RUN  : .\example02.exe
//
// SASS : nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler /Zc:preprocessor ^
//             -c -o example02.o example02.cu
//        cuobjdump -sass example02.o > example02.sass
//        findstr /C:"REDUX" /C:"SHFL" example02.sass
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

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
    do { CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize()); } while (0)

static const unsigned  BS = 256;
static const long long N  = 1LL << 26;    // 67,108,864 -> 256 MiB per buffer

__device__ __forceinline__ float warpReduceSum(float v)
{
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

// One block's worth of work: grid-stride load into a register, shared tree,
// warp tail. Returns the block sum in every thread of warp 0 lane 0.
template <unsigned BLOCK>
__device__ __forceinline__ float blockReduceGridStride(const float* __restrict__ in,
                                                       long long n, float* sdata)
{
    unsigned tid   = threadIdx.x;
    long long i    = (long long)blockIdx.x * (BLOCK * 2) + tid;
    long long step = (long long)BLOCK * 2 * gridDim.x;
    float sum = 0.0f;
    while (i + BLOCK < n) { sum += in[i] + in[i + BLOCK]; i += step; }
    while (i < n)         { sum += in[i];                 i += step; }

    sdata[tid] = sum;
    __syncthreads();
    if (BLOCK >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (BLOCK >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (BLOCK >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
    float w = 0.0f;
    if (tid < 32) {
        w = sdata[tid];
        if (BLOCK >= 64) w += sdata[tid + 32];
        w = warpReduceSum(w);
    }
    return w;      // meaningful only in tid == 0
}

// The ceiling: read every byte, build no tree. Timed inside the same rotated
// sweep as the strategies it is the denominator for.
__global__ void streamCeiling(const float* __restrict__ in, float* out, long long n)
{
    long long i    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)blockDim.x * gridDim.x;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (; i + 3 * step < n; i += 4 * step) {
        a0 += in[i]; a1 += in[i + step];
        a2 += in[i + 2 * step]; a3 += in[i + 3 * step];
    }
    for (; i < n; i += step) a0 += in[i];
    float s = (a0 + a1) + (a2 + a3);
    if (s == 1.2345e-30f) out[blockIdx.x] = s;
}

// ---------------------------------------------------------------------------
// A(a) -- two kernels. Pass 1 writes one partial per block; pass 2 is a single
//         block that sums them in index order.
// ---------------------------------------------------------------------------
__global__ void passOne(const float* __restrict__ in, float* partial, long long n)
{
    __shared__ float sdata[BS];
    float w = blockReduceGridStride<BS>(in, n, sdata);
    if (threadIdx.x == 0) partial[blockIdx.x] = w;
}

__global__ void passTwo(const float* __restrict__ partial, float* out, long long m)
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
    if (tid < 32) { float w = warpReduceSum(sdata[tid]); if (tid == 0) *out = w; }
}

// ---------------------------------------------------------------------------
// A(b) -- one float atomicAdd per block. One launch, no partial array.
//         Module 10 measured float atomicAdd producing 10 distinct bit
//         patterns in 10 runs. This is that, inside a real algorithm.
// ---------------------------------------------------------------------------
__global__ void atomicFinal(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    float w = blockReduceGridStride<BS>(in, n, sdata);
    if (threadIdx.x == 0) atomicAdd(out, w);
}

// ---------------------------------------------------------------------------
// A(c) -- last-block-done flag. Every block stores its partial, fences, then
//         atomically increments a counter; the block that reads gridDim-1 back
//         knows every other block's store is visible and does the final tree.
//         One launch, and the final sum is still computed in INDEX order.
// ---------------------------------------------------------------------------
__device__ unsigned g_blocksDone = 0;

__global__ void lastBlockFinal(const float* __restrict__ in, float* partial,
                               float* out, long long n)
{
    __shared__ float sdata[BS];
    __shared__ bool  amLast;

    float w = blockReduceGridStride<BS>(in, n, sdata);

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = w;
        // Publish the payload BEFORE the ticket (Module 9's release pattern).
        // __threadfence() and not __threadfence_block(): the observer is
        // another SM, and L1 is not coherent across SMs (Module 4).
        __threadfence();
        unsigned ticket = atomicAdd(&g_blocksDone, 1u);
        amLast = (ticket == gridDim.x - 1);
    }
    __syncthreads();          // block-uniform condition below: legal (Module 9)

    if (amLast) {
        unsigned tid = threadIdx.x;
        float sum = 0.0f;
        for (unsigned i = tid; i < gridDim.x; i += blockDim.x) sum += partial[i];
        __syncthreads();
        sdata[tid] = sum;
        __syncthreads();
        for (unsigned s = blockDim.x / 2; s >= 32; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid < 32) {
            float r = warpReduceSum(sdata[tid]);
            if (tid == 0) { *out = r; g_blocksDone = 0; }
        }
    }
}

// ---------------------------------------------------------------------------
// D -- three spellings of the same warp reduction, over unsigned ints.
// ---------------------------------------------------------------------------
template <int MODE>
__global__ void uintReduce(const unsigned* __restrict__ in, unsigned* partial, long long n)
{
    __shared__ unsigned sdata[BS];
    unsigned tid   = threadIdx.x;
    long long i    = (long long)blockIdx.x * (BS * 2) + tid;
    long long step = (long long)BS * 2 * gridDim.x;
    unsigned sum = 0u;
    while (i + BS < n) { sum += in[i] + in[i + BS]; i += step; }
    while (i < n)      { sum += in[i];              i += step; }

    sdata[tid] = sum;
    __syncthreads();
    if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads();
    if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads();

    if (MODE == 2) {
        // cg::reduce over a 32-lane tile. Portable, and the compiler picks the
        // instruction: SHFL.BFLY on sm_89, REDUX where the type allows it.
        cg::thread_block_tile<32> warp =
            cg::tiled_partition<32>(cg::this_thread_block());
        if (tid < 32) {
            unsigned w = sdata[tid] + sdata[tid + 32];
            w = cg::reduce(warp, w, cg::plus<unsigned>());
            if (tid == 0) partial[blockIdx.x] = w;
        }
    } else if (tid < 32) {
        unsigned w = sdata[tid] + sdata[tid + 32];
        if (MODE == 0) {
            w += __shfl_down_sync(0xffffffffu, w, 16);
            w += __shfl_down_sync(0xffffffffu, w,  8);
            w += __shfl_down_sync(0xffffffffu, w,  4);
            w += __shfl_down_sync(0xffffffffu, w,  2);
            w += __shfl_down_sync(0xffffffffu, w,  1);
        } else {
            // sm_80+ hardware reduction instruction: one REDUX.SUM, not five
            // shuffles and five adds.
            w = __reduce_add_sync(0xffffffffu, w);
        }
        if (tid == 0) partial[blockIdx.x] = w;
    }
}

__global__ void uintFinal(const unsigned* __restrict__ p, unsigned* out, long long m)
{
    __shared__ unsigned sdata[256];
    unsigned tid = threadIdx.x, sum = 0u;
    for (long long i = tid; i < m; i += blockDim.x) sum += p[i];
    sdata[tid] = sum; __syncthreads();
    for (unsigned s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out = sdata[0];
}

// ---------------------------------------------------------------------------
// E -- CUB BlockReduce as the block-level primitive.
// ---------------------------------------------------------------------------
__global__ void cubBlockReduce(const float* __restrict__ in, float* partial, long long n)
{
    typedef cub::BlockReduce<float, BS> BR;
    __shared__ typename BR::TempStorage temp;
    long long i    = (long long)blockIdx.x * (BS * 2) + threadIdx.x;
    long long step = (long long)BS * 2 * gridDim.x;
    float sum = 0.0f;
    while (i + BS < n) { sum += in[i] + in[i + BS]; i += step; }
    while (i < n)      { sum += in[i];              i += step; }
    float b = BR(temp).Sum(sum);
    if (threadIdx.x == 0) partial[blockIdx.x] = b;
}

// ---------------------------------------------------------------------------
// F -- the legacy warp-synchronous tail. DO NOT WRITE THIS.
//      Pre-Volta this was correct because a warp had one program counter.
//      Modules 8 and 9 established that sm_70+ removed that guarantee, so it
//      is undefined here. It is shipped only so you can watch what undefined
//      looks like on a machine that often hides it.
// ---------------------------------------------------------------------------
__global__ void legacyVolatileTail(const float* __restrict__ in, float* partial, long long n)
{
    __shared__ float sdata[BS];
    float w = 0.0f;
    unsigned tid   = threadIdx.x;
    long long i    = (long long)blockIdx.x * (BS * 2) + tid;
    long long step = (long long)BS * 2 * gridDim.x;
    float sum = 0.0f;
    while (i + BS < n) { sum += in[i] + in[i + BS]; i += step; }
    while (i < n)      { sum += in[i];              i += step; }
    sdata[tid] = sum;
    __syncthreads();
    if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads();
    if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads();
    if (tid < 32) {
        volatile float* v = sdata;          // NOT synchronization (Module 9)
        v[tid] += v[tid + 32];
        v[tid] += v[tid + 16];
        v[tid] += v[tid +  8];
        v[tid] += v[tid +  4];
        v[tid] += v[tid +  2];
        v[tid] += v[tid +  1];
        w = v[0];
    }
    if (tid == 0) partial[blockIdx.x] = w;
}

// ---------------------------------------------------------------------------
// data fills
// ---------------------------------------------------------------------------
__global__ void fillOnes(float* a, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    for (; i < n; i += s) a[i] = 1.0f;
}

// Dyadic values 2^-10 .. 2^9: wide dynamic range, and every partial sum is
// exactly representable in double, so the double reference is EXACT.
__global__ void fillDyadic(float* a, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    for (; i < n; i += s) {
        unsigned h = (unsigned)i * 2654435761u; h ^= h >> 13;
        int e = (int)(h % 20u) - 10;
        a[i] = ldexpf(1.0f, e);
    }
}

__global__ void fillUint(unsigned* a, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    for (; i < n; i += s) {
        unsigned h = (unsigned)i * 2654435761u; h ^= h >> 15;
        a[i] = h & 0x1Fu;                    // <= 31, so 2^26*31 < 2^31
    }
}

// ---------------------------------------------------------------------------
static unsigned bitsOf(float f) { unsigned u; memcpy(&u, &f, 4); return u; }
static float    floatOf(unsigned u) { float f; memcpy(&f, &u, 4); return f; }

static int countDistinct(const unsigned* v, int n)
{
    int d = 0;
    for (int i = 0; i < n; ++i) {
        int seen = 0;
        for (int j = 0; j < i; ++j) if (v[j] == v[i]) { seen = 1; break; }
        if (!seen) ++d;
    }
    return d;
}

int main(void)
{
    int smCount = 0;
    CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
    int bpsm = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, passOne, BS, 0));
    const int GRID = bpsm * smCount;

    printf("Module 12 example 02 -- finalization, determinism, accuracy\n");
    printf("%d SMs x %d blocks/SM -> grid = %d blocks of %u threads\n",
           smCount, bpsm, GRID, BS);
    printf("N = %lld floats = %.0f MiB\n\n", N, (double)N * 4.0 / 1048576.0);

    float *d_in = nullptr, *d_partial = nullptr, *d_out = nullptr;
    unsigned *d_iu = nullptr, *d_upart = nullptr, *d_uout = nullptr;
    CHECK(cudaMalloc(&d_in,      (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_partial, (size_t)(4 * GRID + 64) * sizeof(float)));
    CHECK(cudaMalloc(&d_out,     sizeof(float)));
    CHECK(cudaMalloc(&d_iu,      (size_t)N * sizeof(unsigned)));
    CHECK(cudaMalloc(&d_upart,   (size_t)(GRID + 64) * sizeof(unsigned)));
    CHECK(cudaMalloc(&d_uout,    sizeof(unsigned)));

    fillDyadic<<<GRID, BS>>>(d_in, N);
    fillUint<<<GRID, BS>>>(d_iu, N);
    CHECK_KERNEL();

    // exact host reference for the dyadic set (all values are powers of two
    // and the running sum stays under 2^53, so double is exact here)
    float* h = (float*)malloc((size_t)N * sizeof(float));
    if (!h) { fprintf(stderr, "host alloc failed\n"); return 1; }
    CHECK(cudaMemcpy(h, d_in, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    double exactDyadic = 0.0;
    for (long long i = 0; i < N; ++i) exactDyadic += (double)h[i];

    // =====================================================================
    printf("=== A: three multi-block finalization strategies ===\n");
    // warm-up, duration based
    {
        cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        float acc = 0.0f; CHECK(cudaEventRecord(a));
        while (acc < 1500.0f) {
            for (int k = 0; k < 20; ++k) passOne<<<GRID, BS>>>(d_in, d_partial, N);
            CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
            CHECK(cudaEventElapsedTime(&acc, a, b));
        }
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); CHECK_KERNEL();
    }

    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    const int NF = 4, NSW = 4, ITER = 20;
    const char* fname[NF] = { "(a) two kernel launches      ",
                              "(b) atomicAdd per block      ",
                              "(c) threadfence + last block ",
                              "    streaming ceiling        " };
    double fbest[NF] = { 1e30, 1e30, 1e30, 1e30 };

    for (int sweep = 0; sweep < NSW; ++sweep) {
        for (int q = 0; q < NF; ++q) {
            int f = (q + sweep) % NF;                     // rotate (spec 12.9)
            CHECK(cudaEventRecord(e0));
            for (int k = 0; k < ITER; ++k) {
                if (f == 0) {
                    passOne<<<GRID, BS>>>(d_in, d_partial, N);
                    passTwo<<<1, 1024>>>(d_partial, d_out, GRID);
                } else if (f == 1) {
                    CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
                    atomicFinal<<<GRID, BS>>>(d_in, d_out, N);
                } else if (f == 2) {
                    lastBlockFinal<<<GRID, BS>>>(d_in, d_partial, d_out, N);
                } else {
                    streamCeiling<<<GRID, BS>>>(d_in, d_partial, N);
                }
            }
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms / ITER < fbest[f]) fbest[f] = ms / ITER;
        }
    }
    CHECK_KERNEL();

    double ceilGBs = (double)N * 4.0 / (fbest[NF - 1] * 1e-3) / 1e9;
    printf("%-30s %9s %9s %9s %9s\n",
           "strategy", "ms", "GB/s", "%ceiling", "launches");
    printf("-----------------------------------------------------------------------\n");
    for (int f = 0; f < NF; ++f) {
        double g = (double)N * 4.0 / (fbest[f] * 1e-3) / 1e9;
        printf("%-30s %9.4f %9.1f %8.1f%% %9s\n", fname[f], fbest[f], g,
               100.0 * g / ceilGBs,
               (f == NF - 1) ? "n/a" : ((f == 0) ? "2" : "1"));
    }
    printf("(ceiling %.1f GB/s = %.1f%% of the 432.0 GB/s pin peak. On this laptop\n"
           " part the same measurement floats between ~373 and ~411 GB/s with the\n"
           " memory P-state; the %%ceiling column is the quantity that reproduces.)\n\n",
           ceilGBs, 100.0 * ceilGBs / 432.0);

    // =====================================================================
    printf("=== B: determinism ===\n");
    unsigned bitsTwo[10], bitsAtom[10], bitsLast[10];
    float r;
    for (int t = 0; t < 10; ++t) {
        passOne<<<GRID, BS>>>(d_in, d_partial, N);
        passTwo<<<1, 1024>>>(d_partial, d_out, GRID);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost)); bitsTwo[t] = bitsOf(r);

        CHECK(cudaMemset(d_out, 0, 4));
        atomicFinal<<<GRID, BS>>>(d_in, d_out, N);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost)); bitsAtom[t] = bitsOf(r);

        lastBlockFinal<<<GRID, BS>>>(d_in, d_partial, d_out, N);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost)); bitsLast[t] = bitsOf(r);
    }
    printf("distinct bit patterns over 10 identical runs, same grid:\n");
    printf("  (a) two kernel launches      : %d   (0x%08x)\n",
           countDistinct(bitsTwo, 10), bitsTwo[0]);
    printf("  (b) atomicAdd per block      : %d\n", countDistinct(bitsAtom, 10));
    for (int t = 0; t < 10; ++t) printf("        run %2d -> 0x%08x  %.6f\n",
                                        t, bitsAtom[t], (double)floatOf(bitsAtom[t]));
    printf("  (c) threadfence + last block : %d   (0x%08x)\n",
           countDistinct(bitsLast, 10), bitsLast[0]);

    printf("\nsame kernel, three DIFFERENT grid sizes (two-kernel strategy):\n");
    int grids[3] = { GRID, GRID / 2, 2 * GRID };
    for (int g = 0; g < 3; ++g) {
        passOne<<<grids[g], BS>>>(d_in, d_partial, N);
        passTwo<<<1, 1024>>>(d_partial, d_out, grids[g]);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost));
        printf("  grid = %5d blocks -> 0x%08x  %.6f\n", grids[g], bitsOf(r), r);
    }
    printf("  A fixed-order tree is reproducible for a FIXED decomposition.\n"
           "  Change the decomposition and you change the summation order.\n\n");

    // =====================================================================
    printf("=== C: floating-point accuracy ===\n");
    printf("%-34s %22s %14s\n", "method", "result", "rel. error");
    printf("----------------------------------------------------------------------\n");

    // --- dataset A: N copies of 1.0f -------------------------------------
    fillOnes<<<GRID, BS>>>(d_in, N);
    CHECK_KERNEL();
    {
        double exact = (double)N;
        float seq = 0.0f;
        for (long long i = 0; i < N; ++i) seq += 1.0f;
        float c = 0.0f, ksum = 0.0f;
        for (long long i = 0; i < N; ++i) {
            float y = 1.0f - c, t = ksum + y;
            c = (t - ksum) - y; ksum = t;
        }
        passOne<<<GRID, BS>>>(d_in, d_partial, N);
        passTwo<<<1, 1024>>>(d_partial, d_out, GRID);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost));
        printf("dataset A: %lld copies of 1.0f, exact = %.1f\n", N, exact);
        printf("%-34s %22.1f %13.3e\n", "  CPU sequential float loop",
               (double)seq, fabs((double)seq - exact) / exact);
        printf("%-34s %22.1f %13.3e\n", "  CPU Kahan compensated float",
               (double)ksum, fabs((double)ksum - exact) / exact);
        printf("%-34s %22.1f %13.3e\n", "  GPU tree reduction (float)",
               (double)r, fabs((double)r - exact) / exact);
    }

    // --- dataset B: dyadic, wide dynamic range ---------------------------
    fillDyadic<<<GRID, BS>>>(d_in, N);
    CHECK_KERNEL();
    {
        double exact = exactDyadic;
        float seq = 0.0f;
        for (long long i = 0; i < N; ++i) seq += h[i];
        float c = 0.0f, ksum = 0.0f;
        for (long long i = 0; i < N; ++i) {
            float y = h[i] - c, t = ksum + y;
            c = (t - ksum) - y; ksum = t;
        }
        passOne<<<GRID, BS>>>(d_in, d_partial, N);
        passTwo<<<1, 1024>>>(d_partial, d_out, GRID);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost));
        printf("\ndataset B: values 2^-10..2^9, exact = %.1f\n", exact);
        printf("%-34s %22.1f %13.3e\n", "  CPU sequential float loop",
               (double)seq, fabs((double)seq - exact) / exact);
        printf("%-34s %22.1f %13.3e\n", "  CPU Kahan compensated float",
               (double)ksum, fabs((double)ksum - exact) / exact);
        printf("%-34s %22.1f %13.3e\n", "  GPU tree reduction (float)",
               (double)r, fabs((double)r - exact) / exact);
    }
    printf("\n  Sequential summation has worst-case error O(N)*eps and, worse,\n"
           "  loses small addends entirely once the accumulator outgrows them.\n"
           "  A tree of depth d has worst-case error O(d)*eps = O(log N)*eps.\n"
           "  The parallel algorithm is not a numerical compromise. It is an\n"
           "  improvement you were going to have to pay for with Kahan anyway.\n\n");

    // =====================================================================
    printf("=== D: three spellings of the warp tail (unsigned) ===\n");
    {
        unsigned long long uexact = 0ull;
        unsigned* hu = (unsigned*)malloc((size_t)N * sizeof(unsigned));
        CHECK(cudaMemcpy(hu, d_iu, (size_t)N * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (long long i = 0; i < N; ++i) uexact += hu[i];
        free(hu);

        const char* dn[3] = { "__shfl_down_sync x5  ",
                              "__reduce_add_sync    ",
                              "cg::reduce(plus<u32>)" };
        double dbest[3] = { 1e30, 1e30, 1e30 };
        unsigned dres[3];
        for (int sweep = 0; sweep < NSW; ++sweep) {
            for (int q = 0; q < 3; ++q) {
                int m = (q + sweep) % 3;
                CHECK(cudaEventRecord(e0));
                for (int k = 0; k < ITER; ++k) {
                    if (m == 0) uintReduce<0><<<GRID, BS>>>(d_iu, d_upart, N);
                    if (m == 1) uintReduce<1><<<GRID, BS>>>(d_iu, d_upart, N);
                    if (m == 2) uintReduce<2><<<GRID, BS>>>(d_iu, d_upart, N);
                }
                CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
                if (ms / ITER < dbest[m]) dbest[m] = ms / ITER;
            }
        }
        for (int m = 0; m < 3; ++m) {
            if (m == 0) uintReduce<0><<<GRID, BS>>>(d_iu, d_upart, N);
            if (m == 1) uintReduce<1><<<GRID, BS>>>(d_iu, d_upart, N);
            if (m == 2) uintReduce<2><<<GRID, BS>>>(d_iu, d_upart, N);
            uintFinal<<<1, 256>>>(d_upart, d_uout, GRID);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(&dres[m], d_uout, 4, cudaMemcpyDeviceToHost));
        }
        printf("exact = %llu\n", uexact);
        printf("%-24s %9s %9s %14s %s\n", "tail", "ms", "GB/s", "result", "");
        for (int m = 0; m < 3; ++m)
            printf("%-24s %9.4f %9.1f %14u %s\n", dn[m], dbest[m],
                   (double)N * 4.0 / (dbest[m] * 1e-3) / 1e9, dres[m],
                   ((unsigned long long)dres[m] == uexact) ? "ok" : "BAD");
        printf("  All three are within noise of each other, because all three are\n"
               "  waiting on DRAM. The tail is ~5 instructions out of the ~2^19\n"
               "  loads a block issues. `cuobjdump -sass` shows what the timer\n"
               "  cannot:\n"
               "    mode 0  -> five SHFL.DOWN + five IADD3\n"
               "    mode 1  -> one REDUX.SUM UR6, R2\n"
               "    mode 2  -> one REDUX.SUM UR6, R2  (cg::reduce lowers to the\n"
               "               hardware instruction by itself on sm_80+ -- but\n"
               "               only for integer types; the float overload has no\n"
               "               hardware form and lowers to five SHFL.BFLY)\n\n");
    }

    // =====================================================================
    printf("=== E: CUB ===\n");
    fillDyadic<<<GRID, BS>>>(d_in, N);
    CHECK_KERNEL();
    {
        void*  d_temp = nullptr;
        size_t tempBytes = 0;
        CHECK(cub::DeviceReduce::Sum(d_temp, tempBytes, d_in, d_out, (int)N));
        CHECK(cudaMalloc(&d_temp, tempBytes));

        double bCub = 1e30, bMine = 1e30, bBlk = 1e30, bCeil = 1e30;
        for (int sweep = 0; sweep < NSW; ++sweep) {
            for (int q = 0; q < 4; ++q) {
                int m = (q + sweep) % 4;
                CHECK(cudaEventRecord(e0));
                for (int k = 0; k < ITER; ++k) {
                    if (m == 0) { passOne<<<GRID, BS>>>(d_in, d_partial, N);
                                  passTwo<<<1, 1024>>>(d_partial, d_out, GRID); }
                    if (m == 1) { cubBlockReduce<<<GRID, BS>>>(d_in, d_partial, N);
                                  passTwo<<<1, 1024>>>(d_partial, d_out, GRID); }
                    if (m == 2) CHECK(cub::DeviceReduce::Sum(d_temp, tempBytes,
                                                             d_in, d_out, (int)N));
                    if (m == 3) streamCeiling<<<GRID, BS>>>(d_in, d_partial, N);
                }
                CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
                float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
                double per = ms / ITER;
                if (m == 0 && per < bMine) bMine = per;
                if (m == 1 && per < bBlk)  bBlk  = per;
                if (m == 2 && per < bCub)  bCub  = per;
                if (m == 3 && per < bCeil) bCeil = per;
            }
        }
        CHECK_KERNEL();
        printf("temp storage cub::DeviceReduce::Sum wants: %zu bytes\n", tempBytes);
        printf("%-34s %9s %9s\n", "implementation", "ms", "GB/s");
        printf("--------------------------------------------------------\n");
        printf("%-34s %9.4f %9.1f\n", "hand-written v6 + finalize", bMine,
               (double)N * 4.0 / (bMine * 1e-3) / 1e9);
        printf("%-34s %9.4f %9.1f\n", "cub::BlockReduce + finalize", bBlk,
               (double)N * 4.0 / (bBlk * 1e-3) / 1e9);
        printf("%-34s %9.4f %9.1f\n", "cub::DeviceReduce::Sum", bCub,
               (double)N * 4.0 / (bCub * 1e-3) / 1e9);

        cubBlockReduce<<<GRID, BS>>>(d_in, d_partial, N);
        passTwo<<<1, 1024>>>(d_partial, d_out, GRID);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost));
        printf("  cub::BlockReduce result %.1f (rel.err %.3e)\n", (double)r,
               fabs((double)r - exactDyadic) / exactDyadic);
        CHECK(cub::DeviceReduce::Sum(d_temp, tempBytes, d_in, d_out, (int)N));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost));
        printf("  cub::DeviceReduce result %.1f (rel.err %.3e)\n", (double)r,
               fabs((double)r - exactDyadic) / exactDyadic);
        CHECK(cudaFree(d_temp));
        printf("  Module 36 covers CUB/Thrust properly. The point here is that\n"
               "  a bandwidth-bound kernel has a hard ceiling, and a library\n"
               "  that reaches it cannot be beaten -- only matched.\n\n");
    }

    // =====================================================================
    printf("=== F: the legacy volatile warp tail (a bug, not a technique) ===\n");
    {
        int mismatches[5];
        for (int t = 0; t < 5; ++t) {
            legacyVolatileTail<<<GRID, BS>>>(d_in, d_partial, N);
            passTwo<<<1, 1024>>>(d_partial, d_out, GRID);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost));
            mismatches[t] = (fabs((double)r - exactDyadic) / exactDyadic > 1e-5);
        }
        int bad = 0; for (int t = 0; t < 5; ++t) bad += mismatches[t];
        printf("  legacy `volatile __shared__` tail: %d of 5 runs wrong, last\n"
               "  result %.1f vs exact %.1f\n", bad, (double)r, exactDyadic);
        printf("  Read Module 8's warning before you draw the wrong conclusion\n"
               "  from that number. The question 'does it produce the right\n"
               "  answer on this chip today' and the question 'is it defined by\n"
               "  the programming model' are different questions, and only the\n"
               "  second one is about your program. It is undefined on sm_70+\n"
               "  either way: `volatile` orders nothing between threads.\n\n");
    }

    free(h);
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_partial)); CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_iu)); CHECK(cudaFree(d_upart)); CHECK(cudaFree(d_uout));
    CHECK(cudaDeviceReset());
    printf("OVERALL: PASS\n");
    return 0;
}
