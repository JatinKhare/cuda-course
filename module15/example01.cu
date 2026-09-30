// ============================================================================
// Module 15 / example01.cu -- The transpose ladder, measured against a copy
//
// GOAL : Transpose has zero arithmetic. Every microsecond it costs is memory
//        behaviour. So the first thing this file measures is NOT a transpose:
//        it is a plain COPY of the same matrix with the same block shape and
//        the same instruction count. That copy is the speed of light for the
//        problem -- 2N traffic, one read and one write of every element, both
//        perfectly coalesced. Every transpose below is reported as a fraction
//        of it.
//
//        Ladder:
//          0  linear 1-D float4 copy            (pure streaming reference)
//          1  2-D tiled copy, loads hoisted     (THE CEILING for this ladder)
//          2  2-D tiled copy, load/store interleaved  (an MLP contrast, M11)
//          3  naive, coalesced read  / strided write
//          4  naive, strided read    / coalesced write
//          5  shared-memory tiled, 32x32 tile   (32-way bank conflict, M7)
//          6  tiled + padded [32][33]
//          7  tiled + XOR swizzle   (zero extra shared memory)
//          8  tiled + padded + diagonal block reordering (partition camping)
//          9  shared-staged COPY -- the tile round trip with no transposition,
//             so the cost of staging is separated from the cost of permuting
//
// BUILD: nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN  : .\example01.exe
//
// SASS : nvcc -arch=sm_89 -O3 -c -o example01.o example01.cu
//        cuobjdump -sass example01.o > example01.sass
//        (compare the LDS/STS pair in transposeTiled<0> and transposeTiled<1>;
//         look for the LOP3 that implements the XOR in transposeSwizzle)
//
// Timing follows AUTHORING_SPEC section 12: 1500 ms duration-based warm-up
// (400 ms ramps the SM clock but NOT the memory P-state), iteration counts
// auto-scaled toward ~10 ms segments, all configurations timed back to back in
// one loop with ROTATED order, SWEEPS >= NCFG, min-of-N, validation in a
// separate second pass.
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

// ---------------------------------------------------------------------------
// Problem sizes.
//   BIG  : 8192 x 8192 floats = 256 MiB per buffer, 512 MiB touched per call.
//          Far beyond the 48 MB L2, and an exact multiple of 32 so the timed
//          ladder contains no boundary arithmetic.
//   RECT : 4093 x 2049 -- neither square nor a multiple of the tile. Used for
//          CORRECTNESS only; at 33.5 MB it is L2-resident and any bandwidth
//          figure taken from it would not be a DRAM measurement.
// ---------------------------------------------------------------------------
static const int BIG_W = 8192, BIG_H = 8192;
static const int RECT_W = 4093, RECT_H = 2049;

static const int TILE  = 32;
static const int BROWS = 8;                  // block is (32, 8): 4 rows/thread
static const int NREG  = TILE / BROWS;       // registers per thread in a copy

static const int NCFG   = 10;
static const int NSWEEP = 10;                // spec 12.9: SWEEPS >= NCFG
static const int CEIL   = 1;                 // index of the matched 2-D copy

// Deterministic, index-derived data. Values are exact in float (< 2^24) and
// distinct enough that a wrong permutation cannot alias into a pass.
__host__ __device__ __forceinline__ float srcValue(long long linearIndex)
{
    unsigned h = (unsigned)linearIndex * 2654435761u;
    h ^= h >> 13;
    h *= 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0x00FFFFFFu);
}

// ---------------------------------------------------------------------------
// 0 -- a plain 1-D streaming copy, float4, grid-stride. No 2-D structure at
//      all. This is the number Modules 11 and 12 call the streaming ceiling,
//      and it exists here to price the 2-D traversal itself.
// ---------------------------------------------------------------------------
__global__ void linearCopy(const float* __restrict__ in, float* __restrict__ out,
                           long long n)
{
    const long long n4 = n / 4;
    const float4* in4  = reinterpret_cast<const float4*>(in);
    float4*       out4 = reinterpret_cast<float4*>(out);
    long long step = (long long)blockDim.x * gridDim.x;
    long long i    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n4; i += step) out4[i] = in4[i];
    // exactly-once tail, outside the vector loop (M5 / M11)
    for (long long t = n4 * 4 + (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < n; t += step)
        out[t] = in[t];
}

// ---------------------------------------------------------------------------
// 1 -- THE CEILING. Same block shape, same tile, same guard, same number of
//      global instructions as every tiled transpose below. All four loads are
//      issued before any store, exactly as the tiled kernels are forced to do
//      by their barrier, so the comparison is not contaminated by a difference
//      in memory-level parallelism (M11).
// ---------------------------------------------------------------------------
__global__ void copyHoisted(const float* __restrict__ in, float* __restrict__ out,
                            int W, int H)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    float v[NREG];
#pragma unroll
    for (int k = 0; k < NREG; ++k) {
        int j = k * BROWS;
        v[k] = (x < W && y + j < H) ? in[(long long)(y + j) * W + x] : 0.0f;
    }
#pragma unroll
    for (int k = 0; k < NREG; ++k) {
        int j = k * BROWS;
        if (x < W && y + j < H) out[(long long)(y + j) * W + x] = v[k];
    }
}

// ---------------------------------------------------------------------------
// 2 -- the same copy with the load and the store of each row adjacent. Same
//      traffic, same sectors, fewer independent loads in flight at once.
// ---------------------------------------------------------------------------
__global__ void copyInterleaved(const float* __restrict__ in, float* __restrict__ out,
                                int W, int H)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS)
        if (x < W && y + j < H)
            out[(long long)(y + j) * W + x] = in[(long long)(y + j) * W + x];
}

// ---------------------------------------------------------------------------
// 3 -- naive, coalesced read / strided write.
//      Read  in[y*W + x]  : lane L supplies x = base+L   -> 4 sectors per warp.
//      Write out[x*H + y] : lane L supplies (base+L)*H+y -> 32 sectors.
// ---------------------------------------------------------------------------
__global__ void naiveCoalRead(const float* __restrict__ in, float* __restrict__ out,
                              int W, int H)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS)
        if (x < W && y + j < H)
            out[(long long)x * H + (y + j)] = in[(long long)(y + j) * W + x];
}

// ---------------------------------------------------------------------------
// 4 -- naive, strided read / coalesced write. The SAME permutation, indexed
//      from the output side instead of the input side.
//      Here (xo, yo) names an element of the OUTPUT, which is H wide, W tall.
// ---------------------------------------------------------------------------
__global__ void naiveCoalWrite(const float* __restrict__ in, float* __restrict__ out,
                               int W, int H)
{
    int xo = blockIdx.x * TILE + threadIdx.x;      // column of out, in [0,H)
    int yo = blockIdx.y * TILE + threadIdx.y;      // row    of out, in [0,W)
    for (int j = 0; j < TILE; j += BROWS)
        if (xo < H && yo + j < W)
            out[(long long)(yo + j) * H + xo] = in[(long long)xo * W + (yo + j)];
}

// ---------------------------------------------------------------------------
// 5 / 6 -- shared-memory tiled. PAD = 0 gives the textbook 32-way column
//          conflict; PAD = 1 gives the [32][33] fix.
//
// The two shared-memory index expressions are NOT symmetric, and that asymmetry
// is the whole difficulty of the kernel:
//     load  phase: tile[threadIdx.y + j][threadIdx.x]   <- tile-local
//     store phase: tile[threadIdx.x][threadIdx.y + j]   <- transposed
// and the GLOBAL indices swap the roles of blockIdx.x and blockIdx.y too.
// ---------------------------------------------------------------------------
template <int PAD>
__global__ void transposeTiled(const float* __restrict__ in, float* __restrict__ out,
                               int W, int H)
{
    __shared__ float tile[TILE][TILE + PAD];

    int x = blockIdx.x * TILE + threadIdx.x;       // column of in
    int y = blockIdx.y * TILE + threadIdx.y;       // row    of in
    for (int j = 0; j < TILE; j += BROWS)
        if (x < W && y + j < H)
            tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * W + x];

    __syncthreads();                               // Module 9 made this precise

    int xo = blockIdx.y * TILE + threadIdx.x;      // column of out, in [0,H)
    int yo = blockIdx.x * TILE + threadIdx.y;      // row    of out, in [0,W)
    for (int j = 0; j < TILE; j += BROWS)
        if (xo < H && yo + j < W)
            out[(long long)(yo + j) * H + xo] = tile[threadIdx.x][threadIdx.y + j];
}

// ---------------------------------------------------------------------------
// 7 -- XOR swizzle. Exactly TILE*TILE floats, not one word more.
//      element (r, c) lives at r*TILE + (c ^ (r & (TILE-1))).
//      Both phases land one word per bank: the store-to-tile phase has r fixed
//      within a warp and c = 0..31, the load-from-tile phase has c fixed and
//      r = 0..31, and XOR by a constant is a bijection on [0,32) either way.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int swz(int r, int c)
{
    return r * TILE + (c ^ (r & (TILE - 1)));
}

__global__ void transposeSwizzle(const float* __restrict__ in, float* __restrict__ out,
                                 int W, int H)
{
    __shared__ float tile[TILE * TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS)
        if (x < W && y + j < H)
            tile[swz(threadIdx.y + j, threadIdx.x)] = in[(long long)(y + j) * W + x];

    __syncthreads();

    int xo = blockIdx.y * TILE + threadIdx.x;
    int yo = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS)
        if (xo < H && yo + j < W)
            out[(long long)(yo + j) * H + xo] = tile[swz(threadIdx.x, threadIdx.y + j)];
}

// ---------------------------------------------------------------------------
// 8 -- padded tile PLUS diagonal block reordering, the classic partition
//      camping mitigation. The launch geometry is unchanged; only the mapping
//      from blockIdx to tile coordinates is rewritten, so that the set of
//      blocks running concurrently is spread across tile rows AND columns
//      rather than marching along one row of tiles at a time.
// ---------------------------------------------------------------------------
__global__ void transposeDiagonal(const float* __restrict__ in, float* __restrict__ out,
                                  int W, int H)
{
    __shared__ float tile[TILE][TILE + 1];

    int bx, by;
    if (gridDim.x == gridDim.y) {
        by = blockIdx.x;
        bx = (blockIdx.x + blockIdx.y) % gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x * blockIdx.y;
        by = bid % gridDim.y;
        bx = ((bid / gridDim.y) + by) % gridDim.x;
    }

    int x = bx * TILE + threadIdx.x;
    int y = by * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS)
        if (x < W && y + j < H)
            tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * W + x];

    __syncthreads();

    int xo = by * TILE + threadIdx.x;
    int yo = bx * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS)
        if (xo < H && yo + j < W)
            out[(long long)(yo + j) * H + xo] = tile[threadIdx.x][threadIdx.y + j];
}

// ---------------------------------------------------------------------------
// 9 -- the tile round trip WITHOUT the transposition. Identical shared-memory
//      traffic, identical barrier, identical instruction count to kernel 6 --
//      but both global accesses are the copy's. Whatever this costs above
//      kernel 1 is the price of staging; whatever kernel 6 costs above this is
//      the price of permuting.
// ---------------------------------------------------------------------------
__global__ void stagedCopy(const float* __restrict__ in, float* __restrict__ out,
                           int W, int H)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS)
        if (x < W && y + j < H)
            tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * W + x];

    __syncthreads();

    for (int j = 0; j < TILE; j += BROWS)
        if (x < W && y + j < H)
            out[(long long)(y + j) * W + x] = tile[threadIdx.y + j][threadIdx.x];
}

// ---------------------------------------------------------------------------
// Validation, on the device so that 256 MiB host buffers are unnecessary.
// out has H columns and W rows; out[r][c] must equal in[c][r] = srcValue(c*W+r).
// ---------------------------------------------------------------------------
__global__ void checkTranspose(const float* __restrict__ out, int W, int H,
                               unsigned* bad)
{
    long long i    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)blockDim.x * gridDim.x;
    long long n    = (long long)W * H;
    unsigned local = 0;
    for (; i < n; i += step) {
        int c = (int)(i % H);          // column of out
        int r = (int)(i / H);          // row    of out
        if (out[i] != srcValue((long long)c * W + r)) ++local;
    }
    if (local) atomicAdd(bad, local);
}

__global__ void checkCopy(const float* __restrict__ out, int W, int H, unsigned* bad)
{
    long long i    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)blockDim.x * gridDim.x;
    long long n    = (long long)W * H;
    unsigned local = 0;
    for (; i < n; i += step)
        if (out[i] != srcValue(i)) ++local;
    if (local) atomicAdd(bad, local);
}

__global__ void fillSource(float* a, long long n)
{
    long long i    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)blockDim.x * gridDim.x;
    for (; i < n; i += step) a[i] = srcValue(i);
}

__global__ void poison(float* a, long long n)
{
    long long i    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)blockDim.x * gridDim.x;
    for (; i < n; i += step) a[i] = -1.0f;
}

// ---------------------------------------------------------------------------
static const char* cfgName[NCFG] = {
    "0 linear 1-D float4 copy (stream) ",
    "1 2-D tiled copy, hoisted (CEILING)",
    "2 2-D tiled copy, interleaved ld/st",
    "3 naive  coalesced rd / strided wr",
    "4 naive  strided rd / coalesced wr",
    "5 tiled  [32][32]  (D=32 conflict)",
    "6 tiled  [32][33]  padded         ",
    "7 tiled  XOR swizzle              ",
    "8 tiled  padded + diagonal blocks ",
    "9 shared-staged copy (no permute) ",
};

// true for the kernels whose output is the transpose; false for the copies
static const bool cfgIsTranspose[NCFG] = { false, false, false, true, true,
                                           true, true, true, true, false };

static void launchCfg(int v, const float* d_in, float* d_out, int W, int H)
{
    // Configuration 4 is indexed from the OUTPUT side, so its grid is sized to
    // the output (H wide, W tall). Everything else is sized to the input.
    dim3 blk(TILE, BROWS);
    dim3 gIn((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    dim3 gOut((H + TILE - 1) / TILE, (W + TILE - 1) / TILE);
    switch (v) {
        case 0: {
            long long n = (long long)W * H;
            long long g = (n / 4 + 255) / 256;
            int grid = (g > 65535) ? 65535 : (int)(g < 1 ? 1 : g);
            linearCopy<<<grid, 256>>>(d_in, d_out, n);
            break;
        }
        case 1: copyHoisted    <<<gIn,  blk>>>(d_in, d_out, W, H); break;
        case 2: copyInterleaved<<<gIn,  blk>>>(d_in, d_out, W, H); break;
        case 3: naiveCoalRead  <<<gIn,  blk>>>(d_in, d_out, W, H); break;
        case 4: naiveCoalWrite <<<gOut, blk>>>(d_in, d_out, W, H); break;
        case 5: transposeTiled<0><<<gIn, blk>>>(d_in, d_out, W, H); break;
        case 6: transposeTiled<1><<<gIn, blk>>>(d_in, d_out, W, H); break;
        case 7: transposeSwizzle<<<gIn,  blk>>>(d_in, d_out, W, H); break;
        case 8: transposeDiagonal<<<gIn, blk>>>(d_in, d_out, W, H); break;
        case 9: stagedCopy     <<<gIn,  blk>>>(d_in, d_out, W, H); break;
        default: break;
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int smCount = 0;
    CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));

    printf("Module 15 example 01 -- the transpose ladder\n");
    printf("GPU: %s, %d SMs, CC %d.%d, L2 = %.1f MB\n",
           prop.name, smCount, prop.major, prop.minor,
           (double)prop.l2CacheSize / 1.0e6);

    const long long nBig = (long long)BIG_W * BIG_H;
    printf("Timed matrix : %d x %d floats = %.0f MiB per buffer, %.0f MiB moved "
           "per call (2N)\n", BIG_W, BIG_H, (double)nBig * 4.0 / 1048576.0,
           2.0 * (double)nBig * 4.0 / 1048576.0);
    printf("Block shape  : (%d, %d) -- each thread moves %d rows of a %dx%d tile\n\n",
           TILE, BROWS, TILE / BROWS, TILE, TILE);

    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  (size_t)nBig * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)nBig * sizeof(float)));
    fillSource<<<2048, 256>>>(d_in, nBig);
    CHECK_KERNEL();

    unsigned* d_bad = nullptr;
    CHECK(cudaMalloc(&d_bad, sizeof(unsigned)));

    cudaEvent_t evStart, evStop;
    CHECK(cudaEventCreate(&evStart));
    CHECK(cudaEventCreate(&evStop));

    // ---- warm-up: 1500 ms, duration based ---------------------------------
    // Spec section 12 rule 4. A 400 ms warm-up ramps the SM clock but not the
    // memory P-state and understates every absolute bandwidth here by ~10%.
    {
        cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        float acc = 0.0f;
        CHECK(cudaEventRecord(a));
        while (acc < 1500.0f) {
            for (int k = 0; k < 10; ++k) launchCfg(CEIL, d_in, d_out, BIG_W, BIG_H);
            CHECK(cudaEventRecord(b));
            CHECK(cudaEventSynchronize(b));
            CHECK(cudaEventElapsedTime(&acc, a, b));
        }
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
        CHECK_KERNEL();
    }

    // ---- auto-scale iteration counts (>= 20, spec section 5) --------------
    int iters[NCFG];
    for (int v = 0; v < NCFG; ++v) {
        launchCfg(v, d_in, d_out, BIG_W, BIG_H);
        CHECK_KERNEL();
        CHECK(cudaEventRecord(evStart));
        for (int k = 0; k < 3; ++k) launchCfg(v, d_in, d_out, BIG_W, BIG_H);
        CHECK(cudaEventRecord(evStop));
        CHECK(cudaEventSynchronize(evStop));
        float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        double per = ms / 3.0;
        int it = (int)(10.0 / (per > 1e-6 ? per : 1e-6));
        if (it < 20) it = 20;
        if (it > 400) it = 400;
        iters[v] = it;
    }

    // ---- TIMING PASS ------------------------------------------------------
    double best[NCFG];
    for (int v = 0; v < NCFG; ++v) best[v] = 1e30;
    for (int sweep = 0; sweep < NSWEEP; ++sweep) {
        for (int q = 0; q < NCFG; ++q) {
            int v = (q + sweep) % NCFG;                  // spec section 12 rule 9
            CHECK(cudaEventRecord(evStart));
            for (int k = 0; k < iters[v]; ++k) launchCfg(v, d_in, d_out, BIG_W, BIG_H);
            CHECK(cudaEventRecord(evStop));
            CHECK(cudaEventSynchronize(evStop));
            float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
            double per = ms / iters[v];
            if (per < best[v]) best[v] = per;
        }
    }
    CHECK_KERNEL();

    const double bytes2N = 2.0 * (double)nBig * 4.0;
    const double ceilMs  = best[CEIL];

    printf("PART A -- the ladder at %d x %d  (min of %d sweeps, rotated order)\n",
           BIG_W, BIG_H, NSWEEP);
    printf("%-36s %9s %10s %9s %9s\n",
           "kernel", "ms", "GB/s(2N)", "%ofcopy", "%of432");
    for (int v = 0; v < NCFG; ++v) {
        double gbs = bytes2N / (best[v] * 1.0e-3) / 1.0e9;
        printf("%-36s %9.4f %10.1f %8.1f%% %8.1f%%\n",
               cfgName[v], best[v], gbs, 100.0 * ceilMs / best[v],
               100.0 * gbs / 432.0);
    }
    printf("\n");

    // ---- VALIDATION PASS (separate, untimed) ------------------------------
    printf("PART B -- validation, square %dx%d and rectangular %dx%d\n",
           BIG_W, BIG_H, RECT_W, RECT_H);

    int fails = 0;
    for (int v = 0; v < NCFG; ++v) {
        poison<<<2048, 256>>>(d_out, nBig);
        CHECK_KERNEL();
        launchCfg(v, d_in, d_out, BIG_W, BIG_H);
        CHECK_KERNEL();
        unsigned zero = 0, bad = 0;
        CHECK(cudaMemcpy(d_bad, &zero, sizeof(unsigned), cudaMemcpyHostToDevice));
        if (cfgIsTranspose[v]) checkTranspose<<<2048, 256>>>(d_out, BIG_W, BIG_H, d_bad);
        else                   checkCopy     <<<2048, 256>>>(d_out, BIG_W, BIG_H, d_bad);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(&bad, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
        printf("  %-36s square   %s (%u mismatches)\n",
               cfgName[v], bad ? "FAIL" : "ok  ", bad);
        if (bad) ++fails;
    }

    // Rectangular, non-multiple-of-tile. Same buffers, smaller footprint.
    {
        const long long nR = (long long)RECT_W * RECT_H;
        fillSource<<<1024, 256>>>(d_in, nR);
        CHECK_KERNEL();
        for (int v = 0; v < NCFG; ++v) {
            poison<<<1024, 256>>>(d_out, nR);
            CHECK_KERNEL();
            launchCfg(v, d_in, d_out, RECT_W, RECT_H);
            CHECK_KERNEL();
            unsigned zero = 0, bad = 0;
            CHECK(cudaMemcpy(d_bad, &zero, sizeof(unsigned), cudaMemcpyHostToDevice));
            if (cfgIsTranspose[v]) checkTranspose<<<1024, 256>>>(d_out, RECT_W, RECT_H, d_bad);
            else                   checkCopy     <<<1024, 256>>>(d_out, RECT_W, RECT_H, d_bad);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(&bad, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
            printf("  %-36s rect     %s (%u mismatches)\n",
                   cfgName[v], bad ? "FAIL" : "ok  ", bad);
            if (bad) ++fails;
        }
    }
    printf("\n");

    // ---- PART C -- the read/write asymmetry, isolated ---------------------
    // Configurations 3 and 4 move identical useful bytes and differ only in
    // which side of the kernel is strided. Module 11 measured write-allocate at
    // 3.95x for partial-sector stores; this is that effect inside an algorithm.
    printf("PART C -- which side should you leave uncoalesced?\n");
    printf("  strided WRITE (cfg 3) : %9.4f ms   %.1f%% of copy\n",
           best[3], 100.0 * ceilMs / best[3]);
    printf("  strided READ  (cfg 4) : %9.4f ms   %.1f%% of copy\n",
           best[4], 100.0 * ceilMs / best[4]);
    printf("  ratio write/read      : %.3fx  (>1 means the strided write is worse)\n\n",
           best[3] / best[4]);

    printf("PART D -- decomposing the ceiling, and the two conflict fixes\n");
    printf("  2-D tiled copy / 1-D linear copy : %.3fx  (cost of the 2-D traversal)\n",
           best[CEIL] / best[0]);
    printf("  interleaved / hoisted copy       : %.3fx  (memory-level parallelism)\n",
           best[2] / best[CEIL]);
    printf("  staged copy / hoisted copy       : %.3fx  (cost of the tile round trip)\n",
           best[9] / best[CEIL]);
    printf("  padded transpose / staged copy   : %.3fx  (cost of the permutation)\n",
           best[6] / best[9]);
    printf("  shared bytes/block, [32][32] : %d\n", (int)(32 * 32 * sizeof(float)));
    printf("  shared bytes/block, [32][33] : %d\n", (int)(32 * 33 * sizeof(float)));
    printf("  conflicted / padded  : %.3fx  (M7 measured 14.8x on an LSU-bound kernel)\n",
           best[5] / best[6]);
    printf("  swizzle   / padded   : %.3fx  (<1 means the swizzle is faster)\n",
           best[7] / best[6]);
    printf("  diagonal  / padded   : %.3fx  (<1 means reordering helped)\n\n",
           best[8] / best[6]);

    // ---- PART E -- the same three tiled kernels with DRAM taken away ------
    // At 8192x8192 the bank conflict is invisible because DRAM is the binding
    // constraint (Module 12's lesson: you cannot rank optimizations without
    // knowing which resource is saturated). Shrink the matrix until both
    // buffers fit in the 50 MB L2 and the shared-memory term stops hiding.
    // NOTHING in this part is a DRAM bandwidth number.
    {
        const int SMALL = 2048;                      // 16 MiB per buffer
        const long long nS = (long long)SMALL * SMALL;
        fillSource<<<1024, 256>>>(d_in, nS);
        CHECK_KERNEL();

        const int ids[3] = { 5, 6, 7 };              // conflicted, padded, swizzle
        double bs[3] = { 1e30, 1e30, 1e30 };
        for (int k = 0; k < 40; ++k) launchCfg(6, d_in, d_out, SMALL, SMALL);
        CHECK_KERNEL();
        for (int sweep = 0; sweep < 3; ++sweep) {
            for (int q = 0; q < 3; ++q) {
                int j = (q + sweep) % 3;
                CHECK(cudaEventRecord(evStart));
                for (int k = 0; k < 100; ++k) launchCfg(ids[j], d_in, d_out, SMALL, SMALL);
                CHECK(cudaEventRecord(evStop));
                CHECK(cudaEventSynchronize(evStop));
                float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
                if (ms / 100.0 < bs[j]) bs[j] = ms / 100.0;
            }
        }
        printf("PART E -- L2-RESIDENT %dx%d (16 MiB/buffer). NOT a DRAM number.\n",
               SMALL, SMALL);
        double b2 = 2.0 * (double)nS * 4.0;
        for (int j = 0; j < 3; ++j)
            printf("  %-36s %9.4f ms  %8.1f GB/s(apparent)  %.3fx of padded\n",
                   cfgName[ids[j]], bs[j], b2 / (bs[j] * 1.0e-3) / 1.0e9, bs[j] / bs[1]);
        printf("\n");
    }

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_bad));
    CHECK(cudaEventDestroy(evStart));
    CHECK(cudaEventDestroy(evStop));
    CHECK(cudaDeviceReset());

    printf("OVERALL: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
