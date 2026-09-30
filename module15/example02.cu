// ============================================================================
// Module 15 / example02.cu -- Tile shape, block shape, and partition camping
//
// GOAL : example01 fixed the tile at 32x32 and the block at (32,8) and asked
//        what the shared-memory layout should be. This file asks the three
//        questions example01 assumed the answer to:
//
//        A. What tile size and block shape? Why is a 32x32 tile not handled by
//           a 32x32 block? Sweep (TILE, BROWS) and read the answer off.
//        B. Is PARTITION CAMPING -- the historically famous transpose
//           pathology, where the blocks running at any instant all target the
//           same DRAM partition / L2 slice -- observable on Ada? Sweep the
//           matrix size so the inter-tile stride is alternately a large power
//           of two and not, and test the classic diagonal-reordering fix.
//        C. What do the boundary guards cost, and what does a matrix whose
//           dimensions are not multiples of the tile cost?
//
// BUILD: nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN  : .\example02.exe
//
// Timing follows AUTHORING_SPEC section 12 throughout: 1500 ms duration-based
// warm-up, back-to-back timing with ROTATED order and SWEEPS >= NCFG in every
// comparison group, min-of-N, validation in a separate pass, ratios reported
// as the stable quantity.
// ============================================================================

#include <cstdio>
#include <cstdlib>
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

__host__ __device__ __forceinline__ float srcValue(long long linearIndex)
{
    unsigned h = (unsigned)linearIndex * 2654435761u;
    h ^= h >> 13;
    h *= 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0x00FFFFFFu);
}

// ---------------------------------------------------------------------------
// The generic padded tiled transpose. TILE is the square tile edge; BROWS is
// the block's y extent, so each thread moves TILE/BROWS rows of the tile.
// The pad of 1 makes the shared pitch TILE+1, which is odd whenever TILE is
// even, and gcd(odd, 32) == 1 is M7's condition for a conflict-free column.
// ---------------------------------------------------------------------------
template <int TILE, int BROWS>
__global__ void tTiled(const float* __restrict__ in, float* __restrict__ out,
                       int W, int H)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
#pragma unroll
    for (int j = 0; j < TILE; j += BROWS)
        if (x < W && y + j < H)
            tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * W + x];

    __syncthreads();

    int xo = blockIdx.y * TILE + threadIdx.x;
    int yo = blockIdx.x * TILE + threadIdx.y;
#pragma unroll
    for (int j = 0; j < TILE; j += BROWS)
        if (xo < H && yo + j < W)
            out[(long long)(yo + j) * H + xo] = tile[threadIdx.x][threadIdx.y + j];
}

// Unguarded twin: legal only when TILE divides both W and H. Used in Part C to
// price the bounds tests, which M3 measured as predication rather than
// branching.
template <int TILE, int BROWS>
__global__ void tTiledNoGuard(const float* __restrict__ in, float* __restrict__ out,
                              int W, int H)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
#pragma unroll
    for (int j = 0; j < TILE; j += BROWS)
        tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * W + x];

    __syncthreads();

    int xo = blockIdx.y * TILE + threadIdx.x;
    int yo = blockIdx.x * TILE + threadIdx.y;
#pragma unroll
    for (int j = 0; j < TILE; j += BROWS)
        out[(long long)(yo + j) * H + xo] = tile[threadIdx.x][threadIdx.y + j];
}

// Diagonal block reordering: same grid, different blockIdx -> tile mapping.
template <int TILE, int BROWS>
__global__ void tDiagonal(const float* __restrict__ in, float* __restrict__ out,
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
#pragma unroll
    for (int j = 0; j < TILE; j += BROWS)
        if (x < W && y + j < H)
            tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * W + x];

    __syncthreads();

    int xo = by * TILE + threadIdx.x;
    int yo = bx * TILE + threadIdx.y;
#pragma unroll
    for (int j = 0; j < TILE; j += BROWS)
        if (xo < H && yo + j < W)
            out[(long long)(yo + j) * H + xo] = tile[threadIdx.x][threadIdx.y + j];
}

// The ceiling: a copy with the same tiling, loads hoisted ahead of stores.
template <int TILE, int BROWS>
__global__ void tCopy(const float* __restrict__ in, float* __restrict__ out,
                      int W, int H)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    float v[TILE / BROWS];
#pragma unroll
    for (int k = 0; k < TILE / BROWS; ++k) {
        int j = k * BROWS;
        v[k] = (x < W && y + j < H) ? in[(long long)(y + j) * W + x] : 0.0f;
    }
#pragma unroll
    for (int k = 0; k < TILE / BROWS; ++k) {
        int j = k * BROWS;
        if (x < W && y + j < H) out[(long long)(y + j) * W + x] = v[k];
    }
}

// ---------------------------------------------------------------------------
// Part B kernels. Same tile, same work, but the OUTPUT leading dimension is a
// free parameter, so the inter-tile output stride (32 * ldOut * 4 bytes) can be
// moved off a power of two without changing a single byte of useful traffic.
// This is M5's row pitch and M10's address spreading, applied to the question
// "does the DRAM partition / L2 slice that a tile lands in matter?"
// ---------------------------------------------------------------------------
__global__ void campCopy(const float* __restrict__ in, float* __restrict__ out,
                         int N, int ldOut)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = in[(long long)(y + k * 8) * N + x];
#pragma unroll
    for (int k = 0; k < 4; ++k) out[(long long)(y + k * 8) * ldOut + x] = v[k];
}

__global__ void campTiled(const float* __restrict__ in, float* __restrict__ out,
                          int N, int ldOut)
{
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
#pragma unroll
    for (int j = 0; j < 32; j += 8)
        tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * N + x];
    __syncthreads();
    int xo = blockIdx.y * 32 + threadIdx.x;
    int yo = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
    for (int j = 0; j < 32; j += 8)
        out[(long long)(yo + j) * ldOut + xo] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void campDiag(const float* __restrict__ in, float* __restrict__ out,
                         int N, int ldOut)
{
    __shared__ float tile[32][33];
    int by = blockIdx.x;
    int bx = (blockIdx.x + blockIdx.y) % gridDim.x;
    int x = bx * 32 + threadIdx.x;
    int y = by * 32 + threadIdx.y;
#pragma unroll
    for (int j = 0; j < 32; j += 8)
        tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * N + x];
    __syncthreads();
    int xo = by * 32 + threadIdx.x;
    int yo = bx * 32 + threadIdx.y;
#pragma unroll
    for (int j = 0; j < 32; j += 8)
        out[(long long)(yo + j) * ldOut + xo] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void checkTransposeLd(const float* __restrict__ out, int N, int ldOut,
                                 unsigned* bad)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    long long n = (long long)N * N;
    unsigned local = 0;
    for (; i < n; i += s) {
        int c = (int)(i % N), r = (int)(i / N);
        if (out[(long long)r * ldOut + c] != srcValue((long long)c * N + r)) ++local;
    }
    if (local) atomicAdd(bad, local);
}

// ---------------------------------------------------------------------------
__global__ void fillSource(float* a, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    for (; i < n; i += s) a[i] = srcValue(i);
}
__global__ void poison(float* a, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    for (; i < n; i += s) a[i] = -1.0f;
}
__global__ void checkTranspose(const float* __restrict__ out, int W, int H,
                               unsigned* bad)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    long long n = (long long)W * H;
    unsigned local = 0;
    for (; i < n; i += s) {
        int c = (int)(i % H), r = (int)(i / H);
        if (out[i] != srcValue((long long)c * W + r)) ++local;
    }
    if (local) atomicAdd(bad, local);
}

// ---------------------------------------------------------------------------
// Part A: the shape sweep. One entry per (TILE, BROWS) pair, plus a copy.
// ---------------------------------------------------------------------------
static const int NSHAPE = 10;
struct Shape { const char* name; int tile, brows; };
static const Shape shapes[NSHAPE] = {
    {"copy ceiling, block (32,8)     ", 32,  8},   // index 0 is the copy
    {"tile 16x16, block (16,16) 256 t", 16, 16},
    {"tile 16x16, block (16, 8) 128 t", 16,  8},
    {"tile 32x32, block (32,32)1024 t", 32, 32},
    {"tile 32x32, block (32,16) 512 t", 32, 16},
    {"tile 32x32, block (32, 8) 256 t", 32,  8},
    {"tile 32x32, block (32, 4) 128 t", 32,  4},
    {"tile 64x64, block (64,16)1024 t", 64, 16},
    {"tile 64x64, block (64, 8) 512 t", 64,  8},
    {"copy ceiling, block (64,16)    ", 64, 16},   // matched ceiling for tile 64
};

static void launchShape(int s, const float* in, float* out, int W, int H)
{
    switch (s) {
        case 0: { dim3 b(32, 8), g((W + 31) / 32, (H + 31) / 32);
                  tCopy<32, 8><<<g, b>>>(in, out, W, H); break; }
        case 1: { dim3 b(16, 16), g((W + 15) / 16, (H + 15) / 16);
                  tTiled<16, 16><<<g, b>>>(in, out, W, H); break; }
        case 2: { dim3 b(16, 8), g((W + 15) / 16, (H + 15) / 16);
                  tTiled<16, 8><<<g, b>>>(in, out, W, H); break; }
        case 3: { dim3 b(32, 32), g((W + 31) / 32, (H + 31) / 32);
                  tTiled<32, 32><<<g, b>>>(in, out, W, H); break; }
        case 4: { dim3 b(32, 16), g((W + 31) / 32, (H + 31) / 32);
                  tTiled<32, 16><<<g, b>>>(in, out, W, H); break; }
        case 5: { dim3 b(32, 8), g((W + 31) / 32, (H + 31) / 32);
                  tTiled<32, 8><<<g, b>>>(in, out, W, H); break; }
        case 6: { dim3 b(32, 4), g((W + 31) / 32, (H + 31) / 32);
                  tTiled<32, 4><<<g, b>>>(in, out, W, H); break; }
        case 7: { dim3 b(64, 16), g((W + 63) / 64, (H + 63) / 64);
                  tTiled<64, 16><<<g, b>>>(in, out, W, H); break; }
        case 8: { dim3 b(64, 8), g((W + 63) / 64, (H + 63) / 64);
                  tTiled<64, 8><<<g, b>>>(in, out, W, H); break; }
        case 9: { dim3 b(64, 16), g((W + 63) / 64, (H + 63) / 64);
                  tCopy<64, 16><<<g, b>>>(in, out, W, H); break; }
        default: break;
    }
}

// ---------------------------------------------------------------------------
// Part B: partition camping. Three kernels per matrix size.
// ---------------------------------------------------------------------------
static void launchCamp(int k, const float* in, float* out, int W, int H)
{
    dim3 b(32, 8), g((W + 31) / 32, (H + 31) / 32);
    switch (k) {
        case 0: tCopy<32, 8>   <<<g, b>>>(in, out, W, H); break;
        case 1: tTiled<32, 8>  <<<g, b>>>(in, out, W, H); break;
        case 2: tDiagonal<32,8><<<g, b>>>(in, out, W, H); break;
        default: break;
    }
}

// ---------------------------------------------------------------------------
static double timeGroup(void (*launch)(int, const float*, float*, int, int),
                        int cfg, int ncfg, int nsweep, int iters,
                        const float* in, float* out, int W, int H,
                        cudaEvent_t a, cudaEvent_t b)
{
    // One configuration of a group; the caller loops with rotation.
    (void)ncfg; (void)nsweep;
    CHECK(cudaEventRecord(a));
    for (int k = 0; k < iters; ++k) launch(cfg, in, out, W, H);
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms / iters;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Module 15 example 02 -- tile shape, block shape, partition camping\n");
    printf("GPU: %s, CC %d.%d, L2 = %.1f MB\n\n",
           prop.name, prop.major, prop.minor, (double)prop.l2CacheSize / 1.0e6);

    // Largest matrix used anywhere below decides the allocation.
    const int MAXN = 8320;
    const long long nMax = (long long)MAXN * MAXN;
    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  (size_t)nMax * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)nMax * sizeof(float)));
    unsigned* d_bad = nullptr;
    CHECK(cudaMalloc(&d_bad, sizeof(unsigned)));

    cudaEvent_t evA, evB;
    CHECK(cudaEventCreate(&evA));
    CHECK(cudaEventCreate(&evB));

    const int N = 8192;
    const long long nN = (long long)N * N;
    fillSource<<<2048, 256>>>(d_in, nMax);
    CHECK_KERNEL();

    // ---- 1500 ms duration-based warm-up (spec 12 rule 4) ------------------
    {
        float acc = 0.0f;
        CHECK(cudaEventRecord(evA));
        while (acc < 1500.0f) {
            for (int k = 0; k < 10; ++k) launchShape(0, d_in, d_out, N, N);
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            CHECK(cudaEventElapsedTime(&acc, evA, evB));
        }
        CHECK_KERNEL();
    }

    // ======================= PART A -- shape sweep =========================
    {
        double best[NSHAPE];
        for (int s = 0; s < NSHAPE; ++s) best[s] = 1e30;
        const int ITERS = 20;
        for (int sweep = 0; sweep < NSHAPE; ++sweep)          // SWEEPS >= NCFG
            for (int q = 0; q < NSHAPE; ++q) {
                int s = (q + sweep) % NSHAPE;
                double per = timeGroup(launchShape, s, NSHAPE, NSHAPE, ITERS,
                                       d_in, d_out, N, N, evA, evB);
                if (per < best[s]) best[s] = per;
            }
        CHECK_KERNEL();

        const double bytes2N = 2.0 * (double)nN * 4.0;
        printf("PART A -- tile and block shape at %d x %d (min of %d rotated sweeps)\n",
               N, N, NSHAPE);
        printf("%-34s %8s %6s %9s %9s %8s\n",
               "configuration", "smem B", "rows", "ms", "GB/s", "%ofcopy");
        for (int s = 0; s < NSHAPE; ++s) {
            int t = shapes[s].tile, br = shapes[s].brows;
            int smem = (s == 0) ? 0 : t * (t + 1) * (int)sizeof(float);
            double gbs = bytes2N / (best[s] * 1.0e-3) / 1.0e9;
            printf("%-34s %8d %6d %9.4f %9.1f %7.1f%%\n",
                   shapes[s].name, smem, t / br, best[s], gbs,
                   100.0 * best[0] / best[s]);
        }
        printf("\n");

        // occupancy the API reports, for the three 32x32 block shapes
        int bpsm = 0;
        printf("  blocks/SM from cudaOccupancyMaxActiveBlocksPerMultiprocessor:\n");
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, tTiled<32, 32>, 1024, 0));
        printf("    tile 32, block (32,32) 1024 thr, 4224 B smem : %d blocks/SM = %d warps\n",
               bpsm, bpsm * 32);
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, tTiled<32, 8>, 256, 0));
        printf("    tile 32, block (32, 8)  256 thr, 4224 B smem : %d blocks/SM = %d warps\n",
               bpsm, bpsm * 8);
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, tTiled<64, 16>, 1024, 0));
        printf("    tile 64, block (64,16) 1024 thr,16640 B smem : %d blocks/SM = %d warps\n\n",
               bpsm, bpsm * 32);
    }

    // =================== PART B -- partition camping =======================
    // The classic pathology: the blocks resident at any instant march along one
    // row of the grid, and on the OUTPUT side consecutive blocks in that row are
    // 32 * ldOut * 4 bytes apart. If the DRAM partition (or L2 slice) is chosen
    // by a few address bits, a stride that is a large power of two sends every
    // one of them to the same partition.
    //
    // The clean experiment holds the matrix, the tile, the grid and every
    // useful byte FIXED and varies only the output leading dimension. Any
    // difference is therefore attributable to where the addresses land, not to
    // how much work was done. A copy with the same ldOut is timed alongside as
    // the control: if the copy moves too, the effect is not about transposing.
    {
        const int N2 = 8192;
        const long long n2 = (long long)N2 * N2;
        fillSource<<<2048, 256>>>(d_in, n2);
        CHECK_KERNEL();
        const int lds[4] = { 8192, 8200, 8224, 8320 };
        printf("PART B -- is partition camping observable on Ada?\n");
        printf("  matrix fixed at %d x %d; only the output leading dimension moves.\n",
               N2, N2);
        printf("%8s %14s %10s %11s %9s %10s %11s\n",
               "ldOut", "tile stride B", "copy GB/s", "tiled GB/s", "%ofcopy",
               "diag GB/s", "diag/tiled");
        dim3 b2(32, 8), g2(N2 / 32, N2 / 32);
        for (int li = 0; li < 4; ++li) {
            int ld = lds[li];
            double best[3] = { 1e30, 1e30, 1e30 };
            for (int k = 0; k < 10; ++k) campTiled<<<g2, b2>>>(d_in, d_out, N2, ld);
            CHECK_KERNEL();
            for (int sweep = 0; sweep < 3; ++sweep)
                for (int q = 0; q < 3; ++q) {
                    int c = (q + sweep) % 3;
                    CHECK(cudaEventRecord(evA));
                    for (int k = 0; k < 20; ++k) {
                        if (c == 0)      campCopy <<<g2, b2>>>(d_in, d_out, N2, ld);
                        else if (c == 1) campTiled<<<g2, b2>>>(d_in, d_out, N2, ld);
                        else             campDiag <<<g2, b2>>>(d_in, d_out, N2, ld);
                    }
                    CHECK(cudaEventRecord(evB));
                    CHECK(cudaEventSynchronize(evB));
                    float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evA, evB));
                    if (ms / 20.0 < best[c]) best[c] = ms / 20.0;
                }
            CHECK_KERNEL();
            double bb = 2.0 * (double)n2 * 4.0;
            printf("%8d %14lld %10.1f %11.1f %8.1f%% %10.1f %10.3fx\n",
                   ld, (long long)32 * ld * 4,
                   bb / (best[0] * 1e-3) / 1e9, bb / (best[1] * 1e-3) / 1e9,
                   100.0 * best[0] / best[1], bb / (best[2] * 1e-3) / 1e9,
                   best[2] / best[1]);
            // correctness of the padded-output variant, checked once per ld
            unsigned zero = 0, bad = 0;
            CHECK(cudaMemcpy(d_bad, &zero, sizeof(unsigned), cudaMemcpyHostToDevice));
            campTiled<<<g2, b2>>>(d_in, d_out, N2, ld);
            CHECK_KERNEL();
            checkTransposeLd<<<2048, 256>>>(d_out, N2, ld, d_bad);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(&bad, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
            if (bad) printf("    !! ldOut %d produced %u mismatches\n", ld, bad);
        }
        printf("\n");
    }

    // ================= PART C -- boundary handling =========================
    {
        fillSource<<<2048, 256>>>(d_in, nMax);
        CHECK_KERNEL();
        dim3 b(32, 8), g(N / 32, N / 32);
        double bg = 1e30, bn = 1e30;
        for (int k = 0; k < 20; ++k) tTiled<32, 8><<<g, b>>>(d_in, d_out, N, N);
        CHECK_KERNEL();
        for (int sweep = 0; sweep < 2; ++sweep)
            for (int q = 0; q < 2; ++q) {
                int c = (q + sweep) % 2;
                CHECK(cudaEventRecord(evA));
                for (int k = 0; k < 20; ++k) {
                    if (c == 0) tTiled<32, 8>       <<<g, b>>>(d_in, d_out, N, N);
                    else        tTiledNoGuard<32, 8><<<g, b>>>(d_in, d_out, N, N);
                }
                CHECK(cudaEventRecord(evB));
                CHECK(cudaEventSynchronize(evB));
                float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evA, evB));
                if (c == 0) { if (ms / 20.0 < bg) bg = ms / 20.0; }
                else        { if (ms / 20.0 < bn) bn = ms / 20.0; }
            }
        printf("PART C -- what the boundary guards cost\n");
        printf("  guarded   (4 predicated tests per phase) : %.4f ms\n", bg);
        printf("  unguarded (legal only when 32 | W, H)    : %.4f ms\n", bn);
        printf("  guarded / unguarded                      : %.3fx\n\n", bg / bn);

        // A large matrix that is NOT a multiple of the tile: 8191 x 8193.
        const int RW = 8191, RH = 8193;
        long long nR = (long long)RW * RH;
        fillSource<<<2048, 256>>>(d_in, nR);
        CHECK_KERNEL();
        dim3 gR((RW + 31) / 32, (RH + 31) / 32);
        double br = 1e30;
        for (int k = 0; k < 20; ++k) tTiled<32, 8><<<gR, b>>>(d_in, d_out, RW, RH);
        CHECK_KERNEL();
        for (int sweep = 0; sweep < 3; ++sweep) {
            CHECK(cudaEventRecord(evA));
            for (int k = 0; k < 20; ++k) tTiled<32, 8><<<gR, b>>>(d_in, d_out, RW, RH);
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evA, evB));
            if (ms / 20.0 < br) br = ms / 20.0;
        }
        double gbsR = 2.0 * (double)nR * 4.0 / (br * 1e-3) / 1e9;
        printf("  %d x %d (neither dimension a multiple of 32): %.4f ms, %.1f GB/s\n",
               RW, RH, br, gbsR);
        printf("  partial tiles: %d of %d blocks touch a boundary (%.1f%%)\n\n",
               (int)(gR.x + gR.y - 1), (int)(gR.x * gR.y),
               100.0 * (gR.x + gR.y - 1) / (double)(gR.x * gR.y));
    }

    // ==================== VALIDATION (separate pass) =======================
    int fails = 0;
    printf("VALIDATION -- every shape, on a square multiple and on 4093 x 2049\n");
    {
        const int cases = 2;
        const int Ws[cases] = { 1024, 4093 };
        const int Hs[cases] = { 1024, 2049 };
        for (int c = 0; c < cases; ++c) {
            long long n = (long long)Ws[c] * Hs[c];
            fillSource<<<512, 256>>>(d_in, n);
            CHECK_KERNEL();
            for (int s = 1; s < NSHAPE - 1; ++s) {
                poison<<<512, 256>>>(d_out, n);
                CHECK_KERNEL();
                launchShape(s, d_in, d_out, Ws[c], Hs[c]);
                CHECK_KERNEL();
                unsigned zero = 0, bad = 0;
                CHECK(cudaMemcpy(d_bad, &zero, sizeof(unsigned), cudaMemcpyHostToDevice));
                checkTranspose<<<512, 256>>>(d_out, Ws[c], Hs[c], d_bad);
                CHECK_KERNEL();
                CHECK(cudaMemcpy(&bad, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
                printf("  %-34s %4dx%-4d %s (%u)\n", shapes[s].name, Ws[c], Hs[c],
                       bad ? "FAIL" : "ok  ", bad);
                if (bad) ++fails;
            }
            // diagonal variant too
            poison<<<512, 256>>>(d_out, n);
            CHECK_KERNEL();
            launchCamp(2, d_in, d_out, Ws[c], Hs[c]);
            CHECK_KERNEL();
            unsigned zero = 0, bad = 0;
            CHECK(cudaMemcpy(d_bad, &zero, sizeof(unsigned), cudaMemcpyHostToDevice));
            checkTranspose<<<512, 256>>>(d_out, Ws[c], Hs[c], d_bad);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(&bad, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
            printf("  %-34s %4dx%-4d %s (%u)\n", "diagonal reorder, tile 32     ",
                   Ws[c], Hs[c], bad ? "FAIL" : "ok  ", bad);
            if (bad) ++fails;
        }
    }
    printf("\n");

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_bad));
    CHECK(cudaEventDestroy(evA));
    CHECK(cudaEventDestroy(evB));
    CHECK(cudaDeviceReset());

    printf("OVERALL: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
