// =====================================================================
// Module 3 / Example 1 : "Index geometry explorer"
//
// GOAL
//   Make the four built-in variables (threadIdx, blockIdx, blockDim,
//   gridDim) concrete, and -- more importantly -- make the *linearization
//   rule* concrete:
//
//       lane-order within a block =
//           threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)
//
//   x varies fastest. Threads 0..31 in that order are warp 0, 32..63 are
//   warp 1, and so on. Nothing else in CUDA decides warp membership.
//
//   Part A: a 1D launch -- show that the unused dimensions are 1, not 0.
//   Part B: a 2D block (32,8) vs a 2D block (8,32) -- identical thread
//           count, completely different warp shape.
//   Part C: a 3D block -- verify the full 3-term formula.
//
//   Every number this program prints is computed on the GPU from the
//   built-ins and checked on the host against the formula above.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN:    .\example01.exe
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

// One record per thread. We record what the thread *sees*, then check it
// on the host. Nothing here uses shared memory or a warp intrinsic -- the
// lane id comes from the PTX special register %laneid, which is simply
// the hardware answer to "where am I inside my warp?".
struct Rec {
    unsigned tx, ty, tz;      // threadIdx
    unsigned bx, by, bz;      // blockIdx
    unsigned bdx, bdy, bdz;   // blockDim
    unsigned gdx, gdy, gdz;   // gridDim
    unsigned lane;            // %laneid : 0..31, hardware truth
};

__device__ __forceinline__ unsigned laneid()
{
    unsigned r; asm volatile("mov.u32 %0, %%laneid;" : "=r"(r)); return r;
}

// Linear thread id *within the block*. This is THE rule.
__device__ __forceinline__ unsigned tidInBlock()
{
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}
// Linear block id within the grid -- same rule, one level up.
__device__ __forceinline__ unsigned bidInGrid()
{
    return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

__global__ void probe(Rec* out, int threadsPerBlock)
{
    unsigned t = tidInBlock();
    unsigned b = bidInGrid();
    unsigned slot = b * threadsPerBlock + t;

    Rec r;
    r.tx = threadIdx.x; r.ty = threadIdx.y; r.tz = threadIdx.z;
    r.bx = blockIdx.x;  r.by = blockIdx.y;  r.bz = blockIdx.z;
    r.bdx = blockDim.x; r.bdy = blockDim.y; r.bdz = blockDim.z;
    r.gdx = gridDim.x;  r.gdy = gridDim.y;  r.gdz = gridDim.z;
    r.lane = laneid();
    out[slot] = r;
}

// ---------------------------------------------------------------------
static int runProbe(const char* title, dim3 grid, dim3 block, Rec** hostOut)
{
    int tpb   = (int)(block.x * block.y * block.z);
    int total = (int)(tpb * grid.x * grid.y * grid.z);

    Rec* d = nullptr;
    CHECK(cudaMalloc(&d, total * sizeof(Rec)));
    CHECK(cudaMemset(d, 0, total * sizeof(Rec)));

    probe<<<grid, block>>>(d, tpb);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    Rec* h = (Rec*)malloc(total * sizeof(Rec));
    CHECK(cudaMemcpy(h, d, total * sizeof(Rec), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d));

    printf("\n===== %s =====\n", title);
    printf("  launched <<<(%u,%u,%u), (%u,%u,%u)>>>  -> %d threads/block, %d total\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z, tpb, total);
    printf("  kernel reports gridDim =(%u,%u,%u)  blockDim=(%u,%u,%u)\n",
           h[0].gdx, h[0].gdy, h[0].gdz, h[0].bdx, h[0].bdy, h[0].bdz);

    *hostOut = h;
    return total;
}

// Verify: does the linearization formula reproduce the hardware's own
// lane id for every thread? If yes, the formula is not a convention we
// invented -- it is what the SM does.
static int verifyLanes(Rec* h, int total, int tpb)
{
    int bad = 0;
    for (int i = 0; i < total; ++i) {
        unsigned t = h[i].tx + h[i].bdx * (h[i].ty + h[i].bdy * h[i].tz);
        if ((t % 32u) != h[i].lane) ++bad;
        if ((int)t >= tpb) ++bad;
    }
    printf("  linearization vs hardware %%laneid : %s (%d mismatches / %d threads)\n",
           bad == 0 ? "MATCH" : "MISMATCH", bad, total);
    return bad;
}

// Print the (tx,ty,tz) membership of one warp of block 0, in lane order.
static void dumpWarp(Rec* h, int tpb, int warp)
{
    printf("  warp %d of block 0, lanes 0..31 = (threadIdx.x, y, z):\n    ", warp);
    int printed = 0;
    for (int t = warp * 32; t < warp * 32 + 32 && t < tpb; ++t) {
        for (int i = 0; i < tpb; ++i) {
            unsigned lin = h[i].tx + h[i].bdx * (h[i].ty + h[i].bdy * h[i].tz);
            if ((int)lin == t && h[i].bx == 0 && h[i].by == 0 && h[i].bz == 0) {
                printf("(%u,%u,%u)", h[i].tx, h[i].ty, h[i].tz);
                if (++printed % 8 == 0) printf("\n    "); else printf(" ");
                break;
            }
        }
    }
    printf("\n");
}

int main(void)
{
    CHECK(cudaSetDevice(0));
    Rec* h = nullptr;
    int total, tpb;
    int fails = 0;

    // ---------------- Part A: 1D ----------------------------------
    // The unused dimensions are 1. Not 0. If they were 0 the product
    // blockDim.x*blockDim.y*blockDim.z would be 0 and the linearization
    // formula would collapse -- CUDA guarantees 1 so the formula is
    // dimension-agnostic and you can always write the full 3-term version.
    tpb   = 256;
    total = runProbe("Part A -- 1D grid, 1D block", dim3(3), dim3(256), &h);
    printf("  threadIdx.y of thread 0 = %u, threadIdx.z = %u   (both 0)\n", h[0].ty, h[0].tz);
    printf("  blockDim.y  = %u, blockDim.z  = %u   (both 1, NOT 0)\n", h[0].bdy, h[0].bdz);
    printf("  gridDim.y   = %u, gridDim.z   = %u   (both 1, NOT 0)\n", h[0].gdy, h[0].gdz);
    fails += verifyLanes(h, total, tpb);
    free(h);

    // ---------------- Part B1: block (32,8) -----------------------
    // 256 threads. blockDim.x == 32 == warp size, so warp w is exactly
    // the row threadIdx.y == w. Each warp covers 32 consecutive x values.
    tpb   = 32 * 8;
    total = runProbe("Part B1 -- 2D block (32,8): warps are rows", dim3(1), dim3(32, 8), &h);
    fails += verifyLanes(h, total, tpb);
    dumpWarp(h, tpb, 1);
    free(h);

    // ---------------- Part B2: block (8,32) -----------------------
    // Same 256 threads. blockDim.x == 8, so one warp spans 4 consecutive
    // threadIdx.y values. If x indexes the fast axis of your array, this
    // warp touches 4 disjoint rows -> 4x the memory transactions.
    // Module 5 turns that sentence into sector counts.
    tpb   = 8 * 32;
    total = runProbe("Part B2 -- 2D block (8,32): warps straddle 4 rows", dim3(1), dim3(8, 32), &h);
    fails += verifyLanes(h, total, tpb);
    dumpWarp(h, tpb, 1);
    free(h);

    // ---------------- Part C: 3D ----------------------------------
    // (4,4,4) = 64 threads = 2 warps. Warp 0 is z=0 and z=1 in full.
    tpb   = 4 * 4 * 4;
    total = runProbe("Part C -- 3D block (4,4,4)", dim3(2, 2, 2), dim3(4, 4, 4), &h);
    fails += verifyLanes(h, total, tpb);
    dumpWarp(h, tpb, 0);
    free(h);

    // Note on %warpid (deliberately not used above): it is the SM warp
    // *slot*, which the PTX ISA documents as not a stable identity. Derive
    // warp membership from tidInBlock()/32, never from %warpid.
    printf("\nOverall linearization check: %s\n", fails == 0 ? "PASS" : "FAIL");

    CHECK(cudaDeviceReset());
    return fails == 0 ? 0 : 1;
}
