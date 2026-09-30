// ============================================================================
// Module 15 / exercise01.cu -- Build the transpose ladder
//
// GOAL : Starting from a measured COPY of the same matrix -- the speed of light
//        for a kernel with zero arithmetic and 2N compulsory traffic -- build
//        three transposes and report each as a fraction of that copy:
//
//          v1  naive, coalesced read / strided write        (you write it)
//          v2  naive, strided read / coalesced write        (given)
//          v3  shared-memory tiled, plain 32x32 tile        (you write it)
//          v4  shared-memory tiled, conflict-free layout    (you design it)
//
//        Every version is validated on BOTH a square power-of-two matrix and a
//        rectangular one whose dimensions are not multiples of the tile. A
//        transpose that is right on the first and wrong on the second is the
//        single most common outcome of this exercise; the harness reports the
//        two separately so you can see which one bit you.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN  : .\exercise01.exe
//
// Optional: nvcc -arch=sm_89 -O3 -Xptxas -v -c -o exercise01.o exercise01.cu
//           cuobjdump -sass exercise01.o | findstr "LDS STS"
//
// Timing follows AUTHORING_SPEC section 12: 1500 ms warm-up, all versions timed
// back to back in one rotated sweep, SWEEPS >= NCFG, min-of-N, validation in a
// separate pass. Report ratios; the absolute ms on this laptop part moves by
// up to 2.5x with thermal and power state.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
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

static const int TILE  = 32;
static const int BROWS = 8;          // block is (32, 8): each thread moves 4 rows
static const int MAXSH = TILE * (TILE + 1);   // you may use at most this many floats

static const int BIG_W = 8192, BIG_H = 8192;      // timed (256 MiB per buffer)
static const int SQ_N  = 2048;                    // validated: square, power of two
static const int RC_W  = 4093, RC_H = 2049;       // validated: neither
static const int SMALL = 2048;                    // L2-resident reveal at the end

static const int NCFG   = 5;         // copy + v1..v4
static const int NSWEEP = 8;         // spec 12.9: SWEEPS >= NCFG (min-of-8)

__host__ __device__ __forceinline__ float srcValue(long long i)
{
    unsigned h = (unsigned)i * 2654435761u;
    h ^= h >> 13;
    h *= 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0x00FFFFFFu);
}

// ===========================================================================
// TODO 4 (PREDICTION) -- fill this in BEFORE you compile anything.
//
// Buckets, as a percentage of the measured copy:
//      1 = below 20%      2 = 20% to 85%      3 = above 85%
//
// Why the boundaries sit where they do. The two naive transposes are the
// least reproducible numbers in this program: a fully strided access is the
// most power-hungry kernel here, so it is the one that loses the most when
// the part is power limited. Measured over many runs on this GPU, v1 lands
// anywhere in 34-64% of the copy and v2 in 72-76%, while v3 and v4 sit at
// 94-101%. The boundaries are placed in the two widest empty gaps in that
// distribution, so no version is within 8 points of a boundary and no bucket
// is decided by the thermal state the run happened to start in. The price is
// that v1 and v2 share a bucket -- the table above still prints both ratios,
// and the difference between them is the point of TODO 1, but it is not a
// difference this machine can score reproducibly.
//
// PRED[0] : v1  naive, coalesced read / strided write
// PRED[1] : v2  naive, strided read / coalesced write
// PRED[2] : v3  tiled, plain 32x32 shared tile
// PRED[3] : v4  tiled, conflict-free shared layout
//
// PRED[4] is a separate question: how much is removing the 32-way bank
// conflict worth on THIS kernel, at 8192x8192?
//      1 = less than 1.25x        2 = 1.25x or more
//
// Leave any entry at 0 and the program will tell you to fill it in.
// ===========================================================================
static const int PRED[5] = { 0, 0, 0, 0, 0 };
// YOUR CODE HERE (replace the five zeros)

// ---------------------------------------------------------------------------
// The ceiling. Given. Same tile, same block, same guards, same instruction
// count as the tiled transposes, with the loads hoisted ahead of the stores so
// the comparison is not contaminated by a difference in memory-level
// parallelism (Module 11).
// ---------------------------------------------------------------------------
__global__ void copyCeiling(const float* __restrict__ in, float* __restrict__ out,
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
// v1 -- naive, coalesced read / strided write.
//
// `in`  is H rows of W floats.   `out` is W rows of H floats.
// (x, y) below names an element of the INPUT: x is its column, y its row.
// The grid is sized to the input.
// ---------------------------------------------------------------------------
__global__ void v1_naive(const float* __restrict__ in, float* __restrict__ out,
                         int W, int H)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS) {
        if (x < W && y + j < H) {
            // TODO 1: one statement. Read the element at input row (y+j),
            // input column x, and store it where the transpose puts it.
            // Both index expressions must be written in terms of x, y+j, W, H.
            // Before you write it, work out with Module 5's procedure how many
            // 32 B sectors ONE WARP touches on each side of this assignment.
            long long src = 0;   // YOUR CODE HERE
            long long dst = 0;   // YOUR CODE HERE
            out[dst] = in[src];
        }
    }
}

// ---------------------------------------------------------------------------
// v2 -- naive, strided read / coalesced write. GIVEN, so that you can measure
//       the asymmetry rather than write it. Note that this kernel is indexed
//       from the OUTPUT side, so its grid is sized to the output: (xo, yo)
//       names an element of `out`, which is H wide and W tall.
// ---------------------------------------------------------------------------
__global__ void v2_naive(const float* __restrict__ in, float* __restrict__ out,
                         int W, int H)
{
    int xo = blockIdx.x * TILE + threadIdx.x;
    int yo = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS)
        if (xo < H && yo + j < W)
            out[(long long)(yo + j) * H + xo] = in[(long long)xo * W + (yo + j)];
}

// ---------------------------------------------------------------------------
// v3 -- shared-memory tiled, plain 32x32 tile.
//
// The block loads a TILE x TILE patch of the input with coalesced reads, and
// writes a TILE x TILE patch of the output with coalesced writes. Between the
// two, the transposition happens inside shared memory, where "contiguous"
// carries no privilege (Module 6, job 2).
//
// Warning before you start: the two shared index expressions are NOT mirror
// images of each other, and the output's global index does not use the same
// block coordinates as the input's. Getting either wrong produces a kernel
// that runs, faults nothing, and passes on a square matrix.
// ---------------------------------------------------------------------------
__global__ void v3_tiled(const float* __restrict__ in, float* __restrict__ out,
                         int W, int H)
{
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;      // column of in
    int y = blockIdx.y * TILE + threadIdx.y;      // row    of in

    for (int j = 0; j < TILE; j += BROWS) {
        if (x < W && y + j < H) {
            // TODO 2a: stage the element at input row (y+j), column x into the
            // tile. Index the tile in the coordinates of the INPUT patch.
            tile[0][0] = in[(long long)(y + j) * W + x];   // YOUR CODE HERE
        }
    }

    // TODO 2b: one statement. Every thread of the block must be able to read
    // cells that other threads of the block wrote. Module 9 made the guarantee
    // you need precise; name the mechanism yourself.
    // YOUR CODE HERE

    // TODO 2c: the output coordinates. `out` has H columns and W rows, and the
    // patch this block produces is NOT at (blockIdx.x, blockIdx.y) of the
    // output grid. Write xo (a column of out) and yo (a row of out).
    int xo = 0;   // YOUR CODE HERE
    int yo = 0;   // YOUR CODE HERE

    for (int j = 0; j < TILE; j += BROWS) {
        if (xo < H && yo + j < W) {
            // TODO 2d: read the staged element that belongs at output row
            // (yo+j), column xo, and store it. The tile subscripts here are
            // not the ones you used in TODO 2a with the roles swapped by
            // accident -- work out which tile cell this thread must read.
            out[(long long)(yo + j) * H + xo] = tile[0][0];   // YOUR CODE HERE
        }
    }
}

// ===========================================================================
// TODO 3 (DESIGN) -- a shared-memory layout with no bank conflicts.
//
// v3's tile is a 32 x 32 float array. The store phase walks a row of it and
// the load phase walks a column. One of those two is a 32-way conflict by
// Module 7's rule. Design a layout that makes BOTH phases conflict-free.
//
// Implement the map from a logical tile cell (r, c), with 0 <= r, c < 32, to
// an offset in a flat shared array, and state how many floats that array needs.
// Constraints the harness enforces:
//   * SH_FLOATS <= 32*33 = 1056
//   * shIdx must be injective on [0,32) x [0,32) and land inside [0, SH_FLOATS)
//   * the store phase (r fixed across a warp, c = 0..31) must have degree <= 2
//   * the load  phase (c fixed across a warp, r = 0..31) must have degree <= 2
// The degree bound is 2 and not 1 on purpose: Module 7 measured the cost law on
// Ada as max(2, D), so a 2-way conflict is free and there is no reason to
// forbid it.
//
// Return -1 from shIdx (the shipped state) and the harness will skip v4 and
// tell you so. Do not name the technique in a comment; there is more than one
// answer that satisfies the constraints, and they do not cost the same.
// ===========================================================================
static const int SH_FLOATS = 0;     // YOUR CODE HERE (must be > 0 and <= 1056)

__host__ __device__ __forceinline__ int shIdx(int r, int c)
{
    (void)r; (void)c;
    return -1;                      // YOUR CODE HERE
}

__global__ void v4_tiled(const float* __restrict__ in, float* __restrict__ out,
                         int W, int H)
{
    __shared__ float tile[MAXSH];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS)
        if (x < W && y + j < H)
            tile[shIdx(threadIdx.y + j, threadIdx.x)] = in[(long long)(y + j) * W + x];

    __syncthreads();

    int xo = blockIdx.y * TILE + threadIdx.x;
    int yo = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BROWS)
        if (xo < H && yo + j < W)
            out[(long long)(yo + j) * H + xo] = tile[shIdx(threadIdx.x, threadIdx.y + j)];
}

// ---------------------------------------------------------------------------
// Harness plumbing. Nothing below here needs to change.
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
    for (; i < n; i += s) a[i] = -7.0f;
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
__global__ void checkCopy(const float* __restrict__ out, long long n, unsigned* bad)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    unsigned local = 0;
    for (; i < n; i += s) if (out[i] != srcValue(i)) ++local;
    if (local) atomicAdd(bad, local);
}

static const char* vName[NCFG] = {
    "copy ceiling            ",
    "v1 coal read / str write",
    "v2 str read / coal write",
    "v3 tiled, plain 32x32   ",
    "v4 tiled, your layout   ",
};

// Warm-up companion (AUTHORING_SPEC 12.4 corollary): a pure-arithmetic kernel
// with no memory traffic. A streaming warm-up alone ramps the memory P-state
// but lets the power manager park the SM clock; this pulls it back up.
__global__ void computeWarm(float* sink, int rounds)
{
    float x = (float)threadIdx.x * 1.0e-3f + 1.0f;
    float y = 0.5f;
#pragma unroll 1
    for (int i = 0; i < rounds; ++i) {
        x = fmaf(x, 0.999f, 0.001f);
        y = fmaf(y, x, 0.5f);
    }
    if (x == -1.0e30f) sink[0] = x + y;   // never true; keeps the loop alive
}

static void launchV(int v, const float* in, float* out, int W, int H)
{
    dim3 blk(TILE, BROWS);
    dim3 gIn((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    dim3 gOut((H + TILE - 1) / TILE, (W + TILE - 1) / TILE);
    switch (v) {
        case 0: copyCeiling<<<gIn,  blk>>>(in, out, W, H); break;
        case 1: v1_naive   <<<gIn,  blk>>>(in, out, W, H); break;
        case 2: v2_naive   <<<gOut, blk>>>(in, out, W, H); break;
        case 3: v3_tiled   <<<gIn,  blk>>>(in, out, W, H); break;
        case 4: v4_tiled   <<<gIn,  blk>>>(in, out, W, H); break;
        default: break;
    }
}

// Host-side simulation of the conflict degree of a layout, phase by phase.
// Exactly Module 7's procedure: bucket DISTINCT WORDS per bank, take the max.
static int degreeOfPhase(bool storePhase)
{
    int worst = 0;
    for (int fixed = 0; fixed < TILE; ++fixed) {
        int words[32][64], nw[32];
        for (int b = 0; b < 32; ++b) nw[b] = 0;
        for (int lane = 0; lane < 32; ++lane) {
            int off = storePhase ? shIdx(fixed, lane) : shIdx(lane, fixed);
            if (off < 0) return 99;
            int bank = off % 32;
            bool seen = false;
            for (int k = 0; k < nw[bank]; ++k)
                if (words[bank][k] == off) { seen = true; break; }
            if (!seen && nw[bank] < 64) words[bank][nw[bank]++] = off;
        }
        for (int b = 0; b < 32; ++b) if (nw[b] > worst) worst = nw[b];
    }
    return worst;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    for (int i = 0; i < 5; ++i)
        if (PRED[i] == 0) { printf("Set TODO 4 (PREDICTION) first.\n"); return 0; }
    for (int i = 0; i < 4; ++i)
        if (PRED[i] < 1 || PRED[i] > 3) { printf("PRED[%d] must be 1, 2 or 3.\n", i); return 0; }
    if (PRED[4] < 1 || PRED[4] > 2) { printf("PRED[4] must be 1 or 2.\n"); return 0; }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Module 15 exercise 01 -- build the transpose ladder\n");
    printf("GPU: %s, CC %d.%d, L2 = %.1f MB\n\n",
           prop.name, prop.major, prop.minor, (double)prop.l2CacheSize / 1.0e6);

    // ---- structural check of TODO 3, before anything is launched ----------
    bool v4ok = false;
    int dStore = 99, dLoad = 99;
    if (SH_FLOATS > 0 && SH_FLOATS <= MAXSH && shIdx(0, 0) >= 0) {
        bool inRange = true, injective = true;
        static char used[32 * 33];
        for (int i = 0; i < MAXSH; ++i) used[i] = 0;
        for (int r = 0; r < TILE && inRange && injective; ++r)
            for (int c = 0; c < TILE; ++c) {
                int o = shIdx(r, c);
                if (o < 0 || o >= SH_FLOATS) { inRange = false; break; }
                if (used[o]) { injective = false; break; }
                used[o] = 1;
            }
        dStore = degreeOfPhase(true);
        dLoad  = degreeOfPhase(false);
        printf("TODO 3 structural check: SH_FLOATS = %d (limit %d), in range %s, "
               "injective %s\n", SH_FLOATS, MAXSH, inRange ? "yes" : "NO",
               injective ? "yes" : "NO");
        printf("  store phase (row across a warp)  degree = %d  (need <= 2)\n", dStore);
        printf("  load  phase (column across warp) degree = %d  (need <= 2)\n", dLoad);
        v4ok = inRange && injective && dStore <= 2 && dLoad <= 2;
    } else {
        printf("TODO 3 not set: v4 will be skipped. (shIdx returns -1 or "
               "SH_FLOATS is out of range.)\n");
    }
    printf("  for reference, v3's plain 32x32 tile: store degree 1, load degree 32\n\n");

    const long long nBig = (long long)BIG_W * BIG_H;
    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  (size_t)nBig * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)nBig * sizeof(float)));
    unsigned* d_bad = nullptr;
    CHECK(cudaMalloc(&d_bad, sizeof(unsigned)));
    fillSource<<<2048, 256>>>(d_in, nBig);
    CHECK_KERNEL();

    cudaEvent_t evA, evB;
    CHECK(cudaEventCreate(&evA));
    CHECK(cudaEventCreate(&evB));

    // ---- warm-up, and a check that it reached the normal operating point --
    // 1500 ms streaming THEN 500 ms compute is the spec 12.4 recipe: the
    // streaming phase ramps the memory P-state, the compute phase pulls the SM
    // clock back up (a streaming-only warm-up leaves it parked low, and the
    // kernels here are not all limited by the same resource, so a half-warmed
    // part moves them by different amounts and the RATIOS this exercise scores
    // are exactly what goes unstable).
    //
    // On this laptop part that is necessary but not sufficient. After a few
    // minutes of back-to-back benchmarking the power manager pins the memory
    // clock down -- measured at 6001 MHz instead of 9001, with SW_POWER_CAP and
    // SW_THERMAL_SLOWDOWN both asserted -- and in THAT state the ratios are not
    // this GPU's ratios at all. Warming harder cannot fix it; only idling can.
    // So: warm, probe the copy ceiling, and if it is far below what this GPU
    // does when healthy, idle and warm again.
    const double CEIL_FLOOR_GBPS = 200.0;   // healthy ~255, power-capped ~118
    for (int attempt = 0; ; ++attempt) {
        float acc = 0.0f;
        CHECK(cudaEventRecord(evA));
        while (acc < 1500.0f) {
            for (int k = 0; k < 10; ++k) launchV(0, d_in, d_out, BIG_W, BIG_H);
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            CHECK(cudaEventElapsedTime(&acc, evA, evB));
        }
        acc = 0.0f;
        CHECK(cudaEventRecord(evA));
        while (acc < 500.0f) {
            for (int k = 0; k < 4; ++k) computeWarm<<<320, 256>>>(d_out, 100000);
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            CHECK(cudaEventElapsedTime(&acc, evA, evB));
        }
        CHECK_KERNEL();

        CHECK(cudaEventRecord(evA));
        for (int k = 0; k < 8; ++k) launchV(0, d_in, d_out, BIG_W, BIG_H);
        CHECK(cudaEventRecord(evB));
        CHECK(cudaEventSynchronize(evB));
        float pms = 0.0f; CHECK(cudaEventElapsedTime(&pms, evA, evB));
        double gbps = 2.0 * (double)nBig * 4.0
                    / (((double)pms / 8.0) * 1.0e-3) / 1.0e9;
        if (gbps >= CEIL_FLOOR_GBPS) {
            printf("warm-up complete: copy ceiling probes at %.1f GB/s\n\n", gbps);
            break;
        }
        if (attempt >= 5) {
            printf("WARNING: the copy ceiling still probes at only %.1f GB/s after "
                   "%d cool-downs.\n         This part is power limited right now. "
                   "The ratios below are real,\n         but they are not this "
                   "GPU's healthy ratios -- let it idle and re-run.\n\n",
                   gbps, attempt);
            break;
        }
        printf("warm-up: copy ceiling probes at only %.1f GB/s (healthy is >= %.0f)"
               " --\n         this part is power or thermally limited. Idling 10 s"
               " and warming again (%d/5).\n", gbps, CEIL_FLOOR_GBPS, attempt + 1);
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    // ---- auto-scale the iteration counts to ~10 ms per segment (spec 12.12) -
    int iters[NCFG];
    for (int v = 0; v < NCFG; ++v) {
        iters[v] = 20;
        if (v == 4 && !v4ok) continue;
        CHECK(cudaEventRecord(evA));
        for (int k = 0; k < 4; ++k) launchV(v, d_in, d_out, BIG_W, BIG_H);
        CHECK(cudaEventRecord(evB));
        CHECK(cudaEventSynchronize(evB));
        float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evA, evB));
        double per = (double)ms / 4.0;
        int n = (per > 1.0e-6) ? (int)(10.0 / per + 0.5) : 20;
        iters[v] = (n < 5) ? 5 : (n > 400 ? 400 : n);
    }
    CHECK_KERNEL();

    // ---- TIMING PASS: rotated order, min of NSWEEP ------------------------
    double best[NCFG];
    for (int v = 0; v < NCFG; ++v) best[v] = 1e30;
    for (int sweep = 0; sweep < NSWEEP; ++sweep)
        for (int q = 0; q < NCFG; ++q) {
            int v = (q + sweep) % NCFG;
            if (v == 4 && !v4ok) continue;
            CHECK(cudaEventRecord(evA));
            for (int k = 0; k < iters[v]; ++k) launchV(v, d_in, d_out, BIG_W, BIG_H);
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evA, evB));
            double per = (double)ms / (double)iters[v];
            if (per < best[v]) best[v] = per;
        }
    CHECK_KERNEL();

    const double bytes2N = 2.0 * (double)nBig * 4.0;
    printf("TIMING at %d x %d (min of %d rotated sweeps)\n", BIG_W, BIG_H, NSWEEP);
    printf("%-26s %9s %9s %9s %9s\n", "version", "ms", "GB/s", "%ofcopy", "bucket");
    int measuredBucket[NCFG];
    for (int v = 0; v < NCFG; ++v) {
        if (v == 4 && !v4ok) { measuredBucket[v] = 0;
            printf("%-26s %9s %9s %9s %9s\n", vName[v], "-", "-", "-", "-"); continue; }
        double frac = 100.0 * best[0] / best[v];
        measuredBucket[v] = (frac < 20.0) ? 1 : (frac < 85.0 ? 2 : 3);
        printf("%-26s %9.4f %9.1f %8.1f%% %9d\n", vName[v], best[v],
               bytes2N / (best[v] * 1.0e-3) / 1.0e9, frac, measuredBucket[v]);
    }
    printf("\n");

    // ---- VALIDATION PASS: square power of two, then rectangular -----------
    int nWrong = 0;
    printf("VALIDATION\n");
    printf("%-26s %14s %14s\n", "version", "2048x2048", "4093x2049");
    {
        const int Ws[2] = { SQ_N, RC_W };
        const int Hs[2] = { SQ_N, RC_H };
        unsigned res[NCFG][2];
        for (int c = 0; c < 2; ++c) {
            long long n = (long long)Ws[c] * Hs[c];
            fillSource<<<1024, 256>>>(d_in, n);
            CHECK_KERNEL();
            for (int v = 0; v < NCFG; ++v) {
                if (v == 4 && !v4ok) { res[v][c] = 0xFFFFFFFFu; continue; }
                poison<<<1024, 256>>>(d_out, n);
                CHECK_KERNEL();
                launchV(v, d_in, d_out, Ws[c], Hs[c]);
                CHECK_KERNEL();
                unsigned zero = 0, bad = 0;
                CHECK(cudaMemcpy(d_bad, &zero, sizeof(unsigned), cudaMemcpyHostToDevice));
                if (v == 0) checkCopy     <<<1024, 256>>>(d_out, n, d_bad);
                else        checkTranspose<<<1024, 256>>>(d_out, Ws[c], Hs[c], d_bad);
                CHECK_KERNEL();
                CHECK(cudaMemcpy(&bad, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
                res[v][c] = bad;
            }
        }
        for (int v = 0; v < NCFG; ++v) {
            char a[24], b[24];
            if (res[v][0] == 0xFFFFFFFFu) snprintf(a, sizeof a, "skipped");
            else snprintf(a, sizeof a, res[v][0] ? "FAIL (%u)" : "ok", res[v][0]);
            if (res[v][1] == 0xFFFFFFFFu) snprintf(b, sizeof b, "skipped");
            else snprintf(b, sizeof b, res[v][1] ? "FAIL (%u)" : "ok", res[v][1]);
            printf("%-26s %14s %14s\n", vName[v], a, b);
            if (res[v][0] != 0xFFFFFFFFu && (res[v][0] || res[v][1])) ++nWrong;
        }
        if (!v4ok) ++nWrong;
    }
    printf("\n");

    // ---- the reveal: the same v3/v4 pair with DRAM taken away -------------
    if (v4ok) {
        const long long nS = (long long)SMALL * SMALL;
        fillSource<<<1024, 256>>>(d_in, nS);
        CHECK_KERNEL();
        double b3 = 1e30, b4 = 1e30;
        for (int k = 0; k < 40; ++k) launchV(3, d_in, d_out, SMALL, SMALL);
        CHECK_KERNEL();
        for (int sweep = 0; sweep < 2; ++sweep)
            for (int q = 0; q < 2; ++q) {
                int v = 3 + ((q + sweep) % 2);
                CHECK(cudaEventRecord(evA));
                for (int k = 0; k < 100; ++k) launchV(v, d_in, d_out, SMALL, SMALL);
                CHECK(cudaEventRecord(evB));
                CHECK(cudaEventSynchronize(evB));
                float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evA, evB));
                if (v == 3) { if (ms / 100.0 < b3) b3 = ms / 100.0; }
                else        { if (ms / 100.0 < b4) b4 = ms / 100.0; }
            }
        printf("THE SAME TWO KERNELS ON AN L2-RESIDENT %dx%d MATRIX\n", SMALL, SMALL);
        printf("  (not a DRAM bandwidth measurement -- both buffers fit in L2)\n");
        printf("  v3 plain 32x32 : %.4f ms\n", b3);
        printf("  v4 your layout : %.4f ms\n", b4);
        printf("  v3 / v4        : %.3fx\n\n", b3 / b4);
    }

    // ---- SCORING ----------------------------------------------------------
    int score = 0, maxScore = 10;
    printf("SCORING\n");
    if (nWrong == 0) { score += 4; printf("  numerics, all four versions, both matrices  : 4/4\n"); }
    else             { printf("  numerics, all four versions, both matrices  : 0/4 "
                              "(%d version(s) wrong or skipped)\n", nWrong); }
    if (v4ok) { score += 1; printf("  TODO 3 layout structurally valid            : 1/1\n"); }
    else      { printf("  TODO 3 layout structurally valid            : 0/1\n"); }

    int predOk = 0;
    for (int v = 0; v < 4; ++v) {
        int m = measuredBucket[v + 1];
        bool ok = (m != 0 && PRED[v] == m);
        if (ok) ++predOk;
        printf("  PRED[%d] = %d, measured bucket %d : %s\n",
               v, PRED[v], m, ok ? "correct" : "WRONG");
    }
    score += predOk;
    printf("  bucket predictions                          : %d/4\n", predOk);

    int gainBucket = 0;
    if (v4ok && best[3] < 1e29 && best[4] < 1e29)
        gainBucket = (best[3] / best[4] < 1.25) ? 1 : 2;
    bool gOk = (gainBucket != 0 && PRED[4] == gainBucket);
    if (gOk) ++score;
    printf("  PRED[4] = %d, measured v3/v4 = %.3fx -> bucket %d : %s\n",
           PRED[4], (v4ok && best[4] < 1e29) ? best[3] / best[4] : 0.0,
           gainBucket, gOk ? "correct" : "WRONG");

    printf("\nSCORE: %d/%d\n", score, maxScore);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_bad));
    CHECK(cudaEventDestroy(evA));
    CHECK(cudaEventDestroy(evB));
    CHECK(cudaDeviceReset());

    printf("OVERALL: %s\n", (score == maxScore) ? "PASS" : "FAIL");
    return (score == maxScore) ? 0 : 1;
}
