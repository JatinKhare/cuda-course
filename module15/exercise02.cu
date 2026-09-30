// ============================================================================
// Module 15 / exercise02.cu -- A layout conversion that is not a plain
//                              transpose: NCHW -> NHWC
//
// GOAL : Convert a 4-D activation tensor from channels-major (NCHW, what
//        cuDNN's older kernels and most textbook code use) to channels-last
//        (NHWC, what Tensor-Core convolution and most modern inference kernels
//        want). You get a working naive kernel and a copy ceiling. You design
//        and write the fast one. There is almost no scaffolding: the grid
//        shape, the thread-to-element mapping, the shared-memory layout and the
//        boundary handling are all yours.
//
//        Index algebra, once, precisely. With n in [0,NI), c in [0,C),
//        h in [0,H), w in [0,W):
//
//            NCHW element (n,c,h,w) lives at  ((n*C + c)*H + h)*W + w
//            NHWC element (n,h,w,c) lives at  ((n*H + h)*W + w)*C + c
//
//        The conversion is therefore, for each image n independently, the
//        transpose of a C x (H*W) matrix into an (H*W) x C matrix. That is the
//        whole problem. Everything else is choosing a tile.
//
//        This is also the general form of the AoS -> SoA conversion Module 5
//        deferred to this module: NHWC is an array of structs whose struct is
//        "the C channels of one pixel", NCHW is the struct of arrays.
//
// PARAMETERS: NI=128 images, C=67 channels, H=W=57.  None of C, H*W is a
//        multiple of 32, so boundary handling is not optional, and C = 67 is
//        barely more than two tiles wide -- a shape that punishes a tile choice
//        made by habit.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
// RUN  : .\exercise02.exe
//
// Timing follows AUTHORING_SPEC section 12.
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

static const int NI = 128, CC = 67, HH = 57, WW = 57;
static const int SPATIAL = HH * WW;                 // 3249
static const long long NTOTAL = (long long)NI * CC * SPATIAL;   // 27,863,424

static const int TILE  = 32;      // fixed by the harness: the tile is 32 x 32
static const int BROWS = 8;       // fixed by the harness: the block is (32, 8)
static const int MAXSH = TILE * (TILE + 1);

__host__ __device__ __forceinline__ float srcValue(long long i)
{
    unsigned h = (unsigned)i * 2654435761u;
    h ^= h >> 13;
    h *= 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0x00FFFFFFu);
}

// ===========================================================================
// TODO 5 (PREDICTION) -- commit before you compile.
//
// PRED[0]: the naive kernel below reads NCHW contiguously and writes NHWC with
//          a stride of C floats. As a fraction of the copy ceiling it will be
//              1 = below 70%     2 = 70% to 82%     3 = above 82%
//
// PRED[1]: your tiled kernel, as a fraction of the copy ceiling:
//              1 = below 70%     2 = 70% to 82%     3 = above 82%
//          (Same bucket edges as PRED[0]. The naive kernel's strided write
//           is the least reproducible number in this program -- measured
//           over many runs it lands anywhere in 29-59% of the copy, because
//           a fully strided write is the most power-hungry kernel here and
//           loses the most when the part is power limited. The tiled kernel
//           sits at 90-100%. Both edges are placed in the empty gaps, at
//           least 8 points from either measured band, so no bucket is
//           decided by the thermal state the run happened to start in.)
//
// PRED[2]: C = 67 means the channel axis is covered by ceil(67/32) = 3 tiles,
//          and the last of those three has only 3 useful columns of 32. What
//          fraction of the blocks in your grid are therefore mostly idle, and
//          what does that do to the achievable bandwidth? Answer with the
//          fraction of TOTAL LAUNCHED TILE-CELLS that are inside the tensor:
//              1 = above 90%     2 = 60% to 90%     3 = below 60%
// ===========================================================================
static const int PRED[3] = { 0, 0, 0 };
// YOUR CODE HERE (replace the three zeros)

// ---------------------------------------------------------------------------
// The ceiling: a plain copy of the same number of elements. 2N traffic, both
// sides perfectly coalesced, no permutation. Nothing you write can beat it.
// ---------------------------------------------------------------------------
__global__ void copyCeiling(const float4* __restrict__ in, float4* __restrict__ out,
                            long long n4)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    for (; i < n4; i += s) out[i] = in[i];
}

// ---------------------------------------------------------------------------
// The naive conversion. GIVEN. One thread per input element, walking NCHW in
// linear order so the read is perfectly coalesced, and paying for it on the
// write side.
// ---------------------------------------------------------------------------
__global__ void naiveNCHW2NHWC(const float* __restrict__ in, float* __restrict__ out,
                               int ni, int c, int h, int w)
{
    long long total = (long long)ni * c * h * w;
    long long step  = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += step) {
        int ww_ = (int)(i % w);
        int hh_ = (int)((i / w) % h);
        int cc_ = (int)((i / ((long long)w * h)) % c);
        int nn_ = (int)(i / ((long long)w * h * c));
        out[(((long long)nn_ * h + hh_) * w + ww_) * c + cc_] = in[i];
    }
}

// ===========================================================================
// TODO 4 (DESIGN) -- the shared-memory layout.
//
// Your kernel will stage a 32 x 32 patch and read it back along the other axis.
// Define the map from a logical tile cell (r, c), 0 <= r, c < 32, to an offset
// in a flat shared array of SH_FLOATS floats.
//
// Enforced: SH_FLOATS in (0, 1056]; shIdx injective on [0,32)x[0,32) and inside
// [0, SH_FLOATS); the row-across-a-warp phase and the column-across-a-warp
// phase must both have conflict degree <= 2 (Module 7's max(2,D) law means a
// 2-way conflict is free on Ada, so there is no reason to demand 1).
//
// Shipped state returns -1, which makes the harness skip your kernel.
// ===========================================================================
static const int SH_FLOATS = 0;     // YOUR CODE HERE

__host__ __device__ __forceinline__ int shIdx(int r, int c)
{
    (void)r; (void)c;
    return -1;                      // YOUR CODE HERE
}

// ===========================================================================
// Your kernel.
//
// TODO 1 (DESIGN): decide what one block does and write the three grid
//   dimensions in gridFor() below. A block owns one 32x32 tile of one image's
//   C x (H*W) matrix. You have three things to cover -- the channel axis, the
//   spatial axis, and the batch -- and three grid dimensions, but the CUDA
//   limits on them are not equal (65535 on y and z, 2^31-1 on x, Module 3), and
//   neither is the cost of the arithmetic that recovers your coordinates.
//
// TODO 2: the load phase. Read from NCHW so that the 32 lanes of a warp supply
//   32 consecutive addresses, and stage into shared memory. Guard both axes.
//   Elements outside the tensor must not be read AND must not leave stale data
//   in the tile -- Module 6's partial-tile hazard, one module on.
//
// TODO 3: the store phase. Write to NHWC so that the 32 lanes of a warp supply
//   32 consecutive addresses. Guard both axes. Note which axis is contiguous in
//   NHWC and what that means for which of your two tile coordinates must vary
//   with threadIdx.x here.
// ===========================================================================
__host__ static void gridFor(int ni, int c, int spatial, dim3* grid)
{
    (void)ni; (void)c; (void)spatial;
    grid->x = 0; grid->y = 0; grid->z = 0;   // YOUR CODE HERE (TODO 1)
}

__global__ void fastNCHW2NHWC(const float* __restrict__ in, float* __restrict__ out,
                              int ni, int c, int spatial)
{
    __shared__ float tile[MAXSH];
    (void)in; (void)ni; (void)c; (void)spatial;
    // Placeholder so the shipped file compiles warning-clean; the condition is
    // never true (threadIdx.x < blockDim.x always). Delete it when you write
    // the kernel.
    if (threadIdx.x >= blockDim.x) out[0] = tile[0];

    // TODO 2: recover this block's (image, channel origin, spatial origin) from
    // blockIdx, then cooperatively load the tile.
    // YOUR CODE HERE

    // (you will need a barrier here)

    // TODO 3: write the tile out in NHWC order.
    // YOUR CODE HERE
}

// ---------------------------------------------------------------------------
// Harness. Nothing below needs to change.
// ---------------------------------------------------------------------------
// Warm-up companion (AUTHORING_SPEC 12.4 corollary): a pure-arithmetic kernel
// with no memory traffic. A streaming warm-up alone ramps the memory P-state
// but lets the power manager drop the SM clock; this pulls it back up.
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
    for (; i < n; i += s) a[i] = -13.0f;
}
// out is NHWC; element (n,h,w,c) at ((n*H+h)*W+w)*C+c must equal the NCHW
// source value at ((n*C+c)*H+h)*W+w.
__global__ void checkNHWC(const float* __restrict__ out, int ni, int c, int h, int w,
                          unsigned* bad)
{
    long long total = (long long)ni * c * h * w;
    long long step  = (long long)blockDim.x * gridDim.x;
    unsigned local = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += step) {
        int cc_ = (int)(i % c);
        long long t = i / c;
        int ww_ = (int)(t % w);
        int hh_ = (int)((t / w) % h);
        int nn_ = (int)(t / ((long long)w * h));
        if (out[i] != srcValue((((long long)nn_ * c + cc_) * h + hh_) * w + ww_))
            ++local;
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

static int degreeOfPhase(bool rowAcrossWarp)
{
    int worst = 0;
    for (int fixed = 0; fixed < TILE; ++fixed) {
        int words[32][64], nw[32];
        for (int b = 0; b < 32; ++b) nw[b] = 0;
        for (int lane = 0; lane < 32; ++lane) {
            int off = rowAcrossWarp ? shIdx(fixed, lane) : shIdx(lane, fixed);
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

    for (int i = 0; i < 3; ++i)
        if (PRED[i] == 0) { printf("Set TODO 5 (PREDICTION) first.\n"); return 0; }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Module 15 exercise 02 -- NCHW -> NHWC\n");
    printf("GPU: %s, CC %d.%d\n", prop.name, prop.major, prop.minor);
    printf("tensor: N=%d C=%d H=%d W=%d -> %lld floats = %.1f MiB per buffer\n",
           NI, CC, HH, WW, NTOTAL, (double)NTOTAL * 4.0 / 1048576.0);
    printf("per image this is a %d x %d transpose\n\n", CC, SPATIAL);

    // ---- structural check of TODO 4 ---------------------------------------
    bool layoutOk = false;
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
        int d1 = degreeOfPhase(true), d2 = degreeOfPhase(false);
        printf("TODO 4: SH_FLOATS=%d, in range %s, injective %s, degrees %d / %d\n",
               SH_FLOATS, inRange ? "yes" : "NO", injective ? "yes" : "NO", d1, d2);
        layoutOk = inRange && injective && d1 <= 2 && d2 <= 2;
    } else {
        printf("TODO 4 not set.\n");
    }

    // ---- structural check of TODO 1 ---------------------------------------
    dim3 grid(0, 0, 0);
    gridFor(NI, CC, SPATIAL, &grid);
    long long tilesNeeded = (long long)((CC + TILE - 1) / TILE)
                          * ((SPATIAL + TILE - 1) / TILE) * NI;
    long long tilesLaunched = (long long)grid.x * grid.y * grid.z;
    bool gridOk = (grid.x > 0 && grid.y > 0 && grid.z > 0 &&
                   grid.y <= 65535 && grid.z <= 65535 &&
                   tilesLaunched >= tilesNeeded &&
                   tilesLaunched <= 2 * tilesNeeded);
    printf("TODO 1: grid = (%u, %u, %u) = %lld blocks; %lld tiles needed -> %s\n",
           grid.x, grid.y, grid.z, tilesLaunched, tilesNeeded,
           gridOk ? "accepted" : "REJECTED (must cover the tensor, y/z <= 65535, "
                                 "and not launch more than 2x the tiles needed)");

    bool userOk = layoutOk && gridOk;
    printf("\n");

    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  (size_t)NTOTAL * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)NTOTAL * sizeof(float)));
    unsigned* d_bad = nullptr;
    CHECK(cudaMalloc(&d_bad, sizeof(unsigned)));
    fillSource<<<2048, 256>>>(d_in, NTOTAL);
    CHECK_KERNEL();

    cudaEvent_t evA, evB;
    CHECK(cudaEventCreate(&evA));
    CHECK(cudaEventCreate(&evB));

    const long long n4 = NTOTAL / 4;
    const int NCFG = 3, NSWEEP = 8;      // spec 12.9: SWEEPS >= NCFG (min-of-8)
    dim3 blk(TILE, BROWS);

    auto launchCfg = [&](int v) {
        if (v == 0) copyCeiling<<<8192, 256>>>(
                        reinterpret_cast<const float4*>(d_in),
                        reinterpret_cast<float4*>(d_out), n4);
        else if (v == 1) naiveNCHW2NHWC<<<8192, 256>>>(d_in, d_out, NI, CC, HH, WW);
        else fastNCHW2NHWC<<<grid, blk>>>(d_in, d_out, NI, CC, SPATIAL);
    };

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
    const double CEIL_FLOOR_GBPS = 200.0;   // healthy ~255, power-capped ~105
    for (int attempt = 0; ; ++attempt) {
        float acc = 0.0f;
        CHECK(cudaEventRecord(evA));
        while (acc < 1500.0f) {
            for (int k = 0; k < 10; ++k) launchCfg(0);
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
        for (int k = 0; k < 8; ++k) launchCfg(0);
        CHECK(cudaEventRecord(evB));
        CHECK(cudaEventSynchronize(evB));
        float pms = 0.0f; CHECK(cudaEventElapsedTime(&pms, evA, evB));
        double gbps = 2.0 * (double)NTOTAL * 4.0
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
    int iters[3] = { 20, 20, 20 };
    for (int v = 0; v < NCFG; ++v) {
        if (v == 2 && !userOk) continue;
        CHECK(cudaEventRecord(evA));
        for (int k = 0; k < 4; ++k) launchCfg(v);
        CHECK(cudaEventRecord(evB));
        CHECK(cudaEventSynchronize(evB));
        float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evA, evB));
        double per = (double)ms / 4.0;
        int n = (per > 1.0e-6) ? (int)(10.0 / per + 0.5) : 20;
        iters[v] = (n < 5) ? 5 : (n > 400 ? 400 : n);
    }
    CHECK_KERNEL();

    double best[3] = { 1e30, 1e30, 1e30 };
    for (int sweep = 0; sweep < NSWEEP; ++sweep)
        for (int q = 0; q < NCFG; ++q) {
            int v = (q + sweep) % NCFG;
            if (v == 2 && !userOk) continue;
            CHECK(cudaEventRecord(evA));
            for (int k = 0; k < iters[v]; ++k) launchCfg(v);
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evA, evB));
            double per = (double)ms / (double)iters[v];
            if (per < best[v]) best[v] = per;
        }
    CHECK_KERNEL();

    const double b2 = 2.0 * (double)NTOTAL * 4.0;
    const char* nm[3] = { "copy ceiling        ", "naive NCHW->NHWC    ",
                          "your tiled NCHW->NHWC" };
    printf("TIMING (min of %d rotated sweeps)\n", NSWEEP);
    printf("%-22s %9s %9s %9s\n", "kernel", "ms", "GB/s", "%ofcopy");
    int bucket[3] = { 0, 0, 0 };
    for (int v = 0; v < 3; ++v) {
        if (best[v] > 1e29) { printf("%-22s %9s %9s %9s\n", nm[v], "-", "-", "-"); continue; }
        double frac = 100.0 * best[0] / best[v];
        printf("%-22s %9.4f %9.1f %8.1f%%\n", nm[v], best[v],
               b2 / (best[v] * 1.0e-3) / 1.0e9, frac);
        if (v == 1) bucket[1] = (frac < 70.0) ? 1 : (frac < 82.0 ? 2 : 3);
        if (v == 2) bucket[2] = (frac < 70.0) ? 1 : (frac < 82.0 ? 2 : 3);
    }
    printf("\n");

    // ---- validation --------------------------------------------------------
    unsigned badNaive = 0, badFast = 0xFFFFFFFFu;
    {
        unsigned zero = 0;
        poison<<<2048, 256>>>(d_out, NTOTAL); CHECK_KERNEL();
        naiveNCHW2NHWC<<<8192, 256>>>(d_in, d_out, NI, CC, HH, WW); CHECK_KERNEL();
        CHECK(cudaMemcpy(d_bad, &zero, sizeof(unsigned), cudaMemcpyHostToDevice));
        checkNHWC<<<2048, 256>>>(d_out, NI, CC, HH, WW, d_bad); CHECK_KERNEL();
        CHECK(cudaMemcpy(&badNaive, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));

        if (userOk) {
            poison<<<2048, 256>>>(d_out, NTOTAL); CHECK_KERNEL();
            fastNCHW2NHWC<<<grid, blk>>>(d_in, d_out, NI, CC, SPATIAL); CHECK_KERNEL();
            CHECK(cudaMemcpy(d_bad, &zero, sizeof(unsigned), cudaMemcpyHostToDevice));
            checkNHWC<<<2048, 256>>>(d_out, NI, CC, HH, WW, d_bad); CHECK_KERNEL();
            CHECK(cudaMemcpy(&badFast, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
        }
        // the copy path is validated too, so a broken harness cannot pass you
        poison<<<2048, 256>>>(d_out, NTOTAL); CHECK_KERNEL();
        copyCeiling<<<8192, 256>>>(reinterpret_cast<const float4*>(d_in),
                                   reinterpret_cast<float4*>(d_out), n4);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(d_bad, &zero, sizeof(unsigned), cudaMemcpyHostToDevice));
        unsigned bc = 0;
        checkCopy<<<2048, 256>>>(d_out, NTOTAL, d_bad); CHECK_KERNEL();
        CHECK(cudaMemcpy(&bc, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
        printf("VALIDATION\n");
        printf("  copy ceiling          : %s\n", bc ? "FAIL" : "ok");
        printf("  naive NCHW->NHWC      : %s\n", badNaive ? "FAIL" : "ok");
        if (badFast == 0xFFFFFFFFu) printf("  your kernel           : skipped\n");
        else printf("  your kernel           : %s (%u mismatches)\n",
                    badFast ? "FAIL" : "ok", badFast);
        printf("\n");
    }

    // ---- scoring -----------------------------------------------------------
    int score = 0, maxScore = 7;
    printf("SCORING\n");
    if (layoutOk) { ++score; printf("  TODO 4 shared layout valid         : 1/1\n"); }
    else            printf("  TODO 4 shared layout valid         : 0/1\n");
    if (gridOk)   { ++score; printf("  TODO 1 grid covers the tensor      : 1/1\n"); }
    else            printf("  TODO 1 grid covers the tensor      : 0/1\n");
    if (badFast == 0) { ++score; printf("  your kernel is correct             : 1/1\n"); }
    else                printf("  your kernel is correct             : 0/1\n");
    bool gate = (badFast == 0 && best[2] < 1e29 && best[0] / best[2] >= 0.80);
    if (gate) { ++score; printf("  your kernel >= 80%% of the copy     : 1/1\n"); }
    else        printf("  your kernel >= 80%% of the copy     : 0/1 (measured %.1f%%)\n",
                       (best[2] < 1e29) ? 100.0 * best[0] / best[2] : 0.0);

    // TODO 5 predictions
    bool p0 = (PRED[0] == bucket[1]);
    bool p1 = (bucket[2] != 0 && PRED[1] == bucket[2]);
    long long cells = tilesNeeded * TILE * TILE;
    double fill = (double)NTOTAL / (double)cells;
    int fb = (fill > 0.90) ? 1 : (fill >= 0.60 ? 2 : 3);
    bool p2 = (PRED[2] == fb);
    if (p0) ++score;
    if (p1) ++score;
    if (p2) ++score;
    printf("  PRED[0] naive bucket   %d, measured %d : %s\n", PRED[0], bucket[1],
           p0 ? "correct" : "WRONG");
    printf("  PRED[1] yours bucket   %d, measured %d : %s\n", PRED[1], bucket[2],
           p1 ? "correct" : "WRONG");
    printf("  PRED[2] tile fill      %d, measured %d (%.1f%% of launched tile cells "
           "hold data) : %s\n", PRED[2], fb, 100.0 * fill, p2 ? "correct" : "WRONG");

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
