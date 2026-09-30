// =============================================================================
// Module 13 / Exercise 3 — single-pass scan by decoupled look-back
//
// GOAL : Scan the whole array in ONE kernel launch. Each tile publishes its own
//        aggregate as soon as it has it, then walks backwards over its
//        predecessors' published state until it finds one that already knows
//        its inclusive prefix, and then publishes its own inclusive prefix so
//        its successors can stop at it. 2N of traffic instead of 4N.
//
//        This is the only kernel in the course in which one block waits on
//        another. Module 9 established that this is normally a structural
//        deadlock. Read TODO 1 before you write anything: the argument for why
//        this particular wait terminates is the exercise.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN  : exercise03.exe
//
// WARNING: a partially-filled set of TODOs can hang the GPU rather than print
// a wrong answer — that is what a missing publish looks like. With all TODOs
// blank the program stops at the prediction gate and does not launch. While
// debugging, put a bounded iteration count on your spin and print how many
// tiles gave up; a bounded spin turns a hang into evidence (Module 9, Ex 2).
//
// TODO 1 — obtain this block's tile index   (design: forward progress)
// TODO 2 — publish the tile aggregate
// TODO 3 — the look-back loop
// TODO 4 — publish the inclusive prefix
// TODO 5 — predict the speedup before running
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                       \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e),             \
               __FILE__, __LINE__);                                            \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

typedef unsigned int u32;

#define BLK   256
#define TILE  1024
#define IPT   (TILE/BLK)

// Tile status. X = nothing published yet, A = aggregate only (this tile's own
// sum), P = inclusive prefix known (everything from tile 0 through this tile).
#define FLAG_X 0u
#define FLAG_A 1u
#define FLAG_P 2u

// ----------------------------------------------------------------------------
// Block scan, given (Exercise 1's warp-shuffle version).
// ----------------------------------------------------------------------------
__device__ __forceinline__ u32 warpInclusiveScan(u32 v, int lane)
{
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        u32 n = __shfl_up_sync(0xffffffffu, v, off);
        if (lane >= off) v += n;
    }
    return v;
}
__device__ u32 blockScanWarp(u32 *s, int tid)
{
    __shared__ u32 warpTot[BLK/32];
    const int lane = tid & 31, wid = tid >> 5;
    u32 x[IPT];
    #pragma unroll
    for (int k = 0; k < IPT; ++k) x[k] = s[tid*IPT + k];
    u32 run = 0;
    #pragma unroll
    for (int k = 0; k < IPT; ++k) { u32 t = x[k]; x[k] = run; run += t; }
    u32 wincl = warpInclusiveScan(run, lane);
    if (lane == 31) warpTot[wid] = wincl;
    __syncthreads();
    if (wid == 0) {
        u32 v = (lane < BLK/32) ? warpTot[lane] : 0u;
        v = warpInclusiveScan(v, lane);
        if (lane < BLK/32) warpTot[lane] = v;
    }
    __syncthreads();
    u32 wexcl = (wid == 0) ? 0u : warpTot[wid - 1];
    u32 texcl = wexcl + wincl - run;
    #pragma unroll
    for (int k = 0; k < IPT; ++k) s[tid*IPT + k] = x[k] + texcl;
    __syncthreads();
    return warpTot[BLK/32 - 1];
}

// ----------------------------------------------------------------------------
// Baseline: the three-kernel scan-then-propagate structure (given).
// ----------------------------------------------------------------------------
__global__ void scanTilesKernel(const u32 * __restrict__ in, u32 * __restrict__ out,
                                u32 * __restrict__ blockSums, int n)
{
    __shared__ u32 s[TILE];
    const int tid = threadIdx.x, base = blockIdx.x * TILE;
    for (int i = tid; i < TILE; i += BLK) s[i] = (base+i < n) ? in[base+i] : 0u;
    __syncthreads();
    u32 total = blockScanWarp(s, tid);
    for (int i = tid; i < TILE; i += BLK) if (base+i < n) out[base+i] = s[i];
    if (tid == 0) blockSums[blockIdx.x] = total;
}
__global__ void scanSumsKernel(u32 *v, int m)
{
    __shared__ u32 s[TILE];
    const int tid = threadIdx.x;
    u32 carry = 0;
    for (int base = 0; base < m; base += TILE) {
        for (int i = tid; i < TILE; i += BLK) s[i] = (base+i < m) ? v[base+i] : 0u;
        __syncthreads();
        u32 tot = blockScanWarp(s, tid);
        for (int i = tid; i < TILE; i += BLK) if (base+i < m) v[base+i] = s[i] + carry;
        carry += tot;
        __syncthreads();
    }
}
__global__ void addOffsetsKernel(u32 * __restrict__ out, const u32 * __restrict__ offs, int n)
{
    const u32 o = offs[blockIdx.x];
    const int base = blockIdx.x * TILE;
    #pragma unroll
    for (int k = 0; k < IPT; ++k) { int j = base + threadIdx.x + k*BLK; if (j < n) out[j] += o; }
}

// ----------------------------------------------------------------------------
// Decoupled look-back
// ----------------------------------------------------------------------------
__global__ void dlbInitKernel(u32 * __restrict__ flags, u32 * __restrict__ ticket, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) flags[i] = FLAG_X;
    if (i == 0) *ticket = 0u;
}

__global__ void dlbScanKernel(const u32 * __restrict__ in, u32 * __restrict__ out,
                              u32 * flags, u32 * aggs, u32 * pfxs, u32 * ticket,
                              int n, int m)
{
    __shared__ u32 s[TILE];
    __shared__ u32 s_tile, s_excl;
    const int tid = threadIdx.x;

    // ---------------------------------------------------------------- TODO 1
    // DESIGN TODO. Set s_tile to the index of the tile this block will own.
    //
    // The obvious answer is blockIdx.x. Before you write it, answer this: the
    // look-back you are about to write has a block spin until a LOWER-numbered
    // tile publishes. Module 9 showed that a block spinning on another block
    // deadlocks, because the spinner occupies the SM slot the producer needs.
    // Decide what has to be true about tile numbering for the spin to be
    // guaranteed to terminate, then pick an assignment that makes it true and
    // write the one-sentence argument in a comment here.
    //
    // `ticket` is a single device word, zeroed by dlbInitKernel before every
    // launch. It is there if you want it.
    //
    // Leave s_tile >= m for a block that has no work; the guard below uses it.
    if (tid == 0) { s_tile = 0xFFFFFFFFu; s_excl = 0u; }
    __syncthreads();
    if (tid == 0) {
        // YOUR CODE HERE
        (void)ticket;
    }
    __syncthreads();
    const u32 utile = s_tile;
    if (utile >= (u32)m) return;           // block-uniform, so the early exit is legal
    const int tile = (int)utile;

    const int base = tile * TILE;
    for (int i = tid; i < TILE; i += BLK) s[i] = (base+i < n) ? in[base+i] : 0u;
    __syncthreads();
    const u32 total = blockScanWarp(s, tid);

    // ---------------------------------------------------------------- TODO 2
    // Publish this tile's state so successors can use it. There are two cases:
    // tile 0 already knows its inclusive prefix (it has no predecessors), every
    // other tile knows only its own aggregate.
    //
    // Two threads on two different SMs are involved, so this is a
    // publish/subscribe (Module 9): the payload has to become visible at the
    // device coherence point before the flag that advertises it. Choose the
    // mechanism; state in a comment which scope you need and why the weaker
    // one is not enough.
    //
    // Exactly one thread should do this.
    if (tid == 0) {
        // YOUR CODE HERE
        (void)total; (void)flags; (void)aggs; (void)pfxs;
    }
    __syncthreads();

    if (tile > 0) {
        // ------------------------------------------------------------ TODO 3
        // The look-back. Compute this tile's EXCLUSIVE prefix — the sum of
        // every element in tiles 0 .. tile-1 — and leave it in s_excl.
        //
        // Requirements:
        //   - use one warp, working on up to 32 predecessors at a time, rather
        //     than one thread walking backwards one tile at a time;
        //   - a predecessor showing FLAG_P contributes its INCLUSIVE prefix and
        //     terminates the walk; predecessors above it (nearer to you)
        //     contribute their aggregates; a predecessor showing FLAG_X has not
        //     published yet and must be waited on;
        //   - a plain load of the flag is not guaranteed to ever observe
        //     another SM's store (Module 4: L1 is not coherent across SMs).
        //     Pick a load that is;
        //   - once you have observed a flag, the payload read must not be
        //     allowed to move ahead of it.
        //
        // __ballot_sync, __ffs, __shfl_down_sync and __shfl_sync are all
        // available and all three are useful here.
        if (tid < 32) {
            const int lane = tid;
            (void)aggs; (void)pfxs; (void)flags;
            // YOUR CODE HERE

            // ------------------------------------------------------- TODO 4
            // Publish this tile's inclusive prefix so that the tiles behind you
            // can stop at you instead of walking all the way to tile 0. Without
            // this, tile k does O(k) work and the whole scan is O(m^2).
            // Also leave the exclusive prefix where the rest of the block can
            // see it.
            if (lane == 0) {
                // YOUR CODE HERE
                (void)total;
            }
        }
        __syncthreads();
    }

    const u32 off = s_excl;
    for (int i = tid; i < TILE; i += BLK) if (base+i < n) out[base+i] = s[i] + off;
}

// ---------------------------------------------------------------- TODO 5 ----
// Predict how much faster the single-pass scan is than the three-kernel scan
// at N = 67,108,861, as a bucket:
//   1: >= 2.0x faster   2: 1.5x .. 2.0x   3: 1.15x .. 1.5x   4: < 1.15x
//
// Both are bandwidth-bound. Write down how many times each one reads N words
// and how many times it writes N words, take the ratio, and then decide
// whether you expect to hit it exactly, beat it, or fall short — and say which
// way the look-back pushes you. Setting this to 0 stops the program.
#define PREDICT_BUCKET 0

static void launchThreeKernel(const u32 *d_in, u32 *d_out, u32 *d_sums, int n, int m)
{
    scanTilesKernel<<<m, BLK>>>(d_in, d_out, d_sums, n);
    scanSumsKernel<<<1, BLK>>>(d_sums, m);
    addOffsetsKernel<<<m, BLK>>>(d_out, d_sums, n);
}
static void launchSinglePass(const u32 *d_in, u32 *d_out, u32 *d_f, u32 *d_a,
                             u32 *d_p, u32 *d_t, int n, int m)
{
    dlbInitKernel<<<(m + 255)/256, 256>>>(d_f, d_t, m);
    dlbScanKernel<<<m, BLK>>>(d_in, d_out, d_f, d_a, d_p, d_t, n, m);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    if (PREDICT_BUCKET == 0) { printf("Set TODO 5 (PREDICTION) first.\n"); return 0; }

    const int nMax = 67108861;
    const int mMax = (nMax + TILE - 1) / TILE;

    printf("Module 13 / Exercise 3 — single-pass scan by decoupled look-back\n\n");

    u32 *h_in  = (u32*)malloc(sizeof(u32)*(size_t)nMax);
    u32 *h_ref = (u32*)malloc(sizeof(u32)*(size_t)nMax);
    u32 *h_out = (u32*)malloc(sizeof(u32)*(size_t)nMax);
    if (!h_in || !h_ref || !h_out) { printf("host alloc failed\n"); return 1; }
    unsigned seed = 424242u;
    for (int i = 0; i < nMax; ++i) { seed = seed*1664525u + 1013904223u; h_in[i] = (seed >> 26) & 3u; }

    u32 *d_in, *d_out, *d_sums, *d_f, *d_a, *d_p, *d_t;
    CHECK(cudaMalloc(&d_in,   sizeof(u32)*(size_t)nMax));
    CHECK(cudaMalloc(&d_out,  sizeof(u32)*(size_t)nMax));
    CHECK(cudaMalloc(&d_sums, sizeof(u32)*(size_t)mMax));
    CHECK(cudaMalloc(&d_f,    sizeof(u32)*(size_t)mMax));
    CHECK(cudaMalloc(&d_a,    sizeof(u32)*(size_t)mMax));
    CHECK(cudaMalloc(&d_p,    sizeof(u32)*(size_t)mMax));
    CHECK(cudaMalloc(&d_t,    sizeof(u32)));
    CHECK(cudaMemcpy(d_in, h_in, sizeof(u32)*(size_t)nMax, cudaMemcpyHostToDevice));

    // ---------------- correctness across awkward grid shapes -----------------
    const int probeSizes[] = { 1, 1023, 1024, 1025, 40959, 245761, 1048573, 4194301, 67108861 };
    const int nProbe = (int)(sizeof(probeSizes)/sizeof(probeSizes[0]));
    int probeFails = 0;
    printf("correctness sweep (single pass) over %d sizes:\n", nProbe);
    for (int k = 0; k < nProbe; ++k) {
        int n = probeSizes[k], m = (n + TILE - 1)/TILE;
        CHECK(cudaMemset(d_out, 0xAB, sizeof(u32)*(size_t)n));
        launchSinglePass(d_in, d_out, d_f, d_a, d_p, d_t, n, m);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out, d_out, sizeof(u32)*(size_t)n, cudaMemcpyDeviceToHost));
        u32 acc = 0; int bad = -1;
        for (int i = 0; i < n; ++i) { if (h_out[i] != acc) { bad = i; break; } acc += h_in[i]; }
        printf("   n=%-9d m=%-6d %s\n", n, m, (bad < 0) ? "PASS" : "FAIL");
        if (bad >= 0) { printf("      first mismatch at %d\n", bad); probeFails++; }
    }

    // ---------------- repeatability: the property a spin-wait must have ------
    const int REPEATS = 200;
    int repeatFails = 0;
    {
        int n = 4194301, m = (n + TILE - 1)/TILE;
        for (int i = 0; i < n; ++i) { /* build reference once */ }
        u32 acc = 0;
        for (int i = 0; i < n; ++i) { h_ref[i] = acc; acc += h_in[i]; }
        for (int r = 0; r < REPEATS; ++r) {
            launchSinglePass(d_in, d_out, d_f, d_a, d_p, d_t, n, m);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            if ((r % 20) == 0) {
                CHECK(cudaMemcpy(h_out, d_out, sizeof(u32)*(size_t)n, cudaMemcpyDeviceToHost));
                for (int i = 0; i < n; ++i) if (h_out[i] != h_ref[i]) { repeatFails++; break; }
            }
        }
        printf("\n%d back-to-back single-pass launches at n=%d: %d bad results, no hangs\n",
               REPEATS, n, repeatFails);
    }

    // ---------------- timing -------------------------------------------------
    const int n = nMax, m = mMax;
    {
        u32 acc = 0;
        for (int i = 0; i < n; ++i) { h_ref[i] = acc; acc += h_in[i]; }
    }
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    {
        float el = 0.0f;
        CHECK(cudaEventRecord(e0));
        do { launchSinglePass(d_in, d_out, d_f, d_a, d_p, d_t, n, m);
             CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
             CHECK(cudaEventElapsedTime(&el, e0, e1)); } while (el < 400.0f);
    }
    int iters;
    {
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < 20; ++i) launchSinglePass(d_in, d_out, d_f, d_a, d_p, d_t, n, m);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        iters = (int)(10.0f/(ms/20.0f)); if (iters < 20) iters = 20; if (iters > 2000) iters = 2000;
    }
    float best[2] = { 1e30f, 1e30f };
    for (int sweep = 0; sweep < 4; ++sweep) {
        for (int q = 0; q < 2; ++q) {
            int c = (q + sweep) % 2;
            CHECK(cudaEventRecord(e0));
            for (int it = 0; it < iters; ++it) {
                if (c == 0) launchThreeKernel(d_in, d_out, d_sums, n, m);
                else        launchSinglePass(d_in, d_out, d_f, d_a, d_p, d_t, n, m);
            }
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms/iters < best[c]) best[c] = ms/(float)iters;
        }
    }
    CHECK(cudaGetLastError());

    int valid[2];
    for (int c = 0; c < 2; ++c) {
        CHECK(cudaMemset(d_out, 0xAB, sizeof(u32)*(size_t)n));
        if (c == 0) launchThreeKernel(d_in, d_out, d_sums, n, m);
        else        launchSinglePass(d_in, d_out, d_f, d_a, d_p, d_t, n, m);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out, d_out, sizeof(u32)*(size_t)n, cudaMemcpyDeviceToHost));
        valid[c] = 1;
        for (int i = 0; i < n; ++i) if (h_out[i] != h_ref[i]) { valid[c] = 0; break; }
    }

    const double bytes2N = 2.0*(double)n*sizeof(u32);
    const char *CN[2] = { "three-kernel (4N traffic)", "single-pass  (2N traffic)" };
    const double traffic[2] = { 2.0*bytes2N, bytes2N };
    printf("\nN = %d, %d tiles\n", n, m);
    printf("%-26s %9s %12s %10s %10s %7s\n",
           "strategy", "ms", "GB/s (2N)", "GB/s (real)", "% of 432", "valid");
    for (int c = 0; c < 2; ++c) {
        double g2 = bytes2N/(best[c]*1e-3)/1e9;
        double gr = traffic[c]/(best[c]*1e-3)/1e9;
        printf("%-26s %9.4f %12.1f %10.1f %9.1f%% %7s\n",
               CN[c], best[c], g2, gr, 100.0*gr/432.0, valid[c] ? "PASS" : "FAIL");
    }
    printf("  2N floor at 432 GB/s = %.4f ms\n", bytes2N/432e9*1e3);

    double ratio = best[0]/best[1];
    int bucket = (ratio >= 2.0) ? 1 : (ratio >= 1.5) ? 2 : (ratio >= 1.15) ? 3 : 4;
    printf("\n  three-kernel / single-pass = %.2fx -> bucket %d   predicted %d\n",
           ratio, bucket, PREDICT_BUCKET);

    int score = 0;
    score += (probeFails == 0)  ? 3 : 0;
    score += (repeatFails == 0) ? 2 : 0;
    score += valid[0] + valid[1];
    score += (bucket == PREDICT_BUCKET) ? 1 : 0;
    printf("\n  score: %d/8\n", score);
    printf("OVERALL: %s\n", (score == 8) ? "PASS" : "FAIL");

    free(h_in); free(h_ref); free(h_out);
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_sums));
    CHECK(cudaFree(d_f)); CHECK(cudaFree(d_a)); CHECK(cudaFree(d_p)); CHECK(cudaFree(d_t));
    CHECK(cudaDeviceReset());
    return (score == 8) ? 0 : 1;
}
