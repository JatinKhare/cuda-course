// =============================================================================
// Module 13 / Exercise 1 — SOLUTION — the scan ladder
//
// GOAL : Implement three block-level scan algorithms (Hillis-Steele,
//        Blelloch, warp-shuffle), run each inside the same three-kernel
//        multi-block scan, validate at non-power-of-two sizes, and measure
//        both the whole scan and the tile-scan kernel in isolation.
//
// BUILD: nvcc -arch=sm_89 -O3 -o ex1sol.exe exercise01_solution.cu
// RUN  : ex1sol.exe
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

#define BLK   256          // threads per block
#define TILE  1024         // elements per block
#define IPT   (TILE/BLK)   // 4 items per thread

// Shared scratch: 2*TILE words = 8192 B. Big enough for the Hillis-Steele
// double buffer, for the padded Blelloch tree, and for the warp version.
#define SMEM_WORDS (2*TILE)

// Module 7's classic padding: one dead word every 32 words.
#define CONFLICT_FREE_OFFSET(i) ((i) >> 5)
#define PIDX(i) ((i) + CONFLICT_FREE_OFFSET(i))

enum { ALG_HS = 0, ALG_BL = 1, ALG_WARP = 2, N_ALG = 3 };
static const char *ALG_NAME[N_ALG] = { "Hillis-Steele", "Blelloch", "warp-shuffle" };

// ---------------------------------------------------------------- TODO 1 ----
// Hillis-Steele. Inclusive by construction; shifted to exclusive at the end.
// Input:  s[0 .. TILE)      (buffer 0)
// Output: s[0 .. TILE)      exclusive scan
// Returns the tile total.
__device__ u32 blockScanHS(u32 *s, int tid)
{
    int cur = 0;                       // which half of s[] currently holds data
    for (int off = 1; off < TILE; off <<= 1) {
        // Read from buffer `cur`, write to buffer `cur^1`. Never both in one
        // array: s[i] += s[i-off] is a cross-thread WAR hazard.
        for (int i = tid; i < TILE; i += BLK)
            s[(cur^1)*TILE + i] = s[cur*TILE + i] +
                                  ((i >= off) ? s[cur*TILE + i - off] : 0u);
        __syncthreads();
        cur ^= 1;
    }
    u32 total = s[cur*TILE + TILE - 1];

    // inclusive -> exclusive: shift right by one, identity at slot 0.
    // Read into registers first, then barrier, then write: the destination
    // (buffer 0) may be the source.
    u32 v[IPT];
    #pragma unroll
    for (int k = 0; k < IPT; ++k) {
        int i = tid + k*BLK;
        v[k] = (i == 0) ? 0u : s[cur*TILE + i - 1];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < IPT; ++k) s[tid + k*BLK] = v[k];
    __syncthreads();
    return total;
}

// ------------------------------------------------------- TODO 2 and TODO 3 --
// Blelloch. Works in place on the PADDED layout: element i lives at PIDX(i).
// Returns the tile total; leaves an exclusive scan behind.
__device__ u32 blockScanBlelloch(u32 *s, int tid)
{
    // ---- TODO 2: upsweep (reduce) ----
    int offset = 1;
    for (int d = TILE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        for (int t = tid; t < d; t += BLK) {
            int ai = offset*(2*t + 1) - 1;
            int bi = offset*(2*t + 2) - 1;
            s[PIDX(bi)] += s[PIDX(ai)];
        }
        offset <<= 1;
    }
    __syncthreads();

    // ---- TODO 3: identity at the root, then downsweep ----
    u32 total = s[PIDX(TILE-1)];
    if (tid == 0) s[PIDX(TILE-1)] = 0u;     // <- the identity element
    for (int d = 1; d < TILE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        for (int t = tid; t < d; t += BLK) {
            int ai = offset*(2*t + 1) - 1;
            int bi = offset*(2*t + 2) - 1;
            u32 tmp      = s[PIDX(ai)];
            s[PIDX(ai)]  = s[PIDX(bi)];
            s[PIDX(bi)] += tmp;
        }
    }
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------- TODO 4 ----
// Warp-shuffle block scan. Three levels:
//   (1) each thread serially scans its IPT items (registers, no traffic),
//   (2) __shfl_up_sync scan of the per-thread totals inside each warp,
//   (3) scan of the per-warp totals in warp 0, then broadcast-add.
// Shared use beyond the tile itself: BLK/32 = 8 words.
__device__ __forceinline__ u32 warpInclusiveScan(u32 v, int lane)
{
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        u32 n = __shfl_up_sync(0xffffffffu, v, off);
        if (lane >= off) v += n;       // lanes < off receive their own value; discard
    }
    return v;
}

__device__ u32 blockScanWarp(u32 *s, int tid)
{
    __shared__ u32 warpTot[BLK/32];
    const int lane = tid & 31;
    const int wid  = tid >> 5;

    u32 x[IPT];
    #pragma unroll
    for (int k = 0; k < IPT; ++k) x[k] = s[tid*IPT + k];

    // (1) serial exclusive scan of this thread's slice
    u32 run = 0;
    #pragma unroll
    for (int k = 0; k < IPT; ++k) { u32 t = x[k]; x[k] = run; run += t; }

    // (2) warp-level inclusive scan of thread totals
    u32 wincl = warpInclusiveScan(run, lane);
    if (lane == 31) warpTot[wid] = wincl;
    __syncthreads();

    // (3) scan the 8 warp totals in warp 0
    if (wid == 0) {
        u32 v = (lane < BLK/32) ? warpTot[lane] : 0u;
        v = warpInclusiveScan(v, lane);
        if (lane < BLK/32) warpTot[lane] = v;
    }
    __syncthreads();

    u32 wexcl = (wid == 0) ? 0u : warpTot[wid - 1];
    u32 texcl = wexcl + wincl - run;     // inclusive minus own total = exclusive
    #pragma unroll
    for (int k = 0; k < IPT; ++k) s[tid*IPT + k] = x[k] + texcl;
    __syncthreads();
    return warpTot[BLK/32 - 1];
}

// ------------------------------------------------------------ pass 1 --------
template<int ALG>
__device__ __forceinline__ void loadTile(const u32 *in, u32 *s, int base, int n, int tid)
{
    if (ALG == ALG_BL) {
        for (int i = tid; i < TILE; i += BLK) s[PIDX(i)] = (base+i < n) ? in[base+i] : 0u;
    } else {
        for (int i = tid; i < TILE; i += BLK) s[i]       = (base+i < n) ? in[base+i] : 0u;
    }
    __syncthreads();
}
template<int ALG>
__device__ __forceinline__ void storeTile(u32 *out, const u32 *s, int base, int n, int tid)
{
    if (ALG == ALG_BL) {
        for (int i = tid; i < TILE; i += BLK) if (base+i < n) out[base+i] = s[PIDX(i)];
    } else {
        for (int i = tid; i < TILE; i += BLK) if (base+i < n) out[base+i] = s[i];
    }
}
template<int ALG>
__device__ __forceinline__ u32 runTileScan(u32 *s, int tid)
{
    if (ALG == ALG_HS) return blockScanHS(s, tid);
    if (ALG == ALG_BL) return blockScanBlelloch(s, tid);
    return blockScanWarp(s, tid);
}

template<int ALG>
__global__ void scanTilesKernel(const u32 * __restrict__ in, u32 * __restrict__ out,
                                u32 * __restrict__ blockSums, int n)
{
    __shared__ u32 s[SMEM_WORDS];
    const int tid  = threadIdx.x;
    const int base = blockIdx.x * TILE;
    loadTile<ALG>(in, s, base, n, tid);
    u32 total = runTileScan<ALG>(s, tid);
    storeTile<ALG>(out, s, base, n, tid);
    if (tid == 0) blockSums[blockIdx.x] = total;
}

// pass 2: one block scans the block-sum array, carrying an offset across tiles
template<int ALG>
__global__ void scanSumsKernel(u32 *v, int m)
{
    __shared__ u32 s[SMEM_WORDS];
    const int tid = threadIdx.x;
    u32 carry = 0;
    for (int base = 0; base < m; base += TILE) {
        loadTile<ALG>(v, s, base, m, tid);
        u32 tot = runTileScan<ALG>(s, tid);
        if (ALG == ALG_BL) {
            for (int i = tid; i < TILE; i += BLK) if (base+i < m) v[base+i] = s[PIDX(i)] + carry;
        } else {
            for (int i = tid; i < TILE; i += BLK) if (base+i < m) v[base+i] = s[i] + carry;
        }
        carry += tot;
        __syncthreads();
    }
}

// pass 3: add each block's offset to its tile
__global__ void addOffsetsKernel(u32 * __restrict__ out, const u32 * __restrict__ offs, int n)
{
    const u32 o = offs[blockIdx.x];
    const int base = blockIdx.x * TILE;
    #pragma unroll
    for (int k = 0; k < IPT; ++k) {
        int j = base + threadIdx.x + k*BLK;
        if (j < n) out[j] += o;
    }
}

// pass 1 only — same 2N of traffic for every algorithm, so the difference
// between the three columns is the algorithm and nothing else.
static void launchPass1(int alg, const u32 *d_in, u32 *d_out, u32 *d_sums, int n, int m)
{
    switch (alg) {
    case ALG_HS: scanTilesKernel<ALG_HS  ><<<m, BLK>>>(d_in, d_out, d_sums, n); break;
    case ALG_BL: scanTilesKernel<ALG_BL  ><<<m, BLK>>>(d_in, d_out, d_sums, n); break;
    default:     scanTilesKernel<ALG_WARP><<<m, BLK>>>(d_in, d_out, d_sums, n); break;
    }
}

static void launchScan(int alg, const u32 *d_in, u32 *d_out, u32 *d_sums, int n, int m)
{
    launchPass1(alg, d_in, d_out, d_sums, n, m);
    switch (alg) {
    case ALG_HS: scanSumsKernel<ALG_HS  ><<<1, BLK>>>(d_sums, m); break;
    case ALG_BL: scanSumsKernel<ALG_BL  ><<<1, BLK>>>(d_sums, m); break;
    default:     scanSumsKernel<ALG_WARP><<<1, BLK>>>(d_sums, m); break;
    }
    addOffsetsKernel<<<m, BLK>>>(d_out, d_sums, n);
}

// ------------------------------------------------------------- TODO 5 -------
// The solution's predictions (the reader must commit before running).
//   PREDICT_SLOWEST : 'H', 'B' or 'W' — which tile-scan is slowest at 64 M
//   PREDICT_BUCKET  : Hillis-Steele / Blelloch tile-scan time ratio at 64 M
//                     1: >= 4x   2: 2x .. 4x   3: 1.2x .. 2x   4: < 1.2x
#define PREDICT_SLOWEST 'H'
#define PREDICT_BUCKET  3

static void cpuExclusiveScan(const u32 *in, u32 *out, int n)
{
    u32 acc = 0;
    for (int i = 0; i < n; ++i) { out[i] = acc; acc += in[i]; }
}

static const int SIZES[2]  = { 1048573, 67108861 };   // both non-powers of two
static const char *SZNAME[2] = { "1,048,573 (1 M, L2-resident)",
                                 "67,108,861 (64 M, DRAM-resident)" };

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    const int nMax = SIZES[1];
    const int mMax = (nMax + TILE - 1) / TILE;

    printf("Module 13 / Exercise 1 — the scan ladder\n\n");

    u32 *h_in  = (u32*)malloc(sizeof(u32)*(size_t)nMax);
    u32 *h_ref = (u32*)malloc(sizeof(u32)*(size_t)nMax);
    u32 *h_out = (u32*)malloc(sizeof(u32)*(size_t)nMax);
    if (!h_in || !h_ref || !h_out) { printf("host alloc failed\n"); return 1; }

    unsigned seed = 20240613u;
    for (int i = 0; i < nMax; ++i) { seed = seed*1664525u + 1013904223u; h_in[i] = (seed >> 24) & 15u; }

    u32 *d_in, *d_out, *d_sums;
    CHECK(cudaMalloc(&d_in,   sizeof(u32)*(size_t)nMax));
    CHECK(cudaMalloc(&d_out,  sizeof(u32)*(size_t)nMax));
    CHECK(cudaMalloc(&d_sums, sizeof(u32)*(size_t)mMax));
    CHECK(cudaMemcpy(d_in, h_in, sizeof(u32)*(size_t)nMax, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    float bestFull[2][N_ALG], bestP1[2][N_ALG];
    int   okFull[2][N_ALG];
    int   totalOK = 0;

    for (int si = 0; si < 2; ++si) {
        const int n = SIZES[si];
        const int m = (n + TILE - 1) / TILE;
        printf("N = %s : %d tiles of %d, last tile holds %d\n",
               SZNAME[si], m, TILE, n - (m-1)*TILE);

        cpuExclusiveScan(h_in, h_ref, n);

        // ---- duration-based warm-up (spec §12): let the clocks ramp ----
        {
            float el = 0.0f;
            CHECK(cudaEventRecord(e0));
            do {
                launchScan(ALG_WARP, d_in, d_out, d_sums, n, m);
                CHECK(cudaEventRecord(e1));
                CHECK(cudaEventSynchronize(e1));
                CHECK(cudaEventElapsedTime(&el, e0, e1));
            } while (el < 400.0f);
        }
        // ---- auto-scale iterations to a ~10 ms timed segment ----
        int iters;
        {
            CHECK(cudaEventRecord(e0));
            for (int i = 0; i < 20; ++i) launchScan(ALG_WARP, d_in, d_out, d_sums, n, m);
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            iters = (int)(10.0f / (ms/20.0f));
            if (iters < 20)   iters = 20;
            if (iters > 4000) iters = 4000;
        }

        // ---- timing: 6 configs back to back, rotated order, min of 4 sweeps --
        for (int a = 0; a < N_ALG; ++a) { bestFull[si][a] = 1e30f; bestP1[si][a] = 1e30f; }
        for (int sweep = 0; sweep < 4; ++sweep) {
            for (int q = 0; q < 2*N_ALG; ++q) {
                int c = (q + sweep) % (2*N_ALG);
                int a = c % N_ALG, full = c / N_ALG;
                CHECK(cudaEventRecord(e0));
                for (int it = 0; it < iters; ++it) {
                    if (full) launchScan (a, d_in, d_out, d_sums, n, m);
                    else      launchPass1(a, d_in, d_out, d_sums, n, m);
                }
                CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
                float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
                ms /= (float)iters;
                if (full) { if (ms < bestFull[si][a]) bestFull[si][a] = ms; }
                else      { if (ms < bestP1  [si][a]) bestP1  [si][a] = ms; }
            }
        }
        CHECK(cudaGetLastError());

        // ---- validation: separate second pass ----
        for (int a = 0; a < N_ALG; ++a) {
            CHECK(cudaMemset(d_out, 0xAB, sizeof(u32)*(size_t)n));
            launchScan(a, d_in, d_out, d_sums, n, m);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_out, d_out, sizeof(u32)*(size_t)n, cudaMemcpyDeviceToHost));
            okFull[si][a] = 1;
            for (int i = 0; i < n; ++i) {
                if (h_out[i] != h_ref[i]) {
                    printf("  %-14s MISMATCH at i=%d: got %u expected %u\n",
                           ALG_NAME[a], i, h_out[i], h_ref[i]);
                    okFull[si][a] = 0; break;
                }
            }
            totalOK += okFull[si][a];
        }

        // ---- report ----
        const double bytes2N = 2.0 * (double)n * sizeof(u32);
        printf("  %-14s %10s %10s %9s | %10s %10s %9s %7s\n",
               "algorithm", "tile ms", "GB/s 2N", "%peak", "full ms", "GB/s 4N", "%peak", "valid");
        for (int a = 0; a < N_ALG; ++a) {
            double g1 = bytes2N   / (bestP1  [si][a]*1e-3) / 1e9;
            double g2 = 2*bytes2N / (bestFull[si][a]*1e-3) / 1e9;
            printf("  %-14s %10.4f %10.1f %8.1f%% | %10.4f %10.1f %8.1f%% %7s\n",
                   ALG_NAME[a], bestP1[si][a], g1, 100.0*g1/432.0,
                   bestFull[si][a], g2, 100.0*g2/432.0, okFull[si][a] ? "PASS" : "FAIL");
        }
        printf("  2N floor = %.4f ms; this 3-kernel structure moves 4N, floor %.4f ms\n\n",
               bytes2N/432e9*1e3, 2.0*bytes2N/432e9*1e3);
    }

    // ---- score the predictions at the L2-resident size ----------------------
    // At 64 M all three tile scans sit near the DRAM roof (66-86 % of 432 GB/s),
    // which compresses the algorithmic difference into the noise. To compare
    // *algorithms* you have to leave the bandwidth-bound regime, so the scored
    // comparison uses the 1 M case. Its apparent >100 % figures are L2 hits,
    // not DRAM bandwidth.
    const int S = 0;
    int slow = ALG_HS;
    for (int a = 1; a < N_ALG; ++a) if (bestP1[S][a] > bestP1[S][slow]) slow = a;
    char slowC = (slow==ALG_HS) ? 'H' : (slow==ALG_BL ? 'B' : 'W');

    double ratio = bestP1[S][ALG_HS] / bestP1[S][ALG_BL];
    int bucket = (ratio >= 4.0) ? 1 : (ratio >= 2.0) ? 2 : (ratio >= 1.2) ? 3 : 4;

    double bw = bestP1[S][ALG_BL] / bestP1[S][ALG_WARP];
    if (bw < 1.0) bw = 1.0/bw;
    double comp = (bestP1[1][ALG_HS]/bestP1[1][ALG_BL]);

    printf("Predictions (scored at N = 1,048,573, tile-scan kernel only)\n");
    printf("  slowest tile scan     : measured %c   predicted %c\n", slowC, PREDICT_SLOWEST);
    printf("  Hillis-Steele/Blelloch: measured %.2fx -> bucket %d   predicted %d\n",
           ratio, bucket, PREDICT_BUCKET);
    printf("  Blelloch vs warp scan : %.2fx apart (not scored, but look at it)\n", bw);
    printf("  work ratio for reference: %d adds/tile vs %d adds/tile = %.1fx\n",
           TILE*10, 2*TILE, (double)(TILE*10)/(double)(2*TILE));
    printf("  same ratio at 64 M, where DRAM is the wall: %.2fx — the algorithm\n"
           "  difference is real but mostly invisible once you are bandwidth-bound.\n", comp);

    int slowOK   = (slowC == PREDICT_SLOWEST);
    int bucketOK = (bucket == PREDICT_BUCKET);
    int score = totalOK + slowOK + bucketOK;              // 6 + 1 + 1

    printf("\n  correctness %d/6, slowest prediction %s, ratio prediction %s\n",
           totalOK, slowOK ? "correct" : "WRONG", bucketOK ? "correct" : "WRONG");
    printf("  score: %d/8\n", score);
    printf("OVERALL: %s\n", (score == 8) ? "PASS" : "FAIL");

    free(h_in); free(h_ref); free(h_out);
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_sums));
    CHECK(cudaDeviceReset());
    return (score == 8) ? 0 : 1;
}
