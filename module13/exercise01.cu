// =============================================================================
// Module 13 / Exercise 1 — the scan ladder
//
// GOAL : Implement three block-level exclusive scans of a 1024-element tile —
//        Hillis-Steele, Blelloch, and a warp-shuffle block scan — and drop each
//        into the same three-kernel device-wide scan. The harness validates all
//        three at two non-power-of-two sizes, times the tile-scan kernel alone
//        (identical global traffic for all three, so the difference is purely
//        the algorithm) and times the full scan, then scores your predictions.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN  : exercise01.exe
//
// Each __device__ scan function receives a pointer to 2*TILE words of shared
// memory and this thread's threadIdx.x, must leave an EXCLUSIVE scan of the
// tile in the layout the matching load/store helper expects, and must return
// the tile total.
//
// TODO 1 — Hillis-Steele
// TODO 2 — Blelloch upsweep
// TODO 3 — Blelloch downsweep
// TODO 4 — warp-shuffle block scan  (design: you choose the decomposition)
// TODO 5 — two predictions, committed before you run
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

// Shared scratch handed to every algorithm: 2*TILE words = 8192 B.
#define SMEM_WORDS (2*TILE)

// Module 7's padding macro. The Blelloch helpers below place element i of the
// tile at PIDX(i); whether that actually helps is the subject of example01.
#define CONFLICT_FREE_OFFSET(i) ((i) >> 5)
#define PIDX(i) ((i) + CONFLICT_FREE_OFFSET(i))

enum { ALG_HS = 0, ALG_BL = 1, ALG_WARP = 2, N_ALG = 3 };
static const char *ALG_NAME[N_ALG] = { "Hillis-Steele", "Blelloch", "warp-shuffle" };

// ============================================================================
// TODO 1 — Hillis-Steele.
//
// On entry s[0 .. TILE) holds the tile. On exit s[0 .. TILE) must hold the
// EXCLUSIVE scan. You have s[TILE .. 2*TILE) as scratch; think about whether
// you need it before you decide you do not. Return the tile total.
//
// Work: O(N log N). Depth: O(log N).
// ============================================================================
__device__ u32 blockScanHS(u32 *s, int tid)
{
    // YOUR CODE HERE
    (void)s; (void)tid;
    return 0u;
}

// ============================================================================
// TODO 2 and TODO 3 — Blelloch, in place, on the PADDED layout: element i of
// the tile lives at s[PIDX(i)].
//
// TODO 2: the upsweep (reduce) phase. After it, s[PIDX(TILE-1)] must hold the
//         sum of the whole tile, and every internal node of the implicit
//         binary tree must hold the sum of its subtree.
// TODO 3: the downsweep phase. Between the two phases exactly one element has
//         to change, and getting that one element wrong produces an answer
//         that is off by a constant everywhere — read the definition of an
//         exclusive scan again and work out which element and what value.
//
// Return the tile total. Note that the downsweep destroys it, so capture it
// before you start.
//
// Work: O(N). Depth: 2*log N.
// ============================================================================
__device__ u32 blockScanBlelloch(u32 *s, int tid)
{
    // ---- TODO 2: upsweep ----
    // YOUR CODE HERE

    // ---- TODO 3: identity element, then downsweep ----
    // YOUR CODE HERE

    (void)s; (void)tid;
    return 0u;
}

// ============================================================================
// TODO 4 — warp-shuffle block scan.  DESIGN TODO.
//
// Requirement, not recipe: produce the same exclusive scan of s[0 .. TILE)
// using no more than BLK/32 words of shared memory beyond the tile itself, and
// no __syncthreads() inside any warp-level step. You choose how the 1024
// elements are divided among the 256 threads and how the partial results are
// combined. `__shfl_up_sync` is available; so is `__shfl_down_sync`.
//
// Two things to get right:
//   - a lane that shuffles from a lane below lane 0 receives its own value,
//     not zero;
//   - converting a per-warp inclusive result into a per-thread exclusive
//     offset needs one subtraction, and it is easy to subtract the wrong thing.
// ============================================================================
__device__ u32 blockScanWarp(u32 *s, int tid)
{
    // YOUR CODE HERE
    (void)s; (void)tid;
    return 0u;
}

// ---------------------------- harness below: do not modify ------------------
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

// ============================================================================
// TODO 5 — predictions. Commit BEFORE you run the program.
//
// PREDICT_SLOWEST : which tile scan is slowest? 'H', 'B' or 'W'.
// PREDICT_BUCKET  : the Hillis-Steele / Blelloch tile-scan time RATIO at
//                   N = 1,048,573, as a bucket:
//                     1 : Blelloch at least 4x faster
//                     2 : 2x .. 4x
//                     3 : 1.2x .. 2x
//                     4 : less than 1.2x apart (either direction)
//
// Before you pick a bucket, count the additions each algorithm performs on a
// 1024-element tile, and write the two numbers down. Then decide whether you
// expect the time ratio to match the work ratio, and why or why not.
// ============================================================================
#define PREDICT_SLOWEST '?'
#define PREDICT_BUCKET  0

static void cpuExclusiveScan(const u32 *in, u32 *out, int n)
{
    u32 acc = 0;
    for (int i = 0; i < n; ++i) { out[i] = acc; acc += in[i]; }
}

static const int SIZES[2]  = { 1048573, 67108861 };
static const char *SZNAME[2] = { "1,048,573 (1 M, L2-resident)",
                                 "67,108,861 (64 M, DRAM-resident)" };

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    if (PREDICT_SLOWEST == '?' || PREDICT_BUCKET == 0) {
        printf("Set TODO 5 (PREDICTION) first.\n");
        return 0;
    }

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
    printf("  same ratio at 64 M, where DRAM is the wall: %.2fx\n", comp);

    int slowOK   = (slowC == PREDICT_SLOWEST);
    int bucketOK = (bucket == PREDICT_BUCKET);
    int score = totalOK + slowOK + bucketOK;

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
