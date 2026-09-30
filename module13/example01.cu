// =============================================================================
// Module 13 / Example 1 — definitions, the ladder, and the bank-conflict question
//
// A: inclusive vs exclusive scan on 16 elements, and the conversion both ways.
// B: three block-scan algorithms measured on the same tile-scan kernel, so the
//    only difference between the columns is the algorithm.
// C: the Blelloch tree with and without the classic CONFLICT_FREE_OFFSET
//    padding, measured, against what Module 7's max(2,D) cost law predicts.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN  : example01.exe
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
#define SMEM_WORDS (2*TILE)

#define CONFLICT_FREE_OFFSET(i) ((i) >> 5)

// ---------------------------------------------------------------- Part A ----
// The two scans, on the host, so the definition is unambiguous.
static void hostInclusive(const u32 *in, u32 *out, int n)
{
    u32 acc = 0;
    for (int i = 0; i < n; ++i) { acc += in[i]; out[i] = acc; }   // includes in[i]
}
static void hostExclusive(const u32 *in, u32 *out, int n)
{
    u32 acc = 0;
    for (int i = 0; i < n; ++i) { out[i] = acc; acc += in[i]; }   // excludes in[i]
}

static void partA(void)
{
    const int n = 16;
    u32 in[16], inc[16], exc[16];
    for (int i = 0; i < n; ++i) in[i] = (u32)(i % 5) + 1u;

    hostInclusive(in, inc, n);
    hostExclusive(in, exc, n);

    printf("=== A. inclusive vs exclusive ===\n");
    printf("  %-10s", "input");    for (int i=0;i<n;++i) printf("%4u", in[i]);  printf("\n");
    printf("  %-10s", "inclusive");for (int i=0;i<n;++i) printf("%4u", inc[i]); printf("\n");
    printf("  %-10s", "exclusive");for (int i=0;i<n;++i) printf("%4u", exc[i]); printf("\n");

    // Conversions. Both are one-liners and both have a trap.
    //   inclusive[i] = exclusive[i] + in[i]          -- always safe
    //   exclusive[i] = inclusive[i] - in[i]          -- safe for a group with
    //                                                   an inverse; NOT for max
    //   exclusive[i] = inclusive[i-1], exclusive[0]=0 -- a shift; safe for any
    //                                                   associative operator,
    //                                                   but it LOSES the total,
    //                                                   which now lives only in
    //                                                   inclusive[n-1].
    int okA = 1, okB = 1;
    for (int i = 0; i < n; ++i) if (exc[i] + in[i] != inc[i]) okA = 0;
    for (int i = 0; i < n; ++i) { u32 e = (i == 0) ? 0u : inc[i-1]; if (e != exc[i]) okB = 0; }
    printf("  exclusive[i] + in[i] == inclusive[i] : %s\n", okA ? "yes" : "no");
    printf("  shift-by-one of inclusive == exclusive: %s\n", okB ? "yes" : "no");
    printf("  total = %u; it is inclusive[n-1], and exclusive[n-1] = %u misses the\n"
           "  last element, which is the single most common scan bug.\n\n",
           inc[n-1], exc[n-1]);

    // Why exclusive is the useful one: it IS the output offset.
    printf("  treat the input as \"how many items element i emits\":\n");
    printf("  element 3 emits %u items and they start at output slot %u = exclusive[3].\n",
           in[3], exc[3]);
    printf("  the inclusive scan would tell you where element 3's run ENDS, which is\n"
           "  not what a writer needs.\n\n");
}

// ---------------------------------------------------------------- Part B ----
enum { ALG_HS = 0, ALG_BL_PAD = 1, ALG_BL_RAW = 2, ALG_WARP = 3, N_ALG = 4 };
static const char *ALG_NAME[N_ALG] = { "Hillis-Steele", "Blelloch +pad",
                                       "Blelloch raw ", "warp-shuffle " };

__device__ u32 blockScanHS(u32 *s, int tid)
{
    int cur = 0;
    for (int off = 1; off < TILE; off <<= 1) {
        for (int i = tid; i < TILE; i += BLK)
            s[(cur^1)*TILE + i] = s[cur*TILE + i] + ((i >= off) ? s[cur*TILE + i - off] : 0u);
        __syncthreads();
        cur ^= 1;
    }
    u32 total = s[cur*TILE + TILE - 1];
    u32 v[IPT];
    #pragma unroll
    for (int k = 0; k < IPT; ++k) { int i = tid + k*BLK; v[k] = (i==0) ? 0u : s[cur*TILE + i - 1]; }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < IPT; ++k) s[tid + k*BLK] = v[k];
    __syncthreads();
    return total;
}

template<bool PAD>
__device__ __forceinline__ int pidx(int i) { return PAD ? (i + CONFLICT_FREE_OFFSET(i)) : i; }

template<bool PAD>
__device__ u32 blockScanBlelloch(u32 *s, int tid)
{
    int offset = 1;
    for (int d = TILE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        for (int t = tid; t < d; t += BLK) {
            int ai = offset*(2*t+1)-1, bi = offset*(2*t+2)-1;
            s[pidx<PAD>(bi)] += s[pidx<PAD>(ai)];
        }
        offset <<= 1;
    }
    __syncthreads();
    u32 total = s[pidx<PAD>(TILE-1)];
    if (tid == 0) s[pidx<PAD>(TILE-1)] = 0u;
    for (int d = 1; d < TILE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        for (int t = tid; t < d; t += BLK) {
            int ai = offset*(2*t+1)-1, bi = offset*(2*t+2)-1;
            u32 tmp = s[pidx<PAD>(ai)];
            s[pidx<PAD>(ai)]  = s[pidx<PAD>(bi)];
            s[pidx<PAD>(bi)] += tmp;
        }
    }
    __syncthreads();
    return total;
}

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
    u32 wexcl = (wid == 0) ? 0u : warpTot[wid-1];
    u32 texcl = wexcl + wincl - run;
    #pragma unroll
    for (int k = 0; k < IPT; ++k) s[tid*IPT + k] = x[k] + texcl;
    __syncthreads();
    return warpTot[BLK/32 - 1];
}

template<int ALG>
__global__ void tileScanKernel(const u32 * __restrict__ in, u32 * __restrict__ out, int n)
{
    __shared__ u32 s[SMEM_WORDS];
    const int tid = threadIdx.x, base = blockIdx.x * TILE;
    if (ALG == ALG_BL_PAD) {
        for (int i = tid; i < TILE; i += BLK) s[pidx<true >(i)] = (base+i<n) ? in[base+i] : 0u;
    } else {
        for (int i = tid; i < TILE; i += BLK) s[i]              = (base+i<n) ? in[base+i] : 0u;
    }
    __syncthreads();

    if      (ALG == ALG_HS)     blockScanHS(s, tid);
    else if (ALG == ALG_BL_PAD) blockScanBlelloch<true >(s, tid);
    else if (ALG == ALG_BL_RAW) blockScanBlelloch<false>(s, tid);
    else                        blockScanWarp(s, tid);

    if (ALG == ALG_BL_PAD) {
        for (int i = tid; i < TILE; i += BLK) if (base+i<n) out[base+i] = s[pidx<true>(i)];
    } else {
        for (int i = tid; i < TILE; i += BLK) if (base+i<n) out[base+i] = s[i];
    }
}

static void launchTile(int alg, const u32 *d_in, u32 *d_out, int n, int m)
{
    switch (alg) {
    case ALG_HS:     tileScanKernel<ALG_HS    ><<<m,BLK>>>(d_in,d_out,n); break;
    case ALG_BL_PAD: tileScanKernel<ALG_BL_PAD><<<m,BLK>>>(d_in,d_out,n); break;
    case ALG_BL_RAW: tileScanKernel<ALG_BL_RAW><<<m,BLK>>>(d_in,d_out,n); break;
    default:         tileScanKernel<ALG_WARP  ><<<m,BLK>>>(d_in,d_out,n); break;
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    partA();

    const int SIZES[2] = { 1048573, 67108861 };
    const char *SZN[2] = { "1,048,573  (8 MB of traffic: L2-RESIDENT)",
                           "67,108,861 (512 MB of traffic: DRAM)" };
    const int nMax = SIZES[1];

    u32 *h_in  = (u32*)malloc(sizeof(u32)*(size_t)nMax);
    u32 *h_ref = (u32*)malloc(sizeof(u32)*(size_t)nMax);
    u32 *h_out = (u32*)malloc(sizeof(u32)*(size_t)nMax);
    if (!h_in || !h_ref || !h_out) { printf("host alloc failed\n"); return 1; }
    unsigned seed = 31337u;
    for (int i = 0; i < nMax; ++i) { seed = seed*1664525u + 1013904223u; h_in[i] = (seed>>24) & 15u; }

    u32 *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  sizeof(u32)*(size_t)nMax));
    CHECK(cudaMalloc(&d_out, sizeof(u32)*(size_t)nMax));
    CHECK(cudaMemcpy(d_in, h_in, sizeof(u32)*(size_t)nMax, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    printf("=== B and C. four tile-scan kernels, identical 2N of global traffic ===\n");
    printf("    (only the shared-memory algorithm differs between columns)\n\n");

    float best[2][N_ALG];

    for (int si = 0; si < 2; ++si) {
        const int n = SIZES[si], m = (n + TILE - 1)/TILE;

        float el = 0.0f;
        CHECK(cudaEventRecord(e0));
        do { launchTile(ALG_WARP, d_in, d_out, n, m);
             CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
             CHECK(cudaEventElapsedTime(&el, e0, e1)); } while (el < 400.0f);

        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < 20; ++i) launchTile(ALG_WARP, d_in, d_out, n, m);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        int iters = (int)(10.0f/(ms/20.0f));
        if (iters < 20) iters = 20; if (iters > 4000) iters = 4000;

        for (int a = 0; a < N_ALG; ++a) best[si][a] = 1e30f;
        for (int sweep = 0; sweep < 4; ++sweep) {
            for (int q = 0; q < N_ALG; ++q) {
                int a = (q + sweep) % N_ALG;
                CHECK(cudaEventRecord(e0));
                for (int it = 0; it < iters; ++it) launchTile(a, d_in, d_out, n, m);
                CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
                float t; CHECK(cudaEventElapsedTime(&t, e0, e1));
                if (t/iters < best[si][a]) best[si][a] = t/(float)iters;
            }
        }
        CHECK(cudaGetLastError());

        // validate every column, in a second pass (spec §12)
        int ok[N_ALG];
        for (int a = 0; a < N_ALG; ++a) {
            CHECK(cudaMemset(d_out, 0xAB, sizeof(u32)*(size_t)n));
            launchTile(a, d_in, d_out, n, m);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_out, d_out, sizeof(u32)*(size_t)n, cudaMemcpyDeviceToHost));
            ok[a] = 1;
            for (int t = 0; t < m && ok[a]; ++t) {
                int base = t*TILE, hi = (base+TILE < n) ? base+TILE : n;
                u32 acc = 0;
                for (int i = base; i < hi; ++i) { if (h_out[i] != acc) { ok[a] = 0; break; } acc += h_in[i]; }
            }
        }

        const double b2n = 2.0*(double)n*sizeof(u32);
        printf("  N = %s\n", SZN[si]);
        printf("  %-14s %9s %10s %9s %9s %6s\n",
               "algorithm", "ms", "GB/s", "% of 432", "vs warp", "valid");
        for (int a = 0; a < N_ALG; ++a) {
            double g = b2n/(best[si][a]*1e-3)/1e9;
            printf("  %-14s %9.4f %10.1f %8.1f%% %8.2fx %6s\n",
                   ALG_NAME[a], best[si][a], g, 100.0*g/432.0,
                   best[si][a]/best[si][ALG_WARP], ok[a] ? "ok" : "BAD");
        }
        printf("  2N floor = %.4f ms\n", b2n/432e9*1e3);
        printf("  Blelloch raw / Blelloch padded = %.2fx  <- what the padding bought\n\n",
               best[si][ALG_BL_RAW]/best[si][ALG_BL_PAD]);
    }

    printf("Reading the bank-conflict result against Module 7\n");
    printf("  The Blelloch tree touches s[offset*(2t+2)-1]. At level L the stride\n");
    printf("  between the addresses a warp uses is 2^(L+1) words, so the conflict\n");
    printf("  degree is D = min(2^(L+1), 32): levels 0..3 give D = 2,4,8,16 and every\n");
    printf("  level from 4 up gives D = 32. Module 7 measured the cost law on Ada as\n");
    printf("  max(2,D), not D, so the D=2 level is genuinely free and the padding buys\n");
    printf("  nothing there -- but it is the only level that is free. Nine of the ten\n");
    printf("  upsweep levels and nine of the ten downsweep levels are conflicted, and\n");
    printf("  the measured win above is the honest total.\n");
    printf("  Padding costs %d extra shared words per block (%d B), which does not\n",
           TILE/32, (int)(TILE/32*sizeof(u32)));
    printf("  change blocks/SM here because %d B + 1024 B reserve still allows the\n",
           (int)(SMEM_WORDS*sizeof(u32)));
    printf("  6 blocks/SM that 256 threads already cap us at.\n");

    free(h_in); free(h_ref); free(h_out);
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    CHECK(cudaDeviceReset());
    return 0;
}
