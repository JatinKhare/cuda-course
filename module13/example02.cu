// =============================================================================
// Module 13 / Example 2 — multi-block scan strategies and the traffic ledger
//
// Four device-wide exclusive scans of the same array, differing only in how the
// per-tile results are stitched together:
//
//   A  scan-then-propagate, 3 kernels   traffic 4N   (read N, write N; read N, write N)
//   B  reduce-then-scan,    3 kernels   traffic 3N   (read N; read N, write N)
//   C  decoupled look-back, 1 kernel    traffic 2N   (read N, write N)      <- the floor
//   D  cub::DeviceScan::ExclusiveSum    traffic 2N   (what you would ship)
//
// The point of the example is that the ranking is predicted by the traffic
// column before any of them is run, because scan is bandwidth-bound.
//
// BUILD: nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler /Zc:preprocessor -o example02.exe example02.cu
//        (the two extra flags are for <cub/cub.cuh> with MSVC; on Linux
//         `nvcc -arch=sm_89 -O3 -o example02 example02.cu` is enough)
// RUN  : example02.exe
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

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

#define FLAG_X 0u
#define FLAG_A 1u
#define FLAG_P 2u

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

// ---- A: scan-then-propagate -------------------------------------------------
__global__ void scanTilesKernel(const u32 * __restrict__ in, u32 * __restrict__ out,
                                u32 * __restrict__ blockSums, int n)
{
    __shared__ u32 s[TILE];
    const int tid = threadIdx.x, base = blockIdx.x*TILE;
    for (int i = tid; i < TILE; i += BLK) s[i] = (base+i<n) ? in[base+i] : 0u;
    __syncthreads();
    u32 total = blockScanWarp(s, tid);
    for (int i = tid; i < TILE; i += BLK) if (base+i<n) out[base+i] = s[i];
    if (tid == 0) blockSums[blockIdx.x] = total;
}
__global__ void scanSumsKernel(u32 *v, int m)
{
    __shared__ u32 s[TILE];
    const int tid = threadIdx.x;
    u32 carry = 0;
    for (int base = 0; base < m; base += TILE) {
        for (int i = tid; i < TILE; i += BLK) s[i] = (base+i<m) ? v[base+i] : 0u;
        __syncthreads();
        u32 tot = blockScanWarp(s, tid);
        for (int i = tid; i < TILE; i += BLK) if (base+i<m) v[base+i] = s[i] + carry;
        carry += tot;
        __syncthreads();
    }
}
__global__ void addOffsetsKernel(u32 * __restrict__ out, const u32 * __restrict__ offs, int n)
{
    const u32 o = offs[blockIdx.x];
    const int base = blockIdx.x*TILE;
    #pragma unroll
    for (int k = 0; k < IPT; ++k) { int j = base + threadIdx.x + k*BLK; if (j<n) out[j] += o; }
}

// ---- B: reduce-then-scan ----------------------------------------------------
__global__ void reduceTilesKernel(const u32 * __restrict__ in, u32 * __restrict__ blockSums, int n)
{
    __shared__ u32 wr[BLK/32];
    const int tid = threadIdx.x, base = blockIdx.x*TILE;
    u32 acc = 0;
    #pragma unroll
    for (int k = 0; k < IPT; ++k) { int j = base + tid + k*BLK; if (j<n) acc += in[j]; }
    // Module 12 owns this: warp reduction, then 8 partials, then one thread.
    #pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if ((tid & 31) == 0) wr[tid>>5] = acc;
    __syncthreads();
    if (tid == 0) { u32 t = 0; for (int i = 0; i < BLK/32; ++i) t += wr[i];
                    blockSums[blockIdx.x] = t; }
}
__global__ void scanWithOffsetKernel(const u32 * __restrict__ in, u32 * __restrict__ out,
                                     const u32 * __restrict__ offs, int n)
{
    __shared__ u32 s[TILE];
    const int tid = threadIdx.x, base = blockIdx.x*TILE;
    for (int i = tid; i < TILE; i += BLK) s[i] = (base+i<n) ? in[base+i] : 0u;
    __syncthreads();
    blockScanWarp(s, tid);
    const u32 o = offs[blockIdx.x];
    for (int i = tid; i < TILE; i += BLK) if (base+i<n) out[base+i] = s[i] + o;
}

// ---- C: decoupled look-back -------------------------------------------------
__global__ void dlbInitKernel(u32 * __restrict__ flags, u32 * __restrict__ ticket, int m)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < m) flags[i] = FLAG_X;
    if (i == 0) *ticket = 0u;
}
__global__ void dlbScanKernel(const u32 * __restrict__ in, u32 * __restrict__ out,
                              u32 *flags, u32 *aggs, u32 *pfxs, u32 *ticket, int n, int m)
{
    __shared__ u32 s[TILE];
    __shared__ u32 s_tile, s_excl;
    const int tid = threadIdx.x;

    if (tid == 0) s_tile = atomicAdd(ticket, 1u);   // dynamic tile index
    __syncthreads();
    const int tile = (int)s_tile;
    if (tile >= m) return;

    const int base = tile*TILE;
    for (int i = tid; i < TILE; i += BLK) s[i] = (base+i<n) ? in[base+i] : 0u;
    __syncthreads();
    const u32 total = blockScanWarp(s, tid);

    if (tid == 0) {
        if (tile == 0) { pfxs[0] = total; __threadfence(); atomicExch(&flags[0], FLAG_P); s_excl = 0u; }
        else           { aggs[tile] = total; __threadfence(); atomicExch(&flags[tile], FLAG_A); }
    }
    __syncthreads();

    if (tile > 0) {
        if (tid < 32) {
            const int lane = tid;
            u32 excl = 0u;
            int look = tile - 1;
            while (true) {
                int j = look - lane;
                u32 f, v;
                if (j >= 0) {
                    do { f = atomicAdd(&flags[j], 0u); } while (f == FLAG_X);
                    __threadfence();
                    v = (f == FLAG_P) ? pfxs[j] : aggs[j];
                } else { f = FLAG_P; v = 0u; }
                u32 pmask = __ballot_sync(0xffffffffu, f == FLAG_P);
                int firstP = pmask ? (__ffs((int)pmask) - 1) : 32;
                u32 c = (lane <= firstP) ? v : 0u;
                #pragma unroll
                for (int o = 16; o; o >>= 1) c += __shfl_down_sync(0xffffffffu, c, o);
                c = __shfl_sync(0xffffffffu, c, 0);
                excl += c;
                if (pmask) break;
                look -= 32;
            }
            if (lane == 0) {
                pfxs[tile] = excl + total;
                __threadfence();
                atomicExch(&flags[tile], FLAG_P);
                s_excl = excl;
            }
        }
        __syncthreads();
    }
    const u32 off = s_excl;
    for (int i = tid; i < TILE; i += BLK) if (base+i<n) out[base+i] = s[i] + off;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    const int n = 67108861;                 // not a multiple of TILE
    const int m = (n + TILE - 1)/TILE;

    printf("Module 13 / Example 2 — multi-block scan strategies\n");
    printf("N = %d, %d tiles of %d (last tile holds %d)\n\n", n, m, TILE, n-(m-1)*TILE);

    u32 *h_in  = (u32*)malloc(sizeof(u32)*(size_t)n);
    u32 *h_ref = (u32*)malloc(sizeof(u32)*(size_t)n);
    u32 *h_out = (u32*)malloc(sizeof(u32)*(size_t)n);
    if (!h_in || !h_ref || !h_out) { printf("host alloc failed\n"); return 1; }
    unsigned seed = 5150u;
    for (int i = 0; i < n; ++i) { seed = seed*1664525u + 1013904223u; h_in[i] = (seed>>26) & 3u; }
    { u32 acc = 0; for (int i = 0; i < n; ++i) { h_ref[i] = acc; acc += h_in[i]; } }

    u32 *d_in, *d_out, *d_sums, *d_f, *d_a, *d_p, *d_t;
    CHECK(cudaMalloc(&d_in,   sizeof(u32)*(size_t)n));
    CHECK(cudaMalloc(&d_out,  sizeof(u32)*(size_t)n));
    CHECK(cudaMalloc(&d_sums, sizeof(u32)*(size_t)m));
    CHECK(cudaMalloc(&d_f,    sizeof(u32)*(size_t)m));
    CHECK(cudaMalloc(&d_a,    sizeof(u32)*(size_t)m));
    CHECK(cudaMalloc(&d_p,    sizeof(u32)*(size_t)m));
    CHECK(cudaMalloc(&d_t,    sizeof(u32)));
    CHECK(cudaMemcpy(d_in, h_in, sizeof(u32)*(size_t)n, cudaMemcpyHostToDevice));

    void *d_temp = NULL; size_t tempBytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, tempBytes, d_in, d_out, n);
    CHECK(cudaMalloc(&d_temp, tempBytes));
    printf("cub::DeviceScan temp storage for N = %d: %zu bytes (%.1f B per tile)\n\n",
           n, tempBytes, (double)tempBytes/(double)m);

    enum { C_STP = 0, C_RTS = 1, C_DLB = 2, C_CUB = 3, NC = 4 };
    const char *CN[NC] = { "A scan-then-propagate", "B reduce-then-scan",
                           "C decoupled look-back", "D cub::DeviceScan" };
    const double TRAFFIC[NC] = { 4.0, 3.0, 2.0, 2.0 };

    #define RUN(c) do {                                                         \
        if ((c) == C_STP) {                                                     \
            scanTilesKernel<<<m,BLK>>>(d_in, d_out, d_sums, n);                 \
            scanSumsKernel<<<1,BLK>>>(d_sums, m);                               \
            addOffsetsKernel<<<m,BLK>>>(d_out, d_sums, n);                      \
        } else if ((c) == C_RTS) {                                              \
            reduceTilesKernel<<<m,BLK>>>(d_in, d_sums, n);                      \
            scanSumsKernel<<<1,BLK>>>(d_sums, m);                               \
            scanWithOffsetKernel<<<m,BLK>>>(d_in, d_out, d_sums, n);            \
        } else if ((c) == C_DLB) {                                              \
            dlbInitKernel<<<(m+255)/256,256>>>(d_f, d_t, m);                    \
            dlbScanKernel<<<m,BLK>>>(d_in, d_out, d_f, d_a, d_p, d_t, n, m);    \
        } else {                                                                \
            cub::DeviceScan::ExclusiveSum(d_temp, tempBytes, d_in, d_out, n);   \
        }                                                                       \
    } while (0)

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    {
        float el = 0.0f;
        CHECK(cudaEventRecord(e0));
        do { RUN(C_CUB); CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
             CHECK(cudaEventElapsedTime(&el, e0, e1)); } while (el < 400.0f);
    }
    int iters;
    {
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < 20; ++i) RUN(C_CUB);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        iters = (int)(10.0f/(ms/20.0f)); if (iters < 20) iters = 20; if (iters > 2000) iters = 2000;
    }
    float best[NC];
    for (int c = 0; c < NC; ++c) best[c] = 1e30f;
    for (int sweep = 0; sweep < 4; ++sweep) {
        for (int q = 0; q < NC; ++q) {
            int c = (q + sweep) % NC;
            CHECK(cudaEventRecord(e0));
            for (int it = 0; it < iters; ++it) RUN(c);
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms/iters < best[c]) best[c] = ms/(float)iters;
        }
    }
    CHECK(cudaGetLastError());

    int ok[NC];
    for (int c = 0; c < NC; ++c) {
        CHECK(cudaMemset(d_out, 0xAB, sizeof(u32)*(size_t)n));
        RUN(c);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out, d_out, sizeof(u32)*(size_t)n, cudaMemcpyDeviceToHost));
        ok[c] = 1;
        for (int i = 0; i < n; ++i) if (h_out[i] != h_ref[i]) { ok[c] = 0; break; }
    }

    const double N4 = (double)n*sizeof(u32);
    const double floor2N = 2.0*N4/432e9*1e3;
    printf("%-23s %8s %8s %11s %11s %9s %6s\n",
           "strategy", "traffic", "ms", "GB/s vs 2N", "GB/s real", "% of 432", "valid");
    for (int c = 0; c < NC; ++c) {
        double g2 = 2.0*N4/(best[c]*1e-3)/1e9;
        double gr = TRAFFIC[c]*N4/(best[c]*1e-3)/1e9;
        printf("%-23s %6.0fN %8.4f %11.1f %11.1f %8.1f%% %6s\n",
               CN[c], TRAFFIC[c], best[c], g2, gr, 100.0*gr/432.0, ok[c] ? "ok" : "BAD");
    }
    printf("\n  2N floor at 432 GB/s = %.4f ms; nothing can be faster than this.\n", floor2N);
    printf("  \"GB/s vs 2N\" is the useful column: it is the rate at which the scan\n"
           "  delivers ANSWERS. \"GB/s real\" is the rate at which each strategy moves\n"
           "  the bytes it chose to move, and shows that A and B are already at the\n"
           "  DRAM roof -- they are not slow kernels, they are kernels doing too much\n"
           "  I/O.\n\n");
    printf("  ratios (stable to ~1%%, absolutes are not):\n");
    for (int c = 0; c < NC; ++c)
        printf("    %-23s %.2fx of cub::DeviceScan\n", CN[c], best[c]/best[C_CUB]);

    free(h_in); free(h_ref); free(h_out);
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_sums));
    CHECK(cudaFree(d_f)); CHECK(cudaFree(d_a)); CHECK(cudaFree(d_p)); CHECK(cudaFree(d_t));
    CHECK(cudaFree(d_temp));
    CHECK(cudaDeviceReset());
    return 0;
}
