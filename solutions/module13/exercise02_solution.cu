// =============================================================================
// Module 13 / Exercise 2 — SOLUTION — ordered stream compaction
//
// GOAL : Produce the list of surviving indices IN INPUT ORDER, deterministically,
//        using an exclusive scan of the predicate; compare against Module 10's
//        atomicAdd ticket compaction, which is faster and unordered.
//
// BUILD: nvcc -arch=sm_89 -O3 -o ex2sol.exe exercise02_solution.cu
// RUN  : ex2sol.exe
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

#define N_ELEMS 33554393          /* prime-ish, not a multiple of TILE */

// The predicate. Keep element i when (data[i] & 7) < 3  => about 37.5 % survive.
__host__ __device__ __forceinline__ u32 keepPred(u32 v) { return ((v & 7u) < 3u) ? 1u : 0u; }

// ----------------------------------------------------------------------------
// Block scan machinery, reused verbatim from Exercise 1 (given, not a TODO).
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
// The unordered reference: Module 10's atomic ticket. Given, complete.
// ----------------------------------------------------------------------------
__global__ void atomicCompactKernel(const u32 * __restrict__ data, u32 * __restrict__ out,
                                    u32 * __restrict__ counter, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        if (keepPred(data[i])) {
            u32 slot = atomicAdd(counter, 1u);   // returns the OLD value (M10)
            out[slot] = (u32)i;
        }
    }
}

// ---------------------------------------------------------------- TODO 1 ----
// Materialise the predicate as a 0/1 array.
__global__ void predicateKernel(const u32 * __restrict__ data, u32 * __restrict__ flags, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) flags[i] = keepPred(data[i]);
}

// ---------------------------------------------------------------- TODO 3 ----
// Scatter: element i goes to slot offsets[i], which is the EXCLUSIVE scan of
// the flags — i.e. the number of survivors strictly before i.
__global__ void scatterKernel(const u32 * __restrict__ flags, const u32 * __restrict__ offsets,
                              u32 * __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) if (flags[i]) out[offsets[i]] = (u32)i;
}

// ---------------------------------------------------------------- TODO 2 ----
// Device-wide exclusive scan, three-kernel scan-then-propagate.
// Traffic ledger: pass 1 reads N and writes N, pass 3 reads N and writes N.
// 4N total. `scanSumsKernel` already loops over its input, so m > TILE is fine.
static void deviceExclusiveScan(const u32 *d_in, u32 *d_out, u32 *d_sums, int n)
{
    const int m = (n + TILE - 1) / TILE;
    scanTilesKernel<<<m, BLK>>>(d_in, d_out, d_sums, n);
    scanSumsKernel<<<1, BLK>>>(d_sums, m);
    addOffsetsKernel<<<m, BLK>>>(d_out, d_sums, n);
}

// --------------------------------------------------- solution-only variant --
// Fused: never materialise the flags. The tile scan applies the predicate as it
// loads, and the offset-add pass scatters directly. Traffic drops from ~8N to
// ~4.4N. This is what TODO 2's design freedom is really worth.
__global__ void scanPredTilesKernel(const u32 * __restrict__ data, u32 * __restrict__ offs,
                                    u32 * __restrict__ blockSums, int n)
{
    __shared__ u32 s[TILE];
    const int tid = threadIdx.x, base = blockIdx.x * TILE;
    for (int i = tid; i < TILE; i += BLK) s[i] = (base+i < n) ? keepPred(data[base+i]) : 0u;
    __syncthreads();
    u32 total = blockScanWarp(s, tid);
    for (int i = tid; i < TILE; i += BLK) if (base+i < n) offs[base+i] = s[i];
    if (tid == 0) blockSums[blockIdx.x] = total;
}
__global__ void addAndScatterKernel(const u32 * __restrict__ data, const u32 * __restrict__ offs,
                                    const u32 * __restrict__ blockOff, u32 * __restrict__ out, int n)
{
    const u32 o = blockOff[blockIdx.x];
    const int base = blockIdx.x * TILE;
    #pragma unroll
    for (int k = 0; k < IPT; ++k) {
        int i = base + threadIdx.x + k*BLK;
        if (i < n && keepPred(data[i])) out[offs[i] + o] = (u32)i;
    }
}

// ---------------------------------------------------------------- TODO 5 ----
// Predicted atomic/scan-based time ratio bucket, measured on the version the
// reader implements (flags + 3-kernel scan + scatter):
//   1: atomics >= 3x faster   2: 1.5x .. 3x   3: 1.05x .. 1.5x
//   4: within 1.05x either way 5: scan-based is faster by >= 1.05x
#define PREDICT_BUCKET 1

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    const int n = N_ELEMS;
    const int m = (n + TILE - 1) / TILE;
    const int gsGrid = 40 * 6;              // wave-sized grid-stride grid

    printf("Module 13 / Exercise 2 — ordered stream compaction\n");
    printf("N = %d (%d tiles, last tile holds %d)\n\n", n, m, n - (m-1)*TILE);

    u32 *h_data = (u32*)malloc(sizeof(u32)*(size_t)n);
    if (!h_data) { printf("host alloc failed\n"); return 1; }
    unsigned seed = 987654321u;
    for (int i = 0; i < n; ++i) { seed = seed*1664525u + 1013904223u; h_data[i] = seed >> 13; }

    // CPU reference: the ordered list of surviving indices.
    u32 *h_ref = (u32*)malloc(sizeof(u32)*(size_t)n);
    int  refCount = 0;
    for (int i = 0; i < n; ++i) if (keepPred(h_data[i])) h_ref[refCount++] = (u32)i;
    printf("survivors: %d of %d (%.2f%%)\n\n", refCount, n, 100.0*refCount/n);

    u32 *d_data, *d_flags, *d_offs, *d_sums, *d_out, *d_cnt;
    CHECK(cudaMalloc(&d_data,  sizeof(u32)*(size_t)n));
    CHECK(cudaMalloc(&d_flags, sizeof(u32)*(size_t)n));
    CHECK(cudaMalloc(&d_offs,  sizeof(u32)*(size_t)n));
    CHECK(cudaMalloc(&d_sums,  sizeof(u32)*(size_t)m));
    CHECK(cudaMalloc(&d_out,   sizeof(u32)*(size_t)n));
    CHECK(cudaMalloc(&d_cnt,   sizeof(u32)));
    CHECK(cudaMemcpy(d_data, h_data, sizeof(u32)*(size_t)n, cudaMemcpyHostToDevice));

    u32 *h_out = (u32*)malloc(sizeof(u32)*(size_t)n);
    u32 *h_out2= (u32*)malloc(sizeof(u32)*(size_t)n);
    unsigned char *seen = (unsigned char*)malloc((size_t)n);
    if (!h_ref || !h_out || !h_out2 || !seen) { printf("host alloc failed\n"); return 1; }

    // ---------------- the three runnable configurations ----------------------
    // 0 = atomic (unordered), 1 = flags + 3-kernel scan + scatter, 2 = fused
    enum { C_ATOMIC = 0, C_SCAN = 1, C_FUSED = 2, NC = 3 };
    const char *CNAME[NC] = { "atomic ticket (unordered)",
                              "scan: flags+scan+scatter",
                              "scan: fused (solution only)" };

    // ---------------------------------------------------------------- TODO 4
    // The total count. The classic off-by-one: the exclusive scan's last entry
    // counts survivors strictly before n-1, so the last element's own flag has
    // to be added back.
    // (computed inside runConfig below)

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    #define RUN_CONFIG(c) do {                                                    \
        if ((c) == C_ATOMIC) {                                                    \
            CHECK(cudaMemset(d_cnt, 0, sizeof(u32)));                             \
            atomicCompactKernel<<<gsGrid, BLK>>>(d_data, d_out, d_cnt, n);        \
        } else if ((c) == C_SCAN) {                                               \
            predicateKernel<<<gsGrid, BLK>>>(d_data, d_flags, n);                 \
            deviceExclusiveScan(d_flags, d_offs, d_sums, n);                      \
            scatterKernel<<<gsGrid, BLK>>>(d_flags, d_offs, d_out, n);            \
        } else {                                                                  \
            scanPredTilesKernel<<<m, BLK>>>(d_data, d_offs, d_sums, n);           \
            scanSumsKernel<<<1, BLK>>>(d_sums, m);                                \
            addAndScatterKernel<<<m, BLK>>>(d_data, d_offs, d_sums, d_out, n);    \
        }                                                                         \
    } while (0)

    // ---- warm-up ----
    {
        float el = 0.0f;
        CHECK(cudaEventRecord(e0));
        do { RUN_CONFIG(C_SCAN);
             CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
             CHECK(cudaEventElapsedTime(&el, e0, e1)); } while (el < 400.0f);
    }
    int iters;
    {
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < 20; ++i) RUN_CONFIG(C_SCAN);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        iters = (int)(10.0f / (ms/20.0f));
        if (iters < 20) iters = 20; if (iters > 2000) iters = 2000;
    }

    float best[NC];
    for (int c = 0; c < NC; ++c) best[c] = 1e30f;
    for (int sweep = 0; sweep < 4; ++sweep) {
        for (int q = 0; q < NC; ++q) {
            int c = (q + sweep) % NC;
            CHECK(cudaEventRecord(e0));
            for (int it = 0; it < iters; ++it) RUN_CONFIG(c);
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms/iters < best[c]) best[c] = ms/(float)iters;
        }
    }
    CHECK(cudaGetLastError());

    // ---------------- validation pass ---------------------------------------
    int  counts[NC];
    int  exactOrder[NC], isPermutation[NC], ascending[NC];

    for (int c = 0; c < NC; ++c) {
        CHECK(cudaMemset(d_out, 0xFF, sizeof(u32)*(size_t)n));
        RUN_CONFIG(c);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // --- TODO 4: recover the count ---
        u32 cnt;
        if (c == C_ATOMIC) {
            CHECK(cudaMemcpy(&cnt, d_cnt, sizeof(u32), cudaMemcpyDeviceToHost));
        } else {
            u32 lastOff, lastFlagSrc;
            CHECK(cudaMemcpy(&lastOff, d_offs + (n-1), sizeof(u32), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(&lastFlagSrc, d_data + (n-1), sizeof(u32), cudaMemcpyDeviceToHost));
            cnt = lastOff + keepPred(lastFlagSrc);     // <- the off-by-one fix
            if (c == C_FUSED) {
                // the fused version never writes the block offset back into
                // d_offs, so the last tile's base has to be added here too
                u32 lastBlockOff;
                CHECK(cudaMemcpy(&lastBlockOff, d_sums + (m-1), sizeof(u32), cudaMemcpyDeviceToHost));
                cnt += lastBlockOff;
            }
        }
        counts[c] = (int)cnt;

        CHECK(cudaMemcpy(h_out, d_out, sizeof(u32)*(size_t)cnt, cudaMemcpyDeviceToHost));

        exactOrder[c] = (int)cnt == refCount;
        if (exactOrder[c])
            for (u32 i = 0; i < cnt; ++i) if (h_out[i] != h_ref[i]) { exactOrder[c] = 0; break; }

        ascending[c] = 1;
        for (u32 i = 1; i < cnt; ++i) if (h_out[i] <= h_out[i-1]) { ascending[c] = 0; break; }

        for (int i = 0; i < n; ++i) seen[i] = 0;
        isPermutation[c] = ((int)cnt == refCount);
        for (u32 i = 0; i < cnt && isPermutation[c]; ++i) {
            u32 v = h_out[i];
            if (v >= (u32)n || !keepPred(h_data[v]) || seen[v]) isPermutation[c] = 0;
            else seen[v] = 1;
        }
    }

    // run the atomic version a second time and compare it with itself
    CHECK(cudaMemset(d_out, 0xFF, sizeof(u32)*(size_t)n));
    RUN_CONFIG(C_ATOMIC);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out2, d_out, sizeof(u32)*(size_t)refCount, cudaMemcpyDeviceToHost));
    CHECK(cudaMemset(d_out, 0xFF, sizeof(u32)*(size_t)n));
    RUN_CONFIG(C_ATOMIC);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, sizeof(u32)*(size_t)refCount, cudaMemcpyDeviceToHost));
    int atomicDiffRuns = 0;
    for (int i = 0; i < refCount; ++i) if (h_out[i] != h_out2[i]) atomicDiffRuns++;

    // ---------------- report -------------------------------------------------
    const double bytesTouched = (double)n*4.0 + (double)refCount*4.0;  // read data, write survivors
    printf("%-28s %9s %9s %8s %8s %8s %8s\n",
           "configuration", "ms", "count", "perm", "ascend", "exact", "GB/s*");
    for (int c = 0; c < NC; ++c) {
        printf("%-28s %9.4f %9d %8s %8s %8s %8.1f\n", CNAME[c], best[c], counts[c],
               isPermutation[c] ? "yes" : "NO", ascending[c] ? "yes" : "no",
               exactOrder[c] ? "yes" : "no",
               bytesTouched/(best[c]*1e-3)/1e9);
    }
    printf("\n  *GB/s counts only the unavoidable traffic (read N + write survivors),\n"
           "   so it is an efficiency score against the %.4f ms floor at 432 GB/s,\n"
           "   not a claim about what each version actually moves.\n",
           bytesTouched/432e9*1e3);
    printf("  atomic run A vs atomic run B: %d of %d positions differ\n",
           atomicDiffRuns, refCount);

    double ratio = best[C_SCAN] / best[C_ATOMIC];
    int bucket = (ratio >= 3.0) ? 1 : (ratio >= 1.5) ? 2 : (ratio >= 1.05) ? 3
               : (ratio >= 1.0/1.05) ? 4 : 5;
    printf("\n  scan / atomic time ratio: %.2fx -> bucket %d   predicted %d\n",
           ratio, bucket, PREDICT_BUCKET);
    printf("  fused scan / atomic     : %.2fx  (what the design TODO is worth)\n",
           best[C_FUSED]/best[C_ATOMIC]);

    int score = 0;
    score += (exactOrder[C_SCAN]      ? 2 : 0);   // ordered and exact
    score += (isPermutation[C_SCAN]   ? 1 : 0);
    score += (counts[C_SCAN] == refCount ? 1 : 0);
    score += (isPermutation[C_ATOMIC] ? 1 : 0);   // atomic must still be a permutation
    score += (!exactOrder[C_ATOMIC]   ? 1 : 0);   // ... and must NOT be ordered
    score += (bucket == PREDICT_BUCKET ? 1 : 0);
    printf("\n  score: %d/7\n", score);
    printf("OVERALL: %s\n", (score == 7) ? "PASS" : "FAIL");

    free(h_data); free(h_ref); free(h_out); free(h_out2); free(seen);
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_data)); CHECK(cudaFree(d_flags)); CHECK(cudaFree(d_offs));
    CHECK(cudaFree(d_sums)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_cnt));
    CHECK(cudaDeviceReset());
    return (score == 7) ? 0 : 1;
}
