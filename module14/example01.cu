// =============================================================================
// Module 14 / Example 1 — The histogram ladder across four input distributions
//
// GOAL
//   Establish the memory floor for a 256-bin byte histogram, then climb a
//   five-rung ladder from "one global atomic per element" to a vectorized,
//   coarsened, replicated, shared-memory-privatized kernel — and measure every
//   rung on FOUR different input distributions, because the distribution is the
//   dominant variable and a single-distribution benchmark is a lie.
//
//   Part A  ladder x distribution  (5 kernels + a streaming ceiling, 4 inputs)
//   Part B  replication factor R sweep, with the shared-memory / occupancy cost
//   Part C  coarsening curve (elements per thread) on two distributions
//   Part D  why v0 is 100x off the floor: global atomic throughput is governed
//           by distinct 32 B SECTORS per warp, not by distinct addresses
//
// BUILD
//   nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN
//   .\example01.exe
//
// SASS (for the ATOMS / RED discussion in the lesson):
//   nvcc -arch=sm_89 -O3 -cubin -o example01.cubin example01.cu
//   cuobjdump -sass example01.cubin
//
// Timing follows AUTHORING_SPEC section 12: 1500 ms duration-based warm-up,
// all configurations back-to-back in one rotated sweep, SWEEPS >= NCFG,
// auto-scaled iteration counts, min-of-N, validation in a separate pass.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            printf("CUDA error %s at %s:%d -> %s\n", #call, __FILE__,          \
                   __LINE__, cudaGetErrorString(_e));                          \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_KERNEL()                                                         \
    do {                                                                       \
        CHECK(cudaGetLastError());                                             \
        CHECK(cudaDeviceSynchronize());                                        \
    } while (0)

// ---------------------------------------------------------------- parameters
static const size_t N     = 201326592;   // 192 MiB of bytes = 4.0x the 48 MB L2
static const int    NBINS = 256;
static const int    BLK   = 256;
static const int    NDIST = 4;
static const double PEAK  = 432.0;       // GB/s, fixed and safe (spec 12.6)

static const char* DIST_NAME[NDIST] = { "uniform", "zipf", "same-bin", "clustered" };

// =============================================================================
// Kernels
// =============================================================================

// The floor: stream every byte, touch it, never store anything the compiler
// can fold away. Four independent accumulators so one thread has 4-way ILP.
__global__ void k_ceiling(const uchar4* __restrict__ in, size_t n4,
                          unsigned int* __restrict__ sink)
{
    unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        uchar4 v = in[i];
        a0 += v.x; a1 += v.y; a2 += v.z; a3 += v.w;
    }
    unsigned int s = a0 + a1 + a2 + a3;
    if (s == 0xFFFFFFFFu) sink[threadIdx.x & 31] = s;   // not provably dead
}

// v0 — one global atomic per element. The baseline everything is measured against.
__global__ void k_global(const unsigned char* __restrict__ in, size_t n,
                         unsigned int* __restrict__ hist)
{
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&hist[in[i]], 1u);
}

// v1 — shared privatization, one element per thread, grid = ceil(N/BLK).
// The textbook privatized histogram with the textbook launch. It is a trap.
__global__ void k_priv_naive(const unsigned char* __restrict__ in, size_t n,
                             unsigned int* __restrict__ hist, int nBins)
{
    extern __shared__ unsigned int s[];
    for (int b = threadIdx.x; b < nBins; b += blockDim.x) s[b] = 0u;
    __syncthreads();

    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&s[in[i]], 1u);

    __syncthreads();
    for (int b = threadIdx.x; b < nBins; b += blockDim.x)
        if (s[b]) atomicAdd(&hist[b], s[b]);
}

// v2 (rmask = 0) and v3 (rmask = R-1) — shared privatization with a
// machine-sized grid (so each thread coarsens over many elements) and R
// replicas of the histogram, laid out REPLICA-MAJOR: s[r * nBins + b].
// Replica selection is per WARP, so within a warp the address pattern is
// identical to the unreplicated kernel and ATOMS.POPC still merges lanes.
__global__ void k_repl(const unsigned char* __restrict__ in, size_t n,
                       unsigned int* __restrict__ hist, int nBins, int rmask)
{
    extern __shared__ unsigned int s[];
    const int tot = nBins * (rmask + 1);
    for (int b = threadIdx.x; b < tot; b += blockDim.x) s[b] = 0u;
    __syncthreads();

    unsigned int* my = s + (size_t)(((int)threadIdx.x >> 5) & rmask) * nBins;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&my[in[i]], 1u);

    __syncthreads();
    for (int b = threadIdx.x; b < nBins; b += blockDim.x) {
        unsigned int sum = 0;
        for (int r = 0; r <= rmask; ++r) sum += s[r * nBins + b];
        if (sum) atomicAdd(&hist[b], sum);
    }
}

// v4 — v3 plus uchar4 loads. Four bins per thread per loop iteration; the
// input side becomes one 4-byte load instead of four 1-byte loads.
__global__ void k_repl_vec(const uchar4* __restrict__ in4, size_t n4,
                           unsigned int* __restrict__ hist, int nBins, int rmask)
{
    extern __shared__ unsigned int s[];
    const int tot = nBins * (rmask + 1);
    for (int b = threadIdx.x; b < tot; b += blockDim.x) s[b] = 0u;
    __syncthreads();

    unsigned int* my = s + (size_t)(((int)threadIdx.x >> 5) & rmask) * nBins;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        uchar4 v = in4[i];
        atomicAdd(&my[v.x], 1u);
        atomicAdd(&my[v.y], 1u);
        atomicAdd(&my[v.z], 1u);
        atomicAdd(&my[v.w], 1u);
    }

    __syncthreads();
    for (int b = threadIdx.x; b < nBins; b += blockDim.x) {
        unsigned int sum = 0;
        for (int r = 0; r <= rmask; ++r) sum += s[r * nBins + b];
        if (sum) atomicAdd(&hist[b], sum);
    }
}

// Part D instrument. Every configuration issues exactly N atomics and presents
// exactly 32 DISTINCT bin addresses per warp. Only the number of distinct 32 B
// sectors those 32 words occupy changes (4 vs 32), plus an optional scramble of
// the lane -> word assignment so that "order within the warp" is controlled for.
template<int SCRAMBLE, int NSECT>
__global__ void k_sector(size_t n, unsigned int* __restrict__ hist)
{
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned int lane = threadIdx.x & 31u;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        unsigned int g = (unsigned int)(i >> 5) & 7u;              // which 32-word group
        unsigned int p = SCRAMBLE ? ((lane * 7u + 3u) & 31u) : lane;
        unsigned int addr = (NSECT == 4) ? (g * 32u + p)           // 32 words, 4 sectors
                                         : ((p * 8u + g) & 255u);  // 32 words, 32 sectors
        atomicAdd(&hist[addr], 1u);
    }
}

// =============================================================================
// Host-side deterministic input generation (and the exact reference histogram,
// accumulated during generation so we never pay for a second CPU pass).
// =============================================================================

static unsigned int xs32(unsigned int* st)
{
    unsigned int x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

static void gen(int dist, unsigned char* h, size_t n, unsigned long long* ref)
{
    for (int b = 0; b < NBINS; ++b) ref[b] = 0ull;
    unsigned int st = 0x12345678u + 7919u * (unsigned)dist;

    for (size_t i = 0; i < n; ++i) {
        unsigned char v;
        switch (dist) {
        case 0:  // uniform over all 256 bins
            v = (unsigned char)(xs32(&st) >> 24);
            break;
        case 1: { // zipf-like: u^6 piles ~40% of the mass into bin 0
            double u = (double)(xs32(&st) >> 8) * (1.0 / 16777216.0);
            double p = u * u * u * u * u * u;
            int b = (int)(p * 256.0);
            if (b > 255) b = 255;
            v = (unsigned char)b;
            break;
        }
        case 2:  // every element in one bin — the pathological case
            v = (unsigned char)37;
            break;
        default: // clustered: runs of 8192 identical values, so a whole warp
                 // (and usually a whole block iteration) shares one bin
            v = (unsigned char)((i >> 13) & 255);
            break;
        }
        h[i] = v;
        ref[v]++;
    }
}

// =============================================================================
// Timing harness (spec section 12)
// =============================================================================

static void warmup(const uchar4* d4, size_t n4, unsigned int* sink, int grid,
                   float targetMs = 1500.0f)
{
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    float acc = 0.0f;
    while (acc < targetMs) {
        CHECK(cudaEventRecord(a));
        for (int i = 0; i < 8; ++i) k_ceiling<<<grid, BLK>>>(d4, n4, sink);
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
        acc += ms;
    }
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
}

// Which kernel a configuration runs.
enum { K_CEIL = 0, K_GLOBAL, K_PRIV1, K_COARSE, K_REPL, K_VEC, NKER };
static const char* KER_NAME[NKER] = {
    "ceiling (stream only)",
    "v0 global atomic",
    "v1 shared priv, 1 elem/thr",
    "v2 shared priv + coarsening",
    "v3 + replication R=8",
    "v4 + uchar4 loads"
};

struct Cfg {
    int ker, dist;
    int grid, smem;
};

static void launch(const Cfg& c, const unsigned char* d_in, const uchar4* d_in4,
                   unsigned int* d_hist, unsigned int* d_sink)
{
    switch (c.ker) {
    case K_CEIL:   k_ceiling<<<c.grid, BLK>>>(d_in4, N / 4, d_sink); break;
    case K_GLOBAL: k_global<<<c.grid, BLK>>>(d_in, N, d_hist); break;
    case K_PRIV1:  k_priv_naive<<<c.grid, BLK, c.smem>>>(d_in, N, d_hist, NBINS); break;
    case K_COARSE: k_repl<<<c.grid, BLK, c.smem>>>(d_in, N, d_hist, NBINS, 0); break;
    case K_REPL:   k_repl<<<c.grid, BLK, c.smem>>>(d_in, N, d_hist, NBINS, 7); break;
    case K_VEC:    k_repl_vec<<<c.grid, BLK, c.smem>>>(d_in4, N / 4, d_hist, NBINS, 7); break;
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("=== Module 14 / Example 1 — histogram ladder x input distribution ===\n");
    printf("N = %zu bytes (%.1f MiB, %.2fx the 48 MB L2), %d bins, block = %d\n",
           N, N / 1048576.0, N / (48.0 * 1024 * 1024), NBINS, BLK);

    // ---------------------------------------------------------- allocation
    unsigned char* h_in = (unsigned char*)malloc(N);
    if (!h_in) { printf("host alloc failed\n"); return 1; }
    unsigned long long* h_ref = (unsigned long long*)malloc((size_t)NDIST * NBINS * sizeof(unsigned long long));

    unsigned char* d_in[NDIST];
    for (int d = 0; d < NDIST; ++d) CHECK(cudaMalloc(&d_in[d], N));
    unsigned int* d_hist; CHECK(cudaMalloc(&d_hist, NBINS * sizeof(unsigned int)));
    unsigned int* d_sink; CHECK(cudaMalloc(&d_sink, 32 * sizeof(unsigned int)));
    CHECK(cudaMemset(d_sink, 0, 32 * sizeof(unsigned int)));

    for (int d = 0; d < NDIST; ++d) {
        gen(d, h_in, N, h_ref + (size_t)d * NBINS);
        CHECK(cudaMemcpy(d_in[d], h_in, N, cudaMemcpyHostToDevice));
    }
    printf("\ninput distributions (fraction of elements in the top bin):\n");
    for (int d = 0; d < NDIST; ++d) {
        unsigned long long mx = 0; int arg = 0;
        for (int b = 0; b < NBINS; ++b)
            if (h_ref[(size_t)d * NBINS + b] > mx) { mx = h_ref[(size_t)d * NBINS + b]; arg = b; }
        int nz = 0;
        for (int b = 0; b < NBINS; ++b) if (h_ref[(size_t)d * NBINS + b]) nz++;
        printf("  %-10s  bin %3d holds %6.2f%%   non-empty bins: %d\n",
               DIST_NAME[d], arg, 100.0 * (double)mx / (double)N, nz);
    }

    // ---------------------------------------------------- launch geometry
    int occGlobal = 0, occRepl = 0, occVec = 0, occCoarse = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occGlobal, (void*)k_global, BLK, 0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occCoarse, (void*)k_repl, BLK, NBINS * 4));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occRepl, (void*)k_repl, BLK, 8 * NBINS * 4));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occVec, (void*)k_repl_vec, BLK, 8 * NBINS * 4));

    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    const int nSM = prop.multiProcessorCount;
    const int gridMachine = occRepl * nSM;              // one full wave of v3
    const int gridNaive   = (int)((N + BLK - 1) / BLK); // v1's textbook grid

    printf("\nlaunch geometry (%d SMs):\n", nSM);
    printf("  blocks/SM: v0 %d (0 B smem) | v2 %d (%d B) | v3 %d (%d B) | v4 %d (%d B)\n",
           occGlobal, occCoarse, NBINS * 4, occRepl, 8 * NBINS * 4, occVec, 8 * NBINS * 4);
    printf("  machine-sized grid = %d blocks; v1's grid = %d blocks\n",
           gridMachine, gridNaive);
    printf("  flush global atomics: v1 %.3g   v2/v3/v4 %.3g   (vs N = %.3g)\n",
           (double)gridNaive * NBINS, (double)gridMachine * NBINS, (double)N);

    // ------------------------------------------------------ configurations
    Cfg cfg[NKER * NDIST];
    int ncfg = 0;
    for (int d = 0; d < NDIST; ++d)
        for (int k = 0; k < NKER; ++k) {
            Cfg c; c.ker = k; c.dist = d;
            switch (k) {
            case K_CEIL:   c.grid = gridMachine; c.smem = 0; break;
            case K_GLOBAL: c.grid = gridMachine; c.smem = 0; break;
            case K_PRIV1:  c.grid = gridNaive;   c.smem = NBINS * 4; break;
            case K_COARSE: c.grid = gridMachine; c.smem = NBINS * 4; break;
            case K_REPL:   c.grid = gridMachine; c.smem = 8 * NBINS * 4; break;
            default:       c.grid = gridMachine; c.smem = 8 * NBINS * 4; break;
            }
            cfg[ncfg++] = c;
        }

    // ------------------------------------------------------------- warm-up
    printf("\nwarm-up (1500 ms, duration-based: ramps the memory P-state, not just the SM clock)...\n");
    warmup((const uchar4*)d_in[0], N / 4, d_sink, gridMachine);

    // ------------------------------------------------------------- timing
    cudaEvent_t evA, evB;
    CHECK(cudaEventCreate(&evA)); CHECK(cudaEventCreate(&evB));

    double best[NKER * NDIST];
    int    iters[NKER * NDIST];
    for (int i = 0; i < ncfg; ++i) { best[i] = 1e30; iters[i] = 0; }

    // Size each timed segment to ~10 ms (spec 12.12).
    for (int i = 0; i < ncfg; ++i) {
        CHECK(cudaMemset(d_hist, 0, NBINS * sizeof(unsigned int)));
        CHECK(cudaEventRecord(evA));
        launch(cfg[i], d_in[cfg[i].dist], (const uchar4*)d_in[cfg[i].dist], d_hist, d_sink);
        CHECK(cudaEventRecord(evB));
        CHECK(cudaEventSynchronize(evB));
        float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
        int it = (int)(10.0f / (ms > 0.0001f ? ms : 0.0001f));
        if (it < 3) it = 3;
        if (it > 120) it = 120;
        iters[i] = it;
    }
    CHECK(cudaGetLastError());

    const int SWEEPS = NKER * NDIST;     // SWEEPS >= NCFG (spec 12.9)
    printf("timing: %d configurations, %d rotated sweeps, per-config iteration\n"
           "counts auto-scaled to ~10 ms segments; reporting min-of-sweeps.\n",
           ncfg, SWEEPS);

    for (int sw = 0; sw < SWEEPS; ++sw) {
        for (int q = 0; q < ncfg; ++q) {
            int p = (q + sw) % ncfg;
            CHECK(cudaEventRecord(evA));
            for (int it = 0; it < iters[p]; ++it)
                launch(cfg[p], d_in[cfg[p].dist], (const uchar4*)d_in[cfg[p].dist], d_hist, d_sink);
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
            double per = (double)ms / iters[p];
            if (per < best[p]) best[p] = per;
        }
    }
    CHECK(cudaGetLastError());

    // -------------------------------------------------- PART A: the table
    printf("\n--- PART A: ladder x distribution (min-of-%d, ms) ---\n", SWEEPS);
    printf("traffic model: 1N = %zu B read. Ceiling is measured in the same sweep.\n\n", N);

    printf("  %-28s", "kernel");
    for (int d = 0; d < NDIST; ++d) printf("%12s", DIST_NAME[d]);
    printf("\n");
    for (int k = 0; k < NKER; ++k) {
        printf("  %-28s", KER_NAME[k]);
        for (int d = 0; d < NDIST; ++d) printf("%12.4f", best[d * NKER + k]);
        printf("\n");
    }

    printf("\n  achieved GB/s on the 1N input stream (and %% of the measured ceiling):\n");
    printf("  %-28s", "kernel");
    for (int d = 0; d < NDIST; ++d) printf("%16s", DIST_NAME[d]);
    printf("\n");
    for (int k = 0; k < NKER; ++k) {
        printf("  %-28s", KER_NAME[k]);
        for (int d = 0; d < NDIST; ++d) {
            double ms = best[d * NKER + k];
            double gbs = (double)N / (ms * 1e6);
            double ceil_ms = best[d * NKER + K_CEIL];
            printf("%10.1f%5.0f%%", gbs, 100.0 * ceil_ms / ms);
        }
        printf("\n");
    }

    printf("\n  speedup over v0 (the global-atomic baseline) — the stable quantity:\n");
    printf("  %-28s", "kernel");
    for (int d = 0; d < NDIST; ++d) printf("%12s", DIST_NAME[d]);
    printf("\n");
    for (int k = K_PRIV1; k < NKER; ++k) {
        printf("  %-28s", KER_NAME[k]);
        for (int d = 0; d < NDIST; ++d)
            printf("%11.2fx", best[d * NKER + K_GLOBAL] / best[d * NKER + k]);
        printf("\n");
    }
    {
        double cg = best[0 * NKER + K_CEIL];
        printf("\n  streaming ceiling: %.4f ms = %.1f GB/s = %.1f%% of the %.0f GB/s peak\n",
               cg, (double)N / (cg * 1e6), 100.0 * (double)N / (cg * 1e6) / PEAK, PEAK);
        printf("  v0 spread across distributions: %.2fx (max/min)\n",
               (best[2 * NKER + K_GLOBAL] > best[0 * NKER + K_GLOBAL]
                    ? best[2 * NKER + K_GLOBAL] : best[0 * NKER + K_GLOBAL]) /
               (best[1 * NKER + K_GLOBAL] < best[0 * NKER + K_GLOBAL]
                    ? best[1 * NKER + K_GLOBAL] : best[0 * NKER + K_GLOBAL]));
    }

    // ------------------------------------------- PART B: replication sweep
    printf("\n--- PART B: replication factor R (v3 kernel, machine grid) ---\n");
    printf("(re-warmed; Part A's sustained streaming engages the power cap, so B and C\n"
           " are internally consistent but their absolute ms are NOT comparable to A's.)\n");
    warmup((const uchar4*)d_in[0], N / 4, d_sink, gridMachine, 1500.0f);
    {
        const int RS[5] = { 1, 2, 4, 8, 16 };
        double rb[5][NDIST];
        int    rg[5], ri[5], ro[5];
        for (int r = 0; r < 5; ++r) {
            int smem = RS[r] * NBINS * 4;
            CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ro[r], (void*)k_repl, BLK, smem));
            rg[r] = ro[r] * nSM;
            CHECK(cudaEventRecord(evA));
            k_repl<<<rg[r], BLK, smem>>>(d_in[0], N, d_hist, NBINS, RS[r] - 1);
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
            int it = (int)(10.0f / (ms > 0.0001f ? ms : 0.0001f));
            ri[r] = it < 3 ? 3 : (it > 120 ? 120 : it);
            for (int d = 0; d < NDIST; ++d) rb[r][d] = 1e30;
        }
        CHECK(cudaGetLastError());
        const int NC = 5 * NDIST;
        for (int sw = 0; sw < NC; ++sw)
            for (int q = 0; q < NC; ++q) {
                int p = (q + sw) % NC, r = p / NDIST, d = p % NDIST;
                int smem = RS[r] * NBINS * 4;
                CHECK(cudaEventRecord(evA));
                for (int it = 0; it < ri[r]; ++it)
                    k_repl<<<rg[r], BLK, smem>>>(d_in[d], N, d_hist, NBINS, RS[r] - 1);
                CHECK(cudaEventRecord(evB));
                CHECK(cudaEventSynchronize(evB));
                float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
                double per = (double)ms / ri[r];
                if (per < rb[r][d]) rb[r][d] = per;
            }
        CHECK(cudaGetLastError());

        printf("   R  smem/blk  blocks/SM  grid   flushAtomics");
        for (int d = 0; d < NDIST; ++d) printf("%12s", DIST_NAME[d]);
        printf("\n");
        for (int r = 0; r < 5; ++r) {
            printf("  %2d  %7d  %9d  %5d   %12.0f", RS[r], RS[r] * NBINS * 4, ro[r],
                   rg[r], (double)rg[r] * NBINS);
            for (int d = 0; d < NDIST; ++d) printf("%12.4f", rb[r][d]);
            printf("\n");
        }
        printf("\n  speedup vs R=1, same distribution:");
        for (int d = 0; d < NDIST; ++d) printf("%12s", DIST_NAME[d]);
        printf("\n");
        for (int r = 0; r < 5; ++r) {
            printf("  R=%-2d                             ", RS[r]);
            for (int d = 0; d < NDIST; ++d) printf("%11.2fx", rb[0][d] / rb[r][d]);
            printf("\n");
        }
    }

    // ------------------------------------------- PART C: coarsening curve
    printf("\n--- PART C: coarsening (elements per thread, v3 with R=8) ---\n");
    warmup((const uchar4*)d_in[0], N / 4, d_sink, gridMachine, 1500.0f);
    {
        const int NCO = 7;
        int gr[NCO];
        double cb[NCO][2];
        int ci[NCO];
        int smem = 8 * NBINS * 4;
        // grid shrinks by 4x each step; C = N / (grid * BLK)
        gr[0] = (int)((N + BLK - 1) / BLK);
        for (int i = 1; i < NCO; ++i) gr[i] = gr[i - 1] / 4;
        for (int i = 0; i < NCO; ++i) {
            CHECK(cudaEventRecord(evA));
            k_repl<<<gr[i], BLK, smem>>>(d_in[0], N, d_hist, NBINS, 7);
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
            int it = (int)(10.0f / (ms > 0.0001f ? ms : 0.0001f));
            ci[i] = it < 3 ? 3 : (it > 120 ? 120 : it);
            cb[i][0] = cb[i][1] = 1e30;
        }
        CHECK(cudaGetLastError());
        const int NC = NCO * 2;
        const int dsel[2] = { 0, 3 };   // uniform and clustered
        for (int sw = 0; sw < NC; ++sw)
            for (int q = 0; q < NC; ++q) {
                int p = (q + sw) % NC, i = p / 2, j = p % 2;
                CHECK(cudaEventRecord(evA));
                for (int it = 0; it < ci[i]; ++it)
                    k_repl<<<gr[i], BLK, smem>>>(d_in[dsel[j]], N, d_hist, NBINS, 7);
                CHECK(cudaEventRecord(evB));
                CHECK(cudaEventSynchronize(evB));
                float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
                double per = (double)ms / ci[i];
                if (per < cb[i][j]) cb[i][j] = per;
            }
        CHECK(cudaGetLastError());

        printf("   grid    elems/thread  flushAtomics   flush/N   uniform ms   clustered ms\n");
        for (int i = 0; i < NCO; ++i) {
            double C = (double)N / ((double)gr[i] * BLK);
            double fa = (double)gr[i] * NBINS;
            printf("  %7d  %12.1f  %12.0f  %8.3f  %11.4f  %13.4f\n",
                   gr[i], C, fa, fa / (double)N, cb[i][0], cb[i][1]);
        }
    }

    // ------------------------ PART D: what actually limits the global kernel
    printf("\n--- PART D: global atomic throughput vs SECTORS per warp ---\n");
    printf("Every row issues exactly %zu atomics and presents exactly 32 DISTINCT\n"
           "bin addresses per warp. Only the sector footprint of those 32 words changes.\n\n", N);
    warmup((const uchar4*)d_in[0], N / 4, d_sink, gridMachine, 1500.0f);
    {
        const int NP = 4;
        double pb[NP];
        int pit[NP];
        for (int i = 0; i < NP; ++i) {
            CHECK(cudaEventRecord(evA));
            switch (i) {
            case 0: k_sector<0, 4><<<gridMachine, BLK>>>(N, d_hist); break;
            case 1: k_sector<1, 4><<<gridMachine, BLK>>>(N, d_hist); break;
            case 2: k_sector<0, 32><<<gridMachine, BLK>>>(N, d_hist); break;
            default: k_sector<1, 32><<<gridMachine, BLK>>>(N, d_hist); break;
            }
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
            int it = (int)(10.0f / (ms > 0.0001f ? ms : 0.0001f));
            pit[i] = it < 2 ? 2 : (it > 120 ? 120 : it);
            pb[i] = 1e30;
        }
        CHECK(cudaGetLastError());
        for (int sw = 0; sw < NP; ++sw)
            for (int q = 0; q < NP; ++q) {
                int p = (q + sw) % NP;
                CHECK(cudaEventRecord(evA));
                for (int it = 0; it < pit[p]; ++it) {
                    switch (p) {
                    case 0: k_sector<0, 4><<<gridMachine, BLK>>>(N, d_hist); break;
                    case 1: k_sector<1, 4><<<gridMachine, BLK>>>(N, d_hist); break;
                    case 2: k_sector<0, 32><<<gridMachine, BLK>>>(N, d_hist); break;
                    default: k_sector<1, 32><<<gridMachine, BLK>>>(N, d_hist); break;
                    }
                }
                CHECK(cudaEventRecord(evB));
                CHECK(cudaEventSynchronize(evB));
                float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
                double per = (double)ms / pit[p];
                if (per < pb[p]) pb[p] = per;
            }
        CHECK(cudaGetLastError());
        static const char* PN[NP] = {
            "32 words in  4 sectors, lane order",
            "32 words in  4 sectors, scrambled",
            "32 words in 32 sectors, lane order",
            "32 words in 32 sectors, scrambled"
        };
        for (int i = 0; i < NP; ++i)
            printf("  %-36s %9.4f ms  %6.2f Gatomic/s\n", PN[i], pb[i], (double)N / (pb[i] * 1e6));
        printf("\n  sector ratio (32 sectors / 4 sectors): %.2fx and %.2fx\n",
               pb[2] / pb[0], pb[3] / pb[1]);
        printf("  scramble ratio (same sector count):    %.2fx and %.2fx\n",
               pb[1] / pb[0], pb[3] / pb[2]);
    }

    // ================================================= VALIDATION (pass 2)
    printf("\n--- validation (separate untimed pass) ---\n");
    int fails = 0;
    for (int d = 0; d < NDIST; ++d) {
        for (int k = K_GLOBAL; k < NKER; ++k) {
            CHECK(cudaMemset(d_hist, 0, NBINS * sizeof(unsigned int)));
            Cfg c = cfg[d * NKER + k];
            launch(c, d_in[d], (const uchar4*)d_in[d], d_hist, d_sink);
            CHECK_KERNEL();
            unsigned int h[NBINS];
            CHECK(cudaMemcpy(h, d_hist, sizeof(h), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int b = 0; b < NBINS; ++b)
                if ((unsigned long long)h[b] != h_ref[(size_t)d * NBINS + b]) bad++;
            if (bad) {
                printf("  [FAIL] %-28s on %-10s : %d bins wrong\n", KER_NAME[k], DIST_NAME[d], bad);
                fails++;
            }
        }
    }
    if (!fails) printf("  [PASS] all %d (kernel, distribution) pairs exact against the CPU reference\n",
                       (NKER - 1) * NDIST);

    // ------------------------------------------------------------- cleanup
    CHECK(cudaEventDestroy(evA)); CHECK(cudaEventDestroy(evB));
    for (int d = 0; d < NDIST; ++d) CHECK(cudaFree(d_in[d]));
    CHECK(cudaFree(d_hist)); CHECK(cudaFree(d_sink));
    free(h_in); free(h_ref);
    CHECK(cudaDeviceReset());

    printf("\nOVERALL: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
