// =============================================================================
// Module 14 / Example 2 — When the bins stop fitting in shared memory
//
// GOAL
//   Example 1 held the bin count at 256 and swept the input distribution.
//   This file holds the distribution axis down to two cases and sweeps the
//   BIN COUNT from 256 to 65536, which is the axis on which shared-memory
//   privatization stops being available at all. Five strategies:
//
//     S0  global atomics                      (always available)
//     S1  shared privatization, uint4 loads   (needs nBins*4 <= 99 KB)
//     S2  multi-pass over bin windows         (P passes => P*N of input traffic)
//     S3  global-memory privatization, G copies + a reduce kernel
//     S4  cub::DeviceHistogram::HistogramEven (production reality)
//
//   plus a streaming ceiling measured in the same rotated sweep.
//
// BUILD
//   nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler /Zc:preprocessor -o example02.exe example02.cu
//   (the -std and /Zc:preprocessor pair is what CCCL headers need under MSVC;
//    Modules 9, 12 and 13 needed the same)
// RUN
//   .\example02.exe
//
// Timing follows AUTHORING_SPEC section 12.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

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
    do { CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize()); } while (0)

static const size_t N    = 67108864;   // 2^26 keys x 4 B = 256 MiB = 5.3x L2
static const int    BLK  = 256;
static const double PEAK = 432.0;

// A key is a 30-bit fraction; the bin is derived on the device so that one
// input array serves every bin count. bin = floor(frac * nBins), exactly.
__device__ __host__ __forceinline__ unsigned int binOf(unsigned int frac, int nBins)
{
    return (unsigned int)(((unsigned long long)frac * (unsigned long long)nBins) >> 30);
}

// =============================================================================
// Kernels
// =============================================================================

__global__ void k_ceiling(const uint4* __restrict__ in, size_t n4, unsigned int* sink)
{
    unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        uint4 v = in[i];
        a0 += v.x; a1 += v.y; a2 += v.z; a3 += v.w;
    }
    unsigned int s = a0 + a1 + a2 + a3;
    if (s == 0xFFFFFFFFu) sink[threadIdx.x & 31] = s;
}

// S0 — one global atomic per element.
__global__ void k_global(const uint4* __restrict__ in, size_t n4,
                         unsigned int* __restrict__ hist, int nBins)
{
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        uint4 v = in[i];
        atomicAdd(&hist[binOf(v.x, nBins)], 1u);
        atomicAdd(&hist[binOf(v.y, nBins)], 1u);
        atomicAdd(&hist[binOf(v.z, nBins)], 1u);
        atomicAdd(&hist[binOf(v.w, nBins)], 1u);
    }
}

// S1 — the whole histogram privatized in shared memory. Requires
// nBins * 4 B of dynamic shared memory per block.
__global__ void k_shared(const uint4* __restrict__ in, size_t n4,
                         unsigned int* __restrict__ hist, int nBins)
{
    extern __shared__ unsigned int s[];
    for (int b = threadIdx.x; b < nBins; b += blockDim.x) s[b] = 0u;
    __syncthreads();

    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        uint4 v = in[i];
        atomicAdd(&s[binOf(v.x, nBins)], 1u);
        atomicAdd(&s[binOf(v.y, nBins)], 1u);
        atomicAdd(&s[binOf(v.z, nBins)], 1u);
        atomicAdd(&s[binOf(v.w, nBins)], 1u);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nBins; b += blockDim.x)
        if (s[b]) atomicAdd(&hist[b], s[b]);
}

// S2 — multi-pass. One pass owns bins [lo, lo+W); everything else is dropped.
// The whole input is re-read once per pass: P passes cost P*N of traffic.
__global__ void k_window(const uint4* __restrict__ in, size_t n4,
                         unsigned int* __restrict__ hist, int nBins, int lo, int W)
{
    extern __shared__ unsigned int s[];
    for (int b = threadIdx.x; b < W; b += blockDim.x) s[b] = 0u;
    __syncthreads();

    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        uint4 v = in[i];
        unsigned int b0 = binOf(v.x, nBins) - (unsigned)lo;
        unsigned int b1 = binOf(v.y, nBins) - (unsigned)lo;
        unsigned int b2 = binOf(v.z, nBins) - (unsigned)lo;
        unsigned int b3 = binOf(v.w, nBins) - (unsigned)lo;
        if (b0 < (unsigned)W) atomicAdd(&s[b0], 1u);
        if (b1 < (unsigned)W) atomicAdd(&s[b1], 1u);
        if (b2 < (unsigned)W) atomicAdd(&s[b2], 1u);
        if (b3 < (unsigned)W) atomicAdd(&s[b3], 1u);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < W; b += blockDim.x)
        if (s[b]) atomicAdd(&hist[lo + b], s[b]);
}

// S3 — privatization one level out: G copies of the histogram in GLOBAL memory,
// block b using copy (b & gmask). Contention falls by up to G; the cost is
// G*nBins*4 B of footprint plus a reduce pass over it.
__global__ void k_gpriv(const uint4* __restrict__ in, size_t n4,
                        unsigned int* __restrict__ part, int nBins, int gmask)
{
    unsigned int* my = part + (size_t)((int)blockIdx.x & gmask) * nBins;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        uint4 v = in[i];
        atomicAdd(&my[binOf(v.x, nBins)], 1u);
        atomicAdd(&my[binOf(v.y, nBins)], 1u);
        atomicAdd(&my[binOf(v.z, nBins)], 1u);
        atomicAdd(&my[binOf(v.w, nBins)], 1u);
    }
}

__global__ void k_fold(const unsigned int* __restrict__ part,
                       unsigned int* __restrict__ hist, int nBins, int G)
{
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nBins;
         b += gridDim.x * blockDim.x) {
        unsigned int t = 0;
        for (int g = 0; g < G; ++g) t += part[(size_t)g * nBins + b];
        hist[b] = t;
    }
}

// =============================================================================
// Host side
// =============================================================================

static unsigned int xs32(unsigned int* st)
{
    unsigned int x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

// dist 0: uniform fraction. dist 1: u^6, so ~40% of the mass lands in the
// lowest 1/256 of the bin range whatever the bin count is.
static void genFrac(int dist, unsigned int* h, size_t n)
{
    unsigned int st = 0x9E3779B9u + 4099u * (unsigned)dist;
    for (size_t i = 0; i < n; ++i) {
        double u = (double)(xs32(&st) >> 8) * (1.0 / 16777216.0);
        double p = (dist == 0) ? u : (u * u * u * u * u * u);
        unsigned int f = (unsigned int)(p * 1073741824.0);   // 2^30
        if (f > 1073741823u) f = 1073741823u;
        h[i] = f;
    }
}

enum { S_CEIL = 0, S_GLOBAL, S_SHARED, S_WINDOW, S_GPRIV, S_CUB, NSTRAT };
static const char* SNAME[NSTRAT] = {
    "ceiling (stream only)",
    "S0 global atomics",
    "S1 shared privatization",
    "S2 multi-pass windows",
    "S3 global priv (G copies)",
    "S4 cub::DeviceHistogram"
};

struct Env {
    const uint4*   in4[2];
    unsigned int*  hist;
    unsigned int*  part;
    unsigned int*  sink;
    void*          cubTmp;
    size_t         cubBytes;
    const unsigned int* rawIn[2];
    int grid, gridShared, G, W, P, nBins, smemShared;
};

static void runStrat(int s, int dist, Env& e)
{
    switch (s) {
    case S_CEIL:
        k_ceiling<<<e.grid, BLK>>>(e.in4[dist], N / 4, e.sink);
        break;
    case S_GLOBAL:
        k_global<<<e.grid, BLK>>>(e.in4[dist], N / 4, e.hist, e.nBins);
        break;
    case S_SHARED:
        k_shared<<<e.gridShared, BLK, e.smemShared>>>(e.in4[dist], N / 4, e.hist, e.nBins);
        break;
    case S_WINDOW:
        for (int p = 0; p < e.P; ++p)
            k_window<<<e.grid, BLK, e.W * 4>>>(e.in4[dist], N / 4, e.hist, e.nBins,
                                               p * e.W, e.W);
        break;
    case S_GPRIV:
        CHECK(cudaMemsetAsync(e.part, 0, (size_t)e.G * e.nBins * sizeof(unsigned int)));
        k_gpriv<<<e.grid, BLK>>>(e.in4[dist], N / 4, e.part, e.nBins, e.G - 1);
        k_fold<<<(e.nBins + BLK - 1) / BLK, BLK>>>(e.part, e.hist, e.nBins, e.G);
        break;
    default:
        cub::DeviceHistogram::HistogramEven(e.cubTmp, e.cubBytes, e.rawIn[dist],
                                            (int*)e.hist, e.nBins + 1, 0,
                                            1073741824, (int)N);
        break;
    }
}

static void warmup(const uint4* d4, unsigned int* sink, int grid, float target)
{
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    float acc = 0.0f;
    while (acc < target) {
        CHECK(cudaEventRecord(a));
        for (int i = 0; i < 8; ++i) k_ceiling<<<grid, BLK>>>(d4, N / 4, sink);
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
        acc += ms;
    }
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("=== Module 14 / Example 2 — bin count vs strategy ===\n");
    printf("N = %zu uint32 keys = %.0f MiB (%.1fx the 48 MB L2)\n",
           N, N * 4.0 / 1048576.0, N * 4.0 / (48.0 * 1024 * 1024));

    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    const int nSM = prop.multiProcessorCount;
    int smemOptin = 0;
    CHECK(cudaDeviceGetAttribute(&smemOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    printf("shared/block: %zu B default, %d B with opt-in; %d SMs\n\n",
           prop.sharedMemPerBlock, smemOptin, nSM);

    CHECK(cudaFuncSetAttribute(k_shared, cudaFuncAttributeMaxDynamicSharedMemorySize, smemOptin));

    // ---------------------------------------------------------- allocation
    unsigned int* h_raw = (unsigned int*)malloc(N * sizeof(unsigned int));
    if (!h_raw) { printf("host alloc failed\n"); return 1; }

    unsigned int* d_raw[2];
    for (int d = 0; d < 2; ++d) {
        CHECK(cudaMalloc(&d_raw[d], N * sizeof(unsigned int)));
        genFrac(d, h_raw, N);
        CHECK(cudaMemcpy(d_raw[d], h_raw, N * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    const int MAXB = 65536;
    const int GMAX = 64;
    unsigned int *d_hist, *d_part, *d_sink;
    CHECK(cudaMalloc(&d_hist, (MAXB + 1) * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_part, (size_t)GMAX * MAXB * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_sink, 32 * sizeof(unsigned int)));
    CHECK(cudaMemset(d_sink, 0, 32 * sizeof(unsigned int)));

    size_t cubBytes = 0; void* cubTmp = NULL;
    cub::DeviceHistogram::HistogramEven(cubTmp, cubBytes, d_raw[0], (int*)d_hist,
                                        MAXB + 1, 0, 1073741824, (int)N);
    CHECK(cudaMalloc(&cubTmp, cubBytes));
    printf("cub::DeviceHistogram temp storage at %d bins: %zu B\n", MAXB, cubBytes);

    Env e;
    e.in4[0] = (const uint4*)d_raw[0]; e.in4[1] = (const uint4*)d_raw[1];
    e.rawIn[0] = d_raw[0];             e.rawIn[1] = d_raw[1];
    e.hist = d_hist; e.part = d_part; e.sink = d_sink;
    e.cubTmp = cubTmp; e.cubBytes = cubBytes;

    int occCeil = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occCeil, (void*)k_ceiling, BLK, 0));
    e.grid = occCeil * nSM;

    printf("machine-sized grid = %d blocks x %d threads\n", e.grid, BLK);
    printf("\nwarm-up (1500 ms)...\n");
    warmup(e.in4[0], d_sink, e.grid, 1500.0f);

    cudaEvent_t evA, evB;
    CHECK(cudaEventCreate(&evA)); CHECK(cudaEventCreate(&evB));

    const int BINS[6] = { 256, 1024, 4096, 8192, 16384, 65536 };
    const char* DN[2] = { "uniform", "zipf" };

    double table[6][2][NSTRAT];
    int    occTab[6];
    int    Wtab[6], Ptab[6], Gtab[6];
    int    valid[6][NSTRAT];

    for (int bi = 0; bi < 6; ++bi) {
        e.nBins = BINS[bi];
        e.smemShared = e.nBins * 4;

        // ---- geometry decisions for this bin count
        int occS = 0;
        int sharedOK = (e.smemShared <= smemOptin - 1024) ? 1 : 0;
        if (sharedOK) {
            CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occS, (void*)k_shared,
                                                               BLK, e.smemShared));
            if (occS < 1) sharedOK = 0;
        }
        e.gridShared = sharedOK ? occS * nSM : e.grid;
        occTab[bi] = sharedOK ? occS : 0;

        // window size: 8192 bins = 32 KB of shared memory, which still leaves
        // 3 blocks resident per SM. P = nBins / W passes over the whole input.
        e.W = e.nBins < 8192 ? e.nBins : 8192;
        e.P = e.nBins / e.W;
        Wtab[bi] = e.W; Ptab[bi] = e.P;

        // G copies of the histogram in global memory, capped at 4 MB of footprint
        e.G = GMAX;
        while (e.G > 1 && (size_t)e.G * e.nBins * 4 > (4u << 20)) e.G >>= 1;
        Gtab[bi] = e.G;

        for (int s = 0; s < NSTRAT; ++s)
            valid[bi][s] = (s == S_SHARED) ? sharedOK : 1;

        // Re-warm before every bin count. The heavy columns (many passes, many
        // atomics) generate enough sustained load to trip the power cap, which
        // would otherwise leave the NEXT column measured in a colder state.
        warmup(e.in4[0], d_sink, e.grid, 1500.0f);

        // ---- auto-scale iteration counts
        const int NC = NSTRAT * 2;
        int iters[NSTRAT * 2];
        for (int c = 0; c < NC; ++c) {
            int s = c / 2, d = c % 2;
            if (!valid[bi][s]) { iters[c] = 1; table[bi][d][s] = -1.0; continue; }
            CHECK(cudaMemset(d_hist, 0, (e.nBins + 1) * sizeof(unsigned int)));
            CHECK(cudaEventRecord(evA));
            runStrat(s, d, e);
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
            int it = (int)(10.0f / (ms > 0.0001f ? ms : 0.0001f));
            iters[c] = it < 2 ? 2 : (it > 30 ? 30 : it);
            table[bi][d][s] = 1e30;
        }
        CHECK_KERNEL();

        // ---- rotated sweep, SWEEPS >= NCFG
        for (int sw = 0; sw < NC; ++sw)
            for (int q = 0; q < NC; ++q) {
                int c = (q + sw) % NC, s = c / 2, d = c % 2;
                if (!valid[bi][s]) continue;
                CHECK(cudaEventRecord(evA));
                for (int it = 0; it < iters[c]; ++it) runStrat(s, d, e);
                CHECK(cudaEventRecord(evB));
                CHECK(cudaEventSynchronize(evB));
                float ms; CHECK(cudaEventElapsedTime(&ms, evA, evB));
                double per = (double)ms / iters[c];
                if (per < table[bi][d][s]) table[bi][d][s] = per;
            }
        CHECK_KERNEL();
    }

    // ------------------------------------------------------------- report
    printf("\n--- strategy availability and cost model ---\n");
    printf("  nBins   smem/blk  blocks/SM  S1?   S2 window x passes   S3 copies (footprint)\n");
    for (int bi = 0; bi < 6; ++bi)
        printf("  %5d   %8d  %9d  %-4s  %6d x %-3d          %2d (%6.2f MB)\n",
               BINS[bi], BINS[bi] * 4, occTab[bi], valid[bi][S_SHARED] ? "yes" : "NO",
               Wtab[bi], Ptab[bi], Gtab[bi], Gtab[bi] * (double)BINS[bi] * 4 / 1048576.0);

    printf("\n--- time (ms, min of %d rotated sweeps) ---\n", NSTRAT * 2);
    printf("  %-27s", "strategy");
    for (int bi = 0; bi < 6; ++bi) printf("%9d", BINS[bi]);
    printf("   (bins)\n");
    for (int d = 0; d < 2; ++d) {
        printf("  [%s]\n", DN[d]);
        for (int s = 0; s < NSTRAT; ++s) {
            printf("  %-27s", SNAME[s]);
            for (int bi = 0; bi < 6; ++bi) {
                if (!valid[bi][s]) printf("%9s", "n/a");
                else printf("%9.4f", table[bi][d][s]);
            }
            printf("\n");
        }
    }

    printf("\n--- speedup over S0 (global atomics), the stable quantity ---\n");
    printf("  %-27s", "strategy");
    for (int bi = 0; bi < 6; ++bi) printf("%9d", BINS[bi]);
    printf("\n");
    for (int d = 0; d < 2; ++d) {
        printf("  [%s]\n", DN[d]);
        for (int s = S_SHARED; s < NSTRAT; ++s) {
            printf("  %-27s", SNAME[s]);
            for (int bi = 0; bi < 6; ++bi) {
                if (!valid[bi][s]) printf("%9s", "n/a");
                else printf("%8.2fx", table[bi][d][S_GLOBAL] / table[bi][d][s]);
            }
            printf("\n");
        }
    }

    printf("\n--- %% of the measured streaming ceiling (1N = %.0f MB read) ---\n",
           N * 4.0 / 1e6);
    for (int d = 0; d < 2; ++d) {
        printf("  [%s]\n", DN[d]);
        for (int s = S_GLOBAL; s < NSTRAT; ++s) {
            printf("  %-27s", SNAME[s]);
            for (int bi = 0; bi < 6; ++bi) {
                if (!valid[bi][s]) printf("%9s", "n/a");
                else printf("%8.0f%%", 100.0 * table[bi][d][S_CEIL] / table[bi][d][s]);
            }
            printf("\n");
        }
    }
    {
        double c = table[0][0][S_CEIL];
        printf("\n  ceiling: %.4f ms = %.1f GB/s = %.1f%% of %.0f GB/s peak\n",
               c, N * 4.0 / (c * 1e6), 100.0 * N * 4.0 / (c * 1e6) / PEAK, PEAK);
    }

    // ============================================== VALIDATION (second pass)
    printf("\n--- validation (separate untimed pass, CPU reference) ---\n");
    unsigned long long* ref = (unsigned long long*)malloc((size_t)MAXB * sizeof(unsigned long long));
    unsigned int* hout = (unsigned int*)malloc((size_t)(MAXB + 1) * sizeof(unsigned int));
    int fails = 0;
    for (int bi = 0; bi < 6; ++bi) {
        e.nBins = BINS[bi]; e.smemShared = e.nBins * 4;
        e.W = Wtab[bi]; e.P = Ptab[bi]; e.G = Gtab[bi];
        int occS = 0;
        if (valid[bi][S_SHARED]) {
            CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occS, (void*)k_shared,
                                                               BLK, e.smemShared));
            e.gridShared = occS * nSM;
        } else e.gridShared = e.grid;

        for (int d = 0; d < 2; ++d) {
            genFrac(d, h_raw, N);
            for (int b = 0; b < BINS[bi]; ++b) ref[b] = 0ull;
            for (size_t i = 0; i < N; ++i) ref[binOf(h_raw[i], BINS[bi])]++;

            for (int s = S_GLOBAL; s < NSTRAT; ++s) {
                if (!valid[bi][s]) continue;
                CHECK(cudaMemset(d_hist, 0, (size_t)(BINS[bi] + 1) * sizeof(unsigned int)));
                runStrat(s, d, e);
                CHECK_KERNEL();
                CHECK(cudaMemcpy(hout, d_hist, (size_t)BINS[bi] * sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost));
                int bad = 0;
                for (int b = 0; b < BINS[bi]; ++b)
                    if ((unsigned long long)hout[b] != ref[b]) bad++;
                if (bad) {
                    printf("  [FAIL] %-27s nBins=%5d %-8s : %d bins wrong\n",
                           SNAME[s], BINS[bi], DN[d], bad);
                    fails++;
                }
            }
        }
    }
    if (!fails) printf("  [PASS] every available (strategy, nBins, distribution) triple exact\n");

    // ------------------------------------------------------------ cleanup
    CHECK(cudaEventDestroy(evA)); CHECK(cudaEventDestroy(evB));
    for (int d = 0; d < 2; ++d) CHECK(cudaFree(d_raw[d]));
    CHECK(cudaFree(d_hist)); CHECK(cudaFree(d_part)); CHECK(cudaFree(d_sink));
    CHECK(cudaFree(cubTmp));
    free(h_raw); free(ref); free(hout);
    CHECK(cudaDeviceReset());

    printf("\nOVERALL: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
