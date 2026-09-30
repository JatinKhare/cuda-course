// =============================================================================
// Module 14 / Exercise 3 — SOLUTION — three defects in a packed histogram
//
// BUILD
//   nvcc -arch=sm_89 -O3 -lineinfo -o exercise03_solution.exe exercise03_solution.cu
// RUN
//   .\exercise03_solution.exe
// =============================================================================

#include <cstdio>
#include <cstdlib>
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
    do { CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize()); } while (0)

static const size_t N        = 67108864;   // 2^26 keys
static const int    BLK      = 256;
static const int    BROKEN_G = 64;         // the grid hist_broken is run at
static const int    BINS[2]  = { 256, 1024 };
static const char*  DNAME[2] = { "uniform", "skewed" };

__device__ __host__ __forceinline__
unsigned int binOf(unsigned int frac, int nBins)
{
    return (unsigned int)(((unsigned long long)frac * (unsigned long long)nBins) >> 30);
}

// ============================================================ TODO 1 answers
// 1 = memcheck, 2 = racecheck, 3 = initcheck --initcheck-address-space shared,
// 4 = none of the four tools reports it.
static int DIAG[3] = { 3, 4, 4 };

// =============================================================================
// The kernel as shipped, with all three defects. Never edited.
// =============================================================================
__global__ void hist_broken(const unsigned int* __restrict__ in, size_t n,
                            unsigned int* __restrict__ hist, int nBins)
{
    extern __shared__ unsigned int s[];
    const int nWords = nBins >> 1;                 // two 16-bit bins per word

    if ((int)threadIdx.x < nWords) s[threadIdx.x] = 0u;      // DEFECT A

    // DEFECT B: no barrier here

    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        unsigned int b = binOf(in[i], nBins);
        atomicAdd(&s[b >> 1], (b & 1u) ? 0x10000u : 1u);     // DEFECT C
    }
    __syncthreads();

    for (int w = threadIdx.x; w < nWords; w += blockDim.x) {
        unsigned int v = s[w];
        if (v & 0xffffu) atomicAdd(&hist[2 * w],     v & 0xffffu);
        if (v >> 16)     atomicAdd(&hist[2 * w + 1], v >> 16);
    }
}

// Dirties shared memory so that a block which fails to zero its private bins
// sees the previous block's leftovers rather than a helpfully-zero SM.
__global__ void dirty_shared(unsigned int* out)
{
    extern __shared__ unsigned int d[];
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) d[i] = 0xDEADBEEFu;
    __syncthreads();
    if (d[threadIdx.x] == 0u) out[0] = 1u;
}

// =============================================================================
// The fixed kernel. TODOs 2, 3 and 4 live here.
// =============================================================================
__global__ void hist_student(const unsigned int* __restrict__ in, size_t n,
                             unsigned int* __restrict__ hist, int nBins)
{
    extern __shared__ unsigned int s[];
    const int nWords = nBins >> 1;

    // TODO 2 — every word, not just the first blockDim.x of them.
    for (int w = threadIdx.x; w < nWords; w += blockDim.x) s[w] = 0u;

    // TODO 3 — the zeroing must be complete and visible to the whole block
    // before any thread's first atomicAdd.
    __syncthreads();

    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        unsigned int b = binOf(in[i], nBins);
        atomicAdd(&s[b >> 1], (b & 1u) ? 0x10000u : 1u);
    }
    __syncthreads();

    for (int w = threadIdx.x; w < nWords; w += blockDim.x) {
        unsigned int v = s[w];
        if (v & 0xffffu) atomicAdd(&hist[2 * w],     v & 0xffffu);
        if (v >> 16)     atomicAdd(&hist[2 * w + 1], v >> 16);
    }
}

// The packed representation is preserved: two bins per 32-bit word.
static int sharedBytes(int nBins) { return (nBins >> 1) * (int)sizeof(unsigned int); }

// ==================================================================== TODO 4
// A 16-bit counter holds 0..65535. The largest value any private bin of a block
// can reach is the number of elements that block processes, which is
// ceil(n / (grid * blockDim)) * blockDim <= n / grid + blockDim. Require that to
// stay at or below 65535 whatever the distribution does, and the grid follows.
// This is the coarsening ceiling the packed representation buys its 2x
// memory saving with.
static int chooseGridSafe(size_t n, int /*nBins*/, int blockDim)
{
    const size_t CAP = 65535;                       // per-bin capacity of a half-word
    size_t need = (n + (CAP - (size_t)blockDim)) / (CAP - (size_t)blockDim);
    int g = (int)need;
    if (g < 1) g = 1;
    if (g > 1000000) g = 1000000;
    return g;
}

// =============================================================== input data
static unsigned int xs32(unsigned int* st)
{
    unsigned int x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return x;
}

// dist 0: uniform. dist 1: 75% of the mass in the lowest 4/1024 of the range,
// which is 1 bin at nBins=256 and 4 bins at nBins=1024.
static void gen(int dist, unsigned int* h, size_t n)
{
    unsigned int st = (dist == 0) ? 0x13579BDFu : 0x2468ACE0u;
    for (size_t i = 0; i < n; ++i) {
        unsigned int r = xs32(&st);
        double u = (double)(xs32(&st) >> 8) * (1.0 / 16777216.0);
        unsigned int f;
        if (dist == 1 && (r & 3u) != 3u) f = (unsigned int)(u * (1073741824.0 / 256.0));
        else                             f = (unsigned int)(u * 1073741824.0);
        h[i] = f > 1073741823u ? 1073741823u : f;
    }
}

static unsigned int fnv1a(const char* s)
{
    unsigned int h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("=== Module 14 / Exercise 3 (SOLUTION) — three defects, packed bins ===\n");
    printf("N = %zu keys; hist_broken is launched with grid = %d, block = %d\n",
           N, BROKEN_G, BLK);

    int grid = chooseGridSafe(N, BINS[1], BLK);
    if (grid <= 0) { printf("Set TODO 4 (chooseGridSafe) first.\n"); return 0; }
    if (DIAG[0] == 0 || DIAG[1] == 0 || DIAG[2] == 0) {
        printf("Set TODO 1 (the three tool codes) first.\n"); return 0;
    }

    unsigned int* h_in = (unsigned int*)malloc(N * sizeof(unsigned int));
    unsigned long long* ref = (unsigned long long*)malloc(1024 * sizeof(unsigned long long));
    unsigned int* h_out = (unsigned int*)malloc(1024 * sizeof(unsigned int));
    if (!h_in || !ref || !h_out) { printf("host alloc failed\n"); return 1; }

    unsigned int* d_in[2];
    for (int d = 0; d < 2; ++d) {
        CHECK(cudaMalloc(&d_in[d], N * sizeof(unsigned int)));
        gen(d, h_in, N);
        CHECK(cudaMemcpy(d_in[d], h_in, N * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    unsigned int* d_hist; CHECK(cudaMalloc(&d_hist, 1024 * sizeof(unsigned int)));
    unsigned int* d_flag; CHECK(cudaMalloc(&d_flag, sizeof(unsigned int)));
    CHECK(cudaMemset(d_flag, 0, sizeof(unsigned int)));

    // ------------------------------------------------- the symptoms, as shipped
    printf("\n--- hist_broken, as shipped (grid = %d) ---\n", BROKEN_G);
    printf("  %-8s %6s   %14s %14s   %s\n", "dist", "nBins", "total", "expected", "ratio");
    for (int bi = 0; bi < 2; ++bi)
        for (int d = 0; d < 2; ++d) {
            CHECK(cudaMemset(d_hist, 0, 1024 * sizeof(unsigned int)));
            dirty_shared<<<BROKEN_G * 4, BLK, 8192>>>(d_flag);
            CHECK(cudaDeviceSynchronize());
            hist_broken<<<BROKEN_G, BLK, sharedBytes(BINS[bi])>>>(d_in[d], N, d_hist, BINS[bi]);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(h_out, d_hist, (size_t)BINS[bi] * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
            unsigned long long tot = 0ull;
            for (int b = 0; b < BINS[bi]; ++b) tot += h_out[b];
            printf("  %-8s %6d   %14llu %14zu   %8.3fx\n",
                   DNAME[d], BINS[bi], tot, N, (double)tot / (double)N);
        }

    // -------------------------------------------------------- the fixed kernel
    printf("\n--- hist_student, your grid = %d (%.0f elements per block) ---\n",
           grid, (double)N / grid);
    printf("  worst-case per-bin count in one block: %.0f  (a 16-bit half holds 65535)\n",
           (double)N / grid + BLK);

    int score = 0, maxScore = 0, allExact = 1;
    for (int bi = 0; bi < 2; ++bi)
        for (int d = 0; d < 2; ++d) {
            gen(d, h_in, N);
            for (int b = 0; b < BINS[bi]; ++b) ref[b] = 0ull;
            for (size_t i = 0; i < N; ++i) ref[binOf(h_in[i], BINS[bi])]++;

            int bad = 0;
            unsigned long long tot = 0ull;
            for (int rep = 0; rep < 3; ++rep) {     // determinism: 3 identical runs
                CHECK(cudaMemset(d_hist, 0, 1024 * sizeof(unsigned int)));
                dirty_shared<<<grid, BLK, 8192>>>(d_flag);
                CHECK(cudaDeviceSynchronize());
                hist_student<<<grid, BLK, sharedBytes(BINS[bi])>>>(d_in[d], N, d_hist, BINS[bi]);
                CHECK_KERNEL();
                CHECK(cudaMemcpy(h_out, d_hist, (size_t)BINS[bi] * sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost));
                tot = 0ull;
                for (int b = 0; b < BINS[bi]; ++b) {
                    tot += h_out[b];
                    if ((unsigned long long)h_out[b] != ref[b]) bad++;
                }
            }
            printf("  [%s] %-8s nBins=%4d : %d wrong bins over 3 runs, total %llu\n",
                   bad ? "FAIL" : "PASS", DNAME[d], BINS[bi], bad, tot);
            if (bad) allExact = 0;
        }
    score += allExact; maxScore += 1;

    int packed = 1;
    for (int bi = 0; bi < 2; ++bi)
        if (sharedBytes(BINS[bi]) > BINS[bi] * 2) packed = 0;
    printf("  [%s] the packed representation is preserved: %d B for %d bins, %d B for %d bins\n",
           packed ? "PASS" : "FAIL", sharedBytes(BINS[0]), BINS[0], sharedBytes(BINS[1]), BINS[1]);
    score += packed; maxScore += 1;

    int gridOK = ((double)N / grid + BLK) <= 65535.0;
    printf("  [%s] the grid bounds the per-block per-bin count below 65536\n",
           gridOK ? "PASS" : "FAIL");
    score += gridOK; maxScore += 1;

    char buf[16];
    snprintf(buf, sizeof(buf), "%d%d%d", DIAG[0], DIAG[1], DIAG[2]);
    unsigned int want = 1096448066u;
    int diagOK = (fnv1a(buf) == want);
    printf("  [%s] TODO 1 diagnosis codes (%s)\n", diagOK ? "PASS" : "FAIL", buf);
    score += diagOK; maxScore += 1;

    printf("\n  score: %d/%d\n", score, maxScore);

    for (int d = 0; d < 2; ++d) CHECK(cudaFree(d_in[d]));
    CHECK(cudaFree(d_hist)); CHECK(cudaFree(d_flag));
    free(h_in); free(ref); free(h_out);
    CHECK(cudaDeviceReset());

    printf("OVERALL: %s\n", score == maxScore ? "PASS" : "FAIL");
    return score == maxScore ? 0 : 1;
}
