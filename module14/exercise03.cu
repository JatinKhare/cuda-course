// =============================================================================
// Module 14 / Exercise 3 — three defects in a packed histogram (debugging)
//
// WHAT THE KERNEL IS SUPPOSED TO DO
//   `hist_broken` is a privatized histogram that halves its shared-memory
//   footprint by packing TWO 16-bit bin counters into every 32-bit word. That is
//   a real technique: 1024 bins cost 2 KB instead of 4 KB, which is the
//   difference between 6 and 5 resident blocks per SM at some sizes.
//
// THE SYMPTOMS (causes NOT given)
//   The harness runs it on two bin counts and two input distributions and prints
//   the totals. As shipped, on this GPU:
//
//     S1  nBins = 256,  uniform input : exactly correct, every time.
//     S2  nBins = 256,  skewed  input : total is about 0.25x the true count,
//                                       and it is perfectly reproducible.
//     S3  nBins = 1024, either input  : total is 15-25x the true count and
//                                       differs from run to run.
//
//   There are THREE defects and only two visible failure modes. One defect
//   produces no symptom at all on this hardware in this configuration; you will
//   find it by reading the kernel and reasoning about what the block is allowed
//   to assume, not by running anything.
//
// WHAT YOU DO
//   TODO 1  commit to which sanitizer tool finds each defect, BEFORE running any
//           of them. Scored against a hash; the answer is not in this file.
//   TODO 2,3  repair two of the defects in `hist_student` (a verbatim copy of
//           `hist_broken`). Stated as requirements, not as mechanisms.
//   TODO 4  repair the third WITHOUT abandoning the packed representation. The
//           harness checks that you kept it.
//
// BUILD
//   nvcc -arch=sm_89 -O3 -lineinfo -o exercise03.exe exercise03.cu
// RUN
//   .\exercise03.exe
//   compute-sanitizer --tool memcheck   .\exercise03.exe
//   compute-sanitizer --tool racecheck  .\exercise03.exe
//   compute-sanitizer --tool initcheck --initcheck-address-space shared .\exercise03.exe
//   compute-sanitizer --tool synccheck  .\exercise03.exe
//
//   (-lineinfo is what makes the tools name a source line. memcheck also prints
//    a benign "Resetting device while there are still other users claiming to
//    use it" warning because main() calls cudaDeviceReset(); that is not a
//    memory error and it is counted in ERROR SUMMARY anyway.)
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

// ==================================================================== TODO 1
// PREDICT BEFORE YOU RUN ANY TOOL.
//
// There are three defects, labelled A, B and C by the order in which they
// appear in `hist_broken` below. For each, say which of these reports it:
//
//     1  compute-sanitizer --tool memcheck
//     2  compute-sanitizer --tool racecheck
//     3  compute-sanitizer --tool initcheck --initcheck-address-space shared
//     4  none of the four tools reports it at all
//
// Fill DIAG[0], DIAG[1], DIAG[2] with defect A's, B's and C's codes. All three
// must be right; there is no partial credit and the answer is stored only as a
// hash. Before you guess at defect B in particular, look carefully at what kind
// of instruction the *conflicting* access is, and ask yourself what a race
// detector is entitled to assume about that kind of instruction.
//
// YOUR CODE HERE
static int DIAG[3] = { 0, 0, 0 };

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

    // TODO 2 — make the initialization of the private bins correct for ANY
    // (nBins, blockDim.x) pair, not just the ones where the arithmetic happens
    // to work out. The harness runs nBins = 256 and nBins = 1024 with 256
    // threads; work out for yourself which of those two the shipped line is
    // wrong for, and why the other one hides it.
    //
    // YOUR CODE HERE
    if ((int)threadIdx.x < nWords) s[threadIdx.x] = 0u;

    // TODO 3 — supply the guarantee the code between here and the accumulation
    // loop is silently assuming. Name it precisely in your own notes: it is one
    // of the two things Module 9 says the construct provides, and only one.
    //
    // YOUR CODE HERE

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

// Two bins per 32-bit word. The harness checks that this still returns no more
// than nBins * 2 bytes, i.e. that you did not "fix" defect C by widening the
// counters. Widening works and is usually right; here it is banned, because the
// point is to find the OTHER fix and to see what it costs.
static int sharedBytes(int nBins) { return (nBins >> 1) * (int)sizeof(unsigned int); }

// ==================================================================== TODO 4
// DESIGN. Defect C is a capacity failure: the representation can hold a value
// only up to some bound, and on one of the two input distributions a private
// bin exceeds it. Widening the counter is banned (see above), so the only
// remaining move is to make the bound unreachable.
//
// Return a grid size for `hist_student` that makes the bound unreachable for
// EVERY possible input distribution, including "all n elements land in one
// bin". Derive it; do not tune it. The harness prints the worst-case per-block
// per-bin count your answer implies and checks it against the bound.
//
// Then write down, for yourself, what this constraint costs: which of the four
// techniques in this module does it put a ceiling on, and what does that ceiling
// do to the flush traffic?
//
// Returning 0 stops the program.
//
// YOUR CODE HERE
static int chooseGridSafe(size_t n, int nBins, int blockDim)
{
    (void)n; (void)nBins; (void)blockDim;
    return 0;
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

    printf("=== Module 14 / Exercise 3 — three defects, packed bins ===\n");
    printf("N = %zu keys; hist_broken is launched with grid = %d, block = %d\n",
           N, BROKEN_G, BLK);

    int grid = chooseGridSafe(N, BINS[1], BLK);
    const int gatesUnset = (grid <= 0) || (DIAG[0] == 0 || DIAG[1] == 0 || DIAG[2] == 0);
    if (gatesUnset)
        printf("\n(TODO 1 and/or TODO 4 are unset: the symptom table below still runs,\n"
               " so that compute-sanitizer has something to look at, and then the\n"
               " program stops without scoring.)\n");

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

    if (gatesUnset) {
        printf("\nSet TODO 1 (the three tool codes) and TODO 4 (chooseGridSafe) first.\n");
        for (int d = 0; d < 2; ++d) CHECK(cudaFree(d_in[d]));
        CHECK(cudaFree(d_hist)); CHECK(cudaFree(d_flag));
        free(h_in); free(ref); free(h_out);
        CHECK(cudaDeviceReset());
        return 0;
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
