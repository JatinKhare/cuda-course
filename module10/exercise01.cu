// =====================================================================
// Module 10 / Exercise 1 : "Three symptoms, how many bugs?"
//
// THE SITUATION
//   `triage` scans a stream of 4,000,000 events. For each event it:
//     (1) computes a severity class 0..7 and counts it,
//     (2) adds the event's value into a global checksum,
//     (3) if the event is critical (class 7), appends its index to a
//         bounded global list of capacity CRIT_CAP.
//
//   It is in production. Three symptoms have been reported:
//
//     S1. The per-class counts vary from run to run and are always too
//         low. Their sum is nowhere near 4,000,000.
//     S2. The checksum is wrong by orders of magnitude, always low, and
//         also varies run to run.
//     S3. The critical-event list is full of holes: slots well below the
//         reported count still hold their initial 0xFFFFFFFF. The
//         reported count is far smaller than the true number of critical
//         events -- and not because the list filled up, since the count
//         never even reaches the capacity.
//
//   Nothing in the file crashes. compute-sanitizer --tool memcheck is
//   clean. The symptoms are all wrong ANSWERS, not wrong ACCESSES.
//
// YOUR JOB
//   Diagnose, classify, and fix. `triage_broken` below is the shipped
//   code and you must NOT edit it -- the harness runs it to show you the
//   damage. `triage_fixed` ships as a byte-for-byte copy of it. Fix
//   `triage_fixed`.
//
//   Before you fix anything, run the tool and fill in TODO 1. The
//   prediction is scored and you cannot pass this exercise without it.
//
//     nvcc -arch=sm_89 -O3 -lineinfo -o exercise01.exe exercise01.cu
//     compute-sanitizer --tool racecheck .\exercise01.exe --racecheck
//
//   The `--racecheck` argument makes the program launch `triage_broken`
//   exactly once on a small input and exit, so the tool's report is about
//   one launch of one kernel and is stable run to run. (The hazard COUNTS
//   still vary; the number of `Error:` blocks does not.) On Windows the
//   launcher is `compute-sanitizer.bat`, which the toolkit puts on PATH.
//
// BUILD: nvcc -arch=sm_89 -O3 -lineinfo -o exercise01.exe exercise01.cu
// RUN:   .\exercise01.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(x) do {                                                      \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                  \
                cudaGetErrorName(e_), __FILE__, __LINE__,                  \
                cudaGetErrorString(e_));                                   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

static const int N        = 4000000;
static const int NCLASS   = 8;
static const int CRIT_CAP = 4096;     // deliberately smaller than the true
                                      // number of critical events
static const int TPB      = 256;

// Severity class of an event value. Identical on host and device so the
// reference cannot drift from the kernel.
__host__ __device__ inline int severity(unsigned int v)
{
    // 0..7, with class 7 ("critical") deliberately rare.
    unsigned int h = v;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return (h % 100u) < 3u ? 7 : (int)(h % 7u);
}

// ---------------------------------------------------------------------
// TODO 1: PREDICTIONS. Run `compute-sanitizer --tool racecheck` on this
//         program BEFORE changing anything, read its output, and fill in
//         all four values. Leave any of them at -1 and the program stops.
//
//   P_HAZARDS  : how many `Error: Race reported ...` blocks does racecheck
//                print for the single triage_broken launch?
//   P_CHECKSUM : does racecheck report the defect responsible for
//                symptom S2?  1 = yes, 0 = no
//   P_SLOTS    : does racecheck report the defect responsible for
//                symptom S3?  1 = yes, 0 = no
//   P_CLASSES  : how many DISTINCT source lines inside triage_broken does
//                racecheck name? (count each line number once)
//
//         Think about what the tool is instrumented to see before you
//         run it. Then run it. If your prediction was wrong, the
//         interesting question is not "what is the answer" but "what is
//         this tool's model of a race, such that this is the answer".
// ---------------------------------------------------------------------
static const int P_HAZARDS  = -1;   // YOUR CODE HERE (TODO 1)
static const int P_CHECKSUM = -1;   // YOUR CODE HERE (TODO 1)
static const int P_SLOTS    = -1;   // YOUR CODE HERE (TODO 1)
static const int P_CLASSES  = -1;   // YOUR CODE HERE (TODO 1)

// =====================================================================
// THE SHIPPED CODE -- DO NOT EDIT. The harness runs it for contrast.
// =====================================================================
__global__ void triage_broken(const unsigned int* __restrict__ ev, int n,
                              unsigned int* gCount,
                              unsigned long long* gChecksum,
                              int* gCritCount, int* gCritList)
{
    __shared__ unsigned int sCount[NCLASS];

    if (threadIdx.x < NCLASS) sCount[threadIdx.x] = 0u;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int v = ev[i];
        int c = severity(v);
        sCount[c] += 1u;
        *gChecksum += (unsigned long long)v;
        if (c == 7) {
            int slot = *gCritCount;
            *gCritCount = slot + 1;
            if (slot < CRIT_CAP) gCritList[slot] = i;
        }
    }
    __syncthreads();

    if (threadIdx.x < NCLASS) gCount[threadIdx.x] += sCount[threadIdx.x];
}

// =====================================================================
// YOUR VERSION. Ships identical to the above. Fix it.
//
// Requirements -- each must hold for EVERY launch configuration, not just
// the one the harness uses:
//
//   TODO 2: Any two threads that classify their events into the same class
//           must BOTH be counted -- whether they are in the same block or
//           in different blocks. There are two places where that can fail
//           and they are not the same kind of failure.
//
//   TODO 3: A block's private counters must be zero before any thread of
//           that block adds to them, and every thread's contribution must
//           be complete before any thread reads them for the flush. State
//           to yourself which mechanism gives you each of those two
//           properties and why one mechanism is not enough for both.
//
//   TODO 4: The checksum must equal the exact sum of all n event values,
//           every run, bit for bit.
//
//   TODO 5: Every critical event must be counted, and the first CRIT_CAP
//           of them (in any order) must occupy slots 0..CRIT_CAP-1 with
//           no slot written twice and no slot left unwritten. No write
//           may ever go past CRIT_CAP. *gCritCount must end up holding
//           the TRUE number of critical events even when that exceeds
//           CRIT_CAP, so the caller can tell that the list overflowed and
//           by how much.
//
//           Note that the last two sentences pull in opposite directions.
//           Decide how to reconcile them; there is more than one workable
//           answer and the harness accepts any of them.
// =====================================================================
__global__ void triage_fixed(const unsigned int* __restrict__ ev, int n,
                             unsigned int* gCount,
                             unsigned long long* gChecksum,
                             int* gCritCount, int* gCritList)
{
    __shared__ unsigned int sCount[NCLASS];

    if (threadIdx.x < NCLASS) sCount[threadIdx.x] = 0u;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int v = ev[i];
        int c = severity(v);
        sCount[c] += 1u;                                   // TODO 2
        *gChecksum += (unsigned long long)v;               // TODO 4
        if (c == 7) {
            int slot = *gCritCount;                        // TODO 5
            *gCritCount = slot + 1;                        // TODO 5
            if (slot < CRIT_CAP) gCritList[slot] = i;
        }
    }
    __syncthreads();                                       // TODO 3

    if (threadIdx.x < NCLASS) gCount[threadIdx.x] += sCount[threadIdx.x];  // TODO 2
}

// ---------------------------------------------------------------------
// The prediction is checked by hash so that this file does not contain the
// answer. Get all four right and the digest matches; there is no partial
// credit and no way to read the answer off the screen.
static unsigned int pred_digest(int a, int b, int c, int d)
{
    unsigned int h = 2166136261u;
    const int v[4] = { a, b, c, d };
    for (int k = 0; k < 4; ++k) {
        h ^= (unsigned int)(v[k] + 7);
        h *= 16777619u;
        h ^= h >> 11;
    }
    return h;
}

int main(int argc, char** argv)
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    // ---- `--racecheck`: one launch of the broken kernel, nothing else ----
    for (int a = 1; a < argc; ++a) {
        if (strcmp(argv[a], "--racecheck") != 0) continue;
        const int SMALL = 8192;
        unsigned int* h = (unsigned int*)malloc(SMALL * sizeof(unsigned int));
        unsigned int s = 20250101u;
        for (int i = 0; i < SMALL; ++i) { s = s * 1664525u + 1013904223u; h[i] = s >> 3; }
        unsigned int* dev = nullptr; unsigned int* dc = nullptr;
        unsigned long long* ds = nullptr; int *dcc = nullptr, *dcl = nullptr;
        CHECK(cudaMalloc(&dev, SMALL * sizeof(unsigned int)));
        CHECK(cudaMalloc(&dc, NCLASS * sizeof(unsigned int)));
        CHECK(cudaMalloc(&ds, sizeof(unsigned long long)));
        CHECK(cudaMalloc(&dcc, sizeof(int)));
        CHECK(cudaMalloc(&dcl, CRIT_CAP * sizeof(int)));
        CHECK(cudaMemcpy(dev, h, SMALL * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dc, 0, NCLASS * sizeof(unsigned int)));
        CHECK(cudaMemset(ds, 0, sizeof(unsigned long long)));
        CHECK(cudaMemset(dcc, 0, sizeof(int)));
        triage_broken<<<SMALL / TPB, TPB>>>(dev, SMALL, dc, ds, dcc, dcl);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        free(h);
        CHECK(cudaFree(dev)); CHECK(cudaFree(dc)); CHECK(cudaFree(ds));
        CHECK(cudaFree(dcc)); CHECK(cudaFree(dcl));
        // No cudaDeviceReset() on this path: under compute-sanitizer it
        // emits a spurious "resetting device while there are still other
        // users" API warning that clutters the report you are here to read.
        printf("racecheck mode: one launch of triage_broken, done.\n");
        return 0;
    }

    printf("Device: %s (sm_%d%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    if (P_HAZARDS < 0 || P_CHECKSUM < 0 || P_SLOTS < 0 || P_CLASSES < 0) {
        printf("Set TODO 1 first: run\n");
        printf("  compute-sanitizer --tool racecheck .\\exercise01.exe --racecheck\n");
        printf("and record what it reports.\n");
        return 0;
    }

    // ---- deterministic input ----
    unsigned int* h_ev = (unsigned int*)malloc((size_t)N * sizeof(unsigned int));
    unsigned int seed = 20250101u;
    for (int i = 0; i < N; ++i) {
        seed = seed * 1664525u + 1013904223u;
        h_ev[i] = seed >> 3;
    }

    // ---- CPU reference ----
    unsigned int refCount[NCLASS];
    memset(refCount, 0, sizeof(refCount));
    unsigned long long refChecksum = 0ull;
    int refCrit = 0;
    for (int i = 0; i < N; ++i) {
        int c = severity(h_ev[i]);
        refCount[c] += 1u;
        refChecksum += (unsigned long long)h_ev[i];
        if (c == 7) ++refCrit;
    }

    unsigned int* d_ev = nullptr;
    unsigned int* d_count = nullptr;
    unsigned long long* d_sum = nullptr;
    int *d_cc = nullptr, *d_cl = nullptr;
    CHECK(cudaMalloc(&d_ev, (size_t)N * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_count, NCLASS * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_cc, sizeof(int)));
    CHECK(cudaMalloc(&d_cl, CRIT_CAP * sizeof(int)));
    CHECK(cudaMemcpy(d_ev, h_ev, (size_t)N * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const int GRID = (N + TPB - 1) / TPB;

    unsigned int  h_count[NCLASS];
    unsigned long long h_sum = 0ull;
    int h_cc = 0;
    int* h_cl = (int*)malloc(CRIT_CAP * sizeof(int));
    char* touched = (char*)malloc((size_t)CRIT_CAP);

    int score = 0, maxScore = 0;

    // ============ run both kernels, twice each, and report ============
    for (int which = 0; which < 2; ++which) {
        const char* name = which ? "triage_fixed " : "triage_broken";
        printf("=== %s ===\n", name);
        unsigned int first[NCLASS]; unsigned long long firstSum = 0ull; int firstCC = 0;
        int stable = 1;

        for (int run = 0; run < 3; ++run) {
            CHECK(cudaMemset(d_count, 0, NCLASS * sizeof(unsigned int)));
            CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
            CHECK(cudaMemset(d_cc, 0, sizeof(int)));
            CHECK(cudaMemset(d_cl, 0xFF, CRIT_CAP * sizeof(int)));

            if (which) triage_fixed <<<GRID, TPB>>>(d_ev, N, d_count, d_sum, d_cc, d_cl);
            else       triage_broken<<<GRID, TPB>>>(d_ev, N, d_count, d_sum, d_cc, d_cl);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(h_count, d_count, NCLASS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(&h_cc, d_cc, sizeof(int), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h_cl, d_cl, CRIT_CAP * sizeof(int), cudaMemcpyDeviceToHost));

            if (run == 0) { memcpy(first, h_count, sizeof(first)); firstSum = h_sum; firstCC = h_cc; }
            else {
                if (memcmp(first, h_count, sizeof(first)) != 0 ||
                    firstSum != h_sum || firstCC != h_cc) stable = 0;
            }

            if (run == 2) {
                long long tot = 0;
                printf("  class counts : ");
                for (int c = 0; c < NCLASS; ++c) { printf("%u ", h_count[c]); tot += h_count[c]; }
                printf("\n  reference    : ");
                for (int c = 0; c < NCLASS; ++c) printf("%u ", refCount[c]);
                printf("\n  sum of counts: %lld (should be %d)\n", tot, N);
                printf("  checksum     : %llu (should be %llu)\n", h_sum, refChecksum);
                printf("  crit count   : %d (should be %d, cap %d)\n", h_cc, refCrit, CRIT_CAP);

                memset(touched, 0, (size_t)CRIT_CAP);
                int dup = 0, unwritten = 0, outOfRange = 0;
                for (int s = 0; s < CRIT_CAP; ++s) {
                    int idx = h_cl[s];
                    if (idx == -1) { ++unwritten; continue; }
                    if (idx < 0 || idx >= N || severity(h_ev[idx]) != 7) ++outOfRange;
                }
                // duplicates among the recorded indices
                for (int s = 0; s < CRIT_CAP; ++s)
                    for (int t = s + 1; t < CRIT_CAP && !dup; ++t)
                        if (h_cl[s] != -1 && h_cl[s] == h_cl[t]) ++dup;
                printf("  crit list    : %d unwritten slots, %s duplicates, "
                       "%d non-critical entries\n\n",
                       unwritten, dup ? "HAS" : "no", outOfRange);

                // ---------------- scoring (fixed kernel only) ----------------
                if (which) {
                    int ok;
                    maxScore = 6;
                    ok = (memcmp(refCount, h_count, sizeof(refCount)) == 0);
                    printf("  [%s] per-class counts exact\n", ok ? "PASS" : "FAIL"); score += ok;
                    ok = (tot == N);
                    printf("  [%s] counts sum to N\n", ok ? "PASS" : "FAIL"); score += ok;
                    ok = (h_sum == refChecksum);
                    printf("  [%s] checksum exact\n", ok ? "PASS" : "FAIL"); score += ok;
                    ok = (h_cc == refCrit);
                    printf("  [%s] gCritCount reports the TRUE critical count (%d)\n",
                           ok ? "PASS" : "FAIL", refCrit); score += ok;
                    ok = (unwritten == 0 && dup == 0 && outOfRange == 0);
                    printf("  [%s] crit list: %d slots filled, unique, all critical\n",
                           ok ? "PASS" : "FAIL", CRIT_CAP); score += ok;
                    ok = stable;
                    printf("  [%s] identical results across 3 runs\n", ok ? "PASS" : "FAIL");
                    score += ok;
                    printf("\n");
                }
            }
        }
        if (!which)
            printf("  results identical across 3 runs: %s\n\n", stable ? "yes" : "no");
    }

    // ============ score the prediction ============
    printf("=== TODO 1 prediction ===\n");
    const unsigned int EXPECTED_DIGEST = 0xC2EE6360u;
    const unsigned int got = pred_digest(P_HAZARDS, P_CHECKSUM, P_SLOTS, P_CLASSES);
    const int pOk = (got == EXPECTED_DIGEST);
    printf("  you predicted: hazards=%d  S2caught=%d  S3caught=%d  lines=%d\n",
           P_HAZARDS, P_CHECKSUM, P_SLOTS, P_CLASSES);
    printf("  [%s] prediction (all four must be right; no partial credit)\n",
           pOk ? "PASS" : "FAIL");
    if (!pOk)
        printf("        re-read the racecheck output, and in particular the\n"
               "        one-line description the tool gives of itself in\n"
               "        `compute-sanitizer --help`.\n");
    printf("\n");

    printf("correctness score: %d/%d\n", score, maxScore);
    const int pass = (score == maxScore && pOk);
    printf("OVERALL: %s\n", pass ? "PASS" : "FAIL");

    free(h_ev); free(h_cl); free(touched);
    CHECK(cudaFree(d_ev)); CHECK(cudaFree(d_count)); CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_cc)); CHECK(cudaFree(d_cl));
    CHECK(cudaDeviceReset());
    return pass ? 0 : 1;
}
