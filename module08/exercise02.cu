// =====================================================================
// Module 8 / Exercise 2 : make the lanes agree.
//
// GOAL
//   A workload in which the amount of work per element is decided by the
//   DATA: element i needs work[i] rounds of refinement, work[i] in 1..8,
//   drawn from a deterministic seeded generator. The obvious kernel --
//   thread t handles element t -- puts all eight work classes in every
//   warp, so every warp runs at the maximum trip count in it. The grid
//   does N * mean(work) rounds of useful arithmetic and pays for
//   N * max(work).
//
//   Your job is to make the warps agree. HOW is up to you. The harness
//   times your version against the naive one, checks that your output is
//   bit-identical to the naive output, measures how homogeneous your
//   warps actually became, and prices the host-side preparation your
//   strategy needs.
//
//   Nothing below tells you which transformation to use. Several work.
//   They do not all cost the same.
//
// WHAT THE HARNESS REPORTS
//   * naive ms, your ms, and the ratio (the ratio is the stable number
//     on this laptop GPU; absolute ms move with the clock),
//   * warp homogeneity: the fraction of warps in which all 32 lanes have
//     the same work[] value,
//   * the host cost of your preparation, and the number of kernel
//     launches over which it pays for itself,
//   * correctness, bit-exact against the naive kernel.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
// RUN:    .\exercise02.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
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

static const int N       = 1 << 20;   // 1,048,576 elements
static const int UNIT    = 128;       // FFMAs in one round of refinement
static const int MAXWORK = 8;         // work[i] is in 1..8
static const int NAIVE_BLOCK = 256;

// ---------------------------------------------------------------------
// The per-element work. Identical in both kernels.
// ---------------------------------------------------------------------
__device__ __forceinline__ float refine(float x, int w)
{
    float a = x;
    for (int r = 0; r < w; ++r)
        for (int j = 0; j < UNIT; ++j) a = fmaf(a, 0.99999f, 1e-6f);
    return a;
}

// ---------------------------------------------------------------------
// The naive kernel: thread t handles element t.
// ---------------------------------------------------------------------
__global__ void refineNaive(const float* __restrict__ x,
                            const int*   __restrict__ work,
                            float*       __restrict__ out, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    out[t] = refine(x[t], work[t]);
}

// =====================================================================
// TODO 1 (design the strategy -- this is the exercise).
//
//   Fill `order` with n integers. The harness passes `order` to your
//   kernel and to nothing else, so its meaning is whatever you decide,
//   with one constraint the harness enforces: after your kernel runs,
//   out[i] must hold refine(x[i], work[i]) for every i in [0, n).
//
//   Return 1 when you have implemented it, 0 to leave it unimplemented.
//
//   Think about: what property must the 32 elements handled by one warp
//   share for the warp to stop paying for max(work)? How cheaply can you
//   arrange for that property to hold? `work` has only MAXWORK distinct
//   values -- does that change the cost of arranging it?
// =====================================================================
static int buildPlan(const int* work, int n, int* order)
{
    (void)work; (void)n; (void)order;
    // YOUR CODE HERE (TODO 1)
    return 0;
}

// =====================================================================
// TODO 2: the kernel that consumes your plan.
//
//   Thread t of this kernel must do the work of exactly one element.
//   Decide which one, read the right inputs, and write the result to the
//   right place. Getting the output index wrong is the classic way to
//   produce a fast kernel that is silently permuted; the harness
//   compares bit-exactly against the naive result, so it will catch you.
// =====================================================================
__global__ void refineFast(const float* __restrict__ x,
                           const int*   __restrict__ work,
                           float*       __restrict__ out,
                           const int*   __restrict__ order, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    int i = -1;                       // element this thread is responsible for
    // YOUR CODE HERE (TODO 2)

    if (i < 0 || i >= n) return;      // leave this guard in place
    out[i] = refine(x[i], work[i]);   // you may rewrite this line if your
                                      // strategy needs a different shape
}

// =====================================================================
// TODO 3: the launch configuration for refineFast.
//
//   Set FAST_BLOCK (threads per block). The grid is derived from it.
//   Your choice interacts with TODO 1: a plan that groups elements is
//   worth nothing if the block shape re-splits the groups.
// =====================================================================
static int FAST_BLOCK = 0;            // YOUR CODE HERE (TODO 3)

// =====================================================================
// TODO 4: before you run anything, commit to a predicted speedup of
//         refineFast over refineNaive, as a ratio (naive_ms / fast_ms).
//         Derive it from the work distribution and the SIMT cost model,
//         not from a guess. Scored within +/- 20%.
// =====================================================================
static const float PREDICTED_SPEEDUP = 0.0f;   // YOUR CODE HERE (TODO 4)

// =====================================================================
// A deterministic index-derived mixer, so the work distribution is
// reproducible on every machine and is not a low-period rand() pattern.
static unsigned mix32(unsigned z)
{
    z += 0x9E3779B9u;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

static float hostRefine(float x, int w)
{
    float a = x;
    for (int r = 0; r < w; ++r)
        for (int j = 0; j < UNIT; ++j) a = fmaf(a, 0.99999f, 1e-6f);
    return a;
}

// Fraction of warps whose 32 elements all carry the same work value.
// `map[t]` is the element index handled by thread t.
static double warpHomogeneity(const int* work, const int* map, int n, int block)
{
    int nthreads = ((n + block - 1) / block) * block;
    int warps = 0, homo = 0;
    for (int w0 = 0; w0 < nthreads; w0 += 32) {
        int first = -1, same = 1, any = 0;
        for (int l = 0; l < 32; ++l) {
            int t = w0 + l;
            if (t >= n) continue;
            int e = map ? map[t] : t;
            if (e < 0 || e >= n) continue;
            if (!any) { first = work[e]; any = 1; }
            else if (work[e] != first) same = 0;
        }
        if (any) { ++warps; homo += same; }
    }
    return warps ? (double)homo / (double)warps : 0.0;
}

int main(void)
{
    CHECK(cudaSetDevice(0));
    printf("=== Module 8 / Exercise 2 : make the lanes agree ===\n");
    printf("N = %d elements, work[i] in 1..%d, one round = %d FFMAs\n",
           N, MAXWORK, UNIT);

    // ---- deterministic input ----------------------------------------
    float* h_x    = (float*)malloc((size_t)N * sizeof(float));
    int*   h_work = (int*)  malloc((size_t)N * sizeof(int));
    int*   h_ord  = (int*)  malloc((size_t)N * sizeof(int));
    float* h_fast = (float*)malloc((size_t)N * sizeof(float));
    float* h_nv   = (float*)malloc((size_t)N * sizeof(float));
    if (!h_x || !h_work || !h_ord || !h_fast || !h_nv) { printf("host alloc failed\n"); return 1; }

    double wsum = 0.0;
    int hist[MAXWORK + 1];
    for (int k = 0; k <= MAXWORK; ++k) hist[k] = 0;
    for (int i = 0; i < N; ++i) {
        h_x[i]    = 0.5f + 1e-4f * (float)(i % 997);
        h_work[i] = 1 + (int)(mix32((unsigned)i + 0x9E3779B9u) & (unsigned)(MAXWORK - 1));
        wsum += h_work[i];
        ++hist[h_work[i]];
        h_ord[i]  = -1;
    }
    printf("work histogram:");
    for (int k = 1; k <= MAXWORK; ++k) printf(" %d:%d", k, hist[k]);
    printf("\nmean work = %.4f rounds, max = %d\n\n", wsum / N, MAXWORK);

    // ---- TODO 1 ------------------------------------------------------
    clock_t c0 = clock();
    int planned = buildPlan(h_work, N, h_ord);
    clock_t c1 = clock();
    double planMs = 1000.0 * (double)(c1 - c0) / (double)CLOCKS_PER_SEC;
    if (!planned) { printf("Set TODO 1 first.\n"); return 0; }

    if (FAST_BLOCK <= 0)           { printf("Set TODO 3 first.\n"); return 0; }
    if (FAST_BLOCK % 32 != 0)      { printf("FAST_BLOCK must be a multiple of 32.\n"); return 0; }
    if (FAST_BLOCK > 1024)         { printf("FAST_BLOCK must be <= 1024.\n"); return 0; }
    if (PREDICTED_SPEEDUP <= 0.0f) { printf("Set TODO 4 first.\n"); return 0; }

    // The plan must be a permutation, or the "bit-exact" test below is
    // meaningless.
    {
        char* seen = (char*)calloc((size_t)N, 1);
        if (!seen) { printf("host alloc failed\n"); return 1; }
        int bad = 0;
        for (int t = 0; t < N; ++t) {
            int e = h_ord[t];
            if (e < 0 || e >= N || seen[e]) { ++bad; break; }
            seen[e] = 1;
        }
        free(seen);
        if (bad) printf("NOTE: order[] is not a permutation of [0,N). "
                        "That is allowed, but your kernel must still write "
                        "every out[i] exactly once.\n");
    }

    // ---- device memory ----------------------------------------------
    float *d_x, *d_out, *d_ref;
    int *d_work, *d_ord;
    CHECK(cudaMalloc(&d_x,    (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_out,  (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_ref,  (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_work, (size_t)N * sizeof(int)));
    CHECK(cudaMalloc(&d_ord,  (size_t)N * sizeof(int)));
    CHECK(cudaMemcpy(d_x,    h_x,    (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_work, h_work, (size_t)N * sizeof(int),   cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ord,  h_ord,  (size_t)N * sizeof(int),   cudaMemcpyHostToDevice));

    const int naiveGrid = (N + NAIVE_BLOCK - 1) / NAIVE_BLOCK;
    const int fastGrid  = (N + FAST_BLOCK  - 1) / FAST_BLOCK;

    // ---- detect an unimplemented TODO 2 ------------------------------
    {
        float nan = nanf("");
        float* poison = (float*)malloc((size_t)N * sizeof(float));
        if (!poison) { printf("host alloc failed\n"); return 1; }
        for (int i = 0; i < N; ++i) poison[i] = nan;
        CHECK(cudaMemcpy(d_out, poison, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
        free(poison);
        refineFast<<<fastGrid, FAST_BLOCK>>>(d_x, d_work, d_out, d_ord, N);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_fast, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
        int untouched = 0;
        for (int i = 0; i < N; ++i) if (h_fast[i] != h_fast[i]) ++untouched;
        if (untouched == N) { printf("Set TODO 2 first.\n"); return 0; }
        if (untouched)
            printf("WARNING: %d of %d outputs were never written by refineFast.\n",
                   untouched, N);
    }

    // ---- clock warm-up (spec 12.4), duration based --------------------
    {
        cudaEvent_t w0, w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        float acc = 0.0f;
        while (acc < 300.0f) {
            CHECK(cudaEventRecord(w0));
            for (int k = 0; k < 10; ++k)
                refineNaive<<<naiveGrid, NAIVE_BLOCK>>>(d_x, d_work, d_ref, N);
            CHECK(cudaEventRecord(w1));
            CHECK(cudaEventSynchronize(w1));
            float ms; CHECK(cudaEventElapsedTime(&ms, w0, w1)); acc += ms;
        }
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    }

    // ---- timing: both configurations back-to-back, min of N sweeps ----
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));
    const int ITERS = 20, SWEEPS = 5;
    float best[2] = { 1e30f, 1e30f };
    for (int s = 0; s < SWEEPS; ++s) {
        for (int m = 0; m < 2; ++m) {
            if (m == 0) refineNaive<<<naiveGrid, NAIVE_BLOCK>>>(d_x, d_work, d_ref, N);
            else        refineFast <<<fastGrid,  FAST_BLOCK >>>(d_x, d_work, d_out, d_ord, N);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < ITERS; ++it) {
                if (m == 0) refineNaive<<<naiveGrid, NAIVE_BLOCK>>>(d_x, d_work, d_ref, N);
                else        refineFast <<<fastGrid,  FAST_BLOCK >>>(d_x, d_work, d_out, d_ord, N);
            }
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaEventSynchronize(ev1));
            float ms; CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            ms /= (float)ITERS;
            if (ms < best[m]) best[m] = ms;
        }
    }
    CHECK(cudaGetLastError());

    // ---- validation, second pass -------------------------------------
    int fails = 0;
    refineNaive<<<naiveGrid, NAIVE_BLOCK>>>(d_x, d_work, d_ref, N);
    CHECK(cudaGetLastError());
    refineFast<<<fastGrid, FAST_BLOCK>>>(d_x, d_work, d_out, d_ord, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_nv,   d_ref, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_fast, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));

    // (a) the naive kernel itself, against a CPU reference, on a sample
    {
        int bad = 0, checked = 0;
        for (int i = 0; i < N; i += 257) {
            float ref = hostRefine(h_x[i], h_work[i]);
            ++checked;
            if (!(fabsf(h_nv[i] - ref) <= 1e-5f * fmaxf(1.0f, fabsf(ref)))) ++bad;
        }
        printf("CPU cross-check of the naive kernel: %d/%d sampled elements agree\n",
               checked - bad, checked);
        if (bad) ++fails;
    }
    // (b) your kernel against the naive kernel, bit-exact, all elements
    {
        int bad = 0, firstBad = -1;
        for (int i = 0; i < N; ++i)
            if (h_fast[i] != h_nv[i]) { ++bad; if (firstBad < 0) firstBad = i; }
        printf("refineFast vs refineNaive: %d mismatches of %d", bad, N);
        if (bad) printf("   (first at i=%d: %.8f vs %.8f)", firstBad, h_fast[firstBad], h_nv[firstBad]);
        printf("\n");
        if (bad) ++fails;
    }

    // ---- diagnostics --------------------------------------------------
    double hNaive = warpHomogeneity(h_work, NULL,  N, NAIVE_BLOCK);
    double hFast  = warpHomogeneity(h_work, h_ord, N, FAST_BLOCK);

    printf("\n--- warp homogeneity (all 32 lanes share one work value) ---\n");
    printf("  naive (thread t -> element t), block %4d : %6.2f%%\n", NAIVE_BLOCK, 100.0 * hNaive);
    printf("  yours (thread t -> element order[t]), block %4d : %6.2f%%\n", FAST_BLOCK, 100.0 * hFast);

    printf("\n--- timing (min of %d sweeps of %d iterations) ---\n", SWEEPS, ITERS);
    printf("  refineNaive : %8.4f ms\n", best[0]);
    printf("  refineFast  : %8.4f ms\n", best[1]);
    double speedup = (double)best[0] / (double)best[1];
    printf("  speedup     : %.3fx    (you predicted %.3fx)\n", speedup, PREDICTED_SPEEDUP);
    double err = fabs(speedup - (double)PREDICTED_SPEEDUP) / speedup;
    int predOk = (err <= 0.20);
    printf("  prediction  : %s (%.1f%% off)\n", predOk ? "WITHIN 20%" : "OUTSIDE 20%", 100.0 * err);
    if (!predOk) ++fails;

    printf("\n--- cost of your host-side preparation ---\n");
    printf("  buildPlan   : %8.3f ms on the host\n", planMs);
    double savedPerLaunch = (double)best[0] - (double)best[1];
    if (savedPerLaunch > 0.0)
        printf("  saves %.4f ms per launch -> pays for itself after %.0f launches\n",
               savedPerLaunch, planMs / savedPerLaunch);
    else
        printf("  your kernel is not faster, so the preparation never pays for itself\n");

    // ---- cleanup ------------------------------------------------------
    free(h_x); free(h_work); free(h_ord); free(h_fast); free(h_nv);
    CHECK(cudaEventDestroy(ev0)); CHECK(cudaEventDestroy(ev1));
    CHECK(cudaFree(d_x)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_ref));
    CHECK(cudaFree(d_work)); CHECK(cudaFree(d_ord));
    CHECK(cudaDeviceReset());

    int pass = (fails == 0) && (speedup > 1.30);
    if (!pass && fails == 0)
        printf("\nCorrect, but the speedup is below the 1.30x bar this exercise asks for.\n");
    printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
