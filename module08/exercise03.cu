// =====================================================================
// Module 8 / Exercise 3 : a kernel that used to be idiomatic.
//
// SYMPTOM
//   `warpSmoothBroken` implements a per-warp ring relaxation: on every
//   step, lane L takes the value currently held by lane L+1 (mod 32),
//   adds one, and stores it as its own new value. After STEPS steps,
//   lane L should hold in[L + STEPS (mod 32)] + STEPS.
//
//   It does not. On this GPU it produces the wrong answer for a large
//   fraction of the elements, and it produces the SAME wrong answer on
//   every run. There is no illegal access; compute-sanitizer memcheck is
//   clean. The kernel contains a __syncthreads() and a `volatile`
//   qualifier, which is what its author believed made it safe.
//
//   Nothing here is about shared-memory bank conflicts (Module 7) or
//   about block-wide barriers (Module 9). The defect is in this module's
//   subject: what the hardware guarantees about the lanes of one warp.
//
// YOUR JOB
//   1. Work out why the kernel is wrong. `warpSmoothBroken` is fixed --
//      do not modify it; it is the evidence.
//   2. Write `warpSmoothFixed` so that it computes the intended result.
//   3. Answer the three diagnostic questions (TODO 2, 3, 4). They are
//      scored; guessing costs you.
//
//   A third kernel, `warpSmoothConverged`, is shipped alongside. It uses
//   the same `volatile __shared__` idiom with no warp-level
//   synchronization, but performs its exchange outside any divergent
//   region. The harness runs it and reports what it does. Read TODO 4
//   before you decide what that result means.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN:    .\exercise03.exe
//
// Worth doing:
//   nvcc -arch=sm_89 -O3 -c -o exercise03.o exercise03.cu
//   cuobjdump -sass exercise03.o > exercise03.sass
//   compute-sanitizer --tool memcheck  .\exercise03.exe
//   compute-sanitizer --tool racecheck .\exercise03.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
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

static const int BLOCK = 128;      // 4 warps per block
static const int GRID  = 4096;
static const int N     = BLOCK * GRID;
static const int STEPS = 8;

// =====================================================================
// THE EVIDENCE. Do not change this kernel.
//
// Written in the pre-Volta warp-synchronous style: `volatile` on the
// shared array, one __syncthreads() to publish the initial load, and no
// synchronization at all inside the warp, because "the 32 lanes of a
// warp execute in lockstep, so they cannot get out of step with each
// other."
// =====================================================================
__global__ void warpSmoothBroken(const int* __restrict__ in,
                                 int* __restrict__ out,
                                 unsigned* __restrict__ maskProbe)
{
    volatile __shared__ int s[BLOCK];

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * BLOCK + tid;
    int lane = tid & 31;
    int base = tid & ~31;

    s[tid] = in[gid];
    __syncthreads();

    float t = 1.0f;
    for (int step = 0; step < STEPS; ++step) {
        if (lane & 1) {
            // odd lanes carry an extra per-lane refinement
            for (int j = 0; j < 300; ++j) t = fmaf(t, 1.00001f, 1e-7f);
            if (blockIdx.x == 0 && step == 0) maskProbe[lane] = __activemask();
            int v = s[base + ((lane + 1) & 31)];
            s[tid] = v + 1;
        } else {
            if (blockIdx.x == 0 && step == 0) maskProbe[lane] = __activemask();
            int v = s[base + ((lane + 1) & 31)];
            s[tid] = v + 1;
        }
    }
    if (t < 0.0f) s[tid] = 0;          // keeps `t` alive; never taken
    out[gid] = s[tid];
}

// =====================================================================
// The same idiom, but with the exchange in a region where the whole warp
// is together. Shipped for TODO 4. Do not change this kernel either.
// =====================================================================
__global__ void warpSmoothConverged(const int* __restrict__ in,
                                    int* __restrict__ out)
{
    volatile __shared__ int s[BLOCK];

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * BLOCK + tid;
    int lane = tid & 31;
    int base = tid & ~31;

    s[tid] = in[gid];
    __syncthreads();

    for (int step = 0; step < STEPS; ++step) {
        int v = s[base + ((lane + 1) & 31)];
        s[tid] = v + 1;
    }
    out[gid] = s[tid];
}

// =====================================================================
// TODO 1 (design the fix).
//
//   Produce the intended result: after STEPS steps, lane L must hold
//   in[L + STEPS (mod 32)] + STEPS, for every warp of every block.
//
//   Requirements, stated as requirements and not as an API call:
//     * every lane must read the value its neighbour held at the START
//       of the step, never a value that neighbour wrote during the same
//       step;
//     * the mechanism you use to guarantee that must be executed by a
//       warp-uniform set of lanes -- a device-wide or block-wide barrier
//       reached by only part of a warp is not a legal fix, and neither
//       is a warp-level one;
//     * the per-lane refinement loop on odd lanes must still happen (it
//       is what the real kernel this is abstracted from was doing), so
//       you cannot simply delete the divergence.
//
//   You may restructure the loop body freely. You may use any shared
//   memory layout you like. Two quite different correct answers exist.
// =====================================================================
__global__ void warpSmoothFixed(const int* __restrict__ in,
                                int* __restrict__ out)
{
    __shared__ int s[BLOCK];

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * BLOCK + tid;
    // Available if you want them:
    //   int lane = tid & 31;
    //   int base = tid & ~31;

    s[tid] = in[gid];
    __syncthreads();

    // YOUR CODE HERE (TODO 1)

    out[gid] = s[tid];
}

// =====================================================================
// TODO 2: the active mask.
//
//   In warpSmoothBroken, the line
//       int v = s[base + ((lane + 1) & 31)];
//   appears twice, once in each arm. Warp 0 of block 0 records
//   __activemask() immediately before it, on step 0, in each arm.
//
//   Give the value recorded by an ODD lane and the value recorded by an
//   EVEN lane. 0 means "not answered".
// =====================================================================
static const unsigned MASK_ODD_ARM  = 0x00000000u;  // YOUR CODE HERE (TODO 2)
static const unsigned MASK_EVEN_ARM = 0x00000000u;  // YOUR CODE HERE (TODO 2)

// =====================================================================
// TODO 3: what kind of failure is this?
//
//   Set BROKEN_IS_DETERMINISTIC to 1 if you expect warpSmoothBroken to
//   produce exactly the same output on every run of this program, and 0
//   if you expect the number of wrong elements to vary run to run.
//   Justify it to yourself from the execution model before answering;
//   the answer tells you which debugging tools are useful here.
// =====================================================================
static int BROKEN_IS_DETERMINISTIC = -1;   // YOUR CODE HERE (TODO 3)

// =====================================================================
// TODO 4: two claims about the classic idiom. Answer 1 for true,
//         0 for false, leave -1 for unanswered.
//
//   (a) VOLATILE_FIXES_IT
//       "The bug in warpSmoothBroken is that the compiler cached s[] in
//        a register. Marking the array `volatile` is what makes the
//        warp-synchronous idiom correct."
//
//   (b) CONVERGED_VARIANT_IS_GUARANTEED
//       "warpSmoothConverged does the same exchange with no warp-level
//        synchronization and the harness reports that it passes.
//        Therefore the CUDA programming model guarantees that this code
//        is correct on sm_89."
// =====================================================================
static int VOLATILE_FIXES_IT             = -1;   // YOUR CODE HERE (TODO 4)
static int CONVERGED_VARIANT_IS_GUARANTEED = -1; // YOUR CODE HERE (TODO 4)

// =====================================================================
static int popc32(unsigned v) { int c = 0; while (v) { c += (int)(v & 1u); v >>= 1; } return c; }

int main(void)
{
    if (MASK_ODD_ARM == 0u || MASK_EVEN_ARM == 0u) { printf("Set TODO 2 first.\n"); return 0; }
    if (BROKEN_IS_DETERMINISTIC < 0)               { printf("Set TODO 3 first.\n"); return 0; }
    if (VOLATILE_FIXES_IT < 0 ||
        CONVERGED_VARIANT_IS_GUARANTEED < 0)       { printf("Set TODO 4 first.\n"); return 0; }

    CHECK(cudaSetDevice(0));
    printf("=== Module 8 / Exercise 3 : a kernel that used to be idiomatic ===\n");
    printf("%d blocks x %d threads = %d elements, %d steps, warp ring rotate\n\n",
           GRID, BLOCK, N, STEPS);

    int* h_in  = (int*)malloc((size_t)N * sizeof(int));
    int* h_out = (int*)malloc((size_t)N * sizeof(int));
    int* h_ref = (int*)malloc((size_t)N * sizeof(int));
    if (!h_in || !h_out || !h_ref) { printf("host alloc failed\n"); return 1; }
    for (int i = 0; i < N; ++i) h_in[i] = (i * 37) % 1013;

    // CPU reference: the intended lockstep semantics.
    for (int b = 0; b < GRID; ++b)
        for (int w = 0; w < BLOCK / 32; ++w) {
            int cur[32], nxt[32];
            for (int l = 0; l < 32; ++l) cur[l] = h_in[b * BLOCK + w * 32 + l];
            for (int st = 0; st < STEPS; ++st) {
                for (int l = 0; l < 32; ++l) nxt[l] = cur[(l + 1) & 31] + 1;
                for (int l = 0; l < 32; ++l) cur[l] = nxt[l];
            }
            for (int l = 0; l < 32; ++l) h_ref[b * BLOCK + w * 32 + l] = cur[l];
        }

    int *d_in, *d_out; unsigned* d_mask;
    CHECK(cudaMalloc(&d_in,  (size_t)N * sizeof(int)));
    CHECK(cudaMalloc(&d_out, (size_t)N * sizeof(int)));
    CHECK(cudaMalloc(&d_mask, 32 * sizeof(unsigned)));
    CHECK(cudaMemcpy(d_in, h_in, (size_t)N * sizeof(int), cudaMemcpyHostToDevice));

    int score = 0, total = 0;

    // ---------------- the broken kernel, several times ----------------
    long long firstBad = -1; int varies = 0;
    unsigned hmask[32];
    for (int trial = 0; trial < 10; ++trial) {
        CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(int)));
        CHECK(cudaMemset(d_mask, 0, 32 * sizeof(unsigned)));
        warpSmoothBroken<<<GRID, BLOCK>>>(d_in, d_out, d_mask);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
        if (trial == 0) CHECK(cudaMemcpy(hmask, d_mask, sizeof(hmask), cudaMemcpyDeviceToHost));
        long long bad = 0;
        for (int i = 0; i < N; ++i) if (h_out[i] != h_ref[i]) ++bad;
        if (trial == 0) firstBad = bad;
        else if (bad != firstBad) varies = 1;
    }
    printf("--- warpSmoothBroken ---\n");
    printf("  wrong elements: %lld of %d (%.2f%%), identical count over 10 runs: %s\n",
           firstBad, N, 100.0 * (double)firstBad / N, varies ? "no" : "yes");

    printf("  recorded __activemask() just before the exchange, warp 0 of block 0:\n");
    printf("    odd  lane (lane 1)  : 0x%08x  popc=%d   you predicted 0x%08x\n",
           hmask[1], popc32(hmask[1]), MASK_ODD_ARM);
    printf("    even lane (lane 0)  : 0x%08x  popc=%d   you predicted 0x%08x\n",
           hmask[0], popc32(hmask[0]), MASK_EVEN_ARM);
    int m1 = (MASK_ODD_ARM == hmask[1]);
    int m2 = (MASK_EVEN_ARM == hmask[0]);
    printf("    TODO 2: %s / %s\n", m1 ? "MATCH" : "MISMATCH", m2 ? "MATCH" : "MISMATCH");
    score += m1 + m2; total += 2;

    int detActual = varies ? 0 : 1;
    int m3 = (BROKEN_IS_DETERMINISTIC == detActual);
    printf("    TODO 3: predicted deterministic=%d, actual=%d  %s\n",
           BROKEN_IS_DETERMINISTIC, detActual, m3 ? "MATCH" : "MISMATCH");
    score += m3; ++total;

    // ---------------- the converged variant ---------------------------
    CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(int)));
    warpSmoothConverged<<<GRID, BLOCK>>>(d_in, d_out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
    long long badConv = 0;
    for (int i = 0; i < N; ++i) if (h_out[i] != h_ref[i]) ++badConv;
    printf("\n--- warpSmoothConverged (same idiom, no divergence) ---\n");
    printf("  wrong elements: %lld of %d\n", badConv, N);
    printf("  (what this does and does not prove is TODO 4b)\n");

    // ---------------- TODO 4 -------------------------------------------
    int m4 = (VOLATILE_FIXES_IT == 0);
    int m5 = (CONVERGED_VARIANT_IS_GUARANTEED == 0);
    printf("\n--- TODO 4 ---\n");
    printf("  (a) VOLATILE_FIXES_IT             = %d   %s\n",
           VOLATILE_FIXES_IT, m4 ? "MATCH" : "MISMATCH");
    printf("  (b) CONVERGED_VARIANT_IS_GUARANTEED = %d   %s\n",
           CONVERGED_VARIANT_IS_GUARANTEED, m5 ? "MATCH" : "MISMATCH");
    score += m4 + m5; total += 2;

    // ---------------- the fixed kernel ---------------------------------
    CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(int)));
    warpSmoothFixed<<<GRID, BLOCK>>>(d_in, d_out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
    long long badFix = 0, firstIdx = -1;
    for (int i = 0; i < N; ++i)
        if (h_out[i] != h_ref[i]) { ++badFix; if (firstIdx < 0) firstIdx = i; }

    printf("\n--- warpSmoothFixed (TODO 1) ---\n");
    if (badFix == N) {
        printf("  every element wrong -- TODO 1 looks unimplemented.\n");
        printf("  Set TODO 1 first.\n");
    } else {
        printf("  wrong elements: %lld of %d", badFix, N);
        if (badFix) printf("   (first at i=%lld: got %d, want %d)",
                           firstIdx, h_out[firstIdx], h_ref[firstIdx]);
        printf("\n");
    }
    int fixOk = (badFix == 0);
    score += fixOk; ++total;

    // ---------------- stability of the fix -----------------------------
    if (fixOk) {
        int stable = 1;
        for (int trial = 0; trial < 10; ++trial) {
            CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(int)));
            warpSmoothFixed<<<GRID, BLOCK>>>(d_in, d_out);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < N; ++i) if (h_out[i] != h_ref[i]) { stable = 0; break; }
            if (!stable) break;
        }
        printf("  10 further runs: %s\n", stable ? "all correct" : "NOT stable");
        score += stable; ++total;
    } else { ++total; }

    printf("\nSCORE: %d/%d\n", score, total);

    free(h_in); free(h_out); free(h_ref);
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_mask));
    CHECK(cudaDeviceReset());
    printf("OVERALL: %s\n", score == total ? "PASS" : "FAIL");
    return score == total ? 0 : 1;
}
