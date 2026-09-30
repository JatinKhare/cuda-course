// =====================================================================
// Module 9 / Exercise 2 SOLUTION : "It hangs. Diagnose it, then design it away."
//
// SYMPTOM
//   A kernel computes a scale factor in one block and applies it in all
//   the others. The blocks that need the scale spin on a global flag
//   until the producing block raises it. With a small grid the program
//   finishes instantly and the answer is right. With a large grid --
//   the same code, the same data, only more blocks -- it stalls for
//   seconds and most blocks report that they gave up waiting.
//
//   A second kernel in this file, a tiled shift with a bounds guard,
//   produces wrong answers only on the last, partially-filled block.
//
//   Nothing in this file is a compile error and nothing is an illegal
//   memory access. compute-sanitizer --tool memcheck reports nothing.
//
// !!  THIS PROGRAM SPINS ON PURPOSE  !!
//   The spin loops here are BOUNDED by an escape counter, so the
//   program always terminates -- but the large-grid case can take
//   several seconds during which the GPU sits at 100% and the desktop
//   may stutter. If you remove the escape counter (and you should, once,
//   deliberately, to see what the real bug looks like) the kernel will
//   never return. Kill it from a second shell:
//       taskkill /F /IM exercise02_solution.exe      (Windows)
//       kill -9 <pid>                       (Linux)
//   The GPU recovers on process exit. No reboot is required. Do not run
//   the unbounded version under a debugger or a profiler.
//
// YOUR JOB
//   TODO 1  commit to two predictions before running anything.
//   TODO 2  compute, from the runtime API, how many blocks of this
//           kernel can be co-resident on this GPU at one time.
//   TODO 3  fix the tiled-shift kernel.
//   TODO 4  make the scale-and-apply computation correct for ANY grid
//           size, on any CUDA device, without a cooperative launch and
//           without any cross-block spinning. This is a design problem,
//           not a one-line fix.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02.cu
// RUN:    .\exercise02_solution.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
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

#define CHECK_KERNEL() do {                                                \
    CHECK(cudaGetLastError());                                             \
    CHECK(cudaDeviceSynchronize());                                        \
} while (0)

static const int   TPB       = 256;
static const int   SHIFT     = 96;
#define SCALE 2.5f
static const long long MAXSPIN = 150000LL;   // escape counter

// =====================================================================
//  PART 1 -- the cross-block spin
// =====================================================================
// The last block of the grid produces `*scale`. Every block, including
// the producer, then waits for the flag and applies the scale.
//
// atomicExch / atomicAdd appear here only to make the flag a
// well-defined, non-cacheable single word. Module 10 is where atomics
// are a topic; here the atomic is nothing but a flag.
__global__ void scale_spin(const float* __restrict__ in,
                           float* __restrict__ out,
                           float* scale, int* flag,
                           int* timeouts, int n)
{
    __shared__ int saw_flag;

    if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {
        *scale = SCALE;
        __threadfence();              // device scope: publish the value
                                      // BEFORE the flag that advertises it
        atomicExch(flag, 1);
    }

    if (threadIdx.x == 0) {
        long long spins = 0;
        int f = 0;
        while ((f = atomicAdd(flag, 0)) == 0 && spins < MAXSPIN) ++spins;
        if (f) __threadfence();       // acquire side: do not read `scale`
                                      // with values cached from before
        else   atomicAdd(timeouts, 1);
        saw_flag = f;
    }
    __syncthreads();                  // block-uniform, well defined

    const float s = saw_flag ? *scale : 0.0f;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = in[gid] * s;
}

// =====================================================================
//  PART 2 -- TODO 3.  A tiled shift. Correct on every full block,
//  wrong on the last one. blockDim.x == TPB always; n is NOT a
//  multiple of TPB.
//
//  Reference semantics: with base = blockIdx.x*TPB and
//  src = base + ((t + SHIFT) % TPB),
//      out[gid] = (src < n) ? in[src] : 0.0f      for every gid < n.
//
//  Do not change the signature and do not change what the kernel
//  computes. Note that `s` is uninitialized for any slot whose owning
//  thread did not write it.
// =====================================================================
__global__ void shift_tile(const float* __restrict__ in,
                           float* __restrict__ out, int n)
{
    __shared__ float s[TPB];
    const int t    = threadIdx.x;
    const int base = blockIdx.x * TPB;
    const int gid  = base + t;

    // The guard was doing two jobs: keeping the global accesses in
    // range, and -- illegally -- deciding who reaches the barrier. Split
    // them. Every thread of the block stages something (a neutral 0 for
    // out-of-range slots), every thread reaches the barrier, and only
    // the global stores stay guarded.
    s[t] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();                       // reached by the whole block
    if (gid < n) out[gid] = s[(t + SHIFT) & (TPB - 1)];
}

// =====================================================================
//  PART 3 -- TODO 4.
//
//  Produce the scale and apply it to all n elements, correctly, for a
//  grid of ANY size, with no cross-block spin-waiting and no
//  cooperative launch. You may add as many __global__ kernels as you
//  like ABOVE this function, and you may issue as many launches as you
//  like inside it.
//
//  Think about what, in the CUDA execution model, is the only ordering
//  primitive that is guaranteed to be visible to every block of a grid
//  no matter how the blocks were scheduled. Module 1 told you that a
//  block, once placed on an SM, runs to completion and never migrates,
//  and that a block that is not resident does not exist. Both halves of
//  that sentence matter here.
//
//  Return 1 when you have implemented it, 0 to skip.
// =====================================================================

__global__ void produce_scale(float* scale)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) *scale = SCALE;
}

__global__ void consume_scale(const float* __restrict__ in,
                              float* __restrict__ out,
                              const float* __restrict__ scale, int n)
{
    const float s   = *scale;
    const int   gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = in[gid] * s;
}

static int apply_scale_any_grid(const float* d_in, float* d_out,
                                float* d_scale, int n, int grid)
{
    // The kernel boundary IS the grid-wide barrier. Every block of
    // launch k has completed, and every memory operation it performed
    // is visible device-wide, before any block of launch k+1 starts.
    // It costs a launch (~4-6 us) and it is the only grid-wide ordering
    // guarantee that holds for an arbitrary grid size.
    produce_scale<<<1, 32>>>(d_scale);
    consume_scale<<<grid, TPB>>>(d_in, d_out, d_scale, n);
    return 1;
}

// =====================================================================
//  YOUR PREDICTIONS
// =====================================================================
// TODO 1: before you run anything.
//   P1  With grid = the number of blocks that fit on the GPU at once,
//       how many blocks will report a spin timeout?
//   P2  With grid = 8192 blocks, how many blocks will report a spin
//       timeout?  Give a NUMBER, not "lots". P2 is accepted within
//       +/-5%, because you should be able to derive it exactly from
//       one quantity this program prints and one you already know.
//   Use -1 for "not answered".
static const long long P1_small_grid_timeouts = 0;
static const long long P2_large_grid_timeouts = 8192 - 240;

// TODO 2: fill in the body. Return the maximum number of blocks of
//         `scale_spin` (at TPB threads per block, 0 bytes of dynamic
//         shared memory) that can be simultaneously resident on this
//         device. Derive it from the runtime API -- do not hard-code
//         40, and do not hard-code 24 either. Return 0 to skip.
static int max_resident_blocks(void)
{
    int sms = 0, perSM = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess)
        return 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &perSM, (const void*)scale_spin, TPB, 0) != cudaSuccess)
        return 0;
    return sms * perSM;
}

// ---------------------------------------------------------------------
static int mismatch(const float* got, const float* ref, int n)
{
    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (fabsf(got[i] - ref[i]) > 1e-5f * fmaxf(1.0f, fabsf(ref[i]))) ++bad;
    return bad;
}

int main(void)
{
    if (P1_small_grid_timeouts < 0 || P2_large_grid_timeouts < 0) {
        printf("Set TODO 1 first.\n");
        return 0;
    }

    const int resident = max_resident_blocks();
    if (resident <= 0) { printf("Set TODO 2 first.\n"); return 0; }

    const int LARGE = 8192;
    const int n     = LARGE * TPB - 37;      // deliberately not a multiple

    float* h_in  = (float*)malloc((size_t)n * sizeof(float));
    float* h_out = (float*)malloc((size_t)n * sizeof(float));
    float* h_ref = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; ++i)
        h_in[i] = (float)((unsigned)(i * 1664525u + 1013904223u) % 977u) * 0.001f;

    float *d_in, *d_out, *d_scale;
    int   *d_flag, *d_to;
    CHECK(cudaMalloc(&d_in,    (size_t)n * sizeof(float)));
    CHECK(cudaMalloc(&d_out,   (size_t)n * sizeof(float)));
    CHECK(cudaMalloc(&d_scale, sizeof(float)));
    CHECK(cudaMalloc(&d_flag,  sizeof(int)));
    CHECK(cudaMalloc(&d_to,    sizeof(int)));
    CHECK(cudaMemcpy(d_in, h_in, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    printf("=== Part 0: what the hardware says ===\n");
    int sms = 0;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int perSM = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM,
              (const void*)scale_spin, TPB, 0));
    const int truth = sms * perSM;
    printf("  SMs = %d, blocks/SM for scale_spin at %d threads = %d\n", sms, TPB, perSM);
    printf("  co-resident blocks (API)      : %d\n", truth);
    printf("  your max_resident_blocks()    : %d  -> %s\n",
           resident, (resident == truth) ? "correct" : "WRONG");

    // ---- Part 1: small grid ------------------------------------------
    printf("\n=== Part 1: cross-block spin ===\n");
    const int SMALL = truth;
    long long to_small = 0, to_large = 0;
    for (int pass = 0; pass < 2; ++pass) {
        const int grid = pass ? LARGE : SMALL;
        const int nn   = pass ? n : SMALL * TPB;
        CHECK(cudaMemset(d_flag,  0, sizeof(int)));
        CHECK(cudaMemset(d_to,    0, sizeof(int)));
        CHECK(cudaMemset(d_scale, 0, sizeof(float)));
        CHECK(cudaMemset(d_out,   0, (size_t)n * sizeof(float)));
        printf("  grid = %5d blocks ... ", grid); fflush(stdout);
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
        CHECK(cudaEventRecord(e0));
        scale_spin<<<grid, TPB>>>(d_in, d_out, d_scale, d_flag, d_to, nn);
        CHECK(cudaEventRecord(e1));
        CHECK_KERNEL();
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
        int to = 0;
        CHECK(cudaMemcpy(&to, d_to, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_out, d_out, (size_t)nn * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < nn; ++i) h_ref[i] = h_in[i] * SCALE;
        const int bad = mismatch(h_out, h_ref, nn);
        printf("%8.2f ms, %6d blocks timed out, %d wrong elems\n", ms, to, bad);
        if (pass) to_large = to; else to_small = to;
    }
    const int p1ok = (P1_small_grid_timeouts == to_small);
    const double lo = 0.95 * (double)to_large, hi = 1.05 * (double)to_large;
    const int p2ok = ((double)P2_large_grid_timeouts >= lo &&
                      (double)P2_large_grid_timeouts <= hi);
    printf("  P1 predicted %lld, observed %lld -> %s\n",
           P1_small_grid_timeouts, to_small, p1ok ? "correct" : "WRONG");
    printf("  P2 predicted %lld, observed %lld (+/-5%% accepted) -> %s\n",
           P2_large_grid_timeouts, to_large, p2ok ? "correct" : "WRONG");
    printf("  ratio observed-timeouts / co-resident-blocks = %.2f\n",
           truth ? (double)to_large / (double)truth : 0.0);

    // ---- Part 2: shift_tile ------------------------------------------
    printf("\n=== Part 2: tiled shift with a bounds guard ===\n");
    for (int b = 0; b < LARGE; ++b)
        for (int t = 0; t < TPB; ++t) {
            const int gid = b * TPB + t;
            if (gid >= n) continue;
            const int src = b * TPB + ((t + SHIFT) & (TPB - 1));
            h_ref[gid] = (src < n) ? h_in[src] : 0.0f;
        }
    CHECK(cudaMemset(d_out, 0, (size_t)n * sizeof(float)));
    shift_tile<<<LARGE, TPB>>>(d_in, d_out, n);
    CHECK_KERNEL();
    CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    const int bad_tile = mismatch(h_out, h_ref, n);
    int bad_tail = 0;
    for (int i = (LARGE - 1) * TPB; i < n; ++i)
        if (fabsf(h_out[i] - h_ref[i]) > 1e-5f) ++bad_tail;
    printf("  wrong elems total %d, of which in the last block %d -> %s\n",
           bad_tile, bad_tail, bad_tile == 0 ? "PASS" : "FAIL");

    // ---- Part 3: TODO 4 ----------------------------------------------
    printf("\n=== Part 3: no cross-block synchronization at all ===\n");
    CHECK(cudaMemset(d_out,   0, (size_t)n * sizeof(float)));
    CHECK(cudaMemset(d_scale, 0, sizeof(float)));
    const int impl = apply_scale_any_grid(d_in, d_out, d_scale, n, LARGE);
    int bad4 = -1;
    if (!impl) {
        printf("  TODO 4 not implemented -- skipping.\n");
    } else {
        CHECK_KERNEL();
        CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) h_ref[i] = h_in[i] * SCALE;
        bad4 = mismatch(h_out, h_ref, n);
        printf("  wrong elems: %d -> %s\n", bad4, bad4 == 0 ? "PASS" : "FAIL");
    }

    const int pass = p1ok && p2ok && (resident == truth) &&
                     (bad_tile == 0) && (bad4 == 0);
    printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");

    free(h_in); free(h_out); free(h_ref);
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_scale));
    CHECK(cudaFree(d_flag)); CHECK(cudaFree(d_to));
    CHECK(cudaDeviceReset());
    return 0;
}
