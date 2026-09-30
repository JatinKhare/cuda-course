// =====================================================================
// Module 9 / Exercise 3 SOLUTION : Fence or barrier? -- a design decision
//
// THE PATTERN
//   Inside one block, threads are paired: thread t and thread
//   t + 128 are each other's peer. The block runs R rounds. In every
//   round each thread computes PAY payload words into shared memory,
//   publishes them, and then consumes its peer's payload from the same
//   round. The result of consuming becomes the seed for the next round,
//   so the rounds are genuinely dependent -- you cannot reorder them.
//
//   The amount of producer work is SKEWED and the skew MOVES: in every
//   round exactly one thread of the block does ~30x the work of the
//   others, and it is a different thread each round.
//
//   Two implementations of the same dataflow:
//
//     variant_flag     point-to-point. Each thread publishes a round
//                      number in its own flag; each consumer waits on
//                      exactly one flag -- its peer's -- and on nothing
//                      else. You supply the ordering.
//
//     variant_barrier  phase-separated. Everybody produces, the whole
//                      block synchronizes once per round, everybody
//                      consumes.
//
//   Both are correct. They do not cost the same, and the reason is not
//   the cost of the instruction.
//
// WHY SPINNING IS LEGAL HERE
//   Every thread of a block is resident on one SM at the same time
//   (Module 1), and on sm_70+ independent thread scheduling guarantees
//   forward progress for diverged threads (Module 8). A spin INSIDE a
//   block therefore terminates. The same spin ACROSS blocks does not --
//   that is Exercise 2. Nothing in this file can hang: the thread you
//   are waiting for is guaranteed to be running.
//
// YOUR JOB
//   TODO 1  producer side: stated as a requirement, not a mechanism.
//   TODO 2  consumer side: the wait, and what it must guarantee.
//   TODO 3  write variant_barrier.
//   TODO 4  predict which variant is faster on this GPU, and by how
//           much, before you measure.
//
//   For TODO 1 and TODO 2 there is more than one construct that makes
//   the program correct, and they do not cost the same either. Decide
//   which you chose and what the alternative would have cost.
//
// BUILD:  nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler /Zc:preprocessor -o exercise03_solution.exe exercise03_solution.cu
//         (libcu++ <cuda/atomic> requires C++17; MSVC additionally
//          requires the conforming preprocessor. On Linux the
//          -Xcompiler flag is unnecessary.)
// RUN:    .\exercise03_solution.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda/atomic>

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

static const int TPB   = 256;
static const int PAY   = 4;
static const int NBLK  = 1024;
static const int PEERD = 128;          // peer = (t + PEERD) % TPB
static const int ROUNDS = 16;
static const int BASEW = 200;          // baseline producer iterations
static const int HEAVY = 6000;         // extra iterations for one thread

// Exact and identical on host and device -- no floating point, so the
// final comparison is an equality, not a tolerance.
__host__ __device__ __forceinline__
unsigned burn(unsigned x, int iters)
{
    for (int i = 0; i < iters; ++i) x = x * 1664525u + 1013904223u;
    return x;
}
// Exactly one heavy thread per block per round, and it moves.
__host__ __device__ __forceinline__
int work_for(int t, int r)
{
    return BASEW + (((t + 37 * r) & (TPB - 1)) == 0 ? HEAVY : 0);
}

// A block-scope atomic reference. Module 10 is where atomics are a
// topic; here the atomic is nothing but a flag, and it exists only so
// that the flag word is a well-defined object the compiler may not keep
// in a register. `memory_order_relaxed` asks for NO ordering, which is
// why the ordering question below is still yours to answer.
using flag_ref = cuda::atomic_ref<int, cuda::thread_scope_block>;

// =====================================================================
//  VARIANT 1 -- point to point
// =====================================================================
__global__ void variant_flag(const unsigned* __restrict__ seed,
                             unsigned* __restrict__ out)
{
    // Two payload buffers, selected by round parity. A pair of peers
    // can never be more than one round apart, so two is enough and
    // there is no write-after-read hazard to synchronize against.
    __shared__ unsigned payload[2][TPB * PAY];
    __shared__ int      flag[TPB];

    const int t    = threadIdx.x;
    const int base = blockIdx.x * TPB;
    const int peer = (t + PEERD) & (TPB - 1);

    flag[t] = 0;
    __syncthreads();         // one-time initialisation of the flags.
                             // Not part of the exercise.

    unsigned acc = seed[base + t];
    flag_ref fr(flag[peer]);

    for (int r = 0; r < ROUNDS; ++r) {
        const int buf = r & 1;

        // ---------------- produce ----------------
        const unsigned v = burn(acc, work_for(t, r));
        for (int k = 0; k < PAY; ++k) payload[buf][t * PAY + k] = v + (unsigned)k;

        // TODO 1: REQUIREMENT -- any thread of this block that observes
        //         flag[t] == r+1 must also observe all PAY payload
        //         words this thread just wrote for round r. Supply
        //         whatever makes that true, and nothing more. In
        //         particular this thread has no reason to wait for
        //         anybody at this point, so do not make it.
        //
        // A block-scope release fence: it orders THIS thread's prior
        // shared writes ahead of everything it does afterwards, as the
        // rest of the block sees it. It blocks nobody.
        __threadfence_block();

        flag_ref(flag[t]).store(r + 1, cuda::memory_order_relaxed);

        // ---------------- consume ----------------
        // TODO 2: REQUIREMENT -- do not read the peer's round-r payload
        //         until the peer has published it, and make sure the
        //         payload words this thread then reads cannot be values
        //         it obtained before it saw the flag reach r+1. The
        //         wait must also terminate: re-reading a value the
        //         compiler hoisted into a register never will.
        //
        // The acquire load does both jobs: it is a real load every time
        // round the loop (so the wait terminates), and it orders the
        // payload reads after it (so they cannot be satisfied from
        // before the flag reached r+1).
        while (fr.load(cuda::memory_order_acquire) < r + 1) { /* spin */ }

        unsigned s = 0;
        for (int k = 0; k < PAY; ++k) s += payload[buf][peer * PAY + k];
        acc = s;
    }
    out[base + t] = acc;
}

// =====================================================================
//  VARIANT 2 -- TODO 3.
//
//  Same dataflow, same output, phase-separated: every thread produces
//  round r, the block synchronizes, every thread consumes round r.
//  No flags, no spinning.
//
//  Decide for yourself what shared state you need and where the
//  synchronization goes -- including how many synchronizations per
//  round you actually need, and why.
//
//  `out[base + t]` must equal exactly what variant_flag produces.
//  Leave the body as shipped and the program reports
//  "TODO 3 not implemented" and skips it.
// =====================================================================
__global__ void variant_barrier(const unsigned* __restrict__ seed,
                                unsigned* __restrict__ out)
{
    const int t    = threadIdx.x;
    const int base = blockIdx.x * TPB;

    // Double-buffered by round parity, exactly as variant_flag is, so
    // that ONE barrier per round suffices: round r+1 writes the buffer
    // nobody is reading, so there is no write-after-read hazard and the
    // only barrier needed is the read-after-write one.
    __shared__ unsigned payload[2][TPB * PAY];

    const int peer = (t + PEERD) & (TPB - 1);
    unsigned acc = seed[base + t];

    for (int r = 0; r < ROUNDS; ++r) {
        const int buf = r & 1;
        const unsigned v = burn(acc, work_for(t, r));
        for (int k = 0; k < PAY; ++k) payload[buf][t * PAY + k] = v + (unsigned)k;

        __syncthreads();   // execution barrier AND block-scope fence.
                           // Both guarantees are used; one instruction
                           // buys both.

        unsigned s = 0;
        for (int k = 0; k < PAY; ++k) s += payload[buf][peer * PAY + k];
        acc = s;
    }
    out[base + t] = acc;
}

// =====================================================================
//  TODO 4 -- predict before you measure.
//
//  PREDICT_FASTER : which variant has the lower min-of-N kernel time on
//                   this GPU?  FASTER_FLAG or FASTER_BARRIER.
//  PREDICT_RATIO  : slower_ms / faster_ms. Accepted within +/-30%.
//
//  Both variants run exactly the same arithmetic, touch exactly the
//  same shared memory and execute exactly the same number of rounds.
//  The only difference is the synchronization. So reason about the
//  critical path, not about instruction counts:
//
//    - In each round, how long does the block take to get past the
//      synchronization point in each design?
//    - The heavy thread MOVES from round to round. Write down the total
//      time for each design as a sum over rounds. One of them is a sum
//      of maxima. The other is closer to a maximum of sums. Those are
//      not the same number.
//    - Module 8: the unit that serializes on a divergent trip count is
//      the warp, not the thread. There are 8 warps here and the peer of
//      warp w is warp w+4.
// =====================================================================
#define FASTER_UNANSWERED 0
#define FASTER_FLAG       1
#define FASTER_BARRIER    2

static const int    PREDICT_FASTER = FASTER_FLAG;
static const double PREDICT_RATIO  = 1.36;

// ---------------------------------------------------------------------
int main(void)
{
    if (PREDICT_FASTER == FASTER_UNANSWERED || PREDICT_RATIO <= 0.0) {
        printf("Set TODO 4 first.\n");
        return 0;
    }

    const int N = NBLK * TPB;

    unsigned* h_seed = (unsigned*)malloc((size_t)N * sizeof(unsigned));
    unsigned* h_out  = (unsigned*)malloc((size_t)N * sizeof(unsigned));
    unsigned* h_ref  = (unsigned*)malloc((size_t)N * sizeof(unsigned));
    unsigned* acc    = (unsigned*)malloc((size_t)TPB * sizeof(unsigned));
    unsigned* pay    = (unsigned*)malloc((size_t)TPB * sizeof(unsigned));
    for (int i = 0; i < N; ++i) h_seed[i] = (unsigned)(i * 2654435761u) ^ 0x9e3779b9u;

    // CPU reference
    for (int b = 0; b < NBLK; ++b) {
        for (int t = 0; t < TPB; ++t) acc[t] = h_seed[b * TPB + t];
        for (int r = 0; r < ROUNDS; ++r) {
            for (int t = 0; t < TPB; ++t) {
                const unsigned v = burn(acc[t], work_for(t, r));
                unsigned s = 0;
                for (int k = 0; k < PAY; ++k) s += v + (unsigned)k;
                pay[t] = s;
            }
            for (int t = 0; t < TPB; ++t) acc[t] = pay[(t + PEERD) & (TPB - 1)];
        }
        for (int t = 0; t < TPB; ++t) h_ref[b * TPB + t] = acc[t];
    }

    unsigned *d_seed, *d_out;
    CHECK(cudaMalloc(&d_seed, (size_t)N * sizeof(unsigned)));
    CHECK(cudaMalloc(&d_out,  (size_t)N * sizeof(unsigned)));
    CHECK(cudaMemcpy(d_seed, h_seed, (size_t)N * sizeof(unsigned),
                     cudaMemcpyHostToDevice));

    // ---- is TODO 3 there? --------------------------------------------
    CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(unsigned)));
    variant_barrier<<<NBLK, TPB>>>(d_seed, d_out);
    CHECK_KERNEL();
    CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(unsigned),
                     cudaMemcpyDeviceToHost));
    int barrier_impl = 0;
    for (int i = 0; i < N; ++i) if (h_out[i] != 0xDEADBEEFu) { barrier_impl = 1; break; }

    // ---- duration-based clock warm-up (spec 12) ----------------------
    {
        cudaEvent_t w0, w1;
        CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        float acc_ms = 0.f;
        while (acc_ms < 400.0f) {
            CHECK(cudaEventRecord(w0));
            for (int i = 0; i < 4; ++i) {
                variant_flag   <<<NBLK, TPB>>>(d_seed, d_out);
                variant_barrier<<<NBLK, TPB>>>(d_seed, d_out);
            }
            CHECK(cudaEventRecord(w1));
            CHECK(cudaEventSynchronize(w1));
            float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, w0, w1));
            acc_ms += ms;
        }
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    }

    // ---- timing: both configs back to back, min of NSWEEP ------------
    // (spec 12: no validation, no allocation and no printing between
    //  timed configurations; min-of-N, not mean.)
    const int NSWEEP = 4, NITER = 20;
    float best[2] = { 1e30f, 1e30f };
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    for (int sweep = 0; sweep < NSWEEP; ++sweep) {
        for (int cfg = 0; cfg < 2; ++cfg) {
            CHECK(cudaEventRecord(e0));
            for (int it = 0; it < NITER; ++it) {
                if (cfg == 0) variant_flag   <<<NBLK, TPB>>>(d_seed, d_out);
                else          variant_barrier<<<NBLK, TPB>>>(d_seed, d_out);
            }
            CHECK(cudaEventRecord(e1));
            CHECK(cudaEventSynchronize(e1));
            float ms = 0.f;
            CHECK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= (float)NITER;
            if (ms < best[cfg]) best[cfg] = ms;
        }
    }
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));

    // ---- validation pass (second pass, after all timing) -------------
    int bad_flag = 0, bad_barrier = -1;
    CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(unsigned)));
    variant_flag<<<NBLK, TPB>>>(d_seed, d_out);
    CHECK_KERNEL();
    CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) if (h_out[i] != h_ref[i]) ++bad_flag;

    if (barrier_impl) {
        bad_barrier = 0;
        CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(unsigned)));
        variant_barrier<<<NBLK, TPB>>>(d_seed, d_out);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) if (h_out[i] != h_ref[i]) ++bad_barrier;
    }

    // ---- report ------------------------------------------------------
    printf("=== correctness (exact integer comparison over %d elements) ===\n", N);
    printf("  variant_flag    wrong: %d -> %s\n", bad_flag, bad_flag ? "FAIL" : "PASS");
    if (barrier_impl)
        printf("  variant_barrier wrong: %d -> %s\n", bad_barrier, bad_barrier ? "FAIL" : "PASS");
    else
        printf("  variant_barrier : TODO 3 not implemented -- skipped\n");

    printf("\n=== timing (min of %d sweeps x %d iterations, %d blocks) ===\n",
           NSWEEP, NITER, NBLK);
    printf("  variant_flag    : %8.4f ms\n", best[0]);
    if (barrier_impl) printf("  variant_barrier : %8.4f ms\n", best[1]);

    int pred_ok = 0, ratio_ok = 0;
    if (barrier_impl) {
        const int    obs_faster = (best[0] <= best[1]) ? FASTER_FLAG : FASTER_BARRIER;
        const double slower = (best[0] > best[1]) ? best[0] : best[1];
        const double faster = (best[0] > best[1]) ? best[1] : best[0];
        const double ratio  = slower / faster;
        pred_ok  = (PREDICT_FASTER == obs_faster);
        ratio_ok = (PREDICT_RATIO >= 0.70 * ratio && PREDICT_RATIO <= 1.30 * ratio);
        printf("  measured ratio  : %.3f  (%s is faster)\n", ratio,
               obs_faster == FASTER_FLAG ? "variant_flag" : "variant_barrier");
        printf("  you predicted   : %s, ratio %.3f -> %s / %s\n",
               PREDICT_FASTER == FASTER_FLAG ? "variant_flag" : "variant_barrier",
               PREDICT_RATIO, pred_ok ? "correct" : "WRONG",
               ratio_ok ? "correct" : "WRONG");
    }

    const int pass = (bad_flag == 0) && barrier_impl && (bad_barrier == 0) &&
                     pred_ok && ratio_ok;
    printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");

    free(h_seed); free(h_out); free(h_ref); free(acc); free(pay);
    CHECK(cudaFree(d_seed)); CHECK(cudaFree(d_out));
    CHECK(cudaDeviceReset());
    return 0;
}
