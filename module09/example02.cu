// =====================================================================
// Module 9 / Example 2 : Fences, scopes, groups, and the block wall
//
// GOAL
//   A barrier and a fence are different tools. A barrier makes threads
//   WAIT for each other. A fence makes one thread's OWN memory
//   operations become observable in program order to a chosen scope --
//   and makes nobody wait for anything.
//
//   Part A : intra-block producer/consumer. The consumer spins on a
//            flag, so the WAITING is already solved by the spin. What is
//            missing is ORDERING, and __threadfence_block() is exactly
//            and only that. Shown with and without.
//   Part B : a fence is not a barrier -- the rotate from Example 1 with
//            __threadfence_block() in place of __syncthreads().
//   Part C : the modern vocabulary: cuda::atomic_ref with explicit
//            cuda::memory_order and cuda::thread_scope_block, and
//            cooperative_groups::this_thread_block() / tiled_partition.
//   Part D : the block wall. Print how many blocks of this kernel can be
//            co-resident on this GPU, which is the number that decides
//            whether a cross-block spin-wait deadlocks. (Exercise 2.)
//
// BUILD:  nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler /Zc:preprocessor -o example02.exe example02.cu
//         (libcu++ <cuda/atomic> needs C++17 and, on MSVC, the conforming
//          preprocessor. On Linux: nvcc -arch=sm_89 -O3 -std=c++17 ... )
// RUN:    .\example02.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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
static const int NBLK  = 2048;
static const int SHIFT = 96;
static const int PAY   = 8;      // payload words per producer

// ---------------------------------------------------------------------
// PART A -- pairwise producer/consumer inside one block
// ---------------------------------------------------------------------
// Thread t is the producer for slot t and the consumer of slot
// peer = (t + 128) % 256. It writes PAY payload words, then raises a
// flag. Its peer spins until the flag is up and then reads the payload.
//
// The spin already supplies the waiting. What it does not supply is any
// reason for the payload stores to be visible BEFORE the flag store.
// Without a fence the hardware and the compiler are both free to let the
// flag store land first, and the consumer reads stale payload.
//
// Spinning inside a block is safe on sm_70+ : every thread of a block is
// resident on one SM simultaneously (Module 1), and independent thread
// scheduling guarantees forward progress for diverged threads (Module 8).
// The same spin ACROSS blocks is a deadlock -- see Part D.
//
// The flag itself is touched with cuda::atomic_ref only so that the
// read/modify is a well-defined non-racing access and the compiler
// cannot cache it in a register. Module 10 explains atomics; here the
// atomic is only a flag, and `memory_order_relaxed` deliberately asks
// for NO ordering, so that the ordering question stays visible.
using flag_ref = cuda::atomic_ref<int, cuda::thread_scope_block>;

template <bool USE_FENCE>
__global__ void pair_produce_consume(const float* __restrict__ in,
                                     float* __restrict__ scratch,
                                     float* __restrict__ out)
{
    __shared__ int flag[TPB];

    const int t    = threadIdx.x;
    const int base = blockIdx.x * TPB;
    float* payload = scratch + (size_t)base * PAY;   // this block's slice

    flag[t] = 0;
    __syncthreads();                       // one-time init; not the point

    // ---- produce -----------------------------------------------------
    for (int k = 0; k < PAY; ++k)
        payload[t * PAY + k] = in[base + t] + (float)k;

    if (USE_FENCE) __threadfence_block();  // <-- the whole example

    flag_ref(flag[t]).store(1, cuda::memory_order_relaxed);

    // ---- consume -----------------------------------------------------
    const int peer = (t + 128) & (TPB - 1);
    flag_ref fr(flag[peer]);
    while (fr.load(cuda::memory_order_relaxed) == 0) { /* spin */ }

    float acc = 0.0f;
    for (int k = 0; k < PAY; ++k) acc += payload[peer * PAY + k];
    out[base + t] = acc;
}

// ---------------------------------------------------------------------
// PART B -- a fence where a barrier was needed
// ---------------------------------------------------------------------
__global__ void rotate_fence_only(const float* __restrict__ in,
                                  float* __restrict__ out)
{
    __shared__ float s[TPB];
    const int t    = threadIdx.x;
    const int base = blockIdx.x * TPB;
    s[t] = in[base + t];
    __threadfence_block();                 // orders MY stores. Waits for
                                           // nobody. Wrong tool.
    out[base + t] = s[(t + SHIFT) & (TPB - 1)];
}

// ---------------------------------------------------------------------
// PART C -- cooperative groups, the explicit-group style
// ---------------------------------------------------------------------
// cg::this_thread_block() names the group a barrier acts on instead of
// leaving it implicit in the call. block.sync() compiles to the same
// bar.sync as __syncthreads(); the gain is that the group is a value you
// can pass to a function, so a device function can state in its
// signature which threads must call it. That is the main reason the
// explicit style is the modern recommendation. Module 29 goes deeper.
__global__ void rotate_cg(const float* __restrict__ in,
                          float* __restrict__ out)
{
    __shared__ float s[TPB];
    cg::thread_block block = cg::this_thread_block();

    const int t    = block.thread_rank();
    const int base = blockIdx.x * TPB;

    s[t] = in[base + t];
    block.sync();                          // == __syncthreads()
    float v = s[(t + SHIFT) & (TPB - 1)];

    // A statically-sized sub-group of 32. warp.sync() == __syncwarp()
    // with the mask implied by the partition.
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float w[TPB];
    w[t] = v;
    warp.sync();                           // warp-scope barrier only
    const int lane = warp.thread_rank();
    const int wbase = (t / 32) * 32;
    out[base + t] = w[wbase + ((lane + 1) & 31)];
}

// ---------------------------------------------------------------------
int main(void)
{
    const int N = NBLK * TPB;

    float* h_in  = (float*)malloc((size_t)N * sizeof(float));
    float* h_out = (float*)malloc((size_t)N * sizeof(float));
    float* h_ref = (float*)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)((unsigned)(i * 1664525u + 1013904223u) % 977u) * 0.001f;

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)N * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    float* d_scr;
    CHECK(cudaMalloc(&d_scr, (size_t)N * PAY * sizeof(float)));

    // ---- PART A ----
    printf("=== PART A: intra-block producer/consumer, spin on a flag ===\n");
    for (int b = 0; b < NBLK; ++b)
        for (int t = 0; t < TPB; ++t) {
            const int peer = (t + 128) & (TPB - 1);
            float acc = 0.0f;
            for (int k = 0; k < PAY; ++k) acc += h_in[b * TPB + peer] + (float)k;
            h_ref[b * TPB + t] = acc;
        }
    const char* an[2] = { "no fence  (payload may lag flag)",
                          "__threadfence_block() before flag" };
    for (int v = 0; v < 2; ++v) {
        CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
        CHECK(cudaMemset(d_scr, 0, (size_t)N * PAY * sizeof(float)));
        if (v == 0) pair_produce_consume<false><<<NBLK, TPB>>>(d_in, d_scr, d_out);
        else        pair_produce_consume<true><<<NBLK, TPB>>>(d_in, d_scr, d_out);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < N; ++i)
            if (fabsf(h_out[i] - h_ref[i]) > 1e-5f * fmaxf(1.0f, fabsf(h_ref[i]))) ++bad;
        printf("  %-36s wrong elems: %d\n", an[v], bad);
    }
    printf("  The spin supplied the WAITING. The fence supplied the ORDERING.\n"
           "  Neither substitutes for the other.\n");
    printf("  Expect BOTH lines to read 0 on this GPU. The difference is not in\n"
           "  the output, it is in the machine code. Run:\n"
           "    cuobjdump -sass example02.exe | findstr MEMBAR\n"
           "  Only the USE_FENCE=true instantiation contains MEMBAR.SC.CTA, and it\n"
           "  sits between the payload STG.E stores and the flag store. Without it\n"
           "  the stores are merely ISSUED in that order -- nothing makes them\n"
           "  COMPLETE in that order. Today the store pipe happens to retire them\n"
           "  in order. That is a property of this chip, not of your program.\n");

    // ---- PART B ----
    printf("\n=== PART B: fence used where a barrier was needed ===\n");
    for (int b = 0; b < NBLK; ++b)
        for (int t = 0; t < TPB; ++t)
            h_ref[b * TPB + t] = h_in[b * TPB + ((t + SHIFT) & (TPB - 1))];
    CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
    rotate_fence_only<<<NBLK, TPB>>>(d_in, d_out);
    CHECK_KERNEL();
    CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    int badb = 0;
    for (int i = 0; i < N; ++i)
        if (fabsf(h_out[i] - h_ref[i]) > 1e-5f) ++badb;
    printf("  __threadfence_block() instead of __syncthreads(): wrong elems: %d\n", badb);
    printf("  A fence orders one thread's own accesses. It never blocks.\n");

    // ---- PART C ----
    printf("\n=== PART C: cooperative groups ===\n");
    CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
    rotate_cg<<<NBLK, TPB>>>(d_in, d_out);
    CHECK_KERNEL();
    CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    int badc = 0;
    for (int b = 0; b < NBLK; ++b)
        for (int t = 0; t < TPB; ++t) {
            const int lane  = t & 31;
            const int wbase = (t / 32) * 32;
            const int src   = wbase + ((lane + 1) & 31);
            const float ref = h_in[b * TPB + ((src + SHIFT) & (TPB - 1))];
            if (fabsf(h_out[b * TPB + t] - ref) > 1e-4f) ++badc;
        }
    printf("  block.sync() + tiled_partition<32>().sync(): wrong elems: %d\n", badc);

    // ---- PART D ----
    printf("\n=== PART D: the block wall (why cross-block spinning deadlocks) ===\n");
    int sms = 0, coop = 0;
    CHECK(cudaDeviceGetAttribute(&sms,  cudaDevAttrMultiProcessorCount, 0));
    CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch,   0));
    int perSM = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &perSM, (const void*)rotate_cg, TPB, 0));
    printf("  SMs                                  : %d\n", sms);
    printf("  max co-resident blocks per SM        : %d  (at %d threads/block)\n", perSM, TPB);
    printf("  => at most %d blocks exist AT ONCE\n", sms * perSM);
    printf("  Launch %d blocks and blocks %d.. are not merely late: they do not\n",
           NBLK, sms * perSM);
    printf("  exist yet, and cannot start until a resident block RETIRES. A\n");
    printf("  resident block that spins waiting for them never retires.\n");
    printf("  cudaDevAttrCooperativeLaunch on this device: %d\n", coop);
    printf("  Cooperative launch caps the grid at the co-residency number so a\n");
    printf("  grid-wide barrier is meaningful. Module 29 owns grid.sync().\n");

    free(h_in); free(h_out); free(h_ref);
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_scr));
    CHECK(cudaDeviceReset());
    return 0;
}
