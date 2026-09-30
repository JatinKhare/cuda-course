// =====================================================================
// Module 10 / Example 1 : "Three instructions, and the gaps between them"
//
// GOAL
//   Show, measurably, that `x[0] += 1` executed by many threads is not one
//   operation but three (load / add / store), that the interleaving destroys
//   almost every update, that a barrier does NOT fix it, and that an atomic
//   does. Then use the value an atomic RETURNS, which is what makes ticket
//   allocation and stream compaction possible.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN:   .\example01.exe
//
// SASS of the racy kernel (see the lesson):
//   nvcc -arch=sm_89 -O3 -cubin -o example01.cubin example01.cu
//   cuobjdump -sass example01.cubin
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ---------------------------------------------------------------------
// Part A -- the read-modify-write race.
//
// `*c += 1` compiles to LDG.E / IADD3 / STG.E. Nothing stops another warp
// from executing its own LDG between this warp's LDG and its STG. Both
// warps then read the same value and store the same value: one increment
// survives, the other is lost.
// ---------------------------------------------------------------------
__global__ void racy_increment(int* c)
{
    *c += 1;
}

// Part B -- the same race, fenced and barriered to the hilt.
//
// Module 9 gave us ordering (a barrier makes all prior accesses by this
// block visible to the block) and a device-scope fence (__threadfence makes
// prior accesses visible device-wide). Neither makes the three instructions
// indivisible. The result is identical to Part A.
__global__ void barriered_increment(int* c)
{
    __syncthreads();
    __threadfence();
    *c += 1;
    __threadfence();
    __syncthreads();
}

// Part C -- the atomic. One instruction, executed at the L2 slice that owns
// the address, with no window for anyone to interleave.
__global__ void atomic_increment(int* c)
{
    atomicAdd(c, 1);
}

// ---------------------------------------------------------------------
// Part D -- atomics return the OLD value.
//
// Stream compaction: each thread that passes a predicate needs a unique,
// dense output slot. `atomicAdd(counter, 1)` returns the value the counter
// had BEFORE this thread's increment, so every thread gets a distinct
// ticket and the tickets are dense. The ORDER of tickets is unspecified;
// the UNIQUENESS is guaranteed. Those are different claims.
// ---------------------------------------------------------------------
__global__ void compact_even(const int* in, int n, int* out, int* counter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = in[i];
    if ((v & 1) == 0) {
        int slot = atomicAdd(counter, 1);   // ticket
        out[slot] = v;
    }
}

// ---------------------------------------------------------------------
// Part E -- float atomicAdd is non-deterministic.
//
// Floating-point addition is not associative. atomicAdd serializes the
// updates in whatever order the hardware happens to produce, and that order
// changes run to run. The sum is correct to within rounding; it is not
// bit-reproducible. This is the single most common source of "my training
// run does not reproduce".
// ---------------------------------------------------------------------
__global__ void float_sum_atomic(const float* x, int n, float* acc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(acc, x[i]);
}

// ---------------------------------------------------------------------
// Part F -- atomicInc does NOT mean atomicAdd(p,1).
//
//   atomicInc(p, limit)  ->  old = *p;  *p = (old >= limit) ? 0 : old + 1;
//
// It is a wrapping increment with a caller-supplied bound, and the wrap
// happens at `limit`, not at `limit+1`. It is a ring-buffer head-pointer
// instruction, not a counter instruction.
// ---------------------------------------------------------------------
__global__ void inc_wrap(unsigned int* p, unsigned int limit, unsigned int steps)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        for (unsigned int s = 0; s < steps; ++s) atomicInc(p, limit);
}

int main(void)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (sm_%d%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    int* d_c = nullptr;
    CHECK(cudaMalloc(&d_c, sizeof(int)));

    // ================= Part A =================
    printf("=== A. `*c += 1` from N threads, no synchronization ===\n");
    printf("%8s %8s %10s %10s %10s %10s\n",
           "blocks", "threads", "expected", "observed", "lost", "% lost");
    const int cfgA[6][2] = { {1,32}, {1,256}, {1,1024}, {4,256}, {40,256}, {1024,256} };
    for (int i = 0; i < 6; ++i) {
        const int nb = cfgA[i][0], nt = cfgA[i][1];
        const int expect = nb * nt;
        CHECK(cudaMemset(d_c, 0, sizeof(int)));
        racy_increment<<<nb, nt>>>(d_c);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        int h = 0;
        CHECK(cudaMemcpy(&h, d_c, sizeof(int), cudaMemcpyDeviceToHost));
        printf("%8d %8d %10d %10d %10d %9.2f%%\n",
               nb, nt, expect, h, expect - h, 100.0 * (expect - h) / expect);
    }
    printf("\n  Read this row by row. A whole warp of 32 lanes issues ONE LDG,\n"
           "  ONE IADD3 and ONE STG; all 32 lanes read the same value and all\n"
           "  32 store the same value, so a warp contributes AT MOST 1. Warps\n"
           "  that overlap in time lose their contribution as well.\n\n");

    // ================= Part B =================
    printf("=== B. The same code with __syncthreads() and __threadfence() ===\n");
    for (int trial = 0; trial < 3; ++trial) {
        CHECK(cudaMemset(d_c, 0, sizeof(int)));
        barriered_increment<<<4, 256>>>(d_c);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        int h = 0;
        CHECK(cudaMemcpy(&h, d_c, sizeof(int), cudaMemcpyDeviceToHost));
        printf("  trial %d: expected 1024, observed %d\n", trial, h);
    }
    printf("\n  Module 9's machinery is ORDERING and VISIBILITY. This race is a\n"
           "  failure of INDIVISIBILITY. A barrier can tell you that everything\n"
           "  before it has happened; it cannot stop something from happening\n"
           "  in the middle of your load-add-store.\n\n");

    // ================= Part C =================
    printf("=== C. atomicAdd ===\n");
    for (int i = 0; i < 6; ++i) {
        const int nb = cfgA[i][0], nt = cfgA[i][1];
        CHECK(cudaMemset(d_c, 0, sizeof(int)));
        atomic_increment<<<nb, nt>>>(d_c);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        int h = 0;
        CHECK(cudaMemcpy(&h, d_c, sizeof(int), cudaMemcpyDeviceToHost));
        printf("  %5d x %-5d expected %8d  observed %8d  %s\n",
               nb, nt, nb * nt, h, (h == nb * nt) ? "exact" : "WRONG");
    }
    printf("\n");

    // ================= Part D =================
    printf("=== D. The return value: ticket allocation ===\n");
    {
        const int N = 100000;
        int* h_in = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; ++i)
            h_in[i] = (int)((((unsigned)i * 1103515245u + 12345u) >> 8) % 1000u);
        int expectedEven = 0;
        long long expectedSum = 0;
        for (int i = 0; i < N; ++i)
            if ((h_in[i] & 1) == 0) { ++expectedEven; expectedSum += h_in[i]; }

        int *d_in = nullptr, *d_out = nullptr, *d_cnt = nullptr;
        CHECK(cudaMalloc(&d_in,  N * sizeof(int)));
        CHECK(cudaMalloc(&d_out, N * sizeof(int)));
        CHECK(cudaMalloc(&d_cnt, sizeof(int)));
        CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_out, 0xFF, N * sizeof(int)));
        CHECK(cudaMemset(d_cnt, 0, sizeof(int)));

        compact_even<<<(N + 255) / 256, 256>>>(d_in, N, d_out, d_cnt);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        int count = 0;
        CHECK(cudaMemcpy(&count, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        int* h_out = (int*)malloc(N * sizeof(int));
        CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

        long long gotSum = 0;
        int nonEven = 0, holes = 0;
        for (int i = 0; i < count; ++i) {
            if (h_out[i] == -1) ++holes;
            else { gotSum += h_out[i]; if (h_out[i] & 1) ++nonEven; }
        }
        printf("  even elements: expected %d, counter says %d  -> %s\n",
               expectedEven, count, (count == expectedEven) ? "exact" : "WRONG");
        printf("  slots 0..count-1 all written (no holes): %s\n", holes ? "NO" : "yes");
        printf("  multiset of values matches (sum %lld vs %lld): %s\n",
               gotSum, expectedSum, (gotSum == expectedSum && nonEven == 0) ? "yes" : "NO");
        printf("  first 8 compacted values: ");
        for (int i = 0; i < 8; ++i) printf("%d ", h_out[i]);
        printf("\n  ...which are NOT in in[]'s order. Tickets are unique and dense,\n"
               "  never ordered. If you need order you need a prefix sum (Module 13).\n\n");

        free(h_in); free(h_out);
        CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_cnt));
    }

    // ================= Part E =================
    printf("=== E. float atomicAdd: correct, but not reproducible ===\n");
    {
        const int N = 1 << 20;
        float* h_x = (float*)malloc(N * sizeof(float));
        // Values spanning several binades, so the rounding actually depends
        // on the order. All-equal values would hide the effect.
        for (int i = 0; i < N; ++i)
            h_x[i] = (float)((((unsigned)i * 2654435761u) >> 8) % 1000u) * 1e-3f
                   + 1e-7f * (float)(i % 97);

        float* d_x = nullptr; float* d_acc = nullptr;
        CHECK(cudaMalloc(&d_x, N * sizeof(float)));
        CHECK(cudaMalloc(&d_acc, sizeof(float)));
        CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

        double ref = 0.0;
        for (int i = 0; i < N; ++i) ref += (double)h_x[i];

        unsigned int firstBits = 0;
        unsigned int seen[16]; int nseen = 0;
        printf("  %5s %18s %14s %12s\n", "run", "bit pattern", "value", "abs err");
        for (int run = 0; run < 10; ++run) {
            const float zero = 0.0f;
            CHECK(cudaMemcpy(d_acc, &zero, sizeof(float), cudaMemcpyHostToDevice));
            float_sum_atomic<<<(N + 255) / 256, 256>>>(d_x, N, d_acc);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            float got = 0.0f;
            CHECK(cudaMemcpy(&got, d_acc, sizeof(float), cudaMemcpyDeviceToHost));
            unsigned int bits; memcpy(&bits, &got, 4);
            if (run == 0) firstBits = bits;
            int isNew = 1;
            for (int k = 0; k < nseen; ++k) if (seen[k] == bits) isNew = 0;
            if (isNew && nseen < 16) seen[nseen++] = bits;
            printf("  %5d       0x%08X %14.6f %12.3e %s\n", run, bits, got,
                   fabs((double)got - ref), (bits == firstBits) ? "" : "<- differs from run 0");
        }
        printf("\n  distinct bit patterns across 10 identical runs: %d\n", nseen);
        printf("  double-precision host reference: %.6f\n", ref);
        printf("  Every one of these is a correct sum. None of them is THE sum.\n"
               "  Modules 41-42 (CUDA for AI) revisit this as the reason ML training\n"
               "  runs do not reproduce bit-for-bit across restarts.\n\n");

        free(h_x);
        CHECK(cudaFree(d_x)); CHECK(cudaFree(d_acc));
    }

    // ================= Part F =================
    printf("=== F. atomicInc's wrap semantics ===\n");
    {
        unsigned int* d_p = nullptr;
        CHECK(cudaMalloc(&d_p, sizeof(unsigned int)));
        printf("  atomicInc(p, limit): old = *p; *p = (old >= limit) ? 0 : old+1\n");
        printf("  %8s %8s %10s   %s\n", "limit", "steps", "final *p", "note");
        const unsigned int cases[6][2] = { {4,1}, {4,4}, {4,5}, {4,6}, {4,10}, {1,3} };
        for (int i = 0; i < 6; ++i) {
            const unsigned int zero = 0;
            CHECK(cudaMemcpy(d_p, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));
            inc_wrap<<<1, 32>>>(d_p, cases[i][0], cases[i][1]);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            unsigned int h = 0;
            CHECK(cudaMemcpy(&h, d_p, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            printf("  %8u %8u %10u   %s\n", cases[i][0], cases[i][1], h,
                   (h == cases[i][1] % (cases[i][0] + 1)) ? "cycle length = limit+1" : "");
        }
        printf("\n  The cycle length is limit+1, not limit: the counter visits\n"
               "  0,1,...,limit,0,... A ring buffer of capacity C therefore needs\n"
               "  atomicInc(head, C-1), and writing C there is an off-by-one that\n"
               "  costs you nothing visible until the buffer wraps.\n\n");
        CHECK(cudaFree(d_p));
    }

    CHECK(cudaFree(d_c));
    CHECK(cudaDeviceReset());
    printf("OVERALL: PASS (demonstration program; see the lesson for the analysis)\n");
    return 0;
}
