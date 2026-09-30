// =====================================================================
// Module 6 / Exercise 3 : "It works on 32 threads"
//
// SYMPTOM
//   `fold_blend` symmetrises a signal block by block: within each
//   window of `blockDim.x` consecutive samples it replaces each sample
//   by the average of itself and its mirror image in that window. It
//   stages the window in shared memory first.
//
//   It passes with 32-thread blocks. It fails with 64, 128, 256 and
//   1024-thread blocks. The number of wrong elements changes from run
//   to run, and on a quiet machine it sometimes passes by accident.
//   The wrong values are not garbage: they are plausible signal
//   values, just the wrong ones.
//
//   There is a second, separate failure. Even after the single-window
//   case is correct at every block size, the multi-window case
//   (`--tiles 8`, where one block sweeps eight consecutive windows)
//   still produces occasional wrong elements, and the errors cluster
//   in the second and later windows of each block.
//
//   Nothing is ever out of bounds. `compute-sanitizer --tool memcheck`
//   reports no errors.
//
// YOUR JOB
//   1. Diagnose both failures. Record the first diagnosis in TODO 1.
//   2. Fix them (TODO 2, TODO 3).
//   3. Confirm with `compute-sanitizer --tool racecheck` that the tool
//      sees what you saw.
//
// BUILD:  nvcc -arch=sm_89 -O3 -lineinfo -o exercise03.exe exercise03.cu
// RUN:    .\exercise03.exe
//         .\exercise03.exe --tiles 8
//         compute-sanitizer --tool racecheck .\exercise03.exe --small
//         compute-sanitizer --tool memcheck  .\exercise03.exe --small
//
// Note: compute-sanitizer is slow. `--small` shrinks the problem to
// 16384 elements and one repeat per block size so the tools finish in
// seconds. The bug is still there.
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
// TODO 1: What is wrong with the single-window case? Set DIAGNOSIS to
//         exactly one of the codes below. The harness scores it, and
//         OVERALL cannot be PASS while it is DIAG_UNSET or wrong.
//
//   DIAG_OOB          a thread reads or writes outside the shared array
//   DIAG_NO_ORDER     a thread reads a shared location before the
//                     thread that writes it has written it
//   DIAG_SMEM_SIZE    the launch requests fewer bytes of shared memory
//                     than the kernel indexes
//   DIAG_BAD_MIRROR   the mirror index formula is wrong
//   DIAG_BANK         shared-memory bank conflicts
// ---------------------------------------------------------------------
enum { DIAG_UNSET = 0, DIAG_OOB = 1, DIAG_NO_ORDER = 2,
       DIAG_SMEM_SIZE = 3, DIAG_BAD_MIRROR = 4, DIAG_BANK = 5 };

static const int DIAGNOSIS = DIAG_UNSET;   // YOUR CODE HERE (TODO 1)

// =====================================================================
// The kernel under investigation.
//
// One block handles `tilesPerBlock` consecutive windows of blockDim.x
// samples. For each window it stages the window in shared memory and
// then writes
//
//     out[base + lane] = 0.5 * ( s[lane] + s[B - 1 - lane] )
//
// where B = blockDim.x and base is the first sample of the window.
// =====================================================================
__global__ void fold_blend(const float* __restrict__ in,
                           float* __restrict__ out,
                           int n, int tilesPerBlock)
{
    extern __shared__ float s[];

    const int B    = (int)blockDim.x;
    const int lane = (int)threadIdx.x;

    for (int t = 0; t < tilesPerBlock; ++t) {
        int base = ((int)blockIdx.x * tilesPerBlock + t) * B;
        int i    = base + lane;

        s[lane] = (i < n) ? in[i] : 0.0f;

        // -------------------------------------------------------------
        // TODO 2: (nothing is missing here -- unless it is)
        // -------------------------------------------------------------
        // YOUR CODE HERE (TODO 2)

        float v = 0.5f * (s[lane] + s[B - 1 - lane]);
        if (i < n) out[i] = v;

        // -------------------------------------------------------------
        // TODO 3: (nothing is missing here either -- unless it is)
        // -------------------------------------------------------------
        // YOUR CODE HERE (TODO 3)
    }
}

// ---------------------------------------------------------------------
static void cpuReference(const float* in, float* out, int n, int B)
{
    for (int base = 0; base < n; base += B)
        for (int lane = 0; lane < B; ++lane) {
            int i = base + lane;
            if (i >= n) break;
            int j = base + (B - 1 - lane);
            float a = in[i];
            float b = (j < n) ? in[j] : 0.0f;
            out[i] = 0.5f * (a + b);
        }
}

// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    int tilesPerBlock = 1;
    int small = 0;                 // --small: tiny problem, 1 repeat.
                                   // Use it under compute-sanitizer,
                                   // which is far too slow otherwise.
    for (int a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "--tiles") && a + 1 < argc)
            tilesPerBlock = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--small"))
            small = 1;
    }
    if (tilesPerBlock < 1) tilesPerBlock = 1;

    CHECK(cudaSetDevice(0));
    printf("=== Module 6 / Exercise 3 : fold_blend, tilesPerBlock = %d ===\n",
           tilesPerBlock);

    const int n = small ? (1 << 14) : (1 << 20);
    const size_t bytes = (size_t)n * sizeof(float);

    float* h_in  = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_got = (float*)malloc(bytes);
    if (!h_in || !h_ref || !h_got) { printf("host alloc failed\n"); return 1; }

    // Deterministic and non-symmetric, so that a value read from the
    // wrong end of the window cannot coincidentally be right.
    for (int i = 0; i < n; ++i)
        h_in[i] = 0.5f * (float)(i % 1021) - 0.125f * (float)(i % 37);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes)); CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const int blockSizes[] = { 32, 64, 128, 256, 1024 };
    const int NB = (int)(sizeof(blockSizes) / sizeof(blockSizes[0]));

    int fails = 0;
    // 8 repeats per configuration: this class of bug is not
    // deterministic and a single run proves nothing.
    const int REPEATS = small ? 1 : 8;

    printf("  %-8s %-10s %-12s %s\n", "block", "grid", "smem/blk", "result");
    for (int b = 0; b < NB; ++b) {
        int B = blockSizes[b];
        int perBlock = B * tilesPerBlock;
        int grid = (n + perBlock - 1) / perBlock;
        size_t smem = (size_t)B * sizeof(float);

        cpuReference(h_in, h_ref, n, B);

        long long worst = 0;
        int badRuns = 0;
        for (int rep = 0; rep < REPEATS; ++rep) {
            CHECK(cudaMemset(d_out, 0, bytes));
            fold_blend<<<grid, B, smem>>>(d_in, d_out, n, tilesPerBlock);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_got, d_out, bytes, cudaMemcpyDeviceToHost));
            long long bad = 0;
            for (int i = 0; i < n; ++i)
                if (!(fabsf(h_got[i] - h_ref[i]) <= 1e-5f * fmaxf(1.0f, fabsf(h_ref[i]))))
                    ++bad;
            if (bad) { ++badRuns; if (bad > worst) worst = bad; }
        }

        if (badRuns == 0)
            printf("  %-8d %-10d %-10zu B PASS  (%d/%d runs clean)\n",
                   B, grid, smem, REPEATS, REPEATS);
        else {
            printf("  %-8d %-10d %-10zu B FAIL  (%d/%d runs wrong, worst %lld/%d elements)\n",
                   B, grid, smem, badRuns, REPEATS, worst, n);
            ++fails;
        }
    }

    static const char* DNAMES[6] = { "(unset)",
        "out-of-bounds shared access", "missing ordering between a shared write and a shared read",
        "shared allocation too small", "wrong mirror index", "bank conflicts" };
    printf("\n  your diagnosis (TODO 1): %s\n",
           (DIAGNOSIS >= 0 && DIAGNOSIS <= 5) ? DNAMES[DIAGNOSIS] : "(invalid)");
    // (deliberately not written as a plain comparison against the right
    //  answer -- this file must not contain it)
    int diagOK = ((DIAGNOSIS * 37 + 11) % 97 == 85);
    printf("  diagnosis: %s\n", diagOK ? "CORRECT" : "WRONG or unset");

    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    free(h_in); free(h_ref); free(h_got);

    printf("\nOVERALL: %s\n", (fails == 0 && diagOK) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (fails == 0 && diagOK) ? 0 : 1;
}
