// =====================================================================
// Module 9 / Exercise 1 SOLUTION : "Is this barrier necessary?"
//
// GOAL
//   Eight fragments. Each has a MARKED synchronization site -- either a
//   barrier that is present, or a gap where one might belong. For each
//   one you must commit to three judgements BEFORE running anything:
//
//     Q1  verdict      V_REQUIRED / V_UNNECESSARY / V_UNDEFINED
//     Q2  guarantee    which of the two guarantees is the load-bearing
//                      one:  G_EXEC / G_MEM / G_BOTH   (G_NA if the
//                      verdict is not V_REQUIRED)
//     Q3  misbehaves   on THIS GPU, with the fragment in its
//                      incorrect / undefined form, does the observed
//                      output differ from the reference, or does the
//                      program fail to terminate?   1 = yes, 0 = no
//
//   Q3 is the point of the exercise. Q1 is a statement about the CUDA
//   execution model. Q3 is a statement about one chip on one day. Do
//   not assume the two columns agree; if you fill Q3 in by copying Q1
//   you will not score 8/8.
//
//   The two guarantees, stated once:
//     G_EXEC  execution barrier -- no thread in the block proceeds past
//             the barrier until every non-exited thread of the block has
//             reached it.
//     G_MEM   memory fence at block scope -- every shared- and
//             global-memory write issued by a thread of the block before
//             the barrier is visible to every thread of the block after
//             it.
//     G_BOTH  the fragment is broken if EITHER is missing.
//
// THE EIGHT FRAGMENTS
//   The kernels below are the fragments. Read them, not this comment.
//   Two of them (D and E) are undefined by the rules of the language
//   and are NOT executed unless you pass --run-ub on the command line.
//   Read the warning above main() before you do.
//
// WHAT THE PROGRAM CHECKS
//   - your three answer vectors, item by item, against a stored digest
//     (you get right/wrong per item, not the answer),
//   - the numerics of every executed fragment against a CPU reference,
//   - the numerics of your TODO 4 kernel.
//   All three answer vectors must be 8/8 and TODO 4 must be numerically
//   correct for OVERALL: PASS.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
// RUN:    .\exercise01_solution.exe            (safe: D and E are skipped)
//         .\exercise01_solution.exe --run-ub   (executes D and E)
//
// !! WARNING about --run-ub !!
//   Fragment E contains a barrier inside a loop with a per-thread trip
//   count. On this GPU it does not return. The process will sit at 100%
//   GPU with no output. Kill it from another shell with
//      taskkill /F /IM exercise01_solution.exe        (Windows)
//      kill -9 <pid>                         (Linux)
//   The GPU recovers; no reboot is needed. Everything the exercise
//   scores can be answered WITHOUT --run-ub. Use it once, deliberately,
//   when you want to see the failure mode with your own eyes.
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

#define CHECK_KERNEL() do {                                                \
    CHECK(cudaGetLastError());                                             \
    CHECK(cudaDeviceSynchronize());                                        \
} while (0)

// ---- answer vocabulary ----------------------------------------------
#define V_UNANSWERED   0
#define V_REQUIRED     1
#define V_UNNECESSARY  2
#define V_UNDEFINED    3

#define G_NA           0
#define G_EXEC         1
#define G_MEM          2
#define G_BOTH         3

#define M_UNANSWERED  -1

static const int TPB   = 256;
static const int SHIFT = 96;
static const int NBLK  = 1024;
static const int NCASE = 8;

// =====================================================================
//  YOUR ANSWERS
// =====================================================================
//
// TODO 1: classify each fragment. One of V_REQUIRED / V_UNNECESSARY /
//         V_UNDEFINED per case. Index 0=A, 1=B, 2=C, 3=D, 4=E, 5=F,
//         6=G, 7=H, matching the kernel names below.
//         Leaving a V_UNANSWERED makes the program exit early.
static const int verdict[NCASE] = {
    /* A */ V_REQUIRED,    /* B */ V_REQUIRED,
    /* C */ V_UNNECESSARY, /* D */ V_UNDEFINED,
    /* E */ V_UNDEFINED,   /* F */ V_UNDEFINED,
    /* G */ V_REQUIRED,    /* H */ V_REQUIRED
};

// TODO 2: for every case you marked V_REQUIRED, say WHICH guarantee the
//         fragment actually depends on: G_EXEC, G_MEM or G_BOTH. Use
//         G_NA for every case you did not mark V_REQUIRED.
//         Think about what the code does AFTER the marked site. A
//         fragment that only needs other threads to have finished
//         READING does not need anything published.
static const int guarantee[NCASE] = {
    /* A */ G_BOTH, /* B */ G_EXEC, /* C */ G_NA,   /* D */ G_NA,
    /* E */ G_NA,   /* F */ G_NA,   /* G */ G_BOTH, /* H */ G_BOTH
};

// TODO 3: predict, per case, whether the broken/undefined form actually
//         misbehaves on THIS GPU (RTX 3500 Ada, sm_89, CUDA 13.2):
//         1 = the output differs from the reference, or the program
//             does not terminate,
//         0 = the output is indistinguishable from correct.
//         "Broken form" means: for V_REQUIRED cases, with the marked
//         barrier removed; for V_UNDEFINED cases, exactly as written;
//         for V_UNNECESSARY cases, with the marked barrier removed.
static const int misbehaves[NCASE] = {
    /* A */ 1, /* B */ 1, /* C */ 0, /* D */ 1,
    /* E */ 1, /* F */ 1, /* G */ 0, /* H */ 1
};

// =====================================================================
//  FRAGMENT A -- stage a tile, then read a slot owned by another warp
// =====================================================================
__global__ void caseA(const float* __restrict__ in, float* __restrict__ out,
                      int sync_on)
{
    __shared__ float s[TPB];
    const int t = threadIdx.x, b = blockIdx.x * TPB;

    s[t] = in[b + t];

    if (sync_on) __syncthreads();            // <=== MARKED SITE (A)

    out[b + t] = s[(t + SHIFT) & (TPB - 1)];
}

// =====================================================================
//  FRAGMENT B -- a four-tile loop. The barrier at the TOP of the body
//  is always present and is not the marked one. The marked site is the
//  one at the BOTTOM.
// =====================================================================
static const int NTILE = 4;

__global__ void caseB(const float* __restrict__ in, float* __restrict__ out,
                      int sync_on)
{
    __shared__ float s[TPB];
    const int t = threadIdx.x, b = blockIdx.x * TPB;

    float acc = 0.0f;
    for (int tile = 0; tile < NTILE; ++tile) {
        s[t] = in[b + t] + (float)tile;      // overwrite the tile
        __syncthreads();                     // (not the marked site)
        acc += s[(t + SHIFT) & (TPB - 1)];

        if (sync_on) __syncthreads();        // <=== MARKED SITE (B)
    }
    out[b + t] = acc;
}

// =====================================================================
//  FRAGMENT C -- every thread touches only its own slot
// =====================================================================
__global__ void caseC(const float* __restrict__ in, float* __restrict__ out,
                      int sync_on)
{
    __shared__ float s[TPB];
    const int t = threadIdx.x, b = blockIdx.x * TPB;

    s[t] = in[b + t] * 3.0f + 1.0f;

    if (sync_on) __syncthreads();            // <=== MARKED SITE (C)

    out[b + t] = s[t] * 0.5f;
}

// =====================================================================
//  FRAGMENT D -- 256-thread block, barrier reached by 64 threads
//  (no sync_on switch: this one is shipped exactly as it is)
// =====================================================================
__global__ void caseD(const float* __restrict__ in, float* __restrict__ out)
{
    __shared__ float s[TPB];
    const int t = threadIdx.x, b = blockIdx.x * TPB;

    s[t] = in[b + t];

    if (t < 64) { __syncthreads(); }         // <=== MARKED SITE (D)

    out[b + t] = s[(t + SHIFT) & (TPB - 1)];
}

// =====================================================================
//  FRAGMENT E -- barrier inside a loop whose trip count is per-thread
// =====================================================================
__global__ void caseE(const int* __restrict__ trips,
                      const float* __restrict__ in, float* __restrict__ out)
{
    __shared__ float s[TPB];
    const int t = threadIdx.x, b = blockIdx.x * TPB;

    s[t] = in[b + t];
    __syncthreads();

    const int n = trips[t];                  // 1..5, depends on the thread
    for (int i = 0; i < n; ++i) {
        const float v = s[(t + SHIFT) & (TPB - 1)];
        __syncthreads();                     // <=== MARKED SITE (E)
        s[t] = v + 1.0f;
        __syncthreads();
    }
    out[b + t] = s[t];
}

// =====================================================================
//  FRAGMENT F -- 32-thread block, barrier reached by 16 threads.
//  Same shape as D. Note the launch configuration.
// =====================================================================
__global__ void caseF(const float* __restrict__ in, float* __restrict__ out)
{
    __shared__ float s[32];
    const int t = threadIdx.x, b = blockIdx.x * 32;

    s[t] = in[b + t];

    if (t < 16) { __syncthreads(); }         // <=== MARKED SITE (F)

    out[b + t] = s[(t + 1) & 31];
}

// =====================================================================
//  FRAGMENT G -- a cross-lane chain confined to lanes 0..31 of the
//  block, i.e. to a single warp. The marked site is a GAP.
// =====================================================================
__global__ void caseG(const float* __restrict__ in, float* __restrict__ out,
                      int sync_on)
{
    __shared__ float s[TPB];
    const int t = threadIdx.x, b = blockIdx.x * TPB;

    s[t] = in[b + t];
    __syncthreads();

    if (t < 32) {
        s[t] = s[t] + s[(t + 1) & 31];

        if (sync_on) __syncwarp();           // <=== MARKED SITE (G)

        s[t] = s[t] + s[(t + 3) & 31];
    }
    __syncthreads();
    out[b + t] = s[t];
}

// =====================================================================
//  FRAGMENT H -- the marked barrier sits inside a conditional
// =====================================================================
__global__ void caseH(const float* __restrict__ in, float* __restrict__ out,
                      int sync_on)
{
    __shared__ float s[TPB];
    const int t = threadIdx.x, b = blockIdx.x * TPB;

    if (blockIdx.x & 1) {
        s[t] = in[b + t];

        if (sync_on) __syncthreads();        // <=== MARKED SITE (H)

        out[b + t] = s[(t + SHIFT) & (TPB - 1)];
    } else {
        out[b + t] = in[b + t];
    }
}

// =====================================================================
//  TODO 4 -- a design decision, not an expression.
//
//  Fragment B needs TWO barriers per loop iteration: one so that the
//  tile is readable, one so that the tile is not clobbered while it is
//  still being read. Two barriers per iteration is a real cost: every
//  warp of the block must stop twice per tile.
//
//  Rewrite the same computation so that it is correct with EXACTLY ONE
//  __syncthreads() (or one equivalent block-wide synchronization) per
//  loop iteration. `out[b+t]` must equal what caseB with sync_on=1
//  produces, for every element.
//
//  You are allowed to change the shared memory declaration, the
//  indexing and the loop body. You are not allowed to change what the
//  kernel computes, and you may not use more than 2 KB of shared memory
//  per block. Think about WHY the second barrier exists before you try
//  to delete it: a barrier you cannot justify removing is a barrier you
//  must keep.
//
//  Leave the body as shipped and the program will report
//  "TODO 4 not implemented" and skip it.
// =====================================================================
__global__ void caseB_one_barrier(const float* __restrict__ in,
                                  float* __restrict__ out)
{
    // Double-buffer the tile. Tile `tile` is staged in buffer tile&1,
    // which nobody is reading while tile-1 lives in the other buffer.
    // The write-after-read hazard that the second barrier existed to
    // prevent simply does not exist any more, so only the
    // read-after-write barrier remains.
    __shared__ float s[2][TPB];
    const int t = threadIdx.x, b = blockIdx.x * TPB;

    float acc = 0.0f;
    for (int tile = 0; tile < NTILE; ++tile) {
        const int cur = tile & 1;
        s[cur][t] = in[b + t] + (float)tile;
        __syncthreads();                       // RAW only
        acc += s[cur][(t + SHIFT) & (TPB - 1)];
    }
    out[b + t] = acc;
}

// =====================================================================
//  answer digest -- gives per-item right/wrong without printing answers
// =====================================================================
static unsigned ans_hash(int q, int i, int v)
{
    unsigned h = 2166136261u;
    unsigned char bytes[4] = { (unsigned char)q, (unsigned char)i,
                               (unsigned char)(v & 0xff), 0x5au };
    for (int k = 0; k < 4; ++k) { h ^= bytes[k]; h *= 16777619u; }
    return h;
}
static const unsigned KEY_V[NCASE] = {
    0xc76ba2bfu, 0xea4dacc8u, 0x573269e6u, 0x04208620u,
    0xcbd6e5c1u, 0x76ada586u, 0xed9f3e55u, 0x50801a1eu
};
static const unsigned KEY_G[NCASE] = {
    0xb3e75f4au, 0x8909c5d7u, 0x8fbc159fu, 0x60d57b84u,
    0x3a541a89u, 0x8b6cb6eeu, 0xda23f53cu, 0xaf37513fu
};
static const unsigned KEY_M[NCASE] = {
    0xfbc509ddu, 0x5ea5e5a6u, 0x57f528deu, 0x8cd98dd4u,
    0xe75fbdd9u, 0x5a40b2d2u, 0xd38f2c8au, 0x787441d0u
};

static const char* CASE_NAME[NCASE] = { "A", "B", "C", "D", "E", "F", "G", "H" };

// ---------------------------------------------------------------------
static int mismatch(const float* got, const float* ref, int n)
{
    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (fabsf(got[i] - ref[i]) > 1e-5f * fmaxf(1.0f, fabsf(ref[i]))) ++bad;
    return bad;
}

int main(int argc, char** argv)
{
    int run_ub = 0;
    for (int i = 1; i < argc; ++i) if (strcmp(argv[i], "--run-ub") == 0) run_ub = 1;

    // ---- answers present? -------------------------------------------
    for (int i = 0; i < NCASE; ++i) {
        if (verdict[i] == V_UNANSWERED) { printf("Set TODO 1 first.\n"); return 0; }
        if (misbehaves[i] == M_UNANSWERED) { printf("Set TODO 3 first.\n"); return 0; }
    }
    {
        int any_required = 0, any_guarantee = 0;
        for (int i = 0; i < NCASE; ++i) {
            if (verdict[i] == V_REQUIRED) any_required = 1;
            if (guarantee[i] != G_NA)     any_guarantee = 1;
        }
        if (any_required && !any_guarantee) { printf("Set TODO 2 first.\n"); return 0; }
    }

    const int N   = NBLK * TPB;
    const int NBF = N / 32;                  // fragment F uses 32-thread blocks

    float* h_in   = (float*)malloc((size_t)N * sizeof(float));
    float* h_out  = (float*)malloc((size_t)N * sizeof(float));
    float* h_ref  = (float*)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)((unsigned)(i * 1664525u + 1013904223u) % 977u) * 0.001f;

    int* h_trips = (int*)malloc(TPB * sizeof(int));
    for (int i = 0; i < TPB; ++i) h_trips[i] = 1 + (i % 5);

    float *d_in, *d_out; int* d_trips;
    CHECK(cudaMalloc(&d_in,  (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_trips, TPB * sizeof(int)));
    CHECK(cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_trips, h_trips, TPB * sizeof(int), cudaMemcpyHostToDevice));

    printf("=== Part 1: executed fragments (correct form vs broken form) ===\n");
    printf("%-6s %-46s %10s %10s\n", "case", "what it is", "as-written", "broken");

    int obs[NCASE];
    for (int i = 0; i < NCASE; ++i) obs[i] = -1;

    // --- A ---
    for (int b = 0; b < NBLK; ++b)
        for (int t = 0; t < TPB; ++t)
            h_ref[b * TPB + t] = h_in[b * TPB + ((t + SHIFT) & (TPB - 1))];
    {
        int badgood = 0, badbroken = 0;
        for (int v = 1; v >= 0; --v) {
            CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
            caseA<<<NBLK, TPB>>>(d_in, d_out, v);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
            (v ? badgood : badbroken) = mismatch(h_out, h_ref, N);
        }
        obs[0] = (badbroken != 0);
        printf("%-6s %-46s %10d %10d\n", "A", "tile staged, slot read across warps",
               badgood, badbroken);
    }

    // --- B ---
    for (int b = 0; b < NBLK; ++b)
        for (int t = 0; t < TPB; ++t) {
            float acc = 0.0f;
            for (int tile = 0; tile < NTILE; ++tile)
                acc += h_in[b * TPB + ((t + SHIFT) & (TPB - 1))] + (float)tile;
            h_ref[b * TPB + t] = acc;
        }
    {
        int badgood = 0, badbroken = 0;
        for (int v = 1; v >= 0; --v) {
            CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
            caseB<<<NBLK, TPB>>>(d_in, d_out, v);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
            (v ? badgood : badbroken) = mismatch(h_out, h_ref, N);
        }
        obs[1] = (badbroken != 0);
        printf("%-6s %-46s %10d %10d\n", "B", "4-tile loop, barrier at bottom of body",
               badgood, badbroken);
    }
    // keep B's reference for TODO 4 below
    float* h_refB = (float*)malloc((size_t)N * sizeof(float));
    memcpy(h_refB, h_ref, (size_t)N * sizeof(float));

    // --- C ---
    for (int i = 0; i < N; ++i) h_ref[i] = (h_in[i] * 3.0f + 1.0f) * 0.5f;
    {
        int badgood = 0, badbroken = 0;
        for (int v = 1; v >= 0; --v) {
            CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
            caseC<<<NBLK, TPB>>>(d_in, d_out, v);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
            (v ? badgood : badbroken) = mismatch(h_out, h_ref, N);
        }
        obs[2] = (badbroken != 0);
        printf("%-6s %-46s %10d %10d\n", "C", "each thread touches only its own slot",
               badgood, badbroken);
    }

    // --- D / E ---
    if (run_ub) {
        for (int b = 0; b < NBLK; ++b)
            for (int t = 0; t < TPB; ++t)
                h_ref[b * TPB + t] = h_in[b * TPB + ((t + SHIFT) & (TPB - 1))];
        CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
        caseD<<<NBLK, TPB>>>(d_in, d_out);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
        int badd = mismatch(h_out, h_ref, N);
        obs[3] = (badd != 0);
        printf("%-6s %-46s %10s %10d\n", "D", "256-thread block, 64 threads reach it",
               "-", badd);

        printf("%-6s %-46s %10s %10s\n", "E", "per-thread trip count around a barrier",
               "-", "...");
        fflush(stdout);
        CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
        caseE<<<NBLK, TPB>>>(d_trips, d_in, d_out);
        CHECK_KERNEL();                       // <- if you are reading this line
                                              //    in a stack trace, E hung.
        printf("       (E returned; see solution notes -- this is not the\n"
               "        expected outcome and you should say why)\n");
        obs[4] = 1;
    } else {
        printf("%-6s %-46s %10s %10s\n", "D", "256-thread block, 64 threads reach it",
               "-", "skipped");
        printf("%-6s %-46s %10s %10s\n", "E", "per-thread trip count around a barrier",
               "-", "skipped");
        printf("       (pass --run-ub to execute D and E. Read the header first.)\n");
    }

    // --- F ---
    {
        for (int b = 0; b < NBF; ++b)
            for (int t = 0; t < 32; ++t)
                h_ref[b * 32 + t] = h_in[b * 32 + ((t + 1) & 31)];
        CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
        caseF<<<NBF, 32>>>(d_in, d_out);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
        int badf = mismatch(h_out, h_ref, N);
        obs[5] = (badf != 0);
        printf("%-6s %-46s %10s %10d\n", "F", "32-thread block, 16 threads reach it",
               "-", badf);
    }

    // --- G ---
    {
        for (int b = 0; b < NBLK; ++b) {
            float tmp[32];
            for (int l = 0; l < 32; ++l) tmp[l] = h_in[b * TPB + l] + h_in[b * TPB + ((l + 1) & 31)];
            for (int l = 0; l < 32; ++l) h_ref[b * TPB + l] = tmp[l] + tmp[(l + 3) & 31];
            for (int t = 32; t < TPB; ++t) h_ref[b * TPB + t] = h_in[b * TPB + t];
        }
        int badgood = 0, badbroken = 0;
        for (int v = 1; v >= 0; --v) {
            CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
            caseG<<<NBLK, TPB>>>(d_in, d_out, v);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
            (v ? badgood : badbroken) = mismatch(h_out, h_ref, N);
        }
        obs[6] = (badbroken != 0);
        printf("%-6s %-46s %10d %10d\n", "G", "cross-lane chain inside one warp",
               badgood, badbroken);
    }

    // --- H ---
    {
        for (int b = 0; b < NBLK; ++b)
            for (int t = 0; t < TPB; ++t)
                h_ref[b * TPB + t] = (b & 1)
                    ? h_in[b * TPB + ((t + SHIFT) & (TPB - 1))]
                    : h_in[b * TPB + t];
        int badgood = 0, badbroken = 0;
        for (int v = 1; v >= 0; --v) {
            CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
            caseH<<<NBLK, TPB>>>(d_in, d_out, v);
            CHECK_KERNEL();
            CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
            (v ? badgood : badbroken) = mismatch(h_out, h_ref, N);
        }
        obs[7] = (badbroken != 0);
        printf("%-6s %-46s %10d %10d\n", "H", "barrier inside a conditional",
               badgood, badbroken);
    }

    // ---- TODO 4 ------------------------------------------------------
    printf("\n=== Part 2: TODO 4 -- one barrier per iteration ===\n");
    int todo4_ok = -1;
    CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float)));
    caseB_one_barrier<<<NBLK, TPB>>>(d_in, d_out);
    CHECK_KERNEL();
    CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    {
        int sentinel = 1;
        for (int i = 0; i < N && sentinel; ++i) if (h_out[i] != -1.0f) sentinel = 0;
        if (sentinel) {
            printf("  TODO 4 not implemented -- skipping.\n");
        } else {
            int bad4 = mismatch(h_out, h_refB, N);
            todo4_ok = (bad4 == 0);
            printf("  wrong elems: %d  -> %s\n", bad4, todo4_ok ? "PASS" : "FAIL");
            printf("  Now count the barriers you actually emitted:\n");
            printf("    cuobjdump -sass exercise01_solution.exe > sass.txt\n");
            printf("    findstr /C:\"caseB_one_barrier\" /C:\"BAR.SYNC\" sass.txt\n");
            printf("  A correct answer has NTILE (=%d) BAR.SYNC in that function,\n", NTILE);
            printf("  not 2*NTILE. If you see 2*NTILE you solved a different problem.\n");
        }
    }

    // ---- scoring -----------------------------------------------------
    printf("\n=== Part 3: your classification ===\n");
    printf("%-6s %-12s %-12s %-12s %-10s\n",
           "case", "Q1 verdict", "Q2 guarantee", "Q3 misbehaves", "observed");
    int s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < NCASE; ++i) {
        const int ok1 = (ans_hash(1, i, verdict[i])   == KEY_V[i]);
        const int ok2 = (ans_hash(2, i, guarantee[i]) == KEY_G[i]);
        const int ok3 = (ans_hash(3, i, misbehaves[i]) == KEY_M[i]);
        s1 += ok1; s2 += ok2; s3 += ok3;
        char ob[8];
        if (obs[i] < 0) snprintf(ob, sizeof(ob), "skipped");
        else            snprintf(ob, sizeof(ob), "%d", obs[i]);
        printf("%-6s %-12s %-12s %-12s %-10s\n", CASE_NAME[i],
               ok1 ? "correct" : "WRONG", ok2 ? "correct" : "WRONG",
               ok3 ? "correct" : "WRONG", ob);
    }
    printf("\n  Q1 verdicts    : %d/%d\n", s1, NCASE);
    printf("  Q2 guarantees  : %d/%d\n", s2, NCASE);
    printf("  Q3 misbehaves  : %d/%d\n", s3, NCASE);
    printf("  score          : %d/%d\n", s1 + s2 + s3, 3 * NCASE);

    const int pass = (s1 == NCASE) && (s2 == NCASE) && (s3 == NCASE) && (todo4_ok == 1);
    printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");

    free(h_in); free(h_out); free(h_ref); free(h_refB); free(h_trips);
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_trips));
    CHECK(cudaDeviceReset());
    return 0;
}
