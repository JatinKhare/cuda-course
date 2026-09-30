// =====================================================================
// Module 4 / Exercise 3 SOLUTION : "Pick the space"
//
// GOAL
//   A kernel needs a small read-only coefficient table: 64 floats, the
//   same table for every thread in the grid, never written by the
//   device. 256 bytes. It has to live somewhere.
//
//   You will implement two access patterns over that table:
//
//     UNIFORM      -- at every step, all 32 lanes of a warp read the
//                     SAME table entry.
//     LANEVARYING  -- at every step, each of the 32 lanes reads a
//                     DIFFERENT table entry.
//
//   and two candidate homes for the table:
//
//     SPACE_CONSTANT  -- a device-side read-only window, 64 KB total,
//                        written from the host before launch, served by
//                        a dedicated small per-SM cache.
//     SPACE_GLOBAL_RO -- an ordinary cudaMalloc'd buffer, read through
//                        a pointer marked const __restrict__ so the
//                        compiler may route it down the read-only path.
//
//   The table fits comfortably in either. The question is which home is
//   right, and whether the answer is the same for both access patterns.
//
//   COMMIT FIRST. Fill in TODO 3 and TODO 4 with your prediction before
//   you build. The harness measures all four combinations and grades
//   your prediction.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN:    .\exercise03.exe
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

#define NC     64          // table entries  (power of two: index masking)
#define REP    64          // passes over the table, to make the reads dominate
#define N      (1 << 20)   // threads
#define VN     8192        // how many outputs the CPU reference checks
                           // (covers every lane residue mod NC many times)
#define ITERS  30

#define SPACE_CONSTANT   0
#define SPACE_GLOBAL_RO  1

// ---------------------------------------------------------------------
// TODO 1 -- the constant-space table.
//
// __constant__ puts the symbol in the 64 KB constant window. Reads are
// served by the per-SM constant cache, whose defining property is that
// a warp in which all 32 lanes request the same address is satisfied by
// ONE broadcast. It has no path for 32 different addresses in one go.
// ---------------------------------------------------------------------
__constant__ float c_tab[NC];

// ---------------------------------------------------------------------
// TODO 2 -- the accessor, resolved at compile time.
//
// The `if` is on a template parameter, so there is no branch in the
// generated code: each instantiation contains exactly one of the two
// loads. SPACE_CONSTANT becomes an LDC (or, when the index is uniform
// and known, a cmem[] operand folded directly into the FMA).
// SPACE_GLOBAL_RO becomes an LDG.
// ---------------------------------------------------------------------
template <int SPACE>
__device__ __forceinline__ float coef(int i, const float* __restrict__ g)
{
    if (SPACE == SPACE_CONSTANT) return c_tab[i];
    else                         return g[i];
}

// ---------------------------------------------------------------------
// UNIFORM: the index `j` is the loop counter. Every lane of the warp
// evaluates the same j at the same time, so all 32 lanes address the
// same table element.
// ---------------------------------------------------------------------
template <int SPACE>
__global__ void k_uniform(const float* __restrict__ x, float* __restrict__ y,
                          int n, const float* __restrict__ g)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float v = x[t], a = 0.0f;
    for (int r = 0; r < REP; ++r) {
        #pragma unroll 8
        for (int j = 0; j < NC; ++j)
            a = fmaf(coef<SPACE>(j, g), v, a);
    }
    y[t] = a;
}

// ---------------------------------------------------------------------
// LANEVARYING: the index is offset by the thread id, so lane L of the
// warp addresses element (base + L + j) mod 64. At every step the 32
// lanes request 32 distinct elements.
// ---------------------------------------------------------------------
template <int SPACE>
__global__ void k_lanevarying(const float* __restrict__ x, float* __restrict__ y,
                              int n, const float* __restrict__ g)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float v = x[t], a = 0.0f;
    int base = t;
    for (int r = 0; r < REP; ++r) {
        #pragma unroll 8
        for (int j = 0; j < NC; ++j)
            a = fmaf(coef<SPACE>((base + j) & (NC - 1), g), v, a);
    }
    y[t] = a;
}

// =====================================================================
static void cpu_uniform(const float* x, const float* tab, float* y, int n)
{
    for (int t = 0; t < n; ++t) {
        float v = x[t], a = 0.0f;
        for (int r = 0; r < REP; ++r)
            for (int j = 0; j < NC; ++j) a = fmaf(tab[j], v, a);
        y[t] = a;
    }
}
static void cpu_lanevarying(const float* x, const float* tab, float* y, int n)
{
    for (int t = 0; t < n; ++t) {
        float v = x[t], a = 0.0f;
        for (int r = 0; r < REP; ++r)
            for (int j = 0; j < NC; ++j) a = fmaf(tab[(t + j) & (NC - 1)], v, a);
        y[t] = a;
    }
}
static int compare(const float* got, const float* ref, int n)
{
    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (fabsf(got[i] - ref[i]) > 1e-4f * fmaxf(1.0f, fabsf(ref[i]))) ++bad;
    return bad;
}

int main(void)
{
    CHECK(cudaSetDevice(0));
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    int constBytes = 0;
    CHECK(cudaDeviceGetAttribute(&constBytes, cudaDevAttrTotalConstantMemory, 0));
    printf("=== %s : constant memory window = %d B ===\n", p.name, constBytes);

    // -----------------------------------------------------------------
    // TODO 3: Which home do you choose for each access pattern?
    //         Set each to SPACE_CONSTANT or SPACE_GLOBAL_RO.
    //         -1 means "not answered".
    // -----------------------------------------------------------------
    int choiceUniform     = SPACE_CONSTANT;    // TODO 3
    int choiceLaneVarying = SPACE_GLOBAL_RO;   // TODO 3

    // -----------------------------------------------------------------
    // TODO 4: For each access pattern, predict the ratio
    //             time(the space you did NOT pick)
    //             --------------------------------
    //             time(the space you DID pick)
    //         A value of 1.0 means "no difference". Values below 1.0
    //         would mean you picked the slower one. Be quantitative:
    //         think about how many distinct addresses a warp presents
    //         per instruction in each case, and what the hardware must
    //         do when that number is greater than one.
    // -----------------------------------------------------------------
    double predRatioUniform     = 3.0;    // TODO 4 -- modest win
    double predRatioLaneVarying = 30.0;   // TODO 4 -- 32-way serialization

    if (choiceUniform < 0 || choiceLaneVarying < 0 ||
        predRatioUniform <= 0.0 || predRatioLaneVarying <= 0.0) {
        printf("\nAnswer TODO 3 and TODO 4 before running.\n");
        return 0;
    }

    // -----------------------------------------------------------------
    const size_t bytes = (size_t)N * sizeof(float);
    float* h_x   = (float*)malloc(bytes);
    float* h_y   = (float*)malloc(bytes);
    float* h_ru  = (float*)malloc(bytes);
    float* h_rl  = (float*)malloc(bytes);
    float  h_tab[NC];
    for (int i = 0; i < NC; ++i)  h_tab[i] = 1.0f / (1.0f + (float)i);
    for (int i = 0; i < N;  ++i)  h_x[i]   = 1.0f + 0.0001f * (float)(i % 1021);

    float *d_x = nullptr, *d_y = nullptr, *d_tab = nullptr;
    CHECK(cudaMalloc(&d_x, bytes));
    CHECK(cudaMalloc(&d_y, bytes));
    CHECK(cudaMalloc(&d_tab, NC * sizeof(float)));
    CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_tab, h_tab, NC * sizeof(float), cudaMemcpyHostToDevice));

    // TODO 1 (host side). cudaMemcpyToSymbol takes the symbol itself,
    // not its address: the runtime resolves the symbol to its device
    // location. You cannot take a host pointer to __constant__ storage.
    CHECK(cudaMemcpyToSymbol(c_tab, h_tab, NC * sizeof(float)));

    cpu_uniform(h_x, h_tab, h_ru, VN);
    cpu_lanevarying(h_x, h_tab, h_rl, VN);

    const int threads = 256;
    const int blocks  = N / threads;
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));

    double ms[2][2];   // [pattern][space], pattern 0 = uniform
    int    bad[2][2];
    float  t;

#define RUN(PAT, SP, KERNEL, REFH)                                            \
    do {                                                                      \
        CHECK(cudaMemset(d_y, 0, bytes));                                     \
        KERNEL<SP><<<blocks, threads>>>(d_x, d_y, N, d_tab);                  \
        CHECK(cudaGetLastError());                                            \
        CHECK(cudaDeviceSynchronize());                                       \
        CHECK(cudaEventRecord(e0));                                           \
        for (int i = 0; i < ITERS; ++i)                                       \
            KERNEL<SP><<<blocks, threads>>>(d_x, d_y, N, d_tab);              \
        CHECK(cudaEventRecord(e1));                                           \
        CHECK(cudaEventSynchronize(e1));                                      \
        CHECK(cudaEventElapsedTime(&t, e0, e1));                              \
        ms[PAT][SP] = t / ITERS;                                              \
        CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));           \
        bad[PAT][SP] = compare(h_y, REFH, VN);                                 \
    } while (0)

    RUN(0, SPACE_CONSTANT,  k_uniform,     h_ru);
    RUN(0, SPACE_GLOBAL_RO, k_uniform,     h_ru);
    RUN(1, SPACE_CONSTANT,  k_lanevarying, h_rl);
    RUN(1, SPACE_GLOBAL_RO, k_lanevarying, h_rl);

    const char* spname[2] = { "constant", "global-ro" };
    printf("\n%-14s %-12s %10s %10s\n", "pattern", "space", "ms", "correct");
    for (int pat = 0; pat < 2; ++pat)
        for (int sp = 0; sp < 2; ++sp)
            printf("%-14s %-12s %10.4f %10s\n",
                   pat ? "lanevarying" : "uniform", spname[sp],
                   ms[pat][sp], bad[pat][sp] ? "NO" : "yes");

    int pass = 1;
    for (int pat = 0; pat < 2; ++pat)
        for (int sp = 0; sp < 2; ++sp)
            if (bad[pat][sp]) {
                printf("  [FAIL] %s/%s produced the wrong result "
                       "(TODO 1 or TODO 2 incomplete)\n",
                       pat ? "lanevarying" : "uniform", spname[sp]);
                pass = 0;
            }
    if (!pass) { printf("\nFAIL\n"); return 1; }

    int bestU = (ms[0][SPACE_CONSTANT] < ms[0][SPACE_GLOBAL_RO]) ? SPACE_CONSTANT : SPACE_GLOBAL_RO;
    int bestL = (ms[1][SPACE_CONSTANT] < ms[1][SPACE_GLOBAL_RO]) ? SPACE_CONSTANT : SPACE_GLOBAL_RO;
    double realU = ms[0][1 - choiceUniform]     / ms[0][choiceUniform];
    double realL = ms[1][1 - choiceLaneVarying] / ms[1][choiceLaneVarying];

    printf("\n--- your answers ---\n");
    printf("  uniform     : chose %-9s (best is %-9s) ratio predicted %.1f, measured %.1f\n",
           spname[choiceUniform], spname[bestU], predRatioUniform, realU);
    printf("  lanevarying : chose %-9s (best is %-9s) ratio predicted %.1f, measured %.1f\n",
           spname[choiceLaneVarying], spname[bestL], predRatioLaneVarying, realL);

    if (choiceUniform != bestU)     { printf("  [FAIL] wrong space for the uniform pattern\n");     pass = 0; }
    if (choiceLaneVarying != bestL) { printf("  [FAIL] wrong space for the lanevarying pattern\n"); pass = 0; }
    if (realU / predRatioUniform > 3.0 || predRatioUniform / realU > 3.0) {
        printf("  [FAIL] uniform ratio prediction off by more than 3x\n"); pass = 0;
    }
    if (realL / predRatioLaneVarying > 3.0 || predRatioLaneVarying / realL > 3.0) {
        printf("  [FAIL] lanevarying ratio prediction off by more than 3x\n"); pass = 0;
    }

    printf("\n%s\n", pass ? "PASS" : "FAIL");

    free(h_x); free(h_y); free(h_ru); free(h_rl);
    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_tab));
    CHECK(cudaDeviceReset());
    return pass ? 0 : 1;
}
