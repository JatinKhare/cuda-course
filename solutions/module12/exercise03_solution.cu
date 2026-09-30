// ============================================================================
// Module 12 / exercise03_solution.cu -- make the answer the same every time
//                                        (SOLVED)
//
// N = 33,554,433 = 2^25 + 1. Nothing here is a multiple of the block size,
// the warp size, or the grid size, and that is deliberate.
//
// Part 1 -- A BUG.
//   `reduceTreeBroken` sums the array and comes back about 5.6% low.
//   It is not a race: `compute-sanitizer --tool racecheck` is clean.
//   It is not an out-of-bounds access: `--tool memcheck` is clean.
//   It is not floating-point noise: 5.6% is five orders of magnitude larger
//   than the worst rounding error a float tree of this depth can produce.
//   The same kernel, on the same buffer, with n = 2^25 instead of 2^25 + 1,
//   is EXACT -- the harness runs both and prints both. Diagnose it, then
//   repair it. Work out exactly how many elements it loses and which ones,
//   because "add a guard somewhere" is not a diagnosis.
//
// Part 2 -- DETERMINISM.
//   Module 10 measured `float atomicAdd` giving ten distinct bit patterns in
//   ten identical runs, and named "fixed-order reduction" as the deterministic
//   alternative without building one. This is where that debt is paid, and
//   there is a second half of it that most people never meet: a fixed-order
//   tree is reproducible only for a FIXED DECOMPOSITION. Change the grid and
//   you change the summation order, and the bits change with it. A library
//   that promises reproducibility has to promise it across grid sizes too,
//   because the grid it picks depends on the GPU it finds.
//
//   You will build two reproducible reductions with different trade-offs:
//   one that keeps float arithmetic and fixes the order, and one that gives
//   up float arithmetic and stops caring about order at all.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
// RUN  : .\exercise03_solution.exe
//        compute-sanitizer --tool memcheck  .\exercise03.exe
//        compute-sanitizer --tool racecheck .\exercise03.exe
//
// PASS : SCORE: 8/8.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(_e), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_KERNEL()                                                         \
    do { CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize()); } while (0)

static const unsigned  BS = 256;
static const long long N  = 33554433LL;        // 2^25 + 1
static const float HEAVY = 1.0e6f;             // the value of the LAST element

__device__ __forceinline__ float warpReduceSum(float v)
{
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

// Fixed-order block tree over a value already held in a register. Correct.
__device__ __forceinline__ float blockTree(float sum, float* sdata)
{
    unsigned tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid <  64) sdata[tid] += sdata[tid +  64];
    __syncthreads();
    float w = 0.0f;
    if (tid < 32) w = warpReduceSum(sdata[tid] + sdata[tid + 32]);
    return w;                                   // valid in thread 0
}

// Fixed-order final pass over m partials, in index order. Correct.
__global__ void finalTree(const float* __restrict__ partial, float* out, long long m)
{
    __shared__ float sdata[1024];
    unsigned tid = threadIdx.x;
    float sum = 0.0f;
    for (long long i = tid; i < m; i += blockDim.x) sum += partial[i];
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) { float w = warpReduceSum(sdata[tid]); if (tid == 0) *out = w; }
}

// ===========================================================================
// THE BROKEN KERNEL. Do not modify it -- write your repair as a new kernel so
// that the harness can print both. One defect, one line.
// ===========================================================================
__global__ void reduceTreeBroken(const float* __restrict__ in, float* partial,
                                 long long n)
{
    __shared__ float sdata[BS];
    long long i    = (long long)blockIdx.x * (BS * 2) + threadIdx.x;
    long long step = (long long)BS * 2 * gridDim.x;
    float sum = 0.0f;
    while (i + BS < n) {
        sum += in[i] + in[i + BS];
        i   += step;
    }
    float w = blockTree(sum, sdata);
    if (threadIdx.x == 0) partial[blockIdx.x] = w;
}

// The non-deterministic one, for contrast. Correct arithmetic, unspecified
// order. Module 10 owns the atomic; Module 12 owns the consequence.
__global__ void reduceAtomic(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    long long i    = (long long)blockIdx.x * (BS * 2) + threadIdx.x;
    long long step = (long long)BS * 2 * gridDim.x;
    float sum = 0.0f;
    while (i + BS < n) { sum += in[i] + in[i + BS]; i += step; }
    while (i < n)      { sum += in[i];              i += step; }
    float w = blockTree(sum, sdata);
    if (threadIdx.x == 0) atomicAdd(out, w);
}

// ===========================================================================
// TODO 1 -- diagnose, then repair.
//
//   (1a) Set DIAG_CODE to the number of the true diagnosis:
//        1  a shared-memory race: a __syncthreads() is missing from blockTree
//        2  the grid is smaller than the data needs, so some blocks never run
//        3  the loop condition silently discards elements that have no partner
//        4  float addition is not associative, so the tree loses precision
//        5  shared-memory bank conflicts corrupt the tree
//        6  the warp tail passes the wrong mask to __shfl_down_sync
//
//        Two of the six are true statements about the CUDA programming model
//        that are nevertheless not the cause here. Being able to say why is
//        the whole point.
//
//   (1b) Write reduceTreeFixed. It must be correct for every n >= 1, for any
//        grid size, and for any block size that is a power of two -- not just
//        for the two values the harness happens to use.
// ===========================================================================
static const int DIAG_CODE = 3;

__global__ void reduceTreeFixed(const float* __restrict__ in, float* partial,
                                long long n)
{
    __shared__ float sdata[BS];
    long long i    = (long long)blockIdx.x * (BS * 2) + threadIdx.x;
    long long step = (long long)BS * 2 * gridDim.x;
    float sum = 0.0f;
    while (i + BS < n) {            // both halves of the pair exist
        sum += in[i] + in[i + BS];
        i   += step;
    }
    while (i < n) {                 // the ragged tail: elements with no partner
        sum += in[i];
        i   += step;
    }
    float w = blockTree(sum, sdata);
    if (threadIdx.x == 0) partial[blockIdx.x] = w;
}

// ===========================================================================
// TODO 2 -- DESIGN. A float reduction whose bits do not depend on the grid.
//
//   launchDeterministic() MUST launch its first kernel with exactly `gridHint`
//   blocks -- the harness checks the return value and will fail you for
//   ignoring it. It must nevertheless produce the identical 32 bits for
//   gridHint = 97, 240 and 1021, and on every run.
//
//   That is the real constraint a reproducible library operates under: the
//   launch shape is chosen by whoever is tuning for the machine, and the
//   answer is not allowed to notice. Separate the two things that the obvious
//   implementation has fused together -- how the data is partitioned, and how
//   many blocks are running -- and make only one of them depend on the grid.
//
//   Return gridHint when implemented, 0 when not.
// ===========================================================================
// The decomposition is a CONSTANT, and only the traversal of it depends on
// the grid. The array is cut into NCHUNK fixed chunks whose boundaries are a
// function of n alone; chunk c is always summed by one block in one fixed
// order and always lands in partial[c]; and the final pass walks partial[]
// in index order. Which block computed which chunk -- the only thing gridHint
// changes -- never enters the arithmetic.
#define NCHUNK 2048

__global__ void detPass1(const float* __restrict__ in, long long n, float* partial)
{
    __shared__ float sdata[BS];
    long long L = (n + NCHUNK - 1) / NCHUNK;        // chunk size: f(n) only
    for (int c = (int)blockIdx.x; c < NCHUNK; c += (int)gridDim.x) {
        long long a = (long long)c * L;
        long long b = a + L; if (b > n) b = n;
        float sum = 0.0f;
        for (long long i = a + threadIdx.x; i < b; i += BS) sum += in[i];
        float w = blockTree(sum, sdata);
        if (threadIdx.x == 0) partial[c] = w;
        __syncthreads();      // WAR: sdata is reused on the next chunk (M9)
    }
}

static int launchDeterministic(const float* d_in, long long n, int gridHint,
                               float* d_partial, float* d_out)
{
    detPass1<<<gridHint, BS>>>(d_in, n, d_partial);
    finalTree<<<1, 1024>>>(d_partial, d_out, NCHUNK);
    return gridHint;
}

// ===========================================================================
// TODO 3 -- DESIGN. The other way to be reproducible: stop using floats.
//
//   Integer addition IS associative and commutative, so a 64-bit integer
//   accumulator gives the same answer for every order, every grid, and every
//   run -- with a plain atomicAdd and no tree discipline at all. The price is
//   that you must choose a fixed-point scale.
//
//   Pick FP_SCALE so that:
//     - no accumulator overflows: every input here is in [0, 1000], and there
//       are 33,554,433 of them;
//     - the quantization error over the whole array stays below the 1e-5
//       relative tolerance the harness applies.
//   Those two requirements pull in opposite directions. Write down the bound
//   for each before you pick a number; there is a wide window and also a wrong
//   answer on each side, and the harness tells you which one you hit.
//
//   Then implement launchFixedPoint(). Same contract: launch with exactly
//   gridHint blocks, return gridHint, produce identical bits everywhere.
// ===========================================================================
// Overflow bound : sum <= 33,554,433 * 1000 ~= 3.36e10, and an unsigned
//                  64-bit accumulator holds 1.8e19, so FP_SCALE < 5.4e8.
// Precision bound: worst-case quantization is n * 0.5 / FP_SCALE, and the
//                  tolerance is 1e-5 * 1.78e7 = 178, so FP_SCALE > 9.4e4.
// Any value in [1e5, 5e8] works. 2^20 is a power of two, which makes the
// scaling exact in binary and keeps the reasoning simple.
static const double FP_SCALE = 1048576.0;   // 2^20

// Integer addition is associative AND commutative, so nothing about the
// order matters: a plain 64-bit atomicAdd per block is bit-reproducible for
// every grid and every run. What you gave up is the float arithmetic itself,
// and with it the dynamic range -- you now have to know the magnitude of your
// data in advance.
__global__ void fpPass(const float* __restrict__ in, long long n,
                       unsigned long long* acc, double scale)
{
    __shared__ unsigned long long sdata[BS];
    unsigned tid   = threadIdx.x;
    long long i    = (long long)blockIdx.x * BS + tid;
    long long step = (long long)BS * gridDim.x;
    unsigned long long sum = 0ull;
    for (; i < n; i += step)
        sum += (unsigned long long)llrint((double)in[i] * scale);

    sdata[tid] = sum;
    __syncthreads();
    for (unsigned k = BS / 2; k > 0; k >>= 1) {
        if (tid < k) sdata[tid] += sdata[tid + k];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(acc, sdata[0]);
}

__global__ void fpFinish(const unsigned long long* acc, float* out, double scale)
{
    *out = (float)((double)(*acc) / scale);
}

static int launchFixedPoint(const float* d_in, long long n, int gridHint,
                            unsigned long long* d_acc, float* d_out)
{
    CHECK(cudaMemsetAsync(d_acc, 0, sizeof(unsigned long long)));
    fpPass<<<gridHint, BS>>>(d_in, n, d_acc, FP_SCALE);
    fpFinish<<<1, 1>>>(d_acc, d_out, FP_SCALE);
    return gridHint;
}

// ===========================================================================
// TODO 4 -- three predictions, committed before running.
//
//   (a) PRED_ATOMIC_REPRO: will the atomicAdd version produce the SAME 32
//       bits on all ten of ten identical runs? 1 = yes, 0 = no. The count
//       of distinct patterns is printed either way.
//
//   (b) PRED_PLAIN_GRID_STABLE: the plain correct tree (reduceTreeFixed
//       followed by finalTree, grid = gridHint) is run at gridHint = 97, 240
//       and 1021. Does it give the same 32 bits all three times? 1/0.
//
//   (c) PRED_DET_COST: your deterministic kernel's time divided into the
//       plain tree's time, at gridHint = 240. Above 1.0 means the
//       deterministic one is faster. Scored within +-30%.
// ===========================================================================
static const int   PRED_ATOMIC_REPRO       = 0;
static const int   PRED_PLAIN_GRID_STABLE = 0;
static const float PRED_DET_COST          = 1.0f;

// ---------------------------------------------------------------------------
static unsigned bitsOf(float f) { unsigned u; memcpy(&u, &f, 4); return u; }
static int countDistinct(const unsigned* v, int n)
{
    int d = 0;
    for (int i = 0; i < n; ++i) {
        int seen = 0;
        for (int j = 0; j < i; ++j) if (v[j] == v[i]) { seen = 1; break; }
        if (!seen) ++d;
    }
    return d;
}

static float runPlain(const float* d_in, long long n, int grid,
                      float* d_partial, float* d_out, int useFixed)
{
    if (useFixed) reduceTreeFixed<<<grid, BS>>>(d_in, d_partial, n);
    else          reduceTreeBroken<<<grid, BS>>>(d_in, d_partial, n);
    finalTree<<<1, 1024>>>(d_partial, d_out, grid);
    CHECK_KERNEL();
    float r; CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost));
    return r;
}

int main(void)
{
    if (DIAG_CODE == 0 || PRED_ATOMIC_REPRO < 0 ||
        PRED_PLAIN_GRID_STABLE < 0 || PRED_DET_COST <= 0.0f) {
        printf("Set TODO 1a and TODO 4 first.\n");
        return 0;
    }

    printf("Module 12 exercise 03 -- determinism and a silent tail\n");
    printf("N = %lld (2^25 + 1), %.1f MiB; element N-1 holds %.0f, the rest [0,1)\n\n",
           N, (double)N * 4.0 / 1048576.0, (double)HEAVY);

    float* h = (float*)malloc((size_t)N * sizeof(float));
    if (!h) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (long long i = 0; i < N; ++i) {
        unsigned u = (unsigned)i * 2654435761u; u ^= u >> 15;
        h[i] = (float)(u & 0xFFFFu) * (1.0f / 65536.0f);
    }
    h[N - 1] = HEAVY;

    double ref = 0.0;
    for (long long i = 0; i < N; ++i) ref += (double)h[i];

    float *d_in = nullptr, *d_partial = nullptr, *d_out = nullptr;
    unsigned long long* d_acc = nullptr;
    CHECK(cudaMalloc(&d_in, (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_partial, (size_t)65536 * sizeof(float)));
    CHECK(cudaMalloc(&d_out, sizeof(float)));
    CHECK(cudaMalloc(&d_acc, sizeof(unsigned long long)));
    CHECK(cudaMemcpy(d_in, h, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Part 1: the bug --------------------------------------------------
    double ref3 = 0.0;
    for (long long i = 0; i < (1LL << 25); ++i) ref3 += (double)h[i];

    float bad  = runPlain(d_in, N, 240, d_partial, d_out, 0);
    float badP = runPlain(d_in, 1LL << 25, 240, d_partial, d_out, 0);
    float good = runPlain(d_in, N, 240, d_partial, d_out, 1);
    printf("=== part 1: the defect ===\n");
    printf("  double reference        %.4f\n", ref);
    printf("  reduceTreeBroken        %.4f   (short by %.4f%%)\n",
           (double)bad, 100.0 * (ref - (double)bad) / ref);
    printf("  same kernel at n = 2^25 %.4f   (reference %.4f, off by %.3e)\n",
           (double)badP, ref3, fabs((double)badP - ref3) / ref3);
    printf("  your reduceTreeFixed    %.4f   (off by %.3e relative)\n",
           (double)good, fabs((double)good - ref) / ref);
    int s2 = (fabs((double)good - ref) <= 1e-5 * ref);
    // and at a second grid size, and at a power-of-two n, to catch a repair
    // that happens to work only for the harness's first configuration
    float good2 = runPlain(d_in, N, 977, d_partial, d_out, 1);
    float good3 = runPlain(d_in, 1LL << 25, 977, d_partial, d_out, 1);
    s2 = s2 && (fabs((double)good2 - ref) <= 1e-5 * ref)
            && (fabs((double)good3 - ref3) <= 1e-5 * ref3);
    printf("  fixed @ grid 977        %.4f   %s\n", (double)good2,
           (fabs((double)good2 - ref) <= 1e-5 * ref) ? "ok" : "WRONG");
    printf("  fixed @ n = 2^25        %.4f   %s\n", (double)good3,
           (fabs((double)good3 - ref3) <= 1e-5 * ref3) ? "ok" : "WRONG");
    int s1 = (DIAG_CODE == 3);
    printf("  diagnosis: you said %d -- %s\n\n", DIAG_CODE, s1 ? "correct" : "wrong");

    // ---- Part 2: determinism ---------------------------------------------
    printf("=== part 2: reproducibility ===\n");
    unsigned atom[10];
    for (int t = 0; t < 10; ++t) {
        CHECK(cudaMemset(d_out, 0, 4));
        reduceAtomic<<<240, BS>>>(d_in, d_out, N);
        CHECK_KERNEL();
        float r; CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost));
        atom[t] = bitsOf(r);
    }
    int nAtom = countDistinct(atom, 10);
    printf("  atomicAdd finalization, 10 runs, same grid : %d distinct patterns\n",
           nAtom);

    int grids[3] = { 97, 240, 1021 };
    unsigned plainBits[3];
    for (int g = 0; g < 3; ++g)
        plainBits[g] = bitsOf(runPlain(d_in, N, grids[g], d_partial, d_out, 1));
    int plainStable = (plainBits[0] == plainBits[1] && plainBits[1] == plainBits[2]);
    printf("  your fixed tree at grids 97/240/1021        : 0x%08x 0x%08x 0x%08x %s\n",
           plainBits[0], plainBits[1], plainBits[2],
           plainStable ? "(stable)" : "(NOT stable)");

    int detGrid = launchDeterministic(d_in, N, 240, d_partial, d_out);
    int fpGrid  = launchFixedPoint(d_in, N, 240, d_acc, d_out);
    if (detGrid == 0 || fpGrid == 0) {
        printf("\nSet TODO 2 and TODO 3 first.\n");
        return 0;
    }

    unsigned detBits[15]; int detIdx = 0, detGridOk = 1;
    for (int g = 0; g < 3; ++g)
        for (int t = 0; t < 5; ++t) {
            int used = launchDeterministic(d_in, N, grids[g], d_partial, d_out);
            CHECK_KERNEL();
            if (used != grids[g]) detGridOk = 0;
            float r; CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost));
            detBits[detIdx++] = bitsOf(r);
        }
    int nDet = countDistinct(detBits, 15);
    float detVal; memcpy(&detVal, &detBits[0], 4);
    printf("  your launchDeterministic, 3 grids x 5 runs  : %d distinct patterns"
           " (0x%08x, %.4f)%s\n", nDet, detBits[0], (double)detVal,
           detGridOk ? "" : "  [IGNORED gridHint]");

    unsigned fpBits[15]; int fpIdx = 0, fpGridOk = 1;
    for (int g = 0; g < 3; ++g)
        for (int t = 0; t < 5; ++t) {
            int used = launchFixedPoint(d_in, N, grids[g], d_acc, d_out);
            CHECK_KERNEL();
            if (used != grids[g]) fpGridOk = 0;
            float r; CHECK(cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost));
            fpBits[fpIdx++] = bitsOf(r);
        }
    int nFp = countDistinct(fpBits, 15);
    float fpVal; memcpy(&fpVal, &fpBits[0], 4);
    printf("  your launchFixedPoint,   3 grids x 5 runs  : %d distinct patterns"
           " (0x%08x, %.4f)%s\n", nFp, fpBits[0], (double)fpVal,
           fpGridOk ? "" : "  [IGNORED gridHint]");
    double fpErr = fabs((double)fpVal - ref) / ref;
    printf("  fixed-point relative error                 : %.3e  (scale %.0f)\n",
           fpErr, FP_SCALE);

    int s3 = (fabs((double)detVal - ref) <= 1e-5 * ref);
    int s4 = (nDet == 1 && detGridOk);
    int s5 = (nFp == 1 && fpGridOk && fpErr <= 1e-5);

    // ---- timing: the two configurations back to back, rotated -------------
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    {
        cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        float acc = 0.0f; CHECK(cudaEventRecord(a));
        while (acc < 1200.0f) {
            for (int k = 0; k < 20; ++k)
                reduceTreeFixed<<<240, BS>>>(d_in, d_partial, N);
            CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
            CHECK(cudaEventElapsedTime(&acc, a, b));
        }
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); CHECK_KERNEL();
    }
    double bPlain = 1e30, bDet = 1e30;
    for (int sweep = 0; sweep < 4; ++sweep) {
        for (int q = 0; q < 2; ++q) {
            int c = (q + sweep) % 2;
            CHECK(cudaEventRecord(e0));
            for (int k = 0; k < 20; ++k) {
                if (c == 0) {
                    reduceTreeFixed<<<240, BS>>>(d_in, d_partial, N);
                    finalTree<<<1, 1024>>>(d_partial, d_out, 240);
                } else {
                    launchDeterministic(d_in, N, 240, d_partial, d_out);
                }
            }
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (c == 0 && ms / 20.0 < bPlain) bPlain = ms / 20.0;
            if (c == 1 && ms / 20.0 < bDet)   bDet   = ms / 20.0;
        }
    }
    CHECK_KERNEL();
    double costRatio = bPlain / bDet;
    printf("\n  plain fixed tree + finalTree : %.4f ms (%.1f GB/s)\n", bPlain,
           (double)N * 4.0 / (bPlain * 1e-3) / 1e9);
    printf("  your deterministic version   : %.4f ms (%.1f GB/s)  ratio %.2fx\n",
           bDet, (double)N * 4.0 / (bDet * 1e-3) / 1e9, costRatio);

    // ---- predictions -------------------------------------------------------
    int atomRepro = (nAtom == 1) ? 1 : 0;
    int s6 = (PRED_ATOMIC_REPRO == atomRepro);
    int s7 = (PRED_PLAIN_GRID_STABLE == (plainStable ? 1 : 0));
    int s8 = (fabs(PRED_DET_COST - costRatio) <= 0.30 * costRatio);
    printf("\npredictions:\n");
    printf("  (a) atomic reproducible?     you said %s, measured %s (%d patterns) %s\n",
           PRED_ATOMIC_REPRO ? "yes" : "no", atomRepro ? "yes" : "no", nAtom,
           s6 ? "MATCH" : "MISS");
    printf("  (b) plain tree grid-stable?   you said %s, measured %s          %s\n",
           PRED_PLAIN_GRID_STABLE ? "yes" : "no", plainStable ? "yes" : "no",
           s7 ? "MATCH" : "MISS");
    printf("  (c) deterministic cost ratio  you said %.2fx, measured %.2fx     %s\n",
           (double)PRED_DET_COST, costRatio, s8 ? "MATCH" : "MISS");

    int score = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;
    printf("\nSCORE: %d/8\n", score);

    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_partial));
    CHECK(cudaFree(d_out)); CHECK(cudaFree(d_acc));
    free(h);
    CHECK(cudaDeviceReset());

    printf("\nOVERALL: %s\n", (score == 8) ? "PASS" : "FAIL");
    return (score == 8) ? 0 : 1;
}
