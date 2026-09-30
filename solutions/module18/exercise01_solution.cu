// =============================================================================
// Module 18 / Exercise 1 — the two-level tile hierarchy, built by hand.
//
// GOAL : Write the register-tiled GEMM. A block owns a BM x BN tile of C and
//        walks K in steps of BK through shared memory (Module 17). Inside that
//        block, EACH THREAD owns a TM x TN sub-tile of C held entirely in
//        registers, so one value read out of shared memory feeds TM (or TN)
//        fused multiply-adds instead of one.
//
//        The arithmetic is trivial. The index arithmetic is not: there are now
//        three coordinate systems (grid -> block tile -> thread tile) and every
//        one of them has a boundary.
//
//        Problem shape: M = 1027, N = 2053, K = 769. None of the three is a
//        multiple of any tile dimension, and none is a power of two.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
// RUN  : exercise01_solution.exe
//
// Useful while you work:
//   nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise01.exe exercise01.cu
//   nvcc -arch=sm_89 -O3 -cubin -o exercise01.cubin exercise01.cu
//   cuobjdump -sass exercise01.cubin
//
// WHAT IS SCORED (10 points; OVERALL: PASS needs all ten)
//   3  your kernel produces the right matrix on the main shape, on BOTH a
//      strictly-positive and a zero-mean dataset, and on two awkward small
//      shapes where every single block is a partial tile
//   1  no element of C is left unwritten (C is prefilled with +infinity)
//   2  your kernel is at least 4.2x the throughput of the Module 17 baseline
//      that this file ships (a correct-but-slow kernel scores 0 here)
//   4  two committed predictions, scored against the measurement
//
// A NOTE ON THE PERFORMANCE GATE: several of these TODOs have more than one
// correct answer, and they do not all run at the same speed. The gate is there
// because the difference between the fastest and the slowest *correct* kernel
// you can write here is about 1.4x, and no correctness test will ever tell you
// which one you wrote.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                       \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        printf("CUDA error %s (%s) at %s:%d\n", cudaGetErrorName(_e),          \
               cudaGetErrorString(_e), __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define M_DIM 1027
#define N_DIM 2053
#define K_DIM  769

// ---- the tile hierarchy. Fixed for you in this exercise; Exercise 2 sweeps it.
#define BM 128        // rows of C per block
#define BN  64        // columns of C per block
#define BK   8        // depth of one shared-memory tile
#define TM   8        // rows of C per thread
#define TN   4        // columns of C per thread
#define NT  ((BM/TM)*(BN/TN))      // threads per block = 16 * 16 = 256
#define AP  (BM + 4)  // row pitch of the transposed A tile, in floats

// =============================================================================
// THE KERNEL YOU WRITE
//
// Shared memory is declared for you, and the declaration already encodes two
// decisions you would otherwise have to make:
//
//   As is the A tile stored TRANSPOSED — As[k][m], not As[m][k]. The thread
//   tile needs TM consecutive values of m at one k, and a transposed tile makes
//   those TM values adjacent in shared memory. (The lesson explains what that
//   buys; look at the SASS afterwards.)
//
//   Its row pitch is BM + 4 rather than BM. Four floats, not one. Work out for
//   yourself, with Module 7's method, what the transposed STORE does to the
//   banks at pitch BM, and why the repair has to be a multiple of 4.
// =============================================================================
__global__ __launch_bounds__(NT) void gemmRegTile(
        int M, int N, int K, float alpha, const float *A, const float *B,
        float beta, float *C)
{
    __shared__ float As[BK][AP];       // As[k][m]  — A tile, transposed
    __shared__ float Bs[BK][BN];       // Bs[k][n]  — B tile, as laid out

    const int tid = threadIdx.x;       // blocks are launched 1-D, NT threads

    // ---------------------------------------------------------------------
    // TODO 1: locate this thread and this block.
    //
    //   rowBase, colBase : the (row, column) of C at the top-left corner of
    //                      this BLOCK's output tile.
    //   tRow, tCol       : this THREAD's position in the (BM/TM) x (BN/TN)
    //                      grid of threads inside the block. Threads are
    //                      launched 1-D, so you derive both from `tid`.
    //
    // Decide which of tRow/tCol varies fastest across a warp, and keep that
    // decision in mind for TODO 4 — it decides whether the stores to C are
    // coalesced (Module 5).
    // ---------------------------------------------------------------------
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;
    const int tRow    = tid / (BN/TN);     // 0 .. BM/TM-1  (16 here)
    const int tCol    = tid % (BN/TN);     // 0 .. BN/TN-1  (16 here)
    // tCol varies fastest across a warp, so the TN stores a thread makes in
    // TODO 4 are adjacent to its neighbour's: the warp covers TN*32 = 128
    // consecutive columns of C in TN store instructions.

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int kt = 0; kt < K; kt += BK) {

        // -----------------------------------------------------------------
        // TODO 2: stage one BM x BK tile of A (transposed, into As) and one
        // BK x BN tile of B (into Bs), cooperatively across all NT threads.
        //
        //   - BM*BK = 1024 elements of A and BK*BN = 512 of B, for NT = 256
        //     threads. Every thread moves several elements.
        //   - Out-of-range elements must become 0.0f, not garbage. THREE
        //     boundaries bite here: rowBase+r >= M, colBase+n >= N, and
        //     kt+c >= K. The last one is the quiet one: K = 769 and BK = 8,
        //     so the final tile has one valid k and seven invalid ones, and
        //     A[(rowBase+r)*K + kt+c] for an invalid c is still a perfectly
        //     legal address — it is the next row of A. Nothing faults.
        //   - The mapping from thread id to (element of the tile) is yours to
        //     choose, and it is not free: it decides the address pattern of
        //     the GLOBAL read. Apply Module 5's sector count to your choice
        //     before you write it down.
        // -----------------------------------------------------------------
        // A tile: BM x BK = 1024 elements, 4 per thread.
        // Consecutive tid -> consecutive k, so 8 lanes cover one row's 8
        // consecutive floats (32 B, one sector) and a warp asks for 4 rows =
        // 4 sectors, all of them fully used. The transpose happens on the
        // SHARED side, where it is free of address-divergence cost.
        #pragma unroll
        for (int u = 0; u < (BM*BK)/NT; ++u) {
            const int idx = tid + u*NT;
            const int r = idx / BK, c = idx % BK;
            As[c][r] = ((rowBase + r) < M && (kt + c) < K)
                     ? A[(size_t)(rowBase + r) * K + kt + c] : 0.0f;
        }
        // B tile: BK x BN = 512 elements, 2 per thread.
        // Consecutive tid -> consecutive n, so a warp reads 32 consecutive
        // floats = 128 B = 4 sectors: perfectly coalesced, and the shared
        // store hits 32 distinct banks.
        #pragma unroll
        for (int u = 0; u < (BK*BN)/NT; ++u) {
            const int idx = tid + u*NT;
            const int k = idx / BN, n = idx % BN;
            Bs[k][n] = ((kt + k) < K && (colBase + n) < N)
                     ? B[(size_t)(kt + k) * N + colBase + n] : 0.0f;
        }

        __syncthreads();

        // -----------------------------------------------------------------
        // TODO 3: the inner product over this tile.
        //
        // For each of the BK values of k in the tile, this thread needs TM
        // values from As and TN values from Bs, and performs TM*TN fused
        // multiply-adds with them. Write it so that each shared-memory
        // location is read ONCE per k and then reused out of registers —
        // that reuse is the entire point of the exercise, and a version that
        // re-reads shared memory inside the TM x TN loop is still correct and
        // will miss the performance gate.
        //
        // Use fmaf(), not `+=  *`.
        // -----------------------------------------------------------------
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float rM[TM], rN[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rM[i] = As[kk][tRow*TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rN[j] = Bs[kk][tCol*TN + j];
            // TM + TN = 12 shared reads have bought TM * TN = 32 fused ops.
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(rM[i], rN[j], acc[i][j]);
        }

        __syncthreads();
    }

    // ---------------------------------------------------------------------
    // TODO 4: write this thread's TM x TN results to C.
    //
    // Honour the BLAS contract: when beta == 0, C is NOT read (Module 16).
    // The harness prefills C with +infinity, and 0.0f * inf is NaN.
    //
    // Guard every element: M = 1027 is not a multiple of BM = 128 and
    // N = 2053 is not a multiple of BN = 64, so the last block row and the
    // last block column are partial. A single guard for the whole thread tile
    // is not enough.
    // ---------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = rowBase + tRow*TM + i;
        if (r >= M) continue;                       // per-row guard
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = colBase + tCol*TN + j;    // per-column guard
            if (c < N) {
                if (beta == 0.0f) C[(size_t)r*N + c] = alpha * acc[i][j];
                else              C[(size_t)r*N + c] = alpha * acc[i][j]
                                                     + beta * C[(size_t)r*N + c];
            }
        }
    }
}

// =============================================================================
// The Module 17 baseline: block tiling only, one output element per thread.
// Complete, correct, and the thing you have to beat by 4.2x.
// =============================================================================
template<int T>
__global__ __launch_bounds__(T*T) void gemmTiled(
        int M, int N, int K, float alpha, const float *A, const float *B,
        float beta, float *C)
{
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty;
    const int col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int kt = 0; kt < K; kt += T) {
        As[ty][tx] = (row < M   && kt + tx < K) ? A[(size_t)row * K + kt + tx] : 0.0f;
        Bs[ty][tx] = (kt + ty < K && col < N)   ? B[(size_t)(kt + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) acc = fmaf(As[ty][k], Bs[k][tx], acc);
        __syncthreads();
    }
    if (row < M && col < N) {
        if (beta == 0.0f) C[(size_t)row*N + col] = alpha * acc;
        else              C[(size_t)row*N + col] = alpha*acc + beta*C[(size_t)row*N + col];
    }
}

// =============================================================================
// Module 16's validator, unchanged. Finiteness first, then a full-coverage
// Freivalds probe, then a sampled exact double reference; the tolerance scales
// with S = sum_k |a_ik||b_kj|, never with |C|.
// =============================================================================
typedef struct {
    int ok, nonfinite;
    double freivalds; int freivaldsRow;
    double sampled;   int sampledRow, sampledCol;
} GemmCheck;

static GemmCheck gemmValidate(int M, int N, int K,
                              float alpha, const float *hA, const float *hB,
                              float beta,  const float *hC0, const float *hC,
                              int sampleStride)
{
    GemmCheck r; r.ok = 1; r.nonfinite = 0;
    r.freivalds = 0.0; r.freivaldsRow = -1;
    r.sampled = 0.0; r.sampledRow = -1; r.sampledCol = -1;
    const double u      = ldexp(1.0, -24);
    const double gammaK = (double)K * u / (1.0 - (double)K * u);

    for (size_t i = 0; i < (size_t)M * N; ++i)
        if (!isfinite(hC[i])) ++r.nonfinite;
    if (r.nonfinite) r.ok = 0;

    double *v   = (double*)malloc(sizeof(double) * N);
    double *Bv  = (double*)malloc(sizeof(double) * K);
    double *aBv = (double*)malloc(sizeof(double) * K);
    unsigned s = 0xB16B00B5u;
    for (int j = 0; j < N; ++j) { s = s*1664525u+1013904223u;
                                  v[j] = 0.5 + (double)((s>>8)&0xFFFFu)/65536.0; }
    for (int k = 0; k < K; ++k) {
        double t = 0.0, tb = 0.0;
        for (int j = 0; j < N; ++j) {
            double b = hB[(size_t)k * N + j];
            t  += b * v[j]; tb += fabs(b) * v[j];
        }
        Bv[k] = t; aBv[k] = tb;
    }
    for (int i = 0; i < M; ++i) {
        double y = 0.0, ybound = 0.0;
        for (int k = 0; k < K; ++k) {
            double a = hA[(size_t)i * K + k];
            y += a * Bv[k]; ybound += fabs(a) * aBv[k];
        }
        double got = 0.0, want = (double)alpha * y;
        for (int j = 0; j < N; ++j) {
            got += (double)hC[(size_t)i * N + j] * v[j];
            if (beta != 0.0f) want += (double)beta * (double)hC0[(size_t)i*N+j] * v[j];
        }
        double c0v = 0.0;
        if (beta != 0.0f)
            for (int j = 0; j < N; ++j) c0v += fabs((double)hC0[(size_t)i*N+j]) * v[j];
        double tol = fabs((double)alpha) * (gammaK + 4.0*u) * ybound
                   + 4.0 * u * fabs((double)beta) * c0v;
        double ratio = (tol > 0.0) ? fabs(got - want) / tol : 0.0;
        if (ratio > r.freivalds) { r.freivalds = ratio; r.freivaldsRow = i; }
    }
    free(v); free(Bv); free(aBv);
    if (!(r.freivalds <= 1.0)) r.ok = 0;

    for (int i = 0; i < M; i += sampleStride)
        for (int j = 0; j < N; j += sampleStride) {
            double acc = 0.0, S = 0.0;
            for (int k = 0; k < K; ++k) {
                double a = hA[(size_t)i * K + k], b = hB[(size_t)k * N + j];
                acc += a * b; S += fabs(a) * fabs(b);
            }
            double want = (double)alpha * acc;
            if (beta != 0.0f) want += (double)beta * (double)hC0[(size_t)i*N+j];
            double tol = fabs((double)alpha) * (gammaK + 4.0*u) * S
                       + 4.0 * u * fabs((double)beta)
                              * (hC0 ? fabs((double)hC0[(size_t)i*N+j]) : 0.0);
            double d = (tol > 0.0) ? fabs((double)hC[(size_t)i*N+j] - want)/tol : 0.0;
            if (d > r.sampled) { r.sampled = d; r.sampledRow = i; r.sampledCol = j; }
        }
    if (!(r.sampled <= 1.0)) r.ok = 0;
    return r;
}

// =============================================================================
// TODO 5 — PREDICTIONS. Commit before you build.
//
// P1: With TM = 8 and TN = 4, how many fused multiply-adds does this kernel
//     perform per SCALAR shared-memory read in the inner loop?  Choose:
//       1 : below 1        2 : 1 to 2        3 : 2 to 3
//       4 : 3 to 5         5 : above 5
//
// P2: By what factor will your register-tiled kernel beat the Module 17
//     baseline that ships in this file?  Choose:
//       1 : under 1.5x     2 : 1.5x to 3x    3 : 3x to 4.5x
//       4 : 4.5x to 7x     5 : over 7x
// =============================================================================
#define PRED_P1 3   // TM*TN/(TM+TN) = 32/12 = 2.67 -> bucket 3
#define PRED_P2 4   // measured 4.8x - 5.8x on this GPU -> bucket 4

// =============================================================================
static int Mg = M_DIM, Ng = N_DIM, Kg = K_DIM;
static const float *dAg, *dBg;
static float *dCg;

static void runMine(void) {
    dim3 gr((Ng + BN - 1)/BN, (Mg + BM - 1)/BM);
    gemmRegTile<<<gr, NT>>>(Mg, Ng, Kg, 1.0f, dAg, dBg, 0.0f, dCg);
}
static void runBase(void) {
    dim3 bl(32,32), gr((Ng+31)/32, (Mg+31)/32);
    gemmTiled<32><<<gr, bl>>>(Mg, Ng, Kg, 1.0f, dAg, dBg, 0.0f, dCg);
}

__global__ void warmStream(const float4 * __restrict__ s, float *o, size_t n) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    float4 acc = make_float4(0,0,0,0);
    for (; i < n; i += gridDim.x*(size_t)blockDim.x) {
        float4 v = s[i]; acc.x+=v.x; acc.y+=v.y; acc.z+=v.z; acc.w+=v.w; }
    if (acc.x == 1e30f) o[0] = acc.x+acc.y+acc.z+acc.w;
}
__global__ void warmFfma(float *o, int iters) {
    float a[8], b = 1.0000001f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = (float)(threadIdx.x + i);
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = fmaf(a[i], b, 1.0f);
    }
    float s = 0; for (int i = 0; i < 8; ++i) s += a[i];
    if (s == 1e30f) o[0] = s;
}
static double timeOne(void (*f)(void), int iters) {
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a)); for (int i = 0; i < iters; ++i) f();
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms / iters;
}

// run the reader's kernel at an arbitrary shape and validate it
static int checkShape(int M, int N, int K, int zeroMean, const char *label)
{
    const size_t sA = (size_t)M*K, sB = (size_t)K*N, sC = (size_t)M*N;
    float *hA = (float*)malloc(sA*4), *hB = (float*)malloc(sB*4), *hC = (float*)malloc(sC*4);
    unsigned st = zeroMean ? 12345u : 1u;
    for (size_t i = 0; i < sA; ++i) { st = st*1664525u+1013904223u;
        float r = (float)((st>>8)&0xFFFFu)/65536.0f;
        hA[i] = zeroMean ? (2.0f*r - 1.0f) : (0.5f + r); }
    for (size_t i = 0; i < sB; ++i) { st = st*1664525u+1013904223u;
        float r = (float)((st>>8)&0xFFFFu)/65536.0f;
        hB[i] = zeroMean ? (2.0f*r - 1.0f) : (0.5f + r); }
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, sA*4)); CHECK(cudaMalloc(&dB, sB*4)); CHECK(cudaMalloc(&dC, sC*4));
    CHECK(cudaMemcpy(dA, hA, sA*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, sB*4, cudaMemcpyHostToDevice));
    float *p = (float*)malloc(sC*4);
    for (size_t j = 0; j < sC; ++j) p[j] = std::numeric_limits<float>::infinity();
    CHECK(cudaMemcpy(dC, p, sC*4, cudaMemcpyHostToDevice)); free(p);

    dim3 gr((N + BN - 1)/BN, (M + BM - 1)/BM);
    gemmRegTile<<<gr, NT>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hC, dC, sC*4, cudaMemcpyDeviceToHost));

    int stride = (M > 512) ? 64 : 1;
    GemmCheck c = gemmValidate(M, N, K, 1.0f, hA, hB, 0.0f, NULL, hC, stride);
    printf("  %-40s nonfinite %8d | Freivalds %9.3g | sampled %9.3g | %s\n",
           label, c.nonfinite, c.freivalds, c.sampled, c.ok ? "PASS" : "FAIL");
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return c.ok;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    if (PRED_P1 == 0 || PRED_P2 == 0) {
        printf("Set TODO 5 (PREDICTIONS) first.\n");
        return 0;
    }

    const int M = M_DIM, N = N_DIM, K = K_DIM;
    const size_t sA = (size_t)M*K, sB = (size_t)K*N, sC = (size_t)M*N;

    printf("=== Module 18 / Exercise 1 - the two-level tile hierarchy ===\n");
    printf("C(%d x %d) = A(%d x %d) * B(%d x %d)\n", M, N, M, K, K, N);
    printf("block tile %d x %d x %d, thread tile %d x %d, %d threads/block\n",
           BM, BN, BK, TM, TN, NT);
    printf("grid = %d x %d blocks; M/BM = %.2f and N/BN = %.2f, so the last block\n"
           "row and the last block column are both partial.\n\n",
           (N+BN-1)/BN, (M+BM-1)/BM, (double)M/BM, (double)N/BN);

    // ---- correctness, four shapes
    printf("-- correctness ----------------------------------------------------\n");
    int ok = 0, nck = 0;
    ok += checkShape(M, N, K, 0, "1027 x 2053 x 769, positive operands");   ++nck;
    ok += checkShape(M, N, K, 1, "1027 x 2053 x 769, zero-mean operands");  ++nck;
    ok += checkShape(37, 53, 11, 0, "37 x 53 x 11  (one partial block)");   ++nck;
    ok += checkShape(129, 65, 9, 1, "129 x 65 x 9  (tile+1 in every axis)");++nck;
    printf("  %d/%d shapes correct\n", ok, nck);

    if (ok == 0) {
        printf("\nNothing correct yet; skipping the timed section.\n");
        printf("\nOVERALL: FAIL\n");
        CHECK(cudaDeviceReset());
        return 1;
    }

    // ---- timing
    float *hA = (float*)malloc(sA*4), *hB = (float*)malloc(sB*4);
    unsigned st = 1u;
    for (size_t i = 0; i < sA; ++i) { st = st*1664525u+1013904223u;
        hA[i] = 0.5f + (float)((st>>8)&0xFFFFu)/65536.0f; }
    for (size_t i = 0; i < sB; ++i) { st = st*1664525u+1013904223u;
        hB[i] = 0.5f + (float)((st>>8)&0xFFFFu)/65536.0f; }
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, sA*4)); CHECK(cudaMalloc(&dB, sB*4)); CHECK(cudaMalloc(&dC, sC*4));
    CHECK(cudaMemcpy(dA, hA, sA*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, sB*4, cudaMemcpyHostToDevice));
    dAg = dA; dBg = dB; dCg = dC;

    printf("\n-- warming up: 1500 ms streaming, then 500 ms compute -------------\n");
    { size_t nb = (size_t)256*1024*1024/16; float4 *ds; float *dsink;
      CHECK(cudaMalloc(&ds, nb*16)); CHECK(cudaMemset(ds, 1, nb*16));
      CHECK(cudaMalloc(&dsink, 4));
      cudaEvent_t w0, w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el = 0; CHECK(cudaEventRecord(w0));
      while (el < 1500.0f) { warmStream<<<320,256>>>(ds, dsink, nb);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el, w0, w1)); }
      el = 0; CHECK(cudaEventRecord(w0));
      while (el < 500.0f) { warmFfma<<<480,128>>>(dsink, 2000);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el, w0, w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
      CHECK(cudaFree(ds)); CHECK(cudaFree(dsink)); }

    void (*runs[2])(void) = { runBase, runMine };
    int iters[2]; double best[2] = {1e30, 1e30};
    for (int i = 0; i < 2; ++i) {
        double t = timeOne(runs[i], 1);
        int n = (int)(10.0/(t > 0 ? t : 0.01));
        if (n < 3) n = 3; if (n > 64) n = 64; iters[i] = n;
    }
    for (int s = 0; s < 4; ++s)                  // SWEEPS >= NCFG, rotated
        for (int q = 0; q < 2; ++q) {
            int p = (q + s) % 2;
            double t = timeOne(runs[p], iters[p]);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    const double flops = 2.0*M*N*K;
    const double gb = flops/(best[0]*1e-3)/1e9, gm = flops/(best[1]*1e-3)/1e9;
    const double speedup = best[0]/best[1];

    printf("\n-- measurement ----------------------------------------------------\n");
    printf("  %-38s %9.4f ms %10.1f GFLOP/s\n", "Module 17 baseline (32x32 tile)", best[0], gb);
    printf("  %-38s %9.4f ms %10.1f GFLOP/s\n", "your register-tiled kernel", best[1], gm);
    printf("  speedup over the baseline                 %6.2f x\n", speedup);
    printf("  fraction of the measured FP32 ceiling     %6.2f %%  (18000 GFLOP/s)\n",
           100.0*gm/18000.0);

    cudaFuncAttributes at; CHECK(cudaFuncGetAttributes(&at, (const void*)gemmRegTile));
    int blk = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blk, (const void*)gemmRegTile, NT, 0));
    printf("  your kernel: %d registers, %d B spilled, %d B shared, %d blocks/SM, %.1f%% occupancy\n",
           at.numRegs, (int)at.localSizeBytes, (int)at.sharedSizeBytes, blk,
           100.0*blk*NT/1536.0);

    // ---- scoring
    const double fplScalar = (double)TM*TN/(TM+TN);
    int trueP1 = (fplScalar < 1) ? 1 : (fplScalar < 2) ? 2 : (fplScalar < 3) ? 3
               : (fplScalar < 5) ? 4 : 5;
    int trueP2 = (speedup < 1.5) ? 1 : (speedup < 3.0) ? 2 : (speedup < 4.5) ? 3
               : (speedup < 7.0) ? 4 : 5;

    printf("\n-- predictions ----------------------------------------------------\n");
    printf("  P1 FMAs per scalar shared read : you said %d, measured/derived %d  (%.2f)  %s\n",
           PRED_P1, trueP1, fplScalar, PRED_P1 == trueP1 ? "correct" : "WRONG");
    printf("  P2 speedup over the M17 kernel : you said %d, measured %d  (%.2f x)  %s\n",
           PRED_P2, trueP2, speedup, PRED_P2 == trueP2 ? "correct" : "WRONG");

    int score = 0;
    if (ok == nck) score += 3;                       // all four shapes
    if (ok == nck) score += 1;                       // includes the +inf check
    if (speedup >= 4.2) score += 2;
    if (PRED_P1 == trueP1) score += 2;
    if (PRED_P2 == trueP2) score += 2;

    printf("\n  score %d/10  (correctness 4, performance gate 2, predictions 4)\n", score);
    if (speedup < 4.2)
        printf("  The gate was missed. Your kernel is correct and is leaving a\n"
               "  factor on the table. Start with TODO 1: work out how many\n"
               "  DISTINCT addresses a warp presents to the As read and to the\n"
               "  Bs read under your choice of tRow/tCol, then do the same for\n"
               "  the other choice. Neither version changes a single output\n"
               "  value, and they are about 1.4x apart.\n");

    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    free(hA); free(hB);
    printf("\nOVERALL: %s\n", score == 10 ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return score == 10 ? 0 : 1;
}
