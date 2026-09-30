// =============================================================================
// Module 16 / Exercise 2 — the mapping decision, analysed and then measured.
//
// GOAL : The naive GEMM kernel contains exactly one free choice that costs
//        nothing to make and an order of magnitude to make badly: which of the
//        block's two axes carries the row index of C and which carries the
//        column index. You will count the sectors a single warp requests under
//        both assignments and every block shape, predict the ranking, and then
//        measure all of it.
//
//        Module 3 Exercise 1 made you meet this on a stencil, where the wrong
//        axis cost 3.1x-3.5x. Module 5 gave you the counting procedure. Here the
//        same decision is worth more, and the count is over two matrices with
//        different leading dimensions at once.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
// RUN  : exercise02.exe
//
// TODO 1 - predictSectors(): the per-warp sector count, by enumeration
// TODO 2 - the second kernel and its launch configuration
// TODO 3 - two predictions, committed before you build
// TODO 4 - DESIGN: use your own model to choose the block shape
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
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
#define THREADS_PER_BLOCK 256

// Mapping 0 ("x -> col"): threadIdx.x carries the column index of C.
// Mapping 1 ("x -> row"): threadIdx.x carries the row index of C.
enum { MAP_XCOL = 0, MAP_XROW = 1 };

// =============================================================================
// Kernel A, supplied: mapping 0.
// =============================================================================
__global__ void gemmXcol(int M, int N, int K, const float *A, const float *B, float *C)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[(size_t)row*K + k] * B[(size_t)k*N + col];
    C[(size_t)row*N + col] = acc;
}

// =============================================================================
// TODO 2 — kernel B: mapping 1.
//
// Same arithmetic, same operands, same output. The ONLY difference is that
// threadIdx.x now carries `row` and threadIdx.y carries `col`. Write it, and
// then set its grid dimensions in `gridFor()` below so that it still covers
// every element of C exactly once. (Both halves of that sentence are the
// exercise: Module 3 Exercise 1's trap was that the axis choice appears twice,
// in the kernel and in the launch, and the two must agree.)
//
// If you leave this unfilled the harness will report every element unwritten
// rather than crashing.
// =============================================================================
__global__ void gemmXrow(int M, int N, int K, const float *A, const float *B, float *C)
{
    // YOUR CODE HERE
    (void)M; (void)N; (void)K; (void)A; (void)B; (void)C;
}

static dim3 gridFor(int mapping, int M, int N, int bx, int by)
{
    if (mapping == MAP_XCOL)
        return dim3((unsigned)((N + bx - 1)/bx), (unsigned)((M + by - 1)/by));
    // TODO 2 (second half): the grid for mapping 1.
    // YOUR CODE HERE
    return dim3(0u, 0u);
}

// =============================================================================
// TODO 1 — predictSectors().
//
// Module 5's procedure, applied to warp 0 of block 0 at k = 0:
//   - a warp is 32 consecutive linearized thread ids, and the linearization is
//     tid = threadIdx.x + blockDim.x * threadIdx.y  (Module 3). So for
//     lane = 0..31:   tx = lane % bx,   ty = lane / bx;
//   - turn (tx, ty) into (row, col) according to `mapping`;
//   - the lane's A address is  &A[row*K + 0], its B address is &B[0*N + col];
//   - convert each byte address to a 32-byte SECTOR index;
//   - count the number of DISTINCT sector indices, separately for A and for B.
//
// Two things people get wrong. Count distinct *sectors*, not distinct lanes:
// 32 lanes that all present the same address cost one sector, not 32 (that is
// a broadcast, and Module 4 and Module 7 both priced it at 1). And a warp is
// not a row of the block unless blockDim.x is a multiple of 32 -- with
// bx = 8 a warp spans four values of threadIdx.y.
//
// Assume bx divides 32 or is a multiple of 32, and that the arrays are
// 32-byte aligned (cudaMalloc guarantees 256).
// =============================================================================
static void predictSectors(int bx, int by, int mapping, int M, int N, int K,
                           int *secA, int *secB)
{
    (void)bx; (void)by; (void)mapping; (void)M; (void)N; (void)K;
    // YOUR CODE HERE
    *secA = 0;
    *secB = 0;
}

// =============================================================================
// TODO 3 — two predictions. Commit before you build.
//
// P1: Over the 12 configurations the harness measures (6 block shapes with
//     bx in {1,2,4,8,16,32} at 256 threads, x 2 mappings), what is the ratio
//     slowest / fastest?  Answer with a bucket:
//        1 = under 1.5x   2 = 1.5x to 2.5x   3 = 2.5x to 6x   4 = over 6x
//
// P2: For mapping 1 ("x -> row"), which value of blockDim.x is FASTEST?
//     Answer with the number itself (one of 1, 2, 4, 8, 16, 32). Scored with
//     a 5% band, because two of the six are genuinely close.
// =============================================================================
#define PREDICT_SPREAD_BUCKET  0     // TODO 3a: 1..4, 0 = unset
#define PREDICT_BEST_BX_XROW   0     // TODO 3b: 1,2,4,8,16 or 32; 0 = unset

// =============================================================================
// TODO 4 — DESIGN: choose the block shape from your own model.
//
// Fill in chooseBx() so that it returns the value of blockDim.x that YOUR
// predictSectors() says is best for the given mapping, searching the candidate
// list {1, 2, 4, 8, 16, 32} at 256 threads per block. Break ties toward the
// larger bx.
//
// Do not hard-code an answer you read off the measured table -- the harness
// runs chooseBx() BEFORE it times anything, prints your choice, and then tells
// you whether the measurement agreed. A hard-coded constant that happens to be
// right teaches you nothing and will not survive the next problem shape.
//
// Then read the result carefully. The model is a count of sectors requested by
// one warp on one instruction. It is not a count of bytes fetched from DRAM,
// it says nothing about how many warps re-request the same sector, and it has
// no term for the L2. Expect it to be an excellent predictor of the ORDER and
// a poor predictor of the RATIO, and be ready to say why.
// =============================================================================
static int chooseBx(int mapping, int M, int N, int K)
{
    (void)mapping; (void)M; (void)N; (void)K;
    // YOUR CODE HERE
    return 0;
}

// ---------------------------------------------------------------- FNV-1a
static unsigned fnv1a(const int *v, int n)
{
    unsigned h = 2166136261u;
    for (int i = 0; i < n; ++i) {
        unsigned x = (unsigned)v[i];
        for (int b = 0; b < 4; ++b) { h ^= (x >> (8*b)) & 0xFFu; h *= 16777619u; }
    }
    return h;
}
#define SECTOR_HASH 0xfdf98fc1u     // hash of the 16 reference sector counts

// ---------------------------------------------------------------- timing
typedef struct { int M,N,K,bx,by,mapping; const float *A,*B; float *C; } GCtx;
static void runG(GCtx *g) {
    dim3 bl((unsigned)g->bx, (unsigned)g->by);
    dim3 gr = gridFor(g->mapping, g->M, g->N, g->bx, g->by);
    if (gr.x == 0u || gr.y == 0u) return;
    if (g->mapping == MAP_XCOL) gemmXcol<<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C);
    else                        gemmXrow<<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C);
}
static double timeIt(GCtx *g, int iters) {
    cudaEvent_t e0,e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) runG(g);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms,e0,e1));
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    return ms/iters;
}

// =============================================================================
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const int M = M_DIM, N = N_DIM, K = K_DIM;
    const size_t sA = (size_t)M*K, sB = (size_t)K*N, sC = (size_t)M*N;
    const double flops = 2.0*M*N*K;

    printf("=== Module 16 / Exercise 2 — the mapping decision ===\n");
    printf("C(%d x %d) = A(%d x %d) * B(%d x %d), %d threads per block\n\n",
           M, N, M, K, K, N, THREADS_PER_BLOCK);

    if (PREDICT_SPREAD_BUCKET == 0 || PREDICT_BEST_BX_XROW == 0) {
        printf("Set TODO 3 (PREDICTIONS) first.\n"); return 0;
    }

    // ---------------------------------------------------- TODO 1 scored
    const int probe[8][3] = {   // {bx, by, mapping}
        {32,  8, MAP_XCOL}, {16, 16, MAP_XCOL}, { 8, 32, MAP_XCOL}, { 1,256, MAP_XCOL},
        {32,  8, MAP_XROW}, {16, 16, MAP_XROW}, { 2,128, MAP_XROW}, { 4, 64, MAP_XROW}
    };
    int got[16]; int anyZero = 0;
    printf("-- TODO 1: your sector model --------------------------------------\n");
    printf("  %-10s %-9s %8s %8s %8s\n", "mapping", "block", "A sect", "B sect", "total");
    for (int i = 0; i < 8; ++i) {
        int a = 0, b = 0;
        predictSectors(probe[i][0], probe[i][1], probe[i][2], M, N, K, &a, &b);
        got[2*i] = a; got[2*i+1] = b;
        if (a <= 0 || b <= 0) anyZero = 1;
        printf("  %-10s (%3d,%3d) %8d %8d %8d\n",
               probe[i][2] ? "x -> row" : "x -> col",
               probe[i][0], probe[i][1], a, b, a+b);
    }
    if (anyZero) { printf("\nSet TODO 1 first.\n"); return 0; }
    const unsigned h = fnv1a(got, 16);
    const int modelOK = (h == SECTOR_HASH);
    printf("  model hash %08x  -> %s\n\n", h, modelOK ? "correct" : "WRONG");

    // ---------------------------------------------------- TODO 4 (before timing)
    const int bxChosenCol = chooseBx(MAP_XCOL, M, N, K);
    const int bxChosenRow = chooseBx(MAP_XROW, M, N, K);
    if (bxChosenCol <= 0 || bxChosenRow <= 0) { printf("Set TODO 4 first.\n"); return 0; }
    printf("-- TODO 4: your model's choice, made before any measurement -------\n");
    printf("  x -> col : blockDim.x = %d\n", bxChosenCol);
    printf("  x -> row : blockDim.x = %d\n\n", bxChosenRow);

    // ---------------------------------------------------- data
    float *dA,*dB,*dC;
    CHECK(cudaMalloc(&dA,sA*4)); CHECK(cudaMalloc(&dB,sB*4)); CHECK(cudaMalloc(&dC,sC*4));
    { float *h2 = (float*)malloc((sB>sA?sB:sA)*4);
      for (size_t i=0;i<sA;++i) h2[i] = 0.5f + (float)(i % 251)/502.0f;
      CHECK(cudaMemcpy(dA,h2,sA*4,cudaMemcpyHostToDevice));
      for (size_t i=0;i<sB;++i) h2[i] = 0.5f + (float)(i % 257)/514.0f;
      CHECK(cudaMemcpy(dB,h2,sB*4,cudaMemcpyHostToDevice));
      free(h2); }

    // correctness of the reader's kernel B, before any timing
    float *hC = (float*)malloc(sC*4), *hRef = (float*)malloc(sC*4);
    {
        GCtx g = {M,N,K,32,8,MAP_XCOL,dA,dB,dC};
        CHECK(cudaMemset(dC,0,sC*4)); runG(&g);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hRef,dC,sC*4,cudaMemcpyDeviceToHost));
    }
    int kernelBok = 1; size_t nbad = 0;
    for (int bx = 1; bx <= 32; bx *= 2) {
        GCtx g = {M,N,K,bx,THREADS_PER_BLOCK/bx,MAP_XROW,dA,dB,dC};
        CHECK(cudaMemset(dC,0,sC*4)); runG(&g);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hC,dC,sC*4,cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < sC; ++i)
            if (!(fabs((double)hC[i] - (double)hRef[i]) <= 1e-5*fabs((double)hRef[i]))) ++bad;
        if (bad) { kernelBok = 0; nbad += bad; }
    }
    printf("-- TODO 2: kernel B correctness (bit-comparable to kernel A) ------\n");
    if (kernelBok) printf("  all six block shapes agree with the mapping-0 kernel.\n\n");
    else { printf("  %zu of %zu elements differ across the six shapes.\n\n", nbad, sC*6); }

    // ---------------------------------------------------- timing, spec 12
    enum { NCFG = 12 };
    GCtx cfg[NCFG]; int bxs[6] = {1,2,4,8,16,32};
    for (int m = 0; m < 2; ++m)
        for (int i = 0; i < 6; ++i) {
            GCtx g = {M,N,K,bxs[i],THREADS_PER_BLOCK/bxs[i],m,dA,dB,dC};
            cfg[m*6+i] = g;
        }
    printf("  warming up 1500 ms ...\n");
    { cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0.0f; CHECK(cudaEventRecord(w0));
      while (el < 1500.0f) { runG(&cfg[11]);
          CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
          CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); }

    double best[NCFG]; int iters[NCFG];
    for (int i = 0; i < NCFG; ++i) {
        best[i] = 1e30;
        double one = timeIt(&cfg[i], 1);
        int it = (int)(10.0/(one > 0.0 ? one : 0.01)); if (it < 1) it = 1; if (it > 64) it = 64;
        iters[i] = it;
    }
    for (int s = 0; s < NCFG; ++s)            // SWEEPS = NCFG, rotated
        for (int q = 0; q < NCFG; ++q) {
            int p = (q+s) % NCFG;
            double t = timeIt(&cfg[p], iters[p]);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    int iFast = 0, iSlow = 0;
    for (int i = 1; i < NCFG; ++i) { if (best[i] < best[iFast]) iFast = i;
                                     if (best[i] > best[iSlow]) iSlow = i; }
    printf("\n-- measured: 12 configurations, one rotated sweep -----------------\n");
    printf("  %-10s %-9s %8s %10s %12s %10s\n",
           "mapping","block","sectors","ms","GFLOP/s","vs best");
    for (int i = 0; i < NCFG; ++i) {
        int a,b; predictSectors(cfg[i].bx, cfg[i].by, cfg[i].mapping, M,N,K, &a,&b);
        printf("  %-10s (%3d,%3d) %8d %10.4f %12.1f %9.2fx%s\n",
               cfg[i].mapping ? "x -> row" : "x -> col", cfg[i].bx, cfg[i].by, a+b,
               best[i], flops/(best[i]*1e-3)/1e9, best[i]/best[iFast],
               (i == iFast) ? "  <- fastest" : "");
    }

    const double spread = best[iSlow]/best[iFast];
    const int bucket = (spread < 1.5) ? 1 : (spread < 2.5) ? 2 : (spread < 6.0) ? 3 : 4;
    int measBestBxRow = bxs[0];
    { int bi = 6; for (int i = 7; i < 12; ++i) if (best[i] < best[bi]) bi = i;
      measBestBxRow = cfg[bi].bx; }
    int measBestBxCol = bxs[0];
    { int bi = 0; for (int i = 1; i < 6; ++i) if (best[i] < best[bi]) bi = i;
      measBestBxCol = cfg[bi].bx; }

    printf("\n-- model vs measurement ------------------------------------------\n");
    printf("  spread slowest/fastest     : %.2fx -> bucket %d   predicted %d  %s\n",
           spread, bucket, PREDICT_SPREAD_BUCKET,
           bucket == PREDICT_SPREAD_BUCKET ? "correct" : "WRONG");
    double bestRow = 1e30, bestCol = 1e30;
    double tPredRow = -1.0, tChoRow = -1.0, tChoCol = -1.0;
    for (int i = 0; i < 6; ++i) {
        if (best[i]   < bestCol) bestCol = best[i];
        if (best[6+i] < bestRow) bestRow = best[6+i];
        if (cfg[i].bx   == bxChosenCol)          tChoCol  = best[i];
        if (cfg[6+i].bx == bxChosenRow)          tChoRow  = best[6+i];
        if (cfg[6+i].bx == PREDICT_BEST_BX_XROW) tPredRow = best[6+i];
    }
    const int p2ok  = (tPredRow > 0.0) && (tPredRow <= 1.05*bestRow);
    const int c4row = (tChoRow  > 0.0) && (tChoRow  <= 1.05*bestRow);
    const int c4col = (tChoCol  > 0.0) && (tChoCol  <= 1.05*bestCol);
    printf("  fastest bx for x -> row    : %d   predicted %d -> %.4f ms vs %.4f ms  %s\n",
           measBestBxRow, PREDICT_BEST_BX_XROW, tPredRow, bestRow,
           p2ok ? "correct (within 5%)" : "WRONG");
    printf("  your model chose bx = %d for x -> row, measurement says %d  %s\n",
           bxChosenRow, measBestBxRow, c4row ? "agree (within 5%)" : "DISAGREE");
    printf("  your model chose bx = %d for x -> col, measurement says %d  %s\n",
           bxChosenCol, measBestBxCol, c4col ? "agree (within 5%)" : "DISAGREE");
    printf("\n  Compare the sector column against the ms column row by row. Then\n"
           "  compare the two ratios: sectors(worst)/sectors(best) against\n"
           "  ms(worst)/ms(best). One of them is much larger than the other, and\n"
           "  the reason is the same one Module 6 gave for the stencil.\n");

    const int score = modelOK + kernelBok
                    + (bucket == PREDICT_SPREAD_BUCKET)
                    + p2ok + c4row + c4col;
    printf("\n  SCORE: %d/6\n", score);
    printf("OVERALL: %s\n", score == 6 ? "PASS" : "FAIL");

    free(hC); free(hRef);
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    CHECK(cudaDeviceReset());
    return score == 6 ? 0 : 1;
}
