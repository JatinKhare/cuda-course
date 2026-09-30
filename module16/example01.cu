// =============================================================================
// Module 16 / Example 1 — the naive GEMM, its BLAS semantics, and how to
//                         validate a floating-point GEMM properly.
//
// GOAL : Establish (a) the one-thread-one-output-element kernel for
//        C = alpha*A*B + beta*C on row-major M x K, K x N, M x N;
//        (b) the validation methodology Modules 17 and 18 will reuse;
//        (c) cuBLAS as the measured reference point, including the
//        column-major argument swap that a row-major caller needs.
//
// BUILD: nvcc -arch=sm_89 -O3 -lcublas -o example01.exe example01.cu
// RUN  : example01.exe
//
// Sections:
//   A  the problem, the traffic ledger, the arithmetic intensity
//   B  the naive kernel and the alpha/beta contract
//   C  the validator: finiteness, Freivalds probe, sampled exact reference
//   D  five deliberately broken kernels vs the validator
//   E  cuBLAS: the swapped-argument call, verified, and timed against naive
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK(call) do {                                                       \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        printf("CUDA error %s (%s) at %s:%d\n", cudaGetErrorName(_e),          \
               cudaGetErrorString(_e), __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define CHECK_CUBLAS(call) do {                                                \
    cublasStatus_t _s = (call);                                                \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
        printf("cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__);     \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// Non-square, non-power-of-two on purpose: every boundary guard is exercised,
// and a transposed result is not even the right shape.
#define M_DIM 1027
#define N_DIM 2053
#define K_DIM  769

// ---------------------------------------------------------------- deterministic
// Index-derived, never unseeded rand().
static unsigned lcg_state = 1u;
static float rnd01(void) { lcg_state = lcg_state * 1664525u + 1013904223u;
                           return (float)((lcg_state >> 8) & 0xFFFFu) / 65536.0f; }

// =============================================================================
// B. The naive kernel.
//
// One thread owns exactly one element of C. It walks the full K-length dot
// product of row `row` of A against column `col` of B.
//
//   A is M x K row-major : A[row*K + k]      contiguous in k
//   B is K x N row-major : B[k*N + col]      stride N in k, contiguous in col
//   C is M x N row-major : C[row*N + col]
//
// Three matrices, three different leading dimensions, and the k index means a
// different thing in each of them. That is the whole difficulty of writing it.
// =============================================================================
__global__ void gemmNaive(int M, int N, int K,
                          float alpha, const float *A, const float *B,
                          float beta, float *C)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;   // x -> the N axis
    const int row = blockIdx.y * blockDim.y + threadIdx.y;   // y -> the M axis
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];

    // BLAS contract: when beta == 0, C is NOT read. C may legally contain
    // uninitialised memory, or NaN, or Inf, on entry. `alpha*acc + 0.0f*C`
    // propagates a NaN; `alpha*acc` does not.
    if (beta == 0.0f) C[(size_t)row * N + col] = alpha * acc;
    else              C[(size_t)row * N + col] = alpha * acc
                                               + beta * C[(size_t)row * N + col];
}

// ---------------------------------------------------------------- D: defects
enum { DEF_NONE = 0, DEF_K_OFF_BY_ONE, DEF_WRONG_GUARD, DEF_BETA_READS,
       DEF_B_ROW_MAJOR_SWAP, DEF_A_COL_MAJOR, N_DEFECT };
static const char *DEFECT_NAME[N_DEFECT] = {
    "correct",
    "k loop runs to K-1 (off-by-one in the reduction)",
    "guard tests col < M instead of col < N",
    "beta applied unconditionally (reads C when beta == 0)",
    "B indexed as if it were column-major: B[col*K + k]",
    "A indexed as if it were column-major: A[k*M + row]"
};

__global__ void gemmDefective(int M, int N, int K,
                              float alpha, const float *A, const float *B,
                              float beta, float *C, int defect)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (defect == DEF_WRONG_GUARD) { if (row >= M || col >= M) return; }
    else                           { if (row >= M || col >= N) return; }

    const int kend = (defect == DEF_K_OFF_BY_ONE) ? K - 1 : K;
    float acc = 0.0f;
    for (int k = 0; k < kend; ++k) {
        float a, b;
        if (defect == DEF_A_COL_MAJOR)        a = A[(size_t)k * M + row];
        else                                  a = A[(size_t)row * K + k];
        if (defect == DEF_B_ROW_MAJOR_SWAP)   b = B[(size_t)col * K + k];
        else                                  b = B[(size_t)k * N + col];
        acc += a * b;
    }
    if (beta == 0.0f && defect != DEF_BETA_READS)
        C[(size_t)row * N + col] = alpha * acc;
    else
        C[(size_t)row * N + col] = alpha * acc + beta * C[(size_t)row * N + col];
}

// =============================================================================
// C. The validator.
//
// A GEMM result cannot be compared bit-exactly against a CPU reference: the
// summation order differs, and float addition is not associative. It also
// cannot be compared with a fixed relative tolerance against |C|, because C
// can be arbitrarily smaller than the terms that produced it (cancellation).
//
// The correct yardstick is the standard backward-error bound for an inner
// product of length K accumulated in precision u:
//
//     | c_hat_ij - c_ij |  <=  gamma_K * S_ij ,
//     S_ij = sum_k |a_ik| * |b_kj| ,   gamma_K = K*u / (1 - K*u) ,  u = 2^-24
//
// So the quantity to threshold is  err_ij / (gamma_K * S_ij), not err_ij or
// err_ij / |c_ij|.  Three checks, cheapest and most complete first:
//
//   1. FINITENESS over all M*N elements.  O(MN).  Must come first: a NaN
//      compares false against everything, so a max-error loop silently
//      *passes* an array full of NaNs.
//   2. FREIVALDS PROBE with a non-negative random vector v.  Checks
//      C*v == alpha*A*(B*v) + beta*C0*v in double, with the bound propagated
//      through the same contraction.  Touches every element of C, and costs
//      O(MN + KN + MK) -- the same order as the problem's compulsory traffic.
//   3. SAMPLED EXACT REFERENCE: recompute a strided sample of (i,j) in double
//      with the exact S_ij, and threshold err/(gamma_K*S_ij).  O(samples*K).
//
// 1 and 2 give full coverage cheaply; 3 gives an exact per-element number on a
// sample. Modules 17 and 18 reuse this function unchanged.
// =============================================================================
typedef struct {
    int    ok;
    int    nonfinite;
    double freivalds;      // worst row ratio, must be <= 1
    int    freivaldsRow;
    double sampled;        // worst sampled scaled error, must be <= 1
    int    sampledRow, sampledCol;
} GemmCheck;

static GemmCheck gemmValidate(int M, int N, int K,
                              float alpha, const float *hA, const float *hB,
                              float beta,  const float *hC0, const float *hC,
                              int sampleStride)
{
    GemmCheck r; r.ok = 1; r.nonfinite = 0;
    r.freivalds = 0.0; r.freivaldsRow = -1;
    r.sampled = 0.0; r.sampledRow = -1; r.sampledCol = -1;

    const double u      = ldexp(1.0, -24);           // fp32 unit roundoff
    const double gammaK = (double)K * u / (1.0 - (double)K * u);

    // ---- check 1: finiteness, every element ----
    for (size_t i = 0; i < (size_t)M * N; ++i)
        if (!isfinite(hC[i])) ++r.nonfinite;
    if (r.nonfinite) r.ok = 0;

    // ---- check 2: Freivalds probe ----
    // v non-negative so that a *systematic* per-element error accumulates
    // coherently along the row instead of cancelling like a random walk.
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
            t  += b * v[j];
            tb += fabs(b) * v[j];
        }
        Bv[k] = t; aBv[k] = tb;
    }
    for (int i = 0; i < M; ++i) {
        double y = 0.0, ybound = 0.0;
        for (int k = 0; k < K; ++k) {
            double a = hA[(size_t)i * K + k];
            y      += a * Bv[k];
            ybound += fabs(a) * aBv[k];
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

    // ---- check 3: sampled exact double reference ----
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

static void printCheck(const char *label, GemmCheck c)
{
    printf("  %-52s nonfinite %8d | Freivalds %10.4g | sampled %10.4g | %s\n",
           label, c.nonfinite, c.freivalds, c.sampled, c.ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------- timing
static double timeBest(void (*run)(void*), void *ctx, int sweeps)
{
    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    // calibrate
    CHECK(cudaEventRecord(e0)); run(ctx); CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float one; CHECK(cudaEventElapsedTime(&one, e0, e1));
    int iters = (int)(10.0 / (one > 0.0f ? one : 0.01f));
    if (iters < 1) iters = 1; if (iters > 64) iters = 64;
    double best = 1e30;
    for (int s = 0; s < sweeps; ++s) {
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) run(ctx);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        double per = ms / iters; if (per < best) best = per;
    }
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    return best;
}

typedef struct {
    int M, N, K; float alpha, beta;
    const float *dA, *dB; float *dC;
    cublasHandle_t h;
} Ctx;

static void runNaive(void *p) {
    Ctx *c = (Ctx*)p;
    dim3 bl(32, 8);
    dim3 gr((c->N + bl.x - 1)/bl.x, (c->M + bl.y - 1)/bl.y);
    gemmNaive<<<gr, bl>>>(c->M, c->N, c->K, c->alpha, c->dA, c->dB, c->beta, c->dC);
}
static void runCublas(void *p) {
    Ctx *c = (Ctx*)p;
    // See section E for the derivation of this argument order.
    CHECK_CUBLAS(cublasSgemm(c->h, CUBLAS_OP_N, CUBLAS_OP_N,
                             c->N, c->M, c->K,
                             &c->alpha, c->dB, c->N, c->dA, c->K,
                             &c->beta,  c->dC, c->N));
}

// =============================================================================
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    const int M = M_DIM, N = N_DIM, K = K_DIM;
    const size_t sA = (size_t)M*K, sB = (size_t)K*N, sC = (size_t)M*N;

    printf("=== Module 16 / Example 1 — naive GEMM, validation, cuBLAS ===\n");
    printf("C(%d x %d) = alpha * A(%d x %d) * B(%d x %d) + beta * C, row-major\n\n",
           M, N, M, K, K, N);

    // -------------------------------------------------- A: the traffic ledger
    const double flops       = 2.0 * M * N * K;
    const double compulsory  = 4.0 * ((double)M*K + (double)K*N + (double)M*N);
    const double requested   = 4.0 * 2.0 * (double)M * N * K;
    printf("-- A. traffic ledger ---------------------------------------------\n");
    printf("  useful work                2*M*N*K            = %.4f GFLOP\n", flops/1e9);
    printf("  compulsory traffic         4*(MK + KN + MN)   = %.3f MB\n", compulsory/1e6);
    printf("  naive kernel REQUESTS      4*2*M*N*K          = %.3f GB\n", requested/1e9);
    printf("  requested / compulsory                        = %.1f x\n", requested/compulsory);
    printf("  arithmetic intensity vs compulsory traffic    = %.2f FLOP/byte\n",
           flops/compulsory);
    printf("  arithmetic intensity vs requested traffic     = %.2f FLOP/byte\n",
           flops/requested);
    printf("  (for a square n x n x n problem the compulsory intensity is n/6;\n"
           "   the requested intensity is 0.25 for every n. Example 2 places\n"
           "   both on a measured roofline.)\n\n");

    // -------------------------------------------------- data
    float *hA = (float*)malloc(sA*4), *hB = (float*)malloc(sB*4);
    float *hC = (float*)malloc(sC*4), *hC0 = (float*)malloc(sC*4);
    lcg_state = 1u;
    // Well-conditioned: strictly positive, so |C_ij| ~ S_ij and there is no
    // cancellation. Section C's commentary explains why this matters.
    for (size_t i = 0; i < sA; ++i) hA[i] = 0.5f + rnd01();
    for (size_t i = 0; i < sB; ++i) hB[i] = 0.5f + rnd01();
    for (size_t i = 0; i < sC; ++i) hC0[i] = (float)((int)(i % 17) - 8) * 0.25f;

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, sA*4)); CHECK(cudaMalloc(&dB, sB*4)); CHECK(cudaMalloc(&dC, sC*4));
    CHECK(cudaMemcpy(dA, hA, sA*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, sB*4, cudaMemcpyHostToDevice));

    dim3 bl(32, 8), gr((N+bl.x-1)/bl.x, (M+bl.y-1)/bl.y);

    // Poison C so that a kernel which reads C when beta == 0 is caught.
    float *poison = (float*)malloc(sC*4);
    for (size_t i = 0; i < sC; ++i) poison[i] = nanf("");

    // -------------------------------------------------- B/C: correct kernel
    printf("-- B/C. the naive kernel and the validator ------------------------\n");
    printf("  grid %u x %u of %u x %u threads = %.2f M threads, one per C element\n",
           gr.x, gr.y, bl.x, bl.y, (double)gr.x*gr.y*bl.x*bl.y/1e6);
    printf("  C is poisoned with NaN before every launch: beta == 0 must not read it.\n\n");

    CHECK(cudaMemcpy(dC, poison, sC*4, cudaMemcpyHostToDevice));
    gemmNaive<<<gr, bl>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hC, dC, sC*4, cudaMemcpyDeviceToHost));
    GemmCheck ck = gemmValidate(M, N, K, 1.0f, hA, hB, 0.0f, NULL, hC, 16);
    printCheck("alpha=1, beta=0", ck);

    CHECK(cudaMemcpy(dC, hC0, sC*4, cudaMemcpyHostToDevice));
    gemmNaive<<<gr, bl>>>(M, N, K, 0.75f, dA, dB, -1.25f, dC);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hC, dC, sC*4, cudaMemcpyDeviceToHost));
    GemmCheck ck2 = gemmValidate(M, N, K, 0.75f, hA, hB, -1.25f, hC0, hC, 16);
    printCheck("alpha=0.75, beta=-1.25", ck2);

    printf("\n  Read the headroom, not just PASS. gamma_K = K*u/(1-K*u) = %.3e at K = %d;\n"
           "  the kernel uses about %.1f%% of the error budget the bound allows.\n",
           (double)K*ldexp(1.0,-24)/(1.0-(double)K*ldexp(1.0,-24)), K, 100.0*ck.sampled);
    printf("  Note what a naive tolerance would do here. The house rule\n"
           "  fabs(a-b) <= 1e-5*max(1,|b|) is a RELATIVE test against |C|. At K = %d\n"
           "  a correct fp32 GEMM has a relative error of order K*u = %.1e, which is\n"
           "  already %.1fx that tolerance before any cancellation. With zero-mean\n"
           "  operands (|C| ~ sqrt(K)*sigma^2 but S ~ K*sigma^2) the same correct\n"
           "  kernel measures ~1e-3 relative and the house rule FAILS it.\n\n",
           K, (double)K*ldexp(1.0,-24), (double)K*ldexp(1.0,-24)/1e-5);

    // -------------------------------------------------- D: the defects
    printf("-- D. five defects vs the three checks ----------------------------\n");
    int caught = 0;
    for (int d = 0; d < N_DEFECT; ++d) {
        CHECK(cudaMemcpy(dC, poison, sC*4, cudaMemcpyHostToDevice));
        gemmDefective<<<gr, bl>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC, d);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hC, dC, sC*4, cudaMemcpyDeviceToHost));
        GemmCheck c = gemmValidate(M, N, K, 1.0f, hA, hB, 0.0f, NULL, hC, 16);
        printCheck(DEFECT_NAME[d], c);
        if (d == DEF_NONE) { if (c.ok) ++caught; }
        else               { if (!c.ok) ++caught; }
    }
    printf("  %d/%d outcomes as required.\n", caught, N_DEFECT);
    printf("  Observe: the NaN-poisoned cases are caught by the FINITENESS check\n"
           "  only. Their Freivalds and sampled numbers are 0.0, because every\n"
           "  comparison against a NaN is false. Order the checks accordingly.\n\n");

    // -------------------------------------------------- E: cuBLAS
    printf("-- E. cuBLAS as the reference point -------------------------------\n");
    printf("  cuBLAS is column-major. A row-major M x N matrix C with leading\n"
           "  dimension N is, byte for byte, the column-major N x M matrix C^T\n"
           "  with leading dimension N. So a row-major C = A*B is the\n"
           "  column-major identity  C^T = B^T * A^T, which cublasSgemm computes\n"
           "  with NO transpose flags if you simply swap the operands:\n\n"
           "    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,\n"
           "                N, M, K,             // m, n, k of the column-major call\n"
           "                &alpha, B, N,        // 'A' = B, lda = N\n"
           "                        A, K,        // 'B' = A, ldb = K\n"
           "                &beta,  C, N);       // ldc = N\n\n"
           "  Nothing is transposed and nothing is copied; the same bytes are\n"
           "  simply read under the other convention. Passing CUBLAS_OP_T instead\n"
           "  is the classic mistake: it compiles, it runs, and on a square\n"
           "  problem it even produces a plausible matrix.\n\n");

    cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
    CHECK(cudaMemcpy(dC, poison, sC*4, cudaMemcpyHostToDevice));
    { const float a1 = 1.0f, b0 = 0.0f;
      CHECK_CUBLAS(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                               &a1, dB, N, dA, K, &b0, dC, N)); }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hC, dC, sC*4, cudaMemcpyDeviceToHost));
    GemmCheck ckb = gemmValidate(M, N, K, 1.0f, hA, hB, 0.0f, NULL, hC, 16);
    printCheck("cublasSgemm, swapped arguments", ckb);

    // ---- timing, spec 12: 1500 ms warm-up, back-to-back, rotated, min-of-N
    printf("\n  warming up (1500 ms, which ramps the memory P-state as well as\n"
           "  the SM clock) ...\n");
    { cudaEvent_t w0, w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el = 0.0f; CHECK(cudaEventRecord(w0));
      while (el < 1500.0f) {
          gemmNaive<<<gr, bl>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
          CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
          CHECK(cudaEventElapsedTime(&el, w0, w1));
      }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); }

    Ctx ctx; ctx.M=M; ctx.N=N; ctx.K=K; ctx.alpha=1.0f; ctx.beta=0.0f;
    ctx.dA=dA; ctx.dB=dB; ctx.dC=dC; ctx.h=h;
    const int SWEEPS = 4;                    // 2 configurations, SWEEPS >= NCFG
    double tn = 1e30, tc = 1e30;
    for (int s = 0; s < SWEEPS; ++s) {
        // rotate which configuration leads the sweep
        if (s & 1) { double a = timeBest(runNaive,  &ctx, 1); if (a<tn) tn=a;
                     double b = timeBest(runCublas, &ctx, 1); if (b<tc) tc=b; }
        else       { double b = timeBest(runCublas, &ctx, 1); if (b<tc) tc=b;
                     double a = timeBest(runNaive,  &ctx, 1); if (a<tn) tn=a; }
    }
    CHECK(cudaGetLastError());

    const double gn = flops/(tn*1e-3)/1e9, gc = flops/(tc*1e-3)/1e9;
    printf("\n  %-28s %10s %12s %10s\n", "kernel", "ms", "GFLOP/s", "% cuBLAS");
    printf("  %-28s %10.4f %12.1f %9.2f%%\n", "naive, (32,8), x->col", tn, gn, 100.0*gn/gc);
    printf("  %-28s %10.4f %12.1f %9.2f%%\n", "cublasSgemm", tc, gc, 100.0);
    printf("\n  naive / cuBLAS time ratio = %.1f x.  Same FLOPs, same hardware,\n"
           "  same fp32 arithmetic, no Tensor Cores on either side.\n", tn/tc);
    printf("  Module 36 owns cuBLAS itself; here it is only the reference point.\n");

    // -------------------------------------------------- cleanup
    CHECK_CUBLAS(cublasDestroy(h));
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC); free(hC0); free(poison);

    const int pass = ck.ok && ck2.ok && ckb.ok && (caught == N_DEFECT);
    printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return pass ? 0 : 1;
}
