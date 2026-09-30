// =============================================================================
// Module 16 / Exercise 1 — SOLUTION
//                          write the naive GEMM, and write the test that can
//                          tell whether it is right.
//
// GOAL : Implement C = alpha*A*B + beta*C for row-major A (M x K), B (K x N),
//        C (M x N) with one thread per element of C, at
//        M = 1027, N = 2053, K = 769 -- non-square, and no dimension a
//        multiple of any block size the harness uses. Then supply the
//        per-element tolerance that a floating-point GEMM has to be judged by.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
// RUN  : exercise01_solution.exe
//
// The harness runs YOUR kernel and four deliberately defective kernels, on two
// datasets with very different conditioning, and scores you on:
//   - your kernel being accepted on both datasets,
//   - all four defects being rejected on both datasets,
//   - your tolerance function having the right shape (it is probed directly).
//
// TODO 1 - this thread's (row, col) and the out-of-range guard      [kernel]
// TODO 2 - the K-length dot product                                 [kernel]
// TODO 3 - the store, honouring the BLAS beta == 0 contract         [kernel]
// TODO 4 - the launch configuration                                 [host]
// TODO 5 - the tolerance law                          [host, DESIGN TODO]
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

#define BLOCK_X 32
#define BLOCK_Y  8

// Written into C before every launch. Two properties are wanted at once:
//   - a correct kernel can never produce it, so an element nobody wrote is
//     detectable as "unwritten";
//   - beta * UNWRITTEN is NOT zero for beta == 0, so a kernel that reads C
//     when it must not is detectable as "nonfinite". 0.0f * inf is NaN.
static const float UNWRITTEN = std::numeric_limits<float>::infinity();

// =============================================================================
// TODO 1, TODO 2, TODO 3 — the kernel.
//
// Layout reminder, row-major, no padding:
//     A is M x K,  element (r, k) lives at A[r*K + k]
//     B is K x N,  element (k, c) lives at B[k*N + c]
//     C is M x N,  element (r, c) lives at C[r*N + c]
//
// Three matrices, three different leading dimensions, and the summation index
// k is the fast axis of one of them and the slow axis of another.
// =============================================================================
__global__ void gemmNaive(int M, int N, int K,
                          float alpha, const float *A, const float *B,
                          float beta, float *C)
{
    // -------------------------------------------------------------------
    // TODO 1: give this thread exactly one element of C.
    //   - derive `row` in [0, M) and `col` in [0, N) from blockIdx/blockDim/
    //     threadIdx;
    //   - return immediately if this thread has no element to compute.
    // Which of the two axes you give to `col` is a decision, not a convention;
    // it is worth one order of magnitude and Exercise 2 measures it. Whatever
    // you choose here, TODO 4 must agree with it.
    // -------------------------------------------------------------------
    // SOLUTION. threadIdx.x is the fast axis of the warp linearization
    // (Module 3), so giving it `col` makes a warp's 32 lanes 32 consecutive
    // columns: B is then read contiguously and A is a broadcast.
    const int col = blockIdx.x * blockDim.x + threadIdx.x;   // the N axis
    const int row = blockIdx.y * blockDim.y + threadIdx.y;   // the M axis
    if (row >= M || col >= N) return;

    // -------------------------------------------------------------------
    // SOLUTION TODO 2.
    // -------------------------------------------------------------------
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];

    // -------------------------------------------------------------------
    // TODO 3: write the result.
    //
    // The BLAS contract for GEMM is not "C = alpha*A*B + beta*C" evaluated
    // literally. When beta is exactly zero, C is *not read*: it is treated as
    // write-only, and its prior contents are irrelevant. That is not a
    // micro-optimization, it is the specification, and callers rely on it --
    // a freshly cudaMalloc'd C is uninitialised memory and may hold anything.
    // Write the store so that both the beta != 0 and the beta == 0 cases are
    // correct. The harness fills C with NaN before every beta == 0 launch.
    // -------------------------------------------------------------------
    // SOLUTION TODO 3. beta == 0 must not read C.
    if (beta == 0.0f) C[(size_t)row * N + col] = alpha * acc;
    else              C[(size_t)row * N + col] = alpha * acc
                                               + beta * C[(size_t)row * N + col];

}

// ---------------------------------------------------------------------------
// Four defective kernels. You are not asked to fix them; the harness runs them
// against YOUR tolerance to check that your tolerance rejects them.
// ---------------------------------------------------------------------------
enum { DEF_K_OFF_BY_ONE = 0, DEF_WRONG_GUARD, DEF_BETA_READS, DEF_B_COL_MAJOR,
       N_DEFECT };
static const char *DEFECT_NAME[N_DEFECT] = {
    "k loop stops one short",
    "guard tests col < M instead of col < N",
    "beta applied unconditionally",
    "B indexed as B[col*K + k]"
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
        float b = (defect == DEF_B_COL_MAJOR) ? B[(size_t)col*K + k]
                                              : B[(size_t)k*N + col];
        acc += A[(size_t)row*K + k] * b;
    }
    if (beta == 0.0f && defect != DEF_BETA_READS) C[(size_t)row*N + col] = alpha*acc;
    else C[(size_t)row*N + col] = alpha*acc + beta*C[(size_t)row*N + col];
}

// =============================================================================
// TODO 5 — the tolerance law.  DESIGN TODO.
//
// You are given, for one element of C:
//     K    the length of the dot product
//     S    sum over k of |A[row][k]| * |B[k][col]|, computed exactly in double
//     ref  the exact value of that element of C, computed in double
// Return the largest |C_gpu - ref| you are willing to accept from a *correct*
// fp32 kernel. alpha = 1 and beta = 0 throughout this exercise.
//
// Requirements the harness checks directly, by calling this function with
// synthetic arguments:
//   (a) it must be proportional to S;
//   (b) doubling K must roughly double it (between 1.5x and 2.5x);
//   (c) at K = 769 it must be at least the theoretical worst case for fp32
//       accumulation of K terms, and no more than 100x that -- a tolerance
//       that accepts everything is not a test;
//   (d) it must be strictly positive even when ref is 0.
//
// And the requirement that actually decides the design: the harness runs two
// datasets. Dataset A has strictly positive operands, so |ref| is about as
// large as S. Dataset B has zero-mean operands, so the terms cancel and |ref|
// can be three orders of magnitude below S while the rounding errors that
// produced it are unchanged. Your tolerance must accept a correct kernel on
// BOTH, and reject all four defects on BOTH.
//
// The unit roundoff of IEEE-754 binary32 is u = 2^-24. Do not guess it; the
// standard error bound for a length-K inner product is worth looking up or
// deriving, and the module lesson states it.
// =============================================================================
static double gemmTolerance(int K, double S, double ref)
{
    // SOLUTION.
    //   |c_hat - c| <= gamma_K * sum_k |a_ik| |b_kj|,  gamma_K = K*u/(1-K*u)
    // plus one more rounding for the final alpha scaling and store, plus an
    // absolute floor so that an exactly-zero reference still has a window.
    // SAFETY is the only judgement call: 4x leaves ~30x of measured headroom
    // on a correct kernel and still rejects the tightest planted defect by
    // more than an order of magnitude.
    const double u      = ldexp(1.0, -24);        // fp32 unit roundoff, 2^-24
    const double gammaK = (double)K * u / (1.0 - (double)K * u);
    const double SAFETY = 4.0;
    (void)ref;                                    // deliberately NOT used
    return SAFETY * gammaK * S + 8.0 * u * S / (double)K;
}

// ---------------------------------------------------------------------------
// Validation, supplied. Three checks, in this order:
//   1. no element left unwritten (sentinel survives)
//   2. every element finite
//   3. sampled exact double reference, judged by YOUR tolerance
// Order matters: a NaN compares false against everything, so a max-error loop
// run first would silently pass an array full of NaNs.
// ---------------------------------------------------------------------------
typedef struct { int ok, unwritten, nonfinite; double worst; int wi, wj; } Verdict;

static Verdict judge(int M, int N, int K, const float *hA, const float *hB,
                     const float *hC, int stride)
{
    Verdict v; v.ok = 1; v.unwritten = 0; v.nonfinite = 0;
    v.worst = 0.0; v.wi = -1; v.wj = -1;
    for (size_t i = 0; i < (size_t)M*N; ++i) {
        if (hC[i] == UNWRITTEN) ++v.unwritten;
        else if (!isfinite(hC[i])) ++v.nonfinite;
    }
    if (v.unwritten || v.nonfinite) v.ok = 0;
    for (int i = 0; i < M; i += stride)
        for (int j = 0; j < N; j += stride) {
            double acc = 0.0, S = 0.0;
            for (int k = 0; k < K; ++k) {
                double a = hA[(size_t)i*K + k], b = hB[(size_t)k*N + j];
                acc += a*b; S += fabs(a)*fabs(b);
            }
            double tol = gemmTolerance(K, S, acc);
            double d   = fabs((double)hC[(size_t)i*N + j] - acc);
            double r   = (tol > 0.0) ? d/tol : 1.0e30;
            if (r > v.worst) { v.worst = r; v.wi = i; v.wj = j; }
        }
    if (!(v.worst <= 1.0)) v.ok = 0;
    return v;
}

static void report(const char *ds, const char *what, Verdict v)
{
    printf("    %-10s %-42s unwritten %7d  nonfinite %7d  worst %10.4g  %s\n",
           ds, what, v.unwritten, v.nonfinite, v.worst, v.ok ? "ACCEPT" : "reject");
}

// =============================================================================
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const int M = M_DIM, N = N_DIM, K = K_DIM;
    const size_t sA = (size_t)M*K, sB = (size_t)K*N, sC = (size_t)M*N;

    printf("=== Module 16 / Exercise 1 — the naive GEMM and its test ===\n");
    printf("C(%d x %d) = alpha * A(%d x %d) * B(%d x %d) + beta * C\n\n",
           M, N, M, K, K, N);

    // ---------------------------------------------------------- TODO 4
    // TODO 4: the launch configuration.
    //   The block shape is fixed at (BLOCK_X, BLOCK_Y) = (32, 8).
    //   Set gridX and gridY so that every element of C is covered exactly
    //   once, given the row/col assignment you chose in TODO 1. Neither M nor
    //   N is a multiple of 32 or of 8.
    //   Getting this inconsistent with TODO 1 is not a compile error and not a
    //   crash: it leaves part of C unwritten, and the harness reports exactly
    //   how many elements were never touched.
    // SOLUTION. x carries col, so gridDim.x is driven by N, gridDim.y by M.
    int gridX = (N + BLOCK_X - 1) / BLOCK_X;
    int gridY = (M + BLOCK_Y - 1) / BLOCK_Y;

    if (gridX <= 0 || gridY <= 0) { printf("Set TODO 4 first.\n"); return 0; }
    if (gemmTolerance(769, 1.0, 1.0) <= 0.0) { printf("Set TODO 5 first.\n"); return 0; }

    // ---------------------------------------------------------- data
    float *hA = (float*)malloc(sA*4), *hB = (float*)malloc(sB*4);
    float *hC = (float*)malloc(sC*4), *hPoison = (float*)malloc(sC*4);
    for (size_t i = 0; i < sC; ++i) hPoison[i] = UNWRITTEN;

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, sA*4)); CHECK(cudaMalloc(&dB, sB*4)); CHECK(cudaMalloc(&dC, sC*4));

    dim3 bl(BLOCK_X, BLOCK_Y), gr((unsigned)gridX, (unsigned)gridY);
    printf("  launch: grid (%u, %u) x block (%u, %u) = %.2f M threads for %.2f M elements\n\n",
           gr.x, gr.y, bl.x, bl.y,
           (double)gr.x*gr.y*bl.x*bl.y/1e6, (double)sC/1e6);

    // ---------------------------------------------------------- probe TODO 5
    printf("-- TODO 5 probed directly ----------------------------------------\n");
    const double u = ldexp(1.0, -24);
    const double bound769 = 769.0*u/(1.0 - 769.0*u);   // the theoretical worst case
    double t1 = gemmTolerance(769, 1.0, 1.0);
    double t2 = gemmTolerance(769, 2.0, 1.0);
    double t3 = gemmTolerance(1538, 1.0, 1.0);
    double t4 = gemmTolerance(769, 1.0, 0.0);
    int pA = fabs(t2 - 2.0*t1) <= 1e-9*t1*2.0;               // proportional to S
    int pB = (t3/t1 >= 1.5) && (t3/t1 <= 2.5);               // roughly linear in K
    int pC = (t1 >= 0.9*bound769) && (t1 <= 100.0*bound769);  // tight enough
    int pD = (t4 > 0.0);                                     // positive at ref = 0
    printf("    (a) proportional to S                 %s\n", pA ? "ok" : "FAILED");
    printf("    (b) doubling K gives %.3fx            %s\n", t3/t1, pB ? "ok" : "FAILED");
    printf("    (c) t(769, S=1)/theoretical = %8.3f  %s\n", t1/bound769, pC ? "ok" : "FAILED");
    printf("    (d) positive when ref == 0            %s\n", pD ? "ok" : "FAILED");
    const int shapeScore = pA + pB + pC + pD;
    printf("    shape score %d/4\n\n", shapeScore);

    // ---------------------------------------------------------- the two datasets
    int accepted = 0, rejected = 0;
    for (int ds = 0; ds < 2; ++ds) {
        const char *dsName = ds ? "zero-mean" : "positive";
        unsigned st = 2463534242u + (unsigned)ds;
        for (size_t i = 0; i < sA; ++i) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            float r = (float)((st >> 8) & 0xFFFFu)/65536.0f;
            hA[i] = ds ? (2.0f*r - 1.0f) : (0.5f + r);
        }
        for (size_t i = 0; i < sB; ++i) {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            float r = (float)((st >> 8) & 0xFFFFu)/65536.0f;
            hB[i] = ds ? (2.0f*r - 1.0f) : (0.5f + r);
        }
        CHECK(cudaMemcpy(dA, hA, sA*4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB, sB*4, cudaMemcpyHostToDevice));

        printf("-- dataset %s -------------------------------------------\n", dsName);

        CHECK(cudaMemcpy(dC, hPoison, sC*4, cudaMemcpyHostToDevice));
        gemmNaive<<<gr, bl>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hC, dC, sC*4, cudaMemcpyDeviceToHost));
        Verdict vy = judge(M, N, K, hA, hB, hC, 16);
        report(dsName, "YOUR kernel", vy);
        if (vy.ok) ++accepted;

        for (int d = 0; d < N_DEFECT; ++d) {
            CHECK(cudaMemcpy(dC, hPoison, sC*4, cudaMemcpyHostToDevice));
            gemmDefective<<<gr, bl>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC, d);
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(hC, dC, sC*4, cudaMemcpyDeviceToHost));
            Verdict v = judge(M, N, K, hA, hB, hC, 16);
            report(dsName, DEFECT_NAME[d], v);
            if (!v.ok) ++rejected;
        }
        printf("\n");
    }

    // ---------------------------------------------------------- the beta path
    printf("-- the alpha/beta path -------------------------------------------\n");
    float *hC0 = (float*)malloc(sC*4);
    for (size_t i = 0; i < sC; ++i) hC0[i] = (float)((int)(i % 17) - 8) * 0.25f;
    CHECK(cudaMemcpy(dC, hC0, sC*4, cudaMemcpyHostToDevice));
    gemmNaive<<<gr, bl>>>(M, N, K, 0.75f, dA, dB, -1.25f, dC);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hC, dC, sC*4, cudaMemcpyDeviceToHost));
    int betaBad = 0; double betaWorst = 0.0;
    for (int i = 0; i < M; i += 37)
        for (int j = 0; j < N; j += 53) {
            double acc = 0.0, S = 0.0;
            for (int k = 0; k < K; ++k) {
                double a = hA[(size_t)i*K+k], b = hB[(size_t)k*N+j];
                acc += a*b; S += fabs(a)*fabs(b);
            }
            double want = 0.75*acc - 1.25*(double)hC0[(size_t)i*N+j];
            double tol  = 0.75*gemmTolerance(K, S, acc) + 8.0*u*fabs(want);
            double r = fabs((double)hC[(size_t)i*N+j] - want)/tol;
            if (r > betaWorst) betaWorst = r;
        }
    betaBad = !(betaWorst <= 1.0);
    printf("    alpha=0.75, beta=-1.25, C prefilled : worst %.4g  %s\n",
           betaWorst, betaBad ? "reject" : "ACCEPT");
    printf("    (the beta == 0 case is already covered above: C was NaN on entry\n"
           "     and a kernel that reads it produces NaN everywhere)\n\n");

    // ---------------------------------------------------------- score
    const int wantAccept = 2, wantReject = 2*N_DEFECT;
    const int score = shapeScore + accepted + rejected + (betaBad ? 0 : 1);
    const int total = 4 + wantAccept + wantReject + 1;
    printf("  tolerance shape       %d/4\n", shapeScore);
    printf("  your kernel accepted  %d/%d datasets\n", accepted, wantAccept);
    printf("  defects rejected      %d/%d\n", rejected, wantReject);
    printf("  alpha/beta path       %d/1\n", betaBad ? 0 : 1);
    printf("  SCORE: %d/%d\n", score, total);
    printf("OVERALL: %s\n", score == total ? "PASS" : "FAIL");

    free(hA); free(hB); free(hC); free(hC0); free(hPoison);
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    CHECK(cudaDeviceReset());
    return score == total ? 0 : 1;
}
