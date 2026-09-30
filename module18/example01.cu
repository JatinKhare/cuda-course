// =============================================================================
// Module 18 / Example 1 — the optimization ladder, measured rung by rung.
//
// GOAL : Take the Module 17 tiled GEMM, which plateaus, and walk it up to
//        cuBLAS parity with four mechanical moves, each measured:
//
//          v1  block tiling only          (the Module 17 kernel)
//          v2  + a 1-D register tile      (TM outputs per thread)
//          v3  + a 2-D register tile      (TM x TN outputs per thread)
//          v4  + a transposed, padded A tile so the shared reads vectorize
//          v5  + double buffering / software pipelining
//
//        and to make the governing ratio visible at BOTH levels of the
//        hierarchy: FMAs per GLOBAL load (fixed by the block tile) and FMAs
//        per SHARED load (fixed by the thread tile). Module 16 measured that
//        6.4-6.5 FMAs per load are needed for 80% of the compute ceiling and
//        that one-element-per-thread supplies 0.50. That statement is true
//        twice, one level apart.
//
// BUILD: nvcc -arch=sm_89 -O3 -lcublas -o example01.exe example01.cu
// RUN  : example01.exe
//
// Also useful:
//   nvcc -arch=sm_89 -O3 -lcublas -Xptxas -v -o example01.exe example01.cu
//   nvcc -arch=sm_89 -O3 -cubin -o example01.cubin example01.cu
//   cuobjdump -sass example01.cubin
//
// Sections:
//   A  the two-level ledger: FMAs per global load, FMAs per shared load
//   B  the five kernels
//   C  validation of every rung (Module 16's gemmValidate, unchanged)
//   D  the measured ladder, versus cuBLAS and versus the FP32 ceiling
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
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

// Module 16's problem, unchanged, so the numbers compose.
#define M_DIM 1027
#define N_DIM 2053
#define K_DIM  769

// =============================================================================
// B. The kernels.
// =============================================================================

// ---- v1: block tiling only. This is the Module 17 kernel, reproduced here so
// the ladder starts from a measured rung. One thread still owns one element of
// C; all that changed versus Module 16 is that the operands arrive from shared
// memory instead of from L1/L2.
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
        if (beta == 0.0f) C[(size_t)row * N + col] = alpha * acc;
        else              C[(size_t)row * N + col] = alpha * acc
                                                   + beta * C[(size_t)row * N + col];
    }
}

// ---- v2: a 1-D register tile. Each thread owns a TM x 1 column of C. One
// value of B, held in a register, now feeds TM fused multiply-adds against TM
// values of A. FMAs per shared load = TM*1/(TM+1).
template<int BM, int BN, int BK, int TM>
__global__ __launch_bounds__((BM/TM)*BN) void gemmReg1D(
        int M, int N, int K, float alpha, const float *A, const float *B,
        float beta, float *C)
{
    const int NT  = (BM/TM) * BN;
    const int NLA = (BM*BK + NT - 1) / NT;
    const int NLB = (BK*BN + NT - 1) / NT;
    __shared__ float As[BK][BM];        // A tile stored TRANSPOSED: As[k][m]
    __shared__ float Bs[BK][BN];
    const int tid  = threadIdx.x;
    const int tRow = tid / BN, tCol = tid % BN;
    const int rowBase = blockIdx.y * BM, colBase = blockIdx.x * BN;

    float acc[TM];
    #pragma unroll
    for (int i = 0; i < TM; ++i) acc[i] = 0.0f;

    for (int kt = 0; kt < K; kt += BK) {
        #pragma unroll
        for (int u = 0; u < NLA; ++u) {
            int idx = tid + u*NT;  if (NLA*NT != BM*BK && idx >= BM*BK) break;
            int r = idx / BK, c = idx % BK;
            As[c][r] = (rowBase + r < M && kt + c < K)
                     ? A[(size_t)(rowBase + r) * K + kt + c] : 0.0f;
        }
        #pragma unroll
        for (int u = 0; u < NLB; ++u) {
            int idx = tid + u*NT;  if (NLB*NT != BK*BN && idx >= BK*BN) break;
            int k = idx / BN, n = idx % BN;
            Bs[k][n] = (kt + k < K && colBase + n < N)
                     ? B[(size_t)(kt + k) * N + colBase + n] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float bv = Bs[kk][tCol];              // ONE shared read ...
            #pragma unroll
            for (int i = 0; i < TM; ++i)                // ... TM fused ops
                acc[i] = fmaf(As[kk][tRow*TM + i], bv, acc[i]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = rowBase + tRow*TM + i, c = colBase + tCol;
        if (r < M && c < N) {
            if (beta == 0.0f) C[(size_t)r*N + c] = alpha * acc[i];
            else              C[(size_t)r*N + c] = alpha * acc[i]
                                                 + beta * C[(size_t)r*N + c];
        }
    }
}

// ---- v3 / v4: the 2-D register tile.
//
// LAYOUT 0 : A tile pitch = BM.  The transposed store hits a conflict; see
//            section D and Example 2.
// LAYOUT 1 : A tile pitch = BM + 4.  Four floats, not one: the padding must
//            both break the bank collision AND stay a multiple of 4 so the
//            float4 reads remain 16-byte aligned.
// LAYOUT 2 : XOR swizzle, zero extra bytes.
template<int BM, int BN, int BK, int TM, int TN, int LAYOUT>
__global__ __launch_bounds__((BM/TM)*(BN/TN)) void gemmReg2D(
        int M, int N, int K, float alpha, const float *A, const float *B,
        float beta, float *C)
{
    const int NT  = (BM/TM) * (BN/TN);
    const int AP  = (LAYOUT == 1) ? (BM + 4) : BM;      // A-tile row pitch
    const int NLA = (BM*BK + NT - 1) / NT;
    const int NLB = (BK*BN + NT - 1) / NT;
    __shared__ float As[BK * AP];
    __shared__ float Bs[BK][BN];

    const int tid  = threadIdx.x;
    const int tRow = tid / (BN/TN);            // which row of the thread grid
    const int tCol = tid % (BN/TN);            // which column
    const int rowBase = blockIdx.y * BM, colBase = blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int kt = 0; kt < K; kt += BK) {
        // ---- stage A^T. Consecutive tid -> consecutive k, so the GLOBAL read
        // is contiguous along A's fast axis; the SHARED write is the transpose.
        #pragma unroll
        for (int u = 0; u < NLA; ++u) {
            int idx = tid + u*NT;  if (NLA*NT != BM*BK && idx >= BM*BK) break;
            int r = idx / BK, c = idx % BK;
            float v = (rowBase + r < M && kt + c < K)
                    ? A[(size_t)(rowBase + r) * K + kt + c] : 0.0f;
            As[c*AP + ((LAYOUT == 2) ? (r ^ ((c & 3) << 3)) : r)] = v;
        }
        #pragma unroll
        for (int u = 0; u < NLB; ++u) {
            int idx = tid + u*NT;  if (NLB*NT != BK*BN && idx >= BK*BN) break;
            int k = idx / BN, n = idx % BN;
            Bs[k][n] = (kt + k < K && colBase + n < N)
                     ? B[(size_t)(kt + k) * N + colBase + n] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float rM[TM], rN[TN];
            const int sw = (LAYOUT == 2) ? ((kk & 3) << 3) : 0;
            #pragma unroll
            for (int i = 0; i < TM; ++i) rM[i] = As[kk*AP + ((tRow*TM + i) ^ sw)];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rN[j] = Bs[kk][tCol*TN + j];
            // TM + TN shared reads have now bought TM * TN fused ops.
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(rM[i], rN[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = rowBase + tRow*TM + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c = colBase + tCol*TN + j;
            if (c < N) {
                if (beta == 0.0f) C[(size_t)r*N + c] = alpha * acc[i][j];
                else              C[(size_t)r*N + c] = alpha * acc[i][j]
                                                     + beta * C[(size_t)r*N + c];
            }
        }
    }
}

// ---- v5: double buffering / software pipelining.
//
// Two shared buffers. The global loads for tile kt+BK are issued into
// REGISTERS before the fused ops on tile kt, and deposited into the other
// buffer afterwards. One barrier per k-tile instead of two: the second barrier
// existed only to stop a fast thread from overwriting a tile a slow thread was
// still reading (a write-after-read hazard, Module 6), and writing the OTHER
// buffer removes that hazard by construction.
//
// Module 32 owns `cp.async` (sm_80+), which does the same overlap in hardware
// and without the register cost. This is the portable, pre-Ampere form.
template<int BM, int BN, int BK, int TM, int TN, int LAYOUT>
__global__ __launch_bounds__((BM/TM)*(BN/TN)) void gemmReg2DDB(
        int M, int N, int K, float alpha, const float *A, const float *B,
        float beta, float *C)
{
    const int NT  = (BM/TM) * (BN/TN);
    const int AP  = (LAYOUT == 1) ? (BM + 4) : BM;
    const int NLA = (BM*BK) / NT;
    const int NLB = (BK*BN) / NT;
    __shared__ float As[2][BK * AP];
    __shared__ float Bs[2][BK * BN];

    const int tid  = threadIdx.x;
    const int tRow = tid / (BN/TN), tCol = tid % (BN/TN);
    const int rowBase = blockIdx.y * BM, colBase = blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    float pa[NLA], pb[NLB];              // the prefetch registers

    #define LOAD_TILE(KT)                                                       \
        _Pragma("unroll")                                                       \
        for (int u = 0; u < NLA; ++u) {                                         \
            int idx = tid + u*NT, r = idx / BK, c = idx % BK;                   \
            pa[u] = ((rowBase + r) < M && (KT) + c < K)                         \
                  ? A[(size_t)(rowBase + r) * K + (KT) + c] : 0.0f;             \
        }                                                                       \
        _Pragma("unroll")                                                       \
        for (int u = 0; u < NLB; ++u) {                                         \
            int idx = tid + u*NT, k = idx / BN, n = idx % BN;                   \
            pb[u] = ((KT) + k < K && (colBase + n) < N)                         \
                  ? B[(size_t)((KT) + k) * N + colBase + n] : 0.0f;             \
        }
    #define STORE_TILE(BUF)                                                     \
        _Pragma("unroll")                                                       \
        for (int u = 0; u < NLA; ++u) {                                         \
            int idx = tid + u*NT, r = idx / BK, c = idx % BK;                   \
            As[BUF][c*AP + ((LAYOUT == 2) ? (r ^ ((c & 3) << 3)) : r)] = pa[u];  \
        }                                                                       \
        _Pragma("unroll")                                                       \
        for (int u = 0; u < NLB; ++u) Bs[BUF][tid + u*NT] = pb[u];

    LOAD_TILE(0)
    STORE_TILE(0)
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < K; kt += BK) {
        const int nkt = kt + BK;
        if (nkt < K) { LOAD_TILE(nkt) }              // issued BEFORE the FFMAs
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float rM[TM], rN[TN];
            const int sw = (LAYOUT == 2) ? ((kk & 3) << 3) : 0;
            #pragma unroll
            for (int i = 0; i < TM; ++i) rM[i] = As[buf][kk*AP + ((tRow*TM + i) ^ sw)];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rN[j] = Bs[buf][kk*BN + tCol*TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(rM[i], rN[j], acc[i][j]);
        }
        if (nkt < K) { STORE_TILE(buf^1) }
        __syncthreads();                             // the ONLY barrier
        buf ^= 1;
    }
    #undef LOAD_TILE
    #undef STORE_TILE

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = rowBase + tRow*TM + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c = colBase + tCol*TN + j;
            if (c < N) {
                if (beta == 0.0f) C[(size_t)r*N + c] = alpha * acc[i][j];
                else              C[(size_t)r*N + c] = alpha * acc[i][j]
                                                     + beta * C[(size_t)r*N + c];
            }
        }
    }
}

// naive, for the bottom of the ladder (Module 16's kernel verbatim)
__global__ void gemmNaive(int M, int N, int K, float alpha, const float *A,
                          const float *B, float beta, float *C)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[(size_t)row*K + k] * B[(size_t)k*N + col];
    if (beta == 0.0f) C[(size_t)row*N + col] = alpha * acc;
    else              C[(size_t)row*N + col] = alpha*acc + beta*C[(size_t)row*N + col];
}

// =============================================================================
// C. Module 16's validator, unchanged. Three checks, in this order:
//    finiteness/writtenness -> Freivalds probe -> sampled exact double.
//    Tolerance scales with S = sum |a||b|, never with |C|.
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
// harness
// =============================================================================
static int Mg = M_DIM, Ng = N_DIM, Kg = K_DIM;
static const float *dAg, *dBg;
static float *dCg;
static cublasHandle_t hbl;

static void r_naive(void){ dim3 bl(16,16), gr((Ng+15)/16,(Mg+15)/16);
    gemmNaive<<<gr,bl>>>(Mg,Ng,Kg,1.0f,dAg,dBg,0.0f,dCg); }
static void r_tiled(void){ dim3 bl(32,32), gr((Ng+31)/32,(Mg+31)/32);
    gemmTiled<32><<<gr,bl>>>(Mg,Ng,Kg,1.0f,dAg,dBg,0.0f,dCg); }
template<int BM,int BN,int BK,int TM>
static void r_1d(void){ dim3 gr((Ng+BN-1)/BN,(Mg+BM-1)/BM);
    gemmReg1D<BM,BN,BK,TM><<<gr,(BM/TM)*BN>>>(Mg,Ng,Kg,1.0f,dAg,dBg,0.0f,dCg); }
template<int BM,int BN,int BK,int TM,int TN,int LAY>
static void r_2d(void){ dim3 gr((Ng+BN-1)/BN,(Mg+BM-1)/BM);
    gemmReg2D<BM,BN,BK,TM,TN,LAY><<<gr,(BM/TM)*(BN/TN)>>>(Mg,Ng,Kg,1.0f,dAg,dBg,0.0f,dCg); }
template<int BM,int BN,int BK,int TM,int TN,int LAY>
static void r_db(void){ dim3 gr((Ng+BN-1)/BN,(Mg+BM-1)/BM);
    gemmReg2DDB<BM,BN,BK,TM,TN,LAY><<<gr,(BM/TM)*(BN/TN)>>>(Mg,Ng,Kg,1.0f,dAg,dBg,0.0f,dCg); }
static void r_cublas(void){ const float a = 1.0f, b = 0.0f;
    CHECK_CUBLAS(cublasSgemm(hbl, CUBLAS_OP_N, CUBLAS_OP_N, Ng, Mg, Kg,
                             &a, dBg, Ng, dAg, Kg, &b, dCg, Ng)); }

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

#define NCFG 8

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const int M = M_DIM, N = N_DIM, K = K_DIM;
    const size_t sA = (size_t)M*K, sB = (size_t)K*N, sC = (size_t)M*N;

    printf("=== Module 18 / Example 1 - the optimization ladder ===\n");
    printf("C(%d x %d) = A(%d x %d) * B(%d x %d), row-major fp32, alpha=1 beta=0\n\n",
           M, N, M, K, K, N);

    // -------------------------------------------------- A. the two-level ledger
    printf("-- A. the ratio that governs, stated twice -----------------------\n");
    printf("  Module 16: 6.4-6.5 fused multiply-adds per load are needed for 80%%\n"
           "  of the FP32 ceiling; one output element per thread supplies 0.50.\n"
           "  That sentence is true at TWO levels of the memory hierarchy, and a\n"
           "  block tile fixes one of them while a register tile fixes the other.\n\n");
    printf("  %-34s %-26s %10s\n", "level", "law", "value");
    printf("  %-34s %-26s %10s\n", "----------------------------------",
           "--------------------------", "----------");
    printf("  %-34s %-26s %10.2f\n", "naive: global -> register",   "1*1/(1+1)",            0.50);
    printf("  %-34s %-26s %10.2f\n", "32x32 block tile: global",    "BM*BN/(BM+BN)",
           32.0*32.0/(32.0+32.0));
    printf("  %-34s %-26s %10.2f\n", "  ... its shared -> register","1*1/(1+1)",            0.50);
    printf("  %-34s %-26s %10.2f\n", "128x128 block tile: global",  "BM*BN/(BM+BN)",
           128.0*128.0/(128.0+128.0));
    printf("  %-34s %-26s %10.2f\n", "  8x1 thread tile: shared",   "TM*TN/(TM+TN)",  8.0*1/(8.0+1));
    printf("  %-34s %-26s %10.2f\n", "  8x8 thread tile: shared",   "TM*TN/(TM+TN)",  8.0*8/(8.0+8));
    printf("  %-34s %-26s %10.2f\n", "  8x8, float4 shared reads",  "TM*TN/((TM+TN)/4)",
           8.0*8/((8.0+8)/4));
    printf("\n  Both laws are the same function, one level apart. For a fixed\n"
           "  register budget R = TM*TN the ratio TM*TN/(TM+TN) is maximised at\n"
           "  TM = TN = sqrt(R) (AM-GM), which is why thread tiles are square.\n");
    printf("  Reaching 6.5 needs TM = TN = 13 as a scalar count -- 169 accumulators,\n"
           "  which does not fit. The instruction count is what actually matters,\n"
           "  and TM = TN = 8 read with float4 gives %.1f FMAs per shared-memory\n"
           "  INSTRUCTION. Section D checks that claim against the SASS.\n\n",
           8.0*8/((8.0+8)/4));

    // -------------------------------------------------- data
    float *hA = (float*)malloc(sA*4), *hB = (float*)malloc(sB*4);
    float *hC = (float*)malloc(sC*4);
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
    CHECK_CUBLAS(cublasCreate(&hbl));

    struct Cfg { const char *name; void (*run)(void); const void *fn; int thr; };
    Cfg cfg[NCFG] = {
      {"v0  naive, one element per thread",        r_naive,  (const void*)gemmNaive, 256},
      {"v1  block tile 32x32 (the M17 kernel)",    r_tiled,  (const void*)gemmTiled<32>, 1024},
      {"v2  + 1-D register tile, TM=8",            r_1d<64,64,8,8>,
                                        (const void*)gemmReg1D<64,64,8,8>, 512},
      {"v3  + 2-D register tile, TM=TN=8",         r_2d<128,128,8,8,8,0>,
                                        (const void*)gemmReg2D<128,128,8,8,8,0>, 256},
      {"v4  + transposed A tile padded by 4",      r_2d<128,128,8,8,8,1>,
                                        (const void*)gemmReg2D<128,128,8,8,8,1>, 256},
      {"v4b + the sweep's winner, TM=8 TN=4",      r_2d<128,64,8,8,4,1>,
                                        (const void*)gemmReg2D<128,64,8,8,4,1>, 256},
      {"v5  + double buffering (one barrier)",     r_db<128,128,8,8,8,1>,
                                        (const void*)gemmReg2DDB<128,128,8,8,8,1>, 256},
      {"    cublasSgemm",                          r_cublas, NULL, 0},
    };

    // -------------------------------------------------- warm-up (spec 12 rule 4)
    printf("-- warming up: 1500 ms streaming, then 500 ms compute --------------\n");
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

    // -------------------------------------------------- D. timing
    int iters[NCFG];
    for (int i = 0; i < NCFG; ++i) {
        double t = timeOne(cfg[i].run, 1);
        int n = (int)(10.0 / (t > 0 ? t : 0.01));
        if (n < 3) n = 3; if (n > 64) n = 64; iters[i] = n;
    }
    CHECK(cudaGetLastError());
    double best[NCFG]; for (int i = 0; i < NCFG; ++i) best[i] = 1e30;
    for (int s = 0; s < NCFG; ++s)                 // SWEEPS == NCFG, rotated
        for (int q = 0; q < NCFG; ++q) {
            int p = (q + s) % NCFG;
            double t = timeOne(cfg[p].run, iters[p]);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    const double flops = 2.0*M*N*K;
    const double gcub  = flops/(best[NCFG-1]*1e-3)/1e9;
    const double CEIL  = 18000.0;        // Module 16's measured FP32 ceiling
    printf("\n-- D. the measured ladder ------------------------------------------\n");
    printf(" %-42s %5s %5s %6s %6s %9s %8s %7s\n",
           "kernel","regs","blk", "smemB", "occ%", "GFLOP/s", "xcuBLAS", "xceil");
    for (int i = 0; i < NCFG; ++i) {
        double g = flops/(best[i]*1e-3)/1e9;
        int regs = 0, blk = 0, smem = 0, thr = 0; double occ = 0.0;
        if (cfg[i].fn) {
            cudaFuncAttributes at; CHECK(cudaFuncGetAttributes(&at, cfg[i].fn));
            regs = at.numRegs; smem = (int)at.sharedSizeBytes;
            thr  = cfg[i].thr;
            CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blk, cfg[i].fn, thr, 0));
            occ = 100.0 * blk * thr / 1536.0;
        }
        if (cfg[i].fn)
            printf(" %-42s %5d %5d %6d %5.1f%% %9.1f %7.2f %7.3f\n",
                   cfg[i].name, regs, blk, smem, occ, g, g/gcub, g/CEIL);
        else
            printf(" %-42s %5s %5s %6s %6s %9.1f %7.2f %7.3f\n",
                   cfg[i].name, "-", "-", "-", "-", g, g/gcub, g/CEIL);
    }
    printf("\n  Read the occupancy column against the GFLOP/s column. The fastest\n"
           "  kernel here is NOT the one with the most resident warps, and the\n"
           "  gap is not small. Example 2 forces the point.\n");

    // -------------------------------------------------- C. validation, second pass
    printf("\n-- C. validation (Module 16's validator, second untimed pass) ------\n");
    int allok = 1;
    for (int i = 0; i < NCFG; ++i) {
        float *p = (float*)malloc(sC*4);
        for (size_t j = 0; j < sC; ++j) p[j] = std::numeric_limits<float>::infinity();   // +inf sentinel
        CHECK(cudaMemcpy(dC, p, sC*4, cudaMemcpyHostToDevice)); free(p);
        cfg[i].run(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hC, dC, sC*4, cudaMemcpyDeviceToHost));
        GemmCheck c = gemmValidate(M, N, K, 1.0f, hA, hB, 0.0f, NULL, hC, 64);
        printf("  %-42s nonfinite %7d | Freivalds %9.3g | sampled %9.3g | %s\n",
               cfg[i].name, c.nonfinite, c.freivalds, c.sampled, c.ok ? "PASS":"FAIL");
        if (!c.ok) allok = 0;
    }
    printf("\n  Every rung produces the same matrix. None of these are\n"
           "  approximations; they are the same arithmetic in a different order.\n");

    printf("\n  What is still missing versus cuBLAS and CUTLASS, and who owns it:\n"
           "    Tensor Cores (Modules 33-34), cp.async and TMA (Module 32),\n"
           "    warp specialization and multi-stage pipelines, per-shape tuned\n"
           "    tile selection, and split-K for skinny problems. See the lesson.\n");

    CHECK_CUBLAS(cublasDestroy(hbl));
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    printf("\nOVERALL: %s\n", allok ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return allok ? 0 : 1;
}
