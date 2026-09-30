// =============================================================================
// Module 17 / Example 1 — the tiled GEMM: structure, boundaries, tile sizes.
//
// GOAL : (a) the canonical shared-memory tiled GEMM, correct at a size that is
//            a multiple of nothing;
//        (b) the generalised rectangular-tile form, whose load mapping is
//            genuinely different from its compute mapping;
//        (c) the tile-shape sweep, timed back-to-back against Module 16's
//            naive kernel and against cuBLAS;
//        (d) the FMAs-per-global-load ledger, which is the hand-off to
//            Module 18.
//
// BUILD: nvcc -arch=sm_89 -O3 -lcublas -o example01.exe example01.cu
// RUN  : example01.exe
//
// Sections:
//   A  the reuse ledger: global loads as a function of the tile shape
//   B  the kernels
//   C  correctness at an awkward size, with Module 16's validator
//   D  the sweep
//   E  the same three numbers on Module 16's exact problem shape
//   F  what tiling did not fix
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

// The module's problem shape. M % 8/16/32 = 3/11/11, N % 8/16/32 = 5/5/5,
// K % 8/16/32 = 7/7/7 -- every tile size used here leaves a partial tile on
// every one of the three axes, and no two of M, N, K are equal.
#define M_DIM 1035
#define N_DIM 1541
#define K_DIM 1063

// Module 16's shape, kept so section E is directly comparable to its table.
#define M16_M 1027
#define M16_N 2053
#define M16_K  769

static unsigned lcg_state = 1u;
static float rnd01(void) { lcg_state = lcg_state * 1664525u + 1013904223u;
                           return (float)((lcg_state >> 8) & 0xFFFFu) / 65536.0f; }

// =============================================================================
// B. The kernels.
// =============================================================================

// --- Module 16's kernel, as the control. -------------------------------------
__global__ void gemmNaive(int M, int N, int K,
                          float alpha, const float *A, const float *B,
                          float beta, float *C)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc = fmaf(A[(size_t)row * K + k], B[(size_t)k * N + col], acc);
    if (beta == 0.0f) C[(size_t)row * N + col] = alpha * acc;
    else              C[(size_t)row * N + col] = alpha * acc
                                               + beta * C[(size_t)row * N + col];
}

// --- the canonical square tiled kernel ---------------------------------------
//
// A block of T x T threads owns a T x T tile of C and marches it along k in
// steps of T. Block shape is (T, T): threadIdx.x -> the N axis, so a warp is
// 32 consecutive columns of C when T = 32 and spans two tile rows when T = 16.
// Module 16 measured x -> col as the good mapping and this keeps it.
//
// Per k-step the block stages As = T x T of A and Bs = T x T of B, then each
// thread adds T terms to an accumulator that lives in a REGISTER across the
// whole tile loop. The register is the only place a partial dot product can
// live: shared memory holds a tile, and a tile is never a whole row of A.
//
// THREE index mappings appear here and they are all different:
//   compute : thread (tx,ty) owns C[row0+ty][col0+tx]
//   A load  : thread (tx,ty) fetches A[row0+ty][k0+tx]   -- tx walks k
//   B load  : thread (tx,ty) fetches B[k0+ty][col0+tx]   -- ty walks k
// The B load is the one people get wrong. It is not "the same as A with the
// indices swapped": in A, threadIdx.x indexes the contraction dimension; in B,
// threadIdx.x indexes the output column. Both are chosen so consecutive tx
// read consecutive global addresses, which is what keeps both loads coalesced.
//
// PAD is the shared-memory row pitch padding. Example 2 measures what it does.
template <int T, int PAD>
__global__ void gemmTiled(int M, int N, int K,
                          float alpha, const float * __restrict__ A,
                          const float * __restrict__ B,
                          float beta, float *C)
{
    __shared__ float As[T][T + PAD];
    __shared__ float Bs[T][T + PAD];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty;          // COMPUTE mapping
    const int col = blockIdx.x * T + tx;

    float acc = 0.0f;                             // lives across the whole loop

    const int nTiles = (K + T - 1) / T;           // ceil, not K/T
    for (int t = 0; t < nTiles; ++t) {
        const int aCol = t * T + tx;              // A LOAD mapping
        const int bRow = t * T + ty;              // B LOAD mapping

        // Out-of-range cells are zero-filled, not skipped. A zero contributes
        // nothing to the dot product, so a zero-padded tile gives the same
        // answer as a shorter one -- and unlike "skip it", every thread still
        // reaches both barriers.
        As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;

        __syncthreads();            // RAW: every cell written before any is read

        #pragma unroll
        for (int k = 0; k < T; ++k)
            acc = fmaf(As[ty][k], Bs[k][tx], acc);

        __syncthreads();            // WAR: every cell read before any is rewritten
    }

    if (row < M && col < N) {
        if (beta == 0.0f) C[(size_t)row * N + col] = alpha * acc;
        else              C[(size_t)row * N + col] = alpha * acc
                                                   + beta * C[(size_t)row * N + col];
    }
}

// --- the general rectangular tiled kernel ------------------------------------
//
// BM x BN outputs, contraction depth BK, BM*BN threads. When BK != BN the
// number of A-tile cells is no longer the number of threads, so the load can
// no longer be one cell per thread: it becomes Module 6's flat strided
// cooperative loop, with BK (resp. BN) on the fast axis so consecutive tid
// read consecutive global addresses.
template <int BM, int BN, int BK, int PAD>
__global__ void gemmTiledRect(int M, int N, int K,
                              float alpha, const float * __restrict__ A,
                              const float * __restrict__ B,
                              float beta, float *C)
{
    __shared__ float As[BM][BK + PAD];
    __shared__ float Bs[BK][BN + PAD];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid  = ty * BN + tx;
    const int nthr = BM * BN;

    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int row  = row0 + ty, col = col0 + tx;

    float acc = 0.0f;
    const int nTiles = (K + BK - 1) / BK;
    for (int t = 0; t < nTiles; ++t) {
        const int k0 = t * BK;
        #pragma unroll
        for (int i = tid; i < BM * BK; i += nthr) {
            const int r = i / BK, c = i - r * BK;
            const int gr = row0 + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
        }
        #pragma unroll
        for (int i = tid; i < BK * BN; i += nthr) {
            const int r = i / BN, c = i - r * BN;
            const int gr = k0 + r, gc = col0 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k)
            acc = fmaf(As[ty][k], Bs[k][tx], acc);
        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta == 0.0f) C[(size_t)row * N + col] = alpha * acc;
        else              C[(size_t)row * N + col] = alpha * acc
                                                   + beta * C[(size_t)row * N + col];
    }
}

// =============================================================================
// C. Module 16's validator, reused unchanged. Do NOT replace this with a
//    relative-to-|C| tolerance; Module 16 measured that rule rejecting a
//    correct GEMM.
// =============================================================================
typedef struct {
    int    ok;
    int    nonfinite;
    double freivalds;
    int    freivaldsRow;
    double sampled;
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

    const double u      = ldexp(1.0, -24);
    const double gammaK = (double)K * u / (1.0 - (double)K * u);

    // ---- check 1: finiteness, every element ----
    for (size_t i = 0; i < (size_t)M * N; ++i)
        if (!isfinite(hC[i])) ++r.nonfinite;
    if (r.nonfinite) r.ok = 0;

    // ---- check 2: Freivalds probe ----
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

// =============================================================================
// warm-up kernels. Spec 12 rule 4: 1500 ms of streaming (ramps the memory
// P-state) then 500 ms of compute (ramps the SM clock without letting the
// memory clock fall back). A GEMM sweep needs both ceilings up.
// =============================================================================
__global__ void streamWarm(const float4 *in, float4 *out, size_t n4)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 s0 = make_float4(0.0f,0.0f,0.0f,0.0f);
    for (; i < n4; i += stride) { float4 v = in[i];
        s0.x += v.x; s0.y += v.y; s0.z += v.z; s0.w += v.w; }
    if (s0.x + s0.y + s0.z + s0.w == 1.2345e30f) out[0] = s0;
}
__global__ void ffmaWarm(float *out, int iters)
{
    float a0=threadIdx.x,a1=a0+1,a2=a0+2,a3=a0+3,a4=a0+4,a5=a0+5,a6=a0+6,a7=a0+7;
    const float b = 1.0000001f, c = 0.9999999f;
    for (int i = 0; i < iters; ++i) {
        a0=fmaf(a0,b,c); a1=fmaf(a1,b,c); a2=fmaf(a2,b,c); a3=fmaf(a3,b,c);
        a4=fmaf(a4,b,c); a5=fmaf(a5,b,c); a6=fmaf(a6,b,c); a7=fmaf(a7,b,c);
    }
    float s=a0+a1+a2+a3+a4+a5+a6+a7;
    if (s == 1.2345e30f) out[0] = s;
}

// =============================================================================
// the configuration table
// =============================================================================
typedef struct {
    const char *name;
    int BM, BN, BK;            // BM==0 marks naive; BM<0 marks cuBLAS
} Cfg;

static const Cfg CFG[] = {
    { "naive (Module 16)",       0,  0,  0 },
    { "square  8x 8",            8,  8,  8 },
    { "square 16x16",           16, 16, 16 },
    { "square 32x32",           32, 32, 32 },
    { "rect 16x16 BK=16",       16, 16, 16 },
    { "rect 16x16 BK= 8",       16, 16,  8 },
    { "rect 16x16 BK=32",       16, 16, 32 },
    { "rect 32x16 BK=16",       32, 16, 16 },
    { "rect 16x32 BK=16",       16, 32, 16 },
    { "rect 64x16 BK=16",       64, 16, 16 },
    { "rect 32x32 BK=16",       32, 32, 16 },
    { "cublasSgemm",            -1, -1, -1 },
};
enum { NCFG = (int)(sizeof(CFG)/sizeof(CFG[0])) };
enum { FIRST_RECT = 4 };       // configs [1,FIRST_RECT) are the square kernel

typedef struct {
    int M, N, K; float alpha, beta;
    const float *dA, *dB; float *dC;
    cublasHandle_t h;
} Ctx;

#define LAUNCH_SQ(t)                                                          \
    do { dim3 bl((t),(t));                                                    \
         dim3 gr((unsigned)((c->N + (t) - 1)/(t)),                            \
                 (unsigned)((c->M + (t) - 1)/(t)));                           \
         gemmTiled<t,0><<<gr,bl>>>(c->M,c->N,c->K,c->alpha,                   \
                                   c->dA,c->dB,c->beta,c->dC);                \
    } while (0)

#define LAUNCH_RE(bm,bn,bk)                                                   \
    do { dim3 bl((bn),(bm));                                                  \
         dim3 gr((unsigned)((c->N + (bn) - 1)/(bn)),                          \
                 (unsigned)((c->M + (bm) - 1)/(bm)));                         \
         gemmTiledRect<bm,bn,bk,0><<<gr,bl>>>(c->M,c->N,c->K,c->alpha,        \
                                              c->dA,c->dB,c->beta,c->dC);     \
    } while (0)

static void launchCfg(int id, Ctx *c)
{
    switch (id) {
      case 0: { dim3 bl(32,8);
                dim3 gr((unsigned)((c->N+31)/32),(unsigned)((c->M+7)/8));
                gemmNaive<<<gr,bl>>>(c->M,c->N,c->K,c->alpha,c->dA,c->dB,
                                     c->beta,c->dC); } break;
      case  1: LAUNCH_SQ( 8); break;
      case  2: LAUNCH_SQ(16); break;
      case  3: LAUNCH_SQ(32); break;
      case  4: LAUNCH_RE(16,16,16); break;
      case  5: LAUNCH_RE(16,16, 8); break;
      case  6: LAUNCH_RE(16,16,32); break;
      case  7: LAUNCH_RE(32,16,16); break;
      case  8: LAUNCH_RE(16,32,16); break;
      case  9: LAUNCH_RE(64,16,16); break;
      case 10: LAUNCH_RE(32,32,16); break;
      case 11: CHECK_CUBLAS(cublasSgemm(c->h, CUBLAS_OP_N, CUBLAS_OP_N,
                                        c->N, c->M, c->K,
                                        &c->alpha, c->dB, c->N, c->dA, c->K,
                                        &c->beta,  c->dC, c->N)); break;
      default: break;
    }
}

// ---- timing --------------------------------------------------------------
static int calibrate(int id, Ctx *c)
{
    cudaEvent_t e0,e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0)); launchCfg(id,c); CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms,e0,e1));
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    int it = (int)(10.0 / (ms > 0.0f ? ms : 0.01f));
    if (it < 1) it = 1;
    if (it > 128) it = 128;
    return it;
}
static double timeOnce(int id, Ctx *c, int iters)
{
    cudaEvent_t e0,e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) launchCfg(id,c);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms,e0,e1));
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    return ms / iters;
}

// =============================================================================
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    const int M = M_DIM, N = N_DIM, K = K_DIM;
    const size_t sA=(size_t)M*K, sB=(size_t)K*N, sC=(size_t)M*N;
    const double flops = 2.0*(double)M*N*K;

    printf("=== Module 17 / Example 1 - the tiled GEMM ===\n");
    printf("C(%d x %d) = alpha*A(%d x %d)*B(%d x %d) + beta*C, row-major fp32\n",
           M, N, M, K, K, N);
    printf("M %% 8/16/32 = %d/%d/%d   N %% 8/16/32 = %d/%d/%d   K %% 8/16/32 = %d/%d/%d\n",
           M%8,M%16,M%32, N%8,N%16,N%32, K%8,K%16,K%32);
    printf("Every tile size below leaves a partial tile on all three axes.\n\n");

    // ------------------------------------------------------------------ A
    printf("-- A. the reuse ledger -------------------------------------------\n");
    printf("  Naive: 2 global loads per FMA, 2*M*N*K = %.3f G load instructions.\n",
           2.0*(double)M*N*K/1e9);
    printf("  A block staging BM x BK of A and BK x BN of B performs\n");
    printf("     BM*BN*BK FMAs  against  BK*(BM + BN) global loads,\n");
    printf("  so FMAs per global load = BM*BN / (BM + BN) = 1/(1/BM + 1/BN):\n");
    printf("  half the harmonic mean of the OUTPUT tile dimensions. BK cancels.\n");
    printf("  BK buys shared memory and barrier amortisation, not reuse.\n\n");
    printf("  %-20s %8s %8s %9s %10s %10s %7s\n",
           "tile", "threads", "smem B", "charged", "FMA/gld", "blocks/SM", "occ");
    for (int i = 1; i < NCFG-1; ++i) {
        const int BM=CFG[i].BM, BN=CFG[i].BN, BK=CFG[i].BK;
        const int thr = BM*BN;
        const int smem = 4*BK*(BM+BN);
        const int charged = ((smem + 1024 + 127)/128)*128;
        const double fpl = 1.0/(1.0/BM + 1.0/BN);
        int bsm = 102400/charged;
        if (bsm > 1536/thr) bsm = 1536/thr;
        if (bsm > 24) bsm = 24;
        printf("  %-20s %8d %8d %9d %10.2f %10d %6.1f%%\n",
               CFG[i].name, thr, smem, charged, fpl, bsm,
               100.0*(double)(bsm*thr)/1536.0);
    }
    printf("\n  Module 16 measured that ~6.4-6.5 FMAs per global load are needed to\n"
           "  reach 80%% of the FP32 ceiling. Every tile from 16x16 up clears it.\n"
           "  Section F explains why clearing it is not enough.\n"
           "  Note the ceiling on BM = BN = T: T*T threads is capped at 1024, so\n"
           "  T <= 32 and FMAs per global load <= 16 for ANY square tile. One\n"
           "  output per thread is what binds it -- which is Module 18's opening.\n\n");

    // ------------------------------------------------------------------ data
    float *hA=(float*)malloc(sA*4), *hB=(float*)malloc(sB*4);
    float *hC=(float*)malloc(sC*4), *hC0=(float*)malloc(sC*4);
    float *poison=(float*)malloc(sC*4);
    lcg_state = 1u;
    for (size_t i=0;i<sA;++i) hA[i] = 0.5f + rnd01();
    for (size_t i=0;i<sB;++i) hB[i] = 0.5f + rnd01();
    for (size_t i=0;i<sC;++i) hC0[i] = (float)((int)(i%17)-8)*0.25f;
    // +inf by bit pattern: the INFINITY macro warns under MSVC's headers.
    { union { unsigned u; float f; } inf; inf.u = 0x7F800000u;
      for (size_t i=0;i<sC;++i) poison[i] = inf.f; }

    float *dA,*dB,*dC;
    CHECK(cudaMalloc(&dA,sA*4)); CHECK(cudaMalloc(&dB,sB*4)); CHECK(cudaMalloc(&dC,sC*4));
    CHECK(cudaMemcpy(dA,hA,sA*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB,hB,sB*4,cudaMemcpyHostToDevice));

    cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
    Ctx ctx; ctx.M=M; ctx.N=N; ctx.K=K; ctx.alpha=1.0f; ctx.beta=0.0f;
    ctx.dA=dA; ctx.dB=dB; ctx.dC=dC; ctx.h=h;

    // ------------------------------------------------------------------ C
    printf("-- C. correctness at an awkward size ------------------------------\n");
    printf("  C is prefilled with +inf before every beta==0 launch, so a never-\n"
           "  written element and a kernel that reads C when beta==0 both show up.\n\n");
    int allOk = 1;
    for (int i = 0; i < NCFG; ++i) {
        CHECK(cudaMemcpy(dC,poison,sC*4,cudaMemcpyHostToDevice));
        ctx.alpha=1.0f; ctx.beta=0.0f; launchCfg(i,&ctx);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hC,dC,sC*4,cudaMemcpyDeviceToHost));
        GemmCheck c1 = gemmValidate(M,N,K,1.0f,hA,hB,0.0f,NULL,hC,64);

        CHECK(cudaMemcpy(dC,hC0,sC*4,cudaMemcpyHostToDevice));
        ctx.alpha=0.75f; ctx.beta=-1.25f; launchCfg(i,&ctx);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hC,dC,sC*4,cudaMemcpyDeviceToHost));
        GemmCheck c2 = gemmValidate(M,N,K,0.75f,hA,hB,-1.25f,hC0,hC,64);

        printf("  %-20s nonfin %7d | Freiv %8.4g | samp %8.4g | a/b %8.4g | %s\n",
               CFG[i].name, c1.nonfinite, c1.freivalds, c1.sampled, c2.sampled,
               (c1.ok && c2.ok) ? "PASS" : "FAIL");
        if (!(c1.ok && c2.ok)) allOk = 0;
    }
    ctx.alpha=1.0f; ctx.beta=0.0f;
    printf("\n  Headroom, not just PASS: every sampled figure is a fraction of the\n"
           "  gamma_K*S budget, and gamma_K = %.3e at K = %d. Note that every\n"
           "  tiled kernel reports the SAME error as the naive one: zero-padding\n"
           "  the partial tile adds exact zeros, and reordering a sum of positive\n"
           "  terms into tiles does not change it here.\n\n",
           (double)K*ldexp(1.0,-24)/(1.0-(double)K*ldexp(1.0,-24)), K);

    // ------------------------------------------------------------------ warm
    printf("-- D. the sweep ---------------------------------------------------\n");
    printf("  warming: 1500 ms stream (memory P-state) then 500 ms FFMA (SM clock)\n");
    { const size_t SB = 256u*1024u*1024u;
      float4 *sIn,*sOut; CHECK(cudaMalloc(&sIn,SB)); CHECK(cudaMalloc(&sOut,SB));
      CHECK(cudaMemset(sIn,0x3c,SB));
      int nsm=0; CHECK(cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0));
      float *fOut; CHECK(cudaMalloc(&fOut,4));
      cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0.0f; CHECK(cudaEventRecord(w0));
      while (el < 1500.0f) { streamWarm<<<nsm*12,256>>>(sIn,sOut,SB/sizeof(float4));
          CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
          CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      el=0.0f; CHECK(cudaEventRecord(w0));
      while (el < 500.0f) { ffmaWarm<<<nsm*6,256>>>(fOut,20000);
          CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
          CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
      CHECK(cudaFree(sIn)); CHECK(cudaFree(sOut)); CHECK(cudaFree(fOut));
      CHECK(cudaGetLastError()); }

    double best[NCFG];
    int iters[NCFG];
    for (int i = 0; i < NCFG; ++i) { iters[i] = calibrate(i,&ctx); best[i] = 1e30; }
    for (int s = 0; s < NCFG; ++s)                 // SWEEPS >= NCFG
        for (int q = 0; q < NCFG; ++q) {
            const int p = (q + s) % NCFG;          // spec 12 rule 9: rotate
            double t = timeOnce(p, &ctx, iters[p]);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    const double gcublas = flops/(best[NCFG-1]*1e-3)/1e9;
    int bestTile = 1;
    for (int i = 2; i < NCFG-1; ++i) if (best[i] < best[bestTile]) bestTile = i;

    printf("\n  %-20s %9s %11s %9s %10s\n",
           "config", "ms", "GFLOP/s", "x naive", "% cuBLAS");
    for (int i = 0; i < NCFG; ++i) {
        const double g = flops/(best[i]*1e-3)/1e9;
        printf("  %-20s %9.4f %11.1f %9.2f %9.1f%%\n",
               CFG[i].name, best[i], g, best[0]/best[i], 100.0*g/gcublas);
    }
    printf("\n  best tile = %s at %.1f GFLOP/s = %.2fx naive = %.1f%% of cuBLAS\n",
           CFG[bestTile].name, flops/(best[bestTile]*1e-3)/1e9,
           best[0]/best[bestTile], 100.0*flops/(best[bestTile]*1e-3)/1e9/gcublas);
    printf("  square 16x16 vs rect 16x16 BK=16 (identical tiles, different loader)\n"
           "  = %.3fx. The difference is the integer division and the strided\n"
           "  loop the general loader needs and the square one does not.\n",
           best[4]/best[2]);

    // ------------------------------------------------------------------ E
    printf("\n-- E. the same three numbers on Module 16's shape (%dx%dx%d) ---\n",
           M16_M, M16_N, M16_K);
    {
        const int m=M16_M, n=M16_N, k=M16_K;
        const size_t a=(size_t)m*k, b=(size_t)k*n, cc=(size_t)m*n;
        const double fl = 2.0*(double)m*n*k;
        float *dA2,*dB2,*dC2;
        CHECK(cudaMalloc(&dA2,a*4)); CHECK(cudaMalloc(&dB2,b*4)); CHECK(cudaMalloc(&dC2,cc*4));
        CHECK(cudaMemset(dA2,0x3c,a*4)); CHECK(cudaMemset(dB2,0x3c,b*4));
        Ctx c2; c2.M=m; c2.N=n; c2.K=k; c2.alpha=1.0f; c2.beta=0.0f;
        c2.dA=dA2; c2.dB=dB2; c2.dC=dC2; c2.h=h;
        const int ids[3] = { 0, bestTile, NCFG-1 };
        double b3[3] = {1e30,1e30,1e30};
        int it3[3];
        for (int i=0;i<3;++i) it3[i]=calibrate(ids[i],&c2);
        for (int s=0;s<3;++s) for (int q=0;q<3;++q) {
            int p=(q+s)%3; double t=timeOnce(ids[p],&c2,it3[p]); if (t<b3[p]) b3[p]=t; }
        CHECK(cudaGetLastError());
        printf("  %-20s %9s %11s %10s\n", "config", "ms", "GFLOP/s", "% cuBLAS");
        for (int i=0;i<3;++i)
            printf("  %-20s %9.4f %11.1f %9.1f%%\n", CFG[ids[i]].name, b3[i],
                   fl/(b3[i]*1e-3)/1e9, 100.0*b3[2]/b3[i]);
        printf("  Module 16 measured naive at 1275-1348 GFLOP/s and cuBLAS at\n"
               "  8150-8705 GFLOP/s on this exact shape.\n");
        CHECK(cudaFree(dA2)); CHECK(cudaFree(dB2)); CHECK(cudaFree(dC2));
    }

    // ------------------------------------------------------------------ F
    printf("\n-- F. what tiling did not fix -------------------------------------\n");
    {
        const int BM=CFG[bestTile].BM, BN=CFG[bestTile].BN, BK=CFG[bestTile].BK;
        const double gl = 1.0/(1.0/BM + 1.0/BN);
        printf("  best tile %d x %d, BK=%d:\n", BM, BN, BK);
        printf("    FMAs per GLOBAL load instruction : %6.2f  (naive 0.50, needed 6.5)\n", gl);
        printf("    FMAs per SHARED load instruction : %6.2f  (two operands per FFMA)\n", 0.5);
        printf("    FMAs per memory instruction, all : %6.2f  (naive 0.50)\n",
               1.0/(2.0 + 1.0/gl));
        printf("  The inner loop is LDS, LDS, FFMA. Tiling changed the OPCODE of the\n"
               "  two operand fetches from LDG to LDS. It did not change that there\n"
               "  are two of them per multiply-add. Module 16's closing sentence --\n"
               "  a cache reduces the cost of a memory instruction, not the number\n"
               "  of them -- applies to a scratchpad just as exactly, and that is\n"
               "  why the measured gain is a small factor and not a large one.\n");
        printf("  Measured: %.1f%% of cuBLAS. Module 18 gives each thread several\n"
               "  outputs so one LDS feeds several FFMAs. example02 measures how\n"
               "  much that is worth, and what the shared-memory bandwidth ceiling\n"
               "  says the limit is, before Module 18 implements it.\n",
               100.0*flops/(best[bestTile]*1e-3)/1e9/gcublas);
    }

    CHECK_CUBLAS(cublasDestroy(h));
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC); free(hC0); free(poison);
    printf("\nOVERALL: %s\n", allOk ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return allOk ? 0 : 1;
}
