// =============================================================================
// Module 17 / Exercise 1 — write the tiled GEMM.
//
// GOAL : Turn Module 16's one-thread-one-output GEMM into a block-cooperative
//        tiled GEMM, correct at a size that is a multiple of nothing, and find
//        out what tiling alone is actually worth.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN  : exercise01.exe
//
// TODO 1 - the A-tile cooperative load, boundaries included       [kernel]
// TODO 2 - the B-tile cooperative load, boundaries included       [kernel]
// TODO 3 - the tile loop: how many tiles, and where the barriers go
//                                                                 [kernel]
// TODO 4 - the store, honouring the BLAS beta == 0 contract       [kernel]
// TODO 5 - PREDICTION, committed before you build     [host, DESIGN TODO]
//
// The problem is C = alpha*A*B + beta*C, row-major fp32, at
//     M = 1035, N = 1541, K = 1063.
// M % 8/16/32 = 3/11/11, N % 8/16/32 = 5/5/5, K % 8/16/32 = 7/7/7.
// There is a partial tile on every axis for every tile size tested, and no
// two of M, N, K are equal. That is deliberate: a square power-of-two test
// hides every bug this exercise can produce.
//
// The harness runs your kernel at T = 8, 16 and 32, on two datasets with
// different conditioning, against Module 16's validator.
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

#define M_DIM 1035
#define N_DIM 1541
#define K_DIM 1063

// =============================================================================
// TODO 5 — PREDICTION. Fill this in BEFORE you build anything.
//
// Module 16 measured the naive GEMM on this class of problem at 1275-1348
// GFLOP/s, cuBLAS at 8150-8705, and established that >= 92 % of the naive
// kernel's 12.97 GB of requested traffic was already being served on chip.
// You are about to replace those global operand loads with shared-memory
// loads.
//
// Predict the ratio (best tiled time) / (naive time) as a bucket:
//    1 = tiled is SLOWER than naive                 (ratio > 1.0)
//    2 = 1.0x to 1.5x faster
//    3 = 1.5x to 3x faster
//    4 = 3x to 6x faster
//    5 = more than 6x faster (i.e. at or past cuBLAS)
//
// Write down your reasoning in one sentence before you look at anything else.
// The single most useful question to ask yourself: which instruction does a
// tiled kernel issue in its inner loop that a naive kernel does not, and which
// one does it stop issuing?
// =============================================================================
#define PREDICTION 0          // <- 1..5, YOUR ANSWER HERE

// =============================================================================
// TODO 1..4 — the kernel.
//
// A block of T x T threads owns a T x T tile of C. blockDim is (T, T), so
// threadIdx.x runs along the N axis of C and threadIdx.y along the M axis.
//
// Layout reminder, row-major, no padding:
//     A is M x K,  element (r, k) at A[r*K + k]      k is A's FAST axis
//     B is K x N,  element (k, c) at B[k*N + c]      k is B's SLOW axis
//     C is M x N,  element (r, c) at C[r*N + c]
//
// The shape of the computation you are writing:
//
//     acc = 0
//     for each tile t along k:
//         stage A's tile and B's tile in shared memory
//         <barrier>
//         for kk in [0, T):  acc += As[...] * Bs[...]
//         <barrier>
//     write acc
//
// `acc` is a register and it must survive every iteration of the tile loop.
// That is the whole idea: shared memory holds a *tile*, never a whole row of
// A or column of B, so the partial dot product cannot live there.
// =============================================================================
template <int T>
__global__ void gemmTiled(int M, int N, int K,
                          float alpha, const float * __restrict__ A,
                          const float * __restrict__ B,
                          float beta, float *C)
{
    __shared__ float As[T][T];      // the A tile: T rows of C x T steps of k
    __shared__ float Bs[T][T];      // the B tile: T steps of k x T cols of C

    const int tx = threadIdx.x, ty = threadIdx.y;

    // this thread's element of C (the COMPUTE mapping)
    const int row = blockIdx.y * T + ty;
    const int col = blockIdx.x * T + tx;

    float acc = 0.0f;

    // -------------------------------------------------------------------
    // TODO 3 (part 1): how many tiles does the k loop run for?
    //   K is 1063 and T is 8, 16 or 32. None of them divides it.
    //   Getting this wrong costs you between 1 and T-1 terms of every dot
    //   product in the matrix, which is a small relative error, which is
    //   exactly why it survives a sloppy tolerance.
    // -------------------------------------------------------------------
    int nTiles = -1;
    // YOUR CODE HERE

    if (nTiles < 0) {           // safety net for the unfilled file
        if (row < M && col < N) C[(size_t)row * N + col] = 0.0f;
        return;
    }

    for (int t = 0; t < nTiles; ++t) {
        const int k0 = t * T;   // first k index of this tile
        (void)k0;               // only here to keep the unfilled file
                                // warning-clean; delete it once you use k0.

        // ---------------------------------------------------------------
        // TODO 1: cooperatively load this block's T x T tile of A into As.
        //
        //   Thread (tx, ty) fetches exactly one element. Work out which one.
        //   Two constraints decide it:
        //     - consecutive tx must read consecutive addresses of A, or the
        //       load is not coalesced (Module 5);
        //     - the element this thread stores must be the one the compute
        //       loop below will read from that slot.
        //
        //   Then handle the boundary. Both row and the k index can be out of
        //   range: M is not a multiple of T and neither is K. Decide what a
        //   cell that has no corresponding element of A should contain, and
        //   make sure EVERY thread executes the store -- including threads
        //   whose output element does not exist. A thread that returns early
        //   here does not arrive at the barrier below.
        // ---------------------------------------------------------------
        // YOUR CODE HERE

        // ---------------------------------------------------------------
        // TODO 2: cooperatively load this block's T x T tile of B into Bs.
        //
        //   This is NOT TODO 1 with the letters changed. In A, threadIdx.x
        //   indexes the contraction dimension k; in B, threadIdx.x indexes
        //   the output column. Write out which global element thread (tx, ty)
        //   must fetch and which slot of Bs it belongs in, and check that
        //   consecutive tx still read consecutive addresses of B.
        //
        //   Boundary again, and it is a different boundary: here the k index
        //   comes from ty and the column from tx.
        // ---------------------------------------------------------------
        // YOUR CODE HERE

        // ---------------------------------------------------------------
        // TODO 3 (part 2): the tile loop needs synchronisation in two places
        // and you must put it in both.
        //
        //   (a) Before the accumulation loop below, every thread must be able
        //       to read any cell of As and Bs and see the value its writer
        //       wrote. Which threads wrote the cells that thread (0,0) reads?
        //   (b) After the accumulation loop, before the next iteration
        //       overwrites the tiles.
        //
        //   Do not take (b) on faith and do not take it on trust that it is
        //   unnecessary. Reason about it: name the two threads involved, say
        //   which one writes and which one reads, and say why the hardware
        //   is permitted to run them in the order that breaks it. Module 9
        //   classified this hazard; Module 6 named it.
        //
        //   Put the first one here.
        // ---------------------------------------------------------------
        // YOUR CODE HERE

        #pragma unroll
        for (int k = 0; k < T; ++k)
            acc = fmaf(As[ty][k], Bs[k][tx], acc);

        // ---------------------------------------------------------------
        // TODO 3 (part 3): and the second one here.
        // ---------------------------------------------------------------
        // YOUR CODE HERE
    }

    // -------------------------------------------------------------------
    // TODO 4: write the result.
    //
    //   Guard first: this thread may own no element of C.
    //
    //   Then the BLAS contract. When beta is exactly zero, C is NOT read --
    //   it is write-only, and its prior contents are irrelevant and may be
    //   anything, including bit patterns that decode as inf or NaN. The
    //   harness fills C with +infinity before every beta == 0 launch, so
    //   `alpha*acc + beta*C[i]` produces 0*inf = NaN in every element and is
    //   caught. Write the store so that both the beta == 0 and the beta != 0
    //   cases are correct.
    // -------------------------------------------------------------------
    // YOUR CODE HERE

    (void)alpha; (void)beta; (void)C; (void)acc; (void)A; (void)B;
}

// =============================================================================
// Module 16's validator, verbatim. Do not replace it with a relative-to-|C|
// tolerance: Module 16 measured the course's default 1e-5*max(1,|ref|) rule
// REJECTING a correct GEMM at K = 769, and K here is 1063.
//
// Three checks, in this order:
//   1. finiteness over all M*N elements   (must be first: NaN compares false
//      against everything, so a max-error loop run first PASSES an all-NaN
//      array)
//   2. Freivalds probe, full coverage, non-negative random vector
//   3. sampled exact double reference, threshold err / (gamma_K * S_ij)
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
    const double u = ldexp(1.0, -24);
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
            double b = hB[(size_t)k*N + j];
            t += b * v[j]; tb += fabs(b) * v[j];
        }
        Bv[k] = t; aBv[k] = tb;
    }
    for (int i = 0; i < M; ++i) {
        double y = 0.0, ybound = 0.0;
        for (int k = 0; k < K; ++k) {
            double a = hA[(size_t)i*K + k];
            y += a * Bv[k]; ybound += fabs(a) * aBv[k];
        }
        double got = 0.0, want = (double)alpha * y;
        for (int j = 0; j < N; ++j) {
            got += (double)hC[(size_t)i*N + j] * v[j];
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
                double a = hA[(size_t)i*K + k], b = hB[(size_t)k*N + j];
                acc += a*b; S += fabs(a)*fabs(b);
            }
            double want = (double)alpha * acc;
            if (beta != 0.0f) want += (double)beta * (double)hC0[(size_t)i*N+j];
            double tol = fabs((double)alpha) * (gammaK + 4.0*u) * S
                       + 4.0*u*fabs((double)beta)
                             * (hC0 ? fabs((double)hC0[(size_t)i*N+j]) : 0.0);
            double d = (tol > 0.0) ? fabs((double)hC[(size_t)i*N+j] - want)/tol : 0.0;
            if (d > r.sampled) { r.sampled = d; r.sampledRow = i; r.sampledCol = j; }
        }
    if (!(r.sampled <= 1.0)) r.ok = 0;
    return r;
}

// ---- the control kernel: Module 16's naive GEMM ----------------------------
__global__ void gemmNaive(int M, int N, int K,
                          float alpha, const float *A, const float *B,
                          float beta, float *C)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc = fmaf(A[(size_t)row*K + k], B[(size_t)k*N + col], acc);
    if (beta == 0.0f) C[(size_t)row*N + col] = alpha*acc;
    else              C[(size_t)row*N + col] = alpha*acc + beta*C[(size_t)row*N+col];
}

// ---- warm-up kernels (spec 12 rule 4) --------------------------------------
__global__ void streamWarm(const float4 *in, float4 *out, size_t n4)
{
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    const size_t st = (size_t)gridDim.x*blockDim.x;
    float4 s0 = make_float4(0,0,0,0);
    for (; i < n4; i += st) { float4 v = in[i];
        s0.x+=v.x; s0.y+=v.y; s0.z+=v.z; s0.w+=v.w; }
    if (s0.x+s0.y+s0.z+s0.w == 1.2345e30f) out[0] = s0;
}
__global__ void ffmaWarm(float *out, int iters)
{
    float a0=threadIdx.x,a1=a0+1,a2=a0+2,a3=a0+3,a4=a0+4,a5=a0+5,a6=a0+6,a7=a0+7;
    const float b=1.0000001f, c=0.9999999f;
    for (int i=0;i<iters;++i){ a0=fmaf(a0,b,c);a1=fmaf(a1,b,c);a2=fmaf(a2,b,c);
        a3=fmaf(a3,b,c);a4=fmaf(a4,b,c);a5=fmaf(a5,b,c);a6=fmaf(a6,b,c);a7=fmaf(a7,b,c); }
    float s=a0+a1+a2+a3+a4+a5+a6+a7;
    if (s == 1.2345e30f) out[0]=s;
}

// ---- launch + timing -------------------------------------------------------
typedef struct { int M,N,K; float alpha,beta; const float *dA,*dB; float *dC; } Ctx;
enum { CFG_NAIVE = 0, CFG_T8, CFG_T16, CFG_T32, NCFG };
static const char *CNAME[NCFG] = { "naive (Module 16)", "tiled T=8",
                                   "tiled T=16", "tiled T=32" };

static void launchCfg(int id, Ctx *c)
{
    switch (id) {
      case CFG_NAIVE: { dim3 bl(32,8);
          dim3 gr((unsigned)((c->N+31)/32),(unsigned)((c->M+7)/8));
          gemmNaive<<<gr,bl>>>(c->M,c->N,c->K,c->alpha,c->dA,c->dB,c->beta,c->dC);
      } break;
      case CFG_T8: { dim3 bl(8,8);
          dim3 gr((unsigned)((c->N+7)/8),(unsigned)((c->M+7)/8));
          gemmTiled<8><<<gr,bl>>>(c->M,c->N,c->K,c->alpha,c->dA,c->dB,c->beta,c->dC);
      } break;
      case CFG_T16: { dim3 bl(16,16);
          dim3 gr((unsigned)((c->N+15)/16),(unsigned)((c->M+15)/16));
          gemmTiled<16><<<gr,bl>>>(c->M,c->N,c->K,c->alpha,c->dA,c->dB,c->beta,c->dC);
      } break;
      default: { dim3 bl(32,32);
          dim3 gr((unsigned)((c->N+31)/32),(unsigned)((c->M+31)/32));
          gemmTiled<32><<<gr,bl>>>(c->M,c->N,c->K,c->alpha,c->dA,c->dB,c->beta,c->dC);
      } break;
    }
}
static double timeOnce(int id, Ctx *c, int it)
{
    cudaEvent_t e0,e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    for (int i=0;i<it;++i) launchCfg(id,c);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms,e0,e1));
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    return ms/it;
}
static int calibrate(int id, Ctx *c)
{
    double ms = timeOnce(id,c,1);
    int it = (int)(10.0/(ms>0.0?ms:0.01));
    if (it<1) it=1; if (it>128) it=128;
    return it;
}

static unsigned lcg = 1u;
static float rnd01(void){ lcg = lcg*1664525u+1013904223u;
                          return (float)((lcg>>8)&0xFFFFu)/65536.0f; }

// =============================================================================
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const int M=M_DIM, N=N_DIM, K=K_DIM;
    const size_t sA=(size_t)M*K, sB=(size_t)K*N, sC=(size_t)M*N;
    const double flops = 2.0*(double)M*N*K;

    printf("=== Module 17 / Exercise 1 - the tiled GEMM ===\n");
    printf("C(%d x %d) = alpha*A(%d x %d)*B(%d x %d) + beta*C\n", M,N,M,K,K,N);
    printf("M %% 8/16/32 = %d/%d/%d  N %% 8/16/32 = %d/%d/%d  K %% 8/16/32 = %d/%d/%d\n\n",
           M%8,M%16,M%32, N%8,N%16,N%32, K%8,K%16,K%32);

    if (PREDICTION < 1 || PREDICTION > 5) { printf("Set TODO 5 first.\n"); return 0; }

    // ---- two datasets with different conditioning ---------------------------
    // positive  : |C| ~ S, so relative error against |C| is meaningful
    // zero-mean : |C| ~ sqrt(K)*sigma^2 but S ~ K*sigma^2, so the terms cancel
    //             and any tolerance scaled by |C| falls apart. Module 16 §8(c).
    float *hA=(float*)malloc(sA*4), *hB=(float*)malloc(sB*4);
    float *hC=(float*)malloc(sC*4), *hC0=(float*)malloc(sC*4);
    float *poison=(float*)malloc(sC*4);
    { union { unsigned u; float f; } inf; inf.u = 0x7F800000u;
      for (size_t i=0;i<sC;++i) poison[i]=inf.f; }
    for (size_t i=0;i<sC;++i) hC0[i] = (float)((int)(i%17)-8)*0.25f;

    float *dA,*dB,*dC;
    CHECK(cudaMalloc(&dA,sA*4)); CHECK(cudaMalloc(&dB,sB*4)); CHECK(cudaMalloc(&dC,sC*4));

    Ctx ctx; ctx.M=M; ctx.N=N; ctx.K=K; ctx.dA=dA; ctx.dB=dB; ctx.dC=dC;
    ctx.alpha=1.0f; ctx.beta=0.0f;

    int score = 0, maxScore = 0;
    printf("-- correctness -----------------------------------------------------\n");
    for (int ds = 0; ds < 2; ++ds) {
        lcg = 1u + (unsigned)ds*7919u;
        if (ds == 0) { for (size_t i=0;i<sA;++i) hA[i]=0.5f+rnd01();
                       for (size_t i=0;i<sB;++i) hB[i]=0.5f+rnd01(); }
        else         { for (size_t i=0;i<sA;++i) hA[i]=2.0f*rnd01()-1.0f;
                       for (size_t i=0;i<sB;++i) hB[i]=2.0f*rnd01()-1.0f; }
        CHECK(cudaMemcpy(dA,hA,sA*4,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB,hB,sB*4,cudaMemcpyHostToDevice));
        printf("  dataset %d (%s):\n", ds, ds==0 ? "positive [0.5,1.5)"
                                                 : "zero-mean [-1,1)");
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

            const int ok = c1.ok && c2.ok;
            printf("    %-18s nonfin %7d | Freiv %9.4g | samp %9.4g | a/b %9.4g | %s\n",
                   CNAME[i], c1.nonfinite, c1.freivalds, c1.sampled, c2.sampled,
                   ok ? "PASS" : "FAIL");
            if (i != CFG_NAIVE) { ++maxScore; if (ok) ++score; }   // 3 per dataset
        }
    }
    ctx.alpha=1.0f; ctx.beta=0.0f;
    printf("\n  gamma_K = %.3e at K = %d. A figure of 0.03 means the kernel used\n"
           "  3%% of the error budget the backward-error bound allows; a figure of\n"
           "  0.9 would mean something is wrong even though it 'passes'.\n\n",
           (double)K*ldexp(1.0,-24)/(1.0-(double)K*ldexp(1.0,-24)), K);

    // ---- timing, spec 12 ----------------------------------------------------
    printf("-- timing ----------------------------------------------------------\n");
    printf("  warming 1500 ms stream + 500 ms FFMA ...\n");
    { const size_t SB=256u*1024u*1024u;
      float4 *si,*so; CHECK(cudaMalloc(&si,SB)); CHECK(cudaMalloc(&so,SB));
      CHECK(cudaMemset(si,0x3c,SB));
      float *fo; CHECK(cudaMalloc(&fo,4));
      int nsm=0; CHECK(cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0));
      cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0.0f; CHECK(cudaEventRecord(w0));
      while (el<1500.0f){ streamWarm<<<nsm*12,256>>>(si,so,SB/sizeof(float4));
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      el=0.0f; CHECK(cudaEventRecord(w0));
      while (el<500.0f){ ffmaWarm<<<nsm*6,256>>>(fo,20000);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
      CHECK(cudaFree(si)); CHECK(cudaFree(so)); CHECK(cudaFree(fo));
      CHECK(cudaGetLastError()); }

    double best[NCFG]; int it[NCFG];
    for (int i=0;i<NCFG;++i){ it[i]=calibrate(i,&ctx); best[i]=1e30; }
    for (int s=0;s<NCFG+1;++s)                        // SWEEPS >= NCFG
        for (int q=0;q<NCFG;++q){ int p=(q+s)%NCFG;
            double t=timeOnce(p,&ctx,it[p]); if (t<best[p]) best[p]=t; }
    CHECK(cudaGetLastError());

    int bt = CFG_T8;
    for (int i=CFG_T8;i<NCFG;++i) if (best[i]<best[bt]) bt=i;
    const double ratio = best[CFG_NAIVE]/best[bt];

    printf("\n  %-18s %9s %11s %10s\n", "config", "ms", "GFLOP/s", "x naive");
    for (int i=0;i<NCFG;++i)
        printf("  %-18s %9.4f %11.1f %10.2f\n", CNAME[i], best[i],
               flops/(best[i]*1e-3)/1e9, best[CFG_NAIVE]/best[i]);

    int bucket;
    if      (ratio <= 1.0) bucket = 1;
    else if (ratio <= 1.5) bucket = 2;
    else if (ratio <= 3.0) bucket = 3;
    else if (ratio <= 6.0) bucket = 4;
    else                   bucket = 5;
    ++maxScore;
    printf("\n  best tiled / naive = %.3fx -> bucket %d. You predicted %d. %s\n",
           ratio, bucket, PREDICTION,
           bucket == PREDICTION ? "CORRECT" : "wrong");
    if (bucket == PREDICTION) ++score;

    printf("\n  Whatever bucket you landed in, write down the answer to this:\n"
           "  the naive inner loop is LDG, LDG, FFMA. What is the tiled inner\n"
           "  loop? Count the memory instructions per multiply-add in each.\n"
           "  Example 2 measures the consequence.\n");

    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC); free(hC0); free(poison);
    printf("\nSCORE: %d/%d\n", score, maxScore);
    printf("OVERALL: %s\n", score == maxScore ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return score == maxScore ? 0 : 1;
}
