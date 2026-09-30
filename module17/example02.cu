// =============================================================================
// Module 17 / Example 2 — why the tiled GEMM stops where it stops.
//
// GOAL : Account for the tiled kernel's throughput from measured ceilings,
//        not from a story. Specifically:
//          - measure the shared-memory read bandwidth of this SM, scalar and
//            vectorised, and show that it, not DRAM and not the FP32 pipe, is
//            what a 2-shared-loads-per-FMA kernel runs into;
//          - measure what happens when the shared-load-to-FMA ratio changes,
//            without implementing register blocking (Module 18 owns that);
//          - count the bank-conflict degree of every shared access in the
//            tiled kernel, and MEASURE what padding does about it;
//          - measure the cost of the two barriers, and of deleting the WAR
//            barrier correctly (double buffering) and incorrectly.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN  : example02.exe
//
// Sections:
//   A  three ceilings: FP32 FFMA, DRAM stream, shared-memory read
//   B  the shared-loads-per-FMA probe  (the Module 18 hand-off number)
//   C  bank-conflict degrees by enumeration, and the padding measurement
//   D  the two barriers, double buffering, and the cost of getting it wrong
//   E  the ledger
//
// No register blocking, no float4 operand loads, no cp.async. Module 18.
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
// ceiling kernels
// =============================================================================
__global__ void ffmaCeiling(float *out, int iters)
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
__global__ void streamCeiling(const float4 *in, float4 *out, size_t n4)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 s0 = make_float4(0.0f,0.0f,0.0f,0.0f);
    for (; i < n4; i += stride) { float4 v = in[i];
        s0.x += v.x; s0.y += v.y; s0.z += v.z; s0.w += v.w; }
    if (s0.x + s0.y + s0.z + s0.w == 1.2345e30f) out[0] = s0;
}

// Shared-memory READ bandwidth, scalar 4-byte accesses.
// Stride 257 words between successive loads: odd, so consecutive lanes hit
// consecutive banks (degree 1), and non-adjacent, so the compiler cannot merge
// them into LDS.64/LDS.128 (spec 12 rule 11 -- confirmed in the SASS).
// Four independent accumulators so the FADD dependence chain is not the limit.
#define SBW_WORDS 4096
__global__ void smemReadScalar(float *out, int iters)
{
    __shared__ float s[SBW_WORDS];
    for (int i = threadIdx.x; i < SBW_WORDS; i += blockDim.x) s[i] = (float)i;
    __syncthreads();
    int base = threadIdx.x;
    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            a0 += s[(base + (j*4+0)*257) & (SBW_WORDS-1)];
            a1 += s[(base + (j*4+1)*257) & (SBW_WORDS-1)];
            a2 += s[(base + (j*4+2)*257) & (SBW_WORDS-1)];
            a3 += s[(base + (j*4+3)*257) & (SBW_WORDS-1)];
        }
        base += 1;      // every address changes every iteration, so the loads
                        // cannot be hoisted out of the loop. Without this the
                        // kernel reports 40 TB/s -- 4x the bank array's
                        // theoretical maximum -- because it issues no LDS at all.
    }
    float v = a0+a1+a2+a3;
    if (v == 1.2345e30f) out[0] = v;
}
// Same traffic through 16-byte accesses: LDS.128.
#define SBW_VEC 1024
__global__ void smemReadVec(float *out, int iters)
{
    __shared__ float4 s[SBW_VEC];
    for (int i = threadIdx.x; i < SBW_VEC; i += blockDim.x)
        s[i] = make_float4((float)i,1.0f,2.0f,3.0f);
    __syncthreads();
    int base = threadIdx.x;
    float4 a0=make_float4(0,0,0,0), a1=a0, a2=a0, a3=a0;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float4 v0 = s[(base + (j*4+0)*257) & (SBW_VEC-1)];
            float4 v1 = s[(base + (j*4+1)*257) & (SBW_VEC-1)];
            float4 v2 = s[(base + (j*4+2)*257) & (SBW_VEC-1)];
            float4 v3 = s[(base + (j*4+3)*257) & (SBW_VEC-1)];
            a0.x+=v0.x; a0.y+=v0.y; a0.z+=v0.z; a0.w+=v0.w;
            a1.x+=v1.x; a1.y+=v1.y; a1.z+=v1.z; a1.w+=v1.w;
            a2.x+=v2.x; a2.y+=v2.y; a2.z+=v2.z; a2.w+=v2.w;
            a3.x+=v3.x; a3.y+=v3.y; a3.z+=v3.z; a3.w+=v3.w;
        }
        base += 1;      // see smemReadScalar: defeats loop-invariant hoisting
    }
    float v = a0.x+a0.y+a0.z+a0.w + a1.x+a1.y+a1.z+a1.w
            + a2.x+a2.y+a2.z+a2.w + a3.x+a3.y+a3.z+a3.w;
    if (v == 1.2345e30f) out[0] = v;
}

// =============================================================================
// the GEMM kernels under study
// =============================================================================

// the canonical tiled kernel of example01, PAD as a template parameter
template <int T, int PAD>
__global__ void gemmTiled(int M, int N, int K, const float * __restrict__ A,
                          const float * __restrict__ B, float *C)
{
    __shared__ float As[T][T + PAD];
    __shared__ float Bs[T][T + PAD];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.0f;
    const int nTiles = (K + T - 1) / T;
    for (int t = 0; t < nTiles; ++t) {
        const int aCol = t*T + tx, bRow = t*T + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) acc = fmaf(As[ty][k], Bs[k][tx], acc);
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// A stored TRANSPOSED in shared memory: As[k][m] instead of As[m][k].
// This is the layout Module 18's register-blocked kernel wants, because it
// lets a thread read a column of the A tile with one vector load. Here it is
// present only to construct a genuine bank conflict: the cooperative STORE
// As[tx][ty] writes down a column of the shared array.
template <int T, int PAD>
__global__ void gemmTiledAT(int M, int N, int K, const float * __restrict__ A,
                            const float * __restrict__ B, float *C)
{
    __shared__ float As[T][T + PAD];          // As[k][m]
    __shared__ float Bs[T][T + PAD];          // Bs[k][n]
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.0f;
    const int nTiles = (K + T - 1) / T;
    for (int t = 0; t < nTiles; ++t) {
        const int aCol = t*T + tx, bRow = t*T + ty;
        As[tx][ty] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) acc = fmaf(As[k][ty], Bs[k][tx], acc);
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// Double-buffered: two shared tiles, ONE barrier per k-step.
// Module 9 established that a WAR hazard needs only the execution half of
// __syncthreads (guarantee G1) and that it is therefore the barrier you can
// design away. Here the write in iteration t+1 targets the buffer that was
// read in iteration t-1, and the single barrier at the top of iteration t+1
// already separates them. Cost: twice the shared memory.
template <int T>
__global__ void gemmTiledDB(int M, int N, int K, const float * __restrict__ A,
                            const float * __restrict__ B, float *C)
{
    __shared__ float As[2][T][T];
    __shared__ float Bs[2][T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.0f;
    const int nTiles = (K + T - 1) / T;

    {   const int aCol = tx, bRow = ty;          // prologue: tile 0 into buf 0
        As[0][ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
        Bs[0][ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f; }

    int buf = 0;
    for (int t = 0; t < nTiles; ++t) {
        __syncthreads();                         // the only barrier
        if (t + 1 < nTiles) {                    // prefetch into the other buffer
            const int aCol = (t+1)*T + tx, bRow = (t+1)*T + ty;
            As[buf^1][ty][tx] = (row < M && aCol < K) ? A[(size_t)row*K + aCol] : 0.0f;
            Bs[buf^1][ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow*N + col] : 0.0f;
        }
        #pragma unroll
        for (int k = 0; k < T; ++k) acc = fmaf(As[buf][ty][k], Bs[buf][k][tx], acc);
        buf ^= 1;
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// *** INTENTIONALLY INCORRECT ***: the WAR barrier removed with no second
// buffer. Present for timing and for the correctness check in section D, which
// reports whether the race actually fires at each tile size. Never ship this.
template <int T>
__global__ void gemmTiledNoWar(int M, int N, int K, const float * __restrict__ A,
                               const float * __restrict__ B, float *C)
{
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.0f;
    const int nTiles = (K + T - 1) / T;
    for (int t = 0; t < nTiles; ++t) {
        const int aCol = t*T + tx, bRow = t*T + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) acc = fmaf(As[ty][k], Bs[k][tx], acc);
        /* MISSING WAR BARRIER */
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// The shared-loads-per-FMA probe. NOT an optimization and NOT a GEMM: it reads
// exactly the same two shared words the tiled kernel reads and then issues R
// FFMAs against them instead of one. The result is arithmetically meaningless.
// This is Module 16 section E, moved one level down the hierarchy.
template <int T, int R>
__global__ void sharedProbe(int M, int N, int K, const float * __restrict__ A,
                            const float * __restrict__ B, float *C)
{
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    const int nTiles = (K + T - 1) / T;
    for (int t = 0; t < nTiles; ++t) {
        const int aCol = t*T + tx, bRow = t*T + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) {
            const float av = As[ty][k], bv = Bs[k][tx];
            #pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = fmaf(av, bv, acc[r]);
        }
        __syncthreads();
    }
    float s = 0.0f;
    #pragma unroll
    for (int r = 0; r < R; ++r) s += acc[r];
    if (row < M && col < N) C[(size_t)row * N + col] = s;
}

// =============================================================================
// C. bank-conflict degree by enumeration (Module 7's method, host side)
// =============================================================================
// bank(addr_bytes) = (addr_bytes / 4) % 32. The degree of a warp access is the
// maximum, over banks, of the number of DISTINCT WORDS that bank must supply.
// Requests for the same word are merged and broadcast for free.
static int degree(const long long *word, int n)
{
    int worst = 1;
    for (int b = 0; b < 32; ++b) {
        long long seen[32]; int ns = 0;
        for (int i = 0; i < n; ++i) {
            if ((int)(word[i] % 32) != b) continue;
            int dup = 0;
            for (int j = 0; j < ns; ++j) if (seen[j] == word[i]) { dup = 1; break; }
            if (!dup && ns < 32) seen[ns++] = word[i];
        }
        if (ns > worst) worst = ns;
    }
    return worst;
}
// warp 0 of a (T, T) block: lane -> (tx, ty) by Module 3's linearization.
static void tileDegrees(int T, int PAD, int k,
                        int *dAcomputeRow, int *dBcompute,
                        int *dAstoreRow, int *dAstoreCol)
{
    const int P = T + PAD;
    long long w[32];
    // compute read of a row-major A tile: As[ty][k]
    for (int l = 0; l < 32; ++l) { int ty = l / T;
        w[l] = (long long)ty * P + k; }
    *dAcomputeRow = degree(w, 32);
    // compute read of B: Bs[k][tx]
    for (int l = 0; l < 32; ++l) { int tx = l % T;
        w[l] = (long long)k * P + tx; }
    *dBcompute = degree(w, 32);
    // cooperative store into a row-major A tile: As[ty][tx]
    for (int l = 0; l < 32; ++l) { int tx = l % T, ty = l / T;
        w[l] = (long long)ty * P + tx; }
    *dAstoreRow = degree(w, 32);
    // cooperative store into a TRANSPOSED A tile: As[tx][ty]
    for (int l = 0; l < 32; ++l) { int tx = l % T, ty = l / T;
        w[l] = (long long)tx * P + ty; }
    *dAstoreCol = degree(w, 32);
}

// =============================================================================
// timing scaffolding
// =============================================================================
typedef struct { int M,N,K; const float *dA,*dB; float *dC; } Ctx;

enum {
    C_T16 = 0, C_T16P, C_T32, C_T32P,
    C_AT16, C_AT16P, C_AT32, C_AT32P,
    C_DB16, C_NOWAR16, C_DB32, C_NOWAR32,
    NCFG
};
static const char *CNAME[NCFG] = {
    "tiled 16, pad 0", "tiled 16, pad 1", "tiled 32, pad 0", "tiled 32, pad 1",
    "A^T   16, pad 0", "A^T   16, pad 1", "A^T   32, pad 0", "A^T   32, pad 1",
    "double-buf 16   ", "no-WAR 16 (WRONG)", "double-buf 32   ", "no-WAR 32 (WRONG)"
};

#define G16 dim3 bl(16,16); dim3 gr((unsigned)((c->N+15)/16),(unsigned)((c->M+15)/16))
#define G32 dim3 bl(32,32); dim3 gr((unsigned)((c->N+31)/32),(unsigned)((c->M+31)/32))

static void launchCfg(int id, Ctx *c)
{
    switch (id) {
      case C_T16:     { G16; gemmTiled<16,0><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      case C_T16P:    { G16; gemmTiled<16,1><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      case C_T32:     { G32; gemmTiled<32,0><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      case C_T32P:    { G32; gemmTiled<32,1><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      case C_AT16:    { G16; gemmTiledAT<16,0><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      case C_AT16P:   { G16; gemmTiledAT<16,1><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      case C_AT32:    { G32; gemmTiledAT<32,0><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      case C_AT32P:   { G32; gemmTiledAT<32,1><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      case C_DB16:    { G16; gemmTiledDB<16><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      case C_NOWAR16: { G16; gemmTiledNoWar<16><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      case C_DB32:    { G32; gemmTiledDB<32><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      case C_NOWAR32: { G32; gemmTiledNoWar<32><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);} break;
      default: break;
    }
}
static void launchProbe(int r, Ctx *c)
{
    dim3 bl(16,16);
    dim3 gr((unsigned)((c->N+15)/16),(unsigned)((c->M+15)/16));
    switch (r) {
      case 0: sharedProbe<16,1><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC); break;
      case 1: sharedProbe<16,2><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC); break;
      case 2: sharedProbe<16,4><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC); break;
      default:sharedProbe<16,8><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC); break;
    }
}

typedef void (*RunFn)(void*);
static double timeOnce(RunFn f, void *p, int iters)
{
    cudaEvent_t e0,e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) f(p);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms,e0,e1));
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    return ms / iters;
}
static int calibrate(RunFn f, void *p)
{
    double ms = timeOnce(f,p,1);
    int it = (int)(10.0 / (ms > 0.0 ? ms : 0.01));
    if (it < 1) it = 1;
    if (it > 128) it = 128;
    return it;
}
static void warmup(RunFn f, void *p, float target)
{
    cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
    float el=0.0f; CHECK(cudaEventRecord(w0));
    while (el < target) { f(p);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
}

// run-function adapters
static int   g_cfg;  static Ctx g_ctx;
static void  runCfg(void *p)   { (void)p; launchCfg(g_cfg, &g_ctx); }
static int   g_probe;
static void  runProbe(void *p) { (void)p; launchProbe(g_probe, &g_ctx); }
typedef struct { const float4 *in; float4 *out; size_t n4; int grid; } SCtx;
static void  runStream(void *p){ SCtx *s=(SCtx*)p;
                                 streamCeiling<<<s->grid,256>>>(s->in,s->out,s->n4); }
typedef struct { float *out; int iters, grid; } FCtx;
static void  runFfma(void *p)  { FCtx *f=(FCtx*)p; ffmaCeiling<<<f->grid,256>>>(f->out,f->iters); }
typedef struct { float *out; int iters, grid, block; } BCtx;
static void  runSbwS(void *p)  { BCtx *b=(BCtx*)p; smemReadScalar<<<b->grid,b->block>>>(b->out,b->iters); }
static void  runSbwV(void *p)  { BCtx *b=(BCtx*)p; smemReadVec<<<b->grid,b->block>>>(b->out,b->iters); }

// =============================================================================
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const int M = M_DIM, N = N_DIM, K = K_DIM;
    const size_t sA=(size_t)M*K, sB=(size_t)K*N, sC=(size_t)M*N;
    const double flops = 2.0*(double)M*N*K;

    printf("=== Module 17 / Example 2 - why the tiled GEMM stops where it stops ===\n");
    printf("C(%d x %d) = A(%d x %d) * B(%d x %d), row-major fp32\n\n", M,N,M,K,K,N);

    int nsm=0; CHECK(cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0));

    float *dA,*dB,*dC;
    CHECK(cudaMalloc(&dA,sA*4)); CHECK(cudaMalloc(&dB,sB*4)); CHECK(cudaMalloc(&dC,sC*4));
    { float *h=(float*)malloc((sB>sA?sB:sA)*4);
      for (size_t i=0;i<sA;++i) h[i] = 0.5f + (float)(i % 251)/502.0f;
      CHECK(cudaMemcpy(dA,h,sA*4,cudaMemcpyHostToDevice));
      for (size_t i=0;i<sB;++i) h[i] = 0.5f + (float)(i % 257)/514.0f;
      CHECK(cudaMemcpy(dB,h,sB*4,cudaMemcpyHostToDevice));
      free(h); }
    g_ctx.M=M; g_ctx.N=N; g_ctx.K=K; g_ctx.dA=dA; g_ctx.dB=dB; g_ctx.dC=dC;

    // ------------------------------------------------------------------ A
    printf("-- A. three ceilings, measured on this GPU ------------------------\n");
    const size_t SBYTES = 256u*1024u*1024u;
    float4 *sIn,*sOut; CHECK(cudaMalloc(&sIn,SBYTES)); CHECK(cudaMalloc(&sOut,SBYTES));
    CHECK(cudaMemset(sIn,0x3c,SBYTES));
    float *fOut; CHECK(cudaMalloc(&fOut,4));

    int bpsmF=0, bpsmS=0, bpsmV=0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsmF,(const void*)ffmaCeiling,256,0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsmS,(const void*)smemReadScalar,256,0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsmV,(const void*)smemReadVec,256,0));

    SCtx sc; sc.in=sIn; sc.out=sOut; sc.n4=SBYTES/sizeof(float4); sc.grid=nsm*bpsmF*2;
    FCtx fc; fc.out=fOut; fc.iters=50000; fc.grid=nsm*bpsmF;
    BCtx bs; bs.out=fOut; bs.iters=4000; bs.grid=nsm*bpsmS; bs.block=256;
    BCtx bv; bv.out=fOut; bv.iters=4000; bv.grid=nsm*bpsmV; bv.block=256;

    // spec 12 rule 4 + its corollary: stream first (memory P-state), then
    // compute (SM clock). Warming with only one understates the other.
    printf("  warming 1500 ms stream + 500 ms FFMA ...\n");
    warmup(runStream,&sc,1500.0f);
    double bS=1e30; { int it=calibrate(runStream,&sc);
        for (int s=0;s<6;++s){ double t=timeOnce(runStream,&sc,it); if(t<bS)bS=t; } }
    warmup(runFfma,&fc,500.0f);
    double bF=1e30; { int it=calibrate(runFfma,&fc);
        for (int s=0;s<4;++s){ double t=timeOnce(runFfma,&fc,it); if(t<bF)bF=t; } }
    double bSS=1e30, bSV=1e30;
    { int itS=calibrate(runSbwS,&bs), itV=calibrate(runSbwV,&bv);
      for (int s=0;s<4;++s) {                       // rotated, 2 configs
        if (s&1) { double a=timeOnce(runSbwS,&bs,itS); if(a<bSS)bSS=a;
                   double b=timeOnce(runSbwV,&bv,itV); if(b<bSV)bSV=b; }
        else     { double b=timeOnce(runSbwV,&bv,itV); if(b<bSV)bSV=b;
                   double a=timeOnce(runSbwS,&bs,itS); if(a<bSS)bSS=a; } } }
    CHECK(cudaGetLastError());

    const double ceilCompute = 2.0*8.0*fc.iters*256.0*fc.grid/(bF*1e-3)/1e9;   // GFLOP/s
    const double ceilDram    = (double)SBYTES/(bS*1e-3)/1e9;                   // GB/s
    const double sbwScalar   = (double)bs.iters*64.0*4.0*256.0*bs.grid/(bSS*1e-3)/1e12; // TB/s
    const double sbwVec      = (double)bv.iters*16.0*16.0*256.0*bv.grid/(bSV*1e-3)/1e12;
    const double clk         = ceilCompute/(nsm*128.0*2.0);                    // GHz

    printf("  FP32 FFMA ceiling        %10.1f GFLOP/s   (implied SM clock %.3f GHz)\n",
           ceilCompute, clk);
    printf("  DRAM read ceiling        %10.1f GB/s      (%.0f%% of the 432.0 GB/s pin peak)\n",
           ceilDram, 100.0*ceilDram/432.0);
    // Module 12 recorded 410.5-410.7 GB/s for this kernel. Reading well below
    // it here is the effect Module 16 documented: the power manager drops the
    // SM clock during a pure-read kernel, and this program measures the stream
    // in the middle of a compute-heavy session. It does not matter for the
    // argument below -- the tiled GEMM is nowhere near DRAM-bound -- but do not
    // quote this line as the machine's streaming ceiling.
    printf("  shared read, 4 B  (LDS)  %10.3f TB/s      = %5.1f B/cycle/SM\n",
           sbwScalar, sbwScalar*1e12/(nsm*clk*1e9));
    printf("  shared read, 16 B (LDS.128) %7.3f TB/s      = %5.1f B/cycle/SM\n",
           sbwVec, sbwVec*1e12/(nsm*clk*1e9));
    printf("\n  The bank array is 32 banks x 4 B = 128 B per cycle per SM. Module 7\n"
           "  measured that a warp's conflict-free 4-byte shared access occupies\n"
           "  the pipeline for TWO cycles even with zero conflicts, so scalar LDS\n"
           "  realises about half the array's bandwidth and a 16-byte access\n"
           "  realises essentially all of it. The measured RATIO of the two rows\n"
           "  (~1.9x) is the stable quantity; the B/cycle column divides by the\n"
           "  clock implied by the FFMA ceiling, and the shared kernel is a\n"
           "  different load that very likely clocks higher, so read that column\n"
           "  as an upper estimate rather than a measurement.\n");
    const double needPerFma = 8.0;                       // 2 operands x 4 B
    const double needTBs = needPerFma * (ceilCompute/2.0) * 1e9 / 1e12;
    printf("\n  A GEMM inner loop that reads BOTH operands from shared memory needs\n"
           "  %.0f bytes of shared traffic per FMA. At the FP32 ceiling of %.0f\n"
           "  GFLOP/s = %.0f G FMA/s that is %.1f TB/s of shared bandwidth.\n",
           needPerFma, ceilCompute, ceilCompute/2.0, needTBs);
    printf("  Available: %.3f-%.3f TB/s. So a two-shared-loads-per-FMA kernel is\n"
           "  capped at %.1f%% - %.1f%% of the FP32 ceiling, i.e. %.0f - %.0f GFLOP/s,\n"
           "  no matter how large the tile is or how well it is laid out.\n",
           sbwScalar, sbwVec, 100.0*sbwScalar/needTBs, 100.0*sbwVec/needTBs,
           ceilCompute*sbwScalar/needTBs, ceilCompute*sbwVec/needTBs);
    printf("  THAT is the wall Module 17 runs into, and it is not DRAM.\n\n");

    // ------------------------------------------------------------------ C
    printf("-- C. bank-conflict degrees, by enumeration -----------------------\n");
    printf("  Module 7's method: bank = (byte address / 4) %% 32, degree =\n"
           "  max over banks of DISTINCT WORDS requested. Warp 0 of a (T,T)\n"
           "  block, k = 3. Cost on Ada is max(2, D), so D <= 2 is free.\n\n");
    printf("  %-6s %-5s %-16s %-14s %-16s %-16s\n",
           "T", "pad", "read As[ty][k]", "read Bs[k][tx]", "store As[ty][tx]",
           "store As[tx][ty]");
    for (int T = 16; T <= 32; T *= 2)
        for (int P = 0; P <= 1; ++P) {
            int d1,d2,d3,d4; tileDegrees(T,P,3,&d1,&d2,&d3,&d4);
            printf("  %-6d %-5d %-16d %-14d %-16d %-16d\n", T,P,d1,d2,d3,d4);
        }
    printf("\n  Every access in the row-major tiled kernel is degree 1 already.\n"
           "  There is nothing for padding to fix. The only degree > 2 in the\n"
           "  table is the store into a TRANSPOSED A tile -- the 'A^T' rows of\n"
           "  section D -- and padding does fix that one.\n");
    printf("\n  Padding is not free even when it fixes nothing. Count the SASS:\n"
           "    nvcc -arch=sm_89 -O3 -cubin -o e2.cubin example02.cu\n"
           "    cuobjdump -sass e2.cubin\n"
           "  Steady-state body of gemmTiled<16,0> : 4 LDS.128 + 16 LDS + 16 FFMA\n"
           "  Steady-state body of gemmTiled<16,1> :             32 LDS + 16 FFMA\n"
           "  Steady-state body of gemmTiled<32,0> : 8 LDS.128 + 32 LDS + 32 FFMA\n"
           "  Steady-state body of gemmTiled<32,1> :             64 LDS + 32 FFMA\n"
           "  As[ty][0..T-1] is contiguous, so ptxas merges four of those reads\n"
           "  into one 16-byte LDS.128 -- but only when the row base is 16-byte\n"
           "  aligned. A pitch of T+1 floats puts row ty at byte 4*(T+1)*ty, which\n"
           "  is not a multiple of 16, and the merge is lost: 20 shared\n"
           "  instructions become 32. Section D measures what that costs.\n\n");

    // ------------------------------------------------------------------ D
    printf("-- D. measured: padding, transposed staging, and the barriers -----\n");
    printf("  All twelve configurations timed back to back, rotated, min of %d\n"
           "  sweeps (spec 12 rules 1, 3, 9). Validation is a separate pass.\n\n",
           NCFG);

    double best[NCFG]; int iters[NCFG];
    for (int i=0;i<NCFG;++i){ g_cfg=i; iters[i]=calibrate(runCfg,NULL); best[i]=1e30; }
    for (int s=0;s<NCFG;++s)
        for (int q=0;q<NCFG;++q) {
            const int p=(q+s)%NCFG;
            g_cfg=p; double t=timeOnce(runCfg,NULL,iters[p]);
            if (t<best[p]) best[p]=t;
        }
    CHECK(cudaGetLastError());

    // correctness pass, after all timing
    float *hC=(float*)malloc(sC*4);
    float *hRef=(float*)malloc(sC*4);
    g_cfg=C_T16; launchCfg(C_T16,&g_ctx);
    CHECK(cudaDeviceSynchronize()); CHECK(cudaMemcpy(hRef,dC,sC*4,cudaMemcpyDeviceToHost));
    int mism[NCFG];
    for (int i=0;i<NCFG;++i) {
        CHECK(cudaMemset(dC,0,sC*4));
        launchCfg(i,&g_ctx); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hC,dC,sC*4,cudaMemcpyDeviceToHost));
        int bad = 0;
        for (size_t e=0;e<sC;++e) {
            double d = fabs((double)hC[e]-(double)hRef[e]);
            if (!(d <= 1e-3 * fabs((double)hRef[e]) + 1e-3)) ++bad;
        }
        mism[i] = bad;
    }

    printf("  %-18s %9s %9s %9s %12s\n",
           "config", "ms", "GFLOP/s", "x tiled16", "mismatches");
    for (int i=0;i<NCFG;++i)
        printf("  %-18s %9.4f %9.1f %9.3f %12d\n", CNAME[i], best[i],
               flops/(best[i]*1e-3)/1e9, best[C_T16]/best[i], mism[i]);

    printf("\n  padding the row-major tile: T=16 %.3fx, T=32 %.3fx\n",
           best[C_T16]/best[C_T16P], best[C_T32]/best[C_T32P]);
    printf("  padding the transposed tile: T=16 %.3fx, T=32 %.3fx\n",
           best[C_AT16]/best[C_AT16P], best[C_AT32]/best[C_AT32P]);
    printf("  double buffering (2 barriers -> 1): T=16 %.3fx, T=32 %.3fx\n",
           best[C_T16]/best[C_DB16], best[C_T32]/best[C_DB32]);
    printf("  deleting the WAR barrier with no second buffer: T=16 %.3fx with\n"
           "  %d wrong elements, T=32 %.3fx with %d wrong elements.\n",
           best[C_T16]/best[C_NOWAR16], mism[C_NOWAR16],
           best[C_T32]/best[C_NOWAR32], mism[C_NOWAR32]);

    // ------------------------------------------------------------------ B
    printf("\n-- B. the shared-loads-per-FMA probe ------------------------------\n");
    printf("  Same two shared reads per k, R fused multiply-adds instead of one.\n"
           "  Arithmetically meaningless; the instruction mix is the point. This\n"
           "  is Module 16 section E moved one level down the memory hierarchy.\n\n");
    const int RS[4] = {1,2,4,8};
    double bp[4]; int ip[4];
    for (int i=0;i<4;++i){ g_probe=i; ip[i]=calibrate(runProbe,NULL); bp[i]=1e30; }
    for (int s=0;s<4;++s)
        for (int q=0;q<4;++q){ int p=(q+s)%4; g_probe=p;
            double t=timeOnce(runProbe,NULL,ip[p]); if (t<bp[p]) bp[p]=t; }
    CHECK(cudaGetLastError());
    printf("  %4s %14s %10s %12s %14s\n",
           "R", "shared ld/FMA", "ms", "GFLOP/s", "% FP32 ceiling");
    for (int i=0;i<4;++i) {
        const double g = flops*RS[i]/(bp[i]*1e-3)/1e9;
        printf("  %4d %14.2f %10.4f %12.1f %13.1f%%\n",
               RS[i], 2.0/RS[i], bp[i], g, 100.0*g/ceilCompute);
    }
    printf("\n  The tiled GEMM is the R = 1 row. Every other row is unreachable by\n"
           "  tiling, because tiling cannot change how many operands one FMA\n"
           "  needs. Only giving a thread more than one output can: with an\n"
           "  Rr x Rc register tile a thread loads Rr + Rc shared words and does\n"
           "  Rr*Rc FMAs, so the ratio becomes (Rr+Rc)/(Rr*Rc). Module 18.\n");

    // ------------------------------------------------------------------ E
    printf("\n-- E. the ledger --------------------------------------------------\n");
    {
        const double gTiled = flops/(best[C_T16]*1e-3)/1e9;
        const double capLo = ceilCompute*sbwScalar/needTBs;
        const double capHi = ceilCompute*sbwVec/needTBs;
        printf("  tiled 16x16, measured                  %8.1f GFLOP/s\n", gTiled);
        printf("  shared-bandwidth cap, all-scalar LDS   %8.1f GFLOP/s\n", capLo);
        printf("  shared-bandwidth cap, all LDS.128      %8.1f GFLOP/s\n", capHi);
        printf("  FP32 ceiling                           %8.1f GFLOP/s\n", ceilCompute);
        printf("\n  The kernel lands BETWEEN the two caps (%.0f%% of the way from\n"
               "  scalar to vector), which is the only region it could land in:\n"
               "  ptxas merges each group of four A reads into one LDS.128 and\n"
               "  leaves the B reads scalar, so the operand traffic is split\n"
               "  across both paths. Nothing here is fitted -- the two caps come\n"
               "  from separate microbenchmarks that never saw the GEMM.\n",
               100.0*(gTiled-capLo)/(capHi-capLo));
        printf("\n  Shared loads per FMA needed to reach 80%% of the FP32 ceiling:\n");
        printf("    available shared bandwidth / (0.8 * required) -> about %.2f\n",
               2.0*sbwVec/(0.8*needTBs));
        printf("  A square Rr x Rr register tile supplies 2/Rr. Solve: Rr >= %.1f.\n",
               2.0/(2.0*sbwVec/(0.8*needTBs)));
        printf("  That is the size of the register tile Module 18 has to build,\n"
               "  derived here without building it.\n");
    }

    free(hC); free(hRef);
    CHECK(cudaFree(sIn)); CHECK(cudaFree(sOut)); CHECK(cudaFree(fOut));
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));

    // The no-WAR kernels are expected to be wrong; everything else must match.
    int pass = 1;
    for (int i=0;i<NCFG;++i) {
        if (i==C_NOWAR16 || i==C_NOWAR32) continue;
        if (mism[i] != 0) pass = 0;
    }
    printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return pass ? 0 : 1;
}
