// =============================================================================
// Module 21 / Example 2 — classification as a procedure, and reconciliation.
//
// GOAL : Take six kernels you have already written in this course, place each
//        one on the hierarchical roofline BEFORE timing it, then time it and
//        reconcile. The reconciliation is the lesson; the table is bookkeeping.
//
//   A  The ceilings, re-measured here (they move between sessions, so a model
//      built on last week's numbers is not a model of this machine today).
//   B  The traffic ledger and the prediction, printed before anything is timed.
//   C  The measurement.
//   D  Reconciliation: measured / predicted, and what the gap means.
//
// The six kernels and where they came from:
//   1 triad   in-place SAXPY  y += a*x        Module 11 (the 3N traffic trap)
//   2 reduce  grid-stride sum                 Module 12 rung v6
//   3 gemmN   naive GEMM                      Module 16
//   4 gemmT   16x16 shared-memory tiled GEMM  Module 17
//   5 gemmR   8x4 register-tiled GEMM         Module 18
//   6 chase   dependent pointer walk          Modules 1 and 4
//
// BUILD: nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN  : example02.exe
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

#define SM_COUNT 40
#define DRAM_PIN_PEAK_GBS 432.0

// Problem sizes. The elementwise sizes are >= 5x the 48 MB L2; the GEMM shape
// is Module 16's, kept so the two modules' tables can be compared directly.
#define NELEM   (32*1024*1024)          // 128 MB per array
#define M_DIM 1027
#define N_DIM 2053
#define K_DIM  769
#define CHASE_LEN   (1<<26)             // 256 MB of indices, 5.3x the L2
#define CHASE_THR   32                  // ONE warp on the whole GPU (M1, M4)
#define CHASE_STEPS 3000

// =============================================================================
// Kernels
// =============================================================================

// --- 1. in-place SAXPY. Module 11: this moves 3N, not 2N. Reading y is not
//        optional just because the source writes it.
__global__ void triad(float *y, const float * __restrict__ x, float a, size_t n)
{
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    const size_t s = gridDim.x*(size_t)blockDim.x;
    for (; i < n; i += s) y[i] = fmaf(a, x[i], y[i]);
}

// --- 2. Module 12 rung v6: grid-stride register accumulation, shared tree,
//        warp-shuffle tail. Reproduced, not re-taught.
template<int BS>
__global__ void reduceV6(const float * __restrict__ in, float *part, size_t n)
{
    __shared__ float sd[BS];
    float v = 0.0f;
    for (size_t i = blockIdx.x*(size_t)BS + threadIdx.x;
         i < n; i += gridDim.x*(size_t)BS) v += in[i];
    sd[threadIdx.x] = v;
    __syncthreads();
    #pragma unroll
    for (int s = BS/2; s > 32; s >>= 1) {
        if ((int)threadIdx.x < s) sd[threadIdx.x] += sd[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float w = sd[threadIdx.x] + sd[threadIdx.x + 32];
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) w += __shfl_down_sync(0xffffffffu, w, d);
        if (threadIdx.x == 0) part[blockIdx.x] = w;
    }
}

// --- 3. Module 16's naive GEMM, best mapping (x -> col).
__global__ void gemmNaive(int M, int N, int K, float alpha,
                          const float * __restrict__ A, const float * __restrict__ B,
                          float beta, float *C)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc = fmaf(A[(size_t)row*K+k], B[(size_t)k*N+col], acc);
    if (beta == 0.0f) C[(size_t)row*N+col] = alpha*acc;
    else              C[(size_t)row*N+col] = alpha*acc + beta*C[(size_t)row*N+col];
}

// --- 4. Module 17's square tiled GEMM.
template<int T>
__global__ void gemmTiled(int M, int N, int K, float alpha,
                          const float * __restrict__ A, const float * __restrict__ B,
                          float beta, float *C)
{
    __shared__ float As[T][T], Bs[T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y*T + ty, col = blockIdx.x*T + tx;
    float acc = 0.0f;
    for (int kt = 0; kt < K; kt += T) {
        As[ty][tx] = (row < M && kt+tx < K) ? A[(size_t)row*K + kt+tx] : 0.0f;
        Bs[ty][tx] = (kt+ty < K && col < N) ? B[(size_t)(kt+ty)*N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) acc = fmaf(As[ty][k], Bs[k][tx], acc);
        __syncthreads();
    }
    if (row < M && col < N) {
        if (beta == 0.0f) C[(size_t)row*N+col] = alpha*acc;
        else              C[(size_t)row*N+col] = alpha*acc + beta*C[(size_t)row*N+col];
    }
}

// --- 5. Module 18's register-tiled GEMM, the 8x4 winner, with the transposed
//        and pad-by-4 A tile.
template<int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__((BM/TM)*(BN/TN))
void gemmReg(int M, int N, int K, float alpha,
             const float * __restrict__ A, const float * __restrict__ B,
             float beta, float *C)
{
    const int NT = (BM/TM)*(BN/TN);
    const int AP = BM + 4;
    __shared__ float As[BK*AP];
    __shared__ float Bs[BK][BN];
    const int tid = threadIdx.x;
    const int tRow = tid/(BN/TN), tCol = tid%(BN/TN);
    const int rowBase = blockIdx.y*BM, colBase = blockIdx.x*BN;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const int NLA = (BM*BK + NT - 1)/NT, NLB = (BK*BN + NT - 1)/NT;
    for (int kt = 0; kt < K; kt += BK) {
        #pragma unroll
        for (int u = 0; u < NLA; ++u) {
            int idx = tid + u*NT; if (NLA*NT != BM*BK && idx >= BM*BK) break;
            int r = idx/BK, c = idx%BK;
            As[c*AP + r] = (rowBase+r < M && kt+c < K)
                         ? A[(size_t)(rowBase+r)*K + kt + c] : 0.0f;
        }
        #pragma unroll
        for (int u = 0; u < NLB; ++u) {
            int idx = tid + u*NT; if (NLB*NT != BK*BN && idx >= BK*BN) break;
            int k = idx/BN, n = idx%BN;
            Bs[k][n] = (kt+k < K && colBase+n < N)
                     ? B[(size_t)(kt+k)*N + colBase + n] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float rM[TM], rN[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rM[i] = As[kk*AP + tRow*TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rN[j] = Bs[kk][tCol*TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(rM[i], rN[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = rowBase + tRow*TM + i; if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c = colBase + tCol*TN + j;
            if (c < N) {
                if (beta == 0.0f) C[(size_t)r*N+c] = alpha*acc[i][j];
                else              C[(size_t)r*N+c] = alpha*acc[i][j] + beta*C[(size_t)r*N+c];
            }
        }
    }
}

// --- 6. The dependent pointer walk. One load's address is the previous load's
//        value, so there is exactly one memory request in flight per thread and
//        no amount of bandwidth helps.
__global__ void chase(const int * __restrict__ nxt, float *out, int steps, int nthr)
{
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= nthr) return;
    int p = t;
    float acc = 0.0f;
    for (int s = 0; s < steps; ++s) { p = nxt[p]; acc += (float)(p & 1023); }
    out[t] = acc;
}

// Warm-up kernels.
__global__ void warmStream(const float4 * __restrict__ s, float *o, size_t n)
{
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    float4 a = make_float4(0,0,0,0);
    for (; i < n; i += gridDim.x*(size_t)blockDim.x) {
        float4 v = s[i]; a.x+=v.x; a.y+=v.y; a.z+=v.z; a.w+=v.w; }
    if (a.x == 1e30f) o[0] = a.x+a.y+a.z+a.w;
}
__global__ void warmFfma(float *o, int iters)
{
    float a[8]; const float b = 1.0000001f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = (float)(threadIdx.x + i);
    #pragma unroll 8
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = fmaf(a[i], b, 1.0f); }
    float s = 0; for (int i = 0; i < 8; ++i) s += a[i];
    if (s == 1e30f) o[0] = s;
}
// Shared-memory ceiling probe (Example 1's honest form).
#define SP_N 2048
#define SP_W 32
__global__ void smemProbe(float *sink, int iters)
{
    __shared__ float s[SP_N];
    for (int i = threadIdx.x; i < SP_N; i += blockDim.x) s[i] = (float)(i & 255);
    __syncthreads();
    float acc = 0.0f; int base = (int)threadIdx.x;
    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int u = 0; u < SP_W; ++u) acc += s[(base + u*33) & (SP_N-1)];
        base += 1;
    }
    if (acc == 1e30f) sink[0] = acc;
}

// =============================================================================
// Harness
// =============================================================================
static double timeFn(void (*f)(void), int iters)
{
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) f();
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms/iters;
}

static float *dX, *dY, *dPart, *dA, *dB, *dC, *dSink, *dChaseOut;
static int   *dNxt;
static float4 *dWarm;

static void kTriad (void) { triad<<<2048,256>>>(dY, dX, 2.0f, (size_t)NELEM); }
static void kReduce(void) { reduceV6<256><<<240,256>>>(dX, dPart, (size_t)NELEM); }
static void kGemmN (void) { dim3 b(16,16), g((N_DIM+15)/16,(M_DIM+15)/16);
                            gemmNaive<<<g,b>>>(M_DIM,N_DIM,K_DIM,1.0f,dA,dB,0.0f,dC); }
static void kGemmT (void) { dim3 b(16,16), g((N_DIM+15)/16,(M_DIM+15)/16);
                            gemmTiled<16><<<g,b>>>(M_DIM,N_DIM,K_DIM,1.0f,dA,dB,0.0f,dC); }
static void kGemmR (void) { dim3 g((N_DIM+63)/64,(M_DIM+127)/128);
                            gemmReg<128,64,8,8,4><<<g,256>>>(M_DIM,N_DIM,K_DIM,1.0f,dA,dB,0.0f,dC); }
static void kChase (void) { chase<<<1,32>>>(dNxt, dChaseOut, CHASE_STEPS, CHASE_THR); }

// Module 16's validator, reduced to the two checks a same-answer sweep needs:
// finiteness FIRST (a NaN compares false against everything), then a full
// Freivalds probe with the tolerance propagated through the same contraction.
static int gemmQuickValidate(int M, int N, int K, const float *hA, const float *hB,
                             const float *hC, double *worst)
{
    const double u = ldexp(1.0,-24);
    const double gammaK = (double)K*u/(1.0-(double)K*u);
    for (size_t i = 0; i < (size_t)M*N; ++i)
        if (!isfinite(hC[i])) { *worst = 1e30; return 0; }
    double *v  = (double*)malloc(sizeof(double)*N);
    double *Bv = (double*)malloc(sizeof(double)*K);
    double *aB = (double*)malloc(sizeof(double)*K);
    unsigned s = 0xB16B00B5u;
    for (int j = 0; j < N; ++j) { s = s*1664525u+1013904223u;
        v[j] = 0.5 + (double)((s>>8)&0xFFFFu)/65536.0; }
    for (int k = 0; k < K; ++k) { double t=0, tb=0;
        for (int j = 0; j < N; ++j) { double bb = hB[(size_t)k*N+j];
            t += bb*v[j]; tb += fabs(bb)*v[j]; }
        Bv[k]=t; aB[k]=tb; }
    double w = 0.0;
    for (int i = 0; i < M; ++i) {
        double y=0, yb=0;
        for (int k = 0; k < K; ++k) { double a = hA[(size_t)i*K+k];
            y += a*Bv[k]; yb += fabs(a)*aB[k]; }
        double got = 0;
        for (int j = 0; j < N; ++j) got += (double)hC[(size_t)i*N+j]*v[j];
        double tol = (gammaK + 4.0*u)*yb;
        double ratio = (tol > 0) ? fabs(got-y)/tol : 0.0;
        if (ratio > w) w = ratio;
    }
    free(v); free(Bv); free(aB);
    *worst = w;
    return w <= 1.0;
}

// The roofline, as a function.
static double attain(double ai, double bw, double peak)
{ double m = ai*bw; return m < peak ? m : peak; }

// =============================================================================
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 21 / Example 2 - classify, predict, measure, reconcile ===\n\n");

    const size_t sA = (size_t)M_DIM*K_DIM, sB = (size_t)K_DIM*N_DIM, sC = (size_t)M_DIM*N_DIM;
    float *hX = (float*)malloc(sizeof(float)*NELEM);
    float *hY = (float*)malloc(sizeof(float)*NELEM);
    float *hA = (float*)malloc(sizeof(float)*sA);
    float *hB = (float*)malloc(sizeof(float)*sB);
    float *hC = (float*)malloc(sizeof(float)*sC);
    int   *hN = (int*)  malloc(sizeof(int)*CHASE_LEN);
    float *hChase = (float*)malloc(sizeof(float)*CHASE_THR);

    unsigned st = 12345u;
    for (int i = 0; i < NELEM; ++i) { st = st*1664525u+1013904223u;
        hX[i] = 0.5f + (float)((st>>9)&0x3FFu)/1024.0f; hY[i] = (float)(i & 255)*0.001f; }
    for (size_t i = 0; i < sA; ++i) { st = st*1664525u+1013904223u;
        hA[i] = 0.5f + (float)((st>>8)&0xFFFFu)/65536.0f; }
    for (size_t i = 0; i < sB; ++i) { st = st*1664525u+1013904223u;
        hB[i] = 0.5f + (float)((st>>8)&0xFFFFu)/65536.0f; }
    // A single cycle of stride 1572869 (a prime) over CHASE_LEN slots: every hop
    // lands in a different 32 B sector and a different 128 B line.
    { const long long P = 25165843LL;
      for (long long i = 0; i < CHASE_LEN; ++i) hN[i] = (int)((i + P) % CHASE_LEN); }

    CHECK(cudaMalloc(&dX, sizeof(float)*NELEM));
    CHECK(cudaMalloc(&dY, sizeof(float)*NELEM));
    CHECK(cudaMalloc(&dPart, sizeof(float)*240));
    CHECK(cudaMalloc(&dA, sizeof(float)*sA));
    CHECK(cudaMalloc(&dB, sizeof(float)*sB));
    CHECK(cudaMalloc(&dC, sizeof(float)*sC));
    CHECK(cudaMalloc(&dNxt, sizeof(int)*CHASE_LEN));
    CHECK(cudaMalloc(&dChaseOut, sizeof(float)*CHASE_THR));
    CHECK(cudaMalloc(&dSink, sizeof(float)*4));
    CHECK(cudaMalloc(&dWarm, (size_t)256*1024*1024));
    CHECK(cudaMemcpy(dX, hX, sizeof(float)*NELEM, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dY, hY, sizeof(float)*NELEM, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dA, hA, sizeof(float)*sA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, sizeof(float)*sB, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dNxt, hN, sizeof(int)*CHASE_LEN, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dWarm, 1, (size_t)256*1024*1024));

    // ---------------------------------------------------------------- A
    // Ceilings. Memory first, after a streaming warm-up; then on-chip, after a
    // compute warm-up. They are not competing configurations, so they are not
    // in the same rotated sweep -- spec SS12 rule 1 is about competitors.
    printf("-- A. ceilings measured in this process ------------------------------\n");
    const size_t wn = (size_t)256*1024*1024/16;
    { cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0; CHECK(cudaEventRecord(w0));
      while (el < 1500.f) { warmStream<<<640,256>>>(dWarm,dSink,wn);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); }
    double bStream = 1e30;
    for (int s = 0; s < 4; ++s) {
        cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        CHECK(cudaEventRecord(a));
        for (int i = 0; i < 12; ++i) warmStream<<<640,256>>>(dWarm,dSink,wn);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        float ms; CHECK(cudaEventElapsedTime(&ms,a,b));
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
        if (ms/12 < bStream) bStream = ms/12;
    }
    const double CEIL_DRAM = (double)(256.0*1024*1024)/(bStream*1e-3)/1e9;

    { cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0; CHECK(cudaEventRecord(w0));
      while (el < 500.f) { warmFfma<<<480,128>>>(dSink,2000);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); }
    double bF = 1e30, bS = 1e30;
    for (int s = 0; s < 4; ++s) {
        cudaEvent_t a,b; float ms;
        CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        CHECK(cudaEventRecord(a));
        for (int i = 0; i < 6; ++i) warmFfma<<<480,128>>>(dSink,8192);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        CHECK(cudaEventElapsedTime(&ms,a,b)); if (ms/6 < bF) bF = ms/6;
        CHECK(cudaEventRecord(a));
        for (int i = 0; i < 6; ++i) smemProbe<<<480,128>>>(dSink,4096);
        CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
        CHECK(cudaEventElapsedTime(&ms,a,b)); if (ms/6 < bS) bS = ms/6;
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    }
    CHECK(cudaGetLastError());
    const double CEIL_FP32 = 480.0*128.0*8192.0*8.0*2.0/(bF*1e-3)/1e9;
    const double CEIL_SMEM = 480.0*128.0*4096.0*SP_W*4.0/(bS*1e-3)/1e9;
    const double CEIL_L2   = 1305.0;    // Module 14's measured figure; not re-measured here

    printf("  DRAM read           %10.1f GB/s   (%.0f%% of the 432 GB/s pin peak)\n",
           CEIL_DRAM, 100.0*CEIL_DRAM/DRAM_PIN_PEAK_GBS);
    printf("  L2 read             %10.1f GB/s   (recorded, Modules 11/14)\n", CEIL_L2);
    printf("  shared read, scalar %10.1f GB/s\n", CEIL_SMEM);
    printf("  FP32 FFMA           %10.1f GFLOP/s\n", CEIL_FP32);
    printf("  machine balance     %10.2f FLOP/byte  (DRAM ridge)\n", CEIL_FP32/CEIL_DRAM);
    printf("  shared ridge        %10.2f FLOP/byte\n\n", CEIL_FP32/CEIL_SMEM);

    // ---------------------------------------------------------------- B
    // The ledger. Every number here is counted by hand from the kernel source,
    // before anything is timed.
    //
    // THREE byte columns, because there are three places bytes move:
    //   dram  compulsory traffic across the pins -- each distinct element once
    //   req   operand bytes a memory INSTRUCTION asks for, in whatever address
    //         space: an LDG that hits in L1, or an LDS. Modules 16 and 17
    //         measured the same law for both, so they share one ceiling.
    //   (L2 sits between them and cannot be separated without ncu -- M23.)
    typedef struct {
        const char *name; void (*run)(void);
        double flops;
        double dramBytes;
        double reqBytes;
        double flopPerInstr;     // FLOPs per warp-instruction, from the SASS
        const char *note;
    } Kern;

    const double FL_GEMM = 2.0*M_DIM*N_DIM*K_DIM;
    const double NMNK    = (double)M_DIM*N_DIM*K_DIM;
    Kern k[6] = {
      { "1 triad  (M11)",  kTriad,
        2.0*NELEM, 12.0*NELEM, 12.0*NELEM, 64.0*1.0/15.0,
        "read x, read y, write y: 3N, not 2N" },
      { "2 reduce (M12)",  kReduce,
        1.0*NELEM, 4.0*NELEM, 4.0*NELEM, 32.0*1.0/9.0,
        "1 add per element read" },
      { "3 gemmN  (M16)",  kGemmN,
        FL_GEMM, 4.0*((double)sA+sB+sC), 8.0*NMNK, 64.0*16.0/87.0,
        "724x between the two byte columns; 8 B per FMA is still REQUESTED" },
      { "4 gemmT  (M17)",  kGemmT,
        FL_GEMM, 4.0*((double)sA+sB+sC), 8.0*NMNK, 64.0*16.0/66.0,
        "tiling changed the opcode LDG->LDS, not the 8 B per FMA" },
      { "5 gemmR  (M18)",  kGemmR,
        FL_GEMM, 4.0*((double)sA+sB+sC), 4.0*(8.0+4.0)/(8.0*4.0)*NMNK,
        64.0*256.0/365.0,
        "(TM+TN)/(TM*TN) floats per FMA = 1.5 B, not 8" },
      { "6 chase  (M1/M4)", kChase,
        (double)CHASE_THR*CHASE_STEPS, 32.0*(double)CHASE_THR*CHASE_STEPS,
        4.0*(double)CHASE_THR*CHASE_STEPS, 32.0*16.0/84.0,
        "4 useful bytes per hop, a whole 32 B sector moved" },
    };

    const double CEIL_ISSUE = CEIL_FP32/64.0;   // G warp-instructions per second

    printf("-- B. the ledger and the prediction, before any timing ---------------\n");
    printf("  AI is FLOP per byte AT THAT LEVEL. p:issue is the plateau implied by\n"
           "  the kernel's own FLOP-per-warp-instruction against %.1f G instr/s.\n\n",
           CEIL_ISSUE);
    printf("  %-16s %9s %9s %9s %9s %9s %10s\n", "kernel",
           "AI(dram)", "AI(req)", "p:dram", "p:onchip", "p:issue", "PREDICT");
    double pred[6]; const char *predLvl[6];
    for (int i = 0; i < 6; ++i) {
        double aiD = k[i].flops/k[i].dramBytes;
        double aiR = k[i].flops/k[i].reqBytes;
        double pD  = attain(aiD, CEIL_DRAM, CEIL_FP32);
        double pR  = attain(aiR, CEIL_SMEM, CEIL_FP32);
        double pI  = CEIL_ISSUE * k[i].flopPerInstr;
        double p = pD; const char *lv = "DRAM";
        if (pR < p) { p = pR; lv = "on-chip"; }
        if (pI < p) { p = pI; lv = "issue";   }
        if (CEIL_FP32 < p) { p = CEIL_FP32; lv = "compute"; }
        pred[i] = p; predLvl[i] = lv;
        printf("  %-16s %9.3f %9.3f %9.1f %9.1f %9.1f %10.1f <- %s\n",
               k[i].name, aiD, aiR, pD, pR, pI, p, lv);
        printf("  %-16s   %s\n", "", k[i].note);
    }
    printf("\n  Note the row that is NOT here: an L2 roofline. Taking the requested\n"
           "  bytes against the %.0f GB/s L2 figure would predict %.0f GFLOP/s for\n",
           CEIL_L2, k[2].flops/k[2].reqBytes*CEIL_L2);
    printf("  gemmN, and that is not a ceiling -- it assumes every request misses\n"
           "  L1, which is exactly what does not happen. A level's roofline binds\n"
           "  only if the traffic you counted actually crosses that level.\n");

    // ---------------------------------------------------------------- C
    printf("\n-- C. measurement (one rotated sweep, SWEEPS >= NCFG) ----------------\n");
    int iters[6]; double best[6];
    for (int i = 0; i < 6; ++i) {
        double t = timeFn(k[i].run, 1);
        int n = (int)(10.0/(t > 1e-3 ? t : 1e-3)); if (n < 3) n = 3; if (n > 64) n = 64;
        iters[i] = n; best[i] = 1e30;
    }
    CHECK(cudaGetLastError());
    for (int s = 0; s < 6; ++s)
        for (int q = 0; q < 6; ++q) {
            int p = (q+s)%6;
            double t = timeFn(k[p].run, iters[p]);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    printf("  %-16s %10s %10s %10s %9s %9s\n", "kernel",
           "ms", "GFLOP/s", "pred GF/s", "meas/pred", "%FP32");
    for (int i = 0; i < 6; ++i) {
        double g = k[i].flops/(best[i]*1e-3)/1e9;
        printf("  %-16s %10.4f %10.2f %10.2f %9.3f %8.2f%%\n",
               k[i].name, best[i], g, pred[i], g/pred[i], 100.0*g/CEIL_FP32);
    }

    // ---------------------------------------------------------------- D
    printf("\n-- D. reconciliation --------------------------------------------------\n");
    for (int i = 0; i < 6; ++i) {
        double g = k[i].flops/(best[i]*1e-3)/1e9;
        double r = g/pred[i];
        double achievedDram = k[i].dramBytes/(best[i]*1e-3)/1e9;
        printf("  %-16s pred %-8s", k[i].name, predLvl[i]);
        if (r > 0.75 && r < 1.30)      printf(" AT THE ROOF (%.2fx).", r);
        else if (r >= 1.30)            printf(" ABOVE the roof (%.2fx): the ledger is wrong.", r);
        else if (r > 0.25)             printf(" below (%.2fx).", r);
        else                           printf(" FAR below (%.4fx).", r);
        printf("  pins: %.2f GB/s (%.2f%%)\n",
               achievedDram, 100.0*achievedDram/CEIL_DRAM);
    }
    printf("\n  What each row is telling you.\n");
    printf("  * triad and reduce are on the DRAM roof. Nothing is left on the table;\n"
           "    the only remaining move is to delete traffic (Module 11's fusion).\n");
    printf("    triad lands a little lower than reduce, reproducibly, and the reason\n"
           "    is the ceiling rather than the kernel: the ceiling here is measured\n"
           "    with a READ-ONLY stream, and triad reads two arrays and writes one.\n"
           "    A mixed read/write stream does not reach a pure-read ceiling. If that\n"
           "    distinction matters to you, measure the ceiling with the access mix\n"
           "    your kernel actually has -- Module 15 made exactly this argument for\n"
           "    the transpose and used a matched COPY as its denominator.\n");
    printf("  * gemmN's COMPULSORY arithmetic intensity is %.0f FLOP/byte, far right\n",
           k[2].flops/k[2].dramBytes);
    printf("    of the %.1f FLOP/byte ridge. A two-axis roofline reads that as\n",
           CEIL_FP32/CEIL_DRAM);
    printf("    COMPUTE BOUND and predicts the FP32 ceiling. It runs at %.1f%% of it.\n",
           100.0*(k[2].flops/(best[2]*1e-3)/1e9)/CEIL_FP32);
    printf("    The compulsory model is not this kernel's model: the kernel REQUESTS\n"
           "    8 B per 2 FLOP and the caches, not the kernel, keep that off the pins\n"
           "    (measured here: only %.2f GB/s crosses them).\n",
           k[2].dramBytes/(best[2]*1e-3)/1e9);
    printf("  * gemmN, gemmT and gemmR are all on the SAME on-chip ceiling, and that\n"
           "    is the module's point. They differ only in AI(req): %.2f, %.2f, %.2f\n",
           k[2].flops/k[2].reqBytes, k[3].flops/k[3].reqBytes, k[4].flops/k[4].reqBytes);
    printf("    FLOP/byte, predicting %.0f / %.0f / %.0f GFLOP/s and measuring\n",
           pred[2], pred[3], pred[4]);
    printf("    %.0f / %.0f / %.0f. Tiling (M17) moved the opcode from LDG to LDS and\n",
           k[2].flops/(best[2]*1e-3)/1e9, k[3].flops/(best[3]*1e-3)/1e9,
           k[4].flops/(best[4]*1e-3)/1e9);
    printf("    left AI alone; register tiling (M18) moved AI, and the throughput\n"
           "    moved with it. A two-axis roofline has no axis to draw any of this on.\n");
    printf("  * chase is the honest failure of the model. Its roofline says %.1f\n", pred[5]);
    printf("    GFLOP/s and it delivers %.4f, a factor of %.0f. Nothing is saturated.\n",
           k[5].flops/(best[5]*1e-3)/1e9, pred[5]/(k[5].flops/(best[5]*1e-3)/1e9));
    printf("    Little's Law: the memory system needs bandwidth x latency =\n"
           "    %.0f kB in flight to run at the ceiling. One warp with ONE dependent\n",
           CEIL_DRAM*1e9*(575.0/1.7e9)/1024.0);
    printf("    load per lane supplies %d x 32 B = %.2f kB, i.e. %.2f%% of it -- and\n",
           CHASE_THR, CHASE_THR*32.0/1024.0,
           100.0*(CHASE_THR*32.0)/(CEIL_DRAM*1e9*(575.0/1.7e9)));
    printf("    %.2f%% of %.1f GB/s is %.2f GB/s, which is what the pins column says.\n",
           100.0*(CHASE_THR*32.0)/(CEIL_DRAM*1e9*(575.0/1.7e9)), CEIL_DRAM,
           k[5].dramBytes/(best[5]*1e-3)/1e9);
    printf("    Being far below every ceiling is a CONCURRENCY fact, not a bandwidth\n"
           "    fact. The roofline cannot see it. Modules 19 and 20 own it.\n");

    // ---------------------------------------------------------------- validation
    printf("\n-- validation (second, untimed pass) ----------------------------------\n");
    int ok = 1;

    CHECK(cudaMemcpy(dY, hY, sizeof(float)*NELEM, cudaMemcpyHostToDevice));
    kTriad(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    { float *t = (float*)malloc(sizeof(float)*NELEM);
      CHECK(cudaMemcpy(t, dY, sizeof(float)*NELEM, cudaMemcpyDeviceToHost));
      double worst = 0;
      for (int i = 0; i < NELEM; i += 997) {
          double ref = 2.0*hX[i] + hY[i];
          double e = fabs(t[i]-ref)/fmax(1.0, fabs(ref));
          if (e > worst) worst = e; }
      printf("  triad   worst relative error %.3e  %s\n", worst, worst < 1e-6 ? "ok":"FAIL");
      if (!(worst < 1e-6)) ok = 0; free(t); }

    kReduce(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    { float *p = (float*)malloc(sizeof(float)*240);
      CHECK(cudaMemcpy(p, dPart, sizeof(float)*240, cudaMemcpyDeviceToHost));
      double got = 0; for (int i = 0; i < 240; ++i) got += p[i];
      double ref = 0; for (int i = 0; i < NELEM; ++i) ref += (double)hX[i];
      double e = fabs(got-ref)/ref;
      printf("  reduce  relative error       %.3e  %s\n", e, e < 1e-5 ? "ok":"FAIL");
      if (!(e < 1e-5)) ok = 0; free(p); }

    { const char *nm[3] = { "gemmN", "gemmT", "gemmR" };
      void (*fn[3])(void) = { kGemmN, kGemmT, kGemmR };
      for (int j = 0; j < 3; ++j) {
          float *poison = (float*)malloc(sizeof(float)*sC);
          for (size_t i = 0; i < sC; ++i) poison[i] = std::numeric_limits<float>::infinity();
          CHECK(cudaMemcpy(dC, poison, sizeof(float)*sC, cudaMemcpyHostToDevice));
          free(poison);
          fn[j](); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
          CHECK(cudaMemcpy(hC, dC, sizeof(float)*sC, cudaMemcpyDeviceToHost));
          double w; int good = gemmQuickValidate(M_DIM,N_DIM,K_DIM,hA,hB,hC,&w);
          printf("  %-7s Freivalds headroom     %.4f      %s\n", nm[j], w, good ? "ok":"FAIL");
          if (!good) ok = 0;
      } }

    kChase(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hChase, dChaseOut, sizeof(float)*CHASE_THR, cudaMemcpyDeviceToHost));
    { int bad = 0;
      for (int t = 0; t < CHASE_THR; ++t) {
          int p = t; double acc = 0;
          for (int s = 0; s < CHASE_STEPS; ++s) { p = hN[p]; acc += (double)(p & 1023); }
          if (fabs(acc - (double)hChase[t]) > 1.0) ++bad; }
      printf("  chase   mismatched threads    %d          %s\n", bad, bad==0?"ok":"FAIL");
      if (bad) ok = 0; }

    CHECK(cudaFree(dX)); CHECK(cudaFree(dY)); CHECK(cudaFree(dPart));
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    CHECK(cudaFree(dNxt)); CHECK(cudaFree(dChaseOut));
    CHECK(cudaFree(dSink)); CHECK(cudaFree(dWarm));
    free(hX); free(hY); free(hA); free(hB); free(hC); free(hN); free(hChase);

    printf("\nOVERALL: %s\n", ok ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return ok ? 0 : 1;
}
