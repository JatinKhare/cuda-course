// =============================================================================
// Module 21 / Exercise 2 — classify six kernels, predict, measure, reconcile.
//
// GOAL : Six kernels you have already written. For each one: count the bytes
//        and the FLOPs, compute the arithmetic intensity at every level of the
//        hierarchy, place it on the roofline, and commit to a prediction --
//        all BEFORE the harness times anything. Then measure and reconcile.
//
//        The kernels, verbatim from the modules that built them:
//          1 triad   in-place SAXPY  y += a*x        Module 11
//          2 reduce  grid-stride sum, rung v6        Module 12
//          3 gemmN   naive GEMM                      Module 16
//          4 gemmT   16x16 shared-memory tiled GEMM  Module 17
//          5 gemmR   8x4 register-tiled GEMM         Module 18
//          6 chase   dependent pointer walk          Modules 1 and 4
//
//        One of the six is far below every ceiling in the model, and the model
//        has no axis that explains it. Finding out which, and why, is TODO 5.
//
// WHAT TO FILL IN
//   TODO 1  fillLedger()        -- FLOPs and both byte counts, for all six
//   TODO 2  classify()          -- the roofline's own verdict
//   TODO 3  PRED_LEVEL[6]       -- what you think actually limits each kernel
//   TODO 4  PRED_BUCKET[6]      -- predicted fraction of the FP32 ceiling
//   TODO 5  bytesInFlight*()    -- Little's Law for the outlier   (DESIGN)
//
// SCORING: 9 points. OVERALL: PASS requires all nine.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
// RUN  : exercise02.exe
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
#define FP32_LANES_PER_SM 128
#define DRAM_LATENCY_CYCLES 575.0        // Module 4, dependent-load latency
#define SECTOR_BYTES 32.0

#define NELEM   (32*1024*1024)
#define M_DIM 1027
#define N_DIM 2053
#define K_DIM  769
#define CHASE_LEN   (1<<26)
#define CHASE_THR   32
#define CHASE_STEPS 3000

// Levels a kernel can be limited by.
#define LVL_DRAM    0
#define LVL_ONCHIP  1
#define LVL_ISSUE   2
#define LVL_COMPUTE 3
#define LVL_LATENCY 4

// =============================================================================
// The six kernels. All six are reproduced from earlier modules and are NOT the
// subject of this exercise -- the analysis is.
// =============================================================================

// 1. Module 11, in-place SAXPY.
__global__ void triad(float *y, const float * __restrict__ x, float a, size_t n)
{
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    const size_t s = gridDim.x*(size_t)blockDim.x;
    for (; i < n; i += s) y[i] = fmaf(a, x[i], y[i]);
}

// 2. Module 12, reduction rung v6.
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

// 3. Module 16, naive GEMM.
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

// 4. Module 17, shared-memory tiled GEMM, T = 16.
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

// 5. Module 18, register-tiled GEMM, TM = 8, TN = 4.
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
            int kk = idx/BN, n = idx%BN;
            Bs[kk][n] = (kt+kk < K && colBase+n < N)
                      ? B[(size_t)(kt+kk)*N + colBase + n] : 0.0f;
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

// 6. Modules 1 and 4, the dependent pointer walk. One warp on the whole GPU.
__global__ void chase(const int * __restrict__ nxt, float *out, int steps, int nthr)
{
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= nthr) return;
    int p = t; float acc = 0.0f;
    for (int s = 0; s < steps; ++s) { p = nxt[p]; acc += (float)(p & 1023); }
    out[t] = acc;
}

// Ceiling probes and warm-up kernels.
__global__ void streamRead(const float4 * __restrict__ s, float *o, size_t n)
{
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    float4 a = make_float4(0,0,0,0);
    for (; i < n; i += gridDim.x*(size_t)blockDim.x) {
        float4 v = s[i]; a.x+=v.x; a.y+=v.y; a.z+=v.z; a.w+=v.w; }
    if (a.x == 1e30f) o[0] = a.x+a.y+a.z+a.w;
}
__global__ void ffmaProbe(float *o, int iters)
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
    if (acc == 1e30f) sink[0] = acc + s[0];
}

// =============================================================================
// TODO 1 — the traffic ledger.
// =============================================================================
typedef struct { double flops, dramBytes, reqBytes, flopPerInstr; } Ledger;

static void fillLedger(Ledger *L)
{
    const double NM = (double)M_DIM*N_DIM*K_DIM;
    const double sA = (double)M_DIM*K_DIM, sB = (double)K_DIM*N_DIM, sC = (double)M_DIM*N_DIM;
    (void)NM; (void)sA; (void)sB; (void)sC;

    // TODO 1: for each of the six kernels fill in
    //   .flops      the useful floating-point operations the kernel performs
    //   .dramBytes  COMPULSORY traffic across the pins: every distinct element
    //               that must be read at least once, plus every element
    //               written, plus -- think about this one -- anything that is
    //               read AND written. Count what the hardware must move, not
    //               what the source text mentions.
    //   .reqBytes   the operand bytes the kernel's memory INSTRUCTIONS ask
    //               for, in whatever address space they live in. A value that
    //               is requested a thousand times is requested a thousand
    //               times whether or not a cache answers.
    //
    // Three of the six have dramBytes == reqBytes. Three do not, and the gap
    // is the whole subject of Modules 16 to 18. One of the six requests fewer
    // bytes than it moves, which is the reverse of all the others.
    //
    // All six kernels launch with alpha = 1, beta = 0.
    // YOUR CODE HERE

    // FLOPs per warp-instruction, from `cuobjdump -sass` loop bodies. GIVEN --
    // you do not have to derive these, but you should know what they mean:
    // one FFMA warp-instruction retires 32 lanes x 2 FLOP = 64 FLOP, and one
    // FADD warp-instruction retires 32.
    L[0].flopPerInstr = 64.0*1.0/15.0;    // 15 instructions,  1 FFMA
    L[1].flopPerInstr = 32.0*1.0/9.0;     //  9 instructions,  1 FADD
    L[2].flopPerInstr = 64.0*16.0/87.0;   // 87 instructions, 16 FFMA
    L[3].flopPerInstr = 64.0*16.0/66.0;   // 66 instructions, 16 FFMA
    L[4].flopPerInstr = 64.0*256.0/365.0; // 365 instructions, 256 FFMA
    L[5].flopPerInstr = 32.0*16.0/84.0;   // 84 instructions, 16 FADD
}

// =============================================================================
// TODO 2 — classify(). The roofline's own verdict.
// =============================================================================
static int classify(double flops, double dramBytes, double reqBytes,
                    double flopPerInstr, double ceilDram, double ceilOnChip,
                    double ceilIssueGI, double ceilFp32, double *predGF)
{
    // TODO 2: return the level that binds, and write the attainable GFLOP/s
    //         into *predGF.
    //
    // Four candidate plateaux:
    //   LVL_DRAM    the DRAM arithmetic intensity against ceilDram  (GB/s)
    //   LVL_ONCHIP  the request-level intensity against ceilOnChip  (GB/s)
    //   LVL_ISSUE   flopPerInstr against ceilIssueGI (G warp-instructions/s)
    //   LVL_COMPUTE the flat FP32 plateau, ceilFp32 (GFLOP/s)
    // Watch the units: GB/s times FLOP/byte is GFLOP/s, and G instructions/s
    // times FLOP/instruction is also GFLOP/s.
    //
    // Return a negative number if this TODO has not been filled in -- the
    // harness uses that to detect it.
    // YOUR CODE HERE
    (void)flops; (void)dramBytes; (void)reqBytes; (void)flopPerInstr;
    (void)ceilDram; (void)ceilOnChip; (void)ceilIssueGI; (void)ceilFp32;
    *predGF = -1.0;
    return -1;
}

// =============================================================================
// TODO 3 — what you believe ACTUALLY limits each kernel.
//
// This is not required to agree with classify(). classify() knows about four
// ceilings; the hardware knows about more. LVL_LATENCY is available and means
// "nothing is saturated; the kernel is waiting on a dependence chain". Use it
// if and only if you think it is true, and be ready to defend it in TODO 5.
//
// Order: triad, reduce, gemmN, gemmT, gemmR, chase.
// =============================================================================
static const int PRED_LEVEL[6] = {
    // TODO 3: one of LVL_DRAM / LVL_ONCHIP / LVL_ISSUE / LVL_COMPUTE /
    //         LVL_LATENCY per kernel.
    // YOUR CODE HERE
    -1, -1, -1, -1, -1, -1
};

// =============================================================================
// TODO 4 — predicted achieved fraction of the MEASURED FP32 ceiling, as a
// bucket. The harness measures the ceiling in the same run, so you are
// predicting a ratio, not an absolute.
//
//   bucket 1: < 1%     2: 1-3%     3: 3-12%     4: 12-25%     5: > 25%
//
// Order: triad, reduce, gemmN, gemmT, gemmR, chase. All six must be right.
// =============================================================================
static const int PRED_BUCKET[6] = {
    // TODO 4: YOUR CODE HERE
    0, 0, 0, 0, 0, 0
};

// =============================================================================
// TODO 5 — DESIGN. Little's Law for the chase kernel.
// =============================================================================
// One of the six kernels will come out far below every ceiling in the model.
// The roofline has nothing to say about it, because the roofline assumes you
// can keep the memory system busy and says nothing about whether you can.
// Little's Law (Module 1) does: a pipeline of latency L running at bandwidth B
// must have B x L bytes outstanding at all times.
//
// Write the two halves. The harness compares their ratio against the fraction
// of the DRAM ceiling the outlier actually achieves; if your reasoning is
// right the two agree to well within a factor of four, which is as much as
// anyone should ask of a Little's Law estimate.
//
// Return a negative number until this TODO is filled in.
static double bytesInFlightNeeded(double bwGBs, double latCycles, double clockGHz)
{
    // TODO 5a: bytes the memory system must have outstanding to run at bwGBs,
    //          given a round trip of latCycles at clockGHz. Mind the units --
    //          GB/s and GHz are both 1e9, and they cancel.
    // YOUR CODE HERE
    (void)bwGBs; (void)latCycles; (void)clockGHz;
    return -1.0;
}
static double bytesInFlightSupplied(double nThreads, double sectorBytes)
{
    // TODO 5b: bytes the outlier kernel actually has outstanding. Look at the
    //          kernel: how many memory requests can one thread have in flight
    //          at once, and how many bytes does the hardware move for each?
    // YOUR CODE HERE
    (void)nThreads; (void)sectorBytes;
    return -1.0;
}

// =============================================================================
// Harness
// =============================================================================
static unsigned fnvNums(const double *v, int n, double scale)
{ char buf[64]; unsigned h = 2166136261u;
  for (int i = 0; i < n; ++i) { snprintf(buf, sizeof buf, "%lld", (long long)llround(v[i]*scale));
      for (char *p = buf; *p; ++p) { h ^= (unsigned char)*p; h *= 16777619u; } }
  return h; }
static unsigned fnvInts(const int *v, int n)
{ char buf[32]; unsigned h = 2166136261u;
  for (int i = 0; i < n; ++i) { snprintf(buf, sizeof buf, "%d", v[i]);
      for (char *p = buf; *p; ++p) { h ^= (unsigned char)*p; h *= 16777619u; } }
  return h; }

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
static float *dX,*dY,*dPart,*dA,*dB,*dC,*dSink,*dChaseOut; static int *dNxt;
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

static int gemmQuickValidate(int M,int N,int K,const float*hA,const float*hB,
                             const float*hC,double*worst)
{
    const double u = ldexp(1.0,-24);
    const double gammaK = (double)K*u/(1.0-(double)K*u);
    for (size_t i = 0; i < (size_t)M*N; ++i) if (!isfinite(hC[i])) { *worst=1e30; return 0; }
    double *v=(double*)malloc(sizeof(double)*N), *Bv=(double*)malloc(sizeof(double)*K),
           *aB=(double*)malloc(sizeof(double)*K);
    unsigned s=0xB16B00B5u;
    for (int j=0;j<N;++j){ s=s*1664525u+1013904223u; v[j]=0.5+(double)((s>>8)&0xFFFFu)/65536.0; }
    for (int k=0;k<K;++k){ double t=0,tb=0;
        for (int j=0;j<N;++j){ double bb=hB[(size_t)k*N+j]; t+=bb*v[j]; tb+=fabs(bb)*v[j]; }
        Bv[k]=t; aB[k]=tb; }
    double w=0;
    for (int i=0;i<M;++i){ double y=0,yb=0;
        for (int k=0;k<K;++k){ double a=hA[(size_t)i*K+k]; y+=a*Bv[k]; yb+=fabs(a)*aB[k]; }
        double got=0; for (int j=0;j<N;++j) got+=(double)hC[(size_t)i*N+j]*v[j];
        double tol=(gammaK+4.0*u)*yb, ratio=(tol>0)?fabs(got-y)/tol:0.0;
        if (ratio>w) w=ratio; }
    free(v); free(Bv); free(aB); *worst=w; return w<=1.0;
}

static int bucketOf(double pct)
{ if (pct < 1.0) return 1; if (pct < 3.0) return 2;
  if (pct < 12.0) return 3; if (pct < 25.0) return 4; return 5; }
static const char *lvlName(int l)
{ static const char *n[5] = {"DRAM","on-chip","issue","compute","latency"};
  return (l>=0 && l<5) ? n[l] : "?"; }

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 21 / Exercise 2 - classify, predict, reconcile ===\n\n");

    Ledger L[6];
    for (int i = 0; i < 6; ++i) { L[i].flops = L[i].dramBytes = L[i].reqBytes = -1.0;
                                  L[i].flopPerInstr = -1.0; }
    fillLedger(L);
    for (int i = 0; i < 6; ++i)
        if (!(L[i].flops > 0 && L[i].dramBytes > 0 && L[i].reqBytes > 0)) {
            printf("Set TODO 1 first.\n"); return 0; }
    { double dummy; if (classify(1,1,1,1,1,1,1,1,&dummy) < 0) {
          printf("Set TODO 2 first.\n"); return 0; } }
    if (PRED_LEVEL[0] < 0 || PRED_BUCKET[0] < 1) { printf("Set TODO 3 and 4 first.\n"); return 0; }
    if (bytesInFlightNeeded(400.0,575.0,1.8) < 0 || bytesInFlightSupplied(32,32) < 0) {
        printf("Set TODO 5 first.\n"); return 0; }

    const size_t sA=(size_t)M_DIM*K_DIM, sB=(size_t)K_DIM*N_DIM, sC=(size_t)M_DIM*N_DIM;
    float *hX=(float*)malloc(sizeof(float)*NELEM), *hY=(float*)malloc(sizeof(float)*NELEM);
    float *hA=(float*)malloc(sizeof(float)*sA), *hB=(float*)malloc(sizeof(float)*sB);
    float *hC=(float*)malloc(sizeof(float)*sC);
    int   *hN=(int*)malloc(sizeof(int)*CHASE_LEN);
    float *hCh=(float*)malloc(sizeof(float)*CHASE_THR);
    unsigned st=12345u;
    for (int i=0;i<NELEM;++i){ st=st*1664525u+1013904223u;
        hX[i]=0.5f+(float)((st>>9)&0x3FFu)/1024.0f; hY[i]=(float)(i&255)*0.001f; }
    for (size_t i=0;i<sA;++i){ st=st*1664525u+1013904223u;
        hA[i]=0.5f+(float)((st>>8)&0xFFFFu)/65536.0f; }
    for (size_t i=0;i<sB;++i){ st=st*1664525u+1013904223u;
        hB[i]=0.5f+(float)((st>>8)&0xFFFFu)/65536.0f; }
    { const long long P=25165843LL;
      for (long long i=0;i<CHASE_LEN;++i) hN[i]=(int)((i+P)%CHASE_LEN); }

    CHECK(cudaMalloc(&dX,sizeof(float)*NELEM)); CHECK(cudaMalloc(&dY,sizeof(float)*NELEM));
    CHECK(cudaMalloc(&dPart,sizeof(float)*240));
    CHECK(cudaMalloc(&dA,sizeof(float)*sA)); CHECK(cudaMalloc(&dB,sizeof(float)*sB));
    CHECK(cudaMalloc(&dC,sizeof(float)*sC));
    CHECK(cudaMalloc(&dNxt,sizeof(int)*CHASE_LEN));
    CHECK(cudaMalloc(&dChaseOut,sizeof(float)*CHASE_THR));
    CHECK(cudaMalloc(&dSink,sizeof(float)*4));
    CHECK(cudaMalloc(&dWarm,(size_t)256*1024*1024));
    CHECK(cudaMemcpy(dX,hX,sizeof(float)*NELEM,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dY,hY,sizeof(float)*NELEM,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dA,hA,sizeof(float)*sA,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB,hB,sizeof(float)*sB,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dNxt,hN,sizeof(int)*CHASE_LEN,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dWarm,1,(size_t)256*1024*1024));

    // ---- ceilings ----------------------------------------------------------
    const size_t wn=(size_t)256*1024*1024/16;
    { cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0; CHECK(cudaEventRecord(w0));
      while (el<1500.f){ streamRead<<<640,256>>>(dWarm,dSink,wn);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); }
    double bStream=1e30;
    for (int s=0;s<4;++s){ cudaEvent_t a,b; float ms;
      CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b)); CHECK(cudaEventRecord(a));
      for (int i=0;i<12;++i) streamRead<<<640,256>>>(dWarm,dSink,wn);
      CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
      CHECK(cudaEventElapsedTime(&ms,a,b));
      CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
      if (ms/12<bStream) bStream=ms/12; }
    const double CEIL_DRAM = 256.0*1024*1024/(bStream*1e-3)/1e9;

    { cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0; CHECK(cudaEventRecord(w0));
      while (el<500.f){ ffmaProbe<<<480,128>>>(dSink,2000);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); }
    double bF=1e30,bS=1e30;
    for (int s=0;s<4;++s){ cudaEvent_t a,b; float ms;
      CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
      CHECK(cudaEventRecord(a));
      for (int i=0;i<6;++i) ffmaProbe<<<480,128>>>(dSink,8192);
      CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
      CHECK(cudaEventElapsedTime(&ms,a,b)); if (ms/6<bF) bF=ms/6;
      CHECK(cudaEventRecord(a));
      for (int i=0;i<6;++i) smemProbe<<<480,128>>>(dSink,4096);
      CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
      CHECK(cudaEventElapsedTime(&ms,a,b)); if (ms/6<bS) bS=ms/6;
      CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); }
    CHECK(cudaGetLastError());
    const double CEIL_FP32 = 480.0*128.0*8192.0*8.0*2.0/(bF*1e-3)/1e9;
    const double CEIL_SMEM = 480.0*128.0*4096.0*SP_W*4.0/(bS*1e-3)/1e9;
    const double CEIL_ISSUE= CEIL_FP32/64.0;
    const double CLOCK_GHZ = CEIL_FP32/(SM_COUNT*(double)FP32_LANES_PER_SM*2.0);

    printf("-- ceilings measured in this process ---------------------------------\n");
    printf("  DRAM %8.1f GB/s   on-chip operand fetch %8.1f GB/s\n", CEIL_DRAM, CEIL_SMEM);
    printf("  FP32 %8.1f GFLOP/s   issue %6.1f G warp-instr/s   clock %.3f GHz\n",
           CEIL_FP32, CEIL_ISSUE, CLOCK_GHZ);
    printf("  DRAM ridge %.2f FLOP/byte   on-chip ridge %.2f FLOP/byte\n\n",
           CEIL_FP32/CEIL_DRAM, CEIL_FP32/CEIL_SMEM);

    // ---- the prediction table ---------------------------------------------
    const char *nm[6] = { "1 triad ", "2 reduce", "3 gemmN ", "4 gemmT ", "5 gemmR ",
                          "6 chase " };
    double pred[6]; int lvl[6];
    printf("-- your classification, before anything is timed ---------------------\n");
    printf("  %-9s %9s %9s %10s %9s %8s %8s\n", "kernel","AI(dram)","AI(req)",
           "roofline","predGF/s","you say","bucket");
    for (int i = 0; i < 6; ++i) {
        lvl[i] = classify(L[i].flops, L[i].dramBytes, L[i].reqBytes, L[i].flopPerInstr,
                          CEIL_DRAM, CEIL_SMEM, CEIL_ISSUE, CEIL_FP32, &pred[i]);
        printf("  %-9s %9.3f %9.3f %10s %9.1f %8s %8d\n", nm[i],
               L[i].flops/L[i].dramBytes, L[i].flops/L[i].reqBytes,
               lvlName(lvl[i]), pred[i], lvlName(PRED_LEVEL[i]), PRED_BUCKET[i]);
    }

    // ---- measurement -------------------------------------------------------
    void (*run[6])(void) = { kTriad, kReduce, kGemmN, kGemmT, kGemmR, kChase };
    int iters[6]; double best[6];
    for (int i = 0; i < 6; ++i) {
        double t = timeFn(run[i],1);
        int n = (int)(10.0/(t>1e-3?t:1e-3)); if (n<3) n=3; if (n>64) n=64;
        iters[i]=n; best[i]=1e30; }
    CHECK(cudaGetLastError());
    for (int s = 0; s < 6; ++s)
        for (int q = 0; q < 6; ++q) {
            int p = (q+s)%6; double t = timeFn(run[p], iters[p]);
            if (t < best[p]) best[p] = t; }
    CHECK(cudaGetLastError());

    printf("\n-- measurement and reconciliation ------------------------------------\n");
    printf("  %-9s %10s %10s %9s %7s %7s %-8s\n", "kernel","GFLOP/s","predGF/s",
           "meas/pred","%FP32","bucket","verdict");
    double gf[6], pct[6];
    for (int i = 0; i < 6; ++i) {
        gf[i]  = L[i].flops/(best[i]*1e-3)/1e9;
        pct[i] = 100.0*gf[i]/CEIL_FP32;
        double r = gf[i]/pred[i];
        printf("  %-9s %10.3f %10.1f %9.4f %6.3f%% %7d %-8s\n", nm[i], gf[i], pred[i],
               r, pct[i], bucketOf(pct[i]),
               (r>0.80&&r<1.25)?"at roof":(r>=1.25?"ABOVE":(r>0.25?"below":"FAR below")));
    }

    // ---- Little's Law for the chase ---------------------------------------
    const double need = bytesInFlightNeeded(CEIL_DRAM, DRAM_LATENCY_CYCLES, CLOCK_GHZ);
    const double got  = bytesInFlightSupplied((double)CHASE_THR, SECTOR_BYTES);
    const double predFrac = got/need;
    const double measFrac = (L[5].dramBytes/(best[5]*1e-3)/1e9)/CEIL_DRAM;
    printf("\n-- TODO 5: why the chase is %4.0fx below its own roofline -------------\n",
           pred[5]/gf[5]);
    printf("  bytes the memory system needs in flight : %10.0f  (%.0f GB/s x %.0f cy / %.3f GHz)\n",
           need, CEIL_DRAM, DRAM_LATENCY_CYCLES, CLOCK_GHZ);
    printf("  bytes this kernel has in flight         : %10.0f  (%d threads x %.0f B sector)\n",
           got, CHASE_THR, SECTOR_BYTES);
    printf("  predicted fraction of the DRAM ceiling  : %10.5f\n", predFrac);
    printf("  measured  fraction of the DRAM ceiling  : %10.5f\n", measFrac);
    printf("  ratio                                   : %10.3f\n", measFrac/predFrac);
    printf("\n  The ratio is not 1.00 and it should not be. Working backwards, the\n"
           "  round trip this kernel actually sees is %.0f cycles, not the %.0f that\n",
           (best[5]*1e-3/CHASE_STEPS)*CLOCK_GHZ*1e9, DRAM_LATENCY_CYCLES);
    printf("  Module 4 measured for a SINGLE-thread chase: a warp issues one LDG\n"
           "  covering 32 independent sectors, and those 32 round trips overlap.\n"
           "  Little's Law gets the order of magnitude and the mechanism right and\n"
           "  the constant wrong, which is all anyone should ask of it.\n");

    // ---- validation --------------------------------------------------------
    printf("\n-- validation (second, untimed pass) ----------------------------------\n");
    int ok = 1;
    CHECK(cudaMemcpy(dY,hY,sizeof(float)*NELEM,cudaMemcpyHostToDevice));
    kTriad(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    { float *t=(float*)malloc(sizeof(float)*NELEM);
      CHECK(cudaMemcpy(t,dY,sizeof(float)*NELEM,cudaMemcpyDeviceToHost));
      double w=0; for (int i=0;i<NELEM;i+=997){ double ref=2.0*hX[i]+hY[i];
          double e=fabs(t[i]-ref)/fmax(1.0,fabs(ref)); if (e>w) w=e; }
      printf("  triad  %.3e  %s\n", w, w<1e-6?"ok":"FAIL"); if (!(w<1e-6)) ok=0; free(t); }
    kReduce(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    { float *p=(float*)malloc(sizeof(float)*240);
      CHECK(cudaMemcpy(p,dPart,sizeof(float)*240,cudaMemcpyDeviceToHost));
      double got2=0; for (int i=0;i<240;++i) got2+=p[i];
      double ref=0; for (int i=0;i<NELEM;++i) ref+=(double)hX[i];
      double e=fabs(got2-ref)/ref;
      printf("  reduce %.3e  %s\n", e, e<1e-5?"ok":"FAIL"); if (!(e<1e-5)) ok=0; free(p); }
    { const char *gn[3]={"gemmN","gemmT","gemmR"};
      void (*gf2[3])(void)={kGemmN,kGemmT,kGemmR};
      for (int j=0;j<3;++j){
        float *po=(float*)malloc(sizeof(float)*sC);
        for (size_t i=0;i<sC;++i) po[i]=std::numeric_limits<float>::infinity();
        CHECK(cudaMemcpy(dC,po,sizeof(float)*sC,cudaMemcpyHostToDevice)); free(po);
        gf2[j](); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hC,dC,sizeof(float)*sC,cudaMemcpyDeviceToHost));
        double w; int good=gemmQuickValidate(M_DIM,N_DIM,K_DIM,hA,hB,hC,&w);
        printf("  %-6s Freivalds headroom %.4f  %s\n", gn[j], w, good?"ok":"FAIL");
        if (!good) ok=0; } }
    kChase(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hCh,dChaseOut,sizeof(float)*CHASE_THR,cudaMemcpyDeviceToHost));
    { int bad=0;
      for (int t=0;t<CHASE_THR;++t){ int p=t; double acc=0;
          for (int s=0;s<CHASE_STEPS;++s){ p=hN[p]; acc+=(double)(p&1023); }
          if (fabs(acc-(double)hCh[t])>1.0) ++bad; }
      printf("  chase  %d mismatched threads  %s\n", bad, bad==0?"ok":"FAIL");
      if (bad) ok=0; }

    // ---- scoring -----------------------------------------------------------
    int score = 0;
    printf("\n-- scoring -----------------------------------------------------------\n");
    { double v[18];
      for (int i=0;i<6;++i){ v[3*i]=L[i].flops; v[3*i+1]=L[i].dramBytes; v[3*i+2]=L[i].reqBytes; }
      unsigned h = fnvNums(v,18,1.0);
      int good = (h == 2878286269u);
      printf("  [%s] 1. traffic ledger matches the reference (2 pts)\n", good?"x":" ");
      score += good ? 2 : 0; }
    { double p; int r[6];
      r[0]=classify(100,100,100,64, 400,5000,300,18000,&p);
      r[1]=classify(1e6,1e3,1e3,64, 400,5000,300,18000,&p);
      r[2]=classify(1e6,1e3,1e7,64, 400,5000,300,18000,&p);
      r[3]=classify(1e6,1e3,1e3,1.0,400,5000,300,18000,&p);
      r[4]=classify(1e9,1e3,1e3,64, 400,5000,300,18000,&p);
      r[5]=classify(2.0,8.0,8.0,64, 400,5000,300,18000,&p);
      unsigned h=fnvInts(r,6); int good=(h==948881140u);
      printf("  [%s] 2. classify() matches the reference on 6 probes\n", good?"x":" ");
      score += good; }
    { unsigned h=fnvInts(PRED_LEVEL,6); int good=(h==3205188794u);
      printf("  [%s] 3. predicted limiting level for all six kernels\n", good?"x":" ");
      score += good; }
    { int nb=0; for (int i=0;i<6;++i) if (PRED_BUCKET[i]==bucketOf(pct[i])) ++nb;
      printf("  [%s] 4. %d/6 performance buckets correct (need 6) (2 pts)\n",
             nb==6?"x":" ", nb);
      score += (nb==6) ? 2 : 0; }
    { double lv[4] = { bytesInFlightNeeded(400.0,575.0,1.8),
                       bytesInFlightNeeded(410.5,575.0,1.831),
                       bytesInFlightSupplied(32.0,32.0),
                       bytesInFlightSupplied(1280.0,32.0) };
      unsigned h = fnvNums(lv,4,100.0); int good = (h==3968990285u);
      printf("  [%s] 5a. bytesInFlight*() match the reference\n", good?"x":" ");
      score += good; }
    { double r = measFrac/predFrac; int good = (r>0.25 && r<4.0);
      printf("  [%s] 5b. Little's Law predicts the chase within 4x (ratio %.3f)\n",
             good?"x":" ", r);
      score += good; }
    printf("  [%s] 6. all six kernels validated\n", ok?"x":" ");
    score += ok;

    CHECK(cudaFree(dX)); CHECK(cudaFree(dY)); CHECK(cudaFree(dPart));
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    CHECK(cudaFree(dNxt)); CHECK(cudaFree(dChaseOut)); CHECK(cudaFree(dSink));
    CHECK(cudaFree(dWarm));
    free(hX); free(hY); free(hA); free(hB); free(hC); free(hN); free(hCh);

    printf("\nSCORE: %d/9\n", score);
    printf("OVERALL: %s\n", score==9 ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return score==9 ? 0 : 1;
}
