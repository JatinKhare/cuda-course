// =============================================================================
// Module 18 / Exercise 3 - design the whole kernel.
//
// This is the point in the course where the scaffolding goes away. You get a
// problem statement, a validator, a timing harness, a baseline to beat and a
// performance gate. Everything else - the decomposition, the tile hierarchy,
// the shared-memory layout, the thread-to-data mapping, the synchronization,
// the vectorization, the boundary handling, the epilogue - is yours.
//
// PROBLEM
//   C = A * B, row-major fp32, alpha = 1, beta = 0 semantics honoured.
//   A is M x K with lda = K, B is K x N with ldb = N, C is M x N with ldc = N.
//   The harness runs you at FOUR shapes, including two small awkward ones in
//   which every block is a partial tile:
//       1027 x 2053 x 769  (positive operands)
//       1027 x 2053 x 769  (zero-mean operands, so |C| << S)
//         37 x   53 x  11
//        129 x   65 x   9
//   C is prefilled with +infinity before every launch. An element you do not
//   write, or a kernel that reads C when beta == 0, is caught.
//
// THE GATE
//   Your kernel must reach at least 4.5x the throughput of the Module 17
//   block-tiled baseline that ships in this file. On this GPU that baseline
//   runs at roughly 1400-1550 GFLOP/s, so the gate is roughly 6500 GFLOP/s,
//   which is in the neighbourhood of cuBLAS on this shape.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN  : exercise03.exe
// Strongly recommended while you work:
//   nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise03.exe exercise03.cu
//   nvcc -arch=sm_89 -O3 -cubin -o exercise03.cubin exercise03.cu
//   cuobjdump -sass exercise03.cubin
// The SASS is the evidence. Count FFMA against everything else in the body
// between the two BAR.SYNCs; if that ratio is not most of the instruction
// stream, the design is wrong and no amount of tuning will save it.
//
// WHAT IS SCORED (10 points; OVERALL: PASS needs all ten)
//   4  correct at all four shapes, with nothing unwritten
//   2  the resource ledger you commit to in TODO 2 matches what the compiler
//      and the occupancy API actually report
//   2  the performance gate
//   2  your committed prediction of where you will land on the FP32 ceiling
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

// =============================================================================
// TODO 1 - the tile hierarchy.
//
// Choose every number. The harness checks them for internal consistency and
// prints the consequences before it compiles anything of yours into a launch.
//
//   BMX, BNX : rows and columns of C owned by one block
//   BKX      : depth of one shared-memory tile
//   TMX, TNX : rows and columns of C owned by one thread
//   THREADSX : threads per block
//
// Constraints the harness enforces:
//   THREADSX == (BMX/TMX) * (BNX/TNX)
//   THREADSX is a multiple of 32 and at most 1024
//   BMX % TMX == 0, BNX % TNX == 0
//   (BMX*BKX) % THREADSX == 0 and (BKX*BNX) % THREADSX == 0
//   the shared memory you declare fits in 48 KB (no opt-in carveout here)
//
// Constraints it does not enforce, and you should think about anyway:
//   the register file is 65536 registers per SM and TMX*TNX of them per thread
//   are accumulators before anything else is allocated; a thread tile that
//   does not fit spills, and a spill is a DRAM access in the innermost loop.
// =============================================================================
#define BMX      0      // YOUR CODE HERE
#define BNX      0      // YOUR CODE HERE
#define BKX      0      // YOUR CODE HERE
#define TMX      0      // YOUR CODE HERE
#define TNX      0      // YOUR CODE HERE
#define THREADSX 0      // YOUR CODE HERE

// =============================================================================
// TODO 2 - the ledger. Compute these from TODO 1's constants, by hand, before
// you write the kernel. The harness checks each one against the compiler and
// the occupancy API.
//
//   LEDGER_ACC    : accumulators held in registers per thread
//   LEDGER_SMEM   : bytes of static shared memory your kernel will declare per
//                   block (the harness compares against cudaFuncGetAttributes,
//                   so this must match your declaration exactly, padding
//                   included)
//   LEDGER_FPGL   : fused multiply-adds your block performs per element it
//                   loads from global memory, rounded to two decimals
//                   (the harness compares to +/- 0.01)
// =============================================================================
#define LEDGER_ACC   0          // YOUR CODE HERE
#define LEDGER_SMEM  0          // YOUR CODE HERE (bytes)
#define LEDGER_FPGL  0.0        // YOUR CODE HERE

// =============================================================================
// TODO 3 - the kernel. All of it.
//
// Signature is fixed so the harness can call it. Everything inside is yours:
// shared memory declaration and layout, the thread-to-element mapping for the
// cooperative loads, the k-tile loop, the barriers, the register tile, the
// inner product, and the epilogue.
//
// Requirements:
//   - correct for arbitrary M, N, K > 0, including K not a multiple of BKX
//   - honours the beta == 0 contract: C is not read
//   - no out-of-bounds access, ever (compute-sanitizer will be run on it)
// =============================================================================
__global__ void gemmYours(int M, int N, int K, float alpha,
                          const float * __restrict__ A,
                          const float * __restrict__ B,
                          float beta, float *C)
{
    // YOUR CODE HERE
    (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)B; (void)beta; (void)C;
}

// =============================================================================
// TODO 4 - the launch configuration.
//
// Fill in the grid and block dimensions your kernel expects. `gemmYours` is
// launched exactly as written below for every shape the harness tests,
// including 37 x 53 x 11.
// =============================================================================
static void launchYours(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C)
{
    dim3 grid(1, 1, 1);      // YOUR CODE HERE
    dim3 block(1, 1, 1);     // YOUR CODE HERE
    gemmYours<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

// =============================================================================
// TODO 5 - PREDICTION. Commit before building.
//
// P1: what fraction of this GPU's measured FP32 ceiling (18 000 GFLOP/s) will
//     your kernel reach on the 1027 x 2053 x 769 shape?
//       1 : under 15%       2 : 15% to 25%      3 : 25% to 35%
//       4 : 35% to 55%      5 : over 55%
// =============================================================================
#define PRED_CEIL 0   // YOUR CODE HERE (1..5)

// =============================================================================
// The Module 17 baseline. Complete; this is what you have to beat by 4.5x.
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
// Module 16's validator, unchanged.
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
// harness
// =============================================================================
#define M_DIM 1027
#define N_DIM 2053
#define K_DIM  769

static int Mg = M_DIM, Ng = N_DIM, Kg = K_DIM;
static const float *dAg, *dBg;
static float *dCg;

static void runYours(void) { launchYours(Mg, Ng, Kg, 1.0f, dAg, dBg, 0.0f, dCg); }
static void runBase(void)  { dim3 bl(32,32), gr((Ng+31)/32,(Mg+31)/32);
    gemmTiled<32><<<gr,bl>>>(Mg,Ng,Kg,1.0f,dAg,dBg,0.0f,dCg); }

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

    launchYours(M, N, K, 1.0f, dA, dB, 0.0f, dC);
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

    if (BMX == 0 || BNX == 0 || BKX == 0 || TMX == 0 || TNX == 0 || THREADSX == 0) {
        printf("Set TODO 1 first.\n"); return 0; }
    if (LEDGER_ACC == 0 || LEDGER_SMEM == 0 || LEDGER_FPGL == 0.0) {
        printf("Set TODO 2 first.\n"); return 0; }
    if (PRED_CEIL == 0) { printf("Set TODO 5 (PREDICTION) first.\n"); return 0; }

    printf("=== Module 18 / Exercise 3 - design the whole kernel ===\n");
    printf("block tile %d x %d x %d, thread tile %d x %d, %d threads/block\n\n",
           BMX, BNX, BKX, TMX, TNX, THREADSX);

    // ---- TODO 1 consistency. The tile constants are copied into runtime
    //      variables (and clamped) so that a zero never reaches a division
    //      while the TODOs are still unfilled.
    int bm = BMX, bn = BNX, bk = BKX, tm = TMX, tn = TNX, th = THREADSX;
    if (tm < 1) tm = 1;  if (tn < 1) tn = 1;  if (th < 1) th = 1;
    int bad = 0;
    if (th != (bm/tm)*(bn/tn))
        { printf("  THREADSX != (BMX/TMX)*(BNX/TNX)\n"); bad = 1; }
    if (th % 32 || th > 1024)
        { printf("  THREADSX must be a multiple of 32 and at most 1024\n"); bad = 1; }
    if (bm % tm || bn % tn)
        { printf("  BMX must be a multiple of TMX and BNX of TNX\n"); bad = 1; }
    if ((bm*bk) % th || (bk*bn) % th)
        { printf("  the tile loads must divide evenly among the threads\n"); bad = 1; }
    if (bad) { printf("\nOVERALL: FAIL\n"); return 1; }

    // ---- correctness, four shapes
    printf("-- correctness ----------------------------------------------------\n");
    int ok = 0;
    ok += checkShape(M_DIM, N_DIM, K_DIM, 0, "1027 x 2053 x 769, positive");
    ok += checkShape(M_DIM, N_DIM, K_DIM, 1, "1027 x 2053 x 769, zero-mean");
    ok += checkShape(37, 53, 11, 0,          "37 x 53 x 11");
    ok += checkShape(129, 65, 9, 1,          "129 x 65 x 9");
    printf("  %d/4 shapes correct\n", ok);

    // ---- TODO 2 ledger
    cudaFuncAttributes at; CHECK(cudaFuncGetAttributes(&at, (const void*)gemmYours));
    int blk = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blk, (const void*)gemmYours, th, 0));
    const double fpglTrue = (double)bm*bn/((double)bm + bn);
    const int    accTrue  = tm*tn;
    printf("\n-- the ledger -----------------------------------------------------\n");
    printf("  accumulators/thread   : you said %6d, true %6d   %s\n",
           LEDGER_ACC, accTrue, LEDGER_ACC == accTrue ? "ok" : "WRONG");
    printf("  shared bytes/block    : you said %6d, true %6d   %s\n",
           LEDGER_SMEM, (int)at.sharedSizeBytes,
           LEDGER_SMEM == (int)at.sharedSizeBytes ? "ok" : "WRONG");
    printf("  FMAs per global load  : you said %6.2f, true %6.2f   %s\n",
           (double)LEDGER_FPGL, fpglTrue,
           fabs((double)LEDGER_FPGL - fpglTrue) <= 0.01 ? "ok" : "WRONG");
    printf("  compiler reports      : %d registers, %d B spilled, %d B shared\n",
           at.numRegs, (int)at.localSizeBytes, (int)at.sharedSizeBytes);
    printf("  occupancy API reports : %d blocks/SM, %.1f%% of 1536 threads\n",
           blk, 100.0*blk*th/1536.0);
    if (at.localSizeBytes > 0)
        printf("  NOTE: your kernel spills %d bytes per thread to local memory,\n"
               "  which Module 4 established is DRAM. Look at the register count.\n",
               (int)at.localSizeBytes);
    int ledgerOK = (LEDGER_ACC == accTrue) && (LEDGER_SMEM == (int)at.sharedSizeBytes)
                && (fabs((double)LEDGER_FPGL - fpglTrue) <= 0.01);

    if (ok == 0) { printf("\nNothing correct yet; skipping the timed section.\n");
                   printf("\nOVERALL: FAIL\n"); CHECK(cudaDeviceReset()); return 1; }

    // ---- timing
    const int M = M_DIM, N = N_DIM, K = K_DIM;
    const size_t sA = (size_t)M*K, sB = (size_t)K*N, sC = (size_t)M*N;
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

    void (*runs[2])(void) = { runBase, runYours };
    int iters[2]; double best[2] = {1e30, 1e30};
    for (int i = 0; i < 2; ++i) {
        double t = timeOne(runs[i], 1);
        int n = (int)(10.0/(t > 0 ? t : 0.01));
        if (n < 3) n = 3; if (n > 64) n = 64; iters[i] = n;
    }
    for (int s = 0; s < 4; ++s)
        for (int q = 0; q < 2; ++q) {
            int p = (q + s) % 2;
            double t = timeOne(runs[p], iters[p]);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    const double flops = 2.0*M*N*K;
    const double gb = flops/(best[0]*1e-3)/1e9, gm = flops/(best[1]*1e-3)/1e9;
    const double sp = best[0]/best[1], frac = gm/18000.0;
    printf("\n-- measurement ----------------------------------------------------\n");
    printf("  %-34s %9.4f ms %10.1f GFLOP/s\n", "Module 17 baseline", best[0], gb);
    printf("  %-34s %9.4f ms %10.1f GFLOP/s\n", "your kernel", best[1], gm);
    printf("  speedup over the baseline            %6.2f x   (gate: 4.50 x)\n", sp);
    printf("  fraction of the FP32 ceiling         %6.2f %%\n", 100.0*frac);

    int truth = (frac < 0.15) ? 1 : (frac < 0.25) ? 2 : (frac < 0.35) ? 3
              : (frac < 0.55) ? 4 : 5;
    printf("\n  P1 fraction of the ceiling : you said %d, measured %d   %s\n",
           PRED_CEIL, truth, PRED_CEIL == truth ? "correct" : "WRONG");

    int score = 0;
    if (ok == 4)       score += 4;
    if (ledgerOK)      score += 2;
    if (sp >= 4.5)     score += 2;
    if (PRED_CEIL == truth) score += 2;
    printf("\n  score %d/10\n", score);

    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    free(hA); free(hB);
    printf("\nOVERALL: %s\n", score == 10 ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return score == 10 ? 0 : 1;
}
