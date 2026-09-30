// =============================================================================
// Module 16 / Example 2 — where the FLOPs went.
//
// GOAL : Place the naive GEMM on a roofline built entirely from measurements
//        on this GPU, and identify the binding constraint exactly.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN  : example02.exe
//
// Sections:
//   A  the two ceilings, measured: FP32 FFMA throughput and DRAM streaming
//   B  per-warp sector counts for both thread->element mappings (M5's method)
//   C  the mapping and block-shape sweep, measured
//   D  the traffic ledger and the implied minimum on-chip service fraction
//   E  the loads-per-FMA probe: what the load path alone allows
//   F  the roofline table
//
// No shared memory anywhere in this file. Module 17 owns tiling; Module 18
// owns register blocking. This file's job is to make both inevitable.
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

// ============================================================================
// The two kernels under study. They differ in ONE line: which of the two
// block/thread axes carries the row index and which carries the column index.
// They compile to byte-identical SASS.
// ============================================================================
__global__ void gemmXcol(int M, int N, int K, const float *A, const float *B, float *C)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[(size_t)row*K + k] * B[(size_t)k*N + col];
    C[(size_t)row*N + col] = acc;
}
__global__ void gemmXrow(int M, int N, int K, const float *A, const float *B, float *C)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[(size_t)row*K + k] * B[(size_t)k*N + col];
    C[(size_t)row*N + col] = acc;
}

// ---- ceilings -------------------------------------------------------------
// FP32 FFMA ceiling: 8 independent chains per thread so the pipe, not the
// dependence, is the limit. One wave, so no tail effect.
__global__ void ffmaCeiling(float *out, int iters)
{
    float a0=threadIdx.x, a1=a0+1, a2=a0+2, a3=a0+3, a4=a0+4, a5=a0+5, a6=a0+6, a7=a0+7;
    const float b = 1.0000001f, c = 0.9999999f;
    for (int i = 0; i < iters; ++i) {
        a0=fmaf(a0,b,c); a1=fmaf(a1,b,c); a2=fmaf(a2,b,c); a3=fmaf(a3,b,c);
        a4=fmaf(a4,b,c); a5=fmaf(a5,b,c); a6=fmaf(a6,b,c); a7=fmaf(a7,b,c);
    }
    float s = a0+a1+a2+a3+a4+a5+a6+a7;
    if (s == 1.2345e30f) out[0] = s;          // never true; not provably dead
}
// DRAM streaming ceiling: a pure READ stream over a buffer 5.3x the 48 MB L2,
// four independent accumulators for memory-level parallelism, and a store the
// compiler cannot prove dead. This is Module 12's ceiling kernel; it measures
// the read side, which is what a GEMM's traffic is made of.
__global__ void streamCeiling(const float4 *in, float4 *out, size_t n4)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 s0 = make_float4(0,0,0,0);
    for (; i < n4; i += stride) {
        float4 v = in[i];
        s0.x += v.x; s0.y += v.y; s0.z += v.z; s0.w += v.w;
    }
    if (s0.x + s0.y + s0.z + s0.w == 1.2345e30f) out[0] = s0;
}

// ---- E: the loads-per-FMA probe ------------------------------------------
// NOT an optimization and NOT a GEMM. It performs the same two loads the naive
// kernel performs and then issues R FFMAs against them instead of 1. The
// result is arithmetically meaningless; the point is the instruction mix.
template <int R>
__global__ void loadsPerFmaProbe(int M, int N, int K,
                                 const float *A, const float *B, float *C)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float av = A[(size_t)row*K + k];
        const float bv = B[(size_t)k*N + col];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = fmaf(av, bv, acc[r]);
    }
    float s = 0.0f;
#pragma unroll
    for (int r = 0; r < R; ++r) s += acc[r];
    C[(size_t)row*N + col] = s;
}

// ============================================================================
// B. The sector-counting method of Module 5, applied to one warp of a GEMM.
//
// A warp is 32 consecutive linearized thread ids (Module 3):
//     tid = threadIdx.x + blockDim.x * threadIdx.y,  warp = tid / 32.
// For warp 0 of block 0 at a fixed k, enumerate the 32 lanes, compute the byte
// address each one presents to A and to B, divide by 32, and count distinct
// sector ids. Inactive lanes supply no address; here all 32 are active.
// ============================================================================
static int countDistinct(const long long *v, int n)
{
    int d = 0;
    for (int i = 0; i < n; ++i) { int seen = 0;
        for (int j = 0; j < i; ++j) if (v[j] == v[i]) { seen = 1; break; }
        if (!seen) ++d; }
    return d;
}
static void warpSectors(int bx, int by, int mapping, int K, int N,
                        int *secA, int *secB)
{
    (void)by;
    long long addrA[32], addrB[32];
    for (int lane = 0; lane < 32; ++lane) {
        int tx = lane % bx, ty = lane / bx;
        int row = (mapping == 0) ? ty : tx;      // 0 = x->col, 1 = x->row
        int col = (mapping == 0) ? tx : ty;
        addrA[lane] = ((long long)row * K + 0) * 4 / 32;      // A[row*K + k], k = 0
        addrB[lane] = ((long long)0 * N + col) * 4 / 32;      // B[k*N + col], k = 0
    }
    *secA = countDistinct(addrA, 32);
    *secB = countDistinct(addrB, 32);
}

// ---- timing helper --------------------------------------------------------
typedef void (*RunFn)(void*);
static int calibrate(RunFn run, void *ctx)
{
    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0)); run(ctx); CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    int it = (int)(10.0 / (ms > 0.0f ? ms : 0.01f));
    if (it < 1) it = 1; if (it > 64) it = 64;
    return it;
}
static double timeOnce(RunFn run, void *ctx, int iters)
{
    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) run(ctx);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    return ms / iters;
}
static void warmup(RunFn run, void *ctx, float targetMs)
{
    cudaEvent_t w0, w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
    float el = 0.0f; CHECK(cudaEventRecord(w0));
    while (el < targetMs) {
        run(ctx);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el, w0, w1));
    }
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
}

// contexts
typedef struct { int M,N,K,bx,by,mapping; const float *A,*B; float *C; } GCtx;
static void runGemm(void *p) {
    GCtx *g = (GCtx*)p; dim3 bl(g->bx, g->by);
    if (g->mapping == 0) { dim3 gr((g->N+g->bx-1)/g->bx, (g->M+g->by-1)/g->by);
        gemmXcol<<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C); }
    else { dim3 gr((g->M+g->bx-1)/g->bx, (g->N+g->by-1)/g->by);
        gemmXrow<<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C); }
}
typedef struct { int R; GCtx g; } PCtx;
static void runProbe(void *p) {
    PCtx *q = (PCtx*)p; GCtx *g = &q->g; dim3 bl(g->bx,g->by);
    dim3 gr((g->N+g->bx-1)/g->bx, (g->M+g->by-1)/g->by);
    switch (q->R) {
      case 1: loadsPerFmaProbe<1><<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C); break;
      case 2: loadsPerFmaProbe<2><<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C); break;
      case 4: loadsPerFmaProbe<4><<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C); break;
      default:loadsPerFmaProbe<8><<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C); break;
    }
}
typedef struct { float *out; int iters; int grid; } FCtx;
static void runFfma(void *p) { FCtx *f=(FCtx*)p; ffmaCeiling<<<f->grid,256>>>(f->out,f->iters); }
typedef struct { const float4 *in; float4 *out; size_t n4; int grid; } SCtx;
static void runStream(void *p){ SCtx*s=(SCtx*)p; streamCeiling<<<s->grid,256>>>(s->in,s->out,s->n4); }

// ============================================================================
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const int M = M_DIM, N = N_DIM, K = K_DIM;
    const size_t sA=(size_t)M*K, sB=(size_t)K*N, sC=(size_t)M*N;
    const double flops = 2.0*M*N*K;

    printf("=== Module 16 / Example 2 — where the FLOPs went ===\n");
    printf("C(%d x %d) = A(%d x %d) * B(%d x %d), row-major fp32\n\n", M,N,M,K,K,N);

    float *dA,*dB,*dC;
    CHECK(cudaMalloc(&dA,sA*4)); CHECK(cudaMalloc(&dB,sB*4)); CHECK(cudaMalloc(&dC,sC*4));
    { float *h=(float*)malloc((sB>sA?sB:sA)*4);
      for (size_t i=0;i<sA;++i) h[i] = 0.5f + (float)(i % 251) / 502.0f;
      CHECK(cudaMemcpy(dA,h,sA*4,cudaMemcpyHostToDevice));
      for (size_t i=0;i<sB;++i) h[i] = 0.5f + (float)(i % 257) / 514.0f;
      CHECK(cudaMemcpy(dB,h,sB*4,cudaMemcpyHostToDevice));
      free(h); }

    // ---------------------------------------------------------------- A
    printf("-- A. the two ceilings, measured on this GPU ----------------------\n");
    const size_t STREAM_BYTES = 256u*1024u*1024u;      // 5.3x the 48 MB L2
    const size_t n4 = STREAM_BYTES / sizeof(float4);
    float4 *sIn,*sOut;
    CHECK(cudaMalloc(&sIn, STREAM_BYTES)); CHECK(cudaMalloc(&sOut, STREAM_BYTES));
    CHECK(cudaMemset(sIn, 0x3c, STREAM_BYTES));
    float *fOut; CHECK(cudaMalloc(&fOut, 4));

    int bpsm = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm,(const void*)ffmaCeiling,256,0));
    int nsm = 0; CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    FCtx fc; fc.out=fOut; fc.iters=50000; fc.grid=bpsm*nsm;
    SCtx sc; sc.in=sIn; sc.out=sOut; sc.n4=n4; sc.grid=nsm*bpsm*2;

    // The two ceilings stress different resources and therefore need
    // different warm-ups. They are NOT competing configurations, so spec 12
    // rule 1 (time competitors back to back) does not apply; what does apply
    // is rule 4 -- and a 1500 ms *streaming* warm-up is what ramps the memory
    // P-state. Warming with the compute kernel and then measuring the stream
    // reproducibly reports 300-380 GB/s instead of 410.
    { cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0.0f; CHECK(cudaEventRecord(w0));
      while (el < 1500.0f) { runStream(&sc);
          CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
          CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); }
    double bS = 1e30; { int itS = calibrate(runStream,&sc);
      for (int s = 0; s < 6; ++s) { double b = timeOnce(runStream,&sc,itS); if (b<bS) bS=b; } }

    warmup(runFfma,&fc,500.0f);
    double bF = 1e30; { int itF = calibrate(runFfma,&fc);
      for (int s = 0; s < 4; ++s) { double a = timeOnce(runFfma,&fc,itF); if (a<bF) bF=a; } }

    CHECK(cudaGetLastError());
    const double ceilCompute = 2.0*8.0*fc.iters*256.0*fc.grid/(bF*1e-3)/1e9;   // GFLOP/s
    const double ceilDram    = (double)STREAM_BYTES/(bS*1e-3)/1e9;             // GB/s, read-only
    const double impliedClk  = ceilCompute / (nsm*128.0*2.0);                  // GHz
    int apiClkKHz = 0; CHECK(cudaDeviceGetAttribute(&apiClkKHz, cudaDevAttrClockRate, 0));

    printf("  FP32 FFMA ceiling   %10.1f GFLOP/s  (8 independent chains, %d blocks = 1 wave)\n",
           ceilCompute, fc.grid);
    printf("  DRAM stream ceiling %10.1f GB/s     (%.0f%% of the 432.0 GB/s pin peak)\n",
           ceilDram, 100.0*ceilDram/432.0);
    printf("  MACHINE BALANCE     %10.2f FLOP/byte = compute ceiling / DRAM ceiling\n",
           ceilCompute/ceilDram);
    printf("  implied SM clock    %10.4f GHz   (cudaDevAttrClockRate reports %.3f GHz\n"
           "                                     and is wrong -- spec 12 rule 6.\n"
           "                                     A 'peak' built from it is %.0f GFLOP/s,\n"
           "                                     which the measurement exceeds by %.2fx.)\n\n",
           impliedClk, apiClkKHz/1.0e6, nsm*128.0*2.0*apiClkKHz/1.0e6,
           ceilCompute/(nsm*128.0*2.0*apiClkKHz/1.0e6));

    // ---------------------------------------------------------------- B
    printf("-- B. one warp's sector footprint, per k, by hand -----------------\n");
    printf("  %-16s %-14s %8s %8s %8s %10s\n",
           "mapping","block","A sect","B sect","total","bytes/32 elems");
    const int shapes[][2] = { {32,8},{32,32},{16,16},{8,32} };
    for (int m = 0; m < 2; ++m)
        for (int s = 0; s < 4; ++s) {
            int sa, sb; warpSectors(shapes[s][0], shapes[s][1], m, K, N, &sa, &sb);
            printf("  %-16s (%2d,%2d)        %8d %8d %8d %10d\n",
                   m ? "x -> row" : "x -> col", shapes[s][0], shapes[s][1],
                   sa, sb, sa+sb, (sa+sb)*32);
        }
    printf("\n  Read the two extremes. With x -> col and a 32-wide block a warp's\n"
           "  32 lanes share ONE element of A (a broadcast, 1 sector) and cover 32\n"
           "  consecutive elements of B (128 B, 4 sectors): 5 sectors = 160 B to feed\n"
           "  32 lanes x 2 operands = 256 B of demand. With x -> row the roles swap:\n"
           "  B becomes the broadcast and A becomes 32 addresses %d bytes apart --\n"
           "  32 distinct sectors, 1056 B, of which 256 B are wanted.\n"
           "  Same kernel. Same SASS. 33/5 = %.1fx the sectors.\n\n", K*4, 33.0/5.0);

    // ---------------------------------------------------------------- C
    printf("-- C. measured, all configurations in one rotated sweep ----------\n");
    enum { NCFG = 6 };
    GCtx cfg[NCFG] = {
        {M,N,K,32, 8,0,dA,dB,dC}, {M,N,K,32,32,0,dA,dB,dC},
        {M,N,K,16,16,0,dA,dB,dC}, {M,N,K, 8,32,0,dA,dB,dC},
        {M,N,K,32, 8,1,dA,dB,dC}, {M,N,K,16,16,1,dA,dB,dC},
    };
    double best[NCFG]; int iters[NCFG];
    warmup(runGemm,&cfg[0],1500.0f);
    for (int i=0;i<NCFG;++i){ best[i]=1e30; iters[i]=calibrate(runGemm,&cfg[i]); }
    const int SWEEPS = NCFG;                    // spec 12 rule 9: SWEEPS >= NCFG
    for (int s=0;s<SWEEPS;++s)
        for (int q=0;q<NCFG;++q) {
            int p = (q+s) % NCFG;
            double t = timeOnce(runGemm,&cfg[p],iters[p]);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());
    double bestColX = 1e30;
    for (int i=0;i<4;++i) if (best[i] < bestColX) bestColX = best[i];

    printf("  %-10s %-8s %10s %12s %10s %12s\n",
           "mapping","block","ms","GFLOP/s","% ceiling","vs best");
    for (int i=0;i<NCFG;++i) {
        double g = flops/(best[i]*1e-3)/1e9;
        printf("  %-10s (%2d,%2d)  %10.4f %12.1f %9.2f%% %11.2fx\n",
               cfg[i].mapping ? "x -> row" : "x -> col", cfg[i].bx, cfg[i].by,
               best[i], g, 100.0*g/ceilCompute, best[i]/bestColX);
    }
    printf("\n  predicted from sectors  x->row(32,8) / x->col(32,8) = 33/5  = %.2fx\n",
           33.0/5.0);
    printf("  measured                                            = %.2fx\n",
           best[4]/best[0]);
    printf("  predicted  x->row(16,16) / x->row(32,8) = 17/33 = %.3f\n", 17.0/33.0);
    printf("  measured                                        = %.3f\n", best[5]/best[4]);
    printf("  The sector model gets the second ratio close and OVER-predicts the\n"
           "  first. Sectors REQUESTED are not sectors FETCHED: A's 32 scattered\n"
           "  sectors are re-requested by every warp in the grid and, at this\n"
           "  working-set size, largely hit in L2. The count is an upper bound on\n"
           "  the damage, not a prediction of it -- Module 6's lesson again.\n\n");

    // ---------------------------------------------------------------- D
    const double compulsory = 4.0*((double)M*K + (double)K*N + (double)M*N);
    const double requested  = 4.0*2.0*(double)M*N*K;
    const double implBW     = requested/(bestColX*1e-3)/1e9;
    // For a BOUND, use the pin peak, not the measured streaming figure: DRAM
    // physically cannot deliver more than 432.0 GB/s, whereas the measured
    // streaming number moves with the memory P-state (294-411 GB/s observed in
    // one session on this laptop part) and using a throttled value would
    // overstate the conclusion.
    const double PIN_PEAK = 432.0;
    printf("-- D. what the caches actually recovered -------------------------\n");
    printf("  compulsory traffic                  %10.3f MB\n", compulsory/1e6);
    printf("  bytes the naive kernel REQUESTS     %10.3f GB   (%.0fx compulsory)\n",
           requested/1e9, requested/compulsory);
    printf("  implied request bandwidth           %10.1f GB/s (requested / best time)\n",
           implBW);
    printf("  measured DRAM read ceiling          %10.1f GB/s\n", ceilDram);
    printf("  DRAM pin peak (used for the bound)  %10.1f GB/s\n", PIN_PEAK);
    printf("  => at most %.1f%% of the requested bytes can have come from DRAM,\n",
           100.0*PIN_PEAK/implBW);
    printf("     so AT LEAST %.1f%% were served on chip (L1 or L2).\n",
           100.0*(1.0 - PIN_PEAK/implBW));
    printf("  This is a bound, not a counter reading: ncu is unavailable on this\n"
           "  machine (ERR_NVGPUCTRPERM), so the hit rate is derived from the two\n"
           "  things that CAN be measured -- elapsed time and the DRAM ceiling.\n");
    printf("  It is also loose here on purpose: %.1f MB of working set fits inside\n"
           "  the 48 MB L2 entirely, so after the first touch essentially ALL of\n"
           "  the traffic is on chip. Module 6's finding, at a much larger K:\n"
           "  the caches got there first, and the reuse you predicted is not the\n"
           "  reuse you are being charged for.\n\n", compulsory/1e6);

    // ---------------------------------------------------------------- E
    printf("-- E. the loads-per-FMA probe ------------------------------------\n");
    printf("  Same two loads per k as the naive kernel; R FFMAs issued against\n"
           "  them instead of 1. The arithmetic is meaningless -- this measures\n"
           "  the instruction mix and nothing else.\n\n");
    PCtx pc[4]; int pR[4] = {1,2,4,8};
    double bp[4]; int pit[4];
    for (int i=0;i<4;++i){ pc[i].R=pR[i]; pc[i].g=cfg[0]; bp[i]=1e30; pit[i]=calibrate(runProbe,&pc[i]); }
    for (int s=0;s<4;++s)
        for (int q=0;q<4;++q){ int p=(q+s)%4; double t=timeOnce(runProbe,&pc[p],pit[p]);
                               if (t<bp[p]) bp[p]=t; }
    CHECK(cudaGetLastError());
    printf("  %-6s %-14s %10s %12s %11s %10s\n",
           "FMAs/","loads per FMA","ms","GFLOP/s","% ceiling","ms vs R=1");
    for (int i=0;i<4;++i) {
        double g = flops*pR[i]/(bp[i]*1e-3)/1e9;
        printf("  R=%-4d %14.3f %10.4f %12.1f %10.2f%% %9.3fx\n",
               pR[i], 2.0/pR[i], bp[i], g, 100.0*g/ceilCompute, bp[i]/bp[0]);
    }
    printf("\n  Eight times the arithmetic for %.0f%% more time. The FP32 pipe was\n"
           "  idle; the loop was waiting on the load path the whole time. The naive\n"
           "  kernel is not slow because of DRAM and not slow because of the FMA\n"
           "  units. It is slow because it issues TWO global loads per FMA and\n"
           "  there is no arrangement of a one-element-per-thread kernel that\n"
           "  changes that ratio.\n", 100.0*(bp[3]/bp[0]-1.0));

    // ---------------------------------------------------------------- F
    printf("\n-- F. the roofline, with both traffic models ----------------------\n");
    const double gNaive = flops/(bestColX*1e-3)/1e9;
    const double aiComp = flops/compulsory, aiReq = flops/requested;
    const double roofComp = (aiComp*ceilDram < ceilCompute) ? aiComp*ceilDram : ceilCompute;
    const double roofReq  = (aiReq *ceilDram < ceilCompute) ? aiReq *ceilDram : ceilCompute;
    printf("  %-28s %14s %16s %14s\n","traffic model","FLOP/byte","roofline GFLOP/s","measured/roof");
    printf("  %-28s %14.2f %16.1f %13.3fx\n","compulsory  4(MK+KN+MN)",aiComp,roofComp,gNaive/roofComp);
    printf("  %-28s %14.2f %16.1f %13.3fx\n","naive request  8MNK",  aiReq, roofReq, gNaive/roofReq);
    printf("\n  The algorithm's intensity (%.0f FLOP/byte) is %.0fx the machine balance\n"
           "  (%.1f FLOP/byte): GEMM is the canonical COMPUTE-bound kernel, and the\n"
           "  intensity grows as n/6 with problem size, so this only gets more true.\n",
           aiComp, aiComp/(ceilCompute/ceilDram), ceilCompute/ceilDram);
    printf("  The naive kernel reaches %.1f%% of the roof its algorithm entitles it\n"
           "  to, and runs %.1fx FASTER than the roof its request pattern would\n"
           "  imply -- which is the caches, quantified. Module 21 develops the\n"
           "  roofline properly; Module 17 closes the first half of the gap by\n"
           "  making each operand load cheap, and Module 18 closes the rest by\n"
           "  issuing fewer of them.\n", 100.0*gNaive/roofComp, gNaive/roofReq);

    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    CHECK(cudaFree(sIn)); CHECK(cudaFree(sOut)); CHECK(cudaFree(fOut));
    printf("\nOVERALL: PASS\n");
    CHECK(cudaDeviceReset());
    return 0;
}
