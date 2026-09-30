// =============================================================================
// Module 16 / Exercise 3 — SOLUTION
//                          the ledger, the roofline, and what has to change.
//
// GOAL : Write the analysis, not the kernel. You supply five functions that
//        together turn a stopwatch reading into a diagnosis: how many bytes the
//        problem must move, how many bytes the naive kernel asks for, where
//        that puts it on a roofline built from two ceilings measured on this
//        machine, how much of its traffic the caches must have absorbed, and
//        how much arithmetic per loaded value would be needed to reach the
//        compute roof.
//
//        The harness checks your arithmetic against hashed reference answers
//        on fixed synthetic inputs, then applies YOUR functions to real
//        measurements taken on this GPU and prints the diagnosis.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
// RUN  : exercise03_solution.exe
//
// TODO 1 - compulsoryBytes()
// TODO 2 - requestedBytes()
// TODO 3 - machineBalance() and rooflineGflops()
// TODO 4 - minOnChipFraction()
// TODO 5 - DESIGN: fmasPerLoadNeeded()
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

// =============================================================================
// TODO 1 — compulsoryBytes()
//
// The number of bytes that MUST cross the DRAM pins at least once for an
// fp32 GEMM C = A*B with A (M x K), B (K x N), C (M x N), counted the way
// Module 11 defined compulsory traffic: every distinct element that is read is
// read once no matter how many times the source text names it, every element
// that is written is written once, and an array that is both read and written
// counts twice. Assume beta == 0, so C is written and not read.
// =============================================================================
static double compulsoryBytes(int M, int N, int K)
{
    // SOLUTION: A read once, B read once, C written once. 4 bytes each.
    return 4.0 * ((double)M*K + (double)K*N + (double)M*N);
}

// =============================================================================
// TODO 2 — requestedBytes()
//
// The number of bytes the NAIVE one-thread-one-output kernel asks the memory
// system for, counted at the granularity of the load instruction's operand
// (4 bytes per lane per load), with no credit for any cache. Every thread runs
// a K-iteration loop containing one load from A and one load from B.
//
// This is not the same quantity as TODO 1 and the difference is the point of
// the module. It is also not the same as the number of SECTORS requested,
// which Exercise 2 counted and which is larger still for a bad mapping.
// =============================================================================
static double requestedBytes(int M, int N, int K)
{
    // SOLUTION: M*N threads, K iterations, 2 loads of 4 bytes each.
    return 4.0 * 2.0 * (double)M * (double)N * (double)K;
}

// =============================================================================
// TODO 3 — machineBalance() and rooflineGflops()
//
// machineBalance: the arithmetic intensity, in FLOP per byte, at which this
//   machine's compute ceiling and memory ceiling are reached simultaneously.
//   Below it a kernel is memory-bound; above it, compute-bound. Both ceilings
//   are passed in, measured, in GFLOP/s and GB/s.
//
// rooflineGflops: the highest throughput the roofline model permits for a
//   kernel of the given arithmetic intensity, given the same two ceilings.
// =============================================================================
static double machineBalance(double ceilGflops, double ceilGBs)
{
    // SOLUTION: the intensity at which the two ceilings intersect.
    // GFLOP/s / (GB/s) = FLOP/byte, the 1e9s cancel.
    return ceilGflops / ceilGBs;
}
static double rooflineGflops(double intensity, double ceilGflops, double ceilGBs)
{
    // SOLUTION: the roofline is the minimum of the flat compute roof and the
    // sloped bandwidth roof.
    const double mem = intensity * ceilGBs;
    return (mem < ceilGflops) ? mem : ceilGflops;
}

// =============================================================================
// TODO 4 — minOnChipFraction()
//
// ncu is unavailable on this machine, so no cache-hit counter can be read.
// Derive a bound instead, from two things that CAN be measured: the kernel's
// elapsed time and the DRAM streaming ceiling.
//
// Given `requested` bytes asked for, an elapsed time of `ms` milliseconds, and
// an UPPER BOUND `ceilGBs` on what DRAM can deliver, in GB/s, return a LOWER
// BOUND on the
// fraction of the requested bytes that must have been serviced somewhere other
// than DRAM. Return a value in [0, 1]; return 0 if no such bound follows.
//
// Be careful about which way the inequality runs, and make sure the function
// still behaves when the kernel is genuinely DRAM-bound.
// =============================================================================
static double minOnChipFraction(double requested, double ms, double ceilGBs)
{
    // SOLUTION. In `ms` milliseconds DRAM can deliver at most
    // ceilGBs*1e9 * ms*1e-3 bytes. Anything the kernel asked for beyond that
    // was served by L1 or L2. If the kernel asked for less than DRAM could
    // have delivered, nothing follows and the bound is 0.
    if (requested <= 0.0 || ms <= 0.0 || ceilGBs <= 0.0) return 0.0;
    const double fromDram = ceilGBs * 1.0e9 * ms * 1.0e-3;
    if (fromDram >= requested) return 0.0;
    return 1.0 - fromDram / requested;
}

// =============================================================================
// TODO 5 — fmasPerLoadNeeded().  DESIGN TODO.
//
// The harness measures a family of kernels that perform exactly the loads the
// naive GEMM performs and then issue R fused multiply-adds against each loaded
// pair instead of one. It hands you `n` observations:
//     fpl[i] = fused multiply-adds issued per global load
//     gf[i]  = the throughput measured at that ratio, in GFLOP/s
//
// Return the value of "FMAs per global load" at which this family would reach
// `targetGflops`. You decide how to turn the observations into an answer: the
// relationship is yours to identify from the data, not to be told. Return 0 if
// the observations do not support an answer.
//
// The number you get back is the whole point of the module, so once the
// harness prints it, read it against this fact: in a one-element-per-thread
// GEMM, every loaded value feeds exactly ONE multiply-add, forever, for any
// block shape, any grid, any launch configuration, on any GPU.
// =============================================================================
static double fmasPerLoadNeeded(double targetGflops, const double *fpl,
                                const double *gf, int n)
{
    // SOLUTION. The observations are proportional: throughput scales with
    // FMAs per load, because the load path -- not the FP32 pipe -- sets the
    // pace and the time per k iteration is unchanged. Fit the single constant
    // c in gf = c * fpl by least squares through the origin, then invert.
    if (n <= 0) return 0.0;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) { num += fpl[i]*gf[i]; den += fpl[i]*fpl[i]; }
    if (den <= 0.0 || num <= 0.0) return 0.0;
    const double c = num/den;                 // GFLOP/s per (FMA per load)
    return targetGflops / c;
}

// ---------------------------------------------------------------------------
// Kernels used by the harness. None of them is part of the exercise.
// ---------------------------------------------------------------------------
__global__ void gemmNaive(int M, int N, int K, const float *A, const float *B, float *C)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[(size_t)row*K + k] * B[(size_t)k*N + col];
    C[(size_t)row*N + col] = acc;
}
template <int R>
__global__ void fmaProbe(int M, int N, int K, const float *A, const float *B, float *C)
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
__global__ void ffmaCeiling(float *out, int iters)
{
    float a0=threadIdx.x,a1=a0+1,a2=a0+2,a3=a0+3,a4=a0+4,a5=a0+5,a6=a0+6,a7=a0+7;
    const float b=1.0000001f, c=0.9999999f;
    for (int i=0;i<iters;++i) {
        a0=fmaf(a0,b,c); a1=fmaf(a1,b,c); a2=fmaf(a2,b,c); a3=fmaf(a3,b,c);
        a4=fmaf(a4,b,c); a5=fmaf(a5,b,c); a6=fmaf(a6,b,c); a7=fmaf(a7,b,c);
    }
    float s=a0+a1+a2+a3+a4+a5+a6+a7;
    if (s == 1.2345e30f) out[0] = s;
}
__global__ void readCeiling(const float4 *in, float4 *out, size_t n4)
{
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x*blockDim.x;
    float4 s = make_float4(0,0,0,0);
    for (; i < n4; i += stride) { float4 v = in[i];
        s.x+=v.x; s.y+=v.y; s.z+=v.z; s.w+=v.w; }
    if (s.x+s.y+s.z+s.w == 1.2345e30f) out[0] = s;
}

// ---------------------------------------------------------------- FNV-1a
static unsigned fnv1a64(const long long *v, int n)
{
    unsigned h = 2166136261u;
    for (int i = 0; i < n; ++i) { unsigned long long x = (unsigned long long)v[i];
        for (int b = 0; b < 8; ++b) { h ^= (unsigned)((x >> (8*b)) & 0xFFull); h *= 16777619u; } }
    return h;
}
#define ANSWER_HASH 0x597e8670u

// ---------------------------------------------------------------- timing
typedef void (*RunFn)(void*);
static double timeIt(RunFn r, void *c, int iters)
{
    cudaEvent_t e0,e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    for (int i=0;i<iters;++i) r(c);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms,e0,e1));
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    return ms/iters;
}
static int calib(RunFn r, void *c)
{ double one = timeIt(r,c,1); int it=(int)(10.0/(one>0.0?one:0.01));
  if (it<1) it=1; if (it>64) it=64; return it; }

typedef struct { int M,N,K,R; const float *A,*B; float *C; } GC;
static void runNaive(void *p){ GC*g=(GC*)p; dim3 bl(16,16);
    dim3 gr((g->N+15)/16,(g->M+15)/16); gemmNaive<<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C); }
static void runProbe(void *p){ GC*g=(GC*)p; dim3 bl(16,16);
    dim3 gr((g->N+15)/16,(g->M+15)/16);
    switch (g->R) {
      case 1: fmaProbe<1><<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C); break;
      case 2: fmaProbe<2><<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C); break;
      case 4: fmaProbe<4><<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C); break;
      default:fmaProbe<8><<<gr,bl>>>(g->M,g->N,g->K,g->A,g->B,g->C); break; } }
typedef struct { float *o; int iters, grid; } FC;
static void runF(void *p){ FC*f=(FC*)p; ffmaCeiling<<<f->grid,256>>>(f->o,f->iters); }
typedef struct { const float4 *in; float4 *out; size_t n4; int grid; } SC;
static void runS(void *p){ SC*s=(SC*)p; readCeiling<<<s->grid,256>>>(s->in,s->out,s->n4); }

// =============================================================================
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const int M = M_DIM, N = N_DIM, K = K_DIM;
    printf("=== Module 16 / Exercise 3 — the ledger and the roofline ===\n");
    printf("C(%d x %d) = A(%d x %d) * B(%d x %d), fp32, beta = 0\n\n", M,N,M,K,K,N);

    // ------------------------------------------------ score the arithmetic
    // Fixed synthetic inputs, so the reference answers are constants and can
    // be hashed rather than printed.
    const double SYN_FPL[4] = { 0.5, 1.0, 2.0, 4.0 };
    const double SYN_GF [4] = { 1250.0, 2500.0, 5000.0, 10000.0 };

    const double a1 = compulsoryBytes(1024, 2048, 512);
    const double a2 = requestedBytes(1024, 2048, 512);
    const double a3 = machineBalance(18000.0, 410.0);
    const double a4 = rooflineGflops(2.0, 18000.0, 410.0);
    const double a5 = rooflineGflops(200.0, 18000.0, 410.0);
    const double a6 = minOnChipFraction(1.0e12, 1.0, 400.0);
    const double a7 = minOnChipFraction(1.0e8, 1.0, 400.0);
    const double a8 = fmasPerLoadNeeded(11250.0, SYN_FPL, SYN_GF, 4);

    if (a1 == 0.0 || a2 == 0.0) { printf("Set TODO 1 and TODO 2 first.\n"); return 0; }
    if (a3 == 0.0 || a4 == 0.0) { printf("Set TODO 3 first.\n"); return 0; }
    if (a8 == 0.0)              { printf("Set TODO 5 first.\n"); return 0; }

    long long enc[8];
    enc[0] = (long long)llround(a1);
    enc[1] = (long long)llround(a2);
    enc[2] = (long long)llround(a3 * 1000.0);
    enc[3] = (long long)llround(a4 * 1000.0);
    enc[4] = (long long)llround(a5 * 1000.0);
    enc[5] = (long long)llround(a6 * 100000.0);
    enc[6] = (long long)llround(a7 * 100000.0);
    enc[7] = (long long)llround(a8 * 1000.0);
    const unsigned h = fnv1a64(enc, 8);
    const int mathOK = (h == ANSWER_HASH);

    printf("-- your analysis functions, on fixed synthetic inputs --------------\n");
    printf("  compulsoryBytes(1024,2048,512)            = %.0f\n", a1);
    printf("  requestedBytes(1024,2048,512)             = %.0f\n", a2);
    printf("  machineBalance(18000 GFLOP/s, 410 GB/s)   = %.4f FLOP/byte\n", a3);
    printf("  rooflineGflops(2,   18000, 410)           = %.2f GFLOP/s\n", a4);
    printf("  rooflineGflops(200, 18000, 410)           = %.2f GFLOP/s\n", a5);
    printf("  minOnChipFraction(1e12 B, 1 ms, 400 GB/s) = %.5f\n", a6);
    printf("  minOnChipFraction(1e8  B, 1 ms, 400 GB/s) = %.5f\n", a7);
    printf("  fmasPerLoadNeeded(11250, synthetic)       = %.4f\n", a8);
    printf("  answer hash %08x -> %s\n\n", h, mathOK ? "all correct" : "at least one is WRONG");

    // ------------------------------------------------ measure
    const size_t sA=(size_t)M*K, sB=(size_t)K*N, sC=(size_t)M*N;
    float *dA,*dB,*dC;
    CHECK(cudaMalloc(&dA,sA*4)); CHECK(cudaMalloc(&dB,sB*4)); CHECK(cudaMalloc(&dC,sC*4));
    { float *h2=(float*)malloc((sA>sB?sA:sB)*4);
      for (size_t i=0;i<sA;++i) h2[i]=0.5f+(float)(i%251)/502.0f;
      CHECK(cudaMemcpy(dA,h2,sA*4,cudaMemcpyHostToDevice));
      for (size_t i=0;i<sB;++i) h2[i]=0.5f+(float)(i%257)/514.0f;
      CHECK(cudaMemcpy(dB,h2,sB*4,cudaMemcpyHostToDevice)); free(h2); }

    const size_t SB = 256u*1024u*1024u, n4 = SB/sizeof(float4);
    float4 *sIn,*sOut; CHECK(cudaMalloc(&sIn,SB)); CHECK(cudaMalloc(&sOut,SB));
    CHECK(cudaMemset(sIn,0x3c,SB));
    float *fo; CHECK(cudaMalloc(&fo,4));
    int bpsm=0, nsm=0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm,(const void*)ffmaCeiling,256,0));
    CHECK(cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0));
    FC fc = { fo, 50000, bpsm*nsm };
    SC sc = { sIn, sOut, n4, nsm*bpsm*2 };

    printf("  measuring: 1500 ms streaming warm-up (memory P-state), then\n"
           "  500 ms compute warm-up (SM clock), then rotated min-of-N sweeps ...\n");
    { cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0.0f; CHECK(cudaEventRecord(w0));
      while (el<1500.0f){ runS(&sc); CHECK(cudaEventRecord(w1));
        CHECK(cudaEventSynchronize(w1)); CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      el=0.0f; CHECK(cudaEventRecord(w0));
      while (el<500.0f){ runF(&fc); CHECK(cudaEventRecord(w1));
        CHECK(cudaEventSynchronize(w1)); CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); }

    double bF=1e30,bS=1e30; int iF=calib(runF,&fc), iS=calib(runS,&sc);
    for (int s=0;s<4;++s) {
        if (s&1){ double x=timeIt(runF,&fc,iF); if(x<bF)bF=x;
                  double y=timeIt(runS,&sc,iS); if(y<bS)bS=y; }
        else    { double y=timeIt(runS,&sc,iS); if(y<bS)bS=y;
                  double x=timeIt(runF,&fc,iF); if(x<bF)bF=x; }
    }
    const double ceilG = 2.0*8.0*fc.iters*256.0*fc.grid/(bF*1e-3)/1e9;
    const double ceilB = (double)SB/(bS*1e-3)/1e9;

    GC gc[5] = { {M,N,K,0,dA,dB,dC}, {M,N,K,1,dA,dB,dC}, {M,N,K,2,dA,dB,dC},
                 {M,N,K,4,dA,dB,dC}, {M,N,K,8,dA,dB,dC} };
    double bg[5]; int ig[5];
    bg[0]=1e30; ig[0]=calib(runNaive,&gc[0]);
    for (int i=1;i<5;++i){ bg[i]=1e30; ig[i]=calib(runProbe,&gc[i]); }
    for (int s=0;s<5;++s)
        for (int q=0;q<5;++q) {
            int p=(q+s)%5;
            double t = (p==0) ? timeIt(runNaive,&gc[0],ig[0]) : timeIt(runProbe,&gc[p],ig[p]);
            if (t<bg[p]) bg[p]=t;
        }
    CHECK(cudaGetLastError());

    const double flops = 2.0*(double)M*N*K;
    const double gNaive = flops/(bg[0]*1e-3)/1e9;

    printf("\n-- measured ceilings ---------------------------------------------\n");
    printf("  FP32 FFMA ceiling   %10.1f GFLOP/s\n", ceilG);
    printf("  DRAM read ceiling   %10.1f GB/s  (%.0f%% of the 432.0 GB/s pin peak)\n",
           ceilB, 100.0*ceilB/432.0);
    printf("  machine balance     %10.2f FLOP/byte   [your TODO 3]\n",
           machineBalance(ceilG, ceilB));

    const double comp = compulsoryBytes(M,N,K), req = requestedBytes(M,N,K);
    const double iComp = flops/comp, iReq = flops/req;
    printf("\n-- your ledger, applied to this problem ---------------------------\n");
    printf("  compulsory bytes            %14.0f  (%.2f MB)\n", comp, comp/1e6);
    printf("  requested bytes             %14.0f  (%.2f GB, %.0fx compulsory)\n",
           req, req/1e9, req/comp);
    printf("  intensity, compulsory model %14.2f FLOP/byte\n", iComp);
    printf("  intensity, requested model  %14.2f FLOP/byte\n", iReq);
    printf("  roofline, compulsory model  %14.1f GFLOP/s\n", rooflineGflops(iComp,ceilG,ceilB));
    printf("  roofline, requested model   %14.1f GFLOP/s\n", rooflineGflops(iReq,ceilG,ceilB));
    printf("  naive kernel measured       %14.1f GFLOP/s  (%.4f ms)\n", gNaive, bg[0]);
    printf("  fraction of the compulsory roof reached : %6.2f%%\n",
           100.0*gNaive/rooflineGflops(iComp,ceilG,ceilB));
    printf("  ratio to the requested roof             : %6.2fx  (above 1 is the caches)\n",
           gNaive/rooflineGflops(iReq,ceilG,ceilB));
    printf("  minimum on-chip service fraction        : %6.2f%%   [your TODO 4]\n",
           100.0*minOnChipFraction(req, bg[0], 432.0));
    printf("  (the bound uses the 432.0 GB/s pin peak, not the measured %.1f GB/s\n"
           "   streaming figure: a bound needs an UPPER bound on DRAM delivery,\n"
           "   and the measured figure moves with the memory P-state)\n", ceilB);

    printf("\n-- the FMAs-per-load family --------------------------------------\n");
    double fpl[4], gfm[4]; const int Rv[4] = {1,2,4,8};
    printf("  %-6s %-16s %10s %12s %11s\n","R","FMAs per load","ms","GFLOP/s","% ceiling");
    for (int i=0;i<4;++i) {
        fpl[i] = Rv[i]/2.0;                                   // 2 loads per k
        gfm[i] = flops*Rv[i]/(bg[i+1]*1e-3)/1e9;
        printf("  R=%-4d %16.3f %10.4f %12.1f %10.2f%%\n",
               Rv[i], fpl[i], bg[i+1], gfm[i], 100.0*gfm[i]/ceilG);
    }
    const double need80 = fmasPerLoadNeeded(0.80*ceilG, fpl, gfm, 4);
    printf("\n  FMAs per global load needed to reach 80%% of the compute ceiling:\n");
    printf("     %.2f      [your TODO 5]\n", need80);
    printf("  A one-element-per-thread GEMM supplies exactly 0.50 (two loads,\n"
           "  one FMA). That is a factor of %.0f, and no choice of block shape,\n"
           "  grid shape, mapping, or launch parameter changes it.\n", need80/0.5);

    printf("\n-- what is left on the table -------------------------------------\n");
    printf("  Exercise 2 showed the mapping is worth up to ~3.6x and that the\n"
           "  best one-element-per-thread kernel on this problem lands near\n"
           "  %.0f GFLOP/s. That is the ceiling of the ADDRESSING fix: it is\n"
           "  bounded by the R=1 row above, which is the same kernel.\n", gNaive);
    printf("  Everything beyond it requires each loaded value to feed more than\n"
           "  one multiply-add. There are exactly two things to arrange:\n"
           "    (1) make the operands cheap to re-read  -> stage them in a\n"
           "        block-scoped scratchpad instead of re-issuing global loads;\n"
           "    (2) make each thread reuse a loaded value across several\n"
           "        outputs it owns, in registers.\n"
           "  (1) is Module 17. (2) is Module 18. The FMAs-per-load number above\n"
           "  is how much of (2) you need; Module 6 already told you (1) alone\n"
           "  does not move a kernel whose instruction mix is unchanged.\n");

    const int score = mathOK;
    printf("\n  SCORE: %d/1  (all eight analysis answers must be right)\n", score);
    printf("OVERALL: %s\n", score ? "PASS" : "FAIL");

    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    CHECK(cudaFree(sIn)); CHECK(cudaFree(sOut)); CHECK(cudaFree(fo));
    CHECK(cudaDeviceReset());
    return score ? 0 : 1;
}
