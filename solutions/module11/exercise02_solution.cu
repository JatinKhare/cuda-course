// =====================================================================
// Module 11 / Exercise 2 -- SOLUTION
//   "Fusion, and the traffic you cannot avoid"
//
// GOAL
//   A four-stage elementwise pipeline is shipped below as four kernels.
//   Predict the speedup available from fusing it -- from traffic counting
//   alone, before you measure -- then build the fused version, then explain
//   the difference between your prediction and the measurement. The gap is
//   not noise and it is not in your favour by accident; you will be asked
//   for the mechanism.
//
// THE PIPELINE
//     stage 1   t1[i] = A*x[i] + B*y[i]
//     stage 2   m [i] = t1[i] + C*z[i]
//     stage 3   t3[i] = fmaxf(m[i], 0)
//     stage 4   d [i] = S*t3[i] + O
//
//   x, y, z are inputs. d is the output. `t1` and `t3` are private
//   temporaries -- nothing outside this pipeline ever reads them.
//
//   *** m IS ALSO A REQUIRED OUTPUT. *** A downstream consumer reads m.
//   You may not treat it as a temporary. This is the constraint that makes
//   the arithmetic interesting: fusion removes the traffic of values that
//   nobody outside the fused region needs, and only that traffic.
//
// WHY THIS MATTERS BEYOND THIS FILE
//   This is the seed of the fused-kernel argument that Parts XIV and XV
//   develop for transformer inference, where a residual-add / normalize /
//   activation / scale chain written as four library calls moves several
//   times the bytes of one hand-written kernel, and the model is memory
//   bound end to end. The arithmetic you do here is the same arithmetic.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
// RUN:   .\exercise02_solution.exe
//
// Worth running while you work:
//   nvcc -arch=sm_89 -O3 -Xptxas -v -cubin -o exercise02.cubin exercise02.cu
//   (the register counts are the input to TODO 5)
//
// WHAT IS CHECKED (7 points; all 7 required for OVERALL: PASS)
//   - TODO 1 and TODO 2 exactly (traffic, in units of N floats)
//   - the partially fused and fully fused kernels, on BOTH outputs
//   - TODO 4: your explanation of the prediction/measurement gap, to 5%
//   - TODO 5: your occupancy prediction
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(x) do {                                                      \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                  \
                cudaGetErrorName(e_), __FILE__, __LINE__,                  \
                cudaGetErrorString(e_));                                   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

static const double PEAK_GBS = 432.0;
static const long long N     = 1LL << 25;      // 33,554,432 floats = 128 MB
static const int  SWEEPS     = 4;

#define P_A  1.5f
#define P_B (-0.5f)
#define P_C  0.25f
#define P_S  2.0f
#define P_O  0.125f

#define GRID_STRIDE(n) \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n); i += (long long)gridDim.x * blockDim.x

// ------------------------------------------------ the pipeline, unfused
__global__ void s1(const float* __restrict__ x, const float* __restrict__ y,
                   float* __restrict__ t1, long long n)
{ for (GRID_STRIDE(n)) t1[i] = P_A*x[i] + P_B*y[i]; }

__global__ void s2(const float* __restrict__ t1, const float* __restrict__ z,
                   float* __restrict__ m, long long n)
{ for (GRID_STRIDE(n)) m[i] = t1[i] + P_C*z[i]; }

__global__ void s3(const float* __restrict__ m, float* __restrict__ t3, long long n)
{ for (GRID_STRIDE(n)) t3[i] = fmaxf(m[i], 0.0f); }

__global__ void s4(const float* __restrict__ t3, float* __restrict__ d, long long n)
{ for (GRID_STRIDE(n)) d[i] = P_S*t3[i] + P_O; }

// ---------------------------------------------------------------------
// TODO 1 (solved): 3 + 3 + 2 + 2 = 10.
static const int UNFUSED_TRAFFIC_N = 10;

// TODO 2 (solved): read x, y, z; write m and d. 3 + 2 = 5. The tempting
// answer is 4 (x, y, z in, d out) -- but m is a REQUIRED output and has
// to reach memory, so it cannot be fused away.
static const int FUSED_TRAFFIC_N   = 5;

// ---------------------------------------------------------------------
// TODO 3a (solved)
__global__ void chain_partial(const float* __restrict__ t1, const float* __restrict__ z,
                              float* __restrict__ m, float* __restrict__ d, long long n)
{
    for (GRID_STRIDE(n)) {
        const float mv = t1[i] + P_C*z[i];
        m[i] = mv;                                   // required output
        d[i] = P_S*fmaxf(mv, 0.0f) + P_O;            // t3 never reaches memory
    }
}

// TODO 3b (solved)
__global__ void chain_fused(const float* __restrict__ x, const float* __restrict__ y,
                            const float* __restrict__ z,
                            float* __restrict__ m, float* __restrict__ d, long long n)
{
    for (GRID_STRIDE(n)) {
        const float a1 = P_A*x[i] + P_B*y[i];        // t1 stays in a register
        const float mv = a1 + P_C*z[i];
        m[i] = mv;                                   // required output
        d[i] = P_S*fmaxf(mv, 0.0f) + P_O;            // t3 stays in a register
    }
}

// ---------------------------------------------------------------------
// TODO 4 (solved): time = traffic / bandwidth, so
//   speedup = (trafficU / bwU) / (trafficF / bwF)
//           = (trafficU / trafficF) * (bwF / bwU)
// The pure traffic model silently sets the second factor to 1. It is not 1:
// the fused kernel keeps more independent streams in flight per thread and
// therefore drives the bus harder than any single stage of the chain does.
static double explainSpeedup(double trafficRatio, double bwFused, double bwUnfused)
{
    return trafficRatio * (bwFused / bwUnfused);
}

// TODO 5 (solved): 0. The register count does rise with the number of fused
// streams, but nowhere near far enough to matter: a bandwidth-bound kernel
// saturates at a small fraction of full occupancy, so there is an enormous
// margin between the occupancy fusion costs and the occupancy it needs.
static const int PREDICT_OCCUPANCY_LIMITS = 0;


// Informational (not a TODO): how far does fusion keep paying? This kernel
// sums M input streams into one output, so its compulsory traffic is
// (M+1)N; the equivalent unfused chain of two-input stages is 3(M-1)N.
// The harness sweeps M and prints both, so that if register pressure or
// stream count ever turns fusion into a loss on this hardware, you will see
// the row where it happens.
template <int M>
__global__ void fuse_depth(const float* const* __restrict__ vs,
                           float* __restrict__ out, long long n)
{
    for (GRID_STRIDE(n)) {
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < M; ++k) acc += (float)(k+1) * vs[k][i];
        out[i] = acc;
    }
}

// ------------------------------------------------------------------ timing
template <typename L>
static double timeCfg(L launch, cudaEvent_t a, cudaEvent_t b)
{
    CHECK(cudaEventRecord(a)); launch(); CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float p = 0.f; CHECK(cudaEventElapsedTime(&p, a, b));
    if (p < 0.0005f) p = 0.0005f;
    int it = (int)(10.0 / p);
    if (it < 20)   it = 20;
    if (it > 5000) it = 5000;
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < it; ++i) launch();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
    return (double)ms / it;
}

// FNV-1a over a 32-bit value: lets the harness check TODO 1 and TODO 2
// without the answers appearing in this file.
static unsigned fnv1a32(unsigned v)
{
    unsigned h = 2166136261u;
    for (int i = 0; i < 4; ++i) { h ^= (v >> (8*i)) & 0xffu; h *= 16777619u; }
    return h;
}
static const unsigned UNFUSED_HASH = 0x6d506bbfu;
static const unsigned FUSED_HASH   = 0xbab8b9c0u;

static long long mism(const float* g, const float* r, long long n)
{
    long long bad = 0;
    for (long long i = 0; i < n; ++i)
        if (fabsf(g[i]-r[i]) > 1e-5f*fmaxf(1.0f, fabsf(r[i]))) ++bad;
    return bad;
}

// Max resident warps per SM given a register count, at `tpb` threads/block.
// Registers are allocated per warp in units of 8 on sm_89.
static int warpsPerSM(int regsPerThread, int tpb)
{
    if (regsPerThread <= 0) return 48;
    const int regsPerWarp = ((regsPerThread * 32 + 255) / 256) * 256;  // 256-reg granularity
    const int warpsByReg  = 65536 / regsPerWarp;
    const int warpsPerBlk = tpb / 32;
    int blocks = warpsByReg / warpsPerBlk;
    if (blocks > 24) blocks = 24;                      // 24 blocks/SM cap
    int warps = blocks * warpsPerBlk;
    if (warps > 48) warps = 48;                        // 1536 threads/SM cap
    return warps;
}

int main(void)
{
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    const int nSM = prop.multiProcessorCount;
    const int TPB = 256, GRID = nSM * 8;

    printf("Module 11 / Exercise 2 -- fusion and traffic accounting\n");
    printf("Device: %s, %d SMs\n", prop.name, nSM);
    printf("N = %lld, one array = %.0f MB, L2 = %.0f MB\n\n",
           N, N*4.0/1048576.0, prop.l2CacheSize/1048576.0);

    if (UNFUSED_TRAFFIC_N <= 0 || FUSED_TRAFFIC_N <= 0) {
        printf("Set TODO 1 and TODO 2 first.\n"); return 0;
    }
    if (PREDICT_OCCUPANCY_LIMITS < 0) { printf("Set TODO 5 first.\n"); return 0; }

    const size_t bs = (size_t)N * sizeof(float);
    float *h_x = (float*)malloc(bs), *h_y = (float*)malloc(bs), *h_z = (float*)malloc(bs);
    float *h_rm= (float*)malloc(bs), *h_rd= (float*)malloc(bs), *h_g = (float*)malloc(bs);
    if (!h_x||!h_y||!h_z||!h_rm||!h_rd||!h_g) { printf("host allocation failed\n"); return 1; }
    for (long long i = 0; i < N; ++i) {
        h_x[i] = (float)(((i*1103515245LL + 12345LL) % 2003) - 1001) * 0.001f;
        h_y[i] = (float)(((i*22695477LL   + 1LL)     % 1999) -  999) * 0.001f;
        h_z[i] = (float)(((i*69069LL      + 5LL)     % 1997) -  998) * 0.001f;
        const float a1 = P_A*h_x[i] + P_B*h_y[i];
        h_rm[i] = a1 + P_C*h_z[i];
        h_rd[i] = P_S*fmaxf(h_rm[i], 0.0f) + P_O;
    }

    float *x,*y,*z,*t1,*t3,*m,*d;
    CHECK(cudaMalloc(&x,bs)); CHECK(cudaMalloc(&y,bs)); CHECK(cudaMalloc(&z,bs));
    CHECK(cudaMalloc(&t1,bs)); CHECK(cudaMalloc(&t3,bs));
    CHECK(cudaMalloc(&m,bs)); CHECK(cudaMalloc(&d,bs));
    CHECK(cudaMemcpy(x,h_x,bs,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y,h_y,bs,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(z,h_z,bs,cudaMemcpyHostToDevice));

    cudaEvent_t e0,e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    { float acc=0.f;
      while (acc < 400.f) {
        CHECK(cudaEventRecord(e0));
        for (int i=0;i<20;++i) s1<<<GRID,TPB>>>(x,y,t1,N);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms=0.f; CHECK(cudaEventElapsedTime(&ms,e0,e1)); acc+=ms; } }
    CHECK(cudaGetLastError());

    // ---- timing: three versions, back to back, rotated, min of SWEEPS ----
    double tU = 1e30, tP = 1e30, tF = 1e30;
    for (int s = 0; s < SWEEPS; ++s) {
        for (int q = 0; q < 3; ++q) {
            switch ((q+s) % 3) {
            case 0: { double ms = timeCfg([&]{
                          s1<<<GRID,TPB>>>(x,y,t1,N);
                          s2<<<GRID,TPB>>>(t1,z,m,N);
                          s3<<<GRID,TPB>>>(m,t3,N);
                          s4<<<GRID,TPB>>>(t3,d,N); }, e0,e1);
                      if (ms<tU) tU=ms; } break;
            case 1: { double ms = timeCfg([&]{
                          s1<<<GRID,TPB>>>(x,y,t1,N);
                          chain_partial<<<GRID,TPB>>>(t1,z,m,d,N); }, e0,e1);
                      if (ms<tP) tP=ms; } break;
            default:{ double ms = timeCfg([&]{
                          chain_fused<<<GRID,TPB>>>(x,y,z,m,d,N); }, e0,e1);
                      if (ms<tF) tF=ms; } break;
            }
        }
    }
    CHECK(cudaGetLastError());

    // ---- validation, untimed second pass ----
    printf("=== validation (untimed second pass) ===\n");
    int pts = 0; const int maxpts = 7;

    const bool u_ok = (fnv1a32((unsigned)UNFUSED_TRAFFIC_N) == UNFUSED_HASH);
    const bool f_ok = (fnv1a32((unsigned)FUSED_TRAFFIC_N)   == FUSED_HASH);
    printf("  [%s] TODO 1 unfused traffic = %dN\n", u_ok?"ok":"  ", UNFUSED_TRAFFIC_N);
    if (u_ok) ++pts;
    printf("  [%s] TODO 2 fused traffic   = %dN\n", f_ok?"ok":"  ", FUSED_TRAFFIC_N);
    if (f_ok) ++pts;

    CHECK(cudaMemset(m,0,bs)); CHECK(cudaMemset(d,0,bs));
    s1<<<GRID,TPB>>>(x,y,t1,N);
    chain_partial<<<GRID,TPB>>>(t1,z,m,d,N);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_g,m,bs,cudaMemcpyDeviceToHost)); long long bpm = mism(h_g,h_rm,N);
    CHECK(cudaMemcpy(h_g,d,bs,cudaMemcpyDeviceToHost)); long long bpd = mism(h_g,h_rd,N);
    printf("  [%s] chain_partial: m %lld bad, d %lld bad\n",
           (bpm||bpd)?"  ":"ok", bpm, bpd);
    if (!bpm && !bpd) ++pts;

    CHECK(cudaMemset(m,0,bs)); CHECK(cudaMemset(d,0,bs));
    chain_fused<<<GRID,TPB>>>(x,y,z,m,d,N);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_g,m,bs,cudaMemcpyDeviceToHost)); long long bfm = mism(h_g,h_rm,N);
    CHECK(cudaMemcpy(h_g,d,bs,cudaMemcpyDeviceToHost)); long long bfd = mism(h_g,h_rd,N);
    printf("  [%s] chain_fused:   m %lld bad, d %lld bad\n",
           (bfm||bfd)?"  ":"ok", bfm, bfd);
    if (!bfm && !bfd) ++pts;
    if (bfm && !bfd)
        printf("       (d is right and m is wrong -- you fused away a required output.)\n");

    // ---- the numbers ----
    // Bandwidth is computed against YOUR traffic numbers, so a wrong TODO 1
    // or TODO 2 shows up as an implausible GB/s rather than being hidden.
    const double bytesU = (double)UNFUSED_TRAFFIC_N * 4.0 * (double)N;
    const double bytesF = (double)FUSED_TRAFFIC_N   * 4.0 * (double)N;
    // the partially fused version writes m and d but still materialises t1
    const double bytesP = bytesF + 2.0*4.0*(double)N;
    const double bwU = bytesU/(tU*1e-3)/1e9;
    const double bwP = bytesP/(tP*1e-3)/1e9;
    const double bwF = bytesF/(tF*1e-3)/1e9;

    printf("\n=== performance ===\n");
    printf("  %-26s %9s %9s %10s %10s\n","version","ms","traffic","GB/s","%ofpeak");
    printf("  %-26s %9.4f %8.0fN %10.1f %9.1f%%\n","unfused (4 kernels)",tU,bytesU/4.0/N,bwU,100*bwU/PEAK_GBS);
    printf("  %-26s %9.4f %8.0fN %10.1f %9.1f%%\n","partial (2 kernels)",tP,bytesP/4.0/N,bwP,100*bwP/PEAK_GBS);
    printf("  %-26s %9.4f %8.0fN %10.1f %9.1f%%\n","fused   (1 kernel)", tF,bytesF/4.0/N,bwF,100*bwF/PEAK_GBS);

    const double predicted = (double)UNFUSED_TRAFFIC_N / (double)FUSED_TRAFFIC_N;
    const double measured  = tU / tF;
    printf("\n  your traffic prediction : %.3fx\n", predicted);
    printf("  measured                : %.3fx\n", measured);
    printf("  gap                     : %+.1f%%\n", 100.0*(measured/predicted - 1.0));

    const double explained = explainSpeedup(predicted, bwF, bwU);
    const bool expOk = (explained > 0.0) && (fabs(explained - measured) <= 0.05*measured);
    printf("  [%s] TODO 4 explained speedup = %.3fx vs measured %.3fx (+-5%%)\n",
           expOk?"ok":"  ", explained, measured);
    if (expOk) ++pts;

    const bool gateOk = (measured >= 1.70);
    printf("  [%s] fused is %.3fx the unfused chain (gate: 1.70x)\n",
           gateOk?"ok":"  ", measured);
    if (gateOk) ++pts;


    // ---- occupancy arithmetic ----
    cudaFuncAttributes aU, aF;
    CHECK(cudaFuncGetAttributes(&aU, (const void*)s1));
    CHECK(cudaFuncGetAttributes(&aF, (const void*)chain_fused));
    const int wU = warpsPerSM(aU.numRegs, TPB), wF = warpsPerSM(aF.numRegs, TPB);
    printf("\n  registers/thread : s1 = %d, chain_fused = %d\n", aU.numRegs, aF.numRegs);
    printf("  max resident warps/SM at %d threads/block : %d vs %d (of 48)\n", TPB, wU, wF);
    const int actual = (wF < 16) ? 1 : 0;   // Example 2: 16 warps/SM is already on the plateau
    printf("  [%s] TODO 5 occupancy prediction = %d, truth = %d\n",
           (PREDICT_OCCUPANCY_LIMITS==actual)?"ok":"  ", PREDICT_OCCUPANCY_LIMITS, actual);
    if (PREDICT_OCCUPANCY_LIMITS==actual) ++pts;


    // ---- informational: fusion depth ----
    {
        // Six DISTINCT input arrays, and d as the output. Repeating an
        // array in this list would let the second read hit in cache and
        // would inflate the GB/s column by exactly the repeat factor.
        const float* hv[6] = { x, y, z, t1, t3, m };
        const float** dvs = NULL;
        CHECK(cudaMalloc((void**)&dvs, 6*sizeof(float*)));
        CHECK(cudaMemcpy((void*)dvs, hv, 6*sizeof(float*), cudaMemcpyHostToDevice));
        const int MS[5] = { 2, 3, 4, 5, 6 };
        double bd[5]; for (int i = 0; i < 5; ++i) bd[i] = 1e30;
        // 6 sweeps over 5 configurations, so every configuration leads a
        // sweep at least once (spec 12.9 -- with SWEEPS < NCFG the first
        // configuration keeps an unfair early-and-cool sample).
        for (int s = 0; s < 6; ++s) {
            for (int q = 0; q < 5; ++q) {
                const int i = (q+s) % 5; double ms = 0.0;
                switch (MS[i]) {
                case 2: ms = timeCfg([&]{ fuse_depth<2><<<GRID,TPB>>>(dvs,d,N); },e0,e1); break;
                case 3: ms = timeCfg([&]{ fuse_depth<3><<<GRID,TPB>>>(dvs,d,N); },e0,e1); break;
                case 4: ms = timeCfg([&]{ fuse_depth<4><<<GRID,TPB>>>(dvs,d,N); },e0,e1); break;
                case 5: ms = timeCfg([&]{ fuse_depth<5><<<GRID,TPB>>>(dvs,d,N); },e0,e1); break;
                default:ms = timeCfg([&]{ fuse_depth<6><<<GRID,TPB>>>(dvs,d,N); },e0,e1); break;
                }
                if (ms < bd[i]) bd[i] = ms;
            }
        }
        CHECK(cudaGetLastError());
        printf("\n=== how deep does fusion keep paying? (informational) ===\n");
        printf("  %6s %10s %10s %12s %12s\n","inputs","traffic","ms","GB/s","vs M=2");
        const double b0 = 3.0*4.0*(double)N/(bd[0]*1e-3)/1e9;
        for (int i = 0; i < 5; ++i) {
            const double nb = ((double)MS[i]+1.0)*4.0*(double)N;
            const double bw = nb/(bd[i]*1e-3)/1e9;
            printf("  %6d %9.0fN %10.4f %12.1f %12.3f\n", MS[i], nb/4.0/N, bd[i], bw, bw/b0);
        }
        printf("  A falling GB/s column would be the point at which fusing more\n"
               "  stops paying. Read it together with the register counts above.\n");
        CHECK(cudaFree((void*)dvs));
    }

    printf("\nScore: %d/%d\n", pts, maxpts);
    printf("OVERALL: %s\n", (pts==maxpts)?"PASS":"FAIL");

    free(h_x);free(h_y);free(h_z);free(h_rm);free(h_rd);free(h_g);
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(x));CHECK(cudaFree(y));CHECK(cudaFree(z));
    CHECK(cudaFree(t1));CHECK(cudaFree(t3));CHECK(cudaFree(m));CHECK(cudaFree(d));
    CHECK(cudaDeviceReset());
    return (pts==maxpts)?0:1;
}
