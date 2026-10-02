// =============================================================================
// Module 21 / Exercise 3 — design to a target fraction of peak.
//
// THE BRIEF
//   A 17x17 convolution over a 2048x2048 image. The shipped `convNaive` is
//   correct and runs at about an eighth of this GPU's FP32 ceiling. Make it
//   reach 28% of the ceiling, and at least 2.5x the naive kernel.
//
//   You are not told which optimization to apply. Derive it. Build the
//   roofline, compute the naive kernel's arithmetic intensity at the level
//   that binds it, work out what intensity the target requires, and let the
//   difference tell you what has to change about the DECOMPOSITION.
//
//   Two warnings, both measured:
//     * one of the obvious moves buys nothing at all here. Staging the input
//       through shared memory changes which cache answers a load; it does not
//       change how many loads there are per FLOP, so it does not move the
//       kernel along the axis that binds it. Check your analysis against that
//       before you write code.
//     * more of the move that does work is not monotonically better. The
//       resource that stops you is not the one you are optimizing.
//
// WHAT TO FILL IN
//   TODO 1  aiNaive(), aiNeeded(), minOutputsPerThread()
//   TODO 2  TILE_X, TILE_Y, OPT -- the decomposition             (DESIGN)
//   TODO 3  the cooperative load into shared memory
//   TODO 4  the accumulation loop
//   TODO 5  PRED_BUCKET -- predicted fraction of the FP32 ceiling
//
// SCORING: 9 points. OVERALL: PASS requires all nine.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN  : exercise03.exe
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

#define RAD   8                       // filter radius
#define DIAM  (2*RAD+1)               // 17 x 17 = 289 taps
#define IW    2048
#define IH    2048
#define TARGET_FRAC 0.28              // of the measured FP32 ceiling
#define TARGET_SPEEDUP 2.5            // over the shipped naive kernel

__constant__ float cW[DIAM*DIAM];

// =============================================================================
// The baseline. Correct, straightforward, and 12-13% of the compute ceiling.
// =============================================================================
__global__ void convNaive(const float * __restrict__ in, float *out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= IW || y >= IH) return;
    float acc = 0.f;
    #pragma unroll
    for (int dy = -RAD; dy <= RAD; ++dy) {
        int yy = min(max(y+dy,0),IH-1);
        #pragma unroll
        for (int dx = -RAD; dx <= RAD; ++dx) {
            int xx = min(max(x+dx,0),IW-1);
            acc = fmaf(in[(size_t)yy*IW+xx], cW[(dy+RAD)*DIAM+(dx+RAD)], acc);
        }
    }
    out[(size_t)y*IW+x] = acc;
}

// =============================================================================
// TODO 1 — the analysis. Do this before you write a single line of kernel.
// =============================================================================
// (a) The arithmetic intensity of convNaive, in FLOP per byte, measured at the
//     level where its operands are actually fetched. Read the inner loop and
//     count: how many bytes does one tap ask the memory system for, and how
//     many floating-point operations does it produce? Note where the WEIGHT
//     comes from -- check the SASS if you are unsure whether it costs a load:
//       nvcc -arch=sm_89 -O3 -cubin -o x.cubin exercise03.cu
//       cuobjdump -sass x.cubin
static double aiNaive(void)
{
    // TODO 1a: YOUR CODE HERE
    return -1.0;
}

// (b) The arithmetic intensity a kernel must have to reach `frac` of the
//     compute ceiling, on a roofline whose sloped ceiling at this level is
//     `ceilOnChipGBs` and whose plateau is `ceilFp32GF`.
static double aiNeeded(double frac, double ceilFp32GF, double ceilOnChipGBs)
{
    // TODO 1b: YOUR CODE HERE
    (void)frac; (void)ceilFp32GF; (void)ceilOnChipGBs;
    return -1.0;
}

// (c) The smallest number of outputs one thread must produce to get there,
//     under the simple model that a thread computing P outputs reuses each
//     value it loads across all P of them. Round the way a design decision has
//     to be rounded.
static int minOutputsPerThread(double needed, double naive)
{
    // TODO 1c: YOUR CODE HERE
    (void)needed; (void)naive;
    return 0;
}

// =============================================================================
// TODO 2 — DESIGN. The tile shape and the blocking factor.
//
// Convention the harness and the kernel skeleton assume: TILE_X x TILE_Y
// threads per block, each thread producing OPT outputs stacked VERTICALLY, so
// one block covers TILE_X columns by TILE_Y*OPT rows of output and must stage
// a (TILE_Y*OPT + 2*RAD) x (TILE_X + 2*RAD) halo tile.
//
// Choose all three. Things worth being deliberate about:
//   * TILE_X decides whether the global loads of the halo tile coalesce;
//   * OPT must be at least what your own TODO 1c demands -- check 2 compares
//     them -- but the halo tax and the register and shared-memory footprint
//     all grow with it, and one of those three bites before you expect;
//   * TILE_X * TILE_Y must not exceed 1024.
// =============================================================================
#define TILE_X 0
#define TILE_Y 0
#define OPT    0

#define SH_W   ((TILE_X + 2*RAD) > 0 ? (TILE_X + 2*RAD) : 1)
#define SH_H   ((TILE_Y*OPT + 2*RAD) > 0 ? (TILE_Y*OPT + 2*RAD) : 1)

// =============================================================================
// TODO 3 and TODO 4 — the kernel.
//
// The output must be numerically identical to convNaive's to within the
// tolerance the harness applies, and EVERY pixel must be written -- the
// harness prefills the output with +infinity, so an unwritten pixel is caught
// rather than silently passing (Module 16's writtenness check).
//
// The image is 2048x2048, which is a multiple of nothing in particular once
// you have chosen a tile, so handle the boundary. The naive kernel clamps at
// the image edge; match it.
// =============================================================================
__global__ void convFast(const float * __restrict__ in, float *out)
{
    __shared__ float s[SH_H][SH_W];
    const int x0 = blockIdx.x*TILE_X, y0 = blockIdx.y*(TILE_Y*OPT);
    const int tid = threadIdx.y*TILE_X + threadIdx.x, nthr = TILE_X*TILE_Y;
    (void)x0; (void)y0; (void)tid; (void)nthr; (void)in;
    if (tid < 0) { out[0] = s[0][0]; return; }   // keeps `s` live before TODO 3

    // TODO 3: stage the (SH_H x SH_W) halo tile into `s`. There are
    //         TILE_X*TILE_Y threads and SH_H*SH_W cells and the two numbers
    //         are not equal, so the load mapping is not the compute mapping
    //         (Module 6). Clamp at the image boundary exactly as convNaive
    //         does. Then make the tile visible to the whole block.
    // YOUR CODE HERE

    // TODO 4: compute OPT outputs per thread. This is where the arithmetic
    //         intensity you designed in TODO 1 either happens or does not:
    //         the number of shared-memory reads per fused multiply-add is
    //         decided entirely by the order of these loops. Writing the
    //         obvious loop nest -- for each output, for each tap -- gives you
    //         back the naive kernel's intensity with extra steps.
    // YOUR CODE HERE

    // TODO 4 (continued): write the OPT results, guarded.
    // YOUR CODE HERE
}

// =============================================================================
// TODO 5 — predicted achieved fraction of the FP32 ceiling for YOUR kernel,
// as a bucket. Commit before you run it.
//   1: < 5%    2: 5-15%    3: 15-28%    4: 28-50%    5: > 50%
// =============================================================================
static const int PRED_BUCKET = 0;

// =============================================================================
// Harness below this line.
// =============================================================================
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

static unsigned fnvNums(const double *v, int n, double scale)
{ char buf[64]; unsigned h = 2166136261u;
  for (int i = 0; i < n; ++i) { snprintf(buf, sizeof buf, "%lld", (long long)llround(v[i]*scale));
      for (char *p = buf; *p; ++p) { h ^= (unsigned char)*p; h *= 16777619u; } }
  return h; }

static float *dIn, *dOut, *dSink; static float4 *dWarm;
static double timeFn(void (*f)(void), int iters)
{
    cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) f();
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms,a,b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms/iters;
}
static void kNaive(void) { dim3 b(32,8), g((IW+31)/32,(IH+7)/8); convNaive<<<g,b>>>(dIn,dOut); }
static void kFast (void) {
    // TILE_X/TILE_Y/OPT are 0 until TODO 2 is filled in; main() refuses to get
    // here in that case, but the grid arithmetic must still compile.
    const int tx = (TILE_X > 0) ? TILE_X : 1;
    const int ty = (TILE_Y > 0) ? TILE_Y : 1;
    const int op = (OPT    > 0) ? OPT    : 1;
    dim3 b(tx,ty), g((IW+tx-1)/tx, (IH+ty*op-1)/(ty*op));
    convFast<<<g,b>>>(dIn,dOut);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 21 / Exercise 3 - design to a target ===\n\n");

    if (aiNaive() <= 0.0 || aiNeeded(0.3,18000.0,5400.0) <= 0.0 ||
        minOutputsPerThread(1.0,0.5) <= 0) { printf("Set TODO 1 first.\n"); return 0; }
    if (TILE_X <= 0 || TILE_Y <= 0 || OPT <= 0) { printf("Set TODO 2 first.\n"); return 0; }
    if (TILE_X*TILE_Y > 1024) { printf("TILE_X*TILE_Y exceeds 1024 threads.\n"); return 0; }
    if (PRED_BUCKET < 1) { printf("Set TODO 5 first.\n"); return 0; }

    float *hIn = (float*)malloc(sizeof(float)*IW*IH);
    float *hW  = (float*)malloc(sizeof(float)*DIAM*DIAM);
    unsigned st = 3u;
    for (size_t i = 0; i < (size_t)IW*IH; ++i) { st = st*1664525u+1013904223u;
        hIn[i] = (float)((st>>8)&1023)/1024.0f; }
    double wsum = 0.0;
    for (int i = 0; i < DIAM*DIAM; ++i) { st = st*1664525u+1013904223u;
        hW[i] = (float)((st>>8)&255)/256.0f; wsum += hW[i]; }
    for (int i = 0; i < DIAM*DIAM; ++i) hW[i] = (float)(hW[i]/wsum);

    CHECK(cudaMalloc(&dIn,  sizeof(float)*IW*IH));
    CHECK(cudaMalloc(&dOut, sizeof(float)*IW*IH));
    CHECK(cudaMalloc(&dSink, 16));
    CHECK(cudaMalloc(&dWarm, (size_t)256*1024*1024));
    CHECK(cudaMemcpy(dIn, hIn, sizeof(float)*IW*IH, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(cW, hW, sizeof(float)*DIAM*DIAM));
    CHECK(cudaMemset(dWarm, 1, (size_t)256*1024*1024));

    // ---- ceilings: 1500 ms stream, then 500 ms compute (spec SS12.4) -------
    { cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0; CHECK(cudaEventRecord(w0));
      while (el<1500.f){ streamRead<<<640,256>>>(dWarm,dSink,(size_t)256*1024*1024/16);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      el=0; CHECK(cudaEventRecord(w0));
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

    const double need = aiNeeded(TARGET_FRAC, CEIL_FP32, CEIL_SMEM);
    const int    minP = minOutputsPerThread(need, aiNaive());
    printf("-- the design brief ---------------------------------------------------\n");
    printf("  %dx%d image, %dx%d filter (%d taps), %.2f GFLOP of useful work\n",
           IW, IH, DIAM, DIAM, DIAM*DIAM, 2.0*IW*IH*DIAM*DIAM/1e9);
    printf("  measured FP32 ceiling          %10.1f GFLOP/s\n", CEIL_FP32);
    printf("  measured on-chip fetch ceiling %10.1f GB/s\n", CEIL_SMEM);
    printf("  on-chip ridge                  %10.2f FLOP/byte\n", CEIL_FP32/CEIL_SMEM);
    printf("  naive kernel AI                %10.2f FLOP/byte -> %.1f%% of the ceiling\n",
           aiNaive(), 100.0*aiNaive()/(CEIL_FP32/CEIL_SMEM));
    printf("  TARGET %.0f%% of the ceiling needs AI >= %.2f FLOP/byte, i.e. at least\n",
           100.0*TARGET_FRAC, need);
    printf("  %d outputs per thread. You chose TILE_X=%d TILE_Y=%d OPT=%d.\n\n",
           minP, TILE_X, TILE_Y, OPT);

    // ---- timing: both kernels back to back, rotated ------------------------
    void (*run[2])(void) = { kNaive, kFast };
    int it[2]; double best[2];
    for (int i=0;i<2;++i){ double t=timeFn(run[i],1);
      int n=(int)(10.0/(t>1e-3?t:1e-3)); if(n<3)n=3; if(n>64)n=64; it[i]=n; best[i]=1e30; }
    CHECK(cudaGetLastError());
    for (int s=0;s<4;++s) for (int q=0;q<2;++q){ int p=(q+s)%2;
      double t=timeFn(run[p],it[p]); if (t<best[p]) best[p]=t; }
    CHECK(cudaGetLastError());

    const double fl = 2.0*(double)IW*IH*DIAM*DIAM;
    const double gN = fl/(best[0]*1e-3)/1e9, gF = fl/(best[1]*1e-3)/1e9;
    const double pctN = 100.0*gN/CEIL_FP32, pctF = 100.0*gF/CEIL_FP32;
    printf("-- measurement --------------------------------------------------------\n");
    printf("  %-10s %10s %12s %10s %10s\n","kernel","ms","GFLOP/s","%ceiling","x naive");
    printf("  %-10s %10.4f %12.1f %9.1f%% %10.2f\n","naive", best[0], gN, pctN, 1.0);
    printf("  %-10s %10.4f %12.1f %9.1f%% %10.2f\n","yours", best[1], gF, pctF,
           best[0]/best[1]);
    { cudaFuncAttributes at; CHECK(cudaFuncGetAttributes(&at,(const void*)convFast));
      int bpsm=0; CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &bpsm,(const void*)convFast, TILE_X*TILE_Y, 0));
      printf("  yours: %d registers, %d B spilled, %d B shared, %d blocks/SM\n",
             at.numRegs, (int)at.localSizeBytes, (int)at.sharedSizeBytes, bpsm); }

    // ---- validation (second, untimed pass) --------------------------------
    printf("\n-- validation (second, untimed pass) ----------------------------------\n");
    float *hA = (float*)malloc(sizeof(float)*IW*IH);
    float *hB = (float*)malloc(sizeof(float)*IW*IH);
    { float *po = (float*)malloc(sizeof(float)*IW*IH);
      for (size_t i=0;i<(size_t)IW*IH;++i) po[i]=std::numeric_limits<float>::infinity();
      CHECK(cudaMemcpy(dOut,po,sizeof(float)*IW*IH,cudaMemcpyHostToDevice));
      kNaive(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(hA,dOut,sizeof(float)*IW*IH,cudaMemcpyDeviceToHost));
      for (size_t i=0;i<(size_t)IW*IH;++i) po[i]=std::numeric_limits<float>::infinity();
      CHECK(cudaMemcpy(dOut,po,sizeof(float)*IW*IH,cudaMemcpyHostToDevice));
      kFast(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(hB,dOut,sizeof(float)*IW*IH,cudaMemcpyDeviceToHost));
      free(po); }

    size_t unwritten = 0;
    for (size_t i=0;i<(size_t)IW*IH;++i) if (!isfinite(hB[i])) ++unwritten;
    printf("  pixels your kernel never wrote : %llu\n", (unsigned long long)unwritten);

    // Sampled exact double reference with the gamma_n * S tolerance (M16).
    const double u = ldexp(1.0,-24);
    const double gam = (double)(DIAM*DIAM)*u/(1.0-(double)(DIAM*DIAM)*u);
    double worst = 0.0; long bad = 0;
    for (int y = 3; y < IH; y += 37)
      for (int x = 5; x < IW; x += 41) {
        double ref = 0.0, S = 0.0;
        for (int dy=-RAD; dy<=RAD; ++dy) { int yy=(y+dy<0)?0:((y+dy>=IH)?IH-1:y+dy);
          for (int dx=-RAD; dx<=RAD; ++dx) { int xx=(x+dx<0)?0:((x+dx>=IW)?IW-1:x+dx);
            double a=hIn[(size_t)yy*IW+xx], w=hW[(dy+RAD)*DIAM+(dx+RAD)];
            ref += a*w; S += fabs(a)*fabs(w); } }
        double e = fabs((double)hB[(size_t)y*IW+x] - ref);
        double tol = 4.0*gam*S;
        if (tol > 0 && e/tol > worst) worst = e/tol;
        if (e > tol) ++bad; }
    printf("  sampled scaled error (1.0 = at tolerance): %.4f, %ld samples over\n",
           worst, bad);
    double dmax = 0.0;
    for (size_t i=0;i<(size_t)IW*IH;++i) { double d=fabs((double)hA[i]-(double)hB[i]);
        if (d>dmax) dmax=d; }
    printf("  max |yours - naive| over the whole image : %.3e\n", dmax);

    // ---- scoring -----------------------------------------------------------
    int score = 0;
    printf("\n-- scoring -----------------------------------------------------------\n");
    { double v[5] = { aiNaive(), aiNeeded(0.30,18000.0,5400.0),
                      aiNeeded(0.80,18000.0,5400.0),
                      (double)minOutputsPerThread(aiNeeded(0.30,18000.0,5400.0), aiNaive()),
                      (double)minOutputsPerThread(aiNeeded(0.80,18000.0,5400.0), aiNaive()) };
      unsigned h = fnvNums(v,5,1000.0); int good = (h == 1886290246u);
      printf("  [%s] 1. aiNaive / aiNeeded / minOutputsPerThread (2 pts)\n", good?"x":" ");
      score += good ? 2 : 0; }
    { int good = (OPT >= minP);
      printf("  [%s] 2. OPT (%d) is at least what your own analysis demands (%d)\n",
             good?"x":" ", OPT, minP); score += good; }
    { int good = (unwritten == 0 && bad == 0 && dmax < 1e-4);
      printf("  [%s] 3. correct: every pixel written, inside tolerance (2 pts)\n",
             good?"x":" "); score += good ? 2 : 0; }
    { int good = (best[0]/best[1] >= TARGET_SPEEDUP);
      printf("  [%s] 4. at least %.1fx the naive kernel (got %.2fx) (2 pts)\n",
             good?"x":" ", TARGET_SPEEDUP, best[0]/best[1]); score += good ? 2 : 0; }
    { int good = (pctF >= 100.0*TARGET_FRAC);
      printf("  [%s] 5. at least %.0f%% of the measured FP32 ceiling (got %.1f%%)\n",
             good?"x":" ", 100.0*TARGET_FRAC, pctF); score += good; }
    { int b = (pctF < 5.0) ? 1 : (pctF < 15.0) ? 2 : (pctF < 28.0) ? 3
            : (pctF < 50.0) ? 4 : 5;
      int good = (PRED_BUCKET == b);
      printf("  [%s] 6. predicted bucket %d, measured bucket %d\n",
             good?"x":" ", PRED_BUCKET, b); score += good; }

    free(hIn); free(hW); free(hA); free(hB);
    CHECK(cudaFree(dIn)); CHECK(cudaFree(dOut)); CHECK(cudaFree(dSink)); CHECK(cudaFree(dWarm));
    printf("\nSCORE: %d/9\n", score);
    printf("OVERALL: %s\n", score==9 ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return score==9 ? 0 : 1;
}
