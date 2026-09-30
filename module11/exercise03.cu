// =====================================================================
// Module 11 / Exercise 3 : "Where the activation stops being free"
//
// GOAL
//   Everything in this module so far has assumed elementwise means memory
//   bound. That assumption has a boundary and you are going to locate it.
//
//   Part 1 is a real kernel: a SiLU-gated activation,
//
//       out[i] = ( x[i] * sigmoid(x[i]) ) * g[i],   sigmoid(v) = 1/(1+e^-v)
//
//   one `expf` and one divide per element. This is the activation in a
//   SwiGLU feed-forward block, and Parts XIV/XV will meet it again inside
//   a transformer. You predict whether it is memory bound, write a
//   fast-intrinsic version, and find out whether the intrinsics bought you
//   anything.
//
//   Part 2 finds the boundary directly: apply a function K times per
//   element and sweep K. You predict the crossover for `sinf` and for
//   `__sinf` BEFORE the sweep runs. Predictions are scored to within a
//   factor of two, because the quantity you are estimating is itself only
//   good to a factor of two.
//
// A NOTE ON -use_fast_math
//   `-use_fast_math` rewrites `expf` as `__expf`, `sinf` as `__sinf`, and
//   `a/b` as `__fdividef(a,b)`, among other things. This file measures the
//   accurate and intrinsic forms side by side in one binary, which is the
//   controlled version of the same experiment: you see the effect of the
//   flag without the flag also changing denormal handling, contraction and
//   everything else at the same time. TODO 5 asks you to predict the
//   result for the SiLU kernel before you see it.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN:   .\exercise03.exe
//
// Worth doing afterwards:
//   nvcc -arch=sm_89 -O3 -cubin -o exercise03.cubin exercise03.cu
//   cuobjdump -sass exercise03.cubin > sass.txt
//   then compare the bodies of the accurate and intrinsic kernels.
//
// WHAT IS CHECKED (7 points; all 7 required for OVERALL: PASS)
//   - TODO 1 exactly, TODO 2 exactly
//   - silu_fast numerics within the stated error budget, and not slower
//   - TODO 4's two crossover predictions, each within a factor of two
//   - TODO 5 exactly
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
static const double REL_BUDGET = 2e-3;         // for silu_fast

#define GRID_STRIDE(n) \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n); i += (long long)gridDim.x * blockDim.x

// ---------------------------------------------------------------------
// TODO 1: the compulsory traffic of the SiLU-gate kernel, in BYTES PER
//   ELEMENT. x and g are inputs, out is a separate output array.
// YOUR CODE HERE
static const int SILU_BYTES_PER_ELEMENT = -1;     // <- replace

// TODO 2: is the SiLU-gate kernel memory bound on this GPU?
//   Set to 1 for yes, 0 for no. Answer from an estimate, not a guess: you
//   know the traffic, you know the streaming rate, and Example 2 measured
//   how many FFMAs an element can afford before arithmetic shows up.
//   Ask yourself how many FP32 instructions one `expf` plus one divide is.
// YOUR CODE HERE
static const int SILU_IS_MEMORY_BOUND = -1;       // <- replace with 0 or 1

// ---------------------------------------------------------------------
// The accurate reference kernel. Do not modify.
__global__ void silu_gate(const float* __restrict__ x, const float* __restrict__ g,
                          float* __restrict__ out, long long n)
{
    for (GRID_STRIDE(n)) {
        const float v = x[i];
        out[i] = (v / (1.0f + expf(-v))) * g[i];
    }
}

// TODO 3: `silu_fast` -- the same function computed with the hardware
//   approximation path instead of the accurate library path.
//
//   Constraints:
//     - max relative error vs a double-precision host reference must be
//       <= 2e-3 (the harness measures it over the whole array);
//     - it must not be more than 5% slower than `silu_gate`.
//   There is more than one intrinsic involved; the divide is not free
//   either. Decide which approximations you are willing to accept, and be
//   ready to say what the error budget of a SwiGLU activation actually is
//   in a network whose weights are stored in 8 bits.
// YOUR CODE HERE
__global__ void silu_fast(const float* __restrict__ x, const float* __restrict__ g,
                          float* __restrict__ out, long long n)
{
    (void)x; (void)g; (void)out; (void)n;
}

// The streaming reference: one read, one write. Timed in the same loop.
__global__ void stream_ref(const float* __restrict__ a, float* __restrict__ o, long long n)
{ for (GRID_STRIDE(n)) o[i] = a[i]; }

// The K-sweep kernels. MODE 0 = FFMA (the pure-memory control),
// 1 = sinf, 2 = __sinf.
template <int K, int MODE>
__global__ void k_apply(const float* __restrict__ x, float* __restrict__ o, long long n)
{
    for (GRID_STRIDE(n)) {
        float v = x[i];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            if      (MODE == 0) v = v * 1.0000001f + 1e-7f;
            else if (MODE == 1) v = sinf(v);
            else                v = __sinf(v);
        }
        o[i] = v;
    }
}

// ---------------------------------------------------------------------
// TODO 4: the crossover predictions, made BEFORE you run the sweep.
//
//   The harness applies the function K times per element for
//   K in {1,2,4,8,16,32,64} and defines the CROSSOVER as the smallest K in
//   that set whose time exceeds 1.25x the memory floor (the time of the
//   FFMA control at K = 1). Predict that K for each function.
//
//   Both are scored to within a factor of two: a prediction of 8 is
//   accepted for a measured 4, 8 or 16. You are not being asked to nail a
//   number, you are being asked to know the order of magnitude and, more
//   importantly, to know that the two answers differ and in which
//   direction.
// YOUR CODE HERE
static const int PREDICT_K_SINF     = -1;     // accurate sinf
static const int PREDICT_K_FASTSINF = -1;     // __sinf intrinsic

// TODO 5: will -use_fast_math (i.e. replacing expf with __expf and the
//   divide with __fdividef) make the SiLU-gate kernel measurably faster?
//   1 for yes, 0 for no. Commit before you run.
// YOUR CODE HERE
static const int FAST_MATH_HELPS_SILU = -1;   // <- replace with 0 or 1

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

// FNV-1a: lets the harness check TODO 1 and TODO 2 without printing the
// answers in this file.
static unsigned fnv1a32(unsigned v)
{
    unsigned h = 2166136261u;
    for (int i = 0; i < 4; ++i) { h ^= (v >> (8*i)) & 0xffu; h *= 16777619u; }
    return h;
}
static const unsigned SILU_BPE_HASH  = 0x8c46f159u;
static const unsigned SILU_MB_HASH   = 0xfb69b604u;
static const unsigned FASTMATH_HASH  = 0x4b95f515u;

static int withinOctave(int pred, int truth)
{
    if (pred <= 0 || truth <= 0) return 0;
    return (pred * 2 >= truth) && (pred <= truth * 2);
}

int main(void)
{
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    const int nSM = prop.multiProcessorCount;
    const int TPB = 256, GRID = nSM * 8;

    printf("Module 11 / Exercise 3 -- where the activation stops being free\n");
    printf("Device: %s, %d SMs\n", prop.name, nSM);
    printf("N = %lld, one array = %.0f MB, L2 = %.0f MB\n\n",
           N, N*4.0/1048576.0, prop.l2CacheSize/1048576.0);

    if (SILU_BYTES_PER_ELEMENT < 0 || SILU_IS_MEMORY_BOUND < 0) {
        printf("Set TODO 1 and TODO 2 first.\n"); return 0;
    }
    if (PREDICT_K_SINF < 0 || PREDICT_K_FASTSINF < 0) {
        printf("Set TODO 4 first.\n"); return 0;
    }
    if (FAST_MATH_HELPS_SILU < 0) { printf("Set TODO 5 first.\n"); return 0; }

    const size_t bs = (size_t)N * sizeof(float);
    float *h_x = (float*)malloc(bs), *h_g = (float*)malloc(bs), *h_o = (float*)malloc(bs);
    if (!h_x || !h_g || !h_o) { printf("host allocation failed\n"); return 1; }
    for (long long i = 0; i < N; ++i) {
        h_x[i] = (float)(((i*1103515245LL + 12345LL) % 12001) - 6000) * 0.001f;  // [-6, 6]
        h_g[i] = (float)(((i*22695477LL   + 1LL)     % 2001)  - 1000) * 0.001f;  // [-1, 1]
    }

    float *x, *g, *o;
    CHECK(cudaMalloc(&x,bs)); CHECK(cudaMalloc(&g,bs)); CHECK(cudaMalloc(&o,bs));
    CHECK(cudaMemcpy(x,h_x,bs,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(g,h_g,bs,cudaMemcpyHostToDevice));

    cudaEvent_t e0,e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    { float acc=0.f;
      while (acc < 400.f) {
        CHECK(cudaEventRecord(e0));
        for (int i=0;i<20;++i) stream_ref<<<GRID,TPB>>>(x,o,N);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms=0.f; CHECK(cudaEventElapsedTime(&ms,e0,e1)); acc+=ms; } }
    CHECK(cudaGetLastError());

    // ---- one timing pass over everything, rotated, min of SWEEPS ----
    const int KV[7] = { 1, 2, 4, 8, 16, 32, 64 };
    double tRef = 1e30, tAcc = 1e30, tFast = 1e30;
    double tK[3][7];
    for (int m = 0; m < 3; ++m) for (int k = 0; k < 7; ++k) tK[m][k] = 1e30;

    const int NCFG = 3 + 3*7;
    for (int s = 0; s < SWEEPS; ++s) {
        for (int q = 0; q < NCFG; ++q) {
            const int c = (q + s) % NCFG;
            double ms = 0.0;
            if (c == 0)      { ms = timeCfg([&]{ stream_ref<<<GRID,TPB>>>(x,o,N); },e0,e1); if (ms<tRef)  tRef=ms;  }
            else if (c == 1) { ms = timeCfg([&]{ silu_gate <<<GRID,TPB>>>(x,g,o,N); },e0,e1); if (ms<tAcc)  tAcc=ms;  }
            else if (c == 2) { ms = timeCfg([&]{ silu_fast <<<GRID,TPB>>>(x,g,o,N); },e0,e1); if (ms<tFast) tFast=ms; }
            else {
                const int r = c - 3, mo = r / 7, ki = r % 7;
                #define DISPATCH(MO)                                                          \
                    switch (ki) {                                                             \
                    case 0: ms = timeCfg([&]{ k_apply<1 ,MO><<<GRID,TPB>>>(x,o,N); },e0,e1); break; \
                    case 1: ms = timeCfg([&]{ k_apply<2 ,MO><<<GRID,TPB>>>(x,o,N); },e0,e1); break; \
                    case 2: ms = timeCfg([&]{ k_apply<4 ,MO><<<GRID,TPB>>>(x,o,N); },e0,e1); break; \
                    case 3: ms = timeCfg([&]{ k_apply<8 ,MO><<<GRID,TPB>>>(x,o,N); },e0,e1); break; \
                    case 4: ms = timeCfg([&]{ k_apply<16,MO><<<GRID,TPB>>>(x,o,N); },e0,e1); break; \
                    case 5: ms = timeCfg([&]{ k_apply<32,MO><<<GRID,TPB>>>(x,o,N); },e0,e1); break; \
                    default:ms = timeCfg([&]{ k_apply<64,MO><<<GRID,TPB>>>(x,o,N); },e0,e1); break; \
                    }
                if      (mo == 0) { DISPATCH(0) }
                else if (mo == 1) { DISPATCH(1) }
                else              { DISPATCH(2) }
                #undef DISPATCH
                if (ms < tK[mo][ki]) tK[mo][ki] = ms;
            }
        }
    }
    CHECK(cudaGetLastError());

    // ---- crossovers, from the data ----
    const double floorMs = tK[0][0];              // FFMA control at K = 1
    int kSinf = 0, kFast = 0;
    for (int k = 0; k < 7; ++k) if (!kSinf && tK[1][k] > 1.25*floorMs) kSinf = KV[k];
    for (int k = 0; k < 7; ++k) if (!kFast && tK[2][k] > 1.25*floorMs) kFast = KV[k];
    if (!kSinf) kSinf = 128;                      // never crossed in range
    if (!kFast) kFast = 128;

    // ---- validation, untimed ----
    printf("=== validation (untimed second pass) ===\n");
    int pts = 0; const int maxpts = 7;

    const bool bpeOk = (fnv1a32((unsigned)SILU_BYTES_PER_ELEMENT) == SILU_BPE_HASH);
    printf("  [%s] TODO 1 SiLU bytes/element = %d\n", bpeOk?"ok":"  ", SILU_BYTES_PER_ELEMENT);
    if (bpeOk) ++pts;
    const bool mbOk = (fnv1a32((unsigned)SILU_IS_MEMORY_BOUND) == SILU_MB_HASH);
    printf("  [%s] TODO 2 memory bound = %d\n", mbOk?"ok":"  ", SILU_IS_MEMORY_BOUND);
    if (mbOk) ++pts;

    CHECK(cudaMemset(o,0,bs));
    silu_gate<<<GRID,TPB>>>(x,g,o,N);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_o,o,bs,cudaMemcpyDeviceToHost));
    double worstAcc = 0.0;
    for (long long i = 0; i < N; ++i) {
        const double ref = ((double)h_x[i] / (1.0 + exp(-(double)h_x[i]))) * (double)h_g[i];
        if (fabs(ref) > 1e-3) { double e = fabs((double)h_o[i]-ref)/fabs(ref); if (e>worstAcc) worstAcc=e; }
    }
    printf("  [%s] silu_gate  max relative error = %.3e (sanity, budget 1e-5)\n",
           (worstAcc<1e-5)?"ok":"  ", worstAcc);

    CHECK(cudaMemset(o,0,bs));
    silu_fast<<<GRID,TPB>>>(x,g,o,N);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_o,o,bs,cudaMemcpyDeviceToHost));
    double worstFast = 0.0; long long zeros = 0;
    for (long long i = 0; i < N; ++i) {
        if (h_o[i] == 0.0f) ++zeros;
        const double ref = ((double)h_x[i] / (1.0 + exp(-(double)h_x[i]))) * (double)h_g[i];
        if (fabs(ref) > 1e-3) { double e = fabs((double)h_o[i]-ref)/fabs(ref); if (e>worstFast) worstFast=e; }
    }
    const bool fastNumOk = (worstFast <= REL_BUDGET) && (zeros < N/2);
    printf("  [%s] silu_fast  max relative error = %.3e (budget %.0e)\n",
           fastNumOk?"ok":"  ", worstFast, REL_BUDGET);
    if (fastNumOk) ++pts;

    const bool notSlower = (tFast <= 1.05*tAcc);
    printf("  [%s] silu_fast is not more than 5%% slower (%.4f vs %.4f ms)\n",
           notSlower?"ok":"  ", tFast, tAcc);
    if (notSlower) ++pts;

    // ---- Part 1 numbers ----
    const double siluBytes = (double)SILU_BYTES_PER_ELEMENT * (double)N;
    const double ceilGBs   = (8.0*(double)N)/(tRef*1e-3)/1e9;
    printf("\n=== Part 1: the SiLU gate ===\n");
    printf("  measured streaming ceiling        : %.1f GB/s (%.0f%% of %.0f)\n",
           ceilGBs, 100.0*ceilGBs/PEAK_GBS, PEAK_GBS);
    printf("  your traffic floor at that rate   : %.4f ms\n", siluBytes/(ceilGBs*1e9)*1e3);
    printf("  %-26s %10s %10s %10s\n", "version", "ms", "GB/s", "x floor");
    printf("  %-26s %10.4f %10.1f %10.3f\n", "silu_gate (expf, /)", tAcc,
           siluBytes/(tAcc*1e-3)/1e9, tAcc/(siluBytes/(ceilGBs*1e9)*1e3));
    printf("  %-26s %10.4f %10.1f %10.3f\n", "silu_fast (yours)", tFast,
           siluBytes/(tFast*1e-3)/1e9, tFast/(siluBytes/(ceilGBs*1e9)*1e3));
    printf("  ratio accurate/fast : %.3fx\n", tAcc/tFast);

    const int fastHelps = (tAcc/tFast >= 1.05) ? 1 : 0;
    const bool fmOk = (fnv1a32((unsigned)FAST_MATH_HELPS_SILU) == FASTMATH_HASH)
                      && (FAST_MATH_HELPS_SILU == fastHelps);
    printf("  [%s] TODO 5 fast-math helps = %d, measured %d\n",
           fmOk?"ok":"  ", FAST_MATH_HELPS_SILU, fastHelps);
    if (fmOk) ++pts;

    // ---- Part 2 sweep ----
    printf("\n=== Part 2: applying a function K times per element (2N traffic) ===\n");
    printf("  memory floor (FFMA control, K=1)  : %.4f ms\n", floorMs);
    printf("  crossover := smallest K with t(K) > 1.25 x floor\n\n");
    const char* nm[3] = { "FFMA", "sinf", "__sinf" };
    printf("  %-8s", "K =");
    for (int k = 0; k < 7; ++k) printf("%9d", KV[k]);
    printf("\n");
    for (int m = 0; m < 3; ++m) {
        printf("  %-8s", nm[m]);
        for (int k = 0; k < 7; ++k) printf("%9.4f", tK[m][k]);
        printf("\n");
    }
    printf("\n  %-8s", "x floor");
    printf("\n");
    for (int m = 0; m < 3; ++m) {
        printf("  %-8s", nm[m]);
        for (int k = 0; k < 7; ++k) printf("%9.2f", tK[m][k]/floorMs);
        printf("\n");
    }
    printf("\n  measured crossover: sinf K = %d, __sinf K = %d%s\n",
           kSinf, kFast, (kSinf>=128||kFast>=128) ? "   (128 = never, in this range)" : "");

    const bool p1 = withinOctave(PREDICT_K_SINF, kSinf) != 0;
    const bool p2 = withinOctave(PREDICT_K_FASTSINF, kFast) != 0;
    printf("  [%s] TODO 4a sinf   predicted %d, measured %d (factor of 2 allowed)\n",
           p1?"ok":"  ", PREDICT_K_SINF, kSinf);
    printf("  [%s] TODO 4b __sinf predicted %d, measured %d (factor of 2 allowed)\n",
           p2?"ok":"  ", PREDICT_K_FASTSINF, kFast);
    if (p1) ++pts;
    if (p2) ++pts;

    printf("\nScore: %d/%d\n", pts, maxpts);
    printf("OVERALL: %s\n", (pts==maxpts)?"PASS":"FAIL");

    free(h_x); free(h_g); free(h_o);
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(x)); CHECK(cudaFree(g)); CHECK(cudaFree(o));
    CHECK(cudaDeviceReset());
    return (pts==maxpts)?0:1;
}
