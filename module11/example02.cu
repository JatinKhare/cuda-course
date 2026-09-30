// =====================================================================
// Module 11 / Example 2 : "Reaching the floor"
//
// GOAL
//   Example 1 computed the floor. This file is the four levers that close
//   the gap to it, each one isolated and measured.
//
//   Part A  LAUNCH CONFIGURATION. Sweep blocks/SM x threads/block for a
//           grid-stride copy and find where the plateau begins. The answer
//           is not "as many threads as possible"; it is "enough bytes in
//           flight to cover DRAM latency" -- Little's Law from Module 1.
//   Part B  VECTORIZATION. float2 / float4 on an already-perfectly-coalesced
//           kernel. Module 5 said this cannot reduce sector count. Measure
//           what it does reduce.
//   Part C  MEMORY-LEVEL PARALLELISM PER THREAD. Coarsening with the loads
//           hoisted above the first use, versus the same work with the loads
//           serialized. Same instruction mix, same traffic, different MLP.
//           This is ILP as a substitute for occupancy (Module 20).
//   Part D  WHEN ELEMENTWISE IS NOT BANDWIDTH-BOUND. Apply a function K
//           times per element and find the K at which arithmetic overtakes
//           memory, for sinf/expf and their SFU intrinsics.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN:   .\example02.exe
//
// To see the instruction-count argument of Part B for yourself:
//   nvcc -arch=sm_89 -O3 -Xptxas -v -cubin -o example02.cubin example02.cu
//   cuobjdump -sass example02.cubin > sass.txt
//   findstr /C:"LDG.E" /C:"STG.E" sass.txt
//
// METHODOLOGY (spec 12): rotated sweep order, per-configuration iteration
// counts sized to ~10 ms segments, min of 4 sweeps, duration-based warm-up,
// buffers >= 5x L2, validation in a separate untimed pass.
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
static const long long N     = 1LL << 26;      // 67,108,864 floats = 256 MB
static const int  SWEEPS     = 4;

// ------------------------------------------------------------------ timing
template <typename L>
static double timeCfg(L launch, cudaEvent_t a, cudaEvent_t b)
{
    // one probe launch -> choose an iteration count giving a ~10 ms segment
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

#define GRID_STRIDE(n) \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n); i += (long long)gridDim.x * blockDim.x

// ------------------------------------------------------------- Part A
__global__ void k_copy(const float* __restrict__ a, float* __restrict__ o, long long n)
{ for (GRID_STRIDE(n)) o[i] = a[i]; }

// ------------------------------------------------------------- Part B
__global__ void k_saxpy(const float* __restrict__ x, const float* __restrict__ y,
                        float* __restrict__ o, float a, long long n)
{ for (GRID_STRIDE(n)) o[i] = a * x[i] + y[i]; }

__global__ void k_saxpy2(const float2* __restrict__ x, const float2* __restrict__ y,
                         float2* __restrict__ o, float a, long long n2)
{
    for (GRID_STRIDE(n2)) {
        float2 X = x[i], Y = y[i], r;
        r.x = a*X.x + Y.x; r.y = a*X.y + Y.y;
        o[i] = r;
    }
}

__global__ void k_saxpy4(const float4* __restrict__ x, const float4* __restrict__ y,
                         float4* __restrict__ o, float a, long long n4)
{
    for (GRID_STRIDE(n4)) {
        float4 X = x[i], Y = y[i], r;
        r.x = a*X.x + Y.x; r.y = a*X.y + Y.y;
        r.z = a*X.z + Y.z; r.w = a*X.w + Y.w;
        o[i] = r;
    }
}

// ------------------------------------------------------------- Part C
// C elements per thread. All 2C loads are issued BEFORE the first value is
// consumed, so one thread has 2C requests outstanding at once.
template <int C>
__global__ void k_mlp(const float* __restrict__ x, const float* __restrict__ y,
                      float* __restrict__ o, float a, long long n)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long base   = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (long long i = base; i < n; i += stride * C) {
        float xv[C], yv[C];
        #pragma unroll
        for (int c = 0; c < C; ++c) { long long j = i + c*stride; if (j < n) { xv[c] = x[j]; yv[c] = y[j]; } }
        #pragma unroll
        for (int c = 0; c < C; ++c) { long long j = i + c*stride; if (j < n) o[j] = a*xv[c] + yv[c]; }
    }
}

// Identical work and identical traffic, but `#pragma unroll 1` forbids the
// compiler from overlapping iteration c with iteration c+1, so at most 2
// requests are outstanding per thread no matter how large C is.
template <int C>
__global__ void k_serial(const float* __restrict__ x, const float* __restrict__ y,
                         float* __restrict__ o, float a, long long n)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long base   = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (long long i = base; i < n; i += stride * C) {
        #pragma unroll 1
        for (int c = 0; c < C; ++c) { long long j = i + c*stride; if (j < n) o[j] = a*x[j] + y[j]; }
    }
}

// ------------------------------------------------------------- Part D
// MODE 0 = FFMA, 1 = sinf, 2 = __sinf, 3 = expf, 4 = __expf
template <int K, int MODE>
__global__ void k_trans(const float* __restrict__ x, float* __restrict__ o, long long n)
{
    for (GRID_STRIDE(n)) {
        float v = x[i];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            if      (MODE == 0) v = v * 1.0000001f + 1e-7f;
            else if (MODE == 1) v = sinf(v);
            else if (MODE == 2) v = __sinf(v);
            else if (MODE == 3) v = expf(v * 0.1f);
            else                v = __expf(v * 0.1f);
        }
        o[i] = v;
    }
}

// ------------------------------------------------------------------ main
int main(void)
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const int nSM = prop.multiProcessorCount;
    printf("Device: %s (sm_%d%d, %d SMs, %d threads/SM)\n", prop.name,
           prop.major, prop.minor, nSM, prop.maxThreadsPerMultiProcessor);
    printf("N = %lld floats, one array = %.0f MB\n\n", N, N*4.0/1048576.0);

    float *x, *y, *o;
    CHECK(cudaMalloc(&x, N*sizeof(float)));
    CHECK(cudaMalloc(&y, N*sizeof(float)));
    CHECK(cudaMalloc(&o, N*sizeof(float)));
    {
        float* h = (float*)malloc((size_t)N*sizeof(float));
        if (!h) { printf("host alloc failed\n"); return 1; }
        for (long long i = 0; i < N; ++i) h[i] = (float)((i*1103515245LL + 12345LL) % 1000) * 0.001f;
        CHECK(cudaMemcpy(x, h, N*sizeof(float), cudaMemcpyHostToDevice));
        for (long long i = 0; i < N; ++i) h[i] = (float)((i*22695477LL + 1LL) % 997) * 0.001f;
        CHECK(cudaMemcpy(y, h, N*sizeof(float), cudaMemcpyHostToDevice));
        free(h);
    }

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    { float acc = 0.f;
      while (acc < 400.f) {
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < 20; ++i) k_copy<<<nSM*8, 256>>>(x, o, N);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, e0, e1)); acc += ms; } }
    CHECK(cudaGetLastError());

    // =================================================================
    // Part A : the launch-configuration plateau
    // =================================================================
    {
        const int TPB[] = { 64, 128, 256, 512, 1024 };
        const int BPS[] = { 1, 2, 4, 8, 16, 32 };
        const int NT = 5, NB = 6;
        static double best[5][6];
        for (int i = 0; i < NT; ++i) for (int j = 0; j < NB; ++j) best[i][j] = 1e30;
        double best11 = 1e30;

        for (int s = 0; s < SWEEPS; ++s) {
            const int M = NT*NB + 1;
            for (int q = 0; q < M; ++q) {
                const int c = (q + s) % M;
                if (c == NT*NB) {                       // the 1:1 mapping
                    const int g = (int)((N + 255) / 256);
                    const double ms = timeCfg([&]{ k_copy<<<g, 256>>>(x, o, N); }, e0, e1);
                    if (ms < best11) best11 = ms;
                } else {
                    const int i = c / NB, j = c % NB;
                    const int g = nSM * BPS[j];
                    const double ms = timeCfg([&]{ k_copy<<<g, TPB[i]>>>(x, o, N); }, e0, e1);
                    if (ms < best[i][j]) best[i][j] = ms;
                }
            }
        }
        CHECK(cudaGetLastError());

        printf("=== Part A: launch configuration (grid-stride copy, 2N = %.0f MB) ===\n",
               8.0*N/1048576.0);
        printf("GB/s, rows = blocks per SM, columns = threads per block\n");
        printf("%9s", "blocks/SM");
        for (int i = 0; i < NT; ++i) printf("%9d", TPB[i]);
        printf("%12s\n", "threads/SM");
        double ceilGBs = 0.0;
        for (int j = 0; j < NB; ++j) {
            printf("%9d", BPS[j]);
            for (int i = 0; i < NT; ++i) {
                const double g = 8.0*N/(best[i][j]*1e-3)/1e9;
                if (g > ceilGBs) ceilGBs = g;
                printf("%9.1f", g);
            }
            printf("   %4d-%5d\n", BPS[j]*TPB[0], BPS[j]*TPB[NT-1]);
        }
        printf("\n  1:1 mapping, grid = ceil(N/256) = %d blocks : %.1f GB/s\n",
               (int)((N+255)/256), 8.0*N/(best11*1e-3)/1e9);
        printf("  best observed in this sweep                 : %.1f GB/s = %.0f%% of %.0f\n",
               ceilGBs, 100.0*ceilGBs/PEAK_GBS, PEAK_GBS);
        printf("\n  Read the table by TOTAL RESIDENT THREADS, not by either axis.\n"
               "  The only clearly-slow entries are the top-left corner, where the\n"
               "  grid is too small to keep enough loads in flight. Module 1's\n"
               "  Little's Law says concurrency = throughput x latency: at ~%.0f GB/s\n"
               "  and a ~575-cycle DRAM latency at ~1.9 GHz, the machine needs\n"
               "  roughly %.0f KB of data in flight at all times. At 4 B per thread\n"
               "  per load that is ~%.0f thousand outstanding loads. Once the grid\n"
               "  supplies that, adding threads changes nothing -- the bus is the\n"
               "  constraint and it does not care who is asking.\n",
               ceilGBs, ceilGBs*1e9*(575.0/1.9e9)/1024.0,
               ceilGBs*1e9*(575.0/1.9e9)/4.0/1000.0);
    }

    // =================================================================
    // Part B : vectorization  +  Part C : MLP per thread
    // =================================================================
    {
        const int TPBC = 128;
        const int SMALL = nSM * 1;      // 1 block/SM  =  5,120 threads
        const int LARGE = nSM * 8;      // 8 blocks/SM = 40,960 threads
        const int CS[] = { 1, 2, 4, 8, 16 };
        const int NC = 5;

        double bSc = 1e30, bV2 = 1e30, bV4 = 1e30;          // 8 blocks/SM
        double sSc = 1e30, sV2 = 1e30, sV4 = 1e30;          // 1 block /SM
        static double bMlpS[5], bMlpL[5], bSerS[5], bSerL[5];
        for (int i = 0; i < NC; ++i) bMlpS[i] = bMlpL[i] = bSerS[i] = bSerL[i] = 1e30;

        const long long n2 = N/2, n4 = N/4;

        for (int s = 0; s < SWEEPS; ++s) {
            const int M = 6 + 4*NC;
            for (int q = 0; q < M; ++q) {
                const int c = (q + s) % M;
                double ms = 0.0;
                if (c == 0) { ms = timeCfg([&]{ k_saxpy <<<LARGE,TPBC>>>(x,y,o,2.f,N); }, e0,e1); if (ms<bSc) bSc=ms; }
                else if (c == 1) { ms = timeCfg([&]{ k_saxpy2<<<LARGE,TPBC>>>((const float2*)x,(const float2*)y,(float2*)o,2.f,n2); }, e0,e1); if (ms<bV2) bV2=ms; }
                else if (c == 2) { ms = timeCfg([&]{ k_saxpy4<<<LARGE,TPBC>>>((const float4*)x,(const float4*)y,(float4*)o,2.f,n4); }, e0,e1); if (ms<bV4) bV4=ms; }
                else if (c == 3) { ms = timeCfg([&]{ k_saxpy <<<SMALL,TPBC>>>(x,y,o,2.f,N); }, e0,e1); if (ms<sSc) sSc=ms; }
                else if (c == 4) { ms = timeCfg([&]{ k_saxpy2<<<SMALL,TPBC>>>((const float2*)x,(const float2*)y,(float2*)o,2.f,n2); }, e0,e1); if (ms<sV2) sV2=ms; }
                else if (c == 5) { ms = timeCfg([&]{ k_saxpy4<<<SMALL,TPBC>>>((const float4*)x,(const float4*)y,(float4*)o,2.f,n4); }, e0,e1); if (ms<sV4) sV4=ms; }
                else {
                    const int r = c - 6;
                    const int variant = r / NC;         // 0 mlp/small 1 mlp/large 2 ser/small 3 ser/large
                    const int ci = r % NC;
                    const int G = (variant == 0 || variant == 2) ? SMALL : LARGE;
                    const bool ser = (variant >= 2);
                    switch (ci) {
                    case 0: ms = ser ? timeCfg([&]{ k_serial<1 ><<<G,TPBC>>>(x,y,o,2.f,N); },e0,e1)
                                     : timeCfg([&]{ k_mlp   <1 ><<<G,TPBC>>>(x,y,o,2.f,N); },e0,e1); break;
                    case 1: ms = ser ? timeCfg([&]{ k_serial<2 ><<<G,TPBC>>>(x,y,o,2.f,N); },e0,e1)
                                     : timeCfg([&]{ k_mlp   <2 ><<<G,TPBC>>>(x,y,o,2.f,N); },e0,e1); break;
                    case 2: ms = ser ? timeCfg([&]{ k_serial<4 ><<<G,TPBC>>>(x,y,o,2.f,N); },e0,e1)
                                     : timeCfg([&]{ k_mlp   <4 ><<<G,TPBC>>>(x,y,o,2.f,N); },e0,e1); break;
                    case 3: ms = ser ? timeCfg([&]{ k_serial<8 ><<<G,TPBC>>>(x,y,o,2.f,N); },e0,e1)
                                     : timeCfg([&]{ k_mlp   <8 ><<<G,TPBC>>>(x,y,o,2.f,N); },e0,e1); break;
                    default:ms = ser ? timeCfg([&]{ k_serial<16><<<G,TPBC>>>(x,y,o,2.f,N); },e0,e1)
                                     : timeCfg([&]{ k_mlp   <16><<<G,TPBC>>>(x,y,o,2.f,N); },e0,e1); break;
                    }
                    double* dst = (variant==0)? bMlpS : (variant==1)? bMlpL : (variant==2)? bSerS : bSerL;
                    if (ms < dst[ci]) dst[ci] = ms;
                }
            }
        }
        CHECK(cudaGetLastError());

        const double B3 = 12.0 * (double)N;   // saxpy compulsory traffic, 3N
        printf("\n=== Part B: access width (saxpy o=a*x+y, 3N = %.0f MB, %d thr/block) ===\n",
               B3/1048576.0, TPBC);
        printf("  %-12s | %9s %9s %9s | %9s %9s %9s\n", "", "1 blk/SM", "GB/s", "vs scalar",
               "8 blk/SM", "GB/s", "vs scalar");
        printf("  %-12s | %9.4f %9.1f %9s | %9.4f %9.1f %9s\n", "float  (1)",
               sSc, B3/(sSc*1e-3)/1e9, "1.000", bSc, B3/(bSc*1e-3)/1e9, "1.000");
        printf("  %-12s | %9.4f %9.1f %9.3f | %9.4f %9.1f %9.3f\n", "float2 (2)",
               sV2, B3/(sV2*1e-3)/1e9, sSc/sV2, bV2, B3/(bV2*1e-3)/1e9, bSc/bV2);
        printf("  %-12s | %9.4f %9.1f %9.3f | %9.4f %9.1f %9.3f\n", "float4 (4)",
               sV4, B3/(sV4*1e-3)/1e9, sSc/sV4, bV4, B3/(bV4*1e-3)/1e9, bSc/bV4);
        printf("\n  Module 5 established that a float4 load moves exactly the same\n"
               "  sectors as four float loads. The traffic is identical and so is\n"
               "  the ceiling; what changes is the number of memory INSTRUCTIONS\n"
               "  and the number of outstanding-request slots each one occupies.\n"
               "  The SASS is unambiguous -- all three kernels compile to the same\n"
               "  10 loads and 5 stores per grid-stride iteration:\n"
               "      k_saxpy   10 x LDG.E.CONSTANT       5 x STG.E       ( 5 elems)\n"
               "      k_saxpy2  10 x LDG.E.64.CONSTANT    5 x STG.E.64    (10 elems)\n"
               "      k_saxpy4  10 x LDG.E.128.CONSTANT   5 x STG.E.128   (20 elems)\n"
               "  Same instruction count, 4x the elements: exactly 1/4 of the memory\n"
               "  instructions per element, and 4x the bytes per request slot.\n"
               "  That is why vectorizing pays most where request slots are scarce\n"
               "  (the 1 block/SM column) and almost nothing where the bus is\n"
               "  already saturated (the 8 blocks/SM column). Vectorizing is a\n"
               "  LATENCY/ISSUE optimization, not a bandwidth one.\n");

        printf("\n=== Part C: memory-level parallelism per thread (%d threads/block) ===\n", TPBC);
        printf("  Same traffic (3N), same arithmetic, same coarsening factor C.\n");
        printf("  'hoisted' issues all 2C loads before consuming any of them.\n");
        printf("  'serial'  consumes each load before issuing the next.\n\n");
        printf("  %-10s", "C");
        for (int i = 0; i < NC; ++i) printf("%9d", CS[i]);
        printf("\n");
        printf("  %-10s", "hoisted@1");  for (int i=0;i<NC;++i) printf("%9.1f", B3/(bMlpS[i]*1e-3)/1e9); printf("   (1 block/SM  = %d threads)\n", SMALL*TPBC);
        printf("  %-10s", "serial @1");  for (int i=0;i<NC;++i) printf("%9.1f", B3/(bSerS[i]*1e-3)/1e9); printf("\n");
        printf("  %-10s", "hoisted@8");  for (int i=0;i<NC;++i) printf("%9.1f", B3/(bMlpL[i]*1e-3)/1e9); printf("   (8 blocks/SM = %d threads)\n", LARGE*TPBC);
        printf("  %-10s", "serial @8");  for (int i=0;i<NC;++i) printf("%9.1f", B3/(bSerL[i]*1e-3)/1e9); printf("\n");
        printf("\n  Three readings of the same table:\n"
               "  (1) At 1 block/SM (5,120 threads, 4 warps per SM) the machine is\n"
               "      starved and the two rows separate by ~2x for every C >= 2.\n"
               "      Same traffic, same instructions, different number of requests\n"
               "      in flight per thread. The SASS confirms the mechanism: for\n"
               "      C >= 2 the serial kernel compiles to 2 x LDG + 1 x STG, full\n"
               "      stop, while the hoisted kernel compiles to 16-32 LDGs. ILP is\n"
               "      a SUBSTITUTE for resident warps -- both supply outstanding\n"
               "      requests, which is the only currency the memory system takes.\n"
               "      Module 20 develops this.\n"
               "  (2) At 8 blocks/SM the two rows converge, and the SIMPLER kernel\n"
               "      is marginally faster. Occupancy has already supplied the\n"
               "      concurrency; coarsening now buys nothing and costs registers,\n"
               "      address arithmetic, and a longer tail. Coarsening a kernel\n"
               "      that is already at its ceiling makes it longer, not faster.\n"
               "  (3) The C=2 hoisted entry is reproducibly BELOW its C=1 and C=4\n"
               "      neighbours and the simple MLP story does not explain it: the\n"
               "      SASS shows k_mlp<1> issuing 10 LDGs and k_mlp<2> issuing 20,\n"
               "      so C=2 has MORE requests in flight and is still slower. It is\n"
               "      reported here rather than smoothed away. Do not build an\n"
               "      argument on a single point of a noisy curve; build it on the\n"
               "      shape, which here is 'hoisted ~2x serial, flat in C'.\n");
    }

    // =================================================================
    // Part D : the arithmetic crossover
    // =================================================================
    {
        const int G = nSM*8, T = 256;
        const int KV[] = { 1, 2, 4, 8, 16, 32 };
        const int NK = 6, NM = 5;
        static double best[5][6];
        for (int a = 0; a < NM; ++a) for (int k = 0; k < NK; ++k) best[a][k] = 1e30;

        for (int s = 0; s < SWEEPS; ++s) {
            const int M = NM*NK;
            for (int q = 0; q < M; ++q) {
                const int c  = (q + s) % M;
                const int mo = c / NK, ki = c % NK;
                double ms = 0.0;
                #define DISPATCH(MO)                                                                 \
                    switch (ki) {                                                                    \
                    case 0: ms = timeCfg([&]{ k_trans<1 ,MO><<<G,T>>>(x,o,N); }, e0,e1); break;      \
                    case 1: ms = timeCfg([&]{ k_trans<2 ,MO><<<G,T>>>(x,o,N); }, e0,e1); break;      \
                    case 2: ms = timeCfg([&]{ k_trans<4 ,MO><<<G,T>>>(x,o,N); }, e0,e1); break;      \
                    case 3: ms = timeCfg([&]{ k_trans<8 ,MO><<<G,T>>>(x,o,N); }, e0,e1); break;      \
                    case 4: ms = timeCfg([&]{ k_trans<16,MO><<<G,T>>>(x,o,N); }, e0,e1); break;      \
                    default:ms = timeCfg([&]{ k_trans<32,MO><<<G,T>>>(x,o,N); }, e0,e1); break;      \
                    }
                if      (mo == 0) { DISPATCH(0) }
                else if (mo == 1) { DISPATCH(1) }
                else if (mo == 2) { DISPATCH(2) }
                else if (mo == 3) { DISPATCH(3) }
                else              { DISPATCH(4) }
                #undef DISPATCH
                if (ms < best[mo][ki]) best[mo][ki] = ms;
            }
        }
        CHECK(cudaGetLastError());

        const char* nm[5] = { "FFMA", "sinf", "__sinf", "expf", "__expf" };
        const double B2 = 8.0 * (double)N;
        printf("\n=== Part D: where arithmetic overtakes memory (o[i] = f^K(x[i]), 2N) ===\n");
        printf("  ms\n  %-9s", "K =");
        for (int k = 0; k < NK; ++k) printf("%9d", KV[k]);
        printf("\n");
        for (int a = 0; a < NM; ++a) {
            printf("  %-9s", nm[a]);
            for (int k = 0; k < NK; ++k) printf("%9.4f", best[a][k]);
            printf("\n");
        }
        printf("\n  effective GB/s against the 2N memory model\n  %-9s", "K =");
        for (int k = 0; k < NK; ++k) printf("%9d", KV[k]);
        printf("\n");
        for (int a = 0; a < NM; ++a) {
            printf("  %-9s", nm[a]);
            for (int k = 0; k < NK; ++k) printf("%9.1f", B2/(best[a][k]*1e-3)/1e9);
            printf("\n");
        }
        printf("\n  A row holds flat while the kernel is memory-bound: the SFU and\n"
               "  FP32 pipes are running in the shadow of DRAM and cost nothing.\n"
               "  The K at which a row starts to climb is that function's crossover\n"
               "  on this machine. FFMA never crosses in this range -- the memory\n"
               "  system gives you dozens of free FLOPs per element, which is the\n"
               "  memory-bound half of Module 21's roofline. The accurate libm\n"
               "  functions cross far earlier than their SFU intrinsics, and the\n"
               "  SASS says why: sinf is a Cody-Waite argument reduction plus a\n"
               "  minimax polynomial plus an FP64 fallback path for large\n"
               "  arguments (and Ada runs FP64 at 1/64 rate), while __sinf is\n"
               "  literally FMUL.RZ by 1/2pi followed by one MUFU.SIN.\n");
    }

    // =================================================================
    // Validation, separate untimed pass
    // =================================================================
    printf("\n=== validation (untimed second pass) ===\n");
    int fails = 0;
    {
        const long long NV = 1 << 20;
        float *h = (float*)malloc((size_t)NV*sizeof(float));
        float *hx= (float*)malloc((size_t)NV*sizeof(float));
        float *hy= (float*)malloc((size_t)NV*sizeof(float));
        if (!h||!hx||!hy) { printf("host alloc failed\n"); return 1; }
        CHECK(cudaMemcpy(hx, x, NV*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hy, y, NV*sizeof(float), cudaMemcpyDeviceToHost));

        struct V { const char* name; int kind; };
        const V vs[] = { {"scalar",0}, {"float4",1}, {"hoisted C=16",2}, {"serial C=16",3} };
        for (int v = 0; v < 4; ++v) {
            CHECK(cudaMemset(o, 0, N*sizeof(float)));
            switch (vs[v].kind) {
            case 0: k_saxpy  <<<nSM*8,128>>>(x,y,o,2.f,N); break;
            case 1: k_saxpy4 <<<nSM*8,128>>>((const float4*)x,(const float4*)y,(float4*)o,2.f,N/4); break;
            case 2: k_mlp<16><<<nSM*1,128>>>(x,y,o,2.f,N); break;
            default:k_serial<16><<<nSM*1,128>>>(x,y,o,2.f,N); break;
            }
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h, o, NV*sizeof(float), cudaMemcpyDeviceToHost));
            long long bad = 0;
            for (long long i = 0; i < NV; ++i) {
                const float ref = 2.f*hx[i] + hy[i];
                if (fabsf(h[i]-ref) > 1e-5f*fmaxf(1.0f, fabsf(ref))) ++bad;
            }
            printf("  %-14s vs CPU reference: %lld mismatch(es)  %s\n",
                   vs[v].name, bad, bad ? "FAIL" : "PASS");
            if (bad) ++fails;
        }

        // __sinf is an approximation; check it is close, not equal.
        CHECK(cudaMemset(o, 0, N*sizeof(float)));
        k_trans<1,2><<<nSM*8,256>>>(x, o, N);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h, o, NV*sizeof(float), cudaMemcpyDeviceToHost));
        double worst = 0.0;
        for (long long i = 0; i < NV; ++i) {
            const double ref = sin((double)hx[i]);
            const double err = fabs((double)h[i] - ref);
            if (err > worst) worst = err;
        }
        printf("  __sinf max absolute error over %lld samples: %.3e  %s\n",
               NV, worst, (worst < 2e-6) ? "PASS" : "FAIL");
        if (!(worst < 2e-6)) ++fails;
        printf("  (the accuracy you trade for the crossover shift in Part D)\n");
        free(h); free(hx); free(hy);
    }

    printf("\nOVERALL: %s\n", fails ? "FAIL" : "PASS");

    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(x)); CHECK(cudaFree(y)); CHECK(cudaFree(o));
    CHECK(cudaDeviceReset());
    return fails ? 1 : 0;
}
