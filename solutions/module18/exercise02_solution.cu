// =============================================================================
// Module 18 / Exercise 2 — the register/occupancy trade, measured.
//
// GOAL : A register-tiled GEMM buys arithmetic intensity with registers. The
//        register file is 65536 32-bit registers per SM and it is shared by
//        every resident thread, so a larger thread tile means fewer resident
//        warps. Somewhere there is an optimum, and this exercise is about
//        finding it from FIRST PRINCIPLES rather than by trying everything.
//
//        You will: compute occupancy by hand and check it against the CUDA
//        API; state the two-level reuse law; predict the winner BEFORE any
//        timing happens; build a cost model that chooses a tile shape from the
//        resource table alone; and predict where the compiler starts spilling.
//
//        The sweep here uses 128 threads per block, so the arithmetic is not
//        the one worked out in Example 2 and you cannot copy the answers.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
// RUN  : exercise02_solution.exe
// Also: nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise02.exe exercise02.cu
//
// WHAT IS SCORED (10 points; OVERALL: PASS needs all ten)
//   3  your occupancy function agrees with
//      cudaOccupancyMaxActiveBlocksPerMultiprocessor on all 11 kernels
//   1  the two-level reuse law (hashed against a reference)
//   2  two predictions committed before the timing runs
//   2  your cost model picks a configuration within 15% of the measured best
//   2  you correctly name the first launch bound at which ptxas spills
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

#define M_DIM 1027
#define N_DIM 2053
#define K_DIM  769
#define NTHREADS 128           // 8 x 16 thread grid, for every configuration
#define BKD 8

// =============================================================================
// TODO 1 — occupancy, by hand.
//
// Return the number of blocks of `threads` threads that can be resident on one
// sm_89 SM, given `regsPerThread` registers per thread and `smemBytes` bytes of
// static shared memory per block. Set *limiter to one of the four strings
// "registers", "shared", "threads/SM", "blocks/SM" naming the binding resource
// (ties: report the first in that order).
//
// Facts you have been given, in Modules 1, 4, 6 and 7:
//   65536 32-bit registers per SM        1536 threads per SM
//   102400 B shared memory per SM        24 blocks per SM maximum
//   1024 B per-block shared-memory driver reserve
//   shared memory is allocated in 128 B granules
//
// One fact you have NOT been given and must get right anyway: registers are not
// allocated per thread. Find the granularity for yourself — the harness runs
// your function against the CUDA occupancy API on eleven real kernels at two
// different block sizes, and a per-thread model matches on some of them and
// not on others. If you get 9/11, you have the wrong granule.
// =============================================================================
static int occupancyBlocks(int regsPerThread, int smemBytes, int threads,
                           const char **limiter)
{
    const int REGS_PER_SM = 65536, THREADS_PER_SM = 1536;
    const int SMEM_PER_SM = 102400, MAX_BLOCKS = 24;
    const int SMEM_RESERVE = 1024, SMEM_GRAN = 128;
    const int REG_GRAN = 8;            // registers are allocated per WARP, in
                                       // granules of 8 per thread (256/warp)
    const int warps = (threads + 31) / 32;
    const int regsPerBlock = ((regsPerThread + REG_GRAN - 1)/REG_GRAN) * REG_GRAN
                           * 32 * warps;
    const int smemPerBlock = ((smemBytes + SMEM_RESERVE + SMEM_GRAN - 1)/SMEM_GRAN)
                           * SMEM_GRAN;
    const int byReg  = regsPerBlock ? REGS_PER_SM / regsPerBlock : MAX_BLOCKS;
    const int bySmem = smemPerBlock ? SMEM_PER_SM / smemPerBlock : MAX_BLOCKS;
    const int byThr  = THREADS_PER_SM / threads;
    int best = byReg;              const char *l = "registers";
    if (bySmem    < best) { best = bySmem;    l = "shared"; }
    if (byThr     < best) { best = byThr;     l = "threads/SM"; }
    if (MAX_BLOCKS< best) { best = MAX_BLOCKS;l = "blocks/SM"; }
    *limiter = l;
    return best;
}

// =============================================================================
// TODO 2 — the two-level reuse law.
//
// fmasPerGlobalLoad(BM, BN): for a block tile of BM x BN walking K in steps of
//   BK, how many fused multiply-adds does the block perform per element of A or
//   B that it loads from global memory? (It does not depend on BK. Show
//   yourself why.)
//
// fmasPerSharedRead(TM, TN): for a thread that owns a TM x TN sub-tile, how
//   many fused multiply-adds per SCALAR value read out of shared memory in the
//   inner loop?
//
// Both are checked against a reference by hash on fixed inputs.
// =============================================================================
static double fmasPerGlobalLoad(int BMv, int BNv)
{
    // Per k-tile a block loads BM*BK + BK*BN elements and performs
    // BM*BN*BK fused ops, so BK cancels.
    return (double)BMv * BNv / ((double)BMv + BNv);
}
static double fmasPerSharedRead(int TMv, int TNv)
{
    // Per value of k a thread reads TM + TN scalars and performs TM*TN ops.
    return (double)TMv * TNv / ((double)TMv + TNv);
}

// =============================================================================
// The kernel. Identical in structure to Exercise 1's; parameterised by tile
// shape and by a minimum-blocks-per-SM launch bound.
// =============================================================================
template<int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void body(int M, int N, int K, float alpha,
        const float *A, const float *B, float beta, float *C)
{
    const int NT = (BM/TM)*(BN/TN);
    const int AP = BM + 4;
    __shared__ float As[BK][AP];
    __shared__ float Bs[BK][BN];
    const int tid  = threadIdx.x;
    const int tRow = tid / (BN/TN), tCol = tid % (BN/TN);
    const int rowBase = blockIdx.y * BM, colBase = blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int kt = 0; kt < K; kt += BK) {
        #pragma unroll
        for (int u = 0; u < (BM*BK)/NT; ++u) {
            const int idx = tid + u*NT, r = idx / BK, c = idx % BK;
            As[c][r] = ((rowBase + r) < M && (kt + c) < K)
                     ? A[(size_t)(rowBase + r)*K + kt + c] : 0.0f;
        }
        #pragma unroll
        for (int u = 0; u < (BK*BN)/NT; ++u) {
            const int idx = tid + u*NT, k = idx / BN, n = idx % BN;
            Bs[k][n] = ((kt + k) < K && (colBase + n) < N)
                     ? B[(size_t)(kt + k)*N + colBase + n] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float rM[TM], rN[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rM[i] = As[kk][tRow*TM + i];
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
        const int r = rowBase + tRow*TM + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = colBase + tCol*TN + j;
            if (c < N) {
                if (beta == 0.0f) C[(size_t)r*N + c] = alpha * acc[i][j];
                else              C[(size_t)r*N + c] = alpha*acc[i][j] + beta*C[(size_t)r*N+c];
            }
        }
    }
}

template<int BM,int BN,int BK,int TM,int TN>
__global__ __launch_bounds__((BM/TM)*(BN/TN)) void gemmK(
        int M,int N,int K,float a,const float*A,const float*B,float b,float*C)
{ body<BM,BN,BK,TM,TN>(M,N,K,a,A,B,b,C); }

template<int BM,int BN,int BK,int TM,int TN,int MINB>
__global__ __launch_bounds__((BM/TM)*(BN/TN), MINB) void gemmKmin(
        int M,int N,int K,float a,const float*A,const float*B,float b,float*C)
{ body<BM,BN,BK,TM,TN>(M,N,K,a,A,B,b,C); }

// =============================================================================
// TODO 3 — PREDICTIONS. Commit before the program is built.
//
// P1: across the six swept tile shapes, the fastest one will turn out to be
//       1 : the configuration with the highest occupancy
//       2 : the configuration with the highest FMAs-per-shared-read
//       3 : neither of those - an interior point of the sweep
//       4 : the configuration with the fewest registers per thread
//
// P2: section B recompiles ONE of these kernels with __launch_bounds__ asking
//     for progressively more resident blocks, up to 100% occupancy. Relative to
//     the same kernel compiled with no such request, the 100%-occupancy build
//     will be:
//       1 : faster by more than 1.2x
//       2 : within 20% either way
//       3 : slower by 1.2x to 3x
//       4 : slower by 3x to 10x
//       5 : slower by more than 10x
// =============================================================================
#define PRED_P1 3   // an interior point: neither extreme wins
#define PRED_P2 5   // the accumulators spill; measured >10x slower

// =============================================================================
// The resource table the harness fills in before any timing happens.
// =============================================================================
typedef struct {
    const char *name;
    int TM, TN, BM, BN;
    int threads;
    int regs;          // from cudaFuncGetAttributes
    int spillBytes;    // from cudaFuncGetAttributes (localSizeBytes)
    int smemBytes;     // from cudaFuncGetAttributes
    int blocksPerSM;   // from the CUDA occupancy API
} TileInfo;

// =============================================================================
// TODO 4 — DESIGN: choose a tile shape from the resource table alone.
//
// You are given, for each of the `n` candidate configurations, everything the
// compiler and the occupancy API know: tile shape, registers, spill bytes,
// shared bytes, and resident blocks per SM. You are NOT given any timing.
//
// Return the index of the configuration you predict will be fastest.
//
// You are scored if your choice is within 5% of the measured best. Note that
// this is a genuine model-building problem and not an application of a rule
// stated anywhere in the lesson: neither "maximise occupancy" nor "maximise
// fmasPerSharedRead" nor "maximise the product of the two" gets it right on
// this data. Think about what the SM is actually short of at each point of the
// sweep, and about what a spill costs (Module 4 priced local memory).
// =============================================================================
static int chooseTile(const TileInfo *t, int n)
{
    // The model. Three saturating terms, one per scarce resource, multiplied:
    //
    //  g = fmasPerGlobalLoad(BM, BN)  - how hard the block tile works each
    //      element it pulls from global memory. This is what separates
    //      TM=8/TN=4 (BM 64 x BN 64, g = 32) from TM=4/TN=8 (BM 32 x BN 128,
    //      g = 25.6) even though the two have identical registers, identical
    //      shared memory, identical occupancy and identical shared-level reuse.
    //      Without this term the model cannot tell them apart, and they differ
    //      by nearly 1.5x in practice.
    //
    //  r = fmasPerSharedRead(TM, TN)  - how hard the thread tile works each
    //      value it pulls out of shared memory.
    //
    //  w = resident warps per SM      - the latency-hiding budget.
    //
    // Each term is fed through x/(x+c), which is monotone and saturating: more
    // is better, and past the knee more stops mattering. The knees (8, 1, 8)
    // are the points at which each resource stops being the binding one; the
    // shared-level knee is small because an LDS is cheap, the global-level one
    // is large because an L2 hit is 241 cycles (Module 4).
    //
    // A spilling configuration is rejected outright: a spill puts a local
    // memory access, i.e. DRAM, inside the innermost loop.
    int best = -1; double bestScore = -1.0;
    for (int i = 0; i < n; ++i) {
        if (t[i].spillBytes > 0) continue;
        const double g = fmasPerGlobalLoad(t[i].BM, t[i].BN);
        const double r = fmasPerSharedRead(t[i].TM, t[i].TN);
        const double w = (double)t[i].blocksPerSM * t[i].threads / 32.0;
        const double score = (g/(g + 8.0)) * (r/(r + 1.0)) * (w/(w + 8.0));
        if (score > bestScore) { bestScore = score; best = i; }
    }
    return best;
}

// =============================================================================
// TODO 5 — where does ptxas start spilling?
//
// Section B compiles the TM=8 TN=8 kernel (128 threads/block) five times, with
// __launch_bounds__(128, B) for B = 1, 2, 4, 8, 12. Name the SMALLEST B for
// which the compiler is forced to spill (localSizeBytes > 0).
//
// You can work this out with a pencil: you know how many registers the kernel
// wants when unconstrained (run -Xptxas -v), you know the register file size,
// and TODO 1 already made you find the allocation granule.
// =============================================================================
#define PRED_SPILL_AT 4   // see the solution notes

// =============================================================================
// harness
// =============================================================================
static int Mg = M_DIM, Ng = N_DIM, Kg = K_DIM;
static const float *dAg, *dBg;
static float *dCg;

template<int BM,int BN,int BK,int TM,int TN>
static void run(void) {
    dim3 gr((Ng+BN-1)/BN, (Mg+BM-1)/BM);
    gemmK<BM,BN,BK,TM,TN><<<gr,(BM/TM)*(BN/TN)>>>(Mg,Ng,Kg,1.0f,dAg,dBg,0.0f,dCg);
}
template<int BM,int BN,int BK,int TM,int TN,int MB>
static void runMin(void) {
    dim3 gr((Ng+BN-1)/BN, (Mg+BM-1)/BM);
    gemmKmin<BM,BN,BK,TM,TN,MB><<<gr,(BM/TM)*(BN/TN)>>>(Mg,Ng,Kg,1.0f,dAg,dBg,0.0f,dCg);
}

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
static unsigned fnv(const char *s) {
    unsigned h = 2166136261u;
    for (; *s; ++s) { h ^= (unsigned char)*s; h *= 16777619u; }
    return h;
}

#define NSW  6      // swept tile shapes
#define NLB  5      // launch-bound variants
#define NCFG (NSW + NLB)

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    const char *lim = "?";
    if (occupancyBlocks(64, 8192, 256, &lim) == 0) { printf("Set TODO 1 first.\n"); return 0; }
    if (fmasPerGlobalLoad(128,128) == 0.0 || fmasPerSharedRead(8,8) == 0.0) {
        printf("Set TODO 2 first.\n"); return 0; }
    if (PRED_P1 == 0 || PRED_P2 == 0) { printf("Set TODO 3 (PREDICTIONS) first.\n"); return 0; }
    if (PRED_SPILL_AT == 0) { printf("Set TODO 5 first.\n"); return 0; }

    const int M = M_DIM, N = N_DIM, K = K_DIM;
    const size_t sA = (size_t)M*K, sB = (size_t)K*N, sC = (size_t)M*N;
    printf("=== Module 18 / Exercise 2 - the register/occupancy trade ===\n");
    printf("problem %d x %d x %d, %d threads per block, BK = %d\n\n",
           M, N, K, NTHREADS, BKD);

    float *hA = (float*)malloc(sA*4), *hB = (float*)malloc(sB*4), *hC = (float*)malloc(sC*4);
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

    struct Cfg { const char *name; void (*run)(void); const void *fn;
                 int thr, TM, TN, BM, BN; };
    Cfg cfg[NCFG] = {
      {"TM=2  TN=2   BM16  BN32",  run<16,32,BKD,2,2>,   (const void*)gemmK<16,32,BKD,2,2>,   128, 2, 2, 16, 32},
      {"TM=4  TN=4   BM32  BN64",  run<32,64,BKD,4,4>,   (const void*)gemmK<32,64,BKD,4,4>,   128, 4, 4, 32, 64},
      {"TM=8  TN=4   BM64  BN64",  run<64,64,BKD,8,4>,   (const void*)gemmK<64,64,BKD,8,4>,   128, 8, 4, 64, 64},
      {"TM=4  TN=8   BM32  BN128", run<32,128,BKD,4,8>,  (const void*)gemmK<32,128,BKD,4,8>,  128, 4, 8, 32,128},
      {"TM=8  TN=8   BM64  BN128", run<64,128,BKD,8,8>,  (const void*)gemmK<64,128,BKD,8,8>,  128, 8, 8, 64,128},
      {"TM=16 TN=8   BM128 BN128", run<128,128,BKD,16,8>,(const void*)gemmK<128,128,BKD,16,8>,128,16, 8,128,128},
      {"8x8, __launch_bounds__(128,1)",  runMin<64,128,BKD,8,8,1>,  (const void*)gemmKmin<64,128,BKD,8,8,1>,  128, 8, 8, 64,128},
      {"8x8, __launch_bounds__(128,2)",  runMin<64,128,BKD,8,8,2>,  (const void*)gemmKmin<64,128,BKD,8,8,2>,  128, 8, 8, 64,128},
      {"8x8, __launch_bounds__(128,4)",  runMin<64,128,BKD,8,8,4>,  (const void*)gemmKmin<64,128,BKD,8,8,4>,  128, 8, 8, 64,128},
      {"8x8, __launch_bounds__(128,8)",  runMin<64,128,BKD,8,8,8>,  (const void*)gemmKmin<64,128,BKD,8,8,8>,  128, 8, 8, 64,128},
      {"8x8, __launch_bounds__(128,12)", runMin<64,128,BKD,8,8,12>, (const void*)gemmKmin<64,128,BKD,8,8,12>, 128, 8, 8, 64,128},
    };

    // ---------------- TODO 1 check, plus the resource table
    printf("-- resources (compiler + occupancy API), before any timing --------\n");
    printf(" %-31s %5s %6s %6s %6s %8s %10s %8s\n",
           "config","regs","spillB","smemB","blk/SM","yours","limiter","occ%");
    int occOK = 0;
    TileInfo info[NCFG];
    for (int i = 0; i < NCFG; ++i) {
        cudaFuncAttributes at; CHECK(cudaFuncGetAttributes(&at, cfg[i].fn));
        int blk = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blk, cfg[i].fn, cfg[i].thr, 0));
        const char *l = "?";
        int mine = occupancyBlocks(at.numRegs, (int)at.sharedSizeBytes, cfg[i].thr, &l);
        if (mine == blk) ++occOK;
        info[i].name = cfg[i].name; info[i].TM = cfg[i].TM; info[i].TN = cfg[i].TN;
        info[i].BM = cfg[i].BM; info[i].BN = cfg[i].BN; info[i].threads = cfg[i].thr;
        info[i].regs = at.numRegs; info[i].spillBytes = (int)at.localSizeBytes;
        info[i].smemBytes = (int)at.sharedSizeBytes; info[i].blocksPerSM = blk;
        printf(" %-31s %5d %6d %6d %6d %8d %10s %7.1f%%%s\n",
               cfg[i].name, at.numRegs, (int)at.localSizeBytes, (int)at.sharedSizeBytes,
               blk, mine, l, 100.0*blk*cfg[i].thr/1536.0,
               (mine == blk) ? "" : "   <-- MISMATCH");
    }
    printf("  occupancy model: %d/%d configurations match the CUDA API\n", occOK, NCFG);

    // ---------------- TODO 2 check
    char buf[256];
    snprintf(buf, sizeof buf, "%.6f|%.6f|%.6f|%.6f",
             fmasPerGlobalLoad(128,128), fmasPerGlobalLoad(64,128),
             fmasPerSharedRead(8,8),     fmasPerSharedRead(16,4));
    const unsigned LAW_HASH = 0x7674d719u;
    int lawOK = (fnv(buf) == LAW_HASH);
    printf("\n  two-level reuse law: %s\n", lawOK ? "correct" : "WRONG");
    printf("    block tile 128x128 -> %8.3f FMAs per global load\n", fmasPerGlobalLoad(128,128));
    printf("    thread tile 8x8    -> %8.3f FMAs per scalar shared read\n", fmasPerSharedRead(8,8));

    // ---------------- TODO 4: the model chooses, before timing
    int pick = chooseTile(info, NSW);
    if (pick < 0 || pick >= NSW) {
        printf("\n  chooseTile() returned %d, which is not in [0,%d). Set TODO 4.\n", pick, NSW);
        printf("\nOVERALL: FAIL\n"); CHECK(cudaDeviceReset()); return 1;
    }
    printf("\n  your cost model picks: %s\n", info[pick].name);

    // ---------------- warm-up and timing
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

    int iters[NCFG]; double best[NCFG];
    for (int i = 0; i < NCFG; ++i) {
        double t = timeOne(cfg[i].run, 1);
        int n = (int)(10.0/(t > 0 ? t : 0.01));
        if (n < 3) n = 3; if (n > 64) n = 64; iters[i] = n; best[i] = 1e30;
    }
    CHECK(cudaGetLastError());
    for (int s = 0; s < NCFG; ++s)
        for (int q = 0; q < NCFG; ++q) {
            int p = (q + s) % NCFG;
            double t = timeOne(cfg[p].run, iters[p]);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    const double flops = 2.0*M*N*K;
    printf("\n-- A. the thread-tile sweep --------------------------------------\n");
    printf(" %-31s %6s %6s %6s %8s %10s\n","config","regs","blk/SM","occ%","f/read","GFLOP/s");
    int bestIdx = 0;
    for (int i = 0; i < NSW; ++i) if (best[i] < best[bestIdx]) bestIdx = i;
    for (int i = 0; i < NSW; ++i)
        printf(" %-31s %6d %6d %5.1f%% %8.2f %10.1f%s\n", cfg[i].name, info[i].regs,
               info[i].blocksPerSM, 100.0*info[i].blocksPerSM*cfg[i].thr/1536.0,
               fmasPerSharedRead(cfg[i].TM, cfg[i].TN), flops/(best[i]*1e-3)/1e9,
               (i == bestIdx) ? "   <-- fastest" : "");

    printf("\n-- B. the same 8x8 kernel, forced to more resident blocks ---------\n");
    printf(" %-31s %6s %6s %6s %6s %10s\n","config","regs","spillB","blk/SM","occ%","GFLOP/s");
    int spillAt = 0; const int lbArg[NLB] = {1,2,4,8,12};
    for (int i = NSW; i < NCFG; ++i) {
        if (!spillAt && info[i].spillBytes > 0) spillAt = lbArg[i - NSW];
        printf(" %-31s %6d %6d %6d %5.1f%% %10.1f\n", cfg[i].name, info[i].regs,
               info[i].spillBytes, info[i].blocksPerSM,
               100.0*info[i].blocksPerSM*cfg[i].thr/1536.0, flops/(best[i]*1e-3)/1e9);
    }

    // ---------------- scoring
    const double gBase = flops/(best[NSW]*1e-3)/1e9;          // launch_bounds(...,1)
    const double gFull = flops/(best[NCFG-1]*1e-3)/1e9;       // 100% occupancy build
    const double ratio = gBase / gFull;
    // Which description fits the measured winner?
    int idxOcc = 0, idxReuse = 0, idxFewReg = 0;
    for (int i = 1; i < NSW; ++i) {
        if (info[i].blocksPerSM*info[i].threads > info[idxOcc].blocksPerSM*info[idxOcc].threads)
            idxOcc = i;
        if (fmasPerSharedRead(info[i].TM, info[i].TN)
            > fmasPerSharedRead(info[idxReuse].TM, info[idxReuse].TN)) idxReuse = i;
        if (info[i].regs < info[idxFewReg].regs) idxFewReg = i;
    }
    int trueP1 = (bestIdx == idxOcc) ? 1 : (bestIdx == idxReuse) ? 2
               : (bestIdx == idxFewReg) ? 4 : 3;
    int trueP2 = (ratio < 1.0/1.2) ? 1 : (ratio < 1.2) ? 2 : (ratio < 3.0) ? 3
               : (ratio < 10.0) ? 4 : 5;
    const double gPick = flops/(best[pick]*1e-3)/1e9;
    const double gBest = flops/(best[bestIdx]*1e-3)/1e9;
    int modelOK = (gPick >= 0.85*gBest);

    printf("\n-- scoring --------------------------------------------------------\n");
    printf("  P1 shape of the winner  : you said %d, measured %d   %s\n",
           PRED_P1, trueP1, PRED_P1 == trueP1 ? "correct" : "WRONG");
    printf("  P2 100%%-occupancy build : you said %d, measured %d  (%.2fx slower)   %s\n",
           PRED_P2, trueP2, ratio, PRED_P2 == trueP2 ? "correct" : "WRONG");
    printf("  cost model              : picked %s at %.0f GFLOP/s, best is %.0f  %s\n",
           info[pick].name, gPick, gBest, modelOK ? "within 15%" : "MISSED");
    printf("  first spilling bound    : you said %d, measured %d   %s\n",
           PRED_SPILL_AT, spillAt, PRED_SPILL_AT == spillAt ? "correct" : "WRONG");

    // ---------------- correctness, second untimed pass
    printf("\n-- correctness (second, untimed pass) ------------------------------\n");
    int numOK = 0;
    for (int i = 0; i < NCFG; ++i) {
        float *p = (float*)malloc(sC*4);
        for (size_t j = 0; j < sC; ++j) p[j] = std::numeric_limits<float>::infinity();
        CHECK(cudaMemcpy(dC, p, sC*4, cudaMemcpyHostToDevice)); free(p);
        cfg[i].run(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hC, dC, sC*4, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (size_t j = 0; j < sC && bad == 0; ++j) if (!isfinite(hC[j])) bad = 1;
        double ref = 0.0, got = 0.0;
        for (int k = 0; k < K; ++k) ref += (double)hA[(size_t)7*K+k]*hB[(size_t)k*N+11];
        got = hC[(size_t)7*N + 11];
        if (!bad && fabs(got - ref) <= 4.0*K*ldexp(1.0,-24)*fabs(ref)) ++numOK;
    }
    printf("  %d/%d kernels produce a finite, correct C\n", numOK, NCFG);

    int score = 0;
    if (occOK == NCFG) score += 3;
    if (lawOK) score += 1;
    if (PRED_P1 == trueP1) score += 1;
    if (PRED_P2 == trueP2) score += 1;
    if (modelOK) score += 2;
    if (PRED_SPILL_AT == spillAt) score += 2;
    if (numOK != NCFG) score = 0;
    printf("\n  score %d/10\n", score);

    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    printf("\nOVERALL: %s\n", score == 10 ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return score == 10 ? 0 : 1;
}
