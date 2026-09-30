// ============================================================================
// Module 12 / exercise01.cu -- build the reduction ladder
//
// GOAL : Six kernels sum the same 2^26 floats. Version 1 is complete. You
//        write the key lines of versions 2 through 6, each of which repairs a
//        specific defect of the one before it. Before you run anything you
//        commit to four predictions about what the measurements will show.
//
//        The harness validates every version against a double-precision CPU
//        reference, times all six plus a pure-streaming ceiling in one rotated
//        sweep, and reports each version as a percentage of that ceiling.
//        Reduction is a BANDWIDTH problem: N floats in, one float out. The
//        only question worth asking about any version is what fraction of the
//        memory system it managed to use.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN  : .\exercise01.exe
//
// SASS : nvcc -arch=sm_89 -O3 -c -o exercise01.o exercise01.cu
//        cuobjdump -sass exercise01.o > exercise01.sass
//        findstr /C:"SHFL" /C:"BAR.SYNC" exercise01.sass
//
// PASS : all six versions numerically correct AND all four predictions right.
//        SCORE: 10/10.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(_e), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_KERNEL()                                                         \
    do { CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize()); } while (0)

static const unsigned  BS = 256;          // threads per block, every version
static const long long N  = 1LL << 26;    // 67,108,864 floats = 256 MiB
static const int NCFG   = 7;              // six versions + the ceiling
static const int NSWEEP = 4;

// ===========================================================================
// TODO 1 -- PREDICTIONS. Commit to all four before you compile.
//
//   Reason them out from the cost model, not from intuition about "more
//   optimized is faster". You know from Module 5 what a streaming
//   kernel costs on this GPU, you know from Module 8 what divergence costs,
//   you know from Module 7 what a bank conflict costs, and you know from
//   Module 9 what a barrier costs. That is enough to answer all four.
//
//   (a) PRED_BIGGEST_STEP: which version gains the most over its immediate
//       predecessor? Answer with the version number, 2..6.
//
//   (b) PRED_TOTAL_SPEEDUP: the ratio v1_time / v6_time. Scored within +-30%.
//
//   (c) PRED_FIRST_AT_95: the LOWEST version number that reaches 95% of the
//       measured streaming ceiling. Answer 1..6, or 7 if none of them do.
//
//   (d) PRED_BARRIER_REMOVABLE: version 5 stops its shared-memory loop at
//       s == 32 and finishes in registers. Consider the __syncthreads() at
//       the BOTTOM of the last shared-memory iteration -- the one that
//       executes after `sdata[tid] += sdata[tid+32]`. Can that particular
//       barrier be deleted without making the kernel incorrect?
//       1 = yes, it can be deleted; 0 = no, it is still required.
//       Justify it in terms of WHICH threads write the slots that are read
//       after it, not in terms of "warps are in lockstep" -- Module 8 spent
//       a whole exercise establishing that that reasoning is dead.
// ===========================================================================
static const int   PRED_BIGGEST_STEP      = 0;    // YOUR CODE HERE (2..6)
static const float PRED_TOTAL_SPEEDUP     = 0.0f; // YOUR CODE HERE
static const int   PRED_FIRST_AT_95       = 0;    // YOUR CODE HERE (1..7)
static const int   PRED_BARRIER_REMOVABLE = -1;   // YOUR CODE HERE (0 or 1)

// ===========================================================================
// v1 -- interleaved addressing with a modulo test. COMPLETE. This is your
//       baseline and the thing every later version is measured against.
// ===========================================================================
__global__ void reduce1(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    unsigned tid = threadIdx.x;
    long long i  = (long long)blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ===========================================================================
// TODO 2 -- v2. Same tree, same number of additions, same order of additions.
//           The ONLY thing you may change is which thread performs which
//           addition. Make the threads that are doing work be a contiguous
//           block of low-numbered threads, so that a warp is either entirely
//           working or entirely idle.
//
//           Do not restructure the tree. If your v2 produces a different
//           floating-point result from v1, you changed more than the mapping.
// ===========================================================================
__global__ void reduce2(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    unsigned tid = threadIdx.x;
    long long i  = (long long)blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        // YOUR CODE HERE
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ===========================================================================
// TODO 3 -- v3. v2 removed the divergence but its shared-memory addresses are
//           now strided. Restructure the tree so that thread `tid` always
//           reads and writes slots that keep a warp's 32 addresses inside one
//           conflict-free set, and so that the surviving values stay packed at
//           the bottom of the array.
//
//           Write the loop yourself, including its bound and its direction.
//           The bound is where people go wrong: v5 below stops early on
//           purpose, and copying that bound into v3 produces a kernel that
//           silently drops 31/32 of every block's data.
// ===========================================================================
__global__ void reduce3(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    unsigned tid = threadIdx.x;
    long long i  = (long long)blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // YOUR CODE HERE  (the whole tree loop, barriers included)

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ===========================================================================
// TODO 4 -- the load. v3 still wastes its first iteration: half the threads of
//           the block do nothing but sit at a barrier. Fix it at the source --
//           halve the number of blocks and have every thread arrive at the
//           tree already holding the sum of TWO input elements.
//
//           `blocks4` below is already halved for you. Return the value that
//           thread `tid` of block `blockIdx.x` should place in sdata[tid].
//           Guard both reads: n need not be a multiple of anything.
// ===========================================================================
__device__ __forceinline__ float loadTwoAndAdd(const float* __restrict__ in,
                                               long long n, unsigned tid)
{
    // YOUR CODE HERE
    // The (void) casts exist only so the shipped file compiles clean. Delete.
    (void)in; (void)n; (void)tid;
    return 0.0f;
}

__global__ void reduce4(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    unsigned tid = threadIdx.x;
    sdata[tid] = loadTwoAndAdd(in, n, tid);
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ===========================================================================
// TODO 5 -- DESIGN. Three pieces, and you choose the shape of the third.
//
//  5a. warpReduceSum(v): reduce one 32-lane warp to a single value using
//      __shfl_down_sync, with no shared memory and no barrier. Five steps.
//
//      Two things will bite you. First, the mask: __shfl_down_sync's first
//      argument names the lanes that must participate, and on sm_70+ that
//      mask is how you CREATE convergence -- it is not a description of what
//      the hardware happens to be doing (Module 8). Passing __activemask()
//      here is a bug even when it returns the value you wanted. Second, the
//      result of a down-shuffle is only meaningful in the LOWER lanes; a lane
//      that shuffles from beyond the warp receives its own value back. Decide
//      which lane you trust and write only from that one.
//
//  5b. reduce5's warp tail: the shared loop below already stops at s == 32.
//      Take it from there.
//
//  5c. reduce6: the fully optimized version. The requirements, not the code:
//        - the grid size must NOT be a function of n. The harness refuses any
//          grid above 4096 blocks and tells you so. Pick it from the machine.
//        - every thread must accumulate MANY input elements into a REGISTER
//          before touching shared memory at all, so that the shared tree and
//          its barriers run once per block instead of once per 512 elements.
//        - the tree itself must have compile-time bounds. BLOCK is a template
//          parameter; use it.
//        - it must be correct for any n, including n not a multiple of
//          anything.
//      Fill chooseGridV6() as well; returning 0 keeps the harness quiet.
// ===========================================================================
__device__ __forceinline__ float warpReduceSum(float v)
{
    // YOUR CODE HERE  (5a)
    return v;
}

__global__ void reduce5(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BS];
    unsigned tid = threadIdx.x;
    sdata[tid] = loadTwoAndAdd(in, n, tid);
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // YOUR CODE HERE  (5b: the warp tail, and the write to out[blockIdx.x])
    (void)sdata;
}

template <unsigned BLOCK>
__global__ void reduce6(const float* __restrict__ in, float* out, long long n)
{
    // YOUR CODE HERE  (5c)   -- you will want  __shared__ float sdata[BLOCK];
    // The three (void) casts below exist only so that the file compiles
    // warning-clean as shipped. Delete them.
    (void)in; (void)out; (void)n;
}

// Return the number of blocks reduce6 should be launched with. Must not
// depend on n. Must be <= 4096.
static int chooseGridV6(void)
{
    // YOUR CODE HERE  (5c)
    return 0;
}

// ===========================================================================
// Everything below is the harness. You do not need to modify it.
// ===========================================================================

// Deterministic fixed-order finalization, identical for every version, so the
// comparison is between the six pass-1 kernels and nothing else.
__device__ __forceinline__ float warpReduceSumRef(float v)
{
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

__global__ void finalizeKernel(const float* __restrict__ partial, float* out, long long m)
{
    __shared__ float sdata[1024];
    unsigned tid = threadIdx.x;
    float sum = 0.0f;
    for (long long i = tid; i < m; i += blockDim.x) sum += partial[i];
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) { float w = warpReduceSumRef(sdata[tid]); if (tid == 0) *out = w; }
}

__global__ void streamCeiling(const float* __restrict__ in, float* out, long long n)
{
    long long i    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)blockDim.x * gridDim.x;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (; i + 3 * step < n; i += 4 * step) {
        a0 += in[i]; a1 += in[i + step];
        a2 += in[i + 2 * step]; a3 += in[i + 3 * step];
    }
    for (; i < n; i += step) a0 += in[i];
    float s = (a0 + a1) + (a2 + a3);
    if (s == 1.2345e-30f) out[blockIdx.x] = s;
}

static int g_blocks[NCFG];

static void launchVersion(int v, const float* d_in, float* d_partial, long long n)
{
    switch (v) {
        case 0: reduce1<<<g_blocks[0], BS>>>(d_in, d_partial, n); break;
        case 1: reduce2<<<g_blocks[1], BS>>>(d_in, d_partial, n); break;
        case 2: reduce3<<<g_blocks[2], BS>>>(d_in, d_partial, n); break;
        case 3: reduce4<<<g_blocks[3], BS>>>(d_in, d_partial, n); break;
        case 4: reduce5<<<g_blocks[4], BS>>>(d_in, d_partial, n); break;
        case 5: reduce6<BS><<<g_blocks[5], BS>>>(d_in, d_partial, n); break;
        case 6: streamCeiling<<<g_blocks[6], BS>>>(d_in, d_partial, n); break;
        default: break;
    }
}

int main(void)
{
    if (PRED_BIGGEST_STEP == 0 || PRED_FIRST_AT_95 == 0 ||
        PRED_BARRIER_REMOVABLE < 0 || PRED_TOTAL_SPEEDUP <= 0.0f) {
        printf("Set TODO 1 first.\n");
        return 0;
    }
    int gridV6 = chooseGridV6();
    if (gridV6 <= 0) {
        printf("Set TODO 5c first (chooseGridV6 returned %d).\n", gridV6);
        return 0;
    }
    if (gridV6 > 4096) {
        printf("chooseGridV6 returned %d. The grid for version 6 must be a\n"
               "property of the machine, not of n, and must be <= 4096.\n", gridV6);
        return 0;
    }

    int smCount = 0;
    CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
    printf("Module 12 exercise 01 -- the reduction ladder\n");
    printf("%d SMs, N = %lld floats = %.0f MiB, block = %u threads\n",
           smCount, N, (double)N * 4.0 / 1048576.0, BS);
    printf("v6 grid = %d blocks\n\n", gridV6);

    g_blocks[0] = g_blocks[1] = g_blocks[2] = (int)(N / BS);
    g_blocks[3] = g_blocks[4] = (int)(N / (BS * 2));
    g_blocks[5] = gridV6;
    g_blocks[6] = gridV6;

    float* h_in = (float*)malloc((size_t)N * sizeof(float));
    if (!h_in) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (long long i = 0; i < N; ++i) {
        unsigned h = (unsigned)i * 2654435761u;
        h ^= h >> 15;
        h_in[i] = (float)(h & 0xFFFFu) * (1.0f / 65536.0f);
    }
    double ref = 0.0;
    for (long long i = 0; i < N; ++i) ref += (double)h_in[i];

    float *d_in = nullptr, *d_partial = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in, (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_partial, (size_t)(N / BS + 64) * sizeof(float)));
    CHECK(cudaMalloc(&d_out, sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    // duration-based clock warm-up: 400 ms ramps the SM clock, but the memory
    // P-state on this laptop part needs longer. See the lesson.
    {
        cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        float acc = 0.0f; CHECK(cudaEventRecord(a));
        while (acc < 1500.0f) {
            for (int k = 0; k < 20; ++k) streamCeiling<<<gridV6, BS>>>(d_in, d_partial, N);
            CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
            CHECK(cudaEventElapsedTime(&acc, a, b));
        }
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); CHECK_KERNEL();
    }

    int iters[NCFG];
    for (int v = 0; v < NCFG; ++v) {
        launchVersion(v, d_in, d_partial, N); CHECK_KERNEL();
        CHECK(cudaEventRecord(e0));
        for (int k = 0; k < 5; ++k) launchVersion(v, d_in, d_partial, N);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        double per = ms / 5.0;
        int it = (int)(10.0 / (per > 1e-6 ? per : 1e-6));
        if (it < 20)  it = 20;
        if (it > 400) it = 400;
        iters[v] = it;
    }

    double best[NCFG];
    for (int v = 0; v < NCFG; ++v) best[v] = 1e30;
    for (int sweep = 0; sweep < NSWEEP; ++sweep) {
        for (int q = 0; q < NCFG; ++q) {
            int v = (q + sweep) % NCFG;          // rotate the sweep order
            CHECK(cudaEventRecord(e0));
            for (int k = 0; k < iters[v]; ++k) launchVersion(v, d_in, d_partial, N);
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms / iters[v] < best[v]) best[v] = ms / iters[v];
        }
    }
    CHECK_KERNEL();

    // validation, separate pass
    float got[NCFG];
    int okCount = 0, okv[NCFG];
    for (int v = 0; v < NCFG - 1; ++v) {
        launchVersion(v, d_in, d_partial, N); CHECK_KERNEL();
        finalizeKernel<<<1, 1024>>>(d_partial, d_out, g_blocks[v]);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(&got[v], d_out, sizeof(float), cudaMemcpyDeviceToHost));
        double rel = fabs((double)got[v] - ref) / fabs(ref);
        okv[v] = (rel <= 1e-5) ? 1 : 0;
        okCount += okv[v];
    }

    double ceilMs  = best[NCFG - 1];
    double ceilGBs = (double)N * 4.0 / (ceilMs * 1e-3) / 1e9;
    const char* nm[NCFG] = {
        "v1 interleaved, tid mod 2s  ", "v2 contiguous index         ",
        "v3 sequential addressing    ", "v4 first add during load    ",
        "v5 warp tail (shuffles)     ", "v6 grid-stride + unrolled   ",
        "   streaming ceiling        " };

    printf("%-30s %8s %9s %9s %9s %8s\n",
           "version", "ms", "GB/s", "%ceiling", "step", "result");
    printf("-------------------------------------------------------------------------------\n");
    for (int v = 0; v < NCFG; ++v) {
        double gbs = (double)N * 4.0 / (best[v] * 1e-3) / 1e9;
        if (v == NCFG - 1)
            printf("%-30s %8.4f %9.1f %8.1f%% %9s %8s\n",
                   nm[v], best[v], gbs, 100.0 * gbs / ceilGBs, "-", "-");
        else
            printf("%-30s %8.4f %9.1f %8.1f%% %8.2fx %8s\n",
                   nm[v], best[v], gbs, 100.0 * gbs / ceilGBs,
                   (v == 0) ? 1.0 : best[v - 1] / best[v],
                   okv[v] ? "ok" : "WRONG");
    }
    printf("\nceiling %.1f GB/s (%.1f%% of the 432.0 GB/s pin peak)\n",
           ceilGBs, 100.0 * ceilGBs / 432.0);
    printf("double reference = %.6f\n", ref);
    for (int v = 0; v < NCFG - 1; ++v)
        printf("  %-30s got %.6f  rel.err %.3e\n", nm[v], (double)got[v],
               fabs((double)got[v] - ref) / fabs(ref));

    // ---- score the predictions -------------------------------------------
    int biggest = 2; double bestStep = 0.0;
    for (int v = 1; v <= 5; ++v) {
        double r = best[v - 1] / best[v];
        if (r > bestStep) { bestStep = r; biggest = v + 1; }
    }
    double total = best[0] / best[5];
    int firstAt95 = 7;
    for (int v = 0; v < NCFG - 1; ++v) {
        double gbs = (double)N * 4.0 / (best[v] * 1e-3) / 1e9;
        if (gbs >= 0.95 * ceilGBs) { firstAt95 = v + 1; break; }
    }

    int p = 0;
    printf("\npredictions:\n");
    int a_ok = (PRED_BIGGEST_STEP == biggest); p += a_ok;
    printf("  (a) biggest single step   predicted v%d, measured v%d (%.2fx)  %s\n",
           PRED_BIGGEST_STEP, biggest, bestStep, a_ok ? "MATCH" : "MISS");
    int b_ok = (fabs(PRED_TOTAL_SPEEDUP - total) <= 0.30 * total); p += b_ok;
    printf("  (b) v1/v6 speedup         predicted %.2fx, measured %.2fx       %s\n",
           (double)PRED_TOTAL_SPEEDUP, total, b_ok ? "MATCH" : "MISS");
    int c_ok = (PRED_FIRST_AT_95 == firstAt95); p += c_ok;
    printf("  (c) first version >= 95%%  predicted v%d, measured v%d            %s\n",
           PRED_FIRST_AT_95, firstAt95, c_ok ? "MATCH" : "MISS");
    int d_ok = (PRED_BARRIER_REMOVABLE == 1); p += d_ok;
    printf("  (d) last barrier of v5's shared loop removable: you said %s  %s\n",
           PRED_BARRIER_REMOVABLE ? "yes" : "no", d_ok ? "MATCH" : "MISS");

    int score = okCount + p;
    printf("\nSCORE: %d/10   (%d/6 versions correct, %d/4 predictions)\n",
           score, okCount, p);

    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_partial)); CHECK(cudaFree(d_out));
    free(h_in);
    CHECK(cudaDeviceReset());

    printf("\nOVERALL: %s\n", (score == 10) ? "PASS" : "FAIL");
    return (score == 10) ? 0 : 1;
}
