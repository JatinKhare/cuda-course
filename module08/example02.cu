// =====================================================================
// Module 8 / Example 2 : the price of divergence, measured; and where
//                        the compiler stops predicating and starts
//                        branching.
//
// GOAL
//   Three measurements, all on the same kernel shape, all timed under
//   the Module-wide benchmarking rules (warm-up, back-to-back configs,
//   min-of-N sweeps, ratios reported as the stable quantity):
//
//   A) BRANCH GROUPING. An if/else whose two arms cost the same, driven
//      by four different predicates over the SAME data and the SAME
//      total work:
//          uniform        : every lane takes arm A
//          warp-uniform   : whole warps take A or B   ((tid/128) test)
//          lane-alternate : threadIdx.x & 1
//          8-lane groups  : (threadIdx.x >> 3) & 1
//      Predict the four ratios before you look.
//
//   B) LOOP TRIP-COUNT DIVERGENCE. A loop whose trip count is per-lane.
//      The warp runs at the MAXIMUM trip count in the warp, not the
//      mean. Compared against a warp-homogeneous arrangement with
//      identical total work.
//
//   C) PREDICATION FLIP POINT. Ten instantiations of the same if/else
//      with N = 1..16 FFMAs per arm. Disassemble and find the N at
//      which the compiler stops emitting predicated instructions and
//      starts emitting a real branch:
//
//        nvcc -arch=sm_89 -O3 -c -o example02.o example02.cu
//        cuobjdump -sass example02.o > example02.sass
//
//      then look for "@P0 FFMA" (predicated) versus "BSSY"/"BRA"/"BSYNC"
//      (real control flow) inside each _Z6flipKerILi..E instantiation.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN:    .\example02.exe
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

// One resident wave on this GPU: 40 SMs x 1536 threads / 256 = 240 blocks.
static const int BLOCK  = 256;
static const int GRID   = 240;
static const int N      = BLOCK * 1920;   // sized for the deeper part-B grid
static const int ARMLEN = 1024;   // FFMAs per arm
static const int REP    = 8;      // outer repeats, to make the kernel long

// ---------------------------------------------------------------------
// A) Four groupings of the same if/else.
//    Both arms cost exactly ARMLEN dependent FFMAs. Only the predicate
//    differs, so total arithmetic across the grid is identical in all
//    four configurations. Any difference is divergence, nothing else.
// ---------------------------------------------------------------------
template <int MODE>
__global__ void groupKer(const float* __restrict__ in, float* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = in[i];
    bool c;
    if      (MODE == 0) c = true;                       // uniform
    else if (MODE == 1) c = (threadIdx.x < 128);        // warp-uniform
    else if (MODE == 2) c = (threadIdx.x & 1) != 0;     // lane-alternate
    else                c = ((threadIdx.x >> 3) & 1) != 0; // 8-lane groups

    for (int r = 0; r < REP; ++r) {
        if (c) { for (int j = 0; j < ARMLEN; ++j) a = fmaf(a, 0.9999f,  0.0001f); }
        else   { for (int j = 0; j < ARMLEN; ++j) a = fmaf(a, 0.9998f, -0.0001f); }
    }
    out[i] = a;
}

// ---------------------------------------------------------------------
// B) Trip-count divergence.
//    trips in {1..8} times TRIPUNIT iterations.
//    MODE 0: trips = lane & 7   -> every warp contains all eight values,
//                                  so the warp runs at trips = 8.
//    MODE 1: trips = blockIdx.x & 7, so every warp of a block agrees.
//                                  Part B uses a grid several waves deep so
//                                  that the block scheduler can refill SMs
//                                  as the cheap blocks retire; otherwise the
//                                  comparison measures load imbalance
//                                  between warps rather than divergence.
//    Total work over the grid is identical.
// ---------------------------------------------------------------------
static const int TRIPUNIT = 1024;

template <int MODE>
__global__ void tripKer(const float* __restrict__ in, float* __restrict__ out)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int trips = (MODE == 0) ? (lane & 7) + 1        // all 8 values in one warp
                            : (int)(blockIdx.x & 7) + 1;  // whole block agrees

    float a = in[i];
    for (int t = 0; t < trips; ++t)
        for (int j = 0; j < TRIPUNIT; ++j) a = fmaf(a, 0.9999f, 0.0001f);
    out[i] = a;
}

// ---------------------------------------------------------------------
// C) Predication flip point. Never launched; present so that the SASS
//    exists in the binary for you to disassemble.
// ---------------------------------------------------------------------
template <int NFFMA>
__global__ void flipKer(const float* __restrict__ in, float* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i], a = v;
    if (threadIdx.x & 1) {
        #pragma unroll
        for (int j = 0; j < NFFMA; ++j) a = fmaf(a, v,  1.0f);
    } else {
        #pragma unroll
        for (int j = 0; j < NFFMA; ++j) a = fmaf(a, v, -1.0f);
    }
    out[i] = a;
}
// The Module 3 bounds guard, in three shapes. Module 3 measured the
// guard as "predicated" and promised Module 8 would make that precise.
// These three kernels are never launched; they exist so that their SASS
// is in the binary. Disassemble them and compare.
//
//   guardTail  -- the guard is the last thing the thread does
//   guardShort -- guarded body is one store, and work follows the guard
//   guardLoad  -- same, but the guarded body contains a global load
__global__ void guardTail(const float* __restrict__ in, float* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 2.0f * in[i] + 1.0f;
}
__global__ void guardShort(float* __restrict__ out, float* __restrict__ tag, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 3.0f;
    tag[i] = (float)i;
}
__global__ void guardLoad(const float* __restrict__ in, float* __restrict__ out,
                          float* __restrict__ tag, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = in[i & (n - 1)];
    if (i < n) out[i] = 2.0f * v + 1.0f;
    tag[i] = (float)i;
}

template __global__ void flipKer<1>(const float* __restrict__, float* __restrict__, int);
template __global__ void flipKer<2>(const float* __restrict__, float* __restrict__, int);
template __global__ void flipKer<3>(const float* __restrict__, float* __restrict__, int);
template __global__ void flipKer<4>(const float* __restrict__, float* __restrict__, int);
template __global__ void flipKer<5>(const float* __restrict__, float* __restrict__, int);
template __global__ void flipKer<6>(const float* __restrict__, float* __restrict__, int);
template __global__ void flipKer<7>(const float* __restrict__, float* __restrict__, int);
template __global__ void flipKer<8>(const float* __restrict__, float* __restrict__, int);
template __global__ void flipKer<12>(const float* __restrict__, float* __restrict__, int);
template __global__ void flipKer<16>(const float* __restrict__, float* __restrict__, int);

// ---------------------------------------------------------------------
static void launchGroup(int m, const float* in, float* out)
{
    switch (m) {
        case 0: groupKer<0><<<GRID, BLOCK>>>(in, out); break;
        case 1: groupKer<1><<<GRID, BLOCK>>>(in, out); break;
        case 2: groupKer<2><<<GRID, BLOCK>>>(in, out); break;
        default:groupKer<3><<<GRID, BLOCK>>>(in, out); break;
    }
}
static const int GRID_B = 1920;      // 8 waves, so blocks can be rebalanced
static void launchTrip(int m, const float* in, float* out)
{
    if (m == 0) tripKer<0><<<GRID_B, BLOCK>>>(in, out);
    else        tripKer<1><<<GRID_B, BLOCK>>>(in, out);
}

// =====================================================================
int main(void)
{
    CHECK(cudaSetDevice(0));
    printf("=== Module 8 / Example 2 : the measured price of divergence ===\n");
    printf("part A grid %d x %d = %d threads (one resident wave), "
           "arm length %d FFMAs, %d repeats\n", GRID, BLOCK, GRID * BLOCK, ARMLEN, REP);
    printf("part B grid %d x %d = %d threads (several waves)\n\n",
           GRID_B, BLOCK, GRID_B * BLOCK);

    float* h_in = (float*)malloc((size_t)N * sizeof(float));
    if (!h_in) { printf("host alloc failed\n"); return 1; }
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f + 1e-4f * (float)(i % 1000);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)N * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));

    // ---- duration-based clock warm-up (spec 12.4) -------------------
    {
        cudaEvent_t w0, w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
        float acc = 0.0f;
        while (acc < 400.0f) {
            CHECK(cudaEventRecord(w0));
            for (int k = 0; k < 20; ++k) groupKer<0><<<GRID, BLOCK>>>(d_in, d_out);
            CHECK(cudaEventRecord(w1));
            CHECK(cudaEventSynchronize(w1));
            float ms; CHECK(cudaEventElapsedTime(&ms, w0, w1)); acc += ms;
        }
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    }

    // ---- A) branch grouping, all four timed back-to-back ------------
    const int ITERS  = 20;
    const int SWEEPS = 6;
    float bestA[4] = { 1e30f, 1e30f, 1e30f, 1e30f };
    for (int s = 0; s < SWEEPS; ++s) {
        for (int m = 0; m < 4; ++m) {
            launchGroup(m, d_in, d_out);                 // warm-up launch
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < ITERS; ++it) launchGroup(m, d_in, d_out);
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaEventSynchronize(ev1));
            float ms; CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            ms /= (float)ITERS;
            if (ms < bestA[m]) bestA[m] = ms;
        }
    }
    CHECK(cudaGetLastError());

    // ---- B) trip-count divergence -----------------------------------
    float bestB[2] = { 1e30f, 1e30f };
    for (int s = 0; s < SWEEPS; ++s) {
        for (int m = 0; m < 2; ++m) {
            launchTrip(m, d_in, d_out);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < ITERS; ++it) launchTrip(m, d_in, d_out);
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaEventSynchronize(ev1));
            float ms; CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            ms /= (float)ITERS;
            if (ms < bestB[m]) bestB[m] = ms;
        }
    }
    CHECK(cudaGetLastError());

    // ---- validation, in a separate pass (spec 12.2) ------------------
    int fails = 0;
    {
        float* h_out = (float*)malloc((size_t)N * sizeof(float));
        if (!h_out) { printf("host alloc failed\n"); return 1; }
        groupKer<0><<<GRID, BLOCK>>>(d_in, d_out);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
        // CPU reference for MODE 0: every lane takes arm A. Done in
        // float with fmaf so it follows the device arithmetic exactly.
        int bad = 0;
        for (int i = 0; i < GRID * BLOCK; i += 4093) {
            float a = h_in[i];
            for (int r = 0; r < REP; ++r)
                for (int j = 0; j < ARMLEN; ++j) a = fmaf(a, 0.9999f, 0.0001f);
            if (!(fabsf(h_out[i] - a) <= 1e-5f * fmaxf(1.0f, fabsf(a)))) ++bad;
        }
        if (bad) { printf("VALIDATION FAILED on %d sampled elements\n", bad); ++fails; }
        free(h_out);
    }

    // ---- report ------------------------------------------------------
    const char* an[4] = { "uniform (all lanes arm A)",
                          "warp-uniform (tid<128)",
                          "lane-alternate (tid&1)",
                          "8-lane groups ((tid>>3)&1)" };
    printf("--- A) same work, same data, four groupings of the branch ---\n");
    printf("  %-30s %10s %10s\n", "configuration", "ms", "vs uniform");
    for (int m = 0; m < 4; ++m)
        printf("  %-30s %10.4f %9.3fx\n", an[m], bestA[m], bestA[m] / bestA[0]);
    printf("\n  Divergence is warp-local. Grouping the SAME split by whole\n"
           "  warps costs nothing; splitting inside a warp costs the sum of\n"
           "  both arms, and it costs that whether the split is 16/16 or\n"
           "  some other partition -- the number of divergent lanes does not\n"
           "  appear in the price.\n\n");

    printf("--- B) loop trip counts: the warp runs at the MAXIMUM ---\n");
    printf("  trip counts 1..8 x %d FFMAs; identical total work both ways.\n", TRIPUNIT);
    printf("  %-30s %10.4f ms\n", "trips = lane&7 (in-warp spread)",  bestB[0]);
    printf("  %-30s %10.4f ms\n", "trips = blockIdx&7 (warp-uniform)",    bestB[1]);
    printf("  ratio %.3fx   (model: max(trips)/mean(trips) = 8/4.5 = %.3f)\n",
           bestB[0] / bestB[1], 8.0 / 4.5);
    printf("\n  A warp cannot retire early for the lanes that finished; it\n"
           "  keeps issuing the loop body with a shrinking active mask until\n"
           "  its longest-running lane is done.\n\n");

    printf("--- C) predication flip point ---\n");
    printf("  This binary contains flipKer<N> for N = 1,2,3,4,5,6,7,8,12,16.\n"
           "  Disassemble and find the N where the two arms stop being\n"
           "  predicated instructions and become a real branch:\n");
    printf("    cuobjdump -sass example02.exe > example02.sass\n");
    printf("  Look for '@P0 FFMA' / '@!P0 FFMA' versus 'BSSY' + 'BRA' + 'BSYNC'.\n");
    printf("  The binary also contains guardTail / guardShort / guardLoad: the\n"
           "  Module 3 bounds guard in three shapes. Compare their SASS. The\n"
           "  guard is not always predicated, and body length is not the only\n"
           "  thing that decides it.\n\n");

    free(h_in);
    CHECK(cudaEventDestroy(ev0)); CHECK(cudaEventDestroy(ev1));
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    CHECK(cudaDeviceReset());
    printf("OVERALL: %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
