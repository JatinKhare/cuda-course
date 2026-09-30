// =====================================================================
// Module 6 / Exercise 1 : "Tile the stencil you already wrote"
//
// GOAL
//   Take the 5-point clamped stencil from Module 3 / Exercise 1 --
//   same formula, same two images, same block shape -- and write a
//   version that stages each block's input tile in shared memory
//   first, then computes entirely out of shared memory.
//
//       out[r][c] = ( 4*in[r][c]
//                   +   in[r-1][c] +   in[r+1][c]
//                   +   in[r][c-1] +   in[r][c+1] ) * 0.125f
//
//   with out-of-image neighbours clamped to the nearest in-image pixel.
//
//   A block computes a TW x TH patch of output. To do that it needs a
//   (TW+2) x (TH+2) patch of input: the interior plus a one-cell HALO
//   (also called ghost cells) on all four sides. That is the whole
//   difficulty of this exercise. The block has TW*TH threads and must
//   load (TW+2)*(TH+2) cells, and those two numbers are not equal, so
//   the load cannot be one cell per thread.
//
//   Note carefully that there are TWO different index mappings in a
//   tiled kernel:
//       the LOAD mapping    : thread -> which tile cell(s) it fetches
//       the COMPUTE mapping : thread -> which output pixel it writes
//   They are not the same function and writing one where you meant the
//   other is the classic tiling bug.
//
//   The harness times your tiled kernel at three tile shapes against an
//   untiled kernel timed in the same sweep, and prints the Module 3
//   measured baseline next to it.
//
// WHAT TO PREDICT (TODO 4, before you run anything)
//   Each input pixel is read by up to 5 threads, so the traffic model
//   says tiling could cut global reads by up to 5x. Module 3 measured
//   the untiled kernel at 0.3213 ms on the 4093x3079 image, which is
//   313.8 GB/s of useful traffic against a 432 GB/s DRAM peak. Decide
//   what tiling will actually buy, and commit to it in TODO 4 before
//   you build. The harness scores it, and OVERALL cannot be PASS
//   unless your prediction is right.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN:    .\exercise01.exe
//
// Useful while debugging:
//   nvcc -arch=sm_89 -O3 -lineinfo -o exercise01.exe exercise01.cu
//   compute-sanitizer --tool memcheck  .\exercise01.exe
//   compute-sanitizer --tool racecheck .\exercise01.exe
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

static const double PEAK_BW = 432.0;   // GB/s on this GPU

// Module 3 / Exercise 1 measured baselines: same kernel, same images,
// block (32,8), on this GPU.
static const double M3_SMALL_MS = 0.0107;   // 1021 x 733
static const double M3_BIG_MS   = 0.3213;   // 4093 x 3079

// ---------------------------------------------------------------------
// TODO 4: Your prediction, committed before you build.
//
//   Set PREDICTION to exactly one of PRED_BIG_WIN, PRED_MODEST_WIN,
//   PRED_WASH, PRED_LOSS, describing what the tiled kernel will do to
//   the *4093 x 3079* runtime relative to the untiled kernel timed in
//   the same sweep.
//
//     PRED_BIG_WIN    tiled is >= 1.50x faster
//     PRED_MODEST_WIN tiled is 1.05x - 1.50x faster
//     PRED_WASH       within +/- 5%
//     PRED_LOSS       tiled is slower: ratio < 0.95x
//
//   Write down your reasoning too -- you will want it later.
//   The file will not run until this is set.
// ---------------------------------------------------------------------
enum { PRED_UNSET = 0, PRED_BIG_WIN = 1, PRED_MODEST_WIN = 2,
       PRED_WASH = 3, PRED_LOSS = 4 };

static const int PREDICTION = PRED_UNSET;   // YOUR CODE HERE (TODO 4)

// ---------------------------------------------------------------------
__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// The untiled kernel, unchanged from Module 3. Do not modify it: it is
// the control in this experiment.
__global__ void stencil5_naive(const float* __restrict__ in,
                               float* __restrict__ out, int h, int w)
{
    int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= h || col >= w) return;

    int rm = clampi(row - 1, 0, h - 1), rp = clampi(row + 1, 0, h - 1);
    int cm = clampi(col - 1, 0, w - 1), cp = clampi(col + 1, 0, w - 1);

    float acc = 4.0f * in[(size_t)row * w + col]
              +        in[(size_t)rm  * w + col]
              +        in[(size_t)rp  * w + col]
              +        in[(size_t)row * w + cm]
              +        in[(size_t)row * w + cp];
    out[(size_t)row * w + col] = acc * 0.125f;
}

// =====================================================================
// The tiled kernel. Launched with blockDim = (TW, TH) and a grid that
// covers the image in TW x TH interior patches.
//
// `tile` is indexed as tile[ly * SW + lx] with 0 <= lx < SW and
// 0 <= ly < SH. Tile cell (ly, lx) corresponds to image pixel
// (row0 + ly - 1, col0 + lx - 1), clamped to the image.
// =====================================================================
template <int TW, int TH>
__global__ void stencil5_tiled(const float* __restrict__ in,
                               float* __restrict__ out, int h, int w)
{
    const int SW = TW + 2;                  // tile width  incl. halo
    const int SH = TH + 2;                  // tile height incl. halo
    __shared__ float tile[(TH + 2) * (TW + 2)];

    const int col0 = (int)(blockIdx.x * TW); // image column of tile[*][1]
    const int row0 = (int)(blockIdx.y * TH); // image row    of tile[1][*]

    // Flat thread id within the block, and how many threads there are.
    const int tid  = (int)(threadIdx.y * TW + threadIdx.x);
    const int nthr = TW * TH;

    // -----------------------------------------------------------------
    // TODO 1: Cooperatively load the whole (SW x SH) tile, halo
    //         included, from `in` into `tile`.
    //
    //         There are SW*SH cells and only nthr threads, and
    //         SW*SH > nthr, so some threads must fetch more than one
    //         cell. Write a loop that covers every cell exactly once,
    //         for every one of the three (TW,TH) shapes the harness
    //         uses below, with no thread reading out of the image.
    //
    //         Out-of-image halo cells must take the value of the
    //         nearest in-image pixel -- the same clamp the formula
    //         uses. Remember that a tile at the right or bottom edge of
    //         the image may also have *interior* cells outside the
    //         image, because the image dimensions are not multiples of
    //         TW or TH.
    //
    //         Think about what 32 consecutive lanes of one warp read on
    //         each iteration of your loop. One obvious way to write
    //         this loop makes the loads uncoalesced; Module 5 gave you
    //         the tool to tell which.
    // -----------------------------------------------------------------
    // YOUR CODE HERE (TODO 1)
    // (this line only keeps the file warning-clean while TODO 1 is
    //  empty; delete it once you have written the loop)
    (void)SW; (void)SH; (void)tid; (void)nthr;

    // -----------------------------------------------------------------
    // TODO 2: Every thread is about to read tile cells that other
    //         threads wrote. Make sure that is safe.
    //         State in a comment which threads wrote the cells that
    //         thread (0,0) reads.
    // -----------------------------------------------------------------
    // YOUR CODE HERE (TODO 2)

    int col = col0 + (int)threadIdx.x;
    int row = row0 + (int)threadIdx.y;
    if (row >= h || col >= w) return;

    // -----------------------------------------------------------------
    // TODO 3: Compute the stencil for pixel (row, col) reading only
    //         from `tile`. No access to `in` is allowed below this
    //         line. You will need this thread's position *within the
    //         tile*, which is not the same as its position within the
    //         image and not the same as threadIdx either.
    // -----------------------------------------------------------------
    float acc = tile[0];   // YOUR CODE HERE (TODO 3)
                           // (tile[0] is only a placeholder that keeps the
                           //  file compiling and warning-clean; replace it)

    out[(size_t)row * w + col] = acc * 0.125f;
}

// ---------------------------------------------------------------------
static void cpuReference(const float* in, float* out, int h, int w)
{
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            int rm = clampi(r - 1, 0, h - 1), rp = clampi(r + 1, 0, h - 1);
            int cm = clampi(c - 1, 0, w - 1), cp = clampi(c + 1, 0, w - 1);
            out[(size_t)r * w + c] = (4.0f * in[(size_t)r * w + c]
                                      + in[(size_t)rm * w + c] + in[(size_t)rp * w + c]
                                      + in[(size_t)r * w + cm] + in[(size_t)r * w + cp])
                                     * 0.125f;
        }
}

typedef void (*Kern)(const float*, float*, int, int);
struct Cfg { const char* name; Kern k; int tw, th; size_t smem; };

static const Cfg CFG[] = {
    { "naive (32,8)",  stencil5_naive,        32,  8,    0 },
    { "tiled 32x8",    stencil5_tiled<32, 8>, 32,  8,  340 * sizeof(float) },
    { "tiled 32x16",   stencil5_tiled<32,16>, 32, 16,  612 * sizeof(float) },
    { "tiled 64x8",    stencil5_tiled<64, 8>, 64,  8,  660 * sizeof(float) },
};
static const int NC = (int)(sizeof(CFG) / sizeof(CFG[0]));

static void launch(int i, const float* d_in, float* d_out, int h, int w)
{
    dim3 blk((unsigned)CFG[i].tw, (unsigned)CFG[i].th);
    dim3 grd((unsigned)((w + CFG[i].tw - 1) / CFG[i].tw),
             (unsigned)((h + CFG[i].th - 1) / CFG[i].th));
    CFG[i].k<<<grd, blk>>>(d_in, d_out, h, w);
}

// Duration-based clock warm-up. This is a laptop part whose SM clock
// swings roughly 0.49-2.04 GHz; a fixed iteration count is not enough,
// the GPU has to be kept busy for a few hundred milliseconds.
static void warmClocks(const float* d_in, float* d_out, int h, int w)
{
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    float ms = 0.0f;
    CHECK(cudaEventRecord(a));
    do {
        for (int i = 0; i < 50; ++i) launch(0, d_in, d_out, h, w);
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        CHECK(cudaEventElapsedTime(&ms, a, b));
    } while (ms < 400.0f);
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
}

// ---------------------------------------------------------------------
static int runSize(int H, int W, double m3ms, double* ratio_out)
{
    const size_t n = (size_t)H * W, bytes = n * sizeof(float);
    float* h_in  = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_got = (float*)malloc(bytes);
    if (!h_in || !h_ref || !h_got) { printf("host alloc failed\n"); return 1; }

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            h_in[(size_t)r * W + c] = 0.001f * (float)((r * 37 + c * 11) % 1000)
                                    + 0.5f * (float)(r % 7) - 0.25f * (float)(c % 13);
    cpuReference(h_in, h_ref, H, W);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes)); CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("\n=== image %d x %d (%.1f MB per array) ===\n", H, W, bytes / 1.0e6);

    // ---- pass 1: timing only. No validation, no printing, no
    //      allocation between configurations -- see the methodology
    //      note in the lesson.
    double best[NC];
    for (int i = 0; i < NC; ++i) best[i] = 1e30;

    warmClocks(d_in, d_out, H, W);

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));

    // Choose the iteration count so that each timed segment lasts about
    // 10 ms. Twenty iterations of a 0.01 ms kernel is 0.2 ms of work,
    // which is short enough for the clock to sag between segments and
    // is the difference between a believable ratio and nonsense.
    int iters;
    {
        float ms = 0.f;
        launch(0, d_in, d_out, H, W);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(t0));
        for (int it = 0; it < 20; ++it) launch(0, d_in, d_out, H, W);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        iters = (int)(200.0 / (double)ms * 20.0);
        if (iters < 20)   iters = 20;
        if (iters > 4000) iters = 4000;
    }

    for (int sweep = 0; sweep < 4; ++sweep)
        for (int i = 0; i < NC; ++i) {
            launch(i, d_in, d_out, H, W);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(t0));
            for (int it = 0; it < iters; ++it) launch(i, d_in, d_out, H, W);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, t0, t1));
            if (ms / iters < best[i]) best[i] = ms / iters;
        }
    CHECK(cudaGetLastError());
    CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));

    // ---- pass 2: correctness
    int fails = 0;
    for (int i = 0; i < NC; ++i) {
        CHECK(cudaMemset(d_out, 0, bytes));
        launch(i, d_in, d_out, H, W);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_got, d_out, bytes, cudaMemcpyDeviceToHost));
        long long bad = 0; size_t first = 0;
        for (size_t j = 0; j < n; ++j)
            if (!(fabsf(h_got[j] - h_ref[j]) <= 1e-5f * fmaxf(1.0f, fabsf(h_ref[j])))) {
                if (!bad) first = j;
                ++bad;
            }
        if (bad) {
            printf("  %-14s FAIL  %lld/%zu wrong; first at (row %zu, col %zu): got %.6f want %.6f\n",
                   CFG[i].name, bad, n, first / W, first % W, h_got[first], h_ref[first]);
            ++fails;
        }
    }

    printf("  %-14s %10s %11s %9s %10s %12s\n",
           "config", "ms", "GB/s eff", "%peak", "smem/blk", "vs naive");
    for (int i = 0; i < NC; ++i) {
        double gbs = 8.0 * (double)n / (best[i] * 1.0e-3) / 1.0e9;
        printf("  %-14s %10.4f %11.1f %8.1f%% %9zu B %11.3fx\n",
               CFG[i].name, best[i], gbs, 100.0 * gbs / PEAK_BW,
               CFG[i].smem, best[0] / best[i]);
    }
    printf("  %-14s %10.4f   <- Module 3 / Exercise 1, same kernel, same image\n",
           "M3 baseline", m3ms);

    *ratio_out = best[0] / best[1];

    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    free(h_in); free(h_ref); free(h_got);
    return fails;
}

// ---------------------------------------------------------------------
int main(void)
{
    CHECK(cudaSetDevice(0));
    printf("=== Module 6 / Exercise 1 : tiled 5-point stencil ===\n");

    if (PREDICTION == PRED_UNSET) {
        printf("\nSet TODO 4 (PREDICTION) first.\n");
        CHECK(cudaDeviceReset());
        return 0;
    }

    int fails = 0;
    double r_small = 0.0, r_big = 0.0;
    fails += runSize(1021,  733, M3_SMALL_MS, &r_small);
    fails += runSize(4093, 3079, M3_BIG_MS,   &r_big);

    int measured = (r_big >= 1.50) ? PRED_BIG_WIN
                 : (r_big >= 1.05) ? PRED_MODEST_WIN
                 : (r_big >= 0.95) ? PRED_WASH
                                   : PRED_LOSS;
    static const char* NAMES[5] = { "(unset)", "a big win (>=1.50x)",
                                    "a modest win (1.05-1.50x)",
                                    "a wash (0.95-1.05x)", "a net loss (<0.95x)" };
    printf("\n--- prediction ---\n");
    printf("  measured tiled-vs-naive on 4093x3079 : %.3fx -> %s\n",
           r_big, NAMES[measured]);
    printf("  you predicted                        : %s\n", NAMES[PREDICTION]);
    int predOK = (PREDICTION == measured);
    printf("  prediction: %s\n", predOK ? "CORRECT" : "WRONG");

    printf("\nOVERALL: %s\n", (fails == 0 && predOK) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (fails == 0 && predOK) ? 0 : 1;
}
