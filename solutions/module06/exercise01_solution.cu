// =====================================================================
// Module 6 / Exercise 1 : SOLUTION
//   Tiled 5-point stencil with a halo, measured against the Module 3
//   baseline.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
// RUN:    .\exercise01_solution.exe
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

static const double PEAK_BW = 432.0;

// Module 3 / Exercise 1 measured baselines, same kernel, same images.
static const double M3_SMALL_MS = 0.0107;   // 1021 x 733,   block (32,8)
static const double M3_BIG_MS   = 0.3213;   // 4093 x 3079,  block (32,8)

enum { PRED_BIG_WIN = 1, PRED_MODEST_WIN = 2, PRED_WASH = 3, PRED_LOSS = 4 };
static const int PREDICTION = PRED_LOSS;

__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------
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

// ---------------------------------------------------------------------
template <int TW, int TH>
__global__ void stencil5_tiled(const float* __restrict__ in,
                               float* __restrict__ out, int h, int w)
{
    const int SW = TW + 2;
    const int SH = TH + 2;
    __shared__ float tile[(TH + 2) * (TW + 2)];

    const int col0 = (int)(blockIdx.x * TW);
    const int row0 = (int)(blockIdx.y * TH);
    const int tid  = (int)(threadIdx.y * TW + threadIdx.x);
    const int nthr = TW * TH;

    // ---- TODO 1 : cooperative load, halo included -------------------
    // SW*SH cells, nthr threads, SW*SH > nthr, so this is a strided
    // loop and NOT one cell per thread.
    for (int idx = tid; idx < SW * SH; idx += nthr) {
        int ly = idx / SW;
        int lx = idx - ly * SW;
        int gy = clampi(row0 + ly - 1, 0, h - 1);
        int gx = clampi(col0 + lx - 1, 0, w - 1);
        tile[idx] = in[(size_t)gy * w + gx];
    }

    // ---- TODO 2 : barrier -------------------------------------------
    __syncthreads();

    // ---- TODO 3 : compute out of shared memory ----------------------
    int col = col0 + (int)threadIdx.x;
    int row = row0 + (int)threadIdx.y;
    if (row >= h || col >= w) return;

    const int lx = (int)threadIdx.x + 1;
    const int ly = (int)threadIdx.y + 1;

    float acc = 4.0f * tile[ly * SW + lx]
              +        tile[(ly - 1) * SW + lx]
              +        tile[(ly + 1) * SW + lx]
              +        tile[ly * SW + (lx - 1)]
              +        tile[ly * SW + (lx + 1)];
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
    { "naive (32,8)",  stencil5_naive,       32,  8,    0 },
    { "tiled 32x8",    stencil5_tiled<32,8>, 32,  8,  340 * sizeof(float) },
    { "tiled 32x16",   stencil5_tiled<32,16>,32, 16,  612 * sizeof(float) },
    { "tiled 64x8",    stencil5_tiled<64,8>, 64,  8,  660 * sizeof(float) },
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

    // -------- pass 1: timing only, all configs back to back ----------
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

    // -------- pass 2: correctness --------------------------------------
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
    printf("=== Module 6 / Exercise 1 : tiled 5-point stencil (SOLUTION) ===\n");

    int fails = 0;
    double r_small = 0.0, r_big = 0.0;
    fails += runSize(1021,  733, M3_SMALL_MS, &r_small);
    fails += runSize(4093, 3079, M3_BIG_MS,   &r_big);

    int measured = (r_big >= 1.50) ? PRED_BIG_WIN
                 : (r_big >= 1.05) ? PRED_MODEST_WIN
                 : (r_big >= 0.95) ? PRED_WASH
                                   : PRED_LOSS;
    static const char* NAMES[5] = { "", "a big win (>=1.50x)", "a modest win (1.05-1.50x)",
                                    "a wash (0.95-1.05x)", "a net loss (<0.95x)" };
    printf("\n--- prediction ---\n");
    printf("  measured tiled-vs-naive on 4093x3079 : %.3fx -> %s\n",
           r_big, NAMES[measured]);
    printf("  you predicted                        : %s\n",
           (PREDICTION >= 1 && PREDICTION <= 4) ? NAMES[PREDICTION] : "(unset)");
    int predOK = (PREDICTION == measured);
    printf("  prediction: %s\n", predOK ? "CORRECT" : "WRONG");

    printf("\nOVERALL: %s\n", (fails == 0 && predOK) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (fails == 0 && predOK) ? 0 : 1;
}
