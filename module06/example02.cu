// =====================================================================
// Module 6 / Example 2 : "What tiling actually buys -- a traffic ledger"
//
// A 2D box filter of radius R over a row-major image, clamped at the
// edges. Two implementations, four radii:
//
//   naive : every thread reads its own (2R+1)^2 neighbourhood straight
//           from global memory.  K = (2R+1)^2 reads per output pixel.
//   tiled : the block cooperatively stages one (TW+2R) x (TH+2R) tile
//           in shared memory, barriers, then every thread reads its
//           neighbourhood out of shared memory.
//
// The point of the example is the arithmetic, not the kernel:
//
//   reads per output, naive  = (2R+1)^2
//   reads per output, tiled  = (TW+2R)(TH+2R) / (TW*TH)      <- halo tax
//   ideal traffic reduction  = the ratio of those two
//
// at TW=32, TH=8:
//
//   R | K = (2R+1)^2 | tile/interior | ideal reduction
//   --+--------------+---------------+-----------------
//   1 |            9 | 340/256=1.33  |  6.8x
//   2 |           25 | 432/256=1.69  | 14.8x
//   3 |           49 | 532/256=2.08  | 23.6x
//   4 |           81 | 640/256=2.50  | 32.4x
//
// The program measures what that is worth in wall-clock time. The gap
// between "ideal traffic reduction" and "measured speedup" is the
// lesson: L1 already captured most of the reuse, so the reduction in
// *DRAM* traffic is nowhere near K.
//
// Timing follows the methodology for this laptop GPU: every
// configuration is timed back to back inside one sweep, four sweeps are
// run, the minimum is kept, and validation happens in a separate pass
// afterwards.
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

static const double PEAK_BW = 432.0;   // GB/s

#define TW 32
#define TH 8

__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------
template <int R>
__global__ void box_naive(const float* __restrict__ in,
                          float* __restrict__ out, int h, int w)
{
    int col = (int)(blockIdx.x * TW + threadIdx.x);
    int row = (int)(blockIdx.y * TH + threadIdx.y);
    if (row >= h || col >= w) return;

    // The column clamps depend only on dx, so hoist them: with R a
    // compile-time constant both loops unroll fully and cc[] stays in
    // registers. Without this the kernel is bound by integer min/max,
    // not by memory, and the measurement below would mean nothing.
    int cc[2 * R + 1];
#pragma unroll
    for (int dx = -R; dx <= R; ++dx) cc[dx + R] = clampi(col + dx, 0, w - 1);

    float acc = 0.0f;
#pragma unroll
    for (int dy = -R; dy <= R; ++dy) {
        const float* p = in + (size_t)clampi(row + dy, 0, h - 1) * w;
#pragma unroll
        for (int dx = 0; dx < 2 * R + 1; ++dx) acc += p[cc[dx]];
    }
    out[(size_t)row * w + col] = acc * (1.0f / ((2 * R + 1) * (2 * R + 1)));
}

// ---------------------------------------------------------------------
// The tiled version. Two index mappings live in this kernel and they
// are NOT the same:
//
//   the LOAD mapping   covers (TW+2R)(TH+2R) tile cells with TW*TH
//                      threads, so each thread loads 1, 2 or 3 cells;
//   the COMPUTE mapping is 1 thread : 1 output pixel.
//
// Conflating them is the classic tiling bug: it leaves the halo
// uninitialised.
// ---------------------------------------------------------------------
template <int R>
__global__ void box_tiled(const float* __restrict__ in,
                          float* __restrict__ out, int h, int w)
{
    const int SW = TW + 2 * R;
    const int SH = TH + 2 * R;
    __shared__ float tile[(TH + 2 * R) * (TW + 2 * R)];

    const int col0 = (int)(blockIdx.x * TW);     // interior origin
    const int row0 = (int)(blockIdx.y * TH);

    const int tid  = (int)(threadIdx.y * TW + threadIdx.x);
    const int nthr = TW * TH;

    // ---- cooperative load: flat, strided, NOT one cell per thread ----
    for (int idx = tid; idx < SW * SH; idx += nthr) {
        int ly = idx / SW;
        int lx = idx - ly * SW;
        int gy = clampi(row0 + ly - R, 0, h - 1);
        int gx = clampi(col0 + lx - R, 0, w - 1);
        tile[idx] = in[(size_t)gy * w + gx];
    }

    __syncthreads();      // barrier; Module 9 makes this precise

    // ---- compute: 1 thread : 1 pixel, indices shifted by R ----------
    int col = col0 + (int)threadIdx.x;
    int row = row0 + (int)threadIdx.y;
    if (row >= h || col >= w) return;

    const int lx = (int)threadIdx.x + R;
    const int ly = (int)threadIdx.y + R;

    float acc = 0.0f;
#pragma unroll
    for (int dy = -R; dy <= R; ++dy)
#pragma unroll
        for (int dx = -R; dx <= R; ++dx)
            acc += tile[(ly + dy) * SW + (lx + dx)];

    out[(size_t)row * w + col] = acc * (1.0f / ((2 * R + 1) * (2 * R + 1)));
}

// ---------------------------------------------------------------------
static void cpu_box(const float* in, float* out, int h, int w, int R)
{
    const float scale = 1.0f / (float)((2 * R + 1) * (2 * R + 1));
    for (int row = 0; row < h; ++row)
        for (int col = 0; col < w; ++col) {
            float acc = 0.0f;
            for (int dy = -R; dy <= R; ++dy)
                for (int dx = -R; dx <= R; ++dx)
                    acc += in[(size_t)clampi(row + dy, 0, h - 1) * w
                              + clampi(col + dx, 0, w - 1)];
            out[(size_t)row * w + col] = acc * scale;
        }
}

typedef void (*Kern)(const float*, float*, int, int);

struct Cfg { const char* name; int R; Kern k; int tiled; };

static void launch(const Cfg& c, const float* d_in, float* d_out, int h, int w)
{
    dim3 blk(TW, TH);
    dim3 grd((unsigned)((w + TW - 1) / TW), (unsigned)((h + TH - 1) / TH));
    c.k<<<grd, blk>>>(d_in, d_out, h, w);
}

// ---------------------------------------------------------------------
int main(void)
{
    CHECK(cudaSetDevice(0));
    printf("=== Module 6 / Example 2 : the tiling traffic ledger ===\n");
    printf("tile interior %dx%d = %d threads per block\n\n", TW, TH, TW * TH);

    const Cfg cfgs[] = {
        { "R=1 naive", 1, box_naive<1>, 0 }, { "R=1 tiled", 1, box_tiled<1>, 1 },
        { "R=2 naive", 2, box_naive<2>, 0 }, { "R=2 tiled", 2, box_tiled<2>, 1 },
        { "R=3 naive", 3, box_naive<3>, 0 }, { "R=3 tiled", 3, box_tiled<3>, 1 },
        { "R=4 naive", 4, box_naive<4>, 0 }, { "R=4 tiled", 4, box_tiled<4>, 1 },
    };
    const int NC = (int)(sizeof(cfgs) / sizeof(cfgs[0]));

    // ---------------- pass 1: correctness, on a small image -----------
    printf("--- correctness (1021 x 733) ---\n");
    int fails = 0;
    {
        const int H = 1021, W = 733;
        const size_t n = (size_t)H * W, bytes = n * sizeof(float);
        float* h_in  = (float*)malloc(bytes);
        float* h_ref = (float*)malloc(bytes);
        float* h_got = (float*)malloc(bytes);
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                h_in[(size_t)r * W + c] = 0.001f * (float)((r * 37 + c * 11) % 1000)
                                        + 0.5f * (float)(r % 7);

        float *d_in, *d_out;
        CHECK(cudaMalloc(&d_in, bytes)); CHECK(cudaMalloc(&d_out, bytes));
        CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

        for (int i = 0; i < NC; ++i) {
            if (!cfgs[i].tiled) cpu_box(h_in, h_ref, H, W, cfgs[i].R);
            CHECK(cudaMemset(d_out, 0, bytes));
            launch(cfgs[i], d_in, d_out, H, W);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_got, d_out, bytes, cudaMemcpyDeviceToHost));
            long long bad = 0;
            for (size_t j = 0; j < n; ++j)
                if (!(fabsf(h_got[j] - h_ref[j]) <= 1e-5f * fmaxf(1.0f, fabsf(h_ref[j]))))
                    ++bad;
            printf("  %-10s %s\n", cfgs[i].name, bad == 0 ? "PASS" : "FAIL");
            fails += (bad != 0);
        }
        CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
        free(h_in); free(h_ref); free(h_got);
    }

    // ---------------- pass 2: timing, on a 50 MB image ----------------
    const int H = 4093, W = 3079;
    const size_t n = (size_t)H * W, bytes = n * sizeof(float);
    printf("\n--- timing (%d x %d, %.1f MB per array, > 48 MB L2) ---\n",
           H, W, bytes / 1.0e6);

    float* h_in = (float*)malloc(bytes);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            h_in[(size_t)r * W + c] = 0.001f * (float)((r * 37 + c * 11) % 1000);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes)); CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));

    double best[8];
    for (int i = 0; i < NC; ++i) best[i] = 1e30;

    // Duration-based clock warm-up: keep the SM busy for a few hundred
    // milliseconds so the laptop part leaves its low P-state before
    // anything is recorded. A fixed iteration count is not enough.
    {
        float ms = 0.f;
        CHECK(cudaEventRecord(t0));
        do {
            for (int i = 0; i < 20; ++i) launch(cfgs[0], d_in, d_out, H, W);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            CHECK(cudaEventElapsedTime(&ms, t0, t1));
        } while (ms < 400.0f);
        CHECK(cudaGetLastError());
    }

    for (int sweep = 0; sweep < 4; ++sweep) {
        for (int i = 0; i < NC; ++i) {
            launch(cfgs[i], d_in, d_out, H, W);            // warm-up launch
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(t0));
            for (int it = 0; it < 20; ++it) launch(cfgs[i], d_in, d_out, H, W);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, t0, t1));
            double per = ms / 20.0;
            if (per < best[i]) best[i] = per;
        }
    }
    CHECK(cudaGetLastError());

    printf("  %-10s %9s %11s %9s %10s %10s %9s\n",
           "config", "ms", "GB/s eff", "%peak", "K reads", "tiled rd", "ideal x");
    for (int i = 0; i < NC; ++i) {
        int R = cfgs[i].R;
        double K     = (2.0 * R + 1) * (2.0 * R + 1);
        double tiler = (double)((TW + 2 * R) * (TH + 2 * R)) / (TW * TH);
        double gbs   = 8.0 * (double)n / (best[i] * 1.0e-3) / 1.0e9;
        printf("  %-10s %9.4f %11.1f %8.1f%% %10.0f %10.2f %8.1fx\n",
               cfgs[i].name, best[i], gbs, 100.0 * gbs / PEAK_BW,
               K, cfgs[i].tiled ? tiler : K, K / tiler);
    }

    printf("\n  %-6s %12s %12s %10s %14s\n",
           "R", "naive ms", "tiled ms", "speedup", "ideal traffic x");
    for (int i = 0; i < NC; i += 2) {
        int R = cfgs[i].R;
        double K     = (2.0 * R + 1) * (2.0 * R + 1);
        double tiler = (double)((TW + 2 * R) * (TH + 2 * R)) / (TW * TH);
        printf("  %-6d %12.4f %12.4f %9.2fx %13.1fx\n",
               R, best[i], best[i + 1], best[i] / best[i + 1], K / tiler);
    }
    {
        double eff1  = 8.0 * (double)n / (best[0] * 1.0e-3) / 1.0e9;
        double frac1 = eff1 / PEAK_BW;
        printf("\n  Why the measured column looks nothing like the ideal column:\n\n");
        printf("  At R=1 the NAIVE kernel already moves %.1f GB/s of useful data,\n"
               "  %.0f%% of the %.0f GB/s DRAM peak. Useful traffic is 8 B/pixel\n"
               "  (one read + one write), so its *actual* DRAM traffic can be at\n"
               "  most %.2fx the compulsory minimum. The traffic model predicted\n"
               "  9x. L1 and L2 had already collapsed 9 reads per pixel down to\n"
               "  about %.1f -- they delivered %.0f%% of the reduction that tiling\n"
               "  was supposed to deliver, for free, before you wrote a line of\n"
               "  shared-memory code.\n\n",
               eff1, 100.0 * frac1, PEAK_BW, 1.0 / frac1, 1.0 / frac1,
               100.0 * (9.0 - 1.0 / frac1) / (9.0 - 340.0 / 256.0));
        printf("  A kernel at %.0f%% of DRAM peak has a hard speed ceiling of\n"
               "  %.2fx no matter what you do on-chip. Tiling cannot beat that,\n"
               "  and at R=1 it does not even reach 1.0x: the halo tax is 33%% more\n"
               "  loads, plus a barrier, plus an extra %zu B of shared memory per\n"
               "  block, in exchange for turning L1 hits into LDS.\n\n",
               100.0 * frac1, 1.0 / frac1, (size_t)(340 * sizeof(float)));
        printf("  By R=4 the naive kernel is down to %.0f%% of peak -- it is no\n"
               "  longer DRAM-bound, it is bound by how many load instructions the\n"
               "  L1 can retire -- and tiling finally shows a profit, because LDS\n"
               "  and LDG contend for different resources.\n\n",
               100.0 * (8.0 * (double)n / (best[6] * 1.0e-3) / 1.0e9) / PEAK_BW);
        printf("  The rule this gives you: tiling pays when reuse K is large AND\n"
               "  the naive version is not already saturating something else.\n"
               "  Modules 15-17 apply it to a problem where K is in the hundreds\n"
               "  and the naive version is nowhere near any ceiling.\n");
    }

    CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); free(h_in);

    printf("\nOVERALL: %s\n", fails == 0 ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return fails == 0 ? 0 : 1;
}
