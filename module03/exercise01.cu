// =====================================================================
// Module 3 / Exercise 1 : "2D stencil on an awkward image"
//
// GOAL
//   Map a 2D grid of threads onto a 2D array that is neither square nor
//   a multiple of the block size, and get every edge case right.
//
//   The computation is a 5-point weighted stencil with clamped
//   (edge-replicated) boundaries:
//
//       out[r][c] = ( 4*in[r][c]
//                   +   in[r-1][c] +   in[r+1][c]
//                   +   in[r][c-1] +   in[r][c+1] ) * 0.125f
//
//   with any out-of-image neighbour replaced by the nearest in-image
//   pixel (clamp). The image is H=1021 rows by W=733 columns, stored
//   row-major with row stride W. Neither dimension is a multiple of 32,
//   neither is a power of two, and H != W -- every partial-block edge
//   case fires, and an index scheme that only happens to work on square
//   power-of-two images will fail loudly.
//
//   Part B re-runs your kernel with four different block shapes that
//   all contain 256 threads and times them, at two image sizes: the
//   1021x733 one (2.9 MB -- fits entirely in this GPU's 48 MB L2) and a
//   4093x3079 one (50 MB per array -- does not fit, so the DRAM
//   transaction count is what you are measuring). Three of the four
//   shapes perform about the same; one is catastrophic. Predict which,
//   and why, before you run it.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN:    .\exercise01.exe
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

// Clamp helper, used by both the CPU reference and the kernel.
__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------
// The kernel.
//
// The array is row-major: element (r, c) lives at in[r * W + c].
// Consecutive c are adjacent in memory; consecutive r are W floats apart.
// ---------------------------------------------------------------------
__global__ void stencil5(const float* __restrict__ in,
                         float* __restrict__ out,
                         int h, int w)
{
    // -----------------------------------------------------------------
    // TODO 1: Compute this thread's global position in the image.
    //         Decide which of the grid's x/y axes maps to the row index
    //         and which maps to the column index, and justify the choice
    //         in terms of what the 32 threads of a single warp will read
    //         and write. Getting this backwards still produces a running
    //         kernel and a plausible-looking image.
    // -----------------------------------------------------------------
    int row = 0;   // YOUR CODE HERE (TODO 1)
    int col = 0;   // YOUR CODE HERE (TODO 1)

    // -----------------------------------------------------------------
    // TODO 2: Guard. The grid necessarily covers more positions than the
    //         image contains (1021 and 733 are not multiples of any
    //         block dimension you will pick). Write the condition that
    //         retires the threads with nothing to do.
    // -----------------------------------------------------------------
    // YOUR CODE HERE (TODO 2)

    int rm = clampi(row - 1, 0, h - 1);
    int rp = clampi(row + 1, 0, h - 1);
    int cm = clampi(col - 1, 0, w - 1);
    int cp = clampi(col + 1, 0, w - 1);

    float acc = 4.0f * in[row * w + col]
              +        in[rm  * w + col]
              +        in[rp  * w + col]
              +        in[row * w + cm]
              +        in[row * w + cp];

    out[row * w + col] = acc * 0.125f;
}

// ---------------------------------------------------------------------
// TODO 3: Given the image size and a block shape, return the grid shape
//         that covers the whole image with no position left out and as
//         few wasted threads as possible. It must be correct for ANY
//         block shape the harness passes in -- (32,8), (8,32) and
//         (256,1) are all used below -- and must stay correct if H or W
//         is changed to something much larger.
// ---------------------------------------------------------------------
static dim3 makeGrid(int h, int w, dim3 block)
{
    (void)h; (void)w; (void)block;
    return dim3(0, 0, 0);   // YOUR CODE HERE (TODO 3)
}

// ---------------------------------------------------------------------
static void cpuReference(const float* in, float* out, int h, int w)
{
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            int rm = clampi(r - 1, 0, h - 1);
            int rp = clampi(r + 1, 0, h - 1);
            int cm = clampi(c - 1, 0, w - 1);
            int cp = clampi(c + 1, 0, w - 1);
            out[r * w + c] = (4.0f * in[r * w + c]
                              + in[rm * w + c] + in[rp * w + c]
                              + in[r * w + cm] + in[r * w + cp]) * 0.125f;
        }
    }
}

static int compare(const float* got, const float* ref, int h, int w, const char* tag)
{
    long long bad = 0;
    int fr = -1, fc = -1;
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            float g = got[r * w + c], e = ref[r * w + c];
            if (!(fabsf(g - e) <= 1e-5f * fmaxf(1.0f, fabsf(e)))) {
                if (bad == 0) { fr = r; fc = c; }
                ++bad;
            }
        }
    if (bad == 0) {
        printf("  %-18s PASS  (all %d x %d = %d elements match)\n", tag, h, w, h * w);
        return 0;
    }
    printf("  %-18s FAIL  %lld / %d elements wrong; first at (row=%d, col=%d): got %.6f, want %.6f\n",
           tag, bad, h * w, fr, fc, got[fr * w + fc], ref[fr * w + fc]);
    return 1;
}

static int runSize(int H, int W)
{
    const size_t n     = (size_t)H * W;
    const size_t bytes = n * sizeof(float);

    float* h_in  = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    if (!h_in || !h_ref || !h_out) { printf("host alloc failed\n"); return 1; }

    // Deterministic, index-derived, and *asymmetric in r and c* so that a
    // transposed or mis-strided index scheme cannot accidentally agree.
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            h_in[(size_t)r * W + c] = 0.001f * (float)((r * 37 + c * 11) % 1000)
                                    + 0.5f   * (float)(r % 7)
                                    - 0.25f  * (float)(c % 13);

    cpuReference(h_in, h_ref, H, W);

    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("\n=== image %d x %d (row-major, stride %d, %.1f MB per array) ===\n",
           H, W, W, bytes / 1.0e6);

    const dim3 blocks[4] = { dim3(32, 8), dim3(8, 32), dim3(256, 1), dim3(1, 256) };
    const char* names[4] = { "block (32,8)", "block (8,32)", "block (256,1)", "block (1,256)" };

    int fails = 0;
    for (int k = 0; k < 4; ++k) {
        dim3 blk  = blocks[k];
        dim3 grid = makeGrid(H, W, blk);

        long long launched = (long long)grid.x * grid.y * grid.z
                           * blk.x * blk.y * blk.z;
        printf("\n--- %s : grid (%u,%u,%u), %lld threads for %zu pixels (%lld idle) ---\n",
               names[k], grid.x, grid.y, grid.z, launched, n, launched - (long long)n);
        if (launched < (long long)n)
            printf("  grid does not cover the image; some pixels can never be written.\n");

        CHECK(cudaMemset(d_out, 0, bytes));

        stencil5<<<grid, blk>>>(d_in, d_out, H, W);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        fails += compare(h_out, h_ref, H, W, names[k]);

        // Timing: warm-up + 20 iterations, cudaEvent_t only.
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
        stencil5<<<grid, blk>>>(d_in, d_out, H, W);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(t0));
        for (int it = 0; it < 20; ++it)
            stencil5<<<grid, blk>>>(d_in, d_out, H, W);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        CHECK(cudaGetLastError());
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, t0, t1)); ms /= 20.f;
        CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));

        // Compulsory traffic: every input pixel is read at least once and
        // every output pixel written once -> 8 B per pixel. Anything above
        // that is re-reads the caches failed to absorb.
        double gbs = 8.0 * (double)n / (ms * 1.0e-3) / 1.0e9;
        printf("  %-18s %8.4f ms   %7.1f GB/s effective   %5.1f%% of peak\n",
               names[k], ms, gbs, 100.0 * gbs / PEAK_BW);
    }

    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    free(h_in); free(h_ref); free(h_out);
    return fails;
}

int main(void)
{
    CHECK(cudaSetDevice(0));
    printf("=== Module 3 / Exercise 1 : 5-point stencil, 2D thread mapping ===\n");

    dim3 probe = makeGrid(1021, 733, dim3(32, 8));
    if (probe.x == 0 || probe.y == 0 || probe.z == 0) {
        printf("\nmakeGrid() returned a zero dimension. Fill in TODO 3 first.\n");
        CHECK(cudaDeviceReset());
        return 0;
    }

    int fails = 0;
    fails += runSize(1021, 733);     // 2.9 MB  -> fits in L2
    fails += runSize(4093, 3079);    // 50.4 MB -> does not fit in L2

    printf("\nOVERALL: %s\n", fails == 0 ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return fails == 0 ? 0 : 1;
}
