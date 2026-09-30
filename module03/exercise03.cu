// =====================================================================
// Module 3 / Exercise 3 : "Which threads are in warp 1?"  (predict first)
//
// GOAL
//   Commit, in writing, to exactly what one warp does -- before the
//   machine tells you. This is the skill Module 5 (coalescing) and
//   Module 8 (SIMT divergence) are built on; if you cannot name the 32
//   threads of a warp and the 32 addresses they touch, you cannot reason
//   about either.
//
// THE SETUP
//   A row-major float matrix, H = 256 rows by W = 512 columns. The
//   kernel is trivial on purpose:
//
//       col = blockIdx.x * blockDim.x + threadIdx.x;
//       row = blockIdx.y * blockDim.y + threadIdx.y;
//       out[row * W + col] = 2.0f * in[row * W + col] + 1.0f;
//
//   It is launched twice with the same 256 threads per block, shaped
//   two different ways:
//
//       config A : block (8, 32)
//       config B : block (32, 8)
//
//   Both produce identical, correct output. They do not produce
//   identical memory behaviour.
//
// WHAT YOU MUST PREDICT (fill in TODO 1..3 below, then run)
//   For BLOCK (0,0) only, and for WARP 1 of that block (the threads whose
//   linearized index within the block is 32..63):
//
//     * how many distinct threadIdx.y values that warp contains,
//     * how many distinct 128-byte-aligned segments of `out` its 32
//       stores fall into (a segment is bytes [128k, 128k+128); the
//       allocation is 256-byte aligned, so element i sits in segment
//       i/32),
//     * the lowest and highest element index of `out` it writes.
//
//   And for the whole 256-thread block (0,0): how many distinct
//   128-byte segments all 256 stores fall into, in each config.
//
//   Write your reasoning down before you run the program. The program
//   prints the truth, including the full lane-by-lane roster of warp 1.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN:    .\exercise03.exe
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

static const int H = 256;
static const int W = 512;
static const int SEG_ELEMS = 32;     // 128 bytes / sizeof(float)

// =====================================================================
//                      YOUR PREDICTIONS
// Leave a field as -1 if you refuse to guess; it will simply be reported
// as a mismatch. Do not run the program before filling these in.
// =====================================================================

// ---------------------------------------------------------------------
// TODO 1: config A, block (8,32). Warp 1 of block (0,0).
// ---------------------------------------------------------------------
static const int A_warp1_distinct_ty     = -1;   // YOUR CODE HERE (TODO 1)
static const int A_warp1_segments        = -1;   // YOUR CODE HERE (TODO 1)
static const int A_warp1_min_element     = -1;   // YOUR CODE HERE (TODO 1)
static const int A_warp1_max_element     = -1;   // YOUR CODE HERE (TODO 1)

// ---------------------------------------------------------------------
// TODO 2: config B, block (32,8). Warp 1 of block (0,0).
// ---------------------------------------------------------------------
static const int B_warp1_distinct_ty     = -1;   // YOUR CODE HERE (TODO 2)
static const int B_warp1_segments        = -1;   // YOUR CODE HERE (TODO 2)
static const int B_warp1_min_element     = -1;   // YOUR CODE HERE (TODO 2)
static const int B_warp1_max_element     = -1;   // YOUR CODE HERE (TODO 2)

// ---------------------------------------------------------------------
// TODO 3: whole block (0,0), all 256 threads -- distinct 128-byte
//         segments written, in each config.
// ---------------------------------------------------------------------
static const int A_block_segments        = -1;   // YOUR CODE HERE (TODO 3)
static const int B_block_segments        = -1;   // YOUR CODE HERE (TODO 3)

// =====================================================================

// The kernel. `trace` is written only by block (0,0): trace[t] is the
// element index of `out` written by the thread whose linearized index
// within the block is t. Nothing here is shared memory or a warp
// intrinsic -- every thread writes its own slot.
__global__ void scaleAndTrace(const float* __restrict__ in,
                              float* __restrict__ out,
                              int* __restrict__ traceElem,
                              int* __restrict__ traceTx,
                              int* __restrict__ traceTy,
                              int h, int w)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= h || col >= w) return;

    int idx = row * w + col;
    out[idx] = 2.0f * in[idx] + 1.0f;

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // The linearization rule. x fastest, then y, then z.
        int t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
        traceElem[t] = idx;
        traceTx[t]   = threadIdx.x;
        traceTy[t]   = threadIdx.y;
    }
}

// ---------------------------------------------------------------------
static int countDistinct(const int* v, int n)
{
    int d = 0;
    for (int i = 0; i < n; ++i) {
        int seen = 0;
        for (int j = 0; j < i; ++j) if (v[j] == v[i]) { seen = 1; break; }
        if (!seen) ++d;
    }
    return d;
}

static void checkPred(const char* label, int predicted, int actual, int* wrong)
{
    if (predicted == actual) {
        printf("    %-46s predicted %6d   actual %6d   MATCH\n", label, predicted, actual);
    } else {
        printf("    %-46s predicted %6d   actual %6d   MISMATCH\n", label, predicted, actual);
        ++*wrong;
    }
}

struct Truth { int distinct_ty, warp1_segments, min_elem, max_elem, block_segments; };

static Truth runConfig(const char* name, dim3 block,
                       const float* d_in, float* d_out,
                       const float* h_in, float* h_out, float* h_ref,
                       int* fails)
{
    const size_t n = (size_t)H * W;
    int tpb = (int)(block.x * block.y * block.z);

    dim3 grid((unsigned)(W / (int)block.x) + (unsigned)((W % (int)block.x) != 0),
              (unsigned)(H / (int)block.y) + (unsigned)((H % (int)block.y) != 0),
              1u);

    int *d_te, *d_tx, *d_ty;
    CHECK(cudaMalloc(&d_te, tpb * sizeof(int)));
    CHECK(cudaMalloc(&d_tx, tpb * sizeof(int)));
    CHECK(cudaMalloc(&d_ty, tpb * sizeof(int)));
    CHECK(cudaMemset(d_te, 0xFF, tpb * sizeof(int)));
    CHECK(cudaMemset(d_tx, 0xFF, tpb * sizeof(int)));
    CHECK(cudaMemset(d_ty, 0xFF, tpb * sizeof(int)));
    CHECK(cudaMemset(d_out, 0, n * sizeof(float)));

    scaleAndTrace<<<grid, block>>>(d_in, d_out, d_te, d_tx, d_ty, H, W);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    int* h_te = (int*)malloc(tpb * sizeof(int));
    int* h_tx = (int*)malloc(tpb * sizeof(int));
    int* h_ty = (int*)malloc(tpb * sizeof(int));
    CHECK(cudaMemcpy(h_te, d_te, tpb * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_tx, d_tx, tpb * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_ty, d_ty, tpb * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- numerical validation ------------------------------------
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        h_ref[i] = 2.0f * h_in[i] + 1.0f;
        if (!(fabsf(h_out[i] - h_ref[i]) <= 1e-5f * fmaxf(1.0f, fabsf(h_ref[i])))) ++bad;
    }

    printf("\n===== %s : block (%u,%u,%u), grid (%u,%u,%u) =====\n",
           name, block.x, block.y, block.z, grid.x, grid.y, grid.z);
    printf("  numerical result: %s (%zu mismatches of %zu)\n",
           bad == 0 ? "PASS" : "FAIL", bad, n);
    if (bad) ++*fails;

    // ---- lane roster of warp 1 -----------------------------------
    printf("  warp 1 of block (0,0) -- lane : (threadIdx.x, threadIdx.y) -> out element -> 128B segment\n");
    int seg[32], tyv[32], elem[32];
    for (int lane = 0; lane < 32; ++lane) {
        int t = 32 + lane;
        elem[lane] = h_te[t];
        tyv[lane]  = h_ty[t];
        seg[lane]  = h_te[t] / SEG_ELEMS;
        if (lane % 4 == 0) printf("   ");
        printf(" %2d:(%2d,%2d)->%6d/s%-5d", lane, h_tx[t], h_ty[t], h_te[t], seg[lane]);
        if (lane % 4 == 3) printf("\n");
    }

    Truth tr;
    tr.distinct_ty     = countDistinct(tyv, 32);
    tr.warp1_segments  = countDistinct(seg, 32);
    tr.min_elem = elem[0]; tr.max_elem = elem[0];
    for (int i = 1; i < 32; ++i) {
        if (elem[i] < tr.min_elem) tr.min_elem = elem[i];
        if (elem[i] > tr.max_elem) tr.max_elem = elem[i];
    }
    int* blkSeg = (int*)malloc(tpb * sizeof(int));
    for (int t = 0; t < tpb; ++t) blkSeg[t] = h_te[t] / SEG_ELEMS;
    tr.block_segments = countDistinct(blkSeg, tpb);
    free(blkSeg);

    printf("  warp 1 spans %d distinct row(s) of the matrix, %d distinct 128B segment(s),\n",
           tr.distinct_ty, tr.warp1_segments);
    printf("  element range [%d, %d] (span of %d elements for 32 stores)\n",
           tr.min_elem, tr.max_elem, tr.max_elem - tr.min_elem + 1);
    printf("  whole block (0,0) touches %d distinct 128B segments with %d stores\n",
           tr.block_segments, tpb);

    free(h_te); free(h_tx); free(h_ty);
    CHECK(cudaFree(d_te)); CHECK(cudaFree(d_tx)); CHECK(cudaFree(d_ty));
    return tr;
}

int main(void)
{
    CHECK(cudaSetDevice(0));
    const size_t n = (size_t)H * W;

    printf("=== Module 3 / Exercise 3 : warp membership and address footprint ===\n");
    printf("matrix %d x %d floats, row-major, row stride %d elements = %d bytes\n",
           H, W, W, (int)(W * sizeof(float)));
    printf("one 128-byte segment = %d consecutive floats\n", SEG_ELEMS);

    float* h_in  = (float*)malloc(n * sizeof(float));
    float* h_out = (float*)malloc(n * sizeof(float));
    float* h_ref = (float*)malloc(n * sizeof(float));
    if (!h_in || !h_out || !h_ref) { printf("host alloc failed\n"); return 1; }
    for (size_t i = 0; i < n; ++i) h_in[i] = 0.001f * (float)(i % 1000);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  n * sizeof(float)));
    CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    int fails = 0;
    Truth A = runConfig("config A", dim3(8, 32), d_in, d_out, h_in, h_out, h_ref, &fails);
    Truth Bt = runConfig("config B", dim3(32, 8), d_in, d_out, h_in, h_out, h_ref, &fails);

    int wrong = 0;
    printf("\n===== your predictions vs the machine =====\n");
    printf("  config A, block (8,32):\n");
    checkPred("warp 1: distinct threadIdx.y values",       A_warp1_distinct_ty, A.distinct_ty,    &wrong);
    checkPred("warp 1: distinct 128B segments",            A_warp1_segments,    A.warp1_segments, &wrong);
    checkPred("warp 1: lowest element index",              A_warp1_min_element, A.min_elem,       &wrong);
    checkPred("warp 1: highest element index",             A_warp1_max_element, A.max_elem,       &wrong);
    printf("  config B, block (32,8):\n");
    checkPred("warp 1: distinct threadIdx.y values",       B_warp1_distinct_ty, Bt.distinct_ty,    &wrong);
    checkPred("warp 1: distinct 128B segments",            B_warp1_segments,    Bt.warp1_segments, &wrong);
    checkPred("warp 1: lowest element index",              B_warp1_min_element, Bt.min_elem,       &wrong);
    checkPred("warp 1: highest element index",             B_warp1_max_element, Bt.max_elem,       &wrong);
    printf("  whole block (0,0):\n");
    checkPred("config A: distinct 128B segments / block",  A_block_segments,    A.block_segments,  &wrong);
    checkPred("config B: distinct 128B segments / block",  B_block_segments,    Bt.block_segments, &wrong);

    printf("\n  predictions correct: %d / 10\n", 10 - wrong);
    printf("\nOVERALL: %s\n", (fails == 0 && wrong == 0) ? "PASS" : "FAIL");

    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);
    CHECK(cudaDeviceReset());
    return (fails == 0 && wrong == 0) ? 0 : 1;
}
