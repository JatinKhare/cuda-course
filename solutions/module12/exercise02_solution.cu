// ============================================================================
// Module 12 / exercise02_solution.cu -- a reduction that is not a sum (SOLVED)
//
// GOAL : Segmented reduction over ragged rows with a NON-COMMUTATIVE operator.
//
//        100,000 rows are packed end to end in one flat array with a CSR-style
//        offset table. Row lengths are wildly skewed: most rows hold a few
//        dozen elements, a few thousand hold several thousand, and ten hold
//        200,000. For every row you must produce two numbers:
//
//          sum  -- the sum of the row's elements
//          best -- the length of the LONGEST RUN of consecutive elements
//                  strictly greater than THR
//
//        `sum` is the reduction you already know. `best` is not: you cannot
//        compute it from the per-element values alone, and swapping two
//        operands of its combine changes the answer. That is the point of the
//        exercise. A reduction is defined by an associative operator over a
//        monoid; the sum version is the least interesting member of the family
//        and it hides every structural decision behind commutativity.
//
//        A correct, complete, one-thread-per-row baseline is supplied. It is
//        slow for reasons Module 8 taught you. Beat it.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
// RUN  : .\exercise02_solution.exe
//
// PASS : every row's `best` exactly right, every row's `sum` within 1e-4
//        relative, a speedup over the baseline of at least 3.0x, and both
//        predictions correct. SCORE: 5/5.
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

static const int   NROWS = 100000;
static const float THR   = -0.5f;     // note: 0.0f IS above this threshold

// ---------------------------------------------------------------------------
// The monoid. `Seg` summarises a CONTIGUOUS span of one row:
//
//   sum   sum of the span's elements
//   len   how many elements the span holds
//   pre   length of the run of above-threshold elements at the span's START
//   suf   length of the run of above-threshold elements at the span's END
//   best  length of the longest run anywhere in the span
//
// A span of length `len` whose elements are all above threshold has
// pre == suf == best == len. That case is what makes the operator interesting.
// ---------------------------------------------------------------------------
struct Seg { float sum; int len, pre, suf, best; };

__host__ __device__ __forceinline__ Seg segLeaf(float x, float thr)
{
    Seg s;
    s.sum = x;
    s.len = 1;
    int a = (x > thr) ? 1 : 0;
    s.pre = a; s.suf = a; s.best = a;
    return s;
}

// ===========================================================================
// TODO 1 -- the monoid: the identity element and the combine.
//
//   combineSeg(a, b) must summarise the span "a immediately followed by b".
//   It must be ASSOCIATIVE. It is NOT commutative, and it must not be: a run
//   that ends at the last element of `a` and continues into the first element
//   of `b` is one run, and no amount of reordering will find it.
//
//   segIdentity() must satisfy combineSeg(segIdentity(), x) == x and
//   combineSeg(x, segIdentity()) == x for every x. Check both directions on
//   paper before you write any kernel; a wrong identity is the single most
//   common way this exercise fails, and it fails only on the rows where the
//   padding lands, which is not all of them.
// ===========================================================================
__host__ __device__ __forceinline__ Seg segIdentity(void)
{
    Seg s;
    s.sum = 0.0f; s.len = 0; s.pre = 0; s.suf = 0; s.best = 0;
    return s;
}

__host__ __device__ __forceinline__ Seg combineSeg(Seg a, Seg b)
{
    Seg r;
    r.sum = a.sum + b.sum;
    r.len = a.len + b.len;
    // a's prefix run extends into b only if ALL of a is above threshold
    r.pre = (a.pre == a.len) ? a.len + b.pre : a.pre;
    // b's suffix run extends back into a only if ALL of b is above threshold
    r.suf = (b.suf == b.len) ? b.len + a.suf : b.suf;
    int j = a.suf + b.pre;                      // the run spanning the joint
    int m = (a.best > b.best) ? a.best : b.best;
    r.best = (j > m) ? j : m;
    return r;
}

// ---------------------------------------------------------------------------
// Mechanical: move a whole Seg down the warp. Given.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Seg shflDownSeg(unsigned mask, Seg s, int off)
{
    Seg r;
    r.sum  = __shfl_down_sync(mask, s.sum,  off);
    r.len  = __shfl_down_sync(mask, s.len,  off);
    r.pre  = __shfl_down_sync(mask, s.pre,  off);
    r.suf  = __shfl_down_sync(mask, s.suf,  off);
    r.best = __shfl_down_sync(mask, s.best, off);
    return r;
}

// ===========================================================================
// TODO 2 -- reduce a full 32-lane warp to a single Seg, held by lane 0.
//
//   Lane L holds the summary of the L-th sub-span, IN ORDER. Five calls to
//   shflDownSeg will get you there, and there are exactly two free choices:
//
//     - which of the two Segs is the left operand of combineSeg, and
//     - the sequence in which you take the five shuffle offsets.
//
//   For a sum BOTH choices are invisible -- every arrangement gives the same
//   answer, which is why the textbook warp reduction never explains either of
//   them and why example01.cu's warpReduceSum may use whichever it likes. For
//   this operator exactly one combination reduces the warp in row order. Write
//   down, for each of the five steps, which spans lane 0 has accumulated; if
//   that list is ever out of order you have the wrong arrangement, and the
//   symptom is a `best` that is right for short rows and wrong for long ones.
// ===========================================================================
__device__ __forceinline__ Seg warpReduceSeg(Seg s)
{
    // Two choices, and both matter here.
    //
    // 1. Operand order: the lane we shuffle FROM is at a HIGHER lane index and
    //    is therefore LATER in the row, so it is the right-hand operand.
    // 2. Offset order: the textbook sum reduction goes 16, 8, 4, 2, 1. That
    //    tree hands lane 0 the spans in the order 0, 16, 8, 24, ... -- fine for
    //    a commutative operator, wrong for this one. Going 1, 2, 4, 8, 16
    //    makes lane 0 accumulate spans 0..1, then 0..3, 0..7, 0..15, 0..31:
    //    contiguous and in order at every step.
    for (int off = 1; off < 32; off <<= 1) {
        Seg t = shflDownSeg(0xffffffffu, s, off);
        s = combineSeg(s, t);
    }
    return s;                                   // valid in lane 0
}

// ---------------------------------------------------------------------------
// The baseline. Complete and correct. One thread per row, sequential scan.
// ---------------------------------------------------------------------------
__global__ void segNaive(const float* __restrict__ data,
                         const int* __restrict__ rowStart,
                         int nRows, float thr, float* outSum, int* outBest)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nRows) return;
    int a = rowStart[r], b = rowStart[r + 1];
    float sum = 0.0f;
    int run = 0, best = 0;
    for (int i = a; i < b; ++i) {
        float x = data[i];
        sum += x;
        if (x > thr) { ++run; if (run > best) best = run; }
        else         { run = 0; }
    }
    outSum[r]  = sum;
    outBest[r] = best;
}

// ===========================================================================
// TODO 3 -- DESIGN. Implement launchSegFast().
//
//   You choose everything: how many kernels, the grid and block shape, which
//   threads cooperate on which row, and how a row longer than one warp is
//   split. The harness only checks the two output arrays and the clock.
//
//   Three constraints you have to reason about, not three hints:
//
//   (1) ORDER. combineSeg is not commutative, so the decomposition has to
//       reproduce the row's element order. Ask yourself what the classic
//       "lane L takes elements L, L+32, L+64, ..." strided loop does to that
//       order, and whether the strided loop's answer for `best` is right or
//       merely plausible. It is not enough for the pieces to be disjoint and
//       cover the row.
//
//   (2) COALESCING. Module 5's rule has not been repealed. The order-preserving
//       decomposition that first suggests itself gives lane L a contiguous
//       chunk of the row, and that is a stride-(chunk) access pattern. There is
//       a decomposition that is both order-preserving and fully coalesced.
//
//   (3) SKEW. Row lengths span four orders of magnitude here. A mapping that
//       gives every row the same amount of hardware is either starved on the
//       long rows or wasteful on the short ones. Module 8 priced exactly this.
//
//   Return 1 when you have implemented it, so the harness knows to run it.
//   Leave it returning 0 and the program exits gracefully.
// ===========================================================================
// One warp per row, and the row is walked in COALESCED TILES OF 32. Lane L
// takes element base+L, so a tile's 32 lanes hold the tile's 32 elements in
// order; reducing the warp in lane order therefore reduces the tile in row
// order, and folding tile summaries into a serial accumulator keeps the tiles
// in row order too. Order-preserving AND fully coalesced.
__global__ void segWarpRows(const float* __restrict__ data,
                            const int* __restrict__ rowStart,
                            const int* __restrict__ rowIdx, int nSel,
                            float thr, float* outSum, int* outBest)
{
    int wpb    = blockDim.x / 32;
    int gwarp  = blockIdx.x * wpb + (int)(threadIdx.x / 32);
    int nwarps = gridDim.x * wpb;
    int lane   = (int)(threadIdx.x & 31u);

    for (int k = gwarp; k < nSel; k += nwarps) {     // warp-uniform trip count
        int r = rowIdx[k];
        int a = rowStart[r], b = rowStart[r + 1];
        Seg acc = segIdentity();
        for (int base = a; base < b; base += 32) {   // warp-uniform
            int i = base + lane;
            Seg leaf = (i < b) ? segLeaf(data[i], thr) : segIdentity();
            Seg t    = warpReduceSeg(leaf);
            if (lane == 0) acc = combineSeg(acc, t);
        }
        if (lane == 0) { outSum[r] = acc.sum; outBest[r] = acc.best; }
    }
}

// One block per long row. The block's warps take CONTIGUOUS, tile-aligned
// segments of the row -- not interleaved tiles -- and thread 0 folds the warp
// summaries together in warp order. This is the skew fix: a 200,000-element
// row gets 8 warps instead of 1, and a 40-element row still gets 1.
#define LONG_THRESH 2048
__global__ void segBlockRows(const float* __restrict__ data,
                             const int* __restrict__ rowStart,
                             const int* __restrict__ rowIdx,
                             float thr, float* outSum, int* outBest)
{
    __shared__ Seg wres[32];
    int r  = rowIdx[blockIdx.x];
    int a  = rowStart[r], b = rowStart[r + 1];
    int nw = (int)(blockDim.x / 32);
    int w  = (int)(threadIdx.x / 32);
    int lane = (int)(threadIdx.x & 31u);

    int tiles = (b - a + 31) / 32;
    int per   = (tiles + nw - 1) / nw;
    int t0    = w * per;
    int t1    = (t0 + per < tiles) ? t0 + per : tiles;

    Seg acc = segIdentity();
    for (int t = t0; t < t1; ++t) {
        int i = a + t * 32 + lane;
        Seg leaf = (i < b) ? segLeaf(data[i], thr) : segIdentity();
        Seg tt   = warpReduceSeg(leaf);
        if (lane == 0) acc = combineSeg(acc, tt);
    }
    if (lane == 0) wres[w] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        Seg tot = segIdentity();
        for (int j = 0; j < nw; ++j) tot = combineSeg(tot, wres[j]);
        outSum[r] = tot.sum; outBest[r] = tot.best;
    }
}

static int launchSegFast(const float* d_data, const int* d_rowStart,
                         int nRows, float thr, float* d_sum, int* d_best)
{
    // The partition depends only on the row-length table, not on the data, so
    // it is built once and amortised over every launch (Module 8 exercise 2
    // made the same argument and priced it).
    static int  inited = 0;
    static int *d_short = nullptr, *d_long = nullptr;
    static int  nShort = 0, nLong = 0;
    static int  warpGrid = 0;

    if (!inited) {
        int* h = (int*)malloc((size_t)(nRows + 1) * sizeof(int));
        CHECK(cudaMemcpy(h, d_rowStart, (size_t)(nRows + 1) * sizeof(int),
                         cudaMemcpyDeviceToHost));
        int* hs = (int*)malloc((size_t)nRows * sizeof(int));
        int* hl = (int*)malloc((size_t)nRows * sizeof(int));
        for (int r = 0; r < nRows; ++r) {
            int L = h[r + 1] - h[r];
            if (L > LONG_THRESH) hl[nLong++] = r; else hs[nShort++] = r;
        }
        CHECK(cudaMalloc(&d_short, (size_t)(nShort + 1) * sizeof(int)));
        CHECK(cudaMalloc(&d_long,  (size_t)(nLong  + 1) * sizeof(int)));
        if (nShort) CHECK(cudaMemcpy(d_short, hs, (size_t)nShort * sizeof(int),
                                     cudaMemcpyHostToDevice));
        if (nLong)  CHECK(cudaMemcpy(d_long,  hl, (size_t)nLong  * sizeof(int),
                                     cudaMemcpyHostToDevice));
        int sm = 0, bpsm = 0;
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, segWarpRows, 128, 0);
        warpGrid = sm * bpsm;
        free(h); free(hs); free(hl);
        inited = 1;
    }

    if (nShort)
        segWarpRows<<<warpGrid, 128>>>(d_data, d_rowStart, d_short, nShort,
                                       thr, d_sum, d_best);
    if (nLong)
        segBlockRows<<<nLong, 256>>>(d_data, d_rowStart, d_long, thr,
                                     d_sum, d_best);
    return 1;
}

// ===========================================================================
// TODO 4 -- two predictions, committed before you run.
//
//   (a) PRED_SPEEDUP: your kernel's time divided into the baseline's. Scored
//       within a factor of 2 in either direction, which is generous, so do
//       not guess -- derive it. The baseline's cost has two separate defects
//       and you should be able to name both and estimate each.
//
//   (b) PRED_STRIDED_OK: suppose someone implements TODO 3 with the standard
//       grid-stride shape -- lane L of the warp accumulates elements
//       a+L, a+L+32, a+L+64, ... into its own Seg, and the warp is then
//       reduced in lane order. Is the resulting `best` correct for every row?
//       1 = yes, 0 = no. The harness runs exactly that kernel and counts the
//       rows it gets wrong, so answer from the algebra, not from hope.
// ===========================================================================
static const float PRED_SPEEDUP     = 8.0f;
static const int   PRED_STRIDED_OK  = 0;

// ---------------------------------------------------------------------------
// The harness runs this deliberately-strided version to score prediction (b).
// It is NOT a model answer for TODO 3.
// ---------------------------------------------------------------------------
__global__ void segStridedProbe(const float* __restrict__ data,
                                const int* __restrict__ rowStart,
                                int nRows, float thr, float* outSum, int* outBest)
{
    int warpsPerBlock = blockDim.x / 32;
    int w    = (blockIdx.x * warpsPerBlock) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (w >= nRows) return;
    int a = rowStart[w], b = rowStart[w + 1];

    Seg acc = segIdentity();
    for (int i = a + lane; i < b; i += 32)
        acc = combineSeg(acc, segLeaf(data[i], thr));
    Seg res = warpReduceSeg(acc);
    if (lane == 0) { outSum[w] = res.sum; outBest[w] = res.best; }
}

// ---------------------------------------------------------------------------
int main(void)
{
    if (PRED_SPEEDUP <= 0.0f || PRED_STRIDED_OK < 0) {
        printf("Set TODO 4 first.\n");
        return 0;
    }

    // ---- deterministic ragged layout ------------------------------------
    int* h_rowStart = (int*)malloc((size_t)(NROWS + 1) * sizeof(int));
    long long total = 0;
    h_rowStart[0] = 0;
    for (int r = 0; r < NROWS; ++r) {
        unsigned h = (unsigned)r * 2654435761u; h ^= h >> 13;
        int len;
        if (r < 10)                 len = 200000;
        else if ((h % 100u) < 95u)  len = 8 + (int)(h % 57u);
        else                        len = 4000 + (int)(h % 8000u);
        total += len;
        h_rowStart[r + 1] = (int)total;
    }
    printf("Module 12 exercise 02 -- segmented non-commutative reduction\n");
    printf("%d rows, %lld elements, %.1f MiB, threshold %.2f\n",
           NROWS, total, (double)total * 4.0 / 1048576.0, (double)THR);
    {
        int mn = 1 << 30, mx = 0;
        for (int r = 0; r < NROWS; ++r) {
            int L = h_rowStart[r + 1] - h_rowStart[r];
            if (L < mn) mn = L;
            if (L > mx) mx = L;
        }
        printf("row lengths: min %d, max %d\n\n", mn, mx);
    }

    float* h_data = (float*)malloc((size_t)total * sizeof(float));
    if (!h_data) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (long long i = 0; i < total; ++i) {
        unsigned h = (unsigned)i * 2246822519u; h ^= h >> 15; h *= 2654435761u;
        h ^= h >> 16;
        h_data[i] = (float)(h & 0xFFFFu) * (2.0f / 65536.0f) - 1.0f;   // [-1,1)
    }

    // ---- CPU reference ---------------------------------------------------
    double* refSum = (double*)malloc((size_t)NROWS * sizeof(double));
    int*    refBest = (int*)malloc((size_t)NROWS * sizeof(int));
    for (int r = 0; r < NROWS; ++r) {
        double s = 0.0; int run = 0, best = 0;
        for (int i = h_rowStart[r]; i < h_rowStart[r + 1]; ++i) {
            s += (double)h_data[i];
            if (h_data[i] > THR) { ++run; if (run > best) best = run; }
            else run = 0;
        }
        refSum[r] = s; refBest[r] = best;
    }

    float *d_data = nullptr, *d_sum = nullptr;
    int   *d_rowStart = nullptr, *d_best = nullptr;
    CHECK(cudaMalloc(&d_data, (size_t)total * sizeof(float)));
    CHECK(cudaMalloc(&d_rowStart, (size_t)(NROWS + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_sum, (size_t)NROWS * sizeof(float)));
    CHECK(cudaMalloc(&d_best, (size_t)NROWS * sizeof(int)));
    CHECK(cudaMemcpy(d_data, h_data, (size_t)total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_rowStart, h_rowStart, (size_t)(NROWS + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));

    float* h_sum  = (float*)malloc((size_t)NROWS * sizeof(float));
    int*   h_best = (int*)malloc((size_t)NROWS * sizeof(int));

    int implemented = launchSegFast(d_data, d_rowStart, NROWS, THR, d_sum, d_best);
    if (!implemented) { printf("Set TODO 3 first.\n"); return 0; }
    CHECK_KERNEL();

    // ---- timing (spec section 12): both configs back to back, rotated ----
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    const int naiveBlocks = (NROWS + 127) / 128;
    {
        cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        float acc = 0.0f; CHECK(cudaEventRecord(a));
        while (acc < 800.0f) {
            for (int k = 0; k < 4; ++k)
                segNaive<<<naiveBlocks, 128>>>(d_data, d_rowStart, NROWS, THR,
                                               d_sum, d_best);
            CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
            CHECK(cudaEventElapsedTime(&acc, a, b));
        }
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); CHECK_KERNEL();
    }

    double bNaive = 1e30, bFast = 1e30;
    for (int sweep = 0; sweep < 4; ++sweep) {
        for (int q = 0; q < 2; ++q) {
            int c = (q + sweep) % 2;
            CHECK(cudaEventRecord(e0));
            for (int k = 0; k < 10; ++k) {
                if (c == 0) segNaive<<<naiveBlocks, 128>>>(d_data, d_rowStart,
                                                           NROWS, THR, d_sum, d_best);
                else        launchSegFast(d_data, d_rowStart, NROWS, THR, d_sum, d_best);
            }
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (c == 0 && ms / 10.0 < bNaive) bNaive = ms / 10.0;
            if (c == 1 && ms / 10.0 < bFast)  bFast  = ms / 10.0;
        }
    }
    CHECK_KERNEL();

    // ---- validation, separate pass ---------------------------------------
    launchSegFast(d_data, d_rowStart, NROWS, THR, d_sum, d_best);
    CHECK_KERNEL();
    CHECK(cudaMemcpy(h_sum, d_sum, (size_t)NROWS * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_best, d_best, (size_t)NROWS * sizeof(int), cudaMemcpyDeviceToHost));
    int badBest = 0, badSum = 0, firstBad = -1;
    for (int r = 0; r < NROWS; ++r) {
        if (h_best[r] != refBest[r]) {
            ++badBest; if (firstBad < 0) firstBad = r;
        }
        double t = fabs(refSum[r]) > 1.0 ? fabs(refSum[r]) : 1.0;
        if (fabs((double)h_sum[r] - refSum[r]) > 1e-4 * t) ++badSum;
    }

    // ---- the strided probe, for prediction (b) ---------------------------
    int stridedBad = 0;
    {
        int wpb = 4;
        int blocks = (NROWS + wpb - 1) / wpb;
        segStridedProbe<<<blocks, wpb * 32>>>(d_data, d_rowStart, NROWS, THR,
                                              d_sum, d_best);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(h_best, d_best, (size_t)NROWS * sizeof(int),
                         cudaMemcpyDeviceToHost));
        for (int r = 0; r < NROWS; ++r) if (h_best[r] != refBest[r]) ++stridedBad;
    }

    double speedup = bNaive / bFast;
    printf("%-28s %10s %10s\n", "kernel", "ms", "GB/s");
    printf("--------------------------------------------------\n");
    printf("%-28s %10.4f %10.1f\n", "baseline (thread per row)", bNaive,
           (double)total * 4.0 / (bNaive * 1e-3) / 1e9);
    printf("%-28s %10.4f %10.1f\n", "your segFast", bFast,
           (double)total * 4.0 / (bFast * 1e-3) / 1e9);
    printf("speedup %.2fx\n\n", speedup);

    printf("correctness: %d/%d rows with a wrong `best`, %d with a wrong `sum`\n",
           badBest, NROWS, badSum);
    if (firstBad >= 0)
        printf("  first wrong row %d: got best=%d, expected %d (len %d)\n",
               firstBad, h_best[firstBad] , refBest[firstBad],
               h_rowStart[firstBad + 1] - h_rowStart[firstBad]);

    int s1 = (badBest == 0);
    int s2 = (badSum == 0);
    int s3 = (speedup >= 3.0);
    int s4 = (PRED_SPEEDUP <= 2.0 * speedup && PRED_SPEEDUP >= 0.5 * speedup);
    int s5 = (PRED_STRIDED_OK == (stridedBad == 0 ? 1 : 0));

    printf("\npredictions:\n");
    printf("  (a) speedup      predicted %.2fx, measured %.2fx   %s\n",
           (double)PRED_SPEEDUP, speedup, s4 ? "MATCH" : "MISS");
    printf("  (b) strided form correct for every row? you said %s;\n"
           "      the probe got %d of %d rows wrong                %s\n",
           PRED_STRIDED_OK ? "yes" : "no", stridedBad, NROWS, s5 ? "MATCH" : "MISS");

    int score = s1 + s2 + s3 + s4 + s5;
    printf("\nSCORE: %d/5  (best %s, sum %s, speed %s, pred-a %s, pred-b %s)\n",
           score, s1 ? "ok" : "no", s2 ? "ok" : "no", s3 ? "ok" : "no",
           s4 ? "ok" : "no", s5 ? "ok" : "no");

    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_data)); CHECK(cudaFree(d_rowStart));
    CHECK(cudaFree(d_sum)); CHECK(cudaFree(d_best));
    free(h_data); free(h_rowStart); free(refSum); free(refBest);
    free(h_sum); free(h_best);
    CHECK(cudaDeviceReset());

    printf("\nOVERALL: %s\n", (score == 5) ? "PASS" : "FAIL");
    return (score == 5) ? 0 : 1;
}
