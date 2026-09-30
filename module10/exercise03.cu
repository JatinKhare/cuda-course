// =====================================================================
// Module 10 / Exercise 3 : "Everything the hardware does not give you"
//
// GOAL
//   There is no atomicArgmax. There is no 64-bit atomic "update these two
//   fields together". atomicCAS is the universal primitive from which any
//   atomic read-modify-write can be built, and this exercise makes you
//   build three things with it -- then shows you how to make one of them
//   disappear into a single hardware instruction by choosing a better
//   encoding.
//
// THE PROBLEM
//   Given x[0..n), produce:
//     (a) argmax(x), with ties broken toward the SMALLEST index;
//     (b) the encoded minimum AND maximum, updated together in one
//         64-bit word by one atomic operation per update.
//
//   Three datasets, and they are the point:
//     A  mixed signs, maximum is positive, maximum value appears 3 times
//     B  every value strictly negative (maximum is a negative number)
//     C  every value identical
//
//   Dataset B is where the well-known shortcut
//       atomicMax((int*)p, __float_as_int(v))
//   quietly produces the wrong answer. Work out why before you write
//   anything. (Hint: write out the 32 bits of -1.0f and of -2.0f and
//   compare them as signed integers, then as unsigned integers.)
//
//   Dataset C is where a compare-and-swap loop with the wrong exit
//   condition either spins forever or silently returns the wrong index.
//
//   The data contains +0.0f but deliberately never -0.0f, because -0.0f
//   is the one place where "compare as floats" and "compare as encoded
//   keys" disagree, and this exercise is not about that argument.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN:   .\exercise03.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
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

static const int N      = 4000003;      // deliberately not a multiple of 32
static const int TPB    = 256;
static const int SWEEPS = 4;
static const int ITERS  = 20;

// Reinterpret a float's bits, on host or device.
__host__ __device__ inline unsigned int fbits(float f)
{
#ifdef __CUDA_ARCH__
    return __float_as_uint(f);
#else
    unsigned int u; memcpy(&u, &f, sizeof(u)); return u;
#endif
}
__host__ __device__ inline float unfbits(unsigned int u)
{
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float f; memcpy(&f, &u, sizeof(f)); return f;
#endif
}

// The RAW packing: the float's bits in the high half, the index in the
// low half. This is the form the CAS loop works on. Note that ordering
// raw packed words as integers is NOT the same as ordering by value --
// that is exactly why a CAS loop is needed here and an atomicMax is not
// enough.
__host__ __device__ inline unsigned long long pack_raw(float v, unsigned int idx)
{
    return ((unsigned long long)fbits(v) << 32) | (unsigned long long)idx;
}
__host__ __device__ inline float        raw_value(unsigned long long p)
{ return unfbits((unsigned int)(p >> 32)); }
__host__ __device__ inline unsigned int raw_index(unsigned long long p)
{ return (unsigned int)(p & 0xFFFFFFFFull); }

// The identity element for the raw form: value -inf, index "worse than
// any real index".
#define RAW_IDENTITY (((unsigned long long)0xFF800000ull << 32) | 0xFFFFFFFFull)

// =====================================================================
//                            YOUR CODE
// =====================================================================

// ---------------------------------------------------------------------
// TODO 1: An order-preserving map from float to unsigned int.
//
//   Return a 32-bit key such that, for any two floats a and b that are
//   not NaN,
//
//       a < b   if and only if   encode_key(a) < encode_key(b)
//
//   as UNSIGNED 32-bit comparison. Total order, no exceptions, negatives
//   included.
//
//   Start from the IEEE-754 layout: one sign bit, then 8 exponent bits,
//   then 23 mantissa bits, in that order, most significant first. For two
//   non-negative floats the bit patterns already compare correctly as
//   unsigned. For two negative floats they compare correctly but
//   BACKWARDS. And every negative float has a bit pattern numerically
//   larger than every non-negative one.
//
//   Two different corrections are needed, one for each half of the number
//   line, and they are not the same correction. One line of code each.
//
//   The harness exhaustively checks monotonicity over a large sample of
//   floats including the awkward ones, so you cannot pass this by luck.
// ---------------------------------------------------------------------
__host__ __device__ inline unsigned int encode_key(float v)
{
    unsigned int b = fbits(v);
    (void)b;
    return 0u;          // YOUR CODE HERE (TODO 1)
}

// ---------------------------------------------------------------------
// TODO 2: The CAS loop.
//
//   `best` points to one 64-bit word in the RAW packing above, already
//   initialised to RAW_IDENTITY. Every thread has one candidate
//   (value, index). Make `*best` end up holding the (value, index) with
//   the largest value, ties going to the smallest index.
//
//   You may use ONLY atomicCAS on `best`. No atomicMax, no atomicMin, no
//   atomicExch. (That is not an arbitrary rule: the raw packing is not
//   monotonic in the value, so no single comparison atomic can do this.
//   TODO 3 is where you get to fix that.)
//
//   Three things go wrong here, in order of how often they are seen:
//     - atomicCAS returns the OLD contents of the word, not a success
//       flag. Looping on the wrong thing gives you either an infinite
//       loop or an exit after a swap that never happened.
//     - when the CAS fails, the value you must re-compare against is the
//       one the CAS just returned, not the one you read before the loop.
//     - a thread whose candidate is not better than the current winner
//       must not enter the loop at all, and must re-test that after every
//       failed attempt. Dataset C (all values equal) will tell you if you
//       got this wrong: every thread's candidate ties with the incumbent,
//       and a loop that retries on a tie never terminates.
// ---------------------------------------------------------------------
__device__ inline void argmax_cas_update(unsigned long long* best,
                                         float v, unsigned int idx)
{
    (void)best; (void)v; (void)idx;
    // YOUR CODE HERE (TODO 2)
}

// ---------------------------------------------------------------------
// TODO 3: Make the CAS loop unnecessary.
//
//   Design a 64-bit packing of (key, index) such that plain unsigned
//   64-bit `atomicMax` on the packed word produces exactly the same
//   answer as TODO 2: largest value wins, ties go to the SMALLEST index.
//
//   `pack_for_max` builds it; `unpack_idx_for_max` recovers the index so
//   the harness can read your answer. They must be exact inverses on the
//   index.
//
//   The value half is straightforward once TODO 1 is right. The index
//   half is the decision: the comparison is "bigger wins" in both halves,
//   but you need "smaller index wins". Choose an encoding of the index
//   that turns one into the other, and make sure the identity element of
//   the whole word (which is what the harness memsets the word to,
//   i.e. all zero bits) is still worse than every real candidate.
// ---------------------------------------------------------------------
__host__ __device__ inline unsigned long long pack_for_max(float v, unsigned int idx)
{
    (void)v; (void)idx;
    return 0ull;        // YOUR CODE HERE (TODO 3)
}
__host__ __device__ inline unsigned int unpack_idx_for_max(unsigned long long p)
{
    (void)p;
    return 0u;          // YOUR CODE HERE (TODO 3)
}

// ---------------------------------------------------------------------
// TODO 4: Two fields, one atomic.
//
//   `bounds` is one 64-bit word holding, from the top down:
//     bits 63..32 : encode_key of the largest value seen so far
//     bits 31..0  : encode_key of the smallest value seen so far
//   initialised to 0x00000000FFFFFFFF (worst possible max, worst
//   possible min).
//
//   Fold one value v into it so that the word is ALWAYS internally
//   consistent -- there must be no instant at which another thread could
//   read a word whose max came from one update and whose min came from a
//   different one. Two separate atomics on the two halves would satisfy
//   the final answer and violate that requirement; do it in one.
//
//   Return early when there is nothing to do. On a monotonically boring
//   input almost every thread has nothing to do, and a loop that runs
//   anyway is the difference between a fast kernel and a slow one.
// ---------------------------------------------------------------------
__device__ inline void bounds_update(unsigned long long* bounds, float v)
{
    (void)bounds; (void)v;
    // YOUR CODE HERE (TODO 4)
}

// ---------------------------------------------------------------------
// TODO 5: Make `argmax_fast` fast.
//
//   `argmax_naive` below does one global atomic per element against a
//   single address. You know from this module exactly what that costs.
//
//   Write `argmax_fast` so that far fewer global atomic instructions
//   target that one address, without changing the answer -- including
//   the tie-break. Everything from Modules 6 through 10 is available:
//   shared memory, barriers, warp intrinsics, the active mask, this
//   module's aggregation ideas. You choose.
//
//   Two things to be careful about:
//     - n is 4,000,003. The last warp of the last block is partial, and
//       any lane you include that has no element must not be able to win.
//     - if you use a barrier, every thread of the block must reach it.
//
//   The harness requires argmax_fast to be at least 5x faster than
//   argmax_naive and to give the identical answer on all three datasets.
// ---------------------------------------------------------------------
__global__ void argmax_fast(const float* __restrict__ x, int n,
                            unsigned long long* best)
{
    // YOUR CODE HERE (TODO 5)
    (void)x; (void)n; (void)best;
}

// =====================================================================
//                          END OF YOUR CODE
// =====================================================================

__global__ void argmax_cas_kernel(const float* __restrict__ x, int n,
                                  unsigned long long* best)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) argmax_cas_update(best, x[i], (unsigned int)i);
}

__global__ void argmax_naive(const float* __restrict__ x, int n,
                             unsigned long long* best)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicMax(best, pack_for_max(x[i], (unsigned int)i));
}

// Fills a small array of 64-bit words with an initial value. Used so every
// timed iteration starts from a fresh accumulator WITHOUT paying for a
// cudaMemset inside the timed region -- otherwise iteration 2 onwards would
// find the answer already in place, every CAS-loop thread would take its
// early-out, and the CAS loop would measure as free.
__global__ void fill_u64(unsigned long long* p, int m, unsigned long long v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) p[i] = v;
}

__global__ void bounds_kernel(const float* __restrict__ x, int n,
                              unsigned long long* bounds)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) bounds_update(bounds, x[i]);
}

// ---------------------------------------------------------------------
static unsigned int lcg(unsigned int& s) { s = s * 1664525u + 1013904223u; return s; }

static void make_dataset(int which, float* h, int n)
{
    unsigned int s = 424242u + 7919u * (unsigned int)which;
    if (which == 0) {                        // A: mixed, max positive, 3-way tie
        for (int i = 0; i < n; ++i) {
            unsigned int r = lcg(s);
            float m = (float)(r >> 12) * 1e-6f;
            h[i] = ((r & 1u) ? -m : m) * 0.5f;
            if ((r >> 1) % 4096u == 0u) h[i] = 0.0f;
        }
        h[7]          =  1234.5f;
        h[n / 2]      =  1234.5f;
        h[n - 11]     =  1234.5f;
    } else if (which == 1) {                 // B: everything strictly negative
        for (int i = 0; i < n; ++i) {
            unsigned int r = lcg(s);
            h[i] = -((float)(r >> 12) * 1e-6f + 1e-3f);
        }
        h[999]        = -1e-3f;
        h[n - 5]      = -1e-3f;
    } else {                                 // C: everything identical
        for (int i = 0; i < n; ++i) h[i] = -3.25f;
    }
}

struct Ref { unsigned int idx; float maxv, minv; };

static Ref cpu_reference(const float* h, int n)
{
    Ref r; r.idx = 0; r.maxv = h[0]; r.minv = h[0];
    for (int i = 1; i < n; ++i) {
        if (h[i] > r.maxv) { r.maxv = h[i]; r.idx = (unsigned int)i; }
        if (h[i] < r.minv) r.minv = h[i];
    }
    return r;
}

int main(void)
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("n = %d\n\n", N);

    // ---- TODO 1 gate + exhaustive-ish monotonicity check ----
    {
        // A spread of awkward floats plus a swept sample.
        const int M = 4096;
        float* probe = (float*)malloc((size_t)M * sizeof(float));
        int m = 0;
        const float NEG_INF = unfbits(0xFF800000u), POS_INF = unfbits(0x7F800000u);
        const float fixed[] = { NEG_INF, -3.4e38f, -1e9f, -1.0f, -1e-9f,
                                -1.17549435e-38f, 0.0f, 1.17549435e-38f,
                                1e-9f, 1.0f, 1e9f, 3.4e38f, POS_INF };
        for (int i = 0; i < (int)(sizeof(fixed) / sizeof(fixed[0])); ++i) probe[m++] = fixed[i];
        unsigned int s = 31415926u;
        while (m < M) {
            unsigned int b = lcg(s);
            float f = unfbits(b);
            if (f == f && !isinf(f) && f != 0.0f) probe[m++] = f;   // skip NaN/inf/zero
        }
        // sort ascending (simple, M is small)
        for (int i = 1; i < m; ++i) {
            float k = probe[i]; int j = i - 1;
            while (j >= 0 && probe[j] > k) { probe[j + 1] = probe[j]; --j; }
            probe[j + 1] = k;
        }
        int mono = 1, allZero = 1;
        for (int i = 0; i < m; ++i) if (encode_key(probe[i]) != 0u) allZero = 0;
        for (int i = 1; i < m && mono; ++i) {
            unsigned int a = encode_key(probe[i - 1]), b = encode_key(probe[i]);
            if (probe[i - 1] < probe[i]) { if (!(a < b)) mono = 0; }
            else                         { if (a != b)   mono = 0; }
        }
        if (allZero) { free(probe); printf("Set TODO 1 first.\n"); return 0; }
        printf("=== TODO 1: order-preserving key ===\n");
        printf("  [%s] encode_key is strictly monotonic over %d sampled floats\n",
               mono ? "PASS" : "FAIL", m);
        printf("  encode_key(-1.0f) = 0x%08X   encode_key(-2.0f) = 0x%08X\n",
               encode_key(-1.0f), encode_key(-2.0f));
        printf("  raw bits  (-1.0f) = 0x%08X   raw bits  (-2.0f) = 0x%08X   "
               "<- note which pair is in the right order\n\n",
               fbits(-1.0f), fbits(-2.0f));
        free(probe);
        if (!mono) { printf("OVERALL: FAIL\n"); return 1; }
    }
    if (pack_for_max(1.0f, 0u) == 0ull && pack_for_max(-1.0f, 5u) == 0ull) {
        printf("Set TODO 3 first.\n"); return 0;
    }

    float* h = (float*)malloc((size_t)N * sizeof(float));
    float* d_x = nullptr;
    unsigned long long* d_w = nullptr;
    CHECK(cudaMalloc(&d_x, (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_w, sizeof(unsigned long long)));

    const int GRID = (N + TPB - 1) / TPB;
    const char* dsName[3] = { "A mixed, 3-way tie", "B all negative    ",
                              "C all identical   " };

    int score = 0, maxScore = 0;

    // ================= correctness, all three datasets =================
    printf("=== correctness ===\n");
    printf("  %-20s %14s %14s %14s %14s\n",
           "dataset", "CPU argmax", "CAS loop", "atomicMax", "fast");
    int casOk = 1, maxOk = 1, fastOk = 1, bndOk = 1;
    for (int d = 0; d < 3; ++d) {
        make_dataset(d, h, N);
        Ref ref = cpu_reference(h, N);
        CHECK(cudaMemcpy(d_x, h, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

        unsigned long long init = RAW_IDENTITY, out = 0ull;

        CHECK(cudaMemcpy(d_w, &init, 8, cudaMemcpyHostToDevice));
        argmax_cas_kernel<<<GRID, TPB>>>(d_x, N, d_w);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&out, d_w, 8, cudaMemcpyDeviceToHost));
        const unsigned int iCas = raw_index(out);

        CHECK(cudaMemset(d_w, 0, 8));
        argmax_naive<<<GRID, TPB>>>(d_x, N, d_w);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&out, d_w, 8, cudaMemcpyDeviceToHost));
        const unsigned int iMax = unpack_idx_for_max(out);

        CHECK(cudaMemset(d_w, 0, 8));
        argmax_fast<<<GRID, TPB>>>(d_x, N, d_w);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&out, d_w, 8, cudaMemcpyDeviceToHost));
        const unsigned int iFast = unpack_idx_for_max(out);

        printf("  %-20s %14u %14u %14u %14u\n", dsName[d], ref.idx, iCas, iMax, iFast);
        if (iCas  != ref.idx) casOk  = 0;
        if (iMax  != ref.idx) maxOk  = 0;
        if (iFast != ref.idx) fastOk = 0;

        // ---- bounds ----
        unsigned long long binit = 0x00000000FFFFFFFFull;
        CHECK(cudaMemcpy(d_w, &binit, 8, cudaMemcpyHostToDevice));
        bounds_kernel<<<GRID, TPB>>>(d_x, N, d_w);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&out, d_w, 8, cudaMemcpyDeviceToHost));
        const unsigned int gotMax = (unsigned int)(out >> 32);
        const unsigned int gotMin = (unsigned int)(out & 0xFFFFFFFFull);
        if (gotMax != encode_key(ref.maxv) || gotMin != encode_key(ref.minv)) {
            bndOk = 0;
            printf("      bounds FAIL: max key 0x%08X want 0x%08X, "
                   "min key 0x%08X want 0x%08X\n",
                   gotMax, encode_key(ref.maxv), gotMin, encode_key(ref.minv));
        }
    }
    maxScore += 4;
    printf("\n  [%s] TODO 2  CAS-loop argmax, all 3 datasets\n", casOk ? "PASS" : "FAIL");
    printf("  [%s] TODO 3  packing + single atomicMax, all 3 datasets\n",
           maxOk ? "PASS" : "FAIL");
    printf("  [%s] TODO 4  packed {max,min} bounds, all 3 datasets\n",
           bndOk ? "PASS" : "FAIL");
    printf("  [%s] TODO 5  argmax_fast agrees on all 3 datasets\n",
           fastOk ? "PASS" : "FAIL");
    score += casOk + maxOk + bndOk + fastOk;
    printf("\n");

    // ================= timing: dataset A, all configs back to back ======
    make_dataset(0, h, N);
    CHECK(cudaMemcpy(d_x, h, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    unsigned long long* d_words = nullptr;
    CHECK(cudaMalloc(&d_words, (size_t)ITERS * sizeof(unsigned long long)));

    for (int i = 0; i < 60; ++i) argmax_naive<<<GRID, TPB>>>(d_x, N, d_words);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

    double tCas = 1e30, tNaive = 1e30, tFast = 1e30, tBounds = 1e30;
#define TIME(store, initval, launch)                                           \
    do {                                                                       \
        { const int r_ = 0; launch; }                                          \
        fill_u64<<<1, 64>>>(d_words, ITERS, (initval));                        \
        CHECK(cudaDeviceSynchronize());                                        \
        CHECK(cudaEventRecord(e0));                                            \
        for (int r_ = 0; r_ < ITERS; ++r_) { launch; }                         \
        CHECK(cudaEventRecord(e1));                                            \
        CHECK(cudaEventSynchronize(e1));                                       \
        float ms_ = 0.f; CHECK(cudaEventElapsedTime(&ms_, e0, e1));            \
        if (ms_ / ITERS < (store)) (store) = ms_ / ITERS;                      \
    } while (0)
    for (int sweep = 0; sweep < SWEEPS; ++sweep) {
        TIME(tCas,    RAW_IDENTITY,
             (argmax_cas_kernel<<<GRID, TPB>>>(d_x, N, d_words + r_)));
        TIME(tNaive,  0ull,
             (argmax_naive     <<<GRID, TPB>>>(d_x, N, d_words + r_)));
        TIME(tFast,   0ull,
             (argmax_fast      <<<GRID, TPB>>>(d_x, N, d_words + r_)));
        TIME(tBounds, 0x00000000FFFFFFFFull,
             (bounds_kernel    <<<GRID, TPB>>>(d_x, N, d_words + r_)));
    }
    CHECK(cudaGetLastError());
#undef TIME

    const double BYTES = (double)N * 4.0;
    printf("=== timing (dataset A, min of %d sweeps x %d iters) ===\n", SWEEPS, ITERS);
    printf("  %-34s %10s %10s %9s\n", "kernel", "ms", "GB/s", "%peak");
    const char* tn[4] = { "CAS loop, one address        ",
                          "atomicMax, one address       ",
                          "argmax_fast (yours)          ",
                          "bounds CAS loop, one address " };
    const double tv[4] = { tCas, tNaive, tFast, tBounds };
    for (int k = 0; k < 4; ++k)
        printf("  %-34s %10.4f %10.1f %8.1f%%\n", tn[k], tv[k],
               BYTES / (tv[k] * 1e-3) / 1e9, 100.0 * BYTES / (tv[k] * 1e-3) / 1e9 / 432.0);
    printf("  (x[] is %.1f MB, comfortably inside the 48 MB L2, so these GB/s\n"
           "   figures are cache bandwidth, not DRAM bandwidth. Read them as a\n"
           "   relative scale, not as a fraction of 432 GB/s.)\n", BYTES / 1048576.0);
    printf("\n  argmax_fast vs argmax_naive: %.2fx\n", tNaive / tFast);
    printf("  CAS loop vs single atomicMax: %.2fx\n\n", tCas / tNaive);

    maxScore += 1;
    const int fastEnough = (tNaive / tFast >= 5.0);
    printf("  [%s] TODO 5  argmax_fast at least 5.00x argmax_naive (got %.2fx)\n",
           fastEnough ? "PASS" : "FAIL", tNaive / tFast);
    score += fastEnough;

    printf("\n  score: %d/%d\n", score, maxScore);
    const int pass = (score == maxScore);
    printf("OVERALL: %s\n", pass ? "PASS" : "FAIL");

    free(h);
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_x)); CHECK(cudaFree(d_w)); CHECK(cudaFree(d_words));
    CHECK(cudaDeviceReset());
    return pass ? 0 : 1;
}
