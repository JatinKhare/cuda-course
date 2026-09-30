// =====================================================================
// Module 4 / Exercise 2 : "The kernel that is slow for no reason"
//                          (debugging / optimization)
//
// SYMPTOM
//   fir_v0 below is a 16-tap normalized FIR filter. Each thread does 16
//   loads from a contiguous window, a max-reduction over 16 values, and
//   16 FMAs. That is a trivially small amount of work per thread and the
//   global memory access pattern is as good as it gets: consecutive
//   threads read consecutive elements.
//
//   It runs at roughly a quarter of the throughput you would expect for
//   a kernel that reads one float and writes one float per thread.
//
//   Build it with:
//       nvcc -arch=sm_89 -O3 -Xptxas -v -c exercise02.cu
//   and read the "Function properties" block that ptxas prints for
//   fir_v0. Something in that block is not zero and should be.
//
// YOUR JOB
//   Diagnose it, then write fir_v1 so that the offending quantity is
//   zero, without changing what the kernel computes. Then deal with the
//   harder version of the same problem in fir_v2, where the tap count
//   genuinely is a runtime value.
//
//   Do not modify fir_v0 -- it is the baseline the harness times against.
//
// BUILD:  nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise02.exe exercise02.cu
// RUN:    .\exercise02.exe
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

#define TAPS    16                // compile-time length used by v0 / v1
#define MAXTAPS 32                // largest length v2 must support
#define N       (1 << 24)         // 16M samples: 64 MB in + 64 MB out,
                                  // comfortably past the 48 MB L2
#define VN      (1 << 20)         // outputs the CPU reference checks
#define ITERS   50

__constant__ float c_h[MAXTAPS];  // filter coefficients, same for all threads

// =====================================================================
// v0 -- BASELINE. Do not modify.
// =====================================================================
__global__ void fir_v0(const float* __restrict__ x, float* __restrict__ y,
                       int n, int taps)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    float win[TAPS];

    for (int i = 0; i < taps; ++i) win[i] = x[t + i];

    float m = 0.0f;
    for (int i = 0; i < taps; ++i) m = fmaxf(m, fabsf(win[i]));

    float inv = 1.0f / (m + 1e-6f);
    float s   = 0.0f;
    for (int i = 0; i < taps; ++i) s = fmaf(c_h[i] * inv, win[i], s);

    y[t] = s;
}

// =====================================================================
// TODO 1: Write fir_v1. Same arithmetic, same result, but the per-thread
//         window must live in the fastest storage the SM has.
//
//         Constraint: the kernel signature is fixed (the harness calls
//         it), and you may assume the filter length is the compile-time
//         constant TAPS.
//
//         Warning: the obvious one-line "fix" is not enough by itself.
//         Whatever you do, re-run with -Xptxas -v and confirm the number
//         you are chasing is actually zero. If it is not, you have not
//         fixed it, you have only moved it.
// =====================================================================
__global__ void fir_v1(const float* __restrict__ x, float* __restrict__ y, int n)
{
    // YOUR CODE HERE (TODO 1)
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) y[t] = 0.0f;       // placeholder so the file compiles as shipped
}

// =====================================================================
// TODO 2: Now the hard case. Suppose the filter length is a property of
//         the *problem*, not of the source file: the caller passes
//         `taps`, and it is 8, 12, 16, or 32.
//
//         Produce a kernel that, for each of those lengths, has exactly
//         the same machine-level property you achieved in TODO 1 -- no
//         per-thread storage outside the register file -- while still
//         letting the host choose the length at run time.
//
//         The mechanism is yours to pick. Think about where the
//         "compile-time constant" can come from when the value is only
//         known at run time, and about which side of the host/device
//         boundary gets to make that choice.
// =====================================================================
template <int TAPS_C>
__global__ void fir_v2(const float* __restrict__ x, float* __restrict__ y, int n)
{
    // YOUR CODE HERE (TODO 2, device side)
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) y[t] = 0.0f;       // placeholder so the file compiles as shipped
}

// ---------------------------------------------------------------------
// TODO 3: Host-side dispatch for fir_v2. Given a runtime `taps` in
//         {8, 12, 16, 32}, launch the right instantiation. Return false
//         for anything else.
// ---------------------------------------------------------------------
static bool launch_v2(const float* x, float* y, int n, int taps,
                      int blocks, int threads)
{
    // YOUR CODE HERE (TODO 3)
    (void)x; (void)y; (void)n; (void)taps; (void)blocks; (void)threads;
    return false;
}

// =====================================================================
// Harness
// =====================================================================
static void cpu_reference(const float* x, const float* h, float* y, int n, int taps)
{
    for (int t = 0; t < n; ++t) {
        float m = 0.0f;
        for (int i = 0; i < taps; ++i) m = fmaxf(m, fabsf(x[t + i]));
        float inv = 1.0f / (m + 1e-6f);
        float s = 0.0f;
        for (int i = 0; i < taps; ++i) s = fmaf(h[i] * inv, x[t + i], s);
        y[t] = s;
    }
}

static int compare(const float* got, const float* ref, int n)
{
    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (fabsf(got[i] - ref[i]) > 1e-5f * fmaxf(1.0f, fabsf(ref[i]))) ++bad;
    return bad;
}

int main(void)
{
    CHECK(cudaSetDevice(0));
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));

    const size_t xbytes = (size_t)(N + MAXTAPS) * sizeof(float);
    const size_t ybytes = (size_t)N * sizeof(float);

    float* h_x   = (float*)malloc(xbytes);
    float* h_y   = (float*)malloc(ybytes);
    float* h_ref = (float*)malloc(ybytes);
    float  h_h[MAXTAPS];

    // Deterministic, index-derived input.
    for (size_t i = 0; i < (size_t)(N + MAXTAPS); ++i)
        h_x[i] = sinf(0.001f * (float)i) + 0.25f * cosf(0.017f * (float)i);
    for (int i = 0; i < MAXTAPS; ++i) h_h[i] = 1.0f / (1.0f + (float)i);

    float *d_x = nullptr, *d_y = nullptr;
    CHECK(cudaMalloc(&d_x, xbytes));
    CHECK(cudaMalloc(&d_y, ybytes));
    CHECK(cudaMemcpy(d_x, h_x, xbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_h, h_h, sizeof(h_h)));

    cpu_reference(h_x, h_h, h_ref, VN, TAPS);

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    float ms = 0.0f;

    // Report what ptxas decided, straight from the driver.
    cudaFuncAttributes a0, a1;
    CHECK(cudaFuncGetAttributes(&a0, fir_v0));
    CHECK(cudaFuncGetAttributes(&a1, fir_v1));
    printf("=== %s ===\n", p.name);
    printf("  fir_v0 : %2d regs, %4zu B local/thread\n", a0.numRegs, a0.localSizeBytes);
    printf("  fir_v1 : %2d regs, %4zu B local/thread\n", a1.numRegs, a1.localSizeBytes);

    // Useful traffic: one float in, one float out per output sample.
    const double gb = ((double)N * 2.0 * sizeof(float)) / 1e9;
    int pass = 1;

    // ---- v0 ---------------------------------------------------------
    fir_v0<<<blocks, threads>>>(d_x, d_y, N, TAPS);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < ITERS; ++i) fir_v0<<<blocks, threads>>>(d_x, d_y, N, TAPS);
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    float ms0 = ms / ITERS;
    CHECK(cudaMemcpy(h_y, d_y, ybytes, cudaMemcpyDeviceToHost));
    int bad0 = compare(h_y, h_ref, VN);
    printf("\n%-8s %9s %10s %9s %8s %8s\n", "version", "ms", "GB/s", "%peak", "local B", "result");
    printf("%-8s %9.4f %10.1f %8.1f%% %8zu %8s\n", "v0", ms0, gb / (ms0 / 1e3),
           100.0 * (gb / (ms0 / 1e3)) / 432.0, a0.localSizeBytes,
           bad0 ? "WRONG" : "ok");
    if (bad0) pass = 0;

    // ---- v1 ---------------------------------------------------------
    CHECK(cudaMemset(d_y, 0, ybytes));
    fir_v1<<<blocks, threads>>>(d_x, d_y, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < ITERS; ++i) fir_v1<<<blocks, threads>>>(d_x, d_y, N);
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    float ms1 = ms / ITERS;
    CHECK(cudaMemcpy(h_y, d_y, ybytes, cudaMemcpyDeviceToHost));
    int bad1 = compare(h_y, h_ref, VN);
    printf("%-8s %9.4f %10.1f %8.1f%% %8zu %8s\n", "v1", ms1, gb / (ms1 / 1e3),
           100.0 * (gb / (ms1 / 1e3)) / 432.0, a1.localSizeBytes,
           bad1 ? "WRONG" : "ok");

    if (bad1) {
        printf("  [FAIL] v1 does not compute the reference result "
               "(still the placeholder?)\n");
        pass = 0;
    } else {
        if (a1.localSizeBytes != 0) {
            printf("  [FAIL] v1 still uses %zu bytes of local memory per thread\n",
                   a1.localSizeBytes);
            pass = 0;
        }
        if (!(ms1 < 0.75f * ms0)) {
            printf("  [FAIL] v1 is not meaningfully faster than v0 (%.4f vs %.4f ms)\n",
                   ms1, ms0);
            pass = 0;
        }
    }

    // ---- v2 : runtime tap count ------------------------------------
    printf("\n--- v2, runtime tap count ---\n");
    const int taplist[4] = { 8, 12, 16, 32 };
    for (int k = 0; k < 4; ++k) {
        int taps = taplist[k];
        float* ref = (float*)malloc(ybytes);
        cpu_reference(h_x, h_h, ref, VN, taps);

        CHECK(cudaMemset(d_y, 0, ybytes));
        bool ok = launch_v2(d_x, d_y, N, taps, blocks, threads);
        if (!ok) {
            printf("  taps=%2d : launch_v2 refused (TODO 2/3 not done)\n", taps);
            pass = 0;
            free(ref);
            continue;
        }
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < ITERS; ++i) launch_v2(d_x, d_y, N, taps, blocks, threads);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        float ms2 = ms / ITERS;

        CHECK(cudaMemcpy(h_y, d_y, ybytes, cudaMemcpyDeviceToHost));
        int bad2 = compare(h_y, ref, VN);
        printf("  taps=%2d : %8.4f ms  %8.1f GB/s  %s\n", taps, ms2,
               gb / (ms2 / 1e3), bad2 ? "WRONG" : "ok");
        if (bad2) pass = 0;
        if (taps == TAPS && !(ms2 < 1.5f * ms1)) {
            printf("  [FAIL] v2 at taps=16 (%.4f ms) is much slower than v1 "
                   "(%.4f ms) -- it is not getting the same treatment\n", ms2, ms1);
            pass = 0;
        }
        free(ref);
    }

    printf("\n%s\n", pass ? "PASS" : "FAIL");

    free(h_x); free(h_y); free(h_ref);
    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaDeviceReset());
    return pass ? 0 : 1;
}
