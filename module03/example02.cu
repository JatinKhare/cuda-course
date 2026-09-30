// =====================================================================
// Module 3 / Example 2 : "Mapping a problem onto threads"
//
// GOAL
//   Three ways to cover n elements with threads, measured side by side:
//
//     v1  1 thread : 1 element        grid = ceil(n / block)
//     v2  1 thread : N elements       grid-stride loop, grid chosen by
//                                     the *machine*, not by n
//     v3  1 thread : N elements       grid-stride loop, but with a
//                                     deliberately silly grid (1 block)
//                                     to show that correctness is
//                                     independent of the launch config
//
//   The kernel is SAXPY: out[i] = a*x[i] + y[i]. It is purely
//   bandwidth-bound: 12 bytes of DRAM traffic per element (read x,
//   read y, write out), so the achieved GB/s tells you immediately how
//   close to the 432.0 GB/s hardware ceiling each variant gets.
//
//   n is deliberately 50,000,003 -- prime-ish, not a multiple of the
//   block size, so the last block is partial and the bounds guard
//   actually fires.
//
//   Part D is host-only: the integer-overflow trap in the standard
//   ceil-divide idiom.
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

static const long long N       = 50000003LL;   // not a multiple of anything useful
static const int       BLOCK   = 256;
static const float     ALPHA   = 2.5f;
static const double    PEAK_BW = 432.0;        // GB/s, RTX 3500 Ada

// ---------------------------------------------------------------------
// v1: one thread per element. The guard is mandatory: the grid covers
// ceil(n/BLOCK)*BLOCK >= n threads, so up to BLOCK-1 threads of the last
// block have no element. Without the guard they read and write past the
// end of the allocation.
//
// Cost of the guard: for every warp except the one straddling the end,
// the predicate is uniformly true, so the compiler emits a setp + a
// predicated (or branch-skipped) region that costs one instruction of
// issue, not a divergence penalty. Module 8 makes the predication story
// precise.
// ---------------------------------------------------------------------
__global__ void saxpy_1to1(const float* __restrict__ x,
                           const float* __restrict__ y,
                           float* __restrict__ out,
                           float a, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)                       // <-- the guard
        out[i] = a * x[i] + y[i];
}

// ---------------------------------------------------------------------
// v2/v3: grid-stride loop. One kernel, any grid size, always correct.
//
//   stride = total number of threads in the grid
//   thread i handles i, i+stride, i+2*stride, ...
//
// Why the stride is gridDim.x*blockDim.x and not blockDim.x: consecutive
// *global* thread ids must map to consecutive elements so that each warp
// still reads 32 consecutive floats = 128 contiguous bytes per iteration.
// A per-block stride would break that. Module 5 explains the cost.
//
// Note the loop condition is still a bounds test -- the guard did not
// disappear, it became the loop condition.
// ---------------------------------------------------------------------
__global__ void saxpy_gridstride(const float* __restrict__ x,
                                 const float* __restrict__ y,
                                 float* __restrict__ out,
                                 float a, long long n)
{
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride)
        out[i] = a * x[i] + y[i];
}

// ---------------------------------------------------------------------
static float timeKernel(void (*launcher)(int, const float*, const float*, float*, float, long long),
                        int grid, const float* x, const float* y, float* out)
{
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    launcher(grid, x, y, out, ALPHA, N);            // warm-up
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    const int ITERS = 20;
    CHECK(cudaEventRecord(t0));
    for (int it = 0; it < ITERS; ++it)
        launcher(grid, x, y, out, ALPHA, N);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    CHECK(cudaGetLastError());

    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    return ms / ITERS;
}

static void launch_1to1(int grid, const float* x, const float* y, float* out, float a, long long n)
{ saxpy_1to1<<<grid, BLOCK>>>(x, y, out, a, n); }

static void launch_gs(int grid, const float* x, const float* y, float* out, float a, long long n)
{ saxpy_gridstride<<<grid, BLOCK>>>(x, y, out, a, n); }

static void report(const char* name, int grid, float ms)
{
    double bytes = 3.0 * (double)N * sizeof(float);          // read x, read y, write out
    double gbs   = bytes / (ms * 1.0e-3) / 1.0e9;
    printf("  %-34s grid=%-8d %8.3f ms   %7.1f GB/s   %5.1f%% of peak\n",
           name, grid, ms, gbs, 100.0 * gbs / PEAK_BW);
}

static int validate(const char* name, const float* h_out, const float* h_x, const float* h_y)
{
    // Check every element -- the interesting failures live at the tail.
    long long bad = 0;
    long long firstBad = -1;
    for (long long i = 0; i < N; ++i) {
        float ref = ALPHA * h_x[i] + h_y[i];
        if (!(fabsf(h_out[i] - ref) <= 1e-5f * fmaxf(1.0f, fabsf(ref)))) {
            if (bad == 0) firstBad = i;
            ++bad;
        }
    }
    printf("  %-34s %s", name, bad == 0 ? "PASS\n" : "FAIL");
    if (bad) printf(" (%lld mismatches, first at i=%lld)\n", bad, firstBad);
    return bad == 0 ? 0 : 1;
}

int main(void)
{
    CHECK(cudaSetDevice(0));
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    printf("=== %s : %d SMs, %d threads/SM ===\n",
           p.name, p.multiProcessorCount, p.maxThreadsPerMultiProcessor);
    printf("n = %lld elements (%.1f MiB per array), block = %d\n",
           N, N * sizeof(float) / 1048576.0, BLOCK);

    size_t bytes = (size_t)N * sizeof(float);
    float *h_x = (float*)malloc(bytes), *h_y = (float*)malloc(bytes), *h_o = (float*)malloc(bytes);
    if (!h_x || !h_y || !h_o) { printf("host alloc failed\n"); return 1; }

    // Deterministic, index-derived initialization. No unseeded rand().
    for (long long i = 0; i < N; ++i) {
        h_x[i] = (float)((i * 1103515245LL + 12345LL) % 1000) * 0.001f;
        h_y[i] = (float)(i % 257) * 0.5f;
    }

    float *d_x, *d_y, *d_o;
    CHECK(cudaMalloc(&d_x, bytes));
    CHECK(cudaMalloc(&d_y, bytes));
    CHECK(cudaMalloc(&d_o, bytes));
    CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    int fails = 0;

    // ---- Part A: 1 thread : 1 element ----------------------------
    // The ceil-divide. n is a long long here precisely so this cannot
    // overflow; see Part D for what happens when it is an int.
    long long gridLL = (N + BLOCK - 1) / BLOCK;
    int grid1 = (int)gridLL;
    printf("\n--- v1: one thread per element ---\n");
    printf("  ceil(%lld / %d) = %d blocks -> %lld threads, %lld idle in the last block\n",
           N, BLOCK, grid1, gridLL * BLOCK, gridLL * BLOCK - N);

    CHECK(cudaMemset(d_o, 0, bytes));
    float ms1 = timeKernel(launch_1to1, grid1, d_x, d_y, d_o);
    CHECK(cudaMemcpy(h_o, d_o, bytes, cudaMemcpyDeviceToHost));
    fails += validate("v1 1:1", h_o, h_x, h_y);
    report("v1 1:1", grid1, ms1);

    // ---- Part B: grid-stride, machine-sized grid -----------------
    // The grid is a function of the GPU, not of n. 1536 threads/SM /
    // 256 threads/block = 6 resident blocks per SM at full occupancy;
    // asking for exactly that many gives one "wave" that stays resident
    // for the whole kernel.
    int blocksPerSM = p.maxThreadsPerMultiProcessor / BLOCK;      // 6
    int gridWave    = blocksPerSM * p.multiProcessorCount;        // 240
    printf("\n--- v2: grid-stride, grid sized to the machine ---\n");
    printf("  %d blocks/SM x %d SMs = %d blocks (one full wave), each thread does ~%lld elements\n",
           blocksPerSM, p.multiProcessorCount, gridWave,
           (N + (long long)gridWave * BLOCK - 1) / ((long long)gridWave * BLOCK));

    CHECK(cudaMemset(d_o, 0, bytes));
    float ms2 = timeKernel(launch_gs, gridWave, d_x, d_y, d_o);
    CHECK(cudaMemcpy(h_o, d_o, bytes, cudaMemcpyDeviceToHost));
    fails += validate("v2 grid-stride (wave)", h_o, h_x, h_y);
    report("v2 grid-stride (wave)", gridWave, ms2);

    // Same kernel, n-sized grid: degenerates to exactly v1 (every thread
    // runs the loop body once). Proof that grid-stride is a superset.
    CHECK(cudaMemset(d_o, 0, bytes));
    float ms2b = timeKernel(launch_gs, grid1, d_x, d_y, d_o);
    CHECK(cudaMemcpy(h_o, d_o, bytes, cudaMemcpyDeviceToHost));
    fails += validate("v2 grid-stride (n-sized)", h_o, h_x, h_y);
    report("v2 grid-stride (n-sized)", grid1, ms2b);

    // ---- Part C: grid-stride, absurd grid ------------------------
    // 1 block = 256 threads on a 40-SM GPU. 39 SMs idle. Still CORRECT.
    // That separation of correctness from configuration is the entire
    // argument for the idiom.
    printf("\n--- v3: grid-stride with a deliberately bad grid ---\n");
    CHECK(cudaMemset(d_o, 0, bytes));
    float ms3 = timeKernel(launch_gs, 1, d_x, d_y, d_o);
    CHECK(cudaMemcpy(h_o, d_o, bytes, cudaMemcpyDeviceToHost));
    fails += validate("v3 grid-stride (1 block)", h_o, h_x, h_y);
    report("v3 grid-stride (1 block)", 1, ms3);

    // ---- Part D: the ceil-divide overflow trap -------------------
    printf("\n--- Part D: integer overflow in (n + block - 1) / block ---\n");
    {
        int  n_int   = 2147483000;        // < INT_MAX = 2147483647
        int  b       = 1024;
        int  bad     = (n_int + b - 1) / b;                       // overflows
        long long ok = ((long long)n_int + b - 1) / b;            // does not
        printf("  n = %d, block = %d\n", n_int, b);
        printf("  int  arithmetic : (n + block - 1) / block = %d   <-- n + 1023 wrapped negative\n", bad);
        printf("  64-bit          : (n + block - 1) / block = %lld\n", ok);
        printf("  safe int form   : n / block + (n %% block != 0) = %d\n",
               n_int / b + (n_int % b != 0));
    }

    printf("\nOVERALL: %s\n", fails == 0 ? "PASS" : "FAIL");

    CHECK(cudaFree(d_x)); CHECK(cudaFree(d_y)); CHECK(cudaFree(d_o));
    free(h_x); free(h_y); free(h_o);
    CHECK(cudaDeviceReset());
    return fails == 0 ? 0 : 1;
}
