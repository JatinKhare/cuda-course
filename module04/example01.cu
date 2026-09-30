// =====================================================================
// Module 4 / Example 1 : "A tour of the memory spaces"
//
// GOAL
//   One runnable file that (a) prints the real capacity of every level
//   of the RTX 3500 Ada memory hierarchy as the driver reports it, and
//   (b) demonstrates the single most misunderstood space: LOCAL memory.
//
//   Two kernels compute the SAME result from the SAME per-thread array.
//   One keeps the array in registers. The other is forced to put it in
//   local memory -- which is DRAM. Build with -Xptxas -v and read the
//   "stack frame" line: that is your local-memory footprint per thread.
//
// BUILD:  nvcc -arch=sm_89 -O3 -Xptxas -v -o example01.exe example01.cu
// RUN:    .\example01.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
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

#define K 16          // per-thread working array length
#define N (1 << 24)   // 16M elements: 64 MB in + 64 MB out, well past the 48 MB L2

// ---------------------------------------------------------------------
// __constant__ : 64 KB device-wide read-only window, cached in a small
// per-SM constant cache. Written from the host with cudaMemcpyToSymbol.
// ---------------------------------------------------------------------
__constant__ float c_w[K];

// ---------------------------------------------------------------------
// __device__ : an ordinary global-memory variable with static storage
// duration. Lives in DRAM for the lifetime of the context, visible to
// every thread of every kernel. Not a register, not "fast".
// ---------------------------------------------------------------------
__device__ float d_scale = 2.0f;

// ---------------------------------------------------------------------
// REGISTER version.
//
// acc[K] is a per-thread array. Every subscript the compiler sees is a
// compile-time constant, because both loops have a compile-time trip
// count and are fully unrolled. ptxas therefore assigns each element its
// own register and the array ceases to exist as an array.
//
// -Xptxas -v reports: "0 bytes stack frame".
// ---------------------------------------------------------------------
__global__ void k_registers(const float* __restrict__ in,
                            float* __restrict__ out, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    float v = in[t];
    float acc[K];

    #pragma unroll
    for (int i = 0; i < K; ++i) acc[i] = v * c_w[i];   // broadcast read of c_w

    float m = 0.0f;
    #pragma unroll
    for (int i = 0; i < K; ++i) m = fmaxf(m, fabsf(acc[i]));

    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < K; ++i) s = fmaf(acc[i], acc[K - 1 - i], s);

    out[t] = s / (m + 1e-6f) * d_scale;
}

// ---------------------------------------------------------------------
// LOCAL MEMORY version.
//
// Identical arithmetic. The only difference: the trip count k arrives as
// a kernel argument, so it is not known at compile time. The loops
// cannot be unrolled, so acc[i] is a *dynamic* index. A register file is
// not addressable -- there is no "load register number i" instruction on
// the SM. The compiler's only option is to place acc in local memory:
// thread-private storage carved out of DEVICE DRAM, addressed through
// the L1 and L2 caches.
//
// -Xptxas -v reports: "64 bytes stack frame" (16 floats x 4 B).
// In SASS the accesses become LDL / STL instead of register operands.
// ---------------------------------------------------------------------
__global__ void k_local(const float* __restrict__ in,
                        float* __restrict__ out, int n, int k)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    float v = in[t];
    float acc[K];

    for (int i = 0; i < k; ++i) acc[i] = v * c_w[i];

    float m = 0.0f;
    for (int i = 0; i < k; ++i) m = fmaxf(m, fabsf(acc[i]));

    float s = 0.0f;
    for (int i = 0; i < k; ++i) s = fmaf(acc[i], acc[k - 1 - i], s);

    out[t] = s / (m + 1e-6f) * d_scale;
}

// ---------------------------------------------------------------------
// Small launch wrappers so both kernels share one timing helper.
// ---------------------------------------------------------------------
static void run_reg(const float* in, float* out, int n, int k)
{ (void)k; k_registers<<<(n + 255) / 256, 256>>>(in, out, n); }

static void run_loc(const float* in, float* out, int n, int k)
{ k_local<<<(n + 255) / 256, 256>>>(in, out, n, k); }

static float bench(void (*fn)(const float*, float*, int, int),
                   const float* in, float* out, int n, int k, int iters)
{
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    fn(in, out, n, k);                       // warm-up
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) fn(in, out, n, k);
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaGetLastError());
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms / iters;
}

static void cpu_reference(const float* in, const float* w, float* out, int n)
{
    for (int t = 0; t < n; ++t) {
        float v = in[t], acc[K];
        for (int i = 0; i < K; ++i) acc[i] = v * w[i];
        float m = 0.0f;
        for (int i = 0; i < K; ++i) m = fmaxf(m, fabsf(acc[i]));
        float s = 0.0f;
        for (int i = 0; i < K; ++i) s = fmaf(acc[i], acc[K - 1 - i], s);
        out[t] = s / (m + 1e-6f) * 2.0f;
    }
}

int main(void)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));

    int l2 = 0, regsSM = 0, smemSM = 0, smemBlockOptin = 0, constMem = 0;
    CHECK(cudaDeviceGetAttribute(&l2,     cudaDevAttrL2CacheSize, dev));
    CHECK(cudaDeviceGetAttribute(&regsSM, cudaDevAttrMaxRegistersPerMultiprocessor, dev));
    CHECK(cudaDeviceGetAttribute(&smemSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    CHECK(cudaDeviceGetAttribute(&smemBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CHECK(cudaDeviceGetAttribute(&constMem, cudaDevAttrTotalConstantMemory, dev));

    printf("=== %s (sm_%d%d) : memory hierarchy ===\n", p.name, p.major, p.minor);
    printf("  Registers / SM           : %d x 32-bit  (%d KB)\n", regsSM, regsSM * 4 / 1024);
    printf("  Registers / thread (max) : 255\n");
    printf("  Shared mem / SM          : %d B (%.0f KB)\n", smemSM, smemSM / 1024.0);
    printf("  Shared mem / block max   : %zu B default, %d B opt-in\n",
           p.sharedMemPerBlock, smemBlockOptin);
    printf("  L1+SMEM unified / SM     : 128 KB (Ada, fixed)\n");
    printf("  L2 cache                 : %d B (%.0f MB)\n", l2, l2 / (1024.0 * 1024.0));
    printf("  Constant memory          : %d B (%.0f KB)\n", constMem, constMem / 1024.0);
    printf("  Global memory            : %.2f GiB\n",
           p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  cudaMalloc alignment     : >= 256 B guaranteed\n");

    // -----------------------------------------------------------------
    // L1 / shared split. On Ada the 128 KB unified block is partitioned
    // between L1 data cache and shared memory. The carveout is a HINT:
    // "give shared memory at least this percentage of the 128 KB".
    // 0 means "I want no shared memory, give the whole block to L1".
    // -----------------------------------------------------------------
    CHECK(cudaFuncSetAttribute(k_registers,
          cudaFuncAttributePreferredSharedMemoryCarveout, 0));   // favour L1

    cudaFuncAttributes fa;
    CHECK(cudaFuncGetAttributes(&fa, k_registers));
    printf("\n--- k_registers static attributes ---\n");
    printf("  numRegs        : %d\n", fa.numRegs);
    printf("  localSizeBytes : %zu\n", fa.localSizeBytes);
    printf("  constSizeBytes : %zu\n", fa.constSizeBytes);

    CHECK(cudaFuncGetAttributes(&fa, k_local));
    printf("--- k_local static attributes ---\n");
    printf("  numRegs        : %d\n", fa.numRegs);
    printf("  localSizeBytes : %zu      <-- bytes of DRAM per thread\n", fa.localSizeBytes);

    // -----------------------------------------------------------------
    // Data
    // -----------------------------------------------------------------
    size_t bytes = (size_t)N * sizeof(float);
    float* h_in  = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_got = (float*)malloc(bytes);
    float  h_w[K];
    for (int i = 0; i < K; ++i) h_w[i] = 0.25f + 0.125f * (float)i;
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f + (float)(i % 97) * 0.01f;

    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_w, h_w, sizeof(h_w)));   // host -> __constant__

    printf("\n  cudaMalloc returned %p and %p (low byte: %02x, %02x -> 256 B aligned)\n",
           (void*)d_in, (void*)d_out,
           (unsigned)((uintptr_t)d_in & 0xff), (unsigned)((uintptr_t)d_out & 0xff));

    cpu_reference(h_in, h_w, h_ref, N);

    // -----------------------------------------------------------------
    // Time + validate both versions
    // -----------------------------------------------------------------
    const int ITERS = 50;
    float ms_reg = bench(run_reg, d_in, d_out, N, K, ITERS);
    CHECK(cudaMemcpy(h_got, d_out, bytes, cudaMemcpyDeviceToHost));
    int bad_reg = 0;
    for (int i = 0; i < N; ++i)
        if (fabsf(h_got[i] - h_ref[i]) > 1e-5f * fmaxf(1.0f, fabsf(h_ref[i]))) ++bad_reg;

    float ms_loc = bench(run_loc, d_in, d_out, N, K, ITERS);
    CHECK(cudaMemcpy(h_got, d_out, bytes, cudaMemcpyDeviceToHost));
    int bad_loc = 0;
    for (int i = 0; i < N; ++i)
        if (fabsf(h_got[i] - h_ref[i]) > 1e-5f * fmaxf(1.0f, fabsf(h_ref[i]))) ++bad_loc;

    // Traffic the algorithm actually demands from global memory:
    // read N floats, write N floats. Local-memory traffic is *extra*.
    double gb = 2.0 * (double)bytes / 1e9;
    printf("\n--- %d elements, %d-entry per-thread array ---\n", N, K);
    printf("  registers : %8.4f ms   %7.1f GB/s  (%4.1f%% of 432)  mismatches=%d\n",
           ms_reg, gb / (ms_reg / 1e3), 100.0 * (gb / (ms_reg / 1e3)) / 432.0, bad_reg);
    printf("  local mem : %8.4f ms   %7.1f GB/s  (%4.1f%% of 432)  mismatches=%d\n",
           ms_loc, gb / (ms_loc / 1e3), 100.0 * (gb / (ms_loc / 1e3)) / 432.0, bad_loc);
    printf("  slowdown  : %.2fx\n", ms_loc / ms_reg);

    printf("\n%s\n", (bad_reg == 0 && bad_loc == 0) ? "PASS" : "FAIL");

    free(h_in); free(h_ref); free(h_got);
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    CHECK(cudaDeviceReset());
    return (bad_reg == 0 && bad_loc == 0) ? 0 : 1;
}
