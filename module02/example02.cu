// =====================================================================
// Module 2 / Example 2 : "The whole pipeline, and the two ways it breaks"
//
// GOAL
//   (1) The canonical five-step CUDA program: allocate on the device,
//       copy host -> device, launch, copy device -> host, free.
//   (2) __host__ __device__ : ONE definition of the math, compiled twice,
//       so the CPU reference and the GPU kernel cannot drift apart.
//   (3) Two failure modes you must be able to recognize on sight:
//         --illegal    a STICKY error: the kernel touches an out-of-range
//                      address. cudaGetLastError() right after the launch
//                      says "success". The error only appears at the next
//                      synchronizing call -- and then the context is dead
//                      and EVERY later CUDA call returns the same error.
//         --hostderef  dereferencing a device pointer on the host. Not a
//                      CUDA error at all: a plain access violation.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN:    .\example02.exe
//         .\example02.exe --illegal
//         .\example02.exe --hostderef      (this one crashes, on purpose)
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#define CHECK_KERNEL() do {                                                \
    CHECK(cudaGetLastError());                                             \
    CHECK(cudaDeviceSynchronize());                                        \
} while (0)

// ---------------------------------------------------------------------
// __host__ __device__ : nvcc emits TWO object codes for this function,
// one x86-64 and one SASS. The host copy is what main() calls; the device
// copy is inlined into the kernel. Same source, so the reference and the
// kernel cannot disagree about what the algorithm is.
//
// (They can still disagree about the last bit: see the tolerance below.)
// ---------------------------------------------------------------------
__host__ __device__ __forceinline__ float smoothstep01(float x)
{
    float t = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
    return t * t * (3.0f - 2.0f * t);
}

// ---------------------------------------------------------------------
// __global__ : device code, host-callable, launched with <<<>>>,
// returns void. Its arguments are copied into a small per-launch
// parameter buffer; `in` and `out` must be DEVICE addresses.
// ---------------------------------------------------------------------
__global__ void apply_smoothstep(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)                       // Module 3 makes indexing rigorous;
        out[i] = smoothstep01(in[i]);// here, one element per thread.
}

// The same kernel with the bounds guard removed. Used only by --illegal.
__global__ void apply_smoothstep_unguarded(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    (void)n;
    out[i] = smoothstep01(in[i]);
}

// ---------------------------------------------------------------------
static void report(const char* what, cudaError_t e)
{
    printf("  %-42s -> %s (%s)\n", what, cudaGetErrorName(e), cudaGetErrorString(e));
}

int main(int argc, char** argv)
{
    bool illegal  = (argc > 1 && strcmp(argv[1], "--illegal")  == 0);
    bool hostderef= (argc > 1 && strcmp(argv[1], "--hostderef")== 0);

    const int n = 1000003;                  // deliberately not a multiple of 256
    const size_t bytes = (size_t)n * sizeof(float);

    CHECK(cudaSetDevice(0));

    // --- host allocation ------------------------------------------------
    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    if (!h_in || !h_out || !h_ref) { fprintf(stderr, "host alloc failed\n"); return 1; }

    // Deterministic initialization: index-derived, spanning [-0.25, 1.25]
    // so the clamp branches inside smoothstep01 are actually exercised.
    for (int i = 0; i < n; ++i)
        h_in[i] = -0.25f + 1.5f * ((float)(i % 4096) / 4095.0f);

    // --- device allocation ----------------------------------------------
    // cudaMalloc writes a DEVICE address into a HOST variable. That is why
    // it takes void** : the pointer itself lives on the host, the memory it
    // names lives on the GPU.
    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc((void**)&d_in,  bytes));
    CHECK(cudaMalloc((void**)&d_out, bytes));

    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;   // 3907 blocks; the last
                                                       // one is partly idle
    printf("n = %d, launch <<<%d, %d>>>  (%d threads, %d beyond the data)\n",
           n, blocks, threads, blocks * threads, blocks * threads - n);

    if (hostderef) {
        // ---------------------------------------------------------------
        // d_in is a perfectly valid 64-bit number. It is just not a number
        // that means anything to the CPU's MMU. The host has no page mapped
        // there, so this is an access violation -- the process dies before
        // any CUDA call is ever made. No cudaError_t is involved.
        // ---------------------------------------------------------------
        printf("\n--hostderef: about to evaluate d_in[0] on the HOST...\n");
        fflush(stdout);
        printf("d_in[0] = %f\n", d_in[0]);     // <-- crashes here
        printf("(if you are reading this line, something is very wrong)\n");
        return 0;
    }

    if (illegal) {
        // ---------------------------------------------------------------
        // Sticky error demonstration.
        // ---------------------------------------------------------------
        // A realistic version of the bug: the grid was sized from the
        // BYTE count instead of the ELEMENT count, so the launch has 4x
        // too many threads -- and the kernel has no bounds guard to
        // absorb them.
        int badBlocks = (int)((bytes + threads - 1) / threads);
        printf("\n--illegal: launching the unguarded kernel as <<<%d, %d>>>\n",
               badBlocks, threads);
        apply_smoothstep_unguarded<<<badBlocks, threads>>>(d_in, d_out, n);

        report("cudaGetLastError() right after launch", cudaGetLastError());
        printf("     ^ the launch configuration was legal, so this is clean.\n"
               "       The kernel has not necessarily even started yet.\n\n");

        report("cudaDeviceSynchronize()", cudaDeviceSynchronize());
        printf("     ^ the error surfaces HERE, at the first synchronizing call.\n\n");

        printf("  From now on the context is poisoned. Every call fails:\n");
        report("cudaGetLastError()", cudaGetLastError());
        report("cudaMemcpy(D2H)", cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        report("cudaMalloc(1 byte)", cudaMalloc((void**)&d_out, 1));
        report("cudaFree(d_in)", cudaFree(d_in));
        report("cudaDeviceReset()", cudaDeviceReset());
        printf("\n  Note two things:\n"
               "   * cudaGetLastError() did NOT clear this one. A sticky error\n"
               "     is a property of the context, not a one-shot status flag.\n"
               "   * cudaDeviceReset() 'succeeds' only because it DESTROYS the\n"
               "     context -- every allocation and every result is gone with it.\n"
               "  Fix the kernel; do not try to clear the error.\n");
        free(h_in); free(h_out); free(h_ref);
        return 1;
    }

    // --- the normal path -------------------------------------------------
    apply_smoothstep<<<blocks, threads>>>(d_in, d_out, n);
    CHECK_KERNEL();

    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // --- CPU reference, calling the SAME function ------------------------
    for (int i = 0; i < n; ++i)
        h_ref[i] = smoothstep01(h_in[i]);

    int bad = 0; double worst = 0.0; int worstIdx = -1;
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)h_out[i] - (double)h_ref[i]);
        double tol = 1e-5 * fmax(1.0, fabs((double)h_ref[i]));
        if (d > worst) { worst = d; worstIdx = i; }
        if (d > tol) ++bad;
    }
    printf("max abs difference vs CPU reference = %.3e at i = %d\n", worst, worstIdx);
    printf("%s (%d mismatching elements)\n", bad == 0 ? "PASS" : "FAIL", bad);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);
    CHECK(cudaDeviceReset());
    return bad == 0 ? 0 : 1;
}
