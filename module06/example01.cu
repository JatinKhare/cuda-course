// =====================================================================
// Module 6 / Example 1 : "Shared memory mechanics"
//
// Five demonstrations, in order:
//
//   A. Scope and lifetime. Every resident block gets its own private
//      copy of a __shared__ array. Two blocks writing the "same"
//      variable do not see each other.
//
//   B. Static vs dynamic declaration. The same kernel written with
//      `__shared__ float tile[N]` and with `extern __shared__`, plus
//      the third launch parameter, produce identical results.
//
//   C. Carving several differently-typed arrays out of ONE dynamic
//      allocation -- and the alignment hazard that lives there.
//      This program also demonstrates the failure, on purpose, and
//      recovers from it.  (Run with --align-bug to see it.)
//
//   D. Capacity vs occupancy. How many blocks fit on an SM as a
//      function of bytes of shared memory per block, measured with the
//      occupancy API rather than guessed.
//
//   E. The 48 KB / 99 KB boundary. A launch asking for 64 KB of
//      dynamic shared memory fails until the kernel opts in via
//      cudaFuncAttributeMaxDynamicSharedMemorySize.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN:    .\example01.exe            (normal)
//         .\example01.exe --align-bug (shows the misaligned-carve fault)
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
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

// ---------------------------------------------------------------------
// A. Scope and lifetime.
//
// `tag` is ONE declaration in the source and N physically distinct
// arrays at run time -- one per resident block. Block b writes b+1 into
// every slot, barriers, then reads slot 0. It must read back b+1, never
// some other block's value, no matter how the blocks interleave.
// ---------------------------------------------------------------------
__global__ void per_block_scope(int* out)
{
    __shared__ int tag[128];

    tag[threadIdx.x] = (int)blockIdx.x + 1;
    __syncthreads();                 // barrier; Module 9 makes this precise

    // Every thread reads a slot written by a *different* thread.
    int seen = tag[(threadIdx.x + 64) % 128];

    if (threadIdx.x == 0) out[blockIdx.x] = seen;
}

// ---------------------------------------------------------------------
// B. Static vs dynamic.
//
// Both kernels do the same thing: stage a block's 256 inputs in shared
// memory, barrier, then have each thread read the element its neighbour
// loaded. The only difference is where the bytes come from.
// ---------------------------------------------------------------------
__global__ void reverse_static(const float* __restrict__ in,
                               float* __restrict__ out, int n)
{
    __shared__ float s[256];                   // size fixed at compile time

    int base = (int)blockIdx.x * 256;
    int i    = base + (int)threadIdx.x;

    s[threadIdx.x] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    if (i < n) out[i] = s[255 - threadIdx.x];
}

__global__ void reverse_dynamic(const float* __restrict__ in,
                                float* __restrict__ out, int n)
{
    extern __shared__ float sdyn[];            // size fixed at launch time

    int base = (int)blockIdx.x * (int)blockDim.x;
    int i    = base + (int)threadIdx.x;

    sdyn[threadIdx.x] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    if (i < n) out[i] = sdyn[blockDim.x - 1 - threadIdx.x];
}

// ---------------------------------------------------------------------
// C. Carving one dynamic allocation into several typed arrays.
//
// The kernel wants three per-block arrays:
//     int    idx[NI]      NI = 33   (deliberately not a nice number)
//     float2 pos[NP]      NP = 64   -- needs 8-byte alignment
//     float  wgt[NP]
//
// The C++ type system will not help here: `extern __shared__` is one
// untyped blob. Offsets are yours to compute, and an offset that is not
// a multiple of the element's alignment is a fault, not a slowdown.
//
// `bug` selects the naive (wrong) offset so the failure is observable.
// ---------------------------------------------------------------------
#define NI 33
#define NP 64

__host__ __device__ __forceinline__ size_t align_up(size_t off, size_t a)
{
    return (off + a - 1) & ~(a - 1);
}

// Host-side byte count, written once and used by both the kernel and
// the launch, so the two can never disagree.
__host__ __device__ __forceinline__ size_t carve_offsets(bool bug,
                                                         size_t* off_pos,
                                                         size_t* off_wgt)
{
    size_t off = (size_t)NI * sizeof(int);              // 132 bytes
    *off_pos = bug ? off : align_up(off, alignof(float2));   // 132 vs 136
    size_t after_pos = *off_pos + (size_t)NP * sizeof(float2);
    *off_wgt = align_up(after_pos, alignof(float));
    return *off_wgt + (size_t)NP * sizeof(float);
}

__global__ void carved(float* out, bool bug)
{
    extern __shared__ char smem[];

    size_t off_pos, off_wgt;
    (void)carve_offsets(bug, &off_pos, &off_wgt);

    int*    idx = reinterpret_cast<int*>(smem);
    float2* pos = reinterpret_cast<float2*>(smem + off_pos);
    float*  wgt = reinterpret_cast<float*>(smem + off_wgt);

    for (int k = (int)threadIdx.x; k < NI; k += (int)blockDim.x)
        idx[k] = k * k;
    for (int k = (int)threadIdx.x; k < NP; k += (int)blockDim.x) {
        pos[k] = make_float2((float)k, (float)(2 * k));   // <-- 8-byte store
        wgt[k] = 0.5f * (float)k;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int k = 0; k < NP; ++k) acc += wgt[k] * (pos[k].x + pos[k].y);
        acc += (float)idx[NI - 1];
        out[blockIdx.x] = acc;
    }
}

// ---------------------------------------------------------------------
// D/E. A kernel whose dynamic shared memory request is a launch-time
// knob, used to map capacity against blocks-per-SM.
// ---------------------------------------------------------------------
__global__ void smem_hog(float* out, int words)
{
    extern __shared__ float hog[];
    for (int k = (int)threadIdx.x; k < words; k += (int)blockDim.x)
        hog[k] = (float)k;
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = hog[words - 1];
}

// =====================================================================
int main(int argc, char** argv)
{
    bool want_bug = (argc > 1 && strcmp(argv[1], "--align-bug") == 0);

    CHECK(cudaSetDevice(0));

    int smemPerBlockDefault = 0, smemPerSM = 0, smemOptin = 0, nSM = 0;
    CHECK(cudaDeviceGetAttribute(&smemPerBlockDefault,
                                 cudaDevAttrMaxSharedMemoryPerBlock, 0));
    CHECK(cudaDeviceGetAttribute(&smemPerSM,
                                 cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    CHECK(cudaDeviceGetAttribute(&smemOptin,
                                 cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CHECK(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, 0));

    printf("=== Module 6 / Example 1 : shared memory mechanics ===\n");
    printf("SMs                              : %d\n", nSM);
    printf("shared mem / block  (default max): %d B  (%.0f KB)\n",
           smemPerBlockDefault, smemPerBlockDefault / 1024.0);
    printf("shared mem / block  (opt-in max) : %d B  (%.0f KB)\n",
           smemOptin, smemOptin / 1024.0);
    printf("shared mem / SM                  : %d B  (%.0f KB)\n",
           smemPerSM, smemPerSM / 1024.0);

    int fails = 0;

    // ---------------- A ------------------------------------------------
    printf("\n--- A. scope and lifetime: one declaration, one array per block ---\n");
    {
        const int NB = 8;
        int* d_out = nullptr;
        CHECK(cudaMalloc(&d_out, NB * sizeof(int)));
        per_block_scope<<<NB, 128>>>(d_out);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        int h[8];
        CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int b = 0; b < NB; ++b) if (h[b] != b + 1) ++bad;
        printf("  block b read back b+1 in %d/%d blocks : %s\n",
               NB - bad, NB, bad == 0 ? "PASS" : "FAIL");
        printf("  (the array is declared once; %d private copies exist at run time)\n", NB);
        fails += (bad != 0);
        CHECK(cudaFree(d_out));
    }

    // ---------------- B ------------------------------------------------
    printf("\n--- B. static vs dynamic declaration ---\n");
    {
        const int n = 256 * 97;
        float *h_in = (float*)malloc(n * sizeof(float));
        float *h_a  = (float*)malloc(n * sizeof(float));
        float *h_b  = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; ++i) h_in[i] = 0.25f * (float)(i % 1021);

        float *d_in, *d_a, *d_b;
        CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        CHECK(cudaMalloc(&d_a,  n * sizeof(float)));
        CHECK(cudaMalloc(&d_b,  n * sizeof(float)));
        CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

        // Static: the size is baked into the SASS, and ptxas reports it.
        reverse_static<<<n / 256, 256>>>(d_in, d_a, n);
        CHECK(cudaGetLastError());

        // Dynamic: the size is the THIRD launch parameter, in BYTES.
        size_t dynBytes = 256 * sizeof(float);
        reverse_dynamic<<<n / 256, 256, dynBytes>>>(d_in, d_b, n);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost));

        int bad = 0;
        for (int i = 0; i < n; ++i) {
            int base = (i / 256) * 256, lane = i % 256;
            float want = h_in[base + 255 - lane];
            if (h_a[i] != want || h_b[i] != want) ++bad;
        }
        printf("  static  : __shared__ float s[256];      size known to ptxas\n");
        printf("  dynamic : extern __shared__ float s[];  size = 3rd launch arg = %zu B\n",
               dynBytes);
        printf("  both kernels agree with the reference : %s\n", bad == 0 ? "PASS" : "FAIL");
        fails += (bad != 0);

        cudaFuncAttributes fa;
        CHECK(cudaFuncGetAttributes(&fa, reverse_static));
        printf("  cudaFuncGetAttributes(reverse_static ).sharedSizeBytes = %zu\n",
               fa.sharedSizeBytes);
        CHECK(cudaFuncGetAttributes(&fa, reverse_dynamic));
        printf("  cudaFuncGetAttributes(reverse_dynamic).sharedSizeBytes = %zu"
               "   <- static part only; the dynamic part is invisible here\n",
               fa.sharedSizeBytes);

        CHECK(cudaFree(d_in)); CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b));
        free(h_in); free(h_a); free(h_b);
    }

    // ---------------- C ------------------------------------------------
    printf("\n--- C. carving one dynamic allocation into three typed arrays ---\n");
    {
        size_t off_pos_ok, off_wgt_ok, off_pos_bug, off_wgt_bug;
        size_t bytes_ok  = carve_offsets(false, &off_pos_ok,  &off_wgt_ok);
        size_t bytes_bug = carve_offsets(true,  &off_pos_bug, &off_wgt_bug);

        printf("  int idx[%d]    : offset %4d, %zu B\n", NI, 0, NI * sizeof(int));
        printf("  float2 pos[%d] : offset %4zu (naive would be %zu), %zu B, needs %zu B alignment\n",
               NP, off_pos_ok, off_pos_bug, NP * sizeof(float2), alignof(float2));
        printf("  float wgt[%d]  : offset %4zu, %zu B\n", NP, off_wgt_ok, NP * sizeof(float));
        printf("  total request  : %zu B  (naive carve would ask for %zu B)\n",
               bytes_ok, bytes_bug);

        float* d_out = nullptr;
        CHECK(cudaMalloc(&d_out, 4 * sizeof(float)));

        carved<<<4, 128, bytes_ok>>>(d_out, false);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        float h[4];
        CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
        // Reference: sum_k 0.5k*(k + 2k) + (NI-1)^2 = 1.5*sum k^2 + 1024
        double ref = 0.0;
        for (int k = 0; k < NP; ++k) ref += 0.5 * k * (3.0 * k);
        ref += (double)(NI - 1) * (NI - 1);
        int bad = (fabs((double)h[0] - ref) > 1e-3 * ref);
        printf("  aligned carve  : got %.1f, want %.1f  -> %s\n",
               h[0], ref, bad ? "FAIL" : "PASS");
        fails += bad;

        if (want_bug) {
            printf("  --align-bug: launching with pos at offset %zu (not a multiple of %zu)\n",
                   off_pos_bug, alignof(float2));
            carved<<<4, 128, bytes_bug>>>(d_out, true);
            cudaError_t e1 = cudaGetLastError();
            cudaError_t e2 = cudaDeviceSynchronize();
            printf("  launch  : %s\n", cudaGetErrorName(e1));
            printf("  execute : %s  <- the 8-byte store to pos[k] faulted\n",
                   cudaGetErrorName(e2));
            printf("  This error is STICKY. Nothing further in this process will work.\n");
            CHECK(cudaFree(d_out));   // will report the sticky error and exit
            return 1;
        }
        printf("  (run with --align-bug to see what the naive offset does)\n");
        CHECK(cudaFree(d_out));
    }

    // ---------------- D ------------------------------------------------
    printf("\n--- D. shared memory per block caps blocks per SM ---\n");
    {
        const int threads = 256;
        int reserved = 0;
        CHECK(cudaDeviceGetAttribute(&reserved,
                                     cudaDevAttrReservedSharedMemoryPerBlock, 0));
        printf("  driver-reserved shared memory per block: %d B\n", reserved);
        printf("  %-12s %-9s %-10s %-11s %-10s %s\n",
               "bytes/block", "charged", "blocks/SM", "threads/SM",
               "occupancy", "limiter");
        const int reqs[] = { 0, 1024, 4096, 8192, 12288, 16384, 25600, 49152 };
        for (int k = 0; k < (int)(sizeof(reqs) / sizeof(reqs[0])); ++k) {
            int nb = 0;
            CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                      &nb, smem_hog, threads, (size_t)reqs[k]));
            int charged   = reqs[k] + reserved;
            int capBySmem = smemPerSM / charged;
            int capByThr  = 1536 / threads;
            const char* lim = (capBySmem >= capByThr) ? "threads/SM"
                                                      : "shared memory";
            char occ[16];
            snprintf(occ, sizeof(occ), "%.1f%%", 100.0 * nb * threads / 1536.0);
            printf("  %-12d %-9d %-10d %-11d %-10s %s\n",
                   reqs[k], charged, nb, nb * threads, occ, lim);
        }
        printf("  blocks/SM = min( 1536/%d , %d/(bytes+%d) , 24 )\n",
               threads, smemPerSM, reserved);
        printf("  Note the +%d: the SM charges every block %d B of shared memory it\n"
               "  never gets to use, which is why 16384 B gives 5 blocks and not 6.\n",
               reserved, reserved);
    }

    // ---------------- E ------------------------------------------------
    printf("\n--- E. the 48 KB default ceiling and the opt-in ---\n");
    {
        float* d_out = nullptr;
        CHECK(cudaMalloc(&d_out, sizeof(float)));

        const size_t big = 64 * 1024;                 // 64 KB > 48 KB default
        int words = (int)(big / sizeof(float));

        smem_hog<<<1, 256, big>>>(d_out, words);
        cudaError_t e = cudaGetLastError();
        printf("  request %zu B without opting in : %s\n", big, cudaGetErrorName(e));
        printf("    (a launch-configuration error -- caught by cudaGetLastError(),\n"
               "     non-sticky, the context survives)\n");

        CHECK(cudaFuncSetAttribute(smem_hog,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)big));
        smem_hog<<<1, 256, big>>>(d_out, words);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        float h = 0.0f;
        CHECK(cudaMemcpy(&h, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        int bad = (h != (float)(words - 1));
        printf("  after cudaFuncSetAttribute(..MaxDynamicSharedMemorySize, %zu) : %s\n",
               big, bad ? "FAIL" : "PASS");
        fails += bad;

        int nb = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, smem_hog, 256, big));
        printf("  blocks/SM at 64 KB/block : %d   (%d KB/SM of shared memory available)\n",
               nb, smemPerSM / 1024);
        printf("  the hard ceiling for a single block is %d B (%.0f KB), not %d KB:\n"
               "  the SM keeps a slice of the 128 KB L1+shared array for L1.\n",
               smemOptin, smemOptin / 1024.0, smemPerSM / 1024);

        CHECK(cudaFree(d_out));
    }

    printf("\nOVERALL: %s\n", fails == 0 ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return fails == 0 ? 0 : 1;
}
