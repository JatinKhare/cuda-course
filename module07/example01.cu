// =====================================================================
// Module 7 / Example 1 : the bank map, the conflict degree, and what
//                        each replay actually costs.
//
// GOAL
//   1. Print the lane -> bank map for a set of shared-memory indexing
//      expressions, so that bank = (byte_addr / 4) % 32 stops being a
//      formula and becomes a table you can read.
//   2. Compute the conflict degree of each expression the same mechanical
//      way Module 5 counted sectors: enumerate 32 lanes, bucket them,
//      take the maximum multiplicity of DISTINCT words per bank.
//   3. Measure the cost, and compare it against the degree.
//
//   Every pattern below reads from the same __shared__ array with the
//   same instruction count and the same number of useful bytes. The only
//   thing that changes is which bank each lane lands in.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN:    .\example01.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <ctime>
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

static const int BANKS  = 32;
static const int BANK_W = 4;    // bytes per bank per cycle
static const int WARP   = 32;

// ============ PART A : the bank arithmetic, on the host ==============

// ARCHITECTURE-SPECIFIC (32 banks x 4 B). PORTABLE CONCEPT: banked SRAM.
static int bank_of(uintptr_t byte_addr) { return (int)((byte_addr / BANK_W) % BANKS); }

// The indexing expressions we analyse. Element type is float (4 B), so
// element index i sits at byte 4*i.
enum Pat { P_ID, P_S2, P_S3, P_S4, P_S8, P_S16, P_S32, P_HALF, P_ZERO, P_S33, P_NUM };
static const char* PAT_NAME[P_NUM] = {
    "s[tid]", "s[2*tid]", "s[3*tid]", "s[4*tid]", "s[8*tid]",
    "s[16*tid]", "s[32*tid]", "s[tid/2]", "s[0]", "s[33*tid]"
};
static int pat_index(int p, int lane) {
    switch (p) {
        case P_ID:   return lane;
        case P_S2:   return 2 * lane;
        case P_S3:   return 3 * lane;
        case P_S4:   return 4 * lane;
        case P_S8:   return 8 * lane;
        case P_S16:  return 16 * lane;
        case P_S32:  return 32 * lane;
        case P_HALF: return lane / 2;
        case P_ZERO: return 0;
        case P_S33:  return 33 * lane;
        default:     return lane;
    }
}

// The mechanical method, mirroring Module 5's sector count:
//   for each lane -> element -> byte address -> (bank, word)
//   bucket the DISTINCT words per bank
//   degree = max bucket size
// Two lanes landing on the SAME word in the same bank are a broadcast and
// cost nothing extra, so we count distinct words, never lanes.
static int conflict_degree(int p, int elemBytes, int* degPerBank /* 32, may be NULL */)
{
    unsigned long long words[BANKS][WARP * 4];
    int                nw[BANKS];
    for (int b = 0; b < BANKS; ++b) nw[b] = 0;

    for (int lane = 0; lane < WARP; ++lane) {
        uintptr_t a0 = (uintptr_t)pat_index(p, lane) * (uintptr_t)elemBytes;
        for (int off = 0; off < elemBytes; off += BANK_W) {   // wide types span words
            uintptr_t a = a0 + off;
            int b = bank_of(a);
            unsigned long long w = a / BANK_W;
            bool seen = false;
            for (int k = 0; k < nw[b]; ++k) if (words[b][k] == w) { seen = true; break; }
            if (!seen) words[b][nw[b]++] = w;
        }
    }
    int mx = 0;
    for (int b = 0; b < BANKS; ++b) {
        if (degPerBank) degPerBank[b] = nw[b];
        if (nw[b] > mx) mx = nw[b];
    }
    return mx;
}

static void print_bank_map(int p)
{
    printf("  %-11s ", PAT_NAME[p]);
    for (int lane = 0; lane < WARP; ++lane)
        printf("%2d%s", bank_of((uintptr_t)pat_index(p, lane) * 4), (lane == 31) ? "" : " ");
    printf("\n");
}

// ============ PART B : the kernels ===================================
//
// One kernel template, one indexing expression per instantiation. Each
// loop iteration adds a multiple of 128 B (32 floats) to the index: that
// changes the ADDRESS but not the BANK, so every iteration has exactly
// the conflict degree computed above and the measurement is a clean
// throughput test. Four independent accumulators keep the FADD
// dependency chain from becoming the bottleneck.

#define TN      4096        // floats of shared memory (16 KB)
#define ITERS   512
#define BLOCKS  320         // 8 blocks per SM on 40 SMs
#define THREADS 256

template <int P>
__global__ void bankKernel(float* out)
{
    __shared__ float s[TN];
    const int t    = threadIdx.x;
    const int lane = t & 31;
    for (int i = t; i < TN; i += blockDim.x) s[i] = (float)(i & 255);
    __syncthreads();      // barrier; Module 9 makes this precise.

    int idx;
    switch (P) {
        case P_ID:   idx = lane;      break;
        case P_S2:   idx = 2 * lane;  break;
        case P_S3:   idx = 3 * lane;  break;
        case P_S4:   idx = 4 * lane;  break;
        case P_S8:   idx = 8 * lane;  break;
        case P_S16:  idx = 16 * lane; break;
        case P_S32:  idx = 32 * lane; break;
        case P_HALF: idx = lane / 2;  break;
        case P_ZERO: idx = 0;         break;
        default:     idx = 33 * lane; break;
    }
    idx &= (TN - 1);

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    #pragma unroll 4
    for (int it = 0; it < ITERS; ++it) {
        int o = (it * 32) & (TN - 1);
        a0 += s[(idx + o      ) & (TN - 1)];
        a1 += s[(idx + o +  32) & (TN - 1)];
        a2 += s[(idx + o +  64) & (TN - 1)];
        a3 += s[(idx + o +  96) & (TN - 1)];
    }
    if (a0 + a1 + a2 + a3 == -12345.f) out[0] = 1.f;   // never true; defeats DCE
}

typedef void (*Launch)(float*);
static void L0(float* d){ bankKernel<P_ID  ><<<BLOCKS,THREADS>>>(d); }
static void L1(float* d){ bankKernel<P_S2  ><<<BLOCKS,THREADS>>>(d); }
static void L2(float* d){ bankKernel<P_S3  ><<<BLOCKS,THREADS>>>(d); }
static void L3(float* d){ bankKernel<P_S4  ><<<BLOCKS,THREADS>>>(d); }
static void L4(float* d){ bankKernel<P_S8  ><<<BLOCKS,THREADS>>>(d); }
static void L5(float* d){ bankKernel<P_S16 ><<<BLOCKS,THREADS>>>(d); }
static void L6(float* d){ bankKernel<P_S32 ><<<BLOCKS,THREADS>>>(d); }
static void L7(float* d){ bankKernel<P_HALF><<<BLOCKS,THREADS>>>(d); }
static void L8(float* d){ bankKernel<P_ZERO><<<BLOCKS,THREADS>>>(d); }
static void L9(float* d){ bankKernel<P_S33 ><<<BLOCKS,THREADS>>>(d); }

int main()
{
    printf("Module 7 / Example 1 - shared memory bank structure\n");
    printf("  %d banks x %d B; bank(addr) = (addr / %d) %% %d\n\n",
           BANKS, BANK_W, BANK_W, BANKS);

    // ---- Part A: the maps -------------------------------------------
    printf("Lane -> bank map, lanes 0..31, float array based at bank 0:\n");
    for (int p = 0; p < P_NUM; ++p) print_bank_map(p);
    printf("\n");

    int    deg[P_NUM];
    double model[P_NUM];
    printf("%-11s %8s %8s   %s\n", "pattern", "degree", "model", "why");
    printf("-----------------------------------------------------------------------\n");
    for (int p = 0; p < P_NUM; ++p) {
        deg[p] = conflict_degree(p, (int)sizeof(float), NULL);
        // Measured cost model on sm_89 for 4 B accesses: a conflict-free
        // warp access already occupies the shared pipeline for 2 cycles,
        // so cost is proportional to max(2, degree), not to degree.
        model[p] = (double)(deg[p] < 2 ? 2 : deg[p]) / 2.0;
        const char* note =
            (p == P_HALF) ? "lane pairs share a word -> broadcast" :
            (p == P_ZERO) ? "all 32 lanes share one word -> broadcast" :
            (p == P_S33)  ? "stride 33 is coprime with 32" :
            (deg[p] == 1) ? "stride coprime with 32" :
            (deg[p] == 32)? "worst case: one bank, 32 distinct words" :
                            "conflicted";
        printf("%-11s %8d %7.1fx   %s\n", PAT_NAME[p], deg[p], model[p], note);
    }

    // ---- Part B: measure ---------------------------------------------
    float* d = NULL;
    CHECK(cudaMalloc(&d, sizeof(float)));
    CHECK(cudaMemset(d, 0, sizeof(float)));

    Launch L[P_NUM] = { L0, L1, L2, L3, L4, L5, L6, L7, L8, L9 };
    double best[P_NUM];
    for (int p = 0; p < P_NUM; ++p) best[p] = 1e30;

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    // Duration-based clock warm-up: this GPU idles near 0.5 GHz and needs
    // seconds, not iterations, to reach a steady SM clock.
    clock_t w0 = clock();
    while ((double)(clock() - w0) / CLOCKS_PER_SEC < 4.0) {
        for (int p = 0; p < P_NUM; ++p) L[p](d);
        CHECK(cudaDeviceSynchronize());
    }

    // Spec 12: every configuration timed back to back, min over 4 sweeps.
    for (int sweep = 0; sweep < 4; ++sweep) {
        for (int q = 0; q < P_NUM; ++q) {
            const int p = (q + sweep) % P_NUM;   // rotate order: no config is
            // always measured right after the clock dip that follows a sync.
            L[p](d); CHECK(cudaDeviceSynchronize());          // warm-up launch
            CHECK(cudaEventRecord(e0));
            for (int i = 0; i < 20; ++i) L[p](d);
            CHECK(cudaEventRecord(e1));
            CHECK(cudaEventSynchronize(e1));
            float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= 20.f;
            if (ms < best[p]) best[p] = ms;
        }
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    printf("\n%-11s %8s %10s %11s %10s\n", "pattern", "degree", "ms", "measured", "model");
    printf("-----------------------------------------------------------------------\n");
    for (int p = 0; p < P_NUM; ++p)
        printf("%-11s %8d %10.4f %10.2fx %9.1fx\n",
               PAT_NAME[p], deg[p], best[p], best[p] / best[P_ID], model[p]);

    printf("\nRatios are the stable quantity; absolute ms move with the clock.\n");
    printf("Read the two rightmost columns together. Where they disagree,\n");
    printf("the hardware is right and the model is incomplete. See lesson.md.\n");

    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d));
    CHECK(cudaDeviceReset());
    printf("\nOVERALL: PASS\n");
    return 0;
}
