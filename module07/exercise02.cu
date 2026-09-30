// =====================================================================
// Module 7 / Exercise 2 : "32 lanes, 32 banks, and only one of them busy"
//
// GOAL
//   The kernel below stages a ROWS x 32 tile in shared memory (the
//   cooperative-load pattern from Module 6) and then has every thread
//   sweep its own row against a weight vector. The load phase is fine.
//   The compute phase is not: it is the single worst access pattern the
//   banking structure admits, and it is costing this kernel most of its
//   throughput.
//
//   You will produce two working alternatives and measure all three.
//
//   Layout NAIVE    : element (r, c) is stored at tile[r * 32 + c].
//   Layout PADDED   : you choose the pitch.  Extra shared memory allowed.
//   Layout SWIZZLED : the shared array must remain exactly
//                     ROWS * 32 floats. Not one word more.
//
//   Both alternatives must satisfy the SAME requirement:
//
//       in the compute phase, the 32 lanes of a warp must address 32
//       DISTINCT banks, without changing the algorithm, without changing
//       which thread produces which output, and without making the load
//       phase conflicted instead.
//
//   The load phase is shared by all three layouts: it writes element
//   (r, c) to whatever location your index function says it lives at.
//   Check that it stays conflict-free. It is easy to fix one phase by
//   breaking the other, and the harness will happily report a "fix" that
//   moved the cost rather than removing it.
//
// PREDICT BEFORE YOU RUN
//   1. The conflict degree of the compute-phase access in layout NAIVE.
//   2. Whether PADDED and SWIZZLED land within 5% of each other.
//   3. Which of the three has the lowest occupancy, and why the answer
//      is not "they all use one 32-wide tile, so they are the same".
//
// WHAT THE PROGRAM CHECKS
//   - all three kernels against a CPU reference (PASS/FAIL each),
//   - that SWIZZLED really did not grow the shared allocation,
//   - your hand-computed occupancy against the CUDA occupancy API,
//   - your predicted conflict degree against the measured slowdown.
//
// BUILD:  nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise02.exe exercise02.cu
// RUN:    .\exercise02.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
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

#define COLS     32            // tile width, one column per lane
#define ROWS    192            // = threads per block = 6 warps
#define REPS     64            // sweeps per thread: makes this shared-bound
#define BLOCKS  320            // 8 per SM on 40 SMs

__constant__ float W[COLS];    // warp-uniform index -> constant broadcast (M4)

// ---------------------------------------------------------------------
// TODO 1: The conflict degree of the compute-phase read in layout NAIVE.
//         One warp = 32 lanes = 32 consecutive rows, all reading the same
//         column c of a 32-wide tile. How many DISTINCT 4 B words does
//         the busiest bank have to deliver for that one instruction?
//         The harness compares this against the measured slowdown.
// ---------------------------------------------------------------------
static const int NAIVE_DEGREE = 0;   // YOUR CODE HERE (TODO 1)

// ---------------------------------------------------------------------
// TODO 2: The padded pitch. PAD_PITCH is the number of floats between
//         the start of row r and the start of row r+1. Choose it so that
//         the compute-phase read hits 32 distinct banks.
//         Must be >= COLS. Extra shared memory is permitted here.
// ---------------------------------------------------------------------
#define PAD_PITCH 32                 // YOUR CODE HERE (TODO 2)

// ---------------------------------------------------------------------
// The three index functions. `pitch` is the row stride in floats.
// ---------------------------------------------------------------------
enum Layout { L_NAIVE = 0, L_PADDED = 1, L_SWIZZLED = 2 };

template <int LAYOUT>
__host__ __device__ __forceinline__ int tile_index(int r, int c)
{
    if (LAYOUT == L_NAIVE)  return r * COLS + c;
    if (LAYOUT == L_PADDED) return r * PAD_PITCH + c;

    // -----------------------------------------------------------------
    // TODO 3: SWIZZLED. Return a location inside a tile of exactly
    //         ROWS * COLS floats, such that
    //           (a) for fixed c, the 32 values you return for 32
    //               consecutive r land in 32 distinct banks;
    //           (b) for fixed r, the 32 values you return for
    //               c = 0..31 land in 32 distinct banks;
    //           (c) the map (r, c) -> location is injective, so nothing
    //               is overwritten.
    //         One arithmetic operation is enough. It is not an add.
    // -----------------------------------------------------------------
    return r * COLS + c;         // YOUR CODE HERE (TODO 3)
}

// ---------------------------------------------------------------------
// TODO 4: How many blocks of `threads` threads, each asking for
//         `smemBytes` bytes of shared memory, can be resident on one SM
//         of this GPU at the same time? Use the three device limits the
//         caller passes in, and note two things the naive division misses:
//           - the driver reserves `reserve` bytes of shared memory per
//             resident block, on top of what the block asked for;
//           - the total is rounded UP to a multiple of the allocation
//             granularity `gran`.
//         Return 0 for the unimplemented stub.
//
//         The harness compares your answer with
//         cudaOccupancyMaxActiveBlocksPerMultiprocessor.
// ---------------------------------------------------------------------
static int max_blocks_per_sm(int smemBytes, int threads,
                             int smemPerSM, int threadsPerSM, int blocksPerSM,
                             int reserve, int gran)
{
    (void)smemBytes; (void)threads; (void)smemPerSM;
    (void)threadsPerSM; (void)blocksPerSM; (void)reserve; (void)gran;
    return 0;    // YOUR CODE HERE (TODO 4)
}

// ============================ the kernel =============================
//
// Module 6 taught this shape: cooperatively stage a tile in shared
// memory, barrier, then reuse it. Everything here is about WHERE in
// shared memory each element lands.

template <int LAYOUT, int PITCH>
__global__ void rowSweep(const float* __restrict__ in, float* __restrict__ out)
{
    __shared__ float tile[ROWS * PITCH];

    const int tid  = threadIdx.x;
    const int base = blockIdx.x * (ROWS * COLS);

    // ---- load phase: flat, so consecutive lanes get consecutive c ----
    for (int i = tid; i < ROWS * COLS; i += ROWS) {
        const int r = i / COLS;
        const int c = i - r * COLS;
        tile[tile_index<LAYOUT>(r, c)] = in[base + i];
    }
    __syncthreads();            // barrier; Module 9 makes this precise.

    // ---- compute phase -----------------------------------------------
    // Thread `tid` owns output row `tid`. Sweep REPS shifted windows so
    // that the kernel is bound by shared-memory traffic and not by the
    // global load that filled the tile.
    float acc = 0.f;
    #pragma unroll 1
    for (int j = 0; j < REPS; ++j) {
        const int rr = (tid + j) % ROWS;          // a different row each j
        // Four columns per iteration, 8 floats apart, and the loop itself
        // is NOT unrolled. Both details are deliberate: they keep every
        // shared access a separate scalar 4 B load, so the bank analysis
        // of Example 1 applies to exactly the instruction being measured.
        // Let the compiler fuse 32 adjacent columns into 128-bit loads and
        // you are timing an instruction you did not analyse.
        #pragma unroll 1
        for (int c = 0; c < 8; ++c) {
            acc += tile[tile_index<LAYOUT>(rr, c     )] * W[(c      + j) & (COLS - 1)];
            acc += tile[tile_index<LAYOUT>(rr, c +  8)] * W[(c +  8 + j) & (COLS - 1)];
            acc += tile[tile_index<LAYOUT>(rr, c + 16)] * W[(c + 16 + j) & (COLS - 1)];
            acc += tile[tile_index<LAYOUT>(rr, c + 24)] * W[(c + 24 + j) & (COLS - 1)];
        }
    }
    out[blockIdx.x * ROWS + tid] = acc;
}

typedef void (*Launch)(const float*, float*);
static void LN(const float* i, float* o){ rowSweep<L_NAIVE,    COLS     ><<<BLOCKS,ROWS>>>(i,o); }
static void LP(const float* i, float* o){ rowSweep<L_PADDED,   PAD_PITCH><<<BLOCKS,ROWS>>>(i,o); }
static void LS(const float* i, float* o){ rowSweep<L_SWIZZLED, COLS     ><<<BLOCKS,ROWS>>>(i,o); }

static const char* LNAME[3] = { "NAIVE", "PADDED", "SWIZZLED" };
static const int   LPITCH[3] = { COLS, PAD_PITCH, COLS };

// ========================== CPU reference ============================
static void cpu_reference(const float* in, const float* w, float* out)
{
    for (int b = 0; b < BLOCKS; ++b) {
        const float* tin = in + (size_t)b * ROWS * COLS;
        for (int r = 0; r < ROWS; ++r) {
            double acc = 0.0;
            for (int j = 0; j < REPS; ++j) {
                const int rr = (r + j) % ROWS;
                for (int c = 0; c < COLS; ++c)
                    acc += (double)tin[rr * COLS + c] * (double)w[(c + j) & (COLS - 1)];
            }
            out[b * ROWS + r] = (float)acc;
        }
    }
}

// =============================== main ================================
int main()
{
    printf("Module 7 / Exercise 2 - a 32-way conflict, and two ways out\n\n");

    if (NAIVE_DEGREE <= 0)   { printf("Set TODO 1 first.\n"); return 0; }
    if (PAD_PITCH < COLS)    { printf("PAD_PITCH must be >= %d.\n", COLS); return 0; }
    if (PAD_PITCH == COLS)   { printf("Set TODO 2 first.\n"); return 0; }
    if (tile_index<L_SWIZZLED>(1, 0) == tile_index<L_NAIVE>(1, 0) &&
        tile_index<L_SWIZZLED>(1, 1) == tile_index<L_NAIVE>(1, 1))
                             { printf("Set TODO 3 first.\n"); return 0; }

    // ---- structural test of TODO 3, before anything is launched ------
    int swizFail = 0;
    {
        static char used[ROWS * COLS];
        for (int i = 0; i < ROWS * COLS; ++i) used[i] = 0;
        for (int r = 0; r < ROWS && !swizFail; ++r)
            for (int c = 0; c < COLS; ++c) {
                const int loc = tile_index<L_SWIZZLED>(r, c);
                if (loc < 0 || loc >= ROWS * COLS) { swizFail = 1; break; }   // in range
                if (used[loc]) { swizFail = 2; break; }                       // injective
                used[loc] = 1;
            }
        for (int c = 0; c < COLS && !swizFail; ++c) {          // column read, 32 rows
            int seen[32] = {0};
            for (int r = 0; r < 32; ++r) seen[tile_index<L_SWIZZLED>(r, c) % 32] = 1;
            int n = 0; for (int b = 0; b < 32; ++b) n += seen[b];
            if (n != 32) swizFail = 3;
        }
        for (int r = 0; r < ROWS && !swizFail; ++r) {          // row write, 32 cols
            int seen[32] = {0};
            for (int c = 0; c < COLS; ++c) seen[tile_index<L_SWIZZLED>(r, c) % 32] = 1;
            int n = 0; for (int b = 0; b < 32; ++b) n += seen[b];
            if (n != 32) swizFail = 4;
        }
    }
    const char* SWIZMSG[5] = { "ok", "out of range", "not injective",
                               "column read still conflicted", "row write now conflicted" };
    printf("TODO 3 structural test: %s\n", SWIZMSG[swizFail]);

    // ---- device limits ------------------------------------------------
    int smemPerSM = 0, threadsPerSM = 0, blocksPerSM = 0;
    CHECK(cudaDeviceGetAttribute(&smemPerSM,    cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    CHECK(cudaDeviceGetAttribute(&threadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor,      0));
    CHECK(cudaDeviceGetAttribute(&blocksPerSM,  cudaDevAttrMaxBlocksPerMultiprocessor,       0));
    int reserve = 0;
    CHECK(cudaDeviceGetAttribute(&reserve, cudaDevAttrReservedSharedMemoryPerBlock, 0));
    const int GRAN = 128;      // shared allocation granularity on sm_89

    if (max_blocks_per_sm(20480, 192, smemPerSM, threadsPerSM, blocksPerSM,
                          reserve, GRAN) == 0) {
        printf("Set TODO 4 first.\n"); return 0;
    }

    // ---- data ---------------------------------------------------------
    const size_t NIN  = (size_t)BLOCKS * ROWS * COLS;
    const size_t NOUT = (size_t)BLOCKS * ROWS;
    float* hIn  = (float*)malloc(NIN  * sizeof(float));
    float* hOut = (float*)malloc(NOUT * sizeof(float));
    float* hRef = (float*)malloc(NOUT * sizeof(float));
    float  hW[COLS];
    for (size_t i = 0; i < NIN; ++i) hIn[i] = (float)((int)(i % 17) - 8) * 0.25f;
    for (int c = 0; c < COLS; ++c)   hW[c]  = (float)((c % 5) - 2) * 0.5f;

    float *dIn = NULL, *dOut = NULL;
    CHECK(cudaMalloc(&dIn,  NIN  * sizeof(float)));
    CHECK(cudaMalloc(&dOut, NOUT * sizeof(float)));
    CHECK(cudaMemcpy(dIn, hIn, NIN * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(W, hW, sizeof(hW)));
    cpu_reference(hIn, hW, hRef);

    Launch L[3] = { LN, LP, LS };
    double best[3] = { 1e30, 1e30, 1e30 };

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    clock_t w0 = clock();
    while ((double)(clock() - w0) / CLOCKS_PER_SEC < 4.0) {
        for (int k = 0; k < 3; ++k) L[k](dIn, dOut);
        CHECK(cudaDeviceSynchronize());
    }
    for (int sweep = 0; sweep < 4; ++sweep)
        for (int q = 0; q < 3; ++q) {
            const int k = (q + sweep) % 3;       // rotate order: no config is
            // always measured right after the clock dip that follows a sync.
            L[k](dIn, dOut); CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(e0));
            for (int i = 0; i < 20; ++i) L[k](dIn, dOut);
            CHECK(cudaEventRecord(e1));
            CHECK(cudaEventSynchronize(e1));
            float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= 20.f;
            if (ms < best[k]) best[k] = ms;
        }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // ---- validate, second pass ---------------------------------------
    int bad[3] = { 0, 0, 0 };
    for (int k = 0; k < 3; ++k) {
        CHECK(cudaMemset(dOut, 0, NOUT * sizeof(float)));
        L[k](dIn, dOut);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hOut, dOut, NOUT * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < NOUT; ++i)
            if (fabsf(hOut[i] - hRef[i]) > 1e-5f * fmaxf(1.0f, fabsf(hRef[i]))) { bad[k]++; }
    }

    // ---- occupancy ----------------------------------------------------
    printf("\nSM limits: shared %d B, threads %d, blocks %d;\n"
           "per-block driver reserve %d B, allocation granularity %d B\n",
           smemPerSM, threadsPerSM, blocksPerSM, reserve, GRAN);
    printf("\n%-9s %6s %9s %9s %8s %9s %11s %9s\n",
           "layout", "pitch", "smem/blk", "yours", "cudaOcc", "occupancy", "ms", "vs best");
    printf("---------------------------------------------------------------------------------------\n");

    double bestMs = best[0];
    for (int k = 1; k < 3; ++k) if (best[k] < bestMs) bestMs = best[k];

    int occFail = 0;
    for (int k = 0; k < 3; ++k) {
        const int smem = ROWS * LPITCH[k] * (int)sizeof(float);
        int api = 0;
        if (k == 0) CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api, (const void*)rowSweep<L_NAIVE,COLS>,          ROWS, 0));
        if (k == 1) CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api, (const void*)rowSweep<L_PADDED,PAD_PITCH>,    ROWS, 0));
        if (k == 2) CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api, (const void*)rowSweep<L_SWIZZLED,COLS>,       ROWS, 0));
        const int mine = max_blocks_per_sm(smem, ROWS, smemPerSM, threadsPerSM,
                                           blocksPerSM, reserve, GRAN);
        if (mine != api) occFail = 1;
        printf("%-9s %6d %9d %9d %8d %8.1f%% %11.4f %8.2fx\n",
               LNAME[k], LPITCH[k], smem, mine, api,
               100.0 * api * ROWS / threadsPerSM, best[k], best[k] / bestMs);
    }

    // ---- results -------------------------------------------------------
    printf("\n%-9s %9s %12s\n", "layout", "numerics", "vs NAIVE");
    printf("--------------------------------------\n");
    for (int k = 0; k < 3; ++k)
        printf("%-9s %9s %11.2fx\n", LNAME[k], bad[k] ? "FAIL" : "PASS", best[0] / best[k]);

    // TODO 1 is checked structurally, against the layout the harness owns.
    int trueDegree;
    {
        int nWords[32]; int seen[32][32];
        for (int b = 0; b < 32; ++b) nWords[b] = 0;
        for (int r = 0; r < 32; ++r) {               // one column, 32 rows
            const int loc  = tile_index<L_NAIVE>(r, 0);
            const int bank = loc % 32;
            bool dup = false;
            for (int k = 0; k < nWords[bank]; ++k) if (seen[bank][k] == loc) dup = true;
            if (!dup) seen[bank][nWords[bank]++] = loc;
        }
        trueDegree = 0;
        for (int b = 0; b < 32; ++b) if (nWords[b] > trueDegree) trueDegree = nWords[b];
    }
    const double measuredSpeedup = best[0] / bestMs;
    const bool   degOk = (NAIVE_DEGREE == trueDegree);
    printf("\nTODO 1: you said the NAIVE compute-phase read has degree %d. %s\n",
           NAIVE_DEGREE, degOk ? "Correct." : "It does not.");
    printf("Degree D costs at most D/2 times a conflict-free access here, so\n");
    printf("the shared-memory term could be up to %.1fx. The whole kernel\n",
           trueDegree / 2.0);
    printf("measured %.2fx. The gap is everything in the loop that is not a\n",
           measuredSpeedup);
    printf("shared load: the global fill, the FFMAs, the loop control. D/2\n");
    printf("bounds the shared-memory term, never the kernel.\n");

    const bool swizNoGrowth = (LPITCH[2] == COLS);
    const bool allPass = !bad[0] && !bad[1] && !bad[2];
    const bool pass = allPass && !swizFail && !occFail && degOk && swizNoGrowth
                      && (best[0] / best[1] > 2.0) && (best[0] / best[2] > 2.0);

    printf("\nnumerics: %s   swizzle structure: %s   swizzle size unchanged: %s\n",
           allPass ? "PASS" : "FAIL", swizFail ? "FAIL" : "ok", swizNoGrowth ? "ok" : "FAIL");
    printf("occupancy model: %s   degree prediction: %s\n",
           occFail ? "FAIL" : "ok", degOk ? "ok" : "MISSED");
    printf("\nOVERALL: %s\n", pass ? "PASS" : "FAIL");

    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(dIn)); CHECK(cudaFree(dOut));
    free(hIn); free(hOut); free(hRef);
    CHECK(cudaDeviceReset());
    return 0;
}
