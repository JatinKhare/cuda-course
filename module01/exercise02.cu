// =====================================================================
// Module 1 / Exercise 2 : "The staircase: what a wave tail costs"
//
// TYPE: predict-the-behavior + performance reasoning.
//
// GOAL
//   A grid is not dispatched "all at once". The GigaThread work
//   distributor places blocks onto SMs only where a free block slot
//   exists. The set of blocks that can be resident simultaneously is one
//   WAVE. Everything beyond a wave waits for a slot to free.
//
//   This program runs a deliberately compute-bound kernel (a long chain
//   of dependent FMAs, no memory traffic to speak of) at six grid sizes:
//
//       half a wave, one wave minus one, exactly one wave,
//       one wave plus one, exactly two waves, two waves plus one.
//
//   Every block does exactly the same amount of work, so the block count
//   is proportional to total work done.
//
// BEFORE YOU BUILD -- write your predictions down. You will be graded by
// your own honesty, not by the program.
//
//   P1. t(one wave + 1 block) / t(one wave) = ?
//   P2. t(half a wave)        / t(one wave) = ?
//   P3. t(two waves)          / t(one wave + 1 block) = ?
//   P4. Which configuration achieves the highest GFLOP/s, and which the
//       lowest? Why?
//   P5. This kernel is compute-bound on purpose. Sketch what would
//       change in the table if it were memory-bandwidth-bound instead,
//       and say why that would make the wave structure harder to see.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
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

static const int   THREADS  = 1024;   // 32 warps: one block alone saturates an SM
static const int   ROUNDS   = 16384;  // dependent FMAs per thread
static const int   ITERS    = 20;     // timed iterations
static const int   WARMUP   = 5;      // untimed launches before each measurement
static const int   REPEATS  = 4;      // full sweeps; we keep the fastest per config
static const int   NCFG     = 6;
static const float A_COEF   = 1.0000001f;
static const float B_COEF   = 1.0e-7f;

// `cycles` is written only by thread 0 of block 0, and only to let the
// host recover the SM clock that was actually in effect during the
// measurement. cudaDevAttrClockRate reports a nominal clock; a laptop
// part runs well above or below it depending on thermal state, so a
// "% of peak" computed from the nominal number is meaningless.
__global__ void fma_chain(const float* __restrict__ in,
                          float* __restrict__ out,
                          long long* __restrict__ cycles,
                          int rounds, float a, float b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    long long c0 = clock64();
    float v = in[i];
    for (int r = 0; r < rounds; ++r)
        v = fmaf(v, a, b);
    out[i] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        cycles[0] = clock64() - c0;
}

static float host_chain(float v, int rounds, float a, float b)
{
    for (int r = 0; r < rounds; ++r) v = fmaf(v, a, b);
    return v;
}

int main(void)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    const int nSMs = p.multiProcessorCount;

    int smClockKHz = 0;
    CHECK(cudaDeviceGetAttribute(&smClockKHz, cudaDevAttrClockRate, dev));
    // 4 processing blocks x 32 FP32 lanes = 128 FMA lanes per SM, 2 FLOP each.
    const double peakGFLOPs = (double)nSMs * 128.0 * 2.0 * (smClockKHz * 1.0e3) / 1.0e9;

    // -----------------------------------------------------------------
    // TODO 1: How many blocks of `THREADS` threads make up exactly ONE
    //         wave of this kernel on this device?
    //
    //         `blocksPerSM` must be how many blocks of THIS kernel the
    //         hardware will actually hold resident on a single SM. That
    //         is a property of the kernel, not only of the device: it
    //         depends on the block's thread, register and shared-memory
    //         footprint, and the binding limit may be any of them.
    //         `p.maxBlocksPerMultiProcessor` is an unconditional ceiling
    //         that ignores all of that, so it is not the answer. Have the
    //         runtime compute the real number for this kernel, then
    //         derive `waveBlocks`.
    // -----------------------------------------------------------------
    int blocksPerSM = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocksPerSM, (const void*)fma_chain, THREADS, 0));
    const int waveBlocks = blocksPerSM * nSMs;     // 1 * 40 = 40    
    const int    waves   = (blocksPerSM + waveBlocks - 1) / waveBlocks;
    const double waveEff = (double)blocksPerSM / ((double)waves * waveBlocks);
    printf("=== %s : %d SMs, SM clock %.3f GHz ===\n",
           p.name, nSMs, smClockKHz / 1.0e6);
    printf("blockDim = %d (%d warps), %d dependent FMAs per thread\n",
           THREADS, THREADS / 32, ROUNDS);
    printf("Resident blocks per SM (occupancy API) : %d\n", blocksPerSM);
    printf("One wave                               : %d blocks\n", waveBlocks);
    printf("Peak FP32 at the API-reported clock    : %.1f GFLOP/s\n\n", peakGFLOPs);

    if (waveBlocks <= 0) { printf("Set TODO 1 first.\n"); return 0; }

    int cfg[NCFG] = { waveBlocks / 2, waveBlocks - 1, waveBlocks,
                      waveBlocks + 1, 2 * waveBlocks, 2 * waveBlocks + 1 };

    int maxBlocks = 0;
    for (int c = 0; c < NCFG; ++c) if (cfg[c] > maxBlocks) maxBlocks = cfg[c];
    const size_t maxN = (size_t)maxBlocks * THREADS;

    float *d_in = nullptr, *d_out = nullptr;
    long long* d_cyc = nullptr;
    CHECK(cudaMalloc(&d_cyc, sizeof(long long)));
    CHECK(cudaMalloc(&d_in,  maxN * sizeof(float)));
    CHECK(cudaMalloc(&d_out, maxN * sizeof(float)));
    float* h_in  = (float*)malloc(maxN * sizeof(float));
    float* h_out = (float*)malloc(maxN * sizeof(float));
    if (!h_in || !h_out) { fprintf(stderr, "host alloc failed\n"); return 1; }
    srand(2024);
    for (size_t i = 0; i < maxN; ++i)
        h_in[i] = 1.0f + (float)(rand() % 1000) * 1.0e-4f;
    CHECK(cudaMemcpy(d_in, h_in, maxN * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    double ms_of[NCFG];
    long long cyc_of[NCFG];

    // ---- PASS 1: timing only ---------------------------------------
    // All timing happens back to back, with no host-side work in between:
    // host work between measurements lets the GPU drop to a lower clock
    // state and silently corrupts the comparison. This is a laptop part
    // whose SM clock swings ~3x between the idle state and the sustained
    // thermally-limited state, so we sweep the whole configuration list
    // REPEATS times and keep the fastest observation of each. Min-of-N is
    // robust against both the initial clock ramp and later throttling;
    // the *ratios* between configurations are what this exercise measures.
    for (int c = 0; c < NCFG; ++c) { ms_of[c] = 1.0e30; cyc_of[c] = 0; }

    for (int rep = 0; rep < REPEATS; ++rep) {
        for (int c = 0; c < NCFG; ++c) {
            const int nBlocks = cfg[c];

            for (int w = 0; w < WARMUP; ++w)
                fma_chain<<<nBlocks, THREADS>>>(d_in, d_out, d_cyc, ROUNDS, A_COEF, B_COEF);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaEventRecord(t0));
            for (int it = 0; it < ITERS; ++it)
                fma_chain<<<nBlocks, THREADS>>>(d_in, d_out, d_cyc, ROUNDS, A_COEF, B_COEF);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaGetLastError());
            CHECK(cudaEventSynchronize(t1));

            float ms_total = 0.0f;
            CHECK(cudaEventElapsedTime(&ms_total, t0, t1));
            const double ms = ms_total / ITERS;
            if (ms < ms_of[c]) {
                ms_of[c] = ms;
                CHECK(cudaMemcpy(&cyc_of[c], d_cyc, sizeof(long long),
                                 cudaMemcpyDeviceToHost));
            }
        }
    }

    // ---- PASS 2: correctness ---------------------------------------
    int allPass = 1;
    for (int c = 0; c < NCFG; ++c) {
        const int nBlocks = cfg[c];
        const size_t n = (size_t)nBlocks * THREADS;

        CHECK(cudaMemset(d_out, 0, maxN * sizeof(float)));
        fma_chain<<<nBlocks, THREADS>>>(d_in, d_out, d_cyc, ROUNDS, A_COEF, B_COEF);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

        int bad = 0, unwritten = 0;
        for (size_t i = 0; i < n; ++i) {
            if (h_out[i] == 0.0f) { ++unwritten; continue; }  // never a valid result
            if (i % 64 == 0) {
                float ref = host_chain(h_in[i], ROUNDS, A_COEF, B_COEF);
                if (fabsf(h_out[i] - ref) > 1e-5f * fmaxf(1.0f, fabsf(ref))) {
                    if (bad < 3)
                        printf("  [MISMATCH] cfg %d i=%zu gpu=%.9g cpu=%.9g\n",
                               c, i, h_out[i], ref);
                    ++bad;
                }
            }
        }
        if (unwritten)
            printf("  [UNWRITTEN] cfg %d: %d elements never stored\n", c, unwritten);
        if (bad || unwritten) allPass = 0;
    }

    // ---- report -----------------------------------------------------
    // Recover the SM clock that was in effect. Use the exactly-one-wave
    // configuration: there, block 0 is resident for the whole kernel, so
    // its cycle count spans the same interval the host timed.
    double ms_one_wave = 0.0, measuredGHz = 0.0;
    for (int c = 0; c < NCFG; ++c) if (cfg[c] == waveBlocks) {
        ms_one_wave  = ms_of[c];
        measuredGHz  = (double)cyc_of[c] / (ms_of[c] * 1.0e6);
    }
    const double measuredPeakGFLOPs = (double)nSMs * 128.0 * 2.0 * measuredGHz;
    printf("Measured SM clock during the sweep     : %.3f GHz "
           "(API reports %.3f GHz)\n", measuredGHz, smClockKHz / 1.0e6);
    printf("Peak FP32 at the measured clock        : %.1f GFLOP/s\n\n",
           measuredPeakGFLOPs);

    printf("%8s %7s %9s %10s %11s %8s %10s %9s\n",
           "blocks", "waves", "wave eff", "ms", "GFLOP/s", "%peak", "ms/wave", "vs wave");
    printf("-------- ------- --------- ---------- ----------- -------- ---------- ---------\n");

    for (int c = 0; c < NCFG; ++c) {
        const int nBlocks = cfg[c];
        const double ms = ms_of[c];

        // -------------------------------------------------------------
        // 
        // : `waves`   -- how many waves this grid takes.
        //         `waveEff` -- the fraction of the machine's resident
        //                      block slots that carry real work, averaged
        //                      over the whole launch (1.0 = perfect).
        //         The self-check below rejects an off-by-one.
        // -------------------------------------------------------------
        const int waves = (nBlocks + waveBlocks - 1) / waveBlocks;
        const double waveEff = (double)nBlocks / ((double)waves * waveBlocks);
        printf("%d, %d", nBlocks, waveBlocks);
        if (waves <= 0) { printf("Set TODO 2 first.\n"); break; }
        if (!((long long)waves * waveBlocks >= nBlocks &&
              (long long)(waves - 1) * waveBlocks < nBlocks)) {
            printf("[FAIL] waves=%d cannot hold %d blocks in waves of %d\n",
                   waves, nBlocks, waveBlocks);
            allPass = 0;
        }
        if (!(waveEff > 0.0 && waveEff <= 1.0)) {
            printf("[FAIL] waveEff=%.3f is not a fraction in (0,1]\n", waveEff);
            allPass = 0;
        }

        // -------------------------------------------------------------
        // TODO 3: Achieved floating-point throughput, in GFLOP/s, for
        //         this configuration. Count only work the machine was
        //         actually asked to do. Idle block slots are not work.
        //         (Careful with what one FMA is worth.)
        // -------------------------------------------------------------
        double gflops = 0;   // YOUR CODE HERE (TODO 3)
        
        printf("%8d %7d %8.1f%% %10.4f %11.1f %7.1f%% %10.4f %8.2fx\n",
               nBlocks, waves, 100.0 * waveEff, ms, gflops,
               100.0 * gflops / measuredPeakGFLOPs, ms / waves, ms / ms_one_wave);
    }

    printf("\nms/wave is near-constant: for a compute-bound kernel the wave, not\n");
    printf("the block count, is the unit of GPU wall-clock time.\n");
    printf("%%peak tracks 'wave eff' almost exactly -- the idle block slots in an\n");
    printf("under-filled wave are throughput thrown away.\n");

    printf("\n%s\n", allPass ? "PASS" : "FAIL");

    free(h_in); free(h_out);
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    CHECK(cudaFree(d_cyc));

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaDeviceReset());
    return allPass ? 0 : 1;
}
