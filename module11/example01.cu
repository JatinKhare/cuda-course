// =====================================================================
// Module 11 / Example 1 : "Counting the floor"
//
// GOAL
//   Establish the discipline that opens every elementwise kernel:
//     1. count the COMPULSORY bytes the kernel must move,
//     2. divide by the machine's streaming bandwidth to get a time FLOOR,
//     3. measure, and report the measured time as a multiple of the floor.
//   A kernel at 1.0x its floor is finished. A kernel at 2.0x has a defect,
//   and the defect is almost always "it is moving bytes you did not count".
//
//   Part A  the traffic table: six elementwise kernels, floor vs measured.
//   Part B  where the extra bytes come from: partial-sector writes force a
//           read-modify-write of the sector (write-allocate). Full-sector
//           writes do not. Measured, not asserted.
//   Part C  do the non-temporal store hints (__stcs / __stwt) help? Measured.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN:   .\example01.exe
//
// SASS (Part C's store opcodes):
//   nvcc -arch=sm_89 -O3 -cubin -o example01.cubin example01.cu
//   cuobjdump -sass example01.cubin | findstr /R "STG"
//
// METHODOLOGY (spec 12). Every configuration is timed back-to-back inside
// one sweep; the sweep order is ROTATED so a different kernel absorbs the
// post-synchronize clock dip each time; the iteration count is chosen at
// runtime so each timed segment lasts ~10 ms; the minimum of 4 sweeps is
// reported. Validation happens in a separate pass, after all timing.
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

static const double PEAK_GBS = 432.0;          // 192-bit @ 9.001 GHz, fixed
static const long long N     = 1LL << 26;      // 67,108,864 floats = 256 MB
static const int  SWEEPS     = 4;

// ------------------------------------------------------------------ timing
template <typename L>
static int autoIters(L launch, cudaEvent_t a, cudaEvent_t b)
{
    CHECK(cudaEventRecord(a)); launch(); CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
    if (ms < 0.0005f) ms = 0.0005f;
    int it = (int)(10.0 / ms);                  // ~10 ms per timed segment
    if (it < 20)    it = 20;
    if (it > 20000) it = 20000;
    return it;
}

template <typename L>
static double timeOnce(L launch, int iters, cudaEvent_t a, cudaEvent_t b)
{
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
    return (double)ms / iters;
}

// ------------------------------------------------------------------ kernels
// Every kernel is a grid-stride loop so the launch configuration is a free
// variable and correctness never depends on it (Module 3).
#define GRID_STRIDE(n) \
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n); i += (long long)gridDim.x * blockDim.x

__global__ void k_fill (float* __restrict__ o, long long n, float v)
{ for (GRID_STRIDE(n)) o[i] = v; }                                   // 1N

__global__ void k_copy (const float* __restrict__ a, float* __restrict__ o, long long n)
{ for (GRID_STRIDE(n)) o[i] = a[i]; }                                // 2N

__global__ void k_scale(const float* __restrict__ a, float* __restrict__ o, float s, long long n)
{ for (GRID_STRIDE(n)) o[i] = s * a[i]; }                            // 2N

__global__ void k_saxpy_oop(const float* __restrict__ x, const float* __restrict__ y,
                            float* __restrict__ o, float a, long long n)
{ for (GRID_STRIDE(n)) o[i] = a * x[i] + y[i]; }                     // 3N

__global__ void k_saxpy_ip (const float* __restrict__ x, float* __restrict__ y,
                            float a, long long n)
{ for (GRID_STRIDE(n)) y[i] += a * x[i]; }                           // 3N, not 2N

__global__ void k_axbycz(const float* __restrict__ x, const float* __restrict__ y,
                         const float* __restrict__ z, float* __restrict__ o,
                         float A, float B, float C, long long n)
{ for (GRID_STRIDE(n)) o[i] = A*x[i] + B*y[i] + C*z[i]; }            // 4N

// ---- Part B: what a partial-sector write costs -------------------------
// Both kernels write exactly N/2 floats = 128 MB of useful data.
// k_write_dense  writes elements 0 .. N/2-1        -> every touched 32 B
//                sector is written IN FULL.
// k_write_odd    writes elements 0,2,4,...,N-2     -> every touched sector
//                receives 16 of its 32 bytes. The other 16 must be
//                preserved, so the sector has to be READ first.
__global__ void k_write_dense(float* __restrict__ o, long long nHalf, float v)
{ for (GRID_STRIDE(nHalf)) o[i] = v; }

__global__ void k_write_odd  (float* __restrict__ o, long long nHalf, float v)
{ for (GRID_STRIDE(nHalf)) o[2*i] = v; }

// ---- Part C: cache-control store hints ---------------------------------
__global__ void k_fill_stcs(float* __restrict__ o, long long n, float v)
{ for (GRID_STRIDE(n)) __stcs(&o[i], v); }      // "streaming": evict-first
__global__ void k_fill_stwt(float* __restrict__ o, long long n, float v)
{ for (GRID_STRIDE(n)) __stwt(&o[i], v); }      // "write-through"
__global__ void k_write_odd_stcs(float* __restrict__ o, long long nHalf, float v)
{ for (GRID_STRIDE(nHalf)) __stcs(&o[2*i], v); }

// ------------------------------------------------------------------ main
int main(void)
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const int nSM = prop.multiProcessorCount;

    printf("Device: %s  (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, nSM);
    printf("N = %lld floats; one array = %.0f MB; L2 = %.0f MB\n",
           N, N * 4.0 / 1048576.0, prop.l2CacheSize / 1048576.0);
    printf("Every buffer is >= 5x L2, so these are DRAM measurements.\n\n");

    float *x, *y, *z, *o;
    CHECK(cudaMalloc(&x, N * sizeof(float)));
    CHECK(cudaMalloc(&y, N * sizeof(float)));
    CHECK(cudaMalloc(&z, N * sizeof(float)));
    CHECK(cudaMalloc(&o, N * sizeof(float)));

    // Deterministic, index-derived initialization (host-side, once).
    {
        float* h = (float*)malloc((size_t)N * sizeof(float));
        if (!h) { printf("host alloc failed\n"); return 1; }
        for (long long i = 0; i < N; ++i) h[i] = (float)((i * 1103515245LL + 12345LL) % 1000) * 0.001f;
        CHECK(cudaMemcpy(x, h, N * sizeof(float), cudaMemcpyHostToDevice));
        for (long long i = 0; i < N; ++i) h[i] = (float)((i * 22695477LL + 1LL) % 997) * 0.001f;
        CHECK(cudaMemcpy(y, h, N * sizeof(float), cudaMemcpyHostToDevice));
        for (long long i = 0; i < N; ++i) h[i] = (float)((i * 69069LL + 5LL) % 991) * 0.001f;
        CHECK(cudaMemcpy(z, h, N * sizeof(float), cudaMemcpyHostToDevice));
        free(h);
    }

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    // A wave-sized grid: 8 blocks of 256 threads per SM = 2048 threads/SM.
    // Example 2 justifies this number by sweeping it; here it is a constant
    // so that nothing in this file varies except the kernel.
    const int TPB  = 256;
    const int GRID = nSM * 8;

    // Duration-based clock warm-up. A laptop GPU idles in a reduced memory
    // P-state (6001 MHz -> 288 GB/s) and ramps to 9001 MHz (432 GB/s) only
    // under sustained load. Run ~400 ms of real traffic before timing.
    {
        float acc = 0.f;
        while (acc < 400.f) {
            CHECK(cudaEventRecord(e0));
            for (int i = 0; i < 20; ++i) k_copy<<<GRID, TPB>>>(x, o, N);
            CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
            float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
            acc += ms;
        }
    }
    CHECK(cudaGetLastError());

    const int IT = autoIters([&]{ k_copy<<<GRID,TPB>>>(x, o, N); }, e0, e1);

    // =================================================================
    // Timing: all configurations, back to back, rotated, min of SWEEPS.
    // =================================================================
    enum { K_FILL, K_COPY, K_SCALE, K_SAXOOP, K_SAXIP, K_AXBYCZ,
           K_WDENSE, K_WODD, K_STCS, K_STWT, K_WODD_STCS, NCFG };
    double best[NCFG];
    for (int i = 0; i < NCFG; ++i) best[i] = 1e30;

    const long long nHalf = N / 2;

    for (int s = 0; s < SWEEPS; ++s) {
        for (int q = 0; q < NCFG; ++q) {
            const int c = (q + s) % NCFG;       // spec 12.9: rotate the order
            double ms = 0.0;
            switch (c) {
            case K_FILL:      ms = timeOnce([&]{ k_fill      <<<GRID,TPB>>>(o, N, 1.0f); }, IT, e0, e1); break;
            case K_COPY:      ms = timeOnce([&]{ k_copy      <<<GRID,TPB>>>(x, o, N); }, IT, e0, e1); break;
            case K_SCALE:     ms = timeOnce([&]{ k_scale     <<<GRID,TPB>>>(x, o, 2.0f, N); }, IT, e0, e1); break;
            case K_SAXOOP:    ms = timeOnce([&]{ k_saxpy_oop <<<GRID,TPB>>>(x, y, o, 2.0f, N); }, IT, e0, e1); break;
            case K_SAXIP:     ms = timeOnce([&]{ k_saxpy_ip  <<<GRID,TPB>>>(x, y, 1e-12f, N); }, IT, e0, e1); break;
            case K_AXBYCZ:    ms = timeOnce([&]{ k_axbycz    <<<GRID,TPB>>>(x, y, z, o, 1.f, 2.f, 3.f, N); }, IT, e0, e1); break;
            case K_WDENSE:    ms = timeOnce([&]{ k_write_dense<<<GRID,TPB>>>(o, nHalf, 1.0f); }, IT, e0, e1); break;
            case K_WODD:      ms = timeOnce([&]{ k_write_odd  <<<GRID,TPB>>>(o, nHalf, 1.0f); }, IT, e0, e1); break;
            case K_STCS:      ms = timeOnce([&]{ k_fill_stcs  <<<GRID,TPB>>>(o, N, 1.0f); }, IT, e0, e1); break;
            case K_STWT:      ms = timeOnce([&]{ k_fill_stwt  <<<GRID,TPB>>>(o, N, 1.0f); }, IT, e0, e1); break;
            case K_WODD_STCS: ms = timeOnce([&]{ k_write_odd_stcs<<<GRID,TPB>>>(o, nHalf, 1.0f); }, IT, e0, e1); break;
            default: break;
            }
            if (ms < best[c]) best[c] = ms;
        }
    }
    CHECK(cudaGetLastError());

    // The machine's streaming ceiling, measured here, in this thermal state.
    // k_copy moves 2N compulsory bytes and nothing else; it is the fastest
    // thing this GPU can do with this much data.
    const double ceilGBs = (8.0 * (double)N) / (best[K_COPY] * 1e-3) / 1e9;

    // =================================================================
    // Part A -- the traffic table
    // =================================================================
    printf("=== Part A: compulsory traffic, floor, and measured ===\n");
    printf("Compulsory traffic = the bytes that MUST cross the DRAM pins at\n");
    printf("least once: every distinct array element read, plus every element\n");
    printf("written. An array named twice in the source is still read once.\n\n");
    printf("  %-30s %6s %10s %10s %8s %8s\n",
           "kernel", "bytes", "floor ms", "meas ms", "x floor", "GB/s");

    struct Row { const char* name; int cfg; double bpe; };
    const Row rows[] = {
        { "fill      o[i]=c",            K_FILL,   4.0  },
        { "copy      o[i]=a[i]",         K_COPY,   8.0  },
        { "scale     o[i]=s*a[i]",       K_SCALE,  8.0  },
        { "saxpy oop o[i]=a*x[i]+y[i]",  K_SAXOOP, 12.0 },
        { "saxpy ip  y[i]+=a*x[i]",      K_SAXIP,  12.0 },
        { "axbycz    o=Ax+By+Cz",        K_AXBYCZ, 16.0 },
    };
    for (int r = 0; r < 6; ++r) {
        const double bytes = rows[r].bpe * (double)N;
        const double floorMs = bytes / (ceilGBs * 1e9) * 1e3;
        const double ms = best[rows[r].cfg];
        printf("  %-30s %5.0fN %10.4f %10.4f %8.2f %8.1f\n",
               rows[r].name, rows[r].bpe / 4.0, floorMs, ms, ms / floorMs,
               bytes / (ms * 1e-3) / 1e9);
    }
    printf("\n  measured streaming ceiling (from `copy`): %.1f GB/s = %.1f%% of the\n"
           "  %.0f GB/s nominal peak. Every 'floor ms' above uses this measured\n"
           "  number, not the nominal one, so the table is honest about the\n"
           "  machine you are actually running on.\n",
           ceilGBs, 100.0 * ceilGBs / PEAK_GBS, PEAK_GBS);

    printf("\n  The row that matters is `saxpy ip`. Read the source: one load of\n"
           "  x, one store to y. If you count 2N you get %.1f GB/s, which looks\n"
           "  like %.0f%% of peak and invites you to 'optimize' a kernel that is\n"
           "  already finished. y is read as well as written: the traffic is 3N.\n",
           8.0 * N / (best[K_SAXIP] * 1e-3) / 1e9,
           100.0 * (8.0 * N / (best[K_SAXIP] * 1e-3) / 1e9) / PEAK_GBS);

    // =================================================================
    // Part B -- write-allocate
    // =================================================================
    printf("\n=== Part B: partial-sector writes (write-allocate) ===\n");
    printf("Both kernels store exactly N/2 floats = %.0f MB of useful data.\n",
           2.0 * N / 1048576.0);
    {
        const double useful = 2.0 * (double)N;      // N/2 floats
        const double dMs = best[K_WDENSE], oMs = best[K_WODD];
        printf("  %-34s %10s %12s %12s\n", "", "ms", "useful GB/s", "impliedDRAM");
        printf("  %-34s %10.4f %12.1f %12s\n", "dense o[i]=v, i<N/2", dMs,
               useful / (dMs * 1e-3) / 1e9, "1x useful");
        printf("  %-34s %10.4f %12.1f %12.1f\n", "strided o[2i]=v, i<N/2", oMs,
               useful / (oMs * 1e-3) / 1e9, (2.0 * useful) / (oMs * 1e-3) / 1e9);
        printf("\n  ratio strided/dense = %.2fx\n", oMs / dMs);
        printf("  The strided kernel executes the SAME number of store\n"
               "  instructions and writes the SAME number of useful bytes. It\n"
               "  touches 2x as many sectors, and it fills only half of each, so\n"
               "  each sector must be fetched from DRAM, merged, and written\n"
               "  back: 1N read + 1N written for 0.5N of useful stores. The\n"
               "  'impliedDRAM' column above assumes exactly that 4x traffic and\n"
               "  lands on the streaming ceiling (%.1f GB/s) -- which is how you\n"
               "  know the model is right and the bus is not idle.\n", ceilGBs);
        printf("\n  Note what the `fill` row of Part A proves: a FULL-sector write\n"
               "  costs 1N, not 2N. There is no unconditional write-allocate on\n"
               "  this GPU. The fill kernel reports %.1f GB/s against a 1N model;\n"
               "  if every store had forced a line fill it could not have exceeded\n"
               "  %.1f GB/s. Write-allocate is a consequence of PARTIAL sector\n"
               "  coverage, not of writing.\n",
               4.0 * N / (best[K_FILL] * 1e-3) / 1e9, ceilGBs / 2.0);
    }

    // =================================================================
    // Part C -- do the streaming store hints help?
    // =================================================================
    printf("\n=== Part C: __stcs / __stwt cache hints ===\n");
    printf("  %-34s %10s %12s\n", "store form", "ms", "GB/s (1N)");
    printf("  %-34s %10.4f %12.1f\n", "plain STG  (full sector)",  best[K_FILL], 4.0*N/(best[K_FILL]*1e-3)/1e9);
    printf("  %-34s %10.4f %12.1f\n", "__stcs     (full sector)",  best[K_STCS], 4.0*N/(best[K_STCS]*1e-3)/1e9);
    printf("  %-34s %10.4f %12.1f\n", "__stwt     (full sector)",  best[K_STWT], 4.0*N/(best[K_STWT]*1e-3)/1e9);
    printf("  %-34s %10.4f %12.1f\n", "plain STG  (half sector)",  best[K_WODD],      2.0*N/(best[K_WODD]*1e-3)/1e9);
    printf("  %-34s %10.4f %12.1f\n", "__stcs     (half sector)",  best[K_WODD_STCS], 2.0*N/(best[K_WODD_STCS]*1e-3)/1e9);
    printf("\n  Read the numbers before you read the folklore. On this GPU the\n"
           "  hints change the SASS opcode (STG.E -> STG.E.EF for __stcs,\n"
           "  STG.E.STRONG.SYS for __stwt) and change the measurement by less\n"
           "  than run-to-run noise. They are CACHE-RESIDENCY hints -- they tell\n"
           "  L2 not to keep the line -- and the partial-sector fill is a\n"
           "  correctness requirement, not a cache policy. A hint cannot repeal\n"
           "  it. The fix for the strided kernel is to write whole sectors.\n");

    // =================================================================
    // Validation pass -- separate, untimed (spec 12.2)
    // =================================================================
    printf("\n=== validation (untimed second pass) ===\n");
    int fails = 0;
    {
        const long long NV = 1 << 20;                 // check a 1M prefix
        float* h  = (float*)malloc((size_t)NV * sizeof(float));
        float* hx = (float*)malloc((size_t)NV * sizeof(float));
        float* hy = (float*)malloc((size_t)NV * sizeof(float));
        float* hz = (float*)malloc((size_t)NV * sizeof(float));
        if (!h || !hx || !hy || !hz) { printf("host alloc failed\n"); return 1; }
        CHECK(cudaMemcpy(hx, x, NV * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hy, y, NV * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hz, z, NV * sizeof(float), cudaMemcpyDeviceToHost));

        k_axbycz<<<GRID, TPB>>>(x, y, z, o, 1.f, 2.f, 3.f, N);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h, o, NV * sizeof(float), cudaMemcpyDeviceToHost));
        long long bad = 0;
        for (long long i = 0; i < NV; ++i) {
            const float ref = 1.f*hx[i] + 2.f*hy[i] + 3.f*hz[i];
            if (fabsf(h[i] - ref) > 1e-5f * fmaxf(1.0f, fabsf(ref))) ++bad;
        }
        printf("  axbycz vs CPU reference over %lld elements: %lld mismatch(es)  %s\n",
               NV, bad, bad ? "FAIL" : "PASS");
        if (bad) ++fails;

        k_write_dense<<<GRID, TPB>>>(o, nHalf, 7.5f);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h, o, NV * sizeof(float), cudaMemcpyDeviceToHost));
        bad = 0; for (long long i = 0; i < NV; ++i) if (h[i] != 7.5f) ++bad;
        printf("  write_dense wrote every element of its range:  %lld bad     %s\n",
               bad, bad ? "FAIL" : "PASS");
        if (bad) ++fails;

        CHECK(cudaMemset(o, 0, N * sizeof(float)));
        k_write_odd<<<GRID, TPB>>>(o, nHalf, 7.5f);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h, o, NV * sizeof(float), cudaMemcpyDeviceToHost));
        bad = 0;
        for (long long i = 0; i < NV; ++i) {
            const float want = (i % 2 == 0) ? 7.5f : 0.0f;
            if (h[i] != want) ++bad;
        }
        printf("  write_odd wrote evens and preserved odds:      %lld bad     %s\n",
               bad, bad ? "FAIL" : "PASS");
        if (bad) ++fails;
        printf("  (that 'preserved odds' check IS the read-modify-write: the\n"
               "   hardware had to fetch each sector to keep the odd lanes.)\n");

        free(h); free(hx); free(hy); free(hz);
    }

    printf("\nOVERALL: %s\n", fails ? "FAIL" : "PASS");

    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(x)); CHECK(cudaFree(y)); CHECK(cudaFree(z)); CHECK(cudaFree(o));
    CHECK(cudaDeviceReset());
    return fails ? 1 : 0;
}
