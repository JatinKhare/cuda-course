// =====================================================================
// Module 5 / Exercise 3 -- SOLUTION
//   "Fix the layout, not the loop"  (row pitch / leading dimension)
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
// RUN:    .\exercise03_solution.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
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

static const double PEAK_GBS = 432.0;

static const long long ROWS       = 1000000LL; // one record per row
static const int       COLS_TOTAL = 65;        // 64 features + 1 label
static const int       COLS_USED  = 32;        // the kernel touches cols 0..31
static const int       PITCH_MAX  = 128;       // allocation is sized for this

// The record is 65 floats wide but the kernel reads a 32-float window out
// of each row. Consecutive windows are therefore separated by a gap of at
// least (pitch-32)*4 >= 132 B, which is more than one sector. That gap is
// what makes row-start alignment matter: no neighbouring warp is there to
// reuse a boundary sector.

// ---------------------------------------------------------------------
// TODO 1 (solved): how many distinct 32 B sectors does the warp handling
// row `row` touch, given a row pitch of `pitch` floats?
//
// The warp reads 128 contiguous bytes beginning at byte offset
// row*pitch*4 from a 256 B-aligned base. Contiguous 128 B spans 4 sectors
// if the start is 32 B aligned and 5 otherwise.
// ---------------------------------------------------------------------
static int sectors_for_row(uintptr_t base, int pitch, long long row)
{
    uintptr_t a0 = base + (uintptr_t)row * (uintptr_t)pitch * sizeof(float);
    uintptr_t a1 = a0 + (uintptr_t)COLS_USED * sizeof(float) - 1;
    return (int)(a1 / 32 - a0 / 32 + 1);
}

static double mean_sectors(uintptr_t base, int pitch)
{
    // The pattern of row-start offsets repeats with period at most 32,
    // so averaging over 64 rows is exact.
    double s = 0.0;
    for (long long r = 0; r < 64; ++r) s += sectors_for_row(base, pitch, r);
    return s / 64.0;
}

// ---------------------------------------------------------------------
// TODO 2 (solved): the three alignment thresholds.
//
// A pitch of P floats makes every row start P*4 bytes apart. For every
// row start to be A-byte aligned (given a base that already is), we need
// P*4 % A == 0, i.e. P % (A/4) == 0. And P must be >= COLS_TOTAL = 65.
//
//   A = 16 B  (what a float4 access requires)     -> P % 4 == 0 -> 68
//   A = 32 B  (one sector: what coalescing wants) -> P % 8 == 0 -> 72
//   A = 128 B (one full L1/L2 line)               -> P % 32 == 0 -> 96
//
// The trap is 68. It satisfies the rule people remember ("multiple of 4
// so float4 works") and still leaves half the rows straddling a sector
// boundary.
// ---------------------------------------------------------------------
static const int MIN_PITCH_16B  = 68;
static const int MIN_PITCH_32B  = 72;
static const int MIN_PITCH_128B = 96;

// TODO 4 (solved): the pitch to ship.
static const int PITCH_CHOSEN   = 96;

// ---------------------------------------------------------------------
// TODO 3 (solved): one warp per row; lane L handles column L.
// ---------------------------------------------------------------------
__global__ void row_window_scale(float* __restrict__ a, long long rows, int pitch)
{
    long long gid  = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long row  = gid >> 5;          // 32 threads per row
    int       lane = (int)(gid & 31);
    if (row >= rows) return;
    long long i = row * (long long)pitch + lane;
    a[i] = a[i] * 2.0f + 1.0f;
}

// Deterministic, index-derived initialisation. At this size a host-side
// init plus an H2D copy of 384 MB per trial would dominate the runtime, so
// we generate the same closed-form values on the device and mirror the
// formula exactly on the host for the reference.
__global__ void fill(float* a, long long rows, int pitch)
{
    long long gid  = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long row  = gid >> 5;
    int       lane = (int)(gid & 31);
    if (row >= rows) return;
    a[row * (long long)pitch + lane] = (float)(((row * 31 + lane * 7) % 97) - 48) * 0.015625f;
}

static float host_init(long long row, int lane)
{
    return (float)(((row * 31 + lane * 7) % 97) - 48) * 0.015625f;
}

// ---------------------------------------------------------------------
template <typename L>
static double time_ms(L launch)
{
    const float WARM_MS = 400.0f;
    cudaEvent_t b, e;
    CHECK(cudaEventCreate(&b)); CHECK(cudaEventCreate(&e));
    float warm = 0.0f;
    while (warm < WARM_MS) {
        CHECK(cudaEventRecord(b));
        for (int i = 0; i < 5; ++i) launch();
        CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
        CHECK(cudaGetLastError());
        float dt = 0; CHECK(cudaEventElapsedTime(&dt, b, e));
        warm += dt;
    }
    const int IT = 20;
    CHECK(cudaEventRecord(b));
    for (int i = 0; i < IT; ++i) launch();
    CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
    CHECK(cudaGetLastError());
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, b, e));
    CHECK(cudaEventDestroy(b)); CHECK(cudaEventDestroy(e));
    return ms / IT;
}

int main(void)
{
    CHECK(cudaSetDevice(0));

    const size_t cap    = (size_t)ROWS * PITCH_MAX * sizeof(float);
    const long long useful = ROWS * (long long)COLS_USED * 4LL * 2LL;  // read+write

    printf("ROWS = %lld, record = %d floats, kernel window = %d floats\n",
           ROWS, COLS_TOTAL, COLS_USED);
    printf("allocation = %.0f MB; touched footprint >= %.0f MB (L2 = 48 MB)\n",
           cap / 1048576.0, ROWS * 128.0 / 1048576.0);
    printf("useful traffic / launch = %.0f MB\n\n", useful / 1048576.0);

    float* d = nullptr;
    CHECK(cudaMalloc(&d, cap));
    const uintptr_t base = (uintptr_t)d;

    if (MIN_PITCH_16B <= 0 || MIN_PITCH_32B <= 0 ||
        MIN_PITCH_128B <= 0 || PITCH_CHOSEN <= 0) {
        printf("Set TODO 2 and TODO 4 to continue.\n");
        CHECK(cudaFree(d)); CHECK(cudaDeviceReset()); return 0;
    }

    // ---- sanity: the thresholds must actually be legal pitches ----
    const int cands[3] = { MIN_PITCH_16B, MIN_PITCH_32B, MIN_PITCH_128B };
    const int wants[3] = { 16, 32, 128 };
    int ok = 1;
    for (int i = 0; i < 3; ++i) {
        if (cands[i] < COLS_TOTAL) {
            printf("  [FAIL] pitch %d is narrower than a record (%d)\n", cands[i], COLS_TOTAL);
            ok = 0;
        } else if ((cands[i] * 4) % wants[i] != 0) {
            printf("  [FAIL] pitch %d does not give %d B-aligned row starts\n",
                   cands[i], wants[i]);
            ok = 0;
        }
    }
    if (!ok) { CHECK(cudaFree(d)); CHECK(cudaDeviceReset()); return 1; }
    printf("  [OK] TODO 2 thresholds are legal and correctly aligned\n\n");

    // ---- Part A: the model ----
    int pitches[5] = { COLS_TOTAL, MIN_PITCH_16B, MIN_PITCH_32B, MIN_PITCH_128B, PITCH_CHOSEN };
    const char* labels[5] = { "65 (as given)", "MIN_PITCH_16B", "MIN_PITCH_32B",
                              "MIN_PITCH_128B", "PITCH_CHOSEN" };

    printf("=== Part A: predicted, from your sector count ===\n");
    printf("  %-16s %7s %14s %14s %12s\n",
           "pitch", "floats", "rowstride B", "mean sectors", "efficiency");
    double modelEff[5];
    for (int i = 0; i < 5; ++i) {
        double ms_ = mean_sectors(base, pitches[i]);
        modelEff[i] = 128.0 / (32.0 * ms_);
        printf("  %-16s %7d %14d %14.3f %11.1f%%\n",
               labels[i], pitches[i], pitches[i]*4, ms_, 100.0*modelEff[i]);
    }

    // ---- Part B: measure ----
    const int TPB  = 256;
    const long long threads = ROWS * 32;
    const int GRID = (int)((threads + TPB - 1) / TPB);

    printf("\n=== Part B: measured (3 passes, last reported) ===\n");
    printf("  %-16s %7s %9s %10s %9s %11s   %s\n",
           "pitch", "floats", "ms", "GB/s", "%ofpeak", "model", "validation");

    double meas[5];
    for (int pass = 0; pass < 3; ++pass) {
        for (int i = 0; i < 5; ++i) {
            const int P = pitches[i];
            fill<<<GRID, TPB>>>(d, ROWS, P);
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

            double ms = time_ms([&]{ row_window_scale<<<GRID, TPB>>>(d, ROWS, P); });
            meas[i] = useful / (ms * 1e-3) / 1e9;

            if (pass < 2) continue;

            // one clean pass for validation
            fill<<<GRID, TPB>>>(d, ROWS, P);
            row_window_scale<<<GRID, TPB>>>(d, ROWS, P);
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

            const long long SAMPLES = 100000;   // sample rows; 1e6*32 on host is wasteful
            float* h = (float*)malloc(SAMPLES * COLS_USED * sizeof(float));
            long long bad = 0;
            for (long long blk = 0; blk < SAMPLES; blk += 1000) {
                long long nrow = 1000;
                CHECK(cudaMemcpy2D(h, COLS_USED*sizeof(float),
                                   d + blk*(long long)P, (size_t)P*sizeof(float),
                                   COLS_USED*sizeof(float), (size_t)nrow,
                                   cudaMemcpyDeviceToHost));
                for (long long r = 0; r < nrow; ++r)
                    for (int c = 0; c < COLS_USED; ++c) {
                        float ref = host_init(blk + r, c) * 2.0f + 1.0f;
                        float got = h[r*COLS_USED + c];
                        if (fabs(got - ref) > 1e-5f * fmaxf(1.0f, fabsf(ref))) ++bad;
                    }
            }
            free(h);

            printf("  %-16s %7d %9.3f %10.1f %8.1f%% %10.1f%%   %s (%lld)\n",
                   labels[i], P, ms, meas[i], 100.0*meas[i]/PEAK_GBS,
                   100.0*modelEff[i], bad ? "FAIL" : "PASS", bad);
        }
    }

    printf("\n=== Predicted vs measured, normalised to MIN_PITCH_128B ===\n");
    printf("  %-16s %14s %14s\n", "pitch", "model ratio", "measured ratio");
    for (int i = 0; i < 5; ++i)
        printf("  %-16s %14.3f %14.3f\n", labels[i],
               modelEff[i]/modelEff[3], meas[i]/meas[3]);

    CHECK(cudaFree(d));
    CHECK(cudaDeviceReset());
    return 0;
}
