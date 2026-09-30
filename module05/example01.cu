// =====================================================================
// Module 5 / Example 1 : "What a warp actually asks the memory system for"
//
// Two halves, deliberately side by side:
//
//   PART A (host, no GPU work): the sector arithmetic. For a given access
//     pattern we enumerate the 32 addresses warp 0 touches, map each to
//     its 32-byte sector id (addr >> 5), count the DISTINCT sectors, and
//     report efficiency = bytes_requested / (32 * distinct_sectors).
//     This is exactly the pencil-and-paper procedure from the lesson,
//     executed on the *real* device pointer, so alignment is real.
//
//   PART B (device): the same patterns run as kernels over a 256 MB
//     buffer (>> the 48 MB L2, so we measure DRAM, not cache), timed with
//     CUDA events. We print effective GB/s (useful bytes / time) and the
//     DRAM GB/s that the Part A model implies. The two columns together
//     are the whole lesson.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN:    .\example01.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
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

static const double PEAK_GBS = 432.0;   // RTX 3500 Ada, 192-bit @ 9.001 GHz DDR
static const int    SECTOR   = 32;      // bytes -- ARCHITECTURE-SPECIFIC (sm_89)

// ---------------------------------------------------------------------
// PART A -- the sector model, on the host.
//
// We describe a pattern by the element index lane L of warp 0 touches.
// Everything else (address, sector, count) follows mechanically.
// ---------------------------------------------------------------------
typedef long long (*IndexFn)(int lane, long long param);

static long long ix_contig   (int lane, long long p) { (void)p; return lane; }
static long long ix_offset   (int lane, long long p) { return lane + p; }        // p = element offset
static long long ix_stride   (int lane, long long p) { return (long long)lane * p; }
static long long ix_broadcast(int lane, long long p) { (void)lane; return p; }
static long long ix_reversed (int lane, long long p) { (void)p; return 31 - lane; }

// Count distinct 32 B sectors covered by warp 0's 32 addresses.
// elemBytes is how many bytes each lane actually loads (4 for float,
// 16 for float4). A lane that loads 16 B can straddle two sectors.
static int sector_count(uintptr_t base, IndexFn f, long long param,
                        int elemBytes, uintptr_t* outSectors, int maxOut)
{
    uintptr_t seen[256];
    int nSeen = 0;
    for (int lane = 0; lane < 32; ++lane) {
        uintptr_t a0 = base + (uintptr_t)(f(lane, param) * elemBytes);
        for (uintptr_t a = a0; a < a0 + (uintptr_t)elemBytes; a += SECTOR) {
            uintptr_t s = a / SECTOR;
            int dup = 0;
            for (int k = 0; k < nSeen; ++k) if (seen[k] == s) { dup = 1; break; }
            if (!dup && nSeen < 256) seen[nSeen++] = s;
        }
        // also the sector containing the LAST byte, in case elemBytes
        // is not a multiple of 32 and the loop above missed it
        uintptr_t sLast = (a0 + elemBytes - 1) / SECTOR;
        int dup = 0;
        for (int k = 0; k < nSeen; ++k) if (seen[k] == sLast) { dup = 1; break; }
        if (!dup && nSeen < 256) seen[nSeen++] = sLast;
    }
    for (int k = 0; k < nSeen && k < maxOut; ++k) outSectors[k] = seen[k];
    return nSeen;
}

static void report_pattern(const char* name, uintptr_t base, IndexFn f,
                           long long param, int elemBytes, int printAddrs)
{
    uintptr_t sec[256];
    int n = sector_count(base, f, param, elemBytes, sec, 256);

    // Bytes the warp actually asked for. Distinct bytes, so a broadcast
    // asks for 4, not 128 -- that is the honest accounting.
    // For these patterns lanes never partially overlap, so:
    long long requested = (f == ix_broadcast) ? elemBytes : 32LL * elemBytes;
    long long moved     = (long long)n * SECTOR;

    printf("  %-28s : %2d sectors, %4lld B requested / %4lld B moved = %6.1f%%\n",
           name, n, requested, moved, 100.0 * (double)requested / (double)moved);

    if (printAddrs) {
        printf("      lane : byte offset from base : sector id (relative to base's sector)");
        for (int lane = 0; lane < 32; ++lane) {
            if (lane % 8 == 0) printf("\n      ");
            uintptr_t a = base + (uintptr_t)(f(lane, param) * elemBytes);
            printf("L%-2d:+%-4lld s%-4lld  ", lane,
                   (long long)(a - base),
                   (long long)(a / SECTOR) - (long long)(base / SECTOR));
        }
        printf("\n");
    }
}

// ---------------------------------------------------------------------
// PART B -- the same patterns as kernels.
//
// Every kernel does one read-modify-write of one float per thread, so
// "useful bytes" is always 8 * nThreads. Only the *address pattern*
// differs. That isolates the variable we care about.
// ---------------------------------------------------------------------
__global__ void k_contig(float* a, long long nThreads)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < nThreads) a[t] = a[t] * 2.0f + 1.0f;
}

__global__ void k_offset(float* a, long long nThreads, int off)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < nThreads) { long long i = t + off; a[i] = a[i] * 2.0f + 1.0f; }
}

__global__ void k_stride(float* a, long long nThreads, int stride)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < nThreads) { long long i = t * stride; a[i] = a[i] * 2.0f + 1.0f; }
}

// Reversed *within the warp*: lane L of each warp touches the element
// that lane 31-L would have touched. Same address SET, different order.
__global__ void k_reversed(float* a, long long nThreads)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < nThreads) {
        long long warpBase = t & ~31LL;
        long long i = warpBase + (31 - (t & 31LL));
        a[i] = a[i] * 2.0f + 1.0f;
    }
}

// ---------------------------------------------------------------------
struct Result { double ms; double effGBs; };

template <typename Launch>
static Result time_kernel(Launch launch, long long usefulBytes)
{
    cudaEvent_t beg, end;
    CHECK(cudaEventCreate(&beg));
    CHECK(cudaEventCreate(&end));

    launch();                                  // warm-up
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    const int ITERS = 25;
    CHECK(cudaEventRecord(beg));
    for (int i = 0; i < ITERS; ++i) launch();
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaGetLastError());

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, beg, end));
    ms /= ITERS;

    CHECK(cudaEventDestroy(beg));
    CHECK(cudaEventDestroy(end));

    Result r;
    r.ms     = ms;
    r.effGBs = (double)usefulBytes / (ms * 1.0e-3) / 1.0e9;
    return r;
}

int main(void)
{
    CHECK(cudaSetDevice(0));

    // 256 MB: 5.3x the 48 MB L2. Any smaller and we would be timing the
    // cache. This is the single most common benchmarking mistake.
    const long long N = 64LL * 1024 * 1024;          // floats
    const long long bytes = N * sizeof(float);
    printf("Buffer: %lld floats = %.0f MB  (L2 is 48 MB)\n\n",
           N, bytes / (1024.0 * 1024.0));

    float* d = nullptr;
    CHECK(cudaMalloc(&d, bytes));
    CHECK(cudaMemset(d, 0, bytes));

    uintptr_t base = (uintptr_t)d;
    printf("cudaMalloc returned %p -> base %% 256 = %llu, base %% 128 = %llu\n",
           (void*)d, (unsigned long long)(base % 256), (unsigned long long)(base % 128));
    printf("(cudaMalloc always returns at least 256 B-aligned memory.)\n\n");

    // ---------------- PART A ----------------
    printf("=== PART A: warp 0's address set, by hand ===\n");
    report_pattern("contiguous, aligned",      base, ix_contig,    0,  4, 1);
    report_pattern("contiguous, +1 float (4B)",base, ix_offset,    1,  4, 0);
    report_pattern("contiguous, +8 floats(32B)",base, ix_offset,   8,  4, 0);
    report_pattern("stride 2 floats",          base, ix_stride,    2,  4, 0);
    report_pattern("stride 6 floats (AoS .x)", base, ix_stride,    6,  4, 0);
    report_pattern("stride 32 floats",         base, ix_stride,   32,  4, 0);
    report_pattern("broadcast (all lanes a[7])",base, ix_broadcast, 7, 4, 0);
    report_pattern("reversed within warp",     base, ix_reversed,   0, 4, 0);
    report_pattern("float4, aligned",          base, ix_contig,     0,16, 0);
    printf("\n  Note: 'reversed' has the SAME sector count as 'contiguous'.\n"
           "  The coalescer sees a SET of addresses, not a sequence.\n\n");

    // ---------------- PART B ----------------
    printf("=== PART B: measured, 256 MB buffer, 25 timed iters ===\n");
    printf("  %-26s %8s %10s %9s %9s %11s\n",
           "pattern", "ms", "effGB/s", "%ofpeak", "model", "impliedDRAM");

    const int TPB = 256;

    // Clock ramp-up. A laptop GPU boots at a low SM/memory clock; the first
    // kernel in a benchmark otherwise reads 10-15%% slow and you will
    // mis-attribute the difference to your access pattern. Burn ~0.3 s first.
    for (int i = 0; i < 500; ++i)
        k_contig<<<(int)((N + TPB - 1) / TPB), TPB>>>(d, N);
    CHECK(cudaDeviceSynchronize());

    // -- contiguous
    {
        long long nt = N;
        Result r = time_kernel([&]{
            k_contig<<<(int)((nt + TPB - 1) / TPB), TPB>>>(d, nt);
        }, 8 * nt);
        double eff = 4.0 / 4.0;
        printf("  %-26s %8.3f %10.1f %9.1f %8.1f%% %11.1f\n", "contiguous aligned",
               r.ms, r.effGBs, 100.0 * r.effGBs / PEAK_GBS, 100.0 * eff, r.effGBs / eff);
    }
    // -- offset by 1 float (misaligned by 4 B)
    {
        long long nt = N - 32;
        Result r = time_kernel([&]{
            k_offset<<<(int)((nt + TPB - 1) / TPB), TPB>>>(d, nt, 1);
        }, 8 * nt);
        double eff = 4.0 / 5.0;
        printf("  %-26s %8.3f %10.1f %9.1f %8.1f%% %11.1f\n", "offset +1 float",
               r.ms, r.effGBs, 100.0 * r.effGBs / PEAK_GBS, 100.0 * eff, r.effGBs / eff);
    }
    // -- offset by 8 floats (32 B: sector-aligned again)
    {
        long long nt = N - 32;
        Result r = time_kernel([&]{
            k_offset<<<(int)((nt + TPB - 1) / TPB), TPB>>>(d, nt, 8);
        }, 8 * nt);
        double eff = 1.0;
        printf("  %-26s %8.3f %10.1f %9.1f %8.1f%% %11.1f\n", "offset +8 floats (32B)",
               r.ms, r.effGBs, 100.0 * r.effGBs / PEAK_GBS, 100.0 * eff, r.effGBs / eff);
    }
    // -- reversed within warp
    {
        long long nt = N;
        Result r = time_kernel([&]{
            k_reversed<<<(int)((nt + TPB - 1) / TPB), TPB>>>(d, nt);
        }, 8 * nt);
        double eff = 1.0;
        printf("  %-26s %8.3f %10.1f %9.1f %8.1f%% %11.1f\n", "reversed in warp",
               r.ms, r.effGBs, 100.0 * r.effGBs / PEAK_GBS, 100.0 * eff, r.effGBs / eff);
    }
    // -- strides
    {
        const int strides[] = {2, 4, 6, 8, 16, 32};
        for (int si = 0; si < 6; ++si) {
            int s = strides[si];
            long long nt = N / s;
            Result r = time_kernel([&]{
                k_stride<<<(int)((nt + TPB - 1) / TPB), TPB>>>(d, nt, s);
            }, 8 * nt);
            // model: 32 lanes * 4 B spread over stride*4 B -> sectors touched
            int sectors = (s * 4 * 32 + 31) / 32;   // bytes spanned / 32
            if (sectors > 32) sectors = 32;         // never more than 1 per lane
            double eff = 128.0 / (32.0 * sectors);
            char nm[64]; snprintf(nm, sizeof(nm), "stride %d floats", s);
            printf("  %-26s %8.3f %10.1f %9.1f %8.1f%% %11.1f\n", nm,
                   r.ms, r.effGBs, 100.0 * r.effGBs / PEAK_GBS, 100.0 * eff, r.effGBs / eff);
        }
    }

    printf("\n  'model' is the efficiency predicted by the Part A sector count.\n"
           "  'impliedDRAM' = effGB/s / model: the bytes DRAM really moved.\n"
           "  When impliedDRAM approaches 432, the bus is saturated and the\n"
           "  only way to go faster is to stop wasting sectors.\n");

    CHECK(cudaFree(d));
    CHECK(cudaDeviceReset());
    return 0;
}
