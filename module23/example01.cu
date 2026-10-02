// =============================================================================
// Module 23 / Example 1 — reconstructing Nsight Compute's
//                         "Memory Workload Analysis" section.
//
// `ncu` cannot run on this machine: every invocation returns
//   ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
//             NVIDIA GPU Performance Counters on the target device 0.
// (verified with Nsight Compute 2026.1.0.0, build 37166530, CUDA 13.2).
//
// So this program BUILDS the section instead of reading it. Every quantity
// printed below is a quantity `ncu` would report, computed here either by
// enumerating the addresses a warp presents (Module 5's and Module 7's
// procedures, run on the host) or by direct measurement. The metric name that
// carries each quantity is printed next to it, so that when you run `ncu` on a
// machine where it works you already know what you are reading.
//
// Covered:
//   A  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
//      l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
//      -> sectors per request, which IS coalescing efficiency       (Module 5)
//   B  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
//      l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum         (Module 7)
//   C  dram__bytes.sum.per_second
//      dram__throughput.avg.pct_of_peak_sustained_elapsed           (Module 12)
//   D  lts__t_sector_hit_rate / l1tex__t_sector_hit_rate, bounded   (Module 16)
//
// BUILD: nvcc -arch=sm_89 -O3 -lineinfo -o example01.exe example01.cu
// RUN  : example01.exe
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                       \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        printf("CUDA error %s (%s) at %s:%d\n", cudaGetErrorName(_e),          \
               cudaGetErrorString(_e), __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// ---- hardware facts (spec table) -------------------------------------------
#define SECTOR_BYTES        32
#define LINE_BYTES         128
#define SMEM_BANKS          32
#define DRAM_PIN_PEAK_GBS  432.0

// 256 MB of floats: 5.3x the 48 MB L2, so every sweep below is a DRAM sweep.
#define NFLOAT   ((size_t)64*1024*1024)
#define NBYTES   (NFLOAT*sizeof(float))

// =============================================================================
// Part A — sectors per request, by enumeration.
//
// `ncu` divides two counters:
//     l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   (32 B sectors moved)
//     l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum  (warp-level LDG count)
// The quotient is the number of 32 B sectors the L1 had to touch for one warp
// instruction. The ideal for a 4-byte load is 4 (32 lanes x 4 B = 128 B,
// exactly 4 sectors); the worst case is 32 (every lane in its own sector).
//
// That quotient is exactly Module 5's hand-computed sector count. Here is the
// counting procedure as code: enumerate the 32 byte addresses, shift right by
// 5, count distinct values.
// =============================================================================
typedef size_t (*AddrFn)(int lane);

static size_t a_contig   (int lane) { return (size_t)lane * 4; }        // in[i]
static size_t a_vec4     (int lane) { return (size_t)lane * 16; }       // in4[i]
static size_t a_stride2  (int lane) { return (size_t)lane * 8; }        // in[2*i]
static size_t a_stride8  (int lane) { return (size_t)lane * 32; }       // in[8*i]
static size_t a_broadcast(int lane) { (void)lane; return 0; }           // in[0]
static size_t a_misalign (int lane) { return 4 + (size_t)lane * 4; }    // in[i+1]
static size_t a_reverse  (int lane) { return (size_t)(31 - lane) * 4; } // in[31-i]

// Count distinct 32 B sectors, and distinct 128 B lines, over one warp.
static void warpFootprint(AddrFn f, int elemBytes, int *sectors, int *lines)
{
    size_t sec[32], lin[32]; int ns = 0, nl = 0;
    for (int lane = 0; lane < 32; ++lane) {
        size_t a0 = f(lane);
        // An access wider than a sector can straddle; walk it byte-block by
        // byte-block so float4 is counted honestly.
        for (int b = 0; b < elemBytes; b += SECTOR_BYTES) {
            size_t s = (a0 + (size_t)b) / SECTOR_BYTES;
            int seen = 0; for (int q = 0; q < ns; ++q) if (sec[q] == s) { seen = 1; break; }
            if (!seen) sec[ns++] = s;
            size_t L = (a0 + (size_t)b) / LINE_BYTES;
            seen = 0; for (int q = 0; q < nl; ++q) if (lin[q] == L) { seen = 1; break; }
            if (!seen) lin[nl++] = L;
        }
    }
    *sectors = ns; *lines = nl;
}

// =============================================================================
// Part B — shared-memory bank conflicts, by enumeration.
//
// `ncu` reports
//     l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
//     l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
// The second counts the wavefronts (replays) the access was split into; the
// first counts the EXTRA ones, i.e. wavefronts - requests. For a 4-byte load
// by a full warp, wavefronts = D, the conflict degree Module 7 defines as the
// maximum, over banks, of the number of DISTINCT WORDS that bank must supply.
// Counting lanes instead of distinct words turns a free broadcast into a
// phantom conflict -- that is Module 7 Exercise 1's trap, and it is the same
// mistake you can make reading the counter.
// =============================================================================
typedef int (*IdxFn)(int lane);   // returns a float index into the shared array

static int s_tid    (int lane) { return lane; }           // s[tid]
static int s_2tid   (int lane) { return 2 * lane; }       // s[2*tid]
static int s_3tid   (int lane) { return 3 * lane; }       // s[3*tid]
static int s_8tid   (int lane) { return 8 * lane; }       // s[8*tid]
static int s_32tid  (int lane) { return 32 * lane; }      // s[32*tid]
static int s_tidhalf(int lane) { return lane / 2; }       // s[tid/2]
static int s_zero   (int lane) { (void)lane; return 0; }  // s[0]

static int conflictDegree(IdxFn f)
{
    int words[SMEM_BANKS][32]; int nw[SMEM_BANKS];
    for (int b = 0; b < SMEM_BANKS; ++b) nw[b] = 0;
    for (int lane = 0; lane < 32; ++lane) {
        int w = f(lane);                 // word index
        int b = w % SMEM_BANKS;          // bank = (addr/4) % 32
        int seen = 0;
        for (int q = 0; q < nw[b]; ++q) if (words[b][q] == w) { seen = 1; break; }
        if (!seen) words[b][nw[b]++] = w;   // DISTINCT WORDS, not lanes
    }
    int d = 0; for (int b = 0; b < SMEM_BANKS; ++b) if (nw[b] > d) d = nw[b];
    return d;
}

// =============================================================================
// Kernels. Each one is the kernel whose addresses Part A enumerates.
// =============================================================================

// Reads every `stride`-th float of `in` over the whole 256 MB buffer. The
// number of USEFUL bytes falls with the stride; the number of sectors the
// memory system has to move does not (at stride 8 the sectors are adjacent and
// every one of them is touched).
template <int STRIDE>
__global__ void strideRead(const float * __restrict__ in, float *sink, size_t nflt)
{
    size_t n = nflt / STRIDE;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (; i < n; i += gridDim.x * (size_t)blockDim.x)
        acc += in[i * STRIDE];
    if (acc == 1e30f) sink[0] = acc;   // never true, never provable
}

// The pure streaming ceiling, for the dram__throughput denominator.
__global__ void streamRead(const float4 * __restrict__ in, float *sink, size_t n4)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    for (; i < n4; i += gridDim.x * (size_t)blockDim.x) {
        float4 v = in[i];
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    if (a.x == 1e30f) sink[0] = a.x + a.y + a.z + a.w;
}

// A correctness kernel so the file validates something real as well as timing.
__global__ void strideCopy(const float * __restrict__ in, float *out, size_t n, int stride)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i * (size_t)stride] * 2.0f + 1.0f;
}

// =============================================================================
// Timing helpers (spec SS12: duration-based warm-up, auto-scaled iteration
// counts, rotated back-to-back sweeps, min-of-N, validation in a second pass).
// =============================================================================
static float4 *gBig4; static float *gBig; static float *gSink;

static double timeLaunch(void (*f)(void), int iters)
{
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) f();
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms / iters;
}
static void lStream(void) { streamRead<<<640,256>>>(gBig4, gSink, NFLOAT/4); }
static void lS1(void)     { strideRead<1><<<640,256>>>(gBig, gSink, NFLOAT); }
static void lS2(void)     { strideRead<2><<<640,256>>>(gBig, gSink, NFLOAT); }
static void lS8(void)     { strideRead<8><<<640,256>>>(gBig, gSink, NFLOAT); }

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 23 / Example 1 - reconstructing Memory Workload Analysis ===\n");
    printf("ncu status on this machine: ERR_NVGPUCTRPERM (counters require\n"
           "elevation). Every number below is reconstructed, never read from a\n"
           "counter. The metric names are the real ones.\n\n");

    int ok = 1;

    // ---------------------------------------------------------------- Part A
    printf("-- A. sectors per request ------------------------------------------\n");
    printf("   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum\n");
    printf("   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum\n");
    printf("   (one fully active warp, one load instruction, base 256 B aligned)\n\n");
    printf("   %-22s %6s %6s %8s %8s %9s\n",
           "source expression", "elemB", "sect", "lines", "sect/req", "used/moved");

    struct { const char *name; AddrFn f; int eb; int expectSec; } pat[7] = {
        { "in[i]",            a_contig,    4,  4 },
        { "in4[i]  (float4)", a_vec4,     16, 16 },
        { "in[2*i]",          a_stride2,   4,  8 },
        { "in[8*i]",          a_stride8,   4, 32 },
        { "in[0]   broadcast", a_broadcast, 4,  1 },
        { "in[i+1] misaligned",a_misalign,  4,  5 },
        { "in[31-i] reversed", a_reverse,   4,  4 },
    };
    for (int p = 0; p < 7; ++p) {
        int s, L; warpFootprint(pat[p].f, pat[p].eb, &s, &L);
        double useful = 32.0 * pat[p].eb;
        double moved  = (double)s * SECTOR_BYTES;
        printf("   %-22s %6d %6d %8d %8.1f %8.1f%%%s\n",
               pat[p].name, pat[p].eb, s, L, (double)s, 100.0*useful/moved,
               s == pat[p].expectSec ? "" : "   <-- MODEL DISAGREES");
        if (s != pat[p].expectSec) ok = 0;
    }
    printf("\n   Reading this in a real report: `ncu` gives you the quotient, not\n"
           "   the table. 4.0 is perfect for a 4-byte load. 32.0 means every lane\n"
           "   is in its own sector and you are moving 8x the bytes you use. The\n"
           "   broadcast row is the one people misread: 1 sector, 12.5%% \"memory\n"
           "   efficiency\", and nothing wrong with it at all.\n\n");

    // ---------------------------------------------------------------- Part B
    printf("-- B. shared-memory bank conflicts ---------------------------------\n");
    printf("   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum\n");
    printf("   l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum\n\n");
    printf("   %-14s %8s %12s %12s %14s\n",
           "expression", "degree", "wavefronts", "conflicts", "Ada cycles");

    struct { const char *name; IdxFn f; int expect; } bp[7] = {
        { "s[tid]",     s_tid,      1 },
        { "s[2*tid]",   s_2tid,     2 },
        { "s[3*tid]",   s_3tid,     1 },
        { "s[8*tid]",   s_8tid,     8 },
        { "s[32*tid]",  s_32tid,   32 },
        { "s[tid/2]",   s_tidhalf,  1 },
        { "s[0]",       s_zero,     1 },
    };
    for (int p = 0; p < 7; ++p) {
        int d = conflictDegree(bp[p].f);
        int cyc = d > 2 ? d : 2;                       // Module 7's max(2,D) law
        printf("   %-14s %8d %12d %12d %14d%s\n",
               bp[p].name, d, d, d - 1, cyc,
               d == bp[p].expect ? "" : "   <-- MODEL DISAGREES");
        if (d != bp[p].expect) ok = 0;
    }
    printf("\n   The trap this metric sets: `s[2*tid]` reports ONE conflict and\n"
           "   costs NOTHING. Module 7 measured the cost law on Ada as max(2,D)\n"
           "   cycles for a 4-byte access, so degree 2 is free. A report showing\n"
           "   a small non-zero conflict count on a 4 B access is not a finding.\n"
           "   (Module 18: on LDS.128 the floor of 2 disappears and cost ~ D.)\n"
           "   `s[tid/2]` is the other trap: bucket LANES per bank and you get\n"
           "   degree 2; bucket DISTINCT WORDS, which is what the hardware does,\n"
           "   and you get 1. The counter counts words.\n\n");

    // ---------------------------------------------------------------- Part C/D
    CHECK(cudaMalloc(&gBig, NBYTES));
    CHECK(cudaMalloc(&gSink, 16));
    gBig4 = (float4*)gBig;
    CHECK(cudaMemset(gBig, 1, NBYTES));

    printf("-- warming 1500 ms streaming (spec SS12 rule 4) ---------------------\n");
    { cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el = 0.f; CHECK(cudaEventRecord(w0));
      while (el < 1500.f) { lStream();
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el, w0, w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1)); }
    CHECK(cudaGetLastError());

    void (*cfg[4])(void) = { lStream, lS1, lS2, lS8 };
    const char *cname[4] = { "streamRead float4", "strideRead<1>",
                             "strideRead<2>",     "strideRead<8>" };
    const double sectPerReq[4] = { 16.0, 4.0, 8.0, 32.0 };
    // Useful bytes actually consumed by the arithmetic.
    const double usefulB[4] = { (double)NBYTES, (double)NBYTES,
                                (double)NBYTES/2.0, (double)NBYTES/8.0 };
    // Bytes the memory system must move: every sector in the 256 MB range is
    // touched by strides 1, 2 and 8 alike (stride 8 floats == one sector).
    const double movedB[4]  = { (double)NBYTES, (double)NBYTES,
                                (double)NBYTES, (double)NBYTES };

    int iters[4]; double best[4];
    for (int i = 0; i < 4; ++i) {
        double t = timeLaunch(cfg[i], 1);
        int n = (int)(10.0 / (t > 1e-3 ? t : 1e-3));
        if (n < 3) n = 3; if (n > 100) n = 100;
        iters[i] = n; best[i] = 1e30;
    }
    CHECK(cudaGetLastError());
    for (int s = 0; s < 4; ++s)
        for (int q = 0; q < 4; ++q) {
            int p = (q + s) % 4;
            double t = timeLaunch(cfg[p], iters[p]);
            if (t < best[p]) best[p] = t;
        }
    CHECK(cudaGetLastError());

    printf("\n-- C. DRAM throughput ----------------------------------------------\n");
    printf("   dram__bytes.sum.per_second\n");
    printf("   dram__throughput.avg.pct_of_peak_sustained_elapsed\n\n");
    printf("   %-20s %9s %10s %12s %10s %9s\n",
           "kernel", "ms", "sect/req", "effective", "impliedDRAM", "%of432");
    for (int i = 0; i < 4; ++i) {
        double eff = usefulB[i] / (best[i] * 1e-3) / 1e9;
        double dr  = movedB[i]  / (best[i] * 1e-3) / 1e9;
        printf("   %-20s %9.4f %10.1f %9.1f GB/s %7.1f GB/s %8.1f%%\n",
               cname[i], best[i], sectPerReq[i], eff, dr, 100.0*dr/DRAM_PIN_PEAK_GBS);
    }
    printf("\n   This is the single most useful pair of numbers in the tool, and\n"
           "   it is also where the misreading happens. `strideRead<8>` delivers\n"
           "   an eighth of the useful bandwidth of `strideRead<1>` while the\n"
           "   DRAM throughput counter sits at the SAME value. Nothing is wrong\n"
           "   with the memory system: 7 of every 8 bytes it moves are bytes you\n"
           "   asked for and never used. dram__throughput tells you the bus is\n"
           "   busy; sectors-per-request tells you whether it is busy on your\n"
           "   behalf. You need both.\n\n");

    double ceilGBs = movedB[0] / (best[0] * 1e-3) / 1e9;

    printf("-- D. cache hit rates ----------------------------------------------\n");
    printf("   l1tex__t_sector_hit_rate.pct , lts__t_sector_hit_rate.pct\n\n");
    printf("   These two have no honest reconstruction from a stopwatch: a hit\n"
           "   and a miss differ in latency, not in anything a kernel can count\n"
           "   for itself. What a stopwatch CAN give you is a BOUND, which is\n"
           "   Module 16's method:\n\n");
    printf("       on-chip service fraction  >=  1 - (pin peak x elapsed) / requested\n\n");
    {
        // strideRead<8> requests NBYTES/8 of useful data but each 4 B load pulls
        // a whole 32 B sector, so the L1 is asked for NBYTES/8 x 8 = NBYTES.
        double requested = (double)NBYTES;             // bytes the LSU asked L1 for
        double couldDram = DRAM_PIN_PEAK_GBS*1e9 * (best[3]*1e-3);
        double frac = 1.0 - couldDram/requested;
        if (frac < 0.0) frac = 0.0;
        printf("   strideRead<8>: requested %.1f MB in %.4f ms; DRAM could have\n"
               "   supplied at most %.1f MB in that time, so at least %.1f%% was\n"
               "   serviced on chip. (Here it is 0%%, and that is the correct\n"
               "   answer: the kernel really does stream the whole buffer.)\n",
               requested/1e6, best[3], couldDram/1e6, 100.0*frac);
    }
    printf("\n   The qualitative half you CAN see without counters: run the same\n"
           "   kernel on a working set that fits in the 48 MB L2 and the apparent\n"
           "   bandwidth exceeds the pin rate. Any figure above 432 GB/s is a\n"
           "   hit-rate measurement wearing a bandwidth's clothes.\n\n");

    // ------------------------------------------------- validation (second pass)
    printf("-- validation (separate pass, untimed) ------------------------------\n");
    {
        const size_t n = 1u << 20;
        float *hIn = (float*)malloc((size_t)8*n*sizeof(float));
        float *dOut; CHECK(cudaMalloc(&dOut, n*sizeof(float)));
        float *hOut = (float*)malloc(n*sizeof(float));
        for (size_t i = 0; i < 8*n; ++i) hIn[i] = (float)((i * 1103515245u + 12345u) & 1023u) * 0.001f;
        CHECK(cudaMemcpy(gBig, hIn, 8*n*sizeof(float), cudaMemcpyHostToDevice));
        strideCopy<<<(unsigned)((n+255)/256),256>>>(gBig, dOut, n, 8);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hOut, dOut, n*sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (size_t i = 0; i < n; ++i) {
            float ref = hIn[i*8]*2.0f + 1.0f;
            if (fabsf(hOut[i]-ref) > 1e-5f*fmaxf(1.0f, fabsf(ref))) { ++bad; }
        }
        printf("   strided copy, %zu elements: %d mismatches\n", n, bad);
        if (bad) ok = 0;
        free(hIn); free(hOut); CHECK(cudaFree(dOut));
    }
    int sane = (ceilGBs > 150.0 && ceilGBs <= DRAM_PIN_PEAK_GBS);
    printf("   streaming ceiling %.1f GB/s inside (150, %.1f] : %s\n",
           ceilGBs, DRAM_PIN_PEAK_GBS, sane ? "yes" : "NO - thermal state?");
    if (!sane) ok = 0;

    CHECK(cudaFree(gBig)); CHECK(cudaFree(gSink));
    printf("\nOVERALL: %s\n", ok ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return ok ? 0 : 1;
}
