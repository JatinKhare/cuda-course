// =====================================================================
// Module 11 / Exercise 1 -- SOLUTION
//   "Saturate the bus, with a budget"
//
// GOAL
//   `blend_v1` below is correct, coalesced, and a grid-stride loop. It is
//   also nowhere near this machine's memory bandwidth. Produce `blend_v2`,
//   which computes exactly the same thing and reaches at least 95% of the
//   streaming ceiling that this program measures in its own timing loop.
//
//   Before you write any code, you must commit to two numbers: how many
//   bytes this operation is OBLIGED to move, and how long that many bytes
//   take at 87% of the 432 GB/s peak (the figure Module 5 measured as this
//   machine's realistic streaming rate). The harness scores both.
//
//   THE BUDGET. `blend_v2` is one stage of a pipeline that must co-reside
//   with another kernel, so it may not have the whole machine:
//
//       gridDim.x * blockDim.x  <=  8192
//       blockDim.x % 32 == 0,  blockDim.x <= 1024,  gridDim.x >= 1
//
//   That constraint is the point of the exercise. Spending the whole budget
//   on threads is necessary and it is not sufficient.
//
// THE OPERATION
//       y[i] = clamp(A*x[i] + B*y[i] + C*x[i]*x[i],  LO, HI)     in place
//   N = 40,000,001, which is not a multiple of 2 or 4.
//   The update is NOT idempotent: an element processed twice is wrong, and
//   the validator will say so.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
// RUN:   .\exercise01_solution.exe
//
// Useful while you work:
//   nvcc -arch=sm_89 -O3 -Xptxas -v -cubin -o exercise01.cubin exercise01.cu
//   cuobjdump -sass exercise01.cubin > sass.txt
//   findstr /C:"Function :" /C:"LDG" /C:"STG" sass.txt
//
// WHAT IS CHECKED (8 points; all 8 required for OVERALL: PASS)
//   - v1 numerics, v2 numerics on its main range, v2 numerics on its tail
//   - v2 correctness at the degenerate launch <<<1,32>>>  (a grid-stride
//     kernel must not depend on the launch shape for correctness)
//   - the thread budget
//   - TODO 1 exactly; TODO 2 to within +-15%
//   - v2 >= 95% of the measured streaming ceiling
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

static const double PEAK_GBS      = 432.0;
static const double REFERENCE_GBS = 0.87 * PEAK_GBS;   // 375.84 GB/s
static const long long N          = 40000001LL;        // odd; N % 4 == 1
static const int    THREAD_BUDGET = 8192;
static const int    SWEEPS        = 4;

#define OP_A  1.25f
#define OP_B  0.75f
#define OP_C  0.5f
#define OP_LO (-2.0f)
#define OP_HI  2.0f

// ---------------------------------------------------------------------
// TODO 1 (solved)
// SOLVED: y is READ and WRITTEN (4 + 4). x is read once no matter how many
// times it is named in the source (4). 4 + 4 + 4 = 12.
static const int COMPULSORY_BYTES_PER_ELEMENT = 12;

// TODO 2 (solved)
// SOLVED: 12 B * 40,000,001 = 480,000,012 B / 375.84e9 B/s = 1.2771e-3 s.
static const double PREDICTED_MS = 1.277;

// ---------------------------------------------------------------------
__host__ __device__ inline float op(float xi, float yi)
{
    float v = OP_A*xi + OP_B*yi + OP_C*xi*xi;
    return fminf(OP_HI, fmaxf(OP_LO, v));
}

// The kernel as shipped. Do not modify it; it is the baseline.
__global__ void blend_v1(const float* __restrict__ x, float* __restrict__ y, long long n)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] = op(x[i], y[i]);
}

// ---------------------------------------------------------------------
// TODO 4 (solved)
// SOLVED: 128-bit access. Each thread handles four consecutive elements
// with one LDG.E.128 per input stream and one STG.E.128 out, so the memory
// instructions per element drop 4x and each outstanding-request slot
// carries 4x the payload. Under an 8,192-thread budget the machine is short
// of requests in flight -- not of bandwidth -- and that is the resource this
// buys. The tail is handled at kernel scope with its own range test.
__global__ void blend_v2(const float* __restrict__ x, float* __restrict__ y, long long n)
{
    const long long n4     = n / 4;                 // whole float4 groups
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long t      = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4*       y4 = reinterpret_cast<float4*>(y);

    for (long long i = t; i < n4; i += stride) {
        float4 X = x4[i], Y = y4[i];
        Y.x = op(X.x, Y.x);
        Y.y = op(X.y, Y.y);
        Y.z = op(X.z, Y.z);
        Y.w = op(X.w, Y.w);
        y4[i] = Y;
    }

    // Tail: elements 4*n4 .. n-1 (exactly one here). NOT nested inside the
    // loop above, and with its own range test, so it is processed exactly
    // once for any grid shape including <<<1,32>>>.
    const long long tailBegin = 4 * n4;
    for (long long j = tailBegin + t; j < n; j += stride)
        y[j] = op(x[j], y[j]);
}

// The streaming reference: one read, one write, nothing else. Timed in the
// same loop as the kernels under test, at the same thermal state, so that
// "% of ceiling" survives the clock drift that "% of 432" does not. The
// budget does not apply to this yardstick.
__global__ void stream_ref(const float* __restrict__ a, float* __restrict__ o, long long n)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x)
        o[i] = a[i];
}

// ---------------------------------------------------------------------
// TODO 3 (solved)
// SOLVED: spend the whole budget, and keep gridDim.x >= nSM so that no SM
// sits idle. 64 blocks x 128 threads = 8,192 = 1.6 blocks per SM. Several
// other pairs measure within noise of this one (see the solution notes);
// what fails is gridDim.x < nSM, which strands SMs, and leaving the budget
// unspent.
static void chooseLaunch(int nSM, int* grid, int* block)
{
    *block = 128;
    *grid  = THREAD_BUDGET / *block;      // 64, and 64 >= nSM
    if (*grid < nSM) *grid = nSM;         // never leave an SM idle
    (void)nSM;
}

// TODO 5 (solved)
// SOLVED: the float4 path covers 4 * (N/4) = 40,000,000 elements; element
// 40,000,000 is the tail.
static const long long V2_MAIN_ELEMENTS = 4 * (N / 4);

// ------------------------------------------------------------------ timing
// Spec 12: one probe launch sizes the iteration count so each timed segment
// is ~10 ms, then the segment is timed with cudaEvent_t.
template <typename L>
static double timeCfg(L launch, cudaEvent_t a, cudaEvent_t b)
{
    CHECK(cudaEventRecord(a)); launch(); CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float p = 0.f; CHECK(cudaEventElapsedTime(&p, a, b));
    if (p < 0.0005f) p = 0.0005f;
    int it = (int)(10.0 / p);
    if (it < 20)   it = 20;
    if (it > 5000) it = 5000;
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < it; ++i) launch();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, a, b));
    return (double)ms / it;
}

// FNV-1a over the 4 bytes of a 32-bit value. Used only to check TODO 1
// without printing the answer in the source.
static unsigned fnv1a32(unsigned v)
{
    unsigned h = 2166136261u;
    for (int i = 0; i < 4; ++i) { h ^= (v >> (8*i)) & 0xffu; h *= 16777619u; }
    return h;
}
static const unsigned BPE_HASH = 0x8c46f159u;

static long long mismatches(const float* got, const float* ref, long long lo, long long hi)
{
    long long bad = 0;
    for (long long i = lo; i < hi; ++i)
        if (fabsf(got[i]-ref[i]) > 1e-5f*fmaxf(1.0f, fabsf(ref[i]))) ++bad;
    return bad;
}

int main(void)
{
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    const int nSM = prop.multiProcessorCount;

    printf("Module 11 / Exercise 1 -- saturate the bus, with a budget\n");
    printf("Device: %s, %d SMs\n", prop.name, nSM);
    printf("N = %lld  (N %% 4 = %lld), one array = %.0f MB, L2 = %.0f MB\n",
           N, N % 4, N*4.0/1048576.0, prop.l2CacheSize/1048576.0);
    printf("Thread budget for the kernel under test: %d\n\n", THREAD_BUDGET);

    if (COMPULSORY_BYTES_PER_ELEMENT <= 0 || PREDICTED_MS <= 0.0) {
        printf("Set TODO 1 and TODO 2 first.\n"); return 0;
    }
    int grid = 0, block = 0;
    chooseLaunch(nSM, &grid, &block);
    if (grid <= 0 || block <= 0) { printf("Set TODO 3 first.\n"); return 0; }
    if (V2_MAIN_ELEMENTS < 0)    { printf("Set TODO 5 first.\n"); return 0; }

    const double bytes = (double)COMPULSORY_BYTES_PER_ELEMENT * (double)N;

    const size_t bs = (size_t)N * sizeof(float);
    float* h_x   = (float*)malloc(bs);
    float* h_y0  = (float*)malloc(bs);
    float* h_ref = (float*)malloc(bs);
    float* h_got = (float*)malloc(bs);
    if (!h_x || !h_y0 || !h_ref || !h_got) { printf("host allocation failed\n"); return 1; }
    for (long long i = 0; i < N; ++i) {
        h_x [i] = (float)(((i * 1103515245LL + 12345LL) % 2003) - 1001) * 0.001f;
        h_y0[i] = (float)(((i * 22695477LL   + 1LL)     % 1999) -  999) * 0.001f;
        h_ref[i] = op(h_x[i], h_y0[i]);
    }

    float *d_x, *d_y, *d_y0, *d_o;
    CHECK(cudaMalloc(&d_x,  bs)); CHECK(cudaMalloc(&d_y,  bs));
    CHECK(cudaMalloc(&d_y0, bs)); CHECK(cudaMalloc(&d_o,  bs));
    CHECK(cudaMemcpy(d_x,  h_x,  bs, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y0, h_y0, bs, cudaMemcpyHostToDevice));
    #define RESET() CHECK(cudaMemcpy(d_y, d_y0, bs, cudaMemcpyDeviceToDevice))
    RESET();

    const int V1_GRID = nSM, V1_BLOCK = 64;        // the shipped configuration

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    { float acc = 0.f;
      while (acc < 400.f) {
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < 20; ++i) stream_ref<<<nSM*8,256>>>(d_x, d_o, N);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms=0.f; CHECK(cudaEventElapsedTime(&ms,e0,e1)); acc += ms; } }
    CHECK(cudaGetLastError());

    // ---- timing pass: all three back to back, rotated, min of SWEEPS ----
    double tRef = 1e30, tV1 = 1e30, tV2 = 1e30;
    for (int s = 0; s < SWEEPS; ++s) {
        for (int q = 0; q < 3; ++q) {
            switch ((q + s) % 3) {
            case 0: { double m = timeCfg([&]{ stream_ref<<<nSM*8,256>>>(d_x, d_o, N); }, e0,e1);
                      if (m < tRef) tRef = m; } break;
            case 1: { double m = timeCfg([&]{ blend_v1<<<V1_GRID,V1_BLOCK>>>(d_x, d_y, N); }, e0,e1);
                      if (m < tV1) tV1 = m; } break;
            default:{ double m = timeCfg([&]{ blend_v2<<<grid,block>>>(d_x, d_y, N); }, e0,e1);
                      if (m < tV2) tV2 = m; } break;
            }
        }
    }
    CHECK(cudaGetLastError());
    const double ceilGBs = (8.0*(double)N)/(tRef*1e-3)/1e9;

    // ---- validation pass, untimed ----
    printf("=== validation (untimed second pass) ===\n");
    int pts = 0; const int maxpts = 8;

    RESET();
    blend_v1<<<V1_GRID,V1_BLOCK>>>(d_x, d_y, N);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_got, d_y, bs, cudaMemcpyDeviceToHost));
    long long badV1 = mismatches(h_got, h_ref, 0, N);
    printf("  [%s] v1 numerics                              %lld mismatch(es)\n", badV1?"  ":"ok", badV1);
    if (!badV1) ++pts;

    RESET();
    blend_v2<<<grid,block>>>(d_x, d_y, N);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_got, d_y, bs, cudaMemcpyDeviceToHost));
    long long badMain = mismatches(h_got, h_ref, 0, V2_MAIN_ELEMENTS);
    long long badTail = mismatches(h_got, h_ref, V2_MAIN_ELEMENTS, N);
    printf("  [%s] v2 numerics, main range [0, %lld)   %lld mismatch(es)\n",
           badMain?"  ":"ok", V2_MAIN_ELEMENTS, badMain);
    printf("  [%s] v2 numerics, tail  [%lld, %lld)   %lld mismatch(es)\n",
           badTail?"  ":"ok", V2_MAIN_ELEMENTS, N, badTail);
    if (!badMain) ++pts;
    if (!badTail) ++pts;

    RESET();
    blend_v2<<<1,32>>>(d_x, d_y, N);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_got, d_y, bs, cudaMemcpyDeviceToHost));
    long long badDeg = mismatches(h_got, h_ref, 0, N);
    printf("  [%s] v2 at the degenerate config <<<1,32>>>   %lld mismatch(es)\n",
           badDeg?"  ":"ok", badDeg);
    if (!badDeg) ++pts;

    const bool budgetOk = ((long long)grid * block <= THREAD_BUDGET) &&
                          (block % 32 == 0) && (block <= 1024) && (grid >= 1);
    printf("  [%s] launch %d x %d = %lld threads, budget %d\n",
           budgetOk?"ok":"  ", grid, block, (long long)grid*block, THREAD_BUDGET);
    if (budgetOk) ++pts;

    // The reference answer is not written out here; it is recovered from a
    // hash so that reading this file does not give TODO 1 away.
    int trueBytes = 0;
    for (int b = 1; b <= 256; ++b) if (fnv1a32((unsigned)b) == BPE_HASH) { trueBytes = b; break; }
    printf("  [%s] TODO 1 compulsory bytes/element = %d\n",
           (COMPULSORY_BYTES_PER_ELEMENT == trueBytes)?"ok":"  ", COMPULSORY_BYTES_PER_ELEMENT);
    if (COMPULSORY_BYTES_PER_ELEMENT == trueBytes) ++pts;

    const double trueFloor = ((double)trueBytes*(double)N)/(REFERENCE_GBS*1e9)*1e3;
    const bool predOk = fabs(PREDICTED_MS - trueFloor) <= 0.15*trueFloor;
    printf("  [%s] TODO 2 predicted floor %.3f ms vs %.3f ms at %.1f GB/s (+-15%%)\n",
           predOk?"ok":"  ", PREDICTED_MS, trueFloor, REFERENCE_GBS);
    if (predOk) ++pts;

    // ---- the performance table ----
    printf("\n=== performance ===\n");
    printf("  your compulsory traffic  = %d B/elem x %lld = %.0f MB\n",
           COMPULSORY_BYTES_PER_ELEMENT, N, bytes/1048576.0);
    printf("  floor at 87%% of peak (%.1f GB/s) = %.3f ms\n", REFERENCE_GBS, trueFloor);
    printf("  measured streaming ceiling      = %.1f GB/s (%.0f%% of %.0f)\n\n",
           ceilGBs, 100.0*ceilGBs/PEAK_GBS, PEAK_GBS);
    printf("  %-28s %9s %9s %9s %10s\n", "version", "ms", "GB/s", "%ofpeak", "%ofceil");
    printf("  %-28s %9.4f %9.1f %8.1f%% %9.1f%%\n", "v1 (as shipped, 40x64)",
           tV1, bytes/(tV1*1e-3)/1e9, 100.0*bytes/(tV1*1e-3)/1e9/PEAK_GBS,
           100.0*bytes/(tV1*1e-3)/1e9/ceilGBs);
    printf("  %-28s %9.4f %9.1f %8.1f%% %9.1f%%\n", "v2 (yours)",
           tV2, bytes/(tV2*1e-3)/1e9, 100.0*bytes/(tV2*1e-3)/1e9/PEAK_GBS,
           100.0*bytes/(tV2*1e-3)/1e9/ceilGBs);
    printf("  speedup v1 -> v2 : %.2fx\n", tV1/tV2);

    const double pctCeil = 100.0*bytes/(tV2*1e-3)/1e9/ceilGBs;
    const bool perfOk = pctCeil >= 95.0;
    printf("\n  [%s] v2 reaches %.1f%% of the measured ceiling (gate: 95.0%%)\n",
           perfOk?"ok":"  ", pctCeil);
    if (perfOk) ++pts;

    printf("\nScore: %d/%d\n", pts, maxpts);
    printf("OVERALL: %s\n", (pts == maxpts) ? "PASS" : "FAIL");

    free(h_x); free(h_y0); free(h_ref); free(h_got);
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d_x)); CHECK(cudaFree(d_y)); CHECK(cudaFree(d_y0)); CHECK(cudaFree(d_o));
    CHECK(cudaDeviceReset());
    return (pts == maxpts) ? 0 : 1;
}
