// =====================================================================
// Module 8 / Example 1 : warp formation, partial warps, and the active
//                        mask as an observable quantity.
//
// GOAL
//   Make the warp visible. This program does four things:
//     1. Launches a 100-thread block and shows that it becomes FOUR
//        warps, the last of which has 28 lanes that are inactive for
//        the whole lifetime of the block and still occupy thread slots.
//     2. Shows that warp membership follows the Module 3 linearization
//        rule, including for a 2-D block.
//     3. Reads the active mask with __activemask() at six labelled
//        points inside nested divergent control flow.
//     4. Timestamps both arms of a divergent if/else with clock64() to
//        show that the hardware runs them SEQUENTIALLY, not in parallel.
//
//   __activemask(), __ballot_sync() and __popc() are used here purely as
//   INSTRUMENTS for observing execution. Module 30 covers the warp
//   intrinsics as tools for building algorithms; this module only uses
//   them as a microscope.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
// RUN:    .\example01.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
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

static int popc32(unsigned v)
{
    int c = 0;
    while (v) { c += (int)(v & 1u); v >>= 1; }
    return c;
}

// ---------------------------------------------------------------------
// 1 + 2. Warp roster. Each thread reports its linearized index within the
//        block, the warp it landed in, its lane, and the active mask its
//        warp presents at this instruction.
// ---------------------------------------------------------------------
struct Report { int tid, warp, lane; unsigned mask; };

__global__ void roster(Report* rep)
{
    // The Module 3 linearization rule: x fastest, then y, then z.
    int tid = threadIdx.x
            + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    Report r;
    r.tid  = tid;
    r.warp = tid >> 5;          // tid / 32
    r.lane = tid & 31;          // tid % 32
    r.mask = __activemask();    // who else is at this instruction?
    rep[tid] = r;
}

// ---------------------------------------------------------------------
// 3. Active masks inside nested divergent control flow, plus a ballot.
//    NPT labelled points; every lane records the mask it saw at each
//    point it reached, and a flag saying whether it reached it at all.
// ---------------------------------------------------------------------
#define NPT 6
__device__ float g_sink;

__global__ void masks(const float* in, unsigned* mask, int* reached,
                      unsigned* ballot)
{
    int lane = threadIdx.x & 31;
    float a  = in[lane];

#define REC(p) do { mask[lane * NPT + (p)] = __activemask();            \
                    reached[lane * NPT + (p)] = 1; } while (0)

    REC(0);                                   // P0: kernel entry

    // A ballot: every lane contributes one bit saying whether it will
    // take the 'then' arm below. __popc counts the set bits. This is an
    // instrument -- it tells us the size of the divergence in advance.
    unsigned b = __ballot_sync(0xffffffffu, (lane & 1) != 0);
    if (lane == 0) *ballot = b;

    if (lane < 24) {
        REC(1);                               // P1: lanes 0..23
        // Long enough that the compiler cannot predicate it away.
        for (int j = 0; j < 40; ++j) a = fmaf(a, 1.0001f, 1e-6f);

        if ((lane & 7) == 0) {
            REC(2);                           // P2: lanes 0, 8, 16
            for (int j = 0; j < 40; ++j) a = fmaf(a, 1.0002f, 2e-6f);
        } else {
            REC(3);                           // P3: the other 21 lanes
            for (int j = 0; j < 40; ++j) a = fmaf(a, 1.0003f, 3e-6f);
        }
        REC(4);                               // P4: inner post-dominator
    }
    REC(5);                                   // P5: outer post-dominator

    g_sink = a;
#undef REC
}

// ---------------------------------------------------------------------
// 4. Sequential execution of divergent paths, timestamped.
//    Each lane records the SM clock at the start of its own arm and at
//    the end. If the two arms ran concurrently the intervals would
//    overlap. They do not.
// ---------------------------------------------------------------------
__global__ void sequencing(const float* in, long long* t0, long long* t1)
{
    int lane = threadIdx.x & 31;
    float a = in[lane];
    long long base = clock64();

    if (lane < 16) {
        t0[lane] = clock64() - base;
        for (int j = 0; j < 512; ++j) a = fmaf(a, 1.00001f, 1e-7f);
        t1[lane] = clock64() - base;
    } else {
        t0[lane] = clock64() - base;
        for (int j = 0; j < 512; ++j) a = fmaf(a, 1.00002f, 2e-7f);
        t1[lane] = clock64() - base;
    }
    g_sink = a;
}

// ---------------------------------------------------------------------
// 4b. Control: the same kernel with NO divergence. Both halves do the
//     identical 512-iteration body, so the warp issues it once.
// ---------------------------------------------------------------------
__global__ void noDivergence(const float* in, long long* t0, long long* t1)
{
    int lane = threadIdx.x & 31;
    float a = in[lane];
    long long base = clock64();
    t0[lane] = clock64() - base;
    for (int j = 0; j < 512; ++j) a = fmaf(a, 1.00001f, 1e-7f);
    t1[lane] = clock64() - base;
    g_sink = a;
}

// =====================================================================
int main(void)
{
    CHECK(cudaSetDevice(0));
    printf("=== Module 8 / Example 1 : warps, lanes, and active masks ===\n\n");

    // ---------------- part 1 : 100 threads, 1-D ----------------------
    {
        const int T = 100;
        Report* d_rep; CHECK(cudaMalloc(&d_rep, T * sizeof(Report)));
        roster<<<1, T>>>(d_rep);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        Report* h = (Report*)malloc(T * sizeof(Report));
        if (!h) { printf("host alloc failed\n"); return 1; }
        CHECK(cudaMemcpy(h, d_rep, T * sizeof(Report), cudaMemcpyDeviceToHost));

        printf("--- part 1 : <<<1, 100>>>  (1-D block of 100 threads) ---\n");
        printf("100 threads -> ceil(100/32) = 4 warps -> 4*32 = 128 thread slots\n");
        printf("%d thread slots are allocated and permanently idle.\n\n", 4 * 32 - T);
        for (int w = 0; w < 4; ++w) {
            int lo = -1, hi = -1, cnt = 0; unsigned m = 0;
            for (int i = 0; i < T; ++i) if (h[i].warp == w) {
                if (lo < 0) lo = h[i].tid;
                hi = h[i].tid; ++cnt; m = h[i].mask;
            }
            printf("  warp %d : tids %3d..%3d  %2d active lane(s)  "
                   "active mask 0x%08x  popc=%d\n",
                   w, lo, hi, cnt, m, popc32(m));
        }
        printf("\n  Warp 3 is a PARTIAL warp. Its 28 missing lanes were never\n"
               "  created, so they are not merely predicated off for one\n"
               "  instruction: they hold register-file space for the whole\n"
               "  lifetime of the block and can never be given work.\n\n");
        free(h); CHECK(cudaFree(d_rep));
    }

    // ---------------- part 2 : 100 threads, 2-D (10,10) --------------
    {
        const int T = 100;
        Report* d_rep; CHECK(cudaMalloc(&d_rep, T * sizeof(Report)));
        roster<<<1, dim3(10, 10)>>>(d_rep);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        Report* h = (Report*)malloc(T * sizeof(Report));
        if (!h) { printf("host alloc failed\n"); return 1; }
        CHECK(cudaMemcpy(h, d_rep, T * sizeof(Report), cudaMemcpyDeviceToHost));

        printf("--- part 2 : <<<1, dim3(10,10)>>>  (2-D block, same 100 threads) ---\n");
        printf("linearized tid = x + 10*y, then cut into groups of 32.\n");
        for (int w = 0; w < 4; ++w) {
            printf("  warp %d :", w);
            int printed = 0, lastTid = -1;
            for (int i = 0; i < T; ++i) if (h[i].warp == w) {
                if (printed == 0)
                    printf(" first lane (x,y)=(%d,%d)", h[i].tid % 10, h[i].tid / 10);
                lastTid = h[i].tid; ++printed;
            }
            printf("  last lane (x,y)=(%d,%d)  [%d lanes]\n",
                   lastTid % 10, lastTid / 10, printed);
        }
        printf("\n  Warp boundaries fall INSIDE rows. A row of this block is not\n"
               "  a warp, and two threads that are neighbours in y can be in\n"
               "  different warps while threads far apart in x are not.\n\n");
        free(h); CHECK(cudaFree(d_rep));
    }

    // ---------------- part 3 : active masks --------------------------
    {
        unsigned *d_mask, *d_ballot; int* d_reached; float* d_in;
        CHECK(cudaMalloc(&d_mask,    32 * NPT * sizeof(unsigned)));
        CHECK(cudaMalloc(&d_reached, 32 * NPT * sizeof(int)));
        CHECK(cudaMalloc(&d_ballot,  sizeof(unsigned)));
        CHECK(cudaMalloc(&d_in,      32 * sizeof(float)));
        CHECK(cudaMemset(d_mask,    0, 32 * NPT * sizeof(unsigned)));
        CHECK(cudaMemset(d_reached, 0, 32 * NPT * sizeof(int)));
        float h_in[32]; for (int i = 0; i < 32; ++i) h_in[i] = 1.0f + 0.01f * i;
        CHECK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));

        masks<<<1, 32>>>(d_in, d_mask, d_reached, d_ballot);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        unsigned hm[32 * NPT], hb; int hr[32 * NPT];
        CHECK(cudaMemcpy(hm, d_mask,    sizeof(hm), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hr, d_reached, sizeof(hr), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&hb, d_ballot, sizeof(hb), cudaMemcpyDeviceToHost));

        const char* pn[NPT] = {
            "P0  entry",
            "P1  inside  if (lane < 24)",
            "P2  inside    if ((lane & 7) == 0)",
            "P3  inside    else",
            "P4  after the inner if/else",
            "P5  after the outer if"
        };
        printf("--- part 3 : active masks in nested divergent control flow ---\n");
        printf("  __ballot_sync(0xffffffff, lane&1) = 0x%08x  popc = %d\n"
               "  (a ballot sizes the split before the branch is taken)\n\n",
               hb, popc32(hb));
        for (int p = 0; p < NPT; ++p) {
            unsigned first = 0; int any = 0, uniform = 1, n = 0;
            for (int l = 0; l < 32; ++l) if (hr[l * NPT + p]) {
                ++n;
                if (!any) { first = hm[l * NPT + p]; any = 1; }
                else if (hm[l * NPT + p] != first) uniform = 0;
            }
            printf("  %-38s : mask 0x%08x  popc=%2d  reported by %2d lane(s)%s\n",
                   pn[p], first, popc32(first), n,
                   uniform ? "" : "   [LANES DISAGREE]");
        }
        printf("\n  P2 + P3 partition P1 exactly. P4 is back to the union: the\n"
               "  compiler placed a reconvergence point there. Lanes 24..31 are\n"
               "  absent from P1..P4 and present again at P5.\n\n");
        CHECK(cudaFree(d_mask)); CHECK(cudaFree(d_reached));
        CHECK(cudaFree(d_ballot)); CHECK(cudaFree(d_in));
    }

    // ---------------- part 4 : paths are sequential ------------------
    {
        long long *d_t0, *d_t1; float* d_in;
        CHECK(cudaMalloc(&d_t0, 32 * sizeof(long long)));
        CHECK(cudaMalloc(&d_t1, 32 * sizeof(long long)));
        CHECK(cudaMalloc(&d_in, 32 * sizeof(float)));
        float h_in[32]; for (int i = 0; i < 32; ++i) h_in[i] = 1.0f + 0.01f * i;
        CHECK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));

        long long t0[32], t1[32], u0[32], u1[32];

        sequencing<<<1, 32>>>(d_in, d_t0, d_t1);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(t0, d_t0, sizeof(t0), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(t1, d_t1, sizeof(t1), cudaMemcpyDeviceToHost));

        noDivergence<<<1, 32>>>(d_in, d_t0, d_t1);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(u0, d_t0, sizeof(u0), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(u1, d_t1, sizeof(u1), cudaMemcpyDeviceToHost));

        long long a0 = t0[0], a1 = t1[0], b0 = t0[16], b1 = t1[16];
        long long lo = a0 < b0 ? a0 : b0, hi = a1 > b1 ? a1 : b1;
        long long ov = (a1 < b1 ? a1 : b1) - (a0 > b0 ? a0 : b0);
        printf("--- part 4 : the two arms of an if/else run sequentially ---\n");
        printf("  divergent kernel, one warp, both arms 512 dependent FFMAs:\n");
        printf("    arm A (lanes  0..15) : cycles [%6lld .. %6lld]  duration %6lld\n",
               a0, a1, a1 - a0);
        printf("    arm B (lanes 16..31) : cycles [%6lld .. %6lld]  duration %6lld\n",
               b0, b1, b1 - b0);
        printf("    interval overlap     : %6lld cycles %s\n", ov < 0 ? 0 : ov,
               ov <= 0 ? "(none)" : "(nonzero -- see solution notes)");
        printf("    whole if/else span   : %6lld cycles\n", hi - lo);
        printf("  control kernel, no divergence, all 32 lanes run one body:\n");
        printf("    span                 : %6lld cycles\n", u1[0] - u0[0]);
        printf("    ratio divergent/uniform = %.2fx\n",
               (double)(hi - lo) / (double)(u1[0] - u0[0]));
        printf("\n  Both arms do the same amount of work. If they ran together the\n"
               "  span would equal one duration. It equals roughly their SUM.\n"
               "  That is the additive cost of divergence, in cycles, measured.\n\n");
        CHECK(cudaFree(d_t0)); CHECK(cudaFree(d_t1)); CHECK(cudaFree(d_in));
    }

    CHECK(cudaDeviceReset());
    printf("OVERALL: PASS\n");
    return 0;
}
