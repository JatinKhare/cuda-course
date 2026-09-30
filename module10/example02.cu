// =====================================================================
// Module 10 / Example 2 : "The price of an address"
//
// GOAL
//   Measure the full contention curve. The number of atomic instructions is
//   held EXACTLY constant (2^20 per launch) in every configuration; only the
//   set of addresses they target changes. Everything that moves in the timing
//   column is therefore contention, not work.
//
//   Five experiments:
//     A  global atomics over K distinct addresses, K = 1 .. 2^20
//     B  the same small K, but with the addresses spread by a word stride
//     C  warp-uniform address (32 lanes -> 1 address) vs lane-varying
//     D  shared-memory privatization vs the same traffic straight to global
//     E  explicit warp-level pre-aggregation with __match_any_sync
//
//   Timing follows the spec's methodology: every configuration is timed
//   back-to-back inside one sweep, the sweep is repeated 4 times, and the
//   MINIMUM is reported. Validation happens in a separate pass afterwards.
//   Ratios are the stable quantity; absolute ms drift with the clock.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
// RUN:   .\example02.exe
//
// SASS (this is Part F of the lesson -- ATOM vs RED):
//   nvcc -arch=sm_89 -O3 -cubin -o example02.cubin example02.cu
//   cuobjdump -sass example02.cubin > sass.txt
//   grep -E "Function : |ATOM|RED|VOTEU|POPC|SHFL" sass.txt
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static const int TPB  = 256;
static const int GRID = 4096;                       // 1,048,576 threads
static const long long NATOM = (long long)TPB * GRID;
static const int SWEEPS = 4;
static const int ITERS  = 20;

// ---------------------------------------------------------------------
// The kernels. Note that `mask` is a RUNTIME argument in every one of
// them. That matters: the compiler cannot prove the address is uniform,
// so it cannot pre-aggregate. Compare k_literal below.
// ---------------------------------------------------------------------

// A: lane-varying address, K = mask+1 consecutive words.
__global__ void k_spread(unsigned int* c, unsigned int mask)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&c[i & mask], 1u);
}

// B: same K, but consecutive bins are `stride` words apart.
__global__ void k_stride(unsigned int* c, unsigned int mask, unsigned int stride)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&c[(i & mask) * stride], 1u);
}

// C: the address is constant across a warp; consecutive warps differ.
__global__ void k_warp_uniform(unsigned int* c, unsigned int mask)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&c[(i >> 5) & mask], 1u);
}

// The non-atomic baseline: the SAME memory traffic, racily. Wrong answer,
// right cost floor. This is what an atomic is being compared against.
__global__ void k_plain_store(unsigned int* c, unsigned int mask)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i & mask] += 1u;
}

// D: per-block privatization in shared memory. K bins private per block,
// K shared atomics' worth of contention absorbed at the SM, then exactly
// K global atomics per block to flush.
__global__ void k_privatized(unsigned int* c, unsigned int mask)
{
    extern __shared__ unsigned int s[];
    const unsigned int K = mask + 1u;
    for (unsigned int t = threadIdx.x; t < K; t += blockDim.x) s[t] = 0u;
    __syncthreads();                       // Module 9: the private copy must
                                           // be zeroed before anyone adds.
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&s[i & mask], 1u);
    __syncthreads();                       // ...and all adds must be done
                                           // before anyone reads for the flush.
    for (unsigned int t = threadIdx.x; t < K; t += blockDim.x)
        if (s[t]) atomicAdd(&c[t], s[t]);
}

// E: explicit warp-level pre-aggregation. Lanes of a warp that share a
// target elect one leader, which performs a single atomic carrying the
// whole warp's contribution.
__global__ void k_warp_aggregated(unsigned int* c, unsigned int mask)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int addr = i & mask;
    unsigned int active = __activemask();
    unsigned int peers  = __match_any_sync(active, addr);  // sm_70+
    int leader = __ffs(peers) - 1;
    unsigned int n = __popc(peers);
    if ((int)(threadIdx.x & 31u) == leader) atomicAdd(&c[addr], n);
}

// F: address is a compile-time constant -> the compiler proves warp
// uniformity and emits the aggregation itself. Same traffic as k_spread
// with mask=0, radically different SASS.
__global__ void k_literal(unsigned int* c)
{
    atomicAdd(&c[0], 1u);
}

// ---------------------------------------------------------------------
static double best_ms(double cur, float ms) { double v = ms / ITERS; return v < cur ? v : cur; }

int main(void)
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("Every configuration below executes EXACTLY %lld atomic instructions.\n",
           NATOM);
    printf("Timing: min of %d sweeps x %d iterations, all configs back to back.\n\n",
           SWEEPS, ITERS);

    unsigned int* d = nullptr;
    const size_t NWORDS = (size_t)1 << 22;             // 4 Mwords = 16 MB
    CHECK(cudaMalloc(&d, NWORDS * sizeof(unsigned int)));
    CHECK(cudaMemset(d, 0, NWORDS * sizeof(unsigned int)));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));

    // Duration-based clock warm-up: run long enough to leave the idle clock
    // state before the first timed iteration (spec 12.4).
    for (int i = 0; i < 300; ++i) k_spread<<<GRID, TPB>>>(d, 0u);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    const int KS[] = { 1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096, 65536, 1048576 };
    const int NK = (int)(sizeof(KS) / sizeof(KS[0]));
    const unsigned int STRIDES[] = { 1u, 8u, 32u, 1024u };
    const int NS = 4;

    double tSpread[16], tWarpU[16], tPlain[16], tPriv[16], tAggr[16];
    double tStride[16][4];
    double tLiteral = 1e30;
    for (int i = 0; i < NK; ++i) {
        tSpread[i] = tWarpU[i] = tPlain[i] = tPriv[i] = tAggr[i] = 1e30;
        for (int j = 0; j < NS; ++j) tStride[i][j] = 1e30;
    }

#define TIME_BLOCK(store, launch)                                              \
    do {                                                                       \
        launch; CHECK(cudaDeviceSynchronize());                                \
        CHECK(cudaEventRecord(e0));                                            \
        for (int r_ = 0; r_ < ITERS; ++r_) { launch; }                         \
        CHECK(cudaEventRecord(e1));                                            \
        CHECK(cudaEventSynchronize(e1));                                       \
        float ms_ = 0.f; CHECK(cudaEventElapsedTime(&ms_, e0, e1));            \
        (store) = best_ms((store), ms_);                                       \
    } while (0)

    // ---------- one loop, all configurations, no printing inside ----------
    for (int sweep = 0; sweep < SWEEPS; ++sweep) {
        TIME_BLOCK(tLiteral, (k_literal<<<GRID, TPB>>>(d)));
        for (int i = 0; i < NK; ++i) {
            const unsigned int mask = (unsigned int)KS[i] - 1u;
            TIME_BLOCK(tSpread[i], (k_spread<<<GRID, TPB>>>(d, mask)));
            TIME_BLOCK(tWarpU[i],  (k_warp_uniform<<<GRID, TPB>>>(d, mask)));
            TIME_BLOCK(tPlain[i],  (k_plain_store<<<GRID, TPB>>>(d, mask)));
            TIME_BLOCK(tAggr[i],   (k_warp_aggregated<<<GRID, TPB>>>(d, mask)));
            const size_t shm = (size_t)KS[i] * sizeof(unsigned int);
            if (shm <= 48u * 1024u)
                TIME_BLOCK(tPriv[i], (k_privatized<<<GRID, TPB, shm>>>(d, mask)));
            if (KS[i] <= 32)
                for (int j = 0; j < NS; ++j)
                    TIME_BLOCK(tStride[i][j], (k_stride<<<GRID, TPB>>>(d, mask, STRIDES[j])));
        }
    }
    CHECK(cudaGetLastError());
#undef TIME_BLOCK

    // ---------- validation, second pass (spec 12.2) ----------
    printf("=== validation pass (correctness, untimed) ===\n");
    {
        int allOk = 1;
        const int checkK[3] = { 1, 256, 4096 };
        for (int t = 0; t < 3; ++t) {
            const unsigned int mask = (unsigned int)checkK[t] - 1u;
            for (int variant = 0; variant < 3; ++variant) {
                CHECK(cudaMemset(d, 0, NWORDS * sizeof(unsigned int)));
                if (variant == 0)      k_spread<<<GRID, TPB>>>(d, mask);
                else if (variant == 1) k_warp_aggregated<<<GRID, TPB>>>(d, mask);
                else                   k_privatized<<<GRID, TPB, (size_t)checkK[t] * 4>>>(d, mask);
                CHECK(cudaGetLastError());
                CHECK(cudaDeviceSynchronize());
                unsigned int* h = (unsigned int*)malloc((size_t)checkK[t] * 4);
                CHECK(cudaMemcpy(h, d, (size_t)checkK[t] * 4, cudaMemcpyDeviceToHost));
                const unsigned int expect = (unsigned int)(NATOM / checkK[t]);
                int ok = 1;
                for (int b = 0; b < checkK[t]; ++b) if (h[b] != expect) ok = 0;
                free(h);
                if (!ok) allOk = 0;
                printf("  K=%-8d %-18s each bin == %-9u  %s\n", checkK[t],
                       variant == 0 ? "plain atomic" : variant == 1 ? "warp-aggregated"
                                                                    : "privatized",
                       expect, ok ? "PASS" : "FAIL");
            }
        }
        // And the racy one, to show it is NOT a correctness-preserving option.
        CHECK(cudaMemset(d, 0, NWORDS * sizeof(unsigned int)));
        k_plain_store<<<GRID, TPB>>>(d, 255u);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        unsigned int h0 = 0;
        CHECK(cudaMemcpy(&h0, d, 4, cudaMemcpyDeviceToHost));
        printf("  K=256    non-atomic baseline  bin 0 == %-9u  (expected %lld) "
               "-- this column is a COST FLOOR, not a kernel\n\n",
               h0, NATOM / 256);
        if (!allOk) { printf("OVERALL: FAIL\n"); return 1; }
    }

    // ---------- A: the contention curve ----------
    printf("=== A. global atomicAdd, %lld atomics over K distinct addresses ===\n", NATOM);
    printf("  %9s %10s %12s %12s %10s %10s\n",
           "K", "ms", "Gatomic/s", "vs K=2^20", "non-atomic", "atom/store");
    for (int i = 0; i < NK; ++i)
        printf("  %9d %10.4f %12.2f %11.2fx %10.4f %9.1fx\n",
               KS[i], tSpread[i], NATOM / (tSpread[i] * 1e-3) / 1e9,
               tSpread[i] / tSpread[NK - 1], tPlain[i], tSpread[i] / tPlain[i]);
    printf("\n  The rightmost column is the honest price of atomicity: an\n"
           "  UNCONTENDED atomic costs about what a plain store costs. The\n"
           "  entire cost of the K=1 row is contention.\n\n");

    // ---------- B: address spreading ----------
    printf("=== B. same K, bins spaced `stride` words apart (ms) ===\n");
    printf("  %9s", "K");
    for (int j = 0; j < NS; ++j) printf(" %12s", "stride");
    printf("\n  %9s", "");
    for (int j = 0; j < NS; ++j) printf(" %9u(%3uB)", STRIDES[j], STRIDES[j] * 4u);
    printf("\n");
    for (int i = 0; i < NK && KS[i] <= 32; ++i) {
        printf("  %9d", KS[i]);
        for (int j = 0; j < NS; ++j) printf(" %12.4f", tStride[i][j]);
        printf("\n");
    }
    printf("\n  K distinct addresses do not buy K-way parallelism unless they\n"
           "  land on K distinct L2 slices. Adjacent words often do not.\n\n");

    // ---------- C: warp-uniform vs lane-varying ----------
    printf("=== C. all 32 lanes of a warp hitting ONE address ===\n");
    printf("  %9s %14s %14s %10s\n", "K", "lane-varying", "warp-uniform", "ratio");
    for (int i = 0; i < NK; ++i)
        printf("  %9d %14.4f %14.4f %9.2fx\n",
               KS[i], tSpread[i], tWarpU[i], tWarpU[i] / tSpread[i]);
    printf("\n  Even with a million distinct addresses in play, making the\n"
           "  address warp-uniform reintroduces most of the contention cost.\n"
           "  It is same-address concurrency that is expensive, not atomics.\n\n");

    // ---------- D + E ----------
    printf("=== D/E. privatization and warp aggregation (ms) ===\n");
    printf("  %9s %12s %12s %10s %12s %10s\n",
           "K", "plain", "privatized", "speedup", "warp-aggr", "speedup");
    for (int i = 0; i < NK; ++i) {
        printf("  %9d %12.4f", KS[i], tSpread[i]);
        if (tPriv[i] < 1e29) printf(" %12.4f %9.2fx", tPriv[i], tSpread[i] / tPriv[i]);
        else                 printf(" %12s %10s", "(>48KB)", "-");
        printf(" %12.4f %9.2fx\n", tAggr[i], tSpread[i] / tAggr[i]);
    }
    printf("\n  Both are contention-reduction techniques and both have a cost\n"
           "  floor. Read the columns for where each stops paying.\n\n");

    // ---------- F: what the compiler already did ----------
    printf("=== F. the compiler's own aggregation ===\n");
    printf("  atomicAdd(&c[i & mask], 1)  with runtime mask == 0 : %.4f ms\n",
           tSpread[0]);
    printf("  atomicAdd(&c[0], 1)         literal address        : %.4f ms\n", tLiteral);
    printf("  ratio: %.2fx for IDENTICAL memory traffic.\n\n", tSpread[0] / tLiteral);
    printf("  Dump the SASS and compare _Z9k_literalPj against _Z8k_spreadPjj:\n"
           "    nvcc -arch=sm_89 -O3 -cubin -o example02.cubin example02.cu\n"
           "    cuobjdump -sass example02.cubin\n"
           "  See the lesson's \"ATOM vs RED\" section.\n\n");

    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    CHECK(cudaFree(d));
    CHECK(cudaDeviceReset());
    printf("OVERALL: PASS\n");
    return 0;
}
