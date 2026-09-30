// =====================================================================
// Module 4 / Exercise 1 : "Measure the hierarchy" (predict + measure)
//
// GOAL
//   Measure the *latency* of a single dependent load at three depths of
//   the memory hierarchy -- L1, L2, DRAM -- with a pointer chase, and
//   find the two capacity cliffs with your own numbers.
//
//   A pointer chase is the only honest way to measure latency. Every
//   load depends on the value returned by the previous one, so the warp
//   can never have two loads in flight and the hardware cannot hide
//   anything. Throughput benchmarks measure bandwidth; this measures
//   the number you actually wait.
//
//   BEFORE YOU BUILD: write down, in the TODO 4 block, the three
//   latencies you expect in cycles. Commit to numbers, not adjectives.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN:    .\exercise01.exe
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

#define NPROBE 6

// ---------------------------------------------------------------------
// The chase kernel. ONE thread, one warp, one block: we want zero
// memory-level parallelism.
//
// buf[] holds a permutation cycle: buf[i] is the *element index* of the
// next node. Following it `steps` times performs `steps` strictly
// dependent loads.
//
// clock64() reads the SM's cycle counter (%globaltimer is wall clock;
// %clock64 is SM clocks, which is what we want).
// ---------------------------------------------------------------------
__global__ void chase(const int* __restrict__ buf, int steps,
                      int* out, long long* cycles)
{
    int p = 0;

    // -----------------------------------------------------------------
    // TODO 3: The measurement below must report *steady-state* latency:
    //         the cost of a load once the hierarchy has settled into the
    //         state this working set produces. As written, the first
    //         time each line is touched it is a compulsory miss all the
    //         way to DRAM, no matter how small the buffer is.
    //
    //         Add whatever is needed here so that the timed region
    //         measures steady state. Do not change the timed region.
    // -----------------------------------------------------------------
    // YOUR CODE HERE (TODO 3)

    long long t0 = clock64();
    for (int i = 0; i < steps; ++i) p = buf[p];
    long long t1 = clock64();

    *cycles = t1 - t0;
    *out    = p;             // keeps the chain from being optimized away
}

// ---------------------------------------------------------------------
// Build a single Hamiltonian cycle over `nodes` nodes spaced
// `strideElems` apart.
//
// The order is randomized *within* 2 MB chunks, and the chunks are
// visited in address order. Two separate concerns are being balanced:
//
//  - randomness inside a chunk defeats any locality the memory system
//    could exploit between one link and the next, which is what we
//    want: every link must be a real, independent lookup;
//
//  - keeping the randomness *local* keeps the GPU's address-translation
//    hardware (TLB) in range. A uniformly random walk over a 384 MB
//    buffer misses the TLB on nearly every link, and you end up
//    measuring page-table walks (~1400 cycles/load here) rather than
//    the memory hierarchy. That is a real effect, but it is not the one
//    this exercise is about.
// ---------------------------------------------------------------------
static void build_chain(int* h, size_t nodes, int strideElems, unsigned seed)
{
    const size_t CHUNK_BYTES    = 2u << 20;      // GPU large-page size
    size_t nodesPerChunk        = CHUNK_BYTES / sizeof(int) / (size_t)strideElems;
    if (nodesPerChunk < 1) nodesPerChunk = 1;
    size_t nChunks              = (nodes + nodesPerChunk - 1) / nodesPerChunk;

    int* order = (int*)malloc(nodes * sizeof(int));
    srand(seed);
    for (size_t c = 0; c < nChunks; ++c) {
        size_t b = c * nodesPerChunk;
        size_t e = b + nodesPerChunk; if (e > nodes) e = nodes;
        for (size_t i = b; i < e; ++i) order[i] = (int)i;
        for (size_t i = e - 1; i > b; --i) {           // Fisher-Yates in-chunk
            size_t j = b + (size_t)rand() % (i - b + 1);
            int t = order[i]; order[i] = order[j]; order[j] = t;
        }
    }
    for (size_t i = 0; i < nodes; ++i)
        h[(size_t)order[i] * strideElems] = order[(i + 1) % nodes] * strideElems;
    free(order);
}

int main(void)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    int l2Bytes = 0;
    CHECK(cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, dev));

    // The unified L1 + shared block on Ada is 128 KB per SM. With no
    // shared memory requested, essentially all of it is available as L1.
    const size_t L1_BYTES = 128 * 1024;

    printf("=== %s : L1+SMEM = %zu KB/SM, L2 = %.0f MB ===\n",
           p.name, L1_BYTES / 1024, l2Bytes / (1024.0 * 1024.0));

    // -----------------------------------------------------------------
    // TODO 1: Choose the distance in BYTES between consecutive nodes of
    //         the chain.
    //
    //         Requirement: following one link must always force a new
    //         request to the memory system. Two nodes that happen to
    //         land in the same cache line would make the second load
    //         free, and the whole curve would flatten.
    //
    //         The L1/L2 line on sm_89 is 128 B and the L2-to-DRAM
    //         transaction granularity is a 32 B sector. Pick a stride
    //         that satisfies the requirement and justify it in one line
    //         in your write-up. There is a smallest correct answer;
    //         find it, and do not go wildly above it (a huge stride
    //         wastes buffer and starts measuring page-table behaviour
    //         instead).
    // -----------------------------------------------------------------
    int strideBytes = 0;   // YOUR CODE HERE (TODO 1)

    if (strideBytes < (int)sizeof(int)) {
        printf("\nSet TODO 1 (strideBytes) first.\n");
        return 0;
    }
    int strideElems = strideBytes / (int)sizeof(int);

    // -----------------------------------------------------------------
    // TODO 2: Fill in SIX probe sizes, in bytes, as
    //           two that must be served by L1,
    //           two that must miss L1 but be served by L2,
    //           two that must miss L2 and reach DRAM.
    //
    //         Derive them from L1_BYTES and l2Bytes -- do not hard-code
    //         a magic constant. The validator will check that the two
    //         probes in each regime measure within 35% of each other,
    //         and that the three regimes are separated by at least 1.8x.
    //
    //         Think carefully about the margins. "Bigger than L2" by
    //         10% is not bigger than L2 in practice: a cache is not a
    //         cliff, it is a hit-rate curve, and 48 MB of L2 is large
    //         enough to keep an entire benchmark buffer resident while
    //         you believe you are measuring DRAM. Choose margins that
    //         make the answer unambiguous.
    // -----------------------------------------------------------------
    size_t probeBytes[NPROBE] = { 0, 0, 0, 0, 0, 0 };
    // YOUR CODE HERE (TODO 2)

    int unset = 0;
    for (int i = 0; i < NPROBE; ++i) if (probeBytes[i] == 0) ++unset;
    if (unset) {
        printf("\nSet TODO 2 (all %d probe sizes) first.\n", NPROBE);
        return 0;
    }

    // -----------------------------------------------------------------
    // TODO 4: Your predictions, in SM cycles per dependent load, BEFORE
    //         you run this. One number per regime. You are allowed to
    //         reason from: an FMA has ~4 cycle latency; the SM must be
    //         able to hide global latency with at most 48 resident
    //         warps; 432 GB/s at ~1.5 GHz.
    // -----------------------------------------------------------------
    double predL1   = 0.0;   // YOUR CODE HERE (TODO 4)
    double predL2   = 0.0;   // YOUR CODE HERE (TODO 4)
    double predDRAM = 0.0;   // YOUR CODE HERE (TODO 4)

    // -----------------------------------------------------------------
    int      *d_out = nullptr;
    long long*d_cyc = nullptr;
    CHECK(cudaMalloc(&d_out, sizeof(int)));
    CHECK(cudaMalloc(&d_cyc, sizeof(long long)));

    double lat[NPROBE];

    printf("\n stride = %d B\n", strideBytes);
    printf("%14s %12s %10s %14s\n", "buffer bytes", "nodes", "steps", "cycles/load");

    for (int k = 0; k < NPROBE; ++k) {
        size_t bytes = probeBytes[k];
        size_t nodes = bytes / sizeof(int) / (size_t)strideElems;
        if (nodes < 16) { printf("  probe %d too small for this stride\n", k); return 1; }

        // One full traversal of every node, so the "working set" really
        // is the whole buffer and not just the part we happened to walk.
        int steps = (int)(nodes > (1u << 22) ? (1u << 22) : nodes);

        int* h = (int*)malloc(bytes);
        build_chain(h, nodes, strideElems, 1234u + k);

        int* d = nullptr;
        CHECK(cudaMalloc(&d, bytes));
        CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));

        chase<<<1, 1>>>(d, steps, d_out, d_cyc);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        long long c = 0;
        CHECK(cudaMemcpy(&c, d_cyc, sizeof(long long), cudaMemcpyDeviceToHost));
        lat[k] = (double)c / steps;

        printf("%14zu %12zu %10d %14.1f\n", bytes, nodes, steps, lat[k]);

        CHECK(cudaFree(d));
        free(h);
    }

    // -----------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------
    double l1   = 0.5 * (lat[0] + lat[1]);
    double l2   = 0.5 * (lat[2] + lat[3]);
    double dram = 0.5 * (lat[4] + lat[5]);

    printf("\n--- predicted vs measured (cycles / dependent load) ---\n");
    printf("%8s %12s %12s %10s\n", "regime", "predicted", "measured", "ratio");
    printf("%8s %12.0f %12.1f %9.2fx\n", "L1",   predL1,   l1,   l1   / (predL1   > 0 ? predL1   : 1));
    printf("%8s %12.0f %12.1f %9.2fx\n", "L2",   predL2,   l2,   l2   / (predL2   > 0 ? predL2   : 1));
    printf("%8s %12.0f %12.1f %9.2fx\n", "DRAM", predDRAM, dram, dram / (predDRAM > 0 ? predDRAM : 1));

    int pass = 1;

    // (a) each regime's two probes must agree -> the size really is in
    //     that regime and not straddling a boundary
    const char* names[3] = { "L1", "L2", "DRAM" };
    for (int r = 0; r < 3; ++r) {
        double a = lat[2 * r], b = lat[2 * r + 1];
        double hi = a > b ? a : b, lo = a > b ? b : a;
        if (hi > 1.35 * lo) {
            printf("  [FAIL] %s probes disagree (%.1f vs %.1f): "
                   "at least one is not in that regime\n", names[r], a, b);
            pass = 0;
        }
    }
    // (b) the regimes must be clearly separated
    if (!(l2 > 1.8 * l1))   { printf("  [FAIL] L2 not clearly slower than L1 (%.1f vs %.1f)\n", l2, l1);   pass = 0; }
    if (!(dram > 1.8 * l2)) { printf("  [FAIL] DRAM not clearly slower than L2 (%.1f vs %.1f)\n", dram, l2); pass = 0; }

    // (c) predictions committed and within 2x
    if (predL1 <= 0 || predL2 <= 0 || predDRAM <= 0) {
        printf("  [FAIL] TODO 4: commit to three predictions before running.\n");
        pass = 0;
    } else {
        double r1 = l1 / predL1, r2 = l2 / predL2, r3 = dram / predDRAM;
        if (r1 > 2.0 || r1 < 0.5 || r2 > 2.0 || r2 < 0.5 || r3 > 2.0 || r3 < 0.5) {
            printf("  [FAIL] at least one prediction is off by more than 2x. "
                   "Work out why before you change the numbers.\n");
            pass = 0;
        }
    }

    printf("\n%s\n", pass ? "PASS" : "FAIL");

    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_cyc));
    CHECK(cudaDeviceReset());
    return pass ? 0 : 1;
}
