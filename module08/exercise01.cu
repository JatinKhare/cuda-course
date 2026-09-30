// =====================================================================
// Module 8 / Exercise 1 : "which lanes are executing right now?"
//                         (predict first, then let the machine score you)
//
// GOAL
//   One warp. One kernel. Eleven labelled program points plus one loop.
//   For each point you must write down, as a 32-bit hex constant, the
//   exact active mask the warp presents when it issues the instruction
//   at that point -- bit L set means lane L is participating.
//
//   Then you must state how many times the warp ISSUES the recording
//   instruction in total, and you must express the rule that produces
//   that number as code (TODO 5).
//
//   The program records the real masks and the real per-site issue
//   counts in hardware and scores your predictions.
//
// THE KERNEL (read it carefully; it is reproduced verbatim below)
//
//     lane = threadIdx.x & 31;
//     REC(0);
//     if (lane < 20) {
//         REC(1);
//         <9 dependent FFMAs>
//         if ((lane & 3) == 0) { REC(2); <9 FFMAs> }
//         else                 { REC(3); <9 FFMAs> }
//         REC(4);
//     } else {
//         REC(5);
//         <9 dependent FFMAs>
//         if (lane >= 28) { REC(6); return; }
//         REC(7);
//     }
//     REC(8);
//     if (lane & 1) { REC(9);  a = a + 1.0f; }
//     else          { REC(10); a = a * 2.0f; }
//     trips = (lane & 7) + 1;
//     for (j = 0; j < trips; ++j) { RECIT(j); <1 FFMA> }
//
//   REC(p)    records __activemask() at site p.
//   RECIT(j)  records __activemask() at loop iteration j.
//   Both also bump a per-lane execution counter, and both bump a
//   per-site WARP-LEVEL issue counter incremented by exactly one lane
//   per issue (the lowest active lane), which is the hardware ground
//   truth for "how many times did the warp issue this instruction".
//
// RULES
//   Work it out on paper from the SIMT contract before you run anything.
//   Bit L of the mask corresponds to lane L. A mask value of 0 is the
//   "not filled in" sentinel; the program refuses to run until every
//   slot is set.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
// RUN:    .\exercise01.exe
//
// Useful, and worth doing before you commit to TODO 3:
//   nvcc -arch=sm_89 -O3 -c -o exercise01.o exercise01.cu
//   cuobjdump -sass exercise01.o > exercise01.sass
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

#define NSITE 11        // REC(0) .. REC(10)
#define MAXIT 8         // maximum loop trip count

// =====================================================================
//                        YOUR PREDICTIONS
// =====================================================================

// ---------------------------------------------------------------------
// TODO 1: the masks on the `lane < 20` side of the outer branch.
//         P0 is kernel entry; P1 is the first instruction inside the
//         taken arm; P2 and P3 are the two arms of the inner branch;
//         P4 is the first instruction after the inner branch.
// ---------------------------------------------------------------------
static const unsigned P0  = 0x00000000u;   // YOUR CODE HERE (TODO 1)
static const unsigned P1  = 0x00000000u;   // YOUR CODE HERE (TODO 1)
static const unsigned P2  = 0x00000000u;   // YOUR CODE HERE (TODO 1)
static const unsigned P3  = 0x00000000u;   // YOUR CODE HERE (TODO 1)
static const unsigned P4  = 0x00000000u;   // YOUR CODE HERE (TODO 1)

// ---------------------------------------------------------------------
// TODO 2: the masks on the `else` side, and at the point where the two
//         sides come back together.
//         P5 is the first instruction of the else arm; P6 is inside the
//         early `return`; P7 is after it; P8 is the first instruction
//         after the whole outer if/else.
//         Think hard about P8.
// ---------------------------------------------------------------------
static const unsigned P5  = 0x00000000u;   // YOUR CODE HERE (TODO 2)
static const unsigned P6  = 0x00000000u;   // YOUR CODE HERE (TODO 2)
static const unsigned P7  = 0x00000000u;   // YOUR CODE HERE (TODO 2)
static const unsigned P8  = 0x00000000u;   // YOUR CODE HERE (TODO 2)

// ---------------------------------------------------------------------
// TODO 3: the two arms of
//             if (lane & 1) { REC(9); a = a + 1.0f; }
//             else          { REC(10); a = a * 2.0f; }
//         The bodies are one arithmetic instruction each. Before you
//         answer, decide what the compiler will emit for a branch this
//         short, and what that implies for the active mask. Then check
//         your answer against the SASS with cuobjdump.
// ---------------------------------------------------------------------
static const unsigned P9  = 0x00000000u;   // YOUR CODE HERE (TODO 3)
static const unsigned P10 = 0x00000000u;   // YOUR CODE HERE (TODO 3)

// ---------------------------------------------------------------------
// TODO 4: the loop
//             trips = (lane & 7) + 1;
//             for (j = 0; j < trips; ++j) { RECIT(j); ... }
//         Give the active mask the warp presents on EACH iteration j,
//         j = 0..7. Remember which lanes are still alive at all.
// ---------------------------------------------------------------------
static const unsigned IT[MAXIT] = {
    0x00000000u,   // j = 0   YOUR CODE HERE (TODO 4)
    0x00000000u,   // j = 1   YOUR CODE HERE (TODO 4)
    0x00000000u,   // j = 2   YOUR CODE HERE (TODO 4)
    0x00000000u,   // j = 3   YOUR CODE HERE (TODO 4)
    0x00000000u,   // j = 4   YOUR CODE HERE (TODO 4)
    0x00000000u,   // j = 5   YOUR CODE HERE (TODO 4)
    0x00000000u,   // j = 6   YOUR CODE HERE (TODO 4)
    0x00000000u    // j = 7   YOUR CODE HERE (TODO 4)
};

// ---------------------------------------------------------------------
// TODO 5 (design, not an expression).
//
//   The kernel gives you, for free, a per-lane execution count:
//       laneCount[lane * (NSITE+1) + site]
//   = how many times lane `lane` executed site `site`. Site index NSITE
//   is the loop body; sites 0..NSITE-1 are REC(0)..REC(10).
//
//   From that table alone, compute how many INSTRUCTION ISSUES the warp
//   spends on each site, and write the per-site answer into issueOut[].
//   The SIMT contract determines the rule; derive it. Do not use the
//   hardware measurement (that is what you are being scored against),
//   and do not hard-code the numbers -- your function must be correct
//   for any laneCount table produced by any kernel of this shape.
//
//   Return the total across all sites, or -1 to say "not implemented".
// ---------------------------------------------------------------------
static long long warpIssuesFromLaneCounts(const int* laneCount,
                                          int nLanes, int nSites,
                                          long long* issueOut)
{
    (void)laneCount; (void)nLanes; (void)nSites; (void)issueOut;
    // YOUR CODE HERE (TODO 5)
    return -1;
}

// =====================================================================
//                            THE KERNEL
// =====================================================================

__device__ float g_sink;

// One block, one warp, so a plain non-atomic increment of issue[] is
// race-free: exactly one lane executes it per issue of the site.
#define LOWEST(m) (__ffs((int)(m)) - 1)

#define REC(p) do {                                                      \
        unsigned m_ = __activemask();                                    \
        mask[lane * NSITE + (p)] = m_;                                   \
        reached[lane * NSITE + (p)] = 1;                                 \
        laneCount[lane * (NSITE + 1) + (p)] += 1;                        \
        if (lane == LOWEST(m_)) issue[(p)] += 1;                         \
    } while (0)

__global__ void probe(const float* __restrict__ in,
                      unsigned* __restrict__ mask,
                      int* __restrict__ reached,
                      int* __restrict__ laneCount,
                      long long* __restrict__ issue,
                      unsigned* __restrict__ itmask)
{
    int lane = threadIdx.x & 31;
    float a  = in[lane];

    REC(0);
    if (lane < 20) {
        REC(1);
        for (int j = 0; j < 9; ++j) a = fmaf(a, 1.0001f, 1e-6f);
        if ((lane & 3) == 0) {
            REC(2);
            for (int j = 0; j < 9; ++j) a = fmaf(a, 1.0002f, 2e-6f);
        } else {
            REC(3);
            for (int j = 0; j < 9; ++j) a = fmaf(a, 1.0003f, 3e-6f);
        }
        REC(4);
    } else {
        REC(5);
        for (int j = 0; j < 9; ++j) a = fmaf(a, 1.0004f, 4e-6f);
        if (lane >= 28) { REC(6); g_sink = a; return; }
        REC(7);
    }
    REC(8);

    if (lane & 1) { REC(9);  a = a + 1.0f; }
    else          { REC(10); a = a * 2.0f; }

    int trips = (lane & 7) + 1;
    for (int j = 0; j < trips; ++j) {
        unsigned m_ = __activemask();
        itmask[lane * MAXIT + j] = m_;
        laneCount[lane * (NSITE + 1) + NSITE] += 1;
        if (lane == LOWEST(m_)) issue[NSITE] += 1;
        a = fmaf(a, 1.0001f, 1e-6f);
    }
    g_sink = a;
}

// =====================================================================
static int popc32(unsigned v) { int c = 0; while (v) { c += (int)(v & 1u); v >>= 1; } return c; }

int main(void)
{
    const unsigned pred[NSITE] = { P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10 };

    // --- refuse to run with unfilled predictions ---------------------
    for (int p = 0; p < NSITE; ++p)
        if (pred[p] == 0u) { printf("Set TODO %d first.\n", p < 5 ? 1 : (p < 9 ? 2 : 3)); return 0; }
    for (int j = 0; j < MAXIT; ++j)
        if (IT[j] == 0u) { printf("Set TODO 4 first.\n"); return 0; }

    CHECK(cudaSetDevice(0));
    printf("=== Module 8 / Exercise 1 : lane-level prediction ===\n");
    printf("one block, one warp, 32 lanes\n\n");

    unsigned *d_mask, *d_it; int *d_reached, *d_laneCount; long long* d_issue;
    float* d_in;
    CHECK(cudaMalloc(&d_mask,      32 * NSITE * sizeof(unsigned)));
    CHECK(cudaMalloc(&d_it,        32 * MAXIT * sizeof(unsigned)));
    CHECK(cudaMalloc(&d_reached,   32 * NSITE * sizeof(int)));
    CHECK(cudaMalloc(&d_laneCount, 32 * (NSITE + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_issue,     (NSITE + 1) * sizeof(long long)));
    CHECK(cudaMalloc(&d_in,        32 * sizeof(float)));
    CHECK(cudaMemset(d_mask,      0, 32 * NSITE * sizeof(unsigned)));
    CHECK(cudaMemset(d_it,        0, 32 * MAXIT * sizeof(unsigned)));
    CHECK(cudaMemset(d_reached,   0, 32 * NSITE * sizeof(int)));
    CHECK(cudaMemset(d_laneCount, 0, 32 * (NSITE + 1) * sizeof(int)));
    CHECK(cudaMemset(d_issue,     0, (NSITE + 1) * sizeof(long long)));

    float h_in[32];
    for (int i = 0; i < 32; ++i) h_in[i] = 1.0f + 0.01f * (float)i;   // deterministic
    CHECK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));

    probe<<<1, 32>>>(d_in, d_mask, d_reached, d_laneCount, d_issue, d_it);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    unsigned hm[32 * NSITE], hit[32 * MAXIT];
    int hr[32 * NSITE], hc[32 * (NSITE + 1)];
    long long hi[NSITE + 1];
    CHECK(cudaMemcpy(hm, d_mask,      sizeof(hm), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hit, d_it,       sizeof(hit), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hr, d_reached,   sizeof(hr), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hc, d_laneCount, sizeof(hc), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hi, d_issue,     sizeof(hi), cudaMemcpyDeviceToHost));

    // ---- reduce the per-lane masks to one value per site -------------
    unsigned actual[NSITE]; int disagree[NSITE], nrep[NSITE];
    for (int p = 0; p < NSITE; ++p) {
        unsigned first = 0; int any = 0, uni = 1, n = 0;
        for (int l = 0; l < 32; ++l) if (hr[l * NSITE + p]) {
            ++n;
            if (!any) { first = hm[l * NSITE + p]; any = 1; }
            else if (hm[l * NSITE + p] != first) uni = 0;
        }
        actual[p] = first; disagree[p] = !uni; nrep[p] = n;
    }
    unsigned actualIt[MAXIT]; int itDisagree[MAXIT];
    for (int j = 0; j < MAXIT; ++j) {
        unsigned first = 0; int any = 0, uni = 1;
        for (int l = 0; l < 32; ++l) {
            unsigned v = hit[l * MAXIT + j];
            if (v) { if (!any) { first = v; any = 1; } else if (v != first) uni = 0; }
        }
        actualIt[j] = first; itDisagree[j] = !uni;
    }

    // ---- score -------------------------------------------------------
    int score = 0, total = 0;
    printf("--- masks at the labelled sites ---\n");
    printf("  %-5s %-12s %-12s %-5s %-7s\n", "site", "predicted", "actual", "popc", "lanes");
    for (int p = 0; p < NSITE; ++p) {
        int ok = (pred[p] == actual[p]);
        score += ok; ++total;
        printf("  P%-4d 0x%08x   0x%08x   %2d    %2d      %s%s\n", p, pred[p], actual[p],
               popc32(actual[p]), nrep[p], ok ? "MATCH" : "MISMATCH",
               disagree[p] ? "   [lanes disagreed!]" : "");
    }
    printf("\n--- masks on each iteration of the variable-trip loop ---\n");
    for (int j = 0; j < MAXIT; ++j) {
        int ok = (IT[j] == actualIt[j]);
        score += ok; ++total;
        printf("  j=%d   0x%08x   0x%08x   %2d     %s%s\n", j, IT[j], actualIt[j],
               popc32(actualIt[j]), ok ? "MATCH" : "MISMATCH",
               itDisagree[j] ? "   [lanes disagreed!]" : "");
    }

    // ---- TODO 5 : the issue-count model ------------------------------
    printf("\n--- instruction issues per site ---\n");
    long long model[NSITE + 1];
    for (int p = 0; p <= NSITE; ++p) model[p] = -1;
    long long modelTotal = warpIssuesFromLaneCounts(hc, 32, NSITE + 1, model);

    long long hwTotal = 0;
    for (int p = 0; p <= NSITE; ++p) hwTotal += hi[p];

    if (modelTotal < 0) {
        printf("  TODO 5 not implemented. Set TODO 5 first.\n");
        printf("  (hardware measured %lld total issues; your model must reproduce it)\n",
               hwTotal);
    } else {
        int allok = 1;
        printf("  %-6s %-14s %-14s %-12s %s\n", "site", "lanes that ran",
               "your model", "hardware", "");
        for (int p = 0; p <= NSITE; ++p) {
            int lanesRan = 0, maxc = 0;
            for (int l = 0; l < 32; ++l) {
                int c = hc[l * (NSITE + 1) + p];
                if (c) ++lanesRan;
                if (c > maxc) maxc = c;
            }
            int ok = (model[p] == hi[p]);
            allok &= ok;
            char nameBuf[8];
            if (p == NSITE) snprintf(nameBuf, sizeof nameBuf, "loop");
            else            snprintf(nameBuf, sizeof nameBuf, "P%d", p);
            printf("  %-6s %-14d %-14lld %-12lld %s\n",
                   nameBuf, lanesRan, model[p], hi[p], ok ? "MATCH" : "MISMATCH");
        }
        int totOk = (modelTotal == hwTotal);
        allok &= totOk;
        printf("  %-6s %-14s %-14lld %-12lld %s\n", "TOTAL", "", modelTotal, hwTotal,
               totOk ? "MATCH" : "MISMATCH");
        score += allok; ++total;
    }
    if (modelTotal < 0) ++total;

    printf("\nSCORE: %d/%d\n", score, total);

    // ---- cleanup -----------------------------------------------------
    CHECK(cudaFree(d_mask)); CHECK(cudaFree(d_it)); CHECK(cudaFree(d_reached));
    CHECK(cudaFree(d_laneCount)); CHECK(cudaFree(d_issue)); CHECK(cudaFree(d_in));
    CHECK(cudaDeviceReset());

    printf("OVERALL: %s\n", score == total ? "PASS" : "FAIL");
    return score == total ? 0 : 1;
}
