// =============================================================================
// Module 23 / Exercise 3 - from a report to a ranked optimization plan.
//
// GOAL : You are given e3_base, a kernel written the way this kind of kernel
//        gets written first, and a CONSTRUCTED Nsight Compute report for it.
//        (`ncu` cannot run on this machine: ERR_NVGPUCTRPERM. The report is
//        assembled from this course's own measurements and is physically
//        consistent with this GPU.)
//
//        The profile shows FOUR separate defects. All four are real. They are
//        NOT equally expensive, and the report contains the evidence for which
//        one is binding -- if you read the right denominator.
//
//        Produce a plan: rank the four fixes by expected payoff, implement the
//        one you ranked FIRST and the one you ranked LAST, and let the harness
//        measure whether your ranking survived contact with the machine.
//
// WHAT TO FILL IN
//   TODO 1  RANK[4]      -- the four fixes, best first                (ANALYSIS)
//   TODO 2  BUCKET_TOP   -- the speedup bucket of your top fix
//   TODO 3  e3_top()     -- implement the fix you ranked first          (DESIGN)
//   TODO 4  e3_bottom()  -- implement the fix you ranked last           (DESIGN)
//   TODO 5  EVIDENCE, MISLEAD -- which profile row predicted the outcome, and
//                                which one is the loudest number that does not
//
// Only positions 1 and 4 of your ranking are scored. The middle two are not,
// and the program explains why in the numbers it prints -- that is itself one
// of the things this exercise exists to teach.
//
// SCORING: 9 points, plus both of your kernels must validate.
//
// BUILD: nvcc -arch=sm_89 -O3 -lineinfo -o exercise03.exe exercise03.cu
// RUN  : exercise03.exe
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                       \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        printf("CUDA error %s (%s) at %s:%d\n", cudaGetErrorName(_e),          \
               cudaGetErrorString(_e), __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define DRAM_PIN_PEAK_GBS 432.0

#define E3_REC     4                              // AoS record: 4 floats
#define E3_N       ((size_t)16*1024*1024)         // records -> 256 MB AoS
#define E3_BLOCKS  20                             // half the SMs get nothing
#define E3_THR     128
#define E3_MLP     4
#define LUTD       32                             // LUT is LUTD x LUTD

// =============================================================================
// ====================  YOUR ANSWERS  =========================================
// =============================================================================
// TODO 1: the four candidate fixes, ranked by EXPECTED payoff, best first.
//   1 CONCURRENCY : machine-sized grid AND several loads in flight per thread
//   2 LAYOUT      : read a packed SoA copy instead of every 4th float of an AoS
//   3 CONFLICTS   : remove the 32-way shared-memory bank conflict
//   4 DIVIDE      : replace the IEEE division with a precomputed reciprocal
// Leave as zeros to have the program print the report and the menus and stop.
static const int RANK[4] = { 0, 0, 0, 0 };     // YOUR CODE HERE

// TODO 2: bucket for the TOP-ranked fix.  1: <1.5x   2: 1.5-6x   3: >6x
static const int BUCKET_TOP = 0;               // YOUR CODE HERE

// TODO 5: which profile row predicted the outcome, and which one misled.
//         Codes are the PROFILE ROWS YOU MAY CITE menu the program prints.
static const int EVIDENCE = 0;                 // YOUR CODE HERE
static const int MISLEAD  = 0;                 // YOUR CODE HERE

// =============================================================================
// The kernel under study, and the two variants.
//
// e3_base has all four defects at once:
//   A  reads img[E3_REC*i] -- one field of a 4-float AoS record: 16 sectors per
//      request where 4 would do                                        (M5)
//   B  reads the staged LUT down a column, s[lane*LUTD + k], which puts all 32
//      lanes on one bank asking for 32 distinct words: degree 32     (M7/M18)
//   C  divides by sd per element                                       (M20)
//   D  is launched as 40 blocks of 128 threads with one load outstanding per
//      thread: 5120 threads, far below the concurrency this memory system
//      needs                                                       (M20/M21)
// =============================================================================
__global__ void e3_base(const float * __restrict__ img,
                        const float * __restrict__ lut,
                        float * __restrict__ out,
                        size_t n, float mean, float sd)
{
    __shared__ float s[LUTD*LUTD];
    for (int t = (int)threadIdx.x; t < LUTD*LUTD; t += (int)blockDim.x)
        s[t] = lut[t];
    __syncthreads();

    const int lane = (int)(threadIdx.x & 31u);
    size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t str = gridDim.x * (size_t)blockDim.x;

    // #pragma unroll 1 keeps the defect honest: without it ptxas unrolls the
    // grid-stride loop and hoists several loads, manufacturing exactly the
    // memory-level parallelism this kernel is supposed to be missing. Spec 12.11.
    #pragma unroll 1
    for (size_t i = gid; i < n; i += str) {
        int   k = (int)((i >> 7) & (LUTD - 1));     // warp-uniform, loop-varying
        float w = s[lane*LUTD + k];                 // defect B: degree 32
        float v = img[E3_REC*i];                    // defect A: 16 sectors/req
        out[i]  = ((v - mean) / sd) * w;            // defect C: IEEE divide
    }
}

// ---- TODO 3 / TODO 4 --------------------------------------------------------
// Two slots. Put the fix you ranked FIRST in e3_top and the fix you ranked LAST
// in e3_bottom. Each must be e3_base plus EXACTLY ONE of the four changes --
// if you fold two fixes into one kernel the measurement stops meaning anything.
//
// Both are launched by the harness through lTop()/lBottom() below; if your fix
// changes the launch shape or the input buffer, change the launcher, not the
// signature. `imgSoA` is a packed copy of field 0, prepared once off the clock,
// and `rsd` is 1.0f/sd, also computed on the host: use them if your fix needs
// them and ignore them if it does not.
//
// Both must produce the same values as e3_base to within fp32 tolerance. The
// validation pass checks a sample of the output against a double reference.
// -----------------------------------------------------------------------------
__global__ void e3_top(const float * __restrict__ img,
                       const float * __restrict__ imgSoA,
                       const float * __restrict__ lut,
                       float * __restrict__ out,
                       size_t n, float mean, float sd, float rsd)
{
    // TODO 3: YOUR CODE HERE
    (void)img; (void)imgSoA; (void)lut; (void)out; (void)n;
    (void)mean; (void)sd; (void)rsd;
}

__global__ void e3_bottom(const float * __restrict__ img,
                          const float * __restrict__ imgSoA,
                          const float * __restrict__ lut,
                          float * __restrict__ out,
                          size_t n, float mean, float sd, float rsd)
{
    // TODO 4: YOUR CODE HERE
    (void)img; (void)imgSoA; (void)lut; (void)out; (void)n;
    (void)mean; (void)sd; (void)rsd;
}

// ---- deterministic device-side initialisation -------------------------------
__global__ void e3_initAoS(float *p, size_t nrec)
{
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t st = gridDim.x*(size_t)blockDim.x;
    for (; i < nrec; i += st) {
        unsigned h = (unsigned)i * 1103515245u + 12345u;
        for (int c = 0; c < E3_REC; ++c)
            p[E3_REC*i + c] = (float)(((h >> (8*c)) & 255u)) * (1.0f/256.0f);
    }
}
// A streaming read used only to ramp the memory P-state (spec SS12 rule 4).
// It is deliberately independent of every kernel under test.
__global__ void e3_warm(const float4 * __restrict__ in, float *sink, size_t n4)
{
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t st = gridDim.x*(size_t)blockDim.x;
    float4 a = make_float4(0.f,0.f,0.f,0.f);
    for (; i < n4; i += st) { float4 v = in[i]; a.x+=v.x; a.y+=v.y; a.z+=v.z; a.w+=v.w; }
    if (a.x == 1e30f) sink[0] = a.x + a.y + a.z + a.w;
}

__global__ void e3_pack(const float *aos, float *soa, size_t nrec)
{
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t st = gridDim.x*(size_t)blockDim.x;
    for (; i < nrec; i += st) soa[i] = aos[E3_REC*i];
}

// =============================================================================
// The constructed report.
// =============================================================================
static void printReport(void)
{
printf(
"=============================================================================\n"
" CONSTRUCTED ncu REPORT  --  NOT produced by ncu on this machine.\n"
" ncu 2026.1.0 is installed here and every invocation returns\n"
"   ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access\n"
"             NVIDIA GPU Performance Counters on the target device 0.\n"
" Assembled from this course's own measurements (M5, M7, M18, M20, M21), so\n"
" it is physically consistent with this GPU.\n"
"=============================================================================\n"
" e3_base(const float*, const float*, float*, unsigned long, float, float)\n"
"   <<<20, 128>>>   16,777,216 records of 4 floats (256 MB), one field read\n"
"-----------------------------------------------------------------------------\n"
" Section: GPU Speed Of Light Throughput\n"
"   Compute (SM) Throughput                      %%            4.90\n"
"   Memory Throughput                            %%           30.25\n"
"   DRAM Throughput                              %%           30.25\n"
"   Duration                                 msecond          2.5672\n"
"   Elapsed Cycles                             cycle       4,878,000\n"
"   SM Frequency                              Ghz              1.90\n"
" Section: Launch Statistics\n"
"   Block Size                                                 128\n"
"   Grid Size                                                    20\n"
"   launch__waves_per_multiprocessor                           0.04\n"
" Section: Occupancy\n"
"   Theoretical Occupancy                        %%          100.00\n"
"   launch__occupancy_limit_registers                            16\n"
"   launch__occupancy_limit_shared_mem                           20\n"
"   launch__occupancy_limit_warps                                12\n"
"   launch__occupancy_limit_blocks                               24\n"
"   Achieved Occupancy                           %%            8.33\n"
"   sm__warps_active.avg.pct_of_peak_sustained_elapsed  %%     4.17\n"
" Section: Memory Workload Analysis\n"
"   dram__bytes.sum                            Mbyte         335.54\n"
"   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum        8,388,608\n"
"   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum         524,288\n"
"     ->  sectors per request                                 16.00\n"
"   smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio\n"
"                                                                8.00\n"
"   l1tex__t_sector_hit_rate                     %%            0.00\n"
"   lts__t_sector_hit_rate                       %%            0.30\n"
"   smsp__sass_inst_executed_op_shared_ld.sum               524,288\n"
"   l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum 16,777,216\n"
"   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum\n"
"                                                        16,252,928\n"
"     ->  wavefronts per shared load request                  32.00\n"
" Section: Compute Workload Analysis\n"
"   sm__inst_executed_pipe_fma.avg.pct_of_peak_...active  %%    0.55\n"
"   sm__inst_executed_pipe_lsu.avg.pct_of_peak_...active  %%    1.45\n"
"   sm__inst_executed_pipe_xu.avg.pct_of_peak_...active   %%    4.70\n"
" Section: Scheduler Statistics\n"
"   Theoretical Active Warps Per Scheduler                    12.00\n"
"   Active Warps Per Scheduler                                 1.00\n"
"   Eligible Warps Per Scheduler                               0.11\n"
"   Issued Warp Per Scheduler                                  0.10\n"
"   No Eligible                                  %%           89.40\n"
"   One or More Eligible                         %%           10.60\n"
" Section: Warp State Statistics\n"
"   Warp Cycles Per Issued Instruction      cycle/inst        10.00\n"
"   Stall Long Scoreboard                   cycle/inst         7.90\n"
"   Stall Short Scoreboard                  cycle/inst         1.20\n"
"   Stall Wait                              cycle/inst         0.45\n"
"   Stall MIO Throttle                      cycle/inst         0.25\n"
"   Stall Not Selected                      cycle/inst         0.02\n"
"   Selected                                cycle/inst         0.10\n"
"   Stall Misc                              cycle/inst         0.08\n"
"   smsp__thread_inst_executed_per_inst_executed.ratio        32.00\n"
"=============================================================================\n");
}

static void printMenus(void)
{
printf(
"\n CANDIDATE FIXES\n"
"   1 CONCURRENCY : a machine-sized grid AND several independent loads in\n"
"                   flight per thread, before any of them is consumed\n"
"   2 LAYOUT      : read a packed SoA copy instead of every 4th float of the\n"
"                   AoS record (the packing is done once, off the clock)\n"
"   3 CONFLICTS   : stage the LUT transposed so the shared read is\n"
"                   conflict-free instead of 32-way\n"
"   4 DIVIDE      : replace the IEEE division by a host-computed reciprocal\n"
"\n PROFILE ROWS YOU MAY CITE\n"
"   1  Eligible Warps Per Scheduler = 0.11  (No Eligible 89.4%%)\n"
"   2  Achieved Occupancy = 8.33%% against a Theoretical 100%%\n"
"   3  l1tex__data_bank_conflicts_...shared_op_ld.sum = 16,252,928\n"
"   4  sectors per global load request = 16.00, bytes/sector = 8.00\n"
"   5  sm__inst_executed_pipe_xu... = 4.70%%  (the divides)\n"
"   6  DRAM Throughput = 30.25%% of peak\n");
}

// =============================================================================
// Harness
// =============================================================================
static unsigned fnv(const char *s)
{ unsigned h = 2166136261u; for (; *s; ++s) { h ^= (unsigned char)*s; h *= 16777619u; } return h; }
static unsigned hashAns(char tag, int i, int v)
{ char b[32]; snprintf(b, sizeof b, "%c%d:%d", tag, i, v); return fnv(b); }

static const unsigned REF_TOP    =  960701160u;   // hashAns('T',0,1)
static const unsigned REF_BUCKET = 4259325639u;   // hashAns('B',0,2)
static const unsigned REF_EVID   = 1747655933u;   // hashAns('E',0,1)
static const unsigned REF_MIS    =  474087829u;   // hashAns('S',0,3)

static float *d_img, *d_soa, *d_lut, *d_out;
static float g_mean = 0.4971f, g_sd = 0.2887f;

static void lWarm (void){ e3_warm<<<640,256>>>((const float4*)d_img, d_out, E3_N*E3_REC/4); }
static void lBase (void){ e3_base   <<<E3_BLOCKS,E3_THR>>>(d_img,d_lut,d_out,E3_N,g_mean,g_sd); }
// Change the <<<grid,block>>> of these two if your fix needs a different launch
// shape. That is part of TODO 3.
static void lTop  (void){ e3_top    <<<E3_BLOCKS,E3_THR>>>(d_img,d_soa,d_lut,d_out,E3_N,g_mean,g_sd,1.0f/g_sd); }
static void lBot  (void){ e3_bottom <<<E3_BLOCKS,E3_THR>>>(d_img,d_soa,d_lut,d_out,E3_N,g_mean,g_sd,1.0f/g_sd); }

static double timeIt(void (*f)(void), int iters)
{
    cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) f();
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms,a,b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    return ms/iters;
}
static void warmFor(float ms, void (*f)(void))
{
    cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
    float el = 0.f; CHECK(cudaEventRecord(w0));
    while (el < ms) { f();
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
}

// Every scored quantity is a ratio formed inside ONE rotated sweep, which spec
// SS12 rule 5b exempts from the operating-point guard.
#define NCFG 3
static void sweep(void (*f[NCFG])(void), double best[NCFG])
{
    int it[NCFG];
    for (int i = 0; i < NCFG; ++i) {
        double t = timeIt(f[i], 1);
        int nn = (int)(10.0/(t > 1e-3 ? t : 1e-3)); if (nn < 3) nn = 3; if (nn > 60) nn = 60;
        it[i] = nn; best[i] = 1e30;
    }
    for (int s = 0; s < NCFG; ++s)
        for (int q = 0; q < NCFG; ++q) {
            int p = (q + s) % NCFG;
            double t = timeIt(f[p], it[p]);
            if (t < best[p]) best[p] = t;
        }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 23 / Exercise 3 - report -> ranked plan -> measure ===\n\n");
    printReport();
    printMenus();

    if (RANK[0] == 0 || BUCKET_TOP == 0 || EVIDENCE == 0 || MISLEAD == 0) {
        printf("\nSet TODO 1, 2 and 5 first (and write TODOs 3 and 4).\n");
        return 0;
    }

    CHECK(cudaMalloc(&d_img, E3_N*E3_REC*sizeof(float)));
    CHECK(cudaMalloc(&d_soa, E3_N*sizeof(float)));
    CHECK(cudaMalloc(&d_out, E3_N*sizeof(float)));
    CHECK(cudaMalloc(&d_lut, LUTD*LUTD*sizeof(float)));
    {
        float hl[LUTD*LUTD];
        for (int t = 0; t < LUTD*LUTD; ++t) hl[t] = 0.5f + (float)((t*37) % 97) * 0.01f;
        CHECK(cudaMemcpy(d_lut, hl, sizeof hl, cudaMemcpyHostToDevice));
    }
    e3_initAoS<<<2048,256>>>(d_img, E3_N);
    e3_pack   <<<2048,256>>>(d_img, d_soa, E3_N);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

    printf("\n-- warming 1500 ms streaming (spec SS12 rule 4) ----------------------\n");
    warmFor(1500.f, lWarm);
    CHECK(cudaGetLastError());

    void (*f[NCFG])(void) = { lBase, lTop, lBot };
    const char *nm[NCFG] = { "e3_base               ",
                             "e3_top     (your #1)  ",
                             "e3_bottom  (your #4)  " };
    double best[NCFG];
    sweep(f, best);
    CHECK(cudaGetLastError());

    // DRAM bytes: 4-float records read (every sector touched) + float output.
    const double movedBase = (double)E3_N*E3_REC*sizeof(float) + (double)E3_N*sizeof(float);
    const double movedSoA  = (double)E3_N*sizeof(float) + (double)E3_N*sizeof(float);

    printf("\n-- measured, all three timed back to back and rotated -----------------\n");
    printf("   %-24s %10s %10s %11s\n", "variant", "ms", "speedup", "DRAM GB/s");
    for (int i = 0; i < NCFG; ++i) {
        // If your fix changed the layout, the bytes it moves changed too.
        double moved = (i > 0 && RANK[i-1] == 2) ? movedSoA : movedBase;
        printf("   %-24s %10.4f %9.2fx %8.1f GB/s\n",
               nm[i], best[i], best[0]/best[i], moved/(best[i]*1e-3)/1e9);
    }

    const double sTop = best[0]/best[1], sBot = best[0]/best[2];
    const int    bTop = (sTop < 1.5) ? 1 : (sTop < 6.0 ? 2 : 3);

    printf("\n   Your ranking: %d > %d > %d > %d\n", RANK[0],RANK[1],RANK[2],RANK[3]);
    printf("   Your top (%d) measured %.2fx; your bottom (%d) measured %.2fx.\n",
           RANK[0], sTop, RANK[3], sBot);
    printf("\n   Positions 2 and 3 of your ranking are NOT scored, and that is\n"
           "   deliberate: on this machine the three non-binding fixes land\n"
           "   within each other's run-to-run spread, so their relative order is\n"
           "   not a real distinction and the harness refuses to pretend it is\n"
           "   (spec 12.5d). The solution notes give the measured numbers.\n");

    // ---- validation, separate untimed pass ---------------------------------
    printf("\n-- validation (separate pass) ----------------------------------------\n");
    int vok = 1;
    {
        const size_t NS = 8192;
        float *hout = (float*)malloc(NS*sizeof(float));
        float *haos = (float*)malloc(NS*E3_REC*sizeof(float));
        float  hl[LUTD*LUTD];
        CHECK(cudaMemcpy(hl, d_lut, sizeof hl, cudaMemcpyDeviceToHost));
        for (int v = 0; v < NCFG; ++v) {
            CHECK(cudaMemset(d_out, 0xff, E3_N*sizeof(float)));
            f[v](); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(hout, d_out, NS*sizeof(float), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(haos, d_img, NS*E3_REC*sizeof(float), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (size_t i = 0; i < NS; ++i) {
                int   lane = (int)(i & 31u);
                int   k    = (int)((i >> 7) & (LUTD - 1));
                double w   = hl[lane*LUTD + k];
                double ref = ((double)haos[E3_REC*i] - (double)g_mean) / (double)g_sd * w;
                if (!(fabs((double)hout[i] - ref) <= 1e-5*fmax(1.0, fabs(ref)))) ++bad;
            }
            printf("   %s sampled mismatches: %d\n", nm[v], bad);
            if (bad) vok = 0;
        }
        free(hout); free(haos);
    }

    // ---- scoring -----------------------------------------------------------
    printf("\n-- scoring -----------------------------------------------------------\n");
    int score = 0;
    int seen[5] = {0,0,0,0,0}, perm = 1;
    for (int i = 0; i < 4; ++i) {
        if (RANK[i] < 1 || RANK[i] > 4 || seen[RANK[i]]) perm = 0;
        else seen[RANK[i]] = 1;
    }
    printf("   [%s] RANK is a permutation of 1..4\n", perm?"x":" "); score += perm;

    int okT = (hashAns('T',0,RANK[0]) == REF_TOP);
    printf("   [%s] top-ranked fix\n", okT?"x":" "); score += 2*okT;

    int okB = (hashAns('B',0,BUCKET_TOP) == REF_BUCKET);
    printf("   [%s] predicted bucket for the top fix\n", okB?"x":" "); score += okB;

    int okM = (BUCKET_TOP == bTop);
    printf("   [%s] measured bucket matches your prediction (you %d, measured %d)\n",
           okM?"x":" ", BUCKET_TOP, bTop); score += okM;

    int okW = (sTop >= 1.5);
    printf("   [%s] your top-ranked fix measured >= 1.5x  (%.2fx)\n", okW?"x":" ", sTop);
    score += okW;

    int okL = (sBot < 1.5);
    printf("   [%s] your bottom-ranked fix measured <  1.5x  (%.2fx)\n", okL?"x":" ", sBot);
    score += okL;

    int okE = (hashAns('E',0,EVIDENCE) == REF_EVID);
    printf("   [%s] the row that predicted the outcome\n", okE?"x":" "); score += okE;
    int okS = (hashAns('S',0,MISLEAD) == REF_MIS);
    printf("   [%s] the row that misleads\n", okS?"x":" "); score += okS;

    if (!vok) printf("   validation FAILED - scoring is void\n");

    CHECK(cudaFree(d_img)); CHECK(cudaFree(d_soa));
    CHECK(cudaFree(d_out)); CHECK(cudaFree(d_lut));

    printf("\nSCORE: %d/9\n", score);
    printf("OVERALL: %s\n", (score == 9 && vok) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (score == 9 && vok) ? 0 : 1;
}
