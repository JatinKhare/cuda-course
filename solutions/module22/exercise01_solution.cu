// =============================================================================
// Module 22 / Exercise 1 — SOLUTION — instrument it, profile it, then fix it.
//
// GOAL : You are given `runNaive()`, a 300-step iterative solver written the
//        way this kind of loop usually gets written first. It produces the
//        right answer. It also leaves the GPU idle for most of its wall time.
//
//        Your job is to find out HOW idle, using Nsight Systems rather than
//        guessing, and then to rewrite the host side so it isn't.
//
//        You must actually run nsys. TODO 5 asks for a number this program
//        cannot measure about itself, and the harness checks it against the
//        bounds it CAN measure. A guess will be rejected.
//
// WHAT TO FILL IN
//   TODO 1  the NvtxRange RAII helper and the NVTX_RANGE macro
//   TODO 2  annotate runNaive() so the timeline localizes the cost
//   TODO 3  bracket the measured region with cudaProfilerStart/Stop
//   TODO 4  runFixed() -- same arithmetic, restructured host loop   (DESIGN)
//   TODO 5  the three numbers you read out of nsys
//
// SCORING: 6 points. OVERALL: PASS requires all six.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
//        (nvtx3 ships with CUDA 13.2 and is header-only: no -l flag.)
// RUN  : exercise01_solution.exe
//
// PROFILE (you need this for TODO 5):
//   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
//        -o ex01 --stats=true --force-overwrite=true exercise01.exe
//
//   On Windows the target's stdout is not forwarded through a redirected
//   pipe, so run the program once normally to see its own report, and once
//   under nsys to collect the profile.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#define CHECK(call) do {                                                       \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        printf("CUDA error %s (%s) at %s:%d\n", cudaGetErrorName(_e),          \
               cudaGetErrorString(_e), __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// 1 Mi cells = 4 MB per buffer. Deliberately chosen so that GPU work per step
// (~100 us of real memory traffic) is a few times the host's per-step cost
// (~35 us). Much smaller and the host is the bottleneck no matter what you do,
// so the fix cannot show anything; much larger and the host cost vanishes into
// the noise. The window where host behaviour is worth fixing is exactly where
// this exercise lives.
#define N        (1 << 20)
#define BLOCK      256
#define STEPS      300
#define OMEGA     0.25f

// =============================================================================
// TODO 1 — the NVTX scope guard.
//
// Build a type whose constructor opens a named NVTX range and whose destructor
// closes it, plus a macro NVTX_RANGE("name") that declares one.
//
// Two requirements that are easy to get wrong:
//   (a) Two NVTX_RANGE uses in the same scope must not collide. A macro that
//       hardcodes the object's name compiles until someone nests two, and then
//       fails in a way that looks unrelated to NVTX.
//   (b) The type must not be copyable. A copy would pop the range twice, and
//       an unbalanced push/pop stack silently reparents every range that comes
//       after it -- which is worse than no instrumentation, because the
//       timeline still looks plausible.
//
// The relevant C API is nvtxRangePushA(const char*) / nvtxRangePop(void).
// =============================================================================
struct NvtxRange {
    explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
    ~NvtxRange()                         { nvtxRangePop(); }
    NvtxRange(const NvtxRange &)            = delete;   // a copy would pop twice
    NvtxRange &operator=(const NvtxRange &) = delete;
};
// Two levels of indirection: `a##b` pastes its arguments BEFORE __LINE__ is
// expanded, so a single-level macro names every object `_nvtxScope__LINE__`
// and two ranges in one scope collide.
#define NVTX_CAT2(a, b) a##b
#define NVTX_CAT(a, b)  NVTX_CAT2(a, b)
#define NVTX_RANGE(name) NvtxRange NVTX_CAT(_nvtxScope, __LINE__)(name)


// -----------------------------------------------------------------------------
// The three kernels. Do not modify them -- the exercise is entirely host-side,
// and the harness checks that both versions produce identical results.
// -----------------------------------------------------------------------------
__global__ void jacobiStep(const float *__restrict__ in, float *__restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float l = in[i > 0 ? i - 1 : 0];
    float c = in[i];
    float r = in[i < n - 1 ? i + 1 : n - 1];
    out[i] = c + OMEGA * (l - 2.0f * c + r);
}

__global__ void scaleBy(float *__restrict__ a, int n, float s)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= s;
}

__global__ void residual(const float *__restrict__ a, const float *__restrict__ b,
                         float *__restrict__ out, int n)
{
    // Block reduction then one atomic per block (Module 12's ladder, Module 10's
    // privatization). A single atomicAdd per THREAD to one address would make
    // this kernel 60x slower than the two it is measuring, which would tell you
    // a great deal about atomic contention and nothing about timelines.
    __shared__ float s[BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (i < n) ? fabsf(a[i] - b[i]) : 0.0f;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, s[0]);
}

// -----------------------------------------------------------------------------
// Event-pair instrumentation. Note what this can and cannot tell you: the
// interval between two enqueued markers includes any stretch where the GPU was
// waiting for the host, so the "GPU time" it reports is an UPPER BOUND on the
// time the GPU spent executing. That is exactly the gap TODO 5 closes with
// nsys, which timestamps kernels on the device.
// -----------------------------------------------------------------------------
struct GpuTimeline {
    cudaEvent_t *beg, *end; int cap, n;
    void init(int c) {
        cap = c; n = 0;
        beg = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * cap);
        end = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * cap);
        for (int i = 0; i < cap; ++i) { CHECK(cudaEventCreate(&beg[i])); CHECK(cudaEventCreate(&end[i])); }
    }
    void open()  { if (n < cap) CHECK(cudaEventRecord(beg[n])); }
    void close() { if (n < cap) { CHECK(cudaEventRecord(end[n])); ++n; } }
    double totalMs() const {
        double s = 0.0;
        for (int i = 0; i < n; ++i) { float ms; cudaEventElapsedTime(&ms, beg[i], end[i]); s += ms; }
        return s;
    }
    void destroy() {
        for (int i = 0; i < cap; ++i) { cudaEventDestroy(beg[i]); cudaEventDestroy(end[i]); }
        free(beg); free(end);
    }
};

static int gBlocks = (N + BLOCK - 1) / BLOCK;

// =============================================================================
// runNaive — the version you are given. Read it before you profile it; then
// check whether the timeline agrees with what you expected.
//
// Every host-side decision in here is a realistic one that somebody made for a
// reason. None of them are typos.
// =============================================================================
static double runNaive(float *a, float *b, float *dRes, float *outResidual,
                       double *gpuMsOut)
{
    NVTX_RANGE("naive");

    GpuTimeline tl; tl.init(STEPS * 3);
    cudaEvent_t w0, w1;
    CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
    CHECK(cudaEventRecord(w0));

    float hostRes = 0.0f;
    for (int s = 0; s < STEPS; ++s) {
        NVTX_RANGE("step");
        // a scratch buffer, sized per step because the step "might" need it
        float *scratch = nullptr;
        { NVTX_RANGE("alloc"); CHECK(cudaMalloc(&scratch, sizeof(float) * N)); }

        {
            NVTX_RANGE("relax");
            tl.open(); jacobiStep<<<gBlocks, BLOCK>>>(a, scratch, N); tl.close();
            CHECK(cudaGetLastError());
            tl.open(); scaleBy<<<gBlocks, BLOCK>>>(scratch, N, 1.0f); tl.close();
            CHECK(cudaGetLastError());
        }

        {
            NVTX_RANGE("converge-check");       // convergence check, every step
            CHECK(cudaMemset(dRes, 0, sizeof(float)));
            tl.open(); residual<<<gBlocks, BLOCK>>>(scratch, a, dRes, N); tl.close();
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpy(&hostRes, dRes, sizeof(float), cudaMemcpyDeviceToHost));
        }

        { NVTX_RANGE("writeback"); CHECK(cudaMemcpy(a, scratch, sizeof(float) * N, cudaMemcpyDeviceToDevice)); }
        { NVTX_RANGE("free");      CHECK(cudaFree(scratch)); }
    }

    CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
    float ms; CHECK(cudaEventElapsedTime(&ms, w0, w1));
    *gpuMsOut = tl.totalMs();
    *outResidual = hostRes;
    tl.destroy();
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    (void)b;
    return ms;
}

// =============================================================================
// TODO 4 (DESIGN) — runFixed.
//
// Produce the same final state in `a` and the same final residual value, with
// the same 900 kernel launches, but without the host-side idle.
//
// You are NOT allowed to change the kernels, reduce the number of launches, or
// change the arithmetic. This is purely a question about when the host talks
// to the device.
//
// Decide for yourself what to do about each of these. Each one is a separate
// decision with a separate justification:
//   - the per-step cudaMalloc / cudaFree of `scratch`
//   - the per-step blocking cudaMemcpy of a single float
//   - the per-step cudaMemset of the residual accumulator
//   - the per-step device-to-device copy of the whole array
//
// Exactly two of those four are already asynchronous with respect to the host:
// they cost you a few microseconds of host time but they do NOT drain the
// queue or stall the CPU. "It synchronizes" is the wrong reason to touch them,
// even if the code ends up faster. Work out which two before you start, and
// check your answer against the cuda_api_sum table rather than against
// intuition -- the API reference is unambiguous here and most people guess it
// wrong at least once.
//
// The harness requires:
//   * the final residual to match runNaive's to within 1e-4 relative,
//   * the final array to match runNaive's elementwise,
//   * a wall-time speedup of at least 3.0x. The ceiling is the naive version
//     you can compute it before you start: the fix can only recover time the
//     host left on the table, so the best possible ratio is the naive version's
//     wall time divided by its GPU time. Work that number out from your own
//     profile and you will know whether you are done.
//   * exactly 900 launches (it counts them).
//
// Signature is fixed; fill in the body. `tl` is already wired up for you so the
// harness can compare the two versions on the same footing.
// =============================================================================
static double runFixed(float *a, float *b, float *dRes, float *outResidual,
                       double *gpuMsOut)
{
    GpuTimeline tl; tl.init(STEPS * 3);
    cudaEvent_t w0, w1;
    CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));

    float hostRes = 0.0f;

    NVTX_RANGE("fixed");

    // FIX 1: one allocation for the whole run, outside the timed region.
    // cudaMalloc is a driver call that synchronizes the device; doing it 300
    // times costs 300 stalls as well as 300 allocations.
    float *scratch = nullptr;
    { NVTX_RANGE("alloc-once"); CHECK(cudaMalloc(&scratch, sizeof(float) * N)); }

    CHECK(cudaEventRecord(w0));
    for (int s = 0; s < STEPS; ++s) {
        NVTX_RANGE("step");
        {
            NVTX_RANGE("relax");
            tl.open(); jacobiStep<<<gBlocks, BLOCK>>>(a, scratch, N); tl.close();
            CHECK(cudaGetLastError());
            tl.open(); scaleBy<<<gBlocks, BLOCK>>>(scratch, N, 1.0f); tl.close();
            CHECK(cudaGetLastError());
        }
        {
            NVTX_RANGE("converge-check");
            // FIX 2: cudaMemsetAsync rather than cudaMemset. This one is a
            // micro-optimisation, NOT a synchronization fix -- see the md.
            CHECK(cudaMemsetAsync(dRes, 0, sizeof(float)));
            tl.open(); residual<<<gBlocks, BLOCK>>>(scratch, a, dRes, N); tl.close();
            CHECK(cudaGetLastError());
            // FIX 3: the readback is GONE from the loop. dRes still ends the
            // loop holding the last step's residual, which is the only value
            // the program actually used.
        }
        { NVTX_RANGE("writeback"); CHECK(cudaMemcpy(a, scratch, sizeof(float) * N, cudaMemcpyDeviceToDevice)); }
    }
    CHECK(cudaEventRecord(w1));
    CHECK(cudaEventSynchronize(w1));

    // The one readback the algorithm genuinely needs, after the loop.
    CHECK(cudaMemcpy(&hostRes, dRes, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(scratch));
    (void)b;

    // ---- leave the code below this line alone --------------------------------
    if (tl.n == 0) {           // TODO 4 not attempted
        tl.destroy();
        CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
        *gpuMsOut = 0.0; *outResidual = 0.0;
        return -1.0;
    }
    float ms; CHECK(cudaEventElapsedTime(&ms, w0, w1));
    *gpuMsOut = tl.totalMs();
    *outResidual = hostRes;
    tl.destroy();
    CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
    return ms;
}

// =============================================================================
// TODO 5 — what nsys told you.
//
// Profile the program with the command in the header. The capture range you
// opened in TODO 3 restricts collection to the two measured versions, so the
// warm-up's 2000 launches and the context setup stay out of the tables.
//
// All three numbers below cover the WHOLE capture range -- both versions
// together, 1800 launches. Read them straight off `nsys stats`:
//
//   NS_KERNEL_NS   cuda_gpu_kern_sum: add the "Total Time (ns)" column over
//                  all three kernel rows. This is time the GPU spent
//                  EXECUTING, timestamped on the device.
//
//   NS_LAUNCH_NS   cuda_api_sum: the "Total Time (ns)" of the
//                  cudaLaunchKernel row. This is host time spent asking for
//                  work. It is not GPU time and no kernel optimisation
//                  touches it.
//
//   NS_SYNCCOST_NS cuda_api_sum: the "Total Time (ns)" of the cudaMemcpy row.
//                  604 of those calls move 4 MB and 301 of them move four
//                  bytes; if this row is large, it is not the bytes.
//
// Leave all three at 0 to skip. The harness will say so and withhold the
// points rather than passing you quietly.
//
// Captured with:
//   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi //        -o ex01sol --force-overwrite=true --stats=true exercise01_solution.exe
// cuda_gpu_kern_sum : residual 21244566 + jacobiStep 7634960 + scaleBy 6801363
// cuda_api_sum      : cudaLaunchKernel 47998696 ns over 1800 calls (26.7 us avg)
// cuda_api_sum      : cudaMemcpy       110502657 ns over 905 calls
// These vary ~20-30% run to run; any capture of your own will pass the gates.
#define NS_KERNEL_NS    35680889.0   // cuda_gpu_kern_sum, all three rows
#define NS_LAUNCH_NS    47998696.0   // cudaLaunchKernel, 1800 calls
#define NS_SYNCCOST_NS 110502657.0   // cudaMemcpy, 905 calls

// -----------------------------------------------------------------------------
static void initField(float *h)
{
    srand(20250901);
    for (int i = 0; i < N; ++i) h[i] = (float)((i * 37) % 101) / 101.0f
                                      + 0.001f * (float)rand() / (float)RAND_MAX;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Module 22 / Exercise 1 — instrument, profile, fix\n");
    printf("N = %d, %d steps, %d launches per version\n\n", N, STEPS, STEPS * 3);

    float *h    = (float *)malloc(sizeof(float) * N);
    float *hA   = (float *)malloc(sizeof(float) * N);
    float *hB   = (float *)malloc(sizeof(float) * N);
    float *a = nullptr, *b = nullptr, *dRes = nullptr;
    CHECK(cudaMalloc(&a, sizeof(float) * N));
    CHECK(cudaMalloc(&b, sizeof(float) * N));
    CHECK(cudaMalloc(&dRes, sizeof(float)));

    // Warm-up: context creation, module load, clocks. Keep it OUT of the
    // capture range -- the first cudaMalloc alone can cost 100 ms.
    initField(h);
    CHECK(cudaMemcpy(a, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    for (int i = 0; i < 2000; ++i) jacobiStep<<<gBlocks, BLOCK>>>(a, b, N);
    CHECK(cudaDeviceSynchronize());

    // TODO 3: open a profiler capture range here, and close it after the two
    //         versions have run (marked below).
    //
    //         Without this, `--capture-range=cudaProfilerApi` collects nothing
    //         and `--stats=true` reports the warm-up's 2000 launches mixed in
    //         with the 1800 you care about. The two calls you need are declared
    //         in <cuda_profiler_api.h>.
    CHECK(cudaProfilerStart());

    double wallN = 0.0, gpuN = 0.0, wallF = 0.0, gpuF = 0.0;
    float resN = 0.0f, resF = 0.0f;

    // Both versions back-to-back, nothing printed in between (spec S12.1).
    initField(h);
    CHECK(cudaMemcpy(a, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    wallN = runNaive(a, b, dRes, &resN, &gpuN);
    CHECK(cudaMemcpy(hA, a, sizeof(float) * N, cudaMemcpyDeviceToHost));

    initField(h);
    CHECK(cudaMemcpy(a, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    wallF = runFixed(a, b, dRes, &resF, &gpuF);
    if (wallF > 0.0) CHECK(cudaMemcpy(hB, a, sizeof(float) * N, cudaMemcpyDeviceToHost));

    // TODO 3b: close the profiler capture range here.
    CHECK(cudaProfilerStop());

    if (wallF < 0.0) {
        printf("Set TODO 4 first.\n");
        CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(dRes));
        free(h); free(hA); free(hB); CHECK(cudaDeviceReset());
        return 0;
    }

    printf("%-16s %12s %14s %12s\n", "version", "wall (ms)", "GPU<=(ms)", "busy<= %");
    printf("%-16s %12.3f %14.3f %11.1f%%\n", "naive", wallN, gpuN, 100.0 * gpuN / wallN);
    printf("%-16s %12.3f %14.3f %11.1f%%\n", "fixed", wallF, gpuF, 100.0 * gpuF / wallF);
    printf("speedup        : %.2fx\n\n", wallN / wallF);

    // ---- scoring -----------------------------------------------------------
    int score = 0;

    bool resOk = fabsf(resN - resF) <= 1e-4f * fmaxf(1.0f, fabsf(resN));
    printf("[%s] 1. final residual matches  (naive %.6f vs fixed %.6f)\n",
           resOk ? "x" : " ", (double)resN, (double)resF);
    score += resOk;

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (fabsf(hA[i] - hB[i]) > 1e-5f * fmaxf(1.0f, fabsf(hA[i]))) ++bad;
    bool fieldOk = (bad == 0);
    printf("[%s] 2. final field matches     (%d/%d cells differ)\n",
           fieldOk ? "x" : " ", bad, N);
    score += fieldOk;

    bool fast = (wallN / wallF) >= 3.0;
    printf("[%s] 3. speedup >= 3.0x         (got %.2fx)\n", fast ? "x" : " ", wallN / wallF);
    score += fast;

    // TODO 5 cross-checks. The event harness gives an UPPER bound on GPU time
    // and the wall time gives an upper bound on everything, so a fabricated
    // number has to survive two inequalities and a ratio.
    double kerMs = NS_KERNEL_NS / 1e6, lauMs = NS_LAUNCH_NS / 1e6, synMs = NS_SYNCCOST_NS / 1e6;
    double gpuBound = gpuN + gpuF;          // event bound, over BOTH versions
    double wallBoth = wallN + wallF;
    int    launches = 2 * STEPS * 3;        // 1800
    bool given = (NS_KERNEL_NS > 0.0) && (NS_LAUNCH_NS > 0.0) && (NS_SYNCCOST_NS > 0.0);
    // Real device execution time must be positive, must be below the event
    // bound (which absorbs launch gaps on top of it), and cannot be a rounding
    // error either. Three inequalities a guessed number has to satisfy at once.
    // 0.15, not 0.30: the event bound is an upper bound whose TIGHTNESS
    // varies run to run (a cold first run inflated it 2x), so a tighter
    // window would reject a correct profile. A fabricated number still has
    // to land inside a 6.7x window AND below the bound.
    bool kerOk = given && kerMs > 0.15 * gpuBound && kerMs < gpuBound;
    double usPerLaunch = 1000.0 * lauMs / (double)launches;
    bool lauOk = given && usPerLaunch > 2.0 && usPerLaunch < 100.0 && lauMs < wallBoth;
    bool synOk = given && synMs > 0.0 && synMs < wallBoth;
    if (!given) printf("[ ] 4-6. TODO 5 not filled in -- profile the program.\n");
    else {
        printf("[%s] 4. nsys kernel time plausible (%.3f ms; event bound %.3f ms)\n",
               kerOk ? "x" : " ", kerMs, gpuBound);
        printf("[%s] 5. nsys launch cost plausible (%.3f ms = %.1f us x %d launches)\n",
               lauOk ? "x" : " ", lauMs, usPerLaunch, launches);
        printf("[%s] 6. nsys memcpy cost plausible (%.3f ms of %.3f ms wall)\n",
               synOk ? "x" : " ", synMs, wallBoth);
        score += kerOk + lauOk + synOk;
    }

    if (given && kerOk) {
        printf("\nOver the whole capture (both versions, %.1f ms wall):\n", wallBoth);
        printf("  GPU actually executing       = %.1f%%   (event bound claimed %.1f%%)\n",
               100.0 * kerMs / wallBoth, 100.0 * gpuBound / wallBoth);
        printf("  host inside cudaLaunchKernel = %.1f%%\n", 100.0 * lauMs / wallBoth);
        printf("  host inside cudaMemcpy       = %.1f%%\n", 100.0 * synMs / wallBoth);
    }

    CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(dRes));
    free(h); free(hA); free(hB);
    CHECK(cudaDeviceReset());

    printf("\nSCORE: %d/6\n", score);
    printf("OVERALL: %s\n", score == 6 ? "PASS" : "FAIL");
    return score == 6 ? 0 : 1;
}
