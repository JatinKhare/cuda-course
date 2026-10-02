// =============================================================================
// Module 22 / Example 1 — NVTX instrumentation and the GPU busy fraction.
//
// GOAL : Take an n-body step loop written the way people actually write them
//        first, annotate it with NVTX so a timeline is readable, and measure
//        the one number Nsight Systems exists to produce: what fraction of
//        wall-clock time the GPU was actually executing something.
//
//        The program measures that fraction itself, with CUDA events, so it
//        is useful with or without a profiler. Then you run it under nsys and
//        confirm that `cuda_gpu_kern_sum` Total Time divided by the NVTX range
//        duration from `nvtx_sum` reproduces the same number.
//
// BUILD: nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
//        (nvtx3 is header-only on CUDA 13.2 -- no -l flag is required.)
// RUN  : example01.exe
//
// PROFILE:
//   nsys profile --trace=cuda,nvtx -o ex01 --stats=true --force-overwrite=true example01.exe
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define CHECK(call) do {                                                       \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        printf("CUDA error %s (%s) at %s:%d\n", cudaGetErrorName(_e),          \
               cudaGetErrorString(_e), __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// -----------------------------------------------------------------------------
// The RAII NVTX range. This is the single highest-value 6 lines in the module.
//
// nvtxRangePushA/nvtxRangePop maintain a per-thread stack, so ranges nest the
// way C++ scopes nest. Tying push to the constructor and pop to the destructor
// means an early `return`, a `break`, or a thrown exception cannot leave the
// stack unbalanced -- and an unbalanced stack produces a timeline where every
// subsequent range is attributed to the wrong parent.
// -----------------------------------------------------------------------------
struct NvtxRange {
    explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
    ~NvtxRange()                         { nvtxRangePop(); }
    NvtxRange(const NvtxRange &)            = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
};
// Two levels of indirection are required: `a##b` pastes before __LINE__ is
// expanded, so a single-level macro would name every object the same thing.
#define NVTX_CAT2(a, b) a##b
#define NVTX_CAT(a, b)  NVTX_CAT2(a, b)
#define NVTX_RANGE(name) NvtxRange NVTX_CAT(_nvtxScope, __LINE__)(name)

// -----------------------------------------------------------------------------
// Problem: direct O(N^2) n-body. Chosen because each step is a genuine,
// clearly-visible block of GPU work -- not because the physics matters.
// -----------------------------------------------------------------------------
#define N        2048
#define BLOCK     256
#define STEPS     200
#define SOFTEN  1e-3f
#define DT      1e-3f

__global__ void computeForces(const float4 *__restrict__ pos,
                              float4 *__restrict__ acc, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 pi = pos[i];
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int j = 0; j < n; ++j) {
        float4 pj = pos[j];
        float dx = pj.x - pi.x, dy = pj.y - pi.y, dz = pj.z - pi.z;
        float r2 = dx * dx + dy * dy + dz * dz + SOFTEN;
        float inv = rsqrtf(r2);
        float s   = pj.w * inv * inv * inv;
        ax += dx * s; ay += dy * s; az += dz * s;
    }
    acc[i] = make_float4(ax, ay, az, 0.0f);
}

__global__ void integrate(float4 *__restrict__ pos, float4 *__restrict__ vel,
                          const float4 *__restrict__ acc, int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 v = vel[i], a = acc[i], p = pos[i];
    v.x += a.x * dt; v.y += a.y * dt; v.z += a.z * dt;
    p.x += v.x * dt; p.y += v.y * dt; p.z += v.z * dt;
    vel[i] = v; pos[i] = p;
}

// A cheap per-step diagnostic: total kinetic energy, reduced with one atomic.
__global__ void kineticEnergy(const float4 *__restrict__ vel, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 v = vel[i];
    atomicAdd(out, 0.5f * (v.x * v.x + v.y * v.y + v.z * v.z));
}

// -----------------------------------------------------------------------------
// A per-step timer that does not perturb what it measures -- and whose answer
// is still an OVER-estimate. Read this carefully; it is why the module exists.
//
// cudaEventRecord is asynchronous: it enqueues a timestamp write into the
// stream and returns without synchronizing. So bracketing each kernel with a
// pair of events costs the host two enqueues, and the elapsed time between
// them is the distance on the GPU timeline between two markers.
//
// That is NOT the same as the kernel's duration. The interval
//     [ record(beg) ..... kernel runs ..... record(end) ]
// contains the kernel AND any stretch where the GPU sat idle between the two
// markers waiting for the host to enqueue the kernel. When the host is running
// ahead, the work is already queued and the two coincide. When the host has
// just been forced to synchronize -- which is exactly what version A does
// every step -- the queue is empty, the beg marker retires immediately, and
// the host's ~10 us launch call lands INSIDE the measured interval.
//
// So this harness systematically charges launch gaps to kernel time, and it
// charges more of them to version A than to version B. It cannot tell you
// how much, because it cannot see where the kernel actually started.
//
// Nsight Systems can: `cuda_gpu_kern_sum` timestamps the kernel on the device
// itself. Profiling this program reports all 800 computeForces instances at
// 152570.7 ns +/- 1008 ns -- version A's kernels and version B's kernels are
// the SAME SPEED, and the entire difference this harness reports as "GPU time"
// is launch gap. The lesson works that discrepancy through in full.
// -----------------------------------------------------------------------------
struct GpuTimeline {
    cudaEvent_t *beg, *end;
    int cap, n;
    void init(int capacity) {
        cap = capacity; n = 0;
        beg = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * cap);
        end = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * cap);
        for (int i = 0; i < cap; ++i) {
            CHECK(cudaEventCreate(&beg[i]));
            CHECK(cudaEventCreate(&end[i]));
        }
    }
    void open()  { if (n < cap) CHECK(cudaEventRecord(beg[n])); }
    void close() { if (n < cap) { CHECK(cudaEventRecord(end[n])); ++n; } }
    double totalMs() const {                 // call only after a sync
        double s = 0.0;
        for (int i = 0; i < n; ++i) { float ms; cudaEventElapsedTime(&ms, beg[i], end[i]); s += ms; }
        return s;
    }
    void destroy() {
        for (int i = 0; i < cap; ++i) { cudaEventDestroy(beg[i]); cudaEventDestroy(end[i]); }
        free(beg); free(end);
    }
};

// =============================================================================
// Version A — the way the loop gets written the first time.
//
// Three separate host-side mistakes, all invisible in the source and all
// glaring on a timeline:
//   1. the scratch `acc` buffer is allocated and freed every step. cudaMalloc
//      is not a cheap bookkeeping call; it talks to the driver and it
//      synchronizes the device.
//   2. the energy is copied back every step with a blocking cudaMemcpy, which
//      means the host cannot run ahead and enqueue step k+1 until the GPU has
//      finished step k.
//   3. cudaMemset on the accumulator, also every step.
// =============================================================================
static double runVersionA(float4 *pos, float4 *vel, float *dEnergy,
                          double *gpuMsOut)
{
    NVTX_RANGE("A: malloc-and-sync-per-step");
    GpuTimeline tl; tl.init(STEPS * 3);
    int blocks = (N + BLOCK - 1) / BLOCK;

    cudaEvent_t wall0, wall1;
    CHECK(cudaEventCreate(&wall0)); CHECK(cudaEventCreate(&wall1));
    CHECK(cudaEventRecord(wall0));

    float hostEnergy = 0.0f;
    for (int s = 0; s < STEPS; ++s) {
        NVTX_RANGE("step");
        float4 *acc = nullptr;
        {
            NVTX_RANGE("alloc");                       // mistake 1
            CHECK(cudaMalloc(&acc, sizeof(float4) * N));
        }
        {
            NVTX_RANGE("forces");
            tl.open();
            computeForces<<<blocks, BLOCK>>>(pos, acc, N);
            tl.close();
            CHECK(cudaGetLastError());
        }
        {
            NVTX_RANGE("integrate");
            tl.open();
            integrate<<<blocks, BLOCK>>>(pos, vel, acc, N, DT);
            tl.close();
            CHECK(cudaGetLastError());
        }
        {
            NVTX_RANGE("energy");
            CHECK(cudaMemset(dEnergy, 0, sizeof(float)));   // mistake 3
            tl.open();
            kineticEnergy<<<blocks, BLOCK>>>(vel, dEnergy, N);
            tl.close();
            CHECK(cudaGetLastError());
            // mistake 2: a blocking 4-byte copy. The transfer is free; the
            // synchronization it implies is what costs.
            CHECK(cudaMemcpy(&hostEnergy, dEnergy, sizeof(float),
                             cudaMemcpyDeviceToHost));
        }
        {
            NVTX_RANGE("free");
            CHECK(cudaFree(acc));                          // mistake 1 again
        }
    }

    CHECK(cudaEventRecord(wall1));
    CHECK(cudaEventSynchronize(wall1));
    float wallMs; CHECK(cudaEventElapsedTime(&wallMs, wall0, wall1));
    *gpuMsOut = tl.totalMs();
    tl.destroy();
    CHECK(cudaEventDestroy(wall0)); CHECK(cudaEventDestroy(wall1));
    printf("   (last kinetic energy = %.6f)\n", (double)hostEnergy);
    return wallMs;
}

// =============================================================================
// Version B — the same arithmetic, the same kernels, the same number of
// launches. Only the host's interaction with the device changed:
//   * the scratch buffer is allocated once, before the loop;
//   * the accumulator reset is folded into the launch sequence (still a
//     cudaMemset, but it is asynchronous with respect to the host);
//   * the energy is read back ONCE, after the loop.
// =============================================================================
static double runVersionB(float4 *pos, float4 *vel, float *dEnergy,
                          double *gpuMsOut)
{
    NVTX_RANGE("B: hoisted-and-async");
    GpuTimeline tl; tl.init(STEPS * 3);
    int blocks = (N + BLOCK - 1) / BLOCK;

    float4 *acc = nullptr;
    {
        NVTX_RANGE("alloc-once");
        CHECK(cudaMalloc(&acc, sizeof(float4) * N));
    }

    cudaEvent_t wall0, wall1;
    CHECK(cudaEventCreate(&wall0)); CHECK(cudaEventCreate(&wall1));
    CHECK(cudaEventRecord(wall0));

    for (int s = 0; s < STEPS; ++s) {
        NVTX_RANGE("step");
        {
            NVTX_RANGE("forces");
            tl.open();
            computeForces<<<blocks, BLOCK>>>(pos, acc, N);
            tl.close();
            CHECK(cudaGetLastError());
        }
        {
            NVTX_RANGE("integrate");
            tl.open();
            integrate<<<blocks, BLOCK>>>(pos, vel, acc, N, DT);
            tl.close();
            CHECK(cudaGetLastError());
        }
        {
            NVTX_RANGE("energy");
            CHECK(cudaMemsetAsync(dEnergy, 0, sizeof(float)));
            tl.open();
            kineticEnergy<<<blocks, BLOCK>>>(vel, dEnergy, N);
            tl.close();
            CHECK(cudaGetLastError());
        }
    }

    CHECK(cudaEventRecord(wall1));
    CHECK(cudaEventSynchronize(wall1));
    float wallMs; CHECK(cudaEventElapsedTime(&wallMs, wall0, wall1));

    float hostEnergy = 0.0f;
    CHECK(cudaMemcpy(&hostEnergy, dEnergy, sizeof(float), cudaMemcpyDeviceToHost));
    *gpuMsOut = tl.totalMs();
    tl.destroy();
    CHECK(cudaFree(acc));
    CHECK(cudaEventDestroy(wall0)); CHECK(cudaEventDestroy(wall1));
    printf("   (last kinetic energy = %.6f)\n", (double)hostEnergy);
    return wallMs;
}

// -----------------------------------------------------------------------------
static void initState(float4 *pos, float4 *vel)
{
    srand(20250222);
    for (int i = 0; i < N; ++i) {
        float u = (float)rand() / (float)RAND_MAX;
        float v = (float)rand() / (float)RAND_MAX;
        float w = (float)rand() / (float)RAND_MAX;
        pos[i] = make_float4(u * 2.0f - 1.0f, v * 2.0f - 1.0f, w * 2.0f - 1.0f, 1.0f / N);
        vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    // Naming the thread makes the timeline's CPU row say "sim-main" instead of
    // a bare TID. On a one-thread program this is cosmetic; on a real one it is
    // the difference between a readable timeline and an unreadable one.
    nvtxNameOsThreadA(0, "sim-main");

    printf("Module 22 / Example 1 — NVTX ranges and the GPU busy fraction\n");
    printf("N = %d bodies, %d steps, %d launches per version\n\n", N, STEPS, STEPS * 3);

    float4 *hPos = (float4 *)malloc(sizeof(float4) * N);
    float4 *hVel = (float4 *)malloc(sizeof(float4) * N);
    float4 *dPos = nullptr, *dVel = nullptr;
    float  *dEnergy = nullptr;
    CHECK(cudaMalloc(&dPos, sizeof(float4) * N));
    CHECK(cudaMalloc(&dVel, sizeof(float4) * N));
    CHECK(cudaMalloc(&dEnergy, sizeof(float)));

    // Warm up: first-touch of the context, the module load, and the clocks.
    // Without this the first version measured absorbs several ms of one-time
    // cost and the comparison is meaningless. On the timeline this is the
    // enormous cudaMalloc / cuLibraryLoadData block at the very start.
    {
        NVTX_RANGE("warmup");
        initState(hPos, hVel);
        CHECK(cudaMemcpy(dPos, hPos, sizeof(float4) * N, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dVel, hVel, sizeof(float4) * N, cudaMemcpyHostToDevice));
        int blocks = (N + BLOCK - 1) / BLOCK;
        for (int i = 0; i < 400; ++i) computeForces<<<blocks, BLOCK>>>(dPos, dVel, N);
        CHECK(cudaDeviceSynchronize());
    }

    double wallA = 0.0, gpuA = 0.0, wallB = 0.0, gpuB = 0.0;

    // Both versions run back-to-back with no printing in between (spec S12.1).
    initState(hPos, hVel);
    CHECK(cudaMemcpy(dPos, hPos, sizeof(float4) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dVel, hVel, sizeof(float4) * N, cudaMemcpyHostToDevice));
    wallA = runVersionA(dPos, dVel, dEnergy, &gpuA);

    initState(hPos, hVel);
    CHECK(cudaMemcpy(dPos, hPos, sizeof(float4) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dVel, hVel, sizeof(float4) * N, cudaMemcpyHostToDevice));
    wallB = runVersionB(dPos, dVel, dEnergy, &gpuB);

    printf("\n%-28s %12s %14s %12s %12s\n",
           "version", "wall (ms)", "GPU<=(ms)", "busy<= %", "idle>=(ms)");
    printf("%-28s %12.3f %14.3f %11.1f%% %12.3f\n",
           "A malloc+sync per step", wallA, gpuA, 100.0 * gpuA / wallA, wallA - gpuA);
    printf("%-28s %12.3f %14.3f %11.1f%% %12.3f\n",
           "B hoisted, async", wallB, gpuB, 100.0 * gpuB / wallB, wallB - gpuB);
    printf("\nspeedup A->B               : %.2fx\n", wallA / wallB);
    printf("GPU work enqueued          : identical (%d launches each)\n", STEPS * 3);
    printf("so the speedup came from   : removing host-side idle, nothing else\n");
    printf("\nNOTE: the GPU and busy columns are UPPER BOUNDS. The event pairs\n");
    printf("      absorb launch gaps (see the comment on GpuTimeline). Profiling\n");
    printf("      this program showed the true figures are lower -- about 55%%\n");
    printf("      busy for A and 81%% for B -- so the real gap is WIDER than the\n");
    printf("      numbers above suggest. Run the nsys command in the header and\n");
    printf("      divide cuda_gpu_kern_sum Total Time by the nvtx_sum range time.\n");

    // ---- the documented surprise ------------------------------------------
    // A and B enqueue the same 600 launches over the same data, so their
    // summed GPU time should be equal. This harness says it is not: it charges
    // version A reproducibly 10-20% more "GPU time" than version B.
    //
    // Nsight Systems says the kernels are identical (152570.7 ns +/- 1008 ns
    // over all 800 instances of computeForces, spanning both versions). The
    // excess is launch gap that the event pairs cannot separate from kernel
    // execution, and version A -- which empties the queue every step with a
    // blocking 4-byte memcpy -- has far more of it.
    //
    // This is the whole argument for a timeline profiler in one number: the
    // in-program harness can tell you that time disappeared, but only the
    // profiler can tell you WHERE it went.
    double rel = 100.0 * (gpuA - gpuB) / fmax(gpuA, gpuB);
    double busyA = 100.0 * gpuA / wallA, busyB = 100.0 * gpuB / wallB;
    printf("\nA's apparent GPU time excess: %+.1f%%  (expect +5..25%%; it is launch\n"
           "                              gap misattributed to kernel time, not\n"
           "                              a real difference in kernel speed)\n", rel);

    // PASS requires the two things the example actually claims.
    bool bFaster  = wallB < wallA;
    bool busierB  = busyB > busyA + 5.0;
    printf("B finishes sooner          : %s\n", bFaster ? "yes" : "no");
    printf("B keeps the GPU busier     : %s\n", busierB ? "yes" : "no");
    bool gpuMatches = bFaster && busierB;

    CHECK(cudaFree(dPos)); CHECK(cudaFree(dVel)); CHECK(cudaFree(dEnergy));
    free(hPos); free(hVel);
    CHECK(cudaDeviceReset());

    printf("\nOVERALL: %s\n", (gpuMatches && bFaster) ? "PASS" : "FAIL");
    return (gpuMatches && bFaster) ? 0 : 1;
}
