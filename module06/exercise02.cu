// =====================================================================
// Module 6 / Exercise 2 : "One blob, three arrays"
//
// GOAL
//   Use DYNAMIC shared memory -- `extern __shared__` plus the third
//   launch parameter -- on a problem with enough reuse that staging
//   actually pays, and carve three differently-typed arrays out of the
//   single untyped allocation without tripping over alignment.
//
// THE PROBLEM
//   Radial-basis scatter interpolation. There are MS source points,
//   each with a position p[j] (float2) and a weight w[j] (float). There
//   are NQ query points q[i] (float2). For every query:
//
//       out[i] = sum over j of  w[j] * phi( |q[i] - p[j]|^2 )
//
//   phi is not an analytic function: it is a table of PROF_N samples of
//   a radial profile on [0,1], linearly interpolated, and zero beyond
//   radius 1. `profile_eval` below does the lookup; use it as given.
//
//   Note what the lookup costs in the untiled kernel. The table index
//   is derived from a distance, so it differs from lane to lane within
//   a warp. Module 4 measured what a lane-varying index does to
//   constant memory (24.6x). Global memory is not as bad, but a warp
//   still issues one gather per lookup, MS times per thread.
//
//   Every thread of a block needs every source point, so each source
//   staged in shared memory is consumed by all TILE threads of the
//   block: the reuse factor K is TILE, not 5. This is the regime where
//   staging is supposed to win.
//
// THE SHARED-MEMORY LAYOUT (fixed -- do not reorder it)
//       float  prof[PROF_N]      the profile table
//       float2 pos [TILE]        one tile of source positions
//       float  wgt [TILE]        one tile of source weights
//   in that order, in one `extern __shared__` allocation. PROF_N is 33
//   on purpose.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
// RUN:    .\exercise02.exe
//
// If the program dies with a CUDA error before printing any timings,
// read the error name carefully and then re-read the layout above.
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

#define PROF_N 33
#define TILE   256

static const int NQ = 262111;   // not a multiple of TILE
static const int MS = 4093;     // not a multiple of TILE either

// Rounds `off` up to the next multiple of `a`. You will want this.
__host__ __device__ __forceinline__ size_t align_up(size_t off, size_t a)
{
    return (off + a - 1) & ~(a - 1);
}

// ---------------------------------------------------------------------
// TODO 1: Return the number of bytes the kernel below needs, for the
//         layout documented in the header. This value is passed as the
//         third launch parameter.
//
//         There are two independent ways to get this wrong. One makes
//         the program die with a specific CUDA error before any result
//         is printed. The other makes every block write past the end of
//         the allocation it asked for -- and on this hardware that is
//         *silent*: no error, no sanitizer report, correct answers.
//         Reason it out rather than testing for it.
//
//         Returning 0 makes main() print a reminder and stop.
// ---------------------------------------------------------------------
static size_t sharedBytesFor(void)
{
    return 0;   // YOUR CODE HERE (TODO 1)
}

// ---------------------------------------------------------------------
__host__ __device__ __forceinline__
float profile_eval(const float* prof, float d2)
{
    if (d2 >= 1.0f) return 0.0f;
    float t = d2 * (float)(PROF_N - 1);
    int   k = (int)t;
    if (k > PROF_N - 2) k = PROF_N - 2;
    float f = t - (float)k;
    return prof[k] + (prof[k + 1] - prof[k]) * f;
}

// The control. Do not modify.
__global__ void rbf_naive(const float2* __restrict__ q,
                          const float2* __restrict__ p,
                          const float*  __restrict__ w,
                          const float*  __restrict__ prof,
                          float* __restrict__ out, int nq, int m)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= nq) return;
    float2 qq = q[i];
    float acc = 0.0f;
    for (int j = 0; j < m; ++j) {
        float2 pp = p[j];
        float dx = qq.x - pp.x, dy = qq.y - pp.y;
        acc += w[j] * profile_eval(prof, dx * dx + dy * dy);
    }
    out[i] = acc;
}

// =====================================================================
__global__ void rbf_tiled(const float2* __restrict__ q,
                          const float2* __restrict__ p,
                          const float*  __restrict__ w,
                          const float*  __restrict__ prof,
                          float* __restrict__ out, int nq, int m)
{
    extern __shared__ char smem[];

    // -----------------------------------------------------------------
    // TODO 2: Point sprof, spos and swgt at the right places inside
    //         `smem`, following the layout in the header.
    //
    //         `smem` is one untyped blob. The compiler will happily
    //         cast any byte offset to any pointer type; the hardware
    //         will not. Work out what each of these three types
    //         requires and make sure the offsets you hand out satisfy
    //         it. Whatever you decide here must agree exactly with
    //         TODO 1.
    // -----------------------------------------------------------------
    float*  sprof = reinterpret_cast<float*>(smem);    // YOUR CODE HERE (TODO 2)
    float2* spos  = reinterpret_cast<float2*>(smem);   // YOUR CODE HERE (TODO 2)
    float*  swgt  = reinterpret_cast<float*>(smem);    // YOUR CODE HERE (TODO 2)

    int i = (int)(blockIdx.x * TILE + threadIdx.x);

    // Note: this thread does NOT return early when i >= nq. Work out
    // why before you write TODO 3.
    float2 qq = q[i < nq ? i : nq - 1];
    float acc = 0.0f;

    for (int t0 = 0; t0 < m; t0 += TILE) {
        // -------------------------------------------------------------
        // TODO 3: Stage one tile of sources (and, once, the profile
        //         table) in shared memory and consume it.
        //
        //         Three things to get right:
        //           (a) the profile table is PROF_N entries and there
        //               are TILE threads, and it only has to be loaded
        //               once for the whole kernel, not once per tile;
        //           (b) the last tile is partial, because MS is not a
        //               multiple of TILE -- decide what the threads
        //               with no source to load should do, and make sure
        //               no thread accumulates a contribution from a
        //               slot that holds nothing;
        //           (c) every thread must observe a complete tile
        //               before it reads one, and no thread may start
        //               overwriting the tile while another thread is
        //               still reading the previous one. Both of those
        //               are requirements on ordering, not on values.
        //
        //         Accumulate into `acc`.
        // -------------------------------------------------------------
        // YOUR CODE HERE (TODO 3)

        // Placeholder: this line exists only so the file compiles
        // warning-clean with TODO 3 empty. Delete it.
        acc += 0.0f * (sprof[0] + spos[0].x + swgt[0] + qq.x
                       + p[t0].x + w[t0] + prof[0]);
    }

    if (i < nq) out[i] = acc;
}

// ---------------------------------------------------------------------
static double cpu_one(const float2* q, const float2* p, const float* w,
                      const float* prof, int i, int m)
{
    double acc = 0.0;
    for (int j = 0; j < m; ++j) {
        float dx = q[i].x - p[j].x, dy = q[i].y - p[j].y;
        acc += (double)w[j] * (double)profile_eval(prof, dx * dx + dy * dy);
    }
    return acc;
}

struct Cfg { const char* name; };
static const Cfg CFG[2] = { { "naive" }, { "tiled (dynamic smem)" } };

static float2 *d_q, *d_p;
static float *d_w, *d_prof, *d_out;
static size_t g_smem;

static void launch(int i)
{
    int grid = (NQ + TILE - 1) / TILE;
    if (i == 0) rbf_naive<<<grid, TILE>>>(d_q, d_p, d_w, d_prof, d_out, NQ, MS);
    else        rbf_tiled<<<grid, TILE, g_smem>>>(d_q, d_p, d_w, d_prof, d_out, NQ, MS);
}

// ---------------------------------------------------------------------
int main(void)
{
    CHECK(cudaSetDevice(0));
    printf("=== Module 6 / Exercise 2 : dynamic shared memory ===\n");

    g_smem = sharedBytesFor();
    printf("  PROF_N=%d  TILE=%d  NQ=%d  MS=%d\n", PROF_N, TILE, NQ, MS);
    printf("  sharedBytes = %zu\n", g_smem);
    if (g_smem == 0) {
        printf("\nSet TODO 1 first.\n");
        CHECK(cudaDeviceReset());
        return 0;
    }

    float2* h_q = (float2*)malloc(NQ * sizeof(float2));
    float2* h_p = (float2*)malloc(MS * sizeof(float2));
    float*  h_w = (float*)malloc(MS * sizeof(float));
    float*  h_o = (float*)malloc(NQ * sizeof(float));
    float   h_prof[PROF_N];
    if (!h_q || !h_p || !h_w || !h_o) { printf("host alloc failed\n"); return 1; }

    for (int i = 0; i < NQ; ++i) {
        h_q[i].x = 4.0f * (float)((i * 7919) % 1013) / 1013.0f;
        h_q[i].y = 4.0f * (float)((i * 6271) % 977)  / 977.0f;
    }
    for (int j = 0; j < MS; ++j) {
        h_p[j].x = 4.0f * (float)((j * 211) % 499) / 499.0f;
        h_p[j].y = 4.0f * (float)((j * 307) % 443) / 443.0f;
        h_w[j]   = 0.01f * (float)(1 + (j % 97));
    }
    for (int k = 0; k < PROF_N; ++k) {
        float r2 = (float)k / (float)(PROF_N - 1);
        h_prof[k] = (1.0f - r2) * (1.0f - r2);
    }

    CHECK(cudaMalloc(&d_q, NQ * sizeof(float2)));
    CHECK(cudaMalloc(&d_p, MS * sizeof(float2)));
    CHECK(cudaMalloc(&d_w, MS * sizeof(float)));
    CHECK(cudaMalloc(&d_prof, PROF_N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, NQ * sizeof(float)));
    CHECK(cudaMemcpy(d_q, h_q, NQ * sizeof(float2), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_p, h_p, MS * sizeof(float2), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, h_w, MS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_prof, h_prof, sizeof(h_prof), cudaMemcpyHostToDevice));

    // ---- pass 1: timing, both configs back to back, min of 4 sweeps --
    double best[2] = { 1e30, 1e30 };
    {
        cudaEvent_t a, b; float ms = 0.f;
        CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        CHECK(cudaEventRecord(a));                 // duration-based warm-up
        do {
            for (int i = 0; i < 10; ++i) launch(1);
            CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
            CHECK(cudaEventElapsedTime(&ms, a, b));
        } while (ms < 400.0f);
        CHECK(cudaGetLastError());

        for (int sweep = 0; sweep < 4; ++sweep)
            for (int c = 0; c < 2; ++c) {
                launch(c);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaEventRecord(a));
                for (int it = 0; it < 20; ++it) launch(c);
                CHECK(cudaEventRecord(b));
                CHECK(cudaEventSynchronize(b));
                CHECK(cudaEventElapsedTime(&ms, a, b));
                if (ms / 20.0 < best[c]) best[c] = ms / 20.0;
            }
        CHECK(cudaGetLastError());
        CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
    }

    // ---- pass 2: correctness ----------------------------------------
    int fails = 0;
    for (int c = 0; c < 2; ++c) {
        CHECK(cudaMemset(d_out, 0, NQ * sizeof(float)));
        launch(c);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_o, d_out, NQ * sizeof(float), cudaMemcpyDeviceToHost));
        long long bad = 0; int first = -1;
        for (int i = 0; i < NQ; i += 17) {
            double ref = cpu_one(h_q, h_p, h_w, h_prof, i, MS);
            if (!(fabs((double)h_o[i] - ref) <= 1e-4 * fmax(1.0, fabs(ref)))) {
                if (first < 0) first = i;
                ++bad;
            }
        }
        printf("  %-22s %s", CFG[c].name, bad ? "FAIL" : "PASS");
        if (bad) {
            printf("  %lld sampled queries wrong; first i=%d: got %.6f want %.6f",
                   bad, first, h_o[first],
                   cpu_one(h_q, h_p, h_w, h_prof, first, MS));
            ++fails;
        }
        printf("\n");
    }

    double pairs = (double)NQ * (double)MS;
    printf("\n  %-22s %10s %14s %12s\n", "config", "ms", "Gpair/s", "vs naive");
    for (int c = 0; c < 2; ++c)
        printf("  %-22s %10.4f %14.2f %11.2fx\n",
               CFG[c].name, best[c], pairs / (best[c] * 1.0e-3) / 1.0e9,
               best[0] / best[c]);

    CHECK(cudaFree(d_q)); CHECK(cudaFree(d_p)); CHECK(cudaFree(d_w));
    CHECK(cudaFree(d_prof)); CHECK(cudaFree(d_out));
    free(h_q); free(h_p); free(h_w); free(h_o);

    printf("\nOVERALL: %s\n", fails == 0 ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return fails == 0 ? 0 : 1;
}
