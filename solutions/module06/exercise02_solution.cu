// =====================================================================
// Module 6 / Exercise 2 : SOLUTION
//   Dynamic shared memory: radial-basis scatter interpolation with a
//   staged source tile and a staged profile table.
//
// BUILD:  nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
// RUN:    .\exercise02_solution.exe
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

static const int NQ = 262111;
static const int MS = 4093;

__host__ __device__ __forceinline__ size_t align_up(size_t off, size_t a)
{
    return (off + a - 1) & ~(a - 1);
}

// ---- TODO 1 : the byte count -----------------------------------------
static size_t sharedBytesFor(void)
{
    size_t off_pos = align_up((size_t)PROF_N * sizeof(float), alignof(float2));
    size_t off_wgt = align_up(off_pos + (size_t)TILE * sizeof(float2),
                              alignof(float));
    return off_wgt + (size_t)TILE * sizeof(float);
}

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

// ---------------------------------------------------------------------
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

// ---------------------------------------------------------------------
__global__ void rbf_tiled(const float2* __restrict__ q,
                          const float2* __restrict__ p,
                          const float*  __restrict__ w,
                          const float*  __restrict__ prof,
                          float* __restrict__ out, int nq, int m)
{
    extern __shared__ char smem[];

    // ---- TODO 2 : carve three typed arrays out of one blob ----------
    const size_t off_pos = align_up((size_t)PROF_N * sizeof(float),
                                    alignof(float2));
    const size_t off_wgt = align_up(off_pos + (size_t)TILE * sizeof(float2),
                                    alignof(float));
    float*  sprof = reinterpret_cast<float*>(smem);
    float2* spos  = reinterpret_cast<float2*>(smem + off_pos);
    float*  swgt  = reinterpret_cast<float*>(smem + off_wgt);

    // ---- TODO 3 : cooperative loads and barriers --------------------
    for (int k = (int)threadIdx.x; k < PROF_N; k += TILE) sprof[k] = prof[k];

    int i = (int)(blockIdx.x * TILE + threadIdx.x);
    float2 qq = q[i < nq ? i : nq - 1];          // clamp, do not return:
                                                 // every thread must reach
                                                 // every barrier below
    float acc = 0.0f;
    for (int t0 = 0; t0 < m; t0 += TILE) {
        int cnt = (m - t0 < TILE) ? (m - t0) : TILE;

        __syncthreads();                  // nobody is still reading the
                                          // previous tile (and sprof is
                                          // complete on the first pass)
        if ((int)threadIdx.x < cnt) {
            spos[threadIdx.x] = p[t0 + threadIdx.x];
            swgt[threadIdx.x] = w[t0 + threadIdx.x];
        }
        __syncthreads();                  // tile is complete before use

        for (int j = 0; j < cnt; ++j) {
            float dx = qq.x - spos[j].x, dy = qq.y - spos[j].y;
            acc += swgt[j] * profile_eval(sprof, dx * dx + dy * dy);
        }
    }
    if (i < nq) out[i] = acc;
}

// ---------------------------------------------------------------------
// The other legal way to handle the partial last tile: instead of
// shortening the inner loop, PAD the tile with zero-weight entries so
// the inner bound is the compile-time constant TILE and ptxas can
// unroll it.
//
// Both versions validate. They are not the same speed -- and the
// faster one is not the one the "give ptxas a constant bound" rule of
// thumb predicts. See the solution notes.
// ---------------------------------------------------------------------
__global__ void rbf_tiled_padded(const float2* __restrict__ q,
                                 const float2* __restrict__ p,
                                 const float*  __restrict__ w,
                                 const float*  __restrict__ prof,
                                 float* __restrict__ out, int nq, int m)
{
    extern __shared__ char smem[];
    const size_t off_pos = align_up((size_t)PROF_N * sizeof(float),
                                    alignof(float2));
    const size_t off_wgt = align_up(off_pos + (size_t)TILE * sizeof(float2),
                                    alignof(float));
    float*  sprof = reinterpret_cast<float*>(smem);
    float2* spos  = reinterpret_cast<float2*>(smem + off_pos);
    float*  swgt  = reinterpret_cast<float*>(smem + off_wgt);

    for (int k = (int)threadIdx.x; k < PROF_N; k += TILE) sprof[k] = prof[k];

    int i = (int)(blockIdx.x * TILE + threadIdx.x);
    float2 qq = q[i < nq ? i : nq - 1];
    float acc = 0.0f;

    for (int t0 = 0; t0 < m; t0 += TILE) {
        int cnt = (m - t0 < TILE) ? (m - t0) : TILE;
        __syncthreads();
        if ((int)threadIdx.x < cnt) {
            spos[threadIdx.x] = p[t0 + threadIdx.x];
            swgt[threadIdx.x] = w[t0 + threadIdx.x];
        } else {
            spos[threadIdx.x] = make_float2(0.0f, 0.0f);
            swgt[threadIdx.x] = 0.0f;           // contributes nothing
        }
        __syncthreads();
#pragma unroll 8
        for (int j = 0; j < TILE; ++j) {
            float dx = qq.x - spos[j].x, dy = qq.y - spos[j].y;
            acc += swgt[j] * profile_eval(sprof, dx * dx + dy * dy);
        }
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

struct Cfg { const char* name; int tiled; };
static const Cfg CFG[3] = { { "naive", 0 },
                            { "tiled, short loop", 1 },
                            { "tiled, padded tile", 2 } };
#define NCFG 3

static float2 *d_q, *d_p;
static float *d_w, *d_prof, *d_out;
static size_t g_smem;

static void launch(int i)
{
    int grid = (NQ + TILE - 1) / TILE;
    if (i == 0)      rbf_naive<<<grid, TILE>>>(d_q, d_p, d_w, d_prof, d_out, NQ, MS);
    else if (i == 1) rbf_tiled<<<grid, TILE, g_smem>>>(d_q, d_p, d_w, d_prof, d_out, NQ, MS);
    else             rbf_tiled_padded<<<grid, TILE, g_smem>>>(d_q, d_p, d_w, d_prof, d_out, NQ, MS);
}

// ---------------------------------------------------------------------
int main(void)
{
    CHECK(cudaSetDevice(0));
    printf("=== Module 6 / Exercise 2 : dynamic shared memory (SOLUTION) ===\n");

    g_smem = sharedBytesFor();
    size_t off_pos = align_up((size_t)PROF_N * sizeof(float), alignof(float2));
    printf("  PROF_N=%d  TILE=%d  NQ=%d  MS=%d\n", PROF_N, TILE, NQ, MS);
    printf("  offsets: prof 0, pos %zu (naive would be %zu), wgt %zu\n",
           off_pos, (size_t)PROF_N * sizeof(float),
           off_pos + (size_t)TILE * sizeof(float2));
    printf("  sharedBytes = %zu\n", g_smem);
    if (g_smem == 0) { printf("\nSet TODO 1 first.\n"); CHECK(cudaDeviceReset()); return 0; }

    float2* h_q = (float2*)malloc(NQ * sizeof(float2));
    float2* h_p = (float2*)malloc(MS * sizeof(float2));
    float*  h_w = (float*)malloc(MS * sizeof(float));
    float*  h_o = (float*)malloc(NQ * sizeof(float));
    float   h_prof[PROF_N];

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
    double best[NCFG] = { 1e30, 1e30, 1e30 };
    {
        cudaEvent_t a, b; float ms = 0.f;
        CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
        CHECK(cudaEventRecord(a));                    // duration-based warm-up
        do {
            for (int i = 0; i < 10; ++i) launch(1);
            CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
            CHECK(cudaEventElapsedTime(&ms, a, b));
        } while (ms < 400.0f);
        CHECK(cudaGetLastError());

        for (int sweep = 0; sweep < 4; ++sweep)
            for (int c = 0; c < NCFG; ++c) {
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
    for (int c = 0; c < NCFG; ++c) {
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

    // ---- report -------------------------------------------------------
    double pairs = (double)NQ * (double)MS;
    printf("\n  %-22s %10s %14s %12s\n", "config", "ms", "Gpair/s", "vs naive");
    for (int c = 0; c < NCFG; ++c)
        printf("  %-22s %10.4f %14.2f %11.2fx\n",
               CFG[c].name, best[c], pairs / (best[c] * 1.0e-3) / 1.0e9,
               best[0] / best[c]);
    printf("\n  reuse K: each staged source is used by all %d threads of the block;\n"
           "  each profile entry is read ~%.0f times per block.\n",
           TILE, pairs / (double)PROF_N / ((double)NQ / TILE));

    CHECK(cudaFree(d_q)); CHECK(cudaFree(d_p)); CHECK(cudaFree(d_w));
    CHECK(cudaFree(d_prof)); CHECK(cudaFree(d_out));
    free(h_q); free(h_p); free(h_w); free(h_o);

    printf("\nOVERALL: %s\n", fails == 0 ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return fails == 0 ? 0 : 1;
}
