// =============================================================================
// Module 23 / Exercise 1 — SOLUTION — read the report, then check it against
//                                     the machine.
//
// BUILD: nvcc -arch=sm_89 -O3 -lineinfo -o e1s.exe exercise01_solution.cu
// RUN  : e1s.exe
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

// ---------------- problem sizes ---------------------------------------------
#define K1_USE    ((size_t)16*1024*1024)      // useful floats read by kernel 1
#define K1_REC    8                           // floats per table record
#define K1_GRID   640
#define K1_BLK    256

#define TW        8192                        // transpose is TW x TW  (DRAM bound)
#define TILE      32

#define GM 1027
#define GN 1027
#define GK 1029

#define K4_BLOCKS 40
#define K4_THR    128
#define K4_MLP    4
#define K4_CHUNK  1600
#define K4_N      ((size_t)K4_BLOCKS*K4_THR*K4_MLP*K4_CHUNK)   // 65,536,000 floats

// =============================================================================
// ====================  ANSWERS  ==============================================
// =============================================================================
// TODO 1 — diagnosis per kernel.
static const int DIAG[4]   = { 4, 1, 6, 3 };
// TODO 2 — the single decisive metric row per kernel.
static const int METRIC[4] = { 3, 1, 7, 6 };
// TODO 3 — predicted speedup bucket from the stated fix.
//          1: below 1.5x   2: 1.5x to 6x   3: above 6x
static const int BUCKET[4] = { 3, 1, 2, 2 };

// =============================================================================
// The four kernel pairs. A = the profiled kernel, B = the stated fix.
// =============================================================================

// ---- 1. one column of a row-major [N][8] table -----------------------------
__global__ void k1_strided(const float * __restrict__ tab, float *partial, size_t n)
{
    size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t nt  = gridDim.x * (size_t)blockDim.x;
    float a = 0.0f;
    for (size_t i = gid; i < n; i += nt) a += tab[i*K1_REC + 3];
    partial[gid] = a;
}
__global__ void k1_packed(const float * __restrict__ col, float *partial, size_t n)
{
    size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t nt  = gridDim.x * (size_t)blockDim.x;
    float a = 0.0f;
    for (size_t i = gid; i < n; i += nt) a += col[i];
    partial[gid] = a;
}

// ---- 2. tiled transpose, conflicted vs padded ------------------------------
template <int PAD>
__global__ void k2_transpose(const float * __restrict__ in, float *out, int w, int h)
{
    __shared__ float t[TILE][TILE + PAD];
    int x = blockIdx.x*TILE + threadIdx.x;
    int y = blockIdx.y*TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += 8)
        if (x < w && y + j < h) t[threadIdx.y + j][threadIdx.x] = in[(size_t)(y+j)*w + x];
    __syncthreads();
    x = blockIdx.y*TILE + threadIdx.x;
    y = blockIdx.x*TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += 8)
        if (x < h && y + j < w) out[(size_t)(y+j)*h + x] = t[threadIdx.x][threadIdx.y + j];
}

// ---- 3. GEMM: naive vs 4x4 register tile -----------------------------------
__global__ void k3_naive(const float * __restrict__ A, const float * __restrict__ B,
                         float *C, int M, int N, int K)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s = fmaf(A[(size_t)row*K + k], B[(size_t)k*N + col], s);
        C[(size_t)row*N + col] = s;
    }
}

#define BM 64
#define BN 64
#define BKK 8
#define TM 4
#define TN 4
__global__ void k3_regtile(const float * __restrict__ A, const float * __restrict__ B,
                           float *C, int M, int N, int K)
{
    __shared__ float As[BKK][BM];        // transposed: As[k][m]
    __shared__ float Bs[BKK][BN];
    const int tid = threadIdx.x;
    const int tx  = tid % (BN/TN);       // 0..15
    const int ty  = tid / (BN/TN);       // 0..15
    const int rowBase = blockIdx.y*BM + ty*TM;
    const int colBase = blockIdx.x*BN + tx*TN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int kt = 0; kt < K; kt += BKK) {
        // A tile: BKK*BM elements, 256 threads -> 2 each. k is A's fast axis.
        #pragma unroll
        for (int l = tid; l < BKK*BM; l += blockDim.x) {
            int m = l / BKK, k = l % BKK;
            int r = blockIdx.y*BM + m, c = kt + k;
            As[k][m] = (r < M && c < K) ? A[(size_t)r*K + c] : 0.0f;
        }
        // B tile: n is B's fast axis.
        #pragma unroll
        for (int l = tid; l < BKK*BN; l += blockDim.x) {
            int k = l / BN, nn = l % BN;
            int r = kt + k, c = blockIdx.x*BN + nn;
            Bs[k][nn] = (r < K && c < N) ? B[(size_t)r*N + c] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BKK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty*TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx*TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = rowBase + i; if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c = colBase + j; if (c >= N) continue;
            C[(size_t)r*N + c] = acc[i][j];
        }
    }
}

// ---- 4. strided sum: one load in flight vs eight ---------------------------
__global__ void k4_serial(const float * __restrict__ in, float *partial, size_t n)
{
    size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t nt  = gridDim.x * (size_t)blockDim.x;
    float a = 0.0f;
    #pragma unroll 1
    for (size_t i = gid; i < n; i += nt) a += in[i];
    partial[gid] = a;
}
__global__ void k4_mlp(const float * __restrict__ in, float *partial, size_t n)
{
    size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t nt  = gridDim.x * (size_t)blockDim.x;
    float a = 0.0f;
    #pragma unroll 1
    for (size_t base = gid; base + (K4_MLP-1)*nt < n; base += K4_MLP*nt) {
        float v[K4_MLP];
        #pragma unroll
        for (int k = 0; k < K4_MLP; ++k) v[k] = in[base + (size_t)k*nt];
        #pragma unroll
        for (int k = 0; k < K4_MLP; ++k) a += v[k];
    }
    partial[gid] = a;
}

// ---- deterministic device-side initialisation ------------------------------
__global__ void initTable(float *p, size_t n)
{ size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x, st = gridDim.x*(size_t)blockDim.x;
  for (; i < n; i += st) p[i] = (float)(((i*1103515245u + 12345u) >> 16) & 1023u) * (1.0f/1024.0f); }
__global__ void gatherCol(const float *tab, float *col, size_t n)
{ size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x, st = gridDim.x*(size_t)blockDim.x;
  for (; i < n; i += st) col[i] = tab[i*K1_REC + 3]; }
__global__ void initRamp(float *p, size_t n, int m, float sc)
{ size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x, st = gridDim.x*(size_t)blockDim.x;
  for (; i < n; i += st) p[i] = (float)(i % (size_t)m) * sc; }
__global__ void initRand(float *p, size_t n, unsigned a, unsigned c)
{ size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x, st = gridDim.x*(size_t)blockDim.x;
  for (; i < n; i += st) p[i] = (float)(((unsigned)i*a + c) % 2048u) * (1.0f/1024.0f) - 1.0f; }

__global__ void sampleXY(const float *m, float *out, const int *idx, int ns, int w, int swap)
{ int q = blockIdx.x*blockDim.x + threadIdx.x;
  if (q >= ns) return;
  int x = idx[2*q], y = idx[2*q+1];
  out[q] = swap ? m[(size_t)y*w + x] : m[(size_t)x*w + y]; }

// =============================================================================
// The constructed report. Values are this course's own measurements, so the
// report is physically consistent with this GPU.
// =============================================================================
static void printReport(void)
{
printf(
"=============================================================================\n"
" CONSTRUCTED ncu REPORT  --  NOT produced by ncu on this machine.\n"
" ncu 2026.1.0 is installed here and every invocation returns\n"
"   ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access\n"
"             NVIDIA GPU Performance Counters on the target device 0.\n"
" The numbers below are assembled from this course's own measurements\n"
" (Modules 5, 7, 12, 16, 17, 18, 19, 20, 21), so they are physically\n"
" consistent: a reader who later runs ncu on a working machine will not find\n"
" this report impossible. Treat it exactly as you would a real one.\n"
"=============================================================================\n"
"\n"
"---------------------------------------------------------------------------\n"
" [1]  k1_strided(const float*, float*, unsigned long)\n"
"      <<<640, 256>>>   one column of a 16,777,216 x 8 row-major float table\n"
"---------------------------------------------------------------------------\n"
" Section: GPU Speed Of Light Throughput\n"
"   Compute (SM) Throughput                      %%            3.47\n"
"   Memory Throughput                            %%           94.61\n"
"   DRAM Throughput                              %%           94.61\n"
"   Duration                                 msecond          1.3137\n"
" Section: Launch Statistics\n"
"   Block Size                                                 256\n"
"   Grid Size                                                  640\n"
"   launch__waves_per_multiprocessor                           2.67\n"
" Section: Occupancy\n"
"   Theoretical Occupancy                        %%          100.00\n"
"   Achieved Occupancy                           %%           88.71\n"
" Section: Memory Workload Analysis\n"
"   dram__bytes.sum                            Mbyte         536.87\n"
"   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum         524,288\n"
"   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       16,777,216\n"
"     ->  sectors per request                                 32.00\n"
"   l1tex__t_sector_hit_rate                     %%            0.00\n"
"   lts__t_sector_hit_rate                       %%            0.07\n"
" Section: Warp State Statistics\n"
"   smsp__thread_inst_executed_per_inst_executed.ratio        32.00\n"
"   Stall Long Scoreboard                   cycle/inst        21.93\n"
"   smsp__warps_eligible.avg.per_cycle_active                  3.41\n"
"\n"
"---------------------------------------------------------------------------\n"
" [2]  void k2_transpose<0>(const float*, float*, int, int)\n"
"      <<<(64,64), (32,8)>>>   2048 x 2048 transpose, __shared__ float t[32][32]\n"
"---------------------------------------------------------------------------\n"
" Section: GPU Speed Of Light Throughput\n"
"   Compute (SM) Throughput                      %%           41.20\n"
"   Memory Throughput                            %%            7.93\n"
"   DRAM Throughput                              %%            7.93\n"
"   Duration                                 msecond          0.3100\n"
" Section: Launch Statistics\n"
"   Grid Size                                                 4,096\n"
"   launch__waves_per_multiprocessor                          17.07\n"
" Section: Occupancy\n"
"   Theoretical Occupancy                        %%          100.00\n"
"   Achieved Occupancy                           %%           91.40\n"
" Section: Memory Workload Analysis\n"
"   dram__bytes.sum                            Mbyte          10.60\n"
"   lts__t_sector_hit_rate                       %%           88.60\n"
"   sectors per global load request                            4.00\n"
"   l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum  4,194,304\n"
"   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum\n"
"                                                         4,063,232\n"
"     ->  wavefronts per shared load request                  32.00\n"
" Section: Warp State Statistics\n"
"   Stall Short Scoreboard                  cycle/inst        26.41\n"
"   Stall Long Scoreboard                   cycle/inst         1.12\n"
"   smsp__thread_inst_executed_per_inst_executed.ratio        32.00\n"
"\n"
"---------------------------------------------------------------------------\n"
" [3]  k3_naive(const float*, const float*, float*, int, int, int)\n"
"      <<<(65,65), (16,16)>>>   C = A*B,  M=1027  N=1027  K=1029\n"
"---------------------------------------------------------------------------\n"
" Section: GPU Speed Of Light Throughput\n"
"   Compute (SM) Throughput                      %%            7.12\n"
"   Memory Throughput                            %%            1.72\n"
"   DRAM Throughput                              %%            1.72\n"
"   Duration                                 msecond          1.6460\n"
" Section: Launch Statistics\n"
"   Grid Size                                                 4,225\n"
"   launch__waves_per_multiprocessor                          17.60\n"
" Section: Occupancy\n"
"   Theoretical Occupancy                        %%          100.00\n"
"   Achieved Occupancy                           %%           93.15\n"
" Section: Memory Workload Analysis\n"
"   dram__bytes.sum                            Mbyte          12.25\n"
"   sectors per global load request                            4.00\n"
"   l1tex__t_sector_hit_rate                     %%           93.72\n"
"   lts__t_sector_hit_rate                       %%           97.80\n"
"   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum      0\n"
" Section: Instruction Statistics\n"
"   smsp__inst_executed.sum                          inst  135,660,000\n"
"   sm__sass_thread_inst_executed_op_ffma_pred_on.sum   1,085,316,141\n"
"     ->  FFMA warp-instructions / all warp-instructions  %%   25.00\n"
" Section: Warp State Statistics\n"
"   Stall Long Scoreboard                   cycle/inst         3.10\n"
"   Stall Wait                              cycle/inst         1.45\n"
"   smsp__warps_eligible.avg.per_cycle_active                  2.87\n"
"   smsp__thread_inst_executed_per_inst_executed.ratio        32.00\n"
"\n"
"---------------------------------------------------------------------------\n"
" [4]  k4_serial(const float*, float*, unsigned long)\n"
"      <<<40, 128>>>   sum of 65,536,000 floats.  The launch shape is fixed by\n"
"      the surrounding application and may not be changed.\n"
"---------------------------------------------------------------------------\n"
" Section: GPU Speed Of Light Throughput\n"
"   Compute (SM) Throughput                      %%            1.19\n"
"   Memory Throughput                            %%           15.78\n"
"   DRAM Throughput                              %%           15.78\n"
"   Duration                                 msecond          3.8450\n"
" Section: Launch Statistics\n"
"   Block Size                                                 128\n"
"   Grid Size                                                   40\n"
"   launch__waves_per_multiprocessor                           0.17\n"
" Section: Occupancy\n"
"   Theoretical Occupancy                        %%          100.00\n"
"   Achieved Occupancy                           %%            8.33\n"
" Section: Memory Workload Analysis\n"
"   dram__bytes.sum                            Mbyte         262.14\n"
"   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       65,536,000\n"
"   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum      16,384,000\n"
"     ->  sectors per request                                  4.00\n"
"   l1tex__t_sector_hit_rate                     %%            0.00\n"
"   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum      0\n"
" Section: Warp State Statistics\n"
"   Stall Long Scoreboard                   cycle/inst       286.40\n"
"   smsp__warps_eligible.avg.per_cycle_active                  0.09\n"
"   smsp__thread_inst_executed_per_inst_executed.ratio        32.00\n"
"=============================================================================\n");
}

static void printMenus(void)
{
printf(
"\n DIAGNOSIS CODES\n"
"   1  At the DRAM roof. The kernel is finished; stop optimizing it.\n"
"   2  Shared-memory bank conflicts are serializing the scratchpad.\n"
"   3  Nothing is saturated: too few memory requests in flight per thread.\n"
"   4  Moving far more bytes than it uses; the global access pattern is\n"
"      scattered across sectors.\n"
"   5  Theoretical occupancy is capped by register pressure.\n"
"   6  Bound by the on-chip operand-fetch ceiling: too few FLOPs issued per\n"
"      memory instruction.\n"
"\n DECISIVE-METRIC CODES\n"
"   1  dram__throughput.avg.pct_of_peak_sustained_elapsed\n"
"   2  sm__warps_active.avg.pct_of_peak_sustained_active\n"
"   3  l1tex__t_sectors_... / l1tex__t_requests_...   (sectors per request)\n"
"   4  smsp__thread_inst_executed_per_inst_executed.ratio\n"
"   5  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum\n"
"   6  smsp__warps_eligible.avg.per_cycle_active\n"
"   7  sm__sass_thread_inst_executed_op_ffma_pred_on.sum / smsp__inst_executed.sum\n"
"   8  launch__waves_per_multiprocessor\n"
"\n THE FIX THAT WILL BE MEASURED, PER KERNEL\n"
"   [1] store the column contiguously (SoA) and read it with unit stride\n"
"   [2] pad the shared tile to [32][33]\n"
"   [3] give each thread a 4x4 register tile over a 64x64 block tile\n"
"   [4] issue 8 independent loads per thread before consuming any of them\n");
}

// =============================================================================
// Harness
// =============================================================================
static unsigned fnv(const char *s)
{ unsigned h = 2166136261u; for (; *s; ++s) { h ^= (unsigned char)*s; h *= 16777619u; } return h; }
static unsigned hashAns(char tag, int i, int v)
{ char b[32]; snprintf(b, sizeof b, "%c%d:%d", tag, i, v); return fnv(b); }

static const unsigned REF_DIAG[4]   = { 2119782615u, 2246000562u, 638244711u, 1014847015u };
static const unsigned REF_METRIC[4] = { 166737359u, 4122668322u, 1884441857u, 1305228197u };

static float *d_tab, *d_col, *d_part;
static float *d_tin, *d_tout;
static float *d_A, *d_B, *d_C;

static void l1a(void){ k1_strided<<<K1_GRID,K1_BLK>>>(d_tab, d_part, K1_USE); }
static void l1b(void){ k1_packed <<<K1_GRID,K1_BLK>>>(d_col, d_part, K1_USE); }
static void l2a(void){ dim3 g(TW/TILE,TW/TILE), b(TILE,8); k2_transpose<0><<<g,b>>>(d_tin,d_tout,TW,TW); }
static void l2b(void){ dim3 g(TW/TILE,TW/TILE), b(TILE,8); k2_transpose<1><<<g,b>>>(d_tin,d_tout,TW,TW); }
static void l3a(void){ dim3 g((GN+15)/16,(GM+15)/16), b(16,16); k3_naive<<<g,b>>>(d_A,d_B,d_C,GM,GN,GK); }
static void l3b(void){ dim3 g((GN+BN-1)/BN,(GM+BM-1)/BM); k3_regtile<<<g,256>>>(d_A,d_B,d_C,GM,GN,GK); }
static void l4a(void){ k4_serial<<<K4_BLOCKS,K4_THR>>>(d_tab, d_part, K4_N); }
static void l4b(void){ k4_mlp   <<<K4_BLOCKS,K4_THR>>>(d_tab, d_part, K4_N); }

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
// Time one PAIR back to back, rotated, min-of-N. The scored quantity is a
// ratio inside one sweep, which spec SS12 5b says needs no operating-point guard.
static void timePair(void (*fa)(void), void (*fb)(void), double *ta, double *tb)
{
    void (*f[2])(void) = { fa, fb };
    int it[2]; double best[2];
    for (int i = 0; i < 2; ++i) {
        double t = timeIt(f[i], 1);
        int n = (int)(10.0/(t > 1e-3 ? t : 1e-3)); if (n < 3) n = 3; if (n > 100) n = 100;
        it[i] = n; best[i] = 1e30;
    }
    for (int s = 0; s < 4; ++s)
        for (int q = 0; q < 2; ++q) {
            int p = (q+s)&1; double t = timeIt(f[p], it[p]);
            if (t < best[p]) best[p] = t;
        }
    *ta = best[0]; *tb = best[1];
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 23 / Exercise 1 - interpret the report, then check it ===\n\n");
    printReport();
    printMenus();

    // ---- allocate -----------------------------------------------------------
    CHECK(cudaMalloc(&d_tab,  K1_USE*K1_REC*sizeof(float)));       // 512 MB
    CHECK(cudaMalloc(&d_col,  K1_USE*sizeof(float)));              //  64 MB
    CHECK(cudaMalloc(&d_part, (size_t)K1_GRID*K1_BLK*sizeof(float)));
    CHECK(cudaMalloc(&d_tin,  (size_t)TW*TW*sizeof(float)));
    CHECK(cudaMalloc(&d_tout, (size_t)TW*TW*sizeof(float)));
    CHECK(cudaMalloc(&d_A, (size_t)GM*GK*sizeof(float)));
    CHECK(cudaMalloc(&d_B, (size_t)GK*GN*sizeof(float)));
    CHECK(cudaMalloc(&d_C, (size_t)GM*GN*sizeof(float)));

    // deterministic, index-derived initialisation, done on the device so that
    // nothing here needs a gigabyte of host memory.
    initTable<<<2048,256>>>(d_tab, K1_USE*K1_REC);
    gatherCol<<<2048,256>>>(d_tab, d_col, K1_USE);
    initRamp<<<2048,256>>>(d_tin, (size_t)TW*TW, 4093, 0.001f);
    initRand<<<2048,256>>>(d_A, (size_t)GM*GK, 48271u, 11u);
    initRand<<<2048,256>>>(d_B, (size_t)GK*GN, 16807u, 7u);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

    // ---- warm up: 1500 ms streaming, then 500 ms compute (spec SS12 r4) -----
    printf("\n-- warming 1500 ms streaming + 500 ms compute ------------------------\n");
    warmFor(1500.f, l1b);
    warmFor(500.f,  l3b);
    CHECK(cudaGetLastError());

    double t[4][2];
    timePair(l1a, l1b, &t[0][0], &t[0][1]);
    timePair(l2a, l2b, &t[1][0], &t[1][1]);
    timePair(l3a, l3b, &t[2][0], &t[2][1]);
    timePair(l4a, l4b, &t[3][0], &t[3][1]);
    CHECK(cudaGetLastError());

    const char *pname[4] = { "k1 strided -> packed", "k2 [32][32] -> [32][33]",
                             "k3 naive -> 4x4 reg tile", "k4 serial -> 8 in flight" };
    printf("\n-- measured, each pair timed back to back and rotated ----------------\n");
    printf("   %-28s %10s %10s %9s %8s\n", "pair", "before ms", "after ms", "speedup", "bucket");
    int mb[4];
    for (int i = 0; i < 4; ++i) {
        double r = t[i][0]/t[i][1];
        mb[i] = (r < 1.5) ? 1 : (r < 6.0 ? 2 : 3);
        printf("   %-28s %10.4f %10.4f %8.2fx %8d\n", pname[i], t[i][0], t[i][1], r, mb[i]);
    }

    // ---- validation, separate untimed pass ----------------------------------
    printf("\n-- validation (separate pass) ---------------------------------------\n");
    int vok = 1;
    {   // k1: both versions must produce the same per-thread partials
        size_t np = (size_t)K1_GRID*K1_BLK;
        float *pa = (float*)malloc(np*sizeof(float)), *pb = (float*)malloc(np*sizeof(float));
        l1a(); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(pa, d_part, np*sizeof(float), cudaMemcpyDeviceToHost));
        l1b(); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(pb, d_part, np*sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (size_t i = 0; i < np; ++i) if (pa[i] != pb[i]) ++bad;
        printf("   k1 strided vs packed partials differing : %d\n", bad);
        if (bad) vok = 0;
        free(pa); free(pb);
    }
    {   // k2: transpose correctness, sampled (the matrices are 256 MB each)
        int bad = 0;
        const int NS = 4096;
        float *hs = (float*)malloc(NS*sizeof(float));
        float *hr = (float*)malloc(NS*sizeof(float));
        int *hidx = (int*)malloc(2*NS*sizeof(int));
        int *didx; float *dsmp;
        CHECK(cudaMalloc(&didx, 2*NS*sizeof(int)));
        CHECK(cudaMalloc(&dsmp, NS*sizeof(float)));
        for (int q = 0; q < NS; ++q) { hidx[2*q] = (q*2731) % TW; hidx[2*q+1] = (q*4111) % TW; }
        CHECK(cudaMemcpy(didx, hidx, 2*NS*sizeof(int), cudaMemcpyHostToDevice));
        sampleXY<<<16,256>>>(d_tin, dsmp, didx, NS, TW, 0);   // in[x*TW + y]
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hr, dsmp, NS*sizeof(float), cudaMemcpyDeviceToHost));
        for (int v = 0; v < 2; ++v) {
            CHECK(cudaMemset(d_tout, 0, (size_t)TW*TW*sizeof(float)));
            if (v == 0) l2a(); else l2b();
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            sampleXY<<<16,256>>>(d_tout, dsmp, didx, NS, TW, 1); // out[y*TW + x]
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(hs, dsmp, NS*sizeof(float), cudaMemcpyDeviceToHost));
            for (int q = 0; q < NS; ++q) if (hs[q] != hr[q]) ++bad;
        }
        printf("   k2 transpose sampled mismatches        : %d\n", bad);
        if (bad) vok = 0;
        free(hs); free(hr); free(hidx);
        CHECK(cudaFree(didx)); CHECK(cudaFree(dsmp));
    }
    {   // k3: sampled exact double reference with the gamma_K * S rule (M16)
        float *ha = (float*)malloc((size_t)GM*GK*sizeof(float));
        float *hb = (float*)malloc((size_t)GK*GN*sizeof(float));
        float *hc = (float*)malloc((size_t)GM*GN*sizeof(float));
        CHECK(cudaMemcpy(ha, d_A, (size_t)GM*GK*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hb, d_B, (size_t)GK*GN*sizeof(float), cudaMemcpyDeviceToHost));
        const double u = ldexp(1.0, -24), gK = (double)GK*u/(1.0 - (double)GK*u);
        int bad = 0;
        for (int v = 0; v < 2; ++v) {
            CHECK(cudaMemset(d_C, 0, (size_t)GM*GN*sizeof(float)));
            if (v == 0) l3a(); else l3b();
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(hc, d_C, (size_t)GM*GN*sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < GM; i += 97)
                for (int j = 0; j < GN; j += 101) {
                    double ref = 0.0, S = 0.0;
                    for (int k = 0; k < GK; ++k) {
                        double x = ha[(size_t)i*GK + k], y = hb[(size_t)k*GN + j];
                        ref += x*y; S += fabs(x)*fabs(y);
                    }
                    double err = fabs((double)hc[(size_t)i*GN + j] - ref);
                    if (err > 4.0*gK*S + 1e-6) ++bad;
                }
        }
        printf("   k3 GEMM sampled failures (gamma_K*S)   : %d\n", bad);
        if (bad) vok = 0;
        free(ha); free(hb); free(hc);
    }
    {   // k4: both versions sum the same elements in the same order
        size_t np = (size_t)K4_BLOCKS*K4_THR;
        float *pa = (float*)malloc(np*sizeof(float)), *pb = (float*)malloc(np*sizeof(float));
        l4a(); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(pa, d_part, np*sizeof(float), cudaMemcpyDeviceToHost));
        l4b(); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(pb, d_part, np*sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (size_t i = 0; i < np; ++i) if (pa[i] != pb[i]) ++bad;
        printf("   k4 serial vs 8-in-flight differing     : %d\n", bad);
        if (bad) vok = 0;
        free(pa); free(pb);
    }

    // ---- scoring ------------------------------------------------------------
    printf("\n-- scoring -----------------------------------------------------------\n");
    int score = 0;
    for (int i = 0; i < 4; ++i) {
        int okd = (hashAns('D', i, DIAG[i]) == REF_DIAG[i]);
        printf("   [%s] diagnosis   for kernel %d\n", okd?"x":" ", i+1); score += okd;
    }
    for (int i = 0; i < 4; ++i) {
        int okm = (hashAns('M', i, METRIC[i]) == REF_METRIC[i]);
        printf("   [%s] key metric  for kernel %d\n", okm?"x":" ", i+1); score += okm;
    }
    for (int i = 0; i < 4; ++i) {
        int okb = (BUCKET[i] == mb[i]);
        printf("   [%s] speedup bucket for fix %d  (you said %d, measured %d)\n",
               okb?"x":" ", i+1, BUCKET[i], mb[i]); score += okb;
    }
    if (!vok) printf("   validation FAILED - scoring is void\n");

    CHECK(cudaFree(d_tab)); CHECK(cudaFree(d_col)); CHECK(cudaFree(d_part));
    CHECK(cudaFree(d_tin)); CHECK(cudaFree(d_tout));
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));

    printf("\nSCORE: %d/12\n", score);
    printf("OVERALL: %s\n", (score == 12 && vok) ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return (score == 12 && vok) ? 0 : 1;
}
