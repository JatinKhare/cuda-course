// ============================================================================
// Module 15 / exercise03.cu -- Count it before you run it
//
// GOAL : Seven real transpose kernels. For each one, work out ON PAPER, for a
//        single warp executing a single instruction:
//
//          (a) how many 32 B sectors the GLOBAL READ touches   (Module 5)
//          (b) how many 32 B sectors the GLOBAL WRITE touches  (Module 5)
//          (c) the bank-conflict degree of the SHARED STORE    (Module 7)
//          (d) the bank-conflict degree of the SHARED LOAD     (Module 7)
//
//        Then implement the two counting procedures so that your own code can
//        check your arithmetic, then let the harness time all seven so you can
//        see which of the four numbers actually predicted anything.
//
//        This is Module 5 exercise 1 and Module 7 exercise 1 applied, at last,
//        to a kernel somebody would ship.
//
// THE SEVEN CONFIGURATIONS (all on a W x H fp32 matrix, W and H multiples of
// 32, so no warp is partial and no guard is ever false):
//
//   1  naive, no shared memory, coalesced read  / strided write, block (32,8)
//   2  naive, no shared memory, strided read    / coalesced write, block (32,8)
//   3  tiled, __shared__ float tile[32][32], block (32,8)
//   4  tiled, __shared__ float tile[32][33], block (32,8)
//   5  tiled, __shared__ float tile[32][34], block (32,8)
//   6  tiled, __shared__ float tile[16][17], block (16,16)
//   7  tiled, 32x32 stored with an XOR swizzle, exactly 1024 floats, block (32,8)
//
//   For 1 and 2, report 0 for (c) and (d): there is no shared memory.
//   For 6, remember which threads form warp 0 -- Module 3's linearization rule
//   is not a formality here.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN  : .\exercise03.exe
//
// Timing follows AUTHORING_SPEC section 12.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(_e), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_KERNEL()                                                         \
    do {                                                                       \
        CHECK(cudaGetLastError());                                             \
        CHECK(cudaDeviceSynchronize());                                        \
    } while (0)

static const int BIG = 8192;          // timed matrix, 8192 x 8192
static const int SMALL = 2048;        // L2-resident reveal
static const int NCFG = 7;

// ===========================================================================
// TODO 3 -- the paper table. Rows are configurations 1..7, columns are
//           { read sectors, write sectors, shared store degree, shared load
//             degree }. Every entry must be filled; leave a zero in the first
//           two columns and the program will stop.
// ===========================================================================
static const int PRED[NCFG][4] = {
    { 0, 0, 0, 0 },   // 1  naive coalesced read / strided write
    { 0, 0, 0, 0 },   // 2  naive strided read / coalesced write
    { 0, 0, 0, 0 },   // 3  tile[32][32]
    { 0, 0, 0, 0 },   // 4  tile[32][33]
    { 0, 0, 0, 0 },   // 5  tile[32][34]
    { 0, 0, 0, 0 },   // 6  tile[16][17], block (16,16)
    { 0, 0, 0, 0 },   // 7  32x32 XOR swizzle, 1024 floats
};
// YOUR CODE HERE

// ===========================================================================
// TODO 5 (PREDICTION) -- two numbers, before you build.
//
// RATIO_DRAM : at 8192 x 8192, config 3 (a real 32-way bank conflict) divided
//              by config 4 (the same kernel, padded). Buckets:
//                  1 = below 1.10x     2 = 1.10x to 2x     3 = above 2x
// RATIO_L2   : the same ratio on a 2048 x 2048 matrix, where both buffers fit
//              inside the 50 MB L2. Same buckets.
// ===========================================================================
static const int RATIO_DRAM = 0;   // YOUR CODE HERE
static const int RATIO_L2   = 0;   // YOUR CODE HERE

// ===========================================================================
// TODO 1 -- Module 5's counting procedure.
//
// `addr` holds the 32 byte addresses one warp's lanes supply for one memory
// instruction. Return the number of distinct 32 B sectors the hardware must
// move. Your implementation must depend only on the SET of addresses, never on
// their order, and each element is 4 bytes so no lane straddles a sector.
// Return -1 (shipped) and the program will stop.
// ===========================================================================
int sectorsOfWarp(const unsigned long long* addr)
{
    (void)addr;
    return -1;   // YOUR CODE HERE
}

// ===========================================================================
// TODO 2 -- Module 7's counting procedure.
//
// `off` holds the 32 shared-memory offsets, IN FLOATS (not bytes), one warp's
// lanes supply for one shared instruction. Return the conflict degree: the
// largest number of DISTINCT WORDS any single bank must supply. Two lanes
// asking a bank for the same word cost one cycle, not two -- get that wrong
// and a broadcast will look like a conflict.
// Return -1 (shipped) and the program will stop.
// ===========================================================================
int degreeOfWarp(const int* off)
{
    (void)off;
    return -1;   // YOUR CODE HERE
}

// ===========================================================================
// TODO 4 (DESIGN) -- generalise the padding rule.
//
// Return the smallest pitch P >= tileW such that a shared tile declared
// `T tile[tileW][P]`, with sizeof(T) == elemBytes, is conflict-free BOTH when a
// warp walks a row (P*r + c, c varying) and when it walks a column (P*r + c,
// r varying). tileW is a multiple of 32; elemBytes is 4, 8 or 16.
//
// The harness brute-forces the true answer for every (tileW, elemBytes) pair it
// tests, including the phase split for 8 B and 16 B elements (Module 7), and
// tells you only whether you agree. Derive the condition; do not search.
// Return 0 (shipped) and this TODO scores zero, but the rest still runs.
// ===========================================================================
int padPitch(int tileW, int elemBytes)
{
    (void)tileW; (void)elemBytes;
    return 0;   // YOUR CODE HERE
}

// ---------------------------------------------------------------------------
// The seven kernels. Given, complete, and correct -- your job is to predict
// them, not to write them.
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ float srcValue(long long i)
{
    unsigned h = (unsigned)i * 2654435761u;
    h ^= h >> 13;
    h *= 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0x00FFFFFFu);
}

__global__ void k1(const float* __restrict__ in, float* __restrict__ out, int W, int H)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        out[(long long)x * H + (y + j)] = in[(long long)(y + j) * W + x];
}

__global__ void k2(const float* __restrict__ in, float* __restrict__ out, int W, int H)
{
    int xo = blockIdx.x * 32 + threadIdx.x;      // column of out
    int yo = blockIdx.y * 32 + threadIdx.y;      // row of out
    for (int j = 0; j < 32; j += 8)
        out[(long long)(yo + j) * H + xo] = in[(long long)xo * W + (yo + j)];
}

template <int PITCH>
__global__ void k345(const float* __restrict__ in, float* __restrict__ out, int W, int H)
{
    __shared__ float tile[32][PITCH];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * W + x];
    __syncthreads();
    int xo = blockIdx.y * 32 + threadIdx.x;
    int yo = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        out[(long long)(yo + j) * H + xo] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void k6(const float* __restrict__ in, float* __restrict__ out, int W, int H)
{
    __shared__ float tile[16][17];
    int x = blockIdx.x * 16 + threadIdx.x;
    int y = blockIdx.y * 16 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[(long long)y * W + x];
    __syncthreads();
    int xo = blockIdx.y * 16 + threadIdx.x;
    int yo = blockIdx.x * 16 + threadIdx.y;
    out[(long long)yo * H + xo] = tile[threadIdx.x][threadIdx.y];
}

__device__ __host__ __forceinline__ int swz(int r, int c) { return r * 32 + (c ^ (r & 31)); }

__global__ void k7(const float* __restrict__ in, float* __restrict__ out, int W, int H)
{
    __shared__ float tile[32 * 32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        tile[swz(threadIdx.y + j, threadIdx.x)] = in[(long long)(y + j) * W + x];
    __syncthreads();
    int xo = blockIdx.y * 32 + threadIdx.x;
    int yo = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8)
        out[(long long)(yo + j) * H + xo] = tile[swz(threadIdx.x, threadIdx.y + j)];
}

__global__ void copyCeiling(const float4* __restrict__ in, float4* __restrict__ out,
                            long long n4)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    for (; i < n4; i += s) out[i] = in[i];
}

__global__ void fillSource(float* a, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    for (; i < n; i += s) a[i] = srcValue(i);
}
__global__ void checkTranspose(const float* __restrict__ out, int W, int H, unsigned* bad)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)blockDim.x * gridDim.x;
    long long n = (long long)W * H;
    unsigned local = 0;
    for (; i < n; i += s) {
        int c = (int)(i % H), r = (int)(i / H);
        if (out[i] != srcValue((long long)c * W + r)) ++local;
    }
    if (local) atomicAdd(bad, local);
}

static void launchCfg(int c, const float* in, float* out, int W, int H)
{
    dim3 g32((W + 31) / 32, (H + 31) / 32), b32(32, 8);
    dim3 g32o((H + 31) / 32, (W + 31) / 32);
    dim3 g16((W + 15) / 16, (H + 15) / 16), b16(16, 16);
    switch (c) {
        case 0: k1     <<<g32,  b32>>>(in, out, W, H); break;
        case 1: k2     <<<g32o, b32>>>(in, out, W, H); break;
        case 2: k345<32><<<g32, b32>>>(in, out, W, H); break;
        case 3: k345<33><<<g32, b32>>>(in, out, W, H); break;
        case 4: k345<34><<<g32, b32>>>(in, out, W, H); break;
        case 5: k6     <<<g16,  b16>>>(in, out, W, H); break;
        case 6: k7     <<<g32,  b32>>>(in, out, W, H); break;
        default: break;
    }
}

// ---------------------------------------------------------------------------
// Address generation. For each configuration, the harness reproduces exactly
// what warp 0 of block (2,2) supplies on the first iteration of each loop.
// Nothing here is a hint: it is the same arithmetic as the kernels above.
// ---------------------------------------------------------------------------
static void addressesFor(int cfg, int W, int H,
                         unsigned long long* rd, unsigned long long* wr,
                         int* shSt, int* shLd, bool* hasShared)
{
    const int bx = 2, by = 2;
    *hasShared = (cfg >= 2);
    if (cfg == 5) {                             // tile 16, block (16,16)
        for (int lane = 0; lane < 32; ++lane) {
            int tx = lane % 16, ty = lane / 16;
            int x = bx * 16 + tx, y = by * 16 + ty;
            rd[lane] = ((unsigned long long)y * W + x) * 4ull;
            int xo = by * 16 + tx, yo = bx * 16 + ty;
            wr[lane] = ((unsigned long long)yo * H + xo) * 4ull;
            shSt[lane] = ty * 17 + tx;
            shLd[lane] = tx * 17 + ty;
        }
        return;
    }
    const int pitch = (cfg == 2) ? 32 : (cfg == 3) ? 33 : (cfg == 4) ? 34 : 0;
    for (int lane = 0; lane < 32; ++lane) {
        int tx = lane, ty = 0;
        if (cfg == 1) {                          // indexed from the output side
            int xo = bx * 32 + tx, yo = by * 32 + ty;
            rd[lane] = ((unsigned long long)xo * W + yo) * 4ull;
            wr[lane] = ((unsigned long long)yo * H + xo) * 4ull;
            shSt[lane] = 0; shLd[lane] = 0;
            continue;
        }
        int x = bx * 32 + tx, y = by * 32 + ty;
        rd[lane] = ((unsigned long long)y * W + x) * 4ull;
        if (cfg == 0) {
            wr[lane] = ((unsigned long long)x * H + y) * 4ull;
            shSt[lane] = 0; shLd[lane] = 0;
            continue;
        }
        int xo = by * 32 + tx, yo = bx * 32 + ty;
        wr[lane] = ((unsigned long long)yo * H + xo) * 4ull;
        if (cfg == 6) { shSt[lane] = swz(ty, tx); shLd[lane] = swz(tx, ty); }
        else          { shSt[lane] = ty * pitch + tx; shLd[lane] = tx * pitch + ty; }
    }
}

// Brute-force truth for TODO 4, including the phase split.
static bool phaseFreeFor(int tileW, int pitch, int elemBytes)
{
    int phases = elemBytes / 4; if (phases < 1) phases = 1;
    int lanesPerPhase = 32 / phases;
    for (int dir = 0; dir < 2; ++dir) {          // 0 = row walk, 1 = column walk
        for (int fixed = 0; fixed < tileW; ++fixed) {
            for (int ph = 0; ph < phases; ++ph) {
                int words[32][8], nw[32];
                for (int b = 0; b < 32; ++b) nw[b] = 0;
                for (int L = 0; L < lanesPerPhase; ++L) {
                    int lane = ph * lanesPerPhase + L;
                    int r = dir ? lane : fixed;
                    int c = dir ? fixed : lane;
                    if (r >= tileW || c >= tileW) continue;
                    long long base = ((long long)r * pitch + c) * elemBytes;
                    for (int k = 0; k < elemBytes; k += 4) {
                        long long w = (base + k) / 4;
                        int bank = (int)(w % 32);
                        bool seen = false;
                        for (int q = 0; q < nw[bank]; ++q)
                            if (words[bank][q] == (int)(w & 0x7fffffff)) { seen = true; break; }
                        if (!seen) {
                            if (nw[bank] >= 8) return false;
                            words[bank][nw[bank]++] = (int)(w & 0x7fffffff);
                        }
                    }
                }
                for (int b = 0; b < 32; ++b) if (nw[b] > 1) return false;
            }
        }
    }
    return true;
}

// Reading a compile-time constant through a volatile pointer stops the
// compiler from folding the "TODO not set" tests away and then warning that
// the whole program is unreachable.
static int peek(const int* p) { const volatile int* q = p; return *q; }

static unsigned fnv(const int* v, int n)
{
    unsigned h = 2166136261u;
    for (int i = 0; i < n; ++i) {
        unsigned x = (unsigned)v[i];
        for (int b = 0; b < 4; ++b) { h ^= (x >> (8 * b)) & 0xFFu; h *= 16777619u; }
    }
    return h;
}
// FNV-1a of the 28 correct answers, row-major. The answers themselves are not
// in this file.
static const unsigned TRUTH_HASH = 0xb51c9c67u;

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    {
        unsigned long long probe[32];
        for (int i = 0; i < 32; ++i) probe[i] = (unsigned long long)i * 4ull;
        int s1 = sectorsOfWarp(probe);
        if (peek(&s1) < 0) { printf("Set TODO 1 first.\n"); return 0; }
        int p2[32];
        for (int i = 0; i < 32; ++i) p2[i] = i;
        int d1 = degreeOfWarp(p2);
        if (peek(&d1) < 0) { printf("Set TODO 2 first.\n"); return 0; }
    }
    for (int r = 0; r < NCFG; ++r)
        if (peek(&PRED[r][0]) == 0 || peek(&PRED[r][1]) == 0)
            { printf("Set TODO 3 first.\n"); return 0; }
    if (peek(&RATIO_DRAM) == 0 || peek(&RATIO_L2) == 0)
        { printf("Set TODO 5 first.\n"); return 0; }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Module 15 exercise 03 -- count it before you run it\n");
    printf("GPU: %s, CC %d.%d, L2 = %.1f MB\n\n",
           prop.name, prop.major, prop.minor, (double)prop.l2CacheSize / 1.0e6);

    // ---- structural tests of TODO 1 and TODO 2 ----------------------------
    int structOk = 0, structTot = 9;
    {
        unsigned long long a[32]; int o[32];
        for (int i = 0; i < 32; ++i) a[i] = 4096ull;                 // broadcast
        if (sectorsOfWarp(a) == 1) ++structOk; else printf("  TODO1 fails: broadcast\n");
        for (int i = 0; i < 32; ++i) a[i] = 4096ull + 4ull * i;      // contiguous
        if (sectorsOfWarp(a) == 4) ++structOk; else printf("  TODO1 fails: contiguous\n");
        for (int i = 0; i < 32; ++i) a[i] = 4096ull + 4ull * (31 - i);  // reversed
        if (sectorsOfWarp(a) == 4) ++structOk; else printf("  TODO1 fails: order dependence\n");
        for (int i = 0; i < 32; ++i) a[i] = 4096ull + 32ull * i;      // stride 8 floats
        if (sectorsOfWarp(a) == 32) ++structOk; else printf("  TODO1 fails: stride 8\n");
        for (int i = 0; i < 32; ++i) a[i] = 4096ull + 8ull * i;       // stride 2 floats
        if (sectorsOfWarp(a) == 8) ++structOk; else printf("  TODO1 fails: stride 2\n");

        for (int i = 0; i < 32; ++i) o[i] = i;
        if (degreeOfWarp(o) == 1) ++structOk; else printf("  TODO2 fails: unit stride\n");
        for (int i = 0; i < 32; ++i) o[i] = 0;                        // broadcast
        if (degreeOfWarp(o) == 1) ++structOk; else printf("  TODO2 fails: broadcast\n");
        for (int i = 0; i < 32; ++i) o[i] = i / 2;                    // broadcast pairs
        if (degreeOfWarp(o) == 1) ++structOk; else printf("  TODO2 fails: broadcast pairs\n");
        for (int i = 0; i < 32; ++i) o[i] = 32 * i;
        if (degreeOfWarp(o) == 32) ++structOk; else printf("  TODO2 fails: 32-way\n");
    }
    printf("Structural tests of your two counting procedures: %d/%d\n\n",
           structOk, structTot);

    // ---- the table your procedures produce --------------------------------
    int computed[NCFG][4];
    for (int c = 0; c < NCFG; ++c) {
        unsigned long long rd[32], wr[32];
        int st[32], ld[32];
        bool hasSh = false;
        addressesFor(c, BIG, BIG, rd, wr, st, ld, &hasSh);
        computed[c][0] = sectorsOfWarp(rd);
        computed[c][1] = sectorsOfWarp(wr);
        computed[c][2] = hasSh ? degreeOfWarp(st) : 0;
        computed[c][3] = hasSh ? degreeOfWarp(ld) : 0;
    }

    const char* cname[NCFG] = {
        "1 naive coal rd / strided wr", "2 naive strided rd / coal wr",
        "3 tile[32][32]              ", "4 tile[32][33]              ",
        "5 tile[32][34]              ", "6 tile[16][17], block(16,16)",
        "7 32x32 XOR swizzle         " };

    printf("%-30s %14s %14s\n", "configuration", "your table", "your procedures");
    printf("%-30s %14s %14s\n", "", "rd wr st ld", "rd wr st ld");
    int rowsMatching = 0;
    for (int c = 0; c < NCFG; ++c) {
        bool same = true;
        for (int k = 0; k < 4; ++k) if (PRED[c][k] != computed[c][k]) same = false;
        if (same) ++rowsMatching;
        printf("%-30s   %2d %2d %2d %2d    %2d %2d %2d %2d   %s\n", cname[c],
               PRED[c][0], PRED[c][1], PRED[c][2], PRED[c][3],
               computed[c][0], computed[c][1], computed[c][2], computed[c][3],
               same ? "" : "<- differ");
    }
    unsigned hPred = fnv(&PRED[0][0], NCFG * 4);
    unsigned hComp = fnv(&computed[0][0], NCFG * 4);
    printf("\n  your hand table  : %s\n", (hPred == TRUTH_HASH) ? "CORRECT" : "wrong");
    printf("  your procedures  : %s\n\n", (hComp == TRUTH_HASH) ? "CORRECT" : "wrong");

    // ---- TODO 4 -----------------------------------------------------------
    int padOk = 0, padTot = 0;
    {
        const int tws[4] = { 32, 64, 96, 128 };
        const int ebs[3] = { 4, 8, 16 };
        printf("TODO 4 -- smallest conflict-free pitch\n");
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 3; ++j) {
                int truth = 0;
                for (int P = tws[i]; P < tws[i] + 64; ++P)
                    if (phaseFreeFor(tws[i], P, ebs[j])) { truth = P; break; }
                int mine = padPitch(tws[i], ebs[j]);
                ++padTot;
                if (mine == truth) ++padOk;
                printf("  tileW %3d, %2d B elements : you say %4d  %s\n",
                       tws[i], ebs[j], mine, (mine == truth) ? "ok" : "no");
            }
        printf("  TODO 4 score: %d/%d\n\n", padOk, padTot);
    }

    // ---- measurement ------------------------------------------------------
    const long long nBig = (long long)BIG * BIG;
    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  (size_t)nBig * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)nBig * sizeof(float)));
    unsigned* d_bad = nullptr;
    CHECK(cudaMalloc(&d_bad, sizeof(unsigned)));
    fillSource<<<2048, 256>>>(d_in, nBig);
    CHECK_KERNEL();

    cudaEvent_t evA, evB;
    CHECK(cudaEventCreate(&evA));
    CHECK(cudaEventCreate(&evB));
    {
        float acc = 0.0f;
        CHECK(cudaEventRecord(evA));
        while (acc < 1500.0f) {
            for (int k = 0; k < 10; ++k)
                copyCeiling<<<8192, 256>>>(reinterpret_cast<const float4*>(d_in),
                                           reinterpret_cast<float4*>(d_out), nBig / 4);
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            CHECK(cudaEventElapsedTime(&acc, evA, evB));
        }
        CHECK_KERNEL();
    }

    const int NT = NCFG + 1;       // 7 configurations + the copy ceiling
    double best[NCFG + 1];
    for (int i = 0; i < NT; ++i) best[i] = 1e30;
    for (int sweep = 0; sweep < NT; ++sweep)
        for (int q = 0; q < NT; ++q) {
            int v = (q + sweep) % NT;
            CHECK(cudaEventRecord(evA));
            for (int k = 0; k < 20; ++k) {
                if (v == NCFG) copyCeiling<<<8192, 256>>>(
                                   reinterpret_cast<const float4*>(d_in),
                                   reinterpret_cast<float4*>(d_out), nBig / 4);
                else launchCfg(v, d_in, d_out, BIG, BIG);
            }
            CHECK(cudaEventRecord(evB));
            CHECK(cudaEventSynchronize(evB));
            float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evA, evB));
            if (ms / 20.0 < best[v]) best[v] = ms / 20.0;
        }
    CHECK_KERNEL();

    double b2 = 2.0 * (double)nBig * 4.0;
    printf("MEASURED at %d x %d (min of %d rotated sweeps)\n", BIG, BIG, NT);
    printf("%-30s %9s %9s %9s\n", "configuration", "ms", "GB/s", "%ofcopy");
    printf("%-30s %9.4f %9.1f %8.1f%%\n", "0 copy ceiling                ",
           best[NCFG], b2 / (best[NCFG] * 1e-3) / 1e9, 100.0);
    for (int c = 0; c < NCFG; ++c)
        printf("%-30s %9.4f %9.1f %8.1f%%\n", cname[c], best[c],
               b2 / (best[c] * 1e-3) / 1e9, 100.0 * best[NCFG] / best[c]);
    printf("\n");

    // ---- validation -------------------------------------------------------
    int nWrong = 0;
    for (int c = 0; c < NCFG; ++c) {
        unsigned zero = 0, bad = 0;
        launchCfg(c, d_in, d_out, BIG, BIG);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(d_bad, &zero, sizeof(unsigned), cudaMemcpyHostToDevice));
        checkTranspose<<<2048, 256>>>(d_out, BIG, BIG, d_bad);
        CHECK_KERNEL();
        CHECK(cudaMemcpy(&bad, d_bad, sizeof(unsigned), cudaMemcpyDeviceToHost));
        if (bad) { ++nWrong; printf("  config %d produced %u mismatches\n", c + 1, bad); }
    }
    printf("VALIDATION: all seven kernels %s\n\n", nWrong ? "FAILED" : "correct");

    // ---- the L2-resident repeat -------------------------------------------
    double s3 = 1e30, s4 = 1e30;
    {
        const long long nS = (long long)SMALL * SMALL;
        fillSource<<<1024, 256>>>(d_in, nS);
        CHECK_KERNEL();
        for (int k = 0; k < 40; ++k) launchCfg(2, d_in, d_out, SMALL, SMALL);
        CHECK_KERNEL();
        for (int sweep = 0; sweep < 2; ++sweep)
            for (int q = 0; q < 2; ++q) {
                int v = 2 + ((q + sweep) % 2);
                CHECK(cudaEventRecord(evA));
                for (int k = 0; k < 100; ++k) launchCfg(v, d_in, d_out, SMALL, SMALL);
                CHECK(cudaEventRecord(evB));
                CHECK(cudaEventSynchronize(evB));
                float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, evA, evB));
                if (v == 2) { if (ms / 100.0 < s3) s3 = ms / 100.0; }
                else        { if (ms / 100.0 < s4) s4 = ms / 100.0; }
            }
    }

    double rD = best[2] / best[3];
    double rL = s3 / s4;
    int bD = (rD < 1.10) ? 1 : (rD <= 2.0 ? 2 : 3);
    int bL = (rL < 1.10) ? 1 : (rL <= 2.0 ? 2 : 3);
    printf("CONFIG 3 / CONFIG 4 -- what removing a 32-way conflict is worth\n");
    printf("  at %dx%d (DRAM bound)    : %.3fx  -> bucket %d\n", BIG, BIG, rD, bD);
    printf("  at %dx%d (L2 resident)   : %.3fx  -> bucket %d\n", SMALL, SMALL, rL, bL);
    printf("\n");

    // ---- scoring ----------------------------------------------------------
    int score = 0, maxScore = 6;
    printf("SCORING\n");
    if (structOk == structTot) { ++score; printf("  structural tests            : 1/1\n"); }
    else printf("  structural tests            : 0/1 (%d/%d)\n", structOk, structTot);
    if (hComp == TRUTH_HASH) { ++score; printf("  TODO 1 + TODO 2 procedures  : 1/1\n"); }
    else printf("  TODO 1 + TODO 2 procedures  : 0/1\n");
    if (hPred == TRUTH_HASH) { ++score; printf("  TODO 3 hand table (28 cells): 1/1 "
                                               "(%d/%d rows agree with your code)\n",
                                               rowsMatching, NCFG); }
    else printf("  TODO 3 hand table (28 cells): 0/1 (%d/%d rows agree with your code)\n",
                rowsMatching, NCFG);
    if (padOk == padTot) { ++score; printf("  TODO 4 padding rule         : 1/1\n"); }
    else printf("  TODO 4 padding rule         : 0/1\n");
    if (RATIO_DRAM == bD) { ++score; printf("  TODO 5 DRAM ratio bucket    : 1/1\n"); }
    else printf("  TODO 5 DRAM ratio bucket    : 0/1 (you said %d, measured %d)\n",
                RATIO_DRAM, bD);
    if (RATIO_L2 == bL) { ++score; printf("  TODO 5 L2 ratio bucket      : 1/1\n"); }
    else printf("  TODO 5 L2 ratio bucket      : 0/1 (you said %d, measured %d)\n",
                RATIO_L2, bL);
    printf("\nSCORE: %d/%d\n", score, maxScore);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_bad));
    CHECK(cudaEventDestroy(evA));
    CHECK(cudaEventDestroy(evB));
    CHECK(cudaDeviceReset());

    printf("OVERALL: %s\n", (score == maxScore && nWrong == 0) ? "PASS" : "FAIL");
    return (score == maxScore && nWrong == 0) ? 0 : 1;
}
