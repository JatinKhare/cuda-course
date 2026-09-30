// =============================================================================
// Module 17 / Exercise 2 — tile-shape design, on paper, then measured.
//
// GOAL : Choose a tile shape from a model you wrote yourself, before any
//        timing happens, and find out whether your model was right. Along the
//        way, work out the bank-conflict degree of every shared access a tiled
//        GEMM performs and decide -- with a number, not with folklore --
//        whether the tile should be padded.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
// RUN  : exercise02.exe
//
// TODO 1 - smemBytesPerBlock(BM,BN,BK)                            [host]
// TODO 2 - blocksPerSM(threads, smemBytes)                        [host]
// TODO 3 - fmasPerGlobalLoad(BM,BN,BK) and sharedLoadsPerFma()    [host]
// TODO 4 - tileWordIndex(...) and conflictDegree(...)             [host]
// TODO 5 - chooseTile() and PAD_PREDICTION       [host, DESIGN TODO]
//
// TODOs 1-4 are checked against FNV-1a hashes of the reference answers, so
// the answers are not in this file. TODO 5 is checked against a measurement.
//
// You are NOT writing a kernel here. The kernel is the one you wrote in
// Exercise 1, generalised to a BM x BN output tile with contraction depth BK
// and a shared-memory row pitch of (BK + PAD) / (BN + PAD).
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

#define M_DIM 1035
#define N_DIM 1541
#define K_DIM 1063

// Machine facts you may use (all established in earlier modules):
//   40 SMs, 1536 threads/SM, 24 blocks/SM, 32 banks x 4 B
//   102400 B of shared memory addressable per SM
//   the driver charges every resident block an extra 1024 B   (Module 6)
//   shared allocations are rounded up to a 128 B granularity  (Module 7)
//   a shared access of degree D costs max(2, D)               (Module 7)
#define SMEM_PER_SM     102400
#define THREADS_PER_SM   1536
#define MAX_BLOCKS_SM      24
#define DRIVER_RESERVE   1024
#define SMEM_GRANULARITY  128

// =============================================================================
// TODO 1 — shared memory per block.
//
// The kernel stages a BM x BK tile of A and a BK x BN tile of B, both fp32,
// with a row pitch of (BK + pad) and (BN + pad) floats respectively. Return
// the number of BYTES the two __shared__ arrays occupy. Return 0 to signal
// "not written yet".
// =============================================================================
static int smemBytesPerBlock(int BM, int BN, int BK, int pad)
{
    (void)BM; (void)BN; (void)BK; (void)pad;
    // YOUR CODE HERE
    return 0;
}

// =============================================================================
// TODO 2 — resident blocks per SM.
//
// Three limiters, and the answer is the smallest of them:
//   - threads: an SM holds at most THREADS_PER_SM threads;
//   - shared memory: an SM has SMEM_PER_SM bytes addressable as shared, but
//     what a block COSTS is not what it asked for -- see DRIVER_RESERVE and
//     SMEM_GRANULARITY above, and get the order of the two operations right;
//   - a hard cap of MAX_BLOCKS_SM blocks.
//
// Module 6 measured this table for 256-thread blocks, which is a useful check:
//     0 B -> 6 blocks, 8192 -> 6, 12288 -> 6, 16384 -> 5, 25600 -> 3,
//     49152 -> 2. Note that 102400/16384 = 6.25, and the answer is 5.
//
// Return 0 to signal "not written yet".
// =============================================================================
static int blocksPerSM(int threads, int smemBytes)
{
    (void)threads; (void)smemBytes;
    // YOUR CODE HERE
    return 0;
}

// =============================================================================
// TODO 3 — the two instruction-mix ratios.
//
// (a) fmasPerGlobalLoad: over one k-step, a block issues some number of global
//     load instructions to fill its two tiles and performs some number of
//     fused multiply-adds. Return the ratio. Work it out from the tile
//     dimensions; do not guess. One of BM, BN, BK does not appear in the
//     answer, and noticing which one is most of the value of this TODO.
//
// (b) sharedLoadsPerFma: in the accumulation loop, how many shared-memory
//     loads does each fused multiply-add require? This one does not depend on
//     the tile shape at all, which is the entire reason Module 18 exists.
//
// Return a negative value to signal "not written yet".
// =============================================================================
static double fmasPerGlobalLoad(int BM, int BN, int BK)
{
    (void)BM; (void)BN; (void)BK;
    // YOUR CODE HERE
    return -1.0;
}
static double sharedLoadsPerFma(void)
{
    // YOUR CODE HERE
    return -1.0;
}

// =============================================================================
// TODO 4 — bank conflicts in the tile.
//
// (a) tileWordIndex(pattern, T, pad, lane, k)
//
//     Warp 0 of a (T, T) block. Module 3's linearisation rule gives
//     lane -> (tx, ty). Return the index, IN 4-BYTE WORDS FROM THE BASE OF THE
//     ARRAY IN QUESTION, that this lane addresses, for four access patterns:
//
//       pattern 0 : the accumulation loop's read of a row-major A tile,
//                   As[ty][k], where As has row pitch (T + pad)
//       pattern 1 : the accumulation loop's read of the B tile, Bs[k][tx],
//                   row pitch (T + pad)
//       pattern 2 : the cooperative store into a row-major A tile, As[ty][tx]
//       pattern 3 : the cooperative store into a TRANSPOSED A tile, As[tx][ty]
//                   -- the layout that lets a thread later read a column of A
//                   with one instruction, which is what Module 18 will want.
//
//     Return -1 for an unknown pattern.
//
// (b) conflictDegree(word, n)
//
//     Given the n word indices a warp's lanes address, return the conflict
//     degree: the maximum, over the 32 banks, of the number of DISTINCT WORDS
//     that bank must supply. Two lanes asking for the same word are merged and
//     broadcast, and cost nothing extra (Module 7). Return 0 if unwritten.
// =============================================================================
static int tileWordIndex(int pattern, int T, int pad, int lane, int k)
{
    (void)pattern; (void)T; (void)pad; (void)lane; (void)k;
    // YOUR CODE HERE
    return -1;
}
static int conflictDegree(const int *word, int n)
{
    (void)word; (void)n;
    // YOUR CODE HERE
    return 0;
}

// =============================================================================
// TODO 5 — DESIGN. Two commitments, both made before any timing happens.
//
// (a) chooseTile(): pick the tile shape you expect to be fastest on the
//     1035 x 1541 x 1063 problem, using YOUR functions above and nothing else.
//     It must be one of the eight shapes the harness compiles:
//
//        (BM,BN,BK,PAD) = ( 8, 8, 8,0) (16,16, 8,0) (16,16,16,0) (16,16,32,0)
//                         (32,16,16,0) (16,32,16,0) (32,32,32,0) (16,16,16,1)
//
//     Print your reasoning somewhere you will still have it after you see the
//     answer. The three quantities you have are FMAs per global load, blocks
//     per SM, and the bank-conflict degrees. At least one of them will point
//     the wrong way; deciding which one to believe is the exercise.
//
// (b) PAD_PREDICTION: the harness times (16,16,16,0) against (16,16,16,1) --
//     the same tile with one extra float of row pitch. Predict the ratio
//     time(pad 0) / time(pad 1):
//        1 = padding is a clear WIN         (ratio > 1.10)
//        2 = padding changes nothing        (0.95 <= ratio <= 1.10)
//        3 = padding is a clear LOSS        (ratio < 0.95)
//     Your TODO 4 answers tell you what padding does to the bank conflicts.
//     Whether that is the whole story is a separate question, and the cheapest
//     way to find out is:
//        nvcc -arch=sm_89 -O3 -cubin -o e2.cubin exercise02.cu
//        cuobjdump -sass e2.cubin
//     and count the shared-memory instructions in the accumulation loop of the
//     two instantiations. Do that BEFORE you commit to a bucket.
// =============================================================================
static void chooseTile(int *BM, int *BN, int *BK, int *PAD)
{
    *BM = 0; *BN = 0; *BK = 0; *PAD = 0;     // <- 0 means "not chosen yet"
    // YOUR CODE HERE
}
#define PAD_PREDICTION 0        // <- 1, 2 or 3, YOUR ANSWER HERE

// =============================================================================
// The kernel. This is Exercise 1's, generalised. You are not editing it.
// =============================================================================
template <int BM, int BN, int BK, int PAD>
__global__ void gemmTiled(int M, int N, int K, const float * __restrict__ A,
                          const float * __restrict__ B, float *C)
{
    __shared__ float As[BM][BK + PAD];
    __shared__ float Bs[BK][BN + PAD];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BN + tx, nthr = BM * BN;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int row = row0 + ty, col = col0 + tx;
    float acc = 0.0f;
    const int nTiles = (K + BK - 1) / BK;
    for (int t = 0; t < nTiles; ++t) {
        const int k0 = t * BK;
        #pragma unroll
        for (int i = tid; i < BM * BK; i += nthr) {
            const int r = i / BK, c = i - r * BK;
            const int gr = row0 + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
        }
        #pragma unroll
        for (int i = tid; i < BK * BN; i += nthr) {
            const int r = i / BN, c = i - r * BN;
            const int gr = k0 + r, gc = col0 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) acc = fmaf(As[ty][k], Bs[k][tx], acc);
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// warm-up kernels (spec 12 rule 4)
__global__ void streamWarm(const float4 *in, float4 *out, size_t n4)
{
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    const size_t st = (size_t)gridDim.x*blockDim.x;
    float4 s0 = make_float4(0,0,0,0);
    for (; i < n4; i += st) { float4 v = in[i];
        s0.x+=v.x; s0.y+=v.y; s0.z+=v.z; s0.w+=v.w; }
    if (s0.x+s0.y+s0.z+s0.w == 1.2345e30f) out[0] = s0;
}
__global__ void ffmaWarm(float *out, int iters)
{
    float a0=threadIdx.x,a1=a0+1,a2=a0+2,a3=a0+3,a4=a0+4,a5=a0+5,a6=a0+6,a7=a0+7;
    const float b=1.0000001f, c=0.9999999f;
    for (int i=0;i<iters;++i){ a0=fmaf(a0,b,c);a1=fmaf(a1,b,c);a2=fmaf(a2,b,c);
        a3=fmaf(a3,b,c);a4=fmaf(a4,b,c);a5=fmaf(a5,b,c);a6=fmaf(a6,b,c);a7=fmaf(a7,b,c);}
    float s=a0+a1+a2+a3+a4+a5+a6+a7;
    if (s==1.2345e30f) out[0]=s;
}

// =============================================================================
// harness
// =============================================================================
typedef struct { int BM,BN,BK,PAD; const char *name; } Shape;
static const Shape SH[] = {
    {  8,  8,  8, 0, " 8x 8 BK= 8 pad0" },
    { 16, 16,  8, 0, "16x16 BK= 8 pad0" },
    { 16, 16, 16, 0, "16x16 BK=16 pad0" },
    { 16, 16, 32, 0, "16x16 BK=32 pad0" },
    { 32, 16, 16, 0, "32x16 BK=16 pad0" },
    { 16, 32, 16, 0, "16x32 BK=16 pad0" },
    { 32, 32, 32, 0, "32x32 BK=32 pad0" },
    { 16, 16, 16, 1, "16x16 BK=16 pad1" },
};
enum { NSH = (int)(sizeof(SH)/sizeof(SH[0])) };

typedef struct { int M,N,K; const float *dA,*dB; float *dC; } Ctx;
static Ctx g_ctx;

static void launchShape(int s, Ctx *c)
{
#define LA(bm,bn,bk,pd) do { dim3 bl((bn),(bm));                                \
        dim3 gr((unsigned)((c->N+(bn)-1)/(bn)),(unsigned)((c->M+(bm)-1)/(bm))); \
        gemmTiled<bm,bn,bk,pd><<<gr,bl>>>(c->M,c->N,c->K,c->dA,c->dB,c->dC);    \
    } while (0)
    switch (s) {
      case 0: LA( 8, 8, 8,0); break;
      case 1: LA(16,16, 8,0); break;
      case 2: LA(16,16,16,0); break;
      case 3: LA(16,16,32,0); break;
      case 4: LA(32,16,16,0); break;
      case 5: LA(16,32,16,0); break;
      case 6: LA(32,32,32,0); break;
      default:LA(16,16,16,1); break;
    }
#undef LA
}
static double timeOnce(int s, Ctx *c, int it)
{
    cudaEvent_t e0,e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0));
    for (int i=0;i<it;++i) launchShape(s,c);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms,e0,e1));
    CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    return ms/it;
}
static int calibrate(int s, Ctx *c)
{
    double ms = timeOnce(s,c,1);
    int it=(int)(10.0/(ms>0.0?ms:0.01)); if(it<1)it=1; if(it>128)it=128; return it;
}

// FNV-1a over a byte stream -- the reference answers are not in this file.
static unsigned fnv(const void *p, size_t n, unsigned h)
{
    const unsigned char *b=(const unsigned char*)p;
    for (size_t i=0;i<n;++i){ h ^= b[i]; h *= 16777619u; }
    return h;
}

// occupancy cross-check: the API knows every limiter
static int apiBlocksPerSM(int BM,int BN,int BK,int PAD)
{
    int b=0; const int thr=BM*BN;
    const void *f=NULL;
    if (BM==8&&BN==8)                 f=(const void*)gemmTiled< 8, 8, 8,0>;
    else if (BM==16&&BN==16&&BK==8)   f=(const void*)gemmTiled<16,16, 8,0>;
    else if (BM==16&&BN==16&&BK==16&&PAD==0) f=(const void*)gemmTiled<16,16,16,0>;
    else if (BM==16&&BN==16&&BK==32)  f=(const void*)gemmTiled<16,16,32,0>;
    else if (BM==32&&BN==16)          f=(const void*)gemmTiled<32,16,16,0>;
    else if (BM==16&&BN==32)          f=(const void*)gemmTiled<16,32,16,0>;
    else if (BM==32&&BN==32)          f=(const void*)gemmTiled<32,32,32,0>;
    else                              f=(const void*)gemmTiled<16,16,16,1>;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b,f,thr,0));
    return b;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const int M=M_DIM, N=N_DIM, K=K_DIM;
    const double flops = 2.0*(double)M*N*K;

    printf("=== Module 17 / Exercise 2 - tile-shape design ===\n");
    printf("C(%d x %d) = A(%d x %d) * B(%d x %d), row-major fp32\n\n", M,N,M,K,K,N);

    // ---- gates ------------------------------------------------------------
    if (smemBytesPerBlock(16,16,16,0) <= 0) { printf("Set TODO 1 first.\n"); return 0; }
    if (blocksPerSM(256, 2048) <= 0)        { printf("Set TODO 2 first.\n"); return 0; }
    if (fmasPerGlobalLoad(16,16,16) < 0.0 || sharedLoadsPerFma() < 0.0)
                                            { printf("Set TODO 3 first.\n"); return 0; }
    if (tileWordIndex(0,16,0,0,0) < 0)      { printf("Set TODO 4a first.\n"); return 0; }
    { int w[32]; for (int i=0;i<32;++i) w[i]=i;
      if (conflictDegree(w,32) <= 0)        { printf("Set TODO 4b first.\n"); return 0; } }
    int cBM,cBN,cBK,cPAD; chooseTile(&cBM,&cBN,&cBK,&cPAD);
    if (cBM <= 0 || cBN <= 0 || cBK <= 0)   { printf("Set TODO 5a first.\n"); return 0; }
    if (PAD_PREDICTION < 1 || PAD_PREDICTION > 3)
                                            { printf("Set TODO 5b first.\n"); return 0; }

    int score = 0;
    const int MAXSCORE = 6;

    // ---- TODO 1 -----------------------------------------------------------
    printf("-- TODO 1: shared bytes per block --------------------------------\n");
    { unsigned h = 2166136261u;
      printf("  %-18s %10s\n", "shape", "bytes");
      for (int i=0;i<NSH;++i) {
          int v = smemBytesPerBlock(SH[i].BM,SH[i].BN,SH[i].BK,SH[i].PAD);
          printf("  %-18s %10d\n", SH[i].name, v);
          h = fnv(&v,sizeof(v),h);
      }
      const int ok = (h == 0x6fc4efb3u);
      printf("  hash %08x : %s\n\n", h, ok ? "correct" : "WRONG");
      if (ok) ++score; }

    // ---- TODO 2 -----------------------------------------------------------
    printf("-- TODO 2: blocks per SM -----------------------------------------\n");
    { unsigned h = 2166136261u; int agree = 1;
      printf("  %-18s %8s %8s %10s %9s %8s\n",
             "shape", "threads", "asked", "charged", "yours", "API");
      for (int i=0;i<NSH;++i) {
          const int thr = SH[i].BM*SH[i].BN;
          const int by  = smemBytesPerBlock(SH[i].BM,SH[i].BN,SH[i].BK,SH[i].PAD);
          const int v   = blocksPerSM(thr, by);
          const int api = apiBlocksPerSM(SH[i].BM,SH[i].BN,SH[i].BK,SH[i].PAD);
          const int chg = ((by + DRIVER_RESERVE + SMEM_GRANULARITY-1)
                           / SMEM_GRANULARITY) * SMEM_GRANULARITY;
          printf("  %-18s %8d %8d %10d %9d %8d%s\n",
                 SH[i].name, thr, by, chg, v, api, (v==api)?"":"   <-- differ");
          if (v != api) agree = 0;
          h = fnv(&v,sizeof(v),h);
      }
      printf("  hash %08x : %s   (matches the occupancy API everywhere: %s)\n\n",
             h, h==0xd228846cu ? "correct" : "WRONG", agree ? "yes" : "NO");
      if (!agree)
          printf("  The API disagrees with your model on at least one shape, and the\n"
                 "  API is right. The three-limiter model above is the one Module 6\n"
                 "  established and it is INCOMPLETE: there is a fourth resource the\n"
                 "  block placement gate checks, and cudaFuncGetAttributes() reports\n"
                 "  how much of it each of these kernels needs. Work out which shape\n"
                 "  it bites on and why -- it is not scored, and it is the most\n"
                 "  useful thing in this exercise.\n");
      if (h==0xd228846cu) ++score; }

    // ---- TODO 3 -----------------------------------------------------------
    printf("-- TODO 3: instruction-mix ratios --------------------------------\n");
    { unsigned h = 2166136261u;
      printf("  %-18s %14s\n", "shape", "FMA/global-ld");
      for (int i=0;i<NSH;++i) {
          double v = fmasPerGlobalLoad(SH[i].BM,SH[i].BN,SH[i].BK);
          printf("  %-18s %14.4f\n", SH[i].name, v);
          long long q = (long long)llround(v*1e6); h = fnv(&q,sizeof(q),h);
      }
      { long long q=(long long)llround(sharedLoadsPerFma()*1e6); h=fnv(&q,sizeof(q),h); }
      printf("  shared loads per FMA : %.4f\n", sharedLoadsPerFma());
      const int ok = (h == 0x77fb5de9u);
      printf("  hash %08x : %s\n", h, ok ? "correct" : "WRONG");
      printf("  Module 16 measured that 6.4-6.5 FMAs per GLOBAL load are needed for\n"
             "  80%% of the FP32 ceiling, and that one output per thread supplies\n"
             "  0.50. Compare both of your columns against that number before you\n"
             "  look at any timing.\n\n");
      if (ok) ++score; }

    // ---- TODO 4 -----------------------------------------------------------
    printf("-- TODO 4: bank-conflict degrees ---------------------------------\n");
    { unsigned h = 2166136261u;
      static const char *PN[4] = { "read As[ty][k]", "read Bs[k][tx]",
                                   "store As[ty][tx]", "store As[tx][ty]" };
      printf("  %-18s %6s %6s %8s\n", "access", "T", "pad", "degree");
      for (int p=0;p<4;++p)
        for (int T=16;T<=32;T*=2)
          for (int pad=0;pad<=1;++pad) {
              int w[32];
              for (int l=0;l<32;++l) w[l] = tileWordIndex(p,T,pad,l,3);
              int d = conflictDegree(w,32);
              printf("  %-18s %6d %6d %8d\n", PN[p], T, pad, d);
              h = fnv(&d,sizeof(d),h);
          }
      const int ok = (h == 0xdd175fcdu);
      printf("  hash %08x : %s\n", h, ok ? "correct" : "WRONG");
      printf("  On Ada a degree-D access costs max(2, D), so D <= 2 is free.\n"
             "  Count how many rows of that table describe an access the kernel\n"
             "  you are about to time actually performs.\n\n");
      if (ok) ++score; }

    // ---- data -------------------------------------------------------------
    const size_t sA=(size_t)M*K, sB=(size_t)K*N, sC=(size_t)M*N;
    float *dA,*dB,*dC;
    CHECK(cudaMalloc(&dA,sA*4)); CHECK(cudaMalloc(&dB,sB*4)); CHECK(cudaMalloc(&dC,sC*4));
    { float *h=(float*)malloc((sB>sA?sB:sA)*4);
      for (size_t i=0;i<sA;++i) h[i]=0.5f+(float)(i%251)/502.0f;
      CHECK(cudaMemcpy(dA,h,sA*4,cudaMemcpyHostToDevice));
      for (size_t i=0;i<sB;++i) h[i]=0.5f+(float)(i%257)/514.0f;
      CHECK(cudaMemcpy(dB,h,sB*4,cudaMemcpyHostToDevice));
      free(h); }
    g_ctx.M=M; g_ctx.N=N; g_ctx.K=K; g_ctx.dA=dA; g_ctx.dB=dB; g_ctx.dC=dC;

    // ---- your choice, printed BEFORE any timing ---------------------------
    int chosen = -1;
    for (int i=0;i<NSH;++i)
        if (SH[i].BM==cBM && SH[i].BN==cBN && SH[i].BK==cBK && SH[i].PAD==cPAD)
            chosen = i;
    printf("-- TODO 5: your commitments --------------------------------------\n");
    if (chosen < 0) {
        printf("  chooseTile returned (%d,%d,%d,pad %d), which is not one of the\n"
               "  eight compiled shapes. Pick one from the list in TODO 5.\n",
               cBM,cBN,cBK,cPAD);
        printf("\nSCORE: %d/%d\nOVERALL: FAIL\n", score, MAXSCORE);
        return 1;
    }
    printf("  tile shape   : %s\n", SH[chosen].name);
    printf("  padding      : bucket %d\n\n", PAD_PREDICTION);

    // ---- timing -----------------------------------------------------------
    printf("-- measurement ----------------------------------------------------\n");
    printf("  warming 1500 ms stream + 500 ms FFMA ...\n");
    { const size_t SB=256u*1024u*1024u;
      float4 *si,*so; CHECK(cudaMalloc(&si,SB)); CHECK(cudaMalloc(&so,SB));
      CHECK(cudaMemset(si,0x3c,SB));
      float *fo; CHECK(cudaMalloc(&fo,4));
      int nsm=0; CHECK(cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0));
      cudaEvent_t w0,w1; CHECK(cudaEventCreate(&w0)); CHECK(cudaEventCreate(&w1));
      float el=0.0f; CHECK(cudaEventRecord(w0));
      while (el<1500.0f){ streamWarm<<<nsm*12,256>>>(si,so,SB/sizeof(float4));
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      el=0.0f; CHECK(cudaEventRecord(w0));
      while (el<500.0f){ ffmaWarm<<<nsm*6,256>>>(fo,20000);
        CHECK(cudaEventRecord(w1)); CHECK(cudaEventSynchronize(w1));
        CHECK(cudaEventElapsedTime(&el,w0,w1)); }
      CHECK(cudaEventDestroy(w0)); CHECK(cudaEventDestroy(w1));
      CHECK(cudaFree(si)); CHECK(cudaFree(so)); CHECK(cudaFree(fo));
      CHECK(cudaGetLastError()); }

    double best[NSH]; int it[NSH];
    for (int i=0;i<NSH;++i){ it[i]=calibrate(i,&g_ctx); best[i]=1e30; }
    for (int s=0;s<NSH;++s)
        for (int q=0;q<NSH;++q){ int p=(q+s)%NSH;
            double t=timeOnce(p,&g_ctx,it[p]); if (t<best[p]) best[p]=t; }
    CHECK(cudaGetLastError());

    int bi=0; for (int i=1;i<NSH;++i) if (best[i]<best[bi]) bi=i;
    printf("\n  %-18s %9s %11s %9s %10s\n",
           "shape", "ms", "GFLOP/s", "x best", "FMA/gld");
    for (int i=0;i<NSH;++i)
        printf("  %-18s %9.4f %11.1f %9.3f %10.2f%s\n", SH[i].name, best[i],
               flops/(best[i]*1e-3)/1e9, best[bi]/best[i],
               fmasPerGlobalLoad(SH[i].BM,SH[i].BN,SH[i].BK),
               i==chosen ? "   <-- your choice" : "");

    // ---- scoring ----------------------------------------------------------
    // Rank, not a percentage. The top two shapes are within a few percent of
    // each other and swap places between runs, and on a thermally throttled
    // laptop the whole table shifts; a fixed percentage tolerance therefore
    // rejects a correct answer at random (spec 12 rule 5: ratios between
    // configurations are stable, absolute margins are not). Top 3 of 8 was
    // stable across every run observed, and it still requires a working model:
    // it excludes 8x8, 32x32, 16x32 and every padded shape.
    int rank = 1;
    for (int i = 0; i < NSH; ++i) if (best[i] < best[chosen]) ++rank;
    printf("\n  fastest measured : %s\n", SH[bi].name);
    printf("  your choice ranked %d of %d (%.1f%% of the best). ",
           rank, NSH, 100.0*best[bi]/best[chosen]);
    if (rank <= 3) { printf("ACCEPTED (top 3 required)\n"); ++score; }
    else            printf("REJECTED\n");

    const double padRatio = best[2]/best[7];       // pad0 / pad1, same tile
    int padBucket;
    if      (padRatio > 1.10) padBucket = 1;
    else if (padRatio >= 0.95) padBucket = 2;
    else                       padBucket = 3;
    printf("  padding: time(pad 0)/time(pad 1) = %.3f -> bucket %d. "
           "You predicted %d. %s\n", padRatio, padBucket, PAD_PREDICTION,
           padBucket==PAD_PREDICTION ? "CORRECT" : "wrong");
    if (padBucket == PAD_PREDICTION) ++score;

    printf("\n  If the padding result surprised you, run\n"
           "    nvcc -arch=sm_89 -O3 -cubin -o e2.cubin exercise02.cu\n"
           "    cuobjdump -sass e2.cubin\n"
           "  and count LDS and LDS.128 in the accumulation loop of\n"
           "  gemmTiled<16,16,16,0> and gemmTiled<16,16,16,1>. Your TODO 4\n"
           "  table is correct and it is not the whole story.\n");

    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    printf("\nSCORE: %d/%d\n", score, MAXSCORE);
    printf("OVERALL: %s\n", score==MAXSCORE ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return score==MAXSCORE ? 0 : 1;
}
