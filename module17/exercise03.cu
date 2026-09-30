// =============================================================================
// Module 17 / Exercise 3 — "it works on 1024 x 1024 x 1024".
//
// SYMPTOM
//   The tiled GEMM below is correct on 512 x 512 x 512. It is correct on
//   1024 x 1024 x 1024. It is correct on every square power-of-two size that
//   has been tried. On 1035 x 1541 x 1063 it is wrong, and it is wrong in
//   three separate ways at once.
//
//   No launch error is reported. compute-sanitizer --tool memcheck is clean.
//   compute-sanitizer --tool racecheck is clean. The kernel does not crash,
//   does not hang, and returns finite numbers for most of the matrix.
//
//   This is the most common class of GEMM bug there is, and a square
//   power-of-two test suite cannot see any of it.
//
// GOAL : Diagnose, fix, and then design the test that would have caught it.
//
// BUILD: nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
// RUN  : exercise03.exe
//
// TODO 1 - diagnosis: three codes                                 [host]
// TODO 2 - fix defect A in gemmTiledStudent                       [kernel]
// TODO 3 - fix defect B in gemmTiledStudent                       [kernel]
// TODO 4 - fix defect C in gemmTiledStudent                       [kernel]
// TODO 5 - design the size set that exposes all three   [host, DESIGN TODO]
//
// The file ships broken on purpose (spec 6 type 3). Running it as shipped
// prints FAIL and shows you the symptom; it does not crash.
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

#define TILE 16

// =============================================================================
// TODO 1 — diagnosis.
//
// Exactly three of the following seven descriptions apply to the kernel below.
// Put their codes in DIAG_A, DIAG_B, DIAG_C in any order.
//
//   1  the k-tile loop count uses a truncating divide, so the final partial
//      tile of the contraction is never processed
//   2  an out-of-range tile cell is left holding whatever was in that shared
//      slot instead of being zero-filled
//   3  the guard on the store compares the column index against the wrong
//      matrix dimension
//   4  the A-tile and B-tile load mappings are swapped, so each thread stages
//      the element the other one needed
//   5  a barrier is missing, so a thread reads tile cells before their writers
//      have written them
//   6  the accumulator lives in shared memory rather than in a register, so
//      threads overwrite each other's partial sums
//   7  the grid is sized with a truncating divide, so the last row and column
//      of blocks are never launched
//
// Four of these seven produce a symptom that IS visible on a square
// power-of-two problem. Identifying which four is most of the work.
// =============================================================================
#define DIAG_A 0        // <- YOUR ANSWER HERE
#define DIAG_B 0        // <- YOUR ANSWER HERE
#define DIAG_C 0        // <- YOUR ANSWER HERE

// =============================================================================
// The kernel. TODOs 2, 3 and 4 are fixes to the code below. Each one is a
// single decision, and the code as shipped is a plausible thing for a
// competent person to have written.
// =============================================================================
__global__ void gemmTiledStudent(int M, int N, int K,
                                 const float * __restrict__ A,
                                 const float * __restrict__ B, float *C)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    // -------------------------------------------------------------------
    // TODO 2: defect A is on the next line. Fix it.
    // -------------------------------------------------------------------
    const int nTiles = K / TILE;

    for (int t = 0; t < nTiles; ++t) {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;

        // ---------------------------------------------------------------
        // TODO 3: defect B is in the next four lines. Fix it.
        //
        // Both guards are present and both index expressions are right.
        // Ask what a thread whose guard is false leaves behind, and what the
        // accumulation loop below does with it.
        // ---------------------------------------------------------------
        if (row < M && aCol < K)
            As[ty][tx] = A[(size_t)row * K + aCol];
        if (bRow < K && col < N)
            Bs[ty][tx] = B[(size_t)bRow * N + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc = fmaf(As[ty][k], Bs[k][tx], acc);

        __syncthreads();
    }

    // -------------------------------------------------------------------
    // TODO 4: defect C is in the guard on the next line. Fix it.
    // -------------------------------------------------------------------
    if (row < M && col < M)
        C[(size_t)row * N + col] = acc;
}

// =============================================================================
// TODO 5 — DESIGN. The test suite that would have caught this.
//
// Supply up to MAX_PROBE candidate problem shapes. For index `i` in
// [0, MAX_PROBE), write (M, N, K) into the out-parameters and return 1; return
// 0 to stop. Constraints: 8 <= each dimension <= 2048, and the total work
// summed over all your shapes must stay under the budget the harness prints
// (so you cannot pass by brute-forcing every size).
//
// Each defect becomes observable only when the problem shape has a particular
// arithmetic property. The harness computes, for the union of your shapes,
// which of these six properties hold:
//
//     bit 0 : K % TILE != 0        bit 3 : M != N
//     bit 1 : M % TILE != 0        bit 4 : K < TILE
//     bit 2 : N % TILE != 0        bit 5 : M > N
//
// THREE of those six are what the three defects need. You score one point for
// each required property your shape set covers. Which three they are is the
// same question as TODO 1, asked a different way: do not answer it by trial.
//
// Two of the three defects need the SAME property, so three shapes is more
// than you need. The cheapest correct answer uses ONE shape, and it is small.
// =============================================================================
#define MAX_PROBE 4
static int probeSize(int i, int *M, int *N, int *K)
{
    (void)i; (void)M; (void)N; (void)K;
    // YOUR CODE HERE
    return 0;
}

// =============================================================================
// Module 16's validator, verbatim.
// =============================================================================
typedef struct {
    int ok, nonfinite;
    double freivalds; int freivaldsRow;
    double sampled;   int sampledRow, sampledCol;
} GemmCheck;

static GemmCheck gemmValidate(int M, int N, int K,
                              float alpha, const float *hA, const float *hB,
                              float beta,  const float *hC0, const float *hC,
                              int sampleStride)
{
    GemmCheck r; r.ok = 1; r.nonfinite = 0;
    r.freivalds = 0.0; r.freivaldsRow = -1;
    r.sampled = 0.0; r.sampledRow = -1; r.sampledCol = -1;
    const double u = ldexp(1.0, -24);
    const double gammaK = (double)K * u / (1.0 - (double)K * u);

    for (size_t i = 0; i < (size_t)M * N; ++i)
        if (!isfinite(hC[i])) ++r.nonfinite;
    if (r.nonfinite) r.ok = 0;

    double *v   = (double*)malloc(sizeof(double) * N);
    double *Bv  = (double*)malloc(sizeof(double) * K);
    double *aBv = (double*)malloc(sizeof(double) * K);
    unsigned s = 0xB16B00B5u;
    for (int j = 0; j < N; ++j) { s = s*1664525u+1013904223u;
                                  v[j] = 0.5 + (double)((s>>8)&0xFFFFu)/65536.0; }
    for (int k = 0; k < K; ++k) {
        double t = 0.0, tb = 0.0;
        for (int j = 0; j < N; ++j) { double b = hB[(size_t)k*N+j];
            t += b*v[j]; tb += fabs(b)*v[j]; }
        Bv[k] = t; aBv[k] = tb;
    }
    for (int i = 0; i < M; ++i) {
        double y = 0.0, yb = 0.0;
        for (int k = 0; k < K; ++k) { double a = hA[(size_t)i*K+k];
            y += a*Bv[k]; yb += fabs(a)*aBv[k]; }
        double got = 0.0, want = (double)alpha * y;
        for (int j = 0; j < N; ++j) {
            got += (double)hC[(size_t)i*N+j] * v[j];
            if (beta != 0.0f) want += (double)beta*(double)hC0[(size_t)i*N+j]*v[j];
        }
        double c0v = 0.0;
        if (beta != 0.0f)
            for (int j = 0; j < N; ++j) c0v += fabs((double)hC0[(size_t)i*N+j])*v[j];
        double tol = fabs((double)alpha)*(gammaK+4.0*u)*yb
                   + 4.0*u*fabs((double)beta)*c0v;
        double ratio = (tol > 0.0) ? fabs(got-want)/tol : 0.0;
        if (ratio > r.freivalds) { r.freivalds = ratio; r.freivaldsRow = i; }
    }
    free(v); free(Bv); free(aBv);
    if (!(r.freivalds <= 1.0)) r.ok = 0;

    for (int i = 0; i < M; i += sampleStride)
        for (int j = 0; j < N; j += sampleStride) {
            double acc = 0.0, S = 0.0;
            for (int k = 0; k < K; ++k) {
                double a = hA[(size_t)i*K+k], b = hB[(size_t)k*N+j];
                acc += a*b; S += fabs(a)*fabs(b);
            }
            double want = (double)alpha*acc;
            if (beta != 0.0f) want += (double)beta*(double)hC0[(size_t)i*N+j];
            double tol = fabs((double)alpha)*(gammaK+4.0*u)*S
                       + 4.0*u*fabs((double)beta)
                             * (hC0 ? fabs((double)hC0[(size_t)i*N+j]) : 0.0);
            double d = (tol > 0.0) ? fabs((double)hC[(size_t)i*N+j]-want)/tol : 0.0;
            if (d > r.sampled) { r.sampled = d; r.sampledRow = i; r.sampledCol = j; }
        }
    if (!(r.sampled <= 1.0)) r.ok = 0;
    return r;
}

// =============================================================================
// harness
// =============================================================================
static unsigned lcg = 1u;
static float rnd01(void){ lcg = lcg*1664525u+1013904223u;
                          return (float)((lcg>>8)&0xFFFFu)/65536.0f; }

typedef struct { float *hA,*hB,*hC; float *dA,*dB,*dC; int M,N,K; } Prob;

static void probAlloc(Prob *p, int M, int N, int K)
{
    p->M=M; p->N=N; p->K=K;
    const size_t sA=(size_t)M*K, sB=(size_t)K*N, sC=(size_t)M*N;
    p->hA=(float*)malloc(sA*4); p->hB=(float*)malloc(sB*4); p->hC=(float*)malloc(sC*4);
    lcg = 1u + (unsigned)(M*7919 + N*104729 + K*15485863);
    for (size_t i=0;i<sA;++i) p->hA[i]=0.5f+rnd01();
    for (size_t i=0;i<sB;++i) p->hB[i]=0.5f+rnd01();
    CHECK(cudaMalloc(&p->dA,sA*4)); CHECK(cudaMalloc(&p->dB,sB*4));
    CHECK(cudaMalloc(&p->dC,sC*4));
    CHECK(cudaMemcpy(p->dA,p->hA,sA*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(p->dB,p->hB,sB*4,cudaMemcpyHostToDevice));
}
static void probFree(Prob *p)
{
    free(p->hA); free(p->hB); free(p->hC);
    CHECK(cudaFree(p->dA)); CHECK(cudaFree(p->dB)); CHECK(cudaFree(p->dC));
}
static void poison(Prob *p)
{
    union { unsigned u; float f; } inf; inf.u = 0x7F800000u;
    const size_t sC=(size_t)p->M*p->N;
    for (size_t i=0;i<sC;++i) p->hC[i]=inf.f;
    CHECK(cudaMemcpy(p->dC,p->hC,sC*4,cudaMemcpyHostToDevice));
}
static GemmCheck runStudent(Prob *p, int stride)
{
    poison(p);
    dim3 bl(TILE,TILE);
    dim3 gr((unsigned)((p->N+TILE-1)/TILE),(unsigned)((p->M+TILE-1)/TILE));
    gemmTiledStudent<<<gr,bl>>>(p->M,p->N,p->K,p->dA,p->dB,p->dC);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(p->hC,p->dC,(size_t)p->M*p->N*4,cudaMemcpyDeviceToHost));
    return gemmValidate(p->M,p->N,p->K,1.0f,p->hA,p->hB,0.0f,NULL,p->hC,stride);
}
// The six-bit property signature of a problem shape (see TODO 5).
static unsigned shapeSignature(int M, int N, int K)
{
    unsigned s = 0;
    if (K % TILE != 0) s |= 1u << 0;
    if (M % TILE != 0) s |= 1u << 1;
    if (N % TILE != 0) s |= 1u << 2;
    if (M != N)        s |= 1u << 3;
    if (K <  TILE)     s |= 1u << 4;
    if (M >  N)        s |= 1u << 5;
    return s;
}
static const char *PROPNAME[6] = {
    "K % TILE != 0", "M % TILE != 0", "N % TILE != 0",
    "M != N",        "K < TILE",      "M > N"
};
// FNV-1a of the required-property mask. The mask itself is not written in this
// file; the harness recovers it at run time by searching the 64 candidates.
static unsigned fnv32(unsigned v)
{
    unsigned h = 2166136261u;
    for (int i = 0; i < 4; ++i) { h ^= (v >> (8*i)) & 0xFFu; h *= 16777619u; }
    return h;
}
#define REQUIRED_MASK_HASH 0x7f8f9f9cu

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Module 17 / Exercise 3 - 'it works on 1024 x 1024 x 1024' ===\n\n");

    int score = 0;
    const int MAXSCORE = 7;

    // ---- the symptom, always shown ----------------------------------------
    printf("-- the symptom -----------------------------------------------------\n");
    { Prob sq; probAlloc(&sq, 512, 512, 512);
      GemmCheck c = runStudent(&sq, 32);
      printf("  512 x 512 x 512       nonfin %8d | Freiv %9.4g | samp %9.4g | %s\n",
             c.nonfinite, c.freivalds, c.sampled, c.ok ? "PASS" : "FAIL");
      probFree(&sq); }
    { Prob aw; probAlloc(&aw, 1035, 1541, 1063);
      GemmCheck c = runStudent(&aw, 64);
      printf("  1035 x 1541 x 1063    nonfin %8d | Freiv %9.4g | samp %9.4g | %s\n",
             c.nonfinite, c.freivalds, c.sampled, c.ok ? "PASS" : "FAIL");
      printf("\n  Read the three numbers, not just the verdict. A non-zero\n"
             "  'nonfin' means elements of C were never written at all, and the\n"
             "  +infinity the harness prefilled them with survived. That is a\n"
             "  different defect from a wrong finite value, and it points at a\n"
             "  different line.\n\n");
      probFree(&aw); }

    // ---- TODO 1 -----------------------------------------------------------
    if (DIAG_A == 0 || DIAG_B == 0 || DIAG_C == 0) {
        printf("Set TODO 1 first.\n");
        printf("\nSCORE: 0/%d\nOVERALL: FAIL\n", MAXSCORE);
        CHECK(cudaDeviceReset());
        return 1;
    }
    printf("-- TODO 1: diagnosis -----------------------------------------------\n");
    { int got[3] = { DIAG_A, DIAG_B, DIAG_C };
      // the three that apply, as a set
      int want[3] = { 1, 2, 3 };
      int hit = 0;
      for (int w = 0; w < 3; ++w)
          for (int g = 0; g < 3; ++g) if (got[g] == want[w]) { ++hit; break; }
      // reject duplicates
      if (got[0]==got[1] || got[1]==got[2] || got[0]==got[2]) hit = 0;
      printf("  you answered {%d, %d, %d} : %d of 3 correct\n\n",
             DIAG_A, DIAG_B, DIAG_C, hit);
      score += hit; }

    // ---- TODOs 2-4: the repaired kernel ------------------------------------
    printf("-- TODOs 2-4: the repaired kernel ----------------------------------\n");
    { int ok = 1;
      const int dims[3][3] = { {512,512,512}, {1035,1541,1063}, {97,61,9} };
      const int strides[3] = { 32, 64, 8 };
      for (int i = 0; i < 3; ++i) {
          Prob p; probAlloc(&p, dims[i][0], dims[i][1], dims[i][2]);
          GemmCheck c = runStudent(&p, strides[i]);
          printf("  %5d x %5d x %5d   nonfin %8d | Freiv %9.4g | samp %9.4g | %s\n",
                 dims[i][0], dims[i][1], dims[i][2],
                 c.nonfinite, c.freivalds, c.sampled, c.ok ? "PASS" : "FAIL");
          if (!c.ok) ok = 0;
          probFree(&p);
      }
      printf("  (the third shape has K < TILE: the ONLY tile is a partial one,\n"
             "   so an un-zero-filled cell is uninitialised shared memory rather\n"
             "   than stale data from a previous tile)\n\n");
      if (ok) ++score; }

    // ---- TODO 5 ------------------------------------------------------------
    printf("-- TODO 5: the size set that would have caught it ------------------\n");
    { int m,n,k, nProbe = 0;
      double work = 0.0;
      const double BUDGET = 4.0e9;          // sum of M*N*K over your shapes
      unsigned sigAll = 0;
      printf("  work budget: sum of M*N*K over all your shapes <= %.1f G\n",
             BUDGET/1e9);
      for (int i = 0; i < MAX_PROBE; ++i) {
          if (!probeSize(i,&m,&n,&k)) break;
          if (m < 8 || n < 8 || k < 8 || m > 2048 || n > 2048 || k > 2048) {
              printf("  shape %d = (%d,%d,%d) is out of range [8,2048]; ignored\n",
                     i,m,n,k);
              continue;
          }
          work += (double)m*n*k;
          if (work > BUDGET) { printf("  budget exceeded at shape %d; stopping\n", i);
                               break; }
          // the repaired kernel must also be correct on every shape you supply
          Prob p; probAlloc(&p,m,n,k);
          const int stride = (m > 256) ? 32 : 1;
          GemmCheck c = runStudent(&p, stride);
          const unsigned sig = shapeSignature(m,n,k);
          printf("  shape %d = %4d x %4d x %4d  props", i, m, n, k);
          for (int b = 0; b < 6; ++b) printf(" %c", (sig >> b) & 1u ? '1' : '.');
          printf("   your kernel: %s\n", c.ok ? "PASS" : "FAIL");
          if (!c.ok) printf("      (your repaired kernel is WRONG on this shape)\n");
          sigAll |= sig;
          probFree(&p);
          ++nProbe;
      }
      if (nProbe == 0) printf("  no shapes supplied (TODO 5 is empty).\n");
      printf("  property order:");
      for (int b = 0; b < 6; ++b) printf("  %d=%s", b, PROPNAME[b]);
      printf("\n");

      unsigned req = 0;
      for (unsigned cand = 0; cand < 64u; ++cand)
          if (fnv32(cand) == REQUIRED_MASK_HASH) { req = cand; break; }
      int need = 0, have = 0;
      for (int b = 0; b < 6; ++b) {
          if (!((req >> b) & 1u)) continue;
          ++need;
          if ((sigAll >> b) & 1u) ++have;
      }
      printf("\n  required properties covered by your set: %d of %d\n", have, need);
      score += have; }

    printf("\nSCORE: %d/%d\n", score, MAXSCORE);
    printf("OVERALL: %s\n", score == MAXSCORE ? "PASS" : "FAIL");
    CHECK(cudaDeviceReset());
    return score == MAXSCORE ? 0 : 1;
}
