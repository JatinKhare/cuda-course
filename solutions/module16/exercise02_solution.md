# Module 16 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

Warning-clean. About 20 s: 1500 ms warm-up plus 12 configurations × 12 rotated
sweeps with iteration counts auto-scaled to ~10 ms segments.

---

## TODO 1 — `predictSectors()`

```cpp
long long sa[32], sb[32];
for (int lane = 0; lane < 32; ++lane) {
    const int tx = lane % bx, ty = lane / bx;
    const int row = (mapping == MAP_XROW) ? tx : ty;
    const int col = (mapping == MAP_XROW) ? ty : tx;
    sa[lane] = ((long long)row * K + 0) * 4 / 32;      // &A[row*K + 0]
    sb[lane] = ((long long)0   * N + col) * 4 / 32;    // &B[0*N + col]
}
// count distinct in sa, count distinct in sb
```

This is Module 5's procedure with nothing GEMM-specific in it: lane → element →
byte address → sector index → count distinct. Two lines carry all the content.

**`tx = lane % bx, ty = lane / bx`** is Module 3's linearization rule read
backwards. A warp is 32 consecutive `tid = threadIdx.x + blockDim.x*threadIdx.y`,
so a warp is a *row* of the block only when `bx >= 32`. At `bx = 8` a warp spans
four values of `threadIdx.y`, which is exactly why the model's answer changes
with `bx` even though the kernel source does not.

**Counting distinct sectors, not distinct lanes.** The single most common error
is to return 32 for the A access under `x -> row` and 32 for the B access under
`x -> col` — i.e. to count lanes. When all 32 lanes present the same address the
coalescer emits **one** sector and broadcasts the word. Module 4 priced this for
constant memory and Module 7 priced it for shared memory; it is the same
mechanism in global memory. `s[31-tid]` in M7 and `A[row*K+k]` here are the same
observation: the coalescer sees a *set* of addresses, not a sequence, and
duplicates are free.

Reference answers, for the eight probed configurations:

| mapping | block | A sectors | B sectors | total |
|---|---|---|---|---|
| x → col | (32, 8) | 1 | 4 | 5 |
| x → col | (16,16) | 2 | 2 | 4 |
| x → col | (8, 32) | 4 | 1 | 5 |
| x → col | (1,256) | 32 | 1 | 33 |
| x → row | (32, 8) | 32 | 1 | 33 |
| x → row | (16,16) | 16 | 1 | 17 |
| x → row | (2,128) | 2 | 2 | 4 |
| x → row | (4, 64) | 4 | 1 | 5 |

Note the last two rows: **the "bad" mapping at `bx = 2` has the same sector
count as the "good" mapping at `bx = 16`.** That is the exercise's payload and
it is visible in the model before anything is measured.

## TODO 2 — kernel B and its grid

```cpp
const int row = blockIdx.x * blockDim.x + threadIdx.x;
const int col = blockIdx.y * blockDim.y + threadIdx.y;
if (row >= M || col >= N) return;
// ... identical loop and store ...
```

and

```cpp
return dim3((unsigned)((M + bx - 1)/bx), (unsigned)((N + by - 1)/by));
```

The grid line is the trap, and it is Module 3 Exercise 1's trap. The axis
decision appears **twice**, once in the kernel and once in the launch. Copying
the mapping-0 grid line here gives a grid covering `min` of the two extents in
each direction: with M = 1027 and N = 2053 it computes 1027 × 1027 elements
(some of them twice, by different blocks, which is harmless only because the
kernel is idempotent) and leaves 1 053 702 elements of C untouched. The harness
compares against kernel A at all six shapes and reports the count.

**The result worth internalising:** the two kernels compile to **byte-identical
SASS**. `cuobjdump -sass` on both gives the same census for the unrolled inner
loop:

```
59 LDG.E   36 IMAD.WIDE   30 FFMA   20 MOV   11 BRA   10 IADD3   6 IMAD ...
```

There is no instruction-level difference at all. The entire effect is in which
32 addresses a warp presents.

## TODO 3 — the predictions

```cpp
#define PREDICT_SPREAD_BUCKET  3     // 2.5x to 6x
#define PREDICT_BEST_BX_XROW   2
```

The spread over twelve configurations measures **3.8–4.0×** here. The sector
model predicts 33/4 = 8.25×, so bucket 4 is the answer the model gives and
bucket 3 is the answer the hardware gives; this is the discrepancy the exercise
wants you to confront. See "Performance reasoning" below.

`bx = 2` for `x -> row` follows straight from the table: 2 rows × 16 columns per
warp gives 2 A sectors + 2 B sectors = 4, the minimum available. It is scored
with a 5 % band because `bx = 2` and `bx = 4` measure within ~1 % of each other
and the ordering between them is not stable run to run.

## TODO 4 — `chooseBx()` (design)

```cpp
int bestBx = 1, bestSec = 1 << 30;
for (int bx = 1; bx <= 32; bx *= 2) {
    int a = 0, b = 0;
    predictSectors(bx, THREADS_PER_BLOCK/bx, mapping, M, N, K, &a, &b);
    if (a + b <= bestSec) { bestSec = a + b; bestBx = bx; }
}
return bestBx;
```

The `<=` gives the larger `bx` on ties, as specified. The point of the TODO is
that the harness calls this **before it times anything** and prints the answer;
a hard-coded constant read off the measured table is detectable by inspection
and, more importantly, does not survive a change of problem shape. Run the
program with `M_DIM`/`N_DIM`/`K_DIM` edited and the model still picks correctly;
a constant does not.

---

## Performance reasoning

The measured table (one run; see below for variance):

| mapping | block | sectors | ms | GFLOP/s | vs best |
|---|---|---|---|---|---|
| x → col | (1,256) | 33 | 9.67 | 335 | 4.02× |
| x → col | (2,128) | 17 | 5.47 | 593 | 2.27× |
| x → col | (4, 64) | 9 | 3.60 | 900 | 1.50× |
| x → col | (8, 32) | 5 | 2.48 | 1307 | 1.03× |
| x → col | (16,16) | **4** | **2.41** | **1348** | **1.00×** |
| x → col | (32, 8) | 5 | 2.42 | 1340 | 1.01× |
| x → row | (1,256) | 5 | 3.86 | 840 | 1.60× |
| x → row | (2,128) | **4** | 2.52 | 1288 | 1.05× |
| x → row | (4, 64) | 5 | 2.53 | 1283 | 1.05× |
| x → row | (8, 32) | 9 | 3.53 | 918 | 1.47× |
| x → row | (16,16) | 17 | 5.58 | 581 | 2.32× |
| x → row | (32, 8) | 33 | 9.08 | 357 | 3.77× |

**The model orders eleven of the twelve correctly.** Group the rows by sector
count and the times fall into bands: 4 sectors → 2.41–2.52 ms, 5 → 2.42–3.86,
9 → 3.53–3.60, 17 → 5.47–5.58, 33 → 9.08–9.67. The one outlier is
`x → row (1,256)` at 5 sectors but 3.86 ms: a block of 1 × 256 covers one row
and 256 columns of C, so its B footprint is 1 KB of *distinct* cache lines and
its A footprint is a single row — the block-level locality is poor even though
the warp-level sector count is good. Sector counting is a per-warp model and has
no term for block shape.

**The mapping is not about the variable names.** `x → row` at `bx = 2` measures
1288 GFLOP/s, within 5 % of the best `x → col` configuration. Setting
`blockDim.x = 2` makes `threadIdx.y` the index that varies across most of a
warp, and `col` varies with `threadIdx.y` in that mapping — so the warp again
spans 16 consecutive columns and the access is coalesced. The rule is Module 3's,
stated once more: **give the fast axis of the warp linearization to the index
with the smallest memory stride.** Which built-in that happens to be is a
consequence of `blockDim`, not of the kernel text.

**Why the model over-predicts the ratio.** Predicted spread 33/4 = 8.25×,
measured 3.8–4.0×. Two reasons, both of them Module 6's:

1. Sectors *requested* are not sectors *fetched*. The 32 scattered A sectors of
   the bad configuration are re-requested by every warp in the grid, and with a
   17.9 MB working set inside a 48 MB L2 they hit almost every time. An L2 hit
   costs ~241 cycles (Module 4) instead of ~575, and costs zero DRAM bytes.
2. The kernel is not sector-bound even at its best. At 1348 GFLOP/s it is at
   7.4 % of the measured FP32 ceiling and the binding constraint is the *number*
   of memory instructions per FMA, not the bytes each one moves. A
   bytes-per-instruction model can order configurations but cannot predict the
   absolute time of a kernel limited by instruction count.

At M = N = K = 1024 (working set 12.6 MB, K a power of two) the same pair of
kernels measures **13.12 ms vs 1.60 ms = 8.2×**, much closer to the model. The
model is not wrong; it is an upper bound on a cost that the caches partly refund.

**Variance.** Absolute times move up to 2× with thermal state on this laptop
part. The ratios in the table above reproduced to within ~5 % across five runs;
the spread bucket landed at 3.3–4.0× every time and the sector-band ordering
never changed. Timing follows spec §12: 1500 ms duration-based warm-up, all
twelve configurations back-to-back in one loop with **rotated** start
(`p = (q + sweep) % 12`) and `SWEEPS = NCFG = 12` so every configuration leads a
sweep exactly once, iteration counts auto-scaled to ~10 ms segments, min-of-12,
and correctness validated in a separate untimed pass before any timing.

---

## Expected output

```
-- TODO 1: your sector model --------------------------------------
  mapping    block       A sect   B sect    total
  x -> col   ( 32,  8)        1        4        5
  x -> col   ( 16, 16)        2        2        4
  x -> col   (  8, 32)        4        1        5
  x -> col   (  1,256)       32        1       33
  x -> row   ( 32,  8)       32        1       33
  x -> row   ( 16, 16)       16        1       17
  x -> row   (  2,128)        2        2        4
  x -> row   (  4, 64)        4        1        5
  model hash fdf98fc1  -> correct

-- TODO 4: your model's choice, made before any measurement -------
  x -> col : blockDim.x = 16
  x -> row : blockDim.x = 2

-- TODO 2: kernel B correctness (bit-comparable to kernel A) ------
  all six block shapes agree with the mapping-0 kernel.

  [12-row measured table as above]

-- model vs measurement ------------------------------------------
  spread slowest/fastest     : 4.02x -> bucket 3   predicted 3  correct
  fastest bx for x -> row    : 2   predicted 2 -> 2.5173 ms vs 2.5173 ms  correct (within 5%)
  your model chose bx = 2 for x -> row, measurement says 2  agree (within 5%)
  your model chose bx = 16 for x -> col, measurement says 16  agree (within 5%)

  SCORE: 6/6
OVERALL: PASS
```

---

## The result that matters

The naive GEMM contains exactly one free choice that costs nothing to make and
almost 4× to make badly, the two kernels that differ by it compile to identical
machine code, and a 20-line hand model of one warp's sector footprint predicts
the ordering of all twelve configurations before anything is run. That is the
whole case for doing the address analysis first. And the twist — that
`blockDim.x = 2` fully repairs the "wrong" mapping — is the reminder that the
warp linearization, not the variable naming, is the thing that decides
coalescing.

**Variation to try:** allocate A with a padded leading dimension
`lda = K + p` instead of `K`, for p = 1, 8 and 32, keeping everything else
identical (the kernel indexes `A[row*lda + k]` and the data is written with the
same stride). `predictSectors` returns 32 for the `x → row` A access at every
p, because the addresses are still 32 distinct sectors. Measure it anyway and
see whether the time moves. Sector counting is one model among several; Module 5
taught row pitch as a global-memory layout decision and Module 7 taught the same
idea one level down for shared-memory banks, and there is a level below the
sector — DRAM page and L2 set behaviour — that neither module opens and that
this experiment pokes at.
