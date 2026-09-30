# Module 03 / Exercise 03 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
```

The matrix is 256 × 512 floats, row-major, row stride 512 elements = 2048 bytes.
A 128-byte segment is 32 consecutive floats, and since 2048 is a multiple of 128,
**every row starts on a segment boundary**. That alignment is what makes the
arithmetic below clean; the last section says what changes when it does not hold.

The index expression is fixed:

```cpp
col = blockIdx.x * blockDim.x + threadIdx.x;
row = blockIdx.y * blockDim.y + threadIdx.y;
out[row * W + col] = ...
```

and we are only asked about block (0,0), where `col == threadIdx.x` and
`row == threadIdx.y`. So the element written by a thread is `ty * 512 + tx`.

## TODO 1 — config A, block (8,32), warp 1

```cpp
static const int A_warp1_distinct_ty = 4;
static const int A_warp1_segments    = 4;
static const int A_warp1_min_element = 2048;
static const int A_warp1_max_element = 3591;
```

Derivation, entirely from the linearization rule.

`tid = tx + blockDim.x * ty = tx + 8*ty`. Warp 1 is `tid ∈ [32, 64)`.
Solve: `8*ty ≤ tid < 8*ty + 8`, so `ty ∈ {4, 5, 6, 7}` and `tx ∈ [0, 8)`.
That is **4 distinct `threadIdx.y` values**, 8 lanes each. Lane order is
`(0,4) … (7,4) (0,5) … (7,5) (0,6) … (7,6) (0,7) … (7,7)`.

Elements: `ty*512 + tx`. Minimum is `4*512 + 0 = **2048**`; maximum is
`7*512 + 7 = **3591**`. Note the warp's address span is 1544 elements for 32
stores — 98% of the span is untouched.

Segments: element `i` is in segment `i/32`. Row 4 contributes elements
2048–2055, all in segment 64. Row 5 → segment 80. Row 6 → 96. Row 7 → 112.
**4 segments**, each only 8/32 = 25% covered.

## TODO 2 — config B, block (32,8), warp 1

```cpp
static const int B_warp1_distinct_ty = 1;
static const int B_warp1_segments    = 1;
static const int B_warp1_min_element = 512;
static const int B_warp1_max_element = 543;
```

`tid = tx + 32*ty`. Warp 1 is `tid ∈ [32,64)` ⇒ `ty == 1`, `tx ∈ [0,32)`.
**1 distinct `threadIdx.y`**. Elements `512 … 543`, a span of exactly 32, which
is exactly one aligned segment: **1 segment**, 100% covered.

Note that `blockDim.x == warpSize` is the special case where "warp" and "row of
the tile" coincide. That is the only reason the folk rule *make blockDim.x a
multiple of 32* exists.

## TODO 3 — whole block (0,0), 256 stores

```cpp
static const int A_block_segments = 32;
static const int B_block_segments = 8;
```

- Config A covers `tx ∈ [0,8)`, `ty ∈ [0,32)` — 32 rows, 8 columns each. Every
  row lands in a different segment (rows are 512 elements = 16 segments apart),
  so **32 segments**, each 25% used.
- Config B covers `tx ∈ [0,32)`, `ty ∈ [0,8)` — 8 rows, 32 columns each.
  **8 segments**, each 100% used.

Same 256 stores; 4× the cache-line footprint in A.

## Common wrong predictions

| Prediction | Why it is wrong |
|---|---|
| A: `distinct_ty = 32` | That is the *block*, not warp 1. A warp is 32 consecutive `tid`, which at `blockDim.x = 8` spans exactly 4 rows. |
| A: `min = 0` | Warp **0** starts at element 0. Warp 1 starts at `tid = 32` ⇒ `ty = 4`. |
| A: `max = 2055` | Forgetting that the warp continues into rows 5, 6, 7. |
| A: `segments = 32` | Confusing per-warp with per-block. One warp is only 4 of the block's 32 rows. |
| B: `min = 0, max = 31` | Again warp 0 versus warp 1. |
| A: `segments = 8` | Assuming each lane gets its own segment; 8 lanes share a row and a segment. |

## Memory reasoning

The number the exercise reports (128-byte segments) is a proxy. The actual
request granularity on this architecture is the **32-byte sector**; an L1/L2 line
is 128 bytes = 4 sectors. So in sector terms:

- Config B, warp 1: 32 consecutive floats = 128 aligned bytes = **4 sectors**.
- Config A, warp 1: 4 runs of 8 floats = 4 runs of 32 aligned bytes =
  **4 sectors**.

**Identical sector counts.** This is the result most readers do not predict, and
it is why Exercise 1 measures (8,32) and (32,8) within 2% of each other on a
large image. The (8,32) block is not "uncoalesced"; it is *fragmented across
cache lines*. It costs 4× the L1 tag/line footprint per block and it wastes 75%
of each line it touches if nothing else in the block reuses that line — but in
this kernel other warps of the same block do touch the rest of each line, so the
line-level waste is recovered within the block.

Where it stops being recoverable is `blockDim.x == 1`: then every lane is a
separate row, the warp requests **32 sectors** for 128 useful bytes (12.5%
efficiency), and no amount of intra-block reuse fixes it. Exercise 1 measures
that case at 3.1× slower.

**Alignment caveat.** All of the above depends on the row stride being a multiple
of 32 floats. Here `W = 512`. If `W` were 513, row `ty` would start at element
`513*ty`, which is not segment-aligned, and a 32-lane contiguous warp would
straddle **5** segments instead of 4 for most rows. That is exactly the problem
`cudaMallocPitch` exists to solve — Module 5.

## Performance reasoning

No timing is reported in this exercise on purpose: the kernel is 2 FLOPs and two
memory operations per thread on a 512 KB matrix that lives entirely in L2, so any
time difference would be launch overhead and noise, and would teach the wrong
lesson. The exercise measures *structure* — which threads, which addresses —
because that is the quantity you can compute on paper and must be able to. The
timing consequences are Exercise 1's job and Module 5's.

## Expected output

```
=== Module 3 / Exercise 3 SOLUTION : warp membership and address footprint ===
matrix 256 x 512 floats, row-major, row stride 512 elements = 2048 bytes
one 128-byte segment = 32 consecutive floats

===== config A : block (8,32,1), grid (64,8,1) =====
  numerical result: PASS (0 mismatches of 131072)
  warp 1 of block (0,0) -- lane : (threadIdx.x, threadIdx.y) -> out element -> 128B segment
     0:( 0, 4)->  2048/s64     1:( 1, 4)->  2049/s64     2:( 2, 4)->  2050/s64     3:( 3, 4)->  2051/s64
     4:( 4, 4)->  2052/s64     5:( 5, 4)->  2053/s64     6:( 6, 4)->  2054/s64     7:( 7, 4)->  2055/s64
     8:( 0, 5)->  2560/s80     9:( 1, 5)->  2561/s80    10:( 2, 5)->  2562/s80    11:( 3, 5)->  2563/s80
    12:( 4, 5)->  2564/s80    13:( 5, 5)->  2565/s80    14:( 6, 5)->  2566/s80    15:( 7, 5)->  2567/s80
    16:( 0, 6)->  3072/s96    17:( 1, 6)->  3073/s96    18:( 2, 6)->  3074/s96    19:( 3, 6)->  3075/s96
    20:( 4, 6)->  3076/s96    21:( 5, 6)->  3077/s96    22:( 6, 6)->  3078/s96    23:( 7, 6)->  3079/s96
    24:( 0, 7)->  3584/s112   25:( 1, 7)->  3585/s112   26:( 2, 7)->  3586/s112   27:( 3, 7)->  3587/s112
    28:( 4, 7)->  3588/s112   29:( 5, 7)->  3589/s112   30:( 6, 7)->  3590/s112   31:( 7, 7)->  3591/s112
  warp 1 spans 4 distinct row(s) of the matrix, 4 distinct 128B segment(s),
  element range [2048, 3591] (span of 1544 elements for 32 stores)
  whole block (0,0) touches 32 distinct 128B segments with 256 stores

===== config B : block (32,8,1), grid (16,32,1) =====
  numerical result: PASS (0 mismatches of 131072)
  warp 1 of block (0,0) -- lane : (threadIdx.x, threadIdx.y) -> out element -> 128B segment
     0:( 0, 1)->   512/s16     1:( 1, 1)->   513/s16     2:( 2, 1)->   514/s16     3:( 3, 1)->   515/s16
     ...
    28:(28, 1)->   540/s16    29:(29, 1)->   541/s16    30:(30, 1)->   542/s16    31:(31, 1)->   543/s16
  warp 1 spans 1 distinct row(s) of the matrix, 1 distinct 128B segment(s),
  element range [512, 543] (span of 32 elements for 32 stores)
  whole block (0,0) touches 8 distinct 128B segments with 256 stores

===== your predictions vs the machine =====
  config A, block (8,32):
    warp 1: distinct threadIdx.y values            predicted      4   actual      4   MATCH
    warp 1: distinct 128B segments                 predicted      4   actual      4   MATCH
    warp 1: lowest element index                   predicted   2048   actual   2048   MATCH
    warp 1: highest element index                  predicted   3591   actual   3591   MATCH
  config B, block (32,8):
    warp 1: distinct threadIdx.y values            predicted      1   actual      1   MATCH
    warp 1: distinct 128B segments                 predicted      1   actual      1   MATCH
    warp 1: lowest element index                   predicted    512   actual    512   MATCH
    warp 1: highest element index                  predicted    543   actual    543   MATCH
  whole block (0,0):
    config A: distinct 128B segments / block       predicted     32   actual     32   MATCH
    config B: distinct 128B segments / block       predicted      8   actual      8   MATCH

  predictions correct: 10 / 10

OVERALL: PASS
```

Shipped with the predictions at `-1`, the program still runs both configs, prints
the full roster, reports `predictions correct: 0 / 10` and `OVERALL: FAIL`. The
numerical result is PASS in both cases — the kernel was never the point.

## The result that matters

You can compute a warp's entire address footprint on paper, before compiling,
from three numbers: `blockDim`, the index expression, and the array stride.
`tid = tx + bdx*(ty + bdy*tz)`, warp `w` is `tid ∈ [32w, 32w+32)`, invert for
`(tx,ty,tz)`, substitute into the index expression. That is the whole method, and
every coalescing argument in Modules 5, 15, 17 and 18 is an application of it.
The surprise this exercise plants is that 4 rows × 8 lanes costs the *same sector
count* as 1 row × 32 lanes — so "warps must be rows" is folklore, while
"`blockDim.x` must be large enough that 32 lanes cover at least a sector" is the
real rule.

Variation to try: change `W` from 512 to 513 and re-derive the segment counts by
hand before re-running. Then try block `(4,64)` and predict all ten numbers.
