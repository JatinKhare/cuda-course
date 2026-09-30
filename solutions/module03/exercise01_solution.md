# Module 03 / Exercise 01 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

## TODO 1 — global row and column

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

The array is row-major with row stride `w`, so the offset is `row * w + col`.
Within a warp, lanes are consecutive values of the linearized thread index, and
x varies fastest — so lanes differ in `threadIdx.x`. Whatever `threadIdx.x`
carries is the axis along which the warp's 32 addresses walk.

- Put `col` on x: lane addresses differ by 1 element = 4 bytes → the warp asks
  for 128 contiguous bytes = 4 aligned 32-byte sectors.
- Put `row` on x: lane addresses differ by `w` elements = 2932 bytes → 32
  separate sectors, 8× the traffic for the same instruction.

This is the *only* reason to prefer one over the other. The arithmetic is
symmetric; the memory system is not.

### Why swapping them is caught, and how it fails

Swapping produces a kernel that compiles, runs, never faults, and is wrong in a
way that depends on the grid shape. With the swap, the x extent of the grid is
still `ceil(w / block.x)` but x now indexes rows, so the grid covers only
`ceil(733/32)*32 = 736` rows of a 1021-row image. Observed:

```
--- block (32,8) : grid (23,128,1), 753664 threads for 748393 pixels (5271 idle) ---
  block (32,8)       FAIL  208863 / 748393 elements wrong; first at (row=736, col=0): got 0.000000, want 0.702125
```

`got 0.000000` is the `cudaMemset` value — rows 736 and beyond were never
written at all. Meanwhile columns 128 and beyond were written by threads that
also wrote *somebody else's* pixel, because `row * w + col` with the two swapped
aliases onto a different element. Notice the kernel is also *slower* (30% of
peak vs 72%) — but you must not rely on noticing that; the validation is what
catches it.

**Common wrong approaches and their symptoms**

| Mistake | Symptom |
|---|---|
| `row` on x, `col` on y | FAIL with a contiguous block of untouched output rows; first mismatch at a row index that is a multiple of `blockDim.x` |
| `blockIdx.x * blockDim.y + threadIdx.x` (wrong `blockDim` component) | works only when the block is square; FAIL on `(32,8)` |
| Using the literal `W` instead of the parameter `w` | passes at 1021×733, FAIL at 4093×3079 |
| Forgetting that `blockIdx`/`blockDim` are unsigned | fine here (values are small); a real hazard once `blockIdx.x * blockDim.x` can exceed 2³² |

## TODO 2 — the bounds guard

```cpp
if (row >= h || col >= w) return;
```

Both axes. `1021 % 8 == 5` and `733 % 32 == 13`, so both the last block row and
the last block column are partial; guarding only one lets the other run off the
end. The grid launches 753,664 threads for 748,393 pixels — 5,271 threads with
nothing to do.

The `return` must come **before** any memory access, including the clamped
neighbour reads. Computing `clampi(row-1, 0, h-1)` for an out-of-range `row` is
harmless arithmetic, but the loads that follow are not.

Cost: for every warp whose 32 lanes agree on the predicate — which is all of
them except the ones straddling an image edge — this is a uniform compare and a
branch that is either taken or not taken by the whole warp. One instruction of
issue, no divergence penalty. There is at most one straddling warp per block-row
edge. This is why "the guard costs a branch" is not an argument against it.

A tempting alternative is `if (row < h && col < w) { ... }` wrapping the body.
Identical semantics; the early `return` keeps the register live range shorter,
which occasionally matters (Module 19).

## TODO 3 — the grid

```cpp
static dim3 makeGrid(int h, int w, dim3 block)
{
    unsigned gx = (unsigned)(w / (int)block.x) + (unsigned)((w % (int)block.x) != 0);
    unsigned gy = (unsigned)(h / (int)block.y) + (unsigned)((h % (int)block.y) != 0);
    return dim3(gx, gy, 1);
}
```

Three points.

1. **x from `w`, y from `h`.** The grid extent must match the axis its index
   carries. Writing `gx` from `h` is the same bug as TODO 1 wearing a different
   hat, and it is the reason the harness passes four different block shapes:
   a `makeGrid` that happens to work for `(32,8)` because the numbers are close
   will fail for `(256,1)`.
2. **The overflow-proof ceil-divide.** `(w + block.x - 1) / block.x` is the
   familiar form and it overflows when `w + block.x - 1 > INT_MAX`. Not here —
   but writing the safe form costs nothing and the habit is worth having.
   `a / b + (a % b != 0)` never forms an intermediate larger than `a`.
3. **`z = 1`, not 0.** `dim3(gx, gy)` would do the same thing via the default
   constructor; the explicit 1 is there to make the point that a 0 would be a
   `cudaErrorInvalidConfiguration` at launch.

## Synchronization / memory reasoning

There is no synchronization in this kernel and none is needed: every output
element is written by exactly one thread and no thread reads an output. The
stencil reads the *input* array, which no thread writes — that separation is
what makes an out-of-place stencil embarrassingly parallel. An in-place stencil
would be a genuine data race (Module 10).

Memory: each output pixel requires 5 input loads. Four of the five are served by
the caches, because neighbouring threads and neighbouring warps request
overlapping rows. The compulsory traffic is one read and one write per pixel =
8 B/pixel, which is what the harness divides by the measured time.

## Performance reasoning

Measured on the RTX 3500 Ada (timings vary run to run by roughly ±30% on the
small image, where the kernel runs for only ~15 µs and launch overhead is a
visible fraction; the large-image numbers are stable to ~2%):

**1021 × 733, 3.0 MB per array — fits entirely in the 48 MB L2**

| Block | grid | ms | effective GB/s | % of peak |
|---|---|---|---|---|
| (32,8) | (23,128) | 0.0107 | 559.5 | 129.5% |
| (8,32) | (92,32) | 0.0150 | 399.1 | 92.4% |
| (256,1) | (3,1021) | 0.0168 | 355.4 | 82.3% |
| (1,256) | (733,4) | 0.0630 | 95.0 | 22.0% |

**4093 × 3079, 50.4 MB per array — exceeds L2**

| Block | grid | ms | effective GB/s | % of peak |
|---|---|---|---|---|
| (32,8) | (97,512) | 0.3213 | 313.8 | 72.6% |
| (8,32) | (385,128) | 0.3275 | 307.8 | 71.3% |
| (256,1) | (13,4093) | 0.3212 | 313.9 | 72.7% |
| (1,256) | (3079,16) | 1.0022 | 100.6 | 23.3% |

Read these carefully; there are three lessons and two of them are
counter-intuitive.

**"129.5% of peak" is not a measurement error.** The 3 MB working set is
resident in L2 after the warm-up launch, so most of the traffic never reaches
DRAM. Any time an "effective bandwidth" exceeds the DRAM ceiling, you have
learned that your benchmark is measuring the cache, not the bus. This is why the
harness runs the second, larger size at all.

**(8,32) is *not* catastrophic.** The folk rule "make `blockDim.x` a multiple of
32 or your warps straddle rows" predicts a disaster here and does not get one:
8 floats is exactly 32 bytes, which is exactly one sector, so a (8,32) warp
requests 4 sectors — the same count as a (32,8) warp. What it costs is cache-line
footprint (32 distinct 128-byte lines per block instead of 8; Exercise 3
measures this) and a slightly worse access pattern for the ±1-row neighbours.
2% at the large size.

**`blockDim.x == 1` is the real cliff: 3.1× slower.** With `blockDim.x == 1`,
consecutive lanes differ in `threadIdx.y`, so consecutive lanes are a whole row
(3079 floats = 12,316 bytes) apart. One store instruction becomes 32 sectors
instead of 4. The 8× sector inflation shows up as 3.1× wall clock rather than 8×
because the L2 absorbs some of the re-reads across warps.

Module 5 replaces this hand-waving with exact transaction counts and the Nsight
Compute metrics that report them.

## Expected output

```
=== Module 3 / Exercise 1 SOLUTION : 5-point stencil, 2D thread mapping ===

=== image 1021 x 733 (row-major, stride 733, 3.0 MB per array) ===

--- block (32,8) : grid (23,128,1), 753664 threads for 748393 pixels (5271 idle) ---
  block (32,8)       PASS  (all 1021 x 733 = 748393 elements match)
  block (32,8)         0.0107 ms     559.5 GB/s effective   129.5% of peak

--- block (8,32) : grid (92,32,1), 753664 threads for 748393 pixels (5271 idle) ---
  block (8,32)       PASS  (all 1021 x 733 = 748393 elements match)
  block (8,32)         0.0150 ms     399.1 GB/s effective    92.4% of peak

--- block (256,1) : grid (3,1021,1), 784128 threads for 748393 pixels (35735 idle) ---
  block (256,1)      PASS  (all 1021 x 733 = 748393 elements match)
  block (256,1)        0.0168 ms     355.4 GB/s effective    82.3% of peak

--- block (1,256) : grid (733,4,1), 750592 threads for 748393 pixels (2199 idle) ---
  block (1,256)      PASS  (all 1021 x 733 = 748393 elements match)
  block (1,256)        0.0630 ms      95.0 GB/s effective    22.0% of peak

=== image 4093 x 3079 (row-major, stride 3079, 50.4 MB per array) ===

--- block (32,8) : grid (97,512,1), 12713984 threads for 12602347 pixels (111637 idle) ---
  block (32,8)       PASS  (all 4093 x 3079 = 12602347 elements match)
  block (32,8)         0.3213 ms     313.8 GB/s effective    72.6% of peak

--- block (8,32) : grid (385,128,1), 12615680 threads for 12602347 pixels (13333 idle) ---
  block (8,32)       PASS  (all 4093 x 3079 = 12602347 elements match)
  block (8,32)         0.3275 ms     307.8 GB/s effective    71.3% of peak

--- block (256,1) : grid (13,4093,1), 13621504 threads for 12602347 pixels (1019157 idle) ---
  block (256,1)      PASS  (all 4093 x 3079 = 12602347 elements match)
  block (256,1)        0.3212 ms     313.9 GB/s effective    72.7% of peak

--- block (1,256) : grid (3079,16,1), 12611584 threads for 12602347 pixels (9237 idle) ---
  block (1,256)      PASS  (all 4093 x 3079 = 12602347 elements match)
  block (1,256)        1.0022 ms     100.6 GB/s effective    23.3% of peak

OVERALL: PASS
```

If TODO 3 is left blank the program exits cleanly with
`makeGrid() returned a zero dimension. Fill in TODO 3 first.`

## The result that matters

The index expression and the grid expression are the *same decision written
twice*, and they must agree: whichever axis `threadIdx.x` carries, `gridDim.x`
must be sized from that axis's extent. Getting the pair consistent-but-swapped
(row on x, and the grid sized from `h`) produces a kernel that is *correct* and
several times slower; getting them inconsistent produces a kernel that is wrong
in a way no amount of staring at a small square test case will reveal — which is
exactly why the image is 1021 × 733.

Variation to try: change `makeGrid` to always return `dim3(gx, gy, 1)` computed
from `h` and `w` *swapped*, and watch how the 1021 × 733 case fails while a
hypothetical 1024 × 1024 case would have passed. Then add a fifth block shape
`dim3(64, 4)` and predict its time before running it.
