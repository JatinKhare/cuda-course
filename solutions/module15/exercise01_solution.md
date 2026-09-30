# Module 15 / Exercise 01 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

Optional, and worth doing:

```
nvcc -arch=sm_89 -O3 -Xptxas -v -c -o exercise01_solution.o exercise01_solution.cu
cuobjdump -sass exercise01_solution.o | findstr "LDS STS LDG STG"
```

---

## TODO 1 — the naive kernel's two index expressions

```cpp
long long src = (long long)(y + j) * W + x;   // 4 sectors per warp
long long dst = (long long)x * H + (y + j);   // 32 sectors per warp
```

**Why it is correct.** `in` has `W` columns, so input element (row `y+j`, column
`x`) is at `(y+j)·W + x`. `out` has `H` columns — it is the transpose, so its
width is the input's height — and the element that was at input `(y+j, x)`
belongs at output row `x`, column `y+j`, hence `x·H + (y+j)`.

**Why the sector counts are what they are.** A warp is 32 consecutive
`threadIdx.x` at fixed `threadIdx.y` (M3's linearization rule), so `x` runs over
32 consecutive values and `y+j` is constant.

- `src`: addresses `4·((y+j)·W + x₀ + L)` for `L = 0..31` — 128 contiguous bytes.
  `x₀` is a multiple of 32 so the run starts on a sector boundary. `128/32 = 4`
  sectors, 100% efficient.
- `dst`: addresses `4·((x₀+L)·H + y+j)` — consecutive lanes `4H = 32768` bytes
  apart. Each lane owns a sector nobody else touches: **32 sectors** for 128
  useful bytes, 12.5% efficient. That is M5's saturation floor, reached by any
  stride of 8 floats or more.

**Common wrong approaches.**

- `out[x*W + y]` — the leading dimension of the *output* is `H`, not `W`. Passes
  on every square matrix, silently shears the result on 4093 × 2049. The harness
  runs both, which is the only reason you would see it.
- Omitting the `(long long)` cast. `x * H` with `x, H` both `int` overflows at
  `2^31` elements; 8192 × 8192 = 67 M is safe, but the habit is not. The cast is
  on the *first* operand so the whole expression promotes.
- Writing `dst` as `(y+j)*H + x` — that is a copy with the wrong pitch, not a
  transpose.

---

## TODO 2 — the tiled kernel

```cpp
// 2a
tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * W + x];

// 2b
__syncthreads();

// 2c
int xo = blockIdx.y * TILE + threadIdx.x;   // column of out, in [0,H)
int yo = blockIdx.x * TILE + threadIdx.y;   // row    of out, in [0,W)

// 2d
out[(long long)(yo + j) * H + xo] = tile[threadIdx.x][threadIdx.y + j];
```

**Why it is correct, phase by phase.**

*Load.* The block owns input tile `(blockIdx.x, blockIdx.y)`. Thread
`(tx, ty)` on iteration `j` reads input row `y+j`, column `x`, and must put it at
the tile-local coordinate that names the same element: row `ty+j`, column `tx`.
The global read has `tx` in the fastest position, so it is 4 sectors.

*Barrier.* The value thread `(tx, ty)` will store in the second phase was written
by a *different* thread — specifically by the thread whose `threadIdx.x` equals
this thread's tile row. Those threads are in different warps. Both of Module 9's
guarantees are required: the execution barrier (the writer must have run) and the
block-scope memory fence (the write must be visible). `__syncwarp()` is not
enough and `volatile` is not synchronization (M9).

*Output coordinates.* The transpose of input tile `(bx, by)` is output tile
`(by, bx)`. The output is `H` wide, so its column index runs over `[0, H)` and
must be the fast axis: `xo = blockIdx.y·TILE + threadIdx.x`. Its row index runs
over `[0, W)`: `yo = blockIdx.x·TILE + threadIdx.y`. **The block indices swap
across the barrier.** This is the step people skip; it is what makes the global
write coalesced.

*Store.* Thread `(tx, ty)` on iteration `j` writes output cell `(yo+j, xo)`.
Output `(r, c)` holds input `(c, r)`, so this cell holds input element
`(xo, yo+j)`. The tile is indexed by (input row within the tile, input column
within the tile). For that element those are

```
input row within tile    = xo     - blockIdx.y*TILE = threadIdx.x
input column within tile = (yo+j) - blockIdx.x*TILE = threadIdx.y + j
```

Hence `tile[threadIdx.x][threadIdx.y + j]`.

That paragraph is deliberately laborious because the shortcut — "just swap the
subscripts" — is how people end up with a kernel that is wrong in a way squares
cannot see.

**Common wrong approaches, and the symptom of each.**

| mistake | symptom |
|---|---|
| `out[(yo+j)*W + xo]` | passes on every square matrix, wrong on 4093 × 2049; values are right, positions are sheared by `r·(W-H)` |
| `xo = blockIdx.x*TILE + tx` (no swap) | wrong everywhere, obviously — the easy one |
| `tile[ty+j][tx]` in *both* phases | a copy, not a transpose; wrong everywhere |
| `tile[tx][ty+j]` in *both* phases | a correct transpose with the conflict on the **store** instead of the load; measures identically (verified: 0.92–1.13× at 8192², 0.97× at 2048²). Not a bug, but not what the exercise asked for. |
| store guard `if (x < W && y+j < H)` reused for the output | drops or duplicates output cells on non-square matrices |
| `return` instead of guarding the statement | a thread that returns before `__syncthreads()` breaks M9's uniformity rule; on sm_89 `bar.sync` counts warps and subtracts exited threads, so this usually corrupts silently rather than hanging |

---

## TODO 3 — designing a conflict-free layout

Two answers satisfy every constraint the harness checks.

**Padding**, which is what the reference solution ships:

```cpp
static const int SH_FLOATS = 32 * 33;                 // 1056, the stated limit
__host__ __device__ int shIdx(int r, int c) { return r * 33 + c; }
```

`bank(r,c) = (33r + c) mod 32 = (r + c) mod 32`. Row phase (`r` fixed, `c`
varying): a permutation of 0..31, `D = 1`. Column phase (`c` fixed, `r` varying):
also a permutation, `D = 1`. The general condition is `gcd(pitch, 32) == 1`,
i.e. **the pitch must be odd** — the exact opposite of every alignment rule M5
gave you for global memory.

**XOR swizzle**, which uses 1024 floats instead of 1056:

```cpp
static const int SH_FLOATS = 32 * 32;
__host__ __device__ int shIdx(int r, int c) { return r * 32 + (c ^ (r & 31)); }
```

`bank(r,c) = (32r + (c ^ r)) mod 32 = c ^ r`. XOR by a fixed value is a bijection
on `[0,32)`, so both phases are permutations and both are `D = 1`. It is
injective because for fixed `r` the map `c ↦ c ^ r` is a bijection, and different
`r` land in disjoint 32-float blocks.

Both score 1/1. They measure the same (see below).

**Common wrong approaches.**

- `pitch = 40` ("a multiple of 8, nicely aligned for `float4`"). `gcd(40,32) = 8`,
  so the column phase is `D = 8`. This is M7 exercise 2's trap and M5 exercise
  3's pitch-68 trap in their third incarnation. The harness prints the degree and
  rejects it.
- `pitch = 34`. `gcd(34,32) = 2`, so `D = 2` — which the harness **accepts**,
  because Module 7 measured the cost law on Ada as `max(2, D)` and a 2-way
  conflict is genuinely free. If you submitted 34, you were right, and you also
  used 1088 floats, which exceeds the 1056 limit; `[32][34]` is 1088. So it does
  not fit. `[32][33]` does.
- Forgetting that `shIdx` is called from the **host** too (the harness simulates
  both phases before launching anything). Marking it `__device__` only will not
  compile.
- Returning a negative offset for some `(r,c)` — the harness's range test catches
  it before any kernel runs, which is the point of doing the check on the host.

---

## Synchronization / memory reasoning

One barrier, between the two phases, outside all control flow. It is required for
both of M9's guarantees. It is *not* required at the end of the kernel: the
kernel exit is itself a release for the block's writes to global memory, and
nothing reads the tile afterwards.

The barrier costs nothing measurable here. M6's convoying argument says a block
runs at the speed of its slowest warp at each barrier; in a transpose all eight
warps of a 256-thread block do exactly four coalesced loads each, so there is no
warp-level slack to expose. The measured cost of the *entire* shared-memory round
trip — 4 `STS`, `BAR.SYNC`, 4 `LDS` — is **0.997× of a plain copy**, i.e. zero.

A thread whose `(x, y+j)` lies outside the input must still execute the barrier.
That is why the guard wraps the assignment and never the control flow around it.

---

## Performance reasoning

Observed on the RTX 3500 Ada, CUDA 13.2, 8192 × 8192, min of 8 rotated sweeps:

```
TIMING at 8192 x 8192 (min of 8 rotated sweeps)
version                           ms      GB/s   %ofcopy    bucket
copy ceiling                  1.4330     374.7    100.0%         3
v1 coal read / str write      4.1831     128.3     34.3%         2
v2 str read / coal write      2.1307     252.0     67.3%         2
v3 tiled, plain 32x32         1.5381     349.0     93.2%         3
v4 tiled, your layout         1.4935     359.5     95.9%         3
```

- **v1 at 34% of copy here, 29-49% across runs.** The sector model says the write moves 8× its useful
  bytes, predicting 2/9 = 22%. The measurement is better because the 32 sectors a
  warp touches are reused by the block's other warps and by the other three
  iterations of the `j` loop before they are evicted from L2; the real
  amplification is nearer 4× than 8×.
- **v2 at 67% of copy.** Same permutation, strided on the *read* side instead.
  A strided write costs about twice a strided read, because a partial-sector
  store forces a read-merge-write at L2 (M5 asserted it, M11 measured it at
  3.95×). **When you must break one side, break the reads.**
  If you remove the bounds guard from v2 it jumps to 88% of copy — the guard
  stops the compiler hoisting all four loads ahead of the stores and costs 1.34×
  in memory-level parallelism. The same guard on v3/v4 costs 0.3%.
- **v3 at 93% and v4 at 96%.** Removing a genuine 32-way bank conflict is worth
  **1.03×** here, which is inside the noise. Observed across all authoring runs:
  0.96× to 1.03× -- in some runs the *conflicted* version is faster. There is no
  signal at this scale.

Then the program re-times the same two kernels on an L2-resident matrix:

```
THE SAME TWO KERNELS ON AN L2-RESIDENT 2048x2048 MATRIX
  (not a DRAM bandwidth measurement -- both buffers fit in L2)
  v3 plain 32x32 : 0.0725 ms
  v4 your layout : 0.0264 ms
  v3 / v4        : 2.739x
```

**2.7× here, 2.7-3.3× across runs.** Nothing about the kernels changed. The binding constraint did. This is
the exercise.

The arithmetic, if you want it: per thread the tiled kernel issues 4 `LDG`,
4 `STG`, 4 `STS`, 4 `LDS`. At `D = 32` the four `LDS` occupy the shared pipeline
for 32 cycles each instead of 2, adding 120 cycles per thread ≈ 960 cycles per
256-thread block. The same block moves 32 KB through DRAM, which at 373 GB/s over
40 SMs takes ≈ 3.5 µs ≈ 7000 cycles at 2 GHz. The shared term fits inside the
memory term with room to spare. Put the data in L2, the memory term drops ~4.5×
(apparent bandwidth 1671 GB/s versus 373), and the shared term is suddenly the
one you are waiting for.

If instead of padding you submitted the XOR swizzle, you will see essentially the
same numbers. Measured `swizzle / padded` on the L2-resident 2048² case over seven runs:
0.92, 0.97, 1.02, 1.03, 1.06, 1.06, 1.22 — a tie. M7 measured padding winning by 19% on
a kernel that did 32 × 64 shared accesses per thread and nothing else; the
transpose does 8, against 8 global accesses, so the swizzle's 8 extra `LOP3`
instructions are invisible.

---

## Expected output

```
Module 15 exercise 01 -- build the transpose ladder
GPU: NVIDIA RTX 3500 Ada Generation Laptop GPU, CC 8.9, L2 = 50.3 MB

TODO 3 structural check: SH_FLOATS = 1056 (limit 1056), in range yes, injective yes
  store phase (row across a warp)  degree = 1  (need <= 2)
  load  phase (column across warp) degree = 1  (need <= 2)
  for reference, v3's plain 32x32 tile: store degree 1, load degree 32

TIMING at 8192 x 8192 (min of 8 rotated sweeps)
version                           ms      GB/s   %ofcopy    bucket
copy ceiling                  1.4330     374.7    100.0%         3
v1 coal read / str write      4.1831     128.3     34.3%         2
v2 str read / coal write      2.1307     252.0     67.3%         2
v3 tiled, plain 32x32         1.5381     349.0     93.2%         3
v4 tiled, your layout         1.4935     359.5     95.9%         3

VALIDATION
version                         2048x2048      4093x2049
copy ceiling                           ok             ok
v1 coal read / str write               ok             ok
v2 str read / coal write               ok             ok
v3 tiled, plain 32x32                  ok             ok
v4 tiled, your layout                  ok             ok

THE SAME TWO KERNELS ON AN L2-RESIDENT 2048x2048 MATRIX
  (not a DRAM bandwidth measurement -- both buffers fit in L2)
  v3 plain 32x32 : 0.0725 ms
  v4 your layout : 0.0264 ms
  v3 / v4        : 2.739x

SCORING
  numerics, all four versions, both matrices  : 4/4
  TODO 3 layout structurally valid            : 1/1
  PRED[0] = 2, measured bucket 2 : correct
  PRED[1] = 2, measured bucket 2 : correct
  PRED[2] = 3, measured bucket 3 : correct
  PRED[3] = 3, measured bucket 3 : correct
  bucket predictions                          : 4/4
  PRED[4] = 1, measured v3/v4 = 1.030x -> bucket 1 : correct

SCORE: 10/10
OVERALL: PASS
```

The correct predictions are `PRED = { 2, 2, 3, 3, 1 }`.

**Variance warning.** This is a 75 W laptop GPU. Absolute times move by up to
2.5× between runs depending on thermal and power state; during authoring the same
binary produced a copy ceiling of 373 GB/s in one run and 98 GB/s twenty minutes
later, with `nvidia-smi` reporting the software power cap and the memory clock
stepping down from 8801 to 8001 MHz. The bucket boundaries (20% and 85%) were
re-placed during post-authoring repair to survive that: across every repeat run
v1 measured 34–64% of copy, v2 72–76%, v3 93–99%, v4 94–101%, so no version is
within 8 points of a boundary. v1 and v2 therefore share a bucket. That is
deliberate: the 8-point gap between their ranges is smaller than the run-to-run
spread of v1 alone, so "v1 is slower than v2" is a claim this machine can *show*
you in the `%ofcopy` column but not one it can *score* reproducibly. Read the
column, not just the bucket.

The harness also refuses to score a deeply power-limited run. After several
minutes of back-to-back benchmarking this part pins its memory clock at 6001 MHz
instead of 9001 with `SW_POWER_CAP` and `SW_THERMAL_SLOWDOWN` asserted, and in
that state v3 falls from 98% of the copy to 81% — the ratios stop being this
GPU's ratios at all. After warming up, the harness probes the copy ceiling; if it
comes back under 200 GB/s it idles for 10 s and warms again, up to five times.

---

## The result that matters

A transpose is a copy with the addresses permuted, so a copy of the same matrix
is its speed of light — and once you measure against that denominator, the
familiar optimizations sort themselves into the ones that move the binding
constraint and the ones that do not. Staging through shared memory is worth
**2.2×** because it converts a 32-sector write into a 4-sector write; removing
the 32-way bank conflict that staging introduces is worth **1.03×**, because the
kernel is waiting for DRAM and the shared-memory pipeline has slack it is not
using. Shrink the matrix until it fits in L2 and the same conflict is worth
**2.7-3.3×**. The optimization did not change; the bottleneck did.

**Variation to try.** Keep v4 and change nothing but the tile: `[64][65]` with a
`(64,16)` block. Shared memory goes from 4224 B to 16640 B and resident blocks
from 6 to 1, which every occupancy heuristic says should hurt — and it measures
**96.5–100.3% of the 32-wide copy and 97.6–100% of its own matched 64-wide
copy** — at the ceiling in every run, the best shape in this module. Work out why before you run it: the answer is about how
many contiguous bytes a warp presents to the DRAM, not about how many warps are
resident.
