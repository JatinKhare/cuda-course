# Module 21 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
exercise03_solution.exe
```

Warning-clean. `SCORE: 9/9`, `OVERALL: PASS`. About 20 s.

---

## The design, derived

The brief is "28% of the FP32 ceiling and 2.5x the naive kernel". Nothing in the
brief names an optimization. Here is the chain that produces one.

**Step 1: which level binds?** The naive kernel's compulsory DRAM traffic is one
read and one write of a 16 MB image — 32 MB for 2.42 GFLOP of work, an
`AI(DRAM)` of 75 FLOP/byte, right of the 48 FLOP/byte ridge. So DRAM does not
bind. The request level does: 289 taps per output, each a 4-byte load.

**Step 2: `AI(request)` of the naive kernel.**

```
per tap: 1 load of 4 bytes, 1 fused multiply-add = 2 FLOP
AI = 2/4 = 0.5 FLOP/byte
```

The subtle part is the *weight*. `cW[...]` is `__constant__` and the loops are
fully unrolled, so the index is a compile-time constant and `ptxas` folds the
weight into the FFMA as a constant-bank operand:

```
FFMA R35, R6, c[0x3][0x0], RZ ;
```

No `LDC`, no second load. Had the weights been in shared memory the intensity
would be `2/8 = 0.25` and the naive kernel would be half as fast. Readers who
assume a weight load get `AI = 0.25`, predict 6.7% of the ceiling, measure 12.8%,
and are a factor of two out in the *baseline* before they design anything.

**Step 3: the ridge and the target.**

```
ridge      = 19 482 GFLOP/s / 5277 GB/s = 3.69 FLOP/byte
naive      = 0.50 / 3.69                = 13.5% of the ceiling   (measured 12.8%)
AI needed  = 0.28 x 3.69                = 1.03 FLOP/byte
P needed   = ceil(1.03 / 0.50)          = 3 outputs per thread
```

**Step 4: what moves `AI(request)`?** Only one thing: reusing a loaded value for
more than one fused multiply-add. A thread that computes `P` outputs and uses
each loaded value for all `P` of them has `AI = 0.5P`. That is register
blocking, and it is forced by the arithmetic, not chosen by taste.

**What does NOT move it, and this is the exercise's first trap:** staging the
input through shared memory. A shared tile changes *which* cache answers the
load; it does not change the number of loads per FLOP. The intensity stays at
0.5 and so does the performance. Measured, with the same kernel at `OPT = 1`:

| kernel | GFLOP/s | % of ceiling | x naive |
|---|---|---|---|
| naive (global loads, no tile) | 2488 | 12.8% | 1.00 |
| **shared tile, 1 output/thread** | **2375** | **12.1%** | **0.96** |
| shared tile, 2 outputs/thread | 4313 | 22.6% | 1.76 |
| **shared tile, 4 outputs/thread** | **7558** | **38.8%** | **3.04** |
| shared tile, 8 outputs/thread | 1870 | 9.8% | 0.76 |

The row everybody writes first is a **4% loss**. This is Module 6's finding in a
new setting — "the caches got there first" — and Module 17's hand-off to Module
18 restated: changing the address space an operand comes from is not the same as
changing how many operands you fetch.

**The second trap is the last row.** `OPT = 8` is a 24% *loss* against the
naive kernel. More blocking is not monotonically better. The halo tax grows
(`(TILE_Y·OPT + 2R)` rows of tile for `TILE_Y·OPT` rows of output: 80/64 at
OPT=8 against 48/32 at OPT=4), the shared footprint grows with it, and the 8
accumulators plus the unrolled `2R+OPT = 24` row loop push the register count
past what lets blocks co-reside. This is M18's occupancy cliff with a different
resource doing the cutting, and M18's lesson stands: the optimum is an interior
point and it is not the maximum of the thing you are optimizing.

---

## TODO 1 — the analysis

```cpp
static double aiNaive(void) { return 2.0/4.0; }

static double aiNeeded(double frac, double ceilFp32GF, double ceilOnChipGBs)
{ return frac * (ceilFp32GF / ceilOnChipGBs); }

static int minOutputsPerThread(double needed, double naive)
{ int p = (int)ceil(needed/naive); return p < 1 ? 1 : p; }
```

`aiNeeded` is "frac of the ridge point", which falls straight out of
`attainable = AI x S` and `target = frac x P`:

```
frac x P = AI x S   =>   AI = frac x P/S = frac x ridge
```

`minOutputsPerThread` must round **up**. Rounding down gives `P = 2` here, which
measures 22.6% against a 28% target: the right analysis with the wrong rounding
fails the gate by 5 points.

## TODO 2 — the decomposition (design)

```cpp
#define TILE_X 32
#define TILE_Y 8
#define OPT    4
```

- `TILE_X = 32` so a warp's 32 lanes cover 32 consecutive columns in the
  cooperative load, which is M5's coalescing requirement on the one part of this
  kernel that touches global memory.
- `TILE_Y = 8` gives 256 threads, the course default, and 6 blocks/SM at
  9216 bytes of shared memory.
- `OPT = 4` is one above the minimum the analysis demands (3), which buys margin
  against the 28% gate without reaching the cliff at 8.

Resources, from `cudaFuncGetAttributes`: **39 registers, 0 spilled, 9216 B
shared, 6 blocks/SM**. Note that the naive kernel uses 40 registers — the fast
kernel is not paying for its 4 accumulators in any visible way, because
`ptxas` was already spending that many on the unrolled 289-tap loop.

`TILE_X = 32, TILE_Y = 4, OPT = 4` measures within 1% and is equally acceptable.

## TODO 3 — the cooperative load

```cpp
    for (int idx = tid; idx < SH_H*SH_W; idx += nthr) {
        int r = idx / SH_W, c = idx % SH_W;
        int yy = min(max(y0 + r - RAD, 0), IH-1);
        int xx = min(max(x0 + c - RAD, 0), IW-1);
        s[r][c] = in[(size_t)yy*IW + xx];
    }
    __syncthreads();
```

The tile is 48 x 48 = 2304 cells and there are 256 threads, so the load mapping
cannot be the compute mapping (M6). The flat strided loop
`for (idx = tid; idx < N; idx += nthr)` is the shape that works for any ratio
and keeps consecutive `idx` on consecutive `c`, so the global reads coalesce.

The clamp is the same one the naive kernel uses, which is what makes the two
kernels comparable at all. A reader who skips the clamp and relies on the image
being large reads out of bounds on the first and last tile rows —
`compute-sanitizer --tool memcheck` catches it, and nothing in the numerical
validator would.

The barrier is a RAW hazard across threads and needs both of M9's guarantees. It
is block-uniform.

## TODO 4 — the accumulation

```cpp
    float acc[OPT];
    #pragma unroll
    for (int q = 0; q < OPT; ++q) acc[q] = 0.f;
    const int lx = threadIdx.x, ly = threadIdx.y;
    #pragma unroll
    for (int r = 0; r < 2*RAD + OPT; ++r) {
        const int sr = ly*OPT + r;
        #pragma unroll
        for (int dx = 0; dx < DIAM; ++dx) {
            const float v = s[sr][lx + dx];
            #pragma unroll
            for (int q = 0; q < OPT; ++q) {
                const int wy = r - q;
                if (wy >= 0 && wy < DIAM) acc[q] = fmaf(v, cW[wy*DIAM + dx], acc[q]);
            }
        }
    }
```

**The loop order is the entire exercise.** The two outer loops walk the *loaded
value*; the innermost loop walks the *accumulators*. One shared read feeds up to
`OPT` fused multiply-adds, which is what `AI = 0.5 x OPT` means in code.

Write it the obvious way instead —

```cpp
    for (int q = 0; q < OPT; ++q)
        for (int dy = 0; dy < DIAM; ++dy)
            for (int dx = 0; dx < DIAM; ++dx)
                acc[q] = fmaf(s[ly*OPT+q+dy][lx+dx], cW[dy*DIAM+dx], acc[q]);
```

— and every FFMA has its own `LDS` again. The intensity is back to 0.5, the
kernel measures 12%, and you have written the shared-tile row of the table
above with four times as much code. It is correct, it validates, and it is the
most instructive wrong answer in this module.

Two details:

- The `if (wy >= 0 && wy < DIAM)` guard is resolved entirely at compile time
  because `r` and `q` are both unrolled compile-time constants. It costs
  nothing; it simply causes `ptxas` not to emit the FFMAs that would be
  out of filter range. The average reuse is therefore
  `OPT x DIAM / (DIAM + OPT - 1) = 4 x 17 / 20 = 3.4`, not 4, so the achieved
  `AI` is `0.5 x 3.4 = 1.7` rather than 2.0. Predicted 1.7 x 5277 = 8971
  GFLOP/s against the compute ceiling; measured 7558, i.e. 84% of the roof, with
  the loader and the epilogue making up the difference.
- `#pragma unroll` on the `r` loop is required for the guard to fold and for
  `cW` to become a constant-bank operand. Without it the kernel is 2x slower
  and the SASS shows `LDC` instructions.

## TODO 5 — the prediction

`PRED_BUCKET = 4` (28–50%). Measured 37.9–38.8%. The model says `1.7 x ridge
fraction = 1.7/3.69 = 46%`, the loader and the epilogue cost some of that, and
28–50% is the bucket that survives the whole observed range.

---

## The SASS, which settles the whole argument

```
cuobjdump -sass exercise03_solution.cubin
```

Full-kernel instruction census (both kernels are fully unrolled, so the whole
body is the loop):

| | `convNaive` | `convFast` (OPT=4) |
|---|---|---|
| total instructions | 1592 | 1648 |
| `FFMA` | 289 | **1156** |
| `LDG` | **289** | 0 (in the compute phase) |
| `LDS` | 0 | **340** |
| `LEA` / `LOP3` (address arithmetic) | 580 / 289 | 23 / 0 |
| FFMA density | **18%** | **70%** |
| `LDC` (weight loads) | 1 | 1 |

Three readings.

1. **Every one of the 289 FFMAs in `convNaive` carries a constant-bank
   operand** — `FFMA R35, R6, c[0x3][0x0], RZ` — so the weight genuinely costs
   no load, and `AI = 2/4 = 0.5` is right. The single `LDC` is in the prologue.
2. **`convFast` performs 1156 FFMAs against 340 shared reads.** That is
   `1156 x 2 / (340 x 4) = 1.70 FLOP/byte`, which is exactly the
   `0.5 x OPT x DIAM/(DIAM+OPT-1) = 0.5 x 3.4` the design predicted. The
   intensity is not an estimate; it is countable in the disassembly.
3. **The naive kernel's FFMA density is 18%,** giving an issue plateau of
   `310 G instr/s x (289 x 64 / 1592) = 3602 GFLOP/s`, against an on-chip
   plateau of `0.5 x 5277 = 2638`. The on-chip ceiling binds, but only by 1.4x
   — and 580 `LEA` plus 289 `LOP3` of clamped-index arithmetic are what put the
   issue ceiling that close. Hoisting the clamp (M6's `example02` trick) would
   push the issue plateau up and change nothing, because it is not the binding
   ceiling. Knowing *which* ceiling binds is what stops you doing that work.

---

## Synchronization / memory reasoning

One `__syncthreads()`, after the cooperative load, before the read loop: RAW
across threads, needs G1 and G2 (M9). There is no WAR hazard because the tile is
written once and never reused across iterations — the kernel has no outer tile
loop — so no second barrier is needed. If you extend this to a tiled sweep over
the image you acquire the second barrier and M17's double-buffering question
with it.

No bank conflicts: `s[sr][lx + dx]` at fixed `(r, dx)` has `lx` running 0..31
over consecutive floats, which is degree 1. The tile pitch is `SH_W = 48` and
`gcd(48, 32) = 16`, which would matter if any access walked the *column*; none
does.

---

## Performance reasoning

The gate is deliberately two-sided: 2.5x over the shipped baseline **and** 28%
of the measured ceiling. The ratio gate is the stable one (M14's finding: ratios
between the reader's own kernels survive thermal state better than percentages
of a separately-measured ceiling), and the fraction-of-ceiling gate is the one
the brief is phrased in. Measured margins: 3.04x against 2.5, and 38.8% against
28. Both exceed spec §12 rule 5d's 8-point requirement.

Reproducibility across five runs: naive 2445–2497 GFLOP/s (12.3–12.8% of
ceiling), fast 7242–7558 (37.2–38.8%), ratio 2.94–3.07. The tightest of those
margins is 9 points on the fraction gate.

---

## Expected output

Real output, one good run:

```
-- the design brief ---------------------------------------------------
  2048x2048 image, 17x17 filter (289 taps), 2.42 GFLOP of useful work
  measured FP32 ceiling             19482.2 GFLOP/s
  measured on-chip fetch ceiling     5276.7 GB/s
  on-chip ridge                        3.69 FLOP/byte
  naive kernel AI                      0.50 FLOP/byte -> 13.5% of the ceiling
  TARGET 28% of the ceiling needs AI >= 1.03 FLOP/byte, i.e. at least
  3 outputs per thread. You chose TILE_X=32 TILE_Y=8 OPT=4.

-- measurement --------------------------------------------------------
  kernel             ms      GFLOP/s   %ceiling    x naive
  naive          0.9745       2487.7      12.8%       1.00
  yours          0.3208       7557.6      38.8%       3.04
  yours: 39 registers, 0 B spilled, 9216 B shared, 6 blocks/SM

-- validation (second, untimed pass) ----------------------------------
  pixels your kernel never wrote : 0
  sampled scaled error (1.0 = at tolerance): 0.0131, 0 samples over
  max |yours - naive| over the whole image : 0.000e+00

-- scoring -----------------------------------------------------------
  [x] 1. aiNaive / aiNeeded / minOutputsPerThread (2 pts)
  [x] 2. OPT (4) is at least what your own analysis demands (3)
  [x] 3. correct: every pixel written, inside tolerance (2 pts)
  [x] 4. at least 2.5x the naive kernel (got 3.04x) (2 pts)
  [x] 5. at least 28% of the measured FP32 ceiling (got 38.8%)
  [x] 6. predicted bucket 4, measured bucket 4

SCORE: 9/9
OVERALL: PASS
```

The `max |yours - naive|` of exactly zero is luck worth not relying on: both
kernels accumulate the 289 taps in the same order (row-major over the filter),
so the floating-point results are bit-identical here. Change `OPT` and the
accumulation order within a thread changes, and the difference becomes a few
ULP. The sampled exact-double check with the `gamma_n x S` tolerance (M16) is
what actually validates the kernel; the bitwise comparison is a bonus.

---

## The result that matters

The roofline did not tell you to use shared memory, and it was right not to:
staging the input through shared memory is a 4% loss here, because it moves
operands to a different address space without changing how many of them each
FLOP needs. What the roofline told you is the number 1.03 FLOP/byte, and the
only transformation that produces it is reusing each loaded value across
multiple outputs. That is a change to the *decomposition* — how work is assigned
to threads — not to the code. Module 6 named this, Module 18 proved it for GEMM,
and here it is on a completely different kernel with the same arithmetic.

**Variation to try.** This filter is a general 17x17; if yours is separable
(`w[dy][dx] = u[dy] v[dx]`), two passes of 17 taps replace one pass of 289 and
the work drops **8.5x**. Implement it and watch what happens to the scoreboard:
the kernel finishes in a fraction of the time and its *percentage of the FP32
ceiling goes down*, because separability reduces `F` without changing `AI`. The
roofline measures a rate, and a target expressed as a fraction of peak will
reject the best algorithm in the room. That is not a flaw in the roofline; it is
a flaw in choosing a rate as a goal.
