# Module 06 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

---

## TODO 1 — the cooperative load, halo included

```cpp
for (int idx = tid; idx < SW * SH; idx += nthr) {
    int ly = idx / SW;
    int lx = idx - ly * SW;
    int gy = clampi(row0 + ly - 1, 0, h - 1);
    int gx = clampi(col0 + lx - 1, 0, w - 1);
    tile[idx] = in[(size_t)gy * w + gx];
}
```

**Why it is correct.** Three things have to hold at once and this form gets all
three for free.

*Coverage.* `idx` runs over `[tid, SW*SH)` in steps of `nthr`, and the union over
all `tid ∈ [0, nthr)` is exactly `[0, SW*SH)`, each value once. For
`TW=32, TH=8`: 340 cells, 256 threads, so threads 0–83 run the loop twice and the
rest once. For `TW=32, TH=16`: 612 cells, 512 threads — same structure. For
`TW=64, TH=8`: 660 cells, 512 threads. No shape-specific code.

*Coalescing.* Consecutive `idx` map to consecutive `lx` within a tile row, which
map to consecutive global columns, which are adjacent in memory. Lane `L` of a
warp supplies address `base + 4L` (except across the `SW`-boundary wrap). By
Module 5's counting procedure that is 4 sectors per 32 lanes at 100 % efficiency,
apart from one extra sector per tile row from the 2-cell halo overhang. Write the
loop the other way —

```cpp
int per = (SW*SH + nthr - 1)/nthr;
for (int k = 0; k < per; ++k) { int idx = tid*per + k; ... }   // WRONG
```

— and lane `L` supplies `base + 4·per·L`, i.e. a stride of 8 bytes at `per=2`.
That is Module 5's stride-2 case: 50 % efficiency, twice the sectors, same
instruction count. It still validates.

*The clamp.* One `clampi` on each axis handles two different problems with one
expression: genuine image-boundary halo (tile row `ly=0` of the topmost block
refers to image row −1) and the partial-tile case (a right-edge block whose
interior columns run past `w-1`). Both need the value of the nearest real pixel,
which is exactly what the untiled formula computes, so the tiled and untiled
kernels agree bit for bit.

**Common wrong approaches.**

| approach | symptom |
|---|---|
| 1:1 load `tile[(ty+1)*SW + tx+1] = in[row*w+col]`, halo unloaded | all four edge strips of every tile read uninitialised SRAM: plausible-looking but wrong values, ~15 % of pixels wrong, changes run to run |
| 1:1 interior + four `if (ty==0)/(ty==TH-1)/(tx==0)/(tx==TW-1)` special cases | **passes.** The four corner cells are never loaded, but a 5-point stencil never reads a corner. This is a legitimate and slightly cheaper load for *this* stencil; it breaks the moment the stencil becomes 9-point. |
| loading with the interior guard `if (row<h && col<w)` still present | right-edge and bottom-edge stripes wrong on the 1021×733 image, correct on any image whose dimensions divide the tile |
| contiguous-chunk-per-thread loop | validates; runs measurably slower; nothing reports why |

---

## TODO 2 — the barrier

```cpp
__syncthreads();
```

Thread (0,0) computes pixel `(row0, col0)`, which sits at tile cell `(1,1)`, and
reads cells `(0,1)`, `(2,1)`, `(1,0)`, `(1,2)`. With `TW=32, TH=8` and the flat
loop above:

- cell `(0,1)` is `idx = 1`, written by thread 1 — same warp;
- cell `(1,0)` is `idx = 34`, written by thread 34 — **warp 1**;
- cell `(2,1)` is `idx = 69`, written by thread 69 — **warp 2**;
- cell `(1,2)` is `idx = 35`, written by thread 35 — warp 1.

So three of thread (0,0)'s four neighbours are written by warps other than its
own. Nothing in the hardware orders warp 0's read after warp 1's write except
the barrier. This is also why the bug would be invisible if you shrank the block
to a single warp — the subject of Exercise 3.

Note there is **no** second barrier needed here: the tile is written once and
read once, and the kernel ends. Rule 2 (after reading, before overwriting) has
nothing to bite on. A reader who adds a second `__syncthreads()` after the
compute loop is not wrong, just slower — and would be *incorrect* if they put it
after the `if (row >= h || col >= w) return;`, where it sits in divergent control
flow. (Module 9.)

---

## TODO 3 — computing from shared memory

```cpp
const int lx = (int)threadIdx.x + 1;
const int ly = (int)threadIdx.y + 1;

float acc = 4.0f * tile[ly * SW + lx]
          +        tile[(ly - 1) * SW + lx]
          +        tile[(ly + 1) * SW + lx]
          +        tile[ly * SW + (lx - 1)]
          +        tile[ly * SW + (lx + 1)];
```

The `+1` is the halo offset, and it is the third distinct index this thread has:
`threadIdx` (position in the block), `(row, col)` (position in the image), and
`(ly, lx)` (position in the tile). Forgetting the `+1` shifts every output by one
pixel diagonally — a result that looks like a correct stencil of a slightly wrong
image, and fails validation everywhere rather than in a stripe, which at least
makes it easy to spot.

Note what is *not* here: no `clampi`. All the boundary logic was paid for once,
in the load. That is the second job of shared memory from the lesson —
decoupling — showing up as a real instruction-count saving in the inner loop.

---

## Synchronization / memory reasoning

One barrier, one direction, no reuse of the buffer. The one subtlety is that the
early-out guard

```cpp
if (row >= h || col >= w) return;
```

must come **after** the barrier, not before it. On the 1021×733 image the
bottom-right block has threads whose interior pixel does not exist; if they
returned before `__syncthreads()`, the remaining threads would wait at a barrier
that those threads can never reach. On sm_89 the hardware's barrier counts only
non-exited threads, so this particular formulation happens not to hang — but
"happens not to hang" is not a design, and the same code with the return inside a
loop absolutely does hang. Keep every thread alive until the last barrier.
Module 9 is where this becomes a rule rather than a warning.

---

## Performance reasoning

**What the traffic model promised.** Each input pixel is read by up to 5 threads;
the 32×8 tile loads 340 cells to produce 256 outputs, so H = 1.33 and the best
achievable reduction in global reads is 5/1.33 = 3.8×.

**What was actually available.** Module 3 measured the untiled kernel at 0.3213 ms
on 4093×3079 = 12.6 Mpixels. Useful traffic is 8 B/pixel, so that is 313.8 GB/s
useful, **72.6 % of the 432 GB/s DRAM peak**. A kernel at 72.6 % of peak in
*useful* bytes has a hard ceiling: even with zero wasted traffic it could only
reach 432/313.8 = **1.38×**. And its actual re-read factor is at most 1.38, not
5 — L1 and L2 had already captured ~90 % of the available reuse. There was never
3.8× on the table; there was at most 1.38×, out of which tiling must pay:

- a **halo tax** of 33 % more global loads (340 vs 256 per block);
- a **barrier** per block, which convoys 8 warps;
- **1360 B of shared memory** per block, which at 256 threads/block does *not*
  reduce occupancy here (6 blocks/SM either way — see Example 1 §D), so this one
  is free at 32×8 but not at 32×16 (2448 B) or 64×8 (2640 B);
- an extra pass over the data: every element is written to shared and read back,
  which adds `STS` + `LDS` instructions that the untiled kernel does not issue.

**Measured.** One representative run (min of 4 sweeps, 400 ms duration-based
clock warm-up, iteration count auto-scaled to ~10 ms per timed segment):

```
=== image 4093 x 3079 (50.4 MB per array) ===
  config                 ms    GB/s eff     %peak   smem/blk     vs naive
  naive (32,8)       0.3187       316.4     73.2%         0 B       1.000x
  tiled 32x8         0.3734       270.0     62.5%      1360 B       0.853x
  tiled 32x16        0.3908       258.0     59.7%      2448 B       0.815x
  tiled 64x8         0.3746       269.1     62.3%      2640 B       0.851x
  M3 baseline        0.3213   <- Module 3 / Exercise 1, same kernel, same image
```

The control reproduces Module 3's 0.3213 ms to within 1 %, which is the
sanity check that makes the rest meaningful. **Tiling is a 15 % loss.** Across
runs the ratio landed between 0.72× and 0.93× and never above 1.0×; absolute
times moved by up to 2.5× with thermal state, which is exactly why the
methodology insists on ratios.

Larger tiles are not better. 32×16 costs 2448 B/block and is the slowest of the
three: more shared memory, more warps to convoy at the barrier, and no reduction
in the halo tax that matters (going 32×8 → 32×16 takes H from 1.33 to 1.20, worth
about 10 % of the *loads* in a kernel whose loads were not the problem).

**Is this a general result?** No, and that is the point. Example 2 runs the same
experiment with radius 1–4 box filters, where *K* goes from 9 to 81 and *K/H*
from 6.8 to 32.4:

| R | K/H | naive % of DRAM peak | tiled vs naive |
|---|---|---|---|
| 1 | 6.8× | 70 % | 0.87× |
| 2 | 14.8× | 54 % | 0.85× |
| 3 | 23.6× | 28 % | 0.92× |
| 4 | 32.4× | 16 % | 1.06× |

Tiling starts winning exactly where the untiled kernel stops being DRAM-bound.
That is the rule, and the 5-point stencil is on the wrong side of it.

---

## Expected output

```
=== Module 6 / Exercise 1 : tiled 5-point stencil (SOLUTION) ===

=== image 1021 x 733 (3.0 MB per array) ===
  config                 ms    GB/s eff     %peak   smem/blk     vs naive
  naive (32,8)       0.0160       375.2     86.9%         0 B       1.000x
  tiled 32x8         0.0180       333.2     77.1%      1360 B       0.888x
  tiled 32x16        0.0150       398.8     92.3%      2448 B       1.063x
  tiled 64x8         0.0161       370.8     85.8%      2640 B       0.988x
  M3 baseline        0.0107   <- Module 3 / Exercise 1, same kernel, same image

=== image 4093 x 3079 (50.4 MB per array) ===
  config                 ms    GB/s eff     %peak   smem/blk     vs naive
  naive (32,8)       0.3187       316.4     73.2%         0 B       1.000x
  tiled 32x8         0.3734       270.0     62.5%      1360 B       0.853x
  tiled 32x16        0.3908       258.0     59.7%      2448 B       0.815x
  tiled 64x8         0.3746       269.1     62.3%      2640 B       0.851x
  M3 baseline        0.3213   <- Module 3 / Exercise 1, same kernel, same image

--- prediction ---
  measured tiled-vs-naive on 4093x3079 : 0.853x -> a net loss (<0.95x)
  you predicted                        : a net loss (<0.95x)
  prediction: CORRECT

OVERALL: PASS
```

Ranges observed over repeated runs on this GPU:

| quantity | range |
|---|---|
| naive, 4093×3079 | 0.279 – 0.74 ms (thermal) |
| tiled 32×8 / naive, 4093×3079 | 0.72 – 0.93× |
| naive, 1021×733 | 0.010 – 0.033 ms |
| tiled 32×8 / naive, 1021×733 | 0.73 – 1.06× |

The small image is 3 MB per array and **fits entirely in the 48 MB L2**, so its
"GB/s effective" figures regularly exceed 100 % of DRAM peak. That number is not
a bandwidth, it is evidence of L2 residency, and it is also why the small image's
ratio is noisier: at 0.01 ms per launch the kernel is short enough that launch
overhead and clock transitions are a visible fraction. The prediction is scored
on the big image for exactly that reason.

---

## The result that matters

A 5-point stencil has *K* ≤ 5 and a 33 % halo tax, and on this GPU the untiled
version was already running at 72.6 % of DRAM peak — so the caches had captured
essentially all of the reuse before any shared memory was involved, and the
theoretical ceiling on *any* optimization was 1.38×. Tiling spent a barrier, 33 %
more loads and an extra `STS`/`LDS` round trip out of that budget and came out
**15 % behind**. The lesson is not "shared memory is useless"; it is that
**arithmetic intensity is a number you compute before you write the kernel**, and
the relevant comparison is not against the naive traffic model but against what
the naive kernel is *actually* achieving. If a kernel is at 70 % of a hardware
ceiling, the most tiling can win is 1.4× and the halo will eat it.

**Variation to try.** Change the stencil to 9-point (add the four diagonals,
weights 1, centre 4, divide by 12). *K* goes from 5 to 9 and H is unchanged at
1.33, so K/H nearly doubles while the naive kernel drops well below the
bandwidth ceiling. Predict the crossover point before you measure it, then
compare against the R=1 row of Example 2's table — the 9-point stencil is exactly
that box filter minus the corner weights.
