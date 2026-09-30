# Module 15 / Exercise 02 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

---

## The problem, restated

NCHW element `(n,c,h,w)` is at `((n*C + c)*H + h)*W + w`.
NHWC element `(n,h,w,c)` is at `((n*H + h)*W + w)*C + c`.

Fold `h` and `w` into a single spatial index `sp = h*W + w`, with
`spatial = H*W`. Then:

```
NCHW:  n*C*spatial + c*spatial + sp        <- c is the slow axis, sp the fast one
NHWC:  n*spatial*C + sp*C     + c          <- sp is the slow axis, c the fast one
```

Per image, that is *exactly* the transpose of a `C × spatial` matrix. There is
no fourth dimension left in the problem; `n` only selects which matrix. Once you
see that, the exercise is Exercise 1 with a batch index and a very lopsided
aspect ratio (67 × 3249).

This is also the general shape of the AoS → SoA conversion Module 5 deferred
here: NHWC is the array-of-structs form (the `C` channels of one pixel are
adjacent), NCHW is the struct-of-arrays form.

---

## TODO 1 — the grid

```cpp
grid->x = (spatial + TILE - 1) / TILE;   // 102
grid->y = (c + TILE - 1) / TILE;         //   3
grid->z = ni;                            // 128
```

39 168 blocks, exactly the number of tiles needed.

**Why this assignment and not another.** Three constraints decide it.

1. `gridDim.y` and `gridDim.z` are limited to 65535; `gridDim.x` is limited to
   `2^31 - 1` (M3). Here nothing exceeds 65535, so the limit does not bind — but
   it would the moment `spatial` grew past 2 M, and the axis you put on `x` is
   the one that scales.
2. `blockIdx.z` is the cheapest coordinate to recover *if* you put the batch
   there, because it needs no arithmetic at all. Packing all three into
   `blockIdx.x` and recovering them with two integer divisions by 102 and 3
   costs real instructions — integer division is not a single instruction — for
   no benefit.
3. The spatial axis is the long one (3249 versus 67), so it belongs on the
   dimension with the largest limit.

**Common wrong approaches.**

- `grid.x = spatial` (one block per spatial index rather than per tile). The
  harness rejects it: it launches 3249/32 = 101× too many blocks, which the
  `tilesLaunched <= 2*tilesNeeded` test catches.
- `grid.y = C` instead of `ceil(C/32)`. 67 blocks in `y` instead of 3, most of
  them redundant. Same rejection.
- Putting `n` on `x` and `spatial` on `z`: works here, fails as soon as
  `spatial/32 > 65535`, i.e. at 2 M pixels — a 1448 × 1448 feature map. Not
  hypothetical.

---

## TODO 2 — the load

```cpp
const long long inBase = (long long)n * c * spatial;
for (int j = 0; j < TILE; j += BROWS) {
    int ch = c0 + threadIdx.y + j;
    int sp = s0 + threadIdx.x;
    float v = 0.0f;
    if (ch < c && sp < spatial) v = in[inBase + (long long)ch * spatial + sp];
    tile[shIdx(threadIdx.y + j, threadIdx.x)] = v;
}
```

**Why it is coalesced.** Within a warp `threadIdx.x` runs 0..31 and
`threadIdx.y` is fixed, so `sp` takes 32 consecutive values at a fixed channel.
In NCHW, consecutive `sp` are consecutive addresses. 128 contiguous bytes, and
`s0` is a multiple of 32 so the run is sector-aligned: **4 sectors per warp**,
100% efficient.

**Why the `float v = 0.0f` matters.** Writing

```cpp
if (ch < c && sp < spatial) tile[...] = in[...];      // WRONG
```

leaves tile cells belonging to out-of-range `(ch, sp)` holding whatever the
previous block left in that SRAM. Shared memory has no initial value (M6). Those
cells are then read in the store phase by threads whose *own* guard passes —
because the store guard tests a different pair of coordinates. With `C = 67` the
last channel tile has 3 valid columns of 32, so 29/32 of its cells are stale, and
they would be written straight into the output. Staging a zero instead is the
cheapest correct fix. (Guarding the store against exactly the same condition is
the other, and it is harder to get right because the coordinates swap.)

This is Module 6's partial-tile hazard, one module on, and it is the reason the
harness poisons the output with `-13.0f` before each run.

---

## TODO 3 — the store

```cpp
const long long outBase = (long long)n * spatial * c;
for (int j = 0; j < TILE; j += BROWS) {
    int ch = c0 + threadIdx.x;              // threadIdx.x now runs over CHANNELS
    int sp = s0 + threadIdx.y + j;
    if (ch < c && sp < spatial)
        out[outBase + (long long)sp * c + ch] = tile[shIdx(threadIdx.x, threadIdx.y + j)];
}
```

**The one decision that matters.** In NHWC the *channel* is the contiguous axis.
So `threadIdx.x` — the axis that varies across a warp — must run over channels
here, the exact opposite of the load phase. Everything else follows: the tile
subscripts swap with it, and the guard's two halves swap with them.

**Why the write is only 4 sectors even though `C = 67`.** A warp writes
`out[base + sp*67 + ch]` for `ch = ch0..ch0+31` at fixed `sp` — 32 consecutive
floats, 128 contiguous bytes. The *stride between warps* is 67 floats, which is
not a multiple of 8, so the runs are not sector-aligned and a warp may straddle
5 sectors rather than 4. That costs at most 25% of the write requests, and in a
streaming kernel the fifth sector is the next `sp`'s first (M5's cross-warp
sharing argument). It is not worth fixing.

**Common wrong approaches.**

- Keeping `threadIdx.x` on the spatial axis in the store phase. The output write
  then has stride `C*4 = 268` bytes: 32 sectors per warp, and you have rebuilt
  the naive kernel with extra steps. It validates. It is 2.6× slower.
- `out[outBase + ch*spatial + sp]` — that is NCHW again. Validates as a copy,
  fails the checker.
- Forgetting `(long long)` on `sp * c`. `spatial * c = 217683` per image and
  `NTOTAL = 27,863,424`, so `int` survives here — but `outBase` must be 64-bit or
  it overflows at `n = 128`. Getting one of the two casts and not the other is a
  silent wrap at large `n`.

---

## TODO 4 — the shared layout

Identical to Exercise 1: `SH_FLOATS = 32*33`, `shIdx(r,c) = r*33 + c`. The XOR
swizzle `r*32 + (c ^ (r & 31))` with `SH_FLOATS = 1024` is equally accepted and
measures the same.

The tile is 32 × 32 even though `C = 67`, because the tile's *shape* is a
property of the warp (32 lanes) and not of the tensor. A 67-wide tile would be a
mistake: you would need a pitch coprime with 32 anyway, and you would waste
shared memory on a dimension that a warp cannot cover in one instruction.

---

## Synchronization / memory reasoning

One `__syncthreads()` between the phases, outside all control flow. Every thread
of the block executes the load loop — including threads whose guard is false,
which is why the guard wraps the assignment and not a `return` (M9's uniformity
rule; on sm_89 a divergent barrier usually corrupts silently rather than
hanging).

Shared memory per block is 4224 B, which at 256 threads leaves the block count
capped by threads, not by shared memory (6 blocks/SM; M6's table).

---

## Performance reasoning

Observed (see the variance note):

```
TIMING (min of 8 rotated sweeps)
kernel                        ms      GB/s   %ofcopy
copy ceiling              1.5521     143.6    100.0%
naive NCHW->NHWC          4.9846      44.7     31.1%
your tiled NCHW->NHWC     1.5763     141.4     98.5%
```

**The tiled conversion reaches the copy ceiling.** 98.5% here; 100.2% and 100.3%
in two other runs, which is noise, not a kernel that beats a copy. Observed range
across authoring runs: **94.3-100.3% of copy**. There is nothing left: the kernel
moves 2N bytes and the machine moves 2N bytes as fast as it can.

**The naive kernel at 25-37% of copy** (31.1% in the run above; 25.1% in the
fastest-machine run, 36.7% in the most throttled one). Its write has stride `C = 67` floats = 268
bytes, so every lane owns a different sector: 32 sectors per warp for 128 useful
bytes. Traffic model: 1N read + 8N write = 9N against the copy's 2N → 22%.
Measured 25-37%. Better than the model for the same reason as Exercise 1 — the
over-fetched sectors are consumed by neighbouring warps before eviction — but
here the margin is small, because `C = 67` is not a multiple of 32 and the reuse
across warps is imperfect.

**PRED[2], and the surprise.** `C = 67` needs `ceil(67/32) = 3` channel tiles, of
which the third contributes only 3 useful columns. Launched tile cells:
`3 × 102 × 128 × 32 × 32 = 40,108,032`; elements: `27,863,424`. **69.5% of the
launched tile cells hold data** — nearly a third of the grid's nominal work is
guarded off.

And it costs nothing. The kernel still reaches the copy ceiling. The reason is
that a guarded-off thread issues **no memory instruction at all** (M5: inactive
lanes supply no address), so a mostly-empty tile consumes a block slot and a few
predicated instructions and no bandwidth. The only cost is the launch and
scheduling of 30% more blocks than strictly necessary, which on a kernel bound by
DRAM is free. **Tile occupancy and memory efficiency are different quantities,
and only one of them is on the critical path.**

If you predicted that the ragged channel axis would cost 30% of the bandwidth,
you made the mistake of pricing *threads* rather than *sectors*.

---

## Expected output

```
Module 15 exercise 02 -- NCHW -> NHWC
GPU: NVIDIA RTX 3500 Ada Generation Laptop GPU, CC 8.9
tensor: N=128 C=67 H=57 W=57 -> 27863424 floats = 106.3 MiB per buffer
per image this is a 67 x 3249 transpose

TODO 4: SH_FLOATS=1056, in range yes, injective yes, degrees 1 / 1
TODO 1: grid = (102, 3, 128) = 39168 blocks; 39168 tiles needed -> accepted

TIMING (min of 8 rotated sweeps)
kernel                        ms      GB/s   %ofcopy
copy ceiling              1.5521     143.6    100.0%
naive NCHW->NHWC          4.9846      44.7     31.1%
your tiled NCHW->NHWC     1.5763     141.4     98.5%

VALIDATION
  copy ceiling          : ok
  naive NCHW->NHWC      : ok
  your kernel           : ok (0 mismatches)

SCORING
  TODO 4 shared layout valid         : 1/1
  TODO 1 grid covers the tensor      : 1/1
  your kernel is correct             : 1/1
  your kernel >= 80% of the copy     : 1/1
  PRED[0] naive bucket   1, measured 1 : correct
  PRED[1] yours bucket   3, measured 3 : correct
  PRED[2] tile fill      2, measured 2 (69.5% of launched tile cells hold data) : correct

SCORE: 7/7
OVERALL: PASS
```

The correct predictions are `PRED = { 1, 3, 2 }`.

**Variance warning.** On this 75 W laptop part the absolute figures move a great
deal with power state. A second run, taken while the GPU was in its throttled
state (memory clock 7001 MHz instead of 8801, `nvidia-smi` throttle reasons
non-zero), gave copy 129.6 GB/s, naive 47.6 GB/s (36.7% of copy) and the tiled
kernel 122.2 GB/s (94.3% of copy) — and still scored 7/7, because every bucket
boundary in this exercise was placed to survive it. Post-authoring repair moved
those boundaries out further still, to 70% and 82%: over many repeat runs the
naive kernel lands anywhere in 29–59% of the copy (a fully strided write is the
most power-hungry kernel here, so it is the one that loses the most when the part
is power limited) and the tiled kernel in 90–100%, which leaves at least 8 points
of clearance on every edge. The harness also probes the copy ceiling after
warming up and, if it comes back under 200 GB/s, idles for 10 s and warms again
rather than scoring a deeply power-limited run. Trust the `%ofcopy` column,
not the GB/s column.

---

## The result that matters

A 4-D tensor permutation looks like a different problem from a matrix transpose
and is not one: fold every axis that stays on the same side of the permutation
into a single index, and `NCHW → NHWC` is a batched `C × HW` transpose with a
batch index that costs one `blockIdx.z`. The reduction is the exercise; once you
have made it, Exercise 1's kernel solves it unchanged and reaches the copy
ceiling. The second lesson is quantitative and counter-intuitive: a channel count
of 67 leaves 30% of your launched tile cells empty and costs **nothing**, because
a predicated-off lane supplies no address and therefore no traffic. Grid
efficiency and memory efficiency are not the same number.

**Variation to try.** Change `C` from 67 to 4 — a realistic input layer, RGBA —
and re-measure. A 32 × 32 tile now wastes 7/8 of its cells and, more to the
point, the NHWC write becomes 16 contiguous bytes per warp-row instead of 128.
Work out what the sector count per warp becomes on the write side, then decide
whether a 32 × 32 tile is still the right shape or whether you want a tile that
is 4 wide and 256 tall. This is the case where the "just use 32 × 32" habit
finally breaks, and it is exactly the shape a convolution input layer has.
