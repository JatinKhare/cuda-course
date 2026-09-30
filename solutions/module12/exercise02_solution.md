# Module 12 / Exercise 02 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

Warning-clean. `SCORE: 5/5`, `OVERALL: PASS`.

---

## The shape of the problem

The exercise exists because the sum reduction hides three decisions behind
commutativity, and none of the three is actually a property of reductions. The
operator here — "sum, and the length of the longest run of consecutive elements
above a threshold" — is associative (so the tree is legal) but **not**
commutative (so every ordering decision becomes visible).

The monoid summarises a *contiguous span*:

```cpp
struct Seg { float sum; int len, pre, suf, best; };
```

`pre` and `suf` are the run lengths at the two ends of the span, `best` is the
longest run anywhere inside it. The reason `pre` and `suf` have to be carried is
the entire content of the exercise: a run can straddle the boundary between two
spans, and neither span can see it.

---

## TODO 1 — the monoid

```cpp
__host__ __device__ __forceinline__ Seg segIdentity(void)
{
    Seg s; s.sum = 0.0f; s.len = 0; s.pre = 0; s.suf = 0; s.best = 0; return s;
}

__host__ __device__ __forceinline__ Seg combineSeg(Seg a, Seg b)
{
    Seg r;
    r.sum = a.sum + b.sum;
    r.len = a.len + b.len;
    r.pre = (a.pre == a.len) ? a.len + b.pre : a.pre;   // a wholly above thr?
    r.suf = (b.suf == b.len) ? b.len + a.suf : b.suf;   // b wholly above thr?
    int j = a.suf + b.pre;                              // the straddling run
    int m = (a.best > b.best) ? a.best : b.best;
    r.best = (j > m) ? j : m;
    return r;
}
```

**Why it is correct.** `pre` of the joined span is `a`'s prefix run, *unless*
`a` is entirely above threshold, in which case the prefix continues into `b` and
becomes `a.len + b.pre`. The test for "entirely above threshold" is
`a.pre == a.len` — which is why `len` must be carried even though nobody reads it
at the end. Symmetrically for `suf`. The only run that neither `a.best` nor
`b.best` can have seen is the one ending at `a`'s last element and starting at
`b`'s first, whose length is exactly `a.suf + b.pre`.

**Associativity.** Check `combineSeg(combineSeg(a,b),c)` against
`combineSeg(a,combineSeg(b,c))` on the case that actually bites: all three spans
entirely above threshold. Left: `(a,b)` has `pre = a.len + b.pre = a.len + b.len`,
then combining with `c` gives `pre = a.len+b.len+c.pre`. Right: `(b,c)` has
`pre = b.len + c.pre`, then `a` in front gives `a.len + b.len + c.pre`. Equal.

**The identity trap.** The requirement is `combineSeg(e, x) == x` **and**
`combineSeg(x, e) == x`. With `e = {0,0,0,0,0}`:
- `combineSeg(e,x)`: `pre = (0 == 0) ? 0 + x.pre : …` → `x.pre` ✓;
  `suf = (x.suf == x.len) ? x.len + 0 : x.suf` → `x.suf` in both branches ✓;
  `best = max(max(0, x.best), 0 + x.pre) = x.best` because `best >= pre` is an
  invariant of the type ✓.
- `combineSeg(x,e)` is symmetric ✓.

**The wrong identity that nearly works**, and the reason `THR` is negative: many
readers pad an incomplete tile with a *zero-valued element* rather than with the
identity — `Seg z = segLeaf(0.0f, thr)`. That has `len = 1`, and since the
threshold here is `-0.5f`, `0.0f > -0.5f`, so the pad is **above threshold** and
extends the row's trailing run by one. The symptom is `best` too large by 1 on
roughly half the rows, and `sum` perfectly correct, which is exactly the kind of
bug that survives a code review. (If `THR` had been positive, the zero pad would
have had `pre = suf = best = 0` and `len = 1`, giving the right `best` and a
wrong `len` — which propagates into `pre` one level up and is wrong again, just
more rarely.)

---

## TODO 2 — the warp reduction

```cpp
__device__ __forceinline__ Seg warpReduceSeg(Seg s)
{
    for (int off = 1; off < 32; off <<= 1) {
        Seg t = shflDownSeg(0xffffffffu, s, off);
        s = combineSeg(s, t);
    }
    return s;                                   // valid in lane 0
}
```

Two decisions, both invisible for a sum, both load-bearing here.

**1. Operand order.** `shflDownSeg` fetches from lane `L + off`, which is a
*later* span, so it must be the right-hand operand: `combineSeg(s, t)`, not
`combineSeg(t, s)`. Swapping them reverses the row. `sum` is unaffected; `best`
is wrong whenever the row is not a palindrome with respect to the threshold.

**2. Offset order, and this is the one that actually caught the author.** The
textbook warp reduction goes `16, 8, 4, 2, 1`. Trace which spans lane 0 holds:

```
off=16 : lane0 = c(0, 16)
off=8  : lane8 = c(8, 24);  lane0 = c( c(0,16), c(8,24) )      <-- spans 0,16,8,24
```

Out of order at the second step. The result is a correct sum and a `best`
computed over a permuted row. Going upward instead:

```
off=1  : lane0 = c(0,1)         lane2 = c(2,3)   ...
off=2  : lane0 = c(0..1, 2..3)  = 0..3
off=4  : lane0 = 0..7
off=8  : lane0 = 0..15
off=16 : lane0 = 0..31
```

Contiguous and in order at every step. This is the same tree with the levels
traversed the other way, and for `+` the two give identical answers; for a
non-commutative operator only the increasing form is a reduction of the sequence.

Note that this does **not** mean `example01.cu`'s `warpReduceSum` is wrong. It is
a sum. The point of putting the two next to each other is that nothing in the
source of the sum version tells you which of its choices were free.

**Symptom of getting this wrong** (the author's first run): 84,249 of 100,000
rows with a wrong `best`, `0` with a wrong `sum`, and the first wrong row off by
exactly 1. Rows shorter than 32 are all correct, because for them lane `L` holds
exactly element `L` and the tree collapses to a single tile.

---

## TODO 3 — the decomposition

Three requirements that pull against each other, and one arrangement that
satisfies all three.

### Order and coalescing are not in conflict, if you tile

The obvious order-preserving mapping gives lane `L` a contiguous chunk of the
row — and that is a stride-`chunk` access pattern, which Module 5 prices at one
sector per lane. The obvious coalesced mapping gives lane `L` elements
`a+L, a+L+32, …` — and that is the strided form the harness probes and which gets
42,860 of 100,000 rows wrong.

The arrangement that is both:

```cpp
Seg acc = segIdentity();
for (int base = a; base < b; base += 32) {        // warp-uniform trip count
    int i = base + lane;
    Seg leaf = (i < b) ? segLeaf(data[i], thr) : segIdentity();
    Seg t    = warpReduceSeg(leaf);               // reduces THIS TILE, in order
    if (lane == 0) acc = combineSeg(acc, t);      // tiles folded in row order
}
```

Within a tile, lane `L` holds element `base + L`, so lane order *is* row order
and `warpReduceSeg` produces the tile's summary correctly. Across tiles, the
serial fold in lane 0 preserves order by construction. The global reads are
`data[base + lane]` — 32 consecutive floats per warp, one 128 B line, perfectly
coalesced.

The price is five shuffle steps (× 5 fields = 25 `SHFL` instructions) per 32
elements. That sounds expensive and is not: the kernel measures 72–194 GB/s
against a baseline of 11–28 GB/s.

### The skew

Row lengths run from 8 to 200,000 — four orders of magnitude. One warp per row
means the ten 200,000-element rows each get one warp and everything waits for
them. The solution splits the rows into two lists and runs two kernels:

```cpp
#define LONG_THRESH 2048
// rows <= 2048 elements : one warp each, grid-stride over the row list
// rows >  2048 elements : one 256-thread block each, the block's 8 warps take
//                         CONTIGUOUS, tile-aligned segments
```

For the long rows, the warps must take **contiguous** segments, not interleaved
tiles — order again — and the eight warp summaries are folded in warp order by
thread 0 through shared memory:

```cpp
int tiles = (b - a + 31) / 32;
int per   = (tiles + nw - 1) / nw;
int t0    = w * per;
int t1    = (t0 + per < tiles) ? t0 + per : tiles;
...
if (lane == 0) wres[w] = acc;
__syncthreads();
if (threadIdx.x == 0) {
    Seg tot = segIdentity();
    for (int j = 0; j < nw; ++j) tot = combineSeg(tot, wres[j]);   // warp order
    ...
}
```

Note that `t1 - t0` differs between warps of the same block, and that is fine —
there is no barrier inside the tile loop. The one `__syncthreads()` is after the
loop, reached by every thread, and the `warpReduceSeg` inside the loop is a
warp-scope operation whose trip count is warp-uniform. Module 9's uniformity rule
is satisfied at the level that matters for each primitive.

### The partition is built once

```cpp
static int inited = 0; ...
if (!inited) { /* copy rowStart to host, split, upload two index lists */ }
```

The partition is a function of the row-length table only, not of the data, so it
is a plan built once and amortised over every launch — the same argument Module 8
exercise 2 made about binning, and with the same caveat: if you launch the kernel
once, the plan is a net loss. The harness calls `launchSegFast` before timing, so
the plan is already built.

**Alternatives that also pass.** A single kernel with one block per row works and
wastes most of a block on the 95% of rows with fewer than 64 elements. A
device-side partition (a pass that counts and a pass that scatters) removes the
host round-trip and needs a scan, which is Module 13. Persistent warps pulling
rows from an atomic work queue (Module 10's ticket allocation) handles the skew
without any partition at all and is arguably the best answer here.

---

## TODO 4 — the predictions

| | answer | measured |
|---|---|---|
| (a) speedup | 8× (anything 3.3–13.3 passes at the measured 6.6×) | 6.6–10.2× |
| (b) is the strided form correct for every row? | **no (0)** | 42,860 / 100,000 rows wrong |

**(a)** The baseline has two independent defects and both are worth naming.
*Divergence*: one thread per row, a warp holds 32 rows, and the warp issues the
loop body `max(len)` times over those 32 rows. With 5% of rows in the 4000–12000
range, most warps contain at least one, so the warp pays ~8000 iterations to do
~400 rows' worth of work — Module 8's `max/mean` bound, around 20× on this
distribution. *Coalescing*: thread `t` walks row `t` sequentially, so a warp's 32
addresses are 32 different rows — one sector per lane, 12.5% efficiency
(Module 5). The two do not multiply, because removing the divergence also fixes
the coalescing; a prediction anywhere in 5–20× is defensible.

**(b)** The strided form is wrong for exactly the rows longer than 32. For a row
of length ≤ 32, lane `L` holds element `a + L` and the lane-ordered reduction is
the row-ordered reduction. Longer than that and lane 0 has accumulated elements
`a, a+32, a+64, …` — a subsequence, not a span — and `combineSeg` is being asked
to join spans that were never adjacent. 42,860 wrong is close to the 5,010 long
rows plus the ~37,850 short-but-longer-than-32 rows, which is a good sanity
check on the reasoning.

---

## Synchronization / memory reasoning

- `warpReduceSeg` uses `__shfl_down_sync(0xffffffff, …)` and requires **all 32
  lanes** to reach it. In `segWarpRows` the enclosing loop bound is
  `base < b` with `a` and `b` read from `rowStart[r]` where `r` is warp-uniform,
  so the trip count is warp-uniform. In `segBlockRows` it is `t < t1` with `t0`
  and `t1` computed from `w = threadIdx.x / 32` — warp-uniform. If you make
  either trip count depend on `lane`, the mask is a lie and the behaviour is
  undefined (Module 8).
- The early `if (w >= nRows) return;` in the probe kernel retires whole warps,
  never part of one, so no `_sync` primitive is left short of participants.
- `segBlockRows`'s single `__syncthreads()` supplies both of Module 9's
  guarantees for `wres[]`: G1 so thread 0 does not read a slot before its warp
  has written it, G2 so what it reads is what was written.

---

## Performance reasoning

Representative run (warm GPU; see the note on variation):

```
Module 12 exercise 02 -- segmented non-commutative reduction
100000 rows, 45438060 elements, 173.3 MiB, threshold -0.50
row lengths: min 8, max 200000

kernel                               ms       GB/s
--------------------------------------------------
baseline (thread per row)       16.6424       10.9
your segFast                     2.5047       72.6
speedup 6.64x

correctness: 0/100000 rows with a wrong `best`, 0 with a wrong `sum`

predictions:
  (a) speedup      predicted 8.00x, measured 6.64x   MATCH
  (b) strided form correct for every row? you said no;
      the probe got 42860 of 100000 rows wrong                MATCH

SCORE: 5/5  (best ok, sum ok, speed ok, pred-a ok, pred-b ok)

OVERALL: PASS
```

A cooler run of the same binary gave `6.5634 ms` baseline, `0.9387 ms` fast,
speedup `6.99×`, and 193.6 GB/s. The speedup is stable at **6.6–10.2×** across
thermal states; the absolute GB/s is not. The pass gate is 3.0×.

**Why only 72–194 GB/s and not 400?** Two reasons, and the honest one first:
the 25 `SHFL` instructions per 32 elements are real work that a sum reduction
does not do, and at this ratio the kernel is not purely memory-bound any more.
Second, the short-row path reads rows of 8–64 elements, so a warp's coalesced
128 B tile is often only partly used and the next row starts at an arbitrary
offset — Module 5's boundary-to-interior argument, with very short interiors.
Neither is a defect to fix; it is what a ragged segmented reduction costs.

**The obvious improvement**, if you want to push it: fold two or four tiles into
each lane before reducing — lane `L` handles the contiguous pair
`(base + 2L, base + 2L + 1)` — which quarters the shuffle count at the price of
a stride-2 read. Try it and measure; on this data it is roughly a wash, which is
itself informative.

---

## Expected output

As pasted above. `SCORE: 5/5`, `OVERALL: PASS`.

---

## The result that matters

Every reduction you will ever write has three ordering decisions in it — which
operand goes left, which direction the tree levels run, and how the array is cut
into pieces — and for `+` all three are free, so the textbook version never
mentions them and you never learn that they were decisions. Swap in an operator
that notices, and two of the three become silent wrong answers that are correct
on 58% of your data. The generalisation is worth stating: **a reduction is a
monoid homomorphism, and the only thing the hardware gives you for free is
associativity.** Everything else you have to arrange.

**Variation to try.** Replace the operator with one that is associative but has
no identity at all — say, "the pair (first element, last element)" restricted to
non-empty spans. You can no longer pad a partial tile, and the entire structure
of the kernel has to change: every out-of-range lane must be excluded from the
reduction rather than neutralised in it. Work out what that does to the mask you
pass to `__shfl_down_sync`, and you will understand why library reductions insist
on an identity.
