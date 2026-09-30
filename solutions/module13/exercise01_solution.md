# Module 13 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o ex1sol.exe exercise01_solution.cu
ex1sol.exe
```

Warning-clean with CUDA 13.2 on sm_89. Runs in about 25 s (two sizes × 4 sweeps
× 6 configurations × ~10 ms segments, plus two 400 ms warm-ups and two full
host-side validations of a 67 M array).

---

## TODO 1 — Hillis–Steele with double buffering

```cpp
int cur = 0;
for (int off = 1; off < TILE; off <<= 1) {
    for (int i = tid; i < TILE; i += BLK)
        s[(cur^1)*TILE + i] = s[cur*TILE + i] + ((i >= off) ? s[cur*TILE + i - off] : 0u);
    __syncthreads();
    cur ^= 1;
}
u32 total = s[cur*TILE + TILE - 1];

u32 v[IPT];
for (int k = 0; k < IPT; ++k) { int i = tid + k*BLK; v[k] = (i == 0) ? 0u : s[cur*TILE + i - 1]; }
__syncthreads();
for (int k = 0; k < IPT; ++k) s[tid + k*BLK] = v[k];
__syncthreads();
return total;
```

**Why it is correct, from the hardware model.** The step `x[i] ← x[i] + x[i-off]`
has thread `i` reading a location that thread `i-off` writes in the same step.
That is a cross-thread **write-after-read** hazard in M9's taxonomy. A barrier
placed *after* the statement orders step `k` against step `k+1`; it does nothing
about two threads racing *inside* step `k`. The SIMT model gives you no ordering
between warps within an instruction stream — the four warp schedulers on the SM
issue independently (M1), so warp 3 can execute its store before warp 0 executes
its load. Reading from one buffer and writing to another removes the hazard by
construction: the read set and the write set are disjoint addresses, so there is
nothing to order.

The final inclusive→exclusive shift is the *same* hazard with the source and
destination being the same buffer when `cur == 0`. Reading all four values into
registers, then a barrier, then writing, is the standard fix and costs one extra
barrier.

**Common wrong approaches and their symptoms.**

| attempt | symptom |
|---|---|
| `s[i] += s[i-off]; __syncthreads();` | wrong at 64+ threads, correct at 32. Results too large in a data-dependent pattern; roughly the upper half of each 32-element group is affected. `compute-sanitizer --tool racecheck` reports it. |
| `u32 t = s[i-off]; __syncthreads(); s[i] += t; __syncthreads();` | **correct**, and a legitimate alternative to double buffering: it costs 2 barriers per level (20 total) instead of 1 (10 total) but saves 4 KB of shared memory. Measured at 0.0351 ms against the double buffer's 0.0333 ms — **1.05x slower**, i.e. the extra 10 barriers cost about 5 %, because this block is not shared-memory limited and so the 4 KB it saves buys nothing back. |
| reading `s[i-off]` without the `i >= off` guard | reads `s[-1]`, which is in-bounds *shared* memory (it is `s[TILE-1]` of the other buffer or garbage), so `memcheck` says nothing and the first few outputs are wrong. |
| forgetting the final shift and returning the inclusive scan | every output off by `x[i]`; validation fails at index 0 with `got 4 expected 0`, which is the fastest possible diagnosis. |

## TODO 2 — Blelloch upsweep

```cpp
int offset = 1;
for (int d = TILE >> 1; d > 0; d >>= 1) {
    __syncthreads();
    for (int t = tid; t < d; t += BLK) {
        int ai = offset*(2*t + 1) - 1;
        int bi = offset*(2*t + 2) - 1;
        s[PIDX(bi)] += s[PIDX(ai)];
    }
    offset <<= 1;
}
__syncthreads();
```

This is **Module 12's tree reduction**, written so that the partial sums stay in
place rather than being compacted. `ai` is the right-most leaf of the left
subtree, `bi` the right-most leaf of the right subtree; after the level, `bi`
holds the sum of both. The `-1` in both index expressions is what makes the
node's value live at the *right* end of its range, which is what the downsweep
then needs.

Note that the barrier is at the *top* of the loop, not the bottom. Either works
as long as there is exactly one per level and one after the loop; putting it at
the top makes the trailing `__syncthreads()` after the loop explicit and easy to
see, which matters because the next thing that happens is a single-thread write.

There is no WAR hazard here: at each level `bi` values are written and `ai`
values are read, and the two sets are disjoint (a node is either a left child or
a right child at a given level, never both). This is why Blelloch needs no second
buffer.

## TODO 3 — the identity element and the downsweep

```cpp
u32 total = s[PIDX(TILE-1)];
if (tid == 0) s[PIDX(TILE-1)] = 0u;
for (int d = 1; d < TILE; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    for (int t = tid; t < d; t += BLK) {
        int ai = offset*(2*t + 1) - 1;
        int bi = offset*(2*t + 2) - 1;
        u32 tmp      = s[PIDX(ai)];
        s[PIDX(ai)]  = s[PIDX(bi)];
        s[PIDX(bi)] += tmp;
    }
}
__syncthreads();
```

**Why `x[N-1] = 0` is the whole reason this produces an exclusive scan.** The
downsweep's invariant is: *node v holds the sum of everything strictly to the
left of v's range.* Setting the root to the identity establishes that invariant
at the top (nothing is to the left of the whole array). Each swap-and-add
propagates it: the left child inherits the parent's value unchanged (nothing new
is to its left), the right child gets the parent's value plus the left subtree's
sum (which the upsweep left in `ai`). At the leaves the invariant *is* the
exclusive scan.

**Common wrong approaches.**

- **Not zeroing at all.** The root still holds `total`, so every output is
  shifted up by exactly `total`: `out[i] = correct[i] + total`. Uniform,
  clean-looking, completely wrong. Validation fails at index 0 with
  `got <total> expected 0`.
- **Zeroing but not capturing `total` first.** The tile total returned is 0, so
  every block reports a zero sum, pass 2 scans an array of zeros, and every tile
  offset is 0. The *first* tile is then correct and all 1023 others are wrong.
  This is the version that passes a single-tile test.
- **Writing the zero from every thread instead of `tid == 0`.** Harmless here
  (all threads write the same value to the same address, which M7 would call a
  broadcast store), but it is a habit that breaks the moment the value is not
  uniform.
- **Forgetting that `offset` must be halved *before* the level's work**, not
  after. Off by one level; the output is a scan of a differently-bracketed tree
  and is wrong in a hard-to-read pattern.

## TODO 4 — warp-shuffle block scan (design)

```cpp
u32 x[IPT];
for (int k = 0; k < IPT; ++k) x[k] = s[tid*IPT + k];
u32 run = 0;
for (int k = 0; k < IPT; ++k) { u32 t = x[k]; x[k] = run; run += t; }   // serial, in registers

u32 wincl = warpInclusiveScan(run, lane);                              // 5 shuffles
if (lane == 31) warpTot[wid] = wincl;
__syncthreads();
if (wid == 0) {                                                        // scan 8 warp totals
    u32 v = (lane < BLK/32) ? warpTot[lane] : 0u;
    v = warpInclusiveScan(v, lane);
    if (lane < BLK/32) warpTot[lane] = v;
}
__syncthreads();
u32 wexcl = (wid == 0) ? 0u : warpTot[wid - 1];
u32 texcl = wexcl + wincl - run;
for (int k = 0; k < IPT; ++k) s[tid*IPT + k] = x[k] + texcl;
```

with

```cpp
for (int off = 1; off < 32; off <<= 1) {
    u32 n = __shfl_up_sync(0xffffffffu, v, off);
    if (lane >= off) v += n;
}
```

**Why this shape.** The decomposition is chosen so that each level uses the
cheapest mechanism that can do the job: registers for the per-thread serial
scan (free), the shuffle network for the intra-warp scan (no memory, no
barrier), and 8 words of shared memory plus two barriers for the only step that
genuinely crosses warps. Total shared traffic beyond the tile load/store: 16
words per block.

**The three places this goes wrong.**

1. **Dropping the `if (lane >= off)`.** `__shfl_up_sync` does not zero-fill: a
   lane whose source index would be negative receives **its own value**. Without
   the guard, lane 0 adds itself repeatedly and lanes 1..`off-1` add stale
   partials. Output is wrong only in the low lanes of each warp — 5 of 32 lanes
   at the first step — so roughly 15 % of elements are wrong and the pattern
   looks random. The alternative fix, `v += (lane >= off) ? n : 0`, is
   equivalent and compiles to the same predicated add (M8).
2. **`texcl = wexcl + wincl`** instead of `wexcl + wincl - run`. `wincl` is the
   *inclusive* warp scan of thread totals, so it already contains this thread's
   own `run`. Subtracting it converts inclusive to exclusive — §2's
   inverse-based conversion, valid here because `+` has an inverse. Getting this
   wrong shifts every thread's block by its own total; the result is monotone
   and plausible.
3. **A missing barrier between writing `warpTot[wid]` and reading it in warp 0.**
   The classic. It passes on this hardware most of the time because warp 0
   usually runs last, which is the most dangerous possible outcome.

**Loading `s[tid*IPT + k]`** gives each thread 4 *consecutive* words — the
"blocked" arrangement — which is a stride-4 access across lanes. By M7 that is
degree `gcd(4,32) = 4`. In practice the compiler merges the four unrolled loads
into one `LDS.128`, which M7's phase model splits into 4 phases of 8 lanes each
reading 128 contiguous bytes: conflict-free. This is the vectorization effect
M7 warned about, working in our favour for once.

A reader who "fixes" this to the striped arrangement `s[tid + k*BLK]` — which
looks more coalesced and is what you would write for a *global* load — gets a
kernel that is 0.956x the time and **wrong**: the per-thread serial scan in step
(1) assumes a thread owns a contiguous run, and under the striped arrangement it
scans four elements that are 256 apart, producing a scan of a permutation. This
is exactly why CUB has a separate `BlockLoad` with a
`BLOCK_LOAD_TRANSPOSE` policy: you load striped (coalesced from global) and then
transpose through shared memory into blocked order before scanning. Measured
here: striped 0.0160 ms and `BAD`, blocked 0.0167 ms and correct. A 4 % speedup
for a wrong answer is the most dangerous kind.

## TODO 5 — the predictions

`PREDICT_SLOWEST 'H'`, `PREDICT_BUCKET 3`.

The slowest is Hillis–Steele at every size measured. The Hillis–Steele/Blelloch
time ratio is **1.48×** in the L2-resident regime, against a work ratio of 5.0×
(10,240 additions per tile versus 2,048). Bucket 3.

Anyone who predicted bucket 1 or 2 made the textbook error of reading a work
complexity as a runtime. Anyone who predicted bucket 4 under-weighted the fact
that Hillis–Steele's extra work is *real* — it is 8,000 extra shared-memory
accesses per tile, and shared memory is not free.

---

## Synchronization / memory reasoning

Barrier count per 1024-element tile scan:

| algorithm | `__syncthreads()` | why |
|---|---|---|
| Hillis–Steele, double-buffered | 10 + 2 | one per level, plus two for the exclusive shift |
| Hillis–Steele, single-buffer + temp | 20 + 2 | two per level |
| Blelloch | 10 + 10 + 1 | one per level each phase, plus one after the identity write |
| warp-shuffle | 2 | only the cross-warp step needs one |

Every barrier in all four is reached by the whole block: the inner loops
`for (i = tid; i < TILE; i += BLK)` have a trip count that depends only on
`TILE` and `BLK`, not on `tid`, and the barriers are outside them. The one place
this could go wrong is the Blelloch loops `for (t = tid; t < d; t += BLK)` where
`d` shrinks below `BLK`: threads with `tid >= d` execute the loop zero times, but
they still *reach* the barrier at the top of the next iteration, because the
barrier is outside the inner loop and the outer loop's trip count is uniform.
M9's uniformity rule is satisfied. Moving the barrier inside the inner loop would
be undefined behaviour and, per M9's measurement, would most likely corrupt
silently rather than hang.

The warp scan contains no `__syncwarp()` because `__shfl_up_sync` carries its own
synchronization: the `_sync` suffix and the `0xffffffff` mask are what create the
convergence that independent thread scheduling (M8) removed the guarantee of.

---

## Performance reasoning

The harness deliberately reports two numbers per algorithm: the tile-scan kernel
alone (2N of traffic for all three, so the comparison isolates the algorithm) and
the full three-kernel scan (4N).

At **N = 1,048,573** the whole working set is 8 MB, comfortably inside the 48 MB
L2, so the tile scan is not DRAM-bound and the algorithm is visible. At
**N = 67,108,861** it is 512 MB and everything is pressed against the DRAM roof.
That is why the ratio between the same two algorithms is 1.48× at 1 M and 1.22×
at 64 M. **The algorithmic difference does not disappear — it becomes
unexploitable**, because the memory system is the constraint. This is the same
lesson M6 delivered about tiling a stencil (a kernel already at 70 % of DRAM peak
has a hard 1.4× ceiling no matter what you do on chip).

The `>100 % of peak` entries at 1 M are L2 hits, labelled as such per the house
rule, not bandwidth claims.

---

## Expected output

Actual run on the RTX 3500 Ada, CUDA 13.2:

```
Module 13 / Exercise 1 — the scan ladder

N = 1,048,573 (1 M, L2-resident) : 1024 tiles of 1024, last tile holds 1021
  algorithm         tile ms    GB/s 2N     %peak |    full ms    GB/s 4N     %peak   valid
  Hillis-Steele      0.0332      253.0     58.6% |     0.0458      366.7     84.9%    PASS
  Blelloch           0.0223      375.4     86.9% |     0.0357      469.3    108.6%    PASS
  warp-shuffle       0.0109      771.9     178.7% |    0.0218      768.7    177.9%    PASS
  2N floor = 0.0194 ms; this 3-kernel structure moves 4N, floor 0.0388 ms

N = 67,108,861 (64 M, DRAM-resident) : 65536 tiles of 1024, last tile holds 1021
  algorithm         tile ms    GB/s 2N     %peak |    full ms    GB/s 4N     %peak   valid
  Hillis-Steele      2.0514      261.7     60.6% |     4.5048      238.4     55.2%    PASS
  Blelloch           1.6878      318.1     73.6% |     3.3978      316.0     73.2%    PASS
  warp-shuffle       1.5791      340.0     78.7% |     3.2425      331.1     76.7%    PASS
  2N floor = 1.2428 ms; this 3-kernel structure moves 4N, floor 2.4855 ms

Predictions (scored at N = 1,048,573, tile-scan kernel only)
  slowest tile scan     : measured H   predicted H
  Hillis-Steele/Blelloch: measured 1.48x -> bucket 3   predicted 3
  Blelloch vs warp scan : 2.06x apart (not scored, but look at it)
  work ratio for reference: 10240 adds/tile vs 2048 adds/tile = 5.0x
  same ratio at 64 M, where DRAM is the wall: 1.22x — the algorithm
  difference is real but mostly invisible once you are bandwidth-bound.

  correctness 6/6, slowest prediction correct, ratio prediction correct
  score: 8/8
OVERALL: PASS
```

**Run-to-run variation.** Absolute times move by up to 30 % with thermal state on
this laptop part — the 64 M warp-shuffle tile scan has been observed between
1.44 ms and 2.05 ms across runs. The scored quantities are ratios for that
reason, and both stay well inside their buckets: across seven runs,
Hillis-Steele/Blelloch at 1 M measured **1.39–1.49x** (bucket 3 spans 1.2–2.0)
and the slowest was Hillis-Steele every time. Blelloch/warp-shuffle at 1 M is
noisier, 1.39–2.27x, which is why it is printed but not scored. The 64 M table
is noisier still: the three fastest sit within 10 % of each other, the
Hillis-Steele/Blelloch ratio moves between 1.18x and 1.29x, and the
Blelloch-versus-warp ordering genuinely flips between runs — which is itself the
result the exercise is trying to deliver.

---

## The result that matters

**A 5× reduction in arithmetic bought 1.48×, and only 1.22× once the array was
big enough to be DRAM-bound.** Scan is a bandwidth problem wearing an algorithms
problem's clothes, and the classical work/depth analysis — which is a good
predictor on a PRAM — is a weak predictor on a machine with 40 SMs, a 432 GB/s
pipe, and a shared-memory crossbar that charges for bank conflicts. The three
things that actually moved the needle were the ones the complexity analysis does
not see: bank conflicts (1.31×), barrier count (Blelloch's 21 versus the warp
scan's 2), and where the data lives.

**Variation to try:** replace the double-buffered Hillis–Steele with the
single-buffer-plus-temporary version (`u32 t = s[i-off]; __syncthreads();
s[i] += t; __syncthreads();`), which halves the shared-memory footprint and
doubles the barrier count, and re-measure. Then raise `TILE` to 2048 and rerun
both — the shared footprint doubles, blocks/SM starts to matter, and the ranking
changes in a way that is worth being able to explain.
