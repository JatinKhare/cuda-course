# Module 10 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

---

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

The requirement was deliberately semantic — *"reduce the number of concurrent
global atomic operations that target the same address, without changing the
result"* — and the four scenarios are chosen so that no single reflex wins
all four. What follows is one correct answer, not the only one; a solution
built on warp-level pre-aggregation plus a runtime switch will also pass, and
the comparison is discussed below.

---

## TODO 1 — the launch configuration, which is two decisions

```cpp
const int blk = 256;
*block = blk;
const size_t priv = (size_t)ncat * sizeof(unsigned int);

if (priv <= (size_t)SMEM_LIMIT) {
    *smemBytes = priv;
    long long byWork    = ((long long)n + (long long)blk*8 - 1) / ((long long)blk*8);
    long long byTraffic = (long long)n / (16LL * (long long)ncat);
    long long g = byWork < byTraffic ? byWork : byTraffic;
    if (g < 80)    g = 80;
    if (g > 65535) g = 65535;
    *grid = (int)g;
} else {
    *smemBytes = 0;
    *grid = (int)(((long long)n + blk - 1) / blk);
}
```

**A private bin is 32 bits, not 64.** The output is `unsigned long long`
because the *global* totals can reach 8×10^9. A *per-block partial* cannot:
the maximum weight is 1000, so a block would need 4,294,968 samples to
overflow 32 bits, and with any grid of more than two blocks no block sees
that many. Halving the bin width halves the shared-memory footprint (which
determines how many blocks are resident) and lets the accumulation use the
cheaper 32-bit `ATOMS` path. This is a genuine decision, and getting it wrong
costs 2× the shared memory for nothing.

**The grid is capped by traffic, not just by work.** Privatization converts
`n` global atomics into `grid × ncat` of them. With the reflexive
`grid = ceil(n / block)` = 31,250 and one sample per thread:

| ncat | `grid × ncat` | vs n = 8,000,000 |
|---|---|---|
| 16 | 500,000 | 16× fewer |
| 4096 | **128,000,000** | **16× more** |

Capping the grid at `n / (16·ncat)` keeps the flush under `n/16` in every
case. Measured choices: 3,907 blocks at `ncat = 16` (62,512 flush atomics)
and 122 blocks at `ncat = 4096` (499,712). The floor of 80 blocks keeps at
least two blocks per SM so the machine is not left idle.

**A measured surprise — read this one carefully.** I predicted that the
uncapped grid would be a disaster at `ncat = 4096`. It is not. Three variants,
measured back to back on this GPU:

| variant | `grid` at ncat=4096 | flush atomics | ms | speedup vs naive |
|---|---|---|---|---|
| capped grid, 8 samples/thread | 122 | 499,712 | 0.2047 | 3.11× |
| uncapped, 8 samples/thread | 3,907 | 16,003,072 | 0.2038 | 3.12× |
| uncapped, 1 sample/thread | 31,250 | 128,000,000 | 0.3798 | 1.67× |

Sixteen million extra global atomics cost **nothing measurable**, and 128
million cost only 1.9×. Both numbers are far better than the naive traffic
argument predicts, and the explanation is in this module's own contention
table: the flush targets 4096 distinct, uncontended addresses, so each atomic
costs what a plain store costs (lesson Part A, atomic/store ratio 0.9× at
K = 4096) — and those 4096 words are 32 KB, permanently L2-resident, so the
traffic never reaches DRAM. The kernel remains bandwidth-bound on its 64 MB
of input either way.

So: the traffic cap is *defensible* (32× fewer global atomic instructions,
and it would matter on a part with less L2 or with more bins) but it is **not
what makes this kernel fast**. What makes it fast is moving the *contended*
updates into shared memory. The grid cap is an optimization of something that
was not the bottleneck, and the honest conclusion is that the third row's 1.9×
comes mostly from processing one sample per thread rather than eight — i.e.
from work per thread, not from atomic traffic.

Removing the `if (s[t] != 0u)` guard from the uncapped variant costs only 7%
(0.2038 → 0.2186 ms), which corroborates the same story.

---

## TODO 2 — the kernel

```cpp
extern __shared__ unsigned int s[];
const int stride = gridDim.x * blockDim.x;
const int i0     = blockIdx.x * blockDim.x + threadIdx.x;

if ((size_t)ncat * sizeof(unsigned int) > (size_t)SMEM_LIMIT) {
    for (int i = i0; i < n; i += stride)
        atomicAdd(&total[cat[i]], (unsigned long long)w[i]);
    return;
}

for (int t = threadIdx.x; t < ncat; t += blockDim.x) s[t] = 0u;
__syncthreads();

for (int i = i0; i < n; i += stride)
    atomicAdd(&s[cat[i]], w[i]);
__syncthreads();

for (int t = threadIdx.x; t < ncat; t += blockDim.x)
    if (s[t] != 0u) atomicAdd(&total[t], (unsigned long long)s[t]);
```

**Why it is correct.** Each block accumulates a complete, private partial sum
of the samples it visited; the grid-stride loop partitions the input exactly,
so every sample is visited by exactly one thread; integer addition is
associative and exact, so the order in which the partials are folded into
`total` is irrelevant and the result is bit-identical every run. (Contrast
with a `float` accumulation, where this argument fails — Example 1 Part E.)

**Why the two barriers are both needed and are not interchangeable.** The
first makes the zeroing visible before any add; the second makes every add
complete before any read. They bracket different intervals. Neither removes
the need for `atomicAdd` in the middle: barriers order threads, not
sub-instruction windows.

**`if (s[t] != 0u)` in the flush.** At `ncat = 4096` with 122 blocks, each
block sees ~65,000 samples spread over 4096 bins, so most bins are non-zero
and this skips little. At `ncat = 4096` with a *sparse* category
distribution it would skip most of the flush. It costs one predicated compare
and can save a global atomic, so it is never a loss.

**The fallback path** matters even though the harness never triggers it
(`4096 × 4 = 16 KB < 48 KB`). A kernel that indexes `s[]` when
`smemBytes == 0` is an out-of-bounds shared access — `compute-sanitizer
--tool memcheck` catches it as `Invalid __shared__ write`. Ship the branch.

**Common wrong approaches and their symptoms:**

| approach | symptom |
|---|---|
| `grid = ceil(n/block)`, one sample per thread | Correct, and **1.67× instead of 3.11× at `ncat = 4096`** — fails the harness's 2.50× gate. Less because of the 128 M flush atomics than because 31,250 blocks of one-sample threads amortise the zero-and-flush over almost no work. |
| `unsigned long long` private bins | Passes, uses 32 KB at `ncat = 4096`, drops to 3 resident blocks/SM, measurably slower. |
| Privatize unconditionally, no `ncat` check | Compiles; at `ncat` large enough to exceed 48 KB the launch fails with `cudaErrorInvalidValue`, reported by `cudaGetLastError()`. |
| Drop the first `__syncthreads()` | Intermittently low counts; `racecheck` names it immediately. |
| Warp-aggregation only (`__match_any_sync`), no privatization | Wins big on `clustered` (lanes collide, so aggregation has something to aggregate) and **nothing at all on `shuffled`**, where a warp's 32 lanes hit 32 different bins and `MATCH.ANY` finds no groups. Fails the ≥2× shuffled test. This is what the two input orders are for. |

That last row is the design insight the exercise is built around:
**privatization attacks contention between blocks; warp aggregation attacks
contention within a warp. They are not substitutes.** Privatization happens
to cover both here, because moving the contention into shared memory makes
the within-warp collisions cheap as well (`ATOMS.POPC.INC`).

---

## TODO 3 — the traffic model

```cpp
if ((size_t)ncat * sizeof(unsigned int) > (size_t)SMEM_LIMIT) return (double)n;
return (double)grid * (double)ncat;
```

`grid × ncat` is an upper bound (the `s[t] != 0` test can only reduce it).
Against the naive kernel's `n = 8,000,000`:

| scenario | model | ratio to naive | measured speedup |
|---|---|---|---|
| ncat = 16 | 62,512 | 128× fewer | 20.9× / 16.6× |
| ncat = 4096 | 499,712 | 16× fewer | 3.1× / 3.6× |

The measured speedup is far below the traffic ratio, and that is the right
answer rather than a disappointment: the fast kernel is **no longer atomic-
bound at all**. It reads `cat[]` and `w[]` — 8 bytes × 8,000,000 = 64 MB —
and runs at 321 GB/s, **74% of the 432 GB/s peak**. It has hit the memory
wall. Removing the remaining 62,512 atomics entirely would change nothing.

That is the correct place to stop optimizing, and knowing *why* you stopped
is the difference between engineering and guessing.

---

## TODO 4 — the two predictions

Measured: **20.9×** at `ncat = 16` clustered, **3.1×** at `ncat = 4096`
clustered. Anything in [10.4, 41.8] and [1.6, 6.2] scores.

The reasoning that gets you there without running anything:

- **ncat = 16.** From the lesson's contention table, 16 hot bins put the
  naive kernel around 10 Gatomic/s. But the distribution is skewed: one
  category carries 49.9% of the traffic, so half the atomics are effectively
  at K = 1 (≈2 Gatomic/s) and the rest at K ≈ 15. The privatized version is
  bandwidth-bound at ~0.2 ms. Naive ≈ 4 ms, so ≈ 20×. Predicting 10–15×
  from the flat-distribution table alone is also within the window.
- **ncat = 4096.** The naive kernel is already near the flat part of the
  curve (lesson: atomic/store ratio 0.9× at K = 4096), so it is close to
  bandwidth-bound and there is at most a small factor available. A prediction
  of 1–2× is the honest reading; 3.1× is a little better than that because
  the *clustered* input still forces same-address collisions within each warp,
  which the lesson's Part C table shows costs ~6× even at large K.

**The trap.** A prediction below 1.0 for `ncat = 4096` is the reasonable
inference from the traffic argument alone — and it is wrong, for the reason
documented under TODO 1: uncontended global atomics against a small,
L2-resident bin array are nearly free, so even 16× *more* of them costs
nothing. If you predicted a regression and did not get one, that is the most
valuable thing this exercise will teach you: the traffic count is a proxy for
cost, and it is only a good proxy while the addresses are contended.

---

## Synchronization / memory reasoning

Three levels of contention are in play and the fix at each level is different:

| level | contention between | mechanism | cost after |
|---|---|---|---|
| within a warp | 32 lanes, same bin | `ATOMS` + hardware popc | ~free |
| within a block | 8 warps, same bin | shared-memory atomic | SM-local, ~40-cycle |
| across blocks | 3907 blocks, same bin | one flush per block | 62,512 global atomics |

The lesson's claim that "shared-memory atomics are 48× cheaper" is what makes
the middle row acceptable. The top row is a hardware gift specific to
Ada/Ampere/Turing shared memory (**ARCHITECTURE-SPECIFIC**); the portable
statement is just that shared atomics are much cheaper than global ones.

---

## Performance reasoning

Observed on the RTX 3500 Ada. Ratios are stable to ~2% across thermal states;
the absolute ms drift.

```
=== timing (min of 4 sweeps x 20 iters, all configs back to back) ===
  scenario               naive ms    fast ms   speedup fast GB/s    %peak     your model
  ncat=16   clustered      4.1608     0.1992    20.89x     321.3    74.4%          62512
  ncat=16   shuffled       3.2938     0.1990    16.56x     321.7    74.5%          62512
  ncat=4096 clustered      0.6370     0.2041     3.12x     313.6    72.6%         499712
  ncat=4096 shuffled       0.7367     0.2041     3.61x     313.6    72.6%         499712
  (naive global atomic instructions per launch: 8000000 in every row)

  grid/block/smem chosen: [3907,256,64B] [3907,256,64B] [122,256,16384B] [122,256,16384B]

=== scoring ===
  [PASS] correct in all four scenarios (4/4)
  [PASS] ncat=16 clustered at least 4.00x  (got 20.89x)
  [PASS] ncat=16 shuffled  at least 2.00x  (got 16.56x)
  [PASS] ncat=4096 at least 2.50x on BOTH orders (got 3.12x, 3.61x)
  [PASS] TODO 4 predictions (2/2)

  score: 9/9
OVERALL: PASS
```

Four things are worth reading off this table.

1. **The naive kernel is 6.5× slower at `ncat = 16` than at `ncat = 4096`**
   (4.16 vs 0.64 ms) for identical work. That is the contention curve,
   reproduced on a real workload.
2. **Clustering hurts the naive kernel at `ncat = 16` and helps it at
   `ncat = 4096`.** At 16 bins, clustered is 4.16 ms against shuffled's
   3.29 ms: a warp whose 32 lanes share a bin is the expensive case (lesson
   Part C). At 4096 bins the order flips — 0.64 clustered against 0.73
   shuffled — because with that many bins the collisions are cheap and what
   dominates instead is locality: clustered input keeps a warp's atomics
   inside one or two L2 slices' worth of address space. Two effects of
   opposite sign, and which one wins depends on the bin count. The input
   order is not a one-line story.
3. **The fast kernel is 0.199–0.205 ms in all four scenarios.** It has
   stopped caring about `ncat` and about the input order entirely, because it
   is bandwidth-bound. That flatness is the real result.
4. **74% of peak is the right number to expect.** The kernel streams two
   `uint32` arrays — 32 MB each, 64 MB total, comfortably past the 48 MB L2,
   so this is a genuine DRAM figure and not a cache artefact. It is not 100%
   because the shared-memory scatter has its own serialization and because
   the access is a gather-free but data-dependent scatter.

---

## Expected output

The full run is quoted above. `OVERALL: PASS` with `score: 9/9`.

Run-to-run variation on this laptop: naive `ncat=16 clustered` ranges
4.16–4.18 ms, fast ranges 0.199–0.206 ms, speedup 20.8–20.9×. The ratio is
the stable quantity, as spec §12 requires.

---

## The result that matters

Privatization is not a technique you apply to atomics; it is a technique you
apply to **contention**, and the harness's `ncat = 4096` column exists to
make that distinction cost you something. The identical kernel that wins 21×
when 16 addresses are hot loses 16× when 4096 are, because `grid × bins`
crossed `n` and the "optimization" started manufacturing more global atomics
than it removed. The number to compute before writing any privatized kernel
is `grid × bins` versus `n`; if it is not much smaller, stop.

**Variation to try:** replace the shared-memory privatization with
`__match_any_sync`-based warp aggregation, keep everything else, and run all
four scenarios. You should see a large win on both `clustered` columns and
almost nothing on the `shuffled` ones — and then combine the two and watch
the combination be slower than privatization alone at `ncat = 16`, because
the aggregation is now working on contention that shared memory already made
free.
