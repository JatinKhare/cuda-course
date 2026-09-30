# Module 14 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

---

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

65,536 bins, 2^26 32-bit keys, 64 MB of device scratch, and a requirement rather
than a technique. What follows is one correct answer. A solution built on global
privatization *without* warp aggregation also passes; a solution built on
multi-pass shared-memory windows does not, and that is what the two-sided gate
is for.

---

## TODO 1 — the two predictions

Measured:

| distribution | speedup of fast over naive |
|---|---|
| flat (uniform over 65,536 bins) | **0.99×** |
| hot (75% of the mass in 32 bins) | **9.68×** |

Anything in [0.5, 2.0] and [3.5, 14] scores.

**The flat column is the whole exercise.** The reasoning that gets you there
without running anything:

Both columns execute exactly 67,108,864 global atomics on exactly the same
65,536 addresses. Only the *distribution over those addresses* differs. From
this module's lesson §7(d) and Module 10's contention table, an uncontended
global atomic costs about what a plain store costs. With 65,536 bins and a flat
input, essentially nothing is contended: the harness reports the naive kernel at
**35% of the measured streaming ceiling** — already within 2.9× of the floor.
An infinitely good contention fix could buy at most 2.9×, and in practice buys
**nothing**, because what is left is not contention. It is the sector-spread
effect of lesson §2: a warp's 32 atomics hit 32 distinct 32 B sectors, costing
8× a packed pattern, and no amount of privatization changes the number of
sectors a warp touches.

The hot column is the opposite: 75% of the atomics pile onto 32 addresses, so
the naive kernel sits at **5% of ceiling** and there is a factor of ~20
theoretically available. 9.68× is a good fraction of it.

**The trap.** Almost everyone predicts a large win on both, because the module
has spent 200 lines teaching that privatization is worth 50–200×. Those figures
are all from the *256-bin* setting, where there are 256 addresses for 61,440
threads. At 65,536 bins the arithmetic is different and so is the answer.

---

## TODO 2 — the accumulation pass

Two independent levers, composed.

**Lever 1 — privatization one level out.** The histogram does not fit in shared
memory; it fits in *global* memory many times over. Keep `G` copies and have
block `k` use copy `k & (G-1)`:

```cpp
__global__ void fast_accum(const uint4* __restrict__ in, size_t n4,
                           unsigned int* __restrict__ part, int nBins, int gmask)
{
    unsigned int* my = part + (size_t)((int)blockIdx.x & gmask) * nBins;
    ...
}
```

This is the same idea as shared-memory privatization with a different venue and
a different cost. It does not make the atomics cheaper *per operation* — they
are still `RED.E.ADD.STRONG.GPU` at the L2 — but it divides the number of
concurrent requests for any one bin by up to `G`.

**Lever 2 — warp-level pre-aggregation.** Module 10 gave the idiom and measured
where it stops paying (27.7× at `K = 1`, 0.46× at `K = 1024`):

```cpp
__device__ __forceinline__ void bump(unsigned int* base, unsigned int b)
{
    unsigned int m = __match_any_sync(0xffffffffu, b);
    unsigned int leader = (unsigned int)__ffs((int)m) - 1u;
    if ((threadIdx.x & 31u) == leader)
        atomicAdd(&base[b], (unsigned int)__popc((int)m));
}
```

The compiler will **not** do this for you here: lesson §"Hardware Mental Model"
shows there is no `VOTEU.ANY` anywhere in a histogram's SASS, because
`hist[in[i]]` cannot be proved warp-uniform. On the `hot` input a warp's 32 lanes
frequently share one of 32 bins, so `__match_any_sync` finds real groups to
collapse. Measured contribution on top of global privatization: **5.44× → 7.59×
at `G = 16`**, i.e. an extra 1.40×.

They attack different levels and they compose. Module 10's Exercise 2 solution
said it directly: *privatization attacks contention between blocks; warp
aggregation attacks contention within a warp; they are not substitutes.* At
65,536 bins there is no shared-memory step to make the within-warp case free, so
for the first time in the course you need both.

**Note the full mask.** `__match_any_sync(0xffffffffu, b)` requires every lane
of the warp to participate. The loop here is a `uint4` grid-stride loop with no
divergent guard, so all 32 lanes are converged at the call site. If you add a
tail guard `if (i < n4)` *around* the `bump` calls, the full mask is wrong and
the result is undefined on the last iteration (Module 8). The harness's `n4` is
exact, so there is no tail — but if you generalise this kernel, that is where it
will break.

---

## TODO 3 — the plan (design)

```cpp
static Plan planFast(int nBins, size_t, int nSM, int blocksPerSM)
{
    Plan p;
    p.block  = 256;
    p.grid   = nSM * blocksPerSM;              // 240
    p.copies = 16;
    while (p.copies > 1 && (size_t)p.copies * nBins * 4u > (8u << 20)) p.copies >>= 1;
    if (p.copies > p.grid) p.copies = p.grid;
    p.scratchBytes = (size_t)p.copies * nBins * sizeof(unsigned int);
    return p;
}
```

`G = 16` at 65,536 bins is **4 MB** — 8% of the 48 MB L2. That constraint is the
design decision and it has a wrong answer on *each* side.

Measured sweep on the `hot` input (same harness, same warm-up):

| G | scratch | ms | vs naive |
|---|---|---|---|
| 8 | 2 MB | 3.73 | 3.41× |
| 16 | 4 MB | 2.34 | 5.44× |
| 64 | 16 MB | 2.03 | 6.27× |
| **256** | **64 MB** | **4.92** | **2.58×** |

and with aggregation on top: `G = 16` → 7.59×, `G = 64` → 7.05×,
`G = 256` → **2.05×**.

**Too few copies** leaves contention on the table (G = 8 is 1.6× worse than
G = 16). **Too many** falls off a cliff: at `G = 256` the privatized array is
64 MB, larger than the 48 MB L2, and every "cheap uncontended atomic" becomes a
DRAM round trip. On the `flat` input `G = 256` measures **0.10× of naive** — ten
times *slower* than doing nothing.

This is Module 4's 48 MB L2 benchmarking hazard reappearing as a design rule:
**a privatized array is only cheap while it is cache-resident.** Module 10's
Exercise 2 found the same thing from the other direction — 16 million extra
flush atomics against a 32 KB L2-resident bin array cost *nothing measurable* —
and this is the point at which that observation acquires a boundary.

The grid is a machine-sized 240 blocks for the usual reason (lesson §4): the
accumulate pass is a coarsened grid-stride loop and the fold pass is
independent, so there is no `grid × nBins` flush term to minimise here — the
fold traffic depends on `G`, not on the grid. Note that this is a genuine
difference from Exercise 1, where the grid *was* the dominant term.

---

## TODO 4 — the fold, and the orchestration

```cpp
__global__ void fast_fold(const unsigned int* __restrict__ part,
                          unsigned int* __restrict__ hist, int nBins, int G)
{
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nBins;
         b += gridDim.x * blockDim.x) {
        unsigned int t = 0u;
        for (int g = 0; g < G; ++g) t += part[(size_t)g * nBins + b];
        hist[b] = t;
    }
}

static void runFast(const Plan& p, const uint4* in4, size_t n4,
                    unsigned int* hist, unsigned int* scratch, int nBins)
{
    CHECK(cudaMemsetAsync(scratch, 0, p.scratchBytes));
    fast_accum<<<p.grid, p.block>>>(in4, n4, scratch, nBins, p.copies - 1);
    fast_fold<<<(nBins + p.block - 1) / p.block, p.block>>>(scratch, hist, nBins, p.copies);
}
```

The traffic the harness prints:

```
cost model: 6.71e+07 global atomics from the accumulate pass (upper bound),
            4.19e+06 B read + 2.62e+05 B written by the fold pass,
            against 6.71e+07 global atomics for the naive kernel.
```

The fold reads 4 MB and writes 256 KB against an input of 256 MB: **1.7% of the
input traffic**, and it is L2-resident, so it does not touch DRAM at all. The
`cudaMemsetAsync` is inside the timed region and is the same 4 MB again.

The fold is a reduction across `G` values per bin, and it is Module 12's
territory: a fixed-order integer sum, associative and exact, so the answer is
bit-identical every run regardless of block scheduling.

**Three wrong orchestrations and their symptoms:**

| mistake | symptom |
|---|---|
| forget the memset | counts monotonically increase across the 3 validation repeats and across timed iterations; the timed loop hides it because only the last result is checked |
| `hist[b] +=` instead of `hist[b] =` in the fold | correct on the first call, 20× too high after the timing loop; the harness's `cudaMemset(d_hist,...)` in the validation pass hides it there but not in the total |
| memset the whole 64 MB budget instead of `p.scratchBytes` | correct, and adds ~0.16 ms — 12% of the fast kernel's total runtime |

---

## Synchronization / memory reasoning

There are no barriers in this solution and that is worth noticing. The
accumulation pass uses no shared memory at all; the only synchronization is the
**kernel boundary** between `fast_accum` and `fast_fold`, which Module 9
established as the only grid-wide barrier available without a cooperative
launch. The atomics in `fast_accum` need no ordering with respect to anything —
they are relaxed, device-scope, and the fold does not run until the accumulate
kernel has retired.

`__match_any_sync` carries its own warp-level synchronization (Module 8's
`_sync` contract), so no `__syncwarp` is needed around it.

---

## Performance reasoning

Observed on a thermally settled machine:

```
your plan: grid 240 x 256, 16 copies, 4194304 B scratch (4.00 MB, 8% of L2)

--- timing (min of 6 rotated sweeps, ms) ---
                                 flat        hot
  ceiling (stream only)        0.6539     0.6532
  hist_naive                   1.8568    12.9556
  your fast version            1.8683     1.3382
  speedup                       0.99x      9.68x
  % of ceiling (fast)             35%        49%
  % of ceiling (naive)            35%         5%

  ceiling 0.6539 ms = 410.5 GB/s = 95% of 432 GB/s peak

--- validation (separate pass) ---
  [PASS] flat  : 0/65536 bins wrong, total 67108864 (expected 67108864)
  [PASS] hot   : 0/65536 bins wrong, total 67108864 (expected 67108864)
  [PASS] hot  distribution at least 5.00x (got 9.68x)
  [PASS] flat distribution no worse than 0.90x (got 0.99x)
  TODO 1 predictions (octave-scored):
    [PASS] flat  predicted 1.00x, measured 0.99x
    [PASS] hot   predicted 7.00x, measured 9.68x

  score: 5/5
OVERALL: PASS
```

Four readings.

1. **The ceiling is 410.5 GB/s, 95% of peak** — the same figure Module 12
   recorded after a 1500 ms warm-up, reproduced independently here. Absolute ms
   degrade to 2.1 ms (126 GB/s) if the GPU has been streaming for a few minutes;
   the ratios above reproduce to ~3% in either state.
2. **Both versions are at 35% of ceiling on `flat`.** Neither is bandwidth-bound
   and neither is contention-bound; both are limited by the sector spread of
   their atomics (lesson §2), which is a property of the *access pattern*, not
   of the strategy. This is the single most useful negative result in the
   exercise.
3. **The fast version is at 49% of ceiling on `hot` and only 35% on `flat`.**
   It is *faster in absolute terms* on the skewed input (1.34 vs 1.87 ms). Skew
   concentrates the atomics onto few bins, so more of a warp's requests coalesce
   into the same sectors and `__match_any_sync` finds more to collapse. **For a
   privatized histogram, a skewed distribution is the easy case.** It is only
   the hard case for the naive one. That inversion is worth internalising.
4. The reflexive answer — take the multi-pass shared-memory approach from
   lesson §7(b), 8 windows of 8,192 bins — measures **2.17× on hot and 0.46× on
   flat** and fails both gates. It reads the input eight times to save atomics
   that were not the bottleneck.

---

## Expected output

Quoted in full above. `score: 5/5`, `OVERALL: PASS`.

Run-to-run: `hot` speedup 9.2–9.9×, `flat` 0.95–1.00× across five runs in
different thermal states. The `flat` figure sitting slightly below 1.00× is
real and reproducible — the memset and the fold are not free — and the gate is
set at 0.90× for that reason.

---

## The result that matters

The gate has two sides because a design exercise with one side teaches you to
optimize, and a design exercise with two sides teaches you to *decide*. The
strategy that wins 9.7× on a skewed input is worth **0.99×** on a flat one with
the same bin count, the same kernel and the same number of atomics. If you ship
the first measurement and not the second, you have shipped a kernel whose
performance you cannot predict.

The quantity to compute before writing anything is the naive version's
**percentage of the streaming ceiling**. At 5% there is 20× on the table. At 35%
there is at most 2.9×, and probably none, because whatever is costing you the
other 65% is not contention.

**Variation to try:** set `p.copies = 256` (64 MB, the whole budget) and re-run.
You will measure 0.10× on `flat` — ten times slower than the naive kernel — and
the explanation is a single number: 64 MB against a 48 MB L2. Then walk it back
to 128, 64, 32 and find the knee. It is the most instructive five minutes in
this module.
