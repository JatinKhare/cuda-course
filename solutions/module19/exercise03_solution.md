# Module 19 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
```

The run takes a couple of minutes. Warm-up is 1500 ms streaming plus 500 ms
compute (spec §12.4); there are two operating-point guards (see the last
section, which is a methodology note worth reading); every configuration is
min-of-5; validation is a separate untimed pass.

---

## TODO 1 — the instrument

```cpp
__device__ __forceinline__ void profIn(Prof p, unsigned long long &t0, unsigned &sm)
{
    if ((threadIdx.x & 31) == 0) {
        sm = smId();
        t0 = clk();
        atomicMin(&p.lo[sm], t0);
    }
}
__device__ __forceinline__ void profOut(Prof p, unsigned long long t0, unsigned sm)
{
    if ((threadIdx.x & 31) == 0) {
        unsigned long long t1 = clk();
        atomicMax(&p.hi[sm], t1);
        atomicAdd(&p.cyc[sm], t1 - t0);
    }
}
```

Four decisions, each of which has a wrong version that compiles and runs.

**One reporter per warp.** Residency is a per-warp property — a warp slot is
occupied or it is not, and the 32 lanes inside it are not separately resident.
Letting every thread report multiplies every quantity by 32 (occupancies come
out above 100 %, which is the tell) and multiplies the atomic traffic by 32 as
well. `(threadIdx.x & 31) == 0` selects lane 0 of each warp, which is correct
for any block size because Module 3's linearisation rule makes `threadIdx.x`
contiguous within a warp for a 1-D block.

**`min` on entry, `max` on exit.** The SM's span is from the first warp that
started to the last that finished. Swapping them gives a negative span, and
because the arrays are unsigned a negative span is an enormous positive number,
so the occupancy comes out at about 1e-10 rather than failing loudly.

**`atomicAdd` of the per-warp lifetime, not of the timestamps.** The numerator
is `Σ (t1 − t0)`, warp by warp. Accumulating `t1` and `t0` into separate sums
and subtracting at the end is algebraically identical and works; accumulating
`t1 − lo[sm]` does not, because `lo` is not final yet.

**Never compare timestamps across SMs.** The per-SM cycle counters are not
synchronised. Measured on this part while authoring: the earliest `clock64()`
values recorded by different SMs during a single launch were **264.378e9 and
264.676e9**, nearly 300 million cycles apart. Every quantity in this file is
accumulated per SM for exactly that reason, and the kernel-wide denominator in
`occElapsed` comes from the host's event timer and a recovered clock rather than
from subtracting one SM's timestamp from another's.

**The `clk()` helper matters.** It is `clock64()` through inline PTX with a
`"memory"` clobber:

```cpp
asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
```

Without the clobber the compiler is entitled to hoist the exit read above the
work it is supposed to be measuring. (On this kernel, measured both ways, it
does not — but "it happens not to today" is not a property you want a
measurement harness to rest on.)

**Calibration.** The harness launches exactly one block per SM — 40 blocks of
256 threads — which makes the answer known by construction: 8 warps of 48 are
resident for the whole kernel, so `occ_active` must be 16.7 %. Measured 16.7 %
with all 40 SMs reporting. A mis-wired instrument cannot pass that.

---

## TODO 2 — the two denominators

```cpp
static double occActive(const ProfData *d)
{ return d->sumCyc / (d->sumSpan * WARPS_PER_SM); }

static double occElapsed(const ProfData *d, double kernelMs, double clockHz)
{ return d->sumCyc / ((double)SM_COUNT * WARPS_PER_SM * (kernelMs * 1e-3) * clockHz); }
```

Same numerator; the denominators answer different questions.

- `occActive` divides by `Σ_SM (span_SM × 48)` — the warp-cycles the SMs could
  have held **while they were busy**. An SM that got no work contributes nothing
  to either half and drops out of the average entirely.
- `occElapsed` divides by `40 × 48 × kernelCycles` — the warp-cycles the whole
  machine could have held for the whole kernel, idle SMs included.

Nsight Compute reports both:
`sm__warps_active.avg.pct_of_peak_sustained_active` and
`..._pct_of_peak_sustained_elapsed`. The first is what is labelled "Achieved
Occupancy" in the Occupancy section and is the one every tutorial quotes.
`ncu` cannot be run on this machine (`ERR_NVGPUCTRPERM`, spec §12), which is
why the counter is built here instead; the construction is the same one the
hardware counter performs.

The clock is recovered from the busiest SM's span divided by the measured wall
time of the most balanced launch — **2.10 GHz**, reproducibly, across every run.
`cudaDevAttrClockRate` reports 1.545 GHz on this GPU (spec §12 rule 6) and using
it would inflate every `occ_elapsed` figure by 36 %.

---

## TODO 3 — the grid-size bound

```cpp
double wave = (double)blocksPerSM * nsm;
double f = (double)grid / wave;
return (f > 1.0) ? 1.0 : f;
```

A grid of `G` blocks cannot occupy more warp slots than it has blocks to put in
them, so

```
achieved <= theoretical * min(1, G / (blocksPerSM * nSM))
```

This is a bound you can compute before writing a profiler, and on an
under-filled grid it is nearly exact. Measured, uniform cost, theoretical 100 %:

| grid | waves | bound | occ_active |
|---|---|---|---|
| 60 | 0.25 | 25 % | **23.9 %** |
| 120 | 0.50 | 50 % | 36.6 % |
| 240 | 1.00 | 100 % | 61.9 % |
| 360 | 1.50 | 100 % | 77.4 % |
| 480 | 2.00 | 100 % | 75.0 % |

The scoring is deliberately asymmetric: the bound must **never be exceeded** on
any row, and on the under-filled rows it must be within 40 %. A reader who
writes `min(1, grid/nsm)` (forgetting blocks per SM) predicts 150 % at grid 60
and fails the upper-bound test on every row; a reader who writes
`grid/(blocksPerSM*nsm)` without the `min` predicts 200 % at two waves and fails
there.

**The gap above one wave is the interesting part**, and it is what the lesson
calls the documented surprise. A full wave, every block resident from the first
cycle, zero tail, zero imbalance, all 40 SM spans within 1 % of one another —
and it achieves 62 % of its theoretical occupancy, not 100 %. The cause is
inside one SM: the warp scheduler is greedy, so warps executing *identical*
work finish at very different times and the slots they vacate stay empty until
the block retires. Measured directly during authoring with a separate probe: all
48 warps on one SM entered within **208 cycles** of each other and their exits
spanned **2.55 million** cycles of a 3.70 million-cycle kernel — a 3× spread
between the fastest and slowest warp doing the same arithmetic.

This is a property of the part and not of the instrument. Nsight Compute counts
allocated warp slots and would report the same figure.

---

## TODO 4 — `chooseGrid`, the design TODO

```cpp
const int wave = blocksPerSM * nsm;
if (nch < wave) { *cannotFill = 1; return nch; }
int g = 2 * wave;
if (g > nch) g = nch;
return g;
```

Three things have to be right.

**Never launch more blocks than there is work.** `min(g, nch)` — empty blocks
cost placement, a launch and a retirement for nothing.

**Report honestly when the machine cannot be filled.** With `nch = 80` and a
wave of 240, no grid can keep more than a third of the warp slots occupied. The
correct return is `nch` with `cannotFill = 1`; anything larger launches blocks
that do no work, and anything smaller is worse. The harness tests exactly this
case because the instinct is to return the wave size unconditionally.

**Go past one wave.** This is the part that is not obvious and that Modules 1
and 3 would not have told you. "One wave" is the natural answer — it is the
grid that exactly fills the machine, it is what a grid-stride loop is usually
sized to, and Module 3 called it the machine-derived grid size. It is not the
best answer here, for a reason that only shows up with a ragged cost
distribution: **the only load balancing you get for free is the work
distributor refilling an SM whose block retired early, and it can only do that
if there are blocks left to place.** At exactly one wave there are none, so an
SM that drew six cheap chunks sits idle while another finishes six expensive
ones.

Measured, imbalanced cost, 480 chunks:

| grid | waves | ms | occ_elapsed |
|---|---|---|---|
| 60 | 0.25 | 2.2415 | 15.8 % |
| 120 | 0.50 | 1.7644 | 31.3 % |
| 240 | 1.00 | 1.9077 | 47.7 % |
| 360 | 1.50 | 1.6159 | 65.8 % |
| 480 | 2.00 | 1.5964 | 65.5 % |

Note the shape: one wave is **slower than half a wave** on this workload, which
is a genuinely counter-intuitive row and is pure imbalance — 240 blocks of
wildly different lengths, no refill possible.

The gates are 1.25× over the quarter-wave baseline and 50 % elapsed occupancy.
Two waves measured **1.39–1.53×** and 65–77 % across runs; one wave measures
1.18–1.29× and 47–48 % and does not clear either gate reliably. Going much past
two waves does not help further — the distributor only needs a small queue — and
eventually costs per-block overhead, which is why the rule is "a small multiple
of a wave" and not "as many blocks as possible".

**What this is not.** It is not a software work queue. An earlier draft of this
exercise had the reader build a persistent-block kernel with an `atomicAdd`
ticket, and it measured no better than simply launching two waves — because
**the GigaThread engine already is a dynamic scheduler at block granularity**.
Launching more, smaller blocks is how you ask it to do its job. The persistent
pattern earns its keep when a block needs to carry state across work items, not
for load balancing.

---

## TODO 5 — the predictions

**P1 = 2: the imbalance shows up in `occ_elapsed`, not in `occ_active`.**

At a fixed grid of exactly one wave, with the same total work and the same mean
cost, going from a uniform to a 1..8× cost distribution moved

```
occ_active   61.9 % -> 65.2 %     (+3.3 points, it went UP)
occ_elapsed  62.0 % -> 47.7 %     (-14.3 points)
```

`occ_active` *rose* because long blocks overlap more of each other's lifetimes
than uniform ones do, so while an SM is busy it is fuller. The cost of the
imbalance is entirely in time the SMs were not busy at all, and that time is
only in the `elapsed` denominator. **If your profiler reports the active
variant, load imbalance is not representable in it.**

The same is true of a tail. Going from 240 to 241 blocks moves `occ_active` by
0.4 points (62.3 vs 61.9) and `occ_elapsed` from 62.0 % to 58.1 %, while the
wall time rises 7 %.

**P2 = 3: the fix does not change theoretical occupancy at all.**

The harness re-queries `cudaOccupancyMaxActiveBlocksPerMultiprocessor` after the
fix and gets the same 6 blocks per SM, because theoretical occupancy is a
function of the *kernel's resource footprint* — registers, shared memory, block
shape — and `chooseGrid` changed none of them. It changed the launch.

This is the whole point of separating the two words. Theoretical occupancy is
what Exercise 1 computes and is often already 100 %; achieved occupancy is what
you are actually getting, and the commonest reason it is far below is a grid
that cannot fill the machine. **No amount of `__launch_bounds__` tuning —
Exercise 2's entire subject — touches this failure mode.** A reader who reaches
for registers here is optimising a resource that is not binding anything.

---

## Expected output

```
shard(): 25 registers, 0 B shared, 256 threads -> 6 blocks/SM,
         THEORETICAL occupancy 100.0%, one wave = 240 blocks

  FFMA ceiling probe: 17143 GFLOP/s

-- instrument calibration: one block per SM ---------------------------
  SMs reporting 40/40, expected occ_active 16.7%, measured 16.7%  ok
  occ_elapsed 15.9% (should track occ_active on a single wave)  ok
  recovered SM clock 2.105 GHz

-- the sweep: 480 chunks of work, six grids, two cost profiles ---------
    grid   waves |  ms(unif)  act(unif)  elap(unif) |   ms(imb)   act(imb)   elap(imb) |     model
      60    0.25 |    1.6527      23.9%       18.0% |    2.2415      25.2%       15.8% |     25.0%
     120    0.50 |    1.2483      36.6%       36.6% |    1.7644      39.5%       31.3% |     50.0%
     240    1.00 |    1.2472      61.9%       62.0% |    1.9077      65.2%       47.7% |    100.0%
     241    1.00 |    1.3384      62.3%       58.1% |    1.7336      65.2%       52.4% |    100.0%
     360    1.50 |    1.2452      77.4%       77.6% |    1.6159      76.3%       65.8% |    100.0%
     480    2.00 |    1.2421      75.0%       75.5% |    1.5964      74.9%       65.5% |    100.0%

  wave model (TODO 3) bounds occ_active correctly on 6/6 grids

-- TODO 4: your grid choice, measured ---------------------------------
  nch = 80 (less than one wave): you chose grid 80, cannotFill 1  ok
  nch = 480: grid 480 -> 1.5933 ms (baseline 2.2497, 1.41x), occ_elapsed 65.3%  ok
  nch = 1920: grid 480 -> 6.0590 ms (baseline 8.4316, 1.39x), occ_elapsed 74.3%  ok

-- TODO 5 ------------------------------------------------------------
  at a fixed grid of one wave, going uniform -> imbalanced moved
  occ_active by +3.3 points and occ_elapsed by -14.3 points
  P1 denominator : truth 2, you said 2  correct
  P2 theoretical : blocks/SM before 6, after 6 -> truth 3, you said 3  correct

  SCORE: 10/10

OVERALL: PASS
```

---

## Measurement hygiene — a finding worth recording

An early verification run of this file scored **8/10 on the correct solution**,
failing both `chooseGrid` gates at 1.23× and 1.21× against a 1.25× threshold.
Every row of the sweep was 2.5× slower than usual (grid 480 uniform: 3.13 ms
against 1.24 ms) while the recovered SM clock was unchanged at 2.10 GHz — so the
kernel really was executing 2.5× more *cycles*, not running at a lower clock.
The FFMA ceiling probe that guards the run reported a perfectly healthy
**17143 GFLOP/s** in the same run.

The lesson: **a small compute probe is not a valid operating-point guard for a
full-machine kernel.** The probe runs 15360 threads on 480 small blocks and
draws little power; it reports a healthy ceiling on a machine where something
else has the GPU or where the part is about to clamp under a real load. The
file now guards on the **balanced two-wave reference launch itself** — the one
it already runs for clock recovery — which measures 1.24–1.26 ms healthy and
3.13 ms contended, with a cut at 1.80 ms, and idles 10 s and re-warms up to five
times before warning and proceeding (spec §12.5b).

Spec §12.5c applies too, and was the proximate cause: the failing run started
30 s after another GPU-heavy Module 19 program. Verify these one at a time.

---

## The result that matters

**Theoretical occupancy and achieved occupancy are different numbers, and
achieved occupancy has two different denominators that disagree about everything
this exercise is about.** A kernel at 100 % theoretical occupancy was running at
24 % achieved because its grid was a quarter of a wave; fixing that cost one
line and 1.4×, and changed the theoretical figure not at all. And of the two
achieved figures, the one almost every tool reports — the "active" variant — is
structurally blind to both tail effects and load imbalance, because an SM that
is doing nothing drops out of its own average. Before you tune a kernel's
registers, find out whether you are even running it on the whole machine.

**Variation to try.** Set the cost profile to uniform and re-run `chooseGrid`'s
gates. Two waves still wins, but only by 1.33× instead of 1.41×, and the
one-wave launch becomes competitive — because with uniform blocks there is
nothing for the distributor to rebalance and the only thing more blocks buys is
a smaller tail. Then make the cost distribution 1..64× instead of 1..8× and
watch the advantage of extra waves grow. The right number of waves is a function
of how ragged your work is, which is a fact about your data and not about the
GPU.
