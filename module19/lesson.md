# Module 19 — Occupancy

> Prerequisites: Module 1 (SM anatomy, the block placement gate, waves and tail
> effects, Little's Law), Module 4 (registers, and that local memory is DRAM),
> Module 6 (shared memory capacity, the 1024 B driver reserve, the 128 B
> granularity), Module 8 (warps, partial warps), Module 18 (`__launch_bounds__`,
> the register granule, and the measurement this module exists to explain).
>
> What this module gives you: the exact arithmetic that decides how many warps
> of your kernel are resident, the exchange rate at which registers buy warps,
> the difference between the occupancy you computed and the occupancy you got,
> and a decision rule for when raising occupancy is worth anything at all.

**Part VI opens here.** Modules 1–18 built kernels. Part VI is about making
them fast on purpose rather than by accident, and it starts with the single
most misused number in GPU programming.

---

## Concept

### 1. What occupancy is, and what it is for

**Occupancy** is the ratio of warps resident on an SM to the maximum number of
warps the SM can hold:

```
occupancy = resident warps per SM / 48          (sm_89)
```

That is the whole definition. It is not a measure of speed, efficiency,
utilisation or parallelism. It is a count of warp *slots* that are occupied.

Occupancy exists because of one mechanism and one mechanism only: **a warp that
is stalled cannot issue, and the scheduler can only switch to a warp that is
resident.** Module 1 measured the resulting curve directly — a dependent
pointer chase scales linearly in throughput up to 3 resident warps, has a knee
at 6, and is flat at 4.87× by 24 warps. Occupancy buys **latency tolerance**.
It buys nothing else. Every argument in this module is a consequence of that
sentence.

Two immediate corollaries, both of which most readers get wrong:

- If a kernel has no latency left to hide, more occupancy buys nothing. The
  flat part of Module 1's curve is the *common* case in a tuned kernel, not the
  exotic one.
- Occupancy is not free. Warps are resident because the SM reserved registers
  and shared memory for them, and those are the same resources your kernel
  wanted for its own data. **Occupancy is something you buy, and the currency
  is the per-thread state your algorithm is allowed to keep.**

Module 18 ended with the sharpest possible statement of the second corollary:
the same GEMM source, compiled five times at five occupancies, measured
**390 GFLOP/s at 100 % occupancy and 7267 GFLOP/s at 33 %** — a factor of
**18.7, in the wrong direction**. And occupancy was not even monotone: 25 % →
33 % was a win, 33 % → 67 % was a 9× loss. This module explains that result,
generalises it, and gives you the arithmetic to predict it on your own kernel.

### 2. The placement gate: four limiters, computed exactly

Module 1 described block placement as a gate with four conditions. Here it is,
with the numbers.

A block is placed on an SM only if all four of the following hold. Dividing the
per-SM budget by the per-block cost gives four candidate block counts, and

```
blocks per SM = min(by registers, by shared memory, by warp slots, by block slots)
```

**(a) Warp slots — 48 per SM.**

```
warpsPerBlock  = ceil(threadsPerBlock / 32)
blocksByWarps  = 48 / warpsPerBlock                    (integer division)
```

Note the `ceil`. The textbook form of this limit is "1536 threads per SM", and
for a block size that is a multiple of 32 the two agree. **For any other block
size they do not.** A 100-thread block is four warps (Module 8): 28 of the 128
lane slots were never created, and they still occupy their warp slot for the
block's lifetime. `1536 / 100 = 15` is wrong; `48 / 4 = 12` is right, and the
occupancy API agrees.

**(b) Block slots — 24 per SM.** A hard architectural cap. It binds only for
very small blocks: a 32-thread block with trivial resources gets 24 blocks, not
48, so it can reach at most `24 × 1 / 48 = 50 %` occupancy. This is the real
reason not to launch 32-thread blocks, and it has nothing to do with
coalescing.

**(c) Shared memory — 102400 B per SM.** Module 6 measured both quantisations:

```
smemPerBlock   = roundUp(requestedBytes + 1024, 128)   // driver reserve, granularity
blocksBySmem   = 102400 / smemPerBlock
```

The 1024 B reserve and the 128 B granularity are why a 16384 B request gives 5
blocks and not 6. The request is the **sum of static `__shared__` and the third
launch parameter**; `cudaFuncGetAttributes().sharedSizeBytes` reports only the
static half, which is a trap worth remembering.

**(d) Registers — and this is the one Module 6's formula omitted.**

```
regsPerWarp    = roundUp(regsPerThread, 8) * 32        // granule: 8 per thread, per warp
warpsPerSlice  = 16384 / regsPerWarp                   // 4 slices of 16384 (Module 1)
blocksByRegs   = (4 * warpsPerSlice) / warpsPerBlock
```

Two quantisations again, and the second one is the one nobody has. Module 18
established the first: **registers are allocated in granules of 8 per thread,
for a whole warp at a time**, so a thread asking for 41 registers is charged for
48 and a warp costs `48 × 32 = 1536`.

The second is Module 1's: the SM's register file is not one pool. It is **four
16384-register slices, one per processing block**, and a warp is assigned to a
processing block and draws its registers entirely from that slice. The
leftovers in each slice cannot be pooled. At 47 registers per thread,
`roundUp(47,8)·32 = 1536` registers per warp:

```
aggregate model : 65536 / 1536            = 42 warps -> 21 blocks of 2 warps
slice model     : 4 * (16384 / 1536) = 40 warps -> 20 blocks
```

The hardware places **20**. Those 2·4 = 8 stranded registers per slice are
unusable. `example01.cu` checks both models against
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on 24 kernels; while authoring
this module the two were checked on **137** kernels spanning block sizes 32 to
1024 and register counts 18 to 177. The slice model was exact on all 137. The
aggregate model was wrong on 10.

**This pays the debt the cross-module index records against Module 6** ("M6's
occupancy formula is incomplete — it omits registers … M19 owes the four-limiter
version"), and it corrects the *form* of Module 18's register term at the same
time. Module 18's statement — granule 8, allocated per warp — is correct and
necessary; it is not sufficient.

Finally, occupancy itself:

```
residentWarps = min(blocksPerSM * warpsPerBlock, 48)
occupancy     = residentWarps / 48
```

Count **warps**, not threads and not blocks. For a 1024-thread block the
hardware places exactly one block — 32 warps of 48 — so the ceiling is 66.7 %,
the figure Module 3 quoted and deferred here.

One more gate exists and is not a limiter: a per-block shared-memory request
above 49152 B is **rejected at launch** (`cudaErrorInvalidValue`) unless you opt
in with `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize,
n)` up to 101376 B (Module 6). That gate returns zero blocks, not fewer blocks.
Similarly, a configuration whose registers do not permit even one block fails at
launch with `cudaErrorLaunchOutOfResources` — a 512-thread block at 136
registers per thread is such a configuration.

### 3. The mechanical procedure

Do this by hand, on your own kernel, once, and you will never again be confused
about why a launch has the occupancy it has.

1. Compile with `-Xptxas -v`. Read **`Used N registers`**, **`N bytes smem`**
   and **`N bytes spill stores`**. (Or call `cudaFuncGetAttributes` and read
   `numRegs`, `sharedSizeBytes`, `localSizeBytes` at runtime, which is what the
   example does so that the numbers cannot go stale.)
2. Add any dynamic shared memory you pass as the third launch parameter.
3. Compute the four candidates above.
4. Take the minimum. **Write down which one it was.** That name is the entire
   actionable content of the calculation: it tells you which resource to spend
   less of, and there is no point optimising any other.
5. Multiply by warps per block, cap at 48, divide by 48.
6. Check against `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. If you
   disagree with it, you are wrong; find out why before you continue.

Step 4 is where readers are surprised. Module 17 measured two tiled-GEMM
kernels with **identical thread counts and identical shared-memory footprints**
differing 3 vs 2 blocks per SM purely because one compiled to 44 registers and
the other to 40. Module 18 searched nine deliberately constructed fp32 GEMM
configurations for one where shared memory binds before registers and did not
find one. For compute kernels on Ada, the answer to "which limiter binds" is
"registers" far more often than anyone expects.

### 4. The register/occupancy exchange rate

This is the module's most useful practical skill, and it is three lines of
algebra.

Ignoring block quantisation for a moment, the register limiter gives

```
residentWarps(R) ~ 4 * floor(16384 / (32 * roundUp(R, 8)))
                 ~ 2048 / R
```

so

```
d(warps) / d(R) ~ -2048 / R^2
```

**The exchange rate is quadratic in the register count, not linear.** Measured
on this GPU at 128 threads per block (the table is printed in full by
`example01.cu`):

| registers/thread | resident warps | warps bought by the next granule of 8 |
|---|---|---|
| 40 | 48 | 0 (already at the ceiling) |
| 48 | 40 | 8 |
| 56 | 36 | 4 |
| 64 | 32 | 4 |
| 80 | 24 | 4 |
| 96 | 20 | 0 |
| 104 | 16 | 4 |
| 120 | 16 | 0 |
| 136 | 12 | 4 |
| 168 | 12 | 0 |

Read it as a trade. Below about 48 registers you are at the warp-slot ceiling
and surrendering registers buys **nothing at all** — any register-reduction
effort there is pure loss. Between 48 and 96 a granule is worth 4–8 warps, which
is a real amount of latency tolerance. Past about 120 a granule is worth one
warp or zero, and you are paying real register pressure for a rounding error.

Three practical consequences.

**(a) The useful question is not "how many registers does my kernel use" but
"how many registers can I afford before I lose a block".** That is the inverse
function, and it is what Exercise 1's TODO 5 asks you to write. At 256 threads
per block the answers are: 128 registers for 2 blocks, 80 for 3, 64 for 4, 40
for 6. Those are the only four numbers that matter at that block size, and
everything between them is free.

**(b) Block size changes the occupancy at a fixed register count**, because a
block is placed whole. At 96 registers per thread:

| threads/block | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| resident warps | 20 | 20 | 20 | 16 | 16 |

Same kernel, same registers, 25 % more occupancy from a smaller block. A large
block has to place all of its warps at once, so it rounds down harder. This is
the quantisation term the `~ 2048/R` approximation drops.

**(c) `__launch_bounds__(T, B)` is this arithmetic run backwards, by the
compiler, whether you want it or not.** Measured, and matching in closed form:

```
register cap = roundDown(65536 / (T * B), 8), capped at 255
```

| `__launch_bounds__` | predicted cap | registers ptxas used | spill bytes |
|---|---|---|---|
| `(128, 1)` | 255 | 145 | 0 |
| `(128, 3)` | 168 | 145 | 0 |
| `(128, 4)` | 128 | 126 | 0 |
| `(128, 5)` | 96 | 96 | 56 |
| `(128, 6)` | 80 | 80 | 120 |
| `(128, 8)` | 64 | 64 | 184 |
| `(128, 12)` | 40 | 40 | 280 |

The second argument of `__launch_bounds__` is **not a hint and not a scheduling
request**. It is a register budget expressed in a confusing unit. When the
budget is below what the kernel wants, `ptxas` reschedules; when rescheduling
runs out, it spills to local memory, which Module 4 established is DRAM.

### 4b. Is a resident warp a resident warp? — and how to design an occupancy sweep

Everything in §4 treats resident warps as fungible: 24 warps is 24 warps,
whether they arrive as six 4-warp blocks or as one 24-warp block. That
assumption deserves a measurement rather than an assertion, for two reasons.
First, a block's warps are distributed round-robin over the SM's **four**
processing blocks (Module 1), so a block whose warp count is not a multiple of 4
loads the four schedulers unequally — a 160-thread block is 5 warps, which is
2/1/1/1. Second, Module 20 reports a reproducible sawtooth in throughput against
blocks per SM on its own harness, and hands this module the warning.

**Measured here, and it does not reproduce.** Same source, ~20 registers, no
spills, no memory traffic, no barriers; blocks per SM forced with *dynamic
shared memory* so that registers, the loop body and the instruction schedule are
byte-identical in every row; warps per SM held fixed within each group; total
FLOPs held fixed within each group; 13 configurations timed back to back in 13
rotated sweeps with ~10 ms segments. `example02.cu` part D, so you can rerun it:

| warps/SM | shape | GFLOP/s | vs best in group |
|---|---|---|---|
| 48 | 12 blocks × 4 warps (128 thr) | 15689 | 0.992 |
| 48 | 6 × 8 (256 thr) | 15809 | 1.000 |
| 48 | 4 × 12 (384 thr) | 15613 | 0.988 |
| 48 | 3 × 16 (512 thr) | 15753 | 0.996 |
| 48 | 2 × 24 (768 thr) | 15639 | 0.989 |
| 24 | 6 × 4 (128 thr) | 15493 | 0.998 |
| 24 | 3 × 8 (256 thr) | 15529 | 1.000 |
| 24 | 2 × 12 (384 thr) | 15206 | 0.979 |
| 24 | 1 × 24 (768 thr) | 15491 | 0.998 |
| 20 | 5 × 4 (128 thr) | 15124 | 0.998 |
| 20 | **4 × 5 (160 thr — 2/1/1/1 per scheduler)** | 15150 | **1.000** |
| 20 | 2 × 10 (320 thr) | 15135 | 0.999 |
| 20 | 1 × 20 (640 thr) | 15084 | 0.996 |

Across four runs every shape lands within **1.4–3.4 %** of the best shape in its
group, with no block shape systematically ahead of another — that band is the
harness's own repeatability, not a signal. The absolute GFLOP/s here are below
Module 16's 17.8–18.3 TFLOP/s ceiling because part D runs after two minutes of
sustained benchmarking; only the within-group ratios are claimed. On a saturating pure-FFMA kernel with no memory
traffic and no barriers, resident warps *are* fungible across block shapes —
including the 5-warp block that loads the schedulers unevenly, which measured
1.000. Module 20's sawtooth does not appear on this kernel, and its proposed fit
`W/(4·ceil(W/4))` evaluates to 1.0 for both of the configurations it contrasts,
so it does not predict them either. Treat the effect as real in its own setting
and unexplained; the mechanism is Module 20's to settle.

**Getting that null result honestly took one more methodological step, and it is
worth knowing about.** Part D reports a **median** over the 13 sweeps, not the
minimum that spec §12.3 prescribes. The reason is measured: 13 configurations ×
13 sweeps × 10 ms is 1.7 s of back-to-back full-machine FFMA, which warms the
part from the first sweep to the last. Each configuration appears exactly once
in every position of the rotation, so under a monotone drift its *minimum* is
always its earliest-position sample — which re-introduces precisely the ordering
bias the rotation exists to remove. With min-of-N the two lowest-indexed shapes
measured **10 % faster** than the other three at the same 48 warps/SM; with the
median they measure equal. **Rotation removes positional bias from the mean, not
from the minimum.** When a sweep is long enough to heat the part, say which
statistic you took and why.

**The methodological point stands regardless, and it is the one to take away.**
An occupancy sweep that changes the block size and the register budget and the
blocks per SM at the same time cannot attribute anything to anything. The rule:

> **Change one axis at a time.** Either hold the block size fixed and vary the
> register budget (what Exercise 2 does: 128 threads — 4 warps — in all nine
> rows, so the scheduler loading is identical and only the warp count moves), or
> hold the resource footprint fixed and vary the block size (what the table
> above does). Prefer block sizes that are a multiple of 128 threads, so that
> the warps divide evenly over the four schedulers and that variable is off the
> table. If you must sweep blocks per SM directly, Module 20 recommends
> restricting it to {1, 2, 3, 4, 8, 12}.

Two related hazards worth knowing, both from Module 20 and both avoided in this
module's harnesses rather than argued about:

- **`clock64()` needs a sanity check against a hardware bound** (spec §12 rule
  13). It is trustworthy for a lone warp and can be wildly wrong at high
  occupancy — Module 20 derived 5.73 instructions per cycle per scheduler, 5.7×
  a hardware maximum, from it. Use `nvidia-smi --query-gpu=clocks.max.sm`
  (**3105 MHz** on this part) as the bound, not the 2.04 GHz figure quoted
  elsewhere; `example02.cu` recovers 2.10–2.12 GHz and checks it against 3105
  before using it.
- **Hold the work inside a loop body constant when sweeping a parameter.** If
  the body's size changes with the sweep variable, `ptxas`'s unroll heuristic
  changes with it and produces a ~25 % artefact that looks exactly like a
  hardware effect. Every sweep in this module compiles the same loop body in
  every row.

### 5. What `__launch_bounds__` costs you, and why it is still worth having

The first argument, `maxThreadsPerBlock`, is almost always worth supplying: it
tells `ptxas` the largest block you will ever launch, which lets it allocate
registers for that case instead of for the 1024-thread worst case, and it makes
a launch with a bigger block a *compile-known* error instead of a runtime
`cudaErrorLaunchOutOfResources`.

The second argument is a loaded weapon. It can only ever *reduce* the register
count, and a reduced register count has exactly two possible outcomes: the
compiler reschedules and you get the extra blocks for free, or the compiler
spills and you have put a DRAM access in your inner loop to buy warps you may
not need.

Exercise 2 measures the whole curve on one source. The shape, on a recursive
cascade kernel whose state lives in registers:

| `__launch_bounds__(128, B)` | registers | spill B | blocks/SM | occupancy | Gelem/s |
|---|---|---|---|---|---|
| 1 (unconstrained) | 177 | 0 | 2 | 16.7 % | 8.56 |
| 2 | 177 | 0 | 2 | 16.7 % | 8.58 |
| 3 | 168 | 0 | 3 | 25.0 % | 10.35 |
| **4** | **122** | **0** | **4** | **33.3 %** | **10.79** |
| 5 | 96 | 72 | 5 | 41.7 % | 7.79 |
| 6 | 80 | 136 | 6 | 50.0 % | 3.30 |
| 8 | 64 | 200 | 8 | 66.7 % | 1.94 |
| 10 | 48 | 264 | 10 | 83.3 % | 1.42 |
| 12 | 40 | 296 | 12 | 100.0 % | 1.45 |

Four things in that table.

1. **The optimum is interior.** Neither end wins. The 100 %-occupancy build is
   **7.4× slower** than the best, and the compiler's own unconstrained choice is
   **1.26× slower** than the best — 7.2–8.8× and 1.09–1.37× over seven runs.
   Raising occupancy helps, right up until it does not.
2. **Occupancy is not monotone**, again: 83.3 % is marginally *slower* than
   100 %. A tuner that hill-climbs on occupancy has no idea what it is doing.
3. **The cap binds long before the compiler spills.** The cap first falls below
   the unconstrained 177 registers at `B = 3`; the first spill is at `B = 5`.
   In between, `ptxas` gave up **55 registers per thread for free** by
   rescheduling. "Reducing the register budget" and "causing a spill" are not
   the same event, and the gap between them is where all the easy occupancy
   lives.
4. **The spill cliff is not the first spilled byte.** Module 18 measured an
   80 B spill that bought a fourth block running **1.12× faster** than the
   unconstrained build. Here a 72 B spill at `B = 5` is already a 1.4× loss.
   Both are correct, and the difference is *what* spilled: the cliff is where
   the values the inner loop touches every iteration stop fitting. In Module
   18's GEMM those were the accumulators; here they are the filter state. A
   spill of addressing temporaries is free; a spill of the recurrence is fatal.
   **`localSizeBytes` tells you there is a spill; only the SASS tells you what
   spilled.**

### 6. Theoretical versus achieved occupancy

Everything above is **theoretical occupancy**: a static calculation from
resource counts, performed by the hardware's placement gate, knowable before the
kernel runs. It is an upper bound.

**Achieved occupancy** is the time-average of the warp slots that were actually
occupied while the kernel ran. It is what Nsight Compute reports as "Achieved
Occupancy", and it is almost always lower. This is the distinction most readers
never learn, and it has a sharp edge: *there are two defensible denominators and
they answer different questions.*

```
occ_active  = sum over warps of (lifetime)  /  sum over SMs of (span_SM * 48)
occ_elapsed = sum over warps of (lifetime)  /  (nSM * 48 * kernel_cycles)
```

`occ_active` normalises each SM by the time **that SM** was busy. `occ_elapsed`
normalises by the whole kernel on the whole machine. Nsight Compute exposes both
(`sm__warps_active.avg.pct_of_peak_sustained_active` and
`..._sustained_elapsed`); almost every tutorial quotes the first.

`ncu` cannot run on this machine (`ERR_NVGPUCTRPERM`, spec §12), so
`example02.cu` builds the counter from `clock64()` and `%smid`: one lane per
warp records entry and exit, and three atomics per warp accumulate
`min(entry)`, `max(exit)` and `sum(lifetime)` per SM. Timestamps are never
compared **across** SMs, because the per-SM cycle counters are not synchronised
— measured 298 million cycles apart on this part.

Three mechanisms make achieved fall below theoretical.

**(a) The grid is too small — by far the most common, and nothing to do with
registers.** A grid of `G` blocks cannot occupy more slots than it has blocks:

```
achieved <= theoretical * min(1, G / (blocksPerSM * nSM))
```

Measured, uniform cost, against a theoretical 100 %:

| grid | waves | occ_active | occ_elapsed | bound |
|---|---|---|---|---|
| 60 | 0.25 | 23.9 % | 18.0 % | 25 % |
| 120 | 0.50 | 36.6 % | 36.5 % | 50 % |
| 240 | 1.00 | 61.9 % | 61.9 % | 100 % |
| 241 | 1.00 | 62.3 % | 57.9 % | 100 % |
| 360 | 1.50 | 77.4 % | 77.5 % | 100 % |
| 480 | 2.00 | 75.0 % | 75.0 % | 100 % |

The bound is tight where it matters and the quarter-wave launch is at a quarter
of its theoretical occupancy no matter what the resource arithmetic says.

**(b) The tail (Module 1's wave analysis), and it is invisible in the metric
you are probably reading.** Going from 240 blocks to 241 moves `occ_active` by
0.4 points — each SM is still busy while it is busy — and moves `occ_elapsed`
from 61.9 % to 57.9 %, with the wall time up 7.5 %, because the denominator now
includes the time 39 SMs spent waiting for one. **If your profiler reports the "active" variant, a tail
effect is literally not representable in it.**

**(c) Load imbalance**, which behaves the same way. With the same total work,
the same mean cost and the same grid, making the per-block cost vary 1..8×
moves `occ_active` by **+3.4** points (it goes *up*: longer blocks overlap more)
and `occ_elapsed` by **−14.4**, and costs 53 % in wall time.

And one mechanism that is **not** on the list, which you must get right:

> **A barrier does not reduce achieved occupancy.** A warp waiting at
> `__syncthreads()`, or stalled on a DRAM load, or stalled on a long-latency
> FFMA chain, is still **resident**. It still owns its warp slot, its registers
> and its share of the block's shared memory, and it is still counted. Neither
> barriers nor memory stalls lower achieved occupancy.

That is exactly why a kernel can sit at 100 % achieved occupancy and issue
almost nothing. Occupancy counts warps that **exist**; the quantity you actually
wanted is warps that are **eligible to issue**. Module 20 owns that distinction
and the stall-reason taxonomy that separates the two. Module 18's 390 GFLOP/s
build was at 100 % occupancy and had 24 of its 64 accumulators in DRAM.

**A documented surprise.** Even a perfectly uniform launch of exactly one wave,
with every block resident from the first cycle, measures **58–70 % of its
theoretical occupancy** on this part, with zero tail, zero imbalance and all 40
SMs reporting spans within 1 % of each other. The cause is inside one SM: the
warp scheduler is greedy, so warps executing **identical** work finish at very
different times, and the slots they vacate stay empty until the block retires.
Measured directly while authoring this module: all 48 warps on an SM entered
within **208 cycles** of each other and their exits spanned **2.55 million**
cycles out of a 3.70 million-cycle kernel — a 3× spread between the fastest and
slowest warp doing the same arithmetic. Nsight Compute would report the same
figure, because it also counts allocated slots. Treat 100 % as unreachable and
compare configurations against each other.

### 7. The API

Four calls and one attribute. All of them are in `example01.cu`.

| what | use it for |
|---|---|
| `cudaFuncGetAttributes(&attr, kernel)` | `numRegs`, `sharedSizeBytes` (static only), **`localSizeBytes` = the spill figure**, `maxThreadsPerBlock` |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, blockSize, dynSmem)` | the oracle. Blocks per SM, exactly. Multiply by `blockSize/32` and divide by 48 yourself |
| `cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, kernel, dynSmemFn, blockSizeLimit)` | a block size that maximises **occupancy** |
| `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` | arg 1: tell `ptxas` the real block size. arg 2: a register budget in disguise |

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` is the one you should actually
call, in your own code, at startup, and print. It is cheap, it is exact, and it
removes an entire class of mistaken belief.

**`cudaOccupancyMaxPotentialBlockSize` chooses for occupancy, not for
performance, and it says so**: the word "performance" does not appear in its
contract. Module 18 measured those two objectives diverging by 18.7× on one
GEMM source. On a latency-bound elementwise kernel its answer is usually fine;
on anything with per-thread state it is a starting point and nothing more. The
same caveat applies with more force to the CUDA Occupancy Calculator
spreadsheet and to every "increase occupancy" recommendation a tool has ever
emitted.

### 8. The decision rule

Put the whole module into one procedure.

> **Raise occupancy only when the profiler shows stalls that more warps would
> cover, and only while the registers it costs are not ones your inner loop
> needs.**

Operationally, in order:

1. **Compute theoretical occupancy and name the binding limiter.** If you do not
   know which of the four it is, you cannot change it.
2. **Compare achieved with theoretical.** If achieved is far below theoretical,
   stop — the fix is the grid, not the registers. A quarter-wave launch at 100 %
   theoretical occupancy is a scheduling bug, and no amount of
   `__launch_bounds__` will touch it.
3. **Ask whether the kernel is latency-bound at all.** Module 11 measured the
   substitute directly: at one block per SM, giving each thread independent
   loads was worth 2×; at eight blocks per SM it was worth 1.00×. A kernel with
   enough instruction-level and memory-level parallelism per thread is already
   tolerant and will not pay you for more warps. Module 20 owns the measurement
   that decides this; Module 23 owns the profiler counters.
4. **If and only if more warps would help, consult the exchange rate.** Below
   ~48 registers per thread it is zero. Above ~120 it is one warp per granule.
   In between it is 4–8 warps per granule and worth having.
5. **Check what spilling would cost before buying.** `localSizeBytes > 0` is not
   automatically fatal — Module 18's 80 B spill was a 1.12× *win* — but a spill
   that reaches the values your innermost loop reads every iteration is a DRAM
   access per iteration, and that is the 18.7×.
6. **Measure.** Occupancy is a proxy. The thing you care about is the thing you
   care about.

Three standing facts to carry out of Part VI's first module: **occupancy is a
means, the optimum is usually interior, and the number your tool reports has a
denominator you should know the name of.**

---

## Hardware Mental Model

**Why the placement gate is four conditions and not one.** An SM is not a pool
of undifferentiated resources. It is 4 processing blocks, each with one warp
scheduler, 12 warp slots and a 16384-register slice, plus one shared 128 KB
L1/shared SRAM and one block-slot table (Module 1). A block arriving at an SM
must find simultaneously: a free entry in the block table, enough free warp
slots, a contiguous shared-memory allocation, and registers *in the slices its
warps will be assigned to*. Any one of those failing stalls the block, and the
GigaThread engine offers it to another SM instead. The four-limiter formula is
not an abstraction over the hardware; it is a transcription of it.

**Why registers quantise twice.** The register file is addressed by a field in
the instruction word, so an allocation is a base plus a size, and the hardware
cannot afford a general allocator. Round the per-thread count up to 8 and you
can describe a warp's allocation in one small field (`8 × 32 = 256` registers,
so 32 sizes span the whole 0–255 range a thread may ask for). Bind the warp to one
slice and the allocation never has to span banks. Both quantisations exist to
make the warp context switch free, which is the property Module 1 called the
GPU's defining trick. **You pay for zero-cost context switching in rounding.**

**Why occupancy is held for the warp's whole lifetime.** A CPU hides latency by
re-ordering and speculating within one instruction stream; a GPU hides it by
having another stream ready. "Ready" means the other warp's architectural state
is *already in the register file*, because copying it there would cost exactly
the latency you were trying to hide. So the register file is statically
partitioned among resident warps for their entire lifetime, and the number of
warps you can have is `registerFileSize / perWarpFootprint` by construction.
Occupancy is not a scheduling policy. It is division.

**Why the optimum is interior.** Two effects move in opposite directions as the
register budget falls. Latency tolerance improves roughly like the warp count,
which goes as `2048/R` — a decelerating gain. Spill traffic grows once `R`
drops below what the live set needs, and it grows *fast*, because what spills
last (and therefore first hurts) is whatever is live across the inner loop.
A decelerating gain against an accelerating loss has an interior maximum. The
only question on a given kernel is where, and the only way to find out is to
know both curves — which is what this module's two exercises make you build.

**Why an "active" denominator cannot see a tail.** The counter is a per-SM
accumulator gated by that SM having work. An SM with nothing to do contributes
nothing to either the numerator or its own denominator; it simply drops out of
the average. That is the right choice if you are asking "when my SMs were
working, how full were they" and the wrong one if you are asking "did I use the
machine". **Both questions are legitimate. Know which one you asked.**

**Ada specifics. ARCHITECTURE-SPECIFIC:** 48 warp slots, 24 block slots, 65536
registers in 4 slices of 16384, an 8-register-per-thread granule, 255 registers
per thread maximum, 102400 B shared per SM with a 1024 B per-block reserve and
128 B granularity, a 49152 B default per-block shared limit opening to 101376 B,
and the greedy warp scheduler whose completion skew puts a practical ceiling of
60–78 % on achieved occupancy.
**PORTABLE CUDA CONCEPT:** occupancy as resident warps over maximum warps; the
placement gate as a minimum over per-resource limits; the register file as the
reason occupancy is bought rather than configured; theoretical versus achieved
and the two denominators; the exchange rate `d(warps)/d(R) ∝ -1/R²`; and the
decision rule in §8.

---

## Code Walkthrough

### `example01.cu` — the arithmetic

No kernel is timed in this file. Part A applies the four-limiter procedure to 24
real kernels and checks every one against the occupancy API:

```cpp
const int regsPerWarp   = roundUp(regsPerThread, REG_GRAN) * 32;
const int warpsPerSlice = REGS_PER_SLICE / regsPerWarp;
cand[0] = (REG_SLICES * warpsPerSlice) / warpsPerBlock;        // registers
cand[1] = SMEM_PER_SM / roundUp(smemBytes + SMEM_RESERVE, SMEM_GRAN);
cand[2] = WARPS_PER_SM / warpsPerBlock;                         // NOT 1536/threads
cand[3] = MAX_BLOCKS_PER_SM;
```

The table it prints has a column per candidate, so you can see *which* of the
four was the minimum on every row; that column is the point of the exercise.
Rows are chosen so that every limiter binds somewhere: registers at
`state<64>` (80 registers, 3 blocks), shared memory at 16384/25600/49152 B
(5/3/2 blocks, reproducing Module 6's table exactly), warp slots at 512 threads,
block slots at 32 threads.

Part B isolates the three rows where the aggregate register model disagrees
with the hardware and works one of them through arithmetically. Part C prints
the exchange-rate table of §4 together with the marginal warps bought by each
granule, and the block-size quantisation table. Part D prints
`cudaFuncGetAttributes`, calls `cudaOccupancyMaxPotentialBlockSize` and
measures the `__launch_bounds__` register cap against `ptxas`'s actual choice on
seven bounds.

The kernel family is a **recursive cascade**: `W` filter states live in
registers across the whole time loop, so `W` is a direct handle on the register
count, and an `SFLOATS`-element `__shared__` array is a handle on the shared
footprint. It is the shape of an IIR filter bank, an RNN step or any per-thread
integrator — a long serial dependence with a large private state, which is
exactly the kernel shape where occupancy decisions bite.

### `example02.cu` — the measurement

The instrument is twelve lines:

```cpp
__device__ __forceinline__ void profIn(Prof p, unsigned long long &t0, unsigned &sm) {
    if ((threadIdx.x & 31) == 0) { sm = smId(); t0 = clk(); atomicMin(&p.lo[sm], t0); }
}
__device__ __forceinline__ void profOut(Prof p, unsigned long long t0, unsigned sm) {
    if ((threadIdx.x & 31) == 0) {
        unsigned long long t1 = clk();
        atomicMax(&p.hi[sm], t1);
        atomicAdd(&p.cyc[sm], t1 - t0);
    }
}
```

One lane per warp (32 reporters would multiply everything by 32), three atomics
per warp over 40 addresses — uncontended by Module 10's measure, and negligible
against a 1.2 ms kernel. `clk()` is `clock64()` through inline PTX with a
`"memory"` clobber so the compiler cannot hoist the second read above the work.

The payload is deliberately compute-bound (spec §12 rule 8): a bandwidth-bound
block runs far faster when it is alone on an SM, which smears every occupancy
effect into a ramp. It is a grid-stride loop over chunks, so the grid is a free
parameter and all twelve configurations compute the same answer — the validation
pass checks that.

The sweep is six grid sizes × two cost profiles, timed back to back in eight
rotated sweeps (`SWEEPS >= NCFG`), after a 1500 ms streaming plus 500 ms compute
warm-up, with validation in a separate pass. **Part D** is §4b's block-shape
experiment on a second, register-light payload whose blocks per SM are forced
with dynamic shared memory, so you can rerun the negative result rather than
take it on trust — and it is the one place in this module that reports a median
rather than a min-of-N, for the reason §4b gives. The SM clock needed by
`occ_elapsed` is recovered from the busiest SM's span divided by the measured
wall time, never from `cudaDevAttrClockRate` (which reads 1.545 GHz against a
measured 2.12 and would make every elapsed figure 27 % optimistic).

---

## Check Your Understanding

1. A kernel uses 42 registers per thread and no shared memory, and is launched
   with 192-thread blocks. Compute its blocks per SM and its occupancy by hand.
   Now your colleague "optimises" the kernel and brings it to 40 registers.
   Compute the new occupancy. They then report a 0 % change in runtime and
   conclude that occupancy does not matter on this GPU. Give the two distinct
   reasons their experiment cannot support that conclusion, and say what you
   would have measured instead.

2. Two kernels have identical theoretical occupancy (50 %), identical register
   counts and identical shared-memory footprints. Kernel A measures an achieved
   occupancy of 49 % and kernel B measures 12 %, using the *elapsed*
   denominator. Both are launched with the same grid of 4096 blocks on the same
   data. List every mechanism that could produce that difference, and for each
   one give a measurement — not a guess — that would confirm or eliminate it.
   Then say which of your mechanisms would still be visible if both numbers had
   been computed with the *active* denominator instead.

3. You are told that a kernel's achieved occupancy is 100 % and that it is
   running at 6 % of the FP32 ceiling. Someone proposes that the next step is to
   reduce its register count so that more warps fit. Explain, from the hardware
   model, why that proposal is guaranteed to be either a no-op or a
   pessimisation, and construct the one measurement that would tell you what to
   do instead.

4. `__launch_bounds__(256, 3)` and `__launch_bounds__(128, 6)` impose the same
   per-thread register cap. Do they produce the same occupancy? Do they produce
   the same *achieved* occupancy on a kernel whose blocks have widely varying
   durations? Argue both from the four-limiter formula and from §6, and give the
   block size you would prefer with a reason that is not "it is a multiple of
   32".

Answers: `solutions/module19/check_your_understanding.md`.

---

## Exercises

### Exercise 1 — `exercise01.cu` — occupancy by hand, then checked against the API

Implement the placement arithmetic and make it agree with
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on **every** kernel in the
table, not on most of them.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

| TODO | requirement |
|---|---|
| 1 | `blocksByRegisters`. Two quantisations have to be in this expression. One of them is Module 18's; the other is in Module 1's description of what an SM is physically made of. A model with only the first is right on most kernels. |
| 2 | `blocksBySharedMemory` and `blocksByWarpSlots`. One row of the table is chosen so that `48 / warpsPerBlock` and `1536 / threads` give different answers. |
| 3 | Combine into a minimum and report the **binding limiter**. |
| 4 | `occupancyPercent`. Count warps. Two rows disagree with any expression written in threads. |
| 5 | **Design:** invert the arithmetic — the largest per-thread register count that still achieves a target number of blocks. Scored by round trip, so it must be exactly the largest. |

Validation: 10 points — 18 real kernels against the API (3), 10 hypothetical
`(registers, shared, threads)` triples against FNV-1a hashes (2), the binding
limiter (2), occupancy (1), and the TODO 5 round trips (2). `OVERALL: PASS`
requires all ten.

### Exercise 2 — `exercise02.cu` — the exchange rate, measured

The harness compiles one source nine times under
`__launch_bounds__(128, minBlocks)` and reports registers, spills, occupancy and
throughput for each. You choose the operating point and justify it in a form the
harness can check.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

| TODO | requirement |
|---|---|
| 1 | `registerCapFor(threads, minBlocks)` in closed form, including the rounding direction and the 255 ceiling. Checked against `ptxas`. |
| 2 | The smallest `minBlocks` at which the cap **binds**. Pencil and paper. Note that this is not where it spills, and the harness prints both. |
| 3 | **Prediction:** how far off the best is the 100 %-occupancy build? |
| 4 | **Prediction:** is the compiler's unconstrained choice the fastest, and if not, which side is it on? |
| 5 | **Design:** `CHOSEN_MINB` plus `CHOSEN_RULE` — the selection rule that picked it. The harness evaluates every candidate rule against the measured *resource* table and checks that yours selects your configuration. |

Validation: 10 points. The chosen operating point must land within 20 % of the
fastest row measured in that run, and must be consistent with the TODO 4
prediction.

### Exercise 3 — `exercise03.cu` — theoretical versus achieved

Build the achieved-occupancy counter yourself, discover that it has two
denominators, and fix a launch that is wasting three quarters of the machine at
100 % theoretical occupancy.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

| TODO | requirement |
|---|---|
| 1 | The instrument: one reporter per warp, three atomics, per SM. Calibrated against a launch whose answer is known by construction. |
| 2 | `occActive` and `occElapsed` — the two denominators. One of them is blind to the effect the rest of the exercise is about. |
| 3 | The grid-size bound on achieved occupancy. Checked as a genuine upper bound on every row. |
| 4 | **Design:** `chooseGrid`, evaluated by measurement on two differently sized ragged workloads, plus one where the honest answer is "this problem cannot fill the machine". |
| 5 | Two predictions: which denominator moves under load imbalance, and what the fix does to *theoretical* occupancy. |

Validation: 10 points, including a 1.25× wall-time gate and a 50 % elapsed-
occupancy gate on both workloads.

---

## Prediction

Commit these to writing before you build anything.

1. A kernel compiles to 47 registers per thread and runs in 64-thread blocks
   with no shared memory. Write down its blocks per SM. Then write down what
   `65536 / (47 × 64)` gives you, and which of the two you expect the hardware
   to agree with.

2. Exercise 2 sweeps one source across nine register budgets from 177 registers
   down to 40. Write down which of the nine you expect to be fastest, as a
   fraction of the way along the sweep, and the ratio between the fastest and
   the 100 %-occupancy build. Most readers put the optimum too close to the
   high-occupancy end and under-estimate the ratio by a factor of three.

3. Example 2 launches exactly one wave of blocks, with uniform per-block work,
   at 100 % theoretical occupancy, and measures achieved occupancy. Write down
   the number you expect. Then write down what you would conclude if the two
   disagreed by 40 %, before you read §6's last paragraph.
