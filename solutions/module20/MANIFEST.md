# Module 20 manifest — Latency Hiding

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module20/lesson.md`, `module20/example0{1,2}.cu`,
`module20/exercise0{1,2,3}.cu`.
Solutions: `solutions/module20/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module20/check_your_understanding.md`,
`solutions/module20/MANIFEST.md`.

All eight `.cu` files verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2 V13.2.51,
RTX 3500 Ada), **warning-clean**. Both examples print `OVERALL: PASS`; all
three solutions print `SCORE: 10/10` and `OVERALL: PASS`. All three shipped
exercises compile with TODOs blank and exit gracefully (`Set TODO 2 first.`,
`Set TODO 1 first.`, `Set TODO 1 first.`). No binaries committed.

**Scope boundary, agreed with Modules 19 and 21.** Module 19 owns the resource
arithmetic (the four placement limiters, the 8-register granule,
`__launch_bounds__`, the register/occupancy exchange rate). Module 21 owns the
roofline. Module 20 owns the **mechanism**: why latency tolerance works, what a
warp is waiting on, and ILP as the alternative to occupancy. None of M19's or
M21's material is re-taught here; both are cited by name.

---

## Concepts taught

- **Little's Law as a calculator, used four times**, each time producing a
  number checkable against hardware: the FP32 pipe (4 instructions in flight per
  scheduler), the memory system (124 KB in flight device-wide), Little's Law
  *inverted* to recover DRAM latency from a bandwidth measurement, and the
  diagnostic form (`supplied / required` = the diagnosis).
- **The two measured constants this module exists to produce:**
  **dependent-FFMA latency 4.05 cycles** and **saturated issue interval 1.067
  cycles** (= 1 warp-instruction per scheduler per clock). Their ratio, 3.8, is
  the concurrency requirement. **ARCHITECTURE-SPECIFIC (sm_89).**
- **TLP and ILP/MLP as substitutes**, with the exchange rate derived rather
  than fitted: **one unit of per-thread ILP replaces one resident warp per
  scheduler = four resident warps per SM**, until the product reaches
  `latency × throughput`, after which both are worthless.
- **The ILP × occupancy surface is a function of the product only.** Measured on
  both pipes. On the memory side, cells sharing a product agree to 0.2–0.5%
  while cells in the same row differ by up to 63%.
- **The stall taxonomy**, as a seven-row table with the "does occupancy help?"
  column: long scoreboard, short scoreboard, barrier, execution dependency,
  instruction fetch, throttle, **not-selected**, drain/tail. With `ncu` metric
  names and the explicit statement that the tool cannot be run here.
- **`not selected` means too MUCH occupancy**, not too little: a scheduler
  issues ≤1 instruction/clock, so the only cure is fewer instructions.
  Constructed and measured (Exercise 2 kernel D).
- **Dependency chains and the critical path**: `T = length × latency`,
  independent of machine width; per-instruction cost
  `max(latency/C, 1/throughput)`.
- **Reassociation is the price of breaking an accumulator chain**, and the
  validator must be written for it (Module 16's `γ_K·S` rule, reused).
- **MLP cost curve for a single warp**: `cycles/step = 241–246 + 15.1 × C`.
  The intercept recovers **Module 4's L2 latency (241.3 cycles) to within 2%**
  without being told. No hard queue limit up to C = 32; the crossover from
  latency-bound to throughput-bound is at `C ≈ L/slope ≈ 16`, practical knee ~8.
- **A `float4` load occupies one outstanding-request slot and carries 4× the
  payload** — concurrency in bytes is the unit, not requests. This is why M5/M11
  called vectorization a latency optimization.
- **The four ways ILP evaporates**: variable compiler unroll factor (measured),
  CSE (M18's 64→12), register pressure (M18's 18.7×), instruction cache.
- **Why the FFMA count per outer-loop iteration must be held constant across
  ILP levels** — otherwise the experiment measures `ptxas`'s unroll heuristic.
  Measured: `C = 2` gets 64 FFMAs between branches where `C = 4` gets 192, and
  comes out 25% low.
- **The decision procedure** (lesson §9): floor → occupancy sweep → distance
  from a bound → read the loop for a dependence → instruction count. Six steps,
  no profiler.
- **`clock64()`'s validity window.** Reliable for a lone warp (cross-checks to
  1.66–2.08 GHz against wall time), unreliable at full occupancy (reports
  5.7 instructions/cycle/scheduler, 5.7× a hardware maximum). Confirms and
  sharpens Module 16's warning.

### New measured finding (see "Cross-module observations")

- **Resident warps are not fungible across block shapes.** 20 warps per SM
  delivered as one 640-thread block run at **1.008** instructions/cycle/scheduler;
  the same 20 warps as five 128-thread blocks run at **0.625**. Sawtooth
  envelope `W/(4⌈W/4⌉)` over blocks/SM = 1..12. Cycle-accurate, placement
  verified even by `%smid`, warp-slot distribution verified identical by
  `%warpid`.

## CUDA API / intrinsics / syntax introduced

- Nothing new in the API. This module is deliberately an *analysis* module and
  reuses: `clock64()` (M1/M4), `#pragma unroll` / `#pragma unroll 1` (M4/M7/M11),
  `fmaf`, `float4` (M5), `__fdividef` (M11), `__restrict__`, `cudaEvent_t`
  timing, `setvbuf(stdout, NULL, _IONBF, 0)`, inline PTX `%smid` (M1) and
  `%warpid` (used during authoring only, not in shipped files).
- SASS read and quoted: `FFMA` with the `.reuse` operand flag, `LDG.E.CONSTANT`,
  `LDG.E.128`, `STG.E`, `IMAD.WIDE.U32`, `MUFU.RCP`, **`FCHK`** (the IEEE-divide
  range check, named by M11 and counted here).
- Nsight Compute metric names given as theory only:
  `smsp__warp_issue_stalled_<reason>_per_warp_active.ratio` for the full reason
  list, plus `--section WarpStateStatistics --section SchedulerStats`.
  **`ncu` is unavailable on this machine (`ERR_NVGPUCTRPERM`); no counter output
  is quoted anywhere in this module.**

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | **The arithmetic of latency on the FP32 pipe.** A: one warp, one SM, ILP 1..10, `clock64()` with a wall-clock cross-check — latency 4.051 cycles, issue interval 1.067, `L/T = 3.81`. B: the 5×5 ILP × occupancy surface, 25 configurations in one rotated sweep (`SWEEPS = NCFG = 25`), printed in GFLOP/s and in % of the best cell. C: the exchange rate, as the smallest occupancy reaching 85% / 90% per ILP level, with the `ILP × warps` product column that reads 4, 4, 4. Validation against a host `fmaf` replay, exact. |
| `example02.cu` | **Little's Law at the memory system.** A: the demand/supply calculation against a ceiling measured in the same program. B: the MLP × occupancy surface, 30 configurations, re-indexed by the product to show the collapse. C: the single-warp MLP cost curve with a least-squares fit whose intercept is M4's L2 latency. Validation: per-thread exact replay of the strided sum for all six MLP levels. |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | fill-in + perf reasoning + prediction (§6 types 1, 6, 2) | 5 | Build `C` independent FFMA chains, sweep ILP × occupancy, and check a *derived* exchange-rate model against the measurement | **TODO 1's requirement (a) is the trap and it is not about ILP at all**: if the FFMA count per outer-loop iteration varies with `C`, `ptxas` unrolls the `C = 2` instantiation 3× less than the others (64 vs 192 FFMAs between branches) and `C = 2` reads 25% low for a reason that has nothing to do with latency. Measured. Also: the identical-seed CSE worry is a **null result** on nvcc 13.2 (fp arithmetic blocks the proof), so the exercise does not claim it. **TODO 2** hides §8's finding — not every `(threads, blocks)` pair with the right product is equivalent; a 160-thread block loads the four schedulers 2/1/1/1 and runs at 0.56 of the ceiling, and blocks/SM ∈ {5,6,7,9,10,11} sits on a reproducible sawtooth. **TODO 5** is scored against the table but must be *derived*; a lookup fitted to the table scores identically and teaches nothing, which is stated in the solution. |
| `exercise02.cu` | debugging + optimization + design (§6 types 3, 4, 5) | 5 | Four kernels, four causes, four different fixes; classify then fix three of them | **The four kernels pair up against the evidence.** A and B both respond "yes" to the occupancy sweep, C and D both respond "no"; separating each pair needs the % -of-bound column or a read of the loop body. The reflexive answer for D is "latency-bound" and it is wrong — D has 64 independent divides at 12 warps/scheduler and is **issue-limited**; 4→12 blocks/SM is worth 1.11× and deleting 256 of 440 SASS instructions is worth 2.03×. **The documented null result:** an earlier form of kernel C computed its coefficient inside the loop, giving three independent instructions per dependent FFMA — already ILP 4 — and four accumulators then bought **1.07×** instead of 3.25×. The wrong fix did nothing because the right fix was already there by accident. Kernel A has *two* valid fixes and the budget gives you only one. Kernel C's block-shape column goes the *wrong* way (0.81×) because 256 threads/block is 20 blocks on 40 SMs. Kernel A's tail: `16777216 / 40960 = 409.6`, so 24,576 elements need a tail loop. |
| `exercise03.cu` | performance reasoning + fill-in (§6 types 6, 1) | 5 | Compute required vs supplied concurrency in bytes, reconcile against the measurement, then close the gap at a fixed thread count | **Counting requests instead of bytes** makes a `float` load and a `float4` load look equivalent, and the whole exercise turns on their 4× difference. **Neither multiplier alone clears the gate**: MLP 4 with scalar loads and MLP 1 with `float4` both supply 80 KB = 0.66 Q* and land near 66% of the ceiling (Example 2's `(1, C=1)` cell measures 61%); you need both, or ~7 scalar loads in flight. Raising the thread count is forbidden and the arithmetic tells you it would take 6× the budget. The payoff is not the score: the reader's own TODO 2 formula, fed the measured 68.3 GB/s, returns **569 cycles** against Module 4's independently measured **575**, and then predicts the baseline bandwidth back to 0.1%. |

Scoring: Ex1 10 points (2 chains-survive, 2 latency prediction, 1+1 the two ILP
gain predictions, 2 the derived model on ≥4 of 5 rows, 2 numerics); Ex2 10
points (4 classification, 3 the four gates, 3 the four speedup predictions
within 1.5× on ≥3); Ex3 10 points (2+2+2 for the three formulas checked exactly
against the harness's own evaluation, 2 the 95%-of-ceiling gate, 2 the
bandwidth prediction within 12%). `OVERALL: PASS` requires full marks plus
clean validation.

## Measured results recorded (RTX 3500 Ada, CUDA 13.2)

### Single warp, dependent FFMA chain (`example01.cu` A)

| ILP | cycles/FFMA | vs ILP 1 |
|---|---|---|
| 1 | **4.051** | 1.00× |
| 2 | 2.055 | 1.97× |
| 3 | 1.387–1.392 | 2.92× |
| 4 | **1.067** | 3.80× |
| 5, 6, 8, 10 | 1.064–1.071 | flat |

**FFMA latency 4 cycles; issue interval 1 cycle; `L/T = 3.8`.** `clock64()`
cross-checked against wall time at 1.66–2.08 GHz.

### ILP × occupancy, FP32 (`example01.cu` B), % of best cell

| warps/sched \ ILP | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 | **24–26%** | 46–51% | 88–96% | 89–97% | 88–96% |
| 2 | 47–51% | 94–99% | 99–100% | 100% | 99–100% |
| 4 | 99% | 99% | 99–100% | 100% | 99–100% |
| 8 | 95–99% | 94–98% | 96–100% | 94–100% | 92–100% |
| 12 | 91–99% | 91–98% | 92–100% | 92–100% | 92–100% |

Best cell 20,965–21,176 GFLOP/s (implies ~2.05 GHz). **ILP 1→16 is 3.64–3.76×
at 1 warp/scheduler and 1.01× at 12.** Occupancy 1→12 at ILP 1 is 3.49–3.84×.

### MLP × occupancy, DRAM (`example02.cu` B), GB/s

| warps/sched \ C | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 1 | **252.5** | 377.2 | 408.7 | 409.9 | 411.1 | 411.0 |
| 2 | 389.5 | 409.7 | 411.6 | 411.6 | 412.0 | 412.0 |
| 4 | 409.3 | 411.9 | 412.1 | 412.3 | 412.4 | 412.3 |
| 8 | 411.0 | 412.3 | 412.5 | 411.8 | 411.0 | 419.8 |
| 12 | 411.6 | 411.5 | 411.1 | 411.2 | 410.9 | 410.1 |

Grouped by product: spread within a product group **0.2–0.5%** (3.2% at product
2); spread within a row up to 63%. Streaming read ceiling **411.9–419.8 GB/s =
95.3–97.2% of the 432 GB/s pin peak**, consistent with M12's 410.5–410.7.

### Single-warp MLP cost (`example02.cu` C, 32 MB L2-resident)

`cycles/step = 240.9 + 15.01·C` (run 1), `245.8 + 15.25·C` (run 2).
Intercept = **M4's L2 latency 241.3**. No hard queue limit to C = 32.
cyc/load falls 290 → 23.

### Little's Law round trip (`exercise03`)

| quantity | value |
|---|---|
| baseline, 5120 threads, 1 scalar load in flight | **68.1–68.4 GB/s** |
| supplied concurrency Q0 | 20,480 B = 20.0 KB |
| derived effective latency | 299.7 ns = **568–571 cycles** (M4: **575**) |
| required concurrency Q* | 123,335 B = 120.4 KB |
| shortfall | 6.02× |
| Q0/L predicted bandwidth | 68.3 GB/s vs 68.3 measured |
| fixed kernel (MLP 4 × `float4`), same 5120 threads | **409.3–409.5 GB/s = 99.4% of the full-occupancy ceiling probe** |

### Exercise 2's four kernels

| | occupancy 1→12 blk/SM | best fix | speedup (3 runs) | mechanism |
|---|---|---|---|---|
| A serialized loads, 1 blk/SM | 2.03–2.24× | hoist 8 loads | **2.07–2.31×** | 2 → 18 `LDG` in the function |
| B `float4` copy, 8 blk/SM | 1.00–1.02× | none exists | **0.99×** | 87–89% of pin peak at 3N |
| C 262,144-long FFMA chain, 5120 threads | n/a (shape only) | 4 accumulators | **2.93–3.25×** | 26% → 85% of the issue ceiling |
| D 64 IEEE divides, 12 blk/SM | 3.65–3.82×, flat past 4 | `__fdividef` | **1.85–2.03×** | 440 → 184 SASS instructions, 15 → 0 `FCHK` |

### The block-shape finding (authoring probes, not shipped)

| configuration | warps/SM | warps/sched | instr/cycle/sched |
|---|---|---|---|
| 1 block × 640 threads | 20 | 5 | **1.008** |
| 5 blocks × 128 threads | 20 | 5 | **0.625** |
| 4 blocks × 128 threads | 16 | 4 | 0.963 |
| 8 blocks × 128 threads | 32 | 8 | 0.963 |
| 4 blocks × 160 threads | 20 | 5 (unbalanced 8/4/4/4) | 0.564 of ceiling |
| 2 blocks × 640 threads | 40 | 10 | 0.993 of ceiling |
| 10 blocks × 128 threads | 40 | 10 | 0.779 of ceiling |

## Assumed from earlier modules

- **M1**: eligible/stalled scoreboard, 4 schedulers × ≤1 instruction/clock, 48
  warp slots, Little's Law, waves and tails, the measured pointer-chase curve
  (linear to 3 warps, knee at 6, 4.87× at 24, per-step latency *worsening* from
  217 to 1070 ns), 40 SMs, 432 GB/s, `%smid` via inline PTX.
- **M3**: grid-stride loops; the linearization rule.
- **M4**: **L1 40.5 / L2 241.3 / DRAM 575 cycles** (all three re-derived or
  cross-checked here); **local memory is DRAM**; 48 MB L2 as the benchmarking
  hazard; `-Xptxas -v`; `cuobjdump -sass`; `clock64()`.
- **M5**: sectors, coalescing, `float4` as a latency/issue optimization with
  identical sector counts, 16 B alignment.
- **M7/M11**: `#pragma unroll 1` as a microbenchmark instrument.
- **M11**: MLP defined; hoisted-vs-serialized coarsening measured at 2.1× / 1.00×
  (the two cells this module generalises); `__fdividef` and `FCHK`;
  the `C = 2` anomaly (explained here as an unroll-factor artefact of the same
  family); compulsory traffic and the floor.
- **M12**: the properly-warmed streaming ceiling 410.5–410.7 GB/s.
- **M16**: the `γ_K·S` tolerance law, reused verbatim for both reassociated
  reductions; the measured FP32 ceiling ~18,000 GFLOP/s (this module measures
  20,900–21,200 on a pure-FFMA kernel and says why); the `clock64()` failure
  mode at full occupancy.
- **M18**: the 18.7× occupancy inversion; the register-tiled GEMM as a kernel
  that is latency-tolerant by construction; CSE eliminating 52 of 64 reads;
  double buffering as a measured loss because the hidden latency was a
  241-cycle L2 hit.
- **M19**: everything about what *limits* occupancy. Not re-taught.
- Spec §12 throughout: 1500 ms stream + 500 ms compute warm-up **plus the
  rule-5b operating-point guard in Exercises 2 and 3**, rotated
  back-to-back sweeps with `SWEEPS ≥ NCFG` (25 and 30 in the examples, 16 and 8
  in Exercise 2), auto-scaled or duration-matched iteration counts, validation
  in a separate untimed pass, min-of-N, ratios as the stable quantity, buffers
  ≥10× L2, every number sanity-checked against a hardware bound, and SASS
  verification of every instruction-level claim.

## Forward-reference debts PAID here

- **M1 → M20: "eligible-vs-stalled stall-reason analysis and ILP as an
  alternative to occupancy."** Delivered as the seven-row taxonomy with the
  does-occupancy-help column, plus the full ILP × occupancy surface on both
  pipes. M1's own curve is re-read in the new language in CYU Q3.
- **M1's knee ("how much occupancy do I need? depends on how many bytes each
  warp has in flight — Modules 19 and 20 pursue this").** Answered
  quantitatively: `Q* = bandwidth × latency`, measured at 120 KB device-wide,
  and the per-thread figure of 2.1 bytes that shows why the question is
  ill-posed without the thread count.
- **M3 → M20: `__restrict__` and ILP.** Partially paid: `__restrict__` is used
  throughout and its role in letting the compiler hoist loads above stores is
  stated, but it is not given a measured section. **Recorded as a residual
  debt** — a `__restrict__`-on/off measurement belongs with M21's or M23's
  treatment, and M18 already recorded a null result for it on GEMM
  (0.991–0.996×).
- **M11 → M20: "ILP vs occupancy, with Exercise 1's 'try this' pointing at
  measuring the exchange rate directly."** Delivered as Exercise 1, which *is*
  that measurement, with a derived model rather than a fitted one.
- **M15 → M20: "the `j` loop as an MLP mechanism."** Paid in the general form
  (Example 2 part B is that loop's parameter sweep); the transpose itself is
  not revisited.
- **M18 → M20: "the other half of the trade."** Delivered, and M18's 18.7×
  result is re-explained in CYU Q3 as the right-hand column of Example 1's
  table.

## Forward references made

- **Module 19 (occupancy)** — named as the owner of every resource question;
  handed the block-shape finding of §8 as something an occupancy sweep must
  control for.
- **Module 21 (roofline)** — named in the decision procedure's step 0 as the
  generalisation of "compute the floor", and as the owner of machine balance.
- **Module 23 (Nsight Compute)** — named as the owner of the stall-reason
  counters. Metric names and section names given; **no counter output quoted**,
  because `ncu` fails with `ERR_NVGPUCTRPERM` here.
- **Module 32 (`cp.async`)** — implicit via M18's double-buffering result,
  cited but not developed.
- **Module 13 (scan)** — cited as the case where a critical path is broken by
  an *algorithm* rather than by ILP.

## Constraints observed

- No shared memory, no barriers, no atomics, no warp intrinsics anywhere in
  Module 20 code. Deliberate: every measurement isolates exactly one stall
  reason, and a barrier would introduce a second.
- No sm_90+ features. No cuBLAS.
- Occupancy is only ever varied over blocks/SM ∈ {1, 2, 4, 8, 12} (plus 3 in an
  authoring probe), for the reason in lesson §8.
- Themes: dependent FFMA chains at parameterised ILP, a streaming read at
  parameterised MLP, a four-kernel diagnosis set, and a strided reduction. The
  dependent-FMA compute-bound kernel is a *reuse* of M1 exercise 2's vehicle,
  which is intentional and stated — M1 measured waves with it, this module
  measures ILP with it. Transpose, GEMM, sorting, sparse formats, n-body and
  large-radius convolution remain untouched.
- Nothing under `module20/` reveals an answer: every prediction is scored
  against something the harness measures at run time, and the three solved
  formulas in Exercise 3 are checked against the harness's independent
  evaluation rather than against stored constants.
- No binaries committed.

## Known issues / honesty notes

- **The block-shape result in lesson §8 is reported without a confirmed
  mechanism.** The fit `W/(4⌈W/4⌉)` is exact over blocks/SM 1..12, the effect
  is cycle-accurate, placement and warp-slot distribution were both verified,
  and a single large block of the same warps does not show it. The hypothesis —
  that warp arbitration has CTA-granular structure — is stated as a hypothesis.
  It is reported rather than smoothed because it will corrupt any occupancy
  sweep built the obvious way.
- **The FP32 ceiling measured here (20,900–21,200 GFLOP/s) is above the index's
  recorded 17,787–18,256.** The kernel is 256 unrolled `FFMA`s with zero memory
  traffic and nothing else, which lets the power manager give the SM clock
  everything; the implied clock is ~2.05 GHz, at the top of this part's
  documented 0.49–2.04 GHz range. The index should widen the FP32 ceiling entry
  rather than treat this as a contradiction.
- **`example02.cu` part B's knee (product 4, 80 KB) is below part A's demand
  figure (124 KB)** and the file says why: `C` counts source-level loads and the
  compiler pipelines across iterations, so the knee is a lower bound on
  concurrency. Exercise 3's `#pragma unroll 1` kernel resolves it at 569 cycles.
  Reported in the file rather than hidden.
- **The identical-seed CSE trap does not exist on nvcc 13.2.** It was expected,
  tested, and found absent (seeding all `C` chains with `1.0f` still yields `C`
  distinct accumulator registers). The exercise therefore does not claim it;
  the solution records the null result.
- **`ncu` remains unavailable** (`ERR_NVGPUCTRPERM`). The entire stall taxonomy
  is taught as theory plus constructed-and-measured experiments. Per the
  standing instruction, no time was spent trying to fix the tool.
- **Spec §12 rule 5b was needed, and it was needed for real.** Exercise 3's
  TODO 5 is the only scored quantity in the module that is an absolute
  bandwidth rather than a ratio. During verification, a run started immediately
  after three other timed programs measured the fixed kernel at **323.6 GB/s —
  still 99.5% of the ceiling measured in the same run**, so the ratio-scored
  TODO 4 passed, but 21% below the healthy figure, so the *correct* TODO 5
  answer scored `FAIL`. Exercises 2 and 3 now both carry Module 15's guard:
  probe a ceiling after warming, and if it is low idle 10 s and re-warm, up to
  five attempts, then warn and proceed. Exercise 3's floor is 370 GB/s, Exercise
  2's is 330 GB/s (it probes its own kernel B). TODO 5's tolerance was also
  widened from 10% to 12%. Exercise 1 scores only ratios and cycle counts and
  needs no guard.
- **Absolute figures move 5–15% with thermal state.** Every claim in the lesson
  and the solution notes is a ratio or a range, and every scored gate sits in
  an empty gap of the measured distribution with ≥14% of margin (spec §12 5d).
- One residual debt is **not** paid: M3's `__restrict__`-and-ILP reference gets
  a qualitative treatment only (see above).

## Cross-module observations for the index

1. **§6b should gain the two constants this module exists for:**
   **dependent-FFMA latency 4.05 cycles** and **saturated issue interval 1.067
   cycles per warp-instruction per scheduler on sm_89**, with the derived
   requirement of **4 independent instructions in flight per scheduler**. These
   are the FP32 analogue of the 575-cycle DRAM figure and they are what make
   M18's occupancy inversion explicable.
2. **§6b should gain the ILP/occupancy exchange rate**: one unit of per-thread
   ILP replaces one resident warp per scheduler (four per SM), up to the
   product `latency × throughput`, after which neither buys anything. Measured
   on both the FP32 pipe (3.64–3.76× at 1 warp/sched, 1.01× at 12) and the
   memory system (cells sharing `warps × MLP` agree to 0.2–0.5%).
3. **⚠️ A new hazard for any occupancy sweep, and M19 in particular.** Resident
   warps delivered as many small blocks are worth less than the same warps
   delivered as few large blocks on a dependency-bound kernel: 20 warps/SM as
   one 640-thread block = 1.008 instr/cycle/scheduler, as five 128-thread
   blocks = 0.625. Blocks/SM of 5, 6, 7, 9, 10, 11 sit on a reproducible
   sawtooth `W/(4⌈W/4⌉)`. Mechanism unconfirmed. **Recommendation for the
   index: sweep occupancy over blocks/SM ∈ {1,2,3,4,8,12} only, or vary the
   block size instead.**
4. **§6b's FP32 ceiling should be widened to 17,787–21,200 GFLOP/s**, with the
   note that the high end is a pure-FFMA kernel with no memory traffic and the
   low end is a kernel that also moves data.
5. **M4's 575-cycle DRAM latency is independently confirmed** by a completely
   different method (Little's Law inverted on a bandwidth measurement at a known
   concurrency): **568–571 cycles**. And **M4's 241.3-cycle L2 latency is
   independently confirmed** as the intercept of a least-squares fit to a
   single-warp MLP cost curve: **240.9 and 245.8** in two runs. Worth recording
   as cross-validated rather than single-sourced.
6. **A second confirmation of M16's `clock64()` caveat, with a number.** At one
   warp per SM, `clock64()` cross-checks against wall time at 1.66–2.08 GHz and
   is trustworthy. At 12 blocks/SM the same arithmetic reports **5.73
   instructions per cycle per scheduler** — 5.7× the hardware maximum of 1 —
   i.e. an implied clock near 0.33 GHz. Spec §12 rule 13 (sanity-check against
   a hardware bound) catches it immediately.
7. **§6 "themes already used" should gain**: parameterised-ILP dependent FFMA
   chains, the MLP × occupancy surface, and the four-cause diagnosis set.
8. **A new entry for spec §12**: when sweeping a parameter that changes how
   much work a loop body does, **hold the work per loop branch constant, not the
   trip count.** Measured here: with the trip count held constant instead,
   `ptxas` unrolled the `C = 2` instantiation 3× less than its neighbours and
   produced a 25% artefact that looks exactly like a hardware effect. This is
   very likely the same mechanism behind M11's unexplained `C = 2` anomaly.
