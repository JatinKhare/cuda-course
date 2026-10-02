# Module 21 manifest — The Roofline Model

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module21/lesson.md`, `module21/example0{1,2}.cu`,
`module21/exercise0{1,2,3}.cu`.
Solutions: `solutions/module21/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module21/check_your_understanding.md`,
`solutions/module21/MANIFEST.md`.

All `.cu` verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2, V13.2.51, RTX 3500
Ada, driver 596.71), warning-clean. No file links cuBLAS. Both examples print
`OVERALL: PASS`; all three solutions print `OVERALL: PASS` (6/6, 9/9, 9/9). All
three shipped exercises compile with TODOs blank and exit gracefully
(`Set TODO 4 first.`, `Set TODO 1 first.`, `Set TODO 1 first.`) returning 0.
No binaries committed.

**Module 21 is the integration module for Parts II–V.** It owns no new CUDA
API. It takes the ceilings Modules 11–18 measured, assembles them into one
model, and classifies six kernels the reader has already written.

---

## Concepts taught

- **Arithmetic intensity**, defined as `F/B`, with the insistence that `B` is
  ambiguous and that an AI without a named level is not a statement.
  **PORTABLE CUDA CONCEPT.**
- **Four byte counts for one kernel**: compulsory, DRAM, requested (what memory
  instructions ask for, any address space), useful. Their measured ratios on
  this course's kernels: 724x (M16's naive GEMM, requested/compulsory), 8x
  (sector amplification in a pointer walk), 1.5x (M11's 3N vs 2N in-place
  SAXPY).
- **The roofline**: `attainable = min(P, AI x S)`; the memory-bound slope; the
  compute-bound plateau; the **ridge point** `P/S` as the machine balance; how
  to place a kernel on it.
- **The hierarchical roofline**: one sloped ceiling per level, each with its own
  ridge point and its own AI for the same kernel, and the binding constraint as
  the `min` over levels.
- **The on-chip operand-fetch ceiling as ONE ceiling shared by L1 and shared
  memory.** They are the same SRAM and the same return path; M16 measured the
  law with `LDG` and M17 with `LDS` and got the same curve to 1%. Measured here
  from the other direction: naive, tiled and register-tiled GEMM all sit on it,
  at 0.90, 1.14 and 1.13 of their predictions.
- **Why a two-axis roofline cannot express tiled GEMM** (M17's explicit debt):
  naive and tiled GEMM have *identical* DRAM AI (181) and *identical* request AI
  (0.25), while running at 6.8% and 8.3% of the compute ceiling; the DRAM axis
  calls both compute bound. The shared axis is the only one on which they are
  distinguishable from a compute-bound kernel at all.
- **A level's roofline binds only if the traffic you counted crosses that
  level.** Requested bytes against the L2 ceiling predicts 326 GFLOP/s for a
  kernel that measures 1169 — 3.6x "above the roof" — because the requests hit
  in L1.
- **The instruction-issue ceiling**: 4 schedulers/SM x 1 instruction/cycle x
  40 SMs = 160 warp-instructions per cycle, measured at **310 G/s**. And the
  unification: **the FP32 plateau is the issue ceiling evaluated at 64 FLOP per
  instruction**, so there is no compute ceiling that is not also an issue
  ceiling. A kernel at FFMA density `d` has a plateau of `d x 64 x issue`.
- **The issue ceiling demonstrated with two kernels differing by one
  instruction**: `fmaf(a,b,1)` vs `fmaf(a,b,1)+b`, which IEEE semantics forbid
  re-associating. SASS loop bodies 68/128 FLOP and 132/192 FLOP; predicted ratio
  0.77, measured 0.78, at an identical issue rate.
- **The measured FP32 "peak" is 94% of the lane peak** because even the best
  loop spends 4 of 68 slots on the counter, compare and branch.
- **Classification as a seven-step procedure**, with a reconciliation table:
  0.8–1.25 = at the roof; >1.25 = the ledger is wrong; 0.25–0.8 = right ceiling
  imperfectly reached; <0.25 = nothing is saturated, go to Little's Law.
- **Where the roofline lies**: perfect overlap; **latency** (measured 58x below
  the roof on a dependent walk, reconciled by Little's Law to within 2x); tail
  and wave effects (M1); and the non-reproducibility of the ceilings themselves
  (279–411 GB/s, same binary, one afternoon).
- **Little's Law as the missing equation**: `bytes in flight = bandwidth x
  latency`; 122–129 kB on this part; a one-warp dependent walk supplies 1024 B
  and achieves 1.7% of the DRAM ceiling, predicted 0.84%.
- **Bounding a microbenchmark before believing it** (spec §12 rule 13), with the
  broken probe shipped alongside the correct one and differing by one statement.
  Hoisted probe measures 32–36 TB/s, 216–332% of the bank array's maximum.
- **Which clock may appear in a bound**: not `cudaDevAttrClockRate` (1.545 GHz,
  produces a "peak" the measurement exceeds by 1.15x), not the clock implied by
  a measurement, **and not the highest clock ever observed** (2.04 GHz — a bound
  built on it rejects a correct 329.6 G instr/s measurement, observed during
  authoring). Only `nvidia-smi --query-gpu=clocks.max.sm` (3105 MHz).
- **Two timing groups, not one rotated sweep**, when the configurations are
  axes of a model rather than competitors: spec §12 rule 1 is about competitors,
  rule 4's corollary is about warm-ups, and they pull in opposite directions
  here. Measured consequence: 326 GB/s + 13 750 GFLOP/s in one sweep versus
  **410 GB/s + 18 642 GFLOP/s** in two.
- **Design to a target as a derivation**: `AI_needed = target x ridge`, then the
  only transformation that raises AI at the on-chip level is reusing a loaded
  value across more than one FLOP — i.e. a change to the decomposition.
- **Staging through shared memory does not raise the on-chip AI.** Measured on
  a 17x17 convolution: shared tile with 1 output per thread is **0.96x** the
  naive kernel. M6's "the caches got there first" and M17's "tiling changed the
  opcode, not the count", on a third kernel.

## CUDA API / intrinsics / syntax introduced

Nothing new, deliberately. Module 21 is an integration module. Reused and
exercised:

- `cudaEvent_t` timing; `cudaFuncGetAttributes` (`numRegs`, `localSizeBytes`,
  `sharedSizeBytes`); `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- `__constant__` + `cudaMemcpyToSymbol` (M4), used in Exercise 3 so the filter
  weight becomes an FFMA constant-bank operand rather than a load
- `__shared__`, `__syncthreads()`, `__shfl_down_sync`, `float4`, `fmaf`,
  `__launch_bounds__`, `#pragma unroll` / `#pragma unroll 1` / `#pragma unroll 8`
- `std::numeric_limits<float>::infinity()` as a writtenness sentinel (M16)
- FNV-1a answer hashing (M11/M12 convention);
  `setvbuf(stdout, NULL, _IONBF, 0)` (M13 convention, spec §5)
- `nvidia-smi --query-gpu=clocks.max.sm --format=csv` — **new use**: the only
  clock figure admissible in a hardware bound
- `nvcc -arch=sm_89 -O3 -cubin` + `cuobjdump -sass`, used to count loop bodies
  rather than to read opcodes
- SASS vocabulary used: `LDS`, `LDS.128`, `LDG`, `FFMA` (incl. the
  `c[0x3][...]` constant-bank operand form), `FADD`, `LEA`, `LOP3`, `IMNMX`,
  `BAR.SYNC.DEFER_BLOCKING`

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | **Building the roofline.** A: five ceilings in two timing groups (1500 ms stream -> DRAM + L2; 500 ms compute -> shared scalar, shared vector, FFMA, issue), each printed against a hardware bound. B: ridge points for all four memory ceilings. C: an ASCII log-log roofline with the known kernels placed on it. D: the hoisted shared probe, rejected, with the SASS loop-body counts (133 instructions/32 LDS versus 35/0) as the evidence. Also the FFMA-vs-mixed pair that isolates the issue ceiling. |
| `example02.cu` | **Classify, predict, measure, reconcile.** Six kernels reproduced verbatim from M11/M12/M16/M17/M18/M1+M4. The full ledger and the prediction printed before anything is timed; one rotated sweep of 6; then a reconciliation that names the verdict per row, prints the DRAM traffic each kernel actually achieves (2.2% of the ceiling for tiled GEMM), and closes the chase kernel with Little's Law arithmetic that lands on the measured number. |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Fill-in + design (§6 types 1, 5) | 5 | Write the DRAM, shared and FP32 probes, derive the four hardware bounds, assemble the roofline; a shipped broken probe must be rejected | **Three.** (a) TODO 2 must defeat *three* independent hazards — bank conflicts, `LDS.128` contraction, and loop-invariant hoisting — and only the third one is visible in the output; the first two silently measure a different thing. (b) TODO 3's obvious `#pragma unroll 1` outer loop gives a loop body of 8 FFMAs + 4 overhead instructions and a reproducible, entirely fictitious ceiling at **13 750 GFLOP/s** instead of 18 642 — the issue ceiling contaminating the measurement that is supposed to be about arithmetic. (c) TODO 4's trap is the clock: a bound built on the **highest clock ever observed on this part (2.04 GHz)** is a reasonable-sounding choice that rejects a correct 329.6 G instr/s measurement of this very program. Only the device maximum (3.105 GHz) is admissible, and it is deliberately loose. |
| `exercise02.cu` | Performance reasoning + design (§6 types 2, 5, 6) | 5 | Six known kernels: ledger, classify, predict the level and the bucket, then reconcile the one the model fails on | **Four.** (a) `gemmTiled`'s requested bytes are **identical** to `gemmNaive`'s — tiling changed the opcode, not the count — and a reader who divides by the tile size predicts 21 700 GFLOP/s and is 13x out. (b) `triad` is 3N not 2N (M11's measured trap); a 2N ledger turns a kernel at 0.91 of its roof into one at 0.60 and invites blaming the hardware. (c) TODO 3 asks what *actually* limits each kernel and is explicitly **not required to agree with the reader's own `classify()`**: the chase kernel's roofline answer is `DRAM` and the true answer is `LATENCY`, and a reader who has correctly run the model and believed it loses the point. (d) the chase requests **fewer** bytes than it moves, the only such row, and getting its `dramBytes` wrong breaks the Little's Law reconciliation rather than the prediction. |
| `exercise03.cu` | CPU->GPU / design from scratch (§6 type 5) + prediction | 5 | 17x17 convolution at 12.8% of the ceiling; reach 28% and 2.5x, with the optimization never named | **Three.** (a) The move everyone makes first — stage the input through shared memory — is a measured **4% LOSS** (0.96x), because it changes which cache answers a load without changing loads per FLOP. (b) The weight comes from `__constant__` and `ptxas` folds it into the FFMA as a `c[0x3][...]` operand, so `AI = 2/4`, not `2/8`; a reader who assumes a weight load is a factor of two out on the *baseline*. (c) `minOutputsPerThread` must round **up**: `P = 2` is the honest result of rounding down and measures 22.6% against a 28% gate. Bonus cliff: `OPT = 8` is **0.76x the naive kernel**, so the design space is not monotone in the thing being optimized (M18's lesson, new resource). |

Scoring: Ex1 6 points (3 probe bands incl. their own bounds, 2 hashed function
checks, 1 rejection of the broken probe); Ex2 9 (ledger 2, classify 1, levels 1,
buckets 2, Little's Law 2, validation 1); Ex3 9 (analysis 2, design consistency
1, correctness 2, speed gate 2, ceiling gate 1, bucket 1). `OVERALL: PASS`
requires full marks in all three.

## Measured results recorded in this module

### Ceilings (RTX 3500 Ada, CUDA 13.2)

| quantity | measured | bound used | index value |
|---|---|---|---|
| DRAM read, 256 MB buffer | **410.1 – 410.8 GB/s** (95.0% of 432); 279–411 across runs | 432.0 GB/s (pin rate) | 410.5–410.7 ✓ |
| L2 read, 24 MB working set | 1259 – 1938 GB/s | — (must exceed DRAM) | ~1305 ✓ |
| shared read, scalar `LDS` | **4777 – 5452 GB/s** | 15 898 GB/s @ 3.105 GHz | 5380–5400 (slightly high; this probe carries address arithmetic) |
| shared read, `LDS.128` | **9019 – 10 008 GB/s** | 15 898 GB/s | 10 240–10 300 ✓ |
| `LDS.128` / scalar ratio | **1.81 – 1.92** | — | 1.90 ✓ (M7's two-cycle floor) |
| FP32 FFMA | **17 123 – 19 847 GFLOP/s** | 31 795 GFLOP/s | 17 787–18 256 ✓ (top of range slightly above) |
| implied SM clock | 1.67 – 1.93 GHz | — | ~1.78 ✓ |
| issue rate | **286 – 330 G warp-instructions/s** | 496.8 G/s | **new** |
| machine balance (DRAM ridge) | 33.6 – 53.2 FLOP/byte | — | 43–44 ✓ (centre of range) |
| on-chip ridge | **2.87 – 3.72 FLOP/byte** | — | **new** |
| `LDS.128` ridge | 1.48 – 1.92 FLOP/byte | — | **new** |

### The issue ceiling isolated

| kernel | SASS loop body | FLOP/instr | GFLOP/s | issue rate |
|---|---|---|---|---|
| `ffmaProbe` | 68 instructions, 64 `FFMA` | 1.88/lane | 18 642 | 309.5 G/s |
| `mixedProbe` | 132 instructions, 64 `FFMA` + 64 `FADD` | 1.45/lane | 14 474 | 311.0 G/s |
| ratio | | | **0.78** | **1.005** |

Predicted ratio from the instruction counts: 0.77.

### The hoisted-probe trap

| | honest | hoisted |
|---|---|---|
| source difference | `base += 1` | — |
| SASS loop body | 133 instructions, **32 `LDS`** | 35 instructions, **0 `LDS`** |
| reported bandwidth | 4777 – 5452 GB/s | **32 965 – 36 472 GB/s** |
| vs the 15 898 GB/s bound | 30 – 34% | **207 – 229%** — rejected |
| ratio | 1.00 | **6.5 – 6.9x** |

### Classification of six known kernels (example02 / exercise02)

| kernel | AI(dram) | AI(req) | binding level | predicted | measured | meas/pred | % of FP32 |
|---|---|---|---|---|---|---|---|
| triad (M11) | 0.167 | 0.167 | DRAM | 62 – 69 | 49 – 62 | 0.79 – 0.91 | 0.25 – 0.33% |
| reduce v6 (M12) | 0.250 | 0.250 | DRAM | 93 – 103 | 80 – 99 | 0.85 – 0.97 | 0.40 – 0.53% |
| naive GEMM (M16) | 181.1 | 0.250 | on-chip | 1295 – 1357 | 1169 – 1343 | **0.90 – 0.99** | 6.1 – 6.8% |
| tiled GEMM (M17) | 181.1 | 0.250 | on-chip | 1295 – 1357 | 1474 – 1634 | **1.14 – 1.24** | 7.6 – 8.3% |
| register-tiled (M18) | 181.1 | 1.333 | on-chip | 6911 – 7248 | 7367 – 8154 | **1.06 – 1.14** | 37.9 – 41.6% |
| pointer chase (M1/M4) | 0.031 | 0.250 | **latency** | 11.6 – 12.8 | 0.21 – 0.23 | **0.016 – 0.019** | 0.001% |

DRAM traffic actually achieved: triad 374 GB/s, reduce 396 GB/s, naive GEMM
**7.1 GB/s (1.7%)**, tiled GEMM **8.9 GB/s (2.2%)**, register-tiled 41 GB/s,
chase 6.8 GB/s.

Little's Law on the chase: needed 122–129 kB in flight, supplied 1.00 kB,
predicted fraction 0.0084, measured 0.0172, **ratio 2.0–2.2**. Implied round
trip 280–288 cycles against M4's 575 for a single-thread chase.

### Exercise 3, the convolution ladder (2048x2048, 17x17, 2.42 GFLOP)

| kernel | GFLOP/s | % of ceiling | x naive |
|---|---|---|---|
| naive, global loads | 2445 – 2497 | 12.3 – 12.8% | 1.00 |
| **shared tile, 1 output/thread** | 2357 – 2375 | 12.1 – 12.4% | **0.95 – 0.96** |
| shared tile, 2 outputs/thread | 4258 – 4352 | 21.8 – 22.6% | 1.71 – 1.76 |
| **shared tile, 4 outputs/thread** | 7242 – 7558 | 37.2 – 38.8% | **2.94 – 3.07** |
| shared tile, 4x4 block, 4/thread | 7315 – 7438 | 37.2 – 38.4% | 2.95 – 2.99 |
| shared tile, 8 outputs/thread | 1838 – 1900 | 9.3 – 9.8% | **0.74 – 0.76** |

SASS census: `convNaive` 1592 instructions, 289 `FFMA`, 289 `LDG`, 580 `LEA`,
289 `LOP3`, FFMA density 18%, every FFMA carrying a `c[0x3][...]` weight
operand. `convFast` (OPT=4) 1648 instructions, **1156 `FFMA`, 340 `LDS`**,
density 70% — giving `AI = 1156x2/(340x4) = 1.70 FLOP/byte`, exactly the
`0.5 x OPT x DIAM/(DIAM+OPT-1)` the design predicted. 39 registers, 0 spilled,
9216 B shared, 6 blocks/SM.

## Assumed from earlier modules

- **M1**: 40 SMs, 4 processing blocks with one scheduler each (the issue
  ceiling), 128 FP32 lanes/SM, **Little's Law**, waves and tail effects, the
  pointer-chase vehicle, ILP as a substitute for occupancy.
- **M2/M3**: launch syntax, `CHECK`, both error checks, grid-stride loops, the
  linearization rule, ceil-divide, bounds guards.
- **M4**: the storage map with latencies (**DRAM 575 cycles**, used in Little's
  Law), the 48 MB L2 as the benchmarking hazard, `__constant__` and the
  broadcast rule, `cuobjdump -sass`, the pointer-chase method.
- **M5**: 32 B sectors and the sector-counting procedure (the chase's 8x
  amplification), `float4`, effective vs DRAM bandwidth.
- **M6**: cooperative loading, load mapping != compute mapping, **"the caches
  got there first"**, the K/H reuse argument, the capacity->occupancy table.
- **M7**: `bank = (addr/4) % 32`, `gcd(k,32)`, broadcast is free, **the
  `max(2,D)` two-cycle floor** (which is why scalar `LDS` gets half the bank
  array), `#pragma unroll 1` as an instrument, the warning that the compiler
  vectorises out from under an analysis.
- **M9**: `__syncthreads()`'s two guarantees, the uniformity rule.
- **M11**: **compulsory traffic and the floor**, `x floor` as a reporting unit,
  the **3N in-place SAXPY** result, write-allocate, MLP, "a model has a domain".
- **M12**: the **1500 ms warm-up** finding and the 410.5 GB/s streaming ceiling;
  reduction rung v6 reproduced verbatim.
- **M14**: ratios between the reader's own kernels as the stable gate.
- **M15**: the matched-ceiling argument (a transpose must be measured against a
  copy, not against a stream) — cited for the triad row.
- **M16**: the GEMM problem statement, the `beta == 0` contract, the
  **724x compulsory/requested ratio**, the FP32 ceiling recovery without
  `cudaDevAttrClockRate`, the two-stage warm-up, `gemmValidate` (the reduced
  two-check form is reused in both GEMM-bearing programs), the FMAs-per-
  operand-fetch law.
- **M17**: **shared-memory read bandwidth as a ceiling** (5.38–5.40 / 10.24–
  10.30 TB/s), the 72 TB/s demand figure, the 7.4–14.2% cap, the loop-invariant
  hoisting failure mode (40 TB/s), tiled GEMM reproduced verbatim.
- **M18**: the two-level reuse law `TM·TN/(TM+TN)`, the register-tiled kernel
  reproduced verbatim, FFMA density as the number that tracked performance,
  the occupancy cliff, `BN/TN >= 16`.
- Spec §12 throughout: 1500 ms stream + 500 ms compute warm-up, rotated
  back-to-back timing with `SWEEPS >= NCFG` within each group, iteration counts
  auto-scaled to ~10 ms segments, min-of-N, validation in a separate untimed
  pass, buffers >= 5x the L2, `cudaDevAttrClockRate` rejected, **rule 13
  (sanity-check against a hardware bound) as the subject of Exercise 1**.

## Forward-reference debts PAID here

- **M3 → M21**: "'achievable' vs 'peak' bandwidth; why a compulsory-traffic
  model can report 62% while the kernel is at its real ceiling." Paid: §2's four
  byte counts, and the triad row where a pure-read ceiling over-predicts a
  read+write kernel by 10%.
- **M5 → M21**: "the 'DRAM is saturated, stop requesting unwanted sectors'
  conclusion is the memory-bound side of Module 21." Paid in §1–3 and in
  Example 2's triad and reduce rows (0.91 and 0.97 of their rooflines).
- **M11 → M21**: "the generalisation of this module's floor", and "the FFMA row
  of the crossover table is the memory-bound plateau drawn from measurement."
  Paid: the floor becomes one of four plateaux, and M11's 3N trap is a scored
  row of Exercise 2's ledger.
- **M15 → M21**: "the copy ceiling as the memory-bound plateau." Paid, and
  extended: M15's insistence on a *matched* denominator is cited as the reason
  the triad row lands at 0.91 rather than 1.00.
- **M16 → M21**: named three times as the owner of the full roofline treatment;
  M16 built a two-point roofline from measured ceilings and said so. Paid: the
  two points become five ceilings and four plateaux, and M16's 181-vs-0.25
  FLOP/byte pair is the worked example of §2.
- **M17 → M21**: "this module adds a third ceiling (shared-memory bandwidth)
  that a two-axis roofline cannot express." Paid in full — the shared axis is
  built, measured, given a ridge point, and shown to predict naive, tiled and
  register-tiled GEMM to within 24%, 14% and 14% from one column.
- **M18 → M21**: "implicit; the module works in % of measured ceilings rather
  than rebuilding a roofline." Paid: the roofline is rebuilt and M18's
  register-tiled kernel is placed on it at 1.333 FLOP/byte.

## Forward references made

- **Module 19 (occupancy)** and **Module 20 (latency hiding)** — named as the
  owners of everything in "where the roofline lies" §6(b). The chase kernel's
  58x deficit is handed to them explicitly, with the Little's Law arithmetic
  done so they can build on it.
- **Module 23 (Nsight Compute)** — named as the owner of the automated roofline
  chart, with the metric names given (`dram__bytes.sum`, `l1tex__t_bytes.sum`,
  `smsp__inst_executed.sum`,
  `sm__sass_thread_inst_executed_op_ffma_pred_on.sum`,
  `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`). `ncu` remains
  unavailable (`ERR_NVGPUCTRPERM`); **no counter output is quoted anywhere** and
  every number in this module was constructed and measured directly. Named
  specifically as the tool that would settle the tiled-GEMM 1.14–1.24x
  over-prediction.
- **Part XIV (Modules 41–42, CUDA for AI)** — named with the content, not just
  the number: a batch-1 decoder matrix-vector product has an arithmetic
  intensity near 0.5 FLOP/byte against a ridge of 45, so inference is a
  bandwidth problem and training is a compute problem.
- **Module 32 (`cp.async`)**, **Modules 33–34 (Tensor Cores)**, **Module 43
  (CUTLASS)** — not mentioned; this module stays out of their territory.

## Cross-module observations for the index

1. **§6b should gain three rows.** (a) The **instruction issue ceiling**:
   160 warp-instructions per cycle device-wide, measured **286–330 G/s**, with
   the unification that the FP32 plateau *is* the issue ceiling at 64 FLOP per
   instruction — so a kernel's compute plateau is `density x 64 x 310` GFLOP/s,
   not a constant. (b) The **on-chip operand-fetch ridge point, 2.87–3.72
   FLOP/byte**, which is the number that predicts all three GEMMs. (c) The
   measured **FP32 peak is 94% of the lane peak** because the best possible loop
   still spends 4 of 68 slots on loop control.
2. **Spec §12 rule 1 needs a scope note.** "Time all configurations back-to-back
   in one loop" applies to *competing* configurations. Ceilings on different
   axes of one model are not competitors and need different warm-ups; putting
   them in one sweep measured 326 GB/s and 13 750 GFLOP/s where two groups
   measured 410 and 18 642. M16 already did this and said so in a comment; it
   should be in the spec.
3. **Spec §12 rule 13 needs the clock clause.** A bound is only a bound if its
   clock is the device *maximum* (`nvidia-smi --query-gpu=clocks.max.sm`,
   3105 MHz here). A bound built on the highest clock ever *observed* (2.04 GHz,
   the figure §12 currently quotes) rejects a correct measurement: Exercise 1's
   issue probe measured 329.6 G instr/s against a 326.4 bound in one run.
4. **A new entry for §12's "the compiler will vectorize AND ELIMINATE"
   catalogue**, with the exact SASS: an under-unrolled FFMA throughput probe
   (`#pragma unroll 1`, 8 chains) measures **13 750 GFLOP/s** reproducibly
   because its loop body is 12 instructions for 8 FFMAs. Nothing is eliminated
   and nothing is vectorised; the loop overhead simply *is* 33% of the issue
   slots. Unroll 8 deep and the identical kernel measures 18 642.
5. **M16's "6.4–6.5 FMAs per operand-fetch instruction" restated geometrically.**
   The threshold is `0.8 x ridge x (bytes per fetch instruction) / 2`, i.e. it is
   the on-chip ridge point in instruction units. Having the ridge point makes the
   threshold derivable rather than empirical.
6. **No inconsistency with Modules 1–18 was found.** Every ceiling this module
   re-measured reproduced the index's recorded value (DRAM 410.1–410.8 against
   410.5–410.7; `LDS.128`/scalar ratio 1.81–1.92 against 1.90; L2 1259–1938
   against ~1305; machine balance centred on 43–47 against 43–44). The one
   number slightly outside its recorded band is the scalar `LDS` figure
   (4777–5452 against M17's 5380–5400), and the reason is visible in the SASS:
   this module's probe spends 65 of 133 loop instructions on address arithmetic
   to defeat hoisting and contraction, so it under-reports the array slightly.
   The *ratio* to the vector probe, which M17 identified as the trustworthy
   quantity, reproduces exactly.

## Constraints observed

- **No new CUDA API.** Everything is reused; this is stated in the lesson and
  enforced in the manifest.
- **No occupancy teaching** (M19) and **no latency-hiding teaching** (M20).
  Both are named as the owners of the "far below every ceiling" case and neither
  is developed: Exercise 2 computes one Little's Law ratio and stops.
- **No `ncu`** (`ERR_NVGPUCTRPERM`), no counter output quoted anywhere. Every
  ceiling, every byte count and every instruction count in this module was
  constructed by a probe or counted in `cuobjdump -sass`.
- **No sm_90+ features**, no `cp.async`, no Tensor Cores, no cuBLAS link.
- Themes: the **17x17 large-radius image convolution** is new and was reserved
  by index §6. The six classified kernels are all reproductions, explicitly
  labelled as belonging to the modules that built them, and none of them is
  re-taught. n-body, sorting and sparse formats remain reserved and unused.
- Non-power-of-two and non-square sizes present throughout: GEMM at
  1027x2053x769, `NELEM = 2^25`, `CHASE_LEN = 2^26` with a prime stride.
- Timing per spec §12: 1500 ms streaming + 500 ms compute warm-ups in the
  documented order, two timing groups with rotation and `SWEEPS >= NCFG` inside
  each, iteration counts auto-scaled to ~10 ms, min-of-N, validation in a
  separate untimed pass, buffers >= 5x the 48 MB L2 for every DRAM figure, every
  figure above 100% of a bound explained rather than hidden, and bucket edges
  placed in the widest empty gaps with >= 8 points of margin.
- Nothing under `module21/` reveals an answer: Exercise 1's bounds and roofline
  functions, Exercise 2's ledger, classification and levels, and Exercise 3's
  analysis are all scored against FNV-1a hashes; every performance prediction is
  scored against a measurement taken in the same run.
- No binaries committed.

## Known issues / honesty notes

- **Tiled GEMM measures 1.14–1.24x its own prediction, reproducibly.** Reported
  as an over-prediction of its cost rather than smoothed. Two candidate
  mechanisms are named in the Exercise 2 solution notes — the `As[ty][k]`
  broadcast (M7: degree 1, one bank access for 32 lanes) and the `LDS.128`
  contraction M17 measured (4 vector + 16 scalar reads for 16 FFMAs) — and the
  counter that would settle it
  (`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`) is unavailable.
- **The triad row measures 0.79–0.91x its roofline.** Explained in the program's
  own output: the DRAM ceiling is measured with a read-only stream and triad
  reads two arrays and writes one. M15's matched-denominator argument applies and
  is cited; the ceiling was not changed, because changing it would have hidden
  the lesson.
- **Absolute figures move with thermal state**, as everywhere in this course.
  The DRAM ceiling measured 279–411 GB/s and the FP32 ceiling 17 123–19 847
  GFLOP/s across runs of the same binaries in one afternoon. Every scored gate
  in the exercises is a ratio, a bucket, or a band wide enough to span that; the
  bucket edges were placed after measuring the real spread.
- **The `LDS.128` probe sometimes exceeds the bound computed at the *implied*
  clock** (not the bound used for scoring, which uses the device maximum). This
  is M17's finding reconfirmed: the shared-memory kernel clocks higher than the
  FFMA kernel, so the implied clock from the FFMA ceiling is not the clock the
  shared probe ran at. Example 1 prints both bounds and says so.
- **This module's scalar `LDS` figure is 1–11% below M17's.** Its probe pays 65
  of 133 loop instructions in address arithmetic to defeat hoisting and vector
  contraction simultaneously, which M17's did not have to do. The ratio to the
  vector probe is unaffected and reproduces M17's 1.90 to within 5%.
- **`compute-sanitizer --tool memcheck` was run on `example01.exe` and
  `exercise03_solution.exe`** (full path: `C:\Program Files\NVIDIA GPU Computing
  Toolkit\CUDA\v13.2\bin\compute-sanitizer.bat`; it is not on `PATH`, as M15
  recorded). Both report **no invalid accesses** — `ERROR SUMMARY: 1 error` is
  entirely the benign `Resetting device while there are still other users
  claiming to use it` API warning that the house `cudaDeviceReset()` convention
  produces (M12's finding, reconfirmed in M14 and here).
  Both programs print `OVERALL: FAIL` *under instrumentation*, because every
  gate in this module is a throughput gate and instrumentation slows the GPU by
  one to two orders of magnitude: `exercise03_solution` measured 0.4% of the
  ceiling instead of 38.8%, and `example01`'s L2 probe fell to 56 GB/s so its
  "L2 > DRAM" sanity check failed. This is expected and is recorded here so
  nobody re-diagnoses it: **run the sanitizer for memory safety and the plain
  binary for scoring.** `exercise02_solution` was not instrumented (a 45 s run
  at a 60x slowdown is 45 minutes); its kernels are the same six reproduced from
  M11–M18, all of which were memchecked in their own modules.
