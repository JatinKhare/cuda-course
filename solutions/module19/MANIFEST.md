# Module 19 manifest — Occupancy

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module19/lesson.md`, `module19/example0{1,2}.cu`,
`module19/exercise0{1,2,3}.cu`.
Solutions: `solutions/module19/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module19/check_your_understanding.md`,
`solutions/module19/MANIFEST.md`.

All `.cu` verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2 V13.2.51, RTX 3500 Ada
Generation Laptop GPU, driver 596.71), **warning-clean**. Both examples print
`OVERALL: PASS`; all three solutions print `OVERALL: PASS` (10/10, 10/10,
10/10). All three shipped exercises compile with TODOs blank and exit gracefully
with `Set TODO n first.` and return 0. No binaries committed.

**Part VI opens here.** Module 19 owns the *resource arithmetic* and the
*decision rule*. It does not re-teach latency-hiding mechanics (Module 20 owns
the warp-scheduler and stall-reason treatment) or the roofline (Module 21), and
says so in the lesson, in both examples and in the Exercise 3 notes.

---

## Concepts taught

- **Occupancy defined as resident warps / 48**, and the insistence that it is a
  count of warp slots and not a measure of speed, efficiency or utilisation.
  **PORTABLE CUDA CONCEPT** (the ratio); **ARCHITECTURE-SPECIFIC** (the 48).
- **Occupancy buys latency tolerance and nothing else**, with Module 1's
  measured pointer-chase curve (linear to 3 warps, knee at 6, flat at 4.87× by
  24) as the evidence, and the two corollaries: past the knee more warps buy
  nothing, and warps are bought with per-thread state.
- **The four placement limiters, exactly.** `blocks/SM = min(registers, shared,
  warp slots, block slots)`, each derived and each demonstrated binding on real
  kernels. **This pays the debt the cross-module index records against Module 6.**
- **Warp slots, not threads/SM.** `48 / ceil(threads/32)`, which differs from
  `1536 / threads` for every block size that is not a multiple of 32 — measured
  at 100 threads: 12 blocks, not 15.
- **Block slots (24)** as the limiter that caps a 32-thread block at 50 %
  occupancy regardless of everything else.
- **Shared memory**: Module 6's `roundUp(bytes + 1024, 128)` into 102400 B,
  reproduced exactly (16384 B → 5, 25600 B → 3, 49152 B → 2), plus the
  static-vs-dynamic trap in `cudaFuncGetAttributes().sharedSizeBytes`.
- **⚠️ The register limiter has TWO quantisations, and Module 18 only had one.**
  Granule 8 registers per thread per warp (M18), *and* the register file is
  **four 16384-register slices** with a warp drawn entirely from one slice (M1).
  `blocksByRegs = 4 * floor(16384 / (roundUp(R,8)*32)) / warpsPerBlock`.
  Verified against `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on **137
  kernels** (block sizes 32–1024, register counts 18–177): the slice model is
  exact on all 137, the aggregate 65536-pool model is **wrong on 10**.
- **The per-block 49152 B shared limit is a gate, not a limiter** — it returns
  zero blocks (a launch error), not fewer blocks; likewise a register demand
  that admits no block at all (`cudaErrorLaunchOutOfResources`).
- **The register/occupancy exchange rate**, as a table and as a derivative:
  `warps ≈ 2048/R`, so `d(warps)/d(R) ≈ −2048/R²`. **Quadratic, not linear.**
  Below ~48 registers a granule is worth zero (warp-slot ceiling); 48–96 it is
  worth 4–8 warps; past ~120 it is worth one warp or zero.
- **The inverse is the usable form**: the largest register count that still
  achieves N blocks. At 256 threads the only numbers that exist are 128 / 80 /
  64 / 40 for 2 / 3 / 4 / 6 blocks, and everything between two steps is free.
- **Block size changes occupancy at a fixed register count** (a block places all
  its warps at once): 96 registers gives 20 warps at 32/64/128 threads and 16 at
  256/512.
- **`__launch_bounds__(T, B)`'s second argument is a register budget in
  disguise**: `roundDown(65536/(T·B), 8)`, capped at 255, verified against
  `ptxas` on 12 bounds across two files.
- **⚠️ The cap binding and `ptxas` spilling are different events**, separated on
  the Exercise 2 kernel by two whole steps of the sweep: the cap first bites at
  `(128,3)` and the first spill is at `(128,5)`, with 55 registers per thread
  surrendered **free** by rescheduling in between. All the cheap occupancy is in
  that gap. (Not predictable with a pencil — only the cap is.)
- **Theoretical vs achieved occupancy**, and **the two denominators**:
  `occ_active` (per-SM busy span, = Nsight's "Achieved Occupancy",
  `sm__warps_active.avg.pct_of_peak_sustained_active`) vs `occ_elapsed` (whole
  machine, whole kernel, `..._sustained_elapsed`). **A tail and a load imbalance
  are structurally invisible in the first.**
- **Building the achieved-occupancy counter from `clock64()` + `%smid` + three
  atomics per warp**, since `ncu` is unavailable (`ERR_NVGPUCTRPERM`); and the
  reason every quantity is per-SM: **the per-SM cycle counters are not
  synchronised** (measured 298 M cycles apart in one launch).
- **The grid-size bound on achieved occupancy**:
  `achieved ≤ theoretical · min(1, grid/(blocksPerSM·nSM))`, measured tight
  below one wave (23.9 % against a bound of 25 %).
- **⚠️ Barriers and memory stalls do NOT lower achieved occupancy.** A warp
  waiting at `__syncthreads()` or on DRAM is still resident and still counted.
  This is exactly why a kernel can be at 100 % achieved occupancy and issue
  nothing — occupancy counts warps that *exist*, not warps that are *eligible*.
  Handed to Module 20.
- **⚠️ Documented surprise: a uniform, exactly-one-wave launch at 100 %
  theoretical occupancy achieves 58–62 %** with zero tail, zero imbalance and all
  40 SM spans within 1 %. Cause isolated: the warp scheduler is greedy, so warps
  doing identical work finish far apart. Measured with a dedicated probe — 48
  warps entering within **208 cycles** and exiting across **2.55 M cycles** of a
  3.70 M-cycle kernel, a 3× spread.
- **⚠️ Are resident warps fungible across block shapes? Measured: yes, to
  within 2 %.** Module 20 handed this module a reported sawtooth in throughput
  against blocks/SM and a warning that an occupancy sweep could pick it up and
  mistake it for resource arithmetic. Tested directly here with a rotated
  14-configuration experiment (identical source, ~20 registers, no spills, no
  memory traffic, no barriers; blocks/SM forced with **dynamic shared memory** so
  registers and the instruction schedule are byte-identical; warps/SM and total
  FLOPs held fixed within each group): over four runs every configuration lands
  within **1.4–3.4 %** of the best shape in its group, with no shape
  systematically ahead — that band is the harness's repeatability, not a signal. Including
  the 160-thread (5-warp) block that loads the four schedulers 2/1/1/1, which
  measured **1.000**. **Negative result; Module 20's effect does not reproduce on
  a saturating FFMA kernel, and its proposed fit `W/(4·ceil(W/4))` evaluates to
  1.0 for both configurations it contrasts.**
- **Sweep-design rule, which stands regardless**: change one axis at a time —
  hold the block size fixed and vary the register budget, or hold the resource
  footprint fixed and vary the block size — and prefer block sizes that are a
  multiple of 128 threads so the warps divide evenly over the four schedulers.
  Hold the loop body constant across a sweep so `ptxas`'s unroll heuristic does
  not change with the sweep variable.
- **`clock64()`-derived figures must be sanity-checked against a hardware bound**
  (spec §12.13), using `nvidia-smi --query-gpu=clocks.max.sm` = **3105 MHz** on
  this part rather than the 2.04 GHz quoted in spec §12 — this module's own
  recovered clock is 2.10–2.12 GHz and would be falsely rejected by 2.04.
  `example02.cu` and `exercise03.cu` both carry the check.
- **The decision rule** (lesson §8): compute theoretical and name the limiter →
  compare with achieved → ask whether the kernel is latency-bound at all →
  consult the exchange rate → price the spill → measure.
- **`cudaOccupancyMaxPotentialBlockSize` chooses for occupancy, not
  performance**, stated plainly with M18's 18.7× divergence as the evidence.
- **Occupancy is not monotone**, confirmed independently of M18: 83.3 % is
  reproducibly slower than 100 % on the Exercise 2 kernel, and 100 % is 7–8×
  slower than 33 %.

## CUDA API / intrinsics / syntax introduced

- `cudaOccupancyMaxPotentialBlockSize` — **new here** (M1/M6/M7/M9/M12/M15 used
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` only)
- `cudaFuncGetAttributes().maxThreadsPerBlock` — new field used here
- `%smid` via inline PTX (M1 used it in exercise01; first load-bearing use in a
  measurement harness)
- `mov.u64 %0, %%clock64;` via inline `asm volatile` **with a `"memory"`
  clobber** — the idiom that stops the compiler reordering a timestamp across
  the work being timed
- `atomicMin` / `atomicMax` on `unsigned long long` (M10 introduced the family;
  the 64-bit min/max forms are new here)
- Reused, not introduced: `__launch_bounds__` (M18), `cudaFuncGetAttributes`
  (M4), `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (M1),
  `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, …)`
  (M6, named only), `__constant__` + `cudaMemcpyToSymbol` (M4), `float4` (M5),
  FNV-1a answer hashing (M11/M12 convention), `setvbuf(stdout, NULL, _IONBF, 0)`
  (M13 convention, now spec §5), `<chrono>`/`<thread>` for the operating-point
  guard (M15 convention)
- Tooling: `nvcc -Xptxas -v` read for `Used N registers`, `N bytes smem`,
  `N bytes spill stores`

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | **The arithmetic, nothing timed.** A: the four candidates, the minimum, the binding limiter and the API's answer for 24 kernels, with every limiter binding somewhere (24/24 agreement). B: the three kernels where the aggregate register model is wrong, with one worked through arithmetically. C: the exchange-rate table (registers → warps, plus the marginal warps per granule) at 128/256/512 threads, and the block-size quantisation table at a fixed register count. D: `cudaFuncGetAttributes`, `cudaOccupancyMaxPotentialBlockSize` with its honest caveat, and the `__launch_bounds__` register cap measured against `ptxas` on seven bounds. Plus a numerical check of the kernel family against a CPU reference (bit-exact). |
| `example02.cu` | **The measurement.** The `clock64()` + `%smid` instrument; a 6-grid × 2-cost-profile sweep (0.25 → 2 waves, uniform and 1..8×) in 8 rotated sweeps after a 1500 ms stream + 500 ms compute warm-up; both occupancy denominators side by side; the SM clock recovered from the busiest SM's span and sanity-checked against 3105 MHz; the tail shown moving only `occ_elapsed`; the imbalance shown moving only `occ_elapsed`; the one-wave residual isolated and explained; **part D**, the block-shape experiment (13 shapes at 3 fixed warp counts, blocks/SM forced with dynamic shared memory, rotated min-of-N) answering Module 20's hazard with a measured negative result; and an explicit section on what achieved occupancy does **not** measure. Validation in a separate pass confirms all 12 configurations compute the same answer. |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Fill-in + design (§6 types 1, 5) | 5 | Implement the four-limiter placement arithmetic; scored against the CUDA occupancy API on 18 real kernels, against FNV-1a hashes on 10 hypothetical triples, and by round trip on the inverse function | **Three.** (a) TODO 1 needs *two* quantisations and Module 18 only supplies one; the aggregate model is right on 127 of 137 kernels and wrong on `(98,0,64)`, `state<30>` at 64 threads and `state<72>` at 96 — and the fully naive `65536/(regs*threads)` is additionally wrong on `(84,0,256)` and `(41,0,256)`. (b) TODO 2's warp-slot limit is **not** `1536/threads`; the 100-thread row gives 12 where that expression gives 15. (c) TODO 5 cannot be inverted algebraically (two floors and a round-up) and the round-trip scoring rejects any answer that is merely sufficient; the intended solution is a 255-step scan, which readers resist writing. Bonus: the 32-thread row is block-slot-limited at 24 blocks = 50 % occupancy, which nobody predicts. |
| `exercise02.cu` | Optimization + prediction + design (§6 types 2, 4, 6) | 5 | One recursive-cascade source compiled nine times under `__launch_bounds__(128, n)`; reader predicts the shape of the curve, chooses the operating point, and names a selection rule the harness evaluates against the measured resource table | **Three.** (a) TODO 2 asks where the register cap *binds*, not where `ptxas` *spills*, and the two differ by two full steps — 55 registers per thread are surrendered for free in between. A reader who conflates them answers 5 instead of 3 and never tries the two configurations where the win is. (b) The optimum is **interior**: 100 % occupancy is 7.2–8.2× slower and the compiler's own unconstrained choice is 1.1–1.4× slower. (c) TODO 5's rule 4, "highest occupancy with a spill under 100 bytes", is a *reasonable* rule that encodes Module 18's correct finding that small spills can win — and it is wrong here, because 72 bytes of spilled filter state is not 80 bytes of spilled addressing temporaries. `localSizeBytes` cannot distinguish them; only the SASS can. Also: 83.3 % occupancy is reproducibly slower than 100 %. |
| `exercise03.cu` | Design ×2 + prediction + instrument build (§6 types 2, 5, 6) | 5 | Build the achieved-occupancy counter from `clock64()`/`%smid`, implement both denominators, bound achieved occupancy from the grid size, and re-size a launch that wastes three quarters of the machine at 100 % theoretical occupancy | **Three.** (a) The two denominators are the exercise: at a fixed grid, load imbalance moves `occ_active` by **+3.3 points (upward)** and `occ_elapsed` by **−14.3**, so the metric every tutorial quotes is blind to it. (b) TODO 4's instinctive answer — exactly one wave, which Modules 1 and 3 both point at — fails both gates, because the only free load balancing on this hardware is the distributor refilling an SM, and at exactly one wave there is nothing left to place; one wave is measurably **slower than half a wave** on a ragged workload. (c) The one-wave uniform launch reaches 62 % of a theoretical 100 % with no tail and no imbalance, which looks like an instrument bug and is the greedy warp scheduler. Also: TODO 1 has four distinct ways to be wrong that still run (32 reporters per warp → occupancy over 100 %; swapped min/max → unsigned underflow → ~1e-10; accumulating raw timestamps; comparing timestamps across SMs). |

Scoring: Ex1 `SCORE: n/10` (3 real kernels + 2 hypothetical + 2 limiter + 1
occupancy + 2 round trip); Ex2 `n/10` (2 cap formula + 1 cap-binds + 2 + 1
predictions + 2 operating point + 2 rule); Ex3 `n/10` (2 instrument + 1 elapsed
+ 2 model + 1 undersized + 2 gates + 1 + 1 predictions). `OVERALL: PASS`
requires full marks in all three.

## Measured results recorded in this module

| Quantity | Measured |
|---|---|
| **Register placement model, verified** | `4 * floor(16384 / (roundUp(R,8)*32)) / warpsPerBlock` — **exact on 137/137 kernels** |
| **Aggregate (M18) register model** | wrong on **10/137**: e.g. 47 regs @64 thr (21 vs 20), 92 @96 (7 vs 6), 78 @160 (5 vs 4), 48 @48/192/200 thr |
| Fully naive `65536/(regs·threads)` | additionally wrong on 84 regs @256 (3 vs 2) and 41 @256 (6 vs 5) |
| blocks/SM vs shared bytes @256 thr | 6 ≤12288 B, **5 @16384**, 3 @25600, 2 @49152 — reproduces M6 exactly |
| 100-thread block | **12 blocks/SM**, not the 15 that `1536/threads` gives; 100 % occupancy with 300 thread slots empty |
| 32-thread block | 24 blocks (block-slot cap) = **50 % occupancy ceiling** |
| 1024-thread block | 1 block = 32 warps = **66.7 %** (M3's deferred figure); unlaunchable above 64 registers/thread |
| **Exchange rate @128 thr** | 48 regs→40 warps, 56→36, 64→32, 72→28, 80→24, 88→20, 96→20, 104→16, 128→16, 136→12, 168→12 |
| Marginal warps per granule | 8 at R=48, 4 at R=56–88, 0 at R=96, 4 at R=104, 0 at 112–128, 4 at 136, 0 past 144 |
| Block-size quantisation @96 regs | 32/64/128 thr → **20 warps**; 256/512 thr → **16 warps** |
| `maxRegistersFor` @256 thr | 2 blocks→128 regs, 3→80, 4→64, 6→40 |
| **`__launch_bounds__` register cap** | `roundDown(65536/(T·B), 8)`, ≤255 — exact on every bound where `ptxas` was forced to it |
| Ex2 kernel, unconstrained | **177 registers**, 0 spill, 2 blocks/SM, 16.7 % occupancy |
| Ex2 cap binds / first spill | `(128,3)` / `(128,5)` — **55 registers per thread given up free** in between |
| **Ex2 sweep (Gelem/s)** | 9.22 / 9.19 / 11.44 / **12.62** / 8.83 / 3.64 / 2.18 / 1.47 / 1.55 at B = 1/2/3/4/5/6/8/10/12 |
| **Ex2: 100 % occupancy vs best** | **7.15 – 8.15× slower** over 4 clean runs (33.3 % occupancy wins) |
| Ex2: unconstrained vs best | **1.09 – 1.37× slower** |
| Ex2: 83.3 % vs 100 % occupancy | 1.47 vs 1.55 Gelem/s — **non-monotone**, reproducibly |
| Ex2 spill ladder | 0/0/0/0/72/136/200/264/296 B at B = 1…12 |
| **Achieved occupancy, uniform, 1 wave, theoretical 100 %** | **61.9 % active / 61.9 % elapsed** — the documented surprise |
| Warp entry spread / exit spread, one SM, identical work | **208 cycles / 2 546 000 cycles** (span 3 697 000) |
| Cross-SM `clock64()` offset in one launch | up to **298 million cycles** — counters are not synchronised |
| Achieved vs grid (uniform) | 0.25 waves 23.9 %, 0.5 → 36.6 %, 1 → 61.9 %, 1.5 → 77.4 %, 2 → 75.0 % |
| Tail (240 → 241 blocks) | `occ_active` 61.9 → 62.3 %, `occ_elapsed` 61.9 → **57.9 %**, wall time +7.5 % |
| Imbalance (1..8× cost, 1 wave) | `occ_active` **+3.4 points**, `occ_elapsed` **−14.4 points**, wall time +53 % |
| Imbalanced: 0.5 waves vs 1 wave | **1.765 vs 1.908 ms** — one wave is slower than half a wave |
| Ex3 `chooseGrid` (2 waves) vs quarter-wave baseline | **1.39 – 1.53×**, `occ_elapsed` 65–77 % |
| One wave on the same gates | 1.18 – 1.29×, `occ_elapsed` 47.3 – 47.7 % — fails both |
| Recovered SM clock | **2.10 – 2.12 GHz** (`cudaDevAttrClockRate` says 1.545 — 27 % low) |
| FFMA ceiling probe during these runs | 16916 – 17297 GFLOP/s |
| **Block shape at fixed warps/SM, 13 shapes, 4 runs** | every shape **0.966–1.000** of the best shape at the same warp count; no systematic ordering |
| …including the 5-warp (160-thread) block, 2/1/1/1 per scheduler | **0.994–1.000** |
| **⚠️ min-of-N vs median on a 1.7 s rotated sweep** | min-of-N reported the two lowest-indexed shapes **10 % fast**; the median reports them equal. Rotation removes positional bias from the mean, not from the minimum |
| Max SM clock for sanity bounds | **3105 MHz** (`nvidia-smi --query-gpu=clocks.max.sm`); recovered clock here 2.10–2.12 GHz |

## Assumed from earlier modules

- **M1**: SM anatomy — 4 processing blocks, one warp scheduler and a
  **16384-register slice** each, 48 warp slots, 24 block slots; the block
  placement gate as four simultaneous conditions; blocks indivisible and
  non-migrating; **waves and tail effects**; Little's Law; and the measured
  latency-hiding curve (linear to 3 warps, knee at 6, flat at 4.87× by 24).
  **Used, not re-taught.**
- **M2/M3**: launch syntax, `CHECK`, `cudaEvent_t` timing, grid-stride loops,
  ceil-divide grid sizing, the linearisation rule (which makes `threadIdx.x & 31`
  a valid lane test), and M3's 66.7 % ceiling for 1024-thread blocks, deferred
  here and now paid.
- **M4**: **local memory is DRAM**, so a spill is a DRAM access; `numRegs` /
  `localSizeBytes` via `cudaFuncGetAttributes`; `-Xptxas -v`; `__constant__`
  broadcast.
- **M6**: shared memory capacity → blocks/SM with the **1024 B driver reserve**
  and **128 B granularity**; the 49152/101376 B per-block limits and the opt-in;
  the measured blocks/SM table, reproduced here exactly.
- **M8**: a partial warp occupies a whole warp slot — the fact that makes
  `48/ceil(T/32)` right and `1536/T` wrong.
- **M10**: atomics, used here only as counters (`atomicMin`/`Max`/`Add` on one
  address per SM), with the contention argument cited for why 3 atomics per warp
  over 40 addresses is free.
- **M11**: ILP/MLP as a substitute for occupancy (2× at 1 block/SM, 1.00× at 8),
  cited in the decision rule as the test for "is this kernel latency-bound".
- **M16**: the FP32 ceiling measurement and the rejection of
  `cudaDevAttrClockRate`; the stream-then-compute warm-up.
- **M17**: registers as the limiter nobody expects (44 vs 40 registers, 3 vs 2
  blocks, identical threads and shared memory).
- **M18**: `__launch_bounds__` as a measurement instrument; the 8-register
  granule; **the 18.7× occupancy result and the non-monotonicity**, which this
  module explains and reproduces on a second, non-GEMM kernel; the
  spill-cliff-is-not-the-first-byte finding, extended here with a
  counterexample.
- **Spec §12 throughout**: 1500 ms stream + 500 ms compute warm-up, rotated
  back-to-back sweeps with `SWEEPS >= NCFG`, auto-scaled iteration counts,
  min-of-N, validation in a separate untimed pass, operating-point guards on
  both scoring harnesses, ratios as the stable quantity, compute-bound payloads
  for occupancy work (rule 8).

## Forward-reference debts PAID here

- **M6 → M19** (recorded in the index as *"M6's occupancy formula is incomplete
  — it omits registers … M19 owes the four-limiter version"*): paid in full, and
  paid with a correction to the *form* of the register term that Module 18 did
  not have. Verified against the occupancy API on 137 kernels.
- **M17 → M19**: *"registers as the fourth placement-gate limiter"*. Paid; M17's
  44-vs-40-register observation is reproduced as the general rule, with the
  granule and slice arithmetic that explains it.
- **M18 → M19**: *"the 14–18.7× 100 %-occupancy result, the non-monotonicity,
  and the by-hand four-limit occupancy arithmetic"*. All three paid; the headline
  result is reproduced independently on a non-GEMM kernel at **7.2–8.2×** and
  the non-monotonicity at a different point of the curve (83.3 % slower than
  100 %).
- **M1 → M19**: *"occupancy arithmetic (registers/shared memory vs resident
  warps)"*. Paid, including the part M1 stated and nobody used — the
  **16384-register slice** is what makes the register term correct.
- **M3 → M19**: *"register and shared-memory pressure added to the
  1536-threads/SM arithmetic; the 66.7 % ceiling of 1024-thread blocks;
  'maximum occupancy ≠ maximum speed'"*. All three paid.
- **M2 → M19**: the one-warp-per-SM latency-bound kernel from M2 ex3's
  performance discussion. Paid as the §8 decision rule's step 3.
- **M4 → M19**: *"register count caps resident warps; `65536 / (regs × 32)`"*.
  Paid **and corrected**: that expression is the aggregate model and is wrong on
  7 % of kernels.
- **M5, M10, M13, M14 → M19**: each named M19 for a piece of occupancy
  arithmetic they did by hand (vectorisation vs occupancy; privatisation
  footprint vs resident blocks; padding vs blocks/SM; `R × nBins × 4 B` vs
  blocks/SM). All are instances of the four-limiter formula and are cited as
  such in the lesson.
- **M7 → M19**: *"Exercise 2 computes blocks-per-SM by hand and checks it
  against the API; M19 owns occupancy properly"*. Paid.
- **M9 → M19** (implicit): M9's co-residency figure
  (`cudaOccupancyMaxActiveBlocksPerMultiprocessor × multiProcessorCount` = 240)
  is the same calculation; the lesson names it as the wave size.

## Forward references made

- **Module 20 (latency hiding / ILP vs occupancy)** — named six times and
  deliberately: it owns the warp-scheduler treatment, the **eligible-vs-resident
  distinction**, the stall-reason taxonomy, and the measurement that decides
  whether a kernel is latency-bound at all. This module states the distinction
  (a stalled warp is still resident, so occupancy cannot see a stall) and stops
  there.
- **Module 21 (roofline)** — named; no roofline is built here.
- **Module 23 (Nsight Compute)** — named as the owner of
  `sm__warps_active.avg.pct_of_peak_sustained_{active,elapsed}`,
  `smsp__warps_eligible`, `smsp__issue_active` and the stall-reason counters.
  **`ncu` remains unavailable (`ERR_NVGPUCTRPERM`); no counter output is quoted
  anywhere.** Every number in this module was constructed and measured directly.
- **Module 29 (cooperative groups)** — named in passing: a cooperative launch's
  grid is capped at `occupancy × nSM`, which is this module's arithmetic used as
  a correctness constraint rather than a performance one.
- **Modules 33–34 / 43** — not named here; occupancy on Tensor-Core kernels is
  the same arithmetic with a different register-per-FLOP ratio, and M18 already
  handed that forward.

## Constraints observed

- **No latency-hiding mechanics** (M20) and **no roofline** (M21). The lesson
  cites Module 1's measured curve as a given and never re-derives it.
- **No GEMM.** Module 18's result is quoted as the motivating measurement and
  reproduced on an unrelated kernel; no GEMM code appears in this module.
- Themes: the **recursive cascade / sequential-state filter** (Examples 1 and 2,
  Exercise 2) and the **ragged chunked workload** (Example 2, Exercise 3) are
  new. None of §6's used themes is repeated; sorting, sparse formats, n-body and
  large-radius convolution remain reserved.
- No sm_90+ features. `%smid` and `clock64()` are labelled
  ARCHITECTURE-SPECIFIC where used for anything but timing.
- Nothing under `module19/` reveals an answer: Exercise 1's hypothetical block
  counts, limiter sequence and occupancy sequence are FNV-1a hashed; its
  real-kernel answers are scored against the CUDA API; Exercise 2's and 3's
  answers are scored against measurements taken in the same run; and the
  reference limiter formula was removed from the Exercise 1 harness after a
  first draft placed it in `main()`.
- Every timed figure follows spec §12, including the **1500 ms + 500 ms**
  two-stage warm-up, rotation with `SWEEPS >= NCFG`, auto-scaled iteration
  counts, min-of-N, and an **operating-point guard on both scoring harnesses**.
- No binaries committed.

## Known issues / honesty notes

- **`ncu` unavailable** (`ERR_NVGPUCTRPERM`), per the standing rule: documented
  as theory, never worked around. The metric names and the exact distinction
  between the two achieved-occupancy counters are given; the counters themselves
  are reconstructed from `clock64()` and `%smid` and the files say so.
- **A perfectly uniform, exactly-one-wave launch achieves 58–62 % of its
  theoretical occupancy**, and this is reported as a finding rather than
  smoothed away. It was chased down with a dedicated probe (entry spread 208
  cycles, exit spread 2.55 M cycles) and attributed to greedy warp scheduling.
  Consequence adopted throughout: compare configurations against each other,
  never against 100 %.
- **An early Exercise 3 design was abandoned after measurement.** It had the
  reader build a persistent-block kernel with an `atomicAdd` work queue to fix
  load imbalance; measured, it was no better than simply launching two waves,
  because **the GigaThread engine already is a dynamic scheduler at block
  granularity**. The exercise now asks for the grid size, which is the move that
  actually works, and the solution notes record the negative result.
- **An early verification run of `exercise03_solution` scored 8/10 on the
  correct answer.** Every row was 2.5× slow at an unchanged 2.10 GHz SM clock —
  i.e. genuinely more cycles, not a lower clock — while the FFMA ceiling probe
  reported a healthy 17143 GFLOP/s. **A small compute probe is not a valid
  operating-point guard for a full-machine kernel**: it draws too little power
  and uses too few SMs. The file now guards on its own balanced two-wave
  reference launch (1.24–1.26 ms healthy, 3.13 ms contended, cut at 1.80 ms)
  with idle-and-rewarm per spec §12.5b. Spec §12.5c applies: verify these one at
  a time.
- **`maxRegistersFor` is solved by a 255-step scan, not in closed form.** That
  is the intended answer, not a shortcut; the expression contains two floors and
  a round-up and has no exact closed-form inverse.
- **Exercise 2 scores correctly even from a badly throttled operating point.**
  One verification run measured every row **2.4× slow** (best 5.25 Gelem/s
  against a healthy 10.2–12.8) while the FFMA probe reported a healthy 16842
  GFLOP/s, and still scored 10/10 — because every scored quantity is a ratio
  inside one rotated sweep. Exercise 3's gates compare two *different launches*
  and are not protected that way, which is why only Exercise 3 needed the
  kernel-based operating-point guard.
- **Absolute throughput moves 10–20 % with thermal state.** Every claim in the
  lesson and the solution notes is a ratio, a bucket, or is quoted with its
  range; the Exercise 2 ordering of `B = 3` and `B = 4` alternates between runs
  (measured over seven clean runs: `B = 4` fastest in six, `B = 3` in one), and
  the top four rows span 73–100 % of their own run's best, so the
  operating-point gate is **20 %** wide and is backed by a self-consistency
  check against the TODO 4 prediction. The first spilling row never exceeds
  72 %, so the gate sits in a 13-point gap.
- **`occ_elapsed` is slightly optimistic by construction.** The SM clock is
  recovered as `maxSpan / elapsed_ms`, and the busiest SM's span is shorter than
  the event-timed kernel by the launch overhead, so the recovered clock is a
  lower bound and the elapsed denominator is correspondingly small. The files
  print this caveat.

## Cross-module observations for the index

1. **⚠️ §6b correction: the register allocation model needs the slice term.**
   The index records *"Register allocation granule: 8 registers per thread,
   allocated per warp. A per-thread model looks nearly right and is wrong."*
   That is correct and incomplete. The per-SM register file is **four
   16384-register slices** and a warp draws entirely from one, so
   `blocksByRegs = 4·floor(16384 / (roundUp(R,8)·32)) / warpsPerBlock`, which is
   **strictly tighter** than `65536 / (roundUp(R,8)·32·warpsPerBlock)`. Verified
   on 137 kernels: slice model 137/137, aggregate model 127/137. §6b should
   carry the corrected form.
2. **§6b should gain the four-limiter formula in full**, including that the warp
   limit is `48/ceil(T/32)` and not `1536/T` (they differ for every block size
   that is not a multiple of 32), and that the 24-block cap puts a hard 50 %
   ceiling on 32-thread blocks.
3. **§6b should gain the exchange rate**: `warps ≈ 2048/R`, marginal value
   `≈ 2048/R²` warps per register — zero below ~48 registers, 4–8 warps per
   granule from 48 to 96, one or zero past 120. And the `__launch_bounds__`
   register cap in closed form, `roundDown(65536/(T·B), 8)`.
4. **New §6b row: achieved occupancy has two denominators and the common one is
   blind to tails and imbalance.** Measured: a tail moves `occ_active` by 0.4
   points and `occ_elapsed` by 4; a 1..8× imbalance moves `occ_active` **up** by
   3.4 points and `occ_elapsed` down by 14.4.
5. **New §6b row: a uniform one-wave launch at 100 % theoretical occupancy
   achieves 58–62 %** because of greedy warp scheduling; 60–78 % is the
   practical ceiling on this part. Any future module quoting achieved occupancy
   should compare against that, not against 100 %.
6. **M18's 100 %-occupancy result replicates off GEMM.** 7.2–8.2× on a
   sequential-state filter at 128 threads, against M18's 14.1–14.6× at 128 and
   18.7× at 256. The sign and the mechanism transfer; the magnitude is
   kernel-specific and scales with how much of the inner loop's live set spills.
7. **Spec §12.5b should gain a qualifier: the operating-point probe must stress
   the same fraction of the machine as the kernel being scored.** A 480-block
   FFMA probe reported 17143 GFLOP/s on a run where a full-machine kernel was
   taking 2.5× its usual cycles. The probe must be a full-machine launch, and
   the cheapest valid one is a reference configuration of the kernel under test.
8. **⚠️ Spec §12.3's min-of-N has a failure mode on long rotated sweeps.** If a
   sweep runs long enough to heat the part monotonically, and every
   configuration visits every position of the rotation exactly once, then each
   configuration's MINIMUM is its earliest-position sample — so the minimum
   re-introduces exactly the ordering bias rotation exists to remove. Measured
   in `example02.cu` part D: 13 configurations × 13 sweeps × 10 ms segments,
   min-of-N reported the two lowest-indexed block shapes **10 % faster** than
   three others at the same 48 warps/SM; the median over the same samples
   reports all five equal, reproducibly. **Rotation removes positional bias from
   the mean, not from the minimum.** Spec §12.3 should carry the qualifier, and
   a long sweep should say which statistic it took.
9. **⚠️ Module 20's block-shape sawtooth does not reproduce here.** M20 reports
   that the same warps/SM gives 1.008 vs 0.625 instructions/cycle/scheduler
   depending on block packaging, and recommends restricting occupancy sweeps to
   blocks/SM ∈ {1,2,3,4,8,12}. A rotated, min-of-N, fixed-warps/SM, fixed-FLOP
   experiment (median over the rotation, not min-of-N -- see observation 8) on a
   saturating FFMA kernel puts all 13 shapes **within 1.4–3.4 %
   of the best shape at the same warp count** over four runs, with no systematic
   ordering, including the 5-warp block M20 predicts should reach 0.56 of
   ceiling. Both can be true — M20's harness derives its figure from `clock64()`,
   which M20 itself measured returning 5.7× a hardware maximum at 12 blocks/SM —
   but the index should record the effect as **harness-dependent and unconfirmed**
   rather than as a property of the hardware. M20's *sweep-design advice* is
   adopted here unconditionally and costs nothing.
10. **Spec §12's SM-clock figure (2.04 GHz) is too low to use as a sanity bound.**
   This module recovers 2.10–2.12 GHz from `clock64()` on a full-machine kernel,
   independently of M16's 2.12 GHz single-warp figure. The right bound is
   `nvidia-smi --query-gpu=clocks.max.sm` = **3105 MHz**; 2.04 produces false
   rejections.
11. **`atomicMin`/`atomicMax` on `unsigned long long`** are the natural way to
   reduce per-SM timestamps and are not used anywhere else in the course.
   Worth recording alongside M10's atomic inventory.
