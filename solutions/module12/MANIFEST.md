# Module 12 manifest — Reduction

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module12/lesson.md`, `module12/example0{1,2}.cu`,
`module12/exercise0{1,2,3}.cu`.
Solutions: `solutions/module12/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module12/check_your_understanding.md`,
`solutions/module12/MANIFEST.md`.

All `.cu` verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2, V13.2.51, RTX 3500
Ada), warning-clean. `example02.cu` additionally needs
`-std=c++17 -Xcompiler /Zc:preprocessor` for `<cub/cub.cuh>` (same pair Module 9
needed for `<cuda/atomic>`). All three solutions print `OVERALL: PASS`
(10/10, 5/5, 8/8). All three shipped exercises compile with TODOs blank and exit
gracefully (`Set TODO 1 first.`, `Set TODO 4 first.`, `Set TODO 1a and TODO 4
first.`). No binaries committed.

---

## Concepts taught

- **Reduction as a monoid homomorphism**: associativity is the licence to
  re-bracket and is the *only* thing the hardware gives you; an identity is
  needed so a thread with no work has something to contribute; **commutativity
  is a separate property that is not required**, and every free choice it hides
  becomes a correctness decision without it. **PORTABLE CUDA CONCEPT.**
- **Work vs depth**: `n-1` operations either way; depth `n-1` sequential vs
  `ceil(log2 n)` as a tree.
- **Reduction is bandwidth-bound**, with the arithmetic done explicitly:
  0.25 FLOP/byte offered against ~42 FLOP/byte needed to balance this GPU, a
  factor of ~170. The floor is `4n / BW`; the addition is free, not cheap; the
  only honest scoreboard is **% of a streaming ceiling measured in the same
  sweep**, not % of 432 GB/s.
- **The six-rung ladder**, each rung a previously taught defect:
  - v1 interleaved `tid % (2*s)` — maximal divergence (M8), 24.6% of ceiling
  - v2 contiguous index `2*s*tid` — divergence removed, stride-`2s` bank
    conflicts introduced (`D = gcd(k,32)`, M7), 35.1%
  - v3 sequential addressing `s = blockDim/2; s > 0; s >>= 1` — conflict-free,
    38.1%
  - v4 first add during load, half the grid — 75.7%, **the largest single step
    (1.94–1.99×)**, and the one the textbook ordering puts fourth
  - v5 warp tail via `__shfl_down_sync` — 99.5%, at the wall
  - v6 grid-stride register accumulation + template-parameter unrolling +
    machine-sized grid — 99.3% at 256 MiB, **1.51× over v5 at 4 MiB**
- **The measured lesson of the ladder**: the two famous optimizations
  (divergence, bank conflicts) are worth 1.45× combined on a kernel that was
  wasting 75% of the memory system, because neither touches the binding
  constraint. You cannot rank optimizations without knowing which resource is
  saturated.
- **Memory-level parallelism as the real content of v4**: two *independent*
  loads per thread instead of one; Little's Law (M1) as the formal statement.
- **The barrier that becomes unnecessary at v5**, and the *correct* reason: the
  reader and writer of `sdata[tid]` after the `s == 32` step are the same
  thread, so neither of M9's guarantees applies. Explicitly **not** "warps are
  in lockstep", which M8 killed.
- **The legacy `volatile __shared__` warp tail as a bug**, never as a technique
  (M9's debt): pre-Volta correctness rested on one PC per warp; `volatile` was
  only stopping register caching and never supplied ordering; ITS removed the
  lockstep. Shipped, run, shown to **pass on this GPU**, with M8's
  dangerous-middle-case framing and the SASS (six `LDS`/`STS` pairs, no
  `WARPSYNC`, no `BAR.SYNC`) as the evidence the source hides.
- **`__shfl_down_sync` semantics**: the mask *creates* convergence on sm_70+ and
  must name the lanes the algorithm requires (not `__activemask()`); results are
  valid **only in the lower lanes**; `_sync` primitives carry their own
  synchronization so no `__syncwarp` is needed between steps.
- **Three multi-block finalization strategies, compared honestly**:
  (a) second kernel launch, (b) `atomicAdd` per block, (c) last-block-done flag
  with `__threadfence()` + `atomicAdd` ticket. **Measured within 0.5% of each
  other and within 1.3% of the streaming ceiling** — the choice is on
  determinism, launch count, memory and scaling, not speed.
- **Why strategy (c) is deterministic even though which block finishes last is
  unspecified**: determinism comes from the *summation order* being fixed, not
  from the *schedule* being fixed.
- **Float `atomicAdd` non-determinism inside a real algorithm** (M10's debt):
  5–9 distinct bit patterns in 10 identical runs.
- **The half of the determinism story M10 could not tell**: a fixed-order tree is
  reproducible only for a **fixed decomposition**. Same kernel, grids
  120/240/480, three different answers, no atomics anywhere. Determinism is a
  property of a kernel *plus a launch configuration*.
- **Two routes to reproducibility and what each costs**: (1) decouple the
  decomposition from the grid — fixed `NCHUNK`, grid only decides traversal;
  measured cost **1.00–1.03×**, i.e. free, *because the kernel is
  bandwidth-bound*; (2) 64-bit fixed-point accumulation — order-independent by
  construction, costs dynamic range, requires deriving an overflow bound and a
  precision bound on the scale.
- **Floating-point accuracy: the tree is MORE accurate**, `O(log n)·eps` vs
  `O(n)·eps`, plus the non-gradual failure mode of sequential accumulation.
  Measured: 2^26 copies of `1.0f` → sequential float **75% low** (saturates at
  exactly 2^24), tree exact; wide-dynamic-range data → sequential 9.0% low,
  Kahan 1.6e-8, tree 6.1e-7. "Reach for the tree before you reach for `double`",
  especially on Ada where FP64 runs at 1/64 rate (M7).
- **`__reduce_add_sync` (sm_80+) is a real hardware instruction**: verified
  `REDUX.SUM UR6, R2` in SASS on sm_89, one instruction replacing five
  `SHFL.DOWN` + five adds, destination a **uniform register**.
  **ARCHITECTURE-SPECIFIC OPTIMIZATION.**
- **`cg::reduce` lowers to `REDUX` by itself** for integer types on sm_80+, and
  to five `SHFL.BFLY` for `float` (no hardware float reduction). Cooperative
  groups is the only one of the three spellings that picks the best available
  instruction.
- **The timer cannot see any of it**: all three warp-tail spellings measured
  within 0.03% on 2^26 elements. Five instructions against ~2^19 loads per
  block, in a kernel waiting on DRAM.
- **CUB as what you would actually ship**: `cub::BlockReduce<T,BS>` with its
  `TempStorage`, `cub::DeviceReduce::Sum` with the two-call temp-storage
  protocol (5119 bytes here). Measured within 0.5% of the hand-written v6 —
  the expected result, and the point of the ceiling argument.
- **Non-commutative reductions** (Exercise 2): the prefix/suffix/best/length
  monoid for "longest run above a threshold"; why `len` must be carried; why the
  identity must be checked in both directions; the **order of the shuffle
  offsets** (1,2,4,8,16 upward, not 16,8,4,2,1 downward) as a correctness
  requirement that `+` hides; the tile-of-32 decomposition that is
  simultaneously order-preserving and coalesced.
- **The missing-ragged-tail bug class** (Exercise 3): an all-or-nothing pair
  guard drops the unpartnered elements; the loss is a function of
  `n mod (2·blockDim)`, is **exactly zero at `n = 2^25`** and one element at
  `n = 2^25 + 1`; **all four `compute-sanitizer` tools report nothing**, because
  the tools check what you did and never what you failed to do.
- **Benchmarking finding added by this module** (extends spec §12): a 400 ms
  duration-based warm-up ramps the SM clock but **not the memory P-state**. The
  same `example01.cu` streaming ceiling measured 372–373 GB/s after a 400 ms
  warm-up and **410.7 GB/s (95.1% of 432)** after 1500 ms — the highest
  streaming figure recorded in this course. All timed files in this module use a
  1500 ms warm-up; `% of ceiling` is the reported stable quantity.

## CUDA API / intrinsics / syntax introduced

- `__shfl_down_sync(mask, var, delta)` — used as an algorithm primitive for the
  first time (M8 used it only as a measured misuse case)
- `__reduce_add_sync(mask, v)` — sm_80+ hardware warp reduction (integer only);
  family `__reduce_{add,min,max,and,or,xor}_sync` named
- `cooperative_groups::reduce(tile, val, cg::plus<T>())`,
  `cg::plus<T>`, `#include <cooperative_groups/reduce.h>`
- `cub::BlockReduce<T, BLOCK>::TempStorage`, `.Sum(v)`;
  `cub::DeviceReduce::Sum(d_temp, tempBytes, in, out, n)` and the two-call
  temp-storage protocol; `#include <cub/cub.cuh>` (lives in
  `include/cccl/cub` in CUDA 13.2, on the default include path)
- `atomicAdd(unsigned long long*, unsigned long long)` as the deterministic
  accumulator
- `__threadfence()` + `atomicAdd` ticket + a shared `bool` as the
  last-block-done idiom (all three components from M9/M10; the *pattern* is new)
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` used to size a grid
  (M6/M9 introduced it for occupancy reporting)
- `llrint()` in device code for fixed-point conversion
- `ldexpf()` in device code (dyadic test data)
- Template-parameter `__global__` with compile-time-resolved tree bounds
  (`if (BLOCK >= 512) …`) — M4's dispatch trick, canonical use
- SASS vocabulary: `REDUX.SUM`, `SHFL.DOWN`, `SHFL.BFLY`, uniform registers
  (`UR6`), `LDS`/`STS`, `BAR.SYNC.DEFER_BLOCKING`
- `cuobjdump -sass` on an object file, filtered with `findstr`
- Build flags: `-std=c++17 -Xcompiler /Zc:preprocessor` for CCCL headers

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | The full six-rung ladder plus a pure-streaming ceiling, all seven timed in one rotated sweep with auto-scaled iteration counts and min-of-4; validation in a separate pass against a double reference. **Part B** times the finalization pass nobody times and shows v6's real win (240 partials vs 262,144: total 0.7355 ms vs v5's 0.7464). **Part C** re-runs all six on 4 MiB of L2-resident data, where v6 is 1.51× v5 and 3.80× v1, explicitly labelled as not a DRAM bandwidth number. Also: how to write a ceiling kernel (four independent accumulators, a store the compiler cannot prove dead). |
| `example02.cu` | **A** three finalization strategies + ceiling in one rotated sweep. **B** determinism: 10 runs × 3 strategies, then the same strategy at 3 grid sizes. **C** accuracy: two datasets × {CPU sequential float, CPU Kahan, GPU tree} against an exact reference. **D** three warp-tail spellings over `unsigned`, timed (identical) and disassembled (different). **E** CUB `BlockReduce` and `DeviceReduce` against the hand-written version. **F** the legacy `volatile` tail, run and shown to pass, with the M8 framing. |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Optimization (§6 type 4) + performance reasoning (type 6) + design | 5 | Build versions 2–6 of the ladder from version 1; four predictions committed before compiling; harness times all six plus a ceiling in one rotated sweep and reports % of ceiling | **TODO 1(a)**: the biggest step is v4 (first add during load), not v2 (remove divergence) — a reader who has learned the ladder's *story* rather than its physics answers v2. **TODO 1(d)**: the last barrier of v5's shared loop IS removable, but the only valid justification is that the reader and writer are the same thread; "warps are in lockstep" gives the right answer for a reason M8 destroyed. **TODO 3**: copying v5's `s > 32` bound into v3 silently discards 63/64 of every block's data. **TODO 4**: the all-or-nothing pair guard `if (i+BS < n) v = in[i] + in[i+BS]` validates perfectly at `n = 2^26` and is Exercise 3's planted bug. **TODO 5**: `__activemask()` as the shuffle mask; writing from lane 31 instead of lane 0; `chooseGridV6()` returning `n/(BS*2)` (correct, exactly as fast as v5, and rejected by a 4096-block gate). |
| `exercise02.cu` | Design (§6 type 5) + fill-in + prediction | 4 | Segmented reduction over 100,000 ragged rows (45.4 M elements, lengths 8…200,000) with a **non-commutative** operator: per-row sum and longest run above threshold | **TODO 2 is the exercise**: the textbook offset order 16,8,4,2,1 hands lane 0 the spans in the order 0,16,8,24,… — correct for a sum, wrong here. Going 1,2,4,8,16 is order-preserving. Symptom: 84,249/100,000 rows with a wrong `best` and **0 with a wrong `sum`**; rows shorter than 32 are all correct. Second trap, **TODO 1**: padding a partial tile with a zero-valued *element* instead of the monoid identity — and `THR = -0.5f` is chosen so that `0.0f` is *above* threshold, making the pad extend the trailing run. Third, **TODO 3**: the order-preserving decomposition and the coalesced decomposition look mutually exclusive; the tile-of-32 arrangement is both. The harness runs a deliberately-strided probe (42,860/100,000 rows wrong) to score the prediction. |
| `exercise03.cu` | Debugging (§6 type 3) + design ×2 + prediction | 4 | `n = 2^25 + 1`. Part 1: a reduction 5.6% low that all four sanitizer tools declare clean and that is **exact at `n = 2^25`**. Part 2: build two bit-reproducible reductions with different trade-offs | The diagnosis list contains **two true statements about CUDA that are not the cause**: "float addition is not associative" (true, subject of part 2, and four orders of magnitude too small to explain 5.6%) and "bank conflicts corrupt the tree" (a category error — conflicts are replays with no correctness component, M7). The 5.6% comes from **one** lost element that happens to hold 1e6; the loss is `{i ∈ [n-256,n) : i mod 512 ∈ [0,256)}`, which is empty at `n = 2^25`. **TODO 2**: the tempting answer is to ignore `gridHint`; the harness checks the return value, so the reader must keep the grid free and fix the decomposition instead — and then discovers it costs 1.00×, which contradicts the intuition that the constraint must be expensive. **TODO 3**: the scale has a wrong answer on *each* side (overflow above 5.4e8, quantization below 9.4e4) and both failures are perfectly reproducible, which is the closing point: a reproducible wrong answer is still a wrong answer. |

Scoring: Ex1 `SCORE: n/10` (6 versions + 4 predictions), Ex2 `SCORE: n/5`
(best exact, sum within 1e-4, ≥3.0× speed gate, 2 predictions), Ex3
`SCORE: n/8`. `OVERALL: PASS` requires full marks in all three.

## Measured results recorded in this module

| Quantity | Value |
|---|---|
| Streaming ceiling, 400 ms warm-up | 372.2 – 373.1 GB/s (86.2–86.4% of 432) |
| Streaming ceiling, 1500 ms warm-up | **410.5 – 410.7 GB/s (95.0–95.1% of 432)** |
| Ladder v1 (interleaved, `%`) | 24.6 – 29.5% of ceiling |
| Ladder v2 (contiguous index) | 35.1 – 36.2% |
| Ladder v3 (sequential addressing) | 38.1 – 42.8% |
| Ladder v4 (first add during load) | 75.3 – 81.6% |
| Ladder v5 (`__shfl_down_sync` tail) | 96.9 – 99.7% |
| Ladder v6 (grid-stride + unrolled) | 99.3 – 99.6% |
| Step v2/v1 | 1.23 – 1.46× |
| Step v3/v2 | 1.03 – 1.18× |
| **Step v4/v3** | **1.93 – 1.99×** (largest) |
| Step v5/v4 | 1.14 – 1.32× |
| Step v6/v5 | 0.99 – 1.01× |
| Total v1 → v6 | 3.32 – 4.27× |
| v6 vs v1 on 4 MiB (L2-resident) | 3.80× |
| v6 vs v5 on 4 MiB (L2-resident) | 1.51× |
| Finalization pass, 262144 partials | 0.0385 – 0.0454 ms |
| Finalization pass, 240 partials | 0.0088 – 0.0125 ms |
| Finalization (a) two kernels / (b) atomic / (c) last-block | 368.2 / 368.9 / 369.9 GB/s — within 0.5% |
| Float `atomicAdd` finalization, 10 runs | 5 – 9 distinct bit patterns |
| Fixed-order tree, 10 runs, fixed grid | **1** bit pattern |
| `__threadfence` last-block, 10 runs | **1** bit pattern |
| Fixed-order tree at grids 120 / 240 / 480 | **3 different answers** |
| Sequential float sum of 2^26 × 1.0f | 16777216 — **75% low** |
| Sequential float sum, dyadic 2^-10..2^9 | 8.96e-2 relative error |
| Kahan float, same data | 1.63e-8 |
| GPU tree, same data | 6.12e-7 |
| `__reduce_add_sync` SASS | `REDUX.SUM UR6, R2` (one instruction) |
| `cg::reduce(plus<unsigned>)` SASS | `REDUX.SUM UR6, R2` (the same) |
| `cg::reduce(plus<float>)` SASS | five `SHFL.BFLY` |
| shuffles / `REDUX` / `cg::reduce`, timed | 0.7223 / 0.7221 / 0.7223 ms — within 0.03% |
| `cub::DeviceReduce::Sum` vs hand-written v6 | 370.3 vs 368.6 GB/s (+0.5%) |
| `cub::DeviceReduce::Sum` temp storage | 5119 bytes |
| Legacy `volatile` warp tail, 5 runs | 0 wrong (passes; still undefined) |
| Ex2 segmented, warp-per-row+tiles vs thread-per-row | 6.6 – 10.2× |
| Ex2 strided (order-breaking) probe | 42,860 / 100,000 rows wrong |
| Ex2 wrong shuffle-offset order | 84,249 / 100,000 rows wrong, 0 wrong sums |
| Ex3 broken kernel at `n = 2^25+1` / `n = 2^25` | 5.6253% low / exact |
| Ex3 deterministic vs plain tree | 1.00 – 1.03× |
| Ex3 fixed-point (scale 2^20) relative error | 1.41e-8 |

## Assumed from earlier modules

- **M1**: SM count 40, 1536 threads/SM, warp = 32, blocks are placed by the
  GigaThread engine and run to completion, waves and tail effects, Little's Law,
  eligible/stalled warp scheduling, 432 GB/s peak, 48 MB L2.
- **M2**: `nvcc -arch=sm_89`, launch syntax, `CHECK` / `CHECK_KERNEL` idiom,
  `cudaGetLastError()` + `cudaDeviceSynchronize()`.
- **M3**: `blockIdx.x*blockDim.x+threadIdx.x`, grid-stride loops, bounds guards,
  ceil-divide, 256 as the default block size.
- **M4**: latencies (L1 40.5 / L2 241.3 / DRAM 575 cycles), **L1 is not coherent
  across SMs** (why `__threadfence()` and not `__threadfence_block()` in the
  last-block strategy), 48 MB L2 as the benchmarking hazard, `cuobjdump -sass`,
  template-parameter dispatch.
- **M5**: sectors and lines, coalescing, the streaming ceiling and the
  effective-vs-DRAM distinction, `float4` named and deferred.
- **M6**: shared memory, static declaration, cooperative loading, the barrier
  used with "Module 9 makes this precise".
- **M7**: `D = gcd(k,32)` for `s[k*tid]`, the `max(2,D)` cost law on Ada,
  conflicts are replays with **no correctness component**, FP64 at 1/64 rate.
- **M8**: divergence is warp-local and costs issue slots (1.96× measured),
  independent thread scheduling, why every warp primitive is `_sync`-suffixed,
  the mask *creates* convergence, `__activemask()` is a diagnostic not an
  argument, the dangerous middle case (converged-region warp-synchronous code
  passing by accident), `max/mean` as the bound on work-sorting speedups.
- **M9**: G1/G2, the uniformity rule and block-uniform conditions, the WAR
  barrier that needs G1 only, `volatile` is not synchronization, fences vs
  barriers, the kernel boundary as the only grid-wide barrier, no cross-block
  synchronization, cooperative groups basics (`tiled_partition<32>`,
  `thread_rank`), the `-std=c++17 -Xcompiler /Zc:preprocessor` build note.
- **M10**: atomics execute at the L2, the contention curve, `RED` vs `ATOM`,
  ticket allocation, float `atomicAdd` non-determinism (10 patterns in 10 runs),
  the order-preserving float→uint encoding.
- **Spec §12** throughout: rotated sweep order, back-to-back timing, validation
  in a separate pass, min-of-N, auto-scaled iteration counts, duration-based
  warm-up (extended to 1500 ms here — see the new finding), buffers ≫ 48 MB,
  ratios and % of a measured ceiling as the stable quantities.

## Forward-reference debts PAID here

- **M8 → M12**: "Module 12 covers reductions"; M8 used `__activemask`,
  `__ballot_sync` and `__popc` only as instruments and shipped no reduction.
  Paid: the full ladder, with M8's divergence measurement as the explanation of
  rung 1 and M8's ITS material as the reason rung 5 is written the way it is.
- **M9 → M12**: the `volatile __shared__` warp-synchronous tail shown there as a
  bug, with `__shfl_down_sync` named as the correct replacement and the algorithm
  deferred here. Paid in the lesson (with SASS), in `example02.cu` part F (run,
  and shown to pass), and in Exercise 1 TODO 5.
- **M9 CYU Q4** asked the reader to choose between `grid.sync()` and two kernel
  launches for summing 2^28 floats. Paid: the three finalization strategies are
  measured, and the two-launch answer is shown to cost 0.5%.
- **M10 → M12**: "fixed-order reduction as the deterministic alternative to
  float `atomicAdd`", explicitly deferred. Paid in `example02.cu` part B and in
  Exercise 3, **including the part M10 could not state**: a fixed-order tree is
  reproducible only for a fixed decomposition.
- **M7 → M12** (implicit): the `max(2,D)` cost law is used to explain why v2's
  bank conflicts are worth so little.

## Forward references made

- **Module 11 (memory-bound kernels / grid-stride at the ceiling)** — `float4`
  vectorized loads are named as the obvious next step and deliberately excluded
  ("Module 5 introduced them and Module 11 owns them at scale"); at 99.5% of the
  ceiling there is nothing left for them to recover.
- **Module 13 (scan)** — named in Exercise 2's solution notes as what a
  device-side partition of the ragged rows would need. No scan appears anywhere
  in this module.
- **Module 14 (histogram)** — not used; no histogram appears.
- **Module 29 (cooperative groups in depth)** — owns `cg::reduce` and the group
  API; this module uses `tiled_partition<32>` and `cg::reduce` and says so.
- **Module 30 (warp-level primitives)** — stated explicitly in the lesson: M30
  owns `__shfl_*_sync` and the ballot family as communication primitives;
  Module 12 uses them because the algorithm demands them and does not develop
  them.
- **Module 36 (CUB / Thrust)** — named as the owner of the library treatment;
  `example02.cu` part E measures `BlockReduce` and `DeviceReduce` and forwards.
- **Modules 22–23 (Nsight)** — not used. `ncu` remains unavailable
  (`ERR_NVGPUCTRPERM`); nothing in this module depends on it.

## Cross-module observations for the index

1. **New course-record streaming bandwidth: 410.7 GB/s (95.1% of 432).** The
   cross-module index records 375–381 GB/s as the best observed streaming
   bandwidth. That figure is a consequence of the **400 ms** duration-based
   warm-up standardised in Modules 6–8, which ramps the SM clock but not the
   memory P-state. With a 1500 ms warm-up the identical kernel on the identical
   data measures 410.5–410.7 GB/s, reproducibly. Modules 11+ measuring DRAM
   bandwidth should use the longer warm-up; §6b of the index should be updated.
2. **Spec §12 should gain a rule**: the warm-up must be long enough for the
   *memory* P-state, not just the SM clock; 400 ms is not.
3. **`compute-sanitizer --tool synccheck` remains useless** (M9's finding
   reconfirmed): it reports nothing on Exercise 3's kernel, as do `memcheck`,
   `racecheck` and `initcheck`. Documented as theory per the standing rule; no
   attempt was made to fix it.
4. **`memcheck` emits a benign `Resetting device while there are still other
   users claiming to use it` API warning** on any program that calls
   `cudaDeviceReset()` — which the house convention requires. It is counted in
   `ERROR SUMMARY` and is not a memory error. Worth noting in the index so
   future modules do not chase it.

## Constraints observed

- **No scan** (Module 13) and **no histogram** (Module 14) anywhere.
- No sm_90+ features. `__reduce_add_sync` is sm_80+ and is explicitly labelled
  ARCHITECTURE-SPECIFIC; every kernel that uses it also has a `__shfl_down_sync`
  sibling in the same file.
- `volatile __shared__` appears exactly once, as a bug, run and disassembled,
  never recommended (spec §9).
- Warp intrinsics used as the algorithm demands, with Module 30's ownership
  stated in the lesson, the manifest and the affected solution notes.
- Non-power-of-two `n` is mandatory and present: Exercise 3 runs at
  `n = 33,554,433 = 2^25 + 1` throughout, Exercise 2's rows are ragged with
  lengths from 8 to 200,000, and Exercise 1's TODO 4 and 5 must guard for
  arbitrary `n` even though the harness uses `2^26`.
- Nothing under `module12/` reveals an answer: the diagnosis in Exercise 3 is a
  six-way multiple choice scored against a constant in `main`, the predictions
  are scored against measurements, and no solution text appears in any exercise
  file.
- All timing follows spec §12 with the ceiling measured **inside** the same
  rotated sweep as the configurations it is the denominator for.
- No binaries committed.
