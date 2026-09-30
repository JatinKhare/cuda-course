# Module 8 manifest

## Concepts taught

- **Warp** as the unit of instruction issue, formed by the Module 3
  linearization rule: `tid = x + bx*(y + by*z)`, `warp = tid/32`,
  `lane = tid%32`. **PORTABLE CUDA CONCEPT** (warps exist, formed by
  linearization); **ARCHITECTURE-SPECIFIC** (the number 32).
- **Partial warp** — a 100-thread block becomes 4 warps; the last has 28 lanes
  that were never created, occupy thread slots and register-file space for the
  block's lifetime, and can never be given work. Measured mask `0x0000000f`.
- **Warp boundaries in multi-dimensional blocks** — a `dim3(10,10)` block cuts
  warps mid-row; a row of a block is not a warp.
- **The SIMT contract** — one warp scheduler issues at most one instruction per
  clock to one warp; every active lane executes it. Cost is counted in *issue
  slots*, not lanes.
- **Active mask** — 32-bit per-instruction lane-participation mask. Masked-off
  lanes produce no register write, no memory access, and supply no address to
  the coalescer (links to M5).
- **Divergence** — lanes of one warp needing different instructions; paths
  executed **sequentially** with complementary masks; cost is **additive
  (sum), not max**. Measured 1.99x on a single warp with `clock64()`, 1.96x on
  a full-grid benchmark.
- **Divergence is warp-local** — proved by measurement: `tid&1` costs 1.97x,
  `tid<128` (warp-uniform) costs 1.02x, over the same data and the same total
  work. The *number* of divergent lanes does not enter the price: a 16/16 split
  and four 8-lane groups both cost 1.96x.
- **n-way divergence** — cost is the sum of all n bodies. With the honest
  caveats: the compiler often converts small switches to arithmetic (measured
  0.94x, i.e. no divergence at all), and when arms genuinely differ the penalty
  can be below n because different arms use different pipelines (an 8-way
  divergence measured 6.1–6.3x, not 8x).
- **Loop trip-count divergence** — the warp issues the body `max(trips)` times
  with a decaying active mask. Measured 1.65x against a `max/mean = 1.78x`
  model, with the shortfall explained.
- **Predication / if-conversion** — short bodies compiled to predicated
  instructions with no branch; both sides issued once, unconditionally, by the
  whole warp. Contrasted with branching, which issues each side once under its
  own mask.
- **The predication flip point on this GPU** — measured at **7 vs 8 FFMAs per
  arm** (CUDA 13.2, `-O3`, sm_89); explicitly labelled a compiler heuristic,
  not an architectural rule.
- **The Module 3 bounds guard, made precise** (M3 debt paid) — three measured
  shapes: guard as tail → `@P0 EXIT`; short guarded body → `@!P0 STG.E`
  (predicated store, predicated address arithmetic); guarded body containing a
  global load → `BSSY`/`@P0 BRA`/`BSYNC`. Body length is not the only criterion.
- **`__activemask()` cannot distinguish predication from branching** — the
  `VOTE` that implements it carries the same predicate, so a predicated-off
  lane is excluded from the mask exactly as a not-taken lane is. Only the SASS
  distinguishes them.
- **Reconvergence, classic model** — one PC per warp, hardware divergence stack,
  guaranteed immediate reconvergence at the **immediate post-dominator**;
  `SSY`/`SYNC` on pre-Volta.
- **Independent thread scheduling (sm_70+)** (M1 debt paid) — per-thread PC and
  call stack; the hardware may interleave divergent paths and **does not
  guarantee reconvergence at the post-dominator**. `BSSY`/`BSYNC` are compiler-
  placed convergence barriers, not a language guarantee. Motivation: ITS makes
  intra-warp fine-grained synchronization possible (the lock/spin deadlock under
  the old model).
- **Why every warp primitive is `_sync`-suffixed with an explicit mask** — the
  mask is how you *create* the synchrony the hardware no longer provides;
  unsuffixed forms are removed, not deprecated.
- **`__syncwarp(mask)`** — warp-scope convergence barrier and compiler barrier;
  must be reached by a warp-uniform set of lanes; distinct from
  `__syncthreads()` (Module 9's subject).
- **Pre-Volta warp-synchronous programming is broken, not deprecated** —
  presented only as a bug to diagnose. `volatile` is a *visibility* mechanism,
  not an *ordering* one; it orders nothing between threads.
- **The dangerous middle case** — converged-region warp-synchronous code
  usually still produces correct results on sm_89 because `nvcc` reconverges at
  post-dominators. Measured: 0 errors out of 524,288. Documented as an accident
  of the code generator, never as a guarantee.
- **Warp-level cost consequences** — `tid % 32` patterns catastrophic,
  `tid / 32` patterns free; sorting/binning work so lanes agree as a real
  technique bounded by `max(work)/mean(work)`, with the reordering cost priced
  and a break-even launch count reported.
- **Lane efficiency** = useful lane-executions / (32 x issues); the single
  figure of merit for divergence (47.7% for Exercise 1's control-flow shape,
  56% for Exercise 2's naive mapping).
- **Benchmarking under spec §12** — `clock64()` inside the kernel as a
  clock-independent measure; duration-based 300–400 ms warm-up; all
  configurations timed back-to-back; min of 5–6 sweeps of 20 iterations;
  validation in a separate pass; ratios reported as the stable quantity. Plus a
  documented pitfall: a per-*warp* "warp-uniform" control config unbalances the
  four processing blocks of the SM and measures imbalance rather than
  divergence (1.09x); per-*block* assignment over a multi-wave grid gives 1.65x.

## CUDA API / intrinsics / syntax introduced

- `__activemask()` — as an **observation instrument only**
- `__ballot_sync(mask, pred)` — as an instrument for sizing a split in advance
- `__popc(v)` / `__ffs(v)` — bit counting and lowest-set-bit, used to pick a
  single reporting lane per issue
- `__syncwarp()` / `__syncwarp(mask)` — warp-scope barrier (the repair for
  warp-synchronous code)
- `__shfl_xor_sync(mask, v, lanemask)` — mentioned once, as the *misuse* case
  (full mask from a half-warp region), not taught as a tool
- `clock64()` — in-kernel cycle counter, used to timestamp divergent arms
- `volatile __shared__` — presented **only** as a bug pattern
- `cudaDevAttrWarpSize` — mentioned as the defensive query
- SASS vocabulary: `@P0` / `@!P0` predication, `VOTE.ANY`, `BSSY`, `BSYNC`,
  `BRA`, `EXIT`, `LDS`, `STS`, `BAR.SYNC`, `WARPSYNC`, `FLO.U32.SH`
- `cuobjdump -sass` on both an object file and an executable (assumed from M4,
  used heavily)
- `compute-sanitizer --tool racecheck` on an intra-warp shared-memory hazard
- Explicit template instantiation of `__global__` templates so their SASS lands
  in the binary without being launched

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Predict-the-behavior + design | 5 | Predict the 32-bit active mask at 11 labelled sites and on all 8 iterations of a variable-trip loop in one warp, then implement the SIMT issue-count model and match the hardware counters | **P8** is `0x0fffffff`, not `0xffffffff`: lanes 28..31 executed `return` and never rejoin the post-dominator. Secondary trap: the one-instruction `if/else` at P9/P10 is if-converted, and the intuitive conclusion "no branch, so all 32 lanes are active" is wrong — `__activemask()` compiles to a *predicated* `VOTE` and reports `0x0aaaaaaa`/`0x05555555`. TODO 5's obvious `sum` or `count-of-lanes` models both fail only at the loop site. |
| `exercise02.cu` | Optimization + design ("choose the strategy") | 4 | Data-dependent per-element work (1..8 rounds); reader must make warps homogeneous and beat the naive kernel by >1.30x with bit-identical output | Writing `out[t]` instead of `out[order[t]]` yields a kernel that is exactly as fast, never faults, and silently permutes the result — caught only because the harness compares bit-exactly against the naive output. Second trap, not a failure but a lesson the harness forces: a `qsort`-based plan is correct and ~10x more host time than the counting sort the 8-valued key admits, and the harness prints the break-even launch count (67) so the reader must confront that the optimization can be a net loss for a single launch. |
| `exercise03.cu` | Debugging (ITS) + design | 4 | A `volatile __shared__` warp ring-rotate written in pre-Volta warp-synchronous style; 100% wrong, deterministically; reader diagnoses and repairs it | The reader is shown a *second* kernel, `warpSmoothConverged`, that uses the identical `volatile`/no-`__syncwarp` idiom and **passes** (0/524,288 wrong). TODO 4b asks whether that proves the programming model guarantees it; the answer is no, and a reader who answers yes has learned the most dangerous possible wrong lesson. Also: `__syncwarp()` placed *inside* the divergent arms looks like the obvious fix and is undefined (a full-mask warp barrier reached by 16 lanes); and `compute-sanitizer --tool memcheck` is clean, so the reader must reach for `racecheck`. |

## Assumed from earlier modules

- **M1:** SM anatomy (4 processing blocks, 1 warp scheduler each, 12 warps per
  processing block, 48 warps / 1536 threads per SM, 40 SMs); eligible vs
  stalled warps and one issue per scheduler per clock; SIMD vs SIMT and the
  implicit active mask; "5120 cores" as 160 instruction streams; the promise
  that ITS would be explained here.
- **M2:** `nvcc -arch=sm_89`, launch syntax, the `CHECK` macro idiom, checking
  both `cudaGetLastError()` and `cudaDeviceSynchronize()`, `compute-sanitizer`
  existing.
- **M3:** the linearization rule (used as the definition of warp membership);
  bounds guards and the claim that they are predicated (paid off here);
  `blockDim` multiple-of-32 rule (explained here); 256 as the default block
  size; grid sizing.
- **M4:** `cuobjdump -sass`, registers-per-warp allocation, shared memory named
  and priced.
- **M5:** predicated-off lanes supply no address to the coalescer; sector
  counting; scattered loads and the extra cost of scattered stores
  (read-modify-write) — used to explain why the gather in Exercise 2 is free
  here and would not be in a memory-bound kernel.
- **M6/M7 (concurrent):** shared memory and `__syncthreads()` are *used* in
  Exercise 3 (one block-level barrier to publish an initial load) but not
  explained; bank conflicts are explicitly ruled out as the cause of the bug.

## Forward references made

- **Module 9 (synchronization)** — owns `__syncthreads()` semantics, memory
  ordering and fences, barriers in divergent control flow, and the full
  "`volatile` is not synchronization" treatment at block and device scope. This
  module uses `__syncwarp()` only as the warp-scope repair for warp-synchronous
  code and says so explicitly in the lesson and in both affected solution notes.
- **Module 10 (races and atomics)** — named as the owner of cross-block races
  and `racecheck` for them; Exercise 1's per-issue counter is explicitly
  constructed to avoid needing atomics (one warp, one lane per issue).
- **Module 12 (reductions)** — named as the owner of reductions; no warp
  reduction is taught or shipped here.
- **Module 16** — named alongside M12 as where device-side binning/scan makes
  the two-level version of Exercise 2's strategy practical.
- **Module 23 (Nsight Compute)** — named as the source of the hardware counter
  equal to the lane-efficiency figure computed by hand in Exercise 1, and as
  where divergence and latency-hiding stalls are reported separately.
- **Module 29 (cooperative groups)** — not used; cooperative groups deliberately
  absent.
- **Module 30 (warp intrinsics as programming tools)** — named repeatedly and
  explicitly as the owner of `__shfl_*_sync` and ballot as *communication
  primitives*. This module states in the Concept section that it uses
  `__activemask()`, `__ballot_sync()` and `__popc()` **only as instruments for
  observing execution**, and the one appearance of `__shfl_xor_sync` is as a
  measured misuse case (49.95% wrong results from a full mask supplied inside a
  half-warp region), not as a technique.

## Measured results recorded in this module

| Measurement | Value |
|---|---|
| Single-warp divergent if/else vs no branch (`clock64()`) | 1.99x, zero interval overlap |
| `tid & 1` vs uniform, full grid | 1.96x |
| `(tid>>3) & 1` (four 8-lane groups) vs uniform | 1.96x |
| `tid < 128` (warp-uniform) vs uniform | 1.02x |
| Trip counts 1..8 in-warp vs warp-uniform | 1.65x (model 1.78x) |
| Predication → branch flip point | 7 vs 8 FFMAs per arm |
| Exercise 2 speedup from binning | 1.58x (model 1.78x); 1.54–1.60x across runs |
| Exercise 2 warp homogeneity | 0.00% naive → 99.98% binned |
| Exercise 2 break-even | 67 launches |
| Exercise 3 broken kernel | 524,288 / 524,288 wrong, identical on 10 runs |
| Exercise 3 converged variant (same idiom, no divergence) | 0 wrong — passes by accident |
| 8-way structurally-distinct divergence | 6.1–6.3x, not 8x (different pipes overlap) |
| 8-way divergence where arms differ only by a constant | 0.94x — compiler converted it to arithmetic |
| Full mask given to `__shfl_xor_sync` from a half-warp region | 49.95% of elements wrong |
