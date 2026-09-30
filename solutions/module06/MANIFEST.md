# Module 06 manifest

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and contains spoilers.

Files: `module06/lesson.md`, `example01.cu`, `example02.cu`,
`exercise01.cu`, `exercise02.cu`, `exercise03.cu`.
Solutions: `solutions/module06/exercise0{1,2,3}_solution.{cu,md}`,
`check_your_understanding.md`, `MANIFEST.md`.
All `.cu` files verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2) on the
RTX 3500 Ada, warning-clean; all solutions print `OVERALL: PASS`; all three
shipped exercises compile with TODOs blank and exit gracefully.

## Concepts taught

- **Shared memory as a software-managed scratchpad, not a cache** — no tags, no
  eviction, no backing store, no policy; *guaranteed* residency versus a cache's
  *predicted* residency. **PORTABLE CUDA CONCEPT** (sizes are Ada-specific).
- **The two jobs**: (1) *buying reuse* — K readers per element becomes 1 global
  read; (2) *decoupling the access pattern from the layout* — load coalesced,
  read in any order. (2) is M5's promised fix for the "neither side coalesces"
  case.
- **Arithmetic-intensity arithmetic**: K (readers per element), H (halo tax =
  tile cells / outputs), K/H as the ceiling on traffic reduction. Worked for the
  5-point stencil (3.8), box filters R=1..4 (6.8 → 32.4), and forward to GEMM.
- **The honest correction**: on Ada, L1/L2 have usually already captured the
  reuse, so shared memory replaces *L1 hits*, not DRAM traffic. A kernel at 72.6 %
  of DRAM peak has a hard 1.38× ceiling regardless of what you do on chip.
  Measured: 5-point stencil tiling is a **net loss (0.85×)**.
- **Scope and lifetime**: per block, allocated at block placement, destroyed at
  retirement; contents undefined at entry; one declaration, N physical arrays.
- **Why the scratchpad is implementable at all**: M1's indivisible, non-migrating
  block guarantee is the precondition. A movable scratchpad is a cache.
- **Cooperative loading**: the LOAD mapping and the COMPUTE mapping are different
  functions; the flat strided loop `for (idx = tid; idx < SW*SH; idx += nthr)` as
  the canonical shape; why the chunk-per-thread alternative is uncoalesced
  (M5 stride-2 case).
- **Halo / ghost cells**; why (TW+2R)(TH+2R) ≠ TW·TH makes the load non-1:1; why
  the clamp in the load simultaneously handles image boundaries and partial tiles;
  why a 5-point stencil never reads the four tile corners.
- **Barriers, used not explained**: `__syncthreads()` as "barrier; **Module 9**
  makes this precise". Two rules of thumb: (1) after writing a tile before
  reading it (RAW); (2) after reading a tile before overwriting it (WAR, only
  exists when a buffer is reused across iterations).
- **Why both hazards vanish at 32 threads** — a single warp supplies the ordering
  accidentally; "works at warp width" is not evidence of correctness.
- **Barrier cost is convoying**, not the `BAR.SYNC` instruction: the block runs at
  the speed of its slowest warp at each barrier; warp-level slack is exposed.
- **Static shared memory**: `__shared__ T a[N]`, visible to `ptxas -v` and to
  `cudaFuncGetAttributes().sharedSizeBytes`.
- **Dynamic shared memory**: `extern __shared__`, the **third launch parameter
  `sharedBytes`** (M2's debt, paid explicitly); a byte count, not an element
  count; exactly one region per kernel, all `extern __shared__` declarations
  alias it; invisible to `cudaFuncGetAttributes`.
- **Carving several typed arrays out of one blob**, with `align_up`; the
  `alignof(float2) = 8` hazard at offset 132; ordering arrays widest-alignment-
  first makes padding unnecessary.
- **The asymmetry of the two carve bugs** (measured): the alignment bug raises
  `cudaErrorMisalignedAddress` at execution time, is sticky, and cannot corrupt;
  the byte-count bug is **silent** — a 1 KB under-request still passes validation
  and `compute-sanitizer --tool memcheck` reports nothing.
- **Capacity and occupancy coupling**: 49152 B/block default, 101376 B opt-in,
  102400 B/SM, **1024 B driver-reserved per block**;
  `blocks/SM = min(1536/threads, 102400/(bytes+1024), 24)`; measured table.
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` as the way to get it right.
- **The >48 KB opt-in**: `cudaFuncSetAttribute(...,
  cudaFuncAttributeMaxDynamicSharedMemorySize, n)`; without it,
  `cudaErrorInvalidValue` at launch (non-sticky); why the per-block ceiling
  (99 KB) is below the per-SM figure (100 KB).
- **L1 and shared memory compete for the same 128 KB SRAM** — the scratchpad is
  taken out of the cache that was already doing the job.
- **Shared memory is a separate address space in the ISA**: `LDS`/`STS` versus
  `LDG`/`STG`; no tag compare is where the latency and the *determinism* come
  from.
- **Benchmarking**: iteration count auto-scaled so each timed segment is ~10 ms
  (20 iterations of a 0.01 ms kernel lets the clock sag between segments and
  produced a spurious 2.3× in an early draft); duration-based 400 ms warm-up;
  min-of-4 sweeps; validate in a second pass; report ratios.
- Explicitly **not** taught: bank conflicts (Module 7 owns them, stated three
  times); memory-model semantics of the barrier (Module 9); atomics (Module 10);
  warp intrinsics (Module 30); cooperative groups.

## CUDA API / intrinsics / syntax introduced

- `__shared__` (static declaration, multi-dimensional and flat)
- `extern __shared__ T name[]`; `extern __shared__ char smem[]` + `reinterpret_cast`
- the third launch parameter: `kernel<<<grid, block, sharedBytes>>>(...)`
- `__syncthreads()` (used; semantics deferred to Module 9)
- `alignof(T)`; `align_up(off, a)` idiom as a `__host__ __device__` helper
- `cudaFuncSetAttribute` with `cudaFuncAttributeMaxDynamicSharedMemorySize`
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- `cudaFuncGetAttributes().sharedSizeBytes`
- `cudaDeviceGetAttribute` with `cudaDevAttrMaxSharedMemoryPerBlock`,
  `cudaDevAttrMaxSharedMemoryPerBlockOptin`,
  `cudaDevAttrMaxSharedMemoryPerMultiprocessor`,
  `cudaDevAttrReservedSharedMemoryPerBlock`, `cudaDevAttrMultiProcessorCount`
- `make_float2`, `float2` in shared memory
- errors named and demonstrated: `cudaErrorMisalignedAddress` (execution-time,
  sticky), `cudaErrorInvalidValue` (launch-time, non-sticky)
- SASS mnemonics named: `LDS`, `STS`, `LDS.64`, `STS.64`, `BAR.SYNC`
- Tooling: `nvcc -Xptxas -v` (`... bytes smem`), `-lineinfo`,
  `compute-sanitizer --tool racecheck --racecheck-report analysis`,
  `compute-sanitizer --tool memcheck`, `cuobjdump -sass`

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Fill in the code + prediction (§6 types 1, 2, 6) | 4 | Tile Module 3's 5-point clamped stencil on the same 1021×733 and 4093×3079 images, three tile shapes, timed against an untiled control and the recorded M3 baselines | The exercise's *purpose* is a trap: the correct answer is that tiling is a **net loss (0.85×)**, and TODO 4 forces the reader to commit to a bucket before measuring. Load-loop traps: the 1:1 load leaves the halo unwritten; the contiguous-chunk-per-thread loop validates but is stride-2 uncoalesced; the interior guard `if (row<h && col<w)` applied to the *load* leaves right/bottom edge stripes wrong only on the non-multiple image; a corner-free halo load is **correct** here because a 5-point stencil never reads a corner. |
| `exercise02.cu` | Fill in the code (design the layout) | 3 | Dynamic shared memory: RBF scatter interpolation, K = 256, staging a source tile *and* a 33-entry lane-varying profile table | Two non-symmetric bugs. `float2` at offset 132 → `cudaErrorMisalignedAddress` (loud, sticky, safe). Byte count `33·4 + 256·8 + 256·4 = 3204` instead of 3208 → **silent**; verified that a full 1 KB under-request still prints PASS and memcheck reports nothing. Plus: `MS = 4093` partial tile (stale slots contaminate every query), and threads with `i >= nq` must **not** return early because they are part of the cooperative load. Third solution-only config shows that padding the tile to a compile-time loop bound is *slower* (1.33× vs 1.40×), contradicting the usual rule of thumb. |
| `exercise03.cu` | Debugging (§6 type 3) | 3 | `fold_blend`, a block-local mirror-and-average; passes at 32 threads, fails at 64+; a second defect survives the first fix | Two instances of the same class with very different loudness: the missing RAW barrier corrupts ~45 % of elements every run, the missing WAR barrier (needed only because the buffer is reused across `--tiles 8` iterations) corrupts ~4.5 % and only when warps drift apart. `memcheck` finds **nothing** (no access is out of bounds); `racecheck` finds both and — the Prediction question — reports the 32-thread configuration too, as a *Warning*, even though it produces correct answers on every run. Distractor diagnoses include bank conflicts, which have no correctness component at all. |

`OVERALL: PASS` in exercises 1 and 3 requires correct numerics **and** a correct
prediction/diagnosis code, so neither can be passed by running first.

Unfilled behaviour: `exercise01` prints `Set TODO 4 (PREDICTION) first.`,
`exercise02` prints `Set TODO 1 first.`, both return 0. `exercise03` is a
debugging exercise and deliberately runs and reports `FAIL` so the reader sees
the symptom; it never crashes.

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | (A) per-block scope: 8 blocks, one declaration, 8 private arrays; (B) static vs dynamic side by side, with `cudaFuncGetAttributes` showing the dynamic size is invisible; (C) carving `int[33]`/`float2[64]`/`float[64]` out of one blob, with `--align-bug` producing a real `cudaErrorMisalignedAddress` and showing it is sticky; (D) the measured bytes-per-block → blocks-per-SM table including the 1024 B driver reservation; (E) 64 KB dynamic request rejected with `cudaErrorInvalidValue`, then accepted after `cudaFuncSetAttribute`, and the 1-block-per-SM consequence |
| `example02.cu` | The traffic ledger: 2D box filter R=1..4, untiled vs tiled, on 4093×3079, with K, halo tax H, ideal K/H and measured speedup in one table (0.87× / 0.85× / 0.92× / 1.06×); computes at runtime the argument that the R=1 naive kernel's true re-read factor is ≤1.4, not 9, because it is already at 70 % of DRAM peak. Also shows the register-resident clamp hoist that keeps the experiment about memory rather than integer min/max |

## Measured numbers recorded (RTX 3500 Ada, CUDA 13.2)

- Untiled 5-point stencil, 4093×3079: **0.3187–0.3262 ms** — reproduces M3's
  0.3213 ms to within 1 %.
- Tiled 32×8 / untiled, 4093×3079: **0.85×** typical, **0.72–0.93×** observed.
  Tiled 32×16: 0.815×. Tiled 64×8: 0.851×.
- Box filter tiled/untiled: R=1 0.87×, R=2 0.85×, R=3 0.92×, R=4 1.06×.
- RBF tiled/untiled: **1.40×** typical, 1.37–1.60× observed. Padded-tile variant
  1.33×.
- Blocks/SM at 256 threads: 6 up to 12288 B, 5 at 16384 B, 3 at 25600 B,
  2 at 49152 B, 1 at 65536 B.
- Driver-reserved shared memory per block: **1024 B**.
- Absolute kernel times on this laptop part vary by up to 2.5× with thermal
  state; every ratio quoted above is stable to ~±0.05.

## Assumed from earlier modules

- **M1**: 40 SMs, 1536 threads/SM, 48 warps, 24 blocks/SM, warp = 32 lanes,
  128 KB unified L1+shared per SM with ≤100 KB addressable as shared,
  blocks are dispatched whole to one SM and never migrate, waves, Little's Law,
  latency hiding by resident warps, 432 GB/s peak.
- **M2**: `nvcc -arch=sm_89`, `__global__`, the four launch parameters (third one
  deferred *to here*), `CHECK` macro idiom, `cudaGetLastError()` +
  `cudaDeviceSynchronize()` two-class error model, sticky vs non-sticky errors,
  `cudaEvent_t` timing with warm-up and ≥20 iterations.
- **M3**: 2-D indexing, the linearization rule, warp formation from linearized
  `threadIdx`, bounds guards, ceil-divide grid sizing, block-shape performance,
  the 66.7 % ceiling of 1024-thread blocks, and **Exercise 1's stencil and its
  measured baselines**, reused verbatim.
- **M4**: register / local / shared / L1 / L2 / global map with latencies, local
  memory is DRAM (hence the register-resident `cc[]` hoist in example02),
  L1 is not coherent across SMs, 48 MB L2 as the benchmarking hazard,
  constant-memory broadcast-vs-serialize and the 24.6× lane-varying penalty,
  `-Xptxas -v`, `cuobjdump -sass`, template-parameter dispatch.
- **M5**: sectors and lines, the sector-counting procedure, alignment thresholds,
  `cudaErrorMisalignedAddress` from a `float4` cast, stride-2 = 50 % efficiency,
  effective vs DRAM bandwidth, >100 % of peak means L2 residency.

## Forward-reference debts paid

- **M2 → M6**: the third launch parameter `sharedBytes` and the 48 KB/99 KB
  limits. Paid in lesson §7 and §9, example01 §B and §E, exercise02 TODO 1.
- **M3 → M6**: "larger blocks amortise work" / block-size amortisation. Paid in
  lesson §3 and §9 and measured in exercise01 (32×8 vs 32×16 vs 64×8 — larger
  tiles reduce the halo tax and lose anyway).
- **M4 → M6**: "shared memory introduced only, Module 6 owns it", and "the right
  answer for a heavily reused lane-varying table". Both paid; exercise02's
  profile table *is* that lane-varying table.
- **M5 → M6**: the case where neither the read nor the write side can be
  coalesced. Paid in lesson §2 (job 2) and §5 (the cooperative load is the
  coalesced half of the decoupling).

## Forward references made

- **Module 7 (bank conflicts)** — stated three times and deliberately excluded:
  lesson §2, §3 (Hardware Mental Model closing), and exercise03's `DIAG_BANK`
  distractor. Framed as "a replay mechanism exactly like M4's constant-memory
  serialization", which is the hand-off M7 is expected to pick up.
- **Module 8 (warps, SIMT, independent thread scheduling)** — "works at 32
  threads" as accidental ordering; ITS on sm_70+ named in exercise03's notes.
- **Module 9 (`__syncthreads` semantics)** — stated at every use of the barrier;
  barrier-in-divergent-control-flow (the early-`return` hazard in exercise01 and
  exercise02); "`volatile` is not synchronization" set up as exercise03's
  suggested variation.
- **Module 10 (races, atomics)** — race vocabulary used informally; atomics
  deliberately absent.
- **Modules 15–17 (transpose, GEMM, tiling at scale)** — the explicit destination
  of the K/H argument; "the transformative wins need register blocking too".
- **Module 17 (software pipelining)** — double-buffering named as the alternative
  to the WAR barrier in exercise03's notes.
- **Module 19 (occupancy)** — the full blocks/SM story; "maximum occupancy is not
  maximum speed".
- **Module 23 (Nsight Compute)** — implied in CYU Q2's "read the counter names
  carefully".
- **Module 26 / 29 (cooperative launch)** — cooperative groups named in CYU Q4 as
  the API whose co-residency precondition is the same constraint as shared
  memory's.
- sm_90 **thread block clusters / distributed shared memory** named in CYU Q4 as
  the hardware answer to "why not a grid-wide scratchpad", with the explicit note
  that Ada does not have it.

## Constraints observed

- No bank-conflict teaching. Every shared access in this module is either
  warp-uniform or unit-stride across lanes, so banks never affect a measurement
  here — stated explicitly in the lesson so M7 is not pre-empted or contradicted.
- No atomics, no warp intrinsics, no cooperative groups, no full memory-model
  treatment.
- No sm_90+ features used; clusters mentioned only as a contrast in an answer.
- Everything builds with `nvcc -arch=sm_89 -O3`, warning-clean.
