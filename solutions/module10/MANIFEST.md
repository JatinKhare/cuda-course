# Module 10 manifest

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module10/lesson.md`, `module10/example0{1,2}.cu`,
`module10/exercise0{1,2,3}.cu`.
Solutions: `solutions/module10/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module10/check_your_understanding.md`,
`solutions/module10/MANIFEST.md`.
All `.cu` verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2, RTX 3500 Ada),
warning-clean; all three solutions print `OVERALL: PASS`; all three shipped
exercises compile with TODOs blank and exit gracefully with `Set TODO 1 first.`

---

## Concepts taught

- **Read-modify-write (RMW)** as three instructions (`LDG` / `IADD3` / `STG`),
  shown in real SASS; the two interleaving windows.
- **Data race**, **lost update**; why the loss is catastrophic rather than
  marginal, and why a warp of 32 lanes contributes at most 1.
- **The atomicity / ordering / visibility distinction** — the module's central
  idea. Barriers order threads, fences order accesses, neither makes a single
  update indivisible. Demonstrated with `__syncthreads()` + `__threadfence()`
  around the race, and with the SASS showing the three instructions untouched
  between `BAR.SYNC` and `MEMBAR.SC.GPU`.
- **Where a device-scope atomic executes**: at the L2, the first level shared
  by all SMs, because L1 is not coherent across SMs (M4/M5 debt). Atomics are
  *shipped to* the memory system, not performed under a lock.
- **Atomic latency is L2 latency** (~241 cycles, M4's measurement); atomic
  throughput is per-L2-slice.
- **Atomics return the OLD value**; **ticket allocation** and stream
  compaction; uniqueness vs ordering (ordering deferred to M13's scan).
- **`atomicInc`/`atomicDec` wrap semantics**: cycle length is `limit+1`; a
  ring buffer of capacity C needs `atomicInc(head, C-1)`.
- **`atomicCAS` as the universal primitive**; the canonical loop; the three
  ways to get the loop condition wrong (exit on the written value, recompute
  from the stale read, retry on a tie → livelock).
- **Multi-field atomic update** by CAS on a packed 64-bit word, and why two
  independent atomics on the two halves is not equivalent (intermediate
  inconsistency).
- **Order-preserving float→uint32 encoding**; the sign-bit trap that makes
  `atomicMax((int*)p, __float_as_int(v))` wrong for negatives; choosing an
  encoding so a hardware instruction replaces a CAS loop.
- **Contention economics**: cost is governed by how many threads target the
  same address, not by how many atomics execute. Full measured curve,
  K = 1 … 2^20.
- **An uncontended atomic costs what a plain store costs** (measured ratio 1.0
  at K ≥ 4096, 48× at K = 1). "Atomics are slow" is false.
- **L2 slice hashing**: K distinct addresses give K-way parallelism only if
  they hash to K slices; adjacent words often do not. Measured with a word-
  stride sweep (1 / 8 / 32 / 1024).
- **Same-address concurrency within a warp** as the dominant variable: a
  warp-uniform address costs the K=1 price regardless of how many distinct
  addresses exist in the array.
- **Shared-memory atomics** (`ATOMS`) vs global (`ATOMG`/`RED`): SM-local,
  block-scoped, 48× cheaper measured; `ATOMS.POPC.INC.32` as Ada's hardware
  aggregation of same-address increments.
- **`ATOM` vs `RED`**: the compiler emits `RED` (no return value, no
  scoreboard dependency) whenever the return value is discarded. Verified in
  SASS.
- **Compiler warp aggregation** (`VOTEU.ANY` + `UFLO` + `POPC` + predicated
  `RED`, and the `POPC`+`SHFL.IDX` form when the return value is used), and
  that it requires provable warp-uniformity — 27.6× for identical traffic.
- **`REDUX.MAX.S32`** for a uniform-address `atomicMax` with discarded return.
- **Privatization** as the general pattern: private copy, zero, barrier,
  accumulate with shared atomics, barrier, flush. The traffic arithmetic
  (`n` global atomics → `n` shared + `g·b` global) and the crossover at which
  it becomes a pessimization.
- **Four contention-reduction moves**: discard the return value, warp-level
  pre-aggregation, privatization, address spreading — each measured, each with
  a measured point where it stops paying. Plus a fifth found in Exercise 3:
  declining to issue the atomic at all.
- **Floating-point `atomicAdd` is non-deterministic**: FP addition is not
  associative, atomics impose an unspecified order, 10 distinct bit patterns
  in 10 identical runs. Deterministic alternatives named (fixed-order
  reduction, integer/fixed-point).
- **`RED.E.ADD.F32.FTZ.RN`**: global float atomics flush denormals.
- Atomic contention as the **third instance of replay**, after M4's
  constant-memory serialization and M7's bank conflicts.

## CUDA API / intrinsics / syntax introduced

- `atomicAdd` (int, unsigned, unsigned long long, float, double; half2/
  bfloat162 forms named), `atomicSub`, `atomicExch`, `atomicMin`, `atomicMax`,
  `atomicAnd`, `atomicOr`, `atomicXor`, `atomicInc`, `atomicDec`, `atomicCAS`
  (32- and 64-bit)
- Shared-memory atomics (same functions, shared-address operand)
- `__match_any_sync` (sm_70+) for warp-level peer discovery
- `__activemask()`, `__ffs`, `__popc` in the aggregation idiom
- `__shfl_down_sync` on `unsigned long long` (used, introduced in M8/M9)
- `__float_as_uint` / `__uint_as_float`, and the `__CUDA_ARCH__`-guarded
  `__host__ __device__` bit-reinterpretation helper
- `compute-sanitizer --tool racecheck`, `--racecheck-detect-level`,
  `NV_COMPUTE_SANITIZER_MAX_RACECHECK_HAZARDS` (mentioned)
- `cuobjdump -sass` reading of `ATOMG` / `RED` / `ATOMS` / `REDUX` / `MEMBAR`
- `-lineinfo` as the flag that makes racecheck name source lines

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | A: racy counter at 6 launch shapes (262,144 threads → 13). B: the same with `__syncthreads()` + `__threadfence()`, still 1. C: `atomicAdd`, exact. D: ticket allocation / stream compaction over 100,000 elements, dense and unordered. E: `float atomicAdd` giving 10 distinct bit patterns in 10 runs. F: `atomicInc` wrap semantics, cycle length `limit+1`. |
| `example02.cu` | The contention curve with the atomic *count* held at exactly 2^20: K = 1…2^20 (55.8× spread), word-stride sweep exposing L2 slice hashing, warp-uniform vs lane-varying, shared privatization (48× at K=1, 0.22× at K=4096), `__match_any_sync` aggregation (27.7× at K=1, 0.46× at K=1024), and literal-vs-runtime address showing the compiler's own aggregation at 27.6×. Spec §12 timing throughout. |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Debugging (spec §6 type 3) + prediction | 5 | Event triage with per-class counts, a global checksum and a bounded critical list; three symptoms, four defects, three race classes | **racecheck reports 2 of the 4 defects and neither of the two with the worst symptoms** — it is a *shared memory* hazard checker by its own `--help` text, and both global RMWs are invisible to it and to memcheck. TODO 5's requirement is self-contradictory on its face (never write past CAP, but report the true count when it exceeds CAP); the resolution is to let the counter run free and use its return value conditionally. The tempting `if (*cnt < CAP) { slot = atomicAdd(...) }` is an out-of-bounds write. The barrier that *is* present (before the flush) is correct and outside divergent control flow; the missing one is after the zeroing. |
| `exercise02.cu` | Design (spec §6 type 5/6) | 4 | Weighted category tally, 8 M samples, 50% of weight in one category; reduce contention without being told how; 4 scenarios × {16,4096} bins × {clustered,shuffled} | TODO 2 never says "privatization". Warp-aggregation alone wins `clustered` and does **nothing** on `shuffled` (a warp's 32 lanes hit 32 bins, `MATCH.ANY` finds no groups) → fails the ≥2× gate. Private bins must be 32-bit, not 64-bit: per-block partials cannot overflow, and the halved footprint doubles resident blocks. **Documented surprise:** the predicted catastrophe from an uncapped grid at ncat=4096 (16 M extra flush atomics) does **not** materialise — they are uncontended and L2-resident, so they are nearly free; the real cost of the reflexive design is one sample per thread, not the atomic count. The harness gates at 2.5× to catch it anyway. |
| `exercise03.cu` | Fill-in + design, hardest | 5 | `atomicCAS` argmax over floats with smallest-index tie-break, a packed {max,min} bounds word, a monotonic encoding that replaces the CAS loop with one `atomicMax`, and a fast version | The sign trap: `atomicMax((int*)p, __float_as_int(v))` orders negatives backwards and dataset B is all-negative. The CAS-loop trap: `atomicCAS` returns the old value, so `while (old != desired)` exits after a failed swap, and a predicate tested only once livelocks on dataset C (all values identical). The packing trap: the index half must be *decreasing* (`~idx`), or the all-zero identity ties with a real candidate at index 0 — and dataset C's answer *is* index 0. **Performance surprise:** the CAS loop is 23× *faster* than the single `atomicMax` on dataset A, because it early-outs and issues almost no atomics — the cheapest atomic is the one you never execute. |

Scoring: Ex1 6 correctness points + a hashed 4-part prediction (no partial
credit, and the answer is not in the file); Ex2 9 points including two
speedup gates and two octave-scored predictions; Ex3 5 points including a
monotonicity sweep over 4096 sampled floats and a 5× speed gate.

## Assumed from earlier modules

- M1: warp = 32 lanes issuing one instruction; SM structure; eligible/stalled.
- M2: launch syntax, `cudaGetLastError` + `cudaDeviceSynchronize`, `CHECK`,
  `__host__ __device__`.
- M3: `blockIdx.x*blockDim.x+threadIdx.x`, grid-stride loops, bounds guards.
- M4: the storage map with measured latencies (L1 40.5 / L2 241.3 / DRAM 575
  cycles); **L1 is not coherent across SMs**; 48 MB L2 as a benchmarking
  hazard; `cuobjdump -sass`; constant-memory replay.
- M5: 32 B sectors, 128 B lines, the sector-counting method, row pitch as a
  layout decision, 432 GB/s peak.
- M6: shared memory, static and dynamic (`extern __shared__`, third launch
  parameter), cooperative loading.
- M7: bank conflicts as replay; padding as an address-layout fix.
- M8: divergence, active masks, independent thread scheduling, `__shfl_*_sync`.
- M9: `__syncthreads()` semantics, the every-thread-must-arrive rule,
  `__threadfence()`, acquire/release, "`volatile` is not synchronization".

## Forward-reference debts PAID here

- **M2** promised `compute-sanitizer --tool racecheck` → Exercise 1, with real
  output and an explicit account of its blind spot.
- **M3** promised races and atomics, and the in-place cross-block race → the
  RMW race is developed in full; Exercise 1's global flush is exactly the
  cross-block case.
- **M4 and M5** both promised that L1's non-coherence across SMs is *why*
  device-scope atomics exist → "Where an atomic executes" in the Hardware
  Mental Model.
- **M4** promised that atomic contention is the same replay mechanism as
  constant-memory serialization → stated and tied to M7's bank conflicts as
  the third instance.
- **M9** owns barriers and fences; this module cites them and does not
  re-teach them, and uses them as the *contrast* that defines atomicity.

## Forward references made

- **Module 12 (reductions)** — fixed-order tree reduction as the
  deterministic alternative to `float atomicAdd`; this module deliberately
  ships no complete reduction.
- **Module 13 (scan)** — prefix sum as the only way to get *ordered*
  compaction; atomic tickets give uniqueness, never order.
- **Module 14 (histogram)** — the full privatized histogram: bin replication,
  more bins than fit in shared memory, coarsening. This module owns the
  mechanism and the traffic arithmetic only; no complete optimized histogram
  is shipped.
- **Module 19 (occupancy)** — shared-memory footprint per block capping
  resident blocks, named in Check-Your-Understanding Q3 as one of two
  mechanisms that make privatization slower.
- **Part XIV / Modules 41-42 (CUDA for AI, LLM inference kernels)** —
  non-reproducibility of training runs caused by
  `float`/`half` `atomicAdd`, named in `example01.cu` Part E and in the lesson.
- Locks/mutexes on GPUs mentioned in prose (Ex3 solution) as what lies beyond
  64-bit CAS, without a module number.

## Constraints observed

- No sm_90+ features; no thread-block clusters, no TMA.
- No complete histogram and no complete reduction (M12/M14 territory);
  all kernels are small and purpose-built.
- Timing per spec §12: all configurations back-to-back in one sweep,
  validation in a separate pass, min of 4 sweeps × 20 iterations,
  duration-based clock warm-up, ratios reported as the stable quantity,
  and every GB/s figure labelled DRAM or L2-resident.
- Nothing under `module10/` reveals an answer: Exercise 1's prediction is
  checked by FNV hash, not by a stored answer key.
- No binaries committed.
