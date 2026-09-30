# Module 14 manifest — Histogram

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module14/lesson.md`, `module14/example0{1,2}.cu`,
`module14/exercise0{1,2,3}.cu`.
Solutions: `solutions/module14/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module14/check_your_understanding.md`,
`solutions/module14/MANIFEST.md`.

All `.cu` verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2 V13.2.51, RTX 3500 Ada,
driver 596.71), warning-clean. `example02.cu` additionally needs
`-std=c++17 -Xcompiler /Zc:preprocessor` for `<cub/cub.cuh>` under MSVC — the
same pair Modules 9, 12 and 13 needed. Both examples print `OVERALL: PASS`; all
three solutions print `OVERALL: PASS` (9/9, 5/5, 4/4). All three shipped
exercises compile with TODOs blank and exit gracefully. No binaries committed.

---

## Concepts taught

- **The histogram as a memory-access problem.** Compulsory traffic is 1N; the
  floor at the course's measured 410.5–410.7 GB/s ceiling is 0.49 ms for 192 MiB.
  A naive implementation runs 100–200× above it, at 0.4–1.3% of the ceiling.
  The module is the distance between those two numbers.
- **Global atomic cost tracks distinct 32 B SECTORS per warp, not distinct
  addresses.** Measured with everything else held constant: 32 distinct bins in
  4 sectors = 25.0 Gatomic/s; the same 32 distinct bins in 32 sectors =
  3.13 Gatomic/s; **7.99×**. Permuting which lane gets which address: **1.00×**.
  This is a **refinement of Module 10's rule**, and it explains why M10's
  `i & mask` benchmark (23.35 Gatomic/s at K=256) is an optimistic bound for
  any K > 8 and a real histogram of random bytes measures 4.0 Gatomic/s.
  Module 5's sector-counting method, applied to atomics.
- **The input distribution is a first-class parameter of the performance model.**
  Four distributions (uniform / zipf / same-bin / clustered), identical
  instruction counts, **2.06× spread** on the naive kernel, driven by two
  distinct mechanisms (K=1 slice serialization vs warp-uniform addresses).
  Every measurement in the module names its distribution.
- **Privatization done properly**: strided cooperative zeroing, the two barriers
  and why they are different guarantees, shared-atomic accumulation, the
  `if (s[b])` flush guard (worth 8× on a sparse distribution, measured as the
  entire difference between two columns of the *same* kernel).
- **Coarsening is the dominant lever, and it has no famous name.** The textbook
  one-element-per-thread privatized histogram converts N contended global
  atomics into N shared atomics **plus N global ones**: measured 8.07 ms against
  1.10 ms for the same kernel at a machine-sized grid — **7.3× at 256 threads,
  14.4× at 128 threads** — with a broad optimum (any grid from 3 K to 12 K
  blocks within 12%) and a turn-up below one full wave.
- **Bin replication**: replica-major layout `s[r*nBins+b]` vs replica-minor
  (`gcd(R,32)`-way bank conflict), per-warp vs per-lane-group replica
  assignment, the `R·nBins·4 B` footprint and its occupancy consequence.
  **Documented null result: 1.00× on all four distributions at R ≤ 8, and
  0.83× at R = 16 purely from 6 → 5 blocks/SM.** Per-lane-group replication
  measures 0.91–0.94× (it defeats `ATOMS.POPC.INC`).
- **Why replication is worthless here**: `ATOMS.POPC.INC.32` already removes
  intra-warp same-address contention; the residual inter-warp contention is
  ~50× cheaper than the memory stream feeding it. Named the case where it *does*
  matter (weighted histograms, where `ATOMS.POPC.INC` cannot be used: measured
  2.79–2.96 ms vs 1.10 ms).
- **What actually binds the privatized histogram: load width.** Scalar
  `unsigned char` stream 152 GB/s; `uchar4` stream 320 GB/s; adding one shared
  atomic per element to either costs **1.7%**. `LDG.E.U8` moves 32 B per warp
  instruction, `LDG.E` on `uchar4` moves 128. Vectorizing takes the kernel from
  46% to **99–100% of the measured ceiling on every distribution**.
- **The flatness of a finished histogram**: v4 measures 0.5005–0.5028 ms on
  inputs that make the naive kernel vary by 2×. A finished histogram has no best
  case and no worst case.
- **Bin counts beyond shared memory.** The availability ladder with measured
  blocks/SM: 256→6, 1024→6, 4096→5, 8192→3, 16384→1 (99 KB opt-in via
  `cudaFuncAttributeMaxDynamicSharedMemorySize`), 65536→impossible.
  **Shared privatization reaches 94% of ceiling at ONE resident block per SM** —
  a bandwidth-bound kernel needs requests in flight, not warps (M11's Little's
  Law).
- **Multi-pass over bin windows costs exactly `P × floor`** — measured at 100%,
  50%, 12% of ceiling for P = 1, 2, 8. No hidden constant. Therefore: last
  resort, largest window you can hold, occupancy be damned.
- **Global-memory privatization with G copies + a fold kernel** as the strategy
  for bin counts that do not fit, with its cost model and its **L2 cliff**:
  G = 256 at 65,536 bins is a 64 MB privatized array against a 48 MB L2 and
  measures **0.10× of the naive kernel**. Module 4's benchmarking hazard
  reappearing as a design rule.
- **At a high bin count with a flat distribution there is nothing to win.**
  S0 at 65,536 uniform bins is already 34–35% of ceiling and the best strategy
  in the table beats it by 1.06×. The gentler form of M10's "privatization is a
  0.22× loss at K = 4096".
- **A skewed distribution is the EASY case for a privatized histogram** and the
  hard case only for the naive one: Exercise 2's fast kernel measures 49% of
  ceiling on the hot input and 35% on the flat one.
- **`cub::DeviceHistogram::HistogramEven`** measured: the two-call
  query/allocate/run protocol, `num_levels = nBins + 1` (boundaries, not
  buckets), 31,457,791 B of temp storage at 65,536 bins (CUB is doing global
  privatization and you can read it off the allocation). **The hand-written
  privatized kernel beats it at every bin count** (1.8× at 256, 2.4–3.4× in the
  middle) — not a defect in CUB, which handles arbitrary ranges, sample types
  and multi-channel images.
- **Padding bins apart is the wrong move for a histogram.** M10's fourth
  contention-reduction move increases the sector count per warp. Named
  explicitly as the one place a previous module's optimization is actively
  wrong here, with the mechanism.
- **`compute-sanitizer --tool racecheck` does not report a missing barrier when
  the conflicting access is a shared ATOMIC.** Verified A/B on the identical
  kernel: `atomicAdd` → 0 hazards; plain `+=` → 2 hazards with 8,990,966 and
  1,003,566 instances. A **second racecheck blind spot** after M10's global-RMW
  finding, and it sits exactly inside the kernel shape this module teaches.
- **`initcheck --initcheck-address-space shared` is the tool that finds an
  unzeroed private bin** (38,361 errors, named to the source line). The symptom
  polarity table: too low+catastrophic = lost update; too low by a multiple of
  65536 = overflow; too high and varying = uninitialized bins; too high by a
  constant = double-processed tail.
- **Packed 16-bit bin counters** as a real 2×-memory technique with a capacity
  invariant, and the derivation `n/grid + blockDim <= 65535` that makes the
  invariant unreachable. **The packed representation's 2× saving is paid for in
  coarsening depth.**
- **A reproducible wrong answer is not a race** (M10's diagnostic rule, applied
  in reverse to identify integer overflow).

## CUDA API / intrinsics / syntax introduced

- `uchar4` and `uint4` as *input* vector types (M5/M11 introduced `float4`);
  `LDG.E.U8.CONSTANT` vs `LDG.E.CONSTANT` in SASS
- `cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376)`
  and `cudaDeviceGetAttribute(..., cudaDevAttrMaxSharedMemoryPerBlockOptin, 0)`
  — named in M6, used for real here at 16,384 bins
- `cub::DeviceHistogram::HistogramEven` (and `HistogramRange`, `MultiHistogramEven`
  named); `num_levels = nBins + 1`
- `compute-sanitizer --tool initcheck --initcheck-address-space shared` used as
  the primary diagnostic (M9/M12 named it; first load-bearing use)
- `__match_any_sync` + `__ffs` + `__popc` as a *shipped* aggregation idiom
  (M10 introduced it as a measurement)
- `cudaMemsetAsync` inside a timed region
- SASS read and quoted: `ATOMS.POPC.INC.32`, `RED.E.ADD.STRONG.GPU`,
  `LDG.E.U8.CONSTANT`, `LDG.E.CONSTANT`, `BAR.SYNC.DEFER_BLOCKING`,
  and the *absence* of `VOTEU.ANY` in every histogram kernel
- FNV-1a hashing of a multi-part answer so the key can live only under
  `solutions/` (M10/M11/M12 convention)

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | 192 MiB of bytes, 256 bins, 4 distributions. **A**: 5-rung ladder + streaming ceiling, 24 configurations in one rotated 24-sweep, all 20 (kernel, distribution) pairs validated. **B**: R ∈ {1,2,4,8,16} × 4 distributions with shared footprint, blocks/SM and flush atomics printed — the documented null result. **C**: coarsening curve, 7 grids from 786,432 to 192 blocks with `flush/N` alongside. **D**: the sector experiment — 32 distinct bins per warp in 4 vs 32 sectors, with a lane-order scramble as the control; 7.99× and 1.00×. Re-warms 1500 ms before B, C and D and says in its own output that absolute ms across parts are not comparable. |
| `example02.cu` | 2^26 uint32 keys, bin count 256 → 65,536, two distributions, five strategies (global / shared-privatized / multi-pass windows / global-privatized+fold / `cub::DeviceHistogram`) + ceiling, with a 1500 ms re-warm and a rotated 12-sweep per bin count. Prints the strategy-availability table (smem/blk, blocks/SM, window × passes, copies × footprint) before any timing. Reports % of a ceiling measured in the same sweep, because the ceiling itself drifts 0.65 → 0.83 ms across the run. |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Optimization (§6 type 4) + prediction (type 6) + design ×2 | 5 | Build the four rungs above a global-atomic 256-bin byte histogram; 6 kernels × 3 distributions in one rotated sweep | **`BLK = 128` with `nBins = 256`**, so `if (threadIdx.x < nBins)` zeroing leaves half the bins holding the previous block's garbage — counts too **high**, varying, and **only `initcheck --initcheck-address-space shared` finds it**. TODO 3's reflexive `ceil(n/BLK)` grid produces 403 M flush atomics against N = 201 M and is **14.4× slow** — it removes the contention and doubles the traffic. TODO 4 is a **null result the reader must measure**: R = 1..8 is 1.00× on every distribution including `same-bin`, and R = 16 is 0.83× from one lost resident block; per-lane-group replication is 0.91–0.94× because it defeats `ATOMS.POPC.INC`; replica-minor layout is a `gcd(R,32)`-way bank conflict. TODO 5 changes zero atomics and is worth **2.19×** — the largest remaining factor is the load width, not the algorithm. TODO 1's trap is predicting three different numbers for v4 (they are all the same) instead of for v0 (which varies 2×). Perf gates are **ratios between the reader's own kernels**, because the ceiling slot drifts with thermal state. |
| `exercise02.cu` | Design (§6 type 5) + performance reasoning (type 6) | 4 | 65,536 bins, 64 MB of scratch, a requirement instead of a technique; gated on **both** a hot and a flat distribution | The gate is two-sided on purpose: ≥ 5.00× on hot **and** ≥ 0.90× on flat. The reflexive answer (multi-pass shared windows, 8 passes) measures **2.17× / 0.46×** and fails both. TODO 3's `G` has a wrong answer on each side: G = 8 is 1.6× worse than G = 16, and **G = 256 is a 64 MB privatized array against a 48 MB L2 and measures 0.10× of naive** — ten times slower than doing nothing. TODO 1's trap is predicting a large win on the flat column; the true answer is **0.99×**, because the naive kernel is already at 35% of ceiling there and what costs it the other 65% is sector spread, not contention. The winning answer needs **two** levers composed (global privatization for cross-block, `__match_any_sync` for intra-warp) — the first workload in the course where neither alone suffices, because there is no shared-memory step to make the intra-warp case free. |
| `exercise03.cu` | Debugging (§6 type 3) + prediction + design | 4 | Packed 16-bit bin counters, three defects, two symptoms; `hist_broken` always runs so the sanitizers have something to chew on even with TODOs blank | Defect B (**missing barrier after zeroing**) is reported by **nothing**: verified A/B, racecheck gives `0 hazards` with `atomicAdd` and `2 hazards / 8,990,966 instances` with a plain `+=` on the identical kernel. The canonical racecheck bug, invisible precisely because a correct privatized histogram must use a shared atomic. Defect C (16-bit wrap) is **reproducible**, which rules out a race and is the diagnostic. Defect A fires only when `nBins/2 > blockDim.x`, i.e. at 1024 bins and not at 256; a `dirty_shared` kernel is run first or a fresh context hides it. TODO 4 **bans widening the counters**, forcing the derivation `n/grid + blockDim <= 65535` → grid ≥ 1,029, and the conclusion that the packed representation's 2× saving is paid for in coarsening depth (4.3× more flush atomics). Predicting `2` (racecheck) for defect B is the reasonable inference from M10 and is wrong; no partial credit. |

Scoring: Ex1 `score: 9/9` (4 kernels exact × 3 distributions, 2 ratio gates,
3 octave-scored predictions); Ex2 `5/5` (exactness on both distributions, 2
speedup gates, 2 octave-scored predictions); Ex3 `4/4` (exactness on 4 scenarios
× 3 repeats, packed-representation check, derived-bound check, FNV-hashed
3-part diagnosis). `OVERALL: PASS` requires full marks.

## Measured results recorded in this module

| Quantity | Measured |
|---|---|
| Streaming ceiling, 1500 ms warm-up, 192 MiB of bytes | 0.4991–0.4997 ms = **403.0–403.4 GB/s (93.3% of 432)** |
| Streaming ceiling, 1500 ms warm-up, 256 MiB of uint32 | 0.6532–0.6539 ms = **410.2–410.9 GB/s (95.0–95.1%)** — reproduces M12 exactly |
| Ceiling under sustained load (power cap) | 0.62 ms (322 GB/s) → 2.12 ms (126 GB/s) |
| 256-bin naive global, uniform | 50.40 ms, **4.0 Gatomic/s**, 1% of ceiling |
| 256-bin naive global, zipf | 76.76 ms, 2.6 Gatomic/s |
| 256-bin naive global, same-bin | 103.89 ms, 1.9 Gatomic/s |
| 256-bin naive global, clustered | 61.04 ms, 3.3 Gatomic/s |
| **naive spread across distributions** | **2.06×** |
| Shared priv, 1 elem/thread (grid = N/BLK), uniform / same-bin | 8.07 / 1.97 ms — same kernel, 4.1× apart, entirely the `if (s[b])` flush guard |
| Shared priv + coarsening (240 blocks) | 1.093–1.096 ms, **46% of ceiling**, flat across all four distributions |
| + replication R = 8 | 1.092–1.095 ms — **1.00×** |
| + `uchar4` loads | 0.5002–0.5028 ms, **99–100% of ceiling**, flat |
| Ladder total, v0 → v4 | **100.2× / 153.3× / 207.6× / 122.0×** (uniform / zipf / same-bin / clustered) |
| Step v1 → v2 (the grid), BLK=256 / BLK=128 | **7.35× / 14.43×** |
| Step v3 → v4 (vectorize) | **2.19×**, zero change in atomic count |
| Replication R = 1/2/4/8/16 (all distributions) | 1.00 / 0.99 / 0.99 / 0.99 / **0.83×** |
| Per-lane-group replication R=8 vs R=1 (clustered / same-bin) | 0.91× / 0.94× |
| Weighted histogram (defeats `ATOMS.POPC.INC`), uniform / clustered / same-bin | 1.10 / 2.89 / 2.78 ms |
| Coarsening curve, grid 786432 → 192 (uniform) | 8.28 / 2.07 / 1.24 / **1.06** / 1.18 / 1.41 / 1.53 ms |
| **Sector experiment**: 32 bins/warp, 4 sectors vs 32 sectors | 8.05 ms (25.0 Gatomic/s) vs 64.29 ms (3.13 Gatomic/s) — **7.99×** |
| Same experiment, lane-order scramble control | **1.00×** |
| `uchar4` stream / `uchar` stream, no atomics | 0.629 / 1.325 ms — **2.11×** |
| `uchar` stream + 1 shared atomic per element | 1.347 ms — atomics cost **1.7%** |
| `uchar4` + 4 atomics vs `uchar4` + 1 atomic per 4 B | 0.6313 / 0.6334 ms — atomic count is free |
| Global atomic throughput vs grid (240 … 786,432 blocks) | **50.4 ms at every grid** — not an MLP effect |
| Shared privatization, % of ceiling at 256/1024/4096/8192/16384 bins | 100 / 100 / 100 / 99 / **94%** |
| blocks/SM vs bin count (256 threads) | 6 / 6 / 5 / 3 / 1 / 0 at 256 / 1024 / 4096 / 8192 / 16384 / 65536 bins |
| Multi-pass, % of ceiling at P = 1 / 2 / 8 | 98–100 / 50 / 12% — exactly `1/P` |
| 65,536 bins, uniform: naive / global-priv / multi-pass / CUB | 1.96 / 1.84 / 5.30 / 4.30 ms |
| 65,536 bins, zipf: naive / global-priv / multi-pass / CUB | 13.61 / 2.32 / 5.29 / 3.25 ms |
| **Global privatization at 65,536 uniform bins** | **1.06× — nothing to win** |
| Ex2 hot input: G = 8 / 16 / 64 / **256** | 3.41 / 5.44 / 6.27 / **2.58×** (64 MB > 48 MB L2) |
| Ex2 flat input, G = 256 | **0.10× of naive** |
| Ex2 final (G = 16 + warp aggregation), hot / flat | **9.68× / 0.99×** |
| Warp aggregation on top of global privatization (hot) | 5.44× → 7.59×, i.e. **+1.40×** |
| `cub::DeviceHistogram` temp storage at 65,536 bins | 31,457,791 B |
| Hand-written shared privatization vs CUB | 1.8× at 256 bins, 2.4–3.4× at 1024–16384 |
| Ex3 broken, uniform/256 · skewed/256 · uniform/1024 · skewed/1024 | 1.000× · **0.251×** · 16.6× · 22.6× |
| Ex3 racecheck: shared `atomicAdd` vs plain `+=`, identical missing barrier | **0 hazards** vs **2 hazards, 8,990,966 + 1,003,566 instances** |
| Ex3 initcheck (shared) on the broken kernel | 38,361 uninitialized-read errors |
| Ex3 derived safe grid at n = 2^26, BLK = 256 | 1,029 blocks (65,474 worst-case per-bin count) |

Absolute ms on this laptop part move by up to 4× with thermal state. Every
claim above is a ratio, a % of a ceiling measured in the same sweep, or is
quoted with its thermal state.

## Assumed from earlier modules

- **M1**: 40 SMs, warp = 32, blocks are indivisible and non-migrating, waves,
  Little's Law, 432 GB/s, 48 MB L2.
- **M2**: `nvcc -arch=sm_89`, launch syntax, `CHECK` / `CHECK_KERNEL`,
  `cudaEvent_t` timing.
- **M3**: grid-stride loops, bounds guards, ceil-divide, the linearization rule.
- **M4**: L1 is not coherent across SMs and L2 is the device coherence point;
  L2 241 / DRAM 575 cycles; **48 MB L2 as the benchmarking hazard, reused here
  as a design rule**; `cuobjdump -sass`.
- **M5**: 32 B sectors and 128 B lines; the sector-counting method (**extended
  to atomics here**); `float4`/`uchar4` vectorized loads; coalescing.
- **M6**: shared memory, dynamic `extern __shared__` and the third launch
  parameter, cooperative loading, the capacity → blocks/SM table
  (6 ≤ 12288 B, 5 @ 16384, 3 @ 25600, 2 @ 49152, 1 @ 65536), the 1024 B driver
  reserve, the 99 KB opt-in.
- **M7**: `bank = (addr/4) % 32`, `D = gcd(stride,32)`, the `max(2,D)` cost law —
  used to reject the replica-minor layout.
- **M8**: warps, divergence, `_sync` intrinsics and why the mask exists,
  `__popc`/`__ffs`.
- **M9**: the two guarantees of `__syncthreads()`; fences are not barriers;
  the kernel boundary as the only grid-wide barrier;
  `initcheck --initcheck-address-space shared`; synccheck detects nothing.
- **M10**: **everything about atomics.** RMW races; atomics execute at the L2;
  `RED` vs `ATOM`; `ATOMS.POPC.INC.32`; the contention curve and the
  addresses-per-warp rule (**refined here to sectors-per-warp**); the
  privatization traffic ledger `n → n shared + g·b global`; where privatization
  and warp aggregation stop paying; `__match_any_sync` aggregation; compiler
  aggregation needs provable warp-uniformity; racecheck's blind spots.
  **Not re-taught anywhere; cited by name throughout.**
- **M11**: compute the floor before writing the kernel; `x floor` and
  % of a measured ceiling as the reporting units; coarsening; vectorization as a
  latency/issue optimization; Little's Law on the memory system (cited to
  explain 94% of ceiling at 1 block/SM); the 1500 ms warm-up finding.
- **M12**: the integer tree fold in Exercise 2's `fast_fold` is labelled as M12's
  reduction and not developed; "the tools check what you did, never what you
  meant"; the 410.5–410.7 GB/s ceiling.
- **M13**: the CCCL build flags; `setvbuf(stdout, NULL, _IONBF, 0)` as the house
  convention; the two-call CUB query/allocate/run protocol.
- **Spec §12** throughout: 1500 ms duration-based warm-up, all configurations
  back-to-back, **rotated sweep order with SWEEPS >= NCFG**, auto-scaled
  iteration counts targeting ~10 ms segments, min-of-N, validation in a separate
  pass, buffers ≥ 4× the L2, ratios as the stable quantity.

## Forward-reference debts PAID here

- **M10 → M14**, in full and by name: *"Module 14 builds the full privatized
  histogram — bin replication to spread residual contention, handling more bins
  than fit in shared memory, and the coarsening ladder."* All three delivered,
  measured, and one of them (replication) reported as a **null result with the
  mechanism**, which is a stronger payment than a fabricated win would have been.
- **M10's "where privatization stops paying" table** is the quantitative
  foundation and it **held up, with one correction and one softening**:
  - *Correction:* M10's Gatomic/s figures are optimistic for K > 8 because its
    `i & mask` address pattern is the densest possible sector packing. The
    sectors-per-warp rule is the general statement; M10's rule is the special
    case where addresses and sectors coincide.
  - *Softening:* M10 measured privatization at **0.22× (a 4.5× loss) at
    K = 4096**. At histogram scale with a correct grid the same physics gives
    **1.06× at K = 65536 uniform** — a no-op, not a loss. M10's loss came from
    `g·b > n` with a one-element-per-thread grid; coarsening removes that term
    and what is left is simply no win. Both are correct and the reconciliation
    is Exercise 2's whole subject.
- **M5 owed** vectorized loads applied at scale; `uchar4` is the punchline of
  the ladder and the sector-counting method is extended to atomics.
- **M6's blocks/SM-versus-shared-bytes table** is used as a design input three
  times (replication factor, bin-count availability, the 99 KB opt-in) and is
  forwarded to M19 for the formal treatment.
- **M9's `initcheck --initcheck-address-space shared`** gets its first
  load-bearing use; Exercise 3 is built on it.

## Forward references made

- **Module 15 (transpose)** — deliberately absent. No transpose appears.
- **Module 19 (occupancy)** — the `R × nBins × 4 B` → blocks/SM arithmetic and
  the 16,384-bin one-block-per-SM case are done by hand and flagged as M19's.
- **Module 23 (Nsight Compute)** — named as where
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and the L2 sector
  counters would settle the Part D mechanism directly. `ncu` remains unavailable
  (`ERR_NVGPUCTRPERM`); every effect here is constructed and measured directly.
- **Module 30 (warp-level primitives)** — owns `__match_any_sync`; Exercise 2
  uses it because the algorithm needs it and does not develop it.
- **Module 36 (CUB / Thrust)** — owns the library treatment;
  `cub::DeviceHistogram` is measured here as a reference point and the honest
  finding (hand-written beats it 1.8–3.4×, and why that is not a criticism) is
  recorded.
- **Sorting** named once, as the other route to a histogram, without a module
  number (it remains reserved in the cross-module index).

## Cross-module observations for the index

1. **New rule for §6b:** *global atomic throughput tracks distinct 32 B sectors
   per warp.* Measured 7.99× for a pure sector-count change with the atomic
   count, the distinct-address count and the lane ordering all held constant.
   Module 10's §6b entries ("max contended atomic penalty 47.9× at K=1",
   "uncontended global atomic ≈ cost of a plain store") remain correct; a third
   entry should record that a *real* data-dependent address pattern at K = 256
   measures 4.0 Gatomic/s against M10's 23.35, and why.
2. **Second racecheck blind spot.** M10 recorded that racecheck is blind to
   global RMW. This module adds: **racecheck does not report a shared-memory
   write-versus-atomic hazard**, verified A/B on one kernel. Worth an index
   entry because it applies to every privatized-accumulator kernel in Part IV.
3. **The 410.5–410.7 GB/s ceiling reproduces independently** (410.2–410.9 here
   on a different kernel and a different buffer size), confirming §6b.
   Also confirmed: the power cap. Example 1's own Parts B/C/D and Example 2's
   per-bin-count columns show the ceiling degrading from 0.65 ms to 0.83 ms and
   as far as 2.12 ms under sustained streaming, which is why **both examples
   re-warm for 1500 ms between parts and say so in their output**. Future
   modules with multi-part harnesses should do the same.
4. **`memcheck`'s benign `Resetting device while there are still other users`
   warning** (M12's note) reconfirmed; it is the *only* thing memcheck reports
   for all three of Exercise 3's defects.
5. **Nothing contradicts Modules 1–13.** The one apparent contradiction
   (M10's 0.22× vs this module's 3.97× for shared privatization at ~4096 bins)
   is a difference in the grid, is reconciled explicitly in the lesson (§7,
   Prediction P2) and in Exercise 2's solution notes, and neither claim is
   adjusted.

## Constraints observed

- **No transpose** (M15), **no GEMM**, no sorting. Reduction (M12) appears only
  as Exercise 2's integer fold, labelled as M12's and not developed; scan (M13)
  does not appear at all.
- Atomics are **used constantly and never re-taught**; M10 is cited by name
  eleven times in the lesson.
- No sm_90+ features. `ATOMS.POPC.INC.32` and the 99 KB shared opt-in are
  labelled **ARCHITECTURE-SPECIFIC**; the portable statements are given
  alongside.
- Themes: image/byte histogram, weighted category histogram and packed-counter
  histogram are new. None of §6's used themes is repeated; transpose, GEMM,
  sorting, sparse formats, large-radius convolution and n-body remain reserved.
- Every timed figure follows spec §12 including the **1500 ms** warm-up and
  `SWEEPS >= NCFG` rotation; every `% of peak` over 100% would be explained (none
  occurs); every absolute figure is accompanied by its thermal state.
- Nothing under `module14/` reveals an answer: Exercise 3's diagnosis is scored
  against an FNV-1a hash, and Exercises 1 and 2 score predictions against
  measurements taken in the same run.
- No binaries committed.

## Known issues / honesty notes

- **`ncu` unavailable** (`ERR_NVGPUCTRPERM`). The sector claim in lesson §2
  rests on a controlled A/B of four kernel instantiations with everything but
  the sector footprint held constant, not on
  `lts__t_sectors_op_atom.sum`. The command line and metric names are given;
  no counter output is quoted anywhere.
- **`synccheck` reports nothing** on Exercise 3's missing barrier, as it has
  reported nothing in every module since M9. Documented as theory per the
  standing rule; no attempt was made to fix it.
- **Bin replication produced no win anywhere in this module.** It is shipped,
  measured on four distributions and five values of R, and reported as a null
  result with the mechanism (`ATOMS.POPC.INC.32`) and with the case where it
  would pay (weighted histograms) measured separately. Module 10 named
  replication as this module's deliverable; delivering it honestly meant
  delivering a negative.
- **Exercise 1's performance gates are ratios between the reader's own kernels,
  not percentages of the ceiling.** An earlier draft gated v4 at ≥ 90% of the
  ceiling and failed on a thermally loaded machine, because the ceiling kernel's
  own slot in the rotated sweep landed in a cooler window than v4's. The
  ceiling is still printed (and the file notes that a settled machine gives
  99–100%), but it is not scored. This is spec §12 rule 5 enforced rather than
  quoted.
- **Exercise 2's `flat` gate is "no worse than 0.90×", not "at least 1.05×".**
  The correct answer measures 0.95–1.00× there, and pretending otherwise would
  have required choosing a distribution that made the module's own conclusion
  false.
- The `clustered` distribution in `example01.cu` has runs of 8,192 identical
  values, which is warp-uniform but *not* L2-hostile; a genuinely sorted input
  would additionally be perfectly bin-local. That variant is suggested in the
  lesson rather than shipped, to keep the configuration count at 24.
