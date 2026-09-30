# Module 17 manifest — Tiled GEMM

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files, all verified on RTX 3500 Ada / CUDA 13.2 (V13.2.51), `nvcc -arch=sm_89 -O3`,
warning-clean, binaries deleted:

| file | build | status |
|---|---|---|
| `module17/lesson.md` | — | — |
| `module17/example01.cu` | `-lcublas` | `OVERALL: PASS` |
| `module17/example02.cu` | | `OVERALL: PASS` |
| `module17/exercise01.cu` | | compiles with TODOs blank, prints `Set TODO 5 first.`, returns 0 |
| `module17/exercise02.cu` | | compiles with TODOs blank, prints `Set TODO 1 first.`, returns 0 |
| `module17/exercise03.cu` | | compiles with TODOs blank, prints the symptom and `OVERALL: FAIL` (debugging exercise — the permitted exception) |
| `solutions/module17/exercise01_solution.cu` | | `SCORE: 7/7`, `OVERALL: PASS` |
| `solutions/module17/exercise02_solution.cu` | | `SCORE: 6/6`, `OVERALL: PASS` |
| `solutions/module17/exercise03_solution.cu` | | `SCORE: 7/7`, `OVERALL: PASS` |
| `solutions/module17/exercise0{1,2,3}_solution.md` | — | — |
| `solutions/module17/check_your_understanding.md` | — | — |

Problem shape throughout: **M = 1035, N = 1541, K = 1063**
(`M % 8/16/32 = 3/11/11`, `N % 8/16/32 = 5/5/5`, `K % 8/16/32 = 7/7/7`).
`example01.cu` §E additionally re-measures on Module 16's shape
1027 × 2053 × 769 so the two modules' tables are directly comparable.

---

## Concepts taught

- **Tiled GEMM**: a `BM × BN` output tile owned by a block, marched along the
  contraction axis in steps of `BK`; the tile loop as a sequence of partial dot
  products.
- **The accumulator lives in a register across the whole tile loop.** Shared
  memory holds a tile and a tile is never a whole row of A, so the partial sum
  has nowhere else to live. Three corollaries: `acc` must not be `__shared__`;
  a thread with no output element still has to reach both barriers, so the guard
  goes on the store and not on kernel entry; the tile loop must run `ceil(K/BK)`.
- **The reuse formula**: FMAs per global load `= BM·BN·BK / (BK·(BM+BN)) =
  1/(1/BM + 1/BN)`. **`BK` cancels** — contraction depth buys shared-memory
  footprint and barrier amortisation, not reuse.
- **Two walls on tile size, and the one people expect is not the one that binds.**
  Shared-memory capacity allows `T ≤ 78` (default) / `T ≤ 112` (opt-in); the
  1024-threads-per-block limit with one output per thread forces `T ≤ 32` and
  therefore FMAs-per-global-load `≤ 16`. The scratchpad has 5× more room than
  the decomposition can use.
- **Three index mappings in one kernel and they are all different**: compute,
  A-load (`tx` walks `k`), B-load (`ty` walks `k`). Forced by layout — `k` is A's
  fast axis and B's slow axis — not by convention.
- **Cooperative loading when `BK ≠ BN`**: the flat strided loop of Module 6,
  with the fast axis chosen for coalescing. Measured cost of the general form
  vs a square specialisation: **5–15 %**.
- **Boundary handling by zero-fill**, and why it is exact rather than
  approximate (a zero term contributes nothing to the sum). Measured: every
  tiled kernel reports the same sampled error as the naive one.
- **The zero-fill redundancy**: an out-of-range A cell and an out-of-range B cell
  occur at the same `k`, so zeroing either side alone is sufficient and the
  kernel passes. Latent NaN hazard when `K < BK` (uninitialised shared memory
  rather than stale data).
- **The two barriers**: RAW needs both `__syncthreads()` guarantees (G1 + G2);
  WAR needs only G1, which is why it is the one you can design away.
- **Double buffering** as the correct removal of the WAR barrier: two tile
  buffers, one barrier per k-step, twice the shared memory. Measured
  1.007–1.009× at `T = 16`, 1.044–1.057× at `T = 32`. (Pays the M6 and M9 debts.)
- **Bank-conflict analysis of a tiled GEMM**: `As[ty][k]` is a broadcast (D = 1),
  `Bs[k][tx]` is unit-stride (D = 1), the cooperative store `As[ty][tx]` is
  D = 1. **A row-major tiled GEMM has no bank conflict to remove.** The only
  D = 32 access is the store into a *transposed* A tile, which padding does fix.
- **Padding the tile costs 1.4–1.5×**, not because of banks but because a pitch
  of `T+1` floats breaks the 16-byte alignment that lets `ptxas` merge four
  contiguous `As[ty][k]` reads into one `LDS.128`. 20 shared instructions become
  32. Spec §12 rule 11 with the sign reversed.
- **Shared-memory read bandwidth as a first-class ceiling**: 32 banks × 4 B =
  128 B/cycle/SM, measured **5.38–5.40 TB/s** scalar and **10.24–10.30 TB/s**
  vectorised (ratio 1.9, which is Module 7's two-cycle floor). A GEMM inner loop
  needs 8 B per FMA = **72 TB/s** at the FP32 ceiling, so a two-shared-loads-
  per-FMA kernel is **capped at 7.4–14.2 % of the FP32 ceiling** regardless of
  tile shape. Measured kernel: 9.4 %, between the two caps, as it must be.
- **The shared-loads-per-FMA probe** (Module 16 §E, one level down): 2 shared
  loads and R FFMAs. R = 1/2/4/8 → 9.4/18.9/36.5/62.7 % of the ceiling, agreeing
  with Module 16's `LDG` table to within 1 %. **The law is about memory
  instructions per arithmetic instruction, not about which address space.**
- **The Module 18 hand-off, derived as a number**: an `Rr × Rc` register tile
  gives `1/Rr + 1/Rc` shared loads per FMA; reaching 80 % of the FP32 ceiling
  needs ≤ 0.35, hence **`Rr ≥ 5.7`, i.e. a 6×6 or 8×8 register tile**.
- **Tile-shape selection**: shared bytes, the 1024 B driver reserve, the 128 B
  granularity, blocks/SM, occupancy — and the measured finding that **the
  FMAs-per-global-load ratio is anti-correlated with speed above 16×16**.
- **Registers are a fourth occupancy limiter** that Module 6's formula omits: the
  `16×32, BK=16` instantiation compiles to 44 registers instead of 40 and drops
  from 3 resident blocks to 2. Hand-compute to understand, call the API to know.
- **The awkward-size bug family**: truncating tile count, un-zero-filled tile
  cell, store guard against the wrong dimension. All three invisible on every
  square power-of-two problem, all three invisible to `memcheck` and `racecheck`,
  all three findable by one 17 × 31 × 23 test.
- **`initcheck --initcheck-address-space shared`** as the tool that sees an
  un-zero-filled tile cell (verified: reports `Uninitialized __shared__ memory
  read of size 16 bytes` — sixteen, because of the `LDS.128` merge).
- **A reproducible wrong answer is not a race** (restating M14's finding in a
  GEMM setting), and its converse: the no-WAR-barrier kernel is a race and is
  *also* reproducible enough to ship.

## CUDA API / intrinsics / syntax introduced

Nothing new. Module 17 is an integration module; every mechanism it uses was
introduced earlier.

Reused: `__shared__` static arrays with a template-parameter extent,
`__syncthreads()`, `fmaf`, `#pragma unroll`, template-parameter `__global__` +
host-side switch dispatch (M4's trick), `__restrict__`,
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaFuncGetAttributes().numRegs`,
`cudaDeviceGetAttribute`, `cudaEvent_t` timing, `cublasSgemm` with the swapped
column-major arguments (M16), `float4`, FNV-1a hashing of reference answers,
`setvbuf(stdout, NULL, _IONBF, 0)`, `nvcc -Xptxas -v`, `nvcc -cubin` +
`cuobjdump -sass`, `compute-sanitizer --tool initcheck --initcheck-address-space
shared`.

SASS vocabulary used: `LDS`, `LDS.128`, `STS`, `FFMA`, `LDG.E.CONSTANT`,
`BAR.SYNC.DEFER_BLOCKING`.

New-ish idiom, worth recording: **`base += 1` inside a shared-bandwidth
microbenchmark loop** to defeat loop-invariant hoisting. Without it the kernel
reports 40 TB/s, four times the bank array's theoretical maximum, because every
`LDS` has been hoisted out of the timing loop.

## Worked examples

| file | what it demonstrates |
|---|---|
| `example01.cu` | the reuse ledger at runtime; the square tiled kernel; the general rectangular tiled kernel with a flat cooperative load; correctness at an awkward size through M16's `gemmValidate()` on two α/β settings; an 11-shape tile sweep against naive and cuBLAS; the same three numbers on M16's shape; the three instruction-mix ratios as the M18 hand-off |
| `example02.cu` | three measured ceilings including **shared-memory read bandwidth, scalar and vectorised**; the bank-conflict degree table by enumeration; SASS instruction counts padded vs unpadded; 12 timed variants (pad 0/1 × row-major/transposed-A × T = 16/32, double-buffered, no-WAR-barrier) with a separate correctness pass; the shared-loads-per-FMA probe; the `Rr ≥ 5.7` derivation |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | fill-in + prediction (spec §6 types 1, 2) | 5 | Write the tiled GEMM: both cooperative loads, the tile-loop bound, both barriers, the store, and a bucket prediction of the speedup | **Three.** (a) The zero-fill is *arithmetically redundant* — fixing only the A line or only the B line makes the kernel PASS the full validator, because out-of-range A and B cells occur at the same `k`; only removing both fails. Verified by measurement, and `initcheck --initcheck-address-space shared` is the only tool that sees the one-sided version. (b) `K/T` instead of `ceil(K/T)` costs 7 of 1063 terms — a 0.66 % perturbation that a relative-to-\|C\| tolerance can miss and the `gamma_K·S` rule rejects at 145×. (c) The missing WAR barrier is **faster** (1.01–1.07×) and wrong in only 1.3 % of elements; at `T = 16` the **Freivalds check passes it** (0.59) and only the sampled check catches it (58.6). It also fires at `T = 8`, contradicting the "you only need it for big blocks" folklore. |
| `exercise02.cu` | analysis + design (spec §6 types 5, 6) | 5 | Shared bytes, blocks/SM, the two instruction-mix ratios, bank degrees for four access patterns, then choose a tile shape and predict what padding does | **Three.** (a) `BK` cancels out of the reuse formula, so the natural "deeper tile = more reuse" instinct is wrong. (b) The FMAs-per-global-load ratio is **anti-correlated with speed above 16×16** — 32×32 has twice the reuse and is 20 % slower — so a reader who trusts their own model picks the wrong tile. (c) The padding prediction: every access is already degree 1 and `max(2,D)` makes degree 2 free, so the analysis says "bucket 2, no effect"; the measurement is **0.65–0.71, a 40 % loss**, and the only way to see it coming is `cuobjdump -sass` and the lost `LDS.128`. Bonus, unscored: the reader's three-limiter occupancy model disagrees with the API on exactly one shape, because of **registers**. |
| `exercise03.cu` | debugging + design (spec §6 types 3, 5) | 5 | Three defects that are all invisible on square power-of-two sizes; diagnose from a 7-option list, fix, then design the minimal test-size set (scored on arithmetic-property coverage, with the required mask stored only as an FNV-1a hash so the exercise file does not reveal the diagnosis) | **Three.** (a) Four of the seven diagnosis options are visible on a square problem and must be ruled out; option 7 (grid sized with a truncating divide) is *also* hidden at 512³ and is distinguished from option 3 only by **which** elements survive the `+inf` poison, not how many. (b) `memcheck` and `racecheck` are both clean, and the failure is fully deterministic — the tools that would normally be reached for are all blind. (c) TODO 5's work budget forces the realisation that two of the three defects need the *same* property of (M,N,K), so **one 17 x 31 x 9 shape suffices** - 0.0000028x the work of the failing shape. Two of the six listed properties are distractors that any awkward shape supplies for free (`M % TILE != 0`, `N % TILE != 0`); a third, `M > N`, flips defect 3 from an unwritten-columns failure to an out-of-bounds-write failure. |

## Scoring

- `exercise01`: 7 points — 3 tile sizes × 2 datasets (6) + the prediction bucket (1).
- `exercise02`: 6 points — 4 hashed analysis TODOs + tile choice ranking in the top 3 of 8 measured shapes + the padding bucket.
- `exercise03`: 7 points - 3 diagnosis codes + the repaired kernel on 3 fixed shapes + 3 required arithmetic properties covered by the reader's size set.

`OVERALL: PASS` requires full marks in every case.

## Measured results

| quantity | measured |
|---|---|
| naive GEMM, 1035×1541×1063 | 1311–1332 GFLOP/s (M16 reported 1275–1348) |
| **tiled 16×16, best** | **1703–1722 GFLOP/s** |
| tiled / naive | **1.29–1.32×** |
| tiled as % of cuBLAS | **17.5–20.2 %** |
| tiled as % of the FP32 ceiling | **9.4 %** |
| cuBLAS SGEMM | **8497–9695 GFLOP/s** (M16 reported 8150–8705; see observation 8) |
| tile ranking (× naive) | 8×8 1.04–1.05, 16×16 **1.29–1.30**, 32×32 1.18–1.19 |
| rectangular tiles | 16×16 BK=32 1.19–1.20×, 32×16 BK=16 1.03–1.09×, 16×32 BK=16 0.89–0.93×, 64×16 BK=16 0.77–0.83×, 32×32 BK=16 0.81–0.82× |
| square specialisation vs general loader, same tile | **1.05–1.15×** |
| **padding a row-major tile** | **0.674–0.706× (T=16), 0.688–0.690× (T=32) — a LOSS** |
| padding a transposed-A tile | 1.053–1.062× (T=16), 1.174–1.178× (T=32) — a win |
| transposed-A staging vs row-major, both unpadded | 0.63–0.64× (T=16), 0.52× (T=32) |
| double buffering (2 barriers → 1) | 1.007–1.009× (T=16), 1.044–1.057× (T=32) |
| WAR barrier deleted, no second buffer | 1.010–1.014× faster, **20 972 / 39 199 wrong elements of 1 594 935** |
| FP32 FFMA ceiling | 17 627–18 122 GFLOP/s, implied clock 1.758–1.770 GHz |
| **shared read, scalar `LDS`** | **5.38–5.40 TB/s** |
| **shared read, `LDS.128`** | **10.24–10.30 TB/s** (ratio 1.90) |
| shared bandwidth needed at 2 loads/FMA and the FP32 ceiling | **72.0–72.5 TB/s** |
| **implied cap on any 2-shared-loads-per-FMA kernel** | **7.4–14.2 % of the FP32 ceiling = 1326–2574 GFLOP/s** |
| shared-loads-per-FMA probe, R = 1/2/4/8 | 9.4 / 18.9 / 36.5 / 62.7 % of the ceiling |
| (M16's `LDG` probe for comparison) | 9.5 / 19.6 / 38.2 / 62.2 % |
| **register-tile size needed for 80 % of the ceiling** | **`Rr ≥ 5.7`, i.e. 6×6 or 8×8** |
| SASS, `gemmTiled<16,0>` accumulation body | 4 `LDS.128` + 16 `LDS` + 16 `FFMA` |
| SASS, `gemmTiled<16,1>` (padded) | 32 `LDS` + 16 `FFMA` |
| SASS, `gemmTiled<32,0>` / `<32,1>` | 8 `LDS.128` + 32 `LDS` + 32 `FFMA` / 64 `LDS` + 32 `FFMA` |
| SASS, `gemmTiledAT<16,*>` (transposed A) | 32 `LDS` + 16 `FFMA` — **no `LDS.128` at any padding** |
| registers, `gemmTiled<16,16,16,0>` | 40; `<16,32,16,0>` uses **44**, dropping 3 blocks/SM to 2 |
| validator headroom, correct kernel | sampled 0.0211–0.0345 of the `gamma_K·S` budget; `gamma_K = 6.336e-5` at K = 1063 |
| validator vs planted defects | `K/T`: 145–152×; both zero-fills removed: 22.7–438×; no WAR barrier: 38.8–74.4×; `col < M`: 523 710 non-finite |

**Reproducibility note (spec §12 rule 5).** Absolute GFLOP/s move ±3 % between
back-to-back runs and can collapse by **2.7×** (naive 1325 → 485, cuBLAS 8755 →
3322) if a sweep is launched immediately after another GPU-heavy process; two
such runs were observed while authoring and discarded. All *ratios* in the table
above reproduced across ≥ 3 clean runs. The `16×16 BK=16` / `16×16 BK=32`
ordering is not stable (they are within 5 %), which is why Exercise 2 accepts any
tile choice ranking in the top 3 of 8 measured shapes.

## Assumed from earlier modules

- **M1**: SM structure, 4 processing blocks, 1536 threads / 24 blocks per SM,
  non-migrating blocks, the placement gate reserving threads + warps + registers
  + shared memory atomically, waves.
- **M3**: the linearization rule `tid = tx + bx·ty` and therefore what a warp is
  in a `(T,T)` block; ceil-divide; bounds guards.
- **M4**: L1 40.5 / L2 241.3 / DRAM 575 cycles; the 128 KB unified L1+SMEM;
  `-Xptxas -v`; `cuobjdump -sass`; template-parameter dispatch.
- **M5**: 32 B sectors; the counting procedure applied to both cooperative loads;
  `float4` alignment.
- **M6**: shared memory as a scratchpad not a cache; the K/H reuse argument and
  the honest "the caches got there first" correction; cooperative loading with
  load-mapping ≠ compute-mapping; the two barrier hazards by name; the
  capacity→occupancy table with the **1024 B driver reserve**; barrier cost as
  convoying; `LDS`/`STS` vs `LDG`/`STG`.
- **M7**: 32 banks × 4 B, `bank = (addr/4) % 32`, degree = max distinct words per
  bank, **broadcast is free**, **cost is `max(2,D)` so degree 2 is free**, the
  two-cycle floor on a conflict-free warp access, padding's `gcd(P,32)=1` rule
  and its occupancy cost, the 128 B allocation granularity, and rule 11's
  warning that the compiler vectorises out from under an analysis.
- **M8**: warp-uniform branches cost one predicate (`beta == 0`).
- **M9**: `__syncthreads()` as two separable guarantees; **WAR needs only G1**;
  exited threads are subtracted from the barrier count and corrupt silently on
  sm_89; double buffering as the way to halve the barrier count.
- **M11**: compulsory traffic; the discipline of a measured ceiling in the same
  sweep; `#pragma unroll 1` and its converse as anti-optimization instruments.
- **M12**: 1500 ms warm-up; the streaming ceiling kernel.
- **M14**: "a reproducible wrong answer is not a race"; `initcheck
  --initcheck-address-space shared` as the tool for uninitialised shared memory.
- **M15**: the general padding rule `gcd(P, 128/E) = 1`; that padding-vs-swizzle
  is only decidable on an LSU-bound kernel.
- **M16**: the GEMM problem statement and `lda/ldb/ldc`; the `beta == 0`
  contract; the compulsory/requested traffic ledger; the FP32 and DRAM ceiling
  measurements and the `cudaDevAttrClockRate` warning; the x→col mapping; the
  loads-per-FMA law and the 6.4–6.5 threshold; `gemmValidate()` **reused
  verbatim**; the cuBLAS swapped-argument call; 1027×2053×769.

## Forward-reference debts PAID here

- **M6 → M15/16/17**: "the explicit destination of the K/H argument." Paid, and
  paid honestly: the K/H ratio for a GEMM tile is `T` with no halo tax, the
  global load count really does fall by `T/2`, and the measured win is **1.30×**.
  M6's closing claim that "the transformative wins need register blocking too" is
  now a measured number rather than an assertion.
- **M6 → M17 (software pipelining)**: "double-buffering named as the alternative
  to the WAR barrier." Paid: `gemmTiledDB` in `example02.cu`, one barrier per
  k-step instead of two, measured 1.007–1.057×, with the barrier counts confirmed
  in the SASS (`BAR.SYNC.DEFER_BLOCKING` 2 → 1).
- **M9 → M17**: "double-buffering to remove the WAR barrier is introduced here as
  the germ of the pipelining there." Paid, together with M9's reason: the WAR
  hazard needs only guarantee G1, which is what makes it removable.
- **M16 → M17**: "the owner of shared-memory tiling, cooperative tile loading and
  double buffering", and "the hand-off is explicit: tiling changes the cost of an
  operand read, not the count of memory instructions per FMA." Paid, and
  quantified: 8.00 FMAs per global load, 0.50 per shared load, 0.47 per memory
  instruction of any kind.
- **M16 → M17/M18**: "`gemmValidate()` — Modules 17 and 18 must reuse this."
  Reused verbatim in `example01.cu`, `exercise01.cu` and `exercise03.cu`,
  including the two-dataset conditioning requirement and reporting the headroom
  rather than a boolean.
- **M7 → GEMM**: M7 stated that the padding-vs-swizzle answer "flips in GEMM,
  where shared-memory capacity limits tile size." Partially addressed: at *this*
  module's tile sizes capacity does **not** limit anything (8 KB of a 48 KB
  budget — the thread count limits it), so the premise of M7's claim does not
  hold until register blocking arrives. Recorded as a correction for M18.

## Forward references made

- **Module 18 (advanced GEMM)** — owns register blocking / thread coarsening as
  a *reuse* mechanism, vectorised `float4` operand loads, `cp.async` and real
  software pipelining, warp-level tiling, and Tensor Cores. This module hands it
  three specific things: (1) the ratio `1/Rr + 1/Rc` and the derived requirement
  **`Rr ≥ 5.7`**; (2) the measured `R = 1/2/4/8` table showing 62.7 % of the FP32
  ceiling is reachable at 0.25 shared loads per FMA; (3) the observation that
  **both** padding and an XOR swizzle destroy the operand contiguity that
  `LDS.128` needs, which neither M7's rule nor M15's rule accounts for, and
  which a `float4`-staged tile must handle by making the pitch odd *in units of
  `float4`*.
- **Module 19 (occupancy)** — named for the full occupancy story, and owed one
  specific item: **registers as the fourth placement-gate limiter**, which this
  module measured biting on `gemmTiled<16,32,16,0>` (44 registers, 3 blocks → 2).
- **Module 21 (roofline)** — named by M16; this module adds a third ceiling
  (shared-memory bandwidth) that a two-axis roofline cannot express.
- **Module 36 (cuBLAS / libraries)** — cuBLAS is used only as a ruler.
- **Module 43 (CUTLASS)** — named as the home of real swizzle layouts and the
  `cp.async` / Tensor-Core pipelines.

## Cross-module observations for the index

1. **Module 16's "6.4–6.5 FMAs per global load" needs restating.** The probe that
   produced it varied arithmetic against a fixed pair of `LDG`s, and the law it
   measured is about **memory instructions per arithmetic instruction on the LSU
   path**, not about the global address space. This module reran the identical
   experiment with `LDS` and got 9.4 / 18.9 / 36.5 / 62.7 % against M16's 9.5 /
   19.6 / 38.2 / 62.2 % — agreement to within 1 %. The index should record the
   quantity as **"FMAs per operand-fetch instruction"**. Without that
   restatement M16's threshold appears to be *met* by a tiled GEMM that reaches
   9.4 % of peak, which reads as a contradiction and is not one.
2. **New measured ceiling for the index's §6b table: shared-memory read
   bandwidth**, 5.38–5.40 TB/s scalar and 10.24–10.30 TB/s vectorised, ratio
   1.90. This is the resource that binds Modules 17 and 18 and it is not
   currently in the table.
3. **Padding folklore now has four data points, not three**: M7 padding beats
   swizzle by 19 % (LSU-bound microbenchmark); M15 padding ties swizzle
   (DRAM-bound transpose); **M17 padding loses to no padding by 1.4–1.5×
   (shared-bandwidth-bound GEMM, no conflicts to remove, `LDS.128` merge
   destroyed)**. The unifying rule to record: *pad only after measuring a degree
   ≥ 4 conflict, and check the SASS for a vector merge you are about to break.*
4. **Spec §12 rule 11 should gain its converse.** The rule currently warns that
   compiler vectorisation can make a measured conflict penalty *smaller* than
   predicted. This module found the opposite failure: an analysis that correctly
   concludes "no conflicts, padding is a no-op" is wrong by 40 % because padding
   destroys a vector merge. Same fix — read the SASS — but the symptom is
   inverted, and an author who has only internalised the first form will not
   look.
5. **Spec §12 should record the loop-invariant-hoisting failure mode.** A shared-
   bandwidth microbenchmark whose addresses do not change with the timing-loop
   index reports **40 TB/s**, four times the bank array's theoretical maximum. A
   physically impossible result is a signal, and every microbenchmark should be
   sanity-checked against a hardware bound before its number is used.
6. **Module 6's occupancy formula is incomplete and should say so.**
   `min(1536/threads, 102400/(bytes+1024), 24)` omits registers, which the
   placement gate checks (M1 listed all four). Measured here: two kernels with
   identical thread counts and identical shared memory differ in resident blocks
   (3 vs 2) purely because one compiles to 44 registers and the other to 40.
8. **cuBLAS's range in the index should be widened.** M16 recorded 8150–8705
   GFLOP/s. This module measured 8497–9695 GFLOP/s for `cublasSgemm` on the same
   1027 x 2053 x 769 shape across clean runs, i.e. up to 11 % above the top of
   M16's range, and 3322 GFLOP/s in a throttled session. The honest figure for
   the index is **8150–9700 GFLOP/s = 45–53 % of the FP32 ceiling**, with the
   usual caveat that a throttled session can halve it.

7. **M7's "the answer flips in GEMM because capacity limits tile size" does not
   apply to M17.** At one output per thread the tile is limited to 32×32 by the
   1024-thread cap and uses 8 KB of a 48 KB budget; capacity is nowhere near
   binding. The premise only becomes true once register blocking decouples tile
   size from thread count, so **the debt is genuinely M18's**, as M15 also
   concluded.

## Constraints observed

- No register blocking, no thread coarsening, no `float4` operand loads as an
  optimization, no `cp.async`, no warp-level tiling, no Tensor Cores. `float4`
  appears only in the warm-up and shared-bandwidth microbenchmark kernels, where
  it is the instrument and not the technique.
- Double buffering appears as the *correct removal of the WAR barrier* (the M6
  and M9 debts) and is measured; software pipelining with asynchronous copies is
  explicitly deferred to M18.
- cuBLAS is a ruler only; M36 owns it.
- No sm_90+ features.
- `ncu` not used (`ERR_NVGPUCTRPERM`); every conflict degree is derived by
  enumeration and every claim is a timing ratio or a SASS instruction count.

## Known issues / honesty notes

- The DRAM ceiling measured inside `example02.cu` reads **278 GB/s**, well below
  Module 12's 410.5–410.7 GB/s, because the stream kernel runs in the middle of a
  compute-heavy session and the power manager trades the SM clock away (M16's
  documented effect). The file says so in a comment and in prose. It does not
  affect any conclusion — the tiled GEMM is nowhere near DRAM-bound — but the
  line must not be quoted as this machine's streaming ceiling.
- The `LDS.128` bandwidth figure divides by the clock implied by the *FFMA*
  ceiling to produce a B/cycle/SM number of ~146, which exceeds the 128 B/cycle
  the bank array can physically deliver. The explanation is that the shared
  kernel almost certainly clocks higher than the FFMA kernel; the ratio between
  the scalar and vector rows (1.90) is the quantity that is trustworthy, and the
  program says so.
- Two of roughly eight `example01.cu` runs produced absolute figures 2.7× low
  across *every* configuration including cuBLAS, immediately after another
  GPU-heavy process. Discarded, and recorded above. Ratios were unaffected.
- `exercise02.cu`'s "fastest measured shape" alternates between `16×16 BK=16` and
  `16×16 BK=32` across runs. The 10 % acceptance tolerance covers it; a tighter
  tolerance would make the exercise fail intermittently on a correct answer.
- The shipped `exercise03.cu` deliberately exits with `OVERALL: FAIL` and a
  non-zero status when the TODOs are blank. This is the spec §5 exception for a
  debugging exercise; it does not crash, does not fault, and prints the symptom.
