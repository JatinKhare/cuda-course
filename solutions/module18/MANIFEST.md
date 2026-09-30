# Module 18 manifest — Advanced GEMM Optimization

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module18/lesson.md`, `module18/example0{1,2}.cu`,
`module18/exercise0{1,2,3}.cu`.
Solutions: `solutions/module18/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module18/check_your_understanding.md`,
`solutions/module18/MANIFEST.md`.

All `.cu` verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2, V13.2.51, RTX 3500
Ada), warning-clean. Only `example01.cu` links cuBLAS
(`nvcc -arch=sm_89 -O3 -lcublas -o example01.exe example01.cu`); every other
file builds with the plain course line. Both examples print `OVERALL: PASS`; all
three solutions print `OVERALL: PASS` (10/10, 10/10, 10/10). All three shipped
exercises compile with TODOs blank and exit gracefully
(`Set TODO 5 (PREDICTIONS) first.`, `Set TODO 1 first.`, `Set TODO 1 first.`).
`compute-sanitizer --tool memcheck` is clean on `exercise03_solution.exe`.
No binaries committed.

**Module 18 closes Part V.** It assumes Module 16 (problem statement, traffic
ledger, loads-per-FMA bound, `gemmValidate`) and Module 17 (shared-memory
tiling: the tile loop, cooperative loading, the two barriers, boundary handling,
tile-size selection, bank analysis of the tiles) and **re-teaches neither**.
Module 17's kernel is reproduced in three files as a labelled baseline only.

---

## Concepts taught

- **The two-level reuse law.** `FMAs per global load = BM·BN·BK/(BM·BK+BK·BN) =
  BM·BN/(BM+BN)` — `BK` cancels — and `FMAs per shared read = TM·TN/(TM+TN)`.
  **The same function, one level of the hierarchy apart.** Module 16's "6.4–6.5
  needed, 0.50 supplied" is true twice; the block tile fixes level 1 and the
  register tile fixes level 2. **PORTABLE CUDA CONCEPT.**
- **Register tiling / thread coarsening for reuse** as a rank-1 update: `TM+TN`
  inputs produce `TM·TN` products, so `TM·TN` accumulators in registers is the
  decomposition that fixes the shared-level ratio.
- **Square thread tiles are optimal by AM–GM**: for fixed `R = TM·TN`,
  `R/(TM+TN)` is maximised at `TM = TN = √R`. Hence `16×4` (3.2) is worse than
  `8×8` (4.0) at identical register cost.
- **6.5 is unreachable as a scalar count** (`TM = TN = 13` → 169 accumulators)
  and reachable as an *instruction* count: `TM·TN/((TM+TN)/4) = 16.0` for
  `8 × 8` read with `float4`. **What matters is instructions, not values.**
- **Coarsening for reuse vs coarsening for MLP** — the debt Module 11 recorded.
  Same code shape, different resource: M11 buys outstanding requests (worthless
  at full occupancy, a regression at C = 8/16); M18 buys operands reused from
  registers (costs occupancy and is worth 5× anyway). Full contrast table.
- **The three coordinate systems** (grid → block tile → thread tile) and the
  fourth, independent one: the **cooperative-load mapping**, which has nothing
  to do with `(tRow, tCol)` — Module 6's "load mapping ≠ compute mapping" at
  block-tile scale.
- **Three boundaries, and what the third one actually protects.** `rowBase+r ≥ M`
  and `colBase+n ≥ N` are correctness *and* safety; `kt+c ≥ K` on the A side is
  **arithmetically redundant** (the B-side guard zeroes the other factor) and is
  required purely for **memory safety** — measured: the unguarded kernel passes
  every numerical check and `compute-sanitizer` reports `Invalid __global__ read`.
- **The register/occupancy cliff, constructed and measured.**
  `__launch_bounds__(threads, minBlocks)` as an instrument: the same source at
  five occupancies, with `ptxas` spilling whatever will not fit.
  **100 % occupancy measured 14.1–18.7× slower than 25–33 % occupancy.**
- **Occupancy is not monotone in the right direction**: 25 % → 33 % is a win,
  33 % → 67 % is a 9× loss. A tuner that hill-climbs on occupancy walks off the
  cliff.
- **The spill cliff is about *which* values spill, not about the first byte.** A
  192–552 B spill that buys a resident block can win (`__launch_bounds__(128,4)`
  measured 1.12× *faster* than the unconstrained build); a spill that reaches
  the accumulators is 14×. Module 4's "local memory is DRAM", in the innermost
  loop.
- **The transposed operand tile**, and the correct reason for it:
  **vectorizability**, not bank conflicts. `As[k][m]` makes the `TM` values a
  thread needs adjacent, so 8 `LDS` become 2 `LDS.128`.
- **Padding an operand tile: the pad size is pinned from two directions at
  once.** For a transposed A tile at pitch `BM` the store is `D = 8`
  (`bank = r mod 32`); pad by 4 gives `bank = (4c + r) mod 32`, `D = 1`; pad by
  1 gives `(c + r) mod 32`, `D = 4` — a partial fix that measures like a fix
  (Module 7 Exercise 2's `PAD_PITCH 40` trap restated). And the pitch must be a
  multiple of 4 or the `float4` reads lose 16-byte alignment. **Both constraints
  select 4 and nothing else.**
- **⚠️ A correction to Module 7's `max(2, D)` law.** The law says a 2-way
  conflict is free on Ada. That is a statement about **4-byte** accesses: a
  32-lane 4-byte read asks for 128 B and the bank array delivers 128 B/cycle, so
  it is under-subscribed and has a spare cycle. An `LDS.128` is phase-split into
  4 phases of **8 lanes**, and one phase asks for exactly 128 B — fully
  subscribed. **On 16-byte shared reads, `cost ∝ D` with no floor of 2.**
  Isolated and measured (see "Cross-module observations" below).
- **The `BN/TN ≥ 16` design rule** that follows: once the thread tile is read
  with `float4`, fewer than 16 threads along the N axis makes the B-tile read a
  2-way conflict in every phase. No resource counter reports this constraint.
- **Vectorized global loads are a null result in GEMM, and misaligned ones are
  illegal.** The global path is 18 instructions out of ~300 per k-tile, so
  cutting it by 4 is invisible (measured 0.94–1.01× with padded leading
  dimensions); and `float4` on `A[row*lda + k]` requires `lda % 4 == 0`, which
  `lda = 769` is not — a `misaligned address` fault, not a slowdown. **The
  leading dimension is a performance parameter, not just bookkeeping** (M16
  taught it as the latter).
- **Double buffering / software pipelining**: two shared buffers remove the
  **write-after-read** hazard (Module 6's name for the second barrier) *by
  construction*, so **one barrier per k-tile suffices**, and the global loads for
  tile k+1 are issued before the FFMAs on tile k. The proof that one barrier is
  enough is given.
- **⚠️ Double buffering is a measured LOSS on this GPU** at every competitive
  tile shape (0.81–0.92×), and a +1.1 % win at the one shape (`4×4`) where it
  costs neither a register granule nor a block. Reason: the 48 MB L2 holds the
  whole working set, so the hidden latency is a 241-cycle L2 hit, and 16–24
  resident warps already cover it. Registers — the binding resource — are traded
  for latency tolerance the kernel does not need. `cp.async` (**Module 32**)
  removes exactly that cost and is why real pipelines are 3–4 stage.
- **Warp-level tiling** named as the third level of the hierarchy
  (`ThreadblockShape → WarpShape → InstructionShape`), introduced conceptually,
  **not implemented**, forward-referenced to CUTLASS (Module 43).
- **Split-K**, properly: the decomposition parallelises M and N and leaves K
  sequential, which fails for small M,N and large K (M = N = 128, K = 65536 is
  one block on 40 SMs). Atomic split-K (cheap, non-deterministic — Module 10's
  ten-bit-patterns-in-ten-runs) versus workspace split-K (deterministic,
  `4·S·M·N` bytes plus a second kernel, which is why cuBLAS reports a workspace
  requirement). The trade is arithmetic intensity against parallelism.
- **Occupancy by hand on sm_89**, all four limits: registers allocated **per
  warp in granules of 8 per thread**, shared memory with the 1024 B per-block
  driver reserve and 128 B granularity, 1536 threads/SM, 24 blocks/SM. Checked
  against `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on 34 distinct kernels
  across the module with no disagreement.
- **Ada's operand-reuse cache**, read off the SASS: 192 of 256 `FFMA`s carry
  `.reuse`, because a rank-1 update reads the same `rM[i]` across `TN`
  consecutive instructions. **ARCHITECTURE-SPECIFIC.**
- **The honest ceiling.** Final kernel at **91–108 % of `cublasSgemm`** on
  1027×2053×769 and **93 %** at 2048³, 41–46 % of the measured FP32 ceiling.
  What is missing and who owns it: Tensor Cores (M33–34), `cp.async`/TMA (M32),
  warp specialization, per-shape tuned tile dispatch, split-K.
- **The methodological point of the module**: a rule of the form "technique X
  beats technique Y" is a rule about **which resource is binding**, and it
  inverts when that changes. Four of the five candidate optimizations in
  Exercise 1 are null results; the SASS `FFMA` density is the only number that
  tracked performance throughout.

## CUDA API / intrinsics / syntax introduced

- **`__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`** — the
  two-argument form. Introduced here as a *measurement instrument* (forcing
  register counts to construct the spill cliff) as well as a tuning knob.
- `cudaFuncGetAttributes` → `numRegs`, **`localSizeBytes`** (the spill figure at
  runtime), `sharedSizeBytes`, `maxThreadsPerBlock`.
- `_Pragma("unroll")` inside a function-like macro (the double-buffer
  load/store macros in `example01.cu`).
- SASS read and quoted: `LDS.128`, `STS`, `LDG.E.CONSTANT`,
  `BAR.SYNC.DEFER_BLOCKING`, `FFMA` with the **`.reuse`** operand flag,
  `IMAD.WIDE`, `LOP3.LUT`.
- Reused, not introduced: `float4` and reinterpreting casts (M5),
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (M7),
  `cublasSgemm` with the swapped-argument row-major call (M16), `fmaf`,
  `cudaEvent_t` timing, `setvbuf(stdout, NULL, _IONBF, 0)`, FNV-1a answer
  hashing (M11/M12), `std::numeric_limits<float>::infinity()` as a
  writtenness sentinel (M16).

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | **The ladder.** A: the two-level ledger printed at runtime (0.50 / 16.00 / 64.00 / 0.89 / 4.00 / 16.00) with the AM–GM argument. B: six kernels — naive, M17 block tile, 1-D register tile, 2-D register tile, transposed+padded A tile, double-buffered — plus `cublasSgemm`. C: every rung validated with M16's `gemmValidate` verbatim, C prefilled with `+inf`. D: all eight timed back-to-back in one rotated sweep (`SWEEPS = NCFG`) after 1500 ms streaming + 500 ms compute warm-up, with registers / blocks-per-SM / occupancy printed next to the throughput so the occupancy inversion is visible in the same table. |
| `example02.cu` | **Resources.** A: the 12-point `TM × TN` sweep at 256 threads with registers, spills, shared bytes, blocks/SM, occupancy and throughput. B: the occupancy cliff — the same 8×8 kernel at `__launch_bounds__(256, n)` for n = 1,2,3,4,6, 7267 → 390 GFLOP/s. C: Module 7's open question — plain vs pad-by-4 vs XOR swizzle, at `BK = 8` and `BK = 32`. D: blocks/SM computed by hand from all four limits and checked against the occupancy API, showing **registers bind in every one of the 23 configurations**. |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Fill-in + design + prediction (§6 types 1, 5, 6) | 5 | Write the `TM × TN` register-tiled kernel — index arithmetic, cooperative staging of a transposed A tile, the register inner product, the guarded epilogue — validated at four shapes and gated at 4.2× the M17 baseline | **The trap is TODO 1, and four of the five things you would expect to be traps are null results.** Measured, all correct, all validated: `tRow`/`tCol` swapped = **0.72×** (fails the gate) because a warp then presents 16 addresses 32 B apart to the `As` `LDS.128`, a 2-way conflict which is *not* free on a 16-byte access; loader mapping `idx % BM` (32 sectors instead of 4) = **0.98×**, because the global path is 18 of ~300 instructions; inner loop not hoisted = **1.05× faster**, because nvcc CSEs all 52 redundant reads; A-tile pad removed = 0.96×. And removing the `kt+c < K` guard = **1.05× and a correct answer**, because the B-side guard zeroes the product — it is `compute-sanitizer` and nothing else that catches the out-of-bounds read. |
| `exercise02.cu` | Predict + design + analysis (§6 types 2, 4, 6) | 5 | Occupancy by hand vs the CUDA API on 11 kernels at **128** threads/block; the two-level law; two predictions; a cost model that picks a tile from the resource table with no timings; the first spilling launch bound | The withheld fact in TODO 1 is that **registers are allocated per warp in granules of 8 per thread** — a per-thread model scores 9/11 and looks nearly right. TODO 4 cannot be solved with occupancy alone (the 100 % row is at 47 % of the winner), with reuse alone (the max-reuse row is at 77 %), or with their product; and `TM=8/TN=4` vs `TM=4/TN=8` are **identical in every column of the table** and 1.48× apart, so the model is forced to use `BM·BN/(BM+BN)` as well. TODO 5's cliff is not at the first spilled byte: `(128,4)` spills 80 B and is *faster* than the unconstrained build. |
| `exercise03.cu` | CPU→GPU / design from scratch (§6 type 5) | 5 | Tile hierarchy, resource ledger committed before the code, **the entire kernel**, the launch, and a ceiling prediction; four shapes and a 4.5× gate | `LEDGER_SMEM` is compared against `cudaFuncGetAttributes` **exactly**, so a reader who computes `8·128·4 + 8·64·4 = 6144` has forgotten the pad — or has not padded, in which case the number matches and the kernel is 5 % slower. The 37×53×11 shape is a 1×1 grid in which 219 of 256 threads write nothing and every guard fires. `float4` on the global loads faults (`lda = 769`), which is not a performance bug. The design space has a cliff at `TM·TN = 256` (spills, 253 GFLOP/s — slower than naive) and a 1.6× trap at `BN/TN < 16` that no resource counter reports. |

Scoring: Ex1 10 points (4 correctness incl. the `+inf` writtenness check,
2 performance gate at 4.2×, 4 predictions); Ex2 10 points (3 occupancy model
11/11, 1 reuse law hash, 2 predictions, 2 cost model within 15 %, 2 spill
threshold); Ex3 10 points (4 correctness, 2 ledger, 2 gate at 4.5×,
2 prediction). `OVERALL: PASS` requires full marks.

## Measured results recorded (RTX 3500 Ada, CUDA 13.2)

### The ladder, M = 1027, N = 2053, K = 769

| rung | GFLOP/s | × cuBLAS | % of 18 000 ceiling |
|---|---|---|---|
| v0 naive (M16) | 1302–1345 | 0.16–0.18 | 7.2–7.5 % |
| v1 block tile 32×32 (M17) | 1372–1543 | 0.17–0.21 | 7.6–8.6 % |
| v2 + 1-D register tile, TM = 8 | 4278–4465 | 0.53–0.60 | 23.8–24.8 % |
| v3 + 2-D register tile, 8×8 | 7033–7802 | 0.86–1.04 | 39.1–43.3 % |
| v4 + transposed, padded A tile | 7373–8138 | 0.91–1.08 | 41.0–45.2 % |
| v4b 8×4 (`BM128 BN64`), the winner | 7399–8336 | 0.91–1.08 | 41.1–46.3 % |
| v5 + double buffering | 5779–6144 | 0.73–0.80 | 32.1–34.1 % |
| `cublasSgemm` | 7258–8370 | 1.00 | 40.3–46.5 % |

At **2048³**: naive 1279, M17 tile 1469, best 9335, cuBLAS 10032 → **93 % of
cuBLAS**, 51.9 % of the ceiling.

### Registers, occupancy, spills (256 threads/block, BK = 8)

| TM × TN | regs | spill B | smem B | blk/SM | occ | FMAs/read | GFLOP/s |
|---|---|---|---|---|---|---|---|
| 1×1 | 38 | 0 | 1152 | 6 | 100.0 % | 0.50 | 1060 |
| 2×2 | 40 | 0 | 2176 | 6 | 100.0 % | 1.00 | 3432 |
| 4×1 | 39 | 0 | 2688 | 6 | 100.0 % | 0.80 | 3140 |
| 4×4 | 64 | 0 | 4224 | 4 | 66.7 % | 2.00 | 6578 |
| 8×1 | 63 | 0 | 4736 | 4 | 66.7 % | 0.89 | 3223 |
| 8×2 | 64 | 0 | 5248 | 4 | 66.7 % | 1.60 | 5808 |
| **8×4** | 80 | 0 | 6272 | 3 | **50.0 %** | 2.67 | **7399** |
| 4×8 | 84 | 0 | 6272 | 2 | 33.3 % | 2.67 | 4725 |
| 8×8 | 124 | 0 | 8320 | 2 | 33.3 % | 4.00 | 7016 |
| 16×8 | 210 | 0 | 12416 | 1 | 16.7 % | 5.33 | 4817 |
| 8×16 | 216 | 0 | 12416 | 1 | 16.7 % | 5.33 | 3650 |
| 16×16 | 64 | **2272** | 16512 | 4 | 66.7 % | 8.00 | **253** |

### The occupancy cliff (same 8×8 kernel, 256 threads)

| `__launch_bounds__` | regs | spill B | blk/SM | occ | GFLOP/s |
|---|---|---|---|---|---|
| `(256,1)` | 124 | 0 | 2 | 33.3 % | 7267 |
| `(256,2)` | 124 | 0 | 2 | 33.3 % | 7297 |
| `(256,3)` | 80 | 192 | 3 | 50.0 % | 2011 |
| `(256,4)` | 64 | 552 | 4 | 66.7 % | 685 |
| `(256,6)` | 40 | 896 | 6 | **100.0 %** | **390** |

**100 % / 33 % = 18.7×.** At 128 threads (Exercise 2) the same construction
gives 14.1–14.6×, and there `(128,4)` — 80 B spilled, 4 blocks — is 1.12×
*faster* than the unconstrained build.

### Module 7's question: padding vs XOR swizzle in GEMM (8×8, 256 threads)

| A-tile layout | regs | smem B | blk/SM | limiter | GFLOP/s | vs plain |
|---|---|---|---|---|---|---|
| `BK=8` plain (D = 8) | 124 | 8192 | 2 | registers | 6587 | 1.00× |
| `BK=8` **pad by 4** (D = 1) | 124 | 8320 | 2 | registers | **7114** | **1.08×** |
| `BK=8` XOR swizzle (D = 2) | 128 | 8192 | 2 | registers | 5761 | 0.87× |
| `BK=32` plain | 196 | 32768 | 1 | **registers** | 4111 | — |
| `BK=32` pad by 4 | 196 | 33280 | 1 | **registers** | 5294 | — |
| `BK=32` XOR swizzle | 226 | 32768 | 1 | **registers** | 5193 | — |

**Padding wins, by 1.23× over the swizzle at `BK=8` and 1.02× at `BK=32`.**
Module 7's prediction (swizzle wins in GEMM, because capacity limits tile size)
**does not hold on Ada for fp32 GEMM**, and both halves of the argument fail:
the swizzle's cost is *larger* than M7 measured, because it makes the A-tile
reads not provably contiguous and so blocks the compiler's contraction into
`LDS.128` (48 scalar `LDS` + 20 `LDS.128` + 42 `LOP3` versus 32 `LDS.128`); and
the capacity premise is false, because **registers bind in every configuration
tested**, including the 32 KB one.

### The `LDS.128` 2-way-conflict finding (isolating experiment)

`BM = 128, BN = 64, BK = 8`, 256 threads; only the thread tile varies.

| | `TM=8 TN=4` | `TM=4 TN=8` |
|---|---|---|
| accumulators / regs / smem / blk / occ / `TM·TN/(TM+TN)` / `BM·BN/(BM+BN)` | 32 / 80 / 6272 / 3 / 50 % / 2.67 / 42.67 | **identical** |
| inner-loop SASS between barriers | 286 instr, 256 `FFMA`, 24 `LDS.128`, 2 `BAR.SYNC` | **identical** |
| `BN/TN` (threads along N) | 16 | 8 |
| B-read bank degree per `LDS.128` phase | D = 1 | **D = 2** |
| predicted cycles/`k` (phases × D) | 12 | 20 → 1.67× |
| measured, 1027×2053×769 | **8186** | **5095** → 1.61× |
| measured, 1024×1024×8192 (epilogue control) | 7224 | 5381 → 1.34× |

### Double buffering

| kernel | regs | smem B | blk/SM | GFLOP/s |
|---|---|---|---|---|
| 8×8 single-buffered, padded | 124 | 8320 | 2 | 7114 |
| 8×8 double-buffered | 144–150 | 16640 | **1** | 5779 |
| 8×8 double-buffered, `__launch_bounds__(256,2)` | 128 | 16640 | 2 (56 B spill) | 6344 |
| 8×4 single-buffered | 80 | 6272 | 3 | 7399 |
| 8×4 double-buffered | 107 | 12544 | 2 | 6833 |
| 4×4 single-buffered | 64 | 4224 | 4 | 6578 |
| **4×4 double-buffered** | 63 | 8448 | 4 | **6651 (+1.1 %)** |

### Instruction mix

| kernel | inner-loop body | `FFMA` density |
|---|---|---|
| M16 naive | `LDG, LDG, IMAD.WIDE, FFMA` | 25 % |
| M18 8×4 register tile | 282–286 instructions: 256 `FFMA`, 24 `LDS.128`, 2 `BAR.SYNC` | **90.8 %** |
| per k-tile, outside the barriers | 6 `LDG.E.CONSTANT`, 6 `IMAD.WIDE`, 6 `STS` | — |
| `.reuse` operand flags | 192 of 256 `FFMA` | — |

### Other measured results / null results

| quantity | measured |
|---|---|
| Register allocation granule, sm_89 | **8 per thread**, per warp; verified against the occupancy API on 34 kernels |
| `float4` global staging with padded `lda`/`ldb` | **0.94–1.01×. Null result.** |
| `float4` global staging with `lda = 769` | **`misaligned address` fault** |
| Explicit `float4` shared reads vs scalar (transposed layout) | within noise — nvcc contracts the scalar reads into the same 24 `LDS.128` |
| Un-hoisted inner loop (64 source reads vs 12) | **1.05× faster**; nvcc CSEs all 52; identical SASS |
| Loader mapping `idx % BM` (32 sectors vs 4) | **0.98×. Null result.** |
| A-tile pad removed (`AP = BM`) | 0.96× |
| `kt+c<K` guard removed on the A side | **correct answer**, 1.05×, and `Invalid __global__ read` under memcheck |
| Shared memory as the occupancy limiter, fp32 GEMM, ≤48 KB static | **never observed** across 9 deliberately constructed candidates; registers bind or tie every time, because `ptxas` spends registers in proportion to `BK` to pipeline the unrolled loop (68 regs at `BK=32` → 128 regs at `BK=128` for the same 4 accumulators) |
| `BM=64 BN=32 BK=128` | does not compile: `uses too much shared data (0xc800 bytes, 0xc000 max)` |

## Assumed from earlier modules

- **M1**: 40 SMs, 4 processing blocks with a 16384-register slice each, 1536
  threads/SM, warps, waves and tails.
- **M2/M3**: launch syntax, `CHECK` idiom, the linearization rule
  `tid = threadIdx.x + blockDim.x·threadIdx.y` and **Exercise 1's axis trap**,
  which this module reproduces a third time at 1.39×.
- **M4**: **local memory is DRAM** and spills are priced accordingly; L1 40.5 /
  L2 241.3 / DRAM 575 cycles; `-Xptxas -v`; `cuobjdump -sass`.
- **M5**: sectors and the sector-counting procedure (applied to the loader, with
  a null result); `float4` alignment requirements; coalesced stores.
- **M6**: shared memory, cooperative loading, **load mapping ≠ compute
  mapping**, the capacity→occupancy arithmetic with the **1024 B driver reserve
  and 128 B granularity**, and the naming of double buffering as the WAR-hazard
  alternative.
- **M7**: 32 banks × 4 B, the degree-counting method, **`cost ∝ max(2, D)`**
  (extended here), broadcast, **phase splitting for 8- and 16-byte accesses**
  (the mechanism behind this module's central bank finding), padding vs XOR
  swizzle and the `PAD_PITCH 40` partial-fix trap, and the open question this
  module settles.
- **M8**: warp-uniform branches are predicated (`beta == 0`).
- **M9**: the two guarantees of `__syncthreads()`, the uniformity rule, and the
  naming of double buffering as the germ of software pipelining.
- **M10**: float `atomicAdd` non-determinism, cited for atomic split-K.
- **M11**: compulsory traffic; **coarsening for MLP**, and its measured
  worthlessness at full occupancy, which this module contrasts with coarsening
  for reuse.
- **M16**: the GEMM problem statement, `lda/ldb/ldc`, the `beta == 0` contract,
  **`gemmValidate` reused verbatim in three files**, the `gamma_K · S` tolerance
  law, the two-dataset conditioning discipline, the cuBLAS column-major swap,
  the 6.4–6.5 vs 0.50 loads-per-FMA result, and **"a cache reduces the cost of a
  memory instruction; it does not reduce the number of them."**
- **M17**: shared-memory tiling in full — the tile loop, cooperative loading,
  the two barriers, boundary handling, tile-size selection, and the bank
  analysis of the tiles. Reproduced only as a labelled baseline.
- Spec §12 throughout: 1500 ms streaming + 500 ms compute warm-up, rotated
  back-to-back timing with `SWEEPS >= NCFG`, auto-scaled iteration counts,
  validation in a separate untimed pass, min-of-N, ratios as the stable
  quantity, `cudaDevAttrClockRate` rejected, and **rule 11 (the compiler
  vectorizes out from under your analysis) confirmed twice**.

## Forward-reference debts PAID here

- **M6 → M18: "register blocking is the missing ingredient."** Delivered and
  measured: 0.50 → 2.67/4.00 FMAs per shared read, 1372 → 7399 GFLOP/s.
- **M11 → M18: coarsening for reuse, explicitly distinguished from coarsening
  for MLP.** Delivered as a section with a full contrast table and the
  observation that the two have *opposite* occupancy behaviour.
- **M7 → M18: the condition under which swizzle beats padding.** Delivered as a
  **measured refutation**: padding wins by 1.23× at `BK = 8` and 1.02× at
  `BK = 32`, the mechanism of the swizzle's cost is identified (it blocks
  `LDS.128` contraction, not merely extra `LOP3`), and Module 7's premise —
  capacity limits tile size — is shown false on Ada for fp32 GEMM with a search
  over 9 candidate configurations. The regimes where the premise *is* true are
  named (Tensor Cores, multi-stage `cp.async`) and handed to M32/M33–34/M43.
- **M6/M9 → M18: double buffering.** Implemented, the one-barrier proof given,
  and measured — as a loss, with the reason.
- **M16 → M18: the honest ceiling.** 91–108 % of cuBLAS at the module shape,
  93 % at 2048³, with the five missing techniques named and owned.
- **M16 CYU Q3 → M18**: "two separate changes rather than one" — the two-level
  law makes that exact.

## Forward references made

- **Module 19 (occupancy)** — named as the owner; handed the 14–18.7×
  100 %-occupancy result, the non-monotonicity finding, and the by-hand
  four-limit occupancy arithmetic.
- **Module 20 (latency hiding / ILP vs occupancy)** — named as the owner of the
  other half of the trade.
- **Module 21 (roofline)** — implicit; the module works in % of measured
  ceilings rather than rebuilding a roofline.
- **Module 32 (`cp.async`, async copy, multi-stage pipelines)** — named
  repeatedly as the reason this module's double buffering loses and modern
  pipelines win. **`cp.async` is mentioned and never used.**
- **Modules 33–34 (Tensor Cores)** — named as the first of the five things
  cuBLAS/CUTLASS do that this kernel does not, and as a regime where Module 7's
  capacity premise becomes true.
- **Module 36 (cuBLAS)** — `cublasSgemm` used as a ruler only.
- **Module 43 (CUTLASS)** — named as the owner of the formalised three-level
  hierarchy (`ThreadblockShape → WarpShape → InstructionShape`), of real swizzle
  layouts, and of split-K in library form.
- **Nsight** — `ncu` remains unavailable (`ERR_NVGPUCTRPERM`). **No counter
  output is quoted anywhere.** Every bank-conflict claim in this module is
  supported by a controlled timing ratio plus the SASS, exactly as Module 7 did.

## Constraints observed

- **No re-teaching of Module 17.** The tiled kernel appears in `example01.cu`,
  `exercise01.cu` and `exercise03.cu` as a complete, labelled baseline only; the
  lesson's §1 states its measured plateau and moves on.
- Non-square, non-power-of-two dimensions throughout: **M = 1027, N = 2053,
  K = 769**, plus 37×53×11 and 129×65×9 where every block is partial, plus
  2048³ and 1024×1024×8192 as controls.
- No sm_90+ features. No `cp.async` (sm_80+ and available, but Module 32 owns
  it). No Tensor Cores. No warp-level tiling implemented.
- `example01.cu` is the only file that links cuBLAS.
- Everything builds warning-clean with `nvcc -arch=sm_89 -O3`.
- Nothing under `module18/` reveals an answer: Exercise 2's reuse law is checked
  by FNV-1a hash, its occupancy model against the CUDA API, its cost model
  against a measurement; Exercise 1's and 3's gates are ratios against a shipped
  baseline; every prediction is scored against something measured at run time.
- No binaries committed.

## Known issues / honesty notes

- **Four of Exercise 1's five candidate optimizations are null or inverted
  results**, all measured and all documented in the solution notes rather than
  quietly dropped. The exercise text was rewritten after measurement so that it
  does not promise penalties that do not exist. This is the single most
  important honesty note in the module.
- **The double-buffering section reports a loss.** It is implemented correctly
  (validated at four shapes), the one-barrier argument is proved, and it is
  slower. Explained rather than removed.
- **Module 7's prediction is refuted.** Reported as a refutation with the
  mechanism, not smoothed into agreement.
- **`ncu` unavailable** (`ERR_NVGPUCTRPERM`). All bank-conflict claims rest on
  controlled timing ratios and SASS instruction counts.
- **`compute-sanitizer` reports `ERROR SUMMARY: 1 error` on the solutions**, and
  it is its own `cudaDeviceReset` warning ("Resetting device while there are
  still other users claiming to use it"), not a kernel error. There are no
  `Invalid` accesses. Under instrumentation the programs also print
  `OVERALL: FAIL`, because throughput-based prediction buckets are scored
  against a 60× slower run; run the sanitizer for memory safety and the plain
  binary for scoring. Recorded so nobody re-diagnoses it.
- **cuBLAS measures 7258–8141 GFLOP/s here against Module 16's recorded
  8150–8705.** The low end appears when the two-stage (stream-then-compute)
  warm-up is used rather than M16 example01's GEMM warm-up. The module reports
  ranges and ratios rather than single figures, and the parity claim is stated
  as parity-within-spread rather than as a win.
- **Absolute throughput moves 10–15 % with thermal state**; every claim is a
  ratio or a range.
- The `8×4` vs `4×8` residual: the `LDS.128` phase model predicts 1.67× and the
  measurement is 1.34–1.61×. The gap is the part of the kernel that is not
  shared-memory-limited. Printed, not hidden.

## Cross-module observations for the index

1. **⚠️ Spec / index correction: Module 7's `max(2, D)` law needs a width
   qualifier.** "A 2-way conflict is free on Ada" holds for **4-byte** shared
   accesses, because a 32-lane 4-byte read asks for 128 B and the bank array
   delivers 128 B per cycle, leaving a spare cycle. An `LDS.128` is phase-split
   into 4 phases of 8 lanes and **one phase already asks for the full 128 B**,
   so a 2-way conflict inside a phase costs a real extra cycle: on 16-byte
   shared reads **`cost ∝ D`, no floor of 2**. Isolated by two configurations
   identical in registers, spills, shared bytes, occupancy, both reuse ratios
   and the entire inner-loop SASS, measured 1.61× apart, with a large-K control
   ruling out the epilogue. This should go into §6b and into any future revision
   of Module 7's cost law.
2. **§5b's M18 debt is paid as a refutation.** "The condition under which
   swizzle beats padding" turns out not to be reachable in fp32 GEMM on Ada:
   registers bind before shared memory in all 9 constructed candidates, because
   `ptxas` spends registers in proportion to `BK`. §5b should record the
   refutation and the two regimes where the premise does hold (Tensor Cores,
   multi-stage `cp.async`).
3. **§6b should gain:** the register allocation granule (8 per thread, per
   warp); the four-limit occupancy formula with the driver reserve and
   granularity; the 100 %-vs-33 % occupancy result (**18.7×**, or 14.3× at 128
   threads/block) as the canonical "occupancy ≠ performance" number; the
   register-tiled SGEMM ceiling (7399–8336 GFLOP/s = 41–46 % of the FP32
   ceiling, 91–108 % of cuBLAS at 1027×2053×769 and 93 % at 2048³); and
   **double buffering as a measured loss on Ada fp32 GEMM** (0.81–0.92×).
4. **§6 "themes already used" should gain**: register-tiled GEMM, the
   `__launch_bounds__` spill cliff as a measurement instrument, the transposed
   operand tile, and the `BN/TN ≥ 16` rule. Warp-level tiling, `cp.async`
   pipelines, Tensor-Core GEMM and split-K **remain unused and reserved** for
   M32, M33–34 and M43.
5. **Spec §12 rule 11 earned a second and third confirmation** (scalar shared
   reads contracted into `LDS.128`; 52 redundant reads CSE'd away so that the
   "wrong" inner loop is 1.05× faster). Worth strengthening the rule's wording
   from "will vectorize" to "will vectorize and eliminate".
6. **A new validation-adjacent finding for the index:** a guard can be
   *arithmetically* redundant and still required for memory safety, and no
   numerical validator — including Module 16's three-stage one — can detect its
   absence. `compute-sanitizer --tool memcheck` is the only instrument. Module
   16's validation methodology should carry this caveat.
