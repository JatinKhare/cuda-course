# Module 15 manifest — Matrix Transpose

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module15/lesson.md`, `module15/example0{1,2}.cu`,
`module15/exercise0{1,2,3}.cu`.
Solutions: `solutions/module15/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module15/check_your_understanding.md`,
`solutions/module15/MANIFEST.md`.

All `.cu` verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2, V13.2.51, RTX 3500
Ada), warning-clean. Both examples print `OVERALL: PASS`. All three solutions
print `OVERALL: PASS` with full scores (10/10, 7/7, 6/6). All three shipped
exercises compile with TODOs blank and exit gracefully
(`Set TODO 4 (PREDICTION) first.`, `Set TODO 5 (PREDICTION) first.`,
`Set TODO 1 first.`) returning 0. No binaries committed.

---

## Concepts taught

- **Transpose as pure data movement.** Zero arithmetic, so every microsecond is
  attributable to a named memory mechanism. **PORTABLE CUDA CONCEPT.**
- **The copy ceiling as the module's methodology backbone.** A transpose and a
  copy of the same matrix have identical compulsory traffic (2N), so the only
  honest denominator is a measured copy with the same block shape and
  instruction count — not 432 GB/s, and not the naive version.
- **The copy ceiling decomposed and shown to have nothing hidden in it**: 1-D
  `float4` stream, 2-D tiled copy with loads hoisted, 2-D tiled copy with
  load/store interleaved, and a copy staged through shared memory all measure
  within 0.1% (373.0–373.3 GB/s). The 2-D traversal is free, the MLP difference
  is free for a copy, and **a full shared-memory round trip is free**.
- **Sector counting applied to both sides of a transpose**: coalesced side 4
  sectors, strided side 32 (M5's saturation floor at stride ≥ 8 floats).
- **The 9N traffic model for the naive transpose** (1N read + 8N write) and why
  it over-predicts: the 32 over-fetched sectors are consumed by neighbouring
  warps and by later `j` iterations before eviction, so the real amplification is
  nearer 4× than 8×. Predicted 22% of copy, measured 29–49%.
- **The read/write asymmetry, measured inside an algorithm**: a strided *write*
  costs ≈2× a strided *read* of the same pattern, because a partial-sector store
  forces a read-merge-write at L2. Measured 43.6% vs 67.5% of copy (guarded).
  **When you must break one side, break the reads.**
- **The bounds guard costs 1.34× on the strided-read naive kernel and 0.3% on
  the tiled one**, with the SASS showing why: without the guard the compiler
  hoists all four `LDG` ahead of all four `STG`; with it, it interleaves them
  three-at-a-time. M11's MLP result appearing uninvited. **Documented surprise.**
- **The canonical tiled transpose**, and its four asymmetries: the shared store
  and load subscripts are not mirror images; `blockIdx.x`/`blockIdx.y` swap roles
  across the barrier; the output's leading dimension is `H` not `W`; the two
  guards test different pairs. **Three of the four pass on a square matrix.**
- **The 32-way column conflict** in a `[32][32]` tile (`bank(r,c) = c`), and both
  of M7's fixes verified structurally and measured.
- **The module's central measurement:** removing a genuine 32-way bank conflict
  is worth **0.96–1.04× at 8192×8192 (DRAM-bound)** and **2.67–3.25× at
  2048×2048 (L2-resident)**. M7 measured the identical conflict at **14.79×** on
  an LSU-bound microbenchmark. All three are correct; the binding resource
  decides. Includes the explicit cycle budget (960 shared-replay cycles per block
  against ~7000 cycles of DRAM time per block).
- **`D/2` bounds the shared-memory term, never the kernel** — M7 said it; this is
  the extreme case of it.
- **Padding vs XOR swizzle on this problem: a tie.** Six measurements of
  `swizzle/padded` on the L2-resident case: 0.92, 0.97, 1.00, 1.02, 1.03, 1.22.
  M7's 19% padding win **does not transfer**, because M7's kernel was LSU-bound
  (32 × 64 shared accesses per thread and nothing else) while the transpose
  issues 8 shared against 8 global accesses and waits on DRAM. The 8 extra
  `LOP3` the swizzle costs are invisible. **The comparison is only decidable on
  an LSU-bound kernel.**
- **Tile shape and block shape, swept**: the 32×32-tile / (32,32)-block
  configuration is the *worst* in every run (67.2 / 76.5 / 78.1% of copy)
  because it gives 1 block/SM = 32 warps **and has no `j` loop**, hence one
  outstanding request per thread. The 64×64 tile with a (64,16) block is
  *also* 1024 threads and 1 block/SM and is at the ceiling, which isolates the
  variable: **the `j` loop is the MLP mechanism, not a convenience, and
  outstanding-requests-per-thread is the quantity, not occupancy.** 16-wide
  tiles are only **1–3%** behind 32-wide ones within a sweep — the sector model
  is right and the "a warp is 32 wide" intuition is wrong. **A 64×64 tile is at
  the ceiling in every run (96.5–100.3% of the 32-wide copy) and is the shape to
  ship.**
- **Partition camping is NOT observable on Ada.** With the matrix, tile, grid and
  useful bytes held fixed at 8192×8192 and only the output leading dimension
  varied (8192 / 8200 / 8224 / 8320, i.e. a 2²⁰-byte inter-tile stride versus
  three non-powers of two), the transpose measures 92.1–96.6% of copy with no
  dip at the power of two — in one run the 2²⁰ stride was the *fastest* of the
  four. **Diagonal block reordering is 6.5–13.5% SLOWER at every stride in every
  run**, and 9–15% slower in the fixed-size sweep. Explained by M10's
  L2 slice *hashing* (a hash destroys the alignment between "power-of-two stride"
  and "one partition") and by the 48 MB L2 with hundreds of tiles in flight.
  **Clean negative result; the 2009 mitigation is a pessimization here.**
- **Boundary handling is a correctness problem, not a performance problem.**
  Guarded vs unguarded tiled transpose: 0.997×. A large non-multiple matrix
  (8191 × 8193) runs at the same speed as 8192², because only 0.8% of blocks
  touch a boundary. The *correctness* half is mandatory and square power-of-two
  test suites are structurally incapable of catching the two classic defects.
- **The warp-spans-two-tile-rows case**: a 16-wide tile with a (16,16) block puts
  `threadIdx.y ∈ {0,1}` in one warp, which changes the conflict degree of both
  phases from the rule's answer (1) to the true answer (2). **A rule has a
  domain.**
- **The general padding rule, derived**: for a `[tileW][P]` tile of `E`-byte
  elements, the column walk is conflict-free iff `gcd(P, 128/E) = 1`, which for
  `E` = 4, 8, 16 is `gcd(P,32)=1`, `gcd(P,16)=1`, `gcd(P,8)=1` — **all three
  satisfied by exactly the odd numbers.** The element size does not enter.
  Generalizes three separate M7 statements into one.
- **Where transpose is the answer**: device-side AoS→SoA (M5's debt), NCHW↔NHWC,
  pre-GEMM layout changes, `cub::BLOCK_LOAD_TRANSPOSE` (M13's name).
- **Benchmarking**: 1500 ms warm-up throughout (spec §12 rule 4); every
  comparison group timed back to back with rotation and `SWEEPS >= NCFG`;
  validation in a separate pass; the L2-resident measurements labelled loudly as
  not DRAM numbers (apparent 1671 GB/s = 387% of peak).

## CUDA API / intrinsics / syntax introduced

Nothing new — that is deliberate. Module 15 is an integration module; every API
it uses came from Modules 1–13. Reused and exercised:

- `__shared__` 2-D and flat arrays, static (M6)
- `__syncthreads()` with both guarantees required (M6 used, M9 explained)
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (M6/M7/M12) — used here to
  explain the 1024-thread row of the shape sweep
- `dim3` 3-D grids with `blockIdx.z` as a batch index (M3), and the 65535 limit
  on `gridDim.y`/`gridDim.z` as a *design constraint* for the first time
- `float4` reinterpretation for the streaming reference (M5/M11)
- `atomicAdd` on a device-side mismatch counter in every validator (M10)
- `#pragma unroll` on compile-time tile loops; template `__global__` on tile and
  block extent (M4's dispatch trick)
- `cuobjdump -sass`, `nvcc -Xptxas -v` (M4)
- SASS vocabulary used: `LDS`, `STS`, `LDG.E.CONSTANT`, `STG.E`, `LOP3.LUT`,
  `BAR.SYNC.DEFER_BLOCKING`, predicated `@!P` forms
- FNV-1a answer hashing so `module15/` contains no answers (M11/M12 convention)
- A `peek()` helper that reads a constant through a `const volatile int*`, to
  stop nvcc folding the "TODO not set" tests and emitting warning #128-D
  ("loop is not reachable") in the shipped exercise

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | **The ladder.** Ten kernels in one rotated sweep of 10: 1-D `float4` copy, 2-D tiled copy hoisted (the ceiling), 2-D tiled copy interleaved, both naive transposes, `[32][32]`, `[32][33]`, XOR swizzle, padded+diagonal, and a shared-staged copy. Part B validates all ten on 8192² **and** 4093×2049. Part C isolates the read/write asymmetry. Part D decomposes the ceiling (2-D traversal 1.001×, MLP 1.000×, tile round trip 0.997×, **the permutation itself 1.036×**). Part E repeats the three tiled variants on an L2-resident 2048² matrix, where the conflict costs 3.0×. |
| `example02.cu` | **Shape, camping, boundaries.** Part A: 10-configuration tile/block sweep (tile 16/32/64 × block (16,16)…(64,16)) plus two matched copies, with the occupancy API printing the blocks/SM that explains the 1024-thread row. Part B: the leading-dimension sweep that kills partition camping, with a copy at the same `ldOut` as the control. Part C: guarded vs unguarded, and 8191×8193. Validation of every shape on 1024² and 4093×2049. |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Fill-in (§6.1) + design (§6.5) + prediction (§6.2/6.6) | 4 | Build naive → tiled → conflict-free from a measured copy; validated on 2048² **and** 4093×2049; five scored predictions | **Three of the four ways to get the tiled kernel wrong pass on a square matrix**, and the harness's rectangular case is the only thing that catches them — chiefly `out[(yo+j)*W + xo]` instead of `*H`. TODO 3 never says "padding" or "swizzle": it asks for an injective `shIdx` with degree ≤ 2 in both phases and ≤ 1056 floats, which admits `[32][33]` and the XOR swizzle and rejects `[32][34]` **on capacity** (1088 > 1056) even though its degree of 2 would be accepted. The degree bound is 2, not 1, because M7's `max(2,D)` law makes a 2-way conflict free. PRED[4] is the headline: removing the 32-way conflict is worth **<1.25×** (measured 0.96–1.03×), and the program then re-times the same pair on an L2-resident matrix where it is worth 2.7–3.3×. |
| `exercise02.cu` | Design (§6.5) + prediction (§6.6), minimal scaffolding | 5 | NCHW→NHWC on a 128×67×57×57 tensor; the reader designs grid, both mappings, the shared layout and the boundary handling | The reduction is the exercise: fold `h,w` into `sp` and it is a batched `C × HW` transpose. **The store phase's contiguous axis is the CHANNEL**, so `threadIdx.x` must run over channels there — the exact reverse of the load — and a reader who keeps `threadIdx.x` on the spatial axis writes a kernel that *validates* and is 2.6× slower. `C = 67` forces boundary handling on an axis only 2.09 tiles wide, and the partial tile must be **zero-filled, not skipped**, or 29/32 of the last channel tile's cells carry the previous block's SRAM into the output. **Documented surprise:** only **69.5% of the launched tile cells hold data** and it costs *nothing* — a predicated-off lane supplies no address, so grid efficiency and memory efficiency are different numbers. The grid TODO is rejected for exceeding 65535 on y/z or for launching >2× the tiles needed. |
| `exercise03.cu` | Predict-the-behavior (§6.2) + PTX/SASS-adjacent analysis + design (§6.5) | 5 | Seven given, correct transpose kernels; predict 28 numbers (read sectors, write sectors, two conflict degrees) by hand, implement M5's and M7's counting procedures, then measure | **Config 5, `tile[32][34]`:** `gcd(34,32)=2` so `D=2`, and it measures within 1% of `[32][33]` — M7's `max(2,D)` law means the second pad byte buys nothing. **Config 6, `tile[16][17]` with a (16,16) block:** a warp spans **two** tile rows, so the pitch-is-odd rule (which would say `D=1`) is outside its domain and the true answer is `D=2` in *both* phases; the read is still 4 sectors and 100% efficient yet the kernel loses 4–12%, a documented model/measurement gap (DRAM page locality). **TODO 4's answer does not depend on `elemBytes`** — `gcd(P,32)`, `gcd(P,16)` and `gcd(P,8)` are all "P odd" — which is the generalization M7 never stated. **The closing point:** of the four hand-computed columns, exactly *one* (write sectors) predicts the measurement; configs 3–7 span degrees {32,1,2,2,1} and land within 3.5% of each other. Also: config 2 (strided read, **unguarded**) measures 95.9% where example01's guarded twin measures 67.5% — the guard costs 1.34× in MLP. TODO 3 is scored by FNV hash; the 28 answers are not in the file. |

Scoring: Ex1 `SCORE: n/10` (4 numerics + 1 structural + 5 predictions), Ex2
`SCORE: n/7` (2 structural + correctness + an 80%-of-copy gate + 3 predictions),
Ex3 `SCORE: n/6` (structural tests, procedures-by-hash, table-by-hash, the
padding rule at 12 points, 2 ratio buckets). `OVERALL: PASS` requires full marks.

## Measured results recorded in this module

| Quantity | Value |
|---|---|
| Copy ceiling, 8192², 2N model, 1500 ms warm-up | **373.1–383.6 GB/s (86.4–88.8% of 432)** |
| 1-D `float4` copy / 2-D tiled copy | 1.001× (the 2-D traversal is free) |
| Interleaved / hoisted copy | 1.000× |
| **Shared-staged copy / plain copy** | **0.997× (the tile round trip is free)** |
| Padded transpose / staged copy | 1.036× (**the permutation itself**) |
| Naive, coalesced read / strided write | 29–49% of copy (typ. 34–44%) |
| Naive, strided read / coalesced write, guarded | 61–76% of copy (typ. 67.5%) |
| Naive, strided read / coalesced write, **unguarded** | **88.3–95.9% of copy** |
| Guard cost, strided-read naive kernel | **1.34×** |
| Guard cost, tiled kernel | 0.997–0.999× (free) |
| Tiled `[32][32]`, D = 32 | 93.2–97.9% of copy |
| Tiled `[32][33]`, padded | 90.4–97.6% of copy |
| Tiled XOR swizzle | 89.1–96.3% of copy |
| **conflicted / padded, DRAM-bound (8192²)** | **0.96–1.04×** |
| **conflicted / padded, L2-resident (2048²)** | **2.67–3.25×** |
| swizzle / padded, L2-resident 2048², 7 runs | 0.92, 0.97, 1.02, 1.03, 1.06, 1.06, 1.22 (**tie**) |
| Extra `LOP3` the swizzle costs | **8** (4 store, 4 load), verified in SASS |
| `[32][32]` vs `[32][33]` SASS | identical instruction counts, 22 regs each; only the `STS` offsets differ (0x400/0x800/0xc00 vs 0x420/0x840/0xc60) |
| Shared bytes, `[32][32]` / `[32][33]` / swizzle | 4096 / 4224 / 4096 |
| L2-resident apparent bandwidth, 2048² | up to 1671 GB/s (387% of peak — **not** a bandwidth) |
| Tile 16×16, block (16,16) / (16,8) | 83.7–100.9% / 85.3–98.6% of copy (1–3% behind tile 32 within any one sweep) |
| Tile 32×32, block (32,32), 1024 thr | **67.2 / 76.5 / 78.1%** — worst in every run (1 block/SM = 32 warps, **and no `j` loop**) |
| Tile 32×32, block (32,16) / (32,8) / (32,4) | 85.1–94.2% / 85.2–97.3% / 83.3–94.6% |
| **Tile 64×64, block (64,16)** | **96.5–98.3% of the 32-wide copy, 97.6–100% of its matched copy** |
| Tile 64×64, block (64,8) | 95.4–103.2% |
| blocks/SM: (32,32)+4224 B / (32,8)+4224 B / (64,16)+16640 B | 1 / 6 / 1 |
| Partition camping, ldOut 8192 / 8200 / 8224 / 8320 | 96.6 / 96.4 / 96.4 / 95.1% of copy (run 2: 92.6 / 92.1 / 93.9 / 92.2) — **no dip at 2²⁰**; in run 1 the power-of-two stride was the *fastest* |
| Diagonal reordering / linear, same ldOut | **1.065–1.135× (slower), every stride, every run** |
| Diagonal reordering, fixed-size sweep (example01) | 1.087–1.145× (slower) |
| 8191 × 8193 tiled transpose | 348.8–353.7 GB/s, same as 8192² |
| Partial tiles at 8191 × 8193 | 512 of 65 792 blocks (0.8%) |
| NCHW→NHWC naive (stride-67 write) | 25.1–36.7% of copy |
| NCHW→NHWC tiled | 94.3–100.3% of copy |
| NCHW→NHWC launched tile cells holding data (C=67) | 69.5%, at no measured cost |
| Conflict on the STORE instead of the LOAD | 0.92–1.13× (no signal) |

## Assumed from earlier modules

- **M1**: 40 SMs, 1536 threads/SM, warp = 32 lanes, blocks indivisible and
  non-migrating (the precondition for a scratchpad), Little's Law and the
  ~116 kB in flight the memory system needs, waves.
- **M2**: `nvcc -arch=sm_89`, launch syntax, `CHECK`/`CHECK_KERNEL`,
  `cudaEvent_t` timing.
- **M3**: the linearization rule (the definition of "a warp's 32 addresses"; it
  is load-bearing in Exercise 3 config 6), 2-D/3-D grids, the 65535 limit on
  `gridDim.y`/`gridDim.z`, ceil-divide grid sizing, bounds guards as
  predication, the 66.7% ceiling of 1024-thread blocks.
- **M4**: the storage map and latencies (L1 40.5 / L2 241.3 / DRAM 575 cycles),
  L1 non-coherence across SMs, 48 MB L2 as the benchmarking hazard, the replay
  mechanism, `cuobjdump -sass`, `-Xptxas -v`, template-parameter dispatch.
- **M5**: the sector-counting procedure (used verbatim), 32 B sectors vs 128 B
  lines, address *set* not sequence, inactive lanes supply no address,
  write-allocate asserted, `float4` and 16 B alignment, row pitch as a layout
  decision, effective vs DRAM bandwidth, >100% of peak means L2.
- **M6**: shared memory as a scratchpad, cooperative loading, the load mapping ≠
  compute mapping distinction, the partial-tile stale-cell hazard, the
  convoying cost of a barrier, capacity → occupancy with the 1024 B reserve.
- **M7**: the bank map `(addr/4) % 32`, degree = max distinct *words* per bank,
  broadcast is free, `gcd(k,32)`, `gcd(pitch,32)==1`, padding vs XOR swizzle and
  their different costs, the phase split for 8/16 B elements, **the `max(2,D)`
  cost law**, the 14.79× reference measurement, "confirm in the SASS that the
  instruction under test is the one executing".
- **M8**: divergence is warp-local; predication vs branching.
- **M9**: `__syncthreads()`'s two guarantees, the uniformity rule, why a
  divergent barrier corrupts silently on sm_89, `volatile` is not
  synchronization.
- **M10**: **L2 slice hashing** — the key fact behind the partition-camping
  negative result; `atomicAdd` (used in every validator); replay as a third
  instance.
- **M11**: compulsory traffic and the floor, `x floor` as the reporting unit,
  write-allocate **measured** (3.95×), MLP and ILP-as-occupancy, `float4` as a
  latency/issue optimization, "a model has a domain".
- **M12**: the 1500 ms warm-up finding and the 410.5 GB/s pure-read ceiling;
  "you cannot rank optimizations without knowing which resource is saturated".
- **M13**: `cub::BLOCK_LOAD_TRANSPOSE` named as the library instance of this
  kernel.
- Spec §12 throughout.

## Forward-reference debts PAID here

- **M7 → M15** (recorded in the cross-module index §5b): "M7 deliberately avoided
  using transpose as its vehicle so M15 keeps it intact." Paid in full; the
  32-way column conflict, padding, and the XOR swizzle are all developed on
  transpose, and M7's `max(2,D)` law and 14.79× measurement are both used and
  both re-contextualized.
- **M5 → M15**: the device-side AoS→SoA conversion and the deinterleave pattern.
  Paid in the lesson's "Where you will meet this again" (AoS→SoA is the `N × C`
  transpose, with the observation that an odd field count is conflict-free with
  no padding) and in Exercise 2, whose NCHW→NHWC *is* the general AoS→SoA
  conversion (NHWC is the array-of-structs form over channels) — stated
  explicitly in the exercise header.
- **M6 → M15**: "the case where neither the read nor the write side can be
  coalesced" was M5's problem, M6's job 2, and M6 named Modules 15–17 as the
  destination of the K/H argument. Paid: the transpose is the canonical instance
  where shared memory buys a *change of access pattern* rather than reuse, and
  the module measures the scratchpad round trip at exactly 0.997× so the reader
  can see that the staging itself is free — the opposite of M6's stencil result,
  with the reason stated.
- **M11 → M15**: M11 named Module 15 as "where shared memory returns for a
  *different* reason than reuse". Paid, and M11's write-allocate measurement is
  reused as the explanation of the read/write asymmetry.
- **M10 → M15** (implicit): M10's L2 slice hashing is the mechanism that makes
  the partition-camping negative result explicable rather than merely observed.

## Forward references made

- **Modules 16–18 (GEMM)** — named as the consumer of layout changes, and as the
  setting where *not* transposing (reading B's columns straight into a padded or
  swizzled shared tile) is the alternative. **Module 18 explicitly owns the
  padding-vs-swizzle decision**, because this module's kernel is not LSU-bound
  and therefore cannot decide it. This is the debt the index records M18 owing,
  restated with the extra evidence that the transpose measures a tie.
- **Module 19 (occupancy)** — the 1024-thread row of the shape sweep is
  explained with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and deferred.
- **Module 20 (latency hiding)** — the `j` loop as an MLP mechanism, and the
  guarded/unguarded SASS difference, both point here.
- **Module 21 (roofline)** — the copy ceiling as the memory-bound plateau.
- **Module 23 (Nsight Compute)** — `ncu` remains unavailable
  (`ERR_NVGPUCTRPERM`); the bank-conflict and sector metric names were given in
  M5 and M7 and are not repeated. **No counter output is quoted anywhere in this
  module**; every number was constructed and measured directly.
- **Module 36 (CUB / Thrust)** — `cub::BLOCK_LOAD_TRANSPOSE` identified as this
  module's kernel at block scope inside a library; `cublas<t>geam` and CUTLASS
  layout transforms named as what you would actually ship.
- **Modules 41–42 (AI kernels)** — NCHW↔NHWC named as the conversion that stands
  between a framework tensor and a Tensor-Core convolution.

## Known issues / honesty notes

- **The machine's power state dominates absolute timings.** During authoring the
  *same binary on the same data* produced a copy ceiling of 373 GB/s in one run
  and **98 GB/s** twenty minutes later, reproducibly across three consecutive
  invocations, with `nvidia-smi` showing the software power cap (`0x4`) and the
  memory clock stepping 8801 → 8001 → 7001 MHz. Every claim in the lesson and the
  solution notes is stated as a ratio and was reproduced in at least three
  separate runs; every scored prediction bucket was widened until it survived the
  full observed spread. Absolute GB/s figures in this module should be read as
  "on a cool machine".
- **The `% of copy` ratios are stable but not perfectly.** The naive
  strided-write kernel measured 29–49% of copy across states; `[32][32]` vs
  `[32][33]` measured 0.96–1.04× (i.e. the sign flips). Both facts are stated in
  the text rather than smoothed.
- **`ncu` unavailable** (`ERR_NVGPUCTRPERM`), per the standing rule: documented
  as theory, not worked around. Every conflict degree in this module was verified
  by host-side simulation of the address map (which the exercises make the reader
  implement) plus a controlled L2-resident timing ratio, never by reading a
  counter.
- **`compute-sanitizer --tool memcheck` reports 0 errors** on a dedicated
  boundary harness that launches every tiled kernel shape in this module
  (tile 16/32/64, pad 0/1, the XOR swizzle, the diagonal remap, and Exercise 2's
  NCHW→NHWC kernel) at 4093×2049, 1023×517, 33×31, 1×1, 31×1, 2049×4093 and
  8191×129 — i.e. every degenerate partial-tile shape, including matrices
  narrower and shorter than one tile. The sanitizer on this machine is the
  CUDA 12.9 build (`.../NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/compute-sanitizer.bat`)
  and is **not on `PATH`**; invoke it by full path.
- **`compute-sanitizer --tool memcheck` emits a benign
  `Resetting device while there are still other users claiming to use it`
  API warning** on every program here, because the house convention calls
  `cudaDeviceReset()`. It is counted in `ERROR SUMMARY` and is not a memory
  error (M12 recorded this first).
- **nvcc warning #128-D ("loop is not reachable")** fires on a shipped exercise
  whose TODO-blank early-exit test is a compile-time constant: the front end
  folds the test, concludes the rest of `main` is dead, and complains about the
  `do { } while (0)` in the `CHECK` macro. Fixed in `exercise03.cu` with a
  `peek()` helper that reads the constant through a `const volatile int*`.
  Worth knowing for future modules that gate on a `static const` prediction.
- **The tile-shape sweep is the noisiest table in the module.** Across three
  good-state runs the `%ofcopy` of a fixed configuration moved by up to 17 points
  (16×16 (16,16): 83.7 / 94.7 / 100.9). An early draft of the lesson claimed a
  16-wide tile "loses 12%" on the strength of one run whose copy sample was
  unlucky; the within-sweep difference is **1–3%**. The text now states only the
  two conclusions that survive every run: the 1024-thread block with no `j` loop
  is reproducibly worst, and the 64-wide tile is reproducibly at the ceiling.
  The DRAM-page-locality explanation for the residual 1–3% is labelled a
  hypothesis, not a measurement, because settling it needs `ncu`.

## Constraints observed

- **No GEMM** (M16–M18 own it) and **no histogram** (M14 owns it). No reduction
  or scan appears; neither is re-taught.
- Themes: matrix transpose was reserved for this module by §6 of the index and is
  used here for the first time. NCHW→NHWC and the AoS→SoA deinterleave are new.
- No sm_90+ features; no TMA, no clusters, no `wgmma`.
- Non-square and non-multiple-of-tile sizes are mandatory and present:
  **4093 × 2049** in every harness's validation pass, **8191 × 8193** timed in
  `example02.cu`, **67 × 3249 per image** in Exercise 2.
- Timing per spec §12 throughout: **1500 ms** duration-based warm-up, all
  configurations in a comparison group timed back to back with rotated order and
  `SWEEPS >= NCFG` (10/10 in `example01`, 10/10 in `example02` Part A, 5/5 in
  Exercise 1, 8/8 in Exercise 3), min-of-N, validation in a separate untimed
  pass, buffers ≥ 5× the 48 MB L2 for every DRAM figure, every L2-resident
  measurement labelled as such, ratios reported as the stable quantity.
- Nothing under `module15/` reveals an answer: Exercise 3's 28-cell table is
  scored by FNV-1a hash, Exercise 1's and 2's layouts are scored by host-side
  structural simulation, and no solution text appears in any exercise file.
- No binaries committed.

---

## Post-authoring repairs

**Defect.** Exercises 1 and 2 intermittently reported `OVERALL: FAIL` on their
own reference solutions. Numerics were never wrong; only the TODO prediction
buckets flipped. Reproduced at roughly 1 run in 3 for Exercise 2 and 1 in 3 for
Exercise 1.

**Root cause (two layers, both confirmed by measurement).**

1. *Under-warmed part.* Both harnesses ran a 1500 ms **streaming-only** warm-up
   and omitted the 500 ms compute phase that spec §12.4's corollary requires.
   The configurations being compared are not all limited by the same resource
   (the copy ceiling and the naive transposes are DRAM-bound; the tiled
   transposes are shared-memory/issue-bound), so a part with its memory P-state
   up and its SM clock parked low moves them by *different* amounts and the
   ratios drift. Both also used a fixed 20 iterations instead of auto-scaling to
   a ~10 ms timed segment (§12.12), and Exercise 2 ran `NSWEEP = 3`.
2. *Bucket edges sitting on top of the measured values* — the dominant cause.
   With the warm-up corrected, the strided-write kernels are still the least
   reproducible numbers in the course: a fully strided write is the most
   power-hungry kernel in either program, so it is the one that loses the most
   when the part is power limited. Measured over 40+ runs, Exercise 1's `v1`
   lands anywhere in **34–64%** of the copy and Exercise 2's naive conversion in
   **29–59%**. The shipped edges were at 60% and 45% respectively — i.e. inside
   those bands. No amount of warm-up fixes that; the edges had to move.

A third effect was found and handled separately: after several minutes of
back-to-back benchmarking this part pins its memory clock at **6001 MHz instead
of 9001** with `SW_POWER_CAP` and `SW_THERMAL_SLOWDOWN` both asserted, and in
that state the ratios are not this GPU's ratios at all (v3 falls from 98% of the
copy to 81%, and v1 and v2 become indistinguishable). It recovers after 20–40 s
of idle. No bucket scheme can be made to span both operating points.

**Changes — identical in the student file and the solution file in each pair.**

`module15/exercise01.cu`, `solutions/module15/exercise01_solution.cu`:
- Added `computeWarm`, a pure-arithmetic no-memory kernel, and a **500 ms
  compute warm-up after** the 1500 ms streaming warm-up (§12.4 corollary).
- Added an **operating-point guard**: after warming, probe the copy ceiling; if
  it comes back below 200 GB/s (healthy ≈ 255, power-capped ≈ 118) idle 10 s and
  warm again, up to five times, then warn and proceed. Adds `<chrono>`/`<thread>`.
- **Auto-scaled iteration counts** to ~10 ms per timed segment (§12.12),
  replacing the fixed 20.
- `NSWEEP` 5 → **8** (still ≥ `NCFG`; min-of-8 instead of min-of-5).
- **Bucket edges 60% / 90% → 20% / 85%**, and `PRED[0]` 1 → **2**. Every
  measured version is now ≥ 8 points from an edge (v1 34–64, v2 72–76, v3 93–99,
  v4 94–101). Cost: v1 and v2 share a bucket. Justification: the gap between
  their ranges is ~8 points, smaller than v1's own run-to-run spread, so
  "strided write is worse than strided read" is a claim this machine can *show*
  in the `%ofcopy` column but cannot *score* reproducibly. Bucket count is
  unchanged at 3, the TODO structure is unchanged, and the exercise remains
  wrong-able: bucket 1 (below 20%) is the classic over-estimate of the strided
  penalty from the sector model, and bucket 3 for v3/v4 requires knowing that a
  tiled transpose really does reach the copy ceiling.

`module15/exercise02.cu`, `solutions/module15/exercise02_solution.cu`:
- Same `computeWarm` + 500 ms compute warm-up, same operating-point guard, same
  auto-scaled iteration counts.
- `NSWEEP` 3 → **8** (`NCFG` is 3, so 3 was the bare minimum under §12.9).
- Timed launches factored into a `launchCfg` lambda so the warm-up, the
  calibration probe and the sweep cannot drift apart.
- **Bucket edges: PRED[0] 45% / 75% and PRED[1] 60% / 85% → a single scheme,
  70% / 82%, for both.** `PRED` is **unchanged** at `{ 1, 3, 2 }`: the naive
  conversion (29–59%) stays in bucket 1 with ≥ 11 points of clearance and the
  tiled kernel (90–100%) stays in bucket 3 with ≥ 8. The ≥ 80% pass gate is
  untouched. Bucket count unchanged at 3.

Solution `.md` answer keys, transcripts and variance paragraphs updated to match.

**Verification.** `nvcc -arch=sm_89 -O3` warning-clean for all eight Module 15
programs. Eight consecutive runs of each solution, the first immediately after a
fresh compile on a cold GPU: **exercise01_solution 8/8 PASS, exercise02_solution
8/8 PASS** (plus an earlier 8/8 pair at slightly tighter edges, 32 scored runs
total, zero failures). The operating-point guard fired on two of those runs and
both still passed. Student `exercise01.cu` and `exercise02.cu` compile with the
TODOs blank and exit gracefully (rc 0, "Set TODO N (PREDICTION) first.").
Regression: `example01`, `example02` and `exercise03_solution` all still
`OVERALL: PASS`, untouched. No binaries left behind.
