# Module 13 manifest — Prefix Sum / Scan

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module13/lesson.md`, `module13/example0{1,2}.cu`,
`module13/exercise0{1,2,3}.cu`.
Solutions: `solutions/module13/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module13/check_your_understanding.md`,
`solutions/module13/MANIFEST.md`.

All `.cu` verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2, RTX 3500 Ada),
warning-clean. `example02.cu` additionally needs
`-std=c++17 -Xcompiler /Zc:preprocessor` for `<cub/cub.cuh>` under MSVC.
All three solutions print `OVERALL: PASS`. All three shipped exercises compile
with TODOs blank and exit gracefully with `Set TODO 5 (PREDICTION) first.`;
with the prediction filled but the code TODOs blank they run to completion and
report `OVERALL: FAIL` without crashing, hanging, or reading out of bounds.

---

## Concepts taught

- **Inclusive vs exclusive scan**, defined over an arbitrary associative `⊕`
  with identity `e`. **Associativity is the only requirement**; commutativity is
  not. **PORTABLE CUDA CONCEPT.**
- **Exclusive is the more useful primitive**: the exclusive scan of "items
  emitted per element" *is* the output offset of each element.
- **The three conversions and their two hazards**: `inc[i] = exc[i] ⊕ x[i]`
  (always safe); `exc[i] = inc[i] ⊖ x[i]` (needs an inverse — wrong for
  `max`/`min`/`or`); `exc[i] = inc[i-1]` (safe for any operator but **loses the
  total**). Hence `count = exc[N-1] + x[N-1]`, a data-dependent off-by-one.
- **Float scan is non-associative but deterministic** — a fixed tiling gives the
  same wrong-vs-serial answer every run, unlike M10's `atomicAdd`. Determinism
  and exactness are different properties.
- **Scan is bandwidth-bound**: 2N minimum traffic, floor = `2·N·4 / 432 GB/s`
  = 1.2428 ms at N = 67,108,861. Every version reported against it.
- **Hillis–Steele**: O(N log N) work, O(log N) depth; 10,240 adds per
  1024-element tile.
- **The double-buffering requirement**, framed as M9's **WAR hazard across
  threads inside one statement**; why a trailing barrier does not fix it; why it
  passes at 32 threads (one warp = one instruction) and fails at 64+; the
  two-barrier-plus-temporary alternative measured at **1.05×** slower.
- **Blelloch**: upsweep + downsweep, O(N) work, 2·log N depth, ~2,048 adds per
  tile. **The upsweep IS Module 12's tree reduction** — said explicitly.
- **The identity insertion `x[N-1] = 0`** as the single line that makes the
  result exclusive; the downsweep invariant ("node holds the sum of everything
  strictly to its left") and why seeding with the identity establishes it.
- **The honest work-efficiency caveat**: work-efficient ≠ faster. Three reasons —
  parallelism collapses near the root (1 active thread of 256 at the top level),
  twice the barriers, and a bank-conflicted access pattern. **Measured 1.48×,
  not the 5× the work ratio suggests, and 1.22× once DRAM-bound.**
- **Bank conflicts in the Blelloch tree**: `D(L) = min(2^(L+1), 32)`, so levels
  0–3 give 2/4/8/16 and levels 4–9 give 32. The classic
  `CONFLICT_FREE_OFFSET(i) = i>>5` padding makes the pitch 33 and `gcd(33,32)=1`.
- **What M7's `max(2,D)` law does to the textbook claim**: level 0 is D=2 and
  therefore **free**, so padding must buy nothing there; the naive D-proportional
  model predicts ~5× and the measured win is **1.31× (L2-resident) / 1.06–1.26×
  (DRAM-bound)**. Reported as a correction to the folklore, not a confirmation.
- **Padding's occupancy cost priced**: 32 extra words (128 B) per block; at 8 KB
  + 1024 B reserve it does not change the 6 blocks/SM that 256 threads already
  cap, so here it is free — with the explicit warning to re-measure where it is
  not.
- **Warp-shuffle block scan** as the modern structure: serial per-thread scan in
  registers → `__shfl_up_sync` scan within the warp → scan of warp totals →
  broadcast-add. Two barriers per tile against Hillis–Steele's 12 and Blelloch's
  21. **1.72× over padded Blelloch, 2.48× over Hillis–Steele.**
- **`__shfl_up_sync` clamps, it does not zero-fill** — the `if (lane >= off)`
  guard is load-bearing.
- **Blocked vs striped item assignment**: `s[tid*IPT+k]` (blocked) is required
  for the per-thread serial scan to mean anything; `s[tid+k*BLK]` (striped) is
  4 % faster and **wrong**, which is why CUB has `BLOCK_LOAD_TRANSPOSE`.
- **The multi-block traffic ledger** — the module's organizing idea:
  scan-then-propagate **4N**, reduce-then-scan **3N**, decoupled look-back
  **2N**, CUB **2N**; the measured ranking is predicted by this column alone.
- **Two bandwidth numbers per kernel**: "GB/s vs 2N" (answers delivered) and
  "GB/s real" (bytes moved). A and B sit at 84.6 % / 86.8 % of DRAM peak and are
  still 2.04× / 1.49× slower than CUB. **They are not slow kernels; they are
  fast kernels doing too much I/O.**
- **Decoupled look-back**: per-tile `{flag, aggregate, inclusive-prefix}` state;
  `X`/`A`/`P` encoding; release (`payload, __threadfence(), flag`) and acquire
  (`atomicAdd(flag,0)` spin, `__threadfence()`, payload); warp-parallel
  look-back with `__ballot_sync` + `__ffs` + warp reduction; publishing the
  inclusive prefix as the difference between **O(m) and O(m²)**.
- **Why three state words and not two** — a reader can observe `A` and then read
  the slot after it has been overwritten with the prefix.
- **The forward-progress argument, in full**: dynamic tile claim via
  `atomicAdd(ticket,1)`; holding ticket `t` implies every tile `< t` is owned by
  an already-**resident** block (M1: indivisible, non-migrating); tile 0 never
  waits; induction. **`blockIdx.x` is not sufficient** and the specific
  deadlocking interleaving is constructed.
- **The assumption that remains**, named and labelled: resident-block forward
  progress is not written down in the programming guide.
  **ARCHITECTURE-SPECIFIC / IMPLEMENTATION-DEPENDENT.**
- **Why ITS (M8) is what makes the divergent per-lane spin legal** — the first
  kernel in the course that positively requires independent thread scheduling.
- **`cub::DeviceScan::ExclusiveSum`**, the two-call query/allocate/run protocol,
  and the measured temp storage: **280,575 B for 65,536 tiles = 4.3 B/tile**,
  i.e. the look-back state and nothing else (CUB packs flag+value into one word).
- **Ordered stream compaction** — M10's debt: flags → exclusive scan → scatter,
  with `count = exc[N-1] + flag[N-1]`. Measured **4.75× slower than the atomic
  ticket**, **2.33× after fusing**; and the atomic version's two consecutive runs
  differ in **12,582,783 of 12,582,896 positions**.
- **The price of ordering is traffic, not atomics** — ~8.4N vs ~1.375N, and
  fusion recovers half of it.
- **Bank conflicts as the fourth instance of replay**, after M4 constant memory,
  M7 shared memory, M10 atomic contention.

## CUDA API / intrinsics / syntax introduced

- `__shfl_up_sync(mask, var, delta)` — **new here** (M8/M12 used `_down` and
  `_xor`); the clamping semantics for `lane < delta`
- `__shfl_sync(mask, var, srcLane)` — broadcast after a warp reduction
- `__ballot_sync` + `__ffs` used as a *search* (nearest lane with a property),
  not as an instrument — M8 used them only for observation
- `<cub/cub.cuh>`: `cub::DeviceScan::ExclusiveSum` (query/allocate/run),
  `InclusiveSum`, `ExclusiveScan`, `DeviceScan::*ByKey` named;
  `cub::BlockScan` and `cub::BlockLoad` / `BLOCK_LOAD_TRANSPOSE` named;
  `cub::DeviceSelect::If` named and deferred to M36
- Build flags for CCCL under MSVC: `-std=c++17 -Xcompiler /Zc:preprocessor`
- `atomicAdd(ptr, 0u)` as an acquire-style *load* (M9 idiom, first real use)
- `atomicExch` as a flag publish at device scope
- `__threadfence()` in a genuine cross-block release/acquire pair
- `setvbuf(stdout, NULL, _IONBF, 0)` — house addition: every file in this module
  is unbuffered so that a hang or an abort cannot swallow the output
- SASS read and quoted: `MEMBAR.SC.GPU` (×4), `ATOMG.E.ADD.STRONG.GPU`,
  `ATOMG.E.EXCH.STRONG.GPU`, `VOTEU.ANY`/`FLO.U32`/`POPC`/`SHFL.IDX`
- Tooling: `nvcc -cubin` + `cuobjdump -sass`,
  `compute-sanitizer --tool memcheck` (clean, quoted)

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | A: inclusive vs exclusive on 16 elements, both conversions checked, the lost-total hazard printed. B+C: four tile-scan kernels (Hillis–Steele, Blelloch padded, Blelloch unpadded, warp-shuffle) in one template with identical 2N traffic, at an L2-resident and a DRAM-resident size, so the columns differ only by algorithm; padded and unpadded Blelloch are the same source instantiated twice. Closes with the level-by-level conflict-degree table and the M7 `max(2,D)` reading. |
| `example02.cu` | The traffic ledger: scan-then-propagate (4N), reduce-then-scan (3N), decoupled look-back (2N), `cub::DeviceScan` (2N), at N = 67,108,861. Reports two bandwidth columns per strategy — answers-per-second vs bytes-moved-per-second — plus CUB's temp-storage size and the ratio table. |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Fill-in + design + prediction (§6 types 1, 2, 6) | 5 | Implement Hillis–Steele, Blelloch, and a warp-shuffle block scan; each runs inside the same 3-kernel device scan; validated at 1,048,573 and 67,108,861 | The prediction *is* the trap: the work ratio is 5:1 and the measured time ratio is 1.48× (bucket 3), and at 64 M it falls to 1.22× because everything is at the DRAM roof. Code traps: (a) `s[i] += s[i-off]` is a cross-thread WAR that is **correct at 32 threads**; (b) forgetting `x[N-1]=0` shifts every output by exactly `total` — uniform, monotone, plausible; (c) zeroing without capturing `total` first returns 0 as the tile sum, which is invisible in a single-block test and produces a per-tile sawtooth in a multi-block one; (d) dropping `if (lane >= off)` in the shuffle scan is wrong only in the low 5 lanes of each warp; (e) `texcl = wexcl + wincl` instead of `… - run` is monotone and wrong; (f) the striped item assignment is 4 % faster and produces a scan of a permutation. |
| `exercise02.cu` | Design + fill-in + prediction (§6 types 1, 5, 6) | 5 | Ordered stream compaction of 33,554,393 elements against M10's atomic ticket; order-exactness and determinism both validated | TODO 4 is the headline: `count = exc[N-1]` is off by one **only when the last element passes the predicate**, and with this seed it does. TODO 2 traps: assuming the 32,768 tile totals fit in one tile (first 1 M right, next 32 M wrong), and launching the offset-add pass with a grid-stride grid (it indexes `offs[blockIdx.x]`). TODO 3 trap: scattering with the *inclusive* scan gives a dense, plausible, uniformly-shifted array and writes one past the logical end without faulting. TODO 1 trap: a non-grid-stride predicate kernel leaves 99.8 % of the flags uninitialised. Result the exercise exists for: the scan version is **4.75× slower**, and fusing recovers 2.04× of that with the same scan underneath. |
| `exercise03.cu` | Design + fill-in + prediction, hardest (§6 types 1, 5, 6) | 5 | Single-pass scan by decoupled look-back: state encoding, fences, warp-parallel look-back, dynamic tile claim; 9-size correctness sweep + 200 repeats + timing | TODO 1 is a correctness-of-argument trap: `blockIdx.x` works on this GPU and is unsound, and the exercise asks for the argument, not the code. TODO 4 is not a correctness bug at all — omitting the prefix publish still gives right answers and turns O(m) into O(m²), which only a large test reveals. TODO 3 traps: `while (f != FLAG_P)` is correct and serialises all 65,536 tiles; a plain (non-atomic) flag load can be hoisted out of the loop; omitting the reader-side `__threadfence()` **passes every test on this hardware** and the only evidence is the SASS. Two-word state (value reused for aggregate and prefix) is a real race. Performance trap: the single-pass version is **1.85× faster while reaching a *lower* % of DRAM peak** (71.6 % vs 77.3 %). |

Scoring: Ex1 6 correctness + 2 predictions = 8; Ex2 6 correctness checks
(order-exact, permutation, count, atomic-is-a-permutation, atomic-is-NOT-ordered)
+ 1 prediction = 7; Ex3 3 (9-size sweep) + 2 (200 repeats) + 2 (both strategies
valid at 64 M) + 1 prediction = 8. In all three, `OVERALL: PASS` requires the
prediction as well as the code.

## Measured results recorded (RTX 3500 Ada, CUDA 13.2)

| Quantity | Measured |
|---|---|
| 2N floor, N = 67,108,861 | 1.2428 ms |
| Tile scan, N = 1,048,573 (L2-resident): Hillis–Steele | 0.0328–0.0333 ms, 253–256 GB/s |
| … Blelloch **padded** | 0.0223–0.0227 ms, 369–375 GB/s |
| … Blelloch **unpadded** | 0.0299 ms, 280 GB/s |
| … warp-shuffle | 0.0109–0.0132 ms (L2-resident, 147–195 % of "peak") |
| **Blelloch unpadded / padded, L2-resident** | **1.31×** |
| **Blelloch unpadded / padded, DRAM-bound** | **1.06–1.26×** |
| **Hillis–Steele / Blelloch, L2-resident** | **1.39–1.49×** over 7 runs (work ratio 5.0×) |
| **Hillis–Steele / Blelloch, DRAM-bound** | **1.18–1.29×** |
| **Blelloch / warp-shuffle, L2-resident** | **1.39–2.27×** (noisy; printed, not scored) |
| Multi-block, N = 67,108,861: scan-then-propagate (4N) | 2.94–3.22 ms, 365 GB/s real, 84.6 % |
| … reduce-then-scan (3N) | 2.15 ms, 375 GB/s real, 86.8 % |
| … decoupled look-back (2N) | 1.70–1.94 ms, 277–317 GB/s, 64.1–73.3 % |
| … `cub::DeviceScan` (2N) | 1.4325–1.4389 ms, 373–375 GB/s, 86.4 % |
| **look-back speedup over 3-kernel** | **1.66–1.85×** over 5 runs (ledger predicts 2.0×) |
| **CUB speedup over this look-back** | **1.18×** |
| CUB temp storage, 65,536 tiles | 280,575 B = 4.3 B/tile |
| Compaction (N = 33,554,393, 37.5 % survive): atomic ticket | 0.688–0.820 ms |
| … flags + 3-kernel scan + scatter | 3.269–3.279 ms (**3.80–4.75×** slower) |
| … fused scan | 1.60–1.69 ms (**1.95–2.44×** slower; ~2.0× from fusion alone) |
| Atomic compaction, two runs, positions differing | 12,582,711 – 12,582,848 of 12,582,896 |
| Hillis–Steele single-buffer+temp / double-buffer | 1.05× |
| Warp scan, striped item assignment | 0.956× the time and **incorrect** |
| Decoupled look-back stress | **7,200 launches × 18 grid sizes (1 … 68,360 tiles), 0 failures, 0 hangs**; plus 9 sizes × 200 repeats in the shipped solution |

Absolute times on this laptop part move up to 30 % with thermal state; every
claim above is a ratio or is quoted as a range.

## Assumed from earlier modules

- **M1**: 40 SMs, 6 blocks/SM at 256 threads → 240 co-resident blocks; blocks are
  indivisible, non-migrating, and **do not exist until placed**; 4 warp
  schedulers per SM issuing independently; waves.
- **M2**: `nvcc -arch=sm_89`, launch syntax, `CHECK` macro,
  `cudaGetLastError()` + `cudaDeviceSynchronize()`, `cudaEvent_t` timing.
- **M3**: grid-stride loops (Ex2's predicate and scatter), bounds guards,
  ceil-divide grid sizing, wave-sized grids (240 blocks).
- **M4**: **L1 is not coherent across SMs, L2 is the device coherence point** —
  the entire justification for `__threadfence()` and for atomic flag loads;
  L2 241 cycles, DRAM 575 cycles; 48 MB L2 as the benchmarking hazard;
  `cuobjdump -sass`.
- **M5**: 32 B sectors, coalescing, 432 GB/s, the traffic-counting method.
- **M6**: shared memory, cooperative loading, the WAR/RAW rules of thumb,
  "works at warp width is not evidence of correctness", double-buffering as a
  memory-for-barriers trade, the "already at 70 % of peak → hard 1.4× ceiling"
  argument.
- **M7**: banks, `bank = (addr/4) % 32`, `D = gcd(stride,32)`, **the `max(2,D)`
  cost law**, padding to an odd pitch, padding's occupancy cost including the
  1024 B driver reserve, and the warning that the compiler vectorizes shared
  accesses out from under your analysis (`LDS.128` phase splitting).
- **M8**: warps, `_sync` intrinsics and why the mask exists, **independent
  thread scheduling** (required by the divergent spin), predication,
  `__ballot_sync`/`__ffs`/`__popc`.
- **M9**: the two guarantees of `__syncthreads()`; the uniformity rule and the
  block-uniform early return; fences vs barriers and the three scopes;
  publish/subscribe with a fence; **"`volatile` is not synchronization"**;
  **"there is no cross-block synchronization"** and the 240-block co-residency
  deadlock — which this module must and does confront head-on.
- **M10**: atomics execute at the L2; `atomicAdd` returns the old value; the
  atomic ticket and its uniqueness-without-ordering property; compiler warp
  aggregation (`VOTEU.ANY`/`POPC`/`SHFL.IDX`), verified again here in SASS;
  `racecheck` is blind to global RMW; non-determinism as a real hazard.
- **M12** (concurrent): the reduction ladder, `__shfl_down_sync` warp reduction,
  multi-block finalization. **Used freely and never re-taught** — the Blelloch
  upsweep and the reduce-then-scan pass 1 are both labelled as M12's reduction.
- Spec §12 timing throughout: back-to-back configurations with **rotated order**,
  validation in a separate pass, min of 4 sweeps, duration-based 400 ms warm-up,
  iteration count auto-scaled to ~10 ms segments, ratios as the stable quantity,
  >100 % of peak explained as L2 residency.

## Forward-reference debts PAID here

- **M10 → M13**: "prefix sum as the only way to get *ordered* compaction; atomic
  tickets give uniqueness, never order." Paid in lesson §10 and Exercise 2 in
  full, with the atomic version measured as a different permutation on every run
  (12,582,783 of 12,582,896 positions differing).
- **M9 → M13** (implicit): M9 named the cross-block spin as the canonical
  deadlock and cooperative launch as the only sanctioned escape. This module
  supplies the *other* escape — a dependency order plus dynamic index
  assignment — and states explicitly which of M9's premises it does and does not
  violate. M9's conclusion is not contradicted: the deadlock argument is
  reproduced and shown to apply to the `blockIdx.x` version.
- **M12 → M13**: the reduction's upsweep reappearing as half of a different
  algorithm is said out loud, twice.

## Forward references made

- **Module 14 (histogram)** — deliberately absent; no histogram appears.
- **Module 15 (transpose)** — deliberately absent, per M7's reservation. The
  blocked-vs-striped discussion names `BLOCK_LOAD_TRANSPOSE` as the place a
  shared-memory transpose is needed, without teaching one.
- **Module 36 (Thrust / CUB)** — named as the owner of CUB proper, including
  `DeviceSelect::If` as the one-line version of Exercise 2 and `DeviceScan`'s
  full policy surface. This module uses `DeviceScan::ExclusiveSum` only as a
  measured reference point.
- **Module 19 (occupancy)** — the padding-vs-blocks/SM arithmetic is done by
  hand and flagged as M19's territory.
- **Module 23 (Nsight Compute)** — named as the tool that would settle the fence
  question, with the standing note that `ncu` is unavailable here
  (`ERR_NVGPUCTRPERM`).
- **Radix sort, CSR row pointers, run-length encoding** named as scan
  applications without module numbers.
- **Segmented scan / `ScanByKey`** named only.

## Constraints observed

- No histogram (M14), no transpose (M15), no GEMM.
- Reduction assumed, not re-taught: the only reduction code in the module is the
  `reduceTilesKernel` in `example02.cu`, explicitly labelled "Module 12 owns
  this."
- No sm_90+ features; no thread-block clusters, no TMA, no cooperative launch.
- Everything builds warning-clean with `nvcc -arch=sm_89 -O3`.
- Nothing under `module13/` reveals an answer; all predictions are gated on a
  reader-supplied value and the answer key lives only under `solutions/`.
- No binaries left behind.

## Known issues / honesty notes

- **`ncu` unavailable** (`ERR_NVGPUCTRPERM`, spec §12). No counter output is
  quoted anywhere. The bank-conflict claim in §6a rests on a controlled A/B
  measurement of the same source instantiated twice, not on
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`.
- **No tool on this machine validates the look-back's fence placement.**
  `racecheck` is a shared-memory checker; `memcheck` is clean (quoted) but
  cannot see a missing fence. The solution says so and offers SASS + stress as
  the only available evidence. Omitting the reader-side `__threadfence()` passes
  every test here, which is documented as a hazard, not hidden.
- **The forward-progress guarantee is not in the programming guide.** Stated in
  the lesson, in CYU Q4, and in the Exercise 3 solution, each time labelled
  implementation-dependent rather than asserted as safe.
- **At 64 M the four tile-scan algorithms are within 20 % of each other and the
  Blelloch/warp-shuffle ordering is not stable run to run.** This is reported as
  a result rather than smoothed over, and is the reason Exercise 1 scores its
  ratio prediction at the L2-resident size instead.
- **The look-back reaches a lower % of DRAM peak than the strategy it beats.**
  Documented prominently rather than buried; it is the module's best single
  illustration that "% of peak" is not a figure of merit.
- Exercise 3 can be made to hang by a partially-filled set of TODOs (a look-back
  without a publish). The file warns about this in its header and recommends a
  bounded spin during development, per M9 Exercise 2.
