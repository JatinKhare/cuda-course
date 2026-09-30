# Cross-module index — state of the course after Modules 1–10

Authors of Modules 11+: read this, then read all ten
`solutions/module{01..10}/MANIFEST.md` files for full detail. This file records
the integration decisions; the manifests record the specifics.

**Parts I–III (Modules 1–10) are COMPLETE and verified.** 10 lessons,
19 worked examples, 29 exercises, 29 verified solutions.

---

## 1. What the reader already knows

| M | Delivered |
|---|---|
| 1 | Latency vs throughput, Little's Law, SIMD vs SIMT, SM internals (4 processing blocks, 1 warp scheduler each, 16384-reg slice, 32 FP32 lanes, 128 KB unified L1+SMEM ≤100 KB shared), eligible/stalled scoreboard, free context switch, warp formation from linearized threadIdx, 6-step launch path (pushbuffer → GigaThread → placement gating → indivisible non-migrating blocks), waves and tail effects |
| 2 | nvcc basics, `__global__`/`__device__`/`__host__`/`__host__ __device__`, all four launch params (3rd and 4th deferred), async launch, `cudaDeviceSynchronize`, two-part error checking, sticky vs non-sticky errors, `CHECK`/`CHECK_KERNEL` macros, device `printf`, cudaMalloc/Memcpy/Free, host-vs-device pointer distinction |
| 3 | All built-ins as `dim3`, 1D/2D/3D, **the linearization rule** (x fastest → warp formation), 1:1 / 1:N / grid-stride mappings, flattening, row- vs column-major, stride ≠ width, bounds guards and their predication cost, block sizing (1024 → 66.7% ceiling on sm_89), ceil-divide overflow, 65535 y/z limit |
| 4 | Full storage map with measured latencies (L1 40.5 / L2 241.3 / DRAM 575 cycles), registers, **local memory is DRAM** (dynamic indexing and spills distinguished), shared memory *named and priced only*, L1 non-coherence across SMs, 48 MB L2 as the benchmarking hazard, global, constant memory broadcast-vs-serialize (86× spread measured), `__ldg`/`const __restrict__`, pinned vs pageable, `-Xptxas -v`, `cuobjdump -sass`, template-parameter dispatch trick |
| 5 | 32 B sectors / 128 B lines, the sector-counting method, alignment, `float4` vectorized loads and tail handling, AoS vs SoA (penalty = record_size / bytes_used, **not** a property of the struct), write coalescing, effective vs implied-DRAM bandwidth, row pitch as a layout decision |

| 6 | Shared memory as software-managed scratchpad; reuse factor K and the halo tax; scope/lifetime tied to non-migrating blocks; static vs dynamic `extern __shared__` + the 3rd launch parameter; carving multiple arrays with alignment; capacity→occupancy (48 KB default / 99 KB opt-in, **1024 B driver reserve**, 128 B granularity); cooperative loading where load-mapping ≠ compute-mapping; halo/ghost cells |
| 7 | 32 banks × 4 B, `bank = (addr/4) % 32`; conflict = N distinct words in one bank → N replays; **broadcast is free**; the degree-counting method; **cost law is `max(2,D)` not `D`, so 2-way conflicts are free on Ada**; padding vs XOR swizzle and their different costs; phase splitting for 8 B and 16 B accesses; `ncu` metric names (tool unavailable here) |
| 8 | Warp formation and partial warps; instruction issue and the SIMT contract; active masks; **divergence is warp-local** (1.96× vs 1.02× measured); cost is additive not max; predication vs real branching with three distinct SASS shapes; the predication flip point (7→8 FFMAs); `BSSY`/`BSYNC`; **independent thread scheduling**, no post-dominator guarantee, why everything is `_sync`-suffixed; `__activemask()` cannot distinguish predication from branching |
| 9 | `__syncthreads()` as **two separate guarantees** (execution barrier + block-scope memory fence); the uniformity rule and UB shapes; **on sm_89 a divergent barrier usually corrupts silently rather than hanging** (`bar.sync` counts warps; exited threads are subtracted); `__syncthreads_count/and/or`; `__syncwarp`; `__threadfence_block/—/_system` and fence-vs-barrier; **`volatile` is not synchronization**; `cuda::atomic_ref`, `cuda::barrier`, memory scopes; cooperative groups basics; no cross-block sync without cooperative launch, and why spin-waiting deadlocks |
| 10 | RMW races and why barriers don't fix them; atomics execute at L2 (the device coherence point) — **the answer to M4/M5's L1-incoherence debt**; the full atomic API incl. `atomicCAS` as the universal primitive and the old-value return; shared vs global atomics (48.2×); **contention economics: cost tracks addresses-per-warp, not atomic count; an uncontended atomic costs what a plain store costs**; L2 slice hashing (K distinct addresses only give K-way parallelism if they hash to K slices); privatization and warp aggregation *and where each stops paying*; `ATOM` vs `RED` in SASS; compiler warp aggregation (`VOTEU.ANY`/`POPC`); float `atomicAdd` non-determinism |

## 2. Forward-reference debts wave 2 MUST pay

Earlier modules promised these by name. Delivering them is not optional.

- **M6 owes:** shared memory proper — the 3rd launch parameter `sharedBytes`
  promised in M2; "block-size amortisation" promised in M3; the "introduced
  only, Module 6 owns it" hand-off from M4; and M5's case where *neither* the
  read nor the write side can be coalesced simultaneously, which is the
  motivating problem for staging through shared memory.
- **M7 owes:** bank conflicts framed as **the same replay mechanism** M4 used
  for constant-memory serialization, and explicitly contrasted with the
  constant-memory *broadcast* case from M4 and M5.
- **M8 owes:** divergence vs mere predication — M3 measured the bounds-guard as
  predicated and promised M8 would make the distinction precise; active masks;
  reconvergence; **independent thread scheduling (sm_70+)**, promised by M1.
- **M9 owes:** `__syncthreads` semantics (used from M6 with "Module 9 makes this
  precise"); memory ordering and fences; **"`volatile` is not synchronization"**,
  promised explicitly by M4; barrier-in-divergent-control-flow.
- **M10 owes:** races; atomics; **cross-SM L1 incoherence as the reason device-scope
  atomics are needed**, promised by M4 and M5; the in-place stencil cross-block
  race promised by M3; `compute-sanitizer --tool racecheck`, promised by M2.

Do NOT re-teach what column 1 already delivered. Cite it and build.

## 3. Continuity opportunities — use these

Reusing an earlier module's exact problem lets the reader measure their own
progress. Strongly preferred over inventing a fresh toy problem.

- **M6 tiled stencil:** M3 exercise 1 is a 5-point clamped stencil on a
  1021×733 and 4093×3079 image. Measured baseline **on the 4093×3079 image**:
  **(32,8) block = 0.3213 ms**, `(1,256)` = 1.0022 ms. (The 1021×733 image runs
  in 0.0107 ms and is L2-resident — do not quote bandwidth from it.) Build the tiled version against these numbers and have
  the harness print the M3 baseline alongside, so the reader sees what shared
  memory did and did not buy. Be honest if the win is small — a 5-point stencil
  has low reuse, and that is itself the lesson about when tiling pays.
- **M7:** M5 exercise 3 taught row pitch as a layout fix for global memory. Bank
  conflict padding is the *same idea one level down*. Make that parallel explicit.
- **M10:** M4's constant-memory serialization and M7's bank conflicts are both
  replay; atomic contention is a third instance. Close the loop.

## 4. House conventions now established — match them

- Output ends with a single `OVERALL: PASS` / `FAIL` line; exercises with
  prediction components score them (e.g. `10/10`) and require both correct
  numerics and correct predictions to pass.
- Unfilled exercises exit gracefully with `Set TODO n first.` and return 0.
  A deliberate-crash debugging exercise is the one permitted exception.
- Deterministic index-derived or seeded initialization. Never unseeded `rand()`.
- Timing per spec §12 (added after wave 1 — read it, it is mandatory): all
  configs timed back-to-back, validation in a second pass, min-of-N over 3–4
  sweeps, duration-based clock warm-up, buffers > 4× the 48 MB L2 for DRAM
  measurements, ratios reported as the stable quantity.
- Report `% of peak` against 432 GB/s, and **explain** any value over 100%
  rather than hiding it (it means L2 residency, or read+write of the same array).
- Solution `.md` quotes real observed output, real `-Xptxas -v`, real SASS,
  real `compute-sanitizer` output.

## 5. Difficulty calibration reached

Wave 1 shipped 3–4 TODOs per exercise, and the best traps were ones where the
**wrong answer still passes validation** — M3's axis assignment (3.5× slower,
no test catches it), M5's tail guard that processes nothing, M4's `#pragma
unroll` that does nothing. Wave 2 must exceed this.

Per spec: **M6–M7 use 2–4 TODOs; M8–M10 use 3–5, and at least one TODO per
exercise in M8–M10 must require designing a strategy, not filling an
expression.** Scaffolding steps down noticeably at M8.

## 5b. Debts owed by Modules 11–18 (recorded by Modules 1–10)

- **M11** owes: grid-stride loops revisited at the bandwidth ceiling (M3 measured
  348.5 GB/s / 80.7% with a wave-sized grid); vectorized `float4` loads and tail
  handling at scale (M5 introduced them).
- **M12** owes: the reduction ladder promised by M8 (divergence), M9 (the
  `volatile` warp-synchronous tail that is **broken** on sm_70+, and
  `__shfl_down_sync` as its correct replacement — M9 and M30 both point here),
  M10 (fixed-order reduction as the **deterministic alternative** to float
  `atomicAdd`, whose non-determinism M10 measured as 10 distinct bit patterns in
  10 runs).
- **M13** owes: ordered compaction, named by M10 as the only route to it.
- **M14** owes: **the full privatized histogram** — M10 shipped only the
  mechanism and the traffic arithmetic and explicitly deferred bin replication,
  >SMEM bin counts, and coarsening here. M10's measured table of *where
  privatization stops paying* (48.2× at K=1, 6.5× at K=32, 0.22× at K=4096) is
  the quantitative foundation M14 must build on.
- **M15** owes: transpose. M7 **deliberately avoided using transpose as its
  vehicle** so M15 keeps it intact. M5 also deferred both the device-side
  AoS→SoA conversion and deinterleave here.
- **M16–M17** owe: the K/H (reuse vs halo) argument's real destination — M6
  showed tiling *loses* on a 5-point stencil and named GEMM as where it wins;
  **register blocking is the missing ingredient M6 named**. M17 also owes
  double-buffering, named by M6 as the WAR-hazard alternative and by M9 as the
  germ of software pipelining.
- **M18** owes: the condition under which swizzle beats padding — M7 measured
  padding winning by 19% on an LSU-bound kernel and stated the answer flips in
  GEMM, where shared-memory capacity limits tile size.

## 6. Themes already used — do not repeat

5-point stencil (M3, M6), SAXPY (M3 example), Horner polynomial (M2),
16-tap FIR (M4), particle AoS/SoA update (M5), pointer chase (M1 example,
M4 ex1), dependent-FMA compute-bound kernel (M1 ex2), column-subset row scan
with pitch (M5 ex3), 2D box filter R-sweep (M6 example), RBF scatter
interpolation (M6 ex2), `fold_blend` (M6 ex3), conflict-degree sweep (M7),
data-dependent work binning (M8 ex2), `warpSmooth` (M8 ex3), 8-fragment
barrier classification (M9 ex1), bounded-spin producer/consumer (M9 ex2),
category counting with contention (M10 ex2), CAS argmax (M10 ex3).

Added by wave 3/4: 4-stage fusion chain and SiLU gate (M11), the six-rung
reduction ladder and segmented ragged-row reduction (M12), Hillis–Steele /
Blelloch / decoupled look-back and ordered compaction (M13), **naive GEMM, the
loads-per-FMA probe, and the Freivalds validation probe (M16)**.

**Still unused and reserved:** tiled GEMM and register blocking (M17/M18),
sorting, sparse formats, n-body, image convolution with large radius.

## 6b. Established measured baselines Modules 11–18 can build against

| Quantity | Measured |
|---|---|
| DRAM pin peak (use for **bounds**) | 432.0 GB/s |
| **FP32 compute ceiling, measured** | **17,787–18,256 GFLOP/s** (implied clock ~1.78 GHz) |
| **Machine balance** | **43–44 FLOP/byte** |
| ⚠️ `cudaDevAttrClockRate`-derived FP32 peak is **too low** | It yields 15,821 GFLOP/s, **below** the measured ceiling by 1.15× — so "% of FP32 peak" built on it exceeds 100%. Never use it. |
| ⚠️ `clock64()` clock recovery has a failure mode | Reads 2.12 GHz for a single resident warp but **0.44–0.66 GHz** for a full-occupancy one-wave grid, because block 0's cycle delta is no longer the kernel's duration. Prefer recovering the clock from a measured saturating FFMA throughput. |
| ⚠️ Ceilings are not session-reproducible | Same binary measured 294–411 GB/s across sessions. `nvidia-smi` during a pure-read kernel shows memory clock 8801 MHz and **SM clock 285 MHz** — the power manager trades away whatever the kernel isn't using. **Convention: measured figure for a ceiling, 432 GB/s pin peak for a bound.** |
| cuBLAS SGEMM | 8150–8705 GFLOP/s = 45–48% of the FP32 ceiling |
| Naive GEMM | 1275–1348 GFLOP/s = **12–13% of cuBLAS**, 7% of ceiling |
| FMAs needed per **operand-fetch instruction** for 80% of ceiling | **6.4–6.5.** ⚠️ **Restated — M16 originally said "per global load", which is wrong and confusing.** M16's probe measured *LSU* behaviour, not the global address space; M17 reran it with `LDS` and got the same curve to within 1%. The threshold applies to a memory instruction of **any** address space. |
| Why that restatement matters | M17's tiled kernel reaches **8.00 FMAs per global load — clearing M16's threshold — while running at 9.4% of peak.** That reads as a contradiction and is not one: its FMAs per *shared* load is still 0.50, and per memory instruction of any kind **0.47, slightly worse than naive.** Tiling changed the opcode, not the count. |
| The same function at both levels (M18) | global: `BM·BN/(BM+BN)` · shared: `TM·TN/(TM+TN)`. **BK cancels out of both.** Square tiles optimal by AM–GM. 6.5 is unreachable as a scalar count (needs TM=TN=13) but reachable as an instruction count: **10.7 FMAs per shared-memory instruction at 8×4, 16.0 at 8×8.** |
| Shared-memory read bandwidth | **5.38–5.40 TB/s scalar `LDS`, 10.24–10.30 TB/s `LDS.128`** (ratio 1.90 = M7's two-cycle floor). A GEMM needing 8 B/FMA at the FP32 ceiling would demand **72 TB/s**, so any 2-shared-loads-per-FMA kernel is capped at **7.4–14.2% of the compute ceiling.** This is the bound that makes register tiling mandatory. |
| Tiled GEMM (M17) | 1703–1722 GFLOP/s = 1.29–1.32× naive, 17.5–20.2% of cuBLAS, 9.4% of ceiling. Best tile **16×16, BK=16**. |
| cuBLAS SGEMM, widened across all measurements | **8150–9700 GFLOP/s = 45–53% of the FP32 ceiling** |
| `__restrict__` on naive GEMM | **Null result** (0.991–0.996×) |
| **Streaming ceiling, properly warmed** | **410.5–410.7 GB/s = 95.0% of peak** (M12, after a **1500 ms** warm-up). This is the real number. |
| ⚠️ Warm-up length changes the answer by 10% | Same binary, same data: **372–373 GB/s after 400 ms, 410.5–410.7 GB/s after 1500 ms.** 400 ms ramps the SM clock but not the memory P-state. Modules 6–11 used 400 ms, so their **ratios are sound but their absolute GB/s figures are ~10% low.** Spec §12 rule 4 now mandates 1500 ms. |
| Earlier (400 ms warm-up) figures, for reference | ~366–395 GB/s; 395.2 for a 1:1 bare copy (M11). Superseded as an absolute ceiling. |
| SAXPY, grid-stride, wave-sized grid | 348.5 GB/s / 80.7% (M3) |
| ⚠️ Grid-stride is not automatically fastest | M3 measured grid-stride 348.5 vs 1:1 297 GB/s. M11 measured the **opposite** for a bare copy: 1:1 = 395.2 vs best grid-stride 384.3. Both are correct — M3's kernel pays a bounds test and a fresh 64-bit index per element with no loop to amortise them. **Frame grid-stride as correctness under any launch shape and a machine-sized grid, not as a speed win.** |
| In-place SAXPY (3N traffic model) | 376.1 GB/s — the naive 2N model reports a misleading 250.8 GB/s |
| Write-allocate | Real: dense vs stride-2 writes of identical useful bytes = 3.95×. But a **full-sector fill hits 375.0 GB/s at 1N — there is no unconditional write-allocate.** |
| `__stcs` / `__stwt` streaming stores | **Null result.** No effect on full- or partial-sector stores. |
| `-use_fast_math` on bandwidth-bound elementwise | **Null result** (0.998×) |
| Transcendental compute/memory crossover (elems/thread) | `sinf` K=8, `__sinf` K=32, `expf` ≈8, `__expf` ≈16–32. FFMA never crosses. |
| Fusion | No cliff to ≥6 input streams; registers 34→80 with no spills. At 4 MB arrays launch overhead (~10 µs) dominates instead. |
| L2-resident apparent bandwidth | up to ~1305 GB/s (302% of "peak") |
| Dependent-load latency | L1 40.5 / L2 241.3 / DRAM 575 cycles |
| Shared-memory conflict cost law | **`max(2, D)` for 4-byte accesses only.** On `LDS.128` (16 B) the phase split leaves no spare cycle: **cost ∝ D, no floor of 2.** M18 isolated this at 1.61× measured vs 1.67× predicted, with byte-identical SASS. |
| Design rule from that | **`BN/TN >= 16` once the thread tile is read with float4** — a constraint no resource counter reports |
| Register allocation granule | **8 registers per thread, allocated per warp**. A per-thread model looks nearly right and is wrong. |
| **"Occupancy ≠ performance", the canonical number** | **100% occupancy is 18.7× SLOWER than 33%** on the same GEMM source via `__launch_bounds__` (390 vs 7267 GFLOP/s). Occupancy is also **non-monotone**: 25→33% wins, 33→67% is a 9× loss. |
| Spills: the cliff is not the first byte | An 80 B spill that buys a 4th block is **1.12× faster**. The cliff is where the *accumulators* spill. |
| Register-tiled SGEMM ceiling reached | **7399–8336 GFLOP/s = 41–46% of FP32 ceiling = 91–108% of cuBLAS** (93% at 2048³) |
| **⚠️ Padding: four measured, mutually contradictory results — read the rule below** | M7 **+19%** (padding wins over swizzle, LSU-bound kernel) · M15 **tie** · M17 **−40%** (padding is a *loss* in row-major tiled GEMM) · M18 **+23%** (padding wins over swizzle in register-tiled GEMM) |
| **The unifying rule** | **1. Pad only after *measuring* a conflict of degree ≥ 4.** M17's row-major tile is degree 1 everywhere (`As[ty][k]` broadcasts, `Bs[k][tx]` is unit-stride) — there was nothing to fix. **2. Before padding, check the SASS for a vector merge you are about to destroy.** A pitch of `T+1` floats breaks the 16 B alignment ptxas needs to merge four contiguous reads into `LDS.128`: M17 measured 20 shared instructions becoming 32. **3. Padding genuinely wins where a real conflict exists** — M17's *transposed* A-tile store (degree 8 at T=16, 32 at T=32) gained 1.05–1.18×; M18's register-tiled A tile gained 1.23×. **4. The pad must be 4 floats, not 1** (M18): pad-by-1 leaves degree 4, a partial fix that looks like a fix, and only a multiple of 4 preserves float4 alignment. |
| ⚠️ M7's "the answer flips in GEMM" — resolved | The premise (capacity binds) is **false for M17**, where one output per thread caps the tile at 32×32 and uses 8 KB of a 48 KB budget. It is **also false for M18**, where registers bind in all 23 configurations tested. The regimes where M7 is right are Tensor Cores and multi-stage `cp.async` — handed to M32/M33–34/M43. |
| ⚠️ M6's occupancy formula is incomplete | It omits **registers**, which M1 listed in the placement gate. M17 measured two kernels with identical threads and identical shared memory differing 3 vs 2 blocks/SM purely on 44 vs 40 registers. M19 owes the four-limiter version. |
| Padding vs XOR swizzle in GEMM | **Padding wins, 1.23× at BK=8** — M7's prediction that the answer flips in GEMM is **refuted**. Registers bind in all 23 configs tested; the swizzle also blocks the compiler's contraction into `LDS.128`. Pad must be **4**, not 1. |
| Double buffering on Ada fp32 GEMM | **A measured LOSS** (0.81–0.92×). The 48 MB L2 holds the working set, so the hidden latency is a 241-cycle L2 hit that 16–24 warps already cover. `cp.async` (M32) is what removes the cost. |
| cuBLAS SGEMM, re-measured | 7258–8370 GFLOP/s under a stream-then-compute warm-up (M16 recorded 8150–8705 under a GEMM warm-up) |
| Uncontended global atomic | ≈ cost of a plain store |
| Max contended atomic penalty | 47.9× (K=1) |
| Divergence cost, 2-way | 1.96× |
| Blocks/SM @256 threads vs shared bytes | 6 ≤12288 B, 5 @16384, 3 @25600, 2 @49152, 1 @65536 |

## 7. Known hardware quirks discovered during wave 1

- SM clock swings 0.49–2.04 GHz; memory P-state ramps 6001 → 8001 → 9001 MHz
  (288 / 384 / 432 GB/s); a software power cap engages after ~30 s of sustained
  streaming. Duration-based warm-ups and min-of-N are mandatory, not optional.
- `cudaDevAttrClockRate` reports 1.545 GHz and is **wrong**; recover the true
  clock via `clock64()` if you need compute-peak percentages.
- `compute-sanitizer` does **not** detect local-memory spills (M4 verified this)
  — it is the right tool for races and illegal accesses, not for everything.
- Pinned vs pageable H2D on this Windows/WDDM driver measures only 1.04–1.15×,
  not the folklore 2×.
