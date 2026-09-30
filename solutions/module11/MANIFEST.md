# Module 11 manifest

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module11/lesson.md`, `module11/example0{1,2}.cu`,
`module11/exercise0{1,2,3}.cu`.
Solutions: `solutions/module11/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module11/check_your_understanding.md`,
`solutions/module11/MANIFEST.md`.

All eight `.cu` files verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2,
RTX 3500 Ada), warning-clean. Both examples print `OVERALL: PASS`; all three
solutions print `OVERALL: PASS` with full scores (8/8, 7/7, 7/7). All three
shipped exercises compile with TODOs blank and exit gracefully with
`Set TODO n first.` and return 0. No binaries committed.

**Part IV opens here.** This module is deliberately not a beginner vector-add
module: the reader already wrote SAXPY in M3 and counts sectors from M5. Its
job is the *discipline* (compute the floor before writing the kernel) and the
*bridge* from Part II's transaction theory to Part IV's algorithms.

---

## Concepts taught

- **Compulsory traffic** — the bytes that must cross the DRAM pins at least
  once: each distinct element read once regardless of how many times the source
  names it, each element written once, an array that is both counted twice.
  **PORTABLE CUDA CONCEPT.**
- **The time floor** = `compulsory_bytes / achievable_bandwidth`, computed
  *before* writing the kernel; and the corollary that a measurement below the
  floor means the model is wrong, not the hardware (usually L2 residency).
- **`x floor` as the reporting unit** — a kernel at 1.0× is finished; GB/s alone
  cannot distinguish a 2N kernel from a 4N one.
- **The in-place read-modify-write trap**: `y[i] += a*x[i]` moves 3N, not 2N.
  Measured: 376.1 GB/s at 3N versus an apparent 250.8 GB/s at 2N, i.e. the wrong
  model turns a finished kernel into an apparent 58%-of-peak defect.
- **Write-allocate as a consequence of partial sector coverage, not of
  writing.** Full-sector stores cost 1N (fill measures 375.0 GB/s against a 1N
  model); a stride-2 store costs 4× (3.95× measured) because every 32 B sector
  is half-written and must be read-merge-written.
- **Where the merge happens** — at L2, which is why a misaligned *streaming*
  kernel escapes the penalty (the neighbouring warp's sector is resident) and a
  scattered one does not.
- **Cache-residency hints are not traffic hints**: `__stcs` / `__stwt` change the
  SASS opcode and change nothing measurable, including on the partial-sector
  case. **Documented null result.**
- **Reading a grid sweep by total resident threads**, not by blocks/SM or
  threads/block; the plateau and its location.
- **Little's Law applied to the memory system**: `bandwidth × latency` bytes in
  flight (≈116 kB here), and the equivalence of "more threads" and "more
  outstanding requests per thread" as ways to supply it.
- **Memory-level parallelism (MLP) per thread** as an explicit quantity, and
  **ILP as a substitute for occupancy** — 2× measured at 1 block/SM, 1.00× at
  8 blocks/SM.
- **Hoisted versus serialized loads**: `#pragma unroll 1` as the instrument that
  isolates MLP with traffic, arithmetic and coarsening factor all held constant.
- **Coarsening a saturated kernel makes it longer, not faster** — measured
  regression at C = 8 and 16 at full occupancy.
- **Vectorization at scale**: identical sector counts, one quarter of the memory
  instructions per element, 4× payload per outstanding-request slot; therefore
  1.11× at 1 block/SM and 1.00× at 8. **A latency/issue optimization, not a
  bandwidth one.**
- **`float4` alignment, truncating `nVec`, and exactly-once tail processing** at
  a size where they must be handled (`N = 40,000,001`, non-idempotent operation).
- **Kernel fusion as traffic deletion**: 10N → 7N → 5N ladder measured at
  3.73 / 2.66 / 1.92 ms.
- **Fusion deletes only the traffic of values with no external consumer** — an
  intermediate somebody else reads is an output and must be written.
- **Decomposing the fusion speedup**: `speedup = (T_u/T_f) × (B_f/B_u)`, and the
  measured fact that the bandwidth ratio is not 1 and changes sign with array
  size (0.971 at 128 MB, 1.112 at 256 MB).
- **Fusion depth**: achieved bandwidth rises ~6% from 2 to 3 input streams and
  is flat to at least 6; register pressure (34 → 40 → 48 → 64 → 80 for 2 → 4 →
  16 → 24 → 32 inputs, no spills) cuts occupancy but never below what a
  bandwidth-bound kernel needs. **Documented negative result: no fusion cliff on
  this hardware.**
- **When elementwise work is not bandwidth-bound**: the arithmetic crossover.
  64 chained FFMAs per element cost 5%; `sinf` crosses at K = 8, `__sinf` at
  K = 32, `expf` at K ≈ 8, `__expf` at K ≈ 16–32.
- **Why the intrinsic shifts the crossover 4× and not 15×**: `MUFU` issues on 4
  SFUs per processing block against 32 FP32 lanes, so one SFU op ≈ 8 FP32 slots.
- **`sinf`'s composition from SASS**: three-term Cody–Waite π/2 reduction plus a
  minimax polynomial plus a guarded **FP64** Payne–Hanek fallback (`I2F.F64.S64`
  / `DMUL` / `F2F.F32.F64`) — and Ada runs FP64 at 1/64 rate.
- **`-use_fast_math` does not speed up a bandwidth-bound elementwise kernel**
  (measured 0.998×). **Documented folklore contradiction.** Also: nvcc 13.2
  defines no preprocessor macro for the flag, so a file cannot detect it.
- **The IEEE divide is not one instruction**: `FCHK` in the SASS, and
  `__frcp_rn` / `__fdividef` as its replacements, with `__fdividef`'s
  `|y| < 2^126` restriction named.
- **Why shared memory does not appear in an elementwise module**: reuse factor
  K = 1 by definition, so Module 6's cost model says it can only lose.
- **Model domains**: at 4 MB arrays the same fused chain measures 3.31× against
  a 2.00× traffic prediction because launch overhead, not bandwidth, is the
  dominant term. A model has a domain; outside it, it predicts a term that no
  longer dominates.

## CUDA API / intrinsics / syntax introduced

- `__stcs`, `__stwt` (and `__ldcs` named) — cache-residency store/load hints;
  their real SASS: `STG.E.EF` and `STG.E.STRONG.SYS`
- `__expf`, `__sinf` — SFU intrinsics; `MUFU.EX2`, `MUFU.SIN`, `MUFU.RCP` in SASS
- `__frcp_rn`, `__fdividef` (with its documented input range)
- `-use_fast_math` as a compiler flag, verified by SASS diff
- `cudaFuncGetAttributes` → `numRegs`, used at runtime to compute resident warps
- `float2` alongside `float4` (M5 introduced `float4`); `LDG.E.64` / `STG.E.64`
- `#pragma unroll 1` as a microbenchmark instrument (M7 introduced it as a
  hazard; here it is used deliberately)
- `cudaDeviceProp.l2CacheSize` for sizing the L2-residency warning
- FNV-1a hashing of expected answers so that answer-checking code can live in
  `moduleNN/` without revealing the answer

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | **Counting the floor.** A: six elementwise kernels with compulsory traffic, floor, measured and `x floor`, against a streaming ceiling measured in the same rotated sweep; the in-place SAXPY row spelled out. B: write-allocate, dense vs stride-2 writes of identical useful byte counts, 3.95× measured, with the validation pass (odd elements preserved) doubling as the proof of the mechanism. C: `__stcs` / `__stwt` null result. |
| `example02.cu` | **Reaching the floor.** A: 5×6 grid sweep plus the 1:1 mapping, with Little's Law arithmetic derived from the measurement. B: `float`/`float2`/`float4` at 1 and 8 blocks/SM with the SASS instruction census. C: hoisted vs `#pragma unroll 1` coarsening at two occupancies, the MLP experiment. D: 5 functions × 6 values of K, the arithmetic crossover, with `__sinf` accuracy checked in the validation pass. |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Optimization + prediction (spec §6 types 4, 6) | 5 | Bring an in-place clamped blend from 24.5% to ≥95% of a measured streaming ceiling under an 8,192-thread budget; `N = 40,000,001` | **Fixing the launch is worth 2.7× and tops out at 66% of the ceiling** — every budget-respecting (grid, block) pair lands in 65–67%, so the gate cannot be met without finding the access-width / MLP lever, which is never named. The operation is **not idempotent**, so a tail placed inside the main loop updates the tail element ~1,220 times and only the tail column reports it. The careful-looking `if (t >= n4 && t < n4+tail)` form never fires, because the grid (8,192) is far smaller than `n4` (10,000,000). `(N+3)/4` overruns by 12 B without faulting. The `<<<1,32>>>` validation catches every kernel that assumes a minimum grid. TODO 1's reference is FNV-hashed so the exercise file does not contain it. |
| `exercise02.cu` | Performance reasoning + design (spec §6 types 5, 6) | 5 | Predict the fusion speedup from traffic alone, build partial and full fusions, then explain the gap | **`m` is a required output**, so the best fused traffic is 5N, not the 4N everyone writes first; the 4N kernel is faster and wrong, and the harness prints "d is right and m is wrong -- you fused away a required output". TODO 4 is an algebraic identity *by design* — the exercise is locating the traffic model's hidden `B_f/B_u = 1` assumption, not predicting a number. TODO 5's folklore answer (fusion kills occupancy) is **wrong here**: registers do climb to 80 at 32 fused inputs, but 24 warps/SM is still 3× what the bus needs. The depth sweep found **no cliff**; an early draft's apparent 413 GB/s at 8 inputs was a repeated pointer in the input list, and one run in six showed a spurious 0.74× cliff that was a memory P-state event. |
| `exercise03.cu` | Predict-then-measure (spec §6 type 2) | 5 | A SwiGLU-style SiLU gate, then the arithmetic crossover for `sinf` / `__sinf` by sweeping K | **`-use_fast_math` does not help** (measured 0.998×) — the folklore answer to TODO 5 is wrong and the exercise scores it. Replacing `expf` but leaving the IEEE divide alone captures half the instruction saving (`FCHK` × 5 in the SASS) and, on a memory-bound kernel, half of nothing. The `__sinf` crossover shift is **4×, not 15×**, because `MUFU` runs on 4 SFUs against 32 FP32 lanes. `__fdividef` silently returns 0 for `|y| ≥ 2^126`, so the validator also counts exact zeros. Crossovers are octave-scored because the estimate is only good to a factor of two. |

Scoring: Ex1 8 points (4 numeric checks incl. a degenerate-launch run, budget,
2 predictions, 1 perf gate); Ex2 7 points (2 hashed traffic answers, 2 kernels,
the reconciliation identity, an occupancy prediction, a 1.70× gate); Ex3 7
points (2 hashed answers, accuracy + not-slower gates on the reader's kernel,
2 octave-scored crossovers, 1 hashed prediction).

## Assumed from earlier modules

- **M1**: Little's Law; SM structure (4 processing blocks, 32 FP32 lanes and 4
  SFUs each); warp scheduling; waves and tail effects; 432 GB/s; 40 SMs.
- **M2**: `nvcc -arch=sm_89`, launch syntax, `CHECK`, both `cudaGetLastError()`
  and `cudaDeviceSynchronize()`, `cudaEvent_t` timing.
- **M3**: grid-stride loops and their four justifications; the linearization
  rule; bounds guards; the measured SAXPY baseline 348.5 GB/s / 80.7%; the
  1:1-vs-grid-stride comparison, which this module revisits and partly reverses.
- **M4**: the storage map and measured latencies (DRAM 575 cycles); 48 MB L2 as
  the benchmarking hazard; `cudaMalloc`'s 256 B alignment; `-Xptxas -v`;
  `cuobjdump -sass`; `const __restrict__` and `LDG.E.CONSTANT`; FP64 at 1/64
  rate on Ada.
- **M5**: 32 B sectors and 128 B lines; the sector-counting procedure; `float4`
  as a latency/issue optimization with identical sector counts; the 16 B
  alignment requirement and `cudaErrorMisalignedAddress`; write-allocate stated;
  the measured 87% streaming ceiling; measuring a ceiling in-loop and reporting
  `% of stream`.
- **M6**: the reuse-factor cost model for shared memory (cited to explain why
  shared memory is absent here).
- **M7**: `#pragma unroll 1` and "confirm in the SASS that the instruction under
  test is the one executing" (cited as the reason the MLP experiment checks its
  own SASS).
- **M9**: barriers in divergent control flow (cited as why not to add one
  between a vectorized main path and its tail).
- **M10**: kernel boundaries as device-wide synchronization (cited in the
  fusion argument).

## Forward-reference debts PAID here

- **M3 owed → paid**: "grid-stride loops revisited at the bandwidth ceiling."
  Full 5×6 sweep plus the 1:1 mapping, with the honest finding that on a bare
  copy the 1:1 mapping is the *fastest* configuration (395.2 GB/s), reversing
  M3's SAXPY result and explaining why (M3's kernel had per-element bounds and
  index work with no loop to amortise it).
- **M5 owed → paid**: "vectorized `float4` loads and tail handling at scale."
  `float2`/`float4` measured at two occupancies with the SASS instruction
  census; tails handled at `N = 40,000,001` with a non-idempotent operation so
  double-processing is detectable.
- **M5's write-allocate claim** was asserted there and is **measured** here
  (3.95×), including the negative half (full-sector writes cost 1N).

## Forward references made

- **Module 12 (reduction)** — named as the case where the cross-thread
  dependence survives fusion and must be paid for, unlike an elementwise chain.
- **Module 15 (transpose)** — named as where shared memory returns for a
  *different* reason than reuse (neither side coalescable simultaneously).
- **Module 18 (advanced GEMM)** — coarsening as a *reuse* mechanism, explicitly
  distinguished from coarsening as an MLP mechanism.
- **Module 19 (occupancy)** — the register/warp arithmetic; "how many warps can
  I actually have"; why `gridDim < nSM` matters more for compute-bound work.
- **Module 20 (latency hiding)** — ILP vs occupancy, with Exercise 1's "try
  this" pointing at measuring the exchange rate directly.
- **Module 21 (roofline)** — stated as the generalisation of this module's
  floor; the FFMA row of the crossover table is named as the memory-bound
  plateau drawn from measurement.
- **Module 23 (Nsight Compute)** — named as where `__stcs`'s actual effect (L2
  hit rate for a *different* array) would be visible. `ncu` is unavailable on
  this machine; no counter output is quoted anywhere in this module.
- **Module 28 (CUDA graphs)** — named in Exercise 2's small-N result as the
  other way to attack launch overhead.
- **Parts XIV / XV (Modules 41–42, LLM inference kernels)** — named three
  times and deliberately: the fusion argument for residual-add / norm /
  activation / scale chains is stated as the same arithmetic as Exercise 2, and
  Exercise 3's SiLU gate is the SwiGLU activation those modules will meet again.
  The 8-bit-weight error-budget argument in Exercise 3's solution is the
  quantization forward reference.

## Constraints observed

- **No reductions** (M12), **no scan** (M13), **no atomics as a topic** (M10 owns
  them; none appear), **no shared-memory tiling** (M6 owns it — and the lesson
  explains *why* an elementwise kernel cannot use it, which is itself content).
- No sm_90+ features.
- Themes not reused from §6 of the cross-module index: the vehicles here are a
  clamped in-place blend, a 4-stage affine/ReLU chain, and a SiLU gate — none
  previously used. Matrix transpose, GEMM, sorting, sparse formats, large-radius
  convolution and n-body remain reserved.
- Timing per spec §12 throughout: rotated sweep order, per-configuration
  iteration counts sized to ~10 ms segments, min of 4 sweeps (6 for the
  5-configuration fusion-depth sweep so every configuration leads once),
  duration-based ~400 ms warm-up, buffers ≥ 2.6× L2 and usually ≥ 5×,
  validation in a separate untimed pass, ratios reported as the stable quantity,
  and every figure above 100% of a reference explained rather than hidden.
- Nothing under `module11/` reveals an answer: TODO 1 in Exercise 1, TODOs 1/2
  in Exercise 2 and TODOs 1/2/5 in Exercise 3 are checked against FNV-1a hashes
  and recovered by search, not stored as literals.
- No binaries committed.

## Known variance and honest caveats

- The measured streaming ceiling ranged **292–395 GB/s** across this authoring
  session depending on thermal state and memory P-state (6001 / 8001 / 9001 MHz
  observed, throttle reasons `0x1` and `0x4` both seen). Every ratio quoted in
  the lesson and the solution notes reproduced to within a few percent; no
  absolute ms figure should be treated as reproducible.
- `example02.cu` Part C's `C = 2` hoisted entry is **reproducibly** below its
  `C = 1` and `C = 4` neighbours and the MLP story does not explain it (the SASS
  shows `k_mlp<2>` issuing *more* loads than `k_mlp<1>`). Reported, not smoothed.
- `ncu` remains unavailable (`ERR_NVGPUCTRPERM`). No hardware counters are
  quoted; every effect in this module is constructed and measured directly by
  the harnesses.
