# Module 16 manifest — Naive GEMM

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module16/lesson.md`, `module16/example0{1,2}.cu`,
`module16/exercise0{1,2,3}.cu`.
Solutions: `solutions/module16/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module16/check_your_understanding.md`,
`solutions/module16/MANIFEST.md`.

All `.cu` verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2, V13.2.51, RTX 3500
Ada), warning-clean. Only `example01.cu` links cuBLAS
(`nvcc -arch=sm_89 -O3 -lcublas -o example01.exe example01.cu`); every other
file builds with the plain course line. Both examples print `OVERALL: PASS`; all
three solutions print `OVERALL: PASS` (15/15, 6/6, 1/1). All three shipped
exercises compile with TODOs blank and exit gracefully
(`Set TODO 4 first.`, `Set TODO 3 (PREDICTIONS) first.`,
`Set TODO 1 and TODO 2 first.`). No binaries committed.

**Part V opens here.** Module 16 owns the GEMM problem statement, the naive
kernel, the traffic and roofline analysis, the validation methodology, and the
proof that the decomposition — not the code — is what has to change. It
deliberately ships **no shared memory and no register blocking**.

---

## Concepts taught

- **GEMM as a problem statement**: `C = alpha*op(A)*op(B) + beta*C`, row-major,
  M×K by K×N into M×N; leading dimension vs width (`lda`/`ldb`/`ldc`);
  non-square, non-power-of-two dimensions as a correctness requirement, not a
  stylistic choice. **PORTABLE CUDA CONCEPT.**
- **The `beta == 0` contract**: the BLAS specification says C is not read when
  beta is exactly zero; `alpha*acc + beta*C` evaluated literally propagates NaN
  from uninitialised memory because `0.0f * NaN = NaN`. The branch is
  warp-uniform (M8) and costs one predicate.
- **The one-thread-one-output-element mapping**, and the fact that it is a
  *decomposition* choice with a hard consequence, not an implementation detail.
- **Compulsory traffic for GEMM** = `4(MK + KN + MN)` with beta = 0 (M11's
  definition applied); **requested traffic** = `8MNK`; the ratio `2n/3` for a
  square problem, measured at **724×** on the module's shape.
- **Two arithmetic intensities, and the gap between them as the module's
  thesis**: compulsory `n/6` FLOP/byte (grows without bound), requested
  **0.25 FLOP/byte for every n**. GEMM is the canonical compute-bound kernel
  *iff* the reuse is arranged.
- **Machine balance** = compute ceiling ÷ memory ceiling, **measured** at
  43.3–44.4 FLOP/byte on this GPU; how to place a kernel on a roofline under
  two different traffic models and read both numbers.
- **Recovering the FP32 ceiling without `cudaDevAttrClockRate`** (spec §12
  rule 6): eight independent FFMA chains, one wave, min-of-N → 17 787–18 256
  GFLOP/s, i.e. **1.15× the "peak" the API's 1.545 GHz implies**. Implied clock
  1.78 GHz falls out of the measurement. M1's `clock64()` recovery is shown to
  work for a single resident warp (2.12 GHz) and **not** under full load, with
  the reason (block 0's cycle count is no longer the kernel's duration).
- **Bounding the cache hit rate with a stopwatch**: `on-chip >= 1 - BW·t/R`,
  and the insistence that the **pin peak (432 GB/s)**, not the measured
  streaming figure, is what a *bound* may use. Measured: **≥ 91.6–93.4 %** of
  requested bytes serviced on chip.
- **Per-warp sector analysis for GEMM** (M5's procedure with two arrays at
  once): `x → col` block (32,8) = 1 A sector (broadcast) + 4 B sectors = 5;
  `x → row` = 32 + 1 = 33. Full 6-shape × 2-mapping table, with the model
  ordering 11 of 12 measured configurations correctly.
- **The mapping is a property of the warp linearization, not of the variable
  names**: `x → row` at `blockDim.x = 2` is *as fast as* the best `x → col`
  configuration, because `threadIdx.y` then carries the fast axis. Closes
  M3 ex1 and M5.
- **Requested sectors are an upper bound on cost, not a prediction**: model
  predicts 8.25× spread, measurement gives 3.3–4.0× at an L2-resident size and
  8.2× at 1024³. M6's finding restated at K = 724.
- **The real bottleneck is instruction mix, not bytes**: two `LDG` plus an
  `IMAD.WIDE` per `FFMA`. Measured with a loads-per-FMA probe: **8× the
  arithmetic for 10–22 % more time**, throughput proportional to FMAs-per-load.
  **6.4–6.5 FMAs per global load** needed for 80 % of the compute ceiling
  versus the **0.5** a one-element-per-thread GEMM can ever supply.
- **A cache reduces the cost of a memory instruction; it does not reduce the
  number of them.** The sentence that makes M17 and M18 forced moves.
- **Validation methodology for floating-point GEMM** (the module's reusable
  deliverable — see the dedicated section below).
- **cuBLAS's column-major convention**: a row-major M×N matrix with `ldc = N`
  *is* the column-major N×M matrix Cᵀ with `ldc = N`, so row-major `C = A·B` is
  column-major `Cᵀ = Bᵀ·Aᵀ` and the correct call swaps the operands with **no**
  transpose flags. Measured at 8150–8705 GFLOP/s, 45–48 % of the FP32 ceiling.
- **Benchmarking finding added by this module**: the two ceilings need
  *different* warm-ups. A 1500 ms streaming warm-up ramps the memory P-state
  (spec §12 rule 4) but a compute ceiling measured straight after it reads
  1.49 GHz-equivalent; a 500 ms FFMA warm-up after it recovers 1.78 GHz without
  the memory clock falling back. Also observed: **`nvidia-smi` shows the SM
  clock dropping to 285 MHz during a pure streaming kernel**, which is why the
  streaming ceiling measured 294–411 GB/s from the identical binary in one
  session.

## The validation methodology (Modules 17 and 18 must reuse this)

`gemmValidate()` in `example01.cu` is the reference implementation. Three
checks, **in this order**:

1. **Finiteness / writtenness over all M·N elements.** Prefill C with
   `+infinity` before every launch. Two properties at once: no correct kernel
   produces `+inf`, so a surviving sentinel means "never written"; and
   `0.0f * inf = NaN`, so a kernel that reads C when `beta == 0` is caught.
   **This must run first** — a NaN compares false against everything, so a
   max-error loop run first silently *passes* an all-NaN array. Verified: the
   beta defect scores `worst = 0` on both numerical checks.
2. **Freivalds probe, full coverage, O(MN + KN + MK).** Draw a **non-negative**
   pseudo-random `v` (non-negative so a systematic per-element error accumulates
   coherently along the row instead of cancelling), and check in double
   `C·v == alpha·A·(B·v) + beta·C0·v` with the tolerance propagated through the
   same contraction: `|alpha|·(gamma_K + 4u)·(|A|(|B||v|))_i + 4u·|beta|·(|C0||v|)_i`.
   Costs the same order as the problem's compulsory traffic.
3. **Sampled exact double reference.** Strided sample of (i, j), exact
   `S_ij = Σ_k |a_ik||b_kj|`, threshold `err / (gamma_K · S_ij) <= 1`.

The tolerance law: `gamma_K = K·u/(1 − K·u)`, `u = 2^-24`, scaled by
**`S = Σ_k |a_ik||b_kj|`, never by `|C|`**, with a safety factor of ~4.
Measured headroom on a correct kernel: 0.009 (positive data), 0.0008
(zero-mean). Measured rejection margins: off-by-one in K 16–29×, B read
column-major 236–1465×, A read column-major 1004–1236×.

Data conditioning: test on **two** datasets — strictly positive in [0.5, 1.5)
(so `|C| ≈ S`, the test has teeth) and zero-mean in [−1, 1) (so `|C| ≪ S` and a
tolerance scaled by `|C|` is forced to fail). Measured: the *same correct
kernel* has a relative-to-|C| error of 1.6e-6 on the first and **9.7e-4** on the
second — 97× past the house `1e-5 * max(1,|ref|)` rule, which therefore rejects
a correct GEMM.

Report the **headroom**, not just PASS.

## CUDA API / intrinsics / syntax introduced

- `<cublas_v2.h>`: `cublasHandle_t`, `cublasCreate`/`cublasDestroy`,
  `cublasSgemm`, `CUBLAS_OP_N` / `CUBLAS_OP_T`, `cublasStatus_t`,
  `CUBLAS_STATUS_SUCCESS`; build flag `-lcublas`. (Module 36 owns the library;
  used here only as a measured reference.)
- `nanf("")`, `std::numeric_limits<float>::infinity()`, `isfinite()` as
  validation instruments; `ldexp(1.0, -24)` for the fp32 unit roundoff.
- `llround` for encoding a double answer into a hashable integer.
- Reused, not introduced: `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
  `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount / cudaDevAttrClockRate)`,
  `cudaEvent_t` timing, `fmaf`, `#pragma unroll`, `float4`, FNV-1a answer
  hashing (M11/M12 convention), `setvbuf(stdout, NULL, _IONBF, 0)` (M13
  convention, now spec §5).
- SASS read and quoted: `LDG.E`, `IMAD.WIDE`, `FFMA` — the instruction census of
  the naive inner loop, shown to be **byte-identical** for both mappings.

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | A: the traffic ledger computed at runtime (724×, 181 vs 0.25 FLOP/byte). B: the naive kernel and the `beta == 0` contract, with C poisoned by `+inf`. C: `gemmValidate()` — the three-check methodology in full, with the tolerance derivation in comments. D: five kernels (one correct, four defective) through the validator, showing which check caught which and that the NaN cases score 0 on both numerical checks. E: the cuBLAS column-major derivation, the swapped-argument call, its validation, and a timed comparison. |
| `example02.cu` | A: both ceilings measured, with the two-stage warm-up and the `cudaDevAttrClockRate` contradiction spelled out. B: per-warp sector counting by literal lane enumeration, 2 mappings × 4 shapes. C: 6 configurations in one rotated sweep (`SWEEPS = NCFG`), model vs measurement. D: the on-chip service bound. E: the loads-per-FMA probe (R = 1,2,4,8) — the module's decisive measurement. F: the roofline table under both traffic models. |

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Fill-in + design (§6 types 1, 5) | 5 | Write the naive GEMM at 1027×2053×769 and the tolerance law that judges it; harness runs your kernel and four defects on two datasets | **TODO 5 is the exercise.** The obvious tolerance — relative to `\|ref\|`, house style — passes the shape probe's proportionality test only if written against `S`, and a reader who scales by `\|ref\|` passes dataset A and fails dataset B, where the *same correct kernel* measures 9.7e-4 relative. The house `1e-5*max(1,\|ref\|)` rule is itself 4.6× too tight at K = 769 and rejects a correct kernel. TODO 3's trap: with C zeroed, the unconditional `alpha*acc + beta*C` is indistinguishable from the correct form — the harness poisons C with `+inf` so `0.0f*inf = NaN` catches it in all 2 108 431 elements. TODO 4 is M3 ex1's trap: the axis decision appears twice and the inconsistent version leaves 1 053 702 elements unwritten without crashing. Also: the sampled numerical check *passes* on the NaN cases (worst = 0), which is why the finiteness check must be ordered first. |
| `exercise02.cu` | Predict + fill-in + design (§6 types 1, 2, 6) | 4 | Count per-warp sectors for 2 mappings × 6 block shapes, predict the spread and the best shape, implement the second kernel, and let your own model choose the block shape before anything is timed | The headline result inverts the expected lesson: the "bad" mapping is **fully repaired by `blockDim.x = 2`** (1288 vs 1348 GFLOP/s), because the warp linearization, not the variable naming, decides which index moves with the fast axis. Counting *lanes* instead of *distinct sectors* gives 32 where the answer is 1 and fails the hash. The grid line in `gridFor()` is M3 ex1's trap again. The scored spread prediction is bucket 3 while the sector model says bucket 4 — the model over-predicts by 2× and the reader has to explain it (L2 absorbs the re-requests; and the kernel is instruction-bound, not sector-bound). One configuration, `x → row (1,256)`, has the minimal-but-one sector count and is 1.6× off the best, because sector counting is a per-warp model with no term for block shape. |
| `exercise03.cu` | Analysis / performance reasoning + design (§6 types 5, 6) | 5 | Write five analysis functions; the harness hashes them against references on fixed synthetic inputs, then applies them to real measurements and prints the diagnosis | TODO 1's trap is counting C twice (right for `beta != 0`, wrong here — a 45 % error in the compulsory figure). TODO 4's marked trap is the degenerate case: when DRAM *could* have supplied everything, no bound follows and the function must return 0, not `1 - 4e11/1e8 = -3999`; the harness probes exactly that. Also the choice of ceiling: a *bound* needs the 432 GB/s pin peak, not the measured streaming figure (294–411 GB/s in one session). TODO 5 is the design TODO: the relationship is not stated, and fitting a line **with an intercept** rather than through the origin gives a nonsense answer at small `fpl`. The payload: 6.47 FMAs per global load needed, 0.50 available, factor 13, and no launch parameter changes it. |

Scoring: Ex1 15 points (4 tolerance-shape probes, 2 accepts, 8 rejects, 1
alpha/beta path); Ex2 6 points (model hash, kernel-B correctness at 6 shapes,
2 predictions, 2 model-vs-measurement agreements scored in a 5 % band); Ex3
1 point, awarded only if all eight hashed analysis answers are correct.
`OVERALL: PASS` requires full marks in all three.

## Measured results recorded (RTX 3500 Ada, CUDA 13.2)

| Quantity | Measured |
|---|---|
| FP32 FFMA ceiling (8 chains, 1 wave, min-of-N) | **17 787 – 18 256 GFLOP/s** |
| Implied SM clock from that ceiling | **1.78 GHz** (API reports 1.545; single-warp `clock64()` reports 2.12) |
| "Peak" from `cudaDevAttrClockRate` | 15 821 GFLOP/s — **exceeded by the measurement by 1.15×** |
| DRAM read streaming ceiling | **410.8 GB/s (95 % of 432)** best; **294–411 GB/s** observed range |
| Machine balance | **43.3 – 44.4 FLOP/byte** |
| Compulsory traffic, 1027×2053×769 | 17.908 MB |
| Requested traffic, naive kernel | 12.971 GB = **724×** compulsory |
| Compulsory arithmetic intensity | 181.08 FLOP/byte (= n/6 for square n) |
| Requested arithmetic intensity | **0.25 FLOP/byte, for every problem size** |
| Naive GEMM, best shape (16,16), x → col | **2.41–2.55 ms, 1275–1348 GFLOP/s** |
| … as a fraction of the FP32 ceiling | **7.0 – 7.4 %** |
| … as a multiple of its own requested-traffic roofline | **12.5 – 14.1×** (i.e. the caches) |
| Minimum on-chip service fraction (bound vs pin peak) | **≥ 91.6 – 93.4 %** |
| `cublasSgemm`, swapped arguments | **0.37–0.40 ms, 8150–8705 GFLOP/s** |
| Naive / cuBLAS | **12–13 % of cuBLAS**, 7.9–8.2× the time |
| cuBLAS as a fraction of the FP32 ceiling | 45–48 % |
| x → row (32,8) / x → col (32,8), 1027×2053×769 | **3.3 – 3.4×** (sector model says 6.6×) |
| Same pair at M = N = K = 1024 | **8.2×** (13.12 ms vs 1.60 ms) |
| Spread over 12 configurations (2 mappings × 6 shapes) | **3.8 – 4.0×** (sector model says 8.25×) |
| x → row (16,16) / x → row (32,8) | measured 0.609–0.614, model 17/33 = 0.515 |
| Best x → col shape | `blockDim.x = 16` (4 sectors) — model and measurement agree |
| Best x → row shape | `blockDim.x = 2` (4 sectors) — **the "bad" mapping repaired** |
| Loads-per-FMA probe, R = 1 / 2 / 4 / 8 | 1275–1283 / 2410–2683 / 4589–5290 / 8622–9307 GFLOP/s |
| … time cost of 8× the arithmetic | **+10 % to +22 %** |
| FMAs per global load for 80 % of the compute ceiling | **6.4 – 6.5** |
| FMAs per global load a 1-element-per-thread GEMM supplies | **0.50** (factor 13 short) |
| Correct kernel, worst sampled scaled error | 0.035 (positive data) / 0.0037 (zero-mean) |
| Correct kernel, worst relative-to-\|C\| error | 1.6e-6 (positive) / **9.7e-4 (zero-mean)** |
| `gamma_K` at K = 769 | 4.584e-05 (**4.6× looser than the house 1e-5 rule**) |
| Defect rejection margins (× tolerance) | k off-by-one 16–29; B col-major 236–1465; A col-major 1004–1236 |
| `beta` read when `beta == 0` | 2 108 431 / 2 108 431 elements non-finite |
| Wrong-dimension guard | 1 053 702 elements never written |
| `__restrict__` on the naive GEMM | **Null result**, 0.991–0.996× |
| SASS of the two mappings | **byte-identical** (59 LDG.E, 36 IMAD.WIDE, 30 FFMA in the unrolled body) |
| Registers used by the naive kernel | 40, no spills, no shared memory |

## Assumed from earlier modules

- **M1**: 40 SMs, 128 FP32 lanes/SM, 1536 threads/SM, warp = 32, waves,
  Little's Law, `clock64()` clock recovery and its one-wave precondition.
- **M2**: `nvcc -arch=sm_89`, launch syntax, `CHECK` macro idiom, both
  `cudaGetLastError()` and `cudaDeviceSynchronize()`, `cudaEvent_t` timing.
- **M3**: the linearization rule `tid = threadIdx.x + blockDim.x*threadIdx.y`
  and warp formation from it; 2-D indexing; bounds guards; ceil-divide grid
  sizing; **the stride-matching rule and Exercise 1's axis trap**, which this
  module reproduces on a problem where it costs more.
- **M4**: the storage map and latencies (L1 40.5 / L2 241.3 / DRAM 575 cycles);
  48 MB L2 as the benchmarking hazard; the broadcast-vs-serialize mechanism;
  `cuobjdump -sass`; `const __restrict__`.
- **M5**: 32 B sectors, 128 B lines, **the sector-counting procedure** (used
  literally, twice), address set not sequence, inactive lanes supply no address,
  row pitch as a layout decision, effective vs DRAM bandwidth.
- **M6**: the K/H reuse argument and, decisively, **"the caches got there
  first"** — a kernel near a hardware ceiling cannot be sped up by removing
  traffic it is not generating, and tiling a 5-point stencil is a 0.85× loss.
  Also M6's naming of **register blocking as the missing ingredient**, which
  this module quantifies (6.5 vs 0.5 FMAs per load) and hands to M18.
- **M8**: warp-uniform branches are predicated and cheap (the `beta == 0`
  branch); divergence is warp-local.
- **M11**: **compulsory traffic**, the floor, `x floor` as a reporting unit, and
  the discipline of computing the model before writing the kernel. Also MLP and
  the observation that coarsening a saturated kernel is a regression, cited as
  the reason M18's coarsening must be *reuse* coarsening.
- **M12**: the 1500 ms warm-up finding and the **410.5–410.7 GB/s** streaming
  ceiling; float summation error growth (`O(n)·eps`) and the `2^26 ones`
  saturation, which is the same hazard as a long K; FNV answer hashing.
- **M13**: `setvbuf(stdout, NULL, _IONBF, 0)` as the house convention.
- Spec §12 throughout: 1500 ms warm-up, back-to-back timing with **rotated**
  start order and `SWEEPS >= NCFG`, iteration counts auto-scaled to ~10 ms
  segments, validation in a separate untimed pass, min-of-N, ratios as the
  stable quantity, `cudaDevAttrClockRate` rejected.

## Forward-reference debts PAID here

- **M6 → M16/M17**: "GEMM is where tiling wins; K/H runs into the hundreds."
  Paid in the half M16 owns: the ratio is computed (724×), the reuse is shown to
  be *already captured by the caches* (≥91.6 % on chip), and the reason tiling
  alone is not the answer is made quantitative (instruction mix, not bytes).
  M17 delivers the tiling; M16 makes it inevitable and explicitly does not
  pre-empt it.
- **M6 → M18**: "register blocking is the missing ingredient." Paid as a
  *number* rather than a technique: 6.4–6.5 FMAs per global load are needed and
  0.50 is what the decomposition supplies. No register blocking is implemented.
- **M3 → M16** (implicit): M3 measured the axis trap at 3.1× (ex1) and 3.5×
  (ex2) and promised the coalescing analysis to M5. Exercise 2 closes the loop on a
  problem where the same decision is worth 3.4–8.2× and where the sector model
  from M5 predicts the ordering of all twelve configurations.
- **M11 → M16** (implicit): M11 named Module 18 as the owner of coarsening *as
  a reuse mechanism, distinguished from coarsening as an MLP mechanism*. M16
  supplies the quantity that makes the distinction concrete.

## Forward references made

- **Module 17 (tiled GEMM)** — named repeatedly as the owner of shared-memory
  tiling, cooperative tile loading, and double buffering. This module ships
  **zero** shared memory. The hand-off is explicit: tiling changes the *cost* of
  an operand read, not the *count* of memory instructions per FMA.
- **Module 18 (advanced GEMM)** — register blocking / thread coarsening /
  vectorized loads / double buffering named as the owner of the FMAs-per-load
  fix. The loads-per-FMA probe deliberately stops one step short: it reuses the
  *same* operand pair R times and computes nothing meaningful, so it measures
  the bound without demonstrating the technique.
- **Module 19 (occupancy)** — named in CYU Q3 as the owner of the register
  budget that bounds how large a per-thread output patch can be.
- **Module 21 (roofline)** — named three times as the owner of the full
  treatment; this module builds a two-point roofline from measured ceilings and
  says so.
- **Modules 22–23 (Nsight)** — `ncu` remains unavailable (`ERR_NVGPUCTRPERM`).
  No counter output is quoted anywhere; the cache-service figure is a *bound*
  derived from elapsed time and the bus width, and the file says so in print.
- **Modules 33–34 (Tensor Cores)** and **Module 43 (CUTLASS)** — named in the
  lesson as what lies beyond M18; not used.
- **Module 36 (cuBLAS / CUB / Thrust)** — named as the owner of the library;
  `cublasSgemm` appears here only as a measured reference point, with the
  column-major argument swap derived rather than asserted.

## Constraints observed

- **No shared memory anywhere in this module** — not in the examples, not in the
  exercises, not in the solutions. Stated in the lesson and in `example02.cu`'s
  header.
- **No register tiling, no thread coarsening as an optimization, no vectorized
  loads as an optimization, no Tensor Cores, no `cp.async`.** The one kernel
  that holds multiple accumulators (`loadsPerFmaProbe<R>` / `fmaProbe<R>`) is
  labelled, in the source and in the lesson, as a probe that computes nothing
  useful and exists solely to measure the instruction-mix bound.
- Non-square, non-power-of-two dimensions throughout: **M = 1027, N = 2053,
  K = 769**, none a multiple of any block dimension used, and M ≠ N so a
  transposed result is not shape-compatible.
- No sm_90+ features.
- Everything builds warning-clean with `nvcc -arch=sm_89 -O3` (plus `-lcublas`
  for `example01.cu`).
- Nothing under `module16/` reveals an answer: Exercise 2's eight reference
  sector counts and Exercise 3's eight reference analysis values are checked by
  FNV-1a hash, Exercise 1's tolerance is checked by probing the reader's
  function for *shape* rather than by comparison against a stored constant, and
  every prediction is scored against a measurement.
- No binaries committed.

## Known issues / honesty notes

- **`ncu` unavailable** (`ERR_NVGPUCTRPERM`). Documented as theory per the
  standing rule; the on-chip service figure is a bound from elapsed time and the
  432 GB/s bus, not a counter reading, and every file that prints it says so.
- **The streaming ceiling is not reliably reproducible on this part.** The same
  binary measured 294–411 GB/s across one session. `nvidia-smi` sampled during
  the streaming kernel shows memory clock 8801 MHz and **SM clock 285 MHz** —
  the power manager trades SM clock away on a kernel that issues no arithmetic,
  and no warm-up length fixes it. Consequence adopted throughout: use the
  **measured** figure for a ceiling and the **432 GB/s pin peak** for a bound.
- **The two ceilings need different warm-ups.** Warming with the compute kernel
  and then measuring the stream reproducibly reports 300–380 GB/s; warming with
  the stream and then measuring compute reports a 1.49 GHz-equivalent ceiling.
  `example02.cu` and `exercise03.cu` do 1500 ms streaming followed by 500 ms
  compute, which recovers both. This is a new finding relative to spec §12
  rule 4 and is recorded for the index.
- **The sector model over-predicts the mapping penalty by ~2×** at this problem
  size (8.25× predicted, 3.8–4.0× measured) and is much closer at 1024³ (8.2×).
  Reported as a result and made a scored question rather than smoothed over.
- **`x → row (1,256)` is the one configuration the sector model mis-orders**
  (5 sectors, 1.6× off the best). Printed, not hidden; the explanation offered
  is block-level footprint, which the per-warp model has no term for.
- **`__restrict__` on the naive GEMM is a null result** (0.991–0.996×),
  measured and recorded rather than assumed helpful.
- **Absolute timings move up to 2.5× with thermal state.** Every claim in the
  lesson and the solution notes is a ratio or is quoted as a range; the worst
  observed run had the FP32 ceiling at 7450 GFLOP/s and the naive kernel at
  465 GFLOP/s, with the *ratios* (7 % of the roof, 13× short on FMAs per load)
  unchanged.

## Cross-module observations for the index

1. **§6b should gain a compute-side row.** The course has a measured DRAM
   ceiling (410.5–410.7 GB/s) but no measured FP32 ceiling. This module
   establishes **17 787–18 256 GFLOP/s** and a **machine balance of 43–44
   FLOP/byte**, and shows that the `cudaDevAttrClockRate`-derived figure
   (15 821 GFLOP/s) is **below** the measured one, so any "% of FP32 peak" built
   on it can and does exceed 100 %.
2. **Spec §12 rule 4 needs a corollary**: the warm-up must match the resource
   being measured. A 1500 ms *streaming* warm-up ramps the memory P-state but
   leaves the SM clock low; a compute warm-up does the reverse. Measuring both
   ceilings requires warming both, in that order.
3. **`clock64()` clock recovery has a documented failure mode.** It reads
   2.12 GHz for a single resident warp and produces 0.44–0.66 GHz for a
   full-occupancy one-wave grid, because block 0's cycle delta is no longer the
   kernel's duration. M1's caveat ("assumes the grid is one wave") is necessary
   but not sufficient. Recovering the clock from a *measured saturating FFMA
   throughput* is the robust alternative and is what this module uses.
4. **§6 "themes already used" should gain**: GEMM in its naive form, the
   loads-per-FMA probe, the Freivalds validation probe. Tiled GEMM and register
   blocking remain reserved for M17/M18 as intended.
