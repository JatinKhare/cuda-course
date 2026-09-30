# Module 07 manifest

## Concepts taught

- **Banked scratchpad memory** — independent one-ported SRAM banks plus a
  crossbar, as the affordable alternative to a 32-ported SRAM.
  **PORTABLE CUDA CONCEPT.**
- **32 banks × 4 B, word-interleaved**, `bank(addr) = (addr / 4) % 32`, period
  128 B. **ARCHITECTURE-SPECIFIC** (sm_89; CC 1.x had 16 banks and half-warp
  resolution; CC 3.x had an optional 8 B bank width via
  `cudaDeviceSetSharedMemConfig`, since removed).
- **Why word-interleaved and not block-mapped** — makes `s[threadIdx.x]` free
  and pushes the pathology onto multiples of 32.
- **The conflict rule, precisely**: group a warp's addresses by bank, then by
  word; same bank + **same word** = **broadcast**, one cycle; same bank +
  **different word** = **replay**, one cycle each.
- **Conflict degree** `D = max over banks of (distinct words that bank supplies)`.
- **Count distinct words, never lanes** — the single line that separates the
  correct model from the naive one.
- **The replay mechanism is M4's**, the same one behind constant-memory
  serialization; and M4's constant broadcast is this module's shared broadcast
  in a different memory. (Third instance, atomic contention, is M10's.)
- **The mechanical counting procedure**, mirroring M5's sector count: lane →
  element → byte → `(bank, word)` → bucket distinct words → take the max.
- **`D = gcd(k, 32)` for `s[k*tid]`** — odd strides always free, powers of two
  cost exactly that power.
- Worked cases: `s[tid]`=1, `s[2*tid]`=2, `s[3*tid]`=1, `s[4*tid]`=4,
  `s[8*tid]`=8, `s[16*tid]`=16, `s[32*tid]`=32, `s[tid/2]`=1 (broadcast pairs),
  `s[0]`=1, `s[31-tid]`=1 (permutation), `s[33*tid]`=1.
- **The 2D row-vs-column case**: `tile[ty][tx]` on a 32-wide tile is `D=1`;
  `tile[tx][ty]` is `D=32`, because `bank(r,c) = c` when the pitch is 32.
- **Padding**: `[32][33]` gives `bank = (r+c) % 32`, a rotation; both the row
  write and the column read become conflict-free. General rule:
  `gcd(pitch, 32) == 1`, i.e. **the pitch must be odd** — the opposite of every
  alignment rule from M5.
- **Padding's occupancy cost**, with the arithmetic: per-block shared request
  + 1024 B driver reserve, rounded up to 128 B granularity, divided into the
  102400 B per SM. Measured: `[192][32]`→4 blocks/SM, `[192][33]`→3.
- **XOR swizzle** `loc = r*32 + (c ^ (r & 31))` — zero extra memory, both phases
  conflict-free, correctness resting on XOR-by-a-constant being a bijection on
  `[0,32)`.
- **Swizzling is free of memory cost, not of instruction cost** — measured 10
  extra `LOP3` and ~19 % slower than padding on a kernel whose tile already
  fits. Padding wins when you have the memory; swizzling wins when capacity
  limits tile size (which is why CUTLASS swizzles).
- **Phase splitting for wide accesses**: the bank array delivers 128 B/cycle, so
  `phases = elemBytes / 4` (4 B → 1 phase of 32 lanes, 8 B → 2 of 16, 16 B → 4
  of 8), with conflicts resolved **independently within a phase**.
- **The folklore correction for `double`**: `sd[2*tid]` is not the 4-way
  conflict naive whole-warp arithmetic predicts; measured 1.65–1.90×. And
  `sd[16*tid]` vs `sd[32*tid]` — naive degrees 16 and 32 — measure **1.00×**
  apart, which is the phase split verified.
- **Padding a double tile takes one `double`, not one `float`** (`sd[17*tid]`
  is conflict-free).
- **A 2-way conflict on floats is free on sm_89.** Measured cost is
  `∝ max(2, D)`, not `∝ D`: a conflict-free 32-lane 4 B access already occupies
  the shared pipeline for two cycles. **ARCHITECTURE-SPECIFIC.**
- **`D/2` bounds the shared-memory term, never the whole kernel** — measured
  whole-kernel speedups of 10.5–15.4× from removing a genuine `D=32`.
- **Conflicts can be data-dependent**: the same 32 index values are
  simultaneously `D=32` at one scale factor and `D=1` at another.
- **`s[f(tid) & (N-1)]` with `N ≤ 32` is unconditionally conflict-free** for any
  `f`, because a 32-element window holds exactly one word per bank.
- **Per-thread private shared scratch laid out `[thread][slot]` with a
  32-multiple slot count is a 32-way conflict** — privacy of data is irrelevant
  to banking. The fix, `[slot][thread]`, is M5's AoS→SoA one level down.
- **Nsight Compute counters**: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_
  op_ld.sum` / `..._op_st.sum`, plus
  `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`.
- **Benchmarking**: configuration order must **rotate across sweeps**, or the
  first-measured configuration absorbs the post-sync clock dip (observed: a
  broadcast measuring 0.75× of conflict-free, from an inflated baseline).
- **Ada runs FP64 at 1/64 rate** — an honest `acc += sd[i]` measures the FP64
  pipe and makes shared-memory banking invisible; reinterpret and XOR instead.
- **Compiler vectorization can invalidate your analysis**: 32 adjacent unrolled
  shared reads become 8 `LDS.128`, turning a 32-way scalar conflict into an
  8-way-per-phase one and collapsing a 16× penalty to 3.15×.

## CUDA API / intrinsics / syntax introduced

- `__double_as_longlong()` (bit reinterpretation to dodge the FP64 pipe)
- `cudaDeviceGetAttribute` with `cudaDevAttrMaxSharedMemoryPerMultiprocessor`,
  `cudaDevAttrMaxThreadsPerMultiProcessor`,
  `cudaDevAttrMaxBlocksPerMultiprocessor`,
  **`cudaDevAttrReservedSharedMemoryPerBlock`** (new here)
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- `cudaDeviceSetSharedMemConfig` / `cudaSharedMemBankSizeEightByte` — named only
  as deprecated history, never used
- `#pragma unroll 1` as a deliberate *anti*-optimization to keep shared accesses
  scalar
- SASS mnemonics named: `LDS`, `LDS.128`, `STS`, `LOP3`
- `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
- Toolchain reuse: `nvcc -Xptxas -v`, `nvcc -cubin` + `cuobjdump -sass`

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Predict-the-behavior + address calculation | 4 | Fill a paper bank table for 8 indexing expressions, implement `bank_of`, the degree counter and the phase split, then score predicted vs measured ratios | Two traps. (a) `s[2*tid]` is a real 2-way conflict and measures **1.00×** — the floor for any 32-lane 4 B access is 2 pipeline cycles, so the model must be `max(2,D)/2`, not `D`. (b) `s[tid/2]` reports degree 2 if you bucket *lanes* per bank instead of *distinct words*; the one missing `if (!seen)` turns a free broadcast into a phantom conflict. Bonus: `dd[2*tid]` is 1.90×, not the 4× naive 4-byte arithmetic gives. |
| `exercise02.cu` | Optimization (v1→v2→v3) + design | 4 | A 32-way column conflict in a `[192][32]` tile; produce a padded fix and a zero-memory fix, with hand-computed occupancy checked against the CUDA API | The TODO never says "padding" or "swizzle"; TODO 3 requires the array stay **exactly `ROWS*32` floats**, which rules padding out and forces the XOR. `PAD_PITCH 40` ("multiple of 8, nicely aligned") gives `gcd(40,32)=8`, a 4× speedup that looks like a fix and leaves 3/4 of the penalty — the shared-memory twin of M5 ex3's pitch 68. TODO 4's occupancy disagrees with the API until you include the **1024 B per-block driver reserve** and the 128 B granularity. And the headline result inverts the folklore: padding is **19 % faster** than the swizzle despite 25 % lower occupancy, because 10 extra `LOP3`s cost more than the fourth resident block was worth. |
| `exercise03.cu` | Predict-the-behavior + design | 4 | Six configurations across two kernels; predict every conflict degree before running | The kernel that *looks* catastrophic (data-dependent shared index) is provably conflict-free for **every possible input**, because `& 31` makes bank collision and word equality the same event. The kernel nobody flags (`scratch[tid][k]`, per-thread private scratch, stride 32 floats) is 32-way. A stride-32 access with a warp-uniform index is a **broadcast**, the cheapest row in the table. And TODO 4's array is simultaneously the worst possible input at `SCALE=32` and perfectly benign at `SCALE=1` — conflicts are a property of the (data, expression) pair. |

Solutions: `solutions/module07/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module07/check_your_understanding.md`.

Worked examples: `example01.cu` (bank maps for 10 expressions + a controlled
conflict-degree sweep establishing `cost ∝ max(2,D)`), `example02.cu` (phase
splitting for 4/8/16 B, including the `dd[(tid%16)*16]` vs `dd[(tid%32)*16]`
pair that measures 1.00× apart against naive degrees of 16 and 32).

## Assumed from earlier modules

- **M1**: 40 SMs, warp = 32 lanes, the SM's 128 KB unified L1+SMEM block with
  ≤100 KB addressable as shared, warp schedulers and issue slots, latency
  hiding by resident warps, ILP versus dependent chains.
- **M2**: `nvcc -arch=sm_89`, launch syntax, the `CHECK` macro idiom, checking
  both `cudaGetLastError()` and `cudaDeviceSynchronize()`.
- **M3**: the linearization rule (`x` fastest → a warp is 32 consecutive
  `threadIdx.x` at fixed `y`), which is what makes "the warp's 32 lanes" a
  well-defined set of addresses; bounds guards.
- **M4**: the replay mechanism and constant memory's broadcast-vs-serialize
  behaviour (24.6× measured) — reused as the framing for this entire module;
  shared memory's location and price; `-Xptxas -v`; `cuobjdump -sass`;
  L1/shared carveout; the 575-cycle global latency used in CYU Q4.
- **M5**: the sector-counting *method* (this module's procedure is its direct
  analogue); per-warp per-instruction analysis; address set not sequence
  (justifying `s[31-tid]`); inactive lanes supply no address; row pitch as a
  layout decision and the pitch-68 trap; AoS vs SoA (Exercise 3's kernel B is
  the same transformation); global broadcast costing one sector.
- **M6**: shared memory proper — `__shared__`, static vs dynamic, the third
  launch parameter, cooperative loading, tiling, barrier placement. Module 7
  **uses** all of it and re-teaches none of it; the kernels here are described
  as "the Module 6 pattern" and the commentary is exclusively about *where in
  shared memory* each element lands.
- Spec §12 timing discipline: back-to-back configurations, second-pass
  validation, min-of-N, duration-based warm-up, ratios as the stable quantity.

## Forward references made

- **Module 8** (warps, divergence, active masks): mentioned where the conflict
  rule is stated for a fully-active warp, noting that inactive lanes supply no
  address.
- **Module 9** (`__syncthreads`, memory ordering): every barrier in every file
  carries the "barrier; Module 9 makes this precise" caveat, per the house
  convention from M6. Also named in CYU Q4 (a faster load phase behind a
  barrier still waits for the slowest warp).
- **Module 10** (atomics, contention): named as the **third** instance of the
  replay mechanism, closing the loop M4 opened and M7 continued.
- **Module 15** (transpose): named as the archetypal kernel where a tile is
  written one way and read the other; deliberately *not* used as the vehicle
  here, so M15 keeps it.
- **Module 19** (occupancy): Exercise 2 computes blocks-per-SM by hand and
  checks it against `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; the lesson
  says explicitly that M19 owns occupancy properly.
- **Module 23** (Nsight Compute): the two bank-conflict metric names and the
  `ncu --metrics` command line are given so the reader can check their own
  answer now; the wavefront vocabulary, Memory Workload Analysis, and
  `--kernel-name` / `--launch-count` hygiene are deferred. Also named as the
  tool that would settle the unexplained `float4` phase-model discrepancy.
- **Module 43** (CUTLASS / modern GEMM): named as the home of real swizzle
  layouts, `cp.async` and Tensor-Core pipelines, and as the setting where the
  capacity argument makes swizzling beat padding.

## Known issues / honesty notes

- **`ncu` was not usable on this machine.** Nsight Compute 2026.1.0 is installed
  at `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\ncu.bat` but
  returns `ERR_NVGPUCTRPERM` — GPU performance-counter access requires
  elevation. Every conflict degree claimed in this module was verified by
  **timing ratio against a controlled degree sweep** (`example01.cu` builds an
  exact degree `D` for `D = 1..32 `and confirms `cost ∝ max(2,D)`), not by
  reading the counter. The lesson says so explicitly rather than implying the
  counters were read.
- **The `float4` phase model is an upper bound, not a law.**
  `qq[(tid%8)*8]` and `qq[(tid%32)*8]` should be equal under a rigid
  four-contiguous-phase model and measure **1.53× apart**. The program prints
  this, flags it as unexplained, and points at M23. The 8 B case is confirmed
  exactly (1.00×); the 16 B case is not. Left as a documented surprise per
  spec §12.
- **Absolute timings on this laptop part move by up to 2× with thermal state.**
  Cold-start runs inflate the conflicted/conflict-free ratios (`s[32*tid]`
  measures 14.8× warm and up to 19× cold). All solution `.md` files quote warm
  numbers and state the observed range.
- Files compile warning-clean with the documented line
  `nvcc -arch=sm_89 -O3`. At `/W4` MSVC emits C4127 ("conditional expression is
  constant") for the template-parameter dispatch in `exercise02.cu` and
  `exercise03.cu`; this is expected for `if (TEMPLATE_PARAM == ...)` and is not
  enabled by the course build line.
