# Module 03 manifest

Files: `lesson.md`, `example01.cu`, `example02.cu`, `exercise01.cu`,
`exercise02.cu`, `exercise03.cu`. (This manifest lives in `solutions/module03/`.)
Solutions: `solutions/module03/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module03/check_your_understanding.md`.
All `.cu` files verified with `nvcc -arch=sm_89 -O3` on the RTX 3500 Ada
(CUDA 13.2); all solutions print `OVERALL: PASS`.

## Concepts taught

- The four built-in variables: `threadIdx`, `blockIdx` (`uint3`), `blockDim`,
  `gridDim` (`dim3`), and that they are read-only special registers, not memory.
- **Unused dimensions are 1, not 0** — `dim3`'s defaulting constructor, and why a
  0 would annihilate the linearization formula.
- 1-D / 2-D / 3-D grids and blocks, and the criteria for choosing each (problem
  geometry vs. the cost of hand-computing an integer division).
- **The linearization rule**: `tid = threadIdx.x + blockDim.x * (threadIdx.y +
  blockDim.y * threadIdx.z)`; x varies fastest.
- **Warp formation**: threads with `tid ∈ [32w, 32w+32)` are warp `w`; the
  partition is fixed at block launch and is the sole determinant of warp
  membership.
- Warp *shape* as a function of `blockDim`: (32,8) → warps are rows;
  (8,32) → warps straddle 4 rows; (1,256) → warps straddle 32 rows.
- Lane id = `tid % 32`, verified against the hardware `%laneid` register.
- `%warpid` is an SM warp slot and is **not** a stable identity — named and
  explicitly rejected as a source of warp membership.
- Grid-level linearization: `bid = blockIdx.x + gridDim.x*(blockIdx.y +
  gridDim.y*blockIdx.z)`; block *dispatch order* is not architecturally
  guaranteed.
- Problem-to-thread mappings: 1 thread : 1 element; 1 thread : N elements
  (thread coarsening); **grid-stride loop** and the four reasons it is the robust
  default (config-independent correctness, machine-sized grids, persistent-kernel
  style, stride = whole grid preserves contiguity).
- "One wave": `blocksPerSM * nSMs` as the machine-derived grid size.
- Flattening multi-dimensional indices; **row-major vs column-major**; row stride
  vs. number of useful columns.
- The stride-matching rule: **give `threadIdx.x` to the axis with the smallest
  memory stride** — invariant under layout change, even though the code is not.
- 3-D flattening `(b*R + r)*C + c` and its stride triple `(R*C, C, 1)`.
- **Bounds checking**: mandatory whenever `n % block != 0`; why the cost is a
  uniform predicate for all but the straddling warp; guarding *every* axis.
- Block-size selection: multiples of 32; why 1024-thread blocks cap at 66.7%
  occupancy on sm_89 (1536 threads/SM ÷ 1024 = 1 block); 128/256 as defaults.
- Grid computation `(n + block - 1) / block` and its **integer-overflow trap**;
  the two safe forms.
- Launch limits: `gridDim.x ≤ 2³¹−1` but `gridDim.y`, `gridDim.z ≤ 65535`;
  `blockDim` product ≤ 1024.
- 32-bit unsigned overflow in `blockIdx.x * blockDim.x`; cast *before* the
  multiply.
- Sector (32 B) vs. cache line (128 B) granularity, at the level needed to count
  a warp's footprint; why 8 contiguous floats is still exactly one sector.
- "Effective GB/s above the DRAM peak" as a diagnostic that the working set is
  L2-resident.

## CUDA API / intrinsics / syntax introduced

- `dim3` construction and defaulting; `uint3`.
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim` (all `.x/.y/.z`).
- Multi-dimensional launch syntax `kernel<<<dim3 grid, dim3 block>>>(...)`.
- PTX special register `%laneid` via inline `asm` (read-only, used for
  verification only); `%warpid` named and rejected.
- `__restrict__` on kernel pointer parameters (mentioned; Module 20 owns it).
- `cudaEvent_t`, `cudaEventCreate/Record/Synchronize/ElapsedTime/Destroy` for
  kernel timing (introduced in Module 2's error-checking harness style; used here
  with warm-up + 20 iterations).
- `cudaDeviceProp.multiProcessorCount`, `.maxThreadsPerMultiProcessor` used to
  derive a wave-sized grid.
- `cudaMallocPitch` — **named as a forward reference only**, not used.

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Fill in the code (2-D) | 3 | 5-point clamped stencil on a 1021×733 and a 4093×3079 row-major image, run with four 256-thread block shapes | TODO 1 and TODO 3 are the *same decision written twice*: if x carries `col` then `gridDim.x` must come from `w`. Swapping x/y leaves 208,863 pixels unwritten and is caught only because the image is non-square and non-power-of-two. The performance trap is the opposite of the folk rule: (8,32) is within 2% of (32,8), while `blockDim.x == 1` is 3.1× slower. |
| `exercise02.cu` | CPU → GPU (spec §6 type 5) | 3 | Port a triple-nested loop over `in[37][1013][577]`; reader designs decomposition, layout, block, grid, flattening; harness requires a genuine z axis | Every axis assignment is *correct*; only one is fast. Handing `threadIdx.x` to the row axis instead of the column axis passes validation and runs **3.5× slower** (0.646 ms → 2.254 ms). A guard on two of three axes also passes on most inputs. `(c+1) % C` costs an integer-division sequence where a compare suffices. |
| `exercise03.cu` | Predict the behavior (spec §6 type 2) | 3 (10 predicted values) | Name the 32 (threadIdx.x, threadIdx.y) pairs of warp 1 and the addresses they touch, for a (8,32) and a (32,8) block; program prints the lane roster and scores the prediction | Most readers predict that the (8,32) warp, spanning 4 rows, costs *more* sectors than the single-row (32,8) warp. It costs **the same 4 sectors** — 8 floats is exactly one 32-byte sector. The real cost is a 4× cache-line footprint *per block* (32 lines vs 8). Also: warp 1, not warp 0 — the off-by-one that makes `min` 2048 rather than 0. |

`OVERALL: PASS` in exercise 3 requires both a correct numerical result and 10/10
predictions, so the file cannot be "passed" by running it first.

## Assumed from earlier modules

- **M1**: SM / warp / block / grid vocabulary; 40 SMs, 1536 threads/SM, 48 warps,
  24 blocks/SM, 65536 registers/SM; blocks are assigned to one SM and do not
  migrate; the concept of a *wave*; 432.0 GB/s peak bandwidth; latency hiding via
  resident warps.
- **M2**: `nvcc -arch=sm_89`; `__global__` / `__device__` / `__host__`; the
  `<<<grid, block>>>` launch; the `CHECK(...)` macro idiom; checking **both**
  `cudaGetLastError()` (launch-time errors) and `cudaDeviceSynchronize()`
  (execution-time errors); `cudaMalloc`/`cudaMemcpy`/`cudaFree`/`cudaDeviceReset`;
  deterministic host-side initialization and PASS/FAIL validation against a CPU
  reference.

## Forward references made

- **Module 5 (global memory & coalescing)** — repeatedly, and deliberately: exact
  transaction/sector counting, alignment, why the row stride should be padded,
  `cudaMallocPitch`. The lesson states that the linearization rule is the input to
  every coalescing argument.
- **Module 6 (shared memory)** — mentioned as the reason larger blocks can amortise
  work; no shared memory is used anywhere in this module.
- **Module 8 (warps and SIMT)** — predication vs. real divergence; why a uniform
  bounds predicate costs one instruction of issue and not a divergence penalty;
  reconvergence.
- **Module 10 (races and atomics)** — an in-place stencil / in-place blend as a
  genuine cross-block data race; atomics named, not used.
- **Module 12 / Module 18** — thread coarsening (1 thread : N elements) as an
  optimization ladder step.
- **Module 19 (occupancy)** — register and shared-memory pressure added to the
  1536-threads/SM arithmetic; "maximum occupancy ≠ maximum speed". The 66.7%
  ceiling of 1024-thread blocks is stated here and deferred there.
- **Module 20 (latency hiding)** — `__restrict__` and ILP.
- **Module 21 (roofline)** — "achievable" vs. "peak" bandwidth; why a
  compulsory-traffic model can report 62% while the kernel is at its real ceiling.
- **Module 29 (cooperative groups)** — grid-wide barriers and cooperative
  launches, as the only mechanism that would make an in-place variant safe; also
  persistent kernels.
