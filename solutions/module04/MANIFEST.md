# Module 04 manifest

## Concepts taught

- **Memory space** as a physical location, not just a C++ storage class
- **Register file** — per-SM SRAM, 65 536 × 32-bit on sm_89, 255/thread ceiling
- **Register allocation for warp lifetime** — why warp switching is free, and why
  `regs_per_thread` caps resident warps (`65536 / (regs × 32)`)
- **Local memory** — thread-private storage in *device DRAM*, cached in L1/L2
- **Two distinct causes of local memory**: register spilling (capacity) and
  dynamic indexing of a per-thread array (addressability)
- **Register file is not addressable** — register numbers are encoded in the
  instruction word, hence no "load register number `i`"
- **`stack frame` vs `spill stores`/`spill loads`** in `-Xptxas -v`, and how to
  tell the two causes apart from that one line
- **Per-thread local memory interleaving** — warp-coalesced but still DRAM
- **Shared memory** — on-chip, per-block scope and lifetime, ~20–30× lower
  latency than global (*introduced only; Module 6 owns it*)
- **Unified L1 + shared memory**, 128 KB/SM on Ada, 100 KB max addressable as
  shared, 48 KB/block default and 99 KB opt-in
- **L1 carveout** as a hint, via `cudaFuncAttributePreferredSharedMemoryCarveout`
- **L1 is per-SM and not coherent across SMs**; L2 is the device-wide coherence
  point; kernel boundaries flush/invalidate L1
- **L2 = 48 MB on this GPU**, and the benchmarking consequence: buffers must be
  sized ≥ 4× L2 to measure DRAM; working set = *sum of all buffers touched*
- **Global memory** — 12 GB GDDR6, 432 GB/s, ~575-cycle dependent-load latency
- **`cudaMalloc` 256 B alignment guarantee** and why it matters for `float4`
- **Constant memory** — 64 KB window + per-SM constant cache
- **The broadcast rule** — constant memory is fast only when all 32 lanes of a
  warp read the same address; divergent constant reads are replayed once per
  distinct address (measured 24.6× penalty)
- **Read-only data path** via `const __restrict__` / `__ldg`; `LDG.E.CONSTANT`
  in SASS; compiler usually infers it on modern architectures
- **Texture objects are mostly legacy for compute** — remaining use is
  interpolation / addressing modes
- **Pageable vs pinned host memory**; why a pageable H2D copy stages through a
  driver bounce buffer (*introduced only; Module 26 owns it*)
- **Pointer chasing** as the method for measuring latency (zero memory-level
  parallelism by construction)
- **Benchmark hygiene**: clock warm-up on a laptop GPU; steady-state vs
  compulsory misses; keeping a random walk within TLB reach
- Compile-time specialization (**templating on a shape**) as the standard way to
  turn a runtime value into a compile-time constant

## CUDA API / intrinsics / syntax introduced

- `__constant__`, `cudaMemcpyToSymbol`
- `__device__` variables (global-memory statics)
- `__shared__` (named and located only; semantics deferred to Module 6)
- `const T* __restrict__`, `__ldg()`
- `cudaFuncGetAttributes` → `numRegs`, `localSizeBytes`, `constSizeBytes`
- `cudaFuncSetAttribute` with `cudaFuncAttributePreferredSharedMemoryCarveout`
  (and `cudaFuncAttributeMaxDynamicSharedMemorySize` mentioned)
- `cudaMallocHost` / `cudaFreeHost`, `cudaHostAlloc` (mentioned)
- `cudaDeviceGetAttribute` with `cudaDevAttrL2CacheSize`,
  `cudaDevAttrMaxRegistersPerMultiprocessor`,
  `cudaDevAttrMaxSharedMemoryPerMultiprocessor`,
  `cudaDevAttrMaxSharedMemoryPerBlockOptin`, `cudaDevAttrTotalConstantMemory`
- `clock64()` device-side cycle counter
- `#pragma unroll`
- `template <int N> __global__` kernels and host-side `switch` dispatch
- `float4` vectorized access (for bandwidth saturation only; Module 5 explains it)
- Toolchain: `nvcc -Xptxas -v`, `cuobjdump -sass`, `compute-sanitizer --tool memcheck`
- SASS mnemonics named: `LDL`, `STL`, `LDG.E`, `LDG.E.CONSTANT`, `FFMA`

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Predict + measure | 4 | One-thread pointer chase across six working-set sizes; find the L1/L2 and L2/DRAM cliffs and commit to predicted latencies first | Choosing a stride smaller than the 128 B line makes each link a free hit and collapses the whole curve to 40/65/100 cycles; and 48 MB of L2 is big enough that an "obviously large" 32 MB buffer measures L2 while you call it DRAM |
| `exercise02.cu` | Debugging / optimization | 3 | A 16-tap FIR runs 3× slow; diagnose from `-Xptxas -v`, get the per-thread window into registers, then handle a genuinely runtime tap count | `#pragma unroll` on a loop whose bound is a kernel *argument* does not remove the dynamic index — the stack frame stays at 64 B. The bound itself must become compile-time, which for TODO 2 means the constant has to come from a template parameter, not from the kernel signature |
| `exercise03.cu` | Performance reasoning | 4 | Choose the memory space for a 256 B read-only table under a warp-uniform index and under a lane-varying index, predicting the ratios quantitatively first | "It fits in 64 KB" is the wrong test. Constant memory under a lane-varying index is 24.6× *slower* than plain global memory, because the hardware replays the instruction once per distinct address — the penalty scales with warp width, not table size |

Solutions: `solutions/module04/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module04/check_your_understanding.md`.

Worked examples: `example01.cu` (space tour + register-vs-local demonstration),
`example02.cu` (L2 residency sweep + pinned vs pageable H2D).

## Assumed from earlier modules

- M1: SM count (40), warp = 32 threads, 48 resident warps/SM, 1536 threads/SM,
  blocks are assigned to one SM and never migrate, 432 GB/s peak derivation
- M2: `nvcc -arch=sm_89`, `__global__`/`__device__`/`__host__`, `<<<>>>` launch
  syntax, `cudaGetLastError` + `cudaDeviceSynchronize` error-checking discipline,
  the `CHECK(...)` macro idiom
- M3: `blockIdx.x * blockDim.x + threadIdx.x`, bounds guards, grid-stride loops
- `cudaEvent_t` timing with warm-up and ≥20 timed iterations

## Forward references made

- **Module 5** (coalescing, transactions, sectors): the 128 B line / 32 B sector
  arithmetic behind Exercise 1's stride, and why a warp's 32 scattered 4 B reads
  cost what they cost
- **Module 6** (shared memory): shared memory is located and priced here but not
  taught; named as the right answer for a heavily reused lane-varying table
- **Module 7** (bank conflicts): shared memory has its own replay mechanism with
  the same architectural cause as the constant-memory replay shown here
- **Module 9** (`__syncthreads`, memory ordering, fences): why `volatile` is not
  a cross-SM synchronization mechanism; what acquire/release actually require
- **Module 10** (atomics): device-scope atomics as the correct tool for
  cross-block visibility
- **Module 12** (mentioned implicitly via benchmarking methodology): sizing
  working sets past L2 for every bandwidth measurement
- **Module 19** (occupancy): register count caps resident warps;
  `65536 / (regs × 32)`; the 142-register instantiation capping at 14 warps
- **Module 26** (pinned memory, async copies, streams): pinned memory is
  measured here but its real payoff — genuine `cudaMemcpyAsync` and copy/compute
  overlap — is deferred
- Nsight Compute *Memory Workload Analysis* named as the profiler section that
  reports local-memory traffic (tooling module)
