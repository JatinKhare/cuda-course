# Module 01 manifest

## Concepts taught

- **Latency** vs **throughput**, as distinct optimization targets.
- **Little's Law** (`concurrency = throughput x latency`) as the governing
  equation for both grid sizing and occupancy targets.
- **Latency-minimizing** (CPU) vs **throughput-maximizing** (GPU) microarchitecture;
  transistor-budget argument; when each wins.
- **ILP / DLP / TLP** as three distinct sources of parallelism.
- **SIMD** vs **SIMT**: per-lane register state, invisible vector width, implicit
  active mask, every load potentially a gather.
- The two leaks in the SIMT abstraction:
  - **warp divergence** and the **active mask** (serialized paths; warp-local,
    not thread-local; branches uniform within a warp are free),
  - **uncoalesced access** and the **32-byte sector** as the unit of memory
    traffic (same instruction count, 8x the bytes).
- **Streaming Multiprocessor (SM)** anatomy on Ada (sm_89): 4 **processing
  blocks** (sub-partitions), each with 1 **warp scheduler** (≤1 instruction per
  clock), a 16,384-register file slice, 32 FP32 lanes (16 of them also INT32),
  one 4th-gen **Tensor Core**, 4 SFU, 8 LD/ST, ≤12 resident warps; plus a
  per-SM 128 KB unified L1 + shared memory block (≤100 KB addressable as shared).
- **CUDA core** = one FP32 lane; 5120 advertised cores = 40 SMs x 128 lanes;
  160 independent instruction streams' worth of issue, not 5120 processors.
- **Warp**, **block (CTA)**, **grid**, **thread**; which levels are physical.
- **Warp formation**: threads linearized as
  `x + y*blockDim.x + z*blockDim.x*blockDim.y`, then cut into groups of 32;
  non-multiple-of-32 blocks waste whole lanes.
- **Warp scheduling per cycle**: **eligible** vs **stalled** warps, scoreboarding,
  one issue per scheduler per clock, ≤4 warp-instructions per SM per clock.
- **Zero-cost context switch**: per-warp register allocation held for the warp's
  lifetime; the dual of register-pressure-limits-occupancy.
- **Physical kernel launch sequence**: asynchronous launch into a **pushbuffer**
  / command buffer; front end; **GigaThread work distributor**; block placement
  gated by block slot / thread slots / registers / shared memory; blocks are
  **indivisible** and **never migrate**; retirement frees resources and the next
  block is placed.
- **Waves**: `blocksPerWave = blocksPerSM x numSMs`,
  `waves = ceil(gridDim / blocksPerWave)`; **wave efficiency**; **tail effect** /
  quantization inefficiency; tail cost ~ `1 / waves`; wall-clock is a staircase
  in waves, not a ramp in blocks.
- Why a grid-wide barrier cannot be cheap (residency guarantee scope).
- Measurement hygiene: `cudaEvent` timing, warm-up, timing separated from
  host-side validation, min-of-N sweeps, and why a nominal clock makes `%peak`
  meaningless on a laptop part.

## CUDA API / intrinsics / syntax introduced

- `kernel<<<grid, block>>>(args)` launch syntax (used; formally taught in Module 2)
- `__global__`, `__device__`, `__host__`, `__forceinline__`, `__restrict__`
- `blockIdx`, `blockDim`, `threadIdx`, `gridDim`
- `cudaSetDevice`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`
  (`cudaDevAttrClockRate`, `cudaDevAttrMemoryClockRate`,
  `cudaDevAttrGlobalMemoryBusWidth`, `cudaDevAttrL2CacheSize`)
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`, `cudaDeviceReset`
- `cudaGetLastError`, `cudaDeviceSynchronize`, `cudaGetErrorName`,
  `cudaGetErrorString`; the `CHECK(...)` macro idiom
- `cudaEvent_t`, `cudaEventCreate/Record/Synchronize/ElapsedTime/Destroy`
- `clock64()`
- Inline PTX: `asm volatile("mov.u32 %0, %%smid;")` (exercise01)
- `fmaf()` in device and host code

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | fill in the code | 4 | Derive peak bandwidth and residency capacities from device attributes, then observe block-to-SM placement via `%smid`. | The DDR factor of 2 in the bandwidth formula (CUDA reports the command clock, not the data rate) — the error is exactly 2x; and using `maxThreadsPerBlock` where `maxThreadsPerMultiProcessor` is meant. |
| `exercise02.cu` | predict the behavior + performance reasoning | 3 | Time a compute-bound kernel at half/-1/exact/+1/2x/2x+1 waves and explain the staircase; predictions P1–P5 committed before building. | TODO 1: `maxBlocksPerMultiProcessor` is a device property named exactly after the thing you want and is the wrong answer — you must ask the runtime for the per-kernel figure. TODO 2 needs ceiling division and a floating-point efficiency. TODO 3: an FMA is 2 FLOPs and idle block slots are not work. |

Worked example: `example01.cu` (no TODOs) — dependent pointer chase at 1..24
resident warps per SM, showing linear throughput scaling then a knee, with
per-access latency never improving.

## Assumed from earlier modules

- Nothing. This is Module 1.
- Assumed from outside the course: strong C/C++, and undergraduate computer
  architecture (pipelining, caches, DRAM latency, out-of-order execution, SIMD).

## Forward references made

- Coalescing, sectors, transaction counting — "Module 5 treats this properly."
- Shared memory and `__syncthreads()` as a usable primitive — Module 6.
- Divergence, predication, reconvergence, active masks, independent thread
  scheduling on sm_70+ — "Module 8 makes this precise."
- Barriers, memory ordering, why `__syncthreads()` is cheap — Module 9.
- Occupancy arithmetic (registers/shared memory vs resident warps) — Module 19.
- Eligible-vs-stalled stall-reason analysis and ILP as an alternative to
  occupancy — Module 20.
- `grid.sync()` / cooperative launch and its co-residency precondition —
  Module 29.
- Grid-stride loops (mentioned in the Prediction section as the thing Module 3
  will add).
- Atomics mentioned only in prose (none used).

## Constraints observed

- No shared memory, no atomics, no `__syncthreads()`, no warp intrinsics, no
  cooperative groups in any Module 1 code.
- Thread indexing limited to `blockIdx.x * blockDim.x + threadIdx.x`.
- Everything builds with `nvcc -arch=sm_89 -O3`, warning-clean, and prints PASS.
- No sm_90+ features.
