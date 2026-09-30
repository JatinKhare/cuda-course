# Module 02 manifest

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers. Readers should not open this
> file before attempting the exercises.

## Concepts taught

- **`.cu` translation unit** — one file containing host code and device code.
- **`nvcc` as a compiler driver** — front-end split (`cudafe++`), device path
  (`cicc` → PTX → `ptxas` → SASS), **fatbinary**, host path delegated to the
  host compiler (MSVC here). High level only.
- **Host code / device code**, **PTX** (virtual ISA) vs **SASS** (Ada machine
  code), `-arch=sm_89`.
- **Execution-space qualifiers**: `__host__`, `__device__`, `__global__`,
  `__host__ __device__` — what each compiles to and who may call it.
- **Kernel** — a `__global__` entry point, instantiated once per thread.
- **Why `__global__` must return `void`** — the launch returns before execution
  and there are N thread instances, so no return value is definable.
- **Dual compilation of `__host__ __device__`** — one source, two object codes;
  guarantees the CPU reference and the kernel are the same algorithm but **not**
  bitwise-identical results (FMA contraction).
- **Launch syntax** `<<<grid, block, sharedBytes, stream>>>` — all four
  parameters introduced; `dim3` shorthand.
- **Kernel arguments are passed by value** into a per-launch parameter buffer.
- **Asynchronous launch** — the launch enqueues and returns; measured host cost
  3–7 µs into a busy queue, 10–20 µs after an idle queue (WDDM), versus
  millisecond-scale kernels.
- **Command queue / GigaThread dispatch** as the hardware reason for asynchrony.
- **Why a host clock cannot time a kernel**; `cudaEvent_t` timing with warm-up
  and ≥ 20 iterations.
- **Launch overhead as a floor** — sub-50 µs kernels are overhead-dominated.
- **`cudaDeviceSynchronize()` as a blunt instrument** — device-wide, destroys
  overlap, names no culprit; development tool, not a release idiom.
- **Stream ordering** and the fact that blocking `cudaMemcpy` on the default
  stream already synchronizes — so an explicit sync before a D2H copy is
  usually redundant.
- **Two-class error model**: launch-configuration errors (caught by
  `cudaGetLastError()` at the launch site) vs execution errors (caught only by a
  synchronizing call).
- **`cudaGetLastError()` clears, `cudaPeekAtLastError()` does not**; the
  last-error slot is a single per-host-thread mutable global.
- **Sticky vs non-sticky errors** — a sticky error destroys the CUDA context;
  every subsequent call (including `cudaFree`, `cudaMalloc`) returns it; no
  in-process recovery; `cudaGetLastError()` does not clear it.
- **`CHECK()` and `CHECK_KERNEL()` macros** — single-evaluation,
  `do{}while(0)`, `__FILE__`/`__LINE__` at the call site, `cudaGetErrorName`
  plus `cudaGetErrorString`; compiling the sync out for release.
- **`compute-sanitizer --tool memcheck`** with `-lineinfo` for localizing
  illegal accesses.
- **Device-side `printf`** — global-memory FIFO with atomic reservation, flushed
  only at synchronization points, 1 MB default with silent oldest-first drop,
  no inter-thread ordering guarantee, perturbs timing and occupancy.
- **Host stdio buffering** interacting with device output; `fflush(stdout)`.
- **`cudaMalloc` / `cudaMemcpy` / `cudaMemset` / `cudaFree`** — byte counts,
  `void**`, 256-byte alignment, `dst` first.
- **Host pointers vs device pointers** — dereferencing a device pointer on the
  host is a segmentation fault, not a CUDA error; passing a host pointer to a
  kernel is `cudaErrorIllegalAddress`; the type system cannot help, hence the
  `h_`/`d_` convention.
- **Bounds guard must cover the load as well as the store**; ceiling-division
  grid sizing.
- **Validation discipline** — CPU reference, tolerance `1e-5 * max(1, |ref|)`,
  PASS/FAIL, deterministic index-derived initialization.

## CUDA API / intrinsics / syntax introduced

- `__global__`, `__device__`, `__host__`, `__host__ __device__`,
  `__forceinline__`
- `kernel<<<grid, block, sharedBytes, stream>>>(args)`
- `dim3`
- `blockIdx.x`, `blockDim.x`, `threadIdx.x` (1-D use only)
- `cudaError_t`, `cudaSuccess`
- `cudaGetLastError`, `cudaPeekAtLastError`, `cudaGetErrorName`,
  `cudaGetErrorString`
- `cudaDeviceSynchronize`
- `cudaSetDevice`, `cudaDeviceReset`
- `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`
- `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`
- `cudaEvent_t`, `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`,
  `cudaEventElapsedTime`, `cudaEventDestroy`
- `printf` in device code
- `fmaf`
- Error enums named explicitly: `cudaErrorInvalidValue`,
  `cudaErrorInvalidConfiguration`, `cudaErrorLaunchOutOfResources`,
  `cudaErrorIllegalAddress`, `cudaErrorMisalignedAddress`,
  `cudaErrorLaunchFailure`, `cudaErrorMemoryAllocation`, `cudaErrorUnknown`
- Tooling: `nvcc -arch=sm_89 -O3`, `-lineinfo`, `-fmad=false`,
  `compute-sanitizer --tool memcheck --show-backtrace no`
- Mentioned, not used: `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, …)`,
  `cudaFuncAttributeMaxDynamicSharedMemorySize`, `cudaMemcpyAsync`,
  `cudaMallocHost`, `cudaStreamSynchronize`, `cudaEventQuery`,
  `cudaMallocManaged`, `__restrict__`

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Fill in the code | 3 | Degree-5 Horner polynomial over 1,000,003 floats, validated against a CPU reference that calls the same `horner5` | TODO 1: the obvious answer `__device__` compiles the kernel and then breaks the *host* pass with "identifier undefined in host code"; only `__host__ __device__` works. TODO 2: guarding only the store is still an out-of-bounds **read** that usually does not fault. |
| `exercise02.cu` | Debugging | 2 | `saxpy`-with-clamp reporting `cudaErrorIllegalAddress` at an obviously-correct `cudaMemcpy` and then at `cudaFree` | Two compounding defects: a byte-count passed where an element count was wanted (grid 4× too large), and a launch checked with `cudaGetLastError()` but never synchronized — so the sticky fault is attributed to three innocent later calls. The tempting "fix" (allocate more) converts a loud crash into a silent wrong answer. |
| `exercise03.cu` | Predict the behavior | 3 | Predict host/device output interleaving, launch-statement cost bucket, and whether an unsynchronized D2H `cudaMemcpy` sees the kernel's results | TODO 3: the intuitive answer ("no — you must call `cudaDeviceSynchronize()` first") is **wrong**; stream ordering plus the blocking form of `cudaMemcpy` already guarantee it. TODO 1's second half: device `printf` lines are *not* in thread-id order — block 1 prints before block 0. |

All three ship compiling and non-crashing with TODOs unfilled
(`exercise01` prints `Set TODO 3 (blocks) first.`, `exercise03` prints a
reminder). `exercise02` is the deliberate-crash exception permitted by the spec
for debugging exercises.

## Worked examples

| File | Demonstrates |
|---|---|
| `example01.cu` | `__global__`, device `printf`, host/device output interleaving, host-clock cost of the launch statement (idle vs busy queue) against event-measured kernel time, `<<<1,1025>>>` caught by `cudaGetLastError()` and shown to be non-sticky, `cudaGetLastError()` clearing on read, device pointer value printed |
| `example02.cu` | Full malloc/memcpy/launch/memcpy/free pipeline with `__host__ __device__` shared math and PASS/FAIL validation; `--illegal` shows a sticky `cudaErrorIllegalAddress` poisoning every subsequent call; `--hostderef` segfaults on a host dereference of a device pointer |

## Assumed from earlier modules

- Module 1: SMs (40 on this GPU), warps of 32, thread blocks, grids, blocks are
  dispatched whole to one SM and do not migrate, the GigaThread engine, the
  1024-threads-per-block and 1536-threads-per-SM limits, 432 GB/s peak
  bandwidth, latency hiding by warp count.
- Module 1's `CHECK()` macro and the two-check idiom, used there without
  explanation; this module explains it.
- Strong C/C++: pointers, `malloc`/`free`, preprocessor macros, `do{}while(0)`,
  IEEE-754 single precision.

## Forward references made

- **Module 3** — thread indexing in depth, multi-dimensional grids, flattening,
  bounds, grid-stride loops. Stated repeatedly that this module keeps indexing
  to the 1-D `blockIdx.x * blockDim.x + threadIdx.x` form on purpose.
- **Module 4** — memory hierarchy; pinned host memory (`cudaMallocHost`);
  `__restrict__` / read-only path.
- **Module 5** — coalescing and the relevance of `cudaMalloc`'s 256-byte
  alignment.
- **Module 6** — the third launch parameter, dynamic shared memory
  (`sharedBytes`), and the 48 KB default / 99 KB opt-in limit.
- **Module 10** — data races; `compute-sanitizer --tool racecheck`.
- **Module 19** — occupancy arithmetic (mentioned via the one-warp-per-SM
  latency-bound kernel in Exercise 3's performance discussion).
- **Module 24** — the fourth launch parameter (streams),
  `cudaStreamSynchronize`, `cudaEventQuery`, `cudaMemcpyAsync` ordering,
  overlap.
- **Module 37** — the compilation pipeline in full: PTX, JIT, fatbinary layout,
  separate compilation. Explicitly deferred in §1 of the lesson.
- Mentioned in prose without a module number: CUDA Dynamic Parallelism
  (`__global__` callable from device code), CUDA Graphs (as the mechanism that
  amortizes launch overhead), unified/managed memory.
