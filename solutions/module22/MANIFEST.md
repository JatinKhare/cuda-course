# Module 22 manifest — Nsight Systems

> **Authoring metadata, not reader material.** The Exercises table names the
> subtle traps and therefore contains spoilers.

Files: `module22/lesson.md`, `module22/example0{1,2}.cu`,
`module22/exercise0{1,2,3}.cu`.
Solutions: `solutions/module22/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module22/check_your_understanding.md`,
`solutions/module22/MANIFEST.md`.

All `.cu` verified with `nvcc -arch=sm_89 -O3` (CUDA 13.2, V13.2.51, RTX 3500
Ada, driver 596.71), warning-clean. Both examples print `OVERALL: PASS`; all
three solutions print `OVERALL: PASS` (6/6, 7/7, 7/7). All three shipped
exercises compile with TODOs blank and exit gracefully (`Set TODO 4 first.`,
`Set TODO 3/4/5 first.`, `Set TODO 4 first (the fused kernel and runFused).`)
returning 0. No binaries, `.nsys-rep` or `.sqlite` committed.

**`nsys` WORKS on this machine** and every profiler number in this module is a
real capture, taken with Nsight Systems 2025.6.3 (not on `PATH`; see the lesson
for the invocation). `ncu` remains blocked — Module 23's problem.

---

## Concepts taught

- **The division of labour between the two Nsight tools.** `nsys` traces a
  timeline and answers *is the GPU busy, and if not why not*; `ncu` reads
  hardware counters and answers *is this kernel efficient*. `nsys` needs no
  counter permission because tracing is not counting. **PORTABLE CUDA CONCEPT.**
- **GPU busy fraction** = device execution time / wall time of an NVTX-delimited
  region, computed from `cuda_gpu_kern_sum` ÷ `nvtx_sum`.
- **"Busy" ≠ "utilized".** A timeline can be solid green at 20% of the machine.
  Explicitly deferred to Module 23.
- **A program cannot reliably measure its own busy fraction.** `cudaEventRecord`
  enqueues a marker; the interval between two markers contains the kernel *and*
  any device idle between them, so an event pair is an **upper bound** on kernel
  duration, loosest exactly when the host is the problem. Demonstrated twice
  (Example 1: 71.4% claimed vs 55.8% true; Exercise 3: four stages differing
  3.9× in arithmetic all reported at ~8 µs).
- **Timeline rows and what each one means**: CPU thread, CUDA API (host time
  inside the call), kernel (device-timestamped), memory ops, streams.
- **A gap means the device has no work queued**; the cause is always host-side.
- **Overlap means two engines were busy**; in the default stream with pageable
  memory it never happens.
- **The host must be allowed to run ahead.** Anything that drains the launch
  queue costs one launch bubble, independent of what it appears to do.
- **Which CUDA calls are synchronous with respect to the host**: pageable
  `cudaMemcpy` (any direction, including `cudaMemcpyToSymbol`), `cudaMalloc`,
  `cudaFree`, `cudaDeviceSynchronize` — versus `cudaMemset`, device-to-device
  `cudaMemcpy` and kernel launches, which are not.
- **The cost of a call is unrelated to the bytes it moves.** A 4-byte D2H copy
  costs more host time than a 4 MB D2D copy.
- **Per-launch host cost on this machine: 8–14 µs**, measured four ways,
  flat in grid size from 1 to 1024 blocks. A kernel shorter than ~10 µs costs
  more to start than to run.
- **The sync tax is constant in kernel duration** (5–14 µs across a 115× sweep),
  because it is the cost of restarting an empty pipeline.
- **Serialization costs in series mask each other.** A ranked `cuda_api_sum`
  tells you what to fix first and never what to fix last.
- **The launch-overhead critical path.** Optimizing a kernel that is not the
  limiting resource is worth nothing; Amdahl applied to the timeline.
- **NVTX instrumentation**: push/pop vs start/end, the per-thread range stack,
  the RAII guard, the deleted copy constructor, the two-level token-paste macro,
  naming OS threads, and the granularity trade-off.
- **`--capture-range=cudaProfilerApi`** as the thing that makes CLI profiling
  usable, and the `cudaProfilerStart` row that is a duration and not a cost.
- **Reading `nsys stats` tables**: `cuda_api_sum`, `cuda_gpu_kern_sum`,
  `cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum`, `cuda_kern_exec_sum`
  (`AAvg`/`QAvg`/`KAvg` decomposition), `nvtx_sum`, `nvtx_gpu_proj_sum`.
- **Methodology, pushed back into the spec:** absolute milliseconds are not
  portable across sessions on this laptop (7.2× swing measured); gates must be
  built from ratios inside one capture.

## CUDA API / intrinsics / syntax introduced

- `#include <nvtx3/nvToolsExt.h>` — header-only on CUDA 13.2, **no link flag**
  (`-lnvToolsExt` is gone).
- `nvtxRangePushA` / `nvtxRangePop` — per-thread nestable range stack.
- `nvtxRangeStartA` / `nvtxRangeEnd` + `nvtxRangeId_t` — thread-independent
  ranges (named, used in prose, not required by any exercise).
- `nvtxNameOsThreadA` — naming a thread on the timeline.
- `#include <cuda_profiler_api.h>`, `cudaProfilerStart()`, `cudaProfilerStop()`.
- `cudaMemsetAsync`.
- `nsys profile` options: `--trace=cuda,nvtx[,osrt]`, `--stats=true`, `-o`,
  `--force-overwrite=true`, `--capture-range=cudaProfilerApi`,
  `--cuda-memory-usage=true`.
- `nsys stats`, `nsys stats --report <name>`, `nsys stats --help-reports`.
- `nsys-ui` named; the module is deliberately CLI-centred.

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | fill-in + design + instrumentation | 5 (6 scored points) | Write the NVTX guard, annotate a 300-step Jacobi solver, open a capture range, rewrite the host loop, report three numbers from a real capture | **Exactly two of the four host-side decisions are already asynchronous** (`cudaMemset` and the 4 MB device-to-device copy). The 4 MB D2D copy *looks* like the expensive one and costs 6.6 µs; the 4-byte blocking D2H copy costs the host >100 ms. Second trap: `cudaFree` is 247 µs per call, 25× a launch, and nobody suspects the allocator. Third: the one-level token-paste macro fails only when two ranges share a scope |
| `exercise02.cu` | debugging / diagnosis + design | 5 (7 scored points) | Diagnose and fix a FIR loop whose wall time is 40× its kernel time; three independent causes, one unnamed | **The third cause (a hidden `cudaDeviceSynchronize` inside a telemetry helper) is 0.1% of `cuda_api_sum` in the slow version** because the blocking copy before it already drained the queue — it is worth 1.19× once the other two are fixed. Second trap: `DIAG_TOP_API` is `cudaMemcpy`, not `cudaDeviceSynchronize`, which is what most people guess. Third: deleting the progress hook is not a fix; the harness counts its calls |
| `exercise03.cu` | prediction + performance reasoning + design | 5 (7 scored points) | Predict the launch cost, predict which of three optimizations wins, predict the margin, write the fused kernel, supply two numbers from a capture | **The longest kernel is 43% of device time and making it free is worth nothing**, because the device is not the limit. Second trap: the program's own event-based per-stage timing reports all four stages at ~8 µs in the *wrong order* for kernels differing 3.9× — the instrument is reporting its own floor, and the exercise cannot be answered without `nsys`. Third: bit-identical fusion is achievable *only because* every staged intermediate was already fp32 in global memory; a reader who "improves" the fused version with a double accumulator fails gate 1 |

Examples: `example01.cu` (NVTX + busy fraction, A/B host-side restructuring of an
n-body loop, PASS); `example02.cu` (launch floor and sync tax measured by sweeping
kernel duration over two decades, PASS).

## Assumed from earlier modules

- M1: the six-step launch path, GigaThread block distribution, waves and tail
  effects (41 blocks = 1.99× of 40), SM count 40.
- M2: asynchronous launch, `cudaDeviceSynchronize`, the `CHECK` macro pattern,
  sticky vs non-sticky errors.
- M3: grid/block sizing, bounds guards.
- M4: pinned vs pageable host memory *named* (the mechanism is M26's).
- M9: kernel boundaries as the only global barrier (used to explain why fusing
  changes the ordering guarantees).
- M10/M12: block reduction + one atomic per block (`residual` in Exercise 1).
- M11: kernel fusion, and the measured ~10 µs launch cost that made a fused
  chain beat its 2.00× traffic prediction at 3.31×. This module measures that
  constant head-on and confirms it.
- M12: `cudaEvent_t` timing, warm-up length, the 1500 ms rule.
- Spec §12: min-of-N / median, rotation, back-to-back timing, auto-scaled
  iteration counts, operating-point guarding.

## Forward references made

- **Module 23 (Nsight Compute)** — every question about what happens *inside* a
  kernel: occupancy, sectors, bank conflicts, stall reasons. Stated as the
  explicit complement in the lesson's opening table.
- **Module 24 (streams)** — the mechanism that makes transfer/compute overlap
  possible at all; named as the fix a timeline with no overlap motivates.
- **Module 25 (events)** — events as a *synchronization* primitive rather than a
  timer.
- **Module 26 (pinned memory)** — why `cudaMemcpyAsync` on pageable memory is
  still synchronizing, and what makes it stop being so.
- **Module 28 (CUDA graphs)** — removing per-launch cost when the kernels cannot
  be fused. Named in Example 2's output and in Exercise 3's.

## Proposed changes to shared files (for the orchestrator)

1. **`AUTHORING_SPEC.md` §12** — the sentence "Modules 22–23 will be taught as
   theory plus screenshots-in-prose rather than live profiling runs" is now
   false for Module 22 and should be struck. Module 22 is built entirely on real
   captures.
2. **`AUTHORING_SPEC.md` §12.5c, strengthen with a measured magnitude.** The
   same binary and the same `nsys` command reported `residual` at **35 067.7 ns**
   in a cool session and **253 413.6 ns** after ~1 h of benchmarking — a **7.2×
   uniform swing** across all three kernels, `nvidia-smi` SM clock at 210 MHz.
   This is much larger than the 1.5× memory-clock drop §12.5 documents, and it
   broke a gate that compared a profiled capture against an un-profiled event
   bound. **Recommended new rule: any gate that compares a number from one run
   against a number from another run must be ratio-based or have a ≥3× window.**
3. **`CROSS_MODULE_INDEX.md` §6b — new rows:**
   - Per-launch host cost (WDDM, this laptop): **8–14 µs**; flat from 1 to 1024
     blocks; `cudaLaunchKernel` average 13 509.6 ns over 81 080 calls and
     8 171.5 ns over 1 600 calls. **Confirms M11's ~10 µs.**
   - Sync tax (`cudaDeviceSynchronize` after every launch): **5–14 µs, constant
     in kernel duration** over a 115× sweep.
   - `cudaFree` cost on a 4 MB buffer: **247 µs**, `cudaMalloc` **142 µs** —
     25× and 14× a kernel launch.
   - Pageable `cudaMemcpy` effective host-side rate, 4 MB: **~11.7 GB/s**
     (359 µs H2D, 344 µs D2H).
   - Minimum on-device kernel cost (grid launch + distribution + first memory
     round trip): **≈1.4 µs**, fitted from stage0/stage1 at rep 330 vs 85
     giving 2.29× rather than the arithmetic 3.88×.
   - Event-pair timing is an **upper bound** on kernel duration; measured
     inflation 13–16 percentage points of busy fraction in Example 1, and a
     complete failure (1.7–3.5× error, wrong ordering) for kernels below the
     launch floor in Exercise 3.
4. **`RESUME_HERE.md`** — mark Module 22 complete: 1 lesson, 2 examples, 3
   exercises, 3 verified solutions, `check_your_understanding.md`, manifest.

## Cross-module inconsistency spotted

- `AUTHORING_SPEC.md` §12's closing section still says Modules 22–23 will be
  theory-only, while the same section's new `nsys`-works block says the
  opposite. The two contradict each other; item 1 above resolves it.
- M11's "~10 µs launch overhead" was a single-session inference from a fusion
  ratio. This module measures it four independent ways and lands at 8–14 µs,
  which **confirms** M11 rather than correcting it. Worth recording as a
  cross-validation rather than a change.
- Example 1's surviving comment claimed `computeForces` at 152 651.8 ns ± 1002 ns
  from an earlier agent's capture. Re-measured here at **152 570.7 ns ± 1007.9 ns**
  — agreement to 0.05%, which is a useful demonstration that device-side kernel
  timing is reproducible even when wall times are not. Updated to the new figure.
