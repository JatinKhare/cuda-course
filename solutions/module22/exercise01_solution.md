# Module 22 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```bash
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
./exercise01_solution.exe

NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
"$NSYS" profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
     -o ex01sol --stats=true --force-overwrite=true ./exercise01_solution.exe
```

No `-l` flag: nvtx3 on CUDA 13.2 is header-only.

---

## TODO 1 — the NVTX scope guard

```cpp
struct NvtxRange {
    explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
    ~NvtxRange()                         { nvtxRangePop(); }
    NvtxRange(const NvtxRange &)            = delete;   // a copy would pop twice
    NvtxRange &operator=(const NvtxRange &) = delete;
};
#define NVTX_CAT2(a, b) a##b
#define NVTX_CAT(a, b)  NVTX_CAT2(a, b)
#define NVTX_RANGE(name) NvtxRange NVTX_CAT(_nvtxScope, __LINE__)(name)
```

**Why it is correct.** `nvtxRangePushA`/`nvtxRangePop` maintain a *per-thread
stack*. Nesting therefore has to match C++ scope nesting exactly, and the only
construct that guarantees that under early `return`, `break`, `goto` and
exceptions is a destructor. The deleted copy constructor closes the other hole:
a copy would run the destructor twice and pop a range it never pushed.

**Why two macro levels.** The token-paste operator `##` suppresses macro
expansion of its operands, so `_nvtxScope##__LINE__` produces the identifier
`_nvtxScope__LINE__`, literally. A second level forces `__LINE__` to expand
first. The symptom of getting this wrong is a redeclaration error naming
`_nvtxScope__LINE__`, which mentions neither NVTX nor the construct you wrote.

**Common wrong approaches.**

| Approach | Symptom |
|---|---|
| Bare `nvtxRangePushA`/`nvtxRangePop` at function top and bottom | works until the first early `return`; then every later range is reparented and the timeline is quietly wrong, not obviously wrong |
| Copyable guard | double pop. On nvtx3 this does not assert; the stack underflows and subsequent pops unwind ranges you still wanted |
| One-level macro | compile error on the second `NVTX_RANGE` in a scope |
| A range per kernel launch (900 of them) | measurable: NVTX push/pop is cheap but not free, and you are now instrumenting at a granularity finer than the thing you are measuring |

---

## TODO 2 — annotating `runNaive`

The shipped solution uses one range for the function, one per step, and one per
*phase* — `alloc`, `relax`, `converge-check`, `writeback`, `free`. Five phase
names, 300 steps, 1800 ranges total. That is cheap and it makes `nvtx_sum`
self-explanatory:

```
 Time (%)  Total Time (ns)  Instances   Avg (ns)      Range
     33.2        322984152        600    538306.9     :step
     28.3        275753540          1  275753540.0    :naive
     13.5        131535602        600    219226.0     :converge-check
      7.7         74611853        300    248706.2     :free
      6.4         61892955        600    103154.9     :relax
      5.5         54010518          1   54010518.0    :fixed
      4.4         43235517        300    144118.4     :alloc
      1.0          9739177        600     16232.0     :writeback
      0.0           213231          1    213231.0     :alloc-once
```

Read the ranking and you have the diagnosis before you open a single other
table: `converge-check` (131.5 ms) is the biggest phase, `free` (74.6 ms) is the
second, `alloc` (43.2 ms) is third, and the phase that does the actual physics —
`relax`, two of the three kernels — is **61.9 ms, fourth**. The one range that
contains no `cudaMalloc`, no `cudaFree` and no blocking copy is a minority of the
run.

`:step`, `:relax`, `:converge-check` and `:writeback` show 600 instances because
both `runNaive` and `runFixed` use them; `:alloc` and `:free` show 300 because
only the naive version has them. That asymmetry is itself the summary of the fix.

---

## TODO 3 — the capture range

```cpp
#include <cuda_profiler_api.h>
...
CHECK(cudaProfilerStart());
   /* both versions */
CHECK(cudaProfilerStop());
```

`--capture-range=cudaProfilerApi` makes the collector ignore everything outside
that bracket. Without it the warm-up's 2000 `jacobiStep` launches and the first
`cudaMalloc` (which can cost 100 ms on a cold context) are mixed into every
average, and the `cudaLaunchKernel` count is 3800 instead of 1800. The
exercise's harness checks per-launch costs, so a polluted capture fails the
gates — correctly.

Note that `cuda_api_sum` then grows a `cudaProfilerStart` row whose "Total Time"
is **the duration of the capture range**, not a cost. Ignore it.

---

## TODO 4 (design) — `runFixed`

```cpp
NVTX_RANGE("fixed");
float *scratch = nullptr;
{ NVTX_RANGE("alloc-once"); CHECK(cudaMalloc(&scratch, sizeof(float) * N)); }

CHECK(cudaEventRecord(w0));
for (int s = 0; s < STEPS; ++s) {
    NVTX_RANGE("step");
    { NVTX_RANGE("relax");
      tl.open(); jacobiStep<<<gBlocks, BLOCK>>>(a, scratch, N); tl.close();
      CHECK(cudaGetLastError());
      tl.open(); scaleBy<<<gBlocks, BLOCK>>>(scratch, N, 1.0f); tl.close();
      CHECK(cudaGetLastError()); }
    { NVTX_RANGE("converge-check");
      CHECK(cudaMemsetAsync(dRes, 0, sizeof(float)));
      tl.open(); residual<<<gBlocks, BLOCK>>>(scratch, a, dRes, N); tl.close();
      CHECK(cudaGetLastError()); }
    { NVTX_RANGE("writeback");
      CHECK(cudaMemcpy(a, scratch, sizeof(float) * N, cudaMemcpyDeviceToDevice)); }
}
CHECK(cudaEventRecord(w1));
CHECK(cudaEventSynchronize(w1));
CHECK(cudaMemcpy(&hostRes, dRes, sizeof(float), cudaMemcpyDeviceToHost));
CHECK(cudaFree(scratch));
```

### The four decisions, and which two were traps

The exercise says *exactly two of the four are already asynchronous with respect
to the host*. They are:

| Decision | Synchronous w.r.t. host? | What to do |
|---|---|---|
| per-step `cudaMalloc` / `cudaFree` | **Yes.** Both talk to the driver and synchronize the device. | **Hoist.** This is the big one. |
| per-step 4-byte blocking `cudaMemcpy` D2H | **Yes.** Pageable host destination; the call does not return until the device has caught up. | **Delete from the loop.** The algorithm only consumes the last value. |
| per-step `cudaMemset` of a 4-byte accumulator | **No.** `cudaMemset` on device memory is asynchronous with respect to the host in the default stream. | Leave it, or switch to `cudaMemsetAsync` as a micro-optimization. Do not claim it as a synchronization fix. |
| per-step 4 MB device-to-device `cudaMemcpy` | **No.** D2D never touches host memory, so there is nothing to stage and nothing to wait for. | Leave it. It is 4 MB of real device traffic — 6.59 µs per copy, 600 copies, 3.96 ms total — and it is honest work. |

The temptation is to rank the four by how expensive they *look*. The 4 MB D2D
copy moves 1000× more bytes than everything else combined and costs 3.96 ms of
device time. The 4-byte D2H copy moves 1200 bytes in total across the whole run
and costs the host **more than 100 ms**. Bytes and cost are unrelated here,
because the cost is the synchronization, not the transfer.

**Common wrong approaches.**

- *Deleting the residual readback entirely and never computing `hostRes`.* The
  harness checks the final residual. The value is genuinely needed — once.
- *Replacing the D2D copy with a pointer swap.* Faster, and it changes the
  launch count and the memcpy count, which the harness checks. More importantly
  it is a different optimization: it removes work, whereas this exercise is about
  removing *waiting*. (Do it afterwards as a variation; it is a further ~1.1×.)
- *Switching the 4-byte D2H to `cudaMemcpyAsync` and keeping it in the loop.*
  With a pageable host destination `cudaMemcpyAsync` is still synchronizing. With
  a pinned destination it would not be — which is Module 26, and it is the right
  instinct one module too early.
- *Calling `cudaMemsetAsync` and reporting it as the fix.* It is worth little.
  Measured: `cudaMemset` 4.26 ms over 300 calls (14.2 µs each); `cudaMemsetAsync`
  1.98 ms over 300 (6.6 µs each). A 2.3 ms saving in a 300 ms run.

---

## TODO 5 — what `nsys` said

```
 ** CUDA API Summary (cuda_api_sum):
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Name
     78.1       1150363144          1  1150363144.0  cudaProfilerStart   <- capture duration, not a cost
      7.3        107603662        905      118899.1  cudaMemcpy
      5.1         74452613        301      247350.9  cudaFree
      3.6         53127210       1800       29515.1  cudaLaunchKernel
      2.9         42834838        301      142308.4  cudaMalloc
      2.3         33715881       3604        9355.1  cudaEventRecord
      0.3          4259925        300       14199.8  cudaMemset
      0.1          1984430        300        6614.8  cudaMemsetAsync

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Name
     61.4         21040592        600   35067.7   residual(const float *, const float *, float *, int)
     20.4          7007053        600   11678.4   jacobiStep(const float *, float *, int)
     18.1          6217168        600   10361.9   scaleBy(float *, int, float)

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):
 Time (%)  Total Time (ns)  Count  Avg (ns)    Operation
     52.4          3956888    600    6594.8    [CUDA memcpy Device-to-Device]
     32.5          2459546    303    8117.3    [CUDA memcpy Device-to-Host]
      9.5           718558      2  359279.0    [CUDA memcpy Host-to-Device]
      5.6           422848    600     704.7    [CUDA memset]
```

So:

```cpp
#define NS_KERNEL_NS    34264813.0   // 21040592 + 7007053 + 6217168
#define NS_LAUNCH_NS    53127210.0
#define NS_SYNCCOST_NS 107603662.0
```

**The three facts in the table.**

1. **Device execution, both versions, 1800 launches: 34.26 ms.** That is 19.0 µs
   per launch, averaged over three kernels of 35.1 / 11.7 / 10.4 µs.
2. **Host time inside `cudaLaunchKernel`: 53.13 ms, 29.5 µs per call.** More than
   the device spent executing. (29.5 µs is inflated by the tracer; an
   un-instrumented probe on this machine measures 7.9–9.2 µs. Both are larger
   than two of the three kernels.)
3. **Host time inside `cudaMemcpy`: 107.60 ms over 905 calls.** Of those 905, 600
   are the 4 MB device-to-device copies, 303 are device-to-host (300 of which are
   *four bytes*), and 2 are the setup uploads. The device-side cost of all of them
   together is 6.4 ms. The other 101 ms is the host standing still.

And the one nobody predicts: **`cudaFree` is 74.45 ms over 301 calls — 247 µs
each, 25× the cost of a kernel launch.** `cudaMalloc` is another 42.83 ms.
Allocation and deallocation are not bookkeeping; they are driver calls that
synchronize the device, and in the naive loop there are 300 of each.

---

## Synchronization / memory reasoning

The naive loop's per-step sequence is:

```
cudaMalloc            -> drains the queue, ~142 us
jacobiStep  (async)
scaleBy     (async)
cudaMemset            -> async, ~14 us of host time
residual    (async)
cudaMemcpy 4 B D2H    -> drains the queue; host blocks until residual finishes
cudaMemcpy 4 MB D2D   -> async
cudaFree              -> drains the queue, ~247 us
```

Two drains per step, plus the fact that `cudaMalloc` and `cudaFree` are
*themselves* expensive even when nothing is queued. The device executes 57.1 µs
of kernel per step and the step takes 538.3 µs (`nvtx_sum`, `:step`). The fixed
version removes both drains and both allocator calls; its `:step` disappears
into a `:fixed` range of 54.01 ms against the naive `:naive` range of 275.75 ms.

Note what is *not* on that list: nothing about the kernels, the memory access
pattern, occupancy, or bank conflicts. `residual` is 61% of device time and it
would be the obvious target for Module 23's tools. Optimizing it to zero would
save 21.0 ms of a 275.8 ms range — **7.6%**. The host-side fix saves 80%.

---

## Performance reasoning

Measured, un-instrumented, on a cool part:

```
version             wall (ms)      GPU<=(ms)     busy<= %
naive                 255.442         27.190        10.6%
fixed                  51.005         37.784        74.1%
speedup        : 5.01x
```

Across several runs the speedup lands at **5.0–6.7×**. The share breakdown the
harness prints from the capture:

```
Share of the three accounted-for costs inside the capture:
  GPU actually executing       = 17.6%  (34.3 ms)
  host inside cudaLaunchKernel = 27.2%  (53.1 ms)
  host inside cudaMemcpy       = 55.2%  (107.6 ms)
```

### ⚠️ The measurement that changed how this exercise is scored

An earlier capture of the *identical binary and identical command* reported:

| | cool session | after ~1 h of benchmarking |
|---|---|---|
| `residual` average | **35 067.7 ns** | **253 413.6 ns** |
| `jacobiStep` average | 11 678.4 ns | 74 649.2 ns |
| `scaleBy` average | 10 361.9 ns | 63 850.1 ns |
| `nvtx_sum` `:naive` | 275.75 ms | 757.24 ms |

A **7.2× swing**, uniform across all three kernels, with `nvidia-smi` showing the
SM clock parked at 210 MHz. Spec §12.5c predicted this and it still caught the
first draft of this exercise, whose gate compared a profiled `nsys` number
against an un-profiled event bound from a different session and failed a correct
solution.

The shipped gates therefore compare the three `nsys` numbers **against each other
and against the launch count**, never against this run's wall clock:

- device time per launch in [4, 600] µs,
- host launch time per call in [2, 150] µs,
- and the structural inequality `memcpy > launch > kernel`, which is the finding
  itself and holds in every capture of this program, cool or throttled.

This is the generalizable rule: **absolute milliseconds are not portable across
sessions on this part; ratios inside one capture are.**

---

## Expected output

```
Module 22 / Exercise 1 — instrument, profile, fix
N = 1048576, 300 steps, 900 launches per version

version             wall (ms)      GPU<=(ms)     busy<= %
naive                 255.442         27.190        10.6%
fixed                  51.005         37.784        74.1%
speedup        : 5.01x

[x] 1. final residual matches  (naive 15.061044 vs fixed 15.061053)
[x] 2. final field matches     (0/1048576 cells differ)
[x] 3. speedup >= 3.0x         (got 5.01x)
[x] 4. nsys kernel time plausible (34.265 ms = 19.0 us x 1800 launches)
[x] 5. nsys launch cost plausible (53.127 ms = 29.5 us x 1800 launches)
[x] 6. nsys memcpy cost dominates (107.604 ms > launch 53.127 > kernel 34.265)

Share of the three accounted-for costs inside the capture:
  GPU actually executing       = 17.6%  (34.3 ms)
  host inside cudaLaunchKernel = 27.2%  (53.1 ms)
  host inside cudaMemcpy       = 55.2%  (107.6 ms)
This run's event bound said the GPU was busy <= 21.2% of 306.4 ms of
wall time. nsys says WHERE the rest went; the event pairs could only
say THAT it went.

SCORE: 6/6
OVERALL: PASS
```

Run-to-run: wall times move ±30% and the speedup lands in 5.0–6.7×. The three
`nsys` figures are from the capture quoted above; a capture of your own will
differ in absolutes and pass the same gates.

---

## The result that matters

**The cost of a CUDA call has almost nothing to do with the number of bytes it
moves.** In this program the 4-byte device-to-host copy is the second most
expensive thing the host does and the 4-megabyte device-to-device copy is nearly
free, because one of them forces the host to wait for the device and the other
does not. The event-pair harness inside the program can tell you that 89% of the
wall time was not kernel execution; only `nsys` can tell you that 55% of it was
spent inside `cudaMemcpy` and another 5% inside `cudaFree`. Optimizing the
dominant *kernel* to zero would have bought 7.6%.

**Variation to try:** replace the 4 MB device-to-device writeback with a pointer
swap between `a` and `scratch` and re-profile. The `cuda_gpu_mem_time_sum` D2D
row vanishes (3.96 ms of device time), the launch count is unchanged, and the
wall time moves by a further ~1.1×. Then ask why that change — which removes
real device work — is worth so much less than removing a copy of four bytes.
