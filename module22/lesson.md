# Module 22 — Nsight Systems: is the GPU busy, and if not, why not?

> Prerequisites: Modules 1–3 (launch path, grids), 2 (async launch, error checking),
> 11 (kernel fusion and the ~10 µs launch cost), 12 (timing methodology)
> What this module gives you: a reproducible way to find out what fraction of your
> program's wall-clock time the GPU spent executing anything at all, and where the
> rest of it went.

---

## Concept

Every performance question about a CUDA program is one of exactly two questions,
and they need different instruments.

| Question | Instrument | Module |
|---|---|---|
| **Is my GPU busy, and if not, why not?** | **Nsight Systems (`nsys`)** — a *timeline tracer* | **this one** |
| Is this kernel efficient while it runs? | Nsight Compute (`ncu`) — a *kernel profiler* | 23 |

The distinction is not a matter of taste. The two tools measure different things
by different mechanisms:

- **`nsys` traces.** It hooks the CUDA driver and runtime, records a timestamped
  event for every API call the host makes and every kernel and copy the device
  executes, and reconstructs a wall-clock timeline. It needs no GPU performance
  counters and therefore no elevated permission. It can tell you *when* a kernel
  ran and for how long. It cannot tell you anything about what happened inside it.
- **`ncu` profiles.** It replays a kernel, reads hardware performance counters,
  and reports sector counts, bank conflicts, warp stall reasons, occupancy. It
  needs counter access. It tells you nothing about the host, the gaps, or any
  kernel it is not currently replaying.

If the GPU is executing 15% of the time, the most efficient kernel in the world
buys you at most 15% of your wall clock. **`nsys` comes first.** It tells you
whether a kernel-level investigation is worth doing at all.

### The one number this module exists to produce

> **GPU busy fraction** = (total device execution time) / (wall-clock time of the
> region you care about)

Both quantities are directly readable from a `nsys` capture: the numerator from
`cuda_gpu_kern_sum` (plus `cuda_gpu_mem_time_sum` if you want to count copies as
"busy"), the denominator from an NVTX range you placed around the region. The
whole module is about computing that number honestly and then reading the gap.

Four things to be precise about before we go further.

**1. "Busy" means "a kernel was resident", not "the machine was well used".**
A grid of 8 blocks on a 40-SM GPU can keep the timeline solid green while using
20% of the machine. `nsys` will tell you the GPU was 100% busy and it will be
telling the truth. Whether that busy time was *efficient* is Module 23's
question, and the two are genuinely independent. This module's job is to find
out whether you have a gap problem; it deliberately says nothing about what the
kernels do.

**2. A gap on the timeline is the GPU waiting for the host.** The device has no
work queued. Something on the CPU has to put work in the queue, and it has not
done so yet. The possible reasons are few and this module enumerates them.

**3. Overlap on the timeline means two engines were busy at once.** Ada has a
copy engine (or several) separate from the SMs, so a transfer and a kernel can
run simultaneously — but only if the program expressed that possibility. In the
default stream with pageable memory it never happens, and the timeline shows a
strict alternation: copy, kernel, copy, kernel. Making that overlap happen needs
streams (Module 24), events (Module 25) and pinned memory (Module 26). This
module's job is to make you *want* those three.

**4. A program cannot reliably measure its own GPU busy fraction.** This is the
deepest point in the module and it is worth stating as a theorem:

> `cudaEventRecord` enqueues a timestamp into a stream. The interval between two
> enqueued markers is the distance on the *device* timeline between them. That
> interval contains the kernel AND any stretch in which the device sat idle
> between the two markers waiting for the host to enqueue the kernel. So an
> event pair bracketing a launch is an **upper bound** on the kernel's duration,
> and the bound is loosest exactly when the host is the problem.

Module 22's examples measure the same quantity both ways and show the difference.
Example 1's event harness reports version A at "≤ 71.4% busy"; the `nsys` capture
of the same binary says **55.8%**. The event pairs could say that time
disappeared. Only the tracer could say where it went.

### The classic findings, all of which this course has already caused

Every item below has a measured instance earlier in this course. Nsight Systems
is how you would have found each of them without knowing to look.

| Finding | Timeline signature | Already seen in |
|---|---|---|
| **Launch overhead dominating** | A dense picket fence of short kernels with gaps of roughly equal width between them | **M11**: a fused chain measured **3.31×** against a **2.00×** traffic prediction at 4 MB arrays, because the unfused version paid ~10 µs of launch cost per kernel and bandwidth was never the limit |
| **Implicit synchronization serializing everything** | Host row shows a long block inside one API call; device row is empty for the same interval; the pattern repeats per iteration | M2's `cudaDeviceSynchronize`; any pageable `cudaMemcpy` |
| **No transfer/compute overlap** | Copy row and kernel row strictly alternate, never coexist | Motivates M24/M26 |
| **CPU-bound gaps** | A long stretch with the device row empty and the CPU row full of application code | Exercise 2 of this module |
| **Tail effects** | The last few hundred microseconds of a kernel show only a handful of SMs busy | **M1**: 41 blocks cost **1.99×** of 40 |

### The launch cost on this machine

This is a Windows laptop on the WDDM driver model, and the per-launch host cost
is large. Measured three different ways while authoring this module:

| Method | Result |
|---|---|
| Example 2: floor of the async launch-rate curve as kernel duration → 0 | **13.86 µs** (that session) |
| A dedicated null-kernel probe, grids of 1 to 1024 blocks | **7.9–9.2 µs**, flat in grid size |
| `nsys` `cuda_api_sum`, `cudaLaunchKernel` average over 81 080 calls | **13.51 µs** |
| `nsys` `cuda_api_sum`, `cudaLaunchKernel` average over 1 600 calls (Exercise 3) | **8.17 µs** |

So: **8–14 µs per launch**, depending on session and on how deep the queue is.
Module 11's "~10 µs" sits in the middle of that band and is the number to carry
around. The consequence is blunt:

> **A kernel shorter than ~10 µs costs more to start than to run.** Exercise 3
> measures a four-stage pipeline whose average kernel is **3.11 µs** against a
> **6.39 µs** launch floor, and finds the GPU idle 51% of the time. No amount of
> kernel optimization touches that 51%.

---

## Hardware Mental Model

### What produces the rows on the timeline

Recall Module 1's six-step launch path. Each step is a row, or part of one, in
`nsys`:

1. **CPU thread row.** Your `main()`, your NVTX ranges, and the OS runtime
   (`--trace=osrt` adds thread states, mutex waits, `nanosleep`, file I/O).
2. **CUDA API row.** One box per runtime/driver call, on the thread that made it.
   The box's *width is host time spent inside the call*, which for an
   asynchronous call is small and for a blocking call is however long the host
   waited. `cudaLaunchKernel` appears here; so does `cudaMemcpy`, and the fact
   that a 4-byte `cudaMemcpy` box is 118 µs wide is the entire diagnosis in
   Exercise 1.
3. **Kernel row(s).** One box per kernel *execution*, timestamped by the device.
   This is ground truth for kernel duration. It has nothing to do with how long
   `cudaLaunchKernel` took.
4. **Memory row(s).** H2D, D2H, D2D, memset — again device-timestamped, on the
   copy engine.
5. **Stream rows.** One lane per CUDA stream. With one stream everything
   serializes into one lane, which is why Module 24 exists.

The correlation arrows between the API row and the kernel row are the thing a
GUI gives you and the CLI does not: they connect a `cudaLaunchKernel` box at
time *t* to the kernel box at time *t + queue delay*. That delay is the gap.

### Why the launch costs 10 µs and why it is a *host* cost

From Module 1: a launch is not a jump. The runtime validates the configuration,
packs the arguments, writes a command into a pushbuffer, and — on WDDM — hands
that buffer to the OS kernel-mode driver, which schedules it onto the hardware
queue. The GPU's GigaThread engine then distributes blocks to SMs.

Two consequences:

- **The cost is paid on the CPU, in `cudaLaunchKernel`, and is nearly
  independent of grid size.** The probe above measured a flat 7.9–9.2 µs from 1
  block to 1024 blocks. You are not paying for blocks; you are paying for a
  driver round trip.
- **If the queue is non-empty, the cost is hidden.** The host enqueues work for
  iteration *k+1* while the device executes iteration *k*. The device never
  sees the 10 µs. **This is why "the host must be allowed to run ahead" is the
  single most important host-side rule in CUDA**, and why anything that drains
  the queue is expensive out of all proportion to what it appears to do.

### What drains the queue

A call is *synchronous with respect to the host* if the host cannot proceed past
it until the device has caught up. Each such call costs you one launch bubble —
the device goes idle, the host restarts the pipeline from empty.

| Call | Drains the queue? |
|---|---|
| `kernel<<<...>>>` | No |
| `cudaMemcpyAsync` on a stream, **pinned** host memory | No |
| `cudaMemcpy`, **pageable** host memory, any direction | **Yes** — the driver must stage through a pinned bounce buffer, and the call does not return until the copy is done |
| `cudaMemcpyToSymbol` from pageable memory | **Yes** — it is a `cudaMemcpy` with a different way of naming the destination |
| `cudaMemset` on device memory | No (asynchronous with respect to the host in the default stream) |
| `cudaMemcpy` device-to-device | No |
| `cudaMalloc` / `cudaFree` | **Yes** — they talk to the driver and synchronize the device |
| `cudaDeviceSynchronize` | **Yes**, by definition |

Two of those surprise people, and both are in this module's exercises. A 4-byte
`cudaMemcpy` is not cheap because it is 4 bytes — the bytes are free and the
synchronization is the whole cost. And `cudaFree` is not bookkeeping: Exercise
1's capture shows **301 `cudaFree` calls totalling 74.5 ms, 247 µs each**, which
is 25× the cost of a launch.

### Why the device's own clock is the only honest source of kernel duration

`nsys` reads kernel start/end timestamps from the device. That is a different
mechanism from `cudaEventElapsedTime`, which measures the distance between two
*enqueued markers*. When the queue is full the two agree; when the queue is
empty the marker retires immediately and the host's 10 µs launch lands inside
the measured interval.

Exercise 3 makes this failure total rather than partial. Its four stages carry
rep counts 330 / 85 / 85 / 85 — a 3.9× spread in arithmetic. An in-program
back-to-back async timing loop reports:

```
event-measured stage times (rep  330/85/85/85):
                                          9.17    6.62    8.00    8.11 us
```

Four nearly equal numbers, for kernels that differ by 3.9×. The instrument is
reporting its own floor. `nsys` on the same binary:

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Name
     43.2          2144342        400    5360.9   stage0
     18.9           937916        400    2344.8   stage1
     19.0           945433        400    2363.6   stage2
     18.9           941530        400    2353.8   stage3
```

5.36 / 2.34 / 2.36 / 2.35 µs. (The ratio is 2.29×, not 3.9×, because the three
short stages are partly floor too — even on-device, a kernel this size has a
~2.3 µs ramp. That residual is a real hardware effect, not an instrument
artefact, and `nsys` is what lets you see the difference.)

---

## Code Walkthrough

### NVTX: making a timeline readable

An un-annotated timeline of a real application is unreadable. You get thousands
of identical `cudaLaunchKernel` boxes and no idea which phase of your program
they belong to. **NVTX** (NVIDIA Tools Extension) is a header-only API that
pushes named, nestable ranges onto a per-thread stack; `nsys` records them and
reports them in `nvtx_sum`, and the GUI draws them as a hierarchy above the API
row.

On CUDA 13.2 the include is:

```cpp
#include <nvtx3/nvToolsExt.h>
```

and **there is no library to link.** The nvtx3 distribution is header-only; the
old `-lnvToolsExt` is gone. (`nvtx3/nvtx3.hpp` is the C++ wrapper; the exercises
use the C API directly so that the RAII guard is something you write rather than
something you include.)

The core API is four calls:

```cpp
nvtxRangePushA("name");   // push onto this thread's range stack
nvtxRangePop();           // pop
nvtxRangeId_t id = nvtxRangeStartA("name");  // thread-independent range
nvtxRangeEnd(id);                            // ends it, possibly on another thread
nvtxNameOsThreadA(0, "sim-main");            // name this thread on the timeline
```

Push/pop is for scoped, nested phases on one thread. Start/end is for ranges
that begin on one thread and end on another (an async job, a producer/consumer
hand-off). Use push/pop unless you need the other.

**The highest-value six lines in this module** are in `example01.cu`:

```cpp
struct NvtxRange {
    explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
    ~NvtxRange()                         { nvtxRangePop(); }
    NvtxRange(const NvtxRange &)            = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
};
#define NVTX_CAT2(a, b) a##b
#define NVTX_CAT(a, b)  NVTX_CAT2(a, b)
#define NVTX_RANGE(name) NvtxRange NVTX_CAT(_nvtxScope, __LINE__)(name)
```

Three details, each of which has bitten somebody:

- **Tying push to the constructor and pop to the destructor** means an early
  `return`, a `break` or a thrown exception cannot leave the stack unbalanced.
  An unbalanced stack does not crash; it silently reparents every subsequent
  range, and the timeline still looks plausible. That is worse than no
  instrumentation.
- **The copy constructor is deleted** for the same reason: a copy would pop
  twice.
- **Two levels of macro indirection are required.** `a##b` pastes its arguments
  *before* `__LINE__` is expanded, so a one-level macro names every object
  `_nvtxScope__LINE__` and two ranges in one scope collide at compile time, in a
  message that mentions neither NVTX nor the line you wrote.

Granularity is a judgement call with a real cost. A range per iteration over 300
iterations is free. A range per kernel launch over 900 launches starts to show up
in the thing you are measuring.

### `example01.cu` — the busy fraction, two ways

The program is a direct O(N²) n-body step loop, 2048 bodies, 200 steps, three
kernels per step. It runs the same 600 launches twice:

**Version A**, written the way the loop gets written first:

```cpp
for (int s = 0; s < STEPS; ++s) {
    NVTX_RANGE("step");
    float4 *acc = nullptr;
    { NVTX_RANGE("alloc"); CHECK(cudaMalloc(&acc, sizeof(float4) * N)); }   // (1)
    ...
    CHECK(cudaMemset(dEnergy, 0, sizeof(float)));                           // (3)
    kineticEnergy<<<blocks, BLOCK>>>(vel, dEnergy, N);
    CHECK(cudaMemcpy(&hostEnergy, dEnergy, sizeof(float),
                     cudaMemcpyDeviceToHost));                              // (2)
    { NVTX_RANGE("free"); CHECK(cudaFree(acc)); }                           // (1)
}
```

Three realistic decisions, none of them typos: a scratch buffer allocated per
step because "the step might need it"; a per-step energy readback because the
diagnostic is useful; a per-step accumulator reset.

**Version B** changes none of the arithmetic, none of the kernels, and none of
the launch count. It hoists the allocation out of the loop, uses
`cudaMemsetAsync`, and reads the energy back once at the end.

Measured on the RTX 3500 Ada (healthy clock state):

```
version                         wall (ms)      GPU<=(ms)     busy<= %   idle>=(ms)
A malloc+sync per step             60.346         43.069        71.4%       17.276
B hoisted, async                   34.713         32.566        93.8%        2.146
speedup A->B               : 1.74x
```

And the same binary under `nsys`:

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)   StdDev (ns)   Name
     97.9        122056588        800  152570.7        1007.9   computeForces
      1.6          2004151        400    5010.4          16.0   kineticEnergy
      0.5           644191        400    1610.5          15.5   integrate

 Time (%)  Total Time (ns)  Instances   Avg (ns)    Range
     17.2         57068215          1  57068215.0   :A: malloc-and-sync-per-step
     11.9         39536303          1  39536303.0   :B: hoisted-and-async
```

Read it. `computeForces` ran 800 times — 400 in the warm-up, 200 in A, 200 in B —
at **152 570.7 ns ± 1007.9 ns**, a 0.7% spread. **A's kernels and B's kernels are
the same speed.** Device execution per step is 152570.7 + 1610.5 + 5010.4 =
159 191.6 ns, so 200 steps is 31.84 ms. Against the NVTX range durations:

| version | NVTX range | true GPU busy | event harness claimed |
|---|---|---|---|
| A | 57.068 ms | **55.8%** | ≤ 71.4% |
| B | 39.536 ms | **80.5%** | ≤ 93.8% |

The event harness was wrong in the right direction — it is an upper bound — and
it was *more* wrong about A, the version whose queue is empty most often. That
asymmetry is why the harness also reports A's apparent GPU time as 5–25% higher
than B's when the profiler says they are identical.

> **Run-to-run caution.** In a session where the laptop had been benchmarking for
> an hour, the same binary reported A/B at 238.1/206.2 ms and a 1.15× speedup,
> with `computeForces` at **951 µs** instead of 152 µs — a 6.2× clock collapse.
> The *ratios* and the *shape* survived; the absolute numbers did not. Spec §12.5c
> exists for this. Cool the part between runs.

### `example02.cu` — putting a number on the launch floor and the sync tax

No profiler needed; the method is pure reasoning about shape. Sweep a kernel's
duration *D* over two decades and time a loop of back-to-back launches:

- When *D* ≫ *L* (the per-launch host cost), the loop costs *D* per iteration.
- When *D* ≪ *L*, the loop costs *L* per iteration.

So **the curve flattens at the bottom, and the height of the floor is *L*
measured directly.** Timing each configuration twice — once with the host
running ahead (`MODE_ASYNC`), once with a `cudaDeviceSynchronize` after every
launch (`MODE_SYNC`) — gives the sync tax as the difference.

```
    work     async us      sync us     sync tax     D est us     busy %
       1       13.864       19.522        5.658     <= floor      71.0%
       4       14.430       19.762        5.331     <= floor      73.0%
      16       15.099       21.511        6.412     <= floor      70.2%
      64       16.842       31.044       14.201     <= floor      54.3%
     256       33.963       44.516       10.553      = async      76.3%
    1024      103.465      113.004        9.539      = async      91.6%
    4096      427.812      439.900       12.088      = async      97.3%
   16384     1588.710     1618.376       29.665      = async      98.2%

launch floor L (async cost of the shortest kernel) : 13.864 us
mean sync tax S (sync - async, across the sweep)   : 11.681 us
```

Two things to take from the table. **The async column is flat for the first four
rows** — a 64× increase in arithmetic changes the iteration time by 21%, because
the program is not executing your kernel, it is executing `cudaLaunchKernel`.
And **the sync tax is roughly constant in *D***: 5–14 µs whether the kernel runs
for 14 µs or 1.6 ms, because it is the cost of restarting an empty pipeline, not
a cost proportional to anything.

Profiling the same binary cross-validates from the other side:

```
 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Name
     10.1       1095356053      81080    13509.6   cudaLaunchKernel
```

**13 509.6 ns of host time per launch**, against the program's own 13 864 ns
floor. The program measured the launch cost from the GPU's side (the device went
idle) and `nsys` measured it from the host's side (the CPU was inside the call).
They agree to 2.6%.

### The `nsys` command lines you actually need

On this machine `nsys` is installed but **not on `PATH`**. Two versions are
present; use 2025.6.3.

```bash
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"

# capture + print all the summary tables
"$NSYS" profile --trace=cuda,nvtx -o out --force-overwrite=true --stats=true ./prog.exe

# re-print tables from an existing capture (no re-run)
"$NSYS" stats out.nsys-rep
"$NSYS" stats --report cuda_api_sum --report cuda_gpu_kern_sum out.nsys-rep
"$NSYS" stats --help-reports
```

| Option | What it does | When you need it |
|---|---|---|
| `--trace=cuda,nvtx` | trace CUDA API + kernels + copies, and NVTX ranges | always |
| `--trace=cuda,nvtx,osrt` | adds OS runtime: thread states, mutexes, blocking syscalls | when the gap is CPU-side and you do not know which CPU call |
| `--stats=true` | print the summary tables after collection | always, on the CLI |
| `-o name` / `--force-overwrite=true` | output file; allow overwrite | always |
| `--capture-range=cudaProfilerApi` | collect only between `cudaProfilerStart()` and `cudaProfilerStop()` | **almost always** — see below |
| `--cuda-memory-usage=true` | track `cudaMalloc`/`cudaFree` and report allocation footprint over time | when you suspect allocator churn |

**`--capture-range=cudaProfilerApi` is the option that makes CLI profiling
usable.** Without it, your warm-up's 2000 launches and the 100 ms first
`cudaMalloc` land in the same tables as the 1800 launches you care about, and
every average is meaningless. With it, you bracket the region in the source:

```cpp
#include <cuda_profiler_api.h>
...
CHECK(cudaProfilerStart());
   /* the region you care about */
CHECK(cudaProfilerStop());
```

and every table describes exactly that region. All three exercises use it.

### Reading the tables

| Report | Rows | What the row means |
|---|---|---|
| `cuda_api_sum` | one per API function | **host** time spent inside the call. A wide `cudaMemcpy` row is the host *waiting*, not bytes moving. |
| `cuda_gpu_kern_sum` | one per kernel | **device** execution time. Ground truth. `Instances` is the launch count. |
| `cuda_gpu_mem_time_sum` | H2D / D2H / D2D / memset | **device**-side transfer time |
| `cuda_gpu_mem_size_sum` | same | bytes, in SI MB (10⁶) |
| `cuda_kern_exec_sum` | one per (API, kernel) pair | `AAvg` = API duration, `QAvg` = queue delay, `KAvg` = kernel duration — the gap, decomposed |
| `nvtx_sum` | one per NVTX range name | your phase names, with totals and instance counts |
| `nvtx_gpu_proj_sum` | NVTX ranges projected onto the GPU | which phase the device work belongs to |

One trap worth naming: **`cuda_api_sum` contains a `cudaProfilerStart` row whose
"Total Time" is the duration of the capture range itself**, not a cost. Ignore it
when looking for the dominant API.

### Computing the busy fraction from the CLI

```
busy = (sum of Total Time over cuda_gpu_kern_sum rows)
       / (Total Time of the enclosing NVTX range in nvtx_sum)
```

Add `cuda_gpu_mem_time_sum` to the numerator if you want to count copies as
"the GPU doing something". Report both: a program at 2.7% kernel-busy and 36%
including copies has a different problem from one at 2.7% both ways. Exercise 2
is exactly that case.

### The GUI

`nsys-ui` opens a `.nsys-rep` and gives you the correlation arrows, the
zoomable timeline, and the per-thread stack of NVTX ranges. It is the better
tool for *finding* a pattern you have not yet named. It is the worse tool for
*recording* one, which is why this module is CLI-centred: every number above was
produced by a command that can be re-run and checked.

---

## Check Your Understanding

1. A colleague profiles a program, finds `cuda_gpu_kern_sum` reports 980 ms of
   kernel time over a 1000 ms NVTX range, and concludes the program is
   GPU-bound and the host is fine. Describe two *materially different*
   situations in which that conclusion is wrong, and say which `nsys` report
   would distinguish them.

2. An iteration of a loop does: `cudaMemcpyToSymbol` (132 bytes, pageable),
   `kernelA<<<>>>`, `kernelB<<<>>>`, `cudaMemcpy` 4 bytes D2H,
   `cudaDeviceSynchronize()`. In `cuda_api_sum`, the `cudaDeviceSynchronize`
   row totals 1.15 ms over 300 calls while `cudaMemcpy` totals 287 ms over 600.
   A reader concludes the `cudaDeviceSynchronize` is harmless and removes only
   the `cudaMemcpy`. Why is that conclusion unsound, and what will the next
   profile look like?

3. You bracket a kernel launch with `cudaEventRecord` on both sides and get
   40 µs. `nsys` says the kernel ran for 9 µs. Both are correct. Explain the
   mechanism precisely enough to predict what the event pair would report if you
   enqueued 50 of those launches before synchronizing, and why.

4. Example 1 reports version B at 93.8% busy by its own event harness, and
   `nsys` says 80.5%. Example 2's `runFast` equivalent in Exercise 2 reports a
   busy fraction *above* 100% when computed the same way. One of those is a
   real measurement artefact with a specific cause and the other is a different
   one. Identify both.

Answers in `solutions/module22/check_your_understanding.md`.

---

## Exercises

All three require a real `nsys` capture. On this machine:

```bash
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
```

### Exercise 1 — instrument it, profile it, then fix it

`module22/exercise01.cu`. A 300-step Jacobi-style solver on 1 Mi cells, written
the way such loops get written first. It is correct. It is also idle most of the
time.

```bash
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
./exercise01.exe
"$NSYS" profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
     -o ex01 --stats=true --force-overwrite=true ./exercise01.exe
```

Five TODOs:

1. The NVTX RAII guard and the `NVTX_RANGE` macro, from scratch.
2. Annotate `runNaive()` so `nvtx_sum` alone localizes the cost.
3. Open and close a `cudaProfilerApi` capture range around the measured region.
4. **(design)** Write `runFixed()`: same 900 launches, same arithmetic, same
   final state, without the host-side idle. Four host-side decisions are in
   play — the per-step `cudaMalloc`/`cudaFree`, the per-step blocking 4-byte
   `cudaMemcpy`, the per-step `cudaMemset`, and the per-step device-to-device
   copy of the whole array. **Exactly two of the four are already asynchronous
   with respect to the host.** Work out which two *before* you start, and check
   your answer against `cuda_api_sum`, not against intuition.
5. Three numbers from your capture: total device execution time, total
   `cudaLaunchKernel` time, total `cudaMemcpy` time.

Validation: final residual matches, final field matches elementwise, wall-time
speedup ≥ 3.0×, and three cross-checks on your `nsys` numbers that a fabricated
value cannot pass. 6 points; `OVERALL: PASS` needs all six.

### Exercise 2 — find the gap

`module22/exercise02.cu`. A 33-tap FIR over 1 Mi samples, 300 times, with a
per-iteration gain. The kernels are fine; the wall time is more than an order of
magnitude above the sum of their durations.

```bash
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
./exercise02.exe
"$NSYS" profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
     -o ex02 --stats=true --force-overwrite=true ./exercise02.exe
```

Five TODOs. Three independent causes; two are named for you, the third is not.
The third one **barely appears in `cuda_api_sum` for the slow version**, because
another serialization point is already paying its bill — you will only see it
after you fix the other two. TODO 5 is the design TODO: eliminate it while
keeping the bookkeeping it was doing (the harness counts).

Validation: bit-identical output, identical reported peak, the progress hook
still called 300 times, speedup ≥ 6.0×, plus three diagnosis numbers including
the kernel total, which the harness needs because the program has no way to
measure it. 7 points.

### Exercise 3 — the critical path of a multi-kernel pipeline

`module22/exercise03.cu`. A four-stage elementwise pipeline, 400 steps, where
stage 0 carries 3.9× the arithmetic of the other three. The program offers three
candidate optimizations and measures all of them.

```bash
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
./exercise03.exe
"$NSYS" profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
     -o ex03 --stats=true --force-overwrite=true ./exercise03.exe
```

Five TODOs. Three are **predictions you commit to before building anything**:
the per-launch host cost, which of the three changes wins, and how much the
winner buys (as a bucket). One is the design TODO: write the fused kernel and
`runFused`, with output required to be **bit-identical** — decide whether that is
even achievable before you write it, and work out why. The last asks for two
numbers from a capture, one of which (`cudaLaunchKernel` count) you can derive
from the source before profiling and should.

Validation: bit-identical fused output, fusion ≥ 1.20×, the three predictions,
the exact launch count, and a final gate asserting two inequalities from your
capture that no in-program instrument could have produced. 7 points.

---

## Prediction

Commit to these in writing before you run anything.

1. **Example 1.** Version A and version B enqueue the same 600 launches over the
   same data. Will the summed *kernel execution time* reported by
   `cuda_gpu_kern_sum` differ between them by more than 2%? Will the summed
   *event-pair* time reported by the program's own harness? If your two answers
   differ, say precisely which mechanism makes them differ.

2. **Exercise 1.** Of the four host-side decisions in `runNaive` — per-step
   `cudaMalloc`/`cudaFree`, per-step 4-byte blocking `cudaMemcpy`, per-step
   `cudaMemset` of a 4-byte accumulator, per-step 4 MB device-to-device copy —
   rank them by how much wall time they cost, highest first. Then predict which
   of the four will have the *largest number of bytes* attached to it. Those two
   rankings are not the same and the difference is the lesson.

3. **Exercise 3.** Stage 0 does 3.9× the arithmetic of stages 1–3. Before
   running: if you could make stage 0 execute in zero time, what fraction of the
   pipeline's wall time would you save? Give a number. Then give the number for
   making *all four* stages execute in zero time. If your second number is not
   close to 100%, you have already predicted the answer to TODO 2.
