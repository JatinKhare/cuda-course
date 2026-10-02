# Module 22 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```bash
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
./exercise02_solution.exe

NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
"$NSYS" profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
     -o ex02sol --stats=true --force-overwrite=true ./exercise02_solution.exe
```

---

## The three causes

All three live in three-line helper functions that are individually correct.
That is the point: none of them is a bug, and no code review of any one of them
in isolation would flag it.

```cpp
static void stageInput(float *dIn, const float *hIn) {            // cause 1
    CHECK(cudaMemcpy(dIn, hIn, sizeof(float) * N, cudaMemcpyHostToDevice));
}

static float readPeak(float *hStage, const float *dOut) {         // cause 2
    CHECK(cudaMemcpy(hStage, dOut, sizeof(float) * N, cudaMemcpyDeviceToHost));
    float m = 0.0f;
    for (int i = 0; i < N; ++i) { float v = fabsf(hStage[i]); if (v > m) m = v; }
    return m;
}

static void recordProgress(int iter, const float *dOut) {         // cause 3
    ++gProgressCalls;
    if (iter < 0) printf("%p\n", (const void *)dOut);
    CHECK(cudaDeviceSynchronize());
}
```

---

## TODO 1 — NVTX instrumentation

Same guard as Exercise 1. The useful granularity here is one range per iteration
plus one per helper call: `stage-input`, `fir`, `rescale`, `telemetry`,
`progress`. Five names over 300 iterations is 1500 ranges and costs nothing
measurable against a 625 ms run. `nvtx_sum` then names the culprit directly.

---

## TODO 3 — the redundant upload

```cpp
// FIX 3: the input is loop-invariant. One upload, before the timed region.
{ NVTX_RANGE("stage-once"); stageInput(dIn, hIn); }
```

`hIn` is filled once, before `main` enters the loop, and never written again.
`stageInput` is idempotent and therefore correct — and it moves **4 MB across
PCIe 300 times**:

```
 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):
 Total (MB)  Count  Avg (MB)   Operation
   1258.291    300     4.194   [CUDA memcpy Host-to-Device]
   1258.291    300     4.194   [CUDA memcpy Device-to-Host]
```

1.26 GB each way for a 4 MB input and a 4 MB output.

**But the bytes are only half of it.** `cudaMemcpy` from *pageable* host memory
is **synchronous with respect to the host**: the driver has to stage through an
internal pinned bounce buffer, and the call does not return until the copy has
completed. So each of those 300 uploads also drained the launch queue. The
device-side transfer time is 107.8 ms; the host-side cost of the same calls is
part of a 287.5 ms `cudaMemcpy` row.

**Common wrong approach:** "it's only 4 MB, PCIe does 12 GB/s, that's 350 µs, so
it can't be the problem." It is 350 µs *per iteration*, 300 times, and that is
105 ms of the 625 ms — before counting the synchronization.

---

## TODO 4 — the per-iteration readback

```cpp
// FIX 4: the telemetry the program actually consumes is the LAST peak.
*peakOut = readPeak(hStage, dOut);      // after the loop, once
```

`readPeak` does two expensive things and the expensive one is not obvious:

- a blocking 4 MB device-to-host copy — again pageable, again a full drain,
  344 µs of device transfer time each;
- a **1 Mi-element host scan**. That is pure CPU time during which the device has
  nothing queued. It is the largest single term in the slow version and it is
  invisible in every CUDA table — it shows up only as a gap on the timeline and
  in the CPU thread row.

The program calls `readPeak` 300 times and prints the result once.

**Common wrong approach:** moving the copy out but keeping a per-iteration host
scan of a stale buffer. The harness checks `peakS == peakF` exactly, and a stale
buffer gives the wrong peak.

---

## TODO 5 (design) — the one that is not named

```cpp
// FIX 5: the bookkeeping still happens, exactly ITERS times. Only the
// cudaDeviceSynchronize() buried inside recordProgress() is gone.
++gProgressCalls;
```

`recordProgress(it, dOut)` reads, at the call site, as pure host accounting. The
`cudaDeviceSynchronize()` is three lines away inside a helper nobody profiles.

### Why it barely shows up in the slow version's profile

This is the subtlest thing in the module and the reason TODO 5 says *"you will
only see it after you fix the other two."*

```
 ** CUDA API Summary (cuda_api_sum):   [capture range = runSlow only]
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Name
     79.2       1130088280          1  1130088280.0   cudaProfilerStart   <- capture duration
     20.1        287507605        600      479179.3   cudaMemcpy
      0.6          8662203        600       14437.0   cudaLaunchKernel
      0.1          1150579        300        3835.3   cudaDeviceSynchronize
```

`cudaDeviceSynchronize` totals **1.15 ms over 300 calls — 3.8 µs each**, 0.1% of
the capture. It looks harmless. It is not harmless; it is *pre-paid*. The
`readPeak` copy immediately before it has already drained the queue, so by the
time `cudaDeviceSynchronize` is called there is nothing left to wait for and it
returns almost instantly.

**Serialization costs in series mask each other. You cannot rank causes by their
`cuda_api_sum` totals when the calls are in sequence.** The only way to price the
third cause is to remove the other two and measure again:

| runFast variant | µs / iteration | cost of the hidden sync |
|---|---|---|
| fixes 3 and 4 only, `recordProgress` kept | **61.7** | — |
| fixes 3, 4 and 5 | **51.7** | **1.19×, ≈ 10 µs per iteration** |

Ten microseconds per iteration is exactly one launch bubble on this machine
(measured launch floor 8–14 µs). That is what a `cudaDeviceSynchronize` costs
when the queue is otherwise full: it empties the pipeline and the host has to
refill it.

**Common wrong approach:** deleting the `recordProgress` call. The harness
requires `gProgressCalls` to advance exactly `ITERS` times inside `runFast`;
deleting the hook changes the program's behaviour rather than its performance.
The fix is to keep the work and drop the synchronization.

---

## TODO 2 — the diagnosis numbers

```cpp
#define DIAG_TOP_API     2            // cudaMemcpy
#define DIAG_H2D_MB   1258.291        // 300 x 4.194 MB
#define DIAG_KERNEL_NS  16653926.0    // fir + rescale
```

```
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Name
     85.0         14153319        300   47177.7   fir(const float *, float *, int)
     15.0          2500607        300    8335.4   rescale(float *, int, float)

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):
 Time (%)  Total Time (ns)  Count  Avg (ns)    Operation
     51.1        107774181    300   359247.3    [CUDA memcpy Host-to-Device]
     48.9        103256562    300   344188.5    [CUDA memcpy Device-to-Host]
```

**`DIAG_TOP_API` is `cudaMemcpy`, not `cudaDeviceSynchronize`.** Most people
guess the sync, because the sync is the thing that "obviously" blocks. The
blocking pageable copies block first and for longer, and they do it 600 times
instead of 300.

The kernel total, 16.65 ms, is the number the program cannot obtain about
itself — which is why the harness asks you for it and then uses it:

```
    GPU busy, runSlow = 2.7%   (16.654 ms of 625.404 ms)
    GPU busy, runFast = 107.3% (16.654 ms of 15.518 ms)
```

**2.7%.** Including the device-side transfer time (211 ms) it is 36%, and that
difference is worth reporting too: a program at 2.7% kernel-busy / 36% including
copies has a transfer problem *and* a gap problem.

The 107.3% is not an error. The kernel total comes from a *profiled* capture and
the wall time from an un-profiled run; the tracer adds a little to every kernel.
It means `runFast` is GPU-bound, which was the goal.

---

## Synchronization / memory reasoning

Per iteration, the slow version:

```
stageInput   : 4 MB H2D, pageable  -> drains the queue, ~359 us of DMA
fir          : async
rescale      : async
readPeak     : 4 MB D2H, pageable  -> drains the queue, ~344 us of DMA
             : 1 Mi-element host scan, ~1.3 ms of pure CPU
recordProgress: cudaDeviceSynchronize -> free, because nothing is queued
```

Device execution: 55.5 µs. Wall: 2085 µs. The GPU is idle for 97% of it, and the
idleness has three distinguishable shapes on the timeline — a copy bar with the
kernel row empty (transfer, no overlap), a long flat stretch with the CPU row
full (host-bound), and a thin gap after each (the launch bubble). All three are
in the module's table of classic findings and all three are in this one loop.

What *would* fix the transfers properly is pinned host memory plus
`cudaMemcpyAsync` on a non-default stream, so that the upload for iteration
*k+1* overlaps the kernel of iteration *k*. That is Modules 24 and 26. This
exercise deliberately solves the problem the cheaper way — by noticing that
neither transfer needed to be in the loop at all. **Removing work beats
overlapping work.**

---

## Performance reasoning

```
version       wall (ms)      us / iter
slow            625.404         2084.7
fast             15.518           51.7
speedup  : 40.30x
```

Across runs the slow version lands at 625–805 ms and the speedup at **40–51×**.
The spread comes almost entirely from the host scan in `readPeak`, which is
sensitive to whatever else the laptop is doing.

Apportionment, from the capture:

| term | per iteration | share of 2085 µs |
|---|---|---|
| host scan in `readPeak` (CPU) | ~1300 µs | 62% |
| 4 MB H2D DMA | 359 µs | 17% |
| 4 MB D2H DMA | 344 µs | 17% |
| `fir` + `rescale` on the device | 55.5 µs | **2.7%** |
| `cudaLaunchKernel` host time | 28.9 µs | 1.4% |
| hidden `cudaDeviceSynchronize` | ~10 µs (unmasked) | 0.5% |

That ordering is the lesson. The thing the program is *for* — filtering a signal
— is 2.7% of its runtime.

---

## Expected output

```
Module 22 / Exercise 2 — find the gap
N = 1048576 samples, 33 taps, 300 iterations, 600 launches per version

version       wall (ms)      us / iter
slow            625.404         2084.7
fast             15.518           51.7
speedup  : 40.30x

[x] 1. filtered signal matches bit-for-bit (0/1048576 differ)
[x] 2. reported peak matches   (0.8007795 vs 0.8007795)
[x] 3. progress hook still called 300 times in runFast (600 total)
[x] 4. speedup >= 6.0x         (got 40.30x)
[x] 5. top cuda_api_sum row identified
[x] 6. H2D traffic in capture  (you said 1258.3 MB, expected 1258.3)
[x] 7. kernel time plausible   (16.654 ms; runFast wall 15.518 ms)

    GPU busy, runSlow = 2.7%   (16.654 ms of 625.404 ms)
    GPU busy, runFast = 107.3%   (16.654 ms of 15.518 ms)
    (runFast can read slightly over 100%: the kernel total
     comes from a PROFILED capture and the wall time does not.
     It means runFast is GPU-bound, which is the goal.)
    The kernels never changed. Only the host did.

SCORE: 7/7
OVERALL: PASS
```

The gate is set at 6.0× rather than near the measured 40× deliberately: the slow
version's dominant term is a host-side scan whose cost depends on the machine's
other load, and spec §12.5d forbids scoring a distinction narrower than the
measured spread.

---

## The result that matters

**A serialization cost can be invisible in the profile because another one is
already paying its bill.** The hidden `cudaDeviceSynchronize()` in this program
shows up in `cuda_api_sum` as 0.1% of the capture — and it is worth 1.19× once
the two copies in front of it are gone. Costs in series mask each other, so a
ranked `cuda_api_sum` table tells you what to fix *first*, never what to fix
*last*. Profile again after every fix; the table is a different table now.

**Variation to try:** keep the per-iteration `readPeak` but make it read back a
single float computed on the device instead of 4 MB scanned on the host. The
transfer and the CPU scan vanish, the drain does not. Predict the new speedup
before you measure it, then check whether you have landed on the launch floor.
