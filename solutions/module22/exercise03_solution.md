# Module 22 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```bash
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
./exercise03_solution.exe

NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
"$NSYS" profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
     -o ex03sol --stats=true --force-overwrite=true ./exercise03_solution.exe
```

---

## TODO 1 — `PRED_LAUNCH_US`

```cpp
#define PRED_LAUNCH_US      10.0
```

Scored to within a factor of 2 against the program's own measurement, which
lands at **5.8–11.3 µs** depending on session. The lesson gives the band from
four independent measurements (8–14 µs); Module 11's "~10 µs" is the number to
remember.

**Why 10 µs and not 1 µs or 100 ns.** A launch is a driver round trip, not a
jump. On WDDM the runtime packs arguments, writes a pushbuffer command, and hands
it to the kernel-mode driver, which schedules it onto a hardware queue. A
dedicated probe in this module measured the cost **flat from 1 block to 1024
blocks** (7.9 → 9.2 µs), confirming you are paying for the round trip and not
for the work.

**Common wrong approach:** reasoning from the GPU's clock. A 2 GHz SM executes
20 000 cycles in 10 µs; nothing about the launch is a device-side cost, so
device-side reasoning gives an answer three orders of magnitude too small.

---

## TODO 2 — `PRED_BEST`

```cpp
#define PRED_BEST              3     // fuse
```

Measured:

```
configuration               us / step      vs base
staged baseline                 25.63        1.00x
halve longest kernel            30.59        0.84x
halve all four                  29.84        0.86x
fuse into one launch            10.84        2.36x
```

Halving the arithmetic — *all* of it, not just stage 0 — does nothing, and in
this run both "optimizations" measured slightly *slower* than the baseline. That
is not a real regression; it is sub-launch-floor noise, and it is exactly what
you should expect from changing a quantity that is not on the critical path.
Repeated runs put both in the 0.84–1.11× band around 1.00×.

### The arithmetic that gives the answer before you measure

From the profile: the four stages take **5.36 / 2.34 / 2.36 / 2.35 µs**, so
12.4 µs of device time per step. The launch floor is ~8 µs and there are four
launches per step, so the host needs ~32 µs per step to issue the work. The host
is the limit; 12.4 µs of device work fits inside it with room to spare.

- *Make stage 0 free:* saves 5.36 µs of device time that was already hidden
  under the host's 32 µs. Saves nothing.
- *Make all four free:* same, 12.4 µs. Saves nothing.
- *Fuse:* four launches become one. The host cost per step falls from ~32 µs to
  ~8 µs, and the device work (now 12.4 µs in one kernel, plus three fewer
  global round trips for the intermediates) becomes the limit. **That is the
  only change that touches the critical path.**

**Common wrong approach:** picking 2, on the grounds that it strictly dominates
1 and is therefore the "most powerful" change. It is the most powerful change *to
a quantity that is not the bottleneck*. This is Amdahl's law applied to the
timeline rather than to the code: the fraction you can affect is the 48% the GPU
was executing, and only when it is the limiting resource.

---

## TODO 3 — `PRED_FUSED_BUCKET`

```cpp
#define PRED_FUSED_BUCKET      3     // 1.8x .. 3.5x
```

Measured **2.14–2.56×** over repeated runs. Buckets are 1.2 / 1.8 / 3.5, placed
so both relevant edges have ≥ 0.3× of margin on the measured spread
(spec §12.5d).

Why not 4× — the naive "four launches become one" argument? Because the fused
kernel still has to do the work. Per step: staged = 4 launches (~32 µs host,
limiting) ; fused = max(1 launch ≈ 8 µs, 12.4 µs of device work + the removed
intermediate traffic) ≈ 11 µs. 25.6 / 10.8 = 2.4. **Fusion converts a
host-limited loop into a device-limited one; it cannot make it faster than the
device.**

---

## TODO 4 (design) — the fused kernel

```cpp
__global__ void fused(const float *__restrict__ in, float *__restrict__ out,
                      int n, int r0, int r1, int r2, int r3,
                      float k0, float k1, float k2, float k3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    x = chain(x, r0, k0);
    x = chain(x, r1, k1);
    x = chain(x, r2, k2);
    x = chain(x, r3, k3);
    out[i] = x;
}
```

```cpp
static double runFused(const float *in, float *out, cudaEvent_t e0, cudaEvent_t e1)
{
    NVTX_RANGE("fused");
    CHECK(cudaEventRecord(e0));
    for (int s = 0; s < STEPS; ++s)
        fused<<<gBlocks, BLOCK>>>(in, out, N, R0, R1, R2, R3, K0, K1, K2, K3);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return 1000.0 * (double)ms / (double)STEPS;
}
```

### Why bit-identical output is achievable

The question the exercise asks you to settle before writing the code. Fusing two
kernels generally does *not* preserve bits — if the intermediate was held in
higher precision, or if the compiler contracts a multiply and an add across the
boundary, the result changes.

Here it is safe, for a specific reason: **every intermediate in the staged
version was already a `float` when it was written to global memory.** `out[i] =
chain(in[i], ...)` stores an fp32 value; the next stage loads exactly that fp32
value. Keeping it in a register instead of round-tripping through DRAM performs
the identical sequence of fp32 operations in the identical order. A `float`
stored and reloaded is bit-exact.

What would break it: making `chain` accumulate in `double` in the fused version
"because we can now"; reordering the stages; or letting the compiler fuse an
`fmaf` across what used to be a kernel boundary. None happen here because `chain`
is already written with explicit `fmaf` and is called in the same order.

Measured: `0/65536 differ`.

**Common wrong approaches.**

- *Fusing by concatenating the rep counts* (`chain(x, r0+r1+r2+r3, k0)`). Faster,
  wrong: each stage has its own `k`.
- *Writing the intermediates to `t0/t1/t2` from inside the fused kernel so the
  code "looks the same".* That restores the global traffic the fusion was
  removing, and it introduces a cross-block ordering requirement the launch
  boundary used to provide for free. (The reason kernel boundaries are the
  standard global barrier in CUDA is Module 9's; nothing inside a grid orders
  blocks against each other.)
- *Fusing with a grid-stride loop and a smaller grid.* Legitimate, and it changes
  the comparison; keep the launch configuration identical if you want the
  measurement to isolate the launch count.

---

## TODO 5 — the profile

The capture range brackets the **baseline staged configuration only**, so the
launch count is derivable from the source: `4 * STEPS = 1600`.

```
 ** CUDA API Summary (cuda_api_sum):
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Name
     98.9       1195858350          1  1195858350.0   cudaProfilerStart  <- capture duration
      1.1         13074439       1600       8171.5    cudaLaunchKernel
      0.0           171451       1600        107.2    cuKernelGetName
      0.0            75975          2      37987.5    cudaEventRecord

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Name
     43.2          2144342        400    5360.9    stage0(const float *, float *, int, int, float)
     19.0           945433        400    2363.6    stage2(const float *, float *, int, int, float)
     18.9           941530        400    2353.8    stage3(const float *, float *, int, int, float)
     18.9           937916        400    2344.8    stage1(const float *, float *, int, int, float)
```

```cpp
#define NS_LAUNCH_COUNT     1600
#define NS_KERNEL_NS     4969221.0     // 2144342 + 945433 + 941530 + 937916
```

The harness turns those into the two inequalities the exercise exists to produce:

```
[x] 7. nsys kernel time: 3.11 us/launch vs 6.39 us launch floor;
       GPU busy 48.5% of the staged run (4.97 ms of 10.25 ms)
```

> **The average kernel in this pipeline is shorter than the call that launches
> it.**

---

## The instrument failing, on purpose

The program also times each stage with a back-to-back async loop — the only
method available to a program measuring itself — and prints:

```
measured launch floor L                 :    6.39 us / launch
event-measured stage times (rep  330/85/85/85):
                                             9.17    6.62    8.00    8.11 us
```

Four nearly equal numbers for kernels whose rep counts differ by 3.9×. The loop
is not measuring the kernels; it is measuring `cudaLaunchKernel`. Against the
`nsys` ground truth of 5.36 / 2.34 / 2.36 / 2.35 µs, every one of the four
in-program numbers is wrong by 1.7–3.5×, and the *ordering* is wrong too.

There is no way to fix this from inside the program. `cudaEventRecord` enqueues a
marker; when the queue is empty the marker retires immediately and the host's
launch call lands inside the measured interval. Only a tracer that reads the
device's own timestamps can separate them.

### A real hardware effect the profile also exposes

`nsys` says stage 0 is **2.29×** stage 1, not the 3.88× its rep count implies. That
gap is not an artefact: a kernel this small has a fixed on-device cost — grid
launch, block distribution by the GigaThread engine, the first memory round trip
— of roughly **1.4 µs**, and `(330·c + 1.4)/(85·c + 1.4) = 2.29` with
`c ≈ 0.012 µs/rep` reproduces both measured durations. Fitting that constant is
only possible because `nsys` gave four device-side durations at two different rep
counts. It is a good illustration of the division of labour: `nsys` found the
constant, and anything you want to say about *why* it is 1.4 µs is Module 23's
territory.

---

## Performance reasoning

| configuration | µs / step | vs baseline | what changed |
|---|---|---|---|
| staged baseline | 25.63 | 1.00× | — |
| halve stage 0 | 30.59 | 0.84× | −5.36 µs of hidden device work |
| halve all four | 29.84 | 0.86× | −6.2 µs of hidden device work |
| **fused** | **10.84** | **2.36×** | 4 launches → 1 |

GPU busy in the baseline: **48.5%**. The other 51.5% is 1600 launches × ~6.4 µs
of host time that the device cannot overlap because there are only four kernels
per step to hide it behind.

Why the baseline is 48% rather than the ~39% that `12.4 / 32` predicts: the host
and device do overlap partially — the host is issuing launch *k+1* while the
device runs kernel *k* — so the step time is not the sum of the two but something
between the larger of them and the sum. 25.6 µs against a 32 µs host estimate and
a 12.4 µs device estimate is consistent with the host being the limit and the
launch cost being nearer 6.4 µs than 8.2 µs in the un-instrumented run.

### What you would do if the kernels could not be fused

This pipeline fuses because the stages are elementwise and share a grid. Real
pipelines often cannot: a reduction between stages, a different launch shape, a
library call in the middle. The lever then is **CUDA graphs (Module 28)**, which
capture the whole sequence once and replay it with a single host-side submission,
removing per-launch cost without changing the kernels at all. Streams
(Module 24) are the other lever when the stages are independent rather than
chained — they are not, here: stage *k+1* reads what stage *k* wrote.

---

## Expected output

```
Module 22 / Exercise 3 — the critical path of a 4-kernel pipeline
N = 65536, 400 steps, 4 launches per step (staged) / 1 (fused)

measured launch floor L                 :    6.39 us / launch
event-measured stage times (rep  330/85/85/85):
                                             9.17    6.62    8.00    8.11 us
  ^ stage 0 carries 3.9x the arithmetic of the others. If these four
    numbers do not reflect that, the instrument is reporting its own
    floor, not the kernels. Only nsys can see through that.

configuration               us / step      vs base
staged baseline                 25.63        1.00x
halve longest kernel            30.59        0.84x
halve all four                  29.84        0.86x
fuse into one launch            10.84        2.36x

[x] 1. fused output is bit-identical to staged (0/65536 differ)
[x] 2. fusion beats the baseline by >= 1.20x (got 2.36x)
[x] 3. PRED_LAUNCH_US within 2x of measured (10.0 vs 6.39 us)
[x] 4. PRED_BEST matches the measurement (you said 3, winner 3)
[x] 5. PRED_FUSED_BUCKET correct (you said 3, measured 2.36x = bucket 3)
[x] 6. nsys cudaLaunchKernel count exact (you said 1600)
[x] 7. nsys kernel time: 3.11 us/launch vs 6.39 us launch floor;
       GPU busy 48.5% of the staged run (4.97 ms of 10.25 ms)

    The average kernel in this pipeline is SHORTER than the call
    that launches it. Halving any kernel -- or all of them --
    cannot touch the 52% of the time the GPU spends waiting.
    Fewer, larger launches is the only lever: fusion here
    (Module 11), or CUDA graphs (Module 28) when the kernels
    cannot be fused.

SCORE: 7/7
OVERALL: PASS
```

Run to run: `L` lands in 5.8–11.3 µs, the baseline in 25–37 µs/step, and the
fusion ratio in 2.1–2.4×. Gate 7's busy figure moves with `L` (14–49% observed)
and the gate accepts 5–60% for that reason.

---

## The result that matters

**Optimize the critical path, and on a timeline the critical path is often not
on the GPU at all.** This pipeline's longest kernel is 43% of its device time and
making it free is worth nothing, because the device was never the limit. The
quantity that decided the answer — 3.11 µs of device execution per launch against
a 6.39 µs launch cost — is one the program structurally cannot measure about
itself: its own event-based timing reported all four stages at ~8 µs, the launch
floor, in the wrong order. Predicting the winner correctly required knowing the
launch cost, and knowing the launch cost required a tracer.

**Variation to try:** raise `R0` to 3300 (10× the arithmetic in stage 0 only) and
re-run. The pipeline becomes device-limited, `PRED_BEST` flips from 3 to 1, and
the fusion ratio collapses towards 1.0×. The same four kernels, the same four
launches, a different answer — which is why you profile rather than apply a rule.
