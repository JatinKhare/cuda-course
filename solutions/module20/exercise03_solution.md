# Module 20 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe

nvcc -arch=sm_89 -O3 -cubin -o x3.cubin exercise03_solution.cu
cuobjdump -sass x3.cubin
```

---

## TODO 1 — supplied concurrency

```cpp
static double suppliedConcurrencyBytes(int threadsPerBlock, int blocksPerSM,
                                       int loadsPerThread, int bytesPerLane)
{
    return (double)threadsPerBlock * blocksPerSM * SM_COUNT
         * loadsPerThread * bytesPerLane;
}
```

**Why bytes.** The DRAM controller is rate-limited in bytes per second, so the
only unit in which Little's Law closes is bytes. Counting *requests* is the
mistake that makes a `float` load and a `float4` load look equivalent when they
differ by 4× in exactly the quantity that matters. Counting *warps* works too,
provided you then multiply by 32 lanes and by the bytes per lane — which is the
same arithmetic.

For the baseline: `128 × 1 × 40 × 1 × 4 = 20,480 B = 20.0 KB`.

## TODO 2 — effective latency

```cpp
static double effectiveLatencyNs(double Qbytes, double B_GBs)
{
    return Qbytes / (B_GBs * 1e9) * 1e9;
}
```

`concurrency = throughput × latency`, so `latency = concurrency / throughput`.
This is only meaningful while the system is latency-bound — which the baseline
certainly is, at 15.8% of pin peak. Applied to the measured 68.3 GB/s:

```
L = 20480 B / 68.3e9 B/s = 299.7 ns = 569 cycles at 1.90 GHz
```

**Module 4 measured the DRAM dependent-load latency as 575 cycles, by an
entirely different experiment (a randomised pointer chase on one thread).**
This exercise recovers it from a *bandwidth* measurement on 5120 threads, and
the two agree to 1%. That agreement is the point of the exercise.

## TODO 3 — required concurrency

```cpp
static double requiredConcurrencyBytes(double B_GBs, double L_ns)
{
    return B_GBs * 1e9 * (L_ns * 1e-9);
}
```

With the measured ceiling of 411.6 GB/s and `L = 299.7 ns`:
**123,335 B = 120.4 KB**, against the baseline's 20.0 KB. **Shortfall 6.02×.**

The check that closes the loop: `Q0 / L = 20480 / 299.7 ns = 68.3 GB/s`, which
is the measured baseline bandwidth to three significant figures. The model is
not fitted; it is the same equation used in both directions.

## TODO 4 — the fixed kernel

```cpp
#define FIX_MLP            4
#define FIX_BYTES_PER_LANE 16
__global__ void kFixed(const float *__restrict__ x, float *part, unsigned n)
{
    const unsigned stride4 = gridDim.x * blockDim.x;
    const unsigned base4   = blockIdx.x*blockDim.x + threadIdx.x;
    const float4 *x4 = (const float4*)x;
    const unsigned n4 = n >> 2;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (unsigned i = base4; i + (FIX_MLP-1)*stride4 < n4; i += FIX_MLP*stride4) {
        float4 v[FIX_MLP];
        #pragma unroll
        for (int c = 0; c < FIX_MLP; ++c) v[c] = x4[i + c*stride4];
        #pragma unroll
        for (int c = 0; c < FIX_MLP; ++c) {
            acc.x += v[c].x; acc.y += v[c].y; acc.z += v[c].z; acc.w += v[c].w;
        }
    }
    part[base4] = (acc.x + acc.y) + (acc.z + acc.w);
}
```

**Two multipliers, and they compose.**

1. **MLP.** Four loads issued before any is consumed: 4×.
2. **Width.** A `float4` load occupies **one** outstanding-request slot and
   carries 16 bytes per lane instead of 4: another 4×.

Supplied concurrency `128 × 1 × 40 × 4 × 16 = 327,680 B = 320 KB = 2.66 × Q*`.
Comfortably past the requirement, which is the right place to be — the knee is
not sharp and overshooting costs four registers.

**Neither multiplier alone is enough.** Each on its own gives
`128 × 40 × 4 × 4 = 81,920 B = 80 KB = 0.66 × Q*`, and Little's Law then
predicts about 66% of the ceiling — Example 2's `(1 warp/scheduler, C = 1)`
cell, which is `float4` with MLP 1, measures **252.5 GB/s = 61%** of its 411.9
ceiling. Both would fail the 95% gate. Getting there with MLP alone needs
`Q*/(5120 × 4 B) = 6.02`, i.e. 7 scalar loads in flight per thread; that also
works and costs three more registers than the `float4` route for the same
concurrency.

**Why the loop body is two loops and not one.** Fusing the issue loop and the
consume loop reinstates a dependence between load `c` and load `c+1`'s issue
slot and collapses `FIX_MLP` to 1. This is Module 11's `#pragma unroll 1`
experiment in reverse.

**The tail.** `NELEM = 5120 × 32 × 800`, so `n4 = NELEM/4` is divisible by
`FIX_MLP × stride4 = 4 × 5120` exactly, and there is no tail. That is a
property of the constants in this file, deliberately chosen so that the
exercise is about concurrency and not about edge cases; do not copy the pattern
without checking.

**Common wrong approaches.**

- *Keeping `#pragma unroll 1` and adding `float4`.* 4× instead of 6×; lands
  around 350–380 GB/s and fails the 95% gate.
- *Removing `#pragma unroll 1` and changing nothing else.* This works
  surprisingly well — `ptxas` will software-pipeline the loop and give you MLP
  2–4 for free — but it is not reliable and it is not something you chose.
  Measure it; then decide whether you want the compiler's unroll heuristic in
  your performance model.
- *Raising the thread count.* The gate exists precisely to forbid this, and the
  file says so. If you try it, the arithmetic in sections 1–3 still works and
  tells you how many threads you would need: `Q* / (1 load × 4 B) = 30,834`
  threads at MLP 1, i.e. 6× the budget. That is the same trade from the other
  side.
- *Dynamically-indexed `float4 v[FIX_MLP]`.* Local memory, Module 4, and the
  measurement becomes a measurement of DRAM round-tripping through the spill
  slots.

## TODO 5 — the prediction

```cpp
#define PRED_FIXED_GBS  410.0
```

Once supplied concurrency exceeds the requirement the kernel is no longer
latency-bound, so the predicted bandwidth is simply the streaming ceiling —
which Module 12 measured at **410.5–410.7 GB/s** after a 1500 ms warm-up and
this program re-measures at 411.6. Predicted 410.0, measured 409.5–409.7, scored to 12%. There is nothing clever
here, and that is the point: the moment you are past the knee, the prediction
stops depending on anything about your kernel.

**Why this file carries an operating-point guard.** TODO 5 is the only scored
quantity in Module 20 that is an *absolute* bandwidth rather than a ratio, and
spec §12 rule 5b requires any harness that scores a performance prediction to
guard its operating point. It was not a hypothetical: during verification, a
run started immediately after three other timed programs measured the fixed
kernel at **323.6 GB/s** — still **99.5% of the ceiling measured in the same
run**, so TODO 4 passed, but 21% below the healthy figure, so the correct
TODO 5 answer scored `FAIL`. `nvidia-smi` in that state shows the memory clock
pinned at 6001 MHz instead of 9001. The file now probes its own ceiling after
warming and, if it reads below 370 GB/s, idles 10 s and warms again, up to five
attempts, then warns and proceeds — the pattern Module 15 established.

## Synchronization / memory reasoning

No barriers, no shared memory, no cross-thread communication. Each thread owns
a disjoint strided subset of the array and writes one partial sum. The
grid-stride access pattern keeps every warp's 32 lanes contiguous, so each
`LDG.E.128` is 4 fully-used sectors (Module 5) — concurrency is the only thing
varying.

The buffer is 500 MB, **10.4× the 48 MB L2**, so nothing is cached between
launches and the measurement is of DRAM, per spec §12 rule 7.

## Performance reasoning

Full output, one run:

```
   buffer                        500.0 MB  (10.4x the 48 MB L2)
   baseline  <<<40,128>>>        7.6715 ms     68.3 GB/s   15.8% of pin peak
   your fix  <<<40,128>>>        1.2810 ms    409.3 GB/s   94.7% of pin peak
   ceiling probe (full occ)      1.2739 ms    411.6 GB/s   95.3% of pin peak

   supplied concurrency Q0         20480 B =   20.0 KB
     = 5120 threads x 1 load x 4 B/lane
   effective latency L             299.7 ns =   569 cycles at 1.90 GHz
   required concurrency Q*        123335 B =  120.4 KB
   shortfall Q*/Q0                  6.02x
   predicted baseline BW = Q0/L =    68.3 GB/s   (measured    68.3)

   your fixed kernel supplies     327680 B =  320.0 KB = 2.66x Q*

   baseline sum 1.965804e+08 vs 1.965760e+08, err/(gamma_K*S) 0.015, unwritten 0
   fixed    sum 1.965756e+08 vs 1.965760e+08, err/(gamma_K*S) 0.001, unwritten 0
```

**The fixed kernel reaches 99.4% of the ceiling probe**, which runs at 12
blocks/SM — twelve times the threads. 5120 threads with the right amount of
work in flight per thread are worth 61,440 threads with the wrong amount. That
sentence is the module.

Sanity checks against hardware bounds: 411.6 GB/s is 95.3% of the 432 GB/s pin
peak and cannot be exceeded; 569 cycles against Module 4's independently
measured 575; and the predicted baseline bandwidth reproduces the measured one
to 0.1%, which would be impossible if the latency-bound model were wrong.

Run-to-run: the baseline measured 68.1–68.4 GB/s and the fixed kernel
409.3–409.5 GB/s across three runs with cool-downs. The derived latency moved
between 568 and 571 cycles.

Note also the two validation lines. The **baseline's** reduction is
*less* accurate than the fixed kernel's (`err/(γ_K·S)` 0.015 vs 0.001), because
the baseline accumulates 25,600 terms into one `float` while the fixed kernel
keeps four partial sums in a `float4`. Breaking a dependency chain usually
*improves* the numerics as a side effect, for exactly the reason multiple
accumulators are a standard summation technique.

## Expected output

`SCORE: 10/10`, `OVERALL: PASS`.

## The result that matters

You can compute, before writing a line of kernel, how many bytes your launch
will have in flight and how many the memory system needs, and the ratio of the
two predicts the achieved bandwidth to within a few percent. The baseline here
is not "badly written" in any way a code review would catch — it is a clean
grid-stride reduction — it is simply 6× short of the machine's concurrency
requirement, and the fix is to pick up either multiplier. **Variation to try:**
set `FIX_MLP` to 1 and keep `float4`, then restore `FIX_MLP = 4` but go back to
scalar `float`. Both supply the same 80 KB and both should land near 66% of the
ceiling; predict the two numbers from Little's Law first, then measure, and
then explain any difference between them using the 15-cycles-per-extra-load
slope from Example 2 part C — the two configurations pay that slope a different
number of times for the same bytes.
