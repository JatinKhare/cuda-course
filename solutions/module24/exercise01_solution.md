# Module 24 / Exercise 01 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -std=c++17 -o exercise01_solution.exe exercise01_solution.cu
exercise01_solution.exe
```

Warning-clean on CUDA 13.2 / sm_89. Run it on its own, with 20–30 s of idle
before it (spec §12.5c) — this file times eleven things and a back-to-back
batch will push the part into a different operating point.

---

## TODO 1 — `chunkOf`

```cpp
static void chunkOf(int k, int n, int nChunks, int* off, int* len)
{
    int c = (n + nChunks - 1) / nChunks;     /* round UP, never down */
    int o = k * c;
    if (o > n) o = n;
    int l = c;
    if (o + l > n) l = n - o;                /* ragged last chunk */
    if (l < 0) l = 0;
    *off = o;
    *len = l;
}
```

**Why this is correct.** `N = 12582917` is deliberately not a multiple of 16.
Rounding the chunk size *down* (`n / nChunks`) produces 16 chunks of 786432 and
leaves the last 5 elements unprocessed — and the output buffer is prefilled with
`-1.0f` precisely so that this shows up as `untouched != 0` rather than as a
silently truncated result. Rounding *up* and clamping the final chunk is the
only one-line partition that is total and non-overlapping.

The `if (o > n) o = n;` clamp matters for the degenerate case the harness
probes, `n = 5, nChunks = 8`: chunk 5 has `k*c = 5` which is fine, but chunks
6 and 7 would compute offsets 6 and 7, past the end. They must come back as
`(n, 0)` — a legitimately empty chunk — not as a negative length.

**Common wrong approaches.**

| Written | Symptom |
|---|---|
| `c = n / nChunks` | last 5 elements never written; `untouched = 1`, `last element: WRONG/unwritten` |
| `len = c` unconditionally | the final `cudaMemcpyAsync` reads 5 elements past `g_dOut`; usually silent, caught by `compute-sanitizer` |
| `off = k * (n / nChunks)`, `len = (k == nChunks-1) ? n - off : n/nChunks` | correct but the last chunk is 16× larger than the rest, which destroys the pipeline's balance and costs ~0.15× |

## TODO 2 — `streamFor`

```cpp
static int streamFor(int k, int nStreams) { return k % nStreams; }
```

The requirement is that **chunks that are in flight at the same time are in
different streams**. With a round-robin map, chunks `k` and `k+1` are always in
different streams, so chunk `k+1`'s `H2D` is free to start while chunk `k`'s
kernel runs. That is the whole purpose of having more than one stream here.

`k / (nChunks / nStreams)` — block assignment — also uses all four streams and
is wrong: chunks 0–3 all land in stream 0, so the first four chunks execute
strictly serially and the pipeline only starts filling at chunk 4.

Returning `0` (one stream for everything) is *correct* and measures 1.00×. It
is the configuration the exercise exists to beat.

## TODO 3 — the pipeline

```cpp
static void readerPipeline(const float* hIn, float* hOut, int nChunks)
{
    for (int k = 0; k < nChunks; ++k) {
        int off, len;
        chunkOf(k, g_n, nChunks, &off, &len);
        if (len <= 0) continue;
        cudaStream_t st = g_s[streamFor(k, N_STREAMS)];

        CHECK(cudaMemcpyAsync(g_dIn + off, hIn + off, (size_t)len * sizeof(float),
                              cudaMemcpyHostToDevice, st));
        condition<<<(len + 255) / 256, 256, 0, st>>>(g_dIn + off, g_dOut + off,
                                                     len, g_iters);
        CHECK(cudaGetLastError());

        if (k >= 1) {                       /* chunk k-1's result, deferred */
            int o2, l2;
            chunkOf(k - 1, g_n, nChunks, &o2, &l2);
            if (l2 > 0)
                CHECK(cudaMemcpyAsync(hOut + o2, g_dOut + o2,
                                      (size_t)l2 * sizeof(float),
                                      cudaMemcpyDeviceToHost,
                                      g_s[streamFor(k - 1, N_STREAMS)]));
        }
    }
    int oL, lL;                             /* the final chunk's deferred copy */
    chunkOf(nChunks - 1, g_n, nChunks, &oL, &lL);
    if (lL > 0)
        CHECK(cudaMemcpyAsync(hOut + oL, g_dOut + oL, (size_t)lL * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              g_s[streamFor(nChunks - 1, N_STREAMS)]));
}
```

**Why it is correct, from the hardware model.**

*Requirement (a), the dependencies.* Chunk `k`'s `H2D`, kernel and `D2H` are
all in the **same stream**, and a stream is an ordered queue. The kernel cannot
start before the copy that fills its input has finished, and the result copy
cannot start before the kernel has finished, because the hardware's stream
front-end will not advance past an incomplete operation. No event is needed;
this is the "cheaper mechanism" the TODO refers to. An event per edge would be
48 events and would express exactly the same thing.

*Requirement (b), the copy engine.* `asyncEngineCount == 1` on this GPU. There
is one DMA engine and it takes transfers in issue order. If `D2H(k)` is issued
immediately after `K(k)`, it reaches the head of the engine's queue while the
kernel it depends on has not started, and `H2D(k+1)` — which is ready **now** —
is stuck behind it. Deferring `D2H(k)` by one loop iteration puts `H2D(k+1)`
in front of it, so the engine always has runnable work.

This is the whole exercise. The naive order measures **0.87×**, i.e. slower
than the serial version it replaces, and this is reproducible: see
`example02.cu` part B, which times all three issue orders side by side.

*Requirement (c), no synchronization.* Everything inside the loop is an enqueue.
A `cudaDeviceSynchronize()` at the bottom of the loop body — a very common
"let's be safe" addition — turns the pipeline back into the serial version
exactly, and still prints the correct answer.

## TODO 4 — the wait

```cpp
static void waitForPipeline(void)
{
    for (int i = 0; i < N_STREAMS; ++i) CHECK(cudaStreamSynchronize(g_s[i]));
}
```

`cudaDeviceSynchronize()` is also correct here and measures the same, because
this program has nothing else running. It is the wrong *habit*: it waits for
work that has nothing to do with this pipeline, which in a real program means
waiting on another subsystem's kernels. Say what you mean.

Waiting on only `g_s[0]` is a real bug and the harness catches it: three
quarters of the output is still in flight, so `untouched ≈ 2300`.

## TODO 5a/5b — the bound and the prediction

```cpp
static double pipelineBound(double h2dMs, double kernelMs, double d2hMs,
                            int copyEngines)
{
    double serial   = h2dMs + kernelMs + d2hMs;
    double copyBusy = (copyEngines >= 2) ? ((h2dMs > d2hMs) ? h2dMs : d2hMs)
                                         : (h2dMs + d2hMs);
    double critical = (copyBusy > kernelMs) ? copyBusy : kernelMs;
    return serial / critical;
}
```

In steady state, every byte must pass through the copy engine(s) and every
element must pass through the SMs. The pipeline cannot finish before the
busiest resource has finished all of its work, so the floor on the time is
`max(copyBusy, K)` and the ceiling on the speedup is `(H + K + D)` divided by
that.

`copyBusy` is the whole point of the `copyEngines` argument. With **two or
more** engines the two directions proceed concurrently and the copy resource is
occupied for `max(H, D)`. With **one** engine the same engine serves both
directions and is occupied for `H + D`. Returning the same expression for both
is the specific wrong answer the harness probes for — two of the five reference
inputs use `copyEngines == 2`, and this device has one, so there is no way to
reach the right answer by measuring.

Measured phases give `H = 4.175`, `K = 5.271`, `D = 3.840`, so
`copyBusy = 8.015 > K` and the bound is `13.286 / 8.015 = 1.658×`.
`PRED_BUCKET = 3` (1.30× .. 2.00×). Note that a reader who assumed two copy
engines would have computed `13.286 / max(4.175, 5.271) = 2.52×` and predicted
bucket 4.

## Synchronization / memory reasoning

Three different mechanisms are doing three different jobs here, and conflating
them is how this exercise goes wrong:

| Job | Mechanism | Cost |
|---|---|---|
| order chunk `k`'s three operations | same stream | **free** |
| keep chunks `k` and `k+1` unordered | different streams | free |
| let the host know everything is done | `cudaStreamSynchronize` ×4 | one blocking call, once |

The pinned host buffer is not a performance tweak in this program — it is a
**correctness condition for the thing being measured**. `cudaMemcpyAsync` out of
pageable memory stages through a driver-owned pinned buffer on the calling
thread before returning, so it is asynchronous in name only.

## Performance reasoning

Observed on this GPU (RTX 3500 Ada Laptop, 40 SMs, power limit 72.8 W):

```
  H2D       4.175 ms  (12.1 GB/s)
  kernel    5.271 ms
  D2H       3.840 ms  (13.1 GB/s)
  bound 1.658x
  serial                         13.353 ms
  your pipeline, pinned           8.706 ms   1.53x    (93% of bound)
  your pipeline, pageable        11.390 ms   1.17x    (not scored)
```

**Where the missing 7% goes.** Ramp and drain. Chunk 0's `H2D` overlaps
nothing and chunk 15's `D2H` overlaps nothing, which costs roughly
`(H + D)/16 = 0.5 ms` on an 8.0 ms floor — about 6%. The remainder is host
issue cost. `example02.cu` part C sweeps the chunk count and shows the same
effect as a curve: 1 chunk → 1.01×, 8 → 1.54×, 16 → 1.55×, 64 → 1.39×.

**The pageable row.** 1.17× against the pinned 1.53×. Note that it is not 1.00×
— chunking recovers a little even from pageable memory, because the host's
staging `memcpy` for chunk `k` happens while chunks `k−1` and `k−2` are still
executing on the GPU. The chunking supplies some of the slack the pinned buffer
would have supplied. It is still a 31% loss, and the host thread is pegged doing
`memcpy` throughout, so it can do nothing else.

> **Operating-point caveat.** An earlier capture of this same binary, taken
> while the platform had the GPU power-limited to 30 W (battery charging from
> 8%, SM clock pinned at 210 MHz), reported H2D at **1.5 GB/s**, bound 1.850×,
> pipeline 1.73× — and the **pageable** row at 1.65×, nearly matching pinned.
> The ratio-to-its-own-bound was stable (94% vs 93%); the absolute numbers and
> the pageable conclusion were not. Check
> `nvidia-smi -q -d POWER | grep "Current Power Limit"` before believing an
> absolute transfer rate on this part.

## Expected output

```
device                 : NVIDIA RTX 3500 Ada Generation Laptop GPU (40 SMs)
asyncEngineCount       : 1
concurrentKernels      : 1
problem                : N = 12582917 floats = 48.0 MB, 3800 FFMA/element
pipeline               : 16 chunks over 4 streams

warming up (1500 ms stream + 500 ms compute) ...
done.

=== measured phase times (pinned, whole buffer) ===============
  H2D       4.175 ms  (12.1 GB/s)
  kernel    5.271 ms
  D2H       3.840 ms  (13.1 GB/s)
  sum      13.286 ms
  serial, three phases back to back:   13.353 ms

=== TODO 1: chunkOf() structural test =========================
  ok (+2)
  TODO 2 diagnostic -- your chunk -> stream map: 0 1 2 3 0 1 2 3 0 1 2 3 ...

=== TODO 5b: pipelineBound() reference check ==================
  bound(H=4.0 K=6.0 D=4.0 engines=1) = 1.7500
  bound(H=4.0 K=6.0 D=4.0 engines=2) = 2.3333
  bound(H=10.0 K=1.0 D=10.0 engines=1) = 1.0500
  bound(H=1.0 K=10.0 D=1.0 engines=1) = 1.2000
  bound(H=2.0 K=3.0 D=7.0 engines=2) = 1.7143
  matches the reference (+2)

  your bound for the measured phases: 1.658x

=== timing (one rotated sweep of 6 configurations, min of 6) ===
  serial                         13.353 ms
  your pipeline, pinned           8.706 ms   1.53x
  your pipeline, pageable        11.390 ms   1.17x   (not scored)

=== correctness ===============================================
  sampled 3075 elements: 0 never written, 0 wrong, worst 0.000e+00
  last element (index 12582916): ok
  ok (+2)

=== performance ===============================================
  measured 1.534x against your bound 1.658x = 93% of bound
  ok (+2)

=== prediction ================================================
  you said bucket 3, the measurement landed in bucket 3
  ok (+2)

SCORE: 10/10
OVERALL: PASS
```

Run to run the phase times move by about ±2% and the achieved fraction of the
bound by ±3 points (observed 91–94%). The bound itself moves with the phase
times because it is computed from them inside the same sweep, which is why the
scored quantity is the *ratio* and not the millisecond figure.

## The result that matters

The ceiling on copy/compute overlap is not a property of your code — it is
`(H + K + D) / max(copyBusy, K)`, and on a device with **one** copy engine
`copyBusy` is `H + D`, not `max(H, D)`. That single integer in `cudaDeviceProp`
moves the ceiling here from 2.53× to 1.66×, and no amount of chunking,
streaming or tuning crosses it. The engineering consequence is that you should
compute the bound *before* writing the pipeline, and if it comes out near 1.0 —
because the kernel dominates, or because the copies do — you should not write
the pipeline at all.

**Variation to try.** Raise `g_iters` from 3800 to 10000 and rerun. The kernel
now dominates the copy engine, `max(copyBusy, K) = K`, and the bound falls
toward `(H + K + D)/K → 1.0`. The pipeline still works perfectly and buys
almost nothing. Then drop `g_iters` to 300 so the copies dominate; the bound
falls again, from the other side. The peak is at `K = H + D`, where the bound
is 2.0 — and reaching it requires balancing the problem, not the code.
