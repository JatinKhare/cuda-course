# Module 22 — Check Your Understanding (answers)

---

## 1. 980 ms of kernel time in a 1000 ms range — why "GPU-bound, host is fine" can be wrong

The conclusion is that the busy fraction is 98% and therefore the host has
nothing left to give. Two materially different ways that is wrong:

**(a) "Busy" is not "utilized."** `cuda_gpu_kern_sum` reports that *a kernel was
resident on the device*, nothing more. A grid of 8 blocks on a 40-SM part keeps
the timeline solid for the whole 980 ms while 32 SMs idle; so does a kernel at
6% of peak because every warp is stalled on DRAM. The busy fraction is blind to
both by construction — this module's instrument measures *when*, not *how well*.
`cuda_gpu_kern_gb_sum` gives the grid and block dimensions per kernel, which
catches the 8-block case immediately; everything else is Module 23's job.

**(b) The 980 ms may be one kernel, not many, and the two have different
remedies.** `cuda_gpu_kern_sum`'s `Instances` column distinguishes them. 980 ms
over 2 instances is a kernel problem. 980 ms over 200 000 instances is 4.9 µs per
kernel — *below this machine's 8–14 µs launch floor* — which means the device
timeline is 98% full only because `nsys` is counting overlapping launches or
because the host happens to be keeping up by a hair, and any small perturbation
will turn it into a picket fence. Check `cuda_kern_exec_sum`: if `QAvg` (queue
delay) is large relative to `KAvg`, the host is already behind.

A third, sneakier version: the NVTX range you divided by might not cover the
expensive part. If `nvtx_sum` shows your range at 1000 ms but the process ran for
8 s, you have measured a well-behaved region of a badly-behaved program.

**The reports that distinguish them:** `cuda_gpu_kern_sum` (instances and
averages), `cuda_gpu_kern_gb_sum` (grid shape), `cuda_kern_exec_sum`
(`AAvg`/`QAvg`/`KAvg` decomposition), `nvtx_sum` (is my range the whole program).

---

## 2. The 1.15 ms `cudaDeviceSynchronize` row — why removing only the `cudaMemcpy` is unsound

The reasoning being used is "cost is proportional to the row total, so the sync
is 0.4% of the memcpy and therefore irrelevant." That is invalid because **the
two calls are in series, and the first one to block pays for both.**

The iteration is: upload, kernelA, kernelB, 4-byte readback, sync. The readback
is a blocking pageable `cudaMemcpy`; it cannot return until kernelA and kernelB
have finished. By the time control reaches `cudaDeviceSynchronize`, the device is
already idle and there is nothing to wait for, so the call returns in 3.8 µs. Its
`cuda_api_sum` total measures *how much work was still outstanding when it was
called*, which is zero — not *how much it costs you*.

**What the next profile looks like.** Remove the `cudaMemcpy` and the
`cudaDeviceSynchronize` inherits the entire wait. Its row grows from 1.15 ms to
roughly whatever the `cudaMemcpy` row minus its DMA time used to be, the
`cudaMemcpy` row loses 300 calls, and **the wall time barely moves** — because
you removed a *symptom* and the *cause* (one full queue drain per iteration) is
still there, now wearing a different name. Measured in Exercise 2: fixing the
transfers but keeping the sync leaves 61.7 µs/iteration; removing the sync as
well gives 51.7 µs, a further 1.19×, which is exactly one launch bubble.

**The generalizable rule:** a ranked `cuda_api_sum` tells you what to fix
*first*. It never tells you what to fix *last*, and it systematically
under-reports every serialization point that sits behind another one. Re-profile
after every fix.

---

## 3. Event pair says 40 µs, `nsys` says 9 µs — the mechanism, and what 50 launches would report

`cudaEventRecord` does not take a timestamp when you call it. It **enqueues a
timestamp-write command into the stream**, which the device executes when it
reaches that point in the stream. `cudaEventElapsedTime` then returns the
distance between the two device-side writes.

So the measured interval is:

```
[ beg marker retires ... device idle, waiting for work ... kernel runs ... end marker retires ]
```

If the queue was empty when you recorded `beg`, the device executes the marker
immediately and then has nothing to do until the host's `cudaLaunchKernel` call —
8–14 µs of driver round trip on this machine — delivers the kernel. That idle
stretch is inside the interval. 40 µs = 9 µs of kernel + ~31 µs of a host that
was not ready. The event pair is an **upper bound** on kernel duration, and the
bound is loosest exactly when the host is the bottleneck, which is exactly when
you most want the truth.

**Fifty launches before synchronizing.** The first iteration still pays the
empty-queue penalty. After that the host runs ahead: while the device executes
kernel *k*, the host is already inside `cudaLaunchKernel` for *k+1*, and by the
time the device finishes, the next kernel is queued. The per-launch event pairs
converge on `max(kernel duration, per-launch host cost)` — here
`max(9, ~10) ≈ 10–12 µs`, not 40 — and the total for 50 launches approaches
`50 × 12 µs`, not `50 × 40 µs`.

Two corollaries worth keeping:

- **The event bound tightens as the queue deepens.** You cannot tell from the
  number alone which regime you are in.
- **It never goes below the true kernel duration.** If your event pair ever
  reports *less* than `nsys`, you have a different problem — most likely you are
  comparing an un-profiled run with a profiled one (see Q4).

---

## 4. 93.8% vs 80.5%, and a busy fraction above 100%

These are two different artefacts and it matters to keep them apart.

**Example 1, version B: harness 93.8%, `nsys` 80.5%.** Same direction, same
mechanism as Q3. The harness's numerator is a sum of event-pair intervals, each
of which is an upper bound on its kernel's duration because it absorbs whatever
launch gap sat between the two markers. The denominator (wall time) is honest.
So the quotient is an **over-estimate of busy**, and 93.8% > 80.5% is the bound
being loose by 13 points. Note that the same harness over-estimates version A by
more (71.4% claimed vs 55.8% true, 16 points), because A drains its queue every
step and therefore has more gap to absorb. The harness is not just wrong; it is
*differentially* wrong in the direction that flatters the worse version.

**Exercise 2, `runFast`: 107.3%.** This cannot be the same thing — an upper-bound
numerator over an honest denominator cannot exceed the truth *within one run*.
The cause is that the two numbers come from **two different runs**: the numerator
is `cuda_gpu_kern_sum` from a *profiled* capture and the denominator is the wall
time of an *un-profiled* execution. Tracing adds a small per-kernel cost, and the
GPU clock differs between the two runs anyway. 107% therefore means "the kernel
total and the wall time agree to within the cross-run noise", i.e. **`runFast` is
GPU-bound**, which was the goal.

The honest way to state it: a busy fraction built from two runs has a few percent
of cross-run error and should be reported as "≈100%", not "107%". A busy fraction
built from one capture (kernel total from `cuda_gpu_kern_sum` divided by an NVTX
range from the *same* capture — as Example 1 does) has no such error, and that is
the form to prefer.

> Related and worse: on this laptop the *absolute* numbers are not comparable
> across sessions at all. The same binary, same command, reported a kernel at
> 35.1 µs in one session and 253.4 µs in another — **7.2×** — with `nvidia-smi`
> showing the SM clock at 210 MHz. Every gate in this module's exercises is
> therefore built from ratios inside a single capture, never from a millisecond
> figure compared across runs.
