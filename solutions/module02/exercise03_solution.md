# Module 02 / Exercise 03 — Solution notes

**Do not read this until you have committed your predictions in writing.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
```

The solution file has the three predictions filled in with the correct answers:

```cpp
static int  predicted_phaseA_order        = 1;
static int  predicted_launch_cost_bucket  = 2;
static bool predicted_memcpy_sees_results = true;
```

## TODO 1 — the ordering is `1` : H1, H2, D…, H3

Host "before launch", host "after the launch statement", **then** all eight
device lines, **then** host "after cudaDeviceSynchronize".

Two independent mechanisms produce this, and you need both to justify it:

**The launch returns immediately.** `announce_and_spin<<<2,4>>>(d_sink, 4000000)`
enqueues and returns in a handful of microseconds. The kernel then spins for a
dependent chain of 4 million FMAs — several milliseconds. The host reaches H2
long before the GPU finishes, and (because each host `printf` is followed by
`fflush(stdout)`) H2 reaches the terminal immediately. So H2 precedes the device
output no matter how the device output is produced.

**Device `printf` is flushed at synchronization points, not at the call.** The
device-side implementation reserves space in a global-memory FIFO with an atomic
bump, writes the format-string pointer and the promoted arguments, and returns.
Nothing is formatted or printed on the device. The *host* runtime drains that
FIFO and does the I/O when it next synchronizes — here, inside
`cudaDeviceSynchronize()`, **before that call returns**. Therefore the device
lines land after H2 and before H3, deterministically.

Why not the other options:

- **2 (H1, D…, H2, H3)** would require the launch to block until the kernel
  produced output. It does not.
- **3 (H1, H2, H3, D…)** would require the flush to happen *after* the sync
  returns. The flush is part of the sync's completion handling. (You *can*
  reach state 3 by deleting `fflush(stdout)` and redirecting stdout to a file:
  then the host's own stdio buffering, not CUDA, reorders the output. That is a
  libc artifact, not a CUDA property — and it is exactly why both examples call
  `fflush`.)
- **4 (varies)** confuses two different things. The *placement of the device
  block relative to H2 and H3* is deterministic. The *order of lines within that
  block* is not.

**The second half of the question: are the eight device lines in global-id
order? No.** Observed, repeatably, on this GPU:

```
  [device] global id 4 (block 1, thread 0)
  [device] global id 5 (block 1, thread 1)
  [device] global id 6 (block 1, thread 2)
  [device] global id 7 (block 1, thread 3)
  [device] global id 0 (block 0, thread 0)
  [device] global id 1 (block 0, thread 1)
  [device] global id 2 (block 0, thread 2)
  [device] global id 3 (block 0, thread 3)
```

Block 1 before block 0. *Within* a block the order is ascending, because the
four threads are lanes 0–3 of a single warp executing one `printf` call, and the
FIFO reservation for a converged warp is done in lane order. *Between* blocks
there is no order at all: the two blocks were dispatched to different SMs by the
GigaThread engine and raced for the FIFO. Which one wins is a property of
dispatch timing, and the fact that it comes out the same way every run on this
machine is an accident of that timing — not a guarantee. Any program whose
correctness depends on device `printf` ordering is broken.

If you predicted "ascending order", the useful lesson is that *nothing* about a
grid is ordered unless you order it: not execution, not memory visibility, not
output.

## TODO 2 — bucket `2` : roughly 1–20 µs

Actual, RTX 3500 Ada Laptop, CUDA 13.2, Windows/WDDM (three consecutive runs
gave 3.45–8.82 µs for the long-kernel case):

```
  host cost/launch, short kernel :     6.58 us
  host cost/launch, long kernel  :     3.45 us
  GPU duration of the long kernel:   583.78 us
  bucket observed = 2, you predicted 2 -> MATCH
```

A launch costs single-digit microseconds of **CPU** time regardless of how long
the kernel runs — 583 µs of GPU work for 3.45 µs of host time, a ratio of ~170×.
That microsecond is spent validating arguments, marshalling the kernel
parameters into the launch packet, and appending the packet to a command buffer.
The GPU is not consulted.

Why not the other buckets:

- **1 (under 1 µs)** — too optimistic. There is a real driver call, a parameter
  copy, and on Windows a WDDM submission decision. Nobody gets below ~1 µs with
  the runtime API. (CUDA Graphs, much later, exist largely to amortize this.)
- **3 (100–1000 µs)** — that is the scale of a *synchronizing* operation, not an
  enqueue.
- **4 (about the kernel's duration)** — that is what you would measure if the
  launch were synchronous. The program explicitly separates the two so you can
  see they are unrelated.

**ARCHITECTURE-SPECIFIC OPTIMIZATION:** the spread between the two host numbers
is a Windows WDDM effect. `example01.cu` shows the same thing more sharply
(17.7 µs for the first launch after the queue drains, 5.0 µs for a launch
appended to a queue the GPU is already working on): when the queue is empty the
driver must submit a fresh command buffer through the OS scheduler; when it is
busy the launch is just an append. On Linux with the native driver the numbers
are lower and far less bimodal. **PORTABLE CUDA CONCEPT:** a launch costs
microseconds of host time, and that is a floor, not a function of the work.

The practical consequence you will use for the rest of the course: **a kernel
that does less than ~50 µs of work is dominated by launch overhead.** Fuse it,
batch it, or make it bigger.

## TODO 3 — `true` : the unsynchronized `cudaMemcpy` sees the results

```cpp
compute<<<blocks, threads>>>(d_in, d_out, n);
CHECK(cudaGetLastError());
// no cudaDeviceSynchronize()
CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
```

Actual:

```
  mismatching elements after an unsynchronized D2H copy : 0 / 1048576
  h_out correct = true, you predicted true -> MATCH
```

This is not luck and it is not a race. Two guarantees stack:

1. **Stream ordering.** Both the launch and the copy are issued to the default
   stream (stream 0, since neither specified one). Operations in a stream
   execute in issue order: the copy cannot begin until the kernel has completed.
2. **The blocking form of `cudaMemcpy` synchronizes.** `cudaMemcpy` (as opposed
   to `cudaMemcpyAsync`) does not return until the copy is complete. Combined
   with (1), it does not return until the kernel is complete either.

So `cudaDeviceSynchronize()` here would be redundant — it would wait for
something that has already been waited for, and it would be one more full
CPU/GPU serialization point. The reflex "I must sync before I read the results"
is the single most common piece of CUDA cargo-cult code.

**Where this stops being true**, and why you should not over-generalize:

| Situation | Safe without an explicit sync? |
|---|---|
| Kernel then blocking `cudaMemcpy`, same stream | Yes — this exercise |
| Kernel then `cudaMemcpyAsync` then read `h_out` | **No.** The async copy returns immediately; you must sync the stream or an event before reading the host buffer. |
| Kernel on stream A, copy on stream B | **No.** Different streams are unordered with respect to each other (Module 24). |
| Kernel writes managed memory, host reads it directly | **No.** Needs a sync. |
| You want to *time* the kernel | The copy synchronizes, but timing with a host clock across it measures copy + kernel + launch. Use events. |

So the correct rule is not "never sync" and not "always sync"; it is **know what
already orders your operations, and add synchronization only where nothing
does.** `cudaDeviceSynchronize()` is the blunt instrument that is always
sufficient and almost never minimal.

## Synchronization / memory reasoning

The program contains exactly four kinds of ordering, and it is worth naming
which does what:

- `cudaDeviceSynchronize()` — host waits for all device work. Used in Phase A to
  force the `printf` flush, and around the timing loops to isolate them.
- `cudaEventRecord` / `cudaEventSynchronize` — a timestamp placed *in the
  stream*, and a host wait on that timestamp. This is how you time GPU work.
- Stream ordering — implicit, free, and what makes Phase C correct.
- `fflush(stdout)` — not a CUDA mechanism at all, but part of why Phase A's
  output order is trustworthy.

## Performance reasoning

Phase B's long kernel is `spin_only<<<40,32>>>` with 200,000 dependent FMAs:
40 blocks of one warp each, one block per SM, so every SM runs a single warp.
That is deliberately terrible occupancy — 1 warp out of 48 resident slots — and
it is why 256 million FMAs (1280 threads × 200,000) take 583 µs. The kernel is latency-bound on the FMA
dependency chain: each `fma` must wait for the previous result (~4 cycles on
Ada), and with only one warp per SM there is nothing else to issue in those
cycles. 200,000 × 4 cycles ≈ 800,000 cycles ≈ 570 µs at ~1.4 GHz, which matches
the measurement. This is Module 1's latency-hiding argument, observed: the
machine is idle 3 cycles out of 4 and the only cure is more warps.

Phase C's `compute` kernel is the opposite — 4096 blocks, trivially parallel,
bandwidth-bound: 8 MB of traffic in roughly 30 µs, i.e. a few hundred GB/s of
the 432 GB/s peak.

## Expected output

Actual output on the RTX 3500 Ada Laptop GPU, CUDA 13.2:

```
--- Phase A ---
  [host] before launch
  [host] after the launch statement
  [device] global id 4 (block 1, thread 0)
  [device] global id 5 (block 1, thread 1)
  [device] global id 6 (block 1, thread 2)
  [device] global id 7 (block 1, thread 3)
  [device] global id 0 (block 0, thread 0)
  [device] global id 1 (block 0, thread 1)
  [device] global id 2 (block 0, thread 2)
  [device] global id 3 (block 0, thread 3)
  [host] after cudaDeviceSynchronize
  (TODO 1: compare the ordering above with your prediction, 1)

--- Phase B ---
  host cost/launch, short kernel :     6.58 us
  host cost/launch, long kernel  :     3.45 us
  GPU duration of the long kernel:   583.78 us
  bucket observed = 2, you predicted 2 -> MATCH

--- Phase C ---
  mismatching elements after an unsynchronized D2H copy : 0 / 1048576
  h_out correct = true, you predicted true -> MATCH

PASS -- 2 of the 2 machine-checkable predictions matched.
(TODO 1 is checked by eye against the Phase A output above.)
```

Run-to-run variation: the two host-cost numbers move around (observed ranges
across three runs: 6.6–20.7 µs short, 3.5–8.8 µs long) because they are host
timings subject to OS scheduling and WDDM batching. The GPU duration is stable
to within ~0.5% (581.9–583.8 µs). The Phase A ordering and the Phase C result
were identical on every run.

## The result that matters

Asynchrony is not an implementation detail you can ignore until you care about
performance — it determines *when you are allowed to look at anything*. The
host runs ahead by microseconds while the GPU works for milliseconds; device
output you "printed" does not exist until something synchronizes; and the
correctness of reading a result depends on which ordering guarantee happens to
cover it, not on whether you remembered to type `cudaDeviceSynchronize()`. Learn
to name the guarantee.

Variation to try, with a warning attached. Allocate `h_out` with
`cudaMallocHost` (pinned) and change the Phase C copy to `cudaMemcpyAsync(h_out,
d_out, bytes, cudaMemcpyDeviceToHost)` — still stream 0, still no sync. Now
there is genuinely no guarantee that `h_out` is populated when the comparison
loop reads it. **Verified result on this machine: it still prints PASS.** The
host spends a million iterations building `h_ref` before it reads `h_out`, and
the copy finishes during that window. That is the lesson: an unsynchronized
async copy is not "a bug that shows up as wrong answers", it is a bug that
produces right answers until the timing shifts — a different GPU, a larger
buffer, a busier machine. Correctness by coincidence is the hardest failure mode
in asynchronous programming, and it is why you reason from the guarantees rather
than from the output. Pinned memory, `cudaMemcpyAsync`, and how to order it
properly are Modules 4 and 24.
