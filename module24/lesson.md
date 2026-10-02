# Module 24 — CUDA Streams

> Prerequisites: Module 1 (waves, tail effects, the launch path), Module 2 (the
> four launch parameters, asynchronous launch, `cudaDeviceSynchronize`, event
> timing), Module 4 (pinned vs pageable host memory), Module 19 (occupancy and
> the four-limiter model), Module 22 (Nsight Systems — used here as an
> instrument, not taught).
>
> What this module gives you: the concurrency model of the CUDA runtime —
> what is ordered, what is not, and what silently orders things behind your
> back — plus the copy/compute overlap pattern, derived as a bound and then
> measured against it, and an honest account of the large class of kernels for
> which none of this buys anything at all.

This module opens **Part VII**. Everything up to here has been about making a
single kernel faster. From here on the subject is **the machine as a whole**:
getting the copy engine, the SMs and the host CPU to be busy at the same time.

---

## Concept

### 1. Module 2 left a parameter unexplained

```cpp
kernel<<<grid, block, sharedBytes, stream>>>(args...);
```

Module 2 introduced all four launch parameters and said of the fourth: *which
stream the launch is queued on; default 0 — Module 24.* Module 6 spent itself
on the third. This module is the fourth.

Module 2 also established the single most important fact about a kernel launch:
**it is asynchronous.** `kernel<<<...>>>()` writes a command into a pushbuffer
and returns; the host carries on while the GPU works. Module 2 measured the
host-side cost of a launch at a few microseconds against kernels lasting
milliseconds.

That asynchrony is already concurrency between the **host** and the **device**.
A stream is what you need to get concurrency *within* the device.

### 2. A stream is an ordered queue

> **PORTABLE CUDA CONCEPT.**
> A **stream** is a sequence of operations that execute **in issue order**.
> Two operations in the **same** stream are ordered: the second does not begin
> until the first has completed. Two operations in **different** streams have
> **no ordering relation at all** unless you create one.

That is the entire model. Everything else in this module is a consequence of
it, or an exception to it.

Three things follow immediately.

**(a) "No ordering" is a permission, not a promise.** Putting two kernels in
two streams does not make them run concurrently. It makes them *allowed* to.
Whether they actually do depends on whether there is hardware left over — which
is §4 below, and which is the part most tutorials skip.

**(b) The ordering you get inside a stream is free.** It costs no event, no
flag, no API call. A chain of dependent operations should be written as one
stream precisely because the dependency is then expressed by the data structure
rather than by synchronization. Exercise 3 turns this into an exam question:
the minimum number of events a dependency graph needs is the number of edges
that same-stream ordering *cannot* supply.

**(c) The absence of ordering is not detectable by testing.** A missing
cross-stream dependency is a race. `example02.cu` part D builds a two-chain
join, removes the one event that orders it, and measures **4062 of 4107 sampled
elements wrong** — but on a faster or slower machine the same code might be
right every time. You cannot test a missing dependency into existence; you have
to reason about it.

### 3. The default stream is not an ordinary stream

Pass no stream, or pass `0`, and the operation goes to the **default stream**.
Under the compilation mode `nvcc` uses unless told otherwise — the **legacy
default stream** — this stream has special semantics:

> An operation issued to the legacy default stream will not begin until all
> previously issued operations in **every blocking stream of the context** have
> completed, and no operation subsequently issued to any blocking stream will
> begin until it has completed.

A "blocking stream" is any stream created by `cudaStreamCreate`. So the legacy
default stream is a **device-wide barrier that you insert by forgetting to type
an argument.**

This is the highest-value fact in the module, so it is measured rather than
asserted. `example01.cu` part C runs the same correct 16-chunk pipeline five
ways. The only difference between rows 2 and 3 is one extra 32-thread kernel
per chunk that touches a single float and is launched with no stream argument:

```
  configuration                                                   ms x serial
  serial (one stream, blocking copies)                        13.351     1.00
  pipeline, clean                                              8.780     1.52
  pipeline + one kernel on the DEFAULT stream                 16.580     0.81
  pipeline + cudaMalloc/cudaFree inside the loop              14.594     0.91
  pipeline + a 4-byte blocking cudaMemcpy inside the loop     15.602     0.86
```

All five rows compute the **same correct answer**. Three of them are slower
than not using streams at all. Nothing in the source says so, no tool reports
an error, and the fastest and slowest differ by **1.89×**.

There are three cures, and they are not equivalent.

**Cure 1 — `cudaStreamNonBlocking`.**

```cpp
cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
```

A non-blocking stream is exempt from the legacy default stream's barrier. The
measurement: with non-blocking streams the poisoned pipeline runs at **1.53×**,
i.e. exactly as fast as the clean one. This is the right default for library
code, which cannot know what the application does with stream 0.

**Cure 2 — `--default-stream per-thread`.** A compile-time flag that redefines
`0` to mean "a distinct, *non-blocking* default stream per host thread". It
cures the problem at the source: there is no longer a stream with device-wide
semantics. Rebuilding `example01.cu` with the flag and nothing else changed:

```
  pipeline + one kernel on the DEFAULT stream                  8.824     1.53   <- cured
  pipeline + cudaMalloc/cudaFree inside the loop              14.815     0.91   <- NOT cured
  pipeline + a 4-byte blocking cudaMemcpy inside the loop      8.825     1.53   <- cured
```

**Cure 3 — pass an explicit stream to every asynchronous call.** Tedious,
and it does not help with cure 2's third row either, because —

### 4. Implicit synchronization: the other silent killer

Several CUDA runtime calls synchronize the **whole device** as a side effect of
doing their actual job. Per the programming guide, these include:

| Call | Why |
|---|---|
| `cudaMalloc`, `cudaFree` | the device allocator has to reorganize address space; it cannot do that under running kernels |
| `cudaHostAlloc`, `cudaMallocHost`, `cudaHostRegister` | page-locking host memory touches driver state the device is using |
| a blocking `cudaMemcpy` touching **pageable** host memory | the driver stages through its own buffer and must know what is in flight |
| `cudaDeviceSetCacheConfig`, L1/shared carve-up changes | reconfigures the SMs |
| a switch of the active device | — |

That is the table. The thing to take from it is the *shape*: **the calls that
synchronize are the calls that change the device's configuration**, and a
steady-state pipeline should not be changing the device's configuration.

> **The rule.** Allocate once, outside the loop. Anything inside the loop that
> is not a kernel launch, an async copy, an async memset, an event record or a
> stream wait is a suspect.

Row 4 of the table above is a 1024-byte `cudaMalloc`/`cudaFree` pair per chunk —
sixteen allocations the program does not need. It costs **1.67×** against the
clean pipeline and is not cured by either of the other two cures, because it
has nothing to do with the default stream. `nsys` sees it directly:

```
 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   ...  Max (ns)         Name
      2.3         50888766         71    716743.2  ...   5585175      cudaFree
```

71 calls to `cudaFree`, averaging **717 µs each**, in a program whose entire
pipeline takes 16 ms. `cudaFree` is not slow; it is *blocking*, and it blocks
for exactly as long as the work it drains.

### 5. The asynchronous API, and the memory it needs

```cpp
cudaMemcpyAsync(dst, src, bytes, kind, stream);
cudaMemsetAsync(ptr, value, bytes, stream);
```

Both enqueue into `stream` and return immediately — **if and only if the host
side of the transfer is page-locked (pinned) memory.**

Module 4 introduced pinned memory with `cudaMallocHost` and measured the raw
bandwidth difference at an unimpressive **1.04–1.15×** on this Windows/WDDM
driver. Re-measured here: 12.0 GB/s pinned vs 11.4 GB/s pageable H2D, a ratio
of **1.07×**.

> **The reframing this module exists to deliver.** On this machine pinned
> memory is worth almost nothing as a bandwidth optimisation. It is worth
> everything as a *concurrency* optimisation. **The overlap is the payoff, not
> the raw copy speed.**

`example02.cu` part A isolates this: one copy in one stream, one independent
kernel in another, nothing else in the program.

```
  case               copy ms     GB/s   both ms   overlap     ideal
  H2D pinned           4.191     12.0     5.682    1.666x    1.794x
  H2D pageable         4.418     11.4     9.904    0.979x    1.837x
  D2H pinned           3.839     13.1     5.589    1.631x    1.728x
  D2H pageable         4.131     12.2     9.539    0.986x    1.783x
```

`overlap = (copy_alone + kernel_alone) / both`. 1.00 means the two ran strictly
one after the other. Pinned reaches 93% of the ideal; **pageable reaches
0.979×, which is to say none at all**, despite the copy itself being only 5%
slower.

The mechanism (Module 26 owns the details): the DMA engine can only read
physical pages that are guaranteed not to move. For pageable memory the driver
copies into its own pinned staging buffer first — **on the calling host thread,
before `cudaMemcpyAsync` returns.** The call is asynchronous in name only.
`nsys` measures exactly this, as the *host-side duration of the API call*:

```
 pageable:  cudaMemcpyAsync   128 calls   avg 205610 ns
 pinned:    cudaMemcpyAsync   160 calls   avg  11706 ns
```

Same bytes, same direction, same hardware. **17.6× more host time**, all of it
spent memcpy'ing on the CPU. The GPU-side transfer durations are nearly
identical; it is the host that is blocked.

### 6. Waiting: four different questions

| Call | Question it answers | Blocks? |
|---|---|---|
| `cudaDeviceSynchronize()` | has *everything* finished? | yes |
| `cudaStreamSynchronize(s)` | has *this stream* finished? | yes |
| `cudaStreamQuery(s)` | has this stream finished **yet**? | **no** |
| `cudaEventSynchronize(e)` / `cudaEventQuery(e)` | has this *point* been reached? | yes / no |

`cudaStreamQuery` returns `cudaSuccess` or `cudaErrorNotReady`. It is a test,
not a synchronization point, and `cudaErrorNotReady` is **not** a sticky error
(Module 2's distinction) — it does not poison the context and it is not cleared
by `cudaGetLastError`. `example01.cu` part E spins on it:

```
  query on an idle stream          : cudaSuccess
  query immediately after a launch : cudaErrorNotReady
  host spun 42148 times before it reported cudaSuccess
```

Prefer `cudaStreamSynchronize` to `cudaDeviceSynchronize`. The device-wide call
is correct but says more than you mean, and in a program with more than one
subsystem it will eventually wait for something that is none of your business.

### 7. Cross-stream dependencies

Same-stream ordering is free but only expresses a chain. For a graph you need
two calls:

```cpp
cudaEventRecord(ev, producerStream);              // mark this point in the queue
cudaStreamWaitEvent(consumerStream, ev, 0);       // consumer waits for that mark
```

**Module 25 covers events properly** — as a timing instrument and in their own
right. Here they are used only as the mechanism for a cross-stream edge, and
for that purpose create them with `cudaEventCreateWithFlags(&ev,
cudaEventDisableTiming)`, which removes the timestamp bookkeeping.

The semantics are worth stating precisely, because the obvious misreading
produces a program that is right most of the time:

> `cudaEventRecord(ev, s)` captures **the work already issued into `s` at the
> moment of the call**. `cudaStreamWaitEvent(t, ev, 0)` makes `t` wait for
> exactly that much, and affects only work issued into `t` **after** the wait.

Neither call blocks the host. Record the event before you have issued the work
you meant to wait for and the wait is satisfied immediately and enforces
nothing — and your program still prints the right answer whenever the producer
happens to win the race.

### 8. Host callbacks

```cpp
cudaLaunchHostFunc(stream, fn, userData);
```

Enqueues a host function into a stream. The driver calls `fn(userData)` on one
of its own threads when the stream reaches that point. It is how you get a
notification without a host thread blocking on `cudaStreamSynchronize`.

> **The rule, and it is absolute: you must not call any CUDA API inside the
> callback.** Not `cudaGetLastError`, not `cudaStreamQuery`, nothing. The
> callback runs inside the driver's own progress machinery; re-entering the
> driver from there is undefined and in practice deadlocks.

What a callback may do: touch host memory, signal a condition variable, push to
a queue, log. What it may not do: anything that would make more GPU work happen
directly. Enqueue that work *before* the callback instead.

(`cudaStreamAddCallback` is the older form and is deprecated. Use
`cudaLaunchHostFunc`.)

### 9. Stream priority

```cpp
int lo, hi;
cudaDeviceGetStreamPriorityRange(&lo, &hi);     // this GPU: lo = 0, hi = -5
cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, hi);
```

**Lower numerical value means higher priority**, which reads backwards and is
the usual first bug. On this device the range is `[0, -5]` — six levels.

Priority is a **hint to the block scheduler**: when blocks from two streams are
competing for placement, blocks from the higher-priority stream are preferred.
It does **not** preempt a running block (Module 1: blocks are indivisible and
non-migrating once placed), and it does nothing at all when the machine is not
oversubscribed. It is useful for latency-sensitive work sharing a device with a
throughput job; it is not a scheduling guarantee.

---

## Hardware Mental Model

### 10. Three engines, and the one that is scarce

A CUDA device presents at least three independent hardware engines to the
driver:

1. the **compute engine** — the GigaThread engine feeding the 40 SMs;
2. one or more **copy (DMA) engines** moving data over PCIe;
3. the host CPU, which issues work into both.

`cudaDeviceProp` reports what you have:

```
device                  : NVIDIA RTX 3500 Ada Generation Laptop GPU (cc 8.9, 40 SMs)
asyncEngineCount        : 1
concurrentKernels       : 1
stream priority range   : least=0  greatest=-5  (6 levels)
```

> **ARCHITECTURE-SPECIFIC, AND IT CHANGES EVERY BOUND IN THIS MODULE:
> `asyncEngineCount == 1` on this GPU.**

`asyncEngineCount` is the number of DMA engines that can run a copy
*concurrently with kernel execution*. One engine means:

- a copy **can** overlap a kernel — the thing this module is about;
- two copies **cannot** overlap each other, in either direction.

So the familiar textbook picture of three-way overlap (H2D of chunk *k+1*, the
kernel of chunk *k*, and D2H of chunk *k−1*, all at once) is **not achievable
here**. Datacentre parts report `asyncEngineCount = 2` or more and get it; this
laptop part does not. Every bound below is computed from `1`, and the module
prints the two-engine bound next to it so the difference is visible.

`concurrentKernels = 1` is a boolean, not a count: it says kernels from
different streams *may* run concurrently. Every device since Fermi reports 1.

### 11. The copy/compute pipeline, and its bound

Split an *N*-element problem into *M* chunks and give the chunks to several
streams. Let `H`, `K`, `D` be the time to move, compute and return the **whole**
buffer. Serially the job costs `H + K + D`. In steady state the pipeline is
limited by whichever engine is busiest:

```
copyBusy = (asyncEngineCount >= 2) ? max(H, D) : (H + D)
speedup  = (H + K + D) / max(copyBusy, K)
```

With this GPU's measured phases:

```
  H2D 4.248 ms | kernel 5.304 ms | D2H 3.880 ms | serial sum 13.432 ms
  copy engines = 1  ->  copy engine is busy 8.128 ms total
  achievable bound  = (H+K+D)/max(copyBusy, K) = 1.653x
  if there were 2+ copy engines it would be  = 2.532x  (NOT available here)
```

Two things to notice. First, **the bound is a modest number.** 1.65× is the
*ceiling*, not the target. Second, the bound is maximised when the copy engine
and the compute engine are equally loaded; it approaches 2× as `K → H + D` and
collapses to 1× when either side dominates. This is the sense in which the
pipeline is a *balance* problem, not a *speed* problem.

### 12. Head-of-line blocking — the defect one copy engine creates

Here is the issue order everybody writes first:

```cpp
for (k = 0; k < M; ++k) {
    cudaMemcpyAsync(dIn + o, hIn + o, len, H2D, s[k % S]);
    kernel<<<..., s[k % S]>>>(dIn + o, dOut + o, len);
    cudaMemcpyAsync(hOut + o, dOut + o, len, D2H, s[k % S]);   // <-- here
}
```

It is correct, it is obvious, and on this GPU it is **slower than not using
streams at all**:

```
  configuration                                        ms x serial % of bound
  serial (blocking copies, one stream)             13.406     1.00        --
  order 1: per chunk H2D,K,D2H                     15.451     0.87       53%
  order 2: per chunk H2D,K,D2H(k-1)                 8.816     1.52       92%
  order 3: two phases, all H2D+K then all D2H       9.032     1.48       90%
```

**Why.** There is one copy engine and it takes transfers in the order they were
issued. `D2H(k)` is issued immediately after `K(k)` and therefore sits at the
head of the engine's queue *waiting for a kernel that has not started*.
`H2D(k+1)` — which is ready to run right now, and whose data the GPU needs
next — is stuck behind it. The engine idles while a runnable transfer waits.
The pipeline degenerates to `H, K, D, H, K, D, …` plus the overhead of having
pretended otherwise.

The two repairs both keep a **ready** transfer at the head of the queue:

- **Order 2 (deferred D2H):** issue chunk *k*'s result copy one iteration
  later, after `H2D(k+1)` and `K(k+1)` have been issued. One iteration of
  software pipelining.
- **Order 3 (two phases):** issue every `H2D`+kernel first, then every `D2H`.
  Simpler, but it needs the whole output resident on the device.

> This defect **does not exist** on a GPU with two copy engines, which is why
> it is absent from most tutorials and why most tutorials' code is slower than
> serial when you run it on a laptop.

### 13. How many chunks?

```
   chunks         ms   x serial % of bound
        1     13.303       1.01        61%
        2      9.451       1.42        86%
        4      8.948       1.50        91%
        8      8.726       1.54        93%
       16      8.638       1.55        94%
       32      9.050       1.48        90%
       64      9.663       1.39        84%
```

Two costs fight each other.

**Ramp and drain.** The first chunk's `H2D` overlaps nothing and the last
chunk's `D2H` overlaps nothing, so a pipeline of `M` chunks wastes roughly
`(H + D)/M` of its time at the two ends. That term falls as `1/M`, which is why
the climb from 1 to 8 chunks is steep and everything past 8 is flat.

**Host-side issue cost.** Each chunk costs three or four runtime calls, a few
microseconds each (Module 2 measured the launch; Module 22 measured it head
on). That term grows linearly in `M`. At 64 chunks the host can no longer stay
ahead of the GPU and the curve turns down.

The optimum is broad — anything from 4 to 32 is within 5% here — which is the
useful engineering conclusion. **Do not tune the chunk count. Pick 8 or 16 and
spend the effort on the issue order instead**, which is worth 1.75× between
order 1 and order 2 and is a correctness-preserving rewrite either way.

Note also: `nChunks` and `nStreams` are **different numbers**. Streams are how
many independent queues exist; chunks are how many pieces of work there are.
Four streams carrying sixteen chunks round-robin is normal and correct. Setting
`nStreams = nChunks` wastes stream objects; setting `nStreams = 1` with
`nChunks = 16` reintroduces full serialization while *looking* chunked.

### 14. Where streams do NOT help — and on 40 SMs that is most of the time

This is the part of the subject that gets left out, and it is the part that
decides whether you should be reading this module at all.

Two **independent** kernels, one per stream, with the grid swept:

```
  blocks  blk/SM   one(ms)   two(ms)   ratio   verdict
       1    0.03     4.912     4.915   1.001   free
      10    0.25     4.620     4.633   1.003   free
      20    0.50     4.186     4.208   1.005   free
      40    1.00     4.175     4.243   1.016   free
      80    2.00     4.255     5.322   1.251   partial
     240    6.00     8.697    14.755   1.697   no room
```

`ratio = both / one`. 1.00 is perfect concurrency; 2.00 is none.

The predictor is **Module 1's wave arithmetic**, not anything in this module.
A kernel that occupies `b` blocks per SM leaves room for a concurrent kernel
only while `b` is below the per-SM block limit that Module 19's four-limiter
model gives. At 40 blocks this kernel is one block per SM, 39/40 of the machine
is idle, and a second copy is literally free. At 240 blocks it is six blocks
per SM, the placement gate is saturated, and the two launches simply queue.

> **The honest statement.** This GPU has **40 SMs**. Any kernel written the way
> this course has spent twenty-three modules teaching you to write them — a
> grid sized to the machine, 100% or near-100% occupancy, several waves of
> blocks — **already owns the whole device, and a second stream cannot give it
> anything.** Concurrent *kernel* execution is for small kernels: tails,
> reductions' last rungs, per-sample work in an inference server, graph nodes
> that are individually tiny.
>
> Copy/compute overlap is different and is almost always worth having, because
> the copy engine is a *separate* piece of hardware that is otherwise idle.
> **That is the one form of concurrency this module recommends unreservedly.**

Exercise 3 measures both sides of this on the same dependency graph. At 16
blocks per kernel (0.4 blocks/SM) the graph runs at **1.76×, which is 101% of
its critical-path bound** — perfect. At 640 blocks per kernel (16 blocks/SM)
the identical graph, identical code, identical events runs at **1.008×, 49% of
a 2.05× bound.** Same program. The hardware decided.

### 15. An operating-point warning specific to this machine

Spec §12.5b exists because this laptop's power manager can move the whole
machine to a different operating point without telling you. During the
authoring of this module the platform dropped the GPU's power limit from 60 W
to **30 W** while the battery charged from 8%, pinning the SM clock at 210 MHz
and the memory clock at 405 MHz with `SW_POWER_CAP` and `SW_THERMAL_SLOWDOWN`
asserted. In that state:

| | 30 W cap | healthy (72.8 W) |
|---|---|---|
| H2D, 48 MB pinned | 1.5 GB/s | 12.0 GB/s |
| the same kernel | 53.4 ms | 5.3 ms |
| pipeline bound | 1.85× | 1.65× |
| measured pipeline | 1.73× (94% of bound) | 1.52× (92% of bound) |
| **pageable** 16-chunk pipeline | 1.65× | **1.15×** |

The **ratios inside one rotated sweep survived** — the pipeline reached 92–94%
of its own bound in both states, which is why every gate in this module's
exercises is a ratio computed inside a single sweep. The **absolute numbers did
not**, and neither did the pageable-memory conclusion: under the cap the copy
engine was so slow that the host's staging memcpy had slack to hide in, and
pageable memory looked almost as good as pinned. It is not. Always check
`nvidia-smi -q -d POWER | grep "Current Power Limit"` before believing an
absolute transfer rate on this part.

---

## Code Walkthrough

### `example01.cu` — the model, measured

**Part A** queries and prints the three capability fields and then says what
each one costs you. The important line is

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("asyncEngineCount        : %d\n", prop.asyncEngineCount);
```

because every bound the file computes afterwards branches on it:

```cpp
double copyTotal = (prop.asyncEngineCount >= 2) ? fmax(h2d, d2h) : (h2d + d2h);
double bound     = (h2d + ker + d2h) / fmax(copyTotal, (double)ker);
```

**Part B** is the concurrency sweep of §14. The kernel is deliberately
compute-bound with one load and one store, because spec §12.8 says a
memory-bound kernel makes a poor occupancy demonstration: a lone
bandwidth-bound block has the whole memory system to itself and runs far faster
than it would in a full wave, which smears the staircase into a ramp.

**Parts C/D** run one pipeline function under four perturbations selected by an
enum, so the pipeline code is byte-identical across rows:

```cpp
if (poison == POISON_DEFAULT_STREAM) {
    touch<<<1, 32>>>(p.dOut);           /* <<< no stream argument >>> */
} else if (poison == POISON_ALLOC_IN_LOOP) {
    void* scratch = nullptr;
    CHECK(cudaMalloc(&scratch, 1024));
    CHECK(cudaFree(scratch));
} else if (poison == POISON_BLOCKING_COPY) {
    int probe = 0;
    CHECK(cudaMemcpy(&probe, p.dOut, sizeof(int), cudaMemcpyDeviceToHost));
}
```

Each of the three is something a reasonable person writes: a debug kernel, a
scratch buffer sized to the chunk, a progress probe. Note that the third one
copies **four bytes**. The volume is irrelevant; the synchronization is the
cost.

The five configurations are timed in one rotated sweep with `SWEEPS == NCFG`
(spec §12.1, §12.9), and validation is a separate pass afterwards (§12.2).

**Part E** demonstrates `cudaStreamQuery` and `cudaLaunchHostFunc`. The
callback is a captureless lambda, which converts to the required
`void (*)(void*)`:

```cpp
CHECK(cudaLaunchHostFunc(s[1], [](void* ud) { ++*(int*)ud; }, &callbackHits));
printf("  callbackHits right after enqueue : %d\n", callbackHits);   /* 0 */
CHECK(cudaStreamSynchronize(s[1]));
printf("  callbackHits after stream sync   : %d\n", callbackHits);   /* 1 */
```

### `example02.cu` — the pipeline, derived and measured

**Part A** is the pinned/pageable isolation of §5. The design point worth
copying: all nine configurations (kernel alone, four copies alone, four
copy+kernel pairs) are in **one** rotated sweep, so the kernel's own duration
cannot drift between the "alone" column and the "both" column. Timing the two
columns in separate loops is the standard way to manufacture a fake overlap
number on this part.

**Part B** computes the bound and measures the three issue orders of §12.
Order 2 is the deferred-D2H software pipeline:

```cpp
cudaMemcpyAsync(dIn + o, hIn + o, len, H2D, st);
condition<<<blocks, 256, 0, st>>>(dIn + o, dOut + o, len, iters);
if (k >= 1) {                                    /* chunk k-1's result */
    chunkOf(k - 1, n, nChunks, &o2, &l2);
    cudaMemcpyAsync(hOut + o2, dOut + o2, l2, D2H, s[(k - 1) % nStreams]);
}
```

followed, after the loop, by the deferred copy for the final chunk. Note that
the deferred copy goes into **chunk k−1's own stream**, not chunk k's — that is
what makes it wait for the right kernel without an event.

**Part C** is the chunk-count sweep of §13.

**Part D** is the smallest interesting DAG: two chains joining. The event is
the only thing holding it together, and the file runs the graph twice, with and
without:

```cpp
if (useEvent) {
    CHECK(cudaEventRecord(depY, g_s[1]));
    CHECK(cudaStreamWaitEvent(g_s[0], depY, 0));
}
addKernel<<<blocks, 256, 0, g_s[0]>>>(dx, dy, dz, M);
```

```
  with cudaStreamWaitEvent : mismatches = 0
  without it               : mismatches = 4062
```

The second number is the one to think about. It is not zero, so the bug is
real; it is also not 4107, so the bug is *intermittent*. A test suite that
sampled 100 elements on a lightly loaded machine would pass.

---

## Check Your Understanding

Answers in `solutions/module24/check_your_understanding.md`. These require
reasoning about the model; none of them is lookupable.

1. A colleague reports that converting their program to four streams made it
   **slower**, and sends you a profile showing that the GPU is busy 100% of the
   time in both versions, with identical total kernel time and identical total
   memcpy time. They conclude streams are useless on this hardware. Give the
   two structurally different explanations that are both consistent with that
   evidence, and name one measurement that distinguishes them.

2. You have a GPU with `asyncEngineCount = 2`. Phase times for the whole buffer
   are H2D = 10 ms, kernel = 4 ms, D2H = 10 ms. Someone proposes splitting into
   many chunks and reports a measured speedup of 2.4×. Is that possible? Derive
   the bound, then explain what the pipeline would have to be doing for the
   claim to be true *without* the measurement being wrong.

3. Two kernels, A and B, are issued to two non-blocking streams with no event
   between them. A writes `buf[0..N)`; B reads `buf[0..N)`. The program prints
   the right answer on 1000 consecutive runs. Your colleague argues this proves
   the driver inserted the dependency for you, citing the fact that CUDA
   "tracks" buffer usage. Rebut this precisely, and then explain why the
   *absence* of a hang (as opposed to a wrong answer) is specifically
   uninformative here.

4. A library function you do not control calls `cudaDeviceSynchronize()` once
   per call. You cannot change it. Your pipeline issues work into four
   non-blocking streams and calls that library function once per chunk.
   `cudaStreamNonBlocking` has already protected you from the legacy default
   stream. Does it protect you from this? Answer for both `cudaStreamSynchronize`
   and `cudaDeviceSynchronize` inside the library, and say what the fix is.

---

## Exercises

### Exercise 1 — `exercise01.cu` — build the overlap pipeline

**Type:** fill-in + design + prediction (§6 types 1, 5, 6). **5 TODOs, 10 points.**

A 48 MB pinned buffer must be sent to the GPU, transformed by a compute-bound
kernel, and brought back. Turn the three serial phases into a chunked pipeline.

```
nvcc -arch=sm_89 -O3 -std=c++17 -o exercise01.exe exercise01.cu
exercise01.exe
```

| TODO | What it requires |
|---|---|
| 1 | `chunkOf(k, n, nChunks, &off, &len)` — partition `[0, n)` with a ragged tail; `n` is prime-ish and is **not** a multiple of `nChunks` |
| 2 | `streamFor(k, nStreams)` — the chunk→stream map |
| 3 | the pipeline itself: the operation sequence, with no synchronization, no allocation and no blocking copy anywhere inside it |
| 4 | the final wait — block until every chunk's result is in host memory, and no longer |
| 5a/5b | predict the speedup bucket, and write `pipelineBound(H, K, D, copyEngines)` correctly for **both** one copy engine and two |

**Validation.** `chunkOf` is checked structurally on six `(n, nChunks)` pairs
including degenerate ones (`nChunks > n`, `nChunks == 1`) before anything runs
on the GPU. `pipelineBound` is checked against a hashed reference on five
synthetic inputs, two of which have `copyEngines == 2` — this device has one, so
you cannot reach the right answer by measurement. The pipeline must produce the
correct answer including the last element, must reach **≥ 70% of your own
bound**, and `PRED_BUCKET` must match the bucket the measurement lands in.
All five gates must pass for `OVERALL: PASS`.

Run it once with the TODOs blank: it prints the three phase times and this
device's `asyncEngineCount` and exits. Those are the inputs to TODO 5a, and you
are expected to compute the prediction rather than guess it.

### Exercise 2 — `exercise02.cu` — the concurrency that isn't there

**Type:** debugging + fill-in + prediction (§6 types 3, 1, 2). **5 TODOs, 10 points.**

A 16-chunk, 4-stream pipeline that produces the right answer, has no error, is
clean under `compute-sanitizer`, and is **slower than the serial version it
replaced**. There are three independent causes. The file states the symptom and
not the cause.

```
nvcc -arch=sm_89 -O3 -std=c++17 -o exercise02.exe exercise02.cu
nvcc -arch=sm_89 -O3 -std=c++17 --default-stream per-thread -o exercise02_pt.exe exercise02.cu
exercise02.exe
```

| TODO | What it requires |
|---|---|
| 1 | name the three causes from a list of eight candidates, five of which are plausible and false |
| 2 | repair the pipeline without changing what it computes |
| 3 | fix the host output buffer |
| 4 | `overlapFactor(start, end, nOps)` — reduce the program's own event timeline to one number that says whether anything overlapped |
| 5 | predict how many of the three causes `--default-stream per-thread` removes, and the bucket the repaired speedup lands in |

The program carries its own instrument: `OP_BEGIN`/`OP_END` bracket each issued
operation with a pair of events and the harness replays them as an ASCII
timeline. Leave those calls where they are. `nsys` shows the same thing more
precisely and the solution notes give the command — **Module 22 owns that
tool**; here it is an instrument.

**Validation.** The three diagnoses are checked against a hash, order
independent. `overlapFactor` is checked against a reference on a synthetic
four-operation timeline. The repaired pipeline must still write every chunk,
must still zero and populate the per-chunk clip counters and the per-chunk
scratch slices, and must reach **≥ 1.35× serial**. Both predictions are scored
separately. 10/10 required.

### Exercise 3 — `exercise03.cu` — design a DAG

**Type:** design from scratch + performance reasoning (§6 types 5, 6).
**5 TODOs, 10 points.**

Nine operations whose dependencies form a graph, not a line: two independent
preprocessing chains of three nodes each, joining into a fused kernel, then a
reduction and a copy out. Express the graph with the **fewest streams and the
fewest events** that still expose every independent pair, then find out what it
is worth.

```
nvcc -arch=sm_89 -O3 -std=c++17 -o exercise03.exe exercise03.cu
exercise03.exe
```

| TODO | What it requires |
|---|---|
| 1 | `MIN_STREAMS` and `MIN_EVENTS` — the design question; the harness hands `issueDag` exactly that many and no more |
| 2 | `criticalPath(d[9])` — the longest path through the DAG from the nine measured node durations |
| 3 | build the graph: every edge enforced, nothing else ordered, no blocking |
| 4 | the wait |
| 5 | two predictions: the speedup bucket at **16 blocks per kernel** and at **640 blocks per kernel** |

**Validation.** `MIN_STREAMS`/`MIN_EVENTS` are hashed — claim too few and you
cannot build a correct graph, claim too many and the TODO scores zero.
`criticalPath` is hashed against a reference on four synthetic duration vectors.
Correctness is checked twice: partial-for-partial against the serial
serialization of the same graph (which catches a missing or mistimed edge), and
against an independent host recomputation of one block (which catches both
versions being wrong the same way). The SMALL configuration must reach ≥ 70% of
your own critical-path bound. Both prediction buckets are scored.

The two predictions are **not the same number**, and the reason is in Module 1.

---

## Prediction

Commit to these in writing before you run anything.

1. **The bound you cannot exceed.** Before running Exercise 1, write down this
   device's `asyncEngineCount` from memory, the formula for the pipeline bound
   that follows from it, and — given that the kernel in that exercise takes
   roughly as long as the two copies put together — the numerical value of the
   bound to one decimal place. Then write down what fraction of it you expect a
   16-chunk pipeline to reach, and why it is not 100%.

2. **The issue order.** Before reading §12, write the obvious per-chunk loop
   (`H2D(k)`, `K(k)`, `D2H(k)`) on paper and predict, with a reason, whether it
   will be faster or slower than not using streams at all on a device with
   **one** copy engine. Commit to a number to two decimal places, then read §12.
   Most people predict "a bit faster".

3. **Where it stops paying.** Exercise 3 runs one dependency graph at two grid
   sizes, 16 and 640 blocks per kernel. Before running it, derive both speedups
   yourself from Module 1's wave arithmetic and Module 19's four-limiter model —
   not from §14's table, which gives you the answer but not the derivation.
   Write down the blocks-per-SM limit for a 128-thread block, the number of
   waves each configuration needs, and the speedup that follows. Then state
   which of the two numbers you could have predicted from *this* module alone,
   and notice that the answer is neither of them.
