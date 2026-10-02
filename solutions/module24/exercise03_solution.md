# Module 24 / Exercise 03 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -std=c++17 -o exercise03_solution.exe exercise03_solution.cu
exercise03_solution.exe
```

Warning-clean. Runs in about 15 s. Give it a cool-down before and after
(spec §12.5c).

---

## TODO 1 — `MIN_STREAMS = 2`, `MIN_EVENTS = 1`

The DAG:

```
    n0 H2D(x) --> n1 preA1 --> n2 preA2 --+
                                          +--> n6 join --> n7 reduce --> n8 D2H
    n3 H2D(y) --> n4 preB1 --> n5 preB2 --+
```

**Streams.** The number of streams you need is the **width of the graph** — the
maximum number of nodes that may be in flight simultaneously — not the number of
nodes. Here the two preprocessing chains are mutually independent and everything
from `n6` onwards is a line, so the width is **2**. Nine streams would also
work and would expose nothing extra; one stream would work and would expose
nothing at all.

The usual wrong answer is 3 ("two chains plus a stream for the join"). The join
has no concurrency to expose: nothing may run alongside it. Put it in whichever
chain's stream you like; it is a line.

**Events.** Count the edges you get for free *before* you count events. There
are eight edges. Six of them (`n0→n1`, `n1→n2`, `n3→n4`, `n4→n5`, `n6→n7`,
`n7→n8`) connect operations that are in the same stream and are supplied by
stream ordering at zero cost. Putting the join in stream 0 makes `n2→n6` a
same-stream edge too, for free. That leaves exactly one edge that crosses
streams — `n5→n6` — and therefore exactly **one** event.

> **The general rule.** `events = edges − (edges you can make same-stream)`.
> Laying out the graph so that the longest path is a single stream minimises
> both. An event per edge is six events and is not wrong, only wasteful — but
> the harness hands you exactly `MIN_EVENTS` events, so the over-claim is
> scored as a design failure rather than discovered later.

## TODO 2 — `criticalPath`

```cpp
static double criticalPath(const double* d)
{
    double chainA = d[0] + d[1] + d[2];
    double chainB = d[3] + d[4] + d[5];
    double head   = (chainA > chainB) ? chainA : chainB;
    return head + d[6] + d[7] + d[8];
}
```

The longest path through a DAG is the lower bound on its makespan given
unlimited resources. Here the structure is simple enough to write in closed
form: the two chains are parallel, so their contribution is the **max**, and
everything after the join is serial, so it is a **sum**.

Probe 3, `{3, 0, 0, 0, 0, 2, 0.5, 0.5, 0.5}`, is the one that catches a solution
that compares only the *last* node of each chain (`d[2]` vs `d[5]`) instead of
the chain totals: it would give `2 + 1.5 = 3.5` instead of the correct
`max(3, 2) + 1.5 = 4.5`.

A solution that returns `sum(d) / something` or that forgets `d[8]` fails probe
2, where the tail terms dominate.

## TODO 3 — the graph

```cpp
static void issueDag(const Bufs& B, cudaStream_t* s, int nStreams,
                     cudaEvent_t* ev, int nEvents)
{
    const size_t bytes = (size_t)B.n * sizeof(float);

    /* chain B, start to finish, in stream 1 */
    CHECK(cudaMemcpyAsync(B.dy, B.hy, bytes, cudaMemcpyHostToDevice, s[1]));
    chainStep<<<B.blocks, BLOCK, 0, s[1]>>>(B.dy, B.n, ITERS, B1);
    chainStep<<<B.blocks, BLOCK, 0, s[1]>>>(B.dy, B.n, ITERS, B2);

    /* the marker goes in AFTER the last node of chain B has been issued */
    CHECK(cudaEventRecord(ev[0], s[1]));

    /* chain A, start to finish, in stream 0 */
    CHECK(cudaMemcpyAsync(B.dx, B.hx, bytes, cudaMemcpyHostToDevice, s[0]));
    chainStep<<<B.blocks, BLOCK, 0, s[0]>>>(B.dx, B.n, ITERS, A1);
    chainStep<<<B.blocks, BLOCK, 0, s[0]>>>(B.dx, B.n, ITERS, A2);

    /* the one edge a stream cannot supply: n5 -> n6, across streams */
    CHECK(cudaStreamWaitEvent(s[0], ev[0], 0));

    joinKernel<<<B.blocks, BLOCK, 0, s[0]>>>(B.dx, B.dy, B.dz, B.n, ITERS / 2);
    reduceKernel<<<B.blocks, BLOCK, 0, s[0]>>>(B.dz, B.n, B.dPart);
    CHECK(cudaMemcpyAsync(B.hPart, B.dPart, (size_t)B.blocks * sizeof(float),
                          cudaMemcpyDeviceToHost, s[0]));
    CHECK(cudaGetLastError());
}
```

**The ordering of the two calls is the subtle part.**

`cudaEventRecord(ev[0], s[1])` captures **the contents of `s[1]` at the moment
of the call**, not "whatever ends up in `s[1]` eventually". `cudaStreamWaitEvent`
makes `s[0]` wait for exactly that much, and affects only work issued into
`s[0]` *after* the wait. So:

- record **after** issuing all of chain B, and **before** issuing the join → correct;
- record **before** issuing chain B's kernels → the event captures an empty (or
  partial) stream, the wait is satisfied immediately, and the join reads `dy`
  while `preB2` is still writing it;
- call `cudaStreamWaitEvent` **after** launching `joinKernel` → the wait orders
  the *reduce* against chain B, not the join.

Both mistakes produce a program that prints the right answer most of the time.
At SMALL (16 blocks) chain B finishes long before the host has issued the join,
so the race is almost never lost. The harness catches it structurally rather
than statistically: it compares your graph's partials against `serialDag`'s
partials element for element, and a mistimed edge shows up the moment the two
chains contend — which is reliably the case at LARGE.

**Which stream carries the join** is free choice. Putting it in `s[1]` works
equally well; you then record the event in `s[0]` instead. Putting it in a third
stream requires a *second* event and fails TODO 1.

**No synchronization anywhere.** `cudaEventRecord` and `cudaStreamWaitEvent` are
both enqueue operations: neither blocks the host. That is what lets the host
issue all nine nodes and get back to doing something else.

## TODO 4 — the wait

```cpp
static void waitForDag(cudaStream_t* s, int nStreams)
{
    for (int i = 0; i < nStreams; ++i) CHECK(cudaStreamSynchronize(s[i]));
}
```

Synchronizing `s[0]` alone is in fact sufficient — `s[0]` carries the terminal
node and already waits on `s[1]` through the event — but syncing both costs
nothing and does not depend on that reasoning staying true if the graph changes.

## Synchronization / memory reasoning

Three different mechanisms, each doing the job it is cheapest at:

| What needs ordering | Mechanism | Cost |
|---|---|---|
| the six intra-chain and post-join edges | same stream | **free** |
| the one cross-stream edge `n5 → n6` | `cudaEventRecord` + `cudaStreamWaitEvent` | one event, two enqueues, no host block |
| "is the whole graph done?" | `cudaStreamSynchronize` | one blocking call, once |

`cudaEventCreateWithFlags(ev, cudaEventDisableTiming)` is the right form here.
A timing-enabled event maintains a timestamp the driver has to read back;
`cudaEventDisableTiming` drops that and leaves a pure dependency marker. The
difference is small but the intent is clear, and `cudaEventElapsedTime` on such
an event correctly returns an error rather than a plausible-looking lie.

**Module 25 covers events properly.** Everything above treats them purely as a
dependency mechanism.

## Performance reasoning

The whole point of the exercise is in these two blocks. Same source, same
events, same streams — only the grid changes.

```
  --- SMALL (16 blocks = 0.40 blocks/SM) ---
    node durations (ms): H2D x=0.033 preA1=1.061 preA2=1.061 H2D y=0.036
                         preB1=1.059 preB2=1.061 join=0.542 reduce=0.034 D2H p=0.032
    serial    4.855 ms | your DAG    2.705 ms | 1.794x
    your critical-path bound 1.757x -> you reached 102% of it

  --- LARGE (640 blocks = 16.00 blocks/SM) ---
    node durations (ms): H2D x=0.048 preA1=4.879 preA2=4.898 H2D y=0.060
                         preB1=4.891 preB2=4.866 join=2.439 reduce=0.023 D2H p=0.034
    serial   22.054 ms | your DAG   20.997 ms | 1.050x
    your critical-path bound 1.790x -> you reached 59% of it
```

**SMALL: 1.79×, which is 102% of the critical path.** Each chain kernel is 16
blocks of 128 threads — 0.4 blocks per SM. Thirty-nine of the forty SMs are idle
while one chain runs. Running both chains concurrently is free, and the graph
executes at its theoretical lower bound.

The 102% is not an error and is worth understanding. Each node's duration is
measured *alone*, with a `cudaDeviceSynchronize` and an event record on either
side, so every measured node carries one launch's worth of ramp-up (~15 µs over
nine nodes ≈ 0.14 ms, about 3% of 4.9 ms). The DAG issues all nine back to back
and amortizes it. The critical path built from per-node measurements is
therefore very slightly pessimistic. Reaching 100–102% of it is the expected
result for a graph this small.

**LARGE: 1.050×, which is 59% of the same bound.** Each chain kernel is now 640
blocks — 16 blocks per SM against a hard limit, from Module 19's four-limiter
model, of `min(24, 48/4) = 12` blocks per SM at 128 threads. One chain kernel
already needs 1.33 waves to clear the machine. Running two of them concurrently
does not make the machine bigger; the blocks simply queue in the GigaThread
engine. The 5% that *is* gained comes from the tail: the last partial wave of
chain A leaves room for the first blocks of chain B.

> **This is the lesson, and it is not a lesson about streams.** The predictor is
> Module 1's wave arithmetic and Module 19's occupancy model. The dependency
> graph, the events and the stream count are identical in both runs and are all
> correct in both runs. Concurrency between kernels is available only in
> proportion to the machine the running kernel is *not* using, and on a 40-SM
> part a kernel written the way this course teaches is using all of it.
>
> The exception — and it is a large and important exception — is **copy/compute
> overlap**, which Exercises 1 and 2 are about. The copy engine is separate
> hardware that a compute kernel never touches, so that overlap is available
> essentially always.

The node durations make the same point arithmetically. SMALL `preA1` = 1.061 ms
for 16 blocks; LARGE `preA1` = 4.879 ms for 640 blocks. Forty times the work for
4.6× the time — the small kernel is latency-bound with the whole machine to
itself, exactly as Module 20's ILP/occupancy exchange-rate model predicts for a
dependent-FFMA chain at under one warp per scheduler.

## Expected output

```
device        : NVIDIA RTX 3500 Ada Generation Laptop GPU (40 SMs), asyncEngineCount = 1
DAG           : 9 nodes, two chains of 3 joining into a reduce
SMALL         : 16 blocks x 128 threads = 0.40 blocks/SM
LARGE         : 640 blocks x 128 threads = 16.00 blocks/SM

warming up (1500 ms stream + 500 ms compute) ...
done.

=== TODO 1: the shape of the graph =============================
  you claim 2 stream(s) and 1 event(s) : minimal (+2)

=== TODO 2: criticalPath() reference check =====================
  probe 0 -> 39.0000
  probe 1 -> 21.0000
  probe 2 -> 2.6250
  probe 3 -> 4.5000
  matches the reference (+2)

=== timing =====================================================
  --- SMALL (16 blocks) ---
    node durations (ms): H2D x=0.033 preA1=1.061 preA2=1.061 H2D y=0.036 preB1=1.059 preB2=1.061 join =0.542 reduce=0.034 D2H p=0.032
    serial    4.855 ms | your DAG    2.705 ms | 1.794x
    your critical-path bound 1.757x -> you reached 102% of it
  --- LARGE (640 blocks) ---
    node durations (ms): H2D x=0.048 preA1=4.879 preA2=4.898 H2D y=0.060 preB1=4.891 preB2=4.866 join =2.439 reduce=0.023 D2H p=0.034
    serial   22.054 ms | your DAG   20.997 ms | 1.050x
    your critical-path bound 1.790x -> you reached 59% of it

=== correctness ================================================
  --- SMALL ---
    0 of 16 partials differ from the serial result (worst 0.000e+00), 0 never written
    total: your DAG 1.251066e+04, serial 1.251066e+04
  --- LARGE ---
    0 of 640 partials differ from the serial result (worst 0.000e+00), 0 never written
    total: your DAG 5.005041e+05, serial 5.005041e+05
  --- host anchor (SMALL, block 0) ---
    host anchor on block 0: GPU 775.779480, host 775.779433, ok
  ok (+2)

=== performance ================================================
  SMALL: 1.794x against a bound of 1.757x : ok (+2)

=== predictions ================================================
  SMALL: you said 3, measured bucket 3 : ok (+1)
  LARGE: you said 1, measured bucket 1 : ok (+1)

SCORE: 10/10
OVERALL: PASS
```

Across four runs on this GPU, SMALL landed in **1.759–1.794×** (100–102% of its
bound) and LARGE in **1.008–1.050×** (49–59%). The bucket edges are at 1.15 and
1.40, which puts both results in the middle of their buckets rather than near an
edge (spec §12.5d).

The host anchor agrees to 4.7e-5 on a value of 776 — a relative error of 6e-8
over 1.25 million dependent FFMAs per element. Host and device `fmaf` are both
correctly rounded, so the per-element results are bit identical; the residual is
the different summation order of the 128 elements (block tree on the GPU, linear
on the host).

## The result that matters

Expressing a dependency graph correctly costs **one event**, because a stream is
already an ordered queue and six of the eight edges come free. Expressing it
*incorrectly* — recording the event before the work it is meant to capture —
costs nothing visible, because the resulting race is almost always won. And
expressing it correctly buys **1.79× at 0.4 blocks per SM and 1.05× at 16 blocks
per SM**, from identical code. The graph is a statement about what *may* happen
concurrently; the occupancy model decides what *does*. Build the graph because
it is the honest description of your dependencies, not because you expect it to
be fast.

**Variation to try.** Sweep `SMALL_BLOCKS` over 16, 40, 80, 160, 320, 640 and
plot the achieved fraction of the critical-path bound. You should recover the
staircase from `example01.cu` part B — free below 40 blocks (one block per SM),
partial at 80, gone by 240 — on a completely different program. Then set
`ITERS` to 20000 so each node lasts ~40 µs instead of ~1 ms, and watch the
SMALL case fall away from its bound as the ~15 µs per-launch cost stops being
amortized: that is the regime where **Module 28's CUDA graphs** are the answer,
not streams.
