# Module 24 / Exercise 02 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -std=c++17 -o exercise02_solution.exe exercise02_solution.cu
exercise02_solution.exe

# for TODO 5a, the same source with one flag changed:
nvcc -arch=sm_89 -O3 -std=c++17 --default-stream per-thread -o exercise02_pt.exe exercise02.cu
exercise02_pt.exe
```

Warning-clean. Run on its own with a cool-down before it (spec §12.5c).

---

## TODO 1 — the three causes: **2, 4, 5**

### Cause 2 — an operation on the legacy default stream

```cpp
CHECK(cudaMemsetAsync(g_dClip + k, 0, sizeof(int)));   /* <-- no stream argument */
```

`cudaMemsetAsync` has a four-argument form; the three-argument call defaults
`stream` to `0`. Under the legacy default stream (what `nvcc` gives you unless
you pass `--default-stream per-thread`) every operation on stream 0 is a
**device-wide barrier**: it waits for all previously issued work in every
blocking stream, and everything subsequently issued into a blocking stream waits
for it. One four-byte memset per chunk, sixteen times, serializes the entire
pipeline.

**The fix** is a one-token edit, and it gives the chunk's ordering for free:

```cpp
CHECK(cudaMemsetAsync(g_dClip + k, 0, sizeof(int), st));
```

Putting it in `st` also places it *before* the kernel in the same stream, which
is exactly the ordering the clip counter needs.

### Cause 4 — a device allocation inside the loop

```cpp
float* scratch = nullptr;
CHECK(cudaMalloc(&scratch, (size_t)blocks * sizeof(float)));
...
CHECK(cudaFree(scratch));
```

`cudaMalloc` and `cudaFree` are **implicit synchronization**: they drain the
device before returning, because the allocator cannot reorganize device address
space underneath running kernels. This is independent of streams and of the
default-stream mode. It is also the reason the `cudaFree` placed immediately
after the kernel launch is "safe" — it blocks until the kernel that is using
`scratch` has finished — which is precisely what makes it so expensive.

**The fix** is to delete both calls and use the slice `main()` already
allocated:

```cpp
condition<<<blocks, BLOCK, 0, st>>>(g_dIn + off, g_dOut + off, len, g_iters,
                                    g_dClip + k,
                                    g_dScratch + (size_t)k * g_stride);
```

`g_stride` is at least any chunk's block count, so chunk `k`'s slice is big
enough. Note the offset: passing the bare `g_dScratch` makes every chunk write
the same prefix, and the harness's per-chunk scratch check — which compares each
chunk's slice against that chunk's own data — reports `per-block scratch BROKEN`
deterministically.

### Cause 5 — the host output buffer is pageable

```cpp
static float* allocHostOut(size_t bytes) { return (float*)malloc(bytes); }
```

`cudaMemcpyAsync` into pageable host memory is asynchronous **in name only**.
The DMA engine can only write physical pages that are guaranteed not to move, so
the driver transfers into its own pinned bounce buffer and then `memcpy`s to the
destination — **on the calling host thread, before the call returns.**

**The fix:**

```cpp
static float* allocHostOut(size_t bytes)
{
    float* p = nullptr;
    CHECK(cudaMallocHost(&p, bytes));
    return p;
}
static void freeHostOut(float* p) { CHECK(cudaFreeHost(p)); }
```

### Why the other five are wrong

| # | Claim | Why it is false here |
|---|---|---|
| 1 | grid too small, chunks serialize on the SMs | each chunk launches 3073 blocks = 77 blocks/SM. The grid is far too **large** for inter-kernel concurrency, not too small — but that is not what is costing the overlap, which is copy/compute, not kernel/kernel |
| 3 | round-robin puts ordered work in different streams | round-robin is the **correct** map. Chunk `k`'s three operations are all in `g_s[k % N_STREAMS]`; nothing that must be ordered crosses a stream |
| 6 | more streams than copy engines serializes | the engine count bounds how many *transfers* run at once, not how many streams may exist. Four streams over one engine is normal and is what the repaired version does |
| 7 | `cudaMemcpyAsync` is only async below 64 KB | invented. There is no such threshold. (There *is* a small-transfer path where the driver copies through the command buffer, but it makes small copies *more* asynchronous, not less) |
| 8 | the driver inserts a dependency to prevent a race | the driver does no such tracking. Streams are ordered queues and nothing else; an unordered access to the same buffer from two streams is a race, and the runtime will let you have it |

Candidates 1 and 8 are the two that catch people, because both describe real
phenomena — kernel concurrency does need a small grid, and races on shared
buffers are real. Neither is the reason *this* program does not overlap.

## TODO 2/3 — the repaired pipeline

Full listing in `exercise02_solution.cu`. The three edits are the three above;
nothing else changes. In particular the issue order — `H2D(k)`, `K(k)`,
`D2H(k−1)` — was already correct in the broken version. That is deliberate: the
exercise is about silent serialization, not about head-of-line blocking (which
is Exercise 1's subject).

## TODO 4 — `overlapFactor`

```cpp
static double overlapFactor(const double* start, const double* end, int nOps)
{
    if (nOps <= 0) return 0.0;
    double busy = 0.0, lo = start[0], hi = end[0];
    for (int i = 0; i < nOps; ++i) {
        busy += end[i] - start[i];
        if (start[i] < lo) lo = start[i];
        if (end[i]   > hi) hi = end[i];
    }
    double span = hi - lo;
    if (span <= 0.0) return 0.0;
    return busy / span;
}
```

Sum of per-operation durations, divided by the wall span from the earliest start
to the latest end. That is the **time-average number of operations in flight**.
If nothing ever overlaps, the busy intervals tile the span and the ratio is 1.0
(or less, if there are gaps — which is why the span and not the sum of gaps is
the denominator: idle time inside the pipeline must count against it).

The synthetic check is `{[0,3], [1,5], [2,6], [4,6]}` → busy = 3+4+4+2 = 13,
span = 6, factor 2.1667.

**The honest limitation, and why `nsys` exists.** `OP_BEGIN` records an event in
stream `st` *before* the operation and `OP_END` records one *after*. An event
records the moment the **stream** reaches that point, which for an idle stream
is immediately — even though the operation itself may then sit in the copy
engine's queue for milliseconds. So `start` is "when the stream accepted the
work", not "when the engine began it", and `overlapFactor` reports an **upper
bound**. In the output below several H2D bars start at 0.43–0.48 ms and end at
1.6–2.2 ms, which is queue time, not transfer time. The number is still a good
*discriminator* — ~1.0 broken, ~3.5 repaired — and that is all the exercise
asks of it. Getting true per-engine begin times requires a tracing profiler.

## TODO 5 — the two predictions

**(a) `PRED_PT_CURED = 1`.** Only cause 2 is a default-stream problem.

The measurement, from `example01.cu` rebuilt with `--default-stream per-thread`
(same binary semantics, three poisons, one table):

```
  pipeline, clean                                              8.872     1.52
  pipeline + one kernel on the DEFAULT stream                  8.824     1.53   <- cured
  pipeline + cudaMalloc/cudaFree inside the loop              14.815     0.91   <- NOT cured
  pipeline + a 4-byte blocking cudaMemcpy inside the loop      8.825     1.53   <- cured
```

The generalisation worth keeping: **per-thread default stream cures
default-stream synchronization, not API synchronization.** A blocking
`cudaMemcpy` on stream 0 is the first kind and is cured. `cudaMalloc`/`cudaFree`
is the second kind and is not. Cause 5 (pageable memory) is neither — it is a
property of the host allocation and no compile flag touches it.

The warning in the TODO text matters: rebuilding `exercise02.cu` with
`--default-stream per-thread` and **measuring** gives you almost nothing,
because causes 4 and 5 each independently serialize the pipeline and mask the
cure. The question is about causes, not milliseconds.

**(b) `PRED_BUCKET = 3`** (1.35× .. 2.00×). Phase times here are H2D ≈ 4.2 ms,
kernel ≈ 5.3 ms, D2H ≈ 3.9 ms with one copy engine, so the bound is
`13.4 / max(8.1, 5.3) = 1.65×` and a 16-chunk pipeline reaches 90–95% of it.

## Synchronization / memory reasoning

The three causes are three *different* mechanisms that produce one
indistinguishable symptom:

| Cause | Mechanism | What it blocks |
|---|---|---|
| 2 | legacy default stream semantics | every blocking stream in the context, on the device |
| 4 | implicit sync in the allocator | the whole device, from the host |
| 5 | driver bounce-buffer staging | the calling host thread |

Only one of them is visible in the device timeline as a barrier. Cause 5 shows
up only as the *host* being late, which is why the API-duration column of a
profile is the right place to look for it.

## Finding it with Nsight Systems

**Module 22 owns this tool.** The capture:

```
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
"$NSYS" profile -o brk --force-overwrite=true --stats=true ./exercise02.exe
"$NSYS" profile -o fix --force-overwrite=true --stats=true ./exercise02_solution.exe
```

On Windows the target's stdout is not forwarded through a redirected pipe, so
run the program once normally to see its own report and once under `nsys` to
collect the profile.

Two rows of `cuda_api_sum` contain the whole diagnosis. **Broken:**

```
 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)   Max (ns)          Name
      2.3         50888766         71    716743.2    636103.0      7640    5585175      cudaFree
      1.2         26318048        128    205609.8    183716.5      7382     678419      cudaMemcpyAsync
```

**Repaired:**

```
      0.4          8613742          7   1230534.6    309516.0      6979    4913711      cudaFree
      0.1          1872876        160     11705.5      9414.0      7359      86461      cudaMemcpyAsync
```

- `cudaFree`: **71 calls at 717 µs average** becomes 7 calls (the teardown
  ones). That is cause 4, measured directly as host blocking time.
- `cudaMemcpyAsync`: **205.6 µs average host time** becomes **11.7 µs** — a
  17.6× reduction for the same bytes over the same link. That is cause 5. The
  *GPU-side* transfer durations in `cuda_gpu_mem_time_sum` barely move; it is
  the host that was blocked.
- Cause 2 does not show in a summary table at all. It is a *shape* in the
  timeline: open the report in the GUI and the four stream rows contain
  strictly non-overlapping bars with a stream-0 memset between every pair.
  The program's own ASCII timeline (below) shows the same shape.

`compute-sanitizer --tool memcheck` and `--tool racecheck` are both clean on the
broken version, as the file header says. No correctness tool can find this
class of bug: the program is correct.

## Performance reasoning

Observed, same GPU, same session, 30 s apart.

**Broken** (as shipped, TODOs blank):

```
=== the symptom ================================================
  serial, three phases, one stream :   13.710 ms
  'pipelined', 4 streams           :   16.943 ms   0.81x

=== the event timeline (first 12 operations) ===================
  op  kind     start       end  timeline (1 column = 0.391 ms)
  0   H2D      0.023     0.295  #
  1   ker      0.297     0.664  ##
  2   H2D      0.715     0.982  .##
  3   ker      0.985     1.415  ..##
  4   D2H      1.427     1.998  ...###
  5   H2D      2.075     2.370  .....##
  6   ker      2.373     2.736  ......#
  7   D2H      2.772     3.209  .......##
  8   H2D      3.274     3.544  ........##
  9   ker      3.547     3.939  .........##
  10  D2H      4.025     4.530  ..........##
  11  H2D      4.607     4.895  ...........##
```

Every bar begins after the previous one ends. No two share a column. The
"pipeline" is a serial program with four unused streams and 48 extra event
records, which is why it is **0.81× — slower than the version it replaced.**

**Repaired:**

```
=== the symptom ================================================
  serial, three phases, one stream :   16.823 ms
  'pipelined', 4 streams           :   10.005 ms   1.68x

=== the event timeline (first 12 operations) ===================
  op  kind     start       end  timeline (1 column = 0.209 ms)
  0   H2D      0.195     0.484  ###
  1   ker      0.487     1.054  ..####
  2   H2D      0.432     0.779  ..##
  3   ker      0.781     1.662  ...#####
  4   D2H      1.056     1.372  .....##
  5   H2D      0.457     1.659  ..######
  6   ker      1.666     2.219  .......####
  7   D2H      1.666     1.958  .......###
  8   H2D      0.482     2.226  ..#########
  9   ker      2.230     2.777  ..........####
  10  D2H      2.225     2.516  ..........###
  11  H2D      1.489     2.784  .......#######
```

Operations 2, 5 and 8 all start before operation 1 ends. `overlapFactor` goes
from ≈ 1.0 to **3.47**.

## Expected output

The full repaired run (phase times move ±2% run to run; the speedup has been
observed in the range **1.52–1.68×** across six runs at the healthy operating
point, and 1.74× under a 30 W power cap):

```
=== TODO 1: diagnosis ==========================================
  you named 2, 4, 5 : correct (+3)

=== TODO 4: overlapFactor() ====================================
  synthetic timeline -> you 2.166667, reference 2.166667 : ok (+2)
  overlapFactor on the traced run above : 3.468

=== TODO 2/3: the repair =======================================
  serial     16.823 ms
  pipeline   10.005 ms   1.68x
  0 wrong, 0 unwritten, clip counters ok, per-block scratch ok
  ok (+3)

=== TODO 5: the two predictions ================================
  (a) per-thread default stream removes 1 of the three : ok (+1)
  (b) bucket: you said 3, measurement is in 3 : ok (+1)

SCORE: 10/10
OVERALL: PASS
```

## The result that matters

There is no error, no warning, no wrong number, and `compute-sanitizer` is
clean — and the program is **0.81× of the serial version it was written to
replace**. Every one of the three bugs is something a competent person writes on
purpose: a memset with the default stream argument omitted, a scratch buffer
sized to the chunk it serves, a host output array allocated with `malloc`.
**Concurrency has no error path.** The only way to know you have it is to
measure that two things happened at once, which is why this exercise makes you
build the instrument before it lets you claim the fix.

**Variation to try.** Fix the three causes one at a time, in each of the six
orders, and record the speedup after each step. You will find that fixing causes
2 and 4 while leaving 5 gives about 1.17× (the pageable pipeline from
Exercise 1), and that fixing *only* 5 gives 0.81× — unchanged. Three
independent serializers mask each other almost perfectly, which is the real
reason this bug class survives code review: the first fix appears to do nothing.
