# Module 13 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o ex3sol.exe exercise03_solution.cu
ex3sol.exe
```

Warning-clean, CUDA 13.2, sm_89. About 30 s: a 9-size correctness sweep, 200
back-to-back launches, a 400 ms warm-up, 4 timed sweeps and two full validations
of a 67 M array.

---

## TODO 1 — the tile index (DESIGN)

```cpp
if (tid == 0) { s_tile = atomicAdd(ticket, 1u); s_excl = 0u; }
__syncthreads();
const u32 utile = s_tile;
if (utile >= (u32)m) return;
const int tile = (int)utile;
```

**Why not `blockIdx.x`.** The look-back makes a block spin until a
*lower-numbered* tile publishes. Module 9's deadlock argument applies in full: a
spinning block holds the SM slot that the block it waits for needs, and a block
that has not been placed does not exist. With `tile = blockIdx.x`, nothing in
the programming model says block 4,999 is placed before block 5,000. If the
GigaThread engine placed blocks in *any* order other than increasing `blockIdx`,
a resident high-numbered block could spin forever on a queued low-numbered one.

**Why the ticket closes the argument.** Let `t` be a tile index handed out by
the atomic. The atomic is the block's *first* action, so a block holding ticket
`t` implies that tickets `0 … t−1` were already returned to blocks that had
therefore already executed an instruction — i.e. every lower-numbered tile is
owned by a block that is **already resident on an SM**. From M1: a resident
block is indivisible, non-migrating, and runs to completion. Tile 0 does not
wait. By induction on `t`, every tile eventually publishes `FLAG_P`, so every
spin terminates.

**The assumption that remains.** The induction needs "a resident block makes
progress." That is a *forward-progress* property of the warp scheduler, and the
CUDA programming guide does not state it. It is true of every NVIDIA GPU shipped
and it is what CUB relies on, but it is an implementation property, not a
language guarantee. **ARCHITECTURE-SPECIFIC / IMPLEMENTATION-DEPENDENT.** If you
write one of these, put that sentence in a comment above it.

**Why the guard is unsigned.** `s_tile` is initialised to `0xFFFFFFFFu` before
the atomic so that an unfilled TODO 1 produces `utile >= m` and a clean early
return. Comparing as `int` would make that `-1 >= m`, which is false, and the
block would compute `base = -1024` and read out of bounds. The comparison being
unsigned is also what makes the "extra blocks" case safe in general.

The early `return` is block-uniform (every thread of the block sees the same
`s_tile`), so the `__syncthreads()` calls after it are legal under M9's
uniformity rule. A per-thread early return before a barrier would be exactly
M9's undefined case (c).

## TODO 2 — publishing the aggregate

```cpp
if (tid == 0) {
    if (tile == 0) {
        pfxs[0] = total;                 // no predecessors: aggregate IS the prefix
        __threadfence();
        atomicExch(&flags[0], FLAG_P);
    } else {
        aggs[tile] = total;
        __threadfence();
        atomicExch(&flags[tile], FLAG_A);
    }
}
```

**Device scope, not block scope.** The consumer is a thread on a *different SM*.
M4: L1 is not coherent across SMs and L2 is the first level they share.
`__threadfence_block()` compiles to `MEMBAR.SC.CTA`, which orders against the
SM's local ordering point only, and would be silently insufficient — it would
work almost all the time, which is worse than failing. `__threadfence()` is
`MEMBAR.SC.GPU` and pushes the payload to L2 before the flag store can become
visible there.

**Why the flag store is an atomic.** `atomicExch` executes at the L2 (M10), so
it cannot sit in a non-coherent L1 line, and it cannot tear. A plain `flags[t] =
FLAG_A` would be a normal store and the compiler is free to cache it. A
`volatile` store would defeat the caching but, per M9, `volatile` supplies no
ordering and no atomicity — it is not a synchronization mechanism and it is not
the right tool here.

**Tile 0 must be special-cased.** If tile 0 publishes only `FLAG_A`, no tile ever
sees a `FLAG_P` and every look-back walks to `j < 0`. That still terminates and
still gives the right answer (the `j < 0` lanes contribute a zero prefix) — but
every tile walks the whole way and the scan becomes O(m²). At m = 65,536 it does
not finish in any useful time.

**Three separate state words, not two.** A tempting compression is to keep one
value slot whose meaning is given by the flag. It is wrong: a reader can observe
`FLAG_A`, then be descheduled, then read the slot *after* the owner has
overwritten it with the inclusive prefix, and add a prefix where an aggregate
was expected. The aggregate and the prefix must live in different words, or —
CUB's approach — in different bit-fields of a single word written atomically.

## TODO 3 — the look-back

```cpp
if (tid < 32) {
    const int lane = tid;
    u32 excl = 0u;
    int look = tile - 1;
    while (true) {
        int j = look - lane;
        u32 f, v;
        if (j >= 0) {
            do { f = atomicAdd(&flags[j], 0u); } while (f == FLAG_X);
            __threadfence();
            v = (f == FLAG_P) ? pfxs[j] : aggs[j];
        } else {
            f = FLAG_P; v = 0u;
        }
        u32 pmask  = __ballot_sync(0xffffffffu, f == FLAG_P);
        int firstP = pmask ? (__ffs((int)pmask) - 1) : 32;
        u32 c = (lane <= firstP) ? v : 0u;
        for (int o = 16; o; o >>= 1) c += __shfl_down_sync(0xffffffffu, c, o);
        c = __shfl_sync(0xffffffffu, c, 0);
        excl += c;
        if (pmask) break;
        look -= 32;
    }
```

Line by line, because every line is load-bearing.

- **`j = look - lane`** puts the *nearest* predecessor in lane 0. That matters
  because the termination condition is "the nearest lane with a known prefix",
  and `__ffs` returns the lowest set bit.
- **`atomicAdd(&flags[j], 0u)` as the load.** A plain `flags[j]` is a normal
  load: the compiler may hoist it out of the loop entirely (it has no reason to
  believe another thread writes it), and even if it does not, it can be served
  from this SM's L1 forever, which is not coherent with the writer's SM. The
  atomic executes at L2 and is re-issued every iteration. `cuda::atomic_ref<u32,
  cuda::thread_scope_device>::load(cuda::memory_order_acquire)` from M9 is the
  modern spelling and generates equivalent code; `volatile` is not an acceptable
  substitute for either.
- **`__threadfence()` after the flag read, before the payload read.** This is
  the *acquire* half. Without it the payload load may be issued before the flag
  load resolves — the hardware is free to reorder two independent loads — and
  you read a stale aggregate for a tile you have just seen publish.
- **`f = FLAG_P; v = 0` for `j < 0`.** Running off the front of the array is
  treated as "found a prefix of value 0", which collapses two loop exits into
  one. It is also *correct* rather than merely convenient: the exclusive prefix
  of tile 0 is the identity.
- **`c = (lane <= firstP) ? v : 0`.** Lanes above `firstP` are already summarised
  by `pfxs[firstP]`, so including them would double-count. Lanes below `firstP`
  contribute aggregates. Lane `firstP` contributes an inclusive prefix. The sum
  over `0..firstP` is exactly this window's contribution.
- **`__shfl_sync(…, c, 0)` after the reduction.** `__shfl_down_sync` leaves the
  total in lane 0 only (M12); every lane needs it because every lane's `excl`
  must agree for the next window's arithmetic and for the `break`.
- **`if (pmask) break;`** — if any lane found a prefix, we are done. Otherwise we
  consumed 32 aggregates and move back 32.

**The spin is divergent-safe.** Each lane spins on its own `j` independently and
they exit at different times; under independent thread scheduling (M8) that is
fine, and the subsequent `__ballot_sync(0xffffffff, …)` re-converges them. Under
the pre-Volta model this loop would have deadlocked — which is the positive side
of ITS that M8 mentioned and this is the first kernel in the course that
actually needs it.

**Wrong approaches and their symptoms.**

| attempt | symptom |
|---|---|
| single-threaded look-back (`if (tid == 0)` walking back one tile at a time) | **correct**, and measurably slower. Try it: the look-back becomes ~32× more L2 round trips deep in the worst case. |
| no `__threadfence()` on the reader side | passes every test on this GPU. The reason is M9's honest note: Ada's load/store unit generally does not reorder these two dependent-looking loads in practice. It is still wrong, and the only evidence is the absence of `MEMBAR.SC.GPU` in the SASS. |
| plain `flags[j]` load in the spin | undefined and unusable. The compiler has no reason to believe another thread writes `flags[j]`, so it is free to hoist the load out of the loop (giving an infinite spin) and the SM is free to satisfy it from a non-coherent L1 line forever. Do not test this by running it; read the SASS instead and look for whether the load is inside the loop body at all. |
| `while (f != FLAG_P)` instead of `while (f == FLAG_X)` | correct but catastrophically slow: every tile waits for its immediate predecessor's *prefix*, which serialises all 65,536 tiles. This is the "chained scan" that decoupled look-back exists to replace. |
| forgetting `__shfl_sync` to broadcast the reduction | only lane 0 has the right `excl`; lane 0 writes `s_excl` so the answer is right, but the `break` condition and any per-lane use go wrong the moment the loop runs twice. |

## TODO 4 — publishing the inclusive prefix

```cpp
if (lane == 0) {
    pfxs[tile] = excl + total;
    __threadfence();
    atomicExch(&flags[tile], FLAG_P);
    s_excl = excl;
}
```

Omitting this is **not a correctness bug** — every tile would still compute the
right answer by walking further back — and that is what makes it interesting.
What it costs is asymptotic: tile `k` would inspect `k` predecessors instead of
`O(1)`, and the total look-back work would be `O(m²)` instead of `O(m)`. At
m = 65,536 the kernel effectively stops finishing. This is the difference between
"decoupled look-back" and "chained scan", and it is one store.

Note that `s_excl` is written by lane 0 and read by the whole block after the
`__syncthreads()` that follows the `if (tid < 32)` region. That barrier supplies
both of M9's guarantees; no fence is needed for it because shared memory is
block-scoped.

## TODO 5 — the prediction

`PREDICT_BUCKET 2` (1.5× .. 2.0×). Measured **1.85×**.

The ledger predicts 4N/2N = 2.0×. We fall slightly short, and the reason is the
look-back: it is a serialised L2 round trip on the critical path of every tile,
it cannot be prefetched (the data does not exist until the tile ahead publishes),
and the fences cost memory-pipeline time. A reader who predicted bucket 1
(≥ 2.0×) reasoned only from traffic; a reader who predicted bucket 3 over-charged
the look-back.

---

## Synchronization / memory reasoning

This kernel contains every synchronization mechanism in the course except
cooperative groups, and each one is doing a different job:

| mechanism | scope | job |
|---|---|---|
| `atomicAdd(ticket, 1)` | device | unique tile claim; the forward-progress precondition |
| `__syncthreads()` (5 of them) | block | publish `s_tile`/`s_excl`, and the tile scan's internal barriers |
| `__threadfence()` (writer) | device | release: payload to L2 before flag |
| `atomicExch(flag, …)` | device | the flag store itself: atomic, uncacheable, single-copy |
| `atomicAdd(flag, 0)` | device | the flag load: forces an L2 read every spin iteration |
| `__threadfence()` (reader) | device | acquire: flag before payload |
| `__ballot_sync` / `__shfl_*_sync` | warp | the look-back's own reduction, no memory involved |

The single most common way to get this wrong is to supply the *waiting* (the
spin) and forget the *ordering* (the fences), or vice versa. M9's framing —
a fence makes nobody wait, a barrier publishes nothing — is the checklist.

`compute-sanitizer --tool memcheck` is clean over the full 18-size stress sweep
(the only line it emits is the benign
`CUDA API Warning: Resetting device while there are still other users claiming
to use it` at `cudaDeviceReset`, which is a sanitizer artifact, not a memory
error):

```
> compute-sanitizer --tool memcheck stress.exe 1
========= COMPUTE-SANITIZER
rep 0 ok (launches so far 18)
STRESS DONE: 18 dlbScan launches over 18 sizes x 1 reps, fails=0
```

`--tool racecheck` cannot help here at all: by its own documentation it is a
*shared-memory* hazard detector, and M10 verified that it is blind to global
read-modify-write patterns. Every cross-block interaction in this kernel is in
global memory through atomics. **There is no tool on this machine that will
validate the fence placement** — `ncu` would not either, and it is unavailable
here (`ERR_NVGPUCTRPERM`, spec §12). The evidence is the SASS plus the stress run.

SASS evidence, from `nvcc -arch=sm_89 -O3 -cubin` + `cuobjdump -sass`, counted
inside `dlbScanKernel`:

```
      2 ATOMG.E.ADD.STRONG.GPU      <- the ticket, and the spin's flag load
      3 ATOMG.E.EXCH.STRONG.GPU     <- the three flag publishes
      4 MEMBAR.SC.GPU               <- two release fences, two acquire fences
```

Four `MEMBAR.SC.GPU`, exactly the four `__threadfence()` calls in the source,
and every atomic carries `.STRONG.GPU` — device scope, executed at L2.

---

## Performance reasoning and the stress evidence

```
N = 67108861, 65536 tiles
strategy                          ms    GB/s (2N) GB/s (real)   % of 432   valid
three-kernel (4N traffic)     3.2163        166.9      333.8      77.3%    PASS
single-pass  (2N traffic)     1.7350        309.4      309.4      71.6%    PASS
  2N floor at 432 GB/s = 1.2428 ms
```

Read both bandwidth columns. The three-kernel version reaches **77 % of DRAM
peak** and the single-pass version only **72 %** — the single-pass kernel is the
*less* efficient user of the memory system, and it is still 1.85× faster,
because it uses half as much of it. This is the central engineering point of the
module and the reason "% of peak" alone is a misleading metric for anything but
a pure streaming kernel.

`example02.cu` puts `cub::DeviceScan` next to this: CUB reaches 86.4 % of peak on
the same 2N, i.e. 1.18× faster than this implementation. The gap is a bigger
tile, vectorized loads, and a look-back that starts before the block scan
finishes. Closing it is not a matter of a better algorithm.

**How hard this was stressed.** Beyond the 9-size sweep and 200 repeats the
program itself runs, the kernel was run standalone **7,200 times across 18 grid
sizes** (1, 2, 1023, 1024, 1025, 2047, 40959, 40960, 40961, 245760, 245761,
1048573, 1048576, 4194301, 16777213, 33554432, 67108859, 69999999 elements —
i.e. 1 to 68,360 tiles, including exactly one wave, one-more-than-a-wave, and
several primes), with full host-side validation on every 10th repetition:
**0 wrong answers, 0 hangs, 0 timeouts.** Sizes below 32 tiles are the ones that
exercise the `j < 0` path in the look-back; sizes just above a wave
(240 blocks × 1024 = 245,760 elements) are the ones that would expose a
scheduling-order assumption.

---

## Expected output

```
Module 13 / Exercise 3 — single-pass scan by decoupled look-back

correctness sweep (single pass) over 9 sizes:
   n=1         m=1      PASS
   n=1023      m=1      PASS
   n=1024      m=1      PASS
   n=1025      m=2      PASS
   n=40959     m=40     PASS
   n=245761    m=241    PASS
   n=1048573   m=1024   PASS
   n=4194301   m=4096   PASS
   n=67108861  m=65536  PASS

200 back-to-back single-pass launches at n=4194301: 0 bad results, no hangs

N = 67108861, 65536 tiles
strategy                          ms    GB/s (2N) GB/s (real)   % of 432   valid
three-kernel (4N traffic)     3.2163        166.9      333.8      77.3%    PASS
single-pass  (2N traffic)     1.7350        309.4      309.4      71.6%    PASS
  2N floor at 432 GB/s = 1.2428 ms

  three-kernel / single-pass = 1.85x -> bucket 2   predicted 2

  score: 8/8
OVERALL: PASS
```

Run-to-run: the single-pass time has been observed between 1.70 and 1.94 ms and
the three-kernel between 2.94 and 3.23 ms; the ratio has measured **1.66–1.85x**
across five runs, inside bucket 2 (1.5–2.0x) every time. It never reaches the
2.0x the traffic ledger predicts, which is the look-back's price.

---

## The result that matters

**A single-pass scan is 1.85× faster than a three-kernel scan while being a
*worse* user of DRAM bandwidth, because the only thing that matters is total
bytes moved.** And the price of that speedup is the only cross-block wait in the
entire course — a construction that is safe for a reason you have to be able to
state in three sentences (tiles claim indices dynamically, so every tile you wait
on is already resident; resident blocks make progress; tile 0 never waits) and
that rests on a forward-progress property nobody wrote down.

**Variation to try:** delete the `atomicExch(&flags[tile], FLAG_P)` in TODO 4 and
run the correctness sweep only, with the 64 M case removed. It still passes. Then
put the 64 M case back and watch a correct algorithm fail to terminate in any
useful time. One store is the difference between O(m) and O(m²), and no test
short of a large one will tell you.

Second variation: replace the three state words with CUB's single packed word
(flag in the top 2 bits, value in the low 30) written and read with a single
32-bit atomic, and delete both `__threadfence()` calls on the reader side. Argue
why that is still correct before you measure it.
