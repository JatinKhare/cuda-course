# Module 13 — Check Your Understanding, answers

---

## Q1

> You implement Hillis–Steele in shared memory with `sdata[i] += sdata[i-off]`
> and a `__syncthreads()` after it. It is correct with a 32-thread block and
> wrong with 256. Which hazard is it, why does the barrier not fix it, why did
> 32 threads pass, and which `compute-sanitizer` tool would have caught it at 32
> threads?

**The hazard is write-after-read (WAR), across threads, within one step.**
Thread `i` reads `sdata[i-off]`; thread `i-off` writes `sdata[i-off]`. Both
happen inside the single statement `sdata[i] += sdata[i-off]`, so they are not
separated by anything. If thread `i-off`'s write lands first, thread `i` adds a
value that has already absorbed `sdata[i-2*off]`, and the result is too large by
that amount.

**Why the barrier does not fix it.** M9's two guarantees, G1 (execution barrier)
and G2 (block-scope memory fence), both order things *across* the barrier. A
barrier placed after the statement correctly orders step `k`'s writes against
step `k+1`'s reads. The race is between two threads *inside* step `k`, on the
same side of both barriers. To fix it you need either two barriers and a
register temporary (read-all, barrier, write-all), or two buffers so the read
set and write set are disjoint addresses.

**Why 32 threads passed.** A 32-thread block is one warp. The warp issues
`sdata[i] += sdata[i-off]` as a single instruction: all 32 lanes perform their
loads, then all 32 perform their stores, because that is what one instruction
means under SIMT (M1, M8). There is no interleaving *because there is no second
instruction stream*. M6 stated this exact warning — "works at warp width is not
evidence of correctness" — and M8 measured a converged-region warp-synchronous
kernel producing 0 errors in 524,288 elements while being formally wrong.

Two further points worth having. First, this is an accident of the code
generator, not a guarantee: under independent thread scheduling (M8) the hardware
is permitted to interleave the two halves of a divergent region, and nothing
promises the load and store phases of one instruction stay adjacent forever.
Second, once the block is 64 threads the two warps are scheduled independently by
two different warp schedulers (M1: one per processing block), and the race is
real.

**The tool: `compute-sanitizer --tool racecheck`.** It is a shared-memory hazard
detector, which is exactly what this is, and M9's Exercise 1 already demonstrated
the key property — it reports hazards on kernels whose *output is bit-exact
correct*, because it analyses the access pattern rather than the result. It
would flag the 32-thread version. `memcheck` would not: no access is out of
bounds. This is the same asymmetry M6's Exercise 3 taught.

---

## Q2

> Blelloch's downsweep is preceded by `x[N-1] = 0`. (a) If you forget the line
> entirely, describe the exact output. (b) Why would a test checking
> `out[N-1] + x[N-1] == total` still pass? (c) If you zero it but forget to save
> the total first, what breaks, and where does it show in a *multi-block* scan?

**(a)** The downsweep's invariant is "node `v` holds the sum of everything
strictly left of `v`'s range," and the root's value is what seeds it. Leaving
`total` at the root seeds every leaf with an extra `total`, and since the
downsweep only ever adds and copies the seed downward, the error is uniform:

```
out[i] = correct[i] + total       for every i
```

In particular `out[0] = total` instead of `0`. The array is still monotone
non-decreasing, still has the right *differences* between adjacent entries, and
looks entirely plausible. Only the absolute values are wrong.

**(b)** That test checks `out[N-1] + x[N-1] == total`, i.e. that the recovered
count is right. With the uniform error, `out[N-1] = correct[N-1] + total`, so the
check computes `correct[N-1] + x[N-1] + total = 2*total`, which fails — *unless*
the test compares against the total it recovered the same broken way, which is
the realistic case: a harness that derives `total` from the scan's own output
rather than from an independent CPU sum will be self-consistent and blind. The
lesson is that a scan must be validated element-wise against an independent
reference, never against a quantity derived from itself.

**(c)** Zeroing without saving means the function returns 0 as the tile total.
In a **single-block** scan that value is usually discarded, so the kernel appears
completely correct and every test passes. In a **multi-block** scan the tile
total is the entire interface between pass 1 and pass 2: every block reports 0,
pass 2 scans an array of zeros and produces an array of zeros, and pass 3 adds 0
to every tile. The result is that **tile 0 is correct and all other tiles are
scanned only within themselves** — each tile restarts from zero. The symptom is a
sawtooth: the output rises within each 1024-element window and drops back at
every window boundary. It is also the bug that a small test (N ≤ TILE) cannot
find, which is why Exercise 1 validates at 1,048,573 and not at 1,024.

---

## Q3

> At 64 M the three-kernel scan runs at 84.6 % of peak and the look-back at
> 73.3 %, yet the look-back is 1.73× faster. Explain. Then: if the spin were
> free, what is the maximum further speedup over CUB, and what would limit it?

**How both are true.** "% of peak" is `bytes actually moved / time / 432 GB/s`.
It measures how efficiently a kernel uses the memory system, not how much work it
gets done. The three-kernel version moves 4N bytes at 84.6 % efficiency; the
look-back moves 2N at 73.3 %. Time is `bytes / (efficiency × peak)`:

```
three-kernel : 4N / (0.846 × 432) 
look-back    : 2N / (0.733 × 432)
ratio        = (4/0.846) / (2/0.733) = 4.728 / 2.729 = 1.73×
```

Halving the traffic beats losing 13 % of efficiency, and it is not close. The
general form: **a kernel's time is set by the product of how much it moves and
how well it moves it, and the first factor is usually the one you can change by
a factor of two.** M6 delivered the same lesson from the other side — a kernel
already at 72.6 % of peak has a hard 1.38× ceiling from on-chip optimization
alone, because there is nothing left to win on the second factor.

**If the spin were free.** The look-back would move 2N at the best streaming
efficiency this part achieves, which M6/M11 measured at 87–88 % (375–381 GB/s).
That puts it at about 1.41 ms, against CUB's measured 1.4389 ms — i.e. **CUB is
already essentially there, and the maximum remaining gain over CUB is ~2 %.**
The limit is the DRAM pipe itself: at 2N of traffic the floor is 1.2428 ms, and
no scan can beat it. This is the answer that matters: once you are at 2N and at
the streaming roof, the scan problem is closed, and any further work has to
change the problem (fuse the producer or the consumer into the scan so the array
never goes to DRAM at all — which is exactly what Exercise 2's fused compaction
does, and what a real pipeline does).

---

## Q4

> The look-back uses `atomicAdd(ticket, 1)` rather than `blockIdx.x`. M9 proved
> a block spinning on another block deadlocks. Give the interleaving under which
> the `blockIdx.x` version deadlocks, explain why the ticket version cannot
> reach it, and identify the assumption the ticket argument still makes that is
> not in the programming guide.

**The deadlocking interleaving.** Let the grid be 65,536 blocks and let the
device hold 240 co-resident blocks (M9 computed this for 256-thread blocks:
6 blocks/SM × 40 SMs). Suppose the GigaThread engine places blocks
`{5000, 5001, …, 5239}` first — nothing forbids this; placement order is
unspecified. Every one of those blocks computes its tile aggregate, publishes
`FLAG_A`, and then spins waiting for tile 4999 to publish. Tile 4999 is owned by
block 4999, which is still in the launch queue. A queued block is not a running
thing (M1); it becomes one only when a resident block retires and frees a slot.
No resident block will retire, because all 240 are spinning. **Deadlock, and it
is structural, not a fairness problem.**

Note that the same argument applies with only *two* blocks out of order, as long
as the higher-numbered one is resident and the lower-numbered one is not and the
machine is full.

**Why the ticket version cannot reach that state.** The block's first action is
`atomicAdd(ticket, 1)`. If a block holds ticket `t`, then tickets `0 … t−1` were
already returned, which means for each of them there exists a block that had
*already executed an instruction* — so it was already placed on an SM. From M1, a
placed block is indivisible, never migrates, and runs to completion. So every
tile a block can possibly wait on is owned by an already-resident block. Tile 0
never waits. Induction on `t`: tile 0 publishes `FLAG_P`; if tiles `0 … t−1` all
eventually publish, tile `t`'s look-back terminates and it publishes too.

The state "resident block waiting on a queued block" is unreachable, because
holding any ticket at all implies every smaller ticket was handed to a block
that is already running.

**The remaining assumption.** The induction step requires that a resident block
*makes progress* — that the warp scheduler will eventually issue instructions to
a spinning warp's siblings and to the warps of every resident block. That is a
**forward-progress guarantee for resident blocks**, and the CUDA programming
guide does not state it. What the guide does give you is that blocks do not
migrate and do not get preempted at block granularity; what it does not give you
is that a resident warp is never starved indefinitely.

In practice every NVIDIA GPU round-robins eligible warps and this holds, which is
why CUB's `DeviceScan` — used by essentially every CUDA application on the
planet — is built on exactly this construction. But it is an implementation
property, and the correct engineering posture is: write the argument down, label
it **ARCHITECTURE-SPECIFIC / IMPLEMENTATION-DEPENDENT**, and stress-test it
(this module's kernel was run 7,200+ times across 18 grid shapes from 1 tile to
68,360 tiles without a hang). A second, weaker assumption worth naming: the
argument also assumes the atomic counter is not reordered with respect to the
block's later loads in a way that would let a block observe a ticket before its
predecessors' atomics are globally ordered — which holds because `atomicAdd`
executes at the L2 (M10) and is therefore already a device-scope serialization
point for that address.
