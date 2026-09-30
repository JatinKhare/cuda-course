# Module 01 — Check Your Understanding (answers)

**Do not read this until you have written your own answers.**

---

## Q1

> Your GPU can keep 1536 threads resident per SM, but each SM has only 4 warp
> schedulers issuing at most one instruction each per cycle. So at most 128
> threads make progress per SM per cycle. What is the point of keeping the other
> 1408 thread contexts resident? Be precise about what resource they consume
> while idle.

The 1408 idle threads are **inventory for the scheduler**. They exist so that in
any cycle where the 4 warps currently issuing go stalled, some other warp is
already eligible and the issue slot is not lost.

Restate it in the units the hardware works in: 1536 threads is 48 warps, 12 per
processing block. Each processing block's scheduler examines its ≤12 resident
warps every clock, partitions them into **eligible** (next instruction's operands
ready, functional unit port free) and **stalled** (waiting on a memory return, a
long-latency ALU result, a barrier, an i-cache miss), and issues exactly one
eligible warp's instruction. The other 11 are not wasted capacity; they are the
probability that the scheduler finds *something* to issue.

The arithmetic: a global load costs ~400–500 cycles. A warp that issues one and
then needs the result is unavailable for ~450 cycles. To keep one scheduler
issuing every cycle from a pool of warps that each stall for 450 cycles after
every load, you need enough warps that their stall windows overlap. Little's Law
again, in cycles instead of bytes.

**What the idle contexts consume:**

- **Registers.** Each resident warp holds a private, disjoint slice of its
  processing block's 16,384-register file, allocated when its block was placed
  and held until the warp exits. This is the expensive resource, and it is what
  usually caps residency below 48 warps.
- **A warp slot** in the scheduler's table (12 per processing block) and, at
  block granularity, **a block slot** (24 per SM).
- **Shared memory**, at block granularity, out of the SM's carve-out.
- **A share of the L1 / unified cache**, which is not partitioned — more resident
  threads means less cache per thread, and this is a real cost, not a bookkeeping
  one.

**What they do not consume:** any issue bandwidth. A stalled warp is examined by
the scheduler and skipped; the examination is free. This is the whole reason the
GPU can afford to keep 48 of them around. It is also the precise sense in which a
GPU context switch is zero-cost: nothing is saved or restored because nothing is
shared — warp 7's registers are different physical storage from warp 8's, and
switching between them is a change of an index in the operand-fetch path.

The trade to remember: registers-per-thread and warps-per-SM are two views of one
budget. 65,536 registers per SM ÷ 48 warps ÷ 32 lanes = 42 registers per thread
if you want full residency. Spend 64 and you can host at most 32 warps, having
given up a third of your latency-hiding inventory before writing any algorithm.

---

## Q2

> A CPU hides a DRAM miss with out-of-order execution and prefetching. A GPU
> hides it differently. Name the mechanism, and state the one thing the
> programmer must supply for it to work.

**Mechanism: zero-cost warp switching by the SM's warp schedulers — thread-level
parallelism used as latency tolerance.** When a warp issues a load and its next
instruction depends on the result, the scoreboard marks the destination register
busy, the warp becomes ineligible, and the scheduler simply picks a different
resident warp next cycle. No state is saved, nothing is speculated, nothing is
replayed. When the data returns, the register is cleared and the warp re-enters
the eligible pool.

Note what is *absent*: there is no out-of-order window, no branch prediction worth
the name, and no hardware prefetcher of consequence. The GPU makes no attempt to
shorten the latency of the individual access — `example01.cu` shows the per-access
latency pinned at ~220 ns and getting *worse*, never better, as you add warps.

**What the programmer must supply: enough independent work in flight** —
concretely, enough resident warps (or, failing that, enough independent memory
operations per thread, which is ILP; Module 20) that at every cycle some warp is
eligible.

"Enough" is a number, and Little's Law gives it: `concurrency = throughput x
latency`. To saturate 432 GB/s at ~450 ns of latency you need ~190 KB of requests
in flight, ~4.8 KB per SM. How many warps that is depends entirely on how many
bytes each warp instruction asks for — a fully coalesced 32-lane float load asks
for 128 bytes, a fully scattered one asks for 1024 bytes. That is why
`example01.cu`'s pathological scattered chase saturates at only ~6 warps per SM
while a well-behaved streaming kernel needs far more. There is no universal
occupancy target; there is only bytes-in-flight.

A one-line version: *a CPU hides latency with cleverness; a GPU hides it with
inventory, and you have to stock the shelves.*

---

## Q3

> `__syncthreads()` synchronizes a block but there is no equally cheap
> `__syncgrid()`. Explain why, using the block-scheduling model above.

Because **a barrier can only be implemented among participants that are
guaranteed to be simultaneously resident, and only blocks carry that
guarantee.**

The block-level case. When the GigaThread engine places a block, it places the
whole block on one SM, and the block stays there until its last thread exits — it
is indivisible and it never migrates. Therefore every warp of the block is
resident on the same SM for the entire lifetime of the block. A barrier over that
set is implementable in hardware with a counter in the SM: each arriving warp
increments it and goes ineligible; when the count reaches the block's warp count
the SM marks them all eligible again. Cost: a few cycles plus whatever imbalance
you created. No memory traffic, no driver involvement.

The grid-level case. A grid of `gridDim` blocks is *not* guaranteed to be
resident all at once. If `gridDim > blocksPerSM x numSMs`, some blocks provably
have not started when others are already running — that is the definition of a
second wave. A naive `__syncgrid()` would then deadlock: the resident blocks wait
at the barrier, never exit, never free their block slots, so the unstarted blocks
can never be dispatched to arrive at the barrier. The blocks holding the machine
are waiting for blocks that are waiting for the machine.

This is not an implementation shortcoming; it is what buys the GPU its
scalability. Because blocks are independent and need not co-reside, the same
binary runs on a 4-SM part and a 144-SM part, and the work distributor can use
whatever SMs happen to be free. Requiring grid-wide co-residency would throw that
away.

What does exist, and at what price:

- **Kernel boundaries** are the ordinary grid-wide barrier. The end of a kernel
  is a point at which all threads have completed and all writes are visible to
  the next kernel. Cost: a launch (microseconds), plus draining and refilling the
  machine.
- **Cooperative groups' `grid.sync()`** (Module 29) exists, but only via a
  cooperative launch, and that launch *fails* unless the grid is small enough to
  be co-resident — i.e. it forces you to launch at most one wave. You do not get
  a free grid barrier; you get an API that enforces the precondition that would
  make one safe, at the cost of capping your grid size.
- Hand-rolled spin barriers on global memory have exactly the deadlock above and
  are a classic way to hang a GPU.

The general principle, which recurs through Modules 9, 10 and 29: **the scope of
a synchronization primitive is bounded by the scope of a residency guarantee.**
Warp < block < grid, with hardware barriers at the first two and only a kernel
boundary at the third.

---

## Q4

> A kernel launches 41 blocks of 256 threads on this 40-SM GPU. Assume each SM
> could hold several such blocks. How many waves execute, and roughly what
> fraction of the GPU's issue capacity is wasted? State your assumption about
> block-to-SM assignment.

**The answer depends entirely on `blocksPerSM`, and that is the point of the
question.** "Assume each SM could hold several such blocks" is the load-bearing
clause, and it changes the answer from the one most people give.

**The naive reading (one block per SM), which is what the question is baiting.**
If only one block fit per SM, a wave would be 40 blocks,
`waves = ceil(41/40) = 2`, and wave efficiency would be `41 / (2 x 40) = 51.2%`.
Wave 1 uses all 40 SMs; wave 2 uses 1 SM and idles 39. Roughly **49% of the
machine-time is wasted**, and the launch takes twice as long as a 40-block launch
would — which is exactly the 1.99x measured in Exercise 2, where `THREADS = 1024`
does force `blocksPerSM = 1`.

**The correct reading for 256-thread blocks.** 256 threads is 8 warps. The
binding limit on sm_89 is 1536 threads per SM, so `blocksPerSM = 6` (the 24-block
slot limit and, for any reasonable register count, the register file, are not
binding). Then:

```
blocksPerWave = 6 x 40 = 240
waves         = ceil(41 / 240) = 1
```

**One wave.** All 41 blocks are resident simultaneously and there is no tail at
all. But the machine is enormously under-filled: 41 of 240 block slots, i.e.
`41 x 256 = 10,496` threads out of a possible 61,440. In terms of resident-block
capacity the launch uses **17%** of the machine, wasting ~83%.

The waste has a different shape in the two readings, and the distinction matters:

- With `blocksPerSM = 1` the waste is **temporal** — the machine is fully engaged
  for half the time and 1/40 engaged for the other half.
- With `blocksPerSM = 6` the waste is **spatial and permanent** — 41 blocks
  spread over 40 SMs means one SM gets 2 blocks and 39 get 1 (see below), so
  every SM is running at 1/6 to 2/6 of its block capacity for the whole kernel,
  and the SMs are also nowhere near enough warps to hide memory latency.

**Assumption about block-to-SM assignment.** I assume the work distributor fills
breadth-first: it hands out one block to each SM in turn before giving any SM a
second block. So 41 blocks land as 40 SMs with 1 block and 1 SM with 2 blocks
(with `blocksPerSM = 6`); or 40 SMs with 1 block, then one straggler after a slot
frees (with `blocksPerSM = 1`). This is what Exercise 1 Part B observes
empirically via `%smid`: 80 blocks on 40 SMs came out as exactly 2 per SM, not 24
stacked on the first few. NVIDIA does not contractually specify the policy, so a
correct answer must state the assumption rather than assert it — and a program
must never depend on it for correctness, only reason about it for performance.

**The lesson.** A block count is meaningless without `blocksPerSM`. "41 blocks on
40 SMs" sounds like a near-perfect fit and is, in fact, either a 2x slowdown or
an 83%-idle machine, depending on a number you have to ask the runtime for.
