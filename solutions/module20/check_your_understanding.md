# Module 20 — Check Your Understanding, answers

---

## Q1

> A kernel runs at 45% of the FP32 ceiling at 100% occupancy. Its inner loop is
> a chain of dependent `FFMA`s. Your colleague proposes raising occupancy by
> reducing the block size. Using the numbers in §1 and §2, state what will
> happen and why, and state the one measurement you would make first to confirm
> your reasoning without a profiler.

**Nothing will happen, and the proposal is also incoherent.**

Incoherent first: the kernel is *already* at 100% occupancy. There is no
occupancy left to raise, and reducing the block size cannot raise it above
1536 threads per SM. (If the block size were already tiny, reducing it further
could *lower* occupancy by hitting the 24-blocks-per-SM cap — Module 19.)

Nothing will happen, second, for a reason that would hold even if there were
headroom. 100% occupancy is 48 warps per SM = **12 warps per scheduler**.
Little's Law says the FP32 pipe needs `latency × throughput = 4 cycles ×
1 instr/cycle = 4` independent instructions in flight. The kernel already
supplies 12 — **three times the requirement**. Example 1's table shows the
whole `warps/scheduler ≥ 4` region flat to within a few percent regardless of
ILP. The dependent chain is already fully covered by TLP; it is not the reason
the kernel is at 45%.

So where does the other 55% go? Not to `execution dependency`. Candidates, in
the order to check them: the kernel's instruction mix is not pure FFMA (every
non-FFMA instruction is an issue slot that produces no FLOPs — Module 18's 90.8%
FFMA density for a *good* GEMM is the yardstick), the loop overhead is a large
fraction of a short body, or there is memory traffic you are not counting.

**The one measurement.** Run the same kernel at half the occupancy — 6 warps
per scheduler. If the time is unchanged, concurrency is not the binding
constraint and the colleague's lever is the wrong one; this takes one extra
launch and no profiler. (If the time *halves* in the good direction you have
discovered Module 18's occupancy inversion, which is a different and more
interesting problem.) The follow-up measurement is `cuobjdump -sass`: count
`FFMA` as a fraction of the inner-loop instruction count. 45% of the ceiling
with an FFMA density of 45% is a kernel that is running perfectly and simply
has other work to do.

---

## Q2

> Kernel P's dominant stall reason is `long scoreboard`; kernel Q's is
> `not selected`. One will get faster if you double its occupancy, one will get
> slower, and one of them is already at a hardware limit. Assign the outcomes.

Note the question has three outcomes for two kernels; one kernel collects two
of them.

**Kernel P (`long scoreboard`) gets faster.** `long scoreboard` means warps are
waiting on global or local memory returns. That is the stall occupancy exists
to fill: each additional resident warp contributes its own outstanding
requests, and Little's Law says the achieved bandwidth rises linearly with
bytes in flight until the bus saturates. Example 2's left column: 252 → 389 →
409 GB/s for 1 → 2 → 4 warps per scheduler at MLP 1.

The caveat that makes this a real answer rather than a slogan: P gets faster
*only while it is below the knee*. Once bytes in flight reach `bandwidth ×
latency` (≈120 KB here) P is at the second outcome — **already at a hardware
limit** — and doubling occupancy does nothing at all. So P collects two of the
three outcomes depending on where it sits, and the way to tell is to compute
its concurrency, not to look at the stall histogram.

**Kernel Q (`not selected`) gets slower.** `not selected` means the warp was
*eligible* and lost the arbitration: another warp on the same scheduler issued
instead. A scheduler issues at most one instruction per clock, so the total
issue capacity is fixed; adding warps cannot create issue slots, it only
increases the number of warps competing for the same ones. The `not selected`
fraction rises mechanically. Worse, the extra warps cost registers and
(usually) shared memory, which on a real kernel means `ptxas` fits into a
smaller register budget and starts spilling — Module 18 measured 18.7× from
exactly that mechanism.

**The mechanism in one sentence each.** P is stalled because the data is not
there; more requesters bring more data in flight. Q is not stalled at all in
any useful sense; more requesters bring more queueing. The fix for Q is to
execute fewer instructions per unit of work — Exercise 2's kernel D, where
replacing the IEEE divide with `__fdividef` cuts the function from 440 to 184
SASS instructions and is worth 2.03×, while going from 4 to 12 warps per
scheduler is worth 1.11×.

---

## Q3

> Module 18: 33% occupancy 18.7× faster than 100%. Module 1: a pointer chase
> 4.87× faster at 24 warps than at 1. Explain with one equation why these are
> not in conflict, and predict which kernel is hurt more by doubling DRAM
> latency at constant bandwidth.

**The equation is `concurrency = throughput × latency`, and the resolution is
that occupancy is not concurrency — it is one of two terms that produce it.**

Write the supply side as `concurrency ≈ warps_resident × (work in flight per
warp)`. The two modules sit at opposite corners:

- **Module 1's pointer chase** has MLP = 1 *by construction*: the address of
  step `s+1` is the value returned by step `s`. The per-warp term is pinned at
  its minimum, so the only way to raise concurrency is to raise the warp count,
  and throughput rises with occupancy until the memory system saturates — which
  it does at ~6 warps per SM, after which the remaining 42 warp slots buy 4%.
- **Module 18's register-tiled GEMM** has an enormous per-warp term: 32–64
  independent accumulator chains per thread. It is past the concurrency
  requirement with **one block per SM**. Occupancy therefore contributes
  nothing — and the registers that would buy it are the same registers holding
  the accumulators, so forcing occupancy up forces the accumulators into local
  memory, which Module 4 established is DRAM. The kernel then acquires the
  `long scoreboard` stalls it never had, in its innermost loop.

In the language of Example 1's table: Module 1's kernel lives in the left
column, where the vertical axis is everything. Module 18's lives in the right
column, where the vertical axis is flat and the cost of moving down it is
paid in the horizontal one. **Occupancy is worth exactly what it adds to the
product, and nothing else.**

**Doubling DRAM latency at constant bandwidth.** The requirement
`concurrency = bandwidth × latency` doubles, so every memory-bound kernel needs
twice the bytes in flight.

- Module 1's pointer chase is hurt **in proportion**: at 1 warp per SM its
  throughput is `1/latency` and halves exactly. At 24 warps it is near its
  plateau, so it gets a partial reprieve — but the plateau itself is a
  concurrency plateau and moves right, so the knee that was at 6 warps moves to
  12 and the achievable throughput at any fixed occupancy below 12 falls.
- Module 18's GEMM is hurt **hardly at all**. Its working set is L2-resident
  (the module measured double buffering as a *loss* for exactly this reason —
  the latency it was hiding was a 241-cycle L2 hit, not a DRAM one), its
  arithmetic intensity is high, and at 46% of the FP32 ceiling it is
  compute-bound, not memory-bound. A longer DRAM latency changes a term that
  does not dominate.

So the answer is Module 1's kernel, and the general rule is that a latency
increase hurts in proportion to how much of your runtime is latency you are
failing to cover.

---

## Q4

> One `float` load per thread in flight, 5120 threads, 68 GB/s. Predict the
> bandwidth at 20,480 threads with the same body, and at 5120 threads with four
> hoisted `float4` loads. State the assumption and when it fails.

**Set up the quantities.** Concurrency in bytes:

```
Q = threads × loads_in_flight × bytes_per_lane
Q0 = 5120 × 1 × 4 = 20,480 B
```

Little's Law inverted gives the effective latency:

```
L = Q0 / B0 = 20,480 B / 68e9 B/s = 301 ns   (≈ 572 cycles at 1.9 GHz)
```

**At 20,480 threads:** `Q = 81,920 B`, so `B = Q/L = 81,920 / 301 ns =
272 GB/s`. Four times the threads, four times the bandwidth — the relation is
linear because nothing else changed.

**At 5120 threads with four hoisted `float4` loads:** `Q = 5120 × 4 × 16 =
327,680 B`, so `B = 327,680 / 301 ns = 1089 GB/s` — which is **impossible**, and
that impossibility is the answer. The prediction is `min(Q/L, B_max)` and
`B_max` is about 411 GB/s measured (432 GB/s at the pins). The second
configuration is past the knee and lands at the ceiling. Exercise 3 measures
409.5 GB/s.

**The assumption.** Both predictions assume `L` is a *constant* — that the
latency of a request does not depend on how many requests are outstanding. That
is the defining assumption of the latency-bound regime and it is what makes
Little's Law a prediction rather than a tautology.

**When it fails — two ways, both measured in this course.**

1. **Above the knee.** Once `Q ≥ B_max × L` the system is throughput-bound,
   requests start queueing, and `L` grows to keep `Q/L = B_max`. Module 1's
   pointer chase shows this directly: per-step latency rises from 217 ns at 1
   warp to 1070 ns at 24, a factor of 4.9 — exactly the factor by which
   throughput stopped improving. The first prediction (272 GB/s) is below the
   ceiling so it is safe; the second is not, which is why it must be clamped.
2. **Below the knee, if the access pattern changes.** `L` is a property of the
   access pattern as well as of the DRAM: 40.5 cycles from L1, 241.3 from L2,
   575 from DRAM (Module 4), and a scattered pattern that touches 32 sectors
   per warp instruction behaves differently from a coalesced one that touches
   4. Converting scalar loads to `float4` does not change `L` — Module 5 proved
   the sector count is identical — which is the reason the second prediction is
   allowed to use the same `L` at all.
