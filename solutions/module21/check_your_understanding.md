# Module 21 — Check Your Understanding, answers

---

## Q1

> A colleague profiles a kernel, computes its compulsory DRAM arithmetic
> intensity as 200 FLOP/byte, observes that the machine balance is 45, and
> concludes the kernel is compute bound and should be rewritten with faster
> math. The kernel runs at 6% of the FP32 ceiling. Name the two distinct things
> that could be true, say which measurement distinguishes them, and explain why
> "compute bound" was never a safe conclusion from that arithmetic.

**The two possibilities.**

1. **The kernel is bound by a level the colleague did not draw.** Compulsory
   DRAM traffic is a property of the *problem*; the kernel's own traffic is a
   property of the *decomposition*. A kernel can have a compulsory intensity of
   200 and a request-level intensity of 0.25, and the second is what the
   hardware sees on every instruction. This is the naive GEMM exactly: 181
   FLOP/byte compulsory, 0.25 requested, 724x between the two columns, 6.5% of
   the ceiling. The binding ceiling is the on-chip operand-fetch one at
   ~5300 GB/s, and `0.25 x 5300 = 1325 GFLOP/s` is 6.8% of 19 500. The model
   predicts the measurement to within 1%; it was just the wrong level.
2. **Nothing is bound at all.** The kernel could be latency limited — too few
   warps, or a dependence chain, or a grid smaller than one wave — in which case
   no ceiling is reached and no roofline statement applies. Module 21's chase
   kernel is at 1.7% of the only ceiling it comes near.

**The measurement that distinguishes them.** Achieved bandwidth at each level,
compared to that level's ceiling. If the request-level traffic divided by the
elapsed time is near the on-chip ceiling, case 1. If *every* level's achieved
bandwidth is a small fraction of its ceiling and the FLOP rate is a small
fraction of the compute ceiling, case 2 — nothing is saturated, and the next
question is Little's Law, not arithmetic intensity.

If `ncu` were available: `dram__bytes.sum` and `l1tex__t_bytes.sum` divided by
`gpu__time_duration.sum`. Without it, as here, you count the bytes by hand from
the source and divide by a measured time, which is what Example 2 does.

**Why the conclusion was never safe.** "Compute bound" is a claim that the
arithmetic pipeline is saturated. An arithmetic intensity computed against a
level the kernel does not stress cannot support that claim — it supports only
"DRAM is not the limit". `AI(DRAM) > ridge(DRAM)` rules out one ceiling. It
says nothing about the other three. The valid inference from a high compulsory
intensity is *negative*: stop looking at DRAM.

---

## Q2

> Module 17 measured tiled GEMM at 9.4% of the FP32 ceiling. The shared
> roofline gives 7.0% for a scalar `LDS` ceiling and 13.0% for an `LDS.128`
> ceiling. Explain why the measurement lands between them, and what you would
> change to move it toward the upper figure. Then explain why that change is
> **not** what Module 18 actually did, and why M18's change was better.

**Why between.** The two ceilings bracket the kernel because its shared-memory
traffic is a *mixture* of the two instruction forms. M17 disassembled it: the
accumulation body of `gemmTiled<16,0>` is **4 `LDS.128` + 16 `LDS` + 16 `FFMA`**.
The `As[ty][k]` reads are contiguous in `k` and `ptxas` merges four of them into
one `LDS.128`; the `Bs[k][tx]` reads are strided by the tile pitch and stay
scalar. A kernel whose shared reads were all scalar would sit at the lower
bound; one whose reads were all vector at the upper. This one is a mix, so it
sits in between, and 9.4% is where the mix puts it.

(There is a second effect pushing the same way: `As[ty][k]` is warp-uniform in
`k`, so it is a **broadcast** — M7, degree 1 — and a broadcast does not consume
32 lanes of bank bandwidth. The byte-counting ledger charges it as if it did,
which is why Exercise 2 measures tiled GEMM 1.14–1.24x *above* its own
prediction.)

**How to move it up.** Make every shared read vectorised: stage the A tile
transposed so that the `TM` values a thread needs are adjacent, and read them as
`float4`. That is M18's `As[k][m]` layout, and the SASS consequence is 8 `LDS`
becoming 2 `LDS.128`. Ceiling-wise, the kernel moves from the 5.2 TB/s line to
the 9.7 TB/s line: a 1.9x improvement, taking it from ~9% to ~17% of peak.

**Why that is not what M18 did.** Changing the instruction form moves the kernel
to a **higher ceiling**; it does not move the kernel along the axis. Its
arithmetic intensity is still 0.25 FLOP/byte, and 1.9x is all it can ever buy.
M18 instead changed the *decomposition*: an `8x4` register tile reads `TM+TN =
12` floats per `TM·TN = 32` FMAs, which is `AI = 1.333 FLOP/byte`, **5.3x**
further right on the same axis. Measured 7399–8336 GFLOP/s, 41–46% of the
ceiling.

Vectorisation buys you a better ceiling once. Raising the arithmetic intensity
buys you a factor proportional to how far you raise it, and you can keep doing
it until something else binds — which for M18 was registers, at `TM·TN = 256`.
And of course M18 did both: the register-tiled kernel's shared reads are
`LDS.128` *and* its intensity is 1.333. The order matters, though. Vectorising
a 0.25 FLOP/byte kernel gets you to 17% of peak and stops; raising the
intensity first is what makes the vectorisation worth anything.

---

## Q3

> A kernel's roofline prediction is 400 GFLOP/s and it measures 520. Give three
> substantively different ways the ledger could be wrong, and for each, the one
> additional number you would compute to test it.

A measurement above its own roof is always a modelling error, never a hardware
surprise. Three distinct causes:

**(a) You counted traffic that does not cross the level you charged it to.**
The canonical case in this module: requested bytes charged against the L2
ceiling. The naive GEMM requests 12.971 GB; against 1305 GB/s that predicts 326
GFLOP/s and the kernel measures 1169 — 3.6x over. The requests mostly hit in L1,
which is five times faster. *Test:* compute achieved bandwidth at the level you
charged — `bytes_counted / elapsed` — and compare it to that level's ceiling. If
it exceeds the ceiling, the traffic is not going there.

**(b) You over-counted the bytes.** Broadcasts are the common case: 32 lanes
reading the same shared word is one bank access, not 32, but a naive
`bytes = lanes x 4` ledger charges 128. The same applies to a warp-uniform
global load (1 sector, M5) and to any value the compiler keeps in a register
across iterations (M18 measured 52 redundant source-level reads CSE'd away and
the "un-hoisted" loop running 1.05x *faster*). *Test:* count memory
**instructions** in the SASS loop body and multiply by the bytes each one
actually moves, instead of counting source-level array references.

**(c) You under-counted the FLOPs, or counted the wrong ones.** If the kernel
performs operations you did not put in `F` — or if you counted an FMA as 1 FLOP
instead of 2 — the numerator is wrong and the measured rate inflates. The
inverse error also exists and is more insidious: counting FLOPs the hardware
does not execute, e.g. taps the compiler eliminated because a guard made them
provably zero. *Test:* count `FFMA` / `FADD` / `FMUL` in the SASS, multiply by
32 lanes and by the launch's warp count, and compare to your `F`.

The general discipline: a measurement above the roof means **go back to the
ledger**, and the fastest way to settle it is almost always the disassembly,
because every column of the ledger is countable there.

---

## Q4

> A kernel's inner loop is one `LDS` followed by one `FFMA`, repeated, with no
> address arithmetic. Compute its plateau on (a) the issue axis and (b) the
> shared-memory axis. Which binds, and what does that tell you about M18's
> `BN/TN >= 16` rule?

Using this module's measured ceilings: issue 310 G warp-instructions/s, shared
scalar 5277 GB/s, FP32 19 482 GFLOP/s.

**(a) Issue axis.** The loop body is 2 instructions producing one FFMA's worth
of work = 64 FLOP per warp-pair-of-instructions, so 32 FLOP per instruction:

```
310 G instr/s x 32 FLOP/instr = 9920 GFLOP/s  =  51% of the compute ceiling
```

**(b) Shared axis.** One scalar `LDS` per FMA is 4 bytes per 2 FLOP, so
`AI = 0.5 FLOP/byte`:

```
0.5 FLOP/byte x 5277 GB/s = 2639 GFLOP/s  =  13.5% of the compute ceiling
```

**(b) binds, by 3.8x.** The shared pipe is the constraint, not the scheduler.
That is the quantitative content of M7's two-cycle floor: a conflict-free
32-lane 4-byte `LDS` occupies the shared pipeline for **two** cycles while the
scheduler could have issued two instructions in that time. The LSU is narrower
than the issue width, so a kernel at one shared read per FMA is wasting half its
issue slots waiting.

**What that has to do with `BN/TN >= 16`.** M18's rule says that once a thread
tile is read with `float4`, fewer than 16 threads along the N axis makes the
B-tile read a 2-way bank conflict in every phase. The reason it costs is exactly
the gap computed above. On a 4-byte access the two-cycle floor gives a 2-way
conflict for free (M7's `max(2,D)` law): the array is under-subscribed and has a
spare cycle. On an `LDS.128`, the access is phase-split into 4 phases of 8 lanes
and **one phase already asks for the full 128 B/cycle** — the array is exactly
subscribed and there is no spare cycle, so `cost ∝ D` with no floor. M18
isolated this with two configurations identical in registers, spills, shared
bytes, occupancy, both reuse ratios and the entire inner-loop SASS, measuring
**1.61x apart** against a predicted 1.67x.

In roofline terms: the `LDS.128` ceiling is twice the scalar one *only while the
accesses are conflict-free*, and a 2-way conflict on a 16-byte access moves you
straight back down to the scalar line. `BN/TN >= 16` is the condition for
staying on the upper ceiling, and no resource counter reports it — which is why
M18 had to find it by constructing a controlled pair and timing them.
