# Module 12 — Check Your Understanding, answers

---

## Q1 — why version 4 is worth 1.99× when versions 2 and 3 were worth 1.45× between them

**What version 3 was doing.** A v3 block holds 256 elements. It issues 256 global
loads — one per thread — and then spends eight barrier-separated tree rounds in
which it issues **no memory traffic at all**. Over the block's lifetime the
memory system sees one burst of requests followed by a long quiet period during
which the block is doing shared-memory arithmetic and convoying at barriers. With
262,144 such blocks and only 240 resident at a time, the machine is running 1092
waves of this pattern, and at any instant a large fraction of the resident blocks
are in their quiet phase.

Little's Law (Module 1) is the formal statement: sustained bandwidth = requests
in flight ÷ latency. DRAM latency here is ~575 cycles (Module 4). To saturate
375 GB/s you need thousands of outstanding sectors, and a thread with exactly one
load in its whole lifetime supplies one.

**What version 4 changes.** Each thread now issues **two independent loads**
before it does anything else. They are independent — neither address depends on
the other's result — so both are in flight simultaneously. Concurrency per thread
doubles, and the first tree level is absorbed into the load phase where it costs
nothing. The block's quiet phase is unchanged in length but now covers twice as
much data.

Measured: 38.1% → 75.7% of the streaming ceiling, a factor of 1.97. Almost
exactly the doubling of memory-level parallelism, which is the clue that this is
a concurrency effect rather than a work effect.

**Eight elements per thread instead of two.** Two bounds, pulling opposite ways.

*The upper bound is Little's Law running out.* Concurrency per thread rises to
eight, but the memory system saturates: at 76% with two loads, there is at most
1.32× of headroom left, so eight elements cannot possibly be worth 4×. In
practice you will measure something between 1.25× and 1.32× and then nothing —
version 5 gets to 99.5% with two elements per thread simply by removing the
barriers, so the headroom was never really about the load count.

*The lower bound is wave quantisation.* Eight elements per thread means
`n / (8 × 256)` = 32,768 blocks, which is 136 waves. That is still deep enough
for the tail effect to be negligible. Push it further — 512 elements per thread
gives 512 blocks, about two waves — and the last wave's imbalance starts to
matter (Module 1's tail effect), and beyond that you are into the v6 regime where
the grid is a property of the machine and the tail effect disappears entirely
because every block runs for the whole kernel.

So: Little's Law bounds the gain from above, wave/tail effects bound the useful
element count from above as well, and between them the answer is "a bit more, and
then nothing." The honest version of the answer is the measurement: v5, with two
elements per thread and three barriers, reaches 99.5%. There was never 4× on the
table.

---

## Q2 — the `volatile` tail that works

**The strongest technical version of the colleague's argument.**

`nvcc` does not emit arbitrary code. At every point where control flow that
diverged must rejoin, the compiler inserts a `BSSY` / `BSYNC` pair — a
convergence barrier — and the hardware honours it: lanes named by the barrier
wait there. The six lines of the `volatile` tail contain no control flow at all;
they are a straight-line sequence of `LDS`/`STS` pairs inside a single
`if (tid < 32)` region that the whole warp enters together. Within a converged
region, Ada's load/store unit issues one warp's shared-memory operations in
program order, and since all 32 lanes issue instruction *k* before any of them
issues instruction *k+1* — because there is only one instruction stream being
issued to the warp and nothing has caused the lanes to separate — lane 4's load
of `v[5]` genuinely does happen after lane 5's store to `v[5]`. The `volatile`
stops the compiler from keeping `v[tid]` in a register across the lines, so the
stores and loads are really there. Zero errors in every run is not luck.

Module 8 measured exactly this and reported it: a converged-region
warp-synchronous exchange produced 0 wrong results out of 524,288.

**Why it is not a defence.**

The argument depends on a property of the *generated code*, not of the *source
program*. `BSSY`/`BSYNC` placement is a compiler decision, and the CUDA
programming model on sm_70+ explicitly does not guarantee that lanes of a warp
are at the same instruction outside a `_sync` primitive. The moment the
compiler's inlining, scheduling or register-allocation decisions change — a new
toolkit, a different optimization level, a caller that inlines something else
around it — the argument evaporates with no source change and no warning. And
the failure mode is not a crash: it is a slightly wrong sum.

The general rule, from Module 9: **"it passed" and "it is correct" are different
claims, and only one of them is about your program.**

**A change to the surrounding code that could break it, with no change to the
tail.** Several work; the cleanest is to put the tail inside a loop whose trip
count varies across the *block* — say, reducing several tiles per block with
`for (t = 0; t < tilesFor(blockIdx.x); ++t)`. That is still block-uniform, so it
is legal, but it changes the code the compiler generates around the tail: with a
loop body large enough, the compiler may sink the `if (tid < 32)` test, re-order
the region relative to the loop's own convergence barriers, or split the
straight-line sequence. Another: add a `__ldg` or a global load *inside* the
`if (tid < 32)` region. Module 8 measured that a guarded body containing a global
load flips the compiler from predication to a real branch with `BSSY`/`BRA`/
`BSYNC` — and once there is a real branch inside the region, the lanes can
separate and the ordering assumption is gone. The tail's six lines are unchanged
in both cases.

---

## Q3 — bit-identical across two GPUs with different SM counts

**Why the results differ.** The grid is
`cudaOccupancyMaxActiveBlocksPerMultiprocessor × multiProcessorCount`. On the
40-SM part that is 6 × 40 = 240; on the 68-SM part it is 6 × 68 = 408. The kernel
therefore produces 240 partial sums on one machine and 408 on the other, and the
array is cut into a different number of pieces at different boundaries. Float
addition is not associative, so a different bracketing of the same 2^26 addends
gives a different result — typically by a few ULP, occasionally by more.

There is no atomic anywhere and no scheduling non-determinism: each machine is
perfectly reproducible with itself. **Determinism is a property of a kernel plus
a launch configuration**, and the launch configuration here is a property of the
hardware. `example02.cu` measures exactly this at 120 / 240 / 480 blocks and gets
three different answers.

**Fix 1 — fix the decomposition, let the grid vary.** Cut the array into a
constant number of chunks `C` whose boundaries depend only on `n`. Chunk `c` is
always summed in the same order and always lands in `partial[c]`; the final pass
walks `partial[0..C)` in index order. The grid only decides which block visits
which chunk, and that never enters the arithmetic. This is Exercise 3's TODO 2.

*Cost:* measured at **1.00–1.03×**, i.e. nothing. You give up the ability to size
the work per block to the machine — if `C` is too small for a very large GPU,
some SMs idle — so `C` must be chosen large enough for any plausible device,
which lengthens the final pass slightly.

**Fix 2 — stop using float.** Accumulate into a 64-bit integer with a fixed-point
scale. Integer addition is associative and commutative and does not round, so
every order gives the same bits, on every machine, with a plain atomic and no
tree discipline at all.

*Cost:* dynamic range. You must know the magnitude of your data before you run,
negative values need a bias, and data spanning more than about 40 binary orders
of magnitude cannot be represented. (A third option, exact float summation via
Kahan or via a fixed-point superaccumulator, buys accuracy as well as
determinism at a real arithmetic cost; on a bandwidth-bound reduction that cost
is also nearly free, which is worth knowing.)

**Why fix 1 being free is not obvious in advance.** The constraint reads like a
scheduling constraint — "you may not adapt the work decomposition to the
machine" — and adapting work to the machine is normally exactly how you get
performance. The reason it costs nothing here is that the kernel is memory-bound:
as long as there are enough independent chunks to keep every SM supplied with
outstanding loads (2048 chunks over 240 blocks is ample), the exact partitioning
is irrelevant, because the bottleneck is the DRAM bus and not the schedule. In a
compute-bound kernel with a long critical path per block, the same constraint
would cost real time. The general form of the answer: **reproducibility is
expensive when the schedule matters and nearly free when the bus is the
bottleneck.**

---

## Q4 — max versus sum, associativity versus accuracy

**`fmaxf`: the second one is never more accurate, and the question is
meaningless.**

`fmaxf` is associative, commutative *and* **exact**: its result is always one of
its two inputs, bit for bit. No rounding ever occurs, so both implementations
return exactly the same bit pattern — the largest element of the array — for
every input and every decomposition. Accuracy is not a dimension along which they
can differ.

(Two footnotes that are worth having. `fmaxf` propagates the non-NaN operand when
one input is NaN, which is what makes it associative in the presence of NaN
where the naive `a > b ? a : b` is not — `a > b` is false when either is NaN, so
the ternary version's answer depends on the order. And the identity is
`-INFINITY`, not `0.0f`, which matters the moment the array can be all-negative:
Module 10's Exercise 3 was built around exactly that sign trap.)

So a max reduction is the clean case: the parallel form is legal because the
operator is associative, and it is bit-identical to the sequential form because
the operator is exact. **Determinism here is free and complete** — the same max
kernel gives the same bits on every grid, every machine, and even with an
atomic max built on Module 10's order-preserving float-to-integer encoding,
because max does not care about order at all.

**`+`: the second one is almost always more accurate, sometimes dramatically so,
and the question is meaningful.**

Float addition is *not* exact — every operation rounds — and it is therefore not
associative. Two consequences, which the question is asking you to keep apart:

- **The algebraic property that makes the parallel form legal** is associativity,
  and float `+` does not have it. The tree is not computing the same expression
  as the loop; it is computing a different bracketing of the same addends. We do
  it anyway, by convention, and that convention is exactly what makes a float
  reduction non-deterministic across decompositions.
- **The numerical property that makes it better** is that the error bound scales
  with the *depth* of the summation tree, not its width. Sequential accumulation
  has depth `n-1` and a worst-case relative error of `O(n)·eps`; the tree has
  depth `log2 n` and `O(log n)·eps`. For `n = 2^26` that is a bound 2.6 million
  times tighter. And the tree avoids the catastrophic regime entirely: once a
  sequential accumulator exceeds `2^24` times a typical addend, each new addend
  rounds away and the sum stops growing. `example02.cu` measures a sequential
  float loop over 2^26 copies of `1.0f` returning `16777216` — **75% low** —
  while the tree returns the exact answer.

The pair is worth remembering as a matched set: for `max`, associativity holds
exactly and the accuracy question is empty; for `+`, associativity holds only
approximately and that approximation is simultaneously the source of the
non-determinism *and* the reason the parallel version is more accurate. You do
not get to have one without the other.
