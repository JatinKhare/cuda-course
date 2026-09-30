# Module 14 — Check Your Understanding (answers)

---

## Q1

> Kernel A: `atomicAdd(&hist[in[i]], 1u)` on global memory. Kernel B: the same
> on a 256-bin `__shared__` array with no zeroing, no barriers and no flush
> (deliberately incorrect; it exists only to be timed). On the `clustered` input
> A is 61.0 ms and B is under 1.2 ms. Explain the ratio from the *instruction*
> each compiles to, then say what happens to each if a warp's 32 lanes hit 32
> different bins instead.

**The ratio.** The two kernels compile to two different instructions with two
different contention stories.

Kernel A emits `RED.E.ADD.STRONG.GPU`. The request leaves the SM and is
performed by the ALU of the L2 slice that owns the address. On `clustered`, all
32 lanes of a warp present the *same* address, so the warp produces 32 requests
to one slice, and the slice serializes them: 32 operations where a spread
pattern would have produced parallelism. Module 10 measured this warp-uniform
case at ~6× the lane-varying case, and it is why `clustered` (61.0 ms) is slower
than `uniform` (50.4 ms) for identical work. The compiler cannot rescue it with
its automatic aggregation (`VOTEU.ANY`/`POPC`/`RED`) because the address comes
from a memory load and cannot be proved warp-uniform.

Kernel B emits `ATOMS.POPC.INC.32`. The shared-memory unit takes the set of
lanes targeting one address, counts them with a population count, and applies
the increment **once**. Thirty-two same-address lanes cost one operation, not
32. The expensive case for A is the *free* case for B.

So the ratio is not "shared memory is faster than global memory" — it is
"same-address concurrency within a warp costs 32 on one instruction and 1 on the
other", multiplied by the L2-versus-SM-local latency difference (Module 4: 241
vs ~40 cycles) that B also avoids.

**Change the input so a warp hits 32 different bins.**

- **Kernel A gets faster.** Thirty-two distinct addresses can be serviced by up
  to 32 slices in parallel instead of one. Measured: 61.0 → 50.4 ms. The win is
  not the 32× the address count suggests, because (i) the slices are chosen by a
  hash and do not distribute perfectly, and (ii) this module's Example 1 Part D
  shows the real unit is the 32 B **sector**: 32 bins scattered over 256 words
  occupy up to 32 sectors and cost 7.99× what the same 32 bins packed into 4
  sectors cost.
- **Kernel B does not change.** The `ATOMS.POPC.INC.32` now merges nothing, so
  it costs 32 lane-operations instead of 1 — but the kernel was never limited by
  that. Example 1 measures the byte-load stream at 152 GB/s and the same stream
  with a shared atomic per element at 149.4 GB/s: **the atomics cost 1.7%.**
  Kernel B is limited by the width of its loads, and the load width did not
  change. Measured: 1.10 ms on `clustered`, 1.10 ms on `uniform`, flat.

The general statement: **contention is a property of the access pattern, and
whether contention costs anything is a property of what else the kernel is
waiting for.**

---

## Q2

> 4,096 bins, 32-bit keys, and an input where fewer than 20 bins are ever
> occupied. Your colleague proposes `R = 8` replication because "the
> distribution is extremely skewed, so contention will be terrible." Argue both
> sides and name the one measurement that decides it.

**The case for.** Twenty occupied bins out of 4,096 is effectively a 20-bin
histogram. Module 10's table puts `K = 16` at 10.1 Gatomic/s against
132 Gatomic/s at `K = 65536` — a 13× contention penalty — and the whole grid is
hammering twenty addresses. If any configuration in this module ought to be
contention-bound, it is this one. Replication divides the *inter-warp*
contention within a block by up to 8.

**The case against, in three parts.**

1. **The contention that replication removes is the cheap kind.** The expensive
   collisions are the 32 lanes of one warp landing on one bin, and Ada's
   `ATOMS.POPC.INC.32` already merges those into a single operation. What is
   left is 8 warps of a block colliding, which costs at most 8 shared-unit
   operations per round while the block consumes hundreds of input bytes. At the
   streaming ceiling the memory system delivers ~5 bytes per SM per cycle. The
   contention is roughly 50× cheaper than the data feeding it. Example 1 Part B
   measures the whole `R` sweep at **1.00×** on four distributions including
   `same-bin`, which is more skewed than this workload.
2. **The footprint is fatal.** `4096 × 4 B = 16 KB` already costs a resident
   block (6 → 5 per SM). `R = 8` is **128 KB**, which exceeds the 101,376 B
   opt-in maximum: `cudaFuncSetAttribute` will fail or the launch will return
   `cudaErrorInvalidValue`. The proposal does not merely underperform; it does
   not run. The largest legal `R` is 4 (64 KB, 1 block/SM), and Example 1
   measured `R = 16` at 16 KB costing 17% purely from the occupancy step.
3. **The flush gets 8× more expensive in shared reads** — `grid × 4096 × 8`
   shared loads to fold — and with only 20 occupied bins the `if (sum)` guard
   was already removing 99.5% of the global flush atomics, so there was nothing
   left to save there either.

**The deciding measurement:** run the *unreplicated* privatized kernel and
report its time as a **percentage of a streaming ceiling measured in the same
sweep**. If it is at 95–100%, the kernel is bandwidth-bound, no contention fix
of any kind can help, and the discussion is over. If it is at 40%, find out what
the missing 60% is before assuming it is contention — in this module it was the
load width every time, and one A/B (scalar load versus vectorized load, atomics
held constant) settles it.

---

## Q3

> Correct at 1,024 bins, counts too low at 64 bins, reproducibly, same input,
> 240 blocks, 2^26 elements. No sanitizer reports anything. Mechanism? And what
> happens at 64 bins with 4,096 blocks?

**Mechanism: private-bin counter overflow.** A "too low, and reproducible"
histogram is not a race — races produce *varying* answers (Module 10: the
diagnostic rule) — and it is not an uninitialized-memory bug, which makes counts
too *high*. Reproducible loss of counts means the accumulator is saturating or
wrapping.

With 240 blocks and 2^26 elements, each block processes 2^26/240 ≈ 279,620
elements. At 1,024 bins the average per-bin per-block count is ~273 and even a
50× skew stays small. At 64 bins the average is ~4,369, and any distribution
that concentrates a quarter of the mass into one bin puts ~70,000 into a single
private counter. If those counters are 16-bit — the packed representation of
Exercise 3, or an `unsigned short` array chosen to halve the shared footprint —
65,535 is the ceiling and everything above it wraps.

The reason it is a function of the *bin count* rather than of `n` is that the
expected occupancy of a bin is `elements_per_block / nBins`: fewer bins means a
higher count per bin, with everything else unchanged.

No sanitizer reports it because there is nothing to report. The shared access is
in bounds, initialized, and synchronized. The wrap is arithmetic, and the
intended meaning of the bits is not expressible to a memory-safety tool. Module
12's Exercise 3 established the general form: **the tools check what you did,
never what you meant.**

**Raising the grid to 4,096 blocks.** Elements per block falls to 2^26/4096 =
16,384, which is below 65,535 *whatever the distribution does* — even if all
16,384 elements land in one bin. The symptom disappears.

**Why it is a fix and not a coincidence.** The bound is derived from `n` and the
grid alone and does not mention the data:

```
max per-bin count in a block <= elements per block <= n/grid + blockDim
require n/grid + blockDim <= 65535
```

A fix that depends on the observed distribution is a coincidence; this one holds
against an adversary. The cost is explicit and should be stated alongside it:
17× more blocks means 17× more flush atomics (`grid × nBins`), which is the
coarsening ceiling the 2× memory saving is paid for with. If `grid × nBins`
approaches `n`, the packed representation has stopped being worth it and you
should widen the counters instead.

---

## Q4

> Module 10: 23.35 Gatomic/s at `K = 256`. Module 14: 4.0 Gatomic/s for a
> 256-bin histogram of random bytes, and 25.0 Gatomic/s for a synthetic kernel
> that also touches 32 distinct bins per warp. All three are correct. Reconcile,
> then give the rule.

**Reconciliation.** All three kernels issue the same number of atomics into the
same number of distinct bins. They differ in **how many distinct 32 B sectors a
single warp's 32 atomics occupy.**

Module 10's benchmark and the 25.0 Gatomic/s synthetic both use an address of
the form `base + lane`: lane `l` touches word `l` of an aligned run, so the
warp's 32 words are 128 contiguous bytes = **4 sectors**. The requests coalesce
on the way to the L2 into four transactions, exactly as 32 coalesced loads do
(Module 5).

The real histogram's bin index comes from the data. A warp's 32 lanes land on 32
addresses scattered across the whole 1 KB bin array, occupying up to **32
sectors**, which is 32 transactions instead of 4.

Example 1 Part D isolates this with everything else held constant — same atomic
count, same 32 distinct addresses per warp, only the sector footprint changed:

```
  32 words in  4 sectors      8.05 ms   25.0 Gatomic/s
  32 words in 32 sectors     64.29 ms    3.13 Gatomic/s
  sector ratio: 7.99x
  scramble ratio (lane order permuted, same sectors): 1.00x
```

**7.99× from the sector count; 1.00× from which lane gets which address.** So
Module 10's table is correct *for its access pattern* and is an optimistic bound
for any `K` above 8, because `i & mask` produces the densest possible packing.
The real histogram's 4.0 Gatomic/s is the same physics at the other end of the
same axis.

**The rule for a colleague.**

> The cost of a warp's global atomics is set by the number of distinct 32 B
> sectors they touch, not by the number of atomics and not by the number of
> distinct bins. Count sectors per warp the way Module 5 taught you to count
> them for loads.

And the second half, which matters more:

> **The source code does not contain the information.** `atomicAdd(&hist[b], 1)`
> is a complete description of the operation and tells you nothing about `b`'s
> distribution across the lanes of a warp — which is the only thing that
> determines the cost. Two identical source lines differ by 8× on this hardware
> depending on the data, and by another 6× depending on whether the address is
> warp-uniform. You cannot predict atomic cost from source; you can only predict
> it from source **plus** a statement about the input distribution. That is why
> every measurement in this module names its distribution, and why a histogram
> benchmark that does not is not a measurement.

A corollary that catches people: Module 10's fourth contention-reduction move —
pad the hot bins apart so they hash to different L2 slices — makes the sector
count *worse* for a histogram, because a warp's bins are already distinct and
spreading them guarantees one sector each. It is the right move for a handful of
hot counters and the wrong move for a histogram, and the difference is exactly
"how many bins does one warp touch".
