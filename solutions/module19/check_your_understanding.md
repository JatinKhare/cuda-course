# Module 19 — Check Your Understanding, answers

---

## Q1 — 42 registers, 192-thread blocks, and a null result

**The arithmetic.** 192 threads is `ceil(192/32) = 6` warps per block.

At **42 registers per thread**:

```
regsPerWarp   = roundUp(42, 8) * 32 = 48 * 32 = 1536
warpsPerSlice = 16384 / 1536        = 10
blocksByRegs  = 4 * 10 / 6          = 6
blocksByWarps = 48 / 6              = 8
blocksBySmem  = 102400 / 1024       = 100
blocksByBlock = 24
blocks/SM     = 6   (registers bind)
occupancy     = 6 * 6 / 48 = 36/48  = 75 %
```

At **40 registers per thread**:

```
regsPerWarp   = 40 * 32 = 1280       (40 is already a multiple of 8)
warpsPerSlice = 16384 / 1280 = 12
blocksByRegs  = 4 * 12 / 6   = 8
blocks/SM     = min(8, 8, 100, 24) = 8   (registers and warp slots tie)
occupancy     = 8 * 6 / 48 = 48/48 = 100 %
```

Two registers took the kernel from 75 % to 100 % occupancy, because 42 rounds up
to 48 and 40 does not. This is the granule crossing from lesson §4, and it is
the one place in the low 40s where surrendering registers is worth anything at
all: at 48 registers a granule is worth 8 warps, at 40 it is worth zero because
the warp-slot ceiling has already been hit.

**Why the experiment cannot support the conclusion — reason 1: the measurement
is on the flat part of the curve, by construction.** Occupancy buys latency
tolerance and nothing else. Module 1 measured the curve: a dependent pointer
chase is linear in throughput to 3 warps, has a knee at 6, and is flat at 4.87×
by 24 warps. Their change moved the kernel from **36** resident warps to **48**
— both far past the knee of any latency curve this hardware has. A null result
there is the *expected* result and says nothing about whether occupancy matters;
it says this kernel was already latency-tolerant at 36 warps. To make occupancy
matter you have to vary it across a range that spans the knee, which on this GPU
means going down to 8–16 warps, and `__launch_bounds__` is how you do that
(Exercise 2).

**Reason 2: they measured theoretical occupancy and reported a runtime, with
nothing connecting the two.** Three things could independently have made the
change invisible:

- *Achieved occupancy may not have moved at all.* If the grid is under one wave
  — or has a long tail, or ragged block durations — the kernel was never at 75 %
  to begin with, and raising the theoretical ceiling to 100 % changes nothing
  that is reachable. Exercise 3's quarter-wave launch sits at 24 % achieved with
  a theoretical 100 %.
- *The 40-register build may have spilled.* `ptxas` was told nothing; if the
  kernel wanted 42 and now reports `localSizeBytes > 0`, two effects of opposite
  sign are cancelling and the net zero is a coincidence.
- *The kernel may not be latency-bound at all.* A bandwidth-bound kernel at the
  DRAM roof, or an issue-bound kernel with high ILP, gains nothing from warps.

**What to measure instead.** Sweep the occupancy of the *same source* with
`__launch_bounds__(192, n)` for `n = 1 …`, check `localSizeBytes == 0` at every
point so the sweep is about warps and not about spills, print achieved occupancy
alongside theoretical so you know the warps you paid for are actually resident,
and report throughput against the full sweep rather than against one adjacent
pair. If the curve is flat from 16 warps upwards, the correct conclusion is
"this kernel is latency-tolerant above 16 warps", which is a useful fact about
the kernel — not a fact about occupancy.

---

## Q2 — 49 % versus 12 % achieved, everything else equal

Both kernels have the same resource footprint, the same theoretical occupancy
(50 %), the same grid and the same data, and they differ 4× in achieved
occupancy measured with the **elapsed** denominator. The numerator of that
fraction is warp-cycles of residency; the denominator is fixed by the wall
clock. So the question is: where did kernel B's warp-cycles go, or where did its
wall clock go.

**Mechanisms, each with the measurement that settles it.**

| mechanism | what it would look like | measurement |
|---|---|---|
| **Grid too small to fill the machine** | both would be low, not one | compute `grid / (blocksPerSM × nSM)`. Identical for both by hypothesis, so this is eliminated by arithmetic alone, with no run |
| **Tail / wave quantisation** | B's grid lands just past a wave boundary | `waves = grid / (blocksPerSM × nSM)`; the tail costs about `1/waves` (Module 1). With 4096 blocks and a 240-block wave that is 17 waves and ≈6 % — far too small to explain 4×, but measure it by re-running at a grid rounded down to an exact multiple of the wave |
| **Load imbalance across blocks** | SMs idle while one finishes | instrument per-block duration with `clock64()` and report `max/mean`; or compute **both** occupancy variants — if B's `occ_active` is near 49 % while its `occ_elapsed` is 12 %, the missing time is SM idle time and this is the cause |
| **Most of B's blocks exit immediately** | the useful work is concentrated in a few blocks | count, device-side, how many blocks executed more than a trivial number of iterations; or look at the sum of per-block durations against the kernel time |
| **Warp-completion skew inside a block** | warps of one block finish 3× apart (lesson §6) | record per-warp lifetimes and compare the spread; this one is *inside* the SM's busy span |
| **A spin-wait or an atomic convoy** | — | eliminated without measuring: a warp waiting on a flag or a barrier is still **resident** and still counted, so serialisation *raises* achieved occupancy. If B were serialising on a lock its achieved occupancy would be higher than A's, not lower |

**Which would still be visible with the active denominator.** This is the part
of the question that matters.

- **Tail and cross-block load imbalance would vanish.** `occ_active` normalises
  each SM by the time *that SM* was busy, so an idle SM contributes nothing to
  either half and drops out of the average. Exercise 3 measured both: at a fixed
  grid, going from uniform to 1..8× costs moved `occ_active` by **+3.3** points
  and `occ_elapsed` by **−14.3**.
- **Warp-completion skew and early-exiting blocks would still be visible**,
  because both happen while the SM is busy and both leave warp slots empty
  inside the measured span.

So the single most informative experiment is to compute **both** denominators
for both kernels. If `occ_active` is similar and `occ_elapsed` differs, the
problem is in the launch — tail or imbalance — and the fix is the grid. If
`occ_active` differs too, the problem is inside the blocks.

---

## Q3 — 100 % achieved occupancy at 6 % of the FP32 ceiling

**Why "reduce registers so more warps fit" is guaranteed to be a no-op or a
pessimisation.**

The first half is arithmetic. Achieved occupancy of 100 % means all 48 warp
slots were occupied essentially all of the time. 48 is a hard architectural
limit; there is no register count at which a 49th warp becomes resident.
Reducing registers cannot add warps, because the warp-slot limiter is already
binding and the register limiter is, by definition, not. Whatever is costing you
94 % of the machine, it is not a shortage of resident warps.

The second half is mechanism. Reducing the register budget has exactly two
possible outcomes. Either `ptxas` reschedules and nothing observable changes
(the no-op), or it runs out of rescheduling and spills to local memory, which
Module 4 established is DRAM with an L1 line in front of it. A spill that
reaches the values the inner loop touches every iteration puts a DRAM round trip
in the innermost loop; Module 18 measured that at **18.7×** and Exercise 2 at
**7–8×** on a different kernel. So the proposal is a coin flip between nothing
and a catastrophe, with no upside available.

**What is actually happening.** Occupancy counts warps that **exist**. The
quantity that produces instructions is warps that are **eligible to issue**. A
warp waiting at `__syncthreads()`, blocked on a long-latency dependency, waiting
on a memory return or losing an issue-slot contest is resident, counted, and
doing nothing. 100 % occupancy and 6 % of peak is the signature of a kernel
whose warps are all present and all stalled.

**The measurement to make instead.** First, the free one: check
`cudaFuncGetAttributes().localSizeBytes`. If it is non-zero, the kernel is
already spilling, the 6 % is probably explained, and the correct move is **more**
registers per thread — the exact opposite of the proposal — bought with a lower
`minBlocksPerMultiprocessor` or by not setting one at all.

If there is no spill, the question becomes "what are the warps stalled on", and
that is a stall-reason question: `smsp__warps_eligible.avg.per_cycle_active`,
`smsp__issue_active.avg.per_cycle_active`, and the
`smsp__pcsamp_warps_issue_stalled_*` family. **Module 20** owns the eligible-vs-
resident distinction and the stall taxonomy; **Module 23** owns the profiler.
`ncu` is unavailable on this machine (`ERR_NVGPUCTRPERM`), so the local
substitute is a controlled A/B: hold occupancy fixed and change one candidate
cause at a time — shorten the dependence chain, add independent accumulators per
thread (Module 11's MLP experiment), remove a barrier, widen the loads — and see
which one moves the throughput. Whichever does is your stall reason.

---

## Q4 — `__launch_bounds__(256, 3)` versus `(128, 6)`

**They impose the same cap.** `roundDown(65536 / (256·3), 8) = roundDown(85, 8)
= 80`, and `roundDown(65536 / (128·6), 8) = 80`. Both allow 80 registers per
thread.

**They produce the same theoretical occupancy.** At 80 registers:

```
regsPerWarp   = 80 * 32 = 2560
warpsPerSlice = 16384 / 2560 = 6
warps by registers = 4 * 6 = 24
```

- 256 threads = 8 warps per block → `24/8 = 3` blocks → 24 warps → **50 %**
- 128 threads = 4 warps per block → `24/4 = 6` blocks → 24 warps → **50 %**

Both requests are satisfied exactly, which is not a coincidence: that is what
the cap formula was constructed to do.

**They do not produce the same achieved occupancy on a kernel with widely
varying block durations.** The block is the unit of placement and of
retirement. With 128-thread blocks the same total work is cut into twice as many
blocks, so:

- the work distributor has twice as many opportunities to refill an SM whose
  block retired early — which Exercise 3 measured as the only free load
  balancing on this hardware;
- the quantum of imbalance is halved: the longest block is a smaller fraction of
  the kernel;
- the tail is half as expensive, because the grid is twice as many waves and
  Module 1's tail cost goes as `1/waves`.

Exercise 3 measured this effect at the grid level — one wave of imbalanced work
reached 47.7 % elapsed occupancy and two waves of the same work reached 65.5 %
— and the block-size version of the argument is the same one.

**Which I would prefer, and why — not "it is a multiple of 32".**

**128 threads**, for three reasons that are all about quantisation:

1. **Placement granularity**, as above: finer blocks give the distributor more
   to work with and shrink both the tail and the imbalance quantum.
2. **Register quantisation rounds down less harshly.** Lesson §4(b): at 96
   registers per thread a 128-thread block gets 20 resident warps and a
   256-thread block gets 16, from the same kernel, because a block must place
   all of its warps at once. The effect is zero at 80 registers and real a
   granule either side of it, and you do not always control which side you land
   on.
3. **It leaves room under the 24-block cap.** Six blocks of 128 is well clear;
   if the kernel later gets cheaper and the register limiter stops binding, the
   128-thread version can grow to 12 blocks (48 warps, 100 %) while the
   256-thread version tops out at 6.

The counterweights, which decide it the other way for some kernels: each block
pays the **1024 B shared-memory driver reserve** and any fixed per-block cost —
a cooperative load prologue, a barrier, a reduction epilogue — so halving the
block size doubles those; and a kernel whose blocks cooperate through shared
memory gets less reuse per block. For a GEMM tile, 256 is usually right. For the
recursive-cascade kernel of Exercise 2, which has no shared memory and no
barriers, 128 is right.
