# Module 23 — Check Your Understanding, answers

---

## Q1

> `ncu` reports `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum =
> 3,145,728` against `smsp__sass_inst_executed_op_shared_ld.sum = 3,145,728`.
> A colleague says "a million conflicts, pad the array." Compute the conflict
> degree, then say when the colleague is right, when padding will do nothing,
> and when padding will make the kernel *slower*. Name the measurement from an
> earlier module that supports each case.

### The degree

The conflicts counter counts the **extra** wavefronts, not the total:

```
wavefronts = requests + conflicts = 3,145,728 + 3,145,728 = 6,291,456
D = wavefronts / requests = 6,291,456 / 3,145,728 = 2
```

**It is a 2-way conflict**, not "a million" of anything. The absolute magnitude
of the conflicts counter is proportional to how many shared loads the kernel
executes, which is a property of the problem size. Always divide.

### When the colleague is right

When `D >= 4` **and** the kernel is not already saturating some other ceiling.
At D = 2 on a 4-byte access the colleague is wrong before anything else is
considered. The cases where padding genuinely wins:

- **M17**, transposed A-tile store, degree 8 at T=16 and 32 at T=32: padding
  measured **1.053–1.062×** and **1.174–1.178×**.
- **M18**, register-tiled GEMM A tile: padding measured **1.23×** over XOR
  swizzle at BK=8, and the pad must be **4 floats, not 1** — a pad of 1 leaves
  degree 4, *a partial fix that measures like a fix*.
- **M7**, isolated degree-32 column access: **10.5–15.4×**, up to 19× cold.

### When padding will do nothing

Two distinct reasons, and they are worth separating.

1. **The degree is below the cost floor.** On Ada a 4-byte full-warp shared
   access costs `max(2, D)` cycles, not `D`. M7 measured `s[2*tid]` — which
   reports exactly this counter value — at **1.00×** the cost of `s[tid]`.
   D = 2 is free. **This is that case.**
2. **The conflict is not on the critical path.** M15 measured the identical
   32-way conflict M7 clocked at 14.79× as worth **0.96–1.03×** inside a
   DRAM-bound 8192² transpose, because the kernel was at ~90% of the pin rate
   and the replays hid behind DRAM latency the SM was waiting for anyway. The
   same conflict in an L2-resident 2048² transpose was worth **2.67–3.25×**.
   M12 saw the same shape: removing divergence *and* conflicts from a kernel
   wasting 75% of its memory system was worth only **1.45× combined**.

### When padding will make it slower

**When the pad breaks the 16 B alignment that `ptxas` needs to contract four
contiguous shared reads into one `LDS.128`.** M17 measured a row-major tiled
GEMM going from 20 shared instructions to **32** when the tile pitch went from
`T` to `T+1`, costing **0.674–0.706× at T=16** — a 40% loss, on a tile whose
conflict degree was 1 everywhere to begin with. Padding also costs shared
capacity, which costs blocks per SM: M7 measured `[192][32]` fitting 4 blocks/SM
and `[192][33]` only 3.

### The rule

**Pad only after measuring a conflict of degree ≥ 4, after checking that the
kernel is not already at some other ceiling, and after checking the SASS for a
vector merge you are about to destroy. Then pad by 4, not by 1.**

---

## Q2

> Kernel A: Achieved Occupancy 94%, `smsp__warps_eligible.avg.per_cycle_active =
> 0.11`. Kernel B: Achieved Occupancy 31%, eligible 2.8. Same cycles. Which has
> more headroom, what would you try on each, and which section next?

### The chain

`resident ≥ eligible ≥ issued`. Achieved occupancy measures **resident**.
Eligible measures how many of those residents could issue on an average cycle.
The ceiling for resident is 12 warps per scheduler on this GPU; for issued it is
1.0 per cycle per scheduler.

### Kernel A — 94% resident, 0.11 eligible

Roughly 11.3 warps resident per scheduler and 0.11 of them eligible: **99% of
the resident warps are parked.** Occupancy is maxed out and buying nothing. This
is Module 19's measured paradox stated in counters — *a warp stalled on a
barrier or a DRAM load is still resident, which is how 100% occupancy and 6% of
peak coexist.*

- **Headroom: a great deal, and none of it on the occupancy axis.** There is no
  occupancy left to add.
- **What to try:** more work in flight *per warp* — memory-level parallelism
  (several independent loads issued before any is consumed) or instruction-level
  parallelism (independent accumulator chains). Module 20 measured the exchange
  rate: one unit of ILP buys exactly one resident warp per scheduler until
  `latency × throughput` is met, and the smallest warps/scheduler reaching 90%
  of peak is 4, 2, 1, 1, 1 for ILP 1, 2, 4, 8, 16 — the product is constant
  at 4.
- **Next section: `WarpStateStats`.** Which stall dominates decides which axis.
  `long_scoreboard` → MLP. `wait` → ILP. `barrier` → rebalance, and occupancy
  will not help. `short_scoreboard` → shared-memory pressure.

### Kernel B — 31% resident, 2.8 eligible

Roughly 3.7 warps resident per scheduler, 2.8 of them eligible, so **76% of what
is resident is ready to go.** The warps it has are being used well; it simply
does not have many.

- **Headroom: less than A's, and it is on the occupancy axis.** With 2.8
  eligible the scheduler is already close to issuing every cycle it can.
- **What to try:** find the limiter, which is a static question — check
  `launch__occupancy_limit_registers`, `_shared_mem`, `_warps`, `_blocks`
  against M19's four-limiter slice model,
  `4*floor(16384/(roundUp(R,8)*32))/warpsPerBlock` etc. If registers bind, a
  `__launch_bounds__` is one line.
- **Next section: `Occupancy`**, and then `LaunchStats` for
  `launch__waves_per_multiprocessor` — if that is below ~1 the grid is simply
  too small and no resource change will help.

### The caution that applies to B

**More occupancy is not automatically better and is not even monotone.** M19
measured a sweep at 9.22 / 9.19 / 11.44 / 12.62 / 8.83 / 3.64 / 2.18 / 1.47 /
1.55 Gelem/s for 1/2/3/4/5/6/8/10/12 blocks per SM — 100% occupancy was
**7.15–8.15× slower** than the 33.3% winner, and 83.3% was reproducibly slower
than 100%. If raising B's occupancy raises its register pressure into a spill of
the *accumulators*, it will lose. Measure, do not assume.

---

## Q3

> Achieved Occupancy went 61.9% → 62.3% and the kernel got 7.5% slower. How?
> Which metric would have shown the regression, and why is it not in the
> Occupancy section?

### How both are true

`ncu`'s "Achieved Occupancy" is
`sm__warps_active.avg.pct_of_peak_sustained_active`. The `active` denominator is
**each SM's own busy cycles**. An SM that has finished its work and gone idle
stops incrementing its numerator *and* its denominator, so its idleness is
arithmetically invisible.

The change described is a **tail**: M19 measured exactly it, going from 240
blocks (one full wave on 40 SMs at 6 blocks each) to 241. The extra block forms
a second wave in which **one SM works and 39 are idle**. The one working SM is
as full as it was before — in fact marginally fuller, because the tail block
runs with no neighbours competing for the memory system — so `_active` ticks
*up* 0.4 points. Meanwhile the kernel cannot finish until that block does, so
wall time rises 7.5%.

### The metric that would have shown it

```
sm__warps_active.avg.pct_of_peak_sustained_elapsed
```

Same numerator, different denominator: the **whole kernel's** elapsed cycles,
for every SM including the idle ones. M19 measured it falling 61.9% → 57.9% on
the same change — a 4.0-point drop against the 0.4-point *rise* in the metric
everybody quotes.

The same pair separates a load imbalance even more starkly. M19's 1..8× cost
imbalance within one wave moved `_active` **up 3.4 points** while `_elapsed`
fell **14.4** and the kernel took **53% longer**.

### Why it is not in the Occupancy section

Because the Occupancy section answers a *different and legitimate* question:
"when this SM was working, how full were its warp slots?" That is the question a
register-pressure or shared-memory-capacity decision needs, and for it the
`active` denominator is correct — you do not want an answer contaminated by how
long some other SM sat idle.

The question "is my kernel wasting the machine" is a different question, and it
needs the `elapsed` form. `ncu` exposes it, but only via `--metrics`:

```
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed \
    --kernel-name myKernel ./prog.exe
```

**Ask for both, every time, and read the gap between them.** The gap is the
machine you are not using. Module 23's Exercise 3 constructs a case where it is
exactly a factor of two, because exactly half the SMs were given no work.

---

## Q4

> Two kernels, both DRAM Throughput 95.0%, one with bytes/sector 4.00 and one
> with 32.00, reading the same array and producing the same answer. Which is
> faster and by how much? Why does Speed of Light give them identical verdicts?
> What would make the slow one's DRAM throughput *drop*, and is that an
> improvement?

### Which is faster

`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` is how many of
each 32-byte sector's bytes the instruction actually consumed.

- **32.00** = every byte of every sector is used. 4 sectors per request,
  perfectly coalesced.
- **4.00** = one 4-byte float per 32-byte sector. 32 sectors per request, the
  saturated worst case for a 4-byte load (Module 5).

Both saturate the bus at 95% of 432 GB/s = **410 GB/s of traffic**. The 32.00
kernel converts all of it into useful data; the 4.00 kernel converts an eighth
of it. **The 32.00 kernel is about 8× faster.** Module 23's Exercise 1 measures
this exact pair at **7.54×**, the shortfall being in the *fast* kernel (a 64 MB
working set is only 1.3× the 48 MB L2 and the launch is short).

### Why Speed of Light cannot tell them apart

`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` is a property of the
**memory controllers**. It answers "is the bus busy?" and the bus does not know
which of the bytes it is moving you will read. The counter that knows is in the
**L1 tag stage**, `l1tex__t_sectors / l1tex__t_requests`, because that is where
a warp's 32 addresses are turned into sector requests and where the waste is
created.

**`dram__throughput` tells you the bus is busy. Sectors-per-request tells you
whether it is busy on your behalf. You need both, and Speed of Light only
carries the first — which is why the decision tree sends a "memory high,
compute low" verdict straight to Memory Workload Analysis rather than calling
the kernel finished.**

### What would make the slow one's DRAM throughput drop

**Fixing it.** Pack the data so the access is unit-stride. The traffic falls 8×,
the kernel finishes ~8× sooner, and the counter may read *lower* than 95%
because the shorter, smaller launch does not sustain the memory P-state as well
— Exercise 1 measured the fixed kernel at 387.4 GB/s against the broken one's
411.2 GB/s.

**Yes, it is an unambiguous improvement, and the DRAM throughput number got
worse.** This is the clearest reason not to optimize a percentage. Optimize
time, or optimize useful bytes per second; the % of peak columns are diagnostic
inputs, not objectives.

(A second thing that would lower it without being an improvement: reducing
concurrency until the kernel becomes latency-bound. Same counter, same
direction, opposite meaning. The `smsp__warps_eligible` row is what separates
the two cases.)
