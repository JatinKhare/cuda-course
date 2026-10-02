# Module 19 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

and the build line that hands you TODO 1 and TODO 2 without running anything:

```
nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise02_solution.exe exercise02_solution.cu
```

The run takes about a minute. It warms up for 1500 ms streaming plus 500 ms
compute (spec §12.4), probes the FFMA ceiling and will idle and re-warm if the
part is power capped (spec §12.5b), times nine configurations in nine rotated
sweeps with auto-scaled iteration counts, and validates in a separate pass.

---

## The kernel, and why this one

```cpp
float st[W];                       // W = 96 recursive stages, live for the whole kernel
for (int t = 0; t < L; ++t) {
    float v = x[t*nsig + sig];
    for (int r = 0; r < 2; ++r)
        for (int i = 0; i < W; ++i) {
            v     = fmaf(cA[i], v, st[i]);
            st[i] = fmaf(cB[i], v, 0.5f*st[i]);
        }
    y[t*nsig + sig] = v;
}
```

Three properties were needed and this shape has all three.

- **The register count is a design parameter, not an accident.** 96 state values
  are live across the entire time loop. `ptxas` cannot schedule them away.
- **The dependence chain is long and the ILP is low.** `v` threads serially
  through 192 fused multiply-adds per input sample. A thread cannot cover its
  own latency, so more warps genuinely help — the low-occupancy end of the sweep
  is a real loss, not an artefact.
- **It is compute-bound.** 768 FLOP per 8 bytes moved is 96 FLOP/byte against a
  machine balance of 43–44 (Module 16), so the DRAM system is not what is being
  measured. Spec §12 rule 8: a bandwidth-bound kernel makes a terrible occupancy
  demonstration.

It is also a real kernel shape — an IIR filter bank, a per-thread RNN cell, any
sequential-state recurrence — which is the category where register pressure and
occupancy actually fight.

---

## TODO 1 — `registerCapFor`

```cpp
int cap = REGS_PER_SM / (threads * minBlocks);   // 65536 / (T*B)
cap = roundDown(cap, REG_GRAN);                  // DOWN to a multiple of 8
if (cap > 255) cap = 255;
```

Two points of care, both of which the harness tests.

**The rounding goes down, not up.** The cap has to be a register count that is
actually allocatable and that still satisfies the request, so it is the largest
multiple of 8 that fits. `__launch_bounds__(128, 5)` gives `65536/640 = 102.4`,
truncated to 102, rounded down to **96**, and `ptxas` reports exactly 96. Round
*up* and you predict 104, which is wrong on three of the five spilling rows.

**255 is a hardware maximum per thread**, so the cap saturates there for small
`minBlocks`. `(128, 1)` predicts 255 and `ptxas` uses 177, which is not a
failure of the formula — the cap is an upper bound, and when it does not bind
`ptxas` uses whatever it likes.

Measured, this run:

| `__launch_bounds__(128, B)` | predicted cap | `numRegs` | spill B |
|---|---|---|---|
| 1 | 255 | 177 | 0 |
| 2 | 255 | 177 | 0 |
| 3 | 168 | 168 | 0 |
| 4 | 128 | 122 | 0 |
| 5 | 96 | **96** | 72 |
| 6 | 80 | **80** | 136 |
| 8 | 64 | **64** | 200 |
| 10 | 48 | **48** | 264 |
| 12 | 40 | **40** | 296 |

Every row where `ptxas` had to spill is a row where it is sitting exactly on the
predicted cap — which is the evidence that the formula is the real one and not a
coincidence.

Note that the cap formula uses the **aggregate** 65536 figure, not the
four-slice model Exercise 1 needed for *placement*. Those are two different
computations by two different pieces of the system: `ptxas` sizes its budget
from the whole file, and the hardware places warps slice by slice. They happen
to agree here on every row, but they are not the same arithmetic and you should
not expect them always to.

---

## TODO 2 — where the cap binds, and the gap that matters

```cpp
for (int b = 1; b <= MAX_BLOCKS_PER_SM; ++b)
    if (registerCapFor(threads, b) < unconstrainedRegs) return b;
```

The unconstrained build uses **177** registers. `registerCapFor(128,1) = 255`
and `registerCapFor(128,2) = 255`, both above 177; `registerCapFor(128,3) = 168`
is the first below it. Answer: **3**.

**The question deliberately is not "where does it spill".** That is not
computable with a pencil and the measurement says why: the cap binds at `B = 3`
and the first spill is at `B = 5`. Between those two bounds `ptxas` surrendered
**177 → 122 = 55 registers per thread**, nearly a third of the budget, purely
by rescheduling: shortening live ranges, re-materialising constants, reordering the
unrolled cascade. All of that is free. Only when the live set of the inner loop
itself — here the 96 filter states — stops fitting does it have to put something
in local memory.

This is the single most useful thing in the exercise. "Constraining the register
budget" and "causing a spill" are different events separated, on this kernel, by
two whole steps of the sweep, and **all of the cheap occupancy lives in the
gap**. A reader who assumes the two coincide will never try `B = 3` or `B = 4`,
which is where the win is.

A plausible wrong answer is **5**, from trying to predict the spill. The harness
scores the cap, says so in the TODO text, and prints the spill point next to it
so the distinction is visible in the output.

---

## TODO 3 — how bad is 100 % occupancy?

**`PRED_MAXOCC_BUCKET = 4`: 5× to 20× slower.** Measured across four clean
runs: **7.15×, 7.23×, 7.42×, 8.15×**. Bucket 4 spans 5–20×, so the answer has
better than 1.4× of margin on both sides.

Most readers answer 2 or 3. The mechanism is not subtle once you see it: at
`__launch_bounds__(128, 12)` the compiler has 40 registers for a kernel with 96
live state values, so 296 bytes per thread go to local memory, and Module 4
established that local memory is DRAM with an L1 line in front of it. The inner
loop then performs an `LDL`/`STL` pair for most of its 192 fused operations per
sample. Occupancy is at its theoretical maximum and the FP32 pipes are idle.

Module 18 measured the same construction on a register-tiled GEMM and got
**18.7×** at 256 threads and **14.1–14.6×** at 128. This kernel gives 7–8×
because its spill is smaller relative to its arithmetic. The number is
kernel-specific; the sign is not.

---

## TODO 4 — is the compiler's own choice the best?

**`PRED_UNCONSTRAINED = 2`: no, and it is on the low-occupancy side.**
Measured ratios of best to unconstrained over seven runs: **1.09× to 1.37×**.
The harness calls the compiler's choice optimal only below 1.03×, so the answer
has ~6 points of margin even on the tightest run observed.

This is worth dwelling on, because it is the opposite of the usual complaint
about `ptxas`. The folklore is that the compiler uses too many registers and you
have to rein it in; what the folklore means by "rein it in" is usually a
`-maxrregcount` that causes a spill. Here `ptxas`'s unconstrained choice is
genuinely too register-hungry — 177 registers for a kernel whose live set is 96
state values plus a handful of temporaries — and the first two steps down cost
**nothing at all** in spill terms while buying 8 more warps per SM.

`ptxas` is optimising for the kernel's own instruction schedule with no model of
how many of these blocks you intend to run. It cannot know that two blocks per
SM is not enough to cover a 192-deep dependence chain. You can.

---

## TODO 5 — the operating point, and the rule

**`CHOSEN_MINB = 4`, `CHOSEN_RULE = 2` (the highest occupancy that costs no
spill).**

The measured sweep, one representative run:

| `__lb__(128,B)` | regs | spill B | blk/SM | warps | occ | Gelem/s |
|---|---|---|---|---|---|---|
| 1 | 177 | 0 | 2 | 8 | 16.7 % | 9.22 |
| 2 | 177 | 0 | 2 | 8 | 16.7 % | 9.19 |
| 3 | 168 | 0 | 3 | 12 | 25.0 % | 11.44 |
| **4** | **122** | **0** | **4** | **16** | **33.3 %** | **12.62** |
| 5 | 96 | 72 | 5 | 20 | 41.7 % | 8.83 |
| 6 | 80 | 136 | 6 | 24 | 50.0 % | 3.64 |
| 8 | 64 | 200 | 8 | 32 | 66.7 % | 2.18 |
| 10 | 48 | 264 | 10 | 40 | 83.3 % | 1.47 |
| 12 | 40 | 296 | 12 | 48 | 100.0 % | 1.55 |

**The optimum is interior**, by a factor of 7–9 on one side and 1.1–1.4 on the
other. The top four rows are all within 1.0–1.37× of each other and their
ordering is *not* stable run to run — across six clean runs `B = 4` was fastest
in five and `B = 3` in one — which is why the operating-point gate accepts
anything within 20 % of the measured best. As a fraction of each run's own best,
over seven runs:

| row | range |
|---|---|
| `B = 1` | 73–89 % |
| `B = 2` | 73–87 % |
| `B = 3` | 82–100 % |
| **`B = 4`** | **85–100 %** |
| `B = 5` | **59–72 %** |

`B = 4` is the robust answer: it never fell below 85 % of its own run's best.
`B = 3` clears the gate in every run observed but came within 2 points of it
once. The first spilling row never exceeded 72 %, so the gate at 80 % sits in an
8-point gap on the reject side (spec §12.5d).

Because `B = 1` can reach 89 % of the best on a lucky run, the harness also
requires the choice to be **self-consistent with TODO 4**: if you answered that
the compiler's unconstrained build is not the fastest, you may not then ship it
as your operating point.

**Why rule 2 is the only one that works.** The harness evaluates all six rules
against the measured resource table — not against the throughput column, which
is the point:

| rule | selects | throughput |
|---|---|---|
| 1 highest occupancy | `B = 12` | 1.55 (12 % of best) |
| **2 highest occupancy with no spill** | **`B = 4`** | **12.62** |
| 3 whatever the compiler picked | `B = 1` | 9.22 (73 %) |
| 4 highest occupancy with spill < 100 B | `B = 5` | 8.83 (70 %) |
| 5 lowest occupancy | `B = 1` | 9.22 |
| 6 the middle of the sweep | `B = 5` | 8.83 |

Rule 4 is the interesting failure. It is a *reasonable* rule — it encodes
"small spills are fine", which Module 18 measured to be true — and it is wrong
here, because what spilled matters more than how much. Module 18's winning
80-byte spill was addressing and scheduling temporaries; the 72 bytes at `B = 5`
here are 18 of the 96 filter states, read and written twice per input sample.
`localSizeBytes` cannot tell those two apart; only the SASS can. **The cliff is
not at the first spilled byte, it is at the first spilled byte of the inner
loop's live set**, and no counter reports that.

So rule 2 is the right one for a kernel like this — but the honest statement is
narrower than "never spill". It is: *when the live set you would spill is the
one the inner loop touches every iteration, do not spill; when it is not, a
spill that buys a block can win.* Rule 2 is the conservative form of that, and
it costs you nothing here because `B = 4` is already at the optimum.

**A note on what this sweep deliberately does not vary.** Every row is a
128-thread block — four warps, one per processing block, so the four schedulers
are loaded identically in all nine configurations. The only thing that changes
is the register budget, and therefore the number of blocks. That is not an
accident: Module 20 reports a throughput sawtooth against blocks per SM on its
own harness, and a sweep that moved the block size *and* the register budget at
once could not tell the two apart. Lesson §4b tests the effect directly on this
hardware (13 block shapes at fixed warps/SM, rotated, min-of-N: spread
0.973–1.006, i.e. no effect) and gives the sweep-design rule: **change one axis
at a time, and prefer block sizes that are a multiple of 128 threads.**

**Why rules have to be about the resource table.** You get the resource table
from `-Xptxas -v` and `cudaFuncGetAttributes` **before you run anything**. A
rule that reads the throughput column is not a rule, it is a measurement, and it
does not transfer to the next kernel. The whole exercise is the construction of
something you can apply to a kernel you have not yet timed.

---

## Performance reasoning

Read the sweep as two curves crossing.

**Latency tolerance** improves with warps. The exchange-rate column shows it
costing roughly 8–26 registers per 4 warps over the useful range, and the
throughput gain from 8 warps to 16 is 1.37×, which is the shape of Module 1's
pointer-chase curve on a 192-deep FFMA chain rather than on memory. It is
*decelerating*: 8 → 12 warps is worth 1.24×, 12 → 16 another 1.10×.

**Spill cost** is zero until the live set stops fitting and then grows fast:
0, 0, 0, 0, then 72 B (×0.70), 136 B (×0.29), 200 B (×0.17), 264 B (×0.12).
Each extra spilled state value is `L × R = 512` extra local-memory round trips
per thread.

A decelerating gain against an accelerating loss has an interior maximum, and
the maximum is at the last configuration before the loss turns on. That is rule
2, stated geometrically.

One more detail worth noticing: `B = 10` (83.3 % occupancy) is **slower** than
`B = 12` (100 %), 1.47 against 1.55 Gelem/s, reproducibly. Occupancy is not even
monotone in this region. Module 18 found the same non-monotonicity at the other
end of its sweep (25 % → 33 % a win, 33 % → 67 % a 9× loss). Any tuner that
hill-climbs on occupancy is walking a surface with no useful gradient.

---

## Expected output

```
-- warming up: 1500 ms streaming, then 500 ms compute ------------------
  FFMA ceiling probe: 17297 GFLOP/s

-- the sweep: same source, nine register budgets -----------------------
  __lb__(B)   regs yourCap  spillB   blk   warps      occ        ms  Gelem/s
          1    177     255       0     2       8    16.7%    1.7051    9.224
          2    177     255       0     2       8    16.7%    1.7123    9.186
          3    168     168       0     3      12    25.0%    1.3751   11.438
          4    122     128       0     4      16    33.3%    1.2463   12.620   <- fastest
          5     96      96      72     5      20    41.7%    1.7811    8.831
          6     80      80     136     6      24    50.0%    4.3168    3.644
          8     64      64     200     8      32    66.7%    7.2161    2.180
         10     48      48     264    10      40    83.3%   10.7196    1.467
         12     40      40     296    12      48   100.0%   10.1588    1.548

-- scoring ------------------------------------------------------------
  unconstrained build uses 177 registers; the cap first BINDS at
  __launch_bounds__(128, 3); you said 3  -> correct
  ...and ptxas does not SPILL until (128, 5). Everything between those
  two bounds is rescheduling: 55 registers per thread given up for free.
  register cap formula matches ptxas on every spilling row: yes
  fastest build is __launch_bounds__(128, 4) at 12.620 Gelem/s
  100%-occupancy build is 8.15x slower  -> bucket 4, you said 4  correct
  compiler's unconstrained build is 1.37x off the best -> answer 2, you said 2  correct
  your operating point minBlocks=4: 12.620 Gelem/s = 100.0% of best  ok
  your rule 2 selects minBlocks=4 from the measured table  ok

  SCORE: 10/10

-- validation (second, untimed pass) -----------------------------------
  worst relative error over all 9 builds: 0  (ok)

OVERALL: PASS
```

**Variance across seven runs.** Absolute throughput on the best row moved
10.2–12.8 Gelem/s on a healthy machine, and one run — taken deliberately
without a cool-down after another Module 19 program — came in at **5.25
Gelem/s, 2.4× slow across every row**, with the FFMA ceiling probe still
reporting a healthy 16842 GFLOP/s. **It scored 10/10 anyway.** Every scored
quantity in this file is a ratio taken inside one rotated sweep, so a uniform
slowdown cancels: the 100 %-occupancy penalty was 7.15–8.83× (bucket 4 in every
run) and the unconstrained penalty 1.09–1.37× (answer 2 in every run).
Register counts, spill figures and block counts were byte-identical in every
run. That is spec §12.5's "report ratios, not absolutes" working as intended —
and it is also the reason Exercise 3, whose gates compare two *different*
launches rather than two columns of one sweep, needed a much stronger
operating-point guard.

**The validation prints a worst relative error of exactly 0**: all nine builds
and the CPU reference evaluate the same `fmaf` sequence in the same order, so
the results are bit-identical. That is a useful property to have in a sweep
whose whole subject is the compiler changing its mind — it means nothing in the
table is a numerical artefact.

---

## The result that matters

**The fastest build of this kernel runs at 33 % occupancy; its 100 %-occupancy
sibling, same source, is 7–8× slower; and the compiler's own unconstrained
choice is 1.1–1.4× slower than both.** The optimum is interior and neither
endpoint is close to it. What makes it findable before you measure is the
distinction TODO 2 forces: the register cap binding is not the same event as
`ptxas` spilling, and on this kernel there are two whole steps of free
occupancy between them. The rule that finds the optimum — *the highest occupancy
that costs no spill* — is computable from `-Xptxas -v` output alone, which is
the only kind of rule worth having.

**Variation to try.** Change `RPASS` from 2 to 1, halving the arithmetic per
input sample while leaving the 96 live state values alone, and re-run. The
kernel becomes much closer to memory-bound, the dependence chain halves, and the
low-occupancy end stops being a loss — which moves the optimum towards the
compiler's unconstrained choice and makes rule 3 look correct. Then put it back
to 2 and note that you have just changed the right answer by changing the
kernel, not the hardware. That is why §8's decision rule starts with "is this
kernel latency-bound at all" and not with "what is its occupancy".
