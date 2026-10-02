# Module 20 — Latency Hiding

> Prerequisites: Module 1 (eligible/stalled warps, Little's Law, the SM's four
> schedulers), Module 4 (the measured latencies: L1 40.5, L2 241.3, DRAM 575
> cycles), Module 11 (MLP, coarsening, the 2.1×/1.00× contrast), Module 19
> (what limits occupancy — this module assumes you can already compute how many
> warps fit).
> What this module gives you: the mechanism. Why latency tolerance works at
> all, exactly what a stalled warp is waiting on, how much of it you can buy
> with instruction-level parallelism instead of occupancy, and a decision
> procedure for a slow kernel.

Module 19 answered *how many warps fit*. This module answers the two questions
that actually matter: **how many do you need**, and **what do you do when you
cannot have them**.

The whole module rests on one equation and two measured numbers.

---

## Concept

### 1. Little's Law is the design document of the machine

For any stable system in which items arrive, spend some time inside, and leave:

```
concurrency = throughput × latency
```

Module 1 stated it. Here we use it as a calculator, four times, and each time
it produces a number you can check against the hardware.

**Use 1 — the FP32 pipe.** On this GPU each of the four warp schedulers in an
SM issues at most one instruction per clock, so the throughput of the FP32 pipe
is **1 warp-instruction per cycle per scheduler**. `example01.cu` measures the
latency of a dependent `FFMA` on a single warp with nothing else resident:

```
 ILP         cycles     cyc/FFMA      wall ms      clk GHz
   1        4148093        4.051       2.3193        1.789
   2        2104092        2.055       1.1173        1.883
   3        1420094        1.392       0.7642        1.858
   4        1092097        1.067       0.5941        1.838
   5        1092095        1.071       0.6267        1.743
   6        1076097        1.068       0.6062        1.775
   8        1092097        1.067       0.5919        1.845
  10        1064090        1.064       0.5748        1.851
```

**Latency 4.05 cycles. Throughput 1.07 cycles per instruction.** Little's Law:

```
concurrency = 1 instr/cycle × 4 cycles = 4 independent FFMAs in flight
              per warp scheduler
```

and the table obeys it exactly: 4.051 → 2.055 → 1.392 → 1.067, which is
`4.05/1`, `4.05/2`, `4.05/3`, `4.05/4`, and then *flat*. One warp with four
independent chains saturates an Ada warp scheduler's FP32 issue port. One warp
with one chain gets 25% of it.

**This is the module in one table.** Four is the number. Everything else is
how you supply it.

**Use 2 — the memory system.** `example02.cu`:

```
measured streaming read ceiling      419.8 GB/s  (97.2% of 432 pin peak)
Module 4 dependent-load latency      575.0 cycles = 303 ns at 1.90 GHz
Little's Law  bytes in flight        127047 B  = 124 KB
              32 B sectors in flight   3970
              per SM                    99.3 sectors
              128 B warp-loads/SM       24.8
```

Now the supply side. An SM holds 1536 threads = **48 warps**. If every resident
warp has exactly one coalesced 128-byte load outstanding, that is 48 warp-loads
= 192 sectors per SM, against a demand of 99. **Ratio 1.93.**

Read that ratio as the design statement it is. The 48-warp slot count is not an
arbitrary number: it is roughly twice what a fully-occupied SM needs to keep
the DRAM interface busy with one outstanding load each. The factor of two is
margin — for warps that are doing arithmetic rather than loading, for warps
sitting at a barrier, for non-uniform behaviour.

And note the per-*thread* figure: 124 KB spread over 61,440 resident threads is
**2.1 bytes per thread**. Less than one `float`. "How many loads should each
thread have in flight?" is not a well-posed question until you say how many
threads there are. The only well-posed quantity is the product.

**Use 3 — reading it backwards.** Exercise 3 pins a loop to genuinely one
outstanding load per thread with `#pragma unroll 1`, at 5120 threads, and
measures:

```
baseline  <<<40,128>>>        7.6715 ms     68.3 GB/s   15.8% of pin peak
supplied concurrency Q0         20480 B =   20.0 KB
effective latency L             299.7 ns =   569 cycles at 1.90 GHz
required concurrency Q*        123335 B =  120.4 KB
shortfall Q*/Q0                  6.02x
predicted baseline BW = Q0/L =    68.3 GB/s   (measured    68.3)
```

Little's Law inverted recovers **569 cycles** of DRAM latency from a bandwidth
measurement — against Module 4's independently measured **575**. Two
experiments, four years of hardware documentation apart in style, agreeing to
1%. The law is not a metaphor.

**Use 4 — the one you will actually do.** Given a kernel, compute what it has
in flight and divide by what the machine needs. That ratio *is* the diagnosis.

### 2. The two currencies: TLP and ILP

A warp scheduler needs N independent instructions in flight. There are exactly
two places to get them.

| | **TLP** (occupancy) | **ILP / MLP** (per-thread) |
|---|---|---|
| Source | more resident warps | more independent work inside one warp |
| Costs | register file, shared memory, warp slots | registers, instruction cache, code size |
| Owned by | Module 19 | this module |
| Set by | the launch configuration and the kernel's footprint | the kernel body |
| Fails when | the problem has no more parallel work, or registers run out | the algorithm is a dependency chain |

They are **substitutes**, and the exchange rate is not a judgement call — it is
`latency × throughput`. `example01.cu` measures the whole 5×5 surface:

```
GFLOP/s   (rows: resident warps per scheduler; cols: independent chains)
warps/s       ILP=1      ILP=2      ILP=4      ILP=8      ILP=16   ILP 1->16
1               5385      10562      20211      20237      19617       3.64x
2              10724      20832      20879      20965      20955       1.95x
4              20794      20775      20822      20955      20747       1.00x
8              19931      19785      20140      20131      20113       1.01x
12             19914      19517      20202      20157      20122       1.01x

same table as % of the best cell (20965 GFLOP/s)
warps/s       ILP=1      ILP=2      ILP=4      ILP=8      ILP=16
1                26%        50%        96%        97%        94%
2                51%        99%       100%       100%       100%
4                99%        99%        99%       100%        99%
8                95%        94%        96%        96%        96%
12               95%        93%        96%        96%        96%
```

(Run-to-run the whole table moves by 5-8% with thermal state; a second run of
the same binary gave 4990 / 9724 / 18454 along the top row and 24 / 46 / 88 %.
The *ratios* within a sweep reproduce to ~1%, which is the quantity the
argument uses. Spec §12 rule 5.)

Three readings, and they are the module's headline:

1. **The top-left cell is 24-26% of peak.** One warp per scheduler running one
   dependent chain. Three quarters of the machine's issue capacity is lost to a
   four-cycle pipeline.
2. **Either axis fixes it, and they fix it to the same place.** Going right
   along the top row: 26 → 50 → 96%. Going down the left column: 26 → 51 →
   99%. Neither axis is privileged.
3. **The surface is a function of the product.** Every cell with
   `warps/scheduler × ILP ≥ 4` is at 88–100%; every cell below it is at
   `product/4` of peak, to within the noise. The exchange rate is exactly
   **one unit of ILP buys one resident warp per scheduler = four resident
   warps per SM.**

The same experiment on the memory side (`example02.cu`, `C` = independent
`float4` loads hoisted per thread):

```
warps/s       C=1       C=2       C=4       C=8       C=16      C=32
1            252.5     377.2     408.7     409.9     411.1     411.0
2            389.5     409.7     411.6     411.6     412.0     412.0
4            409.3     411.9     412.1     412.3     412.4     412.3
8            411.0     412.3     412.5     411.8     411.0     419.8
12           411.6     411.5     411.1     411.2     410.9     410.1

the same cells, indexed by the PRODUCT  (warps/sched x C)
   product      cells  mean GB/s     spread
         1          1      252.5       0.0%
         2          2      383.3       3.2%
         4          3      409.2       0.2%
         8          4      411.1       0.5%
        16          4      411.8       0.3%
        32          4      411.9       0.4%
        64          3      412.0       0.2%
       128          2      411.6       0.3%
```

**Cells with the same product agree to within 0.2–0.5%.** Cells in the same row
or column differ by up to 63%. Occupancy and MLP are not two knobs; they are
two handles on one knob.

This is the general form of Module 11's single measurement — hoisted loads vs
`#pragma unroll 1` was **2.1× at 1 block/SM and 1.00× at 8 blocks/SM**. Module
11 measured two cells of this table; here is the surface they lie on.

### 3. Why you would ever prefer ILP

If they are substitutes, why not always buy occupancy? Four reasons, all of
which you have already met:

- **There may not be enough parallel work.** Exercise 2's kernel C has 5120
  independent items. 5120 threads is 1 warp per scheduler and no launch
  configuration changes that. ILP is the only lever that exists.
- **Occupancy may be unaffordable.** Module 18 measured a register-tiled GEMM
  at 33% occupancy running **18.7× faster** than the same source forced to 100%
  — because the registers that buy occupancy are the registers holding the
  accumulators. A kernel that is latency-tolerant *by construction* (64
  independent accumulator chains per thread) does not need warps. That result
  is unreadable until you have this module's table: at ILP 64 the entire
  occupancy axis has collapsed to "1 warp per scheduler is enough."
- **ILP is free where occupancy is not.** Adding a second independent chain
  costs one register. Adding a warp costs 32× the kernel's whole register
  footprint.
- **Occupancy has a ceiling and ILP effectively does not.** 48 warps/SM is a
  hard cap. Nothing caps how many independent accumulators a thread holds
  except the register file, which is the same budget seen from the other side.

And one reason to prefer occupancy: **ILP must be visible to the compiler.**
Occupancy is a property of the launch; ILP is a property of the generated code,
and §7 below is a catalogue of ways it evaporates.

---

## Hardware Mental Model

### 4. What a stalled warp is actually waiting on

Module 1 gave you the binary: each cycle, a scheduler classifies its resident
warps as **eligible** or **stalled**, picks one eligible warp, and issues. That
binary is too coarse to act on. The hardware distinguishes the *reason*, and
the reason determines the fix.

Here is the taxonomy, as a small table you should memorise. The third column is
the one people get wrong.

| Stall reason | The warp is waiting for | More occupancy helps? | What actually fixes it |
|---|---|---|---|
| **long scoreboard** | a global or local memory return | **yes, strongly** | more warps, or more outstanding loads per warp (MLP), or fewer/wider loads |
| **short scoreboard** | shared memory, or a fixed-latency unit (MUFU, texture) | yes, partly | remove bank conflicts (M7), fewer shared accesses, wider `LDS` |
| **barrier** | other warps of its block to reach `__syncthreads()` | **no** — more blocks, yes; more warps per block, no | fewer barriers, smaller blocks, double buffering, warp-level primitives |
| **execution dependency** (`wait`) | a previous *arithmetic* result, fixed latency | yes, up to 4 warps/scheduler | **ILP: independent chains** |
| **instruction fetch** (`no_instruction`) | the instruction cache | no | smaller loop bodies; stop unrolling |
| **MIO / math / LG throttle** | a shared issue queue that is full | **no — it makes it worse** | fewer instructions of that type |
| **not selected** | nothing; it is eligible and lost the arbitration | **no — this IS too much occupancy** | fewer instructions per unit of work |
| **drain / tail** | the kernel to end | no | load balance, smaller blocks, more waves |

Three of these rows are worth dwelling on.

**`long scoreboard` is the one everybody's mental model is about.** A global
load sets a scoreboard bit; the consumer of that register is not eligible until
it clears. Module 4 priced the wait: 40.5 cycles from L1, 241.3 from L2, 575
from DRAM. This is where occupancy earns its reputation.

**`execution dependency` is the one nobody's mental model is about.** `FFMA`
has a *four*-cycle latency, not a hundred, so it feels like it cannot matter.
The top-left cell of the table in §2 says it costs 76% of the machine. It is
cheap to fix and nearly free to ignore — which is why kernels sit in it.

**`not selected` means you have too much occupancy, not too little.** A warp
reported as `not selected` was ready to go and simply lost: another warp on the
same scheduler was picked. Since a scheduler can issue at most one instruction
per clock, the sum over all warps of (issued + not-selected + stalled) cycles is
fixed. If `not selected` dominates, the kernel's problem is that it executes too
many instructions, and the only cure is to execute fewer. Adding warps raises
`not selected` and changes nothing else. Exercise 2's kernel D is this case:
64 IEEE divides per element at 12 warps per scheduler. Replacing the IEEE
divide with `__fdividef` removes 256 instructions out of 440 from the function
and is worth **1.85–2.01×**; adding warps from 4 to 12 is worth 1.10×.

**ARCHITECTURE-SPECIFIC NOTE.** The names above are Nsight Compute's. The
metrics are
`smsp__warp_issue_stalled_<reason>_per_warp_active.ratio` for
`<reason>` in `long_scoreboard`, `short_scoreboard`, `barrier`, `membar`,
`wait`, `no_instruction`, `imc_miss`, `mio_throttle`, `math_pipe_throttle`,
`lg_throttle`, `tex_throttle`, `drain`, `sleeping`, `misc`, `not_selected`,
`selected`, and the packaged form is

```
ncu --section WarpStateStatistics --section SchedulerStats ./yourkernel.exe
```

**`ncu` cannot be run on this machine** — every invocation fails with
`ERR_NVGPUCTRPERM`, which needs a one-time elevated fix. Module 23 teaches the
tool. Everything in this module is therefore *constructed and measured
directly* rather than read off a counter: each example builds the stall it is
talking about and times it. That is a slower way to learn the taxonomy and a
much more convincing one.

### 5. Dependency chains and the critical path

A sequence of instructions in which each reads what the previous one wrote is a
**dependency chain**. Its execution time is

```
T_chain = (chain length) × (instruction latency)
```

and it is completely independent of how wide the machine is. Four independent
chains of length L take exactly as long as one chain of length L, as long as
`4 ≤ latency × throughput`. The **critical path** of a kernel is its longest
dependency chain; everything else can, in principle, be hidden behind it.

This is why `cyc/FFMA` in §1 falls as `1/ILP` and then stops: with `C` chains
interleaved, the per-instruction cost is `max(latency/C, 1/throughput)`.

Two corollaries you will use:

- **Reassociation is the price of admission.** Splitting
  `s += a[0]; s += a[1]; ...` into four partial sums changes the answer in the
  last bits, because floating-point addition is not associative. That is not a
  bug, but it does mean your validator must be written for it. Module 16's rule
  applies: scale the tolerance by the accumulated magnitude `S`, not by the
  result. Exercise 2's kernel C does exactly this and the measured
  `err/(γ_K·S)` is 0.058 — well inside the bound, and nowhere near what a
  tolerance scaled by `|result|` would have allowed.
- **Some chains cannot be broken.** A pointer chase, a first-order recurrence,
  a scan's carry — these have no independent work by construction. Module 13
  broke one of them (the scan) with an *algorithm*, not with ILP. When the
  critical path is the algorithm, the answer is a different algorithm.

### 6. Memory-level parallelism, and where it stops paying

**MLP** is ILP applied to loads: the number of memory requests one thread has
outstanding at once. Module 11 defined it; here is the cost curve.
`example02.cu` part C puts a single warp on the machine, issues `C` independent
L2-resident loads, and serialises the steps so the measurement is a latency and
not a throughput:

```
   C       cycles     cyc/step     cyc/load     marginal
   1       290445        290.4        290.4           -
   2       300664        300.7        150.3        10.2
   3       311947        311.9        104.0        11.3
   4       318884        318.9         79.7         6.9
   6       345478        345.5         57.6        13.3
   8       371376        371.4         46.4        12.9
  12       411229        411.2         34.3        10.0
  16       472279        472.3         29.5        15.3
  32       745412        745.4         23.3        17.1

fit over C >= 4 :  cycles/step = 245.8 + 15.25 * C
```
(Two runs: intercepts 240.9 and 245.8, slopes 15.01 and 15.25.)

Read the fit. The intercept is **241–246 cycles** — Module 4 measured the L2
dependent-load latency at **241.3**. The probe recovers it to within 2% without
being told what it is looking for. The slope is **15 cycles**: what each
*additional* outstanding warp-load costs in issue slot, address generation and
L2 transport.

So the answer to "how many loads can a thread have in flight?" is: **there is
no hard queue limit up to 32, but there is a crossover.** While `15·C ≪ 241`
the kernel is latency-bound and each extra load is nearly free. Past
`C ≈ 245/15 ≈ 16` the kernel is throughput-bound and extra MLP buys nothing
because there is no longer any latency left to hide. The practical knee is
around 8.

**Three caveats, all of which have bitten this course:**

1. `C` counts loads in the *source*. Nothing stops the compiler from issuing
   the next loop iteration's loads before this one's consumers retire, so the
   real in-flight count is ≥ `C`. That is why `example02.cu` part B's knee
   (product 4, ≈80 KB in flight) is *below* §1's 124 KB demand: it is a lower
   bound on concurrency, not a measurement of it. Exercise 3's `#pragma
   unroll 1` kernel, which genuinely has one load in flight, lands on 569
   cycles and reconciles the two.
2. A `float4` load occupies **one** outstanding-request slot and carries four
   times the payload. Vectorizing multiplies your concurrency in bytes without
   costing a slot — which is exactly why Module 11 measured `float4` at 1.11×
   at low occupancy and 1.00× at high. It is a **latency** optimization.
3. Hoisting only produces MLP if the loads precede *all* the consumers.
   `#pragma unroll 1` is the instrument that removes it; a `v[c]` array indexed
   by a runtime variable is the accident that removes it (the array goes to
   local memory — Module 4).

### 7. Unrolling, and the four ways ILP evaporates

Unrolling is how you ask for ILP. It is not how you get it.

**Way 1 — the compiler was going to do it anyway, differently in each
configuration.** Measured here, and it is why `example01.cu` fixes the number
of FFMAs per outer-loop iteration at 256 for every `C`. With the obvious
formulation instead (`for (t) for (i < C)`), `ptxas` unrolled the `C = 2`
instantiation by 1 and every other instantiation by 3, giving 64 FFMAs between
branches instead of 192 — and the `C = 2` column came out 25% low for a reason
that has nothing to do with ILP. Module 11 recorded a reproducible anomaly at
`C = 2` in its MLP sweep that its story did not explain; this is the same
family of artefact, and the fix is to pin the work per branch rather than the
trip count.

**Way 2 — common subexpression elimination.** Module 18's headline instance:
64 source-level reads CSE'd down to 12, making the deliberately "unoptimized"
kernel **1.05× faster** than the optimized one. If your two chains are provably
equal, you have one chain. (Tested here: nvcc does *not* merge FFMA chains with
identical seeds — fp arithmetic blocks the proof it would need — but it will
merge anything it can prove, and loads are much easier to prove.)

**Way 3 — register pressure.** This is Module 19's half of the trade and the
reason unrolling backfires: `C` independent chains need `C` live registers, `C`
hoisted loads need `C` destination registers, and past the point where the
register file binds, every extra unit of ILP costs you warps. Module 18
measured the extreme: forcing 100% occupancy on a register-tiled GEMM spilled
the accumulators to local memory — which Module 4 established is DRAM — for an
**18.7×** loss. The trade is only favourable while you are below the
`latency × throughput` requirement. Above it, ILP is pure cost.

**Way 4 — instruction cache.** Unrolling a loop 32× produces a loop body that
may not fit in the SM's instruction cache, converting `execution dependency`
stalls into `no_instruction` stalls. This is the stall reason nobody checks.

**Standing rule, from spec §12 rule 11 and confirmed again here: if your
argument depends on an instruction existing, look at the SASS.**
`example01.cu`'s four-chain kernel, verified:

```
/*0170*/   IADD3 R7, R7, 0x1, RZ ;
/*0180*/   FFMA R9,  R9,  R6.reuse, 9.9999999747524270788e-07 ;
/*0190*/   FFMA R11, R11, R6.reuse, 9.9999999747524270788e-07 ;
/*01a0*/   FFMA R13, R13, R6.reuse, 9.9999999747524270788e-07 ;
/*01b0*/   FFMA R15, R15, R6.reuse, 9.9999999747524270788e-07 ;
/*01c0*/   FFMA R9,  R9,  R6.reuse, 9.9999999747524270788e-07 ;
/*01d0*/   FFMA R11, R11, R6.reuse, 9.9999999747524270788e-07 ;
...
```

Four accumulators, `R9 R11 R13 R15`, round-robin, 256 `FFMA` and zero spills in
the function. Compare the `C = 1` instantiation, same 256 `FFMA`, all on `R9`:

```
/*0150*/   FFMA R9, R9, R6, 9.9999999747524270788e-07 ;
/*0160*/   FFMA R9, R9, R6, 9.9999999747524270788e-07 ;
/*0170*/   FFMA R9, R9, R6, 9.9999999747524270788e-07 ;
```

Identical instruction count, identical register count, **3.8× apart**. Nothing
but the register numbers distinguishes them, and no resource counter reports
the difference. This is the module's existence proof that performance is not a
function of the resource table.

### 8. A measured surprise: resident warps are not fungible

While building the occupancy axis for `example01.cu` this module found an
effect that is worth recording because it will corrupt anyone else's occupancy
sweep.

Varying occupancy by changing the number of 128-thread blocks per SM, with a
pure dependent-FFMA kernel at ILP = 1, gives a **sawtooth**:

| blocks/SM (= warps/scheduler) | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| fraction of the issue ceiling | 0.99 | **0.63** | 0.74 | 0.86 | 0.96 | 0.75 | 0.81 | 0.89 | 0.96 |

The envelope fits `W / (4·⌈W/4⌉)` on every point. Block placement was verified
even (`%smid` histogram: exactly `B` blocks on every one of the 40 SMs for
`B = 1..12`), and the effect is cycle-accurate, not a clock artefact — a
`clock64()` trace gives 0.625 instructions/cycle/scheduler at `B = 5` against
1.008 at the control.

The control is the interesting part. **The same 20 warps per SM delivered as
*one* 640-thread block run at 1.008 instructions/cycle; delivered as *five*
128-thread blocks they run at 0.625.** Same warp-slot distribution over the
four sub-partitions (verified by reading `%warpid`), same instruction stream,
same total threads, 1.53× apart. A per-block trace of `clock64()` shows the
five blocks finishing in two groups rather than together.

The honest statement is that the warp scheduler's arbitration has structure at
CTA granularity that the eligible/stalled model does not capture, and that
**resident warps delivered as many small blocks are worth less than the same
warps delivered as few large ones** on a dependency-bound kernel. The practical
consequences:

- when sweeping occupancy by block count, stay on `blocks/SM ∈ {1, 2, 3, 4, 8, 12}`
  — every example and exercise in this module does;
- a block whose warp count is not a multiple of 4 loads the four schedulers
  unequally and the busiest one sets the time (measured: a 160-thread block, 5
  warps, at 4 blocks/SM runs at 0.56 of the ceiling, exactly `20/32`);
- this is a dependency-bound effect. The memory table in §2 does not show it.

Reported, not smoothed. Module 19 should know about it.

### 9. The decision procedure

This is the skill the module exists to produce. You have a slow kernel and no
profiler. Six steps, in order, each of which is one or two launches.

**Step 0 — compute the floor.** Module 11's discipline: count the compulsory
bytes, divide by 410 GB/s, and count the FLOPs, divide by ~20,000 GFLOP/s. The
larger of the two is your floor. If you are within 1.2× of it, stop — there is
nothing here. Module 21 turns this step into the roofline and gives you the
machine-balance argument that says *which* of the two numbers to use.

**Step 1 — does occupancy help?** Launch at 1, 2, 4, 8, 12 blocks per SM (or
whatever the kernel permits; see §8 before you pick the shapes). This is the
single most informative measurement in GPU performance work and it costs five
launches.

- **Time falls as you add warps, and you ran out of warps before it flattened:**
  you are **concurrency-starved**. Go to step 2.
- **Time is flat:** concurrency is not the constraint. Go to step 3.
- **Time *rises* as you add warps:** you are in Module 18's inversion.
  Something you need — registers, shared memory, L2 residency — is being traded
  away to buy warps you do not need. This is Module 19's problem; the fix is
  `__launch_bounds__` in the *restrictive* direction.

**Step 2 — concurrency-starved: which kind, and can you buy more?**
Compute `Q = threads × loads_in_flight × bytes_per_lane` and compare with
`Q* = bandwidth × latency`. If `Q < Q*`:
- if you can raise occupancy, do (it is free and it does not touch the code);
- if you cannot — fixed problem size, register pressure, a co-resident kernel —
  raise per-thread concurrency instead: hoist loads above their consumers, add
  accumulators, widen to `float4`. Both multiply into the same `Q`.

**Step 3 — flat against occupancy: where are you against a bound?** Two
divisions.
- **Achieved bandwidth / 432 GB/s.** Above ~85%: you are **bandwidth-bound**.
  No amount of concurrency or ILP will help and you must move fewer bytes —
  fuse kernels (M11), tile for reuse (M17/M18), change the layout (M5), or
  change the algorithm. Above 100% means you are measuring L2, not DRAM.
- **Achieved FLOP/s / ~20,000 GFLOP/s.** Above ~80%: you are **compute-bound**
  and the only remaining lever is fewer or cheaper instructions.

**Step 4 — flat against occupancy and far from both bounds.** This is the
interesting case and it has two sub-cases, which the *code* distinguishes, not
the timings.
- **The inner loop has a loop-carried dependence** → `execution dependency`.
  Count the independent instructions per dependent one. If that count is below
  `latency × throughput = 4`, add accumulators. If it is already above 4, this
  is not your problem (Exercise 2's measured null result: a version with three
  independent instructions per dependent FFMA gained **1.07×** from four
  accumulators, against 3.25× for the version without them).
- **The inner loop has no dependence worth speaking of** → you are
  **issue-limited** (`not selected`). Count instructions in the SASS. The
  speedup available is the ratio you can cut that count by, and nothing else.
  Exercise 2's kernel D: 440 → 184 instructions, 2.03× measured.

**Step 5 — none of the above.** Barriers (reduce their count, or the block
size), load imbalance and tail effects (Module 1's waves), instruction-cache
pressure from over-unrolling, or an algorithm whose critical path is the
problem. The last one is the only case where the answer is "write a different
kernel", and recognising it early is worth more than any of the above.

**What a profiler adds.** Steps 1, 3 and 4 are exactly what Nsight Compute's
`SchedulerStats` and `WarpStateStatistics` sections report directly, in one run
instead of seven. Module 23 covers it. The reason to be able to do it without
one is that the reasoning is identical either way, and a stall histogram you
cannot interpret is worse than no histogram at all.

---

## Code Walkthrough

### `example01.cu` — latency, throughput, and their product

The kernel is deliberately the smallest thing that can express ILP:

```cpp
template<int C>
__global__ void chainK(float *out, long long *cyc, int iters)
{
    float a[C];
    #pragma unroll
    for (int i = 0; i < C; ++i) a[i] = (float)(threadIdx.x + i + 1) * 1.0e-3f;

    long long t0 = clock64();
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int u = 0; u < BODY / C; ++u)
            #pragma unroll
            for (int i = 0; i < C; ++i) a[i] = fmaf(a[i], FMA_B, FMA_C);
    }
    long long t1 = clock64();
    ...
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
```

Four design decisions, each of which the measurement would be wrong without:

- **`a[i] = fmaf(a[i], ...)` is a true recurrence.** fp32 FMA is neither
  associative nor reassociable, so no compiler is allowed to shorten a chain.
- **`BODY / C` repetitions of a `C`-wide step** keeps the FFMA count per outer
  iteration at 256 for every `C`. See §7, way 1.
- **Both loops over `i` are `#pragma unroll`'d with `C` a template parameter,**
  so `a` is never indexed by a runtime value and stays in registers.
- **The result is stored.** `if (s == 1e30f) out[0] = s;` would also defeat
  dead-code elimination, but an unconditional store also lets the validation
  pass check the arithmetic against a host replay — which it does, to exactly
  zero error, because `fmaf` is the same operation on both sides.

The occupancy dial is a 128-thread block, which puts **exactly one warp on each
of the SM's four schedulers**, so blocks-per-SM and warps-per-scheduler are the
same number, and a grid of `B × 40` blocks is one wave.

The `clk GHz` column is a guard rail. It divides the `clock64()` delta by the
wall-clock time and should land near the real SM clock. It reads 1.66–2.08 GHz
here, which is why these cycle counts are trustworthy. **At full occupancy the
same cross-check reads ~0.33 GHz** — Module 16 recorded this failure mode and
this module reproduced it (a 12-blocks/SM configuration reported 5.7
instructions per cycle per scheduler, which is 5.7× a hardware maximum). Spec
§12 rule 13: a number past a physical bound means the instrument is broken.
`clock64()` is reliable for a lone warp and unreliable for a full SM.

### `example02.cu` — Little's Law at the memory system

Section A does the arithmetic of §1 at run time against the ceiling it measured
in the same program. Section B builds the MLP × occupancy surface with

```cpp
for (unsigned i = base; i + (C-1)*stride < n4; i += C*stride) {
    float4 v[C];
    #pragma unroll
    for (int c = 0; c < C; ++c) v[c] = x[i + c*stride];     // issue all C
    #pragma unroll
    for (int c = 0; c < C; ++c) {                            // then consume
        acc.x += v[c].x; acc.y += v[c].y; acc.z += v[c].z; acc.w += v[c].w;
    }
}
```

The two loops are the whole technique: **issue, then consume.** Fuse them and
`C` collapses to 1 and you have Module 11's `#pragma unroll 1` row. The buffer
size is chosen so that `n4` is divisible by `C × stride` for every `(C, blocks)`
pair in the sweep, so every configuration does exactly the same work with no
tail — otherwise the tail is a different fraction of each cell and the table is
not a table.

Section C is the single-warp MLP cost curve of §6. Its working set is 32 MB:
far larger than the 128 KB L1, comfortably inside the 48 MB L2, so every load
is an L2 hit and DRAM bandwidth cannot enter the measurement. That is the point
of choosing L2 — the quantity being measured is a *latency*, and a latency
probe that collides with a bandwidth ceiling measures the ceiling.

---

## Check Your Understanding

1. A kernel runs at 45% of the FP32 ceiling at 100% occupancy. Its inner loop
   is a chain of dependent `FFMA`s. Your colleague proposes raising occupancy
   by reducing the block size. Using the numbers in §1 and §2, state what will
   happen and why, and state the *one* measurement you would make first to
   confirm your reasoning without a profiler.

2. Two kernels are both reported (hypothetically, by a working `ncu`) as
   spending 70% of their warp-cycles stalled. Kernel P's dominant reason is
   `long scoreboard`; kernel Q's is `not selected`. One of them will get faster
   if you double its occupancy, one will get *slower*, and one of them is
   already at a hardware limit. Assign the outcomes and explain the mechanism
   in each case.

3. Module 18 measured a GEMM at 33% occupancy running 18.7× faster than the
   same source at 100% occupancy. Module 1 measured a pointer chase whose
   throughput rose 4.87× going from 1 to 24 resident warps. Both are correct.
   Using one equation, explain why these two results are not in conflict, and
   predict which of the two kernels would be hurt more by a hypothetical
   architecture change that doubled DRAM latency while leaving bandwidth alone.

4. You have a kernel with exactly one `float` load per thread in flight and
   nothing else, running at 5120 threads, and you measure 68 GB/s. Without
   running anything further, compute the bandwidth you would predict at 20,480
   threads with the same kernel body, and at 5120 threads after converting the
   load to `float4` and hoisting four of them. State the assumption that makes
   both predictions valid and the condition under which it fails.

---

## Exercises

### Exercise 1 — `exercise01.cu` : measure the exchange rate

**Type:** fill in the code + performance reasoning + prediction (spec §6 types
1, 6, 2).

**What the program must accomplish.** Build a kernel parameterised by the
number of independent FFMA chains per thread, sweep it against occupancy to
produce the 5×5 surface of §2, and check a model of the exchange rate against
the measurement. Part A runs the kernel on a single warp and uses the shape of
the `cyc/FFMA` curve to decide, automatically, whether your chains survived
compilation; if they did not, the rest of the experiment is meaningless and the
program says so.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
nvcc -arch=sm_89 -O3 -cubin -o exercise01.cubin exercise01.cu
cuobjdump -sass exercise01.cubin
```

**TODOs.** Five.

1. The chains themselves, plus their seeding. Four requirements are stated in
   the file and three are checked. The hard one is that the number of FFMAs
   between two loop branches must not depend on `C`.
2. The launch geometry for `W` resident warps per scheduler, in exactly one
   wave, with the four schedulers equally loaded. The file warns you that not
   every `(threads, blocks)` pair with the right product is equivalent; §8
   above is why.
3. P1 — the dependent-FFMA latency of this GPU in whole cycles, scored to ±1.
4. P2, P3 — what ILP 1 → 8 is worth at the lowest and the highest occupancy in
   the sweep.
5. P4 — a *model*: a function returning the minimum warps per scheduler needed
   to saturate at a given ILP, derived from your latency answer and the issue
   rate, scored against the measured 85%-of-peak crossing on at least 4 of the
   5 ILP levels.

**Validation.** Every configuration's output is compared against a host replay
of the same `fmaf` chains (exact), every output element is checked to have been
written, and the score is out of 10: 2 for chains that survive, 2 for the
latency prediction, 1 + 1 for P2/P3, 2 for the model, 2 for numerics.
`OVERALL: PASS` requires all ten.

### Exercise 2 — `exercise02.cu` : diagnose, then fix

**Type:** debugging / optimization / design (spec §6 types 3, 4, 5).

**What the program must accomplish.** Four kernels are all far from their
respective ceilings, for four different reasons. You classify each, then write
a faster version of three of them; the fourth cannot be made faster by anything
in this module and the harness checks that your "fix" did not accidentally
change it. Each kernel has a launch configuration you may not change — one of
them has a thread count fixed by the problem itself, not by a budget.

The program prints its evidence before it scores you: every kernel's time
against occupancy (and, for the kernel whose thread count is fixed, against
block *shape* at a constant thread count), and where each sits relative to a
hardware bound. Two of the four kernels respond to occupancy and two do not;
telling the members of each pair apart requires reading the code, not the
table.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

**TODOs.** Five: the four-way classification (design), three replacement
kernels, and four predicted speedups. One of the four predictions is not a
guess — it follows from §1's two numbers.

**Validation.** Each replacement is validated independently: exact for the
elementwise kernels, `err/(γ_K·S) ≤ 1` for the one you had to reassociate, and
a relative tolerance for the one where you are expected to have traded
accuracy. Gates on all four speedups, plus the classification, plus the
predictions. 10 points.

### Exercise 3 — `exercise03.cu` : Little's Law, quantitatively

**Type:** performance reasoning + fill in the code (spec §6 types 6, 1).

**What the program must accomplish.** You are given a streaming reduction at a
launch you cannot change, running at a small fraction of peak. You write three
one-line formulas — supplied concurrency, effective latency, required
concurrency — which the harness evaluates against its own copy and against the
measurement, and then you close the gap. The satisfying part is not the score:
it is that your formula, fed the measured baseline bandwidth, reproduces
Module 4's DRAM latency to within 1%, and then predicts the baseline bandwidth
back to three significant figures.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

**TODOs.** Five: the three formulas, the replacement kernel (≥95% of the
measured streaming ceiling at 5120 threads — there is more than one multiplier
available and they compose), and a prediction of the bandwidth it will reach.

**Validation.** The three formulas are checked exactly against the harness's
own evaluation; the kernel is validated by reducing all 5120 partial sums and
comparing against a double reference with a `γ_K·S` tolerance, plus an
unwritten-output check; the gate is 95% of the measured ceiling and the
prediction is scored to 12%.

---

## Prediction

Commit to these in writing before you run anything.

1. **The number.** Before reading §1's table, predict the dependent-`FFMA`
   latency of this GPU in cycles and the number of independent FFMAs one warp
   must hold to saturate its scheduler's issue port. Then predict what fraction
   of the FP32 ceiling a single warp per scheduler running a single dependent
   chain achieves. One number determines the other two.

2. **The shape.** In `example01.cu`'s 5×5 table, predict the value of the
   cell at (2 warps/scheduler, ILP 2) as a fraction of the best cell, and the
   cell at (1 warp/scheduler, ILP 4). Then predict whether (4, 1) is above or
   below (1, 4), and say which effect would explain a difference.

3. **The memory side.** Exercise 3's baseline has one 4-byte load per thread in
   flight at 5120 threads and measures 68.3 GB/s. Predict, before running
   anything, the bandwidth of the same kernel body at 61,440 threads (full
   occupancy), and name the quantity that must not have changed for your
   prediction to hold.
