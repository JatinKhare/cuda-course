# Module 23 — Nsight Compute

> Prerequisites: Modules 1–21, and Module 22 for the division of labour.
> In particular M5 (sectors), M7 and M18 (bank conflicts), M8 (lane
> efficiency), M12 (the streaming ceiling), M16 (the stopwatch bound on hit
> rate), M19 (occupancy, both denominators), M20 (the stall taxonomy),
> M21 (the hierarchical roofline).
>
> What this module gives you: the ability to read an `ncu` report line by line
> and know, for every number in it, what physical quantity it is counting —
> because you have already measured that quantity yourself, by hand, in an
> earlier module.

---

## Concept

### 1. Why this module is different

Every module from 1 to 21 had to establish by controlled measurement what
`ncu` reports in one command. Module 5 enumerated 32 addresses on paper to get
a sector count. Module 7 bucketed words into banks to get a conflict degree.
Module 19 built a `clock64()` + `%smid` instrument to get achieved occupancy.
Module 21 wrote six probes to get the ceilings of a roofline.

That was not a detour. **The course now contains independently derived ground
truth for nearly every metric that matters**, and this module spends it: each
metric is introduced by naming the earlier measurement it equals.

This is not the usual order. The usual order is "here is a counter, here is
roughly what it means, now go and trust it." The order here is the inverse —
you already know the quantity, and all you are learning is its name and its
denominator. That turns out to be most of what reading a profile actually is.

There is a practical reason for the inversion too. **`ncu` does not run on this
machine.** It is installed (2026.1.0.0, build 37166530) and every invocation
returns:

```
==PROF== Connected to process 11068 (...\t.exe)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA
          GPU Performance Counters on the target device 0. For instructions on
          enabling permissions and to get more information see
          https://developer.nvidia.com/ERR_NVGPUCTRPERM
==PROF== Disconnected from process 11068
```

That is verbatim output, captured on this GPU. §8 explains what the error
means and how an administrator clears it. Everything else in this module is
written so that you can read a real report the first time you meet one.

### 2. Division of labour: `nsys` answers "when", `ncu` answers "why"

Module 22 taught Nsight Systems. The two tools do not overlap, and the line
between them is sharp.

| | Nsight Systems (`nsys`) | Nsight Compute (`ncu`) |
|---|---|---|
| unit of observation | the **timeline** — the whole process | **one kernel launch** |
| mechanism | **tracing**: timestamps CUDA API calls, kernel begin/end, memcpys, NVTX ranges | **counter collection**: reads hardware performance counters inside the SMs, caches and memory controllers |
| sees launch gaps, H2D/D2H overlap, stream concurrency, CPU-side stalls | **yes** | **no** — it sees one kernel and nothing between kernels |
| sees sectors, bank conflicts, stalls, occupancy, instruction mix | **no** | **yes** |
| perturbation | small; the program still runs at roughly normal speed | **large**; see §3 |
| permissions needed | **none** | **elevated, once, by an administrator** |

The permission asymmetry has a concrete cause and it is worth one paragraph.
Tracing only needs timestamps the driver already produces for its own
bookkeeping, and those timestamps describe *your* process. The performance
counters are different: they are **global to the GPU**, they are not
partitioned per process, and they count work done by *every* context resident
on the device. A low-privilege process that could read them could infer the
memory access pattern — and therefore, in published side-channel attacks, key
material — of a *different* user's kernel running on the same GPU. NVIDIA
therefore gates counter access behind an explicit administrative opt-in. That
is why `nsys` works on this machine and `ncu` does not.

**Corollary for how you work:** the first question about a slow program is
almost never a kernel question. Profile with `nsys` first. If the GPU is busy
40% of the wall time, no amount of `ncu` will help you — that is a Module 22
problem. Only once the timeline says "this one kernel is 80% of the GPU time"
does `ncu` become the right tool.

### 3. How `ncu` actually works, and why a profiled run is so slow

An SM does not have a counter for every metric. It has a limited number of
*counter collection slots*, and a given set of counters can only be programmed
into them in certain combinations. `ncu` resolves this by **replay**.

The default replay mode is `--replay-mode kernel`:

1. Before the kernel runs, `ncu` **saves** every piece of device memory the
   kernel could write (it determines this from the launch's parameters and the
   allocations it knows about).
2. It programs one *pass* worth of counters, runs the kernel, reads the
   counters.
3. It **restores** the saved memory so the kernel's inputs are exactly as they
   were, programs the next pass, and **runs the same kernel again**.
4. Repeat until every requested counter has been collected.

Consequences you must internalise:

- **A profiled kernel runs many times.** `--set full` requests ~7300 metrics on
  this chip (see §5) and can need dozens of passes. Combined with the
  save/restore of device memory, a program that takes 2 seconds can take
  several minutes under `ncu`.
- **That slowdown is not a bug and it is not telling you anything about your
  kernel.** Do not read wall-clock time from a profiled run. Read
  `gpu__time_duration.sum`, which is measured on the device.
- **Replay perturbs caches.** Pass 2 of a kernel runs with pass 1's data
  already resident in L2. For a kernel whose working set fits in the 48 MB L2,
  the hit rates `ncu` reports can be *better* than the hit rates the kernel
  gets in production. This is the single most important measurement caveat in
  the tool.
- **Replay is not safe for every kernel.** A kernel that reads device memory it
  did not write — a persistent kernel reading a flag another kernel set, or any
  kernel whose correctness depends on state `ncu` did not know to save — can
  behave differently on the replay passes. `--replay-mode application` reruns
  the whole *application* per pass instead, which is slower but restores the
  state properly; `--replay-mode range` profiles a whole marked range as one
  unit.

This course has already measured the shape of this cost twice without `ncu`.
Module 18 and Module 21 both found their throughput gates failing under
`compute-sanitizer` at roughly **60×** slowdown, which is the same *kind* of
instrumentation tax: the tool is not making your kernel slow, it is running a
different experiment. **Never gate a performance assertion on a run under a
tool.** Measure, then profile.

### 4. The metric naming grammar

`ncu` metric names look forbidding and are completely systematic:

```
   unit __ quantity [_qualifiers] . rollup . submetric
   ────   ──────────────────────   ──────   ─────────
```

**Unit** — *where* in the chip the counter lives:

| unit | hardware |
|---|---|
| `dram__` | the GDDR6 memory controllers — what crosses the pins |
| `lts__` | L2 slices (`t` = tag stage; "sectors" here are L2 sectors) |
| `l1tex__` | the per-SM unified L1/texture/shared unit |
| `smsp__` | **SM sub-partition** = one of the four processing blocks, each with its own warp scheduler and 16384-register slice (Module 1, Module 19) |
| `sm__` | the whole SM (an aggregate over its four `smsp`s) |
| `gpc__` | graphics processing cluster — used here mainly for `gpc__cycles_elapsed` |
| `gpu__` | whole-device composite metrics |
| `launch__` | **not a counter.** Launch configuration and occupancy-calculator output, known statically |

**Rollup** — how the per-unit values are combined: `.sum`, `.avg`, `.max`,
`.min`. `sm__cycles_active.avg` is the average over 40 SMs;
`l1tex__t_sectors...sum` is the total over all 40 L1s.

**Submetric** — what you want *expressed as*:

| submetric | meaning |
|---|---|
| (none) | the raw count |
| `.per_second` | count / wall time |
| `.per_cycle_active` | count / cycles the unit was **active** |
| `.per_cycle_elapsed` | count / cycles of the **whole kernel** |
| `.peak_sustained` | the hardware maximum rate of that counter |
| `.pct_of_peak_sustained_active` | achieved / peak, over **active** cycles |
| `.pct_of_peak_sustained_elapsed` | achieved / peak, over **elapsed** cycles |
| `.ratio` | a quotient of two counters, already formed |

> **The single most important thing in this grammar is `active` vs `elapsed`.**
> `active` divides by the cycles *that unit* was busy. `elapsed` divides by the
> cycles the *kernel* took. They differ by exactly the idleness of that unit,
> and Module 19 measured a case where choosing the wrong one inverts the sign
> of your conclusion (§6b).

### 5. `--set` vs `--section` vs `--metrics`

Three nested levels of "how much do you want":

```
ncu --list-sets          # which bundles exist
ncu --list-sections      # which sections exist
ncu --query-metrics      # every metric name the chip supports
```

Real output of `ncu --list-sets` on this machine (host-side listing works even
though collection does not):

```
Identifier Sections                                                    Enabled Estimated Metrics
---------- ----------------------------------------------------------- ------- -----------------
basic      LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution   yes     213
detailed   ComputeWorkloadAnalysis, LaunchStats, MemoryWorkloadAnalysis,
           MemoryWorkloadAnalysis_Chart, Occupancy, SourceCounters,
           SpeedOfLight, SpeedOfLight_RooflineChart, Tile,
           WorkloadDistribution                                         no      1007
full       (22 sections)                                               no      7314
roofline   SpeedOfLight, 4 hierarchical roofline charts,
           SpeedOfLight_RooflineChart, WorkloadDistribution             no      5932
```

Note the **Estimated Metrics** column: 213 for `basic`, **7314** for `full`.
That number is a direct predictor of how many replay passes you will pay for.

| level | granularity | when |
|---|---|---|
| `--set <id>` | a bundle of sections | `basic` for triage, `full` only when you have narrowed to one kernel |
| `--section <Id>` | one report section | the normal working mode once you know what you are looking for |
| `--metrics a,b,c` | individual counters | scripting, CSV extraction, or a metric no section displays |

Section identifiers you will actually type (all verified present in 2026.1.0):
`SpeedOfLight`, `MemoryWorkloadAnalysis`, `MemoryWorkloadAnalysis_Tables`,
`Occupancy`, `SchedulerStats`, `WarpStateStats`, `ComputeWorkloadAnalysis`,
`InstructionStats`, `LaunchStats`, `SourceCounters`,
`SpeedOfLight_HierarchicalSingleRooflineChart`.

### 6. The ten metrics that matter, and where you already measured them

This is the core of the module. For each metric: the name, what it counts, and
**the module in which you derived the same quantity by hand**.

#### 6a. Achieved DRAM throughput — "is the bus busy?"

```
dram__bytes.sum                                       bytes across the pins
dram__bytes.sum.per_second                            bytes/s
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed    % of peak
```

**Your anchor (M12, M21):** a properly warmed streaming read on this GPU
measures **410.5–410.7 GB/s = 95.0% of the 432.0 GB/s pin peak**. That is the
number `dram__bytes.sum.per_second` prints for a saturating kernel, and
`...pct_of_peak_sustained_elapsed` prints ~95%.

Three things you know that a first-time reader does not:

1. The denominator is the **pin peak**, 432 GB/s, computed from the bus width
   and memory clock — not from anything the kernel did. 95% is the practical
   roof; this course never measured higher.
2. **A high value is not a verdict.** Module 12's §12 rule 7 and Module 5's
   stride experiments: a stride-8 read moves exactly as many bytes as a
   contiguous read and uses one eighth of them. `dram__throughput` is 95% for
   both. The metric that separates them is §6c.
3. **Above 100% means L2.** Module 5 measured an apparent 1360 GB/s (315% of
   peak) on a 32 MB array; M15 measured 1671 GB/s (387%). `dram__bytes` itself
   cannot exceed the pins — but any *derived* bandwidth you compute from your
   own byte model can, and when it does, you are measuring cache residency.

**Caveat `ncu` does not print:** this part is a laptop GPU whose memory P-state
ramps 6001 → 8001 → 9001 MHz. The 400 ms-warm-up figure is 372 GB/s and the
1500 ms figure is 410.5. `ncu` reports what happened in the launch it profiled;
the clock state during that launch is yours to control.

#### 6b. Achieved occupancy — and its two denominators

```
sm__warps_active.avg.pct_of_peak_sustained_active     <- "Achieved Occupancy"
sm__warps_active.avg.pct_of_peak_sustained_elapsed    <- NOT in any section
sm__maximum_warps_per_active_cycle_pct                <- "Theoretical Occupancy"
launch__occupancy_limit_registers / _shared_mem / _warps / _blocks
```

**Your anchor (M19):** theoretical occupancy is
`min` of four limiters, and the register limiter is the **four-slice** form

```
blocksByRegs  = 4*floor(16384 / (roundUp(R,8)*32)) / warpsPerBlock
blocksBySmem  = 102400 / roundUp(smem + 1024, 128)
blocksByWarps = 48 / ceil(threads/32)
blocksByBlock = 24
```

verified exactly on **137 of 137 kernels**. `ncu` prints those four limiters as
`launch__occupancy_limit_*` and you can check them against the formula on the
spot. (The aggregate `65536`-pool model that most tutorials use is wrong on
10 of those 137; it cannot be, because a warp's registers must come from a
single 16384-register slice and stranded registers in one slice cannot be
pooled with another.)

Now the part that matters more. **`ncu`'s "Achieved Occupancy" is
`..._pct_of_peak_sustained_active`, and Module 19 measured that it is
structurally blind to the two things you most want to detect:**

| experiment (M19) | `..._active` | `..._elapsed` | wall time |
|---|---|---|---|
| 240 → 241 blocks (a one-block **tail**) | 61.9 → **62.3** (+0.4) | 61.9 → **57.9** (−4.0) | **+7.5%** |
| 1..8× cost **imbalance**, one wave | 61.9 → **65.3** (+3.4, *up*) | 61.9 → **47.5** (−14.4) | **+53%** |

The mechanism is the denominator. `_active` divides each SM's warp-cycles by
*that SM's own busy span*. An SM that finished early and went idle stops
contributing to its own denominator as well as its numerator, so idleness is
invisible — and an imbalanced SM that stays busy longer looks *better*. `_elapsed`
divides by the whole kernel's span, so idleness costs you.

And the third fact, also M19's: **a barrier or a DRAM stall does not lower
achieved occupancy at all.** A stalled warp is still resident. That is how
100% occupancy and 6% of peak coexist — and it is why the next metric exists.

> `..._pct_of_peak_sustained_elapsed` is **not in any section**. You must ask
> for it with `--metrics`. The default report shows you the blind one.

#### 6c. Sectors per request — this *is* coalescing efficiency

```
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum      32 B sectors moved
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum     warp-level LDG instructions
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio
```

**Your anchor (M5):** take the 32 byte addresses one warp presents, shift each
right by 5, count the distinct values. That is the sector count, and it is
exactly `sectors.sum / requests.sum` for one instruction.

| source expression | sectors/request | bytes/sector | used/moved |
|---|---|---|---|
| `in[i]`, aligned | **4** | 32.00 | 100% |
| `in[i+1]`, 4 B misaligned | **5** | 25.6 | 80% |
| `in[2*i]` | **8** | 16.00 | 50% |
| `in[8*i]` and any wider stride | **32** (saturated) | 4.00 | 12.5% |
| `in[0]` broadcast | **1** | 128.00 | — |
| `float4 in4[i]` | **16** | 32.00 | 100% |

The `.ratio` metric above is the same information in bytes: **32.00 is
perfect** (every byte of every sector is used), 4.00 means seven eighths of
every sector the memory system moved is discarded.

Two traps the counter sets:

- **The broadcast row.** One sector, 128 bytes/sector, and a naive "memory
  efficiency" reading calls it 12.5% efficient. Nothing is wrong with it.
- **The misaligned row.** M5 predicted 80% for a 4-byte misalignment and
  measured **97.9%** — because the neighbouring warp's boundary sector was
  already in cache. The *instruction-level* counter says 5 sectors; the
  *traffic* counters (`lts__`, `dram__`) say almost nothing extra moved. Both
  are right. They are counting at different levels.

#### 6d. Cache hit rates

```
l1tex__t_sector_hit_rate.pct                        L1/TEX hit rate
lts__t_sector_hit_rate.pct                          L2 hit rate
lts__t_sector_op_read_hit_rate.pct / ..._write_...
lts__t_sectors_lookup_miss.sum
```

**Your anchor (M16):** this is the one metric the course could *not* construct
directly, because a hit and a miss differ in latency, not in anything a kernel
can count about itself. What M16 built instead is a **bound**:

```
on-chip service fraction  >=  1 - (pin peak x elapsed) / bytes requested
```

For the naive GEMM, M16 measured **>= 91.6–93.4%** of requested bytes serviced
on chip, and separately that the kernel ran at **12.5–14.1× its own
requested-traffic roofline**. Both are statements about L1+L2 hit rate derived
from a stopwatch and the 432 GB/s pin rate alone. When you finally read
`lts__t_sector_hit_rate.pct` on a working machine, it should land above that
bound — and if it does not, your byte model is wrong.

The qualitative half you can always see: **any apparent bandwidth above
432 GB/s is a hit-rate measurement wearing a bandwidth's clothes.**

**Replay caveat applies hardest here.** See §3: the hit rates `ncu` reports for
an L2-resident working set are measured on a kernel that has already been run
once or more.

#### 6e. L2 traffic

```
lts__t_sectors.sum  /  lts__t_sectors_op_read.sum  /  ..._op_write.sum
lts__throughput.avg.pct_of_peak_sustained_elapsed
```

**Your anchor (M21):** the L2 read ceiling on a 24 MB resident working set
measures **1259–1938 GB/s** (the cross-module index records ~1305 as the
representative value), against 410 GB/s at the pins. That is the second rung of
the hierarchical roofline. `lts__t_sectors_op_read.sum × 32 B / duration` is
the same number, read directly.

Pairing `lts__` with `dram__` is how you answer "did the L2 do its job": L2
sectors in, DRAM bytes out, ratio = the miss rate you are paying.

#### 6f. Shared-memory bank conflicts

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
smsp__sass_inst_executed_op_shared_ld.sum
```

**Your anchor (M7, corrected by M18):** `bank = (addr/4) % 32`. The conflict
degree `D` is the maximum, over banks, of the number of **distinct words** that
bank must supply to one warp. For `s[k*tid]`, `D = gcd(k, 32)`.

**Vocabulary (the thing M7 explicitly deferred to this module).** A
**wavefront** is one pass of the shared-memory crossbar — the hardware splits a
warp's shared access into as many wavefronts as it needs, and
`l1tex__data_pipe_lsu_wavefronts_...` counts them. The *conflicts* counter
counts the **extra** wavefronts, i.e. `wavefronts − requests`. So a
conflict-free 4-byte full-warp load reports 1 wavefront and 0 conflicts; a
32-way conflict reports 32 wavefronts and 31 conflicts.

**The trap, and it is a big one.** `l1tex__data_bank_conflicts_... = 0` is not
a goal and a non-zero value is not a finding:

- **On a 4-byte access the Ada cost law is `max(2, D)` cycles, not `D`.** M7
  measured `s[2*tid]` — which reports one conflict per request — at **1.00×**
  the cost of `s[tid]`. A degree-2 conflict on a 4-byte access is *free*.
  There is a two-cycle floor for any 32-lane 4 B shared access and a 2-way
  conflict hides underneath it.
- **On an `LDS.128` the floor is gone.** M18 isolated this with byte-identical
  SASS: a 16-byte access is phase-split into four phases of 8 lanes, and one
  phase already asks the crossbar for its full 128 B, so there is no spare
  cycle. Cost is **∝ D with no floor**. Measured 1.61× against a predicted
  1.67× for D=2 vs D=1.
- **Therefore: pad only after measuring a conflict of degree ≥ 4** — and check
  the SASS first. M17 measured padding a row-major GEMM tile at **0.674–0.706×,
  a 40% loss**, because the `T+1` pitch broke the 16 B alignment `ptxas` needed
  to merge four contiguous reads into `LDS.128`: 20 shared instructions became
  32. M18 measured that a pad of 1 leaves degree 4 — **a partial fix that looks
  exactly like a fix** — and only a pad of 4 floats both kills the conflict and
  preserves float4 alignment.

So the full reading of this counter is: conflicts > 0, **divide by requests to
get D**, check the access width, apply `max(2,D)` or `∝D` accordingly, and only
then decide whether there is anything to win.

#### 6g. Warp stall reasons

```
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
smsp__average_warps_issue_stalled_selected_per_issue_active.ratio
smsp__average_warp_latency_per_inst_issued.ratio   <- "Warp Cycles Per Issued Instruction"
```

(Also `drain`, `imc_miss`, `membar`, `sleeping`, `branch_resolving`,
`dispatch_stall`, `tex_throttle`, `misc`, `gmma`.)

The unit is **average warps stalled for this reason, per issued instruction** —
i.e. a cycles-per-instruction *breakdown*. The reasons sum to
`smsp__average_warp_latency_per_inst_issued.ratio`, which is the cycles of warp
latency you are paying for each instruction you issue.

**Your anchor (M20).** Module 20 measured the two constants that make this
table readable:

- dependent FFMA **latency = 4.051 cycles**;
- saturated issue interval = **1.067 cycles**;
- so `L/T = 3.8`, and **four independent FFMAs in flight per scheduler
  saturates the pipe**. Equivalently: one unit of ILP buys exactly one resident
  warp per scheduler. M20 measured the exchange exactly — the smallest
  warps/scheduler reaching 90% of peak is 4, 2, 1, 1, 1 for ILP 1, 2, 4, 8, 16;
  the product is constant at 4.

Read the stall table with the question **"does adding warps fix this?"**:

| reason | the warp is waiting for | more occupancy? |
|---|---|---|
| `long_scoreboard` | a global/local load (L2 or DRAM) | **yes**, or more MLP |
| `short_scoreboard` | a shared-memory load, or `MUFU` | yes, or more ILP |
| `wait` | a fixed-latency dependency (an FFMA, 4.05 cycles) | yes, **or ILP** |
| `barrier` | the slowest warp of its own block | **no** — rebalance or shrink blocks |
| `membar` | a fence to drain | no |
| `imc_miss` | the immediate-constant cache | no |
| `no_instruction` | the instruction cache | **no** — the loop body is too big |
| `mio_throttle` | the shared/LSU queue is full | **no** — fewer shared instructions |
| `math_pipe_throttle` | the math pipe is full of *other warps'* work | **NO — it is already saturated** |
| `lg_throttle` | the local/global queue is full | **NO** |
| `not_selected` | another warp was chosen this cycle | **NO — you have too MUCH** |

The last three rows invert the folk advice. M20's Exercise 2 kernel D — 64 IEEE
divides at 12 blocks/SM — is the `not_selected` case made concrete: the fix was
`__fdividef` (440 → 184 SASS instructions, 15 → 0 `FCHK`), worth **1.85–2.03×**,
and *adding warps would have made it worse*.

#### 6h. Eligible warps per scheduler — the metric that explains §6b

```
smsp__warps_active.avg.per_cycle_active       resident
smsp__warps_eligible.avg.per_cycle_active     eligible to issue
smsp__issue_active.avg.per_cycle_active       actually issued
smsp__issue_active.avg.pct_of_peak_sustained_active     "One or More Eligible"
smsp__issue_inst0.avg.pct_of_peak_sustained_active      "No Eligible"
```

Three numbers, one chain: **resident ≥ eligible ≥ issued.** On this GPU the
ceiling for *resident* is 12 warps per scheduler (48 per SM). *Issued* can
never exceed 1.0 per cycle per scheduler.

This is the pair of counters that resolves Module 19's paradox. Achieved
occupancy counts **resident**. A warp stalled on a 575-cycle DRAM load is
resident and not eligible. So:

> **`sm__warps_active...` near 100% together with
> `smsp__warps_eligible.avg.per_cycle_active` near 0 is the signature of a
> latency-bound kernel**, and the fix is never "more occupancy" — you already
> have all of it. The fix is more work in flight per warp (M20's MLP/ILP axis).

Conversely `smsp__issue_active.avg.per_cycle_active` near 1.0 means the
scheduler issues every cycle it can, and you are at the instruction-issue
ceiling that M21 measured at **286–330 G warp-instructions/s** across the
device.

#### 6i. Lane efficiency and instruction mix

```
smsp__thread_inst_executed_per_inst_executed.ratio       "Avg. Active Threads Per Warp"
smsp__thread_inst_executed_pred_on_per_inst_executed.ratio
                                       "Avg. Not Predicated Off Threads Per Warp"
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_alu / _lsu / _xu / _adu / _cbu / _uniform / _fp64
sm__sass_thread_inst_executed_op_ffma_pred_on.sum
smsp__inst_executed.sum
```

**Your anchor (M8):** lane efficiency is `thread_inst_executed / (32 ×
inst_executed)`. Module 8 computed it by hand for two kernels: **47.7%** and
**56%**. Module 8 also measured the costs it predicts — a `tid & 1` split is
**1.96×**, a warp-uniform `tid < 128` split is **1.02×** — and established that
divergence cost is *additive over arms*, not the maximum.

**The pair of metrics is more informative than either alone.** Module 8 proved
`__activemask()` cannot distinguish a predicated-off lane from a
not-taken lane. The counters can: a predicated-off lane **is** active but **is
not** predicated-on, so it is counted by the first metric and not by the
second. **The gap between the two metrics is exactly the predication; a real
branch lowers both.** That is this module paying M8's debt.

**The trap:** the ratio is an average over *issued instructions*, not over
time. A kernel where 1 lane in 32 takes a cheap rare path reports **1.00**
(3.1% lane efficiency) for that branch and may be entirely healthy. Never read
lane efficiency without the instruction count it is averaged over.

**Your anchor for instruction mix (M18, M21):** M18 measured FFMA density in
the SASS inner loop — **25%** for the naive GEMM (`LDG, LDG, IMAD.WIDE, FFMA`),
**90.8%** for the 8×4 register-tiled kernel (256 `FFMA` + 24 `LDS.128` +
2 `BAR.SYNC` in a 282–286 instruction body) — and reported that *FFMA density
was the only number that tracked performance throughout the ladder*.
`sm__inst_executed_pipe_fma...pct_of_peak_sustained_active` is that number, read
off the hardware. M21 then showed why it is the right number: the FP32
"compute plateau" **is** the issue ceiling, and a kernel's plateau is
`density × 64 FLOP/instruction × 310 G instructions/s`.

#### 6j. The roofline

```
ncu --set roofline ./prog.exe
   SpeedOfLight_RooflineChart                      (the classical two-axis one)
   SpeedOfLight_HierarchicalSingleRooflineChart    (FP32, multi-level)
```

**Your anchor (M21), handed to this module explicitly.** M21's closing section
names `dram__bytes.sum`, `l1tex__t_bytes.sum`, `smsp__inst_executed.sum` and
`sm__sass_thread_inst_executed_op_ffma_pred_on.sum` as the counters that would
let you build every column of its tables without writing a probe.

What you bring that the chart does not give you:

1. **M21's five ceilings, measured**: DRAM 410.4 GB/s, L2 1259–1938 GB/s,
   shared scalar `LDS` 4777–5452 GB/s, `LDS.128` 9019–10008 GB/s, FP32 FFMA
   17 123–19 847 GFLOP/s, plus instruction issue 286–330 G instr/s as a sixth
   axis the roofline has no place for.
2. **The knowledge that arithmetic intensity is meaningless without naming its
   level.** Tiled GEMM has `AI(DRAM) = 181`, `AI(L2) = 4.0`, `AI(shared) =
   0.25`. All three are correct. The classical two-axis roofline sees only the
   first and therefore calls naive, tiled and register-tiled GEMM all
   compute-bound while they run at 6.8 / 8.3 / 39.6% of peak. `ncu`'s
   *hierarchical* roofline sections exist because of exactly this.
3. **The fourth bucket.** M21's reconciliation rule: measured/predicted of
   0.8–1.25 means you are at the roof; > 1.25 means your byte ledger is wrong;
   0.25–0.8 means the right ceiling imperfectly reached; **< 0.25 means nothing
   is saturated and the roofline is the wrong model** — go to Little's Law. The
   M1/M4 pointer chase sits **58× below its roofline** and no roofline chart,
   automated or not, will tell you that it needs 122–129 kB in flight and
   supplies 1.00 kB.

### 7. The interpretation workflow, as a decision tree

```
                 ncu --set basic --kernel-name <k> ./prog
                                │
                     Section: GPU Speed Of Light
                                │
        ┌───────────────┬───────┴────────┬────────────────┐
        │               │                │                │
   Memory high     Compute high      BOTH mid         BOTH low
   Compute low     Memory low            │                │
        │               │                │                │
  --section Memory  --section        the binding      NOTHING is
  WorkloadAnalysis  InstructionStats  ceiling is       saturated.
        │           ComputeWorkload   ON CHIP, not     STOP reading
        │               │             at the pins      the roofline.
  sectors/request   FFMA density      (M21: shared     --section
  hit rates         pipe utilization   or L2 or issue)  WarpStateStats
  bank conflicts        │                │             + SchedulerStats
        │               │                │                │
        └───────────────┴────────┬───────┴────────────────┘
                                 │
                      --section WarpStateStats
                      --section SchedulerStats
                 (which stall reason dominates? §6g table)
                                 │
                        eligible warps per scheduler
                        near 0 with high occupancy?
                             -> latency bound, add MLP/ILP
                                 │
                   recompile with -lineinfo, then
                   --set detailed  (SourceCounters)
                   -> per-line sector counts and stalls
```

Rules for this tree:

1. **Speed of Light first, always.** It is 213 metrics, one pass, seconds.
   Everything else is conditional on what it says.
2. **"Both low" is the row everyone misdiagnoses.** A dependent pointer walk
   reports Memory Throughput near 1% and Compute near 0%. A reader who has only
   learned "higher is better" concludes "not memory bound, therefore compute
   bound". It is bound by *neither*: it is bound by latency × insufficient
   concurrency, and Little's Law is the governing equation (M20, M21).
3. **Source attribution is last, and needs `-lineinfo`.** Compile with
   `nvcc -arch=sm_89 -O3 -lineinfo`. Then `--set detailed` or
   `--section SourceCounters` gives you per-source-line sector counts, stall
   samples and instruction counts, and `ncu-ui` shows SASS beside source. Note
   that `-lineinfo` does *not* disable optimisation (unlike `-G`), so the
   attribution is approximate for heavily scheduled code — it is still the
   fastest way to find which of 40 lines is the one.

### 8. The command line, in practice

```bash
# triage: one section, one kernel, first launch only
ncu --set basic --kernel-name myKernel --launch-count 1 ./prog.exe

# skip warm-up launches; profile launches 10..12 of this kernel
ncu --kernel-name myKernel --launch-skip 10 --launch-count 3 ./prog.exe

# regex kernel matching (useful for templates)
ncu --kernel-name-base demangled --kernel-name "regex:gemm.*<16," ./prog.exe

# exactly the metrics you want, machine-readable
ncu --metrics \
  dram__bytes.sum.per_second,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_eligible.avg.per_cycle_active \
  --csv --page raw ./prog.exe > metrics.csv

# capture once, analyse many times (and share the file)
ncu --set full -o report --force-overwrite ./prog.exe     # writes report.ncu-rep
ncu --import report.ncu-rep --page details
ncu --import report.ncu-rep --csv --page raw > all.csv
ncu-ui report.ncu-rep

# compare two reports
ncu --import after.ncu-rep --baseline before.ncu-rep --page details
```

Flags worth memorising:

| flag | why |
|---|---|
| `--kernel-name` / `--kernel-name-base demangled` | without it you profile every kernel, including library ones |
| `--launch-skip N` / `--launch-count M` | **essential.** The first launch of a kernel includes module load and cold caches; profiling it tells you about startup, not steady state. This is the "launch hygiene" M7 deferred here |
| `--target-processes all` | the kernel is in a child process |
| `--replay-mode application\|range\|kernel` | §3 |
| `--cache-control none` | *stop* `ncu` flushing the caches between passes — use when you want the production cache state, not the cold one |
| `--clock-control none` | *stop* `ncu` locking clocks to base. On this laptop part, locking clocks is usually what you want; turn it off only when you are specifically studying clock behaviour |
| `-o` / `--import` / `--csv` / `--page raw` | capture once, analyse offline |

> **`--clock-control` is the one flag this course cares about most.** By default
> `ncu` pins the GPU to its base clock so that two profiles are comparable. That
> makes `ncu`'s *cycle* counts stable and its *time* numbers not comparable to
> your own stopwatch runs, which saw 0.49–2.04 GHz. Compare cycles with cycles.

### 9. What `ncu` cannot see

Four blind spots, all of which this course has measured independently:

1. **Everything between kernels.** `ncu` profiles a launch. Launch gaps,
   H2D/D2H transfers that do not overlap, CPU-side stalls, stream serialisation
   — all invisible. That is Module 22's whole subject. A kernel can be at 95%
   of the DRAM roof inside a program that is 80% idle.
2. **Replay perturbs caches** (§3). The hit rates are measured on a kernel that
   has already run. For an L2-resident working set this flatters the report.
3. **Achieved occupancy is blind to tails and imbalance** (§6b). M19 measured a
   tail moving it 0.4 points while the kernel got 7.5% slower, and an imbalance
   moving it *up* 3.4 points while the kernel got 53% slower.
4. **A counter cannot see a missing bounds check.** M18 measured a GEMM that
   drops a `kt+c<K` guard, returns the exactly correct answer, runs 1.05×
   faster, and is reported by `compute-sanitizer` as `Invalid __global__ read`.
   No profiler metric and no numerical validator detects it. Different tool,
   different question.

And one thing that is *not* a blind spot but is routinely treated as one:
`ncu` reports **per launch**. If your kernel's behaviour depends on which
launch it is — a persistent kernel, a solver whose convergence changes the work
per iteration — profile a representative launch with `--launch-skip`, not the
first one.

### 10. `ERR_NVGPUCTRPERM`, factually

The error means: your user account is not permitted to read the GPU's
performance counters. It is not a licensing problem, not a driver bug, and not
specific to your kernel — `ncu --query-metrics`, which only *enumerates* the
counters, fails with the same error on this machine.

**Why the gate exists:** see §2. The counters are global to the device and
observe every context on it, so they are a cross-process information leak if
left open.

**How an administrator clears it:**

- **Windows.** NVIDIA Control Panel → *Desktop* menu → *Enable Developer
  Settings* → **Developer → Manage GPU Performance Counters** → select
  *Allow access to the GPU performance counters to all users* → Apply, then
  reboot. (Equivalent registry key:
  `HKLM\SOFTWARE\NVIDIA Corporation\Global\NVTweak\RmProfilingAdminOnly = 0`.)
  Changing this requires administrator rights by design.
- **Linux.** Either run the profiler as root, or set the driver module
  parameter persistently:
  ```
  # /etc/modprobe.d/nvidia-profiler.conf
  options nvidia NVreg_RestrictProfilingToAdminUsers=0
  ```
  then `sudo update-initramfs -u` (or the distro equivalent) and reboot.
- Either way it is a **one-time machine-level change**, not a per-run flag.
  There is no `ncu` option that works around it, and you should not look for
  one.

The user of this machine has declined to make that change, which is why this
module is built the way it is.

---

## Hardware Mental Model

### Where the counters physically are

The chip has counter blocks at each level of the hierarchy you learned in
Module 4, and the metric prefix tells you which block:

```
   40 SMs
   ├── 4 sub-partitions each  (smsp__)   warp scheduler, 16384-reg slice,
   │                                     32 FP32 lanes, issue/stall logic
   └── 1 unified L1/TEX/SMEM  (l1tex__)  128 KB; tag stage counts sectors,
                                         data stage counts wavefronts & conflicts
            │
      crossbar
            │
   L2 slices (lts__)        48 MB total; counts sectors, hits, misses, atomics
            │
   memory controllers (dram__)   192-bit GDDR6 @ 9.001 GHz = 432.0 GB/s
```

Two consequences of this layout that explain metric names you will otherwise
find arbitrary:

- **`l1tex__t_*` vs `l1tex__data_*`.** The `t` (tag) stage resolves addresses
  into sectors — hence `t_sectors`, `t_requests`, `t_sector_hit_rate`. The
  `data` stage moves bytes through the bank crossbar — hence
  `data_pipe_lsu_wavefronts`, `data_bank_conflicts`. Global loads are a tag
  problem (which sectors?); shared loads are a data problem (which banks?).
  **That is why coalescing is counted in sectors and bank conflicts are counted
  in wavefronts.** They happen in different stages of the same unit.
- **`smsp__` vs `sm__`.** Anything about *issue* — eligible warps, stall
  reasons, instructions issued — is a property of a single warp scheduler and
  therefore `smsp__`. Anything about the SM as a resource pool — warps
  resident, pipe utilisation — is `sm__`. Module 19 showed this is not
  cosmetic: the register file is **four slices of 16384**, not one pool of
  65536, and the occupancy formula built on the aggregate is wrong on 7% of
  kernels.

### Why there is a counter-slot limit, and what it costs you

The counter hardware is small, fixed-function and shared. Programming it to
observe a particular signal consumes a *collection slot*, and there are only so
many; some signals additionally conflict with each other in the multiplexer.
`ncu`'s job at startup is to solve a bin-packing problem: group the requested
metrics into the minimum number of mutually compatible **passes**.

That is where replay comes from and why `--set full` (7314 metrics) is so much
more expensive than `--set basic` (213). It is also why **two metrics collected
in different passes are measured on different executions of your kernel**. For
a deterministic kernel on a quiet machine that is harmless. For a kernel whose
behaviour depends on cache state, it is exactly the perturbation of §3.

### `active` vs `elapsed`, as hardware

Every unit has two cycle counters: one that ticks whenever the GPU clock ticks
during the kernel (`elapsed`), and one that ticks only while that unit has work
(`active`). `sm__cycles_active.avg` versus `gpc__cycles_elapsed.max` is the gap
between "the SMs were busy" and "the kernel was running".

For a one-wave launch on 40 SMs where one SM gets a block twice as expensive as
the rest, every SM's `active` count is its own busy time, so the per-SM
utilisation ratios all look fine; `elapsed` is the same for all of them and
equals the slowest. **This is Module 19's measurement restated as hardware.**
It is also the reason `ncu`'s Occupancy section shows the `active` form: it is
the one that answers "when this SM was working, how full was it", which is a
legitimate question — just not the question "is my kernel wasting the machine".

### What a "wavefront" is

M7 promised this module would define it. A warp's shared-memory instruction is
presented to the bank crossbar. The crossbar can service, in one pass, any set
of lane requests in which **no two banks are asked for different words**
(same-word requests broadcast for free). If the warp's 32 lanes cannot be
satisfied in one pass, the hardware splits them into the minimum number of
passes that can — each pass is a **wavefront**, and the count is the conflict
degree `D`. A 16-byte access (`LDS.128`) is additionally phase-split into four
phases of 8 lanes before any of this happens, which is why M18's correction to
the cost law exists: a single phase already asks for the crossbar's full 128 B
width, so there is no slack cycle to absorb a 2-way conflict.

---

## Code Walkthrough

Both examples **build** the sections `ncu` would print, rather than reading
them. Every metric name they print is the real one, verified against the
section definitions installed with Nsight Compute 2026.1.0.

### `example01.cu` — reconstructing Memory Workload Analysis

**Part A** implements M5's sector-counting procedure as code — the exact
definition of `l1tex__t_sectors_... / l1tex__t_requests_...`:

```cpp
static void warpFootprint(AddrFn f, int elemBytes, int *sectors, int *lines)
{
    size_t sec[32], lin[32]; int ns = 0, nl = 0;
    for (int lane = 0; lane < 32; ++lane) {
        size_t a0 = f(lane);
        for (int b = 0; b < elemBytes; b += SECTOR_BYTES) {
            size_t s = (a0 + (size_t)b) / SECTOR_BYTES;   // addr >> 5
            /* ...count distinct... */
        }
    }
}
```

The inner loop over `elemBytes` is there so a `float4` is counted honestly: a
16-byte access can straddle a sector boundary and must be charged for both.
The table it prints is §6c's table, with a `MODEL DISAGREES` marker beside any
row where the enumeration contradicts the hand-derived expectation — so the
program fails loudly if the model is wrong, rather than printing a plausible
number.

**Part B** is the bank-conflict counter, and the function is three lines of
real content:

```cpp
int w = f(lane);                 // word index
int b = w % SMEM_BANKS;          // bank = (addr/4) % 32
/* ...insert w into bank b's set of DISTINCT WORDS, not lanes... */
```

Counting *lanes* per bank instead of *distinct words* is the single most common
way to misread this counter, and it is the error that turns a free broadcast
into a phantom 32-way conflict. The printed table includes an `Ada cycles`
column computed as `max(2, D)` so that `s[2*tid]` visibly costs the same as
`s[tid]`.

**Part C** measures DRAM throughput for four kernels whose *useful* bytes
differ by 8× while their *moved* bytes are identical:

```cpp
const double usefulB[4] = { NBYTES, NBYTES, NBYTES/2.0, NBYTES/8.0 };
const double movedB[4]  = { NBYTES, NBYTES, NBYTES,     NBYTES     };
```

The point is the contrast between the two columns it prints from these, which
is §6a point 2: `dram__throughput` is identical for all four and tells you the
bus is busy; sectors-per-request tells you whether it is busy on your behalf.

**Part D** does not fake a hit rate. It states that no stopwatch can produce
one, and then produces M16's *bound* instead.

### `example02.cu` — Speed of Light, Occupancy, Warp State

**Part A** runs one kernel per quadrant of the Speed-of-Light plane: a
streaming read (memory high), a saturating FFMA chain (compute high), a
two-shared-loads-per-FMA kernel (both mid — the on-chip operand ceiling), and a
single-warp dependent pointer walk (both near zero). It prints the decision
tree of §7 underneath, with the fourth row flagged as the one that gets
misdiagnosed.

Note the FFMA kernel's shape, which is M20's and M21's rules applied:

```cpp
#define CHAINS 8
    #pragma unroll 8
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int i = 0; i < CHAINS; ++i) a[i] = fmaf(a[i], b, 1.0f);
    }
```

Eight independent chains because `L/T = 3.8` means four already saturate, and
the outer loop unrolled 8× because M21 measured an under-unrolled FFMA probe
reporting a reproducible and entirely fictitious 13 750 GFLOP/s against 18 642 —
33% of its issue slots were loop overhead.

**Part B** is M19's instrument: `clock64()` at entry and exit, `%smid` to find
the SM, three atomics per warp, and two different denominators computed from
the same three arrays. The reduction is the content:

```cpp
sumA += (double)hRes[i] / (WARP_SLOTS_PER_SM * span);      // this SM's own span
sumE += (double)hRes[i] / (WARP_SLOTS_PER_SM * maxSpan);   // the kernel's span
```

and `maxSpan` is a **maximum of per-SM spans**, never a `max(end) − min(start)`
across SMs, because the per-SM `clock64()` counters are not mutually
synchronised — M19 measured up to 298 million cycles of offset inside a single
launch.

**Part C** reconstructs `smsp__thread_inst_executed_per_inst_executed.ratio`
exactly:

```cpp
unsigned m = __activemask();
int lead = __ffs(m) - 1;
if ((int)(threadIdx.x & 31u) == lead) {
    atomicAdd(&c[0], (unsigned long long)__popc(m));   // thread_inst_executed
    atomicAdd(&c[1], 1ull);                            // inst_executed
}
```

The reporter must be the **lowest set bit** of the active mask, not lane 0:
lane 0 may not be active in the region being measured, and then nothing is
counted at all. The section closes with the stall-reason table of §6g.

---

## Check Your Understanding

**Q1.** You profile a kernel and `ncu` reports
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum = 3,145,728` against
`smsp__sass_inst_executed_op_shared_ld.sum = 3,145,728`. A colleague says
"a million conflicts, pad the array." Compute the conflict degree from those
two numbers, then say under what circumstances the colleague is right, under
what circumstances padding will do nothing, and under what circumstances
padding will make the kernel *slower*. Name the measurement from an earlier
module that supports each of your three cases.

**Q2.** Two kernels, same source, same grid, same block size. Kernel A reports
Achieved Occupancy 94% and `smsp__warps_eligible.avg.per_cycle_active = 0.11`.
Kernel B reports Achieved Occupancy 31% and
`smsp__warps_eligible.avg.per_cycle_active = 2.8`. Both run for the same
number of cycles. Which one has more headroom, what single change would you try
on each, and which `ncu` section do you open next for each? Justify from the
resident/eligible/issued chain, not from "occupancy is good".

**Q3.** You are told that a kernel's Achieved Occupancy went from 61.9% to
62.3% after a change, and that the change made it 7.5% slower. Explain how both
statements can be true simultaneously, name the metric that *would* have shown
the regression, and explain why that metric is not in `ncu`'s Occupancy
section.

**Q4.** `ncu` reports, for a kernel you wrote, DRAM Throughput 95.0% of peak and
`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio = 4.00`.
A second kernel reports DRAM Throughput 95.0% and `...ratio = 32.00`. Both read
the same logical array and produce the same answer. Which is faster, by roughly
what factor, and why does the Speed-of-Light section give them identical
verdicts? What would you have to change for the *slow* one's DRAM throughput to
*drop*, and would that be an improvement?

---

## Exercises

### Exercise 1 — `module23/exercise01.cu` — interpret a report, then check it

A **constructed** `ncu` report for four kernels you have already written in
this course: a strided read of one column of a row-major table (M5), a tiled
transpose with and without padding (M7/M15), a naive GEMM (M16), and a
streaming sum with one load in flight per thread (M20). The report is clearly
labelled constructed, and every number in it is drawn from this course's own
measurements, so it is physically consistent with this GPU.

You diagnose each kernel **from the metrics alone**, name the single decisive
metric row, and predict which bucket the stated fix's speedup falls into. Then
the program **actually runs** all eight kernels, times each pair back to back
with rotation, validates every one numerically, and scores your predictions
against reality.

Interpretation on paper; verification real.

```
nvcc -arch=sm_89 -O3 -lineinfo -o exercise01.exe exercise01.cu
exercise01.exe
```

**TODO 1** — the diagnosis code for each of the four kernels, from a menu of
six. **TODO 2** — the single decisive metric row for each, from a menu of
eight. **TODO 3** — the speedup bucket for each stated fix (`<1.5×`,
`1.5–6×`, `>6×`). **TODO 4** — one of the four "fix" kernels is left for you to
write; the harness will not score anything until it is correct.
**TODO 5** — one metric in the report is *not* evidence for the diagnosis you
gave, even though it looks like it; identify it.

Scoring: 12 points for the interpretation plus validation of all four pairs.
`OVERALL: PASS` requires both.

### Exercise 2 — `module23/exercise02.cu` — derive the counters yourself

This is the module's compensation for the missing tool, and it teaches the
metrics more deeply than reading them would. You write **instrumented kernels
that compute what `ncu` would report about themselves**, and compare them
against **closed-form predictions you also write**:

- sectors per request, from `__match_any_sync` over `addr >> 5`;
- shared-memory conflict degree, counting **distinct words** per bank;
- achieved occupancy under **both** denominators, with `clock64()` and `%smid`;
- lane efficiency, from `__popc(__activemask())`.

```
nvcc -arch=sm_89 -O3 -lineinfo -o exercise02.exe exercise02.cu
exercise02.exe
```

**TODO 1** — the distinct-key count across a warp, and sectors per request on
top of it. **TODO 2** — conflict degree. The obvious implementation is wrong on
exactly two of the eight patterns tested and the program will show you which.
**TODO 3** — the two occupancy reductions, which differ only in a denominator.
**TODO 4** — lane efficiency, where choosing the wrong reporter lane silently
counts nothing. **TODO 5** — **closed-form** predictions for sectors and
conflict degree: no enumeration allowed, a formula in terms of the stride.

Every row is scored against your formula, and three structural facts about the
two occupancy denominators are scored as well. `OVERALL: PASS` requires all of
them.

### Exercise 3 — `module23/exercise03.cu` — from report to ranked plan

A kernel and a constructed profile. You produce an **optimization plan ranked by
expected payoff**, implement the top item and the bottom item, and measure
whether the ranking held.

The kernel is a per-record normalisation written the way this gets written
first, and it has four separate defects, each of which a different section of
the report flags:

| | defect | section that sees it |
|---|---|---|
| A | reads one field of a 4-float AoS record — **16 sectors per request** | Memory Workload Analysis |
| B | reads a staged lookup table down a column — **32-way bank conflict** | Memory Workload Analysis |
| C | an **IEEE division** per element where a reciprocal would do | Compute Workload Analysis |
| D | launched as **20 blocks of 128 threads**, one load outstanding each — half the SMs get nothing | Launch Statistics, Occupancy, Scheduler Statistics |

All four are real. **They are not equally expensive, and one of them is worth
more than the other three put together.** The report contains the evidence for
which — if you read the right denominator.

```
nvcc -arch=sm_89 -O3 -lineinfo -o exercise03.exe exercise03.cu
exercise03.exe
```

**TODO 1** — rank the four candidate fixes by expected payoff.
**TODO 2** — predict the bucket your top-ranked fix will land in. (You can get
this right without writing a line of code: compute the headroom first.)
**TODO 3** — implement the fix you ranked first. **TODO 4** — implement the one
you ranked *last* as well, so the harness can measure the two ends of your
ranking against each other. **TODO 5** — name the profile row that predicted
the outcome, and the one that misleads.

The harness times all three variants back to back with rotation, validates each
against a double-precision reference with the output prefilled with a NaN
pattern, and scores 9 points. **Only positions 1 and 4 of your ranking are
scored** — the program explains in its own output why the middle two cannot
honestly be scored on this machine, and that explanation is itself one of the
things the exercise exists to teach.

---

## Prediction

Commit to these in writing before you run anything.

**P1.** Exercise 1's kernel 2 is a tiled transpose whose shared tile is
`float t[32][32]`, read as `t[threadIdx.x][threadIdx.y + j]`. Compute the
conflict degree. Then predict the speedup from padding it to `[32][33]` on an
**8192×8192** transpose, and separately on a **2048×2048** one, and say why the
two answers are different. (M7 measured 14.79× for the conflict in isolation;
M15 measured something very different for the same conflict inside a transpose.
Both are in this course.)

**P2.** Exercise 2 asks you for a closed form for the conflict degree of
`s[k*tid]`. Write it down now. Then write down what your formula gives for
`s[tid/2]`, which is not of that form, and predict what the *measured* degree
will be. If your two answers differ, you have found the trap before the program
shows it to you.

**P3.** Exercise 3's profile reports **16,252,928** shared-memory bank conflicts
at 32 wavefronts per request, and an `smsp__warps_eligible.avg.per_cycle_active`
of **0.11** against a theoretical 12.00. Before reading any further: predict
which of those two is worth more to fix and by what factor. Then compute the
**headroom** — the profile gives you `dram__bytes.sum` and `Duration`, and this
course has measured the ceiling — and state the largest speedup *any* fix to
this kernel could possibly produce. If your two answers are inconsistent, trust
the headroom.

---

## Where this goes next

- **Module 22 (Nsight Systems)** owns everything between kernels. If you have
  not profiled the timeline first, you may be optimising a kernel that accounts
  for 15% of the wall time.
- **Modules 38–39 (PTX and SASS)** own the level below this one. Every time
  this module said "check the SASS" — M17's destroyed `LDS.128` merge, M18's
  byte-identical inner loops, M21's hoisted loop-invariant — that is where you
  learn to do it properly.
- **Five questions this course could not settle because `ncu` is unavailable
  here** are listed in `solutions/module23/MANIFEST.md`. If you are reading
  this on a machine where the counters are open, they are all one command away,
  and three of them are genuinely open research in miniature.
