# Module 23 manifest — Nsight Compute

**Status: COMPLETE and verified.** Every `.cu` compiles warning-clean with
`nvcc -arch=sm_89 -O3 -lineinfo`; all three solutions print `OVERALL: PASS` on
this GPU; all three shipped exercises compile with their TODOs blank and exit
gracefully.

| file | state |
|---|---|
| `module23/lesson.md` | **new** |
| `module23/example01.cu` | reused from the interrupted run; one gate softened (see below) |
| `module23/example02.cu` | reused; one gate softened |
| `module23/exercise01.cu` | **new**, derived from the surviving solution |
| `module23/exercise02.cu` | **new**, derived from the surviving solution |
| `module23/exercise03.cu` | **new** |
| `solutions/module23/exercise01_solution.cu` | reused; **answers were stale and are corrected** (see below) |
| `solutions/module23/exercise02_solution.cu` | reused; two stale text claims fixed |
| `solutions/module23/exercise03_solution.cu` | **new** |
| `solutions/module23/exercise0{1,2,3}_solution.md` | **new** |
| `solutions/module23/check_your_understanding.md` | **new** |

---

## The constraint this module is built around

`ncu` (Nsight Compute **2026.1.0.0, build 37166530**) is installed at
`C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\ncu.bat` and every
invocation that touches a counter fails. Confirmed twice during authoring,
verbatim:

```
==PROF== Connected to process 11068 (...\t.exe)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA
          GPU Performance Counters on the target device 0. For instructions on
          enabling permissions and to get more information see
          https://developer.nvidia.com/ERR_NVGPUCTRPERM
==PROF== Disconnected from process 11068
```

`ncu --query-metrics` fails with the same error — even *enumerating* the
counters is gated. **No workaround was attempted**, per the standing
instruction.

### What DOES work, and is used

**Host-side listings work** and are quoted verbatim in the lesson:

- `ncu --list-sets` → the real section membership and the **Estimated Metrics**
  column (basic 213, detailed 1007, full **7314**, roofline 5932). That column
  is the lesson's concrete handle on replay cost.
- `ncu --list-sections` → 25 section identifiers, so every `--section` name in
  the lesson is one that exists in 2026.1.0 rather than one from documentation.
- `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\sections\*.section`
  → the `Label:` / `Name:` pairs that define what each section prints.
  **Every metric name in this module was checked against these files**, so the
  display labels ("Achieved Occupancy", "Avg. Active Threads Per Warp",
  "Warp Cycles Per Issued Instruction", "No Eligible") are the real ones for
  this chip and this tool version.

This is why the module can be specific about things like
`smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`
rather than the `smsp__warp_issue_stalled_*_per_warp_active.ratio` family M20
quoted — **both exist; the first is the one the `WarpStateStats` section
actually displays on sm_89 in 2026.1.0.**

---

## Concepts taught

1. **Division of labour `nsys` vs `ncu`**, with the permission asymmetry
   explained: tracing uses timestamps the driver already produces for your
   process; counters are **global to the device** and observe every context on
   it, so they are a cross-process side channel and NVIDIA gates them.
2. **Kernel replay**: save device memory → program one pass of counters → run →
   restore → repeat. Why a profiled run is vastly slower, why wall-clock time
   from a profiled run is meaningless, why **replay perturbs caches**, and the
   `--replay-mode kernel|application|range` trade-off.
3. **The metric naming grammar** — `unit__quantity_qualifiers.rollup.submetric`
   — with the unit table (`dram__`, `lts__`, `l1tex__`, `smsp__`, `sm__`,
   `gpc__`, `gpu__`, `launch__`) tied to the Module 4 storage map, and the
   **`active` vs `elapsed`** distinction as the single most important thing in
   the grammar.
4. **`--set` vs `--section` vs `--metrics`**, with the real `--list-sets` output
   and its metric-count column as the predictor of pass count.
5. **Ten metrics, each anchored to a measurement the reader already made.**
   See the anchor table below.
6. **The interpretation workflow as a decision tree**: Speed of Light → the
   matching section → stalls → `-lineinfo` source attribution; with the
   "both low" quadrant called out as the one that is routinely misdiagnosed.
7. **What `ncu` hides**: everything between kernels (M22), replayed caches,
   achieved-occupancy blindness to tails and imbalance (M19), and the fact that
   no counter detects a missing bounds check (M18).
8. **`ERR_NVGPUCTRPERM` itself** — what it means, why the gate exists, and the
   exact Windows (NVIDIA Control Panel → Developer → Manage GPU Performance
   Counters; `RmProfilingAdminOnly`) and Linux
   (`NVreg_RestrictProfilingToAdminUsers=0`) administrator fixes.
9. **What a wavefront is** — M7's explicitly deferred vocabulary — and why
   coalescing is counted in *sectors* (L1 tag stage) while bank conflicts are
   counted in *wavefronts* (L1 data stage).

---

## The anchor table — this is the module's distinctive angle

Every metric is introduced by naming the earlier module in which the reader
derived the same physical quantity by hand.

| metric | anchored to | the number the reader already has |
|---|---|---|
| `dram__bytes.sum.per_second`, `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | **M12 / M21** | 410.5–410.7 GB/s = 95.0% of the 432.0 pin peak; 372–373 GB/s with only a 400 ms warm-up |
| `lts__t_sectors`, `lts__t_sector_hit_rate.pct` | **M21 / M5 / M15** | L2 read ceiling 1259–1938 GB/s on a 24 MB set; apparent >100% of "peak" means residency, not bandwidth |
| `l1tex__t_sector_hit_rate.pct` | **M16** | the stopwatch *bound* `on-chip ≥ 1 − (pin peak × elapsed)/requested` → ≥ 91.6–93.4% for the naive GEMM |
| `l1tex__t_sectors_... / l1tex__t_requests_...`, `smsp__sass_average_data_bytes_per_sector_...` | **M5** | the sector-counting procedure: 4 / 5 / 8 / 16 / 32 / 1 for `in[i]` / `in[i+1]` / `in[2i]` / `in[4i]` / `in[≥8i]` / `in[0]`. **Sectors per request IS coalescing efficiency** |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`, `l1tex__data_pipe_lsu_wavefronts_...` | **M7 + M18** | `D = gcd(k,32)`; cost `max(2,D)` for 4 B (so D=2 is free, measured 1.00×); **∝D with no floor for `LDS.128`** (M18: 1.61× measured vs 1.67× predicted on byte-identical SASS) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` **and** `..._elapsed`, `launch__occupancy_limit_*` | **M19** | the four-limiter **slice** model `4*floor(16384/(roundUp(R,8)*32))/warpsPerBlock`, exact on 137/137 kernels; and the blindness: a tail moves `_active` +0.4 and `_elapsed` −4.0; an imbalance moves `_active` **up** 3.4 while `_elapsed` falls 14.4 |
| `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` | **M20** | FFMA latency **4.051 cycles**, issue interval **1.067**, `L/T = 3.8`; and the three reasons where more occupancy makes it *worse* |
| `smsp__warps_eligible.avg.per_cycle_active`, `smsp__issue_active...` | **M19 + M20** | resident ≥ eligible ≥ issued; high occupancy with near-zero eligible is the signature of latency binding |
| `smsp__thread_inst_executed_per_inst_executed.ratio` + `..._pred_on_...` | **M8** | hand-computed lane efficiency 47.7% and 56%; divergence 1.96× vs warp-uniform 1.02%. **The *pair* resolves M8's open problem**: `__activemask()` cannot separate predication from branching, but the gap between the two counters is exactly the predication |
| `sm__inst_executed_pipe_fma/_lsu/_xu`, `sm__sass_thread_inst_executed_op_ffma_pred_on.sum` | **M18 + M21** | FFMA density 25% (naive GEMM) vs 90.8% (8×4 register tile); the FP32 plateau *is* the issue ceiling at `density × 64 × 310 G instr/s` |
| the roofline sections | **M21**, which handed this module the debt explicitly | five measured ceilings; AI is meaningless without naming its level (tiled GEMM: 181 / 4.0 / 0.25); the `<0.25` bucket means go to Little's Law |

---

## CUDA API / intrinsics used

Nothing new is introduced as *CUDA*. The exercises use, from earlier modules:
`__match_any_sync`, `__ballot_sync`, `__activemask`, `__popc`, `__ffs`,
`clock64()`, `%smid` inline PTX, `atomicAdd`/`atomicMin`/`atomicMax` on
`unsigned long long`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
`cudaEvent_t` timing, `__constant__`, `#pragma unroll 1`.

New **tool** surface: `ncu --list-sets`, `--list-sections`, `--query-metrics`,
`--set`, `--section`, `--metrics`, `--kernel-name`, `--kernel-name-base`,
`--launch-skip`, `--launch-count`, `--target-processes`, `--replay-mode`,
`--cache-control`, `--clock-control`, `--csv`, `--page raw|details`, `-o`,
`--import`, `--baseline`, `ncu-ui`.

---

## Worked examples

**`example01.cu` — reconstructing Memory Workload Analysis.** Parts A and B are
host-side *enumerations* (sector counting, bank bucketing) that print the real
metric names beside the quantity and flag any row where the enumeration
disagrees with the hand-derived model. Parts C and D are measured: four kernels
whose useful bytes differ 8× while their moved bytes are identical, and M16's
stopwatch bound on hit rate.

**`example02.cu` — Speed of Light, Occupancy, Warp State.** Four kernels, one
per Speed-of-Light quadrant (streaming / FFMA / two-shared-loads-per-FMA /
single-warp pointer chase), M19's `clock64()`+`%smid` occupancy instrument with
both denominators, and an exact reconstruction of
`smsp__thread_inst_executed_per_inst_executed.ratio`.

---

## Exercises

| # | type (spec §6) | TODOs | trap |
|---|---|---|---|
| 1 | 2 (predict) + 6 (perf reasoning) + 1 (fill in) | 5 | **The loudest counter in the report is attached to the defect worth the least.** Kernel 2 has 65 M bank conflicts at degree 32 — the worst the hardware can produce — and removing them measured **0.96×**, because the kernel is at 90% of the DRAM roof. Kernel 4 has a *flawless* access pattern and is 3.3× too slow |
| 2 | 1 (fill in) + 7 (instrumentation) | 5 | **Counting lanes per bank instead of distinct words.** The program prints both columns; they disagree on exactly `s[tid/2]` and `s[0]`, where the lane count invents a 2-way and a 32-way conflict that both cost nothing. Secondary trap: choosing lane 0 instead of the lowest set bit of `__activemask()` as the reporter — it silently counts nothing and still prints a ratio |
| 3 | 4 (optimization) + 6 (perf reasoning) | 5 | **Four real defects, one binding constraint.** The ranking is decided by `smsp__warps_eligible = 0.11`, not by the 16 M bank conflicts. And the fix ranked near the bottom (layout, 1.02×) becomes the **best** fix (2.28×) once the top fix has moved the binding constraint |

Exercise 3 deliberately **refuses to score positions 2 and 3** of the ranking and
says so in its own output: the three non-binding fixes measure 1.02 / 1.07 /
1.15×, inside each other's spread, and spec §12.5d forbids scoring a distinction
narrower than the noise.

---

## Measured results recorded in this module

All with `nvcc -arch=sm_89 -O3 -lineinfo`, one program at a time with ≥60 s
cool-downs (spec §12.5c), 1500 ms streaming warm-up, rotated back-to-back
sweeps, min-of-N, validation in a separate pass.

### example01 — sectors, conflicts, DRAM

```
   kernel                      ms   sect/req    effective impliedDRAM    %of432
   streamRead float4       0.6535       16.0     410.8 GB/s   410.8 GB/s     95.1%
   strideRead<1>           0.6695        4.0     401.0 GB/s   401.0 GB/s     92.8%
   strideRead<2>           0.6540        8.0     205.2 GB/s   410.4 GB/s     95.0%
   strideRead<8>           0.6526       32.0      51.4 GB/s   411.3 GB/s     95.2%
```

**410.8 GB/s = 95.1% of the 432 pin peak**, reproducing M12's 410.5–410.7
exactly. And the point of the table: the four kernels' *useful* bandwidth spans
8× while `dram__throughput` is identical to within 2.4 points.

### example02 — the four Speed-of-Light quadrants

```
   ceilings measured in this run: DRAM 410.9 GB/s, FP32 19841 GFLOP/s

   kernel                     Compute%    Memory%   verdict
   kStream (640,256)              0.5%      95.1%   memory bound
   kFfma (480,128)              100.0%       0.0%   compute bound
   kOperandBound (480,128)        6.8%       0.0%   on-chip operand fetch
   kChase (1,32)                  0.0%       1.5%   LATENCY - nothing saturated
```

**19,841 GFLOP/s** sits inside M20/M21's widened 17,123–21,200 band;
`kOperandBound` at **6.8%** sits just under M21's predicted 7.0–13.0% window
for a two-shared-loads-per-FMA kernel — the shortfall is this probe's own loop
overhead, which M21 measured as the standard hazard of an under-unrolled probe; `kChase` at 1.5% of memory and 0.0% of compute
is the "both low" quadrant.

Occupancy, both denominators:

```
   configuration               blocks   occ_active  occ_elapsed         ms
   uniform, exactly 1 wave        240        67.4%        67.4%     0.348
   uniform, 1 wave + 1 blk        241        67.3%        58.0%     0.383
   1..8x imbalance, 1 wave        240        70.2%        42.3%     0.539
```

Lane efficiency: 16.00 / 32.00 thread-instructions per instruction = 50% / 100%.

### exercise01 — the four pairs

```
   pair                          before ms   after ms   speedup   bucket
   k1 strided -> packed             1.3056     0.1732     7.54x        3
   k2 [32][32] -> [32][33]          1.5217     1.5843     0.96x        1
   k3 naive -> 4x4 reg tile         3.4782     0.9095     3.82x        2
   k4 serial -> 4 in flight         3.9154     1.1733     3.34x        2
   SCORE: 13/13   OVERALL: PASS
```

**k2 at 0.96× reproduces M15's 0.96–1.03× band** for the identical 32-way
conflict inside a DRAM-bound 8192² transpose — and it is a *loss*, i.e. padding
cost more than the conflict it removed.

### exercise02 — reconstructed counters, exact

```
   expression       measured  predicted bytes/sector used/moved
   in[i]                   4          4        32.00     100.0%
   in[i + 1]               5          5        25.60      80.0%
   in[2*i]                 8          8        16.00      50.0%
   in[4*i]                16         16         8.00      25.0%
   in[8*i]                32         32         4.00      12.5%
   in[16*i]               32         32         4.00      12.5%
   in[0]                   1          1       128.00  broadcast
   in[2*i + 1]             8          8        16.00      50.0%

   expression      degree  predicted   wavefronts   conflicts    by-lane
   s[tid]               1          1            1           0          1
   s[2*tid]             2          2            2           1          2
   s[3*tid]             1          1            1           0          1
   s[4*tid]             4          4            4           3          4
   s[8*tid]             8          8            8           7          8
   s[32*tid]           32         32           32          31         32
   s[tid/2]             1          1            1           0          2
   s[0]                 1          1            1           0         32

   configuration               blocks   occ_active  occ_elapsed        ms
   uniform, exactly 1 wave        240        67.3%        67.3%    0.463
   uniform, 1 wave + 1 blk        241        67.3%        58.0%    0.526
   1..8x imbalance, 1 wave        240        70.0%        42.2%    0.738

   predicate                     thr_inst/inst    predicted   lane eff
   p = i & 1        (2-way)              16.00        16.00      50.0%
   p = (i>>5) & 1   (uniform)            32.00        32.00     100.0%
   p = (i&31)==7    (1 of 32)             1.00         1.00       3.1%
   SCORE: 23/23   OVERALL: PASS
```

Every closed-form prediction matched the device-side measurement on every row,
on every run. The `by-lane` column disagrees on exactly the two broadcast rows.

### exercise03 — the ranked plan

```
   variant                          ms    speedup   DRAM GB/s
   e3_base                      2.5682      1.00x    130.7 GB/s
   F1 CONCURRENCY               0.8409      3.05x    399.0 GB/s
   F2 LAYOUT                    2.5245      1.02x     53.2 GB/s
   F3 CONFLICTS                 2.3985      1.07x    139.9 GB/s
   F4 DIVIDE                    2.2354      1.15x    150.1 GB/s
   SCORE: 9/9   OVERALL: PASS
```

Reproduced over three runs: F1 **3.06 / 3.05 / 3.02×**, F4 **1.14 / 1.15 /
1.14×**. The ceiling calculation made *before* implementing anything —
`410.5 / 130.7 = 3.14×` maximum available — predicted the outcome to within 3%.

**And the result that justifies the whole exercise**, measured separately:

```
F1       1.0661 ms  314.7 GB/s
F1+F2    0.4668 ms  287.5 GB/s   speedup over F1 = 2.28x
```

**The same layout fix is worth 1.02× before the concurrency fix and 2.28× after
it.** An optimization plan is not a fixed list; it has to be re-derived every
time the binding constraint moves.

---

## Corrections made to the surviving files

1. **`exercise01_solution.cu`: the shipped `DIAG[1]`/`METRIC[1]` disagreed with
   the file's own answer hashes.** The hashes encoded `DIAG[1]=2`
   (bank conflicts) and `METRIC[1]=5` (the conflicts counter); the visible
   arrays said `1` and `1`. The solution as it stood would have scored 11/12.
   **Resolved in favour of `1`/`1`** — "at the DRAM roof, stop" — which is the
   pedagogically stronger reading and the one M15 measured, and the hashes were
   regenerated. Kernel 2's constructed report was rewritten accordingly: it now
   shows **90.71% DRAM throughput together with 65,011,712 bank conflicts**, so
   the loud counter and the binding ceiling are visible in the same block.
2. **`exercise01_solution.cu`: kernel 2's report described a 2048×2048 transpose
   while the harness times an 8192×8192 one.** All of block [2] was rebuilt for
   the real shape (grid 65,536, 536.87 MB, waves/SM 273.07, 2,097,152 shared
   requests → 67,108,864 wavefronts → 65,011,712 conflicts).
3. **`exercise01_solution.cu`: `K4_N` was 32,768,000 while its report block
   claimed 65,536,000 and 262.14 MB.** `K4_CHUNK` 1600 → 3200, and the sector /
   request counts recomputed (8,192,000 / 2,048,000 = 4.00). The duration
   3.8450 ms now equals 262.14 MB at M20's measured 68.2 GB/s exactly.
4. **`exercise01_solution.cu`: kernel 3's instruction counts.**
   `smsp__inst_executed.sum` set to 135,664,516 so that
   1,085,316,141/32 ÷ 135,664,516 is exactly the 25.00% M18 measured;
   `dram__bytes.sum` set to the exact compulsory 12.67 MB.
5. **`exercise02_solution.cu`, the note left by the interrupted agent
   ("fix Exercise 2's stale text and warning"):** the text claimed a 4-byte
   misalignment "costs a fifth more sectors" (it is a *quarter*: 5 vs 4), and
   the `used/moved` column printed **400.0%** for the broadcast row — the same
   species of lie as calling a broadcast 12.5% efficient. The broadcast row now
   prints `broadcast`, and the misalignment note now also carries M5's measured
   97.9% against its predicted 80%.
6. **Both examples' operating-point gates were hard FAILs below 150 GB/s.**
   Spec §12.5b says *warn and proceed*. During authoring this GPU spent several
   minutes pinned in P8 and produced a **false FAIL**; see the honesty note
   below. The gates now FAIL only on a figure **above** a hardware bound (which
   means the arithmetic is wrong) and **warn** below one, stating that the
   enumerated parts are unaffected.

---

## Forward-reference debts PAID here

| owed by | debt | where paid |
|---|---|---|
| **M5** (`lesson.md:305-310`) | *"Module 23 will have you read exactly that metric"* — `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | lesson §6c; example01 Part A; exercise02 Part A |
| **M7** (`lesson.md:308-332`, `MANIFEST:155-159`) | the **wavefront vocabulary**, Memory Workload Analysis, and `--kernel-name` / `--launch-count` hygiene | lesson "What a wavefront is" (Hardware Mental Model) and §8 |
| **M7** (CYU Q4) | *"conflicts = 0 and the kernel is 4% faster"* | CYU Q1, with M15's 0.96–1.03× and exercise 1's measured **0.96×** |
| **M8** (`lesson.md:476`) | the counter equal to the hand-computed lane efficiency, and divergence vs latency stalls reported separately | lesson §6i; example02 Part C; exercise02 Part D |
| **M8** (open problem) | `__activemask()` cannot distinguish predication from branching | **resolved**: the *pair* `..._per_inst_executed.ratio` and `..._pred_on_per_inst_executed.ratio` separates them; the gap is the predication |
| **M16** (`lesson.md:184`) | *"no cache-hit counter can be read"* → the stopwatch bound | lesson §6d names `l1tex__/lts__t_sector_hit_rate.pct` and keeps M16's bound as the cross-check |
| **M19** (`MANIFEST:297-301`) | ownership of `sm__warps_active.avg.pct_of_peak_sustained_{active,elapsed}`, `smsp__warps_eligible`, `smsp__issue_active`, stall counters | lesson §6b, §6h; example02 Part B; exercise02 Part C; CYU Q3 |
| **M20** (`MANIFEST:95-101`) | the stall-reason counter family and its section names | lesson §6g with the full "does occupancy help?" table |
| **M21** (`lesson.md:679-686`) | the automated roofline chart and `dram__bytes.sum`, `l1tex__t_bytes.sum`, `smsp__inst_executed.sum`, `sm__sass_thread_inst_executed_op_ffma_pred_on.sum` | lesson §6j |
| **M22** | the `nsys`/`ncu` boundary, from the `ncu` side | lesson §2 |

---

## Forward references made

- **Modules 38–39 (PTX / SASS)** — every "check the SASS" in this module
  (M17's destroyed `LDS.128` merge, M18's byte-identical loops, M21's hoisted
  loop-invariant, and the fact that **no `ncu` metric reports the access
  *width***, which is what decides between the `max(2,D)` and `∝D` cost laws).
- **Module 22** — the timeline question that must be answered before any `ncu`
  question is worth asking.
- `module23/lesson.md` §"Where this goes next" points at the five unsettled
  questions below.

---

## Unsettled questions this module could NOT answer, and the exact command

These were all explicitly deferred to Module 23 by earlier modules and remain
open **because the counters are gated on this machine**. On a machine where
`ncu` runs, each is one command.

1. **M5 Exercise 3's residual 6%** between two pitches that are both 100%
   sector-efficient (pitch progression 82 → 89 → 100%, two 100% pitches still
   6% apart).
   `ncu --metrics lts__t_sectors.sum,dram__sectors.sum --kernel-name <k> ./prog`
2. **M7's `float4` phase-model discrepancy.** `qq[(tid%8)*8]` and
   `qq[(tid%32)*8]` should be identical under the rigid 4-phase model and
   measure **1.53× apart**. The 8-byte case is confirmed exact at 1.00×.
   `ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
3. **M11's `__stcs` / `__stwt` null result.** Both measured no effect; the
   hypothesis is that their real effect is on the **L2 hit rate of a *different*
   array**, which no stopwatch can see.
   `ncu --metrics lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum`
4. **M15's conflicted-vs-padded transpose mechanism.** M15 recorded its
   explanation as a hypothesis, not a measurement, for want of DRAM counters.
   Module 23's Exercise 1 reproduces the 0.96× but still cannot say *why* from
   the inside.
   `ncu --section MemoryWorkloadAnalysis --section SpeedOfLight`
5. **M21's tiled-GEMM 1.14–1.24× over-prediction.** Two candidate mechanisms
   (the `As[ty][k]` broadcast, and M17's `LDS.128` contraction), distinguishable
   by one counter.
   `ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,smsp__sass_inst_executed_op_shared_ld.sum`

Also open and named by their modules: **M14's** sector-vs-address atomic
mechanism (`lts__t_sectors_op_atom.sum`) and **M20's** unpaid `__restrict__`
measurement.

---

## Cross-module observations for the index

Proposed additions to `CROSS_MODULE_INDEX.md` §6b (orchestrator to apply):

| Quantity | Measured / established |
|---|---|
| **`ncu` host-side listings work without counter permission** | `--list-sets`, `--list-sections` and the installed `sections/*.section` files give the real metric names, display labels and per-set metric counts (basic **213**, detailed 1007, full **7314**, roofline 5932) on a machine where collection is blocked. `--query-metrics` does **not** work — it fails with `ERR_NVGPUCTRPERM` like everything else |
| **`ncu`'s "Achieved Occupancy" is the `active` form** | `sm__warps_active.avg.pct_of_peak_sustained_active`. The `_elapsed` sibling exists but is **in no section**; it must be requested with `--metrics`. M19's two blindnesses therefore apply to the default report |
| **The stall-reason family displayed on sm_89 in ncu 2026.1.0** | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` (not the `smsp__warp_issue_stalled_<reason>_per_warp_active.ratio` spelling M20 quoted — both exist; the first is what `WarpStateStats` prints). Reasons: `drain, imc_miss, barrier, gmma, branch_resolving, membar, short_scoreboard, sleeping, wait, no_instruction, math_pipe_throttle, tex_throttle, lg_throttle, dispatch_stall, misc, not_selected, selected, long_scoreboard, mio_throttle` |
| **Predication vs divergence is separable by counters** | `smsp__thread_inst_executed_per_inst_executed.ratio` counts active lanes; `smsp__thread_inst_executed_pred_on_per_inst_executed.ratio` counts predicated-on lanes. A predicated-off lane is in the first and not the second; a branched-away lane is in neither. **The gap is exactly the predication.** This resolves M8's recorded open problem |
| **A 32-way shared conflict inside a DRAM-bound 8192² transpose, re-measured** | padding `[32][32]`→`[32][33]` is **0.96×** — a loss. Independently reproduces M15's 0.96–1.03× band |
| **A latency-bound kernel's layout fix is worth nothing until the latency is fixed** | the same AoS→SoA change measured **1.02×** at 2560 threads with 1 load in flight and **2.28×** at 163,840 threads with 4 loads in flight. It also *lowers* achieved DRAM throughput both times (130.7→53.2 GB/s in the first case) |
| **Sector amplification can be load-bearing concurrency** | a warp's single `LDG` over a 16 B stride covers 16 sectors = 512 B in flight; the "fixed" SoA version covers 4 sectors = 128 B. Fixing the layout of a latency-bound kernel reduces its memory-level parallelism. Same mechanism M21 used to reconcile the pointer chase's 280–288-cycle effective latency with M4's 575 |
| **A pre-implementation headroom calculation predicts the bucket** | `ceiling / achieved` = 410.5/130.7 = **3.14× maximum available**; the implemented fix measured **3.02–3.06×**. Do this arithmetic before writing code |
| ⚠️ **New failure mode of this laptop: a persistent P8 lock** | distinct from the `0x24` power-cap state §7 records. Observed for **>10 minutes**, including under 100% GPU utilisation: SM pinned at **210 MHz**, memory at **405 MHz**, P8, 11 W, 49–52 °C, current power limit **30 W** against a 60 W default, throttle reasons `0x24`. A plain float4 streaming read measured **14.1 GB/s** (3.3% of peak) reproducibly across 150 consecutive timed segments. It cleared on its own. **It produced a false `OVERALL: FAIL` on a correct program.** Any harness whose gate is a lower bound on absolute throughput must warn, not fail |

---

## Constraints observed

- Spec §12.5c: every program verified **one at a time with ≥60 s cool-downs**.
  The one violation during authoring (example02 run immediately after
  example01) measured the DRAM ceiling at **127 GB/s**; the same binary after a
  75 s cool-down measured **410.9 GB/s**. The rule is not optional.
- Spec §12 rules 1, 2, 3, 4, 9, 12: back-to-back rotated sweeps, validation in a
  separate pass, min-of-N with `SWEEPS >= NCFG`, 1500 ms streaming warm-up
  (plus 500 ms compute where an on-chip ceiling is measured), auto-scaled
  iteration counts targeting ~10 ms segments.
- Spec §12.5b: every scored quantity in all three exercises is a **ratio formed
  inside one rotated sweep**, which needs no operating-point guard. Verified
  empirically — exercise 1 scored 13/13 and exercise 3 scored 9/9 from the
  degraded P8 operating point as well as from a healthy one.
- Spec §12.5d: exercise 3 **declines to score** the middle two positions of the
  ranking and prints the measured numbers that justify the refusal.
- Spec §12.11: `#pragma unroll 1` on every baseline whose defect is a *lack* of
  memory-level parallelism. An earlier draft of exercise 3 without it ran the
  "defective" baseline at **232 GB/s** instead of 131 — `ptxas` had unrolled the
  grid-stride loop and manufactured the MLP the kernel was supposed to lack.
- Spec §12.13: upper bounds built from `nvidia-smi --query-gpu=clocks.max.sm`
  (3105 MHz → 31,795 GFLOP/s) and the 432.0 GB/s pin rate, never from the
  2.04 GHz observation.
- No `.exe`, `.exp`, `.lib`, `.pdb`, `.ncu-rep` left behind.
- Nothing in `module23/` reveals an answer. All answer hashes are FNV-1a of a
  tagged string; the plaintext appears only under `solutions/`.

---

## Known issues / honesty notes

1. **No real `ncu` output appears anywhere in this module**, and every
   constructed report says so in its own first five lines, quoting the verbatim
   error. Every constructed value is either taken directly from a course
   measurement or derived arithmetically from one, and the derivations close:
   kernel 1's 16,777,216 sectors × 32 B = the 536.87 MB it reports and = 94.61%
   of peak at its stated 1.3137 ms; kernel 3's FFMA count is exactly
   M·N·K = 1,085,316,141; kernel 4's 3.8450 ms is exactly 262.14 MB at M20's
   measured 68.2 GB/s. **A reader who later runs `ncu` on a working machine
   will not find these reports impossible.**
2. **Exercise 1's constructed report gives kernel 2 a DRAM throughput of 90.71%
   and the harness measured 81.7% in-session.** Both lie inside this course's
   documented session-to-session spread for one binary (279–411 GB/s, index
   §6b). The diagnosis is unaffected — at 82% of the pin rate there is still
   nothing for a bank-conflict fix to recover, which is what the measured 0.96×
   confirms — but the two numbers are not the same number and the solution md
   says so.
3. **The P8 lock described in the cross-module table cost real time and produced
   a false FAIL.** It is recorded here in full because it is a *new* failure
   mode: §7 of the index describes a power cap that recovers after 20–40 s idle,
   and this one persisted for over ten minutes through both load and idle. The
   two example programs' gates were rewritten because of it.
4. **M20's and this module's spellings of the stall-reason metrics differ.**
   M20's lesson quotes `smsp__warp_issue_stalled_<reason>_per_warp_active.ratio`;
   this module quotes `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`
   because that is what the installed `WarpStateStatistics.section` lists for
   this chip. Both metric families exist in the tool. **No change to M20 is
   proposed** — its spelling is valid — but the index entry above records which
   one the default report prints, so a reader is not surprised.
5. **Exercise 3's `EVIDENCE` answer is row 1 (eligible warps) and row 2
   (achieved occupancy) is a defensible near-miss** that gets most of the win.
   The solution md discusses this explicitly rather than pretending row 2 is
   wrong. A reader who argues for row 2 has understood the kernel.
6. **`cudaOccupancyMaxActiveBlocksPerMultiprocessor` is used in example02 and
   exercise02 to get blocks/SM** rather than M19's hand formula. The formula is
   taught in the lesson and the API is used in the code; they agree (6 blocks/SM
   at 256 threads for the probe kernel), but the code does not *check* that they
   agree. M19 already verified the formula on 137 kernels, so re-verifying it
   here would be duplication.
