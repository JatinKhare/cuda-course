# Module 20 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe

nvcc -arch=sm_89 -O3 -cubin -o x1.cubin exercise01_solution.cu
cuobjdump -sass x1.cubin
```

---

## TODO 1 — the independent chains

```cpp
template<int C>
__device__ __forceinline__ void chainSeed(float (&a)[C], unsigned tid)
{
    #pragma unroll
    for (int i = 0; i < C; ++i) a[i] = (float)(tid + i + 1) * 1.0e-3f;
}

template<int C>
__device__ __forceinline__ void chainBody(float (&a)[C])
{
    #pragma unroll
    for (int u = 0; u < BODY / C; ++u)
        #pragma unroll
        for (int i = 0; i < C; ++i) a[i] = fmaf(a[i], FB, FC);
}
```

**Why this is right, from the hardware model.** The scheduler's eligibility
test is "are this warp's next instruction's operands ready". `a[i]` is written
by the previous `FFMA` on chain `i` and read by the next one, so chain `i` can
issue at most one instruction every `L` cycles, where `L` is the FP32 pipeline
latency. `C` chains are mutually unrelated, so in any `L`-cycle window the warp
has `C` instructions that pass the eligibility test. The warp's issue rate is
therefore `min(C/L, 1)` instructions per cycle. That is the function the
measurement traces out.

Four properties carry weight:

- **`fmaf(a[i], b, c)` is a true recurrence.** fp32 FMA is not associative and
  nvcc will not reassociate it without `-use_fast_math` (and even then not
  across a recurrence). There is no transformation that shortens a chain.
- **Distinct seeds.** This turns out *not* to be load-bearing on nvcc 13.2 —
  seeding all `C` chains with the same constant still produces `C` distinct
  accumulator registers in the SASS, because the compiler would have to prove
  the chains equal and does not attempt it for fp. It costs nothing and it is
  the right habit; on an integer chain it would matter.
- **`BODY / C` repetitions keeps the FFMA count per outer iteration at 256 for
  every `C`.** This is the one that actually bites. See "common wrong
  approaches".
- **`C` is a template parameter and both `i` loops are fully unrolled**, so `a`
  is never indexed by a runtime value. A runtime index would put `a` in local
  memory — Module 4: local memory is DRAM — and the measurement would become a
  memory measurement.

**SASS proof, `C = 4`:**

```
/*0170*/   IADD3 R7, R7, 0x1, RZ ;
/*0180*/   FFMA R9,  R9,  R6.reuse, 9.9999999747524270788e-07 ;
/*0190*/   FFMA R11, R11, R6.reuse, 9.9999999747524270788e-07 ;
/*01a0*/   FFMA R13, R13, R6.reuse, 9.9999999747524270788e-07 ;
/*01b0*/   FFMA R15, R15, R6.reuse, 9.9999999747524270788e-07 ;
/*01c0*/   FFMA R9,  R9,  R6.reuse, 9.9999999747524270788e-07 ;
/*01d0*/   FFMA R11, R11, R6.reuse, 9.9999999747524270788e-07 ;
/*01e0*/   FFMA R13, R13, R6.reuse, 9.9999999747524270788e-07 ;
/*01f0*/   FFMA R15, R15, R6.reuse, 9.9999999747524270788e-07 ;
```

Instruction census of the whole function: **256 `FFMA`, 7 `IADD3`, 2 `ISETP`,
3 `BRA`, 0 `LDL`/`STL`.** Four accumulators `R9 R11 R13 R15` round-robin.

**SASS proof, `C = 1`:** also 256 `FFMA`, all writing and reading `R9`:

```
/*0150*/   FFMA R9, R9, R6, 9.9999999747524270788e-07 ;
/*0160*/   FFMA R9, R9, R6, 9.9999999747524270788e-07 ;
/*0170*/   FFMA R9, R9, R6, 9.9999999747524270788e-07 ;
```

Same instruction count, same register count, 3.8× apart in time.

**Common wrong approaches.**

| What you write | What happens |
|---|---|
| one accumulator updated `C` times per step | that *is* `C = 1`; Part A reports 4.05 cyc/FFMA for every `C` and the check fails with "the four chains were merged" |
| `for (t) { for (i < C) a[i] = fmaf(...); }` — the natural formulation | compiles, measures, and **lies**. Measured: `ptxas` unrolls the `C = 2` instantiation by 1 and every other instantiation by 3, so `C = 2` has 64 `FFMA` between branches where `C = 4` has 192. `C = 2` then comes out ~25% low on the whole table for a reason that has nothing to do with ILP. (Module 11 recorded an unexplained `C = 2` anomaly in its MLP sweep; this is the same family of artefact.) |
| `float a[C]` with a runtime loop bound | `a` spills to local memory, `-Xptxas -v` shows a stack frame, and the "latency" you measure is DRAM |
| `a[i] += b` instead of `fmaf` | fine, but FADD and FFMA have the same 4-cycle latency on this part, so nothing changes; use whichever |
| not storing `s` | the whole loop is dead code; `cycles` comes back ~30 and the file's own `cyc[0] < 0.05` guard fires |

## TODO 2 — the launch geometry

```cpp
static void setOccupancy(int warpsPerScheduler)
{
    g_threads = 128;                    // 4 warps = one per scheduler
    g_blocks  = warpsPerScheduler;      // blocks per SM
}
```

An Ada SM has four warp schedulers and hands a block's warps to them in linear
warp order, so a 128-thread block contributes exactly one warp to each. `B`
blocks per SM is therefore `B` warps per scheduler, and a grid of `B × 40`
blocks is exactly one wave with `B` blocks on every SM — verified directly with
a `%smid` histogram, which shows exactly `B` on all 40 SMs for `B = 1..12`.

**Why the file warns you off "any pair with the right product".** Two measured
reasons:

1. A block whose warp count is not a multiple of four loads the four schedulers
   unequally and the busiest one sets the time. A 160-thread block is 5 warps →
   2/1/1/1; four such blocks give 8/4/4/4, and the kernel runs at `20/32 =
   0.625` of the ceiling. Measured: **0.564 of the issue ceiling against 0.865
   for the balanced control.**
2. There is a *reproducible* sawtooth in blocks-per-SM that the eligible/stalled
   model does not predict. See the lesson §8: at ILP 1, blocks/SM of
   5, 6, 7, 9, 10, 11 land on `W/(4⌈W/4⌉)` of the ceiling, and the same warp
   count delivered as one large block does not. The sweep in this exercise uses
   `{1, 2, 4, 8, 12}` for exactly that reason.

## TODO 3 — `PRED_LATENCY_CYCLES = 4`

Measured 4.051 cycles/FFMA on a single warp with one chain. The 0.051 is the
loop's own `IADD3`/`ISETP`/`BRA` leaking out of the dependency shadow once per
256 FFMAs.

Answering 4 is not lucky: the issue rate is one instruction per scheduler per
clock and the table's floor is 1.067 cyc/FFMA, so the ratio `4.051/1.067 = 3.80`
is the concurrency requirement, and a short fixed-point FP32 pipeline on
Volta-through-Ada has a 4-cycle result latency.

## TODO 4 — `PRED_ILP_GAIN_LOW = 3.5`, `PRED_ILP_GAIN_HIGH = 1.0`

At 1 warp per scheduler a single chain supplies 1 of the 4 instructions in
flight the pipe needs, so ILP should buy up to 4×, capped by the issue port and
by the loop's own instructions: predict 3.5, measured **3.64–3.76×**.

At 12 warps per scheduler the scheduler already has 12 independent instructions
— 3× the requirement — so ILP has nothing to buy: predict 1.0, measured
**1.01×**.

**This pair of numbers is the module.** The same source change is worth 3.7×
and 1.00× depending on one thing you did not change.

## TODO 5 — the model

```cpp
static int predNeededWarps(int C)
{
    int need = (PRED_LATENCY_CYCLES + C - 1) / C;
    return need < 1 ? 1 : need;
}
```

Little's Law. The scheduler needs `latency × throughput = 4 × 1 = 4`
independent instructions in flight. Each warp supplies `C`. So `⌈4/C⌉` warps,
never fewer than one. Measured against the 85%-of-peak crossing:

| ILP | predicted warps/sched | measured | |
|---|---|---|---|
| 1 | 4 | 4 | ok |
| 2 | 2 | 2 | ok |
| 4 | 1 | 1 | ok |
| 8 | 1 | 1 | ok |
| 16 | 1 | 1 | ok |

5 of 5, exactly, not within a factor of two.

A model fitted to the table instead (e.g. a lookup) scores the same and teaches
nothing. The point of deriving it is that the same derivation, with the DRAM
latency and the DRAM throughput substituted in, gives Exercise 3's answer.

## Synchronization / memory reasoning

There is no shared memory, no barrier and no inter-thread communication in this
exercise at all, and that is deliberate: a barrier would introduce a second
stall reason and the measurement would stop being about one. The only memory
operation is the final store, one per thread.

## Performance reasoning

Full output, one run (RTX 3500 Ada, CUDA 13.2):

```
    ILP         cycles     cyc/FFMA     vs ILP=1
      1        4148092        4.051        1.00x
      2        2104092        2.055        1.97x
      3        1420094        1.387        2.92x
      4        1092097        1.067        3.80x
      6        1076097        1.051        3.85x
      8        1092097        1.067        3.80x

   measured dependent-FFMA latency   L = 4.05 cycles
   measured saturated issue interval T = 1.07 cycles
   Little's Law concurrency L/T        = 3.80 independent FFMAs

   GFLOP/s
   warps/sch     ILP=1      ILP=2      ILP=4      ILP=8      ILP=16    ILP 1->8
   1               5388      10586      20184      20255      20317       3.76x
   2              10719      20860      21022      21012      21003       1.96x
   4              20898      20917      21147      21041      21041       1.01x
   8              20907      20729      21157      21118      21118       1.01x
   12             20870      20785      21176      21099      21128       1.01x

   % of best cell (21176 GFLOP/s)
   warps/sch     ILP=1      ILP=2      ILP=4      ILP=8      ILP=16
   1                25%        50%        95%        96%        96%
   2                51%        99%        99%        99%        99%
   4                99%        99%       100%        99%        99%
   8                99%        98%       100%       100%       100%
   12               99%        98%       100%       100%       100%
```

Absolute GFLOP/s moved 20200–21200 across runs with thermal state; the ratios
within a sweep reproduced to about 1%. The `ILP 1->8` column and the `%` table
are the quantities to trust.

Two sanity checks against hardware bounds (spec §12 rule 13): 21,176 GFLOP/s on
160 schedulers × 32 lanes × 2 FLOP implies a clock of 2.07 GHz, inside the
0.49–2.04 GHz range this part is documented to use (and at the very top of it,
which is what a pure-FFMA kernel with no memory traffic should produce). And
1.067 cycles per FFMA per warp cannot go below 1.0, which it does not.

## Expected output

`SCORE: 10/10`, `OVERALL: PASS`. Reproduced on three separate runs with
cool-downs between them.

## The result that matters

Two numbers — a 4-cycle latency and a 1-instruction-per-cycle issue rate —
predict a 25-cell performance surface that spans 4× in throughput, and they do
it through one multiplication. Occupancy and ILP are not two tuning knobs with
independent effects; they are two ways of supplying the single quantity
`latency × throughput`, and the only reason to prefer one over the other is
which one you can afford. **Variation to try:** change `BODY` from 256 to 16.
The loop's three bookkeeping instructions are then 1 in 19 rather than 1 in 259
and the whole table should lose throughput roughly uniformly — predict how
much before you run it. Then try `BODY = 2048` and ask whether the `ILP=16`
column degrades *differentially*; if it does, you have constructed the
`no_instruction` stall (the unrolled body no longer fits the instruction
cache), which is the one row of the lesson's taxonomy this module does not
build for you. This module did not measure either variant; they are yours.
