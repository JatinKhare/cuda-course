# Module 20 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe

nvcc -arch=sm_89 -O3 -cubin -o x2.cubin exercise02_solution.cu
cuobjdump -sass x2.cubin
```

---

## TODO 1 — the classification

```cpp
static const int diagnosis[4] = { 1, 2, 3, 4 };
```

| | kernel | diagnosis | the evidence that settles it |
|---|---|---|---|
| A | `o[i] = 2x[i] + y[i]`, `#pragma unroll 1`, 1 block/SM | **1 long scoreboard** | occupancy helps (2.24× from 1 to 12 blocks/SM) *and* the kernel is far from the bus at its own budget |
| B | `o = x + y`, `float4`, 8 blocks/SM | **2 bandwidth-bound** | occupancy does not help (1.01×) and it is at **87–89% of the 432 GB/s pin peak** |
| C | 262,144-long accumulator chain, 5120 threads | **3 execution dependency** | thread count cannot change; block *shape* changes nothing; **5236 GFLOP/s against a ~20,000 GFLOP/s ceiling** = 26%, which is Example 1's top-left cell to within a percent |
| D | 64 IEEE divides per element, 12 blocks/SM | **4 not-selected / issue** | occupancy helps a lot up to 4 blocks/SM and then stops (2.413 → 2.177 ms, 1.11×), traffic is **61.6 GB/s** — 14% of the bus — and the function is 440 SASS instructions long |

**The pairing is the exercise.** A and B both respond to the "is it memory?"
question with yes, and C and D both respond with no. Separating A from B needs
the % of peak column; separating C from D needs you to look at the code and
notice that C's inner loop has a loop-carried dependence and D's does not.

The classic wrong answer is to call D "latency-bound" because it is slow and
arithmetic. It is not: at 12 warps per scheduler there are three times as many
eligible warps as issue slots, and the 64 divides are mutually independent so
the ILP is already enormous. D is slow because it executes a quarter of a
million instructions per element-group, and the only cure is to execute fewer.

## TODO 2 — kernel A

```cpp
#define AMLP 8
__global__ void kA_fast(const float *__restrict__ x, const float *__restrict__ y,
                        float *o, unsigned n)
{
    unsigned stride = gridDim.x * blockDim.x;
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    for (; i + (AMLP-1)*stride < n; i += AMLP*stride) {
        float xv[AMLP], yv[AMLP];
        #pragma unroll
        for (int c = 0; c < AMLP; ++c) { xv[c] = x[i + c*stride]; yv[c] = y[i + c*stride]; }
        #pragma unroll
        for (int c = 0; c < AMLP; ++c) o[i + c*stride] = fmaf(2.0f, xv[c], yv[c]);
    }
    for (; i < n; i += stride) o[i] = fmaf(2.0f, x[i], y[i]);   // tail, exactly once
}
```

**Why it is correct.** The original's SASS loop is exactly:

```
/*0070*/   MOV R7, 0x4 ;
/*0080*/   IMAD.WIDE.U32 R2, R0, R7, c[0x0][0x160] ;
/*0090*/   IMAD.WIDE.U32 R4, R0.reuse, R7.reuse, c[0x0][0x168] ;
/*00a0*/   LDG.E.CONSTANT R3, [R2.64] ;
/*00b0*/   LDG.E.CONSTANT R4, [R4.64] ;
/*00c0*/   IMAD.WIDE.U32 R6, R0, R7, c[0x0][0x170] ;
/*00f0*/   ISETP.GE.U32.AND P0, PT, R0, c[0x0][0x178], PT ;
/*0100*/   FFMA R9, R3, 2, R4 ;
/*0110*/   STG.E [R6.64], R9 ;
/*0120*/   @!P0 BRA 0x70 ;
```

Two `LDG`s and one `STG` per iteration, and the `FFMA` at `0x100` consumes both
loads, so the loop cannot advance until they return. `#pragma unroll 1` is what
pins it there: without it `ptxas` would issue the next iteration's `LDG`s
before this iteration's `FFMA`. The budget is one block per SM = 4 warps per
scheduler, each holding 2 outstanding loads, so the supplied concurrency is 8
warp-loads per scheduler. Example 2's knee is 4 — so why is it slow?

Because those are **4-byte-per-lane** loads. 8 warp-loads × 128 B = 1024 B per
scheduler, where Example 2's knee of "4 warp-loads" was 4 × 128 B *float4*
loads = 2048 B. Count bytes, not requests. The fix multiplies by 4 (8 loads
each instead of 2) and gets **2.07–2.31×**, landing on exactly kernel B's
time — which is the right stopping condition, because B is this access
pattern's measured ceiling.

The fixed SASS has **18 `LDG` and 9 `STG`** in the function against the
original's 2 and 1.

**Common wrong approaches.**

- *Vectorising with `float4` instead of hoisting.* Works, and is in fact a
  better answer (fewer instructions as well as more bytes in flight) — but `n`
  is `1 << 24`, so you must still handle the case where the grid-stride loop's
  vector body does not cover the array. It does here, which makes it a trap for
  the next person who changes `NELEM`.
- *Dropping the tail loop.* The validator prefills `o` with zeros and counts
  unwritten elements. `NELEM = 2^24 = 16,777,216` and the hoisted body advances
  by `8 × 5120 = 40,960` per pass; `16777216 / 40960 = 409.6`, so the body makes
  409 full passes and leaves **24,576 elements** unwritten. Without the tail the
  validator reports unwritten outputs and `FAIL`.
- *Writing the tail inside the main loop* with `if (i < n)` — the same trap
  Module 11 Exercise 1 planted. Here the operation is idempotent so it would
  not corrupt the answer, but it reintroduces a predicated store in the hot
  loop.
- *Raising occupancy.* Not available: the launch is the budget. If you try it
  anyway, evidence 1 tells you it would have worked (2.24×) — which is the
  lesson, not a defect: **A has two valid fixes and you were only given one.**

## TODO 3 — kernel C

```cpp
__global__ void kC_fast(const float *__restrict__ x, float *o, int K, unsigned n)
{
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i], w = fmaf(x[i], 0.5f, 1.0e-7f);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int t = 0; t < K / CSTEP; ++t) {
        #pragma unroll
        for (int u = 0; u < CSTEP/4; ++u) {
            s0 = fmaf(v, w, s0);
            s1 = fmaf(v, w, s1);
            s2 = fmaf(v, w, s2);
            s3 = fmaf(v, w, s3);
        }
    }
    o[i] = (s0 + s1) + (s2 + s3);
}
```

Four partial accumulators. The original is one chain at 4.05 cycles per FFMA;
four chains run at the issue rate. Measured **3.25×** against a theoretical
3.8× — the shortfall is the loop's own three instructions per `CSTEP = 16`
FFMAs, which sit in the dependency shadow of the slow version and do not in the
fast one.

**Why both kernels keep `CSTEP` FFMAs between branches.** Without that, `ptxas`
unrolls the two versions differently and some of the measured ratio is a
difference in loop overhead rather than in ILP. Both functions contain exactly
**81 `FFMA`, 6 `ISETP`, 6 `BRA`, 4 `IADD3`** — byte-for-byte comparable
instruction mixes, differing only in which registers the FFMAs name.

**The numerical consequence, and why the validator is built the way it is.**
Four accumulators is a *reassociation* of a floating-point sum. The answer
changes. The validator therefore uses a double-precision reference and the
Module 16 rule: threshold `err / (γ_K · S) ≤ 1` with
`γ_K = K·u/(1 − K·u)` and `S` the accumulated magnitude. Measured ratio
**0.058** — comfortably inside, and the number worth noticing is that the
*absolute* relative error against the double reference is about 1e-3, which the
house default tolerance `1e-5 · max(1,|result|)` would have rejected. A
correct kernel failing a wrongly-scaled tolerance is exactly what Module 16
warned about.

**A measured null result you should know about.** An earlier draft of this
kernel computed a per-iteration coefficient:

```cpp
for (int k = 0; k < K; ++k) s = fmaf(v, 1.0e-7f * (float)(k & 255), s);
```

Four accumulators bought **1.07×** on that version, not 3.3×. The reason is in
the lesson's model: the coefficient computation (`LOP3`, `I2F`, `FMUL`) is
three *independent* instructions per dependent `FFMA`, so the loop already had
ILP 4 — it was already at the `latency × throughput` requirement and there was
nothing left to hide. **The wrong fix did nothing because the right fix had
already been applied by accident.** That is the single most useful thing in
this exercise: before you add ILP, count how much the loop already has.

## TODO 4 — kernel D

```cpp
for (int k = 1; k <= K; ++k) s += __fdividef((float)k, v + (float)k);
```

An IEEE-correct fp32 divide is not an instruction. nvcc emits `MUFU.RCP`, a
Newton refinement in `FFMA`/`FADD`, and an `FCHK` range check that branches to
a slow path. `__fdividef` is `MUFU.RCP` plus `FMUL`, accurate to about 2 ulp,
valid for `|y| < 2^126`.

Measured SASS, whole function:

| | `kD_slow` | `kD_fast` |
|---|---|---|
| total instructions | **440** | **184** |
| `FCHK` | 15 | **0** |
| `MUFU` | 17 | 15 |
| `FFMA` | 87 | 15 |
| `BRA` | 44 | — |

Instruction ratio 2.39×, measured speedup **1.85–2.03×**. The gap is the part
of the loop that did not shrink (`I2FP`, `FADD`, the loads and stores). That
correspondence is the check that the diagnosis was right: on an issue-limited
kernel the speedup should track the instruction count, and it does.

**Common wrong approaches.**

- *More warps.* Evidence 1 shows 4 → 12 blocks/SM is worth 1.11×. The kernel is
  already past the point where warps help.
- *More ILP.* There is nothing to add: the 64 divides are already independent
  and `#pragma unroll 8` already exposes them.
- *`-use_fast_math`.* Would work for this kernel, and Module 11 measured it as a
  **null result** on a bandwidth-bound elementwise kernel — the flag only pays
  where instruction count is the binding constraint, which is precisely the
  diagnosis here. Changing the source instead of the build line makes the
  accuracy trade explicit and local.
- *`__frcp_rn` + `FMUL`.* Correctly rounded reciprocal, then a multiply: fewer
  instructions than the IEEE divide but more than `__fdividef`, and a different
  accuracy point. A legitimate answer; it will clear the 1.40× gate.

## TODO 5 — the predictions

```cpp
static const double predSpeedup[4] = { 2.0, 1.0, 3.2, 2.0 };
```

Measured **2.14 / 0.99 / 3.25 / 2.03**.

C's prediction is the one that is not a guess: it is `latency / throughput`
from Example 1, 4.05/1.07 = 3.8, discounted for the loop's three bookkeeping
instructions per 16 FFMAs → about 3.2.

## Synchronization / memory reasoning

No barriers and no shared memory anywhere in this exercise. Every one of the
four stalls is either a scoreboard wait or an arbitration loss, which is the
point: the taxonomy's `barrier` row needs Module 9's machinery and a different
vehicle, and conflating it with these four is how people end up putting
`__syncthreads()` in places that do not need one.

Kernel A's tail loop is the only correctness-critical piece. Everything else is
performance.

## Performance reasoning

Full output, one run:

```
-- evidence 1: does more occupancy help? (ms, min of 16 sweeps) ------
   kernel             1         2         4        12       1->12      budget
   A              1.172     0.698     0.535     0.524       2.24x           1
   B              0.539     0.519     0.531     0.533       1.01x           8
   C              0.513     0.517     0.513     0.629       0.81x           1
   D              8.195     4.162     2.413     2.177       3.76x          12

-- evidence 2 --------------------------------------------------------
   A at 12 blk/SM :  384.0 GB/s = 88.9% of 432 GB/s pin peak (3N traffic)
   B at 12 blk/SM :  377.6 GB/s = 87.4% of 432 GB/s pin peak (3N traffic)
   C: 262144 FFMAs x 5120 threads -> 5236 GFLOP/s
   D: traffic is 2N = 61.6 GB/s - nowhere near the bus

-- the fixes ---------------------------------------------------------
   k    diagnosis                     orig ms   fixed ms    speedup  predicted
   A    long-scoreboard (global)        1.203      0.561      2.14x      2.00x
   B    bandwidth-bound                 0.573      0.579      0.99x      1.00x
   C    execution dependency            0.515      0.159      3.25x      3.20x
   D    not-selected / issue            2.184      1.073      2.03x      2.00x
```

Note kernel C's `1->12` column reading **0.81×** — the last shape is *slower*.
Those columns are block shapes at a constant 5120 threads, and they are not
equivalent: 32/64/128 threads per block all give 4 warps per SM on all 40 SMs,
i.e. one warp per scheduler, while 256 threads per block is only 20 blocks and
leaves **half the SMs idle** — with 2 warps per scheduler on the 20 that are
busy. Those two changes nearly cancel (half the SMs at twice the issue rate is
the same total issue), which is why the time moves by 1.23× and not by 2×. A
kernel whose thread count is fixed by the problem is also a kernel whose block
size decides how much of the machine it can reach; Module 1's wave arithmetic
reappears as a constraint on the block shape rather than on the grid.

This file also carries Module 15's operating-point guard (spec §12 rule 5b):
after warming it probes kernel B as a ceiling reference and, if that reads
below 330 GB/s, idles 10 s and warms again. The speedup gates are ratios and
survive the power-capped state, but the "87-89% of pin peak" evidence that
identifies B as bandwidth-bound does not — in that state B reads ~300 GB/s and
the reader would classify it wrongly.

Run-to-run spread on the four speedups across three runs with cool-downs:
A 2.07–2.31, B 0.99–1.00, C 2.93–3.25, D 1.85–2.03. The gates (1.70 / 0.90–1.10
/ 2.50 / 1.40) sit in the empty gaps below those bands with 14–30% of margin,
per spec §12 rule 5d.

## Expected output

`SCORE: 10/10`, `OVERALL: PASS`.

## The result that matters

Four kernels, all slow, all slow for different reasons, and each of the four
fixes does **nothing** for the other three. A is cured by concurrency, B cannot
be cured at all, C is cured by breaking a dependency, D is cured by deleting
instructions — and the diagnosis that tells them apart takes two measurements
you can make without a profiler: *does occupancy help*, and *where is it against
a hardware bound*. **Variation to try:** take kernel C's earlier form (compute
`1.0e-7f * (float)(k & 255)` inside the loop) and re-run. The four-accumulator
fix collapses to 1.07× because the coefficient arithmetic was already supplying
the ILP. Then count the independent instructions per dependent one in your own
hot loop before you reach for the fix.
