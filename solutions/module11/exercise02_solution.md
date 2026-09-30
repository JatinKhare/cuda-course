# Module 11 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe

nvcc -arch=sm_89 -O3 -Xptxas -v -cubin -o exercise02.cubin exercise02.cu
```

---

## TODO 1 — the unfused traffic

```cpp
static const int UNFUSED_TRAFFIC_N = 10;
```

| stage | reads | writes | traffic |
|---|---|---|---|
| `t1 = A*x + B*y` | `x`, `y` | `t1` | 3N |
| `m = t1 + C*z` | `t1`, `z` | `m` | 3N |
| `t3 = max(m, 0)` | `m` | `t3` | 2N |
| `d = S*t3 + O` | `t3` | `d` | 2N |
| | | | **10N** |

The assumption that makes this arithmetic valid is stated in the file and it is
not decoration: **the arrays are 128 MB and L2 is 48 MB.** Nothing survives from
one launch to the next, so each intermediate really does round-trip to DRAM. At
N = 2^20 (4 MB per array) the whole working set fits in L2, every stage after
the first hits, and the unfused traffic is 3N total rather than 10N. Check Your
Understanding Q3 is that case.

The usual wrong answer is **8**, from counting stages 3 and 4 as 1N each
("`t3 = max(m,0)` is just a clamp"). Every stage reads an array and writes an
array; the cheapness of the arithmetic has nothing to do with it. That is the
whole thesis of this module.

## TODO 2 — the fused traffic

```cpp
static const int FUSED_TRAFFIC_N = 5;
```

The fused kernel reads `x`, `y`, `z` and writes `m` and `d`: 3 + 2 = **5N**.

**The tempting answer is 4**, from reading "fuse the pipeline" as "produce `d`
from `x, y, z`". The file states, in capitals, that `m` is also a required
output. A downstream consumer reads it. A kernel that keeps `m` only in a
register is faster and wrong, and the harness reports it in a way that names the
mistake:

```
  [  ] chain_fused:   m 33554432 bad, d 0 bad
       (d is right and m is wrong -- you fused away a required output.)
```

This is the general rule, and it is the one worth carrying into Part XIV:
**fusion deletes the memory traffic of values that nobody outside the fused
region needs, and only that traffic.** An intermediate with an external consumer
is not an intermediate; it is an output, and it costs a write no matter how the
kernels are arranged. When someone tells you a fused attention kernel "saves the
attention matrix", check whether anything else needed the attention matrix.

The ladder is worth seeing in full:

| arrangement | materialised | traffic |
|---|---|---|
| 4 kernels | `t1`, `m`, `t3`, `d` | 10N |
| 2 kernels (stage 1, then 2+3+4) | `t1`, `m`, `d` | 7N |
| 1 kernel | `m`, `d` | **5N** |
| 1 kernel, ignoring the `m` requirement | `d` | 4N — **incorrect** |

## TODO 3 — the kernels

```cpp
__global__ void chain_partial(const float* __restrict__ t1, const float* __restrict__ z,
                              float* __restrict__ m, float* __restrict__ d, long long n)
{
    for (GRID_STRIDE(n)) {
        const float mv = t1[i] + P_C*z[i];
        m[i] = mv;                                   // required output
        d[i] = P_S*fmaxf(mv, 0.0f) + P_O;            // t3 never reaches memory
    }
}

__global__ void chain_fused(const float* __restrict__ x, const float* __restrict__ y,
                            const float* __restrict__ z,
                            float* __restrict__ m, float* __restrict__ d, long long n)
{
    for (GRID_STRIDE(n)) {
        const float a1 = P_A*x[i] + P_B*y[i];        // t1 stays in a register
        const float mv = a1 + P_C*z[i];
        m[i] = mv;                                   // required output
        d[i] = P_S*fmaxf(mv, 0.0f) + P_O;            // t3 stays in a register
    }
}
```

Three things to notice.

**`a1` and the `t3` value never get a name in memory.** That is the entire
mechanism. The compiler keeps them in registers because their live ranges do not
escape the loop body. Nothing clever is required and nothing clever should be
attempted — a reader who stages them through shared memory has added a
global→shared store, a barrier and a shared→register load to move a value from a
register to the same register. Module 6's reuse-factor argument says shared
memory pays when `K > 1`; here `K = 1` and the reuse is already happening in the
register file, which is faster and free.

**`m` is read back from a register, not from memory.** `d[i]` is computed from
`mv`, not from `m[i]`. Writing `m[i] = mv; d[i] = P_S*fmaxf(m[i],0)+P_O;` is
also correct — the compiler will forward the store — but it hands the compiler a
load that it has to prove is redundant, and with `__restrict__` absent it could
not. Say what you mean.

**Both loads and both stores per element are independent.** The `x`, `y` and `z`
loads have no dependence on each other, so the compiler issues all three before
consuming any, which is the same memory-level-parallelism argument as
`example02.cu` Part C. This is why the fused kernel's *achieved bandwidth* is
usually at least as good as a single stage's, and sometimes better.

## TODO 4 — reconciling prediction and measurement

```cpp
static double explainSpeedup(double trafficRatio, double bwFused, double bwUnfused)
{
    return trafficRatio * (bwFused / bwUnfused);
}
```

Since `time = traffic / bandwidth`:

```
speedup = t_unfused / t_fused
        = (T_u / B_u) / (T_f / B_f)
        = (T_u / T_f) × (B_f / B_u)
```

The traffic model computes the first factor and **silently assumes the second is
1**. The check is an algebraic identity given the harness's definitions — it will
match to the last digit — and that is deliberate. You are not being asked to
predict a number; you are being asked to locate the hidden assumption. The
useful output is the *value* of `B_f / B_u`, which the harness prints as the
`GB/s` column, and it is not 1.

**Measured on this GPU, and it changes sign with array size.**

| arrays | unfused GB/s | fused GB/s | `B_f / B_u` | predicted | measured |
|---|---|---|---|---|---|
| 128 MB (this exercise, N = 2^25) | 359.7 | 349.1 | 0.971 | 2.000× | **1.941×** |
| 256 MB (the same chain, N = 2^26) | 359.3 | 399.7 | 1.112 | 2.333× | **2.595×** |

At 128 MB the fused kernel falls ~3% short of its traffic prediction; at 256 MB
an eight-input version *beat* its prediction by 11%. Two mechanisms pull in
opposite directions:

- **more MLP per thread helps.** The fused kernel has 3 independent loads per
  element against a stage's 1 or 2, so it fills the outstanding-request
  structure faster.
- **more concurrent DRAM streams hurt.** GDDR6 keeps one activated row per bank.
  Every additional stream is another row that wants to stay open, and past some
  count the controller spends more time on row activations than it saves.

Which one wins depends on how many streams, how large, and what else is
resident. **The traffic model gives you the first-order answer and the sign of
the effect; it does not give you 5% accuracy, and you should not claim it does.**

## TODO 5 — the occupancy question

```cpp
static const int PREDICT_OCCUPANCY_LIMITS = 0;
```

Real `-Xptxas -v` output for this file:

```
ptxas info : Compiling entry function '_Z2s1PKfS0_Pfx' for 'sm_89'
ptxas info : Used 34 registers, used 0 barriers, 384 bytes cmem[0]
ptxas info : Compiling entry function '_Z13chain_partialPKfS0_PfS1_x' for 'sm_89'
ptxas info : Used 40 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info : Compiling entry function '_Z11chain_fusedPKfS0_S0_PfS1_x' for 'sm_89'
ptxas info : Used 40 registers, used 0 barriers, 400 bytes cmem[0]
```

and the harness prints the consequence:

```
  registers/thread : s1 = 34, chain_fused = 40
  max resident warps/SM at 256 threads/block : 48 vs 48 (of 48)
```

40 registers per thread → 40 × 32 = 1,280 registers per warp, rounded up to the
256-register allocation granularity = 1,280 → 65,536 / 1,280 = 51 warps, capped
by the hardware's 48-warp / 1,536-thread limit. **Fusion costs nothing here.**

The interesting part is what happens when you push much further. Compiled
separately, a templated `M`-input fused kernel gives:

| fused inputs | registers | spills | max warps/SM (256 thr/block) |
|---|---|---|---|
| 2 | 34 | 0 | 48 |
| 4 | 40 | 0 | 48 |
| 8 | 40 | 0 | 48 |
| 12 | 40 | 0 | 48 |
| 16 | 48 | 0 | 40 |
| 24 | 64 | 0 | 32 |
| 32 | 80 | 0 | 24 |

Register pressure is real and it does cut occupancy — at 32 fused inputs you are
down to 50% occupancy. **And it still does not matter**, because `example02.cu`
Part A measured this GPU's streaming kernels saturating at roughly 8 resident
warps per SM. 24 warps is three times what the bus needs. There is an order of
magnitude of margin between the occupancy fusion costs and the occupancy a
bandwidth-bound kernel uses.

**Report the honest conclusion:** the standard warning that "fusing too much
kills occupancy" is good advice for compute-bound kernels, where you need
enough warps to hide arithmetic latency and to keep four warp schedulers
issuing. For elementwise streaming on this part it is a non-event, and a reader
who predicted `1` from the folklore learns something more valuable than a reader
who predicted `0` from the arithmetic.

## The fusion-depth sweep

The harness sweeps the number of fused input streams and reports achieved
bandwidth, so the exercise can reveal a cliff if one exists:

```
=== how deep does fusion keep paying? (informational) ===
  inputs    traffic         ms         GB/s       vs M=2
       2         3N     1.1696        344.3        1.000
       3         4N     1.4773        363.4        1.056
       4         5N     1.8555        361.7        1.051
       5         6N     2.2030        365.6        1.062
       6         7N     2.5673        366.0        1.063
```

**There is no cliff on this hardware up to six input streams.** Achieved
bandwidth rises by about 6% from two streams to three and then flattens — more
MLP up to the point where the bus is saturated, and no row-thrashing penalty
after it.

Two methodological notes, both of which cost real time to find:

1. **An early version of this sweep showed the bandwidth column climbing to
   413 GB/s at eight inputs, which is 96% of peak.** It was wrong. The input
   pointer list repeated one array, so the eighth "stream" was a second read of
   a line that was already in cache — seven real streams counted as eight,
   inflating the number by exactly 8/7 = 1.14. The shipped version uses six
   distinct arrays. If a bandwidth measurement exceeds what the rest of your
   experiments produce, look for a repeated pointer before you believe it.
2. **One run in six produced 272 GB/s for `M ≥ 4` and 369 for `M ≤ 3`,** a
   spurious 0.74× "cliff". `nvidia-smi` showed the memory P-state at 7001 MHz
   with throttle reason `0x1` during that run; the drop is the P-state, not the
   stream count. The sweep count was raised from 4 to 6 so that every
   configuration leads a sweep at least once (spec §12.9 — with fewer sweeps
   than configurations, the first configuration keeps an unfairly early and cool
   sample), and it has not recurred. **A "cliff" that appears in one run of six
   is a clock event until proven otherwise.**

## Synchronization / memory reasoning

None required, and this is worth stating explicitly because fusion *looks* like
it should need it. In the unfused pipeline, stage 2 reads what stage 1 wrote —
a genuine cross-block dependence, satisfied by the **kernel boundary**, which is
a device-wide synchronization point that also flushes L1 (Module 4). Fusing the
stages removes the dependence entirely rather than replacing the barrier with
something cheaper: after fusion, the value flows from a register to a register
inside one thread, and there is no inter-thread communication left to order.

That is why elementwise chains fuse trivially and reductions do not. Module 12's
reduction has a real cross-thread dependence that survives fusion and has to be
paid for with a barrier, a shuffle, or an extra kernel launch.

## Performance reasoning

```
=== performance ===
  version                           ms   traffic       GB/s    %ofpeak
  unfused (4 kernels)           3.7314       10N      359.7      83.3%
  partial (2 kernels)           2.6606        7N      353.1      81.7%
  fused   (1 kernel)            1.9222        5N      349.1      80.8%

  your traffic prediction : 2.000x
  measured                : 1.941x
  gap                     : -2.9%
```

All three versions run the bus at 81–83% of nominal peak. **They are all at
their respective ceilings; the only thing that changed is how much work there
was to do.** That is the cleanest possible demonstration of the module's thesis:
for a memory-bound kernel, optimization is traffic reduction, and everything
else is rounding.

Note also that the 4 → 2 → 1 kernel ladder tracks the 10 → 7 → 5 traffic ladder
almost exactly (3.73 / 2.66 / 1.92 ms against 10 / 7 / 5 = 3.73 / 2.61 / 1.87
scaled). Partial fusion is not a consolation prize; it captures its share of the
traffic saving in proportion.

## Expected output

Reproduce the blocks above. A correct submission has:

- `TODO 1 unfused traffic = 10N` and `TODO 2 fused traffic = 5N`;
- both kernels reporting `m 0 bad, d 0 bad` — a nonzero `m` with a zero `d` is
  the "fused away a required output" error and nothing else;
- `TODO 4 explained speedup` matching `measured` (it is an identity; if it does
  not match, your formula is wrong, not the machine);
- `TODO 5 occupancy prediction = 0, truth = 0`;
- `Score: 7/7`.

Absolute ms drift with the clock: across runs the unfused chain measured
3.73–4.27 ms and the fused kernel 1.92–2.16 ms. The measured speedup stayed in
1.94–1.97× every time.

## The result that matters

Fusing an elementwise chain is not an optimization you benchmark your way into —
you can compute the answer before you write the code, because for a memory-bound
kernel time is traffic over bandwidth and fusion changes only the numerator. The
skill being trained is the bookkeeping: which values have external consumers and
therefore must be written, and which are private to the chain and therefore cost
nothing but a register. Everything the rest of this course does to transformer
inference kernels in Parts XIV and XV is that same bookkeeping at larger scale.

**Try this, and be ready to be wrong.** Drop `N` to `1 << 20` — 4 MB per array,
the entire 7-array working set inside the 48 MB L2 — and re-run. The obvious
prediction is that the speedup collapses toward 1, because the intermediates
were never reaching DRAM and so there was no traffic to save. Measured:

```
  version                           ms   traffic       GB/s    %ofpeak
  unfused (4 kernels)           0.0403       10N     1041.9     241.2%
  partial (2 kernels)           0.0202        7N     1451.2     335.9%
  fused   (1 kernel)            0.0122        5N     1724.9     399.3%

  your traffic prediction : 2.000x
  measured                : 3.311x
  gap                     : +65.6%
```

The speedup goes **up**, to 3.31×. Two things happened at once. The >100%-of-peak
figures say what the first one is: the working set is L2-resident, so these are
L2 bandwidth numbers, not DRAM numbers, and the *traffic* saving is worth much
less than 2× because L2 is not the bottleneck. And the second: 0.0403 ms across
four launches is about 10 µs per launch, which on this Windows/WDDM driver is
roughly the launch overhead itself. **In this regime fusion is not saving
bandwidth, it is deleting three kernel launches** — and that is a completely
different argument for the same code change, with a completely different scaling
law. Module 28 (CUDA graphs) attacks the same cost from the other side.

The lesson is not "the traffic model is unreliable". It is that a model has a
domain, the domain here is "working set ≫ L2", and stepping outside it changes
which term dominates. Ask yourself which of your production kernels are in
which regime before you quote a predicted speedup.
