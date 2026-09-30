# Module 11 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe

nvcc -arch=sm_89 -O3 -cubin -o exercise03.cubin exercise03.cu
cuobjdump -sass exercise03.cubin > sass.txt
```

---

## TODO 1 — the activation's traffic

```cpp
static const int SILU_BYTES_PER_ELEMENT = 12;
```

`out[i] = (x[i] * sigmoid(x[i])) * g[i]`: read `x` (4), read `g` (4), write
`out` (4) = **12 B per element**. `x` is named twice in the source and fetched
once — the same rule as Exercise 1.

## TODO 2 — is it memory bound?

```cpp
static const int SILU_IS_MEMORY_BOUND = 1;
```

Do the arithmetic rather than asserting it. 12 B per element at the ~376 GB/s
this GPU streams is

```
12 B / 376e9 B/s = 32 ps per element
```

In 32 ps, 40 SMs × 128 FP32 lanes at ~1.9 GHz retire

```
40 × 128 × 1.9e9 × 32e-12 ≈ 311 lane-operations
```

which, spread over the elements that are in flight simultaneously, is a budget
of **several hundred FP32 instructions per element**. `example02.cu` Part D
measures the same thing empirically from the other end: 64 chained `FFMA`s per
element does not move the runtime at all.

One `expf` plus one IEEE divide is a few tens of instructions. Two orders of
magnitude inside the budget. Memory bound, and not marginally.

This is the estimate you should be able to produce in thirty seconds for any
elementwise kernel, and it is why the next two TODOs have the answers they do.

## TODO 3 — the intrinsic version

```cpp
__global__ void silu_fast(const float* __restrict__ x, const float* __restrict__ g,
                          float* __restrict__ out, long long n)
{
    for (GRID_STRIDE(n)) {
        const float v = x[i];
        out[i] = (v * __frcp_rn(1.0f + __expf(-v))) * g[i];
    }
}
```

Two substitutions, and the second one is the one people forget.

- **`expf` → `__expf`.** `__expf(v)` compiles to `FMUL` by `log2(e)` followed by
  one `MUFU.EX2`. The accurate `expf` adds range reduction and a polynomial.
- **`/` → `__frcp_rn` (or `__fdividef`).** An IEEE-754 single-precision divide is
  *not* one instruction on this hardware. In the SASS below, `silu_gate` contains
  five `FCHK` instructions — one per element of the unrolled loop — which is the
  divide's special-case check (zero, infinity, denormal, NaN), plus a
  Newton–Raphson refinement sequence and a slow-path branch. Replacing `expf`
  and leaving the divide alone captures maybe half the available saving, and on
  a memory-bound kernel captures half of nothing.

Real SASS instruction census, one grid-stride iteration (5 elements) of each:

| | total instructions | `MUFU.EX2` | `MUFU.RCP` | `FCHK` | `FFMA` | `FMUL` | registers |
|---|---|---|---|---|---|---|---|
| `silu_gate` | 448 | 5 | 8 | **5** | 49 | 5 | 33 |
| `silu_fast` | 392 | 5 | 11 | **0** | 15 | 25 | 32 |

`FCHK` disappearing is the divide's special-case path disappearing. The
`MUFU.RCP` count goes *up* (the intrinsic reciprocal is a `MUFU.RCP` plus a
refinement step), while `FFMA` drops from 49 to 15.

**Accuracy.** Measured max relative error over all 33.5 M elements, against a
double-precision host reference:

```
  [ok] silu_gate  max relative error = 2.394e-07 (sanity, budget 1e-5)
  [ok] silu_fast  max relative error = 5.330e-07 (budget 2e-03)
```

`5.33e-7` is about 4.5 ULP of `float` — three and a half orders of magnitude
inside the 2e-3 budget. The design question the TODO asks you to answer is
whether that budget is defensible, and for this workload it plainly is: the
activation sits between two GEMMs whose weights, in the deployment case this
kernel exists for, are quantized to 8 bits. An INT8 weight carries roughly
2⁻⁸ ≈ 4×10⁻³ of relative representational error. Arguing about 5×10⁻⁷ in the
activation is arguing about the seventh decimal place of a number that is only
good to the third.

**What would make the answer different:** an activation inside a training loop
whose gradients are accumulated in fp32 over thousands of steps, where a
systematic bias (not a random error) compounds. `__expf` is monotone and its
error is not a systematic bias, so even there it is usually fine — but "usually"
is the right word, and the way to find out is to run the eval, not to reason
about ULPs.

**A wrong approach worth naming:** `__fdividef(x, y)` is *not* safe for all
inputs. Its documented range is `|y| < 2^126`; outside that it returns 0. Here
`y = 1 + __expf(-v)` with `v ∈ [-6, 6]`, so `y ∈ [1, 404]` and it is fine. Use it
on a denominator you have not bounded and you get silent zeros, which is why the
harness also counts how many outputs are exactly zero.

## TODO 4 — the crossover predictions

```cpp
static const int PREDICT_K_SINF     = 8;
static const int PREDICT_K_FASTSINF = 32;
```

**The reasoning, not the answer.** The budget from TODO 2 is a few hundred
FP32 lane-operations per element, and the harness's crossover threshold is
1.25× the floor, so the crossover is where `K × cost(f)` reaches roughly a
quarter of that budget.

`sinf` on sm_89 is of the order of 30 instructions — a Cody–Waite argument
reduction, a minimax polynomial, and a guarded FP64 fallback for large
arguments. The real SASS:

```
/*01c0*/  FFMA R10, R9, -1.5707962512969970703, R0 ;     <- pi/2, high word
/*01d0*/  FFMA R10, R9, -7.5497894158615963534e-08, R10 ;   pi/2, middle
/*01e0*/  FFMA R10, R9, -5.3903029534742383927e-15, R10 ;   pi/2, low
...
/*0820*/  I2F.F64.S64 R6, R8 ;                            <- FP64 fallback path
/*0860*/  DMUL R6, R6, c[0x2][0x0] ;
/*0870*/  F2F.F32.F64 R6, R6 ;
```

Three chained `FFMA`s against three successively smaller pieces of π/2 is
textbook Cody–Waite: it keeps the reduced argument accurate to more than
single precision by splitting the constant across several floats. The `DMUL`
path is the Payne–Hanek fallback for arguments too large for that to work, and
**Ada runs FP64 at 1/64 rate** — a strong reason to keep `sinf` arguments small
if you must use it at all. (In this sweep the values stay in [-1, 1] after the
first application, so the fallback is predicated off; it still costs
instruction-cache footprint and a branch.)

A few hundred over ~30 is ~10, and 8 is the nearest sweep point. Measured: 8.

`__sinf` is two instructions:

```
/*0550*/  FMUL.RZ R4, R4, 0.15915493667125701904 ;   <- x * 1/(2 pi)
/*0560*/  MUFU.SIN R11, R4 ;                          <- one SFU operation
```

The naive conclusion is a 15× shift in the crossover. The measured shift is 4×.
The reason is that `MUFU` does not execute on the FP32 lanes: each Ada
processing block has **32 FP32 lanes and 4 SFUs**, so a warp's `MUFU`
instruction occupies the SFU for 8 cycles where an `FFMA` occupies the FP32
lanes for 1. `__sinf` is worth roughly 4–8 FFMAs of throughput, not 2. Hence
~4× and not ~15×, and the prediction 8 × 4 = 32. Measured: 32.

Both are scored to within a factor of two because the estimate is only good to
about that. Getting the *ordering* and the *rough ratio* right is the skill;
the exact sweep point is an artifact of the 1.25× threshold.

Measured:

```
  memory floor (FFMA control, K=1)  : 0.7986 ms
  crossover := smallest K with t(K) > 1.25 x floor

  K =             1        2        4        8       16       32       64
  FFMA       0.7986   0.7968   0.7961   0.7966   0.7976   0.7954   0.8416
  sinf       0.8184   0.8463   0.8754   1.1982   2.4061   4.2186   7.6806
  __sinf     0.7963   0.7989   0.7991   0.8043   0.8333   1.0723   2.2019

  x floor
  FFMA         1.00     1.00     1.00     1.00     1.00     1.00     1.05
  sinf         1.02     1.06     1.10     1.50     3.01     5.28     9.62
  __sinf       1.00     1.00     1.00     1.01     1.04     1.34     2.76

  measured crossover: sinf K = 8, __sinf K = 32
```

Read the `FFMA` row across: **64 chained FP32 fused multiply-adds per element
cost 5%**, and the first 32 cost nothing measurable. That row is the
memory-bound plateau of Module 21's roofline drawn directly from measurement,
and it is the single most useful number in this exercise. Whenever you are
tempted to "optimize the math" in an elementwise kernel, ask how your change
compares to 32 free FFMAs.

Read the `sinf` row and note that its slope past the crossover is linear in `K`,
as it must be once memory is fully hidden: 4.22 ms at K=32 and 7.68 at K=64.
Extrapolating backwards, one `sinf` costs about 0.115 ms of pure compute for
33.5 M elements, against a 0.80 ms memory floor — which is the same "7 sinfs
fit inside the floor" statement as the crossover, arrived at without the
threshold.

## TODO 5 — will `-use_fast_math` help?

```cpp
static const int FAST_MATH_HELPS_SILU = 0;
```

```
  measured streaming ceiling        : 337.3 GB/s (78% of 432)
  your traffic floor at that rate   : 1.1938 ms
  version                            ms       GB/s    x floor
  silu_gate (expf, /)            1.1643      345.8      0.975
  silu_fast (yours)              1.1664      345.2      0.977
  ratio accurate/fast : 0.998x
```

**No. Measured 0.998×, i.e. nothing, and slightly on the wrong side of
nothing.** The accurate kernel does ~90 instructions per element against the
intrinsic kernel's ~78, and the difference disappears entirely into DRAM's
shadow, exactly as TODO 2 predicted.

This is a folklore-contradicting result and it is worth stating plainly:
**`-use_fast_math` does not speed up bandwidth-bound elementwise kernels.** It
speeds up kernels whose arithmetic is on the critical path. Applying it to a
streaming activation costs you accuracy, costs you denormal handling
(`-use_fast_math` implies `-ftz=true`), and buys nothing. The correct time to
reach for it is after you have measured that your kernel is *not* at its traffic
floor and have established that the arithmetic is why.

**Verification that the file's comparison is the right proxy for the flag.**
`-use_fast_math` rewrites `sinf` to `__sinf` and `expf` to `__expf` in the
compiler front end. Confirmed directly on this machine — compiling a plain
`sinf` kernel with the flag produces exactly the intrinsic sequence:

```
> nvcc -arch=sm_89 -O3 -use_fast_math -cubin -o fm.cubin fm.cu
> cuobjdump -sass fm.cubin | findstr MUFU.SIN
        /*0560*/   MUFU.SIN R11, R4 ;
        /*06a0*/   MUFU.SIN R17, R17 ;
        /*0710*/   MUFU.SIN R19, R19 ;
        /*07a0*/   MUFU.SIN R17, R18 ;

> nvcc -arch=sm_89 -O3 -cubin -o fmn.cubin fm.cu          (no flag)
> cuobjdump -sass fmn.cubin | findstr /C:"MUFU.SIN" /C:"DMUL"
        /*0860*/   DMUL R6, R6, c[0x2][0x0] ;
```

Zero `MUFU.SIN` and one `DMUL` without the flag; four `MUFU.SIN` and no `DMUL`
with it, for the identical source line `o[i] = sinf(x[i])`.

The reason the exercise measures the two forms in **one binary** rather than
building twice is that the flag changes several things at once (FTZ, contraction,
`__fdividef`, `prec-sqrt`), and a two-binary comparison cannot attribute the
result to any of them. One binary, two kernels, one thermal state, one rotated
sweep.

**There is no preprocessor macro for it.** Neither `__CUDA_FAST_MATH__` nor
`__FAST_MATH__` is defined by nvcc 13.2 under `-use_fast_math`, so a file cannot
detect at compile time whether it was built with the flag. If you need a kernel
to behave differently under fast math, define your own `-D` and pass both.

## Synchronization / memory reasoning

None. Every kernel here is a pure elementwise map with disjoint per-thread
output. The only ordering in the file is the kernel boundary between the timed
launches and the validation pass, and `cudaDeviceSynchronize()` supplies it.

Worth noting for contrast: the K-sweep kernel's inner loop is a **serial
dependence chain** (`v = f(v)` K times) by construction. That is the point — it
is how the experiment makes arithmetic cost show up as latency that cannot be
hidden by ILP within the thread. If the K applications were independent, the
compiler would pipeline them and the crossover would move.

## Performance reasoning

The whole exercise is one measurement made twice, at two scales:

- **at K = 1**, one transcendental per element is free, so `silu_gate` and
  `silu_fast` are indistinguishable and both sit at 0.975× their traffic floor;
- **at K = 8 or 32**, the same transcendental is the entire cost.

Nothing about the hardware changed between those two statements. What changed is
the ratio of arithmetic to bytes, i.e. the **arithmetic intensity**, and the
crossover point is where that ratio crosses the machine's balance. Module 21
draws this as a roofline and gives the balance point a name; this exercise
measures it.

Note that the SiLU kernel reports 345.8 GB/s against a measured streaming
ceiling of 337.3 GB/s — 102.5%. The yardstick `stream_ref` copies between two
arrays (2N); the SiLU kernel reads two arrays and writes a third (3N) and
therefore keeps three DRAM streams in flight instead of two, which on this part
is slightly *better* for the memory controller. The yardstick is a yardstick, not
a physical limit; the physical limit is 432 GB/s and the kernel is at 80% of it.

## Expected output

```
=== validation (untimed second pass) ===
  [ok] TODO 1 SiLU bytes/element = 12
  [ok] TODO 2 memory bound = 1
  [ok] silu_gate  max relative error = 2.394e-07 (sanity, budget 1e-5)
  [ok] silu_fast  max relative error = 5.330e-07 (budget 2e-03)
  [ok] silu_fast is not more than 5% slower (1.1664 vs 1.1643 ms)
...
  measured crossover: sinf K = 8, __sinf K = 32
  [ok] TODO 4a sinf   predicted 8, measured 8 (factor of 2 allowed)
  [ok] TODO 4b __sinf predicted 32, measured 32 (factor of 2 allowed)

Score: 7/7
OVERALL: PASS
```

Across runs the crossovers were stable at 8 and 32 (the `sinf` K=8 entry sits at
1.42–1.50× the floor, comfortably clear of the 1.25× threshold, and K=4 sits at
1.10, comfortably below). Absolute ms varied with the clock by about 10%; the
`x floor` rows did not.

## The result that matters

"Elementwise means memory bound" is true with a quantitative caveat you can now
state: on this GPU, an elementwise kernel moving 8–12 bytes per element gets
**tens of free FP32 operations and roughly four free SFU operations per
element**, and only past that does arithmetic appear in the runtime. That single
number tells you, without an experiment, that a SiLU is free, that a softmax
denominator is free, that `powf(x, 2.5f)` per element is borderline, and that a
32-tap polynomial filter per element is not. It also tells you that
`-use_fast_math` is an answer to a question you have not yet asked.

**Try this:** replace the K-loop body with `__powf(v, 1.1f)` and with
`powf(v, 1.1f)` and re-run. `powf` is `expf(y*logf(x))` with edge-case handling
and is one of the most expensive functions in the library; predict its crossover
from the ratio of its instruction count to `sinf`'s before you measure, then
check the SASS to see how badly you underestimated the edge-case handling.
