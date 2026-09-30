# Module 16 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
```

Warning-clean. About 15 s, most of it the 1500 ms streaming warm-up, the 500 ms
compute warm-up and five configurations × five rotated sweeps.

---

## TODO 1 — `compulsoryBytes()`

```cpp
return 4.0 * ((double)M*K + (double)K*N + (double)M*N);
```

Module 11's definition: every distinct element that is read is read once no
matter how many times the source text names it, every element written is
written once, an array both read and written counts twice. With `beta == 0`, C
is written and **not** read, so it contributes `M*N` and not `2*M*N`. At
1027 × 2053 × 769 this is **17 907 804 bytes = 17.91 MB**.

Two wrong answers and their tells: counting C twice (gives 26.3 MB and a
compulsory intensity of 123 instead of 181 — an honest mistake if you forget the
`beta == 0` contract, and it is the right answer when `beta != 0`); and
forgetting the factor of 4 (gives an intensity in FLOP/element, which is 724 and
looks suspiciously like the requested/compulsory ratio).

## TODO 2 — `requestedBytes()`

```cpp
return 4.0 * 2.0 * (double)M * (double)N * (double)K;
```

`M*N` threads × `K` iterations × 2 loads × 4 bytes = **12.97 GB**, which is
**724×** the compulsory figure. For a square n the ratio is `2n/3`, so it grows
linearly with the problem: at n = 4096 it is 2730×.

The exercise text is explicit that this is counted at the granularity of the
*load operand*, not of the sector. Exercise 2's sector count is a different and
larger number: under the good mapping a warp requests 5 sectors = 160 B for 256 B
of operand demand, so the sector-level figure is `5/8` of... no — it is
`(5 × 32)/(32 × 2 × 4) = 0.625` per requested byte, i.e. *smaller*, because the
A broadcast delivers one sector for 32 lanes' worth of demand. Under the bad
mapping it is `(33 × 32)/256 = 4.125` per requested byte, i.e. four times
*larger*. That the two models disagree in direction is worth a minute's thought;
neither is wrong, they count different things.

## TODO 3 — `machineBalance()` and `rooflineGflops()`

```cpp
double machineBalance(double ceilGflops, double ceilGBs) { return ceilGflops / ceilGBs; }

double rooflineGflops(double intensity, double ceilGflops, double ceilGBs) {
    const double mem = intensity * ceilGBs;
    return (mem < ceilGflops) ? mem : ceilGflops;
}
```

The units work out because GFLOP/s ÷ GB/s = FLOP/byte — the 10⁹s cancel. This is
the only place in the module where the 10⁹ bookkeeping is easy, and it is worth
noticing *why*: both ceilings are quoted per second, and the "per second"
divides out, leaving a property of the machine with no time in it. Machine
balance is a hardware constant; it does not depend on the kernel.

Measured on this GPU: **17 787–18 256 GFLOP/s** compute, **410.8 GB/s** memory,
balance **43.3–44.4 FLOP/byte**. Module 12 estimated ~42 FLOP/byte for the same
part from a shorter measurement; the two agree.

`rooflineGflops` is `min(flat roof, sloped roof)` and nothing else. The common
error is to return the sloped roof unconditionally, which produces the claim
that a kernel with intensity 181 could run at 74 000 GFLOP/s on a machine whose
FP32 units top out at 18 000.

## TODO 4 — `minOnChipFraction()`

```cpp
if (requested <= 0.0 || ms <= 0.0 || ceilGBs <= 0.0) return 0.0;
const double fromDram = ceilGBs * 1.0e9 * ms * 1.0e-3;
if (fromDram >= requested) return 0.0;
return 1.0 - fromDram / requested;
```

The argument: in `ms` milliseconds, DRAM cannot deliver more than
`ceilGBs · 10⁹ · ms · 10⁻³` bytes. If the kernel *asked for* more than that and
finished in that time, the excess was serviced somewhere other than DRAM.
Nothing else is assumed — no hit-rate counter, no cache model, no knowledge of
the access pattern.

**The degenerate case is the marked part of the TODO.** If `fromDram >=
requested` the inequality is satisfied with everything coming from DRAM and
**no bound follows**; the function must return 0, not a negative number. The
harness probes exactly this with `minOnChipFraction(1e8, 1.0, 400.0)`, where
1e8 bytes in 1 ms is 100 GB/s, comfortably under the 400 GB/s ceiling. A
function that returns `1 - 4e11/1e8 = -3999` fails the hash.

**Which ceiling to pass in.** The harness passes the **432.0 GB/s pin peak**,
not the measured streaming figure. A *bound* needs an upper bound on DRAM
delivery. 432 GB/s is a property of a 192-bit bus at 9.001 GHz and cannot be
exceeded; the measured streaming number moves with the memory P-state (this
session saw 294–411 GB/s from the identical binary, with `nvidia-smi` reporting
the SM clock dropping to 285 MHz during a pure streaming kernel) and using a
throttled value would overstate the conclusion. Result on this problem:
**at least 91.6 % of the requested bytes were serviced on chip.**

The bound is deliberately loose here. The working set is 17.9 MB and the L2 is
48 MB, so after first touch essentially *all* of the traffic is on chip and the
true figure is ~99.9 %. What matters is that 91.6 % is *provable* from a
stopwatch and a bus width, with `ncu` unavailable.

## TODO 5 — `fmasPerLoadNeeded()` (design)

```cpp
double num = 0.0, den = 0.0;
for (int i = 0; i < n; ++i) { num += fpl[i]*gf[i]; den += fpl[i]*fpl[i]; }
const double c = num/den;              // GFLOP/s per (FMA per load)
return targetGflops / c;
```

**Identifying the relationship is the exercise.** The observed data:

| R | FMAs per load | ms | GFLOP/s | % ceiling |
|---|---|---|---|---|
| 1 | 0.50 | 2.53 | 1283 | 7.2 % |
| 2 | 1.00 | 2.69 | 2410 | 13.6 % |
| 4 | 2.00 | 2.83 | 4589 | 25.8 % |
| 8 | 4.00 | 3.01 | 8622 | 48.5 % |

Eight times the arithmetic for **19 % more time**. Throughput is very nearly
proportional to FMAs-per-load, which is the signature of a kernel whose clock is
set by one resource — the load path — and whose arithmetic is free until that
resource stops being the constraint. So fit `gf = c · fpl` and invert. A
least-squares fit through the origin, a single-point ratio from the last row, or
a two-point slope all give answers within a few per cent of each other here,
which is the point: you are not being scored on the estimator, you are being
scored on recognising proportionality rather than, say, fitting a line with an
intercept (which would give a negative intercept and a nonsense answer at small
`fpl`).

The synthetic probe data is exactly proportional (`gf = 2500·fpl`), so every
reasonable method returns 4.500 for the probed target and the hash accepts it.
The probe's target is deliberately *not* the one the real measurement uses, so
the answer cannot be copied out of the lesson.

**The answer on real data: 6.47 FMAs per global load** to reach 80 % of the
measured compute ceiling. A one-element-per-thread GEMM supplies **0.50**. That
is a factor of **13**, and it is a property of the decomposition — no block
shape, grid shape, mapping or launch parameter changes it.

---

## Performance reasoning: reading the diagnosis

The program's final table is the module's thesis in four lines:

```
  intensity, compulsory model         181.08 FLOP/byte
  intensity, requested model            0.25 FLOP/byte
  roofline, compulsory model         17786.6 GFLOP/s
  roofline, requested model             102.7 GFLOP/s
  naive kernel measured               1283.5 GFLOP/s
  fraction of the compulsory roof reached :   7.22%
  ratio to the requested roof             :  12.50x
```

The kernel simultaneously reaches **7.2 % of the roof its algorithm entitles it
to** and runs **12.5× faster than the roof its request pattern implies**. Both
numbers are correct and they are the two halves of the diagnosis:

- The 12.5× is the caches. The request pattern would be catastrophic if it
  reached DRAM; it does not, and the ≥91.6 % bound says so without a profiler.
  **Removing traffic that is not reaching DRAM buys nothing** — Module 6 measured
  that on a stencil (0.85× from tiling) and it is equally true here.
- The 7.2 % is the instruction mix. Two `LDG` and an `IMAD.WIDE` per `FFMA`
  means at best one instruction in four is arithmetic, and the LSU and L1 return
  path saturate long before the 128 FP32 lanes per SM do.

So the answer to "what upper bound on speedup is available without changing the
algorithm" is: **the addressing fix, and then nothing.** Exercise 2 measured the
addressing fix at up to 4× (8.2× on a 1024³ problem), and the best
one-element-per-thread configuration lands at ~1283–1350 GFLOP/s, which is the
`R = 1` row of the probe table — the same kernel. Every configuration of the
naive decomposition is bounded by that row.

And the answer to "what must change" is quantitative: **FMAs per global load
must rise from 0.5 to about 6.5.** There are exactly two levers:

1. Make each operand read cheap — stage tiles of A and B in the block-scoped
   scratchpad so the repeated reads are `LDS` against a scratchpad with no tag
   compare and deterministic latency, and the global loads happen once per block
   instead of once per thread. **Module 17.**
2. Make each value loaded into registers feed several FMAs — give each thread a
   patch of C so one A value multiplies several B values. **Module 18.**

Lever 1 alone does not change the count of memory instructions per FMA; it
changes their cost. That is precisely why Module 6 found tiling a 5-point
stencil to be a net loss and named **register blocking** as the missing
ingredient. Lever 2 alone leaves every operand coming from global memory. The
6.47 is how much of lever 2 is needed, and lever 1 is what makes lever 2
affordable.

---

## Expected output

Actual run on the RTX 3500 Ada, CUDA 13.2:

```
-- your analysis functions, on fixed synthetic inputs --------------
  compulsoryBytes(1024,2048,512)            = 14680064
  requestedBytes(1024,2048,512)             = 8589934592
  machineBalance(18000 GFLOP/s, 410 GB/s)   = 43.9024 FLOP/byte
  rooflineGflops(2,   18000, 410)           = 820.00 GFLOP/s
  rooflineGflops(200, 18000, 410)           = 18000.00 GFLOP/s
  minOnChipFraction(1e12 B, 1 ms, 400 GB/s) = 0.99960
  minOnChipFraction(1e8  B, 1 ms, 400 GB/s) = 0.00000
  fmasPerLoadNeeded(11250, synthetic)       = 4.5000
  answer hash 597e8670 -> all correct

-- measured ceilings ---------------------------------------------
  FP32 FFMA ceiling      17786.6 GFLOP/s
  DRAM read ceiling        410.8 GB/s  (95% of the 432.0 GB/s pin peak)
  machine balance          43.30 FLOP/byte   [your TODO 3]

-- your ledger, applied to this problem ---------------------------
  compulsory bytes                  17907804  (17.91 MB)
  requested bytes                12971067512  (12.97 GB, 724x compulsory)
  intensity, compulsory model         181.08 FLOP/byte
  intensity, requested model            0.25 FLOP/byte
  roofline, compulsory model         17786.6 GFLOP/s
  roofline, requested model            102.7 GFLOP/s
  naive kernel measured               1283.5 GFLOP/s  (2.5265 ms)
  fraction of the compulsory roof reached :   7.22%
  ratio to the requested roof             :  12.50x  (above 1 is the caches)
  minimum on-chip service fraction        :  91.59%   [your TODO 4]

-- the FMAs-per-load family --------------------------------------
  R      FMAs per load            ms      GFLOP/s   % ceiling
  R=1               0.500     2.5279       1282.8       7.21%
  R=2               1.000     2.6911       2410.0      13.55%
  R=4               2.000     2.8266       4589.0      25.80%
  R=8               4.000     3.0089       8621.9      48.47%

  FMAs per global load needed to reach 80% of the compute ceiling:
     6.47      [your TODO 5]

  SCORE: 1/1  (all eight analysis answers must be right)
OVERALL: PASS
```

The eight hashed answers are deterministic. The measured block varies with
thermal state: across this session the FP32 ceiling measured **13 210–18 256
GFLOP/s**, the DRAM read ceiling **294–411 GB/s**, and the naive kernel
**1041–1350 GFLOP/s** (up to 465 GFLOP/s when the part was deeply thermally
limited). The *ratios* — 7 % of the compute roof, 12–19× the requested roof, 6–7
FMAs per load needed, ~13× short of the requirement — reproduced across every
run. Quote the ratios; the absolute milliseconds on this laptop part are not
reproducible and the file says so.

---

## The result that matters

You can diagnose this kernel completely without a profiler. Elapsed time, a bus
width, and two microbenchmarked ceilings are enough to prove that ≥91.6 % of its
memory traffic never reaches DRAM, that it is at 7 % of the compute roof its own
algorithm entitles it to, and that closing the gap requires each loaded value to
feed about thirteen times as many multiply-adds as a one-element-per-thread
decomposition can ever supply. That last number is not an optimization
opportunity; it is a proof that the decomposition has to change, and it is why
the next two modules are forced moves rather than tricks.

**Variation to try:** set `M_DIM = N_DIM = K_DIM = 2048` so the working set
(48 MB) no longer fits in the L2, and re-run. Watch `minOnChipFraction` stay
high (the *requested* traffic is still absorbed by L1 and L2 many times over)
while the measured GFLOP/s falls — and then ask which of the two numbers moved
and why. Then set `K_DIM = 128` with M and N unchanged and watch the compulsory
intensity collapse toward the machine balance: the point at which GEMM stops
being a compute-bound problem at all is computable, and it is the reason
"tall-skinny" GEMMs are a separate optimization discipline.
