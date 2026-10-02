# Module 21 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
exercise02_solution.exe
```

Warning-clean. `SCORE: 9/9`, `OVERALL: PASS`. About 45 s; the validation pass
dominates (three full Freivalds probes plus a host-side reduction of 2^25
floats).

---

## TODO 1 — the traffic ledger

```cpp
    // 1 triad
    L[0].flops = 2.0*NELEM; L[0].dramBytes = 12.0*NELEM; L[0].reqBytes = 12.0*NELEM;
    // 2 reduce
    L[1].flops = 1.0*NELEM; L[1].dramBytes =  4.0*NELEM; L[1].reqBytes =  4.0*NELEM;
    // 3 gemmNaive
    L[2].flops = 2.0*NM;    L[2].dramBytes = 4.0*(sA+sB+sC); L[2].reqBytes = 8.0*NM;
    // 4 gemmTiled
    L[3].flops = 2.0*NM;    L[3].dramBytes = 4.0*(sA+sB+sC); L[3].reqBytes = 8.0*NM;
    // 5 gemmReg 8x4
    L[4].flops = 2.0*NM;    L[4].dramBytes = 4.0*(sA+sB+sC);
    L[4].reqBytes = 4.0*(8.0+4.0)/(8.0*4.0)*NM;
    // 6 chase
    L[5].flops     =        (double)CHASE_THR*CHASE_STEPS;
    L[5].dramBytes = 32.0 * (double)CHASE_THR*CHASE_STEPS;
    L[5].reqBytes  =  4.0 * (double)CHASE_THR*CHASE_STEPS;
```

Row by row, with the trap in each.

**1 triad — `12N`, not `8N`.** `y[i] = fmaf(a, x[i], y[i])` reads `x`, reads
`y`, writes `y`. Three streams of 4 bytes. This is M11's measured trap verbatim:
a 2N model on this kernel reports 250.8 GB/s and makes a finished kernel look
42% short, while the 3N model reports 376.1 GB/s. The source text mentions `y`
once on the right-hand side, which is exactly why people miss it.

**2 reduce — `4N`.** 240 partials written against 2^25 elements read; the write
side is 1e-5 of the traffic and rounds away. One `FADD` per element, so `F = N`,
not `2N`.

**3/4 gemm — the two columns that must differ.** `beta == 0`, so by the BLAS
contract (M16) C is not read; compulsory is `4(MK + KN + MN) = 17.9 MB`.
Requested is two 4-byte operands per FMA = `8MNK = 12.97 GB`. The ratio is
**724x**, which is M16's headline number.

**4 vs 3 — the row that surprises people.** `gemmTiled`'s requested bytes are
*identical* to `gemmNaive`'s. The tile loop still executes
`acc = fmaf(As[ty][k], Bs[k][tx], acc)` — two 4-byte operand fetches per FMA.
What tiling changed is the *opcode*: `LDG` became `LDS`. M17 states this as the
hand-off to M18 ("tiling changed the opcode, not the count") and the index
records the measured version: 8.00 FMAs per global load but still 0.50 per
shared load, 0.47 per memory instruction of any kind.

A reader who writes `L[3].reqBytes = 8.0*NM/16.0` has counted the *global* side
only, will predict 21 700 GFLOP/s, measure 1634, and be 13x out.

**5 gemmReg — `(TM+TN)/(TM·TN)` floats per FMA.** Per k-step a thread loads
`TM + TN = 12` floats into registers and performs `TM·TN = 32` FMAs. That is
48 bytes per 32 FMAs = **1.5 B/FMA**, against 8. M18's two-level law:
`FMAs per shared read = TM·TN/(TM+TN)`.

**6 chase — the only row where `req < dram`.** Each hop reads one 4-byte `int`,
and the stride is a prime near 2^25 so every hop lands in a different 32-byte
sector and a different 128-byte line. Four useful bytes, 32 bytes moved. M5's
sector model, at its worst possible efficiency of 12.5%.

Wrong answers and their symptoms:

| mistake | symptom in the output |
|---|---|
| `triad` at 8N | predicts 103 GFLOP/s, measures 62, "below (0.60x)" — and you will blame the hardware |
| GEMM `dramBytes` counting C twice | 45% error in compulsory traffic (M16 ex3's trap); invisible here because DRAM never binds the GEMMs |
| `gemmT.reqBytes` divided by T | predicts above the FP32 ceiling; the `min` clamps it to "compute bound" and the row reads `ABOVE the roof` |
| `chase.dramBytes` at 4 per hop | predicts 103 GFLOP/s instead of 12.8; the measured/predicted ratio becomes 0.002 and Little's Law no longer reconciles |

## TODO 2 — `classify()`

```cpp
    double pD = flops/dramBytes * ceilDram;
    double pR = flops/reqBytes  * ceilOnChip;
    double pI = flopPerInstr * ceilIssueGI;
    double p = pD;  int lv = LVL_DRAM;
    if (pR < p) { p = pR; lv = LVL_ONCHIP;  }
    if (pI < p) { p = pI; lv = LVL_ISSUE;   }
    if (ceilFp32 < p) { p = ceilFp32; lv = LVL_COMPUTE; }
    *predGF = p; return lv;
```

A plain `min` over four plateaux. The units work out without any conversion
constant: `GB/s x FLOP/byte = GFLOP/s`, and `G instructions/s x FLOP/instruction
= GFLOP/s`. A reader who inserts a 1e9 gets answers off by nine orders of
magnitude and notices immediately; the dangerous mistake is subtler — comparing
`AI` values against each other instead of comparing *rates*. Two levels with
different bandwidths cannot be ranked by their intensities.

## TODO 3 — what actually limits each kernel

```cpp
static const int PRED_LEVEL[6] = {
    LVL_DRAM, LVL_DRAM, LVL_ONCHIP, LVL_ONCHIP, LVL_ONCHIP, LVL_LATENCY
};
```

Five of six agree with `classify()`. The sixth does not, and that disagreement
is the exercise. `classify()` reports `LVL_DRAM` for the chase because DRAM is
the lowest of the four plateaux it knows about; the truth is that **none** of
the four is reached, by a factor of 58. The roofline's vocabulary presupposes
that something is saturated. When nothing is, the model does not have a wrong
answer — it has no answer, and `LVL_LATENCY` is the honest one.

A reader who answers `LVL_DRAM` for the chase has correctly run the model and
incorrectly believed it.

## TODO 4 — the performance buckets

```cpp
static const int PRED_BUCKET[6] = { 1, 1, 3, 3, 5, 1 };
```

Buckets are `<1 / 1-3 / 3-12 / 12-25 / >25` percent of the measured FP32
ceiling. Measured: 0.32, 0.50, 6.8, 8.3, 39.6, 0.001.

The reasoning that gets them right is purely the model, once TODO 1 is right:

- triad and reduce: `0.167 x 411 = 69` and `0.25 x 411 = 103` against a ceiling
  of ~19 700 — both about 0.4%, bucket 1.
- gemmN and gemmT: `0.25 x 5430 = 1357`, which is 6.9% — bucket 3, and the
  **same bucket for both**, which is the point.
- gemmR: `1.333 x 5430 = 7240`, 36.6% — bucket 5.
- chase: anything below everything — bucket 1.

The bucket edges sit in the widest empty gaps of the measured distribution
(spec §12 rule 5d). The thinnest margin is gemmR at 39.6% against a 25% edge, a
factor of 1.6.

## TODO 5 — Little's Law (design)

```cpp
static double bytesInFlightNeeded(double bwGBs, double latCycles, double clockGHz)
{ return bwGBs * latCycles / clockGHz; }

static double bytesInFlightSupplied(double nThreads, double sectorBytes)
{ return nThreads * sectorBytes; }
```

The derivation, which is the part worth having done by hand:

```
bytes = bandwidth x latency
      = (bwGBs x 1e9 bytes/s) x (latCycles / (clockGHz x 1e9) seconds)
      = bwGBs x latCycles / clockGHz          -- the 1e9 cancel exactly
```

At 411 GB/s, 575 cycles (M4's measured dependent-load latency) and 1.93 GHz
that is **122 413 bytes**, about 3800 sectors in flight, about 120
fully-coalesced warp loads. The chase kernel is one warp with one dependent load
per lane: **1024 bytes**. Ratio 0.0084.

Measured fraction of the DRAM ceiling: 0.0172. Ratio of ratios: **2.05**.

The factor of two is explained, not swept away. The 575-cycle figure is M4's
measurement of a *single-threaded* chase, where exactly one request is
outstanding. Here a warp issues one `LDG` whose 32 lanes carry 32 independent
addresses, so 32 round trips overlap inside one instruction and the per-hop
latency the warp observes is shorter. Working backwards from the measurement:
`0.4541 ms / 3000 hops x 1.931 GHz = 280 cycles`. Little's Law gets the
mechanism and the order of magnitude; it does not get the constant, and the
harness's 4x tolerance is calibrated to that honesty rather than to an
unobtainable precision.

---

## Synchronization / memory reasoning

Nothing new. `reduceV6` uses M12's barrier structure (shared tree down to 32,
then `__shfl_down_sync` with no barrier because reader and writer of
`sd[tid]` are the same thread — M12's TODO 1(d)). The two GEMM tilings use
M17's RAW+WAR barrier pair. The chase has no shared state at all.

The validation pass prefills C with `+infinity` before each GEMM and runs the
finiteness check **first**, per M16: a NaN compares false against everything, so
a max-error loop run first would pass an all-NaN array.

---

## Performance reasoning

The whole table is the performance reasoning, but three rows deserve a sentence
each.

**gemmT measures 1.20–1.24x its prediction.** This is the one row consistently
above the roof, and it is honest to say we do not have a counter-level
explanation. Two candidates: the `As[ty][k]` read is a broadcast (M17: degree 1,
all 32 lanes of a warp read the same word), and a broadcast does not consume
32 lanes' worth of bank bandwidth — so the kernel's *effective* request rate is
below the 8 B/FMA the ledger charges it. The second is that `ptxas` contracts
four of the `As` reads into `LDS.128` (M17 measured 4 `LDS.128` + 16 `LDS` for
16 FFMAs), and the vector path is 1.9x the scalar one. Both push the same way.
A ledger that counted *instructions* rather than bytes, with the broadcast and
the merge accounted for, would close the gap; `ncu`'s
`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` is the counter that would
settle it, and it is unavailable here (M23).

**triad measures 0.79–0.91x.** The ceiling is measured with a read-only stream
and triad reads two arrays and writes one. A mixed read/write stream does not
reach a pure-read ceiling: this run shows triad putting 374 GB/s on the pins
against a 411 GB/s read ceiling. M15 made the same argument for the transpose
and used a *matched copy* as its denominator rather than a stream.

**chase measures 0.017x.** See TODO 5.

---

## Expected output

Real output, one good run:

```
-- ceilings measured in this process ---------------------------------
  DRAM    411.2 GB/s   on-chip operand fetch   5429.6 GB/s
  FP32  19776.2 GFLOP/s   issue  309.0 G warp-instr/s   clock 1.931 GHz
  DRAM ridge 48.10 FLOP/byte   on-chip ridge 3.64 FLOP/byte

-- your classification, before anything is timed ---------------------
  kernel     AI(dram)   AI(req)   roofline  predGF/s  you say   bucket
  1 triad       0.167     0.167       DRAM      68.5     DRAM        1
  2 reduce      0.250     0.250       DRAM     102.8     DRAM        1
  3 gemmN     181.081     0.250    on-chip    1357.4  on-chip        3
  4 gemmT     181.081     0.250    on-chip    1357.4  on-chip        3
  5 gemmR     181.081     1.333    on-chip    7239.5  on-chip        5
  6 chase       0.031     0.250       DRAM      12.8  latency        1

-- measurement and reconciliation ------------------------------------
  kernel       GFLOP/s   predGF/s meas/pred   %FP32  bucket verdict
  1 triad       62.319       68.5    0.9094  0.315%       1 at roof
  2 reduce      99.297      102.8    0.9660  0.502%       1 at roof
  3 gemmN     1342.608     1357.4    0.9891  6.789%       3 at roof
  4 gemmT     1633.827     1357.4    1.2036  8.262%       3 at roof
  5 gemmR     7828.837     7239.5    1.0814 39.587%       5 at roof
  6 chase        0.220       12.8    0.0172  0.001%       1 FAR below

-- TODO 5: why the chase is   58x below its own roofline -------------
  bytes the memory system needs in flight :     122413
  bytes this kernel has in flight         :       1024
  predicted fraction of the DRAM ceiling  :    0.00837
  measured  fraction of the DRAM ceiling  :    0.01715
  ratio                                   :      2.051

SCORE: 9/9
OVERALL: PASS
```

Observed ranges across runs:

| kernel | measured GFLOP/s | meas/pred | % of FP32 ceiling |
|---|---|---|---|
| triad | 49 – 62 | 0.79 – 0.91 | 0.25 – 0.33% |
| reduce | 80 – 99 | 0.85 – 0.97 | 0.40 – 0.53% |
| gemmN | 1169 – 1343 | 0.90 – 0.99 | 6.1 – 6.8% |
| gemmT | 1474 – 1634 | 1.14 – 1.24 | 7.6 – 8.3% |
| gemmR | 7367 – 8154 | 1.06 – 1.14 | 37.9 – 41.6% |
| chase | 0.21 – 0.23 | 0.016 – 0.019 | 0.001% |

and ceilings DRAM 373–411 GB/s, on-chip 5190–5430 GB/s, FP32 19 450–19 780
GFLOP/s. Every bucket is stable across all of that.

---

## The result that matters

Three GEMMs, three modules, one axis. Naive and tiled have **the same**
request-level arithmetic intensity, 0.25 FLOP/byte, and land within 20% of the
same prediction; register tiling moves that intensity to 1.333 and the
throughput moves with it, 5.8x, almost exactly as predicted. The two-axis
roofline cannot draw any of this: all three kernels have an identical DRAM
arithmetic intensity of 181 FLOP/byte, four times the ridge, and it calls all
three compute bound while they run at 6.8%, 8.3% and 39.6% of the compute
ceiling. The axis you need is the one the classical picture does not have.

And then the sixth row, which has no axis at all. The chase is at 1.7% of the
only ceiling it comes near, and the number that explains it is not a bandwidth
or a FLOP count but 1024 bytes against 122 413. Modules 19 and 20 are about that
number.

**Variation to try.** Change `CHASE_THR` from 32 to 1280 and the launch from
`<<<1,32>>>` to `<<<40,32>>>` — one warp per SM instead of one warp per GPU.
Predict the new measured fraction from Little's Law before you run it, then run
it. The model should track across a 40x change in concurrency, and watching it
do so is the best possible argument for taking Little's Law as seriously as the
roofline.
