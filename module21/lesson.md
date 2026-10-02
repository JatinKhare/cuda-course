# Module 21 — The Roofline Model

> Prerequisites: Modules 1–18. In particular M4 (the storage map), M5 (sector
> counting), M7 (the bank array), M11 (compulsory traffic and the floor),
> M12 (the streaming ceiling), M16 (the two arithmetic intensities of GEMM and
> the measured FP32 ceiling), M17 (shared-memory bandwidth as a ceiling),
> M18 (the two-level reuse law).
>
> What this module gives you: a single picture that holds every ceiling this
> GPU has, a procedure for placing an arbitrary kernel on it *before* you run
> the kernel, and an exact account of the three things the picture cannot see.

---

## Concept

### 1. The one-line model, and why one line is not enough

A kernel performs `F` floating-point operations and moves `B` bytes. Its
**arithmetic intensity** is

```
AI = F / B        [FLOP per byte]
```

If the machine can sustain `P` FLOP/s of arithmetic and `S` bytes/s of
bandwidth, and if the two overlap perfectly, the time is bounded below by
`max(F/P, B/S)` and therefore the achievable rate is bounded above by

```
attainable  =  min( P,  AI x S )      [FLOP/s]
```

Plot that on log–log axes with `AI` on x and attainable rate on y and you get a
diagonal of slope 1 (the memory-bound regime, `AI x S`) meeting a horizontal
plateau (the compute-bound regime, `P`). The meeting point is the **ridge
point** `AI = P/S`, also called the **machine balance**: the arithmetic
intensity at which this machine's arithmetic and its memory system are in
balance. That is the roofline (Williams, Waterman and Patterson, 2009).

It is a good model. On this GPU, in this course, it is also *wrong about three
of the six kernels you have already written*, and the reason it is wrong is
always the same: **`B` is not a single number, because there is not a single
memory.**

Module 16 already showed you the first crack. A naive GEMM's compulsory DRAM
traffic gives `AI = 181 FLOP/byte` on the shape 1027x2053x769 — four times the
ridge point, comfortably compute bound — and the kernel runs at **7% of the
compute ceiling**. Module 17 showed you the second: a tiled GEMM reads two
shared-memory floats per fused multiply-add, which is `2 FLOP / 8 B = 0.25
FLOP/byte`, and it is capped at **7.4–14.2% of the compute ceiling no matter
what DRAM does**. M17 states the conclusion plainly: a two-axis roofline cannot
express tiled GEMM. This module builds the axis it is missing.

### 2. Arithmetic intensity, defined carefully

`AI = F / B` is unambiguous about `F` and completely ambiguous about `B`. There
are at least four defensible byte counts for the same kernel.

| byte count | what it means | measured where |
|---|---|---|
| **compulsory** | each distinct input read once, each output written once | nowhere; it is a property of the problem |
| **DRAM** | what actually crosses the pins | `dram__bytes` (ncu, M23) |
| **requested** | what memory *instructions* ask for, summed | `l1tex__t_bytes` (ncu, M23) |
| **useful** | the bytes the arithmetic consumes | your own arithmetic |

They differ by large factors, and the factors are the content of the earlier
modules:

- **Compulsory vs requested.** M16 measured the naive GEMM requesting
  **724x** its compulsory traffic on this shape (12.971 GB vs 17.908 MB). The
  caches absorb the difference: M16 bounded the on-chip service fraction at
  **>= 91.6%** from elapsed time and the 432 GB/s pin rate alone.
- **Useful vs moved.** M5's sector model. A warp that reads 32 scattered floats
  uses 128 bytes and moves 32 x 32 = 1024. The dependent pointer walk in this
  module's Example 2 uses 4 bytes per hop and moves a 32-byte sector: an 8x
  amplification that no source-level byte count shows.
- **Read-modify-write.** M11's measurement, and the one people get wrong most
  often: `y[i] += a*x[i]` moves **3N**, not 2N, because `y` is read as well as
  written. Under a 2N model the same kernel reports 250.8 GB/s and looks 40%
  short; under the correct 3N model it reports **376.1 GB/s** and is finished.
  The model, not the kernel, was broken.
- **Which level.** Tiled GEMM has `AI(DRAM) = 181`, `AI(L2) = 4.0` and
  `AI(shared) = 0.25` — three different numbers for one kernel. **They are all
  correct.** Each belongs to a different ceiling, and you must compare each to
  its own.

> **Rule.** An arithmetic intensity is meaningless without naming the level it
> was measured at. "This kernel has an AI of 0.25" is not a statement.

### 3. The hierarchical roofline

Draw one sloped ceiling per level of the memory hierarchy, all against the same
compute plateau. Each ceiling has its own ridge point. A kernel appears once per
level, at its own `AI` for that level, and the binding constraint is the
*lowest* of the predicted rates.

Measured on this GPU by `example01.cu` (one representative run; see §7 for the
spread):

| ceiling | measured | ridge point against 18.6 TFLOP/s |
|---|---|---|
| DRAM, read | **410.4 GB/s** (95.0% of the 432 pin peak) | **45.4 FLOP/byte** |
| L2, 24 MB working set | 1259–1938 GB/s | 10–15 FLOP/byte |
| shared / L1, scalar `LDS` | **5224 GB/s** | **3.57 FLOP/byte** |
| shared / L1, `LDS.128` | **9687 GB/s** | **1.92 FLOP/byte** |
| FP32 FFMA | **18 642 GFLOP/s** | — |
| instruction issue | **310 G warp-instructions/s** | — |

The DRAM ridge is the classical machine balance, and it reproduces the
43–44 FLOP/byte the cross-module index records. The *shared* ridge is the number
that explains Part V:

```
tiled GEMM: 2 FLOP per 8 shared bytes = 0.25 FLOP/byte
            0.25 / 3.57 = 7.0% of the compute ceiling
            (with LDS.128: 0.25 / 1.92 = 13.0%)
M17 measured 9.4%. It sits between the two, as it must.
```

and the register-tiled kernel is the *same kernel moved to the right on the same
axis*:

```
8x4 register tile: per k-step a thread reads TM+TN = 12 floats = 48 B
                   and performs TM*TN = 32 FMAs = 64 FLOP
                   AI(shared) = 64/48 = 1.333 FLOP/byte
                   1.333 x 5224 GB/s = 6964 GFLOP/s predicted
M18 measured 7399-8336. Example 2 here measures 7806 against a 6911 prediction.
```

That is the whole of Modules 17 and 18 in two lines of arithmetic, and neither
line mentions DRAM.

#### Why "shared" and "L1" share a ceiling

They are the same SRAM. On Ada an SM has 128 KB of unified L1 + shared memory,
and the data return path is the same. M16 measured the "FMAs needed per global
load" curve with `LDG`; M17 reran it with `LDS` and got the same curve to within
1%. The cross-module index records the restatement: the law is about **memory
instructions of any address space**, not about the global address space.

Example 2 confirms it from the other direction. All three GEMMs in the table
sit on one ceiling:

| kernel | AI(request level) | predicted | measured | ratio |
|---|---|---|---|---|
| naive (M16) | 0.250 | 1359 | 1169 | 0.90 |
| tiled (M17) | 0.250 | 1359 | 1474 | 1.14 |
| register-tiled (M18) | 1.333 | 7248 | 7806 | 1.13 |

Naive and tiled have **identical** request-level intensity, and that is exactly
right: M17's hand-off to M18 says "tiling changed the opcode, not the count",
and here is the model that says the same thing in one column.

#### The level you may not use

There is a fourth row you might want to draw and must not: requested bytes
against the L2 ceiling. For the naive GEMM that is `0.25 x 1305 = 326 GFLOP/s`,
and the kernel measures 1169 — **3.6x above a "ceiling"**. Nothing is wrong with
the hardware. The 12.971 GB of requests never reach L2; most of them hit in L1.

> **Rule.** A level's roofline binds only if the traffic you counted actually
> crosses that level. Counting requested bytes against an L2 ceiling assumes
> every request misses L1, which is a different kernel from the one you have.

### 4. The instruction-issue ceiling

Beyond bytes and FLOPs there is a third resource: **issue slots**. An SM on
sm_89 has 4 processing blocks, each with one warp scheduler that issues at most
one instruction per clock (M1). Device-wide:

```
issue ceiling = 40 SMs x 4 schedulers x 1 instr/cycle x clock
              = 160 warp-instructions per cycle
```

Example 1 measures **310 G warp-instructions/s**, which at the implied 1.82 GHz
is 170 instructions per cycle — within measurement error of 160, given that the
probe kernel clocks slightly above the FFMA kernel.

The interesting part is what this does to the "compute plateau". An FFMA
warp-instruction retires 32 lanes x 2 FLOP = **64 FLOP per issue slot**. So

```
FP32 ceiling = issue ceiling x 64 FLOP/instruction
```

and the flat top of the classical roofline is *not a separate thing*. It is the
issue ceiling evaluated at the maximum possible FLOP density. For a kernel whose
instruction stream is only `d` fraction FFMA, the plateau is `d x 64 x 310`
GFLOP/s, not 18 642.

Example 1 demonstrates this directly with two kernels that differ by one
instruction. `ffmaProbe` issues `a = fmaf(a,b,1.0f)`; `mixedProbe` issues
`a = fmaf(a,b,1.0f) + b`, which IEEE semantics forbid the compiler from
re-associating into a single FFMA. Measured SASS loop bodies: **68 instructions
per 128 FLOPs** and **132 per 192**. Predicted ratio
`(192/132)/(128/68) = 0.77`; measured **0.78**. Same issue rate, less
arithmetic per slot, proportionally less throughput.

This also means the "FP32 peak" you measure is itself 94% of the lane peak,
because even the best possible loop spends 4 of every 68 slots on the loop
counter, the compare and the branch.

M18's instruction-mix table is this axis in the GEMM setting:

| kernel | inner-loop body | FFMA density | issue plateau |
|---|---|---|---|
| naive GEMM | 87 instructions, 16 FFMA | 18% | ~3.7 TFLOP/s |
| tiled GEMM | 66 instructions, 16 FFMA | 24% | ~4.8 TFLOP/s |
| register-tiled 8x4 | 365 instructions, 256 FFMA | 70% | ~13.9 TFLOP/s |

None of those three is *issue* bound — all three are on the operand-fetch
ceiling below — but the issue plateau is what M18's drive toward 90.8% FFMA
density was buying headroom against. A kernel at 18% density has already
forfeited 80% of the machine before a single byte moves.

### 5. Classification as a procedure

Given a kernel, in this order:

1. **Count `F`.** Useful floating-point operations. An FMA is 2. Be explicit
   about conventions (`rsqrtf` is conventionally 2; say which you used).
2. **Count the bytes, three times.** Compulsory across the pins; requested by
   memory instructions; and, if you are being careful, moved in sectors.
3. **Compute `AI` at every level** you counted.
4. **Get the instruction mix.** `cuobjdump -sass`, the loop body between the
   back edge and its branch. FLOPs per warp-instruction.
5. **Predict**: `min` over `AI_dram x S_dram`, `AI_req x S_onchip`,
   `rho x issue`, and `P`. Write down which one won and what rate you expect.
6. **Measure.**
7. **Reconcile.** This is where the learning is:

| measured / predicted | what it means |
|---|---|
| 0.8 – 1.25 | the model is right; the named resource is saturated |
| > 1.25 | **your ledger is wrong.** You counted traffic that does not happen, or at the wrong level |
| 0.25 – 0.8 | the right ceiling, imperfectly reached. Usually overlap, tails, or a second ceiling close behind |
| < 0.25 | **nothing is saturated.** Look at latency (M20), occupancy (M19), or the launch shape (M1's tail effect) |

Note what the last row is *not*. A kernel at 2% of its roofline is not "memory
bound". It is not bound by anything in the model at all. The roofline's whole
vocabulary — DRAM bound, compute bound — presupposes that something is
saturated, and when nothing is, the correct answer is to stop using the
roofline and reach for Little's Law.

### 6. Where the roofline lies to you

Four failures, all of them real and all of them measured in this course.

**(a) It assumes perfect overlap.** `max(F/P, B/S)` is a lower bound on time
only if arithmetic and memory proceed simultaneously. They largely do on a GPU
— that is what warps are for — but not completely. Expect to land a few percent
below, not above.

**(b) It ignores latency entirely.** This is the big one. A kernel can be far
below every ceiling and limited only by dependency chains. Example 2's pointer
walk:

```
AI(DRAM) = 1 FLOP / 32 B = 0.031    ->  roofline predicts 12.8 GFLOP/s
measured                                                   0.22 GFLOP/s
                                                           a factor of 57
```

The roofline has no term for this, and Little's Law does. The memory system is
a pipeline of latency `L` running at bandwidth `S`; to sustain `S` it must have
`S x L` bytes outstanding at all times:

```
needed  = 410 GB/s x 575 cycles / 1.83 GHz = 128 596 bytes
supplied = 32 threads x 1 outstanding load x 32 B sector = 1 024 bytes
ratio = 0.0080
```

and the kernel achieves **0.0175** of the DRAM ceiling — the same number within
a factor of two. (It is a factor of two, not one, because a warp issues a single
`LDG` covering 32 independent sectors and those 32 round trips overlap: the
effective round trip here is ~288 cycles, not the 575 M4 measured for a
single-threaded chase. Little's Law gets the mechanism and the order of
magnitude; it does not get the constant.)

**(c) It ignores tail and wave effects.** M1's staircase. A grid of 241 blocks
on a machine that holds 240 resident runs in the time of two waves, and the
roofline, which knows only totals, predicts one. The error is `1/waves`, so it
is invisible for large grids and dominant for small ones.

**(d) It assumes you can reach the ceilings.** The ceilings themselves are
measurements, and on this part they move. Example 1 measured the DRAM ceiling
anywhere from 279 to 411 GB/s across runs of the same binary in one afternoon.
The cross-module index's standing convention applies: **measured figure for a
ceiling, 432 GB/s pin peak for a bound.** A "% of peak" computed against a
ceiling you measured on a cold machine and applied to a kernel timed on a hot
one is a thermal measurement wearing a performance model's clothes.

To which add a fifth, specific to this course's hardware: `cudaDevAttrClockRate`
reports 1.545 GHz on this part and produces an "FP32 peak" of 15 821 GFLOP/s,
which the measurement **exceeds by 1.15x**. Any roofline whose plateau comes
from that API call has kernels above its own roof. Spec §12 rule 6 forbids it;
this module obeys.

### 7. Measuring the ceilings honestly

Every ceiling in this module is measured, not quoted, and every measurement is
checked against a bound before it is used. This is spec §12 rule 13, and it is
not decoration: M17's first shared-memory bandwidth probe reported **40 TB/s**,
four times the bank array's theoretical maximum, because `ptxas` had hoisted the
loop-invariant address out of the timing loop.

Example 1 ships that bug on purpose, next to the correct version, differing by
one statement. The SASS loop bodies:

```
honest  : 133 instructions between the back edge and the branch, 32 of them LDS
hoisted :  35 instructions,                                       0 of them LDS
```

The hoisted kernel's 32 loads sit above the loop. It is timing 32 FADDs on a
loop-invariant constant and reporting the result as shared bandwidth: **34 TB/s,
216–332% of the bound**. The exercise requires your bound to throw it out.

The bounds themselves, from the spec table and `nvidia-smi
--query-gpu=clocks.max.sm` (3105 MHz):

```
DRAM    : 432.0 GB/s                              (the pin rate)
shared  : 40 SM x 32 banks x 4 B x 3.105 GHz   =  15 898 GB/s
FP32    : 40 SM x 128 lanes x 2 FLOP x 3.105   =  31 795 GFLOP/s
issue   : 40 SM x 4 schedulers x 3.105         =    496.8 G instr/s
```

These are deliberately loose — the device's *maximum* clock is the only clock
figure that makes a legitimate bound, and no kernel runs there. A tighter check
is available once the FP32 ceiling is measured: it implies a clock, and the
other probes can be checked at that clock. Example 1 prints both, and reports
honestly that the `LDS.128` probe sometimes exceeds the tight bound — which is
M17's observation that the shared-memory kernel clocks higher than the FFMA
kernel, and the reason the **ratio** between the scalar and vector rows (1.81–
1.92 measured, 1.9 predicted by M7's two-cycle floor) is the trustworthy
quantity rather than the absolute B/cycle/SM.

#### The warm-up, and why this module is the hard case

Spec §12 rule 4 and its corollary: **1500 ms streaming, then 500 ms compute, in
that order.** This module measures a memory ceiling and a compute ceiling in the
same program, which is exactly the case the two-stage warm-up exists for. The
power manager trades away whatever the running kernel is not using — M16 watched
the SM clock drop to **285 MHz** during a pure-read kernel while the memory
clock sat at 8801 MHz.

One refinement this module adds. Spec §12 rule 1 says to time competing
configurations back to back in one rotated sweep. The DRAM ceiling and the FP32
ceiling are **not competitors**; they are two axes of one model, and they need
different warm-ups. So Example 1 uses two timing groups: warm 1500 ms streaming
and measure the off-chip group, then warm 500 ms compute and measure the on-chip
group, each group rotated internally with `SWEEPS >= NCFG`. An earlier draft put
all seven probes in one sweep and measured 326 GB/s and 13 750 GFLOP/s; split
into two groups it measures **410 GB/s and 18 642 GFLOP/s** — both ceilings,
together, in one run.

---

## Hardware Mental Model

The roofline is a picture of four physically distinct resources, and it is worth
being concrete about what each one *is* on this chip.

**The pins.** 192 bits of GDDR6 at 9.001 GHz, double data rate:
`192/8 x 9.001e9 x 2 = 432.0 GB/s`. This is a wire count times a clock. It
cannot be exceeded, it is shared by all 40 SMs, and it is the only ceiling in
the model that does not scale with the SM clock. That asymmetry is why the power
manager can hold the memory clock up and drop the SM clock to 285 MHz.

**The L2.** 48 MB, device-wide, the coherence point (M4, M10). It is not a
separate pipe so much as a place requests stop missing. Its apparent bandwidth
depends entirely on what fraction of the working set lives there — Example 1
measures 1259–1938 GB/s on a 24 MB working set, 3–5x the pin rate, and M11/M14
recorded up to 1305 GB/s. A kernel "achieving 302% of peak bandwidth" is
reporting L2 residency, which is why spec §12 rule 7 insists on buffers well
past 48 MB for any DRAM figure.

**The L1 / shared SRAM.** 128 KB per SM, up to 100 KB addressable as shared,
32 banks x 4 B, **128 bytes delivered per cycle per SM** (M7). That is the
physical origin of the shared ceiling: 40 x 128 x clock. At 1.8 GHz it is
9.2 TB/s, and M7's two-cycle floor on a conflict-free 32-lane 4-byte access
means a *scalar* `LDS` stream gets about half of it — hence the measured
5.2–5.4 TB/s scalar and 9.0–10.0 TB/s vectorised, a ratio of 1.9.

This is the ceiling that decides Part V, and the reason is a ratio M17 computed:
a GEMM at the FP32 ceiling, reading 8 bytes of operand per FMA, would demand
**72 TB/s** from this array. The array delivers 5–10. The gap is the factor
register tiling exists to close, and it closes it not by making shared memory
faster but by reading fewer bytes from it.

**The schedulers.** 4 per SM, one instruction each per clock, 160 device-wide.
Every instruction competes for these slots: FFMA, LDS, LDG, the integer address
arithmetic, the loop counter, the branch. This is why M18's 90.8% FFMA density
is a performance number and not a curiosity, and why M16's naive GEMM — whose
inner loop is `LDG, LDG, IMAD.WIDE, FFMA` — gave away three quarters of the
machine in the instruction stream before anyone looked at a cache.

**What the roofline cannot see, physically.** Warps. The whole model is about
rates, and every rate on a GPU is achieved by having enough independent work in
flight to cover a latency. Little's Law is the missing equation: ~128 kB in
flight for DRAM on this part, which is ~4000 outstanding 32-byte sectors, which
is ~128 fully-coalesced warp loads. If your kernel cannot supply that, no
ceiling in the model is reachable and the model will not tell you.

---

## Code Walkthrough

### `example01.cu` — building the roofline

**Part A** measures five ceilings in two timing groups. The DRAM probe is the
shape you have seen since M11:

```cpp
__global__ void streamRead(const float4 * __restrict__ src, float *sink, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    for (; i < n; i += gridDim.x * (size_t)blockDim.x) {
        float4 v = src[i];
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    if (a.x == 1e30f) sink[0] = a.x + a.y + a.z + a.w;   // never true, never provable
}
```

256 MB of buffer, 5.3x the L2. The guarded store is the standard idiom for
keeping a measurement alive: the compiler cannot prove the branch is dead, so
neither the loads nor the adds can be eliminated.

The shared probe is where the care goes:

```cpp
    float acc = 0.0f;
    int base = (int)threadIdx.x;
    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int u = 0; u < SPROBE_W; ++u) acc += s[(base + u*33) & (SPROBE_N-1)];
        base += 1;                       // <- the entire reason this is honest
    }
```

Three separate hazards are defeated here, and each of them has a module behind
it. Stride 33 makes `gcd(33,32) = 1`, so the 32 lanes of a warp land in 32
distinct banks and there is no conflict to measure (M7). Stride 33 is also not
1, so consecutive reads are not adjacent and `ptxas` cannot contract four of
them into an `LDS.128` (M7 rule, M18 confirmation) — this probe measures the
scalar instruction it claims to measure. And `base += 1` makes every address
depend on the loop index; without it, the whole sum is loop-invariant and the
loads leave the loop.

The compute probe uses eight independent chains so that one warp's
instruction-level parallelism covers the FFMA latency, and unrolls the outer
loop 8 deep:

```cpp
    #pragma unroll 8
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int i = 0; i < CHAINS; ++i) a[i] = fmaf(a[i], b, 1.0f);
    }
```

An earlier draft used `#pragma unroll 1` on the outer loop. Its loop body was
8 FFMAs plus 4 overhead instructions, and it measured **13 750 GFLOP/s** — a
clean, reproducible, entirely fictitious "ceiling" at exactly 8/12 of the truth.
With the 8-deep unroll the body is 68 instructions per 64 FFMAs and the same
kernel measures **18 642**. The number that changed was the instruction mix, and
this is the issue ceiling making itself felt in the one measurement that is
supposed to be about arithmetic.

**Part B** divides the compute ceiling by each memory ceiling to get the ridge
points, and spells out the one that matters:

```
shared, scalar    5224.3 GB/s    ridge 3.57 FLOP/byte
a kernel reading 2 shared floats per FMA offers 0.25 FLOP/byte,
which is 14x below the ridge: 7.0% of the compute ceiling.
```

**Part C** draws it. The plot is ASCII, log–log, with one line per memory level
and known kernels marked at the AI of the level that binds them.

**Part D** runs the hoisted probe and prints the rejection. This is the part to
read twice.

### `example02.cu` — classify, predict, measure, reconcile

Six kernels, each reproduced verbatim from the module that built it, with the
ledger computed in a struct before anything is timed:

```cpp
      { "4 gemmT  (M17)",  kGemmT,
        FL_GEMM, 4.0*((double)sA+sB+sC), 8.0*NMNK, 64.0*16.0/66.0,
        "tiling changed the opcode LDG->LDS, not the 8 B per FMA" },
```

`FL_GEMM` is `2MNK`. The DRAM column is compulsory traffic, `4(MK+KN+MN)`,
because `beta == 0` means C is not read (M16's contract). The request column is
`8MNK` — two 4-byte operands per FMA — and it is *identical to the naive
kernel's*, which is the sentence the whole module exists to make quantitative.
The last number is FLOPs per warp-instruction, read off `cuobjdump -sass`: the
tiled kernel's loop body is 66 instructions containing 16 FFMAs, so
`64 x 16/66 = 15.5`.

The prediction is the `min` over three plateaux, and the measured table is:

```
  kernel       GFLOP/s   predGF/s meas/pred   %FP32  verdict
  1 triad       61.969       68.3    0.9078  0.330%  at roof
  2 reduce      99.062      102.4    0.9674  0.528%  at roof
  3 gemmN     1168.692     1295.7    0.9019  6.232%  at roof
  4 gemmT     1473.771     1295.7    1.1374  7.858%  at roof
  5 gemmR     7805.680     6910.6    1.1295 41.621%  at roof
  6 chase        0.224       12.8    0.0175  0.001%  FAR below
```

Five of six within 14%, from counting. The sixth is the point of §6(b), and the
program closes the loop with Little's Law in the output rather than in prose.

The reconciliation also prints the DRAM traffic each kernel actually achieves,
which is the evidence that the GEMMs are not DRAM bound: gemmT puts **9.0 GB/s**
across the pins, 2.4% of the ceiling, while sitting on its own roof.

---

## Check Your Understanding

**Q1.** A colleague profiles a kernel, computes its compulsory DRAM arithmetic
intensity as 200 FLOP/byte, observes that the machine balance is 45, and
concludes the kernel is compute bound and should be rewritten with faster math.
The kernel runs at 6% of the FP32 ceiling. Without running anything, name the
two distinct things that could be true, say which measurement distinguishes
them, and explain why "compute bound" was never a safe conclusion from that
arithmetic.

**Q2.** Module 17 measured tiled GEMM at 9.4% of the FP32 ceiling. The shared
roofline gives 7.0% for a scalar `LDS` ceiling and 13.0% for an `LDS.128`
ceiling. Explain why the measurement lands between them, and what you would
change in the kernel to move it toward the upper figure. Then explain why that
change is **not** what Module 18 actually did, and why M18's change was better.

**Q3.** A kernel's roofline prediction is 400 GFLOP/s and it measures 520. Your
colleague says the GPU is faster than its specification. Give three
substantively different ways the ledger could be wrong that all produce a
measurement above the roof, and for each, the one additional number you would
compute to test it.

**Q4.** The FP32 ceiling on this GPU is 160 warp-instructions per cycle times
64 FLOP per FFMA. Suppose you write a kernel whose inner loop is one `LDS`
followed by one `FFMA`, repeated, with no address arithmetic at all (imagine the
compiler folds it into the load offsets). Compute its plateau on (a) the issue
axis and (b) the shared-memory axis, using this module's measured ceilings.
Which binds, by how much, and what does that tell you about why M18's `BN/TN >=
16` rule exists?

Answers in `solutions/module21/check_your_understanding.md`.

---

## Exercises

### Exercise 1 — `exercise01.cu` — build the roofline for this GPU

Measure every ceiling the model needs and make the model refuse a measurement
that cannot physically be true.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
exercise01.exe
```

| TODO | what it requires |
|---|---|
| 1 | The DRAM streaming-read probe kernel. 256 MB buffer, grid-stride, nothing eliminable. |
| 2 | The shared-memory read probe kernel. Exactly `SPROBE_W` scalar shared reads per outer iteration, with three separate compiler and hardware hazards defeated. |
| 3 | The FP32 throughput probe. `CHAINS` fused multiply-adds per outer iteration, arranged so the result is a throughput measurement and not a latency or loop-overhead measurement. |
| 4 | **Design.** `hardwareBound()` for all four levels — derived from wire counts, lane counts, bank counts and scheduler counts, never measured. Decide for yourself which of the available clock figures may legitimately appear in a bound. |
| 5 | `rooflineGFLOPs()` and `ridgeFlopPerByte()`. |

The file ships a second, *broken* shared-memory probe with the loop-invariant
address restored. Check 6 requires your TODO 4 bound to reject it. A bound that
is too loose accepts it; a bound that is too tight rejects your own correct
measurement. The program prints a readable roofline when you are done.

Validation: three measurements inside plausible bands and inside your own
bounds, two hashed function checks, and the rejection. `SCORE: 6/6` required for
`OVERALL: PASS`.

### Exercise 2 — `exercise02.cu` — classify and predict

Six kernels you have already written: M11's in-place SAXPY, M12's reduction v6,
M16's naive GEMM, M17's tiled GEMM, M18's register-tiled GEMM, and the dependent
pointer walk from M1/M4. For each: count the bytes and the FLOPs, compute the
arithmetic intensity at each level, predict the bottleneck and the achievable
fraction of peak — then measure and reconcile.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
exercise02.exe
```

| TODO | what it requires |
|---|---|
| 1 | `fillLedger()` — FLOPs, compulsory DRAM bytes and requested operand bytes for all six. Three of the six have `dram == req`; three do not; one requests *fewer* bytes than it moves. |
| 2 | `classify()` — the roofline's own verdict and predicted rate, as a `min` over four plateaux. Watch the units. |
| 3 | `PRED_LEVEL[6]` — what you believe actually limits each kernel. It is **not** required to agree with `classify()`. |
| 4 | `PRED_BUCKET[6]` — predicted fraction of the measured FP32 ceiling, in five buckets. All six must be right. |
| 5 | **Design.** `bytesInFlightNeeded()` and `bytesInFlightSupplied()` — Little's Law for whichever kernel the model fails on. The harness checks that their ratio predicts the measured ratio. |

All predictions are committed before the harness times anything, and the answer
keys live only under `solutions/`. `SCORE: 9/9` required.

### Exercise 3 — `exercise03.cu` — design to a target

A 17x17 convolution over a 2048x2048 image. The shipped naive kernel is correct
and reaches about an eighth of the FP32 ceiling. **Get to 28% of the ceiling and
at least 2.5x the naive kernel.** You are not told which optimization to apply —
derive it from the roofline.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
exercise03.exe
```

| TODO | what it requires |
|---|---|
| 1 | `aiNaive()`, `aiNeeded()`, `minOutputsPerThread()` — the analysis that decides the design. Check the SASS before you assume where the filter weight comes from. |
| 2 | **Design.** `TILE_X`, `TILE_Y`, `OPT`. Check 2 compares `OPT` against what your own TODO 1 demands. |
| 3 | The cooperative halo load. Load mapping is not compute mapping. |
| 4 | The accumulation. The loop order is the exercise. |
| 5 | `PRED_BUCKET` — predicted fraction of the ceiling, committed before running. |

Two warnings, both measured and both stated in the file: one of the obvious
moves buys nothing at all here, and more of the move that does work is not
monotonically better. Validation prefills the output with `+infinity` so an
unwritten pixel cannot pass, checks a sampled exact double reference against a
`gamma_n x S` tolerance (M16), and compares the whole image against the naive
kernel. `SCORE: 9/9` required.

---

## Prediction

Commit to these in writing before you build anything.

**P1.** Example 1 measures the shared-memory ceiling twice: once with the
address depending on the loop counter and once without. The two kernels differ
by a single `base += 1`. Predict the ratio between the two reported bandwidths
to within a factor of two, and predict how many `LDS` instructions appear in
each one's SASS loop body.

**P2.** Example 2 classifies naive GEMM, tiled GEMM and register-tiled GEMM.
Two of the three have the *same* request-level arithmetic intensity. Say which
two, and predict the ratio of their measured throughputs. If your answer is
"1.0", say what else in the model could account for the difference before you
look.

**P3.** Exercise 3's naive convolution reads one 4-byte value and does one
fused multiply-add per tap. Before computing anything, predict what fraction of
the FP32 ceiling it reaches, and predict whether staging the input through
shared memory — the move everyone reaches for first on a stencil — changes that
fraction. Commit to a number for the shared-memory version relative to the naive
one: faster, slower, or the same.

---

## Where this goes next

- **Module 19 (occupancy)** and **Module 20 (latency hiding)** own the two
  mechanisms that put a kernel below its roofline without any ceiling being
  saturated. Everything in §6(b) is a pointer to them.
- **Module 23 (Nsight Compute)** has an automated roofline chart and the counters
  (`dram__bytes.sum`, `l1tex__t_bytes.sum`, `smsp__inst_executed.sum`,
  `sm__sass_thread_inst_executed_op_ffma_pred_on.sum`) that would let you build
  every column of this module's tables without writing a probe. `ncu` fails with
  `ERR_NVGPUCTRPERM` on this machine and cannot be run here, so M23 teaches it as
  theory and gives the command lines. Everything in this module was constructed
  and measured directly instead, which is slower and, for learning the model, is
  arguably better: you cannot misread a counter you had to build.
- **Part XIV (Modules 41–42, CUDA for AI)** applies this to transformer
  workloads, where the single most important fact is a roofline fact: a
  batch-1 decoder matrix-vector product has an arithmetic intensity of about
  0.5 FLOP/byte against a ridge of 45, so inference is a bandwidth problem and
  training is a compute problem, and they want different kernels.
