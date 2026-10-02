# Module 23 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -lineinfo -o e3s.exe exercise03_solution.cu
e3s.exe
```

The solution file implements all four fixes so that the measured payoff of each
one is on the record. The shipped exercise asks you for only two of them.

---

## The kernel and its four defects

```cpp
__global__ void e3_base(const float * __restrict__ img,
                        const float * __restrict__ lut,
                        float * __restrict__ out,
                        size_t n, float mean, float sd)
{
    __shared__ float s[LUTD*LUTD];
    for (int t = threadIdx.x; t < LUTD*LUTD; t += blockDim.x) s[t] = lut[t];
    __syncthreads();

    const int lane = threadIdx.x & 31u;
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t str = gridDim.x * blockDim.x;

    #pragma unroll 1
    for (size_t i = gid; i < n; i += str) {
        int   k = (int)((i >> 7) & (LUTD - 1));     // warp-uniform, loop-varying
        float w = s[lane*LUTD + k];                 // B: degree 32
        float v = img[E3_REC*i];                    // A: 16 sectors/request
        out[i]  = ((v - mean) / sd) * w;            // C: IEEE divide
    }
}
```

launched `<<<20, 128>>>` — **D**: 2560 threads, one global load outstanding
each, and **half the SMs are given nothing at all**.

All four are genuine. Each one would be flagged by a different section of a real
`ncu` report, and each one has a textbook fix. The exercise is about which of
them is on the critical path.

### Why `#pragma unroll 1` is in the baseline

Without it, `ptxas` unrolls the grid-stride loop and hoists several `LDG`s,
manufacturing exactly the memory-level parallelism defect D is supposed to be
missing. In an earlier draft of this file the baseline ran at **232 GB/s**
instead of **131**, and the whole exercise collapsed to a 1.7× headline.
Spec §12.11 in one sentence: *the compiler will vectorize and eliminate your
accesses out from under your analysis.* Each single-fix variant carries the same
pragma so that exactly one thing differs between any two of them.

---

## TODO 1 — the ranking

```cpp
static const int RANK[4] = { 1, 3, 2, 4 };
```

**Position 1 is CONCURRENCY and it is not close.** The evidence is in the
Scheduler Statistics section, not in the Memory section:

```
 Section: Scheduler Statistics
   Theoretical Active Warps Per Scheduler                    12.00
   Active Warps Per Scheduler                                 1.00
   Eligible Warps Per Scheduler                               0.11
   Issued Warp Per Scheduler                                  0.10
   No Eligible                                  %           89.40
```

Read the chain **resident ≥ eligible ≥ issued**: 1.00 resident of a possible
12.00, of which 0.11 is eligible on an average cycle, of which 0.10 issues. The
schedulers have nothing to do for 89.4% of the cycles the SM is active. Nothing
in the Memory Workload section can change that number, because the warps are not
stalled on *bandwidth* — the Warp State section says they are stalled on
`long_scoreboard` for **7.90 of the 10.00 cycles per issued instruction**, which
is DRAM latency, and the only cures for latency are more warps or more requests
per warp.

Cross-check the occupancy rows and the picture closes:

```
   Theoretical Occupancy                        %          100.00
   Achieved Occupancy                           %            8.33
   sm__warps_active.avg.pct_of_peak_sustained_elapsed  %     4.17
```

Theoretical 100% and achieved 8.33% means the grid, not the resources, is the
limit — the four `launch__occupancy_limit_*` rows are 16/20/12/24, so twelve
blocks per SM would fit and twenty blocks total were launched. And look at the
pair: **8.33 against 4.17, a factor of exactly two.** That is Module 19's
denominator difference making twenty idle SMs visible. `_active` averages over
the SMs that ran something and therefore cannot see them; `_elapsed` divides by
the whole machine and halves. If `ncu` had shown you only its default
"Achieved Occupancy", the twenty dead SMs would be invisible.

**Positions 2, 3 and 4 are, honestly, a guess**, and the harness says so. See
the measurement below.

---

## TODO 2 — the bucket

```cpp
static const int BUCKET_TOP = 2;      // 1.5x to 6x
```

Reasoning before running. The base moves 335.54 MB (268.44 read + 67.11 written)
in 2.5672 ms = **130.7 GB/s**. The ceiling this course has measured repeatedly is
**410.5–410.7 GB/s**. So the absolute maximum available from any fix is
410.7/130.7 = **3.14×**, and no combination of all four fixes can exceed it.
That ceiling calculation is the single most useful thing you can do before
writing any code: it tells you the bucket without implementing anything.

Bucket 3 (>6×) is **arithmetically impossible** here. Bucket 1 (<1.5×) would
require the concurrency fix to recover less than half of a 3.1× gap when the
gap is entirely latency-shaped. Bucket 2.

---

## TODO 3 — the concurrency fix

```cpp
__global__ void e3_fixConcurrency(...)   // launched <<<640, 256>>>
{
    /* ...same LUT staging, same lane, same gid/str... */
    size_t i = gid;
    for (; i + (E3_MLP-1)*str < n; i += E3_MLP*str) {
        float v[E3_MLP];
        #pragma unroll
        for (int q = 0; q < E3_MLP; ++q) v[q] = img[E3_REC*(i + (size_t)q*str)];
        #pragma unroll
        for (int q = 0; q < E3_MLP; ++q) {
            size_t j = i + (size_t)q*str;
            int    k = (int)((j >> 7) & (LUTD - 1));
            out[j] = ((v[q] - mean) / sd) * s[lane*LUTD + k];
        }
    }
    for (; i < n; i += str) { /* ...tail, one at a time... */ }
}
```

**Two changes, and both are needed.**

1. **Grid 20×128 → 640×256.** 2560 threads become 163,840: 64× more warps, and
   all 40 SMs get work.
2. **Four independent loads before any is consumed.** The two inner loops must
   be separate; fusing them into one makes each load's consumer depend on the
   previous load and you are back to one request in flight per thread.

The stride between a thread's four loads is `str`, the whole grid's thread
count — not 1. Striding by 1 would make a warp's single `LDG` cover four times
as many sectors, which is not four independent requests, and it would destroy
the coalescing that `sectors per request` already reports as healthy at the
record level.

Keep the divide and keep the conflict. The exercise is about isolating one
change.

**The tail loop matters.** `n = 16,777,216` and `str = 163,840`: the main loop
stops when fewer than `4*str` elements remain, and the remainder must still be
written. Drop the tail loop and the validation pass catches it — the output
buffer is prefilled with `0xff` (a NaN pattern) before each timed variant
specifically so an unwritten element cannot pass.

---

## TODO 4 — the divide fix (ranked last)

```cpp
    out[i] = ((img[E3_REC*i] - mean) * rsd) * s[lane*LUTD + k];   // rsd = 1.0f/sd
```

`rsd` is computed once on the host. Note that this is **not** the same as
`-use_fast_math` or `__fdividef`: it is exact, it is one FFMA-class operation
instead of a multi-instruction IEEE division sequence, and it is the fix the
`sm__inst_executed_pipe_xu` row is pointing at.

It is also, measurably, worth 1.15×.

---

## TODO 5 — evidence and misdirection

```cpp
static const int EVIDENCE = 1;   // Eligible Warps Per Scheduler = 0.11
static const int MISLEAD  = 3;   // 16,252,928 shared bank conflicts
```

**Evidence (row 1).** Eligible warps near zero is the one number in the report
that identifies the *binding* constraint rather than merely a defect. It says:
the schedulers are idle not because the memory system is saturated (DRAM is at
30%) and not because the math pipes are full (4.9% compute), but because there
is nothing ready to issue. That is latency × insufficient concurrency, and it
names the axis of the fix.

**Misdirection (row 3).** 16,252,928 bank conflicts, 32.00 wavefronts per shared
load request. The worst degree the hardware can produce, and the most impressive
number on the page. Removing it entirely measures **1.07×**.

Two near-misses worth discussing:

- **Row 2, achieved occupancy 8.33%.** This is real and it is *half* of the
  story — it tells you the grid is too small but says nothing about loads in
  flight per thread. A reader who acts on row 2 alone enlarges the grid and gets
  most of the win; the MLP half is the remainder. It is a defensible second
  choice and it is the reason the harness scores row 1 rather than treating the
  two as interchangeable.
- **Row 6, DRAM throughput 30.25%.** It tells you the memory system has headroom.
  It does not tell you why you are not using it, and a reader who concludes
  "not memory bound, so compute bound" from it walks straight into the "both
  low" quadrant that §7 of the lesson exists to warn about.

---

## Performance reasoning, and what the machine said

```
-- measured, all five timed back to back and rotated ------------------
   variant                          ms    speedup   DRAM GB/s
   e3_base                      2.5682      1.00x    130.7 GB/s
   F1 CONCURRENCY               0.8409      3.05x    399.0 GB/s
   F2 LAYOUT                    2.5245      1.02x     53.2 GB/s
   F3 CONFLICTS                 2.3985      1.07x    139.9 GB/s
   F4 DIVIDE                    2.2354      1.15x    150.1 GB/s

   e3_base                sampled mismatches: 0
   F1 CONCURRENCY         sampled mismatches: 0
   F2 LAYOUT              sampled mismatches: 0
   F3 CONFLICTS           sampled mismatches: 0
   F4 DIVIDE              sampled mismatches: 0

SCORE: 9/9
OVERALL: PASS
```

Reproduced across three runs: **3.06 / 3.05 / 3.02×** for F1 and
**1.14 / 1.15 / 1.14×** for F4. The bucket edge at 1.5× sits in the middle of
the widest empty gap in that distribution, which is what spec §12.5d asks for.

**F1 reached 399.0 GB/s — 97% of this course's measured 410.5 GB/s ceiling, and
3.05× of the predicted 3.14× maximum.** The pre-run ceiling calculation was
right to within 3%.

### The F2 row is the most interesting one in the table

```
   F2 LAYOUT                    2.5245      1.02x     53.2 GB/s
```

The layout fix reduced the bytes the kernel moves from 335.54 MB to 134.22 MB —
**a 2.5× reduction in DRAM traffic** — and changed the runtime by **2%**. Read
the throughput column: the kernel's achieved DRAM bandwidth *fell* from
130.7 GB/s to 53.2 GB/s, because it is now doing the same latency-bound work
with less data per request.

That is the lesson of the exercise in one line. **The kernel was never
bandwidth-bound, so removing bandwidth demand bought nothing.** M21 said it
first: measured/predicted below 0.25 means nothing is saturated and the roofline
is the wrong model; go to Little's Law. Here `130.7 / 410.5 = 0.32` — close to
the boundary, and the Scheduler section settles it.

There is a second-order effect worth naming: the AoS read's 16 sectors per
request is also, perversely, *helping* the baseline. A warp's single `LDG` over
a 16 B stride covers 16 distinct sectors = 512 B in flight per request; the SoA
version covers 4 sectors = 128 B. With only one request outstanding per thread,
the AoS version has 4× more bytes in flight. Module 21 measured the same effect
on the pointer chase and used it to reconcile that kernel's effective latency
(280–288 cycles) with M4's single-thread figure of 575. Fixing the layout
without fixing the concurrency makes the memory-level parallelism *worse*.

### F3 and F4

1.07× and 1.15×. Both are real fixes to real defects. Both are a rounding error
against 3.05×. Their relative order is inside their own run-to-run spread
(F3 measured 1.06–1.07×, F4 1.14–1.15×, and with the operating point shifted
they overlap), which is why the harness refuses to score positions 2 and 3 of
the ranking. Pretending to score a distinction narrower than the noise is worse
than admitting it cannot be scored.

---

## Expected output

The constructed report, the two menus, the five-row table above, five
`sampled mismatches: 0` lines, and:

```
SCORE: 9/9
OVERALL: PASS
```

The absolute milliseconds vary with the operating point. During authoring this
GPU spent several minutes pinned in P8 (210 MHz SM, 405 MHz memory, 30 W cap),
during which a plain streaming read measured **14.1 GB/s** instead of 410.8.
Every scored quantity in this exercise is a ratio formed inside one rotated
sweep, which spec §12.5b exempts from the operating-point guard — and the
exercise did in fact score 9/9 from that degraded state as well as from a
healthy one.

---

## The result that matters

**Four real defects, one binding constraint, and the loudest counter in the
report is attached to the defect worth the least.** Rank an optimization plan by
*which ceiling is binding*, never by which counter is largest, and the ordering
falls out before you write a line of code.

The procedure, in four steps, none of which requires implementing anything:

1. Compute the headroom. Bytes moved / duration against 410.5 GB/s: 130.7 of
   410.5, so **nothing can be better than 3.14×.** That fixes the bucket.
2. Ask which ceiling is binding. DRAM 30%, compute 5%, eligible warps 0.11 of
   12 — **none of them.** So the model is Little's Law, not the roofline.
3. The only fixes that move a latency-bound kernel are the ones that increase
   work in flight. Exactly one of the four candidates does.
4. Everything else is real, and goes in the backlog, and is worth doing *after*
   the binding constraint moves — because the ranking changes once it does.

**A variation worth doing, and it closes the loop on step 4:** apply F2 *on top
of* F1 and re-measure. Once the kernel is near the roof it is genuinely
bandwidth-bound, and the layout fix — the one that bought **1.02×** on the
baseline and was ranked near the bottom — becomes the best fix available. It was
checked while writing these notes:

```
F1       1.0661 ms  314.7 GB/s
F1+F2    0.4668 ms  287.5 GB/s   speedup over F1 = 2.28x
```

**The same change, same kernel, same machine: 1.02× before the concurrency fix
and 2.28× after it.** (That run sat at a somewhat depressed operating point —
F1 at 314.7 GB/s rather than 399 — so the ratio is if anything understated.)

The optimization plan is not a fixed list. It is a list that must be re-derived
every time the binding constraint moves, and the first fix you apply is the one
most likely to move it.
