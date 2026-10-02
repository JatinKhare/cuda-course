# Module 21 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
exercise01_solution.exe
```

Warning-clean. `SCORE: 6/6`, `OVERALL: PASS`. The run takes about 25 s, most of
it the two warm-ups.

---

## TODO 1 — the DRAM streaming-read probe

```cpp
__global__ void dramProbe(const float4 * __restrict__ src, float *sink, size_t n)
{
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    float4 a = make_float4(0.f,0.f,0.f,0.f);
    for (; i < n; i += gridDim.x*(size_t)blockDim.x) {
        float4 v = src[i];
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    if (a.x == 1e30f) sink[0] = a.x + a.y + a.z + a.w;
}
```

Three things are load-bearing.

- **`float4`.** With scalar `float` the loop is four times as many memory
  instructions for the same sectors (M5, M11). On a 640x256 launch with
  12 blocks/SM that still saturates, but the margin is smaller and the
  measurement becomes sensitive to the instruction mix rather than the pins.
- **Grid-stride.** The grid is a free parameter and the traversal stays
  contiguous per warp. M3 and M11 both make this argument; M11 adds the
  correction that grid-stride is about correctness under any launch shape, not
  about speed.
- **The unreachable store.** Without it the whole loop is dead code and the
  kernel measures nothing. `a.x == 1e30f` is never true and the compiler cannot
  prove it, which is exactly the property wanted.

Common wrong approaches:

| mistake | symptom |
|---|---|
| no sink, or `if (threadIdx.x == 0) sink[0] = 0.f;` | reports tens of TB/s; the loads were eliminated. Your TODO 4 bound catches it. |
| buffer sized 32 MB "because that is obviously large" | reports >432 GB/s; you measured L2 (M4's hazard, spec §12 rule 7) |
| one thread per element, no loop | fine, actually — M11 measured the 1:1 mapping *fastest* for a bare copy (395.2 GB/s) |

## TODO 2 — the shared-memory read probe

```cpp
    float acc = 0.0f;
    int base = (int)threadIdx.x;
    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int u = 0; u < SPROBE_W; ++u) acc += s[(base + u*33) & (SPROBE_N-1)];
        base += 1;
    }
```

This is the TODO the exercise exists for, and all three hazards it names are
real and all three were found by earlier modules.

**(a) Bank conflicts.** `bank = (addr/4) % 32` (M7). Lane `l` at unroll step `u`
reads element `l + 33u`, so across the 32 lanes of a warp at fixed `u` the bank
is `(l + 33u) mod 32`, a permutation. Degree 1. Had you used stride 32 you would
have measured a 32-way conflict and reported roughly a thirtieth of the array's
bandwidth, with nothing in the output to tell you.

**(b) Vector contraction.** If consecutive source-level reads are *adjacent*,
`ptxas` merges four of them into one `LDS.128`. M7 measured 32 adjacent scalar
reads becoming 8 `LDS.128`; M18 confirmed it twice. The probe would then report
the vector bandwidth while the harness divides by a scalar instruction count.
Stride 33 is non-adjacent, so the merge cannot happen. Verified:

```
cuobjdump -sass : loop body 133 instructions, 32 of them scalar LDS
```

**(c) Loop-invariant hoisting.** This is the one that produced M17's 40 TB/s.
Without `base += 1` every address is a function of `threadIdx.x` alone, the
entire 32-term sum is loop-invariant, and `ptxas` computes it once:

```
honest  loop body: 133 instructions, 32 LDS
hoisted loop body:  35 instructions,  0 LDS
```

Thirty-five instructions, none of them a shared-memory access, timed and
divided by a shared-byte count. The reported number is 32–36 TB/s.

`#pragma unroll 1` on the outer loop is deliberate (M7 introduced it as an
anti-optimization instrument, M11 used it the same way): without it the compiler
may unroll `t` and start merging across iterations again.

## TODO 3 — the FP32 throughput probe

```cpp
    #pragma unroll 8
    for (int t = 0; t < iters; ++t) {
        #pragma unroll
        for (int i = 0; i < CHAINS; ++i) a[i] = fmaf(a[i], b, 1.0f);
    }
```

Two independent requirements.

**ILP.** `CHAINS = 8` independent accumulators. A single chain measures FFMA
*latency*: each instruction waits for the previous one. Eight chains in flight
means the scheduler always has an eligible instruction from this warp alone,
which is M1's point about ILP as a substitute for occupancy, and M11's measured
2x at 1 block/SM.

**Loop overhead.** This is the subtle half, and it is a measured result rather
than a prediction. With `#pragma unroll 1` the SASS loop body is 8 `FFMA` plus
`IADD3`, `ISETP`, `BRA` and a `MOV` — 12 instructions for 8 FFMAs. Since every
instruction costs an issue slot, the measurement caps at 8/12 = 67% of the real
ceiling:

| outer loop | SASS loop body | measured |
|---|---|---|
| `#pragma unroll 1` | 12 instructions, 8 FFMA | **13 750 GFLOP/s** |
| `#pragma unroll 8` | 68 instructions, 64 FFMA | **18 642 GFLOP/s** |

Both numbers are reproducible. The first is a measurement of a loop; the second
is a measurement of a pipeline. Note that even the second is 94% of the lane
peak at the implied clock, because 4 of every 68 slots still go to the loop.
There is no kernel on this machine that reaches 100% of the arithmetic peak,
and the reason is the issue ceiling.

## TODO 4 — the hardware bounds (design)

```cpp
static double hardwareBound(int level)
{
    const double f = SM_CLOCK_MAX_GHZ;              // 3.105, from nvidia-smi
    switch (level) {
        case LEVEL_DRAM:   return DRAM_PIN_PEAK_GBS;                        // 432.0
        case LEVEL_SHARED: return SM_COUNT * (double)SMEM_BANKS * SMEM_BANK_BYTES * f;
        case LEVEL_FP32:   return SM_COUNT * (double)FP32_LANES_PER_SM * 2.0 * f;
        case LEVEL_ISSUE:  return SM_COUNT * (double)SCHEDULERS_PER_SM * f;
        default:           return -1.0;
    }
}
```

giving 432.0 GB/s, 15 897.6 GB/s, 31 795.2 GFLOP/s and 496.8 G warp-instr/s.

**The design decision is the clock.** Four candidates are in scope and only one
of them is admissible:

| candidate | value | admissible? |
|---|---|---|
| `cudaDevAttrClockRate` | 1.545 GHz | **no.** It is neither the boost clock nor an upper bound. Spec §12 rule 6; a "peak" built on it is 15 821 GFLOP/s, which the measurement exceeds by 1.15x |
| the clock implied by the FFMA measurement | ~1.8–1.9 GHz | no. It is a *measurement*, and a bound derived from a measurement cannot police that measurement |
| the highest clock observed in this course | 2.04 GHz | **no**, and this is the trap. It produces a shared bound of 10 445 GB/s and an issue bound of 326.4 G/s, and a good run of this very program measures **329.6** G warp-instructions/s — a correct measurement rejected by a too-tight bound. Observed during authoring |
| `nvidia-smi --query-gpu=clocks.max.sm` | 3.105 GHz | **yes.** The device's own stated maximum |

A bound is a statement about what the silicon can do, not about what it has been
seen doing. The 3.105 GHz bound is loose — the measurements land at 33–63% of
it — and it is still enough to reject the hoisted probe at 216–332%.

The pin rate does not scale with the SM clock at all, which is why the DRAM row
is a constant and why the power manager can run the memory at 8801 MHz while
the SM sits at 285 MHz (M16).

## TODO 5 — the roofline

```cpp
static double rooflineGFLOPs(double ai, double bwGBs, double peakGF)
{ double mem = ai*bwGBs; return (mem < peakGF) ? mem : peakGF; }

static double ridgeFlopPerByte(double peakGF, double bwGBs)
{ return peakGF / bwGBs; }
```

The only trap is units, and it is a pleasant one: GB/s times FLOP/byte is
already GFLOP/s, because the 1e9 appears on both sides. No conversion factor.
Readers who insert one get a model that is off by 1e9 and immediately obvious.

---

## Synchronization / memory reasoning

Only one barrier appears, in the shared probes: `__syncthreads()` after the
cooperative fill of `s[]` and before the read loop. It is a RAW hazard across
threads and needs both of M9's guarantees. It is outside divergent control flow
(the fill loop is a block-uniform `for`), so the uniformity rule is satisfied.

No atomics, no fences, no cross-block communication anywhere in this exercise.

---

## Performance reasoning

The two timing groups are the methodological content. Spec §12 rule 1 says to
time *competing* configurations back to back; the DRAM ceiling and the FP32
ceiling are not competitors, and they need opposite warm-ups. Putting all
probes in one sweep after a combined warm-up measures:

```
DRAM 326.0 GB/s      FP32 13 750 GFLOP/s     (one sweep, both groups mixed,
                                              and the compute probe under-unrolled)
```

Splitting into `{1500 ms stream -> off-chip probes}` then
`{500 ms compute -> on-chip probes}` measures:

```
DRAM 410.8 GB/s      FP32 18 642–19 847 GFLOP/s
```

Both of those reproduce the cross-module index (410.5–410.7 GB/s and
17 787–18 256 GFLOP/s) in the same run.

---

## Expected output

Real output, one good run:

```
=== Module 21 / Exercise 1 - build the roofline ===

-- warming 1500 ms streaming ----------------------------------------
-- warming 500 ms compute -------------------------------------------

-- measured, each against your TODO 4 bound --------------------------
  probe                          measured        bound     frac verdict
  DRAM read                       410.8 GB/s          432.0    95.1%  accepted
  shared read (yours)            5451.6 GB/s        15897.6    34.3%  accepted
  shared read (broken)          35783.5 GB/s        15897.6   225.1%  REJECTED - impossible
  FP32 FFMA                     19334.5 GFLOP/s     31795.2    60.8%  accepted
  issue rate (lower bd)           302.1 Ginstr/s      496.8    60.8%  accepted

-- your roofline ----------------------------------------------------
   '=' DRAM                  410.8 GB/s   ridge   47.06 FLOP/byte
   '-' L2 (recorded)        1305.0 GB/s   ridge   14.82 FLOP/byte
   '.' shared, scalar       5451.6 GB/s   ridge    3.55 FLOP/byte

  machine balance (DRAM ridge) = 47.06 FLOP/byte
  a kernel reading 2 shared floats per FMA offers 0.25 FLOP/byte and is
  therefore capped at 7.0% of the compute ceiling.

-- scoring -----------------------------------------------------------
  [x] 1. DRAM probe in 200..432 GB/s and inside its bound  (410.8)
  [x] 2. shared probe in 2000..12000 GB/s and inside its bound (5451.6)
  [x] 3. FFMA probe above 8000 GFLOP/s and inside its bound (19334.5)
  [x] 4. hardwareBound() matches the reference
  [x] 5. rooflineGFLOPs()/ridgeFlopPerByte() match the reference
  [x] 6. your bound REJECTS the hoisted probe (35783.5 vs 15897.6)

SCORE: 6/6
OVERALL: PASS
```

Observed ranges over several runs on this machine:

| quantity | range |
|---|---|
| DRAM read | 279 – 411 GB/s (thermal; the high end is the real ceiling) |
| shared, scalar `LDS` | 4777 – 5452 GB/s |
| FP32 FFMA | 17 123 – 19 847 GFLOP/s |
| implied SM clock | 1.67 – 1.93 GHz |
| hoisted / honest shared probe | 6.5 – 6.9x |
| hoisted probe vs its bound | 216% – 332% |

The DRAM spread is the documented session-to-session variability (index §6b):
same binary, same data, 292–411 GB/s depending on P-state and thermals.

---

## The result that matters

A microbenchmark is a number until you have bounded it. The hoisted and honest
shared-memory probes differ by one statement, produce identical-looking source,
and report 5.4 TB/s and 35 TB/s; the only thing in the program that
distinguishes them is a bound computed from bank counts and a clock, and the
only thing in the SASS that distinguishes them is which side of the back edge
32 `LDS` instructions sit on. Module 17 shipped the broken version first and
published 40 TB/s internally before catching it. Build the bound before you
build the benchmark.

**Variation to try.** Change the stride in the shared probe from 33 to 32 and
re-run. The measurement drops by roughly an order of magnitude and stays inside
the bound, so nothing flags it — you have silently started measuring a 32-way
bank conflict and calling it bandwidth. Then change it to 1 and look at the
SASS: 32 `LDS` become 8 `LDS.128`, the measurement roughly doubles, and the
harness's byte count (which assumes scalar reads) is now measuring the wrong
instruction. A bound catches the impossible; only the SASS catches the merely
wrong.
