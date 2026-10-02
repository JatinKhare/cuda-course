# Module 23 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -lineinfo -o e1s.exe exercise01_solution.cu
e1s.exe
```

The program allocates ~1.1 GB on the device (a 512 MB AoS table, a 64 MB packed
column, two 256 MB transpose buffers, three small GEMM matrices). That fits
easily in 12 GB but it is not a toy.

---

## TODO 1 — the diagnosis per kernel

```cpp
static const int DIAG[4] = { 4, 1, 6, 3 };
```

### [1] `k1_strided` → **4, scattered sectors**

The report gives you the two counters and their quotient:

```
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum         524,288
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       16,777,216
  ->  sectors per request                                 32.00
```

**32.00 is the saturated worst case for a 4-byte load.** Module 5 established
why: a warp presents 32 addresses, the L1 tag stage turns each into a 32 B
sector index, and once the byte stride reaches 32 every lane owns a distinct
sector. Here the stride is `K1_REC * 4 = 32 B` exactly. 128 useful bytes per
request, 1024 bytes moved — **12.5% of what crosses the pins is used**.

Confirm the arithmetic against the other rows and it closes exactly:
16,777,216 sectors × 32 B = 536.87 MB = `dram__bytes.sum`, in 1.3137 ms =
408.7 GB/s = the 94.61% the Speed-of-Light section reports.

The decisive observation is that DRAM Throughput is **94.61%** and that is *not*
good news. The bus is saturated moving bytes nobody reads.

### [2] `k2_transpose<0>` → **1, at the DRAM roof; stop**

This is the kernel the exercise exists for.

```
DRAM Throughput                              %           90.71
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  65,011,712
  ->  wavefronts per shared load request                  32.00
```

Sixty-five million bank conflicts. A 32-way conflict, the worst there is.
Module 7 measured that exact conflict in isolation at **14.79× warm and up to
19× cold**. Every instinct says fix it.

And it is worth **nothing**, because the kernel is already at 90.71% of the
432 GB/s pin rate. 536.87 MB of compulsory traffic — read the matrix once,
write it once — at 391.9 GB/s. There is no headroom to recover. The conflict
is entirely hidden behind DRAM latency that the SM has to wait for anyway.

Module 15 measured exactly this: *the identical 32-way conflict M7 measured at
14.79× is worth 0.96–1.03× on a DRAM-bound 8192² transpose, and 2.67–3.25× on
an L2-resident 2048² one.* Same conflict, same code, different binding
constraint.

**The rule this teaches: a counter tells you a defect exists. The Speed-of-Light
section tells you whether the defect is on the critical path. Read them in that
order, not the other way round.**

### [3] `k3_naive` → **6, on-chip operand-fetch bound**

```
Compute (SM) Throughput                      %            7.12
Memory  (DRAM) Throughput                    %            1.78
l1tex__t_sector_hit_rate                     %           93.72
lts__t_sector_hit_rate                       %           97.80
smsp__inst_executed.sum                          inst 135,664,516
sm__sass_thread_inst_executed_op_ffma_pred_on.sum   1,085,316,141
  ->  FFMA warp-instructions / all warp-instructions  %   25.00
```

Neither ceiling is reached. The classical two-axis roofline, given
`AI(DRAM) = 181 FLOP/byte`, calls this kernel compute-bound and is wrong —
Module 21 made that the headline. What binds is **on-chip operand fetch**:
25% FFMA density means three of every four issued instructions are address
arithmetic and loads. Module 18 measured 25% for exactly this SASS shape
(`LDG, LDG, IMAD.WIDE, FFMA`) and 90.8% for the register-tiled version, and
reported that *FFMA density was the only number that tracked performance
throughout the ladder*.

Check the arithmetic: FFMA thread-instructions = M·N·K = 1027·1027·1029 =
1,085,316,141 exactly. Divide by 32 for warp instructions: 33,916,129. Divide
by 135,664,516 total: 25.00%. Multiply the FLOPs out — 2·M·N·K / 1.6460 ms =
**1319 GFLOP/s**, which lands inside the 1275–1348 GFLOP/s this course has
measured for a naive GEMM four separate times.

The high hit rates are *not* the finding. They are the reason DRAM throughput
is only 1.78%; they say the caches are absorbing the re-requests, which they
are, which is why the kernel is not memory-bound and also not fast.

### [4] `k4_serial` → **3, nothing saturated; too few requests in flight**

```
Achieved Occupancy                           %            8.33
DRAM Throughput                              %           15.78
sectors per request                                       4.00
Stall Long Scoreboard                   cycle/inst       286.40
smsp__warps_eligible.avg.per_cycle_active                  0.09
```

Perfect coalescing. No bank conflicts. Nothing above 16% of any ceiling. The
access pattern is flawless and the kernel is 6× too slow.

The pair that settles it is **eligible warps 0.09** with **achieved occupancy
8.33%**. 40 blocks of 128 threads on 40 SMs is one block per SM, 4 warps of the
48 slots. Those 4 warps are resident; 0.09 of them is eligible to issue on an
average cycle; the rest are parked on `long_scoreboard`, i.e. waiting for DRAM.

Module 20 measured this exact configuration — 5120 threads, one scalar load in
flight — at **68.1–68.4 GB/s**, derived the supplied concurrency as 20.0 kB
against a required 120.4 kB, and measured the fixed version at **409.3 GB/s**.
262.14 MB / 3.8450 ms = 68.2 GB/s. The report's duration *is* M20's number.

The launch shape is declared fixed, so the only axis left is **memory-level
parallelism per thread** — Little's Law, not occupancy.

---

## TODO 2 — the decisive metric

```cpp
static const int METRIC[4] = { 3, 1, 7, 6 };
```

| kernel | metric | why this one and not a neighbour |
|---|---|---|
| 1 | `l1tex__t_sectors / l1tex__t_requests` | DRAM throughput is 94.6% for both the broken kernel and the fixed one. Only the ratio distinguishes them. |
| 2 | `dram__throughput...elapsed` | it is the only row that says there is no headroom. Every other row in that section is a defect with nowhere to go. |
| 3 | `sm__sass_...op_ffma_pred_on.sum / smsp__inst_executed.sum` | Compute 7.12% tells you it is slow; the density tells you *why* and what to change. |
| 4 | `smsp__warps_eligible.avg.per_cycle_active` | occupancy 8.33% would also catch it, but occupancy suggests "launch more blocks", which the problem statement forbids. Eligible-warps points at concurrency per thread, which is the available fix. |

---

## TODO 3 — the speedup buckets

```cpp
static const int BUCKET[4] = { 3, 1, 2, 2 };   // 1:<1.5x  2:1.5-6x  3:>6x
```

Reasoning, before any measurement:

1. **8×, give or take.** 32 sectors per request becomes 4. The kernel is at the
   DRAM roof in both cases, so the time is proportional to bytes moved and the
   bytes moved fall 8×. Bucket 3.
2. **~1×.** 90.71% of the pin rate leaves 9% of headroom and the conflict is not
   on the critical path. Bucket 1, and it may be a small *loss* — M17 measured
   a `T+1` pitch destroying the `LDS.128` merge for a 40% loss in a different
   kernel, so padding is not free.
3. **4–5×.** M18's ladder: naive 1302–1345, 1-D register tile TM=8 4278–4465,
   2-D 8×8 7033–7802. A 4×4 tile over a 64×64 block tile sits between the
   first two steps. Bucket 2.
4. **3–6×.** M20 measured 6.0× for MLP 4 × `float4` on this shape. This fix is
   MLP 4 with *scalar* loads, so the supplied concurrency is 81.9 kB against a
   required 120.4 kB — about 0.68 of the requirement, so a little short of the
   roof. Bucket 2.

---

## TODO 4 — writing the fix for kernel 4

```cpp
__global__ void k4_mlp(const float * __restrict__ in, float *partial, size_t n)
{
    size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t nt  = gridDim.x * (size_t)blockDim.x;
    float a = 0.0f;
    #pragma unroll 1
    for (size_t base = gid; base + (K4_MLP-1)*nt < n; base += K4_MLP*nt) {
        float v[K4_MLP];
        #pragma unroll
        for (int k = 0; k < K4_MLP; ++k) v[k] = in[base + (size_t)k*nt];
        #pragma unroll
        for (int k = 0; k < K4_MLP; ++k) a += v[k];
    }
    partial[gid] = a;
}
```

Three things in that are load-bearing.

**The stride between the K4_MLP loads is `nt`, the whole grid's thread count,
not 1.** A thread that loads `in[i], in[i+1], in[i+2], in[i+3]` would have its
warp cover four times as many sectors per instruction, which looks like more
bytes in flight but is really the same 32 lanes spread over 16 sectors —
and it breaks coalescing. Striding by `nt` keeps each of the four loads
perfectly coalesced (4 sectors per request) and makes them four *independent*
requests. The report's `sectors per request = 4.00` stays 4.00.

**The two inner loops must be separate.** If you write

```cpp
for (int k = 0; k < K4_MLP; ++k) a += in[base + k*nt];      // WRONG
```

the accumulation makes each load's consumer depend on the previous load's
result, the compiler cannot issue them together, and you have four dependent
loads instead of four independent ones. The whole point is to issue all four
before consuming any.

**`#pragma unroll 1` on the outer loop** stops `ptxas` from unrolling it and
manufacturing even more MLP than you asked for — which would be *faster* but
would make the experiment measure something other than what the TODO says
(spec §12.11). Note that `k4_serial` carries the same pragma for the same
reason: without it the baseline is not the baseline.

**Bit-identical partials.** `base += K4_MLP*nt` with the inner loop adding
`v[0..3]` in order visits exactly the same elements in exactly the same order
as `k4_serial`'s `i += nt`, so the two float sums are bit-identical and the
validation can compare with `==` rather than a tolerance. That is a stronger
check than a tolerance and it is available here for free; take it when you can.

---

## TODO 5 — the claim that is not evidence

```cpp
static const int REDHERRING = 2;
```

> *"[2] 65,011,712 shared bank conflicts and 32.00 wavefronts per shared load
> request mean removing them is the best available fix here."*

The number is real. The conflict is real and is the worst degree possible. The
*inference* is wrong, because the same section reports 90.71% DRAM throughput,
and a defect that is not on the critical path is not a finding.

Claim 5 is the one most people also suspect. It is valid: Achieved Occupancy
89.90% genuinely does rule out *a shortage of resident warps* as kernel 2's
problem. It does not rule out everything — Module 19 measured that achieved
occupancy is blind to tails and imbalance, and Module 20 measured that resident
is not eligible — but the narrow claim it makes is sound.

---

## Performance reasoning, and what the machine said

Measured on this GPU, all four pairs timed back to back, rotated, min of four
sweeps, validated in a separate pass:

```
   pair                          before ms   after ms   speedup   bucket
   k1 strided -> packed             1.3056     0.1732     7.54x        3
   k2 [32][32] -> [32][33]          1.5217     1.5843     0.96x        1
   k3 naive -> 4x4 reg tile         3.4782     0.9095     3.82x        2
   k4 serial -> 4 in flight         3.9154     1.1733     3.34x        2

   k1 strided vs packed partials differing : 0
   k2 transpose sampled mismatches        : 0
   k3 GEMM sampled failures (gamma_K*S)   : 0
   k4 serial vs 4-in-flight differing     : 0

SCORE: 13/13
OVERALL: PASS
```

Four comments.

**k1 measured 7.54×, not 8.00×.** The packed kernel reads 64 MB in 0.1732 ms =
387.4 GB/s, slightly under the 410 GB/s ceiling because 64 MB is only 1.3× the
48 MB L2 and the launch is short. The strided kernel reads 512 MB in 1.3056 ms
= 411.2 GB/s, dead on the ceiling. The shortfall is in the *fixed* kernel, not
the broken one.

**k2 measured 0.96× — padding made it slower.** That is inside M15's measured
band of 0.96–1.03× for this exact comparison and it is the result the exercise
wants. Padding `[32][32]` to `[32][33]` costs 4 KB of shared memory per block,
costs a multiply-add in every index, and buys back a conflict that was hidden.
If you predicted bucket 1 you were right for the right reason; if you predicted
bucket 2 or 3 you trusted a counter over the Speed-of-Light section.

**k3's baseline at 3.4782 ms is slower than the report's 1.6460 ms.** Both are
real: the report is an isolated profile of the naive GEMM; the harness times it
inside a rotated sweep immediately after a 2048-block transpose, in a session
that has been running GEMMs for a minute. The *ratio* is what is scored, and the
ratio is measured inside one sweep (spec §12.5b).

**k2's in-session throughput was 81.7% of peak, not the 90.71% the constructed
report shows.** 536.87 MB in 1.5217 ms = 352.8 GB/s. Both numbers are inside
this course's documented session-to-session spread for the same binary
(279–411 GB/s, cross-module index §6b). The diagnosis is unchanged: at 82% of
the pin rate there is still nothing for a bank-conflict fix to recover, which
is exactly what the 0.96× measurement shows.

---

## Expected output

The report and menus, then the four pairs above, then:

```
SCORE: 13/13
OVERALL: PASS
```

Timings vary. Across runs in one session the four speedups measured
7.5–7.8× / 0.96–1.00× / 3.8–4.1× / 2.1–3.3×. Every bucket assignment was
stable; the bucket edges at 1.5 and 6.0 sit in the two widest empty gaps of
that distribution.

---

## The result that matters

**A profiler counter is evidence that a defect exists. It is never, by itself,
evidence that fixing the defect will make the kernel faster.** Kernel 2 has the
loudest number in the whole report — sixty-five million bank conflicts, 32
wavefronts per request, the worst degree the hardware can produce — and
removing it measured **0.96×**. Kernel 4 has a *flawless* memory access pattern,
four sectors per request, zero conflicts, and it is six times slower than it
should be.

The discipline that separates them is the one in §7 of the lesson: Speed of
Light first, and ask "is any ceiling reached?" before opening any other section.
If one is, the only fixes worth considering are the ones that reduce the work
that ceiling has to do. If none is, stop reading the roofline entirely and go to
eligible warps.

**A variation worth doing:** rerun the k2 pair at `TW = 2048` instead of 8192.
The matrix becomes 16 MB, fits in the 48 MB L2, DRAM stops being the
constraint — and the same padding that measured 0.96× here should measure
2.7–3.3× (M15's number). Nothing about the kernel changed. The *binding
ceiling* changed, and with it the entire optimization plan.
