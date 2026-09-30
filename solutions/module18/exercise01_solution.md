# Module 18 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

Useful while working:

```
nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise01_solution.exe exercise01_solution.cu
nvcc -arch=sm_89 -O3 -cubin -o k.cubin exercise01_solution.cu
cuobjdump -sass k.cubin
compute-sanitizer --tool memcheck .\exercise01_solution.exe
```

Tile hierarchy: `BM = 128, BN = 64, BK = 8, TM = 8, TN = 4`, `NT = 256` threads,
A-tile pitch `AP = BM + 4 = 132` floats.

**Every number in these notes was measured on the RTX 3500 Ada under the
module's warm-up discipline (1500 ms streaming, then 500 ms compute, rotated
min-of-4). Several of them contradict what the obvious analysis predicts; those
are called out rather than smoothed away.**

---

## TODO 1 — locating the thread and the block

```cpp
const int rowBase = blockIdx.y * BM;
const int colBase = blockIdx.x * BN;
const int tRow    = tid / (BN/TN);     // 0..15
const int tCol    = tid % (BN/TN);     // 0..15
```

Blocks are launched 1-D with `NT = 256` threads, so by Module 3's linearization
rule a warp is 32 consecutive values of `tid`. With `tCol = tid % 16`, a warp
covers all sixteen values of `tCol` and two values of `tRow`.

**This is the trap of the exercise, and it costs 1.4×.** Write
`tRow = tid % 16` instead and two things get worse at once.

1. **The stores.** Each store instruction now presents 16 addresses
   `TM·ldc·4 = 8·2053·4 = 65 696` bytes apart: 32 sectors for 128 bytes of
   payload, against 4 sectors in the correct version.
2. **The shared reads** — the larger effect. With `tRow = tid % 16` a warp
   presents 16 distinct `tRow` values to `As[kk][tRow·TM + i]`, whose addresses
   are `TM·4 = 32` bytes apart. An `LDS.128` is phase-split into 4 phases of 8
   lanes (Module 7); within one phase the eight addresses land on bank quads
   0–3, 8–11, 16–19, 24–27, 0–3, … — **a 2-way conflict**. On a 4-byte access
   Module 7's `max(2, D)` law makes a 2-way conflict free; on a 16-byte access
   it is not, because a phase of 8 lanes already asks for the bank array's full
   128 bytes per cycle (lesson §8). Meanwhile `Bs` degenerates to a broadcast.
   Counting phases × degree per value of `k`:

```
correct  (tCol fast):  A 2 x LDS.128 x 4 phases x D=1  +  B 1 x 4 x 1  = 12
swapped  (tRow fast):  A 2 x LDS.128 x 4 phases x D=2  +  B 1 x 4 x 1  = 20
```

**Measured: 5682 GFLOP/s against 7903, a ratio of 1.39×** (the cycle model
predicts 20/12 = 1.67×), which lands at 3.68× over the baseline and **fails the
4.2× gate**. Every validation check passes on both versions; the output is
bit-identical. This is Module 3 Exercise 1's axis trap for the third time in the
course, now acting through a mechanism — `LDS.128` bank phases — that did not
exist when Module 3 planted it.

**`blockIdx.y` carries M and `blockIdx.x` carries N** for the reason Module 16
Exercise 2 established: x is the fast axis, and N is the axis along which C and
B want contiguity. The kernel's decision and the harness's
`dim3 gr((N+BN-1)/BN, (M+BM-1)/BM)` have to agree.

---

## TODO 2 — staging the tiles

```cpp
#pragma unroll
for (int u = 0; u < (BM*BK)/NT; ++u) {          // 1024/256 = 4
    const int idx = tid + u*NT;
    const int r = idx / BK, c = idx % BK;
    As[c][r] = ((rowBase + r) < M && (kt + c) < K)
             ? A[(size_t)(rowBase + r) * K + kt + c] : 0.0f;
}
#pragma unroll
for (int u = 0; u < (BK*BN)/NT; ++u) {          // 512/256 = 2
    const int idx = tid + u*NT;
    const int k = idx / BN, n = idx % BN;
    Bs[k][n] = ((kt + k) < K && (colBase + n) < N)
             ? B[(size_t)(kt + k) * N + colBase + n] : 0.0f;
}
```

### The mapping — and a null result worth more than the penalty would have been

`r = idx / BK, c = idx % BK` makes consecutive thread ids walk `c`, which is
`k`, which is A's contiguous axis. Eight lanes cover one row's eight consecutive
floats — 32 bytes, exactly one sector, fully used — and a warp asks for four
sectors per instruction.

The natural-looking alternative, `r = idx % BM, c = idx / BM`, makes consecutive
lanes walk *rows*, whose addresses are `K·4 = 3076` bytes apart: **32 sectors
instead of 4 for the same bytes.** Module 5's procedure applied unchanged.

**Measured: 7741 GFLOP/s against 7903 — a ratio of 0.98×.** The sector analysis
is correct and the effect is invisible. Two reasons, both of which are the
point:

- **The global path is 18 instructions out of roughly 300 per k-tile**: 6
  `LDG.E.CONSTANT`, 6 `IMAD.WIDE` and 6 `STS`, against 256 `FFMA` and 24
  `LDS.128`. Making 2 % of the instruction stream eight times more expensive in
  *sectors* does not move a kernel whose clock is set by the FP32 pipes. This is
  the block tile doing exactly its job — `BM·BN/(BM+BN) = 42.7` FMAs per global
  load means the global path is not the bottleneck — and it is Module 16's
  argument running in reverse.
- **The swap also improves the other side.** With `r = idx % BM`, consecutive
  `tid` walk `r`, so the transposed store `As[c][r]` hits 32 distinct banks
  (D = 1) instead of the 8-way conflict the `+4` padding exists to repair. One
  side got worse, the other got better.

The transferable discipline: apply the sector count, **then check whether the
instruction it applies to is a measurable fraction of the kernel.** In Module 16
that check came out yes and the mapping was worth 3–8×; here it comes out no.
Write the good mapping anyway — `BK = 8` is not the only tile depth you will
ever use, and at `BK = 32` the loader is four times as much of the kernel — but
do not claim a speedup you have not measured.

### The three guards, and what the third one actually protects

| guard | why | what happens without it |
|---|---|---|
| `rowBase + r < M` | 1027 = 8·128 + 3, the last block row is partial | reads past the end of A |
| `colBase + n < N` | 2053 = 32·64 + 5, the last block column is partial | reads past the end of B |
| `kt + c < K` | 769 = 96·8 + 1, the last **k-tile** is partial | **correct answer, illegal memory access** |

The third row does not say what you expect.

Delete the `(kt + c) < K` term and the kernel still produces the right matrix:
all four shapes PASS, `Freivalds 0.000777`, identical to the guarded version.
The reason is that the **B-side** guard is still there. For `kt + k >= K` the
loader writes `Bs[k][n] = 0.0f`, and the inner product forms
`As[kk][...] · Bs[kk][...]` — garbage × 0 = 0. The junk read from A never
reaches an accumulator. **The K guard on A is arithmetically redundant given the
K guard on B.**

What it is not redundant for is memory safety.
`A[(rowBase+r)*K + kt + c]` with `r = M-1` and `kt + c = 775` is
`(M-1)·769 + 775 = M·769 + 6`: six floats past the end of the allocation. Run
it:

```
compute-sanitizer --tool memcheck .\exercise01_unguarded.exe
========= Invalid __global__ read of size 4 bytes
=========     Access to 0x130970340c is out of bounds
========= Invalid __global__ read of size 4 bytes
=========     Access to 0x1309703410 is out of bounds
...
  4/4 shapes correct
```

**This is the subtle trap.** The numerical validator cannot see it — there is
nothing numerically wrong. Only `compute-sanitizer` can, and only if you run it.
A kernel that reads out of bounds and gets the right answer gets the right
answer *on this allocator, on this run*: `cudaMalloc` happened to leave slack
after the array. Under a different allocation, a different `K`, a
`cudaMallocManaged` buffer or a sub-matrix view with `lda > K`, it faults or
reads someone else's data.

The `0.0f` fill on the guarded path is what makes a padded k-tile
arithmetically identical to a short one, which is why no separate epilogue over
the K remainder is needed.

---

## TODO 3 — the inner product

```cpp
#pragma unroll
for (int kk = 0; kk < BK; ++kk) {
    float rM[TM], rN[TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) rM[i] = As[kk][tRow*TM + i];
    #pragma unroll
    for (int j = 0; j < TN; ++j) rN[j] = Bs[kk][tCol*TN + j];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = fmaf(rM[i], rN[j], acc[i][j]);
}
```

**Twelve shared reads buy thirty-two fused operations**, `TM·TN/(TM+TN) = 2.67`,
which is prediction P1's bucket 3.

### The second null result

The version that looks wrong:

```cpp
for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j)
        acc[i][j] = fmaf(As[kk][tRow*TM+i], Bs[kk][tCol*TN+j], acc[i][j]);
```

`2·TM·TN = 64` shared reads written where 12 are needed. **Measured: 8276
GFLOP/s, 1.05× *faster* than the hoisted version.** The loop bounds are
compile-time constants, the loops unroll fully, and the addresses are affine in
the loop indices, so `nvcc 13.2` common-subexpression-eliminates all 52
redundant reads. The SASS of the two versions is the same: 24 `LDS.128` and 256
`FFMA` between the barriers.

This is spec §12 rule 11 — *the compiler will vectorize your accesses out from
under your analysis* — landing in your favour for once. Three conclusions, and
they are the point of the TODO:

1. **Write the hoisted form anyway.** It states the intent, it does not depend
   on an optimiser heuristic, and it stops being equivalent the moment the
   addressing becomes non-affine (an XOR swizzle — see lesson §8 — a runtime
   pitch, a `volatile`) or the loop stops being unrollable.
2. **Never claim a speedup from a source transformation without reading the
   SASS.** The count that matters is the one `cuobjdump` prints.
3. What makes the register tile work is not which of the two forms you wrote —
   both express the same rank-1 update. It is that the thread owns 32 outputs.
   A one-output-per-thread kernel cannot express the rank-1 update at all,
   whatever you do with its loops.

---

## TODO 4 — the epilogue

```cpp
#pragma unroll
for (int i = 0; i < TM; ++i) {
    const int r = rowBase + tRow*TM + i;
    if (r >= M) continue;
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int c = colBase + tCol*TN + j;
        if (c < N) {
            if (beta == 0.0f) C[(size_t)r*N + c] = alpha * acc[i][j];
            else              C[(size_t)r*N + c] = alpha * acc[i][j]
                                                 + beta * C[(size_t)r*N + c];
        }
    }
}
```

**A guard per element, not per thread tile.** `M = 1027` and `BM = 128` leaves 3
valid rows in the last block row, so in that block the thread with `tRow = 0`
owns rows 0–7 of which only 0–2 exist. A single
`if (rowBase + tRow*TM < M)` writes 5 rows past the end of C — an out-of-bounds
write that `compute-sanitizer` reports. The opposite mistake,
`if (rowBase + tRow*TM + TM - 1 < M)`, writes nothing for that thread and leaves
`3 · 2053 = 6159` elements at `+infinity`, which the finiteness check catches
immediately (`nonfinite 6159`).

**The `beta == 0` branch** is Module 16's contract. The harness prefills C with
`+infinity`, so `alpha*acc + 0.0f*inf` is NaN in every element. The branch is
warp-uniform (`beta` is a kernel argument), so by Module 8 it costs one
predicate.

---

## TODO 5 — the predictions

**P1 = 3.** `TM·TN/(TM+TN) = 32/12 = 2.67`, bucket "2 to 3". Derivable before
the kernel runs; it is a property of the decomposition, not of the code.

**P2 = 4.** Measured speedup over the Module 17 baseline: **4.9–5.8×** across
runs, bucket "4.5× to 7×". A reader who reasons "FMAs per shared read went from
0.50 to 2.67, so about 5×" lands on it for the right reason. A reader who
reasons from occupancy predicts a *slowdown* — the baseline runs one 1024-thread
block per SM at 66.7 % occupancy and this kernel runs three 256-thread blocks at
50 % — and is wrong by a factor of five. That error is the subject of Exercise 2.

---

## Synchronization / memory reasoning

Two `__syncthreads()` per k-tile, and they are different barriers.

The **first** separates the cooperative stores to `As`/`Bs` from the reads of
them: a read-after-write hazard across threads, needing both of Module 9's
guarantees — the execution barrier (every store issued) and the block-scope
memory fence (every store visible). Dropping it is a race that
`compute-sanitizer --tool racecheck` reports and that produces plausible wrong
answers, because most of the time most of the tile is already there.

The **second**, at the end of the loop body, is a **write-after-read** hazard: a
thread that finishes its 32 fused operations early would start overwriting
`As`/`Bs` for tile `kt + BK` while a slower thread is still reading tile `kt`.
Module 6 named this and named double buffering as the alternative; Example 1's
`v5` implements it (and, measured, loses — lesson §9).

Both barriers are at kernel scope and reached unconditionally by all 256
threads, satisfying Module 9's uniformity rule. Note that the epilogue's
`if (r >= M) continue;` is *after* the last barrier: a barrier inside that loop
would be divergent, which on sm_89 corrupts silently rather than hanging.

---

## Performance reasoning

Observed:

```
  Module 17 baseline (32x32 tile)           2.2687 ms     1429.4 GFLOP/s
  your register-tiled kernel                0.4103 ms     7903.2 GFLOP/s
  speedup over the baseline                   5.53 x
  fraction of the measured FP32 ceiling      43.91 %  (18000 GFLOP/s)
  your kernel: 80 registers, 0 B spilled, 6272 B shared, 3 blocks/SM, 50.0% occupancy
```

Range across runs: baseline 1372–1530 GFLOP/s, solution 7399–8336 GFLOP/s,
speedup **4.9–5.8×**, 41–46 % of the ceiling. The ratio is the stable quantity
(spec §12 rule 5).

The full variant table, all correct, all validated, same machine, same session:

| variant | GFLOP/s | vs. solution |
|---|---|---|
| solution | 7903 | 1.00× |
| TODO 1: `tRow`/`tCol` swapped | 5682 | **0.72×** |
| TODO 2: loader mapping `idx % BM` | 7741 | 0.98× |
| TODO 2: A-side `kt+c<K` guard removed | 8336 | 1.05× (and illegal) |
| TODO 3: reads not hoisted | 8276 | 1.05× |
| A-tile pitch `BM` instead of `BM+4` | 7560 | 0.96× |

**One of the five candidate mistakes matters, and it is not the one the
folklore points at.** That table is the exercise.

### Where the 5.5× comes from

Not from traffic. Both kernels stage through shared memory and both have
`BM·BN/(BM+BN) ≥ 16` FMAs per global load, past Module 16's 6.5 threshold;
neither is global-memory-bound. The difference is the *count of memory
instructions per fused operation in the inner loop*. Baseline: one `LDS` for A,
one `LDS` for B, one `FFMA`. Solution: three `LDS.128` and 32 `FFMA`.

SASS, per k-tile:

```
6 x LDG.E.CONSTANT   (+ 6 IMAD.WIDE)
6 x STS
BAR.SYNC.DEFER_BLOCKING
   24 x LDS.128
  256 x FFMA
BAR.SYNC.DEFER_BLOCKING
```

286 instructions between the barriers, 256 of them `FFMA`: **89.5 % arithmetic
density**, against 25 % for Module 16's naive `LDG, LDG, IMAD.WIDE, FFMA`. 192
of the 256 `FFMA`s carry the `.reuse` operand flag. The first few instructions
of the body:

```
/*0b40*/  LDS.128 R16, [R43] ;
/*0b50*/  LDS.128 R20, [R41.X16+0x1080] ;
/*0b60*/  LDS.128 R24, [R43+0x10] ;
/*0b70*/  FFMA R57, R16.reuse, R20, R57 ;
/*0b80*/  FFMA R58, R16.reuse, R21, R58 ;
/*0b90*/  FFMA R59, R16.reuse, R22, R59 ;
/*0ba0*/  FFMA R70, R16,       R23, R70 ;
```

`-Xptxas -v`:

```
ptxas info : Compiling entry function '_Z11gemmRegTileiiifPKfS0_fPf' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 80 registers, used 1 barriers, 6272 bytes smem, 400 bytes cmem[0]
```

80 registers = 32 accumulators + 12 operand registers + addressing and staging.
6272 B shared = `8·132·4` (padded Aᵀ) + `8·64·4` (B). **Check
`0 bytes spill stores` on every build in this module before looking at anything
else.** At 80 registers and 8 warps per block the register limit is
`65536/(80·32·8) = 3` blocks per SM, 50 % occupancy — and this beats every
higher-occupancy configuration in Example 2's sweep.

---

## Expected output

```
=== Module 18 / Exercise 1 - the two-level tile hierarchy ===
C(1027 x 2053) = A(1027 x 769) * B(769 x 2053)
block tile 128 x 64 x 8, thread tile 8 x 4, 256 threads/block
grid = 33 x 9 blocks; M/BM = 8.02 and N/BN = 32.08, so the last block
row and the last block column are both partial.

-- correctness ----------------------------------------------------
  1027 x 2053 x 769, positive operands     nonfinite  0 | Freivalds 0.000777 | sampled  0.0321 | PASS
  1027 x 2053 x 769, zero-mean operands    nonfinite  0 | Freivalds 4.78e-05 | sampled 0.00252 | PASS
  37 x 53 x 11  (one partial block)        nonfinite  0 | Freivalds   0.0249 | sampled   0.187 | PASS
  129 x 65 x 9  (tile+1 in every axis)     nonfinite  0 | Freivalds   0.0107 | sampled   0.249 | PASS
  4/4 shapes correct
...
  P1 FMAs per scalar shared read : you said 3, measured/derived 3  (2.67)  correct
  P2 speedup over the M17 kernel : you said 4, measured 4  (5.53 x)  correct

  score 10/10  (correctness 4, performance gate 2, predictions 4)

OVERALL: PASS
```

The headroom numbers are worth reading. `Freivalds 0.000777` means the kernel is
using 0.08 % of the error budget the backward-error bound allows, and the
zero-mean dataset — the one on which the house `1e-5·max(1,|ref|)` rule *rejects
a correct GEMM* (Module 16) — passes at 4.8e-05.

---

## The result that matters

Register tiling is not an optimization applied to a kernel; it is a different
decomposition, and the thing it changes is the **number of memory instructions
per fused multiply-add** — from 2 in the tiled baseline to 3/32 = 0.094 here.
Every other property people reach for — traffic, cache hit rate, sector counts,
occupancy — either did not move at all or moved the *wrong* way, and the kernel
is 5.5× faster anyway. The `89.5 % FFMA` line in the SASS is the measurement
that says so, and the five-variant table above is the measurement that says the
other four things did not matter.

**Variation to try.** Swap the thread tile to `TM = 4, TN = 8` while leaving the
block tile at `BM = 128, BN = 64` and the block at 256 threads — all of which
remain legal, since `(128/4)·(64/8) = 256`. Registers, shared bytes, blocks/SM,
occupancy, `TM·TN/(TM+TN)`, `BM·BN/(BM+BN)` and the entire inner-loop SASS are
*identical* to the solution's. Measured: **5095 GFLOP/s against 8186**, 1.61×
apart. Then work out, from Module 7's phase-splitting rule applied to the
`LDS.128` that reads `Bs`, why the number of threads along the N axis
(`BN/TN`) has a hard floor of 16. Lesson §8 has the answer; try to get there first.
