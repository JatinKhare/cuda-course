# Module 18 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

This exercise has no single right answer. What follows is *a* design that scores
10/10, the reasoning that produced it, and the measurements that justify each
choice. A different tile shape that clears the gate is equally correct; the part
that is not negotiable is that every constant was chosen from a resource
calculation before the kernel was written.

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe

nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise03_solution.exe exercise03_solution.cu
nvcc -arch=sm_89 -O3 -cubin -o k.cubin exercise03_solution.cu
cuobjdump -sass k.cubin
compute-sanitizer --tool memcheck .\exercise03_solution.exe
```

---

## TODO 1 — the tile hierarchy

```
BMX = 128   BNX = 64   BKX = 8   TMX = 8   TNX = 4   THREADSX = 256
```

The chain of reasoning, in the order it has to happen:

**1. Start from the register file, not from the tile.** 65536 registers per SM.
Anything with `TM·TN > 64` accumulators is at or past the point where a
256-thread block cannot have two copies resident, and `TM·TN = 256` spills
outright (Example 2 measures it at 253 GFLOP/s — worse than naive). So
`TM·TN ∈ {16, 32, 64}`.

**2. `TM·TN/(TM+TN)` wants square, but `BN/TN` wants wide.** AM–GM says the
shared-level reuse is maximised at `TM = TN`. Lesson §8 says `BN/TN` must be at
least 16 or the `LDS.128` that reads `Bs` takes a 2-way bank conflict in every
phase, which is *not* free on a 16-byte access. With 256 threads and a 16×16
thread grid, `BN/TN = 16` exactly — satisfied for any `TN`, provided
`BN = 16·TN`.

**3. Pick `TM ≥ TN`.** Between `8 × 4` (reuse 2.67, `BM×BN = 128×64`,
`BM·BN/(BM+BN) = 42.7`) and `4 × 8` (same reuse, `64×128`, 25.6), the first has
the better block-level reuse, so it is strictly better on the ledger before any
measurement. Example 2 confirms it at 1.48–1.61×.

**4. `BK = 8`** because shared memory and registers both scale with it (the
`BK`-deep loop is fully unrolled, so ptxas's live range grows with it — see the
answer to Check-Your-Understanding question 1), and because
`BM·BN/(BM+BN)` does not depend on `BK` at all, so there is nothing to buy.
`BK = 8` also makes the A-tile load exactly 4 elements per thread and the B-tile
load exactly 2, with no remainder logic.

**5. 256 threads** = `(128/8)·(64/4)`, eight warps, which at 80 registers gives
three resident blocks: 24 warps per SM. Enough to cover the `LDS` latency, not
so many that the register budget per thread collapses.

Consistency, all checked by the harness: `256 = 16·16`; `128 % 8 = 0`;
`64 % 4 = 0`; `(128·8) % 256 = 0`; `(8·64) % 256 = 0`.

---

## TODO 2 — the ledger

```
LEDGER_ACC  = 32        = TMX * TNX
LEDGER_SMEM = 6272      = 8*(128+4)*4  +  8*64*4  =  4224 + 2048
LEDGER_FPGL = 42.67     = 128*64 / (128+64)
```

The `+4` in the shared figure is the A-tile pad, and it is the part readers get
wrong: the harness compares `LEDGER_SMEM` against `cudaFuncGetAttributes`
exactly, so a reader who computed `8·128·4 + 8·64·4 = 6144` has either forgotten
the pad or not added it to the kernel. Both are informative.

---

## TODO 3 — the kernel

### Shared-memory layout

```cpp
const int AP = BMX + 4;                 // A-tile row pitch, in floats
__shared__ float As[BKX * (BMX + 4)];   // A^T : As[k][m]
__shared__ float Bs[BKX * BNX];         // B   : Bs[k][n]
```

**Transposed** because the thread tile needs `TMX = 8` consecutive values of `m`
at one `k`, and in the natural `As[m][k]` layout those are `BKX` floats apart
and cannot be read with `float4`. Transposing makes them adjacent, so eight
`LDS` become two `LDS.128`. That is the only reason to transpose; the bank
behaviour is a separate question with a separate answer.

**Padded by 4** because the transpose creates a conflict on the *store* side. A
thread stages `A[(rowBase+r)][kt+c]` with `r = tid/BKX, c = tid%BKX`, so within a
warp `r` takes 4 values and `c` takes 8. At pitch `BMX`, which is a multiple of
32:

```
bank = (c*BMX + r) mod 32 = r mod 32        (4 banks, 8 words each -> D = 8)
```

At pitch `BMX + 4`:

```
bank = (c*(BMX+4) + r) mod 32 = (4c + r) mod 32
```

and with `c ∈ [0,8)`, `r ∈ [0,4)` the 32 values `4c + r` are distinct: **D = 1**.

**The pad must be 4, and it is pinned from both sides.** Pad by 1, the
folklore default, and you get `(c + r) mod 32`, which collides whenever `c + r`
is equal — `D = 4`, a partial fix that looks like a fix (this is Module 7
Exercise 2's `PAD_PITCH 40` trap in new clothes). Pad by anything not a multiple
of 4 and the `float4` reads lose their 16-byte alignment and the kernel faults.
4 is the smallest value satisfying both.

Measured contribution of the padding on this kernel: **7903 against 7560, about
1.05×.** Small, and worth reporting as small: the conflict is on the *store*
path, which is 6 of roughly 300 instructions per k-tile.

### The cooperative loads

```cpp
#pragma unroll
for (int u = 0; u < (BMX*BKX)/THREADSX; ++u) {      // 4 per thread
    const int idx = tid + u*THREADSX, r = idx / BKX, c = idx % BKX;
    As[c*AP + r] = ((rowBase + r) < M && (kt + c) < K)
                 ? A[(size_t)(rowBase + r)*K + kt + c] : 0.0f;
}
#pragma unroll
for (int u = 0; u < (BKX*BNX)/THREADSX; ++u) {      // 2 per thread
    const int idx = tid + u*THREADSX, k = idx / BNX, n = idx % BNX;
    Bs[k*BNX + n] = ((kt + k) < K && (colBase + n) < N)
                  ? B[(size_t)(kt + k)*N + colBase + n] : 0.0f;
}
```

Consecutive `tid` walks `k` for A (A's contiguous axis; eight lanes cover one
32-byte sector) and `n` for B (32 lanes cover 128 bytes, four sectors, perfectly
coalesced). All three boundaries guarded, out-of-range filled with `0.0f` so a
short k-tile is arithmetically identical to a full one.

The `kt + c < K` guard on the A side is the one that looks removable — it *is*
redundant for the arithmetic, because the B-side guard zeroes the other factor —
and is not. Removing it reads up to six floats past the end of A for `r = M-1`
and `compute-sanitizer --tool memcheck` reports it while every numerical check
still passes. Exercise 1's notes work this through in detail.

**Why no `float4` on the global loads.** `lda = K = 769` and `ldb = N = 2053`,
neither a multiple of 4, so a `float4` load of `A[row*lda + k]` is misaligned and
faults. It is not slower, it is illegal. Even with padded leading dimensions the
measured win is a null result (0.94–1.01×, lesson §7), because the global path
is 6 instructions out of 300.

### The inner loop

```cpp
#pragma unroll
for (int kk = 0; kk < BKX; ++kk) {
    float rM[TMX], rN[TNX];
    #pragma unroll
    for (int i = 0; i < TMX; i += 4)
        *(float4*)(&rM[i]) = *(const float4*)(&As[kk*AP + tRow*TMX + i]);
    #pragma unroll
    for (int j = 0; j < TNX; j += 4)
        *(float4*)(&rN[j]) = *(const float4*)(&Bs[kk*BNX + tCol*TNX + j]);
    #pragma unroll
    for (int i = 0; i < TMX; ++i)
        #pragma unroll
        for (int j = 0; j < TNX; ++j)
            acc[i][j] = fmaf(rM[i], rN[j], acc[i][j]);
}
```

**Three shared-memory instructions buy 32 fused operations** — two `LDS.128` for
the eight A values and one for the four B values. That is 10.7 FMAs per
shared-memory *instruction*, past Module 16's 6.5 threshold, and it is the
number the whole module is aimed at.

`tRow = tid / (BNX/TNX)` and `tCol = tid % (BNX/TNX)`, so `tCol` is the fast
axis: a warp presents 16 distinct addresses to `Bs` (16 bytes apart — one
contiguous 128-byte phase, D = 1) and 2 to `As` (a broadcast). The other
assignment costs 1.39× and is Exercise 1's trap.

### The epilogue

Per-element guards on both axes, and the `beta == 0` branch. `M = 1027` with
`BMX = 128` leaves 3 valid rows in the last block row, so a thread's `i = 0` row
can be in range while its `i = 7` row is not.

---

## TODO 4 — the launch

```cpp
dim3 grid((N + BNX - 1)/BNX, (M + BMX - 1)/BMX, 1);
dim3 block(THREADSX, 1, 1);
```

`blockIdx.x` carries N and `blockIdx.y` carries M, agreeing with the kernel's
`rowBase = blockIdx.y * BMX`. At 37 × 53 × 11 this is a 1 × 1 grid of 256
threads in which every guard fires and 219 of the 256 threads write nothing;
the harness checks it.

---

## TODO 5 — the prediction

**`PRED_CEIL = 4`, 35 % to 55 %.** Measured **41–46 %**.

The reasoning available before measuring: Module 16 measured that throughput is
very nearly proportional to FMAs per memory instruction up to about 6.5, where
it reaches 80 % of the ceiling. This kernel supplies 10.7 per shared instruction
but runs at 50 % occupancy with 24 warps per SM and still pays two barriers and
18 global-path instructions per 256 fused operations. Somewhere under half the
ceiling is the honest estimate, and cuBLAS itself only reaches 40–45 % on this
shape, which brackets it from above.

---

## Synchronization / memory reasoning

Two barriers per k-tile: the read-after-write one before the inner loop, and the
write-after-read one after it (Module 6 named the hazard; Module 9 makes the
semantics precise). Both are at kernel scope and reached unconditionally by all
256 threads, so Module 9's uniformity rule holds.

**Could one of them go?** Yes — that is double buffering, and lesson §9
implements and measures it. On this GPU it loses: the prefetch registers push
the kernel from 3 blocks per SM to 2 (or from 2 to 1 at a larger thread tile)
and the latency it hides is already hidden by the 48 MB L2 and 24 resident
warps. Measured 0.81–0.92× at every tile shape that is otherwise competitive.
`cp.async` (sm_80+, **Module 32**) removes the register cost and changes the
answer; it is deliberately not used here.

`compute-sanitizer --tool memcheck` on the solution reports no memory errors.
(It does report one `ERROR SUMMARY: 1 error`, which is its own
`cudaDeviceReset` warning — *"Resetting device while there are still other users
claiming to use it"* — raised by the sanitizer's instrumentation at exit, not by
the kernel. Under instrumentation the program also reports `OVERALL: FAIL`,
because the prediction bucket is scored against a throughput of 129 GFLOP/s
rather than 8170. Score the run without the sanitizer and check memory safety
with it.)

---

## Performance reasoning

```
-- the ledger -----------------------------------------------------
  accumulators/thread   : you said     32, true     32   ok
  shared bytes/block    : you said   6272, true   6272   ok
  FMAs per global load  : you said  42.67, true  42.67   ok
  compiler reports      : 80 registers, 0 B spilled, 6272 B shared
  occupancy API reports : 3 blocks/SM, 50.0% of 1536 threads

-- measurement ----------------------------------------------------
  Module 17 baseline                    2.1676 ms     1496.1 GFLOP/s
  your kernel                           0.3969 ms     8170.5 GFLOP/s
  speedup over the baseline              5.46 x   (gate: 4.50 x)
  fraction of the FP32 ceiling          45.39 %
```

Observed range across runs: 7399–8336 GFLOP/s, 4.9–5.8× over the baseline,
41–46 % of the 18 000 GFLOP/s ceiling, **91–108 % of `cublasSgemm`** on this
shape (cuBLAS measures 7258–8141 here). At 2048³ the same design measures 9335
against cuBLAS's 10032, i.e. **93 %**, which is the more representative figure.

`-Xptxas -v`:

```
ptxas info : Compiling entry function '_Z9gemmYoursiiifPKfS0_fPf' for 'sm_89'
ptxas info : Function properties for _Z9gemmYoursiiifPKfS0_fPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 80 registers, used 1 barriers, 6272 bytes smem, 400 bytes cmem[0]
```

SASS, per k-tile:

```
6 x LDG.E.CONSTANT   (+ 6 IMAD.WIDE)
6 x STS
BAR.SYNC.DEFER_BLOCKING
   24 x LDS.128
  256 x FFMA          (192 of them carrying .reuse)
BAR.SYNC.DEFER_BLOCKING
```

**282 instructions between the barriers, 256 of them `FFMA`: 90.8 % arithmetic
density.** Module 16's naive kernel is at 25 % (`LDG, LDG, IMAD.WIDE, FFMA`).
The first lines of the body:

```
/*0b40*/  LDS.128 R16, [R43] ;
/*0b50*/  LDS.128 R20, [R41.X16+0x1080] ;
/*0b60*/  LDS.128 R24, [R43+0x10] ;
/*0b70*/  FFMA R57, R16.reuse, R20, R57 ;
/*0b80*/  FFMA R58, R16.reuse, R21, R58 ;
/*0b90*/  FFMA R59, R16.reuse, R22, R59 ;
/*0ba0*/  FFMA R70, R16,       R23, R70 ;
/*0bb0*/  FFMA R68, R17.reuse, R20, R68 ;
```

The `.reuse` flags are Ada's operand-reuse cache being fed by the rank-1
structure: `R16` is the same A value across four consecutive `FFMA`s.

**Read this before tuning anything.** If `FFMA` is not the overwhelming majority
of the instructions between the barriers, the decomposition is wrong and no
parameter sweep will fix it. That was true of Module 16's kernel (25 %) and of
Module 17's (33 %), and it is why both plateau.

---

## Common wrong approaches

| approach | symptom |
|---|---|
| `TM = TN = 16`, "more reuse is better" | 256 accumulators; ptxas allocates 64 registers and spills 2272 bytes; 253 GFLOP/s, *slower than naive*; `-Xptxas -v` says so before you run it |
| chasing occupancy with `__launch_bounds__(256, 6)` | 40 registers, 896 bytes spilled, 100 % occupancy, 390 GFLOP/s |
| `TN = 8` with `BN = 64`, i.e. `BN/TN = 8` | correct, all resources identical, 1.48–1.61× slower; the `LDS.128` on `Bs` takes a 2-way conflict in every phase |
| A tile not transposed | 8 scalar `LDS` instead of 2 `LDS.128`; the shared-instruction count between the barriers rises from 24 to 72 per k-tile, so the FFMA density falls from 90.8 % to about 78 % (arithmetic, not measured: the layout change requires rewriting the reads as well) |
| A tile padded by 1 | `D = 4` instead of `D = 8` — three quarters of the conflict removed, which measures like a fix and is not |
| `float4` on the global loads with `lda = 769` | `misaligned address` fault at the first launch |
| one guard for the whole thread tile in the epilogue | either an out-of-bounds write (memcheck) or 6159 elements left at `+infinity` (the finiteness check) |
| dropping the second barrier without a second buffer | a write-after-read race; `racecheck` reports it, and the answer is wrong intermittently |

---

## The result that matters

You were given a problem statement and a gate, and the kernel that clears it is
four decisions deep: a block tile that fixes `BM·BN/(BM+BN)`, a thread tile that
fixes `TM·TN/(TM+TN)`, a transposed operand tile that turns the second ratio
into an *instruction* ratio, and a pad whose size is pinned simultaneously by a
bank-conflict argument and an alignment argument. None of those is a trick and
none of them was found by trying things: each follows from a count you can do
before writing code, which is why TODO 2 comes before TODO 3.

The single sentence to keep: **the kernel went from 8 % to 45 % of the machine
by deleting memory instructions, not by making memory faster** — 2 memory
instructions per FFMA became 3 per 32 — and the SASS line that reads 90.8 %
`FFMA` is the only evidence that actually matters.

**Variation to try.** Re-run your design at `M = N = 128, K = 65536`. It is the
same total work as a 1024³ GEMM and your kernel will do it at a small fraction
of the throughput, because the grid is a single block and 39 SMs are idle. Then
read lesson §11 on split-K, implement the workspace form, and find the value of
`S` at which the extra `4·S·M·N` of output traffic stops paying for the
parallelism it buys.
