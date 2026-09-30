# Module 17 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

Warning-clean. `SCORE: 6/6`, `OVERALL: PASS`.

The shipped solution prints its hashes rather than checking them, so you can
regenerate the constants in `exercise02.cu` if you change the shape list.

---

## TODO 1 — shared bytes per block

```cpp
return 4 * (BM * (BK + pad) + BK * (BN + pad));
```

Two arrays: `As[BM][BK + pad]` and `Bs[BK][BN + pad]`. The padding is a **row
pitch**, so it is charged once per row, not once per array: `BM + BK` extra
floats, not two.

| shape | bytes |
|---|---|
| 8×8 BK=8 | 512 |
| 16×16 BK=8 | 1024 |
| 16×16 BK=16 | 2048 |
| 16×16 BK=32 | 4096 |
| 32×16 BK=16 | 3072 |
| 16×32 BK=16 | 3072 |
| 32×32 BK=32 | 8192 |
| 16×16 BK=16 **pad 1** | 2176 |

Note how small these are. The default per-block limit is 48 KB and the opt-in
limit is 99 KB; the largest tile here uses 8 KB. **Shared-memory capacity is
not what limits the tile size in this module** — the 1024-threads-per-block
limit is, because one output per thread means `BM·BN ≤ 1024`. That changes in
Module 18, where each thread owns several outputs and the tile can grow to
128×128 with the same thread count; there capacity becomes the binding
constraint and Module 7's swizzle-beats-padding argument starts to apply.

---

## TODO 2 — blocks per SM

```cpp
const int charged = ((smemBytes + DRIVER_RESERVE + SMEM_GRANULARITY - 1)
                     / SMEM_GRANULARITY) * SMEM_GRANULARITY;
int b = THREADS_PER_SM / threads;
const int bs = SMEM_PER_SM / charged;
if (bs < b) b = bs;
if (b > MAX_BLOCKS_SM) b = MAX_BLOCKS_SM;
```

**The order matters.** The driver adds its 1024-byte reservation *first* and
rounds the total up to the 128-byte allocation granularity *second*. Rounding
first and adding second gives a different answer for the pad-1 case
(2176 → 2176 + 1024 = 3200, which is already a multiple of 128; but
16384 → 16384 + 1024 = 17408 vs 16384 rounded then + 1024 = 17408 — equal here,
and not equal for every size). Module 6's measured table is the check:
`102400 / 16384 = 6.25` and the answer is **5**, because 16384 is charged as
17408 and `102400 / 17408 = 5.88`.

| shape | threads | asked | charged | model | API |
|---|---|---|---|---|---|
| 8×8 BK=8 | 64 | 512 | 1536 | 24 | 24 |
| 16×16 BK=8 | 256 | 1024 | 2048 | 6 | 6 |
| 16×16 BK=16 | 256 | 2048 | 3072 | 6 | 6 |
| 16×16 BK=32 | 256 | 4096 | 5120 | 6 | 6 |
| 32×16 BK=16 | 512 | 3072 | 4096 | 3 | 3 |
| **16×32 BK=16** | 512 | 3072 | 4096 | **3** | **2** |
| 32×32 BK=32 | 1024 | 8192 | 9216 | 1 | 1 |
| 16×16 BK=16 pad1 | 256 | 2176 | 3200 | 6 | 6 |

The 8×8 row is worth a moment: 24 blocks/SM is the hard cap, and
`24 × 64 = 1536` threads, so it reaches 100 % occupancy — and is still the
slowest kernel in the table. Occupancy is a permission, not a performance.

### The disagreement, which is the point of the TODO

The occupancy API says **2** for 16×32 BK=16 and the three-limiter model says 3.
The API is right, and the missing limiter is **registers**:

```
$ nvcc -arch=sm_89 -O3 -Xptxas -v -cubin -o s2.cubin exercise02_solution.cu

_Z9gemmTiledILi16ELi32ELi16ELi0EEviiiPKfS1_Pf  Used 44 registers, 3072 bytes smem
_Z9gemmTiledILi32ELi16ELi16ELi0EEviiiPKfS1_Pf  Used 40 registers, 3072 bytes smem
```

Every other shape in the table compiles to 40 registers; the 16×32 shape needs
44, because the B-tile cooperative loop has a different trip count and the
compiler keeps more indices live. At 512 threads that is `512 × 44 = 22 528`
registers per block against the SM's 65 536, and once Ada's per-warp allocation
granularity is applied the SM cannot fit three blocks. Four registers, one
resident block, 33 % of the occupancy.

Module 1 listed the placement gate as reserving *thread slots, warp slots,
registers and shared memory*, atomically. Module 6's occupancy formula quotes
only two of those four because its kernels had no register pressure worth
speaking of. The general rule: **hand-compute occupancy to understand it, and
call `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to know it.**

---

## TODO 3 — the two ratios

```cpp
return (double)BM * (double)BN / ((double)BM + (double)BN);   // FMAs / global load
return 2.0;                                                   // shared loads / FMA
```

**The derivation.** In one k-step a block loads `BM·BK` elements of A and
`BK·BN` elements of B — one global load instruction each — and performs
`BM·BN·BK` fused multiply-adds. So

```
FMAs per global load = BM·BN·BK / (BK·(BM + BN)) = BM·BN/(BM + BN) = 1/(1/BM + 1/BN)
```

**`BK` cancels.** That is the single most useful thing in this TODO. The
contraction depth of the tile does not affect reuse at all: it changes how much
shared memory you hold and how often you cross a barrier, and nothing else.
Readers who expect deeper tiles to be "more reuse" are conflating the two axes.

| shape | FMAs/global load |
|---|---|
| 8×8 | 4.00 |
| 16×16 (any BK) | 8.00 |
| 32×16, 16×32 | 10.67 |
| 32×32 | 16.00 |

Module 16's threshold was **6.4–6.5** FMAs per global load for 80 % of the FP32
ceiling, against the **0.50** that one output per thread supplies. Every shape
from 16×16 up clears the threshold. **And the measured throughput is 9 % of the
FP32 ceiling.** The threshold was necessary, not sufficient, and the reason is
the second ratio.

**The second ratio is 2.0 and does not depend on the tile shape at all.** Every
fused multiply-add needs one element of A and one element of B, and in a tiled
kernel both come from shared memory. `BM`, `BN` and `BK` do not appear. There is
no tile shape that changes it, and that is exactly why Module 18 has to change
the *decomposition* rather than the tile.

There is a ceiling on the first ratio too, and it is worth writing down:
`BM·BN ≤ 1024` because one output per thread, so `1/(1/BM + 1/BN) ≤ 16`, with
equality at 32×32. **16 is the most FMAs-per-global-load that tiling alone can
ever deliver on this hardware.**

---

## TODO 4 — bank conflicts

```cpp
const int P = T + pad;
const int tx = lane % T, ty = lane / T;
switch (pattern) {
  case 0: return ty * P + k;      // As[ty][k]
  case 1: return k  * P + tx;     // Bs[k][tx]
  case 2: return ty * P + tx;     // As[ty][tx]
  case 3: return tx * P + ty;     // As[tx][ty]
}
```

and the degree is the maximum over the 32 banks of the number of **distinct
words** that bank must supply — counting words, not lanes, because two lanes
asking for the same word are merged and broadcast (Module 7).

| access | T | pad | D |
|---|---|---|---|
| read `As[ty][k]` | 16, 32 | 0, 1 | **1** |
| read `Bs[k][tx]` | 16, 32 | 0, 1 | **1** |
| store `As[ty][tx]` | 16 | 0 | 1 |
| store `As[ty][tx]` | 16 | 1 | 2 |
| store `As[ty][tx]` | 32 | 0, 1 | 1 |
| store `As[tx][ty]` | 16 | 0 | **8** |
| store `As[tx][ty]` | 16 | 1 | 2 |
| store `As[tx][ty]` | 32 | 0 | **32** |
| store `As[tx][ty]` | 32 | 1 | 1 |

Work through two of them.

`As[ty][k]` at `T = 32`: a warp is 32 consecutive `tx` at one `ty`, and the
address does not contain `tx` at all. **All 32 lanes present the same address.**
One bank, one word, fanned out by the crossbar — a broadcast, degree 1, free
(Module 7: "the crossbar is a fan-out network"). At `T = 16` a warp spans two
values of `ty`, so there are two distinct addresses in two different banks.
Still 1.

`As[tx][ty]` at `T = 32`, pad 0: word index is `tx * 32 + ty` with `ty = 0` for
the whole warp, so the addresses are `0, 32, 64, ..., 992`. Every one of them is
in bank 0, and they are 32 **distinct** words. **Degree 32** — the worst access
the hardware admits, and the classic one, because it is what a column-major
store into a row-major tile looks like. Padding to 33 makes the address
`tx * 33` with bank `(33·tx) % 32 = tx`, a permutation: degree 1. `gcd(33, 32) = 1`
is Module 7's rule and Module 15's `gcd(P, 128/E) = 1`.

**The row-major tiled kernel performs only the first three patterns, and all
three are already degree 1 or 2.** On Ada the cost law is `max(2, D)`, so degree
2 is free. There is no conflict in the tiled GEMM to remove.

---

## TODO 5 — the design, and the result that contradicts the folklore

**(a) The tile shape.** `16 × 16`, `BK = 16`, no padding. The reasoning from the
three models:

- FMAs per global load: 8.00, comfortably past Module 16's 6.5 threshold. The
  32×32 shape offers 16.00, but once you are past the threshold the extra buys
  nothing, because the threshold was about the *global* load path and that path
  is no longer what binds.
- Blocks per SM: 6 at 256 threads = 100 % occupancy. 32×32 is one 1024-thread
  block per SM = 66.7 %, and its barrier convoys 32 warps instead of 8.
- Bank conflicts: degree 1 everywhere, so nothing to fix and no reason to pad.

Measured:

| shape | ms | GFLOP/s | × best | FMA/global-ld |
|---|---|---|---|---|
| 8×8 BK=8 | 2.752 | 1232 | 0.788 | 4.00 |
| 16×16 BK=8 | 2.436 | 1392 | 0.891 | 8.00 |
| **16×16 BK=16** | **2.170** | **1563** | **1.000** | 8.00 |
| 16×16 BK=32 | 2.278 | 1489 | 0.953 | 8.00 |
| 32×16 BK=16 | 2.502 | 1355 | 0.867 | 10.67 |
| 16×32 BK=16 | 2.877 | 1178 | 0.754 | 10.67 |
| 32×32 BK=32 | 2.719 | 1247 | 0.798 | 16.00 |
| 16×16 BK=16 pad 1 | 3.340 | 1015 | 0.650 | 8.00 |

**The FMA/global-load column is anti-correlated with speed above 16×16.** The
shape with the most reuse in the whole table (32×32, 16.00) is 20 % slower than
the one with half of it. That is not a contradiction; it is the model running
out of domain. Module 16's ratio predicts when the *global* load path stops
being the constraint, and by 16×16 it has stopped. Past that point the
constraints are occupancy and barrier convoying, and both get worse with tile
size.

`16×16 BK=16` and `16×16 BK=32` swap places between runs (they are within 5 % of
each other and the ordering is not stable), which is why the harness scores the choice
by RANK (top 3 of 8) rather than by a percentage margin. On a thermally
throttled laptop the whole table shifts and a fixed margin rejects a correct
answer at random; the ordering does not shift. Their FMA/global-load ratios are
identical; the difference is that `BK = 32` crosses half as many barriers per
unit of work and holds twice as much shared memory. On this problem those two
effects nearly cancel.

**(b) The padding prediction: bucket 3, padding is a clear LOSS.** Measured
`time(pad 0) / time(pad 1) = 0.650–0.706` — the padded kernel is **1.4–1.5×
slower**.

The degree table already says padding cannot help: every access is degree 1 and
`max(2, D)` makes degree ≤ 2 free. That gets you to bucket 2. Bucket 3 needs the
SASS:

```
$ nvcc -arch=sm_89 -O3 -cubin -o e2.cubin exercise02_solution.cu
$ cuobjdump -sass e2.cubin

gemmTiled<16,16, 8,0>  accumulation body:  2 LDS.128 +  8 LDS +  8 FFMA
gemmTiled<16,16,16,0>  accumulation body:  4 LDS.128 + 16 LDS + 16 FFMA
gemmTiled<16,16,32,0>  accumulation body:  8 LDS.128 + 32 LDS + 32 FFMA
gemmTiled<32,32,32,0>  accumulation body:  8 LDS.128 + 32 LDS + 32 FFMA
gemmTiled<16,16,16,1>  accumulation body:             32 LDS + 16 FFMA
```

Every unpadded shape gets one `LDS.128` for every four `LDS`; the padded one
gets none. (Example 2 confirms the same for its own `T = 32` instantiations:
`8 LDS.128 + 32 LDS` unpadded, `64 LDS` padded.)

`As[ty][k]` for `k = 0..T-1` is **contiguous**, so `ptxas` merges four of those
reads into one 16-byte `LDS.128` — which requires the row base to be 16-byte
aligned. With a pitch of `T` floats, row `ty` starts at byte `4·T·ty`, a
multiple of 16 whenever `T` is a multiple of 4. With a pitch of `T + 1`, row
`ty` starts at byte `4·(T+1)·ty`, which is not, and the merge is lost. **20
shared-memory instructions become 32.**

Example 2 measures the shared-memory read bandwidth of this SM directly and
finds `10.2–10.3 TB/s` through `LDS.128` against `5.38–5.40 TB/s` through scalar
`LDS` — a factor of 1.9, which is exactly Module 7's "a conflict-free 32-lane
4-byte access still occupies the pipeline for two cycles". Losing the vector
merge costs you half the bank array's bandwidth on the A operand, and the
measured 1.4–1.5× follows.

This is spec §12 rule 11 with the sign reversed: there, the compiler's
vectorisation *invalidated* a conflict analysis by making the measured penalty
smaller than predicted. Here the compiler's vectorisation is the thing being
destroyed, and the analysis that ignores it predicts "no effect" when the truth
is "40 % slower".

The wider point, and the one Module 18 inherits: **padding is not a free
insurance policy.** Module 7 measured padding *beating* an XOR swizzle by 19 %
on an LSU-bound kernel; Module 15 measured the two tying on a DRAM-bound
transpose; here padding loses to *not padding at all*, on a kernel with no
conflicts to remove. Three modules, three different answers, all correct. Pad
when you have measured a conflict of degree ≥ 4 and you have measured that
removing it helps.

---

## Performance reasoning

Nothing in this exercise is a mystery once you have Example 2's ceiling
measurement. The tiled GEMM reads 8 bytes of shared memory per multiply-add. At
the measured FP32 ceiling of ~18 000 GFLOP/s = 9 000 G FMA/s that would require
**72 TB/s** of shared bandwidth, and the SM array supplies **5.4–10.3 TB/s**.
The kernel is therefore capped at 7–14 % of the FP32 ceiling — 1350–2570 GFLOP/s
— before any question of tile shape arises, and the whole table above lives
inside that band. Tile shape moves you around within a factor of 1.3; it cannot
move you out.

---

## Expected output

```
-- TODO 1: shared bytes per block --------------------------------
  hash 6fc4efb3 : correct
-- TODO 2: blocks per SM -----------------------------------------
  16x32 BK=16 pad0        512     3072       4096         3        2   <-- differ
  hash d228846c : correct   (matches the occupancy API everywhere: NO)
-- TODO 3: instruction-mix ratios --------------------------------
  shared loads per FMA : 2.0000
  hash 77fb5de9 : correct
-- TODO 4: bank-conflict degrees ---------------------------------
  hash dd175fcd : correct

-- TODO 5: your commitments --------------------------------------
  tile shape   : 16x16 BK=16 pad0
  padding      : bucket 3

  fastest measured : 16x16 BK=16 pad0
  your choice ranked 1 of 8 (100.0% of the best). ACCEPTED (top 3 required)
  padding: time(pad 0)/time(pad 1) = 0.650 -> bucket 3. You predicted 3. CORRECT

SCORE: 6/6
OVERALL: PASS
```

Absolute times move ±5 % between runs and the 16×16 BK=16 / BK=32 ordering
flips; the padding ratio (0.65–0.71) and the 8×8 penalty reproduce every time.

---

## The result that matters

You now have three models — reuse, occupancy, bank conflicts — and the exercise
is a demonstration that **each of them has a domain and none of them is the
answer on its own.** The reuse model correctly says 32×32 is best and 32×32 is
20 % slower than 16×16. The conflict model correctly says there are no conflicts
and the folklore fix for conflicts costs 40 %. The occupancy model correctly
says 8×8 reaches 100 % and 8×8 is the slowest kernel in the table. What actually
decides the answer is a resource none of the three models mentions: shared-memory
bandwidth against an instruction mix of two loads per multiply-add.

**Variation to try:** add a `BM = 64, BN = 16, BK = 16` shape (1024 threads,
5120 B, 12.8 FMAs per global load) and predict where it lands before you measure
it. Then explain why a tall-thin tile behaves differently from a short-wide one
of the same area, using Module 3's linearization rule and Module 5's sector
count on the two cooperative loads.
