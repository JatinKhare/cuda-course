# Module 17 — Tiled GEMM

> Prerequisites: Modules 1–13 and 16. Module 6 (shared memory, cooperative loading, the K/H argument), Module 7 (bank conflicts and the `max(2,D)` cost law), Module 9 (barrier semantics, RAW vs WAR) and Module 16 (the GEMM problem, the validator, the loads-per-FMA argument) are all load-bearing.
> What this module gives you: the shared-memory tiled GEMM — the structure, the boundary handling, the tile-size decision — together with a measurement of exactly how far it gets you, which is **1.30× the naive kernel and 20 % of cuBLAS**, and a quantitative account of what stops it there.

---

## Concept

### 1. Where Module 16 left the problem

Module 16 established the following, by measurement, on this GPU:

| quantity | value |
|---|---|
| naive GEMM | 1275–1348 GFLOP/s |
| cuBLAS SGEMM | 8150–8705 GFLOP/s |
| measured FP32 ceiling | 17 787–18 256 GFLOP/s |
| naive as a fraction of cuBLAS | 12–13 % |
| naive as a fraction of the FP32 ceiling | ~7 % |
| requested / compulsory traffic, naive | **724×** |
| of those requested bytes, served on chip | **≥ 92 %** |
| FMAs per global load needed for 80 % of the ceiling | **6.4–6.5** |
| FMAs per global load that one-output-per-thread supplies | **0.50** |

and the sentence that makes this module a forced move:

> *A cache reduces the cost of a memory instruction; it does not reduce the
> number of them.*

The naive kernel's inner loop is `LDG, LDG, FFMA`: two memory instructions per
multiply-add. The L1 and L2 were already absorbing more than 92 % of those
loads, so there was never a large *traffic* win available — and the kernel was
still at 7 % of peak, because the instruction mix, not the byte count, is what
binds.

This module does the one thing shared memory can do about that, and measures
it honestly.

### 2. The reuse argument, made concrete

Give a block of `T × T` threads a `T × T` tile of C. March the tile along the
contraction axis in steps of `T`. In each step the block stages a `T × T` tile
of A and a `T × T` tile of B in shared memory, and then every thread performs
`T` multiply-adds out of the scratchpad.

Count. Per k-step, per block:

```
global load instructions = T·T  (A tile)  +  T·T  (B tile)  =  2T²
fused multiply-adds      = T·T threads × T steps            =  T³
```

so

```
FMAs per global load = T³ / 2T² = T/2
```

and the total number of global load instructions for the whole problem falls
from `2MNK` to `2MNK/T`. Each element staged is used `T` times, once by each of
the `T` threads in its row (for A) or column (for B) of the output tile. That
is the reuse factor, and it is exactly Module 6's *K* with the halo tax *H* equal
to 1 — a GEMM tile has no halo.

More generally, with a `BM × BN` output tile and contraction depth `BK`:

```
FMAs per global load = BM·BN·BK / (BK·(BM + BN)) = 1 / (1/BM + 1/BN)
```

**`BK` cancels.** The contraction depth buys shared-memory footprint and barrier
amortisation; it does not buy reuse. That is worth stating loudly because the
intuition "a deeper tile has more reuse" is very natural and wrong.

Measured, from `example01.cu`:

| tile | threads | shared B | charged | FMAs/global load | blocks/SM | occupancy |
|---|---|---|---|---|---|---|
| 8×8 | 64 | 512 | 1536 | 4.00 | 24 | 100.0 % |
| 16×16 | 256 | 2048 | 3072 | 8.00 | 6 | 100.0 % |
| 32×16 | 512 | 3072 | 4096 | 10.67 | 3 | 100.0 % |
| 64×16 | 1024 | 5120 | 6144 | 12.80 | 1 | 66.7 % |
| 32×32 | 1024 | 8192 | 9216 | **16.00** | 1 | 66.7 % |

Module 16 said the threshold was 6.4–6.5 FMAs per global load. **Every tile from
16×16 up clears it.** And the measured throughput is 1703 GFLOP/s, which is
9.4 % of the FP32 ceiling, not 80 %.

The threshold was necessary and not sufficient, and §7 explains exactly why.

### 3. What T would have to be, and why you cannot have it

Suppose you wanted tiling alone to satisfy a 6.5-FMAs-per-*memory-instruction*
criterion rather than a 6.5-FMAs-per-*global-load* one. The tiled inner loop
issues two shared loads per FFMA regardless of `T`, so the answer is that no `T`
works. But it is still instructive to ask how big `T` can get at all, because
there are two independent walls and only one of them is the one people expect.

**Wall 1, the one people expect: shared-memory capacity.** Two `T × T` float
tiles cost `8T²` bytes. The per-block default limit is 49 152 B and the opt-in
limit is 101 376 B, so capacity alone allows `T ≤ 78` (default) or `T ≤ 112`
(opt-in). Capacity is *not* the binding constraint here.

**Wall 2, the one that actually binds: one output per thread.** A `T × T` tile
computed by `T × T` threads needs `T² ≤ 1024`, so **`T ≤ 32`** and the
FMAs-per-global-load ratio can never exceed **16**, for any tile size, on this
hardware. The largest square tile that fits in *threads* uses 8 KB of a 48 KB
budget.

That asymmetry is the whole hand-off to Module 18. The scratchpad has five times
more room than the decomposition can use, because the decomposition insists that
one thread owns one output. Break that and the tile can grow to 128×128 with the
same 256 threads — and *then* capacity becomes the binding constraint, and
Module 7's swizzle-versus-padding argument starts to matter.

### 4. Why each output still needs the complete row and column

This is the conceptual heart of the module and it is where readers most often
have a vague picture.

`C[i][j]` is a sum of `K` products. `K` is 1063 in this module's problem. Row `i`
of A is 1063 floats = 4252 bytes, and so is column `j` of B. A `16 × 16` tile
pair holds 2048 bytes total, for 256 different outputs. **There is no sense in
which the operands of one output element are ever simultaneously resident.**

What is resident is a *slice*: 16 of the 1063 terms, for all 256 outputs at once.
The tile loop therefore computes a sequence of **partial dot products**:

```
acc  =  sum over t of ( sum over k in tile t of A[i][k]·B[k][j] )
```

and the only place `acc` can live is a **register**, private to the thread,
alive across every iteration of the tile loop. Shared memory is scratch; it is
legitimately and completely destroyed at every iteration. The state that
survives is one float per thread.

Three consequences follow immediately, and each one is a bug class if you miss
it:

1. **`acc` must be declared inside the kernel, outside the tile loop, in
   registers.** Declaring it `__shared__` (a surprisingly common instinct, since
   "the tile is shared, so the sum should be too") makes every thread clobber
   every other thread's partial sum.
2. **You cannot early-return a thread whose output does not exist.** It still has
   to arrive at both barriers, because the block's other threads are waiting for
   it. The guard goes on the *store*, not on kernel entry. Module 9: `bar.sync`
   counts warps and exited threads are subtracted, so a barrier with absent
   threads succeeds with the wrong population and corrupts silently on sm_89.
3. **The tile loop must run `ceil(K/T)` times, not `K/T`.** The last tile is
   partial, and it carries `K mod T` real terms of every dot product in the
   matrix.

### 5. The two barriers

Module 6 named the two hazards and Module 9 made the semantics precise. Apply
them here:

```cpp
As[ty][tx] = ...;  Bs[ty][tx] = ...;
__syncthreads();                     // (a)
for (k = 0; k < T; ++k) acc = fmaf(As[ty][k], Bs[k][tx], acc);
__syncthreads();                     // (b)
```

**(a) is read-after-write.** Thread `(tx, ty)` reads `As[ty][0..T-1]` and
`Bs[0..T-1][tx]`. At `T = 16`, `Bs[k][tx]` for `k = 0..15` was written by threads
with `tid = tx, tx+16, tx+32, ..., tx+240` — **eight different warps**. This
barrier needs both of `__syncthreads()`'s guarantees: the execution barrier (the
writes have happened) and the block-scope memory fence (they are visible).

**(b) is write-after-read.** At the top of the next iteration every thread
overwrites the slot it just read from. A fast warp can reach that store while a
slow warp is still in the accumulation loop. No data is being *published* here —
nothing new is becoming visible — so this barrier needs only the **execution**
guarantee, G1. Module 9's classification is not pedantry: it is what tells you
that this is the barrier you can design away, and how.

**Measured with (b) deleted and no second buffer** (`example02.cu`, section D):

| T | time ratio | wrong elements out of 1 594 935 |
|---|---|---|
| 16 | 1.010× (faster) | **20 972** |
| 32 | 1.071× (faster) | **39 199** |

98.7 % of the matrix is correct, and the kernel is 1–7 % faster. This is the
shape of a bug that ships. Note also that it fires at `T = 8` too, where a block
is only two warps — the folklore that the second barrier only matters for large
blocks is wrong.

**The correct way to remove it: double buffering.** Hold two tile buffers and
alternate. The write in iteration `t+1` targets the buffer that was last read in
iteration `t−1`, and the single remaining barrier at the top of iteration `t+1`
already separates them. This is the debt Module 6 recorded ("double-buffering is
the WAR-hazard alternative") and Module 9 recorded ("the germ of software
pipelining"). Measured cost and benefit:

| T | barriers per k-step | shared memory | speed |
|---|---|---|---|
| 16, plain | 2 | 2048 B | 1.000× |
| 16, double-buffered | 1 | 4096 B | **1.007–1.009×** |
| 32, plain | 2 | 8192 B | 1.000× |
| 32, double-buffered | 1 | 16384 B | **1.044–1.057×** |

A 1–6 % win for twice the shared memory. It is worth more at `T = 32` because a
block there is 32 warps and the convoying cost per barrier is larger. It becomes
genuinely important only when the load is asynchronous and the prefetch can
overlap the compute — which needs `cp.async`, and is **Module 18**.

### 6. Boundary handling: zero-fill, and why it is exact

M, N and K are 1035, 1541 and 1063 in this module. None of them is a multiple of
8, 16 or 32. Every tile size leaves a partial tile on every axis.

The clean rule is: **a tile cell with no corresponding matrix element is filled
with 0.0f, and every thread performs the fill.**

```cpp
As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
```

Correctness: a zero term contributes nothing to a sum. `sum_k a_k b_k` over a
zero-padded tile is *exactly* the sum over the real terms — not approximately.
The validator confirms it: in `example01.cu` every tiled kernel reports the same
sampled error, `0.02686`, as the naive kernel, on both datasets. Padding the
tile with zeros is numerically free.

The alternative — skipping the store when the guard is false — is the defect
Exercise 3 plants. The slot keeps the previous tile's value, or on the first
tile *uninitialised shared memory*. `compute-sanitizer --tool memcheck` is clean
on it; the tool that sees it is

```
compute-sanitizer --tool initcheck --initcheck-address-space shared .\prog.exe
```

which reports `Uninitialized __shared__ memory read of size 16 bytes`. (Sixteen,
not four — the compiler merged four of the A reads into one `LDS.128`. See §8.)

There is a genuinely surprising fact here, measured in Exercise 1: **you only
need one of the two zero-fills.** An out-of-range A cell and an out-of-range B
cell occur at the same value of `k`, so a zero on either side kills the product.
Fixing only the A line makes the kernel pass; fixing only the B line makes it
pass; removing both makes it fail. Do not rely on it — the redundancy disappears
the moment `K < T`, where the un-zero-filled cell is uninitialised rather than
stale — but know that it is there, because it is why a one-sided fix survives a
test suite.

### 7. Bank conflicts in the tile: there are none, and padding costs 40 %

Apply Module 7's method. `bank = (byte address / 4) % 32`; the degree of a warp
access is the maximum, over banks, of the number of **distinct words** that bank
must supply; requests for the same word merge into a broadcast and are free.

Warp 0 of a `(T, T)` block, computed by enumeration in `example02.cu`:

| access | T | pad | degree |
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

`As[ty][k]` at `T = 32` is a **broadcast** — the address contains no `tx`, so all
32 lanes present the same address and the crossbar fans one word out to all of
them. `Bs[k][tx]` is unit-stride across lanes, one word per bank. Both free.

**The row-major tiled kernel performs only the first three patterns. Every one
of them is degree 1 or 2, and Ada's cost law is `max(2, D)`, so degree 2 is
free. There is no bank conflict in a tiled GEMM to remove.**

The only degree-32 access in the table is the store into a *transposed* A tile,
`As[tx][ty]` — the layout Module 18 will want, because it lets a thread read a
column of the A tile with one vector load. Padding fixes that one, as
`gcd(33, 32) = 1` predicts, and it is worth a measured **1.05–1.18×** on that
kernel.

Now the result that contradicts the folklore. Padding the ordinary row-major
tile — the `[T][T+1]` that every textbook recommends — measures:

| kernel | ratio time(pad 0) / time(pad 1) |
|---|---|
| tiled, T = 16 | **0.674–0.706** |
| tiled, T = 32 | **0.688–0.690** |

**Padding makes it 1.4–1.5× slower.** The reason is not banks; it is
vectorisation. `As[ty][0..T-1]` is contiguous, so `ptxas` merges four of those
reads into one 16-byte `LDS.128` — which requires the row base to be 16-byte
aligned. A pitch of `T + 1` floats puts row `ty` at byte `4(T+1)·ty`, which is
not a multiple of 16, and the merge is lost. Real SASS, from
`cuobjdump -sass`:

```
gemmTiled<16,0>  accumulation body:  4 LDS.128 + 16 LDS + 16 FFMA
gemmTiled<16,1>  accumulation body:             32 LDS + 16 FFMA
gemmTiled<32,0>  accumulation body:  8 LDS.128 + 32 LDS + 32 FFMA
gemmTiled<32,1>  accumulation body:             64 LDS + 32 FFMA
```

20 shared instructions become 32. §8 measures why that is worth 1.4×.

This is spec §12 rule 11 with the sign reversed. There, the compiler's
vectorisation invalidated a conflict analysis by making the measured penalty
*smaller* than predicted. Here the compiler's vectorisation is the thing being
destroyed, and an analysis that stops at the bank degrees predicts "no effect"
when the truth is "40 % slower". **Both times the fix is the same: look at the
SASS.**

Three modules now have three different answers about padding — M7 measured
padding beating an XOR swizzle by 19 % on an LSU-bound microbenchmark, M15
measured them tying on a DRAM-bound transpose, and M17 measures padding losing
to no padding at all on a kernel with no conflicts. All three are correct.
**Module 18 owns the padding-versus-swizzle decision** at the tile sizes where
capacity binds; this module's contribution to it is the observation that both
techniques break the operand contiguity that `LDS.128` needs, and that this is a
cost neither the padding rule nor the swizzle rule accounts for.

### 8. What actually binds: shared-memory bandwidth

Here is the number the module exists to produce.

The tiled inner loop reads **two 4-byte operands from shared memory for every
fused multiply-add**: 8 bytes per FMA. That ratio is a property of the
decomposition, not of the tile shape, and no choice of `BM`, `BN`, `BK` or
padding changes it.

At the measured FP32 ceiling of 18 000–18 100 GFLOP/s = 9 000–9 050 G FMA/s, 8
bytes per FMA requires **72 TB/s** of shared-memory bandwidth.

`example02.cu` measures what the SM array actually supplies:

| access width | measured | per SM per cycle |
|---|---|---|
| 4 B (`LDS`) | **5.38–5.40 TB/s** | ~76 B |
| 16 B (`LDS.128`) | **10.24–10.30 TB/s** | ~146 B |

(The per-cycle column divides by the clock implied by the FFMA ceiling; the
shared kernel is a different load and very likely clocks higher, so treat that
column as an upper estimate. The **ratio**, ~1.9×, is the stable quantity, and it
is Module 7's "a conflict-free 32-lane 4-byte access occupies the pipeline for
two cycles" exactly.)

So a two-shared-loads-per-FMA kernel is capped at

```
5.4/72 = 7.5 %   to   10.3/72 = 14.2 %   of the FP32 ceiling
     = 1345 GFLOP/s  to  2574 GFLOP/s
```

**and the measured tiled kernel runs at 1703–1722 GFLOP/s.** It lands between
the scalar cap and the vector cap, which is the only region it could land in,
because `ptxas` merges the A reads into `LDS.128` and leaves the B reads scalar.
Nothing in that calculation is fitted: the two caps come from microbenchmarks
that never saw the GEMM.

That is a complete account. **The tiled GEMM is shared-memory-bandwidth bound,
and the bound is 7–14 % of the FP32 ceiling.** It is not DRAM bound (the whole
17.9 MB working set is L2-resident), it is not occupancy bound (16×16 runs at
100 %), it is not bank-conflict bound (every access is degree 1), and it is not
barrier bound (removing a barrier is worth 1 %).

### 9. The hand-off to Module 18, as a number

The only way past a shared-bandwidth bound is to read fewer bytes per FMA, and
the only way to do that is to make one loaded value feed more than one
multiply-add. A thread that owns an `Rr × Rc` sub-tile of C loads `Rr + Rc`
shared words per k-step and performs `Rr · Rc` FMAs:

```
shared loads per FMA = (Rr + Rc) / (Rr · Rc) = 1/Rr + 1/Rc
```

Set that against the measured bandwidth. To reach 80 % of the FP32 ceiling you
need `0.8 × 72 = 58 TB/s` of demand to fit inside the 10.3 TB/s the array
supplies, i.e. a loads-per-FMA of `2 × 10.3/58 = 0.35`, i.e.

```
Rr = Rc = 2 / 0.35  =>  Rr >= 5.7
```

**A 6×6 register tile, or in practice 8×8.** That is precisely the shape every
production SGEMM kernel uses, and this module derived it without writing one.

`example02.cu` also measures the ceiling directly, using the same trick Module
16 used one level up: hold the two shared loads fixed and issue `R` FFMAs
against them instead of one. The arithmetic is meaningless; the instruction mix
is the point.

| R | shared loads per FMA | GFLOP/s | % of FP32 ceiling |
|---|---|---|---|
| 1 | 2.00 | 1706 | 9.4 % |
| 2 | 1.00 | 3426 | 18.9 % |
| 4 | 0.50 | 6603 | 36.5 % |
| 8 | 0.25 | 11 355 | **62.7 %** |

Compare Module 16's identical table for *global* loads: 9.5 / 19.6 / 38.2 /
62.2 %. The two tables agree to within a percent. **The law is the same law one
level down the hierarchy, and it was never about which memory space the operands
came from — it is about how many memory instructions there are per arithmetic
instruction.**

The `R = 1` row is the tiled GEMM of this module. Every other row is unreachable
by tiling, and reachable by register blocking. That is Module 18.

### 10. The measured summary

`example01.cu`, problem 1035 × 1541 × 1063, min of 12 rotated sweeps after a
1500 ms streaming + 500 ms compute warm-up:

| config | ms | GFLOP/s | × naive | % cuBLAS |
|---|---|---|---|---|
| naive (Module 16) | 2.587 | 1311 | 1.00 | 15.3 % |
| square 8×8 | 2.468 | 1374 | 1.05 | 16.1 % |
| **square 16×16** | **1.991** | **1703** | **1.30** | **19.9 %** |
| square 32×32 | 2.183 | 1554 | 1.19 | 18.2 % |
| rect 16×16 BK=8 | 2.815 | 1205 | 0.92 | 14.1 % |
| rect 16×16 BK=32 | 2.164 | 1567 | 1.20 | 18.3 % |
| rect 32×16 BK=16 | 2.386 | 1421 | 1.08 | 16.6 % |
| rect 16×32 BK=16 | 2.800 | 1211 | 0.92 | 14.2 % |
| rect 64×16 BK=16 | 3.346 | 1013 | 0.77 | 11.9 % |
| rect 32×32 BK=16 | 3.165 | 1071 | 0.82 | 12.5 % |
| `cublasSgemm` | 0.397 | 8549 | 6.52 | 100 % |

Absolute figures move a few percent run to run and can collapse by 2–3× if the
sweep runs immediately after another GPU-heavy process; the **ratios** reproduce
to ±0.02.

Three readings.

**Tiling is worth 1.30×.** Not 5×, not 10×. The textbook presents tiled GEMM as
the flagship demonstration that shared memory transforms a kernel, and on this
machine it moves you from 15 % of cuBLAS to 20 % of cuBLAS.

**The FMAs-per-global-load column is anti-correlated with speed above 16×16.**
32×32 has twice the reuse of 16×16 and is 20 % slower; 64×16 has 1.6× the reuse
and is 40 % slower. Once you are past the point where the global-load path binds
— and 16×16 is past it — more reuse buys nothing and the occupancy it costs is
real.

**The general cooperative loader costs 5–15 %.** `square 16×16` and
`rect 16×16 BK=16` stage identical tiles; the difference is that the rectangular
kernel's load has to work for `BK ≠ BN`, which means an integer division and a
strided loop where the square kernel has a single indexed store. Generality is
not free, and the place it is cheapest to pay for it is a compile-time
specialisation.

---

## Hardware Mental Model

**Why the scratchpad does not change the instruction count.** Module 6 said
shared memory is a scratchpad rather than a cache: nothing arrives unless a
thread puts it there. The corollary that matters here is the mirror image —
nothing *leaves* unless a thread reads it, with an instruction. `LDS` and `LDG`
are different opcodes against different windows (Module 6), but they occupy the
same LSU issue slot and they are counted the same way by the warp scheduler.
Converting `LDG` to `LDS` is a change of *cost per instruction*, which is real
and is worth the 1.30× measured here. It is not a change of *instructions per
FMA*, which stays at 2, and that is what caps the kernel.

**The bank array as a bandwidth, not just a conflict rule.** Module 7 taught
banks as a latency/replay phenomenon: degree `D` costs `max(2, D)`. The same
structure read as a bandwidth is 32 banks × 4 B = **128 B per cycle per SM**, and
that is the number that binds a GEMM. The FP32 lanes want 128 FMAs per cycle per
SM × 2 operands × 4 B = **1024 B per cycle**. The ratio is 8, and the measured
`LDS.128` figure of ~146 B/cycle/SM (upper estimate) is consistent with the array
running at full width. A conflict-free scalar `LDS` gets half of that, because
of Module 7's two-cycle floor — which is why the `LDS.128` merge is worth 1.4×
and why destroying it with a padding row is the most expensive thing in this
module.

**Where the barrier cost actually goes.** Module 6 priced `__syncthreads()` as
convoying: the block runs at the speed of its slowest warp at each barrier. In a
tiled GEMM the barrier is crossed `2·ceil(K/BK)` times — 134 times per block at
`BK = 16, K = 1063`. Measured, removing half of them (double buffering) is worth
1 % at `T = 16` (8 warps per block) and 4–6 % at `T = 32` (32 warps per block).
The convoying cost scales with warps per block, exactly as the model says, and
it is small because every warp in the block is doing identical work with no data
dependence — there is very little skew to expose. Contrast Module 9's
producer/consumer exercise, where a 30× skew made the barrier the whole story.

**Why 1024 threads per block is a trap.** `T = 32` gives the best reuse in the
table and loses. At 1024 threads a single block occupies 1024 of the SM's 1536
thread slots, so only one block fits: 66.7 % occupancy, and no second block to
run while the first is at a barrier. The GigaThread placement gate (Module 1)
reserves thread slots, warp slots, registers *and* shared memory atomically, and
at 1024 threads the thread-slot term alone decides it. The scratchpad in this
kernel is nearly empty — 8 KB of a 48 KB budget — and it does not help.

**The fourth limiter.** Exercise 2 makes the reader build Module 6's occupancy
formula and then compare it against
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`. They disagree on exactly one
shape: `16×32, BK=16`, where the model says 3 blocks/SM and the API says 2. The
missing term is **registers**. That instantiation compiles to 44 registers where
every other one compiles to 40, and `512 × 44 = 22 528` registers per block does
not fit three times into 65 536 under Ada's per-warp allocation granularity.
Four registers, one resident block, a third of the occupancy. Hand-compute
occupancy to understand it; call the API to know it. **Module 19** owns
occupancy properly.

**ARCHITECTURE-SPECIFIC:** the 32×4 B bank array and its 128 B/cycle/SM width,
the `max(2, D)` cost law and its two-cycle floor, the 1024 B driver reserve and
128 B granularity, 1536 threads and 24 blocks per SM, 100 KB of shared memory per
SM, the measured 5.4 / 10.3 TB/s shared-read figures, the 18 000 GFLOP/s FP32
ceiling, and every absolute time in this module.
**PORTABLE CUDA CONCEPT:** the tile-loop structure, the accumulator-in-a-register
argument, the RAW/WAR barrier pair and double buffering, zero-fill boundary
handling, the `1/(1/BM + 1/BN)` reuse formula, the loads-per-FMA argument, and
the conclusion that a scratchpad changes the cost of an operand fetch and not the
number of them.

---

## Code Walkthrough

### `example01.cu` — the kernel, the boundaries, the tile sweep

**§A** computes the reuse ledger at runtime, including the `BK` cancellation and
the `T ≤ 32` wall.

**§B** is the kernel. The three index mappings are the thing to stare at:

```cpp
const int row  = blockIdx.y * T + ty;        // COMPUTE mapping
const int col  = blockIdx.x * T + tx;
...
const int aCol = t * T + tx;                 // A LOAD mapping: tx walks k
const int bRow = t * T + ty;                 // B LOAD mapping: ty walks k
As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
```

Module 6 established that the load mapping is not the compute mapping. A GEMM
has **two** load mappings and they are not each other either. In A,
`threadIdx.x` indexes the contraction dimension; in B, `threadIdx.x` indexes the
output column. That is forced by layout, not by taste: `k` is A's fast axis
(`A[r*K+k]`) and B's slow axis (`B[k*N+c]`), and coalescing requires that
whatever varies with `tx` be the fast axis of the array being read. Get it
backwards on the B side and a warp reads 32 addresses `N·4` bytes apart — 32
sectors instead of 4 (Module 5), on top of the wrong answer.

**The general rectangular kernel** in the same file shows what the load becomes
when `BK ≠ BN`: the tile no longer has one cell per thread, so the load becomes
Module 6's flat strided cooperative loop,

```cpp
for (int i = tid; i < BM * BK; i += nthr) {
    const int r = i / BK, c = i - r * BK;
    ...
}
```

with `BK` on the fast axis so consecutive `tid` read consecutive addresses. This
is the shape you need in general and it costs 5–15 % in integer arithmetic
relative to the square specialisation.

**§C** runs all eleven configurations plus cuBLAS through Module 16's
`gemmValidate()` on two `(alpha, beta)` settings, with C prefilled with `+inf`.
Every tiled kernel reports the *same* sampled error as the naive one, 0.02686 —
tiling does not cost accuracy.

**§D** is the sweep. Warm-up is 1500 ms of streaming followed by 500 ms of FFMA,
per spec §12 rule 4 and its corollary; configurations are timed back to back in
a rotated loop with `SWEEPS = NCFG`; validation happened in §C, before any
timing.

**§E** re-runs naive, best-tiled and cuBLAS on Module 16's exact shape,
1027 × 2053 × 769, so the comparison to that module's table is direct: naive
1337 GFLOP/s against M16's 1275–1348, cuBLAS 8942 against M16's 8150–8705,
tiled 1694 = 18.9 % of cuBLAS.

**§F** prints the three ratios that constitute the hand-off: 8.00 FMAs per
global load (past Module 16's threshold), 0.50 FMAs per shared load
(unchanged from naive), 0.47 FMAs per memory instruction of any kind (slightly
*worse* than naive's 0.50, because the tiled kernel adds the tile stores).

### `example02.cu` — the account

**§A** measures three ceilings. The FP32 and DRAM ones are Module 16's kernels.
The new one is shared-memory read bandwidth, scalar and vectorised. Two details
matter in that microbenchmark:

```cpp
for (int j = 0; j < 16; ++j) { a0 += s[(base + (j*4+0)*257) & (SBW_WORDS-1)]; ... }
base += 1;   // without this the loads are loop-invariant and are hoisted
```

The stride of 257 words is **odd**, so consecutive lanes land on consecutive
banks (degree 1), and **non-adjacent**, so the compiler cannot merge the scalar
loads into `LDS.128` — spec §12 rule 11, applied deliberately. And `base += 1`
is load-bearing: without it every address is loop-invariant, `ptxas` hoists all
64 loads out of the timing loop, and the kernel reports **40 TB/s**, four times
the bank array's theoretical maximum. A microbenchmark that reports a physically
impossible number is telling you it is not measuring what you think.

**§C** enumerates the bank-conflict degrees and prints the SASS instruction
counts for the padded and unpadded tiles.

**§D** times twelve GEMM variants back to back — padded and unpadded, row-major
and transposed A staging, double-buffered, and the deliberately broken
no-WAR-barrier version — and validates them in a separate pass afterwards, so
the "mismatches" column tells you which of them were fast *and wrong*.

**§B** is the shared-loads-per-FMA probe, the direct analogue of Module 16 §E.

**§E** is the ledger and the `Rr ≥ 5.7` derivation.

---

## Check Your Understanding

1. A colleague reports that their tiled SGEMM gets *faster* when they change
   `__shared__ float As[32][32]` to `__shared__ float As[32][33]`, which is the
   opposite of what this module measured. They are not lying and their kernel is
   correct. Give a specific structural difference between their kernel and this
   module's that would produce that sign, and say what single measurement would
   confirm your explanation. Then say what you would expect to happen to *their*
   number if they also switched from `float` to `float4` staging.

2. This module's tiled kernel achieves 8.00 FMAs per global load, comfortably
   past Module 16's measured threshold of 6.4–6.5, and reaches 9.4 % of the FP32
   ceiling rather than 80 %. Module 16's threshold was derived from a correct
   measurement and this module's kernel really does achieve 8.00. Explain
   precisely what Module 16's number was a statement about, identify the
   quantity that would have to have been 6.5 for the prediction to hold, and
   then give the *one* structural change to the kernel that moves that quantity
   without changing the tile size at all.

3. Deleting the second `__syncthreads()` makes the kernel 1–7 % faster and
   produces 20 972 wrong elements out of 1 594 935 at `T = 16`. A reader argues:
   "1.3 % of elements wrong means 1.3 % of the barriers were actually needed, so
   with a 99 % chance of being right per barrier, the hazard is basically never
   real and I could keep the speedup if I re-ran until the answer was right."
   There are at least two independent things wrong with that argument. State
   both, and then explain why the *Freivalds* check passed this kernel at
   `T = 16` (ratio 0.59) while the sampled check rejected it (ratio 58.6) — and
   what that tells you about which check you would want if you could only afford
   one.

4. Suppose NVIDIA doubled the shared-memory bank array to 64 banks × 4 B,
   changing nothing else — same FP32 lane count, same clock, same DRAM. Predict,
   with numbers, what happens to (a) this module's 16×16 tiled kernel, (b) the
   `R = 8` row of the shared-loads-per-FMA probe, and (c) cuBLAS. Then name the
   change to the *decomposition* that would have been worth more than the
   hardware change, and say at what register-tile size the hardware change stops
   mattering at all.

Answers: `solutions/module17/check_your_understanding.md`.

---

## Exercises

### Exercise 1 — `exercise01.cu` — write the tiled kernel

The classic, at a size that is a multiple of nothing.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

| TODO | requirement |
|---|---|
| 1 | The A-tile cooperative load. Which element, why `tx` carries `k`, and what an out-of-range cell must contain. Every thread must execute the store. |
| 2 | The B-tile cooperative load. A different mapping, not TODO 1 with the letters changed, and a different boundary. |
| 3 | The tile-loop count including the partial final tile, and the two barriers. You are told there are two places and asked to justify the second from the hazard, not from habit. |
| 4 | The store: the output guard, and the BLAS `beta == 0` contract. |
| 5 | **Prediction**, committed before building: the best-tiled-over-naive ratio, as a bucket. |

Validation: Module 16's `gemmValidate()` at `T = 8, 16, 32` on two datasets with
different conditioning (positive and zero-mean), plus the `alpha/beta` path,
plus the prediction. `SCORE: 7/7` required for `OVERALL: PASS`.

### Exercise 2 — `exercise02.cu` — tile-shape design

Build the models, commit to a choice, then measure.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

| TODO | requirement |
|---|---|
| 1 | `smemBytesPerBlock(BM,BN,BK,pad)` |
| 2 | `blocksPerSM(threads, bytes)` — three limiters, the 1024 B reserve and the 128 B granularity in the right order. Cross-checked against the occupancy API, which disagrees on exactly one shape. |
| 3 | `fmasPerGlobalLoad(BM,BN,BK)` and `sharedLoadsPerFma()`. One of the three tile dimensions does not appear in the first, and none of them appears in the second. |
| 4 | `tileWordIndex(...)` and `conflictDegree(...)` — four shared access patterns, two tile sizes, padded and unpadded. |
| 5 | **Design:** `chooseTile()` from your own model, and a bucket prediction for what padding does. |

TODOs 1–4 are checked against FNV-1a hashes; TODO 5 against the measurement,
requiring the tile choice to rank in the top 3 of 8. 6 points, all required.

### Exercise 3 — `exercise03.cu` — "it works on 1024 × 1024 × 1024"

A tiled GEMM that is correct on every square power-of-two size and wrong three
different ways on 1035 × 1541 × 1063. Ships broken; the header states the
symptom only.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
compute-sanitizer --tool memcheck .\exercise03.exe
compute-sanitizer --tool initcheck --initcheck-address-space shared .\exercise03.exe
```

| TODO | requirement |
|---|---|
| 1 | Diagnosis: three codes from a list of seven. Four of the seven are visible on a square problem; ruling those out is most of the work. |
| 2–4 | The three fixes. |
| 5 | **Design:** the set of problem shapes that would have caught all three, under a work budget so you cannot brute-force it. Your shapes are scored on which of six listed arithmetic properties they cover; three of the six are what the three defects need, and working out which three is the same question as TODO 1. |

7 points, all required. The smallest correct answer to TODO 5 is a single shape
smaller than 20 x 40 x 10.

---

## Prediction

Commit these to writing before you build anything.

1. Module 16 measured the naive kernel at 12–13 % of cuBLAS. You are about to
   replace both of its global operand loads with shared-memory loads and cut the
   global load count by a factor of 8. Write down the percentage of cuBLAS you
   expect the tiled kernel to reach, **and** write down the instruction sequence
   you expect in its inner loop. If your second answer is `LDS, LDS, FFMA` and
   your first answer is above 50 %, one of them is wrong; work out which before
   you run anything.

2. The tile sizes measured are 8×8, 16×16 and 32×32. Rank them, and separately
   write down the FMAs-per-global-load ratio of each. Then say whether you expect
   your ranking and that ratio to be in the same order, and why.

3. You are about to change `__shared__ float As[16][16]` to `As[16][17]` — the
   textbook padding. Write down the bank-conflict degree of `As[ty][k]` before
   and after, and then predict the runtime ratio to within 20 %. If your two
   answers are "degree 1 both times" and "no change", you have done the analysis
   the module asks for and you will still be wrong by 40 %; the extra step is
   to run `cuobjdump -sass` and count the shared-memory instructions in the
   accumulation loop.
