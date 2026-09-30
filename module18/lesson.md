# Module 18 — Advanced GEMM Optimization

> Prerequisites: Modules 1–17. Module 16 (the GEMM problem statement, the traffic ledger, the loads-per-FMA bound, and `gemmValidate`) and Module 17 (shared-memory tiling: the tile loop, cooperative loading, the two barriers, boundary handling, tile-size selection) are load-bearing and are **not** re-taught here. Module 4 (local memory is DRAM), Module 5 (sectors, `float4`), Module 7 (banks, padding, XOR swizzle, the `max(2,D)` cost law) and Module 11 (coarsening for memory-level parallelism) are used directly.
>
> What this module gives you: the rest of the ladder from a tiled GEMM that has plateaued to a kernel that is at parity with cuBLAS on this GPU — register tiling, the two-level tile hierarchy, vectorization, the register/occupancy cliff, and software pipelining — each step derived, measured, and read off in the SASS.

---

## Concept

### 1. Where Module 17 left you, and why it stopped

Module 17 gave every block a `BM × BN` tile of C and walked K in steps of `BK`
through shared memory. The kernel is correct and it is faster than naive. It is
also nowhere near the machine. Measured on this GPU, at Module 16's shape
M = 1027, N = 2053, K = 769:

| kernel | GFLOP/s | % of the 18 000 GFLOP/s FP32 ceiling |
|---|---|---|
| naive, one element per thread (M16) | 1302–1345 | 7.2–7.5 % |
| block-tiled 32×32, one element per thread (M17) | 1372–1530 | 7.6–8.5 % |

**Tiling bought about 10 %.** That is the whole return on a scratchpad, two
barriers and a boundary-handling rewrite.

Module 16 predicted exactly this, in one sentence: *a cache reduces the cost of
a memory instruction; it does not reduce the number of them.* Shared memory is
a cache you manage by hand. Staging A and B in it turns each operand `LDG` into
an operand `LDS`, which is cheaper — but the inner loop is still

```
LDS   (one value of A)
LDS   (one value of B)
FFMA
```

**two memory instructions per fused multiply-add.** The ratio Module 16
identified as the bottleneck is unchanged. It has simply moved one level down
the hierarchy.

### 2. The governing ratio, stated twice

Module 16's number: **6.4–6.5 fused multiply-adds per load are needed to reach
80 % of the compute ceiling; one output element per thread supplies 0.50.**
That statement is true at *two* levels, and the two are fixed by two different
decisions.

**Level 1 — global to shared.** A block tile of `BM × BN` walking K in steps of
`BK` loads `BM·BK + BK·BN` elements per step and performs `BM·BN·BK` fused
operations with them, so

```
FMAs per global load  =  BM·BN·BK / (BM·BK + BK·BN)  =  BM·BN / (BM + BN)
```

`BK` cancels exactly. It does not appear. A 32×32 block tile gives 16; a
128×64 block tile gives 42.7. **The block tile already fixes level 1** — that
is Module 17's contribution, and it is why Module 17's kernel is not
global-memory-bound.

**Level 2 — shared to register.** Inside the block, one thread that owns one
element of C reads one value of A and one value of B per `k` and does one fused
operation: the ratio is `1·1/(1+1) = 0.50`. The naive kernel's number, again,
one level down.

Give that thread a `TM × TN` sub-tile of C held in registers. Per value of `k`
it reads `TM` values of A and `TN` values of B — and forms all `TM·TN` products:

```
FMAs per shared read  =  TM·TN / (TM + TN)
```

**Same function.** The hierarchy is self-similar: a block tile is to global
memory what a thread tile is to shared memory.

Two consequences fall straight out of the formula.

**Square tiles are optimal.** For a fixed register budget `R = TM·TN`, the
ratio `R/(TM+TN)` is maximised when `TM + TN` is minimised, and by AM–GM that
happens at `TM = TN = √R`. This is why every hand-written SGEMM you will ever
read uses a square-ish thread tile, and it is why a 16×4 tile is worse than an
8×8 tile at the same register cost (3.2 versus 4.0).

**6.5 is not reachable as a scalar count.** `TM = TN` gives `T/2`, so a scalar
ratio of 6.5 needs `T = 13`: 169 accumulators per thread, which does not fit
(the hardware limit is 255 registers per thread and you need room for
everything else). The escape is that **what matters is instructions, not
values.** `TM = 8` consecutive floats in shared memory is *two* `LDS.128`
instructions, not eight `LDS`. Per instruction:

```
FMAs per shared INSTRUCTION  =  TM·TN / ((TM + TN)/4)   [when both sides vectorize]
```

which for `TM = TN = 8` is **16.0**, and for `TM = 8, TN = 4` is **10.7**. Both
are comfortably past 6.5, and the SASS in §8 confirms the count exactly.

### 3. Register tiling is a decomposition change, not a code change

The move is: **each thread computes a `TM × TN` sub-tile of C, holding
`TM × TN` accumulators in registers.**

```cpp
float acc[TM][TN];                       // TM*TN registers, live for the whole kernel
...
for (int kk = 0; kk < BK; ++kk) {
    float rM[TM], rN[TN];
    for (int i = 0; i < TM; ++i) rM[i] = As[kk][tRow*TM + i];   // TM shared reads
    for (int j = 0; j < TN; ++j) rN[j] = Bs[kk][tCol*TN + j];   // TN shared reads
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            acc[i][j] = fmaf(rM[i], rN[j], acc[i][j]);          // TM*TN fused ops
}
```

That is the whole idea. `TM + TN` reads feed `TM × TN` operations, because the
outer product of a length-`TM` vector with a length-`TN` vector has `TM·TN`
entries and costs `TM + TN` inputs. **Register tiling is a rank-1 update.**

The two inner loops must be *separate*. A version that writes

```cpp
for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j)
        acc[i][j] = fmaf(As[kk][tRow*TM+i], Bs[kk][tCol*TN+j], acc[i][j]);   // WRONG
```

is numerically identical and reads shared memory `2·TM·TN` times instead of
`TM + TN` times. It passes every correctness test. Exercise 1 has a performance
gate precisely because nothing else will tell you.

### 4. Coarsening for reuse versus coarsening for MLP — the distinction Module 11 owes you

Module 11 gave one thread `C` elements of a streaming kernel and measured a
2× improvement at low occupancy, then **no** improvement at full occupancy, and
a regression at `C = 8, 16`. Module 11 called that **coarsening for
memory-level parallelism**: the win came from having more independent loads in
flight per thread, so `C` loads' worth of latency overlapped. Its conclusion was
blunt: *coarsening a kernel that is already at its ceiling makes it longer, not
faster.*

The code here has the same shape — one thread, several outputs, an array of
accumulators, `#pragma unroll`. It is a different optimization.

| | M11: coarsening for MLP | M18: coarsening for reuse |
|---|---|---|
| resource bought | outstanding memory requests per thread | operands reused out of registers |
| the quantity that improves | requests in flight (Little's Law) | FMAs per memory instruction |
| what the loads are | `C` **independent** addresses | `TM + TN` addresses reused `TM·TN` ways |
| traffic per output | unchanged | divided by `TM·TN/(TM+TN)` |
| value at full occupancy | **zero, or negative** | the entire optimization |
| failure mode | longer tail, more registers, no gain | register spill |

The decisive difference is the *structure of the reuse*. In M11's SAXPY there is
no reuse to collect: each element of `x` is read once by the algorithm, so the
only thing coarsening can buy is concurrency, and concurrency was already
supplied by occupancy. In GEMM the reuse is in the problem — element `A[i][k]`
is needed by all N outputs of row `i` — and a thread that owns `TM × TN`
outputs can *capture* `TM·TN/(TM+TN)` of it in its own register file. That is
not a latency trick. It deletes instructions.

Which is also why the two have opposite occupancy behaviour. M11's coarsening
is worth nothing at full occupancy; M18's coarsening **costs** occupancy and is
worth a factor of five anyway. §6 measures exactly how much of each.

### 5. The index arithmetic — the part where readers get lost

Three coordinate systems, each with its own boundary.

```
grid        block (bx, by)                -> C tile at (by*BM, bx*BN)
block tile  thread tid in [0, NT)         -> (tRow, tCol) in the (BM/TM) x (BN/TN) thread grid
thread tile (i, j) in [0,TM) x [0,TN)     -> C element (by*BM + tRow*TM + i, bx*BN + tCol*TN + j)
```

with `NT = (BM/TM) · (BN/TN)` threads per block. Launch the block **1-D**; the
2-D thread grid is derived, not declared:

```cpp
const int tRow = tid / (BN/TN);
const int tCol = tid % (BN/TN);
```

`tCol` is the fast axis, so by Module 3's linearization rule a warp spans
`32 · TN` consecutive columns of C. That is what makes the epilogue's stores
coalesced and it is the reason `tCol` — not `tRow` — must come from `tid % ...`.
Get this backwards and every store in the epilogue is strided by `TN·(BN/TN)·4`
bytes.

There is a second, quieter mapping: **the cooperative load is a third,
independent thread-to-data mapping** (Module 6's "load mapping ≠ compute
mapping", at a bigger scale). The `NT` threads stage `BM·BK` elements of A and
`BK·BN` of B, and the mapping has nothing to do with `(tRow, tCol)`:

```cpp
for (int u = 0; u < (BM*BK)/NT; ++u) {
    const int idx = tid + u*NT;
    const int r = idx / BK, c = idx % BK;     // consecutive tid -> consecutive k
    As[c][r] = inBounds ? A[(rowBase + r)*lda + kt + c] : 0.0f;
}
```

`r = idx / BK, c = idx % BK` makes consecutive thread ids walk **k**, which is
A's contiguous axis, so with `BK = 8` eight lanes cover one row's 32 bytes —
exactly one sector, fully used — and a warp asks for four sectors. The obvious
alternative, `r = idx % BM`, makes consecutive lanes walk **rows**, whose
addresses are `lda·4` bytes apart: **32 sectors per warp instead of 4**, for the
same bytes. The kernel is still correct. Module 5's sector count is the only
thing that tells you.

Three boundaries must be handled, and one of them is silent:

- `rowBase + r >= M` — the last block row is partial (1027 = 8·128 + 3).
- `colBase + n >= N` — the last block column is partial (2053 = 32·64 + 5).
- `kt + c >= K` — the last **k-tile** is partial (769 = 96·8 + 1). This is the
  quiet one. `A[(rowBase + r)*K + kt + c]` with `kt + c >= K` is still a legal
  address; it is the *next row of A*. Nothing faults, nothing crashes, and the
  answer is wrong by a small amount that a `1e-5` tolerance will not catch. The
  out-of-range element must be read as `0.0f` so that the extra `k` values
  contribute nothing.

The epilogue needs a guard per element, not per thread tile: a thread whose
first row is in range may have its last row out of range.

### 6. Registers are the scarce resource, and the optimum is not the maximum

`TM × TN` accumulators are live from the first `k`-tile to the last. They cannot
be spilled without destroying the kernel and they cannot be shared. At 256
threads per block:

| TM × TN | accumulators | registers measured | blocks/SM | occupancy | FMAs/shared read | GFLOP/s |
|---|---|---|---|---|---|---|
| 1 × 1 | 1 | 38 | 6 | 100.0 % | 0.50 | 1060 |
| 2 × 2 | 4 | 40 | 6 | 100.0 % | 1.00 | 3432 |
| 4 × 1 | 4 | 39 | 6 | 100.0 % | 0.80 | 3140 |
| 4 × 4 | 16 | 64 | 4 | 66.7 % | 2.00 | 6578 |
| 8 × 1 | 8 | 63 | 4 | 66.7 % | 0.89 | 3223 |
| 8 × 2 | 16 | 64 | 4 | 66.7 % | 1.60 | 5808 |
| **8 × 4** | **32** | **80** | **3** | **50.0 %** | **2.67** | **7399** |
| 4 × 8 | 32 | 84 | 2 | 33.3 % | 2.67 | 4725 |
| 8 × 8 | 64 | 124 | 2 | 33.3 % | 4.00 | 7016 |
| 16 × 8 | 128 | 210 | 1 | 16.7 % | 5.33 | 4817 |
| 8 × 16 | 128 | 216 | 1 | 16.7 % | 5.33 | 3650 |
| 16 × 16 | 256 | 64 + **2272 B spilled** | 4 | 66.7 % | 8.00 | **253** |

Read that table four ways.

**(a) The optimum is an interior point.** The maximum-occupancy configuration
(1×1, 100 %) runs at 14 % of the best. The maximum-reuse configuration that
still fits (16×8, 5.33 FMAs per read) runs at 65 % of the best. The winner is
in the middle, at 50 % occupancy.

**(b) Four registers can cost a third of the occupancy.** `8 × 4` and `4 × 8`
have the same number of accumulators, the same shared-memory footprint, the
same FMAs-per-shared-read, and 80 versus 84 registers — and 3 versus 2 blocks
per SM, because registers are allocated in granules. At 256 threads (8 warps),
80 registers costs `80·32·8 = 20480` registers per block, so three blocks fit in
65536; 84 rounds up to 88 and costs 22528, and only two do. **Four registers —
one allocation granule — is worth a whole block.**

That is one of *three* mechanisms separating those two rows, and it is not the
biggest. The second is the block tile: `8 × 4` means `BM = 128, BN = 64` and
42.7 FMAs per global load, while `4 × 8` means `BM = 64, BN = 128` and 25.6.
**Both levels of the hierarchy move when you change the thread tile, and they
move in opposite directions.** The third is a shared-memory bank effect that no
resource counter reports; §8 isolates it with a controlled experiment in which
registers, occupancy, shared bytes, both reuse ratios and the entire inner-loop
SASS are held identical, and the two configurations still measure 1.61× apart.

**(c) The spill cliff is a cliff, not a slope.** At `16 × 16` the compiler gives
up, allocates 64 registers, and spills 2272 bytes per thread to local memory.
Occupancy goes *up*, to 66.7 %, and throughput collapses to **253 GFLOP/s —
below the naive kernel.** Module 4 established what local memory is: DRAM, with
an L1 line in front of it. A spilled accumulator is a DRAM round trip in the
innermost loop of the innermost loop.

**(d) The FMAs-per-read column is necessary but not sufficient.** It orders the
first five rows correctly and then stops working, because past `8 × 4` the
binding resource stops being the shared-memory path and becomes the register
file. A model that uses only one of the two gets the answer wrong; Exercise 2
asks you to build one that uses both.

#### Forcing the point: the same kernel at five occupancies

Take the `8 × 8` kernel and compile it five times with
`__launch_bounds__(256, n)`. The second argument is a *minimum blocks per SM*
request, and it is a hard instruction to `ptxas`: to fit `n` blocks of 256
threads it must fit in `65536/(256n)` registers per thread, spilling whatever
does not fit. Same source, same algorithm, same instruction schedule up to
register allocation:

| launch bound | registers | spill bytes | blocks/SM | occupancy | GFLOP/s |
|---|---|---|---|---|---|
| `(256, 1)` | 124 | 0 | 2 | 33.3 % | 7267 |
| `(256, 2)` | 124 | 0 | 2 | 33.3 % | 7297 |
| `(256, 3)` | 80 | 192 | 3 | 50.0 % | 2011 |
| `(256, 4)` | 64 | 552 | 4 | 66.7 % | 685 |
| `(256, 6)` | 40 | 896 | 6 | **100.0 %** | **390** |

**100 % occupancy is 18.7× slower than 33 % occupancy.** This is the sharpest
statement of "maximum occupancy is not maximum performance" the course has, and
the mechanism is fully visible: the only thing that changed is where the
accumulators live. Module 19 develops occupancy properly and Module 20 develops
the latency-hiding side of the trade; the number to carry into both is this one.

(Two nuances worth keeping. First, a *small* spill is not always fatal: at 128
threads per block, `__launch_bounds__(128, 4)` spills 80 bytes, buys a fourth
block, and comes out ahead of the unconstrained build. The cliff is at the point
where the spilled values are the ones in the inner loop. Second, `(256,1)` and
`(256,2)` are the *same binary*: `ptxas` was already fitting two blocks, so
asking for two changed nothing.)

### 7. Vectorization, and the transposed A tile

Two separate uses of `float4` (Module 5), with very different returns.

**Shared → register, where it pays.** The thread tile needs `TM` consecutive
values of A at one `k`. In the natural layout `As[m][k]` those are `BK` floats
apart and cannot be vectorized. **Store the A tile transposed** — `As[k][m]` —
and they become adjacent:

```cpp
for (int i = 0; i < TM; i += 4)
    *(float4*)(&rM[i]) = *(const float4*)(&As[kk*AP + tRow*TM + i]);
```

Eight `LDS` become two `LDS.128`. This is the single reason to transpose the A
tile — not bank conflicts, *vectorizability* — and it is what takes the
FMAs-per-shared-instruction ratio past 6.5.

A measurement worth knowing about: **on this compiler you often get it without
asking.** With the transposed layout in place, `nvcc 13.2` contracts the eight
adjacent scalar reads into `LDS.128` by itself; the explicit `float4` version
and the scalar version compile to the same 24 `LDS.128` per k-tile and measure
within noise of each other. This is spec §12 rule 11 (*the compiler will
vectorize your accesses out from under your analysis*) working in your favour
for once. It only happens because the addresses are provably contiguous —
which is exactly what the transpose provides, and exactly what the XOR swizzle
in §8 destroys.

**Global → shared, where it mostly does not.** Vectorizing the tile staging is
the classic next move and it is worth much less than it looks, for two reasons.

*Arithmetic:* with `BM = 128, BN = 64, BK = 8` and 256 threads, a thread issues
6 global loads and 256 fused operations per k-tile. The global path is already
1/40th of the instruction stream. Cutting it by four moves the total by about
1 %.

*Alignment:* a `float4` load of `A[row*lda + k]` requires the **address** to be
16-byte aligned, which requires `lda % 4 == 0`. Here `lda = K = 769`. It is not
merely slower to vectorize — it is **illegal**, and it will fault. The same for
`ldb = N = 2053`. Module 16 taught the leading dimension as a bookkeeping
parameter; here it becomes a performance parameter: real libraries pad `lda` to
a multiple of 4 (or 32) precisely so the staging loads can be vectorized. This
module measures the padded-leading-dimension variant and finds **no reliable
win on this GPU** (0.92–1.08× across runs), because of the arithmetic above.
It is reported as a null result rather than asserted as folklore.

### 8. Bank conflicts at this level — and Module 7's prediction, tested

Register tiling changes the shared-memory access pattern completely, so
Module 7's analysis has to be redone from scratch. Four accesses per k-tile.

**The A-tile transposed store.** Thread `tid` reads `A[(rowBase+r)][kt+c]` with
`r = tid/BK, c = tid%BK`, and writes `As[c][r]`. Within a warp, `r` takes 4
values and `c` takes 8. At pitch `BM` (a multiple of 32):

```
bank = (c·BM + r) mod 32 = r mod 32        (because BM ≡ 0 mod 32)
```

Four banks, each supplying eight distinct words: **D = 8**. This is the one
real conflict in the kernel, and it is created by the transpose that §7 just
argued for.

**The A-tile read**, `As[kk][tRow·TM + i]`: `tRow = tid/(BN/TN)` takes 2 values
across a warp, so the warp presents 2 distinct addresses. **Broadcast, D = 1.**

**The B-tile store**, `Bs[k][n]` with consecutive `tid` walking `n`: 32
consecutive floats, 32 distinct banks. **D = 1.**

**The B-tile read**, `Bs[kk][tCol·TN + j]`: 16 distinct `tCol`, addresses
`TN·4` bytes apart. For `TN = 4` that is 16 B apart and the `float4` phase split
(Module 7) makes each phase contiguous: **D = 1**. For `TN = 8` it is 32 B apart
and two `tCol` values collide per bank: **D = 2**, which by Module 7's
`cost ∝ max(2, D)` law on Ada is **free**.

#### A correction to Module 7's `max(2, D)` law, measured here

Module 7 established, on 4-byte shared accesses, that `cost ∝ max(2, D)` — a
conflict-free 32-lane 4-byte read already occupies the shared pipeline for two
cycles, so **a 2-way conflict is free on Ada**. That law does not survive the
move to `LDS.128`, and this module can measure it cleanly.

A 16-byte access is phase-split into **4 phases of 8 lanes** (Module 7), and one
phase moves 8 × 16 = 128 bytes — which is exactly the bank array's 128 B per
cycle. The phase is already saturating. A 2-way conflict *within a phase* has no
spare cycle to hide in, so it costs a second cycle: on `float4` shared reads
`cost ∝ D`, with no floor of 2.

The experiment that isolates it holds everything else fixed. Take `BM = 128,
BN = 64, BK = 8`, 256 threads, and compare `TM = 8, TN = 4` against
`TM = 4, TN = 8`. Both have 32 accumulators, 80 registers, 6272 shared bytes,
3 blocks/SM, `TM·TN/(TM+TN) = 2.67`, `BM·BN/(BM+BN) = 42.67`, and — verified in
the SASS — **byte-identical inner-loop instruction counts**: 286 instructions
between the barriers, 256 `FFMA`, 24 `LDS.128`, 2 `BAR.SYNC`. Everything the
compiler and the occupancy API report is identical.

The only difference is the bank pattern of the B-tile read, `Bs[kk][tCol·TN+j]`:

| | `TM=8, TN=4` | `TM=4, TN=8` |
|---|---|---|
| threads in x (`BN/TN`) | 16 | 8 |
| B read stride per lane | 16 B | 32 B |
| banks touched, one 8-lane phase | 0–31, all distinct → **D = 1** | 0–3, 8–11, 16–19, 24–27, twice over → **D = 2** |
| A read | 2 × `LDS.128`, broadcast, D = 1 | 1 × `LDS.128`, broadcast, D = 1 |
| shared-pipeline cycles per `k` (phases × D) | 2·4·1 + 1·4·1 = **12** | 1·4·1 + 2·4·2 = **20** |
| predicted ratio | — | **1.67×** |
| measured, 1027×2053×769 | 8186 GFLOP/s | 5095 GFLOP/s → **1.61×** |
| measured, 1024×1024×8192 | 7224 GFLOP/s | 5381 GFLOP/s → **1.34×** |

If the `max(2, D)` law held for 16-byte accesses, the predicted ratio would be
1.00× and the two would measure the same. They do not. The large-K run is the
control that rules out the epilogue: at K = 8192 the epilogue is 1/1024 of the
work and the gap is still 1.34×, so it is the inner loop.

**Practical rule.** Once the thread tile is read with `float4`, the number of
threads along the N axis of the block (`BN/TN`) must be at least 16, or the
B-tile read conflicts. That is a constraint on the thread-grid *shape* that no
resource counter reports, and it is why `TM ≥ TN` rather than `TM ≤ TN` when
you have to choose.

So there is exactly one thing to fix. Three ways to fix it, and this is where
Module 7's open question gets settled.

> Module 7 measured padding beating an XOR swizzle by **19 %** on an LSU-bound
> kernel and wrote: *"padding wins when you have the memory; swizzling wins when
> capacity limits tile size — which is why CUTLASS swizzles. The answer flips in
> GEMM."* This is GEMM.

**Fix 1 — pad the A-tile pitch to `BM + 4`.**

```
bank = (c·(BM+4) + r) mod 32 = (4c + r) mod 32
```

With `c ∈ [0,8)` and `r ∈ [0,4)`, `4c + r` takes 32 distinct values: **D = 1**.
The pad must be **4**, not the folklore 1. Pad by 1 and you get `(c + r) mod 32`,
which collides whenever `c + r` is equal — `D = 4`, a partial fix that looks
like a fix. This is Module 7 Exercise 2's `PAD_PITCH 40` trap in a new costume.
And 4 is forced from the other side as well: the pitch must stay a multiple of 4
or the `float4` reads of §7 lose their 16-byte alignment. **The two constraints
agree on 4 and on nothing else.** Cost: `BK·4·4 = 128` bytes per block, 1.5 % of
the tile.

**Fix 2 — an XOR swizzle**, `As[c][r ^ ((c & 3) << 3)]`. Zero extra bytes. The
XOR is by a multiple of 8, so it permutes whole aligned groups of 8 and the
`float4` reads stay 16-byte aligned. Bank becomes `(r + 8(c mod 4)) mod 32`: 16
banks, 2 words each, **D = 2**, free under `max(2, D)`.

**Measured, same kernel, same tile, 256 threads:**

| A-tile layout | registers | shared B | blocks/SM | GFLOP/s | vs. plain |
|---|---|---|---|---|---|
| plain, pitch `BM` (D = 8) | 124 | 8192 | 2 | 6587 | 1.00× |
| **pad to `BM + 4`** (D = 1) | 124 | 8320 | 2 | **7114** | **1.08×** |
| XOR swizzle (D = 2) | 128 | 8192 | 2 | 5761 | 0.87× |

**Padding wins in GEMM too, by 23 % over the swizzle.** Module 7's prediction
does not hold on this hardware, and both halves of it fail:

*The swizzle's cost is larger than Module 7 measured, not smaller.* Module 7
attributed it to 10 extra `LOP3` instructions. Here the real damage is
different and worse: the XOR makes the eight A-tile reads **not provably
contiguous**, so the compiler cannot contract them into `LDS.128`. The SASS
shows the padded kernel issuing 32 `LDS.128` where the swizzled kernel issues
48 scalar `LDS` plus 20 `LDS.128` plus 42 `LOP3`. The swizzle did not cost an
XOR per address — it cost the vectorization.

*The capacity premise is false on Ada for fp32 GEMM.* Module 7's argument was
that shared memory limits tile size, so bytes spent on padding are bytes not
spent on the tile. Run the same comparison at `BK = 32`, four times the shared
memory:

| layout, BK = 32 | registers | shared B | blocks/SM | limiter | GFLOP/s |
|---|---|---|---|---|---|
| plain | 196 | 32768 | 1 | **registers** | 4111 |
| pad by 4 | 196 | 33280 | 1 | **registers** | 5294 |
| XOR swizzle | 226 | 32768 | 1 | **registers** | 5193 |

At 32 KB of shared memory per block the kernel is **still register-limited**.
102400 bytes of shared memory per SM would admit two of these blocks; 65536
registers admit one. (And the reason is not that 32 KB is small: `ptxas` spends
registers in proportion to `BK` in order to software-pipeline the deeper
unrolled loop — 124 registers at `BK = 8`, 196 at `BK = 32` — so the two limits
move together and shared memory never overtakes. The answer key works through
a search for a counterexample and does not find one.) The gap between padding and swizzling narrows from 1.23× to
1.02× as pressure rises — Module 7's *direction* is right — but the crossover
never arrives, because on Ada an fp32 GEMM runs out of registers long before it
runs out of shared memory. **The premise, not the reasoning, was wrong.**

Where the premise *is* true, and therefore where CUTLASS's swizzles earn their
keep: Tensor-Core GEMM (Modules 33–34), where one register holds an accumulator
for a 16×8×16 tile instead of a single element, so the register-per-FLOP ratio
collapses and shared memory becomes the binding constraint; and multi-stage
`cp.async` pipelines (Module 32), where three or four buffers of a large tile
are live at once. Module 43 covers CUTLASS's layouts in that setting. The
lesson to carry forward is not "padding wins" — it is **measure which resource
is binding before choosing which one to spend**.

### 9. Double buffering and software pipelining

Module 17's tile loop has two barriers, and Module 6 named the reason for the
second one: a **write-after-read hazard**. Threads that finish the inner product
early must not overwrite the tile that slower threads are still reading.

```
loop: [stage tile k] barrier [compute on tile k] barrier
```

Two shared buffers remove the hazard by construction — you write the buffer
nobody is reading — and let the global loads for tile `k+1` be *issued* before
the arithmetic on tile `k`, so their latency overlaps the FFMAs:

```cpp
LOAD_TILE(0); STORE_TILE(0); __syncthreads();
for (kt = 0; kt < K; kt += BK) {
    if (kt + BK < K) LOAD_TILE(kt + BK);     // global -> REGISTERS, issued early
    ... BK rounds of TM x TN fused ops on buffer `buf` ...
    if (kt + BK < K) STORE_TILE(buf ^ 1);    // registers -> the OTHER buffer
    __syncthreads();                          // the only barrier
    buf ^= 1;
}
```

**Why one barrier suffices.** The barrier at the end of iteration `i` guarantees
every thread has finished reading buffer `b` in iteration `i` before any thread
proceeds to iteration `i+1`, where the write target is `b` again (after two
swaps). Within iteration `i`, a thread writing `b^1` cannot collide with a
thread reading `b`. Both hazards are covered by the single barrier; the second
one had nothing left to do. This is Module 9's fence-versus-barrier distinction
applied concretely, and it is the germ of software pipelining Module 9 named.

**Measured — and it loses.**

| kernel | registers | shared B | blocks/SM | GFLOP/s |
|---|---|---|---|---|
| 8×8, single-buffered, padded | 124 | 8320 | 2 | 7114 |
| 8×8, double-buffered, padded | 144–150 | 16640 | **1** | 5779 |
| 8×4, single-buffered | 80 | 6272 | 3 | 7399 |
| 8×4, double-buffered | 107 | 12544 | 2 | 6833 |
| 4×4, single-buffered | 64 | 4224 | 4 | 6578 |
| 4×4, double-buffered | 63 | 8448 | 4 | **6651** |

Double buffering pays exactly once: at `4 × 4`, where it happens to cost neither
a register granule nor a block, it is worth **+1.1 %**. Everywhere else the
prefetch registers push the kernel over a register-file step and cost 10–19 %.

This is a real result and it should not be explained away. The reason it does
not pay here is that **the latency it hides is already hidden.** The whole
17.9 MB working set fits in the 48 MB L2 (Module 16 bounded on-chip service at
≥ 92 %), so the "global" loads being prefetched are mostly ~241-cycle L2 hits,
not ~575-cycle DRAM reads; and with two or three blocks resident there are
16–24 warps per SM to switch between while they land. Double buffering trades
registers — the binding resource — for latency tolerance the kernel does not
need. On a part where the tile really comes from DRAM, or at a problem size well
past L2, the balance changes.

**`cp.async` changes the arithmetic.** On sm_80+ a global→shared copy can bypass
the register file entirely (`cp.async` / `cuda::memcpy_async`), which removes
exactly the cost that sinks this implementation. That is why modern GEMM
pipelines are 3- and 4-stage and this one is not worth 2. **Module 32 owns
`cp.async`**, asynchronous copy, and the multi-stage pipeline; it is named here
and deliberately not used.

### 10. Warp-level tiling — the third level, named

There is a level between the block tile and the thread tile that this module
does not implement: the **warp tile**. A warp's 32 threads cover a
`WM × WN` region of the block tile, and making that region *compact* rather than
scattered means the 32 threads of a warp read overlapping slices of `As` and
`Bs`, so the broadcast and 2-way patterns of §8 are not accidents of the thread
mapping but a designed property. It also gives the register file's operand-reuse
cache something to work with: the SASS in §11 shows 192 of 256 `FFMA`s carrying
the `.reuse` flag, and a warp tile makes that systematic instead of incidental.

CUTLASS formalises exactly this as the hierarchy
`ThreadblockShape → WarpShape → InstructionShape`, and the third element is what
lets the same source target Tensor Cores. **Module 43 covers CUTLASS**; the
reason to know the name now is that when you read a real GEMM kernel, the
three-level hierarchy is the thing you are looking at.

### 11. The honest ceiling

Full ladder, one program, one rotated min-of-N sweep, 1027 × 2053 × 769:

| rung | GFLOP/s | × cuBLAS | % of the 18 000 ceiling |
|---|---|---|---|
| v0 naive (M16) | 1302–1345 | 0.16–0.18 | 7.2–7.5 % |
| v1 block tile 32×32 (M17) | 1372–1543 | 0.17–0.21 | 7.6–8.6 % |
| v2 + 1-D register tile, TM = 8 | 4278–4465 | 0.53–0.60 | 23.8–24.8 % |
| v3 + 2-D register tile, 8×8 | 7033–7802 | 0.86–1.04 | 39.1–43.3 % |
| v4 + transposed, padded A tile | 7373–8138 | 0.91–1.08 | 41.0–45.2 % |
| v4b 8×4, the sweep's winner | 7399–8336 | 0.91–1.08 | 41.1–46.3 % |
| v5 + double buffering | 5779–6144 | 0.73–0.80 | 32.1–34.1 % |
| `cublasSgemm` | 7258–8370 | 1.00 | 40.3–46.5 % |

**The final kernel is at 91–108 % of cuBLAS on this shape** — parity, within the
run-to-run spread of a laptop part. The two moves that mattered are the 1-D
register tile (3.0×) and the second register dimension (1.7× on top). Padding
the A tile is worth a further 1.08×. Everything after that is noise or a loss.

Do not over-read the parity. At 2048³ the same kernel measures 9335 GFLOP/s
against cuBLAS's 10032 — **93 %** — and the gap widens with size, which is the
honest picture: this kernel is competitive on an awkward mid-sized shape where
cuBLAS's heuristics have little to work with, and behind on the shapes cuBLAS
is tuned for. A well-built SGEMM of this kind normally lands at 60–90 % of
cuBLAS; landing at parity on one shape is a statement about that shape.

**What cuBLAS and CUTLASS do that this kernel does not:**

1. **Tensor Cores.** Neither side used them here (this is `cublasSgemm` on fp32
   data with default math mode). Enable TF32 and cuBLAS moves to a different
   ceiling entirely, at reduced mantissa precision. **Modules 33–34.**
2. **`cp.async` and, on sm_90+, TMA.** Asynchronous global→shared copy without
   register staging, which makes deep pipelines affordable — §9's measured loss
   becomes a win. **Module 32.**
3. **Warp specialization.** Dedicating some warps to producing tiles and others
   to consuming them, so the two never contend for issue slots.
4. **Tile shapes tuned per problem size.** cuBLAS dispatches among dozens of
   compiled kernels on `(M, N, K)`, transposition, and alignment. §6's table is
   one point of that search, done by hand.
5. **Split-K.** The one you should know by name.

#### Split-K, properly

Every decomposition in this module parallelises over M and N and leaves K as a
sequential loop inside each block. That is fine when `M·N` is large enough to
fill the machine: here the grid is 9 × 33 = 297 blocks against 40 SMs, seven
waves. It fails completely for a **skinny** problem — small M and N, large K.
Take M = N = 128, K = 65536. The grid is one block. **One SM works and 39 idle**,
no matter how good the kernel is.

**Split-K** partitions the reduction dimension: split K into `S` chunks, launch
`S` times as many blocks, have block `s` compute the partial sum over its chunk,
and combine the `S` partial results. Two combining strategies:

- **Atomic split-K** — each block does `atomicAdd` into C. Cheap, needs no extra
  memory, and by Module 10 its cost tracks addresses-per-warp; but float
  `atomicAdd` is non-deterministic (Module 10 measured ten distinct bit patterns
  in ten runs of the same reduction), so the result is not bitwise reproducible.
- **Workspace split-K** — each block writes its partial `M × N` tile to a
  `S × M × N` workspace, and a second kernel reduces over `S`. Deterministic,
  costs `4·S·M·N` bytes and a second launch. This is what cuBLAS does when it
  reports a workspace requirement.

The trade is arithmetic intensity against parallelism: splitting K by `S`
multiplies the C traffic by `S` while dividing the per-block work by `S`. You
split only until the machine is full. It is the standard answer for the tall-skinny
GEMMs that dominate attention and LLM decode, and Module 43 shows CUTLASS's
version.

---

## Hardware Mental Model

**Why a register can do what a cache cannot.** A register file read is part of
the instruction, not an instruction of its own. `FFMA R57, R16, R20, R57` names
three registers and issues in one slot. An L1 hit costs a separate `LDS`, an
address computation, an issue slot, and a scoreboard dependency; an L2 hit costs
241 cycles of latency on top (Module 4). The register file is the only level of
the hierarchy where an operand read is *free*, and register tiling is the
technique for moving operands there. Everything in this module follows from that
one asymmetry.

**Why the accumulators are the thing you cannot spill.** `acc[TM][TN]` is live
across the entire `K` loop — 97 iterations here — and is read and written once
per `k`. The prefetch registers of §9 are live for one iteration. The `rM`/`rN`
registers are live for one value of `k`. `ptxas` spills in roughly reverse
liveness order, so a register-constrained build spills the accumulators last and
hardest, and when it does you get a local-memory (DRAM) access in the innermost
loop, 256 times per k-tile. That is the 18.7× of §6, and it is why the register
budget must be planned rather than discovered.

**Why occupancy buys less here than anywhere else in the course.** Occupancy
exists to hide latency by having another warp ready to issue. A register-tiled
GEMM's inner loop is 282 instructions of which 256 are `FFMA` with no memory
access between them — there is almost no latency left to hide, and the FP32
pipes are the resource in contention. Two blocks of 8 warps is already more than
enough to keep 4 warp schedulers fed with back-to-back FFMAs. Occupancy stops
being the currency; issue slots and registers are the currency. This is the
general shape of the argument Module 19 and Module 20 will make.

**Where the operand-reuse cache comes in. ARCHITECTURE-SPECIFIC.** The SASS
below shows `.reuse` on 192 of 256 `FFMA`s. Ada's register file has a small
operand-reuse cache in front of it; when consecutive instructions read the same
source register in the same slot, the second read is served from the cache
instead of the register file banks, which relieves register-file port pressure.
A rank-1 update is the ideal shape for it: `rM[i]` is reused across all `TN`
values of `j` in consecutive instructions. You get this for free from the loop
structure, and you lose it if you interleave the products in a different order.

**Ada specifics, labelled. ARCHITECTURE-SPECIFIC:** 65536 registers/SM, 8-register
allocation granule, 255 registers/thread maximum, 1536 threads/SM, 102400 B
shared/SM with a 1024 B per-block reserve and 128 B granularity, 32 banks × 4 B,
the `max(2, D)` conflict cost law, 48 MB L2, the measured 18 000 GFLOP/s ceiling.
**PORTABLE CUDA CONCEPT:** the two-level reuse law `BM·BN/(BM+BN)` and
`TM·TN/(TM+TN)`, the AM–GM argument for square tiles, register tiling as a
rank-1 update, the transposed-operand-tile trick, the WAR hazard that the second
barrier exists for, double buffering, split-K, and "measure which resource is
binding before choosing which to spend".

---

## Code Walkthrough

### `example01.cu` — the ladder

**§A** prints the two-level ledger of §2 at runtime, so that 0.50, 16.00, 64.00,
0.89, 4.00 and 16.00 are numbers you can see rather than claims.

**§B** is the five kernels. The one to read closely is `gemmReg2D`'s inner loop,
which is §3 verbatim, and its loader:

```cpp
for (int u = 0; u < NLA; ++u) {
    int idx = tid + u*NT;
    int r = idx / BK, c = idx % BK;
    float v = (rowBase + r < M && kt + c < K)
            ? A[(size_t)(rowBase + r) * K + kt + c] : 0.0f;
    As[c*AP + ((LAYOUT == 2) ? (r ^ ((c & 3) << 3)) : r)] = v;
}
```

Three decisions in five lines: `idx/BK` versus `idx%BM` (coalescing, §5), the
`kt + c < K` guard (the silent boundary, §5), and `As[c*AP + r]` (the transpose,
§7) with `AP = BM + 4` (the conflict, §8).

**§D** times all eight configurations back to back in one rotated sweep with
`SWEEPS = NCFG`, after a 1500 ms streaming plus 500 ms compute warm-up, and
prints registers, blocks per SM and occupancy next to the throughput. Read the
occupancy column against the throughput column before reading anything else.

**§C** validates every rung with Module 16's `gemmValidate`, unchanged, in a
second untimed pass, with C prefilled to `+infinity`.

### `example02.cu` — resources

**§A** is §6's table. **§B** is the `__launch_bounds__` cliff. **§C** is the
padding-versus-swizzle experiment of §8 at two values of `BK`. **§D** computes
blocks per SM by hand from the four resource limits and checks it against
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, then shows that the register
limit binds in **every single configuration** — including the 32 KB one.

### The SASS, which is the actual evidence

```
nvcc -arch=sm_89 -O3 -cubin -o k.cubin exercise03_solution.cu
cuobjdump -sass k.cubin
```

Per k-tile the whole kernel is:

```
6 x LDG.E.CONSTANT      (+ 6 IMAD.WIDE for the addresses)
6 x STS
BAR.SYNC.DEFER_BLOCKING
--- 282 instructions ---
24  x LDS.128
256 x FFMA
--- end ---
BAR.SYNC.DEFER_BLOCKING
```

and the body between the barriers begins

```
/*0b40*/  LDS.128 R16, [R43] ;
/*0b50*/  LDS.128 R20, [R41.X16+0x1080] ;
/*0b60*/  LDS.128 R24, [R43+0x10] ;
/*0b70*/  FFMA R57, R16.reuse, R20, R57 ;
/*0b80*/  FFMA R58, R16.reuse, R21, R58 ;
/*0b90*/  FFMA R59, R16.reuse, R22, R59 ;
/*0ba0*/  FFMA R70, R16,       R23, R70 ;
/*0bb0*/  FFMA R68, R17.reuse, R20, R68 ;
...
```

Three `LDS.128` — two for the eight A values, one for the four B values — then
32 `FFMA`s. **256 FFMA out of 282 instructions is 90.8 % arithmetic density**,
against 25 % for the naive kernel's `LDG, LDG, IMAD.WIDE, FFMA`. 192 of the 256
carry `.reuse`. There is no clearer statement of what register tiling does: it
did not make the memory faster, it **deleted the memory instructions**.

`-Xptxas -v` on the same kernel:

```
ptxas info : Compiling entry function '_Z9gemmYoursiiifPKfS0_fPf' for 'sm_89'
ptxas info : Function properties for _Z9gemmYoursiiifPKfS0_fPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 80 registers, used 1 barriers, 6272 bytes smem, 400 bytes cmem[0]
```

80 registers for 32 accumulators plus 12 operand registers plus addressing;
6272 bytes of shared memory = `8·132·4` (padded Aᵀ) + `8·64·4` (B); no spills.
`0 bytes spill stores` is the line to check first on every build in this module.

---

## Check Your Understanding

1. §2 shows that the FMAs-per-global-load ratio of a block tile,
   `BM·BN/(BM+BN)`, does not contain `BK`. A colleague concludes that `BK` is
   therefore a free parameter and sets it to 32 to "reduce the number of
   barriers by 4×". Their kernel gets slower. Give the two distinct mechanisms
   by which increasing `BK` costs performance even though it changes neither the
   global traffic nor the arithmetic, and say which of the two you would expect
   to dominate on a GPU with twice this one's register file.

2. The `8 × 4` and `4 × 8` configurations in §6 have identical accumulator
   counts, identical shared-memory footprints and identical values of
   `TM·TN/(TM+TN)`, and measure 1.57× apart. Two mechanisms are named in the
   text. Design a *third* configuration — you may change `BM`, `BN`, `BK`, `TM`,
   `TN` and the thread count — that isolates one of the two mechanisms while
   holding the other fixed, and state what you would expect to measure.

3. §9's double-buffered kernel is slower than the single-buffered one, and the
   explanation given is that it spends registers to hide latency that the L2 was
   already hiding. Someone proposes testing that explanation by re-running both
   kernels at M = N = K = 8192, where the working set is 768 MB and cannot be
   L2-resident. Explain precisely what you would expect to happen to the *ratio*
   between the two kernels, then explain why the experiment as described is
   confounded, and give the change you would make to fix it.

4. Module 7 concluded that an XOR swizzle would beat padding in GEMM because
   shared-memory capacity limits tile size; §8 measured the opposite, on the
   grounds that the register file binds first. Construct the fp32 GEMM
   configuration on *this* GPU for which Module 7's conclusion would in fact be
   correct — give `BM`, `BN`, `BK`, `TM`, `TN`, threads per block, and the
   resulting register and shared-memory footprints — or prove that no such
   configuration exists. If it exists, say whether it would be a kernel anyone
   should ship.

Answers: `solutions/module18/check_your_understanding.md`.

---

## Exercises

### Exercise 1 — `exercise01.cu` — the two-level tile hierarchy, by hand

Write the register-tiled kernel: `BM = 128, BN = 64, BK = 8, TM = 8, TN = 4`,
256 threads. The shared-memory declarations (transposed A tile, pitch `BM + 4`)
are given; everything else is yours.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

| TODO | requirement |
|---|---|
| 1 | `rowBase`, `colBase`, `tRow`, `tCol`. Blocks are launched 1-D; decide which of `tRow`/`tCol` varies fastest across a warp and why. |
| 2 | **Design:** stage both tiles cooperatively. The thread-to-element mapping is yours and is not free — apply Module 5's sector count to your choice. Three boundaries, one of them silent. |
| 3 | The inner product. Each shared location read once per `k`, then reused from registers. |
| 4 | The epilogue: `TM × TN` stores, guarded per element, honouring the `beta == 0` contract. |
| 5 | Two predictions: FMAs per scalar shared read, and the speedup over the Module 17 baseline. |

Validation: 10 points — correctness at four shapes including two where every
block is partial (4), a performance gate at 4.2× the Module 17 baseline (2), and
the two predictions (4). The gate exists because the one mistake that matters
here changes no output value — and, as the answer key documents with
measurements, three of the four mistakes you would expect to matter do not.

### Exercise 2 — `exercise02.cu` — the register/occupancy trade

A sweep at **128** threads per block, so none of Example 2's numbers transfer.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

| TODO | requirement |
|---|---|
| 1 | `occupancyBlocks(regs, smem, threads)` and the binding resource, checked against `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on eleven real kernels. One fact you need is deliberately not given; a model that gets 9/11 has the wrong granule. |
| 2 | The two-level reuse law, both halves, hashed. |
| 3 | Two predictions: what *kind* of configuration wins the sweep, and what forcing 100 % occupancy does. |
| 4 | **Design:** `chooseTile()` — pick the fastest configuration from the resource table alone, with no timings. Neither "maximise occupancy" nor "maximise reuse" nor their product is correct on this data. |
| 5 | The smallest `__launch_bounds__` minimum-blocks argument at which `ptxas` spills. Computable with a pencil. |

Validation: 10 points. All eleven occupancy predictions must match.

### Exercise 3 — `exercise03.cu` — design the whole kernel

Minimal scaffolding: a problem statement, `gemmValidate`, a timing harness, a
baseline, a gate. The kernel body, the shared-memory layout, the loader mapping,
the barriers, the vectorization and the epilogue are all yours.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

| TODO | requirement |
|---|---|
| 1 | The tile hierarchy: `BM, BN, BK, TM, TN`, threads. Checked for internal consistency before anything runs. |
| 2 | The ledger — accumulators per thread, shared bytes per block, FMAs per global load — committed *before* you write the kernel and checked against `cudaFuncGetAttributes`. |
| 3 | **The kernel. All of it.** |
| 4 | The launch configuration, correct at 37 × 53 × 11 as well as 1027 × 2053 × 769. |
| 5 | Where you predict you will land on the FP32 ceiling. |

Validation: 10 points — four shapes correct (4), the ledger (2), a 4.5× gate
over the Module 17 baseline (2), the prediction (2). Read the SASS before you
tune anything: if `FFMA` is not most of the instruction stream between the
barriers, no amount of tuning will fix the design.

---

## Prediction

Commit these to writing before you build anything.

1. Module 17's tiled kernel is at roughly 8 % of this GPU's FP32 ceiling and its
   FMAs-per-global-load ratio is 16 — already past Module 16's 6.5 threshold.
   Write down, in one sentence, why it is nevertheless not compute-bound, and
   then write down the number that *is* 0.50 for that kernel.

2. Write down the `TM × TN` you expect to be fastest at 256 threads per block,
   and the occupancy you expect it to run at. Then write down what you expect
   from the same kernel forced to 100 % occupancy — as a ratio, with a sign.
   Most readers get the first within one step and the second wrong by an order
   of magnitude.

3. Module 7 predicted that an XOR swizzle beats padding in GEMM because capacity
   limits tile size. Before reading §8, write down which of the two you expect to
   win here and, more importantly, **which resource you expect to be limiting
   blocks per SM** in a 128×128×8 fp32 GEMM tile. The second answer is the one
   that decides the first.
