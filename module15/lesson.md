# Module 15 — Matrix Transpose

> Prerequisites: Modules 1–14. You need the sector-counting procedure and
> write-allocate (M5), shared memory and cooperative loading (M6), the bank
> map and the `max(2,D)` cost law (M7), `__syncthreads()`'s two guarantees (M9),
> L2 slice hashing (M10), and the compulsory-traffic floor and the habit of
> reporting `x floor` rather than GB/s (M11).
>
> What this module gives you: the four memory disciplines of Part II — coalescing,
> shared memory, bank conflicts, padding — welded into one skill, on the one
> problem that has no arithmetic to hide behind.

---

## Concept

### Why transpose

```cpp
out[x][y] = in[y][x];
```

That is the whole algorithm. There is no arithmetic. No FLOPs, no reduction
tree, no accumulator, no tolerance, no numerical question of any kind. Every
microsecond a transpose costs is a memory-system event, and you can name which
one. That makes it the cleanest laboratory in the course: when a transpose is
slow, the explanation is coalescing, shared memory, bank conflicts, padding, or
the DRAM partition the addresses landed in — and nothing else.

Module 7 deliberately declined to use transpose as its vehicle so that you would
arrive here with the bank-conflict machinery intact and the problem unspoiled.
This is where it gets spent.

### The speed of light is a copy, and you must measure it first

Module 11 taught you to compute the compulsory traffic before writing the
kernel. For a transpose of an `H × W` matrix of 4-byte elements:

- every element is read exactly once: `4·W·H` bytes;
- every element is written exactly once: `4·W·H` bytes;
- nothing is read twice in the sense that matters, because a correct transpose
  touches each input element once.

So the compulsory traffic is **2N**, identical to a plain copy of the same
matrix. That is not an analogy. A transpose and a copy of the same matrix have
**exactly the same traffic model**, and they differ only in which address each
byte is written to. Therefore:

> **The correct denominator for a transpose is a measured copy of the same
> matrix, with the same block shape and the same instruction count — not
> 432 GB/s, and not the naive transpose you started from.**

This matters more than it sounds. A transpose reported as "1.8× faster than the
naive version" tells you nothing about whether you are finished. A transpose
reported as "94% of a copy of the same matrix" tells you there are six points
left on the table and no more. On this GPU the measured copy ceiling is
**373.1–383.6 GB/s, i.e. 86.4–88.8% of the 432 GB/s peak** — itself below
Module 12's 410.5 GB/s pure-read streaming figure, because a 1:1 read/write
stream is harder on the DRAM than a pure read.

`example01.cu` measures four different copies to make sure the denominator is
honest:

| copy variant | measured | % of the 2-D tiled copy |
|---|---|---|
| 1-D `float4` grid-stride copy | 373.1 GB/s | 100.0% |
| 2-D tiled copy, loads hoisted (the ceiling) | 373.2 GB/s | 100.0% |
| 2-D tiled copy, load/store interleaved | 373.0 GB/s | 99.9% |
| copy **staged through shared memory** | 373.3 GB/s | 100.0% |

Four structurally different kernels, all within 0.1%. The 2-D traversal costs
nothing, the memory-level-parallelism difference between hoisted and interleaved
loads costs nothing *for a copy*, and a full shared-memory round trip — 4 `STS`,
a `BAR.SYNC`, 4 `LDS` per thread — costs **nothing**. The shared-memory
scratchpad is free here, which is exactly why staging is the right answer for
this problem. Remember this row; it is what makes the rest of the module
interpretable.

### Rung 1: the naive transpose, and which side to break

Write the obvious thing:

```cpp
int x = blockIdx.x * TILE + threadIdx.x;   // column of in
int y = blockIdx.y * TILE + threadIdx.y;   // row    of in
out[x * H + y] = in[y * W + x];
```

Count sectors with Module 5's procedure, per warp, per instruction. A warp is 32
consecutive `threadIdx.x` at fixed `threadIdx.y` (M3's linearization rule), so
`x` runs over 32 consecutive values and `y` is fixed.

- **Read** `in[y*W + x]`: byte addresses `4·(y·W + x₀ + L)` for `L = 0..31`.
  128 contiguous bytes, 32 B-aligned because `x₀` is a multiple of 32.
  **4 sectors.** 100% efficient.
- **Write** `out[x*H + y]`: byte addresses `4·((x₀+L)·H + y)`. Consecutive lanes
  are `4H` bytes apart. With `H = 8192` that is 32 KB. Every lane is in its own
  sector. **32 sectors** for 128 useful bytes — **12.5% efficient**, the
  saturation floor M5 established for any stride of 8 floats or more.

So the write moves 8× the bytes it needs. Predicted time, if DRAM traffic is the
only term: read 1N + write 8N = 9N against the copy's 2N, so 2/9 = **22% of
copy**. Measured: **34–49% of copy** across runs, typically 36–44%. The model
over-predicts the damage because the 32 sectors a warp touches are not wasted —
the *next* warp of the same block, and the other three iterations of the `j`
loop, want the neighbouring elements of those same sectors, and by then they are
in L2. Sectors are fetched once and used four times. The traffic amplification
is real but it is closer to 4× than to 8×.

### Rung 2: the same permutation, indexed from the other side

Nothing says the naive kernel has to break the write. Index the grid over the
*output* instead:

```cpp
int xo = blockIdx.x * TILE + threadIdx.x;   // column of out
int yo = blockIdx.y * TILE + threadIdx.y;   // row    of out
out[yo * H + xo] = in[xo * W + yo];         // write coalesced, read strided
```

Identical permutation, identical useful bytes. Now the **read** is 32 sectors
and the **write** is 4.

| | read sectors | write sectors | measured, guarded | measured, unguarded |
|---|---|---|---|---|
| strided **write** | 4 | 32 | 43.6% of copy | — |
| strided **read** | 32 | 4 | 67.5% of copy | **88.3% of copy** |

**A strided write costs about twice what a strided read costs**, and Module 11
already told you why: a partial-sector store cannot be pushed to DRAM, because
DRAM has no byte enables at that granularity. The memory system must read the
sector, merge, and write it back. M11 measured that as 3.95× for a stride-2
store of identical useful bytes; here it is the same mechanism inside a real
algorithm. **When you must leave one side uncoalesced, leave the reads.**

The last two columns of that table are a second lesson, and a surprise. The
strided-read kernel measures 67.5% of copy with the bounds guard
`if (xo < H && yo + j < W)` and **88.3% without it** — a 1.34× penalty for a
predicate that is true every single time on a matrix whose dimensions are
multiples of 32. The tiled kernel pays nothing for the same guard (0.997×,
measured in `example02.cu` Part C). The SASS says why:

```
plain (no guard)                       guarded
  LDG.E.CONSTANT R13, [R6.64]            @!P1 LDG.E.CONSTANT R15, [R2.64]
  LDG.E.CONSTANT R15, [R6.64+0x20]       @!P2 LDG.E.CONSTANT R14, [R2.64+0x20]
  LDG.E.CONSTANT R17, [R6.64+0x40]       @!P3 LDG.E.CONSTANT R0,  [R2.64+0x40]
  LDG.E.CONSTANT R19, [R6.64+0x60]       @!P1 STG.E [R12.64], R15
  STG.E [R4.64], R13                     @!P2 STG.E [R10.64], R14
  STG.E [R2.64], R15                     @!P3 STG.E [R4.64],  R0
  STG.E [R8.64], R17                          LDG.E.CONSTANT R3, [R2.64+0x60]
  STG.E [R10.64], R19                         STG.E [R6.64], R3
```

Same eight memory instructions, different order. Without the guard the compiler
hoists all four loads ahead of all four stores: four independent 32-sector
requests in flight at once. With the guard it interleaves, and at most three are
outstanding. That is Module 11's MLP result, appearing uninvited. It only bites
where the loads are expensive: on the coalesced tiled kernel, whose loads are
4 sectors each, the same reordering is invisible.

### Rung 3: stage the tile in shared memory

The two naive kernels are the two halves of Module 5's unsolvable case: *no
single thread-to-element mapping makes both sides contiguous*. Module 6 named
the fix and deferred the example to here. Load with one mapping, store with
another, and put a scratchpad in between where "contiguous" means nothing:

```cpp
__shared__ float tile[TILE][TILE];

int x = blockIdx.x * TILE + threadIdx.x;      // column of in
int y = blockIdx.y * TILE + threadIdx.y;      // row    of in
for (int j = 0; j < TILE; j += BROWS)
    if (x < W && y + j < H)
        tile[threadIdx.y + j][threadIdx.x] = in[(long long)(y + j) * W + x];

__syncthreads();

int xo = blockIdx.y * TILE + threadIdx.x;     // column of out, in [0,H)
int yo = blockIdx.x * TILE + threadIdx.y;     // row    of out, in [0,W)
for (int j = 0; j < TILE; j += BROWS)
    if (xo < H && yo + j < W)
        out[(long long)(yo + j) * H + xo] = tile[threadIdx.x][threadIdx.y + j];
```

Both global accesses are now 4 sectors per warp. The permutation happens in
SRAM. Measured: **94.0–97.6% of copy.** That is the whole optimization, and it
is worth 2.2× over the strided-write naive kernel.

Four things in that code are the ones people get wrong, and three of them
produce a kernel that runs, faults nothing, and passes on a square matrix:

1. **The shared subscripts are not mirror images.** The load writes
   `tile[ty+j][tx]`; the store reads `tile[tx][ty+j]`. In the load, `ty+j` is
   the row *within the input patch* and `tx` the column. In the store, the
   thread whose `tx` names a column of the *output* must fetch the tile cell
   whose *row* index is `tx`. Swapping them in only one place transposes twice
   and gives you back the identity.
2. **`blockIdx.x` and `blockIdx.y` swap roles across the barrier.** The block
   that consumes input tile `(bx, by)` produces output tile `(by, bx)`. Writing
   `xo = blockIdx.x*TILE + threadIdx.x` instead compiles, runs, and is wrong
   everywhere — that one you will catch.
3. **The output's leading dimension is `H`, not `W`.** This is the one that
   passes. On a square matrix `W == H` and the bug is invisible; on 4093 × 2049
   it either produces garbage or reads out of bounds. Every harness in this
   module validates on a square power-of-two matrix *and* on 4093 × 2049 for
   exactly this reason.
4. **The two guards are different.** The load guard tests against `(W, H)`; the
   store guard tests against `(H, W)`. Same trap as (3), same square-matrix
   camouflage.

### Rung 4: the 32-way bank conflict, and what it is actually worth

Apply Module 7's procedure to `tile[threadIdx.x][threadIdx.y + j]` with a
`[32][32]` float tile. Element `(r, c)` is at flat index `32r + c`, so
`bank(r,c) = (32r + c) mod 32 = c`. **The bank depends only on the column.**

- Load phase, `tile[ty+j][tx]`: `c = tx` varies over 0..31 → 32 distinct banks →
  **D = 1**.
- Store phase, `tile[tx][ty+j]`: `c = ty+j` is *constant across the warp* and
  `r = tx` varies. All 32 lanes hit bank `ty+j`, asking for 32 different words.
  **D = 32.** The worst access the hardware admits.

Both fixes are Module 7's, verbatim:

```cpp
__shared__ float tile[32][33];              // padding: bank = (r + c) % 32
```
```cpp
#define SWZ(r,c) ((r)*32 + ((c) ^ ((r) & 31)))   // swizzle: bank = c ^ r
```

and both drive `D` to 1 in both phases. Now the measurement, which is the reason
this module exists:

| | 8192 × 8192 (DRAM-bound) | 2048 × 2048 (L2-resident) |
|---|---|---|
| `tile[32][32]`, D = 32 | 94.0–97.6% of copy | — |
| `tile[32][33]`, padded | 90.4–96.8% of copy | 1.00× (reference) |
| XOR swizzle, 1024 floats | 89.1–96.3% of copy | 0.92–1.22× |
| **conflicted / padded** | **0.96–1.02×** | **2.94–3.25×** |

At DRAM scale, **removing a genuine 32-way bank conflict buys nothing.** Not
"a little"; nothing, to within run-to-run noise, in either direction. Module 7
measured the identical conflict costing **14.79×** on its microbenchmark and
**12.6–15.4×** on its exercise-2 kernel. The difference is not the hardware. It
is which resource is saturated.

Do the arithmetic. Per thread the tiled transpose issues 4 `LDG`, 4 `STG`,
4 `STS`, 4 `LDS` (confirmed in the SASS below). The four `LDS` cost 2 cycles
each conflict-free and 32 cycles each at D = 32, so the conflict adds
4 × 30 = 120 cycles of shared-pipeline occupancy per thread. Over a 256-thread
block that is 8 warps × 120 = 960 cycles, call it 0.5 µs of LSU time per SM at
2 GHz. In the same block the kernel moves 4096 floats in and 4096 out —
32 KB of DRAM traffic — which at 373 GB/s spread over 40 SMs takes about 3.5 µs.
The shared-memory term fits inside the DRAM term with room to spare, and a term
that fits inside the binding constraint is free.

Shrink the matrix until both buffers live in the 50 MB L2 and DRAM stops being
the constraint, and the same two kernels separate by **3×**. Nothing about the
kernels changed. This is Module 12's lesson restated on new hardware ground:
**you cannot rank optimizations without knowing which resource is saturated.**

**Padding or swizzle?** Module 7 measured padding beating an XOR swizzle by
**19%**, and explicitly warned that the answer flips where shared-memory
capacity limits the tile size. On this problem the answer is neither: they
**tie**. Measured `swizzle / padded` over seven independent runs on the L2-resident
2048² case: 0.92, 0.97, 1.02, 1.03, 1.06, 1.06, 1.22 — a distribution
straddling 1.0.
The reason is visible in the disassembly. The swizzle costs exactly **8 extra
`LOP3` instructions** (four in the store phase, four in the load phase):

```
Function : _Z14transposeTiledILi0EEvPKfPfii     (tile[32][32])
  @!P2 STS [R7.X4+0x400], R14          LDS R13, [R0.X4]
  @!P6 STS [R7.X4+0x800], R12          LDS R11, [R0.X4+0x20]
  @!P0 STS [R7.X4],       R8           LDS R11, [R0.X4+0x40]
  @!P1 STS [R7.X4+0xc00], R16          LDS R7,  [R0.X4+0x60]

Function : _Z14transposeTiledILi1EEvPKfPfii     (tile[32][33])
  @!P2 STS [R7.X4+0x420], R14          LDS R13, [R0.X4]
  @!P6 STS [R7.X4+0x840], R12          LDS R11, [R0.X4+0x20]
  @!P0 STS [R7.X4],       R8           LDS R11, [R0.X4+0x40]
  @!P1 STS [R7.X4+0xc60], R16          LDS R7,  [R0.X4+0x60]

Function : _Z16transposeSwizzlePKfPfii          (XOR swizzle)
  @!P1 LOP3.LUT R5,  R3, 0x1f, R0,  0x78, !PT
  @!P0 LOP3.LUT R11, R3, 0x1f, R18, 0x78, !PT
  @!P4 LOP3.LUT R9,  R3, 0x1f, R16, 0x78, !PT
  @!P3 LOP3.LUT R5,  R3, 0x1f, R8,  0x78, !PT
  ... 4 STS, then 4 more LOP3 and 4 LDS
```

Note also that the padded and unpadded kernels differ **only in the STS
offsets** (`0x400/0x800/0xc00` versus `0x420/0x840/0xc60`, i.e. a row stride of
32 floats versus 33) — same instruction count, same 22 registers, 4096 versus
4224 bytes of shared memory. Padding is genuinely free in instructions.

M7's kernel was **LSU-bound**: its inner loop did 32 × 64 shared accesses per
thread and essentially nothing else, so 10 extra `LOP3`s were 10 extra
instructions on the critical resource. The transpose issues 8 shared accesses
per thread against 8 global ones and is waiting on DRAM the entire time; 8 extra
integer ALU instructions disappear into the shadow of a 575-cycle memory
latency. **The right conclusion is not "padding wins" or "swizzling wins". It is
that the comparison is only decidable on a kernel whose bottleneck is the
shared-memory pipeline, and the transpose is not one.** Ship the padded version
because it is simpler; reach for the swizzle when shared-memory capacity is what
you are short of, which is Module 18's problem, not this one.

### Tile shape and block shape

The classic arrangement is a 32 × 32 tile handled by a **32 × 8** block, with
each thread moving four rows. Why not a 32 × 32 block, one thread per tile cell?

One representative run of `example02.cu` Part A (all ten configurations in one
rotated sweep of ten, min-of-10):

```
configuration                        smem B   rows        ms      GB/s  %ofcopy
copy ceiling, block (32,8)                0      4    1.4366     373.7   100.0%
tile 16x16, block (16,16) 256 t        1088      1    1.5170     353.9    94.7%
tile 16x16, block (16, 8) 128 t        1088      2    1.4950     359.1    96.1%
tile 32x32, block (32,32)1024 t        4224      1    1.8771     286.0    76.5%
tile 32x32, block (32,16) 512 t        4224      2    1.5566     344.9    92.3%
tile 32x32, block (32, 8) 256 t        4224      4    1.5038     357.0    95.5%
tile 32x32, block (32, 4) 128 t        4224      8    1.5181     353.6    94.6%
tile 64x64, block (64,16)1024 t       16640      4    1.4888     360.6    96.5%
tile 64x64, block (64, 8) 512 t       16640      8    1.4324     374.8   100.3%
copy ceiling, block (64,16)           16640      4    1.4486     370.6    99.2%

  blocks/SM from cudaOccupancyMaxActiveBlocksPerMultiprocessor:
    tile 32, block (32,32) 1024 thr, 4224 B smem : 1 blocks/SM = 32 warps
    tile 32, block (32, 8)  256 thr, 4224 B smem : 6 blocks/SM = 48 warps
    tile 64, block (64,16) 1024 thr,16640 B smem : 1 blocks/SM = 32 warps
```

Read this table with discipline, because only one row in it is a reproducible
result. Across three good-state runs the `%ofcopy` column moved as follows:
16×16 (16,16) 83.7 / 94.7 / 100.9; 32×32 (32,8) 85.2 / 95.5 / 97.3; 64×64
(64,16) 96.5 / 98.3 / 113.8. **That spread is larger than most of the
differences in the table**, so the only conclusions worth drawing are the ones
that survive it.

1. **The 1024-thread block is the worst configuration in every run** — 67.2%,
   76.5%, 78.1% — and it is the only row that is reproducibly far from the rest.
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` reports **1 block/SM = 32
   warps**, against 48 for the 256-thread block: a 1024-thread block cannot be
   packed twice into the 1536-thread limit (M3's 66.7% ceiling, met again). And
   it has no `j` loop, so each thread issues exactly one load and one store — one
   outstanding request per thread instead of four. Fewer warps *and* less ILP
   per warp, against a memory system that needs ≈116 kB in flight (M11's Little's
   Law arithmetic). **The `j` loop is not a convenience; it is the
   memory-level-parallelism mechanism, and the 1024-thread block is slow because
   it does not have one.** Note that `tile 64 × 64, block (64,16)` is *also*
   1024 threads and 1 block/SM and is *not* slow — it has a `j` loop of 4. The
   variable is outstanding requests per thread, not occupancy.
2. **A 16-wide tile is consistently 1–3% slower than a 32-wide one within any
   single sweep**, and the effect is small. With `TILE = 16` a warp covers two
   *different rows* of 16 floats each. The sector count is still 4 (16 floats =
   64 B, and the row start is 64 B-aligned), so M5's model says nothing is wrong,
   and the measurement agrees to within a couple of percent. If you expected a
   16-wide tile to be a disaster because "a warp is 32 wide", the sector model
   was right and the intuition was wrong: a 64 B run is two whole sectors, and
   two whole sectors are not a coalescing failure. The residual 1–3% is
   consistent with worse DRAM page locality (two 64 B runs 32 KB apart versus one
   128 B run) but this module cannot measure that directly, and 1–3% is not a
   result worth defending.
3. **A 64 × 64 tile is at the ceiling in every run** — 96.5%, 98.3% and, in the
   run whose copy sample was unlucky, 113.8% of it — and 97.6–100% of its own
   matched 64-wide copy. It costs 16640 B of shared memory and 1 block/SM and it
   never loses. If you want one shape to ship, it is this one; if you want the
   *safe* shape, 32 × 32 with a (32,8) block is never more than a few percent
   behind and uses a quarter of the shared memory.

### Partition camping: the famous pathology, tested honestly

The historical story (NVIDIA's 2009 transpose paper, GT200) is this: DRAM is
split into partitions selected by a few address bits; the blocks resident at any
instant are consecutive in `blockIdx`, so they march along one row of the grid;
on the output side consecutive blocks in a grid row are `TILE · ldOut · 4` bytes
apart; if that stride is a large power of two, every resident block lands in the
same partition and the effective bandwidth collapses to one partition's worth.
The classic mitigation is **diagonal block reordering** — keep the launch
geometry, remap `blockIdx` to tile coordinates so that concurrent blocks are
spread over both axes:

```cpp
int by = blockIdx.x;
int bx = (blockIdx.x + blockIdx.y) % gridDim.x;   // square case
```

`example02.cu` Part B tests it properly: the matrix, the tile, the grid and
every useful byte are held **fixed** at 8192 × 8192, and only the output leading
dimension moves, so the inter-tile stride changes while the work does not.

| `ldOut` | output tile stride | copy | tiled transpose | % of copy | diagonal | diag/tiled |
|---|---|---|---|---|---|---|
| 8192 | 1 048 576 B (2²⁰) | 373.4 | 360.7 | 96.6% | 317.8 | 1.135× |
| 8200 | 1 049 600 B | 372.0 | 358.6 | 96.4% | 331.4 | 1.082× |
| 8224 | 1 052 672 B | 375.0 | 361.3 | 96.4% | 325.8 | 1.109× |
| 8320 | 1 064 960 B | 377.6 | 359.2 | 95.1% | 323.0 | 1.112× |

(GB/s against the 2N model. A second run gave 92.6 / 92.1 / 93.9 / 92.2% and
diag/tiled 1.115 / 1.065 / 1.122 / 1.110×.)

**Partition camping is not observable on this GPU.** A stride of exactly 2²⁰
bytes performs within 1.5% of strides that are not powers of two — and in this
run it was the *fastest* of the four. The ranking is not monotone in the
power-of-two-ness of the stride, and the whole spread is smaller than the
run-to-run variation of the machine. Diagonal reordering is **6.5% to 13.5%
slower at every stride in every run**, and 9–15% slower in the fixed-size sweep
in `example01.cu`. It is a pessimization on Ada, reproducibly.

Two reasons, and Module 10 supplied the first. **L2 slice selection on this GPU
is hashed, not a bit slice**: M10 measured that K distinct addresses give K-way
parallelism only when they hash to K slices, and found that spreading bins by
4 KB halved contention where 32 B spacing made it worse — i.e. the mapping is
not a simple address-bit decode. A hash destroys the alignment between "stride
is a power of two" and "all requests go to one place". Second, the L2 is **48 MB**
and there are 40 SMs each with up to 6 resident blocks; hundreds of tiles are in
flight at once, drawn from many grid rows, and the request stream reaching DRAM
is nothing like the neat marching column the 2009 model assumed.

Diagonal reordering costs because it destroys locality it used to buy: with the
linear mapping, blocks with adjacent `blockIdx.x` read adjacent input tiles and
share L2 lines; the diagonal mapping deliberately scatters them.

**This is a clean negative result and it is worth as much as a positive one.**
The lesson is not "partition camping is a myth" — it was real, and it is real on
hardware with a small L2 and a bit-sliced partition map. The lesson is that a
mitigation carried forward from a 2009 architecture, unmeasured, costs 12% on
this one. Measure the pathology before you apply the cure.

### Boundaries: non-square, non-multiple, and what the guards cost

Everything above assumed the tile divides the matrix. It does not, in general.
Two independent questions.

**Correctness.** A partial tile means some `(x, y)` are outside the input and
some `(xo, yo)` are outside the output, and the two sets are *different* — they
are governed by `(W, H)` and `(H, W)` respectively. Both guards are needed, both
must use the right pair, and a thread that skips the load must not leave a stale
value in the tile that another thread will then store (Module 6's partial-tile
hazard; the harnesses in this module poison the output with `-7.0f` before every
run so that a missed cell cannot pass by accident). A thread outside the input
must still reach the barrier — Module 9's uniformity rule — which is why the
guard is around the *statement*, never around a `return`.

**Cost.** Essentially zero.

| | measured |
|---|---|
| tiled transpose 8192×8192, guarded | 1.5117 ms |
| tiled transpose 8192×8192, unguarded (legal only when 32 divides W and H) | 1.5126 ms |
| ratio | **0.999×** (0.997× in a second run) |
| tiled transpose **8191 × 8193** (neither dimension a multiple of 32) | 1.5179 ms, 353.7 GB/s |

The guard is predication (M3, M8), not branching, and it costs nothing on a
kernel waiting for DRAM. The 8191 × 8193 case runs at the same speed as the
square multiple because only 512 of 65 792 blocks (0.8%) touch a boundary at
all. **Boundary handling is a correctness problem, not a performance problem.**
`example01.cu` validates every kernel on 4093 × 2049 and `exercise01.cu` makes
`OVERALL: PASS` depend on it.

### Where you will meet this again

- **AoS → SoA on the device.** Module 5 deferred it here. An array of `N` records
  of `C` fields is an `N × C` matrix; the SoA form is its transpose. Everything
  in this module applies, with the wrinkle that `C` is small: the tile cannot be
  square in the record axis, and the shared pitch must be chosen against `C`,
  not against 32. When `C` is odd — 7 fields, say — `gcd(C, 32) = 1` and the
  strided shared read is conflict-free with no padding at all.
- **`NCHW → NHWC`.** Exercise 2. Per image this is a `C × (H·W)` transpose, and
  it is the layout conversion that stands between a PyTorch tensor and a
  Tensor-Core convolution.
- **Layout changes before GEMM.** Modules 16–18 need the B operand read down
  columns; a transpose is one of the two ways to arrange that, and *not*
  transposing — reading B's columns straight into a padded or swizzled shared
  tile — is the other. Module 18 owns the capacity argument that decides between
  padding and swizzling, which this module could not decide because its kernel is
  not LSU-bound.
- **`cub::BLOCK_LOAD_TRANSPOSE`.** Module 13 named it: CUB loads a striped
  arrangement from global memory (coalesced) and transposes it through shared
  memory into a blocked arrangement (which per-thread serial scans need). That is
  this module's kernel, at block scope, inside a library. Module 36 covers CUB
  properly.
- **Module 36** also owns `thrust`/`cub` device-wide primitives; a production
  transpose today is `cublas<t>geam` or a CUTLASS layout transform, and you
  should use them — after you can say what they are doing.

---

## Hardware Mental Model

**Why a strided write is worse than a strided read.** Follow one warp's store of
32 floats, 32 KB apart. The coalescer emits 32 sector requests. Each arrives at
L2 covering 4 of the sector's 32 bytes. L2 cannot forward a 4-byte write to
DRAM: the DRAM interface's minimum transaction is a 32 B burst and there is no
byte-enable path behind L2. So the sector must be *filled* from DRAM, merged in
L2, and eventually written back. One store instruction has generated 32 DRAM
reads and 32 DRAM writes. A strided *load* of the same pattern generates 32 DRAM
reads and nothing else. Hence the ≈2× asymmetry, and hence M11's 3.95×
measurement for a stride-2 store where the model said 2×. The merge happens at
L2, which is also why a *streaming* misalignment escapes the penalty entirely —
the neighbouring warp's half of the sector is still resident.

**Why shared memory is free here.** The SM's L1TEX unit contains the tag/data
arrays for L1 and the 32-bank shared array, and they are different ports. A
`STS` does not consume a global-memory request slot, does not allocate an MSHR,
and does not touch L2. A kernel that is waiting on DRAM has its LSU pipeline
mostly idle; putting 8 shared accesses per thread into that idle pipeline costs
nothing measurable. The measured staged-copy row (100.0% of the plain copy)
is the direct proof. This is the *opposite* of Module 6's stencil result, where
tiling was a net loss — and the difference is that M6's tiling added a barrier
and a halo tax to a kernel that was already at 72.6% of peak *and did not need
the scratchpad at all*. Here the scratchpad is not buying reuse; it is buying a
change of access pattern, and that is worth 2.2×.

**Why the barrier costs nothing.** `__syncthreads()` compiles to
`BAR.SYNC.DEFER_BLOCKING`. The block runs at the speed of its slowest warp at
the barrier (M6's convoying argument). In a transpose every warp does exactly
the same amount of work — four coalesced loads — so there is no slack to expose.
Convoying costs you when warps diverge in duration; a transpose is the most
uniform kernel in the course.

**Why `D = 32` is invisible at DRAM scale and 3× at L2 scale.** A conflict is a
*replay*: the instruction re-issues against the bank array once per distinct
word per bank (M7; M4's constant-memory serialization is the same mechanism, and
M10's atomic contention is the third instance). Replays consume LSU issue slots,
not DRAM bandwidth. When DRAM is the binding constraint, extra LSU slots are
spent out of a surplus. When the data is L2-resident the memory side gets ~4×
faster (measured apparent bandwidth 1671 GB/s versus 373) and the LSU becomes the
constraint — at which point the 16× replay factor on one instruction class shows
up as 3× on the kernel, diluted by the `LDG`/`STG` that did not get slower.
`D/2` bounds the shared-memory *term*, never the kernel (M7 said this; here is
the extreme case of it).

**Why the compiler did not vectorize the tile accesses.** M7 warned that 32
adjacent unrolled shared reads become 8 `LDS.128` and silently change the
experiment. Here the four tile reads per thread are `[R0.X4]`, `+0x20`, `+0x40`,
`+0x60` — eight *words* apart, not adjacent — so they stay four scalar `LDS`,
and the 32-way conflict is genuine. Verified in the SASS rather than assumed,
which is the habit M7 demanded.

**Why partition camping stopped mattering.** Two mechanisms. The address-to-slice
map is a **hash** (M10 measured its consequences directly), so "power-of-two
stride" no longer implies "same slice". And the L2 is 48 MB against the GT200's
256 KB: with 40 SMs × 6 resident blocks × 32 KB of working set, the request
stream arriving at the memory controller is drawn from ~240 tiles spread across
many grid rows, not from one marching front. The pathology needed a small cache
and a bit-sliced map; Ada has neither.

**Why tile width matters at all, given identical sector counts.** A 16-wide, a
32-wide and a 64-wide tile all have perfect sector efficiency by M5's count, and
they measure within a few percent of each other with the 64-wide one at the top
in every run. Whatever separates them is below the sector model: a warp in the
64-wide tile presents 256 contiguous bytes where the 16-wide one presents two
disjoint 64 B runs, and longer contiguous runs mean fewer DRAM row activations
per byte delivered and better bank-group interleaving inside the GDDR6 device.
That is a **hypothesis consistent with a 1–3% effect**, not a measured
mechanism — settling it needs `ncu`'s DRAM counters, which are unavailable on
this machine. State it that way. The honest summary is that the sector model
predicts these three shapes to be equal and they are equal to within a few
percent; the model's domain ends where the difference begins (M11).

---

## Code Walkthrough

### `example01.cu` — the ladder

```
nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
.\example01.exe
```

Ten kernels, timed back to back in one rotated sweep of ten (spec §12 rule 9
requires `SWEEPS >= NCFG`), min-of-10, after a **1500 ms** duration-based warm-up.
The warm-up length is not decoration: Module 12 measured the identical ceiling
kernel at 372–373 GB/s after 400 ms and 410.5–410.7 GB/s after 1500 ms, because
400 ms ramps the SM clock but not the memory P-state.

Observed (one good-state run; see the caveat below):

```
kernel                                      ms   GB/s(2N)   %ofcopy    %of432
0 linear 1-D float4 copy (stream)       1.4390      373.1    100.0%     86.4%
1 2-D tiled copy, hoisted (CEILING)     1.4387      373.2    100.0%     86.4%
2 2-D tiled copy, interleaved ld/st     1.4395      373.0     99.9%     86.3%
3 naive  coalesced rd / strided wr      3.3033      162.5     43.6%     37.6%
4 naive  strided rd / coalesced wr      2.1307      252.0     67.5%     58.3%
5 tiled  [32][32]  (D=32 conflict)      1.5307      350.7     94.0%     81.2%
6 tiled  [32][33]  padded               1.5913      337.4     90.4%     78.1%
7 tiled  XOR swizzle                    1.6146      332.5     89.1%     77.0%
8 tiled  padded + diagonal blocks       1.7863      300.6     80.5%     69.6%
9 shared-staged copy (no permute)       1.4382      373.3    100.0%     86.4%
```

Part D decomposes the ceiling so you can see there is nothing hidden in it:

```
  2-D tiled copy / 1-D linear copy : 1.001x  (cost of the 2-D traversal)
  interleaved / hoisted copy       : 1.000x  (memory-level parallelism)
  staged copy / hoisted copy       : 0.997x  (cost of the tile round trip)
  padded transpose / staged copy   : 1.036x  (cost of the permutation)
```

Read that last chain again. Traversing the matrix in 32 × 32 tiles instead of
linearly: free. Hoisting the loads: free. A complete shared-memory round trip
with a barrier: free. **The permutation itself — the only thing a transpose
does — costs 3.6%.** Everything else in this module is about not squandering
that.

Part E repeats the three tiled variants on a 2048 × 2048 matrix whose two
buffers fit inside the 50 MB L2, and labels the result loudly as not a DRAM
number:

```
PART E -- L2-RESIDENT 2048x2048 (16 MiB/buffer). NOT a DRAM number.
  5 tiled  [32][32]  (D=32 conflict)   0.0603 ms   556.9 GB/s(apparent)  3.001x of padded
  6 tiled  [32][33]  padded            0.0201 ms  1671.0 GB/s(apparent)  1.000x of padded
  7 tiled  XOR swizzle                 0.0246 ms  1365.9 GB/s(apparent)  1.223x of padded
```

1671 GB/s is 387% of DRAM peak. It is real and it is not bandwidth; it is L2.
The point of the row is the `3.001x`.

**A caveat you must read before quoting any absolute number from this module.**
This is a 75 W laptop part. During authoring the *same binary on the same data*
produced a copy ceiling of 373 GB/s in one run and 98 GB/s twenty minutes later,
reproducibly, with `nvidia-smi` reporting the software power cap (`0x4`) and the
memory clock stepping 8801 → 8001 MHz. Ratios between configurations survive
that far better than absolutes but not perfectly: the naive strided-write kernel
measured 34–49% of copy across states. Every claim in this lesson that matters is
stated as a ratio and was reproduced in at least three separate runs.

### `example02.cu` — shape, camping, boundaries

```
nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
.\example02.exe
```

Part A is the shape table above, with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
printing the blocks/SM that explains the 1024-thread row. Part B is the
leading-dimension sweep that kills partition camping. Part C prices the guards.
The validation pass runs every shape on 1024 × 1024 and on 4093 × 2049.

The generic kernel is templated on both tile and block extent, which is how the
sweep stays honest — one body, nine configurations:

```cpp
template <int TILE, int BROWS>
__global__ void tTiled(const float* __restrict__ in, float* __restrict__ out,
                       int W, int H)
{
    __shared__ float tile[TILE][TILE + 1];
    ...
}
```

`TILE + 1` is odd whenever `TILE` is even, and `gcd(odd, 32) == 1` is M7's
condition. That one `+ 1` covers tiles of 16, 32 and 64 without a special case —
which is Exercise 3's TODO 4 generalized.

---

## Check Your Understanding

Answers in `solutions/module15/check_your_understanding.md`. None of these can be
looked up.

1. A colleague reports that their transpose runs at 340 GB/s, "79% of the
   432 GB/s peak", and asks whether it is worth optimizing further. Explain why
   that number, on its own, cannot answer the question, and state the *two*
   measurements you would ask them for instead. Then: for a transpose
   specifically, is there a case where 79% of peak means the kernel is finished?
   Give one and justify it from the traffic model.

2. The tiled transpose stores `tile[ty+j][tx]` and loads `tile[tx][ty+j]`. A
   colleague argues that since transposing twice is the identity, you could
   equally store `tile[tx][ty+j]` and load `tile[ty+j][tx]`, and that the two
   versions must perform identically because they move the same bytes through
   the same banks. Exactly one half of that argument is right. Say which, say
   what the conflict degree of each phase becomes in the swapped version, and
   predict the measured ratio between the two versions at 8192 × 8192 and at
   2048 × 2048.

3. You are given a transpose kernel that is correct on every square matrix and
   on every matrix whose dimensions are both multiples of 32, and wrong on
   4093 × 2049. Without reading the kernel, name the two most likely defects
   and give, for each, a 2 × 3 matrix and a tile size for which you could
   distinguish them by looking at the output alone. Then explain why a test
   suite built from square power-of-two matrices — the default choice of almost
   every benchmark — is structurally incapable of finding either.

4. Module 7 measured a 32-way bank conflict costing 14.79× on a microbenchmark.
   This module measures the same conflict costing 1.00× on a DRAM-bound
   transpose and 3.0× on an L2-resident one. All three numbers are correct.
   Construct the general rule that reconciles them, stated as an inequality
   involving the shared-memory term and the memory term, and then use it to
   predict what the conflict would cost in a transpose of a matrix that fits
   entirely in **L1** — and say why you cannot actually build that experiment
   on this GPU.

---

## Exercises

### Exercise 1 — `exercise01.cu`: build the ladder

Starting from a measured copy, write the naive transpose, the tiled transpose,
and a conflict-free shared layout of your own design. Validated on a square
power-of-two matrix **and** on 4093 × 2049.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

| TODO | requirement |
|---|---|
| 1 | The naive kernel's two global index expressions. Count the sectors on each side with M5's procedure before you write them. |
| 2 | The tiled kernel: the shared store index, the mechanism that lets a thread read what another thread wrote, the output block coordinates, and the shared load index. Four separate places to be wrong; three of them pass on a square matrix. |
| 3 | **Design.** A shared-memory layout with conflict degree ≤ 2 in both phases, at most 32×33 floats, expressed as an injective map `shIdx(r,c)`. The harness simulates both phases and tells you the degrees; it does not tell you the technique, and there is more than one answer. |
| 4 | **Prediction.** Which of three bands each of the four versions lands in as a fraction of the copy, plus whether removing the 32-way conflict is worth more or less than 1.25× on this kernel. |

**Validation.** All four versions must be correct on both matrices, TODO 3 must
be injective, in range and conflict-free in both phases, and all five predictions
must be right. `SCORE: 10/10` is required for `OVERALL: PASS`. The program then
re-times your two tiled versions on an L2-resident matrix, which is where the
answer to TODO 4's last part changes.

### Exercise 2 — `exercise02.cu`: NCHW → NHWC

A real layout conversion that is not a plain transpose: convert a
128 × 67 × 57 × 57 activation tensor from channels-major to channels-last. You
get a copy ceiling and a working naive kernel. You design everything else.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

| TODO | requirement |
|---|---|
| 1 | **Design.** The three grid dimensions and what each one indexes. The CUDA limits on `gridDim.y` and `gridDim.z` are 65535 and on `gridDim.x` are not; the harness rejects a grid that does not cover the tensor or that launches more than twice the tiles needed. |
| 2 | The load phase: coalesced over the NCHW-contiguous axis, staged into shared memory, guarded on both axes, with no stale tile cells. |
| 3 | The store phase: coalesced over the NHWC-contiguous axis. Work out which axis that is before you write a subscript. |
| 4 | **Design.** The shared-memory layout, as in Exercise 1. |
| 5 | **Prediction.** The naive kernel's band, your kernel's band, and what fraction of the launched tile cells actually hold data given `C = 67`. |

**Validation.** Every element of the output is checked against the index formula.
Your kernel must reach **80% of the measured copy**. `SCORE: 7/7` required.

### Exercise 3 — `exercise03.cu`: count it before you run it

Seven real transpose kernels, given and correct. For each, predict on paper the
read sectors, the write sectors, and the two shared-memory conflict degrees —
28 numbers — then implement M5's and M7's counting procedures so your own code
can check you, then watch the measurement disagree with all of it.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

| TODO | requirement |
|---|---|
| 1 | `sectorsOfWarp` — M5's procedure. Structurally tested for order-independence, broadcast, contiguous, stride-2 and stride-8. |
| 2 | `degreeOfWarp` — M7's procedure. Structurally tested, including the broadcast-pairs case that returns 2 if you count lanes instead of distinct words. |
| 3 | The 28-cell paper table. Scored against an FNV hash; the answers are not in the file. |
| 4 | **Design.** `padPitch(tileW, elemBytes)` — the smallest conflict-free pitch, for tiles of 32/64/96/128 and elements of 4/8/16 bytes, brute-force checked including the phase split. Derive the condition; do not search. |
| 5 | **Prediction.** What removing the 32-way conflict is worth, at 8192 × 8192 and at 2048 × 2048. |

**Validation.** `SCORE: 6/6` and all seven kernels numerically correct.

---

## Prediction

Commit to these in writing before you build anything.

1. **The copy ceiling.** Before running `example01.cu`, write down what fraction
   of 432 GB/s you expect a 1-read-1-write copy of a 256 MiB matrix to reach, and
   whether staging that copy through shared memory (4 `STS`, a barrier, 4 `LDS`
   per thread, moving not one extra byte through DRAM) will make it slower by
   more than 2%, less than 2%, or not at all. Then write down the number you
   would have predicted for a *pure read* stream, and reconcile the two.

2. **The 32-way conflict.** `tile[32][32]` read down a column is the worst
   shared-memory access the hardware admits, and Module 7 measured it at 14.79×.
   Predict, to one decimal place, the whole-kernel ratio between the unpadded and
   the padded tiled transpose at 8192 × 8192. Then predict it again for a
   2048 × 2048 matrix. If your two numbers are the same, you have not used
   anything Module 12 taught you.

3. **Partition camping.** Predict whether diagonal block reordering will be
   faster, slower, or indistinguishable on this GPU, and by how much. Name the
   specific property of the Ada memory system that decides it, and state what
   would have to be different about the hardware for the 2009 answer to be right
   again.

4. **Padding versus swizzle.** Module 7 measured padding beating an XOR swizzle
   by 19% on its kernel, and warned the answer flips under shared-memory
   pressure. Predict which wins here, and — more important — predict the *size*
   of the gap, in percent. Write down what property of the transpose kernel
   determines your answer, and what measurement would confirm it.
