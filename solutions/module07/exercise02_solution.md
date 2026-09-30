# Module 07 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

## What the kernel was doing wrong

The tile is `ROWS × 32` floats, stored row-major at `r * 32 + c`. The load phase
walks the flat index, so within a warp the row is constant and the column runs
0..31 — 32 distinct banks, conflict-free, exactly as Module 6 taught.

The compute phase is the opposite access. Thread `tid` owns row `tid` and reads
column `c`:

```cpp
acc += tile[tile_index<LAYOUT>(rr, c)] * W[(c + j) & (COLS - 1)];
```

For one instruction, `c` is fixed and the 32 lanes of the warp hold 32
*different* values of `rr` (they are 32 consecutive rows, and `ROWS = 192` is a
multiple of 32 so the `% ROWS` wrap preserves that). The bank is

```
bank(rr, c) = (32*rr + c) % 32 = c
```

— independent of `rr`. **All 32 lanes hit bank `c`, at 32 distinct words.**
That is `D = 32`, the worst case the hardware admits, and it is the canonical
shared-memory bug: the tile is written one way and read the other.

## TODO 1 — the degree

```cpp
static const int NAIVE_DEGREE = 32;
```

The harness verifies this structurally against the layout it owns, rather than
against the clock, because the clock is an upper bound on a *term*, not on the
kernel (see Performance reasoning).

Common wrong answers:

- **1**, from reasoning that each thread reads its own private row so there can
  be no conflict. Privacy of *data* is irrelevant; banks are a property of
  *addresses*. This is the exact mistake Exercise 3 is built around.
- **2**, from noticing that two lanes share... nothing. There is no pairing here.
- **16**, from half-warp thinking, which has not applied since compute
  capability 1.x.

## TODO 2 — the padded pitch

```cpp
#define PAD_PITCH 33
```

With pitch `P`, element `(r, c)` sits at `P*r + c`, so
`bank(r,c) = (P*r + c) % 32`. The column read holds `c` fixed and varies `r`, so
it visits `(P*r) % 32` offset by `c`, and the number of distinct banks is
`32 / gcd(P, 32)`. Conflict-free requires

```
gcd(P, 32) == 1     i.e.    P must be odd
```

`P = 33` is the smallest legal choice above `COLS = 32`. Then
`bank(r,c) = (33r + c) % 32 = (r + c) % 32`: row `r` is rotated by `r` banks, so
the column read visits all 32 banks, and the row write — `r` fixed, `c` varying
— also visits all 32. Both phases fixed with one character.

Common wrong answers and their symptoms:

- **`PAD_PITCH 40`**: "a multiple of 8, so it is nicely aligned." `gcd(40,32)=8`,
  so `D = 8`. The kernel gets 4× faster, which is enough to look like a fix and
  pass a casual review, and leaves three quarters of the penalty in place. This
  is the shared-memory twin of M5 Exercise 3's pitch 68, which satisfied the
  rule everyone remembers (`% 4 == 0`, so `float4` is legal) and was still 11 %
  short. **Alignment intuition is the wrong instinct here: in shared memory you
  want the pitch coprime with 32, i.e. odd, which is the opposite of every
  alignment rule you have learned so far.**
- **`PAD_PITCH 64`**: `gcd(64,32)=32`. A padded tile that is exactly as slow as
  the unpadded one, at double the shared memory. The harness's `> 2.0x` speedup
  requirement fails.
- **`PAD_PITCH 35`, `37`, `65`**: all correct (odd). They work and waste more
  memory than necessary. 33 is the minimum.

## TODO 3 — the swizzle

```cpp
return r * COLS + (c ^ (r & (COLS - 1)));
```

The array is still exactly `ROWS * COLS` floats — that clause in the problem
statement is what rules padding out and forces this.

**Why it works.** The column index is permuted by XOR with the row index. XOR
with a fixed value is an involution and therefore a bijection on `[0, 32)`, so:

- *Injective.* For fixed `r` the map `c ↦ c ^ (r&31)` is a bijection of
  `[0,32)`, so row `r` occupies exactly the same 32 slots it did before, in a
  different order. No element is overwritten and none is lost.
- *Column read conflict-free.* `bank = (32r + (c ^ (r&31))) % 32 = c ^ (r&31)`.
  Hold `c` fixed and run `r` over 32 consecutive values: `r&31` takes all 32
  values, and `c ^ ·` is a bijection, so the 32 lanes cover all 32 banks.
- *Row write conflict-free.* Hold `r` fixed and run `c` over 0..31: `· ^ (r&31)`
  is again a bijection, so the 32 lanes cover all 32 banks.

One XOR, zero bytes, both phases. The structural test in the harness checks all
three of those properties separately, and reports which one failed:

| symptom | cause |
|---|---|
| `out of range` | forgot the `& (COLS-1)` on `r`, so the XOR escapes `[0,32)` |
| `not injective` | used `+` instead of `^` without a modulus: `r*32 + c + r` collides |
| `column read still conflicted` | XORed with something constant across the warp, e.g. `c ^ (c>>1)` |
| `row write now conflicted` | swizzled the row instead of the column, e.g. `(r ^ c)*32 + c` |

That last row is the trap the problem statement warns about explicitly: it is
easy to fix one phase by breaking the other, and a harness that timed only the
compute phase would report a triumph.

**Why XOR and not `(c + r) % 32`.** Rotation works equally well for banking
(`bank = (c + r) % 32`, also a bijection both ways). XOR is preferred in
practice because it is a single `LOP3` with no carry chain and no modulus, and
because it composes cleanly with the larger swizzles real GEMM kernels use,
where the permutation has to respect 128-bit vector granularity as well as bank
granularity. Module 43 covers CUTLASS's layouts.

## TODO 4 — occupancy by hand

```cpp
const int per  = ((smemBytes + reserve + gran - 1) / gran) * gran;
const int bySm = (per > 0) ? (smemPerSM / per) : blocksPerSM;
const int byTh = threadsPerSM / threads;
int n = blocksPerSM;
if (bySm < n) n = bySm;
if (byTh < n) n = byTh;
return n;
```

The two terms the obvious division misses, both queried at runtime:

- **`reserve` = 1024 B per resident block**, from
  `cudaDeviceGetAttribute(..., cudaDevAttrReservedSharedMemoryPerBlock, 0)`.
  The driver takes this out of the SM's shared memory for each block it places.
- **`gran` = 128 B allocation granularity.** Each block's total is rounded up.

Numbers for this exercise:

| layout | pitch | asked | +reserve, rounded | 102400 / that | threads limit | max blocks |
|---|---|---|---|---|---|---|
| NAIVE | 32 | 24576 | 25600 | 4.00 | 1536/192 = 8 | **4** |
| PADDED | 33 | 25344 | 26368 | 3.88 | 8 | **3** |
| SWIZZLED | 32 | 24576 | 25600 | 4.00 | 8 | **4** |

All three agree with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` exactly.
Leave out the 1024 B reserve and NAIVE comes out as 5 and PADDED as 4 — both
wrong, and wrong in a way that would have you believing padding is free here
when it costs 25 % of your occupancy.

That is the real point of TODO 4: **3.1 % more shared memory cost one resident
block out of four.** Occupancy is a step function of shared memory per block,
and a pitch change that looks negligible in bytes can land you on the wrong side
of a step. `ROWS = 192` was chosen precisely so that it does.

## Synchronization / memory reasoning

One `__syncthreads()` between the cooperative load and the compute phase — the
Module 6 pattern. It is required because thread `tid` reads rows
`(tid + j) % ROWS` for `j = 0..63`, i.e. rows written by *other* threads.
Without the barrier this is a read-write race across the block. Module 9 makes
the ordering guarantee precise.

No second barrier is needed: after the load the tile is read-only for the rest
of the kernel's life, and nothing is written to shared memory again.

The barrier is identical in all three layouts, and it has to be — **the
swizzle changes where data lives, never when it is visible.** A common instinct
on first seeing a swizzle is to wonder whether it needs extra synchronization.
It does not. It is a pure address transformation applied identically by the
writer and the reader.

## Performance reasoning

Two deliberate anti-vectorization measures in the compute loop are worth
explaining, because without them the exercise measures the wrong thing.

The first version of this kernel read `tile[rr][c]` for `c = 0..31` with
`#pragma unroll`. Those are 32 *adjacent* floats, and ptxas duly fused them into
**8 `LDS.128` instructions**. Verified in SASS:

```
$ cuobjdump -sass e2.cubin | grep -o 'LDS[^ ;]*' | sort | uniq -c
     64 LDS
      8 LDS.128
```

A 16-byte shared load splits into 4 phases of 8 lanes (Example 2), and the
32-way column conflict becomes an 8-way-per-phase one. Measured speedup from
fixing it collapsed from ~16× to **3.15×** — a real number about a different
instruction than the one the reader was asked to analyse. Worse, the padded
layout has an odd pitch and so cannot be vectorized at all, which made the
comparison between the three layouts meaningless.

The shipped loop therefore reads four columns 8 floats apart per iteration and
is marked `#pragma unroll 1`. Every shared access is a scalar `LDS`, the bank
arithmetic of Example 1 applies verbatim, and all three layouts are compiled the
same way.

**Why the measured speedup is 12.6–15.4× and not 16×.** `D/2` bounds the
**shared-memory term**. The kernel also does a global load per element, an FFMA
per access, a barrier, and loop control, and none of those get slower when the
banks do. Writing the conflict-free time as `L + S` and the conflicted time as
`L + 16S`, a measured 12.6× implies the shared term is about 85 % of the
baseline. Thermal state moves the number: a cold run measured 15.44× and warm
runs measure 12.6×, because the non-shared term does not scale with clock the
same way the LSU term does. **Report the range, not one number.**

**Padding versus swizzling — the surprise.** Measured, warm, reproducible across
runs to about 2 %:

| layout | ms | vs NAIVE | blocks/SM | occupancy |
|---|---|---|---|---|
| NAIVE | 1.4851 | 1.00× | 4 | 50.0 % |
| PADDED | 0.1177 | 12.62× | 3 | 37.5 % |
| SWIZZLED | 0.1412 | 10.52× | 4 | 50.0 % |

**Padding wins by 19 %, despite having 25 % fewer resident blocks.** That is the
opposite of what the "swizzles are the zero-overhead fix" framing suggests, and
it is worth understanding rather than explaining away.

The SASS explains it. Instruction mix per kernel:

| layout | total instructions | `LOP3` | registers |
|---|---|---|---|
| NAIVE | 104 | 5 | 24 |
| PADDED | 136 | 10 | 24 |
| SWIZZLED | 144 | 20 | 23 |

The swizzle costs **10 extra `LOP3` instructions** — one XOR per address — in a
loop whose entire body is address arithmetic plus one FFMA. That is an ~6 %
larger instruction stream on a kernel where the shared pipeline is no longer the
bottleneck once the conflict is gone, and it shows up as ~19 % because those
extra integer ops land on the same issue slots the loads need.

Meanwhile the occupancy loss from padding costs almost nothing, because this
kernel is **throughput-bound on the LSU pipeline, not latency-bound**. Three
blocks of 6 warps already provide 18 resident warps per SM, far more than enough
to keep the shared pipeline saturated; the fourth block adds no throughput.
Occupancy only buys latency hiding, and there is no latency left to hide.

**When the answer flips**, and it is not a hypothetical: when shared memory
capacity is what limits the tile size. A GEMM kernel sized so that its A and B
tiles exactly fill the shared budget cannot pad — the padded tile does not fit,
so you must shrink the tile, which lowers arithmetic intensity, which moves the
kernel from compute-bound to memory-bound. There the swizzle's extra XOR is
irrelevant and its zero memory cost is everything. That is why CUTLASS swizzles.
The lesson is not "swizzle always"; it is **"pad when you have the memory,
swizzle when you do not, and measure, because the XOR is not free."**

## Expected output

Actual output on the RTX 3500 Ada (warm run):

```
TODO 3 structural test: ok

SM limits: shared 102400 B, threads 1536, blocks 24;
per-block driver reserve 1024 B, allocation granularity 128 B

layout     pitch  smem/blk     yours  cudaOcc occupancy          ms   vs best
---------------------------------------------------------------------------------------
NAIVE         32     24576         4        4     50.0%      1.4851    12.62x
PADDED        33     25344         3        3     37.5%      0.1177     1.00x
SWIZZLED      32     24576         4        4     50.0%      0.1412     1.20x

layout     numerics     vs NAIVE
--------------------------------------
NAIVE          PASS        1.00x
PADDED         PASS       12.62x
SWIZZLED       PASS       10.52x

TODO 1: you said the NAIVE compute-phase read has degree 32. Correct.
Degree D costs at most D/2 times a conflict-free access here, so
the shared-memory term could be up to 16.0x. The whole kernel
measured 12.62x. The gap is everything in the loop that is not a
shared load: the global fill, the FFMAs, the loop control. D/2
bounds the shared-memory term, never the kernel.

numerics: PASS   swizzle structure: ok   swizzle size unchanged: ok
occupancy model: ok   degree prediction: ok

OVERALL: PASS
```

Run-to-run variation: NAIVE 1.48–4.26 ms depending on thermal state, speedup
12.6–15.4×, PADDED/SWIZZLED ratio stable at 1.18–1.20×. The occupancy columns
are deterministic.

`ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` would
report roughly `31 × (warps × shared-load instructions)` for NAIVE and `0` for
the other two. That command could not be run on this machine: Nsight Compute
2026.1.0 is installed but returns `ERR_NVGPUCTRPERM` without elevated GPU
performance-counter permissions. The degrees here were verified against the
controlled degree sweep in `example01.cu` instead.

## The result that matters

**A 32-way bank conflict is one transposed index away from a conflict-free
access, and there are two fixes with genuinely different costs: padding spends
shared memory, swizzling spends instructions.** Which one is right depends on
which resource your kernel is actually short of — and on this kernel, at this
tile size, the answer is the one the folklore does not predict: padding is 19 %
faster despite losing a quarter of its occupancy, because the kernel is
throughput-bound and the extra 10 `LOP3`s cost more than the fourth block was
worth.

**Variation to try.** Set `ROWS` to 128 and rerun. The padded tile now fits at
the same 5 blocks per SM as the unpadded one — the occupancy cost of padding
vanishes entirely — and the padded-versus-swizzled gap should widen slightly,
since the swizzle's instruction overhead is unchanged while padding's only
disadvantage is gone. Then set `ROWS` to 320 and check whether padding still
fits under the 48 KB per-block default limit at all; `320 * 33 * 4 = 42240`,
which does, and `384 * 33 * 4 = 50688`, which does not — at which point the
swizzle is not a preference but the only option that compiles.
