# Module 03 — Thread Hierarchy

> Prerequisites: Module 1 (GPU architecture, SIMT, warps/blocks/grids), Module 2 (nvcc, execution-space qualifiers, launch syntax, error checking)
> What this module gives you: the ability to map any problem geometry onto a grid of threads, compute every index correctly at the boundaries, and — the part that actually matters — know in advance which 32 threads form a warp and which addresses that warp will touch.

---

## Concept

### The four built-in variables

Inside a `__global__` or `__device__` function, four variables are always in
scope. You never declare them; the compiler materialises them from hardware
special registers.

| Variable | Type | Meaning | Range |
|---|---|---|---|
| `threadIdx` | `uint3` | this thread's coordinate inside its block | `[0, blockDim)` per component |
| `blockIdx` | `uint3` | this block's coordinate inside the grid | `[0, gridDim)` per component |
| `blockDim` | `dim3` | block shape, in threads | set by the launch |
| `gridDim` | `dim3` | grid shape, in blocks | set by the launch |

`uint3` is a plain struct of three `unsigned int` (`.x`, `.y`, `.z`). `dim3` is
the same thing with a constructor that **defaults the unspecified components to
1**:

```cpp
dim3 a(256);          // (256, 1, 1)
dim3 b(32, 8);        // (32, 8, 1)
dim3 c(16, 16, 4);    // (16, 16, 4)
```

Read that default carefully. In a 1-D launch, `blockDim.y == 1` and
`gridDim.z == 1` — **one, not zero**. `threadIdx.y` is 0 because the thread's
coordinate along an extent of length 1 is 0; the *extent* is 1. The distinction
is not pedantry: every linearization formula below multiplies by `blockDim.y`
and `blockDim.z`, and a 0 there would annihilate the whole expression. Because
CUDA guarantees 1, you may always write the full three-dimensional formula and
it degrades gracefully to the 1-D case.

`dim3` components are `unsigned`. Mixing them with signed `int` in the same
expression triggers the usual C++ integer promotions, and
`blockIdx.x * blockDim.x + threadIdx.x` is computed in **32-bit unsigned**
arithmetic regardless of the type of the variable you assign it to. If the
product can exceed 2³²−1, you must cast *before* multiplying, not after.

### Why three dimensions exist

The GPU does not care. Hardware schedules *warps*, and a warp is 32 consecutive
threads in a single linear order. The 2-D and 3-D forms are a convenience for
the programmer: they let the compiler compute the three index components for you
instead of you dividing and taking remainders, and integer division on NVIDIA
hardware is expensive (no hardware divider; the compiler emits a multi-
instruction reciprocal sequence).

Rules of thumb:

| Grid/block shape | Use when |
|---|---|
| 1-D | the problem is a flat array (vector add, reduction, scan, histogram) |
| 2-D | the problem is a matrix or image and the natural tile is rectangular (stencil, transpose, GEMM tile) |
| 3-D | there is a genuine third index you would otherwise divide out: batch, depth, channel, time step |

A 3-D grid is never *required* — you can always flatten by hand — but writing
`blockIdx.z` costs nothing and writing `b = blockIdx.x / (gridX * gridY)` costs
an integer division per thread.

### The linearization rule

**This is the load-bearing idea of the module.**

Within a block, the threads are ordered by

```
tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)
```

**x varies fastest**, then y, then z. Exactly the memory layout of a C array
declared `T a[Z][Y][X]`.

Threads with `tid` 0–31 form **warp 0**. Threads 32–63 form warp 1. And so on.
There is no other rule, no runtime choice, no dependence on the grid shape. A
block of 256 threads is always exactly 8 warps, cut at those boundaries, whether
the block is declared `(256,1,1)`, `(32,8,1)`, `(8,32,1)` or `(4,8,8)`.

The grid uses the same rule one level up, although the *order in which blocks
are dispatched* is not architecturally guaranteed; only the coordinates are.

The consequence you must internalise:

```
block (32, 8)   -> warp w = the 32 threads with threadIdx.y == w.
                   One warp is one row of the tile.

block (8, 32)   -> warp w = threadIdx.y in {4w, 4w+1, 4w+2, 4w+3}, all x.
                   One warp straddles four rows of the tile.

block (1, 256)  -> warp w = threadIdx.y in [32w, 32w+32), threadIdx.x == 0.
                   One warp straddles thirty-two rows.
```

All three are legal. All three are correct. They differ only in what a *single
memory instruction* asks the memory system for, because a memory instruction is
issued per warp: 32 lanes present 32 addresses simultaneously, and the hardware
coalesces them into the smallest set of transactions that covers them. If the
32 addresses are consecutive 4-byte words, that is 128 contiguous bytes — the
best case. If they are 32 addresses `W` floats apart, that is 32 separate
transactions — up to 32× the traffic for the same instruction.

Module 5 makes the transaction arithmetic exact. Module 8 explains warps as
execution units (divergence, predication, reconvergence). This module's job is
narrower and prior to both: **given a launch configuration and an index
expression, name the 32 threads of a warp and the 32 addresses they touch.**
Exercise 3 is exactly that drill.

### Mapping problems onto threads

**1 thread : 1 element.** The default. Grid size is a function of the problem:

```cpp
int block = 256;
int grid  = (n + block - 1) / block;    // ceil divide
kernel<<<grid, block>>>(p, n);
```

**1 thread : N elements (thread coarsening).** Each thread handles a fixed
number of elements. Reduces total index arithmetic and increases
instruction-level parallelism, at the cost of a launch that is now tied to two
numbers instead of one. Modules 12 and 18 use it heavily.

**Grid-stride loop.** The robust default:

```cpp
__global__ void k(float* a, long long n)
{
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride)
        a[i] = f(a[i]);
}
```

Why this is the form to reach for:

1. **Correctness is independent of the launch configuration.** The same kernel
   is right for `<<<1,256>>>` and `<<<195313,256>>>`. You can therefore tune the
   grid for performance without re-auditing the indexing. Example 2 proves this
   by running the identical kernel at both extremes.
2. **The grid becomes a function of the machine, not of the data.** On this GPU,
   40 SMs × (1536/256) = 240 blocks of 256 threads fill every thread slot. A
   grid of 240 launches exactly one "wave" that stays resident for the kernel's
   whole life — no block scheduling churn, no tail wave.
3. **It enables persistent-kernel styles**, where a fixed set of blocks stays
   resident and consumes work. (Cooperative launches and grid-wide barriers are
   Module 29.)
4. **The stride is `gridDim.x * blockDim.x`, i.e. the whole grid**, so
   consecutive global thread ids still map to consecutive elements on every
   iteration and each warp still reads 128 contiguous bytes. A per-block stride
   would preserve correctness and destroy coalescing.

The bounds check did not disappear in the grid-stride form; it *became* the loop
condition `i < n`.

### Flattening, row-major, and stride

For a 2-D array of `h` rows and `w` columns stored row-major:

```cpp
offset = row * w + col;          // w is the row stride, in elements
```

Consecutive `col` are adjacent in memory; consecutive `row` are `w` elements
apart. Therefore **`threadIdx.x` must carry `col`**. Column-major (Fortran,
cuBLAS's native convention, MATLAB) reverses this: `offset = col * h + row`, and
then `threadIdx.x` must carry `row`. The rule is invariant even though the code
is not:

> Give `threadIdx.x` to the axis with the smallest memory stride.

For 3-D `[B][R][C]` row-major: `offset = (b * R + r) * C + c`, strides
`R*C`, `C`, `1`. So x → c, y → r, z → b.

**Stride vs width.** Nothing requires the row stride to equal the number of
useful columns. Allocating rows padded to a multiple of 128 bytes makes every
row start on a transaction boundary, which matters once you care about sector
counts. `cudaMallocPitch` returns such a padded allocation and reports the pitch
in bytes; you then index `row * (pitch / sizeof(T)) + col`. **Forward reference:
Module 5** covers alignment and why the pitch is worth the wasted bytes. Do not
use it yet; just stop assuming stride == width.

### Bounds checking

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= n) return;
```

Mandatory whenever `n` is not an exact multiple of the block size — which, in
real code, is always. `ceil(n/block) * block − n` threads in the last block have
no element; without the guard they read and write past the end of the
allocation. That is not a hypothetical: it is an out-of-bounds global access,
which `compute-sanitizer --tool memcheck` reports and which, unreported,
silently corrupts whatever follows in the allocator's arena.

The cost is small and worth being precise about. For every warp except the one
or two straddling the boundary, the predicate is **uniform**: all 32 lanes agree.
The compiler emits a compare into a predicate register and either predicates the
following instructions or branches over them; a uniformly-taken branch costs one
instruction of issue and no divergence. Only the straddling warp is genuinely
divergent, and there is at most one such warp per boundary in the whole grid.
Never skip a guard to "save a branch". Module 8 makes the predication story
precise.

Guard **every** dimension. `if (col < w)` alone, on a 2-D launch, lets the
excess rows run.

### Choosing a block size

Three constraints, in order of how often they bind:

1. **Multiple of 32.** A block of 100 threads is rounded up to 4 warps = 128
   thread slots; 28 slots are allocated and idle for the block's whole lifetime.
2. **1024 is usually a bad choice on sm_89.** The hardware limit is 1024 threads
   per block, but only **1536 threads per SM**. 1536/1024 = 1.5, so exactly one
   1024-thread block fits per SM: 1024/1536 = **66.7% occupancy ceiling**, before
   registers or shared memory have said anything. 512 gives 3 blocks = 100%;
   256 gives 6 = 100%; 128 gives 12 = 100% (the max-blocks-per-SM limit is 24).
3. **128 or 256 are the sane defaults.** Smaller blocks schedule more flexibly
   around tail effects; larger blocks amortise per-block work (and, from Module
   6, share more through shared memory).

Module 19 owns occupancy properly — register pressure, shared-memory footprint,
and the fact that maximum occupancy is frequently not maximum speed. For now:
multiples of 32, default to 256, and never pick 1024 without a reason.

### Computing the grid, and the overflow trap

```cpp
int grid = (n + block - 1) / block;
```

is the standard ceil-divide, and it is **wrong for large `n` in `int`
arithmetic**. If `n + block - 1` exceeds `INT_MAX`, the addition overflows —
undefined behaviour, and in practice a negative number, and in practice a launch
with a garbage or negative grid. With `n = 2147483000` and `block = 1024` the
expression evaluates to `-2097151` on this toolchain. Two safe forms:

```cpp
int grid = n / block + (n % block != 0);          // no intermediate overflow
long long grid = ((long long)n + block - 1) / block;   // widen first
```

Also know the launch limits: `gridDim.x` up to 2³¹−1, but **`gridDim.y` and
`gridDim.z` only up to 65535**. A 3-D decomposition that puts a large axis on y
or z fails to launch (`cudaErrorInvalidConfiguration`) — which `cudaGetLastError()`
after the launch will tell you, if you check it.

---

## Hardware Mental Model

**Where the indices come from.** `threadIdx`/`blockIdx` are not memory. They are
read-only special registers (`%tid.x`, `%ctaid.x`, `%ntid.x`, `%nctaid.x` in PTX)
that the SM populates when it creates the thread contexts for a block. Reading
one is a register move, not a load. This is why index arithmetic is cheap and
why there is no reason to cache `threadIdx.x` in a variable for performance —
though you should for readability.

**How a block becomes warps.** When the GigaThread engine assigns a block to an
SM, the SM allocates thread slots and register file space for
`ceil(threadsPerBlock / 32)` warps, and partitions the block's threads into those
warps *in linearized order*. The partition happens once, at block launch, and
never changes. A thread's lane within its warp is `tid % 32` — you can read the
hardware's own answer via the `%laneid` special register, and Example 1 checks
the formula against it for every thread of four different launch geometries.

**Why the partition order dictates memory behaviour.** The SM issues one
instruction per warp per issue slot. A `LDG`/`STG` (global load/store) issued by
a warp presents 32 addresses to the memory pipeline at once. The coalescing
hardware sorts them into the minimum number of aligned transactions. The DRAM
and L2 granularity on this architecture is a **32-byte sector**; an L1/L2 cache
line is 128 bytes = 4 sectors. So:

| Lane address pattern (4-byte elements) | Sectors requested | Useful bytes / fetched bytes |
|---|---|---|
| 32 consecutive, 128-B aligned | 4 | 128/128 = 100% |
| 8 consecutive per row, 4 rows (block (8,32)) | 4 | 128/128 = 100% of sectors, but 4 different cache lines |
| 32 addresses `W` floats apart (blockDim.x == 1) | 32 | 128/1024 = 12.5% |

The middle row is the one people get wrong. A block of (8,32) is *not*
catastrophic on this hardware, because 8 floats is exactly one 32-byte sector, so
the warp still asks for 4 sectors — the same count as the (32,8) case. What it
costs is **cache-line footprint**: the block touches 32 distinct 128-byte lines
instead of 8, so the L1 working set per block quadruples. Exercise 3 measures
both numbers; Exercise 1 measures the resulting time, and finds (8,32) and (32,8)
within a few percent of each other while `blockDim.x == 1` is ~4× slower.

That last case is the real cliff, and it is entirely predicted by the
linearization rule: with `blockDim.x == 1`, consecutive lanes differ in
`threadIdx.y`, so consecutive lanes are a full row apart in memory.

**Occupancy arithmetic, briefly.** An SM on sm_89 holds at most 1536 threads,
48 warps, and 24 blocks, and has 65536 32-bit registers. The binding constraint
is whichever runs out first. 256-thread blocks: 1536/256 = 6 blocks (under the
24-block cap), 100% of thread slots. 1024-thread blocks: 1 block, 66.7%. Module
19 adds registers and shared memory to this arithmetic.

**Why unused dimensions are 1.** The SM computes the block's thread count as the
product `ntid.x * ntid.y * ntid.z` and its warp partition from the same product.
A 0 would mean a block with no threads. The driver validates `dim3` components as
≥ 1 at launch; `dim3`'s constructor defaults exist so that the invariant holds
without the programmer thinking about it.

---

## Code Walkthrough

### `example01.cu` — index geometry explorer

Every thread records what it sees and the host checks the linearization formula
against the hardware's `%laneid`.

```cpp
__device__ __forceinline__ unsigned tidInBlock()
{
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}
```

and on the host:

```cpp
unsigned t = h[i].tx + h[i].bdx * (h[i].ty + h[i].bdy * h[i].tz);
if ((t % 32u) != h[i].lane) ++bad;
```

Across a 1-D launch, two 2-D launches and a 3-D launch — 1792 threads — the
mismatch count is 0. The formula is not a convention this course invented; it is
what the SM does.

Part A prints the degenerate components:

```
  threadIdx.y of thread 0 = 0, threadIdx.z = 0   (both 0)
  blockDim.y  = 1, blockDim.z  = 1   (both 1, NOT 0)
  gridDim.y   = 1, gridDim.z   = 1   (both 1, NOT 0)
```

Part B is the whole point of the module, printed as a roster. With `(32,8)`:

```
  warp 1 of block 0, lanes 0..31 = (threadIdx.x, y, z):
    (0,1,0) (1,1,0) (2,1,0) ... (31,1,0)
```

One warp, one value of `threadIdx.y`, 32 consecutive x. With `(8,32)` — the same
256 threads:

```
  warp 1 of block 0, lanes 0..31 = (threadIdx.x, y, z):
    (0,4,0) ... (7,4,0) (0,5,0) ... (7,5,0) (0,6,0) ... (7,6,0) (0,7,0) ... (7,7,0)
```

Four values of `threadIdx.y`, eight x each. Nothing about the *code* changed;
only `blockDim`.

Part C confirms the third term. With `(4,4,4)`, warp 0 is all 16 threads of
`z == 0` followed by all 16 of `z == 1` — z is the slowest axis, so a warp gets
two complete xy-planes.

### `example02.cu` — mapping n elements onto threads

SAXPY (`out[i] = a*x[i] + y[i]`) over `n = 50,000,003` floats — deliberately not
a multiple of the block size. 12 bytes of compulsory DRAM traffic per element,
so the achieved GB/s is directly comparable to the 432.0 GB/s ceiling.

The 1:1 kernel:

```cpp
long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
if (i < n)
    out[i] = a * x[i] + y[i];
```

The cast is on `blockIdx.x` *before* the multiply. `blockIdx.x * blockDim.x` in
32-bit unsigned overflows at 4.29e9 elements; casting the result afterwards
would not help.

The grid-stride kernel:

```cpp
long long stride = (long long)gridDim.x * blockDim.x;
for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
     i < n; i += stride)
    out[i] = a * x[i] + y[i];
```

Measured on this GPU:

| Variant | grid | ms | GB/s | % of 432 |
|---|---|---|---|---|
| 1 thread : 1 element | 195313 | 2.019 | 297.2 | 68.8% |
| grid-stride, one wave | 240 | 1.722 | 348.5 | 80.7% |
| grid-stride, n-sized grid | 195313 | 1.767 | 339.5 | 78.6% |
| grid-stride, 1 block | 1 | 18.821 | 31.9 | 7.4% |

Three things to read out of that table. First, all four **PASS** — the
grid-stride kernel is correct at a grid of 1 and at a grid of 195313, which is
the argument for the idiom. Second, the wave-sized grid is the fastest: 240
blocks, launched once, resident for the whole kernel, each thread doing ~814
elements; the 195313-block version pays block-scheduling overhead for the same
work. Third, a grid of 1 leaves 39 of 40 SMs idle and gets 7.4% of peak — the
launch configuration is a performance knob, and grid-stride is what lets you turn
it without touching the kernel.

Part D is host-only and shows the ceil-divide overflow described above.

---

## Check Your Understanding

Answer in prose before looking at `solutions/module03/check_your_understanding.md`.

1. A kernel is launched with `dim3 block(16, 16)` and reads
   `A[threadIdx.y * 16 + threadIdx.x]` from a 256-element row-major array. Which
   elements of `A` does warp 2 read, and how many 32-byte sectors does that
   request cost? Now the block is changed to `dim3 block(256)` and the index to
   `A[threadIdx.x]`, and warp 2 is examined again. Compare the two footprints and
   explain the difference purely from the linearization rule.

2. You are told that a kernel with `dim3 block(8, 8, 16)` (1024 threads) has a
   `__global__` load whose address depends only on `threadIdx.z`. How many
   *distinct* addresses does warp 0 present to the memory system, and how many
   does warp 5 present? Justify from the linearization formula, not by running
   anything.

3. A colleague removes the `if (i < n) return;` guard from a kernel, arguing that
   "the array is allocated with `cudaMalloc`, which rounds up, so the extra
   threads land in slack memory and the result is unchanged." Give two separate
   reasons this is wrong — one about correctness, one about what the argument
   reveals about their model of the allocator — and describe the tool output that
   would settle it.

4. A kernel processes `size_t n = 5000000000` elements with `block = 256`. The
   programmer writes
   `int grid = (int)((n + block - 1) / block);` and, inside the kernel,
   `unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) ...`.
   One of those two lines is fine and one is a bug. Say which, why, what the
   observable symptom is, and why `compute-sanitizer --tool memcheck` will
   **not** flag it.

---

## Exercises

### Exercise 1 — `exercise01.cu` (fill in the code, 2-D)

A 5-point weighted stencil with clamped boundaries on a **1021 × 733** row-major
float image — not square, not a power of two, not a multiple of any block
dimension you will pick. The same kernel then runs on a 4093 × 3079 image (50 MB
per array, larger than the 48 MB L2) with four block shapes that all contain 256
threads.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

- **TODO 1** — compute this thread's `row` and `col` from the built-ins. Decide
  which of x/y carries which, and be able to justify it in terms of what a warp
  reads and writes.
- **TODO 2** — the bounds guard.
- **TODO 3** — `makeGrid(h, w, block)`, which must be correct for *any* block
  shape the harness passes in, including `(32,8)`, `(8,32)`, `(256,1)` and
  `(1,256)`, and must remain correct if the image grows.

Validation compares all 748,393 (then 12,602,347) outputs against a CPU
reference with fp32 tolerance and reports the first mismatching `(row, col)`.
The harness also reports whether your grid even covers the image, threads
launched vs pixels, effective GB/s and % of the 432 GB/s peak for each block
shape.

### Exercise 2 — `exercise02.cu` (CPU → GPU)

You are given a triple-nested C++ loop over a batched tensor `in[37][1013][577]`
and a validating harness. Everything else is yours: what one thread computes, the
block shape, the grid shape, and the flattening.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

- **TODO 1** — the kernel. The supplied signature is a suggestion; the contract
  is that all 21,626,537 outputs match the reference.
- **TODO 2** — the block `dim3`.
- **TODO 3** — the grid `dim3`.

The harness enforces three requirements beyond correctness: `blockDim` product
≤ 1024, `gridDim.y`/`gridDim.z` ≤ 65535, and **`gridDim.z * blockDim.z > 1`** —
the decomposition must genuinely use the z axis. It reports threads-per-element
(how much of your grid is waste) and achieved bandwidth. Several correct answers
exist and they are not equally fast.

### Exercise 3 — `exercise03.cu` (predict the behavior)

Before running anything, write down — for block (0,0) and warp 1 only — the
number of distinct `threadIdx.y` values, the number of distinct 128-byte segments
written, and the lowest and highest element index written, for a (8,32) block and
for a (32,8) block; plus the per-block segment count for each. Encode your answers
in TODO 1–3. The program then prints the lane-by-lane roster and the truth, and
scores you out of 10.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

`OVERALL: PASS` requires both a correct numerical result *and* ten correct
predictions. Running it first to read the answers off the screen defeats the
exercise entirely; this is the drill that Modules 5 and 8 assume you can do.

---

## Prediction

Commit to these in writing before you compile anything.

1. In Exercise 1, the four block shapes `(32,8)`, `(8,32)`, `(256,1)` and
   `(1,256)` all contain 256 threads and all produce identical output. Rank them
   by expected runtime on the 50 MB image, and state the ratio between fastest
   and slowest. Most people predict the wrong *pair* as being far apart.

2. In Exercise 2, you will choose one of the three problem axes for `threadIdx.x`.
   Predict the slowdown factor if you had chosen the *row* axis instead of the
   column axis, holding block shape and thread count constant.

3. In Exercise 3, config A uses a (8,32) block, so warp 1 spans four rows of the
   matrix. Predict whether warp 1's 32 stores cost *more* 32-byte sectors than
   config B's single-row warp, and say what, if anything, config A does cost.
