# Module 03 — Check Your Understanding (answers)

---

## 1. `dim3 block(16,16)` reading `A[threadIdx.y*16 + threadIdx.x]`, versus `dim3 block(256)` reading `A[threadIdx.x]`

**Warp 2 in the 2-D case.** The linearization rule gives
`tid = threadIdx.x + 16 * threadIdx.y`. Warp 2 is `tid ∈ [64, 96)`. Solving
`16*ty ≤ tid < 16*ty + 16` gives `ty ∈ {4, 5}` and `tx ∈ [0,16)`: two rows of
the 16×16 tile, sixteen lanes each, in lane order
`(0,4)…(15,4)(0,5)…(15,5)`.

The index expression is `ty*16 + tx`, which is *numerically identical to `tid`*.
So warp 2 reads elements **64…95** — 32 consecutive floats, 128 bytes, starting
at byte offset 256. `cudaMalloc` returns at least 256-byte-aligned memory, so
this is a 128-byte-aligned request: **4 sectors of 32 bytes**, 100% utilised.

**Warp 2 in the 1-D case.** `tid = threadIdx.x`, warp 2 is `tid ∈ [64,96)`, and
the index is `threadIdx.x`. Elements **64…95**. Identical: 4 sectors.

**The point.** The two footprints are the same, and that is not a coincidence.
The index expression `ty*blockDim.x + tx` *is* the linearization formula. Any
time your flattening reproduces the hardware's own thread ordering, a 2-D block
behaves exactly like a 1-D block of the same size. A 2-D block only changes the
memory picture when the flattening uses a stride different from `blockDim.x` —
which is precisely the case in a real matrix kernel, where the stride is the
matrix width, not the block width. That mismatch between `blockDim.x` and the
array stride is where all the interesting behaviour lives.

---

## 2. `dim3 block(8,8,16)`, a load whose address depends only on `threadIdx.z`

`tid = tx + 8*(ty + 8*tz) = tx + 8*ty + 64*tz`, with `tx, ty ∈ [0,8)` and
`tz ∈ [0,16)`. The pair `(tx, ty)` contributes `0…63`, so

```
tz = tid / 64        (exactly, with no remainder interaction)
```

A warp is 32 consecutive `tid`. Any 32-wide window `[32w, 32w+32)` lies entirely
within one 64-wide `tz` bucket, because 32 divides 64. Therefore:

- **Warp 0**: `tid ∈ [0,32)` ⇒ `tz == 0` for all 32 lanes ⇒ **1 distinct
  address**.
- **Warp 5**: `tid ∈ [160,192)` ⇒ `tz == 2` for all 32 lanes ⇒ **1 distinct
  address**.

And in fact *every* warp of this block presents a single address, because
`blockDim.x * blockDim.y = 64` is an exact multiple of the warp size.

The hardware handles this as a **broadcast**: the coalescer sees 32 identical
addresses, issues one 32-byte sector request, and returns the value to all lanes.
It is the cheapest possible global load — one sector for 32 lanes — not the most
expensive, which is the intuition to overwrite.

Change `blockDim` to `(8,8,16)` → `(8,4,32)` and the answer changes: now
`tx + 8*ty` spans only 32 values, `tz = tid/32`, and each warp still has one `tz`
— still broadcast. Change it to `(8,3,...)` and `blockDim.x*blockDim.y = 24` does
not divide 32, so warps straddle two `tz` values and present 2 distinct
addresses. The divisibility of `blockDim.x * blockDim.y` by 32 is what decides
it, and you can read that off the formula without running anything.

---

## 3. "The guard is unnecessary because `cudaMalloc` rounds up"

**Reason 1 — correctness, regardless of what the allocator does.** The claim is
about *not faulting*. It is silent about the writes themselves. Those threads
write values computed from garbage inputs into memory the program does not own
logically, and on the read side they load undefined bytes. If the slack happens
to be the beginning of the *next* live allocation — and suballocators routinely
place allocations back to back — the kernel silently corrupts another array. The
failure surfaces later, in a different kernel, as wrong numbers with no obvious
cause. "It did not crash" is not "it was correct".

Worse, the excess is not bounded by anything small in the 2-D case: with a
`(32,8)` block on a 1021-row image, the last block-row runs 3 rows past the end,
each of `w` elements — thousands of elements, not a handful of bytes.

**Reason 2 — what the argument reveals.** It treats `cudaMalloc` as though it
returned a padded buffer with a documented size, the way `cudaMallocPitch` does.
It does not. `cudaMalloc(&p, bytes)` promises exactly `bytes` of valid storage
(with alignment guarantees on the *start* address, not the end). Any rounding is
an internal implementation detail of the suballocator, undocumented, not
guaranteed, and free to change between CUDA versions and between GPUs. Building
correctness on it is building on a private implementation detail — the same class
of mistake as relying on a `malloc` implementation's bookkeeping bytes.

**The tool that settles it.**

```
compute-sanitizer --tool memcheck .\prog.exe
```

reports, per offending access:

```
========= Invalid __global__ write of size 4 bytes
=========     at 0x... in kernel(float*, int)
=========     by thread (5,7,0) in block (22,127,0)
=========     Address 0x7f... is out of bounds
=========     and is 1234 bytes after the nearest allocation at 0x7f... of size 2993572 bytes
```

Note the phrasing: *after the nearest allocation of size N*. The sanitizer tracks
the logical allocation size, not the physical rounding — which is exactly the
distinction the colleague is eliding. If the access lands inside a *different*
live allocation, memcheck will not flag it at all, which is the strongest
argument of all: the bug can be simultaneously real and invisible to the tool.

---

## 4. `size_t n = 5000000000`, `block = 256`

**The grid line is fine.**

```cpp
int grid = (int)((n + block - 1) / block);   // = 19,531,250
```

`n + block - 1` is computed in `size_t` (64-bit) because `n` is `size_t` — the
usual arithmetic conversions promote the `int` operand. No overflow. The quotient
19,531,250 is well under `INT_MAX` and under the `gridDim.x` limit of 2³¹−1, so
the cast is safe.

**The index line is the bug.**

```cpp
unsigned i = blockIdx.x * blockDim.x + threadIdx.x;   // 32-bit unsigned
```

`blockIdx.x` and `blockDim.x` are both `unsigned int`, so the product is computed
in **32-bit unsigned arithmetic and wraps modulo 2³²**, regardless of the type of
`i`. The largest true value is 19,531,249 × 256 + 255 = 4,999,999,999, which
exceeds 2³²−1 = 4,294,967,295. Every thread whose true global id is ≥ 2³²
receives `true_id − 2³²` instead.

**Symptom.** 705,032,704 threads (the excess above 2³²) alias onto indices
0 … 705,032,703. Those elements are written twice — once by the legitimate
thread and once by a high-id thread computing from the wrong input — and the last
705 million elements of the array are **never written at all**. The output is
correct in the middle, garbage at the front, and stale at the back. Because the
race is between two blocks with no ordering, the garbage is nondeterministic run
to run.

**Why memcheck is silent.** The wrapped index is a *small, in-bounds* value. The
guard `if (i < n)` compares the wrapped `i`, sees a small number, and passes. No
access ever leaves the allocation, so there is no invalid read or write to
report. `--tool racecheck` will not see it either: racecheck detects shared-memory
races within a block, not global-memory races across blocks. This is a class of
bug that only a numerical check against a reference finds — which is why every
exercise in this course ships one.

**The fix.** Widen before the multiply:

```cpp
size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
```

Casting after the multiply (`(size_t)(blockIdx.x * blockDim.x)`) does nothing:
the wrap has already happened. For arrays this large the grid-stride form is the
better answer anyway, since it lets you use a grid of a few hundred blocks and
keeps every index comfortably inside 32 bits — though the loop variable itself
must still be 64-bit.
