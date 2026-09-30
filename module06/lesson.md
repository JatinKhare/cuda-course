# Module 06 — Shared Memory

> Prerequisites: Modules 1–5
> What this module gives you: a software-managed on-chip scratchpad that lets a block buy data reuse and decouple the pattern in which data is *read* from the pattern in which it is *laid out* — plus an honest account of when that is worth doing.

---

## Concept

### 1. What shared memory is

**Shared memory** is a per-block region of on-chip SRAM that you allocate, fill,
and index explicitly. Module 4 located it and priced it: it lives in the same
128 KB array as L1, it is reached at roughly L1 latency, and it is an order of
magnitude closer than DRAM's ~575 cycles.

The critical property is not the latency. It is that shared memory is **not a
cache**. A cache is a hardware policy: it decides what to keep, on a
tag-and-eviction mechanism you do not control, and it is indexed by the same
addresses as the memory it fronts. Shared memory is a **scratchpad**: a flat,
separately-addressed array with no tags, no eviction, no backing store, and no
policy. Nothing arrives in it unless a thread puts it there, and nothing leaves
it except when the block ends.

That is a cost — you write the loads yourself — and it is also the entire point.
A cache is *predictive*; a scratchpad is *guaranteed*. If you have staged a tile
in shared memory, every subsequent read of that tile costs a shared-memory
access. Not "probably". Always.

**PORTABLE CUDA CONCEPT.** Every CUDA-capable GPU since 2007 has had shared
memory with block scope. The *sizes* below are Ada-specific.

### 2. Why it exists: two jobs

**Job 1 — buying reuse.** If each input element is read by *K* different threads
of a block, and the block stages the element once, the block's global traffic for
that element drops from *K* reads to 1. Global traffic falls by up to a factor
of *K*. "Up to", because of the halo tax in §5.

**Job 2 — decoupling the access pattern from the layout.** Module 5 established
that coalescing is a property of the 32 addresses one warp presents on one
instruction. Some algorithms want to read data in an order that cannot be
coalesced, and often the *write* side wants a different order again, so no single
thread-to-element mapping makes both sides contiguous. Shared memory dissolves
the conflict: **load in whatever pattern is coalesced, then read in whatever
pattern the algorithm wants.** The uncoalesced access happens against on-chip
SRAM, where "contiguous" carries no privilege.

Hold that second claim loosely for one more module. Shared memory has its own
access-pattern cost — **bank conflicts** — and **Module 7 owns them entirely**.
Until then, assume shared reads are uniform-cost; the kernels in this module are
written so that assumption does not mislead you.

### 3. Arithmetic intensity, quantified — be sceptical

The traffic argument above is real but incomplete, and reading it uncritically is
how people spend a week tiling a kernel for a 0.85× speedup.

Define, per output element:

| quantity | meaning |
|---|---|
| *K* | how many threads read each input element in the untiled kernel |
| *H* | the halo tax: tile cells loaded ÷ outputs computed |
| *K / H* | the best possible reduction in global reads |

For the 5-point stencil of Module 3, *K* ≤ 5, and with a 32×8 tile
*H* = 34·10 / 256 = 1.33, so *K/H* ≈ 3.8. For a 2D box filter of radius 4,
*K* = 81 and *H* = 2.50, so *K/H* ≈ 32. For a GEMM tile, *K* is the tile
dimension and *K/H* runs into the hundreds — which is why Modules 15–17 are
built on this module.

Now the part that the traffic model does not tell you: **the caches got there
first.** Module 3 measured the untiled 5-point stencil at 0.3213 ms on the
4093×3079 image, which is 313.8 GB/s of *useful* traffic (8 B per pixel: one
read, one write) against a 432 GB/s DRAM peak — **72.6 % of peak**. A kernel
moving 72.6 % of peak in useful bytes cannot be re-reading each input 5 times
from DRAM; at most it is re-reading 432/313.8 = 1.38 times. L1 and L2 had already
collapsed *K* = 5 down to ≈ 1.4 before you wrote any shared-memory code.

Two consequences, and they are the core of this module:

1. **A kernel already near a hardware ceiling cannot be sped up by removing
   traffic it is not generating.** 72.6 % of DRAM peak caps any possible
   speedup at 1.38×, and tiling has to pay a halo tax and a barrier out of that
   budget. Exercise 1 measures the tiled 5-point stencil at **0.85×** of the
   untiled version — tiling it is a net loss.
2. **Tiling pays when *K/H* is large AND the untiled kernel is not already
   saturated.** Exercise 2 uses a problem with *K* = 256 where the untiled kernel
   runs at a fraction of any ceiling, and there shared memory wins by ~1.4–1.6×.

Compare that against the naive expectation of 256× and note how much of the gap
is "L1 was already doing this". The honest headline for Ada: **shared memory
usually replaces L1 hits, not DRAM traffic.** What you are buying is the
difference in cost between an `LDG` that hits in L1 and an `LDS` — real, but a
small constant — multiplied by how many of them there are, plus the *guarantee*
that they hit. The transformative wins (Modules 15–17) come from combining shared
memory with register blocking so that the instruction count falls too.

### 4. Scope and lifetime

Shared memory is **per block**. It is allocated when the block is placed on an SM
and freed when the block retires. It is not addressable from another block, not
addressable from the host, and its contents at block start are undefined. There
is no way to keep anything in it across a kernel launch.

This follows directly from Module 1: a block is dispatched whole to one SM and
**never migrates**. If a block could be suspended on SM 3 and resumed on SM 17,
its scratchpad would have to move with it, and a scratchpad you can move is
either a cache (with coherence traffic) or an enormous copy. Indivisible,
non-migrating blocks are the precondition that makes a per-block scratchpad
implementable at all. The programming model's least convenient rule and its most
useful memory space are the same design decision.

Corollary: **`__shared__` declares one array per resident block, not one array.**
Example 1 §A demonstrates eight blocks writing "the same" variable and each
reading back its own value.

### 5. Cooperative loading and the halo

The load is written by the block as a whole.

A block computing a TW × TH patch of a stencil output needs a
(TW+2R) × (TH+2R) patch of input: the interior plus a **halo** (also called
**ghost cells**) of width *R* on every side. The halo is why the load is
interesting:

- there are (TW+2R)(TH+2R) cells to load and only TW·TH threads;
- those two numbers are not equal, and the first is larger;
- therefore **the load cannot be one cell per thread**.

The standard shape is a flat strided loop over the tile:

```cpp
for (int idx = tid; idx < SW * SH; idx += nthr) {
    int ly = idx / SW, lx = idx - ly * SW;
    tile[idx] = in[ clamp(row0 + ly - R) * w + clamp(col0 + lx - R) ];
}
```

Note the ordering: consecutive `idx` are consecutive tile cells and, within a
row, consecutive global columns — so consecutive lanes of a warp read consecutive
addresses and the load is coalesced. Loop the other way (each thread taking a
contiguous *chunk* of `idx`) and every lane reads a different 32 B sector: same
instruction count, up to 32× the bytes. Module 5's counting procedure applies
unchanged to a cooperative load.

**The classic bug of this module** is conflating the two index mappings that live
in a tiled kernel:

| mapping | domain | used for |
|---|---|---|
| load mapping | tile cells, (TW+2R)(TH+2R) of them | which global element each thread fetches |
| compute mapping | output pixels, TW·TH of them | which output each thread writes |

Writing `tile[ty][tx] = in[...]` — one cell per thread — is the compute mapping
wearing the load mapping's clothes. It compiles, runs, never faults, and leaves
the halo holding whatever the previous block left in that SRAM.

### 6. Barriers

`__syncthreads()` is a **barrier** across all threads of a block: no thread
proceeds past it until every thread of the block has reached it, and shared-memory
writes issued before it are visible to all threads of the block after it.

**Module 9 makes this precise** — the memory-ordering half of that sentence has
real content, and barriers in divergent control flow have rules. For this module
and the next, two rules of thumb suffice:

1. **After writing a tile, before reading it.** Otherwise a thread may read a
   cell its writer has not written yet.
2. **After reading a tile, before overwriting it.** Otherwise a fast warp's next
   write may land before a slow warp's current read. This one is easy to forget
   because there is no *new* data involved — it is a write-after-read hazard, and
   it only exists when a block reuses the same buffer across loop iterations.

Both hazards are invisible with 32-thread blocks, because the 32 threads of a
single warp issue in lockstep from one scheduler. They appear as soon as a block
spans more than one warp. Exercise 3 is built on exactly that.

### 7. Static and dynamic allocation

**Static.** The size is a compile-time constant:

```cpp
__shared__ float tile[16][16];      // 1024 B, baked into the SASS
```

`ptxas` reports it (`-Xptxas -v` → `... bytes smem`) and
`cudaFuncGetAttributes().sharedSizeBytes` returns it.

**Dynamic.** The size is fixed at launch, by the **third launch parameter** —
the one Module 2 introduced and deferred:

```cpp
extern __shared__ float s[];        // unsized, one per kernel
...
kernel<<<grid, block, sharedBytes, stream>>>(args);
```

`sharedBytes` is a **byte count**, not an element count, and it names the size of
the *whole* `extern __shared__` region. There is exactly one such region per
kernel; every `extern __shared__` declaration in a kernel aliases the same base
address. `cudaFuncGetAttributes` does **not** see it — it reports only the static
part — because it is not a property of the function.

Use dynamic when the size depends on a runtime parameter (a tile width chosen by
the host, a per-launch problem size). Use static otherwise: a compile-time size
lets the compiler fold the shared base offsets into the `LDS`/`STS` addressing
and lets the occupancy calculator see the cost without being told.

### 8. Carving one dynamic allocation into several arrays

One `extern __shared__` region, several logical arrays, and no help from the type
system:

```cpp
extern __shared__ char smem[];
float*  prof = (float*) smem;                    // 33 floats  -> 132 B
float2* pos  = (float2*)(smem + 132);            // <-- BUG
```

`132 % 8 != 0`. `float2` is `__align__(8)`, so the compiler emits a 64-bit
`STS.64` / `LDS.64`, and that instruction requires an 8-byte-aligned address. The
kernel dies with **`cudaErrorMisalignedAddress`** — the same error Module 5
produced by casting a mid-array `float*` to `float4*`, for the same reason, one
memory space over. Do it right:

```cpp
__host__ __device__ size_t align_up(size_t o, size_t a) { return (o + a - 1) & ~(a - 1); }

size_t off_pos = align_up(33 * sizeof(float), alignof(float2));   // 136
size_t off_wgt = off_pos + TILE * sizeof(float2);
size_t total   = off_wgt + TILE * sizeof(float);                  // the 3rd launch arg
```

Two hazards for the price of one:

- the **alignment** hazard above, which is loud;
- the **byte count** hazard, which is silent. The naive carve above asks for
  `132 + 8·TILE + 4·TILE` bytes — *four bytes less* than the aligned layout
  needs. Get the alignment right in the kernel and the count wrong at the launch
  and the last array runs off the end of the block's allocation.

  Measured on this GPU: a launch that under-requests by a full **kilobyte** still
  produces correct results and is **not** reported by
  `compute-sanitizer --tool memcheck`. Shared memory is granted to a block in a
  granularity coarser than the request, and the window the hardware bounds-checks
  is the granted size. So the under-request is invisible right up until it is
  catastrophic, at which point it silently overwrites another block's scratchpad.
  Contrast that with the alignment bug, which faults instantly and cannot produce
  a wrong answer. **The loud bug is the safe one.**

The rule: compute the offsets in exactly one place and use that function on both
the host and the device.

### 9. Capacity, and how it costs you occupancy

On this GPU (Ada, sm_89):

| limit | value |
|---|---|
| unified L1 + shared array per SM | 128 KB |
| addressable as shared per SM | **102400 B (100 KB)** |
| per block, default maximum | **49152 B (48 KB)** |
| per block, opt-in maximum | **101376 B (99 KB)** |
| driver-reserved per block | **1024 B** |

A launch requesting more than 48 KB of dynamic shared memory fails with
`cudaErrorInvalidValue` until the kernel opts in:

```cpp
cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
```

This is a *per-kernel*, not per-launch, attribute, and it applies only to the
dynamic portion. The 99 KB ceiling is below the 100 KB per-SM figure because the
SM keeps a slice of the array for L1 and cannot hand a single block the lot.

Shared memory per block caps blocks per SM:

```
blocks_per_SM = min( 1536 / threads_per_block ,
                     102400 / (bytes_per_block + 1024) ,
                     24 )
```

That `+1024` is the driver reservation, and it is why 16384 B/block yields 5
blocks per SM and not the 6 that `102400/16384 = 6.25` suggests. Measured on this
GPU with 256-thread blocks:

| bytes/block | charged | blocks/SM | occupancy | limiter |
|---|---|---|---|---|
| 0 | 1024 | 6 | 100.0 % | threads/SM |
| 8192 | 9216 | 6 | 100.0 % | threads/SM |
| 12288 | 13312 | 6 | 100.0 % | threads/SM |
| 16384 | 17408 | 5 | 83.3 % | shared memory |
| 25600 | 26624 | 3 | 50.0 % | shared memory |
| 49152 | 50176 | 2 | 33.3 % | shared memory |

Do not guess this — `cudaOccupancyMaxActiveBlocksPerMultiprocessor` computes it
including every limiter. Example 1 §D prints the table above.

The trade is real: shared memory buys reuse and spends the resident-warp count
that Module 1 showed is how the SM hides latency. **Module 19** gives the full
occupancy story, including why maximum occupancy is not the same as maximum
speed.

---

## Hardware Mental Model

**Where the bytes are.** Each SM on Ada has one 128 KB SRAM array serving as L1
data cache and shared memory. The split is configurable (Module 4's carveout
hint), with at most 100 KB going to shared. Shared memory and L1 are therefore
*competing for the same transistors*: a kernel using 100 KB/SM of shared memory
has left almost nothing for L1, which is one more reason the L1-vs-scratchpad
comparison in §3 is not one-sided.

**Why there are no tags.** A cache line needs a tag, a valid bit, a replacement
state, and comparators on every access, because the address it holds is not known
until it arrives. A shared-memory address is *known by construction*: the
compiler emits `LDS`/`STS` against a window whose base the SM assigns at block
launch. Address arrives → decode → SRAM read. No tag compare, no miss path, no
fill request, no allocation policy. That is where the latency advantage over an
L1 hit comes from, and it is also why shared-memory latency is *deterministic*
while L1 latency is a distribution.

**A separate address space, in the ISA.** Look at SASS and you will see `LDG.E` /
`STG.E` for global and `LDS` / `STS` for shared. They are different instructions
against different windows. A shared-memory pointer is not a global pointer with a
flag; the generic-addressing machinery that lets you pass `float*` around in
device code resolves to one or the other at the instruction level.

**Allocation at block launch.** When the GigaThread engine places a block on an
SM (Module 1's step 5), it must reserve, atomically: thread slots, warp slots,
registers, **and** shared memory. If any one is unavailable the block does not
launch. This is why shared memory is an occupancy term and not merely a
performance term — it is checked by the same placement gate as registers, before
the block exists.

**Why it can exist at all.** See §4. The scratchpad is affordable because a block
never migrates and never partially retires. Everything about shared memory's
scope, lifetime, and inaccessibility from other blocks is downstream of that one
scheduling guarantee.

**What a barrier costs.** `__syncthreads()` compiles to a `BAR.SYNC`. The SM
tracks per-block arrival counts in hardware; warps that arrive early are marked
stalled and the scheduler runs other warps. The cost is therefore not the barrier
instruction but the **convoying**: the block runs at the speed of its slowest warp
at each barrier, and any warp-level slack that was previously absorbable is now
exposed. In a tiled kernel with a barrier per tile, a block with 8 warps
serializes 8 warps' worth of tail latency at every tile boundary. This is part of
why Exercise 1's tiled version loses.

**The thing this module will not teach you.** Shared memory is built from **32
banks of 4-byte words**, and the cost of a shared access depends on how a warp's
32 addresses distribute across those banks. That is a replay mechanism exactly
like the constant-memory serialization Module 4 measured. **Module 7 owns it.**
Every shared-memory access in this module is either warp-uniform (a broadcast,
which is free) or unit-stride across lanes (one word per bank, also free), so
banks never enter the measurements here. Do not generalize the numbers in this
module to a kernel whose shared access pattern is strided.

---

## Code Walkthrough

### `example01.cu` — mechanics

**§A, scope.** Eight blocks, each writing `blockIdx.x + 1` into every slot of a
128-entry shared array, then reading slot `(tid+64) % 128` — a slot written by a
*different thread* of the same block:

```cpp
__shared__ int tag[128];
tag[threadIdx.x] = (int)blockIdx.x + 1;
__syncthreads();
int seen = tag[(threadIdx.x + 64) % 128];
```

Every block reads back its own value. One declaration; eight simultaneous
physical arrays. The barrier is load-bearing: block 5's thread 0 reads a cell
written by block 5's thread 64, which is in a different warp.

**§B, static vs dynamic.** The same computation twice:

```cpp
__shared__ float s[256];                        // static
extern __shared__ float sdyn[];                 // dynamic
...
reverse_static <<<grid, 256>>>(...);
reverse_dynamic<<<grid, 256, 256*sizeof(float)>>>(...);
```

and the reported attributes:

```
cudaFuncGetAttributes(reverse_static ).sharedSizeBytes = 1024
cudaFuncGetAttributes(reverse_dynamic).sharedSizeBytes = 0
```

The dynamic size is invisible to `cudaFuncGetAttributes` because it is a property
of the *launch*, not of the function. Anything that needs to know — the occupancy
API, for instance — must be told separately.

**§C, carving.** Three arrays (`int[33]`, `float2[64]`, `float[64]`) out of one
blob. The program prints:

```
  int idx[33]    : offset    0, 132 B
  float2 pos[64] : offset  136 (naive would be 132), 512 B, needs 8 B alignment
  float wgt[64]  : offset  648, 256 B
  total request  : 904 B  (naive carve would ask for 900 B)
```

`.\example01.exe --align-bug` takes the naive offset and shows what the hardware
does about it:

```
  launch  : cudaSuccess
  execute : cudaErrorMisalignedAddress  <- the 8-byte store to pos[k] faulted
```

Note which check caught it. The launch configuration was fine; the fault happened
during execution, so only the synchronizing call saw it — exactly Module 2's
two-class error model. And it is **sticky**: the context is dead, every later API
call fails.

**§D, occupancy.** Reproduced as the table in §9 above. The `charged` column is
the point.

**§E, the ceiling.** A 64 KB dynamic request, before and after opting in:

```
  request 65536 B without opting in : cudaErrorInvalidValue
  after cudaFuncSetAttribute(..MaxDynamicSharedMemorySize, 65536) : PASS
  blocks/SM at 64 KB/block : 1
```

64 KB per block means one block per SM: 256 threads out of 1536, 16.7 %
occupancy. The opt-in is available; it is rarely free.

### `example02.cu` — the traffic ledger

A 2D box filter of radius *R* ∈ {1,2,3,4} over the 4093×3079 image, untiled and
tiled, timed in one sweep. The tiled kernel is the canonical shape:

```cpp
__shared__ float tile[(TH + 2*R) * (TW + 2*R)];
...
for (int idx = tid; idx < SW * SH; idx += nthr) {      // LOAD mapping
    int ly = idx / SW, lx = idx - ly * SW;
    tile[idx] = in[ clampi(row0+ly-R,0,h-1) * w + clampi(col0+lx-R,0,w-1) ];
}
__syncthreads();
const int lx = threadIdx.x + R, ly = threadIdx.y + R;  // COMPUTE mapping
for (dy) for (dx) acc += tile[(ly+dy)*SW + (lx+dx)];
```

One detail that is not about shared memory but ruins the measurement if you miss
it: the untiled kernel hoists its column clamps out of the inner loop.

```cpp
int cc[2*R+1];
for (int dx = -R; dx <= R; ++dx) cc[dx+R] = clampi(col+dx, 0, w-1);
```

With `R` a template parameter both loops unroll fully, so `cc[]` stays in
registers (Module 4: a dynamically indexed local array would be DRAM). Without
this hoist the "memory" experiment is really an integer-min/max experiment —
measured 4× slower and the tiling comparison becomes meaningless.

Measured on this GPU (min of 4 sweeps, after a 400 ms clock warm-up):

| R | K = (2R+1)² | halo tax H | ideal K/H | naive ms | tiled ms | measured |
|---|---|---|---|---|---|---|
| 1 | 9 | 1.33 | 6.8× | 0.3317 | 0.3834 | **0.87×** |
| 2 | 25 | 1.69 | 14.8× | 0.4346 | 0.5110 | **0.85×** |
| 3 | 49 | 2.08 | 23.6× | 0.8398 | 0.9150 | **0.92×** |
| 4 | 81 | 2.50 | 32.4× | 1.4493 | 1.3629 | **1.06×** |

Absolute times move ±10 % run to run on this laptop part; the ratios reproduce to
about ±0.05. Two readings:

- The *ideal* column is a fantasy for small *R*, because the naive R=1 kernel is
  already at 70 % of DRAM peak and therefore cannot be re-reading each pixel 9
  times from DRAM. Its true re-read factor is at most 432/304 ≈ 1.4. The caches
  delivered ~99 % of the reduction tiling promised, for free.
- The trend is the real result. Tiling goes from a 13 % loss to a 6 % win as *K/H*
  climbs from 6.8 to 32.4, because the naive kernel falls from 70 % to 16 % of
  DRAM peak and stops being bandwidth-bound. Extrapolate the trend, not the
  ratio: that line, continued, is Modules 15–17.

---

## Check Your Understanding

1. A block is 256 threads and stages a 32×8 tile with a 1-cell halo, i.e. 340
   floats. A colleague proposes "just give each of the 256 threads two cells and
   drop the loop; 340 < 512 so it fits." Their version passes every test on a
   1024×1024 image and fails on a 1021×733 one. Give a concrete (tile, thread)
   pair for which their scheme is wrong, and say what the failure looks like in
   the output image. Then explain why the square power-of-two image hid it.

2. You tile a kernel and it gets 15 % *slower*, with identical results. Your
   profiler says DRAM traffic dropped by 40 %. Both statements are true. Give two
   distinct mechanisms by which removing 40 % of the DRAM traffic can make a
   kernel slower on this GPU, and say what measurement would distinguish them.

3. A kernel declares `extern __shared__ char smem[]` and carves out
   `float a[N]` at offset 0 and `double b[M]` immediately after. It is launched
   with `sharedBytes = N*4 + M*8`. For which values of `N` does this kernel run
   correctly, for which does it fault, and for which does it silently corrupt
   memory? Note that all three outcomes are possible and that *exactly one* of
   them is the dangerous one.

4. Shared memory is per block and dies with the block. Suppose NVIDIA offered a
   per-*grid* scratchpad with the same latency, same capacity per SM, visible to
   every block. Name the specific hardware guarantee from Module 1 that such a
   feature would break, and explain why the resulting thing would necessarily
   behave like a cache rather than like a scratchpad.

Answers: `solutions/module06/check_your_understanding.md`.

---

## Exercises

### Exercise 1 — `exercise01.cu` — tile the stencil you already wrote

Rewrite Module 3 / Exercise 1's 5-point clamped stencil as a tiled kernel, on the
same two images (1021×733 and 4093×3079) so the numbers are directly comparable.
The harness times your tiled kernel at three tile shapes against an untiled
control in the same sweep, and prints the Module 3 baselines (0.0107 ms and
0.3213 ms) alongside.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

| TODO | requirement |
|---|---|
| 1 | The cooperative load, halo included. (TW+2)(TH+2) cells, TW·TH threads. Must be correct for the three tile shapes the harness uses, must clamp out-of-image cells the way the formula does, and must stay coalesced. |
| 2 | Make it safe for a thread to read cells other threads wrote. State in a comment which threads wrote the cells thread (0,0) reads. |
| 3 | Compute the stencil reading only from shared memory. Your position within the tile is neither your position in the image nor `threadIdx`. |
| 4 | **Before building anything**, commit to what tiling will do to the 4093×3079 runtime: big win / modest win / wash / loss. |

Validation: all four configurations must match a CPU reference to
`1e-5 * max(1,|ref|)` on both images, **and** your TODO 4 prediction must match
the measured bucket. `OVERALL: PASS` requires both.

### Exercise 2 — `exercise02.cu` — one blob, three arrays

Radial-basis scatter interpolation: every query point sums a table-driven radial
profile over every source point. Reuse *K* = 256 (each staged source is consumed
by all 256 threads of the block), and the profile-table index is lane-varying —
Module 4's worst case for constant memory. Stage both in dynamic shared memory.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

| TODO | requirement |
|---|---|
| 1 | Compute the byte count for the documented layout (`float[33]`, `float2[256]`, `float[256]`, in that order) and pass it as the third launch parameter. |
| 2 | Carve the three typed pointers out of the one `extern __shared__` blob, in agreement with TODO 1. |
| 3 | Load the profile table (once, not per tile), stream the source tiles, handle the partial last tile (MS = 4093 is not a multiple of 256), and get the ordering right in both directions. |

Validation: both kernels must match a double-precision CPU reference on a strided
sample of the queries, to `1e-4 * max(1,|ref|)`. The harness reports the
tiled-vs-naive ratio.

### Exercise 3 — `exercise03.cu` — debugging: "it works on 32 threads"

A block-local mirror-and-average kernel. Ships broken. The header states the
symptom only.

```
nvcc -arch=sm_89 -O3 -lineinfo -o exercise03.exe exercise03.cu
.\exercise03.exe
.\exercise03.exe --tiles 8
compute-sanitizer --tool racecheck .\exercise03.exe --small
```

| TODO | requirement |
|---|---|
| 1 | Diagnose: set `DIAGNOSIS` to one of five codes. Scored. |
| 2 | Fix the single-window failure. |
| 3 | Fix the multi-window failure, which is a *different* instance of the same class and survives the TODO 2 fix. |

Validation: eight repeats per block size at 32/64/128/256/1024 threads, single-
and multi-window, plus the diagnosis code. `OVERALL: PASS` requires all of it.

---

## Prediction

Commit to these in writing before you build anything.

1. **Exercise 1.** The untiled 5-point stencil runs at 72.6 % of DRAM peak. You
   are about to remove up to 3.8× of its global reads. Write down the tiled
   runtime you expect on the 4093×3079 image, as a multiple of 0.3213 ms, and the
   single sentence of reasoning behind it. Then write down what upper bound
   "72.6 % of peak" places on any speedup at all.

2. **Exercise 2.** Reuse there is *K* = 256, 50× the stencil's. Predict the
   tiled-vs-naive ratio to within a factor of 2, and state which resource you
   expect the tiled version to be limited by afterwards.

3. **Exercise 3.** Before running it: the kernel passes with 32-thread blocks and
   fails with 64. Predict whether `compute-sanitizer --tool racecheck` will report
   a hazard in the **32-thread** case — the one that produces correct answers on
   every run. Justify your answer from what a barrier is for, not from what the
   hardware happens to do.
