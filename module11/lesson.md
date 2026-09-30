# Module 11 — Vector Operations

> Prerequisites: Module 1 (Little's Law, latency hiding), Module 3 (grid-stride
> loops), Module 4 (the memory hierarchy, DRAM latency), Module 5 (coalescing,
> sectors, `float4`, write-allocate)
> What this module gives you: the ability to compute a kernel's runtime *floor*
> before writing it, and to tell — from measurement — whether a kernel has
> reached that floor or still has a defect.

This module opens Part IV. Everything in Parts IV and V is an algorithm:
reduction, scan, histogram, transpose, GEMM. Elementwise vector operations are
the degenerate case of all of them — no reuse, no communication, no reordering
— and that is exactly why they are the right place to learn the discipline that
the rest of the course runs on.

You have already written SAXPY (Module 3) and you already know how to count
sectors (Module 5). This module is not about writing `c[i] = a[i] + b[i]`. It
is about the question that comes *before* you write it: **how fast could this
possibly go, and how will I know when I get there?**

---

## Concept

### The compulsory-traffic floor

**PORTABLE CUDA CONCEPT.** For any kernel, the **compulsory traffic** is the
number of bytes that must cross the DRAM interface at least once, assuming a
perfect implementation:

- every distinct array element that is *read*, counted **once**, regardless of
  how many times the source names it;
- every distinct array element that is *written*, counted once;
- an array that is both read and written counts **twice**.

Divide that by the machine's streaming bandwidth and you have a lower bound on
runtime. On this GPU the nominal peak is 432.0 GB/s (192-bit GDDR6 at 9.001 GHz)
and the realistically achievable streaming rate measured in Module 5 is about
**87% of that, ~376 GB/s**. Use the measured number, not the nominal one.

```
floor_seconds = compulsory_bytes / achievable_bytes_per_second
```

This takes ten seconds and it is the single highest-value habit in GPU
programming. It tells you three things at once:

1. **Whether the problem is worth optimizing at all.** If your kernel already
   runs at 1.05× its floor, there is nothing left; stop.
2. **What the optimization target is.** Not "make it faster" but "get from
   4.2× floor to 1.1× floor".
3. **Whether you counted right.** If a measurement comes out *below* the floor,
   your traffic model is wrong (usually: you forgot something is cached, or
   the buffer fits in L2 — Module 4's 48 MB trap).

**Forward reference.** Module 21 generalises this into the **roofline model**,
where the floor above is the memory-bound side and a second ceiling — peak
FLOP/s — is the compute-bound side. Everything in this module lives on the
horizontal part of that roof, except the last section, which finds the corner.

### Worked traffic counts

| Kernel | Source | Compulsory | Trap |
|---|---|---|---|
| fill | `y[i] = c` | **1N** | none |
| copy | `y[i] = x[i]` | **2N** | none |
| scale | `y[i] = a*x[i]` | **2N** | none |
| saxpy, out of place | `d[i] = a*x[i] + y[i]` | **3N** | none |
| saxpy, in place | `y[i] += a*x[i]` | **3N** | **reading `y` is invisible in the source** |
| polynomial | `y[i] = a*x[i]*x[i] + b*x[i] + c` | **2N** | `x` appears three times, is read once |
| 3-term blend | `d[i] = A*x[i]+B*y[i]+C*z[i]` | **4N** | none |

(`N` here means `N` floats = `4N` bytes.)

The in-place row is the one people get wrong, and getting it wrong is
expensive in both directions. `y[i] += a*x[i]` contains one load and one store
in the source. Count 2N and you will compute an achieved bandwidth about two
thirds of the truth — on this GPU, 251 GB/s instead of 377 — conclude the
kernel is at 58% of peak, and spend a day optimizing a kernel that was already
finished. The read-modify-write really does move three arrays' worth of bytes:
the old `y` has to be fetched before it can be added to.

### Write-allocate: when a store generates a read

Module 5 stated the rule; here is the measurement. A store reaches DRAM as a
32-byte sector. If a warp's stores cover a sector **completely**, the memory
system can write the sector without knowing its previous contents. If they
cover it **partially**, the bytes nobody wrote must be preserved, so the sector
must be read, merged, and written back. That is a **write-allocate**, and it
turns one unit of store traffic into two units of DRAM traffic.

Two kernels that write exactly the same number of useful bytes:

```cpp
__global__ void k_write_dense(float* o, long long nHalf, float v)
{ for (GRID_STRIDE(nHalf)) o[i]   = v; }   // sectors fully covered

__global__ void k_write_odd(float* o, long long nHalf, float v)
{ for (GRID_STRIDE(nHalf)) o[2*i] = v; }   // half of each sector written
```

Measured (`example01.cu`, 128 MB of useful stores each):

| | ms | useful GB/s | implied DRAM GB/s |
|---|---|---|---|
| dense | 0.3690 | 363.7 | 364 |
| strided | 1.4571 | 92.1 | **184 × 2 = 368** (at a 4× traffic model) |

**ratio 3.95×.** The strided kernel executes the same number of store
instructions and writes the same number of useful bytes; it touches twice as
many sectors, and each of those costs a read *and* a write. `0.5N` of useful
stores becomes `1N` read + `1N` written: exactly 4× the traffic, and the
implied DRAM rate lands on the streaming ceiling, which is how you know the
model is right and the bus is not idle.

The converse matters just as much: **there is no unconditional write-allocate
on this GPU.** A `fill` kernel that writes every element measures 375.0 GB/s
against a `1N` model. If every store had forced a line fill it could not have
exceeded half the streaming rate. Write-allocate is a consequence of *partial
sector coverage*, not of writing.

### Non-temporal / streaming stores do not repeal it

CUDA exposes cache-residency hints on loads and stores:

| Intrinsic | Meaning | SASS on sm_89 |
|---|---|---|
| `__stcs(ptr, v)` | "streaming": do not keep this line in cache | `STG.E.EF` (evict-first) |
| `__stwt(ptr, v)` | "write-through" | `STG.E.STRONG.SYS` |
| `__ldcs(ptr)` | streaming load | `LDG.E.EF` |
| `__ldg(ptr)` | read-only path (Module 4) | `LDG.E.CONSTANT` |

Folklore says streaming stores avoid the read-for-ownership. Measured on this
GPU (`example01.cu`, Part C):

| store form | ms | GB/s (1N model) |
|---|---|---|
| plain `STG`, full sector | 0.7159 | 375.0 |
| `__stcs`, full sector | 0.7426 | 361.5 |
| `__stwt`, full sector | 0.7273 | 369.1 |
| plain `STG`, half sector | 1.4571 | 92.1 |
| `__stcs`, half sector | 1.4446 | 92.9 |

**They change the opcode and nothing else measurable.** That is not a
contradiction once you see what they are: hints about *cache residency*, i.e.
about whether L2 should retain the line for a future access. The partial-sector
fill is not a cache policy — it is a correctness requirement, because the bytes
you did not write have to survive. No hint can repeal it. The fix for the
strided kernel is to write whole sectors.

(`__stcs` is still useful: it stops a large streaming output from evicting a
smaller array you want to keep in L2. That is a *different* kernel's problem,
and Module 23 has the metric for it.)

### Grid-stride loops at the ceiling

Module 3 established the grid-stride loop as the default mapping and measured
SAXPY at 348.5 GB/s with a wave-sized grid against 297 GB/s for a 1:1 mapping.
The question Module 3 deferred is: *what grid size?*

`example02.cu` sweeps it. Grid-stride copy, 256 MB in and 256 MB out:

```
GB/s, rows = blocks per SM, columns = threads per block
blocks/SM       64      128      256      512     1024   threads/SM
        1    178.1    287.7    369.6    379.5    379.2      64- 1024
        2    295.5    360.6    379.0    378.8    367.1     128- 2048
        4    362.7    378.6    378.2    361.0    375.7     256- 4096
        8    378.9    377.0    355.2    366.9    379.2     512- 8192
       16    378.0    336.3    363.2    374.0    381.8    1024-16384
       32    339.0    355.8    374.1    376.2    384.3    2048-32768

1:1 mapping, grid = ceil(N/256) = 262144 blocks : 395.2 GB/s
```

Read this table by **total resident threads**, not by either axis. The only
clearly-slow region is the top-left corner. Everything from roughly 4,000
threads upward sits in a 340–385 GB/s band whose internal variation is run-to-run
noise, not structure. There is no "optimal" block size here; there is a
threshold, and above it the bus is the constraint and it does not care who is
asking.

The threshold comes from **Little's Law** (Module 1): to sustain throughput `B`
across a pipeline of latency `L`, you need `B × L` bytes in flight. At 384 GB/s
and a ~575-cycle DRAM latency at ~1.9 GHz, that is

```
384e9 B/s × (575 / 1.9e9) s ≈ 116 kB in flight
```

At 4 bytes per thread per outstanding load, that is ~29,000 outstanding loads.
A grid of 5,120 threads reaches it only if each thread has ~6 loads
outstanding; a grid of 40,960 threads needs less than one each. Both are ways
of buying the same thing. That is the whole of the next two sections.

**A documented surprise, and a correction to a rule you may have taken from
Module 3.** The 1:1 mapping is not merely competitive here — at **395.2 GB/s**
it is the *fastest* configuration in the whole experiment, ahead of the best
grid-stride entry (384.3). Module 3 measured the opposite for SAXPY: 348.5 GB/s
grid-stride against 297 for 1:1. Both measurements are right, and the difference
is the kernel. Module 3's SAXPY does a bounds test and a fresh 64-bit index
computation for every element it touches, with no loop over which to amortise
them; this copy has a body of one load and one store, so the per-element
overhead the grid-stride loop saves is nearly zero, while the 262,144-block
launch gets something in return: the GigaThread engine retires and dispatches
blocks continuously, which smooths the tail and keeps the memory pipeline fed
without any loop-carried address arithmetic.

Do not memorise "grid-stride is faster". The grid-stride loop's guarantees are
**correctness under any launch shape** and **a machine-sized grid**, which is
why every kernel in this module uses one and why Exercise 1 validates at
`<<<1,32>>>`. Speed is not on that list, and on this GPU, for a kernel this
simple, it is not even true.

### Vectorized access at scale

Module 5 proved that a `float4` load moves exactly the same sectors as four
`float` loads. The sector count per useful byte is unchanged, so **vectorizing
cannot raise the bandwidth ceiling.** What it changes is the instruction count
and the number of outstanding-request slots each instruction occupies.

Real SASS from `example02.cu`, one grid-stride iteration of each kernel:

```
k_saxpy   10 x LDG.E.CONSTANT       5 x STG.E        ( 5 elements)
k_saxpy2  10 x LDG.E.64.CONSTANT    5 x STG.E.64     (10 elements)
k_saxpy4  10 x LDG.E.128.CONSTANT   5 x STG.E.128    (20 elements)
```

Identical instruction counts; four times the elements. Per element the scalar
kernel issues 0.75 memory instructions and the `float4` kernel issues 0.1875.
Each 128-bit request also occupies **one** entry in the SM's outstanding-request
tracking structure while returning four times the data, so the same number of
in-flight requests covers four times as much latency.

Measured, saxpy `o = a*x + y`, 3N = 768 MB:

| width | 1 block/SM (5,120 thr) | vs scalar | 8 blocks/SM (40,960 thr) | vs scalar |
|---|---|---|---|---|
| `float` | 349.7 GB/s | 1.000 | 386.4 GB/s | 1.000 |
| `float2` | 384.8 GB/s | **1.100** | 381.4 GB/s | 0.987 |
| `float4` | 389.5 GB/s | **1.114** | 386.6 GB/s | 1.000 |

**Vectorizing pays where request slots are scarce and is nearly free where the
bus is already saturated.** It is a latency/issue optimization, not a bandwidth
one. The practical consequence: reach for it when you are constrained in how
many threads you can run — a small problem, a co-resident kernel, a register-
hungry body — and do not expect it to rescue a kernel that is already at the
ceiling.

**Alignment and tails, now at a size where you must handle them.** Module 5's
three rules still apply and are not negotiable:

1. the pointer must be 16 B aligned — `cudaMalloc` gives ≥ 256 B, but
   `base + 1` does not;
2. `nVec = N / 4` with **truncating** division; `(N+3)/4` reads past the end;
3. the `N % 4` leftovers must be processed **exactly once**, in the same
   launch, from a branch at kernel scope with its own range test.

Exercise 1 uses `N = 40,000,001` and an operation that is *not idempotent*, so
an element processed twice is caught and reported.

### Thread coarsening: memory-level parallelism per thread

**PORTABLE CUDA CONCEPT.** **Memory-level parallelism (MLP)** is the number of
memory requests a single thread has outstanding at once. A thread that issues a
load and immediately uses the result has MLP 1 and stalls for the full DRAM
latency. A thread that issues eight independent loads and then uses them has MLP
8 and stalls once for eight loads' worth of data.

This is **ILP applied to memory**, and it is a *substitute for occupancy*. The
memory system counts requests, not threads. 40 warps each with 1 request
outstanding and 5 warps each with 8 are the same load on the bus.

Coarsening — one thread handling `C` elements — only delivers MLP if the loads
are **hoisted above the first use**:

```cpp
// MLP = 2C : all 2C loads issue before any value is consumed
float xv[C], yv[C];
#pragma unroll
for (int c = 0; c < C; ++c) { xv[c] = x[j]; yv[c] = y[j]; }
#pragma unroll
for (int c = 0; c < C; ++c) { o[j] = a*xv[c] + yv[c]; }

// MLP = 2 : `#pragma unroll 1` forbids overlapping iteration c with c+1
#pragma unroll 1
for (int c = 0; c < C; ++c) o[j] = a*x[j] + y[j];
```

Same traffic, same arithmetic, same `C`. Measured (`example02.cu`, Part C,
saxpy 3N, 128 threads/block):

| GB/s | C=1 | C=2 | C=4 | C=8 | C=16 |
|---|---|---|---|---|---|
| hoisted, 1 block/SM (5,120 threads) | 351.4 | 265.4 | 351.6 | 354.6 | 372.6 |
| serial, 1 block/SM | 343.4 | 153.8 | 169.0 | 169.4 | 169.8 |
| hoisted, 8 blocks/SM (40,960 threads) | 384.1 | 388.8 | 382.5 | 373.6 | 370.1 |
| serial, 8 blocks/SM | 387.3 | 387.7 | 388.1 | 387.9 | 388.0 |

Three readings:

1. **At 1 block/SM the two rows separate by about 2× for every `C ≥ 2`.** The
   SASS confirms the mechanism exactly: for `C ≥ 2` the serial kernel compiles
   to `2 × LDG.E.CONSTANT` and `1 × STG.E`, full stop, while the hoisted kernel
   compiles to 16–32 `LDG`s. Requests in flight is the only currency the memory
   system accepts.
2. **At 8 blocks/SM the rows converge and the simpler kernel is marginally
   ahead.** Occupancy already supplied the concurrency; coarsening now buys
   nothing and costs registers (`C=16` uses 56 registers against `C=1`'s 34),
   address arithmetic and a longer tail. **Coarsening a kernel that is already
   at its ceiling makes it longer, not faster.**
3. **The `C = 2` hoisted entry is reproducibly below its `C=1` and `C=4`
   neighbours, and the simple MLP story does not explain it** — the SASS shows
   `k_mlp<1>` issuing 10 `LDG`s and `k_mlp<2>` issuing 20, so `C=2` has *more*
   requests in flight and is still slower. It is reported rather than smoothed
   away. Build arguments on the shape of a curve, not on a single point.

**Forward reference.** Module 20 (latency hiding) develops ILP-versus-occupancy
properly, including the register-pressure side of the trade. Module 19
(occupancy) supplies the arithmetic for "how many warps can I actually have".
Module 18 uses coarsening as a *reuse* mechanism in GEMM, which is a different
argument from this one.

### Fusion: the traffic you can delete

An elementwise pipeline written as one kernel per stage materialises every
intermediate in DRAM. Consider

```
stage 1   t1 = A*x + B*y          reads x, y   writes t1    3N
stage 2   m  = t1 + C*z           reads t1, z  writes m     3N
stage 3   t3 = max(m, 0)          reads m      writes t3    2N
stage 4   d  = S*t3 + O           reads t3     writes d     2N
                                                          ----
                                                           10N
```

Fused into one kernel, `t1` and `t3` never leave a register. If `m` is also a
required output, the fused kernel reads `x, y, z` and writes `m, d`: **5N**.
Predicted speedup 2.0×.

The arrays are 128 MB and L2 is 48 MB, so nothing survives from one launch to
the next — the intermediate really does round-trip to DRAM. That is the
condition under which this arithmetic is valid, and it is the normal condition
for the tensor sizes in Parts XIV and XV.

**The constraint that makes this interesting is `m`.** Fusion removes the
traffic of values nobody outside the fused region needs, and only that traffic.
An intermediate that someone else consumes must still be written. The reflexive
answer "fuse it all, 4N, 2.5×" produces a kernel that is faster and wrong.

**This is the seed of the fused-kernel argument in Parts XIV and XV.** A
transformer's residual-add → RMSNorm → activation → scale chain, written as four
library calls, moves several times the bytes of one hand-written kernel, and
decode-phase inference is memory bound end to end. The arithmetic there is the
arithmetic here.

Measured (`exercise02.cu` solution, N = 33.5M):

| version | ms | traffic | GB/s |
|---|---|---|---|
| unfused, 4 kernels | 3.7314 | 10N | 359.7 |
| partial, 2 kernels | 2.6606 | 7N | 353.1 |
| fused, 1 kernel | 1.9222 | 5N | 349.1 |

measured speedup 1.941×, predicted 2.000×. All three versions run the bus at
81–83% of nominal peak: they are all at their own ceilings, and the only thing
that changed is how much work there was to do.

### Why the fusion prediction misses, in both directions

The traffic model says `speedup = traffic_unfused / traffic_fused`. That is
true only if both versions move their bytes at the *same rate*. Since
`time = traffic / bandwidth`,

```
speedup = (T_u / B_u) / (T_f / B_f) = (T_u / T_f) × (B_f / B_u)
```

The second factor is what the traffic model silently sets to 1. It is not 1.
On this GPU it has been measured on both sides of 1 for the same chain at
different array sizes — 1.10× at 256 MB arrays (fused *beats* prediction) and
0.97× at 128 MB (fused falls short). A fused kernel has more independent
streams in flight per thread, which helps; it also has more concurrent DRAM
row-buffer streams competing, and one fewer chance for L2 to help, which hurts.

**Does fusing more eventually stop paying?** Exercise 2's harness sweeps the
number of fused input streams and answers with data rather than folklore. On
this GPU, measured:

| inputs | traffic | GB/s | vs 2 inputs |
|---|---|---|---|
| 2 | 3N | 344.3 | 1.000 |
| 3 | 4N | 363.4 | 1.056 |
| 4 | 5N | 361.7 | 1.051 |
| 5 | 6N | 365.6 | 1.062 |
| 6 | 7N | 366.0 | 1.063 |

**It does not, up to at least six input streams.** The register count does
climb with fusion depth (34 → 40 for the kernels above; a 16-input version
reaches 48 and a 32-input version 80, with no spills), but the occupancy it
costs is nowhere near the occupancy a bandwidth-bound kernel needs: 80
registers per thread still allows 24 resident warps per SM, and the grid sweep
above shows the bus saturating at roughly 8. There is an order of magnitude of
margin. The register-pressure warning about fusion is real advice for
*compute-bound* kernels; for elementwise streaming on this part it is a
non-event, and the exercise reports it honestly either way.

### When elementwise work is NOT bandwidth-bound

Everything above assumes arithmetic is free. It is free because there is so
much of it available: 12 bytes per element at ~376 GB/s is ~32 ps per element,
and 40 SMs × 128 FP32 lanes at ~1.9 GHz retire ~9.7×10³ FP32 lane-operations in
that time. The budget is enormous — but it is finite.

`example02.cu` Part D applies a function `K` times per element and sweeps `K`.
Traffic is constant at 2N; only arithmetic moves.

```
ms
K =              1        2        4        8       16       32
FFMA        1.5100   1.5087   1.5093   1.5206   1.5295   1.4754
sinf        1.5709   1.6293   1.7972   2.8437   4.7917   8.3566
__sinf      1.5359   1.5361   1.5395   1.5599   1.6333   2.5300
expf        1.5432   1.5478   1.5006   1.6920   2.2548   4.0143
__expf      1.5783   1.5476   1.5454   1.5569   1.6374   2.8121
```

A row is flat while the kernel is memory bound: the FP32 and SFU pipes are
running in DRAM's shadow and cost nothing. The `K` where a row starts to climb
is that function's crossover.

- **`FFMA` never crosses**, even at `K = 64`. The memory system gives away
  dozens of free FLOPs per element. That is the memory-bound side of the
  roofline, quantified.
- **`sinf` crosses around `K = 8`**, `expf` around `K = 8`.
- **`__sinf` crosses around `K = 32`**, `__expf` around `K = 16–32`.

The SASS says exactly why. Accurate `sinf` on sm_89 is a Cody–Waite argument
reduction, a minimax polynomial, and a guarded fallback for large arguments
that uses **FP64** (`I2F.F64.S64`, `DMUL`, `F2F.F32.F64`) — and Ada runs FP64 at
1/64 rate:

```
/*01c0*/  FFMA R10, R9, -1.5707962512969970703, R0 ;   <- Cody-Waite, pi/2 hi
/*01d0*/  FFMA R10, R9, -7.5497894158615963534e-08, R10 ;      pi/2 mid
/*01e0*/  FFMA R10, R9, -5.3903029534742383927e-15, R10 ;      pi/2 lo
/*0820*/  I2F.F64.S64 R6, R8 ;                          <- FP64 fallback path
/*0860*/  DMUL R6, R6, c[0x2][0x0] ;
/*0870*/  F2F.F32.F64 R6, R6 ;
```

`__sinf` is two instructions:

```
/*0550*/  FMUL.RZ R4, R4, 0.15915493667125701904 ;     <- multiply by 1/(2 pi)
/*0560*/  MUFU.SIN R11, R4 ;                            <- one SFU op
```

`MUFU` runs on the **Special Function Unit**: 4 SFUs per processing block
against 32 FP32 lanes, so an SFU instruction costs roughly 8 FP32 slots of
throughput, not 1. That is why `__sinf` is worth ~4–8 FFMAs rather than ~2, and
why it moves the crossover out by roughly 4× rather than 15×.

**`-use_fast_math`.** The flag rewrites `sinf → __sinf`, `expf → __expf`,
`a/b → __fdividef(a,b)`, enables FTZ and more. Verified on this machine:
compiling a plain `sinf` kernel with `-use_fast_math` produces exactly the
`FMUL.RZ` + `MUFU.SIN` pair above. But read the `K = 1, 2, 4` columns before you
reach for it: **on a bandwidth-bound elementwise kernel it changes nothing**,
because the thing it makes cheaper was never on the critical path. Exercise 3
makes you predict this for a real SwiGLU activation before you measure it.

### Why shared memory does not appear in this module

Module 6 gave shared memory a cost model: it pays when there is a **reuse
factor** `K > 1` — when the same loaded value is consumed by more than one
thread or more than once. An elementwise kernel has `K = 1` by definition. Every
byte is read by exactly one thread and used exactly once. Staging it through
shared memory adds a global→shared store, a barrier, and a shared→register load,
and buys nothing. It also costs occupancy, which for a bandwidth-bound kernel is
the one resource you were not short of.

Knowing *why* a tool does not apply is worth as much as knowing how to use it.
Shared memory returns in Module 15 (transpose), where the reuse factor is still
1 but the *access pattern* cannot be coalesced on both sides at once — a
different justification entirely.

---

## Hardware Mental Model

### Why a memory-bound kernel saturates at low occupancy

Module 1 gave you Little's Law. Apply it to the memory system rather than to
the warp schedulers.

The DRAM interface sustains ~380 GB/s and a dependent load takes ~575 cycles
(Module 4's measurement). To keep the interface busy, ~115 kB must be in flight
at every instant. The SM's Load/Store Unit tracks outstanding requests in a
finite structure; each entry holds one request of 32, 64 or 128 bits.

Two independent knobs fill that structure:

- **threads** — more resident warps, each with a request outstanding;
- **MLP per thread** — more independent requests per thread, from unrolling,
  coarsening, or simply having several input streams.

The hardware sums them. It has no way to distinguish "40,960 threads with 1
request each" from "5,120 threads with 8 each", and the measurements above show
it does not. This is why the grid sweep is flat over a wide region, why
`float4` helps at 1 block/SM and not at 8, and why coarsening is *only* an
optimization for a starved grid.

It is also why **occupancy is the wrong target for these kernels**. A kernel
that needs 8 resident warps per SM to saturate DRAM can afford to spend
registers freely up to the point where it drops below 8. Module 19 will make
that budget precise; the point here is that the budget is large.

### Why a partial-sector store costs a read

DRAM has no byte enables at the granularity the memory system operates on. The
smallest thing GDDR6 will transfer on a 32-bit channel is a 32-byte burst, which
is why the sector is 32 B (Module 5). A store of 16 bytes into a 32-byte sector
is therefore not expressible as a DRAM operation. The memory system's only
option is: read the sector into L2, merge the new bytes, write the sector back.

Notice where this happens: at **L2**, not at DRAM and not at L1. That has two
consequences you can use.

- If the sector is *already* in L2 because a neighbouring warp just touched it,
  the merge is free and no DRAM read occurs. This is why Module 5's `+1 float`
  misalignment measured 98% of contiguous instead of the predicted 80%: warp
  *k*'s fifth sector is warp *k+1*'s first.
- If the kernel's footprint is much larger than L2 and the stores are scattered,
  every merge is a compulsory DRAM read. That is the 4.08× above.

The two cases have the same instruction count and the same source code. Only the
*locality of the store addresses relative to L2 capacity* differs.

### Why cache hints cannot help

`__stcs` sets an eviction-priority bit on the store. It tells L2 "this line is
unlikely to be reused; prefer to evict it". It does not and cannot say "you may
discard the 16 bytes I did not write". Those bytes belong to the program. The
only way to avoid fetching them is to write them, which means changing the
access pattern, not the opcode.

`__stwt`'s SASS on sm_89 is `STG.E.STRONG.SYS` — a *system-scope strong* store,
which is a stronger memory-ordering guarantee than the plain `STG.E`, not a
weaker caching one. On a streaming kernel with no other observer it measures the
same; in a multi-GPU or host-visible context it would not be free. That is a good
reason to prefer `__stcs` when you actually want the streaming behaviour.

### Why the SFU shifts the crossover by 4× and not by 15×

Each Ada processing block has 32 FP32 lanes and **4** Special Function Units. A
warp's `MUFU` instruction therefore occupies the SFU for 8 cycles where an
`FFMA` occupies the FP32 lanes for 1. Replacing a ~30-instruction `sinf`
sequence with a 2-instruction `__sinf` sequence looks like a 15× reduction in
instruction count and delivers about 4× in throughput, because one of the two
remaining instructions is 8× more expensive than the average instruction it
replaced.

This is the general shape of the intrinsic-versus-libm trade and it is why the
answer is always "measure the kernel", never "intrinsics are 10× faster".

### Why fusion raises achieved bandwidth (sometimes)

A fused kernel issues `M` independent loads per element instead of 2. That is
more MLP per thread, so the outstanding-request structure fills faster and DRAM
is asked for more work at once — which, in the measurements above, was worth up
to 10% of achieved bandwidth on top of the traffic saving.

It also concentrates `M+1` concurrent streams onto the same memory controllers.
GDDR6 has an activated row per bank; every additional stream is another row that
wants to stay open. Past some stream count, row thrashing costs more than the
extra MLP earns. On this GPU that point is beyond 6 streams and was not reached;
on a part with fewer banks or a narrower bus it would arrive sooner. The *shape*
of the argument is portable; the *number* is not.

---

## Code Walkthrough

### `example01.cu` — counting the floor

The file is three measurements and one discipline. The discipline is in the
table header:

```cpp
printf("  %-30s %6s %10s %10s %8s %8s\n",
       "kernel", "bytes", "floor ms", "meas ms", "x floor", "GB/s");
```

Every kernel is reported as a multiple of its own floor. A GB/s column alone
would let a 4N kernel and a 2N kernel look comparable; the `x floor` column
makes "is this finished?" a one-glance question.

The floor itself is computed from a **measured** ceiling, not the nominal peak:

```cpp
const double ceilGBs = (8.0 * (double)N) / (best[K_COPY] * 1e-3) / 1e9;
```

`k_copy` moves 2N compulsory bytes and nothing else, and it is timed in the same
rotated sweep as everything it is compared against. On a laptop GPU whose SM
clock swings 0.49–2.04 GHz and whose memory P-state ramps 6001 → 8001 → 9001 MHz,
a ratio against a simultaneously-measured yardstick is the only quantity that
survives.

The in-place SAXPY row exists to be misread, and the program says so:

```cpp
printf("\n  The row that matters is `saxpy ip`. Read the source: one load of\n"
       "  x, one store to y. If you count 2N you get %.1f GB/s, ...");
```

Part B is the write-allocate experiment. The two kernels differ by one
character:

```cpp
__global__ void k_write_dense(float* o, long long nHalf, float v)
{ for (GRID_STRIDE(nHalf)) o[i]   = v; }
__global__ void k_write_odd(float* o, long long nHalf, float v)
{ for (GRID_STRIDE(nHalf)) o[2*i] = v; }
```

and the validation pass is where the mechanism becomes visible rather than
inferred:

```cpp
CHECK(cudaMemset(o, 0, N * sizeof(float)));
k_write_odd<<<GRID, TPB>>>(o, nHalf, 7.5f);
...
for (long long i = 0; i < NV; ++i) {
    const float want = (i % 2 == 0) ? 7.5f : 0.0f;
    if (h[i] != want) ++bad;
}
```

The odd elements still hold their old values. Preserving them *is* the
read-modify-write. The test that proves the kernel correct is the same test that
explains why it is four times slower.

Part C prints the hint comparison and then refuses to dress it up. Reporting a
null result plainly, with the SASS opcode that shows the hint did take effect, is
more useful than a paragraph of theory about non-temporal stores.

### `example02.cu` — reaching the floor

Part A is the grid sweep, and the important line is the summary, not the table:

```cpp
printf("  Read the table by TOTAL RESIDENT THREADS, not by either axis.\n");
```

followed by the Little's Law arithmetic computed from the measurement itself, so
the number of bytes in flight is derived rather than quoted.

Part B measures the vector widths at **two** grid sizes:

```cpp
const int SMALL = nSM * 1;      // 1 block/SM  =  5,120 threads
const int LARGE = nSM * 8;      // 8 blocks/SM = 40,960 threads
```

Measuring only at `LARGE` would have produced the conclusion "vectorization does
nothing", which is true at `LARGE` and false in general. Measuring only at
`SMALL` would have produced "vectorization is worth 1.2×", which is the opposite
error. The two-column table is the finding.

Part C is the MLP experiment, and the entire design is in one pragma:

```cpp
template <int C>
__global__ void k_serial(...)
{
    for (long long i = base; i < n; i += stride * C) {
        #pragma unroll 1
        for (int c = 0; c < C; ++c) { ... o[j] = a*x[j] + y[j]; }
    }
}
```

`#pragma unroll 1` is the only difference from `k_mlp<C>`. Without it the
compiler unrolls, hoists, and produces the fast kernel — which is the same
hazard Module 7 hit when 32 adjacent shared-memory reads became 8 `LDS.128`
instructions and collapsed the effect under study. The SASS is checked, not assumed:
`k_serial<4>` emits `2 × LDG.E.CONSTANT`, `k_mlp<4>` emits 24.

Part D is the crossover sweep. Each configuration gets its own iteration count:

```cpp
float p = 0.f; CHECK(cudaEventElapsedTime(&p, a, b));
int it = (int)(10.0 / p);
```

because the `K=1` kernels run in 1.5 ms and the `K=32 sinf` kernel runs in 8 ms;
a fixed 20 iterations would produce a 30 ms segment for one and a 160 ms segment
for the other, and the clock behaves differently in those two regimes.

---

## Check Your Understanding

Answers in `solutions/module11/check_your_understanding.md`. None of these can
be looked up.

1. A colleague reports that their in-place normalization kernel,
   `x[i] = (x[i] - mean) * inv_std` over 200M floats, achieves 250 GB/s and
   asks how to get it to 376. Give the number they should have computed, the
   number they actually achieved, and what you would tell them to do. Then
   describe a *different* kernel, also a single load and a single store per
   element in the source, for which 250 GB/s really would indicate a defect,
   and say what distinguishes the two cases.

2. You have two candidate implementations of the same elementwise op. A uses
   5,120 threads and `float4` loads with 4 elements per thread. B uses 40,960
   threads and scalar loads with 1 element per thread. Both move the same
   compulsory bytes. Predict their relative speed on this GPU and justify it
   from the outstanding-request argument. Then name one change to the *problem*
   (not the code) that would make A clearly better than B, and one that would
   make B clearly better than A.

3. A pipeline computes `b = f(a)` then `c = g(b)` then `d = h(c)`, all
   elementwise over arrays of size S, with `a` and `d` live outside and `b`, `c`
   private. Write the unfused and fused compulsory traffic as functions of S.
   Now suppose S is small enough that all four arrays fit comfortably in L2.
   State what happens to the predicted speedup and to the measured speedup, and
   explain why they move differently. Finally, state the condition on S under
   which fusion is *not* worth the engineering effort, in terms of a quantity
   you can look up for this GPU.

4. Kernel P writes `out[i] = c` for `i` in `[0, M)`, where `out` is `M` floats.
   Kernel Q writes `out[3*i] = c` for the same `i` in `[0, M)`, where `out` is
   `3M` floats. Both execute `M` store instructions and both write `4M` useful
   bytes; no reads appear in either source. Rank them by wall-clock time,
   quantify the ratio, and state the DRAM traffic of each. Then explain why
   adding `__stcs` to Q does not change the answer, and describe the one change
   to Q's *data layout* — not its code — that would.

---

## Exercises

### Exercise 1 — `exercise01.cu` (optimization, with a constraint)

`blend_v1` is correct, coalesced, and a grid-stride loop, and it runs at about a
quarter of this machine's streaming ceiling. Produce `blend_v2`, reaching ≥ 95%
of the ceiling the program measures in its own timing loop, under a hard budget
of 8,192 threads.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

- **TODO 1** — the compulsory traffic in bytes per element, *before* you write
  any code. Checked exactly, against a hashed reference.
- **TODO 2** — the time floor in milliseconds at 87% of 432 GB/s. Checked to
  ±15%.
- **TODO 3** — the launch configuration, within the budget. There is more than
  one defensible answer; be able to say why yours is one of them, and in
  particular why `gridDim.x` below the SM count is not.
- **TODO 4** — the kernel. You are not told which mechanism to use. `blend_v1`
  is already fully coalesced, so there are no wasted sectors to recover; the
  missing resource is something else and Little's Law says what.
- **TODO 5** — the number of elements your main path covers, so the validator
  can report a tail bug as a tail bug.

Validation: numerics for both versions; the tail range checked separately; a run
at the degenerate launch `<<<1,32>>>` to prove your kernel does not depend on the
launch shape for correctness; the budget; both predictions; and the 95% gate.
Eight points, all eight required.

`N = 40,000,001` and the operation is **not idempotent** — an element processed
twice is wrong and will be reported.

### Exercise 2 — `exercise02.cu` (performance reasoning + design)

Predict the speedup from fusing a four-stage pipeline, build the fused version,
then explain the difference between prediction and measurement.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

- **TODO 1 / 2** — the unfused and best-fused traffic, in units of N floats.
  Both checked against hashed references. One of the two has a constraint in the
  problem statement that most readers will not apply on the first attempt.
- **TODO 3** — the partially fused and fully fused kernels.
- **TODO 4** — the algebra that reconciles the traffic prediction with the
  measurement. Checked to 5%.
- **TODO 5** — an occupancy prediction, committed before running. The harness
  reads the real register counts with `cudaFuncGetAttributes` and prints the
  warp arithmetic.

The harness also sweeps fusion *depth* and prints achieved bandwidth against the
number of fused input streams, so that if fusing more ever stops paying on this
hardware you will see the row where it happens.

### Exercise 3 — `exercise03.cu` (predict-then-measure)

Part 1 is a SwiGLU-style SiLU-gated activation with one `expf` and one divide
per element. Part 2 finds the arithmetic crossover directly by applying a
function `K` times and sweeping `K`.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

- **TODO 1 / 2** — the activation's traffic, and whether it is memory bound.
- **TODO 3** — an intrinsic version within a 2×10⁻³ relative error budget that
  is not slower than the accurate one. Which approximations you accept is your
  decision; be ready to justify the budget for a network with 8-bit weights.
- **TODO 4** — the crossover `K` for `sinf` and for `__sinf`, predicted before
  the sweep runs. Each scored to within a factor of two.
- **TODO 5** — whether `-use_fast_math` will measurably speed up the activation.

Seven points, all seven required.

---

## Prediction

Commit to these in writing before you run anything.

1. **`example01.cu`, Part A.** The in-place SAXPY `y[i] += a*x[i]` and the
   out-of-place `d[i] = a*x[i] + y[i]` do the same arithmetic. State which is
   faster and by what ratio, and give the reason. Then state what the in-place
   version's GB/s would be if you counted its traffic as 2N, and what conclusion
   that number would lead you to.

2. **`example02.cu`, Part C.** At 8 blocks per SM, predict whether the
   `C = 16` hoisted kernel will be faster, slower, or the same as the `C = 1`
   kernel. Give a reason that does not use the word "occupancy".

3. **`exercise03.cu`.** Predict the ratio between the accurate and intrinsic
   SiLU kernels *before* opening the file, then predict the ratio between
   `sinf` and `__sinf` at `K = 32`. One of these two ratios is approximately 1
   and the other is not. Say which, and why the difference is not a
   contradiction.
