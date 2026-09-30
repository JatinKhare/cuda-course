# Module 12 — Reduction

> Prerequisites: Module 1 (SM anatomy, waves, Little's Law), Module 3 (indexing,
> grid-stride loops), Module 5 (coalescing, sectors, the streaming ceiling),
> Module 6 (shared memory), Module 7 (bank conflicts), Module 8 (warps,
> divergence, independent thread scheduling), Module 9 (`__syncthreads`, fences,
> `volatile` is not synchronization), Module 10 (atomics, float `atomicAdd`
> non-determinism).
> What this module gives you: the ability to take any associative operator over
> a large array and turn it into a kernel that runs at the memory bandwidth of
> the machine — and to know, at every step, exactly which hardware resource the
> version you have just written is wasting.

This is where the course's first eleven modules get spent at once. The reduction
ladder is the standard vehicle for that because every rung of it is a different
one of them: rung 1 is Module 8's divergence, rung 2 is Module 7's bank
conflicts, rung 3 is Module 1's idle-lane arithmetic, rung 4 is Module 3's
grid sizing, rung 5 is Module 9's barrier semantics and Module 8's independent
thread scheduling, rung 6 is Module 5's bandwidth ceiling. Nothing new about the
hardware appears in this module. What appears is the discipline of asking, of
each version, *which* resource it is wasting and *how much*.

Three debts are paid here.

- **Module 8** named reductions as this module's property and refused to build
  one out of the warp primitives it was using as instruments.
- **Module 9** showed the `volatile __shared__` warp-synchronous reduction tail
  as a bug, said `__shfl_down_sync` was its correct replacement, and pointed
  here for the algorithm.
- **Module 10** measured float `atomicAdd` producing ten distinct bit patterns
  in ten identical runs, named "fixed-order reduction" as the deterministic
  alternative, and deliberately shipped no reduction. Exercise 3 pays that in
  full, including the half of it Module 10 did not know about: a fixed-order
  tree is reproducible only for a **fixed decomposition**.

---

## Concept

### What a reduction is

A **reduction** collapses a sequence `x[0..n)` to a single value using a binary
operator:

```
reduce(op, x) = x[0] op x[1] op ... op x[n-1]
```

For this to be parallelizable at all, `op` must be **associative**:
`(a op b) op c == a op (b op c)`. Associativity is exactly the licence to
re-bracket, and re-bracketing is the whole algorithm. It is what lets you turn a
chain of `n-1` dependent operations into a tree of depth `ceil(log2 n)`.

`op` also needs an **identity** `e` with `e op x == x op e == x`, so that a
thread with no work has something to contribute. Together `(op, e)` is a
**monoid**. That is the entire mathematical requirement.

Notice what is *not* required: **commutativity**. `a op b == b op a` is a
separate property, and every reduction in this lesson's worked examples has it,
which is precisely why the textbook treatment never mentions it. Exercise 2
removes it, and two decisions that were invisible for a sum — which operand goes
on the left, and the order in which you take the shuffle offsets — become
correctness bugs.

The cost model of the two shapes:

| | work | depth (critical path) |
|---|---|---|
| sequential loop | `n-1` | `n-1` |
| balanced tree | `n-1` | `ceil(log2 n)` |

Both do the same number of operations. The tree does them in `log2 n` rounds
instead of `n`. For `n = 2^26` that is 26 rounds instead of 67 million. **PORTABLE
CUDA CONCEPT.**

### The ceiling: reduction is a bandwidth problem

Before writing a single kernel, work out what the answer is allowed to cost.

A reduction of `n` floats reads `4n` bytes and writes 4. It performs `n-1`
additions. The arithmetic intensity is

```
(n-1) FLOP / 4n bytes  ->  0.25 FLOP per byte
```

This GPU has a 192-bit GDDR6 bus at 9.001 GHz, giving **432.0 GB/s** of pin
bandwidth, and (Module 1) 40 SMs × 128 FP32 lanes at roughly 1.8 GHz, giving
about 18 TFLOP/s of FP32 add throughput. The machine wants about **42 FLOP per
byte** to keep both busy. A reduction offers 0.25. It is memory-bound by a
factor of roughly 170.

Consequences, and they govern everything below:

1. **The floor on the runtime is `4n / BW`.** For `n = 2^26` and a realistic
   375 GB/s that is 0.72 ms. No reduction kernel can be faster.
2. **The addition is free.** Not "cheap" — free. If a version of the kernel is
   slower than the floor, it is not because of arithmetic, and making the
   arithmetic cleverer will not help.
3. **The only honest scoreboard is percentage of the achievable streaming
   bandwidth**, and the achievable figure is not 432. Module 5 recorded
   375–381 GB/s for a pure stream; `example01.cu` measures its own ceiling with a
   read-only kernel inside the same timed sweep, and gets 373–411 GB/s depending
   on the memory P-state. Every version in this module is reported as a
   percentage of *that measured number*, not of 432.

So the ladder below is not a story about getting faster. It is a story about a
kernel that starts out wasting three quarters of the memory system and stops.
**Version 6 is not "fast"; it has merely stopped being wasteful.**

### The ladder

Six versions, each repairing one nameable defect of its predecessor. Measured on
this GPU at `n = 2^26` (256 MiB, comfortably past the 48 MB L2), block size 256,
minimum of four rotated sweeps, with the ceiling measured in the same sweep:

| version | GB/s | % of ceiling | step | the defect it removes |
|---|---|---|---|---|
| v1 interleaved, `tid % (2*s)` | 101.5 | 24.7% | — | — |
| v2 contiguous index `2*s*tid` | 144.8 | 35.2% | 1.43× | **divergence** |
| v3 sequential addressing | 156.4 | 38.1% | 1.08× | **bank conflicts** |
| v4 first add during load | 311.0 | 75.7% | **1.99×** | **idle threads** |
| v5 warp tail, `__shfl_down_sync` | 408.7 | 99.5% | 1.31× | **tail barriers** |
| v6 grid-stride + full unroll | 407.9 | 99.3% | 1.00× | **block count** |
| *streaming ceiling (no tree)* | *410.7* | *100%* | | |

Across three separate runs in different thermal states the absolute GB/s moved
by 10% and the percentages moved by at most 5 points; the ordering and the
step ratios reproduced every time. Total v1 → v6: **3.3–4.3×**.

#### v1 — interleaved addressing with a modulo

```cpp
for (unsigned s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

The working threads are `0, 2s, 4s, …` — spread across the whole block. On the
first iteration every *other* lane of every warp works, which is Module 8's
`tid & 1` pattern, measured there at **1.96×**. On the second iteration one lane
in four, and so on. Every warp of the block is divergent at every step, for all
eight steps, and every warp participates in all eight barriers whether or not it
has any live lane. There is also a `%` — an integer division — on the critical
path, though at 24.7% of the ceiling that is the least of its problems.

#### v2 — contiguous-thread indexing

```cpp
unsigned idx = 2 * s * tid;
if (idx < blockDim.x) sdata[idx] += sdata[idx + s];
```

The *same* tree, the *same* additions, in the *same* order — only the assignment
of additions to threads changed. Now the working threads are `0..(blockDim/2s)`,
a contiguous prefix, so at `s = 1` exactly four of the block's eight warps are
fully active and four are fully idle. Module 8's lesson, stated as a rule: a
branch costs nothing if every lane of the warp agrees. v2 makes them agree.

The bill it introduces: the addresses `sdata[2*s*tid]` have stride `2s`. Module 7
gives the conflict degree of `s[k*tid]` as `gcd(k, 32)`, so `s = 1` is 2-way,
`s = 2` is 4-way, `s = 4` is 8-way, up to 32-way at `s = 16`. Module 7 also
established that the cost law on Ada is `max(2, D)`, so the 2-way step is free
and the later ones are not.

#### v3 — sequential addressing

```cpp
for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

Every access is `sdata[tid]` or `sdata[tid + s]` — unit stride across the warp,
`D = 1`, conflict-free at every step (Module 7). The tree is now
**reversed**: it runs from the widest stride down instead of the narrowest up,
which is what makes the surviving values stay packed at the bottom of the array.

The measured gain over v2 is only 1.03–1.18×, and it should be. The kernel is at
38% of the memory ceiling; the shared-memory replays it just removed were never
the binding constraint. **This is the single most important measurement in the
ladder**: two textbook optimizations, correctly applied, bought 1.45× combined,
and the reason is that neither of them touched the thing that was actually
wrong.

#### v4 — first add during load

What was actually wrong: with one element per thread, half the block does
nothing from the very first tree iteration onward, and the *whole block* is only
alive for `log2(256) = 8` barrier-separated steps that move 256 floats. The
block's useful lifetime is dominated by launch and teardown, not by work.

```cpp
long long i = blockIdx.x * (blockDim.x * 2) + tid;
float v = (i < n) ? in[i] : 0.0f;
if (i + blockDim.x < n) v += in[i + blockDim.x];
sdata[tid] = v;
```

Halve the grid; each thread reads two elements and adds them *before* the tree
starts. Two things improve at once: the first tree level disappears (it happened
in registers, for free, with every thread busy), and there are half as many
blocks — 131,072 instead of 262,144.

Measured **1.94–1.99×**, the largest single step in the ladder, and the one that
takes the kernel from "a shared-memory exercise" to "a memory-bound kernel".

#### v5 — the warp tail

Once `s` falls to 32, every surviving thread is in warp 0. Warps 1–7 are still
executing the loop, still arriving at the barrier, and still contributing
nothing. The final five steps of the tree are a *warp-local* reduction, and a
warp does not need shared memory or barriers to reduce itself:

```cpp
for (unsigned s = blockDim.x / 2; s >= 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
if (tid < 32) {
    float w = warpReduceSum(sdata[tid]);       // five shuffles, no barrier
    if (tid == 0) out[blockIdx.x] = w;
}
```

with

```cpp
__device__ __forceinline__ float warpReduceSum(float v)
{
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;                                  // valid in lane 0
}
```

`__shfl_down_sync(mask, var, delta)` returns the value of `var` held by lane
`lane + delta` of the same warp. If `lane + delta >= 32` the lane receives its
own value back, which is why the result is meaningful **only in the lower
lanes** and why only lane 0 may write.

The `mask` is not decoration. Module 8 established that on sm_70 and later every
thread has its own program counter and the hardware does not guarantee
reconvergence anywhere. The mask is how you **create** the convergence the
hardware no longer gives you: the named lanes are forced to participate in this
instruction together. `0xffffffff` is correct here because all 32 lanes of warp 0
enter the `if (tid < 32)` region. Passing `__activemask()` instead is a bug even
when it happens to return the value you wanted — Module 8 spent an exercise on
exactly that, and the mask you want is a property of your algorithm, not of
where the scheduler has put the warp.

Module 30 owns `__shfl_*_sync` as a topic — shuffles as a general communication
primitive, ballots as a compaction primitive. This module uses them because the
algorithm demands them, and does not develop them further.

**The version you must never write.** The classic form of this tail, from
pre-Volta, is:

```cpp
// BROKEN on sm_70 and later. Present in a great deal of existing code.
if (tid < 32) {
    volatile float* v = sdata;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid +  8];
    v[tid] += v[tid +  4];
    v[tid] += v[tid +  2];
    v[tid] += v[tid +  1];
}
```

Modules 8 and 9 established why: this was correct on Kepler because a warp had
one program counter, so all 32 lanes provably executed line *k* before any lane
executed line *k+1*. The `volatile` was never the synchronization — its only job
was to stop the compiler keeping `v[tid]` in a register across the lines. The
synchronization was free hardware lockstep, and independent thread scheduling
removed it. `volatile` gives no execution ordering between threads, no
visibility guarantee, and no atomicity (Module 9). The idiom is **broken, not
deprecated**.

`example02.cu` part F ships it and runs it. On this GPU, today, it produces the
right answer. Module 8 anticipated exactly this and named it the dangerous middle
case: `nvcc` inserts `BSSY`/`BSYNC` convergence barriers at post-dominators and
the hardware honours them, so a converged-region warp-synchronous exchange
usually still works. That is an observation about one code generator on one day.
It is not a guarantee, it is not portable, and code that is correct only because
of a compiler heuristic breaks silently. The SASS shows what the source hides —
six `LDS`/`STS` pairs with no `WARPSYNC` and no `BAR.SYNC` between them:

```
/*08e0*/  LDS R2, [R0.X4] ;
/*08f0*/  LDS R3, [R0.X4+0x80] ;
/*0920*/  LDS R2, [R0.X4] ;
/*0930*/  LDS R5, [R0.X4+0x40] ;
/*0960*/  LDS R2, [R0.X4] ;
/*0970*/  LDS R7, [R0.X4+0x20] ;
...
/*0a60*/  LDS R7, [RZ] ;
```

Nothing in that instruction stream makes lane 5's store visible to lane 4 before
lane 4's load. The correct repairs are `__syncwarp()` between the steps, or —
better, because the data never needs to be in memory at all — `__shfl_down_sync`.

**The barrier that disappears.** The shared loop in v5 ends with `s == 32`, and
it has a `__syncthreads()` at the bottom of that last iteration. That barrier is
not needed. After the `s == 32` step, the only slots anybody reads are
`sdata[0..31]`, and the thread that reads `sdata[tid]` for `tid < 32` is the
thread that wrote it. There is no cross-thread dependency left to order, so
neither of Module 9's guarantees is required. This is one of the few places in
CUDA where the answer to "can I delete this barrier?" is yes — and note that the
reason has nothing to do with warps being in lockstep. It is that the dependency
is thread-local. Exercise 1 makes you commit to that answer.

#### v6 — grid-stride accumulation and compile-time unrolling

v5 is already at 99.5% of the ceiling, so v6 cannot make the 256 MiB case
faster, and it does not. What it fixes is structural.

```cpp
template <unsigned BLOCK>
__global__ void reduce6(const float* __restrict__ in, float* out, long long n)
{
    __shared__ float sdata[BLOCK];
    unsigned tid   = threadIdx.x;
    long long i    = (long long)blockIdx.x * (BLOCK * 2) + tid;
    long long step = (long long)BLOCK * 2 * gridDim.x;

    float sum = 0.0f;
    while (i + BLOCK < n) { sum += in[i] + in[i + BLOCK]; i += step; }
    while (i < n)         { sum += in[i];                 i += step; }   // tail

    sdata[tid] = sum;
    __syncthreads();
    if (BLOCK >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (BLOCK >=  512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (BLOCK >=  256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (BLOCK >=  128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }

    if (tid < 32) {
        float w = sdata[tid];
        if (BLOCK >= 64) w += sdata[tid + 32];
        w = warpReduceSum(w);
        if (tid == 0) out[blockIdx.x] = w;
    }
}
```

Three changes:

1. **The grid is a property of the machine, not of the data.** It is
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor × multiProcessorCount` = 6 × 40
   = **240 blocks**, one occupancy-limited wave (Module 1, Module 3). Each thread
   walks the array with a grid stride, accumulating into a register. The shared
   tree and its five barriers run **once per block** instead of once per 512
   elements.
2. **The tree bounds are compile-time constants.** `BLOCK` is a template
   parameter, so every `if (BLOCK >= k)` is resolved at compile time and the loop
   is gone. Module 4 introduced the template-parameter dispatch trick; this is
   its canonical use.
3. **The partial array is 240 floats instead of 262,144.**

Point 3 is where the win actually shows up, and it only shows up if you time the
thing everyone forgets to time — the second pass:

| version | partials | pass 1 (ms) | pass 2 (ms) | total (ms) |
|---|---|---|---|---|
| v1 | 262144 | 2.5658 | 0.0432 | 2.6090 |
| v3 | 262144 | 1.6803 | 0.0385 | 1.7188 |
| v5 | 131072 | 0.7219 | 0.0244 | 0.7464 |
| v6 | **240** | 0.7229 | **0.0125** | **0.7355** |

And on data small enough to live in L2 — 4 MiB — where per-block overhead is the
whole story rather than a rounding error:

| version | ms (4 MiB) | vs v6 |
|---|---|---|
| v1 | 0.03708 | 3.80× |
| v3 | 0.02628 | 2.69× |
| v5 | 0.01474 | 1.51× |
| v6 | 0.00977 | 1.00× |

v6 is 1.5× faster than v5 on a small array and identical on a large one. That is
what "fully optimized" buys once you are already at the bandwidth limit: not
speed, but insensitivity to the problem size.

> Vectorized `float4` loads are the obvious next idea and are deliberately not
> here: Module 5 introduced them and Module 11 owns them at scale. At 99.5% of
> the measured ceiling there is nothing left for them to recover.

### Multi-block finalization

Every version above leaves one partial sum per block, and a block cannot talk to
another block (Module 9: there is no cross-block synchronization, and spinning on
a global flag deadlocks). Three strategies, all measured in `example02.cu`:

**(a) A second kernel launch.** Pass 1 writes `partial[blockIdx.x]`; pass 2 is a
single block that sums them in index order. Module 9's rule: **the kernel
boundary is the only grid-wide barrier that holds for an arbitrary grid size.**

**(b) `atomicAdd` per block.** Thread 0 of each block does
`atomicAdd(out, blockSum)`. One launch, no partial array, no second pass.
Module 10: the atomic executes at the L2, it is uncontended at 240 addresses…
no, at *one* address with 240 arrivals, which Module 10's contention curve puts
at the cheap end because 240 atomics over a 0.72 ms kernel is nothing.

**(c) A last-block-done flag with `__threadfence()`.** Every block stores its
partial, fences, and increments a counter; the block that gets ticket
`gridDim.x - 1` knows every other block's store is visible and does the final
tree itself.

```cpp
if (threadIdx.x == 0) {
    partial[blockIdx.x] = w;
    __threadfence();                                  // publish BEFORE the ticket
    unsigned ticket = atomicAdd(&g_blocksDone, 1u);
    amLast = (ticket == gridDim.x - 1);
}
__syncthreads();                                      // block-uniform below: legal
if (amLast) { /* fixed-order tree over partial[0..gridDim) */ }
```

The `__threadfence()` and not `__threadfence_block()`: the observer is a thread
on a different SM, and Module 4 established that L1 is not coherent across SMs,
so device-scope visibility means pushing to L2. This is Module 9's release
pattern with Module 10's atomic as the flag. Note that it is **not** a
cross-block barrier — nobody waits. Each block either is the last one or leaves.

Measured, `n = 2^26`, grid 240:

| strategy | ms | GB/s | % of ceiling | launches |
|---|---|---|---|---|
| (a) two kernel launches | 0.7291 | 368.2 | 98.7% | 2 |
| (b) `atomicAdd` per block | 0.7278 | 368.9 | 98.9% | 1 |
| (c) `__threadfence` + last block | 0.7257 | 369.9 | 99.2% | 1 |

**They are the same speed.** All three are within 0.5% of each other and all
three are within 1.3% of the pure-streaming ceiling. At `n = 2^26` the second
launch costs a few microseconds against 0.73 ms, and 240 atomics cost nothing.
Choose on other grounds:

| | (a) two kernels | (b) atomicAdd | (c) last-block flag |
|---|---|---|---|
| launches | 2 | 1 | 1 |
| extra memory | grid floats | none | grid floats + counter |
| deterministic | **yes** | **no** | **yes** |
| needs the counter reset | no | yes (`out` must be 0) | yes |
| works if a block is the only one | yes | yes | yes |
| gets worse as the grid grows | second pass grows | contention grows | last block's serial work grows |

Strategy (c) is the interesting one pedagogically because its determinism is
easy to get wrong in your head: *which* block finishes last is completely
unspecified, and yet the answer is bit-reproducible — because the final tree
walks `partial[]` in **index** order, and index `c` always holds block `c`'s sum,
computed by block `c` in a fixed order. Determinism comes from the summation
order being fixed, not from the schedule being fixed.

### Determinism

Module 10 measured `float atomicAdd` producing 10 distinct bit patterns in 10
runs and named fixed-order reduction as the alternative. Here is that experiment
inside a real reduction, `n = 2^26`, grid 240, same input every time:

```
distinct bit patterns over 10 identical runs, same grid:
  (a) two kernel launches      : 1   (0x4f4d04ce)
  (b) atomicAdd per block      : 9
        run  0 -> 0x4f4d04cb  3439643392.000000
        run  1 -> 0x4f4d04d1  3439644928.000000
        run  2 -> 0x4f4d04c8  3439642624.000000
        ...
        run  9 -> 0x4f4d04d0  3439644672.000000
  (c) threadfence + last block : 1   (0x4f4d04ce)
```

Float addition is not associative. `atomicAdd` imposes *an* order and does not
specify *which*, so the order is whatever the 240 blocks happened to finish in,
and that changes with the scheduler's mood. The spread here is 9 distinct values
across ~2560 ULP — small in relative terms (1e-6) and completely fatal to any
workflow that diffs two runs.

The half of this that Module 10 could not have told you, because it needs a
complete reduction to demonstrate:

```
same kernel, three DIFFERENT grid sizes (two-kernel strategy):
  grid =   240 blocks -> 0x4f4d04ce  3439644160.000000
  grid =   120 blocks -> 0x4f4d04c0  3439640576.000000
  grid =   480 blocks -> 0x4f4d04d4  3439645696.000000
```

**A fixed-order tree is reproducible only for a fixed decomposition.** Change the
grid and you change how the array is cut up, which changes the bracketing, which
changes the bits. This matters because the grid is normally chosen from the
device — a library compiled once and run on two different GPUs gives two
different answers, with no atomics anywhere. A library that promises
reproducibility has to fix the *decomposition* and let only the *traversal*
depend on the grid. That is Exercise 3's design problem, and the measured cost of
doing it is **1.00–1.03×**, i.e. nothing.

The other route is to stop using floats: accumulate into a 64-bit integer with a
fixed-point scale. Integer addition is associative *and* commutative, so a plain
`atomicAdd(unsigned long long*)` is bit-reproducible for every order, every grid
and every run. What you give up is dynamic range — you must know the magnitude of
your data in advance to pick the scale. Exercise 3 makes you derive both bounds.

### Floating-point accuracy — the parallel version is *better*

Readers usually expect the parallel reduction to be a numerical compromise. It is
the opposite.

Let `eps = 2^-24` be the float unit roundoff. For summation of `n` values the
standard worst-case bound on the relative error is proportional to the **depth**
of the summation tree:

- sequential accumulation has depth `n-1` → error `O(n) * eps`
- a balanced tree has depth `log2 n` → error `O(log n) * eps`

For `n = 2^26` that is a factor of 2.6 million in the bound. And the bound is not
the worst of it. Sequential accumulation has a failure mode that is not
gradual: once the running total exceeds `2^24` times a typical addend, each new
addend rounds to nothing and the sum **stops increasing altogether**.

`example02.cu` part C, `n = 2^26`:

```
dataset A: 67108864 copies of 1.0f, exact = 67108864.0
  CPU sequential float loop                    16777216.0     7.500e-01
  CPU Kahan compensated float                  67108864.0     0.000e+00
  GPU tree reduction (float)                   67108864.0     0.000e+00

dataset B: values 2^-10..2^9, exact = 3439646263.9
  CPU sequential float loop                  3131606016.0     8.956e-02
  CPU Kahan compensated float                3439646208.0     1.626e-08
  GPU tree reduction (float)                 3439644160.0     6.117e-07
```

Dataset A is the stall made visible: the sequential float loop saturates at
exactly `2^24 = 16777216` and returns **75% low**. It is not slowly drifting; it
has stopped. Dataset B, with a wide but ordinary dynamic range, costs the
sequential loop **9%**. The tree costs 6e-7 — close to Kahan compensated
summation, at no extra cost, because the tree structure *is* the compensation.

This is the answer to "should I use `double` on the GPU?" for accumulation: on
Ada, FP64 runs at 1/64 rate (Module 7 measured what that does to a
microbenchmark), and a float tree reduction over 2^26 elements is already within
a few ULP. Reach for the tree before you reach for `double`.

### Warp-level primitives, and the sm_80 hardware reduction

Three ways to write the last five steps:

```cpp
// 1. shuffles
for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);

// 2. the sm_80+ hardware instruction (INTEGER types only)
unsigned s = __reduce_add_sync(0xffffffffu, v);

// 3. cooperative groups
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
unsigned s = cg::reduce(warp, v, cg::plus<unsigned>());
```

`__reduce_add_sync` (and `_min_`, `_max_`, `_and_`, `_or_`, `_xor_`) exists from
sm_80 and is a genuine hardware instruction, not a library helper. Verified on
this sm_89 GPU with `cuobjdump -sass`:

```
== _Z10uintReduceILi1EEvPKjPjx                    (__reduce_add_sync)
        /*08d0*/   REDUX.SUM UR6, R2 ;

== _Z10uintReduceILi0EEvPKjPjx                    (__shfl_down_sync x5)
        /*08d0*/   SHFL.DOWN PT, R3, R2, 0x10, 0x1f ;
        /*08f0*/   SHFL.DOWN PT, R6, R3, 0x8,  0x1f ;
        /*0910*/   SHFL.DOWN PT, R5, R6, 0x4,  0x1f ;
        /*0930*/   SHFL.DOWN PT, R8, R5, 0x2,  0x1f ;
        /*0950*/   SHFL.DOWN PT, R7, R8, 0x1,  0x1f ;

== _Z10uintReduceILi2EEvPKjPjx                    (cg::reduce, unsigned)
        /*08d0*/   REDUX.SUM UR6, R2 ;
```

One instruction instead of ten, and the destination is a **uniform register**
(`UR6`) — the hardware knows the result is the same in every lane.
**ARCHITECTURE-SPECIFIC OPTIMIZATION** (sm_80+). Two things to notice:

- `cg::reduce` lowers to `REDUX` **by itself**. Cooperative groups is not a
  portability tax here; it is the only one of the three that picks the best
  available instruction for you. For `float`, where there is no hardware
  reduction, `cg::reduce` emits five `SHFL.BFLY` instead.
- **The timer cannot tell them apart.** Measured on 2^26 unsigned ints, all
  three within 0.03%:

  ```
  __shfl_down_sync x5      0.7223 ms   371.7 GB/s
  __reduce_add_sync        0.7221 ms   371.7 GB/s
  cg::reduce(plus<u32>)    0.7223 ms   371.7 GB/s
  ```

  Of course they are. Five instructions against the roughly 2^19 loads a block
  issues, in a kernel that is waiting on DRAM. Use `REDUX` because it is correct
  and shorter, not because you measured a win; you will not measure one.

Module 30 owns warp intrinsics as a topic. Module 29 owns cooperative groups in
depth; Module 9 introduced `tiled_partition` and `thread_rank`.

### What you would actually ship: CUB

```cpp
#include <cub/cub.cuh>

// block level
typedef cub::BlockReduce<float, 256> BR;
__shared__ typename BR::TempStorage temp;
float blockSum = BR(temp).Sum(myValue);

// device level
void* d_temp = nullptr; size_t tempBytes = 0;
cub::DeviceReduce::Sum(d_temp, tempBytes, d_in, d_out, n);   // query
cudaMalloc(&d_temp, tempBytes);
cub::DeviceReduce::Sum(d_temp, tempBytes, d_in, d_out, n);   // run
```

The two-call temp-storage protocol is CUB's convention everywhere. In CUDA 13.2
the headers live under `include/cccl/cub` and are on the default include path;
on MSVC you need `-std=c++17 -Xcompiler /Zc:preprocessor`, the same pair
Module 9 needed for `<cuda/atomic>`.

Measured, `n = 2^26`:

| implementation | ms | GB/s |
|---|---|---|
| hand-written v6 + finalize | 0.7283 | 368.6 |
| `cub::BlockReduce` + finalize | 0.7293 | 368.1 |
| `cub::DeviceReduce::Sum` | 0.7248 | 370.3 |

`cub::DeviceReduce::Sum` wanted 5119 bytes of temp storage and beat the
hand-written version by 0.5%, which is noise. **That is the expected result and
it is the point of the ceiling argument**: a bandwidth-bound kernel has a hard
limit, a good library reaches it, and nothing — yours or theirs — can go past it.
The reason to use CUB is not speed; it is that `DeviceReduce` handles every
`n`, every type, every operator, and every architecture, and you do not have to
re-derive the ladder. Module 36 covers CUB and Thrust properly.

(One incidental observation from the same run: `cub::DeviceReduce` returned
3439646208.0 against the exact 3439646263.9, while every version in this module
returned 3439644160.0. CUB's decomposition is different from ours and, on this
data, slightly more accurate. Different decomposition, different bracketing,
different bits — the determinism section again.)

---

## Hardware Mental Model

### Why the tree is free

A block of 256 threads reducing 512 elements performs 511 additions and issues
512 loads. At the DRAM limit, 512 floats take `2048 / 375e9 = 5.5 ns` to arrive.
In 5.5 ns at 1.8 GHz, one SM's four warp schedulers can issue about 40
warp-instructions — roughly 1280 lane-operations. The tree needs 511. The
arithmetic fits in the shadow of the load with room to spare, which is the formal
statement of "the addition is free".

So what were v1, v2 and v3 spending their time on, if not arithmetic? Issue
slots and barriers.

- **v1's divergence costs issue slots, not lanes** (Module 8). An `if` that
  splits a warp issues both sides. Eight tree levels × 8 warps × a divergent
  predicate each is a large multiple of the 40 instruction slots the memory
  system was going to give you for free.
- **v1 and v2 and v3 all pay 8 `BAR.SYNC` per 256 (or 512) elements.** Module 9:
  a barrier costs the idle time of the warps that arrive early, and the block
  runs at the speed of its slowest warp at every one. Eight of them per 1 KB of
  data is an enormous convoy tax.
- **v1–v3 launch 262,144 blocks.** Module 1: blocks are placed by the GigaThread
  engine, gated on resources, and run to completion. At 6 blocks/SM × 40 SMs, the
  machine can hold 240 at a time, so this is 1092 waves deep. Each block's
  lifetime contains a prologue, eight barriers and an epilogue for 512 floats of
  work. v6 launches 240 blocks that each live for the entire kernel.

### Where the memory traffic goes, and why v4 is the big step

Count the DRAM traffic of each version. All of them read `4n` bytes of input. The
difference is the *partial* array:

| version | partial floats written | extra bytes at `n = 2^26` |
|---|---|---|
| v1–v3 | 262144 | 1.0 MB (0.4% of the input) |
| v4–v5 | 131072 | 0.5 MB |
| v6 | 240 | 960 B |

So it is not traffic either. v4's 1.99× comes from somewhere else entirely:
**half of v3's threads never issue a load**. With one element per thread, the
block's 256 loads are issued once and then the block spends eight barrier-
separated rounds doing nothing that touches memory. With two elements per
thread, the same block issues 512 loads, and crucially issues them as **two
independent loads per thread** — Little's Law (Module 1) again: the memory system
needs many outstanding requests to stay saturated, and a thread with two
independent loads in flight supplies twice the concurrency of a thread with one.
v6 takes this further: its inner loop has two independent loads per iteration
and hundreds of iterations, so every thread keeps the pipe full for the whole
kernel.

### Where the atomic executes

Module 10 established it and it is worth restating in this context: a device-scope
atomic is *shipped to* the L2 and performed there, because the L2 is the first
level all SMs share. `atomicAdd(out, blockSum)` from 240 blocks is 240 atomic
operations on one address over 0.73 ms. Module 10's contention curve prices
same-address atomics at up to 48× a plain store — and 240 × 48 store-costs is
still invisible next to 268 MB of DRAM traffic. The atomic is not the problem
with strategy (b). Determinism is.

### The benchmarking result this module adds

Spec §12 already required a duration-based warm-up, iteration counts auto-scaled
to ~10 ms segments, back-to-back timing, rotated sweep order, and min-of-N.
Building this module added one more finding, and it is worth recording because it
changes absolute numbers by 10%:

> **A 400 ms warm-up ramps the SM clock but not the memory P-state.** With the
> 400 ms warm-up established in Modules 6–8, `example01.cu` measured its
> streaming ceiling at 372–373 GB/s. With a 1500 ms warm-up, the same kernel on
> the same data measured **410.7 GB/s** — 95.1% of the 432.0 GB/s pin peak, the
> highest streaming figure recorded anywhere in this course. Module 5 documented
> the P-state ladder (6001 → 8001 → 9001 MHz, i.e. 288 → 384 → 432 GB/s); a
> read-only stream apparently needs more than 400 ms of sustained traffic to be
> promoted to the top rung, and a laptop power cap can demote it again.

The practical rule is the one spec §12 rule 5 already gives: **report ratios and
percentages of a ceiling measured in the same sweep.** Across three runs of
`example01.cu` in different thermal states, the absolute GB/s of every version
moved by 10% together and the `% of ceiling` column moved by at most 5 points.
Do not quote an absolute bandwidth from this machine without saying what the
ceiling measured in the same breath.

---

## Code Walkthrough

### `example01.cu` — the ladder

```
nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
.\example01.exe
```

All six versions plus the streaming ceiling are timed **in one rotated sweep**
(`v = (q + sweep) % 7`), with iteration counts auto-scaled to ~10 ms segments,
minimum of four sweeps, and validation in a separate pass afterwards. Because the
ceiling is one of the seven configurations, it carries exactly the same clock
history as the versions it is the denominator for.

The full output on this GPU:

```
blocks/SM for v6 (occupancy API): 6  ->  grid = 240 blocks
v1-v3 launch 262144 blocks; v4-v5 launch 131072 blocks.

Measured streaming ceiling (read-only, no tree): 0.6536 ms = 410.7 GB/s (95.1% of the 432.0 GB/s pin peak)

version                              ms      GB/s  %ceiling     %peak     vs v1
---------------------------------------------------------------------------------
v1 interleaved, tid mod 2s       2.6452     101.5     24.7%     23.5%     1.00x
v2 contiguous index 2*s*tid      1.8542     144.8     35.2%     33.5%     1.43x
v3 sequential addressing         1.7167     156.4     38.1%     36.2%     1.54x
v4 first add during load         0.8631     311.0     75.7%     72.0%     3.06x
v5 warp tail, __shfl_down        0.6568     408.7     99.5%     94.6%     4.03x
v6 grid-stride + unrolled        0.6580     407.9     99.3%     94.4%     4.02x
   streaming ceiling (no tree)   0.6536     410.7    100.0%     95.1%     4.05x
```

and the validation pass, which is the accuracy section in miniature:

```
double reference = 33554158.000000
  v1 ... v3  got 33554148.000000   rel.err 2.980e-07
  v4, v5     got 33554154.000000   rel.err 1.192e-07
  v6         got 33554158.000000   rel.err 0.000e+00
```

Three different answers from six kernels that all "sum the same array". v1–v3
share a bracketing, v4–v5 share another, and v6's grid-stride register
accumulation is a third — and the third one happens to be exact to the last bit
of the double reference. Nothing is wrong. Read the determinism section again.

`streamCeiling` is worth reading, because measuring a ceiling badly is the usual
way these arguments go wrong:

```cpp
float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
for (; i + 3 * step < n; i += 4 * step) {
    a0 += in[i];  a1 += in[i + step];
    a2 += in[i + 2*step]; a3 += in[i + 3*step];
}
float s = (a0 + a1) + (a2 + a3);
if (s == 1.2345e-30f) out[blockIdx.x] = s;   // never true; keeps the loads live
```

Four independent accumulators so the FP add latency chain never becomes the
bottleneck, and a store the compiler cannot prove is dead so the loads survive
`-O3`. If you write `if (false) out[0] = s;` the compiler deletes the whole loop
and you measure 40,000 GB/s.

### `example02.cu` — everything after the block sum

```
nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler /Zc:preprocessor -o example02.exe example02.cu
.\example02.exe
```

Six parts: the three finalization strategies timed against a ceiling in the same
rotated sweep (A), the determinism measurements (B), the accuracy experiment (C),
the three warp-tail spellings with their SASS (D), CUB (E), and the legacy
`volatile` tail (F). The measured numbers from all six are quoted in the Concept
section above.

Two implementation details worth pointing at.

`lastBlockFinal` puts the `__syncthreads()` **after** thread 0 sets `amLast` and
**outside** the `if (amLast)`:

```cpp
if (threadIdx.x == 0) { ...; amLast = (ticket == gridDim.x - 1); }
__syncthreads();
if (amLast) { ... }
```

`amLast` lives in shared memory, so the barrier is supplying both of Module 9's
guarantees: G1 so that the other 255 threads do not read it before thread 0
writes it, and G2 so that what they read is what was written. The `if (amLast)`
that follows contains barriers of its own and that is legal, because `amLast` is
**block-uniform** — every thread of the block got the same value out of shared
memory. Module 9's uniformity rule is about the block, not about the syntax.

Part F prints:

```
legacy `volatile __shared__` tail: 0 of 5 runs wrong
```

Zero. Read the paragraph above about the dangerous middle case before you draw
any conclusion from that, and then go and read the SASS.

---

## Check Your Understanding

Answers in `solutions/module12/check_your_understanding.md`.

**Q1.** Version 3 of the ladder is conflict-free and non-divergent and runs at
38% of the streaming ceiling. Version 4 changes exactly two things — it halves
the grid and has each thread add two elements during the load — and reaches 76%.
Explain the 1.99× in terms of what the memory system was doing during version
3's eight tree iterations. Then predict what happens to the ratio if version 4 is
changed to read *eight* elements per thread instead of two, and say which of
Module 1's two arguments (Little's Law, or wave/tail effects) bounds the answer
in each direction.

**Q2.** A colleague replaces the `__shfl_down_sync` tail of version 5 with the
legacy `volatile __shared__` tail, runs the harness ten times, gets the exact
same correct answer every time, and concludes that the warning about
independent thread scheduling is overblown on Ada. Give the strongest *technical*
version of their argument — what mechanism actually makes it work? — and then
give the exact reason it is not a defence. Finally: name a change to the
*surrounding* code, with no change to the tail at all, that could break it.

**Q3.** You must sum the same 2^26-element array on two machines, an RTX 3500
Ada with 40 SMs and a hypothetical part with 68 SMs, and the two results must be
bit-identical. Your kernel sizes its grid from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor × multiProcessorCount`. Explain why
the results will differ even though neither machine uses an atomic anywhere, then
give two structurally different fixes and state precisely what each one costs.
One of them costs essentially nothing; say why that is not obvious in advance.

**Q4.** Both of these compute the maximum of an array, and both are correct.
Under what circumstances is the second one *more* accurate than the first, and
under what circumstances is the question meaningless? Now replace `fmaxf` with
`+` in both and answer the same question. Your answer must distinguish the
algebraic property that makes the parallel form legal from the numerical property
that makes it better.

```cpp
// (i)  sequential
float m = x[0]; for (int i = 1; i < n; ++i) m = fmaxf(m, x[i]);
// (ii) tree
/* the ladder's version 6, with += replaced by fmaxf and 0.0f by -INFINITY */
```

---

## Exercises

All three are in `module12/`; solutions and notes in `solutions/module12/`.

### Exercise 1 — `exercise01.cu` : build the ladder

Version 1 is supplied complete. You write the key lines of versions 2 through 6,
and you commit to four predictions before compiling.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe

nvcc -arch=sm_89 -O3 -c -o exercise01.o exercise01.cu
cuobjdump -sass exercise01.o > exercise01.sass
```

- **TODO 1** — four predictions: which step is the largest, the v1/v6 ratio
  (within ±30%), the lowest version that reaches 95% of the measured ceiling, and
  whether the last barrier of version 5's shared loop can be deleted.
- **TODO 2** — version 2's index. Same tree, same additions, same order; only the
  mapping of additions to threads changes.
- **TODO 3** — version 3's loop, bound included. The bound is the trap: version 5
  below stops early on purpose and copying that bound here silently discards
  31/32 of every block's data.
- **TODO 4** — the two-element load.
- **TODO 5 (DESIGN)** — `warpReduceSum` with the right mask, version 5's tail,
  version 6's grid-stride accumulation and compile-time tree, and
  `chooseGridV6()`. The harness rejects any grid above 4096 blocks, so "one block
  per 512 elements" is not available to you.

The harness times all six plus a streaming ceiling in one rotated sweep, validates
each against a double reference, and prints GB/s and % of ceiling. `SCORE: 10/10`
(six versions + four predictions) is required for `OVERALL: PASS`.

### Exercise 2 — `exercise02.cu` : a reduction that is not a sum

Segmented reduction over 100,000 ragged rows (45.4 M elements, lengths from 8 to
200,000) with a **non-commutative** operator: for each row, the sum *and* the
length of the longest run of consecutive elements above a threshold. A correct
one-thread-per-row baseline is supplied and is slow for reasons Module 8 priced.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

- **TODO 1** — the monoid: the identity and the combine. Check the identity in
  *both* directions on paper; a wrong identity fails only on the rows where the
  padding lands.
- **TODO 2** — the 32-lane warp reduction of the monoid. Two free choices, both
  invisible for a sum, and exactly one combination of them is right.
- **TODO 3 (DESIGN)** — the whole decomposition. Order preservation, coalescing
  and a four-orders-of-magnitude length skew pull against each other, and there
  is an arrangement that satisfies all three.
- **TODO 4** — two predictions, one of which the harness tests by running a
  deliberately-strided kernel and counting the rows it gets wrong.

`OVERALL: PASS` needs every row's `best` exactly right, every row's `sum` within
1e-4, a speedup of at least 3.0×, and both predictions.

### Exercise 3 — `exercise03.cu` : make the answer the same every time

`n = 33,554,433 = 2^25 + 1`. Nothing is a multiple of anything.

Part 1 is a debugging exercise: a reduction that comes back 5.6% low, that
`compute-sanitizer` declares clean under `memcheck`, `racecheck`, `initcheck` and
`synccheck`, and that is **exact** at `n = 2^25`. Part 2 is the determinism
problem: build two bit-reproducible reductions with different trade-offs, one
that keeps float arithmetic and fixes the order, and one that abandons float
arithmetic and stops caring about order.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
compute-sanitizer --tool memcheck  .\exercise03.exe
compute-sanitizer --tool racecheck .\exercise03.exe
```

- **TODO 1** — diagnose (six candidates, two of which are true statements about
  CUDA that are not the cause here) and repair.
- **TODO 2 (DESIGN)** — a float reduction that must launch with exactly the grid
  it is handed and must nevertheless give identical bits at grids 97, 240 and
  1021.
- **TODO 3 (DESIGN)** — the fixed-point alternative, including deriving the
  overflow bound and the precision bound on the scale factor. There is a wide
  window and a wrong answer on each side.
- **TODO 4** — three predictions.

`SCORE: 8/8` required.

---

## Prediction

Commit to these in writing before you compile anything.

1. **The shape of the ladder.** Write down, for all six versions, your predicted
   percentage of the streaming ceiling. Then, before looking, answer this:
   version 2 removes divergence and version 3 removes bank conflicts. Module 8
   measured divergence at 1.96× and Module 7 measured a 32-way bank conflict at
   10–15× on an LSU-bound kernel. Predict the v2→v3 step from those numbers, and
   then explain why the number you get is wrong by a factor of five.

2. **The finalization strategies.** Three strategies — two kernels, one atomic
   per block, and a `__threadfence` last-block flag — over 240 blocks and 2^26
   elements. Rank them by speed and commit to the ratio between the fastest and
   the slowest. Separately, rank them by *determinism*, and say which of your two
   rankings you are more confident about and why.

3. **The `REDUX` instruction.** `__reduce_add_sync` replaces five shuffles and
   five adds with a single hardware instruction on sm_80+. Predict the speedup of
   a full 2^26-element reduction that uses it over one that uses shuffles, to one
   decimal place. If your answer is not 1.0, work out how many instructions the
   tail is as a fraction of the block's total, and try again.
