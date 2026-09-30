# Module 09 / Exercise 02 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

Warning-clean. Runs in about 1.1 s, almost all of it in the large-grid spin.
For the tooling section:

```
nvcc -arch=sm_89 -O3 -lineinfo -o e2li.exe exercise02.cu
compute-sanitizer --tool initcheck --initcheck-address-space shared --kernel-regex "kns=shift_tile" .\e2li.exe
compute-sanitizer --tool synccheck --kernel-regex "kns=shift_tile" .\e2li.exe
compute-sanitizer --tool racecheck --kernel-regex "kns=shift_tile" .\e2li.exe
```

---

## TODO 1 — the two predictions

```cpp
static const long long P1_small_grid_timeouts = 0;
static const long long P2_large_grid_timeouts = 8192 - 240;   // = 7952
```

**P1 = 0.** The small grid is exactly the co-residency limit, so every block —
including the producer, which is the last one — is placed at launch. The
producer raises the flag within a few hundred cycles and every consumer sees
it long before the escape counter runs out. Measured: 0 timeouts, 0.04 ms.

**P2 = gridDim − co-resident = 8192 − 240 = 7952.** This is the number you are
asked to derive rather than guess, and the derivation is the whole exercise.

The producer is the **last** block of the grid. Module 1: blocks are placed in
order as slots free up, and a block that is not placed does not exist. So block
8191 is placed last. It cannot be placed until a slot frees. A slot frees only
when a resident block retires. Every resident block is spinning. With an
unbounded spin that is a hard deadlock — nothing ever retires.

With the bounded spin the deadlock becomes a livelock with drip-feed progress:
the 240 resident blocks spin for `MAXSPIN` iterations, give up, and retire;
240 more are placed, spin, give up, retire; and so on. Every block placed
*before* the producer times out. The blocks placed at the same time as the
producer — the last wave, which is the 240 blocks including block 8191 —
succeed. Hence `8192 − 240`.

Measured across runs: **7953, 7954, 7954**. The two-block drift is scheduling
jitter in which blocks happen to share the producer's wave.

Notice what you have just done: you measured `cudaOccupancyMaxActiveBlocksPer
Multiprocessor × multiProcessorCount` *with a stopwatch and a counter*, without
asking the API. The program prints both so you can see they agree.

## TODO 2 — the co-residency number, from the API

```cpp
static int max_resident_blocks(void)
{
    int sms = 0, perSM = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess)
        return 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &perSM, (const void*)scale_spin, TPB, 0) != cudaSuccess)
        return 0;
    return sms * perSM;
}
```

Observed: 40 SMs × 6 blocks/SM = **240**.

**Why it must be computed per kernel, not hard-coded.** `cudaDevAttrMaxBlocks
PerMultiprocessor` on sm_89 is 24, and 1536/256 = 6 threads-wise, but neither
of those is the answer in general — the occupancy API folds in the kernel's
register count and its static plus dynamic shared memory. `scale_spin` uses one
shared int and few registers, so here the limit is the thread budget
(1536 threads/SM ÷ 256 = 6). Add 40 KB of shared memory to the kernel and the
same call returns 2, and the deadlock threshold moves with it. Any code that
reasons about co-residency and hard-codes a number is wrong the first time
someone edits the kernel.

**Common wrong approaches.** Using `cudaDevAttrMaxThreadsPerMultiProcessor /
blockDim` ignores registers and shared memory and over-estimates for any real
kernel. Using `cudaDevAttrMaxBlocksPerMultiprocessor` (24) over-estimates by
4×. Both give a "safe grid size" that is not safe.

## TODO 3 — the tiled shift

Shipped (broken):

```cpp
if (gid < n) {
    s[t] = in[gid];
    __syncthreads();                       // <-- inside the guard
    out[gid] = s[(t + SHIFT) & (TPB - 1)];
}
```

Fixed:

```cpp
s[t] = (gid < n) ? in[gid] : 0.0f;
__syncthreads();                           // reached by the whole block
if (gid < n) out[gid] = s[(t + SHIFT) & (TPB - 1)];
```

**Why the shipped version is wrong.** The guard is doing two jobs. One is
legitimate: keep the global load and store in range. The other is not: it
decides which threads reach the barrier. `n = 8192*256 − 37`, so in block 8191
threads 219..255 fail the guard and do not reach the barrier — a non-uniform
condition around `__syncthreads()`, which is undefined by the Programming Guide
rule. Concretely, those 37 threads also never write `s[219..255]`, so the
threads that read those slots (via the `+96` rotate) read uninitialised shared
memory. Measured: **37 wrong elements, all in block 8191**, `cudaSuccess`
everywhere, no crash.

**Why the fix is correct.** Separate the two jobs. Every thread of the block
stages *something* — the real value if in range, a neutral 0 if not — so the
store is unconditional and every thread reaches the barrier. The guard now
covers only the global accesses. The reference semantics happen to want 0 for
out-of-range sources, so the neutral value is the answer directly; if it were
not, you would guard the read of `s` instead, but you would still stage and
still barrier unconditionally.

**Common wrong approaches.**

- *`if (gid < n) { s[t] = in[gid]; } __syncthreads(); if (gid < n) {...}`* —
  correct about the barrier, but `s[219..255]` is now left uninitialised rather
  than zeroed, so the answer is still wrong (and non-deterministic) for the
  threads that read those slots. This is the version `initcheck` catches; see
  below.
- *Rounding `n` up and over-allocating the input.* Changes the problem.
- *Launching `ceil(n/TPB)` blocks with the last one smaller.* `blockDim` is
  fixed per launch; you cannot make one block smaller. And even if you could,
  the barrier would then be waiting on a differently-sized block, which is
  precisely what makes a partial tile awkward in the first place.

## TODO 4 — no cross-block synchronization at all

```cpp
__global__ void produce_scale(float* scale)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) *scale = SCALE;
}

__global__ void consume_scale(const float* __restrict__ in,
                              float* __restrict__ out,
                              const float* __restrict__ scale, int n)
{
    const float s   = *scale;
    const int   gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = in[gid] * s;
}

static int apply_scale_any_grid(const float* d_in, float* d_out,
                                float* d_scale, int n, int grid)
{
    produce_scale<<<1, 32>>>(d_scale);
    consume_scale<<<grid, TPB>>>(d_in, d_out, d_scale, n);
    return 1;
}
```

**Why it is correct, from the hardware model.** The kernel boundary is the only
grid-wide ordering primitive that holds for an arbitrary grid. Every block of
`produce_scale` has completed and every memory operation it performed is
visible device-wide before any block of `consume_scale` starts. There is no
assumption about co-residency because there is no waiting: `produce_scale`'s
one block retires, and only then is `consume_scale` launched. Note there is no
`__threadfence()` anywhere and none is needed — the flush is part of kernel
completion.

Cost: one extra launch, single-digit microseconds. The whole of Part 3 --
allocation-free, two launches, a 2 M-element device-to-host copy and a full
host-side comparison -- finishes visibly instantly against the spin version's
~950 ms for the same result. The "optimization" of fusing them into one kernel
with a flag is a 1000× regression and a deadlock waiting for a bigger grid.

**Common wrong approaches.**

- *Keeping the spin but using `cudaLaunchCooperativeKernel` and `grid.sync()`.*
  Correct, but it caps the grid at 240 blocks, which means the kernel must be
  rewritten as a grid-stride loop to cover 2 M elements, and it is not portable
  to devices where `cudaDevAttrCooperativeLaunch` is 0. The exercise explicitly
  excludes it. Module 29 covers it.
- *Keeping the spin but making the producer block 0 instead of the last block.*
  This "works" on this GPU today because block 0 is placed first, and it is one
  of the worst bugs in this module: it is a correctness argument that depends
  entirely on the placement order of an unspecified scheduler. Change the grid
  shape, add shared memory, run under MPS, or move to a GPU where the
  distribution across SMs differs, and it stops working. There is no documented
  guarantee that block 0 is scheduled first.
- *Computing the scale redundantly in every block.* Legitimate and often the
  right engineering answer when the producer work is cheap — it trades
  arithmetic for synchronization, which is a good trade on a GPU. It does not
  generalise when the producer phase is a reduction over the whole input, which
  is the case this pattern usually stands in for.

---

## Synchronization / memory reasoning

Three distinct mechanisms appear in this file and it is worth naming which is
which.

1. **`__threadfence()` in `scale_spin`** — device scope, on the producer side,
   between writing `*scale` and raising the flag. This is the publish half of
   publish/subscribe. It has to be device-scope, not block-scope, because the
   consumer is in a *different block* and therefore potentially on a different
   SM, and Module 4 established that L1 is not coherent across SMs. Device
   scope means "visible at the L2."
2. **`__threadfence()` on the consumer side**, after the spin succeeds, before
   reading `*scale`. This is the subscribe half: it prevents the read of
   `*scale` from being satisfied by anything the thread had before it observed
   the flag.
3. **`__syncthreads()` after the spin** — thread 0 of each block does the
   spinning and stores the outcome in a shared variable; the barrier
   distributes that to the other 255 threads. The condition is block-uniform
   (`threadIdx.x == 0` guards the *work*, not the barrier), so it is well
   defined.

All three are correct, and the kernel still deadlocks. That is the lesson: the
memory model was never the problem. **No amount of correct memory ordering can
fix a scheduling impossibility.**

---

## What the sanitizers actually reported

`memcheck` — nothing, on either kernel. There is no illegal access here.

`synccheck` on the broken `shift_tile` — **nothing**:

```
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
```

`racecheck` on the broken `shift_tile` — **nothing**:

```
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

That is correct behaviour on racecheck's part, not a miss: the 37 threads never
write those shared slots at all, so there is no *race* — there is an
uninitialised read. The tool that finds it is `initcheck` with the shared
address space enabled, which is not its default:

```
========= Uninitialized __shared__ memory read of size 4 bytes
=========     at shift_tile(const float *, float *, int)+0x100 in exercise02.cu:137
=========     by thread (123,0,0) in block (8191,0,0)
=========     Address 0x36c
========= Uninitialized __shared__ memory read of size 4 bytes
=========     at shift_tile(const float *, float *, int)+0x100 in exercise02.cu:137
=========     by thread (124,0,0) in block (8191,0,0)
=========     Address 0x370
```

Exactly the right threads (219..255 rotate to readers 123..159) and exactly the
right block (8191, the partial one). Note it must be spelled
`--initcheck-address-space shared`; the default is `global` and reports nothing
here.

**Honest summary of the tooling for this module:** synccheck, the tool
documented for divergent-barrier detection, reported zero errors for every
divergent-barrier construction in Modules 9's exercises. racecheck found the
data races. initcheck (shared) found the uninitialised-shared-slot variant.
Nothing found the deadlock — for that, the tool is `nvidia-smi` showing 100%
utilisation and a process that will not exit.

---

## Performance reasoning

The large-grid spin is not slow because spinning is slow. It is slow because
`MAXSPIN × (gridDim / resident)` is a fixed amount of wasted work:
150 000 iterations × ⌈8192/240⌉ ≈ 34 waves ≈ 5.1 M dependent L2 round-trips.
Measured 880 – 961 ms across runs, i.e. ~190 ns per spin iteration per wave,
which is consistent with 240 blocks hammering one L2 line.

The ratio that matters is the one the program prints:
`observed timeouts / co-resident blocks = 33.14`, which is `8192/240 − 1` to
two digits. Ratios are the stable quantity here; the absolute millisecond
figure moves with the clock state.

---

## Expected output

Observed on the RTX 3500 Ada, CUDA 13.2. The large-grid wall time varies with
clock state, roughly 880–970 ms; the timeout count varies by ±2.

```
=== Part 0: what the hardware says ===
  SMs = 40, blocks/SM for scale_spin at 256 threads = 6
  co-resident blocks (API)      : 240
  your max_resident_blocks()    : 240  -> correct

=== Part 1: cross-block spin ===
  grid =   240 blocks ...     0.04 ms,      0 blocks timed out, 0 wrong elems
  grid =  8192 blocks ...   961.04 ms,   7953 blocks timed out, 2033883 wrong elems
  P1 predicted 0, observed 0 -> correct
  P2 predicted 7952, observed 7953 (+/-5% accepted) -> correct
  ratio observed-timeouts / co-resident-blocks = 33.14

=== Part 2: tiled shift with a bounds guard ===
  wrong elems total 0, of which in the last block 0 -> PASS

=== Part 3: no cross-block synchronization at all ===
  wrong elems: 0 -> PASS

OVERALL: PASS
```

With TODO 3 left as shipped, Part 2 reads
`wrong elems total 37, of which in the last block 37 -> FAIL`.

---

## The result that matters

You predicted a number — 7952 — from two facts about how a GPU places blocks,
and a stopwatch confirmed it to within two blocks. That number is the reason
cross-block spin-waiting is not a technique with caveats but a technique that
does not exist. The resource a spinning block consumes is exactly the resource
the block it is waiting for needs, so waiting makes the thing you are waiting
for less likely to happen. Nothing in the memory model can repair that, which
is why the fix in TODO 4 is not a synchronization primitive at all — it is a
second `<<<>>>`.

**Variation to try:** change the producer from `gridDim.x - 1` to
`gridDim.x / 2` and re-run. The timeout count should drop to
`gridDim/2 - resident`, because only the blocks placed before the producer
waste their timeslice. Predict the number first; the measured value is **3857**
against a prediction of 4096 - 240 = 3856. Then set the producer to block 0 and watch it
succeed — and write down, in one sentence, why that success is not a
correctness argument.
