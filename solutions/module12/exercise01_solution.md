# Module 12 / Exercise 01 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe

nvcc -arch=sm_89 -O3 -c -o exercise01_solution.o exercise01_solution.cu
cuobjdump -sass exercise01_solution.o > exercise01_solution.sass
```

Warning-clean. `SCORE: 10/10`, `OVERALL: PASS`.

---

## TODO 1 — the four predictions

| | answer | measured |
|---|---|---|
| (a) biggest single step | **v4** | v4, 1.94–1.99× |
| (b) v1/v6 speedup | **3.8×** (anything 2.6–5.2 passes) | 3.32–4.27× |
| (c) first version ≥ 95% of ceiling | **v5** | v5 |
| (d) last barrier of v5's shared loop removable | **yes (1)** | yes |

**(a)** The reasoning that gets you there without running anything: v2 and v3
remove divergence and bank conflicts, which are *shared-memory and issue-slot*
costs, in a kernel that is spending 62% of its time waiting for DRAM it has not
managed to keep busy. v4 is the first change that alters how many loads are in
flight. Module 1's Little's Law is the whole argument: the memory system needs
concurrency, and v1–v3 issue one load per thread and then spend eight
barrier-separated rounds issuing none.

A reader who predicts v2 (because "divergence is the classic textbook fix") has
learned the ladder's story rather than its physics.

**(c)** v4 lands at 75–82% of the ceiling in every run; v5 at 96.9–99.7%. The
gate is at 95% precisely because v4 never reaches it and v5 always does. If you
predicted v6, you have not internalised that v5 is already at the wall — there is
nothing for v6 to win on a 256 MiB array.

**(d) is the subtle one, and the justification matters more than the answer.**
After the `s == 32` iteration, the surviving data is in `sdata[0..31]`. The next
thing that happens is `float w = sdata[tid]` for `tid < 32` — and thread `tid`
wrote `sdata[tid]` itself, in the `s == 32` step. There is no cross-thread
dependency left. Module 9's G1 (execution ordering between threads) and G2
(cross-thread visibility) are both vacuous when the reader and the writer are the
same thread; a thread always observes its own stores in program order.

The wrong justification, which gives the right answer for a reason that has been
false since Volta, is "the surviving threads are all in one warp and warps are in
lockstep." Module 8 established that independent thread scheduling removed that
guarantee. If the dependency *were* cross-lane — as it is in the `volatile` tail
— removing the barrier would be undefined, and you would need `__syncwarp()`
instead.

The barriers you may **not** delete are the earlier ones: after the `s == 64`
step, threads 0..31 read `sdata[tid + 32]`, which warp 1 wrote.

---

## TODO 2 — v2, contiguous-thread indexing

```cpp
for (unsigned s = 1; s < blockDim.x; s *= 2) {
    unsigned idx = 2 * s * tid;
    if (idx < blockDim.x) sdata[idx] += sdata[idx + s];
    __syncthreads();
}
```

Why it is correct: the set of `(destination, source)` pairs is identical to v1's.
At step `s`, v1 has thread `2*s*k` add `sdata[2*s*k + s]` into `sdata[2*s*k]`; v2
has thread `k` do the same addition. The additions, their operands and their
order are unchanged — which is why v1, v2 and v3 return **bit-identical**
results (`33554148.000000` in the run below) while v4 and v6 do not.

Why it is faster: the working threads are now `0 .. blockDim/(2s) - 1`, a
contiguous prefix. At `s = 1` warps 0–3 are fully active and warps 4–7 are fully
inactive; at `s = 2` warps 0–1; and so on. Module 8: a branch costs nothing extra
when every lane of the warp agrees. v1 splits every warp at every step.

**Common wrong approaches.**

- `if (idx + s < blockDim.x)` instead of `if (idx < blockDim.x)`. Also correct
  here (the two conditions coincide for power-of-two `blockDim`), and harmless.
- Writing `sdata[tid] += sdata[tid + s]` — that is v3, and it will validate. It
  also makes the v2→v3 step measure 1.00× and the reader will wonder why. The
  harness cannot catch this; only reading your own code can.
- Forgetting that `2 * s * tid` overflows nothing here but is `unsigned`
  arithmetic — for `blockDim = 1024` and `s = 512`, `2*s*tid` reaches 2^20 for
  `tid = 1023`, which is fine, and the guard rejects it.

**What it costs.** The addresses now have stride `2s`. Module 7:
`D = gcd(k, 32)` for `s[k*tid]`, so `s = 1` → 2-way, `s = 2` → 4-way, up to
`s = 16` → 32-way. On Ada the cost law is `max(2, D)`, so the first step is free
and the last four are not. That is what v3 removes.

---

## TODO 3 — v3, sequential addressing

```cpp
for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

The tree runs *downwards* — widest stride first. Every access is `sdata[tid]` or
`sdata[tid + s]`, both unit-stride across the warp, so `D = 1` at every step
(Module 7), and the live values stay packed in `sdata[0..s)`.

**The trap is the bound.** `s > 0` is correct here. `s > 32`, which is what v5
uses, terminates the loop with 64 live values in `sdata[0..63]` and then writes
`sdata[0]` — discarding 63/64 of every block's data. The result is a sum roughly
1/64 of the truth, which is loud; the harness catches it immediately. It is worth
being caught by, because the reason v5 *may* use `s > 32` (or `s >= 32`) is that
v5 replaces the missing iterations with something else, and v3 does not.

The other bound error, `s >= 1` written as `s > 1`, drops exactly the last
addition and returns half the block's sum. Also caught.

**Note what the measurement says**: v3 is only 1.03–1.18× faster than v2. Both
optimizations — divergence removal and conflict removal — are correct, textbook,
and together worth 1.45×. The kernel is still at 38% of the ceiling because
neither of them touched the binding constraint.

---

## TODO 4 — the two-element load

```cpp
__device__ __forceinline__ float loadTwoAndAdd(const float* __restrict__ in,
                                               long long n, unsigned tid)
{
    long long i = (long long)blockIdx.x * (blockDim.x * 2) + tid;
    float v = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    return v;
}
```

Two independent guards, not one. The all-or-nothing form

```cpp
float v = 0.0f;
if (i + blockDim.x < n) v = in[i] + in[i + blockDim.x];   // WRONG
```

silently drops every element in the last partial block that has no partner. It
is the defect Exercise 3 is built around, and at `n = 2^26` — which is an exact
multiple of everything — it produces the correct answer, so the harness here will
not catch it. Write the two guards anyway.

Why it is the biggest step in the ladder: see the solution note for TODO 1(a).
The first tree level now happens in registers with every thread participating,
the grid halves from 262,144 blocks to 131,072, and — the part that matters most
— each thread has **two independent loads in flight** instead of one.

---

## TODO 5 — the warp tail, v6, and the grid

### 5a `warpReduceSum`

```cpp
__device__ __forceinline__ float warpReduceSum(float v)
{
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;                      // valid in lane 0 only
}
```

**The mask.** `0xffffffff` is correct because all 32 lanes of warp 0 enter the
`if (tid < 32)` region. Module 8: on sm_70+ the mask is how you *create* the
convergence the hardware no longer gives you, so it must name the lanes your
algorithm requires, not the lanes the scheduler happens to have. Using
`__activemask()` is wrong even when it returns `0xffffffff` — it is a report on
the past, and the compiler is entitled to have predicated the `VOTE` that
implements it.

**Which lane is valid.** `__shfl_down_sync(mask, v, d)` gives lane `L` the value
held by lane `L + d`; when `L + d >= 32` lane `L` gets its own value back
unchanged. So after the five steps lane 0 holds the full sum, lane 1 holds the
sum of lanes 1..31 plus some double-counting, and so on. Writing from lane 31, or
from `__shfl_sync(mask, v, 0)` broadcast to everybody and then storing from every
lane, are both bugs; the second one is a 32-way race on the same address.

**No barrier is needed between the steps.** `__shfl_*_sync` carries its own
synchronization for the named lanes — that is what the `_sync` suffix means
(Module 8, Module 9). Adding `__syncwarp()` between them is harmless and
pointless.

### 5b — v5's tail

```cpp
if (tid < 32) {
    float w = warpReduceSum(sdata[tid]);
    if (tid == 0) out[blockIdx.x] = w;
}
```

### 5c — v6

The kernel is quoted in full in `lesson.md`. The three requirements and how they
are met:

```cpp
static int chooseGridV6(void)
{
    int sm = 0, bpsm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, reduce6<BS>, BS, 0);
    return sm * bpsm;                 // 40 * 6 = 240
}
```

- **Grid independent of `n`**: one occupancy-limited wave. The harness rejects
  anything above 4096 blocks, which closes off `n / (BS*2)` — the reader who
  writes that gets a kernel that is correct, exactly as fast as v5, and told off.
- **Register accumulation before shared memory**: the `while` loop. Note the
  **two** loops — the paired loop and then a ragged tail loop for the elements
  whose partner is off the end. Omitting the second one is Exercise 3's bug.
- **Compile-time tree**: `BLOCK` is a template parameter, so `if (BLOCK >= 512)`
  is resolved at compile time. Confirm with `cuobjdump -sass`: there is no loop
  and no `ISETP` against a runtime block size in the tree region, just a straight
  run of predicated `LDS`/`STS` pairs and `BAR.SYNC`s, followed by five
  `SHFL.DOWN`.

**Common wrong approaches for 5c.**

- A grid-stride loop with a *single* `in[i]` per iteration. Correct, and slower,
  because it halves the memory-level parallelism per iteration.
- `while (i < n) { sum += in[i] + in[i + BLOCK]; i += step; }` — reads out of
  bounds on the last iteration. `compute-sanitizer --tool memcheck` catches this
  one loudly; it is the *opposite* error to Exercise 3's.
- Using `int` for `i` and `step`. At `n = 2^26` this is still fine; at 2^31 it is
  not, and it is the kind of thing that works for a year.

---

## Synchronization / memory reasoning

The barrier inventory per block, which is the thing the ladder is really
optimizing:

| version | barriers per block | elements per block | barriers per 1024 elements |
|---|---|---|---|
| v1–v3 | 8 | 256 | 32 |
| v4 | 8 | 512 | 16 |
| v5 | 3 | 512 | 6 |
| v6 | 4 | ~280,000 | 0.015 |

Module 9: a barrier costs the idle time of the warps that arrive early, and a
block runs at the speed of its slowest warp at every one. Versions 1–3 pay that
32 times per kilobyte of input.

Every barrier in every version is outside divergent control flow. The
`if (tid < s)` bodies contain no barrier; the barrier is at the bottom of the
loop, and the loop bound depends only on `blockDim`, which is block-uniform.
Moving the `__syncthreads()` inside the `if` is the classic way to turn this
kernel into Module 9's case (a) — undefined, and on sm_89 usually silently wrong
rather than hanging.

---

## Performance reasoning

Full output, one representative run (RTX 3500 Ada, CUDA 13.2, warm):

```
Module 12 exercise 01 -- the reduction ladder
40 SMs, N = 67108864 floats = 256 MiB, block = 256 threads
v6 grid = 240 blocks

version                              ms      GB/s  %ceiling      step   result
-------------------------------------------------------------------------------
v1 interleaved, tid mod 2s       2.6625     100.8     24.6%     1.00x       ok
v2 contiguous index              1.8642     144.0     35.1%     1.43x       ok
v3 sequential addressing         1.7141     156.6     38.2%     1.09x       ok
v4 first add during load         0.8681     309.2     75.3%     1.97x       ok
v5 warp tail (shuffles)          0.6583     407.8     99.3%     1.32x       ok
v6 grid-stride + unrolled        0.6581     407.9     99.4%     1.00x       ok
   streaming ceiling             0.6540     410.5    100.0%         -        -

ceiling 410.5 GB/s (95.0% of the 432.0 GB/s pin peak)
double reference = 33554158.000000
  v1 interleaved, tid mod 2s     got 33554148.000000  rel.err 2.980e-07
  v2 contiguous index            got 33554148.000000  rel.err 2.980e-07
  v3 sequential addressing       got 33554148.000000  rel.err 2.980e-07
  v4 first add during load       got 33554154.000000  rel.err 1.192e-07
  v5 warp tail (shuffles)        got 33554154.000000  rel.err 1.192e-07
  v6 grid-stride + unrolled      got 33554158.000000  rel.err 0.000e+00

predictions:
  (a) biggest single step   predicted v4, measured v4 (1.97x)  MATCH
  (b) v1/v6 speedup         predicted 3.80x, measured 4.05x       MATCH
  (c) first version >= 95%  predicted v5, measured v5            MATCH
  (d) last barrier of v5's shared loop removable: you said yes  MATCH

SCORE: 10/10   (6/6 versions correct, 4/4 predictions)

OVERALL: PASS
```

**Run-to-run variation.** Three runs in different thermal states gave ceilings of
372.2, 372.7 and 410.5 GB/s. Every version's absolute GB/s moved with it; the
`%ceiling` column moved by at most 5 points and the step ratios by less than 0.1.
Observed ranges:

| | v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|---|
| % of ceiling | 24.6–29.5 | 35.1–36.2 | 38.1–42.8 | 75.3–81.6 | 96.9–99.7 | 99.3–99.6 |
| step vs previous | — | 1.23–1.46 | 1.03–1.18 | 1.93–1.99 | 1.14–1.32 | 0.99–1.01 |

If your absolute numbers are 30% below these, your GPU is power-capped; run
`nvidia-smi --query-gpu=clocks.mem,clocks_throttle_reasons.active --format=csv`
and look for memory below 9001 MHz or a `0x4` throttle reason. The ratios are
still valid.

**The three different answers.** v1–v3 agree bit-for-bit with each other (same
bracketing), v4–v5 agree with each other, and v6 is different again — and v6
happens to match the double reference exactly. This is not an accuracy ranking;
it is the determinism lesson, and Exercise 3 develops it.

---

## Expected output

As pasted above. `SCORE: 10/10`, `OVERALL: PASS`. The `iterations per timed
segment` line is not printed by this exercise (it is in `example01.cu`); the
harness auto-scales to ~10 ms and clamps at 20.

---

## The result that matters

The two optimizations everybody knows — remove the divergence, remove the bank
conflicts — bought 1.45× on a kernel that was wasting 75% of the memory system,
and the optimization nobody puts first, *give each thread more than one element
to load*, bought 1.99× on its own and took the kernel from 38% to 76% of the
achievable bandwidth. The lesson is not that divergence and bank conflicts do not
matter; it is that **you cannot rank optimizations without knowing which resource
is saturated**, and for a reduction the answer is always the memory system.
Version 6 is not fast. It has stopped being wasteful, which is a different and
more final thing: there is no version 7.

**Variation to try.** Change `BS` from 256 to 64 and to 1024 and re-run.
Version 6 barely moves; versions 1–3 move a lot, and in opposite directions,
because `BS` changes both the number of tree levels per block and the number of
blocks. Predict the direction for each before you run, then explain the one you
got wrong — there will be one.
