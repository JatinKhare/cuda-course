# Module 12 / Exercise 03 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe

compute-sanitizer --tool memcheck   .\exercise03_solution.exe
compute-sanitizer --tool racecheck  .\exercise03_solution.exe
compute-sanitizer --tool initcheck  .\exercise03_solution.exe
compute-sanitizer --tool synccheck  .\exercise03_solution.exe
```

Warning-clean. `SCORE: 8/8`, `OVERALL: PASS`.

---

## TODO 1a — the diagnosis

**`DIAG_CODE = 3`: the loop condition silently discards elements that have no
partner.**

```cpp
while (i + BS < n) {
    sum += in[i] + in[i + BS];
    i   += step;
}
// and then nothing
```

Work out precisely which elements are lost, because "add a guard somewhere" is
not a diagnosis.

Thread `t` of block `b` starts at `i0 = b*512 + t` with `t` in `[0,256)`, and
steps by `512*gridDim`. So the set of *first* indices, across every thread of
every block, is exactly `{ i : i mod 512 ∈ [0,256) }`, and each such `i` also
covers its partner `i + 256`, whose residue is in `[256,512)`. The two sets are
disjoint and their union is every index — so the pairing is a perfect matching
and nothing is double-counted. The only indices that fall out are the first
indices `i` for which `i < n` but `i + 256 >= n`, i.e. `i ∈ [n-256, n)` with
`i mod 512 ∈ [0,256)`.

For `n = 2^25 + 1 = 33554433`: `n - 1 = 33554432 = 65536 × 512`, so `n-1` has
residue 0, which *is* in `[0,256)`. Exactly **one** element is lost — the last
one. It happens to hold 1,000,000 out of a total of 17,776,946, which is the
5.6%.

For `n = 2^25 = 33554432`: the candidate range `[n-256, n)` has residues
`256 … 511`, none of which is in `[0,256)`. **Zero** elements are lost and the
kernel is exact. The harness prints both:

```
  reduceTreeBroken        16776946.0000   (short by 5.6253%)
  same kernel at n = 2^25 16776946.0000   (reference 16776946.2500, off by 1.490e-08)
```

This is the characteristic signature of a missing ragged tail: the defect is a
function of `n mod (2 × blockDim)`, it is invisible for the power-of-two sizes
everybody tests with, and its magnitude depends entirely on what happens to be
stored in the last few hundred elements.

**Why the other five are wrong, and two of them are worth being tempted by.**

| code | claim | verdict |
|---|---|---|
| 1 | a `__syncthreads()` is missing from `blockTree` | False. There are three, and they bracket the three cross-thread steps correctly. `racecheck` confirms: 0 hazards. |
| 2 | the grid is too small, so some blocks never run | Confused. There is no "block that never runs" — a grid-stride kernel covers the array with whatever grid it is given, and the harness runs it at 240 and 977 with the same shortfall. |
| 3 | the loop condition discards unpartnered elements | **True.** |
| 4 | float addition is not associative | **True as a statement, and not the cause.** This is the tempting one. A float tree over 2^25 elements has a worst-case relative error around `log2(n) * 2^-24 ≈ 1.5e-6`. The observed error is 5.6e-2, four and a half orders of magnitude larger. Non-associativity is real, it is the subject of part 2, and it cannot produce this. |
| 5 | bank conflicts corrupt the tree | **False, and a category error.** Bank conflicts are a *performance* phenomenon — replays — with no correctness component at all (Module 7). Worth being able to reject instantly. |
| 6 | the warp tail passes the wrong mask | False. `0xffffffff` is correct: all 32 lanes of warp 0 enter `if (tid < 32)`. A wrong mask here would be undefined behaviour, not a 5.6% shortfall. |

**What the tools say.** All four `compute-sanitizer` tools are clean on the
broken kernel:

```
== memcheck ==   ERROR SUMMARY: 1 error
                 (a benign "Resetting device while there are still other users"
                  API warning from cudaDeviceReset; no memory errors)
== racecheck ==  RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
== initcheck ==  ERROR SUMMARY: 0 errors
== synccheck ==  ERROR SUMMARY: 0 errors
```

Of course they are. Nothing is out of bounds, nothing is uninitialised, nothing
races, and no barrier is divergent. The kernel simply *does not read* an element
it should have read, and no sanitizer has ever been able to see that. This is the
class of bug the whole toolchain is blind to: **the tools check what you did,
never what you failed to do.** Module 4 made the same point about spills and
Module 9 about the tiled shift that only `initcheck --initcheck-address-space
shared` could see. Here even that does not help.

---

## TODO 1b — the repair

```cpp
__global__ void reduceTreeFixed(const float* __restrict__ in, float* partial,
                                long long n)
{
    __shared__ float sdata[BS];
    long long i    = (long long)blockIdx.x * (BS * 2) + threadIdx.x;
    long long step = (long long)BS * 2 * gridDim.x;
    float sum = 0.0f;
    while (i + BS < n) { sum += in[i] + in[i + BS]; i += step; }
    while (i < n)      { sum += in[i];              i += step; }   // the tail
    float w = blockTree(sum, sdata);
    if (threadIdx.x == 0) partial[blockIdx.x] = w;
}
```

The second loop is the repair. It runs at most once per thread (after it, `i`
has advanced by a full grid stride and is past `n`), and it adds the element
whose partner does not exist.

**Common wrong approaches.**

- `while (i < n) { sum += in[i] + in[i + BS]; i += step; }` — the other
  direction. Reads `in[i + BS]` out of bounds on the last iteration. This is the
  loud version: `memcheck` reports it immediately with an exact address. Prefer a
  bug your tools can see.
- `while (i + BS < n) {…} if (i < n) sum += in[i];` — correct, and equivalent,
  because the tail can only fire once. Fine.
- A guard of the form `(i < n ? in[i] : 0) + (i+BS < n ? in[i+BS] : 0)` inside a
  `while (i < n)` loop — also correct, and it costs two predicates per iteration
  on the hot path rather than zero. Measurably slower; not enough to matter here.
- Special-casing the last block. There is no last block in a grid-stride kernel.

The harness deliberately re-runs the repair at `grid = 977` and at `n = 2^25` to
catch a fix tuned to one configuration.

---

## TODO 2 — a grid-independent float reduction

```cpp
#define NCHUNK 2048

__global__ void detPass1(const float* __restrict__ in, long long n, float* partial)
{
    __shared__ float sdata[BS];
    long long L = (n + NCHUNK - 1) / NCHUNK;        // chunk size: f(n) only
    for (int c = (int)blockIdx.x; c < NCHUNK; c += (int)gridDim.x) {
        long long a = (long long)c * L;
        long long b = a + L; if (b > n) b = n;
        float sum = 0.0f;
        for (long long i = a + threadIdx.x; i < b; i += BS) sum += in[i];
        float w = blockTree(sum, sdata);
        if (threadIdx.x == 0) partial[c] = w;
        __syncthreads();      // WAR: sdata is reused on the next chunk
    }
}

static int launchDeterministic(const float* d_in, long long n, int gridHint,
                               float* d_partial, float* d_out)
{
    detPass1<<<gridHint, BS>>>(d_in, n, d_partial);
    finalTree<<<1, 1024>>>(d_partial, d_out, NCHUNK);
    return gridHint;
}
```

**The idea in one sentence: separate the decomposition from the grid.**

The plain version fuses them — `partial[blockIdx.x]` means "the number of partial
sums equals the number of blocks", so changing the grid changes the bracketing
and therefore the bits. Here the array is cut into `NCHUNK = 2048` chunks whose
boundaries are a function of `n` alone. Chunk `c` is always summed by one block,
always in the same order (`a + tid`, `a + tid + 256`, …, then `blockTree`), and
always lands in `partial[c]`. The final pass walks `partial[0..2047]` in index
order. The only thing `gridHint` changes is **which** block computes which
chunk — and that never enters the arithmetic.

Four things to get right:

1. `L` must be computed from `n` and `NCHUNK` only. Anything involving
   `gridDim.x` reintroduces the dependency.
2. The per-chunk thread mapping must be fixed. `a + threadIdx.x` stepping by
   `BS` is fixed; a grid-stride over the chunk is not.
3. The `for (c = blockIdx.x; c < NCHUNK; c += gridDim.x)` loop condition is
   **block-uniform**, so the `__syncthreads()` inside `blockTree` is legal
   (Module 9's uniformity rule). A per-thread loop bound here would be undefined.
4. The trailing `__syncthreads()` at the bottom of the chunk loop is a
   write-after-read barrier on `sdata`: on the next chunk, thread `t` writes
   `sdata[t]` while threads 0–31 may still be reading `sdata[32..63]` from the
   previous chunk's warp tail. Module 9 calls this the barrier that needs G1 only
   and that double-buffering could remove. Omitting it is a genuine race, and
   `racecheck` finds it.

**What it costs.** Measured: `1.00–1.03×` — nothing. Three runs:

```
  plain fixed tree + finalTree : 0.4209 ms (318.9 GB/s)
  your deterministic version   : 0.4193 ms (320.1 GB/s)  ratio 1.00x
```

That is the result the exercise exists to deliver, and it is not obvious in
advance: the constraint sounds expensive (a fixed decomposition means you cannot
tune the work per block to the machine) and turns out to be free, because the
kernel is bandwidth-bound and `NCHUNK = 2048` chunks over 240 blocks still gives
every SM plenty of independent work. Reproducibility is expensive in a
*compute*-bound kernel and nearly free in a memory-bound one.

**What you gave up**, and the exercise wants you to be able to say it: the
ability to choose the granularity of the partial sums from the device. If
`NCHUNK` were 64 and you ran on a 200-SM GPU, most of the machine would idle.
The fix is to make `NCHUNK` large enough for any plausible machine and accept a
slightly longer final pass — which is exactly what CUB's deterministic modes do.

**Alternatives that also pass.** A two-level fixed tree (fixed 2048 chunks →
fixed 64 groups → 1) is the same idea with a shallower final pass. Ignoring
`gridHint` entirely and always launching 240 blocks would be reproducible but is
rejected by the harness, on purpose: the whole difficulty is doing it *while*
honouring a grid you do not control.

---

## TODO 3 — the fixed-point alternative

```cpp
static const double FP_SCALE = 1048576.0;   // 2^20
```

**Derive both bounds; do not guess.**

- *Overflow.* Every input is in `[0, 1000]` and there are 33,554,433 of them, so
  the true sum is at most `3.36e10`. The accumulator is `unsigned long long`,
  capacity `1.8e19`. Therefore `FP_SCALE < 1.8e19 / 3.36e10 = 5.4e8`.
- *Precision.* Rounding each element to the nearest `1/FP_SCALE` costs at most
  `0.5 / FP_SCALE` per element, so at most `n * 0.5 / FP_SCALE` in total. The
  tolerance is `1e-5` relative on a sum of `1.78e7`, i.e. `178` absolute.
  Therefore `FP_SCALE > 33554433 * 0.5 / 178 = 9.4e4`.

Any value in `[1e5, 5.4e8]` works; the window is nearly four decades wide, and
there is a wrong answer on each side. A power of two makes the scaling exact in
binary and is the conventional choice. `2^20` sits comfortably in the middle.

```cpp
__global__ void fpPass(const float* __restrict__ in, long long n,
                       unsigned long long* acc, double scale)
{
    __shared__ unsigned long long sdata[BS];
    unsigned tid = threadIdx.x;
    long long i = (long long)blockIdx.x * BS + tid, step = (long long)BS * gridDim.x;
    unsigned long long sum = 0ull;
    for (; i < n; i += step)
        sum += (unsigned long long)llrint((double)in[i] * scale);
    sdata[tid] = sum; __syncthreads();
    for (unsigned k = BS / 2; k > 0; k >>= 1) {
        if (tid < k) sdata[tid] += sdata[tid + k];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(acc, sdata[0]);
}
```

**Why the atomic is fine here.** Integer addition is associative *and*
commutative and 64-bit integers do not round, so `atomicAdd` on `acc` gives the
same value for every interleaving. This is the precise sense in which
`atomicAdd` is non-deterministic for `float` and deterministic for integers: the
atomic never promised an order, and only the float operator cared.

Measured: **1 distinct bit pattern over 3 grids × 5 runs**, relative error
`1.4e-8` — better than the float tree's, because the quantization noise is
symmetric and the accumulation is exact.

**What you gave up.** Dynamic range. You must know the magnitude of your data
before you run. Negative values need a bias (or a signed 64-bit accumulator with
`atomicAdd` on `long long`, which CUDA does not provide directly — use
`unsigned long long` with two's-complement wraparound, which works but has to be
argued for). Values spanning more than about 40 binary orders of magnitude cannot
be represented at all. This is the trade every deterministic-reduction library
offers, and it is why they usually ship both.

---

## TODO 4 — the predictions

| | answer | measured |
|---|---|---|
| (a) is the `atomicAdd` version bit-reproducible over 10 runs? | **no (0)** | 5–6 distinct patterns |
| (b) is the plain fixed tree stable across grids 97/240/1021? | **no (0)** | `0x4b87a099 0x4b87a099 0x4b87a098` |
| (c) deterministic / plain cost ratio | **1.0×** | 1.00–1.03× |

**(b) is the one worth dwelling on.** The tree is a *fixed-order* reduction —
every addition happens in a defined sequence, there is no atomic anywhere, and
the answer is perfectly reproducible for a given grid. It is still not
reproducible, because the grid is part of the order. Two of the three grids here
happen to agree (the difference is one ULP and it does not always land); 977
disagrees with 240 by two ULP, which the harness also prints in part 1. The
lesson is that "deterministic" is not a property of a kernel, it is a property of
a kernel *plus a launch configuration*, and a library that chooses its own launch
configuration from the device has not given you determinism at all.

**(c)** The temptation is to predict a slowdown, because the constraint sounds
restrictive. It is free, for the reason given above.

---

## Synchronization / memory reasoning

- `blockTree` has three `__syncthreads()` and they are the minimum: after the
  publishing store, after the 128-step, after the 64-step. There is no barrier
  after the 64-step's *reads* because nothing rewrites `sdata` — except in
  `detPass1`, where the chunk loop does, which is why that kernel has a fourth
  barrier at the bottom of the loop.
- The warp tail reads `sdata[tid] + sdata[tid + 32]` for `tid < 32`. Slots
  32–63 were written by warp 1 at the 64-step, so the barrier before the tail is
  required. It supplies both G1 and G2.
- `atomicAdd(&g_blocksDone, …)` does not appear in this exercise; the
  `__threadfence` last-block strategy is `example02.cu`'s. If you built your
  TODO 2 answer that way it is still reproducible, for the reason given in the
  lesson: the final tree walks `partial[]` in index order regardless of which
  block gets there last.

---

## Performance reasoning / expected output

```
Module 12 exercise 03 -- determinism and a silent tail
N = 33554433 (2^25 + 1), 128.0 MiB; element N-1 holds 1000000, the rest [0,1)

=== part 1: the defect ===
  double reference        17776946.2500
  reduceTreeBroken        16776946.0000   (short by 5.6253%)
  same kernel at n = 2^25 16776946.0000   (reference 16776946.2500, off by 1.490e-08)
  your reduceTreeFixed    17776946.0000   (off by 1.406e-08 relative)
  fixed @ grid 977        17776948.0000   ok
  fixed @ n = 2^25        16776947.0000   ok
  diagnosis: you said 3 -- correct

=== part 2: reproducibility ===
  atomicAdd finalization, 10 runs, same grid : 6 distinct patterns
  your fixed tree at grids 97/240/1021        : 0x4b87a099 0x4b87a099 0x4b87a098 (NOT stable)
  your launchDeterministic, 3 grids x 5 runs  : 1 distinct patterns (0x4b87a098, 17776944.0000)
  your launchFixedPoint,   3 grids x 5 runs  : 1 distinct patterns (0x4b87a099, 17776946.0000)
  fixed-point relative error                 : 1.406e-08  (scale 1048576)

  plain fixed tree + finalTree : 0.4209 ms (318.9 GB/s)
  your deterministic version   : 0.4193 ms (320.1 GB/s)  ratio 1.00x

predictions:
  (a) atomic reproducible?     you said no, measured no (6 patterns) MATCH
  (b) plain tree grid-stable?   you said no, measured no          MATCH
  (c) deterministic cost ratio  you said 1.00x, measured 1.00x     MATCH

SCORE: 8/8

OVERALL: PASS
```

The number of distinct atomic patterns varies run to run — 5, 6, 8 and 9 have
all been observed here, and Module 10 saw 10 on a different kernel. The
prediction is scored as "one, or more than one", which is the distinction that
matters and the only one that is stable.

Bandwidth on this part ranges from 113 GB/s (power-capped) to 396 GB/s
(fully ramped memory P-state) for the same binary on the same data. The ratio
between the two kernels is 1.00–1.03× in every state.

---

## The result that matters

Two independent things go wrong with "just sum the array", and only one of them
has a tool that finds it. The missing tail loop is invisible to every
`compute-sanitizer` mode, exact at power-of-two sizes, and 5.6% wrong at
`2^25 + 1` only because the lost element happened to be large — change the data
and the same bug returns an answer that is wrong in the eighth digit and passes
your test suite forever. The non-determinism is the mirror image: it has no bug
at all, every version is correct, and yet three correct implementations give
three different answers and one of them gives a different answer every time you
run it. **Correct, reproducible and fast are three separate properties, and a
reduction is the smallest program in which you can hold all three in your head at
once.**

**Variation to try.** Set `FP_SCALE` to `2^40` and re-run. The precision bound
is satisfied gloriously and the overflow bound is violated: the accumulator wraps
and the answer becomes nonsense, silently and reproducibly. Then set it to `1e3`
and watch the quantization error alone push you past the `1e-5` gate. Both
failures are deterministic, which is worth noticing — a reproducible wrong answer
is still a wrong answer, and reproducibility is not a correctness argument.
