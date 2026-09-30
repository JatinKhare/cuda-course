# Module 07 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
```

## The two instincts, and which one is wrong

**Kernel A** indexes shared memory with a value read out of a data array:

```cpp
const int base = (idx[t] & 31) * SCALE;
... sA[(base + j*32 + ...) & (SA_FLOATS - 1)] ...
```

Every reviewer flags this. A data-dependent index looks like an uncontrolled
gather, and "uncontrolled gather into a banked memory" sounds like the
definition of a bank conflict.

**Kernel B** gives each thread a private slice of a shared scratch buffer:

```cpp
scratch[t * SCRATCH + k]        // thread t's k-th private float
```

Nobody flags this. There is no sharing at all — thread `t` is the only thread
that ever touches its own slice — so what could possibly conflict?

**Kernel A is fine. Kernel B is a 32-way disaster.** Both for the same reason:
bank membership is a property of *addresses*, and has nothing whatsoever to do
with which thread owns the data or how the index was computed.

## TODO 1 — the six degrees

```cpp
static int predictedDegree[6] = {
     1,   // A1  SCALE=1,  random idx
     1,   // A2  SCALE=1,  adversarial idx
     1,   // A3  SCALE=32, warp-uniform idx
    32,   // A4  SCALE=32, adversarial idx
    32,   // B1  scratch[t * SCRATCH + k]
     1    // B2  your layout
};
```

Derivations:

- **A1, A2 (`SCALE = 1`).** The element index is `idx[t] & 31`, a value in
  `[0, 32)`. Bank = index % 32 = the index itself, and word = the index itself.
  So **the index determines the bank and the word by the same expression**. Two
  lanes are in the same bank if and only if they want the same word, which is a
  broadcast. `D = 1` for every possible content of `idx`. Proved below.
- **A3 (`SCALE = 32`, warp-uniform index).** `idx[t] = (t/32)*7` is constant
  within a warp, so all 32 lanes compute the same `base`, and after the `+ j*32`
  offset they all read the **same word**. One bank, one word, 32 readers:
  **broadcast, `D = 1`.** This is the row that looks worst on paper — a stride
  of 32 floats is the textbook catastrophe — and is the cheapest thing in the
  file.
- **A4 (`SCALE = 32`, 32 distinct low-5-bit values).** `base = (idx & 31) * 32`,
  so bank = `(32k) % 32 = 0` for every lane, and the 32 distinct `k` give 32
  distinct words. **`D = 32`.**
- **B1.** Thread `t`'s element `k` is at `t*32 + k`, so
  `bank = (32t + k) % 32 = k`. For one instruction `k` is the same for all lanes
  and `t` varies over 32 consecutive threads: **all 32 lanes in bank `k`, at 32
  distinct words. `D = 32`.** The per-thread stride is `SCRATCH = 32` floats,
  which is the single worst stride the hardware has.
- **B2.** See TODO 3.

Configurations A3 and B1 are the two that catch people, in opposite directions.

## TODO 2 — the worst case kernel A can reach at `SCALE = 1`

```cpp
static const int WORST_A1_DEGREE = 1;
```

**No content of `idx` can produce a conflict.** Proof:

Let `v_L = idx[L] & 31 ∈ [0, 32)` be lane `L`'s masked value. The element index
is `v_L` (plus a warp-uniform offset that is a multiple of 32 and therefore
changes the word but not the bank, identically for every lane). Then

```
bank(L) = v_L % 32 = v_L          because v_L < 32
word(L) = v_L + (uniform offset)
```

Two lanes `L`, `M` share a bank iff `v_L = v_M`, and in that case they also have
the same word — a broadcast. There is no pair of lanes with the same bank and
different words, so `D = 1`. Always.

The masking is doing all of the work: `& 31` confines the index to a 32-element
window, and a 32-element window of a word-interleaved memory contains **exactly
one word per bank**. Within such a window, bank collision and word equality are
the same event.

The harness brute-forces the claim over 200 000 candidate index sets, including
the all-distinct and all-equal extremes, and reports the maximum degree it
found. It finds 1.

The general statement, worth carrying: **an index of the form `s[f(tid) & (N-1)]`
with `N ≤ 32` is unconditionally conflict-free, whatever `f` is.** Data-dependent
indexing into shared memory is not inherently dangerous; the *range* of the
index relative to 32 is what decides. `s[f(tid) & 1023]` is a different story
entirely — there, a 1024-element window contains 32 words per bank and an
adversarial `f` reaches `D = 32`.

## TODO 3 — kernel B's layout

```cpp
__host__ __device__ __forceinline__ int scratch_index_v2(int t, int k, int nthreads)
{
    return k * nthreads + t;
}
```

Transpose the scratch buffer: index by `[k][t]` instead of `[t][k]`. Exactly the
same `THREADS_B * SCRATCH` floats, not one word more.

`bank = (k*128 + t) % 32 = t % 32`, since `nthreads = 128` is a multiple of 32.
For one instruction `k` is uniform and `t` runs over 32 consecutive threads, so
the 32 lanes occupy 32 consecutive words — **all 32 banks, `D = 1`.** And it
fixes the *fill* loop as well as the read loop, for the same reason: both
phases hold `k` uniform across the warp.

**This is M5's AoS → SoA transformation, one level down the memory hierarchy.**
In M5 the fix for a warp reading field `.x` of 32 adjacent 24-byte structs was
to split the array of structs into a struct of arrays, so the warp's 32 accesses
became contiguous. Here the fix for a warp reading element `k` of 32 adjacent
32-float private records is the same split, with the same effect on the same
kind of counter. The penalty formula is even analogous: M5's AoS cost was
`record_size / bytes_used`; here the cost is `gcd(record_size_in_words, 32)`.
Choose `SCRATCH = 33` and B1 becomes conflict-free too — but that is the padding
fix, and it costs memory, which this TODO forbids.

Common wrong approaches:

- **`t * SCRATCH + ((k + t) & 31)`** — a rotation of `k` within the thread's own
  row. Injective, but the bank is still `(32t + something) % 32 = something`,
  and `something` now *differs* per lane... which actually does work for the
  read. It fails the structural test only if you get the wrap wrong. It is a
  legitimate alternative answer; the harness accepts it.
- **`k * SCRATCH + t`** — uses `SCRATCH` (32) as the stride instead of
  `nthreads` (128). Not injective: `(t=32, k=0)` and `(t=0, k=1)` both map to
  32. The harness reports `not injective`.
- **`t + k * 129`** — works for banking, but `129 * 31 + 127 = 4126 > 4096`.
  The harness reports `out of range`. This is the padding fix in disguise, and
  the problem statement forbade the extra memory.

## TODO 4 — the adversarial index set

```cpp
for (int lane = 0; lane < WARP; ++lane)
    hIdxAdv[lane] = lane;
```

Any 32 values that are **pairwise distinct modulo 32** work. `lane` is the
simplest. At `SCALE = 32` they produce 32 distinct words all in bank 0:
`D = 32`, the maximum.

The point of TODO 4 is the pair of numbers the harness prints:

```
your array gives degree 32 at SCALE=32 (ok) and degree 1 at SCALE=1
```

**The same 32 index values are simultaneously the worst possible input and a
perfectly benign one.** Nothing about the data changed; only the multiplier in
the address computation did. Conflicts are a property of the (data, indexing
expression) pair, and asking "is this data bad?" is not a well-formed question.

A related trap worth naming: a reader who sets `hIdxAdv[lane] = lane * 32`
expecting "a big stride" gets `(lane*32) & 31 == 0` for every lane — the mask
erases the stride and produces a broadcast, degree 1 at both scales. The harness
reports `not maximal`.

## Synchronization / memory reasoning

Both kernels use one `__syncthreads()` between the shared-memory fill and the
read loop. In kernel A it is genuinely required: thread `t` reads
`sA[(idx[t] & 31) + j*32]`, which was written by some other thread. In kernel B
it is *not* strictly required — every thread reads only slots it wrote itself,
in both layouts — but it is kept because removing it is a correctness argument
that depends on the layout, and changing the layout would silently invalidate
it. Module 9 makes the ordering rules precise; the conservative barrier costs
essentially nothing here (one barrier per kernel against `96 × 32` shared reads)
and removes a footgun from TODO 3.

Note that the barrier is unaffected by any of the six configurations. **Bank
conflicts are an intra-warp, intra-instruction phenomenon; no barrier creates or
removes one.**

## Performance reasoning

Measured, warm, reproducible across runs to better than 1 %:

| config | degree | ms | vs its reference |
|---|---|---|---|
| A1 `SCALE=1`, random idx | 1 | 0.0601 | 1.00× |
| A2 `SCALE=1`, adversarial idx | 1 | 0.0599 | 1.00× |
| A3 `SCALE=32`, warp-uniform idx | 1 | 0.0625 | 1.04× |
| A4 `SCALE=32`, adversarial idx | 32 | 0.7420 | **12.36×** |
| B1 `scratch[t*32 + k]` | 32 | 2.9761 | **10.52×** |
| B2 `scratch[k*128 + t]` | 1 | 0.2828 | 1.00× |

Four things to read out of that table:

1. **A1 and A2 are identical (1.00× and 1.00×).** The masked index is
   conflict-free for random data and for data chosen specifically to break it.
   The instinct "data-dependent shared index = danger" is wrong here, and the
   reason is structural, not lucky.
2. **A3 is 1.04×, not 32×.** A stride of 32 floats with a warp-uniform index is
   a **broadcast**, the cheapest access there is. This is the same
   broadcast-versus-serialize split M4 measured for constant memory, where a
   warp-uniform constant read was 24.6× cheaper than a lane-varying one. The
   lesson transfers verbatim: *in a replay-based memory, what matters is how
   many distinct resources the warp asks for, not the stride in the source.*
3. **A3 versus A4 is the whole exercise in one pair.** Identical kernel,
   identical instruction stream, identical addresses-modulo-32 arithmetic —
   and a 12× difference that depends only on the *values* in a device array
   that neither the compiler nor the profiler's static view can see. A conflict
   can be a property of your input data. Profile with representative data.
4. **B1 is 10.5× slower than B2 at zero cost to fix.** "Per-thread private
   scratch" is the most innocent-looking shared-memory idiom there is, and
   laying it out as `[thread][slot]` with a slot count that is a multiple of 32
   is the single worst thing you can do to the bank array.

**Why 12.4× and 10.5× rather than 16×.** `D/2 = 16` bounds the shared-memory
*term*, not the kernel. Each loop iteration also does an FFMA, address
arithmetic, and loop control, and none of those slow down when the banks do. For
B, the non-shared fraction is larger (the index arithmetic involves a multiply
by `nthreads`), which is why B1's ratio is lower than A4's.

**Note on the measurement method.** All six configurations are timed
back-to-back inside one loop, the configuration order **rotates by one each
sweep**, and the reported value is the minimum over four sweeps after a
duration-based (4 s) clock warm-up. The rotation matters: without it, whichever
configuration is measured first in every sweep absorbs the clock dip that
follows the preceding `cudaDeviceSynchronize()`, and in an earlier version of
this harness that inflated the reference configuration by 30 % and made a
broadcast look *faster* than conflict-free. Ratios against a contaminated
baseline are worse than no measurement.

## Expected output

Actual output on the RTX 3500 Ada:

```
TODO 2: you claimed the worst reachable degree at SCALE=1 is 1.
        200000 candidate index sets, including all-distinct and
        all-equal, produced a maximum of 1. ok
TODO 4: your array gives degree 32 at SCALE=32 (ok) and degree 1
        at SCALE=1. Those two numbers are the exercise.
TODO 3: layout test ok, size 4096 floats (B1 uses 4096)

Numerical check vs CPU reference: PASS

config               yours  actual         ms   measured   verdict
---------------------------------------------------------------------------
A1 scale1 random         1       1     0.0601      1.00x   ok
A2 scale1 advers         1       1     0.0599      1.00x   ok
A3 scale32 unif          1       1     0.0625      1.04x   ok
A4 scale32 adver        32      32     0.7420     12.36x   ok
B1 scratch[t][k]        32      32     2.9761     10.52x   ok
B2 your layout           1       1     0.2828      1.00x   ok

Degrees correct: 6/6

A degree of D costs at most D/2 times a conflict-free access,
so the two 32-way rows could have been 16x. A4 measured 12.4x,
B1 measured 10.6x. The rest of each loop -- the FFMAs, the loop
control, the store -- does not slow down when the banks do.

TODO 1: 6/6   TODO 2: ok   TODO 3: ok   TODO 4: ok   numerics: PASS

OVERALL: PASS
```

Absolute times roughly double on a cold GPU; the ratios move by under 5 % across
thermal states, and the degree columns are deterministic.

`ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` would
report zero for A1, A2, A3 and B2 and a large non-zero for A4 and B1. It could
not be run here: Nsight Compute 2026.1.0 is installed on this machine but
returns `ERR_NVGPUCTRPERM` without elevated performance-counter permissions, so
all degrees were verified against timing ratios and against the controlled
degree sweep in `example01.cu`.

## The result that matters

**Bank conflicts are a property of the set of addresses one warp produces, and
nothing else.** Not of who owns the data — kernel B's slices are perfectly
private and 32-way conflicted. Not of how the index was computed — kernel A's
index comes out of memory and is provably safe. Not even of the source-level
stride — A3 has a stride of 32 floats and is free, because all its lanes want
the same word. Every instinct that skips straight to the address set is
reliable; every instinct based on the *shape* of the expression is not.

**Variation to try.** Change kernel A's mask from `& 31` to `& 1023` (and
`SA_FLOATS` stays 2048). Now the reachable window holds 32 words per bank and
TODO 2's proof collapses: find an index set that reaches `D = 32` at
`SCALE = 1`, and convince yourself why no such set existed before. Then set
`SCRATCH` to 33 in kernel B, leaving the `[t][k]` layout alone, and watch the
32-way conflict evaporate — that is the padding fix, and comparing its shared
memory footprint against B2's is the whole padding-versus-swizzle trade-off in
miniature.
