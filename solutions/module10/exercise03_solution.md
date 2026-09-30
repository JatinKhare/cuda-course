# Module 10 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

---

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
nvcc -arch=sm_89 -O3 -cubin -o e3.cubin exercise03_solution.cu
cuobjdump -sass e3.cubin
```

---

## TODO 1 — the order-preserving key, and the sign trap

```cpp
__host__ __device__ inline unsigned int encode_key(float v)
{
    const unsigned int b = fbits(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
```

**Why it is correct.** IEEE-754 binary32 is `[sign | exponent(8) | mantissa(23)]`,
most significant first. That layout is deliberate: for two values of the
*same* sign, comparing the remaining 31 bits as an unsigned integer gives the
same answer as comparing the floats, because exponent dominates mantissa
exactly as it should. Two defects remain:

1. **Every negative pattern is numerically above every non-negative one**,
   because the sign bit is the top bit.
2. **Among negatives the order is reversed**: a larger magnitude means a
   larger exponent field, but a *smaller* value.

One operation fixes both for the negative half: complementing all 32 bits
clears the top bit (pushing negatives below non-negatives) and reverses the
order of the low 31 (undoing the reversal). For the non-negative half only
defect 1 applies, and setting the top bit lifts them above the complemented
negatives. Two different corrections, as the TODO said.

Concretely, from the program's own output:

```
  encode_key(-1.0f) = 0x407FFFFF   encode_key(-2.0f) = 0x3FFFFFFF
  raw bits  (-1.0f) = 0xBF800000   raw bits  (-2.0f) = 0xC0000000
```

−2 < −1. The keys satisfy `0x3FFFFFFF < 0x407FFFFF` — correct. The raw bits
satisfy `0xBF800000 < 0xC0000000` — **backwards**.

**The shortcut, and exactly how it fails.** The widely-copied idiom

```cpp
atomicMax((int*)addr, __float_as_int(v));    // WRONG for negatives
```

reinterprets as *signed*. For non-negative floats that works: the sign bit is
0, so the int is non-negative and ordered correctly. For negative floats the
int is negative, and `-1.0f` → `0xBF800000` → `-1082130432` while `-2.0f` →
`0xC0000000` → `-1073741824`. As signed ints, `-1082130432 < -1073741824`, so
the idiom concludes **−1.0 < −2.0**. Dataset B (all values strictly negative)
makes it return the *minimum*, confidently. `atomicMin` on the unsigned
reinterpretation is the corresponding fix for the negative half, and the
"correct" folklore version is a branch selecting between the two — which is
just this function, written less clearly.

Three checks worth running on any candidate: `-0.0f` maps below `+0.0f`
(here: `0x7FFFFFFF` vs `0x80000000`), `-INF` maps to the smallest key of any
finite float, and the map is a bijection (no two floats share a key). The
harness's monotonicity sweep over 4096 sampled patterns, including denormals
and infinities, catches anything that fails.

---

## TODO 2 — the CAS loop

```cpp
__device__ inline bool raw_better(unsigned long long a, unsigned long long b)
{
    const float va = raw_value(a), vb = raw_value(b);
    if (va != vb) return va > vb;
    return raw_index(a) < raw_index(b);
}

__device__ inline void argmax_cas_update(unsigned long long* best,
                                         float v, unsigned int idx)
{
    const unsigned long long cand = pack_raw(v, idx);
    unsigned long long old = *best;
    while (raw_better(cand, old)) {
        const unsigned long long assumed = old;
        old = atomicCAS(best, assumed, cand);
        if (old == assumed) break;
    }
}
```

**Why it is correct.**

- The **loop condition is the predicate**, not the CAS result. A thread whose
  candidate is not better than the incumbent never enters the loop and never
  issues an atomic. On dataset C (all values identical) the first thread wins
  and every other thread's candidate ties, `raw_better` is false, and they all
  leave immediately.
- The **exit test is `old == assumed`**, i.e. "the word still held what I
  compared against, so my swap happened". Not `old == cand`, which would exit
  whenever someone else had already written my value.
- On failure, `old` holds the *current* contents returned by the CAS, and the
  `while` re-tests `cand` against **that**. Nothing is ever re-derived from
  the pre-loop read.
- **Progress**: every failed CAS means some other thread succeeded, and each
  success strictly increases the incumbent in the `raw_better` order. There
  are finitely many candidates, so the loop terminates. (This is lock-freedom,
  not wait-freedom: an individual thread can retry many times.)

**The plain load `unsigned long long old = *best;`.** It is not atomic in the
formal sense and it may be stale the instant it lands. That is fine — the CAS
re-validates it, and 8-byte aligned accesses are not torn on this hardware.
What would *not* be fine is keeping that first read after a failure. If you
want to be strict, `old = atomicCAS(best, 0ull, 0ull)` gives you an atomic
read, at the price of an atomic for every thread including the ones that were
about to early-out — on dataset A that turns a 0.09 ms kernel into a 2 ms one.

**Common wrong approaches:**

| approach | symptom |
|---|---|
| `while (old != desired)` | Exits after a CAS that failed. Silent undercount; looks exactly like a plain RMW race, but the source contains `atomicCAS` so nobody suspects it. See Check-Your-Understanding Q4. |
| `do { ... } while (assumed != old)` with the predicate tested only once, before the loop | Dataset C **hangs**: every thread ties with the incumbent, the CAS always fails, and nothing re-tests the predicate. This is the livelock the file warns about. |
| Recomputing `desired` from the original `old` rather than from the CAS return | Works while contention is low, loses updates under load. |
| Comparing `cand > old` as packed integers | Compiles, is fast, and is wrong: the *raw* packing is `float bits` in the high half, which is not monotonic in the value. Dataset B fails. This is the same sign trap as TODO 1 wearing different clothes. |

---

## TODO 3 — making the loop unnecessary

```cpp
__host__ __device__ inline unsigned long long pack_for_max(float v, unsigned int idx)
{
    return ((unsigned long long)encode_key(v) << 32)
         | (unsigned long long)(0xFFFFFFFFu - idx);
}
__host__ __device__ inline unsigned int unpack_idx_for_max(unsigned long long p)
{
    return 0xFFFFFFFFu - (unsigned int)(p & 0xFFFFFFFFull);
}
```

**The reasoning.** `atomicMax` compares the whole 64-bit word as one unsigned
integer, which lexicographically means "compare the high half; if equal,
compare the low half". That is exactly the structure of the required order —
*by value, then by index* — provided both halves are monotonic in the right
direction.

- High half: `encode_key` makes it monotonic increasing in the value.
  Bigger wins. Correct.
- Low half: the comparison is also "bigger wins", but the tie-break must be
  "smaller index wins". So the low half must be monotonically *decreasing* in
  the index. `0xFFFFFFFF - idx` (identically `~idx`) does that and is its own
  inverse in the sense needed.

**The identity element.** The harness `memset`s the accumulator to all-zero
bits. Is `0x0000000000000000` worse than every real candidate? The high half
is 0, which is `encode_key` of the bit pattern `0xFFFFFFFF` — a negative NaN,
below every real float, and strictly below `encode_key(-INF) = 0x007FFFFF`.
So yes, and no separate initialisation kernel is needed. If you had chosen
`idx` instead of `~idx` in the low half, the all-zero word would tie with any
candidate at index 0 and the identity would no longer be strictly worse —
a subtle bug that only shows up when the true argmax is index 0. Dataset C's
answer is index 0.

**What this buys.** The CAS loop becomes one `RED.E.MAX.64.STRONG.GPU`. No
retry, no divergence, no loop. And for a read-only-loser thread there is no
early-out either, which is why the naive version of this is *slower* than the
CAS loop on dataset A — see the performance section.

---

## TODO 4 — two fields, one atomic

```cpp
__device__ inline void bounds_update(unsigned long long* bounds, float v)
{
    const unsigned int k = encode_key(v);
    unsigned long long old = *bounds;
    for (;;) {
        const unsigned int mx = (unsigned int)(old >> 32);
        const unsigned int mn = (unsigned int)(old & 0xFFFFFFFFull);
        if (k <= mx && k >= mn) return;               // nothing to do
        const unsigned int nmx = (k > mx) ? k : mx;
        const unsigned int nmn = (k < mn) ? k : mn;
        const unsigned long long want = ((unsigned long long)nmx << 32)
                                      | (unsigned long long)nmn;
        const unsigned long long assumed = old;
        old = atomicCAS(bounds, assumed, want);
        if (old == assumed) return;
    }
}
```

**Why one atomic and not two.** `atomicMax` on the high half and `atomicMin`
on the low half would produce the correct *final* answer, because max and min
are independent. It would not produce a correct *intermediate* word: between
the two atomics there is an instant at which the max came from this thread's
value and the min did not. Any other thread — or a host `cudaMemcpy` from a
concurrent stream, or a subsequent kernel reading it — can observe a `{max,
min}` pair that never corresponded to a single state of the data. The
requirement was consistency at all times, and CAS is the only way to get a
multi-field update that is indivisible as a unit.

This generalises: **CAS on a packed word is how you make a small struct
atomic.** 64 bits is the ceiling on this hardware for a single instruction;
above that you need a lock, and locks on a GPU are a different and much worse
conversation.

**Why the early return matters.** `if (k <= mx && k >= mn) return;` before
the first CAS. On dataset A, after the first few hundred thousand elements
the bounds are already the final ones, and essentially every remaining thread
takes this path and issues no atomic at all. Without it, 4,000,003 threads
each perform at least one CAS on one address, which is the worst pattern in
the module. The re-derivation of `mx` and `mn` from the *returned* `old` after
a failed CAS is the other half of the correctness argument — recomputing only
one of the two fields is the bug the exercise is built around.

---

## TODO 5 — argmax_fast

```cpp
__shared__ unsigned long long s[32];
const int i = blockIdx.x*blockDim.x + threadIdx.x;
const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;

unsigned long long cand = (i < n) ? pack_for_max(x[i], (unsigned int)i) : 0ull;

for (int off = 16; off > 0; off >>= 1) {
    const unsigned long long o = __shfl_down_sync(0xFFFFFFFFu, cand, off);
    if (o > cand) cand = o;
}
if (lane == 0) s[wid] = cand;
__syncthreads();
if (wid == 0) {
    cand = (lane < nw) ? s[lane] : 0ull;
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_down_sync(0xFFFFFFFFu, cand, off);
        if (o > cand) cand = o;
    }
    if (lane == 0 && cand != 0ull) atomicMax(best, cand);
}
```

**The design.** Collapse 256 candidates to 1 inside the block, then do exactly
one global atomic per block. Global atomics on the single hot address drop
from 4,000,003 to 15,626 — a factor of 256. The reduction itself is free
relative to the memory traffic; the kernel ends up bandwidth-bound.

**Why the monotonic packing (TODO 3) is what makes this possible.** The
reduction operator is plain `>` on a 64-bit unsigned. If you had to call
`raw_better` you would be unpacking two floats per shuffle step; more
importantly, the *combine* would not be expressible as a `max`, and the tie
break would have to be threaded through by hand. A good encoding turns an
associative-but-awkward operator into `max`, and `max` composes trivially.

**The two traps the file warns about, and how this handles them.**

- `n = 4,000,003`, so the last block is partial. Out-of-range lanes are given
  the identity `0ull`, which the packing guarantees loses to every real
  candidate. Crucially this is done **without a branch around the reduction**:
  every lane participates in every `__shfl_down_sync`, so the `0xFFFFFFFF`
  mask is honest and no lane is waiting at a shuffle its partner never
  reaches. Writing `if (i < n) { ...reduce... }` instead is the classic way to
  break this on sm_70+ with independent thread scheduling (Module 8).
- `__syncthreads()` sits outside all divergent control flow. Every thread of
  the block reaches it (Module 9). Putting it inside `if (lane == 0)` hangs.

**Why `cand != 0ull` before the atomic.** A block entirely past the end (not
possible here, but possible for other `n`) would otherwise write the identity.
Harmless with `atomicMax`, but free to guard and necessary if you ever switch
the accumulator to something without an identity.

---

## Synchronization / memory reasoning

Three different atomics appear in the SASS, and the difference between them is
the whole module:

```
Function : _Z12argmax_naivePKfiPy
    RED.E.MAX.64.STRONG.GPU [R4.64], R6 ;

Function : _Z17argmax_cas_kernelPKfiPy
    ATOMG.E.CAS.64.STRONG.GPU PT, R2, [R4], R8, R10 ;

Function : _Z13bounds_kernelPKfiPy
    ATOMG.E.CAS.64.STRONG.GPU PT, R2, [R4], R8, R10 ;

Function : _Z11argmax_fastPKfiPy
    SHFL.DOWN PT, R4, R9, 0x10, 0x1f ;      (x20, the two reductions)
    ...
    RED.E.MAX.64.STRONG.GPU [R2.64], R4 ;   (exactly one, per block)
```

`argmax_naive` discards the return value of `atomicMax`, so the compiler emits
`RED` — fire and forget, no scoreboard dependency. A CAS *must* be `ATOMG`,
because the loop's correctness depends on the returned value. That is not a
missed optimization; a compare-and-swap without its return is meaningless.

Note also what is absent: no `VOTEU`/`POPC` warp aggregation in any of these.
The compiler aggregates same-address `atomicAdd`s; it does not (and cannot
generally) aggregate a CAS loop, and it did not aggregate the `atomicMax`
here. `argmax_fast` does the aggregation explicitly with shuffles, which is
the manual version of the same idea one level up.

---

## Performance reasoning

Observed on the RTX 3500 Ada, three consecutive runs. Absolute ms drift with
the clock; the ratios do not.

```
=== timing (dataset A, min of 4 sweeps x 20 iters) ===
  kernel                                     ms       GB/s     %peak
  CAS loop, one address                  0.0876      182.6     42.3%
  atomicMax, one address                 2.0550        7.8      1.8%
  argmax_fast (yours)                    0.0504      317.6     73.5%
  bounds CAS loop, one address           0.1267      126.3     29.2%

  argmax_fast vs argmax_naive: 40.79x
  CAS loop vs single atomicMax: 0.04x
```

Ranges over three runs: CAS loop 0.088–0.100 ms, atomicMax 2.051–2.056 ms,
`argmax_fast` 0.050–0.051 ms, bounds 0.120–0.128 ms;
`argmax_fast / argmax_naive` **40.5–40.9×**. `x[]` is 15.3 MB and therefore
L2-resident, so the GB/s column is cache bandwidth — use it as a relative
scale, not as a fraction of 432 GB/s.

**The headline result is the one nobody predicts: the CAS loop is 23× FASTER
than the single `atomicMax`.**

A CAS loop is strictly more work per update than one `atomicMax` — more
instructions, a possible retry, and `ATOMG` instead of `RED`. It wins anyway,
by a factor of 23, because of the early-out. Dataset A's maximum sits at index
7, so after the first warp or two *every subsequent thread's `raw_better` test
fails and it issues no atomic at all*. 4,000,003 atomics become a few hundred.
Meanwhile `argmax_naive` issues its `atomicMax` unconditionally, all 4,000,003
of them, against one address: 2.05 ms for 4 M operations is 1.95 Gatomic/s,
which is precisely the K = 1 row of the lesson's contention table (1.93
Gatomic/s) — the same hardware limit, reached from a different direction.

**The cheapest atomic is the one you do not execute.** That is a fifth
contention-reduction technique alongside the four in the lesson, and on
monotone-ish data it is the strongest of them.

Two honest qualifications:

- **It is data-dependent.** On an input whose maximum arrives last (a sorted
  ascending array), every thread would pass the test and the CAS loop would be
  the worst kernel here, not the best. Do not generalise from dataset A.
- **`argmax_fast` beats both regardless**, at 0.050 ms, because it removes the
  contention structurally rather than relying on the data. It reads 15.3 MB
  and does 15,626 atomics, and its time is essentially the time to read the
  array. That is the floor, and it is reached without any assumption about
  the input.

The bounds kernel at 0.127 ms is 2.5× the argmax CAS loop, despite having the
same early-out. Two reasons: its predicate fails more often (a value only has
to escape *either* end of the range, and the range settles later than a single
maximum does), and each retry recomputes two fields.

---

## Expected output

Quoted in full above; `OVERALL: PASS` with `score: 5/5`. All three datasets
return the CPU's index from all three kernels, including index **0** on
dataset C (all values identical, tie broken to the smallest index) and index
**999** on dataset B (all values negative).

---

## The result that matters

`atomicCAS` is the universal primitive, and that makes it a trap: because it
can express anything, it is tempting to reach for it whenever the hardware
lacks an instruction. The better move is usually to change the *encoding* so
that the instruction the hardware does have becomes the one you need — here, a
31-line order-preserving map turns "argmax of floats with an index tie-break",
which has no hardware support at all, into a single `RED.E.MAX.64`. And the
measurement then delivers the second lesson, which contradicts the first: on
this data the CAS loop still beats that single instruction by 23×, purely
because it can decline to issue an atomic at all. Structure, then encoding,
then early-out — and measure, because the ordering of those three is not what
you expect.

**Variation to try:** sort dataset A ascending before running, so that every
thread's candidate beats the incumbent at the moment it arrives. Predict what
happens to each of the four kernels first. The CAS loop should collapse from
best to worst, `argmax_naive` should not move at all, and `argmax_fast` should
not move either — which tells you which of the three designs you would ship.
