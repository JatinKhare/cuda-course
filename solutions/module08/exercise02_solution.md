# Module 08 / Exercise 02 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

The workload: `N = 1,048,576` elements; element `i` needs `work[i]` rounds of a
128-FFMA dependent chain, with `work[i] = 1 + (mix32(i) & 7)` in 1..8. The
measured histogram is close to uniform (130,808 … 131,334 per class), giving
`mean(work) = 4.4996` and `max(work) = 8`.

---

## The cost model, before any code

The naive mapping is thread `t` to element `t`. Consecutive threads get
consecutive elements, and `work[]` is a hash, so the 32 elements of a warp are
32 independent draws from 1..8. The probability that all 32 agree is
`8 * (1/8)^32`, i.e. never — the harness measures **0.00%** warp homogeneity.
Every warp therefore runs `max(work over its 32 lanes)`, which with 32 draws is
8 essentially always.

So the naive grid costs `N * 8` rounds of lane-capacity and performs
`N * 4.4996` rounds of useful work: **56% lane efficiency**.

If every warp's 32 elements shared one `work` value, each warp would cost its
own class and the grid would cost `N * 4.4996`. The ceiling on the speedup is

```
max(work) / mean(work) = 8 / 4.4996 = 1.778
```

That is TODO 4's answer, and it is derived, not guessed.

---

## TODO 1 — the plan (the design question)

```cpp
static int buildPlan(const int* work, int n, int* order)
{
    int count[MAXWORK + 2];
    for (int k = 0; k <= MAXWORK + 1; ++k) count[k] = 0;
    for (int i = 0; i < n; ++i) ++count[work[i]];

    int start[MAXWORK + 2];
    int acc = 0;
    for (int k = 0; k <= MAXWORK; ++k) { start[k] = acc; acc += count[k]; }

    for (int i = 0; i < n; ++i) order[start[work[i]]++] = i;
    return 1;
}
```

A **counting sort** by work class. `order[t]` is the element that thread `t`
should process. Elements of class 1 occupy `order[0 .. c1)`, class 2 the next
`c2` slots, and so on.

Why this is correct from the hardware model: thread `t` and thread `t+1` are
adjacent lanes of the same warp unless `t+1` crosses a multiple of 32. Placing
same-class elements contiguously in `t` therefore places them in the same warp.
The only warps that remain mixed are the at most `MAXWORK - 1 = 7` warps that
straddle a class boundary, which out of `N/32 = 32,768` warps is 0.02% — the
harness measures **99.98%** homogeneity.

Why counting sort and not a comparison sort: `work` has only 8 distinct values,
so the ordering is fully determined by a histogram. Two linear passes and an
8-entry table, `O(n)`, no comparisons, no recursion. A `qsort` with a
comparator on `work[order[i]]` produces an equally good permutation and costs
roughly an order of magnitude more host time for no benefit, which matters
because the harness prices the preparation (see below).

### Other strategies that work

- **Make the branch warp-uniform instead of permuting.** Give every *warp* a
  single trip count equal to the maximum over its lanes, and have the lanes that
  need less discard the extra rounds. This removes divergence but not work, so
  it buys nothing here — the whole cost *is* the extra work. It is the right
  answer when the arms differ in kind rather than in length.
- **Convert control flow to arithmetic.** Replace the data-dependent loop with a
  fixed 8-round loop in which round `r` multiplies by a per-lane
  `(r < work[i]) ? 0.99999f : 1.0f`-style selector. Correct, branchless, and
  costs `max` unconditionally — i.e. exactly the naive cost, with none of its
  upside. Useful when the spread is small; useless when it is 8:1.
- **Two-level binning.** Bin at block granularity instead of globally, so the
  permutation is local and can be computed on the device. Achieves nearly the
  same homogeneity with no global sort. This is the production shape, and it is
  what Module 12 and Module 16 build once you have atomics and scan.

### Common wrong approaches

- **Sorting `work[]` itself instead of building a permutation.** You then lose
  the mapping back to `x[]` and `out[]` and cannot write the answer anywhere.
- **Sorting descending vs ascending.** Both give 99.98% homogeneity and
  essentially identical time. The order of the classes does not matter; only
  their contiguity does. (Descending is marginally better for the *tail* of the
  grid — the expensive blocks launch first — but with a grid this deep the
  effect is under the noise floor.)
- **Grouping into runs shorter than 32.** Any scheme that produces
  same-class runs of length `k < 32` leaves every warp mixed and buys nothing.
  The harness's homogeneity line diagnoses this immediately.

---

## TODO 2 — the kernel

```cpp
__global__ void refineFast(const float* __restrict__ x,
                           const int*   __restrict__ work,
                           float*       __restrict__ out,
                           const int*   __restrict__ order, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int i = order[t];
    if (i < 0 || i >= n) return;
    out[i] = refine(x[i], work[i]);
}
```

One line: `int i = order[t];`. Everything downstream — the load of `x`, the load
of `work`, and the store to `out` — uses `i`, not `t`.

**The trap.** Writing `out[t]` instead of `out[i]` gives a kernel that is just
as fast, never reads out of bounds, and produces a permuted result. The harness
compares bit-exactly against the naive output over all 1,048,576 elements, so it
reports mismatches immediately — but in real code nothing would. This is the
same class of error as Module 3's axis assignment: correct-looking, fast, and
silently wrong.

**A note on coalescing, since Module 5 is still fresh.** `order[t]` is
contiguous (`t` and `t+1` are adjacent), so the load of `order` is perfectly
coalesced. The loads of `x[i]` and `work[i]` are *not*: `i` jumps around within
a class, so a warp's 32 addresses are scattered and cost up to 32 sectors
instead of 4. The store to `out[i]` is scattered too, and a scattered store is
worse than a scattered load because a partial-sector store forces a
read-modify-write (Module 5).

None of that shows up in the timing, because this kernel does up to 1024
dependent FFMAs per element and touches 12 bytes. It is compute-bound by three
orders of magnitude. **If the kernel were memory-bound, the permutation would
very plausibly cost more than the divergence it removes**, and the right answer
would be to physically reorder `x` and `work` into class-contiguous arrays
instead of gathering through an index. That is the trade the two-level binning
approach makes.

---

## TODO 3 — the launch configuration

```cpp
static int FAST_BLOCK = 256;
```

The only requirement is that `FAST_BLOCK` be a multiple of 32; the harness
rejects anything else. 256 is the Module 3 default and gives 100% occupancy on
sm_89 (1536/256 = 6 blocks per SM).

**Why it interacts with TODO 1.** Warps are cut out of the linearized thread
index, and `t = blockIdx.x * blockDim.x + threadIdx.x`. If `blockDim.x` is a
multiple of 32 then warp boundaries in `t` fall on multiples of 32 globally, and
a plan that groups elements into runs aligned to 32 in `t` really does group
them into warps. If `blockDim.x` were 96, say (still a multiple of 32, still
fine), boundaries are still at multiples of 32. But if it were 100, every block
would end with a partial warp and the *global* `t` sequence would be
discontinuous relative to warp boundaries, degrading the grouping and wasting
28 lanes per block on top.

---

## TODO 4 — the prediction

```cpp
static const float PREDICTED_SPEEDUP = 1.778f;   // max(work)/mean(work)
```

Measured: **1.578x**, 12.7% below the model, inside the ±20% band.

---

## Synchronization / memory reasoning

No synchronization of any kind: one thread, one element, no shared memory, no
cross-thread communication. The `order` array is read-only on the device. The
permutation is computed on the host and copied once.

That is worth noticing in its own right: the entire optimization in this
exercise is a *layout* change, not a *communication* change. Divergence is
removed by deciding which thread gets which element, which is a Module 3
question answered with a Module 8 criterion.

---

## Performance reasoning

Why the measured 1.578x falls short of the 1.778x model.

1. **The model assumes the grid is issue-bound and perfectly balanced.** After
   sorting, blocks are homogeneous but *unequal*: a block of class-1 elements
   finishes in an eighth of the time of a block of class-8 elements. The grid is
   4096 blocks over 40 SMs, so the block scheduler does rebalance, but the tail
   of the kernel runs with only the expensive blocks resident and fewer eligible
   warps per scheduler, which costs some latency hiding on a dependent-FFMA
   chain. The naive kernel has no such tail: every block is identically
   expensive.

2. **The naive kernel is not exactly at `max = 8` and the sorted kernel is not
   exactly at `mean`.** The 7 straddling warps cost `max` of their two classes,
   and about 1.4% of warps in the naive mapping happen to contain no class-8
   element at all (`(7/8)^32 = 0.0139`, i.e. roughly 460 of the 32,768 warps),
   so the naive kernel pays slightly less than `8` on average and the true
   ratio of the two models is a little under 8/4.4996.

3. **The permuted kernel has worse memory behaviour** (scattered `x`, `work`,
   `out`), which is negligible but not zero.

The ratio is the stable quantity. Across repeated runs on this GPU it sits in
**1.54–1.60x**; the absolute milliseconds move by more than 2x between a cold
GPU and a heat-soaked one, which is why spec §12 exists and why the harness
times both configurations back-to-back inside the same sweep, takes the minimum
of five sweeps, and validates afterwards.

### The preparation cost, and when this is worth doing

```
buildPlan   :    3.000 ms on the host
saves 0.0448 ms per launch -> pays for itself after 67 launches
```

`clock()` on Windows has ~1 ms resolution, so this figure quantises: observed
values across runs are 1, 2 and 3 ms, giving break-even counts of 25, 49–52 and
67 launches. The counting sort over 1M elements really costs about 2 ms. Treat
the break-even as "tens of launches", not as a precise number.

This is the number that decides whether the technique is an optimization or a
pessimization. Sorting to remove divergence is worth it when:

- the permutation is reused across many launches (an iterative solver, a
  time-stepped simulation, a training loop) — 67 launches here;
- or the sort can be done on the device and folded into work you were doing
  anyway;
- or the work spread is much larger than 8:1, which raises the saving without
  raising the sort cost.

It is not worth it for a single launch over freshly generated data. Report the
break-even, not just the speedup.

---

## Expected output

Actual output observed on the RTX 3500 Ada, CUDA 13.2, `-arch=sm_89 -O3`:

```
=== Module 8 / Exercise 2 : make the lanes agree ===
N = 1048576 elements, work[i] in 1..8, one round = 128 FFMAs
work histogram: 1:130936 2:131060 3:131292 4:130936 5:131271 6:131334 7:130808 8:130939
mean work = 4.4996 rounds, max = 8

CPU cross-check of the naive kernel: 4081/4081 sampled elements agree
refineFast vs refineNaive: 0 mismatches of 1048576

--- warp homogeneity (all 32 lanes share one work value) ---
  naive (thread t -> element t), block  256 :   0.00%
  yours (thread t -> element order[t]), block  256 :  99.98%

--- timing (min of 5 sweeps of 20 iterations) ---
  refineNaive :   0.1223 ms
  refineFast  :   0.0775 ms
  speedup     : 1.578x    (you predicted 1.778x)
  prediction  : WITHIN 20% (12.7% off)

--- cost of your host-side preparation ---
  buildPlan   :    3.000 ms on the host
  saves 0.0448 ms per launch -> pays for itself after 67 launches

OVERALL: PASS
```

Absolute times vary with thermal state: `refineNaive` has been observed between
0.122 ms and 0.32 ms across runs. The **speedup is stable at 1.54–1.60x**
(observed 1.546, 1.570, 1.571, 1.578 over four runs), which is why that is the
number the exercise is graded on. The `buildPlan` line quantises to whole
milliseconds and the break-even count moves with it; see above.

---

## The result that matters

Divergence caused by data is a *layout* problem, not a control-flow problem. The
kernel body was never the issue — it is the same `refine()` in both versions.
What changed is which element each lane was handed, and that one decision moved
the grid from paying `max(work)` to paying `mean(work)`. The bound on that
technique is `max/mean` and you can compute it before writing a line of code,
which means you can decide whether the optimization is worth attempting before
attempting it. The second half of the lesson is that the reordering is not free:
always report the break-even launch count alongside the speedup, because a 1.58x
kernel speedup that costs 67 launches of setup is a regression for anyone who
launches once.

**Variation to try.** Change the work distribution from uniform on 1..8 to
heavily skewed — say `work[i] = 1` for 95% of elements and 8 for the rest. Now
`mean` is about 1.35 and `max` is 8, so the model predicts a 5.9x speedup, and
the naive kernel is paying almost 6x for a handful of stragglers. Predict the
new break-even launch count before you measure it. Then make the spread
continuous instead of 8-valued (`work[i] = 1 + (hash(i) % 64)`) and watch the
counting sort stop being obviously the right tool.
