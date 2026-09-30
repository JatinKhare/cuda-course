# Module 10 — Check Your Understanding: answers

---

## Q1 — barriers around a shared-memory `+=`

**Why the colleague is wrong.**

`__syncthreads()` is a *barrier*: it establishes that every thread of the
block has arrived, and that every memory access issued before it is visible
to the block afterwards. Both of those are statements about the boundary
between "before the barrier" and "after the barrier". The race is not at that
boundary. It is *inside* one statement:

```
LDS   R0, [addr]      ;  <-- thread B's LDS can land here
IADD3 R0, R0, 1
STS   [addr], R0      ;  <-- and B's STS here
```

The two barriers sit outside those three instructions and do nothing to the
two gaps between them. **Ordering and visibility are properties of a sequence
of operations; atomicity is a property of a single operation.** No amount of
the former produces the latter.

Note also the scope error hiding in "all threads are synchronized": they are
synchronized *at that point in time*, not *for the duration of the
increment*. A barrier is an instant, not an interval. There is no CUDA
construct that gives you a critical section by holding threads apart in time;
the mechanism is `atomicAdd`, which removes the interval entirely.

**What value the counter holds.**

**1**, and it will be 1 essentially every run.

256 threads is 8 warps. Each warp executes the load as *one* instruction, so
all 32 of its lanes read the same value; it executes the store as one
instruction, so all 32 write the same value. A warp therefore contributes at
most 1, not 32 — the intra-warp losses are structural, not a scheduling
accident.

That leaves 8 warps, which could in principle give anything from 1 to 8. They
all belong to the same block, they were all released by the same barrier at
the same instant, and the counter is a single shared-memory word sitting in
L1/SMEM with ~40-cycle access. All 8 warps issue their loads within a few
cycles of one another, before any of them has stored. So they all read 0 and
all store 1.

`example01.cu` Part B measures exactly 1 on every trial, from 4 blocks of
256 threads — 32 warps, and still 1.

The general shape of the answer matters more than the number: **a racy counter
does not come out slightly low, it comes out catastrophically low.** If your
count is within a few percent of correct, you are looking at a different bug.

---

## Q2 — 1024 packed bins vs 32 bins 4 KB apart

**Prediction: P (1024 consecutive bins) is faster, by about 3×.**

Measured, for 2^20 atomics each:

| kernel | ms |
|---|---|
| P: 1024 consecutive `unsigned int` bins | 0.0132 |
| Q: 32 bins, 4 KB apart | 0.0423 |
| (for reference) 32 consecutive bins | 0.0868 |

**The mechanism.** The serializing resource is the ALU of the L2 slice that
owns the address, and the number of slices your addresses reach is the real
parallelism. Two separate effects are in play and they must not be conflated:

1. *How many distinct addresses.* P has 1024, Q has 32. This favours P by
   construction — 1024 addresses cannot be forced through fewer than 1024
   serialization chains.
2. *How well those addresses spread across slices.* Q's 4 KB spacing is
   chosen so that its 32 bins hash to 32 *different* slices; 32 consecutive
   bins do not, which is why plain 32-bin costs 0.0868 ms and spread 32-bin
   costs 0.0423 ms — a clean 2× for changing nothing but the spacing.

Effect 2 buys Q a factor of ~2. Effect 1 costs it a factor of ~6.6 against
P. Net: P wins by ~3.2×. **Spreading is a multiplier on the parallelism you
have; it does not create parallelism you do not have.** 32 addresses on 32
slices is still 32-way, and 32-way is not enough for 2^20 requests.

**If Q's bins were 32 bytes apart instead of 4 KB**, Q gets *worse*, not
better — measured 0.3416 ms, eight times slower than the 4 KB version and
26× slower than P. Two things go wrong at once:

- 32 B is one sector. Neighbouring bins one sector apart still hash to the
  same or adjacent L2 slices, so the spreading benefit evaporates.
- Unlike the packed case, where all 32 bins lived inside one or two sectors,
  each bin now occupies its own sector, so the same number of requests touch
  32 sectors instead of 2 and the slice must do more fill work per request.

That is why the stride-8 (32 B) column in the lesson's Part B table is the
*worst* column at small K — worse than both the packed layout and the widely
spread one. Spreading helps only once you clear the slice-hash granularity;
below it you pay the cost of scattering without buying the parallelism.

---

## Q3 — privatization made it slower

Two distinct mechanisms, and how to tell them apart.

**Mechanism A — there was no contention to remove.**

If the bins were already spread widely enough that the naive global atomics
were running at roughly plain-store cost (the lesson's table: atomic/store
ratio 1.0 at K ≥ 4096), then privatization removes nothing and adds
everything: `bins` shared writes per block to zero the copy, a barrier, a
second barrier, and up to `grid × bins` global atomics to flush — which can
easily exceed the original `n`. Measured in the lesson: 0.22× at K = 4096.

**Mechanism B — the private copy cost you occupancy.**

A per-block copy of `bins` words is shared memory, and shared memory is a
resident-block resource. Ada has 100 KB of addressable shared memory per SM.
Asking for 32 KB per block caps you at 3 blocks/SM; asking for 48 KB caps you
at 2. Module 19 does the arithmetic properly, but the effect is that the
kernel now has far fewer resident warps with which to hide the (unchanged)
DRAM latency of reading the input, and a memory-bound kernel that loses
occupancy loses throughput even though every individual operation got
cheaper.

**The distinguishing measurements.**

- *For A:* compute `grid × bins` and compare it to `n`. If `grid × bins` is
  within an order of magnitude of `n`, the flush is the whole problem. A
  second confirmation: re-run the **naive** kernel with all bins forced to a
  single index. If it barely slows down, there was no contention to begin
  with — mechanism A.
- *For B:* re-run your privatized kernel with the shared allocation padded to
  a larger size but the same number of bins actually used (or simply halve
  the block count and double the work per thread). If the time tracks the
  *allocation* rather than the *work*, you are occupancy-limited — mechanism
  B. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` reports the resident
  block count directly and will show the cliff.

They can also co-occur, and the two fixes are opposite in direction: A is
fixed by using *fewer* blocks (shrinking `grid × bins`), B by using *smaller*
private copies. Deciding which dominates before changing anything is the
point of the question.

---

## Q4 — the broken CAS loop

```cpp
unsigned old = *p;
do {
    unsigned desired = f(old);
    old = atomicCAS(p, old, desired);
} while (old != desired);
```

**What is wrong.** `atomicCAS` returns **the value that was in `*p` before
the call**, in both the success and the failure case. It is not a success
flag. The loop must exit when the returned value equals what we *compared
against* (`assumed`), because that is what "the swap happened" means. This
loop instead exits when the returned value equals what we *wrote*
(`desired`), which is a different proposition entirely: it exits whenever the
word already happened to contain the value we were about to write, whether or
not we wrote it.

**An interleaving that terminates having performed no swap.**

Let `f(x) = x + 1`, `*p == 5`, threads A and B.

| step | A | B | `*p` |
|---|---|---|---|
| 1 | `old = *p` → 5 | | 5 |
| 2 | | `old = *p` → 5 | 5 |
| 3 | | `desired = 6`; `atomicCAS(p, 5, 6)` → returns 5, writes 6 | 6 |
| 4 | | `5 != 6`, loop again: `desired = f(5) = 6`; `atomicCAS(p, 5, 6)` → **fails** (p is 6), returns 6 | 6 |
| 5 | | `old (6) != desired (6)` is false → **exit** | 6 |
| 6 | `desired = 6`; `atomicCAS(p, 5, 6)` → fails, returns 6 | | 6 |
| 7 | `old (6) != desired (6)` is false → **exit** | | 6 |

Two threads, each of which believes it incremented. Final value 6. One
increment was silently dropped — and note that B dropped one of its *own*
retries at step 5, so the bug is not merely "A and B collided": a single
thread can conclude it succeeded when it did not.

**What it looks like from outside.** Exactly like the plain RMW race this
module opened with — an undercount that varies run to run — except that the
source code now contains `atomicCAS` and therefore *looks* protected, which
makes it far harder to find. `compute-sanitizer --tool racecheck` will not
flag it either: every access to `*p` genuinely is atomic. The tool checks for
unsynchronized accesses, not for incorrect synchronization protocols.

**The correct form:**

```cpp
unsigned old = *p, assumed;
do {
    assumed = old;
    old = atomicCAS(p, assumed, f(assumed));
} while (assumed != old);
```

Two properties to verify every time you write one: the comparison operand and
the loop test use **the same variable** (`assumed`), and the new value is
recomputed **from `assumed`**, not from the stale pre-loop read. A third,
easy to forget: if the update is conditional (`only if my value is larger`),
that condition must be re-tested after every failed CAS, and a thread whose
candidate does not win must leave the loop rather than retry — otherwise a
tie makes it spin forever. Exercise 3's dataset C exists to catch that.
