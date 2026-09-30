# Module 07 — Check Your Understanding, answers

---

## 1. Necessary and sufficient condition for `s[f(tid)]` to be conflict-free

Let `S = {f(0), …, f(31)}` as a **multiset** of element indices into a `float`
array, and let `D(x) = x % 32` be the bank.

**Condition.** The access is conflict-free if and only if

> for every bank `b`, the set `{ x ∈ S : x % 32 == b }` — taken as a set of
> *values*, with duplicates collapsed — has at most one element.

Equivalently, and more usefully: **no two lanes may request different indices
that are congruent mod 32.**

```
conflict-free  ⟺  ∀ i, j :  f(i) ≡ f(j) (mod 32)  ⟹  f(i) = f(j)
```

Written that way it is clearly an equivalence-class statement: the map
`x ↦ x mod 32` must be injective on the *distinct values* appearing in `S`.

**Why "the 32 values are distinct" is not sufficient.** `f(tid) = 32*tid` gives
32 pairwise distinct values — 0, 32, 64, …, 992 — and they are all congruent to
0 mod 32. That is the worst case the hardware admits, `D = 32`. Distinctness of
the *values* says nothing about distinctness of the *banks*.

**Why it is not necessary.** `f(tid) = tid/2` gives the multiset
{0,0,1,1,2,2,…,15,15}: only 16 distinct values, each repeated. It is
conflict-free, because every repetition is a repetition of the *same word*, and
the bank broadcasts. Collisions are harmless precisely when they are total.

The condition to internalise is therefore neither "all different" nor "all the
same" but the hybrid: **per bank, all the same**. The two familiar good cases
— `s[tid]` (all different, all different banks) and `s[0]` (all the same) — are
the two extremes of one rule.

A useful corollary for the common case `f(tid) = k*tid`: distinct lanes give
congruent-mod-32 values exactly when `32 | k*(i-j)`, so the degree is
`gcd(k, 32)`. Odd `k` is always free.

---

## 2. Padding a `[32][32]` tile to `[32][40]`

**The conflict degree is still 8-way, not 1.**

Element `(r, c)` now lives at flat index `40r + c`, so

```
bank(r, c) = (40r + c) % 32 = (8r + c) % 32
```

A column access holds `c` fixed and runs `r` over 0..31. The banks visited are
`(8r + c) % 32`, and since `gcd(8, 32) = 8`, the expression `8r mod 32` takes
only 4 distinct values (0, 8, 16, 24). Thirty-two lanes land in 4 banks, at 32
distinct words, so each of those banks must supply 8 distinct words:

```
D = 8
```

The fix improved the kernel by 4× and left three quarters of the penalty in
place — which is exactly the kind of "fix" that survives code review, because
the profile did get better.

**Why the alignment reasoning is pointed the wrong way.** "Multiple of 8" is a
statement about *byte alignment*: 8 floats = 32 bytes, which is the sector size
that matters in global memory. In shared memory there are no sectors, no cache
lines, and no 32-byte transactions to align to. The quantity that matters is the
**bank stride**, `pitch % 32`, and the requirement is the opposite of alignment:
you want the row stride to be **coprime with 32**, i.e. **odd**. Every property
that makes a pitch attractive for global-memory alignment — being a multiple of
4, 8, or 32 — makes it *worse* for shared-memory banking, because
`gcd(pitch, 32)` is exactly the conflict degree and every such pitch has a large
gcd.

The correct rule: `D = gcd(pitch, 32)` for a column read of a float tile, so
**`pitch` must be odd**. 33 works. 40 gives 8. 64 gives 32 — a padded tile that
is exactly as bad as the unpadded one.

**Relation to M5 Exercise 3.** The structural parallel is exact and the
arithmetic is opposite, which is why they belong side by side:

| | M5 ex3 (global) | M7 (shared) |
|---|---|---|
| granularity | 32 B sector | 4 B bank |
| what must differ per row | nothing — rows must *start aligned* | the starting **bank** |
| condition on pitch (floats) | `pitch % 8 == 0` | `gcd(pitch, 32) == 1` |
| the seductive wrong answer | 68 (`% 4 == 0`, "float4-legal") | 40 or 64 ("nicely aligned") |
| why it is wrong | `68*4 % 32 == 16`, so half the rows straddle | `gcd(40,32) = 8`, so the column hits 4 banks |

In both cases the mistake is the same mistake: applying an alignment rule from
one granularity to a problem governed by a different one. In both cases the
partial fix measures better than the original, which is what makes it dangerous.

---

## 3. `sd[tid]` (double) versus `s[tid]` (float)

**Predicted ratio: 1.0.** They cost the same.

From the phase model:

- `float`, 4 B: `32 × 4 = 128 B` per warp, which is exactly the width of the
  bank array. One phase, conflict-free, and the measured floor for any 32-lane
  4 B shared access is **2 pipeline cycles**.
- `double`, 8 B: `32 × 8 = 256 B` per warp, so the request splits into **2
  phases of 16 lanes**. Sixteen lanes reading 16 consecutive doubles occupy
  `16 × 8 = 128 B` — again exactly the bank array width, with each double
  straddling two adjacent banks and every bank used exactly once. Each phase is
  conflict-free, costs 1 cycle, so the instruction costs **2 cycles**.

Two cycles either way. Measured: `float s[tid]` 0.0590 ms, `double dd[tid]`
0.0685 ms — a 1.16× gap, all of which is the integer XOR the double kernel uses
instead of a floating-point add (Ada runs FP64 at 1/64 rate, so the benchmark
cannot use `+`). The shared-memory term is identical.

The deeper point: **the 4-byte access was not saturating the bank array.** It
asks for 128 B and gets 2 cycles of pipeline; it had a free cycle of slack,
which is also why a 2-way conflict on floats is free. Doubling the request size
consumes the slack rather than adding cost.

**The global-memory analogue, and why it differs.** The same substitution in
global memory — a warp reading 32 contiguous `double`s instead of 32 contiguous
`float`s — moves 256 B instead of 128 B, i.e. 8 sectors instead of 4 (M5). If
the kernel is DRAM-bandwidth-bound, it costs **exactly 2×**. There is no slack
to consume: DRAM bandwidth is a hard 432 GB/s and every byte you request is a
byte you pay for.

The distinction is between a resource priced in **bytes delivered** (global
memory, where the cost function is linear in bytes and the only lever is
requesting fewer) and one priced in **cycles of a fixed-width port** (shared
memory, where the cost function is a step function of the address pattern and
the lever is rearranging addresses). "Fewer bytes is faster" is a reliable
instinct in global memory and a misleading one in shared memory, where
`s[tid/2]` reads half the bytes of `s[tid]` at identical cost and `s[32*tid]`
reads the same bytes as `s[tid]` at 15× the cost.

---

## 4. A 16× penalty removed, 4 % gained

Both explanations below are common, and they are distinguishable by measurement.

**Explanation A — the shared-memory term was a small share of the kernel.**
Conflicts multiply the cost of *shared-memory instructions only*. If the kernel
spends most of its time on global loads, FFMAs, transcendentals, or barrier
waits, then even a 16× reduction on a 5 % slice yields
`1 / (0.95 + 0.05/16) = 1.049`, i.e. about 5 %. Amdahl, applied to an
instruction class rather than a code region. This is the ordinary case, and it
is why "I fixed the bank conflicts" is not by itself a performance claim.

**Explanation B — the kernel was never shared-memory-bound; it was latency- or
bandwidth-bound elsewhere, and the conflict replays were being hidden.** A
32-way conflict costs 31 extra cycles of LSU *occupancy*, but a warp stalled on
a 575-cycle global load (M4) is not issuing anything anyway. With enough
resident warps, the replays fill slots that would otherwise be idle. The
instruction got 16× cheaper and the kernel's critical resource — DRAM
bandwidth, or the dependent-load latency chain — did not change at all, so the
wall clock barely moved.

The two are genuinely different diagnoses with different next steps:
A says "the shared term is now small, go optimise something else";
B says "the shared term was never on the critical path, and you have just
learned the kernel is memory-bound."

**A measurement that distinguishes them.**

Run the kernel with the *conflicted* version but shrink the problem so the
global working set fits in L2 (< 48 MB, per M4/M5 sizing rules), or replace the
global loads with an index-derived constant so there is no global traffic at
all. Then re-measure the conflicted-versus-fixed ratio.

- Under **A**, the shared term is a genuine, small, additive part of the runtime.
  Removing the global traffic removes the *other* term, the shared term becomes
  dominant, and the fixed-versus-conflicted ratio jumps toward 16×.
- Under **B**, the conflict cycles were being overlapped with stalls. Removing
  the stalls exposes them too, so the ratio also grows — but the *absolute*
  runtime of the conflicted version falls far more than the fixed version's
  does, and the two converge from a different direction.

The cleaner discriminator, and the one to reach for first, is the profiler
(Module 23): compare `smsp__throughput.avg.pct_of_peak_sustained_elapsed` for
the LSU pipeline against `dram__throughput.avg.pct_of_peak_sustained_elapsed`
in the conflicted build. If the LSU is near 100 % and DRAM is not, the kernel
was shared-memory-bound and the 4 % is a surprise worth investigating further
(look for a barrier: a 16× faster load phase behind a `__syncthreads()` still
waits for the slowest warp, and Module 9 explains why that changes nothing).
If DRAM is near 100 %, you have explanation B and the 4 % is the whole story.

A third possibility worth ruling out, because it costs thirty seconds: check
that the counter really went to zero for the kernel you think it did. `ncu`
aggregates over all launches of a matching kernel name by default, and a
templated kernel with several instantiations will happily report the
conflict-free instantiation's zero while the conflicted one still runs. Use
`--kernel-name` with a precise regex and `--launch-count 1`.
