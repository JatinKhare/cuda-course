# Module 08 / Exercise 01 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe

nvcc -arch=sm_89 -O3 -c -o exercise01_solution.o exercise01_solution.cu
cuobjdump -sass exercise01_solution.o > exercise01_solution.sass
```

One block, one warp, 32 lanes. `lane == threadIdx.x` throughout, so every mask
can be computed on paper directly from the conditions.

Notation: `bit L` of the mask is lane `L`, so lane 0 is the least significant
bit and lane 31 the most significant. Writing a lane set as hex is the step
most people get wrong the first time — `lanes 0..19` is `0x000fffff`, not
`0xfffff000`.

---

## TODO 1 — the `lane < 20` sub-tree

```cpp
static const unsigned P0  = 0xffffffffu;
static const unsigned P1  = 0x000fffffu;
static const unsigned P2  = 0x00011111u;
static const unsigned P3  = 0x000eeeeeu;
static const unsigned P4  = 0x000fffffu;
```

- **P0 = `0xffffffff`.** Kernel entry. The block is 32 threads, so the warp is
  full; nothing has diverged yet.
- **P1 = `0x000fffff`.** Lanes 0..19, i.e. 20 bits set. `2^20 - 1 = 0xfffff`.
- **P2 = `0x00011111`.** Inside `if ((lane & 3) == 0)`, restricted to the lanes
  that reached it, i.e. lanes in 0..19 that are multiples of 4: 0, 4, 8, 12, 16.
  Five bits: `1 | 1<<4 | 1<<8 | 1<<12 | 1<<16 = 0x00011111`. `popc = 5`.
- **P3 = `0x000eeeee`.** The complement within P1: `P1 ^ P2`. `popc = 15`.
- **P4 = `0x000fffff`.** The inner branch's immediate post-dominator. The
  compiler placed a reconvergence point here, so the mask is `P2 | P3 == P1`
  again. Lanes 20..31 are still elsewhere — they never entered this arm.

The check that catches arithmetic slips: `P2 | P3 == P1` and `P2 & P3 == 0`.
The two arms of a branch always *partition* the mask at the branch.

**Common wrong answers.** `P2 = 0x11111111` — forgetting that lanes 20, 24, 28
are also multiples of 4 but did not reach this point. The symptom is a MISMATCH
with `popc` 8 versus 5, and it is the same class of error as forgetting the
bounds guard in Module 3: you reasoned about the condition and not about who
arrived.

---

## TODO 2 — the `else` sub-tree, and the early `return`

```cpp
static const unsigned P5  = 0xfff00000u;
static const unsigned P6  = 0xf0000000u;
static const unsigned P7  = 0x0ff00000u;
static const unsigned P8  = 0x0fffffffu;
```

- **P5 = `0xfff00000`.** Lanes 20..31: 12 bits, at the top. Complement of P1.
- **P6 = `0xf0000000`.** Inside `if (lane >= 28)`: lanes 28..31.
- **P7 = `0x0ff00000`.** The lanes that reached the `else` arm and did *not*
  take the early return: lanes 20..27. `P6 | P7 == P5`.
- **P8 = `0x0fffffff`.** **This is the one that catches people.**

P8 is the first instruction after the whole outer `if/else`. The instinctive
answer is `0xffffffff`: "the branch is over, everyone is back". It is not,
because lanes 28..31 executed `return` and are **finished**. An exited thread is
not a candidate for any future active mask. So P8 is "all 32 lanes, minus lanes
28..31" = `0x0fffffff`, `popc = 28`.

Stated as a rule: the mask at a join point is the union of the masks of the
paths that *arrive there*, and a path that ends in `return` (or `EXIT` in SASS)
arrives nowhere.

**Common wrong answers.**
- `0xffffffff` — forgetting the `return`. This is the intended trap.
- `0x0ff00000` — forgetting that the `lane < 20` side also rejoins here.
- `0x000fffff` — the mirror image of the previous one.

---

## TODO 3 — the one-instruction `if/else`

```cpp
static const unsigned P9  = 0x0aaaaaaau;    // odd lanes among 0..27
static const unsigned P10 = 0x05555555u;    // even lanes among 0..27
```

Two things have to be right simultaneously.

1. `lane & 1` selects odd lanes for P9 and even lanes for P10 — `0xaaaaaaaa`
   and `0x55555555` in isolation.
2. Only lanes 0..27 are still alive, from TODO 2. Mask both with `0x0fffffff`:
   `0xaaaaaaaa & 0x0fffffff = 0x0aaaaaaa` (14 bits, lanes 1,3,…,27) and
   `0x55555555 & 0x0fffffff = 0x05555555` (14 bits, lanes 0,2,…,26).

Both have `popc = 14`, not 16. `P9 | P10 == P8`.

**The deeper point, and why the TODO asks you to look at the SASS.** A body of
one arithmetic instruction is well inside the compiler's if-conversion
threshold, so a reader who has understood predication expects "no branch is
emitted, therefore all 32 lanes execute the instruction, therefore the mask is
`0xffffffff`". That reasoning is wrong, and it is worth seeing why.

`__activemask()` compiles to `VOTE.ANY Rd, PT, PT`. When the region is
predicated, that `VOTE` is predicated too:

```
ISETP.NE.U32.AND P0, PT, R10, 0x1, PT ;
@!P0 VOTE.ANY R17, PT, PT ;
@P0  VOTE.ANY R13, PT, PT ;
```

A predicated-off lane does not execute the instruction, so it is not counted in
the vote. **Predication and branching produce the same active mask.**
`__activemask()` is therefore useless as a way to tell them apart — it measures
participation, not control-flow shape. Only the disassembly distinguishes them.

(In the *solution* build, as it happens, the recording macro is heavy enough —
a vote, a `FLO`, a compare, two global stores — that the compiler emits a real
`@P0 BRA` for these arms. You can see it in the SASS. The predicted masks are
identical either way, which is exactly the lesson.)

---

## TODO 4 — the variable-trip loop

```cpp
static const unsigned IT[MAXIT] = {
    0x0fffffffu,   // j = 0
    0x0efefefeu,   // j = 1
    0x0cfcfcfcu,   // j = 2
    0x08f8f8f8u,   // j = 3
    0x00f0f0f0u,   // j = 4
    0x00e0e0e0u,   // j = 5
    0x00c0c0c0u,   // j = 6
    0x00808080u    // j = 7
};
```

`trips = (lane & 7) + 1`, so lane `L` runs `(L mod 8) + 1` iterations. Lane `L`
is still in the loop on iteration `j` iff `j < (L mod 8) + 1`, i.e. iff
`(L mod 8) >= j`. Intersect with the surviving lanes `0x0fffffff`.

Derivation for `j = 1`: drop every lane with `L mod 8 == 0`, i.e. lanes 0, 8,
16, 24. Starting from `0x0fffffff` and clearing bits 0, 8, 16, 24 gives
`0x0efefefe`. `popc` falls 28, 24, 20, 16, 12, 9, 6, 3.

(The counts are not a clean multiple of 4 at the end because lanes 28..31 left
earlier: only three lanes have `L mod 8 == 7` among 0..27, namely 7, 15, 23.)

**The whole point of this table.** The warp issues the loop body **8** times.
The mean trip count among the 28 live lanes is about 4.4. The warp is doing
roughly 1.8x the issues its useful work requires, and there is no instant at
which any lane is "skipped ahead" — a lane that has finished sits in the warp
with its bit clear while the warp keeps issuing. This is the shape that
Exercise 2 attacks.

---

## TODO 5 — the issue-count model

```cpp
long long total = 0;
for (int p = 0; p < nSites; ++p) {
    int mx = 0;
    for (int l = 0; l < nLanes; ++l) {
        int c = laneCount[l * nSites + p];
        if (c > mx) mx = c;
    }
    issueOut[p] = (long long)mx;
    total += (long long)mx;
}
return total;
```

**Why `max` and not `sum` or `mean`.** The SIMT contract says one instruction
issue drives every *active* lane of the warp. If 28 lanes each execute a site
once, that is one issue with 28 bits set — not 28 issues. If lane 7 executes a
loop body 8 times and lane 0 executes it once, the warp must issue that body 8
times, and lane 0 is masked off for seven of them. So the number of issues at a
site is the number of *rounds*, which is the maximum per-lane count.

This is the whole cost model of divergence in one line. `sum(counts)` is the
*useful lane-work*; `32 * max(counts)` is the *lane-capacity consumed*; the
ratio is the efficiency of the warp at that site.

**Common wrong approaches.**
- `sum` over lanes — gives 19 × (lanes that ran) and mismatches everywhere. The
  symptom is a model total in the hundreds against a hardware total of 19.
- `count of lanes that ran > 0` — gives 1 for every site including the loop,
  total 12 instead of 19. This is right for straight-line sites and wrong for
  loops, which is the whole content of the question.
- Using `nSites` inconsistently with the row stride. The table is
  `laneCount[lane * (NSITE+1) + site]` and `nSites == NSITE+1` is passed in;
  indexing with `NSITE` instead reads the wrong column and fails at the loop
  site only, which is a confusing symptom.

**How the hardware ground truth is obtained.** Each site also does

```cpp
unsigned m_ = __activemask();
if (lane == __ffs((int)m_) - 1) issue[p] += 1;
```

The lowest set bit of the current mask names one lane, so exactly one lane per
issue performs the increment. There is one warp in the grid, so a plain
non-atomic `+= 1` is race-free — which is why this exercise is allowed to
measure issue counts without touching atomics (Module 10).

---

## Synchronization / memory reasoning

There is none, deliberately. One block of 32 threads, no shared memory, no
`__syncthreads()`, no inter-thread communication of any kind. Every value this
exercise reports is a property of a single warp's control flow, so the only
hazard would be the issue counter, which is protected by the "exactly one lane
per issue" construction above rather than by synchronization.

The `__ballot_sync`/`__activemask` calls do not create synchronization either.
`__activemask()` in particular has no synchronizing effect — it reports, it does
not converge. (`__syncwarp()`, which does converge, is Exercise 3's subject.)

---

## Performance reasoning

This exercise is not timed and should not be. One warp on one SM measures the
divergence *structure*, not its cost; `example02.cu` measures the cost. But the
structure is what predicts the cost:

- 12 sites, 19 issues. If the warp had never diverged, one pass through the
  kernel would have issued each site it reaches once and the loop
  `trips` times — 6 straight-line sites plus `trips` for a uniform trip count.
- The *lane-work* actually performed is
  `32 + 20 + 5 + 15 + 20 + 12 + 4 + 8 + 28 + 14 + 14 = 172` lane-executions at
  the eleven straight-line sites, plus `118` in the loop (three full groups of
  eight lanes contributing `1+2+…+8 = 36` each, and lanes 24..27 contributing
  `1+2+3+4 = 10`), for **290** in total — against `32 * 19 = 608` lane-slots
  consumed. **47.7% lane efficiency** for this control-flow shape.

That number — useful lane-executions over lane-slots consumed — is the single
figure of merit for divergence, and it is exactly what Nsight Compute reports as
average active threads per instruction (Module 23).

---

## Expected output

Actual output observed on the RTX 3500 Ada, CUDA 13.2, `-arch=sm_89 -O3`:

```
=== Module 8 / Exercise 1 : lane-level prediction ===
one block, one warp, 32 lanes

--- masks at the labelled sites ---
  site  predicted    actual       popc  lanes
  P0    0xffffffff   0xffffffff   32    32      MATCH
  P1    0x000fffff   0x000fffff   20    20      MATCH
  P2    0x00011111   0x00011111    5     5      MATCH
  P3    0x000eeeee   0x000eeeee   15    15      MATCH
  P4    0x000fffff   0x000fffff   20    20      MATCH
  P5    0xfff00000   0xfff00000   12    12      MATCH
  P6    0xf0000000   0xf0000000    4     4      MATCH
  P7    0x0ff00000   0x0ff00000    8     8      MATCH
  P8    0x0fffffff   0x0fffffff   28    28      MATCH
  P9    0x0aaaaaaa   0x0aaaaaaa   14    14      MATCH
  P10   0x05555555   0x05555555   14    14      MATCH

--- masks on each iteration of the variable-trip loop ---
  j=0   0x0fffffff   0x0fffffff   28     MATCH
  j=1   0x0efefefe   0x0efefefe   24     MATCH
  j=2   0x0cfcfcfc   0x0cfcfcfc   20     MATCH
  j=3   0x08f8f8f8   0x08f8f8f8   16     MATCH
  j=4   0x00f0f0f0   0x00f0f0f0   12     MATCH
  j=5   0x00e0e0e0   0x00e0e0e0    9     MATCH
  j=6   0x00c0c0c0   0x00c0c0c0    6     MATCH
  j=7   0x00808080   0x00808080    3     MATCH

--- instruction issues per site ---
  site   lanes that ran your model     hardware
  P0     32             1              1            MATCH
  P1     20             1              1            MATCH
  P2     5              1              1            MATCH
  P3     15             1              1            MATCH
  P4     20             1              1            MATCH
  P5     12             1              1            MATCH
  P6     4              1              1            MATCH
  P7     8              1              1            MATCH
  P8     28             1              1            MATCH
  P9     14             1              1            MATCH
  P10    14             1              1            MATCH
  loop   28             8              8            MATCH
  TOTAL                 19             19           MATCH

SCORE: 20/20
OVERALL: PASS
```

Every value here is deterministic; there is nothing to vary run to run.

---

## The result that matters

The active mask is not a metaphor — it is a 32-bit value you can read, and every
one of its bits is derivable on paper from the control flow, *provided you track
two things people forget*: lanes that never entered a region, and lanes that
have exited the kernel. Once you can do that, the cost model falls out
immediately, because the number of issues at a site is the maximum per-lane
execution count, and the lane efficiency is the useful lane-work divided by
`32 x issues`. That one ratio is what every divergence optimization is trying to
raise.

**Variation to try.** Change `trips` from `(lane & 7) + 1` to
`((lane >> 3) & 7) + 1` — the same eight trip counts, but now assigned in
8-lane blocks instead of interleaved. Predict the per-iteration masks and the
issue count before you run it. The issue count does not change (still 8), which
is the point: within a warp it does not matter *how* the trip counts are
arranged, only what the maximum is. Then work out what would have to be true for
the issue count to fall, and you have rediscovered Exercise 2.
