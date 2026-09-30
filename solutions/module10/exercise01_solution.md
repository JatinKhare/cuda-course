# Module 10 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

---

## Compile / run

```
nvcc -arch=sm_89 -O3 -lineinfo -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
compute-sanitizer --tool racecheck .\exercise01_solution.exe --racecheck
```

`-lineinfo` is what makes `racecheck` name source lines instead of raw
instruction offsets. Without it the report is still correct and far less
useful. Do **not** use `-G`: it changes the schedule enough to alter which
hazards appear, and this module is about the optimized code.

There are **four** defects behind the **three** symptoms, in **three**
distinct classes. That mismatch is the exercise.

---

## TODO 1 — what `racecheck` actually reports

Real output, verbatim (function signature elided for width):

```
========= COMPUTE-SANITIZER
========= Error: Race reported between Read access at triage_broken(...)+0x200 in exercise01.cu:119
=========     and Write access at triage_broken(...)+0x80 in exercise01.cu:113 [1052 hazards]
=========     and Write access at triage_broken(...)+0x250 in exercise01.cu:119 [5384 hazards]
=========
========= Error: Race reported between Write access at triage_broken(...)+0x250 in exercise01.cu:119
=========     and Write access at triage_broken(...)+0x80 in exercise01.cu:113 [1 hazards]
=========     and Write access at triage_broken(...)+0x250 in exercise01.cu:119 [503 hazards]
=========
racecheck mode: one launch of triage_broken, done.
========= RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)
```

So:

| prediction | answer |
|---|---|
| `P_HAZARDS` (Error blocks) | **2** |
| `P_CHECKSUM` (S2 caught) | **0** |
| `P_SLOTS` (S3 caught) | **0** |
| `P_CLASSES` (distinct lines named) | **2** — 113 and 119 |

Line 113 is `if (threadIdx.x < NCLASS) sCount[threadIdx.x] = 0u;` and line
119 is `sCount[c] += 1u;`. Both are **shared memory**. Nothing else is
reported.

The hazard *counts* (1052, 5384, 1, 503) vary run to run. The number of
`Error:` blocks and the lines named do not. That is why the exercise asks for
the latter.

### The blind spot, and why it is not a bug in the tool

From `compute-sanitizer --help`:

```
  --tool arg (=memcheck)                Set the tool to use.
                                        racecheck : Shared memory hazard checking
```

The tool documents itself as a **shared memory** hazard checker. It
instruments `LDS`/`STS`/`ATOMS` and tracks, per shared-memory byte, which
threads touched it between barriers. It does not instrument global memory.

The reason is not laziness. Shared memory is per-block, bounded (≤100 KB on
Ada), and its synchronization domain is a barrier that the tool can see. A
global-memory race detector would have to track ownership of every byte of a
12 GB address space against a partial order induced by fences, launches and
stream dependencies across 40 SMs and an unbounded number of blocks. That is
a different and much larger problem.

Module 9 used `racecheck` on eight barrier fragments and it found every
single hazard, including one on a kernel that produced bit-exact correct
output. That is a fair summary of the tool *within its domain*, and it is
worth stating plainly that this exercise is not a correction of Module 9 but
an extension of it: the domain has an edge, and the edge is the word
"shared".

**Practical consequence, and it is the takeaway of this exercise:** a clean
`racecheck` report does not mean you have no races. It means you have no
*shared-memory* races. Both of this exercise's global RMW defects — the
checksum and the slot allocator — are invisible to it, and they are the two
that produced the most spectacular symptoms.

Equally, `--tool memcheck` is clean on the broken program throughout. Every
access is in bounds; the values are just wrong. Module 4 already noted that
`compute-sanitizer` does not see local-memory spills; this is the second
blind spot in the same tool, and the pattern is the same — it detects
*illegal* accesses and *shared-memory* hazards, not *incorrect results*.

**How you find the global races instead.** By reasoning, and by the
signature: a global counter that is 4 orders of magnitude low and varies run
to run is an unprotected RMW, always. `grep` your kernels for `+=`, `++`,
`-=` and `= *p` on any pointer that more than one block can reach.

---

## TODO 2 — two counting defects, two different fixes

The requirement was "any two threads that classify into the same class must
both be counted — same block or different blocks". There are two places that
fails.

**2a, inside the block** (line 119 in the shipped file):

```cpp
sCount[c] += 1u;                    // broken
atomicAdd(&sCount[c], 1u);          // fixed
```

A shared-memory RMW. `racecheck` names this one. Why `ATOMS` and not a
barrier: the threads that collide here are *within* the block, so they are
already "synchronized" in the only sense a barrier offers, and it changes
nothing. Module 10's whole point.

**2b, across blocks** (the flush at the end):

```cpp
gCount[threadIdx.x] += sCount[threadIdx.x];             // broken
atomicAdd(&gCount[threadIdx.x], sCount[threadIdx.x]);   // fixed
```

A **global** RMW, executed by 8 threads of each of 15,625 blocks against 8
addresses. `racecheck` says nothing about it. It is the reason the per-class
counts were low even in the runs where the shared counter happened to behave.

Note that this one is warp-uniform-ish in the worst way — 8 lanes of warp 0
of every block hitting 8 global addresses — which is exactly the contention
pattern the lesson's Part C table measures as expensive. It is correct here
because the flush runs once per block, not once per element. That is
privatization; Exercise 2 makes you build it deliberately.

**Wrong approaches and their symptoms:**

| approach | symptom |
|---|---|
| Fix 2a only | Counts still low and still varying — the flush still races. Easy to mistake for "the atomic didn't work". |
| Fix 2b only | `racecheck` still fires; counts still wrong. |
| Add `__threadfence()` before the flush | No change whatsoever. Visibility was never the problem. |
| Make `gCount` `volatile` | No change, and now slower. Module 9: `volatile` is not synchronization. |

---

## TODO 3 — the missing barrier, and why the other one was fine

```cpp
if (threadIdx.x < NCLASS) sCount[threadIdx.x] = 0u;
__syncthreads();                    // <-- THIS was missing
```

The shipped code zeroes the private counters and then immediately starts
adding to them, with no barrier in between. Warp 0's threads 0–7 do the
zeroing; warps 1–7 are free to run ahead and increment a bin that is about to
be overwritten with 0. That is the `113 ↔ 119` hazard in the racecheck
report: a Write at 113 racing a Read and a Write at 119.

The barrier *after* the accumulation was already present and is also
necessary — for the opposite reason, so that the flush does not read a
counter another warp has not finished writing. **Two barriers, two different
jobs:**

| barrier | guarantees |
|---|---|
| after the zeroing | no thread adds to a bin before it is 0 |
| before the flush | no thread reads a bin before every add landed |

A single barrier cannot do both, because they bracket different intervals.

One subtlety worth noticing: the existing barrier sits **outside** the
`if (i < n) { ... }` block, not inside it and not after a `return`. If the
guard had been written `if (i >= n) return;` the barrier would be in
divergent control flow, some threads would never arrive, and the kernel would
hang — Module 9's rule. The shipped code got this right, which is a small
piece of misdirection.

---

## TODO 4 — the checksum

```cpp
*gChecksum += (unsigned long long)v;              // broken
atomicAdd(gChecksum, (unsigned long long)v);      // fixed
```

A global RMW on **one address** from 4,000,000 threads — the most contended
pattern that exists. `racecheck` does not see it. Observed broken value
23,196,126,914 against a true 1,073,378,141,171,872: **low by a factor of
46,000**, and different every run.

`atomicAdd` on `unsigned long long` is supported from sm_35 and is exact.

Two tempting non-fixes:

- **`double` accumulation with `atomicAdd(double*)`** (sm_60+). Works, is
  slower, and is *not reproducible* — the ordering of the float adds changes
  every run, so the checksum would stop being a checksum. Example 1 Part E.
- **Two 32-bit atomics, one for low and one for high.** Not equivalent: you
  cannot carry across two independent atomics. Silently wrong.

---

## TODO 5 — the slot allocator, and the requirement that pulls both ways

Broken:

```cpp
int slot = *gCritCount;
*gCritCount = slot + 1;
if (slot < CRIT_CAP) gCritList[slot] = i;
```

This is a hand-written, non-atomic `atomicAdd`. Many threads read the same
`slot`, so many threads write the same list entry, and the counter advances
once per issue window rather than once per thread. Observed: reported count
701 against a true 120,149, with 3,367 of the 4,096 list slots never written
— *below* the reported count, which is the giveaway that the counter and the
writes disagree.

Fixed:

```cpp
int slot = atomicAdd(gCritCount, 1);
if (slot < CRIT_CAP) gCritList[slot] = i;
```

**Why this satisfies both halves of the requirement.** The requirement was:
(a) no write past `CRIT_CAP`, and (b) `*gCritCount` must end up holding the
*true* count even when it exceeds the cap. Those pull in opposite
directions — (a) wants the counter to stop, (b) wants it to keep going.

The resolution is that the counter and the capacity are **separate
concerns**. Let the counter run free and count everything; use its return
value as a *conditional* index. Every critical event increments; only the
first `CRIT_CAP` of them store. The caller reads 120,149 and knows the list
holds 4,096 of them, so it can decide what to do about the other 116,053.

**The wrong fix that looks obviously right:**

```cpp
if (*gCritCount < CRIT_CAP) {            // WRONG
    int slot = atomicAdd(gCritCount, 1);
    gCritList[slot] = i;
}
```

The test and the increment are not one operation. Many threads can pass the
test while the counter is at `CRIT_CAP - 1` and then all increment, so `slot`
exceeds the capacity and you write out of bounds. This one *does* get caught
— by `compute-sanitizer --tool memcheck`, as an invalid global write — which
makes it a useful illustration that the two tools catch disjoint things.

**The other wrong fix**, clamping with `atomicMin(gCritCount, CRIT_CAP)` after
the fact, destroys requirement (b): you can no longer tell overflow from an
exact fill.

---

## Synchronization / memory reasoning

The four defects span the three classes this module distinguishes:

| defect | class | scope | detected by |
|---|---|---|---|
| `sCount[c] += 1` | shared RMW | block | `racecheck` |
| missing barrier after zeroing | missing ordering | block | `racecheck` |
| `*gChecksum += v` | global RMW | device | nothing — reasoning only |
| hand-rolled slot allocation | global RMW | device | nothing — reasoning only |

The second row is the only one a barrier fixes, and it is the only one that
is *not* an atomicity problem. That is the module's central distinction in
table form: if the defect is "two operations happened in the wrong order",
you need Module 9's tools; if it is "two operations interleaved inside one
update", you need Module 10's.

---

## Performance reasoning

The fixed kernel is not noticeably slower than the broken one, which is worth
understanding rather than being relieved about.

- `atomicAdd(&sCount[c], 1u)` compiles to `ATOMS` and costs roughly what a
  shared store costs (lesson: 48× cheaper than the equivalent global atomic).
- The two global atomics (`gChecksum`, `gCritCount`) are on single addresses
  and *are* expensive — this is the K=1 row of the contention table — but
  they are 4 M and 120 k operations respectively against a kernel that also
  reads 16 MB of input.
- Every atomic in `triage_fixed` discards its return value except the slot
  allocator, so the compiler emits `RED` for three of the four and `ATOMG`
  only where the ticket is actually consumed. Real SASS:

```
Function : _Z12triage_fixedPKjiPjPyPiS3_
    RED.E.ADD.64.STRONG.GPU  [R8.64], R6 ;      // atomicAdd(gChecksum, v)
    ATOMS.POPC.INC.32 RZ, [R3.X4+URZ] ;         // atomicAdd(&sCount[c], 1u)
@P1 ATOMG.E.ADD.STRONG.GPU PT, R3, [R2.64], R7 ;// slot = atomicAdd(gCritCount,1)
    RED.E.ADD.STRONG.GPU [R2.64], R5 ;          // atomicAdd(&gCount[t], ...)
```

  Three things to notice. The shared counter became a single
  `ATOMS.POPC.INC.32` — the hardware counts the colliding lanes of the warp
  with a population count and applies one increment. The one `ATOMG` is
  predicated (`@P1`) because the compiler warp-aggregated it: one lane
  performs the atomic for the whole warp and the others receive their
  tickets by shuffle. And `RED.E.ADD.64` confirms the checksum really is a
  single 64-bit atomic, not a pair of 32-bit ones.

If you wanted the checksum faster, the move is Exercise 2's: reduce within
the block first, then one global atomic per block. 4,000,000 → 15,625.

---

## Expected output

Observed on the RTX 3500 Ada (CUDA 13.2). The `triage_broken` numbers change
every run; the `triage_fixed` numbers never do, which is the point.

```
Device: NVIDIA RTX 3500 Ada Generation Laptop GPU (sm_89, 40 SMs)

=== triage_broken ===
  class counts : 981 984 984 983 981 985 984 809
  reference    : 553450 555635 555019 554029 554583 555011 552124 120149
  sum of counts: 7691 (should be 4000000)
  checksum     : 23196126914 (should be 1073378141171872)
  crit count   : 701 (should be 120149, cap 4096)
  crit list    : 3367 unwritten slots, no duplicates, 0 non-critical entries

  results identical across 3 runs: no

=== triage_fixed  ===
  class counts : 553450 555635 555019 554029 554583 555011 552124 120149
  reference    : 553450 555635 555019 554029 554583 555011 552124 120149
  sum of counts: 4000000 (should be 4000000)
  checksum     : 1073378141171872 (should be 1073378141171872)
  crit count   : 120149 (should be 120149, cap 4096)
  crit list    : 0 unwritten slots, no duplicates, 0 non-critical entries

  [PASS] per-class counts exact
  [PASS] counts sum to N
  [PASS] checksum exact
  [PASS] gCritCount reports the TRUE critical count (120149)
  [PASS] crit list: 4096 slots filled, unique, all critical
  [PASS] identical results across 3 runs

=== TODO 1 prediction ===
  you predicted: hazards=2  S2caught=0  S3caught=0  lines=2
  [PASS] prediction (all four must be right; no partial credit)

correctness score: 6/6
OVERALL: PASS
```

The broken counts hover around 700–1050 per class across runs — that is
roughly one surviving increment per *issue window*, not per thread.

---

## The result that matters

Three of the four defects are the same bug — an unprotected read-modify-write
— wearing three different costumes: a `+=` on shared memory, a `+=` on global
memory, and a hand-written index allocator that never looks like a `+=` at
all. Only the shared-memory one is visible to `compute-sanitizer --tool
racecheck`, which by its own documentation is a *shared memory* hazard
checker, so the two defects with the most dramatic symptoms are precisely the
ones no tool will hand you. The discipline that replaces the tool is a
pattern-match on the *symptom*: catastrophically low, run-to-run varying, no
fault raised, memcheck clean.

**Variation to try:** delete the `atomicAdd` on `gCritCount` and replace it
with `atomicInc(gCritCount, CRIT_CAP - 1)`, then run the harness. You get a
dense, unique, fully-populated list and a count that is silently wrong
(it wraps), which is a far nastier failure than the one you started with —
and a concrete demonstration of why `atomicInc` is a ring-buffer instruction
and not a counter.
