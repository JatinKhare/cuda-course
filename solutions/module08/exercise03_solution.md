# Module 08 / Exercise 03 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe

compute-sanitizer --tool memcheck  .\exercise03_solution.exe
compute-sanitizer --tool racecheck .\exercise03_solution.exe

nvcc -arch=sm_89 -O3 -c -o exercise03_solution.o exercise03_solution.cu
cuobjdump -sass exercise03_solution.o > exercise03_solution.sass
```

---

## The bug

```cpp
volatile __shared__ int s[BLOCK];
s[tid] = in[gid];
__syncthreads();

for (int step = 0; step < STEPS; ++step) {
    if (lane & 1) {
        for (int j = 0; j < 300; ++j) t = fmaf(t, 1.00001f, 1e-7f);
        int v = s[base + ((lane + 1) & 31)];      // read
        s[tid] = v + 1;                           // write
    } else {
        int v = s[base + ((lane + 1) & 31)];      // read
        s[tid] = v + 1;                           // write
    }
}
```

The intended semantics is a synchronous rotation: on each step, *all* lanes read
their right neighbour's **current** value, and only then does anybody write. The
author believed this happened automatically because "the 32 lanes of a warp are
in lockstep."

The exchange is inside the two arms of a divergent `if`. The warp cannot execute
both arms at once, so it executes one and then the other. Whichever arm runs
second reads values that the first arm has **already overwritten**. Lane `L`
(even) reads `s[L+1]`, which belongs to an odd lane; lane `L` (odd) reads
`s[L+1]`, which belongs to an even lane. Whichever group goes second is reading
the other group's *new* value, not its old one.

The result is wrong for **100%** of the output, deterministically.

Note carefully what this is *not*:

- It is not an out-of-bounds access. `compute-sanitizer --tool memcheck` is
  clean (the only line it emits is a `cudaDeviceReset` warning, which is an
  artifact of the harness, not a memory error).
- It is not a missing `__syncthreads()`. The block-level barrier that publishes
  the initial load is present and correct, and adding more of them would not
  help — indeed a `__syncthreads()` placed inside either arm would be a barrier
  reached by only part of the block, which is itself undefined (Module 9).
- It is not a bank conflict. Every lane reads a distinct bank.
- It is not fixed by `volatile`. See TODO 4a.

### Seeing it in the SASS

```
BSSY B0, 0x1510 ;
STS  [R4.X4], R9 ;              <- the initial store
BAR.SYNC.DEFER_BLOCKING 0x0 ;   <- __syncthreads()
@!P0 BRA 0x1490 ;               <- split on (lane & 1)
     LDS R8, [R5.X4] ;          <- odd arm: read neighbour
     ... 300 FFMAs ...
     STS [R4.X4], R9 ;          <- odd arm: write
     BRA 0x1500 ;
     LDS R6, [R5.X4] ;          <- even arm: read neighbour
     STS [R4.X4], R9 ;          <- even arm: write
BSYNC B0 ;                      <- reconverge, next step
```

Two `LDS`/`STS` pairs in two separate basic blocks, with nothing between them.
The second block's `LDS` executes after the first block's `STS` has retired.
The bug is visible in the instruction stream without running anything.

### `compute-sanitizer --tool racecheck`

Racecheck finds it, and is the right tool:

```
========= Warning: Race reported between Read access at warpSmoothBroken+0x7600 in san.cu:18
=========     and Write access at warpSmoothBroken+0x63d0 in san.cu:16 [2048 hazards]
=========     and Write access at warpSmoothBroken+0x7740 in san.cu:16 [2048 hazards]
=========
========= Warning: Race reported between Write access at warpSmoothBroken+0x7620 in san.cu:18
=========     and Read access at warpSmoothBroken+0x6300 in san.cu:16 [2048 hazards]
...
========= RACECHECK SUMMARY: 16 hazards displayed (0 errors, 16 warnings)
```

(Captured on a reduced 8-block build so the output is readable; the line numbers
are the two arms of the `if`.) Read it as: a read in one arm is unordered with
respect to a write in the other arm. Racecheck is reporting the absence of a
*happens-before* edge, which is exactly what is missing.

Note it reports **warnings, not errors**, and it is reporting hazards on shared
memory between lanes of one warp — precisely the case that used to be free.
Module 10 returns to racecheck for cross-block races.

---

## TODO 1 — the fix

```cpp
float t = 1.0f;
for (int step = 0; step < STEPS; ++step) {
    int v = s[base + ((lane + 1) & 31)];   // whole warp reads
    __syncwarp();
    if (lane & 1) {
        for (int j = 0; j < 300; ++j) t = fmaf(t, 1.00001f, 1e-7f);
    }
    s[tid] = v + 1;                        // whole warp writes
    __syncwarp();
}
```

Two changes, and both are required for the *reasoning* even though the compiler
may only need one of them.

**(1) Hoist the exchange out of the divergent region.** The load and the store
are now executed by the whole warp, in one basic block each. The divergence is
still there — odd lanes still do their 300 FFMAs — but it no longer straddles a
shared-memory access. This is the structural fix and it is the one that matters.

**(2) Make the ordering explicit with `__syncwarp()`.** The first one separates
every lane's read from any lane's write within the same step. The second
separates this step's writes from the next step's reads. `__syncwarp()` with
its default full mask requires all 32 lanes to arrive, which they do, because
it sits outside the `if`. It is also a compiler barrier for shared memory, so
the array no longer needs `volatile` — and `volatile` was never providing
ordering anyway.

The alternative correct answer: keep the divergence where it is but move the
data out of shared memory entirely, exchanging through registers with
`__shfl_sync` and an explicit full mask hoisted outside the branch. Module 30
develops that; the reasoning is the same — the *communication* must be
warp-uniform, whatever medium it uses.

### What the SASS shows after the fix

```
STS  [R0.X4], R5 ;
BAR.SYNC.DEFER_BLOCKING 0x0 ;
LDS  R6, [R3.X4] ;        <- one read, whole warp
BSSY B0, ... ;
@!P0 BRA ... ;            <- divergence, but no shared access inside
BSYNC B0 ;
STS  [R0.X4], R9 ;        <- one write, whole warp
```

One `LDS` and one `STS` per step, both outside the `BSSY`/`BSYNC` region.

**Honest observation:** there is **no `WARPSYNC` instruction anywhere in the
generated code** — `cuobjdump -sass | grep -c WARPSYNC` returns 0. Once the
exchange was hoisted, the compiler could prove the warp is converged at both
`__syncwarp()` points and deleted them. That is legal and expected.

Do not draw the wrong conclusion from it. The `__syncwarp()` calls are not
decoration: they are the statement of what the code requires, and they are what
makes the program correct *by the programming model* rather than by the
compiler's current analysis. If a later change reintroduces a path on which the
warp is not converged there, the barriers become real instructions and the
program stays correct. Remove them and you are back to relying on a heuristic.

### Common wrong approaches

| Attempt | What happens |
|---|---|
| Put `__syncwarp()` inside each arm | Each call is reached by only 16 lanes while its default mask names all 32. Undefined: on this hardware it may appear to work, may return with lanes still split, or may hang. Never place a warp barrier where the warp is split, unless you pass a mask naming exactly the lanes that will arrive. |
| Put `__syncthreads()` inside each arm | A block barrier in divergent control flow. Undefined, and the classic hang. Module 9. |
| Add `__syncthreads()` between the read and the write, outside the arms | Correct, but a bigger hammer than needed: it synchronizes all four warps of the block when only one warp is communicating, and it only works because hoisting the exchange out of the arms was the real fix. A block barrier (`BAR.SYNC`) is substantially more expensive than a warp barrier (`WARPSYNC`). |
| Keep the exchange inside the arms but add a second shared buffer (double-buffer) | Actually correct: if the read comes from buffer A and the write goes to buffer B, the two arms cannot interfere. Costs twice the shared memory and still needs a barrier between steps, but it is a legitimate second solution and the harness accepts it. |
| Remove `volatile` and hope | Changes nothing about the ordering; see below. |
| Delete the odd-lane refinement | Removes the divergence and the bug, and is explicitly forbidden by the TODO, because in the real kernel this was abstracted from, the refinement is the work. |

---

## TODO 2 — the recorded masks

```cpp
static const unsigned MASK_ODD_ARM  = 0xaaaaaaaau;   // lanes 1,3,5,...,31
static const unsigned MASK_EVEN_ARM = 0x55555555u;   // lanes 0,2,4,...,30
```

Warp 0 of block 0 is a full 32-lane warp and nothing has exited, so `lane & 1`
splits it exactly in half. `0xaaaaaaaa | 0x55555555 == 0xffffffff` and
`0xaaaaaaaa & 0x55555555 == 0`.

The reason the exercise asks for this is that it is the **diagnostic**. If you
suspected the exchange was executing with the warp split, this is how you
confirm it in thirty seconds: drop an `__activemask()` next to the suspect
access and look at the `popc`. 16 where you expected 32 tells you immediately
that half the warp is somewhere else.

**Common wrong answer:** `0xffffffff` for both, on the theory that the arms are
short enough to be predicated. They are not — 300 FFMAs plus an `LDS` and an
`STS` is far past the if-conversion threshold, and the SASS shows `BRA`. And
even if they *had* been predicated, the mask would still have been
`0xaaaaaaaa`, because `__activemask()` compiles to a `VOTE` that carries the
same predicate (see Exercise 1, TODO 3).

---

## TODO 3 — deterministic or not

```cpp
static int BROKEN_IS_DETERMINISTIC = 1;
```

Measured: the same `524288 / 524288` wrong elements on every one of ten runs.

This matters for choosing tools. The failure is not a *timing* race in the
CPU-multithreading sense, where two agents contend and the winner varies. It is
the deterministic consequence of a code-generation decision: `nvcc` emitted the
odd arm before the even arm, and it will do so on every launch, so every warp in
every block gets the same wrong answer.

Consequences:

- **Re-running does not help.** A reader who assumes "it's a race, so it will
  sometimes pass" will run it fifty times and learn nothing.
- **`compute-sanitizer --tool racecheck` is the right tool**, because it reasons
  about the *absence of ordering guarantees* rather than about observed
  interleavings, so it flags the hazard even though no interleaving ever varies.
- **Printing intermediate values works**, because the behaviour is reproducible.
- A future compiler that orders the arms the other way would produce a
  *different* deterministic wrong answer, and a future compiler that interleaved
  them would produce a varying one. Determinism today is not a property of the
  program; it is a property of this build.

---

## TODO 4 — the two claims

### (a) `VOLATILE_FIXES_IT = 0` (false)

`volatile` tells the compiler it may not cache the location in a register and
may not elide or reorder accesses to it *within one thread*. That is a
**visibility** guarantee, and a narrow one.

The bug here is an **ordering** problem *between* lanes. `volatile` says nothing
about when lane 3's store becomes visible to lane 2, because under the model
`volatile` was written for, that question did not exist: lockstep made the
answer "immediately, always". Remove lockstep and `volatile` has no replacement
to offer.

Concretely: the broken kernel already has `volatile`, the `LDS` and `STS` are
both present in the SASS in program order, and it is still wrong for 100% of
its output. `volatile` did its job and the program is still broken.

Module 9 develops "`volatile` is not synchronization" at block and device scope,
including what the memory model actually requires. At warp scope, the statement
you need now is simply: `volatile` orders nothing between threads.

### (b) `CONVERGED_VARIANT_IS_GUARANTEED = 0` (false)

`warpSmoothConverged` performs the identical `volatile __shared__` exchange with
no warp-level synchronization, but in a region with no divergence. The harness
reports **0 wrong elements out of 524,288**. It works.

It is not guaranteed.

The CUDA programming model on sm_70+ does not promise that the lanes of a warp
are at the same program counter at any particular instruction. What actually
happens is that `nvcc` emits `BSSY`/`BSYNC` at immediate post-dominators and the
Ada scheduler reconverges there, so in straight-line code the warp really is
together and the exchange really is safe. That is a statement about **this
compiler and this chip**, not about the language.

This is the most dangerous configuration a bug can be in: correct in testing,
undefined in the standard, and one optimisation decision away from becoming the
100%-wrong kernel sitting next to it in the same file. The two kernels differ
only in where the exchange sits relative to an `if`. Nothing in the source
marks one as safe and the other as not.

The rule to take away: **if your program's correctness depends on two lanes of a
warp being at the same instruction, say so in the source** — with `__syncwarp`,
or with a `_sync` intrinsic and an explicit mask. Code that is right by accident
is indistinguishable, at review time, from code that is right on purpose.

---

## Synchronization / memory reasoning

Three scopes appear in this exercise and they are distinct:

| mechanism | scope | what it orders |
|---|---|---|
| `__syncthreads()` | block | all threads of the block; used once, to publish the initial `s[tid] = in[gid]` |
| `__syncwarp(mask)` | warp | the lanes named in `mask`; also a compiler barrier |
| `volatile` | one thread | nothing between threads; only stops register caching |

The initial `__syncthreads()` is genuinely needed and is *not* the bug: the
exchange reads `s[base + ...]`, which is within the warp, so warp scope would
have sufficed for the rotation — but the array is filled by the whole block and
the barrier is the honest way to publish it. (Module 9 makes `__syncthreads`
semantics precise; here it is used, not explained.)

Nothing in this exercise needs atomics, fences, or cooperative groups. The
entire fix is "make the communication happen where the warp is together."

---

## Performance reasoning

The exercise is not timed, and deliberately: the fixed kernel and the broken
kernel do the same work, and the broken one is not faster in any interesting
way. But two performance observations are worth recording.

- **Hoisting the exchange out of the arms is a performance improvement as well
  as a correctness fix.** The broken kernel issues two `LDS` and two `STS` per
  step (one per arm); the fixed kernel issues one of each. The divergence that
  remains covers only the FFMA chain, which is the part that genuinely differs
  per lane.
- **The `__syncwarp()` calls cost nothing here**, because the compiler removed
  them after proving convergence. In code where they survive, a `WARPSYNC` is a
  single instruction with no memory traffic — orders of magnitude cheaper than
  the `BAR.SYNC` that `__syncthreads()` compiles to. Reaching for a block
  barrier when a warp barrier would do is a common and measurable mistake.

---

## Expected output

Actual output observed on the RTX 3500 Ada, CUDA 13.2, `-arch=sm_89 -O3`:

```
=== Module 8 / Exercise 3 : a kernel that used to be idiomatic ===
4096 blocks x 128 threads = 524288 elements, 8 steps, warp ring rotate

--- warpSmoothBroken ---
  wrong elements: 524288 of 524288 (100.00%), identical count over 10 runs: yes
  recorded __activemask() just before the exchange, warp 0 of block 0:
    odd  lane (lane 1)  : 0xaaaaaaaa  popc=16   you predicted 0xaaaaaaaa
    even lane (lane 0)  : 0x55555555  popc=16   you predicted 0x55555555
    TODO 2: MATCH / MATCH
    TODO 3: predicted deterministic=1, actual=1  MATCH

--- warpSmoothConverged (same idiom, no divergence) ---
  wrong elements: 0 of 524288
  (what this does and does not prove is TODO 4b)

--- TODO 4 ---
  (a) VOLATILE_FIXES_IT             = 0   MATCH
  (b) CONVERGED_VARIANT_IS_GUARANTEED = 0   MATCH

--- warpSmoothFixed (TODO 1) ---
  wrong elements: 0 of 524288
  10 further runs: all correct

SCORE: 7/7
OVERALL: PASS
```

Entirely deterministic; ten repetitions of each kernel produced identical
results every time.

---

## An honest note on what could not be reproduced

This exercise was originally designed around the textbook claim that *any*
warp-synchronous code breaks under independent thread scheduling. That claim, as
usually stated, could not be reproduced on this hardware, and the exercise was
rebuilt around what is actually true.

Specifically, the following were all written, compiled at `-O3` for `sm_89`, and
run 20 times each over 524,288 elements, and **all produced correct results with
zero mismatches**:

- a ring rotation through `volatile __shared__` with no `__syncwarp()`, in a
  converged region (this is `warpSmoothConverged`, shipped);
- the same with the array *not* marked `volatile`;
- the same with heavy divergent work placed *before* the exchange within the
  loop body, so that the lanes arrive at very different times;
- a lane-0-writes / all-lanes-read broadcast through `volatile __shared__`.

In every case, `nvcc` 13.2 placed `BSSY`/`BSYNC` at the immediate
post-dominator and the Ada scheduler reconverged there. The hazard that *does*
reproduce, deterministically and at 100%, is the one shipped: an exchange placed
where the warp is genuinely split.

A second, independently reproducible ITS-era failure was also measured while
building this exercise, and is worth knowing even though it belongs to Module
30's subject matter. Calling

```cpp
if (lane & 1) { v = __shfl_xor_sync(0xffffffffu, a, 1); }
```

— a full mask supplied from inside a half-warp region, which is the naive port
of the pre-Volta `__shfl_xor(a, 1)` — produced **wrong data for 49.95% of
elements** (every odd lane got its own value instead of its partner's). Naming a
lane in a `_sync` mask asserts that it will arrive; when it does not, the result
is undefined and on this chip it is silently wrong rather than loud.

So the accurate statement, which is what the lesson makes, is not "warp-
synchronous code always fails on Volta+". It is: **warp-synchronous code has no
defined behaviour on Volta+, it often appears to work because of a compiler
heuristic, and the cases where the heuristic does not cover you fail totally and
silently.** That is worse than a reliable failure, not better.

---

## The result that matters

The pre-Volta guarantee was never "lanes of a warp run in lockstep." It was
"lanes of a warp run in lockstep *while the warp is converged*, and the hardware
guarantees reconvergence at the post-dominator." Independent thread scheduling
removed the second half, and that is what invalidated the idiom. The kernel in
this exercise violates even the *first* half — its exchange sits where the warp
is provably split — which is why it fails totally and deterministically, and why
no amount of `volatile` helps. The fix is not to add a barrier where the bug is;
it is to move the communication to where the warp is whole, and then to write
down, in `__syncwarp()`, the requirement you are depending on.

**Variation to try.** Take `warpSmoothConverged` — the one that passes — and add
a single line inside the loop: `if (lane == 0) { for (int j = 0; j < 300; ++j) t
= fmaf(t, 1.00001f, 1e-7f); }`, placed *between* the read and the write. Predict
whether it still passes. Then compile it and read the SASS before you run it,
and see whether the compiler kept the exchange in one block or split it. You are
now doing the thing this module exists to teach: deciding correctness from the
instruction stream rather than from the test result.
