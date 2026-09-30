# Module 09 / Exercise 01 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
.\exercise01_solution.exe --run-ub
```

Warning-clean. For the tooling sections below:

```
nvcc -arch=sm_89 -O3 -lineinfo -o e1li.exe exercise01_solution.cu
compute-sanitizer --tool racecheck --racecheck-report analysis .\e1li.exe
compute-sanitizer --tool synccheck .\e1li.exe
cuobjdump -sass exercise01_solution.exe > sass.txt
```

---

## The answer key, with reasoning

| | fragment | Q1 verdict | Q2 guarantee | Q3 misbehaves here |
|---|---|---|---|---|
| A | tile staged, slot read across warps | REQUIRED | BOTH | 1 |
| B | 4-tile loop, barrier at bottom of body | REQUIRED | **EXEC** | 1 |
| C | each thread touches only its own slot | UNNECESSARY | NA | 0 |
| D | 256-thread block, 64 threads reach the barrier | UNDEFINED | NA | 1 |
| E | per-thread trip count around a barrier | UNDEFINED | NA | 1 (hangs) |
| F | 32-thread block, 16 threads reach the barrier | UNDEFINED | NA | 1 |
| G | cross-lane chain inside one warp | REQUIRED | BOTH | **0** |
| H | barrier inside a conditional | REQUIRED | BOTH | 1 |

### A — REQUIRED / BOTH

`out[t] = s[(t+96) % 256]`. With 256 threads, `t + 96` is a slot owned by a
thread in a different warp. Thread `t` needs that thread to have executed its
store (G1) and needs the store to be visible (G2). Both. Measured: 78 368 –
83 584 of 262 144 elements wrong without the barrier, ~31%, stable across runs.

### B — REQUIRED / EXEC only

This is the one most people get wrong. The marked barrier is at the *bottom* of
the loop body, after the reads and before the next iteration's write to
`s[t]`. That is a write-after-read hazard. What must this thread wait for? For
the other threads to have **finished reading**. A read publishes nothing, so
there is nothing for the memory-fence half to make visible. The needed
guarantee is G1 alone.

If you answered `G_BOTH` here, you almost certainly did not notice that the
hazard direction is reversed, and you will not spot the double-buffering
opportunity in TODO 4 either — the two observations are the same observation.

Measured: 65 056 – 69 152 wrong without it.

`racecheck` confirms the direction of the hazard, listing the **Read** first
and the **Write** second — read-then-write, i.e. WAR:

```
========= Error: Race reported between Read access at caseB(...)+0x140 in exercise01_solution.cu:173
=========     and Write access at caseB(...)+0x160 in exercise01_solution.cu:171 [984576 hazards]
```

(Line 173 is `acc += s[...]`, line 171 is `s[t] = in[b+t] + tile`. Three such
pairs are reported, one per unrolled iteration boundary. Hazard counts drift a
few percent run to run; the pairs and line numbers do not.)

### C — UNNECESSARY

`s[t] = f(in[t]); barrier; out[t] = s[t]*0.5f`. No thread reads a slot it did
not write. There is no inter-thread dataflow, therefore no hazard, therefore no
barrier needed. Measured 0 wrong either way — and this is the only "0" in the
Q3 column that is a *correct* 0.

(This fragment is also an argument for deleting `s` altogether; the compiler
will do it for you. The barrier is what stops it from doing so.)

### D — UNDEFINED

`if (t < 64) __syncthreads();` in a 256-thread block. The condition is not
block-uniform, so by the Programming Guide rule this is undefined.

It does not hang. Run with `--run-ub`:

```
D      256-thread block, 64 threads reach it            -      64000
```

64 000 of 262 144 elements wrong — 24% — and `cudaGetLastError()` returns
`cudaSuccess`. See "why it does not hang" below.

### E — UNDEFINED, and this one really does hang

```cpp
const int n = trips[t];                  // 1..5
for (int i = 0; i < n; ++i) {
    const float v = s[(t + SHIFT) & (TPB - 1)];
    __syncthreads();
    s[t] = v + 1.0f;
    __syncthreads();
}
```

With `--run-ub` the program prints the E line and never returns. GPU
utilisation pins at 100%. `taskkill /F /IM exercise01_solution.exe` recovers it
cleanly; no reboot, no driver reset.

Why this one hangs when D does not: the threads with small trip counts finish
their loop but do **not** exit the kernel — they still have the `out[...]`
store and the epilogue ahead of them, and more importantly their *warp* cannot
retire until all its lanes are done. The warp is therefore still alive and
still expected at the barrier, while it is sitting past the loop. The
expected-arrival count never drops and the arrival count never reaches it.
That is a genuine deadlock, not a slow kernel.

### F — UNDEFINED, and it corrupts too

Same shape as D but in a **32-thread block**: `if (t < 16) __syncthreads();`.

The temptation is to reason: the barrier counts warps, this block is one warp,
one warp arrives, the barrier releases, so the fragment is harmless. The first
three clauses are true. The conclusion is false, because the bug is not the
barrier at all — it is the *divergence*. Lanes 16–31 do not enter the `if`;
under independent thread scheduling they run ahead and execute
`out[t] = s[(t+1) & 31]` while lanes 0–15 are still at the barrier, having
possibly not yet retired their `STS`. Measured: 8 188 – 8 190 of 262 144 wrong,
reproducible to ±2 elements.

Key lesson: a barrier inside divergent control flow is not made safe by the
warp-granular arrival rule. That rule only explains why you do not get a
*hang*; it says nothing about the data.

### G — REQUIRED / BOTH, and it produces the right answer anyway

```cpp
if (t < 32) {
    s[t] = s[t] + s[(t + 1) & 31];
    /* GAP */
    s[t] = s[t] + s[(t + 3) & 31];
}
```

Lane `t` reads `s[(t+3) & 31]`, which lane `t+3` wrote in the previous
statement. That is a cross-lane read-after-write through memory. Under
independent thread scheduling (Module 8) lanes of one warp have independent
program counters and there is no guarantee that lane `t+3`'s store has
happened. The fragment needs a warp-scope barrier: `__syncwarp()`.

Measured with the gap empty: **0 wrong**, every run.

**This is the whole exercise.** "Passed" is a statement about what the Ada
load/store unit did this afternoon. "Correct" is a statement about what the
CUDA execution model permits. The fragment is undefined, the compiler is free
to reorder the two statements' shared accesses, and a future compiler or a
future architecture may. `racecheck` says so out loud even though the numbers
are right:

```
=========     and Write access at caseG(const float *, float *, int)+0x140 in exercise01_solution.cu:264 [131072 hazards]
=========     and Write access at caseG(const float *, float *, int)+0x1f0 in exercise01_solution.cu:268 [131072 hazards]
=========     and Read access at caseG(const float *, float *, int)+0x120 in exercise01_solution.cu:264 [131072 hazards]
=========     and Read access at caseG(const float *, float *, int)+0x1d0 in exercise01_solution.cu:268 [131072 hazards]
```

131 072 hazards = 1024 blocks × 32 lanes × 4 accesses. If you had shipped this
kernel because the unit test passed, `racecheck` is the only thing standing
between you and a field bug two toolkit versions from now.

Note also what the correct fix *is*: `__syncwarp()`, not `__syncthreads()`.
`__syncthreads()` here would be both wrong (it is inside `if (t < 32)`, which
is not block-uniform — you would be trading a race for undefined behavior) and
expensive.

### H — REQUIRED / BOTH, and legal

```cpp
if (blockIdx.x & 1) { s[t] = in[b+t]; __syncthreads(); out[...] = s[...]; }
else                { out[b+t] = in[b+t]; }
```

The condition depends on `blockIdx`, not `threadIdx`. Every thread of a given
block evaluates it the same way. Either all 256 threads reach the barrier or
none do. Perfectly legal, and necessary for the same reason as A. Measured
29 376 – 31 936 wrong without it — about half of A's count, because only the
odd-numbered blocks take the staging path.

This fragment exists to stop you from pattern-matching on "barrier inside
`if`" as the trigger for UNDEFINED. The trigger is "condition not uniform
across the block."

---

## TODO 4 — one barrier per iteration

```cpp
__global__ void caseB_one_barrier(const float* __restrict__ in,
                                  float* __restrict__ out)
{
    __shared__ float s[2][TPB];
    const int t = threadIdx.x, b = blockIdx.x * TPB;

    float acc = 0.0f;
    for (int tile = 0; tile < NTILE; ++tile) {
        const int cur = tile & 1;
        s[cur][t] = in[b + t] + (float)tile;
        __syncthreads();                       // RAW only
        acc += s[cur][(t + SHIFT) & (TPB - 1)];
    }
    out[b + t] = acc;
}
```

**Why it is correct.** The second barrier in `caseB` existed only to stop
iteration *k+1*'s write from landing on a slot that some thread was still
reading for iteration *k*. Double-buffering by parity puts iteration *k+1*'s
write in the buffer that nobody is reading, so the hazard is gone rather than
synchronized away. Two buffers are enough and not three: iteration *k+2* reuses
buffer `k & 1`, and every thread has passed the barrier of iteration *k+1*
before iteration *k+2*'s write, and that barrier is after all of iteration
*k*'s reads.

**Verification, and this is the part worth doing yourself:**

```
cuobjdump -sass exercise01_solution.exe | findstr /C:"Function" /C:"BAR.SYNC"
```

```
_Z5caseBPKfPfi            : 8 BAR.SYNC
_Z17caseB_one_barrierPKfPf: 4 BAR.SYNC
```

The four-tile loop is fully unrolled in both, so the counts are literal: 2 per
tile versus 1 per tile.

**Common wrong approaches.**

- *Deleting the bottom barrier without double-buffering.* Numerics fail
  (~65 000 wrong) and `racecheck` reports the WAR hazard. This is exactly
  fragment B in its broken form.
- *Moving the barrier from the bottom of the body to the top.* This is the same
  program with the loop rotated; there are still two barriers per iteration
  once you account for the one you need before the first read. `cuobjdump`
  shows 8.
- *Keeping one buffer but giving each thread a private copy of its peer's
  value in a register before the barrier.* This can be made to work for this
  specific access pattern (each thread reads exactly one slot), but it does not
  generalise to a tile that every thread reads many slots of — which is every
  real tiling kernel — and it costs a register per read. The buffer is the
  general answer.
- *Using 3 or more buffers.* Correct but wasteful, and it exceeds the 2 KB
  shared-memory budget the exercise imposes for `NTILE = 4` if you go past two.

---

## Synchronization / memory reasoning

The single organising idea: **classify the hazard first, then pick the
primitive.**

| hazard | needs | cheapest correct primitive |
|---|---|---|
| RAW across warps, same block | G1 + G2 | `__syncthreads()` |
| WAR across warps, same block | G1 | `__syncthreads()`, or eliminate by double-buffering |
| RAW across lanes, same warp | G1 + G2 at warp scope | `__syncwarp()` |
| RAW between a lane's own accesses | nothing | nothing |
| RAW across blocks | nothing can do it | restructure: kernel boundary |

Everything in this exercise is one of the first four rows; the fifth is
Exercise 2.

---

## What the sanitizers actually reported

This is the honest part, and it is not the answer you would expect.

**`compute-sanitizer --tool synccheck` found nothing at all.** Not on fragment
D, not on F, not on the loop-barrier hang in E, not on a `__syncwarp(0xffffffff)`
executed from a half-converged warp. In every configuration tested — with and
without `-lineinfo`, with and without a `--kernel-regex` filter:

```
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
```

synccheck is documented as the divergent-barrier tool and it is the first thing
anyone recommends for this bug class. On CUDA 13.2 / sm_89 it did not detect
any of the four divergent-barrier shapes in this module. Treat this as a
measured limitation, not as evidence that the code is fine.

**`compute-sanitizer --tool racecheck` found everything that was a data race.**
Real output, abridged to the headline lines:

```
========= Error: Race reported between Write access at caseA(const float *, float *, int)+0xd0 in exercise01_solution.cu:149
=========     and Read access at caseA(const float *, float *, int)+0x110 in exercise01_solution.cu:153 [1048576 hazards]
========= Error: Race reported between Read access at caseB(const float *, float *, int)+0x140 in exercise01_solution.cu:173
=========     and Write access at caseB(const float *, float *, int)+0x160 in exercise01_solution.cu:171 [984576 hazards]
=========     and Read access at caseF(const float *, float *)+0x130 in exercise01_solution.cu:247 [524288 hazards]
=========     and Write access at caseG(const float *, float *, int)+0x140 in exercise01_solution.cu:264 [131072 hazards]
========= Error: Race reported between Write access at caseH(const float *, float *, int)+0xd0 in exercise01_solution.cu:284
=========     and Read access at caseH(const float *, float *, int)+0x120 in exercise01_solution.cu:288 [524288 hazards]
========= RACECHECK SUMMARY: 10 hazards displayed (5 errors, 5 warnings)
```

A, B, F, G and H, including **G, which produced numerically correct output**.
racecheck is the right first tool for this whole bug class.

One more measured surprise: running fragment F under synccheck made its
corruption **disappear** (8 188 wrong natively, 0 wrong under the sanitizer).
The sanitizer's instrumentation perturbs the schedule enough to hide the race.
Do not use "it passes under compute-sanitizer" as a correctness argument for
numerics; use the tool for what it reports, not for what the program prints
while it runs.

---

## Performance reasoning

Only TODO 4 has a performance component, and it is a structural one rather than
a measured one: 4 `BAR.SYNC` instead of 8 for the same work, bought with 1 KB
of extra shared memory. The instruction itself is nearly free; what you save is
one block-wide rendezvous per tile, and the cost of a rendezvous is the idle
time of every warp that arrives before the slowest. Exercise 3 turns that idle
time into a number.

---

## Expected output

Observed on the RTX 3500 Ada, CUDA 13.2, `-O3`, no sanitizer. The "broken"
column varies by a few percent run to run for A, B and H; C, F and G are stable
to within a couple of elements.

```
=== Part 1: executed fragments (correct form vs broken form) ===
case   what it is                                     as-written     broken
A      tile staged, slot read across warps                     0      82208
B      4-tile loop, barrier at bottom of body                  0      65056
C      each thread touches only its own slot                   0          0
D      256-thread block, 64 threads reach it                   -    skipped
E      per-thread trip count around a barrier                  -    skipped
       (pass --run-ub to execute D and E. Read the header first.)
F      32-thread block, 16 threads reach it                    -       8190
G      cross-lane chain inside one warp                        0          0
H      barrier inside a conditional                            0      31104

=== Part 2: TODO 4 -- one barrier per iteration ===
  wrong elems: 0  -> PASS
  ...
  A correct answer has NTILE (=4) BAR.SYNC in that function,
  not 2*NTILE. If you see 2*NTILE you solved a different problem.

=== Part 3: your classification ===
case   Q1 verdict   Q2 guarantee Q3 misbehaves observed
A      correct      correct      correct      1
B      correct      correct      correct      1
C      correct      correct      correct      0
D      correct      correct      correct      skipped
E      correct      correct      correct      skipped
F      correct      correct      correct      1
G      correct      correct      correct      0
H      correct      correct      correct      1

  Q1 verdicts    : 8/8
  Q2 guarantees  : 8/8
  Q3 misbehaves  : 8/8
  score          : 24/24

OVERALL: PASS
```

With `--run-ub`, the D line reads `64000` and the program then hangs at E.

Run-to-run ranges over three runs: A 78 368 – 83 584, B 65 056 – 69 152,
F 8 188 – 8 190, H 29 376 – 31 936.

---

## The result that matters

Fragment G is the whole module in eight lines: a kernel that is undefined by
the CUDA execution model, that `racecheck` flags with 131 072 hazards, and that
produces bit-exact correct output on every run on this GPU. Every synchronization
bug you will ever ship looks like G — it looks like D and E only in the ten
minutes before you fix it. The discipline the exercise is trying to install is
that the question "is this barrier necessary?" is answered by naming the hazard
and the guarantee, never by deleting the barrier and re-running the test.

**Variation to try:** change fragment G's chain from `s[(t+3) & 31]` to
`s[(t+3) & 63]` so that it reaches into the second warp's slots, leave the gap
empty, and re-run. The reasoning that made the original "safe" — all
participants in one warp — is now false, and the error count stops being zero.
Then put `__syncwarp()` in the gap and watch it *not* fix it, because
`__syncwarp` synchronizes a warp and the hazard is now between warps. That is
the cheapest possible demonstration that scope is a real parameter and not
decoration.
