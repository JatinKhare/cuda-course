# Module 17 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
```

Warning-clean. `SCORE: 7/7`, `OVERALL: PASS`.

The shipped `exercise03.cu` compiles and runs with the TODOs blank. It does not
crash; it prints the symptom and `OVERALL: FAIL`. That is the house exception
for a debugging exercise — you are meant to see the failure.

---

## The bug class

All three defects belong to one family: **code whose correctness depends on a
dimension being a multiple of the tile size, or on M being equal to N.** A
square power-of-two test suite makes all three disappear simultaneously, which
is why this is the most common real GEMM bug and why Module 16 fixed the course
on 1027 × 2053 × 769 in the first place.

Shipped symptom:

```
  512 x 512 x 512       nonfin        0 | Freiv  0.001557 | samp   0.03449 | PASS
  1035 x 1541 x 1063    nonfin   523710 | Freiv       inf | samp       inf | FAIL
```

Read the columns before you read the verdict. `nonfin = 523710` says half a
million elements of C still hold the `+infinity` the harness prefilled them
with — they were **never written**. That is not "a wrong number"; it is a
*coverage* failure, and it points at the store, not at the arithmetic. The `inf`
in the Freivalds and sampled columns is a consequence: once a non-finite value
is in the array, every comparison against it is false and both error metrics are
meaningless. This is precisely why Module 16 insisted the finiteness check runs
**first**.

---

## TODO 1 — diagnosis: codes 1, 2 and 3

### Code 1 — truncating k-tile count

```cpp
const int nTiles = K / TILE;                 // shipped
const int nTiles = (K + TILE - 1) / TILE;    // fixed
```

`1063 / 16 = 66`, `ceil = 67`. The last 7 values of `k` are never accumulated,
in every one of the 1 594 935 dot products. `512 / 16 = 32` exactly, so this is
invisible on the square test.

Note that this defect *also* makes the kernel wrong in a way that does not
depend on M or N at all. If you only had the square test at 512 you could not
see it; if you had a test at 512 × 512 × **513** you would.

### Code 2 — an out-of-range cell left stale instead of zero-filled

```cpp
if (row < M && aCol < K) As[ty][tx] = A[...];      // shipped: no else
if (bRow < K && col < N) Bs[ty][tx] = B[...];
                                                   // fixed:
As[ty][tx] = (row < M && aCol < K) ? A[...] : 0.0f;
Bs[ty][tx] = (bRow < K && col < N) ? B[...] : 0.0f;
```

Both guards are present and both index expressions are right; the defect is what
the *false* branch does, which is nothing. The slot keeps whatever was there —
the previous tile's value, or on the first tile **uninitialised shared memory**
— and the accumulation loop multiplies it by a genuine element of the other
operand.

Zero is the correct fill because a zero term contributes nothing to a dot
product. It is an identity, not an approximation: the zero-padded tile computes
exactly the same sum as a shorter one would.

Exercise 1's solution notes show the trap in its sharpest form: **fixing only
one of the two lines makes the kernel pass**, because out-of-range A cells and
out-of-range B cells occur at the same `k` and a zero on either side kills the
product. The defect here removes both. Do not take the redundancy as licence.

`compute-sanitizer --tool memcheck` is clean on this — nothing out of bounds is
read. The tool that sees it is:

```
compute-sanitizer --tool initcheck --initcheck-address-space shared .\exercise03.exe
```

which reports, on the one-sided variant used in Exercise 1:

```
========= Uninitialized __shared__ memory read of size 16 bytes
=========     at void gemmTiled<(int)8>(...)+0x4a0
=========     by thread (0,4,0) in block (61,129,0)
=========     Address 0x80
```

Note "size 16 bytes": the read it caught is the merged `LDS.128` of four A
elements, which is also a reminder that the instruction you wrote is not the
instruction that executes.

### Code 3 — store guard against the wrong dimension

```cpp
if (row < M && col < M)   // shipped
if (row < M && col < N)   // fixed
```

`M = 1035 < N = 1541`, so columns 1035..1540 of every row are never written:
`1035 × 506 = 523 710` elements, exactly the `nonfin` count. Had `M > N`
instead, the same line would write *past the end of each row*, corrupting the
next row and, on the last row, running off the allocation — the same code, a
different and much louder failure, decided entirely by the test shape. On a
square problem it is correct.

### The four distractors, and why each one is visible on a square problem

| code | why a square power-of-two test catches it |
|---|---|
| 4 — load mappings swapped | wrong on every element at every shape; not shape-dependent at all |
| 5 — missing barrier | a race is a race at any shape; 512 × 512 × 512 with `TILE = 16` still has 8 warps per block |
| 6 — accumulator in shared memory | every thread's partial sum is clobbered at every shape |
| 7 — grid sized with a truncating divide | at 512 with `TILE = 16` the divide is exact, so this one *is* hidden too — **but** its symptom is an unwritten block row/column, i.e. `nonfin`, and it is distinguishable from code 3 by *which* elements are unwritten: code 7 leaves a whole trailing block row **and** column unwritten, code 3 leaves trailing columns only. Here the trailing rows are written, so it is code 3. |

Distractor 7 is deliberately the hardest to rule out, and the way to rule it out
is to look at *which* elements survived the poison rather than *how many*. A
good debugging habit: when the finiteness check fires, print the bounding box of
the non-finite elements before you touch anything.

---

## TODO 5 — the size set

```cpp
if (i == 0) { *M = 17; *N = 31; *K = 9; return 1; }
return 0;
```

One shape, 4743 multiply-adds, all three required properties covered:

```
  shape 0 =   17 x   31 x    9  props 1 1 1 1 1 .   your kernel: PASS
  property order:  0=K % TILE != 0  1=M % TILE != 0  2=N % TILE != 0
                   3=M != N  4=K < TILE  5=M > N

  required properties covered by your set: 3 of 3
```

The three required properties are **bit 0 (`K % TILE != 0`), bit 3 (`M != N`)
and bit 4 (`K < TILE`)** — mask `0b011001 = 25`, which the harness recovers at
run time by searching for the mask whose FNV-1a hash matches the stored
constant, so that it is not written down in the exercise file.

The reasoning:

| defect | property it needs | why |
|---|---|---|
| 1 truncating tile count | `K % TILE != 0` | if `TILE` divides `K` there is no partial tile to drop |
| 2 stale tile cell | `K % TILE != 0` | the `row ≥ M` and `col ≥ N` cases produce stale cells too, but only in threads whose output is never written, so they cannot affect the answer. Only the `k` tail can. |
| 2 (stronger form) | `K < TILE` | with only one tile, an un-zero-filled cell is genuinely **uninitialised** shared memory rather than the previous tile's value — undefined content rather than merely wrong content, and the case `initcheck` reports |
| 3 wrong store guard | `M != N` | with `M == N` the guard `col < M` is the guard `col < N` |

**Two of the three defects need the same property, so one shape suffices**, and
it can be tiny: `17 × 31 × 9` is 0.0000028× the work of the 1035 × 1541 × 1063
shape and has exactly the same diagnostic power.

That is the transferable lesson. The instinct when a GEMM is wrong at a large
awkward size is to debug at that size; the right move is to find the *smallest*
shape with the same arithmetic properties, because at 17 × 31 × 9 you can print
the whole matrix. Module 16 chose a large awkward shape for a different purpose
— it needed a realistic performance measurement — and those are two different
jobs for two different test cases.

Two properties in the list are **not** required and are there as distractors:
`M % TILE != 0` and `N % TILE != 0` produce partial tiles too, but on axes where
the affected threads write nothing. A reader who assumes "partial tile on every
axis" is the requirement gets them for free from any awkward shape and never
notices that only one of the three axes mattered. `M > N` is the third
distractor, and it is the one worth thinking about: it changes defect 3 from
"columns never written" (caught by the finiteness check) to "writes past the end
of each row" (caught by the value checks, and a genuine out-of-bounds write on
the last row). Same line of code, two different failure modes, selected by the
test shape. The harness's own third fixed shape, 97 × 61 × 9, has `M > N` for
exactly that reason.

---

## Synchronization / memory reasoning

Nothing here is a synchronisation bug, and that is itself worth noticing. Both
barriers are present and correctly placed in the shipped kernel, `racecheck`
reports nothing, and the failure is fully deterministic — the same wrong answer
every run. **A reproducible wrong answer is not a race.** Module 14 made this
point about histograms; it is just as true here, and it should be the first
thing you establish, because it eliminates an entire class of hypotheses (and
the entire `racecheck`/`synccheck` toolchain) in one run.

The defects that remain after that elimination are all *index* bugs, and index
bugs are found by arithmetic on the boundaries, not by tools.

---

## Performance reasoning

Not a performance exercise, but one number is worth recording: the fixed kernel
is the same code as Exercise 1's, so it runs at the same **1.29–1.32× naive**.
Two of the three defects made the kernel *faster* — dropping the last k-tile
removes 0.7 % of the work, and skipping the zero-fill store removes a
predicated `STS`. A "fast" GEMM that does less work than it should is the exact
failure mode that a sloppy tolerance licenses, which is why the validator
thresholds against `gamma_K · S` and reports the headroom rather than a boolean.

---

## Expected output

```
=== Module 17 / Exercise 3 - 'it works on 1024 x 1024 x 1024' ===

-- the symptom -----------------------------------------------------
  512 x 512 x 512       nonfin        0 | Freiv  0.001557 | samp   0.03449 | PASS
  1035 x 1541 x 1063    nonfin        0 | Freiv 0.0005695 | samp    0.0211 | PASS

-- TODO 1: diagnosis -----------------------------------------------
  you answered {1, 2, 3} : 3 of 3 correct

-- TODOs 2-4: the repaired kernel ----------------------------------
    512 x   512 x   512   nonfin        0 | Freiv  0.001557 | samp   0.03449 | PASS
   1035 x  1541 x  1063   nonfin        0 | Freiv 0.0005695 | samp    0.0211 | PASS
     97 x    61 x     9   nonfin        0 | Freiv   0.02033 | samp    0.1981 | PASS

-- TODO 5: the size set that would have caught it ------------------
  work budget: sum of M*N*K over all your shapes <= 4.0 G
  shape 0 =   17 x   31 x    9  props 1 1 1 1 1 .   your kernel: PASS

  required properties covered by your set: 3 of 3

SCORE: 7/7
OVERALL: PASS
```

(In the solution file the "symptom" section already shows the repaired kernel,
because the repairs are in place. Run the shipped `exercise03.cu` to see the
original symptom: `nonfin 523710`, `Freiv inf`, `samp inf`, `FAIL`.)

Deterministic; the error figures are identical run to run because nothing here
is a race.

---

## The result that matters

Three independent defects, all of them in the four lines that handle a boundary,
all of them invisible on every square power-of-two problem, none of them
detectable by `memcheck` or `racecheck`, and all three findable by one 17 × 31 × 9
test that runs in microseconds. The lesson is not "test at awkward sizes" —
everyone says that. It is that **you can decide in advance, from the structure
of the code, exactly which arithmetic property of the test shape each defect
class needs**, and then construct the smallest shape with those properties. The
size of the test has nothing to do with its power.

**Variation to try:** re-run `probeSize` with `(32, 32, 32)`, then `(32, 48, 32)`,
then `(32, 48, 33)`, then `(32, 48, 9)`, and watch the property vector fill in one
bit at a time. Then re-introduce each defect into the kernel by hand and confirm
empirically that the shape with the right property really does catch it — the
harness scores the *properties* because listing the defects in the exercise file
would give the diagnosis away, but the properties are only worth anything if the
implication holds, and you should check it yourself.
