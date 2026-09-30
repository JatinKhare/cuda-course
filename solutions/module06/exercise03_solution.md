# Module 06 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -lineinfo -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
.\exercise03_solution.exe --tiles 8
compute-sanitizer --tool racecheck --racecheck-report analysis .\exercise03_solution.exe --small
compute-sanitizer --tool racecheck --racecheck-report analysis .\exercise03_solution.exe --small --tiles 8
```

---

## The bug class

Both defects are the same class: **a shared-memory access ordered only by
wishful thinking**. `DIAGNOSIS = DIAG_NO_ORDER`.

The kernel does

```cpp
s[lane] = in[i];
float v = 0.5f * (s[lane] + s[B - 1 - lane]);
```

`s[lane]` is written by this thread, so that half is fine. `s[B-1-lane]` is
written by thread `B-1-lane`, which for `B = 256` and `lane = 0` is **thread
255** — warp 7. Nothing in the program orders warp 7's `STS` before warp 0's
`LDS`. The warps are independent instruction streams on (potentially) different
processing blocks of the SM, issued by different warp schedulers, with no
dependency between them that the scoreboard can see.

**Why 32 threads passes.** With `B = 32` the block is a single warp.
`B-1-lane` is always a lane of the *same* warp, and a warp executes one
instruction at a time across all its lanes: every lane's `STS` retires before any
lane's `LDS` issues, because they are the same instruction. The ordering the
programmer wanted is supplied accidentally by the SIMT execution model.

This is the single most important sentence in the exercise: **"it works" at
warp width is not evidence of correctness, it is evidence that you have not yet
crossed a warp boundary.** Relying on it is the "implicit warp-synchronous
programming" that independent thread scheduling on sm_70+ broke for good — the
compiler is free to reorder, and on Volta and later the hardware is free to
schedule lanes of a single warp independently too. Module 8 covers independent
thread scheduling; Module 9 covers why `volatile` does not repair any of this.

**Why the errors are non-deterministic and look plausible.** A losing race means
warp 0 read `s[255]` before warp 7 wrote it, so it read whatever the *previous*
block resident in that SRAM left there — a real sample value from some other
window of the same signal. The output is a valid-looking float, not a NaN, and
the count of wrong elements changes with scheduling.

---

## TODO 1 — diagnosis

```cpp
static const int DIAGNOSIS = DIAG_NO_ORDER;
```

Ruling out the distractors is part of the exercise:

| candidate | why it is wrong here |
|---|---|
| `DIAG_OOB` | `lane` and `B-1-lane` are both in `[0, B)` for every lane. memcheck confirms: no invalid accesses. |
| `DIAG_SMEM_SIZE` | the launch passes `B * sizeof(float)` and the kernel indexes `[0, B)`. Exact. |
| `DIAG_BAD_MIRROR` | `B-1-lane` is the correct mirror. A wrong formula would fail *deterministically*, identically every run, and at `B = 32` as well. The 32-thread pass rules it out on its own. |
| `DIAG_BANK` | bank conflicts are a throughput effect. They cost cycles; they never change a value. If your first instinct was "shared memory misbehaving ⇒ banks", note that bank conflicts have no correctness component at all. |

The general discriminator: **a bug that depends on block size and varies run to
run is a synchronization bug; a bug that is identical every run is a logic bug.**
Running the same input eight times, which the harness does, is the cheapest test
that separates the two.

---

## TODO 2 — rule 1: after writing a tile, before reading it

```cpp
s[lane] = (i < n) ? in[i] : 0.0f;
__syncthreads();                       // <-- TODO 2
float v = 0.5f * (s[lane] + s[B - 1 - lane]);
```

`__syncthreads()` is a `BAR.SYNC`: no thread of the block proceeds until all have
arrived, and shared writes before it are visible to all threads after it. That is
precisely the edge the program was missing between thread 255's store and thread
0's load. (Module 9 gives the memory-ordering half real content; for now, this is
the tool.)

This alone makes `--tiles 1` pass at every block size.

---

## TODO 3 — rule 2: after reading a tile, before overwriting it

```cpp
        float v = 0.5f * (s[lane] + s[B - 1 - lane]);
        if (i < n) out[i] = v;

        __syncthreads();                // <-- TODO 3
    }   // next iteration overwrites s[lane]
```

The second defect is a **write-after-read** hazard and it only exists because the
block reuses one buffer across loop iterations. In iteration `t`, warp 0 reads
`s[255]`. If warp 0 then races ahead into iteration `t+1` and executes
`s[0] = in[...]` while warp 7 is still executing iteration `t`'s read of `s[0]`,
warp 7 gets iteration `t+1`'s value.

Measured with TODO 2 fixed and TODO 3 still missing, `--tiles 8`:

```
  block    grid       smem/blk     result
  32       4096       128        B PASS  (8/8 runs clean)
  64       2048       256        B FAIL  (8/8 runs wrong, worst 47080/1048576 elements)
  128      1024       512        B FAIL  (8/8 runs wrong, worst 46744/1048576 elements)
  256      512        1024       B FAIL  (8/8 runs wrong, worst 76608/1048576 elements)
  1024     128        4096       B FAIL  (8/8 runs wrong, worst 153672/1048576 elements)
```

Compare the damage: ~4.5 % of elements wrong versus ~45 % for the missing
read-after-write barrier. That is characteristic. A missing RAW barrier fails
almost every time because the write genuinely has not happened yet; a missing WAR
barrier fails only when one warp gets a whole loop iteration ahead of another,
which needs the warps to actually diverge in timing. **The WAR bug is the one
that survives your testing and ships.**

Note the alternative fix, which is legitimate and sometimes better:
double-buffer. Use two tiles `s[2][B]` and alternate, and the WAR hazard
disappears without a second barrier — at the cost of twice the shared memory.
That is the entry point to the software-pipelining techniques of Module 17.

---

## Synchronization / memory reasoning

Both barriers are outside any divergent control flow: the `for (t...)` bound is
`tilesPerBlock`, a kernel argument, uniform across the block, and neither
`__syncthreads()` is inside the `if (i < n)` guard. That is deliberate. The guard
protects only the *store*; the loads are made safe by clamping in `cpuReference`'s
mirror image instead of by exiting threads. If you "optimised" by writing

```cpp
if (i >= n) return;            // WRONG
```

at the top of the loop body, the threads of the last partial block would leave
before reaching the barrier and the remaining threads would wait for arrivals
that cannot come. On sm_89 the hardware barrier counts only non-exited threads,
so this specific shape happens not to hang — but move the `return` inside the
loop rather than before it, or run on hardware that counts differently, and it
does. Module 9 makes the rule explicit.

---

## Real `compute-sanitizer` output

### racecheck on the shipped (broken) kernel

```
> compute-sanitizer --tool racecheck --racecheck-report analysis .\exercise03.exe --small
========= COMPUTE-SANITIZER
========= Warning: Race reported between Write access at fold_blend(const float *, float *, int, int)+0x550 in exercise03.cu:100
=========     and Read access at fold_blend(const float *, float *, int, int)+0x580 in exercise03.cu:107 [65536 hazards]
=========
========= Error: Race reported between Write access at fold_blend(const float *, float *, int, int)+0x550 in exercise03.cu:100
=========     and Read access at fold_blend(const float *, float *, int, int)+0x580 in exercise03.cu:107 [65536 hazards]
...
========= RACECHECK SUMMARY: 5 hazards displayed (4 errors, 1 warning)
```

Line 100 is `s[lane] = ...`, line 107 is the read of `s[B-1-lane]`. The tool
names both endpoints of the hazard and the source lines, which is exactly the
information you need. `-lineinfo` is what makes the line numbers appear; without
it you get offsets only.

**Read the "1 warning" carefully — it is the answer to the Prediction question.**
There are five configurations (32/64/128/256/1024) and five reports. The
32-thread configuration, the one that produces *correct output on every run*, is
reported — as a **Warning** rather than an Error, because racecheck can see that
the two accesses were in the same warp and so happened to be ordered by SIMT
execution. It still reports it, because the program contains no ordering
construct and the correctness is accidental. A tool that only reported observed
misbehaviour would be useless for this bug class; racecheck reports the *absence
of ordering*, which is the actual defect.

### racecheck with TODO 2 fixed but TODO 3 missing (`--tiles 8`)

```
========= Error: Race reported between Write access at ...:98
=========     and Read access at ...:105 [384 hazards]
=========
========= Error: Race reported between Read access at ...:105
=========     and Write access at ...:98 [384 hazards]
```

Note the second report has **Read first, Write second** — that is the
write-after-read direction, i.e. the TODO 3 defect, and it is how you tell the
two apart without changing the code.

### memcheck on the broken kernel

```
> compute-sanitizer --tool memcheck .\exercise03.exe --small
========= COMPUTE-SANITIZER
========= CUDA API Warning: Resetting device while there are still other users claiming to use it
...
========= Target application returned an error
========= ERROR SUMMARY: 1 error
```

**memcheck finds nothing.** The single reported "error" is the non-zero exit
status of the program itself plus a WDDM-specific complaint about
`cudaDeviceReset`; there is not one invalid access. That is the correct result
and the point of running it: memcheck answers "did anyone touch memory they
should not have", and here nobody did. Every access was in bounds, correctly
aligned, and to memory the block owns. The bug is in *when*, not *where*. Reach
for `--tool racecheck` when the values are wrong but the addresses are fine, and
for `--tool memcheck` when the program crashes.

### racecheck on the fixed kernel

```
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

---

## Expected output

```
=== Module 6 / Exercise 3 : fold_blend, tilesPerBlock = 8 ===
  block    grid       smem/blk     result
  32       4096       128        B PASS  (8/8 runs clean)
  64       2048       256        B PASS  (8/8 runs clean)
  128      1024       512        B PASS  (8/8 runs clean)
  256      512        1024       B PASS  (8/8 runs clean)
  1024     128        4096       B PASS  (8/8 runs clean)

  your diagnosis (TODO 1): missing ordering between a shared write and a shared read
  diagnosis: CORRECT

OVERALL: PASS
```

For reference, the shipped broken version, `--tiles 1`:

```
  32       32768      128        B PASS  (8/8 runs clean)
  64       16384      256        B FAIL  (8/8 runs wrong, worst 236966/1048576 elements)
  128      8192       512        B FAIL  (8/8 runs wrong, worst 428947/1048576 elements)
  256      4096       1024       B FAIL  (8/8 runs wrong, worst 456296/1048576 elements)
  1024     1024       4096       B FAIL  (8/8 runs wrong, worst 518816/1048576 elements)
```

---

## The result that matters

Two `__syncthreads()` calls, sixteen characters of fix, and the entire difficulty
is that one of them guards a hazard with no new data in it. **Every shared buffer
that is written and read needs a barrier after the write; every shared buffer
that is *reused* needs a second one after the read.** The first bug is loud —
half the output is wrong, every run. The second is quiet: 4.5 % wrong, only
above 32 threads, only when warps drift apart in time, and completely absent from
your unit tests if those use one tile per block. `compute-sanitizer --tool
racecheck` finds both in seconds and flags the ordering gap even in the
configuration that produces correct answers — which is why you run it on code
that passes, not only on code that fails.

**Variation to try.** Remove both barriers and instead declare
`extern __shared__ volatile float s[]`. This is the historical fix and it is
wrong: `volatile` forbids the *compiler* from caching the value in a register,
but it says nothing about the *hardware's* execution order across warps, and
sm_70+ independent thread scheduling means it no longer even holds within a warp.
Measure it, watch it fail, and then read Module 9's treatment of why.
