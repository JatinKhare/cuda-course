# Authoring spec — CUDA course, Parts I–III

Read this in full before writing anything. Every module must conform.

---

## 1. Target hardware (verified, do not re-query unless you need a field)

| Property | Value |
|---|---|
| GPU | NVIDIA RTX 3500 Ada Generation Laptop GPU |
| Compute capability | **8.9** (Ada) |
| SMs | 40 |
| Warp size | 32 |
| Max threads / block | 1024 |
| Max threads / SM | **1536** (48 warps) |
| Max blocks / SM | 24 |
| 32-bit registers / SM | 65536 |
| Shared mem / block (default max) | 49152 B (opt-in up to 99 KB via `cudaFuncAttributeMaxDynamicSharedMemorySize`) |
| Shared mem / SM | 102400 B |
| L2 | 48 MB (50331648 B) |
| Global memory | 12 GB GDDR6, 192-bit @ 9.001 GHz → **432.0 GB/s** peak |
| Toolkit | CUDA **13.2** (V13.2.51), driver 596.71 |
| OS / shell | Windows 11, PowerShell (a bash tool is also available) |

Shared memory on Ada: 128 KB unified L1+SMEM per SM, of which **100 KB max is
addressable as shared**. Shared memory has **32 banks × 4 B**.

**Build line for everything:** `nvcc -arch=sm_89 -o <name>.exe <name>.cu`
Add `-O3`, `-lineinfo`, `-Xptxas -v`, `--ptx` per exercise where relevant.
Never use `-G` except in a debugging exercise that explicitly discusses it
(it disables optimization and changes the performance story).

`sm_89` supports: `cp.async` (sm_80+), async barriers / `cuda::barrier`,
`__shfl_*_sync`, independent thread scheduling (sm_70+), 4th-gen Tensor Cores
incl. FP8. It does **NOT** support: thread block clusters, distributed shared
memory, TMA, `wgmma` (all sm_90+). Never write a non-compiling exercise that
depends on sm_90 features.

---

## 2. Directory layout — exact

```
cuda-course/
  moduleNN/
    lesson.md                  <- the teaching text
    exampleNN.cu               <- worked example(s) from the lesson, runnable
    exerciseMM.cu              <- exercise with TODOs, runnable once filled
  solutions/moduleNN/
    exerciseMM_solution.cu
    exerciseMM_solution.md
    check_your_understanding.md  <- answers to the lesson's conceptual questions
```

`NN` and `MM` are two digits (`module05`, `exercise02`).
**Nothing in `moduleNN/` may reveal an answer.** All answers live under
`solutions/`. Do not put answer hints in exercise comments.

Delete any `.exe`, `.exp`, `.lib`, `.pdb` you produce while verifying — commit
no binaries.

---

## 3. Per-module deliverables

1. `lesson.md` — see §4 for the mandatory section structure.
2. **1–2 worked examples** as runnable `.cu` files, referenced from the lesson.
   These are complete (no TODOs) and demonstrate the concept.
3. **2–3 exercises.** Mix the types (§6). At least one exercise per module must
   be something other than plain fill-in-the-blank.
4. A verified solution `.cu` + solution `.md` for every exercise.
5. `check_your_understanding.md` with full answers to the lesson's conceptual
   questions.

---

## 4. `lesson.md` structure — mandatory headings, in this order

```markdown
# Module NN — <Title>

> Prerequisites: Module X, Y
> What this module gives you: <one sentence>

## Concept
## Hardware Mental Model
## Code Walkthrough
## Check Your Understanding
## Exercises
## Prediction
```

- **Concept** — the idea, rigorously. University level. Assume strong C/C++ and
  basic computer architecture, zero CUDA. Define every term on first use.
- **Hardware Mental Model** — what the silicon does. This is the heart of the
  course. Always answer *why* the hardware behaves this way, not just that it
  does. Tie back to SM structure, warp schedulers, memory transactions,
  register allocation.
- **Code Walkthrough** — walk through `exampleNN.cu`, quoting the important
  fragments inline and explaining them. Do not just say "see the file".
- **Check Your Understanding** — 2–4 conceptual questions, answered in
  `solutions/moduleNN/check_your_understanding.md`. Questions must be
  non-lookupable: they require reasoning about the model, not recall.
- **Exercises** — for each: what the program must accomplish, the exact
  `nvcc` and run commands, what each TODO requires, and what the validation
  will check. Never state the answer.
- **Prediction** — 2–3 things the reader must commit to before running.

Tone: precise, direct, no filler, no cheerleading. Markdown tables where they
help. Code fences tagged `cpp`.

---

## 5. Exercise file requirements — non-negotiable

Every `exerciseMM.cu` must be a **standalone, fully runnable `.cu` file once the
TODOs are filled**. That means it contains:

- all `#include`s,
- a `CHECK(...)` error-checking macro (define it in each file; files are
  standalone, duplication is intended),
- `main()` with allocation, deterministic host-side initialization
  (seeded `rand()` or an index-derived formula — **never** unseeded random),
- the kernel launch, plus **both** `cudaGetLastError()` and
  `cudaDeviceSynchronize()` checks,
- **a CPU reference implementation and a numerical comparison that prints
  PASS/FAIL**, with a tolerance appropriate to the dtype
  (`fabs(a-b) <= 1e-5f * fmaxf(1.0f, fabsf(b))` for fp32 accumulation; be
  stricter for integer kernels).
  **⚠️ That default rule is WRONG for long-accumulation kernels and will reject
  correct code.** Module 16 measured a *correct* GEMM failing it: with zero-mean
  data `|C| << S = Σ|a||b|`, giving relative-to-`|C|` error 9.7e-4, and
  `γ_K = K·u/(1−K·u)` is already 4.58e-5 at K=769 — 4.6× the house rule before
  anything goes wrong. **Scale the tolerance by the magnitude of the
  accumulation (S), never by `|result|`.** For GEMM and any dot-product-shaped
  kernel, reuse `gemmValidate()` from `cuda-course/module16/example01.cu`,
  which does three checks **in this order**:
  1. **Finiteness/writtenness first.** Prefill the output with `+inf`; a
     surviving sentinel means never-written, and `0*inf = NaN` catches a kernel
     that reads C when β==0. This must run first — a NaN compares false against
     everything, so a max-error loop run first *passes* an all-NaN array.
  2. **Freivalds probe** over the full output, O(MN+KN+MK), with a non-negative
     random vector so systematic errors accumulate coherently instead of
     cancelling, tolerance propagated through the same contraction in double.
  3. **Sampled exact double reference**, threshold `err / (γ_K · S_ij) ≤ 1`.
  Test on **two datasets**: positive [0.5,1.5) where `|C|≈S`, and zero-mean
  [−1,1) where `|C|≪S`. The second is what exposes a tolerance scaled by the
  wrong quantity.
  **⚠️ Caveat (Module 18): a numerically perfect kernel can still read out of
  bounds.** M18 measured a GEMM that drops a `kt+c<K` guard, returns the exactly
  correct answer at 1.05×, and is still reported by `compute-sanitizer` as
  `Invalid __global__ read` — the guard is arithmetically redundant because the
  other operand is zero, but it is required for memory safety. **No numerical
  validator, including this one, can detect a missing guard.** Run memcheck in
  addition to validating,
- cleanup (`cudaFree`, `free`, `cudaDeviceReset()`),
- a header comment block: module/exercise, GOAL, BUILD command, RUN command,
- **`setvbuf(stdout, NULL, _IONBF, 0);` as the first line of `main()`.** House
  convention. Buffered stdout is discarded when a process aborts, so a harness
  bug or a device-side fault presents as silently truncated output with no
  indication of where it stopped. Unbuffered output turns that into a visible
  last-line-printed. Module 13 lost real time to exactly this.

The file must **compile as shipped** (with TODOs unfilled) and fail gracefully
or report FAIL — it must never fail to compile, and never crash with an illegal
memory access purely because a TODO is blank. Guard with an early-out like:

```cpp
if (blocks <= 0) { printf("Set TODO n first.\n"); return 0; }
```

unless the exercise is deliberately a debugging exercise about a crash.

**Timing:** where the exercise discusses performance, use `cudaEvent_t` timing
with a warm-up launch plus >= 20 timed iterations, and report both ms and
achieved GB/s (or GFLOP/s), plus % of the 432 GB/s peak. Do not use
`clock()`/`chrono` around async launches.

### TODO style

- Mark with `// TODO n: ...` followed by `// YOUR CODE HERE` on the line(s) to fill.
- Number TODOs sequentially within a file.
- **Module 1–3: 1–3 TODOs per exercise. Modules 4–7: 2–4. Modules 8–10: 3–5,
  and at least one must require designing a strategy, not filling an
  expression.**
- Make at least one TODO per module **subtle**: something where the obvious
  answer is wrong (off-by-one on the last partial tile, a barrier that looks
  needed but isn't, a barrier inside divergent control flow, a `volatile` that
  used to work pre-Volta and no longer does, an index that is coalesced for
  reads but not writes).
- Do not always name the CUDA feature to use. Phrase some TODOs as a
  requirement ("ensure every thread sees the tile before use") and let the
  reader choose the mechanism.

---

## 6. Exercise types — mix them

1. **Fill in the code** — the default.
2. **Predict the behavior** — reader answers in prose before running; the
   program prints the truth. e.g. which lanes execute, what addresses a warp
   touches, how many sectors, is it a race.
3. **Debugging** — ship a kernel that is *intentionally broken* in a specific,
   realistic way. The file header states the symptom, **not** the cause. The
   reader diagnoses. Solution md explains the bug class and how to find it with
   `compute-sanitizer`.
4. **Optimization** — a correct but slow kernel; reader produces v2, v3. The
   harness times all versions side by side and validates each.
5. **CPU → GPU** — give a C++ loop nest, reader designs the whole
   parallelization (decomposition, grid/block, memory layout, sync).
6. **Performance reasoning** — predict faster/slower and the bottleneck before
   measuring; harness then measures.
7. **PTX/SASS** — compile with `--ptx` / `cuobjdump -sass`, connect source to
   instructions. Include the exact commands.

---

## 7. Solution `.md` structure — mandatory

```markdown
# Module NN / Exercise MM — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run
## TODO 1 — <title>          (one section per TODO)
   - the code
   - why it is correct (from the hardware model, not "because it works")
   - common wrong approaches, and what symptom each produces
## Synchronization / memory reasoning
## Performance reasoning
## Expected output
## The result that matters
```

Expected output must be the **actual output you observed** when you ran the
solution on this GPU. Paste real numbers. If timings vary run-to-run, say so
and give a range.

`## The result that matters` is one paragraph: the single insight the exercise
exists to deliver, plus a suggested variation the reader can try.

---

## 8. Verification — required before you report done

For every solution file:

```
nvcc -arch=sm_89 -O3 -o <tmp>.exe <solution>.cu
<tmp>.exe
```

It must compile warning-clean and print PASS. For debugging exercises, also run
`compute-sanitizer --tool memcheck <tmp>.exe` (and `--tool racecheck` where
relevant) on the *broken* version to confirm the tool actually reports the
planted bug, and quote that output in the solution md.

Then confirm the **exercise** file (TODOs unfilled) still compiles.

Delete the binaries afterwards.

---

## 9. Consistency rules across modules

- Never use a concept before the module that introduces it. The order is:
  M1 GPU architecture / SIMT / warps-blocks-grids ·
  M2 nvcc, `__global__/__device__/__host__`, launch syntax, error checking ·
  M3 thread indexing, 1D/2D/3D, flattening, bounds, grid-stride loops ·
  M4 memory hierarchy: registers, local, shared, L1/L2, global, constant,
     read-only, pinned host ·
  M5 coalescing, transactions/sectors, AoS vs SoA, alignment ·
  M6 shared memory, static vs dynamic, tiling, cooperative loading ·
  M7 bank conflicts, broadcast, padding ·
  M8 warps, SIMT, divergence, predication, reconvergence, active masks ·
  M9 `__syncthreads`, barriers, memory ordering, warp sync, cooperative groups
     basics ·
  M10 races, atomics, contention, privatization.
- `__syncthreads()` may be *used* from M6 but is only *explained properly* in
  M9 — in M6/M7 say "barrier; Module 9 makes this precise."
- Shared memory may not appear before M6. Atomics may not appear before M10
  (exception: M1–M5 may mention them in prose as a forward reference).
- **Modern practice only.** Use `__shfl_down_sync` not `__shfl_down`; never
  rely on implicit warp-synchronous programming or `volatile` shared memory as
  a synchronization mechanism (you may present it as a *bug* to diagnose, and
  must then explain why independent thread scheduling on sm_70+ broke it).
  Prefer `cudaDeviceGetAttribute` over deprecated `cudaDeviceProp` clock fields.
- Label anything not portable across architectures as
  **ARCHITECTURE-SPECIFIC OPTIMIZATION**; label the general rule as
  **PORTABLE CUDA CONCEPT**.

---

## 10. Quality bar

Exercises must feel like good graduate homework or a hard GPU interview
question — not textbook drill. A reader who fills every TODO correctly should
have had to *reason about hardware* at least once per exercise. If an exercise
could be completed by pattern-matching the lesson text, it is too easy: rewrite it.

---

## 11. Manifest — required output

When done, write `moduleNN/MANIFEST.md`:

```markdown
# Module NN manifest
## Concepts taught
- bullet list of every concept introduced, with the term used
## CUDA API / intrinsics / syntax introduced
- bullet list
## Exercises
| File | Type | TODOs | One-line description | Subtle trap |
## Assumed from earlier modules
- bullet list
## Forward references made
- bullet list ("mentioned X, said Module Y covers it")
```

This is how modules stay consistent with each other. Be complete and terse.

**Location:** write the manifest to `solutions/moduleNN/MANIFEST.md`, **not** to
`moduleNN/`. It contains a "Subtle trap" column, which would violate §2's rule
that nothing under `moduleNN/` reveals an answer.

---

## 12. Benchmarking methodology on THIS GPU — mandatory

This is a **laptop** part with aggressive clock management. Measured SM clock
swings roughly **0.49–2.04 GHz** across idle, ramped, and thermally-limited
states. Naive timing produces reproducible but entirely fictitious results.
Every timed exercise and every number you paste into a solution md must follow
these rules:

1. **Time all configurations back-to-back in one loop.** Do not validate,
   allocate, or print between timed configurations — those let the clock ramp
   down and silently reorder your results.
   **Scope note (Module 21): this applies to *competing* configurations — ones
   you are comparing against each other.** Ceilings on *different axes* need
   different warm-ups and must be grouped separately. M21 measured one combined
   sweep giving **326 GB/s + 13,750 GFLOP/s**, versus two groups (1500 ms
   stream → off-chip ceilings, then 500 ms compute → on-chip ceilings) giving
   **410 GB/s + 18,642 GFLOP/s**. Group by axis, warm each group for its own
   resource, rotate within a group.
2. **Validate in a separate second pass**, after all timing is done.
3. **Take min-of-N** across at least 3–4 full sweeps, not a single sweep and not
   a mean. The minimum is the least clock-contaminated sample.
   **⚠️ Failure mode on long sweeps (Module 19): rotation removes positional
   bias from the *mean*, not from the *minimum*.** Once a sweep is long enough
   to heat the part monotonically (M19's was 1.7 s of full-machine FFMA, which
   §12.12's ~10 ms segments force), each configuration's minimum is just its
   earliest-position sample, and min-of-N **re-introduces exactly the bias
   rotation exists to remove**. M19 measured the two lowest-indexed shapes
   reporting 10% faster under min-of-N and **equal under the median** of the
   same samples, reproducibly. **If the sweep heats the part, report the
   median and say so.**
4. **Warm-up must be at least 1500 ms — not 400 ms.** This is the single most
   important correction in this section. A 400 ms duration-based warm-up ramps
   the **SM clock** but *not* the **memory P-state**. Module 12 measured the
   identical ceiling kernel, same binary, same data, reproducibly:
   **372.2–373.1 GB/s after 400 ms vs 410.5–410.7 GB/s (95.0% of peak) after
   1500 ms.** Every absolute DRAM figure produced with a 400 ms warm-up is
   ~10% low. Use 1500 ms, then >= 20 timed iterations with `cudaEvent_t`.
   Modules 6–11 were authored under the 400 ms convention; their *ratios* are
   sound (both sides warmed identically) but their absolute GB/s figures are
   understated. Do not "correct" their numbers by arithmetic — if you need a
   comparable absolute figure, re-measure it.
   **Corollary: the warm-up must match the resource you are measuring.** A
   1500 ms *streaming* warm-up ramps the memory P-state but leaves the SM clock
   low; a *compute* warm-up does the reverse. Module 16 measured the power
   manager dropping the SM clock to **285 MHz** during a pure-read kernel
   (memory clock 8801 MHz) — it trades away whatever the kernel is not using.
   To measure both ceilings in one program: **1500 ms stream, then 500 ms
   compute, in that order.** That recovers 18,251 GFLOP/s and 411.0 GB/s
   together.
5. **Report ratios, not just absolutes.** Ratios between configurations are
   far more stable than absolute ms values. If a lesson claims "2x faster",
   that claim must be a measured ratio.
   **⚠️ Correction (Module 15): ratios are NOT "stable to ~1% across thermal
   states". That earlier claim was false.** After several minutes of
   back-to-back benchmarking this part pins its memory clock at **6001 MHz
   instead of 9001**, with `SW_POWER_CAP` and `SW_THERMAL_SLOWDOWN` both
   asserted (`nvidia-smi` throttle reasons `0x24`). In that state the ratios
   are not this GPU's ratios: a tiled transpose fell from 98% of the copy
   ceiling to **81%**, and two versions that are normally distinguishable
   became identical. It recovers after 20–40 s idle. Ratios are stable within
   an operating point, not across this one.
5b. **Any harness that SCORES a performance prediction must guard its
   operating point.** Widening buckets cannot span both states. Required
   pattern, from `module15/exercise01.cu`:
   - after warming, probe a known reference (e.g. the copy ceiling);
   - if it is below a healthy threshold (measured: ~255 GB/s healthy vs
     ~118 GB/s power-capped — use ~200 as the cut), **idle ~10 s and re-warm**,
     up to ~5 attempts;
   - then warn and proceed — never silently skip scoring.
   **Qualifier (Module 19): the probe must stress the same fraction of the
   machine as the kernel it is guarding.** A 480-block FFMA probe reported a
   healthy 17,143 GFLOP/s on a run where a full-machine kernel was taking 2.5×
   its usual *cycles* — and that run then scored 8/10 on a correct solution.
   Guard with a balanced reference launch, not a small one.
   **Also: a harness that scores only *ratios computed inside one rotated
   sweep* needs no guard at all** — M19's Exercise 2 scored 10/10 from a
   2.4×-throttled operating point. Only gates that compare *across* launches
   are exposed.
5c. **Do not validate modules in back-to-back batches.** Running many timed
   programs consecutively induces exactly the state above. It produced a false
   `FAIL` on a Module 15 solution that passes standalone. Put a cool-down
   between programs, or verify them one at a time.
5d. **Place bucket edges in the widest empty gap of the measured
   distribution, and leave ≥8 points of margin.** Measure the real band first
   over many runs. Module 15's strided-write kernels span 34–64% of the copy
   ceiling; the shipped bucket edge sat at 60%, *inside* that band, which no
   amount of warm-up could fix. **If a distinction is narrower than one
   version's own run-to-run spread, it cannot be scored on this machine — show
   it in the printed numbers and say so, rather than pretending to score it.**
6. **`cudaDevAttrClockRate` is NOT the real clock.** It reports 1.545 GHz on
   this GPU while the actual clock runs 1.48–2.04 GHz. Any "% of peak FLOP/s"
   derived from it is meaningless and can exceed 100%. If you need % of compute
   peak, recover the true clock at runtime: record `clock64()` in block 0 and
   compute `cycles / elapsed_ms` on the host. For % of *memory* peak, the
   432 GB/s figure is fixed and safe to use.
7. **Size buffers well beyond the 48 MB L2** when measuring DRAM bandwidth, or
   you are measuring cache. A kernel reporting >100% of peak is measuring L2 —
   say so explicitly rather than reporting the number as a bandwidth.
8. **Memory-bound kernels make poor occupancy/wave demonstrations.** A lone
   bandwidth-bound block has the whole memory system to itself and runs far
   faster than it would in a full wave, which smears a staircase into a ramp.
   Use a compute-bound kernel with one block per SM when you need block
   duration to be independent of what else is resident.

9. **Rotate the sweep order.** Back-to-back timing is not sufficient on its own.
   Measure config `p = (q + sweep) % N` so a different configuration goes first
   in each sweep. Without this, whichever config runs first absorbs the clock
   dip that follows `cudaDeviceSynchronize()`. Verified in Module 7: this
   inflated a baseline by 30% and made a *broadcast* appear 0.75×, i.e. faster
   than conflict-free. With rotation, ratios reproduce to <1%.
   **Rotation only works if `SWEEPS >= NCFG`.** With 4 sweeps over 5
   configurations, config 0 never loses its unfairly early-and-cool sample —
   Module 11 traced a spurious "fusion cliff" to exactly this. Set the sweep
   count to at least the number of configurations being compared.
10. **Ada runs FP64 at 1/64 rate.** Any double-precision microbenchmark that
    does real FP64 arithmetic measures the FP64 pipe and nothing else — Module 7
    saw 2.33 ms for *every* access pattern including a 32-way conflict. If you
    need to move 8-byte data to study memory behavior, use bit reinterpretation
    (`__double_as_longlong` + XOR/integer ops), not FP64 math.
11. **The compiler will vectorize AND ELIMINATE your accesses out from under
    your analysis.** Module 18 confirmed both halves twice: scalar shared reads
    were contracted into `LDS.128`, and a deliberately un-hoisted inner loop
    with 64 source-level reads was CSE'd down to 12 — producing **identical
    SASS and a 1.05× *faster* result than the "optimized" version.** If your
    exercise claims a penalty, verify the penalty exists in SASS before
    shipping the claim.
    **And the converse (Module 17): a correct analysis can be wrong because
    your "fix" destroys a merge.** A textbook-correct "no conflicts here, so
    padding is a no-op" analysis was wrong by **40%** — padding broke the 16 B
    alignment that let ptxas merge four contiguous reads into `LDS.128`, turning
    20 shared instructions into 32. Same fix as rule 11, inverted symptom:
    check the SASS both before and after any layout change.
13. **Sanity-check every microbenchmark against a hardware bound before
    believing it.** Module 17's shared-bandwidth probe reported **40 TB/s** —
    4× the bank array's theoretical maximum — because the compiler hoisted the
    loop-invariant address out of the loop. It only became honest once the
    addresses depended on the loop index. If a number exceeds a physical bound,
    the benchmark is broken, not the hardware.
    **⚠️ Clock clause (Module 21): build the bound from
    `nvidia-smi --query-gpu=clocks.max.sm` (3105 MHz), NOT from the 2.04 GHz
    figure quoted elsewhere in this spec.** That 2.04 is an *observation*, not a
    bound, and a bound built on it **falsely rejects correct measurements** —
    M21 observed it rejecting a valid 329.6 G instr/s result.
14. **An honest-looking probe can still measure the wrong thing with nothing
    eliminated.** Separate from rules 11–12: M21's under-unrolled FFMA probe
    reported a reproducible, entirely fictitious **13,750 GFLOP/s vs 18,642**,
    not because the compiler removed anything, but because a 12-instruction
    loop body containing 8 FFMAs spends **33% of its issue slots on loop
    overhead**. When measuring a *throughput* ceiling, unroll until loop
    overhead is negligible and verify the instruction mix in SASS.
15. **When sweeping a parameter, hold the work per loop branch constant — not
    the trip count.** If the loop body's work varies with the sweep parameter,
    `ptxas` changes its unroll factor and produces a **~25% artefact that looks
    exactly like a hardware effect**. Module 20 traced this precisely (64 vs 192
    FFMAs between branches at C=2) and believes it is the mechanism behind
    Module 11's unexplained C=2 MLP anomaly.
16. **Occupancy sweeps: resident warps are not fungible across block shapes.**
    The same warps/SM packaged differently differs by 1.6× (M20: one 640-thread
    block = 1.008 instr/cycle/scheduler; five 128-thread blocks = 0.625). Sweep
    blocks/SM over **{1, 2, 3, 4, 8, 12} only**, or hold blocks/SM fixed and
    vary block size. Prefer block sizes that are a multiple of 128 threads — a
    block whose warp count isn't a multiple of 4 loads the four schedulers
    unequally.
    32 adjacent unrolled shared-memory reads became 8 `LDS.128` instructions,
    converting a 32-way scalar conflict into 8-way-per-phase and collapsing the
    measured penalty from ~16× to 3.15×. When microbenchmarking a *specific*
    instruction, use `#pragma unroll 1` and non-adjacent access ordering, then
    **confirm in the SASS** that the instruction under test is the one actually
    executing.

12. **Auto-scale the iteration count; a fixed 20 is not enough for short
    kernels.** For a 0.01 ms kernel, 20 iterations is a 0.2 ms timed segment —
    short enough that the clock sags between segments. Module 6 saw this produce
    a fictitious 2.3× in an early draft. Choose the iteration count at runtime so
    each timed segment lasts **~10 ms**, on top of the duration-based warm-up
    (~400 ms) and min-of-N sweeps.

If a measurement contradicts the lesson's prediction, **investigate and explain
it in the solution md**. Do not adjust the claim to fit, and do not quietly drop
the case. A documented surprise is worth more than a clean table.

### ✅ Nsight Systems (`nsys`) WORKS — it is just not on PATH

Verified 2026-10-02 by capturing a real profile with full `--stats=true` output
(kernel times, memcpy breakdown, CUDA API summary). Earlier assumptions that
"all Nsight tooling is broken" were WRONG — `nsys` does *tracing*, which does
not need the GPU performance counters that `ncu` is blocked on.

```
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
"$NSYS" profile -o out --force-overwrite=true --stats=true ./prog.exe
"$NSYS" stats out.nsys-rep
```

Two versions are installed (2025.1.3 and 2025.6.3); use **2025.6.3**. Modules
that trace CPU/GPU timelines, launch overhead, transfer overlap or stream
concurrency should capture **real** profiles and paste **real** output.
Delete `.nsys-rep` and `.sqlite` artifacts when done.

### Broken tooling: document it, never try to fix it

Standing instruction from the user. If a tool on this machine does not work,
**write it up as theory and move on.** Do not spend time diagnosing, escalating,
requesting permissions, or engineering around a broken tool. Give the reader the
command line and the metric names, state plainly that it cannot be run here and
why, and teach the concept without it. Known-broken as of this batch:

- `ncu` — `ERR_NVGPUCTRPERM`, needs elevation. Theory only.
- `compute-sanitizer --tool synccheck` — detects nothing on divergent-barrier
  bugs on CUDA 13.2 / sm_89. Theory only; use `racecheck` and
  `initcheck --initcheck-address-space shared` instead.

Modules 22–23 (Nsight Systems / Nsight Compute) will be taught as theory plus
screenshots-in-prose rather than live profiling runs, unless the user says
otherwise.

### Nsight Compute is currently unavailable on this machine

`ncu` (Nsight Compute 2026.1.0) is installed but every invocation fails with
`ERR_NVGPUCTRPERM — The user does not have permission to access NVIDIA GPU
Performance Counters`. This needs a one-time elevated fix by the user. Until it
is resolved: **give the reader the `ncu --metrics` command line and the exact
metric names**, but do not paste counter output you could not produce, and do
not design an exercise whose validation *depends* on ncu. Build a measurement
harness that constructs the effect under study directly instead.
