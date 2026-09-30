# Resume state — Parts I–V complete, 2026-09-28

## Status

| Modules | State |
|---|---|
| **1–18 (Parts I, II, III, IV, V)** | **COMPLETE and verified.** Every solution compiled and executed on the GPU; every number in the notes is measured. |
| 19–44 + 5 final projects | Not started. |

18 lessons · 35 worked examples · 53 exercises · 53 verified solutions ·
18 answer keys · 18 manifests · 252 files · 0 binaries.

## Next batch — Part VI, Performance Engineering (Modules 19–23)

Suggested wave: **19, 20, 21** then **22, 23** (three at a time; see process
note below).

Modules 19–21 inherit an unusually strong evidence base — these are normally
the hand-wavy modules of a CUDA course, and here they are all measured:

- **M19 Occupancy** is owed the headline result from M18: **100% occupancy is
  18.7× slower than 33%** on the same GEMM source via `__launch_bounds__`, and
  occupancy is **non-monotone** (25→33% wins, 33→67% loses 9×). Also owed:
  **registers are the fourth placement-gate limiter** — M6's occupancy formula
  omits them, and M17 measured two kernels identical in threads and shared
  memory differing 3 vs 2 blocks/SM on 44 vs 40 registers. Also: the register
  granule is **8 per thread, allocated per warp**, and the spill cliff is *not*
  the first spilled byte (an 80 B spill that buys a block is 1.12× faster).
- **M20 Latency hiding** is owed M1's measured curve (dependent pointer chase:
  linear to 3 warps, knee at 6, flat at 4.87× by 24) and M11's ILP/MLP
  measurements (hoisted vs `#pragma unroll 1` = 2.1× at 1 block/SM, 1.00× at 8).
- **M21 Roofline** has all three axes measured: DRAM 410.5–410.7 GB/s, FP32
  17,787–18,256 GFLOP/s, machine balance 43–44 FLOP/byte, **and** the shared
  memory axis M17 added (5.4 TB/s scalar `LDS`, 10.3 TB/s `LDS.128`). M17 notes
  the roofline needs that third axis to explain tiled GEMM at all.
- **M22–23 Nsight** must be **theory only**: `ncu` fails with
  `ERR_NVGPUCTRPERM` (needs elevation the user declined to chase) and
  `compute-sanitizer --tool synccheck` detects nothing on CUDA 13.2. Give the
  command lines and metric names; design no exercise whose validation depends
  on them.

## Process notes that mattered

- **Three agents per wave, not five.** Five concurrent took 8–16 h per module;
  three took 1.5–2 h for the same work. Contention, not workload.
- **Verify by executing, not just compiling.** Module 15 compiled clean and
  had a reference solution that failed its own gate 1 run in 3.
- **Do not validate modules in back-to-back batches** — it induces the
  power-capped state (memory pinned at 6001 MHz) and produces false failures.
  Cool down between programs.
- Per-wave loop: launch 3 → collect manifests → fold corrections into
  `AUTHORING_SPEC.md` and `CROSS_MODULE_INDEX.md` → launch next 3.
- Shared files are edited by the orchestrator only; agents propose changes in
  their manifests.

## Corrections made to already-shipped material (keep doing this)

Later modules have repeatedly falsified earlier ones. That is the process
working, not a defect — but the earlier file must be fixed:

- **M7's `max(2,D)` cost law** now carries a width qualifier in
  `module07/lesson.md`: it holds for **4-byte accesses only**; on `LDS.128`
  cost ∝ D with no floor. M18 isolated this with byte-identical SASS.
- **M16's "6.4–6.5 FMAs per global load"** is restated in the index as **per
  operand-fetch instruction** — M17 and M18 hit this independently.
- **Spec §12's "ratios are stable to ~1% across thermal states"** was false and
  is corrected.
- **Spec §5's default float tolerance** rejects a correct GEMM; the GEMM
  validator in `module16/example01.cu` supersedes it for dot-product kernels.
- **Padding folklore** now has four contradictory measurements (M7 +19%,
  M15 tie, M17 −40%, M18 +23%) and a unifying rule in the index §6b.
