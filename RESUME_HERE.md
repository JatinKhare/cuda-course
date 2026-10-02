# Resume state — paused 2026-10-02

## To resume: one message

> "Continue the CUDA course — next batch is Modules 25, 26, 27."

Everything needed is on disk. No conversation context is required — the
authoring state lives in `AUTHORING_SPEC.md`, `CROSS_MODULE_INDEX.md` and this
file.

## Status

| Modules | State |
|---|---|
| **1–24** | **COMPLETE and verified.** Every solution compiled and executed on the GPU; every number measured. |
| 25–44 + 5 projects | Not started. |

24 lessons · 47 worked examples · 71 exercises · 71 verified solutions.

Parts I–VII(partial) done: fundamentals, memory system, execution, core
parallel algorithms, GEMM, performance engineering, and the first of the
advanced-CUDA modules (streams).

## ✅ Important: `nsys` WORKS (discovered 2026-10-02)

Earlier assumptions that "all Nsight tooling is broken" were **wrong**. `ncu`
needs GPU performance counters (blocked, `ERR_NVGPUCTRPERM`). **Nsight Systems
does tracing and needs no such permission — it was merely not on PATH.**

```
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
"$NSYS" profile -o out --force-overwrite=true --stats=true ./prog.exe
```

Verified producing real `cuda_api_sum`, `cuda_gpu_kern_sum`,
`cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum` tables. Two versions installed;
use **2025.6.3**. So **Module 22 should be built on real captured profiles**,
and Modules 24+ can use `nsys` as an instrument.

`ncu` remains theory-only — Module 23's angle is to teach each metric by
anchoring it to a measurement the course already made independently (sectors
from M5, bank conflicts from M7/M18, the four-limiter slice occupancy model from
M19, the stall taxonomy from M20, the five-ceiling roofline from M21).

## Remaining roadmap

- **Part VII finish** — 25 events, 26 pinned memory, 27 unified memory,
  28 CUDA graphs, 29 cooperative groups
- **Part VIII** — 30 warp primitives in depth (**recharted**: not their
  introduction; they are load-bearing from M10 onward), 31 warp specialization
- **Part IX** — 32 async copy, `cuda::pipeline`, mbarrier
- **Part X** — 33 Tensor Core fundamentals, 34 WMMA/MMA, 35 modern tensor
  pipelines (sm_90+, conceptual with an sm_89 fallback)
- **Parts XI–XVII** — 36 libraries, 37 compilation pipeline, 38 PTX, 39 SASS,
  40 multi-GPU, 41 CUDA for AI, 42 LLM inference kernels, 43 CUTLASS,
  44 modern CUDA 2026
- **5 final projects**

## Process notes

- **Three agents per wave.** Five concurrent took 8–16 h/module; three take
  1.5–2 h for the same work.
- **Verify by executing, not just compiling** — M15 compiled clean with a
  reference solution that failed its own gate 1 run in 3.
- **Verify one program at a time with cool-downs** (spec §12.5c). Back-to-back
  batches induce the power-capped state and produce false failures.
- Per-wave loop: launch 3 → collect manifests → fold corrections into
  `AUTHORING_SPEC.md` and `CROSS_MODULE_INDEX.md` → launch next 3.
- Shared files are edited by the orchestrator only; agents propose changes in
  their manifests.
- **Interrupted agents leave usable partial work.** Resuming with an explicit
  "here is what survived / here is what is missing" prompt works well — proven
  twice.

## Corrections already pushed back into shipped modules

Later modules keep falsifying earlier ones. The earlier file gets fixed:
- **M4's register formula** → corrected to the four-slice model (M19, verified
  on 137 kernels; the aggregate model is wrong on ~7%).
- **M7's `max(2,D)` law** → width qualifier added (4-byte only; `LDS.128` is ∝D).
- **M16's "FMAs per global load"** → restated as *per operand-fetch instruction*.
- **Spec §5's float tolerance** → rejects a correct GEMM; superseded for
  dot-product kernels by `module16/example01.cu`'s validator.
- **Spec §12's "ratios stable to ~1%"** → false, corrected.
- **Min-of-N** → re-introduces positional bias on sweeps that heat the part;
  use the median.

## Unresolved disagreement (do not teach as fact)

**Block-shape fungibility.** M20 measured 20 warps/SM as one 640-thread block at
1.008 instr/cycle/sched vs five 128-thread blocks at 0.625. M19's controlled
experiment found all shapes within 1.4–3.4% with no ordering. M20's own fit does
not predict its own contrast. Recorded in `CROSS_MODULE_INDEX.md` §6b as
unresolved; both modules adopt the conservative methodology regardless.
