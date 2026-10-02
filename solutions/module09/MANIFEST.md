# Module 09 manifest

## Concepts taught

- **The two guarantees of `__syncthreads()`**, stated and named separately:
  - **G1, execution barrier** — no thread of the block proceeds until every
    *non-exited* thread of the block has arrived
  - **G2, memory fence at block scope** — all shared and global writes issued
    before the barrier are visible to the whole block after it
- **Which guarantee a given hazard needs**: RAW across threads needs G1+G2;
  **WAR across threads needs G1 only** (reads publish nothing), and is therefore
  the barrier you can often *delete* by double-buffering
- **Double-buffering by round/tile parity** as the standard way to trade shared
  memory for one fewer barrier per iteration (measured: 8 → 4 `BAR.SYNC`)
- **The uniformity rule**: `__syncthreads()` in conditional code is defined only
  if the condition evaluates identically across the entire block; block-uniform
  conditions (on `blockIdx`, on a kernel argument) are fine
- The three canonical undefined shapes: barrier under a `threadIdx` condition,
  barrier in a loop with a thread-dependent trip count, barrier after an early
  `return`
- **The exited-thread rule** — a thread that has returned is removed from the
  barrier's expected-arrival set, which is why the early-`return` shape usually
  does not hang and must not be relied on
- **Barrier arrival is counted per warp, not per thread** (PTX `bar.sync`
  semantics), which is why a partial-*warp* divergent barrier completes
- **On sm_89, the usual failure mode of a divergent barrier is silent
  corruption, not a hang** — measured for two distinct shapes
- **`__syncthreads_count` / `_and` / `_or`** — barrier plus block-wide
  reduction of a predicate; the mechanism that makes a *block-uniform* loop
  condition, and hence an in-loop barrier, legal by construction
- **`__syncwarp(mask)`** — warp-scope barrier; why ITS (M8) made it necessary
  on sm_70+; when it is required (lanes communicating through memory) and when
  it is not (`_sync` intrinsics carry their own synchronization)
- **Fences vs barriers**: a fence orders *one thread's own* accesses as seen by
  a scope and makes nobody wait; the four scopes
  (`__threadfence_block` / `__threadfence` / `__threadfence_system`), their
  SASS (`MEMBAR.SC.CTA` / `.GPU` / `.SYS`), and the cost argument (scheduler
  time vs memory-pipeline time)
- **When you need a fence but not a barrier**: publish/subscribe, where a spin
  already supplies the waiting and only the payload-before-flag ordering is
  missing
- **`volatile` is not synchronization** (the Module 4 debt): it is a compiler
  directive, gives no G1, no G2, no atomicity, no ordering against non-volatile
  accesses; why the legacy `volatile __shared__` warp-synchronous reduction tail
  is broken on sm_70+
- **The formal CUDA memory model**: `cuda::memory_order_{relaxed, acquire,
  release, acq_rel, seq_cst}` and `cuda::thread_scope_{thread, block, device,
  system}`; a release store == fence + relaxed store
- **Cooperative groups basics**: `this_thread_block()`, `.sync()`,
  `tiled_partition<32>()`, `thread_rank()`; the argument for the explicit-group
  style is interface (the group is a value a function signature can demand),
  not performance
- **`grid.sync()` requires cooperative launch and full co-residency**, so it
  caps the grid at `occupancy × SMs` and forces a grid-stride rewrite
- **There is no cross-block synchronization**: a spinning block consumes the
  exact SM slot the block it waits for needs; deadlock is structural, not a
  fairness bug
- **The kernel boundary is the grid-wide barrier** — the only device-wide
  ordering guarantee that holds for an arbitrary grid size
- **Co-residency as a computable number**: `cudaOccupancyMaxActiveBlocksPer
  Multiprocessor × multiProcessorCount` = 240 on this GPU at 256 threads; a
  bounded spin measures it as `gridDim − 240` timeouts
- **Barrier cost = the block's work skew**, not the instruction: measured
  1.33–1.40× for one `__syncthreads()` per round vs per-pair flags + a fence
  under a moving 30× skew, and **3.49× the other way** with the skew removed
- Tooling reality check: `synccheck` detected **none** of this module's
  divergent barriers; `racecheck` found every data race including one whose
  output was numerically correct; `initcheck --initcheck-address-space shared`
  found the uninitialised-shared-slot case

## CUDA API / intrinsics / syntax introduced

- `__syncthreads()` (semantics, finally), `__syncthreads_count(pred)`,
  `__syncthreads_and(pred)`, `__syncthreads_or(pred)`
- `__syncwarp(mask)` (default mask `0xffffffff`)
- `__threadfence_block()`, `__threadfence()`, `__threadfence_system()`
- `volatile __shared__` — presented as a bug to diagnose, per spec §9
- `<cuda/atomic>`: `cuda::atomic_ref<T, Scope>`, `.load(order)`,
  `.store(v, order)`; `cuda::memory_order_relaxed / acquire / release`;
  `cuda::thread_scope_block / _device / _system`;
  `cuda::atomic_thread_fence` and `cuda::barrier` named only
- `<cooperative_groups.h>`: `cg::this_thread_block()`, `cg::thread_block`,
  `.sync()`, `.thread_rank()`, `cg::tiled_partition<32>()`,
  `cg::thread_block_tile<32>`; `cg::this_grid()` / `grid.sync()` named and
  deferred
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- `cudaDeviceGetAttribute` with `cudaDevAttrMultiProcessorCount`,
  `cudaDevAttrCooperativeLaunch`
- `atomicExch`, `atomicAdd(p, 0)` — used **only** as a flag, with an explicit
  "Module 10 owns atomics" note in every file that uses them
- SASS mnemonics named: `BAR.SYNC.DEFER_BLOCKING`, `WARPSYNC`, `MEMBAR.SC.CTA`
- Toolchain: `compute-sanitizer --tool racecheck --racecheck-report analysis`,
  `--tool synccheck`, `--tool initcheck --initcheck-address-space shared`,
  `--kernel-regex "kns=<name>"`; `cuobjdump -sass` used to count barriers and
  to find `MEMBAR`
- Build flags: `-std=c++17` and (MSVC) `-Xcompiler /Zc:preprocessor` for libcu++

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Classification + predict + design | 4 | Eight fragments; classify each barrier/gap as REQUIRED (with which guarantee) / UNNECESSARY / UNDEFINED, then predict whether the broken form actually misbehaves on this GPU; finally halve the barrier count of a tiled loop | Fragment G is REQUIRED (cross-lane RAW inside one warp, no `__syncwarp`) and produces **bit-exact correct output every run** — racecheck reports 131 072 hazards on a kernel that passes. Fragment B needs G1 **only** (it is WAR, and reads publish nothing), which almost everyone answers as G_BOTH — and missing that is exactly why they then cannot see the double-buffering fix in TODO 4. Fragment H is a barrier inside an `if` and is perfectly legal (the condition is on `blockIdx`), so "barrier inside a conditional" is not the trigger |
| `exercise02.cu` | Debugging / deadlock + design | 4 | A cross-block spin-wait that is instant at 240 blocks and stalls for a second at 8192; plus a tiled shift that is wrong only on the partial last block; reader predicts the timeout count, derives co-residency from the API, fixes the guard, and re-designs away the cross-block dependency | The producer is the **last** block, so the timeout count is `gridDim − co-resident` = 7952, not "a few" and not "all" — and the tempting "fix" of making block 0 the producer works on this GPU while being a correctness argument that rests entirely on unspecified placement order. Also: `memcheck`, `synccheck` **and** `racecheck` all report zero on the broken tiled shift; only `initcheck --initcheck-address-space shared` finds it |
| `exercise03.cu` | Design (mechanism unnamed) + performance prediction | 4 | Paired producer/consumer, 16 dependent rounds, a 30× skew that moves each round; supply the producer ordering and the consumer wait without being told the mechanism, write the barrier variant, and predict which is faster and by how much | Both `__threadfence_block()` and `__syncthreads()` make TODO 1 correct, and leaving TODO 1 *empty* also passes every test on this GPU (Ada retires a thread's shared stores in issue order) — the only evidence is `MEMBAR.SC.CTA` in the SASS. The critical-path argument predicts a 3.6× win for the flag variant; the measured win is 1.33× because 6 blocks/SM of oversubscription hides most of it, and a reader who predicts 3.6× fails the ratio check for the right reason |

Solutions: `solutions/module09/exercise0{1,2,3}_solution.{cu,md}`,
`solutions/module09/check_your_understanding.md`.

Worked examples: `example01.cu` (the two guarantees separated; no-barrier vs
`volatile` vs `__syncthreads`; `__syncthreads_count` convergence loop;
`__syncwarp`), `example02.cu` (fence-based publish/subscribe with SASS
evidence; fence-where-a-barrier-was-needed; cooperative groups; the
co-residency wall).

## Assumed from earlier modules

- M1: SM count (40), 1536 threads/SM, warp = 32, blocks are placed by the
  GigaThread engine, run to completion, never migrate, and **do not exist until
  placed**; eligible/stalled warp scheduling and the free context switch; waves
  and tail effects
- M2: `nvcc -arch=sm_89`, `<<<>>>`, `cudaGetLastError` + `cudaDeviceSynchronize`
  discipline, the `CHECK` macro idiom
- M3: `blockIdx.x * blockDim.x + threadIdx.x`, bounds guards and their cost,
  grid-stride loops (named as the rewrite cooperative launch forces)
- M4: memory spaces; **L1 is not coherent across SMs and L2 is the device-wide
  coherence point** (this is why `__threadfence()` costs more than
  `__threadfence_block()`); the promise that `volatile` would be explained here
- M5: coalescing vocabulary (used incidentally)
- M6: shared memory, static declaration, tiling, cooperative loading, and
  `__syncthreads()` used with the "Module 9 makes this precise" caveat — paid
- M7: shared-memory access patterns (background only)
- M8: warps, divergence, active masks, reconvergence, **independent thread
  scheduling on sm_70+** and per-thread program counters; that a warp
  serializes on a divergent trip count. Module 9 depends on all of this and
  does not re-teach it
- Benchmarking: spec §12 — all configs timed back-to-back in one loop,
  validation in a separate second pass, min-of-N over 4 sweeps, ≥20 timed
  iterations, duration-based clock warm-up, ratios as the stable quantity

## Forward references made

- **Module 10 (races, atomics)**: every use of `atomicExch` / `atomicAdd` /
  `cuda::atomic_ref` in this module is labelled "Module 10 explains atomics;
  here it is only a flag." Atomicity is explicitly *not* a topic here; ordering
  and barriers are
- **Module 12 (reductions)**: named as the place the reduction ladder is built;
  the `volatile` warp-synchronous tail is shown here only as a bug
- **Module 17 (tiled GEMM / software pipelining)**: double-buffering to remove
  the WAR barrier is introduced here as the germ of the pipelining there
- **Module 24 (streams)** [renumbered; this manifest predates the current roadmap]: kernel-boundary ordering is used here; asynchrony
  and overlap are not
- **Module 29 (cooperative groups in depth)**: `grid.sync()`,
  `cudaLaunchCooperativeKernel`, the co-residency cap on cooperative grids,
  multi-grid groups, and `cg::reduce` — introduced by name and by constraint
  only
- **Module 30 (warp-level primitives)**: `__shfl_down_sync` named as the
  correct replacement for the `volatile __shared__` reduction tail; warp
  reduction algorithms deliberately excluded here
- **Module 32 (async copy / `cuda::barrier`)**: split arrive/wait barriers and
  `cuda::pipeline` named as the reason `cuda::barrier` exists
- Nsight Systems / Nsight Compute not used in this module; the tools here are
  `compute-sanitizer` and `cuobjdump`
