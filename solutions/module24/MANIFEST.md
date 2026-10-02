# Module 24 manifest — CUDA Streams (opens Part VII)

## Concepts taught

- **Stream** — an ordered queue of operations; same-stream operations are
  ordered, different-stream operations are not unless you order them.
- **"No ordering" is a permission, not a promise** — two streams make
  concurrency *legal*, the hardware decides whether it *happens*.
- **The legacy default stream** and its device-wide barrier semantics against
  every *blocking* stream in the context.
- **Blocking vs non-blocking streams**; `cudaStreamNonBlocking` as the
  library-safe default.
- **`--default-stream per-thread`** as a compile-time redefinition of stream 0,
  and the measured limit of what it cures: **default-stream synchronization,
  not API synchronization**.
- **Implicit synchronization** — the class of runtime calls that drain the
  device as a side effect (`cudaMalloc`, `cudaFree`, `cudaHostAlloc`,
  `cudaMallocHost`, `cudaHostRegister`, pageable blocking `cudaMemcpy`,
  cache-config changes, device switch), and the rule they generalize to:
  *the calls that synchronize are the calls that change the device's
  configuration.*
- **`cudaMemcpyAsync` only overlaps out of pinned memory**, and why — the driver
  stages pageable transfers through its own pinned buffer **on the calling host
  thread before the call returns**. The reframing: on this Windows/WDDM driver
  pinning is worth 1.07× of bandwidth and ~1.7× of overlap. **The overlap is
  the payoff, not the raw copy speed.**
- **`asyncEngineCount`** — the number of DMA engines; measured **1** on this
  GPU, which makes three-way overlap unachievable and sets every bound in the
  module.
- **The copy/compute pipeline bound**:
  `(H+K+D) / max(copyBusy, K)` with `copyBusy = (engines>=2) ? max(H,D) : H+D`.
  Maximised at `K = H + D`, where it is 2×.
- **Head-of-line blocking on a single copy engine** — the obvious per-chunk
  issue order `H2D(k), K(k), D2H(k)` is *slower than serial*; the deferred-D2H
  and two-phase orders repair it.
- **Chunk count vs stream count** are different numbers; the chunk-count curve
  is ramp/drain (`~(H+D)/M`, falling) against host issue cost (linear, rising),
  with a broad optimum.
- **Blocking vs non-blocking waits**: `cudaDeviceSynchronize` /
  `cudaStreamSynchronize` / `cudaStreamQuery` / `cudaEventQuery`, and why
  `cudaErrorNotReady` is not a sticky error.
- **Cross-stream dependencies** via `cudaEventRecord` + `cudaStreamWaitEvent`,
  with the precise capture semantics (*the event captures what has already been
  issued*) and the resulting "record too early" bug that still prints the right
  answer. **Labelled "Module 25 covers events properly."**
- **Minimum streams = graph width; minimum events = edges that same-stream
  ordering cannot supply.**
- **Host callbacks** — `cudaLaunchHostFunc`, and the absolute prohibition on
  calling any CUDA API inside one. `cudaStreamAddCallback` named as deprecated.
- **Stream priority** — `cudaDeviceGetStreamPriorityRange` (measured `[0, -5]`,
  6 levels), lower value = higher priority, a placement hint and not preemption.
- **Where streams do NOT help** — concurrent kernel execution is available only
  in proportion to the machine the running kernel is not using; on 40 SMs that
  is almost nothing for a properly sized kernel. Copy/compute overlap is the
  exception and is almost always worth having.
- A missing cross-stream dependency **cannot hang** (removing an edge removes
  waits) and so can only be found by reading, not by running.

## CUDA API / intrinsics / syntax introduced

- `cudaStream_t`, `cudaStreamCreate`, `cudaStreamCreateWithFlags`
  (`cudaStreamDefault`, `cudaStreamNonBlocking`), `cudaStreamCreateWithPriority`,
  `cudaStreamDestroy`
- `cudaStreamSynchronize`, `cudaStreamQuery` (and `cudaErrorNotReady`)
- `cudaMemcpyAsync`, `cudaMemsetAsync` (4-argument form)
- the **4th launch parameter**: `kernel<<<grid, block, shmem, stream>>>` — the
  Module 2 debt, paid
- `cudaEventRecord(ev, stream)` into a **non-default** stream,
  `cudaStreamWaitEvent`, `cudaEventCreateWithFlags` + `cudaEventDisableTiming`
  (used as a dependency mechanism only; M25 owns events)
- `cudaDeviceGetStreamPriorityRange`
- `cudaLaunchHostFunc`, `cudaHostFn_t`
- `cudaDeviceProp::asyncEngineCount`, `::concurrentKernels`
- `nvcc --default-stream per-thread`, and the
  `CUDA_API_PER_THREAD_DEFAULT_STREAM` macro for detecting it in source
- `cudaMallocHost` / `cudaFreeHost` **used in anger** (introduced M4; M26 owns
  the mechanism)

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Fill-in + design + prediction (§6 types 1, 5, 6) | 5 | Turn a 48 MB H2D→compute→D2H job on `N = 12582917` floats into a chunked overlap pipeline, derive the bound first, reach ≥70% of it. 10 points. | **Four.** (a) The obvious issue order `H2D(k), K(k), D2H(k)` head-of-line-blocks the single copy engine and measures **0.87×, slower than serial** — and is otherwise perfectly correct, so no test catches it. (b) `chunkOf` with `n/nChunks` instead of `ceil` silently drops the last 5 elements; the harness prefills with `-1.0f` and checks index `N-1` explicitly because nothing else would. (c) `pipelineBound` returning `max(H,D)` for both engine counts is right for every machine the reader can test on and wrong on this one — two of the five hashed probes use `copyEngines == 2`. (d) A `cudaDeviceSynchronize()` at the bottom of the loop body restores exactly the serial time and still passes every correctness check. |
| `exercise02.cu` | Debugging + fill-in + prediction (§6 types 3, 1, 2) | 5 | A correct, sanitizer-clean, three-cause 16-chunk pipeline that runs at **0.81× of serial**. Name all three causes, repair them, build the overlap instrument, predict what `--default-stream per-thread` cures. 10 points. | **Four.** (a) The three causes **mask each other almost perfectly** — fixing only the pageable buffer leaves 0.81× unchanged, so the first correct fix looks like it did nothing. (b) The three-argument `cudaMemsetAsync` defaults to stream 0; the fourth argument is the whole bug and reads as a stylistic omission. (c) Of the eight candidate diagnoses, #1 (grid too small) and #8 (the driver inserts a dependency) both describe real phenomena and are the two that catch people — the grid here is 77 blocks/SM, far too *large*, and the driver tracks no read/write sets at all. (d) Having removed the in-loop `cudaMalloc`, the obvious repair passes the bare `g_dScratch` base pointer, so all 16 chunks alias one prefix; the harness checks each chunk's own slice, so this fails deterministically rather than racily. |
| `exercise03.cu` | Design from scratch + performance reasoning (§6 types 5, 6) | 5 | Express a 9-node DAG (two 3-node chains → join → reduce → D2H) in the minimum streams and events, compute its critical path, then measure it at 16 and at 640 blocks per kernel. 10 points. | **Three.** (a) `cudaEventRecord` captures what is **already issued**; recording before chain B's kernels leaves the wait satisfied immediately, and at SMALL the resulting race is essentially never lost — the harness catches it structurally (partial-for-partial against the serial graph) rather than statistically. (b) The minimum is 2 streams and **1** event, not 3 and 2: the join needs no stream of its own and `n2→n6` becomes free if the join goes in chain A's stream. Over-claiming scores zero even though the program then works. (c) The two prediction buckets differ (**3** and **1**) for a reason that is in Module 1 and Module 19, not in this module — identical correct code measures **1.79× (102% of its bound)** at 0.4 blocks/SM and **1.05× (59%)** at 16 blocks/SM. |

**Scoring.** Ex1 10 points (2 `chunkOf` structure over six `(n, nChunks)` pairs
incl. degenerate · 2 hashed `pipelineBound` over five probes, two with
`copyEngines == 2` · 2 correctness incl. never-written and last-element checks ·
2 ≥70% of the reader's **own** bound · 2 prediction bucket). Ex2 10 points
(3 hashed order-independent diagnosis · 2 hashed `overlapFactor` · 3 repair:
correct + clip counters + per-chunk scratch slices + ≥1.35× · 1 + 1 the two
predictions). Ex3 10 points (2 hashed `MIN_STREAMS`/`MIN_EVENTS` · 2 hashed
`criticalPath` over four probes · 2 correctness, both sizes, vs the serial graph
plus an independent host anchor · 2 SMALL ≥70% of the reader's own
critical-path bound · 1 + 1 the two buckets). **`OVERALL: PASS` requires full
marks in all three.**

**Operating-point discipline.** Every scored ratio in all three exercises is
computed between samples taken inside **one rotated sweep** with
`SWEEPS >= NCFG`, which spec §12.5b exempts from needing a separate
operating-point guard. This was validated the hard way: the whole module was
first measured at a **30 W** power cap (SM clock pinned at 210 MHz) and then
re-measured at the healthy 72.8 W. Absolute figures moved by **8×**; the
achieved-fraction-of-own-bound moved by **1–2 points** and every gate and every
prediction bucket held at both operating points.

## Assumed from earlier modules

- **M1** — waves, tail effects, blocks are indivisible and non-migrating,
  the GigaThread placement gate, the pushbuffer/async-launch path. **This is
  the module that predicts where streams stop helping.**
- **M2** — the four launch parameters (the 4th was explicitly deferred here),
  asynchronous launch and its measured host cost, `cudaDeviceSynchronize`,
  two-part error checking, sticky vs non-sticky errors, `CHECK` macros,
  `cudaEvent_t` timing with warm-up.
- **M3** — grid sizing, bounds guards.
- **M4** — pinned vs pageable host memory, `cudaMallocHost`/`cudaFreeHost`, and
  the measured 1.04–1.15× raw H2D ratio on this driver.
- **M11/M12** — the streaming ceiling and the 1500 ms warm-up requirement.
- **M19** — the four-limiter occupancy model, used to compute that 128-thread
  blocks cap at `min(24, 48/4) = 12` blocks/SM.
- **M20** — dependent-FFMA latency/issue behaviour, which is why the payload
  kernels' duration is a clean function of `iters` and warps-per-scheduler.
- **M22** — Nsight Systems. Used here purely as an instrument
  (`cuda_api_sum`, `cuda_gpu_mem_time_sum`); not re-taught.

## Forward references made

- **Module 25 (events)** — events are used throughout for cross-stream
  dependencies and labelled "Module 25 covers events properly" in every file
  that uses them. M25 owes: `cudaEventElapsedTime` semantics across streams,
  `cudaEventSynchronize`/`cudaEventQuery` in their own right, event pools, and
  the `cudaEventBlockingSync`/`cudaEventDisableTiming` flag space.
- **Module 26 (pinned memory)** — `cudaMallocHost` is used in anger here and the
  bounce-buffer mechanism is asserted, not derived. M26 owes: the page-pinning
  cost, `cudaHostAlloc` flags (`Portable`/`Mapped`/`WriteCombined`),
  `cudaHostRegister`, zero-copy, and why pinned memory is a scarce resource.
  M24 supplies the motivation measured: **1.07× of bandwidth, 1.7× of overlap.**
- **Module 28 (CUDA graphs)** — named twice as the answer when the host-side
  issue cost of a many-node, short-node graph starts to dominate (the right-hand
  tail of the chunk-count curve, and Exercise 3's suggested variation).
- **Module 27 (unified memory)** — named once, in
  `check_your_understanding.md` Q3, as the thing people confuse with
  dependency tracking.
- MPS and multi-context time-slicing named once, in Q4, without a module number.

## Measured results this module contributes (for `CROSS_MODULE_INDEX.md` §6b)

| Quantity | Measured |
|---|---|
| `asyncEngineCount` | **1** — one DMA engine for both directions. Three-way overlap is **not** achievable on this part. |
| `concurrentKernels` | 1 (boolean) |
| Stream priority range | `[0, -5]`, 6 levels, lower = higher priority |
| H2D / D2H, 48 MB **pinned**, healthy | **12.0–12.1 / 13.0–13.1 GB/s** |
| H2D / D2H, 48 MB **pageable** | 11.4 / 12.2 GB/s — **raw ratio only 1.07×**, confirming M4 |
| **Copy/kernel overlap, isolated: pinned** | **1.63–1.67×** of an ideal 1.73–1.79× (93%) |
| **Copy/kernel overlap, isolated: pageable** | **0.979–0.986× — none at all** |
| `cudaMemcpyAsync` **host-side** API duration (`nsys`) | **205.6 µs pageable vs 11.7 µs pinned**, same bytes — 17.6× |
| Pipeline bound, 48 MB, `K ≈ H+D` | **1.65–1.70×** (would be 2.53× with 2 copy engines) |
| Measured 16-chunk / 4-stream pipeline | **1.52–1.68× = 89–94% of its own bound** |
| Same pipeline from **pageable** host memory | **1.15–1.17×** |
| **Obvious issue order `H2D(k),K(k),D2H(k)`** | **0.87× — slower than serial.** Head-of-line blocking on one copy engine. |
| Deferred-D2H / two-phase orders | 1.52× / 1.48× |
| Chunk-count curve (order 2) | 1→1.01, 2→1.42, 4→1.50, 8→1.54, **16→1.55**, 32→1.48, 64→1.39 |
| One kernel on the **legacy default stream**, per chunk | **0.81×** — 1.89× of the win destroyed by one missing argument |
| `cudaMalloc`+`cudaFree` per chunk | **0.91×**; `nsys`: 71 `cudaFree` calls at **717 µs average** |
| 4-byte **blocking** `cudaMemcpy` per chunk | **0.86×** |
| `cudaStreamNonBlocking` cure | restores 1.52–1.53× with the default-stream poison still present |
| **`--default-stream per-thread` cures** | default-stream kernel **yes** (0.81→1.53×), blocking `cudaMemcpy` **yes** (0.86→1.53×), `cudaMalloc` in loop **no** (0.91→0.91×). **Per-thread cures default-stream syncs, not API syncs.** |
| **Two independent kernels, one per stream** (`both/one`) | 1 blk **1.001** · 10 **1.003** · 20 **1.005** · 40 (1/SM) **1.016** · 80 **1.251** · 240 **1.697** |
| **DAG speedup vs critical path, 16 blocks/kernel (0.4 blk/SM)** | **1.76–1.79× = 100–102% of bound** |
| **DAG speedup vs critical path, 640 blocks/kernel (16 blk/SM)** | **1.008–1.050× = 49–59% of the same bound.** Identical code. |
| Missing cross-stream event, 4 Mi-element join | **4062 of 4107 sampled elements wrong** — a race, not a hang, and intermittent |
| ⚠️ **Power-cap episode** (battery at 8%, limit 30 W not 60 W) | SM clock pinned **210 MHz**, memory **405 MHz**, `SW_POWER_CAP`+`SW_THERMAL_SLOWDOWN` asserted. H2D **1.5 GB/s**, kernel 10× slower. **Ratios inside a rotated sweep survived to 1–2 points; absolute figures were 8× off and the pageable-vs-pinned conclusion inverted** (pageable measured 1.65× vs pinned 1.73× under the cap, vs 1.17× vs 1.53× healthy). Check `nvidia-smi -q -d POWER \| grep "Current Power Limit"`. |

## Proposed corrections to shared files

1. **`AUTHORING_SPEC.md` §12** — add a clause to rule 5b: *the power limit
   itself is an operating-point variable on this part, not just the clock.*
   `nvidia-smi --query-gpu=clocks.sm` alone does not reveal a 30 W cap;
   `nvidia-smi -q -d POWER` does. The cap was observed moving between 30 W and
   72.8 W as a function of battery charge state, with the GPU stuck at 210 MHz
   under 100% utilization — a state the existing "recovers after 20–40 s idle"
   note does **not** describe.
2. **`CROSS_MODULE_INDEX.md` §7** — the entry "Pinned vs pageable H2D on this
   Windows/WDDM driver measures only 1.04–1.15×, not the folklore 2×" is
   confirmed (1.07×) but is **incomplete and currently reads as a reason not to
   bother**. Append: *and the raw ratio is not why pinned memory matters —
   `cudaMemcpyAsync` out of pageable memory does not overlap at all
   (0.979× vs 1.666×, M24).*
3. **`CROSS_MODULE_INDEX.md` §6 (themes used)** — add: chunked
   H2D/compute/D2H overlap pipeline, the three-poison pipeline, the 9-node
   two-chain DAG. Remove nothing.
4. **`RESUME_HERE.md`** — Module 24 is complete and verified; update the table.
