# CUDA: A Rigorous Course (2026 edition)

Target machine for this course: **NVIDIA RTX 3500 Ada Generation Laptop GPU**,
compute capability **8.9**, 40 SMs, 1536 threads/SM, 432 GB/s, 48 MB L2,
**CUDA 13.2**.

Everything is written for `-arch=sm_89` unless a module explicitly says
otherwise. Modules covering Hopper/Blackwell-only features (thread block
clusters, distributed shared memory, TMA, `wgmma`) are marked
**ARCHITECTURE-SPECIFIC** and come with a runnable sm_89 fallback.

## Layout

```
cuda-course/
  moduleNN/exerciseMM.cu              <- incomplete, you fill the TODOs
  solutions/moduleNN/exerciseMM_solution.cu
  solutions/moduleNN/exerciseMM_solution.md
```

Do not open `solutions/` until you have submitted an attempt.

## Build convention

```
nvcc -arch=sm_89 -o exerciseMM.exe exerciseMM.cu
.\exerciseMM.exe
```

Extra flags appear per-exercise (`-lineinfo`, `-Xptxas -v`, `-O3`, `--ptx`, `-G`).

**libcu++ on MSVC.** Any file using `cuda::atomic_ref`, `cuda::barrier`,
`cuda::pipeline` or other libcu++ headers needs:

```
nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler /Zc:preprocessor -o <name>.exe <name>.cu
```

`/Zc:preprocessor` selects MSVC's conforming preprocessor, which libcu++
requires. Files that need it say so in their header comment.

**Sanitizer note (verified on CUDA 13.2 / sm_89):** `compute-sanitizer --tool
synccheck` detects **nothing** on divergent-barrier bugs — every shape tried
reported zero errors. Use `racecheck` for races and
`initcheck --initcheck-address-space shared` for uninitialized shared reads;
the latter was the only tool that found one of Module 9's planted bugs.

## Roadmap

### Part I — Fundamentals
| M | Title | Skill developed |
|---|-------|-----------------|
| 1 | Why GPUs | Throughput-vs-latency mental model; what physically happens at kernel launch |
| 2 | First CUDA program | nvcc, execution-space qualifiers, launch syntax, rigorous error checking |
| 3 | Thread hierarchy | Mapping any problem geometry onto grids/blocks; flattening; bounds |

### Part II — Memory system
| M | Title | Skill developed |
|---|-------|-----------------|
| 4 | Memory hierarchy | Knowing which physical storage each declaration lands in |
| 5 | Global memory & coalescing | Computing per-warp address footprints and sector counts |
| 6 | Shared memory | Tiling and cooperative loading |
| 7 | Bank conflicts | Diagnosing and fixing conflicts with padding/swizzling |

### Part III — Execution
| M | Title | Skill developed |
|---|-------|-----------------|
| 8 | Warps and SIMT | Reasoning at lane granularity; divergence, predication, reconvergence |
| 9 | Synchronization | Where barriers are required vs. superstition; memory ordering |
| 10 | Races and atomics | Spotting RMW races; privatization to cut contention |

### Part IV — Core parallel algorithms
| M | Title | Skill developed |
|---|-------|-----------------|
| 11 | Vector operations | Grid-stride loops, vectorized loads, bandwidth ceilings |
| 12 | Reduction (6 versions) | The canonical optimization ladder, naive → warp-shuffle |
| 13 | Scan / prefix sum | Work efficiency, Blelloch vs. Hillis-Steele, decoupled look-back |
| 14 | Histogram | Atomic contention, shared-memory privatization |
| 15 | Transpose | Coalescing + shared memory + bank conflicts in one problem |

### Part V — GEMM
| M | Title | Skill developed |
|---|-------|-----------------|
| 16 | Naive GEMM | Arithmetic-intensity analysis from first principles |
| 17 | Tiled GEMM | Blocking for reuse; why partial tiles still compute full dot products |
| 18 | Advanced GEMM | Register tiling, coarsening, vectorized loads, double buffering |

### Part VI — Performance engineering
| M | Title | Skill developed |
|---|-------|-----------------|
| 19 | Occupancy | Register/shared-memory arithmetic; why max occupancy ≠ max speed |
| 20 | Latency hiding | Eligible vs. stalled warps; ILP as an alternative to occupancy |
| 21 | Roofline | Classifying any kernel as compute- or memory-bound before coding |
| 22 | Nsight Systems | Timeline analysis: launch gaps, copies, stream overlap |
| 23 | Nsight Compute | The ~10 metrics that matter; reading a stall breakdown |

### Part VII — Advanced CUDA
| M | Title | Skill developed |
|---|-------|-----------------|
| 24 | Streams | Real concurrency and dependency expression |
| 25 | Events | Correct GPU timing; cross-stream dependencies |
| 26 | Pinned memory | Why pageable copies stage through a bounce buffer |
| 27 | Unified memory | Page faults, prefetch, oversubscription pitfalls |
| 28 | CUDA graphs | Killing launch overhead in short-kernel pipelines |
| 29 | Cooperative groups | Modern, scoped synchronization |

### Part VIII — Warp-level programming
| M | Title | Skill developed |
|---|-------|-----------------|
| 30 | Warp primitives in depth | **Not their introduction** — `__shfl_up/down/sync`, `__ballot_sync`, `__activemask` are already first-class tools by M13. M30 covers `__match_any_sync`, `__reduce_*_sync`, cooperative-group tiles, sm_70+ semantics, and when warp-level beats shared memory |
| 31 | Warp specialization | Producer/consumer warps inside one block |

### Part IX — Asynchronous data movement
| M | Title | Skill developed |
|---|-------|-----------------|
| 32 | Async copy & pipelines | `cp.async` (sm_80+), `cuda::pipeline`, mbarrier, multi-stage prefetch |

### Part X — Tensor Cores
| M | Title | Skill developed |
|---|-------|-----------------|
| 33 | Tensor Core fundamentals | What an MMA actually executes; FP16/BF16/TF32/FP8 and accumulation |
| 34 | WMMA / MMA programming | Fragment layouts; writing a Tensor Core GEMM |
| 35 | Modern tensor pipelines | TMA, `wgmma`, tensor memory — **ARCHITECTURE-SPECIFIC**, conceptual on sm_89 |

### Part XI–XVII
| M | Title | Skill developed |
|---|-------|-----------------|
| 36 | CUDA libraries | cuBLAS(Lt), cuFFT, cuSPARSE, CUB, Thrust, CCCL — and when not to write a kernel |
| 37 | Compilation pipeline | nvcc phases, fatbins, JIT, `-arch` vs `-code` |
| 38 | PTX | Reading generated virtual ISA |
| 39 | SASS | Using `cuobjdump`/`nvdisasm` to explain compiler decisions |
| 40 | Multi-GPU | P2P, UVA, NVLink vs PCIe, NCCL concepts, scaling limits |
| 41 | CUDA for AI | GEMM/conv/norm/softmax/attention as bandwidth problems |
| 42 | LLM inference kernels | RMSNorm, RoPE, softmax, attention, KV cache, quantization, fusion |
| 43 | CUTLASS / CuTe | Seeing every earlier concept as a named layer in a real library |
| 44 | Modern CUDA 2026 | Clusters, DSMEM, modern barriers, CUDA Tile / Tile IR, current toolchain |

### Final projects
P1 GEMM ladder · P2 reduction library · P3 softmax · P4 attention kernel ·
P5 Nsight-driven optimization of a deliberately slow workload.

## Dependency graph

```
M1 → M2 → M3 ─┬─→ M4 → M5 ──┬─→ M6 → M7 ────┐
              │             │               │
              └─→ M8 → M9 → M10             │
                       │                    │
                       ├─→ M11..M15 ────────┤
                       │                    │
                       └─→ M30 (warps)      └─→ M16 → M17 → M18
                                                          │
                       M19..M23 (perf) ←──────────────────┤
                                                          │
                       M24..M29 (async/host) ─→ M32 ──────┤
                                                          │
                                        M33 → M34 → M35 ──┤
                                                          ↓
                                   M36..M39, M40, M41 → M42 → M43 → M44
```

Hard prerequisites:
- **M5 (coalescing) gates everything.** Every later optimization is an argument
  about memory transactions.
- **M6+M7 gate M15 and M17.** Tiling without bank-conflict awareness produces
  kernels that are correct and slow.
- **M8+M9+M10 gate M12–M14.** Reduction/scan/histogram are warp-behavior
  exercises wearing algorithm costumes.
- **M17+M18 gate M33–M35 and M43.** Tensor Cores and CUTLASS are a re-derivation
  of tiled GEMM with different hardware primitives; arriving early means
  memorizing APIs.
- **M19–M23 run alongside Parts IV–V**, not after: you profile from Module 12 on.
- **M32 (async copy) needs M24 (streams) + M9 (barriers)** before it means
  anything.
