# Module 5 manifest

## Concepts taught

- **Coalescing** — the hardware merging of a warp's 32 addresses into the
  minimum covering set of memory blocks. **PORTABLE CUDA CONCEPT.**
- **Sector (32 B)** — the fill/DRAM-traffic granularity on sm_89.
  **ARCHITECTURE-SPECIFIC.**
- **Cache line (128 B = 4 sectors)** — the tag/replacement granularity;
  distinguished from the sector throughout. **ARCHITECTURE-SPECIFIC.**
- **Per-warp, per-instruction analysis** — coalescing is not per thread and not
  per block; warps cannot merge with each other.
- **Address set, not sequence** — lane order is invisible to the coalescer;
  demonstrated with a reversed-lane kernel that measures identically to the
  contiguous one.
- **Inactive lanes supply no address** — predicated-off lanes cannot force a
  sector.
- **The counting procedure** — enumerate 32 addresses → `addr >> 5` → count
  distinct → `bytes_moved = 32 * S`.
- **Memory efficiency** = `bytes_requested / (32 * distinct_sectors)`.
- Worked efficiencies: contiguous aligned 100 %; misaligned by 4 B 80 %;
  stride-2 50 %; stride-6 (AoS `.x`) 16.7 %; stride ≥ 8 floats 12.5 %
  (saturation at one sector per lane); broadcast 1 sector.
- **Alignment** — `cudaMalloc` guarantees ≥ 256 B; three distinct thresholds:
  16 B (legality of 128-bit access), 32 B (sector, cost), 128 B (line).
- **`cudaErrorMisalignedAddress`** — the failure mode of a 128-bit access on a
  non-16 B-aligned pointer.
- **Vectorized loads** (`float2`/`float4`/`int4`, `reinterpret_cast`) — same
  sectors, fewer instructions and fewer outstanding-request slots; a
  latency/issue optimization, not a bandwidth one.
- **Tail handling** for vectorized kernels when `N % width != 0`.
- **AoS vs SoA**, and the correction to the folklore: the penalty is
  `record_size / bytes_used_per_record`, so AoS costs 6.6× when a kernel reads
  one field of a 6-field struct and 1.35× when it reads all of them.
- **Write coalescing and write-allocate** — partial-sector stores force a
  read-modify-write, so a scattered write costs roughly twice a scattered read;
  a "write-only" kernel can generate DRAM reads.
- **Effective vs DRAM bandwidth** — `dram = effective / efficiency`; a strided
  kernel showing 47 GB/s effective while the bus runs at 375 GB/s.
- **Row pitch / leading dimension** — why padding a record to a multiple of 4
  floats (legal for `float4`) is not enough, and a multiple of 8 (32 B) or 32
  (128 B) is needed.
- **Boundary-to-interior ratio** — misalignment is nearly free in a streaming
  sweep (neighbouring warps share the boundary sector, which hits in cache) and
  genuinely costly when contiguous runs are short and gapped.
- **`const T* __restrict__` and `__ldg`** — read-only path and load hoisting;
  affects scheduling, not sector counts. (Brief.)
- **L1 is not coherent across SMs** — why inter-block communication cannot use
  L1. (Brief, forward-referenced.)
- **Benchmarking methodology** — buffers ≫ the 48 MB L2 (an SoA field of 32 MB
  reports 1360 GB/s, 315 % of DRAM peak); duration-based warm-up because the
  laptop memory P-state ramps 6001 → 8001 → 9001 MHz (288 / 384 / 432 GB/s);
  software power cap (`0x4`) under sustained load; measuring a streaming
  ceiling in the same loop and reporting `% of stream` rather than only
  `% of nominal peak`.

## CUDA API / intrinsics / syntax introduced

- `float2`, `float4`, `int4` and their `__align__(16)` requirement
- `reinterpret_cast<float4*>` on device pointers
- `cudaErrorMisalignedAddress`
- `const T* __restrict__` on kernel parameters; `__ldg()` (mentioned)
- `cudaMemcpy2D` (used in the Exercise 3 validator to sample strided rows)
- `cudaMemcpyDeviceToDevice` (state restore that does not stall the GPU)
- `uintptr_t` address arithmetic on device pointers, host side
- `nvidia-smi --query-gpu=clocks.mem,clocks_throttle_reasons.active`

## Exercises

| File | Type | TODOs | One-line description | Subtle trap |
|---|---|---|---|---|
| `exercise01.cu` | Predict-the-behavior + address calculation | 4 | Fill a paper table of addresses/sectors/efficiency for 7 patterns, implement the counting procedure, then compare predicted vs measured bandwidth ratios | The misaligned (`+1 float`) case measures 98 % of contiguous, not the 80 % the sector model predicts — warp *k*'s 5th sector is warp *k+1*'s 1st and hits in cache. The harness flags it and refuses to explain it. Also: TODO 4's obvious answer `k=1` raises `cudaErrorMisalignedAddress`. |
| `exercise02.cu` | Optimization (v1→v2→v3) | 4 | AoS particle integrator → SoA → `float4`, timed and validated against a CPU reference | `N = 16,000,003`, so `N % 4 == 3`. The tail must be handled *outside* the `i < nVec` branch and with its own range test; the placement `if (i >= nVec && ...)` is correct-looking and processes nothing, because `GRID_V*256 == nVec` exactly. `nVec = (N+3)/4` overruns the allocation. |
| `exercise03.cu` | Design ("decide the strategy") | 4 | A kernel whose indexing is already perfectly contiguous is still slow; the reader must locate the defect in the data layout and choose a row pitch | `MIN_PITCH_16B = 68` satisfies the rule everyone remembers ("multiple of 4 so `float4` works"), improves the measurement from 83.5 % to 91.5 %, and is still 11 % short because `68*4 % 32 == 16`. Also: the two 100 %-efficient pitches measure 6 % apart, which the sector model cannot explain (128 B line straddling, DRAM page locality). |

## Assumed from earlier modules

- M1: SM count (40), warp = 32 lanes, the warp scheduler as issue unit,
  latency hiding via resident warps, 432 GB/s peak, 48 MB L2.
- M2: `nvcc -arch=sm_89`, `__global__`, launch syntax, the `CHECK` macro
  idiom, checking both `cudaGetLastError()` and `cudaDeviceSynchronize()`.
- M3: global thread index from `blockIdx`/`blockDim`/`threadIdx`, 64-bit index
  arithmetic, bounds guards, grid sizing with `ceil`.
- M4: the memory hierarchy — registers, L1, L2, global; `cudaMalloc`;
  `cudaMemcpy`; which declaration lands in which storage.

## Forward references made

- **Shared memory** — named as the fix for patterns where neither reads nor
  writes can be coalesced with a thread-per-element mapping; said Module 6
  covers it (lesson "Check Your Understanding" Q4, Exercise 3 header).
- **Bank conflicts** — contrasted with the broadcast case (32 lanes, same
  address, no serialization) against shared-memory same-bank/different-address
  serialization; said Module 7.
- **`__syncthreads()` / barriers** — mentioned only to say the vectorized
  kernel's two phases need none, and that a barrier in divergent control flow
  is a hang; said Module 9 makes it precise.
- **Atomics** — mentioned in prose as the mechanism for inter-block
  communication that L1's lack of cross-SM coherence forces; said Module 10.
- **Transpose** — the AoS→SoA conversion done on the device, and the
  deinterleave pattern, identified as transpose problems; said Module 15 owns
  them.
- **Nsight Compute metrics** — `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
  named as the hardware counter equal to the number computed by hand here, and
  `lts__t_sectors` / `dram__sectors` as the way to separate the residual 6 % in
  Exercise 3; said Module 23.
- **Occupancy / latency hiding** — the observation that vectorizing helps a
  latency-bound kernel more than a bandwidth-bound one points at Modules 19–20.
- **Roofline** — the "DRAM is saturated, stop requesting unwanted sectors"
  conclusion is the memory-bound side of Module 21.
- **Grid-stride loops** — assumed from M3 and mentioned as a red herring for
  coalescing (a grid-stride loop is still coalesced).
