# Module 1 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

## TODO 1 — peak global memory bandwidth

```cpp
double peakBW_GBs = 2.0 * (double)memClockKHz * 1.0e3 * (busWidthBits / 8.0) / 1.0e9;
```

Three independent facts multiply together:

1. **Bus width → bytes per transfer.** 192 bits = 24 B move across the DRAM
   interface per transfer.
2. **DDR → two transfers per clock.** GDDR6 latches data on both edges of the
   clock, so transfers/s = 2 × clock. CUDA reports the *command clock*
   (9,001,000 kHz = 9.001 GHz), **not** the effective 18 Gbps data rate. This is
   the factor people forget — and the error is exactly 2×, which is why the
   exercise hints at it.
3. **Decimal GB.** NVIDIA's published bandwidth numbers use 1 GB = 10⁹ B.

On the RTX 3500 Ada Laptop: 2 × 9.001e9 × 24 / 1e9 = **432.0 GB/s**, which
matches the published spec exactly. That is your confirmation the formula is
right.

Why you care: 432 GB/s is the number you will divide into every kernel's byte
traffic for the rest of the course. A kernel that touches 1 GB of distinct data
cannot finish faster than ~2.3 ms on this machine, no matter how clever the
arithmetic is.

## TODO 2 — residency

```cpp
int maxWarpsPerSM  = p.maxThreadsPerMultiProcessor / p.warpSize;   // 1536/32 = 48
int maxWarpsDevice = maxWarpsPerSM * p.multiProcessorCount;        // 48*40  = 1920
long long threadsInFlight = (long long)p.maxThreadsPerMultiProcessor
                          * p.multiProcessorCount;                 // 61,440
```

Common wrong approaches:

- Using `maxThreadsPerBlock` (1024) instead of `maxThreadsPerMultiProcessor`
  (1536). Those are unrelated limits: one caps a *software* block, the other
  caps *hardware residency*. On sm_89 they are deliberately not equal, which is
  why 1024-thread blocks are often a bad choice — only one such block fits per
  SM, wasting 512 thread slots (66.7% occupancy ceiling).
- Multiplying by `maxBlocksPerMultiProcessor` (24). 24 blocks × 128 threads =
  3072 > 1536, so the block limit is not the binding constraint here. Whichever
  limit binds first wins; this is exactly the occupancy arithmetic of Module 19.

The meaning of 61,440: that is the number of thread contexts the GPU keeps
*simultaneously alive* — registers allocated, PC live, ready to be scheduled.
It is not "61,440 threads executing per cycle". Per cycle, each SM issues at
most one instruction per warp scheduler (4 schedulers on sm_89), so the machine
retires at most 4 × 40 = 160 warp-instructions per cycle. The gap between 1920
resident warps and 160 issued warp-instructions per cycle *is* the latency-hiding
budget.

## TODO 3 — launch configuration

```cpp
int nBlocks = 2 * p.multiProcessorCount;   // 80
```

Derived from the device, not hard-coded — the same source file must produce a
different launch on a different GPU. Hard-coding 80 is the mistake to avoid.

## TODO 4 — the store, and whether `%smid` is uniform

```cpp
if (threadIdx.x == 0)
    out[blockIdx.x] = smid();
```

- **Is `%smid` uniform across the block?** Yes. A thread block is assigned to
  exactly one SM at launch and never migrates; all of its warps live and die on
  that SM. So every thread in the block reads the same `%smid`. (Contrast with
  `%warpid`/`%laneid`, which are *not* uniform.)
- **So why guard the store?** Because 128 threads writing the same 4 bytes is
  128 redundant stores that the hardware serializes into a write conflict on one
  address — benign in outcome, wasteful in traffic, and the exact habit that
  becomes a genuine data race the moment the stored value varies per thread.
  Designating a leader thread is the idiom; you will use it constantly (reduction
  finalization, histogram flush, block-level flags).
- **Not a race** in the strict sense here: all writers store an identical value,
  so any winner produces the same result. Say so precisely — "benign write
  conflict, not a data race with a nondeterministic outcome."

A subtler correct answer: `if (threadIdx.x == 0)` diverges within warp 0 only
(lane 0 takes the branch, lanes 1–31 are predicated off for one instruction).
Warps 1–3 evaluate the condition false uniformly and skip. The cost is one
predicated store, not a real divergence penalty.

## Expected output (RTX 3500 Ada Laptop, 40 SMs)

```
  SMs                        : 40
  Max threads / SM           : 1536
  Peak BW                    : 432.0 GB/s
  Max resident warps / SM    : 48
  Max resident warps / GPU   : 1920
  Max threads in flight      : 61440
  [OK] derived residency numbers are self-consistent
  SM  0 : 2 block(s)
  ... (every SM gets exactly 2)
  distinct SMs used = 40 / 40, unwritten/invalid entries = 0
```

## The result that matters

Every SM got exactly 2 blocks. The GigaThread engine distributes blocks
round-robin across all SMs *before* stacking a second block on any SM, because
the whole grid fits in one "wave". This is why the concept of a **wave** —
`ceil(nBlocks / blocksResidentPerSM / nSMs)` — governs tail effects: a grid of
41 blocks on 40 SMs takes the same wall-clock as 80 blocks, since the 41st block
starts a second wave with 39 SMs idle. Grid sizing that ignores waves leaves
half the machine idle at the tail.

Try it: change `nBlocks` to `p.multiProcessorCount + 1` and watch the histogram
go lopsided.
