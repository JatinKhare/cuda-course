# Module 04 / Exercise 01 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

Runtime is about 6 s; most of it is the two 192 MB / 384 MB probes, which
perform several million strictly serialized DRAM loads.

## TODO 1 — the stride

```cpp
int strideBytes = 128;   // one full L1/L2 cache line
```

The requirement was: *following one link must always force a new request to the
memory system*. Caches are managed in lines, not words. On sm_89 the L1/L2 line
is 128 B (sectored into four 32 B sectors, which is the granularity of the
L2↔DRAM transfer). If two consecutive nodes of the chain fall in the same line,
the second load is a guaranteed hit no matter how large the buffer is, and every
measurement becomes a weighted average of "line miss" and "31 free hits".

128 B is the *smallest* correct answer. Anything smaller under-reports; anything
much larger wastes buffer and, as discussed at the end, starts to measure
address translation.

**What the wrong answer looks like.** Set `strideBytes = 4` (adjacent `int`s) and
walk the chain in address order instead of randomly, and the entire hierarchy
disappears:

```
     bytes   cyc/load
      8192       40.0
     32768       65.2
   1048576       65.2
  16777216       65.2
  67108864      100.3
 536870912      105.0
```

A "DRAM" latency of 105 cycles. What you measured is 1 miss amortized over 32
hits in the same line: 1/32 × 575 + 31/32 × 40 ≈ 57, plus loop overhead. You can
produce a beautiful, completely meaningless graph this way, and people do.

## TODO 2 — the probe sizes

```cpp
size_t L2 = (size_t)l2Bytes;                 // 50 331 648
size_t probeBytes[NPROBE] = {
    L1_BYTES / 4,      //  32 KB  -- 1/4 of L1
    L1_BYTES / 2,      //  64 KB  -- 1/2 of L1
    L1_BYTES * 8,      //   1 MB  -- 8x L1, 1/48 of L2
    L2 / 6,            //   8 MB  -- still far inside L2
    L2 * 4,            // 192 MB  -- 4x L2
    L2 * 8             // 384 MB  -- 8x L2
};
```

Two design rules are being applied.

**Derive, do not hard-code.** `l2Bytes` comes from
`cudaDeviceGetAttribute(..., cudaDevAttrL2CacheSize, ...)`. The same source file
must produce a correct experiment on a GPU with a 6 MB L2.

**Leave margin, especially above L2.** A cache is not a cliff; it is a hit-rate
curve. Choosing "50 MB, which is bigger than 48 MB" would give you a buffer with
a substantial residual hit rate and a latency somewhere between the L2 and DRAM
plateaus — and you would report that number as "DRAM latency". 4× and 8× L2 give
a residual hit rate near 1/4 and 1/8 respectively for a random walk, which is
why the two probes still agree to about 12%.

The same margin argument applies at the bottom. 128 KB is the *whole* unified
L1+shared block, and some of it is never available as data cache; probing at
128 KB would straddle the boundary. 32 KB and 64 KB are unambiguous.

**This is the trap the exercise exists for.** 48 MB is large enough to hold the
entire working set of almost any benchmark anyone writes casually. If you had
picked 8 MB and 32 MB as your "DRAM" sizes — both perfectly reasonable-sounding
"big" buffers — you would have measured 242 cycles and concluded that global
memory latency on Ada is 242 cycles. It is not; that is L2.

## TODO 3 — steady state

```cpp
    // one full untimed traversal
    for (int i = 0; i < steps; ++i) p = buf[p];

    long long t0 = clock64();
    for (int i = 0; i < steps; ++i) p = buf[p];
    long long t1 = clock64();
```

Every line of the chain is touched once before the clock starts, so the timed
pass sees the steady-state hit/miss distribution that this working-set size
produces, not the compulsory cold misses that every size produces equally.

Because `steps == nodes` for all but the largest probe, the warm-up pass is a
complete traversal: the "working set" in the measurement really is the whole
buffer, not just the subset the chase happened to visit.

Without it, the numbers are not merely shifted, they are incoherent:

```
    bytes      nodes   cyc/load   (no warm-up)
     8192         64       43.2
    65536        512       65.3
  1048576       8192      316.1
  4194304      32768      242.3
 33554432     262144     1224.5
268435456    2097152      614.3
```

Note it is not even monotone. The small sizes are contaminated by cold misses
(which is why 8 KB reads 43 instead of 40), and the mid sizes are dominated by
whatever the previous iteration left in L2.

**Common wrong approach:** warming up by doing `cudaMemcpy` of the buffer, or by
launching the kernel twice. Neither works. The H2D copy writes through L2 with a
streaming policy and does not populate L1; a second launch starts a new kernel,
and the kernel-boundary invalidate flushes L1. The warm-up must be inside the
same kernel invocation, on the same SM, immediately before the measurement.

## TODO 4 — predictions

```cpp
double predL1   = 40.0;
double predL2   = 250.0;
double predDRAM = 500.0;
```

Reasoning that gets you within 2× without looking anything up:

- **L1** must be small — it is on the same SRAM block as shared memory and is
  physically adjacent to the schedulers. An FMA has ~4-cycle latency; an L1 hit
  is the next tier up, tens of cycles, not hundreds. 20–50 is the plausible band.
- **L2** is on-die but across the interconnect from the SM. Call it 5–8× L1.
- **DRAM** you can derive from the machine's own design. The SM can hold 48
  resident warps. For the GPU to hide global latency at all, the latency must be
  roughly within reach of 48 warps × a few independent loads each. Published
  figures for Ampere/Ada land at 400–600 cycles, and the module brief gives that
  range; 500 is the midpoint.

## Synchronization / memory reasoning

There is none, deliberately. One block, one thread, one warp. Any concurrency
would introduce memory-level parallelism, and MLP is exactly what a latency
measurement must exclude: with two independent loads in flight, the second one's
latency is hidden behind the first and the average halves. The dependency
`p = buf[p]` makes that structurally impossible — the address of load *n+1* is
the result of load *n*.

`*out = p` at the end is not decoration. Without a use of `p` that escapes the
kernel, dead-code elimination deletes the entire chase and you measure the
latency of `clock64()`.

## Performance reasoning

The chain is randomized *within* 2 MB chunks and the chunks are visited in
address order. This is not arbitrary. A uniformly random walk over a 384 MB
buffer misses the GPU's address-translation caches on nearly every link, and you
measure page-table walks instead of the memory hierarchy. Measured here with a
uniformly random chain:

```
 201326592    1572864    1572864         1427.5
```

1427 cycles/load, against 583 with the chunked chain. That is a real and
important effect — it is why large, sparse, pointer-heavy GPU data structures
disappoint — but it is not the memory hierarchy, and mixing the two into one
number teaches you nothing about either.

Conversely, keeping the randomness local does not weaken the experiment: a
dependent load cannot be prefetched by construction, so the in-order chunk
sequence costs nothing.

## Expected output

Actual run on the RTX 3500 Ada Laptop GPU. The L1 and L2 numbers are stable to
better than 1% run to run; the DRAM numbers vary by roughly ±10% (observed range
567–682 across runs), because at 4–8× L2 the residual hit rate depends on the
random chain.

```
=== NVIDIA RTX 3500 Ada Generation Laptop GPU : L1+SMEM = 128 KB/SM, L2 = 48 MB ===

 stride = 128 B
  buffer bytes        nodes      steps    cycles/load
         32768          256        256           40.7
         65536          512        512           40.3
       1048576         8192       8192          241.3
       8388608        65536      65536          241.3
     201326592      1572864    1572864          583.6
     402653184      3145728    3145728          566.9

--- predicted vs measured (cycles / dependent load) ---
  regime    predicted     measured      ratio
      L1           40         40.5      1.01x
      L2          250        241.3      0.97x
    DRAM          500        575.3      1.15x

PASS
```

Three flat plateaus, two cliffs, ratios of 6.0× and 2.4×. The cliffs land where
the capacities say they should: between 64 KB and 1 MB (the 128 KB L1), and
between 8 MB and 192 MB (the 48 MB L2).

## The result that matters

**Forty, two hundred and forty, and five hundred and seventy-five.** Those three
numbers are the cost of a value you did not keep close enough, and every
optimization in the rest of this course is an attempt to move an access one step
up that ladder. The second thing to take away is methodological: the 48 MB L2 on
this GPU is big enough to make a carelessly sized benchmark report L2 numbers
and call them DRAM — you must size past it deliberately, every time.

**Variation to try:** rebuild with `strideBytes = 32`. Nothing much happens —
40.2 / 252.4 / 567.7 — because 32 B is one full sector and each link still costs
a distinct sector fill. Now go to `strideBytes = 4` *and* replace the in-chunk
shuffle with an in-order chain. The plateaus collapse to 40 / 65 / 100 cycles
and the hierarchy vanishes. Work out, from the 128 B line and the 32 B sector,
why 32 B was harmless and 4 B was fatal; that is the same arithmetic you will
use for coalescing in Module 5.
