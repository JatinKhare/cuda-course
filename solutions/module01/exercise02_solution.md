# Module 01 / Exercise 02 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

Warning-clean. Runtime ~5 s, most of it the host-side FMA replay in the
validation pass.

---

## TODO 1 — how many blocks make one wave

```cpp
int blocksPerSM = 0;
CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocksPerSM, (const void*)fma_chain, THREADS, 0));
const int waveBlocks = blocksPerSM * nSMs;     // 1 * 40 = 40
```

**Why this is the right question to ask the runtime.** Block placement is gated
by four independent resources (Module 1, §8, step 3): a block slot, thread/warp
slots, registers, and shared memory. `blocksPerSM` is the minimum over all four
*for this kernel*. Only the runtime knows the third one, because only the runtime
knows how many registers `ptxas` gave the kernel. Any hand-derived number is a
guess that silently goes stale the moment you add a variable to the kernel.

Here the binding limit is thread slots: 1536 threads per SM / 1024 threads per
block = 1. It is not registers (this kernel uses a handful) and it is not the
block-slot ceiling.

**Common wrong approaches and the symptom each produces:**

| Wrong choice | Value | Symptom |
|---|---|---|
| `p.maxBlocksPerMultiProcessor` | 24 | `waveBlocks` = 960. The "configurations" become 480/959/960/961/1920/1921 blocks, all of which are many real waves. Every ratio in the table collapses toward 1.0 and the staircase vanishes. |
| `nSMs` (assume 1 block/SM always) | 40 | Right *by accident* for a 1024-thread block. Change `THREADS` to 256 and it is wrong by 6x. |
| `p.maxThreadsPerMultiProcessor / THREADS` | 1 | Right here, wrong for any kernel whose register or shared-memory footprint binds before thread slots do. This is the one that bites in Modules 17–19. |

Note the deliberate trap in the file: `maxBlocksPerMultiProcessor` **is** a real
device property named exactly after the thing you want, and it is exactly the
wrong one. It is an unconditional architectural ceiling, not an answer about your
kernel.

## TODO 2 — waves and wave efficiency

```cpp
const int    waves   = (nBlocks + waveBlocks - 1) / waveBlocks;
const double waveEff = (double)nBlocks / ((double)waves * waveBlocks);
```

`waves` is a **ceiling** division. A partly-filled wave is still a whole wave —
that is the entire point of the exercise. `nBlocks / waveBlocks` (floor) reports
1 wave for 41 blocks and is rejected by the self-check
`(waves-1) * waveBlocks < nBlocks <= waves * waveBlocks`.

`waveEff` must be computed in floating point. `nBlocks / (waves * waveBlocks)` in
integer arithmetic is 1 when the grid is an exact multiple of the wave and 0
otherwise — the self-check `waveEff > 0.0 && waveEff <= 1.0` catches the 0 case
but *not* the case where the division happens to be exact, which is why the
column must be inspected, not just the PASS line.

Interpretation: `waveEff` is the fraction of the machine's resident-block
capacity carrying real work, averaged over the launch. For 41 blocks with a
40-block wave it is 41/80 = 51.2%. For 81 blocks it is 81/120 = 67.5%.

## TODO 3 — achieved GFLOP/s

```cpp
const double gflops = 2.0 * (double)nBlocks * THREADS * (double)ROUNDS
                      / (ms * 1.0e-3) / 1.0e9;
```

Three things to get right:

1. **One FMA is two FLOPs.** `fmaf(v, a, b)` is a multiply and an add. Counting
   it as one halves every number and puts you at ~42% of peak when the machine is
   actually at ~83%.
2. **Count the work requested, not the capacity.** The FLOP count uses
   `nBlocks`, the blocks actually launched — not `waves * waveBlocks`. Idle block
   slots in a partial wave do no work; counting them would make wave efficiency
   invisible, which is exactly the effect we are trying to see.
3. **`ms` is per-launch**, already divided by `ITERS`.

The cross-check that tells you this is right: the 40-block row and the 80-block
row, both at 100% wave efficiency, report the *same* GFLOP/s to within noise
(17395.1 and 17395.1 in the run below). If those two disagree, the FLOP count is
wrong.

---

## Synchronization / memory reasoning

There is none to speak of, and that is deliberate. This kernel has one load, one
store, and 16384 register-resident FMAs per thread. Arithmetic intensity is about
`2 * 16384 FLOP / 8 B = 4096 FLOP/byte`, so the DRAM interface contributes
nothing to the runtime.

That matters because the wave structure is only cleanly visible when **block
duration is independent of how many other blocks are running**. For a
compute-bound kernel with one block per SM, a block's runtime depends only on its
own SM's issue bandwidth, so it is genuinely constant — hence a clean staircase.
Make the kernel memory-bound and blocks start competing for a *shared* resource
(DRAM bandwidth): a block running alone in the tail wave gets the whole memory
system to itself and finishes much faster than it would have in a full wave, so
the staircase smears into a ramp and the tail looks cheaper than it is. It is not
cheaper — the machine is just as idle — but the wall-clock signature is muddier.
(This is prediction P5.)

The one real synchronization subtlety is on the host side: `cudaEventRecord` /
`cudaEventSynchronize`, never `clock()` around an asynchronous launch. And
`cudaGetLastError()` after the launch loop plus a synchronizing call, because a
launch error and an execution error surface at different times.

---

## Performance reasoning — the predictions

**P1. `t(wave + 1) / t(wave)` = 2.0, near exactly.**
41 blocks need two waves. The first 40 run concurrently, one per SM; the 41st
cannot start until a block slot frees, which happens only when one of the first
40 finishes. Since all blocks take the same time, the 41st starts at `T` and ends
at `2T`, with 39 SMs idle for the entire second half. Measured: **1.99x**.

That is a 2.5% increase in work bought at a 100% increase in time. The marginal
cost of the 41st block is 40 blocks' worth of machine-time.

**P2. `t(half a wave) / t(wave)` = 1.0.**
This is the ratio that surprises people. 20 blocks occupy 20 SMs; the other 20
SMs have nothing to do and sit idle. Halving the work does not halve the time
because the resource you removed work from was not the bottleneck — it was
already idle. Measured: **1.00x**, with GFLOP/s halved (8720 vs 17395).

**P3. `t(2 waves) / t(wave + 1)` = 1.0.**
80 blocks and 41 blocks take the same wall-clock time — 0.1543 ms vs 0.1534 ms.
You can double the work of the 41-block launch for free. This is the single most
actionable fact in the module: if your grid has a tail, the tail is already paid
for, and you should fill it.

**P4. Best GFLOP/s: the exact-wave configurations (40 and 80), at ~83% of peak.
Worst: 20 and 41, at ~43%.**
`%peak` tracks `waveEff` almost exactly — 50.0% -> 41.8%, 97.5% -> 81.4%,
100% -> 83.3%, 51.2% -> 42.9%, 67.5% -> 57.2%. The constant offset of ~0.83 is
not wave-related: it is the loop overhead of this kernel. `ptxas` unrolls the FMA
loop by 16, so each group of 16 `FFMA` instructions carries one `IADD3`, one
`ISETP` and one `BRA` — 16 useful instructions out of 19 issued, i.e. 84%. The
dependent-FMA chain itself is fully hidden: with 32 warps per SM there are 8
warps per scheduler and FFMA latency is ~4 cycles, so the chain never limits
issue.

**P5** is answered under "Synchronization / memory reasoning" above.

### Why the file measures the SM clock

`cudaDevAttrClockRate` reports 1.545 GHz on this part. The actual SM clock during
the sweep was **2.039 GHz** in the run below, and **1.480 GHz** in a run taken
after the GPU had been loaded for a minute — a 38% swing. A `%peak` computed from
the nominal clock is therefore meaningless, and can exceed 100%.

The kernel records `clock64()` around its body in block 0, thread 0. For the
exactly-one-wave configuration, block 0 is resident for the whole kernel, so its
cycle count spans exactly the interval the host timed, and `cycles / ms` is the
average SM clock during that measurement. Normalizing against that makes `%peak`
reproducible: across cool and hot runs the absolute ms changed by 38% while
`%peak` stayed at 83.3%.

This is also why the file (a) times all six configurations back to back with no
host work in between, (b) validates in a second pass afterwards, and (c) takes
the minimum of four full sweeps. An earlier version interleaved validation with
timing; the multi-second host replay let the GPU drop to a low clock state, and
the first configuration measured looked 3x faster than the rest — a completely
fictitious result that nevertheless reproduced consistently.

---

## Expected output

Actual output, RTX 3500 Ada Laptop GPU, CUDA 13.2, on a cool GPU:

```
=== NVIDIA RTX 3500 Ada Generation Laptop GPU : 40 SMs, SM clock 1.545 GHz ===
blockDim = 1024 (32 warps), 16384 dependent FMAs per thread
Resident blocks per SM (occupancy API) : 1
One wave                               : 40 blocks
Peak FP32 at the API-reported clock    : 15820.8 GFLOP/s

Measured SM clock during the sweep     : 2.039 GHz (API reports 1.545 GHz)
Peak FP32 at the measured clock        : 20874.9 GFLOP/s

  blocks   waves  wave eff         ms     GFLOP/s    %peak    ms/wave   vs wave
-------- ------- --------- ---------- ----------- -------- ---------- ---------
      20       1     50.0%     0.0770      8720.7    41.8%     0.0770     1.00x
      39       1     97.5%     0.0771     16982.7    81.4%     0.0771     1.00x
      40       1    100.0%     0.0772     17395.1    83.3%     0.0772     1.00x
      41       2     51.2%     0.1534      8965.6    42.9%     0.0767     1.99x
      80       2    100.0%     0.1543     17395.1    83.3%     0.0772     2.00x
      81       3     67.5%     0.2275     11945.1    57.2%     0.0758     2.95x

PASS
```

**Run-to-run variation.** The `ms` column varies by up to ~2x depending on the
GPU's thermal state: a warm run gives 0.1058 / 0.2117 / 0.3134 ms instead of
0.0772 / 0.1534 / 0.2275 ms. The `wave eff`, `%peak` and `vs wave` columns are
stable to within ~1% across all runs observed, because they are ratios (and, for
`%peak`, normalized against the measured clock). Trust the ratios; ignore the
absolute milliseconds.

---

## The result that matters

Wall-clock time on a GPU is quantized in **waves**, not in blocks. 39, 40 and
even 20 blocks all cost the same 0.077 ms; 41 blocks costs 0.153 ms; 80 blocks
also costs 0.153 ms. The marginal block that crosses a wave boundary costs a full
wave of machine time, and every block after it up to the next boundary is free.
Once you have internalized that, grid sizing stops being "launch enough threads"
and becomes "launch a whole number of waves, or enough waves that one partial one
does not matter" — the tail is `~1/waves` of your runtime.

**Variation to try:** change `THREADS` from 1024 to 256 and re-run. The first
thing to check is that your TODO 1 adapts: `blocksPerSM` becomes 6 (1536/256), so
a wave is 240 blocks and the configurations become 120/239/240/241/480/481. If
`waveBlocks` stayed 40 you hard-coded something.

The second thing is more interesting: **the staircase softens.** Measured at
`THREADS = 256`, the ratios become 0.59x / 1.00x / 1.00x / 1.15x / 1.89x / 2.03x
instead of 1.00x / 1.00x / 1.00x / 1.99x / 2.00x / 2.95x. Work out why before
reading on.

Two things changed, and both follow from `blocksPerSM` no longer being 1.

- *Half a wave no longer idles any SM.* 120 blocks with 6 slots per SM spreads as
  3 blocks on every one of the 40 SMs — the machine is fully engaged, just less
  occupied. With half the work and an issue-bound kernel, it takes roughly half
  the time (0.59x). At `THREADS = 1024` the same configuration left 20 SMs
  completely idle, which is why it cost 1.00x.
- *Block duration stops being constant.* With 6 blocks per SM, the SM's 4
  schedulers are shared by 48 warps, so a block takes ~6x as long as it would
  running alone. The single tail block of the 241-block launch runs alone on its
  SM and therefore finishes in a fraction of a full wave's duration — hence 1.15x
  rather than 2x. The `blockDim = 1024` configuration the exercise ships with is
  special precisely because one block already owns the entire SM, which is the
  only way to make block duration independent of what else is resident.

(Caveat if you run this variation: the measured-clock recovery in the file assumes
block 0 spans the whole kernel, which is only guaranteed when `blocksPerSM` is 1
and the grid is one wave. At `THREADS = 256` the reported clock — and therefore
`%peak` — is garbage. The `ms` and `vs wave` columns remain valid.)
