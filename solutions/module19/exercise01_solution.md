# Module 19 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

and, to see where the first two columns of the table come from:

```
nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise01_solution.exe exercise01_solution.cu
```

Nothing in this exercise is timed, so it runs in under a second and its output
is bit-reproducible across runs.

---

## TODO 1 — `blocksByRegisters`

```cpp
static int blocksByRegisters(int regsPerThread, int threads)
{
    if (regsPerThread <= 0 || threads <= 0) return 0;
    const int warpsPerBlock = ceilDiv(threads, 32);
    const int regsPerWarp   = roundUp(regsPerThread, REG_GRAN) * 32;  // granule 8/thread
    const int warpsPerSlice = REGS_PER_SLICE / regsPerWarp;           // 4 slices of 16384
    return (REG_SLICES * warpsPerSlice) / warpsPerBlock;
}
```

Three quantisations, applied in this order, and the order matters.

**Why round to 8 first.** The register file is addressed by a field in the
instruction word, so an allocation is described by a base and a size and the
hardware cannot afford a general allocator. On sm_89 the unit is **8 registers
per thread, allocated for a whole warp at a time** — 256 registers per warp.
Module 18 established this. A thread asking for 41 registers is charged for 48.

**Why divide by 16384 and not 65536.** This is the half Module 18 did not need
and did not state. Module 1 described the SM as **four processing blocks**, each
with one warp scheduler, 12 warp slots and **a 16384-register slice** of the
65536-register file. A warp is assigned to one processing block and draws its
registers entirely from that slice. The leftovers in each slice cannot be
pooled across slices.

The two models differ exactly when `16384 / regsPerWarp` has a remainder large
enough that four of them would have made another warp. `state<30>` at 64
threads is the clean case:

```
47 registers -> roundUp(47,8) = 48 -> 48*32 = 1536 registers per warp
aggregate : 65536 / 1536                 = 42 warps -> 21 blocks of 2 warps
slices    : 4 * floor(16384 / 1536) = 4*10 = 40 warps -> 20 blocks
```

The hardware places **20**, and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
says 20. The eight registers stranded per slice (`16384 − 10·1536 = 1024`, i.e.
two thirds of a warp) are simply unusable.

While authoring this module both models were checked against the occupancy API
on **137 kernels** spanning block sizes 32–1024 and register counts 18–177.
The slice model was exact on all 137; the aggregate model was wrong on 10. The
three that survive into this exercise's table are `state<30>` at 64 threads
(20 vs 21), `state<72>` at 96 threads (6 vs 7), and the hypothetical `(98, 0,
64)` triple (8 vs 9).

**Common wrong approaches and their symptoms.**

| what you wrote | symptom |
|---|---|
| `65536 / (regs * threads)` — no granule at all | wrong on `(84, 0, 256)` (gives 3, answer 2) and `(41, 0, 256)` (gives 6, answer 5). Looks right on most rows because `ptxas` tends to land on register counts where the rounding does not cross a boundary — which is not a coincidence, since `ptxas` is itself occupancy-aware |
| `65536 / (roundUp(regs,8) * 32 * warpsPerBlock)` — Module 18's model | wrong on exactly the three slice rows. Right on 127 of 137 kernels, which is why it survives for years in people's heads |
| rounding the warp count instead of the per-thread count | `roundUp(regs*32, 256)` is the same thing and is fine; `roundUp(regs, 8)` applied after the multiply is not |
| forgetting `ceil` on `threads/32` | only visible at 100 threads, where you get 3 warps per block instead of 4 |

---

## TODO 2 — shared memory and warp slots

```cpp
static int blocksBySharedMemory(int smemBytes)
{
    return SMEM_PER_SM / roundUp(smemBytes + SMEM_RESERVE, SMEM_GRAN);
}
static int blocksByWarpSlots(int threads)
{
    return WARPS_PER_SM / ceilDiv(threads, 32);
}
```

The shared-memory line is Module 6's measured arithmetic, unchanged: a 1024 B
per-block driver reserve, then rounding up to a 128 B allocation granule, into
102400 B per SM. The table reproduces Module 6's measurements exactly —
16384 B → 5 blocks, 25600 B → 3, 49152 B → 2 — and the reserve plus granularity
are the whole reason 16384 B does not give 6.

Note that the request is static **plus** dynamic.
`cudaFuncGetAttributes().sharedSizeBytes` reports only the static half; the
third launch parameter is invisible to it, which is why the harness adds it
explicitly. Two rows of the table are the same kernel with 16384 B supplied
statically and dynamically, and they give the same answer.

**The warp-slot line is the trap.** Everyone writes `1536 / threads`, because
"1536 threads per SM" is the number in every table. The real limit is **48 warp
slots**, and the two expressions agree only when the block size is a multiple
of 32. `state<4>` at 100 threads:

```
1536 / 100          = 15 blocks   (1500 threads -- fits!)
48 / ceil(100/32)   = 12 blocks   (48 warp slots / 4 warps per block)
```

The API says 12. Module 8 explained why: a 100-thread block is four warps, and
the 28 lanes of the fourth warp that were never created still occupy their warp
slot, their register allocation and their share of the block's resources for
the block's whole lifetime. Fifteen such blocks would need 60 warp slots and
there are 48.

`state<4>` at 32 threads is the companion case in the other direction: the
registers permit 64 blocks and the warp slots permit 48, but the **block table
has 24 entries**, so you get 24 blocks of one warp — 24 of 48 warps, a hard
50 % occupancy ceiling. That is the real argument against 32-thread blocks, and
it has nothing to do with coalescing.

---

## TODO 3 — the minimum, and the limiter

```cpp
cand[0] = blocksByRegisters(regsPerThread, threads);
cand[1] = blocksBySharedMemory(smemBytes);
cand[2] = blocksByWarpSlots(threads);
cand[3] = MAX_BLOCKS_PER_SM;
int best = cand[0], which = LIM_REGS;
for (int i = 1; i < 4; ++i) if (cand[i] < best) { best = cand[i]; which = i; }
```

Strict `<` in the loop is what implements "report the first on a tie", which is
the rule the file states. Using `<=` would report block slots on half the rows
and the limiter hash would fail.

**The limiter is the actionable output, not the block count.** Knowing you get
3 blocks per SM tells you nothing you can act on; knowing that *registers* are
why tells you the only lever that exists. Four of the ten hypothetical rows are
register-limited, two shared-limited, two warp-slot-limited and one
block-slot-limited, and a reader who assumed "shared memory is what limits
occupancy" — which is the folk model, because shared memory is the resource you
request explicitly — gets four of them wrong.

Module 17 found this directly: two tiled-GEMM kernels with identical thread
counts and identical shared-memory footprints differed 3 vs 2 blocks per SM
purely because one compiled to 44 registers and the other to 40. Module 18
searched nine deliberately constructed fp32 GEMM configurations for one where
shared memory binds first, including a 32 KB tile, and did not find one.

---

## TODO 4 — occupancy

```cpp
int warps = blocks * ceilDiv(threads, 32);
if (warps > WARPS_PER_SM) warps = WARPS_PER_SM;
return 100.0 * (double)warps / (double)WARPS_PER_SM;
```

Occupancy is **warps over warps**. Two rows disagree with anything written in
threads:

- `(32, 0, 100)`: 12 blocks × 4 warps = 48 warps = **100 %**, while
  `12·100/1536` = 78 %. Occupancy is at its maximum and 300 of the 1536 thread
  slots are empty; those are the lanes of the partial warps, and they are not
  an occupancy problem, they are a lane-efficiency problem (Module 8).
- `(24, 0, 1024)`: 1 block × 32 warps = **66.7 %**, which is the ceiling Module
  3 quoted for 1024-thread blocks and deferred to here. `1·1024/1536` happens to
  give the same number, which is why this row alone does not catch the error.

The `min(…, 48)` clamp matters for `(16, 0, 32)`: 24 blocks × 1 warp = 24, no
clamp needed; but a hypothetical kernel with 48 one-warp blocks would compute
48 and must not exceed it.

---

## TODO 5 — `maxRegistersFor`, the design TODO

```cpp
for (int R = 255; R >= 1; --R)
    if (blocksByRegisters(R, threads) >= targetBlocks) return R;
return 0;
```

The point of this TODO is to notice that **you cannot invert the expression
algebraically**, because it contains two floors and a round-up, and that you do
not need to: `R -> blocksByRegisters(R, threads)` is a non-increasing staircase
on a domain of 255 points, so a linear scan from the top is exact, O(255), and
runs once at startup. A reader who tries to derive a closed form produces
something that is right except on the granule boundaries, which are precisely
the points this function exists to find.

The round-trip scoring is what forces exactness: the harness checks both
`blocksByRegisters(R) >= target` and `blocksByRegisters(R+1) < target`, so an
answer that is merely *sufficient* (say, 8 registers) fails the second test.

The measured answers are the only numbers that matter when you tune:

| target blocks | 256 threads | 128 threads |
|---|---|---|
| 2 | 128 | 128 |
| 3 | 80 | — |
| 4 | 64 | 128 |
| 6 | 40 | — |
| 8 | — | 64 |

Read the 256-thread column as a staircase: anything from 65 to 128 registers
gives you two blocks and **nothing between those two numbers costs you
anything**. If your kernel is at 70 registers and you are considering an
optimisation that would save 4, the answer is that it saves nothing; if it is at
66 and the optimisation saves 2, you have just bought a third block.

---

## Expected output

```
  blocks/SM correct on 18/18 kernels
 ...
 regs   smemB   thr |  reg  shr  wrp  blk |  you limiter         occ
   84       0   256 |    2  100    6   24 |    2 registers     33.3% ok
   41       0   256 |    5  100    6   24 |    5 registers     83.3% ok
   32       0   100 |   16  100   12   24 |   12 warp slots   100.0% ok
   24       0  1024 |    2  100    1   24 |    1 warp slots    66.7% ok
   16       0    32 |  128  100   48   24 |   24 block slots   50.0% ok
   40   12288   256 |    6    7    6   24 |    6 registers    100.0% ok
   40   16384   256 |    6    5    6   24 |    5 shared        83.3% ok
   64   49152   256 |    4    2    6   24 |    2 shared        33.3% ok
   98       0    64 |    8  100   24   24 |    8 registers     33.3% ok
   78       0   160 |    4  100    9   24 |    4 registers     41.7% ok

  10/10 block counts correct; limiter sequence ok; occupancy sequence ok

-- part 3: TODO 5, the inverted arithmetic -----------------------------
  blocks  threads |  max regs round trip
       2      256 |       128 ok    (blocks at R = 2, at R+1 = 1)
       3      256 |        80 ok    (blocks at R = 3, at R+1 = 2)
       4      256 |        64 ok    (blocks at R = 4, at R+1 = 3)
       6      256 |        40 ok    (blocks at R = 6, at R+1 = 5)
       4      128 |       128 ok    (blocks at R = 4, at R+1 = 3)
       8      128 |        64 ok    (blocks at R = 8, at R+1 = 7)
       2      512 |        64 ok    (blocks at R = 2, at R+1 = 1)
      12       64 |        80 ok    (blocks at R = 12, at R+1 = 10)

  SCORE: 10/10

OVERALL: PASS
```

The register counts in part 1 (`40, 48, 64, 80, 128, 34, 32, 36, 47, 92 …`) are
`ptxas` 13.2's choices and could in principle move with a different toolkit;
every scored answer in part 2 and part 3 is a function of literals and cannot.

---

## The result that matters

**The register term of the occupancy formula has two quantisations, not one,
and the second one is a fact about the physical shape of the SM rather than
about the compiler.** Module 6's formula omitted registers entirely; Module 18
added the 8-register granule and was right 127 times out of 137; the missing
piece is that the 65536-register file is four 16384-register slices and a warp
cannot straddle them. The practical form of all of this is not the formula but
its inverse: for a given block size there are only four or five register counts
at which anything changes, and knowing them turns "reduce register pressure"
from a vague aspiration into a target with a number on it.

**Variation to try.** Work out by hand, for a 1024-thread block, the register
count at which the kernel stops being launchable at all. The block needs 32
warps in one SM, and the slices supply `4·floor(16384 / (roundUp(R,8)·32))`:
at 64 registers that is 32 warps exactly — one block, placeable; at 72 it is 28
warps and the block cannot be placed, and the launch fails with
`cudaErrorLaunchOutOfResources`. Then do the same for 512 threads and check it
against the `0 warps` entries in Example 1's exchange-rate table. A 1024-thread
block is not just an occupancy ceiling of 66.7 %; it is a hard register budget
of 64 per thread.
