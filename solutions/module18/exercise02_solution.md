# Module 18 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

and, for TODO 5, the line that gives you the answer without running anything:

```
nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise02_solution.exe exercise02_solution.cu
```

The sweep uses **128 threads per block** throughout, so none of Example 2's
numbers carry over.

---

## TODO 1 — occupancy by hand

```cpp
static int occupancyBlocks(int regsPerThread, int smemBytes, int threads,
                           const char **limiter)
{
    const int REGS_PER_SM = 65536, THREADS_PER_SM = 1536;
    const int SMEM_PER_SM = 102400, MAX_BLOCKS = 24;
    const int SMEM_RESERVE = 1024, SMEM_GRAN = 128;
    const int REG_GRAN = 8;
    const int warps = (threads + 31) / 32;
    const int regsPerBlock = ((regsPerThread + REG_GRAN - 1)/REG_GRAN) * REG_GRAN
                           * 32 * warps;
    const int smemPerBlock = ((smemBytes + SMEM_RESERVE + SMEM_GRAN - 1)/SMEM_GRAN)
                           * SMEM_GRAN;
    const int byReg  = REGS_PER_SM / regsPerBlock;
    const int bySmem = SMEM_PER_SM / smemPerBlock;
    const int byThr  = THREADS_PER_SM / threads;
    /* min of the four, reporting the first that binds */
}
```

**The fact that was withheld: registers are allocated per warp, in granules of
8 per thread** (256 registers per warp) on sm_89. The naive model
`65536 / (regs · threads)` is right whenever `regs` happens to be a multiple of
8 and wrong otherwise, which is why a per-thread model scores 9/11 rather than
0/11 and looks almost right.

The row that separates the two models in this sweep is `TM = 8, TN = 8`:

```
regs = 142, threads = 128, warps = 4
naive :  65536 / (142 * 128)               = 65536 / 18176 = 3.60 -> 3     (right, by luck)
```

and `TM = 16, TN = 8`:

```
regs = 224, threads = 128
naive :  65536 / (224 * 128) = 2.28 -> 2   (right again)
```

and then `TM = 4, TN = 4`:

```
regs = 61, threads = 128, warps = 4
naive   : 65536 / (61 * 128)        = 8.39 -> 8    WRONG
granule : 61 -> 64; 64*32*4 = 8192; 65536/8192 = 8  ... also 8
```

— the two agree here, and the case that actually separates them in this file is
`TM = 2, TN = 2` at 40 registers (both give 12) versus Example 2's 256-thread
`TM = 4, TN = 8` at 84 registers, where the naive model says 3 and the hardware
says 2. The general rule: the models diverge exactly when rounding `regs` up to
the next multiple of 8 crosses a division boundary. Getting 11/11 here without
the granule is possible; getting it on an arbitrary kernel is not, and the
harness says so in the header.

**The other two granularities** come from Module 6: the per-block **1024 B
driver reserve** and the **128 B allocation granularity** for shared memory.
Module 7 Exercise 2 already made you discover both; this is the same arithmetic
with the register term added.

**`limiter` is the part that matters for the rest of the exercise.** In this
sweep it reads `registers` in all eleven rows, which is the observation
Exercise 2 exists to make: for an fp32 register-tiled GEMM on Ada, shared memory
never binds.

---

## TODO 2 — the two-level reuse law

```cpp
static double fmasPerGlobalLoad(int BM, int BN) { return (double)BM*BN/((double)BM+BN); }
static double fmasPerSharedRead(int TM, int TN) { return (double)TM*TN/((double)TM+TN); }
```

Per k-tile a block loads `BM·BK + BK·BN` elements and performs `BM·BN·BK` fused
operations, so `BK` cancels: `BM·BN·BK / (BK(BM+BN)) = BM·BN/(BM+BN)`. Per value
of `k` a thread reads `TM + TN` scalars and performs `TM·TN` fused operations.

**Same function, two levels apart.** That is the whole content of the TODO, and
the reason it is hashed rather than compared to a constant is that the harness
probes four different arguments (`(128,128)`, `(64,128)`, `(8,8)`, `(16,4)`) and
a reader who wrote `TM*TN/2` or `min(TM,TN)` matches on one of them.

Note `fmasPerSharedRead(16,4) = 3.2 < fmasPerSharedRead(8,8) = 4.0` with the
same 64 accumulators. That is the AM–GM argument from lesson §2: for a fixed
product, the ratio is maximised when the factors are equal.

---

## TODO 3 — the predictions

**P1 = 3, "neither — an interior point."** Measured sweep:

| config | regs | blk/SM | occ | FMAs/read | GFLOP/s |
|---|---|---|---|---|---|
| TM=2 TN=2, BM 16, BN 32 | 40 | 12 | **100.0 %** | 1.00 | 3292–3449 |
| TM=4 TN=4, BM 32, BN 64 | 61 | 8 | 66.7 % | 2.00 | 5625–6156 |
| **TM=8 TN=4, BM 64, BN 64** | 96 | 5 | 41.7 % | 2.67 | **6326–7170** |
| TM=4 TN=8, BM 32, BN 128 | 96 | 5 | 41.7 % | 2.67 | 4284–4741 |
| TM=8 TN=8, BM 64, BN 128 | 142 | 3 | 25.0 % | 4.00 | 4556–5107 |
| TM=16 TN=8, BM 128, BN 128 | 224 | 2 | 16.7 % | **5.33** | 4889–6333 |

The maximum-occupancy row is at 47 % of the winner. The maximum-reuse row is at
77 % of it. The winner is in the middle, at 41.7 % occupancy. The prediction is
phrased as a *kind* rather than an index deliberately: the top three rows are
within about 15 % of each other and their ordering moves between runs, but
"neither extreme" is reproducible.

Note also the `TM=8 TN=4` versus `TM=4 TN=8` pair: identical registers,
identical occupancy, identical shared memory, identical reuse, and **1.48×**
apart. That is the `LDS.128` bank effect of lesson §8 (`BN/TN` is 16 in one and
8 in the other), and it is the reason TODO 4's model needs the *global* reuse
term to tell them apart from the resource table alone.

**P2 = 5, "slower by more than 10×".** Measured **14.1–14.6×**.

---

## TODO 4 — the cost model

```cpp
for each candidate:
    if (spillBytes > 0) reject
    g = fmasPerGlobalLoad(BM, BN)
    r = fmasPerSharedRead(TM, TN)
    w = blocksPerSM * threads / 32          // resident warps
    score = (g/(g+8)) * (r/(r+1)) * (w/(w+8))
pick the maximum
```

Three saturating terms, one per scarce resource, multiplied.

**Why saturating.** Each of the three is a resource that helps monotonically and
stops mattering past a knee. More warps hide more latency — until there is no
latency left to hide. More reuse deletes more `LDS` — until `LDS` is no longer
the limit. `x/(x+c)` is the simplest function with that shape; the constant `c`
is the value at which the resource stops being the binding one. A linear model
scores the extremes too highly and picks the 100 %-occupancy row or the
216-register row, both of which are wrong.

**Why three terms and not two.** `TM=8/TN=4` and `TM=4/TN=8` are identical in
registers, spills, shared bytes, occupancy *and* `r`. The only quantity in the
table that separates them is the block tile: `BM·BN/(BM+BN)` is 32 for
`64 × 64` and 25.6 for `32 × 128`. Without the `g` term the model cannot tell
them apart and picks whichever comes first in the array — and they are 1.48×
apart in reality. **The two-level law of TODO 2 is not decoration; it is what
makes the model able to see the difference.**

**Why reject spills outright** rather than penalising them. Module 4: local
memory is DRAM. A spilled accumulator is a DRAM round trip inside the innermost
loop, executed `TM·TN·BK` times per k-tile. There is no coefficient that makes
that comparable to a register.

**The knees.** `c = 1` for the shared level (an `LDS` is cheap, so a small
amount of reuse already removes it as the limit), `c = 8` for the global level
(an L2 hit is 241 cycles — Module 4 — so you need much more reuse before it
stops mattering), `c = 8` warps. These are fitted, not derived, and the notes
say so. The gate is **within 15 % of the measured best**, not within 5 %,
because the top of this sweep is a plateau and the model's job is to land on the
plateau, not to rank inside it.

Measured: the model picks `TM = 8, TN = 4`, which is the measured winner in
every run of this file.

---

## TODO 5 — where ptxas starts spilling

**`PRED_SPILL_AT = 4`.**

With a pencil. `-Xptxas -v` on the unconstrained `TM=8 TN=8` kernel at 128
threads reports 142 registers. To fit `B` blocks of 128 threads (4 warps) on one
SM the compiler must satisfy

```
roundUp(regs, 8) * 32 * 4 * B  <=  65536      =>  roundUp(regs, 8) <= 512 / B
```

| B | registers allowed | 142 fits? |
|---|---|---|
| 1 | 512 (capped at 255) | yes |
| 2 | 256 | yes |
| 4 | 128 | **no** |
| 8 | 64 | no |
| 12 | 42 → 40 | no |

So `B = 4` is the first value that forces the register count below what the
kernel wants, and the compiler makes up the difference by spilling. Measured:

| launch bound | registers | spill bytes | blk/SM | occupancy | GFLOP/s |
|---|---|---|---|---|---|
| `(128, 1)` | 146 | 0 | 3 | 25.0 % | 5106–5282 |
| `(128, 2)` | 146 | 0 | 3 | 25.0 % | 5229–5384 |
| `(128, 4)` | 128 | **80** | 4 | 33.3 % | 5412–5915 |
| `(128, 8)` | 64 | 584 | 8 | 66.7 % | 617–672 |
| `(128, 12)` | 40 | 952 | 12 | **100.0 %** | 331–375 |

(`(128,1)` reports 146 rather than the unconstrained 142 because the launch
bound also fixes `maxThreadsPerBlock`, which changes ptxas's scheduling
slightly.)

**The row worth arguing about is `(128, 4)`.** It spills 80 bytes and is
*faster* than the unconstrained build — 5915 against 5282. A small spill that
buys a whole extra resident block can be a win. What makes the 8- and 12-block
rows catastrophic is not the existence of a spill but its *contents*: at 64 and
40 registers the 64 accumulators cannot possibly be resident, so the spilled
values are the ones read and written 64 times per value of `k`. **The cliff is
at the point where the spilled values are the inner-loop values**, not at the
first byte of spill. That nuance is why the question asks for the first spilling
bound rather than "where does it get slow".

---

## Performance reasoning

The headline is the last two rows of the launch-bound table: **100 % occupancy
is 14.1–14.6× slower than 25 % occupancy, on the same source, with the same
algorithm and the same instruction schedule up to register allocation.**

The mechanism is fully visible and involves nothing subtle. At `(128, 12)` the
compiler has 40 registers to work with and 64 accumulators to place. 24 of them
go to local memory, which Module 4 established is DRAM with an L1 line in front
of it. The inner loop is then

```
LDS.128 x 3
FFMA    x 32, of which ~24 are preceded by an LDL and followed by an STL
```

and the FP32 pipes are idle waiting on the L1/local path. Occupancy is at its
maximum and the machine is doing almost nothing.

This is the empirical heart of the course's occupancy argument. **Module 19**
owns occupancy properly — how to compute it, what the API reports, why the
"achieved" and "theoretical" figures differ. **Module 20** owns the
latency-hiding side: when *does* another warp help, and what is ILP worth
instead. The number to carry into both is 14×, and the sentence to carry is: an
occupancy target is a proxy for a latency-hiding requirement, and when a kernel
has no latency left to hide it is a proxy for nothing.

The second-order finding, for Module 19: **occupancy is not even monotone in the
right direction here.** Going from 25 % to 33.3 % occupancy is a win (5282 →
5915); going from 33.3 % to 66.7 % is a 9× loss. A tuner that hill-climbs on
occupancy walks straight off the cliff.

---

## Expected output

```
-- resources (compiler + occupancy API), before any timing --------
 config                           regs spillB  smemB blk/SM    yours    limiter     occ%
 TM=2  TN=2   BM16  BN32            40      0   1664     12       12  registers   100.0%
 TM=4  TN=4   BM32  BN64            61      0   3200      8        8  registers    66.7%
 TM=8  TN=4   BM64  BN64            96      0   4224      5        5  registers    41.7%
 TM=4  TN=8   BM32  BN128           96      0   5248      5        5  registers    41.7%
 TM=8  TN=8   BM64  BN128          142      0   6272      3        3  registers    25.0%
 TM=16 TN=8   BM128 BN128          224      0   8320      2        2  registers    16.7%
 8x8, __launch_bounds__(128,1)     146      0   6272      3        3  registers    25.0%
 8x8, __launch_bounds__(128,2)     146      0   6272      3        3  registers    25.0%
 8x8, __launch_bounds__(128,4)     128     80   6272      4        4  registers    33.3%
 8x8, __launch_bounds__(128,8)      64    584   6272      8        8  registers    66.7%
 8x8, __launch_bounds__(128,12)     40    952   6272     12       12  registers   100.0%
  occupancy model: 11/11 configurations match the CUDA API

  two-level reuse law: correct
    block tile 128x128 ->   64.000 FMAs per global load
    thread tile 8x8    ->    4.000 FMAs per scalar shared read

  your cost model picks: TM=8  TN=4   BM64  BN64
...
-- scoring --------------------------------------------------------
  P1 shape of the winner  : you said 3, measured 3   correct
  P2 100%-occupancy build : you said 5, measured 5  (14.26x slower)   correct
  cost model              : picked TM=8  TN=4   BM64  BN64 at 7170 GFLOP/s, best is 7170  within 15%
  first spilling bound    : you said 4, measured 4   correct

-- correctness (second, untimed pass) ------------------------------
  11/11 kernels produce a finite, correct C

  score 10/10

OVERALL: PASS
```

Absolute GFLOP/s move by 10–15 % with thermal state; the ratios reproduce to
about 1 %.

---

## The result that matters

**The fastest configuration in this sweep runs at 41.7 % occupancy, and the
100 %-occupancy build of the same kernel is 14× slower.** Occupancy is not a
goal; it is one of three things you are buying with a fixed register file, and
in a register-tiled GEMM it is the one worth the least. The cost model in
TODO 4 makes that concrete: it needs all three of block-level reuse,
thread-level reuse and resident warps, each saturating, and a hard veto on
spills — and with those four pieces it picks the winner from data the compiler
hands you before you run anything.

**Variation to try.** Add `TM = 8, TN = 2` (`BM = 64, BN = 32`, still 128
threads) to the table. Its reuse is 1.6 and its occupancy is high; predict where
it lands with your model before you measure, and then check whether `BN/TN = 16`
— the floor lesson §8 derives — is satisfied. Then try `TM = 2, TN = 8`
(`BM = 16, BN = 128`) and see what happens when it is violated in the other
direction.
