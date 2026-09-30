# Module 18 — Check Your Understanding, answers

---

## 1. `BK` cancels out of the reuse law, so why does `BK = 32` make it slower?

The colleague's arithmetic is right. `FMAs per global load = BM·BN·BK /
(BM·BK + BK·BN) = BM·BN/(BM+BN)`, and `BK` genuinely does not appear. Nor does
the useful arithmetic change. Nor does the total global traffic. And the barrier
count really does drop by 4×, from `2K/8` to `2K/32`. Every one of those claims
survives the measurement.

What `BK` buys is amortisation of the barrier and the loop overhead. What it
costs is **two separate resources, both of which scale linearly with it.**

**Mechanism 1 — shared memory.** The tile is `BK·(BM+pad) + BK·BN` floats.
Quadrupling `BK` quadruples it: 8320 B → 33280 B for a 128×128 tile. On a part
where shared memory is what limits blocks per SM, that is a direct 4× cut in
resident blocks.

**Mechanism 2 — registers.** This is the one that actually bites here, and it is
less obvious. The `BK`-deep inner loop is fully unrolled (`#pragma unroll` on a
compile-time bound). `ptxas` software-pipelines the unrolled body, hoisting
`LDS` instructions well above the `FFMA`s that consume them so the shared-memory
latency is covered. The further it hoists, the more values are simultaneously
live. Measured on the 8×8 kernel:

| | registers | shared B | blocks/SM | limiter | GFLOP/s |
|---|---|---|---|---|---|
| `BK = 8` | 124 | 8320 | 2 | registers | 7114 |
| `BK = 32` | 196 | 33280 | 1 | **registers** | 5294 |

The shared-memory footprint quadrupled, but it is *not* what cost the block:
102400 B/SM would admit three blocks of 33280 B. 65536 registers admit exactly
one block of 256 threads at 196 registers. **On this GPU mechanism 2 dominates.**

There is also a third, smaller cost that is specific to this problem: `K = 769`
and `BK = 32` means the last k-tile carries 1 valid `k` and 31 zeros, so 3.1 %
of the fused operations are wasted. At `BK = 8` the waste is 0.9 %.

**On a GPU with twice the register file**, mechanism 2's threshold doubles: the
`BK = 32` kernel at 196 registers would fit two blocks instead of one, and the
binding resource would become shared memory (131072 registers would admit two
blocks; 102400 B of shared memory admits three, so registers would *still* bind
at 2 — you would need somewhat more than 2× before shared memory takes over).
The honest answer is: **mechanism 1 takes over as soon as the register file
stops binding**, which on a 2× register file happens at a slightly larger `BK`
or a slightly larger thread tile, and the correct procedure is not to guess but
to compute both limits, which is exactly what Example 2 §D and Exercise 2 TODO 1
make you do.

---

## 2. Isolating one of the two mechanisms behind `8 × 4` versus `4 × 8`

The lesson names three, actually: the register granule (80 vs 84 registers → 3
vs 2 blocks/SM), the block tile (`BM·BN/(BM+BN)` = 42.7 vs 25.6), and a
shared-memory bank effect. A good answer isolates *any* of them; here is the one
that isolates the bank effect, because it is the one the resource counters
cannot see.

**The configuration.** Hold the block tile fixed at `BM = 128, BN = 64, BK = 8`
with 256 threads, and vary only the thread tile:

- `TM = 8, TN = 4` → thread grid `(BM/TM) × (BN/TN) = 16 × 16`
- `TM = 4, TN = 8` → thread grid `32 × 8`

Both give `(BM/TM)·(BN/TN) = 256` threads, so both are legal. Now:

| held equal | value |
|---|---|
| accumulators per thread | 32 |
| registers (measured) | 80 |
| shared memory | 6272 B |
| blocks/SM | 3 |
| occupancy | 50 % |
| `TM·TN/(TM+TN)` | 2.67 |
| `BM·BN/(BM+BN)` | 42.67 |
| inner-loop SASS between barriers | 286 instructions, 256 `FFMA`, 24 `LDS.128`, 2 `BAR.SYNC` — **identical** |

Every quantity in Example 2's resource table is identical, and so is the
instruction stream. Both the register-granule mechanism and the block-tile
mechanism have been removed by construction.

**What you should expect to measure.** Not 1.00×. The number of threads along N
is `BN/TN`, which is 16 in one case and 8 in the other, and that is what sets
the stride of the B-tile read `Bs[kk][tCol·TN + j]`:

- `TN = 4`: lanes are 16 bytes apart. An `LDS.128` phase is 8 lanes = 128 bytes
  contiguous = all 32 banks once. **D = 1.**
- `TN = 8`: lanes are 32 bytes apart. A phase of 8 lanes covers 256 bytes with
  gaps, so each bank quad is hit twice with different words. **D = 2.**

And a 2-way conflict on a 16-byte access is *not* free, even though Module 7
showed it is free on a 4-byte access: the `max(2, D)` floor exists because a
32-lane 4-byte read only asks for 128 bytes and the bank array delivers 128
bytes per cycle, so it is under-subscribed and has a spare cycle. An 8-lane
16-byte *phase* asks for exactly 128 bytes — fully subscribed, no spare cycle,
so the conflict costs a real one.

Counting shared-pipeline cycles per value of `k`, as phases × D:

```
TM=8, TN=4 :  A 2 x LDS.128 x 4 phases x D=1  +  B 1 x 4 x 1  =  12
TM=4, TN=8 :  A 1 x LDS.128 x 4 phases x D=1  +  B 2 x 4 x 2  =  20
predicted ratio 20/12 = 1.67x
```

**Measured: 8186 vs 5095 GFLOP/s = 1.61×** at 1027×2053×769, and **7224 vs 5381
= 1.34×** at 1024×1024×8192. The second run is the control that rules out the
epilogue (at `K = 8192` the epilogue is one part in 1024 of the work), and the
residual between 1.34× and 1.67× is the part of the kernel that is not
shared-memory-limited.

A good answer would also note the design rule that falls out: **once the thread
tile is read with `float4`, `BN/TN` must be at least 16.** No counter reports
that constraint; only the bank arithmetic does.

*(An alternative, equally valid isolation: hold `TM = TN = 8` and compare
`BM = 128, BN = 128` at 256 threads against `BM = 64, BN = 128` at 128 threads.
That varies the block tile while holding the thread tile and therefore the
shared-read pattern fixed — the opposite isolation.)*

---

## 3. Testing the "L2 was already hiding it" explanation at 8192³

**What you should expect of the ratio.** Double buffering hides the latency of
the global→shared staging behind the fused operations of the previous tile. At
1027×2053×769 the whole 17.9 MB problem is L2-resident, so that latency is
~241 cycles (Module 4) rather than ~575, and there are 16–24 resident warps to
switch to while it lands; the prefetch therefore hides something that was
already hidden, and the register cost is pure loss (measured 0.81× at 8×8,
0.92× at 8×4). At 8192³ the operands are 768 MB and each block's tile genuinely
comes from DRAM on first touch, so the latency being hidden is larger and the
warp count available to hide it is unchanged. **The ratio should move toward 1
and may cross it.** It will not move dramatically, because the register cost is
unchanged and is what put the kernel on a worse occupancy step.

**Why the experiment as described is confounded.** Changing the problem size
changes *two* things at once:

1. the L2 hit rate of the staging loads — the variable of interest;
2. the grid size, hence the number of waves and the tail effect, hence the
   fraction of the kernel's life spent at full occupancy. At 1027×2053×769 the
   double-buffered kernel launches 9 × 33 = 297 blocks at 1 block/SM, i.e. 7.4
   waves with a 40 % tail; the single-buffered kernel launches the same 297
   blocks at 2 blocks/SM, i.e. 3.7 waves with a different tail. At 8192³ both
   have thousands of blocks and negligible tails. So part of any change in the
   ratio is the tail effect disappearing, not the prefetch working.

There is a third confound that is easy to miss: the two kernels differ in
**both** the pipelining *and* the occupancy (1 vs 2 blocks/SM). The measurement
cannot attribute the difference to either.

**The fix.** Hold occupancy constant and vary only the pipelining. Compile the
single-buffered kernel with `__launch_bounds__(256, 1)` so that `ptxas` is
allowed to use as many registers as the double-buffered version and both run at
1 block/SM, and run both at *both* sizes:

| | small (L2-resident) | large (DRAM) |
|---|---|---|
| single-buffered, forced to 1 block/SM | A | C |
| double-buffered (already 1 block/SM) | B | D |

`B/A` and `D/C` are then clean measurements of what software pipelining is worth
at two different memory-latency regimes, with occupancy, register pressure and
tail effects all held fixed within each column. The claim "the L2 was already
hiding it" predicts `B/A ≈ 1` and `D/C > 1`. Keeping the shared-memory footprint
equal as well (pad the single-buffered kernel's tile to the same 16640 B) closes
the last hole.

---

## 4. Constructing the case where Module 7 would be right

**Within the default 48 KB static shared-memory limit on sm_89, it does not
exist** — and the reason is more interesting than the answer.

The obvious construction is to push `BK` up until shared memory binds. Shared
memory per block is `BK·(BM+pad)·4 + BK·BN·4`, linear in `BK`; the register
count is set by `TM·TN` accumulators, which does not depend on `BK` at all. So
raising `BK` should move the binding resource from registers to shared memory.
It does not, and here is the search:

| BM | BN | BK | TM | TN | threads | regs | shared B | by registers | by shared | actual |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 128 | 32 | 8 | 8 | 256 | 196 | 33280 | **1** | 2 | 1 |
| 64 | 64 | 64 | 4 | 4 | 256 | 120 | 33792 | **2** | **2** | 2 |
| 64 | 64 | 64 | 2 | 2 | 512 | 64 | 33792 | **2** | **2** | 2 |
| 32 | 32 | 128 | 2 | 2 | 256 | 128 | 34816 | **2** | **2** | 2 |
| 32 | 32 | 64 | 2 | 2 | 256 | 68 | 17408 | **3** | 5 | 3 |
| 32 | 64 | 64 | 2 | 4 | 256 | 80 | 25600 | **3** | **3** | 3 |
| 32 | 16 | 128 | 1 | 1 | 512 | 64 | 26624 | **2** | 3 | 2 |
| 16 | 16 | 128 | 1 | 1 | 256 | 64 | 18432 | **4** | 5 | 4 |
| 16 | 32 | 128 | 2 | 2 | 128 | 187 | 26624 | **2** | 3 | 2 |

Registers bind or tie in **every** case. The assumption that broke is "the
register count does not depend on `BK`". It does, strongly, and for the reason
given in the answer to question 1: the `BK`-deep inner loop is fully unrolled,
and `ptxas` software-pipelines the unrolled body by hoisting `LDS` instructions
above the `FFMA`s that consume them. The deeper the unroll, the more values are
live simultaneously. `BM = 32, BN = 32, TM = TN = 2` uses 68 registers at
`BK = 64` and **128** at `BK = 128`, for the same four accumulators. The
register cost tracks `BK` almost as fast as the shared-memory cost does, and the
two limits move together.

Push further and you hit the wall: `BM = 64, BN = 32, BK = 128` does not compile
— `uses too much shared data (0xc800 bytes, 0xc000 max)`. 48 KB is the static
ceiling, and the register limit gets there first.

So the honest answer is a **negative result with a mechanism**: on sm_89, for an
fp32 register-tiled GEMM compiled with a fully unrolled `BK` loop and the
default 48 KB static shared-memory limit, **shared memory never strictly binds
blocks per SM.** Module 7's premise is unreachable within those constraints, and
the swizzle therefore never gets the chance its argument depends on.

To break the negative result you need to relax one of the constraints, and each
relaxation tells you where the technique actually lives:

- **Opt into >48 KB** with `cudaFuncAttributeMaxDynamicSharedMemorySize` (up to
  99 KB on Ada). Now `S` can reach 99 KB while `R` is still bounded by the
  accumulator count, and shared memory can be made to bind. But a single block
  holding 99 KB is 1 block/SM by construction, so you have traded away the only
  thing the swizzle was going to buy you.
- **Stop unrolling the `BK` loop** (`#pragma unroll 1`). Registers stop tracking
  `BK`, shared memory keeps growing, and shared memory binds — at the cost of
  serialising the `LDS`→`FFMA` chain, which is the entire kernel.
- **Change the accumulator-to-FLOP ratio.** This is the real one. On a Tensor
  Core, one register holds part of a 16×8×16 accumulator tile rather than one
  element of C, so the registers needed per unit of work collapse by an order of
  magnitude while the operand tiles do not shrink at all. `R` falls, `S` stays,
  and shared memory becomes the binding resource. **Modules 33–34.**
- **Multi-stage `cp.async` pipelines**, where 3 or 4 buffers of a large tile are
  live at once so `S` is multiplied by the stage count while `R` is not — and
  `cp.async` specifically removes the register staging that would otherwise make
  `R` grow too. **Module 32.**

CUTLASS lives in the last two regimes simultaneously, which is why CUTLASS
swizzles. **Module 43.**

The transferable lesson is §8's last sentence: a rule of the form "technique X
beats technique Y" is almost always a rule about *which resource is binding*,
and it inverts when that changes. Compute both limits before choosing which one
to spend — and check whether the compiler is spending the one you were not
watching.
