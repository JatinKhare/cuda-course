# Module 13 — Prefix Sum / Scan

> Prerequisites: Modules 1–12. In particular M5 (coalescing, the 2N traffic
> argument), M6 (shared memory, cooperative loading), M7 (bank conflicts and the
> `max(2,D)` cost law), M8 (warps, `__shfl_*_sync`), M9 (barriers, fences,
> publish/subscribe, **and why cross-block spinning deadlocks**), M10 (atomics,
> the atomic ticket, non-determinism), M12 (the reduction ladder and
> `__shfl_down_sync`).
>
> What this module gives you: the primitive that turns "I need to know where my
> output goes" into a parallel computation — and the first algorithm in this
> course where blocks cooperate *during* a kernel rather than between kernels.

---

## Concept

### 1. Two definitions, and why one of them is the useful one

Given an array `x[0..N)` and an associative binary operator `⊕` with identity
`e`, the **inclusive scan** is

```
inc[i] = x[0] ⊕ x[1] ⊕ … ⊕ x[i]
```

and the **exclusive scan** is

```
exc[i] = e ⊕ x[0] ⊕ … ⊕ x[i-1]       (so exc[0] = e)
```

For `⊕ = +` and `e = 0` these are the running sums with and without the current
element. `example01.cu` prints both for a 16-element array:

```
  input        1   2   3   4   5   1   2   3   4   5   1   2   3   4   5   1
  inclusive    1   3   6  10  15  16  18  21  25  30  31  33  36  40  45  46
  exclusive    0   1   3   6  10  15  16  18  21  25  30  31  33  36  40  45
```

**Associativity is the whole requirement.** Commutativity is *not* needed — scan
is defined for string concatenation, for matrix products, for `max`. What
associativity buys is re-bracketing: `(x0⊕x1)⊕(x2⊕x3)` equals `x0⊕(x1⊕(x2⊕x3))`,
and re-bracketing is exactly what lets a serial-looking recurrence become a
tree. Keep this in mind for floating point, where `+` is **not** associative and
a GPU scan therefore gives a different answer from the serial loop — but, unlike
M10's `atomicAdd`, it gives the *same* different answer every run, because the
tiling is fixed. Determinism and exactness are different properties.

**Exclusive is the more useful primitive**, and the reason is one sentence:
*the exclusive scan of "how many outputs each element produces" is the offset
where each element's output begins.* If element 3 emits 4 items, `exc[3] = 6`
tells the writer of element 3 to start at slot 6. The inclusive scan would tell
it where its run *ends*, which no writer needs. Every application below is an
instance of this: compaction, radix sort's digit histogram offsets, CSR row
pointers, run-length decoding, bucket boundaries.

### 2. Converting between them, and the two off-by-one hazards

| conversion | formula | requires | hazard |
|---|---|---|---|
| exclusive → inclusive | `inc[i] = exc[i] ⊕ x[i]` | nothing | none |
| inclusive → exclusive | `exc[i] = inc[i] ⊖ x[i]` | an **inverse** | wrong for `max`, `min`, `or` |
| inclusive → exclusive | `exc[i] = inc[i-1]`, `exc[0] = e` | nothing | **loses the total** |

The shift version is the general one, and it is the source of the single most
common scan bug: after shifting, the total `inc[N-1]` is no longer anywhere in
the array. `exc[N-1]` is the sum of everything *except the last element*. Any
code that recovers a count from an exclusive scan must add the last element's
own contribution back:

```
count = exc[N-1] + x[N-1]
```

This is Exercise 2's TODO 4, and it is data-dependent: if the last element
happens to fail the predicate, the wrong formula gives the right answer, and
your test passes.

### 3. Why scan matters

Scan is the primitive that *looks* inherently sequential — `out[i]` depends on
`out[i-1]` — and is not. It is the backbone of:

- **stream compaction** (§7 below, and Exercise 2),
- **radix sort**: each pass scans the per-digit counts to get the destination
  offsets; the sort is a scan with extra steps,
- **sparse matrix row offsets**: CSR's `rowPtr` is the exclusive scan of the
  per-row nonzero counts,
- **run-length encoding and decoding**, quadtree/BVH construction, allocation of
  variable-sized output per thread — anywhere a thread must ask "where do I
  write?" and the answer depends on everybody before it.

**M10 named this module explicitly.** M10's atomic ticket gives every surviving
element a unique slot, which is compaction — but the slot a given element gets
depends on which SM ran it and when. Run it twice, get two different
permutations (M10 measured `float atomicAdd` producing 10 distinct bit patterns
in 10 runs; the ticket is the integer analogue and Exercise 2 measures
12,582,783 of 12,582,896 positions differing between two consecutive runs of the
same kernel on the same data). Scan is the **only** route to *ordered*
compaction, because ordered compaction needs each element to know how many
survivors precede it, and "how many precede it" is a prefix sum by definition.
That debt is paid in §7 and in Exercise 2.

### 4. Scan is bandwidth-bound. Put the floor on the board first.

A scan must read every input and write every output. Nothing else is
*required*. On this GPU:

```
floor(N) = 2 · N · 4 bytes / 432 GB/s
```

For N = 67,108,861 that is **1.2428 ms**. Every number in this module is
reported against that floor, and the entire multi-block section is an argument
about how close you can get to it. The arithmetic (N-1 additions, minimum) is
free by comparison: 67 M adds at ~40 SM × 128 lanes × 1.8 GHz is under 10 µs.

This has a consequence that runs against every textbook treatment of scan: the
*work* complexity of the in-tile algorithm — O(N) versus O(N log N) — is a
second-order effect, because the adds happen in shared memory and registers
while the DRAM pipe is the constraint. Section 6 measures exactly how
second-order.

---

### 5. The ladder, step 1: Hillis–Steele

The **step-efficient** (or "naive") scan. For `off = 1, 2, 4, …, N/2`:

```
x[i] ← x[i] ⊕ x[i - off]      for all i ≥ off
```

After the step with offset `off`, `x[i]` holds the sum of the `2·off` elements
ending at `i`. After `log2 N` steps it holds the inclusive scan.

- **Depth** log2 N = 10 barriers for a 1024-element tile.
- **Work** N per step × log2 N steps = **10,240 additions** per 1024-element
  tile, against the 1023 a serial loop needs. O(N log N).

**The double-buffering requirement.** Written literally, the step is

```cpp
// WRONG
if (i >= off) sdata[i] += sdata[i - off];
__syncthreads();
```

This is a cross-thread **WAR hazard**, the same class M6 taught and M9 made
precise. Thread `i` reads `sdata[i-off]`; thread `i-off` writes `sdata[i-off]`.
Nothing orders those two, so thread `i` may read the *updated* value and add in
elements it has already accounted for. Adding a barrier after the statement does
not help — the race is *inside* the statement, between two threads, in the same
step. You need two barriers and a temporary, or one barrier and two buffers:

```cpp
// RIGHT: read buffer `cur`, write buffer `cur^1`
for (int i = tid; i < TILE; i += BLK)
    s[(cur^1)*TILE + i] = s[cur*TILE + i] + ((i >= off) ? s[cur*TILE + i - off] : 0u);
__syncthreads();
cur ^= 1;
```

Two things to note. First, this costs a second copy of the tile — 8 KB instead of
4 KB per block at TILE = 1024 — which is M9's "double-buffering trades shared
memory for one fewer barrier" in its simplest form. Second, the bug is
**invisible at 32 threads**: with a single warp the reads and writes of one
instruction are issued together and the answer comes out right. M6 warned about
exactly this ("works at warp width is not evidence of correctness"), and it is
why Exercise 1 validates at 1024 elements per tile and not 32.

The final inclusive→exclusive shift has the same hazard if the destination
buffer is the source: read into registers, barrier, write.

### 6. The ladder, step 2: Blelloch (work-efficient)

Blelloch's scan is two tree traversals over an implicit balanced binary tree laid
over the array in place.

**Upsweep (reduce).** This is literally Module 12's tree reduction, and saying so
is the point: *the first half of a scan is a reduction.* For
`d = N/2, N/4, …, 1`, thread `t < d` does

```
x[offset*(2t+2)-1] += x[offset*(2t+1)-1];      offset doubles each level
```

After the upsweep every internal node holds the sum of its subtree, and
`x[N-1]` holds the total.

**The identity insertion.** Before the downsweep, exactly one element changes:

```cpp
total = x[N-1];
x[N-1] = 0;               // the identity element, e
```

This single line is why Blelloch produces an *exclusive* scan. The downsweep
pushes the value at each node down to its children; seeding the root with the
identity means the leftmost leaf receives `e` and every other leaf receives the
sum of everything to its left. Seed it with the total instead and every output is
shifted by exactly `total` — which is the kind of bug that produces a clean,
plausible, uniformly-wrong array. Note also that the assignment destroys the
total, so you have to capture it first.

**Downsweep.** For `d = 1, 2, …, N/2`, halving `offset` each level:

```
t = x[ai];  x[ai] = x[bi];  x[bi] += t;        // swap-and-add
```

- **Depth** 2·log2 N = 20 barriers.
- **Work** N−1 adds up, N−1 adds and N−1 swaps down: **~2,048 additions** per
  1024-element tile. O(N). **5× fewer additions than Hillis–Steele.**

**The honest caveat, and this is the part textbooks skip.** Work-efficient does
not mean faster on a GPU, for three reasons:

1. **Parallelism collapses near the root.** The last upsweep level has one active
   thread out of 256; the first downsweep level likewise. Of the 20 levels, the
   top 3 in each phase have fewer than 32 active threads — one warp or less. The
   *work* is gone but the *depth* is still there, and a level with 1 active
   thread costs a barrier just like a level with 512.
2. **Twice as many barriers.** 20 versus 10. M9: a barrier costs the block's
   work skew, and here the skew is maximal at exactly the levels that do the
   least work.
3. **The strided access pattern is bank-conflicted** (§6a).

The measured answer on this GPU, from `example01.cu` with 1024-element tiles and
all four algorithms in the same kernel with identical global traffic:

| N = 1,048,573 (L2-resident) | ms | GB/s | % of 432 | vs warp scan |
|---|---|---|---|---|
| Hillis–Steele | 0.0328 | 256.1 | 59.3 % | 2.48× |
| Blelloch **+ padding** | 0.0227 | 368.8 | 85.4 % | 1.72× |
| Blelloch, no padding | 0.0299 | 280.5 | 64.9 % | 2.27× |
| warp-shuffle | 0.0132 | 635.9 | 147.2 %† | 1.00× |

† >100 % of DRAM peak means the 8 MB working set is L2-resident, per the house
rule. That is deliberate: to compare *algorithms* you have to get out of the
bandwidth-bound regime, or you are measuring the memory system.

| N = 67,108,861 (DRAM) | ms | GB/s | % of 432 |
|---|---|---|---|
| Hillis–Steele | 2.1009 | 255.5 | 59.2 % |
| Blelloch + padding | 1.7885 | 300.2 | 69.5 % |
| Blelloch, no padding | 1.8931 | 283.6 | 65.6 % |
| warp-shuffle | 1.8109 | 296.5 | 68.6 % |

**So: does Blelloch beat Hillis–Steele? Yes, but by 1.4–1.5×, not 5×** — and
only when it is padded. Unpadded it wins by 1.10× at 1 M. At 64 M, where all
four are within 20 % of the DRAM roof, the three fastest are within 2 % of each
other and the ranking between Blelloch and the warp scan is not even stable
run to run. **The work-efficiency argument is real and it is also nearly
irrelevant at scale.** Do not quote the 5× work ratio as a speedup.

#### 6a. Bank conflicts in the Blelloch tree, and what M7 says about them

The tree touches `s[offset*(2t+2)-1]` with `offset = 2^L` at level `L`. Two
adjacent threads' addresses differ by `2·offset = 2^(L+1)` words, so by M7's
`D = gcd(stride, 32)` rule the conflict degree is

```
D(L) = min(2^(L+1), 32)
```

| level L | 0 | 1 | 2 | 3 | 4 … 9 |
|---|---|---|---|---|---|
| stride (words) | 2 | 4 | 8 | 16 | 32 |
| conflict degree D | 2 | 4 | 8 | 16 | **32** |

The classic fix, straight out of the GPU Gems 3 chapter that made this algorithm
famous:

```cpp
#define CONFLICT_FREE_OFFSET(i) ((i) >> 5)
#define PIDX(i) ((i) + CONFLICT_FREE_OFFSET(i))
```

— insert one dead word every 32, so the effective pitch becomes 33 and
`gcd(33,32) = 1`. It is the same trick as M7's `[32][33]` tile padding and the
same idea as M5's odd row pitch, one level down.

**But M7 measured that the cost law on Ada is `max(2,D)`, not `D`** — a 32-lane
4-byte shared access occupies the pipe for two cycles regardless, so a 2-way
conflict is free. Applied here: **level 0 gets nothing from padding**, because
its D is exactly 2. Level 0 is also the widest level, with 512 active threads.
So the padding should buy less than the naive model predicts.

It does — and it still buys a lot, because levels 1 through 9 are genuinely
conflicted and levels 4–9 are 32-way. Measured with the same source instantiated
twice, once with `PIDX` and once without:

| measurement | ratio |
|---|---|
| Blelloch raw / Blelloch padded, N = 1 M (L2-resident) | **1.31×** |
| Blelloch raw / Blelloch padded, N = 64 M (DRAM-bound) | **1.06–1.26×** (run-dependent) |
| naive `D`-proportional model would predict | ≈ 5× |

The padding costs 32 extra shared words per block (128 B). At 8 KB per block plus
the 1024 B driver reserve that does not change blocks/SM here, because 256-thread
blocks are already capped at 6 blocks/SM by the 1536 threads/SM limit — so on
this configuration the padding is free and worth taking. **On a configuration
where it pushes you over a shared-memory cliff, re-measure: 1.31× is not enough
to pay for losing a resident block.** That is M7's Exercise 2 result (padding
beat the swizzle by 19 % despite 25 % lower occupancy) pointing the other way,
and the general rule is that both must be measured, not assumed.

### 7. The ladder, step 3: the warp-shuffle block scan

This is what a modern block scan actually looks like, and it is not a tree over
shared memory at all. Three levels:

1. **Each thread serially scans its own items.** With 1024 elements and 256
   threads, each thread scans 4 items in registers. Serial work in registers is
   free; this is the same "sequential within a thread, parallel across threads"
   move M12 used to give reduction its first level.
2. **`__shfl_up_sync` scan of the 256 per-thread totals, within each warp.**
   Five steps, no shared memory, no barrier:
   ```cpp
   for (int off = 1; off < 32; off <<= 1) {
       u32 n = __shfl_up_sync(0xffffffffu, v, off);
       if (lane >= off) v += n;
   }
   ```
   Note the guard. A lane shuffling from below lane 0 receives **its own value**,
   not zero — `__shfl_up_sync` clamps rather than zero-filling — so the `if` is
   load-bearing, not defensive. Note also that the WAR hazard of §5 simply does
   not exist here: there is no shared array to race on. The register file is
   private and the shuffle is a single instruction. This is the deepest reason
   the warp version wins.
3. **Scan the 8 per-warp totals in warp 0, then broadcast-add.** Two
   `__syncthreads()` in the entire block scan, against Hillis–Steele's 12 and
   Blelloch's 21.

Measured: 1.72× faster than padded Blelloch and 2.48× faster than Hillis–Steele
in the L2-resident regime. This is the block scan every later kernel in the
course uses, and it is what CUB's `BlockScan` does.

---

### 8. Going multi-block: four strategies and one traffic ledger

A block can scan a tile. The device has to scan 65,536 tiles and give each one
the sum of every tile before it. There are four ways, and the ranking is
determined *before you run anything* by counting DRAM traffic.

| strategy | kernels | traffic | why |
|---|---|---|---|
| **A** scan-then-propagate | 3 | **4N** | pass 1 reads N writes N; pass 3 reads N writes N |
| **B** reduce-then-scan | 3 | **3N** | pass 1 reads N (sums only); pass 3 reads N writes N |
| **C** decoupled look-back | 1 | **2N** | read N, write N. The floor. |
| **D** `cub::DeviceScan` | 1 | **2N** | C, done properly |

Pass 2 in A and B scans the ~65,536 tile totals — 256 KB, negligible.

**A, scan-then-propagate.** Scan each tile, record the tile total, scan the tile
totals, add tile offsets back. Simple, obviously correct, and pays for the
simplicity by writing and re-reading the whole array.

**B, reduce-then-scan.** Notice that pass 1 only needs the *totals*. Replace it
with a pure reduction (M12), which reads N and writes N/1024, then have pass 3
re-read the input and scan it with the offset already known. You have traded one
full write plus one full read for one extra read: 4N → 3N. The input is read
twice, which feels wasteful and is 25 % cheaper. This is CUB's fallback path.

**C, decoupled look-back.** The observation that makes single-pass scan possible:
a tile does not need its predecessors to be *finished*, it needs their *sums*.
A tile can compute its own aggregate immediately, publish it, and then go looking
for enough information to turn aggregates into a prefix.

Each tile owns three words of device state:

```
flag[t] ∈ { X = nothing published, A = aggregate known, P = inclusive prefix known }
agg[t]  = sum of tile t alone
pfx[t]  = sum of tiles 0..t         (only valid when flag[t] == P)
```

The protocol for tile `t`:

1. Scan the tile in shared memory, giving `total`.
2. Publish: `agg[t] = total`, **`__threadfence()`**, `flag[t] = A`.
   Tile 0 publishes `pfx[0] = total` and `flag[0] = P` instead — it has no
   predecessors, so its aggregate *is* its inclusive prefix.
3. **Look back.** Walk backwards from `t-1`. Accumulate `agg[j]` for each
   predecessor whose flag is `A`; the moment you find one whose flag is `P`, add
   its `pfx[j]` and stop — that one value summarises everything before it.
   A predecessor showing `X` has not published yet; spin on it.
4. Publish your own: `pfx[t] = exclusive + total`, `__threadfence()`,
   `flag[t] = P`. **This step is not optional for performance**: without it,
   every tile walks all the way to tile 0 and the scan is O(m²).
5. Add the exclusive prefix to the tile and write it out.

The look-back is warp-parallel, which is what makes it cheap: one warp inspects
32 predecessors at once, `__ballot_sync` finds the nearest lane reporting `P`,
and a warp reduction sums lanes 0 through that one. In the common case every
predecessor already has at least an aggregate and the whole look-back is one
32-wide probe.

Both fences matter and they are M9's publish/subscribe pattern at device scope.
`__threadfence()` is `MEMBAR.SC.GPU`: it pushes the payload out to L2, which M4
established is the device-wide coherence point, *before* the flag that advertises
it becomes visible. Without it a reader can observe `flag == A` and then read a
stale `agg`. On the reader's side the flag load must be an *acquire*: a plain
load may be hoisted out of the spin loop or served forever from a non-coherent
L1, so the flag is read with `atomicAdd(&flag[j], 0u)` (M9's idiom) and followed
by a `__threadfence()` before the payload is read.

#### 8a. Why this is not the deadlock Module 9 forbade

M9 was categorical: **spin-waiting on another block deadlocks**, because the
spinner occupies the SM slot the producer needs, and a block that has not been
placed does not exist. That argument is correct and it applies here. The reason
decoupled look-back terminates rests on two facts, and you must be able to state
both.

**Fact 1: a block only ever waits on lower-numbered tiles.** The dependency graph
is a total order with no cycles, so there is always at least one runnable tile.

Fact 1 alone is *not sufficient*. If the tile a block owns is `blockIdx.x`, then
a resident block 5000 could be spinning on block 4999 which the GigaThread engine
has not placed yet — and nothing in the programming model promises blocks are
launched in increasing `blockIdx` order. Current NVIDIA hardware happens to
dispatch roughly in order, which is why the naive version usually works, and
"usually works" is exactly the failure mode M8 and M9 spent two modules warning
about.

**Fact 2: the tile index is claimed dynamically.** Each block's first act is

```cpp
if (tid == 0) s_tile = atomicAdd(ticket, 1u);
```

Now the argument closes. A block holds ticket `t` only because every ticket
`0 … t-1` was already handed out, which means every lower-numbered tile is owned
by a block that has **already executed an instruction** — i.e. is already
resident on an SM. A resident block is never evicted (M1: blocks are indivisible
and non-migrating) and its warps are always eventually issued. Tile 0 never
waits. By induction on `t`, every tile eventually reaches step 4 and publishes
`P`, so every spin terminates.

This is exactly CUB's design, and it is why CUB's scan uses a "tile counter"
rather than `blockIdx.x`.

**The honest caveat, stated plainly: this is a forward-progress argument, and
the CUDA programming model does not give you forward-progress guarantees for
resident blocks in writing.** What it rests on is (i) blocks do not migrate or
get evicted, and (ii) the warp scheduler does not starve a resident warp
indefinitely. Both are true of every NVIDIA GPU shipped, both are relied on by
CUB and therefore by essentially all CUDA library code, and neither is a
sentence in the programming guide. **ARCHITECTURE-SPECIFIC / IMPLEMENTATION-
DEPENDENT.** If you write a decoupled look-back yourself, write the argument in
a comment, and stress it: Exercise 3's solution and this module's development
ran the kernel **7,400+ times across 18 grid sizes from 1 to 65,536 tiles**
without a hang or a wrong answer, and that is the minimum evidence that should
make you comfortable.

#### 8b. Measured (`example02.cu`, N = 67,108,861, 65,536 tiles)

```
strategy                 traffic       ms  GB/s vs 2N   GB/s real  % of 432  valid
A scan-then-propagate        4N   2.9395       182.6       365.3     84.6%     ok
B reduce-then-scan           3N   2.1486       249.9       374.8     86.8%     ok
C decoupled look-back        2N   1.6958       316.6       316.6     73.3%     ok
D cub::DeviceScan            2N   1.4389       373.1       373.1     86.4%     ok

  2N floor at 432 GB/s = 1.2428 ms
```

Read the two GB/s columns against each other, because that is the whole lesson.
"GB/s real" is the rate at which each strategy moves the bytes *it chose to
move*; A and B sit at 84.6 % and 86.8 % of DRAM peak, which is the best a
streaming kernel does on this part (M6/M11: ~375–381 GB/s observed maximum).
**A and B are not slow kernels. They are fast kernels doing too much I/O.**
"GB/s vs 2N" is the rate at which each one delivers *answers*, and it is the
column that tracks the traffic ledger: 182.6 / 249.9 / 316.6 / 373.1 against a
predicted 1 : 1.33 : 2 : 2 ratio from the traffic column alone.

The one number that does not follow the ledger is mine versus CUB: both move 2N,
CUB is 1.18× faster. The gap is the cost of the look-back and of my simpler
pipeline — CUB uses a larger tile, vectorized loads, and a look-back that starts
speculatively before the block scan finishes. Exercise 3 reproduces this: my
single-pass scan reaches 64–73 % of peak across runs, against CUB's steady 86 %.

### 9. What you would actually ship

```cpp
#include <cub/cub.cuh>

void  *d_temp = nullptr;  size_t tempBytes = 0;
cub::DeviceScan::ExclusiveSum(d_temp, tempBytes, d_in, d_out, n);   // query
cudaMalloc(&d_temp, tempBytes);
cub::DeviceScan::ExclusiveSum(d_temp, tempBytes, d_in, d_out, n);   // run
```

The two-call query/allocate/run protocol is the CUB convention. For
N = 67,108,861 the temp storage is **280,575 bytes — 4.3 bytes per tile**, which
is the decoupled-look-back state array and nothing else: CUB packs the flag into
the high bits of the value so that a tile's state is one 64-bit word that can be
read atomically, eliminating the reader-side fence entirely. `InclusiveSum`,
`ExclusiveScan` with an arbitrary functor, and `DeviceScan::*ByKey` (segmented)
are all there.

CUB ships with CUDA 13.2; no extra install. With MSVC it needs
`-std=c++17 -Xcompiler /Zc:preprocessor`. **Module 36 covers Thrust and CUB
properly**, including `DeviceSelect::If` — which is Exercise 2's ordered
compaction in one line, and is itself implemented on top of `DeviceScan`.

The engineering rule: write the scan yourself once, to understand the traffic
ledger and the look-back argument. Then call CUB.

### 10. Ordered stream compaction — paying M10's debt

Given a predicate, produce the surviving elements **in input order**:

```
flags[i] = pred(x[i])              // 1 or 0
offs     = exclusive_scan(flags)   // where each survivor goes
if (flags[i]) out[offs[i]] = x[i]  // scatter
count    = offs[N-1] + flags[N-1]  // the off-by-one from §2
```

Every survivor's destination is determined by the data alone, so the output is
byte-identical every run and in input order. Compare M10's atomic ticket, which
is one line and gives you a dense array in *arrival* order.

Measured (`exercise02`, N = 33,554,393, 37.5 % survive):

| version | ms | order-exact | run-to-run identical |
|---|---|---|---|
| atomic ticket | 0.688 | no | **no — 12,582,783 of 12,582,896 positions differ** |
| flags + 3-kernel scan + scatter | 3.269 | yes | yes |
| fused scan (predicate and scatter folded in) | 1.601 | yes | yes |

**Ordering costs 4.75×, or 2.33× if you fuse.** That is the real price, and it
is a traffic price, not an atomics price: the naive scan version touches the
array roughly eight times (flags write, flags read, scan pass 1 read+write, pass
3 read+write, scatter reads flags and offsets) where the atomic version touches
it 1.4 times. Fusing the predicate into the tile scan and the scatter into the
offset-add pass removes half of that. Doing the scan with decoupled look-back
and fusing the scatter into it would bring ordered compaction to within about
1.4× of the unordered version — which is roughly where `cub::DeviceSelect::If`
lands, and is the reason nobody writes the atomic version in production unless
they genuinely do not care about order.

---

## Hardware Mental Model

**Why a shared-memory scan is not a memory problem but the global scan is.**
A 1024-element tile is 4 KB. It lives in the SM's 128 KB unified L1+shared block
(M1), reachable with no tag compare and at a fraction of L2's measured 241 cycles (M4, M6). The Blelloch tree does
~2,048 shared accesses per tile; Hillis–Steele does ~20,480. At 8 LD/ST units per
SM processing block, 20,480 shared accesses across 4 processing blocks is on the
order of 640 issue slots per tile — real, but small next to the ~1,024 sectors of
DRAM traffic the same tile generates at 575-cycle latency. This is why the 5×
work ratio collapses to 1.4×.

**Why the conflict degree matters more than the access count.** M7: a conflict is
a *replay* — the same instruction re-issued once per distinct word a bank must
supply. Blelloch's 32-way levels turn one `LDS` into 32 issue slots. Nine of the
twenty levels are 32-way. That is the 1.31× the padding recovers, and it is the
same replay mechanism as M4's constant-memory serialization and M10's atomic
contention — the fourth instance in this course.

**Why the warp scan has no hazard at all.** `__shfl_up_sync` moves a value
between two lanes' *register files* through the SM's crossbar, inside one
instruction, with the warp's 32 lanes participating in lockstep by construction
(the `_sync` mask is what creates that guarantee under independent thread
scheduling, M8). There is no shared location, so there is no RAW, no WAR, and
nothing for a barrier to order. The five shuffle steps of a warp scan are five
instructions. The equivalent five levels of a shared-memory tree are five
instructions *plus five barriers plus their conflict replays*.

**Why the look-back needs the L2 and not just a barrier.** Two blocks on two SMs
communicate. M4: L1 is not coherent across SMs, and L2 is the first level they
share. `__threadfence()` (`MEMBAR.SC.GPU`) is the instruction that says "do not
let anything I issue after this become visible before everything I issued before
it has reached that level." The reader's `atomicAdd(&flag, 0)` executes *at the
L2* (M10) and therefore cannot be served by a stale L1 line. Both halves are
required; a fence without an atomic load, or an atomic load without a fence,
gives you a program that works on this hardware most of the time and is wrong.

**Why the look-back costs ~15 %.** In steady state the look-back is one 32-wide
probe of L2, ~241 cycles (M4). It is on the critical path of every tile, and
unlike the streaming loads it cannot be prefetched, because it does not exist
until the tile ahead publishes. 65,536 tiles across 240 resident blocks means
~273 sequential look-back rounds; at ~241 cycles and 1.8 GHz that is ~37 µs of
pure latency against a 1.24 ms floor — about 3 %, plus the fences, the atomic
traffic, and the loss of overlap. Measured total: 73.3 % of peak versus CUB's
86.4 %.

**Why blocks-per-SM is unchanged by any of this.** 256 threads/block caps you at
1536/256 = 6 blocks/SM. The largest shared request here is 8 KB + 1024 B reserve
= 9216 B, and 102400/9216 = 11. Shared memory is not the binding constraint, so
the Hillis–Steele double buffer and the Blelloch padding are both free in
occupancy terms — which is not something to assume, it is something to check with
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

---

## Code Walkthrough

### `example01.cu` — definitions, the ladder, the padding question

Part A is host-side and exists to pin the definitions down. The two checks it
prints are worth reading as assertions about the algebra:

```cpp
for (int i = 0; i < n; ++i) if (exc[i] + in[i] != inc[i]) okA = 0;
for (int i = 0; i < n; ++i) { u32 e = (i == 0) ? 0u : inc[i-1]; if (e != exc[i]) okB = 0; }
```

and the closing line about the total is the §2 hazard in one sentence:
`exclusive[n-1] = 45` where the total is 46.

Parts B and C are one kernel, templated on the algorithm, so that the *only*
difference between the measured columns is the shared-memory scan. Both the load
and the store are shared by all four configurations:

```cpp
template<int ALG>
__global__ void tileScanKernel(const u32 * __restrict__ in, u32 * __restrict__ out, int n)
{
    __shared__ u32 s[SMEM_WORDS];           // 2*TILE, enough for every variant
    ...
    if      (ALG == ALG_HS)     blockScanHS(s, tid);
    else if (ALG == ALG_BL_PAD) blockScanBlelloch<true >(s, tid);
    else if (ALG == ALG_BL_RAW) blockScanBlelloch<false>(s, tid);
    else                        blockScanWarp(s, tid);
```

The padded and unpadded Blelloch are the *same source*, instantiated twice
through `pidx<PAD>`; nothing else differs, which is what makes the 1.31× a clean
measurement of the padding and not of two different programs. The M7 discipline
is all here: configurations timed back-to-back in one loop with the order
rotated each sweep (`int a = (q + sweep) % N_ALG;`), min of 4 sweeps, a
duration-based 400 ms warm-up, iteration count auto-scaled to a ~10 ms segment,
and validation in a separate second pass.

Note what the kernel does *not* do: there is no pass 2 or pass 3, so the
measurement is pure 2N traffic for every column. That is deliberate. Adding the
other passes would bury the algorithm under 4N of DRAM.

### `example02.cu` — the traffic ledger

The four strategies share a block scan and differ only in the stitching. The
decoupled look-back kernel is the whole of §8 in 50 lines; the part to read
closely is the look-back loop:

```cpp
do { f = atomicAdd(&flags[j], 0u); } while (f == FLAG_X);   // acquire-ish load, spins
__threadfence();                                            // flag before payload
v = (f == FLAG_P) ? pfxs[j] : aggs[j];
...
u32 pmask  = __ballot_sync(0xffffffffu, f == FLAG_P);
int firstP = pmask ? (__ffs((int)pmask) - 1) : 32;
u32 c = (lane <= firstP) ? v : 0u;                          // sum lanes 0..firstP
```

Lane 0 looks at the immediate predecessor and lane 31 at the one 32 tiles back.
`firstP` is the nearest lane with a known inclusive prefix; lanes above it are
irrelevant because that prefix already summarises them. Lanes whose `j < 0` set
`f = FLAG_P, v = 0`, which makes "we ran off the front of the array" the same
case as "we found a prefix" with no extra code — a small thing, but it is the
difference between a loop with one exit and a loop with two.

The reporting is the point of the example:

```cpp
double g2 = 2.0*N4/(best[c]*1e-3)/1e9;          // answers per second
double gr = TRAFFIC[c]*N4/(best[c]*1e-3)/1e9;   // bytes actually moved per second
```

Two bandwidth numbers for the same kernel, and the whole argument of §8b is that
you need both to say anything true about a scan.

---

## Check Your Understanding

1. You implement Hillis–Steele in shared memory with `sdata[i] += sdata[i-off]`
   and a `__syncthreads()` after it. You test it with a 32-thread block and it
   is correct on every input you try. You then test it with a 256-thread block
   and it is wrong. Explain precisely which hazard this is, why the barrier you
   placed does not fix it, and why the 32-thread case passed — and say which
   `compute-sanitizer` tool would have found it at 32 threads.

2. Blelloch's downsweep is preceded by `x[N-1] = 0`. Suppose you forget that line
   entirely. Describe the *exact* output as a function of the correct output,
   and explain why a test that only checks `out[N-1] + x[N-1] == total` would
   still pass. Now suppose instead you set `x[N-1] = 0` but forget to save the
   total first. What breaks, and where does it show up in a multi-block scan
   specifically?

3. At N = 67,108,861 the three-kernel scan runs at 84.6 % of the 432 GB/s peak
   and the decoupled look-back runs at 73.3 %. The look-back version is
   nevertheless 1.73× faster. Explain how both statements are true at once, and
   then answer: if you could make the look-back's spin free, what is the maximum
   speedup you could still gain over CUB, and what would limit you?

4. The decoupled look-back assigns tile indices with `atomicAdd(ticket, 1)`
   rather than using `blockIdx.x`. Module 9 proved that a block spinning on
   another block deadlocks. Construct the specific interleaving under which the
   `blockIdx.x` version deadlocks, explain why the ticket version cannot reach
   that state, and identify the one assumption the ticket argument still makes
   that is *not* written down in the CUDA programming guide.

Answers: `solutions/module13/check_your_understanding.md`.

---

## Exercises

### Exercise 1 — `exercise01.cu` — implement the ladder

Build all three block scans and run each inside the same three-kernel
device-wide scan.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
exercise01.exe
```

- **TODO 1** — Hillis–Steele over a 1024-element tile, exclusive result,
  returning the tile total. You are given `2*TILE` words of shared scratch;
  decide whether you need the second half.
- **TODO 2** — the Blelloch upsweep on the padded layout.
- **TODO 3** — the identity insertion and the downsweep. Exactly one element
  changes between the phases.
- **TODO 4** — **design.** A block scan using no more than `BLK/32` words of
  shared memory beyond the tile and no `__syncthreads()` inside any warp-level
  step. You choose the decomposition of 1024 elements over 256 threads.
- **TODO 5** — two predictions, committed before running: which tile scan is
  slowest, and the Hillis–Steele/Blelloch **time** ratio bucket at
  N = 1,048,573. Count the additions each performs on a 1024-element tile first
  and write both numbers down.

Validation: exact equality against a CPU exclusive scan at **N = 1,048,573** and
**N = 67,108,861** — both non-powers of two, so the last tile is partial (1021
of 1024 elements) in both cases and a scan that ignores the tail fails. `OVERALL:
PASS` requires 6/6 correctness plus both predictions.

The harness reports the tile-scan kernel alone (identical 2N traffic for all
three, so the columns differ only by algorithm) and the full three-kernel scan
(4N), at both sizes. Watch what happens to the ratios between the two sizes.

### Exercise 2 — `exercise02.cu` — ordered stream compaction

Produce the list of surviving indices in input order, deterministically, and
compare with M10's atomic ticket.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
exercise02.exe
```

- **TODO 1** — materialise the predicate into a 0/1 array. The launch is a
  grid-stride grid of 240 blocks, not one thread per element.
- **TODO 2** — **design.** Assemble a device-wide exclusive scan from the three
  supplied kernels. You pick the strategy; before writing it, count how many
  times your plan reads N words and how many times it writes N, and record that
  count in a comment. m = 32,768, so do not assume the tile totals fit in one
  tile.
- **TODO 3** — the scatter. There are two plausible slot formulas and one of
  them produces a dense, permuted, entirely wrong array that looks fine.
- **TODO 4** — recover the total count on the host. The obvious answer is off by
  one on a data-dependent condition.
- **TODO 5** — predict the scan/atomic time ratio bucket.

Validation: your output must equal the CPU reference **element for element**
(not just as a multiset), the recovered count must be exact, the atomic version
must be checked to be a permutation and checked *not* to be ordered, and two
consecutive runs of the atomic version are compared against each other. 7/7 to
pass.

### Exercise 3 — `exercise03.cu` — decoupled look-back

Single-pass scan. The hardest thing in Part IV.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
exercise03.exe
```

- **TODO 1** — **design.** Obtain this block's tile index in a way that makes
  the look-back's spin provably terminate. Write the argument in a comment.
  `blockIdx.x` is the obvious answer and you need to decide whether it is sound.
- **TODO 2** — publish the tile aggregate. Payload before flag, at the right
  scope; say in a comment which scope and why the weaker one is not enough.
- **TODO 3** — the warp-parallel look-back loop, including the choice of load
  for the flag and the ordering between flag and payload.
- **TODO 4** — publish the inclusive prefix. Omitting this is not a correctness
  bug; work out what it costs.
- **TODO 5** — predict the speedup over the three-kernel scan from the traffic
  ledger, then decide whether you expect to hit it.

Validation: a correctness sweep over 9 grid shapes from 1 tile to 65,536 tiles,
200 back-to-back launches checking for wrong answers and hangs, exact validation
of both strategies at 64 M, and the prediction. 8/8 to pass.

**A half-finished decoupled look-back hangs the GPU rather than printing a wrong
answer.** With every TODO blank the program stops at the prediction gate without
launching. While developing, bound your spin and count the tiles that give up —
M9 Exercise 2's technique — so that a hang becomes a number instead of a
reboot.

---

## Prediction

Commit to these in writing before running anything.

1. **The work ratio between Hillis–Steele and Blelloch on a 1024-element tile is
   5:1.** Predict the measured *time* ratio, to the nearest factor of 2, in the
   regime where the tile scan is not DRAM-bound. Then predict it again for the
   DRAM-bound regime and say which direction it moves and why.

2. **The classic `CONFLICT_FREE_OFFSET` padding removes bank conflicts of degree
   2, 4, 8, 16 and 32 from the Blelloch tree.** Given M7's measured `max(2,D)`
   cost law on Ada, predict the speedup the padding buys, and name the one tree
   level for which it must buy exactly nothing.

3. **Decoupled look-back moves 2N bytes and the three-kernel scan moves 4N.**
   Predict the speedup. Then predict whether the single-pass version will reach
   a higher or lower *percentage of DRAM peak* than the three-kernel version,
   and explain why your two predictions are not in contradiction.
