# Module 14 — Histogram

> Prerequisites: Module 5 (sectors, coalescing, `uchar4`/`float4`), Module 6
> (shared memory, dynamic allocation, capacity → occupancy), Module 7 (banks),
> Module 9 (barriers: the two guarantees), Module 10 (atomics, contention
> economics, privatization mechanism), Module 11 (the memory floor, vectorized
> loads, coarsening), Module 12 (reduction — used, not re-taught)
> What this module gives you: the ability to look at a histogram workload —
> a bin count and an input distribution — and choose, before writing anything,
> which of five strategies will win, by roughly how much, and where each one
> stops paying.

Module 10 shipped the *mechanism* of privatization and its traffic arithmetic
and then stopped, explicitly: "Module 14 builds the full privatized histogram —
bin replication to spread residual contention, handling more bins than fit in
shared memory, and the coarsening ladder. This module owns the mechanism and the
arithmetic; Module 14 owns the algorithm." This is that module.

It is also the first module in the course where **the input data is a
first-class parameter of the performance model**. Every number below states its
distribution. A histogram benchmark that does not is worthless, and the tables
in this module are built to show you why.

---

## Concept

### 1. The problem, and its floor

A histogram maps every element of an input array to one of `nBins` buckets and
counts how many land in each:

```cpp
for (size_t i = 0; i < n; ++i) hist[bin(in[i])]++;
```

Module 11's discipline applies before anything else: **compute the floor**.
The compulsory traffic is `n` elements read once, plus `nBins` counters written
once. For a 256-bin histogram of 192 MiB of bytes:

| | bytes |
|---|---|
| input, read once | 201,326,592 |
| output, written once | 1,024 |
| total compulsory | ≈ 201 MB, i.e. **1N** |

At the course's measured streaming ceiling of **410.5–410.7 GB/s** (Module 12,
after a 1500 ms warm-up) the floor is **0.49 ms**. Everything in this module is
an argument about the distance between a real kernel and that number.

**A histogram is a bandwidth-bound kernel that a naive implementation turns into
an atomic-throughput-bound kernel.** `example01.cu` measures the naive version
at **50–103 ms** depending on the input distribution: between 100× and 200×
above the floor, and reaching **0.4–1.3% of the streaming ceiling**. The entire
module is about closing that gap, and the surprise is how few of the moves
actually matter.

### 2. Why the naive version is that bad

Module 10 established the economics: an *uncontended* global atomic costs about
what a plain store costs, and all of the cost of a contended one is contention.
It measured atomic throughput against `K`, the number of distinct bins:

| K (M10) | 1 | 32 | 256 | 4096 | 2^20 |
|---|---|---|---|---|---|
| Gatomic/s | 1.93 | 12.08 | 23.35 | 115.71 | 107.79 |

A 256-bin histogram is the `K = 256` column, so you would expect ~23
Gatomic/s. `example01.cu` Part A measures **4.0 Gatomic/s** on uniformly
distributed bytes — six times worse. That discrepancy is real and it is the
first genuinely new result of this module.

**PORTABLE CUDA CONCEPT.** Module 10's benchmark used `atomicAdd(&c[i & mask])`.
With `mask = 255`, lane `l` of a warp touches word `base + l`: 32 distinct words
occupying **four** contiguous 32 B sectors. A real histogram's bin index comes
from the *data*, so a warp's 32 lanes land on 32 words scattered across the
whole bin array — up to **32 distinct sectors**. Part D of `example01.cu`
isolates exactly this, holding the atomic count and the number of distinct
addresses per warp fixed at 32 and changing only the sector footprint:

```
  32 words in  4 sectors, lane order      8.0507 ms   25.01 Gatomic/s
  32 words in  4 sectors, scrambled       8.0471 ms   25.02 Gatomic/s
  32 words in 32 sectors, lane order     64.2929 ms    3.13 Gatomic/s
  32 words in 32 sectors, scrambled      64.2944 ms    3.13 Gatomic/s

  sector ratio (32 sectors / 4 sectors): 7.99x and 7.99x
  scramble ratio (same sector count):    1.00x and 1.00x
```

**7.99×, and the permutation of lanes within the warp is worth exactly 1.00×.**

So Module 10's rule — *cost tracks addresses per warp, not atomic count* —
needs one refinement at histogram scale:

> **Global atomic cost tracks the number of distinct 32 B sectors a warp's
> atomics touch.** Distinct addresses matter only because distinct addresses
> usually mean distinct sectors. `K` distinct bins that happen to be packed into
> `K/8` sectors behave like `K/8` requests, not `K`.

This is the same sector-counting method Module 5 taught for loads and stores,
applied to atomics. It is also the reason Module 10's table is an *optimistic*
bound for every value of `K` above 8: the benchmark that produced it used the
most sector-dense address pattern that exists.

### 3. The distribution is the dominant variable

Four inputs, identical in every other respect, into the same naive kernel
(`example01.cu`, 192 MiB of bytes, 256 bins):

| distribution | what it is | naive ms |
|---|---|---|
| uniform | every bin equally likely | 50.40 |
| zipf | ~40% of the mass in bin 0 | 76.76 |
| same-bin | every element in bin 37 | 103.89 |
| clustered | runs of 8192 equal values; a warp's 32 lanes always share a bin | 61.04 |

A 2.06× spread on identical code and identical instruction counts. Two distinct
mechanisms are at work and they are worth understanding separately:

- **same-bin** is Module 10's `K = 1`: one L2 slice ALU serializes everything.
- **clustered** is Module 10's *warp-uniform address* case. All 32 lanes of a
  warp present the same address, so the warp contributes one serialized
  operation instead of 32 parallel ones. Module 10 measured 6.25× for this at
  `K = 32`. Note that the compiler's automatic warp aggregation
  (`VOTEU.ANY`/`POPC`/`RED`, Module 10) does **not** fire here: the address
  comes from memory, so it cannot be proved warp-uniform.
- **zipf** is a blend: a large fraction of the atomics behave like `K = 1` and
  the rest like `K ≈ 255`.

**The rule to carry away:** "how many bins" is not enough information to predict
a histogram's cost. You need to know how many bins are *hot*, and whether the
hot ones are hot *within a warp* or merely hot in aggregate. Those are different
diseases with different cures.

### 4. Privatization, done properly

Module 10 gave the five steps. Here they are in a kernel that works for any
`nBins` and any grid:

```cpp
__global__ void hist_shared(const unsigned char* __restrict__ in, size_t n,
                            unsigned int* __restrict__ hist, int nBins)
{
    extern __shared__ unsigned int s[];
    for (int b = threadIdx.x; b < nBins; b += blockDim.x) s[b] = 0u;   // 1
    __syncthreads();                                                   // 2

    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&s[in[i]], 1u);                                      // 3
    __syncthreads();                                                   // 4

    for (int b = threadIdx.x; b < nBins; b += blockDim.x)
        if (s[b]) atomicAdd(&hist[b], s[b]);                           // 5
}
```

Five things in that kernel are load-bearing and at least three of them are
routinely got wrong.

**The zeroing loop is strided, not `if (threadIdx.x < nBins)`.** With
`blockDim.x = 128` and `nBins = 256`, the guarded form initializes half the
bins and leaves the other half holding whatever the previous block left in that
shared memory. Exercise 1 uses exactly that block/bin pair, and Exercise 3 makes
it one of three planted defects. The failure is silent and produces counts that
are *too high*, which is a diagnostic in itself: a lost-update race makes counts
too low (Module 10), an uninitialized-bin bug makes them too high.

**The two barriers are different guarantees.** Module 9: `__syncthreads()`
provides an execution barrier *and* a block-scope memory fence. Barrier 2 is
needed so that a thread's zero is visible to the other threads before their
first add; barrier 4 is needed so that every add is complete before any thread
reads a bin. Removing either one is a genuine data race — and, as Exercise 3
demonstrates with real tool output, **`compute-sanitizer --tool racecheck` does
not report the missing barrier 2 when the conflicting access is an atomic.**
Replace the `atomicAdd` with a plain `+=` in the same kernel and racecheck
immediately reports two hazards with hundreds of thousands of instances.
Module 10 found racecheck blind to global RMW; this is a second blind spot, and
it sits precisely where a correct privatized histogram lives.

**`if (s[b])` in the flush is free and sometimes enormous.** It costs one
predicated compare and removes a global atomic. On the `same-bin` input, 255 of
256 bins are empty in every block, and this single test is why the naive-grid
privatized kernel is 25× faster on `same-bin` than on `uniform`
(`example01.cu`: 1.97 ms vs 8.07 ms for the *same kernel*).

**The traffic arithmetic decides the grid, and it is the whole ballgame.**
Module 10's ledger: `n` global atomics become `n` shared atomics plus
`grid × nBins` global ones. `example01.cu` prints both:

| grid | elems/thread | flush atomics | flush / N | uniform ms |
|---|---|---|---|---|
| 786,432 | 1.0 | 201,326,592 | 1.000 | 8.29 |
| 196,608 | 4.0 | 50,331,648 | 0.250 | 2.07 |
| 49,152 | 16.0 | 12,582,912 | 0.062 | 1.24 |
| 12,288 | 64.0 | 3,145,728 | 0.016 | **1.15** |
| 3,072 | 256.0 | 786,432 | 0.004 | 1.18 |
| 768 | 1,024.0 | 196,608 | 0.001 | 1.41 |
| 192 | 4,096.0 | 49,152 | 0.000 | 1.55 |

The textbook privatized histogram — one element per thread, `grid = ceil(n/BLK)`
— trades `N` contended global atomics for `N` shared atomics **plus `N` global
atomics**. It removes the contention and keeps all of the traffic, and it
measures **8.29 ms against the coarsened version's 1.10 ms**. Privatization
without coarsening is not the optimization; it is half of it.

The curve also turns around: at 192 blocks the machine (40 SMs × 6 resident
blocks = 240) is no longer full. The minimum sits at a grid of a few thousand
blocks, and *any* grid from 12,288 down to 3,072 is within 12% of it. This is a
broad optimum — pick a machine-sized grid and stop.

### 5. Bin replication, and an honest null result

Replication is the technique Module 10 named as this module's: keep `R`
independent copies of the histogram in shared memory, have different threads hit
different copies, and sum the copies at flush time.

```
shared footprint = R * nBins * 4 B
flush cost       = grid * nBins global atomics (unchanged), plus
                   grid * nBins * R shared reads
```

Two design decisions hide in "have different threads hit different copies":

1. **Who shares a copy.** The natural unit is the **warp**: `r = warpId % R`.
   Splitting a *warp* across copies is strictly worse on Ada, because the
   shared-memory unit already merges a warp's same-address increments into one
   `ATOMS.POPC.INC.32` (Module 10). Verified: lane-group replication
   (`r = threadIdx.x & (R-1)`) measures **0.91–0.94× of R=1** on the clustered
   and same-bin inputs — a regression.
2. **The layout.** Replica-major, `s[r * nBins + b]`, keeps a warp's address
   pattern identical to the unreplicated kernel, so Module 7's bank analysis is
   unchanged. Replica-minor, `s[b * R + r]`, gives a fixed `r` a stride of `R`
   across `b`, which is a `gcd(R, 32)`-way bank conflict — 8-way at `R = 8`.

And then the measurement, on all four distributions
(`example01.cu` Part B, 256 bins, 256-thread blocks):

| R | smem/blk | blocks/SM | uniform | zipf | same-bin | clustered |
|---|---|---|---|---|---|---|
| 1 | 1,024 | 6 | 1.00× | 1.00× | 1.00× | 1.00× |
| 2 | 2,048 | 6 | 1.00× | 0.99× | 0.99× | 0.99× |
| 4 | 4,096 | 6 | 0.99× | 0.99× | 0.99× | 1.00× |
| 8 | 8,192 | 6 | 0.99× | 0.99× | 0.99× | 0.98× |
| 16 | 16,384 | **5** | 0.84× | 0.83× | 0.84× | 0.83× |

**Replication buys nothing on a 256-bin counting histogram on this hardware, and
at `R = 16` it costs 17% by dropping a resident block per SM.** That is a
documented negative result, and the reason is worth more than a win would have
been:

- Ada's `ATOMS.POPC.INC.32` already removes *intra-warp* same-address
  contention, which is the expensive kind.
- What is left is *inter-warp* contention on one shared address, which the
  shared-memory unit resolves at roughly one operation per cycle. A block of 8
  warps therefore costs at most 8 shared-unit cycles per round while consuming
  256 input bytes — and at 410 GB/s across 40 SMs the machine can only deliver
  about 5 bytes per SM per cycle. **The memory system is 50× slower than the
  contention it is feeding.** There is nothing to fix.
- The occupancy arithmetic is Module 6's table: 6 blocks/SM at ≤ 12,288 B,
  5 at 16,384 B. Module 19 will formalize it. Crossing that boundary is a real,
  measurable 17%.

**When does replication pay, then?** When `ATOMS.POPC.INC` does not apply.
`ATOMS.POPC.INC.32` is an increment-by-one instruction. A **weighted** histogram
— `atomicAdd(&s[b], w[i])` with a data-dependent weight — cannot use it, and the
lanes of a warp targeting one bin must serialize. Measured on the same harness:
the weighted kernel runs at 2.79–2.96 ms on `clustered` and `same-bin` against
1.10 ms on `uniform`, i.e. intra-warp contention has come back and costs 2.6×.
(Replication still does not fix it, because the serialization is *within* the
warp and replication only separates *different* warps. The fix for that one is
Module 10's `__match_any_sync` aggregation, and Exercise 2 uses it.)

**ARCHITECTURE-SPECIFIC.** Everything in the paragraph above is an Ada/Ampere/
Turing statement. The portable claim is the weaker one: replication reduces
contention between the threads that are assigned different copies, at a cost of
`R×` shared memory and an `R`-term fold at flush. Whether that is worth anything
depends on whether contention was the binding constraint, and on this GPU, for a
counting histogram, it is not.

### 6. What *is* the binding constraint, then

If atomics are free and the traffic model is satisfied, the privatized histogram
should be at the ceiling. It is not — `example01.cu` puts v2 and v3 at **46% of
the streaming ceiling**. One more measurement settles it:

```
  uchar4 stream, no atomics        0.6291 ms   320.0 GB/s
  uchar  stream, no atomics        1.3249 ms   152.0 GB/s
  uchar  load + 1 ATOMS/elem       1.3474 ms   149.4 GB/s
  uchar4 load + 4 ATOMS/elem4      0.6313 ms   318.9 GB/s
  uchar4 load + 1 ATOMS/elem4      0.6334 ms   317.9 GB/s
```

(One sweep, one thermal state; the ratios are what matter.)

A stream of scalar `unsigned char` loads runs at **less than half** the rate of
the same bytes read as `uchar4`, and **adding the shared atomics on top costs
1.7%**. The privatized histogram is not atomic-bound, not contention-bound and
not occupancy-bound. It is bound by the *width of its loads*: `LDG.E.U8` moves
32 bytes per warp instruction where `LDG.E` on a `uchar4` moves 128.

So the last rung of the ladder is Module 5's and Module 11's, not Module 10's:

```cpp
uchar4 v = in4[i];
atomicAdd(&my[v.x], 1u);  atomicAdd(&my[v.y], 1u);
atomicAdd(&my[v.z], 1u);  atomicAdd(&my[v.w], 1u);
```

Identical atomic count, one quarter of the load instructions, and it takes the
kernel from 46% to **99–100% of the measured streaming ceiling on every one of
the four distributions**. The histogram is finished.

### 7. When the bins do not fit

256 bins is 1 KB. 4,096 bins is 16 KB and still fits in the 48 KB default.
8,192 bins is 32 KB — it fits, at 3 blocks per SM. 16,384 bins is 64 KB, which
needs the opt-in:

```cpp
CHECK(cudaFuncSetAttribute(k_shared,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
```

and buys you exactly **one** resident block per SM. 65,536 bins is 256 KB and
there is no launch configuration on any sm_89 part that will hold it.

`example02.cu` sweeps 256 → 65,536 bins with 2^26 32-bit keys, on a flat and a
skewed distribution, against five strategies. The % of the measured streaming
ceiling:

| strategy | 256 | 1024 | 4096 | 8192 | 16384 | 65536 |
|---|---|---|---|---|---|---|
| **[uniform]** | | | | | | |
| S0 global atomics | 4% | 10% | 25% | 43% | 35% | 34% |
| S1 shared privatization | 100% | 100% | 100% | 99% | 94% | **n/a** |
| S2 multi-pass windows | 100% | 100% | 98% | 100% | 50% | 12% |
| S3 global priv, G copies | 33% | 30% | 35% | 44% | 39% | 36% |
| S4 `cub::DeviceHistogram` | 56% | 29% | 34% | 42% | 38% | 15% |
| **[zipf]** | | | | | | |
| S0 global atomics | 3% | 3% | 4% | 5% | 5% | 5% |
| S1 shared privatization | 100% | 100% | 100% | 99% | 95% | **n/a** |
| S2 multi-pass windows | 100% | 100% | 98% | 99% | 50% | 12% |
| S3 global priv, G copies | 28% | 29% | 27% | 32% | 31% | 28% |
| S4 `cub::DeviceHistogram` | 51% | 38% | 31% | 37% | 30% | 20% |

Read four things off this table.

**(a) Shared privatization is at the ceiling wherever it is legal — including at
one resident block per SM.** 16,384 bins gives 64 KB of shared memory, 1 block/SM,
8 resident warps out of 48, and the kernel still reaches 94%. Module 11's Little's
Law argument explains it: a bandwidth-bound kernel needs enough *requests in
flight*, not enough warps, and 40 blocks each running a coarsened grid-stride
loop supply them. **Occupancy is not the figure of merit.**

**(b) Multi-pass costs exactly `P × floor`.** A window of `W` bins requires
`P = nBins/W` passes over the whole input, and the measurements are 100%, 50%,
12% of ceiling at `P = 1, 2, 8`. There is no hidden constant: the model is the
measurement. Multi-pass is therefore a *last* resort, and the right window is
the largest one you can hold, even at the cost of occupancy — which is exactly
what (a) licenses.

**(c) Global-memory privatization is the strategy for very large bin counts.**
`G` copies of the histogram in global memory, block `k` using copy `k & (G-1)`,
followed by a fold kernel. At 65,536 bins it is the fastest available option on
both distributions. Its cost model is worth stating:

```
footprint  = G * nBins * 4 B
extra traffic = footprint read + nBins*4 written by the fold
contention reduced by up to G
```

and it has a hard cliff. Exercise 2 measures `G = 256` at 65,536 bins — a 64 MB
privatized array — at **0.10× of the naive kernel**: once the copies stop fitting
in the 48 MB L2, every "cheap" atomic becomes a DRAM round trip and the cure is
ten times worse than the disease. **The privatized array must stay L2-resident.**
That is the same constraint Module 4 taught as a benchmarking hazard, reappearing
as a design rule.

**(d) At a high bin count with a flat distribution there is nothing to win.**
S0 at 65,536 uniform bins is already at 34% of ceiling and no strategy in the
table beats it by more than 1.06×. Module 10's table said privatization is a
0.22× *loss* at `K = 4096`; the same physics, at histogram scale and with a
proper grid, says the gentler thing: it is a no-op. Before optimizing a
histogram, **measure the naive version's percentage of the ceiling.** If it is
already 34%, the most an infinitely good atomic implementation can buy you is
2.9×, and you should check whether that is worth a week.

### 8. `cub::DeviceHistogram`, measured

```cpp
#include <cub/cub.cuh>
size_t bytes = 0; void* tmp = NULL;
cub::DeviceHistogram::HistogramEven(tmp, bytes, d_in, d_hist,
                                    nBins + 1, lo, hi, (int)n);
cudaMalloc(&tmp, bytes);
cub::DeviceHistogram::HistogramEven(tmp, bytes, d_in, d_hist,
                                    nBins + 1, lo, hi, (int)n);
```

The two-call query/allocate/run protocol is the same one Modules 12 and 13 used
for `DeviceReduce` and `DeviceScan`. Note `num_levels = nBins + 1`, not `nBins`:
CUB takes *boundaries*, not buckets. Temp storage at 65,536 bins measures
**31,457,791 B** — CUB is itself doing global-memory privatization, and you can
read the strategy off the allocation size.

The honest measurement: **the hand-written privatized kernel beats
`DeviceHistogram` at every bin count in the table**, by 1.8× at 256 bins and by
2.4–3.4× in the middle. This is not a defect in CUB. `HistogramEven` handles
arbitrary `[lo, hi)` ranges with a runtime division, arbitrary sample types,
multi-channel interleaved images, and `HistogramRange` handles non-uniform bin
boundaries. You are comparing a general routine against a kernel that hard-codes
`bin = byte`. Module 36 owns the library treatment; the lesson here is only that
**"use the library" is a defensible default and not automatically the fastest
answer**, and that you now have the measurement to decide.

### 9. The decision procedure

This is the deliverable of the module. Given `nBins` and what you know about the
distribution:

1. **Compute the floor** (`n × elementBytes / 410 GB/s`) and measure the naive
   kernel's percentage of it. That bounds every optimization you could do.
2. **Does `nBins × 4 B` fit in shared memory?**
   - ≤ 12 KB: yes, with 6 blocks/SM. Privatize.
   - ≤ 48 KB: yes, at reduced occupancy. Privatize anyway — a bandwidth-bound
     kernel does not need the occupancy.
   - ≤ 99 KB: yes, with the opt-in, at 1 block/SM. Still privatize (94% measured).
   - larger: go to 4.
3. **Choose the grid from the traffic model, not from `n`.** `grid × nBins ≪ n`.
   A machine-sized grid is right over a broad range.
4. **Bins do not fit:** global-memory privatization with `G` copies, sized so
   `G × nBins × 4 B` stays well inside L2. Multi-pass only if you cannot spare
   the global memory, and then with the largest window you can hold.
5. **Is the input skewed or clustered?** If yes, add warp-level aggregation
   (`__match_any_sync`) on top — Exercise 2 measures it worth an extra 1.5× on a
   hot distribution. If no, do not: Module 10 measured it at 0.46× when there is
   no contention to aggregate.
6. **Vectorize the input side.** This is usually the largest remaining factor and
   it has nothing to do with atomics.
7. **Replicate only if you have a reason** — a weighted histogram, or hardware
   without `ATOMS.POPC`. Measure before and after; on this GPU the counting case
   gains nothing and loses 17% if you cross an occupancy boundary.

---

## Hardware Mental Model

### Where each atomic executes, and what serializes

Module 10 established the two venues. Restated with the histogram's numbers:

| | global atomic | shared atomic |
|---|---|---|
| SASS | `RED.E.ADD.STRONG.GPU` (return discarded) | `ATOMS.POPC.INC.32` |
| executes at | the owning L2 slice | the SM's shared-memory unit |
| latency | L2 latency, ~241 cycles (M4) | SM-local, ~40 cycles |
| serializing unit | one slice ALU per address | one shared-memory bank/op |
| intra-warp same address | 32 serialized operations | **1**, merged by popcount |
| measured peak, 256 bins | 3.1–25 Gatomic/s depending on sector spread | not the bottleneck |

From `cuobjdump -sass example01.cubin`, unedited:

```
Function : _Z8k_globalPKhyPj                  // atomicAdd(&hist[in[i]], 1u)
    LDG.E.U8.CONSTANT R2, [R4.64] ;
    RED.E.ADD.STRONG.GPU [R2.64], R7 ;

Function : _Z6k_replPKhyPjii                  // atomicAdd(&my[in[i]], 1u)
    LDG.E.U8.CONSTANT R2, [R2.64] ;
    IMAD R7, R13, c[0x0][0x178], R2 ;
    ATOMS.POPC.INC.32 RZ, [R7.X4+URZ] ;
    BAR.SYNC.DEFER_BLOCKING 0x0 ;

Function : _Z10k_repl_vecPK6uchar4yPjii       // the uchar4 version
    LDG.E.CONSTANT R4, [R4.64] ;              // <- 128 B per warp, not 32
    ATOMS.POPC.INC.32 RZ, [R7.X4+URZ] ;
    ATOMS.POPC.INC.32 RZ, [R8.X4+URZ] ;
    ATOMS.POPC.INC.32 RZ, [R9.X4+URZ] ;
    ATOMS.POPC.INC.32 RZ, [R10.X4+URZ] ;
```

Three things to read off this dump.

**`RED`, not `ATOMG`.** Module 10's free optimization: the return value is
discarded, so the compiler emits the fire-and-forget form with no scoreboard
dependency. Every kernel in this module gets it automatically. If you write
`unsigned old = atomicAdd(...)` and never use `old`, you still get `RED` — but
if you store it anywhere you get `ATOMG` and a stall.

**No `VOTEU.ANY` anywhere.** Module 10 showed the compiler replacing 32
same-address atomics with one when it can *prove* the address is warp-uniform.
It cannot prove anything about `hist[in[i]]`. The `clustered` input is
warp-uniform at run time and gets no help at all, which is precisely why it
costs 61.0 ms against uniform's 50.4. Compiler aggregation is not available to a
histogram; hand-written `__match_any_sync` aggregation is.

**`ATOMS.POPC.INC.32` with a `RZ` destination.** One instruction. `.POPC.INC`
says: take the set of lanes in this warp targeting this shared address, count
them, apply that increment once. `RZ` as the destination is the shared-memory
equivalent of `RED` — no value comes back. This single instruction is why bin
replication is worthless here and why the clustered distribution is the *easy*
case for the privatized kernel even though it is the hard case for the global
one.

### Why the sector, and not the address, is the unit

An atomic is shipped to the L2 as {address, opcode, operand}. The L2 is banked
into slices; a slice owns a set of physical addresses by a hash, and requests
arriving at one slice are tagged by **sector** (32 B), because that is the
granularity at which the L2 tracks and moves data (Module 5). A warp's 32
atomics are coalesced into per-sector requests on the way out of the SM in the
same way a warp's 32 loads are. Thirty-two atomics into four sectors leave the
SM as four requests; thirty-two atomics into thirty-two sectors leave as
thirty-two. Part D's 7.99× is that 8:1 request ratio, undiluted.

The practical consequence for a histogram: **you cannot fix this by spreading
the bins out.** Module 10's fourth contention-reduction move — pad the bins apart
so they hash to different L2 slices — makes the sector count *worse*, not
better, because a warp's bins are already distinct and spreading them guarantees
one sector each. Padding helps when a few hot bins collide on one slice; it
hurts when many bins are touched per warp. Module 10 measured the same
non-monotonicity (its stride-8 column was worse than stride-1 at small `K`).
This is the one place in the module where an optimization from the previous
module is actively the wrong move, and it is worth knowing why.

### The three levels of contention, and which technique attacks which

| level | contending parties | technique | this module's measurement |
|---|---|---|---|
| within a warp | 32 lanes, one bin | `ATOMS.POPC` (shared) or `__match_any_sync` (global) | free in shared; 1.5× on top of global privatization for a hot input |
| within a block | 4–8 warps, one bin | bin replication | **1.00× — nothing to win** |
| across blocks | `grid` blocks, one bin | privatization (shared or global) | 27–39× at 256 bins; 1.06× at 65,536 uniform bins |
| across the device | the flush itself | coarsening (choose the grid) | 7.8× between the worst and best grid |

The row that matters most is the last one, and it is the row nobody optimizes
first.

---

## Code Walkthrough

### `example01.cu` — the ladder × the distribution

Five kernels and a streaming ceiling, on four inputs, in one rotated sweep of
24 configurations (SWEEPS = 24 ≥ NCFG = 24, spec §12.9). All 20 (kernel,
distribution) pairs are validated against an exact reference accumulated during
input generation.

**Part A**, the headline table (ms, min of 24 sweeps):

| kernel | uniform | zipf | same-bin | clustered |
|---|---|---|---|---|
| ceiling (stream only) | 0.4992 | 0.4994 | 0.4991 | 0.4996 |
| v0 global atomic | 50.40 | 76.76 | 103.89 | 61.04 |
| v1 shared priv, 1 elem/thread | 8.07 | 7.61 | 1.97 | 1.96 |
| v2 shared priv + coarsening | 1.096 | 1.094 | 1.093 | 1.093 |
| v3 + replication R=8 | 1.095 | 1.092 | 1.092 | 1.094 |
| v4 + `uchar4` loads | 0.503 | 0.501 | 0.501 | 0.500 |

and as speedup over v0:

| kernel | uniform | zipf | same-bin | clustered |
|---|---|---|---|---|
| v1 | 6.25× | 10.08× | 52.80× | 31.17× |
| v2 | 45.97× | 70.13× | 95.04× | 55.86× |
| v3 | 46.02× | 70.29× | 95.12× | 55.80× |
| **v4** | **100.2×** | **153.3×** | **207.6×** | **122.0×** |

The single most important row is **v4**: 0.500–0.504 ms on every distribution,
99–100% of the ceiling. **The finished kernel does not care what the input looks
like.** Every column of variance in the table above it is variance the naive
kernel had and the finished kernel does not. That flatness is the result; the
speedup column is just how far the starting point was from it, and it varies by
2× purely because v0 varies by 2×.

The second most important row is **v1 versus v2**: 8.07 → 1.10 ms, a **7.36×**
step, from changing nothing but the grid. And v1's own spread — 8.07 ms on
uniform, 1.97 ms on same-bin, *the same kernel* — is the `if (s[b])` flush guard
doing its work when 255 of 256 bins are empty.

**Part B** sweeps `R` and prints the shared footprint, the resident blocks per
SM that `cudaOccupancyMaxActiveBlocksPerMultiprocessor` reports for it, and the
resulting flush atomic count. It is the null result discussed in Concept §5.

**Part C** is the coarsening curve reproduced above.

**Part D** is the sector experiment. It is worth reading the kernel:

```cpp
template<int SCRAMBLE, int NSECT>
__global__ void k_sector(size_t n, unsigned int* hist)
{
    unsigned int lane = threadIdx.x & 31u;
    for (...) {
        unsigned int g = (unsigned int)(i >> 5) & 7u;
        unsigned int p = SCRAMBLE ? ((lane * 7u + 3u) & 31u) : lane;
        unsigned int addr = (NSECT == 4) ? (g * 32u + p)
                                         : ((p * 8u + g) & 255u);
        atomicAdd(&hist[addr], 1u);
    }
}
```

Both forms give each warp exactly 32 **distinct** bins out of 256. The first
packs them into one aligned 128 B block (4 sectors); the second spreads them
8 words apart (32 sectors). `SCRAMBLE` permutes which lane gets which of the 32,
controlling for any lane-ordering effect. This is Module 7's degree-counting
method and Module 12's "hold everything constant except the one variable"
discipline applied to atomics.

Note the file re-warms for 1500 ms before Parts B, C and D. Sustained streaming
engages the power cap (cross-module index §7) and the absolute ms in later parts
are not comparable to Part A's; the file says so in its own output. Ratios
within a part reproduce to ~1%.

### `example02.cu` — the bin-count axis and CUB

Five strategies, six bin counts, two distributions, with a 1500 ms re-warm
before each bin count and a rotated sweep of 12 configurations within it. The
per-bin-count ceiling drifts from 0.65 ms to 0.83 ms across the run, which is
exactly why **% of a ceiling measured in the same sweep** is the reported
quantity and not GB/s.

The strategy-availability table it prints first is the lesson in miniature:

```
  nBins   smem/blk  blocks/SM  S1?   S2 window x passes   S3 copies (footprint)
    256       1024          6  yes      256 x 1            64 (  0.06 MB)
   1024       4096          6  yes     1024 x 1            64 (  0.25 MB)
   4096      16384          5  yes     4096 x 1            64 (  1.00 MB)
   8192      32768          3  yes     8192 x 1            64 (  2.00 MB)
  16384      65536          1  yes     8192 x 2            64 (  4.00 MB)
  65536     262144          0  NO      8192 x 8            16 (  4.00 MB)
```

The `blocks/SM` column falls 6 → 5 → 3 → 1 → 0 as `nBins × 4` crosses 12 KB,
25 KB, 49 KB and the 99 KB opt-in maximum — the same table Module 6 measured for
an unrelated kernel, because it is a property of the hardware and not of the
code. Module 19 will derive it.

---

## Check Your Understanding

Answers in `solutions/module14/check_your_understanding.md`.

**Q1.** Two 256-bin histogram kernels differ in one line. Kernel A does
`atomicAdd(&hist[in[i]], 1u)` on global memory. Kernel B does the same thing but
on a `__shared__` array of 256 bins with no zeroing, no barriers and no flush —
it is deliberately incorrect, and exists only to be timed. On the `clustered`
input (a warp's 32 lanes always share a bin) kernel A is 61.0 ms and kernel B is
under 1.2 ms. Explain the ratio using the *instruction* each compiles to, and
then say what would happen to each if the input were changed so that a warp's 32
lanes hit 32 *different* bins. One of them gets faster and one of them does not
change; say which and why.

**Q2.** You have 4,096 bins, 32-bit keys, and an input you know almost nothing
about except that fewer than 20 bins are ever occupied. Your colleague proposes
bin replication with `R = 8` because "the distribution is extremely skewed, so
contention will be terrible." Give the strongest argument you can that they are
right, then the argument that they are wrong, and say which single measurement
decides it. (`4096 × 4 B = 16 KB`; `8 × 16 KB = 128 KB`.)

**Q3.** A histogram kernel over 2^26 elements privatizes into shared memory and
uses a grid of 240 blocks. It is exactly correct at 1,024 bins and produces
counts that are *too low* at 64 bins, reproducibly, with the same input. No
sanitizer reports anything. Give the mechanism. Then say what happens to the
symptom if you keep 64 bins and raise the grid to 4,096 blocks, and explain why
that is a fix and not a coincidence.

**Q4.** Module 10 measured global atomic throughput at 23.35 Gatomic/s for
`K = 256` distinct bins. This module measures 4.0 Gatomic/s for a 256-bin
histogram of uniformly random bytes and 25.0 Gatomic/s for a synthetic kernel
that also touches 32 distinct bins per warp. All three numbers are correct.
Reconcile them in one paragraph, and then state the rule you would give a
colleague for predicting atomic cost from source code — being explicit about
what information the source code alone does *not* contain.

---

## Exercises

### Exercise 1 — `exercise01.cu` (optimization + prediction + design; 5 TODOs)

Build the four rungs above a global-atomic 256-bin byte histogram: shared
privatization, coarsening, replication, vectorized loads. The harness times six
kernels × three distributions in one rotated sweep of 18 configurations and
validates every one against an exact CPU reference.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
compute-sanitizer --tool racecheck .\exercise01.exe
compute-sanitizer --tool initcheck --initcheck-address-space shared .\exercise01.exe
```

**The block is 128 threads and there are 256 bins.** TODO 1 is three predicted
speedups, committed before you build and scored to within a factor of two.
TODO 2 is the privatized kernel stated as four requirements rather than four
lines. TODO 3 and the second half of TODO 4 are **design** TODOs: the grid, and
the replication factor. TODO 5 is the vectorized version.

Validation: exact counts from all four of your kernels on all three
distributions, the vectorized kernel at ≥ 1.50× the coarsened one on every
distribution, the coarsened kernel at ≥ 20× the baseline on `same-bin`, and all
three predictions within a factor of two. Both performance gates are ratios
between kernels timed in the same sweep, per spec §12 rule 5; the % of ceiling
is printed but not scored, because on a thermally loaded machine the ceiling's
own slot in the rotation can land in a cooler window than the kernel it is the
denominator for. `score: 9/9` for `OVERALL: PASS`.

### Exercise 2 — `exercise02.cu` (design; 4 TODOs)

65,536 bins. The histogram is 262,144 bytes and the largest shared-memory
allocation this GPU permits is 101,376. The move you have spent the whole module
learning is unavailable. You get 64 MB of device scratch, any number of kernel
launches, and a requirement rather than a technique.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

The harness gates on **both** input distributions: ≥ 5.00× on an input with 75%
of its mass in 32 bins, **and** no worse than 0.90× on a flat one. The obvious
adaptation of shared-memory privatization to this bin count fails both halves.
TODO 1 asks for both speedups in advance; one of your two numbers should be
much less exciting than the other, and if it is not, you have not understood
Concept §7(d).

### Exercise 3 — `exercise03.cu` (debugging; 4 TODOs)

A privatized histogram that packs two 16-bit bin counters into every 32-bit
word — a real technique that halves the shared footprint. It contains three
defects. Two produce symptoms; the third produces none at all on this GPU in
this configuration, and you will find it by reading.

```
nvcc -arch=sm_89 -O3 -lineinfo -o exercise03.exe exercise03.cu
.\exercise03.exe
compute-sanitizer --tool racecheck  .\exercise03.exe
compute-sanitizer --tool initcheck --initcheck-address-space shared .\exercise03.exe
```

TODO 1 asks you to commit, **before running any tool**, to which of memcheck,
racecheck, initcheck, or none finds each defect. Scored against a hash; there is
no partial credit and the answer is not in the file. TODO 4 is a design TODO
with a constraint that forbids the obvious fix, and the derivation it forces you
through is the last idea in the module.

---

## Prediction

Commit to these in writing before you run anything.

**P1.** `example01.cu` runs the same global-atomic kernel on four inputs that
differ only in their value distribution. Rank the four — uniform, zipf
(40% in one bin), same-bin (100% in one bin), clustered (a warp's lanes always
share a bin) — from fastest to slowest, and predict the ratio between the
fastest and the slowest to within a factor of two. Then predict the same
ranking for the *finished* kernel (v4), and the ratio there.

**P2.** Module 10 measured shared-memory privatization at 48.2× over plain
global atomics at `K = 1` and at **0.22×** — a 4.5× loss — at `K = 4096`.
`example02.cu` measures shared privatization at 4,096 bins as a **3.97× win**
on a uniform input. Both are correct. Before reading §7, state what is different
about the two setups, and predict which single parameter of Module 10's
experiment you would have to change to reproduce its 0.22× here.

**P3.** Bin replication with `R = 8` costs 8 KB of shared memory per block
instead of 1 KB and requires an 8-term fold at flush time. Predict its speedup
over `R = 1` on the `same-bin` input — where every thread in the block is
hammering one bin — to within a factor of two, and write down the mechanism you
expect to be responsible. Then look at Part B. If you were wrong, the
explanation is a single SASS instruction; find it in the dump.
