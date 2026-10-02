# Module 23 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -lineinfo -o e2s.exe exercise02_solution.cu
e2s.exe
```

---

## TODO 1 — distinct keys across a warp, and sectors per request

```cpp
__device__ __forceinline__ unsigned distinctInWarp(unsigned key)
{
    unsigned m    = __match_any_sync(FULL, key);
    int      lane = (int)(threadIdx.x & 31u);
    bool     lead = (__ffs((int)m) - 1) == lane;
    return (unsigned)__popc(__ballot_sync(FULL, lead));
}

__device__ __forceinline__ unsigned sectorsThisRequest(const void *addr)
{
    return distinctInWarp((unsigned)(((unsigned long long)addr) >> 5));
}
```

`__match_any_sync(mask, key)` (sm_70+) returns, for each lane, the mask of lanes
holding the *same* key. That one instruction does the grouping. Everything after
it is bookkeeping:

- `__ffs(m) - 1` is the lowest set bit of my own group — the group's canonical
  leader. Exactly one lane per distinct value satisfies `lead`.
- `__popc(__ballot_sync(FULL, lead))` counts the leaders, which is the number of
  distinct groups.

Every lane computes the same answer, which matters because the caller must be
able to read it from any lane.

`>> 5` is the sector index: sectors are 32 B. That is Module 5's hand procedure —
"take the 32 byte addresses, shift right by 5, count the distinct values" —
compiled. The quotient `l1tex__t_sectors_... / l1tex__t_requests_...` is defined
to be this number, so the function is not an approximation of the counter, it is
the counter.

**Wrong approaches and their symptoms:**

- *A loop over 32 lanes with `__shfl_sync`.* Correct, 32× slower, and it teaches
  you nothing about why the hardware can do it in one pass.
- *Counting with an atomic into shared memory.* Also correct, and it changes the
  kernel's own shared-memory behaviour, which is the thing Part B measures.
- *Using lane 0 as the leader instead of the lowest set bit of the group mask.*
  Collapses to counting 1 if lane 0 is active and 0 otherwise.

---

## TODO 2 — conflict degree, counting **distinct words**

```cpp
__device__ __forceinline__ unsigned conflictDegree(unsigned wordIndex)
{
    unsigned m    = __match_any_sync(FULL, wordIndex);
    int      lane = (int)(threadIdx.x & 31u);
    bool     lead = (__ffs((int)m) - 1) == lane;     // one lane per DISTINCT word
    unsigned bank = wordIndex & (BANKS - 1);

    unsigned deg = 0;
    #pragma unroll
    for (int b = 0; b < BANKS; ++b) {
        unsigned bm = __ballot_sync(FULL, lead && bank == (unsigned)b);
        if ((unsigned)__popc(bm) > deg) deg = (unsigned)__popc(bm);
    }
    return deg;
}
```

The `lead &&` is the entire exercise. Drop it and you count **lanes** per bank;
keep it and you count **distinct words** per bank, which is what the hardware
does and what the counter reports.

The program prints both, and they disagree on exactly two of the eight rows:

```
   expression      degree  predicted   wavefronts   conflicts    by-lane
   s[tid]               1          1            1           0          1
   s[2*tid]             2          2            2           1          2
   s[3*tid]             1          1            1           0          1
   s[4*tid]             4          4            4           3          4
   s[8*tid]             8          8            8           7          8
   s[32*tid]           32         32           32          31         32
   s[tid/2]             1          1            1           0          2
   s[0]                 1          1            1           0         32
```

`s[tid/2]`: sixteen lanes ask bank 0 for word 0 and sixteen ask bank 1 for
word 1 — wait, more precisely, two lanes per word for sixteen consecutive words
in sixteen consecutive banks. By lanes: 2 per bank. By distinct words: 1 per
bank. **The hardware broadcasts a word to every lane that wants it, for free**
(M7), so the real degree is 1.

`s[0]`: all 32 lanes, one bank, one word. By lanes: 32 — a phantom 32-way
conflict, the worst possible reading. By distinct words: 1. **Free.**

This is the single most common way to misread
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`, and it is why the
lesson spends a paragraph on what a wavefront is.

### The two cost laws, which the degree alone does not give you

- **4-byte accesses: cost ∝ `max(2, D)`.** There is a two-cycle floor for any
  32-lane 4 B shared access, so `s[2*tid]` reports one conflict per request and
  costs **1.00×** (M7 measured it). A small non-zero conflict count on a 4-byte
  access is not a finding.
- **`LDS.128` (16-byte) accesses: cost ∝ `D`, no floor.** M18 isolated this with
  two kernels that are byte-identical in SASS, registers, spills, shared bytes,
  blocks/SM and occupancy, and differ only in the B-tile's per-phase bank
  degree (1 vs 2): predicted 1.67×, measured **1.61×**. A 16 B access is
  phase-split into four phases of 8 lanes, and one phase already asks the
  crossbar for its full 128 B width, so there is no spare cycle for a 2-way
  conflict to hide in.

---

## TODO 3 — the two occupancy denominators

```cpp
unsigned long long maxSpan = 0;
for (int i = 0; i < NSM; ++i)
    if (en[i] > st[i]) {
        unsigned long long sp = en[i] - st[i];
        if (sp > maxSpan) maxSpan = sp;
    }
double sumA = 0.0, sumE = 0.0; int used = 0;
for (int i = 0; i < NSM; ++i) {
    if (en[i] <= st[i]) continue;
    double span = (double)(en[i] - st[i]);
    sumA += (double)res[i] / (WARP_SLOTS_PER_SM * span);        // this SM's span
    sumE += (double)res[i] / (WARP_SLOTS_PER_SM * (double)maxSpan);
    ++used;
}
*occActive  = used ? 100.0 * sumA / used : 0.0;
*occElapsed = 100.0 * sumE / NSM;
```

Three decisions, each of which changes the answer.

**The denominator of `occActive` is per-SM; the denominator of `occElapsed` is
shared.** That is the whole difference between
`..._pct_of_peak_sustained_active` and `..._pct_of_peak_sustained_elapsed`.

**`occActive` averages over `used`; `occElapsed` averages over `NSM`.** An SM
that ran nothing has no "busy span", so it cannot appear in the first average at
all. It *must* appear in the second as a zero, because idle silicon is exactly
what that metric is for.

**`maxSpan` is a maximum of per-SM spans, never `max(en) − min(st)` across SMs.**
The `%clock64` counters are per-SM and are not mutually synchronised: Module 19
measured up to **298 million cycles** of offset inside a single launch.
Subtracting one SM's timestamp from another's produces a number with no physical
meaning, and it will usually be a plausible-looking one.

### What it measures

```
   configuration               blocks   occ_active  occ_elapsed        ms
   uniform, exactly 1 wave        240        67.3%        67.3%    0.463
   uniform, 1 wave + 1 blk        241        67.3%        58.0%    0.526
   1..8x imbalance, 1 wave        240        70.0%        42.2%    0.738
```

- **One extra block** — a tail of one block on one SM while 39 idle — moves
  `occ_active` by 0.0 points and `occ_elapsed` by 9.3, and costs 13.6% of the
  wall time.
- **A 1..8× imbalance** moves `occ_active` **up** by 2.7 points while
  `occ_elapsed` falls 25.1 and the kernel takes **59% longer**.

Module 19 measured the same two effects at 240→241 blocks (`active` +0.4,
`elapsed` −4.0, wall +7.5%) and for the imbalance (`active` +3.4,
`elapsed` −14.4, wall +53%). Same signs, same mechanism, different magnitudes —
this harness uses a shorter kernel.

**`ncu`'s Occupancy section shows `occ_active` and not `occ_elapsed`.** The
`_elapsed` form exists and you can ask for it with `--metrics`, but the number
every tutorial quotes as "achieved occupancy" is the one that cannot see either
of these two effects. A load imbalance makes it go *up*.

> Note the third fact M19 established, which this harness also shows: the
> uniform one-wave case achieves 67.3%, not 100%, with zero tail and zero
> imbalance. The cause is greedy warp scheduling — warps enter together and
> leave spread out. Achieved occupancy below theoretical is not, by itself,
> evidence of anything.

---

## TODO 4 — lane efficiency

```cpp
__device__ __forceinline__ void recordIssue(unsigned long long *c)
{
    unsigned m    = __activemask();
    int      lead = __ffs((int)m) - 1;
    if ((int)(threadIdx.x & 31u) == lead) {
        atomicAdd(&c[0], (unsigned long long)__popc(m));   // thread_inst_executed
        atomicAdd(&c[1], 1ull);                            // inst_executed
    }
}
```

The reporter must be **the lowest set bit of the active mask**, not lane 0. In a
region guarded by `if (i & 1)`, lane 0 is inactive half the time; using
`lane == 0` silently counts only the even-predicate branch and the program still
prints a perfectly plausible ratio. This is the quietest failure in the exercise.

```
   predicate                     thr_inst/inst    predicted   lane eff
   p = i & 1        (2-way)              16.00        16.00      50.0%
   p = (i>>5) & 1   (uniform)            32.00        32.00     100.0%
   p = (i&31)==7    (1 of 32)             1.00         1.00       3.1%
```

Mode 0 issues both arms with 16 lanes on in each, so the average over issued
instructions is 16.00 and lane efficiency is 50%. That is real divergence, and
Module 8 measured its cost at **1.96×** for exactly this predicate.

Mode 2 is the trap. One lane in 32 takes a rare side branch; the counter reports
**1.00**, i.e. **3.1% lane efficiency**, for a kernel that is almost entirely
healthy. The ratio is an average over *issued instructions*, not over time, so a
cheap rare branch sinks it while costing nearly nothing. **Never read lane
efficiency without the instruction count it is averaged over.**

### Divergence vs predication — the pair of counters

Module 8 proved `__activemask()` alone cannot distinguish a predicated-off lane
from a not-taken lane: both are absent from the mask, and the fix is different
in each case. `ncu` gives you a second counter that closes the gap:

```
smsp__thread_inst_executed_per_inst_executed.ratio          "Avg. Active Threads Per Warp"
smsp__thread_inst_executed_pred_on_per_inst_executed.ratio  "Avg. Not Predicated Off Threads Per Warp"
```

A **predicated-off** lane is active but not predicated-on: counted by the first,
not by the second. A **branched-away** lane is neither: counted by neither.
**The gap between the two metrics is exactly the predication; a real branch
lowers both.** That is this module paying M8's debt, and it is a thing the
course could not derive on its own because `__activemask()` does not expose the
predicate.

---

## TODO 5 — the closed forms

```cpp
static int predictSectorsStride(int strideFloats, int baseFloats)
{
    const long long s = 4LL * strideFloats;
    const long long b = 4LL * baseFloats;
    if (s == 0) return 1;
    if (s >= SECTOR_B) return 32;
    const long long lo = b / SECTOR_B;
    const long long hi = (b + 31 * s) / SECTOR_B;
    return (int)(hi - lo + 1);
}
```

Two regimes.

**Once the byte stride reaches the sector size, every lane owns its own sector**
and the answer saturates at 32. It cannot grow past 32 because there are 32
lanes. This is why `in[8*i]` and `in[16*i]` and `in[1024*i]` all report the same
thing — and why a profile showing 32.00 tells you the pattern is scattered but
not *how* scattered.

**Below that, the 32 lanes cover a contiguous byte range** `[b, b + 31s]`, and
the number of sectors it meets is the number of sector boundaries it crosses
plus one: `floor((b+31s)/32) − floor(b/32) + 1`. The `b` term is where
misalignment lives. For `in[i]`: `b = 0`, `b + 31s = 124`, so
`floor(124/32) − floor(0/32) + 1 = 3 − 0 + 1 = **4**`. For `in[i+1]`: `b = 4`,
`b + 31s = 128`, so `4 − 0 + 1 = **5**`.

**Four bytes of misalignment costs a quarter more sectors, on every load,
forever.** Module 5 predicted 80% efficiency for it and *measured 97.9%* —
because the boundary sector was already resident from the neighbouring warp. The
instruction-level counter says 5 and the traffic counters say almost nothing
extra moved, and both are right. They count at different levels. This is the
single best illustration in the course of why you must name the level.

```cpp
static int predictDegreeStride(int strideWords)
{
    if (strideWords == 0) return 1;
    int a = strideWords & 31, b = BANKS;
    if (a == 0) return BANKS;
    while (b) { int t = a % b; a = b; b = t; }
    return a;                                   // gcd(strideWords, 32)
}

static int predictDegreeDivide(int k) { (void)k; return 1; }
```

`D = gcd(k, 32)`. Lanes land in the same bank when `k·lane` is congruent mod 32,
which happens every `32/gcd(k,32)` lanes, giving `gcd(k,32)` lanes per bank — and
their *words* differ by `32·(k/gcd)` floats, so every one of them is a genuine
conflict. That is why `s[3*tid]` is conflict-free (3 is coprime with 32) and
`s[4*tid]` is 4-way.

`s[tid/k]` is **not** of that form and the formula does not apply. The warp asks
for only `32/k` distinct words, they are consecutive, so they occupy `32/k`
consecutive banks with one word each: **degree 1 for every k.** A broadcast, not
a conflict.

---

## Expected output

```
=== Module 23 / Exercise 2 - derive the counters yourself ===
ncu status: ERR_NVGPUCTRPERM. Every quantity below is one the tool
would report; here the kernel computes it about itself.

-- A. sectors per request ------------------------------------------
   expression       measured  predicted bytes/sector used/moved
   in[i]                   4          4        32.00     100.0%
   in[i + 1]               5          5        25.60      80.0%
   in[2*i]                 8          8        16.00      50.0%
   in[4*i]                16         16         8.00      25.0%
   in[8*i]                32         32         4.00      12.5%
   in[16*i]               32         32         4.00      12.5%
   in[0]                   1          1       128.00  broadcast
   in[2*i + 1]             8          8        16.00      50.0%
...
SCORE: 23/23
OVERALL: PASS
```

Parts A, B and D are exact and reproduce bit-for-bit on every run — they are
enumerations, not timings, and they do not depend on the clock. Part C's three
occupancy numbers vary by a point or two between runs; the three structural
facts it scores (tail, imbalance up, imbalance down) were stable across every
run, including runs taken from a power-capped operating point.

---

## The result that matters

**You can build most of Nsight Compute's memory and warp sections out of
`__match_any_sync`, `__ballot_sync`, `__popc`, `__activemask`, `clock64()` and
`%smid`.** Doing it once changes what the tool's output *is* to you: not a
number to look up, but a quantity you know the definition of, the denominator
of, and the failure mode of.

Three of those failure modes are only visible from the inside:

1. The bank-conflict counter counts **distinct words**, not lanes, so a free
   broadcast reports degree 1 — and a reader who assumes lanes will invent a
   32-way conflict that costs nothing.
2. Achieved occupancy's `active` denominator makes a load imbalance look like an
   improvement.
3. Lane efficiency is averaged over issued instructions, so a rare cheap branch
   can report 3.1% on a healthy kernel.

**A variation worth doing:** change `kBanks`'s shared array to `float4` and read
it with 16-byte accesses. The conflict *degrees* your function reports will not
change — but the cost law does, from `max(2,D)` to `∝D` with no floor (M18), so
the same counter value now means something different. Then try to find the
metric in `ncu` that tells you the access width. There isn't one; the SASS
opcode is where that lives, which is Module 39.
