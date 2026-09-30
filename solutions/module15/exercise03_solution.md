# Module 15 / Exercise 03 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
```

---

## TODO 1 — `sectorsOfWarp`

```cpp
int sectorsOfWarp(const unsigned long long* addr)
{
    unsigned long long seen[32];
    int n = 0;
    for (int lane = 0; lane < 32; ++lane) {
        unsigned long long sec = addr[lane] >> 5;        // addr / 32
        bool dup = false;
        for (int k = 0; k < n; ++k) if (seen[k] == sec) { dup = true; break; }
        if (!dup) seen[n++] = sec;
    }
    return n;
}
```

Module 5's procedure, unchanged: map each address to `addr >> 5` and count
**distinct** ids. The `seen` set is order-independent by construction, which is
what the harness's reversed-lane test checks — the coalescer sees a *set* of
addresses, never a sequence, and an implementation that can tell the difference
is not a model of the hardware.

Each element is 4 bytes, so no lane straddles a sector boundary; a `float4`
version would have to walk from `a` to `a + size - 1` in 32 B steps (M5).

**Common wrong approaches.** Sorting the addresses first and counting runs —
works, but conceals that order is irrelevant. Dividing by 128 (the cache *line*)
instead of 32 (the *sector*): the line is the tag granularity, the sector is the
fill and DRAM-traffic granularity, and a strided kernel pays DRAM in 32 B units.
The harness's stride-8 test returns 32 with the right divisor and 32 with the
wrong one, but the stride-2 test returns 8 with the right one and 4 with the
wrong one.

---

## TODO 2 — `degreeOfWarp`

```cpp
int degreeOfWarp(const int* off)
{
    int words[32][32], nw[32];
    for (int b = 0; b < 32; ++b) nw[b] = 0;
    for (int lane = 0; lane < 32; ++lane) {
        int w = off[lane];
        int bank = ((w % 32) + 32) % 32;
        bool seen = false;                      // DISTINCT WORDS, never lanes
        for (int k = 0; k < nw[bank]; ++k) if (words[bank][k] == w) { seen = true; break; }
        if (!seen) words[bank][nw[bank]++] = w;
    }
    int d = 0;
    for (int b = 0; b < 32; ++b) if (nw[b] > d) d = nw[b];
    return d;
}
```

Module 7's procedure. The single line that separates the correct model from the
naive one is `if (!seen)`: same bank + **same word** is a broadcast and costs one
cycle; same bank + **different word** is a replay. Drop it and you count lanes,
and the harness's `off[i] = i/2` test returns 2 instead of 1.

---

## TODO 3 — the paper table

```
configuration                    rd  wr  st  ld
1 naive coal rd / strided wr      4  32   0   0
2 naive strided rd / coal wr     32   4   0   0
3 tile[32][32]                    4   4   1  32
4 tile[32][33]                    4   4   1   1
5 tile[32][34]                    4   4   1   2
6 tile[16][17], block(16,16)      4   4   2   2
7 32x32 XOR swizzle               4   4   1   1
```

Row by row.

**1 and 2.** A warp is 32 consecutive `threadIdx.x` at fixed `threadIdx.y`. The
coalesced side touches 128 contiguous, 32 B-aligned bytes → 4 sectors. The
strided side has lanes `4·8192 = 32768` bytes apart → one sector each → 32. The
two rows are the same permutation indexed from opposite ends, so the 4 and the
32 simply swap columns. No shared memory, so `0` and `0`.

**3, `tile[32][32]`.** Flat index `32r + c`, so `bank = (32r + c) mod 32 = c` —
the bank depends only on the column. The store `tile[ty+j][tx]` has `c = tx`
varying over 0..31: 32 distinct banks, `D = 1`. The load `tile[tx][ty+j]` has
`c = ty+j` **constant across the warp** and `r = tx` varying: all 32 lanes in one
bank, asking for 32 different words, `D = 32`.

**4, `tile[32][33]`.** `bank = (33r + c) mod 32 = (r + c) mod 32`. Row `r` is
rotated by `r` banks. Store (`r` fixed, `c` = 0..31): a permutation, `D = 1`.
Load (`c` fixed, `r` = 0..31): also a permutation, `D = 1`. The general rule is
`gcd(pitch, 32) == 1`.

**5, `tile[32][34]` — the trap.** `bank = (34r + c) mod 32 = (2r + c) mod 32`.
Store (`r` fixed, `c` varying): still a permutation, `D = 1`. Load (`c` fixed,
`r` = 0..31): `2r mod 32` takes 16 values, each hit by two lanes, at two
*different* words. **`D = 2`.**

If you wrote 4 here — reasoning "`gcd(34,32) = 2`, so the degree is 2, and 2-way
conflicts are the same as 4-way conflicts scaled down" — you had the degree
right. If you wrote 1, you assumed any odd-looking pad works. If you wrote 4,
you confused the gcd with the degree: the degree **is** the gcd, not the pitch's
distance from odd. And the measurement is the real punchline: `tile[32][34]`
measures **within 1% of `tile[32][33]`**, because Module 7's cost law on Ada is
`max(2, D)` — a conflict-free 32-lane 4 B access already occupies the shared
pipeline for two cycles, so a 2-way conflict fits inside the slack and is free.
Two extra bytes of pitch per row buy exactly nothing over one.

**6, `tile[16][17]` with a `(16,16)` block — the one that requires care.**
Warp 0 is `threadIdx` linear ids 0..31 (M3), which with a 16-wide block means
`threadIdx.y ∈ {0,1}` and `threadIdx.x ∈ 0..15`. **A warp spans two tile rows.**

- *Global read*, `in[y*W + x]`: lanes 0–15 read 16 consecutive floats = 64 bytes
  at `(by*16)*W + bx*16`, which is 64 B-aligned, so 2 sectors; lanes 16–31 read
  the next row, another 2. **4 sectors**, and the efficiency is still 100%.
  The same for the write. The sector model says a 16-wide tile is perfect, and
  the measurement broadly agrees: config 6 lands 1–3% behind config 4 in every
  sweep. If you expected a 16-wide tile to be a coalescing disaster, the model
  was right and the intuition was wrong — a 64 B run is two whole sectors, not a
  partial one. The residual 1–3% is consistent with worse DRAM page locality
  (two 64 B runs 32 KB apart versus one 128 B run) but is too small, and this
  machine too noisy, to call it measured.
- *Shared store*, offsets `ty*17 + tx`: lanes 0–15 give 0..15 (banks 0..15),
  lanes 16–31 give 17..32 (banks 17..31 and, for offset 32, bank 0). Bank 0 is
  asked for word 0 and word 32 — two distinct words. **`D = 2`.**
- *Shared load*, offsets `tx*17 + ty`: lanes 0–15 give `17·tx`, banks
  {0,2,4,…,14,17,19,…,31}; lanes 16–31 give `17·tx + 1`, banks
  {1,3,…,15,18,20,…,0}. The two sets are disjoint except at bank 0, which gets
  word 0 (from `tx=0, ty=0`) and word 256 (from `tx=15, ty=1`, and `256 mod 32 =
  0`). **`D = 2`.**

Both phases are 2-way, i.e. free on Ada. If you wrote `1, 1` you forgot the warp
spans two rows; if you wrote `1, 32` you applied the 32-wide analysis to a
16-wide tile. The pitch 17 is odd, so `gcd(17,32) = 1`, and the *rule* would say
`D = 1` — but the rule was derived for a warp that covers one full tile row, and
this warp does not. **A rule has a domain.**

**7, the XOR swizzle.** `bank = (32r + (c ^ r)) mod 32 = c ^ r`. Store (`r`
fixed): `c ^ r` over `c = 0..31` is a bijection, `D = 1`. Load (`c` fixed):
`c ^ r` over `r = 0..31` is a bijection, `D = 1`. Same as padding, 32 floats less
memory.

---

## TODO 4 — the general padding rule

```cpp
int padPitch(int tileW, int elemBytes)
{
    (void)elemBytes;
    return (tileW % 2 == 0) ? tileW + 1 : tileW;
}
```

**The element size does not enter the answer.** That is the point of the TODO,
and it is a genuine generalization of three separate statements Module 7 made.

Derivation. Let the pitch be `P` elements of `E` bytes. Element `(r,c)` starts at
byte `(rP + c)·E`, i.e. at word `(rP + c)·E/4`, and occupies `E/4` consecutive
words. The bank array delivers 128 B/cycle, so the warp is split into
`phases = E/4` groups of `32/phases` lanes, and conflicts are resolved
independently within a phase (M7).

A column walk has `c` fixed and `r` running over the `32/phases` lanes of one
phase. The words that phase requests are
`{(rP + c)·E/4 + k : r = 0..L-1, k = 0..E/4-1}` with `L = 32·4/E`. It is
conflict-free iff those `L·E/4 = 32` words occupy 32 distinct banks, which
happens iff `r ↦ (rP·E/4) mod 32` is injective on `r = 0..L-1`. The step is
`P·E/4` and the modulus 32, so the map has period `32/gcd(P·E/4, 32)` and is
injective on `L = 128/E` values iff

```
gcd(P · E/4, 32) = E/4        i.e.        gcd(P, 32·4/E) = 1
```

- `E = 4`  → `gcd(P, 32) = 1`
- `E = 8`  → `gcd(P, 16) = 1`
- `E = 16` → `gcd(P, 8)  = 1`

All three are satisfied by exactly the same set of `P`: **the odd numbers.** A
row walk (`r` fixed, `c` running) is conflict-free for any `P` whatsoever, since
the addresses are consecutive. So the smallest legal pitch is the smallest odd
number `>= tileW`, and since every `tileW` the harness tests is a multiple of 32,
that is `tileW + 1`.

The harness brute-forces the truth with `phaseFreeFor()` and agrees: 33, 65, 97,
129 for every element size.

**Common wrong approaches.** "Pad to a multiple of 8 so `float4` still works" —
M5's alignment instinct applied one level down, where it is not merely irrelevant
but exactly backwards; `gcd(40,32) = 8`. "For `double` you need two extra
elements because a double is two words" — M7 already corrected this
(`sd[17*tid]` is conflict-free); the phase split means one extra `double`, not
one extra `float`'s worth. "The answer depends on `elemBytes`" — it does not,
and noticing that is the exercise.

---

## TODO 5 — what the conflict is worth

`RATIO_DRAM = 1` (below 1.10×), `RATIO_L2 = 3` (above 2×).

Measured: **1.036× at 8192 × 8192** and **2.756× at 2048 × 2048**. Across all
authoring runs, 0.96–1.04× and 2.67–3.25×.

The reasoning is the module's central argument. A conflict is a *replay* in the
LSU pipeline; replays consume issue slots, not DRAM bandwidth. Per thread the
tiled kernel issues 4 `LDG`, 4 `STG`, 4 `STS`, 4 `LDS`. At `D = 32` the four
`LDS` occupy the shared pipeline for 32 cycles each instead of 2, adding ~120
cycles per thread, ~960 per 256-thread block. The same block moves 32 KB through
DRAM: at 373 GB/s over 40 SMs that is ~3.5 µs, ~7000 cycles. The shared term fits
inside the memory term, and a term inside the binding constraint is free.

Put both buffers in the 50 MB L2 and the memory term falls by ~4.5× (apparent
bandwidth ~1671 GB/s against 373). Now the LSU is the constraint and the same
conflict shows up — as 2.8×, not 16×, because it is diluted by the eight global
instructions that did not get slower. See `check_your_understanding.md` Q4 for
the inequality that reconciles this with Module 7's 14.79×.

---

## Performance reasoning: what the table actually predicted

```
MEASURED at 8192 x 8192 (min of 8 rotated sweeps)
configuration                         ms      GB/s   %ofcopy
0 copy ceiling                    1.5435     347.8    100.0%
1 naive coal rd / strided wr      3.2491     165.2     47.5%
2 naive strided rd / coal wr      1.6092     333.6     95.9%
3 tile[32][32]                    1.6377     327.8     94.3%
4 tile[32][33]                    1.5812     339.5     97.6%
5 tile[32][34]                    1.5932     337.0     96.9%
6 tile[16][17], block(16,16)      1.6224     330.9     95.1%
7 32x32 XOR swizzle               1.6215     331.1     95.2%
```

Now compare the four columns you computed against that last column.

- **The write-sector count predicted config 1 and nothing else.** 32 sectors on
  the write side → 47.5% of copy, by far the worst row. Correct, and it is the
  only column in the table that identifies the real problem.
- **The read-sector count over-predicted config 2 badly.** Config 2 also has a
  32-sector side and measures **95.9%**, nearly at the ceiling. Two reasons:
  these kernels carry no bounds guard, so the compiler hoists all four loads
  ahead of all four stores and four 32-sector requests are in flight at once
  (with the guard, the same kernel drops to 67.5% — see the SASS in the lesson);
  and a strided *read* costs one DRAM transaction per sector whereas a strided
  *write* costs a fill plus a writeback, because a partial-sector store must be
  read-merge-written at L2 (M5, M11's 3.95×).
- **Neither shared-memory column predicted anything at all.** Configs 3 through 7
  span conflict degrees of 32, 1, 2, 2 and 1 and land within **3.5%** of each
  other. Ranked by measured time the order is 4, 5, 7, 6, 3 — which is *almost*
  the conflict order, and the spread is smaller than the run-to-run variation of
  the machine, so it is not evidence of anything.

**That is the exercise.** You have now computed, by hand and in code, four
quantities that Modules 5 and 7 spent two lessons teaching you to compute, on a
kernel somebody would ship — and exactly one of them predicted the measurement.
The other three were correct arithmetic about a resource that was not the
bottleneck. Counting is necessary and it is not sufficient: the number you
compute has to be attached to the resource that is saturated, and finding out
which one that is takes a measurement (or a floor calculation, M11).

Run the same seven kernels on a matrix that fits in L2 and the shared-memory
columns start predicting, while the sector columns stop.

---

## Expected output

```
Module 15 exercise 03 -- count it before you run it
GPU: NVIDIA RTX 3500 Ada Generation Laptop GPU, CC 8.9, L2 = 50.3 MB

Structural tests of your two counting procedures: 9/9

configuration                      your table your procedures
                                  rd wr st ld    rd wr st ld
1 naive coal rd / strided wr      4 32  0  0     4 32  0  0
2 naive strided rd / coal wr     32  4  0  0    32  4  0  0
3 tile[32][32]                    4  4  1 32     4  4  1 32
4 tile[32][33]                    4  4  1  1     4  4  1  1
5 tile[32][34]                    4  4  1  2     4  4  1  2
6 tile[16][17], block(16,16)      4  4  2  2     4  4  2  2
7 32x32 XOR swizzle               4  4  1  1     4  4  1  1

  your hand table  : CORRECT
  your procedures  : CORRECT

TODO 4 -- smallest conflict-free pitch
  tileW  32,  4 B elements : you say   33  ok
  tileW  32,  8 B elements : you say   33  ok
  tileW  32, 16 B elements : you say   33  ok
  tileW  64,  4 B elements : you say   65  ok
  tileW  64,  8 B elements : you say   65  ok
  tileW  64, 16 B elements : you say   65  ok
  tileW  96,  4 B elements : you say   97  ok
  tileW  96,  8 B elements : you say   97  ok
  tileW  96, 16 B elements : you say   97  ok
  tileW 128,  4 B elements : you say  129  ok
  tileW 128,  8 B elements : you say  129  ok
  tileW 128, 16 B elements : you say  129  ok
  TODO 4 score: 12/12

MEASURED at 8192 x 8192 (min of 8 rotated sweeps)
configuration                         ms      GB/s   %ofcopy
0 copy ceiling                    1.5435     347.8    100.0%
1 naive coal rd / strided wr      3.2491     165.2     47.5%
2 naive strided rd / coal wr      1.6092     333.6     95.9%
3 tile[32][32]                    1.6377     327.8     94.3%
4 tile[32][33]                    1.5812     339.5     97.6%
5 tile[32][34]                    1.5932     337.0     96.9%
6 tile[16][17], block(16,16)      1.6224     330.9     95.1%
7 32x32 XOR swizzle               1.6215     331.1     95.2%

VALIDATION: all seven kernels correct

CONFIG 3 / CONFIG 4 -- what removing a 32-way conflict is worth
  at 8192x8192 (DRAM bound)    : 1.036x  -> bucket 1
  at 2048x2048 (L2 resident)   : 2.756x  -> bucket 3

SCORING
  structural tests            : 1/1
  TODO 1 + TODO 2 procedures  : 1/1
  TODO 3 hand table (28 cells): 1/1 (7/7 rows agree with your code)
  TODO 4 padding rule         : 1/1
  TODO 5 DRAM ratio bucket    : 1/1
  TODO 5 L2 ratio bucket      : 1/1

SCORE: 6/6
OVERALL: PASS
```

**Variance.** Absolute ms move by up to 2.5× on this laptop part with thermal and
power state. The two scored ratios were stable across every authoring run:
0.96–1.04× at DRAM scale and 2.67–3.25× at L2 scale. Config 2's 95.9% is the
least stable row (88–96% observed) because it is the one whose performance
depends on how aggressively the compiler's load hoisting is rewarded by the
current memory-system state.

---

## The result that matters

You can now compute, before compiling, the four numbers that Part II taught you:
sectors per warp on each side of a global access, and the bank-conflict degree of
each side of a shared access. On this kernel exactly one of those four numbers
predicts the measurement, and the other three are correct arithmetic about a
resource with slack. The skill Part IV needs is not the counting — that is
mechanical — it is knowing which count to believe, which means knowing which
resource is saturated before you start optimizing. A kernel at 95% of a measured
copy has 5% left no matter how bad its bank conflicts look on paper.

**Variation to try.** Add an eighth configuration: `tile[32][40]`, the pitch that
"looks aligned". Predict its two degrees first (`gcd(40,32) = 8`, so the column
walk is `D = 8`, four times worse than config 5's and one quarter of config 3's),
then measure it at 8192 × 8192 and at 2048 × 2048. The DRAM-scale number will be
indistinguishable from configs 3–7; the L2-scale number will land between them,
and where exactly it lands is the most direct measurement of `max(2, D)` that
this module contains.
