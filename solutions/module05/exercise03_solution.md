# Module 5 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
```

## The diagnosis you were asked to make

The kernel's indexing is already the textbook-correct one: 32 consecutive lanes
read 32 consecutive floats. There is no stride in the source. So the usual
checklist — "is the access contiguous?" — passes, and the kernel is still slow.

The thing the checklist misses is that contiguity is necessary but not
sufficient. What the coalescer needs is a 128 B window that is **32 B aligned**.
Row `r` begins at byte `r * pitch * 4` from a 256 B-aligned base, and with
`pitch = 65` that is `260r`. Since `gcd(260, 32) = 4`, the value `260r mod 32`
cycles through `{0, 4, 8, 12, 16, 20, 24, 28}` — zero for one row in eight.

- 1 row in 8: window is 32 B aligned → **4 sectors**.
- 7 rows in 8: window straddles → **5 sectors**.

Mean = `(1*4 + 7*5)/8 = 4.875` sectors for 128 useful bytes → **82.1 %**.

The layout, not the loop, is the defect. The kernel cannot be written any other
way; there is no "more coalesced" mapping of 32 lanes onto 32 adjacent floats.

### Why this case is different from Exercise 1's misalignment

In Exercise 1 the misaligned kernel measured 98 % of the aligned one despite an
80 % model, because warp *k*'s extra boundary sector was also warp *k+1*'s
first sector, and the second request hit in cache. Here that rescue is
unavailable: the warp reads columns 0–31 and then the next warp starts 260 B
later, so the boundary sector at the end of row *r*'s window contains columns
32–33 of row *r* — **bytes nobody reads at all**. There is no neighbour to
share the cost with, so the wasted sector is genuinely fetched from DRAM and
genuinely discarded.

The general rule, and it is the one worth carrying forward: *misalignment costs
in proportion to the boundary-to-interior ratio of the contiguous runs.* Long
runs in a streaming sweep hide it. Short runs separated by gaps expose it.

This is precisely the situation `cudaMallocPitch` exists for.

## TODO 1 — sectors per row

```cpp
static int sectors_for_row(uintptr_t base, int pitch, long long row)
{
    uintptr_t a0 = base + (uintptr_t)row * (uintptr_t)pitch * sizeof(float);
    uintptr_t a1 = a0 + (uintptr_t)COLS_USED * sizeof(float) - 1;
    return (int)(a1 / 32 - a0 / 32 + 1);
}
```

First byte, last byte, sector ids of each, difference plus one. No branch, no
special case for "aligned".

Common wrong approaches:

- `COLS_USED * 4 / 32` (a constant 4). Assumes alignment and so predicts 100 %
  for every pitch — i.e. predicts that this exercise has no content.
- `(a0 % 32 == 0) ? 4 : 5`. Correct *here*, because the window happens to be
  exactly 128 B. Change `COLS_USED` to 33 and it is wrong. Derive from the
  bytes.
- Using `a1 = a0 + COLS_USED * 4` without the `- 1`. Off by one whenever the
  window ends exactly on a sector boundary: reports 5 sectors for the aligned
  case instead of 4, and therefore reports 100 % as unachievable.

## TODO 2 — the three thresholds

Row starts are `pitch * 4` bytes apart. Given a base that is already 256 B
aligned, every row start is `A`-byte aligned iff `(pitch * 4) % A == 0`, i.e.
`pitch % (A/4) == 0`. Also `pitch >= COLS_TOTAL = 65`.

| Boundary | Why you would want it | Condition | Smallest ≥ 65 |
|---|---|---|---|
| 16 B | legality of `float4` / `LDG.E.128` | `pitch % 4 == 0` | **68** |
| 32 B | one sector — what coalescing needs | `pitch % 8 == 0` | **72** |
| 128 B | one full L1/L2 line | `pitch % 32 == 0` | **96** |

```cpp
static const int MIN_PITCH_16B  = 68;
static const int MIN_PITCH_32B  = 72;
static const int MIN_PITCH_128B = 96;
```

**68 is the trap.** "Pad to a multiple of 4 so vector loads work" is the rule
most people carry around, and it is a true statement about *legality*. It does
nothing for *efficiency*: `68 * 4 = 272`, and `272 mod 32 = 16`, so row starts
alternate between offset 0 and offset 16. Half the rows still cost 5 sectors.
Mean 4.5, efficiency 88.9 % — an improvement over 82.1 %, which makes it
especially seductive, and still 11 % short.

Measured: pitch 68 reaches 91.5 % of the pitch-96 bandwidth. It looks like a
fix. It is two thirds of one.

Note that 65 itself is already legal to *allocate* and legal to *access
scalarly*; nothing about the given code is wrong. The exercise is not a bug
hunt.

## TODO 3 — the kernel

```cpp
__global__ void row_window_scale(float* __restrict__ a, long long rows, int pitch)
{
    long long gid  = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long row  = gid >> 5;
    int       lane = (int)(gid & 31);
    if (row >= rows) return;
    long long i = row * (long long)pitch + lane;
    a[i] = a[i] * 2.0f + 1.0f;
}
```

`gid >> 5` and `gid & 31` rather than `/32` and `%32`: identical for unsigned
quantities and clearer about intent — warp *w* of the grid handles row *w*,
lane *L* handles column *L*. Because `blockDim.x = 256` is a multiple of 32, a
warp never spans two rows, so there is no divergence at all inside a warp.

Mistakes to avoid:

- `long long i = row * pitch + lane;` with `pitch` an `int` and `row` promoted
  correctly — fine. But `int i = row * pitch + lane;` overflows at
  1,000,000 × 96 = 96 M (still fits) — and would not at larger sizes. Keep the
  index 64-bit; this is the habit Module 3 established.
- Writing to column `lane` without the `row >= rows` guard. The grid is
  `ceil(rows*32/256)` blocks, which is exact here, but the guard costs nothing
  and the array is not padded at the end.
- Letting a thread touch columns ≥ 32. At pitch 65 that corrupts the label
  column and, at column 65+, the next row's data — and the validator is
  checking columns 0–31 of every sampled row, so you would see it.

## TODO 4 — the pitch to ship

```cpp
static const int PITCH_CHOSEN = 96;
```

The reasoning, before measuring: 72 and 96 have identical predicted efficiency
(4.000 sectors, 100 %). 72 wastes `(72-65)/72 = 9.7 %` of the allocation; 96
wastes `31/96 = 32 %`. On pure sector arithmetic plus memory cost, **72 is the
better engineering choice** and that is a defensible answer.

The measurement says otherwise, and the gap is the interesting part.

## Performance reasoning

Measured on the RTX 3500 Ada Laptop GPU, 1,000,000 rows, 488 MB allocation,
122 MB touched footprint, three passes with the last reported:

```
=== Part A: predicted, from your sector count ===
  pitch             floats    rowstride B   mean sectors   efficiency
  65 (as given)         65            260          4.875        82.1%
  MIN_PITCH_16B         68            272          4.500        88.9%
  MIN_PITCH_32B         72            288          4.000       100.0%
  MIN_PITCH_128B        96            384          4.000       100.0%
  PITCH_CHOSEN          96            384          4.000       100.0%

=== Part B: measured (3 passes, last reported) ===
  pitch             floats        ms       GB/s   %ofpeak       model   validation
  65 (as given)         65     0.806      317.6     73.5%       82.1%   PASS (0)
  MIN_PITCH_16B         68     0.732      349.5     80.9%       88.9%   PASS (0)
  MIN_PITCH_32B         72     0.714      358.7     83.0%      100.0%   PASS (0)
  MIN_PITCH_128B        96     0.677      378.0     87.5%      100.0%   PASS (0)
  PITCH_CHOSEN          96     0.681      376.1     87.0%      100.0%   PASS (0)

=== Predicted vs measured, normalised to MIN_PITCH_128B ===
  pitch               model ratio measured ratio
  65 (as given)             0.821          0.840
  MIN_PITCH_16B             0.889          0.925
  MIN_PITCH_32B             1.000          0.949
  MIN_PITCH_128B            1.000          1.000
  PITCH_CHOSEN              1.000          0.995
```

The ordering is exactly as predicted and the first three ratios track the
sector model to within 2–4 %: 0.821→0.840, 0.889→0.925. Changing nothing but
a padding constant bought **19 %**.

**The discrepancy: pitch 72 measures 0.949, not 1.000.** Both 72 and 96 request
exactly 4 sectors per warp. The sector model says they must be identical. They
are not, by 5–6 %, reproducibly across runs.

What the sector model does not capture:

- **Cache-line and tag granularity.** At pitch 96 the row stride is 384 B = 3
  full 128 B lines, so every row's 128 B window is *exactly one line*, and it
  is line-aligned. At pitch 72 the stride is 288 B, which is not a multiple of
  128; a row's 4 sectors therefore straddle two different 128 B lines for three
  rows out of four (the start offset within a line cycles 0, 32, 64, 96). The sectors fetched are the same 4, but they are tagged
  against twice as many lines, doubling tag-array pressure and halving the
  useful residency of L2.
- **DRAM page and channel locality.** A 384 B stride keeps a DRAM page's worth
  of consecutive rows landing in a predictable channel/bank pattern; 288 B does
  not divide the interleaving granularity as cleanly.

Neither effect is visible at the 32 B sector level, which is exactly the point:
the sector model is the *first* thing to check and it explains the 82 → 89 →
100 progression completely. The last 6 % lives one level down, and Module 23
gives you the metrics (`lts__t_sectors`, `dram__sectors`) to separate them.

**So which do you ship?** 96, if the array fits — 6 % of a memory-bound kernel
for 22 % more memory is usually the right trade in a hot loop, and 96 is also
the pitch `cudaMallocPitch` would hand you on this hardware (it pads to the
texture-alignment granularity, typically 512 B, which subsumes 128 B). 72, if
the array is large enough that 32 % overhead threatens residency, because a
layout that no longer fits in memory has an efficiency of zero. What you must
*not* ship is 68, and the reason you must not is the entire exercise.

## Expected output

Part A is deterministic; reproduce the table above exactly. Part B's absolute
GB/s vary with the memory P-state (318 GB/s for pitch 65 was measured at an
8001 MHz memory clock; at 9001 MHz expect ~355). The **normalised ratios** are
what to check, and the first pass of the three is discarded specifically so
that the row measured first is not penalised by a cold clock.

If all five rows measure within noise of each other, your `sectors_for_row` is
probably returning a constant — check Part A first.

## The result that matters

The kernel was already perfectly written. The slow part was a number chosen by
whoever defined the record format, and the fix was to change that number. Most
real coalescing problems look like this: not a stride you can see in the source,
but an alignment you inherited. Before rewriting a loop, compute where its rows
start.

Try this variation: set `COLS_USED` to 8 instead of 32, so each warp's window
is 32 B rather than 128 B, and re-derive the predicted efficiencies. With a
one-sector window, a misaligned start costs *two* sectors instead of one —
100 % overhead rather than 25 %. Then set `COLS_USED` to 64 and watch the
penalty shrink to 12.5 %. The boundary-to-interior ratio is the whole story,
and you can now compute it.
