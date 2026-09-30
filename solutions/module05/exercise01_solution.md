# Module 5 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

## TODO 1 — address to sector id

```cpp
static unsigned long long sector_of(uintptr_t addr)
{
    return (unsigned long long)(addr / SECTOR_B);   // == addr >> 5
}
```

Sectors are *naturally aligned*: sector `k` covers `[32k, 32k+32)`. So the
sector id is an integer division, not a subtraction from some base. Two
addresses share a sector exactly when their top 59 bits match.

Common wrong approaches:

- `(addr - base) / 32`. This measures the offset's sector, not the address's.
  It gives the right answer here only because `cudaMalloc` happens to return a
  256 B-aligned pointer. Feed it a pointer offset by 4 B and it will report
  everything perfectly aligned, hiding the whole phenomenon the exercise is
  about. Sector alignment is a property of the *absolute* address.
- `addr / 128` (using the line, not the sector). Predicts 1 sector for a
  coalesced warp instead of 4, and 25 % efficiency for stride-4 instead of the
  correct 25 %… by accident. It then predicts 12.5 % for stride-8 *and* for
  stride-32 *and* for stride-64, which is wrong for stride-64.

## TODO 2 — distinct sector count

```cpp
for (int lane = 0; lane < WARP; ++lane) {
    uintptr_t a = base + (uintptr_t)(elem_index(p, lane, param) * sizeof(float));
    unsigned long long s = sector_of(a);
    bool dup = false;
    for (int k = 0; k < n; ++k) if (seen[k] == s) { dup = true; break; }
    if (!dup) { if (n < 64) seen[n] = s; ++n; }
}
```

The structure that matters is the membership test. A version that counts a
"new sector" whenever `s != previous_s` gives the right answer for contiguous
and for stride patterns and the *wrong* answer for anything where lanes revisit
a sector out of order — including the reversed pattern, which is exactly the
case the module uses to make its central point. The coalescer receives 32
addresses in one cycle and computes a set union; the only faithful model is a
set.

Why 32 lanes and not `blockDim.x`: coalescing happens per warp. A 256-thread
block issues eight independent coalescing decisions per instruction. Summing
over a whole block would give a number the hardware never computes.

## TODO 3 — efficiency

```cpp
return (double)requestedBytes / (double)(SECTOR_B * distinctSectors);
```

Requested over moved. Not moved over requested (that is the *waste factor*,
≥ 1), and not `1 - moved/requested` (meaningless). Sanity check: contiguous
gives `128 / (32*4) = 1.0`; stride-32 gives `128 / (32*32) = 0.125`.

The program asserts `efficiency * 32 * S == requested`, which catches an
inverted formula immediately.

## TODO 4 — the alignment probe

```cpp
const int k_align = 4;
```

A 128-bit access compiles to `LDG.E.128` / `STG.E.128`, and the hardware
requires the address to be **16 B aligned**. `cudaMalloc` guarantees the base
is at least 256 B aligned, so `d + k` (which is `base + 4k` bytes) is 16 B
aligned iff `4k % 16 == 0`, iff `k % 4 == 0`. The smallest strictly positive
such `k` is **4**.

Wrong answers and their exact symptoms:

| `k_align` | What happens |
|---|---|
| `1`, `2`, `3` | Launch succeeds; `cudaDeviceSynchronize()` returns `cudaErrorMisalignedAddress`. The program prints that name and stops. The context is now poisoned — every subsequent CUDA call in the process fails. |
| `8` | Works, but is not the smallest. |
| `0` | Would work, but the TODO says *strictly positive* — and the point is to find the granularity, not to dodge it. |

The observed message:

```
  float4 load at +4 floats succeeded -> address is 16 B aligned.
```

Note carefully what `k = 4` does **not** buy you. `base + 16 B` is 16 B aligned
but *not* 32 B aligned, so a warp of scalar loads starting there still costs 5
sectors instead of 4. Legality and efficiency are two different thresholds:
16 B for the instruction, 32 B for the sector. Exercise 3 turns that distinction
into a trap.

## Memory reasoning — the paper table

| Pattern | Byte offsets (warp 0) | Sectors | Moved | Efficiency |
|---|---|---|---|---|
| `a[t]` | 0, 4, …, 124 | 4 | 128 | 100 % |
| `a[t+1]` | 4, 8, …, 128 | 5 | 160 | 80 % |
| `a[t+8]` | 32, 36, …, 156 | 4 | 128 | 100 % |
| `a[2t]` | 0, 8, …, 248 | 8 | 256 | 50 % |
| `a[8t]` | 0, 32, …, 992 | 32 | 1024 | 12.5 % |
| `a[32t]` | 0, 128, …, 3968 | 32 | 1024 | 12.5 % |
| `a[6t]` | 0, 24, …, 744 | 24 | 768 | 16.7 % |

Two rows worth dwelling on.

*`a[8t]` and `a[32t]` are both 12.5 %.* Once the stride reaches 8 floats
(32 B), every lane already owns a private sector, and increasing the stride
further cannot make it worse. 32 sectors is the ceiling: one per lane. This is
why "stride 1000" is not a thousand times worse than "stride 8" — the sector
model saturates. (It does keep getting worse in *other* currencies: TLB
coverage and DRAM page locality, visible in the measured stride-32 row below.)

*`a[6t]` is 16.7 %, not 12.5 %.* A 24 B stride means some pairs of lanes share
a sector: lanes 0 and 1 hit bytes 0 and 24, both in sector 0. Four lanes out of
every three sectors double up, giving 24 sectors rather than 32. This is the
AoS case, and the exact figure `128/768 = 1/6` is the "24 B of struct, 4 B
used" ratio, as it must be.

## Performance reasoning

Measured on the RTX 3500 Ada Laptop GPU, 256 MB buffer, 25 timed iterations
after a 500-launch warm-up:

```
=== PART B: measured (25 iters, 256 MB buffer) ===
  pattern                       ms    effGB/s   %ofpeak    modelEff  impliedDRAM
  contiguous                 1.569      342.1     79.2%      100.0%        342.1
  offset k=1 float           1.603      334.9     77.5%       80.0%        418.7
  offset k=8 floats          1.563      343.6     79.5%      100.0%        343.6
  stride s=2                 1.562      171.8     39.8%       50.0%        343.7
  stride s=8                 1.577       42.6      9.9%       12.5%        340.4
  stride s=32                0.412       40.7      9.4%       12.5%        325.5
  AoS .x (stride 6)          1.572       56.9     13.2%       16.7%        341.6

  kernel numerics: PASS (0 mismatches)

=== Predicted vs measured, normalised to 'contiguous' ===
  pattern                   model ratio measured ratio      err
  contiguous                      1.000          1.000     0.0%
  offset k=1 float                0.800          0.979    22.4%
  offset k=8 floats               1.000          1.004     0.4%
  stride s=2                      0.500          0.502     0.5%
  stride s=8                      0.125          0.124    -0.5%
  stride s=32                     0.125          0.119    -4.9%
  AoS .x (stride 6)               0.167          0.166    -0.1%
  patterns where the sector model mispredicts by >15%: 1
```

Absolute GB/s varies run to run between roughly **250 and 380 GB/s** for the
contiguous case, depending on the memory P-state and whether the power cap has
engaged. The **ratios** are stable to about ±2 % once warm, which is why the
exercise judges you on ratios. If your contiguous number is below ~290 GB/s,
run `nvidia-smi --query-gpu=clocks.mem,clocks_throttle_reasons.active
--format=csv`: `7001`/`8001` MHz instead of `9001` caps you at 336/384 GB/s
rather than 432, and `0x4` means a software power cap.

**Look at the first column of times.** Five of the seven kernels take ~1.57 ms
despite processing between 1/32 and 1 of the elements. The stride-8 kernel
launches one eighth as many threads as the contiguous one and takes *the same
wall-clock time*. That is the entire module in one observation: the runtime is
set by sectors moved, and all five of those kernels move 512 MB of sectors.

The `impliedDRAM` column confirms it: 340–344 GB/s for contiguous, stride-2,
stride-8, and AoS alike. The bus is equally saturated in all of them. No amount
of extra occupancy or ILP will help the strided kernels, because there is no
idle resource to fill.

**Stride 32 measures 325 implied DRAM, not 342.** With a 128 B stride the warp
spans 4 KB, the grid sweeps 256 MB through the TLB in 4 KB hops, and DRAM row
locality degrades. Below the sector model there is a second, coarser locality
effect you will meet again in Module 15.

### The one that disagrees: `offset k=1`

Predicted ratio 0.800, measured 0.979 — off by +22 %, and the only row flagged.
The implied DRAM figure is **418.7 GB/s**, higher than the 342 GB/s the
contiguous kernel actually achieved. A derived quantity exceeding a measured
ceiling is a proof that the model over-counts.

The mechanism: warp *k* reads bytes `[128k+4, 128k+132)`, which touches sectors
`4k … 4k+4`. Warp *k+1* reads `[128k+132, 128k+260)`, touching `4k+4 … 4k+8`.
**Sector `4k+4` is requested by both.** In a streaming kernel those warps run
within microseconds of each other, so the second request hits in L1 or L2. DRAM
delivers 128 B per warp, exactly as in the aligned case; only the on-chip
request count rose from 4 to 5 per warp.

So the sector count correctly predicts **L1 request pressure** and
over-predicts **DRAM traffic** whenever neighbouring warps overlap. Whether
that matters depends on which resource is binding. Here DRAM binds, so
misalignment is nearly free.

The natural follow-up — *when is misalignment actually expensive?* — is
Exercise 3. The answer is: when the contiguous runs are short and separated by
gaps, so that no neighbour exists to absorb the boundary sector.

## Expected output

Part A is deterministic and will match the table above exactly. Part B's
absolute numbers will differ; the `err` column should show one flagged row
(`offset k=1`, positive error) and everything else within a few percent. If
several rows are flagged, your GPU had not settled — run it again.

## The result that matters

You can now compute a kernel's DRAM traffic before writing it: enumerate one
warp's addresses, divide by 32, count distinct values. That number, times 32,
divided by 432 GB/s, is a hard lower bound on the kernel's runtime. Everything
in Parts IV–X of this course is an attempt to reduce that count.

And you have seen the model's edge: it counts *requests*, which equal DRAM
traffic only when warps do not share sectors. Try it — change `k_pattern`'s
offset case to use a *per-block* offset (`i = t + blockIdx.x`) so that each
block starts at a different misalignment, and watch whether the 80 % prediction
gets closer or further from the measurement. Then predict what happens if you
run the offset kernel on a buffer small enough to sit in L2.
