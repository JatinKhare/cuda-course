# Module 13 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o ex2sol.exe exercise02_solution.cu
ex2sol.exe
```

Warning-clean, CUDA 13.2, sm_89. About 20 s, most of it in the three host-side
validation passes over 12.6 M survivors.

---

## TODO 1 — the predicate kernel

```cpp
__global__ void predicateKernel(const u32 * __restrict__ data, u32 * __restrict__ flags, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) flags[i] = keepPred(data[i]);
}
```

The launch is `<<<240, 256>>>` — a wave-sized grid, 40 SMs × 6 blocks/SM (M1,
M3) — not one thread per element. 240 × 256 = 61,440 threads for 33,554,393
elements, so a grid-stride loop is mandatory; writing only
`if (i < n) flags[i] = ...` leaves 99.8 % of the flags uninitialised. The symptom
is spectacular and immediate (the count comes out near zero or wildly wrong), so
this one diagnoses itself.

The predicate is `(v & 7) < 3`, which survives 3 of 8 values, and the harness
reports 12,582,896 of 33,554,393 = 37.50 %.

## TODO 2 — the device-wide exclusive scan (DESIGN)

```cpp
static void deviceExclusiveScan(const u32 *d_in, u32 *d_out, u32 *d_sums, int n)
{
    const int m = (n + TILE - 1) / TILE;          // 32,768 tiles
    scanTilesKernel<<<m, BLK>>>(d_in, d_out, d_sums, n);   // read N, write N
    scanSumsKernel<<<1, BLK>>>(d_sums, m);                 // 128 KB, negligible
    addOffsetsKernel<<<m, BLK>>>(d_out, d_sums, n);        // read N, write N
}
```

**Traffic: 4N.** That is the number the TODO asks you to write in a comment, and
the reason it asks is that the whole exercise turns on it.

The two things that go wrong here:

- **Assuming `m` fits in one tile.** m = 32,768 and TILE = 1024. A single-block
  pass 2 that scans only the first 1024 tile totals leaves tiles 1024..32,767
  with an offset of whatever was in `d_sums` before, so the first 1 M elements
  are right and the remaining 32 M are wrong. `scanSumsKernel` as supplied loops
  over its input with a running `carry` precisely so this works; a reader who
  writes their own pass 2 must reproduce that.
- **Launching pass 3 with a grid-stride grid.** `addOffsetsKernel` indexes
  `offs[blockIdx.x]`, i.e. it assumes one block per tile. Launching it with 240
  blocks silently applies tile 0..239's offsets to the first 240 tiles and
  nothing to the rest.

`scanSumsKernel` costs 32,768 words read and written twice — 256 KB against
268 MB — under 0.1 % of the traffic. Do not optimise it.

### The fused variant (third configuration)

The exercise ships a second scan-based configuration; the solution fills it in
because it is what the design freedom in TODO 2 is actually worth:

```cpp
// pass 1: apply the predicate AS the tile is loaded — flags[] never exists
for (int i = tid; i < TILE; i += BLK) s[i] = (base+i < n) ? keepPred(data[base+i]) : 0u;
...
// pass 3: add the block offset and scatter in the same kernel
if (i < n && keepPred(data[i])) out[offs[i] + o] = (u32)i;
```

Traffic drops from roughly 8N (write flags, read flags, scan 4N, scatter reads
flags and offsets and writes survivors) to roughly 4.4N. Measured 1.60 ms against
the unfused 3.27 ms — **2.04× from fusion alone, with the identical scan
algorithm underneath.** The predicate is recomputed instead of stored, which
costs two integer instructions per element and saves two DRAM round trips per
element. That trade is almost always right on this hardware.

## TODO 3 — the scatter

```cpp
for (; i < n; i += stride) if (flags[i]) out[offsets[i]] = (u32)i;
```

`offsets[i]` is the **exclusive** scan of the flags, which is by definition the
number of survivors strictly before `i` — i.e. exactly the slot this survivor
should occupy. The two plausible-but-wrong alternatives:

| wrong version | what happens |
|---|---|
| `out[inclusive[i]] = i` (using an inclusive scan) | every survivor lands one slot too high; `out[0]` is never written and the last survivor writes at index `count`, one past the end of the logical output. Still dense-looking, still a permutation of a *shifted* set, and it is an out-of-bounds write when the output buffer is sized exactly `count`. Here the buffer is size `n` so it does not fault — which is why `memcheck` is silent and the validation has to be exact rather than "looks reasonable". |
| `out[offsets[i] - 1] = i` | reads `out[-1]` for the first survivor. |
| dropping the `if (flags[i])` | every element writes, non-survivors overwrite their successors' slots; the output is a mess but the *count* is still right. |

**Write the index, not the value.** Compacting indices rather than payloads is
what makes the validation exact and cheap: the reference is the sorted list of
surviving indices, so "is it a permutation" is a bitmap check and "is it in
order" is a single scan for a descent. It is also what a real `select` returns,
because the caller usually wants to gather several payload arrays with the same
index list.

## TODO 4 — the count

```cpp
u32 lastOff, lastSrc;
cudaMemcpy(&lastOff, d_offs + (n-1), 4, cudaMemcpyDeviceToHost);
cudaMemcpy(&lastSrc, d_data + (n-1), 4, cudaMemcpyDeviceToHost);
cnt = lastOff + keepPred(lastSrc);
```

The exclusive scan's last entry counts survivors **strictly before** index n−1.
The last element's own flag has to be added back. This is §2 of the lesson, and
it is data-dependent in the nastiest way: **if element n−1 happens to fail the
predicate, `cnt = lastOff` is correct.** With this seed and predicate,
`data[33554392]` survives, so the naive version reports 12,582,895 instead of
12,582,896 and silently truncates the last survivor. Change the seed and the bug
disappears.

Alternatives that are also correct: read `flags[n-1]` instead of recomputing the
predicate (one fewer branch, one more array that has to exist — the fused version
cannot do it); or have pass 2 write the grand total to a dedicated device word
and copy that.

## TODO 5 — the prediction

`PREDICT_BUCKET 1` — atomics at least 3× faster. Measured **4.75×**.

The arithmetic you were asked to do:

| version | words read | words written | total |
|---|---|---|---|
| atomic ticket | N | 0.375 N | **1.375 N** |
| flags + 3-kernel scan + scatter | N (predicate) + N (scan p1) + N (scan p3) + N (scatter flags) + N (scatter offs) | N (flags) + N (scan p1) + N (scan p3) + 0.375 N | **~8.4 N** |
| fused | N + N + N | N + 0.375 N | **~4.4 N** |

8.4 / 1.375 = 6.1, and the measured ratio is 4.75 — better than the ledger
predicts, because the atomic version's 12.6 M scattered single-word stores are
less efficient per byte than the scan version's fully coalesced streams. The
fused version's ledger predicts 3.2 and measures 2.33, same reason.

---

## Synchronization / memory reasoning

Nothing in the scan path needs a fence: the three kernels are separated by
kernel boundaries, which M9 established are the only device-wide ordering
guarantee that holds for an arbitrary grid size. That is the entire reason the
three-kernel structure exists and the entire reason it costs 4N.

The atomic path needs no ordering either, only atomicity: `atomicAdd` returns the
old value and that value is unique across the device by construction (M10). Note
what it does *not* give you — any relationship between the returned slot and the
element's index. The compiler emits warp-aggregated `ATOMG` here. Confirmed in the SASS with
`nvcc -arch=sm_89 -O3 -cubin` + `cuobjdump -sass`:

```
VOTEU.ANY   UR6, UPT, PT ;
FLO.U32     R6, UR6 ;
POPC        R7, UR6 ;
@P0 ATOMG.E.ADD.STRONG.GPU PT, R3, [R2.64], R7 ;
POPC        R5, R8 ;
SHFL.IDX    PT, R4, R3, R6, 0x1f ;
```

This is M10's compiler-aggregation idiom exactly: one lane issues a single
atomic adding the warp's whole population count, the result is broadcast with
`SHFL.IDX`, and each lane derives its own slot from a `POPC` of the lanes below
it. So 32 lanes hitting the same counter cost roughly one atomic plus a
one-instruction warp scan.
**The fast unordered compaction is itself doing a one-warp-wide scan inside the
hardware.** That is worth sitting with: the difference between the two versions
is not scan-versus-no-scan, it is the *scope* over which the scan is done.

---

## Performance reasoning

```
configuration                       ms     count     perm   ascend    exact    GB/s*
atomic ticket (unordered)       0.6878  12582896      yes       no       no    268.3
scan: flags+scan+scatter        3.2688  12582896      yes      yes      yes     56.5
scan: fused (solution only)     1.6006  12582896      yes      yes      yes    115.3
```

The `GB/s*` column deliberately counts only the *unavoidable* traffic — read N,
write the survivors, 0.4272 ms at 432 GB/s — so it is an efficiency score
against a common floor rather than a bandwidth claim about what each version
moves. The atomic version reaches 62 % of that floor; the naive scan 13 %; the
fused scan 27 %.

**The result the exercise exists to produce:** two consecutive runs of the
atomic kernel, on identical input, with no changes of any kind, differ in
**12,582,783 of 12,582,896 positions**. Not a few. Essentially all of them. The
output is a *correct compaction* every time and a *different* correct compaction
every time. If anything downstream — a checksum, a reduction with floating-point
payloads, a regression test, a bisect — depends on the order, that pipeline is
not reproducible and the cause will be very hard to find, because the kernel
itself is not wrong.

---

## Expected output

Actual run, RTX 3500 Ada, CUDA 13.2:

```
Module 13 / Exercise 2 — ordered stream compaction
N = 33554393 (32768 tiles, last tile holds 985)

survivors: 12582896 of 33554393 (37.50%)

configuration                       ms     count     perm   ascend    exact    GB/s*
atomic ticket (unordered)       0.6878  12582896      yes       no       no    268.3
scan: flags+scan+scatter        3.2688  12582896      yes      yes      yes     56.5
scan: fused (solution only)     1.6006  12582896      yes      yes      yes    115.3

  *GB/s counts only the unavoidable traffic (read N + write survivors),
   so it is an efficiency score against the 0.4272 ms floor at 432 GB/s,
   not a claim about what each version actually moves.
  atomic run A vs atomic run B: 12582783 of 12582896 positions differ

  scan / atomic time ratio: 4.75x -> bucket 1   predicted 1
  fused scan / atomic     : 2.33x  (what the design TODO is worth)

  score: 7/7
OVERALL: PASS
```

Run-to-run: the atomic version has been observed between 0.688 ms and 0.820 ms
and the unfused scan between 3.269 ms and 3.279 ms; the ratio has measured
**3.80–4.75x** across five runs, safely inside bucket 1 (>= 3x), and the fused
version 1.95–2.44x. The number of differing positions between two atomic runs
has been 12,582,711 / 12,582,783 / 12,582,810 / 12,582,830 / 12,582,848 — always
within 0.002 % of "all of them".

---

## The result that matters

**Ordering is not free, and its price is bandwidth, not atomics.** The atomic
ticket is 4.75× faster than the straightforward scan-based compaction not because
atomics are fast (they are, but that is M10's point) but because the scan-based
version touches the array about six times more. Fusing the predicate into the
scan and the scatter into the propagate pass recovers half of that with no change
to the algorithm, and doing the scan with decoupled look-back instead of three
kernels (Exercise 3) would bring it to within roughly 1.4× — which is where
`cub::DeviceSelect::If` lives, and is why nobody ships the atomic version when
order matters.

**Variation to try:** change the predicate's selectivity from 37.5 % to 1 % and
to 99 %, and re-measure both. The scan version's cost barely moves — it is
proportional to N, not to the number of survivors. The atomic version's cost
moves a lot, because its contention (M10: cost tracks addresses-per-warp) and its
scatter traffic both scale with the survivor count. Work out whether there is a
selectivity at which the gap closes, and predict the direction before measuring.
