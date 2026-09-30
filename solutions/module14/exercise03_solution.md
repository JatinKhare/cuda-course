# Module 14 / Exercise 3 — Solution notes

**Do not read this until you have submitted your own attempt.**

---

## Compile / run

```
nvcc -arch=sm_89 -O3 -lineinfo -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
```

Three defects in a kernel that packs two 16-bit bin counters into every 32-bit
shared word. Two produce symptoms. The third produces none on this GPU in this
configuration, which is the point.

---

## The three defects

```cpp
__global__ void hist_broken(const unsigned int* __restrict__ in, size_t n,
                            unsigned int* __restrict__ hist, int nBins)
{
    extern __shared__ unsigned int s[];
    const int nWords = nBins >> 1;

    if ((int)threadIdx.x < nWords) s[threadIdx.x] = 0u;      // DEFECT A
                                                             // DEFECT B: no barrier
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = ...; i < n; i += stride) {
        unsigned int b = binOf(in[i], nBins);
        atomicAdd(&s[b >> 1], (b & 1u) ? 0x10000u : 1u);     // DEFECT C
    }
    __syncthreads();
    ...
}
```

**A — the guarded zeroing initializes only `min(blockDim.x, nWords)` words.**
At `nBins = 256`, `nWords = 128 ≤ 256`, so every word is zeroed and the defect
is invisible. At `nBins = 1024`, `nWords = 512 > 256`, so words 256–511 hold
whatever the previous resident block left there. This is why the symptom is a
function of `nBins` and not of the input. The harness runs a `dirty_shared`
kernel first specifically so that "whatever was left there" is
`0xDEADBEEF` rather than a helpfully zeroed SM — without it the defect hides on
a freshly booted context and you would call it flaky.

**B — no barrier between the zeroing and the first `atomicAdd`.** `__syncthreads()`
provides two guarantees (Module 9) and the code needs both: the execution
barrier so that warp 7 does not add before warp 0 has zeroed, and the block-scope
memory fence so that warp 0's store is visible when warp 7 reads it. It is a
genuine data race with undefined behaviour under the CUDA memory model. On this
GPU, with this scheduler, it happens not to manifest — warps 0–3 are issued
first and the zeroing is four instructions long.

**C — a 16-bit counter holds 0..65535.** Each block processes
`n / grid ≈ 67,108,864 / 64 ≈ 1,048,576` elements. On the skewed input 75% of
them land in the lowest 4/1024 of the range — which at `nBins = 256` is **one
bin** — so that bin's private counter reaches ~786,000 and wraps twelve times.
Each wrap silently loses 65,536 counts from the low half **and carries 1 into
the high half**, i.e. into bin `b+1`. The total comes out at 0.251× and bin 1 is
a handful too high. On the uniform input the maximum per-bin count is ~4,096 and
the defect is invisible.

---

## TODO 1 — which tool finds which

**Answer: `DIAG = {3, 4, 4}`.**

Verified by running all four tools on the shipped `exercise03.exe`:

```
##### memcheck
========= COMPUTE-SANITIZER
========= CUDA API Warning: Resetting device while there are still other users claiming to use it
========= ERROR SUMMARY: 1 error

##### racecheck
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)

##### initcheck --initcheck-address-space shared
========= Uninitialized __shared__ memory read of size 4 bytes
=========     at hist_broken(const unsigned int *, unsigned long long, unsigned int *, int)+0x200 in exercise03.cu:112
=========     by thread (1,0,0) in block (0,0,0)
=========     Address 0x690
========= ERROR SUMMARY: 38361 errors

##### synccheck
========= ERROR SUMMARY: 0 errors
```

memcheck's single "error" is the benign `cudaDeviceReset()` API warning that the
house convention (`setvbuf` + `cudaDeviceReset`) provokes in every file in this
course. It is counted in `ERROR SUMMARY` and is not a memory error. Module 12
recorded the same thing.

**Defect A → 3 (initcheck).** 38,361 uninitialized shared reads, named to the
source line. This is the tool's exact job and it does it perfectly.

**Defect B → 4 (nothing finds it), and this is the finding worth keeping.**

racecheck is *the* shared-memory hazard checker. Module 10 used it successfully
on exactly this shape of bug ("Drop the first `__syncthreads()` → intermittently
low counts; racecheck names it immediately"). Here it reports zero. The
difference is one word of the conflicting access:

| accumulation written as | racecheck |
|---|---|
| `atomicAdd(&s[b>>1], ...)` | **0 hazards displayed (0 errors, 0 warnings)** |
| `s[b>>1] += ...` | `2 hazards displayed (2 errors, 0 warnings)`, with 8,990,966 and 1,003,566 instances on the Read/Write pair |

Same kernel, same missing barrier, same schedule. **racecheck treats an atomic
as a synchronizing access and does not pair it against the ordinary store that
zeroed the bin.** Module 10 documented racecheck's blindness to global RMW; this
is a second blind spot, and it sits precisely inside the one kernel shape this
entire module has been teaching you to write. A privatized histogram *must* use
shared atomics, therefore a privatized histogram is exactly where racecheck
cannot see your missing barrier.

`synccheck` reports nothing, as it has in every module since Module 9 (standing
note: it detects nothing on divergent-barrier or barrier-omission bugs on
CUDA 13.2 / sm_89; documented as theory, never fixed).

**Defect C → 4.** No sanitizer models the *semantics* of a packed
representation. `atomicAdd(&s[w], 1u)` on a legally allocated, correctly
initialized, correctly synchronized 32-bit word is a valid operation. That the
programmer intended the low 16 bits to be a separate counter is not expressible
to the tool. This is the same class as Module 12's Exercise 3 finding: **the
tools check what you did, never what you meant.**

The partial-credit trap: predicting `2` for defect B is the reasonable inference
from Module 10, and it is wrong. There is no partial credit here on purpose.

---

## TODO 2 — the zeroing

```cpp
for (int w = threadIdx.x; w < nWords; w += blockDim.x) s[w] = 0u;
```

Strided, not guarded. This form is correct for every `(nBins, blockDim.x)` pair,
including `nWords < blockDim.x` (the extra threads simply do not enter the loop)
and `nWords` not a multiple of `blockDim.x`.

The flush loop at the bottom of the shipped kernel is already written this way —
`for (int w = threadIdx.x; w < nWords; w += blockDim.x)` — which is the strongest
hint in the file. The zeroing loop and the flush loop must iterate over the same
set; if one of them is guarded and the other strided, the block flushes words it
never initialized. Reading those two loops next to each other finds defect A in
about ten seconds and is a better technique than running anything.

---

## TODO 3 — the barrier

```cpp
    for (int w = threadIdx.x; w < nWords; w += blockDim.x) s[w] = 0u;
    __syncthreads();                                     // <-- TODO 3
```

Both guarantees are required, and neither of them is provided by the atomic.
Module 10's sentence, restated: a fence orders your accesses, a barrier orders
your threads, and an atomic makes one update indivisible. Three different
things; you need two of them here and the code had only one.

**Do not "fix" this with `__threadfence_block()`.** A fence supplies the
visibility half and not the execution half: warp 7 can still reach its
`atomicAdd` before warp 0 has issued its store, fence or no fence. This is the
distinction Module 9 built and Module 10 restated, and it is the reason the
correct answer is a barrier specifically.

---

## TODO 4 — the capacity bound (design)

The constraint forbids widening the counters, so the only move left is to make
the bound unreachable:

```cpp
static int chooseGridSafe(size_t n, int, int blockDim)
{
    const size_t CAP = 65535;
    size_t need = (n + (CAP - (size_t)blockDim)) / (CAP - (size_t)blockDim);
    int g = (int)need;
    return g < 1 ? 1 : (g > 1000000 ? 1000000 : g);
}
```

**The derivation.** A private bin of a block can be incremented at most once per
element that block processes, and the adversarial distribution ("every element
in one bin") achieves that bound exactly. A grid-stride loop with `grid` blocks
of `blockDim` threads gives each block at most
`ceil(n / (grid·blockDim)) · blockDim ≤ n/grid + blockDim` elements. Require

```
n/grid + blockDim <= 65535
```

and solve for `grid`. At `n = 2^26` and `blockDim = 256` that is
**grid ≥ 1,029**, and the harness prints the worst case your answer implies:

```
--- hist_student, your grid = 1029 (65218 elements per block) ---
  worst-case per-bin count in one block: 65474  (a 16-bit half holds 65535)
```

Note it is derived from `n` alone, not from the distribution. A bound that
depends on the data is not a bound.

**What it costs, which is the point of the TODO.** The packed representation
puts a hard ceiling on **coarsening** — the single most valuable technique in
this module (Exercise 1: 14.4× from the grid alone). With 32-bit counters you
would choose a machine-sized grid of 240 blocks; here you are forced to 1,029,
which is 4.3× more blocks and therefore 4.3× more flush atomics
(1,029 × 1,024 ≈ 1.05 M instead of 246 K). Against `n = 67 M` that is still only
1.6% of the input, so it is affordable — but the trade is explicit and it will
not be affordable at every `(n, nBins)`. **The 2× shared-memory saving is paid
for in coarsening depth**, and if the saving does not buy you a resident block
per SM (Module 6's table: the boundaries are at 12,288 / 16,384 / 25,600 /
49,152 B), you have paid for nothing.

**Wrong answers and their symptoms:**

| answer | symptom |
|---|---|
| widen to `unsigned int` per bin | correct and fast; the harness fails the "packed representation preserved" check, which is the whole constraint |
| `grid = n / 65535` without the `+ blockDim` term | off by one block-worth of elements; passes on this input, and is wrong for an adversarial one. The harness's own arithmetic check catches it |
| flush and re-zero every 65535 elements inside the loop | correct, and legitimate — it needs two extra barriers per flush interval and multiplies the flush atomics by the number of intervals. Measure it: at these sizes it is slower than raising the grid |
| bound from the observed maximum bin count | not a bound |

---

## Synchronization / memory reasoning

The fixed kernel has two barriers and they are the two from lesson §4. The
second one (before the flush) was present in the shipped file and is correct —
which is itself instructive: the presence of one barrier is not evidence that
the other is unnecessary, and a reader who sees `__syncthreads()` in the kernel
and stops looking has been had.

The flush is a WAR-free read of a region only this block wrote, so no third
barrier is needed. The global `atomicAdd` in the flush is required (many blocks
target the same `hist[b]`) and its return value is discarded, so it compiles to
`RED` (Module 10).

---

## Expected output

```
=== Module 14 / Exercise 3 (SOLUTION) — three defects, packed bins ===
N = 67108864 keys; hist_broken is launched with grid = 64, block = 256

--- hist_broken, as shipped (grid = 64) ---
  dist      nBins            total       expected   ratio
  uniform     256         67108864       67108864      1.000x
  skewed      256         16843519       67108864      0.251x
  uniform    1024       1114997106       67108864     16.615x
  skewed     1024       1516864430       67108864     22.603x

--- hist_student, your grid = 1029 (65218 elements per block) ---
  worst-case per-bin count in one block: 65474  (a 16-bit half holds 65535)
  [PASS] uniform  nBins= 256 : 0 wrong bins over 3 runs, total 67108864
  [PASS] skewed   nBins= 256 : 0 wrong bins over 3 runs, total 67108864
  [PASS] uniform  nBins=1024 : 0 wrong bins over 3 runs, total 67108864
  [PASS] skewed   nBins=1024 : 0 wrong bins over 3 runs, total 67108864
  [PASS] the packed representation is preserved: 512 B for 256 bins, 2048 B for 1024 bins
  [PASS] the grid bounds the per-block per-bin count below 65536
  [PASS] TODO 1 diagnosis codes (344)

  score: 4/4
OVERALL: PASS
```

The `nBins = 1024` totals vary run to run (16.4–16.6× and 22.6× observed across
runs) because they are reading uninitialized shared memory; the `nBins = 256`
skewed figure of 16,843,519 is **perfectly reproducible**, because integer
overflow is deterministic. That contrast — one symptom noisy, one symptom
stable — is itself a diagnostic: *a reproducible wrong answer is not a race.*
Module 12's Exercise 3 made the same point from the other side.

The arithmetic checks out exactly. 64 blocks, ~786,432 elements per block in the
hot bin, 786,432 / 65,536 = 12 wraps per block, 64 × 12 × 65,536 = 50,331,648
counts lost, and 67,108,864 − 50,331,648 = 16,777,216. Observed: 16,843,519.
The residual 66,303 is the carry into bin 1 plus the non-hot 25% of the data.

---

## The result that matters

Three defects, three different mechanisms, and **the tools found one of them**.
The one they found was the one you would also have found by reading the zeroing
loop next to the flush loop. The two they missed were a missing barrier — the
canonical racecheck bug, invisible here only because the conflicting access is
an atomic — and a semantic overflow that no memory-safety tool can model.

Tools bound your search; they do not conduct it. The discipline that finds all
three is the one this module has been teaching all along: **state the invariant
the representation requires, then prove the code maintains it.** For the packed
counters that invariant is "no half-word exceeds 65535", it is a statement about
`n/grid`, and the moment you write it down the fix is forced.

**Variation to try:** change the accumulation to `s[b>>1] += ...` (dropping the
atomic, which introduces a fourth defect) and run racecheck again. It will
report the missing barrier immediately, with millions of instances — the exact
hazard it refused to report one character earlier. Then put the atomic back and
watch it go quiet. That A/B is the most useful thing in this exercise and it
takes two minutes.
