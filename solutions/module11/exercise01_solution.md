# Module 11 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

Useful while working:

```
nvcc -arch=sm_89 -O3 -Xptxas -v -cubin -o exercise01.cubin exercise01.cu
cuobjdump -sass exercise01.cubin > sass.txt
findstr /C:"Function :" /C:"LDG" /C:"STG" sass.txt
```

---

## TODO 1 — the compulsory traffic

```cpp
static const int COMPULSORY_BYTES_PER_ELEMENT = 12;
```

The operation is

```
y[i] = clamp(A*x[i] + B*y[i] + C*x[i]*x[i], LO, HI)
```

Count *distinct array elements*, not source occurrences:

| array | role | bytes |
|---|---|---|
| `x` | read (named **twice**, fetched once) | 4 |
| `y` | read | 4 |
| `y` | written | 4 |

Total **12 B per element**.

Why "named twice, fetched once" is right from the hardware model: the second
occurrence of `x[i]` is the *same address*. The compiler keeps the loaded value
in a register — the SASS shows one `LDG.E.CONSTANT` per element and two `FFMA`s
consuming it — and even if it did not, the second load would hit in L1 with the
line still resident. Either way DRAM moves those 4 bytes once.

Why `y` counts twice: `B*y[i]` reads it and the assignment writes it. The write
is a full-sector write (every lane of every warp writes its own `float`, and a
warp covers 128 contiguous bytes = 4 whole sectors), so there is no
write-allocate surcharge on top — but the *read* is real and is the thing the
source hides. This is the in-place trap from the lesson, and it is the reason
this operation was chosen.

**Common wrong answers and the symptom each produces:**

| answer | reasoning | what it does to your numbers |
|---|---|---|
| **8** | "one load, one store" | you will compute 2/3 of the true bandwidth and conclude a finished kernel is at 58% of peak |
| **16** | counted `x` twice | your floor is 33% too high; a correct kernel will appear to beat its own floor, which should immediately tell you the model is wrong |
| **4** | counted the store only | the harness rejects it, and a correct kernel would appear to run at 3× the streaming ceiling |

The habit worth taking away: **if a measurement comes out below your floor, the
model is wrong, not the hardware.** That is also how you catch an accidentally
L2-resident benchmark.

## TODO 2 — the floor

```cpp
static const double PREDICTED_MS = 1.277;
```

```
12 B × 40,000,001  = 480,000,012 B
480,000,012 / 375.84e9 B/s = 1.2771e-3 s = 1.277 ms
```

`REFERENCE_GBS = 0.87 × 432.0 = 375.84 GB/s` — Module 5's measured streaming
rate for this GPU, not the nominal peak. Predicting against 432 would give
1.111 ms and set you chasing a number the memory controller has never produced.

The harness allows ±15% precisely because this figure is only good to about
that. What is *not* allowed to be sloppy is TODO 1: the byte count is exact.

## TODO 3 — the launch configuration

```cpp
static void chooseLaunch(int nSM, int* grid, int* block)
{
    *block = 128;
    *grid  = THREAD_BUDGET / *block;      // 64
    if (*grid < nSM) *grid = nSM;
}
```

Two constraints, both from the hardware:

1. **Spend the budget.** 8,192 threads is what you are allowed; using fewer
   throws away outstanding-request capacity you were given for free.
2. **`gridDim.x >= nSM`.** A block does not migrate (Module 1) and cannot be
   split across SMs, so a grid of 32 blocks leaves 8 of this GPU's 40 SMs with
   nothing to do — and with them, their share of the L1/LSU request capacity.

Measured, all five budget-respecting shapes, `blend_v1` (scalar) and a
`float4` version:

(measured ceiling for this run: 369.0 GB/s)

| grid × block | threads | scalar GB/s | % of ceiling | `float4` GB/s | % of ceiling |
|---|---|---|---|---|---|
| 40 × 64 (as shipped) | 2,560 | 90.9 | 24.6% | — | — |
| 64 × 128 | 8,192 | 241.1 | 65.4% | 380.2 | 103.0% |
| 32 × 256 | 8,192 | 246.4 | 66.8% | 383.9 | 104.1% |
| 128 × 64 | 8,192 | — | — | 386.0 | 104.6% |
| 16 × 512 | 8,192 | 245.8 | 66.6% | 386.1 | 104.6% |
| 8 × 1024 | 8,192 | 243.8 | 66.1% | — | — |
| *512 × 128 (budget removed)* | *65,536* | *347.9* | *94.3%* | — | — |
| *256 × 256 (budget removed)* | *65,536* | *347.6* | *94.2%* | — | — |

Read the scalar column first: **every budget-respecting launch configuration
lands between 65% and 67% of the ceiling.** Fixing the launch alone moves you
from 25% to 67% and then stops. That is the exercise: the launch is one lever,
it is worth 2.7×, and it cannot get you to 95%.

The two italic rows show what the budget is actually costing you. Lift it to
65,536 threads and the *scalar* kernel reaches 94.3% — one percentage point
short of the gate, using eight times the machine. The `float4` kernel clears the
gate comfortably on one eighth of it. That comparison, not the raw speedup, is
the point: per-thread memory-level parallelism and occupancy are two prices for
the same good, and the exercise makes you pay in the currency you have.

(The 32×256 and 16×512 rows nominally violate `grid >= nSM`. They measure fine
here because a bandwidth-bound kernel with a grid-stride loop keeps the bus busy
from fewer SMs; the constraint matters much more for compute-bound work, and
Module 19 quantifies it. Spending the budget as 64×128 or 128×64 is the answer
that is defensible without a measurement.)

## TODO 4 — the kernel

```cpp
__global__ void blend_v2(const float* __restrict__ x, float* __restrict__ y, long long n)
{
    const long long n4     = n / 4;
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long t      = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4*       y4 = reinterpret_cast<float4*>(y);

    for (long long i = t; i < n4; i += stride) {
        float4 X = x4[i], Y = y4[i];
        Y.x = op(X.x, Y.x);
        Y.y = op(X.y, Y.y);
        Y.z = op(X.z, Y.z);
        Y.w = op(X.w, Y.w);
        y4[i] = Y;
    }

    const long long tailBegin = 4 * n4;
    for (long long j = tailBegin + t; j < n; j += stride)
        y[j] = op(x[j], y[j]);
}
```

**Why this is the right lever, argued rather than pattern-matched.**
`blend_v1` is already perfectly coalesced: lane *L* of warp *w* touches
`base + 4*(32w + L)`, so each memory instruction covers 128 contiguous bytes =
4 whole sectors, 100% efficiency. There are no wasted bytes to recover. So the
kernel is not losing to *traffic*; it is losing to *concurrency*.

Little's Law (Module 1) says the memory system needs `bandwidth × latency` bytes
in flight. At ~380 GB/s and a ~575-cycle DRAM latency at ~1.9 GHz that is about
116 kB, i.e. ~29,000 outstanding 4-byte loads. The budget gives 8,192 threads.
Even with the compiler's 5× unroll of the grid-stride loop, the scalar kernel has
roughly 8,192 × 10 = 82,000 *elements* in flight but only 4 bytes per request —
and it is the request *slots*, not the bytes, that the LSU tracks.

A `float4` load occupies one request slot and returns 16 bytes. Same slots, four
times the payload. The SASS confirms the instruction side exactly: the scalar
kernel emits `10 × LDG.E.CONSTANT` and `5 × STG.E` per grid-stride iteration for
5 elements; the `float4` kernel emits `10 × LDG.E.128.CONSTANT` and
`5 × STG.E.128` for 20 elements. Identical instruction count, one quarter of the
memory instructions per element.

**Other answers that also pass.** Coarsening with hoisted loads (`C = 4` or
more, all loads issued before the first use) reaches the gate too, and so does
`float2` combined with coarsening. That is intentional: the TODO states a
requirement, not a mechanism, and the requirement is "more requests in flight
per thread". Any route to that works. What does *not* work is anything that
only changes the grid.

**The alignment argument you must be able to give.** `reinterpret_cast<float4*>`
is legal here because `float4` is declared `__align__(16)`, the `LDG.E.128`
instruction requires a 16-byte-aligned address, and `cudaMalloc` returns
pointers aligned to at least 256 B. We cast the allocation *base* with zero
offset, so `y4[i]` is at `base + 16i` — always 16 B aligned. Cast a pointer that
has been offset by an odd number of floats and the kernel raises
`cudaErrorMisalignedAddress` at the next synchronize (Module 5, Exercise 1).

## TODO 5 — the main range and the tail

```cpp
static const long long V2_MAIN_ELEMENTS = 4 * (N / 4);    // 40,000,000
```

`N = 40,000,001`, so `N / 4 = 10,000,000` whole `float4` groups covering
40,000,000 elements, and element 40,000,000 is the tail.

`N / 4` uses **truncating** division. `(N + 3) / 4` is the classic wrong answer:
it rounds up to 10,000,001, and the last `float4` reads 12 bytes past the end of
the allocation. It will usually not fault, because `cudaMalloc` pads; it shows up
as garbage in the last elements, or as nothing at all until someone changes `N`.
`compute-sanitizer --tool memcheck` reports it as an invalid `__global__` read of
size 16.

### The trap: the operation is not idempotent

```
y[i] = clamp(A*x[i] + B*y[i] + C*x[i]*x[i], LO, HI)
```

writes `y` from `y`. Applying it twice gives a different answer. That makes
"exactly once" a testable property, and the validator tests it.

The tail loop is written at **kernel scope**, not nested inside the `i < n4`
loop, and with its **own** range test. Wrong placements and their symptoms:

| attempt | symptom |
|---|---|
| tail inside the `i < n4` loop body | the tail element is processed once per loop iteration that thread executes — with a grid of 8,192 and `n4` of 10,000,000, thread 0 runs the loop ~1,220 times, so `y[40000000]` is updated 1,220 times. The tail column reports 1 mismatch and nothing else does. |
| `if (t == 0) for (j = tailBegin; j < n; ++j) …` | correct, but serialises the tail onto one thread. With a 1-element tail it is unmeasurable; write it this way for a 4,000-element tail and you have a kernel whose runtime is set by one thread. |
| `if (t >= n4 && t < n4 + tailCount)` with `j = tailBegin + (t - n4)` | correct **only if the grid exceeds `n4`**. Here the grid is 8,192 and `n4` is 10,000,000, so no thread ever satisfies `t >= n4` and the tail is silently never processed. The tail column reports 1 mismatch; the main column reports 0. This is the one that looks most careful and is most wrong. |
| a second kernel launch for the tail | correct. Costs ~5 µs of launch. Acceptable — the point of doing it in one launch is to see that it is unnecessary. |

### Why `<<<1,32>>>` is in the validator

A grid-stride loop's contract is that correctness does not depend on the launch
shape. The degenerate configuration — one block, one warp, 32 threads, for
40 million elements — exercises the loop at 1,250,000 iterations per thread and
catches every kernel that quietly assumes `gridDim.x * blockDim.x >= n` or
`>= n4`. It is slow (about 40 ms) and it runs in the untimed validation pass, so
it costs nothing but the truth.

Note that the tail loop above is written as a `for`, not an `if`. With
`tailCount = 1` an `if (t < tailCount)` would be equivalent, but the loop form
is correct for any tail size at any grid size, which is the same property the
main loop has. Writing the two halves in the same idiom is not decoration; it is
what makes the `<<<1,32>>>` test pass without thinking about it.

## Synchronization / memory reasoning

There is none, and there must not be any. Every thread owns a disjoint set of
elements, so no thread ever reads a value another thread wrote. No barrier is
required between the main loop and the tail — a reader who adds
`__syncthreads()` there has added a block-wide barrier inside a kernel where
different threads execute different numbers of loop iterations, which in the
worst arrangement is exactly Module 9's divergent-barrier undefined behaviour.

The in-place update is safe for the same reason the lesson's warning about
in-place stencils (Module 3, Module 10) does not apply: element *i* is written
only from element *i*. Change the operation to read `y[i-1]` and it becomes a
cross-block race with no barrier that can fix it.

## Performance reasoning

Measured on the RTX 3500 Ada, N = 40,000,001:

```
=== validation (untimed second pass) ===
  [ok] v1 numerics                              0 mismatch(es)
  [ok] v2 numerics, main range [0, 40000000)   0 mismatch(es)
  [ok] v2 numerics, tail  [40000000, 40000001)   0 mismatch(es)
  [ok] v2 at the degenerate config <<<1,32>>>   0 mismatch(es)
  [ok] launch 64 x 128 = 8192 threads, budget 8192
  [ok] TODO 1 compulsory bytes/element = 12
  [ok] TODO 2 predicted floor 1.277 ms vs 1.277 ms at 375.8 GB/s (+-15%)

=== performance ===
  your compulsory traffic  = 12 B/elem x 40000001 = 458 MB
  floor at 87% of peak (375.8 GB/s) = 1.277 ms
  measured streaming ceiling      = 368.9 GB/s (85% of 432)

  version                             ms      GB/s   %ofpeak    %ofceil
  v1 (as shipped, 40x64)          5.3117      90.4     20.9%      24.5%
  v2 (yours)                      1.2483     384.5     89.0%     104.2%
  speedup v1 -> v2 : 4.26x

  [ok] v2 reaches 104.2% of the measured ceiling (gate: 95.0%)

Score: 8/8
OVERALL: PASS
```

**Absolute numbers drift; ratios do not.** Across runs the measured ceiling
ranged 292–369 GB/s depending on thermal state and memory P-state, and `v2`
ranged 312–385 GB/s with it. The `%ofceil` column stayed between 103.8% and
106.7% every time, and the `v1 → v2` speedup between 3.9× and 4.5×. Report the
ratio.

**Why `v2` exceeds 100% of the "ceiling".** The yardstick `stream_ref` is
`o[i] = a[i]` across **two distinct 153 MB arrays**; `blend_v2` reads `x` and
read-modify-writes `y`, so its store hits the sector its own load just brought
into L2, and it touches two arrays rather than two arrays plus a distinct output
page. Slightly better DRAM row locality, slightly above the reference. Module 5
saw the same effect and the same conclusion applies: the streaming reference is
a yardstick, not a physical ceiling. The physical ceiling is 432 GB/s, and `v2`
is at 89% of it.

**Decomposing the 4.26×.** The two levers are separable and were measured
separately:

```
v1 at 40x64  (2,560 threads, scalar)  :  24.6% of ceiling
v1 at 64x128 (8,192 threads, scalar)  :  65.4% of ceiling     <- launch alone: 2.66x
v2 at 64x128 (8,192 threads, float4)  : 103.0% of ceiling     <- width: a further 1.58x
```

Neither lever alone reaches the gate. That is the design of the exercise.

## Expected output

Reproduce the block above. A correct submission has:

- all four numeric checks reporting 0 mismatches, **including the `<<<1,32>>>`
  row** — a nonzero there and zeros elsewhere means your kernel assumes a
  minimum grid size;
- the tail row separately at 0 — a `1` there is a TODO 5 bug and nothing else;
- `%ofceil` for `v2` in the 100–115% band;
- `Score: 8/8`.

If `v2` lands between 60% and 75% of the ceiling, you fixed the launch and not
the access. That is the halfway point and it is a good place to stop and re-read
the Little's Law paragraph before continuing.

## The result that matters

A perfectly coalesced kernel can still run at a quarter of the memory bandwidth,
and no amount of sector counting will tell you why, because the sector count is
already optimal. The missing resource is **requests in flight**, and you can buy
it with threads *or* with wider loads *or* with more independent loads per
thread — the memory system cannot tell them apart. When you are free to choose
the grid, threads are the cheapest currency. When you are not, per-thread
memory-level parallelism is the only one left, and this exercise removes the
cheap option so you have to find the other.

**Try this:** raise `THREAD_BUDGET` to 65,536 and re-measure `blend_v1` at
`512 × 128`. It gets to 94.3% of the ceiling with no vectorization at all — one
point short of the gate, on eight times the threads. Then halve the budget
repeatedly and plot where the scalar kernel falls off. You will have measured,
for this GPU and this kernel, the exchange rate between occupancy and
per-thread memory-level parallelism, which is the quantity Module 20 is about.
