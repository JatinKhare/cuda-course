# Module 5 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

## TODO 1 — the layout transformation

```cpp
#define UPLOAD_FIELD(dst, field) do {                                  \
    for (long long i_ = 0; i_ < N; ++i_) h_tmp[i_] = h_p[i_].field;    \
    CHECK(cudaMemcpy(dst, h_tmp, fldBytes, cudaMemcpyHostToDevice));   \
} while (0)
```

**The cost, which you were asked to state.** One invocation reads the whole
366 MB AoS array on the host (touching 4 of every 24 bytes, so the host cache
behaves as badly as the GPU would) and sends 61 MB over PCIe. Six invocations:
2.2 GB of host reads, 366 MB over PCIe. On a Gen4 x8 laptop link at ~12 GB/s
that is ~30 ms for the transfers alone, and the gather loops cost more.

**When it repays.** The SoA kernel saves 2.05 ms − 1.53 ms ≈ 0.52 ms per
launch on this machine. Ignoring the gather and counting only PCIe, the
transform pays for itself after roughly 60 timesteps; counting the host gather,
several hundred. So:

- A pipeline that uploads once and integrates for thousands of steps: convert,
  obviously.
- A one-shot kernel over data that arrives as AoS: converting on the host is a
  clear loss. Convert *on the device* instead (one strided read, one coalesced
  write — you pay the bad pattern once rather than every step), or restructure
  the producer so the data is born as SoA.
- The best answer is usually the third one. Layout is a property of the data
  structure, not of the kernel, and the right place to fix it is where the data
  is created.

A common wrong instinct is to write a device-side transpose kernel and call the
problem solved. That kernel has exactly the access pattern this module is
teaching you to avoid, and it is the subject of Module 15.

## TODO 2 — the SoA kernel

```cpp
__global__ void update_soa(float* __restrict__ x, float* __restrict__ y,
                           float* __restrict__ z,
                           const float* __restrict__ vx,
                           const float* __restrict__ vy,
                           const float* __restrict__ vz,
                           long long n, float dt)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}
```

Why it is correct from the hardware model: on each of the six memory
instructions, lane `L` of warp `w` supplies address `base + 4*(32w + L)`. The
32 addresses are `{base+128w, …, base+128w+124}` — 128 contiguous bytes from a
128 B-aligned start (because `cudaMalloc` gives ≥ 256 B alignment and `128w` is
a multiple of 128). Four sectors, all four fully consumed. 100 %.

`__restrict__` earns its place here. Without it the compiler must assume `x`
and `vx` may alias, so it cannot issue the `y` and `z` loads before the `x`
store retires. With it, all six loads can be hoisted and issued back to back,
which is what keeps six independent memory streams in flight. It does not
change the sector count; it changes how many requests are outstanding
simultaneously, which is what hides latency.

Common wrong approaches:

- Using `threadIdx.x` alone as the index: only block 0's data is updated, the
  rest of the array keeps its initial values, and the validator reports ~N
  mismatches.
- A grid-stride loop with a stride that is not a multiple of 32: each iteration
  still reads 32 contiguous floats per warp, so it is *still* coalesced — this
  one is a red herring, and worth confirming for yourself.
- Declaring the parameters `float* __restrict__` for *all six* including the
  velocities, then writing to a velocity array anywhere: undefined behaviour,
  and the compiler will happily reorder your stores. The velocities are
  `const` here for that reason.

## TODO 3 — `nVec` and the vectorised fast path

```cpp
const long long nVec = N / 4;          // 4,000,000 for N = 16,000,003
```

Integer division truncates, which is what you want: `nVec` counts *whole*
`float4` groups. `(N + 3) / 4` is the classic wrong answer — it rounds up, and
the last group reads four floats of which only three exist. With `N` = 16,000,003
that is a 4-byte overrun past the end of a `cudaMalloc` region; it will
usually *not* fault (the allocation is padded), so the bug shows up as garbage
in the last element, or as nothing at all until the array size changes.

```cpp
float4*       x4  = reinterpret_cast<float4*>(x);
const float4* vx4 = reinterpret_cast<const float4*>(vx);

float4 a = x4[i], b = vx4[i];
a.x += b.x*dt; a.y += b.y*dt; a.z += b.z*dt; a.w += b.w*dt;
x4[i] = a;
```

**Why the cast is legal.** `float4` is declared with `__align__(16)`, so the
compiler will emit `LDG.E.128`, and that instruction requires a 16 B-aligned
address. `cudaMalloc` returns ≥ 256 B-aligned pointers and we are casting the
allocation base with zero offset, so every `x4[i]` is at `base + 16i` — always
16 B aligned. Cast a pointer that has been offset by an odd number of floats
and you get `cudaErrorMisalignedAddress` (Exercise 1, TODO 4).

**What you actually buy.** A warp of 32 lanes each loading a `float4` covers
512 contiguous bytes = 16 sectors for 512 requested bytes: 100 %, the same
efficiency as the scalar version. The sector count per useful byte is
*unchanged*. The gains are:

- one instruction instead of four, so a quarter of the issue slots;
- one entry in the SM's outstanding-request tracking structure instead of four,
  covering four times as much data — so the same number of in-flight requests
  hides four times as much latency;
- a quarter of the address arithmetic.

Measured here, that is worth about 1 % on the fully coalesced kernel (it is
already DRAM-bound) but would be worth much more on a kernel with fewer resident
warps. The lesson is that vectorising is a *latency/issue* optimization, not a
*bandwidth* one. People who expect a 4× speedup have confused the two.

## TODO 4 — the tail

```cpp
const long long tailBegin = 4 * nVec;
const long long tailCount = n - tailBegin;
if (i < tailCount) {
    long long j = tailBegin + i;
    x[j] += vx[j] * dt;
    y[j] += vy[j] * dt;
    z[j] += vz[j] * dt;
}
```

Placed at kernel scope, **not** nested inside the `i < nVec` branch, and using
a *separate* range test.

Why this is correct: the grid is sized over `nVec`, so every index `0 …
nVec-1` exists exactly once in the grid. Threads `0, 1, 2` additionally handle
elements `16,000,000 … 16,000,002`. Each tail element is therefore written by
exactly one thread. Only one warp in the entire grid evaluates `i < tailCount`
as true for any lane; every other warp evaluates it uniformly false and skips
the block with a single predicated branch. The cost is one comparison per
thread — unmeasurable.

Wrong placements, and the symptom each produces:

| Attempt | Symptom |
|---|---|
| Tail inside `if (i < nVec) { … }` | Threads 0–2 are inside that branch, so it *works* here — but only because `tailCount < nVec`. Change `N` so the tail index exceeds `nVec` and it silently skips the tail. A correctness bug that hides behind a particular array size is the worst kind. |
| `if (i == 0) for (j = tailBegin; j < n; ++j) …` | Correct, but serialises three elements onto one thread while 4 M threads wait for the kernel to drain. Measurable only as a tiny tail latency here; a disaster if the tail were larger. |
| `if (i >= nVec && i < nVec + tailCount)` with `j = tailBegin + (i - nVec)` | Correct **only if the grid is larger than `nVec`**. With `GRID_V = ceil(nVec/256)` and `nVec` a multiple of 256, there are no threads with `i >= nVec`, so the tail is never processed. `tailBad` reports 9 (3 elements × 3 fields). |
| A separate 1-block kernel launch for the tail | Correct. Costs an extra ~5 µs launch. Acceptable, but the point of the exercise is to see that it is unnecessary. |
| Rounding `nVec` up and letting the last `float4` overrun | Reads 4 B past the array. Usually no fault; `compute-sanitizer --tool memcheck` reports an invalid `__global__ read of size 16`. |

The harness reports tail mismatches separately precisely so these show up as
`(9, of which 9 in the tail)` rather than as a vague failure.

## Synchronization / memory reasoning

There is none, deliberately. Every thread owns a disjoint set of elements, so
there is no ordering requirement between threads and no barrier is needed. This
is worth noticing because the vectorised kernel *looks* like it has two phases
(fast path, tail) and beginners frequently insert a `__syncthreads()` between
them. It would be harmless here but it would also be superstition — and if the
tail branch were inside divergent control flow it would be a hang. Module 9
makes this precise.

## Performance reasoning

Measured, 16,000,003 particles, 549 MB of useful traffic per launch, two passes
with the second reported:

```
N = 16000003  (N % 4 = 3)
AoS array 366 MB, one SoA field 61 MB, L2 = 48 MB
useful traffic / launch = 549 MB

nVec = 4000000, 4*nVec = 16000000, tail = 3 element(s)

  version                   ms       GB/s   %ofpeak  %ofstream   validation
  v1 AoS                 2.046      281.5     65.2%      75.7%   PASS (0)
  v2 SoA                 1.533      375.7     87.0%     101.1%   PASS (0)
  v3 SoA+float4          1.520      379.0     87.7%     102.0%   PASS (0, of which 0 in the tail)

  measured streaming ceiling (1 read + 1 write) : 371.7 GB/s = 86% of the 432 GB/s nominal peak
```

Absolute GB/s drift with the memory P-state and the power cap; across runs
`v1` measured 191–282 GB/s and `v2`/`v3` 258–380 GB/s. The **`%ofstream`**
column is far steadier: v1 lands on 75.7–75.8 % every single run, v2 and v3 on
89–103 %. That column exists
because a laptop GPU's clocks are not a constant, and a ranking built on
absolute numbers taken minutes apart is not a ranking.

**v2 and v3 exceed 100 % of the streaming reference.** That is not an error.
The reference kernel is `out[i] = in[i]*2+1` on two distinct arrays: every byte
read is a compulsory miss and every byte written is a full-sector store to a
different page. The SoA integrator reads and writes *the same* `x` array, so
the store hits the sector the load just brought in, and it writes full sectors.
Slightly better DRAM page locality, slightly above the reference. Treat the
reference as a yardstick, not a ceiling.

**Why v1 is only 1.34× slower, not 6×.** This is the point the exercise is
built around, and it is the correction to the AoS folklore. The kernel reads
*all six* fields of each particle and writes three. A warp's six load
instructions each cover the same 768 B / 24 sectors; the first misses and the
other five hit in L1. DRAM moves 768 B per warp and every byte is used. The
remaining 24 % gap comes from:

- **request count**, not bytes: 6 × 24 = 144 sector-requests per warp against
  SoA's 6 × 4 = 24, so the L1/LSU path does six times the work for the same
  DRAM traffic;
- **partial-sector writes**: the three stores write 12 of every 24 bytes, so
  most sectors are partially written and must be read-merged-written rather
  than simply written.

Compare with `example02.cu` Part 1, where the kernel touches *one* field: there
AoS measures 57.0 GB/s against SoA's 374.0, a factor of **6.6**. Same layout,
same hardware; the only difference is what fraction of each fetched sector the
warp consumes. **The AoS penalty is not a property of the struct. It is a
property of the access.**

## Expected output

Reproduce the block quoted above. Requirements for a correct submission:

- all three rows `PASS (0)`;
- v3's parenthetical reads `(0, of which 0 in the tail)` — a nonzero second
  number is a TODO 4 bug and nothing else;
- `%ofstream` roughly 76 / 101 / 102.

If v2 and v3 are within noise of each other, that is correct and expected: both
are DRAM-bound at ~100 % efficiency, and vectorising cannot beat the bus.

## The result that matters

The layout you choose determines how many of the bytes DRAM delivers your warp
actually uses, and that fraction — not the instruction count, not the FLOPs —
sets the runtime of a memory-bound kernel. But the fraction depends on the
*access*, not the layout alone: AoS costs 6× when you read one field and 1.3×
when you read them all. Before converting anything to SoA, ask what fraction of
each record the hot kernel touches.

Try this: change `update_aos` so it updates only `x` (drop the `y` and `z`
lines) and adjust `useful` to 8 B per particle. The AoS row should collapse to
roughly 1/6 of the SoA row, reproducing `example02.cu` Part 1 — and you will
have derived the folklore and its exception from the same program.
