# Module 14 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

---

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

The harness times six kernels × three distributions (18 configurations) in one
rotated sweep of 18 sweeps, auto-scales the iteration counts to ~10 ms segments,
warms up for 1500 ms, and validates in a separate untimed pass.

---

## TODO 1 — the three predictions

Measured on this GPU, thermally settled:

| distribution | v4 over v0 |
|---|---|
| uniform | **100.0×** |
| same-bin | **205.7×** |
| clustered | **121.8×** |

Anything within a factor of two scores. The reasoning that gets you there:

**Start from v4, not from v0.** The finished kernel reads 192 MiB once, through
`uchar4` loads, into per-block shared bins. It is a pure streaming kernel with
an increment attached, and it runs at the streaming ceiling — **0.500–0.504 ms
in all three columns**. Predict that first and predict it as a constant, because
that is the whole point of the exercise. If you predicted three different
numbers for v4 you have not yet believed that privatization removes the
distribution from the problem.

**Then v0 is the only variable.** From Module 10's contention economics:

- `same-bin` is `K = 1`: one L2 slice ALU, everything serializes. Module 10's
  worst case, and it measures 103.1 ms.
- `clustered` is the *warp-uniform address* case: a warp's 32 lanes present one
  address, so the warp contributes one serialized operation rather than 32
  parallel ones. Module 10 measured that at ~6× even with many distinct bins in
  play; here it costs 61.2 ms against uniform's 50.4.
- `uniform` is `K = 256` with a data-dependent address, 50.4 ms.

Dividing gives 100×, 206×, 122×. A prediction of "100× / 200× / 100×" is a
clean pass.

**The trap in this TODO** is predicting from the technique list rather than from
the physics: readers who reason "privatization is worth 48× (Module 10), so the
answer is about 48× everywhere" get uniform right-ish and both others wrong,
*and* they are right for the wrong reason — the 48× is Module 10's `K = 1`
figure, not its `K = 256` figure.

---

## TODO 2 — the privatized kernel

```cpp
__global__ void hist_shared(const unsigned char* __restrict__ in, size_t n,
                            unsigned int* __restrict__ hist, int nBins)
{
    extern __shared__ unsigned int s[];
    for (int b = threadIdx.x; b < nBins; b += blockDim.x) s[b] = 0u;   // (a)
    __syncthreads();                                                   // (b)

    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&s[in[i]], 1u);                                      // (c)
    __syncthreads();                                                   // (d)

    for (int b = threadIdx.x; b < nBins; b += blockDim.x)
        if (s[b]) atomicAdd(&hist[b], s[b]);                           // (e)
}
```

**(a) is the trap the block size was chosen to spring.** `BLK = 128` and
`nBins = 256`. `if (threadIdx.x < nBins) s[threadIdx.x] = 0u;` initializes bins
0–127 and leaves 128–255 holding whatever the *previous block that occupied this
SM's shared memory* wrote there. The symptom is counts that are **too high**,
not too low, and they vary run to run because which block ran before you is a
scheduling accident.

That polarity is a diagnostic worth memorising:

| symptom | cause |
|---|---|
| counts too **low**, catastrophically | lost-update race — a missing atomic (M10) |
| counts too **low** by a multiple of 65536 | counter overflow (Exercise 3) |
| counts too **high**, varying run to run | uninitialized private bins |
| counts too **high** by a constant | double-processing the tail (M11) |

`compute-sanitizer --tool initcheck --initcheck-address-space shared` reports
this one directly as `Uninitialized __shared__ memory read of size 4 bytes`, and
it is the *only* one of the four tools that does.

**(b) and (d) are different guarantees and cannot be merged or dropped.**
Module 9: `__syncthreads()` is an execution barrier *and* a block-scope memory
fence. (b) exists so that thread 5's zero of bin 200 is visible to thread 90
before thread 90 adds to bin 200. (d) exists so that every add is retired before
any thread reads a bin for the flush. A common wrong answer is to keep (d) and
drop (b) on the grounds that "the accumulate loop has a barrier after it" —
it does, and it is the wrong end.

**Measure this one thing yourself:** drop (b), rebuild, and run
`compute-sanitizer --tool racecheck`. It reports **zero hazards**. Exercise 3
documents why in full: racecheck does not flag a plain write racing with an
`atomicAdd` on the same shared address. Change (c) to a plain `s[in[i]] += 1u`
and racecheck instantly reports two hazards with ~9 million instances. This is
Module 10's "racecheck has blind spots" finding, in a second place, and it is
exactly where a correct privatized histogram lives.

**(c) must be a grid-stride loop, not `if (i < n)`.** TODO 3 is going to launch
this kernel with 480 blocks against 201 million elements. A kernel written
`if (i < n) atomicAdd(...)` silently histograms 0.03% of the input and the
harness catches it immediately.

**(e)'s `if (s[b])` is worth measuring.** It costs one predicated compare and
saves a global atomic. On `same-bin`, 255 of 256 bins are empty in every block
and it removes 99.6% of the flush. This is visible in the v1 row of the results
table below: the same kernel takes 15.90 ms on `uniform` and 2.32 ms on
`same-bin`, and the entire difference is this one test.

---

## TODO 3 — the grid (design)

```cpp
static int chooseGrid(int nSM, int blocksPerSM, size_t, int)
{
    int g = nSM * blocksPerSM;
    return g < 1 ? 1 : g;
}
```

40 SMs × 12 resident blocks (128 threads, 4 KB shared) = **480 blocks**.

The requirement was two-sided and the harness prints both sides:

```
flush global atomics: v1 4.03e+08  v2 1.23e+05  v3/v4 1.23e+05   (N = 2.01e+08)
```

v1's textbook grid of `ceil(N/128) = 1,572,864` blocks produces **403 million**
flush atomics against an input of 201 million elements. It converts `N`
contended global atomics into `N` shared atomics **plus 2N global atomics**. It
removes the contention and doubles the traffic. Measured: 15.90 ms on uniform
against v2's 1.10 ms — a **14.4× penalty for the grid alone**, with identical
kernel source.

`example01.cu` Part C sweeps this directly and shows the optimum is broad:
any grid from ~3,000 to ~12,000 blocks is within 12% of the best, and the curve
only turns back up below 240 blocks, where the machine stops being full. So:
machine-sized grid, and do not tune it further.

**Common wrong answers:**

| answer | symptom |
|---|---|
| `ceil(n / blockDim)` | correct, 14× slow, fails the 20× same-bin gate |
| a fixed 65535 | correct, ~1.3× slow; grid×nBins = 16.8 M, still 12× less than N, and the real cost is the zero-and-flush per block |
| `nSM` (one block per SM) | correct, ~1.5× slow — 40 blocks cannot fill 40 SMs' worth of memory parallelism |
| grid computed from `n` with no occupancy input | usually fine; the point of passing `blocksPerSM` is that the answer must change when the shared footprint does, which TODO 4 makes it do |

---

## TODO 4 — replication (design + code)

```cpp
unsigned int* my = s + (size_t)(((int)threadIdx.x >> 5) & rmask) * nBins;
...
for (int b = threadIdx.x; b < nBins; b += blockDim.x) {
    unsigned int sum = 0u;
    for (int r = 0; r <= rmask; ++r) sum += s[r * nBins + b];
    if (sum) atomicAdd(&hist[b], sum);
}
```

```cpp
static int chooseR(int nBins, int blockDim)
{
    int warps = blockDim / 32;              // 4 at BLK = 128
    int R = 1;
    while (R * 2 <= warps && (size_t)R * 2 * nBins * 4 <= 8192u) R *= 2;
    return R;                               // 4
}
```

Two decisions and one measurement.

**Decision 1: one replica per warp, `r = threadIdx.x >> 5`.** Not
`r = threadIdx.x & (R-1)`. Ada's shared-memory unit executes
`ATOMS.POPC.INC.32`, which merges all the lanes of one warp targeting one
address into a single increment. Splitting a warp across replicas destroys that
merge and gains nothing, because the lanes of a warp were never the expensive
contenders. Measured on this harness: lane-group replication is **0.91–0.94× of
R=1** on `clustered` and `same-bin` — a regression.

**Decision 2: replica-major, `s[r * nBins + b]`.** With this layout a warp's
address pattern is `r` fixed, `b` varying — bit-for-bit the unreplicated
pattern, so Module 7's bank analysis is unchanged. The replica-minor layout
`s[b * R + r]` gives a fixed `r` a stride of `R` across `b`, which is a
`gcd(R, 32)`-way conflict: 4-way at `R = 4`, 8-way at `R = 8`. Module 7's
`max(2, D)` law makes the 4-way case cost 4 replays per access on the hot path.

**And the measurement, which is a null result:**

```
  v2 hist_shared, your grid               1.1018      1.0981      1.0982
  v3 hist_repl, your grid and R           1.1028      1.0999      1.1007
```

**1.00× on all three distributions, including `same-bin`, where every thread in
every block is hammering one bin.** `example01.cu` Part B extends this to
R ∈ {1,2,4,8,16} over four distributions: flat to within 2% up to R = 8, and
**0.83× at R = 16**, where 16 KB of shared memory drops the SM from 6 resident
blocks to 5 (Module 6's table; Module 19 formalises it).

Why nothing: the intra-warp collisions are free (`ATOMS.POPC`), and the
remaining inter-warp collisions cost at most one shared-unit operation per warp
per round — 4 operations per 128 input bytes at `BLK = 128`. At the streaming
ceiling the memory system delivers roughly 5 bytes per SM per cycle. The
contention replication exists to fix is **about 50× cheaper than the data it is
waiting on**.

A full-credit answer to this TODO is `R = 1` with the reasoning written down.
The harness accepts any `R` in `[1, blockDim/32]` precisely so that the reader
who reasons it out and the reader who implements it both get the same score and
the same lesson. **What is graded is that you measured it.**

**Where replication *would* pay:** when `ATOMS.POPC.INC` does not apply, i.e.
increments by a data-dependent weight. Try it: change the accumulate line to
`atomicAdd(&my[v], (unsigned)(v & 7u) + 1u)` and re-measure. `clustered` and
`same-bin` jump from 1.10 ms to 2.79–2.96 ms, because the lanes of a warp now
serialize. (Replication still will not fix *that*, because the serialization is
inside the warp — `__match_any_sync` aggregation is the fix, and Exercise 2 uses
it. Replication only separates different warps.)

---

## TODO 5 — the vectorized version

Identical to TODO 4 except the input side:

```cpp
uchar4 v = in4[i];
atomicAdd(&my[v.x], 1u);  atomicAdd(&my[v.y], 1u);
atomicAdd(&my[v.z], 1u);  atomicAdd(&my[v.w], 1u);
```

**Exactly the same number of atomic operations. 1.96–2.19× faster.** This is the
biggest single step in the whole ladder after the grid fix, and it has nothing
to do with atomics at all.

The mechanism is Module 5's, measured directly:

```
  uchar4 stream, no atomics        0.6291 ms   320.0 GB/s
  uchar  stream, no atomics        1.3249 ms   152.0 GB/s
  uchar  load + 1 ATOMS/elem       1.3474 ms   149.4 GB/s
  uchar4 load + 4 ATOMS/elem4      0.6313 ms   318.9 GB/s
  uchar4 load + 1 ATOMS/elem4      0.6334 ms   317.9 GB/s
```

A scalar `unsigned char` load moves 32 bytes per warp instruction; a `uchar4`
load moves 128. Adding the shared atomics on top of either costs **1.7%**. The
privatized histogram was never atomic-bound; it was bound by the width of its
loads, and rows 2 and 3 of that table prove it by holding everything else fixed.

In the SASS the change is one opcode:

```
LDG.E.U8.CONSTANT R2, [R2.64] ;     // v2/v3: 32 B per warp instruction
LDG.E.CONSTANT    R4, [R4.64] ;     // v4:   128 B per warp instruction
```

Both kernels then issue the same `ATOMS.POPC.INC.32`.

---

## Synchronization / memory reasoning

Three sync points, three different reasons:

| where | why | what breaks without it |
|---|---|---|
| after zeroing | make the zeros visible block-wide | counts too high, varying; **racecheck is silent** |
| after accumulate | make every add complete before the flush reads | counts too low, varying; racecheck silent for the same reason |
| the `atomicAdd` itself | indivisibility, which no barrier provides (M10) | catastrophic loss, ~1 count per warp |

Neither barrier removes the need for the atomic and the atomic removes the need
for neither barrier. Module 10 stated this; this exercise is where it costs
something.

---

## Performance reasoning

Full observed output, thermally settled:

```
your choices: R = 4 (4096 B shared, 12 blocks/SM), grid = 480
flush global atomics: v1 4.03e+08  v2 1.23e+05  v3/v4 1.23e+05   (N = 2.01e+08)

--- timing (min of 18 rotated sweeps, ms) ---
  kernel                                 uniform    same-bin   clustered
  ceiling (stream only)                   0.4997      0.4994      0.4995
  v0 global atomic                       50.4091    103.1264     61.1516
  v1 hist_shared, grid=ceil(N/BLK)       15.8979      2.3150      2.3171
  v2 hist_shared, your grid               1.1018      1.0981      1.0982
  v3 hist_repl, your grid and R           1.1028      1.0999      1.1007
  v4 hist_vec (uchar4)                    0.5039      0.5012      0.5020

  speedup over v0, and % of the measured ceiling:
  v1 hist_shared, grid=ceil(N/BLK)        3.17x/  3%     44.55x/ 22%     26.39x/ 22%
  v2 hist_shared, your grid              45.75x/ 45%     93.91x/ 45%     55.68x/ 45%
  v3 hist_repl, your grid and R          45.71x/ 45%     93.76x/ 45%     55.56x/ 45%
  v4 hist_vec (uchar4)                  100.03x/ 99%    205.74x/100%    121.81x/ 99%

  ceiling: 0.4994 ms = 403.1 GB/s = 93.3% of 432 GB/s peak

--- validation ---
  [PASS] v1 hist_shared, grid=ceil(N/BLK)   exact on all 3 distributions
  [PASS] v2 hist_shared, your grid          exact on all 3 distributions
  [PASS] v3 hist_repl, your grid and R      exact on all 3 distributions
  [PASS] v4 hist_vec (uchar4)               exact on all 3 distributions
  [PASS] v4 at least 1.50x v2 on every distribution (worst 2.19x)
  [PASS] v2 at least 20x v0 on same-bin (got 93.9x)
  TODO 1 predictions (octave-scored, v4 vs v0):
    [PASS] uniform    predicted   100.0x, measured   100.0x
    [PASS] same-bin   predicted   200.0x, measured   205.7x
    [PASS] clustered  predicted   120.0x, measured   121.8x

  score: 9/9
OVERALL: PASS
```

The step sizes, which are the stable quantities:

| step | uniform | same-bin | clustered |
|---|---|---|---|
| v0 → v1 (privatize) | 3.17× | 44.6× | 26.4× |
| v1 → v2 (coarsen) | **14.43×** | 2.11× | 2.11× |
| v2 → v3 (replicate) | 1.00× | 1.00× | 1.00× |
| v3 → v4 (vectorize) | **2.19×** | 2.19× | 2.19× |

**Run-to-run variance.** On a thermally settled machine the ceiling measures
0.4991–0.4997 ms (403 GB/s, 93% of peak). Under sustained load it degrades to
0.62 ms (322 GB/s) and every absolute figure moves with it; the ratios above
reproduce to ~1% in both states, which is why the scored gates are ratios
(spec §12 rule 5). If your ceiling row reads 0.62 rather than 0.50, let the GPU
idle for two minutes and re-run.

---

## The result that matters

The two optimizations with a name — privatization and replication — are worth
3.2× and 1.00× on the uniform input. The two with no name — *choose a grid from
the traffic model* and *widen the load* — are worth 14.4× and 2.2×. The famous
moves attack contention, and this kernel stops being contention-bound after the
first of them; everything after that is Module 11's memory-floor discipline
wearing a histogram costume. **When a kernel reaches 46% of the streaming
ceiling, the remaining factor of two is not in the algorithm.**

The second result is the flatness of the v4 row: 0.5012–0.5039 ms across inputs
that make the naive kernel vary by 2×. A finished histogram does not have a
best case and a worst case. If yours does, you are not finished.

**Variation to try:** change the accumulate to add a data-dependent weight
(`atomicAdd(&my[v], (v & 7u) + 1u)`) and re-run all four rungs. `ATOMS.POPC.INC`
is an increment-by-one instruction and disappears from the SASS; the clustered
and same-bin columns jump to 2.8–3.0 ms; and for the first time in this module
you have a workload where warp-level aggregation is worth something. Then check
whether replication helps — it will not, and working out why is the whole point.
