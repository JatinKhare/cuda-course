# Module 17 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

Warning-clean. `SCORE: 7/7`, `OVERALL: PASS`.

---

## TODO 3 (part 1) — the number of tiles

```cpp
int nTiles = (K + T - 1) / T;          // ceil, not K/T
```

`K = 1063`; `1063 % 8 = 1063 % 16 = 1063 % 32 = 7`. `K / T` runs 132, 66 and 33
tiles respectively and drops the final 7 values of `k` from **every** dot
product in the matrix.

Why it matters that this is easy to miss: 7 terms out of 1063 is a 0.66 %
perturbation of a sum of positive numbers. Against a relative-to-`|C|`
tolerance of `1e-5` it is caught, but against a sloppier one it is not, and
against a test matrix of zero-mean data the *relative* error would be smaller
still. Measured through Module 16's validator:

```
tiled T=8    Freiv 145   samp 151.9   FAIL
tiled T=16   Freiv 145   samp 151.9   FAIL
tiled T=32   Freiv 145   samp 151.9   FAIL
```

145× the bound. The scaled tolerance leaves two orders of magnitude of headroom
on a correct kernel and still rejects this by two orders of magnitude the other
way.

**Note the shape of the failure.** The error is identical at all three tile
sizes. That is the signature of a defect that is a property of the *algorithm*
rather than of the *schedule*: nothing about it depends on how many warps are
in a block.

---

## TODO 1 — the A-tile cooperative load

```cpp
const int aCol = k0 + tx;      // tx walks the CONTRACTION axis of A
As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
```

Three separate decisions are packed into that line.

**Which element.** Thread `(tx, ty)` must fetch `A[row0 + ty][k0 + tx]`. The
compute loop reads `As[ty][k]` for `k` in `[0, T)`, so slot `As[ty][tx]` must
hold the element of A at row `row0 + ty` and contraction index `k0 + tx`. The
row index of the A tile is the same as the row index of the C tile; the column
index is *not* the column index of the C tile, it is a position along `k`.

**Why `tx` and not `ty` carries `k`.** A is row-major with stride `K`, so
consecutive `k` are consecutive addresses. A warp is 32 consecutive linearised
thread ids (Module 3), i.e. consecutive `tx` at fixed `ty` when `T = 32`, or
two rows of 16 when `T = 16`. Putting `k` on `tx` makes the warp read
`A[row*K + k0 .. k0+31]` — 128 contiguous bytes, 4 sectors, perfectly coalesced
(Module 5). Swap `tx` and `ty` in the address and the same warp reads 32
addresses `K*4 = 4252 B` apart: **32 sectors instead of 4.**

**The boundary.** `row` can exceed `M − 1` (because `M % T ≠ 0`) and `aCol` can
exceed `K − 1` (because `K % T ≠ 0`). The cell has no corresponding element of
A, and the correct thing to put there is **0.0f**, because a zero contributes
nothing to a dot product: `sum_k a_k b_k` over a zero-padded tile equals the sum
over the real terms. It is not an approximation, it is an identity.

**The part that is a real hazard.** The store must be executed by *every*
thread, including threads whose output element does not exist. The obvious
wrong shape is

```cpp
if (row >= M || col >= N) return;      // WRONG here, correct in the naive kernel
```

at the top of the kernel. In the naive kernel that early return is right. In a
tiled kernel it removes threads from the block before the barrier, and
`__syncthreads()` in the presence of exited threads is exactly the case Module 9
covered: `bar.sync` counts arrivals per warp and exited threads are subtracted,
so the barrier "succeeds" with a subset of the block and the remaining threads
read tile cells nobody wrote. On sm_89 this corrupts silently rather than
hanging. The guard belongs on the *store to C*, not on entry.

### The subtle trap, measured

The zero-fill is logically required on both the A tile and the B tile. It is
**arithmetically redundant**: you only need one of them.

Measured on the shipped harness:

| variant | result |
|---|---|
| both zero-fills present (correct) | PASS, samp 0.0269 |
| A zero-filled, **B not** (`if (...) Bs[ty][tx] = ...;`) | **PASS**, samp 0.0269 |
| B zero-filled, **A not** | **PASS**, samp 0.0269 |
| **neither** zero-filled | FAIL, samp 30.5 / 162.7 / 410.8 at T = 8/16/32 |

The reason is that the only cells whose contents can affect a *written* output
are the ones in the partial **k** tile, and there `aCol ≥ K` and `bRow ≥ K` fire
together on the same `k` index. `0 * garbage` is 0, so one zero kills the term.
The `row ≥ M` and `col ≥ N` cases never affect a written output, because the
thread that would read them writes nothing.

Do not rely on this. Two reasons:

1. On the **first** tile the un-zero-filled slot is not stale, it is
   *uninitialised shared memory*. Its contents are undefined, and `0.0f * inf`
   is `NaN`, not 0. This kernel never hits that case at these dimensions
   (`aCol ≥ K` can only occur on the last tile) but a kernel with `K < T` does,
   immediately.
2. It couples two pieces of code that have no business being coupled. The next
   person who changes the A-side guard silently breaks the B side.

Write both. The measurement is here because "my test passes" is not the same
claim as "my kernel is correct", and this is a case where the gap is visible.

---

## TODO 2 — the B-tile cooperative load

```cpp
const int bRow = k0 + ty;      // ty walks the CONTRACTION axis of B
Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
```

**This is not TODO 1 with the letters swapped, and that is the point of the
exercise.** In A, `threadIdx.x` indexes the contraction dimension. In B,
`threadIdx.x` indexes the output column and `threadIdx.y` indexes the
contraction dimension. The two load mappings are genuinely different functions
of `(tx, ty)`, and neither of them is the compute mapping.

The reason is a layout fact, not a convention: `k` is A's **fast** axis
(`A[r*K + k]`) and B's **slow** axis (`B[k*N + c]`). Coalescing requires that
whatever varies with `tx` be the fast axis of the array being read. For A that
forces `k` onto `tx`; for B it forces the column onto `tx`. The contraction
index therefore has to move to `ty` in the B load. Module 6 called this "the
load mapping is not the compute mapping"; a GEMM has *two* load mappings and
they are not each other.

**The common wrong answers and what each produces:**

| wrong version | symptom |
|---|---|
| `Bs[ty][tx] = B[(size_t)(k0+tx)*N + (col0+ty)]` (mappings swapped) | wrong answer everywhere, Freivalds ~10², and 32 sectors per warp instead of 4 |
| `Bs[tx][ty] = B[(size_t)bRow*N + col]` (slot transposed) | wrong answer; also turns a degree-1 shared store into a degree-`T` one |
| guard `bRow < K` omitted | reads `B` up to `T*N` floats past the end; `compute-sanitizer --tool memcheck` reports it immediately |
| guard `col < N` omitted | reads into the next row of B — no fault, wrong values, silent |

---

## TODO 3 (parts 2 and 3) — the two barriers

```cpp
__syncthreads();        // RAW: every cell written before any is read
...accumulate...
__syncthreads();        // WAR: every cell read before any is rewritten
```

**The first barrier (read-after-write).** Thread `(0,0)` reads `As[0][k]` for
every `k` in `[0, T)`. Those cells were written by threads `(0,0) .. (T-1,0)`.
At `T = 32` that is the whole first warp; at `T = 16` those threads are
`tid = 0..15`, which are in the same warp as `(0,0)` — but `Bs[k][0]` for
`k = 0..15` was written by threads `tid = 0, 16, 32, ..., 240`, which span
**eight warps**. Without a barrier, warp 0 can run ahead and read cells warp 7
has not written. This barrier needs **both** halves of `__syncthreads()`: the
execution barrier so the writes have happened, and the block-scope memory fence
so they are *visible* (Module 9, guarantee G1 + G2).

**The second barrier (write-after-read).** At the top of iteration `t+1`, every
thread overwrites `As[ty][tx]` and `Bs[ty][tx]`. A fast warp can reach that
store while a slow warp is still reading iteration `t`'s tile. No new data is
involved — nothing is being published — so this barrier needs only guarantee
**G1**, the execution barrier. That asymmetry is exactly why this is the barrier
you can design away: give the block two tile buffers and alternate between
them, and there is no write-after-read to order. Example 2 measures the
double-buffered version.

**Measured, with the WAR barrier deleted and no second buffer:**

| T | result | Freivalds | sampled |
|---|---|---|---|
| 8 | FAIL | 1.35 | 61.1 |
| 16 | FAIL | 0.59 | 58.6 |
| 32 | FAIL | 18.8 | 38.8 |

Two things to notice. First, it fires at every tile size here, *including*
`T = 8` where the block is only two warps — which is worth knowing, because the
usual folklore is "you only need the second barrier for big blocks." Second, at
`T = 16` the **Freivalds probe alone would have passed it** (0.59 ≤ 1) and only
the sampled per-element check caught it. A race corrupts a small number of
elements in a data-dependent pattern; a row-summed probe partially cancels it.
That is why the validator runs three checks and not one.

Example 2 measures the same kernel in the timing harness and finds
**21 000 wrong elements out of 1 594 935 at T = 16 and 39 000 at T = 32** — i.e.
98.7 % of the matrix is correct. A test that checks a handful of elements and
declares victory will ship this.

---

## TODO 4 — the store and the `beta == 0` contract

```cpp
if (row < M && col < N) {
    if (beta == 0.0f) C[(size_t)row * N + col] = alpha * acc;
    else              C[(size_t)row * N + col] = alpha * acc
                                               + beta * C[(size_t)row * N + col];
}
```

The guard here rather than at kernel entry, for the barrier reason above.

The `beta` branch is Module 16's, unchanged: when `beta` is exactly zero the
BLAS specification says `C` is not read, and the harness fills it with
`+infinity` to prove it. `0.0f * inf` is `NaN`, so the unconditional form
produces 1 594 935 non-finite elements and the finiteness check catches every
one. The branch is warp-uniform — `beta` is a kernel argument — so by Module 8
it costs one predicate and nothing else.

Using `col < M` instead of `col < N` (the classic) leaves columns 1035..1540
never written. The `+infinity` poison survives, and the finiteness check reports
**523 710 non-finite elements**. Note that this defect is *invisible* on a
square problem, which is why `M ≠ N` is not negotiable.

---

## TODO 5 — the prediction

**Bucket 2: 1.0×–1.5× faster.** Measured **1.29–1.32×**, best at `T = 16`.

The reasoning that gets you there in one sentence: the naive inner loop is
`LDG, LDG, FFMA` and the tiled inner loop is `LDS, LDS, FFMA`, so the *count* of
memory instructions per multiply-add is unchanged at two, and what you have
bought is only the difference in cost between an L1-or-L2-hitting global load
and a shared load. Module 16 had already measured that ≥ 92 % of the naive
kernel's requested bytes never reached DRAM, so there was never a large traffic
win available.

Readers who predict bucket 4 or 5 are reasoning from the traffic ledger — global
loads really do fall by a factor of `T/2` — and the traffic ledger is not what
binds. This is Module 6's finding in its strongest form and it is the reason
Module 18 exists.

---

## Synchronization / memory reasoning

The accumulator is the conceptual centre of the kernel. `acc` is a **register**,
private to one thread, and it is alive across the whole tile loop. It has to be:
shared memory holds a `T × T` tile, and the dot product needs the complete row
of A and the complete column of B — `K = 1063` elements each, which is 4252
bytes per operand per output and cannot be resident. The tile loop is therefore
a *partial-sum* loop: each iteration adds `T` of the `K` terms, and the only
state that crosses an iteration boundary is one float per thread. Everything in
shared memory is scratch and is legitimately destroyed every iteration — which
is precisely why the WAR barrier exists.

`-Xptxas -v` for the shipped solution, `T = 16`:

```
ptxas info : Used 37 registers, used 1 barriers, 2048 bytes smem, 400 bytes cmem[0]
             0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

2048 B = two 16×16 float tiles. `used 1 barriers` is the barrier *resource*
count, not the number of `BAR.SYNC` instructions; the SASS contains two.

---

## Performance reasoning

Measured, min of 5 rotated sweeps after a 1500 ms stream + 500 ms FFMA warm-up:

| config | ms | GFLOP/s | × naive |
|---|---|---|---|
| naive (Module 16) | 2.560 | 1324 | 1.00 |
| tiled T = 8 | 2.475 | 1370 | 1.03 |
| **tiled T = 16** | **1.971** | **1721** | **1.30** |
| tiled T = 32 | 2.157 | 1572 | 1.19 |

Absolute figures move ±3 % run to run and can collapse by 2–3× if the sweep is
run immediately after another GPU-heavy process; the **ratios** reproduce to
about ±0.02.

`T = 8` is nearly worthless: 64 threads per block is two warps, the barrier
convoys them constantly, and the FMAs-per-global-load ratio is only 4. `T = 32`
loses to `T = 16` despite twice the reuse, because 1024 threads per block means
**one block per SM** (1536 threads/SM ÷ 1024 = 1) and 66.7 % occupancy, and
because every barrier now convoys 32 warps instead of 8. Exercise 2 does this
arithmetic properly.

---

## Expected output

```
=== Module 17 / Exercise 1 - the tiled GEMM ===
C(1035 x 1541) = alpha*A(1035 x 1063)*B(1063 x 1541) + beta*C
M % 8/16/32 = 3/11/11  N % 8/16/32 = 5/5/5  K % 8/16/32 = 7/7/7

-- correctness -----------------------------------------------------
  dataset 0 (positive [0.5,1.5)):
    naive (Module 16)  nonfin       0 | Freiv 0.0005871 | samp   0.02686 | a/b   0.02686 | PASS
    tiled T=8          nonfin       0 | Freiv 0.0005871 | samp   0.02686 | a/b   0.02686 | PASS
    tiled T=16         nonfin       0 | Freiv 0.0005871 | samp   0.02686 | a/b   0.02686 | PASS
    tiled T=32         nonfin       0 | Freiv 0.0005871 | samp   0.02686 | a/b   0.02686 | PASS
  dataset 1 (zero-mean [-1,1)):
    naive (Module 16)  nonfin       0 | Freiv  3.16e-05 | samp  0.001395 | a/b  0.001469 | PASS
    tiled T=8          nonfin       0 | Freiv  3.16e-05 | samp  0.001395 | a/b  0.001469 | PASS
    tiled T=16         nonfin       0 | Freiv  3.16e-05 | samp  0.001395 | a/b  0.001469 | PASS
    tiled T=32         nonfin       0 | Freiv  3.16e-05 | samp  0.001395 | a/b  0.001469 | PASS

  gamma_K = 6.336e-05 at K = 1063.

-- timing ----------------------------------------------------------
  config                    ms     GFLOP/s    x naive
  naive (Module 16)     2.5603      1324.4       1.00
  tiled T=8             2.4753      1369.9       1.03
  tiled T=16            1.9706      1720.7       1.30
  tiled T=32            2.1573      1571.8       1.19

  best tiled / naive = 1.299x -> bucket 2. You predicted 2. CORRECT

SCORE: 7/7
OVERALL: PASS
```

Note the **identical error figures for the tiled and naive kernels** on both
datasets. Reordering the summation into tiles does not change the answer here at
all: the terms are all positive in dataset 0, and in dataset 1 the reordering is
a permutation of the same additions whose worst-case bound is unchanged.
Zero-padding contributes exact zeros. A tiled GEMM is not less accurate than a
naive one — if anything, blocking a long accumulation is a mild *improvement*
(Module 12's tree-vs-serial result), though at `T = 16` over `K = 1063` the
effect is far below the measurement floor.

---

## The result that matters

Tiling a GEMM is the textbook's flagship example of shared memory paying off,
and on this machine it is worth **1.30×** — not 5×, not 10×. The traffic
argument is correct (global load instructions really do fall by a factor of
`T/2 = 8`) and it is *not what binds*, because the naive kernel's loads were
already being served on chip and because the tiled inner loop still issues two
memory instructions per multiply-add, just with a different opcode. Every
instinct that says "shared memory is fast, so use shared memory" is answered by
this number. The move that actually changes the instruction mix is giving each
thread more than one output, and Exercise 2 computes how much of one you need.

**Variation to try:** delete the WAR barrier and run the harness ten times.
Record how many of the ten runs report `PASS` at `T = 8`, and how the Freivalds
figure moves relative to the sampled figure. Then run
`compute-sanitizer --tool racecheck` on the same binary and compare what it
reports against what the validator caught.
