# Module 16 — Naive GEMM

> Prerequisites: Modules 1–13. Module 5 (sectors and coalescing) and Module 6 (the reuse-vs-cache argument) are load-bearing; Module 11's compulsory-traffic discipline is the backbone of the analysis.
> What this module gives you: the general matrix multiply as a problem statement, the simplest correct CUDA kernel for it, a validation methodology that is actually capable of failing, and a quantitative account of why that kernel reaches roughly 6–10 % of this GPU's FP32 throughput — precise enough that the next two modules are forced moves rather than tricks.

---

## Concept

### 1. The problem

**GEMM** — general matrix–matrix multiply — is the BLAS level-3 operation

```
C := alpha * op(A) * op(B) + beta * C
```

This module fixes `op = identity` and works entirely in **row-major**, fp32:

| matrix | shape | element (i, j) lives at |
|---|---|---|
| A | M × K | `A[i*K + j]` |
| B | K × N | `B[i*N + j]` |
| C | M × N | `C[i*N + j]` |

with

```
C[i][j] = alpha * sum over k in [0,K) of A[i][k] * B[k][j]  +  beta * C[i][j]
```

Three things about that definition earn their keep immediately.

**It is not square.** M, N and K are three independent dimensions. Everything
in this module and in Modules 17–18 runs at **M = 1027, N = 2053, K = 769** —
none of them a power of two, none a multiple of any block size used, all three
different. A square power-of-two test hides four distinct bug classes at once:
a missing boundary guard, a transposed result, a leading dimension confused
with a size, and a grid whose axes are swapped. With M ≠ N a transposed result
is not even the right shape.

**The leading dimension is not the width.** `A[i*K + k]` uses K as A's *stride*.
In real code a submatrix of a larger matrix has a stride larger than its width
— the BLAS parameter `lda` — exactly as Module 5's row pitch was larger than
the number of useful columns. This module keeps `lda = K`, `ldb = N`,
`ldc = N`, and says so, because Module 17 will have to keep track of them
separately.

**`beta == 0` means C is not read.** This is specified, not an optimization.
The reference BLAS says that when `beta` is exactly zero, `C` need not be set
on input. Callers rely on it: a freshly `cudaMalloc`'d `C` is uninitialised
device memory and may hold anything, including bit patterns that decode as NaN.
Writing

```cpp
C[idx] = alpha * acc + beta * C[idx];     // WRONG when beta == 0
```

propagates that NaN, because `0.0f * NaN` is NaN, not 0. The correct kernel
branches:

```cpp
if (beta == 0.0f) C[idx] = alpha * acc;
else              C[idx] = alpha * acc + beta * C[idx];
```

The branch is warp-uniform — `beta` is a kernel argument, identical for all
32 lanes — so by Module 8 it costs one predicate evaluation and nothing else.
This is the single most commonly botched line in hand-written GEMM kernels, and
it is botched *silently*: with a zeroed C it produces the right answer.

### 2. The naive kernel

One thread computes one element of C:

```cpp
const int col = blockIdx.x * blockDim.x + threadIdx.x;
const int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row >= M || col >= N) return;
float acc = 0.0f;
for (int k = 0; k < K; ++k)
    acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
C[(size_t)row * N + col] = alpha * acc;          // beta == 0 case
```

There is nothing else to it. You already have every ingredient: Module 3's 2-D
indexing and bounds guard, Module 2's launch and error checks. The difficulty is
entirely in the indexing — three matrices, three leading dimensions, and the
summation index `k` is the *fast* axis of A and the *slow* axis of B — and in
the launch configuration agreeing with the axis assignment inside the kernel.

### 3. Counting the traffic, twice

Module 11 defined **compulsory traffic**: the bytes that must cross the DRAM
pins at least once, counting each distinct element read once regardless of how
many times the source names it. For a GEMM with `beta == 0`:

```
compulsory = 4 * (M*K + K*N + M*N) bytes
```

At M = 1027, N = 2053, K = 769 that is **17.91 MB**.

Now count what the naive kernel actually *asks for*. There are M·N threads;
each runs K iterations; each iteration issues one 4-byte load from A and one
from B:

```
requested = 4 * 2 * M * N * K bytes
```

which is **12.97 GB** — **724× the compulsory figure**. For a square n×n×n
problem the ratio is `2n³·4 / (12n²) = 2n/3`, so it grows linearly with the
problem: at n = 4096 the naive kernel requests 2730× the compulsory traffic.

The useful work is fixed at **2·M·N·K FLOPs** (one multiply and one add per
term). Divide it by each traffic figure and you get two very different
arithmetic intensities:

| model | intensity | for a square n |
|---|---|---|
| compulsory | 2MNK / (4(MK+KN+MN)) | **n / 6** FLOP/byte |
| naive request | 2MNK / (8MNK) | **0.25** FLOP/byte, for every n |

Those two numbers are the whole module. The **algorithm** has an arithmetic
intensity that grows without bound with problem size. The **naive
implementation** has an arithmetic intensity of one quarter, forever. GEMM is
the canonical compute-bound kernel *if and only if* you arrange to get the
reuse; the naive kernel does not arrange it, and asks the memory system for
0.25 FLOP/byte's worth of service.

### 4. Machine balance, and where each of those lands

The **machine balance** of a processor is the arithmetic intensity at which its
compute ceiling and its memory ceiling are hit simultaneously. Below it a
kernel is memory-bound; above it, compute-bound. You need two measured numbers.

**Do not compute the FP32 peak from `cudaDevAttrClockRate`.** Spec §12 rule 6:
this GPU reports 1.545 GHz and runs at 1.48–2.04 GHz. A "peak" of
40 SM × 128 FP32 lanes × 2 FLOP × 1.545 GHz = 15 821 GFLOP/s is not a peak —
`example02.cu` *measures* **≈18 250 GFLOP/s**, which is 1.15× the alleged
maximum. Module 1 recovered the true clock with `clock64()`; that works for a
single resident warp (it reads 2.12 GHz here) but not under a full machine
load, where block 0's cycle count is no longer the kernel's duration. The
robust method, and the one this module uses, is to **measure the ceiling
directly**: eight independent FFMA chains per thread, one wave, min-of-N. The
implied clock falls out as `ceiling / (40 · 128 · 2)` and comes to **1.78 GHz**
— which is the honest number, measured under exactly the load it describes.

For the memory ceiling, the course already has one: Module 12's properly-warmed
streaming read, **410.5–410.7 GB/s = 95 % of the 432 GB/s pin peak**, after a
**1500 ms** warm-up. `example02.cu` reproduces that figure — but only sometimes.
Across one authoring session the same binary measured **294–411 GB/s**, with
`nvidia-smi` showing the memory clock at 8801 MHz and the **SM clock dropping to
285 MHz** during the streaming kernel. This laptop part's power manager will
trade SM clock away on a kernel that is not issuing arithmetic, and no warm-up
length fixes it. Take the best figure as the ceiling and treat the rest as
noise; and when you need a *bound* rather than a *ceiling*, use the 432 GB/s pin
peak, which is a property of the bus and cannot be throttled away.

```
machine balance = 18250 GFLOP/s / 410.7 GB/s = 44.4 FLOP/byte
```

(Module 12 quoted ~42 FLOP/byte for the same GPU from a shorter measurement;
the two agree.)

Now place both intensities:

| model | intensity | vs machine balance | roofline says |
|---|---|---|---|
| compulsory, 181 FLOP/byte | 4.1× the balance | compute-bound | 18 250 GFLOP/s |
| naive request, 0.25 FLOP/byte | 1/178 of the balance | memory-bound | 103 GFLOP/s |

And the measurement: the naive kernel runs at **1275–1345 GFLOP/s** on this
problem. That is **6–10 % of the roof its algorithm entitles it to** — and
simultaneously **12–16× faster than the roof its request pattern implies.**

Both statements are true, and the second is the more interesting one. It is
the caches, quantified: **Module 21** develops the roofline properly.

### 5. What the caches actually recovered — and how to measure it without a profiler

`ncu` cannot run on this machine (`ERR_NVGPUCTRPERM`), so no cache-hit counter
is available. You can still get a rigorous **bound**, from two things a
stopwatch can see.

In `t` seconds, DRAM can deliver at most `BW_ceiling · t` bytes. If the kernel
asked for `R` bytes and finished in `t`, then at least `R − BW·t` of them came
from somewhere else:

```
on-chip fraction  >=  1 - (BW_ceiling * t) / R
```

Use the **pin peak** here, not the measured streaming figure: a bound needs an
upper bound on what DRAM can deliver, and 432 GB/s is one by construction while
the measured 294–411 GB/s is not. Measured: R = 12.97 GB, t = 2.41 ms,
BW ≤ 432 GB/s → `1 − 1.04 GB / 12.97 GB` = **at least 92 % of the requested
bytes were serviced on chip.** Not "probably". At least.

This is deliberately a loose bound on this problem, and the looseness is
instructive: the entire working set is 17.91 MB and the L2 is **48 MB**, so
after the first touch essentially *all* the traffic is on chip and the true
figure is close to 99.9 %. That is Module 6's finding in its strongest form —
*the caches got there first*. Module 6 measured a 5-point stencil where L1/L2
had already collapsed a predicted reuse factor of 5 down to ≈1.4, so tiling it
was a 0.85× net loss. Here the predicted reuse factor is **724**, and the caches
have already collapsed nearly all of it.

Which raises the question this module exists to answer: **if the caches already
captured the reuse, why is the kernel still at 7 % of peak?**

### 6. Per-warp access analysis, and the mapping that costs an order of magnitude

Apply Module 5's sector-counting procedure to one warp on one iteration of the
`k` loop. A warp is 32 consecutive linearized thread ids (Module 3):
`tid = threadIdx.x + blockDim.x * threadIdx.y`, so with `blockDim.x = 32` a warp
is 32 consecutive `threadIdx.x` at one `threadIdx.y`.

**Mapping "x → col"** (`col` from `threadIdx.x`), block (32, 8):

- `A[row*K + k]`: `row` is the same for all 32 lanes. **One address. One
  sector.** This is a broadcast, and Module 4 and Module 7 both priced a
  broadcast at one access.
- `B[k*N + col]`: 32 consecutive `col`, so 32 consecutive floats = 128 B =
  **4 sectors**, perfectly coalesced.
- **5 sectors = 160 B** to feed 32 lanes × 2 operands = 256 B of demand.

**Mapping "x → row"** (`row` from `threadIdx.x`), block (32, 8):

- `A[row*K + k]`: 32 different rows, addresses `K*4 = 3076 B` apart. **32
  distinct sectors.**
- `B[k*N + col]`: `col` is the same for all lanes. **1 sector.**
- **33 sectors = 1056 B** for the same 256 B of demand.

Same kernel. Same arithmetic. **Byte-identical SASS** — `cuobjdump` shows both
compile to 59 `LDG.E`, 36 `IMAD.WIDE`, 30 `FFMA` in the unrolled body. The only
difference is which index moves with the fast axis of the warp. 33/5 = 6.6× the
sectors.

Module 3 Exercise 1 taught exactly this trap on a stencil, where swapping the
axes left 208 863 pixels unwritten and `blockDim.x == 1` cost 3.1×; Module 3
Exercise 2 measured the axis assignment alone at 3.5× on a triple-nested port.
Here it costs **3.3–3.9× measured** on this problem shape and **8.2×** on a
1024³ one (13.12 ms vs 1.60 ms, measured). Exercise 2 makes you count it and then measure it.

The sharpest form of the lesson is what happens when you vary `blockDim.x`.
Measured, 256 threads per block throughout:

| blockDim.x | x → col sectors | x → col ms | x → row sectors | x → row ms |
|---|---|---|---|---|
| 1 | 33 | 8.66 | 5 | 3.79 |
| 2 | 17 | 5.31 | **4** | **2.60** |
| 4 | 9 | 3.35 | 5 | 2.73 |
| 8 | 5 | 2.48 | 9 | 3.58 |
| 16 | **4** | **2.40** | 17 | 5.65 |
| 32 | 5 | 2.42 | 33 | 9.22 |

The sector model orders all twelve configurations correctly. And the "bad"
mapping is **completely repaired by setting `blockDim.x = 2`** — because then
`threadIdx.y` is what varies across a warp, and `col` varies with `threadIdx.y`.
The mapping was never about the names `row` and `col`. It is about which index
moves with the fast axis of the warp linearization, and `blockDim` decides that
just as much as the kernel body does.

Note also what the model does *not* get right: it predicts a 33/4 = 8.25× spread
and the measurement is 3.3–3.9×. Sectors *requested* are not sectors *fetched*.
The 32 scattered A-sectors are re-requested by every warp in the grid and mostly
hit in L2. The count is an upper bound on the damage, not a prediction of it.

### 7. The actual bottleneck: two loads per FMA

Fix the mapping and you are at ~1350 GFLOP/s, which is 7.4 % of the measured
FP32 ceiling. The traffic argument cannot explain the remaining 13×: we just
established that 92 %+ of the requests never reach DRAM.

Look at the instruction mix instead. The inner loop is

```
LDG.E  (A)        one 32-bit global load
LDG.E  (B)        one 32-bit global load
FFMA              one fused multiply-add
```

plus an `IMAD.WIDE` per B load, because B's stride `N` is a runtime value the
compiler cannot strength-reduce away. **Two global load instructions per
multiply-add.** The FP32 pipes can retire 128 FMA lanes per SM per cycle; the
LSU and the L1 return path cannot come close to feeding them at a 2:1 ratio.

`example02.cu` measures this directly with a probe kernel that performs
*exactly the same two loads* and then issues R FFMAs against them instead of 1.
The arithmetic is meaningless; the instruction mix is the point. Measured:

| R | loads per FMA | ms | GFLOP/s | % of ceiling |
|---|---|---|---|---|
| 1 | 2.00 | 2.59 | 1252 | 9.5 % |
| 2 | 1.00 | 2.51 | 2583 | 19.6 % |
| 4 | 0.50 | 2.57 | 5046 | 38.2 % |
| 8 | 0.25 | 3.16 | 8212 | 62.2 % |

**Eight times the arithmetic for 22 % more time** (10–22 % across runs). The FP32 pipe was idle; the
loop was waiting on the load path the entire time. Throughput is very nearly
proportional to FMAs-per-load, which is the signature of a kernel whose clock
is set by one resource and one only.

Extrapolate: reaching 80 % of the compute ceiling needs roughly **6.4 fused
multiply-adds per global load.** A one-element-per-thread GEMM supplies exactly
**0.5** — two loads, one FMA — and *no* choice of block shape, grid shape,
mapping, or launch parameter changes that number. It is a property of the
decomposition, not of the code.

That is the whole setup for the rest of Part V. There are exactly two things
you can do about a 2:1 load-to-FMA ratio:

1. **Make each operand load cheap.** Stage a tile of A and a tile of B in the
   block-scoped scratchpad and read them with `LDS` instead of re-issuing
   `LDG`. This is **Module 17**. Note carefully what it does *not* do: it does
   not change the *count* of memory instructions per FMA. Module 6 already told
   you that a kernel whose instruction mix is unchanged does not move.
2. **Make each loaded value feed several FMAs.** Give each thread several
   output elements so one loaded A value multiplies several B values already in
   registers. This is **register blocking / thread coarsening**, and it is
   **Module 18** — the ingredient Module 6 named by name as "the missing one".

The measured 6.4-versus-0.5 is how much of (2) you need. Neither module is a
trick; both are the only available moves.

### 8. Validating a floating-point GEMM

This is usually done badly, and a GEMM test that cannot fail is worse than no
test, because it licenses everything downstream. Four separate problems.

**(a) You cannot compare bit-exactly.** The GPU sums K terms in a different
order than any CPU reference, and fp32 addition is not associative (Module 12).
Some tolerance is required.

**(b) The house tolerance is wrong for GEMM.** `fabs(a-b) <= 1e-5 * max(1,|b|)`
is a relative test against |C|. The standard backward-error bound for a
length-K inner product accumulated at unit roundoff `u` is

```
| c_hat - c |  <=  gamma_K * S ,    S = sum_k |a_ik| * |b_kj| ,
gamma_K = K*u / (1 - K*u) ,          u = 2^-24 = 5.96e-8 for fp32
```

At K = 769, `gamma_K = 4.58e-5` — already **4.6× looser than the house
tolerance** before anything goes wrong. A correct kernel at large K fails the
house test. This is Module 12's `2^26 ones saturating at 2^24` hazard wearing a
different hat: long sequential fp32 accumulation is exactly what a GEMM's inner
loop is.

**(c) Relative-to-|C| is the wrong yardstick anyway.** With zero-mean operands
the K terms cancel: `|C| ~ sqrt(K)·sigma²` while `S ~ K·sigma²`. Measured here,
same correct kernel, same K:

| dataset | worst relative error vs \|C\| | worst error / (gamma_K · S) |
|---|---|---|
| positive operands | 1.6e-6 | 0.035 |
| zero-mean operands | **9.7e-4** | 0.0037 |

The relative-to-|C| number moves by 600× between two datasets on which the
kernel is *equally correct*. The scaled number does not. **Scale the tolerance
by S, not by |C|.**

**(d) A loose tolerance must still catch real bugs.** With
`tol = 4 · gamma_K · S`, measured margins on the same problem:

| defect | error / tolerance |
|---|---|
| correct kernel | 0.009 (positive), 0.0008 (zero-mean) |
| `k` loop stops one short | 16× / 29× — **rejected** |
| B indexed column-major | 236× / 1465× — **rejected** |
| guard tests `col < M` | 1 053 702 elements never written — **rejected** |
| `beta` applied when `beta == 0` | 2 108 431 non-finite — **rejected** |

Two orders of magnitude of headroom on a correct kernel, and still one to three
orders of margin on the tightest planted defect.

#### The methodology this module establishes (Modules 17 and 18 reuse it)

Three checks, **in this order**:

1. **Finiteness / writtenness, over all M·N elements.** Prefill C with a value
   no correct kernel can produce (`+inf` works, and has the second property
   that `0.0f * inf` is NaN, so it catches both "nobody wrote this" and "you
   read C when `beta == 0`"). **This check must come first**: a NaN compares
   false against everything, so a max-error loop run first silently *passes* an
   array full of NaNs. `example01.cu` demonstrates exactly that.
2. **A Freivalds probe, full coverage, O(MN + KN + MK).** Pick a non-negative
   pseudo-random vector `v` and check, in double,
   `C·v == alpha·A·(B·v) + beta·C0·v`, with the tolerance propagated through
   the same contraction: `gamma_K · |alpha| · (|A|(|B||v|))_i`. Every element of
   C participates. The cost is the same order as the problem's *compulsory
   traffic* — you can afford to run it on every size you test. `v` must be
   non-negative so that a systematic per-element error accumulates coherently
   along the row instead of cancelling like a random walk.
3. **A sampled exact reference.** Recompute a strided sample of (i, j) in
   double with the exact `S_ij` and threshold `err / (gamma_K · S_ij)`. Gives
   an exact per-element number and a location to look at.

Plus a conditioning decision: test on **both** a strictly positive dataset (so
`|C| ≈ S` and errors are visible as relative errors) and a zero-mean one (so
the test is forced to be scaled correctly). And report the **headroom**, not
just PASS: a test that passes at 0.009 of tolerance is telling you something
different from one that passes at 0.98.

### 9. cuBLAS as the reference point

`cublasSgemm` is the number to be measured against. It is column-major, and
getting the call right is a standard trip hazard.

A row-major M×N matrix with leading dimension N is, **byte for byte**, the
column-major N×M matrix Cᵀ with leading dimension N. Nothing needs to move. So
the row-major identity `C = A·B` is the column-major identity

```
C^T = B^T * A^T
```

and cuBLAS computes exactly that with **no transpose flags** if you swap the
operands:

```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,                 // m, n, k of the column-major call
            &alpha, dB, N,           // 'A' := B, lda = N
                    dA, K,           // 'B' := A, ldb = K
            &beta,  dC, N);          // ldc = N
```

The instinct to reach for `CUBLAS_OP_T` is the mistake: that asks cuBLAS to
transpose data that is already in the layout it wants, and on a square problem
it produces a plausible-looking wrong matrix. Nothing is transposed and nothing
is copied here — the same bytes are read under the other convention.

Measured on this GPU, fp32, no Tensor Cores on either side:

| kernel | ms | GFLOP/s | % of cuBLAS | % of the 18 250 GFLOP/s ceiling |
|---|---|---|---|---|
| naive, best mapping and shape | 2.41–2.55 | 1275–1345 | 12–13 % | 7.0–7.4 % |
| `cublasSgemm` | 0.37–0.40 | 8150–8705 | 100 % | 45–48 % |

**The naive kernel is at 12–13 % of cuBLAS and ~7 % of the machine's measured
FP32 ceiling.** cuBLAS reaches 45–48 % of that ceiling on this awkward
non-power-of-two shape; the remaining half is what Modules 17, 18, 33–34 and 43
are about. **Module 36** owns cuBLAS and the library ecosystem; here it is a
ruler.

---

## Hardware Mental Model

**Why the mapping matters so much, in terms of the memory pipeline.** A warp's
load instruction presents 32 addresses to the L1. The coalescer reduces them to
the minimum covering set of 32-byte sectors (Module 5), and the L1 returns
data at a fixed width per cycle. A 5-sector warp request and a 33-sector warp
request take the same issue slot and occupy the return path for very different
numbers of cycles. Nothing about the *instruction* changed — which is why the
SASS is identical — and everything about the *service time* did.

**Why the broadcast is free.** When all 32 lanes present the same address the
coalescer emits one sector and the crossbar fans the word out to 32 lanes. This
is the same mechanism Module 4 measured for constant memory (broadcast cheap,
lane-varying 24.6× penalty) and Module 7 measured for shared memory (same bank,
same word = broadcast; same bank, different word = replay). GEMM's A access
under the good mapping is a global-memory broadcast, and it costs one sector.

**Where the naive kernel's time actually goes.** With `blockDim = (16,16)` the
kernel uses 40 registers and no shared memory, so occupancy is not the
constraint — the occupancy API reports the maximum 6 blocks/SM at 256 threads.
The SASS inner loop is unrolled far enough to have ~16 loads in flight per
thread, so memory-level parallelism (Module 11) is not the constraint either.
What *is* the constraint is the ratio: every FFMA is preceded by two `LDG`s and
an `IMAD.WIDE`, so at best one instruction in four is arithmetic, and the LSU
and L1 return path saturate long before the FP32 lanes do. The probe in §7
proves it by holding the loads fixed and varying only the arithmetic: the time
barely moves.

**Why the L2 is doing so much work and it still does not help.** 48 MB of L2
is large enough to hold this entire problem. Every one of the 12.97 GB of
requests is being serviced, and ≥92 % of them on chip. But an L2 hit is still
~241 cycles (Module 4), it still consumes an L1 miss slot, it still occupies
the LSU, and it is still one instruction per operand. **A cache reduces the
cost of a memory instruction; it does not reduce the number of them.** That
sentence is the reason Modules 17 and 18 exist and the reason Module 6's
stencil experiment came out the way it did.

**Why the intensity grows with n and the implementation's does not.** The
mathematical object has O(n³) work over O(n²) data. Every element of A is
needed by N different outputs and every element of B by M different outputs.
The reuse is *in the problem*. A one-element-per-thread decomposition throws it
all away at the thread boundary: thread (i, j) shares its entire A row with the
N−1 threads in its row of C and its entire B column with the M−1 threads in its
column, and has no mechanism to exploit that except to hope for a cache hit.
The reuse factor available to a *block* that cooperates is the tile dimension;
the reuse factor available to a *thread* that owns several outputs is the
number of outputs it owns. Modules 17 and 18 collect them, in that order.

**Ada specifics, labelled.** **ARCHITECTURE-SPECIFIC:** 32-byte sectors, 128-byte
lines, 48 MB L2, 128 FP32 lanes/SM, 40 SMs, the measured 18 250 GFLOP/s and
410.7 GB/s ceilings and the 44 FLOP/byte balance derived from them.
**PORTABLE CUDA CONCEPT:** the compulsory-vs-requested traffic distinction, the
arithmetic-intensity argument, the loads-per-FMA argument, the warp-level
address analysis, the GEMM error bound, and the cuBLAS column-major convention.

---

## Code Walkthrough

### `example01.cu` — the kernel, the contract, and the test

**§A** prints the ledger from §3 above, computed at runtime, so the 724× and
the two intensities are not numbers you have to take on faith.

**§B** is the kernel. The line worth staring at is the store:

```cpp
if (beta == 0.0f) C[(size_t)row * N + col] = alpha * acc;
else              C[(size_t)row * N + col] = alpha * acc
                                           + beta * C[(size_t)row * N + col];
```

The harness fills C with `+infinity` before every `beta == 0` launch. A kernel
that writes `alpha*acc + beta*C` unconditionally produces `0.0f * inf` = NaN in
**all 2 108 431 elements** and is caught. With C zeroed instead of poisoned it
would pass.

**§C** is `gemmValidate()`, the function Modules 17 and 18 will reuse verbatim.
Read the ordering comment; the finiteness pass has to be first. The Freivalds
tolerance is the interesting line:

```cpp
double tol = fabs(alpha) * (gammaK + 4.0*u) * ybound
           + 4.0 * u * fabs(beta) * c0v;
```

`ybound` is `(|A|(|B||v|))_i` — the same contraction as the value being
checked, with absolute values throughout. That is what makes the bound exact
rather than a guess: it is the error bound for the inner product, summed along
the row with the same weights.

**§D** runs five kernels — one correct and four defective — through the
validator and prints what each check saw. The two NaN-poisoned failures report
Freivalds = 0 and sampled = 0: those checks *passed*, because NaN compares false
against everything. That row is the reason the finiteness check exists.

**§E** derives and performs the cuBLAS swapped-argument call, validates its
result with the same function, and times it against the naive kernel with a
1500 ms warm-up and rotated min-of-N sweeps.

### `example02.cu` — where the FLOPs went

**§A** measures both ceilings. Note the warm-up: 1500 ms of the *streaming*
kernel (which ramps the memory P-state — spec §12 rule 4) followed by 500 ms of
the FFMA kernel (which ramps the SM clock without letting the memory clock fall
back). Warming with only one of them understates the other ceiling by 10–25 %;
this was verified while writing the module and the intermediate numbers are in
the solution notes.

**§B** implements the sector count by literal enumeration:

```cpp
for (int lane = 0; lane < 32; ++lane) {
    int tx = lane % bx, ty = lane / bx;
    int row = (mapping == 0) ? ty : tx;
    int col = (mapping == 0) ? tx : ty;
    addrA[lane] = ((long long)row * K + 0) * 4 / 32;
    addrB[lane] = ((long long)0 * N + col) * 4 / 32;
}
```

This is Module 5's procedure with no GEMM in it: lane → element → byte → sector
→ count distinct. The warp decomposition `tx = lane % bx, ty = lane / bx` is
Module 3's linearization rule read backwards, and it is why `blockDim.x = 8`
makes a warp span four rows of the block.

**§C** times six configurations back-to-back in one rotated sweep with
`SWEEPS = NCFG`, per spec §12 rules 1, 3 and 9.

**§D** computes the on-chip service bound of §5.

**§E** is the loads-per-FMA probe. The probe kernel is explicitly *not* an
optimization and computes nothing meaningful:

```cpp
const float av = A[(size_t)row*K + k];
const float bv = B[(size_t)k*N + col];
#pragma unroll
for (int r = 0; r < R; ++r) acc[r] = fmaf(av, bv, acc[r]);
```

The same two loads; R FFMAs instead of one. This is the module's decisive
measurement and it deliberately stops one step short of the real technique.

**§F** prints the roofline table with both traffic models.

---

## Check Your Understanding

1. A colleague replaces `C[idx] = alpha*acc + beta*C[idx]` with the
   two-branch form, and their GEMM starts returning different numbers in a
   downstream application even though `beta` was 0 in both versions and both
   pass a `cudaMemset(C, 0, ...)`-based unit test. Explain how this is
   possible, and give the specific circumstance in which the *original* code
   was the one producing the wrong answer.

2. §6's sector model predicts a spread of 33/4 = 8.25× across the twelve
   configurations and the measurement is 3.3–3.9×. Give two distinct
   mechanisms by which the requested-sector count can overstate the cost, and
   for each one, describe a change to the *problem size* (not the kernel) that
   would make the measured spread move toward the predicted 8.25×. Then say
   which of the two would make it move further.

3. The probe in §7 shows that issuing 8 FFMAs per loaded pair instead of 1 costs
   22 % more time and delivers 6.6× the throughput. A reader concludes: "so if I
   just make each thread compute 8 output elements, I get 6.6×." Name the
   resource that this conclusion ignores, say what would have to be true of the
   8 outputs for the conclusion to hold at all, and explain why the honest
   version of the claim requires *two* separate changes rather than one.

4. You are asked to validate a GEMM at K = 4 096 with operands drawn from a
   heavy-tailed distribution, where a few entries of A are 10⁶ times larger
   than the rest. Your `gamma_K * S` tolerance passes. A colleague points out
   that the *relative* error of some elements of C is 10 %. Both of you are
   right. Explain what has actually happened, decide whether the kernel should
   be accepted, and state what you would change about the *test data* — not the
   tolerance — to get a test that answers the question you care about.

Answers: `solutions/module16/check_your_understanding.md`.

---

## Exercises

### Exercise 1 — `exercise01.cu` — the naive kernel and the test that can fail it

Write the kernel, and write the tolerance law that judges it.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

| TODO | requirement |
|---|---|
| 1 | This thread's `(row, col)` in C, from the built-ins, plus the out-of-range guard. Neither M nor N is a multiple of the block dimensions. |
| 2 | The K-length dot product. Three matrices, three leading dimensions. |
| 3 | The store, honouring the `beta == 0` contract. C is prefilled with `+infinity` before every `beta == 0` launch. |
| 4 | The launch configuration, consistent with TODO 1's axis assignment. |
| 5 | **Design:** `gemmTolerance(K, S, ref)`. The harness probes it directly for shape — proportionality to S, growth with K, tightness — and then runs your kernel and four defective kernels through it on **two datasets with different conditioning**. |

Validation: 15 points. Tolerance shape 4, your kernel accepted on both
datasets 2, four defects rejected on both datasets 8, the α/β path 1.
`OVERALL: PASS` requires all of them.

### Exercise 2 — `exercise02.cu` — the mapping decision

Count first, predict second, measure third.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

| TODO | requirement |
|---|---|
| 1 | `predictSectors()`: the per-warp A-sector and B-sector counts, by enumerating warp 0 of block 0 at `k = 0`. Checked against a hash of the reference answers for eight (mapping, block) pairs. |
| 2 | The second kernel — `threadIdx.x` carries `row` — **and** its grid dimensions. The two must agree; Module 3 Exercise 1's trap is that the axis decision appears twice. |
| 3 | Two predictions, committed before building: the slowest/fastest spread across all twelve configurations, as a bucket; and which `blockDim.x` is fastest for the `x → row` mapping. |
| 4 | **Design:** `chooseBx()` — use *your own* model to pick the block shape, for each mapping, before any timing happens. Do not hard-code a measured answer; the harness prints your choice before it measures. |

Validation: 6 points — model hash, kernel-B correctness at six block shapes,
both predictions, and both model choices agreeing with the measurement to
within 5 %.

### Exercise 3 — `exercise03.cu` — the ledger, the roofline, and what has to change

Write the analysis, not the kernel. Five functions; the harness checks your
arithmetic against hashed references on fixed synthetic inputs and then applies
your functions to real measurements.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

| TODO | requirement |
|---|---|
| 1 | `compulsoryBytes(M, N, K)` |
| 2 | `requestedBytes(M, N, K)` — what the naive kernel's loads actually ask for |
| 3 | `machineBalance()` and `rooflineGflops()` from two measured ceilings |
| 4 | `minOnChipFraction()` — the bound of §5, from elapsed time and the DRAM ceiling alone, with the degenerate case handled |
| 5 | **Design:** `fmasPerLoadNeeded()` — given the measured loads-per-FMA family, the ratio required to reach a target throughput. You identify the relationship from the data; you are not told it. |

Validation: one point, awarded only if all eight probed answers are correct.
The program then prints the diagnosis in full, ending with the number of fused
multiply-adds per global load you would need and the number a
one-element-per-thread GEMM can ever supply.

---

## Prediction

Commit these to writing before you build anything.

1. The naive kernel requests **724×** the compulsory traffic of this problem.
   Write down the percentage of those requested bytes you expect to be served
   by L1 or L2 rather than DRAM, and — separately — the percentage of the
   machine's FP32 peak you expect the kernel to reach. Most readers get the
   first one roughly right and the second one wrong by a factor of five,
   because they assume the two are the same question.

2. Swapping which block axis carries `row` and which carries `col` changes the
   per-warp sector count from 5 to 33. Predict the measured time ratio, then
   predict what happens to that ratio if the matrices are made four times
   larger in every dimension. State which of your two predictions you are less
   sure of and why.

3. Before running Exercise 3: write down how many fused multiply-adds each
   loaded value would have to feed for this kernel to reach 80 % of the FP32
   ceiling, and how many it feeds now. Then write down, in one sentence, what
   has to change about the *decomposition* — not the code — to close the gap.
   If your sentence mentions shared memory and nothing else, re-read §7.
