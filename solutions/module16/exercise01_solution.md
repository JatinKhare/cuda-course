# Module 16 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

Warning-clean. Runs in a few seconds; most of that is the host-side double
reference over 8 385 sampled elements × 769 terms × 5 kernels × 2 datasets.

---

## TODO 1 — the thread's `(row, col)` and the guard

```cpp
const int col = blockIdx.x * blockDim.x + threadIdx.x;   // the N axis
const int row = blockIdx.y * blockDim.y + threadIdx.y;   // the M axis
if (row >= M || col >= N) return;
```

**Why this assignment and not the other.** Module 3's linearization rule says
`tid = threadIdx.x + blockDim.x * threadIdx.y`, so with `blockDim.x = 32` a warp
is 32 consecutive values of `threadIdx.x` at one `threadIdx.y`. Giving
`threadIdx.x` to `col` makes a warp span 32 consecutive columns of C, which
makes the `B[k*N + col]` access 32 consecutive floats — 128 B, 4 sectors — and
the `A[row*K + k]` access a single address shared by all 32 lanes, which is a
broadcast costing 1 sector. The other assignment costs 33 sectors instead of 5.
Exercise 2 measures it; this exercise only asks you to be consistent.

**Wrong approaches and their symptoms.**

- Guarding only one axis (`if (row >= M) return;`). The grid is
  `ceil(2053/32) × ceil(1027/8) = 65 × 129`, so `col` reaches 2079 and `row`
  reaches 1031. Writing `C[row*N + col]` with `col = 2053..2079` lands in the
  *next row* of C and with `row = 1027..1030` lands past the end of the
  allocation. The harness reports it as corrupted rows, or
  `cudaErrorIllegalAddress` from the synchronizing call. Both tell you
  immediately; the danger is that on a square power-of-two problem neither
  happens.
- `int col = blockIdx.x * blockDim.x + threadIdx.x;` with 32-bit arithmetic is
  fine here (2079 × 2053 fits), but the *addressing* is not: `row * K + k` at
  M = 100 000 would overflow a 32-bit int. The solution casts to `size_t`
  before the multiply, which is Module 3's rule.

## TODO 2 — the dot product

```cpp
float acc = 0.0f;
for (int k = 0; k < K; ++k)
    acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
```

Three leading dimensions, and `k` is the **fast** axis of A and the **slow**
axis of B. The three ways to get this wrong that the harness plants as defects
are exactly the three people write:

| wrong form | what it is | how it shows up |
|---|---|---|
| `B[(size_t)col * K + k]` | B read as if column-major | sampled error 236× (positive data), 1465× (zero-mean) |
| `A[(size_t)k * M + row]` | A read as if column-major | Freivalds 1004×, sampled 1236× |
| `for (k = 0; k < K-1; ...)` | off-by-one in the reduction | Freivalds 43×, sampled 63× |

Note how small the last one's margin is compared with the other two, and how it
is still an order of magnitude past the tolerance. That margin is what TODO 5 is
really being scored on.

## TODO 3 — the `beta == 0` contract

```cpp
if (beta == 0.0f) C[(size_t)row * N + col] = alpha * acc;
else              C[(size_t)row * N + col] = alpha * acc
                                           + beta * C[(size_t)row * N + col];
```

The BLAS specification says C need not be set on input when `beta` is exactly
zero. `alpha*acc + beta*C` evaluated literally reads C, and `0.0f * NaN` is
NaN, `0.0f * inf` is NaN. The harness prefills C with `+infinity`, and the
unconditional form produces NaN in **2 108 431 of 2 108 431** elements.

The branch is **warp-uniform** — `beta` is a kernel argument, identical for all
32 lanes — so by Module 8 it is a predicate evaluation on a uniform condition
and costs one instruction of issue, not a divergence penalty. Do not be tempted
to "avoid the branch" here.

**The reason this bug is dangerous rather than annoying:** with `C` zeroed —
which every unit test does — the two forms agree exactly. It is invisible until
a caller hands you uninitialised memory, at which point it produces NaN and the
blame lands somewhere else.

## TODO 4 — the launch configuration

```cpp
int gridX = (N + BLOCK_X - 1) / BLOCK_X;     // 2053 -> 65
int gridY = (M + BLOCK_Y - 1) / BLOCK_Y;     // 1027 -> 129
```

`x` carries `col`, so `gridDim.x` is driven by **N**. This is Module 3
Exercise 1's trap verbatim: the axis decision appears twice — once in the kernel
and once in the launch — and the two must agree. Writing `gridX` from M and
`gridY` from N with this kernel covers a 1027 × 1024-ish region and leaves most
of C untouched; the harness reports the count of unwritten elements rather than
crashing, because the `+infinity` sentinel survives.

## TODO 5 — the tolerance law (design)

```cpp
const double u      = ldexp(1.0, -24);                 // fp32 unit roundoff
const double gammaK = (double)K * u / (1.0 - (double)K * u);
const double SAFETY = 4.0;
(void)ref;                                             // deliberately unused
return SAFETY * gammaK * S + 8.0 * u * S / (double)K;
```

**Why `S` and not `ref`.** The standard backward-error bound for a length-K
inner product accumulated at unit roundoff `u` is

```
| c_hat - c |  <=  gamma_K * S ,   S = sum_k |a_ik| |b_kj| ,
gamma_K = K*u / (1 - K*u)
```

`S` is the sum of the *magnitudes* of the terms; `|c|` is the magnitude of their
sum. Those differ by the condition number of the inner product, which is
unbounded. Measured on this exercise's two datasets with the *same correct
kernel*:

| dataset | worst relative error vs \|C\| | worst error / (gamma_K · S) |
|---|---|---|
| positive operands | 1.6e-6 | 0.035 |
| zero-mean operands | **9.7e-4** | 0.0037 |

A tolerance scaled by `|ref|` has to be ~1e-3 to accept the zero-mean dataset,
and at 1e-3 it accepts the off-by-one-in-K defect too. A tolerance scaled by `S`
accepts both datasets with two orders of magnitude of headroom and still
rejects the tightest defect by 16–29×. That is why the harness runs two
datasets: **the obvious answer passes on one of them.**

**Why 2^-24 and not 1e-7.** `u` for IEEE binary32 is `2^-24 = 5.96e-8` (half
the machine epsilon `2^-23`). Using `1e-7` inflates the tolerance by 1.7×, which
is harmless, but using `1e-5` or `FLT_EPSILON` without thinking is not: the
shape probe checks `t(769, S=1)` against the theoretical `gamma_769 = 4.584e-5`
and requires `0.9x <= t <= 100x`.

**Why a safety factor at all, and why 4.** The bound is a worst case for the
*rounding of the products and additions*, but the GPU also rounds the final
`alpha * acc` and the store, and Modules 17 and 18 will sum in a different
bracketing again. 4× leaves the correct kernel at 0.009 of tolerance and the
tightest defect at 16×, i.e. three orders of magnitude of separation. A safety
factor of 100 would still reject every planted defect here but would start to
accept a genuinely wrong kernel at larger K, which is why the probe caps it.

**The extra `8*u*S/K` term** is an absolute floor so that a legitimately zero
`S` (an all-zero row) still yields a strictly positive tolerance; the shape
probe's requirement (d) tests exactly that. Returning `0.0` for an all-zero row
makes every subsequent comparison `d/0 = inf` and fails a correct kernel.

---

## The validator's ordering, and why it is not negotiable

```
1. unwritten sentinel (+inf survives)   -> whole array
2. finiteness                           -> whole array
3. sampled scaled error                 -> strided sample
```

Check 3 run first would **pass** an array of NaNs, because `NaN > worst` is
false and the running maximum never updates. The observed output makes this
concrete:

```
positive   beta applied unconditionally   unwritten 0  nonfinite 2108431  worst 0  reject
```

Worst = 0. The numerical check saw nothing. Only the finiteness pass caught it.
The same phenomenon appears in the wrong-guard row: `worst 0.03452`, i.e. every
*sampled and finite* element was correct, while 1 053 702 elements had never
been written at all.

---

## Expected output

Actual run on the RTX 3500 Ada, CUDA 13.2:

```
=== Module 16 / Exercise 1 — the naive GEMM and its test ===
C(1027 x 2053) = alpha * A(1027 x 769) * B(769 x 2053) + beta * C

  launch: grid (65, 129) x block (32, 8) = 2.15 M threads for 2.11 M elements

-- TODO 5 probed directly ----------------------------------------
    (a) proportional to S                 ok
    (b) doubling K gives 2.000x            ok
    (c) t(769, S=1)/theoretical =    4.000  ok
    (d) positive when ref == 0            ok
    shape score 4/4

-- dataset positive -------------------------------------------
    positive   YOUR kernel                            unwritten       0  nonfinite       0  worst   0.009259  ACCEPT
    positive   k loop stops one short                 unwritten       0  nonfinite       0  worst       16.1  reject
    positive   guard tests col < M instead of col < N unwritten 1053702  nonfinite       0  worst        inf  reject
    positive   beta applied unconditionally           unwritten       0  nonfinite 2108431  worst          0  reject
    positive   B indexed as B[col*K + k]              unwritten       0  nonfinite       0  worst      236.3  reject

-- dataset zero-mean -------------------------------------------
    zero-mean  YOUR kernel                            unwritten       0  nonfinite       0  worst  0.0007686  ACCEPT
    zero-mean  k loop stops one short                 unwritten       0  nonfinite       0  worst      28.59  reject
    zero-mean  guard tests col < M instead of col < N unwritten 1053702  nonfinite       0  worst        inf  reject
    zero-mean  beta applied unconditionally           unwritten       0  nonfinite 2108431  worst          0  reject
    zero-mean  B indexed as B[col*K + k]              unwritten       0  nonfinite       0  worst       1465  reject

-- the alpha/beta path -------------------------------------------
    alpha=0.75, beta=-1.25, C prefilled : worst 0.0006958  ACCEPT

  tolerance shape       4/4
  your kernel accepted  2/2 datasets
  defects rejected      8/8
  alpha/beta path       1/1
  SCORE: 15/15
OVERALL: PASS
```

The numbers above are deterministic — no timing is involved — and reproduce
exactly run to run.

---

## The result that matters

A GEMM test is only a test if it can fail, and the two ways to make it unable to
fail are opposite errors: a tolerance scaled by `|C|` **rejects correct
kernels** at large K or on ill-conditioned data, and a tolerance made loose
enough to fix that **accepts wrong ones**. The escape is to scale the tolerance
by the quantity the error bound actually depends on — `Σ|a||b|`, not `|C|` — and
then to condition the *test data* so that the two coincide. That is two separate
decisions, and this exercise scores you on both by running a dataset where they
coincide and one where they do not.

**Variation to try:** change `K_DIM` to 4099 and re-run. `gamma_K` rises to
2.4e-4, the off-by-one defect's margin *falls* (each dropped term is a smaller
share of the sum), and you can find the K at which a 4× safety factor stops
separating them. That value of K is the point at which fp32 accumulation stops
being a good enough validation substrate and you need a double or
split-accumulator reference — which is what a production GEMM test suite
actually does.
