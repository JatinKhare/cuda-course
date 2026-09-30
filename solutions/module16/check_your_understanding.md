# Module 16 — Check Your Understanding, answers

---

## Q1 — the `beta == 0` branch changes results even though `beta` was 0 both times

**What happened.** The original line

```cpp
C[idx] = alpha * acc + beta * C[idx];
```

*reads* `C[idx]` unconditionally. With `beta == 0.0f` the product `beta * C[idx]`
is 0 for every finite value of `C[idx]`, so the two versions agree on every
input for which `C` holds finite data. The unit test does
`cudaMemset(C, 0, ...)` first, which guarantees finite data, so the test cannot
distinguish them.

The application does not memset. It calls the GEMM on a buffer obtained from
`cudaMalloc`, whose contents are **undefined** — whatever the last user of that
physical memory left there. If any of those bit patterns decodes as a NaN or an
infinity, then `0.0f * NaN = NaN` and `0.0f * inf = NaN`. The original code
writes NaN into those elements of C and the fixed code does not. So the change
in results is the *fix* becoming visible, not the fix breaking something.

**The circumstance where the original was the wrong one.** Precisely: any call
with `beta == 0` on a `C` buffer that has not been initialised and happens to
contain a NaN or infinity bit pattern. That is exactly the case the BLAS
specification was written to cover — it says C "need not be set on input" when
beta is zero — and it is common in practice, because the whole point of
`beta = 0` is "I am overwriting C, do not bother reading it".

There is a second, rarer circumstance worth naming: `beta == 0` with `C`
containing a *very large* finite value is fine, but `alpha == 0` with `A`
containing a NaN is not symmetric — the standard also specifies that when
`alpha == 0` the product need not be formed. A kernel that computes the dot
product anyway and multiplies by `alpha = 0` will propagate NaNs out of A. Same
class of bug, other operand.

**How to test for it.** Prefill C with a value no correct kernel can produce and
that is *not* annihilated by multiplication by zero. `+infinity` is ideal:
`0.0f * inf` is NaN, so the defect is loud, and no correct GEMM on finite data
produces an infinity, so a surviving `+inf` means "nobody wrote this element".
`example01.cu` and `exercise01.cu` both do this, and it catches the defect in
**2 108 431 of 2 108 431 elements**.

---

## Q2 — the sector model predicts 8.25× and the measurement is 3.3–3.9×

**Mechanism 1: sectors requested are not sectors fetched.** The sector count is
a per-warp, per-instruction quantity: it says how many distinct 32-byte blocks
*this* warp's 32 addresses touch. It says nothing about whether those blocks are
already resident. Under the `x → row` mapping, warp *w* of block *b* reads A
rows `32b .. 32b+31` at column k; the *next* warp in the same block reads the
same 32 rows at column k for a different output column, and so does every other
block in the same row-strip of C. The 32 scattered sectors are re-requested
enormously often and hit in L1 and L2 almost every time. The model charges full
price for each request; the hardware charges an L2 hit.

**Mechanism 2: the kernel is not sector-bound in the first place.** Even under
the good mapping the kernel is at 7 % of the compute ceiling and the binding
constraint is the *number of memory instructions per FMA*, not the bytes those
instructions move (§7 of the lesson: 8× the arithmetic for 10 % more time). A
model of bytes-per-instruction cannot predict the runtime of a kernel whose
limit is instructions-per-FMA; it can only predict the *ordering*, because
within one mapping more sectors does mean more service time per instruction.

**Problem-size changes that move the measurement toward 8.25×.**

- *Mechanism 1:* make the working set far exceed the 48 MB L2, so the
  re-requests miss. `4(MK + KN + MN) > 4 × 48 MB` needs roughly
  n ≳ 2000 for a square problem; at n = 4096 the three matrices are 64 MB each.
  Then the scattered A sectors are genuine DRAM traffic and the good mapping's
  4-sector footprint is genuine coalescing.
- *Mechanism 2:* increase **K only**, keeping M and N modest. A larger K
  increases the work per thread without increasing the size of C or the number
  of threads, so the fraction of runtime spent in the inner loop rises and the
  per-instruction service cost — which is what the sector count measures —
  becomes a larger share of the total.

**Which moves it further:** the L2-overflow change (mechanism 1), and by a wide
margin. Mechanism 2 changes the weighting of a term; mechanism 1 changes the
term itself, from an L2 hit at ~241 cycles to a DRAM miss at ~575 cycles *and*
from zero DRAM bytes to 33/5 of the compulsory bytes. The measured ratio at
1024³ (working set 12 MB, still L2-resident) is already 8.2×, higher than at
1027×2053×769, which is consistent: the shorter K there means less time for the
caches to amortise, and the effect is in the predicted direction.

---

## Q3 — "8 FFMAs per loaded pair was 6.6×, so 8 outputs per thread gives 6.6×"

**The resource the conclusion ignores: registers, and through them occupancy.**
The probe kernel keeps `R` accumulators in registers and reads *the same two
operands* for all of them. A real 8-output-per-thread GEMM needs the
accumulators **and** the operands that distinguish the outputs: if a thread owns
an 8-wide row of C it must hold 8 distinct B values (or reload them), and if it
owns a 2×4 patch it must hold 2 A values and 4 B values plus 8 accumulators. The
probe's register cost is 8 floats; a real 8-output tile's is closer to 16–20,
on top of the 40 the naive kernel already uses. At 1536 threads/SM and 65 536
registers/SM the budget is ~42 registers per thread at full occupancy, so this
is not a free parameter. Module 19 owns that arithmetic; Module 18 spends it.

**What would have to be true of the 8 outputs.** They have to **share
operands**. If a thread computes 8 outputs that share no A row and no B column,
it issues 16 loads per 8 FMAs and the ratio is unchanged — you have coarsened
without reusing, which Module 11 already measured as a regression on a saturated
kernel. The 8 outputs must form a contiguous patch of C: an 8-wide row shares
one A value across all 8; a 2×4 patch loads 2 A values and 4 B values to produce
8 FMAs, giving 6 loads per 8 FMAs = 1.33 FMAs per load, against the naive 0.5.
The patch shape *is* the optimization.

**Why the honest claim needs two changes.** Even a 2×4 patch still issues global
loads, and each one is an `LDG` with L1/L2 latency and an `IMAD.WIDE` of address
arithmetic behind it. Reducing the count from 2-per-FMA to 0.75-per-FMA helps,
but the remaining loads are still expensive, and every thread in the block is
still independently fetching values its neighbours are fetching too. So:

1. **Module 17** stages a tile of A and a tile of B in shared memory, so the
   repeated operand reads become `LDS` against a scratchpad with no tag compare
   and deterministic latency, and the global loads happen once per block instead
   of once per thread.
2. **Module 18** gives each thread a patch of C so the values loaded from the
   scratchpad into registers each feed several FMAs.

Neither alone gets there. (1) alone leaves the instruction count per FMA
unchanged — two `LDS` per `FFMA` instead of two `LDG` — which is exactly why
Module 6 measured tiling a 5-point stencil as a 0.85× *loss*. (2) alone leaves
every operand coming from global memory. The combination is what makes GEMM the
kernel where tiling finally pays, and it is why Module 6 named register blocking
as the missing ingredient rather than calling shared memory the answer.

---

## Q4 — K = 4096, heavy-tailed operands, `gamma_K * S` passes, relative error 10 %

**What has actually happened.** `S_ij = Σ_k |a_ik||b_kj|` is dominated by the
handful of enormous terms. `C_ij = Σ_k a_ik b_kj` is *not*, if those enormous
terms cancel — and with mixed signs they partially do. So `S` is ~10⁶ times
larger than `|C|`, the tolerance `gamma_K · S` is ~10⁶ times larger than
`gamma_K · |C|`, and a 10 % relative error in C sits comfortably inside it.

This is not a defect in the bound. The bound is *tight*: it is achieved, to
within a small constant, by exactly this kind of data. The absolute error
produced by fp32 accumulation really is of order `K · u · S`, because the
intermediate partial sums really do reach magnitude `S` before cancelling. The
computation is **backward stable** and the problem is **ill-conditioned**: the
condition number of this inner product is `S / |C| ≈ 10⁶`, and a backward-stable
algorithm on a problem with condition number κ delivers a relative error of
order `κ · K · u`. 4096 · 6e-8 · 10⁶ ≈ 25 %. Ten per cent is *better* than the
generic prediction.

**Should the kernel be accepted?** Yes — as a GEMM. Nothing is wrong with it;
the same 10 % would be produced by a different-but-correct summation order, by
cuBLAS, and by a single-threaded C loop in a different order. Rejecting it would
be rejecting fp32, not rejecting the kernel. What must *not* happen is
concluding that the downstream application is fine: an application that needs
those elements of C to 3 digits needs a different precision or a different
formulation, and that is a numerical-analysis decision, not a kernel-correctness
one.

**What to change about the test data.** Make the test problem
**well-conditioned**, so that `S ≈ |C|` and the scaled tolerance and the
relative tolerance coincide. Concretely: draw the operands **strictly positive**
and of similar magnitude — uniform on [0.5, 1.5) is what this module uses — so
there is no cancellation, `S` and `|C|` agree to within a small factor, and an
error of 10 % of `|C|` is 10 % of `S` and fails the bound by four orders of
magnitude. Keep the ill-conditioned dataset too, as a *second* case, because it
is the one that proves the tolerance is scaled by the right quantity — but do
your **bug hunting** on the well-conditioned one, where the test has teeth.

The general rule this establishes: the tolerance must be scaled by `S` so that
correct kernels are never rejected, and the *data* must be conditioned so that
`S ≈ |C|` and incorrect kernels are never accepted. Those are two different
jobs and you need both. `exercise01.cu` scores the reader on exactly this by
running both datasets and requiring the same tolerance to accept the correct
kernel and reject four defects on each.
