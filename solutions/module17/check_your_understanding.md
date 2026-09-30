# Module 17 — Check Your Understanding: answers

---

## 1. The colleague whose `[32][33]` padding makes their kernel faster

**The structural difference: their kernel reads or writes the shared tile down a
column, and this module's kernel never does.**

The full degree table from `example02.cu` has exactly one entry above 2, and it
is the *store* into a transposed A tile, `As[tx][ty]`:

| access | T = 32, pad 0 | T = 32, pad 1 |
|---|---|---|
| `As[ty][k]` (this module's compute read) | 1 | 1 |
| `Bs[k][tx]` (this module's compute read) | 1 | 1 |
| `As[ty][tx]` (this module's cooperative store) | 1 | 1 |
| **`As[tx][ty]`** (a column-wise store) | **32** | **1** |

A kernel that stages A transposed — which is the natural thing to do as soon as
a thread owns a column of outputs and wants to read `A[k][m0..m0+7]` with one
instruction — performs that store, and it is a genuine 32-way conflict. Padding
to 33 turns `bank = (32·tx + ty) % 32 = ty` (one bank, 32 distinct words) into
`bank = (33·tx + ty) % 32 = (tx + ty) % 32` (a permutation), degree 1. Measured
in this module: padding that kernel is worth **1.05× at T = 16 and 1.18× at
T = 32**, in the opposite direction from the row-major case.

Other structural differences that would produce the same sign: reading a *column*
of the B tile (`Bs[k0..k0+7][tx]` staged as `Bs[n][k]`), or any access whose
stride across lanes is a multiple of 32 words.

**The confirming measurement:** enumerate their access. Take the 32 lanes of
warp 0, compute the word index each one addresses for the access in question,
reduce mod 32, and count distinct words per bank — the procedure from
`example02.cu` §C and Module 7. If any degree is ≥ 4, padding is buying something
real. Then confirm in the SASS that the instruction you analysed is the
instruction that executes: `cuobjdump -sass` and count `LDS`, `LDS.64`,
`LDS.128` and `STS` in the accumulation body, padded and unpadded. Two counts
and two degrees settle it.

**What happens if they also switch to `float4` staging.** Their padding almost
certainly breaks.

- A `float4` shared access requires a **16-byte-aligned** address. A pitch of 33
  floats puts row `r` at byte `132·r`, and `132 % 16 = 4`. Every odd row is
  misaligned and the `LDS.128` either cannot be emitted or faults
  (`cudaErrorMisalignedAddress` — Module 6's carve-alignment hazard, one memory
  space over).
- So the pitch for a `float4`-staged tile must be a multiple of 4 floats, i.e.
  odd *in units of `float4`*. Module 15's general rule is `gcd(P, 128/E) = 1` for
  `E`-byte elements; for `E = 16` that is `gcd(P, 8) = 1` with `P` measured in
  `float4`s, i.e. `P` odd. In floats that means `4·(odd)`: `[32][36]` works
  (9 `float4` per row, odd), `[32][40]` does not (10, even).
- And a `float4` access is phase-split (Module 7): the 32 lanes' 512 bytes cannot
  cross the 128-byte-per-cycle array in one go, so conflicts are resolved
  independently per phase. Their degree analysis has to be redone per phase.

The prediction: their `[32][33]` becomes `[32][36]`, their conflict is still
fixed, and their throughput goes up again because the vector path is now
available. This is the interaction Module 18 has to manage, and it is why
production GEMM kernels swizzle rather than pad — a swizzle costs no bytes and,
chosen as an XOR on *whole `float4`s*, preserves 16-byte alignment, which a
+1-float pitch cannot.

---

## 2. Why 8.00 FMAs per global load does not deliver 80 % of peak

**What Module 16's number was a statement about.** Its probe held two loads fixed
and varied the arithmetic:

```cpp
const float av = A[row*K + k];      // one LDG
const float bv = B[k*N + col];      // one LDG
for (r = 0; r < R; ++r) acc[r] = fmaf(av, bv, acc[r]);
```

What it measured is the throughput of a kernel as a function of **memory
instructions issued per arithmetic instruction**, on the LSU and L1TEX return
path. It happened to use `LDG` because Module 16 had no shared memory, and the
number was reported as "FMAs per global load" — but the mechanism it measured is
the issue and service cost of an LSU instruction, and *nothing in it is specific
to the global address space*.

`example02.cu` runs the identical experiment with `LDS`:

| R | loads per FMA | M16 (`LDG`), % of ceiling | M17 (`LDS`), % of ceiling |
|---|---|---|---|
| 1 | 2.00 | 9.5 % | 9.4 % |
| 2 | 1.00 | 19.6 % | 18.9 % |
| 4 | 0.50 | 38.2 % | 36.5 % |
| 8 | 0.25 | 62.2 % | 62.7 % |

The two tables agree to within a percent. Same law, different opcode.

**The quantity that had to be 6.5.** Not FMAs per *global* load — FMAs per
**memory instruction of any kind**, or equivalently per operand fetch. The tiled
kernel's value is:

| ratio | naive | tiled 16×16 |
|---|---|---|
| FMAs per global load | 0.50 | **8.00** |
| FMAs per shared load | — | **0.50** |
| FMAs per memory instruction, all | 0.50 | **0.47** |

Tiling moved the first number by 16× and left the third one where it was — in
fact very slightly worse, because the tiled kernel adds two `STS` per tile that
the naive kernel does not have. The 1.30× it delivers comes entirely from `LDS`
being cheaper to service than `LDG`, which is a change of cost per instruction.

**The one structural change that moves it without changing the tile size.**
Give each thread more than one output element. Keep the output tile at 16×16 but
launch 64 threads instead of 256, each owning a 2×2 sub-tile; per k-step a
thread then loads 2 words of A and 2 of B and performs 4 FMAs, so loads per FMA
falls from 2.00 to 1.00. Generally, an `Rr × Rc` register tile gives
`1/Rr + 1/Rc`. The *tile* is unchanged; the *decomposition* is not. That is
exactly the distinction Module 16 asked the reader to write in one sentence, and
it is Module 18.

---

## 3. The "1.3 % wrong means the barrier is basically never needed" argument

**First thing wrong: that is not what the 1.3 % measures.** The write-after-read
hazard is either present in the program or it is not, and here it is present in
every single one of the 66 tile iterations of every one of the 6 336 blocks. The
20 972 wrong elements are the subset where the hazard *happened to be observed*
under this particular schedule, on this occupancy, with this input, on this
GPU, in this thermal state. Change the block count, the clock, the co-resident
kernel, the L2 state, or the architecture, and the number moves. It is not a
probability attached to the barrier; it is a sample from an unspecified
distribution over schedules.

Worse, the program is undefined — Module 9's rule is that a data race on shared
memory has no defined outcome at all, not "the outcome of one of the two
orderings". A compiler is entitled to assume the race does not happen and to
reorder around it. The observed 1.3 % is not a bound on anything.

**Second thing wrong: "re-run until the answer is right" requires knowing what
right is.** If you have a trusted reference to compare against, you do not need
to compute the GEMM. If you do not, you cannot filter. And the corruption here is
not independent noise across runs — the schedule is largely determined by the
launch configuration, so the same elements tend to be wrong, and the errors
accumulate in the same direction. Re-running samples from a narrow, biased
distribution, not from a distribution centred on the truth.

A third, if you want it: 1–7 % faster is the payment, and the payment for it in
Module 18 is repaid legitimately by double buffering, which removes the same
barrier and is *correct*.

**Why Freivalds passed and the sampled check failed.** The Freivalds probe
computes, for each row `i`, `sum_j C[i][j]·v[j]` and compares it against the
same contraction computed in double from A and B, with a tolerance
`gamma_K · (|A|(|B||v|))_i` — a bound scaled by the magnitude of the **whole
row's** contraction. A race corrupts a handful of the 1541 elements of that row.
Their individual errors are large relative to their own `gamma_K·S_ij`, but when
they are summed into a row total alongside 1500 correct elements and compared
against a row-sized tolerance, they are diluted by a factor of order the row
length. At `T = 16` that dilution took the ratio to 0.59, just under the
threshold.

The sampled check compares one element against **its own** bound,
`err / (gamma_K · S_ij)`, with no dilution. It measured 58.6.

**Which check you would want if you could only afford one:** it depends on the
shape of the error you expect, and that is the real lesson.

- A **systematic** error — a wrong index, a dropped k-tile, a transposed operand
  — perturbs every element in the same direction, so the row sum *accumulates*
  it coherently (which is why Module 16 insists `v` be non-negative). Freivalds
  catches those with full coverage at `O(MN + KN + MK)` cost, and the sampled
  check would too but only if it happened to sample one.
- A **localised** error — a race, an off-by-one on one boundary tile, a single
  unwritten block — is diluted by any row-sum probe and needs a per-element
  comparison.

A race is localised, so here you would want the sampled check. But a strided
sample can miss a localised error entirely; it only caught this one because
1.3 % of 1.59 M elements is 20 000 of them, spread widely enough that a
16 × 24 sample hits several. The defensible answer is that you cannot pick one,
which is precisely why `gemmValidate()` runs three: finiteness for coverage
failures, Freivalds for systematic errors, sampling for localised ones. Each is
blind to a class the others see.

---

## 4. If the bank array were 64 banks × 4 B

The array would deliver 256 B per cycle per SM instead of 128, so the measured
shared read bandwidth would roughly double: ~10.8 TB/s scalar and ~20.6 TB/s
vectorised. Everything else — 128 FP32 lanes per SM, ~1.78 GHz, 432 GB/s of
DRAM — is unchanged, so the FP32 ceiling stays at ~18 000 GFLOP/s and the demand
stays at 8 bytes per FMA = 72 TB/s for a two-loads-per-FMA kernel.

**(a) The 16×16 tiled kernel: roughly 1.7–2.0×, to about 2900–3400 GFLOP/s.**
It is squarely shared-bandwidth bound — measured 1703 GFLOP/s against a
1345–2574 GFLOP/s band — so doubling the binding resource roughly doubles it.
It would not be a clean 2×, because at ~3400 GFLOP/s (19 % of the FP32 ceiling)
the secondary costs this module measured as 1–5 % each (barrier convoying, the
`STS` pair, the integer address arithmetic, the `ceil` tail) become a larger
share of the total. Predict 1.7–1.9× and expect the kernel to become
*instruction-issue* bound rather than bandwidth bound.

**(b) The `R = 8` probe row: very little, perhaps 1.1×.** At `R = 8` the kernel
reads 2 shared words per 8 FMAs = 1 byte per FMA, so its demand at the FP32
ceiling is `1 × 9050 G = 9.05 TB/s` — already just under the 10.3 TB/s the
*existing* array supplies. Shared bandwidth is barely binding there, which is
exactly why it reaches 62.7 % of the ceiling rather than 9.4 %. What limits it
instead is instruction issue: 2 `LDS` + 8 `FFMA` is still 10 instructions for 8
arithmetic ones, plus the loop and the barriers. Doubling the banks does not
remove an instruction.

**(c) cuBLAS: essentially nothing, 1.00–1.10×.** cuBLAS measures 8549 GFLOP/s =
47 % of the ceiling, which places it in the same regime as the `R = 8` row — it
uses a register tile of roughly 8×8 with vectorised shared loads, so its
shared-bandwidth demand is already about 1 byte per FMA. It was written by people
who arranged not to need the second bank array. What limits it is the same thing
that limits `R = 8`: instruction issue, register file pressure, and the fact that
a real GEMM also has to handle the `K` tail, `alpha/beta`, and the epilogue.

**The decomposition change worth more than the hardware change.** Register
blocking. Measured here, moving from 2.00 to 0.25 loads per FMA is worth
**6.7×** (9.4 % → 62.7 % of the ceiling) on identical silicon. Doubling the bank
array is worth at most 2×, and only to the kernel that has not been
register-blocked. A factor of 6.7 from a source rewrite against a factor of 2
from a die respin is the entire argument for why Modules 17 and 18 are two
modules and not one.

**Where the hardware change stops mattering at all.** A square `Rr × Rr` register
tile reads `2/Rr` shared words per FMA = `8/Rr` bytes. Set that against the
existing 10.3 TB/s at a demand of 9050 G FMA/s:

```
8/Rr × 9050e9 <= 10.3e12   =>   Rr >= 7.0
```

**At an 8×8 register tile the current 32-bank array is already sufficient**, and
64 banks would buy nothing whatsoever. That is not a coincidence: 8×8 is the tile
size production SGEMM kernels use, and it is the size at which the existing
hardware stops being the constraint. The bank array was sized for the kernel
people were expected to write.
