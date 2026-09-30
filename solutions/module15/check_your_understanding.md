# Module 15 — Check Your Understanding, answers

---

## Q1 — "340 GB/s, 79% of peak. Worth optimizing?"

**Why the number cannot answer the question.** `% of 432 GB/s` is a ratio between
*useful bytes delivered* and a *nominal DRAM peak*. It conflates two independent
things: how much traffic the kernel generates, and how fast the bus ran while it
did. A transpose that moves 2N useful bytes at 340 GB/s and a transpose that
moves 2N useful bytes while the bus actually carried 8N — because every store was
a partial-sector read-modify-write — both report a useful-bandwidth figure, and
only one of them has anything left to fix. This is exactly Module 5's
effective-versus-DRAM distinction and Module 11's `x floor` argument: GB/s alone
cannot distinguish a 2N kernel from a 9N one.

It is also not a *reachable* ceiling. 432 GB/s is the pin rate. Nothing on this
GPU reaches it. A pure-read stream reaches 410.5 GB/s (Module 12, 1500 ms
warm-up); a 1:1 read/write copy — the actual shape of a transpose — reaches
373–384 GB/s. Measured against the right denominator, 340 GB/s is 89–91%, not
79%.

**The two measurements to ask for.**

1. **A copy of the same matrix, same block shape, same instruction count, timed
   in the same rotated sweep.** That is the speed of light for a 2N kernel and
   the only honest denominator.
2. **Whether the buffers exceed the 48 MB L2.** If they do not, the number is an
   L2 measurement and is not comparable to anything.

(A useful third: the measured `% of copy` for the naive version, so you know how
much of the distance has already been travelled.)

**When 79% of peak means finished.** If the copy ceiling itself measured 79% of
peak in the same sweep — which happens on this part whenever the memory P-state
has not fully ramped, or under the software power cap — then a transpose at 79%
*is* at 100% of copy, and the traffic model says a 2N kernel cannot do better
than a 2N copy. The kernel is finished. The correct response to "79% of peak" is
never to optimize; it is to measure the copy.

---

## Q2 — Swapping which phase carries the conflict

**Which half of the argument is right.** The conclusion is right; the reason is
wrong.

The colleague is correct that transposing in shared memory is symmetric: you may
store the tile transposed and load it straight, instead of storing it straight
and loading it transposed. Both produce the correct output, and both move
identical bytes through global memory.

They are wrong that the two versions go "through the same banks". The bank map
of a `[32][32]` float tile is `bank(r,c) = (32r + c) mod 32 = c`. A warp has
`threadIdx.x` varying and `threadIdx.y` fixed, so:

| version | store phase | store D | load phase | load D |
|---|---|---|---|---|
| canonical | `tile[ty+j][tx]` → `c = tx` varies | **1** | `tile[tx][ty+j]` → `c = ty+j` fixed | **32** |
| swapped | `tile[tx][ty+j]` → `c = ty+j` fixed | **32** | `tile[ty+j][tx]` → `c = tx` varies | **1** |

The conflict does not disappear. It **moves from the `LDS` to the `STS`**. The
degrees are not the same in each phase; the *multiset* of degrees is.

**Predicted ratio.** Since the total number of replayed wavefronts is identical —
4 instructions at D = 32 either way — the two versions should measure the same to
within noise at both sizes. There is no reason for a store replay to cost
differently from a load replay: the replay loop is in the same L1TEX unit and
neither touches DRAM.

Measured:

```
N=8192  load-conflict 1.90423 ms   store-conflict 1.74776 ms   B/A 0.918x
N=8192  load-conflict 1.73773 ms   store-conflict 1.95850 ms   B/A 1.127x
N=2048  load-conflict 0.06226 ms   store-conflict 0.06037 ms   B/A 0.970x
N=2048  load-conflict 0.06215 ms   store-conflict 0.06034 ms   B/A 0.971x
```

At 8192 × 8192 the ratio is 0.92 in one run and 1.13 in the next — pure
run-to-run variation, no signal. At 2048 × 2048 the store-conflicted version is
reproducibly 3% faster, which is far smaller than the 3× that separates either of
them from the padded version, and is not worth a story.

**The point of the question:** "it moves the same bytes" is not an argument about
bank conflicts at all. Bank conflicts are not a traffic phenomenon; they are a
replay phenomenon in the LSU pipeline (Module 7; Module 4's constant-memory
serialization and Module 10's atomic contention are the same mechanism). Two
kernels can move byte-for-byte identical traffic and differ 16× in shared-memory
cost. Here they happen not to, because the *degree* structure is the same — but
you have to count to know that.

---

## Q3 — Correct on squares and on multiples of 32, wrong on 4093 × 2049

**The two most likely defects.**

**(a) The output's leading dimension is `W` where it should be `H`.**
`out[(yo+j) * W + xo]` instead of `out[(yo+j) * H + xo]`. On a square matrix
`W == H` and the expression is literally the same. On a non-square matrix every
output row is placed at the wrong stride, so the output is a systematic
shear: the element belonging at `(r, c)` lands at linear index `r·W + c` instead
of `r·H + c`, i.e. displaced by `r·(W - H)`. With `W > H` the tail of the array
is written past the end of the logical output (and, if the allocation is exactly
`W·H` floats, is still in range, so `compute-sanitizer` reports nothing).

**(b) The guards use the wrong pair of bounds.** The load guard must test
`(x < W && y+j < H)`, the store guard `(xo < H && yo+j < W)`. Using `(W, H)` in
both is invisible on a square matrix and, on a non-square one, either drops
output elements (leaving whatever `poison` wrote) or writes outside the output.

**Distinguishing them by looking at the output.** Take `W = 3`, `H = 2` — a
2 × 3 input, so the output is 3 × 2 — and a tile of 32 (one block, every guard
exercised). Number the input `in[y][x] = 10y + x`:

```
in =  0  1  2          correct out =  0 10
      10 11 12                        1 11
                                      2 12
```

- Defect (a), leading dimension `W = 3` instead of `H = 2`, writes value
  `in[c][r]` to linear index `r*3 + c`. Linear output becomes
  `[0, 10, ?, 1, 11, ?]` for the first six slots: the correct values appear, in
  the correct order, **spread out with a hole every third slot**, and the last
  two elements are written past index 5. The signature is *right values, wrong
  positions, regular stride*.
- Defect (b), both guards testing `(W, H)`, drops every output cell with
  `xo >= H` — here `xo >= 2`, which is none, so pick `W = 2, H = 3` instead and
  the third output column vanishes. The signature is *right values in the right
  positions, with a rectangular block of untouched cells* still holding the
  poison value.

Right-values-wrong-places versus right-places-missing-values separates them
immediately, and a 2 × 3 matrix is small enough to print.

**Why square power-of-two test suites cannot find either.** Both defects are
substitutions of `W` for `H`. On a square matrix `W` and `H` are the same
integer, so the buggy and the correct program are the *same program* — not
"accidentally agreeing", but textually equivalent after constant folding. No
amount of running it, at any size, with any data, under any sanitizer, can
distinguish them. A power-of-two dimension additionally hides every partial-tile
path, because `32 | 2^k` for `k >= 5`, so the guards are never false and a
missing guard is never exercised. The two properties a transpose test must have
are therefore **`W != H`** and **`32 ∤ W`, `32 ∤ H`** — which is why every harness
in this module uses 4093 × 2049, and why 8191 × 8193 appears in `example02.cu`.

---

## Q4 — Reconciling 14.79×, 1.00× and 3.0×

**The general rule.** A kernel's time is set by its binding resource. Write
`T_mem` for the time the memory system needs to deliver the kernel's compulsory
traffic and `T_lsu` for the time the SM's load/store pipeline needs to issue and
retire the kernel's memory instructions, including replays. To a first
approximation these overlap, so

```
T  ≈  max(T_mem, T_lsu)
```

and a bank conflict of degree `D` multiplies only the shared-memory part of
`T_lsu`. Split `T_lsu = T_shared + T_other`. Then the measured whole-kernel
penalty of a conflict is

```
                max( T_mem , T_other + (D/2)·T_shared )
penalty  =    -----------------------------------------
                max( T_mem , T_other +       T_shared )
```

(the `D/2` rather than `D` is Module 7's `max(2,D)` law: a conflict-free 32-lane
4 B shared access already occupies the pipeline for two cycles).

Two limits fall straight out:

- **`T_mem` dominates both numerator and denominator** → penalty → 1. The
  conflict is free. This is the DRAM-bound transpose: `T_mem ≈ 3.5 µs` per block
  against a shared term of ≈ 0.5 µs even at `D = 32`.
- **`T_mem` is negligible and `T_shared` dominates `T_other`** → penalty →
  `D/2 = 16`. This is Module 7's microbenchmark, which was built precisely to
  make `T_other` and `T_mem` vanish: `#pragma unroll 1`, four independent
  accumulators, one tiny shared array, no global traffic in the loop at all.
  Measured 14.79×, within 8% of the 16× bound.
- **In between**, with `T_mem` removed but `T_other` comparable to `T_shared`:
  penalty → `(T_other + 16·T_shared)/(T_other + T_shared)`. The L2-resident
  transpose measured **2.94–3.25×**; solving for the split gives
  `T_shared / T_other ≈ 0.16`, i.e. the four `LDS`/`STS` pairs account for about
  one seventh of the non-DRAM work, the rest being the `LDG`/`STG` that did not
  get slower, the address arithmetic and the barrier. That is a plausible budget
  for a kernel with 8 global and 8 shared instructions per thread.

So the rule is: **`D/2` bounds the shared-memory term, never the kernel, and the
measured penalty is that bound diluted twice — once by the instructions that do
not conflict, and once by whatever the kernel is actually waiting for.**

**Prediction for an L1-resident transpose.** With DRAM *and* L2 both removed,
`T_mem` collapses to L1 latency (40.5 cycles, M4) and the `LDG`/`STG` term
shrinks sharply, so `T_shared / T_other` rises and the penalty should climb
above the 3× measured at L2 — towards, but still well short of, 16×, because the
kernel still issues eight global instructions per thread that are unaffected. A
defensible estimate is 5–8×.

**Why you cannot build that experiment on this GPU.** Three independent reasons,
and the first is fatal.

1. **A transpose has no reuse.** Each input element is read exactly once and each
   output element written exactly once. There is nothing for L1 to *hit on*
   within a kernel launch: every line fetched is used for the bytes it contains
   and never touched again. "L1-resident" is not a property you can arrange by
   shrinking the matrix, because residency is irrelevant when the reuse factor
   is 1. (Repeating the kernel in a loop does not help: L1 is 128 KB per SM
   shared with shared memory, and the matrix would have to be smaller than
   40 × 128 KB *and* each block would have to land on the same SM each launch,
   which the GigaThread engine does not promise.)
2. **L1 is not coherent across SMs** (M4, M5), so the output written by one SM is
   not readable through another SM's L1 anyway; the store path goes to L2
   regardless.
3. **Shared memory and L1 are the same 128 KB array.** Taking 4224 B per block ×
   6 blocks out of it to hold the tiles directly reduces the L1 you are trying to
   measure, so the experiment perturbs its own independent variable.

The right way to see the effect is the one this module uses: keep the kernel
identical and shrink the *memory* term by moving the data from DRAM to L2, which
is a 4× change in `T_mem` and is enough to flip the answer from 1.00× to 3.0×.
