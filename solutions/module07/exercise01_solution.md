# Module 07 / Exercise 1 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

## The paper table, filled in

| # | expression | bytes | lane → bank (first 8) | distinct words in busiest bank | D |
|---|---|---|---|---|---|
| 0 | `s[tid]` | 4 | 0 1 2 3 4 5 6 7 | 1 | **1** |
| 1 | `s[2*tid]` | 4 | 0 2 4 6 8 10 12 14 | lanes 0,16 → words 0,32, both bank 0 | **2** |
| 2 | `s[8*tid]` | 4 | 0 8 16 24 0 8 16 24 | lanes 0,4,8,…,28 → 8 words, bank 0 | **8** |
| 3 | `s[32*tid]` | 4 | 0 0 0 0 0 0 0 0 | all 32 lanes, 32 words, bank 0 | **32** |
| 4 | `s[tid/2]` | 4 | 0 0 1 1 2 2 3 3 | lane pairs want the **same word** | **1** |
| 5 | `s[31-tid]` | 4 | 31 30 29 28 27 26 25 24 | a permutation of case 0 | **1** |
| 6 | `dd[tid]` | 8 | 0/1, 2/3, 4/5, … | 2 phases × 1 cycle | **1 per phase** |
| 7 | `dd[2*tid]` | 8 | 0/1, 4/5, 8/9, … | 2 phases × 2 cycles | **2 per phase** |

For the general `s[k*tid]` case the answer is `D = gcd(k, 32)`, which covers
rows 0–3 at a glance and explains why `s[3*tid]` and `s[33*tid]` (from the
lesson) are free while `s[32*tid]` is a catastrophe.

## TODO 1 — `bank_of`

```cpp
static int bank_of(uintptr_t byte_addr)
{
    return (int)((byte_addr / BANK_W) % BANKS);
}
```

Correct because the array is **word-interleaved**: bank membership is a property
of the 4-byte word index, not of the byte. Divide first, then take the modulus.

Common wrong approaches:

- `byte_addr % 32`. Takes the modulus of the *byte* address. Off by a factor of
  4 in the stripe: it maps 32 consecutive bytes (8 words) onto 32 banks, and the
  structural test rejects it immediately because it is not a bijection over 32
  consecutive words — it hits only banks 0, 4, 8, …, 28.
- `(byte_addr / 4) % 16`. The 16-bank layout of compute capability 1.x. Returns
  values in `[0,16)`, fails the bijection test.
- `(byte_addr >> 2) & 31`. Correct and identical for non-negative addresses;
  the structural test passes. Fine.

The structural test deliberately never states the formula. It checks four
*properties* a 32 × 4 B striped memory must have: values in `[0,32)`, a bijection
over 32 consecutive words, a period of 128 bytes, and indifference to the low
two address bits. Those four facts pin the function down without spelling it.

## TODO 2 — `degree_in_group`

```cpp
static int degree_in_group(const int* elemIdx, int n, int elemBytes)
{
    if (n <= 0) return 0;
    long long words[BANKS][WARP * 4];
    int       nw[BANKS];
    for (int b = 0; b < BANKS; ++b) nw[b] = 0;

    for (int i = 0; i < n; ++i) {
        const uintptr_t a0 = (uintptr_t)elemIdx[i] * (uintptr_t)elemBytes;
        for (int off = 0; off < elemBytes; off += BANK_W) {
            const uintptr_t a = a0 + off;
            const int       b = bank_of(a);
            const long long w = (long long)(a / BANK_W);
            bool seen = false;
            for (int k = 0; k < nw[b]; ++k) if (words[b][k] == w) { seen = true; break; }
            if (!seen) words[b][nw[b]++] = w;
        }
    }
    int mx = 0;
    for (int b = 0; b < BANKS; ++b) if (nw[b] > mx) mx = nw[b];
    return mx;
}
```

Three things must be right, and each has a characteristic failure:

1. **Count distinct words, not lanes.** The `if (!seen)` guard is the entire
   broadcast rule. Drop it and `s[tid/2]` reports 2 and `s[0]` reports 32 — the
   two rows the exercise exists to teach. The hardware's crossbar drives one
   bank output to as many lanes as want it; there is no arbitration to pay for.
2. **Take the maximum over banks, not the sum and not the average.** The
   instruction retires when the *last* wavefront lands, and a bank that owes `D`
   words forces `D` wavefronts. A sum over banks would report 32 for the
   conflict-free case.
3. **A wide element spans several words.** The inner `off` loop is what makes
   the function work for `double`. An 8-byte element at byte `8j` occupies words
   `2j` and `2j+1`, in two adjacent banks. Omit the loop and every `double`
   pattern reports the degree of its first word only.

The harness's consistency test is deliberately weak — it checks only
`ceil(distinctWords/32) ≤ D ≤ distinctWords`, where `distinctWords` is computed
without reference to banking — because a strong test would have to contain the
answer. It catches the crude failures (returning a constant, returning a sum
over banks, returning `n`), and it lets the subtle ones through on purpose.

The real check on TODO 2 is the **prediction score** in the final table. If you
counted lanes instead of words, you predicted 2.00× for `s[tid/2]` and the clock
says 1.00×; if you forgot the `off` loop, your `dd[2*tid]` cycle count is wrong
and the ratio misses. Your model has to survive contact with the hardware, which
is the only test that was ever going to mean anything.

## TODO 4 — `cycles_for_pattern`

```cpp
static int cycles_for_pattern(int p)
{
    const int eb     = PAT_ELEM_BYTES[p];
    int       phases = eb / BANK_W; if (phases < 1) phases = 1;
    const int lanes  = WARP / phases;

    int total = 0;
    for (int ph = 0; ph < phases; ++ph) {
        int idx[WARP];
        for (int i = 0; i < lanes; ++i) idx[i] = pat_index(p, ph * lanes + i);
        total += degree_in_group(idx, lanes, eb);
    }
    return total;
}
```

The bank array delivers `32 × 4 = 128 B` per cycle. A warp asking for
`32 × elemBytes` bytes therefore needs at least `elemBytes/4` cycles no matter
what, and the hardware realises that floor by cutting the warp into that many
**contiguous** lane groups and resolving conflicts inside each.

Common wrong approaches:

- **No split at all** (`return degree_in_group(all 32 lanes, eb)`). Predicts
  `dd[2*tid]` at 4 cycles relative to `dd[tid]`'s 2, which is ratio 2.0 — which
  happens to be *the same answer* here. The split only becomes visible when the
  two phases need different numbers of words; `example02.cu`'s
  `dd[(tid%16)*16]` versus `dd[(tid%32)*16]` pair is the case that separates
  them (naive: 16 vs 32; truth: equal; measured: 1.00×).
- **Interleaved rather than contiguous groups** (even lanes in phase 0, odd
  lanes in phase 1). Gives a different answer for the `(tid%16)*16` pattern, and
  the measurement rules it out: the contiguous split predicts equality with
  `(tid%32)*16` and the interleaved split predicts a factor of 2. The hardware
  measures 1.00×.
- **Multiplying instead of summing** (`phases * degree_of_whole_warp`). Double
  counts: it charges every phase for words that belong to another phase.

## Synchronization / memory reasoning

Each kernel does a cooperative fill of the shared array and then a single
`__syncthreads()` before any thread reads. That is the Module 6 pattern verbatim
and is the only barrier in the file: after it, the tile is read-only for the
rest of the kernel, so no second barrier is needed. Module 9 makes the ordering
guarantee precise; for now, "every thread in the block has completed its writes
before any thread proceeds" is the property being relied on.

Nothing in this exercise involves cross-warp communication, and nothing involves
the memory model beyond that one barrier. Bank conflicts are an *intra-warp,
intra-instruction* phenomenon: no amount of synchronization creates or removes
one.

## Performance reasoning

The measurement is a throughput test, and three details make it one:

- **The loop advances the index by 32 floats = 128 B per step.** That is one
  full trip round the bank array, so each iteration reads a *different word* in
  the *same bank* as the previous. Conflict degree is invariant over the whole
  loop, which is the property a throughput measurement needs.
- **Four independent accumulators.** One `acc +=` chain serializes on the FADD
  latency and hides the banks entirely — in an early version of this harness
  with a single accumulator, `s[2*tid]` and `s[32*tid]` measured 11.25 and 71.13
  cycles per load against a conflict-free 11.00, which is a *latency* signal
  (`11 + 2(D-2)`), not a throughput one.
- **Configuration order rotates across sweeps.** Without rotation, whichever
  configuration is measured first in every sweep sits right after a
  `cudaDeviceSynchronize()` and eats the clock dip that follows. An early
  version of this file reported `s[tid/2]` at 0.75× of `s[tid]` for exactly that
  reason — the baseline was inflated, not the broadcast made fast. Rotating
  `p = (q + sweep) % N_PAT` removes it, and the run-to-run spread on every ratio
  dropped below 1 %.

**Why `dd[...]` accumulates with an integer XOR.** Ada runs FP64 at 1/64 the
FP32 rate. With `a += dd[i]` every double pattern measured 2.33 ms — identical
for `dd[tid]` and `dd[32*tid]` — because the kernel was bound by the FP64 pipe
and the banks were invisible. `__double_as_longlong` keeps the load an honest
`LDS.64` while making the arithmetic free.

## Expected output

Actual output on the RTX 3500 Ada, warm, two consecutive runs agreeing to
better than 1 %:

```
TODO 1 structural test : ok
TODO 2 consistency test: ok

pattern      bytes   cycles predicted
-------------------------------------------
s[tid]           4        1     1.00x
s[2*tid]         4        2     1.00x
s[8*tid]         4        8     4.00x
s[32*tid]        4       32    16.00x
s[tid/2]         4        1     1.00x
s[31-tid]        4        1     1.00x
dd[tid]          8        2     1.00x
dd[2*tid]        8        4     2.00x

Numerical check vs CPU reference: PASS

pattern       cycles         ms    measured   predicted   verdict
--------------------------------------------------------------------------
s[tid]             1     0.0585       1.00x       1.00x   ok
s[2*tid]           2     0.0585       1.00x       1.00x   ok
s[8*tid]           8     0.2215       3.79x       4.00x   ok
s[32*tid]         32     0.8646      14.79x      16.00x   ok
s[tid/2]           1     0.0584       1.00x       1.00x   ok
s[31-tid]          1     0.0585       1.00x       1.00x   ok
dd[tid]            2     0.0594       1.00x       1.00x   ok
dd[2*tid]          4     0.1126       1.90x       2.00x   ok

Predictions within 30%: 8/8

TODO 1: ok   TODO 2: ok   TODO 4: ok   numerics: PASS   predictions: 8/8

OVERALL: PASS
```

Absolute times move a lot with clock state — a cold first run measured 0.1233 ms
for `s[tid]` (2.1× the warm value) and pushed `s[32*tid]` to 18.94×. Let the
machine warm up and run it twice. The ratios in the table above are reproducible
to under 1 %; `s[32*tid]` ranges 14.7–14.8× warm and up to ~19× cold.

The `predicted` column here is the solved TODO 3, namely
`max(2, cycles) / max(2, cycles_of_reference)`.

## The result that matters

**The correct model is `cost ∝ max(2, D)`, not `cost ∝ D`, and the quantity you
bucket is distinct *words*, not lanes.** Those two corrections between them
produce every surprise in the table. Counting lanes turns the free `s[tid/2]`
broadcast into a phantom 2-way conflict and makes you "optimise" a kernel that
was already optimal. Forgetting the floor of 2 makes you predict that
`s[2*tid]` costs double when it costs nothing, and — far more expensively —
makes you spend a week removing 2-way conflicts from a code base where they were
already free. The paying cases start at `D = 4`, where the cost is exactly `D/2`
and worth every minute.

**Variation to try.** Change `pat_index` to `return (lane * 5) % 40;` and work
out the degree by hand before running it. The stride is odd, which the `gcd`
rule says is free — but the modulus is not 32, and the wrap changes which lanes
collide. Then try `(lane % 4) * 8`, which touches only 4 distinct words with 32
lanes and is conflict-free for a reason that has nothing to do with stride.
