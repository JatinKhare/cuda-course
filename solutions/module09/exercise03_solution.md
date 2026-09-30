# Module 09 / Exercise 03 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler /Zc:preprocessor -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
```

Warning-clean. `<cuda/atomic>` needs `-std=c++17`; without it the build fails
with `libcu++ requires at least C++ 17`. On MSVC it additionally needs
`/Zc:preprocessor`, otherwise the build fails with
`MSVC/cl.exe with traditional preprocessor is used`. On Linux neither extra
flag beyond `-std=c++17` is required.

---

## TODO 1 — the producer side

```cpp
for (int k = 0; k < PAY; ++k) payload[buf][t * PAY + k] = v + (unsigned)k;

__threadfence_block();          // <-- TODO 1

flag_ref(flag[t]).store(r + 1, cuda::memory_order_relaxed);
```

**The requirement was:** any thread that observes `flag[t] == r+1` must also
observe the PAY payload words written for round r, and this thread must not
wait for anybody.

**Why `__threadfence_block()` is right.** The needed guarantee is pure
ordering, at block scope, one-directional: my prior stores before my later
store. That is the definition of a fence. `__threadfence_block()` compiles to
`MEMBAR.SC.CTA`, which touches the memory pipeline and not the warp scheduler,
so it blocks nobody. Block scope is the correct scope because every observer is
in the same block and the payload is in shared memory; `__threadfence()` would
be correct but would push ordering out to the L2 for no reason.

**The equally correct alternative, and it is worth knowing they are the same
thing:**

```cpp
flag_ref(flag[t]).store(r + 1, cuda::memory_order_release);   // no fence
```

A release store is defined as "no memory operation of this thread may be
reordered after this store", which is exactly the fence plus the relaxed store
folded into one operation. Same SASS, same guarantee, and it says the intent in
the type system rather than in a comment. Prefer it in new code; recognise the
fence form because you will read it in every existing CUDA codebase.

**Common wrong approaches.**

- *Nothing at all.* Passes. On this GPU, every run, 0 wrong out of 262 144.
  Ada's LSU retires one thread's shared stores in issue order, so the payload
  lands before the flag whether or not you asked for it. This is the Exercise 1
  lesson again: your test suite is not testing this. The only observable
  difference is the presence of `MEMBAR.SC.CTA` in the SASS.
- *`__syncthreads()` instead.* It *is* sufficient — it orders and it waits —
  but it is the wrong instrument, and in this kernel it is also actively
  harmful: putting a block-wide barrier in the flag variant makes the flags and
  the spin redundant while retaining their cost, and re-imposes the block-wide
  skew that the design exists to avoid. It turns the fast variant into a slower
  copy of the slow one.
- *`volatile` on the payload array.* Does nothing for ordering between the
  non-volatile flag store and the volatile payload stores — `volatile` orders
  volatile accesses only with respect to each other. See the lesson.

## TODO 2 — the consumer side

```cpp
while (fr.load(cuda::memory_order_acquire) < r + 1) { /* spin */ }
```

**The requirement was:** do not read the peer's round-r payload before the peer
publishes it; do not use payload values obtained before the flag was seen to
reach r+1; and terminate.

The acquire load does all three:

- it is an atomic load, so the compiler must actually emit a load every time
  round the loop — a plain `while (flag[peer] < r+1) {}` on a non-atomic,
  non-volatile `int` is a loop on a loop-invariant expression and the compiler
  is entitled to hoist it and spin forever;
- `memory_order_acquire` forbids any later memory operation of this thread from
  being reordered before it, which is precisely "the payload reads cannot be
  satisfied from before the flag was seen set";
- `< r + 1` rather than `== 0` is what makes the flag work across 16 rounds
  without a reset — the peer's flag is monotonically increasing, so there is no
  round in which it has to be cleared, and therefore no extra synchronization
  to clear it safely.

**Equally correct alternative:** `while (fr.load(cuda::memory_order_relaxed)
< r + 1) {} __threadfence_block();` — the relaxed load keeps the loop honest,
the fence after it supplies the acquire ordering. Same guarantee, one more
instruction.

**Common wrong approaches.**

- *Omitting the wait entirely* (the shipped state). The program terminates and
  reports **136 498 of 262 144 wrong** — about 52%. This is the one place in
  Module 9 where the broken version fails loudly, because the skew guarantees
  that the peer genuinely has not finished.
- *`volatile int* flag` with a plain loop.* Terminates, and happens to be
  right on this GPU, but supplies no acquire ordering and is exactly the idiom
  the lesson tells you not to write.
- *`__syncthreads()` in place of the wait.* Deadlocks or is undefined
  depending on where you put it: the threads are at different rounds, so a
  barrier here is inside control flow that is not block-uniform.

**Why spinning is legal at all here** — and this is the contrast with Exercise
2, so make sure you can state it: all 256 threads of a block are placed on one
SM simultaneously (Module 1), so the thread you are waiting for is *running*,
not queued. On sm_70+, independent thread scheduling additionally guarantees
forward progress for diverged threads (Module 8), so the spinning warp cannot
starve the warp it is waiting for. Neither condition holds across blocks, which
is why the identical code shape deadlocks there.

## TODO 3 — the barrier variant

```cpp
__shared__ unsigned payload[2][TPB * PAY];

const int peer = (t + PEERD) & (TPB - 1);
unsigned acc = seed[base + t];

for (int r = 0; r < ROUNDS; ++r) {
    const int buf = r & 1;
    const unsigned v = burn(acc, work_for(t, r));
    for (int k = 0; k < PAY; ++k) payload[buf][t * PAY + k] = v + (unsigned)k;

    __syncthreads();                       // G1 + G2, once per round

    unsigned s = 0;
    for (int k = 0; k < PAY; ++k) s += payload[buf][peer * PAY + k];
    acc = s;
}
out[base + t] = acc;
```

The design decision hidden in "how many barriers per round" is the same one as
Exercise 1's TODO 4: with a single payload buffer you need two barriers per
round (RAW then WAR); double-buffering by round parity removes the WAR hazard
and leaves one. Two buffers suffice because a thread and its peer can never be
more than one round apart — each is blocked on the other.

Shared memory: 2 × 256 × 4 × 4 B = 8 KB, well inside the 48 KB default, and the
same footprint as `variant_flag` so the occupancy comparison is fair.

---

## Synchronization / memory reasoning

Both variants are correct; they synchronize different *sets* of threads.

| | who waits for whom | what orders the writes |
|---|---|---|
| `variant_flag` | thread t waits for thread (t+128) only | `__threadfence_block()` / release store |
| `variant_barrier` | every thread waits for every thread | the barrier's own fence (G2) |

`variant_flag` needs an explicit fence because the waiting mechanism (a spin on
a flag) carries no ordering of its own. `variant_barrier` needs no fence
because `__syncthreads()` *is* a fence at block scope — one instruction buys
both guarantees, which is the whole argument for using a barrier when you need
both.

---

## Performance reasoning

Measured, min of 4 sweeps × 20 iterations, after a duration-based (400 ms)
clock warm-up, 1024 blocks × 256 threads, ROUNDS = 16:

| variant | ms | ratio |
|---|---|---|
| `variant_flag` (fence + per-pair flags) | 0.0936 – 0.0947 | 1.00 |
| `variant_barrier` (one `__syncthreads()` per round) | 0.1262 – 0.1276 | **1.33 – 1.36** |

Reproducible to under 1% across runs once the warm-up is in place. Without the
duration-based warm-up the absolute times swing from 0.09 to 0.24 ms and the
*direction* of the comparison flips — which is exactly the failure mode spec
§12 exists to prevent, and is worth reproducing once by deleting the warm-up
block.

**Why the flag variant wins.** The skew moves: in every round exactly one
thread of the block does `BASEW + HEAVY = 6200` iterations against everyone
else's 200. Module 8: the unit that serializes on a divergent trip count is the
warp, so the heavy *thread* makes its whole warp heavy.

- `variant_barrier` pays, per round, the max over all 8 warps. Exactly one warp
  is heavy each round, so every round costs `BASEW + HEAVY`. Total over 16
  rounds: **a sum of maxima**, 16 × 6200 ≈ 99 200 units.
- `variant_flag` pays, per thread, the max over its own two-cycle dependency
  (t and t+128 are mutual peers, so warp *w* depends on warp *w+4* and vice
  versa). A warp-pair is heavy only in the rounds where the moving heavy thread
  lands in one of its 64 threads, i.e. a quarter of the rounds. Its total is
  **nearer a maximum of sums**, 16 × 200 + 4 × 6000 ≈ 27 200 units.

The ratio those two numbers predict is about 3.6×. The measured ratio is 1.33.
**The gap is real and worth understanding rather than explaining away:** at
1024 blocks the SM holds 6 blocks at once, so a warp stalled at a barrier is
not idle SM time — the scheduler issues from the other five blocks' warps. The
block-level critical path is 3.6× worse; the *device* throughput is only 1.33×
worse, because oversubscription hides most of it. That is Module 1's latency
hiding doing its job, applied to synchronization latency rather than memory
latency.

Two supporting measurements, taken by editing `NBLK` (not part of the shipped
harness):

| grid | blocks/SM | ratio |
|---|---|---|
| 40 (one block per SM) | 1 | **1.40** |
| 240 (one full wave, full occupancy) | 6 | 1.10 |
| 1024 (oversubscribed) | 6 | 1.33 |

The trend is not monotonic and the honest answer is that it is not a clean
one-variable experiment: at 40 blocks there is nothing to hide behind, so the
structural advantage shows most clearly; at 240 the tail effect (a single wave,
so the kernel ends when the slowest block ends) compresses the difference; at
1024 there are enough waves that the per-block ratio reasserts itself with
hiding applied. If you predicted 3.6× and measured 1.33×, you reasoned
correctly about the critical path and did not account for oversubscription —
which is the more useful half of the lesson.

**The design rule to take away:** a barrier costs you *the block's* worst case
at every barrier. Point-to-point synchronization costs you *your own
dependency's* worst case. When the block is balanced, use the barrier: one
instruction, both guarantees, no spinning, no flags, less code. When the block
is badly skewed and the dependencies are narrow, the fence-plus-flag design is
worth its complexity — and measure, because oversubscription may already be
hiding the difference you are paying complexity to remove.

---

## Expected output

```
=== correctness (exact integer comparison over 262144 elements) ===
  variant_flag    wrong: 0 -> PASS
  variant_barrier wrong: 0 -> PASS

=== timing (min of 4 sweeps x 20 iterations, 1024 blocks) ===
  variant_flag    :   0.0947 ms
  variant_barrier :   0.1262 ms
  measured ratio  : 1.333  (variant_flag is faster)
  you predicted   : variant_flag, ratio 1.360 -> correct / correct

OVERALL: PASS
```

Observed ratios over four runs: 1.333, 1.356, 1.356, 1.362. The shipped
prediction is 1.36 and the ±30% window accepts anything from 0.93 to 1.73, so
a reader who predicts "the flag version, somewhere around 1.5×" passes and a
reader who predicts "3.6× from the critical path" fails the ratio check and is
then told, here, why.

With TODO 2 left blank, `variant_flag` reports 136 498 wrong and the program
still terminates and prints `OVERALL: FAIL`.

---

## The result that matters

The requirement "ensure the consumer observes the producer's writes" has two
correct answers and they are not interchangeable in cost. `__syncthreads()`
supplies both guarantees in one instruction and is the right default — but it
charges every warp in the block for the slowest warp in the block, at every
single barrier. `__threadfence_block()` supplies only the ordering guarantee
and charges nothing to anybody, which is sufficient precisely when some other
mechanism is already supplying the waiting. Choosing between them is not a
question about which primitive is faster; it is a question about whether the
dependency structure of your algorithm is all-to-all or point-to-point. This
exercise is the smallest program in which those two answers differ by a
measurable amount.

**Variation to try:** set `HEAVY` to 0 so that every thread does identical
work, and re-run. The skew disappears, the barrier's "cost" collapses to the
instruction itself, and `variant_barrier` should become the faster of the two —
it has no atomics, no spin loop, and half the shared-memory traffic on the flag
array. Predict the direction before you run it. Measured: `variant_flag`
0.0649 ms against `variant_barrier` 0.0186 ms -- the barrier version is
**3.49× faster** once the skew is gone. That flip is the proof that the 1.33×
was never about the price of `BAR.SYNC`.
