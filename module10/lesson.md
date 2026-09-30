# Module 10 — Race Conditions and Atomics

> Prerequisites: Module 1 (SMs, warps), Module 2 (launch, error checking),
> Module 3 (indexing), Module 4 (memory hierarchy, L1 non-coherence),
> Module 5 (sectors, transactions), Module 6–7 (shared memory, banks),
> Module 8 (warps, divergence, active masks), Module 9 (barriers, fences,
> memory ordering)
> What this module gives you: the ability to decide, for any concurrent
> update in a kernel, whether you need an atomic — and then to predict what
> that atomic will cost from the addresses it touches, before you run it.

Module 9 taught you how to make one thread's writes *visible* to another, and
how to make two threads *agree on an order*. This module is about the thing
those tools do not give you: **indivisibility**. It is the last module of
Part III, and it closes a loop opened in Module 4 — why L1 is not coherent
across SMs, and what the hardware does instead.

---

## Concept

### 1. `x += 1` is not an operation

Write this in a kernel:

```cpp
__global__ void racy_increment(int* c) { *c += 1; }
```

and compile it. The SASS for `sm_89` is, in its entirety:

```
LDG.E   R0, [R2.64] ;
IADD3   R5, R0, 0x1, RZ ;
STG.E   [R2.64], R5 ;
```

Three instructions. A **load**, an **arithmetic op**, and a **store**, with
two gaps between them in which the rest of the machine keeps running. The C
source has one `+=` and the hardware has no instruction that means `+=` on
memory unless you ask for one.

**PORTABLE CUDA CONCEPT.** A **read-modify-write (RMW)** is any update whose
new value depends on the old value. Executed concurrently by two threads
without protection, it is a **data race**: the C++ and CUDA memory models say
the program has undefined behaviour, and in practice what you observe is
**lost updates**.

### 2. Counting the losses

Trace two threads, A and B, doing `*c += 1` on a counter holding 0:

| time | thread A | thread B | memory |
|---|---|---|---|
| 0 | `LDG` → 0 | | 0 |
| 1 | | `LDG` → 0 | 0 |
| 2 | `IADD3` → 1 | `IADD3` → 1 | 0 |
| 3 | `STG` 1 | | 1 |
| 4 | | `STG` 1 | 1 |

Two increments, final value 1. One update is lost. Nothing was corrupted, no
fault was raised, and both threads believe they succeeded.

Now scale it up. The crucial step is that **a warp is not 32 independent
threads here**. All 32 lanes execute the same `LDG` in the same instruction,
so they all read the same value; they all execute the same `STG`, so they all
write the same value. From the memory system's point of view, a warp of 32
lanes incrementing one address contributes **at most 1**, no matter how many
lanes are active. `example01.cu` measures exactly this:

| blocks | threads | expected | observed | lost |
|---|---|---|---|---|
| 1 | 32 | 32 | 1 | 31 |
| 1 | 256 | 256 | 1 | 255 |
| 1 | 1024 | 1024 | 1 | 1023 |
| 4 | 256 | 1024 | 2 | 1022 |
| 40 | 256 | 10240 | 1 | 10239 |
| 1024 | 256 | 262144 | 13 | 262131 |

Read the 1024×256 row carefully. 262,144 threads produced the number 13.
Not a small error — the answer is off by a factor of 20,000. This is the
signature of an unprotected RMW on a hot address: not noise around the right
answer, but a result determined by *how many times the memory system happened
to be observed*, which is roughly the number of non-overlapping issue windows.

**The diagnostic rule.** A racy counter does not come out "a bit low". It
comes out catastrophically low, and it varies run to run. If your result is
within a few percent of correct, the bug is something else.

### 3. Why Module 9's tools do not fix this

This is the central conceptual distinction of the module. Barriers and fences
solve a *different* problem.

| Mechanism | What it guarantees | What it does not |
|---|---|---|
| `__syncthreads()` | Every thread of the block has reached this point, and every memory access they made before it is visible to the block after it | Nothing about what happens *between* two of your own instructions |
| `__threadfence()` | Accesses this thread made before the fence become visible device-wide before accesses it makes after | Same |
| `atomicX()` | This entire read-modify-write happens as one indivisible step with respect to every other atomic on the same address | Ordering with respect to *non*-atomic accesses (that still needs a fence) |

Ordering and visibility are properties of the *sequence* of operations.
Indivisibility is a property of *one* operation. A barrier can tell you that
everything before it has finished. It cannot stop a thread in another block
from executing its `LDG` in the two-instruction gap inside yours, because that
gap is not "before" or "after" anything — it is inside.

`example01.cu` proves it by brute force. This kernel:

```cpp
__global__ void barriered_increment(int* c)
{
    __syncthreads();
    __threadfence();
    *c += 1;
    __threadfence();
    __syncthreads();
}
```

gives 1, from 1024 threads, on every trial. And the SASS shows why:

```
BAR.SYNC.DEFER_BLOCKING 0x0 ;
MEMBAR.SC.GPU ;
LDG.E   R0, [R2.64] ;      <- the race is still here,
IADD3   R5, R0, 0x1, RZ ;  <- verbatim,
STG.E   [R2.64], R5 ;      <- untouched by either barrier
MEMBAR.SC.GPU ;
BAR.SYNC.DEFER_BLOCKING 0x0 ;
```

The three instructions are unchanged. All the synchronization did was
surround them.

> **If you remember one sentence from this module:** a fence orders your
> accesses, a barrier orders your threads, and neither makes any single
> update indivisible. Only an atomic does that.

### 4. The atomic API

An atomic performs the whole read-modify-write as one indivisible step and
**returns the value that was there before**. The return value is not a
nicety; it is what makes the whole family useful for allocation.

| Function | Effect (indivisibly) | Notes |
|---|---|---|
| `atomicAdd(p, v)` | `old = *p; *p = old + v` | `int`, `unsigned`, `unsigned long long`, `float`, `double` (sm_60+), `__half`/`__half2`/`__nv_bfloat16`/`__nv_bfloat162` (sm_70/sm_80+) |
| `atomicSub(p, v)` | `old = *p; *p = old - v` | `int`, `unsigned` only. For others, add the negation. |
| `atomicExch(p, v)` | `old = *p; *p = v` | The only one that is not a "modify"; also defined for `float` |
| `atomicMin` / `atomicMax` | `*p = min/max(*p, v)` | `int`, `unsigned`, `long long`, `unsigned long long` — **no float version** |
| `atomicAnd/Or/Xor` | bitwise | integer types |
| `atomicInc(p, limit)` | `old = *p; *p = (old >= limit) ? 0 : old+1` | **wrapping**, `unsigned` only |
| `atomicDec(p, limit)` | `old = *p; *p = (old == 0 \|\| old > limit) ? limit : old-1` | **wrapping**, `unsigned` only |
| `atomicCAS(p, cmp, v)` | `old = *p; if (old == cmp) *p = v` | returns `old` **always**, success or not |

Three traps hide in that table.

**`atomicInc` is not `atomicAdd(p,1)`.** It is a ring-buffer head pointer.
The cycle length is `limit + 1`, not `limit`: the counter visits
`0, 1, …, limit, 0, …`. `example01.cu` measures it: with `limit = 4`, five
steps return the counter to 0. A ring buffer of capacity `C` therefore takes
`atomicInc(head, C-1)`. Passing `C` is an off-by-one that produces no error
and costs you one slot at the first wrap.

**There is no `atomicMax` for `float`.** Section 6 is about building one.

**`atomicCAS` returns the old value, not a success flag.** `old == cmp`
means you won; `old != cmp` means you lost *and* `old` is the current truth.
Looping on the wrong one of those gives you either an infinite loop or a
silent exit after a swap that never happened.

**Relationship to Module 9's `cuda::atomic_ref`.** Module 9 introduced
`cuda::atomic_ref<T, Scope>` as the modern vocabulary for ordering, and used
`.store(v, memory_order_release)` / `.load(memory_order_acquire)` on flags.
The `atomicX()` intrinsics in this module are the same hardware operations
with **relaxed** ordering and **device** scope hard-wired:
`atomicAdd(p, v)` is exactly
`cuda::atomic_ref<int, cuda::thread_scope_device>(*p).fetch_add(v, cuda::memory_order_relaxed)`.
Relaxed means *atomicity only, no ordering* — which is precisely the
separation this module is built on. If you need an atomic **and** ordering,
you need both, either as `atomic_ref` with a stronger order or as an
intrinsic plus a fence. The `_block` and `_system` scoped variants
(`atomicAdd_block`, `atomicAdd_system`) exist for the same reason
`thread_scope_block` does: a narrower scope is cheaper, because a narrower
scope can be enforced closer to the SM.

### 5. The return value: tickets

```cpp
if (predicate) {
    int slot = atomicAdd(counter, 1);   // the value BEFORE my increment
    out[slot] = v;
}
```

Every thread that passes the predicate receives a **distinct** value, and the
values are **dense** from 0 upward. That is stream compaction in three lines,
and `example01.cu` verifies it over 100,000 elements: exactly 50,000 slots
filled, no duplicates, no holes.

What you do **not** get is order. The compacted output of `example01.cu`
begins `724 544 934 540 …`, which is not the input order of the even
elements. Uniqueness is guaranteed; ordering is not, and never will be.
Ordered compaction needs a prefix sum — Module 13.

### 6. `atomicCAS` is the universal primitive

Any atomic read-modify-write whatsoever can be built from compare-and-swap:

```cpp
old = *p;
do {
    assumed = old;                              // what I think is there
    desired = f(assumed);                       // what I want instead
    old = atomicCAS(p, assumed, desired);       // swap iff still assumed
} while (assumed != old);                       // retry with the truth
```

Read the loop condition carefully. `atomicCAS` returns the prior contents.
If they equal `assumed`, the swap happened and we are done. If not, `old`
now holds whatever someone else wrote, and the next iteration recomputes
`desired` **from that**, not from the stale first read. Recomputing from the
stale read is the classic bug; so is looping `while (old != desired)`, which
mistakes the new value for a success flag.

Two things this unlocks:

- **Operations the hardware lacks**, e.g. `atomicMax` on `float`, or an
  argmax that must break ties by index. Exercise 3.
- **Multi-field updates**, e.g. keeping a `{min, max}` pair consistent in one
  64-bit word. Two separate atomics would give the right final answer while
  permitting an observer to read a word whose halves came from different
  updates.

The cost is that a CAS loop **can retry**, and under heavy contention it
retries a lot. Escaping the loop by choosing a better *encoding* — so that a
single hardware atomic suffices — is usually worth more than optimizing the
loop. Exercise 3 makes you do both and compare.

---

## Hardware Mental Model

### Where an atomic executes, and why it must be there

Module 4 established that each SM has its own L1, and that **L1 is not
coherent across SMs**: SM 3 writing a line does not invalidate SM 17's copy.
Module 4 and Module 5 both promised this module would explain what that costs
you. Here it is.

A device-scope atomic cannot execute in L1, because the L1 of the SM doing it
is not a point that the other 39 SMs agree on. It must execute at the first
level of the hierarchy that is **shared by all SMs and single-valued for any
given address**. On this GPU that is the **L2**, which is physically banked
into slices, each slice owning a fixed subset of physical addresses by a hash
of the address bits.

**PORTABLE CUDA CONCEPT.** A global atomic is not "load, modify, store"
performed with a lock held. The operation is *shipped to the memory system*.
The requesting SM sends {address, opcode, operand} to the owning L2 slice;
the slice's ALU performs the RMW locally, and returns the old value if the
requester asked for it. No line is ever resident in the requesting SM's L1
for the duration. There is no lock, no window, and nothing to interleave —
which is exactly what "indivisible" means here.

Three consequences follow immediately, and they explain everything in the
measurement sections.

1. **A `__threadfence()` is not a substitute.** A fence makes your prior
   writes visible at the device scope. It does nothing about the fact that
   another SM read the value before you wrote it. Visibility ≠ atomicity.
2. **Atomic latency is L2 latency.** Module 4 measured L2 at ~241 cycles and
   L1 at ~40. An atomic never gets the 40.
3. **Atomic throughput is per-slice.** The unit that serializes is the L2
   slice's ALU for one address. Two atomics to *different* slices proceed in
   parallel; two to the *same address* cannot, by definition.

That third point is the whole economics of this module.

### Shared-memory atomics are a different instruction

`atomicAdd(&sharedArray[i], v)` compiles to `ATOMS`, not `ATOMG`. It executes
in the SM's shared-memory unit, which is private to the block anyway, so
there is no coherence problem to solve and no trip to L2. The scope is
correspondingly narrower: a shared-memory atomic is indivisible with respect
to other threads **of the same block** and says nothing to anyone else.

The Ada shared-memory unit also has dedicated hardware for the common cases.
`example02.cu`'s privatized kernel compiles its `atomicAdd(&s[i], 1u)` to a
single

```
ATOMS.POPC.INC.32 RZ, [R3.X4+URZ] ;
```

— one instruction that, for all the lanes of a warp targeting the same shared
address, counts them with a population count and applies the increment once.
Same-address contention *within a warp* is essentially free in shared memory.

**ARCHITECTURE-SPECIFIC OPTIMIZATION.** `ATOMS.POPC.INC` is an sm_7x+ Ada/
Ampere/Turing feature and applies to increment-by-one specifically. The
portable claim is the weaker one: shared-memory atomics are much cheaper than
global ones. The measurement below quantifies "much".

### The third instance of replay

Module 4 showed constant memory serializing one replay per distinct address
across a warp (a 24.6× penalty under a lane-varying index). Module 7 showed
shared memory serializing one replay per bank conflict. Atomic contention is
the **same architectural idea a third time**: a unit that can service one
request per cycle, presented with N requests that it cannot merge, takes N
cycles. In all three cases the cost is driven by *how many distinct
serialization events* the access pattern forces, and in all three cases the
fix is to change the addresses, not the instruction.

### ATOM vs RED: the optimization you get for free

**PORTABLE CUDA CONCEPT, with an architecture-specific encoding.** There are
two machine instructions, not one:

| SASS | Meaning |
|---|---|
| `ATOMG.E.ADD.STRONG.GPU PT, R5, [R4.64], R9` | atomic; **returns** the old value into `R5` |
| `RED.E.ADD.STRONG.GPU [R4.64], R7` | **red**uction; performs the update, returns nothing |

`RED` is fire-and-forget. The SM issues it and never waits: there is no
destination register, so no scoreboard dependency, so no stall. `ATOMG` has a
return value, which means a result must come back from L2 and the consuming
instruction must wait for it.

**The compiler emits `RED` whenever you discard the return value.** This is
real, it is free, and almost nobody knows it is happening. Verify it
yourself:

```
nvcc -arch=sm_89 -O3 -cubin -o example02.cubin example02.cu
cuobjdump -sass example02.cubin
```

From that dump, unedited:

```
Function : _Z8k_spreadPjj              // atomicAdd(&c[i & mask], 1u);
    RED.E.ADD.STRONG.GPU [R2.64], R5 ;

Function : _Z12compact_evenPKiiPiS1_   // int slot = atomicAdd(counter, 1);
    ATOMG.E.ADD.STRONG.GPU PT, R3, [R2.64], R9 ;
```

Same source function, different instruction, decided purely by whether you
used the result. **Do not assign an atomic's return value to a variable you
do not need.** It is the cheapest optimization in CUDA.

### Warp aggregation, which the compiler may already have done

Look at this SASS for `atomicAdd(&c[0], 1u)` — a literal address, return
value discarded:

```
Function : _Z9k_literalPj
    VOTEU.ANY UR4, UPT, PT ;              // which lanes are active?
    UFLO.U32  UR5, UR4 ;                  // lowest set bit -> the leader
    POPC      R5,  UR4 ;                  // how many lanes are there?
@P0 RED.E.ADD.STRONG.GPU [R2.64], R5 ;    // ONE atomic, adding the count
```

The compiler proved the address is uniform across the warp, and replaced 32
same-address atomics with one atomic of the summed value. It does this even
when you *use* the return value — in `compact_even` it follows the `ATOMG`
with a `POPC` of the prefix mask and a `SHFL.IDX` to hand each lane its own
correct ticket.

This is a **compiler** optimization, not a hardware one, and it needs the
address to be provably warp-uniform. Change `&c[0]` to `&c[i & mask]` with
`mask` a runtime argument that happens to be 0, and the traffic is identical
but the transformation is impossible. `example02.cu` measures both:

| kernel | ms for 2^20 atomics |
|---|---|
| `atomicAdd(&c[0], 1u)` — literal | 0.0197 |
| `atomicAdd(&c[i & mask], 1u)`, `mask == 0` at runtime | 0.5424 |

**27.6× for identical memory traffic.** This is why you should never reason
about atomic cost from the source alone.

**ARCHITECTURE-SPECIFIC.** `atomicMax` with a discarded return and a uniform
address goes further and emits `REDUX.MAX.S32`, a warp-level reduction
instruction, before the single `RED`.

---

## Code Walkthrough

### `example01.cu` — the race, and the pieces of the API

Parts A/B/C are the three tables above: race, race-with-barriers, atomic.
Part D is ticket allocation. Part F is `atomicInc`'s wrap. Part E deserves
its own discussion.

**Part E: `atomicAdd` on `float` is correct and not reproducible.**

```cpp
__global__ void float_sum_atomic(const float* x, int n, float* acc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(acc, x[i]);
}
```

Every individual add is indivisible. But floating-point addition is **not
associative**: `(a+b)+c` and `a+(b+c)` differ in the last bits, and an atomic
imposes *some* order without promising *which* order. Run it ten times on the
same 2^20 inputs:

```
    run        bit pattern          value      abs err
      0       0x48FFBCBE  523749.937500    5.392e+00
      1       0x48FFBCEF  523751.468750    3.861e+00 <- differs from run 0
      2       0x48FFBCE9  523751.281250    4.048e+00 <- differs from run 0
      ...
      9       0x48FFBCC7  523750.218750    5.111e+00 <- differs from run 0

  distinct bit patterns across 10 identical runs: 10
```

Ten runs, ten different answers. All ten are correct sums; none is *the* sum.

Consequences you will meet again:

- A kernel with `float`/`double`/`half` `atomicAdd` is **not bit-reproducible
  across runs on the same machine with the same input**. If you need
  reproducibility (regression tests, debugging a divergence between two
  training runs, certification), you cannot use one.
- The error is not bounded by one rounding. It accumulates: the observed
  absolute error against a `double` reference ranges over 3.3–5.4 here, on a
  sum of magnitude 5×10^5.
- The deterministic alternative is a **fixed-order reduction** — a tree with
  a fixed shape, or integer/fixed-point accumulation. Module 12 builds the
  tree; Part XIV (Modules 41-42, CUDA for AI and LLM inference kernels)
  returns to this as *the* reason ML training runs do not
  reproduce bit-for-bit.

Note also that the SASS for the float add is
`RED.E.ADD.F32.FTZ.RN.STRONG.GPU`: **FTZ**, flush-to-zero. Global float
atomics on NVIDIA hardware flush denormals regardless of your `-ftz` setting.

### `example02.cu` — the contention curve

Every configuration executes **exactly 1,048,576 atomic instructions**. Only
the address pattern changes. Timing follows spec §12: all configurations
back-to-back in one sweep, min of 4 sweeps × 20 iterations, validation in a
separate pass.

**A. Distinct addresses.** `atomicAdd(&c[i & mask], 1u)`, K = mask+1:

| K | ms | Gatomic/s | vs K=2^20 | non-atomic store | atomic/store |
|---|---|---|---|---|---|
| 1 | 0.5424 | 1.93 | 55.8× | 0.0113 | 47.9× |
| 2 | 0.5399 | 1.94 | 55.5× | 0.0136 | 39.6× |
| 4 | 0.2740 | 3.83 | 28.2× | 0.0108 | 25.4× |
| 8 | 0.1399 | 7.50 | 14.4× | 0.0082 | 17.0× |
| 16 | 0.1035 | 10.13 | 10.6× | 0.0093 | 11.1× |
| 32 | 0.0868 | 12.08 | 8.9× | 0.0095 | 9.2× |
| 256 | 0.0449 | 23.35 | 4.6× | 0.0094 | 4.8× |
| 1024 | 0.0132 | 79.38 | 1.4× | 0.0087 | 1.5× |
| 4096 | 0.0091 | 115.71 | 0.93× | 0.0096 | 0.9× |
| 65536 | 0.0079 | 132.99 | 0.81× | 0.0109 | 0.7× |
| 2^20 | 0.0097 | 107.79 | 1.00× | 0.0101 | 1.0× |

Read the last column first. It compares the atomic against a **racy plain
store** doing the same traffic — i.e. against the cost of the memory
operation with the atomicity removed. At K ≥ 4096 the ratio is 1.0: **an
uncontended atomic is free.** At K = 1 it is 48×. Every bit of the cost of a
contended atomic is contention. "Atomics are slow" is false; "concurrency on
one address is slow" is true, and would be true of any mechanism.

The curve is not smooth, and the kink at K=2 is real: K=1 and K=2 cost the
same, then K=4 halves. Investigate it by spreading the bins out.

**B. Where the addresses land.** Same K, bins `stride` words apart (ms):

| K | stride 1 (4 B) | stride 8 (32 B) | stride 32 (128 B) | stride 1024 (4 KB) |
|---|---|---|---|---|
| 1 | 0.5427 | 0.5425 | 0.5426 | 0.5422 |
| 2 | 0.5419 | 0.8112 | 0.6350 | **0.2743** |
| 4 | 0.2742 | 0.6761 | 0.3249 | **0.1391** |
| 8 | 0.1399 | 0.6761 | 0.3234 | **0.0702** |
| 16 | 0.1035 | 0.3409 | 0.1674 | **0.0370** |
| 32 | 0.0868 | 0.3416 | 0.0819 | 0.0423 |

With 4 KB spacing, K = 2 *does* halve — 0.5422 → 0.2743 — and keeps halving
cleanly to K = 16. Adjacent words do not, because the L2 slice hash maps
neighbouring words to the same slice: two "distinct addresses" that share a
slice share its ALU and do not run in parallel.

Stride 8 (32 B, one sector apart) is *worse* than stride 1 at small K. Two
bins one sector apart force two sector fills where packed bins forced one,
and the slice mapping still does not separate them.

**The rule that survives:** contention is governed by the number of distinct
**L2 slices** your hot addresses hash to, not by the number of distinct
addresses. `K` is an upper bound on your parallelism, never a guarantee.

**C. Same-address concurrency is the real variable.** Compare a lane-varying
address against one that is uniform across each warp:

| K | lane-varying ms | warp-uniform ms | ratio |
|---|---|---|---|
| 1 | 0.5424 | 0.5421 | 1.00× |
| 8 | 0.1399 | 0.5422 | 3.88× |
| 32 | 0.0868 | 0.5423 | 6.25× |
| 1024 | 0.0132 | 0.0870 | 6.59× |
| 2^20 | 0.0097 | 0.0627 | 6.44× |

The warp-uniform column is **flat at 0.542 ms for every K up to 64**. A
million distinct addresses in the array buys nothing if each warp's 32 lanes
all pile onto one of them. Even at K = 2^20 — where 32,768 warps each own a
private address — making the address warp-uniform costs 6.4×.

**D/E. The two ways out.**

| K | plain ms | privatized ms | speedup | warp-aggregated ms | speedup |
|---|---|---|---|---|---|
| 1 | 0.5424 | 0.0113 | **48.2×** | 0.0196 | **27.7×** |
| 4 | 0.2740 | 0.0125 | 21.9× | 0.0365 | 7.5× |
| 16 | 0.1035 | 0.0109 | 9.5× | 0.0534 | 1.9× |
| 32 | 0.0868 | 0.0133 | 6.5× | 0.0870 | 1.00× |
| 256 | 0.0449 | 0.0451 | 1.00× | 0.0451 | 1.00× |
| 1024 | 0.0132 | 0.0145 | 0.91× | 0.0289 | 0.46× |
| 4096 | 0.0091 | 0.0416 | **0.22×** | 0.0289 | 0.31× |

Both techniques are contention-reduction techniques, and **both become
pessimizations once the contention is gone.** At K = 4096 privatization is
4.5× *slower* than doing nothing, because each block now pays 4096 shared
writes to zero its copy and up to 4096 global atomics to flush it, in
exchange for removing contention that was not there.

**Shared vs global, directly:** at K = 1 the privatized kernel — which does
the same 2^20 updates plus the zeroing and the flush — runs in 0.0113 ms
against the global kernel's 0.5424 ms. **48× for identical semantics.**

### Privatization: the traffic arithmetic

The pattern, in general:

1. Each block allocates a **private copy** of the accumulator in shared memory.
2. Zero it. `__syncthreads()`.
3. Accumulate into the private copy with **shared-memory** atomics.
4. `__syncthreads()`.
5. Flush: one **global** atomic per bin per block.

Both barriers are required and they are required for different reasons — the
first for the zeroing to be visible, the second for the accumulation to be
complete. Module 9 gives you the vocabulary; note that neither barrier
removes the need for the atomics in step 3, which is this module's whole
point restated.

The traffic argument, for `n` updates over `b` bins with `g` blocks:

| | global atomics | shared atomics |
|---|---|---|
| naive | `n` | 0 |
| privatized | `g · b` (at most) | `n` |

Privatization pays when `g · b ≪ n` **and** the naive version was actually
contended. Both conditions matter. With `n = 2^20`, `b = 1` and `g = 4096`
you trade 1,048,576 heavily-contended global atomics for 1,048,576 cheap
shared atomics plus 4,096 global ones — a 256× reduction in global atomic
traffic, and the measured 48×. With `b = 4096` and the same `g` you would
trade 1,048,576 global atomics for **16,777,216** of them, which is why that
row of the table reads 0.22×.

**Forward reference:** Module 14 builds the full privatized histogram — bin
replication to spread residual contention, handling more bins than fit in
shared memory, and the coarsening ladder. This module owns the mechanism and
the arithmetic; Module 14 owns the algorithm.

### Reducing atomic pressure: the four moves

1. **Do not use the return value** unless you need it. `RED` instead of
   `ATOM`. Free.
2. **Aggregate within the warp** before the atomic. `__match_any_sync` gives
   each lane the mask of lanes sharing its target; one leader per group does
   one atomic carrying the group's total. The compiler already does this when
   it can prove uniformity; do it by hand when it cannot. Worth 27.7× at
   K = 1 and **0.46× at K = 1024** — it is not free, so gate it on knowing
   you have contention.
3. **Privatize**, per block or per warp, as above.
4. **Spread the addresses.** Pad your bins apart, or hash the index, so that
   hot bins land on different L2 slices (Part B's 4 KB column). This is
   Module 5's row-pitch argument and Module 7's bank-padding argument, one
   level further out — same idea, third venue.

And the move that beats all four: **use the return value to allocate once
instead of updating many times.** One `atomicAdd` returning a base index, and
then plain stores into `[base, base+k)`, replaces `k` atomics with 1.

---

## Check Your Understanding

Answers in `solutions/module10/check_your_understanding.md`.

**Q1.** A kernel has 256 threads in one block, all executing
`shared_counter += 1` on a `__shared__ int` with a `__syncthreads()` before
and after. Your colleague says the barriers make it safe because "all threads
are synchronized". Give the precise reason they are wrong, and then say what
value the counter will actually hold and why that specific number.

**Q2.** Two kernels, both doing 2^20 `atomicAdd`s, both with the return value
discarded. Kernel P spreads them over 1024 consecutive `unsigned int` bins.
Kernel Q spreads them over 32 bins that are 4 KB apart. Naively, P has 32×
more parallelism. From the measurements above, predict which is faster and
explain the mechanism. Then explain why the answer would change if the 32
bins in Q were 32 bytes apart instead.

**Q3.** You replace `atomicAdd(&hist[b], 1)` with a shared-memory privatized
version and the kernel gets *slower*. The input, the grid and the number of
bins are unchanged. Give two distinct mechanisms that could cause this, and
for each, one measurement that would distinguish it from the other.

**Q4.** A CAS loop is written as

```cpp
unsigned old = *p;
do {
    unsigned desired = f(old);
    old = atomicCAS(p, old, desired);
} while (old != desired);
```

It compiles and usually produces the right answer. State precisely what is
wrong with the loop condition, construct an interleaving of two threads in
which it terminates having performed no swap, and say what the resulting bug
looks like from outside the kernel.

---

## Exercises

### Exercise 1 — `exercise01.cu` (debugging; 5 TODOs)

`triage` scans 4,000,000 events, counts them into 8 severity classes,
maintains a global checksum, and appends critical events to a bounded list.
Three symptoms are reported; the file states them and not their causes. There
are more defects than symptoms, and they are not all the same class of bug.

```
nvcc -arch=sm_89 -O3 -lineinfo -o exercise01.exe exercise01.cu
compute-sanitizer --tool racecheck .\exercise01.exe --racecheck
.\exercise01.exe
```

TODO 1 is a **prediction** about what `racecheck` will report, committed
*before* you look, and scored. Module 9 introduced the tool and found every
hazard it went looking for; this exercise is the counterexample, and knowing
where the counterexample lives is the difference between using a sanitizer
and trusting one. Before you run it, decide from first
principles what a race detector can and cannot instrument, and read the
one-line description the tool gives of itself in `compute-sanitizer --help`.
TODOs 2–5 are the fixes, stated as requirements rather than as mechanisms.
TODO 5 contains a requirement that pulls in two directions at once; reconcile
it.

Validation: exact per-class counts, exact checksum, a fully-populated
duplicate-free critical list, a critical *count* that survives overflow of
the list, and identical results across three consecutive runs.

### Exercise 2 — `exercise02.cu` (design + performance reasoning; 4 TODOs)

A weighted category tally with a heavily skewed distribution — one category
carries half the weight. `tally_naive` does one global atomic per sample.
Write `tally_fast`.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

**TODO 2 is the design TODO and does not name a technique.** The requirement
is "reduce the number of concurrent global atomic operations that target the
same address, without changing the result". The harness runs four scenarios —
`{16, 4096}` categories × `{clustered, shuffled}` input — precisely because
the one-line answers each win one column and lose another. `ncat` is a
runtime argument. TODO 4 asks you to predict two speedup ratios before
building; they will not be remotely similar, and the second is very sensitive
to a decision that barely moves the first.

Validation: exact integer results in all four scenarios, ≥ 4× on the
contended one, ≥ 2× on the shuffled one, ≥ 2.5× on both `ncat = 4096`
orders, and both predictions within a factor of 2.

### Exercise 3 — `exercise03.cu` (`atomicCAS`; 5 TODOs; hardest)

Build three things the hardware does not provide: an order-preserving
`float → uint32` key, an argmax by CAS loop with ties broken to the smallest
index, and a `{max, min}` pair updated consistently in one 64-bit atomic.
Then make the CAS loop unnecessary by choosing a better encoding, and make
the whole thing fast.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

Three datasets: mixed signs with a three-way tie, **all values strictly
negative**, and all values identical. Dataset B is where the well-known
`atomicMax((int*)p, __float_as_int(v))` shortcut silently returns the wrong
answer; dataset C is where a CAS loop with the wrong exit condition spins
forever. Work out both on paper before writing code.

Validation: correct index on all three datasets from all three kernels,
correct bounds, monotonicity of your key checked over 4096 sampled floats
including infinities and denormals, and `argmax_fast` at least 5× the naive
kernel.

---

## Prediction

Commit to these in writing before you build anything.

**P1.** In Part A of `example01.cu`, one block of 1024 threads increments a
counter racily. You predicted the answer would be "some number well below
1024". Predict the actual number, to within a factor of 2, and justify it
from how many *instructions* the warps issue, not from how many threads
there are.

**P2.** `example02.cu` times `atomicAdd(&c[i & mask], 1u)` with mask = 0
against a plain racy `c[i & mask] += 1u` with the same mask. Both touch one
address from 2^20 threads. Predict the ratio of their run times, and say
which is faster. Then predict the ratio at mask = 0xFFFFF (2^20 bins), and
say what the change in that ratio tells you about what an atomic actually
costs.

**P3.** In the privatization table, shared-memory privatization beats the
plain global atomic by 48× at K = 1 and loses to it by 4.5× at K = 4096.
There is a crossover. Predict the K at which the two are equal, from the
traffic arithmetic and the grid size (4096 blocks of 256 threads, 2^20
updates), before looking at the table.
