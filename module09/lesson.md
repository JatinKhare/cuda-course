# Module 9 — Synchronization: Barriers, Fences, and Memory Ordering

> Prerequisites: Module 1 (SMs, blocks, warps, block scheduling), Module 2
> (launch, error checking), Module 4 (memory spaces, L1 non-coherence),
> Module 6 (shared memory, tiling), Module 7 (bank conflicts), Module 8
> (warps, divergence, active masks, reconvergence, independent thread
> scheduling)
> What this module gives you: the ability to look at any CUDA kernel and say,
> for every synchronization point in it, *which* guarantee it supplies, *which*
> guarantee the code actually needs, and whether the two match.

Since Module 6 you have been writing `__syncthreads()` into tiled kernels and
being told "barrier; Module 9 makes this precise." This is that module. You have
also been told, in Module 4, that `volatile` is not synchronization and that
Module 9 would explain why. That debt is paid here too.

The central claim of this module is that **`__syncthreads()` is two things at
once**, that most CUDA programmers know only one of them, and that the one they
do not know is the one that makes tiling correct.

---

## Concept

### The two guarantees

**PORTABLE CUDA CONCEPT.** `__syncthreads()` makes exactly two promises. Learn
them as two separate sentences, because bugs violate them separately.

> **G1 — execution barrier.** No thread of the block proceeds past the barrier
> until every non-exited thread of the block has reached it.
>
> **G2 — memory fence at block scope.** Every write to shared memory and to
> global memory performed by a thread of the block *before* the barrier is
> visible to every thread of the block *after* the barrier.

G1 alone is useless. Suppose the hardware gave you G1 and nothing else: all 256
threads rendezvous, and then thread 0 reads `s[200]`. Thread 200 executed its
store before the rendezvous, so it "happened" — but "happened" in whose view?
The store may still be sitting in a store queue; the compiler may have decided
to keep `s[200]`'s value in a register and never emit the store at all; thread
0's read may have been hoisted above the barrier by the compiler. G2 is the
promise that none of that can happen. It is a promise about *visibility*, and
it is the reason the tiled kernels of Module 6 are correct.

G2 alone is also useless, and this is the half people skip. A fence says "my
prior writes are visible to the block *once you look*." It does not say anybody
has looked, or that anybody has even run yet. If warp 7 has not been issued a
single instruction, no fence executed by warp 0 conjures warp 7's data into
existence. That is what G1 is for.

The diagnostic question for any suspected synchronization bug is therefore:

| Symptom | Guarantee violated |
|---|---|
| A thread read a slot whose owner had not run yet | G1 |
| A thread read a slot whose owner *had* run, but got a stale value | G2 |
| A thread overwrote a slot another thread was still reading | G1 (the readers had not finished) |
| A thread hung | G1 (some threads never arrived) |

The fourth row is the one everybody can already diagnose. The interesting bugs
are in rows 1–3.

### Which guarantee does a given barrier supply the *work* for?

`__syncthreads()` always supplies both. The question that matters for design is
which one your code *depends on*, because that determines whether a cheaper
primitive would do. Two canonical cases from tiling:

```cpp
for (int tile = 0; tile < nTiles; ++tile) {
    s[t] = g[tile * TPB + t];      // stage
    __syncthreads();               // (1) read-after-write:  needs G1 AND G2
    acc += f(s[...]);              // use other threads' slots
    __syncthreads();               // (2) write-after-read:  needs G1 ONLY
}
```

Barrier (1) is a read-after-write (RAW) hazard across threads: I must wait for
you to write (G1) *and* I must see what you wrote (G2).

Barrier (2) is a write-after-read (WAR) hazard: on the next iteration I will
overwrite `s[t]`, and I must not do that while you are still reading the old
value. What do I need? I need you to have *finished reading*. Reads publish
nothing, so there is nothing for G2 to make visible. Barrier (2) is a pure
execution barrier. Recognising this matters, because a WAR barrier can often be
*deleted* rather than kept — by double-buffering the tile, so that the write
lands in a buffer nobody is reading. That trade (one extra buffer of shared
memory for one fewer barrier per iteration) is Exercise 1's design TODO and is
the foundation of the software pipelining you will meet in Module 17.

### The undefined-behavior rule

**PORTABLE CUDA CONCEPT.** From the CUDA C++ Programming Guide, on
`__syncthreads()`:

> `__syncthreads()` is allowed in conditional code but only if the conditional
> evaluates identically across the entire thread block, otherwise the code
> execution is likely to hang or produce unintended side effects.

The operative word is **identically across the entire thread block**. A barrier
is a property of the *block*, and the block has one barrier resource per
barrier site. If the block splits and only some threads arrive, the barrier's
arrival count and the block's thread count disagree, and the CUDA memory model
simply stops defining what happens.

Three canonical shapes, all undefined:

```cpp
// (a) barrier inside a thread-dependent condition
if (threadIdx.x < 64) { ...; __syncthreads(); }          // blockDim.x == 256

// (b) barrier inside a loop with a thread-dependent trip count
for (int i = 0; i < trips[threadIdx.x]; ++i) { ...; __syncthreads(); }

// (c) barrier after an early return taken by some threads
if (gid >= n) return;
...
__syncthreads();
```

A barrier in a **block-uniform** branch is completely fine:

```cpp
if (blockIdx.x & 1) { ...; __syncthreads(); }            // uniform per block
if (nTiles > 4)     { ...; __syncthreads(); }            // uniform: same for all
```

because every thread of the block evaluates the condition the same way, so
either all of them arrive or none of them do. Note that "uniform" means uniform
*within a block*; different blocks may take different sides, because barriers
never span blocks.

#### What the guide says about *exited* threads — read this carefully

G1 as stated above says "every **non-exited** thread." That wording is
deliberate and it is the source of the most confusing behavior in this whole
area. A thread that has returned from the kernel is removed from the barrier's
expected-arrival set. Consequently case (c) above frequently does **not** hang:
the threads that returned early are no longer counted, so the survivors'
barrier completes.

That does not make case (c) correct. It makes it *undiagnosable by hanging*.
The program is still undefined by the language rule quoted above, it is still
computing with data from a block that silently changed size, and the next
toolkit, the next architecture, or the next inlining decision may schedule it
differently. **Do not write code whose correctness depends on the exited-thread
rule.**

There is a second, lower-level reason that divergent barriers often fail to
hang, and you need it to reason about what you will actually observe. The PTX
`barrier.sync` / `bar.sync` instruction that `__syncthreads()` compiles to
counts **warps, not threads**: if any thread of a warp arrives at the barrier,
the whole warp is counted as having arrived. That is why
`if (threadIdx.x < 16) __syncthreads();` in a **32-thread block** completes —
the block's one warp does arrive. Exercise 1 makes you predict exactly this.

**The practical consequence, and it is the most important sentence in this
module:** on sm_89 with CUDA 13.2, the usual failure mode of a divergent
barrier is **silent wrong answers, not a hang**. Chasing "it didn't hang, so
the barrier must be fine" is how these bugs survive into production.

### The barrier-plus-reduction variants

Three intrinsics do a barrier *and* a block-wide reduction of a per-thread
predicate, returning the same value to every thread:

| Intrinsic | Returns |
|---|---|
| `__syncthreads_count(pred)` | the number of threads in the block for which `pred != 0` |
| `__syncthreads_and(pred)` | non-zero iff `pred != 0` for **all** threads of the block |
| `__syncthreads_or(pred)` | non-zero iff `pred != 0` for **at least one** thread |

All three carry the full G1 + G2 semantics of `__syncthreads()` and all three
are subject to the same uniformity rule.

They are not a convenience. They are the mechanism that makes an *iterative*
kernel legal. Consider a block-local convergence loop:

```cpp
int active;
do {
    ... one sweep, which needs a barrier inside it ...
    active = __syncthreads_count(i_changed_something);
} while (active > 0);                     // block-uniform by construction
```

The loop condition must be identical in every thread, or the barrier inside the
loop body is a case-(b) violation. `__syncthreads_count` guarantees that by
construction: its return value is a block-wide reduction, so it is the same
number in every thread. Writing the same thing with a shared counter and a
separate `__syncthreads()` is three instructions instead of one and is easy to
get subtly wrong. `__syncthreads_or(work_remaining)` is the canonical
"is anyone still busy?" flag and `__syncthreads_and(converged)` the canonical
"has everyone converged?" test.

### `__syncwarp(mask)` — the warp-scope barrier

Module 8 established that on sm_70 and later, **independent thread scheduling**
gives each thread its own program counter, and the hardware may schedule
diverged sub-groups of a warp independently and interleave them arbitrarily.
Lanes of one warp are no longer guaranteed to be at the same instruction.

Everything that used to be free therefore is not. Pre-Volta, this was correct
by construction:

```cpp
s[lane] = ...;              // one PC per warp, so this store
x = s[lane ^ 1];            // provably retired before this load issued
```

On sm_70+ it is undefined. The fix is a **warp-scope barrier**:

```cpp
s[lane] = ...;
__syncwarp();               // default mask 0xffffffff
x = s[lane ^ 1];
```

`__syncwarp(mask)` is G1 + G2 restricted to the lanes named in `mask`. Its
rules:

- Every lane named in `mask` must execute the *same* `__syncwarp` call with the
  *same* mask. Naming a lane that is not going to reach this call is undefined,
  exactly as with `__syncthreads()`.
- It is much cheaper than `__syncthreads()` — no block-wide rendezvous, no
  traffic to the block's barrier resource — but it synchronizes *nothing*
  outside the warp.
- **You do not need it** when the only data flow is a lane reading its own
  data, or when the cross-lane communication is done by an intrinsic that
  carries its own synchronization: `__shfl_sync`, `__ballot_sync`,
  `__any_sync` and friends take a mask and synchronize the named lanes
  themselves. That is why the `_sync` suffix exists and why the non-`_sync`
  forms were removed.
- **You do need it** whenever lanes communicate through *memory* — shared or
  global — rather than through a `_sync` intrinsic.

The trap, and Exercise 1 will spring it on you: on this GPU, a warp-local
shared-memory exchange with no `__syncwarp()` *produces the right answer*.
The eight lanes' stores and loads happen to be issued in order by the
load/store unit. That is a fact about Ada's memory pipeline, not a fact about
your program. `compute-sanitizer --tool racecheck` reports the race anyway;
the solution notes show it doing so.

### Fences are not barriers

**PORTABLE CUDA CONCEPT.** A **fence** orders *one thread's own* memory
operations as seen by other threads. It makes nobody wait.

| | makes threads wait | orders my accesses |
|---|---|---|
| `__syncthreads()` | yes, block | yes, block |
| `__syncwarp(mask)` | yes, warp | yes, warp |
| `__threadfence_block()` | **no** | yes, block |
| `__threadfence()` | **no** | yes, device |
| `__threadfence_system()` | **no** | yes, host + all devices |

Precisely: `__threadfence_block()` guarantees that all writes the calling
thread made before the fence are observed by every thread *of its block* to
have occurred before all writes it makes after the fence, and similarly that
reads before the fence are not satisfied by values from after it.
`__threadfence()` extends the observer set to every thread on the device;
`__threadfence_system()` extends it to the host and to peer devices.

Two rules follow:

- **You need a fence but not a barrier** when the waiting is already handled by
  something else — typically a flag that the consumer spins on — and all you
  need is that the payload be visible before the flag is. This is the classic
  publish/subscribe shape:

  ```cpp
  payload[...] = ...;          // produce
  __threadfence_block();       // publish the payload BEFORE the flag
  flag = 1;                    // advertise
  ```

  without which the flag store may become visible before the payload stores,
  and the consumer reads garbage even though it waited correctly.

- **You need a barrier but not a fence** essentially never — every CUDA barrier
  includes the fence at its own scope. But you need a *barrier rather than a
  fence* whenever the waiting is not handled by anything else, which is almost
  every tiling kernel. Substituting `__threadfence_block()` for
  `__syncthreads()` in a tiled kernel compiles, runs, and produces wrong
  answers, because it makes nothing wait. Example 2 measures exactly that.

Scope costs money. `__threadfence()` has to push stores out to the L2 (the
device-wide coherence point Module 4 identified, because L1 is not coherent
across SMs); `__threadfence_block()` only has to order accesses within one SM's
L1 and shared memory. Use the narrowest scope that is correct.

### `volatile` is not synchronization

**PORTABLE CUDA CONCEPT, and the debt Module 4 promised.**

`volatile` in CUDA C++ means what it means in C: the compiler must not cache
the object in a register, must not eliminate accesses to it, and must not
reorder *volatile* accesses with respect to *each other*. That is the whole
specification. In particular `volatile` gives you:

- no execution ordering between threads (no G1),
- no visibility guarantee across threads (no G2),
- no ordering with respect to *non-volatile* accesses,
- no atomicity.

It is a compiler directive. Synchronization is a hardware and memory-model
problem. They are different layers.

The legacy idiom this kills is the warp-synchronous reduction tail:

```cpp
// BROKEN on sm_70 and later. Do not write this.
if (tid < 32) {
    volatile float* v = sdata;
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid +  8];
    ...
}
```

This was correct before Volta for one reason only: a warp had a single program
counter, so all 32 lanes executed `v[tid] += v[tid+32]` before any lane
executed `v[tid] += v[tid+16]`. The `volatile` was there solely to stop the
compiler from keeping `v[tid]` in a register across the lines — it was
*never* providing the synchronization. The synchronization was free hardware
lockstep.

Independent thread scheduling (Module 8) removed the lockstep. The `volatile`
is still doing its one job, and that job was never the important one. The
correct modern forms are `__syncwarp()` between the steps, or — better, since
the data never needs to be in memory — `__shfl_down_sync`, which Module 30
covers. Example 1 runs the `volatile` version of a shared-memory rotate next to
the barrier version and reports the error count of each; on this GPU the
`volatile` version is wrong for about 43% of elements.

### The modern vocabulary: `cuda::atomic_ref`, `cuda::barrier`, scopes

**PORTABLE CUDA CONCEPT.** Since CUDA 10.2 there is a *formal* memory model for
CUDA, aligned with the C++11 memory model, with one addition: every operation
carries a **thread scope** as well as a memory order. The libcu++ headers
expose it:

```cpp
#include <cuda/atomic>

cuda::atomic_ref<int, cuda::thread_scope_block> f(flag);
f.store(1, cuda::memory_order_release);          // publish
while (f.load(cuda::memory_order_acquire) == 0) {}   // subscribe
```

- **Memory orders**: `memory_order_relaxed` (atomicity only, no ordering),
  `memory_order_acquire`, `memory_order_release`, `memory_order_acq_rel`,
  `memory_order_seq_cst`. A release store paired with an acquire load on the
  same object gives you exactly the publish/subscribe guarantee that the
  `__threadfence_block()` + plain store idiom gives you, but says so in the
  type system instead of in a comment.
- **Thread scopes**: `cuda::thread_scope_thread`, `_block`, `_device`,
  `_system`. These are the same four scopes as the `__threadfence*` family,
  now attached to individual operations rather than to a separate fence.
  `cuda::atomic_thread_fence(order, scope)` is the standalone fence form.
- `cuda::barrier<cuda::thread_scope_block>` is a reusable, *split* barrier:
  you can `arrive()` and later `wait()`, doing useful work in between, which is
  what makes asynchronous copy pipelines possible. Module 29 and Module 32 use
  it; here you only need to know it exists.

The old intrinsics are not deprecated and are not going away; they are what the
new vocabulary compiles to. Exercise 3 uses `cuda::atomic_ref` for a flag and
leaves the ordering decision to you deliberately, because writing
`store(1, memory_order_release)` and writing `__threadfence_block(); store(1,
memory_order_relaxed);` produce the same machine code and the same guarantee,
and knowing that is the point.

> Build note: `<cuda/atomic>` requires `-std=c++17`, and on MSVC also
> `-Xcompiler /Zc:preprocessor`. Example 2 and Exercise 3 say so in their
> headers.

### Cooperative groups, briefly

**PORTABLE CUDA CONCEPT.** Cooperative groups make the set of threads a barrier
acts on into a *value*:

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

cg::thread_block block = cg::this_thread_block();
block.sync();                                   // == __syncthreads()

cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
warp.sync();                                    // == __syncwarp() with the
                                                //    partition's mask
int lane = warp.thread_rank();
```

`block.sync()` compiles to the same `BAR.SYNC` as `__syncthreads()`. There is
no performance argument here. The argument is interface: because the group is a
value, a device function can take it as a parameter and thereby *state in its
signature* which threads must call it. `__syncthreads()` buried inside a
`__device__` function is an unwritten contract with every caller;
`void f(cg::thread_block g)` is a written one. That is why the explicit-group
style is the modern recommendation.

There is also `cg::this_grid()` with `grid.sync()`, a grid-wide barrier. Be
clear about what it costs:

- it requires the kernel to be launched with `cudaLaunchCooperativeKernel`, not
  `<<<>>>`;
- it requires **every block of the grid to be co-resident** — the runtime will
  refuse a grid larger than `cudaOccupancyMaxActiveBlocksPerMultiprocessor ×
  SM count`;
- so you cannot simply "add a grid barrier" to an existing kernel with a large
  grid. You must first make the grid small enough that the whole thing fits on
  the GPU, and then write the kernel as a grid-stride loop (Module 3) so that a
  small grid still covers all the data.

Module 29 covers this properly. What you need from it now is the reason the
constraint exists, which is the next section.

### There is no cross-block synchronization. None.

**PORTABLE CUDA CONCEPT, and the most important negative result in the
course.** Consider:

```cpp
// DEADLOCK. Do not write this.
if (blockIdx.x == 0) {
    result = compute();
    __threadfence();
    atomicExch(&flag, 1);
} else {
    while (atomicAdd(&flag, 0) == 0) { /* spin */ }
    use(result);
}
```

The fence and the atomic are both correct. The code still deadlocks, and the
reason has nothing to do with memory ordering.

Module 1's launch path: the GigaThread engine places blocks on SMs subject to
resource gating, a block once placed **runs to completion and never migrates**,
and a block that has not been placed **does not exist** — it is an entry in a
work queue, not a running thing. On this GPU, a 256-thread block of a typical
kernel achieves 6 blocks/SM × 40 SMs = **240 co-resident blocks**. Launch 8192
blocks and the other 7952 are queued.

Now the deadlock is obvious. Blocks 0..239 are resident and spinning. Block
5000, which they are waiting for, cannot start, because starting requires a
free block slot, and a slot frees only when a resident block *retires*, and no
resident block will ever retire because they are all spinning. The spin is
consuming the exact resource the producer needs.

This is not a fairness problem that a better scheduler could fix. The block
scheduler is not preemptive at block granularity, by design — that is part of
why launch is cheap and why occupancy math works at all.

Three consequences you must internalise:

1. **Spin-waiting on a global flag between blocks is a deadlock**, unless you
   can prove all participating blocks are co-resident, which for a
   general-purpose kernel you cannot.
2. **The kernel boundary is the grid-wide barrier.** Every block of launch *k*
   has completed, and every memory operation it performed is visible
   device-wide, before any block of launch *k+1* begins. It costs a launch
   (single-digit microseconds) and it is the only grid-wide ordering that holds
   for an arbitrary grid size. Split the kernel.
3. **Cooperative launch exists precisely to remove assumption (1)** by capping
   the grid at the co-residency limit, so that "all blocks are resident" is no
   longer an assumption but an enforced precondition.

A bounded spin — one with an escape counter — does not deadlock but does not
help either: it converts the deadlock into a livelock in which almost every
block wastes its whole timeslice and then gives up. Exercise 2 measures that;
the number of blocks that give up turns out to be `gridDim.x` minus the
co-residency limit, which is a rather direct way of measuring Module 1's model.

---

## Hardware Mental Model

### What `BAR.SYNC` actually is

On Ada, each SM has a small set of hardware **barrier resources** per resident
block (16 of them; `__syncthreads()` always uses barrier 0). The SASS
instruction is `BAR.SYNC.DEFER_BLOCKING 0x0`. The mechanism is:

1. A warp executing `BAR.SYNC` increments the block's arrival counter for that
   barrier and **is marked stalled** by the warp scheduler. It stops being
   eligible. Module 1's scoreboard picks another warp to issue from — this is
   exactly the "free context switch" from Module 1, and it is why a barrier
   costs nearly nothing when the block has other warps with work to do.
2. When the arrival counter reaches the block's expected count, the barrier
   *releases* and all the waiting warps become eligible again.
3. The `DEFER_BLOCKING` variant lets the warp continue issuing independent
   instructions it has already fetched until it actually needs the barrier —
   an instruction-level optimization, not a semantic one.

Two things follow directly:

- **The counter counts warps, not threads.** Increment happens once per warp
  that executes the instruction, regardless of how many lanes are active. This
  is the precise hardware reason a partial-*warp* divergent barrier completes
  and a partial-*block* one may not.
- **Exited warps are subtracted from the expected count.** A warp whose threads
  have all returned is no longer expected. This is the hardware realisation of
  the "non-exited" wording in G1, and the reason that a case-(c) barrier after
  an early return often completes rather than hangs.

The cost of a barrier is therefore not the instruction. It is the **idle time
of the warps that arrive early**, and that idle time is the max over the block
of "how long did each warp take to get here." A block whose warps do wildly
different amounts of work pays that skew at every barrier. Exercise 3 measures
exactly this: replacing one block-wide barrier per round with per-pair flags
and a fence gives a measured **1.33–1.40×** on this GPU, entirely because the
block stops paying `sum over rounds of max over warps` and starts paying
something nearer `max over pairs of sum over rounds`.

### What a fence actually is

`__threadfence_block()` is `MEMBAR.SC.CTA` in SASS; `__threadfence()` is
`MEMBAR.SC.GPU`; `__threadfence_system()` is `MEMBAR.SC.SYS`. A membar does not
touch the warp scheduler at all. It goes to the memory pipeline and says: do
not let any memory operation issued after this point become visible at the
named scope before every memory operation issued before it has. On Ada this
means draining the relevant write path — for CTA scope, the SM's local ordering
point; for GPU scope, out to the L2 (Module 4: L1 is not coherent across SMs,
so device-scope visibility means L2).

This is why the two are not substitutes and why the cost profile is completely
different. A barrier costs *scheduler* time proportional to the block's work
skew. A fence costs *memory pipeline* time proportional to how far the writes
have to be pushed. A block with perfectly balanced warps pays almost nothing
for `__syncthreads()`; a thread with a deep store queue pays real cycles for
`__threadfence()`.

### Why the broken versions often still work

Three separate mechanisms conspire to hide synchronization bugs on this
hardware, and you should be able to name all three:

1. **In-order store issue within a thread.** Ada's load/store unit generally
   retires one thread's stores to one memory space in issue order. So a missing
   `__threadfence_block()` between a payload store and a flag store usually
   does not manifest. Example 2 demonstrates this honestly: both variants print
   zero errors, and the only observable difference is that one of them contains
   `MEMBAR.SC.CTA` in its SASS and the other does not.
2. **Warp-granular barrier arrival.** Divergent barriers whose divergence is
   *within* a warp do not deadlock, because the hardware counts the warp.
3. **Exited-thread removal.** Divergent barriers where the non-arriving threads
   simply leave do not deadlock, because the expected count shrinks.

None of these is a language guarantee. All three can change with a compiler
version, an inlining decision, or a new architecture. The engineering rule is
the one Exercise 1 is built around: **"it passed" and "it is correct" are
different claims, and only one of them is about your program.**

---

## Code Walkthrough

### `example01.cu` — the two guarantees, separated

**Part A** runs one dataflow three ways. 4096 blocks of 256 threads stage a
tile and then rotate it by 96 elements, which crosses warp boundaries:

```cpp
s[t] = in[base + t];
/* nothing, or volatile, or __syncthreads() */
out[base + t] = s[(t + SHIFT) & (TPB - 1)];
```

Thread `t` reads a slot owned by thread `t+96`, which is in a different warp.
Measured output:

```
variant                         wrong elems  guarantees supplied
no barrier                           455018  none
volatile __shared__                  450115  none
__syncthreads()                           0  G1 + G2
```

Out of 1 048 576 elements, 43% are wrong without a barrier — and the
`volatile` version is wrong by essentially the same amount. `volatile` changed
what the compiler may do and nothing at all about what the scheduler may do.
That is the Module 4 debt, paid with a number.

**Part B** is the `__syncthreads_count` convergence loop. The key line is:

```cpp
active = __syncthreads_count(changed);   // barrier + block-wide sum
} while (active > 0);                    // identical in every thread
```

The loop runs 37 sweeps in every block, which is `1 + max(target)` over the
block's 256 cells — the whole block is held by its slowest cell, which is
exactly what a block-uniform loop condition means. There is a second, earlier
`__syncthreads()` inside the body separating the reads from the writes; that
one is a WAR barrier and needs only G1.

**Part C** is the `__syncwarp()` honesty check. A single 32-thread warp rotates
a shared array by one lane, with and without `__syncwarp()`:

```
  no __syncwarp   (UNDEFINED)    wrong elems: 0
  __syncwarp()    (correct)      wrong elems: 0
```

Both correct. The undefined one is still undefined. If you take nothing else
from this module, take the habit of not treating that first line as evidence.

### `example02.cu` — fences, scopes, groups, and the block wall

**Part A** is publish/subscribe inside a block. Thread `t` writes 8 payload
words to global scratch, then raises `flag[t]`; its peer spins on the flag and
reads the payload. The only difference between the two instantiations is
`__threadfence_block()` before the flag store. Both print zero errors — and the
real evidence is in the SASS:

```
# USE_FENCE=false
STG.E [R6.64],      R11
... 8 stores ...
STG.E [R6.64+0x1c], R23
(no membar)

# USE_FENCE=true
STG.E [R6.64],      R5
... 8 stores ...
STG.E [R6.64+0x1c], R23
MEMBAR.SC.CTA                <-- here
```

Reproduce it with `cuobjdump -sass example02.exe | findstr MEMBAR`. The stores
are *issued* in the right order in both; only one of them is *ordered*.

**Part B** substitutes `__threadfence_block()` for `__syncthreads()` in the
Part-A rotate of Example 1:

```
  __threadfence_block() instead of __syncthreads(): wrong elems: 162706
```

A fence makes nobody wait. This is the single crispest demonstration in the
module that the two primitives are not interchangeable.

**Part C** is the cooperative-groups form of the same rotate, using
`block.sync()` and then `cg::tiled_partition<32>(block).sync()` for a
warp-local exchange. Zero errors, same machine code, better-documented intent.

**Part D** prints the block wall:

```
  SMs                                  : 40
  max co-resident blocks per SM        : 6  (at 256 threads/block)
  => at most 240 blocks exist AT ONCE
  cudaDevAttrCooperativeLaunch on this device: 1
```

240. Remember that number; Exercise 2 makes you predict it and then measures
`8192 - 240` blocks giving up.

---

## Check Your Understanding

Answers in `solutions/module09/check_your_understanding.md`.

1. A colleague removes the second `__syncthreads()` from the bottom of a tiled
   matrix-multiply loop body, runs the test suite, and it passes. They argue:
   "the barrier at the top of the next iteration already separates my write
   from your read, so the one at the bottom was redundant." Construct the
   concrete interleaving that breaks their kernel, and say which of G1 and G2
   the missing barrier was supplying. Then describe a change to the *data
   layout* that would make their claim true.

2. You have a kernel in which thread 0 of each block computes a 16-byte
   descriptor into shared memory and all 256 threads then read it. A reviewer
   says "use `__threadfence_block()`, it is cheaper than a barrier." Under what
   circumstance is the reviewer right, under what circumstance are they wrong,
   and what would you have to add to the kernel to make them right? Is the
   result then actually cheaper? Answer in terms of what each primitive does to
   the warp scheduler.

3. Exactly one of the following two fragments is well-defined. Say which,
   explain the rule that decides it, and then explain why *both* of them are
   likely to run to completion and produce plausible-looking output on an
   sm_89 GPU. `blockDim.x` is 256 in both.

   ```cpp
   // (i)
   if (threadIdx.x < 128) { s[threadIdx.x] = f(); __syncthreads(); }
   // (ii)
   if (nIter > 3) { s[threadIdx.x] = f(); __syncthreads(); }   // nIter is a kernel arg
   ```

4. You must sum 2^28 floats. Design A launches one kernel of 240 blocks that
   uses `grid.sync()` between the partial-sum phase and the final-combine
   phase. Design B launches two kernels back to back. Both are correct. Give
   two distinct reasons a production library would ship B, and one specific
   circumstance in which A is genuinely the better choice.

---

## Exercises

All three exercises are in `module09/`. Solutions and notes are in
`solutions/module09/`.

### Exercise 1 — `exercise01.cu` — "Is this barrier necessary?"

Eight fragments, each with a marked synchronization site. For each you supply
three judgements: a verdict (`V_REQUIRED` / `V_UNNECESSARY` / `V_UNDEFINED`),
which guarantee it depends on (`G_EXEC` / `G_MEM` / `G_BOTH` / `G_NA`), and
whether the broken form actually misbehaves *on this GPU*. The third column is
the point of the exercise: it is a question about one chip, and it does not
have the same answer as the first.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
.\exercise01.exe --run-ub      # executes the two undefined fragments; read
                               # the header warning first -- one of them hangs
```

- **TODO 1–3** are the three answer vectors. They are scored item by item
  against a stored digest, so you get right/wrong without being shown the
  answer.
- **TODO 4** is a design problem: rewrite fragment B so that it needs exactly
  one `__syncthreads()` per loop iteration instead of two, without changing
  what it computes, in under 2 KB of shared memory. The harness validates the
  numerics; you verify the barrier count yourself with `cuobjdump -sass`.

Validation: 24/24 on the three vectors plus correct numerics on TODO 4.

### Exercise 2 — `exercise02.cu` — "It hangs. Diagnose it, then design it away."

A kernel where one block produces a scale factor and all the others spin on a
global flag until it appears. With a small grid it is instant and correct; with
a large grid it stalls for about a second and most blocks report giving up.
A second kernel, a tiled shift with a bounds guard, is wrong only on the last
partially-filled block.

**This program spins on purpose.** The spins are bounded by an escape counter
so it always terminates, but the large-grid case runs the GPU at 100% for
roughly a second. The header tells you how to kill it if you remove the bound.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

- **TODO 1** two numeric predictions, committed before running. The second one
  is accepted only within ±5%, because you should be able to derive it exactly.
- **TODO 2** compute, from the runtime API, how many blocks of this kernel can
  be co-resident. No hard-coded constants.
- **TODO 3** fix the tiled shift. The guard is doing two jobs and one of them
  is not allowed.
- **TODO 4** the design TODO: make the scale-and-apply computation correct for
  *any* grid size on *any* device, with no cross-block spinning and no
  cooperative launch. You may add kernels and issue as many launches as you
  like.

Validation: both predictions correct, TODO 2 matching the API, and both kernels
numerically exact.

### Exercise 3 — `exercise03.cu` — fence or barrier?

Paired producer/consumer inside a block, 16 dependent rounds, with a skew that
moves: in every round exactly one thread of the block does 30× the work, and it
is a different thread each round. Two implementations of identical dataflow:
point-to-point flags, and a block-wide barrier per round.

```
nvcc -arch=sm_89 -O3 -std=c++17 -Xcompiler /Zc:preprocessor -o exercise03.exe exercise03.cu
.\exercise03.exe
```

- **TODO 1** and **TODO 2** are stated as *requirements*, not as mechanisms.
  More than one construct makes each correct and they do not cost the same.
- **TODO 3** write the barrier variant. Decide for yourself how many barriers
  per round you actually need.
- **TODO 4** predict which variant is faster and by what ratio, before
  measuring. Accepted within ±30%.

Validation: exact integer equality against a CPU reference for both variants,
plus a correct prediction of the direction and the magnitude of the difference.

---

## Prediction

Commit to these in writing before you compile anything.

1. In Example 1 Part A, the no-barrier and `volatile` variants will each get
   some fraction of the 1 048 576 elements wrong. Predict that fraction to
   within a factor of two, and predict which of the two is worse. Justify the
   ordering from the definition of `volatile`, not from intuition.

2. Exercise 2 launches 8192 blocks of 256 threads, of which the **last** one is
   the producer. Predict how many blocks will report a spin timeout. You need
   two numbers from Module 1 to do it and one of them the program prints for
   you. Write the formula down, not just the answer.

3. Exercise 3 runs the same arithmetic twice, once with per-pair flags and a
   fence and once with one `__syncthreads()` per round. Predict which is
   faster. Then predict what happens to the ratio if the grid is reduced from
   1024 blocks to 40 — one block per SM — and say why. One of those two
   predictions is much harder than the other, and the solution notes report a
   result that is not monotonic.
