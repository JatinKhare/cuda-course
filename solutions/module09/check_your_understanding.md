# Module 9 — Check Your Understanding: answers

---

## 1. The "redundant" WAR barrier in a tiled loop

**The interleaving.** The loop body is

```cpp
for (tile...) {
    s[t] = g[tile * TPB + t];      // W
    __syncthreads();               // (1) RAW barrier — kept
    acc += f(s[peer(t)]);          // R
    __syncthreads();               // (2) WAR barrier — deleted by the colleague
}
```

With (2) removed, nothing prevents this order of events. Warp 0 is released by
barrier (1) of iteration *k*, executes its read `R`, loops, and executes its
write `W` of iteration *k+1*, all before warp 7 — released by the *same*
barrier (1) — has executed its read `R` of iteration *k*. Warp 7 then reads
`s[peer]` and gets iteration *k+1* data.

Note carefully that barrier (1) of iteration *k+1* does **not** save them.
Warp 0 reaches that barrier only *after* its `W`; the whole point of a barrier
is that it orders things around itself, and warp 7's stale read happened before
warp 0 got there. A barrier separates "before" from "after"; it cannot reach
backwards and un-corrupt a slot that was clobbered on the other side of it.

**Which guarantee.** G1, the execution barrier, alone. The hazard is
write-after-read. The thing warp 0 must wait for is warp 7 *finishing a read*.
Reads publish nothing, so there is nothing for G2 to make visible. This is one
of the few places in CUDA where you can point at a barrier and say "only half
of this is doing work" — and it is exactly the observation that lets you delete
it.

**The layout change that makes the claim true: double-buffering.** Declare
`__shared__ float s[2][TPB]` and stage iteration `tile` into `s[tile & 1]`.
Now iteration *k+1*'s write lands in the buffer that nobody is reading, because
the readers of iteration *k* are reading `s[k & 1]`. The WAR hazard does not
exist, and the colleague's kernel is correct with one barrier per iteration.
Iteration *k+2* reuses buffer `k & 1`, and by then every thread has passed
barrier (1) of iteration *k+1*, which is after all of iteration *k*'s reads —
so two buffers is enough, not three.

The cost is `TPB * sizeof(float)` extra bytes of shared memory per block. On
this GPU with 256 threads and floats that is 1 KB out of a 48 KB default
budget, against saving one block-wide rendezvous per tile. That trade is
essentially always worth taking, and it is the germ of the software pipelining
in Module 17. This is TODO 4 of Exercise 1; the measured SASS barrier count
drops from 8 to 4 for a four-tile loop.

---

## 2. `__threadfence_block()` instead of a barrier for a shared descriptor

**As written, the reviewer is wrong.** Thread 0 writes the descriptor; threads
1–255 read it. `__threadfence_block()` executed by thread 0 makes thread 0's
writes *visible* once somebody looks, but it does not make threads 1–255 wait
for thread 0 to have run. Warp 3 may be scheduled first, read 16 bytes of
uninitialised shared memory, and proceed. The missing guarantee is G1, and a
fence does not supply G1 at any scope. There is no ordering here at all without
something that blocks.

**What you would have to add to make them right.** You have to supply the
waiting some other way, which in practice means a flag and a spin:

```cpp
if (threadIdx.x == 0) {
    desc[0] = ...; desc[1] = ...; desc[2] = ...; desc[3] = ...;
    __threadfence_block();                                  // publish payload
    flag_ref(ready).store(1, cuda::memory_order_relaxed);   // advertise
}
// every thread, including 0:
while (flag_ref(ready).load(cuda::memory_order_acquire) == 0) { }
use(desc);
```

That is correct, and it is legal specifically because all 256 threads are
co-resident on one SM (Module 1) and independent thread scheduling guarantees
forward progress for diverged threads (Module 8).

**Is it cheaper? No — in this shape it is strictly worse.** Reason about the
warp scheduler:

- `__syncthreads()` makes an arriving warp **stalled**: it leaves the eligible
  set, the scheduler issues from other warps, and it costs the SM nothing while
  it waits. This is Module 1's free context switch. Its only real cost is the
  idle time of warps that arrive early, i.e. the block's work skew at that
  point.
- The spin loop makes the waiting warp **eligible and issuing**. Every
  iteration is an instruction issued by a warp scheduler that could have issued
  something useful. Seven warps busily loading the same shared word is seven
  warps of issue bandwidth burnt, plus contention on that word.

Here the skew is nearly the whole block waiting for one thread either way, so
the barrier's "cost" is the same waiting time — but spent stalled instead of
spinning. The barrier wins.

**When does the reviewer's instinct pay off?** When the waiting is
*point-to-point* rather than all-to-one, so that most warps never wait at all.
That is Exercise 3: each consumer waits on exactly one peer, most warps never
block on the block's slowest thread, and the flag+fence design measures
1.33–1.40× faster than one barrier per round. The rule is not "fences are
cheaper than barriers." It is: **a barrier costs you the block's worst case;
point-to-point synchronization costs you your own dependency's worst case. Use
the barrier unless the block's worst case is much worse than yours.**

---

## 3. Which fragment is well-defined

**(ii) is well-defined. (i) is undefined.**

The rule is the one in the Programming Guide: `__syncthreads()` in conditional
code is allowed only if the condition evaluates **identically across the entire
thread block**.

- In (i) the condition is `threadIdx.x < 128`. Half the block evaluates it
  true and half false. The barrier is reached by 128 of 256 threads. Undefined.
- In (ii) the condition is `nIter > 3`, where `nIter` is a kernel argument —
  the same value in every thread of every block. Either all 256 threads reach
  the barrier or none do. Well-defined. (It would also be well-defined for a
  condition on `blockIdx`, which is uniform within a block though not across
  the grid.)

**Why both run to completion and look plausible on sm_89.** Three separate
mechanisms:

1. **The barrier counts warps, not threads.** `BAR.SYNC` increments the
   arrival counter once per warp that executes it, regardless of the active
   mask. In (i), warps 0–3 (threads 0–127) execute the barrier and warps 4–7 do
   not — so the counter reaches 4 and the block expects 8. That *would* hang,
   except for:
2. **Exited threads are removed from the expected count.** Warps 4–7 in (i)
   fall straight through to the end of the kernel and exit. Once they exit they
   are no longer expected at the barrier, the expected count drops to 4, and
   the barrier releases. The kernel returns normally with
   `cudaGetLastError() == cudaSuccess`.
3. **The corruption is data-dependent and partial.** Only the threads whose
   reads happened to race lose. In the measured version of this shape
   (Exercise 1 fragment D, 1024 blocks × 256 threads) 64 000 of 262 144
   elements were wrong — 24%. Three quarters of the output is right, and no
   error is reported anywhere.

The engineering point: a divergent barrier on this hardware is far more likely
to produce *silently wrong answers* than a hang. "It didn't hang" is not
evidence of anything. `compute-sanitizer --tool synccheck` did not flag any of
these shapes either (see the Exercise 1 solution notes); the tool that found
them was `--tool racecheck`, and for the uninitialised-shared-slot variant,
`--tool initcheck --initcheck-address-space shared`.

---

## 4. `grid.sync()` versus two kernel launches for a 2^28-element sum

**Two reasons a library ships B (two launches):**

1. **Portability and grid-size freedom.** `grid.sync()` requires
   `cudaLaunchCooperativeKernel` and requires *every* block to be co-resident.
   The runtime enforces this: the grid may not exceed
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor × multiProcessorCount`, which
   on this GPU is 240 blocks at 256 threads. That number is a property of the
   *kernel's* register and shared-memory usage on the *specific* GPU. A library
   kernel whose maximum legal grid changes when a user bumps an unroll factor,
   or moves to a different card, or is run under MPS or in a
   green-context/partitioned environment, is a support burden. Design B has no
   such constraint and its grid can be tuned freely for the machine.
   `cudaDevAttrCooperativeLaunch` is also not guaranteed non-zero on every
   supported platform.

2. **Occupancy and register pressure.** A single fused kernel must carry the
   register and shared-memory footprint of *both* phases for its entire
   lifetime, because a block's resource allocation is fixed at placement
   (Module 1) and is never resized. The partial-sum phase and the combine phase
   have different footprints; fusing them means both run at the occupancy of
   the worse one. Two kernels are each compiled and scheduled with their own
   footprint. On a reduction, the second kernel is also trivially small — the
   launch overhead (single-digit microseconds) is dwarfed by the ~600 µs that
   2^28 floats take to stream at 432 GB/s.

   A third reason, if you want one: with two launches the intermediate array is
   an ordinary buffer the caller can inspect, reuse across calls, or feed into
   a different second stage. Fusion hides it.

**When A is genuinely better: many small, dependent iterations.** If the
structure is not "one reduce" but "iterate this reduce-then-broadcast step 500
times to convergence", design B pays 1000 kernel launches. At ~5 µs each that
is 5 ms of pure overhead, which for a small working set can exceed the
arithmetic entirely. Worse, every launch boundary flushes L1 and forces the
data back through L2, so a working set that would have stayed resident in the
SMs' L1 across iterations is re-fetched every time. A cooperative kernel with
`grid.sync()` keeps the blocks — and their registers, their shared memory, and
their L1 lines — alive across all 500 iterations. Iterative solvers, BFS
frontier expansion, and some graph analytics are the standard examples, and it
is exactly the case Module 29 builds towards.
