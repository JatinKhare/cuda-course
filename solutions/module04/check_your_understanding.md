# Module 04 — Check Your Understanding: answers

---

## 1. `float tmp[8]` fully unrolled, yet `128 bytes stack frame`

The array is 32 bytes; the stack frame is 128. So the array is at most a quarter
of it, and something else is in there too. Two distinct mechanisms produce a
non-zero stack frame, and this kernel has evidence of at least one of them
beyond the array.

**Mechanism A — register spilling (a capacity problem).** The kernel's live
values exceed the register budget, either the architectural 255/thread ceiling
or a lower budget you imposed with `-maxrregcount` or `__launch_bounds__`. ptxas
picks victims and stores them to local memory. `tmp[]` may be among the victims
even though it is perfectly register-allocatable in principle: once ptxas is
short of registers, sixteen scalars that happen to have come from an array are
as spillable as anything else.

**Mechanism B — something else in the kernel is still dynamically indexed (an
addressability problem).** A *different* array, a `switch` lowered to a jump
table, a local `struct` whose address is taken and passed to a `__device__`
function that was not inlined, or a recursive/indirect call needing an ABI stack
frame. The "fully unrolled `tmp[]`" is a red herring; the stack frame belongs to
its neighbour.

**How to tell them apart from compiler output alone:** read the spill counters,
which ptxas reports separately from the frame size.

```
    128 bytes stack frame, 96 bytes spill stores, 96 bytes spill loads
```
→ Mechanism A dominates. 96 of the 128 bytes are spilled registers. Confirm by
looking at `Used NNN registers`: it will be at or near your budget. The fix is to
reduce live range pressure, not to change any index.

```
    128 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```
→ Mechanism B. Nothing was spilled; ptxas deliberately placed 128 bytes of
*addressable* per-thread storage. Register count will be comfortably below the
ceiling. The fix is to find the dynamic index — `cuobjdump -sass` and look at
what address the `LDL`/`STL` instructions compute.

The general rule, worth memorizing: **frame size tells you how much local memory
you have; the spill counters tell you why.**

---

## 2. 700 GB/s on 20 MB, 280 GB/s on 400 MB

The colleague is wrong twice.

**What is happening.** The peak DRAM bandwidth of this GPU is 432 GB/s
(192-bit × 9.001 GHz × 2 for DDR). 700 GB/s is physically impossible from DRAM.
The 20 MB run never reached DRAM: 20 MB fits comfortably inside the 48 MB L2, so
after the warm-up iteration the entire working set was L2-resident and the
kernel was measuring L2 bandwidth, which on this part is several times the DRAM
figure. The 400 MB run is 8× L2, so its hit rate is small and it is a genuine
DRAM measurement.

**The kernel's real DRAM bandwidth is 280 GB/s**, which is 65% of peak — an
ordinary, unremarkable number for a streaming kernel. The kernel scales fine;
the *benchmark* did not measure what it claimed to.

**What you would have to change about the small run.** Nothing about the size —
20 MB will always be L2-resident, and you cannot fix that by running longer.
You have three options, in decreasing order of honesty:

1. **Re-size.** Make the total working set ≥ 4× L2 (192 MB here) and report one
   number. This is the right answer for a bandwidth benchmark.
2. **Relabel.** Keep the 20 MB run but report it as "L2 bandwidth", not "memory
   bandwidth", and stop comparing it to the 432 GB/s peak.
3. **Flush between iterations.** Evict L2 before each timed iteration (touch a
   ≥48 MB scratch buffer, or use `cudaCtxResetPersistingL2Cache` /
   `cudaAccessPolicyWindow` controls). This makes the small run measure cold
   misses, which is a *different* quantity again and usually not what you want.

Note the accounting trap that makes this easy to get wrong: if the kernel reads
one buffer and writes another, the working set is **2× the buffer size**, so the
cliff is at 24 MB per buffer, not 48.

---

## 3. Block A on SM 3 writes; block B on SM 17 reads a stale value

**Why B may see stale data.** A's store is absorbed by SM 3's L1. L1 on this
architecture is a **per-SM, non-coherent** cache: there is no snooping protocol,
no invalidation messages, nothing that tells SM 17 that SM 3 has a dirty line.
Two independent failures can occur:

- A's write may still be sitting dirty in SM 3's L1 and never have reached a
  place SM 17 can see it, and
- B may be reading a copy of the line that SM 17's own L1 cached *before* A
  wrote, and which nothing has invalidated.

**Where they would agree: L2.** L2 is device-wide and is the coherence point on
this GPU — it is the first level both SMs physically share. A write that reaches
L2 and a read that misses L1 and goes to L2 will see each other. (Kernel
boundaries give you this for free: the implicit device-wide synchronization at
the end of a launch flushes dirty L1 lines and invalidates L1, which is why the
"write in kernel 1, read in kernel 2" idiom is always safe.)

**Why `volatile` is not sufficient.** `volatile` is a *compiler* directive. It
stops the compiler from caching the value in a register and from reordering or
eliding the access — it guarantees an actual load/store instruction is emitted
each time. It says nothing about which *hardware* caches that instruction is
allowed to hit in, and nothing about ordering with respect to other accesses.
In practice on modern NVIDIA hardware `volatile` does cause the compiler to emit
loads that bypass L1, so it often appears to work, and that is exactly what
makes it dangerous: it is a coincidence of the current code generator, not a
guarantee in the memory model.

What you actually need is two things, and `volatile` supplies neither:

1. **Scope and caching semantics** — explicit device-scope atomics or
   `ld.global.cg` / `st.global.cg`-class operations (in CUDA C++, relaxed atomics
   or `cuda::atomic_ref` with `thread_scope_device`), which are architecturally
   defined to reach the coherence point.
2. **Ordering** — a fence (`__threadfence()` / `cuda::atomic_thread_fence`) to
   guarantee that A's *data* writes are visible before A's *flag* write is, and a
   corresponding acquire on B's side. Without it, B can observe the flag set and
   the data stale, which is a reordering bug, not a caching bug.

And even with all of that, there is a third problem this question hides: blocks
A and B are not guaranteed to be co-resident. B may be scheduled only after A
has retired, or A may never be scheduled until B releases its SM — spinning on a
flag across blocks can deadlock. Module 9 makes ordering precise; Module 10
covers atomics.

---

## 4. A 48 KB read-only table indexed by a data-dependent permutation

**No. Do not put it in constant memory**, even though it fits.

**Cost model.** The constant cache has a one-address-per-cycle datapath, built
for broadcast. When a warp presents *k* distinct addresses in one access, the
hardware replays the instruction *k* times. For a data-dependent permutation,
the expected number of distinct addresses among 32 lanes indexing a 12 288-entry
table is essentially 32. So:

```
cost(constant, k distinct) ≈ k × cost(constant, 1 distinct) ≈ 32 × broadcast
```

That is the whole warp's load serialized into 32 back-to-back accesses, for
every such instruction, on every warp.

Compare the alternative. In **global memory with `const T* __restrict__`**, the
same 32 scattered 4 B reads are issued as one instruction and the coalescer
splits them into sectors. Worst case they fall in 32 different 32 B sectors and
you pay 32 sector transactions — but those transactions are *pipelined and
overlapped*, not serialized instruction replays, and they are served by an L1
and L2 sized to absorb them. A 48 KB table also sits very comfortably in the
48 MB L2 and largely in the 128 KB L1, so the steady-state hit rate is high.
Measured on the analogous 64-entry case in Exercise 3, the global path was
**24.6× faster** than the constant path for a lane-varying index.

**What to use instead**, in order of preference:

1. **Global memory with `const T* __restrict__`.** Simple, correct, and routes
   through the read-only data cache. This is the default answer.
2. **Shared memory**, if the table is reused enough to justify staging it — but
   48 KB is the entire default per-block shared budget on sm_89, so a copy would
   leave nothing for anything else and would cap you at one block per SM. You
   would need to tile the table. (Module 6.) And shared memory has its own
   replay mechanism for scattered indices — bank conflicts — which Module 7
   shows costs up to 32× for exactly the same architectural reason.
3. **Reorder the work** so the index becomes warp-uniform: sort or bucket the
   threads by `perm[t]` so that each warp touches one region of the table. If you
   can achieve that, constant memory becomes the best option again. This is
   usually the highest-value optimization available and it is not a memory-space
   choice at all.

The generalization: `__constant__` is not "fast read-only storage", it is a
**broadcast** mechanism. The question to ask is never "does it fit in 64 KB?",
it is "how many distinct addresses does one warp present per instruction?"
