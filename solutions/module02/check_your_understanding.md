# Module 02 — Check Your Understanding (answers)

---

## 1. "My kernel runs in 4 microseconds" for a 400 MB input

**The bound.** A kernel that must read 400 MB cannot finish faster than

```
400e6 B / 432e9 B/s = 9.26e-4 s = 926 µs
```

and that assumes perfect coalescing, 100% of theoretical peak bandwidth, and
that nothing is written. The true floor is higher: at a realistic 80–85% of
peak for a streaming kernel, ~1.1 ms; if it also writes 400 MB, ~2.2 ms. So the
claimed 4 µs is *two to three orders of magnitude* below a hard physical limit.
No amount of cleverness inside the kernel gets you there, because DRAM cannot
deliver the bytes faster.

**What the 4 µs actually measured.** The host-side cost of the
`kernel<<<…>>>(…)` statement: argument validation, marshalling the parameters
into a launch packet, and appending that packet to a command buffer. The
measurement is real and reproducible — it is just not a measurement of the
kernel. `example01.cu` measures the same quantity deliberately and gets 5.04 µs.

The fix is CUDA events, which are timestamps recorded *in the stream* alongside
the work:

```cpp
cudaEventRecord(e0);  kernel<<<…>>>(…);  cudaEventRecord(e1);
cudaEventSynchronize(e1);  cudaEventElapsedTime(&ms, e0, e1);
```

plus a warm-up launch and ≥ 20 iterations.

**Why adding a sync to the *previous* iteration changes the number.** Because
the cost of a launch depends on the state of the command queue when it is
issued. If the previous iteration ended with `cudaDeviceSynchronize()`, the
queue is drained and the driver must build and submit a fresh command buffer to
the OS — on Windows/WDDM that is the expensive path. If the previous iteration
did not sync, the GPU is still working and the new launch is appended to a
buffer already in flight — a cheap path. `example01.cu` measures both on this
machine: **17.70 µs** after an idle queue versus **5.04 µs** into a busy one.
Same code, same kernel, 3.5× difference, entirely due to host-side driver
behaviour. **ARCHITECTURE-SPECIFIC**: the magnitude is a WDDM property. The
lesson is portable: a host-side timing near an async launch measures the driver,
and the driver's cost is context-dependent, so the number is not even a stable
wrong answer.

---

## 2. Kernel A, kernel B, one sync, `cudaErrorIllegalAddress`

**Which kernel faulted? You cannot tell.** Both launches went into the same
stream; the single `cudaDeviceSynchronize()` waits for both and reports the
first error the context recorded. The `cudaGetLastError()` after B's launch
returned `cudaSuccess`, which tells you only that *B's launch configuration* was
accepted — it says nothing about whether A had already faulted, because a
launch-config check does not wait for anything. (It is also worth noting that
had A faulted *before* B was launched, B's launch would have failed too, since
the context is dead — but with stream-ordered execution and a host that ran
ahead, B is typically enqueued long before A starts executing.)

**The smallest change that tells you:** a synchronizing check after *each*
launch.

```cpp
A<<<…>>>(…);  CHECK(cudaGetLastError());  CHECK(cudaDeviceSynchronize());
B<<<…>>>(…);  CHECK(cudaGetLastError());  CHECK(cudaDeviceSynchronize());
```

i.e. `CHECK_KERNEL()` after both. Now the error is reported at the launch that
caused it.

**What it costs:** the sync after A serializes CPU and GPU and prevents B from
being enqueued while A runs, so you lose all launch-overhead overlap and
possibly all kernel overlap. In a hot loop this is a large, real slowdown.
Hence: `CHECK_KERNEL()` is a development tool; in release builds you keep the
`cudaGetLastError()` and compile out the sync.

**A case where even that leaves you unsure:** yes, more than one.

- If A's fault is *asynchronous in the hardware sense* — e.g. a store that is
  still in flight in the memory pipeline when A's blocks retire — the fault can
  be surfaced at the next synchronization rather than A's. In practice the sync
  after A does capture it, but the general guarantee is "by the next
  synchronizing call", not "at the exact instruction".
- If the fault is a **race** rather than a fixed bad index, inserting the sync
  changes the timing and the fault may stop reproducing (a Heisenbug). Module 10
  deals with races; `compute-sanitizer --tool racecheck` is the tool.
- If A corrupted memory *without* going out of bounds — writing the wrong valid
  address — there is no fault at all, and B may be the one that reads the
  garbage and computes a wrong answer with zero errors reported. Error checking
  cannot find that; validation against a reference can.

The reliable move once you see `cudaErrorIllegalAddress` is not to bisect with
syncs, it is `compute-sanitizer --tool memcheck` with `-lineinfo`, which names
the kernel, the source line, the block, and the thread.

---

## 3. "Defensive" `cudaGetLastError()` everywhere

The scheme: call `cudaGetLastError()` at the top of `main`, after every
`cudaMalloc`, before every launch, and after every launch; log only non-success.

**Scenario where it hides a real bug.**
`cudaGetLastError()` **clears** the last-error slot. Consider:

```cpp
kernelA<<<1, 2000>>>(…);            // 2000 > 1024: rejected, slot := InvalidValue
… some code …
cudaGetLastError();                 // "defensive" pre-launch clear: logs nothing?
kernelB<<<blocks, 256>>>(…);
if (cudaGetLastError() != cudaSuccess) log(…);   // clean
```

If the pre-launch "defensive" call is the one that reads A's error, and it is
placed somewhere the author considers routine hygiene rather than a real check
(or it logs into a stream nobody reads), A's failure is consumed and discarded.
`kernelB`'s check then reports success and the program proceeds with A's output
never having been computed. The result is a **silent wrong answer with no error
anywhere** — verified in Exercise 2's variation: deleting the launch-config
check and launching `<<<blocks, 2000>>>` yields
`FAIL (999759 mismatching elements of 1000003, 0 CUDA errors reported)`.

The two-check idiom does not have this hole because there is exactly one
`cudaGetLastError()` per launch, immediately after it, and its result is acted
on.

A second, more insidious variant: the scheme never syncs. So it catches launch
configuration errors only, and every illegal-address bug in the program is
reported against whichever `cudaMemcpy` happens to come next. That is exactly
the symptom in Exercise 2.

**Scenario where it invents a bug that does not exist.**
Some CUDA runtime calls legitimately fail as part of normal probing, and leave
an error in the slot even though the program handled it. The canonical case:

```cpp
cudaError_t e = cudaMalloc(&p, huge);       // returns cudaErrorMemoryAllocation
if (e != cudaSuccess) { huge /= 2; e = cudaMalloc(&p, huge); }   // handled, fine
…
kernel<<<g,b>>>(p, …);
CHECK(cudaGetLastError());     // may report the *earlier* allocation failure
```

The same happens with `cudaSetDevice` probing devices that are in exclusive
mode, `cudaDeviceGetAttribute` on an unsupported attribute, or occupancy/driver
queries that are expected to fail. A stale non-sticky error sits in the slot,
and the next "defensive" check attributes it to a launch that was perfectly
fine. You then spend an afternoon debugging a healthy kernel.

Both failure modes have the same root: **the last-error slot is a single
mutable global, and a check is only meaningful if it is paired with the specific
operation that could have set it.** Read it once, immediately after the thing
you are checking, and act on the value. If you need to inspect without
consuming, use `cudaPeekAtLastError()`.

---

## 4. Why `__device__` may return values and `__global__` may not

**Who invokes it, and when does the invocation complete?**

A `__device__` function is called **by a thread that is already running on the
GPU**, from within device code, through an ordinary (usually inlined) call. The
call is synchronous in the only sense that matters: the calling thread's program
counter does not advance past the call until the callee has returned, and the
return value is delivered in a register belonging to that same thread. Because
it is an ordinary function in an ordinary calling convention, it can do
everything an ordinary function can: return a value, take a reference, be
recursive (given enough stack — device stack is finite and per-thread, and
recursion costs local memory), be overloaded, be a template.

A `__global__` function is not called. It is **launched**: the host enqueues a
request, and the launch statement returns — typically microseconds later, and
*before the function has executed a single instruction*. There is therefore no
point in the host's program at which a return value could be delivered. The
launch statement has already completed and the host has moved on. To return a
value you would need the host to block until the kernel finished, which would
make every launch synchronous and destroy the queueing model the whole
architecture is built on (see §"Hardware Mental Model" in the lesson).

There is a second, independent reason: a kernel is instantiated once per thread
across the grid. `<<<3907,256>>>` creates 1,000,192 instances of the body.
"The" return value of that is not a well-defined object. Even if the launch were
synchronous, the language would have to invent a rule for which thread's value
wins.

So kernels communicate through memory: the host allocates a device buffer,
passes the pointer in the parameter buffer, the kernel writes to it, and the
host reads it back after an operation that orders the two. `void` is not a
restriction the language imposes for tidiness — it is the only type consistent
with asynchronous, many-instance execution.

(Corollary worth internalizing: the same argument explains why kernel arguments
are passed **by value** into a small parameter buffer. A reference or a host
pointer would name storage the GPU cannot reach, and the launch returns before
the kernel runs, so the host's stack frame holding that storage may be gone by
the time the kernel executes.)
