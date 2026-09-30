# Module 02 / Exercise 02 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

Diagnosing the original:

```
nvcc -arch=sm_89 -O3 -lineinfo -o exercise02.exe exercise02.cu
.\exercise02.exe
compute-sanitizer --tool memcheck --show-backtrace no .\exercise02.exe
```

## The bug class

Two independent defects, and they compound. That is the point of the exercise:
the second one is what makes the first one hard to find.

**Defect A (root cause) — a unit confusion at an API boundary.** The wrapper

```cpp
static void launch_saxpy_clamp(const float* d_in, float* d_out, size_t size, …)
{
    const int blocks = (int)((size + threads - 1) / threads);
    saxpy_clamp<<<blocks, threads>>>(d_in, d_out, (int)size, …);
}
```

takes a parameter called `size` and is called as

```cpp
launch_saxpy_clamp(d_in, d_out, bytes, a, b, lo, hi);    // bytes, not elements
```

`bytes` is `n * sizeof(float)` = 4,000,012. So the grid is sized for 4,000,012
"elements" — 15,626 blocks instead of 3,907 — and the kernel's bound `n` is
also 4,000,012, so the `if (i < n)` guard lets every one of those threads
through. Threads with `i >= 1,000,003` read `in[i]` and write `out[i]` up to
~12 MB past the end of two 4,000,012-byte allocations. The GPU MMU faults.

Note what *does not* catch this: the guard is present and correct, the
allocations are correctly sized, the memcpys are correctly sized, and the
kernel is correct. The only wrong thing in the program is which variable is
passed at one call site. `size_t` versus `int` did not help, because both
counts are integers. Units are not in the type system unless you put them
there — which is why the fix renames the parameter as well as changing the
argument.

**Defect B (why it is misreported) — a launch checked without a sync.** The
wrapper called only

```cpp
CHECK(cudaGetLastError());
```

That check asks "did the driver reject the launch?" The launch configuration
was perfectly legal (15,626 × 256 is within every limit), so the answer is
`cudaSuccess`. At that instant the kernel may not have executed a single
instruction. The fault happens microseconds later, in hardware, and is reported
to the host by the next call that waits for the device — here the D2H
`cudaMemcpy`.

And because `cudaErrorIllegalAddress` is **sticky**, the context is destroyed,
so both `cudaFree` calls report the same error too. Three error messages, none
of them at the faulting line, all of them identical.

## TODO 1 — make the failure point at the right line

```cpp
saxpy_clamp<<<blocks, threads>>>(d_in, d_out, nElems, a, b, lo, hi);

CHECK(cudaGetLastError());        // launch-configuration errors
CHECK(cudaDeviceSynchronize());   // errors raised during execution
```

Both calls, in that order, immediately after the launch. Why both:

| Check | Detects | Cannot detect |
|---|---|---|
| `cudaGetLastError()` | too many threads/block, zero-sized grid, too much dynamic shared memory, insufficient registers | anything the kernel does while running — the kernel has not run yet |
| `cudaDeviceSynchronize()` | illegal address, misaligned address, device `assert`, launch timeout | nothing that (1) catches, because the launch never happened |

With the sync in place and the root cause still present, the reported line moves
from the memcpy to the launch site — which is the whole diagnostic value. It is
also why `CHECK_KERNEL()` exists as a single macro: it is too easy to write one
of the two and think you are covered.

Cost: `cudaDeviceSynchronize()` after every launch serializes CPU and GPU. Keep
it in development builds, compile it out for release:

```cpp
#ifdef NDEBUG
#  define CHECK_KERNEL() CHECK(cudaGetLastError())
#else
#  define CHECK_KERNEL() do { CHECK(cudaGetLastError());                      \
                              CHECK(cudaDeviceSynchronize()); } while (0)
#endif
```

**Wrong answers.** Replacing `cudaGetLastError()` with `cudaDeviceSynchronize()`
alone: a bad launch config is stored in the last-error slot and *not* reported
by the sync, so `<<<1,2000>>>` would silently do nothing. Calling
`cudaPeekAtLastError()` instead of `cudaGetLastError()`: works for detection,
but leaves the error in the slot so the *next* launch's check reports a stale
error. Adding a `cudaGetLastError()` before the launch "to clear the slot":
plausible-looking and harmful — see question 3 of Check Your Understanding.

## TODO 2 — fix the root cause

```cpp
static void launch_saxpy_clamp(const float* d_in, float* d_out, int nElems, …)
{
    const int threads = 256;
    const int blocks  = (nElems + threads - 1) / threads;
    saxpy_clamp<<<blocks, threads>>>(d_in, d_out, nElems, …);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}
…
launch_saxpy_clamp(d_in, d_out, n, a, b, lo, hi);     // n, not bytes
```

Changing only the call site (`bytes` → `n`) also fixes it. Changing the
parameter name and type as well is the better answer, because it makes the
wrong call impossible to write next time.

The plausible-but-wrong alternative fix is to enlarge the allocations so the
overrun lands inside them. That makes the symptom go away and leaves you
computing 4 million elements when you have 1 million of data — silently wrong
output, no error at all. Whenever a fix to a memory fault is "allocate more",
stop and re-derive the loop bound.

## How compute-sanitizer finds it

`compute-sanitizer --tool memcheck` instruments every global load and store and
validates the address against the live allocation table. Build with `-lineinfo`
so it can map the faulting instruction back to a source line. Real output from
the broken program on this machine (trimmed; it printed 7,587 errors):

```
========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 4 bytes
=========     at saxpy_clamp(const float *, float *, int, float, float, float, float)+0x90 in exercise02.cu:72
=========     by thread (0,0,0) in block (3928,0,0)
=========     Access to 0x13097d6000 is out of bounds
=========     and is 22,261 bytes after the nearest allocation at 0x1309400000 of size 4,000,012 bytes
=========
========= Invalid __global__ read of size 4 bytes
=========     at saxpy_clamp(const float *, float *, int, float, float, float, float)+0x90 in exercise02.cu:72
=========     by thread (1,0,0) in block (3928,0,0)
=========     Access to 0x13097d6004 is out of bounds
=========     and is 22,265 bytes after the nearest allocation at 0x1309400000 of size 4,000,012 bytes
=========
… 7,487 more …
CUDA error cudaErrorUnknown at exercise02.cu:119 -> unknown error
CUDA error cudaErrorUnknown at exercise02.cu:133 -> unknown error
CUDA error cudaErrorUnknown at exercise02.cu:134 -> unknown error
FAIL (999759 mismatching elements of 1000003, 1 CUDA errors reported)
========= Target application returned an error
========= ERROR SUMMARY: 7587 errors
```

Every number in there is a clue:

- **`in exercise02.cu:72`** — the faulting instruction is the `in[i]` load in
  the kernel, not the memcpy the program blamed.
- **`block (3928,0,0)`** — the first faulting block. 3907 blocks would have
  sufficed; block 3907 is the first one entirely past the data, and by block
  3928 the addresses have walked past the allocation's padding. So the grid is
  too big — go look at how `blocks` was computed.
- **`22,261 bytes after the nearest allocation … of size 4,000,012 bytes`** —
  4,000,012 is `1000003 * 4`, confirming the allocation is the intended size.
  The *access* is out of range, not the buffer.
- Note that under the sanitizer the host-side errors change from
  `cudaErrorIllegalAddress` to `cudaErrorUnknown`, because the sanitizer
  intercepts the fault first. Do not let that distract you; the sanitizer's own
  report is the authoritative one.

After the fix:

```
========= COMPUTE-SANITIZER
========= CUDA API Warning: Resetting device while there are still other users claiming to use it
=========
PASS (0 mismatching elements of 1000003, 0 CUDA errors reported)
========= ERROR SUMMARY: 1 error
```

The remaining "error" is a benign API warning raised by `cudaDeviceReset()`
while the sanitizer itself holds a reference to the device. It is not a memory
error and it does not occur outside the sanitizer.

## Synchronization / memory reasoning

There is no intra-kernel synchronization here at all — every thread is
independent. The only ordering that matters is host-to-device: the kernel must
finish before the D2H copy reads `d_out`. A blocking `cudaMemcpy` on the
default stream provides that ordering by itself, which is precisely why the
program still produced *a* result (a wrong one) and why the missing
`cudaDeviceSynchronize()` was invisible until something faulted.

## Performance reasoning

The broken version launched 15,626 blocks instead of 3,907 and read/wrote four
times the necessary bytes — so even if the allocations had been large enough to
absorb it, it would have been 4× slower on a kernel that is purely
bandwidth-bound (8 MB of traffic for 1 M elements, two FLOPs each). "It got
slower and the answers got weird" is a symptom of the same class of bug.

## Expected output

Broken version, actual:

```
CUDA error cudaErrorIllegalAddress at exercise02.cu:119 -> an illegal memory access was encountered
CUDA error cudaErrorIllegalAddress at exercise02.cu:133 -> an illegal memory access was encountered
CUDA error cudaErrorIllegalAddress at exercise02.cu:134 -> an illegal memory access was encountered
FAIL (999759 mismatching elements of 1000003, 1 CUDA errors reported)
```

Line 119 is the D2H `cudaMemcpy`; 133 and 134 are the two `cudaFree` calls. The
`1 CUDA errors reported` in the PASS/FAIL line counts only the errors seen
*before* that line printed. `cudaDeviceReset()` returns success because tearing
down an already-dead context is exactly its job.

Fixed version, actual:

```
PASS (0 mismatching elements of 1000003, 0 CUDA errors reported)
```

## The result that matters

**The line a CUDA error is reported on is the line that happened to
synchronize, not the line that is wrong.** Because a launch is asynchronous and
because an illegal access is sticky, one bad index in a kernel produces a
cascade of identical errors attributed to `cudaMemcpy`, `cudaFree`, and
anything else that follows — none of which are guilty. The only defence is
structural: check both the launch-configuration error and a synchronizing error
at every launch during development, so that the first report is co-located with
the cause, and reach for `compute-sanitizer` the moment you see
`cudaErrorIllegalAddress`.

Variation to try: delete `CHECK(cudaGetLastError())` from the fixed wrapper,
keep the sync, and change the launch to `<<<blocks, 2000>>>` (2000 > 1024).
Verified result: `FAIL (999759 mismatching elements of 1000003, 0 CUDA errors
reported)`. The kernel never ran, `cudaDeviceSynchronize()` returned success
because there was nothing to wait for, `d_out` still holds whatever was in that
page, and you get a silent wrong answer with no error message at all — the
mirror image of the bug you just fixed.
