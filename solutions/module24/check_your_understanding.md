# Module 24 — Check Your Understanding, answers

---

## Question 1

> A colleague reports that converting their program to four streams made it
> **slower**, and sends a profile showing the GPU busy 100% of the time in both
> versions, with identical total kernel time and identical total memcpy time.
> They conclude streams are useless on this hardware. Give the two structurally
> different explanations that are both consistent with that evidence, and name
> one measurement that distinguishes them.

The evidence rules out a great deal. Identical total kernel time means the
kernels themselves did not change. Identical total memcpy time means the same
bytes moved at the same rate. "GPU busy 100%" — which almost always means
`cuda_gpu_kern_sum` total divided by wall time, or the utilization row in a
timeline — means there was no idle *compute* time to recover. So neither
version is wasting GPU cycles, and yet one is slower.

**Explanation A — there was never any concurrency available, and the stream
machinery is pure overhead.** If the kernel already occupies the whole machine
(Module 1's wave arithmetic; §14 of the lesson measured the staircase: free at
≤40 blocks, gone by 240), splitting the work across four streams changes
nothing about how it executes. What it *adds* is per-chunk host work: three or
four runtime calls per chunk instead of three for the whole buffer, at a few
microseconds each. Chunking a job into 64 pieces costs ~0.5–1 ms of pure issue
overhead on this part, and if the win is zero the overhead is the entire
difference. `example02.cu` part C's right-hand tail (64 chunks → 1.39× where 16
chunks → 1.55×) is this effect visible even *when* there is a win.

**Explanation B — the chunking introduced head-of-line blocking on a
single-copy-engine device.** With `asyncEngineCount == 1` and the obvious
per-chunk issue order `H2D(k), K(k), D2H(k)`, each chunk's result copy reaches
the head of the DMA engine's queue while the kernel it depends on has not
started, blocking the next chunk's *input* copy, which was ready. The engine
idles behind a non-runnable transfer. Measured: **0.87× — slower than serial**,
with the same kernels, the same bytes, and the same total transfer time,
because the transfers all still happened and all still took the same time. They
just happened at the wrong moments.

**The distinguishing measurement.** Total times cannot tell these apart, because
both preserve them. You need a quantity sensitive to *when* things happened:

- the cheapest discriminator is **the serial baseline against the one-chunk
  pipeline**. Set `nChunks = 1` and rerun. Under A the time is unchanged
  (there was nothing to gain, nothing to lose); under B the time *improves*
  back to roughly serial, because with one chunk there is no second transfer to
  block.
- the direct one is the **copy engine's idle fraction**: capture with `nsys` and
  compare the DMA engine's busy time against the wall span of the pipeline
  region. Under A the engine is busy for exactly `H + D` out of a wall time of
  `H + K + D` and that is simply the bound; under B the engine's busy time is
  the same but it is *interleaved with gaps that line up with kernel
  durations*, which is the signature.
- `overlapFactor` from Exercise 2 — sum of operation durations over the wall
  span — separates them too: ≈1.0 under B (nothing concurrent), ≈1.0 under A as
  well but with the pipeline *wall time* unchanged from serial rather than
  worse.

A third possibility worth naming, though the question's evidence argues against
it: the conversion introduced an implicit synchronization (Exercise 2's cause 4
or 5). That would usually show up as the GPU *not* being 100% busy, since a
drained device is an idle device, so the stated evidence makes it less likely —
but "100% busy" measured as a ratio over a *longer* wall time can hide a lot,
and the first thing to check is always the host-side API durations.

---

## Question 2

> A GPU with `asyncEngineCount = 2`. Phase times for the whole buffer: H2D =
> 10 ms, kernel = 4 ms, D2H = 10 ms. Someone reports a measured speedup of
> 2.4×. Is that possible? Derive the bound, then explain what the pipeline
> would have to be doing for the claim to be true without the measurement being
> wrong.

**The bound.** Serial is `10 + 4 + 10 = 24 ms`. With two or more copy engines
the two directions proceed concurrently, so the copy resource is busy for
`max(H, D) = 10 ms`, and the compute resource for `K = 4 ms`. The pipeline
cannot finish before its busiest resource does:

```
speedup <= (H + K + D) / max(max(H, D), K) = 24 / max(10, 4) = 2.4x
```

So **2.4× is exactly the bound** — not impossible, but attainable only if ramp,
drain, issue overhead and every scheduling imperfection sum to zero. Measuring
*exactly* the bound is the result to be suspicious of, not the result to
celebrate.

**What would have to be true.** Three candidate reconciliations, in order of
likelihood:

1. **It is measurement noise against a ceiling the pipeline is genuinely close
   to.** This problem is extremely copy-dominated (`H + D = 20` against `K = 4`),
   so the pipeline is essentially "run both copy engines flat out and compute in
   the cracks". There is very little for the pipeline to get wrong. 2.35–2.40×
   with ±2% run-to-run variation rounds to 2.4. Take the minimum over several
   sweeps and the number will sit a few percent below the bound.
2. **The serial baseline is inflated.** If the "serial" version used pageable
   host memory and the pipeline used pinned, the baseline paid the driver's
   staging `memcpy` and the pipeline did not — the comparison then measures
   pinning, not pipelining, and can exceed the bound computed from the
   pipeline's own phase times. The fix is to measure `H`, `K` and `D` with the
   *same* allocation the pipeline uses, inside the same rotated sweep.
3. **The phases were measured separately from the pipeline, at a different
   operating point.** On a clock-managed part, phases measured cold and a
   pipeline measured warm produce a ratio with no physical meaning. Spec §12.1
   exists for this.

The honest answer to give the colleague: the number is at the bound, so either
the pipeline is perfect or the baseline is wrong, and the second is far more
common. Ask to see the serial baseline's allocation.

A secondary point worth making: with `K = 4` against `H + D = 20`, this problem
barely needs a pipeline. Even on a **one**-engine device the bound would be
`24 / max(20, 4) = 1.2×`. Almost all of the 2.4× comes from having two copy
engines, i.e. from the hardware, not from the code.

---

## Question 3

> Kernels A and B in two non-blocking streams, no event. A writes `buf[0..N)`;
> B reads it. The program prints the right answer on 1000 consecutive runs. A
> colleague argues this proves the driver inserted the dependency, citing that
> CUDA "tracks" buffer usage. Rebut this precisely, and explain why the
> *absence of a hang* is specifically uninformative.

**The rebuttal.** The CUDA runtime does not track data dependencies between
operations in different streams, and there is no mechanism by which it could.
A kernel launch is a pointer-and-argument blob pushed into a command queue; the
driver does not know, and cannot know, which bytes the kernel will touch — the
addresses are computed on the device at run time from `blockIdx`/`threadIdx`,
from values read out of memory, from indirection. The *only* ordering the
runtime provides is the one documented: operations within a stream are ordered,
operations across streams are not. Everything the driver "tracks" is lifetime
and residency (does this allocation still exist, is this host memory pinned),
not read/write sets.

What the colleague is probably remembering is one of three real things, none of
which applies:

- **Unified/managed memory** page migration, where the driver *does* intervene
  on access — but on faults, not on dependencies, and it still does not order
  two concurrent kernels (Module 27).
- **CUDA graphs** (Module 28), where you declare the dependencies explicitly in
  the graph and the runtime then enforces them. The enforcement comes from your
  declaration, not from analysis.
- **The legacy default stream**, which *does* insert device-wide ordering — which
  is exactly why the question specifies `cudaStreamNonBlocking`. Programs that
  "work" under the legacy default stream and break the day someone switches to
  `--default-stream per-thread` are this misconception's most common casualty.

**Why 1000 passing runs prove nothing.** A is launched first, so in practice the
GigaThread engine places A's blocks first, and if A is long relative to the host
issue gap and the machine has no room for both, B's blocks do not even get
placed until A's start retiring. The race is won by the same side every time
under one set of conditions — and lost when the conditions change: a smaller
grid for A (so B fits alongside), a faster GPU, a slower host, a different
driver's placement policy, another process sharing the device, or simply a
different `nChunks`. `example02.cu` part D shows the intermittency directly:
removing the one necessary event leaves **4062 of 4107 sampled elements wrong** —
not 4107, and on a differently loaded machine potentially 0.

**Why the absence of a hang is specifically uninformative.** The failure mode of
a missing cross-stream dependency is *not* a deadlock. A deadlock requires a
cycle in a wait graph; a missing edge **removes** waits, it does not add them.
So the program is guaranteed to terminate — it simply computes with stale data.
This is the opposite of the intuition built up around locks and barriers on the
CPU, where forgetting to acquire something usually shows up eventually as a
hang. Here the only observable is a wrong number, sometimes, and if B reads data
that happens to be initialized to something plausible (zeros, the previous
iteration's values, the same values A was going to write) there may not even be
a wrong number.

The practical consequence: cross-stream dependencies must be established by
reading the code, not by running it. That is also why `compute-sanitizer
--tool racecheck` does not help — it reasons about shared memory within a block,
not about global memory across streams.

---

## Question 4

> A library you do not control calls `cudaDeviceSynchronize()` once per call.
> Your pipeline uses four non-blocking streams and calls it once per chunk.
> `cudaStreamNonBlocking` has already protected you from the legacy default
> stream. Does it protect you from this? Answer for both
> `cudaStreamSynchronize` and `cudaDeviceSynchronize` inside the library, and
> say what the fix is.

**No, and the distinction between the two calls is the whole answer.**

`cudaStreamNonBlocking` is narrow. It says one thing: *this stream is exempt
from the legacy default stream's implicit synchronization.* It is a statement
about stream 0's special semantics and about nothing else. It does not make a
stream invisible, unstoppable, or immune to explicit synchronization.

**If the library calls `cudaStreamSynchronize(itsOwnStream)`** — you are fine.
That call blocks the host until *that* stream drains. Your four streams are
different streams; their work keeps flowing to the device while the library's
host thread waits. You lose only the host time, and if the library's stream is
short you lose almost nothing. This is the well-behaved case and is what library
code should do.

**If the library calls `cudaDeviceSynchronize()`** — you are not fine, and
`cudaStreamNonBlocking` does not help at all. `cudaDeviceSynchronize` waits for
**all** work in the context, in every stream, non-blocking or not. Called once
per chunk it drains your pipeline sixteen times. Your pipeline is now a serial
program with extra steps, and the symptom is exactly Exercise 2's: correct
answer, no error, measurably slower than not using streams.

(The same reasoning applies to `cudaMalloc`, `cudaFree` and friends inside the
library — Exercise 2's cause 4. Implicit synchronization in the allocator is
device-wide and no stream flag exempts you from it. This is why "the library
allocates a workspace on every call" is a performance bug in a pipelined
program even though it is invisible in a serial one.)

**The fixes, in order of preference.**

1. **Hoist the library call out of the steady-state loop.** Call it once before
   the pipeline and once after, not once per chunk. Usually possible and
   usually the right answer: a per-chunk call to a library that synchronizes is
   a sign the chunk boundary is in the wrong place.
2. **Use a stream-aware entry point if one exists.** Most CUDA libraries
   (cuBLAS, cuFFT, cuDNN, Thrust via its execution policies) take a stream and
   do not synchronize; the synchronizing call is usually a convenience wrapper.
   `cublasSetStream(handle, s)` and then issuing into `s` removes the problem
   entirely.
3. **Move the library onto its own CUDA context, or its own process.** Implicit
   and explicit synchronization are *context*-scoped, not device-scoped. This is
   heavy — a context switch on the device is expensive and the two contexts
   time-slice rather than run concurrently unless MPS is in use — but it is the
   only option when the library is a binary blob with no stream parameter.
4. **Overlap at a coarser grain.** If the library call must stay inside the
   loop, accept that each iteration is a serialization point and make the
   iterations large enough that the drain is amortized. This is giving up, but
   it is giving up with a number attached.

What does *not* work: creating more streams, raising stream priority, switching
to `--default-stream per-thread`, or wrapping the library call in your own
stream. None of those changes what `cudaDeviceSynchronize` waits for.
