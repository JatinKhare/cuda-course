# Module 02 — First CUDA Program

> Prerequisites: Module 1 (SMs, warps, blocks, grids, the GigaThread engine)
> What this module gives you: the ability to write, launch, and — crucially — *correctly diagnose* a CUDA program, starting from the fact that a kernel launch does not run the kernel.

---

## Concept

### 1. A `.cu` file is two programs in one file

A `.cu` file is C++ with a small syntactic extension. It contains code for two
different instruction set architectures:

- **host code** — compiled to x86-64, linked into your `.exe`, runs on the CPU;
- **device code** — compiled to PTX and then to SASS (the actual Ada
  instruction set), embedded in the same `.exe` as a *fatbinary*, uploaded to
  the GPU by the driver at first use.

`nvcc` is a **compiler driver**, not a compiler. Given `foo.cu` it:

1. runs the CUDA front end (`cudafe++`) to split the translation unit into a
   device part and a host part;
2. compiles the device part to PTX (a virtual ISA) with `cicc`, then to SASS
   for each requested architecture with `ptxas`;
3. packs the PTX and SASS into a fatbinary with `fatbinary`;
4. rewrites the host part — in particular every `<<< >>>` becomes ordinary
   function calls into the CUDA runtime — and hands it to the **host compiler**
   (MSVC `cl.exe` on this machine, gcc/clang elsewhere);
5. links everything.

`-arch=sm_89` tells step 2 which SASS to generate. That is why the build line
for this course is `nvcc -arch=sm_89 …`: without it you get whatever default
architecture the toolkit picks, and possibly a JIT compile at run time.

**Module 37 covers the compilation pipeline in detail** — PTX, JIT, fatbinary
layout, separate compilation, `-rdc`. For now, the only thing you must believe
is: *one source file, two target machines, and a function must be labelled for
which machine it belongs to.*

### 2. Execution-space qualifiers

| Qualifier | Compiled for | Callable from | Notes |
|---|---|---|---|
| `__host__` | CPU | host code | The default. Writing it is optional. |
| `__device__` | GPU | device code | Not visible to host code at all. |
| `__global__` | GPU | **host** (and, with CUDA Dynamic Parallelism, device) | A *kernel*. Launched with `<<< >>>`. **Must return `void`.** |
| `__host__ __device__` | CPU **and** GPU | both | nvcc compiles the body twice and emits two object codes. |

The distinction people get wrong is `__global__` vs `__device__`.
`__global__` is the *entry point*: the host asks the GPU to run it, and the
hardware instantiates it once per thread across a grid. `__device__` is an
ordinary function that already-running device threads call — it is per-thread,
usually inlined, and there is no grid associated with it.

**Why must `__global__` return `void`?** Because the launch statement completes
before the kernel does (see §3), so there is no moment at which a return value
could be handed back to the caller. There is also no single value to return:
the kernel runs in thousands of thread instances, each of which would have its
own. Kernels communicate results by writing to memory, and the host reads that
memory afterwards. Trying to `return` a value from a `__global__` function is a
compile error, not a runtime surprise.

`__host__ __device__` is how you avoid maintaining two copies of the same
formula. Both worked examples use it so that the CPU reference and the GPU
kernel are compiled from *the same source lines* — if the two disagree
numerically, you know the difference is floating-point, not a transcription
error.

A caveat that bites later: inside a `__host__ __device__` function you may not
call host-only library functions (`std::vector`, `malloc`, `<iostream>`) —
the device compilation of the body must be valid device code too.

### 3. Launch syntax, and the four configuration parameters

```cpp
kernel<<<grid, block, sharedBytes, stream>>>(args...);
```

| Parameter | Type | Meaning | Where it is covered |
|---|---|---|---|
| `grid` | `int` or `dim3` | number of **blocks** | Module 1, Module 3 |
| `block` | `int` or `dim3` | **threads per block**; max 1024 on sm_89 | Module 1, Module 3 |
| `sharedBytes` | `size_t` | bytes of **dynamic shared memory** per block; default 0 | **Module 6** |
| `stream` | `cudaStream_t` | which **stream** the launch is queued on; default 0 | **Module 24** |

In this module you will only ever write the two-parameter form. Both omitted
parameters default to "0", which means "no dynamic shared memory" and "the
default stream".

`grid` and `block` are really `dim3` (three unsigned fields, `.x/.y/.z`, unset
fields default to 1). Writing `<<<3907, 256>>>` is shorthand for
`<<<dim3(3907,1,1), dim3(256,1,1)>>>`. **Module 3 owns multi-dimensional
indexing**; everything here uses the one-dimensional form

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

and nothing more.

The arguments in `(args...)` are copied, by value, into a small per-launch
parameter buffer that the driver ships to the GPU alongside the launch. That is
why a pointer argument must be a **device** address: the GPU will dereference
it, and it dereferences it in the GPU's address space.

### 4. The launch is asynchronous. This is the module.

`kernel<<<...>>>(...)` **enqueues** work and returns. It does not wait for the
kernel to start, let alone finish. On this machine the launch statement costs
about **3–7 µs** of host time when the queue is already busy (measured in
`example01.cu`), while the kernel it enqueued may run for milliseconds.

Three consequences, all of which you must internalize now:

**(a) You cannot time a kernel with a CPU clock around the launch.**
`clock()`, `std::chrono`, `QueryPerformanceCounter` around a launch measure the
enqueue, not the work. You would conclude your kernel takes 4 µs. Use CUDA
events (`cudaEventRecord` / `cudaEventElapsedTime`), which are timestamps
recorded *in the GPU's stream*, next to the work itself. Every timing in this
course uses events with a warm-up launch and ≥ 20 timed iterations.

**(b) You cannot see a kernel's error at the launch site.** The launch
statement can only report problems the *driver* can detect before submitting:
an illegal configuration. Anything the kernel does wrong while running is
detected later, by hardware, and is reported by the next call that actually
waits for the GPU.

**(c) Launch overhead is a real cost at small sizes.** A kernel that does 5 µs
of work is 50% launch overhead. This is why "one kernel per tiny operation" is
a bad pattern and why kernel fusion exists. You will meet this again in the
optimization modules.

What the launch actually does, in hardware terms: the runtime writes a command
packet (kernel entry point address, grid/block dimensions, parameter buffer,
shared-memory request) into a command buffer; the driver submits that buffer to
a hardware queue; the GPU's front end (the **GigaThread engine**, Module 1)
pops it and begins dispatching blocks to SMs. The CPU is only involved in the
first step.

### 5. `cudaDeviceSynchronize()` — the blunt instrument

`cudaDeviceSynchronize()` blocks the calling host thread until **everything**
previously submitted to the device — every stream, every kernel, every async
copy — has completed. It is correct, and it is coarse:

- it serializes CPU and GPU, destroying any overlap you may have arranged;
- it waits for work you did not care about;
- it says nothing about *which* operation failed, only that something did;
- in a loop it is a performance bug: sync-per-iteration turns a pipeline into a
  ping-pong.

Use it (i) while debugging, (ii) at a genuine program-wide barrier, (iii) in the
`CHECK_KERNEL()` idiom below during development. The finer-grained tools —
`cudaStreamSynchronize`, `cudaEventSynchronize`, `cudaEventQuery` — arrive in
Module 24.

The most common misconception it causes: *"I must call
`cudaDeviceSynchronize()` before I can copy results back."* You usually must
not, because the ordinary blocking `cudaMemcpy` on the default stream already
synchronizes with respect to previously issued work. Exercise 3 makes you
commit to an answer on this before you find out.

### 6. Error checking, done properly

CUDA runtime functions return `cudaError_t`. The launch statement does not
return anything — so the runtime keeps a **per-host-thread "last error"** slot,
and you query it.

```cpp
cudaError_t cudaGetLastError(void);   // returns the stored error AND clears it
cudaError_t cudaPeekAtLastError(void);// returns it, leaves it in place
```

The clearing behaviour matters. If you call `cudaGetLastError()` for logging in
one place, the next check will see `cudaSuccess` and conclude everything is
fine. Conversely, if you *never* clear, a stale error from an earlier operation
gets blamed on an innocent later one. Rule: **one `cudaGetLastError()` per
launch, immediately after the launch, and nowhere else.** Use
`cudaPeekAtLastError()` when you want to inspect without consuming.

#### Two checks, because there are two classes of error

```cpp
kernel<<<grid, block>>>(args);
CHECK(cudaGetLastError());        // (1) launch-configuration errors
CHECK(cudaDeviceSynchronize());   // (2) errors raised during execution
```

**(1) catches what the driver rejected before submission**, for example:

| Mistake | Error |
|---|---|
| `<<<1, 1025>>>` (max is 1024 on sm_89) | `cudaErrorInvalidValue` / `cudaErrorInvalidConfiguration` |
| grid dimension 0 | `cudaErrorInvalidConfiguration` |
| dynamic shared memory > 48 KB without opt-in | `cudaErrorInvalidValue` |
| kernel needs more registers/shared memory than an SM has | `cudaErrorLaunchOutOfResources` |

*(The exact code for "too many threads" is toolkit-version dependent — CUDA
13.2 reports `cudaErrorInvalidValue` for `<<<1,1025>>>`. Check the name, not
your memory of it.)*

**(2) catches what the hardware raised while the kernel was running**:
`cudaErrorIllegalAddress` (a load or store outside any valid allocation),
`cudaErrorMisalignedAddress`, `cudaErrorLaunchTimeout`, device-side `assert`
failures. These *cannot* be seen by check (1), because at the time (1) runs the
kernel may not have executed a single instruction.

Omit (2) and the error does not disappear — it surfaces at the next
synchronizing call, which is usually a `cudaMemcpy` fifty lines away. That is
the entire premise of Exercise 2.

#### Sticky vs non-sticky errors

| | Non-sticky | Sticky |
|---|---|---|
| Examples | `cudaErrorInvalidValue`, `cudaErrorMemoryAllocation`, bad launch config | `cudaErrorIllegalAddress`, `cudaErrorMisalignedAddress`, device assert, `cudaErrorLaunchFailure` |
| Cause | The runtime refused the request | The GPU faulted while executing |
| Effect on the context | none | **destroyed** |
| Recovery | clear it and carry on | none, within this process |
| Cleared by `cudaGetLastError()`? | yes | **no** |

A sticky error means the CUDA **context** — the GPU-side address space, module
table, and allocation bookkeeping for your process — has been torn down. Every
subsequent CUDA call in that process returns the same error, including
`cudaFree` and `cudaMalloc`, which are not touching anything illegal. The only
"recovery" is to destroy the context (`cudaDeviceReset()`, which throws away all
your allocations and results) or to restart the process. Run
`example02.exe --illegal` and watch the cascade.

The practical consequence: **an illegal memory access has no local symptom.**
It poisons everything after it. If you see six identical errors in a row from
unrelated API calls, the real fault is the earliest one — and it is almost
always a kernel.

#### The macros

Every file in this course defines these two:

```cpp
#define CHECK(x) do {                                                      \
    cudaError_t e_ = (x);                                                  \
    if (e_ != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                  \
                cudaGetErrorName(e_), __FILE__, __LINE__,                  \
                cudaGetErrorString(e_));                                   \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

#define CHECK_KERNEL() do {                                                \
    CHECK(cudaGetLastError());                                             \
    CHECK(cudaDeviceSynchronize());                                        \
} while (0)
```

Details that are not incidental:

- `do { … } while (0)` so the macro is a single statement and survives
  `if (c) CHECK(f()); else …`.
- The argument is evaluated **exactly once** into `e_`. `if ((x) != cudaSuccess)
  { … (x) … }` would call the function twice.
- `__FILE__`/`__LINE__` expand at the *call site*, which is the whole point.
- Both `cudaGetErrorName` (the enum spelling, greppable) and
  `cudaGetErrorString` (prose).
- `CHECK_KERNEL()` contains a full device synchronize, so it is a **development
  tool**. In a release build you keep the `cudaGetLastError()` and drop the
  sync, or compile the sync out behind `#ifndef NDEBUG`. Shipping a
  `cudaDeviceSynchronize()` after every launch is a self-inflicted performance
  bug.

`compute-sanitizer --tool memcheck ./prog.exe` is the tool that turns "illegal
address somewhere" into "line 72, block 3928, thread 0, 22,261 bytes past a
4,000,012-byte allocation". Build with `-lineinfo` so it can name the line. It
slows execution by roughly an order of magnitude; that is a fine trade.

### 7. `printf` from device code

Device code may call `printf`. It is genuinely useful and has four caveats:

1. **It is buffered.** Output goes into a fixed-size circular device buffer
   (default 1 MB, settable via `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, …)`).
   If the buffer overflows, the **oldest** entries are discarded silently.
2. **It is flushed only at synchronization points** — kernel completion
   observed by `cudaDeviceSynchronize`, a blocking `cudaMemcpy`, stream/event
   synchronize, or `cudaDeviceReset`. Not at the `printf` itself. So device
   output appears *after* host output that was printed later in program order.
   In `example01.cu` the host's "AFTER launch statement" line always precedes
   the device lines.
3. **Ordering among threads is not guaranteed.** Each thread's individual
   `printf` is atomic with respect to other threads, but the interleaving is
   whatever order the warps reached the call and the buffer was drained. On this
   GPU `<<<2,4>>>` reliably prints block 1 before block 0 — reliably, but not
   *guaranteed*, and you must never write code that depends on it.
4. **It perturbs what you are measuring.** Every call writes to global memory
   and serializes; a `printf` inside a hot loop changes the register count, the
   occupancy, and the timing. Never leave one in a kernel you are profiling.

Mix host and device `printf` and you get a third hazard: the host's stdout is
block-buffered when redirected to a file or pipe, so the *apparent* order can
change between a console run and `prog.exe > out.txt`. Both examples call
`fflush(stdout)` after each host `printf` so the ordering you observe is the
real one.

### 8. Host pointers and device pointers

```cpp
float* d_x = nullptr;
cudaMalloc((void**)&d_x, bytes);   // d_x is a HOST variable holding a DEVICE address
cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
kernel<<<g,b>>>(d_x, …);
cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
cudaFree(d_x);
```

- `cudaMalloc` takes `void**` because it must *write* a pointer value into a
  variable that lives on the host. The memory it names lives in GPU DRAM.
- The returned address is at least 256-byte aligned — relevant to coalescing in
  Module 5.
- `cudaMemcpy(dst, src, count, kind)` — **count is in BYTES**, and `dst` comes
  first, like `memcpy`. The blocking form on the default stream synchronizes
  with respect to previously issued device work, which is why Exercise 3's
  Phase C is safe.
- `cudaMemset` also takes a **byte** count and sets **bytes**, not elements.
  `cudaMemset(d_x, 1, n*sizeof(float))` does not fill an array with 1.0f.

**A device pointer dereferenced on the host is a segmentation fault.** Not a
CUDA error — the CUDA runtime is never even called. `d_x` holds something like
`0x1302400000`, a perfectly ordinary 64-bit integer that names a page in the
GPU's address space. The CPU's MMU has nothing mapped there, so `*d_x` is an
access violation and your process dies immediately. Symmetrically, passing a
host pointer to a kernel gives you `cudaErrorIllegalAddress` at the next
synchronizing call. The compiler cannot catch either one: both are `float*`.
The `h_`/`d_` naming convention exists precisely because the type system will
not help you. Run `example02.exe --hostderef` to see the crash.

(Unified/managed memory, `cudaMallocManaged`, blurs this deliberately; it is a
later topic. Until then, keep the two worlds separate in your head.)

---

## Hardware Mental Model

**Why is the launch asynchronous?** Because the GPU is a separate device at the
end of a PCIe link, driven by a command queue, exactly like a disk controller or
a NIC. The CPU writes a command packet describing the launch into a ring buffer
in memory; the GPU's front end reads it. A round trip across PCIe costs on the
order of a microsecond. If every launch were synchronous, a program issuing
1000 kernels would spend milliseconds of pure latency doing nothing, and the
GPU would idle between kernels while the CPU decided what to do next. Queueing
is what lets the CPU run ahead and keep the front end fed. Asynchrony is not a
convenience feature bolted on; it is the only way the architecture makes sense.

This also explains the ~3–7 µs floor on launch cost. That is the CPU-side cost
of validating arguments, marshalling the parameter buffer, and appending to the
command buffer — plus, on Windows WDDM, the cost of the driver deciding whether
to submit the buffer to the OS scheduler now or batch it with more work. The
first launch after the queue has drained is measurably more expensive (~10–20 µs
here) than a launch appended to a queue the GPU is already working through
(~3–7 µs), because the former forces a submission. **ARCHITECTURE-SPECIFIC
OPTIMIZATION**: the WDDM batching behaviour is a Windows driver-model property;
on Linux with the native driver the numbers are lower and less bimodal. The
*existence* of a microsecond-scale floor is a **PORTABLE CUDA CONCEPT**.

**Why must the error be reported late?** An illegal address is detected by the
memory management unit inside the GPU, when a thread issues a load or store to a
page with no valid mapping. That happens potentially milliseconds after the host
executed the launch statement, on a different chip, with the host thread long
gone. There is no mechanism by which the launch statement could have known. The
fault is recorded by the hardware, propagated to the driver on the next
interaction, and returned to the host on the next call that waits.

**Why is it sticky?** The MMU fault leaves the SM in an undefined state
mid-kernel: warps are aborted with registers and shared memory holding
arbitrary values, and in-flight memory transactions may or may not have landed.
The driver cannot reason about which of your allocations are still coherent, so
it does the only sound thing and invalidates the entire context. Contrast a
non-sticky error like `<<<1,1025>>>`: the driver rejected the request before any
hardware state changed at all, so nothing needs to be invalidated.

**Why is device `printf` buffered?** Because there is no path from an SM to
your terminal. The device-side `printf` implementation reserves space in a
global-memory FIFO with an atomic bump, copies the format string pointer and the
promoted arguments into it, and returns. The *host* runtime walks that buffer
and does the actual formatting and I/O when it next synchronizes. So: the
ordering you see is the order threads won the atomic, not the order they
executed; overflow drops the oldest records because the FIFO wraps; and the
formatting cost is paid on the CPU. It also explains caveat 4 — that atomic bump
is a contended global-memory atomic, so a `printf` in a hot loop serializes your
warps.

**Why does a thread block map to exactly one SM?** (Module 1, restated because
it is about to matter.) The block's threads share registers and shared memory
allocated from one SM's partition, and a block cannot migrate. So the
`blockIdx.x * blockDim.x + threadIdx.x` index you compute is stable for a
thread's whole lifetime, and the grid size you choose determines how many
blocks the GigaThread engine has to spread over 40 SMs.

---

## Code Walkthrough

### `example01.cu` — the launch is asynchronous

The kernel is as small as it gets:

```cpp
__global__ void hello(void)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("    [device] block %d thread %d (global id %d)\n",
           blockIdx.x, threadIdx.x, gid);
}
```

`__global__`, returns `void`, takes no arguments, and reads three built-in
variables that exist only in device code. Launched with `hello<<<2, 4>>>();` —
2 blocks × 4 threads = 8 instances of this function body.

Part 1 brackets the launch with host prints and an explicit sync:

```cpp
printf("  [host] BEFORE launch\n");              fflush(stdout);
hello<<<2, 4>>>();
CHECK(cudaGetLastError());
printf("  [host] AFTER launch statement, BEFORE cudaDeviceSynchronize\n");
fflush(stdout);
CHECK(cudaDeviceSynchronize());
printf("  [host] AFTER cudaDeviceSynchronize\n"); fflush(stdout);
```

Observed:

```
  [host] BEFORE launch
  [host] AFTER launch statement, BEFORE cudaDeviceSynchronize
    [device] block 1 thread 0 (global id 4)
    [device] block 1 thread 1 (global id 5)
    [device] block 1 thread 2 (global id 6)
    [device] block 1 thread 3 (global id 7)
    [device] block 0 thread 0 (global id 0)
    ...
  [host] AFTER cudaDeviceSynchronize
```

Two things to read off it. The host's post-launch line comes out **before** any
device output — the launch returned immediately, and the device FIFO had not
been drained yet. And block 1 printed before block 0: there is no ordering
guarantee between blocks, in output or in execution.

Part 2 measures the launch. Note which clock is used for what:

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
busy<<<blocks, threads>>>(d_out, n, kIters);
auto t1 = std::chrono::high_resolution_clock::now();   // CPU time for the CALL
```

versus

```cpp
CHECK(cudaEventRecord(evStart));
for (int r = 0; r < kReps; ++r) busy<<<blocks, threads>>>(d_out, n, kIters);
CHECK(cudaEventRecord(evStop));
CHECK(cudaEventSynchronize(evStop));                   // GPU time for the WORK
```

Measured on the RTX 3500 Ada:

```
  launch after an idle queue      :    17.70 us (CPU time)
  launch into a busy queue        :     5.04 us (CPU time)
  kernel actually ran for         :  2981.38 us (GPU time)
  work / enqueue ratio            :    591.0x
```

The two host numbers move run to run (observed 12–18 µs idle-queue, 5.0–5.4 µs
busy-queue across several runs) because they are host timings subject to OS
scheduling and WDDM batching; the GPU number is stable to ~1% (2841–2981 µs).

A CPU clock around the launch would have told you this 3 ms kernel takes 5 µs.

Part 3 issues `busy<<<1, 1025>>>` — one more thread per block than sm_89
allows:

```
  cudaGetLastError() after <<<1,1025>>> : cudaErrorInvalidValue (invalid argument)
  cudaGetLastError() a second time      : cudaSuccess  <- cleared by the first read
  a legal launch after the failed one   : OK (error was NON-STICKY)
```

Three lessons in four lines: the launch-config error *is* visible at the launch
site without syncing; `cudaGetLastError()` clears (which is why the second read
says success, and why you must not sprinkle it around); and the context
survived, so this class of error is recoverable.

### `example02.cu` — the pipeline and its two failure modes

The shared math is written once:

```cpp
__host__ __device__ __forceinline__ float smoothstep01(float x)
{
    float t = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
    return t * t * (3.0f - 2.0f * t);
}
```

and used from both sides — `out[i] = smoothstep01(in[i])` in the kernel,
`h_ref[i] = smoothstep01(h_in[i])` in `main`. Default run:

```
n = 1000003, launch <<<3907, 256>>>  (1000192 threads, 189 beyond the data)
max abs difference vs CPU reference = 0.000e+00 at i = -1
PASS (0 mismatching elements)
```

Exactly zero difference here, because the expression contains no FMA-contractible
`a*b+c` pattern that the two compilers could handle differently. Do not expect
that in general (Exercise 1 differs by 4×10⁻⁶ for exactly that reason).

`--illegal` runs a bounds-guard-free kernel with a grid sized from the byte
count. Real output:

```
  cudaGetLastError() right after launch      -> cudaSuccess (no error)
  cudaDeviceSynchronize()                    -> cudaErrorIllegalAddress (...)
  From now on the context is poisoned. Every call fails:
  cudaGetLastError()                         -> cudaErrorIllegalAddress (...)
  cudaMemcpy(D2H)                            -> cudaErrorIllegalAddress (...)
  cudaMalloc(1 byte)                         -> cudaErrorIllegalAddress (...)
  cudaFree(d_in)                             -> cudaErrorIllegalAddress (...)
  cudaDeviceReset()                          -> cudaSuccess (no error)
```

Read it carefully. The launch-site check is clean. `cudaGetLastError()` does
**not** clear the sticky error. `cudaMalloc` of one byte fails. And
`cudaDeviceReset()` "succeeds" only because its job is to destroy the context,
which was already dead.

`--hostderef` evaluates `d_in[0]` on the host and the process dies with a
segmentation fault before printing anything further. No `cudaError_t` is
produced, because no CUDA function was called.

---

## Check Your Understanding

Answers in `solutions/module02/check_your_understanding.md`. Reason them out
before looking.

1. A colleague reports that their kernel "runs in 4 microseconds" for a 400 MB
   input, measured with `std::chrono` around the launch. Compute the fastest
   this kernel could possibly be on a 432 GB/s device, and use that number to
   say precisely what their 4 µs measured. Then explain why their result would
   *change* — without any code change — if they had called
   `cudaDeviceSynchronize()` at the top of the previous loop iteration.

2. A program launches kernel A, then kernel B, then calls `cudaDeviceSynchronize()`
   once, which returns `cudaErrorIllegalAddress`. There is exactly one
   `cudaGetLastError()` call in the program and it is after B's launch; it
   returned `cudaSuccess`. Which kernel faulted? What is the smallest change to
   the program that would tell you, and what does it cost? Is there a case where
   even that change leaves you unsure?

3. Someone proposes "defensive" error handling: call `cudaGetLastError()` at the
   top of `main`, after every `cudaMalloc`, before every launch, and after every
   launch, logging only when it is non-`cudaSuccess`. Describe a concrete
   scenario in which this scheme *hides* a real bug that the two-check idiom
   would have caught. Then describe a second scenario in which it *invents* a
   bug that does not exist.

4. Why can a `__device__` function be recursive, have a return value, and take a
   reference parameter, while a `__global__` function must return `void`? Answer
   in terms of who invokes each and when the invocation completes — not "because
   the spec says so".

---

## Exercises

### Exercise 1 — `exercise01.cu` (fill in the code, 3 TODOs)

Evaluate `p(x) = 2x⁵ − 3x⁴ + 0.5x³ + x² − 4x + 7` elementwise over
`n = 1,000,003` floats using Horner's rule, and validate against a CPU
reference that calls the *same* source function.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

- **TODO 1** — give `horner5` the execution-space qualifier(s) it needs. Find
  every call site first. The obvious minimal answer compiles the device side
  and breaks the host side.
- **TODO 2** — write the kernel body: 1-D global index, one element per thread,
  and a guard that protects **both** the load and the store. A guard on the
  store alone is still an out-of-bounds read.
- **TODO 3** — compute the block count. `n` is not a multiple of 256.

Validation: element-wise comparison against the CPU reference with tolerance
`1e-5 * max(1, |ref|)`; prints the worst absolute difference, its index, and
PASS/FAIL. As shipped (TODOs blank) the file compiles and prints
`Set TODO 3 (blocks) first.`

### Exercise 2 — `exercise02.cu` (debugging, 2 TODOs)

A `saxpy`-with-clamp program that reports `cudaErrorIllegalAddress` at a
`cudaMemcpy` whose arguments are obviously correct, and then reports the same
error from `cudaFree`. The file header states the symptom only.

```
nvcc -arch=sm_89 -O3 -lineinfo -o exercise02.exe exercise02.cu
.\exercise02.exe
compute-sanitizer --tool memcheck .\exercise02.exe
```

- **TODO 1** — make the program blame the line that is actually at fault.
- **TODO 2** — fix the root cause (a one-line change elsewhere in the file).

Validation: PASS requires both zero mismatching elements *and* zero reported
CUDA errors. Expect the broken version to crash-ish and print FAIL; that is the
starting point, not a bug in the exercise.

### Exercise 3 — `exercise03.cu` (predict the behavior, 3 TODOs)

Commit to three predictions in writing, then run once.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

- **TODO 1** — the interleaving of host and device output around an async
  launch (choose one of four orderings), plus a written justification for
  whether the device lines come out in thread-id order.
- **TODO 2** — which bucket the host-side cost of the launch statement falls
  into, for a kernel with milliseconds of GPU work.
- **TODO 3** — whether a `cudaMemcpy` D2H issued immediately after a launch,
  with **no** `cudaDeviceSynchronize()`, returns the kernel's results or stale
  data. Justify from the semantics of the call, not from what feels safe.

Validation: the program measures TODO 2 and TODO 3 itself and prints
MATCH/MISMATCH plus PASS/FAIL; TODO 1 you check by eye against the Phase A
output. As shipped it prints a reminder to fill in the predictions.

---

## Prediction

Commit to these in writing before you compile anything.

1. In `example01.cu` Part 3, after the illegal `<<<1,1025>>>` launch, does the
   *next*, legal launch succeed, or does the program die? State which class of
   error you think `<<<1,1025>>>` is and why the answer follows from that.

2. `example02.cu --illegal` prints the result of `cudaGetLastError()`
   immediately after the launch of an out-of-bounds kernel. Will it be
   `cudaSuccess` or `cudaErrorIllegalAddress`? Now predict the return value of
   the `cudaFree` twelve lines later, and say what the two answers together
   imply about where in your code you should look when you see a CUDA error.

3. In Exercise 1, the GPU result and the CPU result come from the same source
   lines for `horner5`. Predict whether the worst absolute difference will be
   exactly 0, around 10⁻⁷, around 10⁻⁵, or around 10⁻². Name the mechanism that
   produces whatever difference you predict.
