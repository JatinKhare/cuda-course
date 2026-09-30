# Module 02 / Exercise 01 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise01_solution.exe exercise01_solution.cu
.\exercise01_solution.exe
```

## TODO 1 — the execution-space qualifier

```cpp
__host__ __device__ __forceinline__ float horner5(float x)
```

`horner5` has two call sites on opposite sides of the host/device line:

- `out[i] = horner5(x[i]);` inside `__global__ void eval_poly` — device code;
- `h_ref[i] = horner5(h_x[i]);` inside `main` — host code.

nvcc compiles the translation unit twice. In the device pass, only entities
visible to device code exist; in the host pass, only entities visible to host
code exist. A function annotated `__host__ __device__` is emitted in both
passes, producing an x86-64 body and an Ada SASS body from one set of source
lines. That is the entire point: the CPU reference and the GPU kernel cannot
drift apart through a transcription error, because there is nothing to
transcribe.

**The obvious wrong answer is `__device__` alone.** It makes the kernel compile
and then breaks the host pass:

```
error: identifier "horner5" is undefined in host code
```

or, depending on the toolkit's phrasing, `calling a __device__ function
("horner5") from a __host__ function ("main") is not allowed`. The error does
not appear at the line you edited, which is what makes it worth planting.

Other wrong answers and their symptoms:

| Attempt | Symptom |
|---|---|
| `__host__` (as shipped) | Fine until TODO 2 is written; then `calling a __host__ function from a __global__ function is not allowed`. |
| `__global__` | `a __global__ function must return void` plus `a __global__ function cannot be called from device code` (without CDP). Kernels are entry points, not helpers. |
| Duplicating the body: one `__device__` copy and one `__host__` copy | Compiles, validates, and is exactly the maintenance hazard the qualifier exists to prevent. If you did this, you did not answer the question. |

`__forceinline__` is orthogonal — it is an inlining hint for the device
compiler, not an execution space. The device compiler would almost certainly
inline a leaf function like this anyway; stating it makes the intent explicit
and keeps the kernel free of a call frame.

## TODO 2 — the kernel body

```cpp
__global__ void eval_poly(const float* x, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = horner5(x[i]);
}
```

The point that is graded is the **placement of the guard**. This is wrong:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
float v = horner5(x[i]);        // <-- out-of-bounds READ for i >= n
if (i < n) out[i] = v;
```

It "works" on most runs and is still a bug. `n = 1000003` with 3907 blocks of
256 threads launches 1,000,192 threads, so 189 threads have `i >= n` and read
up to 756 bytes past the end of `d_x`. Whether that faults depends on the
allocator's granularity — `cudaMalloc` rounds up, so those bytes usually land
inside padding the driver happens to own, and nothing happens. Until the
allocation size changes, or the driver version changes, and then it faults in
production. `compute-sanitizer --tool memcheck` reports it immediately as
`Invalid __global__ read of size 4 bytes`, which is why you run it on code that
passes.

Why must the guard exist at all? Blocks are the unit of dispatch: the
GigaThread engine hands whole blocks to SMs, so you cannot launch 1,000,003
threads. You launch a whole number of blocks and mask off the excess. Module 3
generalizes this (grid-stride loops, multi-dimensional bounds); here the 1-D
form is all that is needed.

Note also `const float*` on the input. It costs nothing and it documents the
access pattern; from Module 4 on it also lets you ask for the read-only data
path with `__restrict__`.

## TODO 3 — the block count

```cpp
int blocks = (n + threads - 1) / threads;   // (1000003 + 255) / 256 = 3907
```

Integer ceiling division. `n / threads` gives 3906, covering 999,936 elements
and silently dropping the last 67. Verified: it prints
`FAIL (67 mismatching elements of 1000003)` with the worst difference at
exactly `i = 999936` — a run of failures starting at a multiple of the block
size is the tell-tale signature of a truncating grid computation. `n / threads + 1` happens
to be right here but is wrong whenever `n` *is* a multiple of `threads`: it
launches one entirely idle block. The idle block costs a dispatch and a few
microseconds; more importantly it is a latent off-by-one that will eventually
meet a kernel without a guard.

3907 blocks over 40 SMs is ~98 blocks per SM — many waves, so the tail effect
of the partly-idle final block is negligible here.

## Synchronization / memory reasoning

There is no synchronization in this kernel, and none is needed: every thread
reads one element and writes one element, and no thread reads anything another
thread wrote. This is an *embarrassingly parallel map*. The only ordering
requirement in the whole program is between the kernel and the D2H copy, and
that is supplied for free by the blocking `cudaMemcpy` on the default stream.

`CHECK_KERNEL()` after the launch does both checks. With a correct TODO 3 the
launch config is legal, so `cudaGetLastError()` returns success; with a correct
TODO 2 nothing faults, so `cudaDeviceSynchronize()` returns success too. Break
either and you get exactly one of the two.

## Performance reasoning

This kernel reads 4 bytes and writes 4 bytes per element and does 5 FMAs. At
1,000,003 elements that is 8.0 MB of traffic and 5 MFLOP — an arithmetic
intensity of 0.625 FLOP/byte, far below the ridge point of this machine
(~16 TFLOP/s FP32 ÷ 432 GB/s ≈ 37 FLOP/byte). It is **memory bound by a
factor of ~60**, and the polynomial is free: you could evaluate degree 50 here for
no additional time. That is the intuition to carry forward — on a GPU, "make
the math cheaper" is almost never the optimization.

The exercise does not time anything, deliberately. Timing belongs with events
and warm-ups, and Exercise 3 introduces that.

## Expected output

Actual output on the RTX 3500 Ada Laptop GPU, CUDA 13.2:

```
n = 1000003, threads/block = 256, blocks = 3907  -> 1000192 threads launched
worst |gpu - cpu| = 4.292e-06 at i = 364 (gpu -6.9441495, cpu -6.9441452)
PASS (0 mismatching elements of 1000003)
```

The worst difference is **not zero**, and it is not a mistake. The device
compiler contracts `p * x + c` into a single `fma.rn.f32` — one rounding for
the multiply-add. MSVC on the host emits a separate multiply and add — two
roundings — unless it happens to use FMA. Five contracted steps in a chain
whose intermediate values reach ~30 in magnitude accumulate to ~4×10⁻⁶
absolute, i.e. ~6×10⁻⁷ relative, which is about 5 ULP. Well inside the
`1e-5 * max(1,|ref|)` tolerance, and well outside "bitwise identical".

Verified: rebuilding with `nvcc -arch=sm_89 -O3 -fmad=false` gives
`worst |gpu - cpu| = 0.000e+00`, i.e. the difference collapses to exactly zero — at the cost
of roughly doubling the instruction count of the polynomial. Do not do that to
"fix" a validation failure; fix the tolerance, or compare against a
double-precision reference.

## The result that matters

One set of source lines, two code generators, two results that differ in the
last few bits — and a validation scheme that is written in terms of a tolerance
rather than equality is the only reason the program says PASS. `__host__
__device__` buys you a single definition of the algorithm; it does not buy you
bitwise agreement, because the two compilers are allowed to contract
floating-point expressions differently. Every GPU program you write from here
on needs a tolerance chosen from the dtype and the length of the dependent
chain, not from hope.

Variation to try: change the tolerance to exact equality and watch how many of
the 1,000,003 elements fail. Then raise the degree of the polynomial (add terms
to the Horner chain) and watch the worst difference grow roughly with the
number of contracted steps — that is error accumulation in a dependent chain,
measured rather than asserted.
