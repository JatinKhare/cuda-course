# Module 04 / Exercise 02 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

## The diagnosis

The symptom given in the header was "roughly a quarter of the throughput you
would expect", with a pointer at `-Xptxas -v`. This is the verbatim output for
the shipped `fir_v0`:

```
ptxas info    : Compiling entry function '_Z6fir_v0PKfPfii' for 'sm_89'
ptxas info    : Function properties for _Z6fir_v0PKfPfii
    64 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 30 registers, used 0 barriers, 64 bytes cumulative stack size, 376 bytes cmem[0]
```

**`64 bytes stack frame`** is the whole story. 64 B = 16 floats = `win[TAPS]`.
The array is not in registers; it is in **local memory**, which is device DRAM.

Note carefully what is *not* there: `0 bytes spill stores, 0 bytes spill loads`.
This is not register pressure. The kernel uses 30 registers out of a budget of
255. Nothing was spilled. The array was never a candidate for the register file
in the first place.

The cause is the loop bound:

```cpp
__global__ void fir_v0(..., int n, int taps)
...
    for (int i = 0; i < taps; ++i) win[i] = x[t + i];
```

`taps` is a kernel argument, so the trip count is unknown at compile time, so
the loop cannot be unrolled, so `win[i]` is a **dynamic index**. The SM's
register file is not addressable — register numbers are encoded in the
instruction word; there is no "load register number R[i]" instruction. An array
that must be indexed by a runtime value therefore cannot live in registers, and
ptxas puts it in the only per-thread space that has addresses: local memory.

You can confirm it in SASS:

```
$ nvcc -arch=sm_89 -O3 -c exercise02_solution.cu
$ cuobjdump -sass exercise02_solution.o
        Function : _Z6fir_v0PKfPfii
        /*01f0*/   LDG.E.CONSTANT R4, [R2.64] ;
        ...
        /*0330*/   STL.128 [R0], R4 ;
        /*0340*/   STL.128 [R0+0x10], R8 ;
        /*0350*/   STL.128 [R0+0x20], R12 ;
        /*0360*/   STL.128 [R0+0x30], R16 ;
```

24 `LDL`/`STL` (load-local/store-local) instructions in `fir_v0`; **zero** in
`fir_v1`. The kernel loads the window from global memory into registers and then
immediately writes it back out to DRAM, because it has been told to build an
addressable array.

### What compute-sanitizer tells you (nothing)

```
> compute-sanitizer --tool memcheck exercise02_solution.exe
========= COMPUTE-SANITIZER
...
v0         92.4364        1.5      0.3%       64       ok
v1         74.5981        1.8      0.4%        0       ok
========= ERROR SUMMARY: 1 error
```

The single reported "error" is a benign `cudaDeviceReset` API warning
("Resetting device while there are still other users claiming to use it"). There
are **no illegal accesses, no invalid reads, no uninitialised values** — a local
memory spill is not a correctness bug, it is a placement decision, and memcheck
is the wrong instrument entirely. Note also that under the sanitizer everything
runs ~150× slower and the harness's timing-based assertions fail; never read
performance numbers from a sanitizer run.

The right instruments for this bug class are, in order: `-Xptxas -v` /
`cudaFuncGetAttributes` (is there a stack frame at all?), `cuobjdump -sass`
(where are the `LDL`/`STL`?), and Nsight Compute's *Memory Workload Analysis*
section, which reports local load/store traffic as a separate line item.

## TODO 1 — `fir_v1`

```cpp
__global__ void fir_v1(const float* __restrict__ x, float* __restrict__ y, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    float win[TAPS];

    #pragma unroll
    for (int i = 0; i < TAPS; ++i) win[i] = x[t + i];

    float m = 0.0f;
    #pragma unroll
    for (int i = 0; i < TAPS; ++i) m = fmaxf(m, fabsf(win[i]));

    float inv = 1.0f / (m + 1e-6f);
    float s   = 0.0f;
    #pragma unroll
    for (int i = 0; i < TAPS; ++i) s = fmaf(c_h[i] * inv, win[i], s);

    y[t] = s;
}
```

```
ptxas info    : Compiling entry function '_Z6fir_v1PKfPfi' for 'sm_89'
ptxas info    : Function properties for _Z6fir_v1PKfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 30 registers, used 0 barriers, 372 bytes cmem[0]
```

**Why it is correct, from the hardware model.** With a compile-time trip count
the loop is fully unrolled, so every occurrence of `win[i]` becomes `win[0]`,
`win[1]`, … `win[15]` — sixteen distinct scalar values with no surviving array
semantics. ptxas allocates sixteen registers and the array disappears. Register
count is unchanged at 30, because those sixteen values were already live in
registers in `fir_v0`; `fir_v0` was merely *also* writing them to DRAM and
reading them back.

**Common wrong approaches:**

| Attempt | Symptom |
|---|---|
| Add `#pragma unroll` to `fir_v0`'s loops without changing `taps` | Stack frame stays at 64 B. `#pragma unroll` with no count on a loop whose bound is a runtime variable cannot fully unroll; the compiler may emit a partially unrolled loop with a residual, and the index is still dynamic. The pragma is not a magic word. |
| `#pragma unroll 16` on the runtime-bound loop | Same. ptxas must still emit the guard and residual for `taps != 16`, so `win[]` remains addressable. |
| Declare `win` as `volatile float win[TAPS]` "to stop the compiler moving it" | Stack frame *grows* and the kernel gets slower. `volatile` forces every access to memory — it is the opposite of what you want. |
| Move `win` to `__shared__` | Works, gets 0 local bytes, and is a bad answer: you have traded free registers for a scarce per-SM resource that now limits occupancy, to store data that is strictly per-thread and never shared. Shared memory is for inter-thread reuse (Module 6). |
| Fuse the three loops into one so no array is needed | Legitimate and fast, but it changes the algorithm: you cannot compute `m` (a max over the whole window) before you have seen the whole window, so the normalized output would be wrong. The harness catches it. |
| Recompute `x[t+i]` in the third loop instead of storing `win` | Also legitimate; the redundant loads hit L1 (windows overlap 15/16). It benchmarks close to `fir_v1`. It is a fine answer as long as you *chose* it rather than stumbled into it. |

**Is `#pragma unroll` required here?** No — `nvcc -O3` fully unrolls a
16-iteration constant-bound loop on its own, and the kernel is already clean
without the pragma. Write it anyway. It documents the requirement, and it stops
a future edit (`#define TAPS 512`) from silently reintroducing a 2 KB stack
frame per thread with no diagnostic.

## TODO 2 / TODO 3 — `fir_v2`, runtime tap count

The tap count is genuinely a runtime value, so the compile-time constant has to
come from somewhere else. It comes from a **template parameter**, with the host
choosing the instantiation:

```cpp
template <int TAPS_C>
__global__ void fir_v2(const float* __restrict__ x, float* __restrict__ y, int n)
{
    ...
    float win[TAPS_C];
    #pragma unroll
    for (int i = 0; i < TAPS_C; ++i) win[i] = x[t + i];
    ...
}

static bool launch_v2(const float* x, float* y, int n, int taps,
                      int blocks, int threads)
{
    switch (taps) {
        case  8: fir_v2< 8><<<blocks, threads>>>(x, y, n); return true;
        case 12: fir_v2<12><<<blocks, threads>>>(x, y, n); return true;
        case 16: fir_v2<16><<<blocks, threads>>>(x, y, n); return true;
        case 32: fir_v2<32><<<blocks, threads>>>(x, y, n); return true;
        default: return false;
    }
}
```

The key idea, and it is the one worth carrying forward: **the compile-time
constant does not have to be known when you write the source, only when the
kernel is compiled.** Each instantiation is a separately compiled kernel image
in which `TAPS_C` is a literal. The runtime value never crosses into device
code as a variable — the *host* consumes it, in a `switch`, and picks a kernel.

Each instantiation is clean and each has a register count proportional to its
window:

```
ptxas info    : Compiling entry function '_Z6fir_v2ILi32EEvPKfPfi' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 45 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compiling entry function '_Z6fir_v2ILi16EEvPKfPfi' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 30 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compiling entry function '_Z6fir_v2ILi12EEvPKfPfi' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 26 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compiling entry function '_Z6fir_v2ILi8EEvPKfPfi' for 'sm_89'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 0 barriers, 372 bytes cmem[0]
```

22 → 45 registers as the window grows from 8 to 32. That is the trade you have
made explicit: window length now shows up directly in register pressure, and
therefore (Module 19) in occupancy. At 45 registers you can still hold 48 warps
(65536 / (45 × 32) = 45.5 → 45 warps, essentially the cap); push `TAPS_C` to 128
and you would be back in trouble, this time via spills rather than dynamic
indexing. A 512-tap filter should *not* use this structure.

The cost of templating is binary size: one kernel image per supported length.
For a small fixed set of shapes this is almost always worth it, and it is how
every production GPU library (cuDNN, CUTLASS, cuBLASLt) handles fixed-shape
kernels.

**Other acceptable answers:** `if constexpr` dispatch, a macro-generated switch,
or NVRTC/JIT specialization. Not acceptable: `-maxrregcount` (it controls
spilling, not addressability, and cannot help here), or hoping `taps` gets
constant-folded (it cannot; it arrives through `cmem[0]` kernel-parameter space
at run time).

## Performance reasoning

Measured, RTX 3500 Ada, N = 16 777 216 samples (64 MB in + 64 MB out, past L2):

```
version         ms       GB/s     %peak  local B   result
v0          1.3428      100.0     23.1%       64       ok
v1          0.4523      296.7     68.7%        0       ok

--- v2, runtime tap count ---
  taps= 8 :   0.4258 ms     315.2 GB/s  ok
  taps=12 :   0.4293 ms     312.6 GB/s  ok
  taps=16 :   0.4437 ms     302.5 GB/s  ok
  taps=32 :   0.6228 ms     215.5 GB/s  ok
```

**3.0×**, from deleting one kernel argument.

Now account for it. The GB/s column counts only the traffic the algorithm
demands: 4 B in + 4 B out per output sample. `fir_v1` reaches 69% of the 432
GB/s peak, which is what a well-behaved streaming kernel with 16× read overlap
(served by L1) looks like on this part.

`fir_v0` demands the same 8 B/thread of useful traffic and additionally moves
64 B/thread to local memory and 64 B/thread back. Naïvely that is 136/8 = 17×
the traffic and should be 17× slower. It is 3×. Two reasons:

1. **Local memory is cached.** The spill store and the reload happen a few
   hundred instructions apart and the line is still in L1. Most of the local
   traffic never reaches DRAM at all — which is why the kernel is "only" 3×
   slower and not 17×, and why local-memory bugs are easy to miss in profiles
   that only look at DRAM counters.
2. **The stores still cost instructions and L1 bandwidth.** 24 extra `LDL`/`STL`
   per thread compete for the same load/store units and the same L1 ports as the
   real loads. At 61 440 resident threads the L1 write bandwidth becomes the
   binding constraint long before DRAM does.

The lesson is that local memory is expensive even when it hits in cache. If it
had missed — a larger array, or lower occupancy leaving less L1 per thread — the
factor would have been far worse.

**Why does `taps=32` drop to 215 GB/s?** 45 registers per thread, plus twice the
per-thread work; the kernel becomes partly instruction-bound. Nothing pathological.

Run-to-run variation on `ms` is roughly ±10% on a laptop GPU with variable
clocks; the *ratio* v0/v1 is stable at 2.9–3.2×.

## Expected output

```
=== NVIDIA RTX 3500 Ada Generation Laptop GPU ===
  fir_v0 : 30 regs,   64 B local/thread
  fir_v1 : 30 regs,    0 B local/thread

version         ms       GB/s     %peak  local B   result
v0          1.3428      100.0     23.1%       64       ok
v1          0.4523      296.7     68.7%        0       ok

--- v2, runtime tap count ---
  taps= 8 :   0.4258 ms     315.2 GB/s  ok
  taps=12 :   0.4293 ms     312.6 GB/s  ok
  taps=16 :   0.4437 ms     302.5 GB/s  ok
  taps=32 :   0.6228 ms     215.5 GB/s  ok

PASS
```

## The result that matters

A per-thread array indexed by anything the compiler cannot evaluate at compile
time is **not** in registers — it is in DRAM, and the compiler will not warn
you. The diagnostic is one flag (`-Xptxas -v`, look for a non-zero *stack
frame*, which is distinct from a non-zero *spill*), and the fix is almost always
to make the index a compile-time constant, by templating if necessary. Add
`-Xptxas -v` to your standard build line and read it every time; this is a
five-second check that routinely finds a 3× regression.

**Variation to try:** instantiate `fir_v2` at 64, 128 and 256 taps and watch the
register count climb. Measured:

```
fir<64>  :   0 bytes stack frame,  0 spill stores,  0 spill loads, 76 registers
fir<128> :   0 bytes stack frame,  0 spill stores,  0 spill loads, 142 registers
fir<256> :  72 bytes stack frame, 68 spill stores, 68 spill loads, 255 registers
```

At 256 taps the array is still fully unrolled and statically indexed, but there
are simply not 256 registers to put it in — the per-thread ceiling is 255 — so
ptxas spills. You now have the *second*, entirely different route into local
memory, and you can read the two apart from one line of compiler output: a
non-zero stack frame with zero spills is an addressability problem, a non-zero
stack frame with non-zero spills is a capacity problem. Note also that even
`fir<128>` at 142 registers caps you at 14 resident warps out of 48, which is
the occupancy conversation Module 19 picks up.
