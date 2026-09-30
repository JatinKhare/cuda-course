# Module 04 — GPU Memory Hierarchy

> Prerequisites: Module 1 (SM structure, warps/blocks/grids), Module 2 (nvcc, kernel qualifiers, error checking), Module 3 (thread indexing, grid-stride loops)
> What this module gives you: a physical map of every place a value can live on an RTX 3500 Ada, what it costs to reach each one, and the syntax that decides which one your variable lands in.

---

## Concept

A CUDA kernel is a C++ function, so every variable in it has a *storage class* in the ordinary C++ sense. On a GPU, storage class also means **physical location**, and the physical locations differ from each other by three orders of magnitude in latency. Choosing the wrong one is not a style issue; it is the difference between 40 cycles and 600.

CUDA exposes seven device-side spaces plus one host-side distinction.

**Registers.** Per-thread, private, on-chip. The SM's register file is real SRAM sitting next to the ALUs. A variable you declare as a plain scalar local (`float x;`) is a register. Scope: one thread. Lifetime: from the point of definition to the end of the kernel — but the *allocation* lasts the whole lifetime of the thread's warp, because registers are assigned once when a block is launched onto an SM and are never re-assigned while it is resident.

**Local memory.** Per-thread, private — and **not on-chip**. This is the trap the module exists for. "Local" describes *scope*, not *location*. Local memory is a per-thread slice of device DRAM, in an address-swizzled layout, cached in L1 and L2 like any other global access. A variable ends up here when the compiler cannot keep it in registers.

**Shared memory.** Per-block, on-chip SRAM. Declared `__shared__`. Every thread in a block sees the same shared array; threads in a different block see a different one. Lifetime: exactly the block's residency on the SM. Latency roughly 20–30× lower than global memory. **Module 6 owns shared memory** — this module only tells you that it exists, where it is, and what it costs. Module 7 covers its bank structure.

**L1 / unified data cache.** Per-SM, hardware-managed. On Ada this is the *same* 128 KB SRAM array as shared memory; the two split it. Not programmer-addressable.

**L2 cache.** Device-wide, 48 MB on this GPU, hardware-managed, the point at which all SMs see a coherent view of memory.

**Global memory.** Device DRAM. 12 GB of GDDR6 on a 192-bit bus at 9.001 GHz DDR → 432 GB/s. Allocated with `cudaMalloc` or declared `__device__`. Scope: the whole device, plus the host via copies. Lifetime: until you free it.

**Constant memory.** A 64 KB read-only window into device memory, declared `__constant__`, written by the host with `cudaMemcpyToSymbol`, and served to the SMs by a small dedicated per-SM constant cache. Fast for one very specific pattern and slow for its opposite — see below.

**Read-only / texture path.** Not a separate memory, but a separate *route* to global memory through the SM's read-only data cache. Requested with `const T* __restrict__` or `__ldg()`.

**Host memory: pageable vs pinned.** Ordinary `malloc` memory is pageable — the OS may relocate or page out its physical frames at any time. A DMA engine cannot chase a moving target, so the driver copies your pageable buffer into a small internal *pinned staging (bounce) buffer* first, then DMAs from there. `cudaMallocHost` / `cudaHostAlloc` give you page-locked memory the DMA engine can read directly. **Module 26 owns host memory and async transfers**; this module measures the difference so you know it is real.

### The table

Measured on an RTX 3500 Ada (sm_89) with the code in this module. Latency is cycles for a single *dependent* load; capacity is per SM unless stated.

| Space | Physical location | Scope | Lifetime | Latency (cycles) | Capacity on sm_89 | How you declare it |
|---|---|---|---|---|---|---|
| Register | SM register file (SRAM) | thread | thread (allocated for the warp's residency) | ~1 (operand read, no load) | 65 536 × 32-bit / SM = 256 KB; **255 / thread** | `float x;` — any scalar local |
| Local | **device DRAM**, cached in L1/L2 | thread | thread | 40 / 241 / 575 (L1 / L2 / DRAM hit) | 512 KB / thread architectural max | `float a[N];` that the compiler cannot register-allocate |
| Shared | on-chip SRAM, shares 128 KB with L1 | block | block residency | ~20–30 | 100 KB / SM addressable; 48 KB / block default, 99 KB opt-in | `__shared__ float s[N];` (Module 6) |
| L1 / unified | on-chip SRAM, per SM | SM (**not coherent across SMs**) | hardware-managed | **40** | 128 KB unified L1+SMEM / SM | not addressable; carveout hint only |
| L2 | on-die SRAM, device-wide | whole device (coherence point) | hardware-managed | **241** | **48 MB** | not addressable |
| Global | GDDR6 DRAM | device + host | until `cudaFree` | **575** | 12 GB @ 432 GB/s | `cudaMalloc`, `__device__` |
| Constant | DRAM window + per-SM constant cache | device, read-only | context | ~L1 when broadcast; **×32 when divergent** | 64 KB total | `__constant__ float t[N];` + `cudaMemcpyToSymbol` |
| Read-only path | global memory via read-only cache | device, read-only in-kernel | — | same as global | shares L1 | `const T* __restrict__`, `__ldg()` |
| Host pageable | system RAM, movable | host | `free` | — | system RAM | `malloc` — 10.9 GB/s H2D |
| Host pinned | system RAM, page-locked | host | `cudaFreeHost` | — | system RAM (limited) | `cudaMallocHost` — 12.1 GB/s H2D |

Every latency in that table is a number you will measure yourself in Exercise 1.

---

## Hardware Mental Model

### Registers, and why they cap parallelism

Each SM on sm_89 has **65 536 32-bit registers**. That is a fixed physical array. When the GigaThread engine places a block on an SM, it must carve out `registers_per_thread × threads_per_block` registers from that array and hold them for the block's entire residency. Registers are never spilled and re-loaded by the scheduler the way a CPU's context switch spills; a CUDA warp's registers stay allocated from the moment it is launched until it retires. *That is precisely what makes warp switching free* — there is no state to save, so the warp scheduler can pick a different warp every cycle at zero cost.

The consequence is arithmetic:

```
resident_warps_per_SM  <=  65536 / (regs_per_thread * 32)
```

At 32 registers per thread you can hold 64 warps — above the 48-warp hardware limit, so registers are not binding. At 64 registers per thread you get 32 warps, and you have lost a third of your latency-hiding capacity. At 128 registers you get 16 warps. The per-thread ceiling is **255 registers**; ask for more and the compiler spills. Occupancy as a formal subject is **Module 19**; for now, note only that register count is one of the three things that decides it.

You read your kernel's register count from the compiler, not from a guess:

```
nvcc -arch=sm_89 -O3 -Xptxas -v -c example01.cu
```

```
ptxas info    : Compiling entry function '_Z11k_registersPKfPfi' for 'sm_89'
ptxas info    : Function properties for _Z11k_registersPKfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 372 bytes cmem[0]
```

Three numbers matter there: **24 registers**, **0 bytes stack frame**, **0 bytes spill**.

### Local memory: the highest-value item in this module

"Local memory" is a naming accident that has cost the GPU community an enormous amount of wasted time. It is **DRAM**. It is as far away as any other global access. The only thing "local" about it is that each thread gets its own copy, and that the hardware interleaves the per-thread copies so that thread *t*'s word *i* is adjacent to thread *t+1*'s word *i* — which makes a warp's access to `a[i]` perfectly coalesced, but does not make it fast.

Two things put a variable there.

**1. Running out of registers.** If the kernel's live values exceed 255 per thread, or exceed whatever budget `-maxrregcount` / `__launch_bounds__` imposed, ptxas spills the excess. You see this as non-zero **`spill stores` / `spill loads`**.

**2. A per-thread array indexed by something that is not a compile-time constant.** This is the common case, and it has nothing to do with running out of registers. The register file is *not addressable*. There is no SASS instruction meaning "read register number R7". Register numbers are encoded in the instruction word. So if the compiler cannot prove which element of `a[]` you want at each use, it has no choice: the array must go somewhere with real addresses, and that means local memory. You see this as a non-zero **`stack frame`**.

Here are the two kernels from `example01.cu`. They compute identical results. The only difference is that one loop bound is a compile-time constant and the other is a kernel argument:

```cpp
    float acc[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) acc[i] = v * c_w[i];      // K is #define'd
```

```cpp
    float acc[K];
    for (int i = 0; i < k; ++i) acc[i] = v * c_w[i];      // k is a kernel argument
```

and here is what ptxas says about them, verbatim:

```
ptxas info    : Compiling entry function '_Z7k_localPKfPfii' for 'sm_89'
ptxas info    : Function properties for _Z7k_localPKfPfii
    64 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 36 registers, used 0 barriers, 64 bytes cumulative stack size, 376 bytes cmem[0]
ptxas info    : Compiling entry function '_Z11k_registersPKfPfi' for 'sm_89'
ptxas info    : Function properties for _Z11k_registersPKfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 372 bytes cmem[0]
```

**64 bytes stack frame** = 16 floats × 4 B. That is the array, in DRAM, per thread. With 1536 threads per SM × 40 SMs, that is 3.9 MB of DRAM traffic the algorithm never asked for.

Note the wording carefully. Modern ptxas prints **`stack frame`** for local-memory allocation and **`spill stores` / `spill loads`** for register spills specifically. Older documentation and older toolkits talk about "lmem". They are the same storage; `stack frame` is the total local footprint, spills are the subset of it caused by register pressure. Both are local memory, both are DRAM, and both are reported by this one flag.

If you want to see it in the instruction stream, `cuobjdump -sass` shows `LDL` / `STL` (load/store local) instead of register operands:

```
$ cuobjdump -sass exercise02_solution.o
        Function : _Z6fir_v0PKfPfii
        /*0330*/   STL.128 [R0], R4 ;
        /*0340*/   STL.128 [R0+0x10], R8 ;
        ...
```

24 `LDL`/`STL` instructions in the slow kernel, zero in the fixed one.

### Shared memory and the unified 128 KB

Ada gives each SM a single 128 KB SRAM block that serves as both L1 data cache and shared memory. Up to **100 KB** of it can be addressed as shared memory; the rest is always L1. A single block may statically request 48 KB by default and up to 99 KB after opting in with `cudaFuncAttributeMaxDynamicSharedMemorySize`.

You can bias the split:

```cpp
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0);   // favour L1
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100); // favour shared
```

It is a **hint**. The driver honours the block's actual static + dynamic shared-memory request first, and uses the carveout only to decide what to do with what is left over.

The consequence people trip over: **L1 is per-SM and is not coherent across SMs.** SM 3 has no idea what SM 17 wrote into its L1. There is no snooping, no invalidation protocol between SMs. The point at which two SMs agree on the contents of memory is **L2**, and getting there requires either finishing the kernel (the implicit device-wide sync at kernel boundaries flushes and invalidates L1) or using operations that explicitly bypass or write through L1. This is why a "global flag" written by one block and polled by another is a bug you cannot fix with `volatile` alone, and why Module 9 has a lot to say about memory ordering.

### L2 is 48 MB, and that changes how you benchmark

48 MB is not a small cache. It is larger than the entire working set of most textbook benchmarks. The practical consequence:

> **If your benchmark buffers fit in 48 MB, you are measuring L2, not DRAM.**

`example02.cu` sweeps a streaming copy across the boundary. Real output:

```
 MB each   workset MB         ms       GB/s    %peak  verdict
       4            8     0.0165      508.8   117.8%  IMPOSSIBLE from DRAM -> L2 hit
       8           16     0.0250      672.0   155.6%  IMPOSSIBLE from DRAM -> L2 hit
      16           32     0.0257     1305.5   302.2%  IMPOSSIBLE from DRAM -> L2 hit
      24           48     0.0473     1064.8   246.5%  IMPOSSIBLE from DRAM -> L2 hit
      32           64     0.2380      281.9    65.3%  real DRAM
      64          128     0.5322      252.2    58.4%  real DRAM
     256          512     1.7763      302.2    70.0%  real DRAM
     512         1024     3.5570      301.9    69.9%  real DRAM
```

1305 GB/s on a GPU whose DRAM interface tops out at 432 GB/s. The number is not wrong; it is simply not a DRAM measurement. The copy touches two buffers, so the working set is 2 × the buffer size, and the cliff appears exactly where 2 × size crosses 48 MB.

**Rule for the rest of this course:** to measure DRAM bandwidth, size your total working set to at least 4× L2 — 192 MB or more on this machine — and say so in your write-up. Every benchmark in Modules 5, 6, 12 and 19 depends on this.

### Global memory: latency, bandwidth, alignment

432 GB/s at a core clock near 1.5 GHz is roughly 290 bytes per cycle for the whole GPU, or about 7 bytes per SM per cycle. Meanwhile a single dependent global load costs ~575 cycles. Those two facts together are the entire reason the GPU is built the way it is: the only way to convert 575-cycle latency into 432 GB/s of delivered bandwidth is to have hundreds of independent requests in flight, which is what 1920 resident warps are for.

`cudaMalloc` guarantees the returned pointer is aligned to at least **256 bytes**. This is not a detail: it means the base of every allocation is aligned to a 128 B cache line and to a 32 B sector, so `((float4*)p)[i]` is always naturally aligned and a warp reading `p[0..31]` always touches exactly the minimum number of transactions. Misalignment in practice comes from *your* offsets (`p + 3`), not from the allocator. Module 5 makes the transaction arithmetic precise.

### Constant memory and the broadcast rule

`__constant__` data lives in a 64 KB device-memory window and is served by a small, dedicated, per-SM **constant cache**. That cache is built around one assumption: at any instruction, all 32 lanes of the warp want the *same* address. When that holds, the cache does a single lookup and broadcasts the value to all 32 lanes — and if the index is a literal, ptxas will not even emit a load, it folds the constant-bank address straight into the FMA as an operand.

When it does not hold, the hardware has no wide path to fall back on. It **replays the instruction once per distinct address**. 32 distinct addresses means up to 32 serialized accesses for one instruction.

The measurement, from `exercise03_solution.cu`, is brutal and worth memorizing:

```
pattern        space                ms    correct
uniform        constant         1.1591        yes
uniform        global-ro        4.3803        yes
lanevarying    constant       100.1714        yes
lanevarying    global-ro        3.5361        yes
```

Same table, same 256 bytes, same amount of arithmetic. Choosing `__constant__` for a lane-varying index made the kernel **28× slower** than an ordinary global array. Choosing it for a uniform index made it **3.8× faster**. The space is not "fast" or "slow"; the *pairing of space and access pattern* is.

The rule: **`__constant__` is for values that are uniform across the warp** — filter coefficients indexed by a loop counter, problem dimensions, transformation matrices, quantization scales. It is never for a lookup table indexed by thread ID.

### The read-only path

Global data that a kernel only reads can be routed through the SM's read-only data cache (historically the texture cache). You request it by promising the compiler two things at once: the data is `const`, and the pointer does not alias any pointer the kernel writes through.

```cpp
__global__ void k(const float* __restrict__ in, float* __restrict__ out, int n);
```

On Ada, `const __restrict__` is normally sufficient — you can confirm it in SASS, where the loads become `LDG.E.CONSTANT` rather than plain `LDG.E`. You saw exactly that in the `fir_v1` disassembly above. `__ldg(ptr)` is the explicit intrinsic form, and is rarely needed on modern architectures because the compiler does this analysis itself. Add `__restrict__` anyway: without it the compiler must assume `in` and `out` may overlap, and that alone blocks the optimization, reordering, and vectorization you want.

**Texture objects** (`cudaTextureObject_t`, `tex2D<>()`) still exist and still work. For compute kernels they are **mostly legacy**: their remaining reasons to exist are hardware-interpolated sampling, normalized coordinates, and free clamp/wrap/mirror addressing modes. If you only want caching, use `const __restrict__`.

### Host memory

A `cudaMemcpy` from a `malloc`'d buffer cannot be a straight DMA. The host OS owns those physical frames and may unmap, relocate or page them out at any moment; a DMA engine programmed with a physical address would then scribble on whatever moved in. The driver's solution is a pinned staging buffer it allocated at init: it `memcpy`s a chunk of your pageable buffer into the staging buffer, DMAs that, and repeats. The extra CPU-side copy is the cost, and the transfer cannot be asynchronous because the driver must do the staging copy on the calling thread.

`cudaMallocHost` (or `cudaHostAlloc` with flags) page-locks the allocation up front, so the DMA engine reads your buffer directly. Measured here:

```
--- Part B: host-to-device copy, 128 MB ---
  pageable (malloc)      :  10.91 GB/s
  pinned   (cudaMallocHost):  12.06 GB/s
  speedup                : 1.11x
```

(Observed 1.04–1.15x across runs; pinned is consistently 12.0–12.2 GB/s, pageable
10.4–11.4 GB/s.)

11% for a plain synchronous copy — smaller than the folklore 2×, because the Windows driver's staging pipeline overlaps the `memcpy` with the DMA well. The *throughput* win is modest; the reason you actually use pinned memory is that it is a **precondition for `cudaMemcpyAsync` to be genuinely asynchronous** and for copy/compute overlap. **Module 26** covers that. Pin sparingly: page-locked pages are removed from the OS's pool and pinning gigabytes will degrade the whole machine.

---

## Code Walkthrough

### `example01.cu` — the tour, and the local-memory demonstration

The file opens by declaring three things in three different spaces:

```cpp
__constant__ float c_w[K];      // 64 B in the constant window
__device__  float  d_scale = 2.0f;   // a global-memory variable, in DRAM
```

`c_w` is filled from the host with `cudaMemcpyToSymbol(c_w, h_w, sizeof(h_w))`. Note that you pass the *symbol*, not a pointer — you cannot take a host pointer to constant storage; the runtime resolves the symbol to its device location. `d_scale` looks like a constant but is not: it is an ordinary DRAM word, and reading it costs a global load.

The two kernels differ in exactly one token, and the program reports the difference twice — once as a compile-time fact and once as a run-time fact. The compile-time fact comes from the driver, without needing to read compiler output:

```cpp
cudaFuncAttributes fa;
cudaFuncGetAttributes(&fa, k_local);
printf("  localSizeBytes : %zu\n", fa.localSizeBytes);
```

```
--- k_registers static attributes ---
  numRegs        : 24
  localSizeBytes : 0
  constSizeBytes : 64
--- k_local static attributes ---
  numRegs        : 36
  localSizeBytes : 64      <-- bytes of DRAM per thread
```

`cudaFuncGetAttributes` is the programmatic form of `-Xptxas -v`, and it works on a shipped binary. Use it in asserts.

The run-time fact:

```
--- 16777216 elements, 16-entry per-thread array ---
  registers :   0.4513 ms     297.4 GB/s  (68.8% of 432)  mismatches=0
  local mem :   1.2395 ms     108.3 GB/s  (25.1% of 432)  mismatches=0
  slowdown  : 2.75x
```

(Observed slowdown ranges 2.5–3.2x across runs; the clocks on a laptop GPU move.)

Read the GB/s column carefully. It counts only the traffic the *algorithm* demands: one float in, one float out. The register version achieves 69% of peak, which is a respectable streaming copy. The local-memory version appears to achieve 25%, not because the DRAM got slower, but because the DRAM is now also carrying 64 bytes per thread of traffic that the algorithm never asked for. The bandwidth was spent; it just was not spent on your data.

The carveout hint is exercised once, on a kernel that uses no shared memory:

```cpp
cudaFuncSetAttribute(k_registers, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
```

and the allocation alignment is printed straight from the pointer:

```
cudaMalloc returned 0000001307600000 and 000000130B600000 (low byte: 00, 00 -> 256 B aligned)
```

### `example02.cu` — L2 residency and host staging

Part A is a `float4` grid-stride copy with a **fixed** grid of 16 blocks/SM, so the only variable across the sweep is the buffer size. Two details are load-bearing.

First, the clock warm-up:

```cpp
for (int i = 0; i < 300; ++i)
    stream_copy<<<p.multiProcessorCount * 16, 256>>>(a, b, wb / sizeof(float4));
```

A laptop GPU idles at a low SM and memory clock. Without this, the first few points of the sweep are measured at the wrong frequency and come out nonsensical — on this machine, the un-warmed first run reported 155 GB/s for a 1 MB buffer and 91 GB/s at 32 MB, which reverses the entire conclusion. Put a warm-up in every GPU benchmark you write, forever.

Second, the working-set accounting:

```cpp
double ws = 2.0 * mb[k];    // one source buffer + one destination buffer
```

The cliff is at `ws = 48 MB`, i.e. 24 MB per buffer, not 48 MB per buffer. Benchmarks that only count the input buffer place the cliff at the wrong size and then conclude the cache is twice as large as it is.

Part B allocates the same 128 MB twice, once pageable and once pinned, warms each path three times, and times 20 `cudaMemcpy` calls of each. The validation is deliberately loose (`pinned > 0.9 × pageable`) because the gap on this platform is around 10% and run-to-run noise is a few percent.

---

## Check Your Understanding

1. A kernel declares `float tmp[8];` and only ever accesses it as `tmp[0]` through `tmp[7]` with literal subscripts, inside a loop that `#pragma unroll` fully unrolls. `-Xptxas -v` still reports `128 bytes stack frame` — 4× the size of the array. Give two distinct mechanisms that could produce this, and say how you would distinguish them from the compiler output alone.

2. You benchmark a kernel on a 20 MB input buffer and measure 700 GB/s, then run the identical kernel on a 400 MB buffer and measure 280 GB/s. A colleague concludes the kernel "does not scale". What is actually happening, what is the kernel's real DRAM bandwidth, and what would you have to change about the *small* run to make its number meaningful?

3. Block A, running on SM 3, writes a value to a `cudaMalloc`'d location. Block B of the *same kernel launch*, running on SM 17, reads that location. Explain why block B may observe a stale value, name the level of the hierarchy at which the two blocks would agree, and explain why marking the pointer `volatile` does not by itself make the program correct.

4. You have a 48 KB read-only table. Thread `t` needs `table[perm[t]]`, where `perm` is a data-dependent permutation, so within a warp the 32 lanes read 32 unrelated entries. The table fits in the 64 KB constant window. Should it go there? Justify your answer with a cost model in terms of instruction replays, and name the space you would use instead.

Answers: `solutions/module04/check_your_understanding.md`.

---

## Exercises

### Exercise 1 — `exercise01.cu` (predict + measure)

Measure the latency of a single dependent load at three depths of the hierarchy with a one-thread pointer chase, and locate the two capacity cliffs.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

Four TODOs:

- **TODO 1** — choose the byte distance between consecutive chain nodes, such that following one link always forces a new request to the memory system. There is a smallest correct answer; going far above it starts measuring something other than the data hierarchy.
- **TODO 2** — supply six probe buffer sizes, two per regime (L1-resident, L2-resident, DRAM), derived from `cudaDevAttrL2CacheSize` and the 128 KB L1 rather than hard-coded. Choose margins that make each classification unambiguous.
- **TODO 3** — inside the kernel, make the timed region report steady-state latency rather than cold-start compulsory misses. The timed region itself must not change.
- **TODO 4** — commit to three predicted latencies, in cycles, before building.

Validation checks that the two probes within each regime agree to 35%, that consecutive regimes differ by at least 1.8×, and that each of your three predictions is within 2× of the measurement. It prints PASS or FAIL.

### Exercise 2 — `exercise02.cu` (debugging / optimization)

A 16-tap normalized FIR filter runs at about a quarter of the throughput its memory traffic implies. The file header gives you the symptom and tells you which compiler flag to look at. It does not tell you the cause.

```
nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise02.exe exercise02.cu
.\exercise02.exe
```

Three TODOs:

- **TODO 1** — write `fir_v1`, computing exactly what `fir_v0` computes, with the per-thread window in the fastest storage the SM has. The header warns you that the obvious one-line fix is not sufficient on its own; verify with `-Xptxas -v` rather than assuming.
- **TODO 2** — `fir_v2`, where the tap count is genuinely a runtime value in {8, 12, 16, 32}, must reach the same machine-level property. You choose the mechanism.
- **TODO 3** — the host-side dispatch that makes TODO 2 work.

`fir_v0` is the timed baseline; do not modify it. Validation requires all versions numerically correct against a CPU reference, `localSizeBytes == 0` for `fir_v1` (queried from the driver, so you cannot fake it), `fir_v1` at least 25% faster than `fir_v0`, and `fir_v2` at 16 taps within 1.5× of `fir_v1`.

### Exercise 3 — `exercise03.cu` (performance reasoning)

A 64-entry, 256-byte read-only coefficient table has to live somewhere. You implement two access patterns over it — one where all 32 lanes of a warp read the same entry, one where all 32 read different entries — and choose a home for the table in each case.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

Four TODOs:

- **TODO 1** — declare the device-side table meeting a stated requirement (the requirement is stated; the feature is not named), and get the host's copy into it.
- **TODO 2** — a compile-time-selected accessor for each of the two candidate spaces.
- **TODO 3** — commit to a choice of space for each access pattern.
- **TODO 4** — commit to a *quantitative* predicted speed ratio for each, before building.

The harness measures all four combinations. Validation requires every combination numerically correct, both of your choices to match the empirically faster space, and both ratio predictions within 3× of the measurement.

---

## Prediction

Commit to these in writing before you run anything.

1. **The cliffs.** For Exercise 1, at what two buffer sizes will the measured latency jump, and by what factor each time? Give six numbers: two sizes and three latencies. You have been told the capacities; you have not been told the latencies.

2. **The spill tax.** In Exercise 2, `fir_v0` moves 64 extra bytes per thread through DRAM that the algorithm never asked for, against 8 bytes of useful traffic per thread. Predict the ratio `time(v0) / time(v1)`. Will it be 9× (the traffic ratio), or less, or more? State which effect dominates and why.

3. **The constant-memory cliff.** In Exercise 3, one of the four combinations will be dramatically worse than the other three. Name it, and predict its slowdown factor relative to the best option for the *same access pattern*. The warp is 32 lanes wide — does your number reflect that?
