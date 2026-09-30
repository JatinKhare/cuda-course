# Module 01 — Why GPUs

> Prerequisites: none beyond strong C/C++ and an undergraduate computer-architecture
> course (pipelining, caches, DRAM, out-of-order execution).
> What this module gives you: a physically accurate model of what a GPU is, what a
> warp is, and exactly what the machine does between the instant you write
> `kernel<<<grid, block>>>(...)` and the instant the last thread retires.

---

## Concept

### 1. Two different optimization targets

A CPU and a GPU are built from the same transistors and are solving the same
problem — *get work done* — but they optimize different objective functions.

**Latency** is the time from the start of one operation to its completion,
measured in seconds (or clock cycles) per operation.
**Throughput** is the number of operations completed per unit time.

These are not reciprocals. A machine can complete a billion operations per second
while each individual operation takes a microsecond, provided a thousand of them
are in flight at once. The relationship is **Little's Law**, which holds for any
stable queueing system:

```
concurrency = throughput x latency
```

Read it three ways and all three are useful:

- *How much parallelism do I need?* `concurrency = throughput x latency`.
- *What throughput can I get?* `throughput = concurrency / latency`.
- *What is my effective latency?* `latency = concurrency / throughput`.

A **latency-minimizing** processor attacks the second term. A CPU core spends
most of its transistor budget on machinery whose only purpose is to shorten the
time-to-answer of a *single* instruction stream: deep out-of-order windows,
register renaming, branch prediction, speculative execution, multi-level
prefetchers, and a large private cache hierarchy. Arithmetic units are a small
fraction of the die. If your program is one dependent chain of operations, this
is the only strategy that works, and a CPU is unbeatable at it.

A **throughput-maximizing** processor attacks the first term. It accepts that a
DRAM access will cost hundreds of nanoseconds and makes no attempt to shorten it.
Instead it spends its transistor budget on (a) arithmetic units, and (b) the
ability to hold an enormous number of independent instruction streams in flight,
so that whenever one stalls another is ready. There is no out-of-order engine, no
branch predictor worth the name, and the per-thread cache is small. The bet is
that you brought enough parallel work to cover the latency.

| | Latency-minimizing (CPU) | Throughput-maximizing (GPU) |
|---|---|---|
| Objective | shortest time for one stream | most streams finished per second |
| Latency hidden by | OOO execution, prefetch, speculation | switching to another warp |
| Threads in flight per core | 1–2 (SMT) | up to 48 warps = 1536 threads per SM |
| Context switch cost | hundreds of cycles (save/restore) | **zero** (contexts never leave the register file) |
| Cache per thread | megabytes | ~100 bytes of L1 |
| Transistors spent on control | most of them | few |
| Fails when | work is embarrassingly parallel | work is one dependent chain |

The quantitative statement of the bet, on this GPU: DRAM latency is roughly
400–500 ns, and peak DRAM bandwidth is 432 GB/s. Little's Law says you must keep
about `432e9 B/s x 450e-9 s ~= 190 KB` of memory requests in flight *at all
times* to run the memory system at its rated speed. That is ~6000 outstanding
32-byte requests, spread over 40 SMs — about 150 per SM. A single thread can have
one outstanding dependent load. You need thousands of threads. This is not a
style preference; it is arithmetic.

### 2. Parallelism, precisely

Three kinds of parallelism matter, and CUDA exposes all three:

- **ILP — instruction-level parallelism.** Independent instructions within one
  thread. A CPU extracts it dynamically with an out-of-order window; a GPU
  depends on the compiler to schedule it statically. Module 20.
- **DLP — data-level parallelism.** The same operation applied to many data
  elements. This is what SIMD units exploit.
- **TLP — thread-level parallelism.** Many independent instruction streams. This
  is the GPU's primary latency-hiding resource.

**PORTABLE CUDA CONCEPT.** A GPU converts TLP into latency tolerance. If you do
not supply TLP (or, failing that, ILP), no amount of hardware helps.

### 3. SIMD and SIMT

**SIMD** (single instruction, multiple data) is what an AVX-512 unit does. One
instruction, one architectural vector register, 16 lanes of `float`. The vector
width is part of the ISA and therefore part of your source code. Conditional
execution requires explicit mask registers. A gather is a distinct, expensive
instruction. You, or your compiler's vectorizer, are responsible for the
transformation from scalar loops to vector code.

**SIMT** (single instruction, multiple threads) is NVIDIA's variant. You write
ordinary scalar code for one thread. The hardware groups threads into fixed
bundles of 32 called **warps**, and issues one instruction per cycle to the whole
warp; all 32 lanes execute that instruction, each on its own data, with its own
private registers and its own view of memory.

The difference that matters:

| | SIMD | SIMT |
|---|---|---|
| Lane state | one shared vector register file | 32 independent per-thread register sets |
| Vector width in source | explicit (`__m512`) | invisible (you write scalar code) |
| Divergent control flow | explicit mask registers | hardware active mask, implicit |
| Per-lane addressing | special gather/scatter instruction | every load is potentially a gather |
| Programming model | vectorize the loop | write one thread, launch a million |

SIMT is a strictly more convenient abstraction. It is also **leaky**, and the two
leaks below are the source of most GPU performance problems. Everything in this
course from Module 5 onward is, in one way or another, about them.

#### Leak 1 — divergence

Nothing in the source of

```cpp
if (x[i] > 0) a(); else b();
```

says that the hardware has only one instruction pointer per warp. When lanes of
one warp disagree on the branch, the hardware cannot issue two different
instructions in the same cycle. It executes `a()` with an **active mask** that
disables the lanes that went the other way, then executes `b()` with the
complementary mask. Both paths are executed serially; the lanes that are masked
off consume issue slots and produce nothing. A warp in which all 32 lanes take
different paths through a 32-way `switch` runs at 1/32 of peak, and the source
code gives no hint of it.

Crucially, divergence is a *warp-local* phenomenon. An `if` on which every lane
of a warp agrees costs nothing — the hardware evaluates the predicate, finds the
mask uniform, and takes one path. "Avoid branches" is folklore; "avoid branches
whose outcome varies *within* a warp" is the real rule. Module 8 makes this
precise, including how Volta-and-later **independent thread scheduling** gives
every thread its own program counter and why that broke a decade of code that
assumed implicit warp-synchronous execution.

#### Leak 2 — uncoalesced access

Nothing in the source of

```cpp
float v = in[idx];
```

says that this is not a per-thread operation. It is a per-*warp* operation. The
memory pipeline collects the 32 addresses generated by the 32 lanes, and issues
the minimum number of **sectors** (32-byte aligned chunks of a cache line) that
cover them.

- If lane `L` requests `in[base + L]` with `base` suitably aligned, the 32 lanes
  cover 128 contiguous bytes = 4 sectors. One request, four sectors, 100% of the
  fetched bytes used.
- If lane `L` requests `in[base + L * 64]`, the 32 lanes land in 32 different
  sectors. Same source line, same instruction count, **8x the memory traffic**,
  and 1/8 of the fetched bytes used.

`example01.cu` in this module deliberately does the second thing, because the
pathological case is the clearest way to measure pure latency. Module 5 treats
coalescing properly.

There are further leaks — warp-level collectives need explicit masks, the shared
memory banking structure is visible, `__syncthreads()` is a block-scope object
with no thread-scope equivalent — and each gets its own module. The point to
carry out of Module 1 is that the SIMT abstraction is a *performance* fiction,
not a correctness fiction: your code will produce the right answer either way.
It will merely be eight times slower, and nothing in the source says so.

---

## Hardware Mental Model

### 4. Anatomy of one Ada SM

The **Streaming Multiprocessor (SM)** is the unit of replication. This GPU has
40 of them. Everything below describes one SM, compute capability 8.9.

```
+---------------------------------------------------------------+
|  SM  (one of 40)                                              |
|  L1 instruction cache                                         |
|  +---------------+ +---------------+ +------------+ +--------+|
|  | proc block 0  | | proc block 1  | | proc blk 2 | | pb 3   ||
|  | warp sched    | | warp sched    | |    ...     | |  ...   ||
|  | dispatch 1/clk| | dispatch 1/clk| |            | |        ||
|  | 16384 x 32b   | | 16384 x 32b   | |            | |        ||
|  |   registers   | |   registers   | |            | |        ||
|  | 32 FP32 lanes | | 32 FP32 lanes | |            | |        ||
|  |  (16 also INT)| |  (16 also INT)| |            | |        ||
|  | 1 Tensor Core | | 1 Tensor Core | |            | |        ||
|  | 4 SFU, 8 LD/ST| | 4 SFU, 8 LD/ST| |            | |        ||
|  | <=12 resident | | <=12 resident | |            | |        ||
|  |     warps     | |     warps     | |            | |        ||
|  +---------------+ +---------------+ +------------+ +--------+|
|  128 KB unified L1 data cache + shared memory  (<=100 KB SMEM)|
+---------------------------------------------------------------+
        |                                                    |
        +------------------ 48 MB L2 (chip-wide) ------------+
                                   |
                        12 GB GDDR6, 192-bit, 432 GB/s
```

Four **processing blocks** (NVIDIA also calls them SM sub-partitions). Each is
very nearly an independent scalar processor:

| Per processing block | Per SM (x4) |
|---|---|
| 1 warp scheduler, issues **at most 1 instruction per clock** | 4 |
| 16,384 32-bit registers (64 KB) | 65,536 registers (256 KB) |
| 32 FP32 lanes, of which 16 also handle INT32 | 128 FP32 lanes / 64 INT32 |
| 1 fourth-generation Tensor Core | 4 |
| 4 SFUs (transcendentals), 8 LD/ST units | 16 SFU, 32 LD/ST |
| up to 12 resident warps | **48 resident warps = 1536 threads** |

Shared across the SM: the 128 KB unified L1-data/shared-memory block (of which at
most 100 KB can be carved out as addressable shared memory on sm_89; the default
static limit is 48 KB per block), the instruction cache, the texture units, and
one RT core.

Two consequences worth internalizing now:

- **A "CUDA core" is one FP32 lane.** This GPU's advertised 5120 CUDA cores is
  exactly `40 SMs x 128 lanes`. It is not 5120 independent processors; it is 160
  independent instruction streams' worth of issue capability (40 SMs x 4
  schedulers), each driving 32 lanes.
- **A warp's registers live in exactly one processing block's register file
  slice.** That is why warps are bound to a processing block for life, and why
  the register file is the resource that most often limits how many warps fit.

### 5. The warp scheduler, cycle by cycle

Each processing block holds up to 12 resident warps. Every clock, its scheduler:

1. Looks at all resident warps and classifies each as **eligible** (its next
   instruction has all operands ready and the required functional unit's issue
   port is free) or **stalled** (waiting on a memory return, a long-latency ALU
   result, a barrier, a texture fetch, an instruction-cache miss, ...).
2. Picks **one** eligible warp by a priority heuristic (roughly: least-recently
   issued, so as to be fair).
3. Issues that warp's next instruction to the 32 lanes of a functional unit.

Dependences are tracked with a **scoreboard**: long-latency results are marked
busy on issue and cleared on write-back, and a warp whose next instruction reads
a busy register is simply not eligible. There is no renaming, no speculation, no
replay — the warp just waits.

The number that matters: **at most 4 warp-instructions per SM per clock**, i.e.
at most 128 lanes' worth of work. With 48 warps resident, 44 of them are doing
nothing in any given cycle. That is not waste — that is the design. Those 44
warps are the inventory from which the scheduler draws whenever the 4 it is
currently issuing go stalled. A resident warp costs registers and a warp slot; it
costs **no issue bandwidth at all** while it waits.

### 6. Why the context switch is free

An operating system thread switch is expensive because the new thread's
architectural state must be loaded into the one physical register file. A GPU
never does this. When a block is placed on an SM, the hardware **allocates each
of its warps a private, disjoint range of the processing block's register file**,
and that allocation is held for the entire lifetime of the warp. Warp 7's
registers are physically different flip-flops from warp 8's registers. Switching
from warp 7 to warp 8 is a change of an index in the operand-fetch path — one
cycle, no save, no restore.

The price is paid at a different time and in a different currency: the register
file is finite (65,536 registers per SM), so the number of registers your kernel
uses per thread directly caps how many warps can be resident. A kernel using 32
registers per thread can host `65536 / (32 x 32) = 64` warps' worth of registers
— above the 48-warp hardware cap, so registers are not binding. A kernel using 64
registers per thread can host only 32 warps, and you have lost a third of your
latency-hiding inventory before writing a line of algorithm. Module 19 turns this
into arithmetic; for now, just note that "zero-cost context switch" and "register
pressure limits occupancy" are the same fact seen from two sides.

### 7. Grids, blocks, warps — and which are hardware

CUDA gives you a three-level hierarchy. Only some of it is physical.

| Level | What it is | Hardware reality |
|---|---|---|
| **grid** | all threads of one kernel launch, up to 3D | a software concept; the work distributor's queue |
| **block** (CTA) | a group of up to 1024 threads, up to 3D | **physical**: the unit of SM assignment; can share memory and synchronize |
| **warp** | 32 threads | **physical**: the unit of instruction issue |
| **thread** | one lane | **physical**: owns registers, has its own PC on sm_70+ |

A **block** (formally a *cooperative thread array*, CTA) is the contract between
you and the scheduler. Its guarantee is: *all threads of a block are resident on
one SM simultaneously, for the whole life of the block.* That guarantee is what
makes shared memory (Module 6) and `__syncthreads()` (Module 9) implementable at
all. Its cost is that blocks must be small enough to fit, and that no similar
guarantee exists at grid level.

Warps are formed by a fixed, documented rule. The block's threads are
**linearized** as

```
tid_linear = threadIdx.x
           + threadIdx.y * blockDim.x
           + threadIdx.z * blockDim.x * blockDim.y
```

and then warp `w` is exactly threads `tid_linear` in `[32w, 32w+32)`. Warp
membership is therefore fully determined by `blockDim` and `threadIdx`, and is
decided before the block starts running. Two corollaries you will use constantly:

- A block whose size is not a multiple of 32 still allocates whole warps. A
  96+1 = 97-thread block occupies 4 warps, and 31 lanes of the last warp are
  permanently inactive — they consume issue slots and registers and do no work.
- `threadIdx.x` is the *fastest-varying* index. If you want adjacent lanes of a
  warp to touch adjacent memory (Module 5), `threadIdx.x` must map to the
  fastest-varying data dimension. Getting this backwards is the single most
  common cause of an 8x slowdown in beginner CUDA code.

---

### 8. What physically happens at `kernel<<<grid, block>>>(args)`

This is the part to remember. Walk it in order.

**Step 1 — the launch is a write to a buffer, not a call.**
The `<<< >>>` syntax compiles (Module 2) into calls that marshal the arguments
and push a *launch command* — grid dimensions, block dimensions, dynamic shared
memory size, the kernel's entry address, the argument buffer — into a
**pushbuffer** (also called a command buffer): a region of memory that the CPU
writes and the GPU's front end reads. The CPU then returns immediately. Nothing
has executed yet. This is why kernel launches are **asynchronous** and why a bug
inside your kernel is reported at the *next* synchronizing call rather than at
the launch. It is also why `clock()`/`std::chrono` around a launch measures the
cost of writing a command packet (a few microseconds) and not the kernel.

**Step 2 — the front end and the GigaThread engine.**
The GPU's front end reads the command, and the **GigaThread work distributor**
takes ownership of the grid. Its job for the whole life of the kernel is: hand
blocks to SMs.

**Step 3 — block placement, gated by four resources.**
The distributor will place block `b` on SM `s` only if SM `s` currently has all
of the following free:

1. a **block slot** (max 24 resident blocks per SM on sm_89),
2. enough **thread/warp slots** (max 1536 threads = 48 warps per SM),
3. enough **registers** — `regsPerThread x threadsPerBlock`, allocated in
   granular chunks out of the 65,536-register file,
4. enough **shared memory** — the block's static plus dynamic shared memory, out
   of the SM's shared-memory carve-out.

Whichever of the four binds first determines how many blocks of *your* kernel fit
on an SM. This number is a property of the kernel, not only of the device, and
CUDA exposes an occupancy API to compute it (you will need it in Exercise 2;
Module 19 derives it by hand).

When the block is placed, the hardware allocates its registers and shared memory,
creates its warps, sets their PCs to the kernel entry point, and marks them
resident. From the next cycle they compete for issue like everyone else.

**Step 4 — blocks are indivisible and never migrate.**
A block is placed on one SM in its entirety or not at all. Once placed it stays
there until every one of its threads has exited. There is no preemption of a
block onto a different SM, no spilling of a half-run block, no work stealing.
(Compute preemption exists for whole contexts at a much coarser granularity; it
is irrelevant to this model.) Consequences:

- Any two threads of a block share an L1, a shared-memory carve-out, and a clock
  domain — hence `__syncthreads()` is cheap.
- Two threads in *different* blocks may be on different SMs, may not be resident
  at the same time at all, and in a multi-wave grid provably are not. Hence there
  is no cheap grid-wide barrier.
- Load imbalance between blocks cannot be repaired by the hardware. A block that
  runs 10x longer than its peers holds an SM hostage.

**Step 5 — the steady state.**
Each SM now holds some number of blocks, i.e. some number of warps, spread over
its 4 processing blocks. Every cycle, each of the 4 schedulers picks one eligible
warp and issues. When a warp issues a global load it goes stalled for hundreds of
cycles; the scheduler simply does not consider it again until the data returns.
Throughput is preserved exactly as long as *some* other warp is eligible.
`example01.cu` measures this directly.

**Step 6 — retirement and refill.**
When the last warp of a block exits, the block's registers, shared memory, warp
slots and block slot are released, and the GigaThread engine immediately places
the next undispatched block of the grid there — if there is one. Blocks are
dispatched roughly in increasing `blockIdx` order, but *completion* order is
arbitrary and you must never rely on it.

### 9. Waves and the tail

Define

```
blocksPerSM    = min over the four resource limits (a property of the kernel)
blocksPerWave  = blocksPerSM x numSMs
waves          = ceil(gridDim.x / blocksPerWave)
```

A **wave** is one full complement of simultaneously-resident blocks. A grid of
`blocksPerWave` blocks fills the machine exactly once. A grid of
`blocksPerWave + 1` blocks does not fill it 2.5% more — it needs a **second
wave**, in which 1 block runs and the rest of the machine idles until it
finishes. When all blocks take similar time, the wall-clock cost is therefore a
**staircase in `waves`, not a ramp in `gridDim`**.

Define **wave efficiency** as `gridDim / (waves x blocksPerWave)`: the fraction
of the machine's resident-block capacity that carries real work, averaged over
the launch. A grid of 41 blocks with a 40-block wave has a wave efficiency of
51.2%, and you should expect to pay for it. The unused half is called the **tail**
or **quantization inefficiency**.

Two practical rules follow, and they survive into every later module:

- **PORTABLE CUDA CONCEPT.** Size grids in whole waves where you can, or make the
  grid large enough (many waves) that one partial wave is amortized. Tail cost as
  a fraction of runtime is roughly `1 / waves`.
- A grid smaller than one wave leaves SMs literally idle. Halving the block count
  from a full wave to half a wave does not halve the time; it halves the
  throughput and leaves the time unchanged.

Exercise 2 measures the staircase.

---

## Code Walkthrough

`example01.cu` — *latency hiding is a concurrency budget.*

The experiment isolates one variable. Every thread performs an identical amount
of work: a chain of 256 **dependent** loads. Dependent means the address of step
`s+1` is the value returned by step `s`, so a single thread can never have more
than one memory request outstanding. All memory-level parallelism must come from
*other warps*. We then vary only one thing — how many warps are resident per SM —
and measure throughput.

The chain:

```cpp
__global__ void chase(const unsigned int* __restrict__ next,
                      unsigned int* __restrict__ out,
                      int steps)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = start_index(tid);
    for (int s = 0; s < steps; ++s)
        idx = next[idx];
    out[tid] = idx;
}
```

There is nothing here for a compiler to help with. It cannot unroll usefully (the
next address is unknown), cannot prefetch (same reason), and cannot vectorize. In
SASS this is 256 iterations of `LDG; ` — load, stall, repeat.

The table is generated from a permutation, so successive addresses are
uncorrelated:

```cpp
// Full-period LCG permutation on Z_2^26:  a = 1 (mod 4), c odd.
unsigned int successor(unsigned int x)
{ return (1103515245ull * x + 12345ull) & MASK; }
```

Because the successor of an address is effectively random, each lane of a warp
touches a different 32-byte sector — the worst case of Leak 2 above. That is
deliberate: it makes each load cost one full sector and one full latency, which
is exactly what we want to count. The table is 32 MB, which fits inside the 48 MB
L2, so this stays a *latency* experiment and does not collide with the DRAM
bandwidth ceiling.

The occupancy dial is the block size:

```cpp
// blockDim = 32 => one warp per block => warpsPerSM == blocksPerSM.
const int warpsPerSM[] = { 1, 2, 3, 4, 5, 6, 8, 12, 16, 24 };
...
const int nBlocks = W * nSMs;
chase<<<nBlocks, 32>>>(d_next, d_out, STEPS);
```

One warp per block is an unusual choice and it is the trick that makes the
experiment clean. Because a block is one warp, "blocks resident per SM" and
"warps resident per SM" are the same number, and because the grid is always
exactly `W x 40` blocks, every configuration is exactly one wave with `W` warps
on every SM. No tail, no imbalance, no dependence on the register file.

Timing is `cudaEvent` based with a warm-up launch and 30 timed iterations, for
the reasons in Step 1 above:

```cpp
chase<<<nBlocks, 32>>>(d_next, d_out, STEPS);   // warm-up, untimed
CHECK(cudaDeviceSynchronize());
CHECK(cudaEventRecord(t0));
for (int it = 0; it < ITERS; ++it)
    chase<<<nBlocks, 32>>>(d_next, d_out, STEPS);
CHECK(cudaEventRecord(t1));
CHECK(cudaEventSynchronize(t1));
```

Correctness is checked by re-walking every thread's chain on the host with the
same `successor` table and comparing the final index — an exact integer match, no
tolerance.

Measured output on the RTX 3500 Ada Laptop GPU:

```
 warps  blocks  threads        loads        ms   Gload/s     GB/s*    %peak  ns/step  vs 1w
------ ------- -------- ------------ --------- --------- --------- -------- -------- ------
     1      40     1280       327680     0.056      5.89     188.5    43.6%   217.33  1.00x
     2      80     2560       655360     0.057     11.41     365.1    84.5%   224.40  1.94x
     3     120     3840       983040     0.058     17.05     545.6   126.3%   225.20  2.90x
     4     160     5120      1310720     0.061     21.49     687.6   159.2%   238.27  3.65x
     5     200     6400      1638400     0.065     25.20     806.3   186.6%   254.00  4.28x
     6     240     7680      1966080     0.072     27.22     871.1   201.6%   282.13  4.62x
     8     320    10240      2621440     0.095     27.62     883.7   204.6%   370.80  4.69x
    12     480    15360      3932160     0.141     27.84     890.9   206.2%   551.73  4.73x
    16     640    20480      5242880     0.185     28.34     907.0   210.0%   722.53  4.81x
    24     960    30720      7864320     0.274     28.70     918.5   212.6%  1070.27  4.87x
```

Read three things out of it.

**(a) The left half of the table is Little's Law, verbatim.** From 1 to 3 warps
per SM the throughput multiplier is 1.00x, 1.94x, 2.90x — dead linear. The
`ms` column barely moves (0.056 -> 0.058) while the number of loads performed
triples. The machine is not going faster; it is doing more things at once for the
same money. Concurrency = throughput x latency, with latency pinned at ~220 ns.

**(b) Latency never improves.** The `ns/step` column is the round-trip time of one
dependent load as seen by one thread. It starts at 217 ns and only ever gets
*worse* (1070 ns at 24 warps/SM, where requests queue behind each other). A GPU
never makes an individual operation faster. If your problem is one dependent
chain, a GPU is the wrong machine and this column is the proof.

**(c) There is a knee, and it is early.** Throughput saturates around 6 warps per
SM at ~4.7x, and the remaining 42 warp slots buy 4%. The reason is that this
access pattern is extraordinarily greedy: each warp instruction asks for 32
separate sectors. Six such warps per SM already put ~190 sectors in flight per SM
— close to the ~150 that Little's Law said the memory system could absorb. A
kernel whose warps ask for less per instruction (a coalesced load asks for 4
sectors, not 32) needs correspondingly *more* warps to reach saturation. "How
much occupancy do I need?" has no universal answer; it depends on how many bytes
each warp has in flight. Module 19 and Module 20 pursue this.

(The `%peak` column exceeds 100% legitimately: the 32 MB table is L2-resident and
L2 bandwidth is several times the 432 GB/s of DRAM. The DRAM interface is almost
idle during this test.)

---

## Check Your Understanding

Answer these before reading `solutions/module01/check_your_understanding.md`.

1. Your GPU can keep 1536 threads resident per SM, but each SM has only 4 warp
   schedulers issuing at most one instruction each per cycle. So at most 128
   threads make progress per SM per cycle. What is the point of keeping the other
   1408 thread contexts resident? Be precise about what resource they consume
   while idle.

2. A CPU hides a DRAM miss with out-of-order execution and prefetching. A GPU
   hides it differently. Name the mechanism, and state the one thing the
   programmer must supply for it to work.

3. `__syncthreads()` synchronizes a block but there is no equally cheap
   `__syncgrid()`. Explain why, using the block-scheduling model above.

4. A kernel launches 41 blocks of 256 threads on this 40-SM GPU. Assume each SM
   could hold several such blocks. How many waves execute, and roughly what
   fraction of the GPU's issue capacity is wasted? State your assumption about
   block-to-SM assignment.

---

## Exercises

### Exercise 1 — `exercise01.cu` : hardware census + block-to-SM residency

**Type:** fill in the code.

**What the program must accomplish.** Part A turns the numbers the driver reports
into the capacities that actually constrain you: the theoretical peak global
memory bandwidth derived from the memory clock and bus width, and the residency
figures (resident warps per SM, resident warps per GPU, total threads in flight)
derived from the device properties. Part B launches a kernel in which one thread
per block records the hardware SM identifier it ran on, read through inline PTX,
and prints a histogram of blocks per SM.

```
nvcc -arch=sm_89 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

**TODOs.** Four of them: (1) the bandwidth formula — there is one factor most
people get wrong, and the error is exactly 2x in a known direction; (2) the three
residency numbers, derived from queried fields and never hard-coded; (3) a launch
configuration of exactly two blocks per SM at 128 threads per block, computed from
the device properties; (4) the store in `report_sm`, plus a written justification
of whether the SM identifier is uniform across a block.

**Validation.** The program cross-checks your residency numbers for internal
consistency (`warps/SM x 32 == threads/SM`, `threadsInFlight == warps/GPU x 32`)
and prints `[OK]` or `[FAIL]`. The histogram must account for every block, with
zero unwritten entries.

### Exercise 2 — `exercise02.cu` : the staircase — what a wave tail costs

**Type:** predict the behavior + performance reasoning.

**What the program must accomplish.** It runs a deliberately compute-bound kernel
(a long chain of dependent FMAs, negligible memory traffic) at six grid sizes
expressed relative to one wave: half a wave, one wave minus one block, exactly one
wave, one wave plus one block, exactly two waves, two waves plus one block. Every
block does identical work, so block count is proportional to total work. The
program reports, per configuration: elapsed ms, achieved GFLOP/s, percent of peak
at the SM clock it actually measured, ms per wave, and the ratio to the
exactly-one-wave time.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

**Before you build**, commit to answers for the five predictions in the file
header (P1–P5). They ask for the ratios `t(wave+1)/t(wave)`, `t(half)/t(wave)`,
`t(2 waves)/t(wave+1)`, which configuration achieves the best and worst GFLOP/s,
and what would change if the kernel were memory-bandwidth-bound instead. Write
them down; the program prints the truth.

**TODOs.** Three:

1. `blocksPerSM` and `waveBlocks` — how many blocks of this kernel form exactly
   one wave on this device. Note the phrasing: *of this kernel*. The device's
   unconditional `maxBlocksPerMultiProcessor` is not the answer; you need the
   number that accounts for this block's thread, register and shared-memory
   footprint. Find the runtime call that computes it.
2. `waves` and `waveEff` — the number of waves the grid takes, and the fraction
   of the machine's resident block slots that carry real work, averaged over the
   launch. An off-by-one is rejected by the self-check.
3. `gflops` — achieved floating-point throughput for the configuration. Count
   only work the machine was actually asked to perform, and be careful about what
   a single fused multiply-add is worth.

**Validation.** Correctness is checked independently of your TODOs: every output
element is compared against a host-side replay of the same FMA chain (sampled)
and every element is checked to have been written at all (the buffer is zeroed
first and a valid result is never exactly zero). The TODOs are checked by
invariants: `waves` must be the smallest integer with
`waves x waveBlocks >= gridDim`, and `waveEff` must lie in `(0, 1]`. The program
prints `PASS` only if all of that holds.

The file times everything first and validates afterwards, and takes the fastest of
four sweeps. That is not paranoia: on a laptop GPU the SM clock swings by nearly
3x between the idle state and the sustained thermal state, and host-side work
between two GPU measurements is enough to invalidate the comparison. The
measured-clock readout in the output shows you how far the nominal clock is from
reality.

---

## Prediction

Commit to these in writing before you run anything in this module.

1. **The knee.** In `example01.cu`, at how many resident warps per SM will
   dependent-load throughput stop improving? Give a number, and give the
   reasoning (Little's Law, sectors in flight) that produced it. Then also
   predict what happens to the *per-step latency* as you add warps: down, flat,
   or up?

2. **The staircase.** In `exercise02.cu`, the kernel is compute-bound and every
   block does identical work. Work out `blocksPerWave` for a 1024-thread block on
   this 40-SM GPU, then predict `t(waveBlocks + 1) / t(waveBlocks)` and
   `t(waveBlocks / 2) / t(waveBlocks)`. One of these two ratios surprises almost
   everyone; decide in advance which.

3. **Grid sizing.** You have a perfectly parallel problem of 10 million elements
   and a kernel that fits 6 blocks of 256 threads per SM. State how many blocks
   you would launch and why, in terms of waves — before Module 3 teaches you the
   grid-stride idiom that makes the question interesting.
