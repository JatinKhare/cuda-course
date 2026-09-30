# Module 5 — Global Memory and Coalescing

> Prerequisites: Module 1 (warps, SMs), Module 2 (launch, error checking),
> Module 3 (thread indexing), Module 4 (memory hierarchy)
> What this module gives you: the ability to compute, on paper, exactly how
> many bytes DRAM must move to service a given warp's load — and therefore
> to predict a kernel's bandwidth before you write it.

Every optimization in the rest of this course is an argument about memory
transactions. Module 12's reduction ladder, Module 15's transpose, Module 17's
tiled GEMM, Module 42's attention kernels — each one is, underneath, a
rearrangement of which thread touches which address. If you cannot count
sectors, those modules degrade into memorized recipes. This module is the
counting.

---

## Concept

### The unit of memory traffic is not the byte

A CUDA thread loading a `float` looks, in source, like it asks for four bytes.
It does not. The memory system has no mechanism for moving four bytes.

**PORTABLE CUDA CONCEPT.** Global memory is accessed in fixed-size aligned
blocks. A warp's memory instruction is *coalesced* by the hardware into the
minimum set of those blocks that covers the addresses the warp's active lanes
supply. Every byte inside a fetched block travels whether or not anyone wanted
it. Therefore: **move contiguous data with contiguous threads.**

**ARCHITECTURE-SPECIFIC (sm_89 / Ada, and in fact sm_50 onward).** The block
is a **32-byte sector**, naturally aligned: sector *k* covers bytes
`[32k, 32k+32)`. The L1 and L2 cache line is **128 B = 4 sectors**, but the
line is the *tag* granularity while the sector is the *fill* granularity — a
miss on one sector of a line fetches that sector, not the whole line. On this
GPU, "how much did DRAM move?" is answered in units of 32.

Three qualifications you must internalise now, because they are the source of
almost every wrong prediction:

1. **Coalescing is per warp, per instruction.** Not per thread: a single
   thread's load is never "coalesced" or "uncoalesced" in isolation. Not per
   block: warp 0 and warp 1 of the same block are serviced independently.
   The unit of analysis is *32 lanes executing one memory instruction*.
2. **The hardware sees a set of addresses, not a sequence.** Lane order is
   irrelevant. If lanes 0..31 touch the same 32 addresses in reverse, the
   sector count is identical.
3. **Inactive lanes contribute nothing.** A lane predicated off by a branch or
   a bounds check supplies no address, so it cannot force an extra sector.

### The counting procedure

For one warp executing one memory instruction:

1. Write down the 32 byte addresses `a_0 .. a_31` the active lanes supply.
2. Map each to its sector id: `s_i = a_i / 32` (i.e. `a_i >> 5`). If a lane
   loads more than 4 B (e.g. a 16 B `float4`), it may span two sectors — take
   every sector from `a_i/32` to `(a_i + size - 1)/32`.
3. Count the **distinct** sector ids. Call it `S`.
4. `bytes_moved = 32 * S`.
5. `efficiency = bytes_requested / bytes_moved`.

That is the whole model. Everything below is that procedure applied.

### Worked cases

Assume `float* a` from `cudaMalloc`, so `(uintptr_t)a % 256 == 0`, and let
`t` be the global thread index with lane `L = t % 32` in warp 0.

| Access | Byte offsets touched | Distinct sectors | Moved | Efficiency |
|---|---|---|---|---|
| `a[t]` | 0,4,…,124 | 4 (`0,1,2,3`) | 128 B | **100 %** |
| `a[t+1]` | 4,8,…,128 | 5 (`0..4`) | 160 B | **80 %** |
| `a[t+8]` | 32,…,156 | 4 (`1..4`) | 128 B | **100 %** |
| `a[2*t]` | 0,8,…,248 | 8 | 256 B | **50 %** |
| `a[6*t]` (AoS `.x`) | 0,24,…,744 | 24 | 768 B | **16.7 %** |
| `a[32*t]` | 0,128,…,3968 | 32 | 1024 B | **12.5 %** |
| `a[7]`, all lanes | 28 only | 1 | 32 B | 4 B asked, 32 B moved |
| `a[31-L]` | 0,4,…,124 | 4 | 128 B | **100 %** |

Read the last two rows carefully.

- **Broadcast.** All 32 lanes reading the same address is *one* sector. The
  hardware broadcasts the value to all lanes; there is no serialization and no
  replay. The efficiency number (4 B wanted / 32 B moved = 12.5 %) is
  misleading as a performance figure: 32 B is the cheapest possible request,
  and the value will be in L1 for every subsequent warp. Uniform access is
  *good*; the efficiency metric just does not model reuse.
- **Reversed.** Same sector count as contiguous, because the coalescer sees
  `{0,4,...,124}` either way. This is the crispest demonstration that order
  does not matter — only the address **set**.

Note also `a[t+8]`: offsetting by 8 floats (32 B) restores 100 %. It is not
*contiguity from the base pointer* that matters, it is whether the warp's
128-byte window is 32 B-aligned.

### Alignment

`cudaMalloc` returns pointers aligned to at least **256 B**. This is a
guarantee you may rely on: the base of any `cudaMalloc` allocation starts a
sector and a line.

Alignment is destroyed by arithmetic, not by allocation. `d + 1` is 4 B into a
sector; `d + 4` is 16 B in; `d + 8` starts a fresh sector. This matters in two
distinct ways that are easy to conflate:

| Offset from a 256 B-aligned base | 16 B aligned? | 32 B aligned? | Consequence |
|---|---|---|---|
| `+1 float` (4 B) | no | no | `float4` access is **illegal**; scalar access costs an extra sector |
| `+4 floats` (16 B) | yes | no | `float4` legal; scalar access still costs an extra sector |
| `+8 floats` (32 B) | yes | yes | fully aligned |

A 128-bit (`float4`/`int4`) access **requires** a 16 B-aligned address. Violate
it and the kernel does not silently slow down — it raises
`cudaErrorMisalignedAddress`, surfacing at the next `cudaDeviceSynchronize()`.
This is a hardware requirement of the `LDG.E.128` instruction, not a software
check.

The 32 B question is separate and is about *cost*, not legality.

### Vectorized loads

```cpp
float4* a4 = reinterpret_cast<float4*>(a);
float4  v  = a4[i];          // one LDG.E.128
```

A warp of 32 lanes each loading a `float4` covers 512 contiguous bytes = 16
sectors, for 512 requested bytes: still 100 %. **The sector count per byte is
unchanged.** So what does vectorizing buy?

- **Instruction count.** One `LDG.E.128` replaces four `LDG.E.32`. Fewer
  instructions issued means fewer cycles spent in the memory pipeline's issue
  stage, which matters once you are close to the bandwidth ceiling and the
  kernel becomes issue-limited rather than DRAM-limited.
- **Outstanding-request slots.** Each SM can track a bounded number of
  in-flight memory requests. One 128-bit request occupies one slot and returns
  four times the data of a 32-bit one, so the same number of slots covers four
  times as much latency. This is the real reason vectorizing helps a
  latency-bound kernel.
- **Register pressure and address math.** One address computation instead of
  four.

Constraints, all of which the exercises make you handle:

1. the pointer must be 16 B aligned;
2. the element count is divided by the vector width, and **the remainder must
   be handled** — `N = 16,000,003` leaves a tail of 3;
3. the *stride between what consecutive threads load* is now 16 B, so your
   index arithmetic changes shape.

### AoS vs SoA — and when AoS is fine

The canonical example:

```cpp
struct Particle { float x, y, z, vx, vy, vz; };   // 24 B
Particle p[N];
```

A warp reading `p[i].x` supplies addresses 24 B apart, spanning 768 B, landing
in **24 distinct sectors** for 128 useful bytes: **16.7 %**. Split into six
arrays `float x[N], y[N], ...` and the same read is **4 sectors, 100 %** — a
6× reduction in DRAM traffic.

But the folklore "AoS bad, SoA good" is wrong as stated, and `example02.cu`
measures the correction. A kernel that reads *every* field of every particle
also covers those same 768 B / 24 sectors — and now uses all of them. The DRAM
traffic is identical to SoA. Measured on this GPU:

| Workload | AoS | SoA | ratio |
|---|---|---|---|
| touch one field (`x *= 2`) | 57.0 GB/s | 374.0 GB/s | **6.6×** |
| touch every field (`pos += vel*dt`) | 280.1 GB/s | 379.2 GB/s | **1.35×** |

**The rule is not about structs. It is about the fraction of each fetched
sector that the warp consumes.** AoS costs you exactly when a kernel touches a
minority of each record. That is also why the residual 1.35× exists at all:
even when every byte is used, AoS needs six separate instructions to gather
what SoA gets in one contiguous run per stream, and partial-sector writes cost
extra (below).

### Writes

Stores coalesce by the same rule: the warp's store addresses are reduced to the
minimum covering set of sectors. There is one asymmetry.

If a warp writes only *part* of a sector, the memory system cannot simply push
32 B to DRAM — DRAM has no byte enables at that level of the hierarchy. It must
first **read** the sector, merge the new bytes, and write it back. A "write
only" kernel with a scattered or misaligned store pattern therefore generates
read traffic that appears nowhere in your source. Full-sector writes avoid the
fill entirely.

Consequence: a strided *write* is typically worse than a strided *read* of the
same pattern, because it costs a read plus a write instead of a read. When you
have to choose which side of a kernel to leave uncoalesced, leave the reads.

### Measuring it honestly

Two numbers, and you must never confuse them.

- **Effective bandwidth** = *useful* bytes / time. Useful means bytes your
  algorithm needed. This is the number that tells you how fast your program is.
- **DRAM bandwidth** = bytes actually moved / time. This is the number that
  tells you whether the bus is saturated.

They are related by the efficiency you just computed:
`dram = effective / efficiency`.

From `example01.cu` on this GPU (256 MB buffer, read-modify-write, one float
per thread):

| Pattern | effective GB/s | model efficiency | implied DRAM GB/s |
|---|---|---|---|
| contiguous | 342.3 | 100 % | 342 |
| stride 2 | 188.8 | 50 % | 378 |
| stride 4 | 94.4 | 25 % | 378 |
| stride 6 | 62.7 | 16.7 % | 376 |
| stride 8 | 46.9 | 12.5 % | 375 |

The right-hand column is the punchline. The stride-8 kernel looks catastrophic
at 46.9 GB/s effective — 11 % of the 432 GB/s peak — while the DRAM bus is
running at essentially the same rate as the perfectly coalesced kernel. It is
not idle. It is saturated moving bytes nobody asked for. You cannot fix that
kernel by adding more warps or more ILP; the only lever is to stop requesting
sectors you will not consume.

**Methodology warnings.**

1. **Your buffer must be far larger than L2.** This GPU has a **48 MB** L2 —
   unusually large, and large enough to hide a benchmarking mistake completely.
   At `N = 8M` floats, a single SoA field is 32 MB, fits in L2, and the kernel
   reports **1360 GB/s** — 315 % of DRAM peak. That number is real; it is just
   not a DRAM measurement. Size every buffer at 4× L2 or more.
2. **Warm up by duration, not by iteration count.** A laptop GPU idles in a
   reduced memory P-state (6001 MHz here) and ramps to 8001 and then 9001 MHz
   under sustained load. Peak bandwidth at those states is 288 / 384 / 432
   GB/s. A cold first measurement reads up to 3× slow, and if that first
   measurement is your baseline, every ratio you compute afterwards is wrong.
   Check with:
   `nvidia-smi --query-gpu=clocks.mem,clocks_throttle_reasons.active --format=csv`
   (`0x4` in the throttle field means a software power cap — common on laptops
   after ~30 s of streaming load).
3. **Prefer ratios and "% of a measured streaming ceiling" over "% of nominal
   peak."** `exercise02.cu` measures a plain 1-read-1-write stream in the same
   loop as the versions under test, precisely so the ranking survives clock
   drift.
4. Use `cudaEvent_t` around the launches, with ≥ 20 timed iterations. Never
   `chrono` around an async launch.

### Two short items

**`const __restrict__` and `__ldg`.** Marking a pointer
`const T* __restrict__` tells the compiler the data is read-only for the
kernel's lifetime and does not alias the outputs. On sm_35+ this lets it emit
`LDG` (load through the read-only/texture path) and, more importantly, hoist
loads above stores. `__ldg(ptr)` is the explicit spelling of the same thing and
is essentially never needed in modern code — write `const __restrict__` and let
the compiler do it. Neither changes the sector count; they change scheduling.

**L1 is not coherent across SMs.** Each SM has its own L1. A write by one SM is
not guaranteed visible in another SM's L1 without going through L2. This is why
inter-block communication needs Module 9's memory fences and Module 10's
atomics, and why you cannot use L1 to pass data between blocks. Within a block
it is a non-issue, because a block lives on exactly one SM.

---

## Hardware Mental Model

Why 32 bytes, and why per warp?

**The warp is the issue unit, so it is also the coalescing unit.** Module 1
established that an SM's warp scheduler issues one instruction for all 32 lanes
at once. When that instruction is `LDG`, the Load/Store Unit receives 32
addresses in the same cycle group. Having them all simultaneously is exactly
what makes merging possible: the address coalescer is a piece of combinational
logic that takes 32 addresses and emits the distinct sector requests. It has no
buffer in which to accumulate addresses across instructions, which is why warp
1's request can never be merged with warp 0's, even if they are adjacent. What
*does* rescue adjacency across warps is the cache, one level up — and that is a
different mechanism with different (capacity-limited) behaviour.

**Why a sector rather than a byte?** Every request carried through the memory
hierarchy needs a tag, a crossbar routing decision, an entry in a
miss-status-holding register, and a DRAM burst. The fixed overhead per request
is large and nearly independent of the payload; GDDR6 on a 32-bit channel
delivers a minimum burst of exactly 32 B. Making the minimum transfer 32 B
amortises that overhead. Making it larger (say 128 B, as on very old GPUs)
would waste more on scattered access. 32 B is the compromise, and it is why a
perfectly coalesced warp needs exactly 4 requests, not 1 and not 32.

**Why 128 B lines on top of 32 B sectors?** The line is the unit of tag storage
and replacement; the sector is the unit of fill and of DRAM traffic. Tagging at
128 B keeps the tag array small (one tag for four sectors), while sectoring
lets a miss fetch only the 32 B actually needed. A strided kernel therefore
pollutes the cache in 128 B units but only *pays DRAM* in 32 B units — which is
why the stride-8 and stride-32 rows of the table above are both 12.5 % and not
12.5 % and 3.1 %.

**Where the "uncoalesced" cost actually lands.** A warp whose addresses span 32
sectors does not execute 32 instructions. It executes one instruction that
produces 32 sector requests. Those requests queue in the LSU and the L1-to-L2
interface. The warp stalls until all 32 return. Nsight Compute calls this
`stall_long_scoreboard` and reports
`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` — the sector count you just
computed by hand. Module 23 will have you read exactly that metric; you should
already be able to predict it.

**Why misalignment is usually cheaper than the model says.** In the table
above, `a[t+1]` is predicted at 80 % but measures at 97.9 % of contiguous. The
model is not wrong about the *request* count — the warp really does issue 5
sector requests. But warp *k*'s fifth sector is warp *k+1*'s first sector. In a
streaming kernel the neighbour is running concurrently on the same or a nearby
SM, and the second request hits in L1 or L2. DRAM moves the same bytes; only
the on-chip request count went up. Misalignment becomes genuinely expensive
when there is no neighbour to share the boundary sector with — that is, when
the contiguous runs are short and separated by gaps. Exercise 3 constructs
exactly that case, and there the penalty is real and measurable.

**Why AoS sometimes costs nothing.** Same mechanism. A warp reading all six
fields of 32 particles issues six instructions, each covering the same 24
sectors. The first instruction misses; the other five hit in L1. Total DRAM
traffic: 768 B per warp, every byte used. The *sector-request* count is
6 × 24 = 144 where SoA needs 6 × 4 = 24, and that request-count difference is
what the residual 1.35× gap measures. The sector model predicts DRAM traffic;
it does not predict L1 request pressure. Knowing which of the two is binding is
the difference between a guess and a diagnosis.

---

## Code Walkthrough

### `example01.cu` — the model and the measurement, side by side

The file has two halves that compute the same quantity two ways.

**Part A** implements the counting procedure on the host, using the *real*
device pointer so the alignment is genuine:

```cpp
static int sector_count(uintptr_t base, IndexFn f, long long param,
                        int elemBytes, uintptr_t* outSectors, int maxOut)
{
    uintptr_t seen[256];
    int nSeen = 0;
    for (int lane = 0; lane < 32; ++lane) {
        uintptr_t a0 = base + (uintptr_t)(f(lane, param) * elemBytes);
        for (uintptr_t a = a0; a < a0 + (uintptr_t)elemBytes; a += SECTOR) {
            uintptr_t s = a / SECTOR;
            ...
```

Three things to notice. It iterates over lanes but stores into an
order-independent `seen` set — the code structurally cannot produce a different
answer for a reversed pattern. It walks from `a0` to `a0 + elemBytes` so that a
16 B `float4` correctly claims both sectors it straddles. And `base` is the
actual `cudaMalloc` result, so if the allocator ever returned something less
aligned, the printed numbers would change.

The pattern functions are the entire vocabulary of this module:

```cpp
static long long ix_contig   (int lane, long long p) { return lane; }
static long long ix_stride   (int lane, long long p) { return (long long)lane * p; }
static long long ix_broadcast(int lane, long long p) { return p; }
static long long ix_reversed (int lane, long long p) { return 31 - lane; }
```

`ix_reversed` exists purely to make the point that the answer is a property of
the set. Running the file prints 4 sectors for both `ix_contig` and
`ix_reversed`.

**Part B** runs each pattern as a kernel. All of them do the identical amount
of arithmetic and touch the identical number of *useful* bytes per thread:

```cpp
__global__ void k_stride(float* a, long long nThreads, int stride)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < nThreads) { long long i = t * stride; a[i] = a[i] * 2.0f + 1.0f; }
}
```

The read-modify-write shape is deliberate: read and write share the same
address pattern, so one efficiency number describes both halves and we are not
averaging two different behaviours.

The warm-up is not decoration:

```cpp
for (int i = 0; i < 500; ++i)
    k_contig<<<(int)((N + TPB - 1) / TPB), TPB>>>(d, N);
CHECK(cudaDeviceSynchronize());
```

Without it, the first pattern measured is 2–3× slow and every ratio in the
table is garbage. Delete it and re-run if you want to see how badly.

The output's last column, `impliedDRAM = effective / model`, is the file's
reason for existing. When it reads ~376 for every strided case *and* for the
contiguous case, you have proven that the bus is equally busy in all of them
and the only difference is how much of that traffic was wanted.

One row deserves suspicion: `offset +1 float` reports an implied DRAM rate of
**420 GB/s**, against a contiguous measurement of 342. A number above what the
contiguous kernel achieves is a signal that the model is over-counting, and the
explanation is the cross-warp sector sharing described above. Take the habit:
when a derived quantity exceeds a physical ceiling, the model is wrong, not the
hardware.

### `example02.cu` — AoS, SoA, and the honest comparison

Part 1 touches one field:

```cpp
__global__ void scale_x_aos(Particle* p, long long n)
{   ... if (i < n) p[i].x *= 2.0f; }        // stride 24 B -> 24 sectors/warp

__global__ void scale_x_soa(float* __restrict__ x, long long n)
{   ... if (i < n) x[i] *= 2.0f; }          // stride  4 B ->  4 sectors/warp
```

Measured: 57.0 vs 374.0 GB/s. The ratio 6.6 is close to the 6.0 the sector
model predicts (24/4), the extra coming from the partial-sector writes the AoS
version forces.

Part 2 touches every field, and the gap collapses to 1.35×. The file is
structured so you read both tables together; a reader who sees only one of them
learns a rule that is half wrong.

The vectorized kernel shows the mechanical shape of a 128-bit access:

```cpp
float4* x4 = reinterpret_cast<float4*>(x);
float4 a = x4[i], b = vx4[i];
a.x += b.x*dt; a.y += b.y*dt; a.z += b.z*dt; a.w += b.w*dt;
x4[i] = a;
```

`N` is a multiple of 4 here so no tail is needed — a convenience Exercise 2
removes.

Finally, note the sizing comment in the header. At `N = 8M` a single SoA field
is 32 MB, it fits in the 48 MB L2, and `scale_x_soa` reports 1360 GB/s. The
file uses `N = 24M` so that one field is 96 MB and the measurement is real.

---

## Check Your Understanding

Answers in `solutions/module05/check_your_understanding.md`. Reason them out
before looking; none of these can be looked up.

1. Warp A's 32 lanes read `a[0], a[1], ..., a[31]`. Warp B's 32 lanes read the
   same 32 elements but in reverse lane order. Warp C's lanes all read `a[0]`.
   Rank A, B and C by the number of 32 B sectors requested, and separately by
   the time you would expect each to take. Explain any place where the two
   rankings disagree, and say what that disagreement tells you about the
   efficiency metric.

2. A kernel does `out[i] = in[i]` over 64M floats and achieves 340 GB/s
   effective. You change it to `out[i] = in[i + 1]` — a misalignment of 4 B on
   the read side only. The sector model says the read is now 80 % efficient,
   predicting ~300 GB/s. You measure 336. Give the mechanism. Then describe,
   concretely, a kernel whose access pattern is misaligned by the same 4 B but
   for which the 80 % prediction *would* be accurate, and explain what is
   structurally different about it.

3. Two candidate layouts for 10⁷ records of 8 floats each. Layout X is AoS.
   Layout Y is SoA. Kernel P reads field 0 of every record. Kernel Q reads all
   8 fields of every record. Kernel R reads fields 0 and 1 of every record.
   For each of P, Q, R, state which layout moves fewer bytes through DRAM and
   by what factor, computing the sector counts. Then: for the case where the
   two layouts move the *same* number of bytes, explain why one is nonetheless
   measurably faster.

4. You are told a kernel writes (and never reads) one `float` per thread, with
   threads writing to addresses 64 B apart. A colleague argues the kernel
   cannot generate any DRAM read traffic, since there are no loads in the
   source. Explain why they are wrong, quantify the DRAM traffic per warp, and
   describe the change to the *data layout* (not the code) that would eliminate
   the read traffic.

---

## Exercises

### Exercise 1 — `exercise01.cu` (predict-the-behavior + address calculation)

Compute the per-warp address footprint of seven access patterns on paper, then
implement the counting procedure so the program checks your arithmetic, then
predict the relative bandwidths and let a timed run judge you.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

- **TODO 1** — map a byte address to its 32 B sector id.
- **TODO 2** — count the distinct sectors warp 0's 32 addresses fall into. Your
  implementation must be insensitive to lane order; if it is not, it is not a
  model of the coalescer.
- **TODO 3** — the efficiency formula.
- **TODO 4** — the smallest strictly positive element offset `k` for which a
  `float4` access at `d + k` is legal. Guess wrong and the program reports a
  specific CUDA error rather than bad numbers. Read the error name.

Validation: sector counts must lie in [1,32] and be arithmetically consistent
with your efficiency; the kernel's numerics are checked against a CPU
reference; and a final table compares your model's bandwidth ratios against the
measured ones. **One pattern will disagree by more than 15 %.** Identifying
which, and explaining why, is the exercise.

### Exercise 2 — `exercise02.cu` (optimization: AoS → SoA → vectorized)

A correct AoS particle integrator is given. Produce an SoA version and then a
128-bit version. `N = 16,000,003`, so `N % 4 == 3`.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

- **TODO 1** — the layout transformation. State its cost in bytes before you
  write it, and how many timesteps must run before it repays.
- **TODO 2** — the SoA kernel, written so that every one of its six memory
  instructions spans exactly 128 contiguous bytes per warp.
- **TODO 3** — `nVec`, and the `float4` fast path. Be ready to justify why the
  `reinterpret_cast` is legal.
- **TODO 4** — the tail. Three elements must be updated exactly once each, by
  the same launch. Placement matters; the validator reports tail mismatches
  separately from the rest so a tail bug is unmistakable.

Validation: all three versions are checked field-by-field against a CPU
reference, and the harness reports each version as a percentage of a streaming
ceiling it measures in the same loop.

### Exercise 3 — `exercise03.cu` (design: fix the layout, not the loop)

A kernel processes columns 0..31 of every row of a 1,000,000 × 65-float record
array, one warp per row, lane *L* on column *L*. The indexing is already the
obvious contiguous one. It is still slow. You may not use shared memory,
atomics or warp intrinsics, and you may not change what the kernel computes.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

- **TODO 1** — sectors touched by the warp handling row *r*, as a function of
  pitch. Derive it from the first and last byte; do not special-case.
- **TODO 2** — the smallest pitch ≥ 65 giving 16 B-, 32 B- and 128 B-aligned
  row starts. Three different numbers. The one most people quote is not the one
  that fixes this.
- **TODO 3** — the kernel.
- **TODO 4** — the pitch you would ship, given that memory is not free.
  Two of your three thresholds have identical predicted efficiency; decide
  between them on reasoning first, then see whether the measurement agrees, and
  if it does not, work out what the sector model omits.

Validation: the thresholds are checked for legality and alignment, the
transformed columns are checked against a CPU reference for a sampled 100,000
rows, and predicted/measured ratios are printed for all five pitches.

---

## Prediction

Commit to these in writing before you run anything.

1. **`example01.cu`, Part B.** The stride-2, stride-4 and stride-8 kernels each
   do a quarter, an eighth and a sixteenth of the work of the contiguous one
   (fewer threads). Will their *wall-clock times* be smaller, larger, or
   roughly equal to the contiguous kernel's? Give a number for the ratio of
   stride-8's time to contiguous's time, and justify it from the sector model
   rather than from the thread count.

2. **`exercise01.cu`.** Exactly one of the seven patterns will have a measured
   bandwidth ratio that differs from the sector model's prediction by more than
   15 %. Name it before you run, and say whether the measurement will be
   *better* or *worse* than the model predicts.

3. **`exercise03.cu`.** Two of the pitches you compute in TODO 2 have the same
   predicted efficiency (100 %). Predict whether they will measure the same, and
   if not, which will be faster and by roughly what margin. Name the physical
   mechanism you think is responsible.
