# Module 07 — Shared Memory Bank Conflicts

> Prerequisites: Modules 1–6. You need warps and the linearization rule (M3), the
> replay mechanism and constant-memory broadcast-vs-serialize (M4), the
> sector-counting method and row pitch (M5), and shared memory itself — static
> and dynamic allocation, the third launch parameter, cooperative loading,
> tiling, halos, barrier placement (M6).
>
> What this module gives you: the ability to look at any shared-memory indexing
> expression and say, before compiling, how many cycles one warp's access costs
> — and three ways to fix it when the answer is bad.

---

## Concept

Module 6 sold you shared memory on latency: roughly 20–30× faster than global,
software-managed, per-block. That price is the *good* case. Shared memory has a
second cost model layered on top of latency, one that depends entirely on which
addresses the 32 lanes of a warp ask for, and it can cost you a factor of 16
without changing a single line of algorithm.

### The banked scratchpad — **PORTABLE CUDA CONCEPT**

Shared memory is not one monolithic SRAM. It is a collection of independent
memory **banks**, each with its own address decoder and its own output port.
Each bank can service **one request per cycle**. A single-ported SRAM that could
serve 32 arbitrary addresses per cycle would need 32 read ports and would be
enormous; 32 one-ported banks plus a crossbar is the affordable design. Every
GPU vendor's scratchpad memory works this way, and so do most register files,
vector-machine memories, and multi-bank caches. The idea is portable. The
numbers are not.

### The bank map on sm_89 — **ARCHITECTURE-SPECIFIC**

On Ada (and on every NVIDIA architecture from Kepler onward, in the default
configuration) shared memory has:

- **32 banks**, matching the warp width;
- **4 bytes per bank per cycle** — one 32-bit word;
- a **word-interleaved** (striped) mapping, not a block mapping.

The map is:

```
bank(byte_address) = (byte_address / 4) % 32
```

Consecutive 4-byte words go to consecutive banks, wrapping every 32 words
(every 128 bytes). So `float s[64]`: `s[0]` is in bank 0, `s[1]` in bank 1, …
`s[31]` in bank 31, `s[32]` back in bank 0.

Three historical notes so you recognise old code and old advice:

- **Compute capability 1.x** had 16 banks and resolved conflicts over
  half-warps. Advice written for that hardware ("pad to 17") is wrong now.
- **Compute capability 2.x (Fermi)** introduced 32 banks.
- **Compute capability 3.x (Kepler)** offered an optional **8-byte bank width**
  via `cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)`, which
  made `double` arrays conflict-free by default. That mode was removed; the
  API still exists on sm_89 but is deprecated and has no effect. Do not write
  code that depends on it.

Anything you read that names a bank count, a bank width, or a magic padding
constant is architecture-specific. The *method* — map lane to bank, count
multiplicity — is not.

### The conflict rule, stated precisely

Consider one **warp** executing one shared-memory **instruction** (this is
exactly the granularity M5 used for coalescing: per warp, per instruction, never
per thread and never per block).

The 32 lanes produce up to 32 addresses. Group them by bank. Within a bank,
group them by **word**.

- If a bank is asked for **one distinct word**, it delivers that word once and
  the crossbar **broadcasts** it to every lane that wanted it. **One cycle**,
  regardless of how many lanes asked. Two lanes, or thirty-two, same cost.
- If a bank is asked for **N distinct words**, the hardware **replays** the
  instruction against that bank N times. **N cycles.**

The cost of the whole instruction is set by the **busiest bank**:

```
conflict degree D = max over banks of (number of DISTINCT words that bank must supply)
```

`D = 1` is called **conflict-free**. `D = 32` is the worst case a 4-byte access
can reach: all 32 lanes want 32 different words that all live in one bank.

Two sentences that are easy to get backwards, so read them twice:

> **Same bank, same word → broadcast, free.**
> **Same bank, different word → serialize, one replay each.**

### This is M4's mechanism, not a new one

Module 4 measured constant memory: a warp in which all 32 lanes read the same
constant address costs one access; a warp in which they read 32 different
constant addresses costs 86× more, because the hardware **replays the
instruction once per distinct address**. That was a 64 KB window behind a
per-SM constant cache. This module is the same sentence with a different
noun. Shared memory replays the instruction once per distinct word per bank.

The identical mechanism, a third time, is atomic contention — which Module 10
closes the loop on. Replay is the GPU's universal answer to "one instruction,
several resources, not enough ports." Whenever you see a memory in this course
serialize, ask what the port is and what the unit of replay is.

Note the symmetry with M5's broadcast case: 32 lanes reading the same *global*
address cost one 32 B sector, because the coalescer collapses identical
addresses. The shared-memory crossbar does the same collapse for identical
words. Different memory, same economics: **the hardware charges you for distinct
resources touched, not for lanes.**

### Computing the degree by hand

This mirrors M5's sector-counting procedure exactly, one level down. M5:
*enumerate 32 addresses, shift right by 5, count distinct sector ids.* Here:

1. Write down the element index each of lanes 0..31 will use. Only lane matters
   — `threadIdx.x % 32` after linearization (M3).
2. Convert to a byte address: `index * sizeof(element)` (plus the array base,
   which you may take as bank 0 — a `__shared__` array is 4-byte aligned at
   worst, and the base only rotates the whole map).
3. Compute `word = addr / 4` and `bank = word % 32`.
4. Bucket the **distinct words** per bank.
5. `D` = the largest bucket.

Worked cases, all on `__shared__ float s[...]`:

| expression | lane → bank | distinct words per busy bank | D |
|---|---|---|---|
| `s[tid]` | `tid` | 1 | **1** |
| `s[2*tid]` | `2*tid % 32` = 0,2,4,…,30,0,2,… | lanes 0 and 16 → words 0 and 32, both bank 0 | **2** |
| `s[3*tid]` | `3*tid % 32`, a permutation | 1 | **1** |
| `s[32*tid]` | `32*tid % 32` = 0 for every lane | words 0,32,64,…,992 — 32 of them, all bank 0 | **32** |
| `s[tid/2]` | 0,0,1,1,2,2,… | lanes 2k and 2k+1 want the **same word** | **1** |
| `s[0]` | 0 | one word, 32 readers | **1** |
| `s[33*tid]` | `33*tid % 32` = `tid` | 1 | **1** |

Two of those rows are where people go wrong.

`s[tid/2]` *looks* like a 2-way conflict: two lanes per bank. It is not. The two
lanes want the identical word, and that is a broadcast. Counting **lanes** per
bank gives 2; counting **distinct words** per bank gives 1. Count words.

`s[3*tid]` and `s[33*tid]` are conflict-free because 3 and 33 are **coprime with
32**. In general, `s[k*tid]` has `D = gcd(k, 32)` — a one-line consequence of the
fact that `k*tid mod 32` takes `32/gcd(k,32)` distinct values, each hit
`gcd(k,32)` times, by distinct lanes and therefore at distinct words. Odd
strides are always free. Every power of two up to 32 costs exactly that power
of two.

### The 2D case, which is where this actually bites

```cpp
__shared__ float tile[32][32];
```

Element `(r, c)` lives at flat index `r*32 + c`, so `bank(r,c) = (r*32 + c) % 32
= c % 32 = c`. **The bank depends only on the column.**

- **Row access** `tile[threadIdx.y][threadIdx.x]`: within a warp, `threadIdx.x`
  runs 0..31 and `threadIdx.y` is constant (M3's linearization rule). So `c`
  varies over all 32 values → 32 distinct banks → **D = 1**.
- **Column access** `tile[threadIdx.x][threadIdx.y]`: now `r` varies 0..31 and
  `c` is constant. Every lane lands in bank `c`, and the 32 rows are 32
  different words. **D = 32.** This is the worst access the hardware admits,
  and it is one transposition away from the free one.

Any time a kernel writes a tile one way and reads it the other — transpose,
matrix multiply's B operand, a column-wise reduction, an FFT butterfly stage —
this is the access you have written.

### Fix 1: padding

```cpp
__shared__ float tile[32][33];   // one extra column, never read
```

Now `(r, c)` is at `r*33 + c`, so `bank(r,c) = (33r + c) % 32 = (r + c) % 32`.
Row `r` is rotated by `r` banks relative to row 0.

- Column access (`r` = 0..31, `c` fixed): banks are `(r + c) % 32`, which is a
  permutation of 0..31 → **D = 1**. Fixed.
- Row access (`r` fixed, `c` = 0..31): banks are `(r + c) % 32`, also a
  permutation → still **D = 1**. Not broken.

**This is exactly M5 Exercise 3's row pitch, one level down the hierarchy.**
There, you chose a pitch so that every row of a 2D array in *global* memory
started on a 32-byte sector boundary. Here you choose a pitch so that every row
starts on a different *bank*. Same decision — the leading dimension is not the
row length — driven by a different granularity: 32 B sectors there, 4 B banks
here. Keep the two straight: 68 was a bad global pitch because
`68*4 % 32 == 16`; 33 is a good shared pitch because `gcd(33, 32) == 1`.

The general rule for a `[R][C]` float tile read down a column: pad to a pitch
`P` with `gcd(P, 32) == 1`. `P = C + 1` works whenever `C` is even; if `C` is
already odd, `P = C` needs no padding at all.

**What padding costs.** Shared memory is a per-SM resource of 100 KB, and it
caps how many blocks can be resident (Module 19 owns occupancy properly; you
have enough from M1 and M4 to do the arithmetic). A `[192][32]` float tile is
24576 B; padded to `[192][33]` it is 25344 B. Add the 1024 B the driver reserves
per resident block, round up to the 128 B allocation granularity, and:

```
102400 / 25600 = 4 blocks/SM        (unpadded)
102400 / 26368 = 3 blocks/SM        (padded)
```

3.1 % more memory, 25 % fewer resident blocks. That is measured in Exercise 2.
At real GEMM tile sizes, where shared memory is the binding constraint on how
large a tile you can hold, this is the difference between a tile that fits and
one that does not.

### Fix 2: swizzling (XOR permutation)

Padding buys bank separation with bytes. A **swizzle** buys it with arithmetic.

```cpp
// element (r, c) is stored at:
r * 32 + (c ^ (r & 31))
```

The array is still exactly `32*32` floats. For each row `r`, the column index is
permuted by XOR with `r`. XOR with a fixed value is a **bijection** on
`[0, 32)`, so nothing collides and nothing is lost.

Verify both directions:

- **Column access**, `c` fixed, `r` = 0..31:
  `bank = (r*32 + (c ^ r)) % 32 = (c ^ r) % 32 = c ^ r`.
  As `r` runs over 0..31, `c ^ r` runs over all 32 values exactly once (XOR by a
  fixed `c` is a bijection). **D = 1.**
- **Row access / the cooperative load**, `r` fixed, `c` = 0..31:
  `bank = c ^ r`, again a permutation of 0..31. **D = 1.**

Both phases conflict-free, zero extra bytes. This is why **CUTLASS and every
modern hand-written GEMM kernel swizzle rather than pad**: at the tile sizes
those kernels use, shared memory capacity determines the tile size, which
determines the arithmetic intensity, which determines whether the kernel is
compute-bound. Spending 3 % of shared memory on padding is spending 3 % of the
tile. Module 43 covers CUTLASS's swizzle layouts and the `cp.async` /
Tensor-Core pipelines they feed.

Swizzling is not free of cost, only free of *memory* cost. It adds an XOR (a
`LOP3` in SASS) to every address computation. In Exercise 2 the swizzled kernel
emits 10 more `LOP3` instructions than the padded one and runs ~19 % slower,
even though it keeps 25 % more blocks resident. That is a real, measured result,
and the correct conclusion is not "swizzles are bad" — it is that on a kernel
whose tile already fits comfortably, padding is the cheaper fix, and the swizzle
earns its keep when capacity is what you are short of.

### Wider than 4 bytes: `double` and `float4`

A bank is 4 bytes wide. The whole bank array therefore delivers at most
**32 × 4 = 128 bytes per cycle**. But a warp of 32 `double` lanes asks for 256
bytes, and a warp of 32 `float4` lanes asks for 512 bytes. No arrangement of
addresses makes that fit in one cycle.

So the hardware **splits the request into phases**:

| element | bytes/warp | phases | lanes per phase |
|---|---|---|---|
| `float` (4 B) | 128 | 1 | 32 |
| `double` (8 B) | 256 | 2 | 16 |
| `float4` (16 B) | 512 | 4 | 8 |

**Conflicts are resolved independently within a phase.** Lanes in different
phases never conflict with each other, because they were never going to be
served in the same cycle.

This is where folklore goes wrong. Take `sd[2*tid]` on `__shared__ double
sd[...]`. Naive 4-byte arithmetic: element `2*tid` is at byte `16*tid`, word
`4*tid`, bank `4*tid % 32` — 8 distinct banks, 32 lanes, so "4-way conflict, 4×
slower." Measured: **1.65–1.90×**, because the correct count is per phase
(lanes 0–15 alone produce a 2-way conflict, and there are 2 phases), not
per warp.

The cleanest experimental proof of the split, from `example02.cu`:

```
double dd[(tid%16)*16]      naive degree 16      2.3082 ms
double dd[(tid%32)*16]      naive degree 32      2.2575 ms      ratio 0.98
```

Whole-warp arithmetic says the second pattern touches twice as many distinct
words and should cost twice as much. It costs the same, to within measurement
noise, because the extra 16 words belong to the *second* phase, which was
already paying 16 cycles for its own 16 words. Doubling the work in a phase that
was already saturated buys nothing and costs nothing.

Be careful about believing a model further than it has been measured. The same
example shows the 16-byte case **not** obeying the rigid four-contiguous-phase
model: `qq[(tid%8)*8]` and `qq[(tid%32)*8]` should be equal by that model and
measure 1.53× apart, with the *smaller* working set cheaper. The hardware is
evidently doing something smarter when several phases need the same words. Treat
the phase model as an **upper bound** for 16 B accesses, verify with a
measurement, and reach for the profiler when it matters.

One practically useful consequence: padding a double tile takes **one double**,
not one float. `sd[17*tid]` is conflict-free (measured 1.00× of `sd[tid]`);
`sd[16*tid]` is a full disaster.

### Measuring it — **forward reference to Module 23**

Nsight Compute counts bank conflicts directly:

```
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    --kernel-name <regex> ./your.exe
```

`..._op_ld` is loads, `..._op_st` is stores. The number reported is the count of
**extra** wavefronts caused by conflicts — a conflict-free kernel reports zero,
and a kernel with `D`-way conflicts reports roughly `(D-1)` per warp-instruction.
The companion counters
`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` (total wavefronts) and
`smsp__sass_average_data_bytes_per_wavefront_mem_shared` complete the picture.
Module 23 covers the profiler properly, including how to read the Memory
Workload Analysis section and what a "wavefront" is in NVIDIA's counter
vocabulary.

> **A note about this machine.** Nsight Compute 2026.1.0 is installed here, but
> profiling requires elevated GPU performance-counter permissions
> (`ERR_NVGPUCTRPERM`) which this account does not have. Every conflict degree
> claimed in this module was therefore verified by **timing ratios** against a
> controlled degree sweep, not by reading the counter. If you can run `ncu`,
> run it — the counter is the ground truth and a timing ratio is an inference.
> The commands above are correct; try them.

### What a conflict actually costs on sm_89

Here is the measured law, from a sweep that constructs an *exact* conflict degree
`D` for `D` = 1..32 (`example01.cu`):

| D | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| measured ratio | 1.00 | 0.99 | 1.92 | 3.80 | 7.47 | 14.79 |
| `D/2` | 0.5 | 1 | 2 | 4 | 8 | 16 |

Cost is proportional to `max(2, D)`, not to `D`. Two facts fall out:

> **⚠️ Width qualifier (measured in Module 18 — read this before you generalize).**
> **`max(2, D)` holds for 4-byte shared accesses only.** The floor of 2 exists
> because a 32-lane 4 B read asks for 128 B while the bank array delivers
> 128 B/cycle — the request is under-subscribed, so there is a spare cycle and a
> 2-way conflict hides inside it.
>
> An `LDS.128` (a `float4` shared read) is phase-split into 4 phases of 8 lanes,
> and **one phase already asks for the full 128 B**. There is no spare cycle, so
> on 16-byte shared reads **cost is proportional to `D` with no floor of 2** — a
> 2-way conflict is no longer free.
>
> Module 18 isolated this cleanly: two GEMM kernels with identical accumulators,
> registers, shared bytes, blocks/SM, occupancy, reuse ratios, and
> **byte-identical inner-loop SASS**, differing only in B-read conflict degree
> (D=1 vs D=2), measured **1.61× apart** against a 1.67× cycle-model prediction.
> This matters the moment you vectorize a tile read, which is exactly what a
> fast GEMM does.

1. **A conflict-free 32-lane 4 B shared access already occupies the shared
   memory pipeline for two cycles.** You cannot go faster than that, which is
   why `D = 1` and `D = 2` measure identically.
2. **A 2-way bank conflict is free on this hardware.** `s[2*tid]` is a genuine
   2-way conflict by the rule and costs nothing measurable. Do not spend
   engineering effort removing 2-way conflicts on Ada. Do spend it on 4-way and
   worse, where the cost is exactly `D/2`.

The measured ratios run 5–10 % below `D/2` because the timing loop also contains
address arithmetic and an FFMA, which do not get slower when the banks do.
`D/2` is an upper bound on the **shared-memory term**, never on the kernel.
Exercise 2's whole kernel, with a genuine `D = 32`, measures 12.6–15.4× rather
than 16×, and Exercise 3's measures 10.5–12.9×, for exactly that reason.

---

## Hardware Mental Model

Recall the SM from Module 1: four processing blocks, one warp scheduler each,
and a shared 128 KB unified L1 + shared memory block of which up to 100 KB is
addressable as shared. That block is the **L1TEX unit**, and it contains:

- the **tag/data arrays** used as L1 cache for global traffic,
- the **shared memory array**, organised as 32 banks × 4 B,
- an **address generation stage** that takes a warp's 32 addresses,
- a **crossbar** connecting 32 bank outputs to 32 lane inputs,
- and a **replay loop**.

For a shared load the sequence is:

1. The LSU pipeline receives one shared-memory instruction with a 32-lane active
   mask (M8 covers masks; here assume all 32 active).
2. Address generation produces 32 byte addresses and derives `(bank, word)` for
   each. Inactive lanes supply no address — the same rule M5 established for
   coalescing.
3. The unit partitions the request into **wavefronts**: maximal subsets of the
   32 requests that can be satisfied in one cycle, i.e. subsets in which no two
   *distinct* words share a bank. Requests for the same word are merged into one
   bank read plus a crossbar fan-out.
4. Each wavefront takes one cycle through the bank array and the crossbar.
5. The instruction does not retire until the last wavefront lands. Its latency
   is the conflict-free latency plus `(wavefronts - 1)` extra cycles, and its
   *occupancy of the LSU pipeline* is `wavefronts` cycles — which is what
   throughput-bound kernels actually feel.

The greedy partition in step 3 is exactly why the degree is
`max over banks of distinct words`: a bank that must supply `D` words forces at
least `D` wavefronts, and no bank forces more.

**Why broadcast is free.** The crossbar is a fan-out network, not a set of
point-to-point wires. Once bank `b` has driven word `w` onto its output, routing
that value to one lane costs the same as routing it to twenty. There is no
arbitration to do — every requesting lane wants the same bits. This is the same
reason the constant cache broadcasts for free, and the same reason M5's
32-lanes-one-address global read costs one sector.

**Why the floor is 2 cycles and not 1.** The measured data (D=1 and D=2 cost
identically, and cost is linear in D from D=3 upward) says a 32-lane 4 B request
occupies the shared pipeline for two cycles even when perfectly conflict-free.
A `float4` conflict-free access measures 1.89× a `float` one — consistent with
4 phases against 2 pipeline slots, not 4 against 1. The clean interpretation is
that the LSU issues a warp's shared access over two pipeline slots, so the bank
array has two cycles of slack that a 2-way conflict fits inside. Whatever the
exact microarchitectural reason, treat it as measured behaviour of sm_89, not as
a portable law: it is the reason the "model" column in `example01.cu` uses
`max(2, D)/2` rather than `D`.

**Why the bank map is word-interleaved.** If shared memory were block-mapped —
bank 0 owning the first 1/32 of the array, bank 1 the next — then the single
most common access in all of GPU programming, `s[threadIdx.x]`, would put all 32
lanes in bank 0 and serialize 32 ways. Word interleaving makes the common case
free and pushes the pathology onto strides that are multiples of 32, which are
rarer and which you can always fix by changing a pitch. The designers optimised
for the access you write without thinking.

**Why this is per-warp.** Banks are a resource of the SM's L1TEX unit, shared by
all warps resident on the SM. But conflict *resolution* happens per instruction:
the unit takes one warp's 32 addresses and partitions them. Two different warps
hitting the same bank in the same cycle is not a "conflict" in this sense — it
is ordinary pipeline contention, and it is why a heavily conflicted kernel does
not slow other warps down 32×, it just consumes 32× as many LSU slots. This
matters for reasoning: you fix conflicts by changing the address set *within* a
warp, never by changing how many warps you run.

**Where the bytes go.** On Ada the 128 KB L1TEX block is split between L1 cache
and shared memory by a carveout (M4 introduced `cudaFuncAttributePreferred
SharedMemoryCarveout`). Shared memory allocations are rounded up to a **128 byte
granularity**, and the driver charges each resident block an extra **1024 bytes**
on top of what it asked for. Both facts are needed to predict occupancy
correctly, and both are why the hand-computed and API-reported occupancies in
Exercise 2 agree only when you include them. Query them with
`cudaDeviceGetAttribute(..., cudaDevAttrReservedSharedMemoryPerBlock, 0)`.

---

## Code Walkthrough

### `example01.cu` — the bank map and the cost law

Build and run:

```
nvcc -arch=sm_89 -O3 -o example01.exe example01.cu
.\example01.exe
```

The host half is the mechanical procedure, written out:

```cpp
static int bank_of(uintptr_t byte_addr) { return (int)((byte_addr / BANK_W) % BANKS); }
```

and the degree calculation, which is the only part with any subtlety:

```cpp
for (int lane = 0; lane < WARP; ++lane) {
    uintptr_t a0 = (uintptr_t)pat_index(p, lane) * (uintptr_t)elemBytes;
    for (int off = 0; off < elemBytes; off += BANK_W) {   // wide types span words
        uintptr_t a = a0 + off;
        int b = bank_of(a);
        unsigned long long w = a / BANK_W;
        bool seen = false;
        for (int k = 0; k < nw[b]; ++k) if (words[b][k] == w) { seen = true; break; }
        if (!seen) words[b][nw[b]++] = w;      // DISTINCT words, not lanes
    }
}
```

Note the `if (!seen)`. Drop it and you are counting lanes, and `s[tid/2]` comes
out as a 2-way conflict instead of the broadcast it is. That single line is the
difference between the naive model and the correct one.

The device half is one templated kernel:

```cpp
template <int P>
__global__ void bankKernel(float* out)
{
    __shared__ float s[TN];
    ...
    #pragma unroll 4
    for (int it = 0; it < ITERS; ++it) {
        int o = (it * 32) & (TN - 1);          // +128 B: new word, SAME bank
        a0 += s[(idx + o      ) & (TN - 1)];
        a1 += s[(idx + o +  32) & (TN - 1)];
        ...
    }
```

Two design decisions carry the measurement:

- **`o` advances by 32 floats = 128 bytes.** That is exactly one full trip
  around the bank array, so every iteration reads a *different word* in the
  *same bank* as the last. The conflict degree is constant for the whole loop,
  which is what makes a throughput measurement meaningful.
- **Four independent accumulators.** A single `acc +=` chain would serialize on
  the ~4-cycle FADD latency and hide the bank behaviour entirely. This is the
  same ILP argument M1 made about dependent FMA chains.

Observed output on this GPU (warm; ratios, not absolutes, are the stable part):

```
pattern       degree         ms    measured      model
s[tid]             1     0.0584       1.00x       1.0x
s[2*tid]           2     0.0580       0.99x       1.0x
s[3*tid]           1     0.0581       1.00x       1.0x
s[4*tid]           4     0.1119       1.92x       2.0x
s[8*tid]           8     0.2220       3.80x       4.0x
s[16*tid]         16     0.4360       7.47x       8.0x
s[32*tid]         32     0.8634      14.79x      16.0x
s[tid/2]           1     0.0580       0.99x       1.0x
s[0]               1     0.0584       1.00x       1.0x
s[33*tid]          1     0.0581       0.99x       1.0x
```

Everything the Concept section claimed is in that table: the `gcd` law
(`s[3*tid]` free, `s[8*tid]` exactly 8-way), the broadcast rows at 1.00×, the
padding row at 0.99×, the 32-way disaster at 14.79×, and the 2-way conflict
costing nothing at all.

### `example02.cu` — phases

```
nvcc -arch=sm_89 -O3 -o example02.exe example02.cu
.\example02.exe
```

The double kernel contains one line that is worth explaining because it looks
like a trick:

```cpp
a0 ^= __double_as_longlong(s[(idx + o) & (TND - 1)]);
```

Ada runs FP64 at **1/64** the FP32 rate. Write the honest `acc += sd[i]` and the
kernel measures the FP64 pipeline, reporting 2.33 ms for *every* pattern
including the 32-way one — the banks become invisible. Reinterpreting the loaded
bits as an integer and XOR-accumulating keeps the load a genuine 8-byte shared
load (`LDS.64`) while making the arithmetic free. If you ever benchmark a
`double` kernel on a consumer or workstation Ada/Ampere part, check whether you
are measuring what you think you are.

The host model is the phase split, in six lines:

```cpp
int phases = elemBytes / BANK_W; if (phases < 1) phases = 1;
int lanes  = WARP / phases;
int total  = 0;
for (int ph = 0; ph < phases; ++ph)
    total += group_cost(p, elemBytes, ph * lanes, lanes, wrapMask);
return total < 2 ? 2 : total;
```

Observed output (warm):

```
pattern                phases  naive  cycles        ms   measured     model
float  s[tid]               1      1       2    0.0590      1.00x     1.00x
float  s[2*tid]             1      2       2    0.0586      0.99x     1.00x
float  s[32*tid]            1     32      32    0.8636     14.63x    16.00x
double dd[tid]              2      2       2    0.0685      1.00x     1.00x
double dd[2*tid]            2      4       4    0.1129      1.65x     2.00x
double dd[4*tid]            2      8       8    0.2231      3.26x     4.00x
double dd[17*tid]           2      2       2    0.0659      0.96x     1.00x
double dd[(tid%16)*16]      2     16      32    0.8631     12.61x    16.00x
double dd[(tid%32)*16]      2     32      32    0.8648     12.63x    16.00x
float4 qq[tid]              4      4       4    0.1122      1.00x     1.00x
float4 qq[2*tid]            4      8       8    0.2222      1.98x     2.00x
float4 qq[17*tid]           4      4       4    0.1116      0.99x     1.00x
float4 qq[(tid%8)*8]        4      8      32    0.5582      4.97x     8.00x
float4 qq[(tid%32)*8]       4     32      32    0.8647      7.70x     8.00x
```

Five things to take from it:

1. `double dd[tid]` costs the same as `float s[tid]` (0.0685 vs 0.0590 — the
   9 % is the integer XOR). Two phases of one cycle each equals the 4-byte
   floor of two cycles. **Using doubles in shared memory is not inherently
   slower.**
2. `double dd[2*tid]` costs 1.65×, not the 4× naive arithmetic predicts.
3. The `(tid%16)*16` / `(tid%32)*16` pair measures **1.00×** apart despite
   naive degrees of 16 and 32. That is the phase split, measured.
4. `float4 qq[2*tid]` costs 1.98× — a 2-way conflict that is **not** free,
   unlike the float case, because a 16 B access already uses every one of its
   four phases at full width and has no slack left.
5. The `float4` `(tid%8)*8` / `(tid%32)*8` pair measures 1.53× apart where the
   model says 1.00×. Documented, unexplained, and flagged in the program's own
   output. A model you have measured and found wanting is worth more than one
   you have only read.

---

## Check Your Understanding

Answers in `solutions/module07/check_your_understanding.md`. None of these can
be looked up; all require reasoning about the model.

1. A warp executes `v = s[f(tid)]` on `__shared__ float s[1024]`, where `f` is
   an arbitrary function. Give a necessary and sufficient condition on the
   **multiset** `{f(0), …, f(31)}` for this access to be conflict-free. Then
   explain why the condition "the 32 values are distinct" is neither necessary
   nor sufficient.

2. A colleague fixes a 32-way column conflict in a `__shared__ float
   tile[32][32]` by changing the declaration to `tile[32][40]`. They chose 40
   because it is a multiple of 8 and therefore "nicely aligned for `float4`".
   What conflict degree does the column access now have, and why is the
   alignment reasoning not just irrelevant but actively pointed in the wrong
   direction? Relate your answer to M5 Exercise 3's pitch 68.

3. A kernel reads `sd[tid]` from `__shared__ double sd[256]` and a second reads
   `s[tid]` from `__shared__ float s[512]`. Both are conflict-free by any
   counting method. The double version moves twice the bytes. Predict the ratio
   of their costs and justify it from the phase model — then say what the
   *global*-memory analogue of this situation is and why the answer there is
   different.

4. You have fixed a 32-way conflict, `ncu` now reports
   `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum = 0`, and the
   kernel is 4 % faster. Give two distinct, concrete explanations for why
   removing a 16× penalty on one instruction class bought 4 %, and describe a
   measurement that distinguishes them.

---

## Exercises

### Exercise 1 — `exercise01.cu`: bank maps by hand, then measured

**What the program must accomplish.** Reproduce the conflict-degree calculation
in code, predict the relative cost of eight indexing expressions from a paper
table, and check both against a measurement and a CPU reference.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe
```

Before writing any code, fill the paper table in the file header for the eight
expressions: `s[tid]`, `s[2*tid]`, `s[8*tid]`, `s[32*tid]`, `s[tid/2]`,
`s[31-tid]`, `dd[tid]`, `dd[2*tid]` (the last two are `double`).

- **TODO 1** — `bank_of(byte_addr)`. The harness tests structural properties: it
  must be a bijection over 32 consecutive words, have period 128 B, and ignore
  the low 2 bits of the address.
- **TODO 2** — `degree_in_group(elemIdx, n, elemBytes)`: the number of cycles a
  group of `n` lanes needs, for any element size. The harness independently
  computes how many distinct words the warp touches and checks your answer lies
  between `ceil(words/32)` and `words`.
- **TODO 3** — your eight predicted relative costs. Scored in a 30 % band.
- **TODO 4** — `cycles_for_pattern(p)`: split the warp into contiguous groups
  small enough that the group's request fits in the 128 B the bank array
  delivers per cycle, and sum.

**Validation.** Structural tests on TODOs 1 and 2, a CPU reference for all eight
kernels, and a score on your predictions. `OVERALL: PASS` needs correct
arithmetic, `PASS` numerics, and at least 6 of 8 predictions in band.

At least one of your eight predictions will miss. The harness names which and
refuses to explain.

### Exercise 2 — `exercise02.cu`: fix a 32-way conflict, twice

**What the program must accomplish.** A tiled kernel with a genuine 32-way
column conflict, plus two working alternatives you write, all three validated
and timed side by side with their occupancy.

```
nvcc -arch=sm_89 -O3 -Xptxas -v -o exercise02.exe exercise02.cu
.\exercise02.exe
```

- **TODO 1** — the conflict degree of the compute-phase read as shipped.
  Checked structurally against the layout the harness owns.
- **TODO 2** — a padded pitch. Extra shared memory is allowed here.
- **TODO 3** — a second layout meeting the same bank requirement in which the
  shared array is **exactly `ROWS * 32` floats**. Not one word more. The harness
  checks range, injectivity, the column read, **and the row write** — it is easy
  to fix one phase by breaking the other.
- **TODO 4** — `max_blocks_per_sm(...)`, by hand, from the device limits. Your
  answer is compared against
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. Two things the obvious
  division misses will make you disagree with the API until you include them;
  the TODO comment names both.

**Validation.** All three kernels against a CPU reference, the structural tests
on TODO 3, your occupancy against the API, your degree against the true degree,
and a requirement that both fixes beat the original by more than 2×.

### Exercise 3 — `exercise03.cu`: which of these is the disaster?

**What the program must accomplish.** Six configurations across two kernels;
predict the conflict degree of every one before running anything.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe
```

Kernel A indexes shared memory with a value loaded from a data array — the
pattern everyone flags on sight. Kernel B gives every thread its own private
slice of a shared scratch buffer and reads `scratch[tid][k]` — the pattern
nobody flags at all. Exactly one of those instincts is correct.

- **TODO 1** — the conflict degree of all six configurations. Checked exactly.
- **TODO 2** — over **every possible** content of the index array, the largest
  conflict degree kernel A can be made to exhibit at `SCALE == 1`. The harness
  brute-forces 200 000 candidate index sets and prints what it found. Reason
  about what the mask does to the reachable set of banks *and* of words; do not
  guess from the shape of the expression.
- **TODO 3** — a layout for kernel B's scratch buffer using exactly the same
  number of bytes, with the column read conflict-free.
- **TODO 4** — the index values that *maximise* kernel A's degree at
  `SCALE == 32`. The harness reports the degree your array achieves at both
  scales; the pair of numbers is the point.

**Validation.** CPU reference for both kernels, the brute-force check on TODO 2,
a degree check on TODO 4, structural tests on TODO 3, and all six degrees
correct. Nothing less than 6/6 passes.

---

## Prediction

Commit to these in writing before you run anything.

1. **`s[2*tid]` is a 2-way bank conflict by the rule.** Give the factor by which
   you expect it to be slower than `s[tid]`, to one decimal place. Then give the
   factor for `s[4*tid]`. If your two answers are 2.0 and 4.0, you have made a
   prediction the hardware will contradict exactly once — decide now which one
   and why.

2. **The 32×32 column access costs 32 cycles instead of 1.** Exercise 2's kernel
   does that access 32 × 64 times per thread and nothing else of consequence.
   Predict the whole-kernel speedup from fixing it. State whether your number is
   above or below 16×, and name the specific thing that puts it on that side.

3. **Padding a `[192][32]` float tile to `[192][33]` costs 3.1 % more shared
   memory and one resident block per SM (4 → 3, i.e. 25 % occupancy).** The XOR
   swizzle costs no memory and no occupancy at all. Predict which of the two
   fixed kernels is faster, and by how much. Write down what would have to be
   true for the answer to flip.
