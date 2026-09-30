# Module 06 / Exercise 2 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

The shipped solution carries a third configuration the exercise does not ask for
(`tiled, padded tile`); see "Performance reasoning".

---

## TODO 1 — the byte count

```cpp
static size_t sharedBytesFor(void)
{
    size_t off_pos = align_up((size_t)PROF_N * sizeof(float), alignof(float2));
    size_t off_wgt = align_up(off_pos + (size_t)TILE * sizeof(float2),
                              alignof(float));
    return off_wgt + (size_t)TILE * sizeof(float);
}
```

With `PROF_N = 33`, `TILE = 256`:

| array | type | alignment | offset | bytes |
|---|---|---|---|---|
| `prof` | `float[33]` | 4 | 0 | 132 |
| `pos` | `float2[256]` | **8** | **136** (not 132) | 2048 |
| `wgt` | `float[256]` | 4 | 2184 | 1024 |
| | | | **total** | **3208** |

The arithmetic everyone does first is `33·4 + 256·8 + 256·4 = 3204`. That is four
bytes short and, more importantly, it implies `pos` starts at byte 132.

**Why the `align_up` is not optional.** `float2` is declared
`__align__(8)`, so `ptxas` emits a 64-bit shared access (`STS.64` / `LDS.64`) for
`spos[k] = ...`. A 64-bit access requires an 8-byte-aligned address, and
`132 % 8 == 4`. Verified by building the solution with the `align_up` removed:

```
CUDA error cudaErrorMisalignedAddress at exercise02_solution.cu:253 -> misaligned address
```

The fault happens at execution time, not at launch — the launch configuration was
perfectly valid — so only the synchronizing call reports it, and the error is
**sticky**: the context is dead. This is Module 5's `cudaErrorMisalignedAddress`
(`float4` cast on an unaligned `float*`) in a different memory space, for exactly
the same reason: the *instruction* carries the alignment requirement, not the
pointer type you wrote in C++.

**Why the byte count is the dangerous half.** Get the alignment right in the
kernel and leave the launch asking for 3204 and you have a 4-byte overrun in
every block, forever. Measured, deliberately, with a *kilobyte* of under-request
(`sharedBytesFor()` returning 2184, so the whole `wgt` array is outside the
request):

```
  sharedBytes = 2184
  naive                  PASS
  tiled, short loop      PASS
  tiled, padded tile     PASS
OVERALL: PASS
```

and `compute-sanitizer --tool memcheck` reports no invalid access. Shared memory
is granted to a block in a granularity coarser than the request, and the window
the hardware bounds-checks against is the granted size, not the asked-for one —
so the under-request is invisible until it is large enough to smash a neighbour,
and then it is invisible in a different way. **The alignment bug is the safe one
because it is loud. The byte-count bug is the one to be afraid of.** The only
defence is structural: compute offsets and total in one `__host__ __device__`
function and call it from both sides, which is what the solution does.

---

## TODO 2 — the carve

```cpp
extern __shared__ char smem[];

const size_t off_pos = align_up((size_t)PROF_N * sizeof(float), alignof(float2));
const size_t off_wgt = align_up(off_pos + (size_t)TILE * sizeof(float2),
                                alignof(float));
float*  sprof = reinterpret_cast<float*>(smem);
float2* spos  = reinterpret_cast<float2*>(smem + off_pos);
float*  swgt  = reinterpret_cast<float*>(smem + off_wgt);
```

Declaring it as `char[]` and doing byte arithmetic is the idiom. Do not write

```cpp
extern __shared__ float smem_f[];
extern __shared__ float2 smem_f2[];      // aliases smem_f at offset 0
```

and expect two arrays. There is exactly **one** dynamic shared region per kernel;
every `extern __shared__` declaration names the same base address. That form
compiles, is legal, and gives you two views of the same bytes — which is
occasionally what you want and almost never what you meant.

The offsets could equally be computed from 0 and added (as here) or with a
running pointer and `align_up` on the pointer value. Offsets from 0 are safer:
the base of the dynamic region is guaranteed suitably aligned for any type, so an
offset that is a multiple of `alignof(T)` yields an address that is too, whereas
pointer arithmetic invites a `uintptr_t` round-trip that is easy to get wrong.

**Common wrong approaches.**

| approach | symptom |
|---|---|
| `float2* spos = (float2*)(smem + PROF_N*4)` | `cudaErrorMisalignedAddress`, sticky, every later API call fails |
| order the arrays `pos, wgt, prof` instead | works (16-byte-aligned base, then 8-byte, then 4-byte — alignment is monotonically decreasing so no padding is ever needed). **Ordering the arrays from widest to narrowest alignment makes the problem disappear.** The exercise fixes the order specifically to prevent you from dodging it. |
| host and kernel each compute offsets independently | works until someone edits one of them |

---

## TODO 3 — cooperative load, partial tile, barriers

```cpp
for (int k = (int)threadIdx.x; k < PROF_N; k += TILE) sprof[k] = prof[k];

int i = (int)(blockIdx.x * TILE + threadIdx.x);
float2 qq = q[i < nq ? i : nq - 1];
float acc = 0.0f;

for (int t0 = 0; t0 < m; t0 += TILE) {
    int cnt = (m - t0 < TILE) ? (m - t0) : TILE;

    __syncthreads();                       // rule 2 (and covers sprof on pass 0)
    if ((int)threadIdx.x < cnt) {
        spos[threadIdx.x] = p[t0 + threadIdx.x];
        swgt[threadIdx.x] = w[t0 + threadIdx.x];
    }
    __syncthreads();                       // rule 1

    for (int j = 0; j < cnt; ++j) {
        float dx = qq.x - spos[j].x, dy = qq.y - spos[j].y;
        acc += swgt[j] * profile_eval(sprof, dx * dx + dy * dy);
    }
}
if (i < nq) out[i] = acc;
```

Four things worth defending.

**(a) The profile table is loaded once, before the loop.** `PROF_N = 33 < TILE`,
so the strided loop runs once for threads 0–32 and zero times for the rest — the
standard shape for "fewer cells than threads", and the mirror image of Exercise
1's "more cells than threads". Loading it inside the tile loop is correct and
costs 16 redundant loads plus a barrier's worth of convoying per tile.

**(b) The partial tile.** `MS = 4093 = 15·256 + 253`. On the last pass three
threads have no source to stage. If you let them store anyway you read
`p[4093..4095]` — three elements past a `cudaMalloc` allocation, which will
usually *not* fault (the allocation is padded to 256 B) and will usually give you
garbage that three threads then multiply into every query's answer. If you let
them skip the store but still run the inner loop to `TILE`, you consume three
stale slots left by the previous tile. Either bug affects **every** query, so the
validator catches it immediately — this is the rare case where the obvious bug is
also the obvious failure.

**(c) Why there is no early return.** `NQ = 262111` is not a multiple of 256, so
the last block has threads with no query. They must not `return`, because they
have to keep reaching both barriers on every tile — they are part of the
cooperative load. Hence the clamp `q[i < nq ? i : nq-1]` (a harmless duplicate
read) and the guard on the store only. Exiting early here is the barrier-in-
divergent-control-flow hazard; Module 9 makes the rule precise.

**(d) Two barriers, not one.** The first is rule 2 — no thread may overwrite
`spos`/`swgt` while another is still reading the previous tile. Putting only the
second barrier in gives a kernel that passes on small inputs and fails
intermittently on large ones; that is precisely Exercise 3's TODO 3 defect.
Placing the rule-2 barrier at the *top* of the loop rather than at the bottom is
a small elegance: it also orders the `sprof` load on the first iteration, so the
profile table needs no barrier of its own.

---

## Synchronization / memory reasoning

Two barriers per tile, 16 tiles, 8 warps per block: 32 `BAR.SYNC` per block. The
cost is convoying, not the instruction — at each barrier the block runs at the
speed of its slowest warp. That is affordable here because the inner loop is 253
or 256 iterations of real arithmetic per tile, so the barrier is amortized over
~2000 instructions. In Exercise 1 the same barrier is amortized over five adds,
which is a large part of why that one loses.

---

## Performance reasoning

**Where the win comes from.** Two separate effects:

1. **The source tile.** Each staged source is consumed by all 256 threads of the
   block, so K = 256. But the untiled reads of `p[j]` and `w[j]` are *warp-uniform*
   — all 32 lanes read the same address — which is a 1-sector broadcast that hits
   in L1. So most of this K was already being captured, exactly as in Exercise 1.
2. **The profile table.** `profile_eval`'s index is derived from a distance, so
   it is **lane-varying**: 32 lanes, up to 32 different table entries, once per
   source, i.e. 4093 times per thread. This is Module 4's lane-varying-table case,
   the one where constant memory measured 24.6× slower than global. In global
   memory a lane-varying gather still goes through the L1 tag path with the
   distinct-line count as its cost; in shared memory it is a direct SRAM read.
   This is where most of the measured gain comes from, and it is the specific
   thing Module 4 promised Module 6 would fix.

**Measured**, min of 4 sweeps, 400 ms duration-based clock warm-up, all
configurations timed back to back and validated afterwards:

```
  config                         ms        Gpair/s     vs naive
  naive                      4.1929         255.86        1.00x
  tiled, short loop          2.9965         358.03        1.40x
  tiled, padded tile         3.1528         340.27        1.33x
```

Across repeated runs: naive 3.93–4.59 ms, tiled/naive **1.37–1.60×**, typically
~1.40×. Absolute times swing with thermal state; the ratio does not.

**Put that next to Exercise 1.** Same GPU, same methodology, same week:

| | K/H | tiled vs naive |
|---|---|---|
| Exercise 1 (5-point stencil) | 3.8 | **0.85×** |
| Example 2 (box filter R=4) | 32.4 | **1.06×** |
| Exercise 2 (this) | 256 | **1.40×** |

Fifty times the reuse buys about 1.6× more speedup than the stencil got. That
sublinearity is the honest shape of shared memory on a modern GPU with a large
L1: **you are not removing DRAM traffic, you are removing L1 tag lookups**, and
the price of an L1 hit is only a small multiple of the price of an `LDS`. The
regime where shared memory is genuinely transformative (Modules 15–17) needs one
more ingredient this kernel does not have: register blocking, so that the number
of memory instructions falls as well as their cost.

**The documented surprise.** The third configuration, `tiled, padded tile`, pads
the last tile with zero-weight entries so the inner loop bound is the
compile-time constant `TILE` instead of the runtime `cnt`. The standard rule of
thumb says a constant bound lets `ptxas` unroll and should be faster. It is
consistently **slower** — 1.33× against 1.40×, reproduced across runs, with and
without `#pragma unroll 8`. The short-loop version's `cnt` equals `TILE` on 15 of
16 tiles, so the loop is the same length in practice; what the padded version
adds is an `else` branch executing two shared stores on every tile for lanes that
have nothing to store, and (with unrolling) more live registers. The rule of
thumb assumed the constant bound bought something; here the bound was already
effectively constant and the padding was pure overhead. Measure, do not assume.

---

## Expected output

```
=== Module 6 / Exercise 2 : dynamic shared memory (SOLUTION) ===
  PROF_N=33  TILE=256  NQ=262111  MS=4093
  offsets: prof 0, pos 136 (naive would be 132), wgt 2184
  sharedBytes = 3208
  naive                  PASS
  tiled, short loop      PASS
  tiled, padded tile     PASS

  config                         ms        Gpair/s     vs naive
  naive                      4.1929         255.86        1.00x
  tiled, short loop          2.9965         358.03        1.40x
  tiled, padded tile         3.1528         340.27        1.33x

  reuse K: each staged source is used by all 256 threads of the block;
  each profile entry is read ~31752 times per block.

OVERALL: PASS
```

3208 bytes per block is small enough that occupancy is unaffected: at 256 threads
the limit is `min(1536/256, 102400/(3208+1024), 24) = min(6, 24, 24) = 6` blocks
per SM, 100 % occupancy, the same as the untiled kernel. Everything above 12288
bytes per block would start costing blocks — see Example 1 §D.

---

## The result that matters

`extern __shared__` hands you one untyped blob and the third launch parameter,
and everything else is your arithmetic. The two mistakes available are not
symmetric: the **alignment** mistake faults immediately with
`cudaErrorMisalignedAddress` and cannot corrupt anything, while the **byte-count**
mistake is silent — verified here at a full kilobyte of under-request, passing
validation with memcheck reporting nothing. Structure the code so the mistake is
impossible: one `__host__ __device__` function computes the offsets and the
total, the host passes its return value at launch, the kernel uses the same
offsets, and there is no second place for the two to disagree.

**Variation to try.** Reorder the layout to `pos, wgt, prof` (widest alignment
first) and watch every `align_up` become a no-op. Then ask whether you want to
depend on that: add a `double` accumulator array to the front and the padding is
back. Then try raising `TILE` to 1024 with 1024-thread blocks — shared memory per
block goes to 12424 B, occupancy is still capped by threads (1536/1024 = 1 block,
66.7 %, Module 3's ceiling), and the reuse factor K doubles. Predict which of
those two effects wins before you measure.
