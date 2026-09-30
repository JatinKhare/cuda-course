# Module 06 — Check Your Understanding (answers)

---

## Q1 — "just give each of the 256 threads two cells"

**The scheme.** 340 tile cells, 256 threads, so the proposal is something like
"thread `t` loads cell `t`, and threads `t < 84` also load cell `256 + t`." That
does cover all 340 cells exactly once, so as an *enumeration* it is fine. The bug
is not in the count. It is in what the proposer almost always writes instead,
which is the 1:1 version:

```cpp
tile[(threadIdx.y + 1) * SW + (threadIdx.x + 1)] = in[row * w + col];   // interior only
```

plus a handful of "and the edge threads also fetch the halo" special cases. With
`TW=32, TH=8` the interior is 256 cells and the halo is 84, and the 84 split as
2·32 (top and bottom rows) + 2·8 (left and right columns) + **4 corners**.

**A concrete failure.** Take the corner cell `tile[0][0]` — tile coordinates
(ly=0, lx=0), i.e. image pixel `(row0-1, col0-1)`. It is not on the top edge
*only* nor on the left edge *only*, so a load that handles "row −1" with
`threadIdx.y == 0` and "column −1" with `threadIdx.x == 0` as two independent
special cases fetches it twice or not at all depending on how the two conditions
are combined, and the usual formulation (`if (ty==0) load above; if (tx==0) load
left;`) misses all four corners entirely.

**But the 5-point stencil never reads a corner.** `tile[ly±1][lx]` and
`tile[ly][lx±1]` only ever touch the four edge strips. So the corner omission is
*invisible* for this stencil — which is itself worth knowing, and is why the
solution notes for Exercise 1 accept a corner-free load as correct. The failure
the question asks about is a different one:

**The real failure is the last partial tile.** 1021 is not a multiple of 8 and
733 is not a multiple of 32. The rightmost column of blocks has `col0 = 704`,
covering columns 704…735, but the image only has columns up to 732. A 1:1 load
guarded by `if (row < h && col < w)` — the guard everyone writes — **skips** the
load for threads whose interior pixel is outside the image. Those tile cells are
then never written. Now look at which cells the *in-range* threads read:
thread `(tx=28, ty=k)` writes pixel 732, the last real column, and reads
`tile[ly][lx+1]` = tile column 30, which corresponds to image column 733 — a halo
cell that had to be **clamped to column 732**, and which the skipped thread never
wrote.

Symptom: a one-pixel-wide vertical stripe of wrong values down the right-hand
edge of the image, and a one-pixel horizontal stripe along the bottom, with the
wrong values being whatever the previously-resident block left in that SRAM —
so they are *plausible image values from elsewhere in the image*, not NaNs, and
they change between runs. `1021 × 733` also makes the stripes land at different
tile offsets in x and y, which is why one axis can look fine while the other is
broken.

**Why 1024×1024 hid it.** 1024 is a multiple of both 32 and 8. Every tile is
full, every thread's interior pixel is in range, no load is ever skipped, and the
only cells needing the clamp are on the outer boundary of the image — where a
global clamp in the load is also what the untiled reference does. The correct and
the broken versions agree everywhere. This is the same class of hiding that
Module 3's exercise was built to prevent: **test dimensions must be coprime with
every block dimension you will ever use.**

---

## Q2 — 40 % less DRAM traffic, 15 % slower

Both statements can be true because DRAM traffic is only the bottleneck when the
kernel is DRAM-bound. Two distinct mechanisms:

**Mechanism A — occupancy collapse.** The shared memory the tiles occupy is
charged at block placement (plus the 1024 B driver reservation), and it caps
blocks per SM:
`blocks_per_SM = min(1536/threads, 102400/(bytes+1024), 24)`.
Going from 8 KB to 25 KB per 256-thread block takes you from 6 blocks/SM (1536
threads, 100 % occupancy) to 3 (768 threads, 50 %). Fewer resident warps means
less memory-level parallelism — Little's Law from Module 1 — so the *remaining*
60 % of the traffic is issued with half the outstanding requests and takes
longer in wall-clock terms than the original 100 % did.

**Mechanism B — barrier convoying.** A barrier per tile forces the block to run
at the speed of its slowest warp at every tile boundary. Previously the scheduler
could hide one warp's long-latency load behind another warp's arithmetic
indefinitely; now that slack is cut off `tiles` times per block. The cost scales
with warps-per-block and with the number of barriers, and is entirely independent
of how many bytes moved.

(There is a third, which the module measures: the traffic that "disappeared" was
never going to DRAM in the first place — it was hitting in L1 — so the 40 %
reduction the profiler reports may be a reduction in *L1-to-SM* traffic, not DRAM
traffic. Read the counter names carefully.)

**Distinguishing them.** Change one thing without changing the other:

- Re-run the tiled kernel with the shared allocation artificially padded (add a
  dummy `__shared__ char pad[N]`) so occupancy drops *further* with no change to
  the algorithm. If the slowdown tracks the padding, it is A.
- Or query `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for both kernels: if
  they report the same blocks/SM, A is excluded outright and you are looking at B.
- To confirm B, increase the tile size so that the number of barriers per output
  element falls while occupancy is held constant (bigger tile, fewer tiles, same
  bytes per thread). If the gap closes, it was barrier cost.

---

## Q3 — `float a[N]` then `double b[M]`, launched with `N*4 + M*8`

`double` is 8-byte aligned; `b` is placed at byte offset `4N`.

- **N even** → `4N % 8 == 0`. The carve is aligned, the byte count `4N + 8M` is
  exactly the space used, and the kernel is **correct**.
- **N odd, offset used as written (`smem + 4*N`)** → `4N % 8 == 4`. Every
  `LDS.64`/`STS.64` to `b[]` is misaligned and the kernel dies with
  **`cudaErrorMisalignedAddress`** at execution time (not at launch; and the error
  is sticky). This is loud, immediate, and *safe* — the program cannot produce a
  wrong answer.
- **N odd, offset rounded up correctly in the kernel (`align_up(4N, 8)` = `4N+4`)
  but the launch still passing `4N + 8M`** → the carve is aligned, every access
  is legal, and the last 4 bytes of `b[M-1]` fall **outside the block's shared
  allocation**. This is the dangerous one.

**How dangerous — measured, because the answer is worse than the folklore.**
Exercise 2's solution was rebuilt with `sharedBytesFor()` returning 2184 instead
of 3208 — a full kilobyte short, with every block's last array entirely outside
the requested region. The result:

```
  sharedBytes = 2184
  naive                  PASS
  tiled, short loop      PASS
  tiled, padded tile     PASS
OVERALL: PASS
```

and `compute-sanitizer --tool memcheck` reported **no invalid access**. The
under-request is silent on this hardware: shared memory is allocated to a block
in a granularity larger than the request, the per-block window the hardware
bounds-checks against is the granted size rather than the asked-for size, and
memcheck models the same thing. So an under-request corrupts only when it spills
past the *granted* region, at which point it silently smashes another block's
scratchpad or the driver's 1024 B reservation.

The lesson, then, is the reverse of what you would expect: the **alignment** bug
is the benign one, because it faults loudly and immediately and cannot produce a
wrong answer. The **byte count** bug is the one to be afraid of, because neither
the hardware nor the tools will tell you, and the only defence is computing the
offsets and the total in a single `__host__ __device__` function that both sides
call.

---

## Q4 — a hypothetical grid-wide scratchpad

**The guarantee it breaks: blocks are placed on exactly one SM, are indivisible,
never migrate, and — crucially — are *not all resident simultaneously*.**

A grid of 4096 blocks on a 40-SM GPU runs in waves; block 4000 does not exist
while block 0 is running. For a scratchpad to be "visible to every block" it
would have to be visible to blocks that have already retired and to blocks that
have not been created — which means its contents must survive block retirement,
which means it is not per-block storage at all.

Suppose instead it is one shared region per SM, aliased across blocks. Then:

1. **It cannot be private, so it must be coherent.** Two blocks on two different
   SMs writing "the same" grid-scratchpad address must see each other's writes,
   or the abstraction is a lie. SM-local SRAM is not coherent across SMs (Module
   4) — that is precisely why L1 is not, and why device-scope atomics exist. To
   make it coherent you need a directory or a shared backing point, i.e. the
   thing L2 already is. At that point you have built a cache.

2. **It cannot be allocated at block launch, so it needs a replacement policy.**
   Per-block allocation works because the lifetime is known: reserve at
   placement, free at retirement. Grid-wide data has no such bracket. Capacity is
   finite and the grid's working set is not, so the hardware has to decide what
   to keep — a replacement policy, which is the defining feature of a cache and
   the defining *absence* in a scratchpad.

3. **It destroys the scheduling freedom that makes the model scale.** Module 1's
   whole argument for why a CUDA grid runs on 4 SMs or 144 SMs unchanged is that
   blocks are independent and the GigaThread engine may run them in any order,
   any number at a time. A grid-wide scratchpad creates cross-block dependencies
   on a resource whose availability depends on co-residency, which is exactly the
   restriction that makes cooperative launch (`grid.sync()`, Module 29) a special
   API with a co-residency precondition rather than the default.

So the answer is that the feature already exists, twice: as **L2** (coherent,
device-wide, with a replacement policy — a cache) and, in a restricted form, as
**distributed shared memory across a thread block cluster** on sm_90+, which buys
cross-block shared access precisely by introducing a *new* co-residency guarantee
(a cluster's blocks are guaranteed co-resident on one GPC). Ada (sm_89) does not
have clusters. Note the shape of the sm_90 solution: it did not remove the
residency guarantee, it enlarged the unit it applies to — which is the strongest
evidence that the guarantee, not the SRAM, is the scarce thing.
