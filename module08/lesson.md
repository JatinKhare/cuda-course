# Module 08 — Warps and SIMT Execution

> Prerequisites: Module 1 (SM anatomy, warp schedulers, SIMD vs SIMT),
> Module 2 (launch syntax, error checking), Module 3 (the linearization rule,
> bounds guards), Module 4 (storage classes), Module 5 (per-warp memory
> behaviour).
> What this module gives you: the exact rule for which lanes of a warp execute
> which instruction at which time, what that costs, and why every warp-level
> primitive on sm_70 and later carries an explicit mask.

Modules 3 and 5 taught you to name the 32 threads of a warp and the 32
addresses they touch. This module is about the other half of the warp: not
*where* it reads, but *what it executes*, and when some of its lanes do not.

Three debts are paid here. Module 1 promised that independent thread scheduling
would be explained. Module 3 promised that the difference between a bounds
guard that is predicated and a branch that genuinely diverges would be made
precise. Module 1 also promised that "divergence is warp-local" would be turned
from a slogan into a measurement.

---

## Concept

### The warp

A **warp** is 32 threads of one block that the hardware treats as a single
schedulable entity. Membership is fixed at block launch and follows the Module 3
**linearization rule** exactly:

```
tid  = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)
warp = tid / 32
lane = tid % 32
```

`lane` is the thread's position inside its warp, 0..31. Nothing you write can
change this mapping; it is not a scheduling decision the driver makes, it is
arithmetic on the thread's own index. **PORTABLE CUDA CONCEPT** (that warps
exist and are formed by linearization). **ARCHITECTURE-SPECIFIC** (that the
number is 32; query `cudaDevAttrWarpSize` if you need to be defensive, but no
shipping NVIDIA GPU has ever used another value).

### Partial warps

A block of 100 threads becomes `ceil(100/32) = 4` warps. Warps 0, 1, 2 hold
tids 0..31, 32..63, 64..95. Warp 3 holds tids 96..99 — four threads — and 28
lanes that **do not exist**.

Those 28 lanes are not "predicated off for a while". They are:

- permanently inactive for the whole lifetime of the block,
- counted against the SM's 1536 thread slots,
- allocated register file space, because registers are allocated per warp at a
  granularity of 32 lanes regardless of how many are live,
- never able to receive work.

That is the hardware reason for the Module 3 rule "make the block size a
multiple of 32". A block of 100 costs the same occupancy resources as a block
of 128 and does 78% of the work. `example01.cu` prints the roster and the
active mask `0x0000000f` that warp 3 presents.

The same arithmetic applies in 2-D and 3-D. A `dim3(10,10)` block also has 100
threads and also has one partial warp, but now the warp boundaries fall *inside*
rows: warp 0 runs from `(x,y) = (0,0)` to `(1,3)`, and warp 1 starts at `(2,3)`.
A row of that block is not a warp. Warps are cut out of the linearized sequence,
not out of the index space.

### The SIMT contract

One **warp scheduler** issues **at most one instruction per clock to at most one
warp**. Every **active** lane of that warp executes that instruction, on its own
private registers, against its own private addresses. Module 1 gave you the
counts: 4 processing blocks per SM on Ada, one scheduler each, so at most 4
warp-instructions per SM per clock, driving at most 128 lanes.

The consequence that matters for this module:

> The unit of instruction issue is the **warp**, not the thread. Anything that
> makes the lanes of a warp want to execute *different* instructions is paid for
> in extra issues.

### The active mask

Every instruction the hardware issues carries a 32-bit **active mask**: bit `L`
set means lane `L` participates, bit `L` clear means lane `L` is present but
produces no architectural effect — no register write, no memory access, no
address supplied to the coalescer (this is the Module 5 rule that predicated-off
lanes cannot force a sector).

`__activemask()` returns the mask of lanes that are executing *that*
`__activemask()` instruction with the caller. `__ballot_sync(mask, pred)`
returns the mask of lanes in `mask` whose `pred` is nonzero, and `__popc(m)`
counts set bits.

> **Scope note.** In this module `__activemask()`, `__ballot_sync()` and
> `__popc()` are **instruments for observing execution**, in the same way an
> oscilloscope is an instrument for observing a circuit. Module 30 covers the
> warp intrinsics as **tools for building algorithms** (shuffles as a
> communication primitive, ballot as a compaction primitive), and Module 12
> covers reductions. Nothing in this module builds an algorithm out of them,
> and you should not either, yet.

There is one thing `__activemask()` is not: it is **not** a way to discover
which lanes the compiler *intended* to be active. It reports what is true at the
instant it executes, which on sm_70+ depends on where the hardware happens to
have the warp's lanes. That is the subject of the reconvergence section, and it
is why `__activemask()` is a diagnostic and never an argument you should pass to
a `_sync` primitive.

### Divergence

**Divergence** is when the lanes of one warp need to execute different
instructions. It happens when a branch condition is not uniform across the warp.

The hardware cannot issue two different instructions to one warp in one cycle,
so it executes the paths **sequentially**, once per path, with the mask set to
the lanes that belong to that path:

```cpp
if (cond) A();      // issued with mask = lanes where cond
else      B();      // issued with mask = lanes where !cond
```

Cost is **additive**: `cost(if/else) = cost(A) + cost(B)`, not `max`. Both
bodies are issued, in some order, and during each one the lanes that do not
belong are idle. `example01.cu` part 4 timestamps this directly with `clock64()`
inside a single warp:

```
arm A (lanes  0..15) : cycles [  2483 ..   4587]  duration   2104
arm B (lanes 16..31) : cycles [    28 ..   2430]  duration   2402
interval overlap     :      0 cycles (none)
whole if/else span   :   4559 cycles
control kernel, no divergence, all 32 lanes run one body:
  span               :   2295 cycles
ratio divergent/uniform = 1.99x
```

The `else` arm ran first, the `then` arm second, they did not overlap by a
single cycle, and the whole construct took 1.99x what one arm takes. Both arms
did identical work, so a machine that ran them in parallel would have shown
1.0x.

**Divergence is warp-local.** This is the single most misunderstood point in
CUDA, so state it as a rule and then prove it:

> A branch costs nothing extra if **every lane of the warp agrees**, no matter
> how much the warps disagree with each other.

`tid % 2` and `(tid / 32) % 2` select the same two code paths over the same
data with the same total work. The first splits every warp; the second assigns
whole warps. Measured on this GPU (`example02.cu`, part A, min of 6 sweeps):

| grouping of the branch | ms | vs uniform |
|---|---|---|
| uniform, all lanes take arm A | 0.0634 | 1.000x |
| warp-uniform, `threadIdx.x < 128` | 0.0647 | **1.020x** |
| lane-alternate, `threadIdx.x & 1` | 0.1248 | **1.969x** |
| 8-lane groups, `(threadIdx.x >> 3) & 1` | 0.1242 | 1.960x |

Two readings. First, the warp-uniform branch is free: 2% over no branch at all,
which is the cost of evaluating the predicate. Second, the *number* of divergent
lanes does not appear in the price. A 16/16 split and a split into four groups
of 8 both cost 1.96x, because in both cases the warp must issue both bodies
once. What matters is only whether the warp splits at all.

The generalisation to *n* paths: if the lanes of a warp spread over `n` distinct
bodies, the warp issues all `n`, and the cost is the **sum** of all `n` bodies
while the useful work is one of them. A 32-way `switch (lane)` in which every
lane picks a different arm runs at 1/32 of peak, and the source code gives no
hint of it. This is why `tid % 32` patterns are catastrophic and why
`tid / 32` patterns are free.

Two honest caveats, both measured while writing this module:

- The compiler often rescues you by converting a small `switch` into
  arithmetic. An 8-arm switch whose arms differed only in a constant compiled
  into a single arm with a selected constant — zero divergence, and the timing
  showed it (0.94x). If your arms differ only in data, expect the compiler to
  merge them. Read the SASS before believing an *n*-way divergence claim.
- When the arms genuinely differ, the measured penalty can be below `n`. An
  8-way divergence over eight structurally different bodies (FMA chain, sqrt
  chain, `__sinf`, reciprocal, `__expf`, `__logf`, …) measured **6.1–6.3x**, not
  8x, because the eight bodies use different pipelines (FMA, SFU) and the
  scheduler overlaps them better than it overlaps eight copies of the same
  body. `n` is an upper bound on the penalty, not a law.

### Loop divergence

A loop whose trip count varies per lane is divergence in the cheapest possible
disguise. The warp issues the body until its **longest-running lane** is done,
with a shrinking active mask:

```cpp
int trips = f(lane);
for (int j = 0; j < trips; ++j) body();
```

The warp costs `max(trips)` issues of `body()`. The useful work is
`mean(trips)`. The active mask *decays* across iterations: it starts as the set
of lanes that entered the loop and loses a lane each time one finishes. For a
full warp with `trips = lane + 1`, iteration 0 runs with `0xffffffff`,
iteration 1 with `0xfffffffe`, iteration 2 with `0xfffffffc`, and iteration 31
with `0x80000000` — one lane, one issue, 31 lanes idle. Exercise 1 makes you
predict a less obvious version of this table.

Measured (`example02.cu` part B), trip counts 1..8 with identical total work,
spread inside the warp versus made warp-uniform: **1.65x**, against a model
prediction of `max/mean = 8/4.5 = 1.78x`. The gap is discussed under
Performance below; it is not measurement noise.

### Predication

For a short body the compiler does not emit a branch at all. It computes the
condition into a **predicate register** and attaches the predicate to the
instructions of the body. Every lane issues those instructions; the ones whose
predicate is false discard the result.

This is **not** divergence in the scheduling sense — there is no second issue of
anything, no branch unit involvement, no reconvergence point. It is also not
free: the body is issued once, for everybody, whether or not anybody needs it.
Predication trades "issue the body twice, once per side" for "issue both sides
once, unconditionally". For short bodies that is a win; for long ones it is a
disaster, which is why the compiler has a threshold.

Module 3 measured the bounds guard

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= n) return;
```

as predicated and deferred the details to here. Here they are, from
`cuobjdump -sass` on `example02.exe`, three shapes of the same guard:

**1. The guard is the last thing the thread does** (`guardTail`) — the compiler
emits a *predicated exit*, and nothing else changes:

```
ISETP.GE.AND P0, PT, R4, c[0x0][0x170], PT ;
@P0 EXIT ;
MOV R5, 0x4 ;
IMAD.WIDE R2, R4, R5, c[0x0][0x160] ;
LDG.E.CONSTANT R2, [R2.64] ;
FFMA R7, R2, R7, 1 ;
STG.E [R4.64], R7 ;
EXIT ;
```

One compare, one predicated `EXIT`. No branch instruction, no reconvergence
bookkeeping. Out-of-range lanes stop; the rest carry on with the same code.

**2. Guarded body is one store, work follows the guard** (`guardShort`) — the
store itself is predicated:

```
ISETP.GE.AND P0, PT, R0.reuse, c[0x0][0x178], PT ;
@!P0 MOV R9, 0x40400000 ;
@!P0 IMAD.WIDE R2, R0, R3, c[0x0][0x160] ;
@!P0 STG.E [R2.64], R9 ;
STG.E [R4.64], R7 ;
EXIT ;
```

This is the `@P0 STG` case. Three instructions carry the predicate; all 32 lanes
issue them; only the in-range lanes actually write. Note that the address
computation is predicated too, so an out-of-range lane never forms an
out-of-range address.

**3. Same shape, but the guarded body contains a global load** (`guardLoad`) —
the compiler emits a **real branch**:

```
BSSY B0, 0x160 ;
ISETP.GE.AND P0, PT, R4, c[0x0][0x178], PT ;
@P0 BRA 0x150 ;
   LDG.E.CONSTANT R2, [R2.64] ;
   FFMA R9, R2, R9, 1 ;
   STG.E [R4.64], R9 ;
BSYNC B0 ;
STG.E [R6.64], R11 ;
EXIT ;
```

Body length is not the only thing that decides. A global load costs hundreds of
cycles; issuing one unconditionally for lanes that will throw the result away
is worse than branching around it, so the compiler branches. **None of this
changes the correctness of the guard, and none of it is a reason to skip one.**

### Where the compiler flips

`example02.cu` contains `flipKer<N>` for N = 1..8, 12, 16: an `if (threadIdx.x &
1)` whose two arms are each `N` dependent FFMAs. Disassembling the binary gives
a sharp threshold on this GPU with CUDA 13.2 at `-O3`:

| N per arm | code shape |
|---|---|
| 1 … 7 | fully predicated, no branch |
| 8, 12, 16 | `BSSY` / `BRA` / `BSYNC` — real control flow |

At N = 7 (14 instructions total across both arms):

```
ISETP.NE.U32.AND P0, PT, R5, 0x1, PT ;
@!P0 FFMA R7, R4.reuse, R4.reuse, 1 ;
@P0  FFMA R5, R4, R4, -1 ;
@!P0 FFMA R7, R4.reuse, R7, 1 ;
@P0  FFMA R5, R4, R5, -1 ;
...                                  (7 of each, interleaved)
@!P0 FFMA R7, R4.reuse, R7, 1 ;
@P0  FFMA R7, R4, R5, -1 ;
STG.E [R2.64], R7 ;
```

Both arms are in the instruction stream, interleaved, every lane issues all 14.
At N = 8 the same source becomes:

```
BSSY B0, 0x210 ;
ISETP.NE.U32.AND P0, PT, R4, 0x1, PT ;
@P0 BRA 0x180 ;
   FFMA R3, R2, R2, 1 ;      (8 of them)
   ...
   BRA 0x200 ;
   FFMA R3, R2, R2, -1 ;     (8 of them)
   ...
BSYNC B0 ;
STG.E [R2.64], R5 ;
```

`BSSY B0, <addr>` pushes a reconvergence point onto a hardware stack of
convergence barriers; `BSYNC B0` waits there for the lanes that went the other
way. This is the Volta+ replacement for the pre-Volta `SSY`/`.S` `SYNC`
mechanism; the name changed because the semantics did.

**The threshold is a compiler heuristic, not an architectural rule. Do not build
correctness on it, and re-check it whenever you change toolkit version.**

One consequence worth internalising: **`__activemask()` cannot tell you which
one you got.** In the predicated form the `VOTE` instruction that implements
`__activemask()` is itself predicated, so a predicated-off lane does not execute
it and is not counted. A predicated region and a branched region report the same
mask. Only the SASS distinguishes them.

### Reconvergence, and what changed at Volta

**The classic model (pre-Volta, up to sm_62).** A warp had one program counter
and a hardware stack of divergence tokens. At a divergent branch the hardware
pushed the reconvergence address — the **immediate post-dominator** of the
branch, the first instruction that every path must reach — executed one side,
popped, executed the other, and rejoined. Reconvergence at the post-dominator
was **architecturally guaranteed and immediate**. That guarantee is what made
"warp-synchronous programming" work: because all 32 lanes were at the same PC
whenever the code was not inside a divergent region, you could exchange data
between lanes through shared memory with no synchronization at all, as long as
you marked the array `volatile` so the compiler would not cache it in a
register.

**Independent thread scheduling, sm_70 and later, which includes your sm_89.**
Every thread has its **own program counter and call stack**. The warp still
issues one instruction at a time to a set of lanes, but the hardware chooses
that set, and it may:

- interleave the two sides of a divergent branch, issuing a few instructions of
  one and then a few of the other;
- leave lanes at different points for arbitrarily long;
- **not reconverge at the post-dominator at all.**

Nothing in the ISA requires the lanes to come back together at any particular
instruction. The compiler inserts `BSSY`/`BSYNC` pairs where it wants
reconvergence, and the hardware honours those — but that is a property of the
code the compiler chose to emit, not a guarantee you may assume about the
source you wrote.

Why did NVIDIA give this up? Because the lockstep guarantee made a large class
of programs impossible to write. Under the old model, if lane 0 holds a lock
that lane 1 is spinning on, and both are in the same warp, the warp deadlocks:
the hardware will not schedule lane 0's release until the spin loop of lane 1
exits, and it never will. Independent thread scheduling makes fine-grained
synchronization between threads of the same warp *possible*, at the price of
making implicit synchronization between them *illegal*.

The three consequences you must carry forward:

1. **Every warp-level primitive is now `_sync`-suffixed and takes an explicit
   mask**: `__shfl_sync`, `__shfl_down_sync`, `__ballot_sync`, `__all_sync`,
   `__any_sync`, `__match_any_sync`. The mask names the lanes that must
   participate. The hardware makes them converge for that instruction. The
   mask is how you *create* the synchrony the hardware no longer gives you for
   free. The unsuffixed forms (`__shfl`, `__ballot`, …) are removed, not merely
   deprecated.

2. **`__syncwarp(mask)`** is the explicit convergence barrier for a warp. All
   lanes named in `mask` must reach it. It is also a compiler barrier for
   shared and global memory accesses. It is *not* `__syncthreads()`: it
   synchronizes one warp, not a block. (Module 9 owns block-level barriers,
   memory ordering, and fences; this module only needs `__syncwarp` as the
   warp-scope repair for code that used to rely on implicit lockstep.)

3. **Pre-Volta warp-synchronous programming is broken, not deprecated.** Code
   of the form

   ```cpp
   volatile __shared__ float s[64];
   s[tid] = v;
   // no barrier: "the warp is in lockstep"
   float u = s[tid ^ 1];
   ```

   has no defined behaviour on sm_70+. `volatile` was never a synchronization
   mechanism — it only stops the compiler caching the location in a register.
   It says nothing about *when* another lane's store becomes visible, because
   under the old model "when" was not a question anyone had to ask. Exercise 3
   ships a kernel written this way, and it is wrong on this GPU by 100% of its
   output. (Module 9 develops "`volatile` is not synchronization" in full, for
   block and device scope; here you only need the warp-scope case.)

   **Never present the old idiom as valid.** If you meet it in existing code,
   it is a bug to be diagnosed, and the diagnosis is this module's subject.

A subtlety you must not misread, and which Exercise 3 makes you confront
directly: on this GPU, with this compiler, a *converged-region* warp-synchronous
exchange usually still produces the right answer. `nvcc` emits `BSSY`/`BSYNC` at
immediate post-dominators, and the hardware does reconverge there. That is an
observation about today's code generator. It is not a guarantee, it is not
portable across toolkit versions, and it is not something you may rely on. Code
that is correct only because of the current heuristic is code that will break
silently.

---

## Hardware Mental Model

Put the pieces from Module 1 together.

An Ada SM has **four processing blocks** (sub-partitions). Each has one warp
scheduler, a 16,384-register slice, 32 FP32 lanes, 4 SFUs, 8 LD/ST units, and up
to 12 resident warps (48 per SM = 1536 threads). A warp is assigned to one
processing block for its entire life; `warp_in_block % 4` decides which.

Per clock, each scheduler:

1. looks at its resident warps and marks the **eligible** ones — those whose
   next instruction has no outstanding dependency (scoreboard) and whose
   required functional unit is free;
2. picks **one**;
3. issues its next instruction, with that warp's current active mask, to the
   lanes the mask selects.

Everything in this module follows from step 3 costing one issue slot regardless
of how many bits the mask has.

- **Divergence costs issue slots, not lanes.** An `if/else` split 31/1 costs
  exactly as much as a split 16/16: two issues per instruction pair. The lone
  lane is not cheaper. This is why the measured cost of the 8-lane-group
  grouping equals the cost of the alternating grouping.

- **Divergence does not cost memory bandwidth by itself.** A masked-off lane
  supplies no address, so a divergent load fetches fewer sectors, not more
  (Module 5). What it costs is a second *instruction*, and the second load's 
  addresses are a different, usually equally scattered, set.

- **Idle lanes are not reclaimed.** The 16 masked-off lanes of a 16/16 split are
  not handed to another warp. Lane `L` of warp `W` is hard-wired to a register
  file slice and a datapath slot; there is no mechanism to fill it from
  elsewhere. This is the difference between SIMT and a CPU's out-of-order
  execution, and it is why "the GPU has 5120 cores" is a misleading number
  (Module 1): the machine has 160 independent instruction streams' worth of
  issue, and a divergent warp wastes lanes of one of them.

- **Latency hiding is unaffected by divergence.** A warp that is executing arm A
  with half its lanes masked is still a warp the scheduler can switch away from
  when it stalls. Divergence costs throughput, not the ability to hide latency.
  The two failure modes are independent, and Nsight reports them separately
  (Module 23).

- **Predication moves the cost from the scheduler to the pipeline.** A
  predicated instruction occupies an issue slot and a functional unit and
  produces nothing for the masked lanes. For a two-instruction body that is
  cheaper than the branch bookkeeping; for a two-hundred-instruction body it is
  not, and for a body containing a 500-cycle global load it is not even close.

- **Under ITS the "current active mask" is a property of the hardware's
  scheduling choice**, not of your source. Two lanes at the same source line can
  be at different PCs, and `__activemask()` will say so. The exercise-1 loop
  masks are that fact made visible.

### Why sorting work so lanes agree is a real technique

If the amount of work per element is data-dependent, a thread-per-element
mapping puts every work class in every warp, and every warp pays `max`. If you
first **permute the elements so that elements of the same class land in the same
warp**, every warp pays its own class and the grid pays `mean`.

The speedup is bounded by `max(work) / mean(work)`. It is a real, standard
optimization — ray tracers sort rays by direction, sparse solvers bin rows by
nonzero count, and every production sorting network sorts keys into buckets
before doing per-bucket work. The catch is that the permutation is not free, and
Exercise 2 makes the harness price it: the reordering must either be cheap
enough relative to what it saves, or be amortised over many launches. Report the
break-even launch count, not just the speedup.

---

## Code Walkthrough

### `example01.cu` — making the warp visible

**Part 1** launches `<<<1, 100>>>` and reports, per warp, the tids it contains
and the active mask at the first instruction:

```
  warp 0 : tids   0.. 31  32 active lane(s)  active mask 0xffffffff  popc=32
  warp 1 : tids  32.. 63  32 active lane(s)  active mask 0xffffffff  popc=32
  warp 2 : tids  64.. 95  32 active lane(s)  active mask 0xffffffff  popc=32
  warp 3 : tids  96.. 99   4 active lane(s)  active mask 0x0000000f  popc=4
```

`0x0000000f` is the machine telling you that 28 of warp 3's lanes are gone
before the kernel has executed a single useful instruction.

**Part 2** launches the same 100 threads as `dim3(10,10)` and prints where each
warp begins and ends in `(x,y)`:

```
  warp 0 : first lane (x,y)=(0,0)  last lane (x,y)=(1,3)  [32 lanes]
  warp 1 : first lane (x,y)=(2,3)  last lane (x,y)=(3,6)  [32 lanes]
```

Warp 0 ends mid-row and warp 1 starts mid-row. If your branch condition is
`threadIdx.y < k`, whether it diverges depends on `blockDim.x`, not on `k`.

**Part 3** records `__activemask()` at six labelled points in nested control
flow:

```cpp
    REC(0);                                   // P0: kernel entry
    if (lane < 24) {
        REC(1);                               // P1
        for (int j = 0; j < 40; ++j) a = fmaf(a, 1.0001f, 1e-6f);
        if ((lane & 7) == 0) { REC(2); ... }  // P2
        else                 { REC(3); ... }  // P3
        REC(4);                               // P4
    }
    REC(5);                                   // P5
```

and prints

```
  P0  entry                              : mask 0xffffffff  popc=32
  P1  inside  if (lane < 24)             : mask 0x00ffffff  popc=24
  P2  inside    if ((lane & 7) == 0)     : mask 0x00010101  popc= 3
  P3  inside    else                     : mask 0x00fefefe  popc=21
  P4  after the inner if/else            : mask 0x00ffffff  popc=24
  P5  after the outer if                 : mask 0xffffffff  popc=32
```

Read it as arithmetic on sets: `P2 | P3 == P1`, `P2 & P3 == 0`, `P4 == P1`,
`P5 == P0`. The masks are exactly the lane sets you would compute on paper from
the conditions. The ballot printed above them, `0xaaaaaaaa`, is the same
information obtained *before* the branch — which is what makes ballots useful
for sizing work.

The 40-FFMA bodies are there on purpose: they are long enough to be past the
predication threshold, so these are real branches with real reconvergence
points. Shorten them and the masks do not change, but the SASS does.

**Part 4** is the timing shown earlier. Note how it is built: one warp, both
arms doing the identical 512-FFMA dependent chain, `clock64()` read at the start
and end of each arm, plus a control kernel with no branch at all. Using
`clock64()` inside the kernel sidesteps the whole laptop-clock problem — cycles
are cycles regardless of what frequency the SM is running at.

### `example02.cu` — pricing it

Part A is the four-grouping experiment already quoted. Its construction is the
point: `groupKer<MODE>` is one kernel, the two arms are the same length, the
grid is one resident wave (240 blocks of 256 = 61,440 threads = 40 SMs x 1536),
and only the predicate differs between the four instantiations. Total
arithmetic across the grid is *identical* in all four. Anything the timer shows
is divergence.

Timing follows spec §12: a duration-based 400 ms warm-up to get the clock
ramped, then six sweeps in which all four configurations are timed
back-to-back, 20 iterations each, minimum taken. Validation happens afterwards,
in a separate pass, against a CPU reference computed with `fmaf` in float so
that it follows the device arithmetic exactly. Ratios are the reported quantity;
the absolute milliseconds move by 2x between a cold and a heat-soaked run and
the ratios do not.

Part B, trip-count divergence, is the same idea for loops, and it contains a
trap worth understanding. The obvious "warp-uniform" control — give each *warp*
of a 256-thread block a different trip count — makes the comparison measure the
wrong thing, because warps `w` and `w+4` share a processing block and the trip
counts no longer balance across the four schedulers. The version shipped assigns
the trip count per **block** (`blockIdx.x & 7`) and uses a grid several waves
deep, so the block scheduler rebalances. That took the measured ratio from
1.09x (broken control) through 1.39x (per-warp, rebalanced by hand) to 1.65x
(per-block, deep grid), against a model of 1.78x.

Part C is not run at all. It compiles ten instantiations of `flipKer<N>` and
three shapes of the Module 3 guard into the binary purely so you can
disassemble them. The commands are printed by the program.

---

## Check Your Understanding

Answers in `solutions/module08/check_your_understanding.md`.

**Q1.** A kernel is launched with `dim3(48, 4)` blocks — 192 threads. Inside it,
every thread evaluates `if (threadIdx.y < 1) { A(); } else { B(); }`, and `A`
and `B` are each 200 instructions long, with no predication. How many of the
block's warps diverge, and what is the block's total instruction issue count for
this construct? Now change the block to `dim3(32, 6)` — same 192 threads, same
condition, same code. Answer again. Explain the difference in terms of the
linearization rule, and say what it implies about choosing block shapes when a
branch condition is written on `threadIdx.y`.

**Q2.** Two kernels compute the same thing. Kernel X contains
`if (lane < 16) heavy(); else heavy();` where both calls are the same 300-
instruction function. Kernel Y contains just `heavy();`. Predict the ratio of
their runtimes and justify it. Then predict what happens to that ratio if
`heavy()` is reduced to three instructions, and explain which mechanism takes
over.

**Q3.** You are told that a warp executed a particular `__activemask()`
instruction and got `0x0000ffff`. List everything this does and does not tell
you about the other 16 lanes. In particular: can you conclude that they are
executing a different instruction right now? Can you conclude that they will
rejoin you at the end of the enclosing `if`? Can you safely pass `0xffffffff`
to a `__shfl_sync` on the next line?

**Q4.** The pre-Volta idiom below was correct on Kepler and is undefined on
sm_89. Identify *both* independent reasons it can fail, and explain why fixing
only one of them (adding `volatile`, say) leaves a program that may pass every
test you run and still be wrong.

```cpp
__shared__ int s[32];
s[lane] = value;
int neighbour = s[(lane + 1) & 31];
```

---

## Exercises

### Exercise 1 — `exercise01.cu` : lane-level prediction

**What the program must accomplish.** One block of 32 threads runs a kernel with
nested divergent control flow, an early `return` inside one arm, a short
`if/else`, and a loop whose trip count is `(lane & 7) + 1`. Eleven labelled
sites record `__activemask()`; the loop records it per iteration. The kernel also
maintains a per-lane execution counter at every site and a hardware ground-truth
**issue counter** incremented by exactly one lane per issue. You predict all the
masks and supply a model that reproduces the issue counts.

```
nvcc -arch=sm_89 -O3 -o exercise01.exe exercise01.cu
.\exercise01.exe

nvcc -arch=sm_89 -O3 -c -o exercise01.o exercise01.cu
cuobjdump -sass exercise01.o > exercise01.sass
```

**TODOs.**
1. The masks at P0..P4 — the `lane < 20` sub-tree, including both arms of the
   inner branch and their reconvergence point.
2. The masks at P5..P8 — the `else` sub-tree, the site inside the early
   `return`, and the site after the whole outer branch. P8 is not what it looks
   like.
3. The masks at P9 and P10, the two arms of a one-instruction `if/else`.
4. The mask on each of the eight iterations of the variable-trip loop.
5. **Design.** Implement `warpIssuesFromLaneCounts()`: given only the per-lane
   execution-count table, compute how many instruction issues the warp spent at
   each site. Derive the rule from the SIMT contract. It must reproduce the
   hardware counters for every site, including the loop.

**Validation.** Each mask is scored MATCH/MISMATCH against the recorded value;
the issue model must agree with the hardware counter at every site and in
total; the program prints `SCORE: n/20` and `OVERALL: PASS` only on 20/20.

### Exercise 2 — `exercise02.cu` : make the lanes agree

**What the program must accomplish.** One million elements; element `i` needs
`work[i]` rounds of refinement with `work[i]` in 1..8, drawn from a
deterministic index-derived hash. The naive kernel maps thread `t` to element
`t`, so every warp holds all eight classes and pays `max`. You must make the
warps agree and beat the naive kernel by at least 1.30x while producing
bit-identical output.

```
nvcc -arch=sm_89 -O3 -o exercise02.exe exercise02.cu
.\exercise02.exe
```

**TODOs.**
1. **Design.** `buildPlan(work, n, order)` — fill `order` with whatever your
   strategy needs. The harness constrains only the result: `out[i]` must end up
   holding `refine(x[i], work[i])` for every `i`. Nothing tells you which
   transformation to use; more than one works and they do not cost the same.
2. The kernel that consumes your plan. Getting the output index wrong produces
   a fast kernel with silently permuted results; the harness compares
   bit-exactly against the naive output.
3. The launch configuration. It interacts with TODO 1: a plan that groups
   elements is worth nothing if the block shape re-splits the groups.
4. A predicted speedup, committed before running, derived from the work
   distribution and the SIMT cost model. Scored within ±20%.

**Validation.** The naive kernel is cross-checked against a CPU reference on a
sample; your kernel is compared bit-exactly against the naive kernel on all
1,048,576 elements; warp homogeneity is measured and printed for both mappings;
the timing follows spec §12 and reports the ratio; the host cost of your
preparation is measured and turned into a break-even number of launches.
`OVERALL: PASS` requires correctness, the prediction within ±20%, and a speedup
above 1.30x.

### Exercise 3 — `exercise03.cu` : a kernel that used to be idiomatic

**What the program must accomplish.** `warpSmoothBroken` is a per-warp ring
rotation written in the pre-Volta warp-synchronous style: `volatile __shared__`,
one `__syncthreads()` to publish the initial load, and no warp-level
synchronization, because "the lanes of a warp are in lockstep". It is wrong on
this GPU. `compute-sanitizer --tool memcheck` is clean. You diagnose it and
write a correct version.

```
nvcc -arch=sm_89 -O3 -o exercise03.exe exercise03.cu
.\exercise03.exe

compute-sanitizer --tool memcheck  .\exercise03.exe
compute-sanitizer --tool racecheck .\exercise03.exe
nvcc -arch=sm_89 -O3 -c -o exercise03.o exercise03.cu
cuobjdump -sass exercise03.o > exercise03.sass
```

**TODOs.**
1. **Design.** Write `warpSmoothFixed`. The requirements are stated as
   requirements, not as an API call: every lane must read the value its
   neighbour held at the *start* of the step; whatever mechanism you use to
   guarantee that must be reached by a warp-uniform set of lanes; and the
   per-lane refinement must still happen, so deleting the divergence is not an
   answer. Two quite different correct solutions exist.
2. The active mask recorded immediately before the exchange in each arm of the
   broken kernel.
3. Whether the broken kernel's wrong answer is the *same* wrong answer on every
   run — and therefore which debugging tools are useful.
4. Two claims to mark true or false: that `volatile` is what makes the
   warp-synchronous idiom correct, and that a converged-region variant which
   passes today is therefore guaranteed by the programming model.

**Validation.** The broken kernel is run ten times and its mismatch count is
reported along with whether it varies; your fixed kernel must match the CPU
reference exactly, and must do so on ten consecutive runs; TODOs 2–4 are scored;
`SCORE: 7/7` is required for `OVERALL: PASS`.

---

## Prediction

Commit to these in writing before you run anything.

1. **The grouping experiment.** `example02.cu` part A runs the same branch over
   the same data under four groupings. Write down all four ratios relative to
   the uniform case. In particular commit to a number for the 8-lane-group case
   *before* you see it, and say whether you expect it to be above, below, or
   equal to the 16/16 alternating case, and why.

2. **The guard.** Module 3 told you the bounds guard is predicated. Before you
   disassemble `guardTail`, `guardShort` and `guardLoad`, predict which of the
   three the compiler will predicate and which it will branch, and state the
   criterion you think it is using. Then check, and revise your criterion if it
   was wrong.

3. **Exercise 3's converged variant.** `warpSmoothConverged` performs the same
   `volatile __shared__` exchange with no warp-level synchronization, but in a
   region with no divergence. Before running, predict whether it produces the
   right answer on this GPU, and separately predict whether the CUDA programming
   model guarantees that it will. These are two different questions and they
   have two different answers.
