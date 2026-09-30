# Module 08 — Check Your Understanding, answers

---

## Q1 — block shape decides whether a `threadIdx.y` branch diverges

**Setup.** 192 threads, `if (threadIdx.y < 1) A(); else B();`, `A` and `B` each
200 instructions, no predication. 192 threads = 6 warps either way.

### Block `dim3(48, 4)`

The linearization rule gives `tid = threadIdx.x + 48 * threadIdx.y`, so
`threadIdx.y = tid / 48`. The condition is true exactly for `tid` in `[0, 48)`.
Warp `w` owns `tid` in `[32w, 32w+32)`:

| warp | tids | `threadIdx.y` values | branch |
|---|---|---|---|
| 0 | 0..31 | 0 | A only — uniform |
| 1 | 32..63 | 0 (tids 32..47), 1 (tids 48..63) | **both — diverges** |
| 2 | 64..95 | 1 | B only — uniform |
| 3 | 96..127 | 2 | B only — uniform |
| 4 | 128..159 | 2, 3 | B only — uniform |
| 5 | 160..191 | 3 | B only — uniform |

**One warp diverges.** Note warp 4 spans two distinct `threadIdx.y` values and
still does not diverge: what matters is whether the *condition* differs, not
whether the index does.

Issue count: five uniform warps issue 200 each = 1000. Warp 1 issues both arms
= 400. **Total 1400 warp-instruction issues.**

### Block `dim3(32, 6)`

Now `tid = threadIdx.x + 32 * threadIdx.y`, so `threadIdx.y = tid / 32 = warp`.
Warp `w` contains exactly the threads with `threadIdx.y == w`, so the condition
is constant within every warp.

**Zero warps diverge. Total 1200 issues** — 14% fewer, for identical work and
identical source.

### What it implies

A condition written on `threadIdx.y` is warp-uniform if and only if
`blockDim.x` is a multiple of 32 (so that each warp lies within a single `y`)
or the condition's boundary happens to fall on a multiple of 32. With
`blockDim.x = 48` the boundary at `tid = 48` splits warp 1; with
`blockDim.x = 32` every `y` is a whole number of warps.

This is the same rule as Module 3's "give `threadIdx.x` to the axis with the
smallest memory stride", arriving from the execution side instead of the memory
side, and it usually agrees with it: making `blockDim.x` a multiple of 32 is
good for coalescing *and* makes `y`- and `z`-conditions warp-uniform for free.
When the two rules conflict, measure — but they rarely do.

---

## Q2 — the same body on both sides

**Kernel X:** `if (lane < 16) heavy(); else heavy();` with `heavy()` 300
instructions. **Kernel Y:** `heavy();`

**Predicted ratio: 2.0x.** The condition is not warp-uniform, so the warp splits
and issues `heavy()` twice — once with mask `0x0000ffff` and once with mask
`0xffff0000`. Cost is additive, not `max`, and it is additive in *issues*, so
the fact that the two bodies are textually identical is irrelevant: they are two
separate regions of the instruction stream with two separate reconvergence
points. The measured value in `example02.cu` part A for exactly this shape is
**1.96x**; the missing 4% is the one-time predicate evaluation and branch
bookkeeping amortised over 300 instructions, plus the fact that the two halves
occasionally overlap in the pipeline.

The trap this question exists for: a reader who thinks the compiler will notice
the two arms are identical and merge them. It does not, and you can check —
`flipKer` in `example02.cu` has two arms differing only in a sign and the
compiler emits both. (It *does* sometimes merge arms that differ only in a
constant operand; see the honest caveat in the lesson. Merging control flow into
arithmetic is a different transformation from noticing two blocks are equal.)

**With `heavy()` reduced to three instructions:** the mechanism changes. The
compiler is now below its if-conversion threshold (measured at 7 instructions
per arm on this GPU with CUDA 13.2 at `-O3`), so it emits no branch at all: six
predicated instructions, each issued once by the whole warp. The *ratio against
kernel Y is still about 2x* — six issues where Y needs three — but the constant
in front of it is now three instructions instead of 300, so in any real kernel
the difference disappears into the noise, and the branch-unit and reconvergence
overhead that made the 300-instruction case expensive is gone entirely.

The point: the 2x does not disappear, it changes mechanism. Divergence pays
`2 x body` in issue slots through the branch unit; predication pays `2 x body`
in issue slots through the arithmetic pipes. What predication buys is the
removal of the branch overhead, which is why it wins for tiny bodies and loses
for big ones.

---

## Q3 — what `__activemask() == 0x0000ffff` tells you

**It tells you exactly one thing:** at the instant that `VOTE` instruction
executed, the 16 lanes 0..15 executed it together, and lanes 16..31 did not.

**What it does not tell you.**

- **It does not tell you that lanes 16..31 are executing a different
  instruction right now.** They may be anywhere: in the other arm, already past
  the join, stalled on a memory dependency, or never created at all (a partial
  warp at the block boundary — a 20-thread block reports `0x000fffff` at every
  site).
- **It does not tell you that they will rejoin you.** Under independent thread
  scheduling there is no architectural guarantee of reconvergence at the
  immediate post-dominator. `nvcc` emits `BSSY`/`BSYNC` pairs and the hardware
  honours those, so in practice you usually do reconverge — but that is a
  property of the code the compiler chose to emit, and a different optimisation
  decision, a different toolkit version, or a `return` inside the other arm
  changes it.
- **It does not tell you the value is stable.** Two lanes at the same source
  line can observe different masks. The variable-trip loop in Exercise 1 is one
  demonstration; a loop that the compiler peels or unrolls into a main body plus
  a remainder gives another, because the "same source line" becomes two
  distinct SASS sites.

**Can you pass `0xffffffff` to `__shfl_sync` on the next line?** No. Naming a
lane in the mask asserts that it will arrive at that instruction. If it does
not, the behaviour is undefined: measured on this GPU, a `__shfl_xor_sync`
called with a full mask from inside a half-warp branch returned wrong data for
**49.95%** of elements — every odd lane got its own value instead of its
partner's. In other shapes it can hang.

**Can you pass the harvested `__activemask()` instead?** Also no, and this is
the subtler error. The mask you harvest is a snapshot of an implementation
decision. If it happens to be `0x0000ffff` on one run and `0x000000ff` on the
next because the scheduler split the group differently, your algorithm silently
computes over a different set of lanes. The correct discipline is to name the
lanes you *require* — a compile-time or logically-determined mask — and make
the control flow guarantee they are there. Module 30 develops this into the
standard idiom.

---

## Q4 — the two independent reasons the pre-Volta idiom fails

```cpp
__shared__ int s[32];
s[lane] = value;
int neighbour = s[(lane + 1) & 31];
```

### Reason 1 — the compiler (a visibility problem)

Without `volatile`, `s[]` is an ordinary object. The compiler may keep
`s[lane]` in a register and never issue the `STS`, or may hoist the `LDS` of
`s[(lane+1)&31]` above the `STS`, because within the abstract machine of a
single thread the two accesses are to different locations and there is no
ordering constraint between them. This is a *compile-time* failure: the
instructions you assumed exist are not in the binary. This is what `volatile`
was there to prevent, and for that narrow job it works.

### Reason 2 — the hardware (an ordering problem)

Even if both instructions are emitted, in program order, nothing makes lane `L`
execute its `STS` before lane `L+1` executes its `LDS`. Under the pre-Volta
model that was free: one PC per warp meant that when the warp was converged, all
32 lanes executed the `STS` in the same cycle and the `LDS` in the next. Under
independent thread scheduling each lane has its own PC and the hardware may run
them in any interleaving it likes. This is a *run-time* failure, and `volatile`
has nothing to say about it — `volatile` constrains the compiler, not the
scheduler.

### Why fixing only one leaves a program that passes every test

Add `volatile` and both instructions appear in the binary. Run it: on this GPU,
in a region where the compiler has not split the warp, `nvcc` emits `BSSY`/
`BSYNC` at post-dominators and the hardware reconverges, so the lanes really are
together and the answer really is right. Exercise 3 ships exactly that kernel
(`warpSmoothConverged`) and the harness reports **0 wrong elements out of
524,288**.

That is the worst possible outcome, because it is an accident. The same source,
with the exchange moved inside a divergent `if/else`, is wrong for **100%** of
its output — also in Exercise 3. Nothing in the language distinguishes the two
cases; the difference is entirely in what the code generator decided to do. A
program that is correct because of a heuristic is a program that breaks on the
next toolkit release, at a customer site, with no source change to blame.

### The fix

Make the requirement explicit rather than inherited:

```cpp
__shared__ int s[32];           // no volatile needed
s[lane] = value;
__syncwarp();                   // all 32 lanes; also a compiler barrier
int neighbour = s[(lane + 1) & 31];
```

or, better where it applies, take shared memory out of the picture entirely and
use a register-to-register exchange with an explicit mask (Module 30). Either
way the mechanism is *named in the source*, which is the whole point: the
Volta-and-later model did not remove a capability, it removed an assumption you
were never allowed to write down.
