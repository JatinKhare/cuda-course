# Module 04 / Exercise 03 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise03_solution.exe exercise03_solution.cu
.\exercise03_solution.exe
```

Runtime is about 4 s, almost all of it in the one combination that is
pathologically slow — which is itself the answer to the exercise.

## TODO 1 — the table

```cpp
__constant__ float c_tab[NC];
...
CHECK(cudaMemcpyToSymbol(c_tab, h_tab, NC * sizeof(float)));
```

The requirement stated in the file was: 64 floats, device-wide visibility, never
written by device code, contents supplied by the host, *and placed so that a
warp in which all 32 lanes request the same element is serviced in a single
operation*. That last clause is the definition of the constant cache's broadcast
path, and `__constant__` is the only way to ask for it.

Two API details worth stating precisely:

- `cudaMemcpyToSymbol` takes the **symbol**, not its address. `c_tab` here is not
  a host pointer that you may dereference; the runtime performs a symbol lookup
  to find the device-side location. `cudaMemcpyToSymbol(&c_tab, ...)` compiles
  under some toolchains and is wrong.
- There is no `cudaMalloc` for constant memory. The 64 KB window is carved at
  module-load time from the compiled symbols. It is a static resource; `nvcc`
  reports your usage as the `cmem[3]` figure in `-Xptxas -v`.

## TODO 2 — the accessor

```cpp
template <int SPACE>
__device__ __forceinline__ float coef(int i, const float* __restrict__ g)
{
    if (SPACE == SPACE_CONSTANT) return c_tab[i];
    else                         return g[i];
}
```

The `if` tests a template parameter, so it is resolved at compile time and no
branch survives into SASS. Each instantiation contains exactly one load. The
`SPACE_CONSTANT` instantiation emits a constant-bank access — and where the
index is also a literal, ptxas does not emit a load at all, it folds the
`c[0x3][off]` operand directly into the `FFMA`. The `SPACE_GLOBAL_RO`
instantiation emits `LDG.E.CONSTANT`, the read-only-cache form of a global load,
which it is allowed to use because `g` is `const __restrict__`.

## TODO 3 / TODO 4 — the answers

```cpp
int choiceUniform     = SPACE_CONSTANT;
int choiceLaneVarying = SPACE_GLOBAL_RO;
double predRatioUniform     = 3.0;
double predRatioLaneVarying = 30.0;
```

### The reasoning you were meant to produce, before measuring

Count **distinct addresses presented by one warp per instruction**.

**Uniform pattern**: 1 distinct address per warp per access. The constant cache
is designed for exactly this: one tag lookup, one broadcast to 32 lanes, and
often no load instruction at all because the address is a literal that becomes
an instruction operand. The global read-only path must still issue an `LDG`,
which occupies an LSU slot and a cache lookup per warp even though all lanes
want the same line. Predict constant memory wins by a small integer factor —
call it 2–4×.

**Lane-varying pattern**: 32 distinct addresses per warp per access. The global
path handles this natively: 32 lanes × 4 B = 128 B, which is a single cache line
and a single transaction — the *best possible* case for a coalesced load
(Module 5). The constant cache has **no wide path**. Its datapath assumes one
address, so the hardware must **replay the instruction once per distinct
address**: up to 32 serialized accesses for one instruction. Predict constant
memory loses by something on the order of 32× — the warp width.

That is why the answer is not "constant memory is fast". The pairing of *space*
and *access pattern* is what is fast or slow. A single 256-byte table is the
right choice for constant memory under one indexing scheme and a 28× pessimization
under the other.

### Measured

```
pattern        space                ms    correct
uniform        constant         1.1575        yes
uniform        global-ro        4.3809        yes
lanevarying    constant       100.1370        yes
lanevarying    global-ro        4.0760        yes
```

- Uniform: constant **3.8×** faster than global-ro. Predicted 3.0.
- Lane-varying: constant **24.6×** *slower* than global-ro. Predicted 30.

The lane-varying constant number is the one to remember: **100 ms for a kernel
that does 4 ms of work**, because a 256-byte table was put one level too clever.
The replay factor measured at ~25 rather than a clean 32 because the four warp
schedulers can overlap other warps' replays with each other, and because the
inner loop was unrolled by 8 so some address reuse survives.

Also read the *uniform/global-ro* row against the *lane-varying/global-ro* row:
4.38 vs 4.08 ms. Global memory does not care which of the two patterns you use,
because both are a single 128 B transaction per warp. Constant memory cares by a
factor of 86.

## Synchronization / memory reasoning

None required: the table is read-only for the entire kernel, so there is no
ordering question and no race. That is precisely the precondition for
`__constant__` and for the read-only path — both are architecturally read-only
for the duration of a kernel launch. If any thread wrote the table, neither
mechanism would be legal: `__constant__` is not writable from device code at
all, and the read-only cache has no coherence with the write path, so a write
through a different pointer would not be visible. The compiler's `__restrict__`
promise is what lets it assume this, and breaking that promise is undefined
behaviour, not a slowdown.

## Performance reasoning

The kernel is deliberately arithmetic-heavy relative to its memory footprint
(64 table reads × 64 repetitions per thread, one global load and one global
store) so that the table access dominates and the DRAM traffic does not. This is
a microbenchmark shape, not a realistic kernel shape — but the *ratio* it
exposes is exactly what you would see inside the inner loop of a real stencil,
FIR filter, or small dense matrix transform.

Where the broadcast rule bites in practice:

| Data | Index | Correct home |
|---|---|---|
| Convolution/filter coefficients | loop counter (uniform) | `__constant__` |
| Problem dimensions, strides, scale factors | none (scalar) | `__constant__`, or just a kernel argument (also `cmem`) |
| Camera / transform matrix | uniform, or `blockIdx`-derived | `__constant__` |
| Per-class bias in a classifier | `threadIdx`-derived | global + `const __restrict__` |
| Gather/scatter lookup table | data-dependent | global + `const __restrict__` |
| Small table reused by every thread, indexed per-thread | per-thread | shared memory (Module 6), not constant |

Note the last row: when the index varies across lanes *and* the table is reused
heavily, the right answer is usually neither of the two spaces this exercise
offered — it is shared memory, which does have a 32-wide datapath. Module 6
introduces it and Module 7 explains why even shared memory has a version of this
problem (bank conflicts) with a very similar replay cost.

## Expected output

Actual run, RTX 3500 Ada Laptop GPU. The two fast rows vary by a few percent run
to run; the slow row is stable at 100 ± 1 ms.

```
=== NVIDIA RTX 3500 Ada Generation Laptop GPU : constant memory window = 65536 B ===

pattern        space                ms    correct
uniform        constant         1.1575        yes
uniform        global-ro        4.3809        yes
lanevarying    constant       100.1370        yes
lanevarying    global-ro        4.0760        yes

--- your answers ---
  uniform     : chose constant  (best is constant ) ratio predicted 3.0, measured 3.8
  lanevarying : chose global-ro (best is global-ro) ratio predicted 30.0, measured 24.6

PASS
```

## The result that matters

`__constant__` is not a synonym for "fast read-only memory". It is a **broadcast**
mechanism: it is the fastest thing on the chip when all 32 lanes of a warp want
the same address, and roughly a warp-width penalty when they do not, because the
hardware serially replays the instruction once per distinct address. Before you
put a table in constant memory, ask one question — *within a warp, at a single
instruction, how many distinct elements are being requested?* If the answer is
not 1, use global memory with `const __restrict__`.

**Variation to try:** change the lane-varying index from `(base + j) & (NC-1)`
to `(base/32 + j) & (NC-1)`, so that all 32 lanes of a warp again share an index
but different warps use different ones. Predict the constant-memory time before
you run it. Measured: **100.14 ms → 3.82 ms**, a 26× recovery from changing one
`/ 32`. It does not reach the 1.16 ms of the fully uniform kernel, because the
index is now warp-uniform but not a compile-time literal, so ptxas must emit a
real constant-cache load rather than folding the address into the `FFMA`
operand. The point: the broadcast requirement is per *warp*, not per block and
not per grid — uniformity across the 32 lanes is the only thing the hardware
cares about, and you can often restructure an index to get it.
