# Module 03 / Exercise 02 — Solution notes

**Do not read this until you have submitted your own attempt.**

## Compile / run

```
nvcc -arch=sm_89 -O3 -o exercise02_solution.exe exercise02_solution.cu
.\exercise02_solution.exe
```

## The decomposition, stated before any code

The loop nest is

```cpp
for (b = 0; b < B; ++b)
  for (r = 0; r < R; ++r)
    for (c = 0; c < C; ++c)
      out[(b*R + r)*C + c] = 0.5*in[(b*R+r)*C + c]
                           + 0.25*in[(b*R+r)*C + (c+1)%C]
                           + 0.25*in[((b+1)%B*R + r)*C + c];
```

Three facts decide the whole design.

1. **The iterations are independent.** No iteration reads an element of `out`,
   and no iteration writes `in`. Therefore one thread per iteration is legal
   with no synchronization of any kind.
2. **The strides are 1 (c), C (r), R·C (b).** The layout is `[B][R][C]`
   row-major, so `c` is the fast axis.
3. **Lanes of a warp differ in the linearized thread index, x fastest.**
   Therefore `threadIdx.x` must carry `c`. Everything else follows.

So: **one thread computes exactly one output element**, x → c, y → r, z → b.

## TODO 1 — the kernel

```cpp
__global__ void batched_blend(const float* __restrict__ in,
                              float* __restrict__ out,
                              int b_, int r_, int c_)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (c >= c_ || r >= r_ || b >= b_) return;   // all three axes

    int cNext = c + 1; if (cNext == c_) cNext = 0;
    int bNext = b + 1; if (bNext == b_) bNext = 0;

    size_t base     = ((size_t)b     * r_ + r) * c_;
    size_t baseNext = ((size_t)bNext * r_ + r) * c_;

    out[base + c] = 0.50f * in[base + c]
                  + 0.25f * in[base + cNext]
                  + 0.25f * in[baseNext + c];
}
```

Points that are decisions, not typing:

- **Guard all three axes.** `C = 577` is not a multiple of 128, `R = 1013` is not
  a multiple of 2, `B = 37` is not a multiple of anything. Guarding two of three
  leaves the third free to run past the end. The harness would report a FAIL
  (and `compute-sanitizer` an out-of-bounds store) but you should be able to
  predict which one before running it.
- **`(c+1) % C` becomes a compare.** `c` is in `[0, C)`, so `c+1` is in `[1, C]`
  and can exceed the range by at most one. `if (cNext == c_) cNext = 0;` is
  exactly equivalent and is one predicated instruction. The `%` form compiles to
  an integer division sequence (NVIDIA GPUs have no hardware integer divider; the
  compiler emits a reciprocal-multiply plus fixups, roughly 10–20 instructions
  for a non-constant divisor). Two of those per thread over 21.6 M threads is
  real arithmetic — though on this memory-bound kernel it is hidden. Using `%`
  is not wrong; knowing it is not free is the point.
- **`size_t` for the base offsets.** `b * r_ + r` peaks at 37·1013 ≈ 37 k and
  times 577 is 21.6 M — fits in `int` here. The cast is a habit that costs one
  extra register and protects you the day `B` becomes 4096. Cast *before* the
  multiply, not after.
- **`__restrict__` on both pointers.** Promises no aliasing between `in` and
  `out`, which lets the compiler keep the three loads in flight simultaneously
  instead of ordering them against the store. Module 20 returns to this.

## TODO 2 — the block

```cpp
dim3 block(128, 2, 1);      // 256 threads
```

- `blockDim.x = 128` is 4 warps' worth of contiguous columns. No warp ever
  straddles a row boundary: with `tid = tx + 128*ty`, warps 0–3 are `ty == 0` and
  warps 4–7 are `ty == 1`, each covering 32 consecutive `c`.
- 256 threads/block → 1536/256 = **6 blocks resident per SM**, under the 24-block
  cap, 100% of the thread slots. A 1024-thread block would cap at 66.7% because
  only one fits in 1536 slots.
- `blockDim.z = 1` with `gridDim.z = B`: the batch axis is expressed through the
  grid rather than the block. Putting batches inside a block would be worse —
  threads in the same block would touch memory `R*C*4` = 2.3 MB apart, and a
  block is the unit that shares an L1.
- Why not `(256,1,1)`? `C = 577`, so `ceil(577/256) = 3` blocks in x cover 768
  columns — 33% of the x threads wasted. `(128,2)` covers 640 columns in 5
  blocks: 10% waste. Measured threads-per-element: 1.110.

## TODO 3 — the grid

```cpp
dim3 grid( (unsigned)(C / (int)block.x) + (unsigned)((C % (int)block.x) != 0),
           (unsigned)(R / (int)block.y) + (unsigned)((R % (int)block.y) != 0),
           (unsigned)(B / (int)block.z) + (unsigned)((B % (int)block.z) != 0) );
```

→ `(5, 507, 37)`.

The 65535 limit on `gridDim.y` and `gridDim.z` is the constraint to check here.
`R = 1013` and `B = 37` are comfortably under it, but the general lesson is:
**the large axis belongs on x**, because only `gridDim.x` goes to 2³¹−1. Had the
batch been 100,000, a 3-D grid with `gridDim.z = B` would fail to launch with
`cudaErrorInvalidConfiguration`, reported by the `cudaGetLastError()` immediately
after the launch — which is why Module 2 insisted on that check.

## Synchronization / memory reasoning

None required, for the reason given above: disjoint writes, read-only input.
If the kernel were in-place (`out == in`), thread `(b,r,c)` would read
`in[b][r][c+1]`, which thread `(b,r,c+1)` is concurrently overwriting. There is
no block-level barrier that fixes that — the two threads may be in different
blocks, and there is no grid-wide barrier in an ordinary launch. The only fixes
are double-buffering (what this kernel does) or a cooperative launch (Module 29).

Memory traffic: the kernel issues three loads and one store per element. The
`base + c` and `base + cNext` loads are the same cache line for 31 of every 32
lanes, so the second is nearly free. The `baseNext + c` load is a *third
independent stream*, 2.3 MB ahead in the address space; with 86.5 MB arrays and
a 48 MB L2, roughly a third of those hit. Compulsory traffic is 8 B/element
(read the array once, write it once); the measured 267.8 GB/s against a
compulsory-traffic model means actual bus traffic is substantially higher than
compulsory.

## Performance reasoning

Measured (run-to-run variation ~2%):

| Decomposition | launch | ms | GB/s (compulsory model) | % of peak |
|---|---|---|---|---|
| x→c, y→r, z→b (this solution) | `<<<(5,507,37),(128,2,1)>>>` | 0.6459 | 267.8 | 62.0% |
| x→r, y→c, z→b | `<<<(8,289,37),(128,2,1)>>>` | 2.2541 | 76.8 | 17.8% |

**3.5× slower, same thread count, same block shape, still PASS.** That is the
exercise. Hand `threadIdx.x` to the row axis and lanes become `C = 577` floats
apart; one store instruction turns from 4 sectors into 32. The validation cannot
catch it because the answer is *right*. Only the clock notices.

Why 62% of peak rather than ~95%: three read streams (one of them 2.3 MB ahead)
plus a write stream, against a compulsory model that only counts one read and one
write. The kernel is memory-bound and near the achievable ceiling for its actual
traffic; it is the model in the denominator that is optimistic. Module 21
(roofline) formalises "achievable" versus "peak".

## Expected output

```
=== Module 3 / Exercise 2 SOLUTION : batched blend, B=37 R=1013 C=577 (21626537 elements, 86.5 MB) ===

launch <<< (5,507,37), (128,2,1) >>>  = 256 threads/block, 24011520 threads total
  threads per element = 1.110
  VALIDATION        PASS  (21626537 elements)
  TIME                0.6459 ms     267.8 GB/s effective    62.0% of peak

OVERALL: PASS
```

Shipped with TODOs unfilled, the program prints
`Set TODO 2 (block) and TODO 3 (grid) first; a dim3 component is still 0.`
and exits 0 without launching.

## The result that matters

Porting a loop nest is not "put the outer loop on `blockIdx` and the inner loop
on `threadIdx`". It is a stride-matching problem: enumerate the memory stride of
each loop index, sort them, and give `threadIdx.x` the smallest. Every other
assignment produces a program that passes every test you can write and runs
several times slower — a category of bug that unit tests structurally cannot
find, and the reason this course spends Modules 3, 5 and 21 on it.

Variation to try: put the batch axis on `threadIdx.z` instead of `blockIdx.z`
(e.g. `block(64,1,4)`, `grid(ceil(C/64), R, ceil(B/4))`) and explain the
slowdown in terms of what a single block's L1 working set now has to hold.
