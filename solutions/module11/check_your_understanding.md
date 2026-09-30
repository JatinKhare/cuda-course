# Module 11 — Check Your Understanding, answers

Read these only after you have written your own answers down.

---

## Q1 — the in-place normalization that is already finished

> A colleague reports that their in-place normalization kernel,
> `x[i] = (x[i] - mean) * inv_std` over 200M floats, achieves 250 GB/s and asks
> how to get it to 376. Give the number they should have computed, the number
> they actually achieved, and what you would tell them to do. Then describe a
> *different* kernel, also a single load and a single store per element in the
> source, for which 250 GB/s really would indicate a defect, and say what
> distinguishes the two cases.

**The number they computed.** Their source has one load and one store, so they
counted 8 B per element: 200e6 × 8 = 1.6 GB, and they divided that by their
measured time to get 250 GB/s.

**The number they should have computed.** `x` is read *and* written. The
read-modify-write moves 12 B per element:

- read `x[i]` — 4 B
- write `x[i]` — 4 B
- and the read is a compulsory DRAM fetch of the old value, which is the third
  4 B their count is missing. Concretely: the array is 800 MB, so the fetch
  cannot come from the 48 MB L2.

2.4 GB rather than 1.6 GB. Their achieved bandwidth is therefore
250 × 12/8 = **375 GB/s**, which is 87% of the 432 GB/s peak and exactly the
streaming rate this GPU produces.

**What you tell them.** The kernel is finished. There is nothing to do. If they
want it faster the only remaining moves are algorithmic: fuse it with whatever
produces `x` or whatever consumes it, so the array does not round-trip to DRAM
at all, or move to a narrower dtype. Both change the traffic; nothing that
leaves the traffic at 12 B per element can help.

**The contrasting kernel.** Anything where the load and the store go to
*different* arrays and the write covers whole sectors:
`y[i] = (x[i] - mean) * inv_std`, out of place. That really is 8 B per element,
so 250 GB/s really would be 58% of achievable, and the first thing to check is
the launch configuration (is the grid large enough to keep ~116 kB in flight?)
and then the access width.

**What distinguishes them.** Whether the output array is also an input. It is a
property of the *dataflow*, not of the source syntax — both kernels have exactly
one subscripted load and one subscripted store. This is why the traffic count
has to be done on the dataflow graph and cannot be done by counting `[` in the
source.

A third case worth having in your head: `y[2*i] = (x[i] - mean) * inv_std`,
which has one load and one store and moves **16 B** per element, because the
strided store covers half of each sector and triggers a read-modify-write of the
output as well.

---

## Q2 — 5,120 fat threads versus 40,960 thin ones

> A uses 5,120 threads and `float4` loads with 4 elements per thread. B uses
> 40,960 threads and scalar loads with 1 element per thread. Both move the same
> compulsory bytes. Predict their relative speed on this GPU and justify it from
> the outstanding-request argument. Then name one change to the *problem* (not
> the code) that would make A clearly better than B, and one that would make B
> clearly better than A.

**Prediction: within a few percent of each other, with B very slightly ahead.**

Count requests in flight per SM. A has 5,120/40 = 128 threads per SM, each
issuing one 16-byte request per stream, i.e. 128 × 16 = 2 kB per stream per SM
per issue round. B has 40,960/40 = 1,024 threads per SM, each issuing one 4-byte
request: 1,024 × 4 = 4 kB. Multiply by the compiler's unroll factor (5 for a
grid-stride loop of this shape on this compiler) and both are comfortably past
the ~116 kB the whole device needs in flight. **Both saturate; neither can go
faster than the bus.**

`example02.cu` Part B measures exactly this pair and gets 389.5 GB/s for the
`float4`-at-1-block/SM configuration against 386.4 for scalar-at-8-blocks/SM —
a 0.8% difference, which is noise. The two are interchangeable *at this problem
size on this GPU* precisely because the memory system counts requests, not
threads, and both configurations supply enough.

**A change that makes A better:** shrink the problem, or make the kernel share
the machine. If the array is 2 M elements instead of 67 M, B's 40,960 threads
have 50 elements each and the grid-stride loop is short enough that the tail and
the launch ramp matter; A's 5,120 threads with `float4` extract the same
bandwidth from a smaller launch. More sharply: add a co-resident kernel that
takes 7/8 of the machine (Exercise 1's budget). B cannot exist; A can.

Equivalently, make the kernel body register-hungry — a 20-input fused
elementwise chain at 80 registers per thread caps occupancy at 24 warps per SM,
and A's shape fits inside that while B's does not.

**A change that makes B better:** make the access pattern unable to vectorize.
If the kernel reads `x[perm[i]]` for an arbitrary permutation `perm`, `float4`
is illegal (the addresses are neither contiguous nor 16 B aligned), and A's only
remaining source of requests-in-flight is per-thread coarsening, which does not
help a gather because each of the 4 elements is a separate sector anyway. B's
thread count is then the only lever. Same effect from an odd element size: a
3-float record, or an `N` whose alignment cannot be guaranteed because the
buffer is a sub-view of a larger allocation at a non-multiple-of-4 offset.

---

## Q3 — fusing a three-stage chain, and what L2 does to the answer

> A pipeline computes `b = f(a)`, `c = g(b)`, `d = h(c)`, all elementwise over
> arrays of size S, with `a` and `d` live outside and `b`, `c` private. Write the
> unfused and fused compulsory traffic as functions of S. Now suppose S is small
> enough that all four arrays fit comfortably in L2. State what happens to the
> predicted speedup and to the measured speedup, and explain why they move
> differently. Finally, state the condition on S under which fusion is *not*
> worth the engineering effort.

**Traffic.** Let `S` be the element count and `w` the element size in bytes.

- Unfused: stage 1 reads `a` writes `b` (2Sw), stage 2 reads `b` writes `c`
  (2Sw), stage 3 reads `c` writes `d` (2Sw). **6Sw.**
- Fused: reads `a`, writes `d`. **2Sw.**
- Predicted speedup **3.0×**.

**When everything fits in L2.** The predicted speedup does not change at all —
it is computed from the source, and the source does not know how large L2 is.
The *measured* speedup does change, and in a way most people get backwards.

The naive expectation is that it collapses toward 1, because `b` and `c` never
reached DRAM so there was no DRAM traffic to save. That part is right: the
traffic saving becomes an L2-traffic saving, and L2 on this GPU delivers
~1,300 GB/s against DRAM's ~376, so the same byte count costs about a quarter as
much time.

But a second term takes over. At small `S` each kernel runs in microseconds, and
on this machine a kernel launch costs about 10 µs. The unfused version pays
three launches and the fused version one. Measured on Exercise 2's four-stage
chain at 4 MB per array, the speedup went **up**, to 3.31× against a 2.00×
traffic prediction, and the `%ofpeak` column read 241–399% — the unmistakable
signature of an L2-resident measurement. Fusion was no longer saving bandwidth;
it was deleting kernel launches.

**They move differently** because the traffic model has a domain, and the domain
is "working set much larger than L2, kernel duration much larger than launch
overhead". Outside it the model is not wrong so much as irrelevant: it is
predicting a term that is no longer dominant.

**When fusion is not worth the effort.** Two regimes, at opposite ends:

- **`S·w` well below the L2 size (48 MB on this GPU), and the chain is long
  enough that launch overhead is already amortised.** Then the intermediates
  live in L2, the saving is a fraction of a cheap resource, and the fused kernel
  is harder to test and harder to reuse. Note this is a narrow window, because at
  genuinely small `S` launch overhead brings the payoff back.
- **When an intermediate has an external consumer.** Fusion can only delete the
  traffic of values nobody outside the region needs. A chain where every stage's
  output is also somebody's input has zero fusable traffic no matter how long it
  is.

The number to look up for this GPU: **`cudaDevAttrL2CacheSize` = 50,331,648 B**.
Compare `(number of distinct arrays) × S × w` against 4× that before quoting a
predicted speedup, per the benchmarking rule that a working set under ~4× L2 is
measuring cache.

---

## Q4 — one third of the stores, four times the time

> Kernel P writes `out[i] = c` for `i` in `[0, M)`, where `out` is `M` floats.
> Kernel Q writes `out[3*i] = c` for the same `i`, where `out` is `3M` floats.
> Both execute `M` store instructions and both write `4M` useful bytes; no reads
> appear in either source. Rank them by wall-clock time, quantify the ratio, and
> state the DRAM traffic of each. Then explain why adding `__stcs` to Q does not
> change the answer, and describe the one change to Q's *data layout* — not its
> code — that would.

**Ranking: Q is much slower. Roughly 6× on this GPU.**

**P.** A warp's 32 lanes write `out[32w .. 32w+31]`, i.e. 128 contiguous bytes
from a 128 B-aligned base. That is exactly 4 sectors, all four **completely
covered**. No fill is needed. DRAM traffic: **4M bytes written, 0 read.**

**Q.** A warp's 32 lanes write `out[96w], out[96w+3], …` — byte addresses 12 B
apart spanning 384 B, which is 12 sectors. Each sector receives either 2 or 3 of
its 8 floats, so **every sector is partially covered** and every one must be
read, merged, and written back.

Per warp: 12 sectors × 32 B = 384 B read **and** 384 B written = 768 B of DRAM
traffic for 128 B of useful stores. Over the whole kernel:
**12M bytes read + 12M bytes written = 24M bytes**, against P's 4M.

That is a traffic ratio of 6, and since both kernels saturate the bus, the time
ratio is also about 6. (`example01.cu` measures the stride-2 version of exactly
this experiment and gets 3.95× against a predicted 4×, with the implied DRAM
rate landing on the streaming ceiling — which is how you confirm the model
rather than merely assert it.)

Note the accounting carefully: Q executes **one third** the store instructions of
a dense kernel over the same `3M`-element array, writes **one third** the useful
bytes, and still takes longer than P, which writes the same useful bytes densely.
The store count is irrelevant; the sector coverage is everything.

**Why `__stcs` does not help.** `__stcs` sets an eviction-priority bit: it tells
L2 "this line is unlikely to be reused, prefer to evict it". It is a statement
about *cache residency*. The read Q generates is not a caching decision — the
bytes Q did not write belong to the program and must survive the write-back, so
the memory system has no choice but to fetch them. No hint can authorise
discarding them. Measured in `example01.cu`: plain `STG` 1.4571 ms, `__stcs`
1.4446 ms — identical within noise, despite the SASS genuinely changing from
`STG.E` to `STG.E.EF`.

**The data-layout change.** Make Q's writes cover whole sectors. Concretely: the
`3M` array is an interleaved structure with stride 3, and Q is touching field 0.
Split it into three separate arrays of `M` floats each — the AoS→SoA transform
of Module 5 — and Q becomes `out0[i] = c`, which is P. Traffic drops from 24M to
4M, a 6× improvement, and nothing about Q's *code* changed except the name of
the array it writes.

The general statement: a partial-sector write is a symptom of a layout in which
the things one kernel touches are not adjacent. The fix is always to make them
adjacent, and it is always a change to where the data lives rather than to how
the loop is written.
