# Module 5 — Check Your Understanding (answers)

---

## 1. Warps A (contiguous), B (reversed), C (broadcast)

**Sector counts.** A and B both request **4 sectors**. C requests **1**.

A and B are equal because the address coalescer is presented with the 32
addresses of one instruction simultaneously and computes their *set* of
sectors. It has no representation of which lane produced which address, so
`{0,4,…,124}` and `{124,120,…,0}` are literally the same input. Lane order is
not a thing the hardware can observe at this stage.

C's 32 lanes supply one distinct address, hence one sector. There is no replay
and no serialization: the return path broadcasts the loaded value to all lanes.
(This is *unlike* a shared-memory bank conflict, Module 7, where 32 lanes
hitting one bank at *different* addresses do serialize — the distinction is
same-address vs same-bank.)

**Expected time.** C fastest, then A and B tied.

**Where the rankings disagree.** By the efficiency metric,
`bytes_requested / bytes_moved`, C scores 4 / 32 = **12.5 %** — the same score
as a pathological stride-32 gather — while A and B score 100 %. Yet C is the
fastest of the three.

What that tells you: efficiency measures *waste within a single instruction*.
It has no term for **reuse**. C moves 32 B once; the sector then sits in L1 and
every subsequent warp reading the same value hits. Over a kernel, C's total
DRAM traffic is 32 B, against A's 128 B per warp. The metric, applied to one
instruction, says C is eight times worse; the machine says it is many times
better.

The practical consequence: never optimize the efficiency number in isolation.
It is a correct model of DRAM traffic only when each sector is touched by one
warp once. Uniform (thread-invariant) access is a *good* pattern that the
metric scores badly — which is also why the compiler goes out of its way to
recognise it and, where it can prove uniformity, hoist the load entirely.

---

## 2. `out[i] = in[i+1]` — model says 80 %, measurement says 336/340

**The mechanism.** The model counts *requests*, and it is right: warp *k* reads
bytes `[128k+4, 128k+132)`, spanning sectors `4k … 4k+4` — five of them. But
warp *k+1* reads `[128k+132, 128k+260)`, spanning `4k+4 … 4k+8`. **Sector
`4k+4` appears in both.**

In a streaming kernel, warps *k* and *k+1* are co-resident and execute within
microseconds. Whichever issues first misses to DRAM; the other hits in L1 (if
on the same SM) or L2 (if not). DRAM therefore still moves exactly 128 B of new
data per warp. Only the *on-chip request count* rose, from 4 to 5 per warp, and
that path is not the bottleneck — the DRAM bus is. Hence ~99 % of the aligned
bandwidth.

Stated generally: the sector count predicts L1/LSU request pressure exactly,
and predicts DRAM traffic only when warps do not share boundary sectors.

**A kernel where the 80 % prediction would be accurate.** You need the
contiguous runs to be short and separated by gaps, so no neighbouring warp
exists to absorb the boundary sector.

Concrete construction: a record array of `P` floats per record with
`P*4 % 32 != 0`, where each warp reads only columns `0..31` of one record.
Warp *r* reads `[260r, 260r+128)` for `P = 65`; the boundary sector it drags in
contains columns 32–33 of record *r*, which **no warp reads**. That sector is
fetched from DRAM and discarded. The measured penalty then matches the model to
within a few percent.

That is precisely Exercise 3: predicted 82.1 %, measured 84.0 % of the aligned
case.

The structural difference in one sentence: in the streaming case the "wasted"
bytes are somebody else's useful bytes; in the windowed case they are nobody's.

---

## 3. AoS vs SoA for 10⁷ records × 8 floats, kernels P / Q / R

Record = 32 B. A warp handles 32 consecutive records, spanning 1024 B of AoS.

**Kernel P — reads field 0 only.**

- AoS: addresses 32 B apart, 32 distinct sectors, 128 useful bytes.
  → 1024 B moved, **12.5 %**.
- SoA: 32 contiguous floats, 4 sectors, 128 B moved, **100 %**.
- SoA moves **8× fewer bytes**. Per warp: 128 B vs 1024 B.

**Kernel Q — reads all 8 fields.**

- AoS: eight load instructions, each covering the same 1024 B / 32 sectors.
  First misses, other seven hit L1. DRAM moves 1024 B per warp, all used.
- SoA: eight streams × 4 sectors = 32 sectors = 1024 B, all used.
- **Identical DRAM traffic.** Neither layout wins on bytes.

**Kernel R — reads fields 0 and 1.**

- AoS: two instructions over the same 32 sectors. Fields 0 and 1 are adjacent
  (bytes 0–7 of each record), so both live in the same sector as each other.
  DRAM moves 1024 B; 256 B are used. **25 %**.
- SoA: two streams × 4 sectors = 8 sectors = 256 B, all used. **100 %**.
- SoA moves **4× fewer bytes**.

Note the pattern: the AoS penalty is `record_size / used_bytes_per_record`,
capped by the fact that a warp never touches more than 32 sectors. Reading
*k* of 8 fields gives a ratio of `8/k`, which is why it is 8×, 4× and 1× for
P, R and Q.

**Why SoA is still faster for Q, despite identical byte counts.** Two reasons,
both measured in `example02.cu` and Exercise 2 (280 GB/s AoS vs 379 GB/s SoA,
a 1.35× gap on identical DRAM traffic):

1. **Request count.** AoS issues 8 × 32 = 256 sector-requests per warp against
   SoA's 8 × 4 = 32. The seven L1 hits are cheap in DRAM terms but they still
   consume LSU issue slots, L1 tag lookups and MSHR entries. Once DRAM is not
   the binding resource, this path becomes it.
2. **Partial-sector writes.** If the kernel also *writes* a subset of the
   fields, AoS writes fragments of sectors. A partially written sector cannot
   be pushed to DRAM as-is; it must be read, merged, and written back. SoA's
   stores fill whole sectors and skip the fill entirely.

---

## 4. The write-only kernel with 64 B-strided stores

**Why the colleague is wrong.** DRAM does not accept partial-sector writes.
The smallest thing the memory system can hand to DRAM is a 32 B sector. If a
warp's store covers only part of a sector, the hardware must first **fetch**
that sector (from L2 or DRAM), merge the new bytes into it, and write the
merged sector back. This is *write-allocate* behaviour and it appears nowhere
in the source: the kernel has no load instruction and generates read traffic
anyway.

The only stores that avoid the fill are those that cover a full sector, because
then there is nothing to preserve.

**Quantifying it.** Threads write one 4 B float at addresses 64 B apart. Per
warp: 32 addresses spanning 2048 B, each in its own sector (64 B > 32 B), so
**32 distinct sectors**, each only 4/32 full.

| Traffic | Bytes per warp |
|---|---|
| Useful (what the program wanted written) | 128 |
| Sector fills (reads) | 32 × 32 = 1024 |
| Sector write-backs | 32 × 32 = 1024 |
| **Total DRAM** | **2048** |

Efficiency 128 / 2048 = **6.25 %** — half the 12.5 % a strided *read* of the
same pattern would score, because the write pays both directions.

This is the asymmetry worth remembering: **a scattered write costs about twice
a scattered read.** When a kernel must leave one side uncoalesced, leave the
reads.

**The data-layout fix.** Do not change the kernel; change where the outputs
live. The 64 B stride means the destinations are one field of a 16-float
record. Split that field into its own contiguous array — SoA — and the same 32
lanes write 128 contiguous bytes: **4 sectors, fully covered, no fill, no
read traffic at all.**

| | sectors | fill reads | write-backs | total DRAM per warp |
|---|---|---|---|---|
| strided (64 B) | 32 | 1024 B | 1024 B | 2048 B |
| contiguous | 4 | 0 | 128 B | 128 B |

A 16× reduction, and the kernel source is unchanged apart from the base
pointer.

(If the interleaved layout is fixed by an external interface and cannot be
changed, the remaining option is to have each warp buffer a full sector's worth
of results and write them together — which needs a scratchpad the warp shares.
That is shared memory, Module 6, and for the two-dimensional version of exactly
this problem, Module 15.)
