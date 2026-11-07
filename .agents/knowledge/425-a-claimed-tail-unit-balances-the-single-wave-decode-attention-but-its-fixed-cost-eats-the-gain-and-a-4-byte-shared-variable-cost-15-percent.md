<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 425. A claimed tail unit balances the single-wave decode attention, but its fixed cost eats the gain, and a 4-byte shared variable cost 15 percent

**Lesson:** A claimed tail unit can fill the idle multiprocessors and still lose overall, because
each unit repeats a fixed cost. Measure that per-unit cost and the bound on recoverable idle time
before building a balancer, and compare against the base binary in the same session.

## What happened

Question (the brief of 2026-11-01, knowledge 423's design a): the last tiles of each key split as a
unit of their own with its own partial record, claimed through an atomic ticket (a counter the last
block of the launch resets), so a block that finished early takes the tail of one that is late. Does
it bring `paged_flash_decode_gqa_hd256_kvbf16` / `_kv8` / `_kvfp8` at 8K x 16 towards the byte roof?

## Why

### Measured (GH200 gpu1 idle, `decode_attention_matrix_cell_report` split + merge us a layer, same session base and change, 16 x 8192 bf16 unless stated)

- Exact: three tiers, ragged / scattered / shared blocks / NaN-poisoned records / splits with no
  live key, tails of 1, 3 and 15 tiles, three launches over one slab returning the same bits with
  the counter back at zero: every case passed (the spec was written first and is in the experiment
  patch, not shipped).
- Balance: `sm__cycles_active` (avg / max / min of 132 SMs, elapsed) went from 225 k / 241 k / **136
  k** / 248 k to 238 k / 249 k / **212 k** / 254 k. The idle SMs were fed; the work grew by 6 % (13
  k cycles an SM), which is the fixed cost of the stolen units (query fragments built from global,
  pipeline fill, four-warp merge, record write: ~5 us each, 2 units a block with a tail of 1/8).
- Net: 196.0 us base, 199.8 us with the tail at 1/8 of a split; 202, 205, 193, 199 us at 1/2, 1/4,
  1/16, 1/32 (the noise of the card is +-3 us). int8 and FP8 likewise 0 to +4 %, 1 row +10-17 % (the
  extra unit is a larger share of a short launch). Multi-wave cells (64 x 8K, 16 x 32K) were never
  planned a tail and sit within +-1 %.
- Bound on any such scheme: the SMs with one block are 3 % of the work (8192 tiles / 132 SMs = 62
  against 64 tiles on a full SM), so a perfect balance is worth 3-5 %, ~10 us a layer, 0.15 ms of a
  17 ms step. A flat cut (264 equal blocks) cannot do better and needs a block to cross (sequence,
  kv head) pairs, which the one-record-per-block kernel does not.
- Not shipped: the host and kernels stay as on origin/main. The patch is kept at
  `~/project/shared/tmp/decattn-tail-experiment.patch`.

## What to do

### Traps found on the way (each cost a measured slowdown with no tail running at all)

- **A 4-byte `__shared__` variable shifted the dynamic ring**: the key/value ring starts right after
  the static shared buffers, and the kernels' `cp.async` / `ldmatrix` patterns are conflict-free
  only with the ring 128-byte aligned. One `__shared__ unsigned int` (static 12544 -> 12560 bytes)
  made every bf16 cell 15 % slower, 64 x 65536 included (`cuobjdump -res-usage` SHARED 13568 ->
  13584 shows it). Pad a new static variable to a whole 128-byte row, and read the whole row, or the
  compiler shrinks it again.
- **Two inlined copies of a kernel body** (head and tail as two instantiations) raised `_kv8` from
  168 to 204 registers: 3 blocks of 128 threads need <= 170, so residency fell to 2 a multiprocessor
  and int8 / FP8 got 25-35 % slower. `__launch_bounds__(THREADS, 3)` restores 168; always compare
  `cuobjdump -res-usage` registers against the base before timing.
- **Shared arrays declared inside a device function are per instantiation**: two instantiations
  doubled the static shared memory to 26 KB and one block a multiprocessor fitted. File scope shares
  them.
- **Compare against the base binary in the same session**: the card read 27.1 us (1 x 8K) in one run
  and 29.2 in the next; the 64-row cells repeat within 1 %.

### Rules

- Measure per-unit fixed cost before splitting work into units that a block claims late: a unit with
  its own query prep, pipeline fill and merge costs ~5 us here, 2.5 % of a 200 us launch each.
- A single-wave launch's idle fraction is `1 - sm__cycles_active.avg / elapsed` (9 % base), and what
  a scheme can recover is bounded by the SMs' unequal block counts (3 %) plus the speed spread;
  check both before building a balancer.
