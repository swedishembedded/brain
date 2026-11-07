<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 426. The unified pool's 8K x 64 cost was memory mapped on the serving thread, at the first touch of a block and at every admission, not a slower kernel

**Lesson:** Driver memory-mapping calls on the serving thread cost throughput even when kernel
totals are unchanged. When the card idles with equal kernel time, look for such calls at first touch
and at admission, and keep them out of those paths.

## What happened

Question (the brief of 2026-11-01): commit 6382ed2c7 (knowledge 424) cost 8K x 64 1.5-3 % output
tok/s (368 -> 360), bf16 KV, f32 state, no speculation. Why?

### Measured (nsys of the 8K x 64 cell, gpu1, base 76243a969 against 46b5bf21d)

- Output tok/s under nsys 369.1 -> 347.8; the kernel total is the same (87.54 s against 87.52 s;
  decode attention 6037 against 6058 ms, the GEMMs within 0.3 %), so it is not a kernel: the card
  idled. Gaps over 5 ms between kernels: 330 -> 541.

## Why

- The difference is the CUDA driver calls of the sparse buffers on the serving thread:
  `cuMemSetAccess` 27472 calls 2357 ms, `cuMemCreate` 27472 calls 1473 ms (base: none), `cuMemUnmap`
  8544 calls 592 ms. Two sources:
  1. **First touch**: a block is backed when the allocator first hands it out, 32 planes x (create +
     set access) ~ 4.7 ms; the ~700 blocks of 64 streams of 8K were backed inside the cell (3.3 s of
     94).
  2. **Admission**: a sequence's state row backed ~140 granule-planes (~20 ms) and unbacked the
     blocks lent for it (3-4 blocks at 1.3 ms), and the release undid it; a server at its limit
     replaces a finished sequence at once, so every sequence paid ~30 ms.

## What to do

- Fixes: all blocks backed at load (`StateLedger::back_all_blocks`; a block a state row displaces
  still gives its memory back), and a released state row stays backed while other sequences run (at
  most 8, none once idle; the next sequence takes that row first, and the retained rows go before an
  admission is refused). Calls during the cell fell from ~24000 to ~3500 (the one-time lend of the
  first rows).

### Rules

- A memory operation of the driver on the serving thread costs a hundred microseconds to
  milliseconds and takes a lock: keep it out of admission and first touch; count `cuMem*` calls in
  an nsys trace of any change that maps memory on demand.
- Compare kernel totals before calling a throughput loss a kernel loss: equal totals with more idle
  gaps point at the host.

## Evidence

- Served A/B (no nsys, `matrix.sh`, bf16 KV, f32 state, no speculation, interleaved, output tok/s):
  8K x 64 main 353.6 / 360.2 / 344.9 / 351.9 (mean 352.7), fixed 362.2 / 360.3 (361.3, +2.4 %); 8K x
  16 main 291.0 / 301.1, fixed 309.6 / 307.7 (+4 %); 8K x 1, 32K x 16, 64K x 4 within noise (+-1 %).
  The remaining ~2 % to the 368 of the separate pools is not in a kernel or a driver call that this
  trace names (median ITL 34.3 against 33.2 ms).
