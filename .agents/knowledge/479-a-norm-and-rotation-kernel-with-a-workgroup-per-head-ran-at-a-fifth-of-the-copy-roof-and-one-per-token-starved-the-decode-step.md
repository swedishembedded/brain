<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 479. A norm-and-rotation kernel with a workgroup per head ran at a fifth of the copy roof, and one block per token starved the decode step

**Lesson:** a per-row kernel sized for the portable tier (one 64-lane workgroup per head row, each lane computing its
own rotation angles) is latency- and trig-bound at prefill row counts: the angles depend only on the token's position,
so the rows of one token share them. But the grid has to stay wide at decode row counts too: a block per token gave a
decode step of 16 tokens 16 blocks and doubled its cost. Size the block by a fixed group of heads, not by the token.

## What happened

After knowledge 478 the largest glue kernel of a Qwen3-4B prefill (bf16 weights, 8K-row chunks, GH200) was the
portable `qknorm_rope_fused`: 0.43 ms for the 32 query heads and 0.14 ms for the 8 key heads per layer, 0.6 TB/s on
268 MB of query rows. A native form with one block per token (angles once per token, four heads at a time) halved
the prefill cost (2.5 -> 1.2 ns a prefill token at c=64) but took a decode step's QK norm from 0.27 to 0.52 ms, since a
step's 16-64 tokens made 16-64 blocks. One block per four heads of a token keeps the decode step where it was and
the prefill at 1.6 ns a token.

## What to do

- `qknorm_rope_rows` (`crates/kernels-cuda/cu/qknorm_rope_rows.cu`, `gpu_core::provider::library::qk_norm_rope`) is
  bit-identical to `qknorm_rope_fused` (the 64-lane accumulation, the left fold of the partials, every rounding):
  gate `crates/gpu-core/tests/it/qknorm_rope_native.rs` (mutation-checked with `rsqrtf` for the norm). The qwen3 serving
  engine takes it wherever the device offers it; `engine_glue` holds a bf16 tape to no portable `qknorm_rope_fused`.
- Measure a glue kernel's replacement at both the prefill and the decode row counts of the same run before keeping it.

## Evidence

Same command as knowledge 478, c=64, nsys glue by launch phase: QK norm per prefill token 2.52 ns (portable) -> 1.25 ns
(block per token) -> 1.61 ns (block per four heads); per decode step 0.27 -> 0.52 -> 0.26 ms. Client c=64 3.46 -> 3.66
req/s, TTFT p50 9.3 -> 8.9 s; c=16 3.07 -> 2.98 / 3.07 / 3.17 req/s over three runs (within that cell's run-to-run
spread).
