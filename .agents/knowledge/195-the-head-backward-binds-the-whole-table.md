<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 195. The LM head's backward bound the whole table

The forward applied the LM head a vocab tile at a time, because a storage
binding is capped at 2 GiB on every backend. The backward did not: its input
gradient bound the whole `[vocab, d_model]` fp32 weight in one dispatch.

A `[152064, 3584]` head is 2.18 GB. A 7B LoRA fine-tune loaded, built its
forward, and then failed creating the backward's bind group
(`matmul_dx_reg`, binding range 2179989504 above the limit of 2147483644),
after the multi-minute load of the base. Nothing smaller shows it: a 1.5B
decoder's head is 0.93 GB, so every test and every small real run passed.
Any decoder with `vocab * d_model * 4` above 2 GiB was affected: Qwen2.5-7B
and the R1 distills on it, Llama-3's 128k vocabulary at d = 4096 included.

The backward now follows the forward's tiling: `matmul_dx_tile` accumulates
each tile's rows of `W` into `d_xn`, and `matmul_dw_tile` writes each tile's
rows of the head's weight gradient when the head trains. Each dispatch binds
only its rows and reads its columns of the logits gradient through the full
row stride.

A test can force the tiling with `BRAIN_TILE_BUDGET_WORDS`, which is how the
gradient is gated against the untiled build on a tiny model. The binding
limit itself cannot be lowered on a device, so what such a test asserts is the
recorded dispatches (one tile kernel per tile) next to the numbers.
