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

The backward now follows the forward's tiling, and both run on the GEMM
kernels an untiled head uses. The first version of the fix applied each tile
with one-thread-per-output kernels, which is what the forward's `matmul_tile`
does. It was correct and unusable: profiled on the 7B fine-tune at 478
tokens, the forward's tile dispatches took about 12 s each, the backward's
about 1.1 s each, and a 1.8k-token step ran for minutes (the profiler even
discards the forward's timings as implausible). Each vocab tile is now cut
into passes of at most a quarter of the tile budget in columns; a pass is one
GEMM against its run of head rows into a dense scratch, plus `copy_cols` to
move the pass's columns into place in the logits (forward) or out of the
logits gradient (backward). The same 1.8k-token fine-tune went from 14.5
minutes to under 3, with the same losses.

The same profile showed the bf16 frozen base's input gradient running the
naive `matmul_dx#w=bf16` (every linear of every layer): the register-tiled
`matmul_dx_reg` now has the bf16 storage variant too.

A test can force the tiling with `BRAIN_TILE_BUDGET_WORDS`, which is how the
gradient is gated against the untiled build on a tiny model. The binding
limit itself cannot be lowered on a device, so what such a test asserts is the
recorded dispatches (the copies, and no `matmul_tile`) next to the numbers.
