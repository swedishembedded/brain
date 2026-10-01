<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 196. A decode-only build prefills in chunks, not one decode tape per token

`Qwen::prefill` fed the KV cache one position at a time: each prompt token
recorded and submitted a whole single-row decode tape, with the fence moved to
the end. It saved the readbacks and nothing else, so a prompt of a few thousand
tokens (an agent's system prompt plus its tool schema) cost a few thousand
tapes of matrix-vector work, where one chunk of rows is a GEMM per linear that
the weights are read once for.

A decode-only build sized every activation buffer for one row, which is what
made a batched forward impossible on it. It is now sized for one prefill chunk
(`DECODE_PREFILL_ROWS` rows, independent of the context, so the context stays
bounded by the KV cache alone; `footprint::estimate_vram_bytes` counts the same
rows through `activation_rows`). `prefill` runs the prompt through the ordinary
batched layer forward with `Attend::Cache`: RoPE at the chunk's absolute
positions, K/V appended to the decode cache, and `block::gqa_chunk_step`
attending each row to the cache up to its own position. The KV cache is the
`num_blocks = 1` case of the paged pool that kernel already addresses, so no
new kernel was needed beyond registering `paged_flash_prefill` for
`head_dim <= 128` next to the existing `head_dim = 256` one.

Things that decide whether it is correct:

- The fused kernel only exists on a device with workgroup reductions. Where
  the materialised score/softmax/apply triad runs instead (the CPU backend,
  or a head_dim with no fused kernel), a chunk's `[rows, n_heads, start +
  rows]` scores must fit `scores`, which holds `n_heads * ctx`. The chunk
  length is cut to fit (never below one row), so a long context shortens
  chunks on that path instead of failing or capping the context.
  `block::gqa_chunk_fused` is the one place that decides which path a device
  takes, and the sizing asks it.
- Token rows and raw embedding rows (vision features) interleave in a prompt;
  a chunk ends at every change of kind.
- A chunk only reads back at the end of the prompt, and only the last row's
  final norm is computed (into `xn_final` row 0, where the next `step` and
  `decode_logits` read it).
- DeepStack additions index the whole image sequence, so they are skipped for
  a chunk; a model that uses them prefills per row through `prefill_mrope`.

Parity is held by `tests/batched_prefill.rs` against the `step` walk over the
same inputs: logits within fp32 summation order (1e-4 of the logit scale, 1e-3
at the int8 tier), the same greedy continuation, mixed token/embedding
prompts, a runtime and a folded LoRA adapter, caller-sized chunks with the
cancel token polled between them, and a submission count that does not grow
with the prompt.
