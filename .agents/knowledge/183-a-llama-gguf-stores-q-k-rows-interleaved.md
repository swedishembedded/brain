<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 183. A Llama GGUF stores its q/k rows interleaved

llama.cpp's converter permutes a llama-architecture checkpoint's q and k
projections before it writes them (`conversion/llama.py`,
`LlamaModel.permute`, `undo_permute = True`). Each head's rows are
reshaped `(2, head_dim/2)` and transposed, so HF row `half * hd/2 + i` of a
head is stored at `2i + half`. llama.cpp's llama RoPE kernel expects that
layout. brain's RoPE rotates HF's half-split pairs.

- **A Llama GGUF read as a plain rename computes attention on the wrong
  pairs.** It builds, every shape checks, and the model is quietly wrong.
  Qwen2 and Qwen3 conversions do not permute (their llama.cpp RoPE is the
  NEOX mode, which matches HF).
- **The fix is a row permutation, not a rename.**
  `qwen3::gguf_import::llama_unpermute_order` gives the source row for each
  brain row: `h*hd + 2i + half`. `checkpoint::remap::Fetch::RowPermute`
  applies it on every read path.
- **Quantized rows move without being decoded.** A permutation that only
  worked in f32 would have pushed exactly the q/k projections through a
  dequant. Every other tensor keeps its own quantization. `raw_blocks` now
  returns a `Cow`, so `RowPermute` hands back owned bytes with whole
  block-rows moved.

A llama GGUF also carries no `attention.key_length`. Its head size is
`rope.dimension_count`, which the converter writes from the HF `head_dim`.
A llama3 RoPE scaling travels as the `rope_freqs.weight` tensor of
divisors. Linear and YaRN scaling travel as `rope.scaling.*` keys. All of
these are read into the config; `rope_freqs.weight` used to be dropped.

`deepseek_gguf_parity` checks this against GGUFs made by llama.cpp's own
converter from deepseek-coder-1.3b-instruct (llama, linear RoPE ×4). At f16
and Q8_0, every position predicts the same token as the HF checkpoint.

Two more findings came out of the same work:

- **The GGUF resident's single-sequence path bought nothing.** It built its
  model at `Dtype::F32`, so it expanded the quantized file exactly as the
  batched engine would, while giving up batching and the paged KV. A GGUF
  now goes through the engine like every other format.
- **Neither path serves the file at its own quantization.** That is open
  work.
