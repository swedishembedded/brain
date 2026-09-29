<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Qwen2-family decoders (not yet servable)

Dense decoder-only transformers in the Qwen2 layout (`Qwen2ForCausalLM`):
the [Qwen3](qwen3.md) decoder with a bias on the query/key/value projections
and without QK-norm - grouped-query attention, rotary position embeddings,
SwiGLU MLPs. brain runs them on the same decoder as Qwen3, so everything the
Qwen3 page describes about precision tiers, sharding and serving applies
once a checkpoint loads.

The DeepSeek checkpoints of this family are the R1 reasoning distills:

| Checkpoint | Parameters |
|---|---|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.8B |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7.6B |

brain recognizes the architecture (HF class `Qwen2ForCausalLM`, GGUF
`qwen2`), but importing and serving these checkpoints is not available yet.

Package: `brain-qwen3`.
