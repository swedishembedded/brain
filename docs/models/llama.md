<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Llama-family decoders (not yet servable)

Dense decoder-only transformers in the Llama layout (`LlamaForCausalLM`):
pre-norm RMSNorm, rotary position embeddings, SwiGLU MLPs, grouped-query or
full multi-head attention, no attention bias and no QK-norm. brain runs them
on the same decoder as [Qwen3](qwen3.md) - a Llama checkpoint is that
decoder with QK-norm off - so everything the Qwen3 page describes about
precision tiers, sharding and serving applies once a checkpoint loads.

The DeepSeek checkpoints of this family:

| Checkpoint | Parameters | RoPE scaling |
|---|---|---|
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 8.0B | llama3 (x8) |
| `deepseek-ai/deepseek-coder-1.3b-base` / `-instruct` | 1.3B | linear (x4) |
| `deepseek-ai/deepseek-coder-6.7b-base` / `-instruct` | 6.7B | linear (x4) |
| `deepseek-ai/deepseek-coder-7b-base-v1.5` / `-instruct-v1.5` | 6.9B | none |
| `deepseek-ai/deepseek-llm-7b-base` / `-chat` | 6.9B | none |
| `deepseek-ai/deepseek-math-7b-base` / `-instruct` | 6.9B | none |

brain recognizes the architecture (HF class `LlamaForCausalLM`, GGUF
`llama`), but importing and serving these checkpoints is not available yet.

Package: `brain-qwen3`.
