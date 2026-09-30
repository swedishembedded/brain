<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Llama-family decoders

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
`llama`). `brain pull deepseek-ai/<checkpoint>` fetches one (as safetensors
when the repo ships them, otherwise its `pytorch_model*.bin` shards), and
brain serves it from those files as downloaded, at its own configuration,
RoPE scaling included, reading each checkpoint's own tokenizer pipeline,
chat template and stop tokens. `brain serve` serves a pulled checkpoint under
its `deepseek-ai/<checkpoint>` id; the R1 distills' reasoning comes back as
`reasoning_content`. At fp32 a 7-8B checkpoint needs more than one 24 GB card;
`--qwen-weights-int8` serves it on one.

A llama.cpp GGUF of any of them serves too, straight off the file
(`brain serve` with the `.gguf` as the checkpoint, or `brain import` to
write a brain checkpoint). llama.cpp stores a Llama GGUF's q/k projections
with each head's rows interleaved. brain reorders whole rows as it reads
them, quantized ones included, and reads the RoPE scaling from the file:
the `rope.scaling.*` keys, or `rope_freqs.weight` for llama3. The quantized
file is expanded to f32 (or requantized to int8 with `--qwen-weights-int8`)
as it is uploaded; serving it at its own quantization is not supported yet.

Package: `brain-qwen3`.
