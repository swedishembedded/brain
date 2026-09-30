<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Qwen2-family decoders

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
`qwen2`). `brain pull deepseek-ai/<checkpoint>` fetches one (as safetensors
when the repo ships them, otherwise its `pytorch_model*.bin` shards), and
brain serves it from those files as downloaded, at its own configuration,
RoPE scaling included, reading each checkpoint's own tokenizer pipeline,
chat template and stop tokens. `brain serve` serves a pulled checkpoint under
its `deepseek-ai/<checkpoint>` id; the R1 distills' reasoning comes back as
`reasoning_content`. At fp32 a 7-8B checkpoint needs more than one 24 GB card;
`--qwen-weights-int8` serves it on one.

A llama.cpp GGUF of either serves too, straight off the file, its q/k/v
biases and RoPE settings read from it. The quantized file is expanded to f32
(or requantized to int8 with `--qwen-weights-int8`) as it is uploaded;
serving it at its own quantization is not supported yet.

Package: `brain-qwen3`.
