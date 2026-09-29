<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Qwen3-VL-30B-A3B (not yet servable)

Qwen3-VL's vision tower (ViT + PatchMerger + DeepStack, shared with
[Qwen3-VL](qwen3vl.md)) in front of a top-k sparse Mixture-of-Experts Qwen3
decoder: 128 experts, no shared expert, about 3B of its 31B parameters active
per token. Upstream: `Qwen/Qwen3-VL-30B-A3B-Instruct`.

brain recognizes the architecture (HF class `Qwen3VLMoeForConditionalGeneration`,
GGUF `qwen3vlmoe`) and has its configuration and model structure, but no real
checkpoint has been imported yet. It is not something you can run as a model
today; for vision-language inference use [Qwen3-VL](qwen3vl.md) or the other
models on the [vision-language page](vlm.md).

Package: `brain-qwen3vlmoe`.
