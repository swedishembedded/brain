<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Janus-Pro

One Llama decoder that both understands images and generates them.
Understanding reads SigLIP-L features through an MLP aligner; generation
predicts VQ-16 image tokens with its own head, under classifier-free
guidance, and decodes the finished token grid to pixels.

| Checkpoint | Parameters | Licence |
|---|---|---|
| `deepseek-ai/Janus-Pro-7B` | 7.4B | MIT code, DeepSeek Model License |

**Status: recognized, not yet served.** brain identifies the checkpoint (its
config names no class, only DeepSeek-VL's `model_type`; the generation heads
tell them apart) and reads its configuration. The towers, generation loop
and serving are not implemented, so `brain serve` does not list it and
`brain pull` downloads it without a way to load it.

Package: `brain-januspro`.
