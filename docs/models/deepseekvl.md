<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# DeepSeek-VL

A Llama decoder reading image features from a hybrid vision tower: SAM-B
over the image at 1024 pixels and SigLIP-L over it at 384, joined by a split
MLP aligner.

| Checkpoint | Parameters | Licence |
|---|---|---|
| `deepseek-ai/deepseek-vl-7b-chat` | 7.3B | DeepSeek License |

**Status: recognized, not yet served.** brain identifies the checkpoint
(HF class `MultiModalityCausalLM`, which Janus-Pro shares; the generation
heads only Janus-Pro configures tell them apart) and reads its
configuration. The tower, aligner and composite are not implemented, so
`brain serve` does not list it and `brain pull` downloads it without a way
to load it.

Package: `brain-deepseekvl`.
