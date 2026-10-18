<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# MiniCPM-V 4.6

A NaViT-style SigLIP vision tower whose token count is cut by an in-tower
2x2 window merger and a 2x2 MLP merger, feeding the Qwen3.5-0.8B hybrid
Gated-DeltaNet/GQA decoder. Images, several images per prompt, and video.

| Checkpoint | Parameters | Licence |
|---|---|---|
| `openbmb/MiniCPM-V-4.6` | 1.3B | Apache-2.0 |

**Status: in progress.** The decoder loads straight from the checkpoint
directory through `brain-qwen35` and matches the reference on text-only
prompts (logits and greedy decode). The vision side is not wired yet; the
plan and the gates are in the port's roadmap.

## Architecture

- **Vision tower**: 27 SigLIP layers of width 1152 (16 heads, MLP 4304,
  tanh GELU), patch 14. Every view keeps its own aspect ratio: position
  embeddings come from a 70x70 learned table at the nearest position, and
  all views of a request are packed into one sequence with attention kept
  inside each view.
- **Window merger**: after encoder layer 6, attention inside 2x2 windows,
  then each window's 4 tokens are concatenated and projected back to one
  (plus their mean). The remaining 20 layers run on a quarter of the tokens.
- **Merger**: 2x2 concatenation, LayerNorm, a two-layer MLP to the decoder's
  width 1024.
- **Downsample mode**: `16x` (both mergers, the default) or `4x` (the <!-- perf-number: mode names, not measurements -->
  merger only, four times more visual tokens for fine detail). Same weights.
- **Decoder**: Qwen3.5 text, 24 layers (3 Gated DeltaNet : 1 GQA), tied
  embeddings, vocabulary 248094. Visual features replace the placeholder
  rows of the input embeddings; positions are plain 1-D.
