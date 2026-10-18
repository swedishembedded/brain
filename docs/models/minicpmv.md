<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# MiniCPM-V 4.6

A NaViT-style SigLIP vision tower whose token count is cut by an in-tower
2x2 window merger and a 2x2 MLP merger, feeding the Qwen3.5-0.8B hybrid
Gated-DeltaNet/GQA decoder. Images, several images per prompt, and video.

| Checkpoint | Parameters | Licence |
|---|---|---|
| `openbmb/MiniCPM-V-4.6` | 1.3B | Apache-2.0 |

**Status: served.** `brain serve` lists the model as `brain/minicpmv` when
the checkpoint is in the models directory, and `/v1/chat/completions` (or
D-Bus `Run`, action `generate`) takes a conversation with up to 8 images:
OpenAI `image_url` data URLs over HTTP, or `image`, `image1`, ... blobs over
D-Bus. Each image part becomes a placeholder where it stands; an image sent
without one opens the last user turn. Tools, `tool_choice`, stop strings,
`enable_thinking` and the sampling parameters behave as for the other chat
models; tool calls come back structured. Requests that arrive together
decode as one batch.

The checkpoint is loaded as downloaded (one BF16 safetensors file, run in
fp32) and the whole composite sits on one card: about 5.2 GB of weights, the
decoder's KV pool for 4 sequences of 16k tokens, and the tower's scratch.

## Getting started

```bash
brain pull openbmb/MiniCPM-V-4.6
brain minicpmv generate --image photo.jpg --prompt "Describe this image." --temp 0
brain minicpmv generate --image diagram.png --downsample 4x --max_slices 36 --prompt "Explain the diagram."
brain serve --openai 8789 --api-keys-out /tmp/keys.json
python3 samples/python/vision/minicpmv-chat/minicpmv_chat.py --image a.png --image b.png \
    --question "How do these differ?" --base-url http://127.0.0.1:8789 --keys-file /tmp/keys.json
```

| option | effect |
|---|---|
| `--downsample 16x` / `4x` | visual tokens per view: one per 4x4 patches (default) or per 2x2 (four times the tokens, for small text and fine detail) |
| `--max_slices N` | the most slices a high-resolution image is cut into (default 9; 36 for detailed documents) |
| `--use_image_id false` | drop the `<image_id>N</image_id>` tag before each image |
| `--enable_thinking` | let the model reason in a `<think>` block first (off by default, as the checkpoint's template) |
| `--tools JSON` | OpenAI function definitions; replies come back as `tool_calls` |

## How it is validated

Against the reference implementation (`transformers`' `minicpmv4_6`, fp32
on CPU) on the real checkpoint, from goldens written by
`tools/goldens/minicpmv_dump_reference.py`:

- **Processor**: resized views byte for byte (torch's uint8 antialiased
  bicubic, not Pillow's), slice grids, normalized patches, the rendered chat
  template and the token ids, identical across the slicing branches, 4x,
  slicing off, no image ids and several images per prompt.
- **Vision**: every stage of the tower, the window merger and the merger at
  cosine >= 0.9999999999, at 16x, 4x and with two images packed together.
- **Decoder**: text-only logits at cosine 1.0000000000.
- **End to end**: the next-token logits after the prompt and the whole
  greedy continuation identical to the reference for a sliced photo (16x and
  4x), a document and two images in one prompt, decoded alone and batched.

## Limits

- Video is not wired yet.
- Images are decoded by brain's own codecs; a JPEG can decode a pixel or two
  differently from PIL's libjpeg, which can move a long greedy continuation.
- Fine-tuning is not wired yet.

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
