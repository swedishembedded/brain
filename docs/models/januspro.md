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

**Status: served.** `brain serve` lists the model as `brain/januspro` when
the checkpoint is in the models directory, with two actions:
`/v1/chat/completions` (action `generate`, the same conversation handling as
DeepSeek-VL) and `/v1/images/generations` (action `text2image`) with
`"size": "384x384"`, the one size the model draws; any other size is a 400.
The two run on different builds of the checkpoint, each on one card: the
chat build (SigLIP tower about 3 GB, bf16 decoder about 15 GB before its KV
cache) and the drawing build (about 19.5 GB at 1024 tokens a sequence). The
scheduler swaps them on a card that cannot hold both, or keeps one on each
card of a two-card host. Every context is what
the card leaves room for, never a fixed figure. The checkpoint is loaded as
downloaded (bf16 `pytorch_model-*.bin` shards):

- **Understanding** (`januspro::model::load_understanding`): the same
  composite as DeepSeek-VL, with Janus-Pro's single tower, roles and image
  tags. At the checkpoint's own bf16 weights (about 14 GB) every stage
  matches the reference implementation, and 16 greedy tokens are identical
  to its fp32 output.
- **Text to image** (`januspro::t2i::TextToImage`): each image is a
  conditional and an unconditional sequence decoded together on the paged
  serving engine, with the 576 sampled tokens decoded by the VQ-16. The
  engine keeps the decoder at the checkpoint's own bf16 (about 14 GB);
  int8 is not offered, because guidance magnifies its error.

```rust
let mut t2i = januspro::t2i::TextToImage::load(dir, 1, qwen3::Dtype::BF16, 1024)?;
let req = januspro::t2i::Request { prompt: "A red apple on a wooden table.", cfg_weight: 5.0, temperature: 1.0, seed: 7 };
let images = t2i.generate(&req, &|| false, &mut |_, _| {})?;
```

The checkpoint's config names no class, only DeepSeek-VL's `model_type`; the
generation heads that only Janus-Pro configures tell the two apart.

Package: `brain-januspro`.
