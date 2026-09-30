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

**Status: runs as a library, not yet served.** `brain-januspro` loads the
checkpoint as downloaded (bf16 `pytorch_model-*.bin` shards) and runs both
halves:

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
let mut t2i = januspro::t2i::TextToImage::load(dir, 1, qwen3::Dtype::BF16)?;
let req = januspro::t2i::Request { prompt: "A red apple on a wooden table.", cfg_weight: 5.0, temperature: 1.0, seed: 7 };
let images = t2i.generate(&req, &|| false, &mut |_, _| {})?;
```

The checkpoint's config names no class, only DeepSeek-VL's `model_type`; the
generation heads that only Janus-Pro configures tell the two apart.
`brain serve` does not list the model yet.

Package: `brain-januspro`.
