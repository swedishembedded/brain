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

### Fine-tuning

```text
brain januspro finetune --mode understanding|generation --weights deepseek-ai/Janus-Pro-7B \
    --dataset DIR --out DIR [--rank 16 --steps 200 --lr 1e-4 --aligner-lr X --cfg-dropout 0.1 ...]
```

Both modes train a LoRA over the frozen bf16 decoder (`--base-dtype f32` to
hold it at fp32) and read the checkpoint as downloaded; nothing is written
beside it. The flags are `brain deepseekvl finetune`'s.

- **understanding** is that command's trainer with Janus-Pro's tower and
  roles: `DIR/train.jsonl` holds an `image` and the `messages` to learn,
  and `--out` receives `adapter.safetensors` and `aligner.safetensors`.
- **generation** teaches the decoder to draw. Each line of `DIR/train.jsonl`
  is a `prompt` and the `image` (relative to `DIR`) to draw for it:

  ```json
  {"prompt": "A golden retriever puppy.", "image": "dog.png"}
  ```

  Every image is encoded once by the frozen VQ-16, which is then released.
  A step feeds the decoder the prompt, the begin-of-image tag and the first
  575 of the image's 576 codes (each through the code embedding and the
  generation aligner), and the cross-entropy of the generation head's
  prediction at each of the 576 positions from the tag on is differentiated
  back through all of it. The adapter, the generation head, the generation
  aligner and the code embedding train; `--cfg-dropout` of the steps (a
  tenth by default) run on a padded prompt, which is what classifier-free
  guidance contrasts against. `--out` receives `adapter.safetensors` and
  `generation.safetensors`.

An understanding fine-tune is served by starting `brain serve` with
`BRAIN_JANUSPRO_TUNED=<out dir>` (the adapter is attached at run time, the
trained aligner replaces the checkpoint's); a generation fine-tune is served
by `BRAIN_JANUSPRO_TUNED_GENERATION=<out dir>` (its adapter is folded into the
drawing engine's decoder, its head, aligner and code embedding replace the
checkpoint's).

A fine-tune kept in `<model dir>/adapters/<owner>/<name>/<tag>/` is listed as a
model of its own, `brain/januspro:<owner>:<name>:<tag>`, beside the base. One
holding `generation.safetensors` serves `text2image`; any other serves
`generate`.

As a library: `januspro::train::{finetune_understanding, finetune_generation}`,
or `GenTrainer` for your own loop.

The checkpoint's config names no class, only DeepSeek-VL's `model_type`; the
generation heads that only Janus-Pro configures tell the two apart.

Package: `brain-januspro`.
