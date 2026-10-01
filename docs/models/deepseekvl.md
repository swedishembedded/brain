<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# DeepSeek-VL

A Llama decoder reading image features from a hybrid vision tower: SAM-B
over the image at 1024 pixels and SigLIP-L over it at 384, joined by a split
MLP aligner.

| Checkpoint | Parameters | Licence |
|---|---|---|
| `deepseek-ai/deepseek-vl-7b-chat` | 7.3B | DeepSeek License |

**Status: served.** `brain serve` lists the model as `brain/deepseekvl` when
the checkpoint is in the models directory, and
`/v1/chat/completions` (or D-Bus `Run`, action `generate`) takes a
conversation with up to 8 images: OpenAI `image_url` data URLs or Anthropic `image`
blocks over HTTP (up to 8 per request, the 9th is a 400), or `image`,
`image1`, ... blobs over D-Bus. Each image part becomes a
placeholder where it stands in the message; an image sent without one opens
the last user turn. Decoding is greedy and stops on the end-of-sentence token
or the next `User:` turn.

The checkpoint is loaded as downloaded and run whole: preprocessing, both
towers, the aligner and the decoder at the checkpoint's own fp16. On the
real checkpoint every stage matches the reference implementation, and 16
greedy tokens are identical to its fp32 output.

**Placement.** The towers hold about 8 GB (SAM's attention at 1024 pixels)
and the decoder about 15 GB before its KV cache, which takes 0.94 MiB per
context token. With two cards the towers and the decoder each get one. The
served context is whatever KV cache fits the decoder's card, up to the
checkpoint's 16384 tokens; each image is 576 of them. A single 24 GB card
cannot hold both parts with a useful context, and the model is then not
served.

As a library:

```rust
use deepseekvl::prompt::{Role, Turn};

let m = deepseekvl::load(dir, qwen3::Dtype::F16, 2048)?; // or model::load_placed(dir, dtype, model::place(..)?)
let turns = [Turn { role: Role::User, content: "<image_placeholder>Describe this image.".into() }];
let ids = m.prompt_ids(&turns)?;
let embeds = m.image_embeds(&[image])?;
let reply = m.generate_greedy(&ids, &embeds, 256, &mut |_| true)?;
```

### Fine-tuning

```text
brain deepseekvl finetune --weights deepseek-ai/deepseek-vl-7b-chat \
    --dataset DIR --out DIR [--rank 16 --steps 200 --lr 1e-4 --aligner-lr X ...]
```

`DIR/train.jsonl` holds one example per line: an `image` (a path relative to
`DIR`) and `messages`, ending with the assistant reply to learn:

```json
{"image": "cat.png", "messages": [
  {"role": "user", "content": "<image_placeholder>\nWhat is this?"},
  {"role": "assistant", "content": "A cat."}]}
```

The towers stay frozen: each image's features are extracted once and the
towers released, so the decoder gets the whole card. The aligner trains from
the checkpoint's own weights (at `--aligner-lr`, a tenth of `--lr` unless
set), and the decoder trains as a LoRA over its frozen base, held at `bf16`
unless `--base-dtype f32`. Only the reply and its end-of-sentence are
supervised; each example carries one image. The checkpoint is read as
downloaded and nothing is written beside it: `--out` receives
`adapter.safetensors` and `aligner.safetensors`. The schedule and optimiser
flags (`--weight-decay`, `--grad-clip`, `--warmup`, `--min-lr`) are
`brain qwen3 finetune --lora`'s.

As a library: `deepseekvl::train::finetune`, or `Frontend::prepare` and
`Trainer` for your own loop.

The checkpoint is recognized by its HF class `MultiModalityCausalLM`, which
Janus-Pro shares. The generation heads that only Janus-Pro configures tell the
two apart.

Package: `brain-deepseekvl`.
