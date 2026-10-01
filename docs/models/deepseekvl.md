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
and the decoder about 15 GB before its KV cache, which a served build holds
as int8 (about 0.25 MiB per context token). With two cards the towers and the decoder each get one. The
served context is whatever KV cache fits the decoder's card, up to the
checkpoint's 16384 tokens; each image is 576 of them. A single 24 GB card
cannot hold both parts at the checkpoint's fp16 with a useful context, so the
decoder is then built with int8 linears (about 8 GB) instead; any host with
room for the fp16 decoder keeps it.

**Batching.** The decoder is `brain-qwen3`'s paged serving engine. Requests
that reach the model together are prefilled one after another and then decode
as one batch, a step per token for all of them, on one KV pool of the loaded
context that they share (up to four sequences at a time). A request that does
not fit the pool beside those in flight waits for the batch; one that could
never fit is refused on its own.

As a library:

```rust
use deepseekvl::prompt::{Role, Turn};

let mut m = deepseekvl::load(dir, qwen3::Dtype::F16, 2048)?; // or model::load_placed(dir, dtype, model::place(..)?)
let turns = [Turn { role: Role::User, content: "<image_placeholder>Describe this image.".into() }];
let ids = m.prompt_ids(&turns)?;
let embeds = m.image_embeds(&[image])?;
let reply = m.generate_greedy(&ids, &embeds, 256, &mut |_| true)?; // or generate_batch(&requests, ..)
```

### Fine-tuning

```text
brain deepseekvl finetune --weights deepseek-ai/deepseek-vl-7b-chat \
    --dataset DIR --out DIR [--rank 16 --steps 200 --lr 1e-4 --aligner-lr X --batch N ...]
```

`DIR/train.jsonl` holds one example per line: an `image` (a path relative to
`DIR`), or `images` (a list of paths, in the order the messages place them),
and `messages`, ending with the assistant reply to learn:

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
supervised, and the messages place one `<image_placeholder>` per image.
`--batch N` averages the gradients of N examples into each step. A decoder
that does not fit one card is split across the cards by what each has free,
with no flag (the images are spliced into the first stage, and their
gradient comes back from it); `--device` pins one card. The checkpoint is
read as downloaded and nothing is written beside it: `--out` receives
`adapter.safetensors` and `aligner.safetensors`. The schedule and optimiser
flags (`--weight-decay`, `--grad-clip`, `--warmup`, `--min-lr`) are
`brain qwen3 finetune --lora`'s.

To serve the result, start `brain serve` with `BRAIN_DEEPSEEKVL_TUNED=<out dir>`:
the adapter is folded into the decoder's weights as they load and the
trained aligner replaces the checkpoint's. The served model
is then the fine-tune.

A fine-tune can instead be kept beside the checkpoint, in
`<model dir>/adapters/<owner>/<name>/<tag>/` (the layout a text adapter has),
and `brain serve` then lists it as a model of its own,
`brain/deepseekvl:<owner>:<name>:<tag>`, next to the base: one process serves
the base and every stored fine-tune, each loaded when first asked for. A
host that cannot hold two of them at once replaces the idle one when the
other is asked for.

As a library: `deepseekvl::train::finetune`, or `Frontend::prepare` and
`Trainer` for your own loop.

The checkpoint is recognized by its HF class `MultiModalityCausalLM`, which
Janus-Pro shares. The generation heads that only Janus-Pro configures tell the
two apart.

Package: `brain-deepseekvl`.
