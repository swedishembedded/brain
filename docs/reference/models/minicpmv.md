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
D-Bus, which also takes a `video` blob of decoded frames (with meta `{fps}`
the frames are sampled as the reference processor does; without it they are
used as given, thinned evenly to `max_frames`). Each image part becomes a placeholder where it stands; an image sent
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
brain minicpmv generate --video_path clip.mp4 --prompt "What happens in this video?"
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
| `--video_path FILE` | a video on this host: frames picked by index from the container's metadata - one per second, or every tenth of a second thinned to `max_frames` for a clip longer than that many seconds |
| `--max_frames N` | most frames one video contributes (default 128) |

## Fine-tuning

```bash
brain minicpmv finetune --dataset train.jsonl --held-out val.jsonl --label acme:receipts:v1
brain serve --openai 8789          # now also lists brain/minicpmv:acme:receipts:v1
brain minicpmv export-peft --tune acme:receipts:v1 --out receipts-peft
```

The dataset is `generic-messages-v2` JSONL - one conversation per line,
every message with an explicit `train` flag - whose message content may be a
list of parts, media referenced by path relative to the file:

```json
{"messages": [
  {"role": "user", "train": false, "content": [
    {"type": "image", "path": "images/0001.png"},
    {"type": "text", "text": "What is the total on this receipt?"}]},
  {"role": "assistant", "train": true, "content": "42.50 EUR"}]}
```

Each assistant turn marked `train` is one training sample, rendered as the
model is asked at inference (the checkpoint's template, no thinking block)
with only the answer supervised. Every record, file and turn is checked
before the first step; a bad one fails naming its line.

Each part trains on its own:

| flag | part | modes (default first) |
|---|---|---|
| `--decoder` | the Qwen3.5 decoder | `lora:16:32` (rank:alpha), `full`, `frozen` |
| `--freeze-connector` | the merger | trained unless given |
| `--vision` | the SigLIP tower and window merger | `frozen`, `lora:R:A` (on every layer's qkv, out, fc1, fc2), `full` |

| option | effect |
|---|---|
| `--epochs N` / `--batch N` | passes over the data (default 3) and turns per AdamW step (default 1) |
| `--lr X` | peak rate of the decoder and of a LoRA tower (default 1e-4) |
| `--connector-lr X`, `--vision-lr X` | peak rates of the connector (default a tenth of `--lr`) and of the tower (`--lr` under LoRA, a fiftieth in full): Adam moves every weight by about the rate each step, and these weights are small enough that a LoRA-sized rate scrambles them |
| `--warmup N`, `--min-lr-ratio R` | the cosine schedule every part follows: warmup over N steps (default a twentieth of the run), then down to R of each peak (default 0.1) |
| `--weight-decay W`, `--grad-clip C` | AdamW decay (default 0) and global gradient clip (default 1; 0 disables) |
| `--block T` | tokens per sample (default: the longest sample) |
| `--downsample`, `--max-slices`, `--max-frames` | the processor options, as for `generate` |
| `--checkpoint-every N`, `--resume` | save the whole training state every N steps (always at each epoch's end); `--resume` continues from it, drawing the same batches |
| `--label owner:name:tag` / `--out DIR` | where the tune goes: beside the checkpoint in the model store under the label (served as `brain/minicpmv:<label>`), or a directory |
| `--activations keep|recompute` | what the backward keeps of the forward in the parts that train: the decoder as `qwen35` does, and a vision tower that trains always rebuilds its blocks, so `keep` is refused by name there; absent leaves each part's default |
| `--device gpuN`, `--seed S` | the card; the seed of the data order and adapter initialisation (random and printed when not given) |

A tune directory holds `tune.json`, the decoder's LoRA adapter (or its full
weights), and the vision side's trained tensors and LoRA factors. Serving
applies it as the model loads: the vision LoRA and the decoder adapter are
folded into the weights, so a tune runs exactly as fast as the base.
`export-peft` writes it as one PEFT adapter over the Hugging Face checkpoint:
the decoder's and the tower's LoRA pairs under their HF modules (a pair on
the tower's fused `qkv` becomes three that share `A`), a connector or tower
trained in full as `modules_to_save`. A decoder trained in full is no adapter
and is refused.

Training runs in fp32 on one card. With tower LoRA the tower keeps its
full-size gradient buffers and a copy of its base weights beside the
decoder: about 15 GB, and a supervised turn with one 448-pixel view takes
several seconds on a P40. Training the tower is what lets a task that hinges on what
the image shows learn quickly: on the codeword task below, 32 steps with
tower LoRA reach a held-out loss of 0.0056, while the same budget with the
tower frozen learns the answers' form but stays at chance on which answer
the image calls for.

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
- **Video**: the frame indices for every recorded clip metadata (short clips,
  long ones thinned to the budget, odd frame rates), and a synthetic clip end
  to end - video features at cosine 1.0000000000 and the greedy reply
  identical.
- **Training gradients**: every vision-side gradient (tower, window merger,
  merger, tower LoRA, at 16x and 4x) and the composite's (decoder LoRA,
  merger and tower LoRA through the splice) against finite differences of
  the loss.
- **Training behaviour** (a tiny model): frozen parts stay bit-identical
  while trained parts move, the loss falls on an overfit run, and a run
  saved and restored halfway ends bit-identical to an uninterrupted one.
- **Fine-tuning the real checkpoint**: a synthetic task (the colour of a disc
  on noise names a made-up codeword) with decoder LoRA, the connector and
  tower LoRA all training (batches of 3, 8 epochs at `--lr 5e-4`) cuts the
  held-out loss on unseen images from 11.6 to 0.0056, and the saved tune,
  loaded as serving loads it, answers every unseen image with its codeword.
- **Training forward and reload**: the trainer's label loss for the golden
  photo and answer is the reference's (0.0396951 vs 0.0396935), and with
  every trained part moved, the next-token loss the trainer computes and the
  one the saved tune gives when served are the same number.
- **PEFT export**: a deterministic tune (decoder LoRA on all twelve
  projections, tower LoRA at another rank and scale, the connector in full)
  exported and applied to the reference model by
  `tools/goldens/minicpmv_peft_reference.py` gives the next-token logits
  brain gives with the tune itself (cosine 1.0000000000, rel_l2 6.5e-5)
  and the same greedy reply. Without the `peft` package installed the script
  applies the adapter by PEFT's LoRA rule (`W += lora_alpha / r * B A`,
  `modules_to_save` loaded in place) and records that it did.

## Limits

- `stack_frames > 1` is refused: the reference builds its per-second
  composites and then processes the raw frames, so its own forward fails on
  its output and there is nothing correct to match.
- The OpenAI HTTP surface carries images, not video; video arrives over
  D-Bus (`video` blob) or the CLI (`--video_path`).
- Images are decoded by brain's own codecs; a JPEG can decode a pixel or two
  differently from PIL's libjpeg, which can move a long greedy continuation.
- PEFT export needs at least one LoRA part, and a decoder trained in full
  is not exported (its weights stay in the tune directory).

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
