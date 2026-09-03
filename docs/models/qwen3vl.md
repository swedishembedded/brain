# Qwen3-VL (image or short video + text -> text)

A general image + text -> text model - ask it a question about an image, not
just "describe this." ViT + PatchMerger + DeepStack vision tower spliced
into a Qwen3 decoder (interleaved M-RoPE). It also accepts a short,
pre-decoded video clip (see [Video input](#video-input) below) instead of a
single image. For dedicated single-purpose captioning instead, see
[FastVLM](fastvlm.md); both are compared on the
[vision-language overview](vlm.md).

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Sampling (temperature/top-k/top-p) | [x] |
| LoRA fine-tune         | [ ] |
| CLI                    | [x] |
| HTTP API               | [x] (OpenAI + Anthropic chat, auto-exposed - `generate`'s shape matches `apiserve::catalog::api_caps`'s chat classification) |
| D-Bus                  | [x] (`Run`/`Subscribe`, `examples/vision/qwen3vl_caption.py`) |
| Batched/streaming serving | [x] (streams tokens; does not batch concurrent requests - see "Hardware and limits" below) |

## Getting the weights

Model id: `brain/qwen3vl` (the served id itself is never fetched, per this
project's naming grammar) - but `Qwen/Qwen3-VL-4B-Instruct` auto-fetches (⤓, opt-in
`--autofetch`) on first CLI use, no env var needed. To point at a different checkpoint, set
`BRAIN_QWEN3VL_WEIGHTS` (overridable per call via the `weights` param) to
either of the two checkpoint layouts brain reads. Which one a path is, is
sniffed, never declared:

**HuggingFace safetensors** - a directory holding `config.json` +
`model.safetensors[.index.json]` + `tokenizer.json`.

**GGUF (llama.cpp)** - a vision-language GGUF release is **two files**, and
both are required:

| File | What it is |
|---|---|
| `Qwen3-VL-4B-Instruct-Q8_0.gguf` | the language half (`general.architecture = "qwen3vl"`) |
| `mmproj-F16.gguf` | the vision tower (`clip.projector_type = "qwen3vl_merger"`) |

Point `weights` at the language half, or at the directory holding both. The
projector is found beside it by its own metadata rather than by filename, so
any of the `mmproj-BF16` / `mmproj-F16` / `mmproj-F32` spellings resolve, and
the pair actually used is printed on every load.

A language half with **no** projector beside it is refused, naming the
missing file. This is deliberate: a decoder without its vision tower is still
a fluent language model, so it would answer "describe this image" with a
confident description of an image it never saw, and nothing downstream could
tell.

```bash
brain pull https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/blob/main/Qwen3-VL-4B-Instruct-Q8_0.gguf
brain pull https://huggingface.co/unsloth/Qwen3-VL-4B-Instruct-GGUF/blob/main/mmproj-F16.gguf
```

Only the **dense** Qwen3-VL configuration is servable. The 30B-A3B release
(`general.architecture = "qwen3vlmoe"`, a top-k sparse-MoE decoder - a
different HF class from this page's dense model, `Qwen3VLMoeForConditionalGeneration`)
is now a **named, recognized architecture** (`crates/qwen3vlmoe`) rather than
an unknown one, so a GGUF carrying that tag gets a specific "not yet
importable, no real release available to verify a tensor mapping against"
error instead of "brain has never heard of this architecture." It is **not**
loadable, servable, or reachable through any `brain` command yet - no import
path, no CLI verb, no capability wiring exist for it.

## Running it

```bash
BRAIN_QWEN3VL_WEIGHTS=/path/to/qwen3-vl \
  brain qwen3vl generate --prompt "Describe this image." --max_new 64 \
    --in image=photo.ppm --out text=answer.txt
```

It is also **served**: with `BRAIN_QWEN3VL_WEIGHTS` set, `brain serve --dbus`
registers a residency adapter (`crates/cli/src/resident_qwen3vl.rs`) that
builds the checkpoint ONCE (device-placed, GPU or CPU) and reuses it across
requests, reachable over D-Bus (`Run`/`Subscribe` -
`examples/vision/qwen3vl_caption.py`) and, because `generate` matches the
chat-capable shape `apiserve::catalog::api_caps` looks for, automatically on
the OpenAI/Anthropic `/v1/chat/completions` and `/v1/messages` surfaces too -
no per-model HTTP or D-Bus code was added for this. `max_pixels` and
`precision` are both part of the resident's identity, so a request that asks
for a bigger capacity or a different precision tier builds (and budgets) a
separate instance rather than silently reusing or evicting another one; see
`resident_qwen3vl.rs`'s module doc for the derived (not measured - no real
checkpoint has been run through this resident on this machine)
`FP32_BYTES`/`INT8_BYTES` footprint arithmetic.

## Options

- `prompt` - the instruction/question.
- `max_new` - max tokens to generate.
- `image` input - a still image: raw HWC f32 pixels in `[0,1]`, with `{w,h}`
  metadata. Exactly one of `image`/`video` is required.
- `video` input + `fps` - a short clip; see [Video input](#video-input).
- `weights` - per-call override of the checkpoint (in place of
  `BRAIN_QWEN3VL_WEIGHTS`): a safetensors directory, a GGUF language half, or
  the directory holding a GGUF pair.

## Multiple images in one request

`generate` takes up to 8 images: `image` (required) plus `image1`..`image7`
(optional), each the same wire shape as `image`. They must be supplied
**contiguous from `image`** - `image` and `image1` set with `image2` absent
is a 2-image request, and a later `image3` is never read. Each image gets its
own smart-resize (they need not share a resolution or aspect ratio) and its
own run of visual tokens in the prompt, in key order, ahead of your question
text.

```bash
BRAIN_QWEN3VL_WEIGHTS=/path/to/qwen3-vl \
  brain qwen3vl generate --prompt "What changed between these two photos?" \
    --in image=before.ppm --in image1=after.ppm --out text=answer.txt
```

`max_pixels` still bounds each image's own resize budget; the resident's
total visual-token capacity is sized for 8 images at that budget each, so a
request combining many large images can still be refused (naming how many
tokens it needed) rather than silently corrupted or truncated.

## Video input

`generate` also accepts a `video` input in place of `image`: N pre-decoded
RGB frames (`capability::blob::decode_video`'s wire format), plus a required
`fps` (the clip's real, constant frame rate). **brain does not decode video
containers** (no mp4/mkv demuxing or codec support anywhere in this crate) -
a caller hands over already-decoded frames, the same contract
`qwen3omnimoe`'s omni provider and `sam2`'s video path already use.

Each frame group's position on the decoder's temporal (T) axis is driven by
its REAL elapsed time (`frame_index / fps`, scaled by the checkpoint's
`tokens_per_second`), not by frame count - two clips with the same frame
count but different real durations get different temporal positions. This
generalizes Qwen2.5-VL's own T-RoPE formula to real per-frame timing; it is
**not** Qwen3-VL's own newer text-token timestamp design (see
`.agents/roadmap/vlm.md` for exactly what upstream mechanism this does and
does not reproduce).

Bounded, deliberately: at most 32 frames per request (a request over the
limit is refused by name, before any checkpoint is touched), no container
decoding, no streaming/hours-long video. A clip without a known `fps` is
refused rather than guessed at, since there is no meaningful real-time
position without one.

Programmatic callers of the `captioner::Captioner` trait (`crates/
captioner`) reach this automatically: a `captioner::Clip` with more than one
frame routes through the video path, using the clip's own `fps`. There is no
`brain label` CLI verb for a video folder yet - `brain label images` remains
images-only (it writes `data::imageset`'s format; a video-folder labeler
writing `data::videoset`'s format is a separate, unbuilt driver, not this
change).

## How long a caption takes, and the one knob that changes it

Both halves of the model run on the placed device: the ViT tower is a second
kernel set on the decoder's own card, not a second device.

Cost is dominated by **prefill**, and prefill is weight-bandwidth bound - the
decoder reads every one of its fp32 weights once per prompt token. So the
number that decides how long a caption takes is how many **visual tokens**
the image becomes, and that is what `--max-pixels` (`brain label images`) or
the `max_pixels` action parameter sets. Halving the pixel budget roughly
halves the time. A smaller budget is less of the image, so it is a real
tradeoff and brain does not pick it for you; `qwen3vl_bench` prints the
cost curve for your own hardware.

`crates/qwen3vl/src/bin/qwen3vl_bench.rs` is the profiler:

```bash
qwen3vl_bench vision                       # the tower at the real geometry, no checkpoint needed
qwen3vl_bench caption --image photo.jpg    # the real checkpoint, per stage, with tok/s
```

`caption` mode drives the same served path `brain label images` does, and
reports each stage against the machine's own measured roofline plus the
weight-bandwidth ceiling a batch-1 decode cannot beat.

## The int8 decoder tier

`--precision int8` (or the `precision` action parameter) builds the decoder's
per-layer linears as packed int8 instead of fp32. The vision tower and the LM
head stay fp32. Because captioning is weight-bandwidth bound, reading a
quarter of the weight bytes per token is a large speed-up.

**It is lossy, it is never the default, and it should not be chosen from a
speed number alone.** Quantization perturbs the decode hidden state enough to
flip a greedy argmax, and one flipped token rewrites the rest of the caption.
In practice int8 and fp32 captions of the same photograph usually differ.
Many differences are cosmetic - the same scene described in a different
order, or as prose instead of a list - but some are not: the two tiers can
name a different object, or a different number of them, in the same part of
the picture. If the captions are training data, that is the decision, not the
speed.

```bash
qwen3vl_bench compare --dir <dir>     # both tiers, side by side, with the divergence
```

That mode captions the same images at both tiers and prints both texts in
full alongside the time and a word-overlap figure. Read the captions: a
similarity score cannot tell you whether a difference is cosmetic or
substantive, and that is the whole question.

Every load states the tier that ACTUALLY ran. A device that cannot serve a
packed int8 dot has the request promoted back to fp32, and that is reported
as a warning rather than passing silently.

## Sampling

`temp`/`top_k`/`top_p`/`seed` action parameters (same names, defaults and
bounds as `qwen3::caps`) select the decode policy; `temp <= 0` (the default)
is deterministic greedy argmax, matching this action's original behaviour.
A real `temp` samples via the same temperature/top-k/nucleus algorithm
`qwen3::sample` uses.

## Context

Decoder context is a resident-build-time capacity, not a per-request
parameter (the same philosophy `max_pixels` already uses): `$BRAIN_QWEN3VL_CTX`
(default 24576, mirroring `qwen3`'s own `BRAIN_QWEN_CTX`) sizes the KV cache
this resident's decoder is built with, clamped DOWN to the checkpoint's own
declared `max_position_embeddings` so a request never allocates past what the
checkpoint was actually trained for. A request whose prompt + `max_new`
exceeds that resident's built context is refused BY NAME, naming both the
built capacity and the checkpoint's real ceiling - never silently truncated.

This decode path allocates a plain linear fp32 KV cache
(`Qwen::new_shard_dt_decode`), not the paged/int8 cache `qwen3::serve::Engine`
uses to reach a real checkpoint's native (typically 262144-token) length
affordably - at the 4B config's shape, the full native length would be tens
of gigabytes of KV cache alone for one request. Reaching that native length
here (rather than just raising the env default within what a card can hold)
needs the same paged-KV/M-RoPE-aware serving `qwen3vl` does not have yet -
see `.agents/roadmap/vlm.md`.

## Hardware and limits

Served (CLI, D-Bus, HTTP) one request at a time, fp32 by default (int8
decoder opt-in), with real temperature/top-k/top-p sampling (greedy by
default - see "Sampling" above). `run_batch` is the documented serial
default: unlike Moondream 3 (whose vision tower attends within each crop
independently and so batches across requests), Qwen3-VL's vision tower feeds
directly into the decoder's own incremental KV-cache splice, so every
request is its own multi-step decode with its own prompt, its own
image-token placement and its own KV cache - there is no stage that is both
shared across requests and independent of each one's state. No
LoRA/fine-tuning command yet.

Prefill runs one token at a time, so its cost is linear in the prompt and
close to the card's memory bandwidth. There is no batched prefill and no
batching across images, so a folder is captioned one photograph at a time;
both are recorded, with measured numbers and the memory arithmetic that
bounds them, in `.agents/roadmap/vlm.md`.
