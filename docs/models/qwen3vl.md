# Qwen3-VL (image + text -> text)

A general image + text -> text model - ask it a question about an image, not
just "describe this." ViT + PatchMerger + DeepStack vision tower spliced
into a Qwen3 decoder (interleaved M-RoPE). For dedicated single-purpose
captioning instead, see [FastVLM](fastvlm.md); both are compared on the
[vision-language overview](vlm.md).

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Sampling (temperature/top-k/top-p) | [x] |
| LoRA fine-tune         | [ ] |
| CLI                    | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [ ] |
| Batched/streaming serving | [ ] |

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

## Options

- `prompt` - the instruction/question.
- `max_new` - max tokens to generate.
- `image` input - raw HWC f32 pixels in `[0,1]`, with `{w,h}` metadata.
- `weights` - per-call override of the checkpoint (in place of
  `BRAIN_QWEN3VL_WEIGHTS`): a safetensors directory, a GGUF language half, or
  the directory holding a GGUF pair.

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

No D-Bus/HTTP serving adapter yet - CLI only, one request at a time, fp32
by default (int8 decoder opt-in). Does not batch concurrent requests. No
LoRA/fine-tuning command yet.

Prefill runs one token at a time, so its cost is linear in the prompt and
close to the card's memory bandwidth. There is no batched prefill and no
batching across images, so a folder is captioned one photograph at a time;
both are recorded, with measured numbers and the memory arithmetic that
bounds them, in `.agents/roadmap/vlm.md`.
