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
| LoRA fine-tune         | [ ] |
| CLI                    | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [ ] |
| Batched/streaming serving | [ ] |

## Getting the weights

Model id: `brain/qwen3vl` (the served id itself is never fetched, per this
project's naming grammar) - but `Qwen/Qwen3-VL-4B-Instruct` auto-fetches (⤓)
on first CLI use, no env var needed. To point at a different checkpoint, set
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

Only the **dense** Qwen3-VL configuration is built. A GGUF whose
`general.architecture` names an architecture brain has no row for (the
30B-A3B release declares the MoE `qwen3vlmoe`) is refused by that name rather
than loaded as the adjacent dense model.

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

## Hardware and limits

No D-Bus/HTTP serving adapter yet - CLI only, one request at a time, fp32,
greedy decoding. Does not batch concurrent requests. No LoRA/fine-tuning
command yet.

Prefill runs one token at a time, so its cost is linear in the prompt and
close to the card's memory bandwidth. There is no batched prefill and no
int8 decoder tier for this model yet; both are recorded, with measured
numbers, in `.agents/roadmap/vlm.md`.
