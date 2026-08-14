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

Model id: `brain/qwenvl` (reserved vendor - never auto-fetched).
`BRAIN_QWEN3VL_WEIGHTS` - a checkpoint directory holding `config.json` +
`model.safetensors[.index.json]` + `tokenizer.json` (overridable per call
via the `weights` param).

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
- `weights` - per-call override of the checkpoint directory (in place of
  `BRAIN_QWEN3VL_WEIGHTS`).

## Hardware and limits

No D-Bus/HTTP serving adapter yet - CLI only, one request at a time, fp32,
greedy decoding. Does not batch concurrent requests. No LoRA/fine-tuning
command yet.
