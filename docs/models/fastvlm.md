# FastVLM (image captioning)

Apple's FastVLM: a dedicated image captioner - point it at an image and get
a description back. FastViTHD hybrid conv/attention vision tower + an
`mlp2x_gelu` projector spliced into a Qwen2-configured decoder. For a general
"ask a question about this image" model instead of single-purpose
captioning, see [Qwen3-VL](qwen3vl.md); both are compared on the
[vision-language overview](vlm.md).

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [ ] |
| CLI                    | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched/streaming serving | [x] (streams tokens over CLI/D-Bus; does not batch concurrent requests) |

## Getting the weights

Model id: `brain/fastvlm` (a `brain/`-namespaced served id, per this
project's naming grammar, is never itself fetched from the network) - but
`apple/FastVLM-0.5B` auto-fetches (⤓, opt-in `--autofetch`) on first CLI use, no env var needed.
To point at a different checkpoint, set `BRAIN_FASTVLM_WEIGHTS` to a
directory holding `config.json` + `model.safetensors` + `tokenizer.json`
(overridable per call via the `weights` param).

## Running it

```bash
BRAIN_FASTVLM_WEIGHTS=/path/to/fastvlm \
  brain fastvlm caption --prompt "Describe this image." --max_new 48 \
    --in image=photo.ppm --out text=caption.txt
```

Over D-Bus:

```python
from brain_py.dbus import BrainDBus
with BrainDBus() as brain:
    out = brain.subscribe(
        "brain/fastvlm", "caption", {"prompt": "Describe this image.", "max_new": 48},
        blobs={"image": hwc_f32_bytes},
        meta={"image": {"media": "Image", "meta": {"w": w, "h": h}}},
    )
```

See [`examples/dbus/brain_dbus.py`](../../examples/dbus/brain_dbus.py) for the
connection setup this snippet builds on.

## Options

- `prompt` - the instruction (default `"Describe this image."`).
- `max_new` - max tokens to generate (default `48`).
- `precision` - `fp32` (default) or `int8` for the language decoder; the
  vision tower always runs fp32 regardless of this setting.
- `image` input - raw HWC f32 pixels in `[0,1]`, with `{w,h}` metadata.
- `weights` - per-call override of the checkpoint directory (in place of
  `BRAIN_FASTVLM_WEIGHTS`).

## Hardware and limits

`caption` is not chat-shaped (it isn't named `generate`), so it is not
exposed on the OpenAI/Anthropic-compatible HTTP APIs - CLI and D-Bus only.
Does not batch concurrent requests. No LoRA/fine-tuning command yet.
