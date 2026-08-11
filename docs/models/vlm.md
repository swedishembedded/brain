# Vision-Language Models (FastVLM + Qwen3-VL)

Two image-understanding models, both image (+ text) in, text out. **FastVLM**
is a dedicated image captioner — point it at an image and get a description
back. **Qwen3-VL** is a general image + text → text model — ask it a
question about an image, not just "describe this." Reach for FastVLM for
fast, single-purpose captioning; reach for Qwen3-VL when you need to prompt
about an image's content.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [ ] |
| CLI (`brain do`)       | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] (FastVLM only) |
| Batched/streaming serving | [x] (FastVLM streams tokens over CLI/D-Bus; neither model batches concurrent requests) |

## Getting the weights

- **FastVLM** — id `brain/fastvlm`. `BRAIN_FASTVLM_WEIGHTS` — a checkpoint
  directory holding `config.json` + `model.safetensors` + `tokenizer.json`
  (overridable per call via the `weights` param).
- **Qwen3-VL** — id `brain/qwenvl`. `BRAIN_QWENVL_WEIGHTS` — a checkpoint
  directory holding `config.json` + `model.safetensors[.index.json]` +
  `tokenizer.json` (overridable per call via the `weights` param).

Both are reserved vendor `brain/` — never auto-fetched.

## Running it

FastVLM — caption an image:

```bash
BRAIN_FASTVLM_WEIGHTS=/path/to/fastvlm \
  brain do brain/fastvlm caption --prompt "Describe this image." --max_new 48 \
    --in image=photo.ppm --out text=caption.txt
```

FastVLM over D-Bus:

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

Qwen3-VL — ask a question about an image:

```bash
BRAIN_QWENVL_WEIGHTS=/path/to/qwen3-vl \
  brain do brain/qwenvl generate --prompt "Describe this image." --max_new 64 \
    --in image=photo.ppm --out text=answer.txt
```

## Options

- `prompt` — the instruction/question (FastVLM defaults to
  `"Describe this image."`).
- `max_new` — max tokens to generate (FastVLM default `48`).
- `precision` (FastVLM only) — `fp32` (default) or `int8` for the language
  decoder; the vision tower always runs fp32 regardless of this setting.
- `image` input — raw HWC f32 pixels in `[0,1]`, with `{w,h}` metadata.
- `weights` — per-call override of the checkpoint directory (in place of the
  `BRAIN_*_WEIGHTS` env var).

## Hardware and limits

- FastVLM's `caption` action is not chat-shaped (it isn't named `generate`),
  so it is not exposed on the OpenAI/Anthropic-compatible HTTP APIs; only
  CLI and D-Bus.
- Qwen3-VL has no D-Bus/HTTP serving adapter yet — CLI (`brain do`) only, one
  request at a time, fp32, greedy decoding.
- Neither model batches concurrent requests.
- No LoRA/fine-tuning command for either model today.
