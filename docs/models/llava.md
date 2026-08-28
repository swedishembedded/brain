# LLaVA-1.5-13B

LLaVA-1.5-13B pairs a CLIP-L/14@336 vision tower with a Vicuna-1.5 (LLaMA-2
13B) decoder to answer questions about, or produce a detailed caption for, an
image. It is brought into this workspace as [SUPIR](supir.md)'s optional
captioner - SUPIR's restoration prompt can come from LLaVA describing the
low-quality input, or from a prompt supplied directly. For a general
"ask a question about this image" model compared against, see
[FastVLM](fastvlm.md)/[Qwen3-VL](qwen3vl.md); all three are compared on the
[vision-language overview](vlm.md).

## Status

The graph, tokenizer, config presets, `mlp2x_gelu` projector, `vicuna_v1`
prompt template, image-token splice, INT8 decoder path and serving action are
implemented and weight-free gated. **Not yet exercised against real
checkpoint bytes** - LLaVA-1.5-13B is a multi-ten-GB download and none was
fetched this session; see `.agents/roadmap/llava.md` for the specific gap and
what would close it.

## Support

| Capability | Supported |
|---|---|
| Inference (`caption`) | [x] (untested against real weights - see Status) |
| INT8                   | [x] (`qwen3::Qwen::new_shard_i8`, reused unmodified) |
| CLI (`brain <arch> <action>`)       | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

Model id: `brain/llava` (a `brain/`-namespaced served id is never itself
fetched from the network, and this architecture has no `default_ref` - point
it at weights you obtained yourself). Set `BRAIN_LLAVA_WEIGHTS` to a directory
holding `config.json` + `model.safetensors` + `tokenizer.json` (overridable
per call via the `weights` param).

## Running it

```bash
BRAIN_LLAVA_WEIGHTS=/path/to/llava-v1.5-13b \
  brain llava caption --prompt "Describe this image and its style in a very detailed manner." \
    --max_new 128 --in image=photo.ppm --out text=caption.txt
```

Over D-Bus:

```python
from brain_py.dbus import BrainDBus
with BrainDBus() as brain:
    out = brain.subscribe(
        "brain/llava", "caption",
        {"prompt": "Describe this image and its style in a very detailed manner.", "max_new": 128},
        blobs={"image": hwc_f32_bytes},
        meta={"image": {"media": "Image", "meta": {"w": w, "h": h}}},
    )
```

See [`examples/dbus/brain_dbus.py`](../../examples/dbus/brain_dbus.py) for the
connection setup this snippet builds on.

## Options

- `prompt` - the instruction (default the SUPIR caption prompt, "Describe
  this image and its style in a very detailed manner.").
- `max_new` - max tokens to generate (default `128`).
- `precision` - `fp32` (default) or `int8` for the Vicuna decoder; the
  CLIP-L336 vision tower always runs fp32 regardless of this setting.
- `image` input - raw HWC f32 pixels in `[0,1]`, with `{w,h}` metadata.
- `weights` - per-call override of the checkpoint directory (in place of
  `BRAIN_LLAVA_WEIGHTS`).

## Hardware and limits

`caption` is not chat-shaped (it isn't named `generate`), so it is not
exposed on the OpenAI/Anthropic-compatible HTTP APIs - CLI and D-Bus only.
Does not batch concurrent requests. No LoRA/fine-tuning command. Multi-turn
conversation beyond a single caption call is out of scope (SUPIR only ever
needs one caption per image) - see `.agents/roadmap/llava.md`.
