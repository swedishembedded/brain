# Vision-language models (`crates/fastvlm`, `crates/qwenvl`, `crates/moondream`)

Two VLMs in brain have a working generation loop: FastVLM (`brain/fastvlm`,
captioning) and Qwen3-VL-4B (`brain/qwenvl`, image + text → text — see its
section below). Moondream 3 shares the same vision→projector→decoder shape but
is forward-only today — see the note at the bottom.

## Model id and weights

- **Id:** `brain/fastvlm` — reserved vendor `brain/`, never auto-fetched.
- **Weights:** `BRAIN_FASTVLM_WEIGHTS` — a checkpoint **directory** holding
  `config.json` + `model.safetensors` + `tokenizer.json`; overridable per call via
  the `weights` param.

## Surfaces

CLI (`brain do`) and D-Bus (registered as a stateless resident) — not HTTP. Its
one action is named `caption`, not `generate`, so it is correctly excluded from
HTTP chat; its required `image` input blob excludes it from text-to-image too.
(Before the Phase-0 fix `caption` was incorrectly advertised on chat and would
404/400 — it no longer is.)

## Inference

### CLI

No dedicated `brain fastvlm` verb; the generic pair:

```bash
brain caps brain/fastvlm
brain do brain/fastvlm caption --prompt "Describe this image." --max_new 48 \
    --precision fp32 --in image=photo.ppm --out text=caption.txt
```

One action, **`caption`** (`.streaming()`, per-token `Progress`):
- params: `weights` (default `$BRAIN_FASTVLM_WEIGHTS`), `prompt` (default
  `"Describe this image."`), `max_new` (default `48`), `precision` (default
  `"fp32"`, or `"int8"` — per-channel weights + dynamic activation quant, but
  **only on the Qwen2 decoder**; the MobileCLIP-L vision tower always runs fp32
  on CPU regardless of `precision`).
- input: `image` (required, `Media::Image`, raw HWC f32 pixels in `[0,1]`, meta
  `{w,h}`).
- output: `text`.

### D-Bus

Registered as a stateless resident (`FastVlmProvider`, see
`crates/cli/src/resident.rs`) — the same `caption` action over the generic
`Run`/`Subscribe` methods. There is no example script yet for FastVLM
specifically; use `brain_dbus.py`'s generic client directly:

```python
from brain_py.dbus import BrainDBus
with BrainDBus() as brain:
    out = brain.subscribe(
        "brain/fastvlm", "caption", {"prompt": "Describe this image.", "max_new": 48},
        blobs={"image": hwc_f32_bytes},
        meta={"image": {"media": "Image", "meta": {"w": w, "h": h}}},
    )
```

See [`examples/dbus/brain_dbus.py`](../../../examples/dbus/brain_dbus.py) for the
connection setup this snippet builds on.

## Training / Fine-tune / LoRA

A full training loop exists (`crates/fastvlm/src/train_smoke.rs`: `zero_grads` →
image-splice `forward` → `backward` → AdamW, driving loss 3.12 → 0.01 on a single
image→caption example — see `docs/models/vlm/VALIDATION.md`) but has no CLI verb —
no command to give.

## Not supported

`training` (verb), `finetune`, `LoRA`, `QLoRA`, `batch > 1`, `HTTP`.

## Qwen3-VL-4B (`brain/qwenvl`) — servable, validation-tier

`crates/qwenvl` serves a real `generate` action (registered in
`crates/cli/src/catalog.rs`): smart-resize preprocessing, image splice, and a
KV-cache greedy decode with M-RoPE/DeepStack (`Qwen3Vl::generate`). fp32,
greedy argmax only, one request at a time.

- **Weights:** `BRAIN_QWENVL_WEIGHTS` — a checkpoint directory (`config.json` +
  `model.safetensors[.index.json]` + `tokenizer.json`); overridable per call via
  the `weights` param.
- **Surfaces:** `brain caps`/`brain do` only — no residency adapter yet, so no
  D-Bus/HTTP (the same state fastvlm started in):

```bash
brain do brain/qwenvl generate --prompt "Describe this image." --max_new 64 \
    --in image=photo.ppm --out text=answer.txt
```

## Moondream 3 — not yet runnable

`crates/moondream` implements the same vision-encoder → projector → decoder
shape, gradient-checked and import-covered, but has never emitted a generated
token: it exposes only a training `forward()` that returns a scalar loss
(`crates/moondream/src/model.rs:71`/`:88`), with no `generate`/`greedy`/
`max_new`/KV-decode function anywhere in the crate. See `crates/moondream`
directly, and the capability matrix in `docs/models/vlm/VALIDATION.md`.

## See also

- Crate: `crates/fastvlm`
- Workstream ledger: [`VALIDATION.md`](VALIDATION.md) — this directory has no
  `status.md`; `VALIDATION.md` is the equivalent ledger.
