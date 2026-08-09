# Vision-language models (`crates/fastvlm`, `crates/qwenvl`, `crates/moondream`)

FastVLM is the only VLM in brain with a working generation loop. Qwen3-VL-4B and
Moondream 3 also live under `crates/` and share the same vision→projector→decoder
shape, but are forward-only today — see the note at the bottom.

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

## Qwen3-VL-4B and Moondream 3 — not yet runnable

`crates/qwenvl` and `crates/moondream` implement the same vision-encoder →
projector → decoder shape, gradient-checked and import-covered, but neither has
ever emitted a generated token: each exposes only a training `forward()` that
returns a scalar loss (`crates/qwenvl/src/model.rs:128`,
`crates/moondream/src/model.rs:71`/`:88`), with no `generate`/`greedy`/`max_new`/
KV-decode function anywhere in either crate — and `qwenvl` is not even a
dependency of `crates/cli`. See `crates/qwenvl` and `crates/moondream` directly,
and the capability matrix in `docs/models/vlm/VALIDATION.md`.

## See also

- Crate: `crates/fastvlm`
- Workstream ledger: [`VALIDATION.md`](VALIDATION.md) — this directory has no
  `status.md`; `VALIDATION.md` is the equivalent ledger.
