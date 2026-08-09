# CLIP family (`crates/clip`)

Text and image encoders used as conditioning towers by other models: CLIP-L
and OpenCLIP-bigG/14 (SDXL's two text encoders; CLIP-L alone is also FLUX.1's
pooled-vector encoder) and EVA-CLIP-L/336 (the image tower PuLID conditions
on).

## Model id and weights

- **Id:** `brain/clip` — reserved vendor `brain/`, never fetched.
- **Weights:** `BRAIN_CLIP_DIR` — a checkpoint root in the SDXL layout:
  `text_encoder/` (CLIP-L) and/or `text_encoder_2/` (OpenCLIP-bigG) weight
  dirs, `tokenizer/` and/or `tokenizer_2/` BPE dirs (at least one tokenizer
  dir must exist), and `EVA02_CLIP_L_336_psz14_s6B.pt` at the root for the
  image tower (`embed_image` fails without it; `embed_text` never touches
  it).

## Surfaces

D-Bus and `brain do` only. Not HTTP: `POST /v1/embeddings` always dispatches
an action literally named `"embed"` (`crates/apiserve/src/openai.rs`), and
this manifest has neither action named that (`embed_text`/`embed_image`) —
`crates/apiserve/src/catalog.rs`'s embeddings classifier requires the exact
name for precisely this reason, so `brain/clip` is correctly absent from
`/v1/models` rather than listed and then failing. Not chat (no `generate`
action) and not image-generation (no `Image` output, no `prompt` param).

## Inference

### D-Bus / `brain do`

No CLI verb. Two actions:
- `embed_text` — params `text` (required), `tower` (`clip_l`, 768-d,
  default; or `openclip_bigg`, 1280-d); output `embedding` (`Media::Bytes`,
  f32 little-endian) — the projected `text_embeds` when the tower has a
  projection head, else the pooled EOS row.
- `embed_image` — required input `image` (`Media::Image`, resized to the
  336² EVA-CLIP-L input); output `embedding` (`Media::Bytes`, f32
  little-endian, L2-normalised CLS embedding).

```bash
brain caps brain/clip
BRAIN_CLIP_DIR=<ckpt-root> brain do brain/clip embed_text --text "a photo of a cat" \
    --tower clip_l --out embedding=text.bin --json
BRAIN_CLIP_DIR=<ckpt-root> brain do brain/clip embed_image \
    --in image=photo.ppm --out embedding=img.bin --json
```

No dedicated example script exists under `examples/` — `examples/embedding/`
is LFM2.5-Encoder's, not CLIP's, and nothing else in the tree drives
`embed_text`/`embed_image`. Use `brain_dbus.py`'s `Run` directly over a
private session bus, or the no-bus `brain do` form above.

## Training

Text towers only: `ClipText::new_train_on` adds the reverse pass, gated by
`gradcheck::check_clip` (CLIP-L shape), `check_clip_bigg` (OpenCLIP-bigG
shape), `check_clip_tiled` (tiled backward GEMMs) — but there is no CLI
verb, no command to give. The EVA-CLIP-L/336 **image** tower is forward-only
end to end (`crates/clip/src/lib.rs`): no training graph exists for it at
all.

## Not supported

training (verb) for the image tower entirely, finetune, LoRA, QLoRA, HTTP,
batch > 1 for `embed_image` (the EVA tower is built at b=1; `embed_text`
batches for real via `Session::embed_text_batch`)

## See also

- Crate: `crates/clip`
- Ledger: [docs/imaging/plan.md](../../imaging/plan.md) (no clip-specific
  status.md exists) — Phase 3b table and §3.3 "Training / finetuning"
