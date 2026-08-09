# SAM 2.1 (`crates/sam2`)

Promptable image segmentation: point and box prompts on an image produce a
mask. Image path only — no video memory-bank tracking (the reference's
propagate-through-video mode is not implemented).

## Model id and weights

- **Id:** `brain/sam2` — reserved vendor `brain/`, never fetched
  (brain_modelstore auto-fetch applies to the qwen/glm/lfm-family models,
  not this one).
- **Weights:** `BRAIN_SAM2_WEIGHTS` — a `sam2.1_hiera_{tiny,large}.pt`
  release checkpoint (or an equivalent `.safetensors`); must exist on disk
  or the resident does not register at all.
- **Variant:** `BRAIN_SAM2_VARIANT` — `tiny` (default) or `large`. A call's
  `variant` param overrides this per request, which keys a separate
  resident instance.

## Surfaces

D-Bus and `brain do` only. The one action, `segment`, requires an input
`image` blob, so it fails HTTP's image-generation shape (needs a pure
text→image action with no required input) and is not named `generate` or
`embed`, so it is simply absent from `/v1/chat/completions`,
`/v1/images/generations` and `/v1/embeddings` — not listed, not broken.

## Inference

### D-Bus / `brain do`

No CLI verb. One action, `segment`: params `variant` (`tiny`|`large`,
default `tiny`), `points` (`"x,y;x,y;…"` in source-image pixels), `labels`
(`"1;0;…"`, 1=foreground/0=background, default all-foreground), `box`
(`"x1,y1,x2,y2"`), `multimask` (bool, default `true`); required input
`image` (`Media::Image`); output `mask` (`Media::Mask`, sigmoid probability
on the source-image grid — threshold at 0.5). Batches BY IMAGE, not by
request: N prompts on one frame cost one Hiera trunk pass plus N decoder
passes (`resident_sam2::Sam2Instance::run_batch`).

```bash
brain caps brain/sam2
BRAIN_SAM2_WEIGHTS=<ckpt>/sam2.1_hiera_tiny.pt \
  dbus-run-session -- bash -c 'brain serve --dbus & sleep 3
    python3 examples/vision/segment_image.py --image photo.ppm --point 614,430 --concurrent 4'

# or with no bus at all:
BRAIN_SAM2_WEIGHTS=<ckpt>/sam2.1_hiera_tiny.pt \
  brain do brain/sam2 segment --points "614,430" \
    --in image=photo.ppm --out mask=mask.ppm --json
```
Reference client: [`examples/vision/segment_image.py`](../../../examples/vision/segment_image.py)
(prompts are in source-image pixels; `--point x,y` repeatable, `--box
x1,y1,x2,y2`, `--concurrent N` submits N prompts at once to exercise the
image-batched path).

## Training

`crates/sam2/src/train.rs` implements a real, finite-difference-gradchecked
mask-decoder trainer (`gradcheck::check_sam2` / `check_sam2_on`, trunk + FPN
neck frozen, 128 tensors), but there is no CLI verb — no command to give.

## Not supported

training (verb), finetune, LoRA, QLoRA, HTTP, video/memory-bank tracking
(forward-only, image path only)

## See also

- Crate: `crates/sam2`
- Ledger: [docs/imaging/plan.md](../../imaging/plan.md) §4 "Phasing" (no
  sam2-specific status.md exists)
