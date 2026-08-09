# Real-ESRGAN 4x super-resolution (`crates/upscale`)

Super-resolves an image with the Real-ESRGAN RRDBNet generator (the
discriminator is training-time only and not part of the released checkpoint —
see [Not supported](#not-supported)).

## Model id and weights

- **Id:** `brain/upscale` — reserved vendor `brain/`, never fetched.
- **Weights:** `BRAIN_ESRGAN_WEIGHTS` — a released RRDBNet checkpoint file, e.g.
  `RealESRGAN_x4plus.pth` (the shape is derived from the tensors, so `x2plus`
  and `x4plus_anime_6B` also work).

## Surfaces

CLI and D-Bus. Not HTTP: the action is `upscale`, not `generate` (no `chat`),
and it requires an `image` input blob, so it does not qualify as the `image`
(text-to-image) capability either — correctly absent from `/v1/models` and
`/v1/images/generations`.

## Inference

### CLI
No dedicated `brain upscale` verb. Use the generic pair:
```bash
brain caps brain/upscale
brain do brain/upscale upscale --tile 0 --in image=photo.ppm --out image=upscaled.ppm --json
```

### D-Bus
One action, `upscale`: required input `image` (`Media::Image`, RGB in `[0,1]`),
one int param `tile` (default `0` = whole image; process in tiles of this many
input pixels a side when the image is too large for peak VRAM at once), output
`image` (RGB in `[0,1]`, `scale`x the input — 4x on the released `x4plus`
checkpoint).

There is no standalone example script for `brain/upscale` — it is normally
exercised as the tail stage of `brain/imgpipe`:
```bash
BRAIN_SAM2_WEIGHTS=... BRAIN_RESTORE_WEIGHTS=... BRAIN_ESRGAN_WEIGHTS=... \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 3
    python3 examples/imaging/edit_pipeline.py --image photo.ppm --point 614,430 --upscale'
```
Reference client: [`examples/imaging/edit_pipeline.py`](../../../examples/imaging/edit_pipeline.py)
(`--upscale`, `--tile N`). To drive `brain/upscale` alone over D-Bus with no
example script, use `brain_dbus.py`'s `Run` directly with the args above.

## Not supported

`training`, `finetune`, `LoRA`, `QLoRA`, `batch > 1`, `HTTP`. The discriminator
is a training-time-only component and was never part of the released
checkpoint, so there is no training path here at all — not even a lib-only
trainer.

## See also

- Crate: [`crates/upscale`](../../../crates/upscale)
- No dedicated `status.md` exists for this crate; the closest ledger is
  [`docs/imaging/plan.md`](../../imaging/plan.md).
- Composed pipeline this feeds as a tail stage: [`../imgpipe/readme.md`](../imgpipe/readme.md)
