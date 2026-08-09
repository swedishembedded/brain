# Imaging pipeline (`crates/imgpipe`)

Composes segment -> refine mask -> restore -> optional upscale tail into ONE
capability call, so intermediates (mask, restored image) never cross the
process/bus boundary and pixels outside the mask come back bit-identical. Each
stage dispatches into the same providers `brain caps`/`brain do` use for the
underlying models — it holds no weights of its own.

## Model id and weights

- **Id:** `brain/imgpipe` — reserved vendor `brain/`, never fetched.
- **Weights:** no env var of its own. Each stage resolves its own model's
  weights the normal way, and a stage whose model is unconfigured fails with
  THAT model's own error:
  - `segment` -> `BRAIN_SAM2_WEIGHTS`
  - `restore` -> `BRAIN_RESTORE_WEIGHTS`
  - `upscale` -> `BRAIN_ESRGAN_WEIGHTS`

## Surfaces

CLI and D-Bus — registered as a stateless resident in
`crates/cli/src/resident.rs` (built from `crate::catalog::provider(imgpipe::caps::MODEL)`,
the same stage registry `brain do` uses, so nothing can drift between the two
paths). Not HTTP: the action is `run`, not `generate` (no `chat`), and it
requires an `image` input blob, so it does not qualify as the `image`
(text-to-image) capability either.

## Inference

### CLI
No CLI verb beyond the generic pair:
```bash
brain caps brain/imgpipe
brain do brain/imgpipe run \
  --stages '{"stages":[{"op":"segment","points":[[120,80]]},{"op":"dilate","radius":4},{"op":"restore","w":0.7},{"op":"upscale"}]}' \
  --in image=photo.ppm --out image=pipeline.ppm --out mask=pipeline_mask.ppm --json
```

### D-Bus
One action, `run`: required string param `stages` (a JSON stage list, see
above), required input `image` (`Media::Image`), two outputs — `image` (the
composited result) and `mask` (`Media::Mask`, the mask actually composited
with, at the *output* resolution).

Stage ops (`allowed()` in `crates/imgpipe/src/lib.rs`): `segment`
(`points`/`boxes`), `dilate`/`erode`/`feather` (`radius`), `invert` (no
params), `restore` (`w`), `upscale` (`tile`). `upscale` changes the image size,
so it must be the last stage — asking for it elsewhere is rejected by position,
not silently reordered.

```bash
BRAIN_SAM2_WEIGHTS=... BRAIN_RESTORE_WEIGHTS=... BRAIN_ESRGAN_WEIGHTS=... \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 3
    python3 examples/imaging/edit_pipeline.py --image photo.ppm --point 614,430 --upscale'
```

Reference client: [`examples/imaging/edit_pipeline.py`](../../../examples/imaging/edit_pipeline.py)
(`--image`, `--point x,y` repeatable, `--box x1,y1,x2,y2`, `--dilate N`,
`--feather N`, `--restore w`, `--upscale`, `--tile N`, `--out dir`).

## Not supported

`training`, `finetune`, `LoRA`, `QLoRA`, `HTTP`. This crate composes other
models' inference; it has nothing of its own to train.

## See also

- Crate: [`crates/imgpipe`](../../../crates/imgpipe)
- Workstream ledger: no `status.md` for this crate — the closest is
  [`docs/imaging/plan.md`](../../imaging/plan.md).
- Stages: [`../restore/readme.md`](../restore/readme.md),
  [`../upscale/readme.md`](../upscale/readme.md)
