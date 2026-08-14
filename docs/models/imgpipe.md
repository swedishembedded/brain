# Imgpipe (composed imaging pipeline)

Chains multiple imaging operations — segment, mask refine, face restore,
upscale — into a single call instead of separate round-trips. Reach for it
when you want to apply several imaging steps to one image in one request
(e.g. "restore and upscale the face at this point") rather than orchestrating
each model call yourself and shuttling intermediate images and masks back and
forth. See [Imaging overview](../inference/imaging.md) for how the individual
stages relate.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| CLI (`brain do`)       | [x] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

Model id: `brain/imgpipe`. It has no weights of its own — each stage resolves
its own model's weights the normal way, and a stage whose model is
unconfigured fails with that model's own error:

- `segment` → `BRAIN_SAM2_WEIGHTS`
- `restore` → `BRAIN_CODEFORMER_WEIGHTS`
- `upscale` → `BRAIN_ESRGAN_WEIGHTS`

## Running it

```bash
brain caps brain/imgpipe
brain do brain/imgpipe run \
  --stages '{"stages":[{"op":"segment","points":[[120,80]]},{"op":"dilate","radius":4},{"op":"restore","w":0.7},{"op":"upscale"}]}' \
  --in image=photo.ppm --out image=pipeline.ppm --out mask=pipeline_mask.ppm --json
```

Over D-Bus, the single action is `run`: a `stages` JSON parameter (as above),
input `image`, and two outputs — the composited `image` and the `mask` that
was actually composited with, at the output resolution.

## Options

Stage ops, applied in order:

- `segment` — `points` and/or `boxes` to select a region
- `dilate` / `erode` / `feather` — `radius`, mask post-processing
- `invert` — no params
- `restore` — `w`, the [restore](restore.md) fidelity dial
- `upscale` — `tile`, see [upscale](upscale.md); since it changes the image
  size, `upscale` must be the last stage in the list

## Hardware and limits

Imgpipe composes other models' inference and holds no weights or training
path of its own. Intermediate results (mask, restored image) never leave the
process, and pixels outside the mask come back bit-identical to the input.
No batching, and no HTTP endpoint — use `brain do` or D-Bus.
