# SAM 2.1 (segmentation)

Promptable image segmentation: give it an image plus a point, a box, or both,
and it returns a mask for the object you pointed at. Reach for it whenever
you need to cut an object out of a photo — background removal, region
selection for editing, building a mask for another pipeline — without
training a detector for that specific object class.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| CLI (`brain do`)       | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched serving        | [x] |

## Getting the weights

Model id: `brain/sam2` — not auto-fetched. Point `BRAIN_SAM2_WEIGHTS` at a
`sam2.1_hiera_{tiny,large}.pt` release checkpoint (or an equivalent
`.safetensors`); it must exist on disk or the model does not register at
all. `BRAIN_SAM2_VARIANT` picks `tiny` (default) or `large`; a per-request
`variant` param overrides it.

## Running it

No dedicated `brain sam2` verb — use the generic `brain do` / D-Bus action
`segment`. It takes a required `image` input plus prompt params, and returns
a `mask` (sigmoid probability on the source-image grid; threshold at 0.5):

```bash
brain caps brain/sam2

BRAIN_SAM2_WEIGHTS=<ckpt>/sam2.1_hiera_tiny.pt \
  brain do brain/sam2 segment --points "614,430" \
    --in image=photo.ppm --out mask=mask.ppm --json
```

Over D-Bus, the same action takes the same params:

```bash
BRAIN_SAM2_WEIGHTS=<ckpt>/sam2.1_hiera_tiny.pt \
  dbus-run-session -- bash -c 'brain serve --dbus & sleep 3
    python3 examples/vision/segment_image.py --image photo.ppm --point 614,430 --concurrent 4'
```

Reference client: `examples/vision/segment_image.py` — prompts are given in
source-image pixels; `--point x,y` is repeatable, `--box x1,y1,x2,y2` sets a
box prompt, `--concurrent N` submits N prompts at once to exercise the
image-batched path.

## Options

| Param | Effect |
|---|---|
| `variant` | `tiny` (default) or `large` — overrides `BRAIN_SAM2_VARIANT` for this request |
| `points` | `"x,y;x,y;…"` in source-image pixels |
| `labels` | `"1;0;…"` — 1 = foreground, 0 = background (default: all-foreground) |
| `box` | `"x1,y1,x2,y2"` |
| `multimask` | bool, default `true` |

Multiple prompts on the same image are cheap: N prompts on one frame cost one
image-encoder pass plus N decoder passes, so batching prompts (not requests)
is the efficient way to segment many regions of one photo.

## Hardware and limits

Image path only — there is no video mode: the reference implementation's
propagate-through-video (memory-bank) tracking is not implemented, so
frame-to-frame consistency across a video is not available. There's no
training or fine-tune verb, and no HTTP surface — this model is reached
through `brain do` or D-Bus.
