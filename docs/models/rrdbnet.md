# Upscale (4x super-resolution)

Upscales an image 4x using a Real-ESRGAN-style generator — useful for
restoring detail and sharpness lost to compression, resizing, or a low source
resolution. Feed it any RGB image and get back a larger, sharper version.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| CLI (`brain do`)       | [x] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

Model id: `brain/rrdbnet`. Set `BRAIN_ESRGAN_WEIGHTS` to a released RRDBNet
checkpoint file, e.g. `RealESRGAN_x4plus.pth` (`x2plus` and
`x4plus_anime_6B` checkpoints also work — the scale factor is read from the
checkpoint itself).

## Running it

```bash
brain caps brain/rrdbnet
brain do brain/rrdbnet upscale --tile 0 \
    --in image=photo.ppm --out image=upscaled.ppm --json
```

Over D-Bus, the single action is `upscale`: input `image`, one int param
`tile`, output `image` scaled up by the checkpoint's factor (4x on the
released `x4plus` checkpoint).

## Options

- `tile` (default `0` = whole image at once) — process the image in tiles of
  this many input pixels a side, for images too large to upscale in one pass
  within available VRAM.

## Hardware and limits

Tiling trades a bit of speed for lower peak VRAM on large images. There is no
training or fine-tuning path for this model — the released checkpoint ships
inference-only weights. No batching beyond one image per request, and no HTTP
endpoint — use `brain do` or D-Bus. Often used as the final stage of an
[imgpipe](imgpipe.md) pipeline.
