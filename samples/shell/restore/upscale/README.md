# sample: restore/upscale (shell)

Upscale an image 4x with Real-ESRGAN, straight from the CLI - no server, no
D-Bus.

```bash
brain pull schwgHao/RealESRGAN_x4plus   # once
samples/shell/restore/upscale/upscale.sh photo.png photo-4x.jpg
```

## What it demonstrates

Any format brain decodes goes in; the output format follows the extension
you type (`.png`, `.jpg`/`.jpeg`, `.ppm`) - an unsupported extension is an
error rather than a file that lies about itself. `--tile` bounds device
memory: the image is upscaled in tiles of that size rather than whole, so a
large input doesn't need a proportionally large card - lower it if you run
out of memory; it does not change the result.

Generation models cap out well below print resolution, so the usual shape
is: generate at what the model does well, then upscale here (e.g. after
`samples/python/imagegen/flux2-klein/generate.py` or an LTX-2.5 video
frame).

## What it needs

A `RealESRGAN_x4plus.pth` checkpoint in the model store.

## Options

| flag / env | default |
|---|---|
| `<image>` | *required* |
| `[out]` | `<image>-4x.jpg` |
| `TILE` (env) | `128` |
| `DEVICE` (env) | `gpu0`/`gpu1`/`cpu`, passed as `--device` - CPU handles this fine if the cards are busy |
| `BRAIN` (env) | `./target/release/brain` |
