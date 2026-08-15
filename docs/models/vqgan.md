# VQGAN (image ↔ codebook encode/decode)

A VQ autoencoder that converts an image into a grid of discrete codebook
indices (`encode`) and back into an image (`decode`). It's mainly a building
block other imaging models are built on - [restore](restore.md)'s face
restoration uses this same codec - but it's also usable standalone, for
example to compress an image down to a compact set of integer codes, or to
see what the model's prior reconstructs from a code grid.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| CLI (`brain <arch> <action>`)       | [x] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

Model id: `brain/vqgan`. Set `BRAIN_VQGAN_WEIGHTS` to a released checkpoint
(`vqgan_code1024.pth` or `codeformer.pth`), or to a directory holding one -
a directory resolves to `vqgan_code1024.pth` first, falling back to
`codeformer.pth`. Both checkpoints share tensor names but are different
weights and produce different codes; which one loaded is logged.

## Running it

```bash
brain caps brain/vqgan
brain vqgan encode --in image=face.ppm --out codes=codes.bin --json
brain vqgan decode --in codes=codes.bin --out image=recon.ppm
```

Over D-Bus, `encode` and `decode` share one resident instance per square
`size`: `encode` takes an `image` and returns `codes` (little-endian `u32`
indices plus a `{lh, lw, codebook_size}` shape); `decode` takes `codes` and
returns an `image`.

## Options

- `size` (default `512`, must be a multiple of 32) - the square resolution to
  encode at / decode to.

## Hardware and limits

Encode and decode always deal in a single image per call - no batching. No
fine-tuning path is exposed on the CLI, and there is no HTTP endpoint - use
`brain do` or D-Bus. Pick one checkpoint (`vqgan_code1024.pth` or
`codeformer.pth`) and stick with it: codes from one are not compatible with
the other.
