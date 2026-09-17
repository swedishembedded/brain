# sample: restore/vq-roundtrip (D-Bus)

`vq_roundtrip.py` drives `brain/vqgan`'s `encode`/`decode` actions over
D-Bus: image in, a discrete code grid out, then straight back to an image.
`encode` and `decode` are separate actions on purpose - the whole point of
a discrete latent is that the **codes travel**: a 512x512 RGB image is
786,432 bytes; its 16x16 code grid is 256 indices, 1 KiB as `u32` (320
bytes at 10 bits each).

```bash
BRAIN_VQGAN_WEIGHTS=/path/to/vqgan_code1024.pth \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/restore/vq-roundtrip/vq_roundtrip.py --image face.ppm --corrupt 20
```

## What it demonstrates

```
256 indices, 207 distinct of 1024
quantisation MSE (mean squared distance to the chosen code): 5.9234
786432 B of pixels -> 1024 B of u32 codes (768x)
```

The codes come back as a raw `Media::Bytes` blob (`u32` little-endian, meta
`{lh, lw, codebook_size}`) and go straight back into `decode` unchanged.
`--corrupt N` zeroes N of them first - the cheapest way to see what one
index is worth; an out-of-range index is a clean error, never the
out-of-bounds gather the underlying `embed` kernel would otherwise do.

Both actions share **one** resident instance: `instance_key` is the square
`size`, not the action name, so a round trip builds the graph once -
`brain.stats()`'s `builds` staying at `1` across encode+decode is the
evidence.

## What it needs

`BRAIN_VQGAN_WEIGHTS` pointing at a released checkpoint (or its directory -
see `restore-face/README.md` for the directory-resolution caveat it shares
with CodeFormer's weights).

## Options

| flag | default |
|---|---|
| `--image PATH` | *required* - binary PPM (P6) to round-trip |
| `--size N` | `512` (square side the graph is built for, multiple of 32) |
| `--corrupt N` | `0` (also decode with N codes replaced by code 0) |
| `--out DIR` | `/tmp` |
