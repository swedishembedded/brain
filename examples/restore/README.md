# Face restoration and the VQ latent, over D-Bus

Two models from the CodeFormer stack driven through `com.swedishembedded.Brain1`
with the generic `Run` method - images and code grids travel as file descriptors
(memfd/dmabuf), not as bytes marshalled through D-Bus.

| model | actions | weights env |
|---|---|---|
| `restore` | `restore_face` - a degraded face + `w` → a restored 512² face | `BRAIN_CODEFORMER_WEIGHTS` (`codeformer.pth`, or its directory) |
| `vqgan` | `encode` → codebook indices, `decode` → an image | `BRAIN_VQGAN_WEIGHTS` (a released checkpoint, or its directory) |

```bash
brain caps brain/restore
brain caps brain/vqgan
```

## Run it

```bash
BRAIN_CODEFORMER_WEIGHTS=/path/to/codeformer \
BRAIN_VQGAN_WEIGHTS=/path/to/codeformer/vqgan_code1024.pth \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 3
    python3 examples/restore/restore_face.py --image face.ppm --w 0,0.5,1
    python3 examples/restore/vq_roundtrip.py --image face.ppm --corrupt 20'
```

Or with no bus at all:

```bash
BRAIN_CODEFORMER_WEIGHTS=… brain codeformer restore_face --w 0.5 \
    --in image=face.ppm --out image=restored.ppm --json
BRAIN_VQGAN_WEIGHTS=…  brain vqgan encode --in image=face.ppm --out codes=codes.bin --json
BRAIN_VQGAN_WEIGHTS=…  brain vqgan decode --in codes=codes.bin --out image=recon.ppm
```

> **Point the env var at the file you mean.** `codeformer.pth` and
> `vqgan_code1024.pth` ship in the same release directory and share every VQ
> tensor name, so a directory resolves to `vqgan_code1024.pth` first and the
> server logs `vqgan: <dir> -> <file>` so the choice is never silent. They are
> different weights and produce different codes.

## `restore_face.py` - the fidelity dial

`w` is CodeFormer's identity-fidelity control: **0 = maximum quality** (the code
prediction alone drives the generator), **1 = maximum fidelity** to the input
(the encoder features are injected at full strength).

```
  w = 0.00  512x512   34338.4 ms  mean|out-in| 0.03633
  w = 0.50  512x512     929.7 ms  mean|out-in| 0.02953
  w = 1.00  512x512     945.1 ms  mean|out-in| 0.02754
```

Higher `w` tracks the input more closely - visible in the last column, and the
reason the sweep is worth running on your own photo.

The whole sweep runs on **one** resident instance: `w` lives in a one-element
device buffer read by `scale_add`, so changing it is a buffer write, not a graph
rebuild. Only the first call pays the 377 MB import and the upload - which is
what the 30× drop after the first row shows. (`builds` in `brain.stats()` counts
every model the server has built, so it is only `1` when `restore` is the only
one served; what matters is that the sweep does not *increase* it.)

> Measured on a Tesla P40 with a **debug** build; the first call includes the
> import. Use `make release` for representative latency.

### Scope

The action takes an **aligned** 512² face and returns one - the reference CLI's
`cropped_faces/` → `restored_faces/` step. CodeFormer's alignment template is
facexlib's 512² one, which is *not* `arcface::ARCFACE_DST_112` rescaled, so the
face stack in `examples/vision/` is not chained in automatically: wiring the
wrong template would quietly degrade every restoration. Use
`examples/vision/face_id.py`'s `detect` to locate faces, crop, and feed the crop
here.

## `vq_roundtrip.py` - image → codes → image

`encode` and `decode` are separate actions because the whole point of a discrete
latent is that the **codes travel**. A 512² RGB image is 786 432 bytes; its 16×16
code grid is 256 indices - 1 KiB as `u32`, 320 bytes at 10 bits each:

```
  256 indices, 207 distinct of 1024
  quantisation MSE (mean squared distance to the chosen code): 5.9234
  786432 B of pixels -> 1024 B of u32 codes (768x)
```

The codes come back as a raw `Media::Bytes` blob (`u32` little-endian, meta
`{lh, lw, codebook_size}`) and go straight back into `decode` unchanged.
`--corrupt N` zeroes N of them first, which is the cheapest way to see what one
index is worth. An out-of-range index is a clean error, never the out-of-bounds
gather the underlying `embed` kernel would otherwise do.

Both actions share ONE resident instance: `instance_key` is the square `size`,
not the action name, so a round trip builds the graph once.

## What is NOT here

Neither model batches. Both are **recorded step lists over fixed buffers**
(`CodeFormer::new` / `Vqgan::new` size every buffer from one `[3, H, W]` image),
so there is no N axis to widen at call time - the default serial `run_batch`
stands, with the reason stated in `crates/cli/src/resident_restore.rs`. What does
amortise is residency: a `w` sweep, or an encode/decode pair, costs one build.
