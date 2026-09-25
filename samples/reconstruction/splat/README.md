<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: reconstruction/splat

A folder of photographs in, a 3D Gaussian Splatting scene out, through the
public `brain` SDK. No camera poses, no EXIF and no model weights are needed.

```bash
make samples/reconstruction/splat/build
make samples/reconstruction/splat/run ARGS="--photos ~/captures/can --out out/can.ply"
brain splat view out/can.ply
```

## What it demonstrates

* `brain::Reconstruction::builder().photos(..).run()` is the whole
  photogrammetry pipeline in one call. Structure from motion recovers every
  photograph's camera and a sparse point cloud; the photographs are resampled
  to ideal pinhole cameras; the fit grows the points into the scene.
* `save_ply`, `save_cameras` and `render` cover what an application does with
  the result. The PLY is the standard splat format that viewers open, fitted
  under the same 0.3 px dilation those viewers render.
* The sample names one SDK surface, `three-d`, and links 31 brain crates. It
  does not link the model store resolver or any model architecture.

## What it needs

* **A GPU.** The fit runs on the device the SDK's `Device` selects.
* **Photographs of one static scene**, all from one camera at one zoom:
  * Photograph the subject from all around.
  * Overlap neighbouring shots generously, so every part of the scene is in at
    least three photographs.
  * A textured floor or table under the subject helps registration a great
    deal.
  * Keep people and pets out of the frame, and don't move anything between
    shots.

  The sample exits with a message when it gets fewer than two photographs.

## Options

| flag | default |
|---|---|
| `--photos DIR` | required |
| `--out PATH` | `out/reconstruction.ply` |
| `--width N` | `1024` (training resolution across) |
| `--iterations N` | `3000` |
| `--max-gaussians N` | `500000` |

It writes the scene, the cameras (`<out>.cameras.json`, which
`brain splat render --cameras` reads) and a render of the first placed
camera (`<out>.view0.png`) to compare with that photograph.
