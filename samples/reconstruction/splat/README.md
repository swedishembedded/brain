<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: reconstruction/splat

A folder of photographs in, a 3D Gaussian Splatting scene out, through the
public `brain` SDK. No camera poses, no EXIF and no model weights are needed.

```bash
make samples/reconstruction/splat/build
make samples/reconstruction/splat/run ARGS="--photos ~/captures/scene --out out/scene.ply"
brain splat view out/scene.ply
```

## What it demonstrates

* `brain::Reconstruction::builder().photos(..).run()` is the whole
  photogrammetry pipeline in one call, the same one `brain splat train` runs:
  * **Structure from motion** recovers every photograph's camera and
    calibrates its real lens (pinhole, Brown-Conrady or fisheye, chosen by
    the photographs themselves).
  * **Multi-view stereo** measures each photograph's surfaces - range, normal
    and confidence per pixel, through that lens - and fuses them into a
    dense starting scene of thin gaussians lying in those surfaces.
  * **The fit** renders every pixel along its own ray through the lens, so no
    photograph is resampled or undistorted. The stereo's range and normals
    act as priors, and density control grows detail where the error is.
    Alongside the scene it fits the photometric camera: each photograph's
    exposure (starting from its EXIF) and white balance, the lens's
    vignetting, and the sensor's colour matrix and response.
* `save_ply`, `save_cameras`, `save` and `render` cover what an application
  does with the result. The PLY is the standard splat format that viewers
  open, with the anti-aliasing filter the fit used baked in.
* The sample names one SDK surface, `three-d`, and does not link the model
  store resolver or any model architecture.

## What it needs

* **A GPU.** Stereo and the fit run on the device the SDK's `Device` selects.
* **Photographs of one static scene:**
  * Photograph the subject from all around.
  * Overlap neighbouring shots generously, so every part of the scene is in at
    least three photographs.
  * Textured surfaces register and measure well. Plain walls, glass and
    mirrors do not.
  * Keep people and pets out of the frame, and don't move anything between
    shots.

  The sample exits with a message when it gets fewer than two photographs.

## Options

| flag | default |
|---|---|
| `--photos DIR` | required |
| `--out PATH` | `out/reconstruction.ply` |
| `--max-width N` | `2048`: the photographs are halved exactly until they fit |
| `--iterations N` | about 500 visits per photograph, from 3000 to 30000 |
| `--max-gaussians N` | what the dense start needs, plus a quarter |
| `--sparse` | start from structure from motion's points instead of stereo |
| `--no-camera-model` | do not fit the photometric camera (per-photograph exposure and white balance, vignetting, colour matrix, response) |

It writes these files:

* the scene;
* the cameras (`<out>.cameras.json`, which `brain splat render --cameras`
  reads);
* a directory `<out>.bundle/` holding the scene, the cameras and
  `reconstruction.json` (which photograph each camera is, and what structure
  from motion, stereo and the fit measured);
* a render of the first placed camera (`<out>.view0.png`), to compare with
  that photograph.
