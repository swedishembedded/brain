<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/worldmirror2

WorldMirror-2 multi-view 3D reconstruction as a standalone program: give it
photographs of a scene, or a video of one, and one feed-forward pass returns a
Gaussian-splat scene and the camera each frame was taken from - no poses, no
calibration, no structure-from-motion step.

The program is WorldMirror-2's served actions (`reconstruct`, `import`) with a
command line built from their manifest, so everything `brain serve` exposes
for the model is here with the same parameters and defaults. On top of that it
carries `export-npu`, which is an application rather than the served model: it
writes ONNX graphs of the model's stages for the Intel NPU.

## Build

```bash
make samples/models/worldmirror2/build
```

## Weights

The checkpoint is found in the model store: a `.safetensors` holding the
released WorldMirror-2 tensors (the reference `model.safetensors`, or the
container `import` writes) is recognised by its content. `--weights PATH`
names one outright. The released weights are under Tencent's community
license, which excludes the EU; treat them as research material.

`import` converts the reference release once into brain's own container, with
the model's configuration embedded and every tensor shape-verified, streaming
one tensor at a time:

```bash
worldmirror2=target/release/sample-models-worldmirror2
$worldmirror2 import --source model.safetensors --dest mirror.safetensors
```

## Run

A folder of photographs, or a video, to a scene and its cameras:

```bash
$worldmirror2 reconstruct --images photos/ --out scene=scene.ply --out cameras=cameras.json
$worldmirror2 reconstruct --images orbit.mp4 --out scene=scene.ply --out cameras=cameras.json
```

`--images` is a directory of frames (PNG, JPEG, PPM, BMP or TIFF, read in name
order, so a capture's own numbering is the order the model sees) or a video
file. Any size and aspect ratio works as long as every frame shares one. Look
at the result with `sample-models-splat view scene.ply`.

A long capture is thinned before the model, because its global attention is
quadratic in frame count: a video is capped at 48 frames spread across the
whole clip unless `--max-frames` says otherwise.

```bash
$worldmirror2 reconstruct --images orbit.mp4 --fps 2 --max-frames 24 --out scene=scene.ply
```

A turntable, where an object rotates in front of a still background, needs the
background masked away (white keeps), one mask for the whole capture:

```bash
$worldmirror2 reconstruct --images turntable.mp4 --mask mask.png --out scene=scene.ply
```

Known cameras, in the shape the `cameras` output has, condition the model and
become the cameras the geometry is built through; one per selected frame:

```bash
$worldmirror2 reconstruct --images photos/ --poses cameras.json --out scene=scene.ply
```

Depth and normal maps, and the frames exactly as the model saw them (what a
later fit must be scored against):

```bash
$worldmirror2 reconstruct --images photos/ --maps --frames-out \
  --out scene=scene.ply --out depth=depth.mp4 --out normals=normals.mp4 --out frames=seen.mp4
```

Every other choice is a parameter with the reference's default: how the
scene is filtered (`--min-opacity`, `--gs-mask-threshold`,
`--edge-depth-threshold`, `--min-support`, `--conf-percentile`), which head
places the gaussians (`--position-from`, `--normals-from`), duplicate fusion
and outlier rejection (`--prune`, `--fuse-depth`, `--max-scale-quantile`),
orientation (`--surface-align`, `--keep-camera-frame`) and the working
resolution (`--target-size`).

Export a stage of the model for OpenVINO on the NPU or CPU, with the weights
external (`model.onnx` + `model.onnx.data`):

```bash
$worldmirror2 export-npu --stage dino --out out/mirror-dino.onnx
$worldmirror2 export-npu --stage trunk --frames 4 --hp 37 --wp 37 --out out/mirror-trunk.onnx
$worldmirror2 export-npu --stage heads --out out/
```

`sample-models-worldmirror2 --help` lists the actions, `sample-models-worldmirror2
<action> --help` every parameter with its default. `--device gpu0` places the
run.

## What it needs

A GPU with room for the 1.26B-parameter fp32 model (about 5 GB) plus the
frames' activations; a shape too large for the device is refused with a
message naming `--target-size` and `--max-frames`. The checkpoint, in the model
store or named with `--weights`.

## What it shows

How a program embeds one brain model through its served actions:
`invoke::Model` names the manifest, the weights resolver and how the provider
is built, and `invoke::main_with` does the rest - help from the static
manifest, a directory or a video to a clip blob, outputs to files - while an
`invoke::Workflow` adds `export-npu`, which has no request and response to
serve.
