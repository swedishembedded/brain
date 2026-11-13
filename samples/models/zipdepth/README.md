<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/zipdepth

ZipDepth as a standalone program: depth from one image, a view of it (the
photo beside its colorized depth, fog, depth of field or a stereogram),
training and the INT8 calibration report - and two windows, `show` for one
image and `camera` for a live webcam, that cycle through the views as you
watch, on brain's engine or the Intel NPU.

Every action is ZipDepth's served one, with a command line built from its
manifest, so everything `brain serve` exposes for the model is here with the
same parameters and defaults. The windows are written in this sample: they
drive the model directly so a live camera keeps the engine busy on the next
frame while the last one is drawn.

## Build

```bash
make samples/models/zipdepth/build
Z=target/release/sample-models-zipdepth
```

## Weights

The released checkpoint lives in the upstream repository; put it in the
model store (or name any file with `--weights <path>`):

```bash
mkdir -p ~/.local/share/brain/models/brain/zipdepth
curl -L -o ~/.local/share/brain/models/brain/zipdepth/zipdepth_base.pth \
  https://github.com/fabiotosi92/ZipDepth/raw/main/checkpoints/zipdepth_base.pth
```

## Run

```bash
$Z infer --image photo.png --out depth=depth.png                       # the depth map
$Z infer --image photo.png --view side --out view=side.png             # photo | depth
$Z show --image photo.png                                              # a window: v view, [ ] colormap, Esc quit
$Z camera                                                              # the webcam, live
$Z camera --npu --weights zipdepth_npu.pth                             # on the Intel NPU
$Z train --save mine.safetensors --steps 200 --seed $RANDOM            # train (fine-tunes the store's checkpoint)
$Z calib --images calib/                                               # which layers resist INT8
```

`$Z <action> --help` and `$Z show --help` list every flag with its default;
`--device cpu|gpu` and `--backend` place the run.

## What it shows

- A per-model program is the model's served actions plus what is an
  application: `invoke::main_with` runs the actions and hands the windows
  (`invoke::Workflow`) the resolved checkpoint.
- The window draws with `zipdepth::view::render`, the renderer the served
  `infer` uses for its `view` output, so a saved view matches the window.
