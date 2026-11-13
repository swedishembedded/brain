# Depth (ZipDepth)

Monocular relative-depth estimation: point it at a single image or a live
camera feed and it produces a per-pixel depth map, no stereo rig or second
camera needed. It's a small, fast, pure-convolutional network, so it's the
model to reach for when you want depth in realtime - live webcam preview,
depth-of-field or fog effects on a photo, or even an autostereogram - rather
than a slow, large-scale depth transformer.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| CLI (`brain <arch> <action>`)       | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

Model id: `brain/zipdepth`. Not auto-fetched (no HuggingFace host exists for
this checkpoint). The upstream [ZipDepth](https://github.com/fabiotosi92/ZipDepth)
project ships its released checkpoints in its own repository, e.g.
`checkpoints/zipdepth_base.pth` (the base, unfold-upsampler variant). Put it
in the model store, where every command finds it:

```bash
mkdir -p ~/.local/share/brain/models/brain/zipdepth
curl -L -o ~/.local/share/brain/models/brain/zipdepth/zipdepth_base.pth \
  https://github.com/fabiotosi92/ZipDepth/raw/main/checkpoints/zipdepth_base.pth
```

or name a file outright with `--weights <path>`. A checkpoint `train` writes
(safetensors) is found and loaded the same way. The variant (base, or the
NPU blend layout) is read from the file's own tensor shapes.

## Running it

`brain zipdepth <action>` runs the served actions; `sample-models-zipdepth`
(`make samples/models/zipdepth/build`) runs the same actions plus the image
and webcam windows. `<action> --help` lists every parameter with its default.

```bash
# The depth map, and a view of it (no display needed)
brain zipdepth infer --image photo.png --out depth=depth.png
brain zipdepth infer --image photo.png --view side --out view=side.png

# A window over one image or a live webcam (Linux V4L2, YUYV):
# v cycles the view, [ ] the colormap, Esc quits
Z=target/release/sample-models-zipdepth
$Z show --image photo.png
$Z camera --res 640x480

# Pick the accelerator: --device cpu|gpu, --backend wgpu|vulkan|cuda, or the
# Intel NPU (the blend checkpoint, compiled through OpenVINO) in the windows
$Z show --image photo.png --device cpu
$Z camera --npu --weights zipdepth_npu.pth
```

### Views

`--view` picks what `infer` draws into its `view` output, and what a window
starts on (`v` cycles them):

| view | what you see |
|---|---|
| `side` | the photo \| colorized depth, side by side (a window's default) |
| `depth` | colorized depth |
| `fog` | depth-graded fog composited onto the photo |
| `blur` | depth-of-field blur (far = blurred) |
| `stereo` | random-dot autostereogram (free-view straight on) |
| `stereo-image` | autostereogram patterned with the photo itself |
| `stereo-dual` | cross-eye left \| right pair |

`--colormap turbo|gray|grayinv` colors the depth; `--stripes N` sets how many
times a stereogram's pattern repeats.

### Training / fine-tuning

```bash
brain zipdepth train --save zipdepth.safetensors --steps 50 --batch 2 --width 64 --height 64 --seed $RANDOM
brain zipdepth train --save ft.safetensors --weights zipdepth_base.pth --seed $RANDOM   # fine-tune
```

`train` starts from the checkpoint in the store when there is one (or the one
`--weights` names) and from freshly initialized weights otherwise, or when
`--fresh true` asks. `--activations keep|recompute` sets what the backward
keeps of the forward: every normed unit's intermediates, or only each unit's
input, rebuilt in the backward for the same gradients at lower memory.

### INT8 calibration report

```bash
brain zipdepth calib --images calib/
```

Per-layer activation outlier ratios (absmax / p99.99), the most INT8-hostile
first, and the encoder's mean against the decoder's: the data for choosing
which layers keep full precision when quantizing.

## Options

| Flag | Action | Effect |
|---|---|---|
| `--weights <path>` | every action, the windows | the checkpoint, instead of the store's |
| `--input N` | `infer`, `calib`, the windows | model input side (rounded to 32); smaller is faster, quadratically |
| `--view`, `--colormap`, `--stripes` | `infer`, the windows | what is drawn (above) |
| `--bench N` | `infer` | steady-state per-frame timing over N warm re-inferences |
| `--steps`, `--batch`, `--width`, `--height`, `--lr`, `--wd`, `--seed`, `--fresh`, `--activations` | `train` | the training run |
| `--max N`, `--top N` | `calib` | images read, layers listed |
| `--npu` | the windows | run on the Intel NPU (the blend checkpoint) |
| `--device-path <dev>`, `--res WxH` | `camera` | the webcam and the size asked of it |
| `--scale N`, `--frames N` | the windows | window pixel scale; close after N frames |

Served (`brain serve`, D-Bus), the same three actions are offered; a serving
process loads a carded checkpoint found in its models directory, or
`BRAIN_ZIPDEPTH_WEIGHTS`, and places `infer` on an NPU when one is budgeted.

## Hardware and limits

Output is unbounded non-negative **inverse depth**, relative only - there is
no metric-scale mode. `camera` is Linux/V4L2 (YUYV) only; an MJPEG-only
webcam is rejected. Training uses a built-in synthetic generator today, not
real depth datasets. HTTP is not wired up for this model - use
`brain zipdepth <action>`, D-Bus, or the sample.
