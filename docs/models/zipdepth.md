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
this checkpoint to fetch from) - point it at a released ZipDepth checkpoint
(`.pth` or an equivalent `.safetensors`) yourself. The upstream
[ZipDepth](https://github.com/fabiotosi92/ZipDepth) project ships its
released checkpoints directly in its own GitHub repo, e.g.
`checkpoints/zipdepth_base.pth` (the base/unfold-upsampler variant this
crate's importer expects, verified to load and produce a real depth map -
see `scripts/demo/quickstart.sh`):

- Standalone demo (`brain zipdepth --image` / `--camera`): pass `--weights
  <path>` directly.
- Served (via `brain do` / D-Bus, or the `--infer npu` path): set
  `BRAIN_ZIPDEPTH_WEIGHTS=<path>`.

The checkpoint variant (base vs. the NPU-blend layout) is auto-detected from
the file's own tensor names - you never need to pass `--variant` by hand.

## Running it

```bash
export ZIPDEPTH_PTH=/path/to/zipdepth_base.pth

# Still image (window; Esc quits, [ ] cycle colormaps, v cycles views)
brain zipdepth --image photo.ppm --weights $ZIPDEPTH_PTH

# Headless: writes the composite PPM + a content hash, no display needed
DISPLAY= brain zipdepth --image photo.ppm --weights $ZIPDEPTH_PTH \
    --headless --out out/depth.ppm

# Webcam, realtime (Linux/V4L2, YUYV)
brain zipdepth --camera --weights $ZIPDEPTH_PTH

# Pick the accelerator
brain zipdepth --image photo.ppm --weights $ZIPDEPTH_PTH --device cpu       # all cores
brain zipdepth --image photo.ppm --weights $ZIPDEPTH_PTH --device gpu       # wgpu (default)
brain zipdepth --image photo.ppm --weights $ZIPDEPTH_PTH --device vulkan    # native Vulkan
brain zipdepth --image photo.ppm --infer npu --weights zipdepth_base_npu.pth
                                                       # Intel NPU via ONNX/OpenVINO
```

The CLI reads PPM (P6); convert anything else first, e.g. `convert photo.jpg
photo.ppm`.

Served through the capability system over D-Bus, the model takes one `image`
input and returns a `depth` map via the `infer` action (`brain caps
brain/zipdepth` lists it) - zipdepth has its own dedicated CLI module, so
`brain zipdepth --image ...` above is how the CLI specifically reaches it,
not a second `brain zipdepth infer ...` form.

### Views

`--view MODE`, cycled live with `v`:

| mode | what you see |
|---|---|
| `side` | RGB \| colorized depth, side by side (default) |
| `depth` | full-frame colorized depth |
| `fog` | depth-graded fog composited onto the RGB |
| `blur` | depth-of-field blur (far = blurred) |
| `stereo` | random-dot autostereogram (Magic-Eye; free-view straight on) |
| `stereo-image` | textured autostereogram from the image itself |
| `stereo-dual` | cross-eye left\|right pair |

`[` / `]` cycle colormaps (turbo / gray / grayinv) without re-inference;
`--stripes N` sets the stereogram pattern repeat count.

### Training / fine-tuning

```bash
brain zipdepth train --out out/zipdepth.safetensors --steps 50 --batch 2 --size 64x64
brain zipdepth train --out out/ft.safetensors --weights zipdepth_base.pth   # fine-tune
```

## Options

| Flag / env | Effect |
|---|---|
| `--weights <path>` | ZipDepth checkpoint for the standalone demo |
| `BRAIN_ZIPDEPTH_WEIGHTS` | checkpoint path for the served (`brain do`/D-Bus) path |
| `--device cpu\|gpu\|vulkan` | pick the compute backend for the engine path |
| `--infer engine\|npu` | run brain's own engine (default) or the Intel NPU via OpenVINO |
| `--input N` | model input side (rounded to a multiple of 32); smaller is faster, quadratically |
| `--view MODE` | initial view (see table above); `v` cycles live |
| `--colormap turbo\|gray\|grayinv` | initial depth colormap |
| `--stripes N` | autostereogram pattern repeat count |
| `--camera` / `--device-path <dev>` / `--res WxH` | webcam capture options (Linux/V4L2 only) |
| `--headless --out <file>` | write a composite PPM instead of opening a window |
| `--bench N` | print steady-state per-frame timing over N warm re-inferences |
| `brain zipdepth calib --report --weights <pth> --images <dir>` | per-layer INT8 outlier report, used to pick which layers to keep in FP when quantizing |

## Hardware and limits

Output is unbounded non-negative **inverse depth**, relative only - there is
no metric-scale mode. `--camera` is Linux/V4L2 (YUYV) only; an MJPEG-only
webcam is rejected. Training uses a built-in synthetic generator today, not
real depth datasets. HTTP is not wired up for this model - use `brain do`,
D-Bus, or the standalone `brain zipdepth` demo verb.
