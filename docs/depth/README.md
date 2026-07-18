# Monocular depth in brain — ZipDepth

brain trains, quantizes and runs **ZipDepth** (6.1M-param pure-conv monocular
depth, MIT) end to end: load the released pretrained checkpoints, run realtime
depth from an image or webcam on **CPU / GPU (wgpu, Vulkan) / Intel NPU**, and
fine-tune with brain's own training loop. The model matches the reference
PyTorch implementation exactly (cosine 1.000000 on real checkpoints), and the
whole backward is gradient-checked (`tests/p3_gradcheck.rs`).

`docs/depth/STATUS.md` is the workstream ledger (what landed, in what order,
and every trap the gates caught). This file is the user-facing guide.

## Quick start

```bash
make release
export ZIPDEPTH_PTH=/path/to/zipdepth_base.pth      # released checkpoint

# Still image (window; Esc quits, [ ] cycle colormaps, v cycles views)
./target/release/brain depth --image photo.ppm --weights $ZIPDEPTH_PTH

# Headless: writes the composite PPM + a content hash
DISPLAY= ./target/release/brain depth --image photo.ppm --weights $ZIPDEPTH_PTH \
    --headless --out out/depth.ppm

# Webcam, realtime (Linux/V4L2, YUYV)
./target/release/brain depth --camera --weights $ZIPDEPTH_PTH

# Pick the accelerator
brain depth --image … --device cpu            # WGSL->Cranelift JIT, all cores
brain depth --image … --device gpu            # wgpu (default)
brain depth --image … --device vulkan         # native Vulkan (ash)
brain depth --image … --infer npu --weights zipdepth_base_npu.pth --variant npu
                                              # Intel NPU via ONNX/OpenVINO

# Steady-state timing (skips one-time model build + BN packing)
brain depth --image … --headless --bench 20
```

The CLI reads PPM (P6). Convert anything else with ImageMagick:
`convert photo.jpg photo.ppm`. The checkpoint variant is auto-detected from the
file's own tensor names (`--variant` exists but is never needed).

## Views

`--view MODE`, cycled live with `v`:

| mode | what you see |
|---|---|
| `side` | RGB \| colorized depth, side by side (default) |
| `depth` | full-frame colorized depth |
| `fog` | depth-graded fog composited onto the RGB |
| `blur` | depth-of-field blur (far = blurred) |
| `stereo` | random-dot autostereogram (Magic-Eye; free-view straight on) |
| `stereo-image` | textured autostereogram from the image itself |
| `stereo-dual` | cross-eye L\|R DIBR pair |

`[` / `]` cycle colormaps (turbo / gray / grayinv) without re-inference;
`--stripes N` sets stereogram pattern repeats.

## How inference is structured

- `depth::Predictor` reproduces the reference preprocessing exactly:
  aspect-preserving resize so the shorter side is 384 (both dims rounded to a
  multiple of 32) — NOT a letterboxed square — model forward at that size, then
  bilinear resize of the depth back onto the frame's own grid. The model is
  rebuilt only when the target size changes, so a fixed-resolution camera
  stream builds once.
- Output is unbounded non-negative **inverse depth**, relative only. The
  colorizer uses robust p2/p98 bounds (EMA-smoothed in camera mode) so a lone
  specular spike cannot swing the hue.
- In eval mode every dense conv+BN unit runs as ONE fused register-tiled
  dispatch (`conv_act_reg`, act selector 0-3), not conv2d + bn_eval + act —
  see `docs/PERFORMANCE.md` for what this bought on each backend.

## NPU (Intel AI Boost)

`--infer npu` walks the model graph, emits fp32 ONNX
(`npu::depth_topology::build_depth_graph`), fuses each RepVGG block to one
biased 3×3 (`depth::fuse`), folds BN into every conv, compiles via OpenVINO and
runs on `/dev/accel/accel0`. Parity: OpenVINO-CPU is graph-EXACT vs brain
(cosine 1.00000); the NPU itself is cosine 0.99998 (fp16 internals), ~30 fps.
Needs the `zipdepth_base_npu.pth` variant (its upsampler is NPU-node-only).
`brain depth calib --report` prints the per-conv INT8 outlier report used to
pick the (encoder-side) layers to keep in FP when quantizing.

## Training / fine-tuning

`brain depth train` runs brain's own loop on RGB→inverse-depth pairs:
forward → masked L1 → backward → AdamW, gradient-faithful to the master
gradcheck. See `brain depth train --help` for the dataset layout; the
overfit-one-batch sanity path is `tests/` — loss must strictly decrease.

## Makefile targets

```bash
make depth/demo    ZIPDEPTH_PTH=…   # windowed still-image demo
make depth/smoke   ZIPDEPTH_PTH=…   # headless deterministic render + hash
make depth/camera  ZIPDEPTH_PTH=…   # webcam window
```
