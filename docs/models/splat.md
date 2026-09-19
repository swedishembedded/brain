# 3D Gaussian Splatting (splat)

brain's own from-scratch 3D Gaussian Splatting renderer and optimizer: load
a `.ply` scene, render still images or fly through it interactively, or fit
a new scene against a set of posed photos. Reach for it to view or inspect
any Gaussian-splat scene - including ones produced by [mirror](worldmirror2.md) -
or to optimize a scene of your own from a set of camera-posed images. The
same rendering kernels run on any GPU and on the CPU, so it works without
a discrete graphics card.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| CLI (`brain <arch> <action>`)       | [ ] (has its own `brain splat …` subcommand instead - see "Running it" below) |
| HTTP API               | [ ] `render`/`fit` both take a REQUIRED `scene` input blob, so neither is a text-to-image/chat action |
| D-Bus                  | [x] `render` (one-shot) and `fit` (streaming); `view` is human-in-the-loop and is not served |
| Batched serving        | [ ] |

## Getting the weights

There's no fetched model here - splat works directly on Inria-format `.ply`
scene files, whether produced by `brain worldmirror2 infer`, brain's own `fit`
(below), or another Gaussian Splatting tool.

## Running it

```bash
brain splat info   scene.ply
brain splat render scene.ply --out img.ppm [--eye x,y,z --target x,y,z]
brain splat view   scene.ply         # interactive fly-through (WASD + mouse)
brain splat fit     scene.ply --images photos/ --cameras cameras.json --out out/fitted.ply
```

`view` opens an interactive window. With no `--eye`, the camera auto-frames
the scene from its bounds. Viewer controls:

| Key | Action |
|---|---|
| W A S D | move |
| Space / C | up / down |
| Shift | sprint |
| `m` | toggle mouse-look |
| Arrow keys | look |
| `[` / `]` | render quality (1x, 1/2, 1/4 resolution) |
| `v` | color / depth view |
| `p` | screenshot |
| Enter | reset camera |
| Esc | quit |

`fit` optimizes an existing `.ply` scene against a set of posed target
photos - the same rasterizer, run backward. `--cameras` takes the
`cameras.json` format `brain worldmirror2 infer` produces (a list of camera poses
and intrinsics); `--images` is a directory of P6 PPM photos (or a
comma-separated list) in the same order as the cameras, and each image's
size must match its camera. This is how you turn a starting point cloud
(from `brain worldmirror2`, or your own SfM/COLMAP output converted to `.ply`)
into a scene that actually reproduces your photos.

## Options

| Flag | Command | Effect |
|---|---|---|
| `--out <path>` | `render`, `fit` | output file (`img.ppm` / `fitted.ply`) |
| `--width` / `--height` | `render`, `view` | output resolution |
| `--eye x,y,z --target x,y,z [--up x,y,z]` | `render`, `view` | explicit camera; omit both for auto-framing |
| `--fov D` | `render`, `view` | vertical field of view in degrees |
| `--depth` | `render` | render the depth view instead of color |
| `--bg r,g,b` | `render`, `view` | background color |
| `--aa` | `render` | opacity compensation for the dilation below |
| `--eps2d X` | `render` | anti-alias dilation in px^2 (default `0.3`). **This is a blur** - see below |
| `--naive` | `render` | use the reference (non-tiled) rasterizer instead of the tiled pipeline |
| `--bench N` | `render` | print steady-state per-frame timing over N warm re-renders |
| `--isect-cap N` | `render` | tile-instance budget (default: 8 per gaussian). A frame that exceeds it drops its depth-latest splats and says `CLAMPED` |
| `--frames N` | `view` | exit the viewer after N frames (scripted/headless runs) |
| `--cameras <path>` | `fit` | camera poses/intrinsics (default `out/mirror/cameras.json`) |
| `--images <dir\|list>` | `fit` | target photos, one per camera, in order |
| `--iters N` | `fit` | optimization steps (default `200`) |
| `--eps2d X` | `fit` | the dilation to optimize UNDER (default `0.3`); render the result with the same value |
| `--lr X` | `fit` | learning rate (default `5e-3`) |

## Sharpness, and the anti-alias dilation

Every splat's screen-space covariance gets a constant added to its diagonal
before rasterizing - `eps2d`, in pixels squared, `0.3` by default, the
reference 3DGS value. It is there because splats smaller than a pixel alias
badly as the camera moves, and it is a low-pass filter: it costs detail, and
on a reconstruction it costs a lot of it.

Measured on a scene built to be a perfect representation of a test image (one
gaussian per pixel, 0.25 px across), rendered from the camera it was built
for, as a fraction of the source image's high-frequency content:

| `--eps2d` | fraction of the source's detail | PSNR |
|---|---|---|
| `0.3` (default) | 0.30 | 21.5 dB |
| `0.1` | 0.80 | 34.7 dB |
| `0` | 0.98 | 44.9 dB |

And on a real six-photograph reconstruction, rendered from a camera the model
recovered:

| `--eps2d` | fraction of the photograph's detail | PSNR |
|---|---|---|
| `0.3` (default) | 0.42 | 21.5 dB |
| `0.15` | 0.54 | 20.9 dB |
| `0.05` | 0.65 | 20.0 dB |
| `0` | 0.73 | 19.2 dB |

Note which way PSNR moves. Blurring a slightly-wrong scene IMPROVES its
mean-squared error while destroying its detail, so accuracy alone will tell
you the default is the best setting and it is not. That is what
`splat::quality::sharpness_ratio` exists to measure, and what the
`s6`/`s7` test gates hold.

### Which value to use

**The dilation is part of the forward model `fit` inverts, not a display
setting.** The optimizer folds compensation for it into the gaussians, so a
fitted scene has to be rendered at the value it was fitted under. `fit` prints
that value when it finishes, and both commands take `--eps2d`.

Measured on the same six-photograph reconstruction after a full-resolution
fit at the 0.3 default:

| rendered at | fraction of the photograph's detail | PSNR |
|---|---|---|
| `0.3` (**what the fit used**) | **0.92** | **29.1 dB** |
| `0.2` | 1.22 | 28.5 dB |
| `0.1` | 1.67 | 26.3 dB |
| `0.05` | 1.90 | 24.4 dB |

Above 1.0 is not extra sharpness, it is aliasing: detail the photograph does
not contain, produced by removing a blur the gaussians were shaped to cancel.
Both measures agree that matching is correct.

So:

* **fitted scene** - render at the `--eps2d` it was fitted under. Changing it
  is a way to make the scene wrong, not a quality dial.
* **raw feed-forward scene, never fitted** - nothing has compensated for
  anything, so lowering it does recover real detail (0.42 to 0.73 in the table
  above). `--eps2d 0.05` is a reasonable still-frame setting there.
* **moving camera** - keep the default. The aliasing it prevents is worse than
  the sharpness it costs.

## When a render says CLAMPED

The tiled rasterizer sorts one instance per (gaussian, tile) pair out of a
buffer sized 8 instances per gaussian. Big screen-space gaussians touch dozens
of tiles each, so a scene whose gaussians have GROWN - which is every scene
`fit` has optimized - can exceed that budget. The overflowing tail is dropped
rather than the frame failing, and because the tail is depth-latest, what you
see is a missing occluder: a bright flare where something should have been in
front of something else.

`--isect-cap N` raises the budget; the render line says how many instances the
frame actually wanted, so `--isect-cap` that number and re-render.

## Serving (D-Bus)

Model id: `brain/splat`. Always served - the scene arrives as request bytes,
so there is no checkpoint to configure.

```bash
brain caps brain/splat
```

`render` (one-shot): takes `scene` (Inria-layout binary PLY) plus the same
camera/size/depth/background params as `brain splat render` above, returns
`image`. `fit` (streaming): takes `scene` plus `video` (N target views,
concatenated interleaved-HWC f32 RGB frames - the same convention every other
video input uses) and a `views` param (the camera array as JSON, the shape
`cameras.json` uses), returns the optimized `scene` and reports the
per-iteration MSE as progress; cancellable mid-run like any other long-running
served action. `view` is deliberately not served - it is an interactive
WASD/mouse loop with no request/response shape. See
[`samples/python/vision/splat/splat_render.py`](../../samples/python/vision/splat/splat_render.py) and
[`splat_fit.py`](../../samples/python/vision/splat/splat_fit.py).

## Hardware and limits

Runs on any wgpu-supported GPU or on the CPU - no CUDA or vendor-specific
GPU is required. Only spherical-harmonics degree 0 (flat per-splat color)
actually renders today; higher-order SH coefficients in a `.ply` are parsed
and preserved on round-trip (so re-saving a scene doesn't lose them) but
don't yet affect the rendered image. `fit` optimizes an existing set of
gaussians against posed photos - it does not run structure-from-motion or
recover camera poses itself; bring your own `cameras.json`.
