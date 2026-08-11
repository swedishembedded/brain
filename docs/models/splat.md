# 3D Gaussian Splatting (splat)

brain's own from-scratch 3D Gaussian Splatting renderer and optimizer: load
a `.ply` scene, render still images or fly through it interactively, or fit
a new scene against a set of posed photos. Reach for it to view or inspect
any Gaussian-splat scene — including ones produced by [mirror](mirror.md) —
or to optimize a scene of your own from a set of camera-posed images. The
same rendering kernels run on any GPU and on the CPU, so it works without
a discrete graphics card.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| CLI (`brain do`)       | [ ] |
| HTTP API               | [ ] |
| D-Bus                  | [ ] |
| Batched serving        | [ ] |

## Getting the weights

There's no fetched model here — splat works directly on Inria-format `.ply`
scene files, whether produced by `brain mirror infer`, brain's own `fit`
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
photos — the same rasterizer, run backward. `--cameras` takes the
`cameras.json` format `brain mirror infer` produces (a list of camera poses
and intrinsics); `--images` is a directory of P6 PPM photos (or a
comma-separated list) in the same order as the cameras, and each image's
size must match its camera. This is how you turn a starting point cloud
(from `brain mirror`, or your own SfM/COLMAP output converted to `.ply`)
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
| `--aa` | `render` | enable anti-aliasing |
| `--naive` | `render` | use the reference (non-tiled) rasterizer instead of the tiled pipeline |
| `--bench N` | `render` | print steady-state per-frame timing over N warm re-renders |
| `--frames N` | `view` | exit the viewer after N frames (scripted/headless runs) |
| `--cameras <path>` | `fit` | camera poses/intrinsics (default `out/mirror/cameras.json`) |
| `--images <dir\|list>` | `fit` | target photos, one per camera, in order |
| `--iters N` | `fit` | optimization steps (default `200`) |
| `--lr X` | `fit` | learning rate (default `5e-3`) |

## Hardware and limits

Runs on any wgpu-supported GPU or on the CPU — no CUDA or vendor-specific
GPU is required. Only spherical-harmonics degree 0 (flat per-splat color)
actually renders today; higher-order SH coefficients in a `.ply` are parsed
and preserved on round-trip (so re-saving a scene doesn't lose them) but
don't yet affect the rendered image. `fit` optimizes an existing set of
gaussians against posed photos — it does not run structure-from-motion or
recover camera poses itself; bring your own `cameras.json`.
