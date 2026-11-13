<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/splat

3D Gaussian Splatting as a standalone program: render a scene from any camera,
fit it to photographs, inspect, re-frame, chain and thin scenes, turn a folder
of photographs into a finished scene, and fly through the result in a window.

The program is splat's served actions (`render`, `fit`, `info`, `orient`,
`merge`, `prune`) with a command line built from their manifest, so everything
`brain serve` exposes for the model is here with the same parameters and
defaults. On top of that it carries what is an application rather than the
served model: `sfm` and `train` (photographs to cameras and a scene) and `view`
(an interactive window).

## Build

```bash
make samples/models/splat/build
```

## Inputs

No weights: a scene is a binary `.ply` in the Inria layout (what `train`,
`fit`, other splat tools and `brain worldmirror2` write), cameras are a JSON
array in the `fit` action's `views` shape (what `sfm`, `train`, `fit
--pose-lr` and `orient` write), photographs are a folder of images.

## Run

Photographs to a scene and its cameras, then look at it:

```bash
splat=target/release/sample-models-splat
$splat train --images photos/ --out scene=scene.ply --out cameras=cameras.json
$splat view scene.ply
```

Inspect a scene and render it, from the default framing, from a pose, or from
one of the cameras it was trained with, next to its photograph:

```bash
$splat info --scene scene.ply
$splat render --scene scene.ply --out image=front.png
$splat render --scene scene.ply --eye 0,-1,-4 --target 0,0,0 --width 1920 --height 1080 --out image=pose.png
$splat render --scene scene.ply --cameras cameras.json --view 3 --out image=view3.png
$splat render --scene scene.ply --depth --out image=depth.png
```

Fit a scene against photographs, with cameras refined, density control and the
3DGS objective:

```bash
$splat fit --scene scene.ply --cameras cameras.json --images photos/ \
  --iters 3000 --densify 100 --max-gaussians 2000000 --sh-degree 3 \
  --loss l1-ssim --pose-lr 0.0001 --out scene=fitted.ply --out cameras=refined.json
```

Re-frame upright, thin duplicates, chain chunks of a long capture:

```bash
$splat orient --scene scene.ply --cameras cameras.json --out scene=up.ply --out cameras=up.json
$splat prune --scene scene.ply --cameras cameras.json --voxel-pixels 2 --out scene=thin.ply
$splat merge --scenes a.ply,b.ply --cameras a.json,b.json --overlap 8 --prune 0.01 \
  --out scene=merged.ply --out cameras=merged.json
```

Structure from motion alone, for inspection or for `fit`:

```bash
$splat sfm --images photos/ --out scene=sparse.ply --out cameras=cameras.json
```

`view` takes the scene first and `render`'s flags that shape a frame (`--width`,
`--height`, `--fov`, `--bg`, `--eye`/`--target`, `--cameras`/`--view`, `--depth`,
...), plus `--frames N` to quit after N frames and `--shots DIR` for screenshots.
WASD move, Space and C go up and down, Shift sprints, `m` captures the mouse
for look (the arrows look too), `[` `]` trade resolution for speed, `v` toggles
colour and depth, `p` saves a screenshot, Enter returns to the start, Esc
quits. Headless, set `SDL_VIDEODRIVER=dummy`.

`sample-models-splat --help` lists the actions, `sample-models-splat <action>
--help` every parameter with its default. `--device gpu0` places the run.

## What it shows

How a program embeds one brain model through its served actions:
`invoke::Model` names the manifest and how the provider is built, and
`invoke::main_with` does the rest - help from the static manifest, input files
to blobs, outputs to files - while an `invoke::Workflow` adds the window, which
has no request and response to serve. `sfm` and `train` use the same action
contract over `recon`, which depends on `splat` and so cannot be part of its
provider.
