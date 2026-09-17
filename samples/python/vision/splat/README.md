# sample: vision/splat (D-Bus)

Two Python clients against `brain/splat` (3D Gaussian Splatting) over
D-Bus: `splat_render.py` (one posed view of a scene) and `splat_fit.py`
(optimize a scene against N posed target views). Unlike every other model
in this tree, `brain/splat` needs no checkpoint at all - the scene
(Inria-layout binary PLY) arrives as request bytes, so it's always served.

```bash
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/splat/splat_render.py --scene scene.ply --out render.ppm
```

## `splat_render.py` - one posed view in, one image out

A plain `Run`, not streaming - there's nothing to report progress about
for a single rasterizer pass. With neither `--eye` nor `--target` given,
the server auto-frames the scene from its own bounds (the same default
`brain splat render` uses).

```bash
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/splat/splat_render.py --scene scene.ply \
      --eye 0,0,3 --target 0,0,0 --out render.ppm
```

| flag | default |
|---|---|
| `--scene PATH` | *required* - Inria-layout binary PLY |
| `--out PATH` | `render.ppm` |
| `--width` / `--height` | `960` / `720` |
| `--fov` | `60.0` (vertical FOV, degrees) |
| `--eye` / `--target` | scene auto-framed if either is omitted |
| `--bg` | `0,0,0` |
| `--depth` | off - render expected depth instead of color |

## `splat_fit.py` - N posed views in, an optimized scene out

A `Subscribe`: an initial scene plus a `cameras.json` (the format `brain
mirror infer`/`brain splat fit` write) and one PPM per camera, in order.
The rasterizer backward pass (`crates/splat/src/opt.rs::fit`), driven
remotely and cancellable mid-run like any other long training action - the
per-iteration MSE streams as it improves, exactly like `brain splat fit`'s
own `println!`.

```bash
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/splat/splat_fit.py --scene init.ply \
      --cameras out/mirror/cameras.json --images out/mirror \
      --out fitted.ply --iters 200
```

| flag | default |
|---|---|
| `--scene PATH` | *required* |
| `--cameras PATH` | *required* - `cameras.json` |
| `--images PATH` | *required* - directory of `*.ppm` (sorted) or a comma-separated list, one per camera |
| `--out PATH` | `fitted.ply` |
| `--iters N` | `200` |
| `--lr F` | `5e-3` |
| `--min-scale F` | `1e-4` |

## What it demonstrates

Every posed-view input in this repo (here and in `worldmirror2-reconstruct`)
shares one convention: N images pack into a single `video` blob, every
frame sharing one `(w, h)`. `splat_fit.py::load_video` is the reference
implementation of that packing.

## Not served

The interactive WASD/mouse fly-through (`brain splat view`) is
deliberately NOT served: it's human-in-the-loop with no request/response
shape.
