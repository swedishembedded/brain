# 3D: reconstruction and rendering

brain can turn a handful of ordinary photos into a navigable 3D scene, and
render, view, or further optimize a 3D Gaussian Splatting scene once you
have one.

## Capabilities

### Multi-view reconstruction - `brain worldmirror2`

Feed it several photos of a scene or object from different angles and it
reconstructs a 3D scene as a Gaussian Splatting point cloud, with per-frame
depth, normal, and confidence maps and recovered camera poses - no manual
structure-from-motion pipeline needed. One feed-forward pass, no per-scene
optimisation. See [the WorldMirror-2 page](../models/worldmirror2.md).

### Rendering and fitting - `brain splat`

brain's own 3D Gaussian Splatting renderer and optimizer: load a `.ply`
scene (including one produced by `worldmirror2`) to render still images or fly
through it interactively, or fit a scene against a set of camera-posed
photos. Runs on any GPU or the CPU. See
[the 3D Gaussian Splatting page](../models/splat.md).

## The round trip

The two compose into one pipeline, and the second stage is not optional if you
care about the result: `worldmirror2 infer` gets you a scene and the camera
poses for it in one pass, and `splat fit` then optimises that scene against the
same photos, using the poses the first stage recovered.

```bash
brain worldmirror2 infer --weights mirror.safetensors --images photos/ \
    --prune 0.002 --out scene
brain splat fit scene/scene.ply --cameras scene/cameras.json --images photos/ \
    --iters 280 --out fitted.ply
brain splat view fitted.ply
```

`--images` for `fit` must be the SAME frames, at the size the recovered
cameras were written for (`width`/`height` in `cameras.json`) - `fit` asserts
on a mismatch rather than fitting against a rescaled target.
