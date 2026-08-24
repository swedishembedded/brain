# 3D: reconstruction and rendering

brain can turn a handful of ordinary photos into a navigable 3D scene, and
render, view, or further optimize a 3D Gaussian Splatting scene once you
have one.

## Capabilities

### Multi-view reconstruction — `mirror`

Feed it several photos of a scene or object from different angles and it
reconstructs a 3D scene as a Gaussian Splatting point cloud, with per-frame
depth, normal, and confidence maps and recovered camera poses — no manual
structure-from-motion pipeline needed. See
[the WorldMirror-2 page](../models/worldmirror2.md).

### Rendering and fitting — `splat`

brain's own 3D Gaussian Splatting renderer and optimizer: load a `.ply`
scene (including one produced by `mirror`) to render still images or fly
through it interactively, or fit a new scene against a set of camera-posed
photos. Runs on any GPU or the CPU. See
[the 3D Gaussian Splatting page](../models/splat.md).

Together they cover the round trip: photos in via `mirror`, an interactive
or rendered scene out via `splat`.
