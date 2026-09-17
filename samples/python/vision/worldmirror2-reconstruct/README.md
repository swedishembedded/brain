# sample: vision/worldmirror2-reconstruct (D-Bus)

`worldmirror2_reconstruct.py` drives `brain/worldmirror2`'s one-shot
`reconstruct` action over D-Bus: N *unposed* images in, a Gaussian-splat
scene plus the per-frame cameras WorldMirror-2 itself predicted out - this
model estimates poses, it does not take them as input.

```bash
BRAIN_WORLDMIRROR2_WEIGHTS=<mirror.safetensors> \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/worldmirror2-reconstruct/worldmirror2_reconstruct.py \
      --images a.ppm,b.ppm,c.ppm --out scene.ply

brain splat view scene.ply
```

## What it demonstrates

`reconstruct` is a plain `Run` - a single feed-forward pass, nothing to
stream. The N input images pack into one `video` blob (the same convention
`splat_fit.py`'s target views use: every frame shares one `(w, h)`, which
must also land on a whole number of the model's 14px patches).
`min_opacity`/`max_depth`/`prune_voxel` mirror `brain worldmirror2 infer`'s
own flags exactly; `--maps` additionally fetches a per-frame depth-map
video, written out as `..._depth_NN.ppm`. Render the result with
`brain splat view scene.ply` or `samples/python/vision/splat/splat_render.py`.

## What it needs

`BRAIN_WORLDMIRROR2_WEIGHTS` pointing at the checkpoint.

## Options

| flag | default |
|---|---|
| `--images PATH` | *required* - directory of `*.ppm` (sorted) or a comma-separated list, in order |
| `--out PATH` | `scene.ply` |
| `--cameras PATH` | `<out>_cameras.json` |
| `--min-opacity F` | `0.01` |
| `--max-depth F` | `0.0` (off) |
| `--prune-voxel F` | `0.0` (off) |
| `--maps` | off - also fetch a per-frame depth-map video |
