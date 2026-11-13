# sample: vision/worldmirror2-reconstruct (D-Bus)

`worldmirror2_reconstruct.py` drives `brain/worldmirror2`'s one-shot
`reconstruct` action over D-Bus: N *unposed* images in, a Gaussian-splat
scene plus the per-frame cameras WorldMirror-2 itself predicted out - this
script sends no poses; the model estimates them.

```bash
BRAIN_WORLDMIRROR2_WEIGHTS=<mirror.safetensors> \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/worldmirror2-reconstruct/worldmirror2_reconstruct.py \
      --images a.ppm,b.ppm,c.ppm --out scene.ply

make samples/models/splat/run ARGS="view scene.ply"
```

## What it demonstrates

`reconstruct` is a plain `Run` - a single feed-forward pass, nothing to
stream. The N input images pack into one `video` blob (the same convention
`splat_fit.py`'s target views use: every frame shares one `(w, h)`; the action
resizes them to the model's 14px patch grid, `--target-size` caps the longest
edge). The flags mirror `brain worldmirror2 reconstruct`'s own parameters
(`--param NAME=VALUE` reaches the rest, such as `position_from=blend`); the
scene comes back as the `scene` PLY and the cameras as the `cameras` JSON
blob. `--maps` additionally fetches the per-frame depth and normal clips,
written out as `..._depth_NN.ppm` and `..._normals_NN.ppm`. Render the result
with `sample-models-splat view scene.ply` or
`samples/python/vision/splat/splat_render.py`.

## What it needs

`BRAIN_WORLDMIRROR2_WEIGHTS` pointing at the checkpoint (the scheduled
service's setting; a direct run resolves it from the model store, see
`sample-models-worldmirror2`).

## Options

| flag | default |
|---|---|
| `--images PATH` | *required* - directory of `*.ppm` (sorted) or a comma-separated list, in order |
| `--out PATH` | `scene.ply` |
| `--cameras PATH` | `<out>_cameras.json` |
| `--min-opacity F` | `0.01` |
| `--max-depth F` | `0.0` (off) |
| `--prune F` | `0.002` (`0` = off) |
| `--target-size N` | `952` |
| `--stride N` | `1` |
| `--max-frames N` | `0` (all) |
| `--keep-camera-frame` | off - re-frame the scene upright |
| `--maps` | off - also fetch the per-frame depth and normal clips |
| `--param NAME=VALUE` | any other `reconstruct` parameter |
