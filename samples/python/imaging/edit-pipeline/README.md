# sample: imaging/edit-pipeline

The whole imaging pipeline in one D-Bus call: **segment -> refine the mask ->
restore -> (optionally) upscale.** `edit_pipeline.py` drives `brain/imgpipe`'s
`run` action as a single request instead of four round trips.

```bash
BRAIN_SAM2_WEIGHTS=... BRAIN_CODEFORMER_WEIGHTS=... BRAIN_ESRGAN_WEIGHTS=... \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/imaging/edit-pipeline/edit_pipeline.py --image photo.ppm --point 614,430 --upscale
```

## What it demonstrates

Why one call rather than four:

* the intermediate mask and image **never cross the bus** - four separate
  calls would marshal a full-resolution image out and back three times;
* the composite happens **once**, at the end, so pixels outside the mask
  come back bit-identical instead of surviving three lossy round trips;
* a stage whose model is not configured fails with **that model's** own
  `set BRAIN_...` message, because the pipeline dispatches through the
  capability registry rather than linking the models.

`upscale` is a **tail**: it changes the image size, so it must be the last
stage and runs after the composite. Asking for it in the middle is rejected
by position rather than silently reordered. The returned mask travels at the
*output* size, so it still describes the image it came back with.

The stages and their parameters are the ones `brain caps brain/imgpipe`
lists - the sample builds the JSON, it does not define it.

## What it needs

- `BRAIN_SAM2_WEIGHTS`, `BRAIN_CODEFORMER_WEIGHTS`, `BRAIN_ESRGAN_WEIGHTS` -
  only the stages you actually request need their weights configured.
- `jeepney` - `pip install -e brain-py`.

## Options

| flag | default |
|---|---|
| `--image PATH` | *required* - binary PPM (P6) |
| `--point x,y` | repeatable - foreground click for segmentation |
| `--box x1,y1,x2,y2` | unset - box prompt for segmentation |
| `--dilate N` | `4` - grow the mask by N px (0 = off) |
| `--feather N` | `3` - soften the mask edge by N px (0 = off) |
| `--restore X` | `0.7` - face-restoration fidelity dial |
| `--upscale` | off - add the x4 super-resolution tail |
| `--tile N` | `0` - upscale tile size (0 = whole image) |
| `--out DIR` | `/tmp` - where to write the result and the mask |
