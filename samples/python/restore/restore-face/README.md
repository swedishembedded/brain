# sample: restore/restore-face (D-Bus)

`restore_face.py` drives `brain/codeformer`'s `restore_face` action over
D-Bus, sweeping `w` - CodeFormer's identity-fidelity dial - against one
aligned face crop. Images travel as file descriptors (sealed memfd), not
bytes marshalled through D-Bus.

```bash
BRAIN_CODEFORMER_WEIGHTS=/path/to/codeformer \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/restore/restore-face/restore_face.py --image face.ppm --w 0,0.5,1
```

## What it demonstrates

`w` is a one-element device buffer read by `scale_add`, **not** a recorded
graph constant: **0 = maximum quality** (the code prediction alone drives
the generator), **1 = maximum fidelity** to the input (encoder features
injected at full strength). A whole sweep therefore runs on **one**
resident instance - one buffer write per value, no graph rebuild:

```
  w = 0.00  512x512   34338.4 ms  mean|out-in| 0.03633
  w = 0.50  512x512     929.7 ms  mean|out-in| 0.02953
  w = 1.00  512x512     945.1 ms  mean|out-in| 0.02754
```

The 30x drop after the first row is the import + upload paid once; higher
`w` tracking the input more closely shows up directly in the last column.
`brain.stats()`'s `builds` count staying at `1` across the sweep is the
same evidence from the scheduler's side, not just the clock.

## What it needs

`BRAIN_CODEFORMER_WEIGHTS` pointing at `codeformer.pth` (or its directory -
`vqgan_code1024.pth` ships alongside it in the same release and shares
every VQ tensor name, so a directory resolves to `vqgan_code1024.pth`
first; the server logs `vqgan: <dir> -> <file>` so the choice is never
silent).

**Scope**: the action takes an **aligned** 512x512 face and returns a
restored one - the reference CLI's `cropped_faces/` -> `restored_faces/`
step. CodeFormer's alignment template is facexlib's 512x512 one, not
`arcface::ARCFACE_DST_112` rescaled, so this is not chained automatically
after face detection: wiring the wrong template would quietly degrade
every restoration. Use `samples/python/vision/face-id/face_id.py`'s
`detect` to locate a face in a full photo, crop it, then feed the crop
here.

## Options

| flag | default |
|---|---|
| `--image PATH` | *required* - binary PPM (P6) of an aligned face |
| `--w LIST` | `0,0.25,0.5,0.75,1` (comma-separated fidelity values to sweep) |
| `--out DIR` | `/tmp` |
