# sample: vision/mirage-perception (D-Bus)

`mirage_perception.py` drives `Cactus-Compute/mirage`: `describe` (scene,
objects with boxes, scores and masks, text lines, relations, reasoning) and
`extract` (a form of choice, count and text fields) over D-Bus with
`brain_py`.

```bash
brain pull Cactus-Compute/mirage
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/mirage-perception/mirage_perception.py --image photo.ppm --concurrent 4
```

## What it demonstrates

* **Two typed actions on one resident model.** `BrainDBus.describe` and
  `.extract` build the generic `Run` call (the image travels as an fd blob) and
  return a `Perception`: the engine-shaped JSON as a `dict`, with the cost of
  the request in `.usage`.
* **Batching.** With `--concurrent N` the same request is sent from N
  connections at once. The scheduler merges them into one device batch
  (`usage.batch` shows its size) and the sample checks every answer equals the
  single-request answer. On the CPU backend the answers are bit-identical; on
  an accelerator the GEMM selector may pick another kernel for another batch
  size, so scores and mask pixels at the threshold may differ in the last place
  and the sample compares everything else (boxes, names, text, relations).

## What it needs

The released checkpoint (about 240 MB), pulled into the model store. Input is a
binary PPM (P6). Without the model served the sample exits 77 (skip).

## Options

| flag | default |
|---|---|
| `--image PATH` | *required* |
| `--form JSON` | a two-field animal/count form |
| `--max-tokens N` | model default |
| `--concurrent N` | `1` |
