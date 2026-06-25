# brain-py

A small Python package that drives the **`brain`** executable as an
event-driven subprocess: send an image in, get the detected boxes (and an
annotated image) back. It speaks `brain`'s JSONL-over-stdio event protocol and
correlates concurrent requests by `req_id`.

## Install

```bash
# editable install from the repo root (pulls in Pillow)
pip install -e brain-py
# or just the runtime dep if you only need the module on PYTHONPATH
pip install pillow
```

You also need the `brain` binary. Build it once:

```bash
cargo build --release       # produces ./target/release/brain
```

`BrainClient` locates the binary via (in order): a `brain_bin=` argument, the
`$BRAIN_BIN` env var, `./target/release/brain`, `./target/debug/brain`, then
`brain` on `$PATH`.

## Quick start (runs out of the box)

```bash
# Generate a synthetic image, run the built-in FAKE detector, save annotated.png:
python -m brain_py.examples.detect_image --out annotated.png

# With a real tiny YOLO checkpoint trained on `brain data gen detect`:
python -m brain_py.examples.detect_image \
    --weights /tmp/bp.weights --out annotated.png --conf 0.05
```

Library usage:

```python
from PIL import Image
from brain_py import BrainClient, annotate

img = Image.open("photo.png")
with BrainClient(yolo="/tmp/bp.weights", conf=0.05) as client:
    dets = client.detect(img)          # -> list[Detection], in img pixel coords
annotate(img, dets).save("annotated.png")
```

## The event protocol

`brain run [--yolo W] [--gpt W] [--conf X]` reads ONE JSON event per line on
stdin and writes events (one per line) on stdout; stderr is logs. Each request
may carry a top-level `"req_id"` string, which the runtime **echoes on every
response event** for that request — this is how concurrent requests over the
single stdio stream are demultiplexed.

| Direction | Event |
|-----------|-------|
| request   | `{"req_id":"r1","event":"camera_frame","format":"rgb8","w":W,"h":H,"data":"<base64 of W*H*3 RGB8 bytes>"}` (or `"path":"file"`) |
| request   | `{"req_id":"r1","event":"user_text","text":"..."}` |
| response  | `{"req_id":"r1","event":"object_detected","dets":[[x1,y1,x2,y2,conf,class],...],"labels":[...]}` |
| response  | stream of `{"req_id":"r1","event":"brain_text_chunk","text":"..","seq":N,"done":false}` ending with `done:true` |
| lifecycle | `{"event":"ready"}` (no `req_id`), `{"event":"error","message":...}` |

### BrainClient design: reader thread + condition-guarded queue + req_id demux

* `BrainClient` spawns `brain run ...` with `subprocess.Popen` in text mode,
  line-buffered, with `BRAIN_DEVICE=cpu` in the environment.
* A **background reader thread** blocks on `stdout.readline`, parses each line
  as JSON, and routes it **by `req_id`** into a per-request buffer.
* All shared state lives behind a single `threading.Condition` — the
  *condition-guarded queue*. The reader appends events and `notify_all()`s;
  `detect()` / `chat()` `wait_for()` on the condition until their own `req_id`
  is complete (with a timeout).
* One-shot responses (`object_detected`) complete the request immediately;
  streaming responses accumulate `brain_text_chunk`s until `done:true`.
* This decouples brain's emission order from caller order, so multiple `detect`
  / `chat` calls can be in flight and are demuxed by `req_id`.
* `close()` (or the context manager) closes stdin (EOF ends brain's loop),
  joins the reader thread, and terminates the process.

The `--conf` flag is a CLI-only addition to `brain run` (threaded through to the
`YoloDetect` adapter): a tiny model trained for only a few hundred steps emits
low-confidence boxes that the default `0.25` gate would drop, so the demo lowers
it. It has no effect on the fake detector. (`$BRAIN_CONF` works too.)

## What should the input image contain?

**(a) The tiny demo model** — trained on the synthetic `brain data gen detect`
data: an image of **solid-colored shapes (rectangles / circles) on a dark-grey
background**, where the colors match the `gen_detect` class palette
(`BG = (32,32,32)`; class 0 = red `(220,40,40)`, 1 = green `(40,200,40)`,
2 = blue `(60,80,230)`, ...). The model is trained at 128×128 (MultiObject
preset, 3 classes). `brain_py/examples/make_test_image.py` generates exactly
such an in-distribution image, and the example uses it when `--image` is
omitted. A real photo will be out-of-distribution for this tiny model.

**(b) Real `yolov8n` weights** exported via `tools/yolo_export` (COCO, 80
classes): an **ordinary photo containing common objects** (person, car, dog,
bicycle, ...) at any resolution — it gets letterboxed to 640×640 internally.
This path is slow on the CPU JIT and requires the offline weight export first.

## Tests

```bash
pip install pytest
python -m pytest brain-py/tests        # CI-fast: uses brain's built-in fake detector
```

The CI tests run `brain run` with **no `--yolo`**, so brain uses its built-in
fake detector (a fixed deterministic box) and fake echo text model — fast, no
model load, no JIT. They exercise the real subprocess, the JSONL protocol, the
reader thread, and the `req_id` demux. The slow real-model demo is
`brain_py/examples/detect_image.py`, run manually.
```
