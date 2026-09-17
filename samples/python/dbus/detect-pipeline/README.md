# sample: dbus/detect-pipeline

A full multi-model pipeline over D-Bus: **generate -> detect -> draw**.
`detect_pipeline.py` generates an image with z-image, runs YOLOv8 object
detection on it, and saves the image with labeled boxes drawn over the
detections - every step over `com.swedishembedded.Brain1` (protocol in
`samples/python/dbus/brain-dbus/README.md`), exchanging the image as a file
descriptor. Both models are served and scheduled by the same residency
Executor.

```bash
BRAIN_S3DIT_DIT=... BRAIN_YOLOV8=... \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/dbus/detect-pipeline/detect_pipeline.py
```

## What it demonstrates

* Chaining two unrelated models through the *same* D-Bus surface, in the same
  process, with the image never touching disk between steps: `Subscribe`
  streams the generated image out as a blob, `Run` passes it straight back in
  as a `detect` input blob.
* `brain.stats()` at the end - both jobs went through the one scheduler.
* A clean `skip()` (bats-compatible exit 77) when either `brain/s3dit` or
  `brain/yolov8` isn't served, so this runs safely with partial weights.

## What it needs

- `BRAIN_S3DIT_*` (z-image) and `BRAIN_YOLOV8` (detection weights) exported
  before `brain serve --dbus`. Without both, the script prints which model is
  missing and skips rather than failing.
- `pip install -e brain-py` (jeepney with fd passing) and Pillow (`PIL`, for
  drawing the labeled boxes).

## Options

Environment variables read at import time (no CLI flags):

| var | default | meaning |
|---|---|---|
| `OUT` | `/tmp` | directory for `step1_generated.png` / `step3_boxes.png` |
| `SIZE` | `512` | generated image width/height |
| `STEPS` | `8` | z-image denoise steps |
| `PROMPT` | *(a sample photo prompt)* | text2image prompt |
