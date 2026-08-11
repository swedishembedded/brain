# YOLOv8-style object detector

An anchor-free, single-stage real-time object detector. brain's implementation
is byte-compatible with the canonical Ultralytics YOLOv8n graph, so the
official pretrained weights import and run unchanged, and the detector can
also be trained or fine-tuned from scratch on your own data. Reach for it
whenever you need bounding-box detection — as a one-shot `detect` call, wired
into an event-driven pipeline, or exported to run on an Intel NPU.

## Support

| Capability | Supported |
|---|---|
| Inference               | [x] |
| Training from scratch   | [x] |
| Fine-tuning             | [x] |
| CLI (`brain do`)        | [x] |
| Intel NPU export        | [x] |
| Batched serving         | [x] |

## Getting the weights

Reference the model as `Ultralytics/YOLOv8` and brain fetches and converts
the official nano (`n`) weights itself on first use — no manual `.pt` export,
no extra setup. The first call downloads `yolov8n.pt` and converts it
in-process; every call after that just loads the cached checkpoint.

Detection has no HTTP route, so it's served over D-Bus:

```bash
brain serve --dbus
```

then call `Ultralytics/YOLOv8`'s `detect` action with an RGB image blob.

`brain do` does not auto-fetch — it only reaches models already registered
locally by name. To use a checkpoint you've already converted (or to import a
YOLOv8-compatible checkpoint of your own), point `BRAIN_YOLO` at its
`.safetensors` file to register it directly, bypassing auto-fetch entirely.
Auto-fetch only pulls the `n` variant; larger YOLOv8 sizes need this manual
path.

## Running it

```bash
# train a detector from scratch
brain yolo train --weights out/yolo.safetensors

# fine-tune an existing checkpoint
brain yolo fine-tune --weights out/yolo.safetensors

# evaluate: reports precision, recall, mAP@0.5
brain yolo eval --weights out/yolo.safetensors --data data/detect

# run detection on an image
brain yolo detect --weights out/yolo.safetensors --image photo.jpg --conf 0.25 --iou 0.45
```

`brain yolo train`'s defaults are 200 steps, batch size 4, learning rate
1e-3, weight decay 1e-2 — override as needed.

For continuous inference, an event-driven controller can map a camera frame
straight to a detection event:

```bash
brain run --yolo out/yolo.safetensors
```

## Options

- `--conf` — confidence threshold below which a candidate box is dropped
  (default `0.25`).
- `--iou` — IoU threshold used by NMS to suppress overlapping boxes of the
  same class; detection keeps at most 300 boxes per image.
- `--weights` — path to a `.safetensors` checkpoint.
- Input images are letterboxed (aspect-preserving resize + grey pad) onto the
  model's square input — 640x640 for the pretrained `Ultralytics/YOLOv8n`
  graph.
- `--data` — a dataset directory of images with normalized center-xywh box
  labels, used for training and evaluation.
- `BRAIN_YOLO` — registers a local checkpoint directly, bypassing auto-fetch.

## Intel NPU export

brain can quantize the detector to INT8 and run it on an Intel NPU (Meteor
Lake and newer) via OpenVINO, through a separate export -> quantize -> run ->
bench pipeline (`brain npu export/quantize/run/bench`, or
`brain yolo detect --device npu` as a shortcut). See
[the NPU export page](npu.md) for the full command reference, and
[the configuration reference](../../using/configuration.md) and the NPU note
in [the hardware page](../../introduction/hardware.md) for setup.

## Hardware and limits

- Runs on CPU and any wgpu-supported GPU; the NPU path is a separate
  compiled-graph pipeline, not a third live backend, and always finishes box
  decoding and NMS on the host CPU.
- Batch size is fixed at model-configuration time — no dynamic per-request
  batching.
- Auto-fetch only provides the `n` (nano) variant; larger YOLOv8 sizes must
  be imported manually via `BRAIN_YOLO`.
- Detection is not exposed over HTTP — only D-Bus, `brain do`, and the `brain
  yolo`/`brain run` CLI paths.
