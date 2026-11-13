<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/yolov8

The YOLOv8 detector as a standalone program: detection (optionally telling
which person each person box holds, with SCRFD and ArcFace), training from
scratch, fine-tuning a checkpoint (appending classes, freezing what must not
move), the mAP report - and the Intel NPU route: export to ONNX, INT8
quantization, a no-NPU accuracy simulation, and detection through OpenVINO.

`detect`, `train`, `finetune` and `eval` are the model's served actions, with a
command line built from their manifest, so everything `brain serve` exposes
for the model is here with the same parameters and defaults. The `npu-*`
actions are written in this sample: they need the NPU crate, which depends on
the model crates, and they write files for a runtime brain does not itself
serve through.

## Build

```bash
make samples/models/yolov8/build
Y=target/release/sample-models-yolov8
```

## Weights

Pull the official nano checkpoint into the model store (`Ultralytics/YOLOv8`),
or name any brain-format checkpoint with `--weights <file>`. Identity
verification also needs `glintr100.onnx` and `scrfd_10g_bnkps.onnx` in the
store (the antelopev2 release), or `--components arcface.weights=<file>`.

## Run

```bash
$Y detect --image photo.png --out detections=boxes.json                  # boxes in image pixels
$Y detect --image group.png --identity-ref ref.bin --identity-name alice # who is each person (ref.bin: `arcface embed --out embedding=ref.bin`)
$Y train --data data/detect --save mine.safetensors --seed $RANDOM       # from scratch
$Y finetune --data mine/ --save tuned.safetensors --nc 81 --freeze-backbone --freeze-reg-head --train-classes 80 --wd 0
$Y eval --data data/detect --split all --conf 0.001 --dump-preds preds.jsonl
$Y npu-export --out onnx=yolo.onnx                                       # fp32 ONNX, pure Rust
$Y npu-quantize --calib calib/ --out onnx=yolo.int8.onnx                 # INT8 Q/DQ ONNX, pure Rust
$Y npu-sim --data data/detect                                            # fp32 vs INT8 mAP, no NPU
$Y npu-run --onnx yolo.int8.onnx --image photo.png                       # on the NPU (needs OpenVINO)
```

`$Y <action> --help` lists every flag with its default; `--device cpu|gpu` and
`--backend` place the run.

## What it shows

- A per-model program is the model's served actions plus what is an
  application: `invoke::main_with` runs them all through one provider whose
  manifest is the served one extended with the `npu-*` actions.
- The two-stage identity verification lives in the served `detect`; the face
  models are the model store's, supplied by `yolov8-faces`, so the detector
  crate itself links neither.
