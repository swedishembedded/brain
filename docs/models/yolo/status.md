# yolo — workstream ledger

From-scratch anchor-free detector (CSP backbone → PAN-FPN neck → decoupled DFL
head), byte-compatible with canonical `yolov8n` for weight import. The
user-facing guide is `readme.md`; the NPU export/quantize/run pipeline is
`npu.md`. This file is the workstream ledger (what landed, the parity gates,
what remains).

## Done

- **Architecture** — one parametric graph, two configs (`config.rs`):
  `YoloConfig::tiny(nc)` (128 px, reg_max 8) and `YoloConfig::yolov8n()` (640 px,
  nc 80, reg_max 16). CSP backbone (Conv-BN-SiLU, Bottleneck, C2f, SPPF) →
  PAN-FPN neck → decoupled DFL head.
  `crates/yolo/src/{model,blocks,head,config,net}.rs`.
- **Byte-compatible weight import** — `full_param_list` matches canonical
  `yolov8n` 1:1; import via `tools/yolo_export/export_yolov8.py`.
- **Loss** — Task-Aligned Assigner + BCE/CIoU/DFL with the verified Ultralytics
  gains (λ_box=7.5, λ_cls=0.5, λ_dfl=1.5); CIoU `α` detached; BN running-stats
  EMA (momentum 0.1).
- **Inference** — `Yolo::detect`: letterbox → eval-BN forward → DFL decode →
  sigmoid+argmax → class-aware NMS (`max_det=300`) → un-letterbox;
  `detect_batch` for batched forward.
- **Eval** — `crates/eval/src/detection.rs`: mAP@0.5, precision/recall (reuses
  `yolo::boxmath::iou`); golden-tested in `crates/eval/tests`.
- **Synthetic data** — `crates/data/src/gen_detect.rs`: five presets with exact
  painted-pixel GT boxes, deterministic per seed.
- **NPU path** — `brain npu {export,quantize,check,run,bench,sim}`: pure-Rust
  export/quantize/sim (no NPU needed) + OpenVINO INT8 PTQ (per-channel weights,
  per-tensor act, symmetric Q/DQ; the head's final 1×1 convs stay fp32). DFL+NMS
  stay on the host.
- **Shared vision blocks** — `crates/vision` (Conv/Bottleneck/C2f/SPPF, kernel
  lookup by name), extracted with `tests/p1_forward_pin.rs` landed first to pin
  the pre-refactor baseline (see `docs/models/depth/status.md` P1).

## Parity ladder

| Gate | What | Result |
|---|---|---|
| P1 forward pin | tiny(2)@128 cls+box logits | 45,696 logits pinned bitwise (FNV-1a) in train AND eval; run-to-run + thread-split independent |
| P3 whole-net gradcheck | directional FD, `eps=5e-4`, `atol=4e-3, rtol=8e-2` | every param passes; 277 tensors checked |
| P4 detection-loss gradcheck | full loss, frozen assigner, `eps=5e-4` | passes `atol=5e-2, rtol=8e-2` |
| P8 name-map / counts | canonical yolov8n layout | 57 full Convs, 297 tensors, 3,167,776 params; names unique |
| PyTorch parity | head logits vs exported weights | `max_abs_err < 1e-3` (env-gated, skips without weights) |
| NPU INT8 parity | fp32 vs INT8 fake-quant head logits | cls/box cosine > 0.95 (no NPU needed) |
| P5 overfit | one-image / tiny-dataset | loss 454→4.3 (one-image), recall 3/3; ~399→~5 (8-img), recall 12/12 |
| P7 capability/negative | localization/classification/bg/shuffled | mAP50 0.50 / IoU 0.54; acc 1.0; bg 0 FPs; shuffled-label 0.00 vs clean 0.75 |

## Serving contract

yolo ships the **full** serving contract — it is the canonical pattern AGENTS.md
points other models at ("the pattern to copy: `resident.rs` (yolo/z-image)"):

- **Capability** — `crates/yolo/src/caps.rs`: `YoloProvider` + `DetectAction`
  (`detect`), static weight-free manifest.
- **Residency** — `YoloResident` in `crates/cli/src/resident.rs`, env-gated on
  `BRAIN_YOLO` (`BRAIN_YOLO_BATCH` sets the forward batch), COCO-80 labels
  attached to detections.
- **Batching** — `YoloInstance::run_batch` chunks invocations to the model's
  batch and runs one `detect_batch` per chunk (a genuine batched forward).
- **D-Bus + example** — reachable over `com.swedishembedded.Brain1` via `Run`;
  `examples/dbus/detect_pipeline.py` (z-image generate → yolo detect → draw
  boxes, all over D-Bus with fd-passed images).

## Remaining

- **PyTorch parity is env-gated, not default CI** — `tests/parity.rs` skips
  without `YOLO_PARITY_WEIGHTS`/`YOLO_PARITY_ACTS` (no torch/GPU in the brain
  box; the 640 px forward on the CPU JIT is slow).
- **Live NPU run/bench need hardware** — `brain npu run`/`bench`/`check` need
  OpenVINO + an Intel NPU; `brain npu sim` (the accuracy gate) does not.
- **`scale` and `multi_object` capability tests are `#[ignore]`d** (heavier; run
  with `--ignored`).
- **`fine-tune --freeze-backbone` is a no-op** — accepted but the model exposes
  no per-parameter freeze API; it fine-tunes the whole network from the
  pretrained init (documented in `yolo_cli.rs`).
- **Calibration representativeness** — the synthetic 128 px `data/detect` set is
  the CI vehicle, not representative for 640 px; `npu.md` recommends ~300
  representative 640 px images for `yolov8n` calibration.

## See also

- `docs/models/yolo/readme.md` — full architecture, kernels, test suite, worked example.
- `docs/models/yolo/npu.md` — NPU export→quantize→compile→run pipeline.
- `docs/models/depth/status.md` — the shared `crates/vision` extraction history.
- `docs/serving-contract.md` — the five obligations yolo satisfies.
