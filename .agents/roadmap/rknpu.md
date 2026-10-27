<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# rknpu - roadmap

Scope: native Rockchip RV1103/RV1106 NPU support (own compiler and `/dev/rknpu` runtime, no vendor SDK), a lean low-RAM build of brain for 32-bit ARM boards, and YOLOv5 detection served on such a board.

The reference board is a Luckfox Pico Plus (RV1103: one Cortex-A7, 64 MB DDR of which
Linux sees 33 MB, 24 MB CMA, uClibc-ng 1.0.31, `rknpu` driver v0.9.2 non-IOMMU). The
target is real-time object detection on that board with brain as the only inference
engine: brain loads a brain checkpoint, compiles it for the NPU itself and serves it.

## Decisions

- **No vendor runtime and no offline vendor conversion in the product path.** brain
  emits the NPU task program from its own graph and submits it through the kernel
  driver's ioctls. The register and container knowledge is ported from open-rknpu
  (MIT, RV1103/RV1106-specific, board-verified), attributed in `/NOTICE.md`. Its
  provenance includes black-box vendor-toolkit comparisons and static analysis by its
  authors; porting it was approved by the project owner on 2026-10-27.
- **The vendor toolkit is a black-box oracle only:** it converts the same ONNX graph
  outside this repository so output tensors and latency can be compared. Nothing
  from it (binary, header, model, capture) enters the repository.
- **First end-to-end model is YOLOv5 with ReLU** (Rockchip's `yolov5s_relu` export of
  the Ultralytics architecture): every hot op is a Conv with a fused ReLU, the
  anchor head needs only a CPU sigmoid, and SiLU (which needs a general LUT plus Mul
  on this NPU) is avoided. Weights are treated as AGPL-3.0 like YOLOv8's.
- **The board runs a sample, not the CLI:** `brain-cli` links every model family
  unconditionally, so the on-board program is the standalone sample
  `samples/npu/rockchip`, which links only the NPU runtime, the model and the
  host-side detection crates it uses. It is one process that owns the whole
  pipeline: camera capture, NPU inference, box overlay, H.264 encoding and an RTSP
  server, so no HTTP stack, D-Bus daemon or second process costs RAM. The refactor
  that serves every lean target is making the crates it links independent of
  `gpu-core` and its backends (detection box math and NMS move out of
  `crates/yolov8` into a host-only crate both detectors share), not feature-gating
  the CLI.
- **The camera, ISP 3A, scaler and encoder are the board's own media libraries**
  (`librockit`, `librkaiq`, `librga` under `/oem/usr/lib`), loaded at run time
  through brain's own FFI declarations. Their headers are Apache-2.0 (rockit's are
  `GPL-2.0 WITH Linux-syscall-note OR Apache-2.0`); no library is redistributed.
  The RTSP/RTP server is brain's own.
- **Toolchain:** `armv7-unknown-linux-uclibceabihf` via `-Zbuild-std` with the
  Luckfox uClibc cross gcc as linker (proven: a Rust binary built this way runs on
  the board). A static `armv7-unknown-linux-musleabihf` build also runs, and is
  usable once nothing needs the board's shared libraries.

## Open

- [ ] M0: oracle baseline. `yolov5s_relu.onnx` converted by the vendor toolkit and run
  on the board with the vendor runtime by our own harness, recording output tensors
  for fixed inputs and the per-inference latency. Lives outside this repository.
- [ ] M1: lean build. Detection box math and NMS in a host-only crate; the board
  sample builds for the uClibc target; a `make` target that builds it and one
  that deploys it to `brain/` in the board user's home directory (SD card, never
  the RAM-backed `/tmp`). Gate: the sample's idle RSS measured on the board, with
  the budget set from the measurement.
- [ ] M2: YOLOv5 in brain, sharing `crates/yolov8`'s blocks, NMS and letterbox: arch
  row, import, CPU fp32 forward parity against an ONNX golden, int8 PTQ.
- [ ] M3: the rknpu crate. ioctl runtime (buffers, submit, sync), compiler for
  Conv+ReLU, Add, channel Concat, nearest 2x Upsample; SPPF's 5x5 MaxPool on the CPU.
  Gate: each layer byte-exact against brain's int8 reference on the board, then the
  whole graph against the M0 oracle.
- [ ] M4: still-image mode. The sample detects on an image file and prints the boxes
  as JSON, the end-to-end check against the M0 oracle and brain's CPU reference.
- [ ] M5: real-time stream. The sample takes the camera (the board's `rkipc` daemon
  stopped), runs detection on every frame it can, draws the boxes and serves H.264
  over RTSP. Measured: end-to-end FPS, NPU latency and total RAM on the board.

## Measured

Nothing yet.
