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

## Done

- [x] M1: lean build. The board sample builds for a Cortex-A7 uClibc target spec
  with NEON (`scripts/targets/armv7-cortexa7-linux-uclibceabihf.json`; Rust's own
  uClibc ARMv7 spec disables NEON), `make samples/npu/rockchip/board/{build,deploy}`
  copies it and the model to `brain/` on the board (never `/tmp`). Box math, NMS and
  the COCO names are host-only in `imaging` (`nms`, `coco`), shared by both detectors.
  The sample links the SDK's `vision` surface (39 brain crates, budget in its
  `Cargo.toml`); its RSS on the board is 3 MB idle and 10 MB streaming.
- [x] M2: YOLOv5 in brain (`crates/yolov5`, arch row `yolov5`, no default ref so
  nothing is fetched). `tools/yolov5_export/export_yolov5.py` converts Rockchip's
  `yolov5s_relu.onnx` (121 tensors, 7,226,029 parameters; batch norm folded; the stem is
  a 2x2/s2 space-to-depth conv, SPPF is six chained 3x3 pools; the sigmoid is inside the
  graph) into a brain `.safetensors` with a `brain.config` header. fp32 forward through
  `gpu_core` for any multiple-of-32 input; anchor decode and NMS on the host (NMS hoisted
  into `imaging::nms`, shared with YOLOv8); `DetectionPipeline` resolves YOLOv5 from the
  checkpoint. Parity against onnxruntime (`crates/yolov5/tests/parity.rs`, goldens from
  `tools/goldens/yolov5_dump_reference.py`), max abs diff of the post-sigmoid heads: noise
  320 3.7e-6, noise 640 4.4e-6, photo 320 9.4e-6, photo 640 1.8e-5 (gate 1e-4).
  Int8 quantization lives in `crates/rknpu` (M3), not here.
- [x] M3: the rknpu crate. `/dev/rknpu` driver (NPU memory, cache sync, linked
  one-submission jobs); dense int8 conv with fused ReLU, any stride, even kernels grown
  to odd, height tiling against the CNA buffer; Add lowered to concat + a 1x1 `[I | I]`
  conv so it runs on the NPU; concat by plane aliasing; max pool and 2x upsample on
  NEON between NPU segments; liveness-based tensor memory reuse; weights streamed
  from the mapped checkpoint layer by layer (never a layer's float copy); NPU memory
  split into ~1 MiB buffers because the board's CMA is fragmented. Gates: every
  YOLOv5-at-320 layer shape and the whole YOLOv5s topology byte-exact against the
  plan's host simulation on the board (`make crates/rknpu/board/check/run`); the int8
  network tracks float onnxruntime on the golden photograph
  (`crates/yolov5/tests/npu_int8.rs`).
- [x] M4: still-image mode. `sample-npu-rockchip --model local/yolov5s-relu --image
  FILE` prints JSON boxes. COCO val2017 #139 on the board: ten detections (tv 0.89,
  person 0.62, chair 0.56, ...) identical to `yolov5::npu::simulate_detect` on the host.
- [x] M5: real-time stream. `--stream`: camera (ISP 3A) -> 320x320 RGA tap ->
  `DetectionPipeline` on the NPU on its own thread -> boxes drawn into NV12 -> H.264
  -> RTSP, verified with ffmpeg from a host (boxes and labels on the decoded frames).

## Open

- [ ] M0: oracle baseline against the vendor toolkit and runtime. Superseded as a
  gate by the host simulation (exact) and the float reference (accuracy); still useful
  as a speed comparison.
- [x] Capture-to-stream latency (target ~20 ms): a complete frame reaches the RTSP
  socket in p50 3.9 ms / p95 4.6 ms. The sensor timestamp is the start of readout;
  the SC3336's only mode reads out over 33 ms (`/proc/rkcif-mipi-lvds`: `readout: 33
  ms`) and the online ISP finishes with it (`/proc/rkisp-vir0`: `delay: 32ms`), so
  start of readout -> sent is 37 ms, of which 33.5 ms no downstream pipeline can
  remove. The sample's latency report splits the two.
- [ ] Detection throughput: 15 inferences/s, 46 ms each, of which the NPU is ~23 ms.
  Writing the 3-channel input costs ~6 ms because each pixel fills a 16-lane atom;
  lowering the 2x2/s2 stem to a CPU space-to-depth on input would quarter that and
  1.2 MB of NPU memory. Max pool on the NPU's PPU would take the last CPU op off the
  path.
- [ ] SiLU (stock YOLOv5/YOLOv8 checkpoints) needs a LUT lowering on the NPU.
- [ ] The model must load before the camera starts: with the camera running, the
  board's CMA is too fragmented for the model's buffers.

## Measured

- 2026-10-27, `make crates/rknpu/board/check/run` on the Luckfox Pico Plus: dense int8
  convolution, every YOLOv5-at-320 layer shape, byte-exact against
  `conv::reference` (0 differing bytes of 2.2M). NPU time, median of 5: stem
  320x320x3->12 k2s2 (as k3) 1.50 ms in 18 tasks; 160x160x12->32 k3 1.36 ms;
  160x160x32->64 k3s2 1.13 ms; 20x20x256->256 k3 1.48 ms; 10x10x1024->512 k1
  0.70 ms; head 40x40x128->255 k1 0.43 ms. Check process VmHWM 4.2 MB.
- 2026-10-27, YOLOv5s topology at 320x320 on the board, byte-exact, 172 tasks: NPU
  23 ms, CPU ops 3 ms (max pool 1.3 with NEON, upsample 0.7, cache 0.3), input write
  6.6 ms, whole run 33 ms; 9.6 MB of NPU memory.
- 2026-10-27, live stream on the Luckfox Pico Plus (800x448 at 30 fps, SC3336): video
  30.0 fps; frame captured -> last RTP packet p50 3.9 ms / p95 4.6 ms; sensor readout
  + ISP 33.5 ms; start of readout -> sent p50 37 ms / p95 38 ms;
  detection 15 inferences/s, 46 ms per inference, boxes drawn ~84 ms after capture;
  VmRSS 10 MB (peak 11.4); CMA 20.2 of 24.5 MB. Still image 640x426: 71 ms; model
  load 4.8 s.
