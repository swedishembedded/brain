# Running brain's YOLO detector on the Intel NPU (OpenVINO)

brain can **quantize the YOLO detector to INT8, compile it into a real graph, and
run it on an Intel NPU** (NPU 3720 / Meteor Lake and newer) via **OpenVINO**.

This path is **different in kind** from `--device cpu|gpu`. Those run brain's own
WGSL kernels op-by-op through the `Gpu`/`Step` seam. The NPU is a **whole-graph
compiler**: OpenVINO ingests an ONNX graph, compiles it for the NPU, and runs it.
So the NPU is **not** a third `gpu-core` backend — it is a separate
**export → quantize → compile → run** pipeline. DFL decode + NMS stay on the host
in pure Rust (NPUs don't do NMS), reusing the exact same post-processing as the
engine path.

## Pipeline at a glance

```
.weights ──export──▶ yolo.onnx (fp32) ──┐
   │                                     ├─▶ OpenVINO compile_model("NPU") ─▶ run ─▶ host DFL+NMS ─▶ boxes
   └──quantize(calib)─▶ yolo.int8.onnx ──┘
```

- **export / quantize / sim** are **pure Rust** and run on any machine (no NPU, no
  OpenVINO) — they live in `crates/onnx` (ONNX serializer) + `crates/npu`.
- **run / bench** (and `check`'s compile step) need OpenVINO + an Intel NPU.

## Crates

| Crate | Role |
|---|---|
| `crates/onnx` | pure-Rust ONNX graph model + serializer (vendored `prost` bindings; no `protoc` in the build) |
| `crates/npu` | YOLO→ONNX export, BN fold, brain-native INT8 PTQ, fake-quant simulator, OpenVINO runtime |

OpenVINO is a **default dependency** on x86_64 linux/windows (so `--device npu`
works with no special rebuild), pulled in with the `openvino` crate's
`runtime-linking` feature: the shared library is loaded at **run time**, so the
build stays green on machines without OpenVINO installed — a missing runtime only
surfaces when you actually open a session.

## Commands

```bash
# fp32 ONNX (pure Rust)
brain npu export   --weights out/yolo.weights --out out/yolo.onnx [--input S --opset 13]

# INT8 Q/DQ ONNX via brain-native PTQ (pure Rust): calibrate over representative
# images, compute symmetric per-tensor activation scales + per-channel weight scales
brain npu quantize --weights out/yolo.weights --calib calib/ --out out/yolo.int8.onnx \
                   [--input S --num-calib 300 --scales-out out/scales.json]

# structural ONNX check (always) + compile/op-coverage on a device (needs OpenVINO)
brain npu check    --onnx out/yolo.int8.onnx [--device NPU]

# run on the NPU; output format identical to `brain yolo detect`
brain npu run      --onnx out/yolo.int8.onnx --image sample.ppm --device NPU \
                   [--conf 0.25 --iou 0.45 --nc 80 --reg-max 16 --cache-dir out/npu-cache \
                    --hint latency|throughput --turbo --allow-fallback]

# latency p50/p99 + throughput
brain npu bench    --onnx out/yolo.int8.onnx --device NPU --hint throughput [--iters 200 --warmup 20]

# fp32 vs INT8 mAP@0.5 with NO NPU (fake-quant simulation) — the accuracy gate
brain npu sim      --weights out/yolo.weights --data data/detect [--calib calib/ --num-calib 300]

# convenience: route `yolo detect` through the NPU (auto-exports fp32)
brain yolo detect  --weights out/yolo.weights --image sample.ppm --device npu
```

Makefile: `make export/yolo-onnx`, `make quantize/yolo`, `make sim/yolo-int8`
(pure Rust); `make run/yolo-npu`, `make bench/yolo-npu` (need the NPU).

## Install (Meteor Lake / NPU 3720)

1. Install **OpenVINO 2024.x** runtime and the **Intel NPU driver** (Linux:
   `intel-driver-compiler-npu` / `intel-level-zero-npu` + level-zero; Windows: the
   Intel NPU driver from Intel's site).
2. Source the OpenVINO environment so the loader can find `libopenvino_c`:
   `source /opt/intel/openvino_2024/setupvars.sh` (sets `INTEL_OPENVINO_DIR` and
   `LD_LIBRARY_PATH`).
3. Build brain normally — no special flags: `make release`. Verify the NPU is
   visible with `brain npu check --onnx out/yolo.onnx --device NPU` (it lists the
   OpenVINO devices and tries to compile).

## Constraints

- **Static shapes.** The NPU plugin requires fixed dims; the export is static
  `[1,3,S,S]`. Letterboxing absorbs arbitrary input sizes; `run` validates the
  input against the compiled model's shape.
- **Op coverage.** Only backbone/neck/head go to ONNX (Conv, Sigmoid+Mul = SiLU,
  MaxPool, Resize, Concat, Split, Add). DFL decode + NMS stay on the host. Use
  `brain npu check` (and the OpenVINO exec-graph) to confirm ops land on the NPU
  rather than silently falling back to CPU.
- **Mixed precision.** Backbone/neck convs are INT8 (per-channel weights,
  per-tensor activations, symmetric, zero-point 0, ONNX Q/DQ form — what
  `NPU_QDQ_OPTIMIZATION` consumes). The head's final 1×1 convs are kept fp32 so the
  detection logits stay full precision.

## Accuracy methodology

The exported INT8 graph's arithmetic is reproduced **exactly** in brain's own
engine (fp32 "fake-quant"): weights replaced by `dequant(quant(fold_bn(w)))`
per-channel, BN folded into Conv+bias, and a per-tensor activation quant→dequant on
each conv input. This lets you measure INT8 accuracy with **no NPU**:

- `brain npu sim` reports `mAP@0.5` for fp32 vs the INT8 simulation on a dataset.
- CI gate (`crates/npu/tests/parity.rs`): cosine similarity of fp32 vs INT8-sim
  head logits (≈1.0 — the synthetic-shapes detector is near-lossless under INT8).

On the NPU, additionally compare OpenVINO's outputs to the fp32/sim reference
(`crates/npu/tests/`, gated by `BRAIN_NPU_AVAILABLE` + a present NPU).

## Performance knobs

`PERFORMANCE_HINT` (`--hint latency|throughput`), `CACHE_DIR` (`--cache-dir`, warm
recompiles), `NPU_TURBO` (`--turbo`), `NPU_TILES`, and
`NPU_COMPILATION_MODE_PARAMS=optimization-level=2 …` are set on the device before
compile. `--profile` enables `PERF_COUNT`.

## Calibration data

For `yolov8n @ 640` use ~300 representative **640px** images (a directory of binary
PPM/P6 files, or a brain detection-dataset dir). The synthetic 128px `data/detect`
set is the tiny-model CI vehicle, not representative for the 640px canonical model.
