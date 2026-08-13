# Running the YOLO detector on the Intel NPU (OpenVINO)

brain can quantize the YOLO detector to INT8, compile it into a real graph,
and run it on an Intel NPU (NPU 3720 / Meteor Lake and newer) via OpenVINO.

This path is different in kind from `--device cpu|gpu`, which run brain's
engine op-by-op. The NPU is a whole-graph compiler: OpenVINO ingests an ONNX
graph, compiles it for the NPU, and runs it — an export -> quantize ->
compile -> run pipeline, not a third live backend. Box decoding and NMS
always run on the host in Rust after the NPU returns raw logits (NPUs don't
do NMS).

## Pipeline at a glance

```
.safetensors ──export──▶ yolo.onnx (fp32) ──┐
   │                                     ├─▶ compile for NPU ─▶ run ─▶ host decode+NMS ─▶ boxes
   └──quantize(calib)─▶ yolo.int8.onnx ──┘
```

`export`, `quantize`, and `sim` are pure Rust and run on any machine — no NPU
or OpenVINO install required. `run`, `bench`, and `check`'s compile step need
OpenVINO plus an Intel NPU.

## Commands

```bash
# fp32 ONNX (pure Rust)
brain npu export   --weights out/yolo.safetensors --out out/yolo.onnx [--input S --opset 13]

# INT8 Q/DQ ONNX via brain-native post-training quantization (pure Rust):
# calibrate over representative images, compute activation + weight scales
brain npu quantize --weights out/yolo.safetensors --calib calib/ --out out/yolo.int8.onnx \
                   [--input S --num-calib 300 --scales-out out/scales.json]

# structural ONNX check (always) + compile/op-coverage check on a device (needs OpenVINO)
brain npu check    --onnx out/yolo.int8.onnx [--ov-device NPU]

# run on the NPU; output format identical to `brain yolo detect`
brain npu run      --onnx out/yolo.int8.onnx --image sample.ppm --ov-device NPU \
                   [--conf 0.25 --iou 0.45 --nc 80 --reg-max 16 --cache-dir out/npu-cache \
                    --hint latency|throughput --turbo --allow-fallback]

# latency p50/p99 + throughput
brain npu bench    --onnx out/yolo.int8.onnx --ov-device NPU --hint throughput [--iters 200 --warmup 20]

# fp32 vs INT8 mAP@0.5 with no NPU present (fake-quant simulation) — the accuracy gate
brain npu sim      --weights out/yolo.safetensors --data data/detect [--calib calib/ --num-calib 300]

# convenience: route `yolo detect` through the NPU (auto-exports fp32)
brain yolo detect  --weights out/yolo.safetensors --image sample.ppm --device npu
```

## Install (Meteor Lake / NPU 3720)

1. Install OpenVINO 2024.x runtime and the Intel NPU driver (Linux:
   `intel-driver-compiler-npu` / `intel-level-zero-npu` plus level-zero;
   Windows: the Intel NPU driver from Intel's site).
2. Source the OpenVINO environment so the runtime loader can find it:
   `source /opt/intel/openvino_2024/setupvars.sh`.
3. Build and run brain normally — no special flags needed. Verify the NPU is
   visible with `brain npu check --onnx out/yolo.onnx --ov-device NPU` (it lists
   the OpenVINO devices and tries to compile).

## Constraints

- **Static shapes.** The NPU plugin requires fixed dims; export is a static
  `[1,3,S,S]` graph. Letterboxing absorbs arbitrary input sizes; `run`
  validates the input against the compiled model's shape.
- **Op coverage.** Only backbone/neck/head go to ONNX (Conv, Sigmoid+Mul as
  SiLU, MaxPool, Resize, Concat, Split, Add). Box decoding and NMS stay on
  the host. Use `brain npu check` to confirm ops actually land on the NPU
  rather than silently falling back to CPU.
- **Mixed precision.** Backbone/neck convolutions are INT8 (per-channel
  weights, per-tensor activations, symmetric, zero-point 0). The head's
  final 1x1 convolutions are kept fp32 so detection logits stay full
  precision.

## Accuracy methodology

The exported INT8 graph's arithmetic can be reproduced exactly in brain's own
engine as an fp32 "fake-quant" simulation, so INT8 accuracy can be measured
without any NPU present:

```bash
brain npu sim --weights out/yolo.safetensors --data data/detect
```

This reports fp32 vs INT8-simulated mAP@0.5 on a dataset, and closely tracks
what you'll see running the same graph on-device.

## Performance knobs

`--hint latency|throughput` picks OpenVINO's performance hint, `--cache-dir`
enables warm recompiles, `--turbo` enables NPU turbo mode, and `--profile`
enables per-op timing.

## Calibration data

For the 640px pretrained model, use roughly 300 representative 640px images
(a directory of binary PPM/P6 files, or a brain detection-dataset directory)
for `--calib`. A small synthetic dataset is fine for quick smoke-testing but
isn't representative of real-world calibration for the 640px model.
