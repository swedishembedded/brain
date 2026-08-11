# GLM on the Intel NPU

Like YOLO and Qwen3, GLM reaches the Intel NPU through a separate export →
compile → run path (`--device npu` is not the same code path as `--device
cpu|gpu`, since the NPU is a whole-graph OpenVINO compiler, not a per-op
backend).

```bash
brain glm export --weights <checkpoint> --out model.onnx --seq <T>
```

This exports a fixed-sequence-length ONNX graph. Because brain's
mixture-of-experts forward already evaluates every expert densely and masks
by the gate, the exported graph has no data-dependent routing — every expert
runs on every token, which trades some wasted compute for a graph the NPU's
static-shape compiler can actually run.

## What works today

fp32 export is implemented and validated on real NPU hardware (graph
compiles, runs, and its output matches brain's own CPU/GPU forward). Wiring
that exported graph into `brain glm infer --device npu` — and an INT8 export
path — are not done yet; today `export` produces a graph you can run through
OpenVINO yourself, but `infer --device npu` does not yet exist. `--device
cpu`/`--device gpu` are fully supported for GLM today.
