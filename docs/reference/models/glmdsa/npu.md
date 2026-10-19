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

`brain glm infer --device npu` exists and runs an fp32 greedy decode end to
end: it exports the graph, compiles it with OpenVINO, and decodes through it
directly - no separate `export` step needed to use the NPU from `infer`.
Sampling beyond greedy (temperature/top-k/top-p) is not available on this
path, same as GLM's CPU/GPU decode.

`brain glm export --int8` also works, producing a weight-only INT8 ONNX graph
(~4x smaller than the fp32 export) that compiles and runs correctly through
OpenVINO - but `infer --device npu` does not yet expose an `--int8` flag, so
today INT8 is reachable only via `export` (run the exported graph through
OpenVINO yourself), not through `infer` end to end.

`--device cpu`/`--device gpu` are fully supported for GLM today.
