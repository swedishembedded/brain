# GLM on the Intel NPU

Like YOLO and Qwen3, GLM reaches the Intel NPU through a separate export →
compile → run path (the NPU is a whole-graph OpenVINO compiler, not a per-op
backend, so it is not a `--device` of the served actions). The path is two
actions of `samples/models/glmdsa`, which needs the `npu` crate:

```bash
G=target/release/sample-models-glmdsa
$G export --weights <checkpoint> --seq <T> --out onnx=model.onnx
$G npu-generate --weights <checkpoint> --prompt "..." --max-new N --out text=out.txt
```

`export` writes a fixed-sequence-length ONNX graph. Because brain's
mixture-of-experts forward already evaluates every expert densely and masks
by the gate, the exported graph has no data-dependent routing — every expert
runs on every token, which trades some wasted compute for a graph the NPU's
static-shape compiler can actually run.

## What works today

`npu-generate` runs a greedy decode end to end: it exports the graph in
memory, compiles it with OpenVINO (`--ov-device npu|cpu|gpu|auto`, the NPU by
default) and decodes through it - no separate `export` step is needed.
Sampling beyond greedy (temperature/top-k/top-p) is not available on this
path.

`--int8` on either action selects a weight-only INT8 graph (~4x smaller than
fp32) that compiles and runs through OpenVINO.

Both actions read a brain checkpoint; `import` converts a Hugging Face
release into one. `generate` on the CPU or a GPU is the served action of
`brain glmdsa`.
