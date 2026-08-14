# glmdsa - roadmap

GLM-5.2 (`glm_moe_dsa`) decoder - MLA + sigmoid `noaux_tc` MoE + DSA indexer +
MTP - on brain's fp32/WGSL engine. Core forward/backward, learnability,
indexer distillation, KV-cache decode, and HF import are done and
parity-gated against the reference implementation. `brain glmdsa export`
produces an ONNX graph validated on real NPU hardware (fp32 only).

## Not yet done

- [ ] GLM serving contract - not yet discoverable/scheduled/batched/driven
      over D-Bus or the HTTP APIs like the other models
- [ ] `brain glmdsa infer --device npu` - the exported ONNX graph isn't wired
      into an inference command yet; today `export` hands you a graph to run
      through OpenVINO yourself
- [ ] INT8 weight-only export for the NPU path (fp32 only today)
- [ ] Full-size config (78 layers, 256 experts, ~155k vocab) is not runnable
      locally - only used for import shape validation; tests/training run on
      small presets
- [ ] Migrating the training graph (`build_forward`/`build_backward`) onto
      the row-compacted sparse MoE dispatch - a measured ~7x speedup exists
      for the inference-only path already landed, but training needs a
      correctly-designed row-compacted backward (scatter-add semantics) that
      has not been attempted, and the speedup is from a synthetic
      microbenchmark at this model's shape, not from a real checkpoint

Not deferrals, by design: the DSA indexer's backward is forward-only (trained
solely via distillation, detached from the LM loss); DSA sparse selection and
MTP are not exported to the NPU, which runs dense attention with MTP as a
host-side draft loop.
