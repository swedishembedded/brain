# glmdsa - roadmap

GLM-5.2 (`glm_moe_dsa`) decoder - MLA + sigmoid `noaux_tc` MoE + DSA indexer +
MTP - on brain's fp32/WGSL engine. Core forward/backward, learnability,
indexer distillation, KV-cache decode, and HF import are done and
parity-gated against the reference implementation. `brain glmdsa export`
produces an ONNX graph validated on real NPU hardware (fp32 only).

## Not yet done

- [x] GLM discovery and the direct action path. `glmdsa::caps` now carries a
      **weight-free** manifest plus a `GlmProvider`, wired into
      `cli::catalog::models()`, so `brain caps` lists `brain/glm` and
      `brain glmdsa generate` runs it with no checkpoint on the box.

      Scheduling and D-Bus/HTTP serving were never actually missing -
      `cli::resident_llm::GlmResident` has always implemented `ResidentModel`
      and been registered in `resident.rs::build_executor`, which the serving
      contract accepts. What was missing is that its manifest is only built
      when `BRAIN_GLMDSA_WEIGHTS` is set, so on a box with no GLM checkpoint
      the model did not appear in `brain caps` **at all**, while every other
      model advertises itself weight-free and takes `weights` as a request
      parameter. Discovery that depends on deployment state is discovery a
      client cannot rely on.

      `GlmResident::manifest` now returns `glmdsa::caps::manifest_resident()`
      (the same definition minus the `weights` param the service supplies
      itself) rather than building its own `ActionSpec`, so the served and
      direct surfaces cannot advertise different parameters for one action.
      The model ref stays `brain/glm`, which is what `modelref::alias::ROWS`,
      `perf_cli` and the checkpoint `ModelCard` already use - `glmdsa` is the
      *architecture* id, a different namespace.
- [ ] `Instance::run_batch` for GLM is the serial default and does not yet say
      why - the decoder is autoregressive, so the honest options are batching
      the prefill or adopting `model::serve::PagedDecoder` (see
      `.agents/rules/serving-contract.md`), not a comment
- [ ] A runnable `examples/` client for GLM, like the other served models have
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
