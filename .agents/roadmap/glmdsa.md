# glmdsa - roadmap

GLM-5.2 (`glm_moe_dsa`) decoder - MLA + sigmoid `noaux_tc` MoE + DSA indexer +
MTP - on brain's fp32/WGSL engine. Core forward/backward, learnability,
indexer distillation, KV-cache decode, and HF import are done and
parity-gated against the reference implementation. `brain glmdsa export`
produces an ONNX graph validated on real NPU hardware (fp32 only).

## Not yet done

- [ ] The RMSNorm backward (`rms_inv`/`rmsnorm_dx`) and the DSA indexer's
      LayerNorm are still per-element kernels; only `norm_fwd` selects the
      coalesced `rmsnorm_rows` (measured 23.5x on one decode token's norms,
      186 ms -> 7.9 ms at the 78-layer shape)
- [ ] `norm_fwd` normalizes at the 1e-6 `rmsnorm.wgsl` hardcodes, not at
      `GlmConfig::rms_eps` (1e-5); `rmsnorm_eps`/`rmsnorm_eps_rows` are not
      registered, so the config field is round-tripped but not honoured
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
- [ ] **A streaming `generate`, which is what puts GLM on the HTTP chat
      routes.** `crates/apiserve/src/catalog.rs::api_caps` derives HTTP
      exposure from action SHAPE, not from a per-model list:
      `/v1/chat/completions` and `/v1/messages` need an action named
      `generate` that is `.streaming()`, takes `prompt`/`messages`/`text`, and
      outputs `Media::Text`. `glmdsa::caps` satisfies every clause except
      `streaming`, so GLM is on `brain do` and D-Bus but not on the HTTP
      dialects. The char-level GPT baseline is in the same position
      (`resident_llm::GptResident` passes `generate_spec(..., chat=false)`);
      qwen3, qwen35moe and qwen35 all pass `true` and are exposed.

      The fix is NOT to add `.streaming()` to the spec - that would make the
      manifest claim something the action does not do. `sample::generate_kv`
      produces its tokens and returns; per-token `Progress::delta` emission has
      to exist first, and then the flag describes it. Doing it in that order is
      the difference between a served streaming endpoint and a manifest that
      lies to the router.
- [x] `crates/cli/src/resident_llm.rs::GlmInstance::run` now calls
      `glmdsa::sample::generate_kv` (the KV-cached fast path), matching the
      direct `brain glmdsa generate` path (`glmdsa::caps::GenerateAction`,
      which already called `generate_kv`). Before this the served and direct
      surfaces silently disagreed about how GLM samples - not an RNG-ordering
      bug (`sample_logits` draws exactly one `rng.next_f32()` per emitted
      token, identically ordered in both `generate` and `generate_kv`), but a
      real numerical one: `generate_kv` applies GLM's untied `lm_head` on the
      host in a scalar loop, agreeing with the device path's logits to only
      ~1e-3 - enough to flip a sampled token at the served default
      (`temp=0.8`, `top_k=40`) even though bit-identity still holds at
      `temperature=0` (greedy; see the extended
      `sample::kv_gen_tests::generate_kv_matches_recompute_greedy`).
- [x] `Instance::run_batch` for GLM is the serial default, and now SAYS why in
      a comment (`GlmInstance`'s `run_batch`) plus the fuller rationale in
      `GlmResident`'s module doc - **this item's original wording above was
      itself wrong and is corrected here, not obeyed literally.** Both
      "honest options" it named - batching the prefill, or adopting
      `model::serve::PagedDecoder` - require a batched forward this model
      does not have: `Glm::step` takes one token id and one KV-cache,
      `Glm::logits_all_compact` takes one window, `Glm::set_batch` builds one
      TRAINING sequence - there is no N axis anywhere in the MLA attention or
      the sigmoid `noaux_tc` MoE dispatch to widen. Writing a `run_batch`
      override that loops `run()` would also be a byte-for-byte redundant
      copy of `residency::model`'s own default (`crates/residency/src/
      model.rs` lines ~53-58) - this repo's established convention for
      exactly this situation is a comment where the override would go, not a
      written-out duplicate (`resident_asr.rs`'s Qwen3-ASR, `resident_restore.
      rs`'s CodeFormer/VQGAN - both comment-only, no override). **Real
      batched serving needs a batched MLA + sigmoid `noaux_tc` MoE forward
      first** - new kernel/architecture work, explicitly out of scope here,
      tracked as its own item below.
  - [ ] **Batched MLA + sigmoid `noaux_tc` MoE forward for GLM** - the actual
        missing prerequisite for both real batched serving and the
        `qwen35moe`/`qwen3`-style paged-KV `Scheduler` machinery. Until this
        exists, GLM's `run_batch` correctly stays the residency default
        serial loop (see above); do not reach for `PagedDecoder` without it.
  - [ ] `GlmInstance`'s served (resident) path has no automated test on this
        box: `GlmResident::activate` requires `BRAIN_GLMDSA_WEIGHTS` to point
        at a checkpoint with an embedded char vocab, and no such fixture is
        committed (this repo's convention gitignores test fixtures/goldens
        under `testdata/`). A real, disclosed coverage gap, not a hidden one -
        `crates/glmdsa/src/sample.rs`'s `kv_gen_tests` cover the sampling
        function itself at the model level instead. Closing this needs a tiny
        committed GLM checkpoint with a char vocab, which is its own item, not
        something to manufacture as a side effect of unrelated work.
- [x] A runnable `examples/` client for GLM, like the other served models have
      (`examples/llm/glmdsa.py`, D-Bus only - GLM's `generate` is not
      `.streaming()`, see the item above, so it is not reachable over
      `/v1/chat/completions`/`/v1/messages`; `--openai`/`--anthropic` are
      refused with that explanation rather than silently doing nothing).
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
