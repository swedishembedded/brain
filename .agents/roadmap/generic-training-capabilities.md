# generic training capabilities - roadmap

A model that implements a few capability traits gets LoRA/QLoRA SFT, DPO, GRPO,
distillation and the conformance suite without per-model glue. What exists:
`model::training::TokenObjectives` (with the logit scale), the typed
`Unsupported` refusal, `model::adapter::sites::LoraSites`, `model::conformance`
(+ `rl::conformance`), the SDK's `ARCHS` table and `score_tokens`.

## Not yet done

- [ ] qwen35 and qwen35moe: implement `TokenObjectives` and
      `model::adapter::sites::LoraSites` (GDN layers' `in_proj_qkv/z/b/a/out_proj`,
      GQA `q/k/v/o`, all layers `gate/up/down`), flip `TOKEN_OBJECTIVES` in
      `crates/sdk/src/study.rs`, and register them in
      `crates/rl/tests/conformance.rs` (one `subject!` line plus a `Subject`).
- [ ] Conformance C8 (`chunked_head_ce_equals_unchunked`) is an ignored test
      that fails when run; give `Subject` a chunk-size hook once the shared
      chunked head cross-entropy lands and make it check loss, log-probabilities
      and gradients against the unchunked head.
- [ ] Conformance tiers: register the NF4 base tier (C9) for the models that get it.
- [ ] Move the serving `AdapterBank` (`qwen3/src/serve/adapters.rs`, built from
      `QwenConfig` and its own seven-site `LoraSite` enum) behind `LoraSites`, so
      qwen35 can use it, and stop `catalog/src/adapter_registry.rs` importing
      qwen3 types. Regression gates: `catalog/tests/multi_adapter.rs`,
      `qwen3/tests/engine_adapter_bank.rs`, `apiserve/tests/api.rs`. Also let
      `Qwen::attach_adapter_pairs` call `adapter::sites::validate` instead of its
      own shape checks.
- [ ] The chat fine-tuners (`ChatFineTune`, `ChatFullFineTune`,
      `PreferenceFineTune`) still call `qwen3::finetune`; a generic trainer over
      `Model` + `LoraSites` would let `ArchEntry::chat_fine_tune` be true for
      another architecture.
- [ ] `Improve::run` boxes its environment and verifier to dispatch through the
      table; if that proves a cost, make the table entry generic instead.
