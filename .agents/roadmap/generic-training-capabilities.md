# generic training capabilities - roadmap

A model that implements a few capability traits gets LoRA/QLoRA SFT, DPO, GRPO,
distillation and the conformance suite without per-model glue. What exists:
`model::training::TokenObjectives` (with the logit scale), the typed
`Unsupported` refusal, `model::adapter::sites::LoraSites`, `model::conformance`
(+ `rl::conformance`), the SDK's `ARCHS` table and `score_tokens`.
qwen3, qwen35, qwen35moe and gpt2 are registered in the conformance suite.

## Not yet done

- [ ] qwen35 and qwen35moe batched Gated-DeltaNet training on the CPU backend:
      a batch of more than one row aborts with heap corruption in the backward
      (knowledge 291). Find the CPU kernel that mishandles `b > 1`, then drop
      `require_accelerator` from `crates/rl/tests/conformance.rs`.
- [ ] The qwen35/qwen35moe conformance run takes minutes, nearly all in the
      finite-difference check; give `Subject` a tensor or direction budget.
- [ ] Conformance C8 (`chunked_head_ce_equals_unchunked`) is an ignored test
      that fails when run; give `Subject` a chunk-size hook once the shared
      chunked head cross-entropy lands and make it check loss, log-probabilities
      and gradients against the unchunked head.
- [ ] Conformance tiers: qwen35 registers bf16 and NF4, qwen3 bf16 and int8.
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
