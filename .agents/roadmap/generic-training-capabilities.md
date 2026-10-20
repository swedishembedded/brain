# generic training capabilities - roadmap

A model that implements a few capability traits gets LoRA/QLoRA SFT, DPO, GRPO,
distillation and the conformance suite without per-model glue. What exists:
`model::training::TokenObjectives` (with the logit scale), the typed
`Unsupported` refusal, `model::adapter::sites::LoraSites`, `model::conformance`
(+ `rl::conformance`; C1-C9 including the chunked head, with a per-subject
tensor/direction budget), `model::score` (held-out scoring over any `Model`),
`model::finetune` (the `ChatTrainer` seam the generic full-parameter, LoRA and
preference fine-tuners run on), the SDK's `ARCHS` table and `score_tokens`.
qwen3, qwen35, qwen35moe and gpt2 are registered in the conformance suite
(`rl/tests/conformance.rs`); qwen3 registers the bf16 and int8 frozen-base
tiers, qwen35 bf16 and NF4. Status re-read against the source on 2026-10-20.
The tier, adapter and trainer work is part of the larger plan in
`quantized-training.md`, which this file feeds.

## Not yet done

- [ ] Conformance tiers: register the tiers still missing as they land (FP8; an
      int8 tier for qwen35 once its trainer accepts one). `Subject::tiers` is the
      hook; each tier carries its own gradient tolerance.
- [x] The serving `AdapterBank` is `model::adapter::bank` over `LoraSites` /
      `AdapterServing`; qwen3 and qwen35 (fp32 and GGUF engines) serve adapters
      from it as overlays without folding into the base, and
      `catalog/src/adapter_registry.rs` names no architecture (the family table
      is `catalog/src/served_family.rs`). `Qwen::attach_adapter_pairs` checks
      shapes with `adapter::sites::validate`. Regression gates:
      `catalog/tests/multi_adapter.rs`, `qwen3/tests/engine_adapter_bank.rs`,
      `qwen35/tests/adapter_bank.rs`, `apiserve/tests/api.rs`.
- [ ] `ChatFullFineTune` and `PreferenceFineTune` run through
      `model::finetune::ChatTrainer` (qwen3, qwen35). Not yet: qwen35moe's
      trainer; a Hugging Face exporter for qwen35 full fine-tunes (it writes
      brain-format weights); preference-pair scoring for qwen35 (it needs a
      reference-adapter scorer over `Model`); a dtype-aware optimizer state
      behind `model::finetune::OptimizerState` (lane Q5).
- [ ] `Improve::run` boxes its environment and verifier to dispatch through the
      table; if that proves a cost, make the table entry generic instead.
