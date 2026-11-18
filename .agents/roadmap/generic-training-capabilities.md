# generic training capabilities - roadmap

Scope: Generic LoRA/QLoRA/DPO/GRPO/distillation over capability traits.

A model that implements a few capability traits gets LoRA/QLoRA SFT, DPO, GRPO,
distillation and the conformance suite without per-model glue. What exists:
`model::training::TokenObjectives` (with the logit scale), the typed
`Unsupported` refusal, `model::adapter::sites::LoraSites`, `model::conformance`
(+ `rl::conformance`; C1-C9 including the chunked head, with a per-subject
tensor/direction budget), `model::score` (held-out scoring over any `Model`),
`model::finetune` (the `ChatTrainer` seam the generic full-parameter, LoRA and
preference fine-tuners run on), the SDK's `ARCHS` table and `score_tokens`.
qwen3, qwen35, qwen35moe and gpt2 are registered in the conformance suite
(`rl/tests/conformance.rs`); qwen3 registers the bf16, int8, NF4 and FP8
frozen-base tiers, qwen35 bf16, NF4 and FP8. DPO, GRPO and top-K distillation are
`ModelFit` visitors in `rl::fit`; `model::acceptance` runs one LoRA workflow over
qwen3, qwen35, qwen35moe and flux2 (`rl/tests/lora_acceptance.rs`); a trainer
declares its base formats (`ChatTrainer::LORA_BASES`). Status re-read against the source on 2026-10-20.
The tier, adapter and trainer work is part of the larger plan in
`quantized-training.md`, which this file feeds.

`brain::toy::write_toy_chat_checkpoint` writes a 2-layer, 32-wide `qwen3` checkpoint with a byte-level tokenizer and a bare
chat template (stops at `<|im_end|>`, 256-token context) that a LoRA run trains and samples on the CPU in seconds;
`sdk/tests/it/toy_training.rs` holds a held-out cipher task on it (a random model answers none, the adapter answers unseen prompts).
An orchestrator's learning-method smokes build on it. Not supported on it: `merge_adapter` (it needs a
Hugging Face directory), so on-policy methods continue an adapter instead of folding it.

## Training regimes (plan `training-regimes`, lane 1)

The acceptance levels are the plan's (D5): L1 an fp64 hand oracle, each token's weight minus the derivative of its loss
in its log-probability past the clip too; L2 `gradcheck::directional_check` on a toy `qwen3` (`rl/tests/<name>_gradcheck.rs`);
L3 one step lowers the regime's own loss on every conformance family (`rl::conformance`), or the family refuses it
typed; L4 the `brain/train` params validated, another objective's refused by name, the dry run. L5-L7 (the caller's
view, a served toy that learns, real weights) are other lanes' and are not claimed here.

| Regime | Level | Evidence |
|---|---|---|
| Dr. GRPO (`LossAgg::ConstantLength`, mean-centred) | L4 | L1 `rl` `pg_batch::every_tokens_weight_is_minus_the_derivative_of_its_loss_inside_and_past_the_clip`, `grpo::each_aggregation_divides_a_step_as_its_definition_says`; L2 `rl/tests/dr_grpo_gradcheck.rs`; L3 `rl::conformance::policy_gradient_regimes`; L4 `trainjob` `the_constant_length_aggregation_takes_its_constant_or_the_runs_max_block` |

## Not yet done

- [ ] Conformance tiers: register the tiers still missing as they land (an
      int8 tier for qwen35 once its trainer accepts one: today it refuses I8 and
      Q4 bases by name, and C8 now also holds the weighted loss). `Subject::tiers` is the
      hook; each tier carries its own gradient tolerance.
- [x] The serving `AdapterBank` is `model::adapter::bank` over `LoraSites` /
      `AdapterServing`; qwen3 and qwen35 (fp32 and GGUF engines) serve adapters
      from it as overlays without folding into the base, and
      `llmserve/src/adapter_registry.rs` names no architecture (each resident
      hands it its `adapter_release::ServedFamily`). `Qwen::attach_adapter_pairs` checks
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
