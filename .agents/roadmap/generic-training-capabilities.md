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

The named algorithms (`rl::objective::external::PolicyGradientAlgorithm`: `grpo`, `dr-grpo`, `dapo`, `rloo`) are the
configurations the rows below are held to: `policy_gradient_regimes` trains each one's expansion on every family,
`each_algorithm_expands_to_the_settings_that_define_it` (`rl`), `a_named_algorithm_is_the_base_its_requests_settings_override`
(`trainjob`) and `every_named_algorithm_trains_and_the_result_records_the_settings_it_expanded_to` (`trainjob/tests/it/run.rs`,
toy `qwen3`) hold the wire to them.

| Regime | Level | Evidence |
|---|---|---|
| RLOO (`AdvantageNorm::LeaveOneOut`, contract v2) | L4 | L1 `rl` `advantage::every_normalisation_matches_its_fp64_oracle`, `a_pair_by_hand`, `leave_one_out_is_scaled_mean_centring_at_a_fixed_size_and_not_across_sizes`, `external_policy_gradient::the_advantages_are_each_groups_rewards_under_the_runs_normalisation`, the token weight as for every regime (`pg_batch`); L2 `rl/tests/rloo_gradcheck.rs`; L3 `rl::conformance::policy_gradient_regimes`; L4 `trainjob` `the_advantage_normalisation_is_named_on_the_wire` |
| DAPO (clip-higher 0.28, token-sum, `truncated: keep \| mask`, z-score) | L4 | L1 the token weight past an asymmetric band (`pg_batch`), the advantages of kept and masked truncated rollouts against hand values (`rl/tests/it/policy_gradient_stability.rs` `a_run_that_keeps_truncated_rollouts_trains_them_with_their_rewards_counted`, `data` `a_truncated_rollout_counts_only_where_the_run_keeps_it`, `the_rules_between_records_refuse_the_record_that_breaks_them`); L2 `rl/tests/dapo_gradcheck.rs`; L3 `rl::conformance::policy_gradient_regimes`; L4 `trainjob` `the_truncated_rollout_handling_is_named_on_the_wire`, the dry run `a_dry_run_counts_a_truncated_rollout_in_its_group_only_where_the_run_would_keep_it`. The over-length reward penalty is the caller's |
| KTO (`objective: feedback`, `generic-feedback-v1`) | L4 | L1 `rl` `feedback::the_term_is_the_papers_loss_and_its_weight_is_its_gradient`, `the_reference_point_is_the_clamped_mean_of_the_mismatched_log_ratios`; L2 `rl/tests/kto_gradcheck.rs` (desirable and undesirable, the reference point held); L3 `rl::conformance` (`kto (desirable)`, `kto (undesirable)` on every family, refused typed by a model without token objectives); L4 `trainjob` `a_feedback_request_takes_its_own_settings_and_refuses_the_rest_by_name`, `a_feedback_job_trains_a_qwen3_base_on_judged_completions`, the resumed job `a_stopped_lora_job_of_every_objective_continues_to_the_uninterrupted_adapter`, the dry run `a_feedback_dataset_is_checked_as_the_trainer_reads_it`; SDK `feedback_finetune.rs` (the judgement, not the text, orders the completions); CLI `brain finetune --objective feedback --desirable-weight --undesirable-weight`, `cli` `finetune_objectives::a_feedback_run_weighs_the_two_judgements_and_reports_their_log_ratios` |
| Distillation as a served job (`objective: distill`, `teacher: base \| start \| adapter:<digest>`, `DistillTopK`) | L4 | L1 `rl` `distill` unit tests (the top-K weight identity, the mean over scored positions); L2 `rl/tests/distill_gradcheck.rs`; L3 `rl::conformance` (`distill` on every family); L4 `trainjob` `a_distill_request_names_its_teacher_and_refuses_the_rest_by_name`, `a_distill_job_copies_an_adapter_of_the_store_into_a_fresh_one`, the resumed job (byte-identical adapter), SDK `distill_finetune.rs` (a fresh adapter closes its cross-entropy to a trained teacher; a continued one taught by the base moves back); CLI `--objective distill --teacher base|start|adapter:FILE --top-k`, `cli` `finetune_objectives::a_distillation_is_taught_by_the_base_the_start_adapter_or_an_adapter_file` |
| Advantage floor (`advantage_floor: zero`, positive-only arms) | L4 | L1 `rl` `advantage::the_zero_floor_clamps_after_the_whole_group_set_the_baseline`, through `open` in `the_advantages_are_each_groups_rewards_under_the_runs_normalisation`; the token weight is every regime's; L4 `trainjob` `the_advantage_floor_is_named_on_the_wire`, CLI `--advantage-floor` |
| IPO, SimPO, ORPO (`PairLoss`, beside DPO) | L4 (`objective: preference`, `loss`, `gamma`; `trainjob` `the_preference_objective_names_its_pair_loss`, `a_preference_job_trains_with_every_pair_loss`) | L1 `rl` `dpo::every_pair_loss_matches_its_oracle_and_its_weights_are_its_gradient` (fp64 oracles from each paper, weights by central differences, the reference-free losses blind to the reference); L2 `rl/tests/preference_gradcheck.rs`; L3 `rl::conformance` (`ipo`, `simpo`, `orpo` on every family); SDK `preference_finetune.rs` `every_pair_loss_trains_and_is_recorded`; CLI `--objective preference --loss dpo|ipo|simpo|orpo --gamma`, `cli` `finetune_objectives::a_preference_run_trains_the_named_pair_loss_and_scores_the_held_out_pairs` |
| Dr. GRPO (`LossAgg::ConstantLength`, mean-centred) | L4 | L1 `rl` `pg_batch::every_tokens_weight_is_minus_the_derivative_of_its_loss_inside_and_past_the_clip`, `grpo::each_aggregation_divides_a_step_as_its_definition_says`; L2 `rl/tests/dr_grpo_gradcheck.rs`; L3 `rl::conformance::policy_gradient_regimes`; L4 `trainjob` `the_constant_length_aggregation_takes_its_constant_or_the_runs_max_block` |

## Release checks for training and serving builds

A new build never becomes the default for a result already proven on an older one without these (standing rule, 2026-11-19: never
degrade a proven result unknowingly). The reference adapter is whichever adapter the operator pins as champion, with its recipe
and records; reproducing it must not need an expensive retrain, so the checks are tiered and stop early:

1. **Fixed-batch numerical parity.** On fixed real records the loss, gradients and post-step weights of the new path equal the old
   path's within the old path's own run-to-run band.
2. **Trajectory band.** About 20 training steps stay inside the old path's band (loss curve and held-out loss).
3. **Teacher-forced logit parity for serving.** The reference adapter served on the new build gives the same per-token logprobs
   as on the pinned build, within the documented numerics (an untruncated draw's `policy_trace` equals the decode's own logprobs).
4. **Small paired dev, only if tier 3 differs.** Early stop on a paired difference against the pinned build.

Reproducing the reference recipe within its seed band on the new build is the final gate before a build is frozen as the default.
Each run uses its own frozen binary copy (commit and sha recorded); never switch mid-run.

## Prompt parity and numerics rules (landed, kept as rules)

- A served prompt is the checkpoint template's rendering of the request: no hidden system text, tool schemas keep the request's
  member order. A parameter a model does not declare is refused by name (`unsupported_parameter`); `reasoning_effort` is declared
  only by flavors whose template reads it, with no server default. Every route is admitted through `bridge::admit`: declared
  defaults, undeclared parameters refused. An omitted sampler draws at the checkpoint's declared defaults, not at parser constants.
  Test: `qwen3` `served_prompt_template`. Results from builds before 1bdd5400c were served with an injected instruction.
- The record contract trains the token ids as sampled, admitted by read-back, plus `prompt_token_ids` checked against the rendered
  context (`prompt_token_ids_mismatch`).
- An untruncated draw's `policy_trace` is the decode's own logprobs (the GPU parity gate is tightened); a truncated draw's trace
  comes from decode, not a serial re-prefill. Numerics of the 27B FP8 model: KV tier is not the gap (0.019 int8 against 0.018 f32),
  sequence ESS about 0.5, so use `--importance truncate:2` for its policy gradient.
- The replies of every chat route carry the `brain` identity object (base and adapter digests), so an examiner can name the served
  model; a reply without it cannot be named in a verdict.

## Open (from the 2026-11-17..19 validation work)

- Batch-invariant decode: a request is repeatable alone but still depends on co-batching (remaining 2.6e-3 on the 27B; exact decode
  costs 0.34x throughput). Prefill FP8 E4M3 activations against bf16 decode, and two bf16 weight-rounding candidates, are the suspects.
- `preserve_thinking` is declared and inert for Qwen3 models.
- A frozen `brain-<sha>` binary finds no worker beside it unless the frozen layout is one directory per commit; supervision stays off
  otherwise.
- The on-card LoRA footprint test leaves +1.3-2.2 GiB unitemised against the estimate; qwen35 and Ripple estimates still use the old
  rate; the CPU tightness test is 4.5 KB short on CUDA (pre-existing).
- A served training job needs its client to cancel it on interrupt (owner-only cancel and `GET /v1/jobs` landed).
- Served preference/feedback jobs: KTO's reference pass was 55 minutes and 170 s per step; see `training-efficiency.md`.
- Distillation from a teacher served elsewhere (step 10b) below is still the open training-capability item.

## Not yet done

- [ ] Distillation from a teacher served elsewhere (step 10b of the plan, not built): the student would read the
      teacher's top-K from records rather than from a forward of its own base. Needed, in order: (1) the served `score`
      path returns, per generated position, the K largest `(id, logprob)` of the full untempered-then-tempered softmax
      (`model::readout::topk_rows_steps` over the scored rows in `llmserve::score`; qwen3 and the qwen35 scorer, the
      latter another lane's file); (2) `apiserve::sampling::refuse_unsupported_openai` lifts its `top_logprobs` refusal
      when `policy_trace` is set (K bounded, say 20) and the trace carries `generation_top_logprobs`; (3) the record
      contract takes `teacher_top_logprobs` (ids and values per supervised token) on an assistant message, kept in a
      prepared-split sidecar like `behavior_logprobs`; (4) `DistillTeacher::Recorded` builds `DistillExample`s from it
      (renormalised `TopK`). Tests: the trace's top-K equals the teacher-forced logits' at f32 on the toy, and a
      distillation from recorded top-K equals one from the live teacher of the same base.

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
