<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->
# training correctness - roadmap

Scope: Source-level training-correctness audit (2026-11-08): items T001-T060, their gates and the conformance-suite plan.

Goal: no training run silently optimises a different objective, consumes different supervision or updates different
state than its validated plan declares; training stays architecture-generic (shared paths, refusal by missing
capability, never by architecture name). The audit is source-level, so each item is a hypothesis until checked
against the code: a finding that does not hold is recorded as such with its evidence, and a test pinning the correct
behaviour still lands. Each item completes only with its regression test (red before, green after) and the gate named.

Format: `- [ ] T### severity - file pointer - required completion gate`.

## Items
- [x] T001 P0 - crates/model/src/train.rs (mask read, `.ok()` fallback) - mask problem refuses before step 1; corrupt-byte/length test
- [x] T002 P0 - crates/model/src/train.rs (example-indexed dataset fallback) - oversized example refuses or splits into masked windows; no cross-example supervision test
- [x] T003 P0 - crates/rl/src/objective/grpo.rs (RatioBasis::Trainer) - stale rollouts: behaviour policy identity; two-policy ratio test (refuses a stale or mismatching engine; reconstructing the behaviour policy from a frozen snapshot is not implemented)
- [x] T004 P0 - crates/rl/src/objective/grpo.rs (top-k/top-p old_lp) - exact truncated likelihood or refuse; truncated-sampling test
- [x] T005 P0 - crates/rl/src/objective/grpo_term.rs (capped log-ratio / k3) - forward and backward agree; FP64 reference + finite differences
- [x] T006 P0 - crates/rl/src/objective/dpo.rs (empty step) - empty step changes no weights, moments or step count
- [x] T007 P0 - crates/rl/src/objective/distill.rs (empty step) - same as T006 for distillation
- [x] T008 P0 - crates/model/src/train.rs (fit_controlled) - transactional update gate in the shared loop; strict mode; NaN injection test (shared gate landed as `model::stepgate`; `FitOpts::strict_updates` is exposed on the CLI, the SDK and the served job; GRPO and external policy gradient report their verdicts to the shared applied-update count)
- [x] T009 P1 - crates/rl/src/objective/distill.rs - one normalisation convention; dense-softmax mean-KL oracle, unequal lengths
- [x] T010 P1 - crates/rl/src/objective/dpo.rs (pack_row) - streaming DPO refuses truncation
- [x] T011 P1 - crates/rl/src/objective/distill.rs (pack_row, TopK::from_logits) - refuse truncation and zero-information batches; validated teacher distribution
- [x] T012 P1 - crates/model/src/train.rs (schedule_fingerprint) - identity covers Adam beta1/beta2/eps; resume refuses a change
- [x] T013 (adam, gemm precision, native sparse attention in the loop's fingerprint; wave 2 added the resolved base tier, rendering and modules_to_save to the ChatFineTune identity and the reference to the DPO one) P1 - crates/model/src/train.rs (schedule_fingerprint) - identity covers GEMM compute precision and the whole resolved run configuration
- [x] T014 (the warm start is announced and returned as StartKind; it is not written into the checkpoint header, which is each model's own save) P1 - crates/model/src/train.rs (build_or_warm_start) - separate exact resume, weight warm start, adapter continuation (names and lineage)
- [x] T015 P1 - crates/rl/src/objective/grpo.rs (attach_references) - incomplete references refuse; unsupported EOS positions distinguished
- [x] T016 (recorded behaviour log-probs are the default exact mode; the sampler policy identity is not stored per record) P1 - crates/rl/src/objective/external.rs - explicit exact vs proxy behaviour logprobs with provenance
- [x] T017 P1 - crates/rl/src/objective/external.rs - missing rollout sidecars never select imitation semantics silently
- [x] T018 P1 - crates/model/src/train.rs (loss_probe) - pure probe; persistent objective cursor; resume equivalence test (probe is pure for DPO and distillation; a resumed run continues the objectives' own pair, example and closure-fed streams, tested for dataset lengths not divisible by the step size)
- [x] T019 (model::training_plan::ValidatedTrainingPlan built by the ChatFineTune, ChatFullFineTune and PreferenceFineTune builders, hence the CLI and the served job; per-architecture CLIs that build FitOpts directly are checked by the shared loop only) P1 - central ValidatedTrainingPlan: options, schedules, DPO beta (ablation flag), GRPO and policy-gradient numerics, distill row length; token ids, rewards and advantages checked by the objectives before a step
- [x] T020 (ids, probabilities, normalisation, zero-information, k=0; vocabulary range and discarded tail mass are not checked yet) P1 - crates/rl/src/objective/distill.rs - duplicate/out-of-range ids, non-finite or unnormalised teacher probabilities (with T011)
- [x] T021 P1 - DPO reference checkpoint - reference (base digest, continued-adapter digest, base format) in the run identity
- [x] T022 (strict always, not a mode: chat, rollout, teacher-context and preference records parse into typed `deny_unknown_fields` structs and a stray or missing field refuses the record by line and field: `data/src/chat.rs::from_jsonl_rejects_an_unexpected_top_level_field` and its siblings, `data/src/preference.rs::malformed_pairs_are_refused_by_line_and_field`, `data/tests/it/chat_image_parts.rs::what_the_reader_refuses_is_refused_by_line`, `data/tests/it/chat_teacher_context.rs::a_record_that_misstates_its_teacher_context_is_refused_naming_the_field`) P1 - structured datasets - strict schema mode
- [x] T023 (declared-vs-built tensors and sizes, full-FT trained set, adapter pairs on declared tensors, run in the conformance suite for gpt2, qwen3, qwen35, qwen35moe, ricursive; which sites an architecture adapts is its own policy and is not asserted) P1 - per architecture - expected-vs-actual trainable tensor audit
- [x] T024 (frozen-base tiers: conformance C9 for qwen3, qwen35 and ricursive, `model/tests/it/qops_frozen_lora_grad.rs::every_weight_tier_gives_the_lora_gradients_of_the_decoded_reference` against f64 over host-decoded weights, `qwen3/tests/it/lora_packed_bases.rs`; C9 has no tier to hold for qwen35moe, whose LoRA trainer declares an f32 base only (`qwen35moe/src/finetune.rs` `LORA_BASES`) and refuses resident narrow weights, nor for gpt2, which trains no adapter; optimizer state: `optim/tests/it/adamw_bf16_parity.rs` and `adamw_q8_parity.rs` hold device bf16 and int8 AdamW to the host reference step; backend: `qwen35/tests/it/lora_cuda_matches_cpu.rs::the_cuda_fp32_tier_trains_like_the_cpu_backend`, `qwen3/tests/it/train_cuda_matches_cpu.rs` and `qwen35moe/tests/it/train_cuda_matches_cpu.rs` (LoRA and full fine-tune, three AdamW steps on tiny random weights: every loss and trained-tensor gradient within the binary32 round-off bound of `testutil/src/training.rs`, derived from the longest contraction and the reductions in series; the CUDA side placed by `gpu_core::testgpu::on_cuda`, which skips naming a missing card, a device that does not open within 30 s, or a process whose backend is not CUDA)) P1 - verification gap - cross-precision and cross-backend training parity
- [x] T025 (`qwen3/tests/it/accumulation_normalisation.rs`: sequential and overlapped-pipeline accumulation over records of 1 and 5 supervised positions equal one forward of both as one batch, which found and fixed the pipeline defect of knowledge #453; ranks: `model/tests/it/distributed_model.rs::reduce_grads_weighted_gives_the_per_token_mean_of_unequal_ranks`, weighted by the step's supervised count in `model::train`; objectives: `rl/tests/it/external_policy_gradient_scale.rs`, `rl/tests/grpo_fixes.rs::token_level_normalisation_is_length_invariant`, `rl/tests/distill_normalisation.rs`) P1 - verification gap - distributed and accumulated token normalisation
- [x] T026 (SFT `sdk/tests/it/chat_finetune.rs::a_cancelled_run_resumes_to_the_uninterrupted_result` and the selecting run, qwen35 `chat_finetune_qwen35.rs::a_cancelled_qwen35_run_resumes_to_the_uninterrupted_adapter`, DPO `preference_finetune.rs::a_cancelled_preference_run_resumes_to_the_uninterrupted_result` and `rl/tests/objective_cursor.rs`, distillation `objective_cursor.rs`, external policy gradient `chat_finetune_policy_gradient.rs::a_cancelled_policy_gradient_run_resumes_to_the_uninterrupted_adapter` (with its step statistics), GRPO `objective_cursor.rs::a_resumed_grpo_run_continues_its_rollout_stream` (stops inside a group, with its counters): each bit-identical to the uninterrupted run; the GRPO and policy-gradient cases found the defects of knowledge #454. An external rollout engine is rebuilt by its caller on a resume and synced at the next group, which the uninterrupted run may not do at that step) P1 - verification gap - objective-cursor resume
- [ ] T027 (`rl/tests/grpo_fixes.rs`: `a_truncated_completion_is_rewarded_on_the_trained_span`, `the_policy_learns_to_stop`, `a_reference_without_the_stop_token_position_is_complete`, `an_incomplete_reference_is_refused_not_padded_with_behaviour_logprobs`; `rl/tests/it/policy_gradient_stability.rs::a_truncated_rollout_is_never_drawn_and_a_dataset_of_only_those_is_refused`. Open: an overlong-length penalty is not offered (it belongs to the DAPO preset of T039), so its contract has nothing to test yet) P1 - verification gap - EOS, truncation and overlong reward contracts
- [x] T028 (probes and splits: `rl/tests/it/document_study.rs::no_probe_answer_appears_in_any_trained_span`, `rl/src/continual.rs` frozen-probe refusals, `rl/src/improve.rs::explore_anchor_split_panics_on_a_colliding_seed`, `sdk/tests/it/emit_held_out_records.rs`, the leak refusal of `sdk/tests/it/decision_train.rs`; a `ChatFineTune` or preference held-out set that repeats training records is used deliberately to measure fit, so it is scored and the overlap reported, not refused: `held_out_repeats` (and `held_out_text_repeats`) on the outcome, in the training record and in the served job's report, `sdk/tests/it/chat_finetune.rs::a_held_out_set_that_repeats_training_records_reports_the_overlap`, `sdk/tests/it/preference_finetune.rs::held_out_pairs_that_repeat_training_pairs_report_the_overlap`) P1 - verification gap - held-out reuse
- [x] T029 (GRPO KL `rl/tests/grpo_fixes.rs::the_logged_kl_is_the_analytic_one`; GRPO clip fraction `rl/tests/grpo_fixes.rs::the_logged_clip_fraction_is_the_one_the_band_gives` (both edges and the inside of the band reached, counted from an independent forward; logged as `grpo_clip_frac`, `_high`, `_low` and kept in the run's `GrpoStats`); DPO margin `rl/tests/dpo_objective.rs::the_logged_margin_is_the_margin_an_independent_forward_measures` (each conversation scored alone by a separately built model, normalised in f64, rows of unequal length); distillation KL `rl/tests/distill_normalisation.rs`; external policy gradient KL, entropy, length and gradient norm `rl/tests/it/external_policy_gradient.rs::the_kl_term_is_the_k3_estimator_against_the_reference_it_names`, `policy_gradient_stability.rs::the_step_reports_the_entropy_estimate_the_response_length_and_the_gradient_norm`, clip fraction `rl/src/objective/pg_batch.rs`; numerics gate `rl/src/objective/numerics.rs`) P1 - verification gap - method metrics
- [x] T030 (fold equals live adapter `qwen3/tests/it/lora_adapter_file.rs`, `qwen35/tests/it/lora_serve_fold.rs`; variants `qwen3/tests/it/adapter_variants.rs`, `qwen35/tests/it/adapter_variants.rs`; PEFT export `qwen3/tests/it/peft_export.rs` and against transformers `minicpmv/tests/it/peft_export.rs`; save and reload C6 in the conformance suite; the saved adapter and the saved full fine-tune served through each family's engine at the trainer's loss `rl/tests/it/served_conformance.rs::qwen3_serves_the_artifacts_it_trains` and the qwen35 and qwen35moe siblings) P1 - verification gap - adapter merge, export and reload parity
- [ ] T031 (`qwen35/tests/it/vision_train.rs::the_training_forward_has_the_logits_serving_produces_for_the_same_prompt`, `qwen35/tests/it/vl.rs::mrope_positions_match_get_rope_index`, `gradcheck/tests/it/qwen35_image_record_fd.rs`, `data/tests/it/chat_image_parts.rs`, real weights `qwen35/tests/vision_train_real.rs` against transformers. Open: no test asserts that the image-placeholder positions carry no target through the dataset path into the vision trainer; the fixtures build those targets by hand) P1 - verification gap - multimodal alignment
- [x] T032 (flow targets from the formula: `wan/src/modelgrad.rs`, `ltxv/src/modelgrad.rs`, `ltxv/src/av_modelgrad.rs`, `s3dit/src/finetune.rs`, `minimaxh3/src/grad.rs`; f64 finite differences of the velocity loss: `flux2/tests/it/model_grad.rs::full_model_gradcheck`, `ltxv`, `s3dit` and `wan` block gradchecks, `gradcheck/tests/it/adapter_kinds_fd.rs`; `flux2::modelgrad::make_flow_batch` and its paired form in FP64: `flux2/tests/it/flow_batch_convention.rs::a_flow_batch_is_the_rectified_flow_formula_in_fp64` (`x_σ = (1 − σ)·x₀ + σ·ε`, `v = ε − x₀`, `t = σ`, references clean, the joint layout's RoPE tables) and `::the_velocity_target_is_the_step_the_euler_sampler_takes` (the target is `d x_σ / dσ`, so the sampler's Euler step from σ lands on the batch at σ', from pure noise at σ = 1 to `x₀`); the schedule: `flux2/tests/it/training_matches_deployment.rs`) P1 - verification gap - flow and diffusion oracles
- [ ] T033 (`crates/rlcd`: policy gradient and scoring against finite differences, the proper reward against the published reference values, ECE, NLL and AUROC closed forms. Open: no end-to-end run is held to a calibration bound on held-out data) P1 - verification gap - RLCD calibration
- [ ] T034 (every verifier is programmatic and unit-tested in `rl/src/verify.rs`; the promotion gate scores a candidate on held-out tasks, `rl/tests/improve_cycle.rs`. Open: there is no learned verifier whose generalisation could be measured; it depends on T042 and T043) P1 - verification gap - verifier generalisation
- [ ] T035 P2 - feature - full-vocab distillation API
- [ ] T036 P2 - feature - top-K tail mass
- [ ] T037 P2 - feature - online DPO
- [ ] T038 P2 - feature - IPO/KTO/ORPO/SimPO
- [ ] T039 P2 - feature - DAPO and Dr.GRPO presets
- [ ] T040 P2 - feature - GSPO
- [ ] T041 P2 - feature - actor-critic PPO
- [ ] T042 P2 - feature - PRM/critic
- [ ] T043 P2 - feature - learned reward models
- [ ] T044 P2 - feature - leave-one-out baselines
- [ ] T045 P2 - feature - KL/entropy control
- [ ] T046 P2 - feature - RL replay with corrections
- [ ] T047 P2 - feature - continual-learning regularisers
- [ ] T048 P2 - feature - failure supervision
- [ ] T049 P2 - feature - objective capability registry
- [ ] T050 P2 - feature - exposure of GRPO/top-K distill to downstream callers
- [x] T051 P2 - feature - qwen3 external policy gradient (done in 95c1a48d4)
- [ ] T052 P2 - feature - quantized training gaps
- [ ] T053 P2 - feature - batched KV-cached GRPO rollouts
- [ ] T054 P2 - feature - multi-card served training
- [ ] T055 P2 - feature - training certification manifest
- [ ] T056 P2 - feature - independent FP64 oracle suite
- [ ] T057 P2 - feature - data/config fuzzing
- [ ] T058 P2 - feature - cross-backend determinism policy
- [ ] T059 P2 - feature - capability regression benchmarks
- [ ] T060 P2 - feature - architecture training certification matrix

## Conformance suite plan (target)

- [x] A. FP64 scalar loss oracle plus finite-difference / second-autograd gradient checks at extreme inputs.
      LM head: `model/tests/it/head_ce.rs::extreme_logits_match_the_host_reference_in_both_tails` (logits in the
      hundreds, both tails) beside its normal-range f64 and finite-difference tests; DPO:
      `rl/src/objective/dpo.rs::pair_term_matches_the_fp64_oracle_at_extreme_margins`; GRPO:
      `rl/src/objective/grpo_term.rs::the_term_matches_the_fp64_reference_at_normal_extreme_and_boundary_inputs`;
      distillation: `rl/tests/distill_normalisation.rs`; whole models: `crates/gradcheck`, `rl/tests/dpo_gradcheck.rs`,
      `rl/tests/grpo_gradcheck.rs`, `rl/tests/it/external_policy_gradient_gradcheck*.rs`.
- [x] B. Reference AdamW step (moments, bias correction, decoupled weight decay, clipping, accumulation) through every
      precision, streaming, offload and multi-GPU mode, with token-weighted global normalisation.
      Step: `qwen3/tests/it/adamw_hyperparameters.rs::betas_and_eps_are_honoured` (two steps against a host AdamW with
      decoupled decay), `optim/src` `adamw_with_clip_matches_hand_computation` and
      `f32_streamed_step_equals_reference_adamw_for_any_tile`, `optim/tests/it/offload_parity.rs::scale_and_clip_follow_torch_on_device_and_host`;
      state precision: `optim/tests/it/adamw_bf16_parity.rs`, `adamw_q8_parity.rs`; streaming: the
      `*_streams_its_layers_through_a_one_layer_window_and_resumes_exactly` family and
      `qwen3/tests/it/layer_streamed_full_ft.rs::a_layer_streamed_run_ends_exactly_where_the_resident_run_does`;
      multi-GPU: `qwen3/tests/it/shard_microbatch.rs::microbatch_pipeline_grad_parity` and `dp_parity.rs` (two cards);
      normalisation: T025.
- [x] C. Exact resume: data order, objective cursors, weights, moments, schedule, RNG, reference identity, artifact hashes.
      Bit-identical weights, moments, adapters and digests after a cancel and resume for SFT, DPO, distillation,
      GRPO, qwen35 and the external policy gradient (T026); a changed option, rendering or base format is refused (`sdk/tests/it/chat_finetune.rs::a_cancelled_run_resumes_to_the_uninterrupted_result`);
      the reference is the run's start (`rl/tests/it/lazy_scoring.rs::a_resumed_run_scores_under_the_adapter_it_started_from_not_the_one_it_restored`);
      moments and tiles `optim/tests/it/streamed_state.rs::resume_from_checkpoint_is_bit_identical_to_an_uninterrupted_run`.
- [x] D. Reload the saved artifact from disk and evaluate it through the serving API.
      `rl/tests/it/served_conformance.rs::qwen3_serves_the_artifacts_it_trains` and the qwen35 and qwen35moe siblings:
      the saved adapter and the saved full fine-tune, read back into each family's serving engine, score the
      trained records at the loss the trainer measured; `sdk/tests/it/chat_finetune.rs::a_fine_tune_exports_an_adapter_that_scores_and_serves`
      serves the adapter by its digest through `ChatPipeline`.

## Findings

- Policy-gradient update scale (2026-11-08, served `brain/train` on a 4B model: no change at lr 1e-6, divergence at 2e-5).
  Checked against an FP64 reference on a tiny CPU model with unequal record lengths and both advantage signs
  (`crates/rl/tests/it/external_policy_gradient_scale.rs`): the step's gradient is the documented aggregation (sequence
  mean: mean over records of each record's token mean; token sum: all tokens over the step's token count) to 0.5%;
  advantages are not renormalised in brain; negative advantages are not clipped on the upper side; the k3 KL is a
  per-token mean. The shared loop gives AdamW the accumulated gradient divided by `grad_accum`. AdamW normalises the
  gradient, so a step moves each adapter element by about the learning rate whatever the advantage scale, the
  aggregation or the KL coefficient (a 1000x advantage gives the same step), and the policy change per step is linear
  in the learning rate. No objective defect found. The one silent factor was the number of records read: a run draws
  `steps * grad_accum * batch` records, 100 by default, so 2400 rollouts at default steps train on 4% of them; the run
  now says so when it opens.
- [x] T061 P0 - crates/rl/src/objective/numerics.rs, external.rs - serve-versus-train numerics gate before an RL arm (step-0 ratio, gap, mismatch KL, ESS, gradient norms; bounds by flag; non-zero exit); `grpo-corrections.md`, knowledge #446
- [x] T062 P0 - crates/rl/src/objective/external.rs (rule) - per-step guard on KL and gradient norm that skips and logs an update (knowledge #440 is the evidence); entropy estimate, response length, clip fraction per edge per step; clip-higher; truncated importance sampling and mask; truncated rollouts and observation tokens masked
- [x] T063 P1 - crates/sdk/src/finetune.rs, crates/backend-cuda/src/backend.rs - a device out-of-memory in a chat fine-tune is `Error::DeviceOutOfMemory` (and `[unavailable]` through `brain/train`), not an unwind; a graph launch that runs out of memory after its first graph unwinds with the allocator's typed payload so the scheduler backs off (knowledge #445)
