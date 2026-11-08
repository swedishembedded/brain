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
- [x] T008 P0 - crates/model/src/train.rs (fit_controlled) - transactional update gate in the shared loop; strict mode; NaN injection test (shared gate landed as `model::stepgate`; `FitOpts::strict_updates` is not yet exposed as a CLI flag; GRPO and external policy gradient keep their own verdict counters)
- [ ] T009 P1 - crates/rl/src/objective/distill.rs - one normalisation convention; dense-softmax mean-KL oracle, unequal lengths
- [ ] T010 P1 - crates/rl/src/objective/dpo.rs (pack_row) - streaming DPO refuses truncation
- [ ] T011 P1 - crates/rl/src/objective/distill.rs (pack_row, TopK::from_logits) - refuse truncation and zero-information batches; validated teacher distribution
- [ ] T012 P1 - crates/model/src/train.rs (schedule_fingerprint) - identity covers Adam beta1/beta2/eps; resume refuses a change
- [ ] T013 P1 - crates/model/src/train.rs (schedule_fingerprint) - identity covers GEMM compute precision and the whole resolved run configuration
- [ ] T014 P1 - crates/model/src/train.rs (build_or_resume) - separate exact resume, weight warm start, adapter continuation (names and lineage)
- [x] T015 P1 - crates/rl/src/objective/grpo.rs (attach_references) - incomplete references refuse; unsupported EOS positions distinguished
- [ ] T016 P1 - crates/rl/src/objective/external.rs - explicit exact vs proxy behaviour logprobs with provenance
- [ ] T017 P1 - crates/rl/src/objective/external.rs - missing rollout sidecars never select imitation semantics silently
- [ ] T018 P1 - crates/model/src/train.rs (loss_probe) - pure probe; persistent objective cursor; resume equivalence test
- [ ] T019 P1 - central ValidatedTrainingPlan - every numeric configuration refused at every entry point
- [ ] T020 P1 - crates/rl/src/objective/distill.rs - duplicate/out-of-range ids, non-finite or unnormalised teacher probabilities (with T011)
- [ ] T021 P1 - DPO reference checkpoint - reference digest in run identity
- [ ] T022 P1 - structured datasets - strict schema mode
- [ ] T023 P1 - per architecture - expected-vs-actual trainable tensor audit (full FT / LoRA target coverage)
- [ ] T024 P1 - verification gap - cross-precision/backend parity, distributed token normalisation, objective-cursor resume, EOS/truncation/overlong reward contracts, held-out reuse, method metrics, adapter merge/export/reload parity, multimodal alignment, flow/diffusion oracles, RLCD calibration, verifier generalisation (T024-T034 share one row each; split as they land)
- [ ] T025 P1 - verification gap - cross-precision/backend parity, distributed token normalisation, objective-cursor resume, EOS/truncation/overlong reward contracts, held-out reuse, method metrics, adapter merge/export/reload parity, multimodal alignment, flow/diffusion oracles, RLCD calibration, verifier generalisation (T024-T034 share one row each; split as they land)
- [ ] T026 P1 - verification gap - cross-precision/backend parity, distributed token normalisation, objective-cursor resume, EOS/truncation/overlong reward contracts, held-out reuse, method metrics, adapter merge/export/reload parity, multimodal alignment, flow/diffusion oracles, RLCD calibration, verifier generalisation (T024-T034 share one row each; split as they land)
- [ ] T027 P1 - verification gap - cross-precision/backend parity, distributed token normalisation, objective-cursor resume, EOS/truncation/overlong reward contracts, held-out reuse, method metrics, adapter merge/export/reload parity, multimodal alignment, flow/diffusion oracles, RLCD calibration, verifier generalisation (T024-T034 share one row each; split as they land)
- [ ] T028 P1 - verification gap - cross-precision/backend parity, distributed token normalisation, objective-cursor resume, EOS/truncation/overlong reward contracts, held-out reuse, method metrics, adapter merge/export/reload parity, multimodal alignment, flow/diffusion oracles, RLCD calibration, verifier generalisation (T024-T034 share one row each; split as they land)
- [ ] T029 P1 - verification gap - cross-precision/backend parity, distributed token normalisation, objective-cursor resume, EOS/truncation/overlong reward contracts, held-out reuse, method metrics, adapter merge/export/reload parity, multimodal alignment, flow/diffusion oracles, RLCD calibration, verifier generalisation (T024-T034 share one row each; split as they land)
- [ ] T030 P1 - verification gap - cross-precision/backend parity, distributed token normalisation, objective-cursor resume, EOS/truncation/overlong reward contracts, held-out reuse, method metrics, adapter merge/export/reload parity, multimodal alignment, flow/diffusion oracles, RLCD calibration, verifier generalisation (T024-T034 share one row each; split as they land)
- [ ] T031 P1 - verification gap - cross-precision/backend parity, distributed token normalisation, objective-cursor resume, EOS/truncation/overlong reward contracts, held-out reuse, method metrics, adapter merge/export/reload parity, multimodal alignment, flow/diffusion oracles, RLCD calibration, verifier generalisation (T024-T034 share one row each; split as they land)
- [ ] T032 P1 - verification gap - cross-precision/backend parity, distributed token normalisation, objective-cursor resume, EOS/truncation/overlong reward contracts, held-out reuse, method metrics, adapter merge/export/reload parity, multimodal alignment, flow/diffusion oracles, RLCD calibration, verifier generalisation (T024-T034 share one row each; split as they land)
- [ ] T033 P1 - verification gap - cross-precision/backend parity, distributed token normalisation, objective-cursor resume, EOS/truncation/overlong reward contracts, held-out reuse, method metrics, adapter merge/export/reload parity, multimodal alignment, flow/diffusion oracles, RLCD calibration, verifier generalisation (T024-T034 share one row each; split as they land)
- [ ] T034 P1 - verification gap - cross-precision/backend parity, distributed token normalisation, objective-cursor resume, EOS/truncation/overlong reward contracts, held-out reuse, method metrics, adapter merge/export/reload parity, multimodal alignment, flow/diffusion oracles, RLCD calibration, verifier generalisation (T024-T034 share one row each; split as they land)
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

- [ ] A. FP64 scalar loss oracle plus finite-difference / second-autograd gradient checks at extreme inputs.
- [ ] B. Reference AdamW step (moments, bias correction, decoupled weight decay, clipping, accumulation) through every
      precision, streaming, offload and multi-GPU mode, with token-weighted global normalisation.
- [ ] C. Exact resume: data order, objective cursors, weights, moments, schedule, RNG, reference identity, artifact hashes.
- [ ] D. Reload the saved artifact from disk and evaluate it through the serving API.
