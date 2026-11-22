<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# generic-learning

Scope: brain owns a thinking budget that always terminates, scoring of given tokens under a given context for any model or adapter, KL-to-champion on a rehearsal set in SFT and PG, and on-policy distillation from a teacher model: the model half of the generic learning loop. Items GL-R1 to GL-R4 and the brain-side stop list.

Source: the 2026-11-21 training-methods review (read-only advisory). The same review is recorded as `generic-learning.md` in the sibling repositories of the stack; the full plan E0-E5 is in the campaign driver's ledger.

## The generic loop and its invariants

The loop needs only the verifier, the tools, a token budget and state forks, and routes each task by its measured pass rate under the current policy, never by task content. Every repo holds the same four rules:

1. **Same observation.** A token is a training target only if it was sampled in a context the student will also see. Privileged sources (a teacher, a hinted self, an oracle solution) supply states, not text.
2. **Reasoning length is a budget.** A trial passes only if the verifier passes within `B` completion tokens and `N` turns. Native reasoning mode, one mode and one budget for teacher capture, training data and every exam arm; length is controlled by the budget and by lowest-cost selection among verified passes, never by prompt text.
3. **Evaluation that can see the effect.** At least 60 tasks over at least 20 families, a reference arm between about 0.15 and 0.85, three tiers (unseen instance, held-out variant, held-out family; only the third supports a transfer claim), wall clock an infrastructure backstop.
4. **No string matching.** No filter reads the prose of a trace or the content of a task.

## Capabilities

| Id | Capability | Status | Acceptance | Depends on |
|---|---|---|---|---|
| GL-R1 | A thinking budget that always closes: the thinking block ends at the budget even when the model never emits `</think>`, and the reasoning count is enforced, not only observed | done (#540): `thinking_budget` is per turn over every block; a reopened block is closed at once; a call written inside the block is never counted or cut, and running out after it ends the turn; `reasoning_tokens` counts exactly the tokens the parse reads as reasoning | a spec: a model that never closes its block is cut at the budget and answers; same behaviour on every architecture, no per-model flag (the scheduler enforces it for every model whose frontend names its tags; specs in #540) | none |
| GL-R2 | A scoring call: log-probabilities of given tokens under a given context for any model or adapter | done: the `score` action (`POST /v1/run`, `model` the base or adapter id) takes token records `{"prompt_token_ids", "token_ids"}` scored as given, at a `temperature` (default 1), beside chat records; every scheduler-backed chat model declares it (`docs/reference/api/http.md`) | one API call scores a token sequence under a context for base and adapter; used for the importance-ratio diagnostic and later logit distillation | none |
| GL-R3 | KL-to-champion on a rehearsal set in SFT and PG, as a shared-path term with backward and gradcheck | partial: recorded behaviour, reference pass and PG exist | the term is in the shared training path, gradchecked, and reported with training tokens per second | none |
| GL-R4 | On-policy distillation from a teacher model: the student samples, another model scores the samples, the student is trained toward the reverse KL on its own samples, the estimator chosen by comparing the two tokenizers | done: `data::tokenizer_identity::TokenizerIdentity::check` decides; `teacher_logprobs` (token-level `log q - log p_old` per token, identical tokenizers only, refused otherwise with the typed `DistillRefusal`) and `teacher_logprob` (sequence-level `log q(text) - log p(text)` with a leave-one-out baseline, any teacher) on rollout records, trained by the shared policy gradient (`rl::objective::onpolicy_distill`, `rl::objective::external`); `ChatFineTune::distill_teacher`, train job `distill_teacher`. Qwen3-4B and Qwen3.8-27B differ (#554), so 4B from 27B takes the sequence-level path. Not run on real weights yet; the orchestrator's sampling and `score` calls are outside brain | gradcheck of the token-level term on a tiny model, zero term and gradient for teacher == student, the sequence-level estimator equal to the exact reverse-KL gradient on an enumerable toy with the baseline lowering its variance, refusal by type (`rl/tests/it/onpolicy_distill*.rs`, `rl/src/objective/onpolicy_distill.rs`) | GL-R2 |

The 27B and the 4B do not share a tokenizer (#554), so on-policy distillation between them is sequence-level (GL-R4). It is to be weighted by verified outcomes (the teacher fails 29% of the time): the record's `reward` advantage is added to the distillation advantage.

## Stop doing (brain)

- Length control through template switches or per-model flags; control it by the budget (GL-R1).
