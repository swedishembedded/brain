<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# generic-learning

Scope: brain owns a thinking budget that always terminates, scoring of given tokens under a given context for any model or adapter, and KL-to-champion on a rehearsal set in SFT and PG: the model half of the generic learning loop. Items GL-R1 to GL-R3 and the brain-side stop list.

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
| GL-R1 | A thinking budget that always closes: the thinking block ends at the budget even when the model never emits `</think>`, and the reasoning count is enforced, not only observed | gap (handover: the budget and the count still end only at `</think>`) | a spec: a model that never closes its block is cut at the budget and answers; same behaviour on every architecture, no per-model flag | none |
| GL-R2 | A scoring call: log-probabilities of given tokens under a given context for any model or adapter | not confirmed to exist as a general endpoint | one API call scores a token sequence under a context for base and adapter; used for the importance-ratio diagnostic and later logit distillation | none |
| GL-R3 | KL-to-champion on a rehearsal set in SFT and PG, as a shared-path term with backward and gradcheck | partial: recorded behaviour, reference pass and PG exist | the term is in the shared training path, gradchecked, and reported with training tokens per second | none |

Open question for later on-policy logit distillation: whether the 27B and the 4B share a tokenizer; unconfirmed. Such distillation must be weighted by verified outcomes (the teacher fails 29% of the time).

## Stop doing (brain)

- Length control through template switches or per-model flags; control it by the budget (GL-R1).
