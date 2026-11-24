<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Asynchronous policy gradient - roadmap

Scope: Online policy-gradient training while the same process serves the policy: the sampler's top-K and the score-centred objective, a streaming training job that consumes rollout groups as an orchestrator produces them and publishes each new adapter version to the serving bank, bounded staleness and token backpressure, and the invariants that gate it.

Reference design: an actor-learner split in the style of TRL's `AsyncGRPOTrainer` (source pinned under
`~/resources/reference/trl`, not a dependency). brain owns the arithmetic and the serving side; the orchestrator that
schedules rollouts and decides which experience qualifies is a caller of the API, never named here.

Order of work (each step validated before the next; no step mixes two changes in one learning experiment):
mathematics first, then provenance, then the streaming job, then performance driven by profiles.

## Invariants (each one is a model-free test before any GPU run)

1. Behaviour: a record's behaviour log-probability is the distribution that drew the token (whole-row softmax at the
   sampling temperature; a truncated sampler - top-k, top-p, min-p - is refused for this objective until its support
   set is replayed).
2. Trajectory: every sampled token is trained once, against the exact context it was sampled in.
3. Normalisation: the gradient of one optimiser step is the sum over its policy tokens divided by the step's total
   policy-token count, however the tokens are split into micro-batches (and, once data-parallel, ranks).
4. Publication: every served reply names the policy version that decoded it, and a published version becomes routable
   under continuous load within a bounded time.
5. Score centring: with the sampler's full distribution as its top-K, the centred gradient equals the exact
   expectation `-A (f(r_a) grad log pi(a) - E_{v~mu}[f(r_v) grad log pi(v)])` on a tiny vocabulary.

## Open

- [ ] **Sparse extra targets in the LM-head loss** (`model::head_ce`, `model::training::TokenObjectives`): per
      position up to K `(id, weight)` pairs added to the weighted cross-entropy gradient
      (`d logits = W softmax - w e_y - sum_j c_j e_{v_j}`, `W = w + sum_j c_j`) and the trainer's log-probability at
      those ids read back after the forward. One shared implementation for every model on `HeadCe`; a model without it
      is refused by name.
- [ ] **Score-centred policy-gradient objective** (`rl::objective`): TRL's estimator (masked importance weight
      `f(r) = r 1[low <= r <= high]`, centring over the sampler's top-K plus a tail bucket), a separate algorithm next
      to the clipped surrogate, sharing `pg_batch`. Analytic tests on a tiny vocabulary against the exact expectation,
      a finite-difference gradcheck, the sampler-mass and masked-fraction telemetry.
- [ ] **Rollout record fields** (`data::chat_rollout`, contract v2, additive): `behavior_top_ids` /
      `behavior_top_logprobs` (K per sampled token) and `policy_version` (oldest version among the requests that
      produced the record).
- [ ] **The sampler's top-K in `policy_trace`** (`policy_trace: {top_log_probs: K}`): per generated position the K most
      probable ids and their whole-row log-probabilities at the sampling temperature, from the arithmetic that drew the
      token (the score fallback produces the same figures), shared by every architecture that scores continuations.
- [ ] **Adapter routes refresh under load**: a re-published adapter id is routed to its new version for new requests
      while older requests finish on the old one (today a route refreshes only when its lane is idle).
- [ ] **Streaming policy-gradient job** (`crates/trainjob`): `brain/train` with `stream` instead of `data`; records of
      complete groups are appended to the running job (`POST /v1/jobs/{id}/records`, D-Bus equivalent), deduplicated
      by group key, held in a token-bounded queue (a full queue is a 429 naming the allowance); a step trains when its
      token budget is filled; every `publish_every` steps the adapter is published to the serving bank under a fixed
      id as the next version; records older than `max_staleness` versions are dropped and counted; `finish` drains
      and ends. Events: `step`, `published {version, digest, step}`, `consumed {groups, tokens, dropped_stale}`, queue
      depth. Resume: the job reports the group keys it consumed; a client re-sends the rest.
- [ ] Per-step telemetry for async runs: staleness mean/max, trained tokens per step, time the learner waited for
      records, records' time in queue.
- [ ] End-to-end on Qwen3-4B (one card): serving and the streaming job in one process, rollouts from an orchestrator;
      profile (nsys) before optimising anything.
- [ ] Later, each driven by a profile: group-prefix packing in training, token-budget micro-batches, support-set replay
      for truncated samplers.
