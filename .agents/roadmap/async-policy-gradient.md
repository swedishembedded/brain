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

## Findings that shape the order

- A truncated sampler makes the policy trace the wrong behaviour denominator: the trace is the token's whole-row
  log-probability at the temperature (knowledge 520, 528), while agent runs sample with top_k/top_p cuts. The real
  behaviour probability is `p / mass(support)`, so the clipped surrogate on such records starts from a ratio of
  `mass < 1`, not 1. Online runs therefore sample untruncated (top_k 0, top_p 1, min_p 0) until the trainer's
  probabilities can be renormalised over the sampler's support (support replay, below); the score-centred estimator
  refuses truncated records by name.
- Leases mostly exist: a reply's adapter identity is captured at admission, a replaced adapter drains, each release has
  its own KV namespace. The gap is the route refresh under load.
- The in-process `Grpo` stays the synchronous reference; asynchrony serves externally produced experience first.

The cross-repository plan, its package numbering (A01-A53) and its review are kept by the orchestrating application
that drives this API; brain's packages are A03-A10, A16-A21, A25-A26, A28-A29, A32-A33, A37-A49 and are the items below.

## Open

- [ ] **One policy-gradient configuration** (A03-A04; `engineering-cleanup.md` EC-1, EC-2): settings declared once
      for objective, SDK, CLI and the served job; `Grpo` and the external objective share one loss config (dual clip,
      step guards included); the estimator (`ClippedSurrogate | ScoreCentred`) is a variant of the shared `pg_batch`
      path, independent of the experience source and scheduling.
- [ ] **Model-free oracles** (A05-A06): a K-action categorical policy on host parameters with fixed rewards, every
      estimator's sampled gradient against its own exact enumerated expectation (complete groups for group-standardised
      advantages); repartitioning the same tokens across micro-batches leaves the step's gradient unchanged; golden
      fixtures from the pinned TRL code for advantages, masks and centring coefficients (an isolated tool, CPU only).
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
- [x] **The sampler's top-K in `policy_trace`** (`policy_trace: {"top_log_probs": K}`, K in 1..=20; params `policy_trace` +
      `policy_trace_top_log_probs` over D-Bus): per generated position (stop token included) `generation_top_ids` /
      `generation_top_log_probs`, the K most probable ids and their whole-row log-probabilities at the sampling
      temperature, from `model::logprobs::scored_position` (one row, one arithmetic: the sampled token's
      `generation_log_probs` entry is its top entry bit for bit). Shared by every decoder that scores continuations
      (`PagedDecoder::score_continuation_top`: qwen3, qwen35 incl. int8/fp8 residents, ouro; conformance-checked; a decoder
      that cannot refuses by name before decoding). Tests: `llmserve/tests/it/outcome_report.rs`,
      `qwen3/tests/it/engine_score_continuation.rs`, `catalog/tests/it/scheduled_batch.rs` (decode trace == score trace
      within 2e-3, entries == trainer), `apiserve/tests/it/api.rs` (shape, bounds, names). API audit: one integer param
      bounded by the action's spec (1..=20, refused by name), an object with no other key, no path or secret; the reply
      grows by at most 20 ids and 20 figures per generated token, within the existing `max_new`/body limits.
      Decision: a request with a top list takes its figures from the scoring pass (`log_prob_source: "score"`), not the
      decode steps: the device readout only keeps a candidate list for truncated draws, so a whole-row top-K from the
      decode would need a device top-k over every row. Open (profile first): serve the top list from the decode steps for
      whole-row draws, which removes the extra prefill (knowledge #528). This gates the score-centred estimator on a card:
      with a top list the sampled token's own figure is the prefill's too, and on a card a prefill rounds differently from
      the decode that drew the token (knowledge #520), so the behaviour invariant holds only up to that gap. The
      clipped-surrogate path keeps the decode-sourced trace.
- [x] **Adapter routes refresh under load**: an open group refreshed its adapter routes only when its lane went idle
      (`OpenGroup::Refresh`), so a lane that never drained kept routing a re-published id to the old version. Verified
      first (the new test failed: the first request after the swap named version 1). Now `run_open_scheduled` refreshes the
      bank and routes (`AdapterSync::sync`, `&self`) before it admits a request while the frontend is stale: the bound is
      "every request admitted after `LiveAdapters::serve` returned runs on the new version", with no wait for a drain.
      Running sequences keep their weights (a replaced release drains in the bank; a job's identity is captured at
      admission; KV namespaces stay per digest). `OpenGroup` and the idle hand-back are gone. Test:
      `catalog/tests/it/multi_adapter.rs` (three back-to-back request streams on the tiny qwen3 resident, probe after the
      swap, every reply's text equals its named release's folded server), `scheduled_batch.rs`. Not exercised: a bank with
      fewer device slots than live releases (the new version may wait for a slot while the old one drains).
- [ ] **Streaming policy-gradient job** (`crates/trainjob`): `brain/train` with `stream` instead of `data`; records of
      complete groups are appended to the running job (`POST /v1/jobs/{id}/records`, D-Bus equivalent), deduplicated
      by group key, held in a token-bounded queue (a full queue is a 429 naming the allowance); a step trains when its
      token budget is filled; every `publish_every` steps the adapter is published to the serving bank under a fixed
      id as the next version; records older than `max_staleness` versions are dropped and counted; `finish` drains
      and ends. Events: `step`, `published {version, digest, step}`, `consumed {groups, tokens, dropped_stale}`, queue
      depth. Resume: the job reports the group keys it consumed; a client re-sends the rest.
      Receipts separate applied (the group's gradient is in an update) from committed (that update is in a durable
      checkpoint); a client re-sends every group not committed, and the job deduplicates by key.
      Design notes from the paused lane: a generic job inbox beside the cancel token (a worker-process job refuses it by
      name); a streaming objective that lays out a whole step before its first backward; a publisher seam implemented
      in-process by the serving binary; `finish` ends a run as completed, not cancelled.
- [ ] **Run-level policy identity** (A16): training run, committed update, base and adapter digests beside the serving
      version, in every reply and every `published` event.
- [ ] Per-step telemetry for async runs: staleness mean/max, trained tokens per step, time the learner waited for
      records, records' time in queue.
- [ ] End-to-end on Qwen3-4B (one card): serving and the streaming job in one process, rollouts from an orchestrator;
      profile (nsys) before optimising anything.
- [ ] **Support replay for truncated samplers**: the sampler's support per position in the trace, the trainer's
      log-probabilities renormalised over it, so cut samplers can train without bias.
- [ ] Later, each driven by a profile and recorded as a knowledge entry (bottleneck, baseline, evidence, change,
      oracle, result, decision): colocated serve-and-train scheduling and whether the frozen base can be shared
      (A37-A39); merging append-only calls of one session and group-prefix packing in training, against the per-call
      reference (A40-A43); head and coefficient cost (A44-A45); multi-actor publication (A46-A48).
