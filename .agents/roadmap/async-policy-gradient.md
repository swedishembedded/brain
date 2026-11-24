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

## Done

- [x] **Sparse extra targets in the LM-head loss** (`model::head_ce::SparseTargets`, `TokenObjectives::
      enable_sparse_targets` / `set_sparse_target_ids` / `set_sparse_target_weights` / `batch_sparse_target_logprobs`,
      `Capability::SparseTargets`): per position up to K `(id, weight)` pairs in one backward pass. `ce_sparse_targets`
      (WGSL, one thread per logit, a K-term scan, no atomics) writes `W softmax - w e_y - sum_j c_j e_{v_j}` from the
      plain CE gradient before the soft cap and the scales, so the extra terms pass through both; `ce_gather_logprob`
      reads `log p(v_j) = z_v - z_y - nll` from the chunk's logits without a second pass over the row. Implemented on
      every `HeadCe` trainer (qwen3, qwen35, qwen35moe, inkling2, ricursive, `PipelineModel` delegates); gpt2 (no
      chunked head) refuses by the trait default. No CUDA candidate: the CE elementwise kernels have none and no gate
      requires one. Tests: `model` `it::head_ce_sparse` (f64 oracle over 150 tokens / 64-row chunks / 250-row vocab
      windows, F32 and BF16, soft cap, duplicate ids, label ids, unused slots, ignored rows; all-unused equals the
      weighted path to 1e-6; refusals), and conformance C3b `sparse_targets_match_their_label_equivalents` on every
      registered model (one backward with extra targets equals the sum of label-equivalent weighted backwards to 1e-4
      of the largest gradient; read-back log-probabilities equal the label path's to 1e-5; verified to fail with the
      qwen35 backward wiring removed, where a finite-difference form at the conformance tolerances did not).
- [x] **Rollout record fields** (`data::chat_rollout`, contract v2, additive): `behavior_top_ids` /
      `behavior_top_logprobs` (K per sampled token beside `behavior_logprobs`, one K per record and per prepared split,
      distinct ids, whole-row log-probabilities of mass at most one; rules `behavior_top_count`, `behavior_top`),
      kept as `<split>.btk.bin` / `<split>.btl.bin` at the positions of `<split>.blp.bin` (`read_behavior_top`);
      `policy_version` (oldest version among the record's requests) and `sampler` (`SamplerDecl`, `truncates()`) per
      example. Ids are checked against the vocabulary by the trainer, which knows it. Tests: `data`
      `it::chat_rollout_records` (positions, packing, a 14-case refusal table, mixed K, provenance) and the v2
      fixtures.

- [x] **Streaming policy-gradient job** (lane agrpo-stream, 2026-11-24; `crates/trainjob/src/stream.rs`,
      `rl::objective::stream_pg`, `brain::RolloutStream`, `capability::inbox`, `trainjob::bank`): `brain/train` with
      `stream` (a JSON document: `publish_adapter`, `publish_every`, `step_tokens`, `queue_tokens`, `max_staleness`?,
      `max_steps`?, `top_k`?, `start_adapter`?) instead of `data`, exactly one of them. Records reach the running job as job input
      (`POST /v1/jobs/{id}/records[?epoch=N]` / `finish`, D-Bus `Input(job, kind, query, fd)`), checked as a dataset
      is (`Asked::samples`, `check_rollouts_together` with the base's tokenizer digest), each group's advantages
      computed on arrival by the batch path's own `external::advantages` over `SplitRollouts::of_split`; a step takes
      whole groups in arrival order until it holds `step_tokens` policy tokens and divides by the step's own count
      (token-sum) or sequences x constant (constant-length) however many micro-batches hold it; every `publish_every`
      steps the adapter is offered in process to the serving bank (`ServingBank`: `verify_adapter_among` +
      `LiveAdapters::serve`) and `published {version, digest, step}` follows once the registry lists it; groups more
      than `max_staleness` versions behind are dropped and reported. Receipts: `applied {step, groups[, skipped]}`,
      `committed {step, groups}` (after a training-state write, a stop, the job's completion), `dropped`, and a
      `stream` frame with the run epoch. Full queue: 429 naming `allowance`, never while the queue holds less than one
      step. Tests: `trainjob` `it::stream` (lifecycle over HTTP incl. a reply from the published version and
      `/v1/models`, duplicate, stale drop, `410` after finish; the queue's allowance and no refusal below a step;
      one-POST vs many-POST first version byte-identical; a resumed run takes again exactly the applied-but-uncommitted
      group, `410` to the old id, `409` naming the new epoch; the score-centred estimator over a top-2; door refusals),
      `apiserve` `a_jobs_input_is_its_owners_alone_and_a_job_that_reads_none_says_so`, `it::dbus`
      `a_stream_job_over_the_bus_takes_its_records_by_input_and_finishes`, `model` `a_fed_objective_ends_the_run_complete_or_stops_it_resumable`,
      `rl` `stream_pg` unit tests, `capability` `inbox`.
      Decisions: (1) the job inbox is generic (`capability::inbox`, beside `CancelToken`): an `Inbox` on the
      `Invocation`, armed by the HTTP and D-Bus surfaces before the door check; a job declares at its door that it
      will read it (`will_read`), so records sent before it starts wait for it; a supervised worker's invocation
      carries none and a stream job is refused at the door without one. (2) `model::Objective::supply` (before each
      step: wait for data, `Finished` ends the run completed, `Stopped` interrupts it resumably) and `state_saved`
      (after a training-state write) are the loop's two new seams; `FitControl::publish` writes the adapter every N
      steps and hands it over before the step is reported. (3) The consumed keys ride in the objective's cursor, so they
      are durable atomically with the weights they trained (stronger than the run record); the run record keeps the
      epoch (`runs::Record::epoch`, +1 per job that continues the run). (4) The stream's settings are a `stream`
      document param like `data`; the estimator's settings stay the `pg_settings` table's params, and
      `check_streamable` refuses what the stream does not train: `behaviour start_of_run_proxy` (the versions that
      sampled are not the run's start), importance correction, the numerics gate, kv round trip, `seq-mean` (a step's
      sequence count varies), teacher contexts and teacher scores, images, `cards`/`job`, `grad_accum` > 1,
      evaluations, `adapter_id` (the card is `publish_adapter`). The score-centred estimator trains through the stream
      over the records' sampler top-K: its K is a stream setting (`top_k`), since the head's extra targets are sized
      before the first record arrives; a record of another K is refused, as is a truncating sampler
      (`external::check_centrable`, shared with the batch path). (5) Records sent before the learner is positioned wait
      (a resumed run must restore its consumed keys before it can call one a duplicate); an answer not given within two
      minutes is 503, never 429, so a full-queue 429 means exactly that. (6) A stream's warmup is at most 10 steps; an
      open-ended stream (no `max_steps`) keeps its peak rate.
      Open: a group whose
      oldest record has no `policy_version` is never stale (unmeasurable); the consumed-key list grows with the run
      (in the state's cursor); the published adapter is a snapshot card without base digest (it binds by base id);
      a `--serve-adapters` file of the same id would fight the stream for it (refuse an id the directory owns);
      `steps` is not read by a stream job (`max_steps` is).
- [x] Per-step telemetry for async runs (lane agrpo-stream): every streamed step frame's `pg` carries
      `staleness_mean`, `staleness_max` (versions behind the last published one), `policy_tokens`,
      `learner_wait_seconds`, `queue_groups`, `queue_tokens`, `time_in_queue_seconds` (the step's groups' mean),
      `dropped_stale`. Asserted in `trainjob` `it::stream`.

## Open

- [x] **One policy-gradient configuration** (A03-A04; `engineering-cleanup.md` EC-1, EC-2): settings declared once
      for objective, SDK, CLI and the served job; `Grpo` and the external objective share one loss config (dual clip,
      step guards included); the estimator (`ClippedSurrogate | ScoreCentred`) is a variant of the shared `pg_batch`
      path, independent of the experience source and scheduling. Done: EC-1 (`rl::objective::pg_settings`, one row per
      setting) and EC-2 (`rl::objective::pg_loss::PgLoss` in both configs); the `estimator` setting (wire `clipped` |
      `score-centred`, `PgLoss::estimator`) and its band (`centring_low` / `centring_high`, 0.5 / 5.0) are three
      `WhenNotDefault` rows, so the identity pins of runs at the default estimator stay green.
- [ ] **Model-free oracles** (A05-A06). Done: `it::pg_estimator_oracle::sampled_group_estimates_converge_to_their_exact_expectations`
      (four actions, groups of three with z-scored advantages, both estimators' means over 4000 sampled groups within
      5 standard errors of their exact expectations over all 64 groups; under a constant advantage the score-centred
      expectation is zero and the clipped surrogate's is not); invariant 3,
      `it::external_policy_gradient_scale::a_token_sum_steps_gradient_does_not_depend_on_how_its_records_are_split_into_micro_batches`
      (four records of 1-5 tokens as 4x1, 2x2, 1x4 micro-batches: one gradient to 1e-4 of the largest). Open: golden
      fixtures from the pinned TRL code for advantages, masks and centring coefficients (an isolated tool, CPU only).
- [x] **Score-centred policy-gradient objective** (`rl::objective`): the estimator is a variant of the shared
      path (`pg_batch::PgEstimator::{ClippedSurrogate, ScoreCentred(CentringBand)}`, the term in
      `score_centring::centred_term`: `f(r) = r 1[low <= r <= high]`, head over the sampler's top-K, tail bucket,
      weights `A f(r_a)` at the token and `-A residual_j` at the listed ids through `SparseTargets`), selected by
      `estimator: score-centred` (served param, `--estimator`, `PgLoss::estimator`). `ExternalPolicyGradient` reads the
      split's `.btk`/`.btl` into each batch (`Laid::top`), installs the ids after `set_batch` and before the forward,
      and `prepare` enables the model's extra targets with the records' K. Refused by name: records without a top-K
      (or a policy example with a supervised token without one), an example whose `sampler` truncates, an importance
      correction, behaviour other than `recorded`, a top-K id past the vocabulary (`check_data`), a model without
      extra targets (`Capability::SparseTargets`), and online GRPO (its rollouts record no top-K). Each step reports
      `sampler_mass` and `centred_masked_fraction` (`metrics()` `pg_sampler_mass`, `pg_centred_masked_fraction`, so
      the served step frame's `pg` map). Tests: `score_centring` unit tests, `it::pg_estimator_oracle`,
      `it::score_centred_gradcheck`, `it::score_centred_objective` (a step on the records' top-K with its mass, every
      refusal), `grpo_fixes::numeric_settings_no_run_can_honour_are_refused` (online GRPO), and the SDK's
      `it::chat_finetune_score_centred` (tiny qwen3 and qwen35 LoRA trained 10 steps end to end on the CPU: rewarded
      actions up, penalised down, finite stats, first step unmasked, the setting on the card; refusals through the
      SDK). Open: a GPU run waits for the decode-sourced top list (the note on the policy-trace item below).
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
      Decision (landed, superseded below): the first cut took every top list from the scoring pass.
      Served from the decode (2026-11-24, `agrpo-topk`): a request whose draw is over the whole row (sampler `top_k` 0,
      `top_p` 1, `min_p` 0, no processor or thinking budget, `temp` above 0) and whose decoder reads rows back gets
      the sampled token's figure and the top list from the decode step that drew the token: `log_prob_source: "decode"`,
      no scoring pass, so the figures are the draw's own arithmetic and not a prefill's (knowledge #520, #528). The
      scheduler asks per row (`SampleParams::trace_top` -> `RowDraw::top`) and every decoder answers through the generic
      readout: `RowDrawer` extracts the best `K` candidates of the whole batch in the step's one extraction (`topk_rows_steps`
      over all rows: one launch on CUDA, `3K` on the portable path, whatever the row count) and a row's list is
      `logit / temp - best - ln(mass)` from the statistics the draw reduced; the host fallback (`draw_rows_host`) uses
      `model::logprobs::scored_position`, so a decode list and a scored list are the same numbers for the same row.
      The drawn token's figure is computed with the list's own formula (bit-equal to its entry). Per step it costs
      the extraction for the batch and `rows x K` (id, logit) readbacks, only on steps with a row that asked (measured on
      the portable CPU path: 2 draw launches without a list, 2 + 3K with, for 3 and for 12 rows alike; on a card the figure
      is a profile's to give). A row that asked takes single steps (no keyed decode window), since a window reads back
      tokens only.
      What stays on the score path, by name: a draw that cuts the row (`top_k`, nucleus, min-p: a list under the
      survivors would be of a distribution the token was not drawn from), greedy, speculated and host-sampled rows
      (processor, thinking budget), a decoder without row readback or without the candidate extraction kernels (the model
      must register `READOUT_KERNELS` beside `DRAW_KERNELS`), and a list wider than the device keeps
      (`MAX_DRAW_CANDIDATES`). For those `log_prob_source` is `"score"`, and the request is still refused by name where
      the decoder cannot score a top list. Tests: `model/tests/it/{readout_draw,serve_keyed_draw}.rs`,
      `llmserve/tests/it/outcome_report.rs` (mixed batch, scoring passes counted), `catalog/tests/it/scheduled_batch.rs`
      (tiny qwen35 end to end vs the trainer), `serve::conformance::keyed_draws_answer_the_top_list_they_are_asked_for`
      on qwen3, qwen35 and ouro. For a card: measure the extraction launch per step (nsys), the readback bytes, and that
      the decode figures equal the scoring pass's within the gap of knowledge #520 on the production engines; no speedup
      is claimed here.
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
- [ ] **Run-level policy identity** (A16): training run, committed update, base and adapter digests beside the serving
      version, in every reply and every `published` event.
- [ ] End-to-end on Qwen3-4B (one card): serving and the streaming job in one process, rollouts from an orchestrator;
      profile (nsys) before optimising anything.
- [ ] **Support replay for truncated samplers**: the sampler's support per position in the trace, the trainer's
      log-probabilities renormalised over it, so cut samplers can train without bias.
- [ ] Later, each driven by a profile and recorded as a knowledge entry (bottleneck, baseline, evidence, change,
      oracle, result, decision): colocated serve-and-train scheduling and whether the frozen base can be shared
      (A37-A39); merging append-only calls of one session and group-prefix packing in training, against the per-call
      reference (A40-A43); head and coefficient cost (A44-A45); multi-actor publication (A46-A48).
