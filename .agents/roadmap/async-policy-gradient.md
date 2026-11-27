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
   sampling temperature; for a truncated sampler - top-k, top-p, min-p - the softmax over its support, which the record
   carries and the trainer replays: the trainer's figure is renormalised over the same support, so the ratio at the
   step that sampled is one).
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
  `mass < 1`, not 1. Support replay (below, lane agrpo-support) removes it: the trace reports each token's support and
  its figure within it, the records carry them, and the trainer renormalises over the support, so both estimators train
  cut samplers without bias (`rl` `it::support_replay`: ratio one at step 0 replayed, the support's mass without).
  Without a support the score-centred estimator still refuses truncated records by name, and the clipped surrogate
  trains them with a loud warning naming the bias (`sequences_cut_without_support`). Debt: that warning path stays
  until online runs record supports by default; then the clipped surrogate refuses a truncated record without one too.
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
      (in the state's cursor); a `--serve-adapters` file of the same id would fight the stream for it (refuse an id the directory owns);
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
- [x] **Model-free oracles** (A05-A06). Done: `it::pg_estimator_oracle::sampled_group_estimates_converge_to_their_exact_expectations`
      (four actions, groups of three with z-scored advantages, both estimators' means over 4000 sampled groups within
      5 standard errors of their exact expectations over all 64 groups; under a constant advantage the score-centred
      expectation is zero and the clipped surrogate's is not); invariant 3,
      `it::external_policy_gradient_scale::a_token_sum_steps_gradient_does_not_depend_on_how_its_records_are_split_into_micro_batches`
      (four records of 1-5 tokens as 4x1, 2x2, 1x4 micro-batches: one gradient to 1e-4 of the largest). TRL goldens
      (lane agrpo-conf, 2026-11-25): `tools/goldens/trl_score_centring_reference.py` runs TRL's own
      `AsyncGRPOTrainer.compute_loss` (score_centering, locals read at a stop after the per-token loss) and
      `AsyncRolloutLoop.compute_advantages` from the pinned tree (commit 6267b72, trainer SHA-256 in the fixture; torch
      2.14.1+cpu in the isolated venv `~/project/shared/venvs/trl-golden`), writing
      `crates/rl/tests/fixtures/trl_score_centring.json`; `it::trl_goldens` holds `centred_term` (mask, weight,
      `-A residual`, tail, per-token loss) to it within 2e-6 relative on six tokens, and the z-score advantages up to the
      one named difference (epsilon 1e-4 here, 1e-8 there, asserted as exactly that; NaN reward = a masked rollout; a
      group of one scored rollout refused here, trained at advantage 0 there). It found knowledge #583 (a whole listed
      sampler distribution left a rounding-noise tail; fixed). Exactly at a head edge the float32/float64 decisions can
      still differ by one ulp; the cases stay a hair inside and outside.
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
- [x] **Cross-architecture conformance of the asynchronous path** (A49, lane agrpo-conf, 2026-11-25):
      `rl::conformance_async::run_async` on every registered trainer (`rl/tests/conformance.rs`, `*_trains_online`),
      tiny random models on the CPU, held to `ASYNC_MATRIX`, which the capability gate holds the declaration's
      `online_pg`, `score_centring` and `policy_publication` cells to both ways. Per architecture: groups fed to a
      running stream train (four applied steps, the adapter moves); a run stopped after two steps and resumed trains
      every group exactly once (the committed ones are duplicates); the score-centred weights installed as one backward
      with extra targets equal the sum of label-equivalent weighted backwards (1e-4 of the largest gradient; zeroing the
      extra coefficients fails it on qwen3 and inkling2); every published version is read, bound and attached by the
      architecture's own serving bank family (`ServedFamily`: qwen3's, `qwen35::spec::served_family`,
      `ricursive::resident::family`, `inkling2::serve::served_family`, the last two moved out of the catalog so the
      resident and the suite share one). Matrix: qwen3, qwen35, ricursive, inkling2 hold all three; qwen35moe refuses
      all three (its `ChatTrainer` declares no policy gradient, asserted against the constant).
- [x] **Run-level policy identity** (A16, lane agrpo-prov): a published version's card records
      `TrainingProvenance::run` `{run, epoch, update}` beside its `base_digest` (`ChatFineTune::run_lineage`; the fit
      loop's `AdapterPublish::provenance` writes it on every published snapshot, the exported adapter carries it too);
      a server reads it from the card (`llmserve::adapter_release`, the one read) into `AdapterIdentity::lineage`
      (`residency::PolicyLineage {run, epoch, update, base_digest, adapter_digest}`), so the serving-local `version` and
      the immutable identity travel together through admission: `published {version, digest, step, lineage}`, a reply's
      `brain.adapter.lineage`, `policy_trace.policy.adapter.lineage` and the model card. The run is the `resume_key`,
      else `job:<32 hex>` drawn once (named in the `stream` frame); the update is the step, told apart across resumes
      by the epoch. One JSON form (`AdapterIdentity::to_json`/`from_json`, `deny_unknown_fields`) for every surface. A
      card naming a run without a base digest, a run id outside `[A-Za-z0-9._:-]{1,160}` or a malformed base digest is
      refused by name (the card is hostile input that every reply repeats). Tests: `trainjob` `it::stream` (every
      published frame's lineage is this run, epoch, step and file; `/v1/models`, a reply's `brain` and its policy trace
      name the last version's lineage; a resumed run publishes under its key at epoch 2; the one-POST/many-POST
      versions are compared tensor by tensor, since the cards now differ by run), `llmserve`
      `a_published_adapter_names_its_run_from_its_own_card` (bound, unbound and among bases; refusals),
      `a_request_keeps_the_lineage_of_the_version_it_was_admitted_under`, `residency`
      `an_identity_reads_back_from_its_json_and_a_foreign_field_is_refused`, `apiserve`
      `a_chat_reply_returns_a_policy_trace_only_when_asked` and the adapter card/reply tests. The `stream` frame also names the base's
      `tokenizer_digest` (`RolloutStream::tokenizer_digest`), so an orchestrator records the model and tokenizer a run
      trained without a probe. API audit
      (api-security.md): no new route or param; the new fields are read-only facts of a release (no path, bounded,
      validated at the card), behind the key like the rest of the reply; the run id exposes a `resume_key` to callers
      that may use the adapter, which opens nothing (a key resumes only the identical request, `runs::open`).
- [x] **Server build and record contract, for an orchestrator's revision manifest** (A01's brain side, lane
      agrpo-prov): `GET /v1/capabilities` carries `server {engine {version, commit?, dirty?}, backend?,
      training_records {contract, version, digest}?}` (`apiserve::ServerIdentity`, given by `brain serve`). The commit
      and dirty flag are embedded by the SDK's build script (`brain::build`); the record contract's digest is
      `data::chat_rollout::CONTRACT_DIGEST`, the digest of the v2 example set (`contract_digest`, held to the set by a
      test), which a producer compares with its copy's. Tests: `apiserve`
      `the_capability_list_names_the_server_build_and_its_record_contract`, `data`
      `the_reported_contract_digest_is_the_digest_of_the_example_set`, `brain` `build::tests`. Open: D-Bus has no
      method for it yet (`Version` stays the version string), and `tests/e2e/api_conformance.bats` does not assert
      it (not run in this lane: e2e suites take a card).
- [x] **Serving beside a streamed run on one device** (lane agrpo-e2e2, knowledge 584): a workload held its device
      lane for its whole run, so a stream job whose server serves its policy on the same device waited for ever for
      records whose requests queued behind it. `capability::lease::DeviceLease` on every workload job (armed by the
      executor), a serving and a workload lane per device, `Msg::Lend`/`Msg::Reclaim` in the dispatcher (a lent
      device takes serving; a reclaimed one starts nothing new and its open groups take no late jobs until it is
      back); `RolloutFeed` lends at every step boundary. Tests: `residency` `it::admission::a_workload_that_lends_*`,
      `capability` `lease` unit test, `trainjob` `it::stream` (replies from version 1 and 2 while the run goes on).
      Open: a GPU run (two threads now reach one device, one at a time); other training runs still hold their device
      between steps (they need not wait on their own server; lending in the fit loop would be the same call).
- [ ] **Run-level policy identity** (A16): training run, committed update, base and adapter digests beside the serving
      version, in every reply and every `published` event.
- [x] **The campaign's suites on CUDA** (lane agrpo-cuda, GH200, `BRAIN_BACKEND=cuda`): two failures that passed on the
      CPU were TF32 rounding read as defects, not wrong gradients (knowledge 587). `qwen35`/`qwen35moe` implement
      `Model::new_with_gemm`, so a run's `FitOpts::gemm_precision` reaches them; the extra-target and score-centring
      identity checks build with `model::conformance::build_exact` (the fp32 reference tier); the `rl` policy-gradient
      tests' model is built at that tier. Tests: `qwen35` `lora_gemm_precision::a_runs_gemm_precision_reaches_the_training_build`,
      `qwen35moe` `lora_freezes_base::a_runs_gemm_precision_reaches_the_training_build`; `rl` `--test conformance` 15/15
      and `--test it` 111/111 on CUDA. Open: `ricursive`, `ouro` and `inkling2` build at one fixed tier
      and ignore a run's `gemm_precision` (`Auto` there trains on the fp32 kernels); implementing `new_with_gemm`
      changes the arithmetic of their default runs on CUDA, so it lands with a step-parity check per model.
- [ ] End-to-end on Qwen3-4B (one card): serving and the streaming job in one process, rollouts from an orchestrator;
      profile (nsys) before optimising anything. **Partial (lane agrpo-real, 2026-11-25, brain 760c04d3b):** Qwen3-0.6B
      and Qwen3-4B trained online on a GH200 through an orchestrator's online stage, with versions published, receipts
      applied and committed, and no group trained twice. A container restart then lost the run directories and the
      nsys report, and the cards were withdrawn; the figures below were read from them before that. Engine findings:
      int8 KV cache vs the clipped surrogate (knowledge 588: 8% of tokens clipped at version 1, none with
      `--kv-cache f32`); the served 4B fell back to the CPU beside a full-size training reservation (589); and the
      `dropped` and first `published` frames come late (590). One nsys capture of a short 4B run (8 groups of 8,
      `--kv-cache f32 --weight-format bf16 --ctx 2048`, `max_block 1024`, LoRA rank 16; the card shared with another
      tenant's busy 63 GB server, so absolute decode speed is not attributable) gave this breakdown of a 175.7 s run:
      - startup: 27 s to the first scored group (two base loads plus the policy's first publication);
      - serving: `serve.decode` NVTX ranges 131.0 s wall over 1701 ranges, CUDA-graph time 84.6 s (32319 launches of
        2.6 ms), `serve.admit_prefill` 5.7 s;
      - training: one step, a 5.3 s GPU window with 1.5 s of kernels (forward and other 1.23 s, backward 0.23 s,
        optimiser 0.05 s, split by kernel names);
      - learner wait: 147.2 s (84% of the wall);
      - rollouts: 63 finished, 466 tokens generated and 243 trained. One rollout ran away to the 2048-token context
        three times (about 44 s per attempt, `context_length_exceeded`, retried as an infrastructure failure), and
        its group was scored at +164 s;
      - device: this process kept the card busy 89.2 s of the 170.9 s capture (52%, union of kernels, graphs and
        copies); `nvidia-smi` read about 99% only because the other tenant's server shares the card;
        `cuStreamSynchronize` accounted for 104 s of host time;
      - memory: the run added at most 28.8 GB to the card;
      - device-lease handoffs: not measured, because no NVTX range marks lend and reclaim.
      Proposals, ranked by measured share of the wall (knowledge-entry form; the orchestrator's ledger holds the run-level
      ones):
      1. Runaway rollouts (74% of this run): bound a rollout's reply and treat context exhaustion as truncation.
         the orchestrator's output bound landed on its lane branch, but its after-measurement was not taken.
      2. Learner starvation (84%): the run is generation-bound, so training-side work cannot shorten it.
      3. Startup, 27 s with two copies of the base: share the frozen base (A37-A39).
      4. NVTX for the lease and for the fit loop's step phases: lend, reclaim, forward, backward and optimiser. The
         breakdown above had to infer them from kernel names.
      5. A short step is launch- or sync-bound (28% busy in its window): low priority while the learner waits.
      Re-profile on an exclusive card before acting on decode or step speed.
- [ ] **Frames when the facts happen** (knowledge 590): emit `dropped` when `RolloutFeed` drops a stale group while it
      waits, and emit the initial `published` frame as soon as the run's first version is routable. Both are now
      emitted from the step callback. Clients work around both today (they count no stale held group, and they poll
      `/v1/models`).
- [ ] **Lossy serving under a stream job** (knowledge 588, 589): refuse, or warn loudly about, a stream job whose
      policy is served over a lossy KV cache or has fallen back to the host; measure fp8 KV against f32 at version 1.
- [x] **Support replay for truncated samplers**: the sampler's support per position in the trace, the trainer's
      log-probabilities renormalised over it, so cut samplers can train without bias.
      Serving (lane agrpo-support, 2026-11-25): `policy_trace: {"support": N}` (N in 1..=256,
      `llmserve::MAX_TRACE_SUPPORT`; alone or beside `top_log_probs`; param `policy_trace_support` over D-Bus) returns
      `support_cap`, `generation_sampler_log_probs` (each generated token's figure under the survivors it was drawn from,
      stop token included; `generation_log_probs` stays the whole row's) and the ragged `generation_support_offsets` /
      `generation_support_ids` (best first, the token among them), all from the decode step that drew the token. A cut
      draw that asks (`SampleParams::trace_support` -> `RowDraw::support`, `RowDraw::reports_support`) is answered
      `Drawn::Supported`: a `top_k` row's support is its candidate list's cut as `draw_list.wgsl` counted it (new `kept`
      output), and a nucleus or min-p row without a `top_k` is drawn from its best `N + 1` candidates against the whole
      row's mass instead of the threshold search (`RowDraw::list_drawn`), so a survivor past the cap shows in the list
      and is reported as an overflow, never a shortened set; the host reference (`draw_row_supported`, used by
      `draw_rows_host` and the admission draw) is the same arithmetic. A top list asked beside it comes from the same
      candidates. Such rows take single steps (no keyed window). Refused by name before decoding: without
      `policy_trace`, greedy, `top_k` above N, a decoder without row readback (`Scheduler::reports_sampler_supports`);
      a position not drawn by a step that reports it (`SupportGap::Unreported`) or a draw over the cap
      (`SupportGap::Overflow`) fails the request by name: a scoring pass is never substituted, it cannot know the
      survivors. A draw that cuts nothing reports empty supports and its sampler figures are the whole row's. Tests:
      `model` `it::readout_draw::a_cut_draw_reports_the_support_it_drew_from` (device and host reference against the
      exact truncation's survivors and figures for top-k, nucleus, min-p and their combinations, overflows; checked to
      fail with the kernel's count off by one), `it::serve_keyed_draw::a_cut_draw_reports_the_support_of_every_token_it_drew`
      (scheduler, stop token, overflow, host and device double), the generic conformance
      (`serve::conformance::keyed_draws_are_the_rows_draws` now asks two cut rows for their support, run on every
      serving engine: qwen3, qwen35 incl. edgerock, qwen35moe, ouro, deepseek2, huginn, inkling2, ricursive),
      `llmserve` `it::outcome_report` (reply shape, both figures, no scoring pass, every refusal and the overflow),
      `apiserve` `it::api::a_policy_trace_object_asks_for_the_sampler_support_and_is_bounded`, `catalog`
      `it::scheduled_batch::a_cut_draws_support_comes_from_the_decode_and_the_trainer_replays_it` (tiny qwen35: the
      trainer's teacher-forced figure renormalised over the reported support equals the sampler's within 2e-3).
      API audit (api-security.md): one integer param bounded by the action spec (1..=256, refused by name as
      `policy_trace.support`), the `policy_trace` object takes no other key, no path or secret; the reply grows by at most
      N ids, one offset and one figure per generated token, within the existing `max_new` and body limits.
      Records (lane agrpo-support): contract v2 gains, additively, `behavior_support_offsets` / `behavior_support_ids` /
      `behavior_support_logprobs` on a supervised answer (all three, beside `behavior_logprobs`; ragged, offsets one
      more than the tokens, from 0, rising at every token, ending at the ids; ids distinct, at most
      `MAX_BEHAVIOR_SUPPORT` 1024, the sampled token among them where `token_ids` are given; each figure finite, at most
      0, at least the whole-row one). Rules `behavior_support_count` and `behavior_support`; a `baseline_only` record
      carries none. Kept as `<split>.bso.bin` (u64 offsets) / `.bsi.bin` / `.bsl.bin` with `behavior_support_max` in
      `<split>.rollouts.json` (`read_behavior_support`), and on `EncodedSample` / `EncodedSplit::behavior_support` for
      the stream path. Fixtures: `chat_rollout.jsonl` line 9 (a `top_k` 2 rollout with its supports),
      `invalid/behavior_support_count.jsonl`, `invalid/behavior_support.jsonl`; `CONTRACT_DIGEST`
      `sha256:4fb13e2f610fb2d3d4ede4731c5da509d2ccfae1071143c796015e824a4bc20b`. Tests: `data`
      `it::chat_rollout_records` (positions in the sidecars, packing two widths, a 15-case refusal table, stale removal),
      the SDK's `it::training_records` (every fixture's verdict).
      Training (lane agrpo-support): a position whose record carries a support (`pg_batch::SamplerSupport`) is trained
      on `log pi_S = l_y - logsumexp_S l`: the support's ids are the head's extra targets (`slot_ids`: a position's
      support, else its top-K under centring; one width `slot_width` = max(top K, widest support), enabled in
      `prepare`), the forward reads the whole-row figures at them, `model::head_ce::renormalise_over_support`
      renormalises on the host in f64, and `support_coefficients` (`c_j = omega_j - Omega pi_j`) make the one backward the
      renormalised loss's gradient: the whole-row softmax cancels, no new kernel (knowledge #618). The clipped
      surrogate's ratio is `pi_S / mu_S` against `behavior_support_logprobs`; the score-centred estimator centres over the
      support (TRL's behaviour: `q_S = mu(v) / mass_mu(S)` from the record's whole-row top-K, `log mass = blp - bsl`, a
      listed id outside the support dropped, the tail the rest of the support); the KL term stays on the whole row. Read
      from `<split>.bso/.bsi/.bsl` in `ExternalPolicyGradient` and per example in the stream (`stream.support`, 1 to
      1024, sizes the slots; a record with supports and no width, or a wider one, refused by name). Refused by name: a
      supervised token outside its support, a support id past the vocabulary (`check_data`), supports under an importance
      correction, a model whose head takes no extra targets. `external::check_centrable`'s truncated-sampler refusal is
      lifted for an example whose every supervised token has a support, kept otherwise. Tests: `rl`
      `it::support_replay` (a row drawn from tiny Qwen3.5 by `draw_row_supported` with top_k 5 / top_p 0.9: replayed,
      the step-0 ratio is one within 2e-5 at every token; not replayed, it equals the support's mass, below 0.999; and
      gradchecks of the support-normalised clipped and score-centred losses on tiny Qwen3 and Qwen3.5 at
      `failures(2e-4, 2e-2)`, which a coefficient with `Omega` off by 10% fails), `it::score_centred_objective::
      records_from_a_cut_sampler_train_when_their_supports_are_replayed` (both estimators step at ratio one on-policy,
      every refusal, the warning count), `it::trl_goldens::support_replay_is_trls` (TRL's own `compute_loss` with
      `sampling_mask` and its fused head `ChunkedLogProbFunction` from the pinned tree: the support-normalised
      log-probability, mask, tail, per-token loss and the gradient in the logits, within 2e-6 relative; fixture extended
      by `tools/goldens/trl_score_centring_reference.py`, the older cases unchanged), `rl::conformance_async::
      support_replay` on every registered trainer (renormalised figure equals a softmax over the support of the model's
      own `logits_all`, the installed weights one backward equal to their label equivalents; held where score centring
      holds: qwen3, qwen35, ricursive, inkling2; qwen35moe refuses, no policy gradient), `model`
      `it::head_ce_sparse::a_loss_renormalised_over_a_support_is_one_backward_of_its_extra_targets`, `stream_pg`
      `a_stream_takes_supports_within_its_width_and_refuses_the_rest`.
      Cost (`ce_sparse_targets`' per-logit scan of the K slots, measured on the CPU backend before using it at support
      widths): 1024 rows x 32,000-token vocabulary, the sparse backward 0.45 s at K 1 and 0.52 s at K 256 against
      0.42-0.44 s for the weighted backward, so the scan was kept (knowledge #618). Open: measure it on a card at K 256
      (nsys); in-process online GRPO still records no supports (its sampler is the run's own).
- [x] **Session-merged samples** (A40-A41, lane rl-merge, 2026-11-27; knowledge 619). Measured first: a stream step
      weighs an episode by its call records (`token-sum`: by their policy tokens; `constant-length` and the offline
      `seq-mean`: by their count), so a 60-call episode outweighs a 2-call one 30:1 at equal call length
      (`rl` `it::session_samples::a_streamed_step_weighs_an_episode_by_its_call_records_so_sixty_calls_outweigh_two_thirty_to_one`),
      and each call re-forwards the context the earlier calls built. The merged form is the existing packed rollout
      record: one record of the whole session, every sampled answer `"train": true` with `token_ids`,
      `prompt_token_ids`, `behavior_logprobs` (and top-K / support when sampled so), trained with `pack_answers: true`
      (`ChatSample::answers_packed`; batch, SFT and the stream job alike). New: a packed record that names
      `prompt_token_ids` on any answer is held to the append-only property, every later answer with behaviour
      log-probabilities naming `token_ids` + `prompt_token_ids` that equal the session so far, else refused by rule
      `session_not_append_only` naming the answer (`data` `it::chat_session_records`: a template that drops an earlier
      answer's reasoning is refused, each of its call records valid alone). A packed record naming no
      `prompt_token_ids` keeps the older packed semantics (the template's history; unvalidated, as before). The contract
      example set and `CONTRACT_DIGEST` are unchanged (the rule depends on the run's packing, not on the record). The
      stream now trains `seq-mean` too (each row's terms by its own supervised count through `PgBatch::scale`, the step
      by its sequences), so merged sessions can weigh every episode alike; `token-sum` stays the setting under which a
      merged sample equals its per-call records. Oracles: `rl` `it::session_samples` (merged vs per-call gradient under
      `token-sum`, relative gap 3.4e-7; `seq-mean` over two merged sessions is the mean of each alone), and the A49
      conformance check `merged sessions` on every architecture (qwen3, qwen35, qwen35moe, ricursive, inkling2 at the
      exact fp32 GEMM, tolerance 1e-4 of the largest gradient; checked to fail with one answer unmasked). Cost on the
      CPU toy (4 sessions x 12 calls, 48-token context, 12-token answers, 20-token observations): 11328 sequence tokens
      (13568 forwarded) per call against 1648 (1664) merged, forward+backward 0.43 s against 0.05 s, same 576 policy
      tokens. Producer contract (for an orchestrator's call projection): emit one merged record per exact rollout when
      every call is exact and call `k+1`'s prompt ids start with call `k`'s prompt ids followed by its generated ids
      (the tokens between are the next turns, as the template renders them); otherwise, or when brain refuses the
      record with `session_not_append_only`, emit the per-call records. Its messages are the last request's messages
      plus the last answer, with each assistant turn the session sampled marked `train` and carrying that call's
      fields; `reward`, `group`, `policy_version` (the oldest call's), `sampler`, `tokenizer_digest` once on the record (calls drawn under different sampler controls stay per call);
      no `rollout` key needed (the record is the whole rollout). Send with `pack_answers: true` and a `max_block` that
      holds the whole session; choose `loss_agg` `token-sum` to reproduce the per-call reference or `seq-mean` to weigh
      episodes alike.
- [ ] **Group-prefix packing** (A42-A43): the rollouts of a group share their prompt, which every rollout's row
      forwards again. Packing a group into one row needs what no forward has yet: per-token position ids that restart
      at each branch and an attention mask that is a tree (each continuation sees the prefix and itself), in every
      architecture's attention, plus a state fork at the branch point for the recurrent and linear-attention mixers
      (qwen35's gated delta net, ricursive), with the gradient of the shared prefix accumulated once. The oracle would
      be the per-rollout rows' gradient, as for merged sessions. Not started: measure the prompt share of a group's
      forwarded tokens on a real run first (an agent session's prompt is a small share of its merged length).
- [ ] Later, each driven by a profile and recorded as a knowledge entry (bottleneck, baseline, evidence, change,
      oracle, result, decision): colocated serve-and-train scheduling and whether the frozen base can be shared
      (A37-A39); head and coefficient cost (A44-A45); multi-actor publication (A46-A48).
