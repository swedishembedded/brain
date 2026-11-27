<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Metrics and health: what a server and a router report

`brain serve` answers three machine-readable documents, all behind the surface's key, on every provider port:

| Route | Schema | What it is for |
|---|---|---|
| `GET /health`, `/v1/health` | `brain.server.health` v1 | Cheap, polled every few seconds by a router: is the server up, how much does it take at once, what has it served. |
| `GET /metrics`, `/v1/metrics` | `brain.server.metrics` v1 | Everything an observer needs to see the server's state: each model's residency, scheduler, back-off and rates, each card's memory, each training job's phases. |
| `GET /health`, `/v1/health`, `/metrics`, `/v1/metrics` on `brain serve --upstream ...` | `brain.router.health` v1 | The router's view of its fleet, per upstream. |

## Stability

Each document names its schema and version in `schema` and `schema_version`. Within a version fields are only
added, never renamed, removed or given another meaning or unit; a reader ignores fields it does not know. A change
that would rename or remove a field, or change its unit, is a new version. Conformance tests pin every field of each
document against what the server and router answer, and `make check/metrics-fields` holds this page to the pinned
lists, so a field cannot disappear, be renamed or go undocumented silently.

Conventions: `_bytes` is bytes, `_secs` seconds (a float), `_unix_ms` milliseconds since the Unix epoch (wall clock of
the server), `_per_s` per second, `_tokens` and `_blocks` counts. A counter (`requests_completed`, `decoded_tokens`,
`spills`, ...) starts at zero when the process starts and only grows; a reader takes a rate from two readings and
treats a smaller value, or a changed `instance`, as a restart. `null` means "not known" or "does not apply", never
zero. Devices are named as `--device` names them: `gpu0`, `npu0`, `cpu`. Paths below write an array's items as
`[]`.

## `brain.server.metrics`

| Field | Meaning |
|---|---|
| `schema` | `"brain.server.metrics"`. |
| `schema_version` | `1`. |
| `instance` | The process's identity for its lifetime (the same as `/health`'s): a change means the server restarted. |
| `time_unix_ms` | When the document was taken. |
| `queue_depth` | Jobs waiting for a lane now, over every model. |
| `serving_held` | Every card the server computes on runs a workload (a fine-tune) that has not lent it: nothing sent here runs until it ends. |
| `models` | Every model the server lists, workloads (`brain/train`) excepted: one entry per model id, adapters included. |
| `models[].model` | The model id requests name. |
| `models[].slot_group` | The model whose instance serves it: itself, or for an adapter its base. Models of one slot group share one batch and one residency; `jobs`, `totals` and `replicas` are on the slot group's own entry only. |
| `models[].residency` | Where the slot group's instance is. |
| `models[].residency.state` | `resident` (built on a device), `building` (claimed and being built or promoted), `warm` (demoted off its card, promoted back on the next request), `evicted` (built once, evicted since), `unservable` (its last claim was refused for memory, see the reason), or `not_loaded` (never built). Over several replicas: the most available state any replica is in. |
| `models[].residency.unservable_reason` | Why the last claim was refused (a card short of memory right now, or a model no card can ever hold); `null` when it was not. |
| `models[].residency.replicas_resident` | Replicas built on a device now. |
| `models[].residency.devices` | Devices the instance occupies (resident, building or warm). |
| `models[].residency.building_secs` | How long the build in progress has run; `null` when none is. |
| `models[].residency.last_build` | The most recent build of any replica; `null` before the first. |
| `models[].residency.last_build.ended_unix_ms` | When it ended. |
| `models[].residency.last_build.duration_secs` | How long it took, from the claim to the instance being ready. |
| `models[].residency.last_build.succeeded` | Whether it produced an instance. |
| `models[].residency.last_eviction` | The most recent eviction of any replica; `null` before the first. |
| `models[].residency.last_eviction.at_unix_ms` | When it was evicted. |
| `models[].residency.last_eviction.resident_secs` | How long it had been resident since its last build; `null` when not known. |
| `models[].residency.last_eviction.demoted` | `true` when it went to the warm tier (kept off its card), `false` when it was dropped. |
| `models[].served` | What requests for this model id were answered with since the process started. |
| `models[].served.requests_completed` | Requests answered with a result. |
| `models[].served.prompt_tokens` | Prompt tokens of those results. |
| `models[].served.completion_tokens` | Completion tokens of those results. |
| `models[].served.cancelled` | Requests their caller cancelled. |
| `models[].served.refused` | Requests answered with an error, by cause. |
| `models[].served.refused.<key>` | The count for one cause, keyed by it: the error's failure class (`invalid_request`, `unavailable`, `timeout`, `unsupported_capability`, `not_installed`, `failed_to_load`, `backend_failure`), or `internal` for an error that names none. A cause appears once it has happened. |
| `models[].jobs` | The executor's jobs for the slot group now. |
| `models[].jobs.queued` | Jobs waiting for a lane. |
| `models[].jobs.running` | Jobs running on a lane (a batch's requests each count). |
| `models[].totals` | Sums over `replicas[].scheduler`: what a dashboard reads first. Each is `null` without a scheduler-backed replica. |
| `models[].totals.kv_total_tokens` | KV pool tokens. |
| `models[].totals.kv_claimable_tokens` | KV tokens a new request can claim. |
| `models[].totals.kv_total_blocks` | KV pool blocks. |
| `models[].totals.kv_free_blocks` | KV blocks free. |
| `models[].totals.prefix_cached_blocks` | KV blocks the prefix cache holds. |
| `models[].totals.admitted_width` | Sequences admitted at once now. |
| `models[].totals.running_sequences` | Sequences decoding. |
| `models[].totals.waiting_sequences` | Requests waiting in the schedulers. |
| `models[].totals.decode_tokens_per_s` | Decoded tokens per second over the window. |
| `models[].totals.prefill_tokens_per_s` | Prefilled tokens per second over the window. |
| `models[].replicas` | Each built replica that batches through the continuous-batching scheduler (text models), as last published (a few times a second while it runs, and on every read while it is idle). |
| `models[].replicas[].config` | The replica's instance configuration (`#replica=N` marks a replica past the first). |
| `models[].replicas[].scheduler` | The replica's scheduler. |
| `models[].replicas[].scheduler.kv` | Its paged KV pool. |
| `models[].replicas[].scheduler.kv.block_tokens` | Tokens one block holds. |
| `models[].replicas[].scheduler.kv.total_blocks` | Blocks in the pool. |
| `models[].replicas[].scheduler.kv.free_blocks` | Blocks no sequence and no cached prefix holds. |
| `models[].replicas[].scheduler.kv.prefix_cached_blocks` | Blocks the prefix cache holds (given up when the pool needs them). |
| `models[].replicas[].scheduler.kv.claimable_tokens` | Tokens a new request can claim: free blocks and blocks only the prefix cache holds, as of the last iteration (an upper bound). |
| `models[].replicas[].scheduler.kv.total_tokens` | The pool's capacity in tokens. |
| `models[].replicas[].scheduler.sequences` | What it holds. |
| `models[].replicas[].scheduler.sequences.running` | Sequences decoding. |
| `models[].replicas[].scheduler.sequences.prefilling` | Sequences whose prompt is being read in. |
| `models[].replicas[].scheduler.sequences.waiting` | Requests submitted and not admitted yet. |
| `models[].replicas[].scheduler.sequences.swapped` | Admitted sequences whose KV is parked in host memory, waiting for device blocks. |
| `models[].replicas[].scheduler.admission` | How many sequences it admits at once. |
| `models[].replicas[].scheduler.admission.admitted_width` | The most sequences (decoding and prefilling) it admits now: its batch, or the back-off limit. This is the replica's share of the model card's `decode_slots`. |
| `models[].replicas[].scheduler.admission.max_width` | Its batch: the most it ever admits. |
| `models[].replicas[].scheduler.admission.backoff` | Its back-off after the card refused memory (knowledge #613). |
| `models[].replicas[].scheduler.admission.backoff.limited` | Whether it admits fewer than its batch now. |
| `models[].replicas[].scheduler.admission.backoff.limit` | The width it is held to; `null` while not limited. |
| `models[].replicas[].scheduler.admission.backoff.failed_at_width` | The fewest sequences in flight at a failure since it last drained or the card gained room; `null` when none. |
| `models[].replicas[].scheduler.admission.backoff.cause` | What the last failure was: `device_out_of_memory` (the card refused an allocation: narrows the width) or `kv_pool_dry` (the KV pool ran out of blocks: narrows only when it repeats); `null` before any. |
| `models[].replicas[].scheduler.admission.backoff.last_device_oom_unix_ms` | When the card last refused an allocation; `null` if never. |
| `models[].replicas[].scheduler.admission.backoff.last_pool_dry_unix_ms` | When the KV pool last ran dry; `null` if never. |
| `models[].replicas[].scheduler.admission.backoff.calm_iterations` | Iterations with work and no failure since the last failure or growth: the recovery's progress. |
| `models[].replicas[].scheduler.admission.backoff.grows_after_iterations` | The calm iterations at which the limit next grows; `null` while not limited. |
| `models[].replicas[].scheduler.admission.backoff.next_limit` | The limit after that growth; `null` when it is the whole batch again (or there is no limit). |
| `models[].replicas[].scheduler.throughput` | Its tokens. |
| `models[].replicas[].scheduler.throughput.window_secs` | The window the rates are measured over (10). |
| `models[].replicas[].scheduler.throughput.decode_tokens_per_s` | Tokens sampled per second over the last window (or since the replica was built, when younger). |
| `models[].replicas[].scheduler.throughput.prefill_tokens_per_s` | Prompt tokens computed by prefill per second over the same window (prefix-cache hits are not computed). |
| `models[].replicas[].scheduler.throughput.decoded_tokens` | Tokens sampled since the replica was built, stop tokens included. |
| `models[].replicas[].scheduler.throughput.prefilled_tokens` | Prompt tokens computed by prefill since the replica was built. |
| `cards` | Every device the residency manager budgets: accelerators by index, then the host (`cpu`). |
| `cards[].device` | Its name. |
| `cards[].total_bytes` | Its capacity as the manager holds it: configured, then re-measured as the card's free memory changes. |
| `cards[].reserved_bytes` | Headroom kept free on purpose (`--reserve-gb`). |
| `cards[].used_bytes` | Bytes charged to what is placed there: resident instances, builds in progress, training runs. |
| `cards[].free_bytes` | Bytes a new placement could take now. |
| `cards[].ceiling_bytes` | The most one placement could ever take there (the card's size less the reserve). |
| `cards[].training_reservation_bytes` | Of `used_bytes`, what running training jobs reserved: never given to serving while they run. |
| `cards[].serving_held` | A running training job holds the card and has not lent it: nothing serves on it until the job ends. |
| `training_jobs` | Every workload job (`brain/train` actions: `train`, `full_train`, `pretrain`, ...) not ended, and the last 16 that ended, in submit order. |
| `training_jobs[].id` | The executor's job id (submit order, per process). |
| `training_jobs[].model` | The workload model (`brain/train`). |
| `training_jobs[].action` | The action (`train`, `full_train`, ...). |
| `training_jobs[].state` | `queued` (waiting for its card), `running`, `completed`, `failed` or `cancelled`. |
| `training_jobs[].submitted_unix_ms` | When it was submitted. |
| `training_jobs[].started_unix_ms` | When it got its card(s); `null` while queued. |
| `training_jobs[].ended_unix_ms` | When it was answered; `null` while not. |
| `training_jobs[].queued_secs` | How long it waited for its card(s): until it started, or until now. |
| `training_jobs[].wall_secs` | From submission to its end, or to now. |
| `training_jobs[].report` | The job's own account of itself, as of `report.as_of_unix_ms` (published at every stage and step); `null` until it starts. |
| `training_jobs[].report.action` | The action. |
| `training_jobs[].report.devices` | The cards it runs on. |
| `training_jobs[].report.as_of_unix_ms` | When the account was taken. |
| `training_jobs[].report.current_phase` | What it does now: `staging`, `model_load`, `training` (between its steps), `waiting_for_card`, `ended`, or `other`. |
| `training_jobs[].report.steps` | Its optimizer steps. |
| `training_jobs[].report.steps.done` | Steps completed in the run (a resumed run continues the count). |
| `training_jobs[].report.steps.total` | Steps the run is asked for. |
| `training_jobs[].report.optimizer_tokens` | Record tokens trained on inside optimizer steps, every attempt. |
| `training_jobs[].report.optimizer_tokens_per_s` | `optimizer_tokens` per second of `phases_secs.optimizer_steps`: the rate inside optimizer steps. |
| `training_jobs[].report.oom_retries` | Times the card refused an allocation and the run was tried again from its last saved step. |
| `training_jobs[].report.wall_secs` | Its wall time as of the account: queued plus running. |
| `training_jobs[].report.outside_optimizer_steps_share` | The share of `wall_secs` outside optimizer steps (0 to 1). |
| `training_jobs[].report.phases_secs` | `wall_secs` split into phases: every second is in exactly one, and they add up to `wall_secs`. |
| `training_jobs[].report.phases_secs.waiting_for_card` | Queued for its card(s), and backing off after the card refused an allocation. |
| `training_jobs[].report.phases_secs.staging` | Fetching and verifying datasets and starting adapters into the job's directory. |
| `training_jobs[].report.phases_secs.model_load` | Building the trainer before its first step (base weights, tokenized datasets), once per attempt. |
| `training_jobs[].report.phases_secs.optimizer_steps` | Inside optimizer steps: drawing batches, forward, backward, optimizer. |
| `training_jobs[].report.phases_secs.evaluation` | Held-out and monitoring passes, with the evaluation snapshots and best parameters they keep. |
| `training_jobs[].report.phases_secs.checkpoint_save` | Writing checkpoints, the training state and published adapter versions. |
| `training_jobs[].report.phases_secs.waiting_for_records` | A streamed run waiting for its next records. |
| `training_jobs[].report.phases_secs.finalize` | After the last step: the final save and publishing the adapter. |
| `training_jobs[].report.phases_secs.other` | Whatever the rest does not name: the time since a running job's last step until its next says what it was, a failed attempt's unwind. |

## `brain.server.health`

| Field | Meaning |
|---|---|
| `schema` | `"brain.server.health"`. |
| `schema_version` | `1`. |
| `status` | `"ok"`. |
| `instance` | The process's identity for its lifetime: a change means the server restarted. |
| `decode_slots` | The sequences the served models decode at once, summed over every model the server could load. |
| `free_slots` | Of those, how many are free now; 0 while `serving_held`. |
| `serving_held` | Every card runs a workload (a fine-tune) that has not lent it: a router sends nothing here. |
| `kv_free_tokens` | KV tokens a new request can claim, summed over the paged text models; `null` when none is built. |
| `kv_total_tokens` | Their pools' total tokens; `null` when none is built. |
| `edge_concurrency` | Requests the surface admits at once before it sheds with 503. |
| `queue_depth` | Jobs waiting for a lane now. |
| `served` | What the process has served, over every surface (counters). |
| `served.requests` | Finished generations. |
| `served.prompt_tokens` | Their prompt tokens. |
| `served.completion_tokens` | Their completion tokens. |
| `served.cached_prompt_tokens` | Their prompt tokens served from the prefix cache. |

## `brain.router.health`

Answered with 200 while some upstream is in rotation, else 503 with the same body.

| Field | Meaning |
|---|---|
| `schema` | `"brain.router.health"`. |
| `schema_version` | `1`. |
| `status` | `"ok"`, or `"unavailable"` when no upstream is in rotation. |
| `decode_slots` | The decode slots of the upstreams in rotation, summed: the most requests the router sends at once. |
| `capacity` | The same as `decode_slots`. |
| `in_flight` | Requests the router has sent and not finished. |
| `queued` | Requests waiting in the router for a slot. |
| `queue_bound` | The most that may wait; one more is refused with 503. |
| `upstream_queue_depth` | Jobs the upstreams in rotation report waiting, summed. |
| `tokens_per_s` | Completion tokens per second of the upstreams in rotation, between their last two probes. |
| `keying` | How the router keyed prompts for prefix affinity (counters). |
| `keying.token_keyed_requests` | Requests keyed by their tokens. |
| `keying.text_keyed_requests` | Requests keyed by their text (no tokenizer for the model). |
| `keying.encoded_tokens` | Tokens the router encoded. |
| `keying.memo_tokens` | Tokens it took from its memo of earlier prompts instead. |
| `keying.encode_seconds` | Seconds it spent encoding. |
| `upstreams` | One entry per upstream, by index (addresses stay with the operator). |
| `upstreams[].index` | Its index in the router's list. |
| `upstreams[].healthy` | Its last probe answered. |
| `upstreams[].serving_held` | Its health says every card is held by a workload. |
| `upstreams[].draining` | It takes nothing new (an operator drained it). |
| `upstreams[].stalled` | It held work and made no progress for the stall bound: out of rotation. |
| `upstreams[].in_flight` | Requests the router has sent it and not finished. |
| `upstreams[].capacity` | Its decode slots: the most it is sent at once. |
| `upstreams[].queue_depth` | Jobs it reports waiting. |
| `upstreams[].tokens_per_s` | Its completion tokens per second between its last two probes. |
| `upstreams[].prefix_hit_rate` | Its share of prompt tokens served from its prefix cache between its last two probes; `null` before it served a prompt. |
| `upstreams[].idle_connections` | Idle keep-alive connections the router holds to it. |
| `upstreams[].free_slots` | Of its decode slots, those free by its last health. |
| `upstreams[].kv_free_tokens` | Its claimable KV tokens by its last health; `null` when it reports none. |
| `upstreams[].models` | Models it lists. |
| `upstreams[].spills` | Requests the router sent it as a spill: the model was not resident there, every upstream where it was carried the spill load, and this one was idle (counter). |
| `upstreams[].balance_moves` | Requests it took although their session or their longest cached prefix was on another upstream, because that one carried more than the balance bound (counter). |
| `upstreams[].stalls` | Times it was found stalled and taken out of rotation (counter). |
