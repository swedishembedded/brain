# Training as a served job

`brain serve --training` lets a client fine-tune a chat model's LoRA adapter on
the machine that serves it, without a training process of its own. The client
starts a job, follows its steps, and collects the adapter; brain schedules the
run on the same cards that serve inference, so one service owns the devices.

The job is the model `brain/train` with one action, `train`. It trains

- **supervised**: imitation of the supervised turns of `generic-messages-v2`
  chat records, with held-out scoring, a monitoring set, replay datasets and
  early stopping, exactly as `brain::ChatFineTune` does;
- **by policy gradient** (`objective: policy_gradient`): the clipped objective
  on rollout records that carry an `advantage`, or a `teacher_context` for
  self-distillation (a Qwen3.8 base; see `brain::ChatFineTune::policy_gradient`
  for the record format and the objective's settings);
- **by direct preference optimization** (`objective: dpo`): `dataset` and
  `held_out` hold `generic-preference-v1` pairs (a prompt and the `chosen` and
  `rejected` assistant turns), as `brain::PreferenceFineTune` trains them. The
  reference is the policy the run starts from (the base, or the `start_adapter`
  continued), frozen, so no second model is held.

## Serving it

```
brain serve --models-dir /models --training --artifact-store /shared/artifacts \
    --openai 0.0.0.0:8788 --api-key-file /etc/brain/key --dbus \
    --tls-cert /etc/brain/tls/server.crt --tls-key /etc/brain/tls/server.key
```

A surface on an address other than loopback needs TLS (see
[Transport security](../serving/serving.md#transport-security)).

`--training` turns the workload on; it is off otherwise. `--artifact-store DIR`
names the directory bulk data moves through (below); it must exist. The base is
named by model id in `--models-dir`, as everywhere (`brain pull
Qwen/Qwen3.8-27B-FP8`).

## What moves, and how

No dataset or adapter travels through the API as a message body.

- **Over D-Bus** a dataset is an input file descriptor of `Subscribe` (named
  `dataset`, `held_out`, `held_out_text`, `monitor`, `adapter` for a starting
  adapter, `replay_0` .. `replay_7`) and the trained adapter comes back as an
  output descriptor when the call sets `return_adapter`.
- **Over HTTP, and across machines,** a dataset is named by the blake3 digest of
  its bytes. The client writes the file as `<DIR>/<64 lowercase hex digits>`
  (to a temporary name in the same directory, then rename, so a reader never
  sees half of it) and sends the digest. The server resolves the digest in the
  store it was started with, copies the file into the job's private directory
  while hashing it, and refuses it if it does not hash to its name. The trained
  adapter is published to the store the same way and its digest is in the
  result. A client never names a path; a digest is exactly 64 lowercase hex
  digits and cannot spell one.

## The request

`POST /v1/jobs` (HTTP) or `Subscribe` (D-Bus) with model `brain/train`, action
`train` and these params (ranges are in `GET /v1/capabilities`):

| param | meaning |
|---|---|
| `base` | the base model, `<vendor>/<repo>` in the model store (required) |
| `data` | a JSON document naming the datasets (required), below |
| `objective` | `sft` (default), `policy_gradient` or `dpo` |
| `steps`, `grad_accum`, `rank`, `alpha`, `lr`, `weight_decay`, `seed` | the optimiser's settings; what is not named is the trainer's own default |
| `strict_updates` | abort at the first optimiser step whose loss or gradient is not finite (default off: such a step is skipped, the weights untouched, and the run aborts on the third in a row) |
| `max_block` | the longest training row in tokens (default 8192). The memory reserved for the run is sized to it and a longer record is refused by name |
| `thinking`, `pack_answers` | how records are rendered |
| `base_format` | how the frozen base is held: `bf16`, `fp8`, `nf4`, ...; default the tier that fits the card |
| `eval_every`, `patience`, `replay_share`, `keep_evaluations` | monitoring and early stopping |
| `adapter_id` | the id on the adapter's card |
| `return_adapter` | also return the adapter as an output blob (D-Bus) |
| `clip_eps`, `kl_beta`, `reference`, `behaviour`, `loss_agg`, `temperature`, `teacher`, `advantage_bound`, `sequence_advantage_cap` | the policy-gradient objective's settings (only with `objective: policy_gradient`) |
| `behaviour` | where the importance ratio's behaviour log-probabilities come from. `recorded` (default): the sampler's own `behavior_logprobs`, on every supervised token of every record with an advantage; a dataset missing any is refused. `start_of_run_proxy`: the start-of-run policy's where none was recorded, exact only if the rollouts were sampled from it; the training record states which was used |
| `beta`, `nll_weight`, `keep_reasoning` | the preference objective's settings (only with `objective: dpo`): the loss temperature (default 0.1), the weight of the supervised likelihood of the chosen answer added to it (default 0) and whether the reasoning of each candidate is trained on |
| `arch` | the base's architecture, only for a checkpoint directory with no `config.json` |

`data` is `{"train": SRC, "held_out"?: SRC, "held_out_text"?: SRC, "monitor"?:
SRC, "start_adapter"?: SRC, "replay"?: [SRC], "weighted_replay"?: [{SRC fields,
"share"}]}` where `SRC` is `{"digest": "<blake3>"}` or `{"blob": "<input
name>"}`. Unknown fields anywhere are refused, as is a blob that is named but
not attached or attached but not used.

A preference run takes only what it uses: `held_out_text`, `monitor`, replay sets, `replay_share`,
`weight_decay`, `eval_every`, `patience`, `keep_evaluations`, `pack_answers`, `thinking: false` and the
policy-gradient settings are refused by name, never ignored. Each step holds both candidates of a pair,
so the memory reserved is sized to two `max_block` rows.

A request that could never run is a 400 (HTTP) or a bus error before anything
starts: an unknown base, a digest the store does not hold, a setting the
objective cannot use, a run that needs more than the largest card holds.

## Following a job

Over HTTP, `GET /v1/jobs/{id}/events` is a server-sent event stream: a `frame`
event for each thing the run reports, then one `done` event with the status
(`succeeded`, `failed` with the reason, `cancelled`). The last 4096 frames are
kept, so a client that connects late or reconnects with `Last-Event-ID` replays
what it missed. `GET /v1/jobs/{id}` and `.../result` work as for every job, and
`DELETE /v1/jobs/{id}` cancels: the run stops at the next optimiser step.
Over D-Bus the frames are `progress` frames whose `event` is the same object,
and `Cancel(job)` cancels.

A frame is `{"type": "stage", "stage": ...}` while the datasets are staged and
the base is loaded, then one per optimiser step:

```
{"type": "step", "step": 7, "steps": 100, "loss": 1.82, "lr": 0.0003,
 "grad_norm": 0.41, "seconds": 93.2,
 "pg": {"loss": ..., "clip_fraction": ..., "kl": ..., "ratio_mean": ..., "ratio_min": ...,
        "ratio_max": ..., "mean_advantage": ..., "scoring_seconds": ...}}
```

`pg` is present for a policy-gradient run, with the teacher-advantage statistics
(`token_advantage_mean`, `token_advantage_max`, `token_advantage_clip_fraction`)
when records carry a teacher context. `seconds` is the time since training
began. Step frames over D-Bus are droppable like every progress frame: a
subscriber that does not read them loses frames, never the result.

The result is `outputs` (the report: `objective`, `adapter_digest`, the scores before and
after, the monitoring curve, `selected_step`, the policy-gradient totals,
`base_tier` and why; for `dpo`, `beta`, `nll_weight` and `train_score` and `held_out_score`, each
`{accuracy, mean_margin, pairs, skipped}`: the share of pairs the tuned policy prefers the chosen
answer on more than its reference does, and the mean margin in nats). The trainer's own training record, which holds every
step's statistics, is published to the artifact store beside the adapter
(`record_digest`), and returned as the blob `record` when the call sets
`return_adapter` or the server has no store.

## Scheduling

The run is a workload of the serving executor, so it takes devices from serving
and gives them back:

- It needs the bytes of the frozen base at its tier, the activations of one
  `max_block` row and the adapter with its optimiser state. If a card has them
  free, the run takes the one with the most room.
- Otherwise it evicts the least valuable **idle** serving instances on the best
  candidate card, whole instances at a time. A replica evicted this way is
  rebuilt by the next request for it, on a card with room.
- It never takes memory from an instance a request is running on. While serving
  is busy on every card that could hold the run, the run waits; past the
  admission deadline (`--admit-deadline-ms`) it is refused (HTTP 429, a bus
  `error` frame), and the client retries.
- One run holds one card at a time, and several runs use several cards. The
  trainer is one card per run; a run that needs more than the largest card
  holds is refused at once, with what it needs.
- When a run ends, by completion, failure or cancel, it gives its card back at
  once, and its files are removed.

## Refitting while sampling (asynchronous rollouts)

A learning loop that keeps sampling while it trains publishes each trained
adapter to a running server (`brain serve --serve-adapters DIR`, see
[Serving LoRA adapters](../serving/serving.md#the-adapter-bank)) and samples
from the adapter's model id. The server tags every sample with the policy
version that drew it, so the trainer can decide what is too stale to train on:

- The adapter's `version` (in `GET /v1/models` and in the `adapter` object of
  every reply and of a `policy_trace`) is 1 for the first release of its card
  id and +1 for each later release with different weights. A sequence runs on
  the release it started on and never spans a refit, so every token of a reply
  carries the one version it names.
- The loop keeps the mapping from its own optimiser step to the version it
  published; a sample whose version is more than the loop's staleness bound
  behind the version it trains is masked or dropped. Nothing on the server
  interprets the bound.
- A refit never reads what an earlier release cached: each release has its own
  prefix-cache namespace, so the first request after a refit recomputes its
  prompt and later ones share it.
- Several adapters are served at once, each under its own id, in the same
  batches; a version is per adapter id. The rank cap (`--adapter-rank-cap`,
  default 32) is fixed when the server starts, so an adapter above it cannot be
  hot-loaded.

## Security

The surface is behind the server's key like every route and takes no parameter
that names a file: the work directory, the model store and the artifact store
are the operator's. The artifact store is the one place a client can put bytes
the server will read, and it reads a file only by a digest it has verified. Run
the store on a directory only the clients you trust can write to. See
[`docs/guides/serving/security.md`](../serving/security.md).
