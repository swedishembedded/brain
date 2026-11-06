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
  for the record format and the objective's settings).

## Serving it

```
brain serve --models-dir /models --training --artifact-store /shared/artifacts \
    --openai 0.0.0.0:8788 --api-key-file /etc/brain/key --dbus
```

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
| `objective` | `sft` (default) or `policy_gradient` |
| `steps`, `grad_accum`, `rank`, `alpha`, `lr`, `weight_decay`, `seed` | the optimiser's settings; what is not named is the trainer's own default |
| `max_block` | the longest training row in tokens (default 8192). The memory reserved for the run is sized to it and a longer record is refused by name |
| `thinking`, `pack_answers` | how records are rendered |
| `base_format` | how the frozen base is held: `bf16`, `fp8`, `nf4`, ...; default the tier that fits the card |
| `eval_every`, `patience`, `replay_share`, `keep_evaluations` | monitoring and early stopping |
| `adapter_id` | the id on the adapter's card |
| `return_adapter` | also return the adapter as an output blob (D-Bus) |
| `clip_eps`, `kl_beta`, `reference`, `loss_agg`, `temperature`, `teacher`, `advantage_bound`, `sequence_advantage_cap` | the policy-gradient objective's settings (only with `objective: policy_gradient`) |
| `arch` | the base's architecture, only for a checkpoint directory with no `config.json` |

`data` is `{"train": SRC, "held_out"?: SRC, "held_out_text"?: SRC, "monitor"?:
SRC, "start_adapter"?: SRC, "replay"?: [SRC], "weighted_replay"?: [{SRC fields,
"share"}]}` where `SRC` is `{"digest": "<blake3>"}` or `{"blob": "<input
name>"}`. Unknown fields anywhere are refused, as is a blob that is named but
not attached or attached but not used.

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

The result is `outputs` (the report: `adapter_digest`, the scores before and
after, the monitoring curve, `selected_step`, the policy-gradient totals,
`base_tier` and why). The trainer's own training record, which holds every
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

## Security

The surface is behind the server's key like every route and takes no parameter
that names a file: the work directory, the model store and the artifact store
are the operator's. The artifact store is the one place a client can put bytes
the server will read, and it reads a file only by a digest it has verified. Run
the store on a directory only the clients you trust can write to. See
[`docs/guides/serving/security.md`](../serving/security.md).
