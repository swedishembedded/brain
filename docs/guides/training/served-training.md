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

## Full-parameter tuning and merging

Two more actions of `brain/train` produce a checkpoint instead of an adapter.

- **`full_train`** trains every weight of the base (fp32, the AdamW moments and master weights in host memory, the
  card holding the weights, their gradients and one step's activations; any architecture with a full-parameter
  trainer: Qwen3, Qwen3.5, Qwen3.5 MoE, Ripple, Ouro). `data` takes
  `train`, `replay` and `weighted_replay` only; held-out sets, a monitor and a starting adapter are refused by name.
  Params: `base`, `data`, `steps`, `lr`, `weight_decay`, `grad_accum`, `strict_updates`, `seed`, `max_block`,
  `thinking`, `arch`, `replay_share`, `train_precision` (`key=value,...`, e.g. `state=int8,sr=on`: what each
  trainable parameter, its gradient and its optimiser state are stored as; fp32 everywhere by default, and an unknown
  key or value is refused before anything loads), `install_as`, `resume_key`, `checkpoint_every` (below),
  `exit_entropy` (below).
- **`merge`** folds a LoRA adapter into the weights it targets (`W + alpha/r B A`). The adapter is the input blob
  `adapter` or `adapter_digest` (exactly one); the work is host-side and takes no card.

A checkpoint is a directory and an artifact is one file, so a checkpoint leaves as a **manifest**: each file is
published to the artifact store by digest, and the manifest, `{"format": "brain-checkpoint-manifest-v1", "files":
[{"name", "digest", "bytes"}]}`, is published too. The result carries `checkpoint_digest` (the manifest's) and `files`.
`install_as` additionally installs the checkpoint in the server's model store as `local/<install_as>` (letters,
digits, `.`, `_`, `-`; a name that exists is refused), where a later job names it as its `base`. A server with no
artifact store can only install. The files of an installed checkpoint are as large as the model: the operator owns
the model store's disk.

## A model from nothing

Two more checkpoint actions create a model and pretrain it, for any architecture that declares a from-scratch
initialization (`ouro`, `qwen3`; the SDK's `brain::from_scratch_architectures` lists them). The sequence is
`init_model`, then `pretrain`, then `full_train` on chat records, each installing its checkpoint for the next.

**`init_model`** learns a byte-level BPE vocabulary on a text corpus, checks a chat template and writes a randomly
initialised checkpoint. It is host work and takes no card.

| param | meaning |
|---|---|
| `arch` | the architecture, a registry id that declares the capability (required) |
| `config` | the model's configuration, a JSON object in the architecture's `config.json` spelling, at most 64 KiB (required); `vocab_size` may be left out, and given must equal the tokenizer's |
| `tokenizer` | `{"vocab_size": N, "special_tokens": [...], "eos_token": T, "bos_token"?: T, "pad_token"?: T, "split_digits"?: bool}` (required): the vocabulary is the 256 bytes, the merges learned on the corpus, then the special tokens in declared order; the special tokens are matched whole and never learned as pieces, and the eos/bos/pad tokens must be among them. `split_digits` (default off) makes every digit its own pre-token, so a number is read one token per digit and no merge holds a digit; the `tokenizer.json` declares the split (a `Digits` step before the byte-level one) |
| `chat_template` | the Jinja chat template; without one the model is a base model. It is refused unless it holds to the contract in [Ouro](../../reference/models/ouro.md#the-chat-template-a-from-scratch-model-meets) |
| `sampling` | a JSON object in `generation_config.json` spelling (`do_sample`, `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`): the sampling the model is served with when a request names none, written into its `generation_config.json` and carried by every checkpoint trained from it (`{"do_sample": false}` serves it greedily). Without it a chat model is served with the chat fallback (temperature 0.7, top-k 20, top-p 0.95). An unknown key or an out-of-range value is refused |
| `corpus_digest` or the blob `corpus` | the corpus, JSONL with one `{"text": ...}` document per line (exactly one of the two) |
| `tokenizer_bytes` | the vocabulary is learned from at most this many bytes of the first documents (default 64 MiB) |
| `seed` | the initialization's seed (random when omitted; reported). The same request writes the same files |
| `install_as`, `resume_key` | as for `full_train` |

The checkpoint holds the weights (`model.brain.safetensors` for a brain-format family, a `transformers` directory
for qwen3), `tokenizer.json`, `tokenizer_config.json` (the template and the special tokens) and
`generation_config.json` (the eos id). The report adds `arch`, `seed`, `vocab_size`, `merges`, `params`, `config`,
`tokenizer_digest` and `tool_calls` (the template serves tool calls).

**`pretrain`** trains every weight of a base on raw text: each document of the corpus is encoded with the base's own
tokenizer (special tokens matched whole, so conversations serialized in the model's template train its own tokens)
and followed by the separator token; the documents form one stream, the run draws windows of `block + 1` tokens from
it with every position supervised, and a held-out corpus is scored every `eval_every` steps. `data` is `{"train":
SRC, "held_out"?: SRC}` (the blobs are `dataset` and `held_out`); params `steps`, `batch` (rows per micro-batch,
default 8), `block` (tokens per row, default 256, at most the model's positions), `grad_accum`, `lr` (default
1e-3), `min_lr` (default a tenth of `lr`), `warmup` (default a twentieth of the run), `weight_decay` (default 0.1),
`grad_clip` (default 1), `eval_every` (needs `held_out`), `eval_batches` (default 8), `separator` (default
`<|endoftext|>`), `strict_updates`, `seed`, `arch`, `install_as`, `resume_key`, `checkpoint_every`, `exit_entropy`.
For an architecture whose depth is chosen by exit gates (Ouro), `exit_entropy` is the weight `beta` of the
entropy-regularised objective of arXiv:2510.25741: the exit-weighted loss minus `beta` times the entropy of each
supervised position's exit distribution, which keeps training from collapsing the exit mass onto one pass. It
defaults to that recipe's weights, 0.1 for `pretrain` and 0.05 for `full_train`; 0 turns it off; any other weight
is refused for an architecture without exit gates. The report carries the weight used. The step frame of
an evaluating step carries `eval_loss`; the report adds `evaluations` (`[{step, train_loss, eval_loss}]`),
`documents`, `tokens`, `held_out_documents`, `held_out_tokens`, `separator` and `separator_id`.

## Resuming a job

`train` (every objective: `sft`, `policy_gradient`, `dpo`), `init_model`, `pretrain` and `full_train` take a
`resume_key` (1 to 128 letters, digits, `.`, `_`, `-`, not starting with a dot; needs an artifact store), and every
one but `init_model` a `checkpoint_every`. The key names one run:

- The run's **identity** is the blake3 digest of the action, every param (defaults filled) but `resume_key` and
  `checkpoint_every`, and the digests of the attached blobs. A key whose run has another identity is refused.
- The run keeps a record and its exact training state in the artifact store's directory, under
  `.runs/<blake3 of the key>/`: the record (status, seed, result) written atomically, the training state (weights,
  AdamW moments, schedule position, data order and generator) saved every `checkpoint_every` steps (default 25) and
  when the job is cancelled, itself written atomically. An omitted `seed` is drawn once and kept.
- **Submitting the same request again** after the job, or the server, died or was cancelled continues it from the
  last saved state, and it ends at the weights a run that was never stopped writes (for `train`, the adapter file
  byte for byte, so its `adapter_digest` is the uninterrupted run's; the policy gradient's start-of-run policy and
  DPO's reference stay the policy the run began from, not the one the state restored); the report's `resume` is
  `{"key", "state": "continued", "resumed_at": <step>}` (`"fresh"` for a first run, `"restarted"` for one that died
  before saving any state). Its step frames continue the step count.
- Submitting the request of a **completed** run answers with the result it recorded (`resume.state: "completed"`)
  without training, once every artifact it names is verified to be in the store (for `train`, the adapter, the
  training record and every kept evaluation; with `return_adapter` the adapter and the record come back as blobs,
  read from the store). An `install_as` the run made
  before dying is recognized as its own when its files are byte-identical.
- A key whose job is still running is refused, as is a record or a training state that cannot be read or that
  belongs to another run: nothing is silently restarted; submit under a new key. The training state is removed
  when the run completes; the record stays.

## Timeline models

`timeline_train` trains a timeline (survival) model, or an ensemble, on `timeline-v1` subjects: the params of the
model's own `train` action (see [Horizon](../../reference/models/horizon.md)) without `out` and `force`, the datasets
as the input blobs `dataset` and `held_out` or `dataset_digest` and `held_out_digest`, and `install_as`. The model
leaves like a full-parameter checkpoint (a manifest of per-file digests, `checkpoint_digest`); `install_as` installs
it as `local/<name>` with a manifest of family `horizon`, and the server then serves it as a model of its own.

## Checking a dataset first

`check_dataset` reads a dataset with the base's own tokenizer and chat template, as the trainer would, and reports
without training or claiming a card. Params: `base` (not for a `timeline` dataset), `kind` (`chat` for `generic-messages-v2` records, `preference`
for `generic-preference-v1` pairs, `timeline` for `timeline-v1` subjects, each line validated and no token counted), the dataset as the input blob `dataset` or as `dataset_digest` (exactly one),
`thinking` and `pack_answers` (rendered as a run with those settings would), `max_block`, and `per_record` (with
`offset` and `limit`) to also list each valid record's length. The result:

| field | meaning |
|---|---|
| `records`, `valid_records` | non-blank lines, and those that parse and encode |
| `examples` | what the valid records are trained as (a record of several answers can be several examples; a pair is two) |
| `tokens`, `supervised_tokens` | per example: `total`, `min`, `mean`, `p50`, `p90`, `p99`, `max`, in the base's tokens |
| `examples_without_supervision` | examples that would train on nothing |
| `over_max_block`, `over_max_block_lines` | examples longer than `max_block` (a run refuses them rather than cutting them) and the first lines holding one |
| `invalid_total`, `invalid` | records the trainer refuses, and the first 20 as `{line, error}` |
| `per_record`, `per_record_next` | with `per_record` only: `[line, tokens]` for each valid record of the page, `tokens` being its longest example (what the training row has to hold, a teacher's rendering included); `per_record_next` is the `offset` of the next page, `null` after the last. A page holds at most `limit` records (default 50000, at most 200000); the whole file is read for every page, so the totals above are the same on each. |

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

### Storage quotas

The store and every job's private directory are bounded, so one caller cannot
fill the disk the other jobs share. Sizes are a byte count or `512M`, `8G`,
`8GiB`, `1T` (binary multiples).

| flag | default | bounds |
|---|---|---|
| `--artifact-max-bytes SIZE` | 64GiB | one published artifact |
| `--artifact-store-max-bytes SIZE` | 1TiB | the artifact store in total, concurrent publishes included |
| `--job-staging-max-bytes SIZE` | 128GiB | what one job keeps in its directory: staged datasets, checkpoints, temporaries |
| `--min-free-bytes SIZE` | 10GiB | free space on the work directory's and the store's filesystems below which new work is refused; `0` turns the floor off |

A publish claims its bytes against the store maximum as it copies and gives
them back on every way it can end, so a write that would pass a limit fails
with a clear error, leaves no partial artifact (publishing stays atomic) and
frees its claim. Staged inputs are charged as they are copied; what the trainer
writes is measured before anything is published, and the whole directory is
removed when the job ends however it ends. A publish needs room for its whole
size even when the artifact turns out to be one the store already holds. The
quota flags need `--training`. The free-space floor is read with `statvfs` on
Unix and is not enforced elsewhere.

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
| `adapter_id` | the id on the adapter's card, an adapter model reference `<vendor>/<repo>:<owner>:<name>:<tag>` (any other id is refused before the run, since a server would not offer the adapter under it) |
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
 "grad_norm": 0.41, "seconds": 93.2, "eval_loss": 1.95,
 "pg": {"loss": ..., "clip_fraction": ..., "kl": ..., "ratio_mean": ..., "ratio_min": ...,
        "ratio_max": ..., "mean_advantage": ..., "scoring_seconds": ...}}
```

`eval_loss` is present on a step that evaluated a held-out set. `pg` is present for a policy-gradient run, with the teacher-advantage statistics
(`token_advantage_mean`, `token_advantage_max`, `token_advantage_clip_fraction`)
when records carry a teacher context. `seconds` is the time since training
began. Step frames over D-Bus are droppable like every progress frame: a
subscriber that does not read them loses frames, never the result.

The result is `outputs` (the report: `objective`, `adapter_digest`, the scores before and
after, `held_out_repeats` and `held_out_text_repeats` (held-out examples that are also trained on), the monitoring curve, `selected_step`, the policy-gradient totals,
`base_tier` and why; for `dpo`, `held_out_repeats` (held-out pairs that are also training pairs), `beta`, `nll_weight` and `train_score` and `held_out_score`, each
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
  is busy on every card that could hold the run, the run waits, but not
  indefinitely: once it has waited `--train-max-wait-ms` (default 60000) the
  cards it could use are held for it. New requests then go to other cards (or
  wait), and the batches on the held cards take no new requests, so the first
  held card whose running requests finish goes to the run. The run therefore
  starts within that wait plus the longest request already running on a held
  card, under any request traffic. Should it still not start by the admission
  deadline (`--admit-deadline-ms`; a card another process filled, say) it is
  refused (HTTP 429, a bus `error` frame), the hold ends, and the client
  retries.
- One run holds one card at a time, and several runs use several cards. The
  trainer is one card per run; a run that needs more than the largest card
  holds is refused at once, with what it needs.
- When a run ends, by completion, failure or cancel, it gives its card back at
  once, and its files are removed.
- A `train`, `full_train` or `pretrain` run never lands on the CPU backend
  silently. On a server started on the CPU (`--device cpu`, `--backend cpu`) it
  trains there and says so in the server log. Where the CPU would only be the
  fallback (the server sees no card, or the placer finds none with room), the
  run is refused with the reason unless the request sets `allow_cpu: true`: a
  card trains orders of magnitude faster.

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
