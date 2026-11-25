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
  on rollout records that carry a `reward` and the `group` it is compared
  within (training-record contract version 2; the job computes each rollout's
  advantage by `advantage_norm`), or a `teacher_context` for self-distillation (a Qwen3.8 base; see `brain::ChatFineTune::policy_gradient`
  for the record format and the objective's settings), or another model's scores of the sampled answers for on-policy
  distillation from that model (`distill_teacher`, below);
- **by preference optimization** (`objective: preference`): `dataset` and
  `held_out` hold `generic-preference-v1` pairs (a prompt and the `chosen` and
  `rejected` candidates, each one assistant turn or a continuation of turns that
  starts and ends with one, such as a tool call, its result and the answer), as `brain::PreferenceFineTune` trains them, by the pair
  loss `loss`: `dpo` (default), `ipo`, `simpo` (with `gamma`) or `orpo`. The
  reference of `dpo` and `ipo` is the policy the run starts from (the base, or the
  `start_adapter` continued), frozen, so no second model is held; `simpo` and `orpo`
  read none;
- **by feedback** (`objective: feedback`, KTO): `dataset` holds `generic-feedback-v1` judged completions (a prompt, one
  `completion`, a candidate as in a preference pair, and `desirable`), as `brain::FeedbackFineTune` trains them, against the same frozen reference; a step's
  `grad_accum` examples estimate the reference point, and the result reports `desirable_examples`,
  `undesirable_examples` and each kind's mean log-ratio to the start (`desirable_logratio`, `undesirable_logratio`);
- **by distillation** (`objective: distill`): the adapter is trained toward a teacher's `top_k` (default 16) most likely
  next tokens at every supervised position of `generic-messages-v2` records, as `brain::DistillFineTune` does. The
  teacher is a policy of the same base: `teacher: base` (default), `start` (the base with the `start_adapter`, which
  the student continues too) or `adapter:<blake3 digest>` (an adapter in the artifact store, verified like a dataset).
  It scores every record once before the first step. The result names the teacher (`teacher`,
  `teacher_adapter_sha256`) and the student's cross-entropy to it before and after (`initial_loss`, `final_loss`). A
  teacher served elsewhere (another base) is not offered here: its top-K log-probabilities would have to ride a recorded
  trace, which `policy_trace` does not carry. Distillation from another model on the student's own samples is the
  policy gradient's `distill_teacher`.

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
  `exit_entropy` (below), `validate_only`, and `cards`, `job` and `sharding` (below). Every setting the trainer
  would refuse without reading a weight (an architecture option, a `train_precision` the architecture does not run,
  an `exit_entropy` for a model without exit gates, an optimiser plan that does not hold, a sharding the precision
  cannot take, a run no sharding fits on the largest card) is refused at submission, a 400 or a bus error.
  `validate_only: true` makes those checks and answers `{"status": "valid", "arch", "steps", "grad_accum",
  "max_block", "lr", "weight_decay", "exit_entropy", "train_precision", "cards", "sharding", "device_bytes",
  "footprint"}` (`device_bytes` is what each card is charged) without training: host work that takes no card. The
  dataset's records are checked by `check_dataset`.
- **`full_train` on several cards and servers** takes `cards` and `job` exactly as `train` does (below): one
  data-parallel rank per claimed card, the same `seed` on every rank, rank 0 streaming the steps and answering with
  the report of a one-card run, whose `parallel` object says how it was spread; a server of a job that does not
  hold rank 0 answers `{"status": "completed", "objective": "full_sft", "job", "ranks": [first, last]}`. A run on
  several cards or servers resumes on the cards and sharding it began on ([Resuming a job](#resuming-a-job)). `sharding` says what the ranks split among themselves:
  `replicated` (each holds the fp32 weights and gradients), `optimizer` (ZeRO-1: each also holds its slice of the
  f32 master copy and AdamW moments on the card), `gradients` (ZeRO-2: those and the reduced gradient slice, half
  the gradient traffic) or `parameters` (ZeRO-3: the weights too, gathered layer by layer; it needs a resident
  `train_precision` with bf16 gradients and a layer window, such as `weights=bf16,grad=bf16,window=2`, and is
  refused at submission otherwise). Every claimed card is charged one rank's footprint under that sharding.
  The optimizer and gradient stages take f32 AdamW state only (another `train_precision` than `moments=host` is
  refused) and keep every gradient in the model's own buffers, so they spare host memory and the update, not the
  card: only `parameters` makes a rank smaller than a replica. Unnamed, the run takes the **first of `replicated`,
  `gradients`, `parameters` whose rank fits the server's largest card** and whose precision admits it (`optimizer`
  is never chosen: it holds what `gradients` holds and moves twice the gradient traffic; one rank is always
  replicated); when none fits, the request is refused with what each needs and what the card holds. With a `job`,
  the plan's `sharding` is the run's, and a named one must equal it.
- **`merge`** folds a LoRA adapter into the weights it targets (`W + alpha/r B A`). The adapter is the input blob
  `adapter` or `adapter_digest` (exactly one); the work is host-side and takes no card. Host-side work (a merge,
  `init_model`, `check_dataset`, a `validate_only` check) runs in host memory whatever `--device` the server was
  given, a server of one card included.

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

## Speech fine-tuning

**`speech_sft`** fine-tunes a Qwen3-TTS checkpoint (Talker and code predictor) on recorded speech, with new languages
and named voices, and exports it in the upstream layout of its base: the base's `model.safetensors` names and dtypes
(BF16 for the released checkpoints), its tokenizer, `generation_config.json` and `speech_tokenizer/`, and a
`config.json` that names the trained languages and voices, so brain and the upstream `qwen-tts` package load it
unchanged. A checkpoint with named voices is a `custom_voice` checkpoint (see
[Qwen3-TTS](../../reference/models/qwen3tts.md#trained-checkpoints)).

The datasets are training shards (`brain-tts-shard-v1`, one file of tokenised transcripts, `[frames, 16]` codec codes,
x-vectors, voice and language per utterance; the `encode` action of a Qwen3-TTS model produces each utterance's
arrays). Params:

| param | meaning |
|---|---|
| `base` | the Qwen3-TTS base, `<vendor>/<repo>` in the model store (required) |
| `dataset_digest` or the blob `dataset` | the training shard (exactly one of the two) |
| `held_out_digest` or the blob `held_out` | a held-out shard (at most one), scored every `eval_every` steps |
| `languages` | JSON object naming every language the shards hold (required): `{"<name>": {"new": true, "init_from": ["english", ...]}}` for one the base lacks, its codec row starting at the mean of those (every base language when omitted; every new language of a run names the same `init_from`), or `{"<name>": {"new": false}}` for one of the base's. Names are 1 to 64 lowercase `[a-z0-9_-]`; at most 256, the document at most 64 KiB |
| `voices` | JSON object naming every voice the shards train (default `{}`): `{"<name>": {"new": true}}`, its row initialised from its speaker's x-vectors, or `{"new": false}` for one of the base's |
| `steps` or `epochs` | the run's length: optimiser steps, or passes over the training shard (exactly one) |
| `lr`, `weight_decay`, `warmup`, `schedule`, `grad_clip` | the optimiser: peak rate, AdamW decay, steps of linear warmup, `cosine` (to a tenth by the last step) or `constant` after it, the global gradient-norm clip |
| `batch_utterances`, `grad_accum` | utterances per micro-batch, micro-batches per optimiser step |
| `max_frames` | the most codec frames a record may hold: a longer one is refused by name, never cut; it sizes the build |
| `residual_weight` | the weight of the code predictor's loss beside the Talker's (upstream's 0.3) |
| `streaming_fraction` | the share of drawn records trained in the streaming text layout; the rest train non-streaming |
| `talker_lr_multiplier`, `code_predictor_lr_multiplier`, `codec_table_lr_multiplier`, `new_row_lr_multiplier` | the learning-rate multiplier of the Talker decoder and head, the code predictor, the codec rows the base trained, and the new languages' and voices' rows |
| `voice_outlier_fraction`, `free_row_fraction` | the share of a new voice's x-vectors left out of the centroid its row starts at, and the norm (as a share of the median named row's) below which a codec row is free for a new name |
| `eval_every`, `eval_batches` | score the held-out shard every this many steps, over this many micro-batches (0: every record) |
| `patience` | stop once the held-out loss has not improved for this many evaluations, and publish the checkpoint of the best evaluation (needs `eval_every`) |
| `keep_best` | publish the checkpoint of the evaluation with the lowest held-out loss rather than the last step's, still running every step (implied by `patience`; needs `eval_every`) |
| `length_bucket_steps` | micro-batches whose utterances are regrouped by length so each pads little (default 16; 1 draws them unsorted) |
| `seed` | the batch order and the layout draws (random when omitted; reported) |
| `gt_weight` | the weight of the ground-truth objective (`talker_ce + residual_weight * code_predictor_ce`) in the loss (default 1; 0 learns from the teacher alone) |
| `teacher` | distil from this frozen Qwen3-TTS checkpoint, `<vendor>/<repo>` in the model store (see below) |
| `talker_kd_weight` | the weight of the Talker's codebook-0 KL to the teacher, times the squared temperature (default 1; needs `teacher`) |
| `mtp_kd_weight` | the weight of the code predictor's residual-codebook KLs to the teacher, each a mean over the frames, summed over the codebooks, times the squared temperature (default 0.02, `residual_weight`'s 0.3 spread over 15 codebooks; needs `teacher`) |
| `hidden_kd_weight`, `hidden_kd_layers` | the weight of matching the Talker's residual stream to the teacher's through a learned linear projection per pair, and the pairs as a JSON list `[[student layer, teacher layer], ...]` (the stream entering the layer; the layer count names the stream after the last layer). Off by default; the projections are never exported |
| `kd_temperature` | the softmax temperature of both KLs (default 1) |
| `cards`, `job` | one data-parallel rank per card of this server, and this server's part of a job over several servers, as for `train` ([One job over several servers](#one-job-over-several-servers)) |
| `install_as`, `resume_key`, `checkpoint_every`, `allow_cpu` | as for `full_train` |
| `validate_only` | stage and check the shards (format, codebooks and x-vector width against the base, every language and voice named) and the names against the base, and answer `{"status": "valid", "records", "held_out_records", "languages", "voices", "device_bytes"}` without training: host work |

Every setting left out is the objective's own default, which `GET /v1/capabilities` lists as the param's
`implied_default`.

**Distillation.** With a `teacher`, every micro-batch is also run through the teacher, teacher forced on the same
histories, and the student learns its distributions: the loss is `gt_weight * (ground truth) + talker_kd_weight * T^2 *
KL(teacher || student)` over the Talker's supervised rows `+ mtp_kd_weight * T^2 * sum of the residual codebooks' KLs`
`+ hidden_kd_weight * the mapped pairs' mean squared errors`. The teacher is held frozen at bf16 beside the student on
every rank's card (no gradient, no optimizer state), so a run on several cards or servers distils exactly as it
trains. The student is the `base`: a checkpoint of the same upstream layout, which may be narrower, shallower or have a
trimmed text vocabulary, exported in its own shapes. Every student text token must be a token of the teacher's
tokenizer (`vocab.json` and the added tokens of `tokenizer_config.json`), its text specials the teacher's under that
map and its codec specials the same rows; the teacher must have every language and voice the shards train (at any
row). A record in the x-vector slot needs a teacher of the student's width. Each is refused by name before training.
The held-out score (`eval_loss`, what `patience` and `keep_best` select on) is then the distillation's terms alone,
the distance to the teacher; step frames and the report's `last_step` add `talker_kl`, `code_predictor_kl` and
`hidden_mse` (held out: `held_out_talker_kl`, ...), and the report names the `teacher`. A rank is charged the teacher's
parameters, its activations and the logits of both models beside its own.

The languages and voices named are exactly the shards' own. A language or voice the request calls new that the base already names, one it calls the base's that the base lacks,
an `init_from` that is not a base language, a shard language or voice the request does not name, and a shard of
another codebook count or x-vector width are refused by name. Each rank is charged four copies of the Talker's and the
code predictor's f32 weights, a workspace and the activations of a micro-batch of `batch_utterances` records of
`max_frames` frames on its card (`validate_only` answers it as `device_bytes`), so a batch the card cannot hold is
refused at the door. Step frames carry `step`, `steps`, `loss`
(`talker_ce + residual_weight * code_predictor_ce`), `talker_ce`, `code_predictor_ce`, `eos_accuracy` (end-of-speech
targets the Talker's greedy pick got right), `frames` and `frames_per_s`, `grad_norm`, `lr`, `seconds`, and after an
evaluation `eval_loss`, `held_out_talker_ce`, `held_out_code_predictor_ce` and `held_out_eos_accuracy`. The result is
the report (`objective: "speech_sft"`, `base`, `seed`, `initial_loss`, `final_loss`, `steps_completed`,
`evaluations`, `last_step` with the measurements above, `held_out_curve` (each evaluation's `step`, `train_loss` and
`held_out_loss`), `checkpoint` (`"last"`, or `"best"` when the run selected on held-out loss, with `best`: its `step`
and `held_out_loss`), `stopped_early`, `languages` and `voices` with the codec row of each,
`tts_model_type`, `trained_tensors`, and on several cards `parallel`: the group as `train` reports it), `checkpoint_digest` and `files`, and with `install_as` the installed id. Installed as
`local/<name>`, the checkpoint is served (from the server's next start) as a model of its own under that id:
`POST /v1/audio/speech` with `"model": "local/<name>"`, `"voice": "<voice>"` and `"language": "<language>"`.

## Resuming a job

`train` (every objective: `sft`, `policy_gradient`, `preference`, `feedback`, `distill`), `init_model`, `pretrain`, `full_train` and `speech_sft` take a
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
- A run on several cards (`cards`) or servers (`job`) keeps each rank's state apart, in numbered generations every
  rank commits together (each rank its file, then rank 0's manifest once every rank's is durable), and resumes from
  the newest one every rank committed. It resumes only on the layout it began on: the same number of ranks, the same
  ranks on this server and the same `sharding`, since each rank's state is its own shard of the records (and of the
  optimizer); the same key on another layout is refused naming both (`its run began on 2 rank(s), ranks 0..2 on this
  server, sharding gradients, and this request asks 1 rank(s), ...`).
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
for `generic-preference-v1` pairs, `feedback` for `generic-feedback-v1` judged completions, `timeline` for `timeline-v1` subjects, each line validated and no token counted), the dataset as the input blob `dataset` or as `dataset_digest` (exactly one),
`thinking` and `pack_answers` (rendered as a run with those settings would), `truncated` (`mask` default or `keep`: the
rules between rollout records read as a policy-gradient run with that setting would, a kept truncated rollout counting
in its group), `max_block`, and `per_record` (with
`offset` and `limit`) to also list each record's token counts, or its refusal, one page at a time. The result:

| field | meaning |
|---|---|
| `records`, `valid_records` | non-blank lines, and those that parse and encode |
| `examples` | what the valid records are trained as (a record of several answers can be several examples; a pair is two) |
| `tokens`, `supervised_tokens` | per example: `total`, `min`, `mean`, `p50`, `p90`, `p99`, `max`, in the base's tokens |
| `examples_without_supervision` | examples that would train on nothing |
| `over_max_block`, `over_max_block_lines` | examples longer than `max_block` (a run refuses them rather than cutting them) and the first lines holding one |
| `invalid_total`, `invalid` | records the trainer refuses, and the first 20 as `{line, error, rule, field}`: `rule` names the rule of the training-record contract (version 2) the record broke (`behavior_logprob_count`: `behavior_logprobs` does not hold one value per token of its answer; `behavior_top_count` and `behavior_top`: `behavior_top_ids` and `behavior_top_logprobs` (the sampler's top-K per token) do not number the answer's tokens, or are not one distribution's top-K (a ragged K, a repeated id, a value above 0, a mass above 1, one without the other or without `behavior_logprobs`); `behavior_support_count` and `behavior_support`: `behavior_support_offsets`, `behavior_support_ids` and `behavior_support_logprobs` (a cut sampler's support per token) do not number the answer's tokens, or are not a support (offsets that do not frame the ids or leave a token empty, a repeated id, more than 1024 ids, the sampled token outside its support, a value above 0 or below the token's `behavior_logprobs`, one without the others or without `behavior_logprobs`); `token_ids_mismatch`: `token_ids` neither are the tokens the template renders for the answer nor read back as that answer; `prompt_token_ids_mismatch`: `prompt_token_ids` are not the context the template renders before the answer; `missing_reward`: a rollout field, or a `group`, on a record with no `reward`, `teacher_context` or `teacher_logprobs`; `teacher_scores`: a teacher model's `teacher_logprobs` or `teacher_logprob` stated so no distillation estimator can use them (both forms, a score on some answers only, beside a `teacher_context`, or a per-answer score without `behavior_logprobs`, `reward` and `group`, or with a `rollout` key); `teacher_logprob_count`: `teacher_logprobs` does not hold one value per token of its answer; `missing_group`: a `reward` or `rollout` without a `group`; `advantage_removed`: version 1's precomputed `advantage`; `loss_weight`: a message's `loss_weight` that is not a finite number above 0, on a message that is not supervised, or on a rollout record; and, between the records of the file, `rollout_mismatch`: two records of one `rollout` with different `reward` or `truncated`; `group_of_one`: a group with one rollout that was not truncated; `tokenizer_mismatch`: a `tokenizer_digest` that is not the base's own tokenizer's; `mixed_tokenizer_digest`: one other than an earlier record names, where the base's is not known; `baseline_trains`: a `baseline_only` record that trains something) and `field` the record field, as a JSON path (`messages[1].behavior_logprobs`); both are `null` for any other refusal. A record that breaks a rule of its own is refused for that. Example records of the contract, valid and refused, are in brain's source tree under `crates/data/tests/fixtures/training-records/v2` |
| `per_record`, `per_record_refused`, `per_record_next` | with `per_record` only, one page of the records (non-blank lines; `offset` counts valid and refused alike). `per_record` holds `[line, tokens, prompt, trained, examples]` for each valid record of the page: `prompt` and `trained` are the unsupervised and the supervised tokens of its longest example, counted in the base's tokenizer and chat template exactly as training encodes it (so `prompt + trained` is that example's length), `tokens` is what the training row has to hold (equal to `prompt + trained` unless a teacher's rendering is longer), `examples` how many examples the record is trained as. `per_record_refused` holds the refusals of the page as `{line, error, rule, field}` (as `invalid`, but not capped at 20; `error` at most 512 characters). `per_record_next` is the `offset` of the next page, `null` after the last. A page holds at most `limit` records, counts and refusals together (default 50000, at most 200000; a larger `limit` is refused as a bad request naming it), so the response is bounded whatever the dataset size; the whole file is read for every page, so the totals above are the same on each. |

## Serving it

```
brain serve --models-dir /models --training --artifact-store /shared/artifacts \
    --openai 0.0.0.0:8788 --api-key-file /etc/brain/key --dbus \
    --tls-cert /etc/brain/tls/server.crt --tls-key /etc/brain/tls/server.key
```

A surface on an address other than loopback needs TLS (see
[Transport security](../serving/serving.md#transport-security)).

`--training` turns the workload on; it is off otherwise. `--artifact-store DIR`
names the directory bulk data moves through (below); it must exist.
`--train-job-secret FILE` and `--train-job-coordinator HOST:PORT` let the
server run ranks of a job spread over several servers ([below](#one-job-over-several-servers)). The base is
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
| `objective` | `sft` (default), `policy_gradient`, `preference`, `feedback` or `distill` |
| `steps`, `grad_accum`, `rank`, `alpha`, `lr`, `weight_decay`, `seed` | the optimiser's settings; what is not named is the trainer's own default, which `GET /v1/capabilities` lists as the param's `implied_default` |
| `strict_updates` | abort at the first optimiser step whose loss or gradient is not finite (default off: such a step is skipped, the weights untouched, and the run aborts on the third in a row) |
| `max_block` | the longest training row in tokens (default 8192). The memory reserved for the run is sized to it and a longer record is refused by name |
| `cards` | how many of the server's cards the run trains on at once (default 1, at most the cards the server has). Each card runs one data-parallel rank holding the run's whole footprint and its own share of the records, all with the same `seed` (one is drawn for all of them when the request names none), and every step's gradients are reduced over the cards, so each step sees `cards` times the records of a one-card step. The result is the same document as on one card, from rank 0; the same request trains the same adapter, byte for byte, though not the adapter of a one-card run. Every objective but `feedback`, which is refused by name (see [why](#which-objectives-train-as-ranks)) |
| `job` | makes the run this server's part of one data-parallel job spread over several servers (below); every objective but `feedback`, with `seed` |
| `thinking`, `pack_answers` | how records are rendered |
| `base_format` | how the frozen base is held: `bf16`, `fp8`, `nf4`, ...; default the tier that fits the card |
| `eval_every`, `patience`, `replay_share`, `keep_evaluations` | monitoring and early stopping |
| `adapter_id` | the id on the adapter's card, an adapter model reference `<vendor>/<repo>:<owner>:<name>:<tag>` (any other id is refused before the run, since a server would not offer the adapter under it) |
| `return_adapter` | also return the adapter as an output blob (D-Bus) |
| `algorithm`, `clip_eps`, `kl_beta`, `reference`, `behaviour`, `loss_agg`, `loss_agg_length`, `truncated`, `advantage_floor`, `temperature`, `teacher`, `advantage_bound`, `sequence_advantage_cap`, `advantage_norm`, `estimator`, `centring_low`, `centring_high` | the policy-gradient objective's settings (only with `objective: policy_gradient`). Each one's `implied_default` in `GET /v1/capabilities` is what the trainer applies when it is omitted (`brain::PolicyGradientConfig::default()`); one without (`clip_eps_high`, `max_step_kl`, `max_step_grad_norm`, `advantage_bound`, `sequence_advantage_cap`) is off unless named, and `loss_agg_length` defaults to the request's `max_block` |
| `algorithm` | a named policy-gradient algorithm whose settings are the run's defaults: `grpo` (`z-score`, `seq-mean`, a symmetric clip band, the KL term), `dr-grpo` (`mean-centred`, `constant-length` over `max_block`, `kl_beta` 0), `dapo` (`z-score`, `clip_eps_high` 0.28, `token-sum`, `truncated: mask`, `kl_beta` 0) or `rloo` (`leave-one-out`, `token-sum`). Every setting named beside it overrides it (`dr-grpo` takes `loss_agg_length` alone). The result names the algorithm (`algorithm`) and every setting the run trained with (`policy_gradient.settings`) |
| (record) `policy_version`, `sampler` | on a rollout record: the oldest policy version behind it (an integer) and the sampling controls the decoder applied (`temperature`, `top_k`, `top_p`, `min_p`); kept per example in the prepared split, see the SDK reference |
| (record) `baseline_only` | a rollout record with `"baseline_only": true` (and its `reward` and `group`, no message `"train": true`) sets its group's baseline and is never trained; `check_dataset` counts it as a valid record of no example and refuses one that trains something by rule `baseline_trains` |
| `advantage_floor` | `none` (default) or `zero`: every advantage is clamped at zero after it is computed against its whole group, so a rollout under the group's baseline still sets the baseline but carries no policy gradient (a positive-only run; with `kl_beta` above 0 the KL term still reaches its tokens). No named algorithm sets it |
| `truncated` | what a rollout record marked `"truncated": true` does: `mask` (default; never drawn, scored or trained, and its reward takes no part in its group's advantages: DAPO's overlong filtering) or `keep` (trained like any rollout, the reward its collector gave it counted in its group) |
| `loss_agg`, `loss_agg_length` | how a step's token terms are divided: `seq-mean` (default; each sequence by its own token count, a batch of one sequence), `token-sum` (every token of a step by the step's token count) or `constant-length` (each sequence's sum by `loss_agg_length` tokens, default `max_block`, averaged over the sequences: Dr. GRPO, whose gradient is not rescaled by a sequence's length). `loss_agg_length` with another aggregation is refused |
| `advantage_norm` | how each group's rewards become the advantages of its rollouts: `mean-centred` (default, `r - mean`), `z-score` (`(r - mean) / (std + 1e-4)`) or `leave-one-out` (`r_i` less the mean of the others). The records the run trains on (the dataset and its replay sets together) are held to the contract's rules between records: a group of one, two records of one rollout that disagree, or two tokenizer digests fail the job naming the rule and the example |
| `behaviour` | where the importance ratio's behaviour log-probabilities come from. `recorded` (default): the sampler's own `behavior_logprobs`, on every supervised token of every record with a reward; a dataset missing any is refused. `start_of_run_proxy`: the start-of-run policy's where none was recorded, exact only if the rollouts were sampled from it; the training record states which was used |
| `beta`, `nll_weight`, `keep_reasoning` | the preference objective's settings (only with `objective: preference`; `beta` and `keep_reasoning` with `feedback` too): the loss temperature (default 0.1), the weight of the supervised likelihood of the chosen answer added to it (default 0) and whether the reasoning of each candidate is trained on |
| `distill_teacher` | with `objective: policy_gradient`, on-policy distillation from another model: `<vendor>/<repo>` in the server's model store, the teacher whose scores of the student's sampled answers the records carry (`teacher_logprobs` per token, `teacher_logprob` per answer; the `score` action of that model gives both). Only its tokenizer is read and compared with the base's: per-token scores train only when the two are identical and are refused otherwise; per-answer scores (each rollout's `log q(text) - log p(text)` against the leave-one-out mean of its group, added to its reward advantage) train with any teacher and are the higher-variance path. A teacher not in the store is refused at submission. The result's `policy_gradient.settings` name the comparison (`teacher_tokenizer`) and the estimator (`distill_estimator`) |
| `teacher`, `top_k` | with `objective: distill`, the teacher (`base`, `start` or `adapter:<blake3 digest>`) and the tokens it keeps per position (1 to 256, default 16); `teacher` is also the policy gradient's (`base` or `start`). A distillation refuses the policy-gradient and preference settings, a held-out, monitoring or replay set, and the monitoring settings by name |
| `loss`, `gamma` | the preference objective's pair loss (only with `objective: preference`): `dpo` (default), `ipo` (the per-token mean log-ratio margin regressed to `1 / (2 beta)`), `simpo` (length-normalised, reference-free, target margin `gamma`, default 0.5) or `orpo` (the chosen answer's likelihood plus `beta` times its odds ratio over the rejected one, reference-free); `gamma` with another loss is refused |
| `desirable_weight`, `undesirable_weight` | the feedback objective's weights of a desirable and of an undesirable example's term (default 1 each; only with `objective: feedback`) |
| `arch` | the base's architecture, only for a checkpoint directory with no `config.json` |

`data` is `{"train": SRC, "held_out"?: SRC, "held_out_text"?: SRC, "monitor"?:
SRC, "start_adapter"?: SRC, "replay"?: [SRC], "weighted_replay"?: [{SRC fields,
"share"}]}` where `SRC` is `{"digest": "<blake3>"}` or `{"blob": "<input
name>"}`. Unknown fields anywhere are refused, as is a blob that is named but
not attached or attached but not used.

A preference or feedback run takes only what it uses (a feedback run takes no `held_out` either): `held_out_text`, `monitor`, replay sets, `replay_share`,
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

A run is detached from the client that started it: closing the event stream,
or the client exiting, does not stop it (it keeps its card and publishes its
adapter when it ends). A client that is interrupted and wants the run stopped
cancels it explicitly; one that lost the id finds it in `GET /v1/jobs` (HTTP)
or `Jobs()` (D-Bus), which list the caller's own jobs.

A frame is `{"type": "stage", "stage": ...}` while the datasets are staged and
the base is loaded, then one per optimiser step:

```
{"type": "step", "step": 7, "steps": 100, "loss": 1.82, "lr": 0.0003,
 "grad_norm": 0.41, "seconds": 93.2, "eval_loss": 1.95,
 "pg": {"loss": ..., "clip_fraction": ..., "kl": ..., "ratio_mean": ..., "ratio_min": ...,
        "ratio_max": ..., "mean_advantage": ..., "scoring_seconds": ...}}
```

`loss` is the step's training loss, the per-token mean over the step's records; on several cards it is over every
rank's records (what one card accumulating them would report), not the first rank's share. `eval_loss` is present on
a step that evaluated a held-out set. `pg` is present for a policy-gradient run, with the teacher-advantage statistics
(`token_advantage_mean`, `token_advantage_max`, `token_advantage_clip_fraction`)
when records carry a teacher context. `seconds` is the time since training
began. Step frames over D-Bus are droppable like every progress frame: a
subscriber that does not read them loses frames, never the result.

The result is `outputs` (the report: `objective`, `adapter_digest`, the scores before and
after, `held_out_repeats` and `held_out_text_repeats` (held-out examples that are also trained on), the monitoring curve, `selected_step`, the policy-gradient totals,
`base_tier` and why; for `preference`, `loss` (and `gamma`), `held_out_repeats` (held-out pairs that are also training pairs), `beta`, `nll_weight` and `train_score` and `held_out_score`, each
`{accuracy, mean_margin, pairs, skipped}`: the share of pairs the tuned policy prefers the chosen
answer on more than its reference does, and the mean margin in nats). The trainer's own training record, which holds every
step's statistics, is published to the artifact store beside the adapter
(`record_digest`), and returned as the blob `record` when the call sets
`return_adapter` or the server has no store.

Every chat fine-tune report but `feedback`'s, adapter or full (`objective` `full_sft`), and the
`report.json` `brain finetune --out` writes, has a `parallel` object that
records how the run was spread, as it ran. Fields that were not observed are
`null`:

| Field | Meaning |
|---|---|
| `world_size` | ranks in the whole group, across every server (1 for a run that is not data-parallel) |
| `ranks` | `[first, end)`: the ranks this server ran |
| `devices` | one `{rank, device, name}` per rank this server ran: `device` is `gpu<N>` or `cpu`, `name` is the card's name as its driver reports it |
| `collective` | what summed the gradients: `nccl` on the cards, or `host` through the host transport; `null` without a group |
| `transport` | how the ranks reached each other on the host side: `in_process` (threads of one server) or `tcp` (sealed TCP between processes or servers) |
| `network` | the network NCCL picked between nodes; always `null` for now, because NCCL does not report which transport a communicator chose |
| `sharding` | what the group sharded: `replicated`, `optimizer`, `gradients` or `parameters`; `null` without a group |

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
- A run holds the cards it asked for (`cards`, default one), and several runs
  use several cards. Each card must hold one rank's footprint (a `full_train`
  rank's under its sharding); a run that needs more than the largest card holds
  is refused at once, with what it needs, and so is one asking for more cards
  than the server has. A run on
  several cards takes them all at once: the cards with room, then cards that
  evicting idle instances frees, and nothing is evicted while the cards it
  needs cannot all be had (a card busy serving); it waits as a one-card run
  waits.
- When a run ends, by completion, failure or cancel, it gives its cards back at
  once, and its files are removed. On several cards a cancel stops every rank
  after the same step, and a rank that fails fails the run at once: the others
  stop in their current reduction (on the cards too) with its reason.

### One job over several servers

An orchestrator runs one training job over the cards of several servers (on
different machines) by sending each server the same `train` request with a
`job` document naming that server's part:

```json
{"plan": {"job": "run-17", "world_size": 8, "tp": 1, "pp": 1, "dp": 8,
          "coordinator": "10.0.0.1:29500", "handshake_timeout_ms": 600000, "op_timeout_ms": 600000},
 "first_rank": 4, "secret_sha256": "<64 hex digits>"}
```

- `plan` is the job plan every server gets unchanged (the `--job-plan` file of
  `brain finetune`): data-parallel only (`tp` and `pp` 1, `dp` the world size;
  `sharding` replicated for `train`, any for `full_train`), optionally `collectives` and `bucket_mib`. At most
  4096 ranks, timeouts at most a day, the document at most 4 KiB.
- The server runs ranks `first_rank` to `first_rank + cards - 1`, one per card
  it claims; the ranges of all servers cover the plan's ranks once.
- `secret_sha256` is the SHA-256 of the job secret's 32 bytes
  (`xxd -r -p secret | sha256sum`). The secret itself never travels over the
  API: each server is started with it, `--train-job-secret FILE` (the
  owner-only file `brain finetune --job-secret` reads; the orchestrator puts it
  on each machine out of band), and refuses a job whose digest is not its
  secret's. A server started without it refuses every `job`.
- The server holding rank 0 binds the plan's `coordinator`, which must be the
  address it was started with, `--train-job-coordinator HOST:PORT`; every other
  rank connects to it. Every connection proves the secret and is sealed, as for
  a launched job. A server without `--train-job-coordinator` runs only a job's
  other ranks.
- Every server's ranks join within the plan's `handshake_timeout_ms` (each
  server's run waits for its cards first, so leave room), or the job fails on
  every server and each gives its cards back. A rank that fails, or a server
  whose run is cancelled, fails the job: the ranks on that server stop at once,
  and the other servers' ranks within `op_timeout_ms`.
- The server holding rank 0 answers with the run's result and publishes the
  adapter as a one-server run does. Every other server answers
  `{"status": "completed", "objective", "job", "ranks": [first, last],
  "adapter_sha256"}` and publishes nothing; every rank holds the same adapter.
- `seed` is required (every rank must start from the same initialisation),
  and so is the same `data` on every server, each naming datasets its own store
  or the call holds.
- With a `resume_key` (sent to every server) each server keeps its own ranks'
  state in its own artifact store, every `checkpoint_every` steps, as one
  generation the ranks of every server commit together. After the job failed
  (a server died, or a run was cancelled, which fails the job on every
  server), sending every server its same request again resumes the job from
  the newest generation every rank committed, and it trains the adapter the
  uninterrupted job trains. A server whose part completed answers its
  recorded result without joining the job again; a server that completed
  while another did not record its completion leaves that one to fail at the
  handshake, and its key must then be replaced on every server.
- A `train`, `full_train` or `pretrain` run never lands on the CPU backend
  silently. On a server started on the CPU (`--device cpu`, `--backend cpu`) it
  trains there and says so in the server log. Where the CPU would only be the
  fallback (the server sees no card, or the placer finds none with room), the
  run is refused with the reason unless the request sets `allow_cpu: true`: a
  card trains orders of magnitude faster.

### Which objectives train as ranks

Data parallelism changes nothing in what a step optimises only when the objective
is a mean of terms that each depend on one record: every rank then computes the
gradient of its own records' terms, and their weighted sum over the group is the
gradient of the whole batch.

- `sft`, `policy_gradient` and `full_train` are such sums (a token's or a
  rollout's loss; a group's advantages are computed from the records' rewards
  before training) and train as ranks.
- `preference` (DPO, IPO, SimPO, ORPO: a loss per pair, the frozen reference
  being the starting policy every rank holds) and `distill` (a KL per token to
  the teacher) are such sums too, with each pair or record as the unit, and
  train as ranks: rank `r` of `N` trains the `n`-th pair or record drawn when
  `n % N == r`, so a step of `N` ranks trains what one card accumulating `N`
  pairs or records (`grad_accum`) trains. Every rank scores the frozen
  reference of every pair, or has the teacher score every record, before the
  first step: that pass takes as long on `N` cards as on one.
- `speech_sft` is such a sum (each utterance's teacher-forced losses) and trains as ranks: rank `r` of `N` trains
  the `n`-th record drawn when `n % N == r` (within each length bucket), and rank 0 scores the held-out shard for
  every rank. On several cards a run with a patience or `keep_best` holds the best evaluation's parameters on every
  rank, in host memory (about 6.5 GB for the 1.7B). The server of rank 0 publishes the checkpoint; every other server
  answers `{"status": "completed", "objective": "speech_sft", "job", "ranks": [first, last], "steps_completed"}`.
- `feedback` (KTO) is not: its reference point is a KL estimate over the step's
  batch, from mismatched pairs of that batch, so a rank holding a shard would
  estimate a different reference point and optimise a different objective unless
  the estimate itself were summed over the group first. It is refused by name.

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

## Training while serving: a streaming job

`brain/train` with `objective: policy_gradient` and a `stream` document instead of `data` trains on rollout groups
an orchestrator sends while the job runs, and serves each new adapter version from the same process. Any architecture
whose trainer declares the policy gradient takes it. The server must serve the base with an adapter bank
(`--adapter-slots`, `--serve-adapters` or another adapter flag), and the job is started over HTTP or D-Bus, the
surfaces that deliver its input.

```
POST /v1/jobs
{"model": "brain/train", "action": "train", "params": {
   "base": "Qwen/Qwen3-4B", "objective": "policy_gradient", "loss_agg": "token-sum", "max_block": 4096,
   "resume_key": "run-7", "checkpoint_every": 4,
   "stream": "{\"publish_adapter\": \"Qwen/Qwen3-4B:lab:policy:v1\", \"publish_every\": 1,
               \"step_tokens\": 16384, \"queue_tokens\": 65536, \"max_staleness\": 2}"}}
```

| `stream` field | Meaning |
|---|---|
| `publish_adapter` | the adapter id every version is served under (`<vendor>/<repo>:<owner>:<name>:<tag>`) |
| `publish_every` | steps between publications; the final adapter is published when the job ends |
| `step_tokens` | policy tokens (supervised tokens of rollouts) one optimiser step trains: a step takes whole groups, in arrival order, until it holds at least this many |
| `queue_tokens` | policy tokens the queue holds, at least `step_tokens` |
| `max_staleness` | optional: a group whose oldest record's `policy_version` is more than this many versions behind the last published one is dropped, not trained |
| `max_steps` | optional: the job ends after this many steps and the learning rate decays over them; unset, the job runs until `finish` and keeps its peak rate after a warmup of at most 10 steps |
| `top_k` | with `estimator: score-centred` (required then, refused otherwise): the sampler top-K every record carries (`behavior_top_ids`, `behavior_top_logprobs`), at most 256 |
| `support` | optional, 1 to 1024: the widest sampler support a record may carry per token (`behavior_support_*`, from a policy trace with `support`); the trainer renormalises its log-probabilities over each token's support. Records with supports are refused without it, and so is one wider than it |
| `start_adapter` | optional: `{"digest": ...}` or `{"blob": "adapter"}`, the adapter the run continues |

The estimator and its settings are the request's policy-gradient params, as for a dataset: the clipped surrogate,
or `score-centred` over the records' sampler top-K. A step is normalised over
its own tokens, so `loss_agg` is `token-sum` or `constant-length`; `behaviour` is `recorded` (every policy token
carries the sampler's `behavior_logprobs`); importance correction, teacher contexts, images, `cards`, `job`,
`grad_accum` above 1, evaluations and `adapter_id` are refused.

**Records.** `POST /v1/jobs/{id}/records[?epoch=N]` (D-Bus `Input(job, "records", query, fd)`) takes a JSONL body
of training-record contract v2 rollout records; a request's records form complete groups (every record of a group key
is in the one request). They are checked as a dataset is, the base's tokenizer digest included, and each group's
advantages are computed when it arrives. The answer:

```
200 {"accepted": ["g1"], "duplicates": ["g0"], "zero_advantage": 0, "queued_groups": 3, "queued_tokens": 9120,
     "allowance": 56416, "epoch": 2, "version": 5}
```

A group key the run already took (queued, trained or dropped) is a `duplicate`: re-sending is idempotent. A group
whose rewards are all equal is accepted, counted in `zero_advantage`, and trains nothing. A full queue answers
`429` with `Retry-After` and `"detail": {"allowance", "queued_tokens", "offered_tokens", "epoch"}`: nothing of the
request is queued. It is never `429` while the queue holds less than `step_tokens`. A record the contract refuses is
`400` naming it; `?epoch=N` naming another epoch than the run's, or records after `finish`, is `409` with
`"detail": {"epoch"}`; a job that has ended is `410`, and an id the server does not know (it restarted) is `404`.
Records sent before the job is ready (it is loading the base, or a resumed run is restoring its state) wait for it;
a job that has not answered within two minutes is `503` with `Retry-After` (retry the same request).

**Finish.** `POST /v1/jobs/{id}/finish` (D-Bus `Input(job, "finish", "{}", empty fd)`) answers
`{"finishing": true, "queued_groups", "queued_tokens", "epoch"}`; the job trains what is queued, publishes the final
adapter to the bank and the artifact store, and ends with the result a dataset run reports. `DELETE` cancels as for
every job: the run writes its state and stops.

**Events.** Beside the `step` frames, whose `pg` object also carries `staleness_mean`, `staleness_max`,
`policy_tokens`, `learner_wait_seconds` (the time the learner waited for the step's records), `queue_groups`,
`queue_tokens`, `time_in_queue_seconds` (the step's groups' mean) and `dropped_stale`, the job reports:

```
{"type": "stream", "run": "my-run", "epoch": 2, "tokenizer_digest": "sha256:...", "publish_adapter": "...", "version": 5, "step_tokens": 16384, "queue_tokens": 65536, "max_staleness": 2}
{"type": "applied", "step": 7, "groups": ["g1", "g2"]}                 (+ "skipped": why, when a guard skipped the update)
{"type": "published", "version": 6, "digest": "sha256:...", "step": 7,
 "lineage": {"run": "my-run", "epoch": 2, "update": 7, "base_digest": "sha256:...", "adapter_digest": "sha256:..."}}
{"type": "dropped", "reason": "stale", "groups": ["g0"]}
{"type": "committed", "step": 8, "groups": ["g1", "g2", "g0"]}
```

`applied` follows the step whose optimiser update holds the groups' gradient; `committed` follows the write of a
training state that holds it (every `checkpoint_every` steps of a `resume_key` run, a cancel, and the job's
completion); a version is `published` once a request for `publish_adapter` is routed to it, and staleness is measured
against it. `version` counts the releases this server offered under the id; `lineage` names the weights for good: the
run (`resume_key`, or `job:<hex>` drawn for a run without one, as the `stream` frame says), the job of it (`epoch`), the
optimiser update (`update`, the step), the base and the file. The adapter's card records it, so every reply the
version decodes names it under `brain.adapter.lineage` (and `policy_trace.policy.adapter.lineage`), on this server or
any other that serves the file.

**Resuming.** Started again with the same `resume_key`, the run continues from its last saved state at the next
epoch (the `stream` frame and every answer carry it). The queue is not kept: the client re-sends every group it has
no `committed` receipt for, and the run answers the committed ones as duplicates. A client that names the epoch it
believes in (`?epoch=N`) learns at its next call that another job took the run over.

## Security

The surface is behind the server's key like every route and takes no parameter
that names a file: the work directory, the model store and the artifact store
are the operator's. The artifact store is the one place a client can put bytes
the server will read, and it reads a file only by a digest it has verified. Run
the store on a directory only the clients you trust can write to. See
[`docs/guides/serving/security.md`](../serving/security.md).
