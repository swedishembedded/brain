# LoRA fine-tuning (Qwen)

brain can fine-tune a Qwen3 model on your own chat-style data using LoRA
(Low-Rank Adaptation), producing a small adapter rather than a full copy of
the model's weights.

Only LoRA fine-tuning is supported today, not full-parameter retraining or
preference-optimization methods like DPO.

## Running a fine-tune

```
brain qwen3 finetune --lora RANK --weights BASE --adapter OWNER/NAME[:TAG] --dataset DIR
```

- `--lora RANK` - the LoRA rank (higher = more capacity, more adapter
  parameters).
- `--weights BASE` - the base model to fine-tune from, either a model-store
  reference or a direct path to a checkpoint.
- `--adapter OWNER/NAME[:TAG]` - where to store the resulting adapter.
  Running the same command again with the same `--adapter` overwrites that
  tag in place, so retraining is just rerunning the command.
- `--dataset DIR` - a directory of chat-template-driven supervised
  fine-tuning (SFT) data. Each training example is a multi-turn
  conversation (a sequence of role-tagged messages - user, assistant, and
  optionally tool calls/results), rendered through the base model's own
  chat template. Only the messages marked as trainable in a given example
  contribute to the loss, so a dataset can mix turns you want the model to
  learn from with context turns it shouldn't be scored against.

This target is also wrapped by `make train/qwen/lora DATASET=<dir>
ADAPTER=<owner/name[:tag]>`, which builds the release binary first.

## Adapter references

A fine-tuned adapter is addressed the same way as any other model: as a
named reference of the form

```
Qwen/Qwen3-0.6B:owner:name:tag
```

and stored on disk under `adapters/<owner>/<name>/<tag>/`. A base model and
any number of fine-tunes of it can sit side by side as distinct, separately
selectable models.

## Evaluating an adapter

```
brain qwen3 eval --weights BASE --jsonl held-out.jsonl [--adapter OWNER/NAME[:TAG]]
```

scores a checkpoint on held-out data - either the base model alone, or the
base model against an adapter side by side - reporting an explicit verdict
on whether the adapter beats the base.

## Serving cost

A served adapter is folded into its base weights at load time, so there is
no extra per-token cost to serving a fine-tuned model versus the base
model.

## Teaching a model a document, with a gate

```
brain document-study --arch <name> --weights BASE --dataset facts.json \
                     --adapter-dir DIR --report report.json
```

trains a LoRA adapter on a batch of frozen
`{fact, probe_question, expected_answer}` triples and decides, against a
pre-registered statistical bar, whether the result is good enough to
promote. It is one command, not a pipeline: the dataset goes in, a verdict
comes out.

- `--arch <name>` - which architecture the base checkpoint is
  (`qwen3`, `qwen35`, `qwen35moe`; defaults to `qwen3`).
- `--dataset FILE.json` -
  `{"cycles": [[{"fact": …, "probe_question": …, "expected_answer": …}, …]],
  "anchors": [ … ]}`. Each entry of `cycles` is one batch the study trains
  and gates as a unit; `anchors` is the behaviour suite every batch
  rehearses so a new document cannot quietly destroy what the model already
  did. Unknown or missing fields are rejected by name before any training
  starts.
- `--report FILE.json` - **always** written, whether the study promotes or
  rejects: per batch, the pass rate on its frozen probes before and after
  training, the gate's own p-value and effect size, the reason for a
  rejection, and the same row for a control arm whose gate is a coin flip -
  the comparison that says whether the real gate carried any information.
- `--adapter-dir DIR` - written to **only** on a promote, as
  `adapter-NNNNNN.safetensors`. That is exactly the name and ordering
  `brain serve --watch-adapters DIR` looks for, so a promoted adapter is
  picked up by a running server with no restart.

The probe questions are never trained on - only the facts are - so the
score is a held-out measurement rather than a memorisation check. A run
scoring fewer probes than the pre-registered floor is reported as
`"preregistered": false`: it exercised the machinery, but it is not a
result.
