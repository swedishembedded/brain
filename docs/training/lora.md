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
