# LoRA fine-tuning (Qwen)

brain can fine-tune a Qwen3 model on your own chat-style data using LoRA
(Low-Rank Adaptation), producing a small adapter rather than a full copy of
the model's weights.

`brain qwen3 finetune` trains LoRA adapters. Full-parameter retraining
(`brain::ChatFullFineTune`) and preference optimization with DPO
(`brain::PreferenceFineTune`) are available through the SDK; see
[`docs/reference/sdk.md`](../../reference/sdk.md). Qwen3.8-27B has its own verb,
described under [Qwen3.8-27B](#qwen38-27b-brain-qwen35-finetune).

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
- `--activations keep|recompute` - what the backward keeps of the forward. A
  Qwen3 decoder always recomputes each layer, so `recompute` is accepted and
  `keep` is refused by name before the first step. `brain qwen3 train` and
  `brain qwen3 finetune` (full) take it too, also as `activations=` in
  `--train-precision`; naming both with different values is refused.

This target is also wrapped by `make qwen/finetune DATASET=<dir>
ADAPTER=<owner/name[:tag]>`, which builds the release binary first.

## Qwen3.8-27B (`brain qwen35 finetune`)

```
brain qwen35 finetune DATA_DIR --base BASE --out OUT --mode lora [--base-format f32|bf16|nf4|fp8]
    [--lora-rank R] [--lora-alpha A] [--seq-len T] [--recompute on|off] [--steps N] [--lr X]
    [--gdn-segment N] [--bf16-residuals on|off] [--mlp-rows N] [--eval-interval N] ...
```

- `--base BASE` - a brain-native `.safetensors` checkpoint, or (with
  `--base-format bf16`, `nf4` or `fp8`) a HuggingFace checkpoint directory such as
  `Qwen/Qwen3.8-27B-FP8`, which is streamed from disk one tensor at a time
  (about 1 GiB of host memory) and never held whole on the host.
- `--base-format` - how the frozen base is held on the device (earlier
  spelling `--base-dtype`; one flag for every model, an unsupported format is
  refused naming the model and its supported set). `f32` (the default) keeps
  the whole model resident in fp32 and can train in `--mode full` or
  `--mode lora`, but the 27B does not fit; `bf16`, `nf4` (4-bit, QLoRA) and
  `fp8` (block-scaled E4M3) shrink the base and train LoRA adapters only, so
  combining them with `--mode full` is refused.
- `--lora-rank`, `--lora-alpha` - adapter shape (aliases `--rank`, `--alpha`;
  alpha defaults to twice the rank).
- `--lora-variant V`, `--lora-plus-ratio X` - model-agnostic adapter variants,
  the same for every LoRA-capable verb: `V` is a comma list of `rs` (rsLoRA:
  the update is scaled by `alpha / sqrt(rank)`) and `fa` (LoRA-FA: the `A`
  matrices stay at their initial values and only `B` trains); `X` makes the `B`
  matrices train at `X` times the learning rate (LoRA+). Continue an adapter
  with the same options. The SDK's `ChatFineTune` exposes `rank_stabilized`,
  `lora_plus` and `freeze_a`.
- `--seq-len` - training sequence length (alias `--block`).
- `--recompute` - recompute layer activations in the backward pass instead of
  keeping them; the default for `bf16` and `nf4`. The `f32` resident trainer
  keeps all activations, so `--recompute on` is refused there.
- `--activations keep|recompute` - the same choice as `--recompute off|on` for a
  `--mode lora` run (naming both with different values is refused), and the
  `activations=` key of `--train-precision` for a `--mode full` run.
- `--backend cuda|cpu|vulkan` is the global backend flag, as for every command.

- `--gdn-segment`, `--bf16-residuals`, `--mlp-rows` - the long-sequence
  settings of a `bf16`/`nf4` base: segmented Gated-DeltaNet layers (tokens per
  segment), a bf16 residual stream, the MLP run in token chunks. Rows of 2048
  tokens or more at batch 1 get 2048 / on / 2048 by themselves; naming a flag
  replaces that choice, and `0` turns `--gdn-segment` and `--mlp-rows` off.
  A segment needs a batch of one.
- `--eval-interval`, `--eval-batches` - evaluate on the held-out split every N
  steps (`0` never) on that many batches.
- Every optimizer step prints its loss, learning rate and wall time to stderr,
  and the base load prints its duration and the settings in force.

A batch wider than one with more than 8192 tokens per micro-step is refused
with the equivalent `--batch 1 --grad-accum G`.

### Using the adapter on the quantised base: `brain qwen35 sample-adapter`

```
brain qwen35 sample-adapter --base DIR --prompt TEXT [--prompt TEXT ...] [--adapter FILE]
    [--base-format nf4|bf16|fp8|f32] [--max-new N] [--chat] [--no-adapter | --compare]
```

Streams the base onto the device at `--base-format` with the adapter overlaid
(no fold into a host copy, which the 27B does not fit) and decodes greedily by
repeated full forwards at a fixed length, without a KV cache: slow, but exactly
the forward the adapter was trained through. `--no-adapter` decodes the base
alone and `--compare` prints the adapter's and the base's answer for each
prompt. The tokenizer is `tokenizer.json` in the base directory unless
`--tokenizer` names one.

The `f32` tier is the existing trainer for small and merged checkpoints.

## Images and video: `brain minicpmv finetune`

MiniCPM-V 4.6 fine-tunes on conversations with image and video parts, the
decoder (LoRA, full or frozen), the connector and the vision tower (LoRA,
full or frozen) each trained on its own, at its own rate:

```bash
brain minicpmv finetune --dataset train.jsonl --held-out val.jsonl --vision lora:4 --label acme:receipts:v1
brain minicpmv export-peft --tune acme:receipts:v1 --out receipts-peft
```

The tune is served as `brain/minicpmv:acme:receipts:v1`; the export is one
PEFT adapter over the Hugging Face checkpoint. Dataset format, options and
what is verified: [MiniCPM-V](../../reference/models/minicpmv.md#fine-tuning).

## Adapter references

A fine-tuned adapter is addressed the same way as any other model: as a
named reference of the form

```
Qwen/Qwen3-0.6B:owner:name:tag
```

and stored on disk under `adapters/<owner>/<name>/<tag>/`. For an adapter of
your own, `Qwen/Qwen3-0.6B+name` is the short way to say
`Qwen/Qwen3-0.6B:local:name:latest`, and `Qwen/Qwen3-0.6B+name:tag` names a tag;
the long form is what is printed. A base model and
any number of fine-tunes of it can sit side by side as distinct, separately
selectable models.

Training with an adapter name that already exists continues that adapter from
its own factors, at the rank and alpha it was trained at (a different rank is
refused); `--fresh` starts over from the base and overwrites it.

## Evaluating an adapter

```
brain qwen3 eval --weights BASE --jsonl held-out.jsonl [--adapter OWNER/NAME[:TAG]]
```

scores a checkpoint on held-out data - either the base model alone, or the
base model against an adapter side by side - reporting an explicit verdict
on whether the adapter beats the base.

## Serving cost

`brain serve` does not fold a served adapter into a second copy of the
weights. The base and every adapter run on one resident copy: a request for an
adapter adds a low-rank product per targeted projection per token, for its own
rows only, and a request for the base pays nothing. Memory is the adapter bank
(`--adapter-slots`, `--adapter-rank-cap`, `--adapter-targets`), a few percent
of the base, instead of a full weight copy per adapter. See
[`docs/guides/serving/serving.md`](../serving/serving.md#serving-lora-adapters).

The SDK pipelines (`ChatPipeline`, `TextGenerationPipeline`) fold an
adapter named at load into an fp32 base. On an int8 base, and
for an adapter switched in with `attach_adapter`, they apply it beside the
resident base instead, which keeps the correction exact and needs no reload.
The cost is an extra low-rank product per projection per token, which
`qwen_bench lora-decode` measures against a folded adapter and against no
adapter on the machine in hand.

## Teaching a model a document, with a gate

This is a **library capability**, reached through the SDK rather than a
`brain` subcommand, because the thing that makes it useful is that it is
embeddable - an agent deciding to teach itself something reaches for a
library, not a subprocess:

```rust
use brain::DocumentStudy;

let outcome = DocumentStudy::from_pretrained("Qwen/Qwen3-0.6B")?
    .dataset("facts.json")
    .adapter_dir("adapters/")
    .report("report.json")
    .run()?;
```

`samples/study/document/` is the worked example driving it from a command
line:

```bash
make samples/study/document/run ARGS="--weights BASE --dataset facts.json \
                                     --adapter-dir DIR --report report.json"
```

It trains a LoRA adapter on a batch of frozen
`{fact, probe_question, expected_answer}` triples and decides, against a
pre-registered statistical bar, whether the result is good enough to
promote. The dataset goes in, a verdict comes out.

- `--arch <name>` / `.arch(..)` - which architecture the base checkpoint is
  (`qwen3`, `qwen35`, `qwen35moe`; defaults to `qwen3`).
  `DocumentStudy::architectures()` lists them at run time.
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
  picked up by a running server with no restart and offered as a model of its
  own (its card id) beside the base.

The probe questions are never trained on - only the facts are - so the
score is a held-out measurement rather than a memorisation check. A run
scoring fewer probes than the pre-registered floor is reported as
`"preregistered": false`: it exercised the machinery, but it is not a
result.
