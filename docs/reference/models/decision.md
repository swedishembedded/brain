# Decision models

A decision model reads a situation, reads a list of options **the caller
supplies in the request**, and returns a calibrated probability for each. It
generates no text. One loaded model answers a three-option question and a
two-hundred-option one, because the output space lives in the request rather
than in a final layer.

Two architectures implement the same contract:

| | `decide` | Laya |
|---|---|---|
| backbone | MiniLM-family encoder, 22M parameters, plus a cross-attention head | ModernBERT-large, 395M parameters, plus Laya's own decision head |
| weights | trained by you: `DecisionPipeline::save_model` writes the directory | released as `convaiinnovations/laya`, ready to use |
| state | encoded once per request, shared by every question; long states are windowed | re-packed and re-encoded for every question; trained context is 512 tokens |
| page | [Option head](optionhead.md) | this page |

The English root Laya checkpoint is the one supported. Its multilingual and
typed-decisions variants are different backbones and are not served.

## Questions

Three types, all answered about one state:

| Type | Options | Answer |
|---|---|---|
| `choice` | 1 to 255 named options, each with an optional description | the selected option and a probability for every option |
| `score` | 2 to 10 ordered levels, worst first | the 0-based expectation over the levels, and a probability per level |
| `noul` | an implicit true/false, with optional readings of each | one probability on `[0, 1]` |

`confidence` is `1 - H/ln k` for a `choice` or `score` (1 for a certain answer,
0 for a uniform one, whatever `k` is). A `noul` has none: a two-outcome
probability already is its own confidence.

Questions are independent: no answer conditions another, and the order out is
the order in.

## Serving

Each decision checkpoint is its own model, addressed by its model-store id
(`convaiinnovations/laya`, `swedishembedded/triage`), so two checkpoints are two
models a caller selects by name and the name survives a restart. A checkpoint
outside the store is `local/<directory name>`.

```bash
brain pull convaiinnovations/laya
brain serve --openai 8788 --dbus
```

`brain serve` registers every decision model it finds in the model store: a Laya
checkpoint, and a `decide` directory written by `save_model` (a bare encoder has
no trained head and is not a decision model). `--decision-model REF|DIR` names
one explicitly, and may be repeated; a named model is registered even when its
weights are missing, so a client is told it is not installed instead of being
told nothing exists. `--decision-default ID` names the model `auto` resolves to.

Nothing is loaded at registration. The scheduler loads a model on the first
`decide` and evicts it when the card is needed. A model is placed on a GPU only.

`brain caps decide` lists the family. `brain decide decide --state "..."
--questions '[...]'` runs it in-process against the default model store (the
`--model` flag takes an id, a checkpoint directory, or `auto`, the default),
`brain decide models` lists what is installed without loading it and `brain
decide select` says which model `auto` would pick.

### The `decide` action

Reachable as `decide` on the model's own id: over D-Bus `Run`, over HTTP
`POST /v1/run`, and from the CLI.

| param | |
|---|---|
| `state` | the situation |
| `state_format` | `text` (default), taken verbatim; or `json`, a JSON document where a JSON string is text and anything else is structured state **kept in the caller's key order** |
| `questions` | JSON, in either spelling below |
| `deadline_ms` | cooperative deadline, 1 to 600000, default 60000 |

```json
{"model": "convaiinnovations/laya", "action": "decide", "params": {
  "state": "my payment failed again today",
  "questions": "[{\"id\": \"team\", \"type\": \"choice\", \"instruction\": \"who handles this\",
                  \"options\": [{\"id\": \"billing\", \"description\": \"payments\"}, \"tech\"]},
                 {\"id\": \"anger\", \"type\": \"score\", \"instruction\": \"how upset\",
                  \"options\": [\"calm\", \"annoyed\", \"furious\"]},
                 {\"id\": \"urgent\", \"type\": \"noul\", \"instruction\": \"is this urgent\"}]"}}
```

`questions` is an array of `{id, type, instruction, options | criteria}` as
above, or an object of name to `{type, instructions, criteria}` (the
`/v1/systemone` spelling: `criteria` is an ordered label-to-description map for a
`choice`, an ordered list for a `score`, a `{"true", "false"}` pair for a
`noul`). At most 64 questions and 1 MiB of state.

The response (`outputs`):

```json
{"model": "convaiinnovations/laya", "backend": "laya", "latency_ms": 41.2,
 "max_input_tokens": 512,
 "flags": {"state_truncated": false, "unsupported_question_types": []},
 "answers": [
  {"id": "team", "type": "choice", "selected": "billing", "selected_index": 0, "confidence": 0.91,
   "ranked": [{"id": "billing", "index": 0, "probability": 0.98}, {"id": "tech", "index": 1, "probability": 0.02}],
   "state": {"tokens": 31, "kept": 31, "truncated": false}},
  {"id": "anger", "type": "score", "score": 1.3, "confidence": 0.4,
   "legend": ["calm", "annoyed", "furious"],
   "ranked": [{"id": "1", "index": 1, "label": "annoyed", "probability": 0.7}, "..."],
   "state": {"tokens": 31, "kept": 31, "truncated": false}},
  {"id": "urgent", "type": "noul", "noul": 0.78,
   "ranked": [{"id": "true", "index": 1, "probability": 0.78}, {"id": "false", "index": 0, "probability": 0.22}],
   "state": {"tokens": 31, "kept": 31, "truncated": false}}]}
```

* `ranked` is the **full** distribution, best first (ties by position), each
  candidate with its id and its position in the list the caller sent.
* `backend` is `laya` or `decide`; `latency_ms` is the model's inference time for
  the call, not queueing or loading.
* `state.truncated` is true when the model answered from a **prefix** of the
  state. Laya's trained context is 512 tokens and it does not extrapolate
  (`max_input_tokens` reports the figure); `decide` windows a long state and
  never drops any of it, so it reports `windows` instead, and refuses a request
  that does not fit.
* A question whose type the model does not support is answered
  `{"id", "type", "unsupported": true, "reason"}` and named in
  `flags.unsupported_question_types`; a request in which no question is
  supported fails.

Within a request the `decide` architecture encodes the state once for all its
questions; Laya re-encodes per question. Across requests an instance answers
serially, because each request packs its own state into one bounded sequence and
there is no shared pass to exploit. A model has one forward in flight at a time;
the scheduler queues the rest under its admission deadline.

### Failures

Every failure carries a class a client can branch on without reading prose. Over
HTTP it is the error `code`, with the status below; over D-Bus it is the
`[class]` tag at the start of the error text.

| class | HTTP | meaning |
|---|---|---|
| `not-installed` | 404 | the model is named but its weights are not on this host; the message says what to `brain pull` |
| `failed-to-load` | 500 | the weights are present and cannot be loaded: corrupt or incompatible files, a hash that does not match the manifest, a declared context the checkpoint does not hold, an encoder with no trained head, a device that could not be built |
| `unsupported-capability` | 422 | the model cannot answer any question type asked |
| `timeout` | 504 | `deadline_ms` passed; no partial answers are returned |
| `invalid-request` | 422 | malformed, or outside the published limits (255 options, 2 to 10 levels, 64 questions, 1 MiB) |
| `backend-failure` | 500 | the backend faulted while running a valid request |

The deadline is cooperative: it is checked before the model runs and, for Laya,
between questions. A forward pass already on the device is not interrupted, so a
call can overrun by one pass. Time spent queueing or loading is not charged to
it.

Error messages name no filesystem path; the full error goes to the server log.

### Is it ready?

`POST /v1/run` on `brain/decision` with action `models` lists every decision
model the host knows with its state, and loads nothing:

```json
{"models": [{"id": "convaiinnovations/laya", "state": "ready", "backend": "laya",
             "configured": false, "validated": false, "manifest": {"...": "..."}},
            {"id": "acme/gone", "state": "not-installed", "missing": ["the model directory"],
             "hint": "run `brain pull acme/gone`", "configured": true, "validated": false}]}
```

`state` is `ready`, `not-installed` (files a load reads are absent) or
`incompatible` (the files are there and are not a servable decision model, or
the manifest does not parse). The query reads the filesystem, so a model pulled
after the server started is `ready` immediately; serving it takes a restart.

## The model manifest and validation

A decision checkpoint may carry `decision.manifest.json` beside its weights:

```json
{"schema": 1,
 "capability": "decide",
 "architecture": "laya",
 "weights_sha256": "sha256:...", "tokenizer_sha256": "sha256:...",
 "lineage": {"base": "convaiinnovations/laya", "trained_on": "...", "recipe": "..."},
 "max_input_tokens": 512,
 "question_types": ["choice", "score", "noul"],
 "validation": {"status": "validated", "profile": "ticket-routing", "evidence": "..."}}
```

Every field is optional. A checkpoint with no manifest is **unvalidated**, and
nothing makes a model validated except someone writing a `validated` status
with the `profile` it was measured on **and** the `weights_sha256` it applies
to: validation measures particular weights, so a claim not bound to them would
stay true after the weights changed. At load, a declared hash that does not
match its file, or a `max_input_tokens` the checkpoint does not hold, is a
`failed-to-load`. A manifest that is present and wrong makes the model
`incompatible` rather than quietly unvalidated. `architecture` must agree with
the directory's layout. `weights_sha256` is the digest of `model.safetensors`; for
a Laya head that references its backbone (below) it is the digest of the head,
the base it names and its calibration, so a validation stays bound to exactly
that model. Hashes are checked when a model loads, not when it is
listed, so a listing reports what the manifest records.

## Choosing a model: `auto`

`"model": "auto"` (the default of `/v1/systemone`) resolves, in order, to:

1. the model named by `--decision-default`, as named, validated or not;
2. the smallest-id **installed, validated Laya** model;
3. the smallest-id **installed, validated `decide`** model;
4. nothing, and the call fails with `not-installed` and the reason.

An unvalidated model is never picked automatically, and neither is one that is
not installed. A configured default that does not exist selects nothing rather
than something else. `POST /v1/run` on `brain/decision` with action `select`
returns `{"chosen", "rule", "reason"}` for `auto` or any id, again loading
nothing; `rule` is `configured`, `validated-laya`, `validated-decide` or
`none`.

## `POST /v1/systemone`

The shared wire shape, a thin layer over the same `decide` action:

```json
{"model": "auto",
 "state": {"ticket": "payments failing"},
 "questions": {
   "team":   {"type": "choice", "instructions": "who handles this",
              "criteria": {"billing": "payments, refunds", "tech": "outages"}},
   "anger":  {"type": "score", "instructions": "how upset", "criteria": ["calm", "annoyed", "furious"]},
   "urgent": {"type": "noul", "instructions": "is this urgent"}}}
```

```json
{"model": "convaiinnovations/laya",
 "answers": {
   "team":   {"type": "choice", "choice": "billing", "confidence": 0.91,
              "probabilities": {"billing": 0.98, "tech": 0.02}},
   "anger":  {"type": "score", "score": 1.3, "confidence": 0.4,
              "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
              "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}},
   "urgent": {"type": "noul", "noul": 0.78}},
 "usage": {"output_tokens": 0},
 "brain": {"backend": "laya", "latency_ms": 41.2, "max_input_tokens": 512,
           "flags": {"state_truncated": false, "unsupported_question_types": []}}}
```

Question names, option order and the state's key order are the caller's own and
survive to the model; `probabilities` are keyed in the caller's option order,
not best first. A request outside the limits (more than 255 options, a score
with fewer than 2 or more than 10 levels, an unknown type, a missing field) is a
422 naming the question; a body that is not JSON is a 400; an unknown model is a
404. `brain` is an additive block a client of the shared shape can ignore.

## Decision examples: the training data format

Training, evaluation and promotion read outcome-labelled choice decisions from
JSONL, one JSON object per line, written by an exporter (an agent runtime's
benchmark harness, for instance). `decisions::examples` is the only reader.
Unknown fields are ignored.

```json
{"decision_type": "tool_routing",
 "state_ref": {"run": "r1", "task": "t1", "suite": "tool_selection", "seed": 3, "question_id": "q"},
 "state": "find the function that parses headers",
 "question": {"id": "q", "type": "choice", "instruction": "which tool"},
 "options": [{"id": "grep", "description": "search text"}, {"id": "read"}],
 "label": "grep", "label_index": 0, "label_source": "oracle",
 "outcome": {"task_passed": true, "llm_tokens_saved": 120, "recovery_calls": 0},
 "split": "train", "family": "tool_selection", "seed": 3}
```

| field | rule |
|---|---|
| `decision_type`, `split`, `label_source` | required; `split` is `train`, `validation` or `test`; `label_source` is `oracle`, `counterfactual_replay` or `scorer` |
| `state` | text, or any JSON (its key order is kept); falls back to `question.state` |
| `question` | `{id, type: "choice", instruction}`; only `choice` is trained on |
| `options` | 1 to 255 entries, each an id or `{id, description?}`, ids unique; the text a model reads is the description, else the id |
| `label` | one of the option ids; a `label_index`, if present, must agree |
| `family`, `seed` | on the line, else `state_ref.suite` and `state_ref.seed`; a family is required |
| `outcome` | optional; `task_passed`, `llm_tokens_saved`, `recovery_calls`, each optional |

An example's identity is its `state_ref` (plus `question.id` when the
`state_ref` is a plain string). A bad line, an unknown label id or a repeated
identity is refused with an error that starts `line N:`.

**Leakage.** A held-out score means nothing if the held-out example or its task
was seen in training, so a file is refused, with a report naming the splits and
lines, when any of these groups spans more than one split: the task
(`state_ref.task`, or the whole `state_ref` when it names none), a family and
seed together, or an identical state, instruction and option set. A family may
appear in several splits through different seeds; a family that appears in one
split only is a held-out family.

## Training a decision model

`brain decide inspect --data FILE` validates a file of examples (schema,
duplicate ids, leakage) and prints its counts, label entropy and
majority-label baseline per split; `--json` prints the same machine-readably.

```bash
brain decide train --data decisions.jsonl --base sentence-transformers/all-MiniLM-L6-v2 \
    --out /models/triage-v1 --steps 600 --batch 16
```

| flag | |
|---|---|
| `--data FILE` | the JSONL above; only its `train` split is trained on |
| `--base REF` | a `decide` checkpoint or a stock Laya checkpoint: a model-store id or a directory. A trained Laya directory is not a base; name the stock checkpoint it was trained from |
| `--out DIR` | the checkpoint to create; it must not exist, and a failed run leaves nothing behind |
| `--freeze-encoder` / `--finetune-encoder` | head only (the default) or encoder and head. A Laya base is always head only and `--finetune-encoder` is refused for it |
| `--steps N`, `--batch N`, `--seed N` | optimizer updates (600), examples per update (16), seed (drawn at random and printed) |
| `--activations keep\|recompute` | what the backward keeps of the forward: every layer's activations, or only each layer's input, rebuilt in the backward for the same gradients at lower memory. Absent keeps them; a backend that cannot run the choice refuses it by name |

Each case brings its own question and options. A step draws cases at random
and shows each one's options in a fresh seeded order, so a position never
stands in for reading the option text. The same data, base, flags and seed give
the same weights.

The checkpoint's `decision.manifest.json` records its lineage and never claims
validation:

```json
{"schema": 1, "capability": "decide", "architecture": "decide",
 "weights_sha256": "sha256:...", "tokenizer_sha256": "sha256:...",
 "question_types": ["choice"],
 "lineage": {"base": "sentence-transformers/all-MiniLM-L6-v2", "base_sha256": "sha256:...",
             "data_sha256": "sha256:...", "split_counts": {"train": 800, "validation": 100, "test": 100},
             "hyperparameters": {"steps": 600, "batch": 16, "freeze_encoder": true, "final_loss": 0.41},
             "seed": 7, "brain_commit": "8b29b3e9e", "trained_on": "...", "recipe": "..."},
 "validation": {"status": "unvalidated"}}
```

`base_sha256` identifies the base checkpoint (`modelstore::fetch::base_digest`),
`data_sha256` the exact bytes of the JSONL, and `brain_commit` the code that
trained it, `-dirty` when its tree had uncommitted changes. An unvalidated
model is served only when named, never chosen by `auto`.

### Training a Laya base

Laya is the preferred stock decision model, and it trains the same way:

```bash
brain decide train --data decisions.jsonl --base convaiinnovations/laya \
    --out /models/acme/routing-v1 --steps 600 --batch 16
```

The run uses the Laya arm's own head objective (REINFORCE with a group
baseline over a strictly proper scoring reward, a cosine learning rate and
annealed exploration), with the same case draws, batch semantics (one optimizer
step per `--batch` cases, the summed gradient rescaled to a mean) and seed
determinism as a `decide` run. The 395M-parameter trunk is **frozen**: it is
held as weights only, with no gradient or optimizer moments, and its reverse
pass is not run. That is what makes training fit in memory at all, and a
frozen trunk keeps the lineage honest, since the base digest names exactly the
trunk in the result.

**`--finetune-encoder` is refused for a Laya base.** Training the trunk costs a
gradient and two optimizer moments for every trunk weight, and the result could
only be saved as a full copy of the weights; a head saved beside the published
trunk would not be the model that was measured. Train the head.

**The directory holds the head, not the backbone.** A stock Laya directory's
`model.safetensors` is the trunk and the head together; copying it for every
experiment would put the unchanged trunk on disk each time. A trained directory
is instead:

```text
routing-v1/
  decision.manifest.json    lineage, hashes, validation (unvalidated until promoted)
  laya.base.json            the backbone: {"schema": 1, "id": "convaiinnovations/laya",
                            "sha256": "sha256:...", "path": "<where it was>"}
  head.safetensors          the trained decision head
  rl_agent_config.json      } copied from the base: context lengths, and the
  encoder/config.json       } temperature tables that turn the head's logits
  tokenizer/tokenizer.json  } into probabilities; small
```

`brain-decisions` loads it exactly as it loads a stock checkpoint: it is a
model of its own, addressed by its store id, and `decide`, `eval`, `promote`,
`rollback` and `auto` need nothing Laya-specific.

**Base reference semantics.** `sha256` is `modelstore::fetch::base_digest` of
the stock directory, that is, the digest of its weight files, and it is the only
thing that identifies the base. At load the backbone is looked for, in order, in
the model store the trained directory sits in (`<root>/<vendor>/<repo>`), in the
default model store, and at the recorded `path`; the first directory that exists
**and** hashes to `sha256` is used. A base that is absent fails the load
(`failed-to-load`, naming the id and "run `brain pull <id>`"); one that is
present with other bytes is refused, since a head trained against one trunk
answers noise on another. The base must therefore stay installed to serve the
trained directory, and hashing its weight files is part of every load (it reads
them in full). A head file that does not cover the whole head, an edited head,
or an edited base reference fails the load the same way. A base named by directory path gets the id
`local/<directory name>`, which only the recorded `path` can locate; name it by
store id to make the trained directory relocatable.

The manifest records what a `decide` run's does (base id and digest, data
digest, split counts, steps, batch, seed, commit) with `"architecture": "laya"`
and `question_types: ["choice"]`, because choice is the one question type the
run trained; `score` and `noul` questions to it are flagged as unsupported
rather than answered from a head that never saw them. `weights_sha256` covers
the head file, the base digest and `rl_agent_config.json`. The base's
temperature tables are kept unchanged, so check the trained model's calibration
with `brain decide eval` (`ece`) before trusting its probabilities.

Memory and speed follow from the frozen trunk. Serving a trained directory
costs what serving the base costs (the head is a small addition), and the
resident reservation counts the base's weights. A training step encodes every
case through the whole trunk (the trunk is frozen but the head learns from its
output, so there is no sharing across a batch) and runs the head's backward
only; step time is therefore dominated by the trunk's forward and grows with
`--batch` and with the sequence length of the cases. Plan runs by measuring a
few steps on your own data and device.

## Evaluating a decision model

```bash
brain decide eval --model /models/triage-v1 --data decisions.jsonl --split test \
    --costs costs.json --out /tmp/triage-v1.eval.json
```

`--model` is a checkpoint directory or a store id (a stock model measures the
same way; a checkpoint whose manifest declares a hash its file does not match is
refused). `--split` is `validation` or `test`. The command prints a table, or
with `--json` the record; `--out` also writes the record to a file. A cost file
is `{"false_exclusion": 5.0, "wasted_tokens": 1.0}`.

| measure | meaning |
|---|---|
| `accuracy` | the share of examples whose most probable option is the labelled one |
| `baseline` | the larger of the *majority* baseline (always answer the `train` split's most common label id; the evaluated split's own majority when the file has no `train` examples, flagged `majority_from_train: false`) and the *chance* baseline (mean of `1/k`); `lift` is accuracy minus baseline |
| `nll` | mean `-ln p(label)` |
| `brier` | mean over examples of the sum over options of `(p - 1{label})^2` |
| `ece`, `ada_ece` | calibration of the top probability: examples binned by `p_max` into 15 equal-width (ECE) or equal-count (AdaECE) bins, the count-weighted mean gap between a bin's mean `p_max` and its accuracy; `reliability_bins` lists the bins |
| `coverage_accuracy` | examples ranked by the served `confidence` (`1 - H/ln k`); the accuracy of the most confident tenth, two tenths, ... of them |
| `order_instability` | every example asked again with its options in a seeded random order; the share whose winning option id changes (a model that reads the options scores 0) |
| `regret` | with a cost file: the mean realized cost per example of acting on the model's own confidence. The action is *trust* the top option (wrong costs `false_exclusion`, the right option was excluded) or *defer* and keep every option (a right top option costs `wasted_tokens`); the model trusts when its `p_max` makes trusting the cheaper expected action. `always_trust` and `always_defer` are reported beside it |
| `families` | accuracy, baseline and NLL within each task family |

## Promoting a checkpoint

A trained checkpoint is `unvalidated`, and `auto` never picks one. Promotion is
the only step that changes that, and it does not trust the checkpoint's own
numbers: it loads the weights, measures them on the **test** split of the data
file now, and compares the measurement with a policy.

```bash
brain decide promote --model /models/triage-v2 --data decisions.jsonl --policy policy.json
brain decide rollback --model /models/triage-v2
```

```json
{"schema": 1,
 "profile": "tool-routing",
 "min_test_examples": 100,
 "min_accuracy_lift": 0.10,
 "max_ece": 0.10,
 "max_order_instability": 0.05,
 "max_regret": 1.0,
 "costs": {"false_exclusion": 5.0, "wasted_tokens": 1.0},
 "required_families": ["tool_selection", "noisy_output"],
 "order_seed": 0}
```

| key | gate |
|---|---|
| `profile` (required) | the task profile written into the validation; letters, digits, `.`, `_`, `-` |
| `min_test_examples` (required) | the test split holds at least this many examples |
| `min_accuracy_lift` | accuracy minus the baseline accuracy is at least this |
| `max_ece` | expected calibration error is at most this |
| `max_order_instability` | at most this share of winners change when the options are reordered |
| `max_regret` | mean cost per example of the model's own trust-or-defer policy is at most this; needs `costs` |
| `required_families` | each family has test examples and, when `min_accuracy_lift` is set, meets it within the family |

At least one of the four quality gates is required, unknown keys are refused,
and `order_seed` seeds the reordering. The measures are the ones
[above](#evaluating-a-decision-model).

**A failed gate changes nothing.** Each failed gate is printed with its
observed value, its limit and the margin, the exit code is 1, and no file is
written. A checkpoint trained on one family and promoted under a policy that
requires another fails: the family has no test examples, or the model does not
beat that family's baseline.

**A passed promotion** writes, in this order: the evaluation record
(`decision.evaluation.json`: the report, the policy, the verdict, and the
digests of the weights and data) beside the manifest; the superseded checkpoint's manifest back to `unvalidated`; the
new manifest's `validation` (`validated`, the profile, the evidence file) and
`weights_sha256`, the digest of exactly the weights that were measured; and the
pointer `decision.promoted.<profile>.json` in the directory holding the
checkpoints, written atomically:

```json
{"schema": 1, "profile": "tool-routing",
 "current": {"dir": "triage-v2", "weights_sha256": "sha256:...", "evidence": "decision.evaluation.json"},
 "last_known_good": {"dir": "triage-v1", "weights_sha256": "sha256:...", "evidence": "decision.evaluation.json"}}
```

So at most one checkpoint per profile is validated, and the checkpoints of a
profile are siblings. A crash part-way leaves a profile with no validated
checkpoint (`auto` then chooses nothing), never an unmeasured one; running the
command again completes it. Two promotions of one profile cannot interleave: a
lock file named in the error is held for the duration.

**Rollback** makes the last-known-good checkpoint the validated one again and
un-validates the current one. The model named must be the profile's current
checkpoint, and the last-known-good must still hold the weights that were
validated, or nothing changes. After a rollback there is no last-known-good
until the next promotion.

## What the checkpoint knows

Laya is a System 1 model trained on agent and routing decisions over text. It
chooses among actions your system defines, triages, and judges whether to
escalate. It does not do arithmetic or compare numbers, and a threshold rule in
a criterion is text, not a test. Option text must describe, not label. When a
question is outside what it knows, `confidence` is usually the tell. See
`samples/decision/json` for measured runs.

## Not served

The act/escalate head of Laya is reachable through the SDK only
(`DecisionPipeline::route`). Batching across requests is the serial default,
for the reason given above. Training is `brain decide train`, the SDK and
`samples/decision`: the served surface is inference.

## Samples

| Sample | |
|---|---|
| `python/decision/served-decide` | the served model over plain HTTP, no SDK; exits 77 saying what to `brain pull` when there is nothing to ask |
| `decision/json` | the same wire shape on stdin/stdout, through the SDK |
| `decision/triage` | trains a `decide` head on BANKING77 |
