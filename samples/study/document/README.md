# sample: study/document

Teach a model a batch of documents, and gate whether it learned them.

```bash
make samples/study/document/run ARGS="--weights Qwen/Qwen3-0.6B \
                                     --dataset facts.json \
                                     --adapter-dir adapters/ \
                                     --report report.json"
```

## What it demonstrates

* **A study is a library capability, not an engine verb.** Everything this
  program does beyond parsing a command line is one call to
  `brain::DocumentStudy`. It trains a LoRA adapter on frozen
  `{fact, probe_question, expected_answer}` triples, scores the result against
  a pre-registered bar, and publishes the adapter only if the gate promoted.
* **The control arm runs beside it, always.** A null-gate arm is trained on
  the same probes with the gate's decision rule deliberately broken, so the
  reported number says something about the *gate* rather than only about the
  run. Both arms appear in the report, per cycle, with their own p-values and
  effect sizes - not collapsed into one scalar.
* **A rejection is a first-class outcome.** If the gate does not promote,
  nothing is published and the report names which check failed. That is the
  normal path for a study that did not teach the model anything, and it is
  reported rather than smoothed over.
* **The dataset is validated before anything expensive happens.** `--dry-run`
  runs the exact same structural and semantic checks the real study applies -
  same code, so the two cannot come to disagree - with no weights resolution,
  no checkpoint load and no device anywhere in reach.

## Usage

```text
--weights BASE        checkpoint path, model directory, or vendor/repo ref
--dataset FILE.json   frozen {fact, probe_question, expected_answer} batches
--adapter-dir DIR     where a PROMOTED adapter is published
--report FILE.json    machine-readable verdict (optional)
--arch NAME           which architecture's Model impl to use (default qwen3)
--dry-run             validate the dataset and exit
```

Plus the study's own knobs: `--lora`/`--alpha`, `--steps`, `--seqs`/`--batch`/
`--lr`, `--eval-per-cycle`, `--work-dir`, `--models-dir`, `--seed`/
`--null-gate-seed`, `--quiet`.

Both seeds are drawn randomly when not given, and the drawn value is printed
**before** the run starts - so an interrupted study is still reproducible.

## The dataset

One JSON object: `cycles` (one batch of triples per study cycle) and
`anchors` (a suite mixed into every cycle's draw, so cycle 1's training
distribution is not one document alone).

```json
{
  "cycles": [[{"fact": "...", "probe_question": "...", "expected_answer": "..."}]],
  "anchors": [{"fact": "...", "probe_question": "...", "expected_answer": "..."}]
}
```

Parsed into typed structs with `deny_unknown_fields` and no `Option` members,
so a missing, mistyped or extra field is a loud failure naming the field
rather than a plausible-looking default that trains silently on the wrong
thing. A cycle below the pre-registered held-out floor is refused by name.

## What it needs

* A base checkpoint whose architecture is registered for a study
  (`brain::DocumentStudy::architectures()` lists them - the Qwen-family
  decoders today). The bound is the TOKENIZER, not the model: the curriculum
  is hard-wired to an HF `tokenizer.json` BPE.
* The base's directory must hold that `tokenizer.json` and a
  `tokenizer_config.json` carrying a chat template.
* A GPU for anything beyond a toy fixture - this trains.

## Why it is a sample rather than a `brain` subcommand

The study is architecture-agnostic machinery: `rl::continual::run_study` is
generic over `model::Model` and the curriculum names no model type at all. It
is also one workflow among many that could be built on that machinery, and the
thing that makes it useful to a caller is that it is *embeddable* - an agent
deciding to teach itself something reaches for a library, not a subprocess.

So the capability lives in the SDK (`brain::DocumentStudy`, the `study`
surface) and this sample is the worked example of driving it.
