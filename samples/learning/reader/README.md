<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# reader - teaching a model a tool that does not exist, and checking whether it worked

The hard part of demonstrating learning is not the learning. It is producing
a number someone else can check. "It got better at text" is not checkable,
and neither is a score a model gave itself.

So this sample **invents a command-line tool that does not exist**, writes
that tool's manual pages, and ships the tool as a real, executable, strictly
parsed program. Nothing scores the model except running what it wrote. The
baseline is genuinely zero because the tool did not exist until the seed was
drawn: a base model asked how to use it reaches for CSS margins and building
sealants.

```console
$ sample-learning-reader score --probes sft/ --model Qwen/Qwen3-0.6B
  NotThisTool  want haskctl prune --margin 7d
       got "It seems there might be a typo. The term \"7d\" doesn't make sense
            in the context of setting margins. Let me know if you're asking
            about setting margins in CSS!"
score 0/33 on held-out questions, every answer executed
```

## Running is the judge, not string similarity

A candidate answer is correct when RUNNING it produces the same observable
result as running the reference. That is a stronger claim than "it looks
similar" and a different one from "it parses":

| candidate | verdict |
|---|---|
| `haskctl seal --force-label main` | correct |
| `haskctl seal --force-label main --max-quota 64M` | **valid, and a different command** |
| `haskctl seal --force-label banana!!` | `BadValue { wants: "NAME" }` |
| `haskctl seal main` | `UnexpectedValue("main")` |
| `hask seal --force-label main` | `NotThisTool` |

The second line is why this executes rather than parses. It is a perfectly
valid invocation, it is not the command that was asked for, and only running
both says so. Every one of those wrong answers passes a fuzzy string check.

**Executing a model's guess is safe by construction.** The tool has no
filesystem, no network and no state that outlives the call. The worst a
wrong answer can do is produce a different transcript.

## The pipeline

Training a model on raw text teaches it to CONTINUE that text. Asking it a
question afterwards is a different task in a different format. So the corpus
is turned into question/answer pairs first, and the model is fine-tuned on
those - training and use are the same shape.

```console
$ sample-learning-reader generate --corpus corpus/ --seed 1
invented haskctl and zornsvc, 10 subcommands between them

$ sample-learning-reader phrase --corpus corpus/ --out big/ --model Qwen/Qwen3-0.6B --passes 6
answers     18  built by the tool and verified to run
questions  425  phrased for them
training   211  of 425 rows survived (50%)
  deduped  94  repeated question(s)
distinct   117  rows written

$ sample-learning-reader sft --in big/ --out R/ --held-out 2
train       56  rows, answer supervised and question masked
validate    28  phrasings spent on deciding when to stop
probe       33  held-out phrasings, never used for any decision
```

**The tool writes the answers; the model only phrases the questions.** Asked
to invent both halves, a model invents wrong answers in the right format: in
this corpus 26 of 28 generated commands were refused by the tool that would
have to run them, and the 2 that passed answered a different question than
the one they were paired with. A model asked only to phrase a question for
an answer that is already correct cannot produce a wrong answer at all.

The answers come from the tool's own grammar and are **executed before being
offered**, so a defect in the generator is caught before it becomes a
training row.

## The answer being right does not make the pairing right

A model asked for several questions about one command drifts onto the other
flags in front of it, and the answer is attached regardless. `Tool::mismatch`
refuses six kinds of that, mechanically, from the tool's own vocabulary:

- the question names a flag the answer does not set
- the answer sets an optional flag the question never asked for
- the question does not ask for a command at all ("what is the purpose of")
- the question quotes a placeholder (`TIME`) instead of a value
- the question names a flag this tool does not have
- the question quotes its own answer

Stated as what a question must contain rather than as a list of what it must
not: the ways of asking what something IS are endless, and a blacklist is
always one phrasing out of date.

## Three splits, and why not two

`validation` is spent on deciding when to stop. `probe` is never used for
any decision. Stopping early on the probe would make it a selection
criterion and the reported number a maximum over a search.

Splits are by PHRASING, not by answer: the probe asks for something the
training half teaches, in words it never used. Splitting by answer would ask
about a command the training half never mentioned, which measures the split
rather than the model.

`audit` re-checks the written files rather than trusting the stage that
wrote them - 17 checks, non-zero exit on any:

```console
$ sample-learning-reader audit --sft R/
  no question is both trained on and probed            PASS
  no question both decides the stop and reports the result PASS
  every probe answer is taught by the training half    PASS
  every probe answer executes                          PASS
  ...
train 56 / validation 28 / probe 33
every check passed: these splits are testing what they claim to
```

It earns its keep. Asked for eight phrasings the model repeats itself, and a
repeated question landed in two halves of one split - which turns a held-out
score into a recall score. The guard refused to write it.

## Stop before memorising

`--patience N` watches the held-out loss and keeps the checkpoint from
before it turned. On this data:

```text
step  20  train 1.3943  eval 1.0987
step  40  train 0.0384  eval 0.3818   <- best
step  60  train 0.0012  eval 0.4369   <- turned
step 100  train 0.0002  eval 0.4395   stopped, kept step 40
```

Training loss reaches 0.0002 while the held-out loss has been rising since
step 40. Without this the run trains to the end and saves the memorised
model.

## The control arm is the point

`sft --control` writes the same questions, the same answers, the same row
count - with every pairing deliberately wrong. Both arms then get the same
step budget and the same stopping rule, and are scored on a byte-identical
probe set.

They do not run for the same NUMBER of steps, and that is the rule working
rather than a confound: each arm stops where its own held-out loss turns. The
control stops earlier because it runs out of things to learn earlier, which is
itself part of the result.

A probe that scores as well from the control as from the real thing is not
measuring the mapping from a request to a command, and whatever it IS
measuring would have been reported as learning.

## What it does on this corpus

Qwen3-0.6B, LoRA rank 8, 56 training rows over 17 commands. Both arms get the
same step budget and stopping rule; all seven runs are scored on the same 33
held-out phrasings, and every answer is executed.

| arm | seed 1 | seed 2 | seed 3 | held-out loss |
|---|---|---|---|---|
| base, no adapter | 0/33 | - | - | - |
| **real** | **27/33** | **27/33** | **28/33** | 0.020 / 0.025 / 0.010 |
| control (pairings deranged) | 1/33 | 0/33 | 2/33 | 0.307 / 0.254 / 0.264 |

The arms separate on every seed, with the worst real run 25 answers clear of
the best control run. The control is not merely lower - it is at the floor,
and it got there having learned the tool's grammar perfectly:

```console
$ sample-learning-reader score --probes R/ --model Qwen/Qwen3-0.6B --adapter ctrl-1
  valid, different command     want haskctl budget-rebuild --skip-depth 16
                               got "haskctl seal --force-label main --region"
score 1/33 on held-out questions, every answer executed
```

Every one of those runs. None of them answers the question. That is what the
control is for: a probe those answers had scored well on would have been
measuring format, and format is exactly what both arms learn.

Where the real arm misses, it misses for a reason worth stating. Five of the
six seed-1 failures are on answers the training half taught with exactly ONE
phrasing - and there are exactly six such answers. Everything shown twice or
more was learned. The measured limit here is one-shot generalisation, and
`phrase --passes N` is the dial that moves it.

## One example per row, and why the number depends on it

Before this sample could report anything, `model::load_dataset` had to stop
packing. A chat dataset is one token stream with `<|endoftext|>` between
examples, and a training row used to be an arbitrary window of it:

```text
block_size                     1024 tokens
one question/answer example      42 tokens
-> ~24 examples per row, all attending to each other
48% of held-out examples had their own answer verbatim, earlier in the row
```

A model can drive that loss to zero by copying a neighbour. At serving time
there is one question and nothing to copy, so the training regime was one
that never occurs in use - and held-out loss scored the copying, reporting
0.0353 while the model answered 25% of questions. One example per row costs
nothing (the row is sized to the data: `--block 57` here, not 1024) and is
the difference between the table above and no measurable effect at all.

## How long it takes

One P40, end to end: training is 5-6 minutes for a real arm and 2-3 for a
control, on 330-480 steps before the held-out loss turns. Serving costs 5.5 s
to load the model and adapter, then **1.8 s per answer** (1.8 s median, 2.2 s
slowest) - 60 s for all 33 questions at 7.6 tokens/s.


## Running it

```bash
cargo build --release -p sample-learning-reader
```

`generate`, `report`, `selftest`, `sft` and `audit` need no model. `phrase`,
`score`, `battery` and `read` take `--model REF`, resolved by brain's own
model handler.

| verb | |
|---|---|
| `generate` | invent the tools and write the corpus, its lanes and their labels |
| `phrase` | the tool's answers, the model's questions, verified by execution |
| `sft` | split into train / validation / probe; `--control` writes the null arm |
| `audit` | prove the splits are what they claim, before anything trains |
| `score` | ask the held-out questions and RUN each answer |
| `read` | the continual reader: read a directory, deciding what is worth learning |
| `battery` | the reader's own held-out capability tasks |
| `report` | every episode the reader recorded, and where each one stopped |
| `selftest` | check every reader verdict against its lane's label |

## The continual reader

`read` is a separate path with its own design: it walks a directory, decides
per episode whether what it read is worth learning, and records why it
refused. `generate` writes nine labelled lanes for it, and `selftest` checks
every verdict against its lane's label:

| lane | what it is | expected |
|---|---|---|
| `learn/` | the real manuals for both tools | learned |
| `repeat/` | a manual already read | refused, for one forward pass |
| `noise/` | valid text with no structure in it | refused, before the model is touched |
| `contradict/` | a correction claiming a flag takes a value when it does not | refused |
| `degenerate/` | one line many times over | refused |
| `shortcut/` | a page containing a battery answer verbatim | refused at ingest |
| `format/` | the shape of a manual describing nothing | either |
| `counterfact/` | near-identical flags with opposite meanings | either |
| `rare/` | one subcommand documented once and never again | learned |

That path trains on raw next-token continuation and is measured by a
question-answering battery, which is the mismatch the pipeline above exists
to fix. Its measured result on this corpus is a battery that does not move.

## Limitations

The probe measures **unseen phrasing to correct command, verified by
execution**. Composing commands never trained on is out of its reach, and
necessarily so: this pipeline fine-tunes on question/answer pairs only, so the
model never sees the manual at training or inference time.

The sample is the instrument rather than a verdict on any particular model or
tool. A run that gains nothing reports that, and a run whose control is not
separated from it reports that too.
