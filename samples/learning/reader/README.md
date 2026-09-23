<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# reader - a model that keeps reading, and can say what it gained

Point a model at a directory and leave it. It decides, per episode, whether
what it just read is worth learning, whether it learned it, and whether that
cost anything it already knew. Everything it learns lives in one run
directory you can stop and reopen.

## What it demonstrates

The hard part of showing continual learning is not the learning, it is
having a claim someone else can check. "It got better at text" is not
checkable.

So this sample **invents a command-line tool that does not exist**, writes
that tool's manual pages as the corpus, and ships the tool's own grammar
parser as the judge. Before reading, the model writes invocations the tool
rejects. After reading, it writes invocations the tool accepts. The baseline
is provably zero because the tool did not exist until the seed was drawn,
and nothing scores the model except a parser.

```console
$ sample-learning-reader generate --corpus corpus/
invented vex3 and quil7, 9 subcommands between them

$ sample-learning-reader battery --model Qwen/Qwen3-0.6B --run-dir run/
capability 0/5

$ sample-learning-reader read --corpus corpus/ --run-dir run/ --model Qwen/Qwen3-0.6B
read 11 episodes, promoted 4
bank 4 earlier episodes, detection latency 1 episodes

$ sample-learning-reader battery --model Qwen/Qwen3-0.6B --run-dir run/
capability 4/5
```

The battery is the result. `read`'s promote rate is plumbing: a run whose
promote rate rises while the battery stays flat has failed.

**Two tools are generated, disjoint by construction, and read in sequence.**
That is what makes catastrophic forgetting legible: the question is whether
learning the second destroys the first, and a shared flag between them would
mean learning one reinforces the other.

## It also demonstrates refusing

A corpus of only learnable material would show that a reader can learn and
nothing about whether it can say no. `generate` writes nine lanes, each
labelled, and `selftest` checks every episode's verdict against its label:

| lane | what it is | expected |
|---|---|---|
| `learn/` | the real manuals for both tools | learned |
| `repeat/` | a manual already read | refused, for one forward pass |
| `noise/` | valid text with no structure in it | refused, before the model is touched |
| `contradict/` | a page claiming a flag takes a value when it does not | refused |
| `degenerate/` | one line five hundred times | refused |
| `shortcut/` | a page containing a battery answer verbatim | refused at ingest |
| `format/` | the shape of a manual describing nothing | either |
| `counterfact/` | near-identical flags with opposite meanings | either |
| `rare/` | one subcommand documented once and never again | learned, and still working at the end |

`selftest` exits non-zero if the reader learned something it should have
refused.

## Running it

```bash
make samples/learning/reader/build
./target/release/sample-learning-reader generate --corpus corpus/
```

`generate`, `report` and `selftest` need no model. `battery` and `read` take
`--model REF`, resolved by brain's own model handler, so a reference that is
not on disk is fetched.

| verb | |
|---|---|
| `generate` | invent the tools and write the corpus, lanes and labels |
| `battery` | score the held-out tasks against what is served now |
| `read` | read the corpus; resumes if the run directory has been read before |
| `report` | every episode recorded, and where each one stopped |
| `selftest` | check every verdict against its lane's label |

## What it does not claim

Nothing here says a particular model will learn this tool. The sample
provides the instrument; the number it produces is whatever the model and
the budget actually achieve, and a run that gains nothing reports that.
