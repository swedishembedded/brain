<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/gpt2

The dense GPT baseline: train it on a character dataset (alone or as one rank of a distributed job), generate text and evaluate a checkpoint, as a standalone program.

Every action is the model's served one, with a command line built from its
manifest, so everything `brain serve` exposes for the model is here with the
same parameters and defaults. The model has no released checkpoint: `train`
writes one, and the other actions name it with `--weights`.

## Build

```bash
make samples/models/gpt2/build
G=target/release/sample-models-gpt2
```

## Run

```bash
$G train --data data/calc --save calc.safetensors --steps 2000 --seed $RANDOM
$G generate --weights calc.safetensors --prompt "12+30=" --max-new 8 --seed $RANDOM
$G eval --weights calc.safetensors --data data/calc --seed $RANDOM

# one rank of a distributed job (every rank shares the plan and the secret)
$G train --data data/calc --save calc.safetensors --job-plan plan.json --job-rank 0 --job-secret job.secret
```

A dataset directory holds `train.bin`, `val.bin` and `meta.json` (the character vocabulary); `brain data` prepares one.

`$G <action> --help` lists every parameter with its default; `--device`
places the run.
