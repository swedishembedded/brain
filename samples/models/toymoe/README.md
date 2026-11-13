<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/toymoe

The toy sparse Mixture-of-Experts transformer: train it on its synthetic corpus, evaluate how well a checkpoint continues it, and sample from it, as a standalone program.

Every action is the model's served one, with a command line built from its
manifest, so everything `brain serve` exposes for the model is here with the
same parameters and defaults. The model has no released checkpoint: `train`
writes one, and the other actions name it with `--weights`.

## Build

```bash
make samples/models/toymoe/build
M=target/release/sample-models-toymoe
```

## Run

```bash
$M train --save moe.safetensors --steps 2000 --seed $RANDOM
$M eval --weights moe.safetensors
$M generate --weights moe.safetensors --prompt 1,2,3,4 --max-new 64 --seed $RANDOM
$M generate --weights moe.safetensors --prompt 1,2,3 --logits-only true
```

`eval` scores the training and validation orbits of the corpus and one the model never saw: whether it learned the substitution table rather than the sequences.

`$M <action> --help` lists every parameter with its default; `--device`
places the run.

## Federated experts

Each expert spans every layer, so one can be trained on its own against the
frozen shared backbone and assembled back later:

```bash
$M split --weights moe.safetensors --dest shards/
$M train_expert --weights moe.safetensors --expert 0 --dest exp0/ --steps 500 --seed $RANDOM
$M train_expert --weights moe.safetensors --expert 1 --dest exp1/ --steps 500 --seed $RANDOM
$M assemble --base shards/ --overlays exp0/,exp1/ --dest final.safetensors
$M verify --dir shards/
```

See [federated-experts](../../../docs/guides/training/federated-experts.md).
