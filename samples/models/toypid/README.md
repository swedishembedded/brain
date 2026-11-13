<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/toypid

The PID event/effect control transformer: train it on simulated plants, run a checkpoint in closed loop against each plant's oracle controller, and time one inference, as a standalone program.

Every action is the model's served one, with a command line built from its
manifest, so everything `brain serve` exposes for the model is here with the
same parameters and defaults. The model has no released checkpoint: `train`
writes one, and the other actions name it with `--weights`.

## Build

```bash
make samples/models/toypid/build
P=target/release/sample-models-toypid
```

## Run

```bash
$P train --save pid.safetensors --steps 2000 --seed $RANDOM
$P rollout --weights pid.safetensors
$P profile --weights pid.safetensors --cycles 100
```

`train` ends with the same closed-loop report `rollout` gives: each training plant and each interpolated plant it never saw, the model's tracking error against the plant's oracle.

`$P <action> --help` lists every parameter with its default; `--device`
places the run.
