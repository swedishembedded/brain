<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/diamond

DIAMOND, the Atari-100k diffusion world model, as a standalone program: import
the reference checkpoint, export its UNet for the Intel NPU, roll the model out
headless over an action script, record and replay episodes, benchmark it and
fine-tune it on what you recorded - and `play`, an interactive window where
the keyboard is the game controller and the model dreams the next frame.

Every action is DIAMOND's served one, with a command line built from its
manifest, so everything `brain serve` exposes for the model is here with the
same parameters and defaults. The window is written in this sample.

## Build

```bash
make samples/models/diamond/build
D=target/release/sample-models-diamond
```

## Weights

Convert an official Atari-100k `agent.pt` (for example `Breakout.pt` from the
`eloialonso/diamond` Hugging Face repository) once; the result is found in the
model store, or named with `--weights`:

```bash
$D import --source Breakout.pt --dest ~/.local/share/brain/models/brain/diamond/breakout.safetensors
```

## Run

```bash
SEED=$RANDOM                                                          # printed back by every run
$D rollout --action_seq 1,2,0,3 --seed $SEED --out clip=rollout.mp4      # headless frames, a rollout hash
$D rollout --model fake --action_seq 1,2,0 --hashes                  # the GPU-free test model
$D rollout --frames 300 --record episode/ --seed $SEED                 # record a session as an episode dataset
$D replay --episode episode/ --verify --seed $SEED                     # regenerate it and compare frame by frame
$D bench --frames 50                                                 # ms per frame
$D bench --profile                                                   # time per kernel of one UNet forward
$D finetune --data episode/ --save tuned.safetensors --seed $SEED      # fine-tune on the recording
$D export --dest diamond.onnx                                        # the UNet as ONNX
$D rollout --onnx diamond.onnx --action_seq 1,1,1                    # ... run on the Intel NPU
$D play --scale 10                                                   # the window: Space fire, A/D move, Esc quit
```

`$D <action> --help` and `$D play --help` list every flag with its default;
`--device cpu|gpu` and `--backend` place the run.

## What it shows

- A per-model program is the model's served actions plus what is an
  application: `invoke::main_with` runs the actions and hands the `play`
  workflow (`invoke::Workflow`) the resolved checkpoint.
- The window builds its model with `diamond::worldrun::build` from the same
  params the served `rollout` takes, so a window session and a headless
  rollout of one seed start from the same world, and `--record` writes the
  dataset `replay` and `finetune` read.
- The SDL renderer is a window parameter (`--renderer`), not process state.
