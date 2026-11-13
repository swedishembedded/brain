<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/wan

Wan2.1 text-to-video as a standalone program: a clip from a prompt, or a
LoRA trained on a folder of clips, with a report that shows what the adapter
changed.

The program is Wan's served actions with a command line built from their
manifest, so everything `brain serve` exposes for the model is here with the
same parameters and defaults.

## Build

```bash
make samples/models/wan/build
```

## Weights

```bash
brain pull Wan-AI/Wan2.1-T2V-1.3B
```

The DiT, VAE, umT5 text encoder and tokenizer resolve from the model store;
`--dit`, `--vae`, `--text-encoder` and `--tokenizer` name one outright.

## Run

```bash
W=target/release/sample-models-wan

# 81 frames at 832x480, written with the clip's frame rate
$W t2v --prompt "a red fox running through snow" --out video=fox.mp4

# a smaller, faster draft: int8 DiT, the text encoder on the second card
$W --device gpu0 t2v --prompt "..." --frames 33 --width 480 --height 288 --dit-dtype int8 --te-device gpu1 --out video=draft.mp4

# a video LoRA, then the base-vs-adapted report next to it
$W lora_train --data clips/ --save fox.brain --steps 200 --seed $RANDOM --report fox-report/
```

`sample-models-wan --help` lists the actions, `sample-models-wan <action>
--help` every parameter with its default.
