<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/glmdsa

The GLM-5.2 decoder (MLA + sigmoid noaux_tc MoE + DSA indexer) from the command line: train it from scratch or fine-tune a checkpoint on a character dataset, evaluate it, generate text, import the Hugging Face release, and export or decode it on the Intel NPU, as a standalone program.

`train`, `finetune`, `eval`, `generate` and `import` are the model's served
actions, with a command line built from its manifest, so everything
`brain serve` exposes for the model is here with the same parameters and
defaults. `export` and `npu-generate` are this program's own: they write ONNX
and run it through OpenVINO, which needs the `npu` crate.

The checkpoint is resolved from the model store, as `brain glmdsa <action>`
does: a brain checkpoint (`import`, `train` and `finetune` write one with a
`glm` card, so the store finds it) or a downloaded Hugging Face directory,
which `generate` serves as released with its own tokenizer. `--weights PATH`
names one outright. `train` and `import` need none.

## Build

```bash
make samples/models/glmdsa/build
G=target/release/sample-models-glmdsa
```

## Run

```bash
$G train --data data/calc --save calc.safetensors --steps 2000 --seed $RANDOM
$G eval --weights calc.safetensors --data data/calc --seed $RANDOM
$G generate --weights calc.safetensors --prompt "12+30=" --max-new 8 --seed $RANDOM --out text=out.txt
$G finetune --weights calc.safetensors --data data/calc --save calc-tuned.safetensors --seed $RANDOM

# a Hugging Face release into a brain checkpoint
$G import --source <hf_dir> --dest glm.safetensors

# the NPU route
$G export --weights calc.safetensors --seq 64 --out onnx=calc.onnx
$G npu-generate --weights calc.safetensors --prompt "12+30=" --max-new 8 --out text=out.txt
```

A dataset directory holds `train.bin`, `val.bin` and `meta.json` (the
character vocabulary); `brain data` prepares one. `train` takes a shape preset
(`--size tiny|small|base`) that `--layers`, `--d-model`, `--heads` and
`--experts` override.

`$G <action> --help` lists every parameter with its default; `--device`
places the run. `npu-generate` is greedy and picks its OpenVINO device with
`--ov-device`.
