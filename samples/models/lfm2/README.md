<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/lfm2

LFM2.5-Encoder as a standalone program: fill masked words, embed documents of
up to 8k tokens, and continue the encoder's masked-language training on your
own text, or export it and time it on the Intel NPU.

Every action is the model's served one, with a command line built from its
manifest, so everything `brain serve` exposes for the model is here with the
same parameters and defaults.

## Build

```bash
make samples/models/lfm2/build
L=target/release/sample-models-lfm2
```

## Weights

Import the reference release once into brain's checkpoint and keep it in the
model store, where every action finds it with its `tokenizer.json`:

```bash
brain pull LiquidAI/LFM2.5-350M
S=~/.local/share/brain/models
$L import --source "$S/LiquidAI/LFM2.5-350M" --dest "$S/brain/lfm2/lfm2-350m.safetensors"
cp "$S/LiquidAI/LFM2.5-350M/tokenizer.json" "$S/brain/lfm2/"
```

`--weights` and `--tokenizer` name either outright.

## Run

```bash
$L fill_mask --text "The capital of France is <|mask|>."
$L embed --text "a document to embed" --normalize true --out embeddings=doc.f32
$L tokenize --corpus notes.txt --dest data/notes
$L finetune --data data/notes --save notes.safetensors --steps 200 --seed $RANDOM
$L eval --data data/notes --weights notes.safetensors --seed $RANDOM
```

The encoder on the Intel NPU (OpenVINO and an NPU are needed to compile and
time it; the export alone runs anywhere):

```bash
$L npu-export --seq 1024 --quant int8 --out out/lfm.onnx
$L npu-bench --seq 8192 --quant f16 --iters 20 --compare true --ov-device npu
```

`$L <action> --help` lists every parameter with its default; `--device`
places the run. A long document is read from a file with `--in text=doc.txt`
instead of `--text`.
