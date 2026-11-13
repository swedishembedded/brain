<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/qwen35moe

Qwen3.5-35B-A3B, the hybrid Gated-DeltaNet/GQA sparse-MoE decoder, as a
standalone program: `generate` with every parameter the served action has,
and `export`, the decoder as an ONNX graph for OpenVINO and the NPU.

`generate` is the model's served action, with a command line built from its
manifest, so this program, `brain qwen35moe generate` and `brain serve` take
the same parameters with the same defaults. `export` needs the NPU crate, so it
lives here rather than in the model.

## Build

```bash
make samples/models/qwen35moe/build
Q=target/release/sample-models-qwen35moe
```

## Weights

The checkpoint is a brain one in the model store, with the `tokenizer.json`
of the release beside it:

```bash
brain import-gguf Qwen3.5-35B-A3B-Q8_0.gguf --id brain/qwen35moe
```

`--weights` and `--tokenizer` name others outright; a `.gguf` given as
`--tokenizer` lends its embedded tokenizer, and without a tokenizer the prompt
and the result are token ids.

## Run

```bash
$Q generate --prompt "2+2=" --out text=answer.txt --seed $RANDOM
$Q generate --messages '[{"role":"user","content":"Say hello in French."}]' --chat true --out text=hello.txt
$Q generate --prompt "..." --precision int8 --out text=out.txt          # int8 weights, fp32 KV
$Q export --dest qwen35moe.onnx --seq 32                                 # fixed-length prefill graph
```

`$Q <action> --help` lists every parameter with its default and bounds.
