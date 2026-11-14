<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/qwen3

The Qwen3 dense decoder (grouped-query attention, QK-norm, RoPE, SwiGLU) from the command line: generate text, train it from scratch, fine-tune every weight or a LoRA adapter, score it on held-out chat data, calibrate its INT8 KV cache, convert it to and from Hugging Face, GGUF and PEFT, and export or decode it on the Intel NPU, as a standalone program.

Every capability of the model is a served action (`generate`, `train`,
`finetune`, `lora_train`, `lora_gate`, `eval`, `calib`, `toolcall_gen`,
`toolcall_eval`, `import`, `export`, `export_peft`, `embed`), with a command
line built from its manifest, so everything `brain serve` exposes for the model
is here with the same parameters and defaults. `export-onnx`, `npu-generate`
and `npu-precompile` are this program's own: they write ONNX and run it
through OpenVINO, which needs the `npu` crate.

Concurrent, batched decoding is `brain serve --openai PORT` (paged KV cache,
continuous batching, int8 KV by default); this program runs one request at a
time.

The checkpoint is resolved from the model store, as `brain qwen3 <action>`
does: a brain checkpoint (`import`, `train` and `finetune` write one with a
`qwen` card, so the store finds it), a GGUF release, or a downloaded Hugging
Face directory. `--weights PATH` names one outright, and `--tokenizer PATH` its
tokenizer. `train`, `import`, `export_peft` and `toolcall_gen` need no
checkpoint.

## Build

```bash
make samples/models/qwen3/build
G=target/release/sample-models-qwen3
```

## Run

```bash
# generate from the stored checkpoint (a seed is drawn and echoed when omitted)
$G generate --prompt "Hello" --chat --max-new 64 --seed $RANDOM

# train a small decoder from scratch on a tokenized dataset (`brain data` prepares one)
$G train --data data/calc --save calc.safetensors --steps 2000 --seed $RANDOM
$G finetune --weights calc.safetensors --data data/calc --save calc-tuned.safetensors --seed $RANDOM

# a LoRA adapter on chat data, kept in the model store, then scored against the base
$G lora_train --rank 8 --adapter-name my-org/my-finetune --dataset train.jsonl --validation validation.jsonl --seed $RANDOM
$G eval --adapter-name my-org/my-finetune --samples validation.jsonl --out report=report.txt

# a Hugging Face release into a brain checkpoint, and back out
$G import --source <hf_dir> --dest qwen.safetensors
$G export --weights qwen.safetensors --format hf --out qwen-hf
$G export_peft --adapter adapter.safetensors --out peft-adapter

# the NPU route
$G export-onnx --weights qwen.safetensors --seq 64 --out qwen.onnx
$G npu-precompile --weights qwen.safetensors --seq 64 --npu-cache /tmp/npu-cache
$G npu-generate --weights qwen.safetensors --tokenizer tokenizer.json --prompt "Hello" --seq 64 --npu-cache /tmp/npu-cache
```

`$G <action> --help` lists every parameter with its default; `--device` places
the run. A chat dataset is JSONL, one `{"messages": [...]}` per line with each
message's `train` flag set. `npu-generate` is greedy and picks its OpenVINO
device with `--ov-device`; asked for the NPU it refuses a graph compiled for
another device unless `--allow-fallback` says that result is wanted.
