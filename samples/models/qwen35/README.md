<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/qwen35

The Qwen3.8-27B dense hybrid decoder (Gated-DeltaNet and grouped-query attention layers) from the command line: generate text, fine-tune every weight or a LoRA adapter on a base held at fp32, bf16, nf4 or fp8, read what an adapter changes against the base it was trained on, and import a GGUF or Hugging Face release, as a standalone program.

Every capability of the model is a served action (`generate`, `finetune`,
`sample_adapter`, `import`), with a command line built from its manifest, so
everything `brain serve` exposes for the model is here with the same parameters
and defaults. There is no action of this program's own: the model has no
ONNX/NPU route. Concurrent, batched decoding is `brain serve --openai PORT`
(paged KV cache, continuous batching); this program runs one request at a time.

The checkpoint is resolved from the model store, as `brain qwen35 <action>`
does: a brain checkpoint (`import` writes one with a `qwen35` card, so the
store finds it) or a downloaded Hugging Face directory. `--weights PATH` names
one outright, and `--tokenizer PATH` its tokenizer (or a GGUF release whose
embedded tokenizer is used). `import` needs no checkpoint.

## Build

```bash
make samples/models/qwen35/build
G=target/release/sample-models-qwen35
```

## Run

```bash
# generate from the stored checkpoint (a seed is drawn and echoed when omitted)
$G generate --prompt "Hello" --chat --max-new 64 --seed $RANDOM

# the same through a trained LoRA adapter, folded into the weights once
$G generate --prompt "Hello" --chat --adapter adapter.safetensors --seed $RANDOM

# a GGUF release into a brain checkpoint (MTP head included)
$G import --source Qwen3.8-27B-Q8_0.gguf --dest qwen35.safetensors

# a Hugging Face release re-encoded once at nf4, so a LoRA run copies codes instead of requantizing FP8
$G import --source Qwen3.8-27B-FP8 --dest Qwen3.8-27B-nf4 --weight-format nf4

# fine-tune every weight of a brain checkpoint on a tokenized dataset (`brain data` prepares one)
$G finetune --weights qwen35.safetensors --data data/facts --save tuned.safetensors --steps 200 --seed $RANDOM

# a LoRA adapter on a base streamed at nf4, kept in the model store under a name
$G finetune --weights Qwen3.8-27B-FP8 --data data/facts --mode lora --base-format nf4 --rank 16 --adapter-name facts --seed $RANDOM

# the same from chat JSONL, the adapter written to a path
$G finetune --weights Qwen3.8-27B-FP8 --dataset train.jsonl --mode lora --base-format nf4 --out adapter=adapter.safetensors --seed $RANDOM

# the adapter's answers beside the base's, on the same streamed base
$G sample_adapter --weights Qwen3.8-27B-FP8 --adapter adapter.safetensors --prompt "Who wrote the facts?" --mode compare --max-new 16
```

`$G <action> --help` lists every parameter with its default; `--device` places
the run. A full run needs the base at fp32 (`--base-format f32`, the default)
and writes a checkpoint; a LoRA run on a Hugging Face directory, or on a base
held narrow, streams the base tensor by tensor and writes only the adapter, as
the `adapter` output, to `--save`, or under `--adapter-id`/`--adapter-name` in
the model store, where a later run of the same name continues it unless
`--fresh` is given. `--activations recompute` trades time for memory on any run
(`--recompute` is the same choice for a streamed base), and the long-sequence
settings (`--gdn-segment`, `--bf16-residuals`, `--mlp-rows`) turn on by
themselves for rows of 2048 tokens or more at batch 1. `sample_adapter` is
greedy and decodes by full forwards with no KV cache: slow, but exactly the
forward the adapter was trained through.
