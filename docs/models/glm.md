# GLM

A from-scratch GLM-5.2 decoder: a compressed, low-rank attention design that
keeps per-token attention state small, a mixture-of-experts routing layer
(a sparse set of expert MLPs selected per token, plus one always-on shared
expert), an optional attention indexer that narrows which past tokens each
query attends to for speed at longer context, and an optional head that
predicts several tokens ahead for faster generation. Reach for it when you
want to train, finetune, evaluate, or run inference against a from-scratch
GLM-family MoE decoder, or import official GLM-5.2 HuggingFace weights.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| INT8                   | [ ] |
| CLI (`brain do`)       | [ ] |
| HTTP API               | [ ] |
| D-Bus                  | [ ] |
| Batched serving        | [ ] |

## Getting the weights

Import official GLM-5.2 HuggingFace weights (single or sharded safetensors):

```bash
brain glm import --hf <hf_dir> --out glm.safetensors
```

There's no auto-fetch by model id yet — either import a checkpoint yourself
or train/finetune your own from scratch. For CLI inference against an
existing checkpoint, point `BRAIN_GLM_WEIGHTS` at it.

## Running it

```bash
brain glm train    <data_dir> --size tiny|small|base [...]
brain glm finetune  --weights F [...]
brain glm infer     --weights F --prompt "..."
brain glm eval      --weights F --data <data_dir>
brain glm import    --hf <hf_dir> --out glm.safetensors
brain glm export    --weights F --out model.onnx --seq T
```

`--size tiny|small|base` selects a runnable preset shape; the CLI grammar is
shared with `gpt`/`qwen`. `export` produces an ONNX graph for the Intel NPU
path (below) rather than a servable checkpoint.

## Options

- `--size tiny|small|base` — model size preset.
- `--device cpu|gpu` — backend selection.
- `--seq T` (export) — sequence length baked into the exported ONNX graph.

## Hardware and limits

Forward and backward passes are backprop-verified — the attention/MoE core,
the optional attention indexer, and the optional multi-token-prediction head
each have their gradients checked against finite differences, and the model
demonstrably learns on held-out data. The indexer is trained separately from
the language-model loss, by a distillation step that teaches it to match the
dense attention pattern; it needs a trained backbone first, since at random
initialization there's nothing meaningful to distill yet.

The full published GLM-5.2 shape (78 layers, 256 experts, roughly 155k
vocabulary) is far too large to run locally today and is used only to
validate that HuggingFace imports resolve to the right shapes — train,
finetune, and evaluate against the `tiny`/`small`/`base` presets instead.

GLM also has a validated **Intel NPU export path**: `brain glm export`
produces a dense-expert (every expert computed, non-selected ones masked to
zero) fp32 ONNX graph that has been confirmed to run correctly on real NPU
hardware. NPU INT8 weight-only quantization for GLM isn't implemented yet,
and the attention indexer and multi-token-prediction head aren't part of the
exported graph — the NPU path runs dense attention, and multi-token
prediction there would need a separate host-side draft loop. See
[glm/npu.md](glm/npu.md) for the export design.

Serving GLM over HTTP or D-Bus alongside brain's other decoders isn't wired
up yet — use the CLI directly for train/finetune/infer/eval.
