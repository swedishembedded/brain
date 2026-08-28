# Qwen3.5-35B-A3B (hybrid GDN/GQA sparse-MoE decoder)

A 40-layer hybrid decoder: 3:1 Gated DeltaNet (chunked linear-attention) to
GQA layers, a 256-expert top-8 sigmoid-gated-shared-expert sparse MoE on
every layer, a sigmoid attention-output gate, partial RoPE + M-RoPE on the
GQA layers, and a spliced vision front end reusing Qwen3-VL's ViT +
PatchMerger as-is.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| LoRA fine-tune         | [x] (rank-8 on the 9 targetable GDN/GQA projections; never the MoE experts) |
| CLI                    | [x] |
| HTTP API               | [x] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

Model id: `brain/qwen35moe`. GGUF streaming import (`brain import`, dispatched
by the file's own `general.architecture = qwen35moe`) - never a whole-model
fp32 disk intermediate, since the real checkpoint is roughly 140 GB at fp32.
INT8 quantization and cross-GPU pipeline sharding are available for
checkpoints that exceed one card.

The importer un-transforms `ssm_a` on the way in (`A_log = ln(-ssm_a)`):
llama.cpp's converter stores `-exp(A_log)` so `ggml_ssm_scan` can use it
directly, while `model::gdn`'s decay gate implements the reference formula and
wants the original `A_log`. That fix landed with the sibling `qwen35` GGUF
resident, which found it on real weights - see `.agents/rules/lessons.md`
#70. **It has NOT been re-validated end to end on a real `qwen35moe`
checkpoint** (none is available on the development box); the synthetic-fixture
import test covers the transform, real generation does not.

## Running it

```bash
brain qwen35moe import --gguf model-Q4_K_M.gguf --out qwen35moe.safetensors
brain qwen35moe infer --weights qwen35moe.safetensors --prompt "..."
brain qwen35moe export --weights qwen35moe.safetensors --out model.onnx
```

## Hardware and limits

No D-Bus/HTTP serving adapter yet - CLI only.
