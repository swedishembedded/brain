# Vision-language models — capability & validation matrix

Three VLMs run in brain as first-class architectures: **Qwen3-VL-4B**, **FastVLM-0.5B**,
and **Moondream 3**. All share one shape — vision encoder → connector/projector →
autoregressive text decoder, with image embeddings spliced into the decoder's stream.

This document tracks, per capability, *what is implemented and how it is validated*.
The validation ladder, weakest → strongest evidence:

1. **Shape / finiteness** — a forward runs and produces finite, correctly-shaped output.
2. **Gradient check** — analytic backward == central finite-difference (self-consistent math).
3. **Reference parity** — brain's output matches the HuggingFace model on the *real* weights.
4. **End-to-end generation** — greedy-decodes a real caption / answer.
5. **Training convergence** — finetune actually drives the loss down on real data.

## Reference parity (tier 3) + generation (tier 4) — the strongest signals we have

| Check | Model | Result | How |
|---|---|---|---|
| **Decoder logits** (24 layers, GQA, SwiGLU, tied head) | FastVLM-0.5B | **mean\|Δ\|≈3e-6, max\|Δ\|≈6e-5, argmax agrees everywhere** | `crates/fastvlm/src/parity.rs` vs `transformers` on the real bf16 checkpoint |
| **Greedy generation** | FastVLM-0.5B | **brain decodes the identical token stream as HF** — "Name three primary colors." → **"Red, Blue, and Yellow."** | same test; brain argmax-decodes 8 tokens `[6033,11,8697,11,323,25462,13,151645]`, matching HF exactly |

This validates the **shared decoder backbone** — the same block math (`crates/model/src/block.rs`)
all three models decode with — against the actual reference model to fp32 reassociation
noise, *and* the full decode loop (embed → 24 layers → tied head → argmax → append) producing
a real, coherent, correct answer. It exercises the real-weight path end to end:
`checkpoint::safetensors` (bf16→f32) → `import::map_decoder` → `Qwen::new` → `logits_all`.

## Capability matrix

Legend: ✅ implemented + validated · 🟡 implemented, validation pending · ⬜ not yet built · — n/a

| Capability | Qwen3-VL-4B | FastVLM-0.5B | Moondream 3 |
|---|---|---|---|
| Image → training loss (fwd) | ✅ | ✅ | ✅ |
| Gradient-faithful backward | ✅ decoder + ViT | ✅ decoder + FastViTHD | ✅ decoder + SigLIP ViT |
| Checkpoint import (name-coverage) | ✅ | ✅ | ✅ (662 tensors) |
| **Decoder reference parity** | 🟡 (same harness, 8 GB ckpt) | ✅ (mean\|Δ\|≈3e-6) | 🟡 (MoE, 28 GB ckpt — per-block) |
| Vision-encoder reference parity | 🟡 | 🟡 | 🟡 |
| **Greedy generation (text) matches HF** | 🟡 | ✅ ("Red, Blue, and Yellow.") | 🟡 |
| Full image → caption (vision + generate) | ⬜ | 🟡 (needs vision parity) | ⬜ |
| Multi-crop / dynamic resolution | ✅ smart-resize | ✅ pad-to-square | ✅ overlap multi-crop |
| MoE expert sharding | — | — | ✅ (federated round-trip) |
| Data-/pipeline-parallel | ✅ (trait + splice seam) | ✅ | ✅ |
| Video | ⬜ deferred | — | — |
| Region / point / detect heads | — | — | ⬜ deferred (Phase 3.9) |

## Known gaps ("all capabilities" caveats)

- **Generation loop.** The composites expose a training `forward` (image → CE loss) but
  no greedy-decode `generate` yet — so no model produces a caption/answer end-to-end today.
  FastVLM-0.5B is the first target (it is the only checkpoint that loads whole in RAM).
- **Moondream spatial heads.** The region/point/detect heads (grounding, pointing, object
  detection) were deferred; the importer *recognizes* their tensors but they are not built.
  Moondream's caption + visual-query paths are complete.
- **Qwen3-VL video.** Images only; the temporal/video path is deferred.
- **Big-checkpoint parity.** Qwen3-VL (8 GB) and Moondream (28 GB) exceed a single fp32
  load; their parity runs stream weights per block (`safetensors` mmap) rather than loading whole.

## Reproducing

```
# 1. reference dumps (needs torch + transformers, CPU is fine)
python3 tools/fastvlm_decoder_dump_reference.py

# 2. brain parity (CPU-JIT backend — the tied vocab table exceeds GPU binding limits)
BRAIN_DEVICE=cpu cargo test -p brain-fastvlm --release fastvlm_decoder_logits -- --nocapture
```
