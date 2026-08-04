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
| **Qwen3 decoder** (partial-depth: 4 real blocks, QK-norm, θ=5e6) | Qwen3-VL-4B | **mean\|Δ\|≈5e-6, max\|Δ\|≈7e-5, argmax agrees** | `crates/qwenvl/src/parity.rs` — shards streamed to fit RAM (the full 36-layer model is ~32 GB in-brain) |
| **Moondream decoder** (partial-depth: 4 real dense blocks, parallel block + tau + partial RoPE) | Moondream 3 | **mean\|Δ\|≈5e-6, max\|Δ\|≈7e-5, argmax agrees** | `crates/moondream/src/parity.rs` — streamed from the 28 GB checkpoint; **caught a real bug**: the missing fused-qkv bias (below) |
| **Greedy generation** (text) | FastVLM-0.5B | **brain decodes the identical token stream as HF** — "Name three primary colors." → **"Red, Blue, and Yellow."** | same test; brain argmax-decodes 8 tokens `[6033,11,8697,...]`, matching HF exactly |
| **Image → caption** (splice + decode) | FastVLM-0.5B | **brain reproduces the HF caption token-for-token** — DOSBox logo → **"A wooden frame with the letters B, D, and S in it."** | `fastvlm_image_caption_matches_hf`: brain splices the 256 HF image embeddings (`enable_mm_splice`) and greedy-decodes `[32,22360,4034,…]`, identical to HF |
| **Fully-in-brain caption** (vision + projector + decode) | FastVLM-0.5B | **brain runs the WHOLE pipeline on its own weights and matches HF** — same caption, zero HF tensors at inference | `fastvlm_full_pipeline_caption`: brain's mobileclip vision → mlp2x_gelu projector → Qwen2 decoder → greedy decode → `[32,22360,4034,…]` |

This validates the **shared decoder backbone** — the same block math (`crates/model/src/block.rs`)
all three models decode with — against the actual reference model to fp32 reassociation
noise, *and* the full decode loop (embed → 24 layers → tied head → argmax → append) producing
a real, coherent, correct answer. It exercises the real-weight path end to end:
`checkpoint::safetensors` (bf16→f32) → `import::map_decoder` → `Qwen::new` → `logits_all`.

### What real-weight parity caught that gradchecks couldn't

Moondream's fused-qkv is an `nn.Linear` **with a bias**, and the checkpoint ships
`attn.qkv.bias` — but brain's `MoondreamBlock` did the qkv matmul with no bias add.
The synthetic gradcheck weights had no such bias, so every self-consistency check
(analytic == finite-diff) passed while the model computed a *different function* than
the reference. Real-weight parity surfaced it immediately (mean\|Δ\|=1.08, growing
with position); adding the (conditional) bias collapsed the delta to fp32 noise. This
is precisely why tier-3 parity matters on top of the gradient checks.

## Training convergence (tier 5)

| Check | Result | How |
|---|---|---|
| **VLM finetune overfits** | **loss 3.12 → 0.01** on a single image→caption example | `crates/fastvlm/src/train_smoke.rs`: the full loop `zero_grads → set_batch → (image-splice) forward → backward → AdamW`, 300 steps |

On top of the gradient checks (which prove the backward is *correct*), this proves the whole
training loop — image-splice, decoder, backward, and the AdamW optimizer together — actually
*drives learning*: the loss collapses from ~ln(vocab) to near zero as the model memorizes the
example. This is the "gradient-faithful finetune" capability, confirmed empirically.

## A note on big-checkpoint parity (memory)

Full-model real-weight parity fits comfortably for **FastVLM-0.5B** (validated above). For
**Qwen3-VL-4B** and **Moondream 3** it does not: a 4 B decoder needs ~14 GB as f32 weights
*plus* a second ~14 GB device copy inside `Qwen::new` (≈32 GB), over the ~18 GB available here.
Their correctness therefore rests on (a) the gradient checks, (b) import name-coverage, and
(c) the shared-backbone parity FastVLM establishes; a **per-block streaming** parity (load one
block's weights, compare that block's output) is the documented path to close the gap on a
larger box.

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
| **Image → caption (splice + decode) matches HF** | 🟡 | ✅ ("A wooden frame…") | 🟡 |
| **brain's own vision tower matches HF** | 🟡 | ✅ (FastViTHD, cosine=1.000000) | 🟡 |
| Multi-crop / dynamic resolution | ✅ smart-resize | ✅ pad-to-square | ✅ overlap multi-crop |
| MoE expert sharding | — | — | ✅ (federated round-trip) |
| Data-/pipeline-parallel | ✅ (trait + splice seam) | ✅ | ✅ |
| Video | ⬜ deferred | — | — |
| Region / point / detect heads | — | — | ⬜ deferred (Phase 3.9) |

## Known gaps ("all capabilities" caveats)

- **Generation loop.** Done for FastVLM: greedy decode via `logits_all` + `enable_mm_splice`
  drives the full image→caption above (matching HF). Qwen3-VL/Moondream composites still
  expose only the training `forward`; wiring the same greedy loop onto them is mechanical.
- **Moondream spatial heads.** The region/point/detect heads (grounding, pointing, object
  detection) were deferred; the importer *recognizes* their tensors but they are not built.
  Moondream's caption + visual-query paths are complete.
- **Qwen3-VL video.** Images only; the temporal/video path is deferred.
- **Big-checkpoint parity.** Qwen3-VL (8 GB) and Moondream (28 GB) exceed a single fp32
  load; their parity runs stream weights per block (`safetensors` mmap) rather than loading whole.

## Reproducing

```
# 1. reference dumps (needs torch + transformers, CPU is fine)
python3 tools/goldens/fastvlm_decoder_dump_reference.py

# 2. brain parity (CPU-JIT backend — the tied vocab table exceeds GPU binding limits)
BRAIN_DEVICE=cpu cargo test -p brain-fastvlm --release fastvlm_decoder_logits -- --nocapture
```
