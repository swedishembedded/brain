<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# ouro

Scope: The Ouro looped language model (`crates/ouro`).

Brain's share of the looped-decoder work: Ouro reuses one 48-layer stack for
`total_ut_steps` passes per token, with per-pass attention history, an exit
gate and a pass-selected or exit-weighted head.

## Done

- Configuration and parameter layout pinned against the released
  `ByteDance/Ouro-2.6B-Thinking` header: 48 layers, hidden 2048, 16 query and
  16 KV heads of width 128, SwiGLU 5632, four passes, untied head, gated exit.
  The layout test checks 533 tensors and 2,667,974,657 parameters, so a shape
  drift fails before a 5 GiB import does.
- Hugging Face safetensors import (single file and sharded), whole-tensor
  rename with two-way coverage, no transpose.
- Forward graph: embedding applied once; each pass runs the shared stack,
  final RMSNorm and exit gate; the next pass consumes the previous pass's
  **normed** output (the reference reassigns `hidden_states = norm(...)`).
- Tiny reference parity (`tools/goldens/ouro_dump_reference.py`): every pass's
  hidden state, gate logit and head output match `modeling_ouro.py` at the
  same weights, CPU fp32, on the tiny shape.
- Exit rule: cumulative exit mass over the gate outputs, released threshold
  (1.0 selects the final pass).
- Released-checkpoint parity: the real 2.6B weights imported from the Hugging
  Face directory reproduce the reference's per-pass hidden state and logits at
  rel_l2 3-8e-6 on the first four passes (Tesla P40, wgpu).
- `brain ouro import|generate`, the arch row, the docs page and the model
  index row; a `catalog` resident that serves the same engine over HTTP,
  D-Bus and the SDK with the checkpoint's own BPE tokenizer and ChatML
  rendering.
- Whole-prompt prefill through the paged pools (one pass per recurrent step,
  `m = n`), gated against the full-sequence forward; per-pass index buffers
  keep every pass's attention on its own block table.
- Paged continuous-batching engine (`crate::serve::Engine`): `T` pools per
  layer, one block table per pass per sequence, admission charged
  `passes × blocks_for`; prefill and greedy/top-k decode go through the same
  paged step. Two interleaved sequences with different prompt lengths
  reproduce the model's own full-sequence greedy stream exactly.
- Incremental decode with one KV history per `(pass, layer)`: a step runs the
  new token through every pass, each appending its own K/V at the absolute
  position, and the gated result equals the full-sequence forward at every
  position and every pass on cpu, wgpu, native Vulkan and CUDA.
- Training objective and backward: every pass's CE weighted by that pass's exit
  probability, the gate's own gradient included (and zero on the final pass,
  whose probability is the remaining mass), accumulated into the one shared
  parameter set. The weighted loss matches the reference arithmetic recomputed
  from its own logits and gates; finite differences of the same forward pass
  on every parameter (the shared 2-layer stack visited three times) on cpu,
  wgpu, native Vulkan and CUDA.

## Left to do

- End-to-end serving on the real checkpoint: measure prefill/decode
  throughput, and run the brain-served model through an agent workload.
  Prefix reuse and host offload stay off for the looped path until they are
  pass-aware.
- Prefill memory: the whole-prompt path materializes `[heads, n, n]` scores,
  fine at agent prompt lengths but not at the checkpoint's full context; a
  chunked path (paged chunk attention) is the next scaling step.
- LoRA fine-tuning: adapter sites on the shared stack's linears, with the
  frozen-base path (full-weight training allocates gradients and AdamW moments
  for every parameter and does not fit a 2.6B model on one card).
- A runnable sample and its e2e registration.
- Threshold-selected decode across a batch whose rows exit on different
  passes.
