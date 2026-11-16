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
  layer, one allocator and block table per pass per sequence (a shared
  allocator charged a sequence `T x` its context against one plane's blocks,
  which exhausted the pool); prefill and greedy/top-k decode go through the
  same paged step. Two interleaved sequences with different prompt lengths
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

## Training

- The objective is the reference's: every pass's cross-entropy weighted by
  that pass's exit probability, summed over passes, averaged over supervised
  rows; the gate's gradient is derived analytically (product-exclusion form,
  valid when a gate saturates).
- Gated by finite differences on every parameter on cpu, wgpu and cuda, and by
  the host-recomputed reference arithmetic for the weighted loss.
- `brain ouro train` runs it end to end: the tiny fixture checkpoint falls
  3.37 -> 1.99 nats in 40 AdamW steps; the released 2.6B checkpoint falls
  1.76 -> 0.20 in 6 steps on one 64-token batch and the saved checkpoint is
  served. Full fine-tuning of the released checkpoint does not fit one 24 GB
  card (weights, gradients and AdamW moments are four times its fp32
  weights), so that run updates on the host tier.
- Two bugs the first real run exposed were in the new verb, not the model:
  the batch was never set, and AdamW's one-based bias correction was given a
  zero-based step.

## Training from scratch (2026-11-16)

- `ouro::finetune::OuroTrainer` is a `model::finetune::ChatTrainer`: the generic full-parameter chat fine-tune
  (role-masked from the template's own rendering), raw-text pretraining and exact resume run on the shared fit
  loop; `ouro::train::train` (the served `brain/ouro train`) runs on the same loop. It declares `FROM_SCRATCH`
  (a seeded `init::init_weights` checkpoint, `model.brain.safetensors` with a card of family `ouro`).
- The from-scratch sequence (init, pretrain with resume, chat fine-tune) is the conformance test
  `crates/sdk/tests/it/from_scratch.rs`, run over every declaring architecture; over the API in
  `crates/trainjob/tests/it/{create,pretrain,resume}.rs`.
- Served with the checkpoint's own chat template and tool-call form (`catalog::resident_ouro::chat_format`), a
  served prompt equal token for token to the training rendering (`crates/catalog/tests/it/ouro_served.rs`), context
  capped at the model's positions, installed models served under `local/<name>`. `samples/models/ouro` runs it.
- Measured on the CPU backend (Cranelift WGSL JIT, 72 threads, this node, 2026-11-16, `samples/models/ouro`):
  the tiny size (150k parameters, rows of 64, batch 8) trains about 1.4k tokens/s over a whole run and its loss
  falls 5.96 -> 1.21 in 200 steps; the default size (hidden 256, 4 layers, 4 passes, 4 heads, vocabulary 4096,
  5.3M parameters, rows of 1024, batch 4) about 61 tokens/s over a whole run. See below for what the step spends.

## Left to do

- CPU training rate of the from-scratch sizes (numbers above). Profiled (2026-11-16, `brain --device cpu
  --kernel-profile ouro train` on the default size, rows of 1024, batch 4, this node's CPU backend, 288 threads): a
  step is 55 s, 74 tokens/s; `gqa_bwd_dscores` (the materialized attention's score gradient, 32 calls of 1.6 s)
  takes 40 % of the kernel time, `matmul_dw` 22 %, `ce_grad` 9 % (every pass's `[rows x vocab]` logits),
  `attn_softmax` 8 %, the forward `matmul` 6 %. The training graph materializes `b x h x t x t` scores per layer and
  pass; a flash-style training attention (the GQA trainers' `flash_gqa_train` path) and a CPU GEMM candidate for
  `matmul_dw` are what make a 5M-parameter model train at host speed. The run's initial loss probe (five forwards,
  16 s) is paid once per run.
- Teacher-forced `score` for ouro (log-probabilities of a continuation, for policy gradient and served
  `policy_trace`): not built.
- LoRA on the looped stack: not built (`OuroTrainer::train_lora` refuses by name).

- End-to-end serving has one measured turn: an agent runtime against the
  served 1.4B checkpoint at a 6144-token context completed a tool-using task -
  it called the file-read tool, the tool returned the file's content, and its
  answer carried the number the file held (about 292 s, two model requests and
  one compaction retry). The turn needs the served template's reasoning block
  off: with it on, the model spent its whole output budget thinking and never
  answered. A 2.6B checkpoint in fp32 holds 4096 tokens only on a host, not on
  a 24 GB card; lower-precision serving weights (bf16 storage) are the step
  that puts a 6k context on one card. Prefix reuse and host offload stay off
  for the looped path until they are pass-aware.
- Decode speed: one token's graph is recorded once; a step writes the new
  K/V slot, two rope uniforms and the block table, then submits. A 16-token
  greedy continuation on the 1.4B checkpoint at a 6144-token capacity,
  including its load and a 5-token prefill, takes 10.5 s end to end.
- Prefill speed: the whole-prompt path now uses the flash causal attention
  kernel (no score slabs) and the register-tiled GEMM for prefill-sized linear
  slabs (the naive kernel still serves a decode row and the CPU JIT). A
  160-token prompt takes that path and matches the model's own greedy tokens;
  chunked paged prefill bounds memory beyond ~6k tokens.
- A served end-to-end check of a from-scratch model emitting a tool call over `/v1/chat/completions` (needs a model
  trained far enough to call tools).
- Threshold-selected decode across a batch whose rows exit on different
  passes.
