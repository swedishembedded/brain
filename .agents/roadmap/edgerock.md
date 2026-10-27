# edgerock - roadmap

Scope: EdgeRock-4B windowed hybrid decoder with a bounded anchor memory (`crates/qwen35` config variant, `model::anchor_memory`).

EdgeRock (`rockmatrix-ai/EdgeRock`, HF `EdgeRockForCausalLM`, text only, tied head,
546 tensors / 4.58B parameters) is the Qwen3.5 hybrid decoder (`crates/qwen35`) with
two changes on every full-attention layer: a causal sliding window on the attention
(`swa_window`, 1024), and an **anchor memory** beside it (`anchor_memory`: capacity
1024 records per KV group, commit blocks of `min(256, window)` tokens). The memory
reads the layer's residual input, keeps per KV group the best-scoring written
records of completed commit blocks (score descending, position ascending), reads
them by softmax attention with a zero sink, gates the read per head and projects it
with its own `o_proj`; the result is added to the attention output before the
residual add. It is a row in the arch registry (`edgerock`, package `brain-qwen35`),
not a crate: the memory math is generic (`model::anchor_memory`, four WGSL kernels).

The reference is `tools/goldens/edgerock_dump_reference.py` (an independent torch
implementation on the installed `transformers.models.qwen3_5`; the upstream code is
never run). `python3 tools/goldens/edgerock_dump_reference.py --seed 7` writes the
goldens under `testdata/golden/edgerock/{tiny_text,tiny_ties}/`.

## Done

- [x] M0: goldens dumper (`tiny_text`: random weights, the bank evicts; `tiny_ties`:
  `admission_up` zeroed so every score is 0 and the tie rule alone decides).
- [x] M1: arch row; `Qwen35Config::{swa_window, anchor_memory}` + `edgerock_tiny()`
  (pinned to the dumper) + `edgerock_4b()`; HF `config.json` parsing that refuses an
  unknown `fixed_memory_revision`, a `fixed_memory_swa_layers` that is not exactly
  the full-attention layers, and every missing/zero `independent_memory` field;
  `param_list()` `blocks.{l}.memory.*` (the three memory RMSNorms are `(1+w)` norms
  and take the import fold); the text-only `model.layers.*` layout through the one
  `import::classify` (also the lazy `fp8_source`); init; forward through
  `Qwen35::logits_all` (windowed `gqa_mixer_fwd_windowed`, memory added before the
  residual add); decode, chunked prefill, streaming and training paths refuse the
  config loudly (`Qwen35Config::require_plain_decoder`, a training build with an
  anchor memory is refused at construction).
  Gates: `crates/qwen35/tests/edgerock_golden_parity.rs` (both variants, CPU and
  default backend), `edgerock_config.rs`, `edgerock_hf_fixture.rs` (the release
  layout through config, lazy source, eager import and a forward),
  `crates/model/tests/anchor_memory.rs` (selection and indexed read against host
  definitions), the import and init unit tests.

Parity at the tiny dims (`edgerock_golden_parity`, per stage cosine / max_abs /
rel_l2, 24 stages per variant): cosine 1.0000000000 at every stage; worst max_abs
4.1e-5 (CPU backend) and 2.9e-5 (default backend, wgpu), worst rel_l2 4.0e-6. The
bank of both memory layers (`layer{3,7}.mem.refs`, 360 slots each) equals the
reference exactly in both variants on both backends; flipping the tie rule in
`anchor_select.wgsl` fails the ties variant and the model-crate selection test.

- [x] M2: training at tiny scale. `model::anchor_memory::anchor_memory_bwd` (generic) differentiates the
  memory end to end: the output projection and per-head gate, the indexed read (dP, softmax backward,
  query gradient, then key/value gradients **gathered** per record through an inverse bank map built
  by `anchor_slot_inverse`, so no atomics and `O(rows * heads * capacity * head_dim)`), the rotary and
  per-head norms, the writer MLP and the input norm, with both `u` paths (query/gate and writer/key/value)
  joined. The admission scorer gets no gradient and is `Role::Frozen` (a discrete sort reads it; weight
  decay must not touch it either). Six new kernels (`anchor_read_bwd_{dp,q,kv}`, `anchor_slot_inverse`,
  `anchor_read_gate_{bwd,wgrad}`) around `softmax_k_dx`, `row_dot`, `rmsnorm_bwd`, `swiglu_bwd`. The
  windowed attention backward needs nothing new: its saved probabilities are zero outside the window.
  Wired into `backward_over` for Full layers (the memory's `d_h` joins the layer's residual gradient;
  the recompute policy rebuilds the memory's activations through `layer_forward`). The training
  refusal is gone. LoRA targets `memory.{q,k,v,o}_proj` and `memory.write_{gate,up,down}_proj`
  (`memory_lora_targets()`, asked for through `lora_cfg_for(&cfg, rank, alpha)`; `LoraSites`,
  `hf_linear_name`), never the scorer. The arch row declares
  `full_ft` and `lora(F32)` with the matrix rows that back them.
  Gates: `gradcheck::check_edgerock` (b = 2, every trainable tensor of the memory and the windowed
  layer, banks pinned: a finite-difference step that moves a bank fails loudly instead of measuring a
  jump; rel error 1e-4 to 3e-2 over the memory tensors, no hollow memory entry),
  `check_edgerock_lora` (all seven memory adapter pairs), `model/tests/gqa_mixer_windowed.rs`
  (the window's forward and backward against finite differences), `model/tests/anchor_memory.rs`
  (the read backward against an explicit scatter), `qwen35/tests/edgerock_training.rs` (a batch of two
  sequences equals two single batches in loss and gradients, a fixed-batch loop cuts the loss by 40
  percent, zeroing `read_scale` moves the trained loss, the scorer is never optimised),
  `lora_freezes_base.rs` (memory base, norms, gate and scorer unchanged under LoRA).
- [x] CUDA parity. The memory's projections are `MATMUL` kernel steps, exact fp32 on every backend; only
  `Ops::matmul` takes the TF32 tensor-core products, and those perturb the *residual stream* the
  scorer reads, not the scorer's own arithmetic. So the banks are exact exactly where the whole model
  is: `edgerock_golden_parity` runs CUDA pinned to `GemmPrecision::Fp32Reference` (stages to 1e-4, banks
  exactly equal, both variants), and at the default tier (TF32) with the `lora_gemm_precision`
  convention (cosine 0.999) and at least 95 percent of bank slots equal: near-tied scores swap rank
  there (1 to 2 percent of slots on `tiny_text`), which the released bf16 model does as well.

- [x] M5 (forward part): the released weights against the independent reference
  forward (`tools/goldens/edgerock_dump_real_reference.py`, fp32 CPU torch, a
  1400-token prompt that slides the window and commits five blocks, then asks about
  a fact stated at its start). `crates/qwen35/tests/edgerock_real_weights.rs`
  (ignored: needs the store checkpoint and the dumped golden), bf16 weight tier on
  the CPU backend: every `layer{i}.out` cosine >= 0.9999999997 (worst max_abs
  1.9e-3 at layer 26, rel_l2 <= 2.3e-5), tail logits rel_l2 4.1e-6, the same greedy
  answer, and 99.54-99.85% of the slots of every memory bank equal (near-tied
  scores swap rank). The reference itself answers the probe correctly with the
  memory on (" 4817.", the fact sits ~1380 tokens back, outside the window) and
  wrongly with `read_scale` zeroed (" 48617").

## Not yet done

- [ ] M3: decode state. Per sequence and memory layer: the committed bank (`key`/`value`
  `[groups, capacity, D]`, `score` and `position` `[groups, capacity]`), the pending
  block, and the window's ring KV; step, chunked prefill and batched decode parity
  against `logits_all`; state bytes independent of length.
- [ ] M4: serving: engine on the generic scheduler, snapshot prefix cache over the
  fixed state, conformance suite, catalog, capabilities, CLI, `brain pull
  rockmatrix-ai/EdgeRock`, model page promoted from "recognized" to usable.
- [ ] M5: real weights (BF16 load, reference parity with the CPU torch reference,
  generation sanity, long-context retrieval).
- [ ] M6: performance, training included: real-weight training throughput and memory (the saved `probs`
  are `[rows * heads, capacity + 1]` per memory layer, held until the backward or recomputed).
  Inference and training: The memory read materialises `[rows * heads, capacity + 1]`
  logits and probabilities and the windowed attention the `[B,H,T,T]` slabs (no
  windowed flash kernel yet); `anchor_select` ranks by counting, `O((capacity + chunk)^2)`
  per group and block; both want the cooperative/flash treatment per `kernels.md` F.
- [ ] The memory's projections are plain fp32 matmuls over the parameter store: no
  quantized tier, no adapter site. Quantizing them is only safe for the read side
  (`q/k/v/o`, writer); the scorer stays fp32.
- [ ] A real-weight parity run must first confirm `read_scale != 0` on the memory
  layers: a zero read gate silences the memory entirely, and parity would then pass
  without exercising it (the tiny goldens randomise every parameter for the same
  reason).
