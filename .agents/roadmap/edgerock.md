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

## Not yet done

- [ ] M2: backward + gradcheck of the memory (the admission selection is
  non-differentiable: gradients flow through the read, the gate, the written keys and
  values, not through the scores) and of the windowed attention; LoRA on the memory
  projections; batched training `b > 1` (the memory kernels already take `n_seq`; the
  rope tables are indexed `row % rows`, check them at `b > 1`).
- [ ] M3: decode state. Per sequence and memory layer: the committed bank (`key`/`value`
  `[groups, capacity, D]`, `score` and `position` `[groups, capacity]`), the pending
  block, and the window's ring KV; step, chunked prefill and batched decode parity
  against `logits_all`; state bytes independent of length.
- [ ] M4: serving: engine on the generic scheduler, snapshot prefix cache over the
  fixed state, conformance suite, catalog, capabilities, CLI, `brain pull
  rockmatrix-ai/EdgeRock`, model page promoted from "recognized" to usable.
- [ ] M5: real weights (BF16 load, reference parity with the CPU torch reference,
  generation sanity, long-context retrieval).
- [ ] M6: performance. The memory read materialises `[rows * heads, capacity + 1]`
  logits and probabilities and the windowed attention the `[B,H,T,T]` slabs (no
  windowed flash kernel yet); `anchor_select` ranks by counting, `O((capacity + chunk)^2)`
  per group and block; both want the cooperative/flash treatment per `kernels.md` F.
- [ ] CUDA parity. With the default GEMM precision the CUDA backend runs `matmul`
  as TF32 tensor-core products: every stage of the tiny golden differs by about 3e-3
  relative (the plain qwen35 path too), and the admission scores perturbed that much
  reorder near-tied bank slots (4 of 360 at layer 3 of `tiny_text`). Admission scoring
  must be exact fp32 on every device (route `AdmissionDown`/`AdmissionUp` to the
  fp32 reference GEMM), and a real-weight check should judge the bank by its scores
  and the logits, not by slot equality, since the released model itself runs bf16.
- [ ] The memory's projections are plain fp32 matmuls over the parameter store: no
  quantized tier, no adapter site. Quantizing them is only safe for the read side
  (`q/k/v/o`, writer); the scorer stays fp32.
- [ ] A real-weight parity run must first confirm `read_scale != 0` on the memory
  layers: a zero read gate silences the memory entirely, and parity would then pass
  without exercising it (the tiny goldens randomise every parameter for the same
  reason).
