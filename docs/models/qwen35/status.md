# qwen35 — workstream ledger

Qwen3.5-35B-A3B: a hybrid decoder — 3-of-4 layers are **Gated DeltaNet**
(chunked linear attention), 1-of-4 are **GQA full attention** with a sigmoid
output gate and `partial_rotary_factor` RoPE — plus a 256-expert top-8
sparse MoE (softmax router, sigmoid-gated shared expert) on every layer, and
a natively-multimodal vision tower. Tracked from `.todo/qwen-3.5-support.md`;
plan approved 2026-08-09. This ledger records what's done, what's verified,
and what's still open, per `AGENTS.md`'s "write down what you learned in the
same change" rule — updated as the work lands, not after the fact.

## Sources

Built against the real checkpoint, not a secondhand description
(`docs/porting-playbook.md` §0): `Qwen/Qwen3.5-35B-A3B`'s `config.json`,
`model.safetensors.index.json`, `tokenizer_config.json`, and HuggingFace
Transformers' `modeling_qwen3_5_moe.py`/`configuration_qwen3_5_moe.py`
(main branch, fetched 2026-08-09), saved under
`/data/workspace/resources/qwen3.5/`. Weights are fetched as **GGUF**
(`bartowski/Qwen_Qwen3.5-35B-A3B-GGUF`, Q4_K_M + Q8_0 + the f16 mmproj), not
HF safetensors — this environment's measured HuggingFace throughput is
~3.2 MB/s regardless of connection parallelism, so fetching the ~72 GB bf16
checkpoint just to re-quantize it ourselves would cost hours more than
fetching an already-quantized GGUF directly.

## Done

- **P0 — config + param layout** (`crates/qwen35/src/config.rs`):
  `Qwen35Config` with the hybrid layer-type schedule (`LayerType::{Linear,
  Full}`, generated from `full_attention_interval` by the exact reference
  formula — verified against the real checkpoint's explicit `layer_types`
  list: full at 1-indexed multiples of 4, i.e. 0-indexed `{3,7,...,39}`),
  per-layer-type parameter shapes (GDN's `in_proj_{qkv,z,a,b}`/`conv1d`/
  `A_log`/`dt_bias`/`norm`/`out_proj`; GQA's doubled `q_proj` for the
  value+gate split), and a 256-expert MoE `param_list()` where every expert
  keeps its own indexed tensor name (`blocks.{l}.mlp.experts.{e}.{leaf}`) —
  matching `omni`'s convention, not concatenated into one `[E,ff,d]` buffer,
  since `model::moe`'s dispatch reads one 2-D expert weight per call. `tiny()`
  keeps every dimension that's distinct in the real config pairwise distinct
  (`docs/lessons.md` #4). JSON round-trip covers every field including LoRA
  (`docs/lessons.md` #23). Reuses `qwen::LoraCfg` rather than duplicating it.
- **P1 — GGUF import** (`crates/qwen35/src/import.rs`): streams every
  simple tensor one at a time via `checkpoint::gguf::MmapGguf`'s new
  `TensorSource` impl (P1a below); the three per-layer expert-stack tensors
  are read whole (~1 GB fp32 dequantized at the real shape, never the whole
  checkpoint) and sliced per expert on the host (verified contiguous and
  expert-index-ordered by a dedicated test, not assumed). **The full
  llama.cpp GGUF tensor-naming map was derived empirically** — range-read
  the real header directly off the partially-downloaded file, decoded by
  hand against the GGUF spec, cross-checked every mapped tensor's SHAPE
  (not just its name) against the real HF `index.json` — documented
  table-by-table in `import.rs`'s module doc. Key findings:
  - `qwen35moe.block_count = 41`, not 40: llama.cpp folds the MTP layer
    into the same `blk.N` index space at block 40 (confirmed via
    `blk.40.nextn.*` + its own attention tensors, matching HF's
    `mtp.layers.0.*`). MTP is out of scope for this port (deferred exactly
    like GLM's `check_glm_mtp`); `n_layers = block_count - 1`, and every
    `blk.40.*` tensor is dropped **loudly and counted**, never silently.
  - llama.cpp's GGUF conversion **already split** HF's fused
    `mlp.experts.gate_up_proj` into separate `ffn_gate_exps`/`ffn_up_exps`
    tensors — this import does not need to split a fused gate+up weight
    itself, only slice the already-separate per-expert stacks.
  - llama.cpp maps Gated DeltaNet onto its generic Mamba/SSM tensor-naming
    scheme (`ssm_alpha`/`ssm_beta`/`ssm_conv1d`/`ssm_norm`/`ssm_out`,
    `attn_qkv`/`attn_gate` reused from the attention side) — names bear no
    lexical resemblance to HF's `in_proj_{a,b,qkv,z}`/`conv1d`/`norm`/
    `out_proj`; every mapping is proven by shape equality, not name
    similarity.
  - `Qwen35Config`'s linear-attention head-shape fields
    (`linear_num_{key,value}_heads`, `linear_{key,value}_head_dim`) are
    derived from the REAL tensor shapes of the first Gated-DeltaNet block,
    not trusted from the GGUF KV's `qwen35moe.ssm.*` key names alone (those
    reuse generic Mamba field names — e.g. `ssm.time_step_rank` appears to
    actually hold `num_v_heads` — whose exact semantic mapping is
    plausible-but-unproven; the KV's `ssm.group_count` is used only as a
    cross-check assertion against the shape-derived value).
  - `import_gguf` fails loudly on any two-way coverage gap (every planned
    tensor written exactly once, every mapped source tensor consumed) —
    mirrors `qwen::import::brain_init_from_hf`'s discipline for a streaming
    writer.
  - Verified against the REAL checkpoint (`BRAIN_QWEN35_GGUF`-gated test)
    while ~49% downloaded: header (753 tensor infos + all KV) decoded
    correctly; `config_from_gguf`'s derived shape matches the real
    `config.json` exactly on every field, including the full-attention
    layer index set; the GGUF-embedded tokenizer extracts via
    `data::qwen_tokenizer::QwenBpe::from_gguf` and round-trips a real
    sentence through encode/decode exactly.
- **P1a — `TensorSource` for `MmapGguf`** (`crates/checkpoint/src/gguf.rs`,
  shared-library level, not qwen35-specific): gives GGUF checkpoints the
  same streaming construction path safetensors already has via
  `WeightReader` — peak host allocation ≈ one tensor's fp32 expansion,
  never the whole model. A dequant failure (IQ/TQ/MXFP4, still unimplemented)
  panics rather than reporting "not found", per the "fail loudly at the
  boundary" rule.
- **P2 — vision tower reuse decision**: verified (not yet wired — awaits
  P4/the forward pass) that Qwen3.5's vision config is **numerically
  identical** to `crates/qwenvl`'s existing `VisionConfig::qwen3_omni()`
  preset (depth 27, hidden 1152, 16 heads, intermediate 4304, patch 16,
  temporal patch 2, spatial merge 2, 2304 position embeddings, out_hidden
  2048) except `deepstack_indexes` (empty for 3.5, `[8,16,24]` for Omni).
  `VisionEncoder::encode_with_taps(..., &[])` already handles an empty taps
  list with no special-casing (confirmed by reading it — `encode()` itself
  is defined as exactly that call). **No fork of `crates/qwenvl` is needed**:
  `qwen35` will depend on `brain-qwenvl` directly and reuse
  `VisionEncoder`/`PatchMerger`/`vision_position_ids`/`vision_rope_tables`/
  `pos_embed_bilinear` as-is with a `VisionConfig{deepstack_indexes: vec![],
  ..VisionConfig::qwen3_omni()}` — a config-level reuse, not a code fork.
- **P3 — `rope2d_partial` kernel** (`crates/kernels/wgsl/rope2d_partial.wgsl`,
  `crates/model/src/block.rs::{rope2d_partial_fwd, rope2d_partial_bwd}`,
  shared-library level): Qwen3.5's full-attention layers combine
  `partial_rotary_factor=0.25` (only 64 of `head_dim=256` channels rotate)
  with interleaved M-RoPE — a combination neither existing RoPE kernel
  supported (`rope2d.wgsl` is table-driven but assumes the whole head
  rotates; `rope_partial.wgsl` supports a partial rotated prefix but
  computes its angle analytically, not from a per-axis table). The new
  kernel combines both: table-driven multi-axis lookup (reuses
  `qwenvl::mrope::mrope_tables` unchanged, called with `head_dim=rot_dim`)
  over only the first `2*half` channels of each head, addressed via the
  FULL `head_dim` stride so heads after the first aren't corrupted.
  `sign=-1` gives the exact inverse for backward — no separate `_bwd`
  kernel needed. Verified against a host-oracle test (rotated-prefix math
  AND bit-identical untouched tail) on both `BRAIN_DEVICE=cpu` and the
  default GPU backend (`docs/lessons.md` #5).
- **P4 — array-free 256-expert MoE router** (`crates/kernels/wgsl/
  router_gate.wgsl`, `router_gate_train.wgsl`, shared-library level, fixed
  IN PLACE not forked): these kernels hard-capped at `MAX_EXPERTS=128` via
  `var<function> array<f32,128>`/`array<bool,128>` scratch —
  `docs/lessons.md` #35b had already fixed the sibling `router_bwd.wgsl`
  this way and guarded (but not fixed) `router_gate_sigmoid.wgsl`, but left
  these two, since nothing had needed more than 128 experts yet. Qwen3.5's
  256 experts is the first real consumer past that wall. Fixed the same
  array-free way: the softmax numerator is stashed in the `gate`/`probs`
  OUTPUT buffer itself instead of a cached array, and the only remaining
  function-scope array (`sel_idx`, top-k selection bookkeeping) is bounded
  by `top_k` via a fixed `MAX_TOP_K=32`, never by `n_experts`. Verified
  against a real (non-same-composition) top-k host oracle at 8, 129, and
  256 experts on both backends; every existing MoE/GLM test re-run clean
  (no regression for existing 128-expert callers). See `docs/lessons.md`
  #35c for the full writeup. `router_gate_sigmoid.wgsl` (GLM's router,
  unused by Qwen3.5) is untouched, still behind its documented
  `assert!(<=64)` guard.

- **P5 — Gated DeltaNet chunked-recurrence FORWARD** (`crates/model/src/
  gdn.rs`, new `bmm`/`bmm_acc` general batched matmul primitive + 8 GDN-
  specific kernels + `exp.wgsl`/`sub.wgsl`, all shared-library level):
  `torch_chunk_gated_delta_rule` transcribed step-for-step. `bmm`/`bmm_acc`
  fill a real gap (no batched matmul — both operands varying per batch —
  existed anywhere in this engine before). The sequential per-chunk cumsum
  and the UT-transform's forward substitution are each one host dispatch per
  row (CPU JIT allows one `workgroupBarrier()` per kernel, so a true
  parallel scan can't fit in one dispatch); `gdn_ut_step.wgsl` keeps the
  frozen `attn0` and evolving `t_mat` as separate buffers specifically to
  avoid a same-buffer read/write race across threads in one dispatch, which
  WGSL's lack of cross-invocation ordering makes real (PyTorch's in-place
  single-tensor update is safe only because Python statement sequencing has
  no WGSL equivalent). Requires the caller to produce every per-token buffer
  **chunk-major** (`[n_chunks,B,H,C,D]`, chunk outermost) rather than the
  reference's `[B,H,T,D]` — documented at length in `gdn.rs`'s module doc;
  `qwen35::model`'s wiring (P8) must honor this. Verified against an
  independently re-derived f64 host oracle (plain nested loops, natural
  `[b,h,t,d]` indexing) at a tiny two-chunk shape with pairwise-distinct
  dims, on both backends: worst |delta| 8.99e-8 (GPU) / 1.61e-7 (CPU JIT).
  The oracle explicitly encodes a real, easy-to-miss reference detail
  confirmed against the live source: HF's `value = attn @ v_beta` (step 8)
  REASSIGNS the function's own `value` parameter, so every later `v_i` in
  the reference's per-chunk loop means that reassigned tensor, never the
  raw input value.
- **P6 — INT4 (q4) weight quantization at the shared `model::int8`-sibling
  level** (`crates/model/src/int4.rs`, `matmul_q4_dyn.wgsl`/
  `matmul_q4_gemv.wgsl`/`moe_linear_gated_q4.wgsl`): per-channel symmetric
  4-bit packing (8 nibbles/`u32`, range `[-7,7]` to stay exactly symmetric
  around zero, mirroring int8's `[-127,127]`), W4A8 (activations stay on the
  existing int8 dynamic-quant path unchanged — the only new device math is
  the 4-bit weight unpack), a naive-not-register-tiled GEMM (correctness-
  first per `docs/porting-playbook.md` §10). Notable finding: this engine's
  CPU-JIT (`crates/wgsl-cpu`) supports neither the WGSL `extractBits`
  builtin nor calling a user-defined WGSL function at all — both would have
  silently broken `BRAIN_DEVICE=cpu`; sign-extension is inlined via
  shift-up/arithmetic-shift-down directly in each kernel body instead
  (mirroring `dot4I8Packed`'s own CPU lowering). Verified bit-identical
  between the GPU and CPU-JIT backends (integer accumulation); measured
  tolerance at tiny synthetic shapes: cosine 0.998–0.999, rel-L2 4.8–5.8%.

## In progress

(nothing actively in flight right now — see "Not started" for the next
pieces, gated on P8's forward pass existing.)

## Not started

- **P2b — Gated DeltaNet BACKWARD**: `crates/model/tests/gdn_chunk_bwd.rs`
  is a documented `#[ignore]`d stub, not a silent gap. Needs a reverse sweep
  through the UT-transform (`i` from `chunk-1` down to `1`, mirroring
  `gdn_ut_step.wgsl`'s forward sweep) plus backward for every other step
  (mostly more `bmm`/`bmm_acc` calls with permuted operands), gradchecked at
  the same tiny shape `gdn_chunk_fwd.rs` uses, both backends. Required per
  this session's own AGENTS.md policy update (full backward is the default,
  not an opt-out) — blocks P9 and P12 below.
- **Attention output gate wiring**: `q_proj`'s doubled width (value + sigmoid
  gate) composes from existing `sigmoid.wgsl`+`mul.wgsl` — no new kernel
  needed, just wiring in the forward pass (P8 below).
- **P7 — device-side sparse MoE decode dispatch**: deliberately deferred to
  the performance-optimization phase (after P8 exists), per
  `docs/porting-playbook.md` §10 "correct-then-freeze" — `model::moe::
  expert_fwd`/`expert_fwd_compact` are correct for decode today, just not
  optimally fast at 256 experts (256×5=1280 dispatches/layer). Designing the
  dispatch redesign before the real forward pass exists risks building the
  wrong shape; revisit once P11 (serving) needs real decode throughput.
- **P8 — forward pass** (`crates/qwen35/src/model.rs`): GDN layer, GQA
  layer, MoE layer, vision splice — the SSA activation-cache convention,
  `Mixer::{Gdn,Gqa}` enum per layer (GLM's `Mlp::Dense`/`Mlp::Moe` precedent).
  Stage parity → single-forward parity → composed-loop parity against a
  golden dump. **Environment gap**: `tools/goldens/qwen35_dump_reference.py`
  needs `torch`+`transformers`, neither installed in this environment — the
  dumper can be written and documented but must be *run* on a machine with
  those deps; flag explicitly rather than silently skipping stage-parity
  when this is reached.
- **P9 — backward + `gradcheck::check_qwen35`**: full backward for GDN,
  gated GQA, MoE, vision; f64 FD oracle at a tiny hybrid config on both
  backends. Per the (approved, revised) project policy, this is required —
  not deferred as forward-only.
- **P10 — INT8/INT4 quantized inference wiring** for qwen35 specifically
  (`crates/qwen35/src/q8.rs`/`q4.rs`), once P8/P9 land.
- **P11 — `qwen35::serve::Engine`**: `model::serve::PagedDecoder` impl; KV
  paging for the 10 full-attention layers, a small O(1)-in-sequence-length
  per-sequence recurrent-state buffer (not paged) for the 30 GDN layers;
  layer-range `Shard`/`Pipeline` across both P40s at INT8/INT4 residency
  (35B doesn't fit one 24 GB card).
- **P12 — LoRA + memory-bounded finetune validation**: base frozen in
  INT4/INT8 (never fp32/bf16), only LoRA adapters trainable, on real
  weights, measured loss descent, bounded peak RSS/VRAM.
- **P13 — full serving contract**: `Qwen35Resident`, `caps.rs` (model id
  must be byte-exact `Qwen/Qwen3.5-35B-A3B`), D-Bus verification, OpenAI/
  Anthropic auto-exposure via `catalog::api_caps`, examples + manifest.
- **P14 — best-effort NPU export**: a `gdn_chunk` ONNX emitter (no existing
  linear-attention topology to copy) + a sparse gather-based expert-dispatch
  emitter (256-way dense unroll is impractical, ~30k MatMul nodes); fixed-`T`
  cache-free prefill only, explicitly stopping at "compiles + best-effort
  OpenVINO compile attempt" without chasing a working NPU run.
- **AGENTS.md policy edit**: replace the "imported models ship forward-only"
  allowance with "full backward + gradcheck is the default expectation" —
  land alongside P9.
- **Qwen3.5-27B (dense)**: a second named config once the shared hybrid code
  is proven — the dense sibling is the same code with `n_experts=1`-shaped
  MoE-as-dense and no linear-attention layers, additive not a fork.
