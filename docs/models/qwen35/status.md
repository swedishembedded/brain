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

- **P0 — config + param layout** (`crates/qwen35moe/src/config.rs`):
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
  (`docs/lessons.md` #23). Reuses `qwen3::LoraCfg` rather than duplicating it.
- **P1 — GGUF import** (`crates/qwen35moe/src/import.rs`): streams every
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
    mirrors `qwen3::import::brain_init_from_hf`'s discipline for a streaming
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
  `qwen35moe::model`'s wiring (P8) must honor this. Verified against an
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

- **P8 — forward pass, TEXT ONLY** (`crates/qwen35moe/src/model.rs`,
  `crates/qwen35moe/src/init.rs`): assembles this session's already-verified
  primitives (`model::gdn`, `model::moe`, `model::block`'s GQA/
  `rope2d_partial`) into the full 40-layer hybrid decoder — no new math
  derived here, pure wiring, following GLM's per-layer-enum/SSA-cache
  precedent (with the enum axis inverted: GDN-vs-GQA on the attention
  sublayer, MoE universal every layer, opposite of GLM's dense-vs-MoE MLP
  axis). Two new small kernels: `gdn_decay_gate.wgsl` (the decay gate,
  numerically-stable `softplus`) and `gdn_layout_permute.wgsl` (the
  token-major ↔ chunk-major permute at the GDN layer boundary — a genuine
  5-index permute neither existing layout kernel covers; its flat-index
  arithmetic checked by hand against `model::gdn`'s own documented
  convention). Everything else reuses existing kernels, notably
  `concat_split.wgsl` for both the GDN qkv split (whole-row) and the GQA
  q_proj value/gate split (per-head interleaved, confirmed against the real
  reference's `torch.chunk(...,dim=-1)` semantics).

  **Resolved empirically, not by guessing**: the reference's
  `Qwen3_5MoeRMSNorm` computes `output*(1+weight)` with `weight`
  zero-initialized — but reading real F32 norm-weight tensors directly out
  of the (by then fully downloaded) real GGUF checkpoint
  (`blk.{0,1}.attn_norm.weight`, `output_norm.weight`) shows them centered
  around 1.0–2.6, not 0.0 — confirming llama.cpp's GGUF conversion already
  bakes the `+1` in. This engine's shared `rmsnorm.wgsl` (plain multiply,
  used by every model) is therefore exactly correct for our GGUF import
  path as-is, no fix needed. Recorded here so it is never "corrected" later
  by someone who reads only the reference source and not the actual
  checkpoint data.

  Implements the full `model::Model` trait; `backward`/`zero_grads`/
  `adamw_step` panic with a named message pointing at `model::gdn`'s missing
  backward (P2b below) rather than silently no-op'ing.

  Verified: existing config/import tests unaffected; a tiny-config smoke
  test (both layer types, 6-expert MoE, a genuine 3-chunk GDN recurrence)
  passes on both `BRAIN_DEVICE=cpu` and the default GPU backend — finite,
  deterministic across repeated `forward()` calls. Zero warnings; kernel
  table current (384 kernels).

  **Honest scope note, unchanged**: true numerical parity against the real
  HF reference is not achievable in this environment (no `torch`/
  `transformers` installed) — only structural correctness is claimed.
  **Vision splice is explicitly NOT in this pass** — text-only, tracked
  separately as its own task.

- **P10a — INT8 (DP4A) quantized inference, SINGLE-GPU** (`crates/qwen35moe/src/
  q8.rs`): mirrors `qwen3::q8::Q8`'s recipe (weights quantized once via
  `model::int8::quantize_weight`, activations quantized on-device per
  forward with a dynamic per-token scale), adapted for qwen35's two mixer
  shapes (GDN vs GQA) and its 256-routed-expert MoE. Quantized: every mixer
  projection + every routed expert's gate/up/down (the dominant parameter
  mass — 256×3 tensors/layer vs. 4-5 mixer tensors). Left fp32: the router
  (a noised logit can flip which experts get selected, a qualitatively
  worse failure mode than a noised activation, and not a throughput
  bottleneck), the shared expert (1/256th the mass, no int8 kernel exists
  for it), embeddings/`lm_head` (embed is a gather not a GEMM; `lm_head`
  kept fp32 for the same precision-sensitive-logits reason as the router,
  matching `qwen3::q8`'s own choice). Reuses `model::moe`'s existing
  `Lin8`/`ExpertScratch8`/`expert_fwd_i8` unchanged via a thin adapter.

  Two real findings during integration: (1) initially wired the STATIC
  per-tensor-scale `matmul_i8` instead of the DYNAMIC `matmul_i8_dyn`
  `Q8::quant`/`mm8` actually produce — caught by a bind-group-count
  mismatch; (2) confirmed `matmul_i8_dyn`/`moe_linear_gated_i8` cannot run
  on the CPU (Cranelift JIT) backend at all (a pre-existing, engine-wide
  limitation, not a new bug — `qwen3::q8`'s own tests never exercise an int8
  forward on CPU either) — handled by asserting the exact known panic on
  CPU and running the real dual-backend-parity check
  (`docs/lessons.md` #5) against fp32 on the GPU backend instead.

  Verified (independently reproduced): int8 vs fp32 forward at a small
  int8-packable config — **cosine=1.000000000, rel_l2=0.0000066** on real
  P40 hardware, mutation-verified sensitive to a real regression (scaling
  one weight's scale by 1.3× moves rel_l2 to 6.9e-4; by 50× drops cosine to
  0.40). All 21 pre-existing + new tests pass, zero regression.

  **Single-GPU only** — multi-GPU layer sharding and GGUF-direct
  per-expert-slice streaming (required at the real 35B scale, since even
  int8 is ~35GB and doesn't fit one 24GB P40) are separate, larger
  follow-on work (P11 below). A research pass into the existing precedent
  (`crates/omni/src/int8_thinker_resident.rs`, validated on real 2×P40
  hardware) found: partitioning is by contiguous LAYER RANGE (never by
  expert — an expert-split would break the "quantize the shared input once"
  contract), cross-device handoff is a plain host round-trip (`gpu.read` →
  `gpu.write`, "negligible bytes... at any realistic decode batch"), and
  placement goes through `residency::MultiDeviceResidentModel`/
  `Executor::register_multi` (NOT `model::shard::Shardable`/`Pipeline`,
  which is training-shaped and requires a whole-model host `HashMap` up
  front — a real non-starter at 35B). The one genuinely new piece P11 needs
  beyond that precedent: GGUF has no per-expert tensor names (each layer's
  256 experts are one stacked `[256,ff,hidden]` tensor), so a quantized
  loader must read that whole stack once (~1GB transient) and slice+quantize
  256× before dropping it, rather than the one-`with_tensor`-call-per-name
  loop `Q8::build`/`Int8ThinkerResident`'s own loaders use today.

- **P2b — Gated DeltaNet BACKWARD** (`model::gdn::gdn_chunk_bwd`): full
  reverse-mode backward through all 11 forward steps, closing the gap P5
  left open. A complete hand derivation (every one of ~20 backward steps,
  including the UT-transform's forward-substitution reverse sweep) was
  worked out first and independently spot-checked kernel-by-kernel against
  the implementation before merging — the two highest-risk kernels (the
  UT-transform backward pair, the two-direction decay-mask reduction)
  matched the derivation exactly on inspection.

  New `gdn_chunk_fwd_train` (a training-mode forward variant, sharing a
  refactored `gdn_chunk_fwd_prefix` with the unchanged inference
  `gdn_chunk_fwd`) additionally saves per-chunk history — backward runs the
  chunk loop in REVERSE and needs every chunk's snapshot of
  `q_scaled`/`decay_scale`/`decayed_k`/`v_prime`/`v_new` and the full
  per-chunk recurrent-state history, not just the last one the inference
  path keeps.

  9 new kernels. The two hardest: `gdn_ut_bwd_dattn0`/`gdn_ut_bwd_dtmat`
  (one dispatch pair per row, `i` from `chunk-1` down to `1`, mirroring the
  forward's own per-row shape; frozen forward values and the evolving
  gradient kept in separate buffers for the same race-avoidance reason the
  forward kernel does — WGSL invocations within one dispatch have no
  read-before-write ordering). `gdn_decay_mask_bwd` needs a row-sum AND a
  column-sum over the same `[bhc,c,c]` tensor (the causal decay mask's
  `g_cs[i]` and `g_cs[j]` roles) — solved as one kernel dispatched twice
  with a mode flag (no atomics, so one thread can't cheaply do both
  directions at once). Plus `gdn_chunk_reverse_cumsum_step` (suffix-sum),
  `row_dot` (a new generic per-row dot-product primitive needed at five
  different backward sites), and three smaller elementwise/reduction
  kernels for the state-decay/decay-scale terms.

  Verified: finite-difference gradcheck at the same tiny two-chunk shape
  `gdn_chunk_fwd.rs` uses, both backends — worst relative error **2.7e-5
  (GPU/Vulkan) / 6.0e-5 (CPU/Cranelift JIT)**, both ~15-40× tighter than the
  1e-3 target. `gdn_chunk_fwd_train`'s forward output matches
  `gdn_chunk_fwd`'s bit-for-bit (identical error numbers to the pre-existing
  forward test), confirming the shared-prefix refactor changed no behavior.

- **P9 — qwen35-level backward + `gradcheck::check_qwen35`** (`crates/qwen35moe/
  src/model.rs`; one new kernel, `gdn_decay_gate_bwd.wgsl`): wires the three
  already-proven primitives (`model::gdn::gdn_chunk_bwd`, `model::block::
  gqa_bwd`, `model::moe::moe_layer_bwd`) together into `Qwen35::backward()`,
  which no longer panics. Pure integration for everything except the one new
  kernel — the gradient math itself was not re-derived here.

  **Two construction paths, not one**: `Qwen35::new`/`new_i8` are unchanged
  (still `Role::Frozen`, still panic on `backward()`/`zero_grads()`/
  `adamw_step()`); a new `Qwen35::new_train` builds every weight
  `Role::Trainable` (full-parameter — no LoRA subset, per the approved scope)
  and makes `forward()` additionally populate a `TrainActs` activation cache
  (`RefCell<Option<TrainActs>>`, taken and consumed by the very next
  `backward()` call) that `layer_gdn_fwd`/`layer_gqa_fwd`/`moe_sublayer`
  populate via a new `is_train`-gated branch mirroring the existing `q8`
  int8 branch. The GDN branch's key wiring point: `layer_gdn_fwd`'s training
  path calls `gdn_chunk_fwd_train` (saving the per-chunk history
  `gdn_chunk_bwd` needs) instead of the inference `gdn_chunk_fwd`, via new
  owned `GdnScratchTrainBufs`/`GdnBwdScratchBufs` wrappers mirroring the
  existing `GdnScratchBufs`.

  **The one new kernel**, `gdn_decay_gate_bwd.wgsl`: backward of
  `g = -exp(A_log)*softplus(a_proj+dt_bias)` w.r.t. `a_proj` (`d_a_proj =
  d_g * -exp(A_log)*sigmoid(a_proj+dt_bias)`, recomputing the un-saved
  pre-activation); `d_A_log`/`d_dt_bias` are NOT computed inside it (per this
  repo's "one kernel, one job" convention) — they compose from its output via
  `mul.wgsl`+`bias_grad.wgsl` and `bias_grad.wgsl` alone, respectively.
  Everything else reuses existing kernels: `concat2.wgsl` (twice, for the
  3-way qkv split's adjoint) for the GDN mixer, once for GQA's 2-way q-gate
  split; `gdn_layout_permute.wgsl` again with the flag flipped for its own
  backward (a permutation's adjoint is the inverse permutation, no new
  kernel); `l2norm_scale_dx.wgsl`, `kv_expand_bwd.wgsl`, `conv1d_bwd`
  (`audio::conv`), `sigmoid_bwd.wgsl`, `silu_bwd.wgsl`, `rope2d_partial_bwd`
  (sign=-1, already built for exactly this), `matmul_dx.wgsl`/
  `matmul_dw.wgsl` (naive tier only — no tiled-GEMM selection, correctness-
  first per `docs/porting-playbook.md` §10, appropriate at this tiny
  gradcheck scale), all newly REGISTERED in qwen35's own `PIPELINES` (they
  already existed in `crates/kernels`).

  **The one hand-derived (not composed) piece**: the sigmoid-gated shared
  expert's backward. `model::moe` has no helper for it — `crates/glm`'s
  shared expert has no gate at all, and no other model composes this exact
  shape — so it is hand-derived from `scale_row.wgsl` (self-adjoint w.r.t.
  its scaled operand), `row_dot.wgsl`, `sigmoid_bwd.wgsl`, `swiglu_bwd`, and
  `proj_bwd`. **Ordering constraint discovered while wiring it**:
  `moe_layer_bwd`'s own contract requires its caller-supplied
  `router_weight_bwd` to be the FIRST (`acc=0`) touch to the shared MoE-input
  gradient `d_xn2` — but the shared expert's own chain also writes `d_xn2`.
  Resolved by ordering, not a new accumulator: the routed-MoE backward
  (`moe_layer_bwd`, which establishes `d_xn2`'s base value per its own
  documented phase order) runs FIRST in program order, and the shared
  expert's three `d_xn2` touches all use `acc=1` (accumulate) on top —
  correct because this engine's `Step`s execute strictly in submission
  order, the same assumption every other multi-source accumulator in this
  file already relies on.

  **`moe_layer_bwd`'s phase ordering, followed exactly as documented**:
  Phase A (every expert's `d_gate` column, `expert_dgate`, needed before
  Phase B since the router needs the WHOLE `d_gate` row) → Phase B
  (`router_bwd`'s kernel-level router backward, THEN the router weight's own
  dense-linear backward, supplied here as `router_weight_bwd_steps`) → Phase
  C (every expert's SwiGLU backward, `expert_bwd`, accumulating into
  `d_xn2`). `moe_layer_bwd` itself enforces this order (I only supply the
  buffers/ids and the caller-owed `router_weight_bwd_steps`, not the
  ordering).

  **Gradcheck**: went through the generic `gradcheck` crate machinery
  (`gradcheck::check_qwen35`, `crates/gradcheck/src/lib.rs`), NOT a bespoke
  `crates/qwen35moe/tests/` harness — `Qwen35` already implements the full
  `model::Model` trait (forward, backward, zero_grads, adamw_step, read_grad,
  read/write_weight), so the blanket `impl<M: model::Model> CheckModel for M`
  makes it checkable with zero new test-harness code, exactly like
  `check_qwen`/`check_glm`/`check_moe`. A hybrid config smaller than
  `Qwen35Config::tiny()` (`n_layers: 4, n_experts: 3, top_k: 3` — `top_k ==
  n_experts` removes the hard top-k selection boundary finite differences
  cannot see through, the same mitigation `check_moe`/`check_glm` already
  use; `t: 12` still gives GDN a genuine 3-chunk recurrence at chunk size 4)
  keeps `directional_check`'s `O(n_dirs)`-forward-passes-per-tensor cost
  manageable. Verified on **both** `BRAIN_DEVICE=cpu` and the default GPU
  backend (`docs/lessons.md` #5): both pass at the workspace's own fp32
  GPU-directional-check tolerance `(atol=4e-3, rtol=8e-2)` — the same gate
  `check_qwen`/`check_glm`/`check_seq2seq`/`check_moe` all use (the tighter
  "block FD < 1e-4, model FD < 1e-3" figures in `docs/porting-playbook.md`
  §8 are FLUX.2's own f64 HOST-trainer check, a different, much tighter
  regime than the fp32-GPU `directional_check` family qwen35 belongs to).
  Observed worst relative error: **1.87e-2** (CPU/Cranelift JIT,
  `blocks.3.self_attn.o_proj.weight`) / **1.20e-2** (GPU/Vulkan,
  `blocks.3.self_attn.q_proj.weight`) — both well inside the gate, in line
  with every sibling model's own numbers at this tolerance.

  **The first passing run was hollow for the whole GDN branch — caught on
  review, not by the checker.** Every `blocks.{0,1,2}.linear_attn.*`
  parameter's finite-difference numeric gradient came back *exactly* `0.0`,
  and the `(4e-3, 8e-2)` gate's absolute floor swallowed the distinction from
  a genuinely tiny one. Direct probing (perturb `A_log` by ±10 by hand,
  re-`forward()`, compare loss) confirmed the loss was bit-identical —
  `tiny()`'s standard `std=0.02` init, cascaded through `in_proj_qkv` then a
  *depthwise* causal conv1d (only 4 summed terms per channel, far less
  central-limit averaging than a `d_model`-wide matmul) then SiLU, collapses
  `query`/`key` to ~1e-6 before `l2norm_scale.wgsl`, whose `eps=1e-6`
  (correct for the real `d_model=2048`) then dominates the normalization —
  the entire recurrence downstream reads back at ~1e-11 regardless of decay/
  beta/gate parameters. Full writeup: `docs/lessons.md` #40. Fixed in the
  **test harness only** (`qwen35_gradcheck_harness`, `crates/gradcheck/src/
  lib.rs`, overrides `in_proj_qkv.weight`/`conv1d.weight` to `std=1.0` post-
  init) — neither `qwen35moe::init`'s production init nor `model::gdn`'s `eps`
  needed to change. Re-verified after the fix: every GDN-layer parameter
  (`in_proj_qkv.weight`, `conv1d.weight`, `in_proj_z.weight`,
  `out_proj.weight`, `norm.weight`) now shows real, non-floor-dominated
  agreement, and the new `check_qwen35_a_log_elementwise` test (per-entry FD
  on `blocks.2.linear_attn.A_log`, following `check_t5_rel_bias_elementwise`'s
  own precedent for a cross-stage-folded parameter `directional_check` can't
  resolve) confirms `A_log`'s own gradient directly. A follow-up ad hoc
  elementwise probe on `blocks.2.ln1.weight` at `eps=1e-2` initially looked
  alarming (up to 3x mismatch, one sign flip) but resolved cleanly at
  `eps=1e-3` — finite-difference truncation error from the UT-transform's
  nonlinearity at too-large an eps, not a backward bug (T5's own elementwise
  doc already documents this exact eps-sensitivity U-curve).

  **Not fully confident on** (flagged rather than silently assumed):
  `router_bwd`'s `Softmax` variant's aux-loss `fe` (expert-usage-fraction)
  scratch buffer's own zero/overwrite contract was trusted from
  `model::moe`'s existing (independently gradient-checked) composition
  rather than re-derived here; the naive (non-tiled) `matmul_dx`/`matmul_dw`
  choice for every weight gradient is a real perf gap at the 35B scale (out
  of scope for this integration pass, called out for the eventual
  performance-optimization pass per `docs/porting-playbook.md` §10).

- **CLI entry point** (`crates/cli/src/qwen35moe_cli.rs`, commit `4a7922b8`):
  `brain qwen35moe import --gguf F --out qwen35.safetensors` (thin wrapper
  over `import_gguf`) and `brain qwen35moe infer --weights F (--tokenizer T
  | --gguf G) --prompt "..."` (loads via `checkpoint::load(path).by_role("")`
  — the same simplest load path every other model crate uses — and generates
  via the new `qwen35moe::sample::generate_kv`, mirroring `qwen3::sample`'s
  own structure over `Qwen35::step`). `--gguf` on `infer` re-opens the
  original checkpoint's embedded tokenizer directly (`import_gguf` only
  writes tensors, not the tokenizer) so a full `import` -> `infer` round
  trip needs no separate `tokenizer.json` extraction step.

- **P14 — best-effort NPU export** (done, `crates/npu/src/qwen35moe_{topology,export}.rs`,
  commit `87fad8ae`): a `gdn_chunk` ONNX emitter (cumsum via `MatMul` against
  a triangular matrix, the UT-transform's `(I-attn0)^-1` via its exact
  Neumann series since `attn0` is nilpotent, the cross-chunk recurrence
  statically unrolled per chunk — no existing linear-attention topology to
  copy) + a sparse gather-based expert-dispatch emitter (per-layer `[E,in,
  out]` stacked initializer, `Softmax->TopK->ReduceSum->Div->Gather`, cost
  scales with `top_k` not `E` — a 256-way dense unroll would be ~30k MatMul
  nodes). Router-math translation verified against `model::moe`'s real
  algorithm via a standalone cross-check (2000 random cases, `E` up to 256):
  worst gate-value diff 5.96e-8, selected-expert sets exact every time.
  Fixed-`T` cache-free prefill only, text-only, explicitly stopping at
  "compiles + best-effort OpenVINO compile attempt" without chasing a
  working NPU run — confirmed the real stopping point on this box (NPU
  compiler plugins present, no core OpenVINO runtime, probing it hangs
  rather than failing cleanly) rather than assuming one. `brain qwen35moe
  export --weights F --out model.onnx --seq T` is the CLI entry point.

## In progress

- **P11 — `qwen35moe::serve::Engine`**: `model::serve::PagedDecoder` impl; KV
  paging for the 10 full-attention layers, a small O(1)-in-sequence-length
  per-sequence recurrent-state buffer (not paged) for the 30 GDN layers;
  layer-range `Shard`/`Pipeline` across both P40s at INT8/INT4 residency
  (35B doesn't fit one 24 GB card). Broken into sub-steps since a full paged
  multi-sequence engine is a lot of surface area to review at once — climbing
  the same "small proven piece, then integrate" ladder P9 (backward) used.
  - **P11a — decode-step primitives, shared-library level** (done,
    `crates/model/src/gdn.rs`, commit `f0cdb28c`): `gdn_recurrent_step` (the
    single-token GDN state update, derived directly from the reference
    `torch_recurrent_gated_delta_rule` in `modeling_qwen3_5_moe.py` —
    decay state, subtract what it already predicts, scale by beta, rank-1
    update, read back out; composes entirely from EXISTING kernels
    (`gdn_state_decay`/`bmm`/`bmm_acc`/`sub`/`row_scale`) — no new kernel
    needed here) and `gdn_causal_conv1d_step` (+ one new kernel,
    `causal_conv1d_step.wgsl`: the streaming, ring-buffer-state sibling of
    `conv1d_fwd`'s whole-sequence causal depthwise conv — re-running the
    whole-sequence kernel every decode step would be `O(T^2)`). Both
    inference-only (no backward). Validated against already-proven
    references: `gdn_recurrent_step` vs. `gdn_chunk_fwd` at `chunk=1`
    (worst delta 1.2e-7 GPU / 8.9e-8 CPU), `gdn_causal_conv1d_step` vs.
    `conv1d_fwd` over a short sequence (bit-identical, both backends).
  - **P11b — single-sequence incremental decode wiring** (done,
    `crates/qwen35moe/src/model.rs`, commit `12438b9e`): `Qwen35::step`
    (+ `reset_decode_cache`/`decode_pos`), analogous to `qwen3::Qwen::step`/
    `decode_at` — persistent per-layer state (plain non-paged KV cache for
    GQA layers via `model::block::gqa_decode_step`, `gdn_recurrent_step`'s
    `state` + `gdn_causal_conv1d_step`'s `hist` for GDN layers), single
    sequence, fp32, text-only. Single-position M-RoPE (no exact
    qwen35moe-internal precedent — `qwen3::Qwen` has no GDN layers to mirror
    against) resolved by recomputing a fresh 1-row `qwenvl::mrope::
    mrope_tables` table per step, mirroring `qwen3::Qwen::step_mrope`'s own
    per-step-table pattern for the same structural reason
    (`rope2d_partial_fwd`'s `row % tmod` addressing can't reach into a larger
    precomputed table at `rows=1`). Validated against `logits_all` (whole-
    sequence prefill) replayed step-by-step over the same tokens at a tiny
    hybrid config (both layer types, a real multi-chunk GDN prefill): worst
    maxabs **2.98e-8** on both backends — fp32 machine epsilon, since both
    paths run the identical kernels in a different dispatch order, not
    merely "close by tolerance". No decode-specific MoE function was needed
    (`moe_sublayer` already works correctly at `n=1`, confirmed by the
    passing test).
  - **P11c — `qwen35moe::serve::Engine`, single-GPU** (done,
    `crates/qwen35moe/src/serve.rs`, commit `92f23669`): `PagedDecoder`
    impl, single GPU, one truly-active sequence at a time. Solves the "two
    kinds of per-sequence state" problem without touching the shared
    `PagedDecoder` trait or `model::paged`: `block_size == max_seq_len`
    makes a sequence's first (and only) physical block id a stable key into
    a private `GdnSlot` map the `Engine` owns. Required refactoring
    `Qwen35::run_decode_step`/`layer_gdn_decode_step`/`layer_gqa_decode_step`
    to take their per-sequence resources as an explicit `DecodeCaches`
    parameter instead of reading instance-wide fields — `Qwen35::step`
    itself unchanged (thin wrapper), confirmed by `decode_step.rs` still
    passing bit-for-bit. Deliberately deferred (named in the module doc, not
    silently absorbed): prefix-cache reuse, chunked/batched prefill,
    multi-sequence GPU batching, int8/int4 paged KV, speculative decode,
    multi-GPU layer sharding. Validated: admit one request, drive to
    completion through `Scheduler<Engine>`, compare against
    `Qwen35::step`-driven generation — exact token match on both backends.
  - **Not started**: multi-sequence GPU batching, prefix-cache reuse,
    chunked prefill, int8/int4 paged KV, speculative decode, and multi-GPU
    layer-range sharding across both P40s — P11c's own "Deliberately
    deferred" list, in priority order for whoever picks this up next.
- **P8b — vision splice**: image/video token embedding into the decoder's
  input stream, reusing `crates/qwenvl`'s `VisionEncoder`/`PatchMerger`/
  position helpers as-is (verified numerically identical vision config to
  `VisionConfig::qwen3_omni()`, P2's finding — no fork needed), plus real
  3D M-RoPE position derivation from image grids (today's `model.rs` only
  handles the degenerate text-only all-axes-equal case).
- **P7 — device-side sparse MoE decode dispatch**: deliberately deferred to
  the performance-optimization phase, per `docs/porting-playbook.md` §10
  "correct-then-freeze" — `model::moe::expert_fwd`/`expert_fwd_compact` are
  correct for decode today, just not optimally fast at 256 experts
  (256×5=1280 dispatches/layer). Revisit once P11 (serving) needs real
  decode throughput.
- **⚠ Storage constraint discovered while validating against the real
  checkpoint**: brain's native checkpoint format stores every tensor as
  plain F32 (`StWriter::create`'s `Dtype::F32` plan) — a full import of this
  35B-parameter model would be **~140 GB** (`35e9 × 4 bytes`), which fits in
  neither this box's 93 GB tmpfs (used for the GGUF downloads) nor
  comfortably in `/data/workspace`'s remaining free space alongside
  everything else. **A full-precision `.brain.safetensors` import of this
  model is not a viable step on this hardware, and should not be attempted
  as an intermediate stage.** P10/P11's resident-loading path MUST construct
  INT8/INT4 device buffers directly from the GGUF import (streaming,
  `checkpoint::gguf::MmapGguf`'s per-tensor `TensorSource`, one tensor
  dequantized-then-immediately-quantized-and-uploaded at a time — never
  materializing a whole-model F32 intermediate on disk or in host RAM) —
  matching the original plan's "prefer the direct mmap-streaming GGUF path"
  intent, now with a concrete number behind why it is not optional.
- **P10b — INT4 quantized inference** for qwen35 specifically
  (`crates/qwen35moe/src/q4.rs`, mirroring P10a's `q8.rs` shape but over
  `model::int4`), once useful — P10a (int8, single-GPU) already landed.
- **P12 — LoRA + memory-bounded finetune validation**: base frozen in
  INT4/INT8 (never fp32/bf16), only LoRA adapters trainable, on real
  weights, measured loss descent, bounded peak RSS/VRAM.
- **P13 — full serving contract**: `Qwen35Resident`, `caps.rs` (model id
  must be byte-exact `Qwen/Qwen3.5-35B-A3B`), D-Bus verification, OpenAI/
  Anthropic auto-exposure via `catalog::api_caps`, examples + manifest.
- **Qwen3.5-27B (dense)**: a second named config once the shared hybrid code
  is proven — the dense sibling is the same code with `n_experts=1`-shaped
  MoE-as-dense and no linear-attention layers, additive not a fork.
