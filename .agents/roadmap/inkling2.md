<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# inkling2

Scope: Inkling 2 Mini (`Inkling2LoopForConditionalGeneration`): a looped MoE decoder with depth-attention residuals, plus a vision tower, served and trained from the published checkpoint, with every new building block landing as a shared op.

Status: **planned, nothing implemented.** This ledger is the plan from a read-only audit
(2026-11-21) of the published checkpoint, the reference implementation and brain's current
crates. Every item below names its owning crate and the gate that closes it.

## Pinned sources

- Checkpoint `thinkingmachines/Inkling-2-Mini` at revision
  `ed40e7da625f594fc9c4990f5f66be24873c22d7` (Apache-2.0): `config.json`,
  `hf_quant_config.json`, one 19.5 GB `model-00001-of-00001.safetensors`, optional
  `mtp.safetensors` (636 MB, outside the index), `tokenizer.json`,
  `chat_template.jinja`, `generation_config.json`, `processor_config.json`.
- Reference math: Hugging Face transformers at commit
  `2f1bc5e321b9a736200a590d6e7743ccdeb60772`, `models/inkling2/modeling_inkling2.py`
  (attention, MoE, convolutions, depth residuals, vision),
  `models/inkling2_loop/modular_inkling2_loop.py` (loop and message gates),
  `core_model_loading.py` (`Inkling2LoopDequantizeExperts`) and
  `conversion_mapping.py` (checkpoint names to modules).

## Facts pinned from the checkpoint and the reference

These were checked against the tensor headers and the reference code. Several of them
contradict a plausible reading of the release notes. Where they do, the checked fact is
the one that holds.

**Topology**
- 24 stored decoder layers, run as 48 logical visits: a prelude (layers 0-5), a core
  (6-17) run 3 times with shared weights, and a coda (18-23). `is_varloop: false`, so
  the pass count is fixed.
- Passes 2 and 3 restart from the prelude output and prelude block states. The previous
  pass enters only through messages.

**Depth-attention residuals (block variant)**
- The carried hidden state is the partial sum of the current 6-layer block.
- At a block-boundary layer (index % 6 == 0) the partial sum is appended to `block_states`
  and the residual restarts: the attention output is not added to it.
- Each sublayer input is a softmax over depth of the RMS-normed `[block_states...; partial]`,
  using a learned f32 query and scale `hidden^-0.5`.
- Parameters:
  - `attn_res.attn_queries.{i}` belongs to layer `i+1`; layer 0 has none.
  - Full-attention layers (5, 11, 17, 23) have 3 queries, giving separate q/k/v inputs.
    They also have separate `q/k/v_input_norm` in place of `attn_norm`.
  - `mlp_queries.{i}` is one per layer; `output_query` feeds the final norm.

**Messages between passes**
- For each core layer and pass p > 1, a learned query bank attends over the previous
  pass's core block states plus its final partial.
- This is per token, across depth only, so it needs no cache beyond the per-visit KV and
  conv state.
- Query layout: `middle_cycle_cross_pass_queries` [24, 2304] holds the (q, mlp) pairs and
  `..._kv_queries` holds the (k, v) pairs, per core layer.
- Message gate: `x' = (1 - tanh(Wf h)) * x + tanh(Wg h) * m`, with
  `h = Wd [rms(x); rms(m)]`, rank 64.
- Arithmetic is f32. The `controllers.attn` gate is the q gate.

**Attention**
- 24 q heads and 8 KV heads of width 128, with no RoPE.
- q: learned per-(head,dim) scale, then L2 clip to 22.63.
- k: L2 clip only, no scale.
- v: RMSNorm with no gamma.
- Relative-position bias:
  - Computed as `wr_du` (hidden to 24x16) then a scaled L2 clip (8.0), dotted with a
    [16, 1024] distance bank. The bias is zero at distance >= 1024.
  - The bias is present in the sliding layers too.
  - Full layers additionally scale q and the relative states by
    `tau(t) = 1 + 0.1 ln(max((t+1)/8192, 1))`.
- Sinks are hidden-state conditioned (`attn_sink_proj` [24, 2304]): one logit per
  token per head joins the softmax with a zero value. They are not a constant per head.
- Output: `sigmoid(gate_attn x) * attn` (in f32), then `wo_ud`.
- 20 sliding layers with window 1024, and full attention on layers 5, 11, 17 and 23.
  That gives 40 local and 8 global visits.

**Short convolutions** (k=4, depthwise, f32)
- K and V in sliding layers use a static conv, `(conv(x)+x)/sqrt(5)`.
- K and V in full layers use static taps plus per-token per-KV-head taps from
  `weight_proj` [32, 2304], `(dyn(x)+x)/3`. Tap 0 is the current token.
- `attn_sconv` and `mlp_sconv` follow every attention and MoE output,
  `(conv(x)+x)/sqrt(5)`.
- Each visit holds 4 conv states. Packed rows reset at document boundaries (`seq_idx`).

**MoE**
- 128 experts, top-8, no shared expert.
- Router: sigmoid scores (f32); top-k on `score + bias`; the weights are the unbiased
  scores renormalised (`+1e-20`) and multiplied by 8.
- Expert activation: `softcap4(g) * sigmoid(g) * softcap25(u)`.
- Outputs are accumulated in f32.
- `w13` is row-interleaved gate/up (`g0,u0,g1,u1,...`), and so is its scale tensor.

**Embedding and head**
- The embedding is followed by an RMSNorm (`embed_norm`).
- The unembedding is untied, with 201,024 rows of which the first 200,058 are valid.
- Final logit softcap is 30.

**Quantization**
- Layers 0-19 experts are NVFP4: packed E2M1 (low nibble first), an E4M3 scale per 16
  inputs, and a per-expert second level `amax[e] / 2688`.
  The denominator read from the file is 2688 = 6 x 448: the standard NVFP4 global
  scale, one per expert.
- Layers 20-23 experts are MXFP8: E4M3 elements with an E8M0 scale per 32 inputs.
- Everything else is BF16 or F32, as declared per tensor in `hf_quant_config.json`
  (`quant_method: "inkling2"`, `model.parameter_groups[]`).
- `config.json`'s `modelopt/NVFP4` is not accurate for most tensors.
- The reference dequantizes experts to bf16 and does **not** quantize activations.
  The sidecar's `required_activation_quantization` describes the vendor's runtime:
  NVFP4 per-token, rule `fouroversix_mae`, and MXFP8.

**Vision**
- ViT width 1024, 23 layers, 16 heads, LayerNorm with bias, layer-scale.
- Positions: a learned 32x32 position table interpolated bilinearly per image, plus axial
  2-D RoPE inside attention.
- Each image starts with a class token, and attention is block-diagonal per image.
- Patches are 20 px, merged 2x2 (4096), then `proj_in` / GELU / `proj_out` produce 2304.
- Preprocessing:
  - No resize: pad to multiples of 40 with -1 on the 0-255 scale.
  - Alpha is composited onto an 8-px 204/153 checkerboard.
  - Mean and std are 0.5.
  - One `<|unused_200054|>` placeholder per 40x40 region.

**Tokens and chat**
- EOS for sampling is 200006 and pad is 199999. A message ends with `<|end_message|>`
  (200010).
- The template takes a numeric `reasoning_effort` (default 0.9). Tools are declared as
  sorted-key JSON in a system `tool_declare` message.

**MTP** (optional)
- A 3-layer block drafter (block 8) with RoPE theta 1e6 and context K/V projections over
  target taps at logical depths 5, 23, 41 and 48.
- It adds a rank-256 Markov head. The tensor names (`dflare_*`, `k_proj_ctx`) put it in
  the DFlash family.

**Sizes**
- About 24.5B expert parameters: 11.5 GB as NVFP4 and 4.2 GB as MXFP8.
- About 1.9B dense parameters: 3.8 GB as bf16.
- KV per visit is 4 KiB per token (bf16): 32 KiB per token over the 8 global visits,
  plus 40 x 1024 local rows.
- Conv state is about 80 KB per visit, or 3.8 MB per sequence.

## What brain already has (reuse, do not copy)

- Looped execution in two bespoke crates: ouro (whole-stack loop) and huginn
  (prelude/core/coda). Both have:
  - per-visit KV planes in paged engines;
  - recompute-from-boundary training;
  - `model::streamed::RevisitingPass` (gradients carried across visits, one update per
    step).
- `model::moe`:
  - The `RouterKind::SigmoidNoAuxTc` router (`router_gate_sigmoid.wgsl`, plus backward)
    takes a selection-only bias, an unbiased weight and a route scale.
  - Row-compacted expert training (`expert_fwd_compact`, `moe_layer_bwd`).
  - An NVFP4 `MicroBank` with per-expert f32 globals.
  - NVFP4 gather and grouped MoE kernels (`moe_nvfp4_*`, WGSL and CUDA).
- `model::ops::Weight::Micro{Nvfp4|Mxfp8E4m3|...}` gives a frozen-tier forward, dX and dW.
  `tier_of` already maps E4M3 + Block(32) E8M0 to MXFP8.
- `checkpoint::packed` has E2M1, E4M3 and E8M0 formats, block scales, and codecs whose
  nibble order matches the reference.
- `dyn_group_conv1d.wgsl` is exactly the full-layer dynamic K/V conv, forward only.
- `conv1d*` (with backward) and `causal_conv1d_step` cover static convs. lfm2 already
  trains a short conv.
- `gqa_mixer` provides the sigmoid output gate, the sliding window (materialised scores
  only) and ring KV rows (`ring_row`, `RingChunk`).
- `attn_scores_causal_bias` provides a dense additive-bias attention with backward, used
  as the oracle for the new bias op.
- `tau_scale` (+`_ds`) multiplies per (head, row), and `l2norm_scale` (+backward) is the
  per-dim scaled L2 norm.
- Serving:
  - `PagedDecoder` + `Scheduler`;
  - `SnapshotPrefixCache` for hybrid state;
  - `PromptEmbeds` splice;
  - `data::chat_template` (minijinja over the checkpoint's own template);
  - generic speculative verify and commit;
  - `enable_taps` / `read_taps`;
  - `dflash2` (block drafter with dynamic conv).
- Training:
  - `ChatTrainer`, `TokenObjectives`, `rl::fit` (DPO/GRPO/... as generic visitors);
  - `head_ce` chunked LM-head CE;
  - `adapter` LoRA sites and serving bank;
  - `memplan`, ZeRO-1/2, data parallel;
  - conformance C1-C9 and the training-capability matrix gate.
- Vision: `model::vit` (LayerNorm with bias, layer-scale, backward), `projector`,
  `vlm::splice`, and the qwen3vl 2x2 merger.

## Gaps found in brain (each a defect today or a missing shared op)

- `checkpoint::packed::hfquant` knows only the ModelOpt `quantization{...}` sidecar.
  Pointed at this repo it misreads silently:
  - The `parameter_groups` file is ignored.
  - `config.json` makes every tensor "NVFP4".
  - NVFP4 experts (U8, no `_scale_2`) load as plain u8.
  - MXFP8 experts (U8 E8M0 scale) load as unscaled FP8.
  - The `_amax` and `_outer_scale_denominator` sidecars surface as parameters.

  A wrong-but-loading checkpoint is the worst failure mode, so I0.1 comes first.
- No rank-3 stacked-expert view for scaled or sub-byte tensors (`SlicedReader` refuses
  them), and no row-stride view for interleaved gate/up.
- No MXFP8 GEMV and no MXFP8 MoE kernel on any backend (`tier-gemm.allow` records the
  GEMV gap). `MicroBank` is NVFP4 only.
- No windowed flash or paged attention: windows exist only as materialised `[B,H,T,T]`
  scores.
- None of these exists: hidden-conditioned sinks, distance-bank relative bias, softcap
  (anywhere), gamma-free RMSNorm, scaled L2 clip with threshold, depth attention, or
  message gate.
- The MoE expert activation is hard-wired SwiGLU.
- `dyn_group_conv1d` has no backward.
- `head_ce` and the serving head have no logit softcap and no valid-vocab mask.
- Training has no packed rows with document-boundary state resets, and no routing replay.
- There is no LoRA site for stacked experts, and no training backward through `Micro`
  expert banks.
- Looped serving is bespoke per crate. Ouro and huginn disable prefix reuse because a
  key that knows about visits does not exist.
- Chat: there is no template flavor for Inkling's tool and thinking wire format, and
  `end_of_turn()` exists only for the Qwen format.

## Design rule for this port

The model crate (`crates/inkling2`) holds only the following:
- configuration;
- the checkpoint name map;
- the declared topology (the visit plan);
- the layer recipe that wires shared ops in the reference order;
- residency, catalog and capability glue.

Every operation above that a future model could use lands in its owning shared crate.
Each such op is:
- forward, backward and gradcheck;
- WGSL reference, with CUDA through wgsl-cuda;
- autotuned candidates registered through `gpu_core::provider`;
- named for what it computes, never for Inkling.

`inkling2` is one crate for the family: the non-looped Inkling 2 variants are the same
recipe with an empty loop, as an arch row and config, not a new crate.

## Work items

The phases are ordered by dependency. Items inside a phase can run in parallel.

### Phase 0: load exactly what was published (checkpoint, model)

- **I0.1** `checkpoint::packed::hfquant`: a second sidecar schema, selected by shape
  (`model.parameter_groups[]` with explicit tensor lists, also `mtp.parameter_groups[]`),
  never by the string "inkling2".
  - Per-tensor declarations override `config.json`'s global algorithm.
  - Map `weight_transform` and `weight_scale_format` (`cast`, `nvfp4_nearest` +
    `amax_denominator`, `mxfp8_nearest`) to declarations. Refuse unknown values by name.
  - Cross-check every declaration against the stored dtype and the shape of the scale
    sibling. A mismatch is an error, never a reinterpretation.
  - Gate: unit tests on the real sidecar plus the real header, red today.
  - Done (2026-11-21): `HfQuant` parses the `parameter_groups` shape strictly (`TensorDeclaration`,
    `DeclaredWeight::{Cast, Nvfp4 { global }, Mxfp8, Undeclared}`, `Nvfp4Global`, `ActivationQuant`); `PackedReader`
    reads each tensor as declared and refuses undeclared tensors and dtype/scale-shape contradictions
    (`tests/it/parameter_groups_release.rs`, finding #549).
- **I0.2** `checkpoint::packed::{encoding,codec,reader}`:
  - A `QuantEncoding::mxfp8()` constructor.
  - An NVFP4 second level derived from `amax / denominator` sidecars, folded to the
    per-slab f32 global at load (one division, carried).
  - A per-leading-axis slab granularity for rank-3 stacked tensors.
  - `_weight`/`_weight_scale`/`_weight_amax`/`_weight_outer_scale_denominator` sidecar
    naming, hidden from the parameter list.
  - Gate: decode of a real expert slab equals the reference dequant formula, bit-exact in
    f32, for one NVFP4 layer and one MXFP8 layer.
  - Done (2026-11-21): `QuantEncoding::{mxfp8, nvfp4_per_slab}`, `ScaleGranularity::Slab`, the `_scale_2` level
    folded once from `_amax / _outer_scale_denominator`, sidecars hidden; scale levels apply finest first. Gate:
    `a_decoded_expert_slab_equals_the_reference_dequantization_bit_for_bit` (golden from
    `tools/goldens/inkling2_expert_dequant_reference.py` in `testdata/golden/inkling2/experts/`).
- **I0.3** `checkpoint::packed::slice` (+concat):
  - Per-expert 2-D views of stacked scaled and sub-byte tensors.
  - A stride-2 row view that splits interleaved gate/up for both the payload and the
    scales.
  - Kernels may instead consume the interleaved layout directly as a declared `GluLayout`;
    decide in I1.3.
  - Done (2026-11-21): `SlicedReader::with_view` takes scaled and sub-byte sources (whole rows; a per-slab level
    becomes the view's per-tensor level) and `with_slab_views` cuts a stacked tensor into per-expert matrices in the
    single-expert encoding (NVFP4 with its own f32 global, MXFP8). Rows stay as stored (interleaved gate/up); no
    stride-2 view, per the decision below: the model declares the layout.
- **I0.4** `modelstore::recipe`: make sure `brain pull` lands every file the model needs,
  including `mtp.safetensors` (it is not in the index); check what the pull fetches today. `checkpoint::weightio` opens it as a second, optional source
  with its own unprefixed names.
  - Done (2026-11-21): the transformers recipe missed `hf_quant_config.json` (a pull would have
    read every tensor through `config.json`'s wrong global NVFP4), `processor_config.json` and
    `mtp.safetensors`; it now fetches them when listed (`MTP_WEIGHTS`, plan test
    `a_release_lands_its_quantization_declaration_processor_and_drafter`). `mtp.safetensors` opens as its own
    `WeightReader` with unprefixed names and reads under the `mtp` section of the same declaration.
  - Left: `brain pull` still refuses the architecture until I2.2's arch row; then check that it accepts the
    existing store entry as is and writes its checksums.
- **I0.5** `tools/goldens/inkling2_dump_reference.py`: a tiny random-config golden
  (2 prelude, 2 core x 3 passes, 2 coda, block size 2, 8 experts top-2, d 64), recording:
  - the hidden state at every logical visit;
  - every block state and message;
  - router indices and weights;
  - conv states;
  - logits.

  Also a real-weight dump of the same tensors for 3 prompts, run once on the GH200 and
  kept under `testdata/`.

  Done (2026-11-21): `tools/goldens/inkling2_dump_reference.py tiny|real` writes
  `testdata/golden/inkling2/{tiny,real}` (`make testdata/fetch` links them from
  `BRAIN_GOLDEN_MIRROR`). Tiny: 6 layers, block 2, core 2-3 x 3 passes (10 visits), sliding
  and full layers in every segment, window 4, rel extent 6, log floor 4, caps chosen so
  clips and softcaps fire (counts in the manifest); the checkpoint is f32 in the published
  naming, proven by a round trip through the reference's own conversion mapping. Per visit:
  `in/out`, `block_states`, `attn_in`, `mlp_in`, `messages`, `gated_{q,k,v,mlp}`, `attn`,
  `attn_conv`, `mlp`, `mlp_conv`, `rel_states`, `position_bias`, router logits/indices/weights,
  plus `relative_distance.{full,sliding}`, `final_in`, `final_norm`, `logits`, the conv
  states after a cached prefill and four token-by-token decode steps (decode equals the
  uncached forward within 2.5e-5). Real: bf16 reference on CPU, three chat prompts (22-27
  tokens), per-visit `in/out` and router choices for all 48 visits, top-64 logits per
  position and full rows at -1, -2, -4 (77 MB). Reference facts found on the way:
  - the reference ignores `local_layer_ids` and derives layer kinds from `(i + 1) % 6`
    (they agree for the published config); the tiny config writes both;
  - the chat template has no generation prompt: after `<|end_message|>` the model itself
    emits `<|message_model|>` (all three real prompts' argmax);
  - the reference's cache slots are physical layers for the prelude, first core pass and
    coda, then one appended copy of the core per extra pass (manifest `cache_slot`).

  Left: a bf16-vs-f32 reference spread for the real thresholds (an f32 load needs about
  110 GB of host RAM).

### Phase 1: shared ops (forward + backward + gradcheck, WGSL first, CUDA through the seam)

- **I1.1** `kernels` + `model::hostmath`: `softcap` / `softcap_bwd`, `y = c tanh(x/c)`.
  Used for the logit cap (30) and the expert caps (4, 25).
  Done (2026-11-21): kernels `softcap` / `softcap_bwd` (params `total, cap` as f32 bits; the backward takes the
  forward input), `model::hostmath::{softcap, softcap_grad}`, spec `crates/model/tests/it/softcap.rs`
  (host reference + f64 finite differences, saturation without NaN). CPU backend verified; CUDA via the generated seam, not exercised.
- **I1.2** `kernels`: `rmsnorm_nogain` (+dx) and `l2clip_scale` (+dx, d-scale), with an
  optional scale and a threshold. This generalises `l2norm_scale`.
  Done (2026-11-21): kernels `rmsnorm_nogain` / `rmsnorm_nogain_dx` and `l2clip_scale` / `l2clip_scale_dx` /
  `l2clip_scale_ds` (workgroup per vector, dispatched with `Dispatch::Workgroups`; the dx kernel also writes the
  per-element scale partials that `_ds` reduces over vectors, so the norm is computed once), hostmath references
  `rmsnorm_nogain_rows` / `l2clip_scale_rows`, spec `crates/model/tests/it/norm_clip_ops.rs`. `l2norm_scale` is
  kept: it normalises (eps under the root), the clip only removes the excess (eps added to the norm), so they are
  different operators, not one with options. CPU backend verified; CUDA not exercised.
- **I1.3** `model::moe` + `moe_linear_gated*`, `moe_nvfp4_*`:
  - A declared `GatedAct` (SwiGLU, clamped SwiGLU, softcap-sigmoid with two caps) with
    backward, plus a declared gate/up layout (concat or interleaved).
  - Check `router_gate_sigmoid` at `n_group = topk_group = 1` and the `+1e-20` renorm
    against the reference.
  - Gate: `moe_block_gradcheck` per activation and layout.
  Done (2026-11-21): kernels `glu_act` / `glu_act_bwd` and module `model::glu` (`GatedAct::{SwiGlu,
  SoftcapSigmoid{gate_cap,up_cap}}`, `GluLayout::{Separate,Concat,Interleaved}`, `GluSpec`, `glu_act_step`,
  `glu_act_bwd_step`, `expert_fwd_glu`, `expert_bwd_glu`). The projection kernels (`moe_linear_gated*`, `moe_nvfp4_*`,
  `moe_i8_*`) are layout-agnostic GEMVs over `[E, n, k]` banks, so a fused `[E, 2 ff, d]` interleaved bank is read by
  them unchanged with `n = 2 ff`; the layout is declared where gate and up meet, the activation kernel, which reads
  the interleaved projection output in place. `GluSpec::default()` (SwiGLU, separate) is `silu_mul` /
  `silu_bwd_da` / `silu_bwd_db` to the bit and `expert_fwd_glu` / `expert_bwd_glu` reproduce `expert_fwd` /
  `moe_layer_bwd`'s experts bit for bit (tested). Router: `router_gate_sigmoid` / `router_bwd_sigmoid` at
  `n_group = topk_group = 1` matched the reference except the denominator, a floor `max(sum, 1e-20)` where the reference
  adds `sum + 1e-20`; both kernels now add (differs only when the selected scores are below ~1e-12), spec
  `router_sigmoid_reference.rs` at E=128, top-8, scale 8 including that case. Gates: `moe_block_gradcheck` per
  activation and layout (5 cases + bit identity + interleaved == permuted concat), `glu_act.rs`. Left: the row-compacted
  (`MoeActs::compact`) and grouped (`expert_fwd_grouped`) paths and the CUDA-only fused NVFP4 gate/up launch
  (`Fused::MoeNvfp4GateUp`, SwiGLU only) still take the default activation; a model on another activation uses the
  unfused slot path (`glu_act_step` after the projection GEMVs). CPU backend verified; CUDA not exercised.
- **I1.4** `kernels` + `model::moe` + `model::ops::sink`:
  - An MXFP8 `MicroBank` (no global scale).
  - MXFP8 GEMV and MXFP8 MoE gather and grouped kernels.
  - Rows in `tier-gemm.map`. `tier-gemm.allow` only shrinks.
  - The NVFP4 MoE kernels must accept per-expert globals derived from amax.
  - Done (2026-11-21): `matmul_mxfp8_gemv` (MAXM 1/2/4/8, `Ops::small_m_gemv` MXFP8 arm, tier-gemm row, the
    `Mxfp8E4m3 gemv` gap removed), `moe_mxfp8_gemv_gather` and `moe_mxfp8_grouped`: one top-level barrier, no
    workgroup table, so they run on the CPU JIT too (`@cpu yes`); CUDA through the WGSL translation. One order of
    arithmetic: gather, grouped and dense GEMV agree to the bit (`gpu-core/tests/it/moe_mxfp8.rs`, run on CPU).
    `MicroBank` holds NVFP4 or MXFP8 E4M3 (`upload_micro_bank` from per-expert views; `MicroBank::gather` /
    `grouped` dispatch either kind; `micro_bank_kernels()`); a stacked NVFP4 release's `amax / denominator` lands
    in the bank's `global` (`model/tests/it/micro_bank_stacked.rs`). The kernels are layout-agnostic (a fused
    gate/up bank is read with `n = 2 * ff`; `model::glu` applies the declared layout and activation).
  - Left: the NVFP4 bank products and the MXFP8 `Ops` route were not run here (no GPU); `qops_tier_coverage` and
    `micro_bank_stacked` cover them on a card. MXFP8 has no tensor-core batched path (tier-gemm `fused` gap stays);
    no native CUDA build; E5M2 MXFP8 has no GEMV.
- **I1.5** Attention kernels (`flash_attn_causal_gqa*`, `paged_flash_*`, `gqa_mixer`),
  each an optional declared input, off by default:
  - (a) A causal sliding window in the flash prefill, decode and paged kernels, forward
    and backward. This also removes the `[B,H,T,T]` cost the windowed mixer pays today.
  - (b) Per-(row, head) sink logits joining the softmax denominator with a zero value,
    with a d-sink gradient.
  - (c) A distance-bank additive bias `bias[h,i,j] = <rel[i,h,:], bank[:, i-j]>` for
    `0 <= i-j < extent`, else 0, computed in-tile (never materialised), with d-rel and
    d-bank gradients.
  - (d) A per-row query scale (the `tau(t)` factor computed on the device from position).

  Gate: equality against the dense `attn_scores_causal_bias` oracle and finite
  differences; CUDA included.
  - Done (2026-11-21): `model::flash_ext` over the `flash_attn_gqa_ext*` kernels (forward, `_bwd_prep`
    (D, lse, d-sink), `_bwd_dq` (dQ, d-rel), `_bwd_dkv`, `_bwd_dbank`), all four options declared in
    `AttnOptions` and off by default, inputs refused unless declared. Distances and `tau` come from absolute
    positions; the forward reads a contiguous layout or paged / ring KV (f32 or bf16 planes) through a
    per-sequence index, so chunked prefill, decode, a mixed batch and a ring of `window` rows compute the
    same logits. `gqa_mixer_fwd_windowed` runs it (no `[B,H,T,T]` slab, `AttnSaved::Windowed`), and
    `gqa_mixer::ring_chunk_attend` is the ring prefill with any options. Gated against a host f64 oracle
    for every option combination, every gradient by element-wise central differences, the window and
    `n_floor` boundaries, and whole == chunked == decode == ring (`crates/model/tests/it/flash_ext.rs`);
    verified on the CPU backend (the qwen35 EdgeRock suite included).
  - Left: the kernels are the barrier-free reference tier (one invocation per row and head); a tiled
    workgroup form for GPUs (and its native CUDA twin) is not written: no barrier-capable device was free
    to gate one, so it is Phase 7 work, gated against these kernels. CUDA was not exercised (the tests run
    there unchanged through `testgpu::dev`). The causal flash trio, the tensor-core and native paths and
    the paged flash decode kernels take none of the options; callers with an option call `flash_ext`. The
    ext kernels have no `gpu_core::cost` formula yet (they run unpriced). Training reads the contiguous
    layout only (no packed-row document resets: I4.4).
- **I1.6** `kernels`: the `dyn_group_conv1d` backward trio (dx, d-base, d-dyn), plus a
  single-token decode form with ring state. Make the static conv's residual-and-scale
  `(conv(x)+x)*s` one fused parameter of the step and prefill kernels. Document-boundary
  masking via a segment-id input.
  Done (2026-11-21): one kernel family for static and dynamic short convolutions over token-major rows:
  `dyn_group_conv1d` (extended in place: params `flags, res_scale`, bindings `seg`, `hist`; `flags == 0` is the previous
  kernel exactly, its one caller, dflash2, passes 0), `dyn_group_conv1d_dx` / `_dbase` / `_ddyn`, `dyn_group_conv1d_step`
  (ring state per sequence) and `conv_state_tail` (the state a prefill chunk leaves), with the typed wrapper
  `model::short_conv` (`ShortConvShape`, `short_conv_fwd/_bwd/_step`, `conv_state_tail_step`). Options: fused
  `(conv(x) + x) * s`, document `seg` ids, carried history, purely static taps, and a base read in the published
  depthwise-weight layout `[c, taps]` (last tap current). The static prefill/step kernels `conv1d` and
  `causal_conv1d_step` are not extended: their params are an ABI with native twins and a fixed call contract
  (`gdn_causal_conv1d_step`, the cost model), so the static case is the same family with `FLAG_NO_DYN` (tested equal
  to `causal_conv1d_step` to rounding, state bit-equal). Spec `crates/model/tests/it/short_conv.rs`: f64 oracle over the
  option matrix, finite-difference backward, step == block, chunked prefill == whole, no cross-document or
  cross-block reads. A carried history has no gradient (inference input). CPU backend verified; CUDA not exercised.
- **I1.7** `model` (new module `depth_attn`): softmax-over-depth mixing of a retained
  stack `[D, rows, dim]` (RMS-normed) with `Q` learned queries.
  - Backward fans gradients into every retained state.
  - `model::saved` gains a retained-state store whose entries accumulate gradients from
    several consumers.
  - This is the block-AttnRes primitive. Plain residuals are its degenerate case.

  Done (2026-11-21): `model::depth_attn::{DepthAttn, DepthAttnIds, kernels}` over
  `model::saved::{RetainedStates, retained_kernels}`; kernels `depth_attn_fwd`,
  `depth_attn_bwd`, `depth_attn_bwd_query` (one thread per row / query element, CPU JIT
  safe) and `axpy_at`; host reference `hostmath::depth_attention`. Spec in
  `crates/model/tests/it/depth_attn.rs`: forward = host reference, backward = per-element
  f64 central differences for every retained state, the partial and every query, two
  consumers of one slot sum their gradients, and every tiny-golden attention input plus the
  final readout reproduce the reference within 1e-5 (CPU backend).
  Left: a workgroup-per-row candidate behind `gpu_core::provider` once the model runs and a
  profile shows these on the gap list; CUDA not exercised.
- **I1.8** `model` (compose first, fuse only on profile evidence): the message gate
  `(1 - tanh(a)) x + tanh(b) m`, with forward and backward over existing
  `rmsnorm`/`concat2`/`matmul`/`tanh` kernels.

  Done (2026-11-21): `model::message_gate::{MessageGate, MessageGateIds, MessageGateWeights,
  MessageGateActs, MessageGateScratch, kernels}`, composed from `rmsnorm_nogain(_dx)`,
  `concat2`/`concat_split`, the `Ops` products, `tanh_act(_bwd)`, `mul`, `sub`, `add2`.
  Spec in `crates/model/tests/it/message_gate.rs`: forward and every input and weight
  gradient equal f64 central differences, weight gradients accumulate over calls, and all
  16 q/k/v/mlp gate outputs of the tiny golden's looped visits reproduce within 1e-5 from
  the checkpoint's `controllers.*` weights (CPU backend). Left: a fused kernel only if a
  profile of the model puts the gate on the gap list.
- **I1.9** `model::head_ce` + the serving head: a logit softcap and a valid-vocab bound
  in the chunked CE (forward and backward) and in every serving readout path (`logits`,
  top-k, keyed draws, full rows, `score_continuation`). `vocab()` reports the valid size.
  Gate: the serve conformance full-row and keyed-draw tests.
  Done (2026-11-21): kernels `logit_cap_mask` / `logit_cap_mask_bwd` (in place on a `[rows, vocab]` block: soft cap on
  the valid columns, a `-1e30` floor on the padding columns; the backward reads the transformed logits, since the cap's
  derivative is `1 - (y/c)^2`) and `model::head_transform::HeadTransform { softcap, valid_vocab }` with the device
  steps (`step`, `bwd_step`), the host forms (`served`, `trim`, `logprob_at`, `served_vocab`) and validation.
  `HeadCe::set_head_transform` applies it in the chunked CE forward and backward (cap after the logit scale; the
  gradient passes through the cap and a padded column carries probability exactly 0 and no gradient; labels in the
  padding are refused). `PagedDecoder::head_transform()` (default identity) declares it: such a decoder serves
  transformed values from `logits`, the admission picks, candidates, keyed draws, full rows and the rows it scores
  over, with `vocab()` the valid size and rows of exactly that length; new conformance property
  `a_declared_head_transform_is_honoured` (in `run_all`). No engine declares a transform yet: each engine opts in by
  applying `HeadTransform::step` to its device logits block before `readout` reads it. Specs: `head_ce.rs`
  (softcap + mask vs an f64 reference and finite differences, padding rows dominating the raw logits),
  `head_transform.rs` (padded columns never picked, drawn or returned on the device or host; a scheduler run with a
  declaring decoder; the conformance property accepts an honest decoder and refuses one that serves raw rows).
  CPU backend verified; CUDA not exercised (the native top-k / draw launches read the transformed block like any
  other, but were not run).

### Phase 2: the model (inkling2)

- **I2.1** `model` (new `looped` module): a declared visit plan (prelude range, core range
  x passes, coda range). It maps each logical visit to its physical layer, attention kind
  (full or window n) and state slots.
  - It is used by inkling2's forward, backward, serving and streaming.
  - It is shaped so ouro and huginn can migrate to it (tracked in their ledgers; not a
    blocker).

  Done (2026-11-21): `model::looped::{VisitPlan, LoopShape, Visit, Segment, Attention}`.
  `VisitPlan::new(shape, |layer| Attention::Full | Attention::Window(n))` lists every
  logical visit in forward order with its physical `layer`, `segment`, `pass`, `attention`
  and state `slot` (one per visit, or `share_core_slots(r)` for a ring); `pass(p)`,
  `prelude()`, `coda()`, `visits_of(layer)`, `boundaries()` (visit i reads boundary i,
  writes i+1), `revisits()` (drives `RevisitingPass`), `count(attention)` (pool charging),
  `slot_owners()`. `crates/model/tests/it/visit_plan.rs` holds it to the tiny golden's
  recorded visit order. Ouro and huginn migrated onto it in the same series (their
  forward, backward, decode and paged engines; outputs, gradients and served tokens
  bit-identical before and after on cpu). The reference's own cache numbering differs
  (manifest `cache_slot`); brain's slot is the visit index.
- **I2.2** `crates/inkling2`:
  - Config (strict serde, `deny_unknown_fields`, every field validated).
  - The `arch` row for `Inkling2LoopForConditionalGeneration` / `inkling2_loop`, plus the
    non-loop family row.
  - The name map from the published names, with two-way coverage.
  - Strict per-tensor dtype and shape checks.
- **I2.3** `inkling2`: the forward recipe over I1 ops in reference order, including:
  - the block-boundary residual restart;
  - the q/k/v depth queries of the full layers;
  - message queries (q, mlp) and (k, v) re-ordered at load.

  Gates:
  - Tiny golden at every logical visit on cpu, wgpu, Vulkan and CUDA.
  - Real-weight parity on the GH200 for the hidden state per visit, router agreement and
    final logits, with thresholds set from the reference's own bf16-vs-f32 spread.
- **I2.4** Decide the parity target for the expert precision.
  - Default: dequantized weights with unquantized activations, which is the HF reference.
  - Opt-in tier: the vendor's activation quantization (NVFP4 per-token "four over six" and
    MXFP8), measured as KL and top-1 against the bf16 reference.

  Precision is a metric, so the low-precision fast path is opt-in.

### Phase 3: serving

- **I3.1** `model::serve` + `paged`: a generic visit-aware engine core composing:
  - the paged growing KV for full visits (8);
  - a ring or retired window KV for local visits (40);
  - per-visit conv-state rows charged to the pool (`sequence_state_blocks`).

  Admission uses `blocks_for` times visits by kind.
- **I3.2** `model::snapshot_prefix`: prefix entries spanning all visits (full-visit blocks
  shared, ring and conv rows snapshotted).
  - This is the visit-aware identity ouro and huginn lack.
  - Gate: the serve conformance prefix tests.
- **I3.3** `data` + chat: an Inkling template flavor and tool/thinking scanner. Pass
  numeric `reasoning_effort` as a template variable. A stop set of 200006 plus 200010,
  which needs a format-level stop API in place of the Qwen-only `end_of_turn()`.
  - Verify that `QwenBpe` (or the generic HF BPE) reproduces `tokenizer.json` on a corpus
    of special-token and multilingual goldens.
  - Done (2026-11-22):
    - `QwenBpe::from_file` reproduces the `tokenizers` ids and decode of the checkpoint on 96
      hostile strings (every special token inline), unchanged.
    - The checkpoint's own template renders 72 reference conversations byte for byte, refusals
      included. `tojson` now takes the `transformers` keywords (#550).
    - `ReasoningEffort` (named or a validated level) is applied through the render `extra` map.
    - `TemplateFlavor::TypedMessages`, read by `data::message_scan::MessageScanner` behind
      `ChatScanner`, is selected from the template's markers (`ChatTemplate::reply_flavor`).
    - `ChatFormat::end_of_turn()` returns the format's stop tokens (`ChatTemplate::end_of_turn_markers`);
      `stop_ids` takes any list, and a real-checkpoint test pins `[200006, 200010]`.
  - Left: stopping on 200010 ends a generation after the first message, so a reply that thinks
    and then answers needs the serving lane to continue past a message boundary (or to stop on
    200006 alone for one-shot replies).
- **I3.4** `catalog::resident_inkling2` and the registrations (arch, catalog, serving
  model_dir, modelstore family, cli alias):
  - the `generate` action, cancellable, with host-resolved weights;
  - an estimate of weights + KV + state + scratch;
  - `serve::conformance::run_all` on the tiny config, plus `inkling2_served.rs`;
  - `docs/reference/models/inkling2.md`;
  - a sample.
- **I3.5** Acceptance through the public API with the vendor's published verification
  suite (coding, vision, long context to 1M), after operator parity.

### Phase 4: training on the frozen quantized base

The first trainable target needs no streaming: the base is about 19.5 GB.
- **I4.1** `model::moe`: a backward through frozen `Micro` expert banks (NVFP4 and MXFP8
  grouped dX). The router gradient is optional, behind a declared flag.
- **I4.2** `model::adapter`: stacked-expert LoRA sites (per-expert A/B over a bank), plus
  sites for:
  - attention q/k/v/o/gate;
  - relative states;
  - sink projection;
  - message-gate down/forget/gain;
  - depth queries.

  Serving bank overlays for each.
- **I4.3** `inkling2`:
  - A `ChatTrainer` plus `TokenObjectives`, with `LORA_BASES` naming the native
    NVFP4/MXFP8 base.
  - `RevisitingPass` accumulation for the core.
  - A model-owned saved structure: block states, plus each pass's core block states for
    the next pass's messages (about 2 rows per token per pass).
  - Gates:
    - finite differences on the tiny config;
    - `elementwise_check` on every shared core parameter;
    - conformance C1-C9;
    - capability-matrix rows.
- **I4.4** `model::train` + `data`: packed rows with document-boundary resets of
  attention, every conv state and the relative-position origin. Gate: no supervision or
  state crosses a boundary.
- **I4.5** `rl` + `model::rollout`:
  - routing replay (rollout expert ids recorded per logical visit and forced in the
    trainer);
  - an inkling2 rollout engine over the serving engine;
  - the serve-vs-train log-prob gap measured before any reward is read.

### Phase 5: vision

- **I5.1** `inkling2` over `model::vit`:
  - interpolated learned positions, using the shared bilinear position-table op (add it
    if absent);
  - axial 2-D RoPE;
  - a class token per image;
  - block-diagonal spans;
  - the 2x2 merge projector.
- **I5.2** `imaging`: the preprocessor (pad, checkerboard alpha, normalisation, region
  grid), byte-exact against the reference processor.
- **I5.3** Image-text parity and the `PromptEmbeds` splice into the scheduler. Frozen-tower
  LoRA comes after that.

### Phase 6: full fine-tuning

- **I6.1** Trainable bf16 expert banks with f32 master and optimizer state on the host
  (GH200 C2C), or ZeRO across 4 cards.
- **I6.2** `model::streamed`: a non-f32 window plus `RevisitingPass` for a model whose
  frozen layers stay quantized.
- **I6.3** Expert-parallel training layout (`memplan`, `distributed`).

### Phase 7: throughput

Driven by the per-device gap list, as new candidates only:
- grouped native NVFP4 and MXFP8 expert GEMMs (Hopper decodes to FP8/BF16 tensor-core
  input; FP4 MMA only where the device declares it);
- the fused 4-message controller;
- windowed flash with sink and bias;
- graph replay;
- the MTP drafter through `dflash2` with a tap API keyed by logical depth and rollback
  over rings and conv rows.

## Decisions (2026-11-21)

- I2.4: the default serves the reference numerics (dequantized expert weights,
  unquantized activations). The vendor's activation-quantized numerics are an opt-in tier,
  measured as KL and top-1 against the bf16 reference.
- I1.3 / I0.3: the kernels consume the interleaved gate/up layout as a declared
  `GluLayout`, so the published bytes are served as downloaded; there is no
  de-interleaving at load.
- I2.1: ouro and huginn migrate onto the shared visit plan in the same series that
  introduces it.
- The checkpoint lives only in brain's model store as
  `thinkingmachines/Inkling-2-Mini`, and no copy of the weights lives elsewhere.
  - Today `brain pull` refuses it ("unsupported architecture", exit 1) until the arch row
    (I2.2) exists. The store entry was therefore filled from the pinned revision with the
    Hugging Face CLI.
  - Once I2.2 lands, `brain pull` must accept the entry as is and write its checksums
    (I0.4).
