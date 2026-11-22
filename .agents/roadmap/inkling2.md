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
  - Done (2026-11-22): `crates/inkling2` with `config` (`CheckpointConfig` strict at every level, `Inkling2Config`
    validated: family from `model_type`, layer kinds from `local_layer_ids` and refused when `layer_types` or, without
    it, the reference's `(i + 1) % 6` rule disagrees, the loop as whole blocks, valid vocab, caps) and `names` (the
    published-name manifest and `check`: every parameter present once with its shape and an encoding its role
    admits, nothing left over; vision tower included, `mtp.safetensors` is a separate source). One `arch` row
    `inkling2` covers both classes and both `model_type`s (the plain family is a config, not a crate). Gate:
    `tests/it/layout.rs` - the real header is consumed exactly once (1116 parameters, 26,388,641,280 elements, the
    decoder 26,070,385,152), the tiny reference checkpoint likewise, wrong configs refused by name. `brain pull
    thinkingmachines/Inkling-2-Mini` now plans the existing store entry as complete (no network, no file touched);
    no pull path writes per-file checksums for an entry already on disk (a download verifies the hub's sha256), so
    there is nothing more for I0.4 to write.
- **I2.3** `inkling2`: the forward recipe over I1 ops in reference order, including:
  - the block-boundary residual restart;
  - the q/k/v depth queries of the full layers;
  - message queries (q, mlp) and (k, v) re-ordered at load.

  Gates:
  - Tiny golden at every logical visit on cpu, wgpu, Vulkan and CUDA.
  - Real-weight parity on the GH200 for the hidden state per visit, router agreement and
    final logits, with thresholds set from the reference's own bf16-vs-f32 spread.
  - Done (2026-11-22): `inkling2::model` - `Inkling2::load(ops, dir)` (header held to the name map first),
    `forward(tokens)` (logits `[T, valid_vocab]`, capped), `forward_probed` with a `Probe`, `kernels()`,
    `visit_plan(cfg)` over `model::looped`. One retained-state slot per block, the first core block (the prelude's
    output) shared by every pass; `flash_ext` with window, sink and distance bias, `tau` scaling q and the bias on
    full layers; `short_conv` reading the published depthwise weights, the dynamic taps' `weight_proj` rows reordered
    at load from `[kv head][tap]` to the `[tap][kv head]` the kernel reads; `message_gate`; `router_gate_sigmoid`
    then a dense f32 bank (`expert_fwd_glu`) or the published NVFP4/MXFP8 `MicroBank`s (`gather` per slot,
    `glu_act` softcap-sigmoid interleaved, `moe_group_combine` in ascending expert order, the reference's
    accumulation order); `HeadTransform` for the cap and the valid rows. To run the NVFP4 banks on the CPU JIT,
    `moe_nvfp4_gemv_gather` / `moe_nvfp4_grouped` now decode E2M1 in registers with one barrier (bit-identical
    arithmetic; the workgroup lookup table is gone; the native CUDA twin is unchanged; GPU WGSL speed not
    measured). Gates (CPU backend): `tests/it/tiny_parity.rs` - all 10 visits (in/out, block states, attention and
    MoE inputs, messages, gated q/k/v/mlp, attention, convolutions and the state they leave, relative states,
    router logits, choices and weights) and the logits within 5e-5 of each tensor's scale (measured at most
    2.9e-5); `tests/it/real_parity.rs` - the released weights (about 25 GB resident, 5 min load, 45 s per prompt):
    every visit's in/out and three full logit rows within 1/100 of the reference's own bf16-vs-f32 distance from
    the f32 reference (measured 1/1000), router agreement with the f32 run at least the bf16 run's, the f32 run's
    greedy token at every position. The f32 run is a new golden, `real-f32` (dumper mode of that name, about
    110 GB); the bf16 run is up to 8e-2 of a visit's scale from it and shares down to 95.8% of its choices.
    Finding: the tiny golden's core is one block, so its messages attend over the partial alone and do not pin
    the (q, mlp)/(k, v) query packing; the real gate does (two states per stack).
  - Left: both gates on wgpu, Vulkan and CUDA (they run there unchanged through `testgpu::dev`; no device was
    free); a cached prefill and decode (the tiny golden's `decode*` and `prefill.*` keys, I3.1); the grouped expert
    product for prefill (`moe_route_*`); training (I4.3).
- **I2.4** Decide the parity target for the expert precision.
  - Default: dequantized weights with unquantized activations, which is the HF reference.
  - Opt-in tier: the vendor's activation quantization (NVFP4 per-token "four over six" and
    MXFP8), measured as KL and top-1 against the bf16 reference.

  Precision is a metric, so the low-precision fast path is opt-in.
  - Done (2026-11-22): the default is the reference numerics - expert banks read as published (NVFP4, MXFP8),
    activations unquantized f32 - and it is gated against the f32 reference (I2.3).
  - Done (2026-11-22): the vendor's activation-quantized tier, opt-in and generic. `--linear-activations declared`
    (`LinearActivations::Declared`, never the default or `auto`) makes each expert product read its activation rows
    round-tripped through the format `hf_quant_config.json` declares for it (`ActivationQuant::scheme` in
    `checkpoint::packed::hfquant`, every field checked, anything else refused by name): `model::act_quant::qdq_step`
    over `act_qdq_nvfp4_mxfp8.wgsl`, host reference `model::hostmath::{nvfp4_four_over_six_qdq, mxfp8_qdq}`, bit for
    bit on the CPU backend (`model` test `act_quant`, mutation-checked on the E2M1 tie and the strict four-vs-six rule).
    `Inkling2` applies it to the token rows before `w13` (once per token) and to each slot's activation before `w2`,
    and refuses `declared` on a checkpoint declaring none and `bf16`/`e4m3` (no such product here).
    Spec (source: vLLM main 8c17e469 `vllm/models/inkling2/nvidia/ops/{nvfp4,mxfp8,activation,moe}.py`, the vendor's
    runtime; "four over six" is arXiv 2512.02010, `mit-han-lab/fouroversix` `ScaleRule.mae`; copies and a torch
    transliteration in the reference store `inkling2-actq/`, which agrees with the host reference on 147,456 of
    147,456 values):
    - NVFP4 (layers 0-19): per row `amax = max|x|`, `enc = 1536/amax` (1536 = 6 x 256, not 6 x 448: the six-based
      block scale tops out at 256 so its 1.5x four-based twin fits E4M3), `dec = 1/enc`, `outer = amax * (1/1536)`.
      Per 16-block: `s6 = e4m3_rne(min(bmax * (enc * (1/6)), 448))`, `s4 = e4m3_rne(min(bmax * (enc * 0.25), 448))`;
      each candidate `r = (e2m1_rne_sat(x * (1/(s * dec))) * s) * outer`; error = pairwise-tree sum of `|r - x|`;
      the four-based scale only when strictly smaller. Dequant = `r` (the GEMM's row scale folds `outer` with the
      weight's `amax/denominator`).
    - MXFP8 (layers 20-23): per 32-block `raw = bmax * (1/448)`, E8M0 exponent = `raw` rounded UP to a power of two
      (`(bits + 0x7FFFFF) & 0x7F800000`), values `e4m3_rne(clamp(x/scale, +-448))`.
    - Deviations, stated: brain quantizes its f32 rows (the vendor's are bf16); the vendor multiplies the routing weight
      into the activation before quantizing `w2`'s input, brain after the product (NVFP4 is invariant to a per-row
      scale up to f32 rounding; MXFP8's power-of-two grid is not, so layers 20-23 round on a shifted grid); a row with
      amax below 2^-100 or an MXFP8 block whose `raw` is subnormal reads as zero. The round trip feeding f32 products is
      the quantized product up to accumulation order; a packed W4A4/W8A8 product is a speed item with this as oracle.
    - Measured (CPU backend, the released checkpoint, the three golden prompts, 76 positions; `inkling2` test
      `activation_tier`), with the prompt's expert products grouped by expert: against brain's reference numerics mean
      KL 0.085 nats, top-1 0.934; against the f32 reference's full rows (last 1/2/4, 9 rows) KL 0.029 nats where the
      reference's own bf16 run is 0.006 and brain's reference numerics 2.7e-11; greedy agreement with the f32 run 0.934
      (reference numerics 1.000). With the products gathered per slot instead (same tier, another f32 summation
      order) the same measurement gave 0.118 nats, top-1 0.882 and 0.032 nats: a rounding boundary crossed by a last-bit
      difference flips a code, so the tier is not smooth in its inputs and a 76-position estimate resolves its KL to
      about +-0.03 nats. The reference numerics moved by 1e-11 between the two orders. The per-product round trip's
      relative L2 error is 8.0% (NVFP4) and 2.7% (MXFP8) on heavy-tailed rows.
  - Left: the tier on CUDA/Vulkan (the CUDA bit-exactness test exists and skips without a selected card; not run);
    the serving engine taking `--linear-activations` (its `PolicySupport` row); a split of the KL between the NVFP4 and
    MXFP8 layers; the routing-weight-before-quantization order if the MXFP8 layers' share matters.

### Phase 3: serving

- **I3.1** `model::serve` + `paged`: a generic visit-aware engine core composing:
  - the paged growing KV for full visits (8);
  - a ring or retired window KV for local visits (40);
  - per-visit conv-state rows charged to the pool (`sequence_state_blocks`).

  Admission uses `blocks_for` times visits by kind.
  Done (2026-11-22): `model::serve::visit_state` (`VisitLayout`, `VisitPool<P>`, `VisitStateShape`, `ConvState`,
  `VisitPoolSizing`, `SlotKv`, `Resumed`). Every full slot is a paged plane addressed by the SAME block ids, so the
  scheduler's one `BlockTable` covers all full visits and `blocks_for(tokens)` charges `full slots x 2 x block` rows per
  block (`VisitLayout::block_bytes`); every windowed slot keeps a ring of exactly its window per sequence and every slot
  its declared f32 convolution states: together one state row per sequence (`row_bytes`), rows `0..live_rows` for running
  sequences (`max_batch`), then snapshot rows. The state rows are a reserved store beside the block pool (no sparse
  backing exists off CUDA), so `sequence_state_blocks` stays 0 and `pool_bytes` reports blocks plus every row; `sizing_for`
  splits a byte budget. `attend_chunk` (paged append + `flash_ext_paged_fwd`, or `ring_chunk_attend`) and
  `attend_decode` (append at the block / ring row, then each row through its own table) take any `AttnOptions`;
  `conv_rows` / `store_conv_rows` move convolution state between rows and the `short_conv` batch layout. Spec:
  `crates/model/tests/it/serve_visit_state.rs`, a tiny looped decoder (2 prelude + 2 core x 2 + 1 coda, full and
  window-5 layers, three conv states, sinks) on the pool: the conformance suite (f32 and bf16 KV, prefills sliced),
  and every served logit of a sequence prefilled in slices and decoded far past the window equal to the bit to a
  stateless whole-sequence forward, on a state row another sequence used first. CPU backend; CUDA not exercised.
  Left: state rows charged to the block pool need sparse backing (`qwen35`'s `unipool` is the CUDA mechanism, model
  specific today); a row-indexed conv step would remove the gather/scatter around `short_conv_step`; ouro and huginn
  can adopt the pool (noted in their ledgers).
- **I3.2** `model::snapshot_prefix`: prefix entries spanning all visits (full-visit blocks
  shared, ring and conv rows snapshotted).
  - This is the visit-aware identity ouro and huginn lack.
  - Gate: the serve conformance prefix tests.
  Done (2026-11-22): `VisitPool::{begin, file_prefix, cached_prefix_blocks, reclaim, reclaim_keeping, evict_namespace,
  prefix_stats}` over `SnapshotPrefixCache` (extended with `held_blocks`, no parallel cache): one entry per prompt and
  namespace spans every visit, its table a fork of the full visits' blocks (a resumed sequence copies a partly filled
  tail block before writing), its snapshot row every ring and conv state after the prompt, its payload the engine's
  (the last hidden row: a repeat computes nothing). Admission is charged only the blocks a hit does not adopt. Spec:
  the conformance prefix tests and `a_prefix_hit_restores_every_visits_state_exactly` (an extension and two repeats of a
  cached prompt serve the cold run's logits to the bit; removing the snapshot restore or the tail copy fails it).
  Left: no branch points or checkpoint-before-tail filing yet (`branch_point`, knowledge #347): an engine files where its
  prefill ends; the inkling2 engine should file a chat prompt a few tokens short of its end.
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
  - Done (2026-11-22): engine `inkling2::serve::Engine: PagedDecoder` (`EngineOptions { kv_tier, sizing:
    VisitPoolSizing, max_seq_len, prefill_chunk }`, `Engine::{layout, new, open, model, pool}`, `device_ops()`), one
    recipe with the uncached forward: `model.rs`'s `trunk` runs over `Rows::{Whole, Chunk, Decode}` - attention through
    `VisitPool::attend_chunk` / `attend_decode`, all four convolution states per visit through `conv_rows` /
    `store_conv_rows` (history + `conv_state_tail_step` in a chunk, `short_conv_step` in decode), the published banks
    grouped by expert for whole/chunk passes (`MicroBank::grouped` over the new `model::ops::MicroRoute`: barrier-free
    `moe_route_count_serial` / `moe_route_emit_serial` + `moe_route_scan`, so it runs on the CPU JIT) and gathered per
    slot in decode; head transform declared (cap 30, valid 200,058), greedy/top-k reduced on the device; prompt
    embeddings spliced after the embedding norm (`embed_width`, `prefill_chunk_embeds`); prefix entries filed
    `CHECKPOINT_TAIL` (16) tokens short of a prompt of 32+ (knowledge #347), content-keyed (`VisitPool::begin` /
    `cached_prefix_blocks` / `reclaim_keeping` take a `snapshot_prefix::Prompt`, `file_prefix_content`); routing recorded
    per logical visit (`routing_calls`, I4.5's log). Gates (CPU): `inkling2/tests/it/serve.rs` - the tiny golden's
    `prefill.visit*.conv*` states and `decode0..3.logits` (whole and sliced prefill), served logits == uncached forward
    at every position (prefill whole / in 4s / token by token, batched beside another sequence), `conformance::run_all`,
    routing log == the uncached router at every computed position (`SchedulerRollout::record_routing(true)`), and the
    released weights: cached greedy decode of the three real-golden prompts (6 tokens) == the uncached forward's argmax
    (about 25 GB resident, 12 min); `real_parity` still passes with the grouped prefill (worst 1/1000 of the bf16
    spread). Serving: `catalog::resident_inkling2` (`Inkling2Resident::from_hf_dir`, estimate = shard bytes + the
    pool + step buffers, f32 K/V, ctx 16384 unless `--ctx`, `--max-batch` live rows and as many snapshot rows),
    `generate_spec(..).params(template_params(TypedMessages))`, `parse_request_as` with the checkpoint's own template
    (TypedMessages scanner), stop = `stop_ids` over `end_of_turn()`; registrations: catalog row (`FamilySpec`
    inkling2), serving `model_dir` (`card.family == "inkling2"`), cli alias `inkling2`; docs page, sample
    `samples/python/llm/inkling2`; `catalog/tests/it/inkling2_served.rs` (served chat within ctx, effort refused out of
    range, image refused, `certify` cancellation, estimate >= weights + pool). Stop set decision: TypedMessages
    `end_of_turn_markers()` is the sampling eos alone (`<|end_message|>` closes the thinking message too); test that a
    think-then-answer stream is cut only at its end.
  - Left: images through the request path (apiserve media -> `VisionTower::embed_prompt` -> `PromptEmbeds`; the engine
    splice and content-keyed prefixes are in, the front end is not: an image request is refused by name); adapters in
    the engine (an `AdapterBank::new_with_experts` over the decoder's linear sites and the w13/w2 expert sites, corrections
    after each product; the catalog path `llmserve::adapter_release` now reads stacked expert pairs through
    `parse_adapter_with_experts`, done 2026-11-22, so only the engine side is missing);
    a recorded/graph decode step (every pass allocates its buffers, attention and conv state submit per visit; no
    profile yet); bf16 K/V as a measured opt-in; MTP drafting.
- **I3.5** Acceptance through the public API with the vendor's published verification
  suite (coding, vision, long context to 1M), after operator parity.

### Phase 4: training on the frozen quantized base

The first trainable target needs no streaming: the base is about 19.5 GB.
- **I4.1** `model::moe`: a backward through frozen `Micro` expert banks (NVFP4 and MXFP8
  grouped dX). The router gradient is optional, behind a declared flag.
  - Done (2026-11-22): kernels `moe_{nvfp4,mxfp8}_dx_gather` / `_dx_grouped` (the transposed bank products, one thread
    per output, barrier-free, CPU JIT; grouped is the gather form to the bit) behind `MicroBank::{gather_dx,
    grouped_dx}` and the id-range forms `gather_at` / `gather_dx_at` (in `micro_bank_kernels()`). The row-compacted
    training path takes a declared activation and bank: `ExpertSet { glu: GluSpec, bank: ExpertBank::{Dense(&[GluWeights]),
    Micro { gate_up, up, down }} }`, `MoeActs::compact_for(g, shape, host_gate, &spec)`, `experts_fwd_train_set`,
    `moe_layer_bwd_set(g, RouterGrad::{Full{..} | Detached}, ..)` (a frozen bank takes no weight gradients, refused by
    name); `COMPACT_TRAIN_KERNELS` now carries `glu_act` / `glu_act_bwd`; the SwiGLU `experts_fwd_train` /
    `moe_layer_bwd` share the same compacted frame unchanged. Grouped path: `expert_fwd_grouped_glu` +
    `GroupedExpertScratch::new_for`. Gates: `moe_micro_bwd.rs` (bank dX vs the f64 transposed product of the decoded
    weights for both kinds; a softcap-sigmoid interleaved MXFP8 layer's output vs the f64 forward and its dX vs f64
    central differences with the router differentiated and detached; refusals), `moe_block_gradcheck`
    (`compacted_softcap_sigmoid_experts_pass_the_gradcheck`, compacted == dense declared path), `moe_grouped_parity`
    (interleaved grouped == separate grouped); existing moe gradchecks, deepseek2/glmdsa/qwen35moe suites green.
  - Left: the NVFP4 bank's forward products need workgroup reductions, so the NVFP4 layer-level check skips on the
    CPU backend (its dX kernels ran); CUDA not exercised; the dX kernels are the reference tier (no tiled/tensor-core
    candidate, no `gpu_core::cost` formula); the grouped forward over a `MicroBank` stays the serving path's
    (`MicroBank::grouped`), training reads the bank through the compacted gather.
- **I4.2** `model::adapter`: stacked-expert LoRA sites (per-expert A/B over a bank), plus
  sites for:
  - attention q/k/v/o/gate;
  - relative states;
  - sink projection;
  - message-gate down/forget/gain;
  - depth queries.

  Serving bank overlays for each.
  - Done (2026-11-22): training through the stacked-expert adapters on the row-compacted path:
    `model::moe::{ExpertLora { a: [E, r, k], b: [E, n, r], rank, scale }, ExpertLoraGrads, ExpertAdapters { gate_up, up,
    down }, ExpertAdapterGrads, experts_fwd_train_adapted, moe_layer_bwd_adapted, EXPERT_LORA_KERNELS}`
    (`crates/model/src/moe/expert_lora.rs`; `experts_fwd_train_set` / `moe_layer_bwd_set` are these with no adapters).
    Per expert, on its routed rows only, `proj += scale * B_e (A_e x)` after the frozen bank product (gate_up before the
    activation, down before the scatter); the backward ADDS the routed experts' `dA_e` / `dB_e` (an unrouted expert's
    range is not written) and the adapters' share of `d_h` and `d_x`. No new kernel: `moe_linear_gated` / `_dx` / `_dw`
    (element offsets into the stacked factors), `axpy`, `grad_scale`; `A_e x` is recomputed in the backward, nothing
    saved. Adapters beside a dense (trainable) bank are refused. Gate `crates/model/tests/it/moe_expert_lora.rs`: every
    A/B entry of gate_up and down adapters over a frozen MXFP8 softcap-sigmoid interleaved bank vs f64 central
    differences, the idle expert's gradient bits untouched, `d_x` vs f64 central differences with the router
    differentiated and detached, refusals. CPU backend; NVFP4 skips there (bank forward needs workgroup reductions);
    CUDA not exercised.
  - Done (2026-11-22): stacked-expert sites, file format and serving overlay. `adapter::sites::{ExpertSite { name, leaf,
    layer, experts, out, inn, top_k }, LoraSites::expert_sites (default empty), expert_tensor_shapes,
    validate_with_experts, SiteMismatch::ExpertShape}`, `Model::adapter_expert_sites`, `TargetSelector::matches_expert`;
    expert leaves resolve in `AdapterTargets` / `AdapterServing::all_leaves`. File: `{bank}.lora_a [E, r, in]`,
    `{bank}.lora_b [E, out, r]` (real rank-3 shapes, written by `save_lora_with_lineage`); `DeviceAdapter::experts:
    Vec<ExpertPair>` filled only by `read_adapter_with_experts` / `parse_adapter_with_experts`; the linear readers
    (`read_adapter`, `parse_adapter`, `FoldedSource::open`, `AdapterFile::read`) refuse a stacked pair by name
    (knowledge #553: `StModel` now carries the header shapes). Fold: `fold_adapter_into` folds a stacked pair into an
    f32 bank held as one `[E, out, in]` tensor, expert by expert; a quantized bank has no fold (refused, served beside
    the bank). Serving: `AdapterBank::new_with_experts(gpu, sites, expert_sites, cfg, rows, layers)` holds `slots * E`
    sub-slots per targeted expert site and `AdapterBank::correct_experts(gpu, steps, bank_name, x, xdiv, ids, out,
    slots)` corrects a routed bank product `[slots, n]` after it (kernel `lora_expert_slots` composes
    `row_slot[s / top_k] * E + ids[s]`, `embed` gathers the token rows when `xdiv > 1`, then the existing
    `lora_bank_rows`); kernels `model::dispatch::LORA_EXPERT_BANK_PIPELINES`. Gate
    `crates/model/tests/it/adapter_expert_sites.rs`: one SGD step over a frozen MXFP8 bank, save, read back (shapes,
    card, bits), bank registration, a mixed batch served (adapted rows = the training forward, others the base bits),
    budget = allocation, per-expert fold, refusals. Ordinary linear LoRA over 2-D `Weight::Micro`: `qops_frozen_lora_grad`
    runs every tier including nvfp4 / mxfp4 / mxfp8 forward, dX, LoRA dA/dB on the CPU backend (only the bf16
    tile-decode case skips there); nothing missing.
  - Left: the generic catalog path (`llmserve::adapter_release` -> `parse_adapter`) still reads linear pairs only, so a
    stacked adapter reaches a bank through `parse_adapter_with_experts` in the model's own loading until the catalog
    switches (every `AdapterBank::register` already checks expert pairs); adapters beside a dense expert bank in
    training; LoRA-FA (gradient buffers for B only) on expert sites; the attention / relative /
    sink / message-gate / depth-query sites are ordinary `LinearSite`s the model declares; CUDA not exercised.
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
  - Done (2026-11-22): `inkling2::train` - `Inkling2Train` (`model::Model` + `TokenObjectives`), `TrainConfig`
    (`ModelConfig`; `tiny_config`, `tiny_config_every_expert_routed`), `sites` (`lora_sites`, `expert_sites`,
    `DEFAULT_TARGETS`, `targets`), `finetune::Inkling2Trainer` (`ChatTrainer`), SDK row `inkling2` (study + chat
    trainer). The backward walks the visits in reverse, recomputing each from its saved input partial (saved: every
    visit's input partial and the `RetainedStates` block states; the prelude-output slot shared by every core pass,
    the previous pass's own core block states read by the next pass's messages); `dB[j]` per partial, slot gradients
    collected at the retaining visit, the core's parameters accumulated over the passes; the message queries gathered
    from and scattered back into the two published tensors. Ops: `depth_attn`, `message_gate`, `flash_ext` (window,
    sink, distance bias, tau, packed documents), `short_conv` (static/dynamic, segments), `l2clip_scale`,
    `rmsnorm_nogain`, compacted MoE (`experts_fwd_train_adapted` / `moe_layer_bwd_adapted`, `RouterGrad::Full` or
    detached by `set_detached_router`), chunked `HeadCe` with `HeadTransform`. Full fine-tune (f32, float experts) and
    LoRA over the frozen base: bf16 or f32 linears through `AdapterRuntime`, the published NVFP4/MXFP8 banks read in
    place with stacked per-expert pairs, and the f32 matrices no linear reads (depth/readout/message queries,
    message-gate factors) through the new shared `model::adapter::merged::MergedLora` (`W + s B A` formed per step,
    pair gradients projected from `dW`). Routing replay: one MoE call per logical visit (`routing_calls`),
    `router_fwd_replay` after every router, `Inkling2Train::taken_routing` hands the last forward's routing back as a
    replay. Packed rows (`reads_packed_rows`, `Batch::LmPacked`). `HeadCe` fixed: the logit scale multiplies the
    capped logits (knowledge #561). Gates (CPU backend): training forward == tiny reference logits and its loss ==
    their CE (`tests/it/training.rs`); `gradcheck` `inkling2_fd` - every parameter of a full fine-tune (directional),
    every shared core tensor and the message queries entry by entry, every adapter tensor of a LoRA build (routing
    pinned by replay: a finite difference must not flip a top-k, knowledge #562); rl conformance `inkling2_conforms` (C1-C4, C6, C10
    and every objective's step; the subject routes both of two experts, since the suite cannot pin a routing);
    bf16 base == f32 over rounded values; packed row == documents alone; LoRA run lowers the loss and its saved adapter
    reloads; `rl/tests/grpo_inkling2.rs` (on-policy ratio, routing replay applied and exact on the trainer's own
    routing); capability matrix rows.
  - Done (2026-11-22): image-text training - `train::image_text::ImageTextTrainer` over `Inkling2Train` and a
    `VisionTower::new_trainable` on the decoder's device (the training kernel list carries `vision::train_pipelines`):
    `encode_train`, `splice_embeds` into the batch's flattened rows (`Inkling2Train::set_image_rows`, after the
    embedding norm as the reference scatters), the decoder's forward and backward, the image rows' gradient taken out
    of the embedding gradient (`take_image_gradients` via `PromptEmbeds::take_row_gradients`, so the placeholder's token
    row gets none) into `VisionTower::backward`; frozen tower by default, tower and merger LoRA opt-in, one AdamW per
    store (each clipped to its own norm). Gate `tests/it/image_text.rs`: tiny decoder + tiny tower, the tower adapters'
    gradient equals central differences of the whole image-text loss (decoder routing pinned), and both adapters lower
    the loss.
  - Done (2026-11-22): `Inkling2Trainer::serving_engine` opens `serve::Engine` over the release (f32 K/V, one
    sequence) for `served_loss` and precision searches; its served logits equal the trainer's at every position
    (`the_serving_engine_serves_what_the_trainer_trains`). An adapter is refused by name.
  - Done (2026-11-22): real-weight QLoRA smoke run (`tests/it/real_lora.rs`, skips by name without the store entry):
    the release as published (NVFP4 layers 0-19 and MXFP8 20-23 read in place, bf16 dense linears), a rank-4 adapter
    on every site including the stacked expert pairs, one 19-token row, AdamW 2e-3: loss 1.240 -> 0.958/0.690 -> 0.204/
    0.145 -> 0.008 after three steps (two seeds); the adapter saved and read back (`continue_from`) over a fresh load
    computes the trained loss within 1e-4. CPU backend, release build: load 47 s, about 450 s per step (forward,
    per-visit recompute and backward); in about half of the stack samples taken during the steps a single one of the
    288 threads was busy, which fits the CPU backend running a dispatch of at most 16Ki invocations on the calling
    thread (the per-expert bank and adapter products are that small) - a hypothesis, not measured per kernel (the CPU
    backend has no kernel timing).
  - Left: the served engine applies no adapter (no `AdapterBank` in `serve::Engine`; the stacked expert pairs cannot
    be folded into quantized banks), so a trained adapter is not served yet and `serving_engine(.., Some(adapter), ..)`
    refuses; C8 (chunked head) and C9 (tiers) are not run by the conformance subject (the tiny rows fit one head
    chunk; bf16 is gated in `training.rs` with the routing pinned); expert pairs over float banks (the shared op
    refuses them, I4.2 Left), so a small checkpoint's adapter leaves `w13_weight`/`w2_weight` out; full fine-tune of the
    released 26B needs a streamed window (I6.2), host/ZeRO optimizer state (I6.1) or expert parallelism (I6.3); the
    training forward allocates per stage and syncs on the host once per MoE call (the compacted routing reads the
    gate) - no recorded tape, no CUDA run; `export_checkpoint` refuses to fold an adapter (its expert pairs correct
    quantized banks).
- **I4.4** `model::train` + `data`: packed rows with document-boundary resets of
  attention, every conv state and the relative-position origin. Gate: no supervision or
  state crosses a boundary.
  Done (2026-11-22): `model::flash_ext` takes declared packed documents on the contiguous layout
  (`AttnOptions::documents`, `AttnInputs::doc_pos` = each row's in-document position from
  `data::packing::document_positions`): forward and every gradient (dQ, dK/dV, d-sink, d-rel, d-bank) see only
  same-document pairs at in-document positions (window, bias distance, tau); paged forward unchanged and refuses it.
  `model::short_conv`'s existing `seg` input is the conv-state reset (reused, not duplicated). Data: `data::packing`
  (`RowPacker` greedy whole-example packing, `PackedBatch`, `PackingError`), `TokenDataset::packed_batch` /
  `example_len` (IGNORE on each document's last token, padding its own segment). Plumbing: `Batch::LmPacked { tokens,
  targets, segments }`, `Model::reads_packed_rows` (default false), `training::Capability::PackedRows`,
  `model::causal_lm_packed` (epoch order, exact resume through its cursor, held-out batches packed too; refuses a
  model that does not read packed rows, a token stream and a sharded run). Specs: `model/tests/it/flash_ext.rs`
  (packed oracle + finite differences, each packed document == alone with every option on), `short_conv.rs`
  (packed == alone, bit for bit), `packed_rows.rs` (conv + attention + CE: packed loss and gradients == documents
  run alone), `data/tests/it/packed_rows.rs`, `model::train` unit tests. CPU backend verified; CUDA not exercised.
  Left: no model reads packed rows yet (inkling2 opts in: `reads_packed_rows`, `LmPacked` in `set_batch`); packed
  runs are not sharded; the training-capability matrix gets a row once a model declares it.
- **I4.5** `rl` + `model::rollout`:
  - routing replay (rollout expert ids recorded per logical visit and forced in the
    trainer);
  - an inkling2 rollout engine over the serving engine;
  - the serve-vs-train log-prob gap measured before any reward is read.

  Done (2026-11-22): routing replay as generic infrastructure. Kernel `router_gate_replay` +
  `model::moe::{RouterOverride, router_fwd_replay, ROUTER_REPLAY_KERNEL}`: dispatched after `router_fwd_kind` on the
  same gate, it forces a recorded top-k selection per row with the router's own weights over the current logits
  (softmax or sigmoid, renormalised `+1e-9` floor / `+1e-20` added, times the routed scale), leaves rows whose ids are
  `model::routing::UNRECORDED` as routed, and sums the selected scores in the router's greedy order, so replaying the
  router's own selection is bit-identical to it; `router_bwd` / `router_bwd_sigmoid` read the selection as the gate's
  nonzero pattern and differentiate it unchanged (spec `crates/model/tests/it/router_replay.rs`: bit identity,
  formula against f64, unrecorded rows, device-forward finite differences and an f64 reference of the backward).
  `model::routing`: `RouterCallShape`, `RoutingRecord { policy_version, positions, recorded, calls: [RoutedCall {
  top_k, experts }] }` keyed by MoE call in forward order (one call per visit of a looped layer), `validate` naming
  the field (serde `deny_unknown_fields`), `RoutingLog` (engine side), `RoutingReplay` (a batch's records laid out per
  call). Recording: `PagedDecoder::routing_calls` (default `None`), `BlockTable::routing_log_mut` written by the
  engine for every position it computes (prefix-cache hits stay unrecorded, the log survives preemption),
  `Scheduler::record_routing`, `SchedulerRollout::record_routing` -> `Completion::routing`. Trainer:
  `TokenObjectives::{routing_calls, set_routing_replay}` (refusal by default, `Capability::RoutingReplay`). Gate
  `crates/model/tests/it/routing_replay_rollout.rs`: a tiny MoE served through the scheduler with its router summed in
  the other order flips a near-tie; without replay the trainer's log-probability is off by > 0.05, with replay it
  equals the served one (< 1e-3, the rounding of a cancelling router), softmax and sigmoid. GRPO:
  `GrpoOptions::routing_replay` trains and scores rows with their recorded routing (refused by name on a model that
  cannot replay); every GRPO run reports `GrpoStats::{serve_gap, serve_gap_replayed}` and every external
  policy-gradient run `PolicyGradientStats::serve_train_gap` (grpo-corrections.md). CPU backend verified; CUDA not
  exercised. Left: no engine or trainer implements the seam yet: the inkling2 engine records into the block table's
  log per (visit, layer) call and its `ChatTrainer` passes `RoutingReplay::upload_call(g, call, top_k)` to
  `router_fwd_replay` after each router; external policy-gradient records carry no routing (no record format for it);
  `PipelineModel` does not forward the replay to its stages.

### Phase 5: vision

- **I5.1** `inkling2` over `model::vit`:
  - interpolated learned positions, using the shared bilinear position-table op (add it
    if absent);
  - axial 2-D RoPE;
  - a class token per image;
  - block-diagonal spans;
  - the 2x2 merge projector.

  Done (2026-11-22): `inkling2::vision` (`VisionWeights::load`, `VisionTower::{new, encode, encode_traced}`,
  `pipelines()`). Interleaved-pair RoPE is the new `rope2d_pairs` kernel filling `VitKernelIds::rope2d` (same Params as
  `rope2d`, backward is sign -1); the position table is `model::postable` (`bilinear_grid_plan`, `DeviceTaps`) over the
  new `rows_gather_weighted` and its CSR-transposed `rows_gather_weighted_bwd`, both with finite-difference specs and
  host references in `hostmath`. The shared block's layer scale, LayerNorm bias and per-image spans are used as they
  were. Parity on the CPU backend, f32: tiny random tower in the published naming (embeddings, `ln_pre`, every layer,
  merger input and output, worst rel L2 5e-7) and the published weights on two images (embeddings, layers 0/11/22,
  last hidden, merger, worst rel L2 5e-6); images in a batch equal images alone (block-diagonal). CUDA and Vulkan not
  exercised; the fused attention and register-tiled GEMM rows are registered and resolved by name, so a GPU takes them,
  but their parity on this tower is unmeasured.

  Left: training the tower needs `vit_block_fwd_cached`/`vit_block_bwd` wired per layer (the `rope2d` slot with sign -1
  already inverts the pair rotation), the patch-embedding weight gradient, the class-token rows' gradient,
  `DeviceTaps::table_grad_step` for the position table, the merger through a trainable `MlpProjector`, and a
  `bias: false` projector config (zero biases are fed today, exact but a parameter that does not exist). The projector
  is rebuilt per forward (its activations are sized for the region count), which re-uploads 100 MB of merger weights per
  call; serving wants a capacity-sized projector or a two-matmul step list on resident weights. TF32 tensor-core
  products (`vit_block_fwd_tc`) are not passed. Tower activations are f32 end to end; the reference's bf16 rounding
  points are not reproduced.

  Done, training (2026-11-22): `vision::{VisionTower::new_trainable, encode_train, backward, set_saved_policy,
  zero_grads, lora_store, store, embed_prompt_train, VisionPass, LoraTraining, train_pipelines}` over a `ParamStore`
  (published names, q/k/v fused as `fused_qkv_name`). Per-layer backward on `vit_block_fwd_cached`/`vit_block_bwd` with
  `SavedPolicy::{Keep, Recompute}` (bit-identical gradients); patch-embedding, class-token (both parameters),
  position-table (`DeviceTaps::table_grad_step`) and `ln_pre` gradients; the backward stops at the lowest tensor that
  needs a gradient. Shared pieces: `ProjectorConfig::without_bias` (no zero biases), `MlpProjector::{with_buffers,
  resized, forward_rows, backward_rows}` (capacity-sized resident merger sharing the store's weights; serving no longer
  re-uploads 100 MB per call), `adapter::fold::{fold_fused, Slice, FusedTarget}` plus `GradTarget::offset` (several
  adapter sites in one fused weight), `PromptEmbeds::take_row_gradients` (the splice backward), `impl LoraSites for
  VisionConfig` (q, k, v, out_proj, c_fc, c_proj, proj_in, proj_out). Gates on the CPU backend: finite differences of
  every parameter class and of adapter factors on a random tower, keep == recompute bitwise, a frozen-tower LoRA step
  lowers a toy loss, existing vision parity unchanged (tiny and published tower).
  Left: CUDA/Vulkan not exercised; weights without gradient buffers still get dense dW computed into a shared sink
  (a skip-dW option in `vit_block_bwd` would save about a third of the backward of a LoRA run); training caches
  attention probabilities per image (bounded by `MAX_CACHE_BYTES`, needs hidden size a multiple of 64), so large
  images need a flash backward; adapters are plain LoRA with one rank (fold), other kinds are refused; tower updates
  are host-side through the store (no device optimizer wiring in this crate).
- **I5.2** `imaging`: the preprocessor (pad, checkerboard alpha, normalisation, region
  grid), byte-exact against the reference processor.

  Done (2026-11-22): `imaging::regions` (`RegionTiling::{preprocess, preprocess_bytes}`, `composite_over_checkerboard`,
  `RegionImage`) and the alpha-keeping decode `imaging::decode_alpha_bounded`. Bit-identical `pixel_values` and grid for
  RGB, RGBA (every alpha class), grayscale, odd sizes, 1x1 and exact multiples (PNG sources). The reference processor
  cannot be imported in the pinned venv (no torchvision), so the dumper replays its `convert_to_rgb` and `_preprocess`
  on torch tensors; the checkerboard, the -1 pad and the fused mean/std are taken from the processor source, not from
  `processor_config.json` (which carries only mean, std, rescale and the 40-pixel size). Sides over 4096 pixels and
  empty images are refused by name; the pixel bound is applied from the header before decoding.

  Left: the exactness claim is for 8-bit sources decoded losslessly. JPEG is decoded by `image`, not libjpeg, and may
  differ by an IDCT rounding step from PIL; 16-bit PNGs go through `image`'s 16-to-8-bit conversion, which is not PIL's.
- **I5.3** Image-text parity and the `PromptEmbeds` splice into the scheduler. Frozen-tower
  LoRA comes after that.

  Done, tower half (2026-11-22): `vision::{expand_placeholders, splice_embeds, VisionTower::embed_prompt}` and
  `ImageProcessorSpec` (typed `processor_config.json`, refuses a recipe other than the trained one). The chat template
  renders one `<|unused_200054|>` per image; the processor expands it to one per 40x40 region; the features of all
  images in image order fill the placeholder positions in order (`get_placeholder_mask`). `embed_prompt` from PNG bytes
  equals the tower on the reference processor's own pixels. Left for the model lane: the serving splice of
  `PromptEmbeds` into the scheduler, and image-text logit parity once the decoder is up.

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
  - Done (2026-11-22): the tap API is keyed by logical depth (`PagedDecoder::{enable_taps, tap_depths, read_taps}`,
    `TapPool::depths`: a decoder layer of a plain stack, a visit of a looped one, each decoder documenting what its row
    holds). Verify rollback over the visit pool: `VisitPoolSizing::undo_rows` reserves one undo state row per live row,
    `VisitPool::checkpoint` keeps every ring and convolution state before a verify writes ahead and
    `VisitPool::rollback` restores them; the engine recomputes the kept rows (full visits need nothing: rejected
    positions are truncated and rewritten). Spec `serve_visit_state.rs`
    (`a_verify_that_keeps_part_of_its_rows_leaves_every_ring_and_convolution_as_decoding_them`, and the conformance
    suite's `speculation_equals_plain_greedy` on the tiny looped decoder, which now speculates); removing the rollback
    fails both.

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

## GPU validation (2026-11-22)

One GH200 (CUDA 13 and the native Vulkan backend, adapter `NVIDIA GH200 120GB (Vulkan compute, ash + naga
WGSL->SPIR-V)`), main at `a799c05b6` plus the three fixes below; every suite run whole, both backends.

| Suite | CUDA | Vulkan |
|---|---|---|
| `brain-model` it: the campaign modules (softcap ... gqa_mixer_windowed), `bf16_roundtrip`, `qops_frozen_lora_grad` | 95/95 | 95/95 (serial) |
| `head_ce::bf16_head_at_real_width_runs_4096_tokens_and_reports_its_time` (GPU-only, ignored elsewhere) | pass: forward 0.68 s, backward 1.7-2.1 s | - |
| `brain-model` `moe_block_gradcheck` | 17/17 | 17/17 |
| `gpu-core` it `moe_mxfp8` | 2/2 | 2/2 |
| `gradcheck` it `ouro_fd`, `huginn_fd`, `backend_parity` | 4/4 | 4/4 |
| `ouro` it, `huginn` it | 24/24, 19/19 | 24/24, 19/19 |
| `inkling2` it (layout, tiny_parity, real_parity, serve incl. the released weights' cached greedy decode, vision, vision_train) | 25/25 | 25/25 |
| `catalog` it `inkling2_served` | 2/2 | 2/2 |

Defects found and fixed:
- `micro_bank_stacked` held the NVFP4 bank's gather and grouped products to the bit on CUDA, where both run native
  twins that sum in other orders; each is now held to the f64 oracle and the bits only on the WGSL tier, as
  `gpu-core`'s `moe_nvfp4` gate already states.
- The default arithmetic (`GemmPrecision::Auto`) routed through bf16 tensor-core products while the reference keeps the
  router in binary32: on CUDA the last visits drifted to 7.5x the bf16 reference run's distance, KL up to 1.1e-1 and a
  greedy token led by 0.65 logits changed. The router is now an f32 weight under `fp32_kernels` (#557).
- Vulkan reported no bf16/f16 storage, so `Ops::lands` widened such weights to f32 (`head_ce` x3, `qops_tier_coverage`
  failed); fixed in `backend-vulkan` and `ArchDesc::numeric_view` (#558).
- Not fixed, pre-existing (#300): tests sharing the pooled Vulkan device concurrently lose it (`flash_ext`,
  `serve_visit_state`: DEVICE_LOST, a SIGSEGV in `router_replay`); every one passes alone or with `--test-threads=1`.

Real weights (`real_parity`, binary32 products): every visit and the three full logit rows within 1.2e-4 to 3.2e-4 of
the reference's own bf16-vs-f32 distance (CUDA and Vulkan alike; the gate allows 1e-2), router agreement at least the
bf16 run's, the f32 run's greedy token at every position. The default arithmetic, the served path's precision metric
against the f32 golden (worst KL of the three full rows; greedy agreement at every position), on CUDA: KL 2.2e-2 /
3.6e-5 / 7.0e-3 nats for the three prompts (bf16 reference run 4.4e-2 / 2.4e-3 / 6.3e-3), greedy 22/22, 27/27, 26/27
(the change at a near tie, lead 4.2e-2). On Vulkan the default is binary32 (no library GEMM): KL 5e-11.

Uncached forward of one prompt (22-27 tokens): CUDA 0.51-0.54 s warm at the default arithmetic (0.57-0.66 s binary32
with every visit read back); Vulkan 4.7-7.2 s. The model loads in about 40 s from a page-cached file.

Profile (nsys, CUDA, the `real_parity` run before the rebase: 9 forwards, 4.35 s of kernels, about 0.48 s per forward),
the Phase 7 gap list:
1. `depth_attn_fwd` 69%: 2.8 ms a launch, 120 launches a forward; the barrier-free reference kernel (one invocation
   per row) for a softmax over at most 8 states.
2. `moe_nvfp4_gemv_gather` 15% (+ `moe_mxfp8_gemv_gather` 1.3%): 0.8 ms a launch; `Inkling2::forward` runs the decode
   regime's per-slot gather for a whole prompt (the served engine groups by expert in prefill).
3. `flash_attn_gqa_ext` 4.4% (0.44 ms a launch, barrier-free reference), `matmul_gemv_w_bf16` 4.2%,
   `router_gate_sigmoid` 3.1% (0.31 ms for 25 x 128 scores, barrier-free).
4. Host: about 3,500 `cuMemAlloc`/`cuMemFree` pairs and 3,000 launches a forward (2.5 s in `cuMemFree` over the run):
   per-step storage instead of retained blocks or a step arena.
