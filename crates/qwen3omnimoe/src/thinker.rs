// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni's Thinker text decoder: a Qwen3-style GQA+QK-norm+RoPE
//! attention stack (identical shape to `qwen3::Qwen`'s, and to `qwen3tts::talker`'s
//! reuse of it) over a sparse top-k MoE FFN (`model::moe`, no shared
//! expert).
//!
//! **Why this is a new, dedicated forward rather than an extension of
//! `qwen3::Qwen`**: `qwen3::Qwen`'s attention/RoPE/QK-norm/DeepStack-splice
//! machinery is already exactly right for Thinker (confirmed by reading
//! `forward_steps` in full - same RMSNorm, same per-head QK-norm, same
//! `block::gqa_fwd`, an existing `mrope`/`rope2d_step` path, and DeepStack
//! splice support already wired for the vision-language case). But its
//! internals (sharding, LoRA, int8, KV-cache) are all interleaved in one
//! large constructor with no seam to swap out just the dense SwiGLU MLP for
//! `model::moe`'s sparse one, unlike `crates/glm`, which already carries an
//! `Mlp::Dense`/`Mlp::Moe` enum at exactly that point (giving `qwen3::Qwen`
//! the same seam `glm` has is the "one
//! implementation" answer and the natural following step). This module is
//! deliberately narrower than `qwen3::Qwen` in every other respect
//! (forward-only, single device, no sharding/LoRA/int8/KV-cache) so it can
//! be validated against real weights now - it composes the SAME shared
//! primitives `qwen3::Qwen` itself is built from
//! (`model::block::{rmsnorm_fwd,rope2d_fwd,gqa_fwd}`), not a re-derivation of
//! the attention math.
//!
//! **M-RoPE**: every layer takes the table-driven [`model::block::rope2d_fwd`]
//! path (the same kernel `qwen3::Qwen::rope2d_step` dispatches for Qwen3-VL),
//! fed a `[n, head_dim/2]` `cos`/`sin` table the caller builds with
//! `qwen3vl::mrope::{get_rope_index, mrope_tables}`. There is deliberately no
//! separate "plain RoPE" code path: for a token stream where all three axes
//! carry the same position (pure text, or pure audio), Omni's interleaved
//! M-RoPE collapses exactly to ordinary half-split
//! RoPE (`qwen3vl::mrope`'s own `diagonal_positions_collapse_to_plain_rope`
//! test proves this), so a caller with no image/video/audio span just builds
//! that degenerate diagonal table via `get_rope_index(tokens, image_token_id,
//! &[])` (empty grids) rather than reaching for a second kernel - one
//! implementation for both cases, avoiding the risk of wiring the wrong
//! RoPE kernel into a second, parallel path.
//!
//! **Multimodal splice**: not this module's concern. A caller with image/
//! audio embeddings splices them into the token-embedding buffer via
//! `model::vlm::splice_fwd` (`splice.wgsl`, in [`thinker_pipelines`]) BEFORE
//! calling [`decode`] - `decode` and [`layer_fwd`] only ever see an
//! already-assembled `[n, d]` embedding sequence, exactly like `x` in
//! `qwen3::Qwen::write_img_embeds`'s contract.

use gpu_core::{DeviceBuffer, Gpu};
use model::block::{self, gqa_attn_sublayer_decode_step, gqa_attn_sublayer_fwd, rmsnorm_fwd, GqaAttnDims, GqaAttnIds, GqaAttnWeights, GqaDecodeIds, KernelIds};
use model::int8::{quant_rows_steps, QuantRows};
use model::moe::{expert_fwd, expert_fwd_i8, router_fwd, ExpertScratch, ExpertScratch8, MoeIds, MoeIds8};

use crate::config::MoeTextConfig;
use crate::int8_resident::ThinkerLayerExperts8;

/// Kernel pipeline this module dispatches. Forward-only: the backward slots
/// `KernelIds` carries (`rms_inv`/`rmsnorm_dx`/`rmsnorm_dw`/`rope_bwd`/
/// `gqa_d*`/`silu_da`/`silu_db`) are never reached, so they point at index 0
/// (`rmsnorm`) rather than a real backward kernel - harmless since nothing
/// dispatches them, and it keeps this list short.
pub fn thinker_pipelines() -> &'static [(&'static str, &'static str)] {
    &[
        ("rmsnorm", kernels::RMSNORM),               // 0
        ("rope2d", kernels::ROPE2D),                  // 1 -- table-driven M-RoPE; see the module doc.
        ("gqa_scores", kernels::GQA_SCORES),          // 2
        ("attn_softmax", kernels::ATTN_SOFTMAX),      // 3
        ("gqa_apply", kernels::GQA_APPLY),            // 4
        ("matmul", kernels::MATMUL),                  // 5
        ("add2", kernels::ADD2),                      // 6
        ("router_gate", kernels::ROUTER_GATE),        // 7
        ("moe_linear_gated", kernels::MOE_LINEAR_GATED), // 8
        ("silu_mul", kernels::SILU_MUL),              // 9
        ("scale_add", kernels::SCALE_ADD),            // 10
        ("splice", kernels::SPLICE),                  // 11 -- for a caller wiring multimodal input; see the module doc.
        ("kv_append", kernels::KV_APPEND),                   // 12 -- KV-cache decode; see `layer_decode_step`'s doc.
        ("attn_decode_scores", kernels::ATTN_DECODE_SCORES), // 13
        ("decode_softmax", kernels::DECODE_SOFTMAX),         // 14
        ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),   // 15
        // int8 MoE expert dispatch -- see `moe_sublayer`'s `int8_experts`
        // parameter and `crate::int8_resident`. Registered unconditionally
        // (harmless, matching every other kernel here, whether or not a
        // given resident instance ever passes `Some`).
        ("max_abs_row", kernels::MAX_ABS_ROW),               // 16
        ("quant_pack", kernels::QUANT_PACK),                 // 17
        ("moe_linear_gated_i8", kernels::MOE_LINEAR_GATED_I8), // 18
        // General (non-MoE) int8 GEMM -- see `lm_head_fwd_i8`'s doc.
        ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),           // 19
        // O(T*head_dim)-memory causal GQA attention -- see
        // `attn_ids`/`flash_attn_causal_gqa.wgsl`'s doc for the real
        // ERROR_OUT_OF_DEVICE_MEMORY this closes.
        ("flash_attn_causal_gqa", kernels::FLASH_ATTN_CAUSAL_GQA), // 20
        // Coalesced RMSNorm -- the throughput twin of index 0, selected by
        // `block::rms_variant` inside `block::rmsnorm_fwd`.
        ("rmsnorm_rows", kernels::RMSNORM_ROWS),             // 21
    ]
}

fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: 0,
        rms_inv: block::UNREGISTERED,
        rmsnorm_dx: block::UNREGISTERED,
        rmsnorm_dx_rows: block::UNREGISTERED,
        rmsnorm_dw: block::UNREGISTERED,
        // This model rotates through `rope2d` (`GqaAttnIds::rope2d`), not
        // `block::rope_fwd`, and has no backward here. `0` is `rmsnorm` - a
        // live kernel - so these were misroutes waiting to happen, not
        // placeholders; `UNREGISTERED` is out of range and fails loudly.
        rope: block::UNREGISTERED,
        rope_bwd: block::UNREGISTERED,
        gqa_scores: 2,
        gqa_apply: 4,
        attn_softmax: 3,
        gqa_dscores: block::UNREGISTERED,
        gqa_dv: block::UNREGISTERED,
        gqa_dq: block::UNREGISTERED,
        gqa_dk: block::UNREGISTERED,
        silu_mul: 9,
        silu_da: block::UNREGISTERED,
        silu_db: block::UNREGISTERED,
        rmsnorm_rows: RMSNORM_ROWS,
    }
}

fn moe_ids() -> MoeIds {
    MoeIds { router_gate: 7, linear_gated: 8, silu_mul: 9, scale_add: 10 }
}

fn moe_ids8() -> MoeIds8 {
    MoeIds8 { linear_gated_i8: 18, silu_mul: 9, scale_add: 10, quant: [16, 17] }
}

fn decode_ids() -> GqaDecodeIds {
    GqaDecodeIds { kv_append: KV_APPEND, attn_decode_scores: 13, decode_softmax: 14, attn_decode_apply: 15 }
}

/// The hoisted attention sublayer's kernel indices, resolved against
/// [`thinker_pipelines`]'s ordering - see `model::block::GqaAttnIds`.
fn attn_ids() -> GqaAttnIds {
    // flash_causal_gqa: Some(20) -- the Thinker prefills a real agent's whole
    // conversation (system prompt + tool schemas can run thousands of
    // tokens), where gqa_fwd's O(T*T) materialized scores/probs is the real
    // ERROR_OUT_OF_DEVICE_MEMORY source; see flash_attn_causal_gqa.wgsl's doc.
    GqaAttnIds { kernels: kernel_ids(), matmul: MATMUL, add2: ADD2, rope2d: ROPE2D, kv_append: KV_APPEND, decode: decode_ids(), flash_causal_gqa: Some(20) }
}

fn attn_dims(cfg: &MoeTextConfig) -> GqaAttnDims {
    GqaAttnDims { hidden: cfg.hidden, head_dim: cfg.head_dim, n_heads: cfg.n_heads, n_kv_heads: cfg.n_kv_heads, use_qk_norm: cfg.use_qk_norm }
}

fn attn_weights<'a>(w: &ThinkerLayerWeights<'a>) -> GqaAttnWeights<'a> {
    GqaAttnWeights { ln1: w.ln1, wq: w.wq, wk: w.wk, wv: w.wv, wo: w.wo, q_norm: w.q_norm, k_norm: w.k_norm }
}

const MATMUL: usize = 5;
const ADD2: usize = 6;
const ROPE2D: usize = 1;
/// `model::vlm::splice_fwd`'s kernel index - exposed for a caller assembling
/// a multimodal embedding sequence before [`decode`]; see the module doc.
pub const SPLICE: usize = 11;
const KV_APPEND: usize = 12;
// Coalesced RMSNorm - the throughput twin of index 0, selected by
// `block::rms_variant` inside `block::rmsnorm_fwd`.
const RMSNORM_ROWS: usize = 21;

/// One decoder layer's weights, keyed exactly as they arrive from
/// `qwen3omnimoe::import` (`thinker.blocks.{l}.*`, prefix already stripped by the
/// caller - see [`ThinkerLayer::new`]'s doc).
pub struct ThinkerLayerWeights<'a> {
    pub ln1: &'a DeviceBuffer,
    pub wq: &'a DeviceBuffer,
    pub wk: &'a DeviceBuffer,
    pub wv: &'a DeviceBuffer,
    pub wo: &'a DeviceBuffer,
    pub q_norm: &'a DeviceBuffer,
    pub k_norm: &'a DeviceBuffer,
    pub ln2: &'a DeviceBuffer,
    pub router: &'a DeviceBuffer,
    /// Expert `e`'s `(gate.weight, up.weight, down.weight)`, indexed
    /// `0..n_experts`.
    pub experts: &'a [(DeviceBuffer, DeviceBuffer, DeviceBuffer)],
}

/// A layer's persistent incremental-decode KV cache: `[cap, n_kv_heads*
/// head_dim]` buffers the caller (`crate::generate`) sizes once for the whole
/// generation (`cap` = prompt length + max new tokens) and owns across every
/// call - [`layer_fwd`] only ever WRITES into these (a bulk prefill fill, when
/// given `Some`), never reads them back; [`layer_decode_step`] does both.
pub struct ThinkerLayerCache<'a> {
    pub kcache: &'a DeviceBuffer,
    pub vcache: &'a DeviceBuffer,
}

/// Runs one Thinker decoder layer forward: `x [n, d] -> [n, d]` (`out`),
/// alongside three intermediates a parity test needs to localize a
/// divergence to a specific stage: the router's raw logits `[n, n_experts]`,
/// the post-attention/pre-MoE residual (`xmid`), and the dense post-topk-
/// renorm gate `[n, n_experts]` `model::moe`'s expert loop actually consumes.
/// `n` is the sequence length (batch folded in, matching every other
/// forward-only harness in this engine). `cos`/`sin` are the `[n, head_dim/2]`
/// M-RoPE tables (`qwen3vl::mrope::mrope_tables`) - see the module doc for why
/// there is one RoPE path, not a separate "plain" and "M-RoPE" one.
///
/// `cache`, when `Some`, bulk-fills this layer's persistent KV cache with the
/// `n` positions' post-RoPE key/value rows (`model::block::kv_cache_fill`) as
/// a side effect - the prefill half of [`crate::generate`]'s KV-cache decode
/// loop, letting [`layer_decode_step`] continue attending from `pos = n`
/// onward without recomputing anything this call already did. Purely
/// additive: `out`/`router_logits`/`xmid`/`gate` are identical whether `cache`
/// is `Some` or `None`.
///
/// Note for callers comparing `out` against a full-model reference dump:
/// `out` is this ONE layer's raw output. A real decoder stack's top-level
/// `model.norm` (applied once, after every layer) is not part of this
/// function and must be applied separately if the comparison target is a
/// `last_hidden_state`-shaped tensor (see `thinker_layer_parity.rs`'s module
/// doc for the real bug this distinction caught; [`decode`] applies it).
///
/// `int8_experts`: see [`moe_sublayer`]'s doc - `Some` swaps only the
/// routed-expert dispatch to int8; `None` (every caller before this
/// parameter existed) is the original fp32 path, unchanged.
#[allow(clippy::too_many_arguments)]
pub fn layer_fwd(g: &Gpu, cfg: &MoeTextConfig, w: &ThinkerLayerWeights, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, n: u32, cache: Option<&ThinkerLayerCache>, int8_experts: Option<&ThinkerLayerExperts8>) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let xmid = gqa_attn_sublayer_fwd(g, &attn_ids(), &attn_dims(cfg), &attn_weights(w), x, cos, sin, n, cache.map(|c| (c.kcache, c.vcache)));
    let (out, router_logits, gate) = moe_sublayer(g, cfg, w, &xmid, n, int8_experts);
    (out, router_logits, xmid, gate)
}

/// The MoE FFN sublayer shared by [`layer_fwd`] (full batched forward) and
/// [`layer_decode_step`] (single-token KV-cache decode) - the two attention
/// shapes differ, but the post-attention residual `xmid [n, d] -> out [n, d]`
/// math is identical either way (router -> top-k renorm gate -> per-expert
/// SwiGLU -> residual add), so it lives once here instead of twice.
///
/// `int8_experts`: `Some(store)` dispatches every routed expert through
/// [`model::moe::expert_fwd_i8`] against `store`'s resident packed weights
/// (`crate::int8_resident::ThinkerInt8Store`) instead of `w.experts`' fp32
/// ones - `w.experts` is simply UNUSED in that branch (attention/router/
/// norms still come from `w`, unchanged; only the expert linears are
/// swapped). `None` (every caller before this parameter existed) is the
/// original fp32 path, bit-for-bit unchanged.
fn moe_sublayer(g: &Gpu, cfg: &MoeTextConfig, w: &ThinkerLayerWeights, xmid: &DeviceBuffer, n: u32, int8_experts: Option<&ThinkerLayerExperts8>) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let ids = kernel_ids();
    let mids = moe_ids();
    let d = cfg.hidden;

    let xn2 = g.storage((n * d) as u64);
    let router_logits = g.storage((n * cfg.n_experts) as u64);

    // ONE accumulated batch, ONE submit -- was one submit PER EXPERT (128 at
    // Thinker's scale, 6144 across 48 layers per token), each a real
    // encode+queue-submit+pipeline-barrier round trip. `expert_fwd`'s own
    // 5-step sequences already rely on multi-step ordering being preserved
    // within a single submit (the same guarantee `forward_steps()` batches an
    // entire model's forward into one submit on) -- accumulating every
    // expert's steps here is that same, already-relied-on contract, not a
    // new one. This is what `omni_bench`'s own module doc names as the
    // reason it cannot use `gpu_core::profile` at all ("submits eagerly...
    // so there is no single Step list to hand that profiler") -- fixing it
    // here unblocks that profiler for every subsequent optimisation pass on
    // this model, not just the submit-count win itself.
    let mut steps = vec![
        rmsnorm_fwd(g, &ids, xmid, w.ln2, &xn2, d, n),
        g.step(MATMUL, &[&xn2, w.router, &router_logits], &[n, d, cfg.n_experts], n * cfg.n_experts),
    ];

    let shape = cfg.moe_shape(n);
    let gate = g.storage((n * cfg.n_experts) as u64);
    steps.push(router_fwd(g, &mids, &shape, &router_logits, &gate, true, 1.0));

    let moe_ff = cfg.moe_intermediate;
    let moe_out = g.storage((n * d) as u64);
    match int8_experts {
        None => {
            let scratch = ExpertScratch {
                gate_pre: &g.storage((n * moe_ff) as u64),
                up: &g.storage((n * moe_ff) as u64),
                h: &g.storage((n * moe_ff) as u64),
                expert_out: &g.storage((n * d) as u64),
            };
            for (e, (gw, uw, dw)) in w.experts.iter().enumerate() {
                steps.extend(expert_fwd(g, &mids, &shape, &xn2, &gate, gw, uw, dw, &scratch, &moe_out, e as u32, e != 0));
            }
        }
        Some(store) => {
            let mids8 = moe_ids8();
            // xn2 quantized ONCE, shared by every expert -- expert_fwd_i8's
            // own doc: "every expert reads the same quantized activation, so
            // quantizing it 128 times would be pure waste."
            let xq = g.storage((n * d / 4) as u64);
            let sx = g.storage(n as u64);
            steps.extend(quant_rows_steps(g, QuantRows { kernels: mids8.quant, x: &xn2, sx: &sx, xq: &xq }, 0, n, d));
            let scratch8 = ExpertScratch8 {
                gate_pre: &g.storage((n * moe_ff) as u64),
                up: &g.storage((n * moe_ff) as u64),
                h: &g.storage((n * moe_ff) as u64),
                hq: &g.storage((n * moe_ff / 4) as u64),
                sh: &g.storage(n as u64),
                expert_out: &g.storage((n * d) as u64),
            };
            for e in 0..cfg.n_experts as usize {
                let (gw, uw, dw) = store.lin8_at(e);
                steps.extend(expert_fwd_i8(g, &mids8, &shape, &xq, &sx, &gate, gw, uw, dw, &scratch8, &moe_out, e as u32, e != 0));
            }
        }
    }

    let out = g.storage((n * d) as u64);
    steps.push(g.step(ADD2, &[xmid, &moe_out, &out], &[n * d], n * d));
    g.submit(&[], &steps);

    (out, router_logits, gate)
}

/// One incremental KV-cache decode step: a SINGLE new token's embedding row
/// `x [1, d]` through this layer, attending against `cache`'s `pos+1` valid
/// positions (`model::block::gqa_decode_step`) instead of recomputing full
/// causal attention over a growing sequence - the O(cached length), not O(T²),
/// twin of [`layer_fwd`]. `cos`/`sin` are the `[1, head_dim/2]` M-RoPE table
/// for this ONE token's absolute 3-axis position (`qwen3vl::mrope::mrope_tables`
/// called with a single-element `positions` slice) - `rope2d_fwd`'s table-driven
/// kernel needs no separate "decode" variant, unlike `qwen3::Qwen`'s `ROPE_AT`
/// (Thinker's RoPE
/// path was already row-driven, so a 1-row table IS the decode case). `cap` is
/// the cache's allocated capacity (must match what [`layer_fwd`]'s prefill
/// call and every prior decode step against this same cache used).
///
/// Returns this token's new hidden row `[1, d]`. The MoE tail is
/// [`moe_sublayer`], shared unchanged with [`layer_fwd`]. `int8_experts`:
/// see [`moe_sublayer`]'s doc.
#[allow(clippy::too_many_arguments)]
pub fn layer_decode_step(g: &Gpu, cfg: &MoeTextConfig, w: &ThinkerLayerWeights, cache: &ThinkerLayerCache, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, pos: u32, cap: u32, int8_experts: Option<&ThinkerLayerExperts8>) -> DeviceBuffer {
    let xmid = gqa_attn_sublayer_decode_step(g, &attn_ids(), &attn_dims(cfg), &attn_weights(w), (cache.kcache, cache.vcache), x, cos, sin, pos, cap);
    let (out, ..) = moe_sublayer(g, cfg, w, &xmid, 1, int8_experts);
    out
}

/// All 48 decoder layers plus the top-level final RMSNorm
/// (`thinker.model.norm.weight`) - the weights [`decode`] needs beyond one
/// layer's own [`ThinkerLayerWeights`].
pub struct ThinkerWeights<'a> {
    pub layers: &'a [ThinkerLayerWeights<'a>],
    pub final_norm: &'a DeviceBuffer,
}

/// Runs the full Thinker text-decoder stack: `x [n, d] -> [n, d]`, one
/// [`layer_fwd`] per entry in `w.layers` chained residual-to-residual, then
/// `model.norm` - the piece [`layer_fwd`] deliberately leaves out (see its
/// doc). `x` is an already-assembled embedding sequence: plain token
/// embeddings for pure text, or the same buffer after a caller has spliced in
/// image/audio embeddings via `model::vlm::splice_fwd` at the appropriate
/// rows (`SPLICE`, in [`thinker_pipelines`]) - this function has no opinion
/// on how `x` was built, only on what to do with it. `cos`/`sin` are the
/// M-RoPE tables for this same sequence (`qwen3vl::mrope`), shared unchanged
/// across every layer (position doesn't change with depth).
pub fn decode(g: &Gpu, cfg: &MoeTextConfig, w: &ThinkerWeights, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let mut h = x.clone(); // DeviceBuffer is Arc-backed; cheap, aliases the same buffer.
    for layer in w.layers {
        let (out, ..) = layer_fwd(g, cfg, layer, &h, cos, sin, n, None, None);
        h = out;
    }
    final_norm(g, cfg, w.final_norm, &h, n)
}

/// The top-level final RMSNorm (`thinker.model.norm.weight`) [`layer_fwd`]
/// deliberately leaves out - factored out of [`decode`] so a caller that
/// streams layer weights one at a time instead of holding `[ThinkerLayerWeights]`
/// resident (`crate::generate`, for a real-weight generation loop too large to
/// keep GPU-resident all at once) can apply it without re-deriving it.
pub fn final_norm(g: &Gpu, cfg: &MoeTextConfig, norm_w: &DeviceBuffer, h: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let ids = kernel_ids();
    let normed = g.storage((n * cfg.hidden) as u64);
    g.submit(&[], &[rmsnorm_fwd(g, &ids, h, norm_w, &normed, cfg.hidden, n)]);
    normed
}

/// `hidden [n, d] -> logits [n, vocab]` via `thinker.lm_head.weight`
/// (`[vocab, d]`, untied from the embedding table -
/// `MoeTextConfig`'s real source config has `tie_word_embeddings: false`).
pub fn lm_head_fwd(g: &Gpu, lm_head_w: &DeviceBuffer, hidden: &DeviceBuffer, n: u32, d: u32, vocab: u32) -> DeviceBuffer {
    let out = g.storage((n * vocab) as u64);
    g.submit(&[], &[g.step(MATMUL, &[hidden, lm_head_w, &out], &[n, d, vocab], n * vocab)]);
    out
}

/// Kernel indices [`lm_head_fwd_i8`] dispatches - `matmul_i8` is a general
/// int8 GEMM (`matmul_i8_dyn.wgsl`, the SAME kernel [`model::moe::
/// shared_expert_fwd_i8`]'s own dense linears already use), not an MoE-only
/// primitive, so this is a fresh (non-MoE) int8 dispatch surface rather than
/// a reuse of [`moe_ids8`].
pub struct LmHeadIds8 {
    pub matmul_i8: usize,
    /// `crate::int8::quant_rows_steps`'s `[max_abs_row, quant_pack]` pair.
    pub quant: [usize; 2],
}

/// int8 counterpart of [`lm_head_fwd`]: `hidden` is quantized once (a single
/// `[n, d]` activation, not reused across N experts, so there is no "quantize
/// once, share across readers" concern the MoE path has), `lm_head_w` is
/// ALREADY a packed [`model::moe::Lin8`] view (the checkpoint's real
/// `lm_head.weight` is quantized on disk by `qwen3omnimoe::import::should_quantize`
/// like every other rank-2 `k%32==0` weight - a caller loads it via
/// `crate::int8_resident`'s `load_lin8` rather than dequantizing, unlike
/// [`crate::int8_thinker_resident::load_mat`]'s current always-dequantize
/// path). vocab is not required to be a multiple of 32 for the OUTPUT side
/// (only `d`, the K dimension, needs to be - the packing constraint is on
/// the CONTRACTED dimension, matching every other int8 GEMM in this crate).
pub fn lm_head_fwd_i8(g: &Gpu, ids: &LmHeadIds8, lm_head_w: model::moe::Lin8, hidden: &DeviceBuffer, n: u32, d: u32, vocab: u32) -> DeviceBuffer {
    let xq = g.storage((n * d / 4) as u64);
    let sx = g.storage(n as u64);
    let mut steps = quant_rows_steps(g, QuantRows { kernels: ids.quant, x: hidden, sx: &sx, xq: &xq }, 0, n, d).to_vec();
    let out = g.storage((n * vocab) as u64);
    steps.push(g.step(ids.matmul_i8, &[&xq, lm_head_w.wq, &sx, lm_head_w.sw, &out], &[n, d / 4, vocab], n.div_ceil(128) * vocab.div_ceil(128) * 256));
    g.submit(&[], &steps);
    out
}

/// The coalesced RMSNorm this model now selects (`rmsnorm_rows`, via
/// `block::rms_variant` inside `block::rmsnorm_fwd`) is NOT bit-identical to
/// the per-element `rmsnorm` it replaced: 64 partial sums fold in a different
/// order. It was adopted for throughput, so what it computes is gated here,
/// against a HOST reference, at the shapes THIS model's decode tape really
/// dispatches - narrow rows are where the two reduction orders differ most,
/// and they are also the whole reason the swap is worth making.
#[cfg(test)]
mod rmsnorm_variant_agreement {
    use super::*;

    /// The slot really names the coalesced kernel. A registration this model
    /// gets wrong by one index does not fail - it silently dispatches a
    /// DIFFERENT kernel through the RMSNorm bindings.
    #[test]
    fn the_registered_slot_names_the_coalesced_kernel() {
        assert_eq!(thinker_pipelines()[kernel_ids().rmsnorm_rows].0, "rmsnorm_rows");
    }

    #[test]
    fn the_decode_tape_norms_match_the_host_reference() {
        // `layer_decode_step` runs `block::gqa_attn_qkv` (ln1 + the two
        // QK-norms) and `moe_sublayer` (ln2) at `n = 1`, then one final norm -
        // and three of those four are norms this crate never dispatches
        // itself, they live inside the shared builder.
        let c = crate::config::MoeTextConfig::thinker_defaults();
        let shapes = [
            (1, c.hidden, "ln1/ln2/final norm at decode"),
            (c.n_heads, c.head_dim, "q_norm at decode"),
            (c.n_kv_heads, c.head_dim, "k_norm at decode"),
        ];
        let gpu = gpu_core::testgpu::dev(thinker_pipelines());
        block::assert_rmsnorm_variant_agrees(&gpu, &kernel_ids(), &shapes);
    }
}
