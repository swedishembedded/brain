// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni's Talker text decoder: the same GQA+QK-norm+M-RoPE attention
//! stack as [`crate::thinker`] (Talker's decoder layer reuses
//! `Qwen3OmniMoeThinkerTextAttention` verbatim in the reference - see
//! `MoeTextConfig::talker_defaults`'s doc comment) over a sparse top-k MoE
//! FFN **plus an always-active shared expert**
//! (`model::moe::shared_expert_fwd`) - the one architectural difference from
//! Thinker's MoE block. `Qwen3OmniMoeTalkerTextSparseMoeBlock.forward`:
//! `expert_output + sigmoid(shared_expert_gate(x)) * shared_expert(x)`.
//!
//! Not built by converting `qwen3tts::talker`'s existing `TalkerModel` (the
//! Qwen3-TTS dense Talker, wrapping [`qwen3::Qwen`] with `tie_embeddings =
//! false`) to MoE: that model is a genuinely different, already-shipping
//! architecture, and `qwen3::Qwen` has the same missing MoE seam
//! `crate::thinker`'s module doc explains for the Thinker case. This module
//! is [`crate::thinker`]'s sibling, composed from the same
//! `model::block`/`model::moe` primitives, for the same "validate against
//! real weights now, give `qwen3::Qwen` the seam later" reasoning.
//!
//! Not this module's concern (same seam contract as `crate::thinker`):
//! `accept_hidden_layer` (Talker consumes Thinker's hidden state at a given
//! layer, not just its own token embeddings), the codec-id sampling loop,
//! and the code predictor (`crate::code_predictor` /
//! `qwen3tts::mtp`) are a caller's job, layered on top of [`decode`].

use gpu_core::{DeviceBuffer, Gpu};
use model::block::{self, gqa_attn_sublayer_decode_step, gqa_attn_sublayer_fwd, rmsnorm_fwd, GqaAttnDims, GqaAttnIds, GqaAttnWeights, GqaDecodeIds, KernelIds};
use model::int8::{quant_rows_steps, QuantRows};
use model::moe::{
    expert_fwd, expert_fwd_i8, router_fwd, shared_expert_fwd, shared_expert_fwd_i8, ExpertScratch, ExpertScratch8, MoeIds, MoeIds8, SharedExpertIds, SharedExpertIds8, SharedExpertScratch,
    SharedExpertScratch8,
};

use crate::config::MoeTextConfig;
use crate::int8_resident::TalkerLayerExperts8;

/// Kernel pipeline this module dispatches. Forward-only, same convention as
/// `crate::thinker::thinker_pipelines`: unused backward slots point at index
/// 0 rather than a real backward kernel.
pub fn talker_pipelines() -> &'static [(&'static str, &'static str)] {
    &[
        ("rmsnorm", kernels::RMSNORM),                    // 0
        ("rope2d", kernels::ROPE2D),                       // 1
        ("gqa_scores", kernels::GQA_SCORES),               // 2
        ("attn_softmax", kernels::ATTN_SOFTMAX),           // 3
        ("gqa_apply", kernels::GQA_APPLY),                 // 4
        ("matmul", kernels::MATMUL),                       // 5
        ("add2", kernels::ADD2),                           // 6
        ("router_gate", kernels::ROUTER_GATE),             // 7
        ("moe_linear_gated", kernels::MOE_LINEAR_GATED),   // 8
        ("silu_mul", kernels::SILU_MUL),                   // 9
        ("scale_add", kernels::SCALE_ADD),                 // 10
        ("splice", kernels::SPLICE),                       // 11 -- see crate::thinker's module doc; same seam.
        ("sigmoid", kernels::SIGMOID),                     // 12 -- shared-expert gate.
        ("scale_row", kernels::SCALE_ROW),                 // 13 -- shared-expert gate scale.
        ("kv_append", kernels::KV_APPEND),                   // 14 -- KV-cache decode; see thinker::layer_decode_step's doc.
        ("attn_decode_scores", kernels::ATTN_DECODE_SCORES), // 15
        ("decode_softmax", kernels::DECODE_SOFTMAX),         // 16
        ("attn_decode_apply", kernels::ATTN_DECODE_APPLY),   // 17
        // int8 MoE dispatch (routed experts + shared expert) -- see
        // `moe_sublayer`'s `int8_experts` parameter and `crate::int8_resident`.
        // Registered unconditionally, same rationale as `crate::thinker`'s
        // identical trailing block.
        ("max_abs_row", kernels::MAX_ABS_ROW),                 // 18
        ("quant_pack", kernels::QUANT_PACK),                   // 19
        ("moe_linear_gated_i8", kernels::MOE_LINEAR_GATED_I8), // 20
        ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),             // 21 -- shared expert's own dense linears.
    ]
}

fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: 0,
        rms_inv: block::UNREGISTERED,
        rmsnorm_dx: block::UNREGISTERED,
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
        rmsnorm_rows: block::UNREGISTERED,
    }
}

fn moe_ids() -> MoeIds {
    MoeIds { router_gate: 7, linear_gated: 8, silu_mul: 9, scale_add: 10 }
}

fn moe_ids8() -> MoeIds8 {
    MoeIds8 { linear_gated_i8: 20, silu_mul: 9, scale_add: 10, quant: [18, 19] }
}

fn shared_expert_ids() -> SharedExpertIds {
    SharedExpertIds { matmul: MATMUL, silu_mul: 9, sigmoid: SIGMOID, scale_row: SCALE_ROW, add2: ADD2 }
}

fn shared_expert_ids8() -> SharedExpertIds8 {
    SharedExpertIds8 { matmul_i8: MATMUL_I8_DYN, matmul: MATMUL, silu_mul: 9, sigmoid: SIGMOID, scale_row: SCALE_ROW, add2: ADD2, quant: [18, 19] }
}

fn decode_ids() -> GqaDecodeIds {
    GqaDecodeIds { kv_append: KV_APPEND, attn_decode_scores: 15, decode_softmax: 16, attn_decode_apply: 17 }
}

/// The hoisted attention sublayer's kernel indices, resolved against
/// [`talker_pipelines`]'s ordering - see `model::block::GqaAttnIds`.
fn attn_ids() -> GqaAttnIds {
    // flash_causal_gqa: None -- the Talker's own sequence lengths are small
    // (bounded assistant-turn generation, not a long system prompt + tool
    // schemas), so the O(T*T) path this leaves unchanged is not the real
    // problem `flash_attn_causal_gqa.wgsl` closes for the Thinker.
    GqaAttnIds { kernels: kernel_ids(), matmul: MATMUL, add2: ADD2, rope2d: ROPE2D, kv_append: KV_APPEND, decode: decode_ids(), flash_causal_gqa: None }
}

fn attn_dims(cfg: &MoeTextConfig) -> GqaAttnDims {
    GqaAttnDims { hidden: cfg.hidden, head_dim: cfg.head_dim, n_heads: cfg.n_heads, n_kv_heads: cfg.n_kv_heads, use_qk_norm: cfg.use_qk_norm }
}

fn attn_weights<'a>(w: &TalkerLayerWeights<'a>) -> GqaAttnWeights<'a> {
    GqaAttnWeights { ln1: w.ln1, wq: w.wq, wk: w.wk, wv: w.wv, wo: w.wo, q_norm: w.q_norm, k_norm: w.k_norm }
}

const MATMUL: usize = 5;
const ADD2: usize = 6;
const ROPE2D: usize = 1;
const SIGMOID: usize = 12;
const SCALE_ROW: usize = 13;
const KV_APPEND: usize = 14;
const MATMUL_I8_DYN: usize = 21;
/// `model::vlm::splice_fwd`'s kernel index - see `crate::thinker`'s module doc.
pub const SPLICE: usize = 11;

/// One decoder layer's weights, keyed exactly as they arrive from
/// `qwen3omnimoe::import` (`talker.blocks.{l}.*`, prefix stripped by the caller) --
/// same shape as [`crate::thinker::ThinkerLayerWeights`] plus the shared
/// expert's own MLP + sigmoid-gate weight.
pub struct TalkerLayerWeights<'a> {
    pub ln1: &'a DeviceBuffer,
    pub wq: &'a DeviceBuffer,
    pub wk: &'a DeviceBuffer,
    pub wv: &'a DeviceBuffer,
    pub wo: &'a DeviceBuffer,
    pub q_norm: &'a DeviceBuffer,
    pub k_norm: &'a DeviceBuffer,
    pub ln2: &'a DeviceBuffer,
    pub router: &'a DeviceBuffer,
    /// Expert `e`'s `(gate.weight, up.weight, down.weight)`, indexed `0..n_experts`.
    pub experts: &'a [(DeviceBuffer, DeviceBuffer, DeviceBuffer)],
    /// The always-active shared expert's own `(gate.weight, up.weight, down.weight)`.
    pub shared_expert: (&'a DeviceBuffer, &'a DeviceBuffer, &'a DeviceBuffer),
    /// `mlp.shared_expert_gate.weight`, `[1, hidden]` -- the per-token sigmoid gate.
    pub shared_expert_gate: &'a DeviceBuffer,
}

/// A layer's persistent incremental-decode KV cache -- same shape/contract
/// as `crate::thinker::ThinkerLayerCache` (which see).
pub struct TalkerLayerCache<'a> {
    pub kcache: &'a DeviceBuffer,
    pub vcache: &'a DeviceBuffer,
}

/// Runs one Talker decoder layer forward: `x [n, d] -> [n, d]` (`out`), plus
/// the same three diagnostic intermediates
/// [`crate::thinker::layer_fwd`] returns (router logits, `xmid`, the dense
/// post-topk-renorm gate) -- everything through the routed-expert combine is
/// identical to Thinker's; only the final combine step differs (adds the
/// shared expert). `cos`/`sin` are the `[n, head_dim/2]` M-RoPE tables
/// (`qwen3vl::mrope::mrope_tables`), same contract as Thinker's. `cache`, when
/// `Some`, bulk-fills this layer's persistent KV cache -- see
/// `crate::thinker::layer_fwd`'s doc for why this is additive and cheap.
/// `int8_experts`: see [`moe_sublayer`]'s doc -- `Some` swaps the routed AND
/// shared expert dispatch to int8; `None` (every caller before this
/// parameter existed) is the original fp32 path, unchanged.
#[allow(clippy::too_many_arguments)]
pub fn layer_fwd(g: &Gpu, cfg: &MoeTextConfig, w: &TalkerLayerWeights, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, n: u32, cache: Option<&TalkerLayerCache>, int8_experts: Option<&TalkerLayerExperts8>) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let xmid = gqa_attn_sublayer_fwd(g, &attn_ids(), &attn_dims(cfg), &attn_weights(w), x, cos, sin, n, cache.map(|c| (c.kcache, c.vcache)));
    let (out, router_logits, gate) = moe_sublayer(g, cfg, w, &xmid, n, int8_experts);
    (out, router_logits, xmid, gate)
}

/// The MoE FFN sublayer (routed experts + always-active shared expert) shared
/// by [`layer_fwd`] and [`layer_decode_step`] -- see `crate::thinker::
/// moe_sublayer`'s doc for why this is factored out once instead of twice.
///
/// `int8_experts`: `Some(store)` dispatches every routed expert through
/// [`expert_fwd_i8`] AND the shared expert through [`shared_expert_fwd_i8`]
/// against `store`'s resident packed weights
/// (`crate::int8_resident::TalkerInt8Store`) instead of `w`'s fp32 ones --
/// `w.experts`/`w.shared_expert`/`w.shared_expert_gate` are simply UNUSED in
/// that branch (attention/router/norms still come from `w`). The routed and
/// shared experts share ONE quantization of `xn2` (`xq`/`sx`) -- the same
/// "quantize once, every reader shares it" discipline Thinker's own int8
/// branch already relies on. `None` (every caller before this parameter
/// existed) is the original fp32 path, bit-for-bit unchanged.
fn moe_sublayer(g: &Gpu, cfg: &MoeTextConfig, w: &TalkerLayerWeights, xmid: &DeviceBuffer, n: u32, int8_experts: Option<&TalkerLayerExperts8>) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let ids = kernel_ids();
    let mids = moe_ids();
    let d = cfg.hidden;

    let xn2 = g.storage((n * d) as u64);
    let router_logits = g.storage((n * cfg.n_experts) as u64);

    // ONE accumulated batch, ONE submit -- the same fix Thinker's
    // moe_sublayer got (ae72d8f: one submit PER EXPERT was 128 real
    // encode+queue-submit+pipeline-barrier round trips per layer; at
    // Talker's scale ~2,560 per codec token across 20 layers). This copy
    // MISSED that fix -- exactly the re-divergence risk that motivated
    // hoisting the attention sublayer into model::block; the MoE tail stays
    // model-local only because the shared expert genuinely differs.
    let mut steps = vec![
        rmsnorm_fwd(g, &ids, xmid, w.ln2, &xn2, d, n),
        g.step(MATMUL, &[&xn2, w.router, &router_logits], &[n, d, cfg.n_experts], n * cfg.n_experts),
    ];

    let shape = cfg.moe_shape(n);
    let gate = g.storage((n * cfg.n_experts) as u64);
    steps.push(router_fwd(g, &mids, &shape, &router_logits, &gate, true, 1.0));

    let moe_ff = cfg.moe_intermediate;
    let se_ff = cfg.shared_expert_intermediate;
    let moe_out = g.storage((n * d) as u64);
    match int8_experts {
        None => {
            let scratch = ExpertScratch {
                gate_pre: &g.storage((n * moe_ff) as u64),
                up: &g.storage((n * moe_ff) as u64),
                h: &g.storage((n * moe_ff) as u64),
                expert_out: &g.storage((n * d) as u64),
            };
            let routed_out = g.storage((n * d) as u64);
            for (e, (gw, uw, dw)) in w.experts.iter().enumerate() {
                steps.extend(expert_fwd(g, &mids, &shape, &xn2, &gate, gw, uw, dw, &scratch, &routed_out, e as u32, e != 0));
            }

            // Shared expert: always active (no gating), reads the SAME xn2 the
            // routed experts read, added to routed_out via a fresh buffer (never
            // in place -- see shared_expert_fwd's doc).
            let se_ids = shared_expert_ids();
            let se_scratch = SharedExpertScratch {
                gate_pre: &g.storage((n * se_ff) as u64),
                up: &g.storage((n * se_ff) as u64),
                h: &g.storage((n * se_ff) as u64),
                mlp_out: &g.storage((n * d) as u64),
                gate_logits: &g.storage(n as u64),
                gate_scalar: &g.storage(n as u64),
                scaled: &g.storage((n * d) as u64),
            };
            let (sgw, suw, sdw) = w.shared_expert;
            steps.extend(shared_expert_fwd(g, &se_ids, n, d, se_ff, &xn2, sgw, suw, sdw, Some(w.shared_expert_gate), &se_scratch, &routed_out, &moe_out));
        }
        Some(store) => {
            let mids8 = moe_ids8();
            // xn2 quantized ONCE, shared by every routed expert AND the
            // shared expert -- see this function's doc.
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
            let routed_out = g.storage((n * d) as u64);
            for e in 0..cfg.n_experts as usize {
                let (gw, uw, dw) = store.lin8_at(e);
                steps.extend(expert_fwd_i8(g, &mids8, &shape, &xq, &sx, &gate, gw, uw, dw, &scratch8, &routed_out, e as u32, e != 0));
            }

            let se_ids8 = shared_expert_ids8();
            let se_scratch8 = SharedExpertScratch8 {
                gate_pre: &g.storage((n * se_ff) as u64),
                up: &g.storage((n * se_ff) as u64),
                h: &g.storage((n * se_ff) as u64),
                hq: &g.storage((n * se_ff / 4) as u64),
                sh: &g.storage(n as u64),
                mlp_out: &g.storage((n * d) as u64),
                gate_logits: &g.storage(n as u64),
                gate_scalar: &g.storage(n as u64),
                scaled: &g.storage((n * d) as u64),
            };
            let (sgw, suw, sdw) = store.shared_lin8();
            steps.extend(shared_expert_fwd_i8(g, &se_ids8, n, d, se_ff, &xq, &sx, &xn2, sgw, suw, sdw, Some(&store.shared_expert_gate), &se_scratch8, &routed_out, &moe_out));
        }
    }

    let out = g.storage((n * d) as u64);
    steps.push(g.step(ADD2, &[xmid, &moe_out, &out], &[n * d], n * d));
    g.submit(&[], &steps);

    (out, router_logits, gate)
}

/// One incremental KV-cache decode step -- the O(cached length) twin of
/// [`layer_fwd`], same contract as `crate::thinker::layer_decode_step`
/// (which see for the full doc: `cos`/`sin` are a 1-row M-RoPE table for
/// this token's absolute position, `cap` is the cache's allocated capacity).
/// Returns this token's new hidden row `[1, d]`. `int8_experts`: see
/// [`moe_sublayer`]'s doc.
#[allow(clippy::too_many_arguments)]
pub fn layer_decode_step(g: &Gpu, cfg: &MoeTextConfig, w: &TalkerLayerWeights, cache: &TalkerLayerCache, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, pos: u32, cap: u32, int8_experts: Option<&TalkerLayerExperts8>) -> DeviceBuffer {
    let xmid = gqa_attn_sublayer_decode_step(g, &attn_ids(), &attn_dims(cfg), &attn_weights(w), (cache.kcache, cache.vcache), x, cos, sin, pos, cap);
    let (out, ..) = moe_sublayer(g, cfg, w, &xmid, 1, int8_experts);
    out
}

/// All decoder layers plus the top-level final RMSNorm -- same composition
/// contract as [`crate::thinker::decode`] (which see for the multimodal
/// splice and `x`-assembly seam, identical here).
pub struct TalkerWeights<'a> {
    pub layers: &'a [TalkerLayerWeights<'a>],
    pub final_norm: &'a DeviceBuffer,
}

/// Runs the full Talker text-decoder stack: `x [n, d] -> [n, d]`, one
/// [`layer_fwd`] per entry in `w.layers` chained residual-to-residual, then
/// `model.norm`. See [`crate::thinker::decode`]'s doc -- identical contract.
pub fn decode(g: &Gpu, cfg: &MoeTextConfig, w: &TalkerWeights, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let mut h = x.clone();
    for layer in w.layers {
        let (out, ..) = layer_fwd(g, cfg, layer, &h, cos, sin, n, None, None);
        h = out;
    }
    let ids = kernel_ids();
    let normed = g.storage((n * cfg.hidden) as u64);
    g.submit(&[], &[rmsnorm_fwd(g, &ids, &h, w.final_norm, &normed, cfg.hidden, n)]);
    normed
}
