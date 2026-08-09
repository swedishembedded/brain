// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni's Talker text decoder: the same GQA+QK-norm+M-RoPE attention
//! stack as [`crate::thinker`] (Talker's decoder layer reuses
//! `Qwen3OmniMoeThinkerTextAttention` verbatim in the reference — see
//! `MoeTextConfig::talker_defaults`'s doc comment) over a sparse top-k MoE
//! FFN **plus an always-active shared expert**
//! (`model::moe::shared_expert_fwd`) — the one architectural difference from
//! Thinker's MoE block. `Qwen3OmniMoeTalkerTextSparseMoeBlock.forward`:
//! `expert_output + sigmoid(shared_expert_gate(x)) * shared_expert(x)`.
//!
//! Not built by converting `tts::talker`'s existing `TalkerModel` (the
//! Qwen3-TTS dense Talker, wrapping [`qwen::Qwen`] with `tie_embeddings =
//! false`) to MoE: that model is a genuinely different, already-shipping
//! architecture, and `qwen::Qwen` has the same missing MoE seam
//! `crate::thinker`'s module doc explains for the Thinker case. This module
//! is [`crate::thinker`]'s sibling, composed from the same
//! `model::block`/`model::moe` primitives, for the same "validate against
//! real weights now, give `qwen::Qwen` the seam later" reasoning.
//!
//! Not this module's concern (same seam contract as `crate::thinker`):
//! `accept_hidden_layer` (Talker consumes Thinker's hidden state at a given
//! layer, not just its own token embeddings), the codec-id sampling loop,
//! and the code predictor (`crate::code_predictor` /
//! `tts::mtp`) are a caller's job, layered on top of [`decode`].

use gpu_core::{DeviceBuffer, Gpu};
use model::block::{gqa_decode_step, gqa_fwd, kv_cache_fill, rmsnorm_fwd, rope2d_fwd, Gqa, GqaDecodeIds, KernelIds};
use model::moe::{expert_fwd, router_fwd, shared_expert_fwd, ExpertScratch, MoeIds, SharedExpertIds, SharedExpertScratch};

use crate::config::MoeTextConfig;

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
    ]
}

fn kernel_ids() -> KernelIds {
    KernelIds {
        rmsnorm: 0,
        rms_inv: 0,
        rmsnorm_dx: 0,
        rmsnorm_dw: 0,
        rope: 0,
        rope_bwd: 0,
        gqa_scores: 2,
        gqa_apply: 4,
        attn_softmax: 3,
        gqa_dscores: 0,
        gqa_dv: 0,
        gqa_dq: 0,
        gqa_dk: 0,
        silu_mul: 9,
        silu_da: 0,
        silu_db: 0,
    }
}

fn moe_ids() -> MoeIds {
    MoeIds { router_gate: 7, linear_gated: 8, silu_mul: 9, scale_add: 10 }
}

fn shared_expert_ids() -> SharedExpertIds {
    SharedExpertIds { matmul: MATMUL, silu_mul: 9, sigmoid: SIGMOID, scale_row: SCALE_ROW, add2: ADD2 }
}

fn decode_ids() -> GqaDecodeIds {
    GqaDecodeIds { kv_append: KV_APPEND, attn_decode_scores: 15, decode_softmax: 16, attn_decode_apply: 17 }
}

const MATMUL: usize = 5;
const ADD2: usize = 6;
const ROPE2D: usize = 1;
const SIGMOID: usize = 12;
const SCALE_ROW: usize = 13;
const KV_APPEND: usize = 14;
/// `model::vlm::splice_fwd`'s kernel index — see `crate::thinker`'s module doc.
pub const SPLICE: usize = 11;

/// One decoder layer's weights, keyed exactly as they arrive from
/// `omni::import` (`talker.blocks.{l}.*`, prefix stripped by the caller) --
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
/// (`qwenvl::mrope::mrope_tables`), same contract as Thinker's. `cache`, when
/// `Some`, bulk-fills this layer's persistent KV cache -- see
/// `crate::thinker::layer_fwd`'s doc for why this is additive and cheap.
#[allow(clippy::too_many_arguments)]
pub fn layer_fwd(g: &Gpu, cfg: &MoeTextConfig, w: &TalkerLayerWeights, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, n: u32, cache: Option<&TalkerLayerCache>) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let ids = kernel_ids();
    let (d, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let (hq, hkv) = (nh * hd, nkv * hd);

    let xn1 = g.storage((n * d) as u64);
    let mut steps = vec![rmsnorm_fwd(g, &ids, x, w.ln1, &xn1, d, n)];

    let q_pre = g.storage((n * hq) as u64);
    let k_pre = g.storage((n * hkv) as u64);
    let v = g.storage((n * hkv) as u64);
    steps.push(g.step(MATMUL, &[&xn1, w.wq, &q_pre], &[n, d, hq], n * hq));
    steps.push(g.step(MATMUL, &[&xn1, w.wk, &k_pre], &[n, d, hkv], n * hkv));
    steps.push(g.step(MATMUL, &[&xn1, w.wv, &v], &[n, d, hkv], n * hkv));

    let (q, k) = if cfg.use_qk_norm {
        let q = g.storage((n * hq) as u64);
        let k = g.storage((n * hkv) as u64);
        steps.push(rmsnorm_fwd(g, &ids, &q_pre, w.q_norm, &q, hd, n * nh));
        steps.push(rmsnorm_fwd(g, &ids, &k_pre, w.k_norm, &k, hd, n * nkv));
        (q, k)
    } else {
        (q_pre, k_pre)
    };
    steps.push(rope2d_fwd(g, ROPE2D, &q, cos, sin, n, nh, hd, hq));
    steps.push(rope2d_fwd(g, ROPE2D, &k, cos, sin, n, nkv, hd, hkv));
    g.submit(&[], &steps);

    if let Some(c) = cache {
        g.submit(&[], &[kv_cache_fill(g, KV_APPEND, &k, c.kcache, n, nkv, hd), kv_cache_fill(g, KV_APPEND, &v, c.vcache, n, nkv, hd)]);
    }

    let scores = g.storage((nh * n * n) as u64);
    let probs = g.storage((nh * n * n) as u64);
    let ctx = g.storage((n * hq) as u64);
    let ga = Gqa { b: 1, t: n, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
    g.submit(&[], &gqa_fwd(g, &ids, &ga, &q, &k, &v, &scores, &probs, &ctx));

    let proj = g.storage((n * d) as u64);
    let xmid = g.storage((n * d) as u64);
    g.submit(&[], &[g.step(MATMUL, &[&ctx, w.wo, &proj], &[n, hq, d], n * d)]);
    g.submit(&[], &[g.step(ADD2, &[x, &proj, &xmid], &[n * d], n * d)]);

    let (out, router_logits, gate) = moe_sublayer(g, cfg, w, &xmid, n);
    (out, router_logits, xmid, gate)
}

/// The MoE FFN sublayer (routed experts + always-active shared expert) shared
/// by [`layer_fwd`] and [`layer_decode_step`] -- see `crate::thinker::
/// moe_sublayer`'s doc for why this is factored out once instead of twice.
fn moe_sublayer(g: &Gpu, cfg: &MoeTextConfig, w: &TalkerLayerWeights, xmid: &DeviceBuffer, n: u32) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let ids = kernel_ids();
    let mids = moe_ids();
    let d = cfg.hidden;

    let xn2 = g.storage((n * d) as u64);
    let router_logits = g.storage((n * cfg.n_experts) as u64);
    g.submit(
        &[],
        &[
            rmsnorm_fwd(g, &ids, xmid, w.ln2, &xn2, d, n),
            g.step(MATMUL, &[&xn2, w.router, &router_logits], &[n, d, cfg.n_experts], n * cfg.n_experts),
        ],
    );

    let shape = cfg.moe_shape(n);
    let gate = g.storage((n * cfg.n_experts) as u64);
    g.submit(&[], &[router_fwd(g, &mids, &shape, &router_logits, &gate)]);

    let moe_ff = cfg.moe_intermediate;
    let scratch = ExpertScratch {
        gate_pre: &g.storage((n * moe_ff) as u64),
        up: &g.storage((n * moe_ff) as u64),
        h: &g.storage((n * moe_ff) as u64),
        expert_out: &g.storage((n * d) as u64),
    };
    let routed_out = g.storage((n * d) as u64);
    for (e, (gw, uw, dw)) in w.experts.iter().enumerate() {
        let steps = expert_fwd(g, &mids, &shape, &xn2, &gate, gw, uw, dw, &scratch, &routed_out, e as u32, e != 0);
        g.submit(&[], &steps);
    }

    // Shared expert: always active (no gating), reads the SAME xn2 the
    // routed experts read, added to routed_out via a fresh buffer (never
    // in place -- see shared_expert_fwd's doc).
    let se_ff = cfg.shared_expert_intermediate;
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
    let moe_out = g.storage((n * d) as u64);
    g.submit(
        &[],
        &shared_expert_fwd(g, &se_ids, n, d, se_ff, &xn2, sgw, suw, sdw, Some(w.shared_expert_gate), &se_scratch, &routed_out, &moe_out),
    );

    let out = g.storage((n * d) as u64);
    g.submit(&[], &[g.step(ADD2, &[xmid, &moe_out, &out], &[n * d], n * d)]);

    (out, router_logits, gate)
}

/// One incremental KV-cache decode step -- the O(cached length) twin of
/// [`layer_fwd`], same contract as `crate::thinker::layer_decode_step`
/// (which see for the full doc: `cos`/`sin` are a 1-row M-RoPE table for
/// this token's absolute position, `cap` is the cache's allocated capacity).
/// Returns this token's new hidden row `[1, d]`.
#[allow(clippy::too_many_arguments)]
pub fn layer_decode_step(g: &Gpu, cfg: &MoeTextConfig, w: &TalkerLayerWeights, cache: &TalkerLayerCache, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, pos: u32, cap: u32) -> DeviceBuffer {
    let ids = kernel_ids();
    let dids = decode_ids();
    let (d, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let (hq, hkv) = (nh * hd, nkv * hd);
    let n = 1u32;

    let xn1 = g.storage(d as u64);
    let mut steps = vec![rmsnorm_fwd(g, &ids, x, w.ln1, &xn1, d, n)];

    let q_pre = g.storage(hq as u64);
    let k_pre = g.storage(hkv as u64);
    let v = g.storage(hkv as u64);
    steps.push(g.step(MATMUL, &[&xn1, w.wq, &q_pre], &[n, d, hq], n * hq));
    steps.push(g.step(MATMUL, &[&xn1, w.wk, &k_pre], &[n, d, hkv], n * hkv));
    steps.push(g.step(MATMUL, &[&xn1, w.wv, &v], &[n, d, hkv], n * hkv));

    let (q, k) = if cfg.use_qk_norm {
        let q = g.storage(hq as u64);
        let k = g.storage(hkv as u64);
        steps.push(rmsnorm_fwd(g, &ids, &q_pre, w.q_norm, &q, hd, nh));
        steps.push(rmsnorm_fwd(g, &ids, &k_pre, w.k_norm, &k, hd, nkv));
        (q, k)
    } else {
        (q_pre, k_pre)
    };
    steps.push(rope2d_fwd(g, ROPE2D, &q, cos, sin, n, nh, hd, hq));
    steps.push(rope2d_fwd(g, ROPE2D, &k, cos, sin, n, nkv, hd, hkv));
    g.submit(&[], &steps);

    let scores = g.storage((nh * cap) as u64);
    let probs = g.storage((nh * cap) as u64);
    let ctx = g.storage(hq as u64);
    g.submit(&[], &gqa_decode_step(g, &dids, nh, nkv, hd, pos, cap, &q, &k, &v, cache.kcache, cache.vcache, &scores, &probs, &ctx));

    let proj = g.storage(d as u64);
    let xmid = g.storage(d as u64);
    g.submit(&[], &[g.step(MATMUL, &[&ctx, w.wo, &proj], &[n, hq, d], n * d)]);
    g.submit(&[], &[g.step(ADD2, &[x, &proj, &xmid], &[n * d], n * d)]);

    let (out, ..) = moe_sublayer(g, cfg, w, &xmid, n);
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
        let (out, ..) = layer_fwd(g, cfg, layer, &h, cos, sin, n, None);
        h = out;
    }
    let ids = kernel_ids();
    let normed = g.storage((n * cfg.hidden) as u64);
    g.submit(&[], &[rmsnorm_fwd(g, &ids, &h, w.final_norm, &normed, cfg.hidden, n)]);
    normed
}
