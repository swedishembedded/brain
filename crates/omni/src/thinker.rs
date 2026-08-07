// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni's Thinker text decoder: a Qwen3-style GQA+QK-norm+RoPE
//! attention stack (identical shape to `qwen::Qwen`'s, and to `tts::talker`'s
//! reuse of it) over a sparse top-k MoE FFN (`model::moe`, no shared
//! expert).
//!
//! **Why this is a new, dedicated forward rather than an extension of
//! `qwen::Qwen`**: `qwen::Qwen`'s attention/RoPE/QK-norm/DeepStack-splice
//! machinery is already exactly right for Thinker (confirmed by reading
//! `forward_steps` in full — same RMSNorm, same per-head QK-norm, same
//! `block::gqa_fwd`, an existing `mrope`/`rope2d_step` path, and DeepStack
//! splice support already wired for the vision-language case). But its
//! internals (sharding, LoRA, int8, KV-cache) are all interleaved in one
//! large constructor with no seam to swap out just the dense SwiGLU MLP for
//! `model::moe`'s sparse one, unlike `crates/glm`, which already carries an
//! `Mlp::Dense`/`Mlp::Moe` enum at exactly that point (see
//! `docs/models/omni/status.md`'s M6 design note for the two ways to close
//! this gap; giving `qwen::Qwen` the same seam `glm` has is the "one
//! implementation" answer and the natural following step). This module is
//! deliberately narrower than `qwen::Qwen` in every other respect
//! (forward-only, single device, no sharding/LoRA/int8/KV-cache) so it can
//! be validated against real weights now — it composes the SAME shared
//! primitives `qwen::Qwen` itself is built from
//! (`model::block::{rmsnorm_fwd,rope2d_fwd,gqa_fwd}`), not a re-derivation of
//! the attention math.
//!
//! **M-RoPE**: every layer takes the table-driven [`model::block::rope2d_fwd`]
//! path (the same kernel `qwen::Qwen::rope2d_step` dispatches for Qwen3-VL),
//! fed a `[n, head_dim/2]` `cos`/`sin` table the caller builds with
//! `qwenvl::mrope::{get_rope_index, mrope_tables}`. There is deliberately no
//! separate "plain RoPE" code path: for a token stream where all three axes
//! carry the same position (pure text, or pure audio — see the M6 design
//! note), Omni's interleaved M-RoPE collapses exactly to ordinary half-split
//! RoPE (`qwenvl::mrope`'s own `diagonal_positions_collapse_to_plain_rope`
//! test proves this), so a caller with no image/video/audio span just builds
//! that degenerate diagonal table via `get_rope_index(tokens, image_token_id,
//! &[])` (empty grids) rather than reaching for a second kernel — one
//! implementation for both cases, per the M6a lesson about wiring the wrong
//! RoPE kernel into a second, parallel path (`docs/lessons.md`, status.md's
//! M6a entry).
//!
//! **Multimodal splice**: not this module's concern. A caller with image/
//! audio embeddings splices them into the token-embedding buffer via
//! `model::vlm::splice_fwd` (`splice.wgsl`, in [`thinker_pipelines`]) BEFORE
//! calling [`decode`] — `decode` and [`layer_fwd`] only ever see an
//! already-assembled `[n, d]` embedding sequence, exactly like `x` in
//! `qwen::Qwen::write_img_embeds`'s contract.

use gpu_core::{DeviceBuffer, Gpu};
use model::block::{gqa_fwd, rmsnorm_fwd, rope2d_fwd, Gqa, KernelIds};
use model::moe::{expert_fwd, router_fwd, ExpertScratch, MoeIds};

use crate::config::MoeTextConfig;

/// Kernel pipeline this module dispatches. Forward-only: the backward slots
/// `KernelIds` carries (`rms_inv`/`rmsnorm_dx`/`rmsnorm_dw`/`rope_bwd`/
/// `gqa_d*`/`silu_da`/`silu_db`) are never reached, so they point at index 0
/// (`rmsnorm`) rather than a real backward kernel — harmless since nothing
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

const MATMUL: usize = 5;
const ADD2: usize = 6;
const ROPE2D: usize = 1;
/// `model::vlm::splice_fwd`'s kernel index — exposed for a caller assembling
/// a multimodal embedding sequence before [`decode`]; see the module doc.
pub const SPLICE: usize = 11;

/// One decoder layer's weights, keyed exactly as they arrive from
/// `omni::import` (`thinker.blocks.{l}.*`, prefix already stripped by the
/// caller — see [`ThinkerLayer::new`]'s doc).
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

/// Runs one Thinker decoder layer forward: `x [n, d] -> [n, d]` (`out`),
/// alongside three intermediates a parity test needs to localize a
/// divergence to a specific stage: the router's raw logits `[n, n_experts]`,
/// the post-attention/pre-MoE residual (`xmid`), and the dense post-topk-
/// renorm gate `[n, n_experts]` `model::moe`'s expert loop actually consumes.
/// `n` is the sequence length (batch folded in, matching every other
/// forward-only harness in this engine). `cos`/`sin` are the `[n, head_dim/2]`
/// M-RoPE tables (`qwenvl::mrope::mrope_tables`) — see the module doc for why
/// there is one RoPE path, not a separate "plain" and "M-RoPE" one.
///
/// Note for callers comparing `out` against a full-model reference dump:
/// `out` is this ONE layer's raw output. A real decoder stack's top-level
/// `model.norm` (applied once, after every layer) is not part of this
/// function and must be applied separately if the comparison target is a
/// `last_hidden_state`-shaped tensor (see `thinker_layer_parity.rs`'s module
/// doc for the real bug this distinction caught; [`decode`] applies it).
#[allow(clippy::too_many_arguments)]
pub fn layer_fwd(g: &Gpu, cfg: &MoeTextConfig, w: &ThinkerLayerWeights, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, n: u32) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let ids = kernel_ids();
    let mids = moe_ids();
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

    let scores = g.storage((nh * n * n) as u64);
    let probs = g.storage((nh * n * n) as u64);
    let ctx = g.storage((n * hq) as u64);
    let ga = Gqa { b: 1, t: n, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
    steps.extend(gqa_fwd(g, &ids, &ga, &q, &k, &v, &scores, &probs, &ctx));

    let proj = g.storage((n * d) as u64);
    steps.push(g.step(MATMUL, &[&ctx, w.wo, &proj], &[n, hq, d], n * d));
    let xmid = g.storage((n * d) as u64);
    steps.push(g.step(ADD2, &[x, &proj, &xmid], &[n * d], n * d));

    let xn2 = g.storage((n * d) as u64);
    steps.push(rmsnorm_fwd(g, &ids, &xmid, w.ln2, &xn2, d, n));

    let router_logits = g.storage((n * cfg.n_experts) as u64);
    steps.push(g.step(MATMUL, &[&xn2, w.router, &router_logits], &[n, d, cfg.n_experts], n * cfg.n_experts));
    g.submit(&[], &steps);

    // Router gate (top-k + renorm) needs its own submit boundary from the
    // per-expert loop below only in the sense that the gate buffer must be
    // fully written first -- `router_fwd` -> one more submit, then the
    // expert loop reads it. Kept as a second pass for clarity; cheap either
    // way (router is a single small kernel).
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
    let moe_out = g.storage((n * d) as u64);
    for (e, (gw, uw, dw)) in w.experts.iter().enumerate() {
        let steps = expert_fwd(g, &mids, &shape, &xn2, &gate, gw, uw, dw, &scratch, &moe_out, e as u32, e != 0);
        g.submit(&[], &steps);
    }

    let out = g.storage((n * d) as u64);
    g.submit(&[], &[g.step(ADD2, &[&xmid, &moe_out, &out], &[n * d], n * d)]);

    (out, router_logits, xmid, gate)
}

/// All 48 decoder layers plus the top-level final RMSNorm
/// (`thinker.model.norm.weight`) — the weights [`decode`] needs beyond one
/// layer's own [`ThinkerLayerWeights`].
pub struct ThinkerWeights<'a> {
    pub layers: &'a [ThinkerLayerWeights<'a>],
    pub final_norm: &'a DeviceBuffer,
}

/// Runs the full Thinker text-decoder stack: `x [n, d] -> [n, d]`, one
/// [`layer_fwd`] per entry in `w.layers` chained residual-to-residual, then
/// `model.norm` — the piece [`layer_fwd`] deliberately leaves out (see its
/// doc). `x` is an already-assembled embedding sequence: plain token
/// embeddings for pure text, or the same buffer after a caller has spliced in
/// image/audio embeddings via `model::vlm::splice_fwd` at the appropriate
/// rows (`SPLICE`, in [`thinker_pipelines`]) — this function has no opinion
/// on how `x` was built, only on what to do with it. `cos`/`sin` are the
/// M-RoPE tables for this same sequence (`qwenvl::mrope`), shared unchanged
/// across every layer (position doesn't change with depth).
pub fn decode(g: &Gpu, cfg: &MoeTextConfig, w: &ThinkerWeights, x: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let mut h = x.clone(); // DeviceBuffer is Arc-backed; cheap, aliases the same buffer.
    for layer in w.layers {
        let (out, ..) = layer_fwd(g, cfg, layer, &h, cos, sin, n);
        h = out;
    }
    let ids = kernel_ids();
    let normed = g.storage((n * cfg.hidden) as u64);
    g.submit(&[], &[rmsnorm_fwd(g, &ids, &h, w.final_norm, &normed, cfg.hidden, n)]);
    normed
}
