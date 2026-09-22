// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `crates/modernbert`'s kernel set, and its indices resolved by NAME.
//!
//! Same reasoning as `crates/decide/src/kern.rs`: one device, one pipeline
//! list, every index looked up by name at build time rather than assumed
//! from position, so a kernel this crate forgot to register fails loudly at
//! construction instead of running whatever happens to sit at that slot.
//!
//! Deliberately a SMALL list next to `decide::kern::PIPELINES`: ModernBERT's
//! own trunk has no GQA/causal path, no learned position or token-type
//! table, no bias-add anywhere (`attention_bias`/`mlp_bias`/`norm_bias` are
//! all false on the released config), and this milestone's attention
//! dispatch deliberately skips the key-minor transpose optimisation and the
//! fused flash kernels - see `model.rs`'s module doc for why the plain
//! materialized rungs are the right starting point here.
//!
//! `laya.rs`'s decision head is the opposite shape (standard biased
//! `nn.TransformerEncoderLayer`s, see its own module doc), which is why
//! `bias_add`/`layernorm`/`relu_inplace` are registered below even though
//! nothing in the trunk needs them - one pipeline list for the whole crate,
//! same reasoning as `decide::kern::PIPELINES` serving both `Encoder` and
//! `Head`.
//!
//! **Laya M5 (seeded backward) grew this list, but added ZERO new WGSL
//! files.** Every gradient this crate needs - the no-bias LayerNorm's dx and
//! dgamma, GeGLU's backward, the windowed attention backward, the head's
//! plain-ReLU backward, the act_head detach's column split - turned out to
//! already exist as either a genuinely bias/window-agnostic kernel (`ln_stats`,
//! `layernorm_dx`, `layernorm_dgamma` never read a `beta` binding at all, so
//! they are the correct no-bias dx/dgamma kernels unmodified - the M2 plan
//! note predicting a new `layernorm_nobias_dx` kernel was wrong, corrected
//! here) or a kernel already built for a different model
//! (`leaky_relu`/`leaky_relu_bwd` at `slope=0.0` for plain ReLU,
//! `concat_split` for the act_head detach's column slice). See `model.rs`'s
//! and `laya.rs`'s own module docs for the per-piece reasoning.

use gpu_core::Gpu;

pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("emb_bwd", kernels::EMB_BWD),
    ("matmul", kernels::MATMUL),
    ("add2", kernels::ADD2),
    ("gelu_erf", kernels::GELU_ERF),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    // GeGLU: `mlp_out = gelu(wi_u) * wi_v`. No fused GeGLU kernel - the split
    // into `wi_u`/`wi_v` is two ordinary GEMMs against SLICED halves of the
    // one fused `mlp.wi.weight` (contiguous by output row, since the weight
    // is `[out_features, in_features]` row-major - see `model.rs`'s
    // `mlp_step`), so no new kernel is needed for the split itself; this is
    // the existing `mul` the repo's own GEGLU decision note in `mul.wgsl`
    // already names. Its backward COMPOSES from the same `mul` kernel plus
    // `gelu_erf_bwd` - see `model.rs`'s backward module doc.
    ("mul", kernels::MUL),
    // Weight-only LayerNorm - `norm_bias: false` on every trunk norm, a
    // genuinely bias-free kernel rather than a biased one fed a zeroed
    // buffer. See `crates/kernels/wgsl/layernorm_nobias.wgsl` and
    // `block::layernorm_nobias_fwd`.
    ("layernorm_nobias", kernels::LAYERNORM_NOBIAS),
    // The trunk's LayerNorm backward - `ln_stats`/`layernorm_dx`/
    // `layernorm_dgamma` never bind a `beta` buffer at all (checked directly
    // against their own WGSL source), so they are ALREADY the correct
    // no-bias dx/dgamma kernels; no `layernorm_nobias_dx` kernel exists or is
    // needed. `layernorm_dbeta` is deliberately NOT registered here - the
    // trunk has no bias to have a gradient.
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("rope_base", kernels::ROPE_BASE),
    ("rope_base_bwd", kernels::ROPE_BASE_BWD),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross_acc", kernels::ATTN_BWD_DK_CROSS_ACC),
    ("attn_bwd_dv_cross_acc", kernels::ATTN_BWD_DV_CROSS_ACC),
    // The bidirectional LOCAL-window twin (Laya M1). Local-attention layers
    // must use this rung unconditionally until the fused flash kernel grows
    // its own window support - see `model.rs`'s module doc.
    ("attn_scores_cross_win", kernels::ATTN_SCORES_CROSS_WIN),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("bias_grad", kernels::BIAS_GRAD),
    // Laya M3's head only: standard biased Linear/LayerNorm plus a plain
    // (non-gated) ReLU FFN activation - see `laya.rs`'s module doc for why
    // this is the opposite bias/activation convention from the trunk above.
    ("bias_add", kernels::BIAS_ADD),
    ("layernorm", kernels::LAYERNORM),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("relu_inplace", kernels::RELU_INPLACE),
    // Plain ReLU forward/backward for a TRAINABLE head, `slope=0.0` -
    // `leaky_relu`/`leaky_relu_bwd` at that slope compute exactly plain
    // ReLU and its adjoint, so no new kernel is written for it. Non-in-place
    // (unlike `relu_inplace`) because the backward needs the pre-activation
    // value back - see `laya.rs`'s module doc.
    ("leaky_relu", kernels::LEAKY_RELU),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    // Laya M5's act_head detach (see `laya.rs`'s module doc): splits the
    // `[q, d+4]` concat gradient back down to its `[q, d]` `pooled` share,
    // discarding the last 4 (host-feature) columns entirely - the device-side
    // half of reproducing the real checkpoint's `softmax(logits).detach()`
    // training recipe. An existing kernel (built for NCHW concat backward
    // elsewhere in the workspace; `H=W=1` here), not a new one.
    ("concat_split", kernels::CONCAT_SPLIT),
    // AdamW and its gradient-clipping stage - registered so
    // `LayaHead::adamw_step_scaled` (Laya M6) has a device-resident optimizer
    // to step. Same five kernels `decide::kern::PIPELINES` registers for the
    // same reason; see that list's own comment.
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    // The COOPERATIVE grad-norm pair - never indexed directly, `optim::Optim`
    // resolves these by name and prefers them over the two single-thread
    // kernels above wherever the device can run a workgroup barrier. Laya's
    // head is far smaller than `decide`'s encoder (see `laya.rs`'s own
    // parameter count), but registering the cooperative pair costs nothing
    // and keeps this list one honest copy of `decide::kern::PIPELINES`'s
    // reasoning rather than a smaller one nobody re-derived.
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("clip_coef_wg", kernels::CLIP_COEF_WG),
];

macro_rules! ids {
    ($($f:ident => $n:literal),+ $(,)?) => {
        /// One resolved index per kernel, built once per device.
        #[derive(Clone, Copy, Debug)]
        pub struct Ids { $(pub $f: usize),+ }

        impl Ids {
            /// Resolve against a device built with [`PIPELINES`]. A missing
            /// kernel is named, because the alternative is dispatching
            /// whatever occupies that slot.
            pub fn resolve(g: &Gpu) -> Ids {
                Ids {
                    $($f: g.kernel_index($n).unwrap_or_else(|| {
                        panic!("kernel {:?} is not registered on this device - build it with modernbert::kern::PIPELINES", $n)
                    })),+
                }
            }
        }
    };
}

ids! {
    embed => "embed",
    emb_bwd => "emb_bwd",
    matmul => "matmul",
    add2 => "add2",
    gelu_erf => "gelu_erf",
    gelu_erf_bwd => "gelu_erf_bwd",
    mul => "mul",
    layernorm_nobias => "layernorm_nobias",
    ln_stats => "ln_stats",
    layernorm_dx => "layernorm_dx",
    ln_dgamma => "layernorm_dgamma",
    rope_base => "rope_base",
    rope_base_bwd => "rope_base_bwd",
    scores_cross => "attn_scores_cross",
    softmax_cross => "attn_softmax_cross",
    apply_cross => "attn_apply_cross",
    dscores_cross => "attn_bwd_dscores_cross",
    dq_cross => "attn_bwd_dq_cross",
    dk_cross_acc => "attn_bwd_dk_cross_acc",
    dv_cross_acc => "attn_bwd_dv_cross_acc",
    scores_cross_win => "attn_scores_cross_win",
    matmul_dx => "matmul_dx",
    matmul_dw => "matmul_dw",
    bias_grad => "bias_grad",
    bias_add => "bias_add",
    layernorm => "layernorm",
    ln_dbeta => "layernorm_dbeta",
    relu_inplace => "relu_inplace",
    leaky_relu => "leaky_relu",
    leaky_relu_bwd => "leaky_relu_bwd",
    concat_split => "concat_split",
    adamw => "adamw",
    gradnorm_sq => "gradnorm_sq",
    grad_scale => "grad_scale",
    clip_coef => "clip_coef",
    grad_scale_buf => "grad_scale_buf",
}

impl Ids {
    /// The optimizer's five kernel slots, in the order `optim::Optim::new`
    /// takes them - see `decide::kern::Ids::optimizer`'s own doc.
    pub fn optimizer(&self) -> optim::Optim {
        optim::Optim::new(self.adamw, self.gradnorm_sq, self.grad_scale, self.clip_coef, self.grad_scale_buf)
    }
}
