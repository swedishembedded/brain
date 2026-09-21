// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `crates/modernbert`'s kernel set, and its indices resolved by NAME.
//!
//! Same reasoning as `crates/decide/src/kern.rs`: one device, one pipeline
//! list, every index looked up by name at build time rather than assumed
//! from position, so a kernel this crate forgot to register fails loudly at
//! construction instead of running whatever happens to sit at that slot.
//!
//! Deliberately a SMALL list next to `decide::kern::PIPELINES`: ModernBERT
//! has no GQA/causal path, no learned position or token-type table, no
//! bias-add anywhere in the trunk (`attention_bias`/`mlp_bias`/`norm_bias`
//! are all false on the released config), and this milestone's attention
//! dispatch deliberately skips the key-minor transpose optimisation and the
//! fused flash kernels - see `model.rs`'s module doc for why the plain
//! materialized rungs are the right starting point here.

use gpu_core::Gpu;

pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("add2", kernels::ADD2),
    ("gelu_erf", kernels::GELU_ERF),
    // GeGLU: `mlp_out = gelu(wi_u) * wi_v`. No fused GeGLU kernel - the split
    // into `wi_u`/`wi_v` is two ordinary GEMMs against SLICED halves of the
    // one fused `mlp.wi.weight` (contiguous by output row, since the weight
    // is `[out_features, in_features]` row-major - see `model.rs`'s
    // `mlp_step`), so no new kernel is needed for the split itself; this is
    // the existing `mul` the repo's own GEGLU decision note in `mul.wgsl`
    // already names.
    ("mul", kernels::MUL),
    // Weight-only LayerNorm - `norm_bias: false` on every trunk norm, a
    // genuinely bias-free kernel rather than a biased one fed a zeroed
    // buffer. See `crates/kernels/wgsl/layernorm_nobias.wgsl` and
    // `block::layernorm_nobias_fwd`.
    ("layernorm_nobias", kernels::LAYERNORM_NOBIAS),
    ("rope_base", kernels::ROPE_BASE),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    // The bidirectional LOCAL-window twin (Laya M1). Local-attention layers
    // must use this rung unconditionally until the fused flash kernel grows
    // its own window support - see `model.rs`'s module doc.
    ("attn_scores_cross_win", kernels::ATTN_SCORES_CROSS_WIN),
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
    matmul => "matmul",
    add2 => "add2",
    gelu_erf => "gelu_erf",
    mul => "mul",
    layernorm_nobias => "layernorm_nobias",
    rope_base => "rope_base",
    scores_cross => "attn_scores_cross",
    softmax_cross => "attn_softmax_cross",
    apply_cross => "attn_apply_cross",
    scores_cross_win => "attn_scores_cross_win",
}
