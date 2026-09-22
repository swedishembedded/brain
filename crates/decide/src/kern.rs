// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision model's kernel set, and its indices resolved by NAME.
//!
//! The encoder and the head run on ONE device with ONE pipeline list, so they
//! must agree on what index 7 means. Hard-coded per-module constants cannot
//! give that: two lists that each start at zero silently disagree the moment
//! both halves exist, and the symptom is a plausible number from the wrong
//! kernel rather than a crash.
//!
//! So the list is declared once here and every index is looked up by name at
//! build time. Adding a kernel to [`PIPELINES`] cannot renumber anything, and a
//! kernel a model forgot to register fails by name at construction instead of
//! running whatever happens to sit at that index.

use gpu_core::Gpu;

/// Every kernel either half dispatches, forward and backward.
pub const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    // The `embed` gather's inverse for UNIQUE indices, ASSIGNING rather than
    // accumulating. The head reads its state rows out of the packed hidden
    // states by index instead of by a row offset, so its reverse pass has to
    // put them back the same way - see `Head::set_call` for why an offset is
    // not an option on this card.
    ("row_scatter", kernels::ROW_SCATTER),
    ("emb_bwd", kernels::EMB_BWD),
    // The compact twin of the scatter above, resolved by name through
    // `block::EmbBwdIds`. A call looks up at most a few hundred of this
    // model's 30522 vocabulary rows, and the reference kernel spends an
    // invocation per (table row, channel) regardless.
    ("emb_bwd_uniq", kernels::EMB_BWD_UNIQ),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    // The same kernel retiled to a 64x64 output tile. Registered because a
    // 128x128 tile is the wrong shape for an ENCODER's linears: at a few
    // hundred packed rows, `proj` and `fc2` are 541x384 outputs, which is
    // fifteen workgroups - half the card idle whatever the inner loop does.
    // See `Encoder::gemm` for the rule that picks between them.
    ("matmul_reg3_64", kernels::MATMUL_REG3_64),
    // Attention scores against a KEY-MINOR copy of K, and the transpose that
    // produces it. `attn_scores_cross` reads K with the key index as the
    // fastest thread index while K is key-MAJOR, so consecutive lanes read
    // addresses a fused row apart and every load is its own transaction - a
    // defect that kernel's own header documents and says cannot be fixed
    // inside it. This is the fix it names.
    ("kv_k_headt", kernels::KV_K_HEADT),
    ("attn_scores_cross_kt", kernels::ATTN_SCORES_CROSS_KT),
    // Fused bidirectional attention. One dispatch per span in place of the
    // transpose/scores/softmax/apply quartet, with the score slab never
    // written to memory at all. Measured over eight packed decisions, that
    // quartet was 64% of the forward pass while being 7% of its arithmetic.
    // Forward-only and workgroup-cooperative - see `Encoder::build_steps` for
    // the gate.
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("flash_attn_bidir_reg", kernels::FLASH_ATTN_BIDIR_REG),
    ("flash_attn_bidir_reg2", kernels::FLASH_ATTN_BIDIR_REG2),
    ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT),
    // The same arithmetic over ALL spans in one dispatch. A packed request
    // here is a couple of long windows and ten short option slots, and one
    // dispatch per span is twelve to twenty-four workgroups against a card
    // with thirty compute units.
    ("flash_attn_bidir_spans", kernels::FLASH_ATTN_BIDIR_SPANS),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("add2", kernels::ADD2),
    ("gelu_erf", kernels::GELU_ERF),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    ("layernorm", kernels::LAYERNORM),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    // The cooperative LayerNorm trio. Never indexed directly: `block::
    // LayerNormIds::resolve` finds these by name and selects between them and
    // the reference per device.
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    // The workgroup-per-query-row twin of the line above. Never indexed
    // directly: `block::CrossBwdIds::resolve` finds it by name. The reference
    // kernel gives ONE thread a span's whole `(key, head_dim)` double loop and
    // then walks it twice, which is why it cost more than every GEMM in the
    // reverse pass put together at this model's shapes.
    ("attn_bwd_dscores_cross_rows", kernels::ATTN_BWD_DSCORES_CROSS_ROWS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross_acc", kernels::ATTN_BWD_DK_CROSS_ACC),
    ("attn_bwd_dv_cross_acc", kernels::ATTN_BWD_DV_CROSS_ACC),
    // The reverse twin of `flash_attn_bidir_spans`: the same six-kernel span
    // attention backward, over EVERY span in one dispatch each instead of one
    // dispatch per span per kernel per layer. Never indexed directly -
    // `block::SpansBwdIds::resolve` finds them by name and `Encoder::
    // rebuild_bwd` falls back to the per-span path when they are absent.
    //
    // The count is the point. At eighteen options a request is nineteen
    // spans, so the per-span path records 114 attention dispatches per layer;
    // this records 6. And because they address a span by DATA rather than by
    // a non-zero storage-binding offset, the reverse pass stops tripping
    // `backend-wgpu`'s Intel ANV sliced-binding workaround, which gives every
    // dispatch in a serialised flush its own queue submit and fence.
    ("attn_scores_spans", kernels::ATTN_SCORES_SPANS),
    ("attn_softmax_spans", kernels::ATTN_SOFTMAX_SPANS),
    ("attn_bwd_dscores_spans", kernels::ATTN_BWD_DSCORES_SPANS),
    ("attn_bwd_dq_spans", kernels::ATTN_BWD_DQ_SPANS),
    ("attn_bwd_dk_spans", kernels::ATTN_BWD_DK_SPANS),
    ("attn_bwd_dv_spans", kernels::ATTN_BWD_DV_SPANS),
    // AdamW and its gradient-clipping stage.
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    // The COOPERATIVE grad-norm pair. Never indexed directly either:
    // `optim::Optim` resolves these by name and uses them in place of the two
    // above wherever the device can run a workgroup barrier.
    //
    // Registering them is the entire opt-in, and it is not optional in
    // practice. `gradnorm_sq` reduces a whole tensor from ONE thread, and this
    // model's largest tensor is the `vocab x d_model` token embedding - 11.7M
    // floats. That single thread was 97% of a training step. See
    // `tests/optimizer_kernels.rs`.
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
                        panic!("kernel {:?} is not registered on this device - build it with decide::kern::PIPELINES", $n)
                    })),+
                }
            }
        }
    };
}

ids! {
    embed => "embed",
    row_scatter => "row_scatter",
    emb_bwd => "emb_bwd",
    matmul => "matmul",
    matmul_reg3 => "matmul_reg3",
    matmul_reg3_64 => "matmul_reg3_64",
    kv_k_headt => "kv_k_headt",
    scores_cross_kt => "attn_scores_cross_kt",
    flash_bidir => "flash_attn_bidir",
    flash_bidir_reg => "flash_attn_bidir_reg",
    flash_bidir_reg2 => "flash_attn_bidir_reg2",
    flash_bidir_split => "flash_attn_bidir_split",
    flash_bidir_spans => "flash_attn_bidir_spans",
    matmul_dx => "matmul_dx",
    matmul_dw => "matmul_dw",
    matmul_dx_reg => "matmul_dx_reg",
    matmul_dw_reg => "matmul_dw_reg",
    bias_add => "bias_add",
    bias_grad => "bias_grad",
    add2 => "add2",
    gelu_erf => "gelu_erf",
    gelu_erf_bwd => "gelu_erf_bwd",
    layernorm => "layernorm",
    ln_stats => "ln_stats",
    layernorm_dx => "layernorm_dx",
    ln_dgamma => "layernorm_dgamma",
    ln_dbeta => "layernorm_dbeta",
    scores_cross => "attn_scores_cross",
    softmax_cross => "attn_softmax_cross",
    apply_cross => "attn_apply_cross",
    dscores_cross => "attn_bwd_dscores_cross",
    dq_cross => "attn_bwd_dq_cross",
    dk_cross_acc => "attn_bwd_dk_cross_acc",
    dv_cross_acc => "attn_bwd_dv_cross_acc",
    adamw => "adamw",
    gradnorm_sq => "gradnorm_sq",
    grad_scale => "grad_scale",
    clip_coef => "clip_coef",
    grad_scale_buf => "grad_scale_buf",
}

impl Ids {
    /// The optimizer's five kernel slots, in the order `optim::Optim::new`
    /// takes them.
    pub fn optimizer(&self) -> optim::Optim {
        optim::Optim::new(self.adamw, self.gradnorm_sq, self.grad_scale, self.clip_coef, self.grad_scale_buf)
    }
}
