// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FastViTHD vision encoder — built on brain's `crates/vision` Conv framework.
//!
//! Five stages (RepMixer conv token-mixing in 0–2, self-attention in 3–4) built
//! from the shared `Conv`/`ConvSpec` primitives + the attention kernels; no
//! net-new device kernels. This module grows the encoder incrementally; today it
//! establishes the conv pipeline + `Ctx` scaffolding (the same pattern yolo/depth
//! use) and a smoke that runs a `Conv` in this crate.

use std::sync::OnceLock;

use vision::{ConvKernelIds, Ctx, Shape};

/// Kernel registry for the FastViTHD tower: the conv/BN/act family (RepMixer,
/// MobileOne, ConvFFN, PatchEmbed, SE) plus the transpose + attention kernels the
/// stage-4/5 attention blocks need. Resolved by name, so a superset is fine —
/// absent kernels map to `NONE` and only error if actually dispatched.
pub const PIPELINES: &[(&str, &str)] = &[
    // --- conv / bn / activation ---
    ("conv2d", kernels::CONV2D),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("conv2d_gd", kernels::CONV2D_GD),
    ("conv2d_gd_reg", kernels::CONV2D_GD_REG),
    ("conv2d_gd_dx", kernels::CONV2D_GD_DX),
    ("conv2d_gd_dw", kernels::CONV2D_GD_DW),
    ("conv2d_tiled", kernels::CONV2D_TILED),
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("conv_act", kernels::CONV_ACT),
    ("conv_act_tiled", kernels::CONV_ACT_TILED),
    ("conv_act_reg", kernels::CONV_ACT_REG),
    ("conv_epilogue", kernels::CONV_EPILOGUE),
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    ("bn_stats", kernels::BN_STATS),
    ("bn_running", kernels::BN_RUNNING),
    ("bn_train", kernels::BN_TRAIN),
    ("bn_eval", kernels::BN_EVAL),
    ("bn_dstats", kernels::BN_DSTATS),
    ("bn_dx", kernels::BN_DX),
    ("bn_dgamma", kernels::BN_DGAMMA),
    ("bn_dbeta", kernels::BN_DBETA),
    ("silu", kernels::SILU),
    ("silu_bwd", kernels::SILU_BWD),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    ("im2col", kernels::IM2COL),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("avgpool2d", kernels::AVGPOOL2D),
    ("avgpool2d_dx", kernels::AVGPOOL2D_DX),
    ("add2", kernels::ADD2),
    ("add_inplace", kernels::ADD_INPLACE),
    ("add_chan_bcast", kernels::ADD_CHAN_BCAST),
    ("add_chan_bcast_dv", kernels::ADD_CHAN_BCAST_DV),
    ("scale_chan", kernels::SCALE_CHAN),
    ("chan_place", kernels::CHAN_PLACE),
    // --- transpose + attention (stage-4/5 blocks) ---
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("layernorm", kernels::LAYERNORM),
    ("gelu", kernels::GELU),
    ("gelu_erf", kernels::GELU_ERF),
    ("matmul", kernels::MATMUL),
    ("matmul_rows", kernels::MATMUL_ROWS),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
];

/// Cached conv kernel-id resolution against [`PIPELINES`].
pub fn ids() -> &'static ConvKernelIds {
    static IDS: OnceLock<ConvKernelIds> = OnceLock::new();
    IDS.get_or_init(|| ConvKernelIds::resolve(PIPELINES))
}

/// Build a fresh conv `Ctx` bound to `gpu` and the tower kernel ids.
pub fn ctx(gpu: &gpu_core::Gpu) -> Ctx<'_> {
    Ctx::new(gpu, ids())
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;
    use gpu_core::Gpu;
    use paramstore::ParamStore;
    use std::collections::HashMap;
    use vision::Conv;

    #[test]
    fn conv_runs_in_fastvlm_crate() {
        // Prove the conv framework is wired: one strided conv 3→16 over 8×8 → 4×4.
        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ctx(&gpu);
        let in_shape = Shape::new(1, 3, 8, 8);
        let conv = Conv::new(&ctx, "c0", in_shape, 16, 3, 2, 1, true);

        let plist = conv.param_list();
        let mut rng = Rng::new(1);
        let init: HashMap<String, Vec<f32>> = plist
            .iter()
            .map(|(n, sz)| {
                let v = if n.ends_with("running_var") {
                    vec![1.0; *sz]
                } else if n.contains("running_mean") {
                    vec![0.0; *sz]
                } else {
                    (0..*sz).map(|_| (rng.next_f32() - 0.5) * 0.2).collect()
                };
                (n.clone(), v)
            })
            .collect();
        let ps = ParamStore::new(&gpu, plist, &init);

        let img: Vec<f32> = (0..in_shape.numel() as usize).map(|_| rng.next_f32() - 0.5).collect();
        let img_b = gpu.storage_init("img", &img);
        conv.forward(&ctx, &ps, &img_b);
        let out = gpu.read(conv.out(), conv.out_shape.numel() as usize);
        assert_eq!(out.len(), (16 * 4 * 4) as usize, "3→16 stride-2 over 8×8 → 16×4×4");
        assert!(out.iter().all(|v| v.is_finite()), "conv output must be finite");
    }
}
