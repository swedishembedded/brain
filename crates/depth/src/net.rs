// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ZipDepth's kernel registry.
//!
//! The pipeline order here is ZipDepth's own and bears no relation to yolo's —
//! which is exactly why the shared `vision` blocks resolve their kernels BY NAME
//! ([`vision::ConvKernelIds::resolve`]) rather than by a positional constant. This
//! model registers only what it dispatches; anything it omits resolves to
//! `vision::NONE` and panics naming the kernel if reached, instead of silently
//! running whatever happens to sit at that index.
//!
//! Note there is no `relu` entry: `leaky_relu` with `slope = 0` IS ReLU in both
//! directions (`v>=0 -> v` else `0*v`; `x>=0 -> dy` else `0*dy`), and `slope` is a
//! bit-cast f32 in the uniform, so ZipDepth's activation costs no new kernel.

use std::sync::OnceLock;

use vision::ConvKernelIds;

/// Kernel registry passed to `Gpu::new`/`Gpu::new_cpu`.
///
/// Ordered by concern, not by any external contract — nothing indexes this array
/// positionally.
pub const PIPELINES: &[(&str, &str)] = &[
    // ---- conv. BOTH forms are registered, and both are used: ZipDepth's dense
    // units (the stem, most QARepBlocks, SPPF, SE, GlobalContextBlock) are
    // groups=1/dilation=1 and route to `conv2d`, which `backend-cpu` fast-paths
    // through AVX2/winograd by NAME. Only the genuinely grouped/dilated units
    // (MinimalMultiScale's depthwise branches, MinimalCrossScale, the fusion
    // projections, the NPU upsampler's depthwise 5x5) go to `conv2d_gd`.
    // `vision::ConvSpec::is_dense` makes that choice per unit.
    ("conv2d", kernels::CONV2D),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("conv2d_gd", kernels::CONV2D_GD),
    // Register-tiled grouped forward — taken over conv2d_gd when present. The
    // grouped 1x1 fusion projections + dilated depthwise branches were the
    // hottest kernel of a frame (56% CPU) as naive/scalar dispatches.
    ("conv2d_gd_reg", kernels::CONV2D_GD_REG),
    ("conv2d_gd_dx", kernels::CONV2D_GD_DX),
    ("conv2d_gd_dw", kernels::CONV2D_GD_DW),
    // Biased convs (head_half, mask_pred.3, GlobalContextBlock) are all DENSE, so
    // the fused conv+per-channel-bias kernel covers every one of them.
    //
    // NOT `bias_add`: it is `out[idx] += bias[idx % n]`, i.e. [M,N] row-major with
    // the biased dim TRAILING — a LINEAR-layer bias. In NCHW the channel is not
    // the trailing dim, so it silently indexes garbage. yolo's head.rs:14-29
    // documents the workaround (a host-built [C*HW] broadcast) and then replaces
    // the forward with conv_bias exactly as here; `bias_grad` is still the right
    // BACKWARD via the same [M=N, N=C*HW] view plus a host spatial reduce.
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("bias_grad", kernels::BIAS_GRAD),
    // ---- fused eval: conv -> BN(eval) affine -> act, one dispatch. The act
    // SELECTOR in the kernels' uniform (0 none, 1 relu, 2 silu, 3 sigmoid) is
    // what lets a ReLU model fuse — every dense+BN unit takes conv_act_reg
    // instead of conv2d + bn_eval + leaky_relu (3 full-tensor passes -> 1,
    // ~8x less input traffic on the GPU). Grouped/dilated units still run
    // unfused. conv_act/conv_act_tiled are the BRAIN_NAIVE_CONV/
    // BRAIN_TILED_CONV comparison variants of the same fusion.
    ("conv_act", kernels::CONV_ACT),
    ("conv_act_reg", kernels::CONV_ACT_REG),
    ("conv_act_tiled", kernels::CONV_ACT_TILED),
    // ---- batchnorm ----
    ("bn_stats", kernels::BN_STATS),
    ("bn_running", kernels::BN_RUNNING),
    ("bn_train", kernels::BN_TRAIN),
    ("bn_eval", kernels::BN_EVAL),
    ("bn_dstats", kernels::BN_DSTATS),
    ("bn_dx", kernels::BN_DX),
    ("bn_dgamma", kernels::BN_DGAMMA),
    ("bn_dbeta", kernels::BN_DBETA),
    // ---- activations (relu == leaky_relu at slope 0) ----
    ("leaky_relu", kernels::LEAKY_RELU),
    ("leaky_relu_bwd", kernels::LEAKY_RELU_BWD),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    // ---- spatial ----
    ("maxpool5", kernels::MAXPOOL5), // LightweightSPPF: K/pad are params
    ("maxpool5_dx", kernels::MAXPOOL5_DX),
    ("avgpool2d", kernels::AVGPOOL2D), // SE pool, cross-scale down, strip pool
    ("avgpool2d_dx", kernels::AVGPOOL2D_DX),
    ("resize_bilinear", kernels::RESIZE_BILINEAR),
    ("resize_bilinear_dx", kernels::RESIZE_BILINEAR_DX),
    ("resize_nearest", kernels::RESIZE_NEAREST),
    ("resize_nearest_dx", kernels::RESIZE_NEAREST_DX),
    ("pixel_shuffle", kernels::PIXEL_SHUFFLE),
    ("pixel_shuffle_dx", kernels::PIXEL_SHUFFLE_DX),
    // ---- attention / context ----
    ("softmax_k", kernels::SOFTMAX_K), // both the 9-neighbour axis AND (M=1) the map
    ("softmax_k_dx", kernels::SOFTMAX_K_DX),
    ("weighted_gap", kernels::WEIGHTED_GAP),
    ("weighted_gap_dx", kernels::WEIGHTED_GAP_DX),
    ("weighted_gap_dm", kernels::WEIGHTED_GAP_DM),
    ("add_chan_bcast", kernels::ADD_CHAN_BCAST),
    ("add_chan_bcast_dv", kernels::ADD_CHAN_BCAST_DV),
    ("broadcast_add_hw", kernels::BROADCAST_ADD_HW),
    ("broadcast_add_hw_da", kernels::BROADCAST_ADD_HW_DA),
    ("convex_upsample", kernels::CONVEX_UPSAMPLE),
    ("convex_upsample_dmask", kernels::CONVEX_UPSAMPLE_DMASK),
    ("convex_upsample_dd", kernels::CONVEX_UPSAMPLE_DD),
    // ---- elementwise / plumbing ----
    ("add2", kernels::ADD2),
    // MinimalCrossScale's `x + 0.3*delta`: a scaled accumulate needs no scale kernel.
    ("axpy", kernels::AXPY),
    ("mul", kernels::MUL),
    ("scale_chan", kernels::SCALE_CHAN),
    ("concat2", kernels::CONCAT2),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("chan_place", kernels::CHAN_PLACE),
    // ---- loss ----
    ("masked_l1", kernels::MASKED_L1),
    ("masked_l1_grad", kernels::MASKED_L1_GRAD),
    // ---- optimizer ----
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
];

/// Optimizer kernel indices. These ARE looked up positionally (`Optim::new` takes
/// bare indices), so unlike the block kernels they need constants — kept next to
/// the array they index.
pub fn optim_ids() -> (usize, usize, usize, usize, usize) {
    let k = |n: &str| PIPELINES.iter().position(|(m, _)| *m == n).expect("optimizer kernel missing");
    (k("adamw"), k("gradnorm_sq"), k("grad_scale"), k("clip_coef"), k("grad_scale_buf"))
}

/// The loss kernels' indices (`masked_l1`, `masked_l1_grad`) — positional for
/// the same reason as [`optim_ids`].
pub fn loss_ids() -> (usize, usize) {
    let k = |n: &str| PIPELINES.iter().position(|(m, _)| *m == n).expect("loss kernel missing");
    (k("masked_l1"), k("masked_l1_grad"))
}

/// The shared conv blocks' kernel ids, resolved by name against [`PIPELINES`].
pub fn ids() -> &'static ConvKernelIds {
    static IDS: OnceLock<ConvKernelIds> = OnceLock::new();
    IDS.get_or_init(|| ConvKernelIds::resolve(PIPELINES))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_kernel_name_exists() {
        for (name, _) in PIPELINES {
            assert!(
                kernels::ALL.iter().any(|(n, _)| n == name),
                "PIPELINES registers `{name}`, which is not in the kernel registry"
            );
        }
    }

    #[test]
    fn no_duplicate_registrations() {
        let mut seen = std::collections::HashSet::new();
        for (name, _) in PIPELINES {
            assert!(seen.insert(*name), "`{name}` registered twice");
        }
    }

    /// The blocks reach their kernels through `Ctx.ids`, so anything ZipDepth's
    /// blocks dispatch must resolve. A missing one would panic at build time
    /// naming the kernel — this catches it in a fast unit test instead.
    #[test]
    fn the_conv_block_kernels_all_resolve() {
        let i = ids();
        for (id, what) in [
            (i.bn_stats, "bn_stats"),
            (i.bn_train, "bn_train"),
            (i.bn_eval, "bn_eval"),
            (i.bn_dstats, "bn_dstats"),
            (i.bn_dx, "bn_dx"),
            (i.bn_dgamma, "bn_dgamma"),
            (i.bn_dbeta, "bn_dbeta"),
            (i.add2, "add2"),
            (i.concat2, "concat2"),
            (i.maxpool5, "maxpool5"),
        ] {
            assert_ne!(id, vision::NONE, "`{what}` did not resolve");
        }
    }

    /// Kernels ZipDepth genuinely does not use resolve to NONE, and that is the
    /// CORRECT outcome: `need()` then panics naming the kernel rather than
    /// dispatching a wrong one.
    ///
    /// `conv2d` is deliberately NOT in this list. An earlier version asserted it
    /// was absent, on the assumption that ZipDepth is grouped conv throughout —
    /// which is false: the stem, most QARepBlocks, SPPF, SE and
    /// GlobalContextBlock are all groups=1, and routing them through `conv2d_gd`
    /// would forfeit the AVX2/winograd fast path for nothing. Both forms are
    /// registered; `ConvSpec::is_dense` picks per unit.
    #[test]
    fn kernels_zipdepth_does_not_use_resolve_to_none() {
        let i = ids();
        assert_eq!(i.silu, vision::NONE, "ZipDepth is ReLU, not SiLU");
        assert_eq!(i.upsample2, vision::NONE, "ZipDepth resizes to arbitrary sizes, not 2x");
    }

    /// The fused eval kernels ARE registered: since the act selector landed the
    /// fused path serves ReLU (and None/Sigmoid) units, so ZipDepth's dense+BN
    /// convs run one `conv_act_reg` dispatch instead of conv2d + bn_eval + act.
    /// (An earlier version pinned `conv_act_reg == NONE` — "the fused path is
    /// SiLU-only" — which stopped being true, and was the 3x-dispatch perf bug.)
    #[test]
    fn the_fused_eval_kernels_are_registered() {
        let i = ids();
        assert_ne!(i.conv_act_reg, vision::NONE, "dense+BN units fuse through conv_act_reg");
        assert_ne!(i.conv_bias_reg, vision::NONE, "dense biased convs take the register-tiled kernel");
    }

    /// ...and the dense conv IS registered, precisely so ZipDepth's many
    /// groups=1 units keep the fast path.
    #[test]
    fn both_conv_forms_are_registered() {
        let i = ids();
        assert_ne!(i.conv2d, vision::NONE, "dense units need conv2d for the CPU fast path");
        assert_ne!(i.conv2d_gd, vision::NONE, "grouped/dilated units need conv2d_gd");
    }
}
