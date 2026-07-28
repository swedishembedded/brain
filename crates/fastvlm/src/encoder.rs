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

use gpu_core::DeviceBuffer;
use paramstore::ParamStore;
use vision::{Act, Conv, ConvKernelIds, ConvSpec, Ctx, Norm, Shape};

/// Index of a manually-dispatched kernel (not covered by `ConvKernelIds`) in
/// [`PIPELINES`], by name. Panics if absent — a programming error.
fn kidx(name: &str) -> usize {
    PIPELINES.iter().position(|(n, _)| *n == name).unwrap_or_else(|| panic!("kernel `{name}` not in FastViTHD PIPELINES"))
}

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

/// The atomic FastViTHD conv primitive: a (grouped/depthwise/dense) conv with an
/// optional BatchNorm, an optional **per-channel** bias (`add_chan_bcast`, which
/// unlike the dense-only fused `conv_bias` handles grouped convs), and an optional
/// **erf**-GELU. Every FastViTHD conv (MobileOne, RepMixer, ConvFFN's fc1/fc2,
/// RepCPE, PatchEmbed, conv_exp) is one of these. Weight keys: `{prefix}.conv.*`
/// (+ `.bn.*` when `bn`) and `{prefix}.bias` (when `bias`).
pub struct ConvUnit {
    conv: Conv,
    bias_name: Option<String>,
    gelu: bool,
    out_shape: Shape,
    biased: DeviceBuffer,
    activated: DeviceBuffer,
    k_add_chan: usize,
    k_gelu: usize,
}

impl ConvUnit {
    #[allow(clippy::too_many_arguments)]
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, cout: u32, k: u32, stride: u32, pad: u32, groups: u32, bn: bool, bias: bool, gelu: bool) -> ConvUnit {
        let spec = ConvSpec {
            cout,
            k,
            stride,
            pad,
            groups,
            dilation: 1,
            norm: if bn { Norm::Bn } else { Norm::None },
            act: Act::None,
            bias: false, // bias is applied by us (per-channel, grouped-safe)
        };
        // ConvNames::brain appends `.conv.weight`/`.bn.*` to the prefix.
        let conv = Conv::with_spec(ctx, prefix, in_shape, spec, true);
        let out_shape = conv.out_shape;
        let n = out_shape.numel() as u64;
        ConvUnit {
            bias_name: bias.then(|| format!("{prefix}.bias")),
            gelu,
            out_shape,
            biased: ctx.gpu.storage(n),
            activated: ctx.gpu.storage(n),
            k_add_chan: kidx("add_chan_bcast"),
            k_gelu: kidx("gelu_erf"),
            conv,
        }
    }

    pub fn out_shape(&self) -> Shape {
        self.out_shape
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut p = self.conv.param_list();
        if let Some(b) = &self.bias_name {
            p.push((b.clone(), self.out_shape.c as usize));
        }
        p
    }

    /// Run conv → (+bias) → (GELU); returns the final output buffer.
    pub fn forward<'a>(&'a self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) -> &'a DeviceBuffer {
        self.conv.forward(ctx, ps, x_in);
        let mut cur: &DeviceBuffer = self.conv.out();
        if let Some(b) = &self.bias_name {
            let (nn, c, hw) = (self.out_shape.n, self.out_shape.c, self.out_shape.h * self.out_shape.w);
            ctx.gpu.submit(&[], &[ctx.step(self.k_add_chan, &[cur, ps.w(b), &self.biased], &[nn, c, hw], self.out_shape.numel())]);
            cur = &self.biased;
        }
        if self.gelu {
            let tot = self.out_shape.numel();
            ctx.gpu.submit(&[], &[ctx.step(self.k_gelu, &[cur, &self.activated], &[tot], tot)]);
            cur = &self.activated;
        }
        // SAFETY of lifetimes: `cur` borrows either self.conv.out(), self.biased,
        // or self.activated — all owned by `self`, so `'a` is valid.
        cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;
    use gpu_core::Gpu;
    use paramstore::ParamStore;
    use std::collections::HashMap;
    use vision::Conv;

    /// Random init for a param list, keeping BN running stats sane (var=1, mean=0)
    /// so eval-mode BN never hits sqrt of a negative.
    fn rand_init(plist: &[(String, usize)], rng: &mut Rng) -> HashMap<String, Vec<f32>> {
        plist
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
            .collect()
    }

    #[test]
    fn conv_unit_depthwise_bias_gelu_runs() {
        // RepMixer-style unit: depthwise 3×3, no BN, +bias, +erf-GELU.
        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ctx(&gpu);
        let in_shape = Shape::new(1, 16, 8, 8);
        let unit = ConvUnit::new(&ctx, "u", in_shape, 16, 3, 1, 1, 16, false, true, true);
        let plist = unit.param_list();
        assert!(plist.iter().any(|(n, _)| n == "u.conv.weight"), "depthwise conv weight present");
        assert!(plist.iter().any(|(n, _)| n == "u.bias"), "per-channel bias present");

        let mut rng = Rng::new(2);
        let ps = ParamStore::new(&gpu, plist.clone(), &rand_init(&plist, &mut rng));
        let x: Vec<f32> = (0..in_shape.numel() as usize).map(|_| rng.next_f32() - 0.5).collect();
        let xb = gpu.storage_init("x", &x);
        let out = unit.forward(&ctx, &ps, &xb);
        let outv = gpu.read(out, unit.out_shape().numel() as usize);
        assert_eq!(outv.len(), (16 * 8 * 8) as usize, "depthwise 3×3 pad 1 preserves 8×8");
        assert!(outv.iter().all(|v| v.is_finite()), "output finite");
        // erf-GELU has a global minimum ≈ -0.17, so post-activation values can't go
        // much below that — a sanity check that the GELU actually ran.
        assert!(outv.iter().cloned().fold(f32::INFINITY, f32::min) > -0.2, "GELU floor");
    }

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
