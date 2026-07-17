// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ZipDepth's architecture config and its parameter layout.
//!
//! The param NAMES are the reference checkpoint's own `state_dict` keys, not a
//! brain-flavoured renaming. That is deliberate and follows yolo, which mirrors
//! Ultralytics' names: it makes weight import a 1:1 name match instead of a
//! hand-maintained translation table, and it makes [`ZipConfig::param_list`]
//! directly checkable against a real `.pth` (see `tests/p1_param_layout.rs`).
//!
//! Everything here is device-free, so the layout can be verified — and weights
//! initialised — with no GPU.

/// Which extra global-context modules the encoder gets.
///
/// Verified against the reference: `EfficientGlobalAttention` is gated on
/// `full` ONLY (`architecture.py:552-553`), and the default is `balanced`
/// (`:511`, `:587`). So **the released 6.1M model contains no ViT-style token
/// attention at all** — it is conv + pooling throughout, which is precisely what
/// makes it a realistic from-scratch target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GlobalMode {
    /// No StripPoolingAttention, no GlobalContextBlock, no EGA.
    None,
    /// + StripPoolingAttention (stage2) + GlobalContextBlock (stage3). THE DEFAULT.
    Balanced,
    /// + EfficientGlobalAttention (stage4). Not implemented — see `blocks.rs`.
    Full,
}

/// ZipDepth's architecture. Field-for-field `MODEL_CONFIGS[variant]`
/// (`architecture.py:10-43`) plus the two forward-shaping flags.
#[derive(Clone, Debug)]
pub struct ZipConfig {
    /// Per-stage channel schedule.
    pub dims: [u32; 4],
    /// QARepBlocks per stage.
    pub depths: [u32; 4],
    /// EfficientGlobalAttention heads. Unused at `GlobalMode::{None,Balanced}`.
    pub heads: u32,
    /// Decoder base width; the fusion widths derive from it (see `dec_chans`).
    pub dec_ch: u32,
    /// Width of the half-resolution fusion + head.
    pub half_dec_ch: u32,
    pub global_mode: GlobalMode,
    /// `true` -> FastConvexUpsample's softmax-over-9 unfold path
    /// (`zipdepth_base.pth`); `false` -> the learned nearest/bilinear blend
    /// (`zipdepth_base_npu.pth`). The two released checkpoints are DIFFERENT
    /// models here — every shared tensor differs, despite the README's claim
    /// that they share encoder/decoder weights — so this flag picks which
    /// checkpoint the layout matches, not a runtime option.
    pub upsample_unfold: bool,
    /// Model input side. The predictor resizes the short side to this and rounds
    /// each axis to a multiple of 32; the NPU additionally needs it square.
    pub input: u32,
}

impl ZipConfig {
    /// `base` — the released 6.1M model. 6,788,254 trainable params unfused;
    /// the 6.1M headline is POST-fusion (RepVGG's 3x3+1x1+identity collapse to a
    /// single 3x3), so the checkpoint stores the larger, unfused form.
    pub fn base() -> ZipConfig {
        ZipConfig {
            dims: [48, 96, 192, 384],
            depths: [2, 2, 6, 2],
            heads: 4,
            dec_ch: 96,
            half_dec_ch: 32,
            global_mode: GlobalMode::Balanced,
            upsample_unfold: true,
            input: 384,
        }
    }
    pub fn small() -> ZipConfig {
        ZipConfig { dims: [24, 48, 96, 192], depths: [2, 2, 4, 2], dec_ch: 32, half_dec_ch: 24, ..ZipConfig::base() }
    }
    pub fn large() -> ZipConfig {
        ZipConfig { dims: [64, 128, 256, 384], depths: [2, 4, 10, 4], heads: 8, dec_ch: 192, half_dec_ch: 48, ..ZipConfig::base() }
    }
    pub fn giant() -> ZipConfig {
        ZipConfig { dims: [96, 192, 384, 512], depths: [2, 4, 14, 6], heads: 8, dec_ch: 288, half_dec_ch: 64, ..ZipConfig::base() }
    }

    /// The stem's first-conv width: `dims[0] / 2`.
    pub fn half_ch(&self) -> u32 {
        self.dims[0] / 2
    }

    /// Decoder widths `(ch4, ch3, ch2, ch1)` from `dec_ch`
    /// (`architecture.py:447-451`). Note `ch2` is `int(dec_ch * 1.5)` — a
    /// truncating cast, so an odd `dec_ch` rounds DOWN.
    pub fn dec_chans(&self) -> (u32, u32, u32, u32) {
        let d = self.dec_ch;
        (d * 3, d * 2, (d * 3) / 2, d)
    }
}

/// Reference `_pick_groups` (`architecture.py:300-305`), verbatim.
///
/// Must be replicated EXACTLY: it decides each grouped conv's weight shape
/// `(out, in/groups, k, k)`, so a divergence here is a shape mismatch on import —
/// or, worse, a silently different model that still loads.
pub fn pick_groups(in_ch: u32, out_ch: u32, max_g: u32) -> u32 {
    for g in [max_g, 2, 1] {
        if in_ch % g == 0 && out_ch % g == 0 {
            return g;
        }
    }
    1
}

/// A parameter: reference `state_dict` key + shape.
pub type Param = (String, Vec<usize>);

fn conv(out: &mut Vec<Param>, name: &str, cout: u32, cin_per_g: u32, k: u32, bias: bool) {
    out.push((format!("{name}.weight"), vec![cout as usize, cin_per_g as usize, k as usize, k as usize]));
    if bias {
        out.push((format!("{name}.bias"), vec![cout as usize]));
    }
}

/// BatchNorm's four learnable/running tensors.
///
/// `num_batches_tracked` is deliberately absent: it is an int64 step counter that
/// no inference or training path here reads, and BN is folded into the conv for
/// export anyway. The importer skips it by name.
fn bn(out: &mut Vec<Param>, name: &str, c: u32) {
    for t in ["weight", "bias", "running_mean", "running_var"] {
        out.push((format!("{name}.{t}"), vec![c as usize]));
    }
}

/// `ConvBN` = conv(bias-free) + BN (+ ReLU, no params).
fn conv_bn(out: &mut Vec<Param>, name: &str, cin: u32, cout: u32, k: u32, groups: u32) {
    conv(out, &format!("{name}.conv"), cout, cin / groups, k, false);
    bn(out, &format!("{name}.bn"), cout);
}

/// `QARepBlock`: a 3x3 branch and a 1x1 branch, each conv+BN, plus an optional
/// RAW identity add.
///
/// The identity branch has **no BN** (`architecture.py:89-108`) — unlike
/// canonical RepVGG. That is why its fuse contributes a kernel but no bias term.
fn qarep(out: &mut Vec<Param>, name: &str, cin: u32, cout: u32) {
    conv(out, &format!("{name}.branch_3x3.0"), cout, cin, 3, false);
    bn(out, &format!("{name}.branch_3x3.1"), cout);
    conv(out, &format!("{name}.branch_1x1.0"), cout, cin, 1, false);
    bn(out, &format!("{name}.branch_1x1.1"), cout);
}

/// `UltraLightFusion`: two grouped 1x1 projections + BN.
fn fusion(out: &mut Vec<Param>, name: &str, high_ch: u32, low_ch: u32, out_ch: u32) {
    let gh = pick_groups(high_ch, out_ch, 4);
    let gl = pick_groups(low_ch, out_ch, 4);
    conv(out, &format!("{name}.proj_high"), out_ch, high_ch / gh, 1, false);
    conv(out, &format!("{name}.proj_low"), out_ch, low_ch / gl, 1, false);
    bn(out, &format!("{name}.bn"), out_ch);
}

impl ZipConfig {
    /// Every parameter, in the reference's own `state_dict` order.
    ///
    /// Device-free by construction, so it can be diffed against a real `.pth`
    /// before a single kernel is dispatched — which is how the layout is gated.
    pub fn param_list(&self) -> Vec<Param> {
        let mut p: Vec<Param> = Vec::new();
        let d = self.dims;
        let half = self.half_ch();
        let use_global = self.global_mode != GlobalMode::None;

        // Normalization buffers live IN the state_dict: the reference applies
        // ImageNet mean/std inside forward (`architecture.py:641`), not in the
        // preprocessing. They are constants, but they ship with the weights.
        p.push(("mean".into(), vec![1, 3, 1, 1]));
        p.push(("std".into(), vec![1, 3, 1, 1]));

        // ---- encoder: stem (/4) ----
        conv_bn(&mut p, "encoder.stem_half", 3, half, 3, 1);
        conv_bn(&mut p, "encoder.stem_quarter", half, d[0], 3, 1);

        // ---- stage 1 ----
        for i in 0..self.depths[0] {
            qarep(&mut p, &format!("encoder.stage1.{i}"), d[0], d[0]);
        }
        // ---- stage 2 (/8) ----
        qarep(&mut p, "encoder.down2", d[0], d[1]);
        let mut idx = 0u32;
        for _ in 0..self.depths[1] {
            qarep(&mut p, &format!("encoder.stage2.{idx}"), d[1], d[1]);
            idx += 1;
        }
        // MinimalMultiScale is appended UNCONDITIONALLY (`architecture.py:531`);
        // two depthwise 3x3 branches (dilation 1 and 2) + BN.
        conv(&mut p, &format!("encoder.stage2.{idx}.branch1"), d[1], 1, 3, false);
        conv(&mut p, &format!("encoder.stage2.{idx}.branch2"), d[1], 1, 3, false);
        bn(&mut p, &format!("encoder.stage2.{idx}.bn"), d[1]);
        idx += 1;
        if use_global {
            // StripPoolingAttention: depthwise 1x1 gate + BN.
            conv(&mut p, &format!("encoder.stage2.{idx}.gate_conv.0"), d[1], 1, 1, false);
            bn(&mut p, &format!("encoder.stage2.{idx}.gate_conv.1"), d[1]);
        }

        // ---- stage 3 (/16) ----
        qarep(&mut p, "encoder.down3", d[1], d[2]);
        let mut idx = 0u32;
        for _ in 0..self.depths[2] {
            qarep(&mut p, &format!("encoder.stage3.{idx}"), d[2], d[2]);
            idx += 1;
        }
        // ChannelAttention (SE) is unconditional (`:542`). reduction=8,
        // hidden = max(dim/8, 4). Both convs are BIAS-FREE.
        let hidden = (d[2] / 8).max(4);
        conv(&mut p, &format!("encoder.stage3.{idx}.fc.0"), hidden, d[2], 1, false);
        conv(&mut p, &format!("encoder.stage3.{idx}.fc.2"), d[2], hidden, 1, false);
        idx += 1;
        if use_global {
            // GlobalContextBlock. reduction=4, hidden = max(dim/4, 8). These
            // convs DO have bias.
            let gh = (d[2] / 4).max(8);
            conv(&mut p, &format!("encoder.stage3.{idx}.context_weight"), 1, d[2], 1, true);
            conv(&mut p, &format!("encoder.stage3.{idx}.transform.0"), gh, d[2], 1, true);
            bn(&mut p, &format!("encoder.stage3.{idx}.transform.1"), gh);
            conv(&mut p, &format!("encoder.stage3.{idx}.transform.3"), d[2], gh, 1, true);
        }

        // ---- stage 4 (/32) ----
        qarep(&mut p, "encoder.down4", d[2], d[3]);
        for i in 0..self.depths[3] {
            qarep(&mut p, &format!("encoder.stage4.{i}"), d[3], d[3]);
        }
        // GlobalMode::Full would append EfficientGlobalAttention here. Not
        // implemented, and not needed: no released checkpoint uses it.

        // ---- SPPF ----
        let spp_hidden = d[3] / 4;
        conv_bn(&mut p, "encoder.spp.cv1", d[3], spp_hidden, 1, 1);
        conv_bn(&mut p, "encoder.spp.cv2", spp_hidden * 4, d[3], 1, 1);

        // ---- MinimalCrossScale: grouped 1x1, no bias, no BN ----
        let g_lh = pick_groups(d[3], d[2], 4);
        let g_hl = pick_groups(d[2], d[3], 4);
        conv(&mut p, "encoder.cross_scale.low_to_high", d[2], d[3] / g_lh, 1, false);
        conv(&mut p, "encoder.cross_scale.high_to_low", d[3], d[2] / g_hl, 1, false);

        // ---- decoder ----
        let (ch4, ch3, ch2, ch1) = self.dec_chans();
        conv_bn(&mut p, "decoder.proj4", d[3], ch4, 1, 1);
        fusion(&mut p, "decoder.fuse3", d[2], ch4, ch3);
        fusion(&mut p, "decoder.fuse2", d[1], ch3, ch2);
        fusion(&mut p, "decoder.fuse1", d[0], ch2, ch1);
        fusion(&mut p, "decoder.fuse_half", half, ch1, self.half_dec_ch);
        conv(&mut p, "decoder.head_half", 1, self.half_dec_ch, 3, true);

        // ---- FastConvexUpsample ----
        let fc = self.half_dec_ch;
        if self.upsample_unfold {
            // mask_pred: conv3x3 -> BN -> ReLU -> conv1x1(-> 9*S*S, with bias)
            let h = (fc / 4).max(8);
            conv(&mut p, "decoder.convex_up.mask_pred.0", h, fc, 3, false);
            bn(&mut p, "decoder.convex_up.mask_pred.1", h);
            conv(&mut p, "decoder.convex_up.mask_pred.3", 9 * 2 * 2, h, 1, true);
        } else {
            // where_conv: 1x1 -> BN -> ReLU -> depthwise5x5 -> BN -> ReLU -> 1x1
            let h = (fc / 2).max(8);
            conv(&mut p, "decoder.convex_up.where_conv.0", h, fc, 1, false);
            bn(&mut p, "decoder.convex_up.where_conv.1", h);
            conv(&mut p, "decoder.convex_up.where_conv.3", h, 1, 5, false);
            bn(&mut p, "decoder.convex_up.where_conv.4", h);
            conv(&mut p, "decoder.convex_up.where_conv.6", 1, h, 1, false);
        }
        p
    }

    /// Total trainable + buffer elements (what the `.pth` stores, excluding the
    /// int64 `num_batches_tracked` counters).
    pub fn numel(&self) -> usize {
        self.param_list().iter().map(|(_, s)| s.iter().product::<usize>()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_groups_matches_the_reference_rule() {
        // Prefers max_g, then 2, then 1 — first g dividing BOTH.
        assert_eq!(pick_groups(384, 192, 4), 4);
        assert_eq!(pick_groups(192, 384, 4), 4);
        assert_eq!(pick_groups(6, 4, 4), 2); // 4 divides 4 but not 6
        assert_eq!(pick_groups(3, 4, 4), 1); // neither 4 nor 2 divides 3
        assert_eq!(pick_groups(2, 2, 4), 2);
    }

    #[test]
    fn dec_chans_truncates_the_1_5x_width() {
        let c = ZipConfig::base();
        assert_eq!(c.dec_chans(), (288, 192, 144, 96));
        // int(dec_ch * 1.5) TRUNCATES: an odd dec_ch rounds down.
        let odd = ZipConfig { dec_ch: 33, ..ZipConfig::base() };
        assert_eq!(odd.dec_chans().2, 49, "int(33*1.5) == 49, not 50");
    }

    #[test]
    fn base_layout_is_the_released_shape() {
        let c = ZipConfig::base();
        assert_eq!(c.half_ch(), 24);
        let p = c.param_list();
        // The released zipdepth_base.pth has 278 keys, 43 of which are int64
        // num_batches_tracked (one per BatchNorm) that we deliberately omit.
        assert_eq!(p.len(), 278 - 43, "param count != the checkpoint's float tensors");
        // ...and 6,802,927 total elements, of which the 43 counters are 1 each.
        assert_eq!(c.numel(), 6_802_927 - 43, "element count != the checkpoint's");
    }

    #[test]
    fn npu_variant_swaps_only_the_upsampler() {
        let unfold = ZipConfig::base();
        let npu = ZipConfig { upsample_unfold: false, ..ZipConfig::base() };
        let (a, b) = (unfold.param_list(), npu.param_list());
        let shared = |v: &Vec<Param>| -> Vec<String> {
            v.iter().map(|(n, _)| n.clone()).filter(|n| !n.starts_with("decoder.convex_up")).collect()
        };
        assert_eq!(shared(&a), shared(&b), "the two checkpoints must agree outside convex_up");
        // 278 vs 283 keys; minus their 43/44 counters -> 235 vs 239 floats.
        assert_eq!(b.len(), 283 - 44, "npu param count != the checkpoint's float tensors");
        assert_eq!(npu.numel(), 6_801_324 - 44, "npu element count != the checkpoint's");
    }
}
