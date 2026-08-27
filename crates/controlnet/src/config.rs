// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `ControlNetModel` configuration and its canonical brain-side tensor
//! manifest.
//!
//! ## A ControlNet is defined BY its backbone, not alongside it
//! Every number a ControlNet shares with the UNet it conditions —
//! `block_out_channels`, `down_block_types`, `layers_per_block`,
//! `transformer_layers_per_block`, `attention_heads`, `cross_attention_dim`,
//! the two GroupNorm epsilons, the whole `text_time` added-conditioning chain —
//! *must* be that UNet's, because the ControlNet's down and mid blocks are a
//! trainable COPY of the UNet's. So [`ControlNetConfig`] holds a
//! [`UNetConfig`] rather than restating those fields, and its manifest is
//! literally a **filter of the backbone's manifest** plus the two things that
//! are genuinely new: the conditioning-image embedder and the zero-convs.
//!
//! Restating them would be the classic drift: a ControlNet whose
//! `transformer_layers_per_block` says `[1, 2, 10]` while the UNet's says
//! `[1, 2, 4]` imports cleanly, runs, and produces residuals that are wrong
//! everywhere below level 1.

use sdxlunet::config::UNetConfig;

use crate::adapter::InjectionPoint;

/// One diffusers `ControlNetModel`.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlNetConfig {
    /// The backbone whose early blocks this ControlNet copies.
    pub backbone: UNetConfig,
    /// Channels of the conditioning IMAGE (3 for rgb/bgr, 1 for a raw depth or
    /// edge map that the reference still ships as 3).
    pub conditioning_channels: u32,
    /// `ControlNetConditioningEmbedding`'s per-stage widths — `[16, 32, 96,
    /// 256]` for every released SD/SDXL ControlNet. `len() - 1` of these are
    /// stride-2 stages, so the embedder downsamples by `2^(len-1)`, which MUST
    /// equal the VAE's factor (8) or the embedding does not land on the latent
    /// grid. [`ControlNetConfig::cond_downscale`] is that number.
    ///
    /// [`ControlNetConfig::validate`] deliberately does **not** check it — a
    /// config alone does not know its backbone's VAE factor. The check lives in
    /// [`crate::model::ControlNet::new`], which asserts the embedder lands at
    /// exactly the recorded latent size; that is a named error rather than a
    /// mis-sized `add` deeper in the graph.
    pub conditioning_embedding_out_channels: Vec<u32>,
}

impl ControlNetConfig {
    /// The InstantID SDXL ControlNet (`ControlNetModel/config.json`), which is
    /// also the shape of every `diffusers/controlnet-*-sdxl-1.0` release.
    pub fn sdxl() -> ControlNetConfig {
        ControlNetConfig {
            backbone: UNetConfig::sdxl_base(),
            conditioning_channels: 3,
            conditioning_embedding_out_channels: vec![16, 32, 96, 256],
        }
    }

    /// A tiny variant matching [`UNetConfig::tiny`], for the weight-free smoke
    /// test. One stride-2 stage, so the embedder downsamples by 2.
    pub fn tiny() -> ControlNetConfig {
        ControlNetConfig {
            backbone: UNetConfig::tiny(),
            conditioning_channels: 3,
            conditioning_embedding_out_channels: vec![6, 10],
        }
    }

    /// How much the conditioning embedder downsamples: `2^(stages - 1)`.
    pub fn cond_downscale(&self) -> u32 {
        1 << (self.conditioning_embedding_out_channels.len() as u32 - 1)
    }

    /// Structural checks that a wrong config would otherwise turn into a
    /// plausible-looking forward.
    pub fn validate(&self) -> Result<(), String> {
        if self.conditioning_embedding_out_channels.len() < 2 {
            return Err("controlnet: the conditioning embedder needs at least 2 stages".into());
        }
        if self.backbone.block_out_channels.is_empty() {
            return Err("controlnet: the backbone has no levels".into());
        }
        Ok(())
    }

    /// The injection points this ControlNet produces residuals for, at a
    /// `h × w` latent — the same names and the same order as the UNet's
    /// `ControlAdapter` impl derives from `skip_stack()`.
    ///
    /// Derived from the BACKBONE's skip stack, not from the zero-conv list, so
    /// the two can only ever agree: `controlnet_down_blocks.k` conditions
    /// `skip_stack()[k]` by construction in diffusers, and this says so.
    pub fn injection_points(&self, h: u32, w: u32) -> Vec<InjectionPoint> {
        let mut v = Vec::new();
        for (k, (c, sh, sw)) in self.residual_shapes(h, w).into_iter().enumerate() {
            let name = if k + 1 == self.n_points() { "mid".to_string() } else { format!("down.{k}") };
            v.push(InjectionPoint::spatial(name, c, sh, sw));
        }
        v
    }

    /// Number of injection points: `skip_stack().len()` down + 1 mid.
    pub fn n_points(&self) -> usize {
        self.backbone.skip_stack().len() + 1
    }

    /// `(channels, h, w)` of every residual at a `h × w` latent, in injection
    /// order.
    ///
    /// The down-path portion is [`UNetConfig::skip_shapes`] - hoisted there
    /// because it is backbone math, not ControlNet math, and the mid block's
    /// own residual sits at the same `(channels, h, w)` the down path's LAST
    /// entry already reached: the coarsest level pushes no downsampler after
    /// its resnets, so `skip_shapes`'s final entry already IS `(cmid, ch, cw)`.
    pub fn residual_shapes(&self, h: u32, w: u32) -> Vec<(u32, u32, u32)> {
        let mut v = self.backbone.skip_shapes(h, w);
        let mid = *v.last().expect("levels >= 1");
        v.push(mid);
        v
    }

    /// Canonical brain-side tensor manifest: `(name, shape)` for every
    /// parameter the graph binds.
    ///
    /// Three parts, in this order:
    /// 1. the backbone manifest **filtered** to what a ControlNet keeps — the
    ///    conditioning chain, `conv_in`, the down blocks and the mid block. A
    ///    ControlNet has no up path, no `conv_norm_out` and no `conv_out`;
    /// 2. `controlnet_cond_embedding.*` — the conditioning-image embedder;
    /// 3. `controlnet_down_blocks.{k}` / `controlnet_mid_block` — the
    ///    zero-convs, one 1x1 conv per injection point.
    ///
    /// Part 1 being a filter rather than a re-derivation is the point: the
    /// three host-side fusions (`attn1.qkv`, `attn2.kv`, the split GEGLU) are
    /// the backbone's, described once, in `sdxlunet::config`.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut v: Vec<(String, Vec<usize>)> = self
            .backbone
            .tensor_manifest()
            .into_iter()
            .filter(|(n, _)| Self::is_controlnet_half(n))
            .collect();

        // ---- the conditioning-image embedder ----
        let ce = &self.conditioning_embedding_out_channels;
        let conv = |v: &mut Vec<(String, Vec<usize>)>, p: &str, cin: u32, cout: u32| {
            v.push((format!("{p}.weight"), vec![cout as usize, cin as usize, 3, 3]));
            v.push((format!("{p}.bias"), vec![cout as usize]));
        };
        conv(&mut v, "controlnet_cond_embedding.conv_in", self.conditioning_channels, ce[0]);
        for i in 0..ce.len() - 1 {
            // Each pair is (same-resolution conv, stride-2 widening conv) — see
            // `ControlNetConditioningEmbedding.__init__`.
            conv(&mut v, &format!("controlnet_cond_embedding.blocks.{}", 2 * i), ce[i], ce[i]);
            conv(&mut v, &format!("controlnet_cond_embedding.blocks.{}", 2 * i + 1), ce[i], ce[i + 1]);
        }
        conv(
            &mut v,
            "controlnet_cond_embedding.conv_out",
            *ce.last().expect("validated >= 2 stages"),
            self.backbone.block_out_channels[0],
        );

        // ---- the zero-convs ----
        let zero = |v: &mut Vec<(String, Vec<usize>)>, p: &str, c: u32| {
            v.push((format!("{p}.weight"), vec![c as usize, c as usize, 1, 1]));
            v.push((format!("{p}.bias"), vec![c as usize]));
        };
        for (k, c) in self.backbone.skip_stack().into_iter().enumerate() {
            zero(&mut v, &format!("controlnet_down_blocks.{k}"), c);
        }
        zero(&mut v, "controlnet_mid_block", *self.backbone.block_out_channels.last().expect("levels >= 1"));
        v
    }

    /// Which of the backbone's tensors a ControlNet also has.
    ///
    /// A prefix test, deliberately: `up_blocks.*`, `conv_norm_out.*` and
    /// `conv_out.*` are exactly the tensors a ControlNet does NOT ship, and the
    /// importer's two-way coverage turns a wrong answer here into a named
    /// error rather than a silent shape mismatch.
    fn is_controlnet_half(name: &str) -> bool {
        name.starts_with("time_embedding.")
            || name.starts_with("add_embedding.")
            || name.starts_with("conv_in.")
            || name.starts_with("down_blocks.")
            || name.starts_with("mid_block.")
    }
}

/// The kind of a backbone level, re-exported so a caller configuring a
/// ControlNet does not have to depend on `crates/sdxlunet` directly.
pub use sdxlunet::config::BlockKind as LevelKind;

#[cfg(test)]
mod tests {
    use super::*;
    use sdxlunet::config::BlockKind;

    #[test]
    fn sdxl_conditioning_embedder_downsamples_by_the_vae_factor() {
        // The embedder must land the conditioning image on the LATENT grid, so
        // its downscale has to equal the SDXL VAE's 8. A 5-stage embedder
        // (downscale 16) would produce a half-size embedding whose add against
        // `conv_in` would fail on length — but only at run time.
        assert_eq!(ControlNetConfig::sdxl().cond_downscale(), 8);
        assert_eq!(ControlNetConfig::tiny().cond_downscale(), 2);
    }

    #[test]
    fn sdxl_has_nine_down_points_and_one_mid() {
        let c = ControlNetConfig::sdxl();
        assert_eq!(c.n_points(), 10);
        let p = c.injection_points(32, 32);
        assert_eq!(p.len(), 10);
        assert_eq!(p[0].name, "down.0");
        assert_eq!(p[8].name, "down.8");
        assert_eq!(p[9].name, "mid");
    }

    /// The residual shapes, read straight off the diffusers golden's
    /// `out.down{k}` / `out.mid` at a 32x32 latent. The spatial half is the
    /// error-prone one: the downsampler's residual is at the COARSER size (it
    /// is pushed after the stride-2 conv), so `down.3` is 16x16 while `down.2`
    /// is 32x32 at the same 320 channels.
    #[test]
    fn sdxl_residual_shapes_match_the_reference() {
        let want = [
            (320, 32, 32),
            (320, 32, 32),
            (320, 32, 32),
            (320, 16, 16),
            (640, 16, 16),
            (640, 16, 16),
            (640, 8, 8),
            (1280, 8, 8),
            (1280, 8, 8),
            (1280, 8, 8),
        ];
        assert_eq!(ControlNetConfig::sdxl().residual_shapes(32, 32), want);
    }

    /// The manifest must be exactly the released checkpoint's tensor count once
    /// the backbone's three fusions are accounted for.
    ///
    /// The InstantID `ControlNetModel` ships **844** tensors. Its
    /// `BasicTransformerBlock` count is 34 (down level 1: 2 attentions x 2
    /// blocks; level 2: 2 x 10; mid: 1 x 10), and each costs a net -1 tensor
    /// (-2 for the fused qkv, -1 for the fused kv, +2 for the split GEGLU).
    #[test]
    fn sdxl_manifest_matches_the_checkpoint_count() {
        let c = ControlNetConfig::sdxl();
        let b = &c.backbone;
        let mut tb = 0usize;
        for i in 0..b.levels() {
            if b.down_block_types[i] == BlockKind::CrossAttn {
                tb += (b.layers_per_block * b.transformer_layers_per_block[i]) as usize;
            }
        }
        tb += b.transformer_layers_per_block[b.levels() - 1] as usize; // mid
        assert_eq!(tb, 34, "the ControlNet half of SDXL has 34 BasicTransformerBlocks");
        let m = c.tensor_manifest();
        assert_eq!(m.len(), 844 - tb, "manifest {} tensors", m.len());
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate names in the manifest");
    }

    /// A ControlNet has no up path and no output head — if any of those leaked
    /// through the filter the importer would demand tensors the checkpoint does
    /// not have.
    #[test]
    fn the_manifest_drops_exactly_the_up_path_and_the_head() {
        let m = ControlNetConfig::sdxl().tensor_manifest();
        for (n, _) in &m {
            assert!(
                !n.starts_with("up_blocks.") && !n.starts_with("conv_norm_out") && !n.starts_with("conv_out"),
                "{n} is not part of a ControlNet"
            );
        }
        // ... and it does keep the whole conditioning chain, which is the half
        // that a naive "down + mid" filter would drop.
        for want in ["time_embedding.linear_1.weight", "add_embedding.linear_2.bias", "conv_in.weight"] {
            assert!(m.iter().any(|(n, _)| n == want), "manifest is missing {want}");
        }
    }

    /// One zero-conv per injection point, each square at that point's width.
    #[test]
    fn every_injection_point_has_a_zero_conv() {
        let c = ControlNetConfig::sdxl();
        let m = c.tensor_manifest();
        for (k, (ch, _, _)) in c.residual_shapes(32, 32).into_iter().enumerate() {
            let name = if k + 1 == c.n_points() {
                "controlnet_mid_block.weight".to_string()
            } else {
                format!("controlnet_down_blocks.{k}.weight")
            };
            let (_, shape) = m.iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(*shape, vec![ch as usize, ch as usize, 1, 1], "{name}");
        }
    }
}
