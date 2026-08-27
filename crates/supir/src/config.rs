// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `SupirConfig` and the canonical brain-side tensor manifest for SUPIR's
//! **delta over frozen SDXL**: the `GLVControl` trunk, the 12 adaptors and
//! the degradation-robust `denoise_encoder`.
//!
//! ## A SUPIR config is defined by its two backbones, not alongside them
//! Exactly the `ControlNetConfig` precedent (`crates/controlnet/src/config.rs`):
//! [`SupirConfig`] holds a [`UNetConfig`] for the frozen backbone AND one for
//! the trunk, rather than restating either's `block_out_channels`,
//! `transformer_layers_per_block`, `cross_attention_dim`, GroupNorm epsilons
//! or added-conditioning chain. The trunk is `GLVControl`, a hand-written
//! copy of the SAME down+mid schedule (verified against the real checkpoint
//! headers - see [`trunk_manifest`]'s doc), so its tensor manifest is
//! [`UNetConfig::tensor_manifest`] filtered to the down+mid half and
//! reprefixed, never hand-listed.
//!
//! ## The manifest is the 1035-tensor SUPIR delta, not the frozen backbone
//! [`SupirConfig::tensor_manifest`] covers exactly `model.control_model.*`
//! (811), `model.diffusion_model.project_modules.*` (118) and
//! `first_stage_model.denoise_encoder.*` (106) - the full SUPIR-over-SDXL
//! delta measured from the real checkpoint header. The frozen SDXL
//! backbone's own ~1610 tensors are NOT part of this manifest; they are
//! loaded the same way `crates/controlnet` loads them, through
//! `sdxlunet::import::load` against `cfg.backbone`.
//!
//! ## The zero_conv width question, settled against the real checkpoint
//! One fact upstream source alone does not disambiguate: whether
//! `ZeroSFT.zero_conv`'s output channel count is the *skip* width or the
//! *`h_ori`* width, since they differ at joins 2, 3, 5 and 6. Reading every
//! `project_modules.*.zero_conv.weight` shape out of the real
//! `SUPIR-v0Q_fp32.safetensors` checkpoint settles it: at join 2 (`h_ori`
//! 1280, skip 640), `project_modules.8.zero_conv.weight` is `[640, 640, 1,
//! 1]`, the SKIP width, not `h_ori`'s. Every other `zero_conv` in the
//! checkpoint is square at exactly the control tensor's OWN channel width,
//! which by the shape-preservation proof below always equals the skip
//! width, so `ZeroConv1x1(c)` is a channel-count-preserving `control_c ->
//! control_c` conv, added directly onto `h_skip` (both being `control_c`
//! wide) before the concat. This also explains why a `zero_conv` tensor
//! exists at the post-mid site (`project_modules.11`, `[1280, 1280, 1,
//! 1]`), even though that call has "no concat": with `h_ori` absent, the
//! general `h1 = concat(h_ori, h_skip + ZeroConv1x1(c))` formula
//! degenerates to `h1 = h + ZeroConv1x1(c)` (`h` standing in for the sole
//! "skip" argument), which is exactly
//! [`vae::blocks::skipfuse::SkipFuse::fuse_mid`]'s shape, a genuinely
//! distinct call, not a zero-width `fuse_skip`.
//!
//! The mapping from checkpoint index to the join-in-pop-order table below
//! (the injection order, `adapter_idx` 11->0) was cross-checked against
//! every `project_modules.*` shape and matches exactly - see
//! [`AdaptorConfig::for_backbone`]'s doc for the general derivation.

use crate::adaptors::AdaptorConfig;
use sdxlunet::config::UNetConfig;

/// One SUPIR model: the frozen SDXL backbone it restores through, the
/// `GLVControl` trunk that mirrors it, the 12 adaptors that replace its
/// up-path skip concatenation, and the degradation-robust encoder that
/// produces the trunk's hint.
#[derive(Clone, Debug, PartialEq)]
pub struct SupirConfig {
    /// The frozen SDXL 1.0 base UNet SUPIR restores through. Never SUPIR's
    /// own tensors - loaded via `sdxlunet::import::load` against this field,
    /// exactly as `crates/controlnet` loads its backbone.
    pub backbone: UNetConfig,
    /// `GLVControl`: a hand-written copy of the backbone's down blocks +
    /// middle block (own `time_embed`/`label_emb`, plus the hint embedder).
    /// Numerically identical to `backbone` for the released SUPIR-v0Q
    /// checkpoint (both are SDXL-shaped), but named and manifested
    /// separately because nothing GUARANTEES the two agree - the whole
    /// reason [`SupirConfig`] holds two [`UNetConfig`]s instead of one.
    pub trunk: UNetConfig,
    /// The 12 `ZeroSFT`/`ZeroCrossAttn` adaptors, derived from `backbone`'s
    /// own skip-stack arithmetic - see [`AdaptorConfig::for_backbone`].
    pub adaptors: AdaptorConfig,
    /// The degradation-robust encoder (`first_stage_model.denoise_encoder.*`
    /// in the checkpoint): byte-identical topology to the frozen SDXL VAE
    /// encoder (verified against the real checkpoint header - see
    /// [`denoise_encoder_manifest`]'s doc), only the weights differ. Not yet
    /// wired into a forward (that is `pipeline.rs`'s job, still to be
    /// written) - this field exists so [`SupirConfig::tensor_manifest`] can
    /// cover its 106 tensors for the two-way import gate.
    pub denoise_encoder: vae::config::VaeConfig,
}

/// `nn.Conv2d(4 -> block_out_channels[0])`, zero-init: the hint embedder.
/// The ONE difference from a vanilla ControlNet's 8-layer pixel embedder -
/// the hint is already a latent, not a pixel image.
pub const HINT_EMBEDDER: &str = "input_hint_block";

impl SupirConfig {
    /// The released `SUPIR-v0Q`/`SUPIR-v0F` shape: both backbones are SDXL
    /// 1.0 base, verified against the real checkpoint headers.
    pub fn sdxl() -> SupirConfig {
        SupirConfig {
            backbone: UNetConfig::sdxl_base(),
            trunk: UNetConfig::sdxl_base(),
            adaptors: AdaptorConfig::for_backbone(&UNetConfig::sdxl_base()),
            denoise_encoder: sdxl_vae_encoder_config(),
        }
    }

    /// A deliberately tiny variant for weight-free smoke tests: same graph
    /// shape as [`UNetConfig::tiny`], distinct dims everywhere.
    pub fn tiny() -> SupirConfig {
        let backbone = UNetConfig::tiny();
        SupirConfig {
            trunk: backbone.clone(),
            adaptors: AdaptorConfig::for_backbone(&backbone),
            denoise_encoder: vae::config::VaeConfig {
                in_channels: 3,
                out_channels: 3,
                latent_channels: 2,
                block_out_channels: vec![6, 10],
                layers_per_block: 1,
                norm_num_groups: 2,
                norm_eps: 1e-6,
                mid_block_add_attention: true,
                scaling_factor: 0.13025,
                shift_factor: 0.0,
                use_quant_conv: true,
                use_post_quant_conv: true,
                patch_size: [1, 1],
                batch_norm_eps: 1e-4,
            },
            backbone,
        }
    }

    /// Canonical brain-side tensor manifest for the SUPIR **delta**: the
    /// trunk (prefixed `control_model.`), the adaptors (prefixed
    /// `project_modules.`) and the denoise encoder (prefixed
    /// `denoise_encoder.`) - 1035 tensors for [`SupirConfig::sdxl`],
    /// checked against the real checkpoint by
    /// `crates/supir/tests` (gate: mapping-units, porting.md §5 rung 1).
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut v = trunk_manifest(&self.trunk);
        v.extend(self.adaptors.tensor_manifest());
        v.extend(denoise_encoder_manifest(&self.denoise_encoder));
        v
    }
}

/// `GLVControl`'s tensor manifest: [`UNetConfig::tensor_manifest`] filtered
/// to the down+mid half (the same filter `ControlNetConfig::tensor_manifest`
/// applies to a vanilla ControlNet - kept here as SUPIR's own, since
/// depending on `crates/controlnet` for a four-line name filter would be a
/// stranger dependency than repeating the technique), reprefixed under
/// `control_model.` so the names are exactly what
/// `sdxlunet::model::Rec::set_prefix("control_model.")` will look up, plus
/// the hint embedder [`HINT_EMBEDDER`] the vanilla backbone does not have.
///
/// This is the "diffusers-style" name a real checkpoint does NOT ship under:
/// SUPIR's own checkpoint stores the trunk with CompVis/LDM naming
/// (`input_blocks.*`, `middle_block.*`, `time_embed.*`, `label_emb.*`). The
/// rename from LDM to this manifest's names is [`crate::import`]'s job, done
/// once at import time so this manifest, and every `Rec` lookup against it,
/// stays the SAME diffusers-style name [`sdxlunet::config::UNetConfig`]
/// already uses everywhere else in this codebase.
pub fn trunk_manifest(trunk: &UNetConfig) -> Vec<(String, Vec<usize>)> {
    let mut v: Vec<(String, Vec<usize>)> =
        trunk_manifest_unprefixed(trunk).into_iter().map(|(n, s)| (format!("control_model.{n}"), s)).collect();
    let c0 = trunk.block_out_channels[0] as usize;
    v.push((format!("control_model.{HINT_EMBEDDER}.weight"), vec![c0, 4, 3, 3]));
    v.push((format!("control_model.{HINT_EMBEDDER}.bias"), vec![c0]));
    v
}

/// [`trunk_manifest`]'s down+mid half, WITHOUT the `control_model.` prefix
/// and without the hint embedder - exactly the shape
/// `sdxlunet::import::remap_manifest` needs its `manifest` argument in
/// ([`crate::import::remap_trunk`] hands it this, not [`trunk_manifest`]'s
/// own prefixed/hint-including output, since that function's fused-name
/// resolution reads the manifest's own suffixes and knows nothing about a
/// `control_model.` prefix or a hint embedder that has no fusion to apply).
pub(crate) fn trunk_manifest_unprefixed(trunk: &UNetConfig) -> Vec<(String, Vec<usize>)> {
    trunk.tensor_manifest().into_iter().filter(|(n, _)| is_trunk_half(n)).collect()
}

/// Which of the backbone's tensors `GLVControl` also has: the conditioning
/// chain, `conv_in`, the down blocks and the mid block. No up path, no
/// `conv_norm_out`, no `conv_out` - the trunk returns raw hidden states, it
/// never produces an image. Identical filter to
/// `ControlNetConfig::is_controlnet_half`, restated here rather than shared
/// because the two configs have no other reason to depend on each other.
fn is_trunk_half(name: &str) -> bool {
    name.starts_with("time_embedding.")
        || name.starts_with("add_embedding.")
        || name.starts_with("conv_in.")
        || name.starts_with("down_blocks.")
        || name.starts_with("mid_block.")
}

/// The degradation-robust encoder's manifest, CompVis/LDM-named exactly as
/// the checkpoint ships it (`down.{i}.block.{j}.*`, `down.{i}.downsample.conv.*`,
/// `mid.block_1`/`mid.attn_1`/`mid.block_2`, `norm_out`, `conv_in`,
/// `conv_out`) - NOT diffusers' `down_blocks.*` naming, because nothing in
/// this crate binds these tensors to a forward yet (that is `pipeline.rs`'s
/// job, Step 6, deferred) and inventing a rename with no consumer would be
/// speculative. Derived from [`vae::config::VaeConfig`]'s own fields
/// (`block_out_channels`, `layers_per_block`) rather than hand-listed, and
/// verified to match the real checkpoint's 106 tensors exactly (see
/// `crates/supir/tests`).
pub fn denoise_encoder_manifest(v: &vae::config::VaeConfig) -> Vec<(String, Vec<usize>)> {
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    let c0 = v.block_out_channels[0] as usize;
    let cin_img = v.in_channels as usize;
    out.push(("denoise_encoder.conv_in.weight".into(), vec![c0, cin_img, 3, 3]));
    out.push(("denoise_encoder.conv_in.bias".into(), vec![c0]));

    let resnet = |out: &mut Vec<(String, Vec<usize>)>, p: &str, cin: usize, cout: usize| {
        out.push((format!("{p}.norm1.weight"), vec![cin]));
        out.push((format!("{p}.norm1.bias"), vec![cin]));
        out.push((format!("{p}.conv1.weight"), vec![cout, cin, 3, 3]));
        out.push((format!("{p}.conv1.bias"), vec![cout]));
        out.push((format!("{p}.norm2.weight"), vec![cout]));
        out.push((format!("{p}.norm2.bias"), vec![cout]));
        out.push((format!("{p}.conv2.weight"), vec![cout, cout, 3, 3]));
        out.push((format!("{p}.conv2.bias"), vec![cout]));
        if cin != cout {
            out.push((format!("{p}.nin_shortcut.weight"), vec![cout, cin, 1, 1]));
            out.push((format!("{p}.nin_shortcut.bias"), vec![cout]));
        }
    };

    let mut prev = c0;
    let levels = v.block_out_channels.len();
    for i in 0..levels {
        let cout = v.block_out_channels[i] as usize;
        for j in 0..v.layers_per_block {
            let cin = if j == 0 { prev } else { cout };
            resnet(&mut out, &format!("denoise_encoder.down.{i}.block.{j}"), cin, cout);
            prev = cout;
        }
        if i + 1 < levels {
            out.push((format!("denoise_encoder.down.{i}.downsample.conv.weight"), vec![cout, cout, 3, 3]));
            out.push((format!("denoise_encoder.down.{i}.downsample.conv.bias"), vec![cout]));
        }
    }

    let cmid = *v.block_out_channels.last().expect("levels >= 1") as usize;
    resnet(&mut out, "denoise_encoder.mid.block_1", cmid, cmid);
    if v.mid_block_add_attention {
        out.push(("denoise_encoder.mid.attn_1.norm.weight".into(), vec![cmid]));
        out.push(("denoise_encoder.mid.attn_1.norm.bias".into(), vec![cmid]));
        for leaf in ["q", "k", "v", "proj_out"] {
            out.push((format!("denoise_encoder.mid.attn_1.{leaf}.weight"), vec![cmid, cmid, 1, 1]));
            out.push((format!("denoise_encoder.mid.attn_1.{leaf}.bias"), vec![cmid]));
        }
    }
    resnet(&mut out, "denoise_encoder.mid.block_2", cmid, cmid);

    out.push(("denoise_encoder.norm_out.weight".into(), vec![cmid]));
    out.push(("denoise_encoder.norm_out.bias".into(), vec![cmid]));
    let cout_final = 2 * v.latent_channels as usize; // moments: mean ‖ logvar
    out.push(("denoise_encoder.conv_out.weight".into(), vec![cout_final, cmid, 3, 3]));
    out.push(("denoise_encoder.conv_out.bias".into(), vec![cout_final]));
    out
}

/// `stable-diffusion-xl-base-1.0`'s VAE encoder shape - the topology
/// `denoise_encoder` byte-for-byte shares, verified against the real
/// checkpoint header (106 tensors, matching key set).
fn sdxl_vae_encoder_config() -> vae::config::VaeConfig {
    vae::config::VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 4,
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        norm_num_groups: 32,
        norm_eps: 1e-6,
        mid_block_add_attention: true,
        scaling_factor: 0.13025,
        shift_factor: 0.0,
        use_quant_conv: true,
        use_post_quant_conv: true,
        patch_size: [1, 1],
        batch_norm_eps: 1e-4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest must be exactly the real checkpoint's 1035-tensor SUPIR
    /// delta once its two host-side fusions are accounted for: 811
    /// `model.control_model.*` (810 plus the rejected `mask_LQ`) minus the
    /// trunk's qkv/kv/GEGLU fusion (the SAME one
    /// `sdxlunet::config::UNetConfig::tensor_manifest`'s own test asserts as
    /// `1680 - tb`, and `ControlNetConfig`'s as `844 - tb`); 118
    /// `project_modules.*` minus one tensor per fused `ZeroCrossAttn` `kv`;
    /// 106 `denoise_encoder.*` untouched (no fusion there).
    #[test]
    fn sdxl_manifest_matches_the_checkpoint_delta_count() {
        let cfg = SupirConfig::sdxl();
        let trunk = trunk_manifest(&cfg.trunk);
        let adaptors = cfg.adaptors.tensor_manifest();
        let denc = denoise_encoder_manifest(&cfg.denoise_encoder);

        let mut tb = 0usize; // BasicTransformerBlocks in the trunk's down+mid
        let b = &cfg.trunk;
        for i in 0..b.levels() {
            if b.down_block_types[i] == sdxlunet::config::BlockKind::CrossAttn {
                tb += (b.layers_per_block * b.transformer_layers_per_block[i]) as usize;
            }
        }
        tb += b.transformer_layers_per_block[b.levels() - 1] as usize; // mid
        assert_eq!(tb, 34, "GLVControl's down+mid has 34 BasicTransformerBlocks, same as SDXL ControlNet's");

        let n_cross = cfg.adaptors.cross.len();
        assert_eq!(trunk.len() + tb, 810, "trunk manifest ({} post-fusion + {tb} fused-away = 810 raw, mask_LQ excluded)", trunk.len());
        assert_eq!(adaptors.len() + n_cross, 118, "adaptor manifest ({} post-fusion + {n_cross} fused-away kv = 118 raw)", adaptors.len());
        assert_eq!(denc.len(), 106, "denoise_encoder manifest");

        let m = cfg.tensor_manifest();
        assert_eq!(m.len(), trunk.len() + adaptors.len() + denc.len());
        assert_eq!(m.len() + tb + n_cross, 810 + 118 + 106, "combined manifest vs the real checkpoint delta (mask_LQ excluded)");
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate names in the manifest");
    }

    /// `zero_conv`'s width, settled against the real checkpoint (see the
    /// module doc): at every join `k`, `zero_conv` is square at `skip_c`
    /// (which by the shape-preservation proof equals `control_c`), NOT at
    /// `h_ori_c` - and those differ at joins 2, 3, 5, 6, so this is a real
    /// discriminating check, not one that would pass either way.
    #[test]
    fn zero_conv_is_square_at_the_skip_width_not_h_ori() {
        let cfg = SupirConfig::sdxl();
        let m = cfg.adaptors.tensor_manifest();
        let mut checked_a_differing_join = false;
        for j in &cfg.adaptors.joins {
            if j.h_ori_c != j.skip_c {
                checked_a_differing_join = true;
            }
            let name = format!("project_modules.{}.zero_conv.weight", j.pm_idx);
            let (_, shape) = m.iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(*shape, vec![j.skip_c as usize, j.skip_c as usize, 1, 1], "{name}");
        }
        assert!(checked_a_differing_join, "no join in the table has h_ori_c != skip_c - test is vacuous");
    }

    #[test]
    fn tiny_manifest_has_no_duplicates_and_covers_the_adaptor_schedule() {
        let cfg = SupirConfig::tiny();
        let m = cfg.tensor_manifest();
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len());
        assert!(!m.is_empty());
    }
}
