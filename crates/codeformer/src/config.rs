// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CodeFormer configuration: what the VQ autoencoder does **not** cover.
//!
//! `crates/vqgan` already owns the encoder / codebook / generator schedule
//! ([`VqganConfig`]) and this config embeds it verbatim — CodeFormer *is* a
//! `VQAutoEncoder` subclass (`codeformer_arch.py:159`), constructed with exactly
//! `VQAutoEncoder(512, 64, [1,2,2,4,4,8], 'nearest', 2, [16], codebook_size)`.
//!
//! What is added here, mirroring `codeformer_arch.py`:
//!
//! * the **code-prediction Transformer** — `feat_emb` (a 256→512 linear over the
//!   flattened encoder output), a learned `position_emb[T, 512]`, nine
//!   `TransformerSALayer`s, and the `idx_pred_layer` head (LayerNorm + a
//!   *biasless* 512→1024 linear) whose argmax replaces the nearest-neighbour
//!   codebook search;
//! * the **controllable feature transformation** (`Fuse_sft_block`) at the four
//!   `connect_list` resolutions, each a `ResBlock(2C→C)` over the channel
//!   concatenation of the encoder and generator features followed by two
//!   `Conv3×3 → LeakyReLU(0.2) → Conv3×3` towers producing a `scale` and a
//!   `shift`;
//! * the **fidelity dial `w`** — see [`CodeFormerConfig`]'s note on direction.
//!
//! The fuse tap tables are transcribed from `codeformer_arch.py:200-206` rather
//! than derived: the reference hardcodes them, and a derivation that happened to
//! agree today would be a second source of truth tomorrow. [`FUSE_TAPS`] carries
//! all six entries the reference declares; a test checks every one against the
//! block schedule's own channel/resolution bookkeeping, so a transcription slip
//! fails loudly instead of fusing the wrong pair of features.

use vqgan::config::{Block, VqganConfig};

/// One controllable-feature-transformation tap, transcribed from
/// `codeformer_arch.py`'s `channels` / `fuse_encoder_block` /
/// `fuse_generator_block` dictionaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuseTap {
    /// Spatial size the tap lives at (the reference's dict key, a string there).
    pub size: u32,
    /// Channel count at that size — the `in_ch`/`out_ch` of its `Fuse_sft_block`.
    pub channels: u32,
    /// `encoder.blocks` index whose output is the *encoder* feature.
    pub enc_block: usize,
    /// `generator.blocks` index **after** which the fuse is applied.
    pub gen_block: usize,
}

/// Every tap the reference declares, coarsest first. `connect_list` selects a
/// subset; the released `codeformer.pth` carries `fuse_convs_dict` for
/// `['32','64','128','256']` only (72 tensors = 4 × 18).
pub const FUSE_TAPS: [FuseTap; 6] = [
    FuseTap { size: 16, channels: 512, enc_block: 18, gen_block: 6 },
    FuseTap { size: 32, channels: 256, enc_block: 14, gen_block: 9 },
    FuseTap { size: 64, channels: 256, enc_block: 11, gen_block: 12 },
    FuseTap { size: 128, channels: 128, enc_block: 8, gen_block: 15 },
    FuseTap { size: 256, channels: 128, enc_block: 5, gen_block: 18 },
    FuseTap { size: 512, channels: 64, enc_block: 2, gen_block: 21 },
];

/// `CodeFormer.__init__` hyperparameters plus the VQ autoencoder it subclasses.
#[derive(Clone, Debug, PartialEq)]
pub struct CodeFormerConfig {
    /// The `VQAutoEncoder` half — owned by `crates/vqgan`, not restated here.
    pub vqgan: VqganConfig,
    /// Transformer width (`dim_embd`).
    pub dim_embd: u32,
    pub n_head: u32,
    pub n_layers: u32,
    /// `dim_mlp = dim_embd * 2` in the reference constructor — note that
    /// `TransformerSALayer`'s own default is 2048 and is **overridden**.
    pub dim_mlp: u32,
    /// Number of latent positions the `position_emb` parameter covers. The
    /// reference's 256 = 16×16, i.e. a 512×512 input downscaled 32-fold, and
    /// the parameter is not interpolated — the model is fixed to that size.
    pub latent_size: u32,
    /// Spatial sizes the CFT connects, coarsest first.
    pub connect: Vec<u32>,
    /// `nn.LayerNorm` epsilon (torch's default).
    pub ln_eps: f32,
    /// `nn.LeakyReLU(0.2)` negative slope inside the scale/shift towers.
    pub leaky_slope: f32,
}

impl CodeFormerConfig {
    /// The released `codeformer.pth` preset — `inference_codeformer.py:78`:
    /// `CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
    /// connect_list=['32','64','128','256'])`.
    pub fn codeformer() -> CodeFormerConfig {
        CodeFormerConfig {
            vqgan: VqganConfig::codeformer(),
            dim_embd: 512,
            n_head: 8,
            n_layers: 9,
            dim_mlp: 1024,
            latent_size: 256,
            connect: vec![32, 64, 128, 256],
            ln_eps: 1e-5,
            leaky_slope: 0.2,
        }
    }

    /// Attention head width. 512 / 8 = 64.
    pub fn head_dim(&self) -> u32 {
        self.dim_embd / self.n_head
    }

    /// The taps `connect` selects, ordered by **generator** block index — the
    /// order the generator walk applies them in.
    pub fn taps(&self) -> Vec<FuseTap> {
        let mut v: Vec<FuseTap> =
            FUSE_TAPS.iter().copied().filter(|t| self.connect.contains(&t.size)).collect();
        assert_eq!(
            v.len(),
            self.connect.len(),
            "connect_list {:?} names a size with no tap in FUSE_TAPS",
            self.connect
        );
        v.sort_by_key(|t| t.gen_block);
        v
    }

    /// Input image size this config's `position_emb` fixes the model to.
    pub fn img_size(&self) -> u32 {
        let side = (self.latent_size as f64).sqrt() as u32;
        assert_eq!(side * side, self.latent_size, "latent_size {} is not square", self.latent_size);
        side * self.vqgan.downscale()
    }

    /// The `fuse_convs_dict.{size}` prefix for a tap.
    pub fn fuse_prefix(t: &FuseTap) -> String {
        format!("fuse_convs_dict.{}", t.size)
    }

    /// The `ft_layers.{i}` prefix for a transformer layer.
    pub fn layer_prefix(i: usize) -> String {
        format!("ft_layers.{i}")
    }

    /// Every tensor in the **checkpoint**, with its expected shape: the VQGAN
    /// 329 plus CodeFormer's own. `codeformer.pth` carries exactly 515.
    ///
    /// This is the contract [`crate::import`] validates against the source in
    /// both directions. The tensors the *forward graph* reads are
    /// [`CodeFormerConfig::runtime_manifest`], which differs by the fused
    /// `in_proj` split.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut m = self.vqgan.tensor_manifest();
        let (e, mlp) = (self.dim_embd as usize, self.dim_mlp as usize);

        m.push(("position_emb".into(), vec![self.latent_size as usize, e]));
        m.push(("feat_emb.weight".into(), vec![e, self.vqgan.emb_dim as usize]));
        m.push(("feat_emb.bias".into(), vec![e]));

        for i in 0..self.n_layers as usize {
            let p = CodeFormerConfig::layer_prefix(i);
            m.push((format!("{p}.self_attn.in_proj_weight"), vec![3 * e, e]));
            m.push((format!("{p}.self_attn.in_proj_bias"), vec![3 * e]));
            m.push((format!("{p}.self_attn.out_proj.weight"), vec![e, e]));
            m.push((format!("{p}.self_attn.out_proj.bias"), vec![e]));
            m.push((format!("{p}.linear1.weight"), vec![mlp, e]));
            m.push((format!("{p}.linear1.bias"), vec![mlp]));
            m.push((format!("{p}.linear2.weight"), vec![e, mlp]));
            m.push((format!("{p}.linear2.bias"), vec![e]));
            for n in ["norm1", "norm2"] {
                m.push((format!("{p}.{n}.weight"), vec![e]));
                m.push((format!("{p}.{n}.bias"), vec![e]));
            }
        }

        // idx_pred_layer = Sequential(LayerNorm(512), Linear(512, 1024, bias=False)).
        m.push(("idx_pred_layer.0.weight".into(), vec![e]));
        m.push(("idx_pred_layer.0.bias".into(), vec![e]));
        m.push(("idx_pred_layer.1.weight".into(), vec![self.vqgan.codebook_size as usize, e]));

        for t in self.taps() {
            let p = CodeFormerConfig::fuse_prefix(&t);
            let c = t.channels;
            // `encode_enc = ResBlock(2*in_ch, out_ch)`: cin != cout, so it has
            // the 1×1 `conv_out` shortcut. Exactly the VQGAN ResBlock tensor
            // set, so it is spelled by the same helper the VQGAN config uses.
            vqgan::config::block_tensors(
                &format!("{p}.encode_enc"),
                &Block::Res { cin: 2 * c, cout: c },
                &mut m,
            );
            for tower in ["scale", "shift"] {
                for idx in [0usize, 2] {
                    m.push((
                        format!("{p}.{tower}.{idx}.weight"),
                        vec![c as usize, c as usize, 3, 3],
                    ));
                    m.push((format!("{p}.{tower}.{idx}.bias"), vec![c as usize]));
                }
            }
        }
        m
    }

    /// Every tensor the **forward graph** reads.
    ///
    /// Identical to [`CodeFormerConfig::tensor_manifest`] except at the
    /// attention projection. `nn.MultiheadAttention` fuses q|k|v into one
    /// `in_proj_weight[3E, E]`, but this port cannot use it whole: the
    /// reference adds the position embedding to **q and k only**
    /// (`with_pos_embed(tgt2, query_pos)`), so q/k and v read *different*
    /// inputs. The fused weight is therefore split at import into a contiguous
    /// `qk[2E, E]` (q rows then k rows — already adjacent in the checkpoint) and
    /// a `v[E, E]`, which is also what the attention kernels want: `q` and `k`
    /// share one buffer at stride `2E` with `q_off = 0, k_off = E`, and `v` is
    /// its own buffer at stride `E`.
    pub fn runtime_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let e = self.dim_embd as usize;
        let mut m: Vec<(String, Vec<usize>)> = self
            .tensor_manifest()
            .into_iter()
            .filter(|(n, _)| !n.ends_with(".self_attn.in_proj_weight"))
            .filter(|(n, _)| !n.ends_with(".self_attn.in_proj_bias"))
            .collect();
        for i in 0..self.n_layers as usize {
            let p = CodeFormerConfig::layer_prefix(i);
            m.push((format!("{p}.self_attn.qk.weight"), vec![2 * e, e]));
            m.push((format!("{p}.self_attn.qk.bias"), vec![2 * e]));
            m.push((format!("{p}.self_attn.v.weight"), vec![e, e]));
            m.push((format!("{p}.self_attn.v.bias"), vec![e]));
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transcribed tap table must agree with the block schedule it indexes:
    /// the encoder block's output channels, the generator block's output
    /// channels, and the spatial size both sit at.
    #[test]
    fn fuse_taps_match_the_block_schedule() {
        let cfg = CodeFormerConfig::codeformer();
        let enc = cfg.vqgan.encoder_blocks();
        let gen = cfg.vqgan.generator_blocks();
        let img = cfg.img_size();
        assert_eq!(img, 512);

        // Walk each net once, recording (out_channels, out_size) per block.
        let walk = |blocks: &[Block], start: u32| -> Vec<(u32, u32)> {
            let mut res = start;
            blocks
                .iter()
                .map(|b| {
                    match b {
                        Block::Down { .. } => res /= 2,
                        Block::Up { .. } => res *= 2,
                        _ => {}
                    }
                    (b.out_channels(), res)
                })
                .collect()
        };
        let enc_shape = walk(&enc, img);
        let gen_shape = walk(&gen, img / cfg.vqgan.downscale());

        for t in FUSE_TAPS {
            assert_eq!(
                enc_shape[t.enc_block],
                (t.channels, t.size),
                "encoder tap {} (block {})",
                t.size,
                t.enc_block
            );
            assert_eq!(
                gen_shape[t.gen_block],
                (t.channels, t.size),
                "generator tap {} (block {})",
                t.size,
                t.gen_block
            );
        }
    }

    #[test]
    fn taps_are_ordered_by_generator_block() {
        let t = CodeFormerConfig::codeformer().taps();
        assert_eq!(t.iter().map(|t| t.size).collect::<Vec<_>>(), vec![32, 64, 128, 256]);
        assert_eq!(t.iter().map(|t| t.gen_block).collect::<Vec<_>>(), vec![9, 12, 15, 18]);
        assert_eq!(t.iter().map(|t| t.enc_block).collect::<Vec<_>>(), vec![14, 11, 8, 5]);
    }

    /// 515 is the tensor count the reference `load_state_dict` reports for
    /// `codeformer.pth` (329 VQGAN + 186 CodeFormer).
    #[test]
    fn manifest_is_the_whole_checkpoint() {
        let cfg = CodeFormerConfig::codeformer();
        let m = cfg.tensor_manifest();
        assert_eq!(m.len(), 515, "checkpoint tensor count");
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate tensor name");
        assert!(names.contains("encoder.blocks.24.weight"));
        assert!(names.contains("position_emb"));
        assert!(names.contains("ft_layers.8.self_attn.in_proj_weight"));
        assert!(names.contains("idx_pred_layer.1.weight"));
        assert!(names.contains("fuse_convs_dict.256.encode_enc.conv_out.weight"));
        assert!(names.contains("fuse_convs_dict.32.shift.2.bias"));
        assert!(!names.contains("idx_pred_layer.1.bias"), "the head linear has no bias");
    }

    #[test]
    fn runtime_manifest_splits_the_fused_in_proj() {
        let cfg = CodeFormerConfig::codeformer();
        let r = cfg.runtime_manifest();
        // -18 fused (2 per layer × 9), +36 split (4 per layer × 9).
        assert_eq!(r.len(), 515 - 18 + 36);
        let by: std::collections::HashMap<&str, &Vec<usize>> =
            r.iter().map(|(n, s)| (n.as_str(), s)).collect();
        assert_eq!(by["ft_layers.0.self_attn.qk.weight"], &vec![1024, 512]);
        assert_eq!(by["ft_layers.0.self_attn.qk.bias"], &vec![1024]);
        assert_eq!(by["ft_layers.0.self_attn.v.weight"], &vec![512, 512]);
        assert!(!by.contains_key("ft_layers.0.self_attn.in_proj_weight"));
    }
}
