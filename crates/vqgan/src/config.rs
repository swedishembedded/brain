// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! VQGAN configuration and the block schedule it determines.
//!
//! Mirrors `basicsr/archs/vqgan_arch.py`'s `VQAutoEncoder` /`Encoder`
//! /`Generator` constructors exactly. Both nets are a **flat `nn.ModuleList`**
//! (`encoder.blocks.{i}` / `generator.blocks.{i}`), so the config's job is to
//! reproduce that list — index for index — because the checkpoint's tensor
//! names are positional.
//!
//! The same block family also describes LlamaGen's `VQModel` (Janus-Pro's
//! image tokenizer, [`VqganConfig::llamagen_vq16`]). Its differences are
//! config knobs, not a second schedule: one more residual block per decoder
//! level ([`VqganConfig::dec_res_blocks`]), a SiLU after each head GroupNorm
//! ([`VqganConfig::head_act`]), 1×1 `quant_conv`/`post_quant_conv` bridges
//! between a wide latent and a narrow codebook ([`VqganConfig::z_channels`]),
//! an L2-normalised codebook search ([`VqganConfig::codebook_l2_norm`]) and
//! `beta` on the commitment term ([`VqganConfig::beta_on`]). LlamaGen names
//! its modules hierarchically (`encoder.conv_blocks.{level}.res.{j}`), so
//! [`VqganConfig::llamagen_block_paths`] records, for every flat index, the
//! reference module it is - the one source of truth [`crate::import`]'s name
//! map is derived from.
//!
//! Note that `attn_resolutions` is resolved against `img_size` at
//! **construction** time: the AttnBlock positions are frozen into the module
//! list and do not change with the runtime input size.

/// One entry of the reference `nn.ModuleList`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Block {
    /// `nn.Conv2d(cin, cout, 3, stride 1, pad 1)` — the `conv_in`/`conv_out`
    /// heads. Weights at `{prefix}.{weight,bias}`.
    Conv { cin: u32, cout: u32 },
    /// `ResBlock(cin, cout)`: norm1→swish→conv1→norm2→swish→conv2 plus a 1×1
    /// `conv_out` shortcut when `cin != cout`.
    Res { cin: u32, cout: u32 },
    /// `AttnBlock(c)`: single-head spatial self-attention, scale `c^-0.5`.
    Attn { c: u32 },
    /// `nn.Conv2d(cin, cout, 1)` - LlamaGen's `quant_conv`/`post_quant_conv`
    /// bridges between the latent and the codebook width.
    Proj { cin: u32, cout: u32 },
    /// `Downsample(c)`: `F.pad(x,(0,1,0,1))` then `Conv2d(c,c,3,stride 2,pad 0)`
    /// at `{prefix}.conv`. Halves H and W.
    Down { c: u32 },
    /// `Upsample(c)`: nearest-2× interpolate then `Conv2d(c,c,3,1,1)` at
    /// `{prefix}.conv`. Doubles H and W.
    Up { c: u32 },
    /// The head `GroupNorm(32, c, eps 1e-6)`, followed by a SiLU when `silu`.
    /// basicsr's VQGAN has **no activation** there (unlike the diffusers VAE
    /// head, which is norm→SiLU→conv); LlamaGen's does
    /// (`norm_out → nonlinearity → conv_out`).
    Norm { c: u32, silu: bool },
}

impl Block {
    /// Channel count of this block's output.
    pub fn out_channels(&self) -> u32 {
        match *self {
            Block::Conv { cout, .. } | Block::Res { cout, .. } | Block::Proj { cout, .. } => cout,
            Block::Attn { c } | Block::Down { c } | Block::Up { c } | Block::Norm { c, .. } => c,
        }
    }

    /// The reference class name, as recorded in the golden manifest's topology.
    pub fn class(&self) -> &'static str {
        match self {
            Block::Conv { .. } | Block::Proj { .. } => "Conv2d",
            Block::Res { .. } => "ResBlock",
            Block::Attn { .. } => "AttnBlock",
            Block::Down { .. } => "Downsample",
            Block::Up { .. } => "Upsample",
            Block::Norm { .. } => "GroupNorm",
        }
    }
}

/// Which VQ loss term [`VqganConfig::beta`] weights. The two released
/// families disagree, and a finite-difference check cannot tell them apart
/// (it gates the backward against whatever forward is emitted), so the
/// placement is configuration, pinned by reading each reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BetaTerm {
    /// `beta · ||sg[z] - q||²`, the term that reaches the CODEBOOK. What
    /// `basicsr`'s `vqgan_arch.py:55` executes - its own line-29 comment
    /// calls `beta` the commitment cost, but the comment and the code
    /// disagree there, and the code is what trained the released weights.
    Codebook,
    /// `beta · ||z - sg[q]||²`, the term that reaches the ENCODER - the VQ-VAE
    /// paper's commitment cost, and LlamaGen's
    /// `commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)`.
    Commitment,
}

/// `VQAutoEncoder` hyperparameters.
#[derive(Clone, Debug, PartialEq)]
pub struct VqganConfig {
    pub in_channels: u32,
    pub out_channels: u32,
    /// Base width; per-level channels are `nf * ch_mult[i]`.
    pub nf: u32,
    pub ch_mult: Vec<u32>,
    /// Residual blocks per resolution level in the ENCODER.
    pub res_blocks: u32,
    /// Residual blocks per resolution level in the GENERATOR. `basicsr` uses
    /// `res_blocks` on both sides; LlamaGen's `Decoder` runs
    /// `num_res_blocks + 1`.
    pub dec_res_blocks: u32,
    /// Spatial resolutions (at `img_size` scale) that carry an `AttnBlock`
    /// after every residual block.
    pub attn_resolutions: Vec<u32>,
    /// Resolution the module list was constructed for. Frozen into the block
    /// schedule; the runtime input may differ.
    pub img_size: u32,
    pub codebook_size: u32,
    pub emb_dim: u32,
    /// `Some(z)`: the encoder ends at `z` channels and a 1×1 `quant_conv`
    /// (`z → emb_dim`) / `post_quant_conv` (`emb_dim → z`) pair bridges it to
    /// the codebook (LlamaGen: 256 ↔ 8). `None`: the encoder's last conv
    /// emits `emb_dim` directly and the generator reads it (`basicsr`).
    pub z_channels: Option<u32>,
    /// A SiLU between each head GroupNorm and its conv ([`Block::Norm`]).
    pub head_act: bool,
    /// L2-normalise the query rows AND the codebook before the
    /// nearest-neighbour search, and gather decode-side codes from the
    /// normalised codebook (LlamaGen `codebook_l2_norm=True`: every use of
    /// `embedding.weight` goes through `F.normalize`, so the stored table is
    /// never read raw).
    pub codebook_l2_norm: bool,
    /// Weight of one of the two VQ loss terms (training only; unused by the
    /// forward). Which one is [`VqganConfig::beta_on`].
    pub beta: f32,
    pub beta_on: BetaTerm,
    pub norm_groups: u32,
    pub norm_eps: f32,
}

impl VqganConfig {
    /// The CodeFormer / `vqgan_code1024` preset — `codeformer_arch.py:166`:
    /// `VQAutoEncoder(512, 64, [1,2,2,4,4,8], 'nearest', 2, [16], 1024, 256)`.
    pub fn codeformer() -> VqganConfig {
        VqganConfig {
            in_channels: 3,
            out_channels: 3,
            nf: 64,
            ch_mult: vec![1, 2, 2, 4, 4, 8],
            res_blocks: 2,
            dec_res_blocks: 2,
            attn_resolutions: vec![16],
            img_size: 512,
            codebook_size: 1024,
            emb_dim: 256,
            z_channels: None,
            head_act: false,
            codebook_l2_norm: false,
            beta: 0.25,
            beta_on: BetaTerm::Codebook,
            norm_groups: 32,
            norm_eps: 1e-6,
        }
    }

    /// LlamaGen's `VQ_16` (`tokenizer/tokenizer_image/vq_model.py`), which
    /// Janus-Pro vendors verbatim as its image tokenizer (`gen_vision_model`,
    /// built with the defaults): `ch=128`, `ch_mult=[1,1,2,2,4]`,
    /// `num_res_blocks=2` (the decoder runs 3), `z_channels=256`, a
    /// 16384×8 L2-normalised codebook, `commit_loss_beta=0.25`.
    ///
    /// LlamaGen places an `AttnBlock` after every residual block of the LAST
    /// (lowest-resolution) level whatever the input size; here that is
    /// `attn_resolutions = [img_size / 16]`, with `img_size` Janus-Pro's 384.
    pub fn llamagen_vq16() -> VqganConfig {
        VqganConfig {
            in_channels: 3,
            out_channels: 3,
            nf: 128,
            ch_mult: vec![1, 1, 2, 2, 4],
            res_blocks: 2,
            dec_res_blocks: 3,
            attn_resolutions: vec![24],
            img_size: 384,
            codebook_size: 16384,
            emb_dim: 8,
            z_channels: Some(256),
            head_act: true,
            codebook_l2_norm: true,
            beta: 0.25,
            beta_on: BetaTerm::Commitment,
            norm_groups: 32,
            norm_eps: 1e-6,
        }
    }

    /// Total spatial downscale of the encoder: one `Downsample` per level
    /// except the last. `[1,2,2,4,4,8]` → 32.
    pub fn downscale(&self) -> u32 {
        1 << (self.ch_mult.len() as u32 - 1)
    }

    /// Width of the encoder's last conv and the generator's first: the latent
    /// before `quant_conv` and after `post_quant_conv`.
    fn latent_channels(&self) -> u32 {
        self.z_channels.unwrap_or(self.emb_dim)
    }

    /// `Encoder.blocks`, index for index.
    pub fn encoder_blocks(&self) -> Vec<Block> {
        self.encoder_schedule().into_iter().map(|(b, _)| b).collect()
    }

    /// `Generator.blocks`, index for index.
    pub fn generator_blocks(&self) -> Vec<Block> {
        self.generator_schedule().into_iter().map(|(b, _)| b).collect()
    }

    /// Every block of both nets as `(flat prefix, LlamaGen module path)`:
    /// `("encoder.blocks.3", "encoder.conv_blocks.0.downsample")`,
    /// `("encoder.blocks.23", "quant_conv")`,
    /// `("generator.blocks.0", "post_quant_conv")`, …. The flat prefix is what
    /// [`Self::tensor_manifest`] and the forward's taps use; the path is where
    /// the reference keeps the same module.
    pub fn llamagen_block_paths(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (net, sched) in [("encoder", self.encoder_schedule()), ("generator", self.generator_schedule())] {
            for (i, (_, path)) in sched.into_iter().enumerate() {
                out.push((format!("{net}.blocks.{i}"), path));
            }
        }
        out
    }

    /// The encoder schedule, each block beside its LlamaGen module path.
    fn encoder_schedule(&self) -> Vec<(Block, String)> {
        let n = self.ch_mult.len();
        let mut curr_res = self.img_size;
        let mut out = vec![(Block::Conv { cin: self.in_channels, cout: self.nf }, "encoder.conv_in".to_string())];
        // `in_ch_mult = (1,) + ch_mult`
        for i in 0..n {
            let mult_in = if i == 0 { 1 } else { self.ch_mult[i - 1] };
            let mut cin = self.nf * mult_in;
            let cout = self.nf * self.ch_mult[i];
            for j in 0..self.res_blocks {
                out.push((Block::Res { cin, cout }, format!("encoder.conv_blocks.{i}.res.{j}")));
                cin = cout;
                if self.attn_resolutions.contains(&curr_res) {
                    out.push((Block::Attn { c: cin }, format!("encoder.conv_blocks.{i}.attn.{j}")));
                }
            }
            if i != n - 1 {
                out.push((Block::Down { c: cin }, format!("encoder.conv_blocks.{i}.downsample")));
                curr_res /= 2;
            }
        }
        let mid = self.nf * self.ch_mult[n - 1];
        out.push((Block::Res { cin: mid, cout: mid }, "encoder.mid.0".into()));
        out.push((Block::Attn { c: mid }, "encoder.mid.1".into()));
        out.push((Block::Res { cin: mid, cout: mid }, "encoder.mid.2".into()));
        out.push((Block::Norm { c: mid, silu: self.head_act }, "encoder.norm_out".into()));
        out.push((Block::Conv { cin: mid, cout: self.latent_channels() }, "encoder.conv_out".into()));
        if let Some(z) = self.z_channels {
            out.push((Block::Proj { cin: z, cout: self.emb_dim }, "quant_conv".into()));
        }
        out
    }

    /// The generator schedule, each block beside its LlamaGen module path.
    /// LlamaGen's `decoder.conv_blocks` is indexed in execution order, so its
    /// level 0 is the lowest-resolution one (`ch_mult[n-1]`).
    fn generator_schedule(&self) -> Vec<(Block, String)> {
        let n = self.ch_mult.len();
        let mut cin = self.nf * self.ch_mult[n - 1];
        let mut curr_res = self.img_size >> (n as u32 - 1);
        let mut out = Vec::new();
        if let Some(z) = self.z_channels {
            out.push((Block::Proj { cin: self.emb_dim, cout: z }, "post_quant_conv".to_string()));
        }
        out.push((Block::Conv { cin: self.latent_channels(), cout: cin }, "decoder.conv_in".into()));
        out.push((Block::Res { cin, cout: cin }, "decoder.mid.0".into()));
        out.push((Block::Attn { c: cin }, "decoder.mid.1".into()));
        out.push((Block::Res { cin, cout: cin }, "decoder.mid.2".into()));
        for (level, i) in (0..n).rev().enumerate() {
            let cout = self.nf * self.ch_mult[i];
            for j in 0..self.dec_res_blocks {
                out.push((Block::Res { cin, cout }, format!("decoder.conv_blocks.{level}.res.{j}")));
                cin = cout;
                if self.attn_resolutions.contains(&curr_res) {
                    out.push((Block::Attn { c: cin }, format!("decoder.conv_blocks.{level}.attn.{j}")));
                }
            }
            if i != 0 {
                out.push((Block::Up { c: cin }, format!("decoder.conv_blocks.{level}.upsample")));
                curr_res *= 2;
            }
        }
        out.push((Block::Norm { c: cin, silu: self.head_act }, "decoder.norm_out".into()));
        out.push((Block::Conv { cin, cout: self.out_channels }, "decoder.conv_out".into()));
        out
    }

    /// Every tensor the forward graph reads, with its expected shape — the
    /// contract [`crate::import`] validates in both directions.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut m = Vec::new();
        for (net, blocks) in
            [("encoder", self.encoder_blocks()), ("generator", self.generator_blocks())]
        {
            for (i, b) in blocks.iter().enumerate() {
                let p = format!("{net}.blocks.{i}");
                block_tensors(&p, b, &mut m);
            }
        }
        m.push((
            "quantize.embedding.weight".to_string(),
            vec![self.codebook_size as usize, self.emb_dim as usize],
        ));
        m
    }

    /// The LlamaGen/Janus name map: every [`Self::tensor_manifest`] name
    /// beside the name LlamaGen's `VQModel.state_dict()` gives the same
    /// tensor, in manifest order.
    ///
    /// Derived from [`Self::llamagen_block_paths`], never written out by hand:
    /// a block's flat prefix becomes its module path, and the one leaf the two
    /// families spell differently - the resnet's 1×1 projection shortcut,
    /// `conv_out` in `basicsr`, `nin_shortcut` in LlamaGen - is renamed.
    pub fn llamagen_tensor_names(&self) -> Vec<(String, String)> {
        let blocks = self.encoder_blocks().into_iter().chain(self.generator_blocks());
        let mut out = Vec::new();
        for ((flat, path), b) in self.llamagen_block_paths().into_iter().zip(blocks) {
            let mut m = Vec::new();
            block_tensors(&flat, &b, &mut m);
            for (name, _) in m {
                let leaf = &name[flat.len() + 1..];
                let leaf = match (b, leaf.strip_prefix("conv_out.")) {
                    (Block::Res { .. }, Some(rest)) => format!("nin_shortcut.{rest}"),
                    _ => leaf.to_string(),
                };
                out.push((name, format!("{path}.{leaf}")));
            }
        }
        let cb = "quantize.embedding.weight".to_string();
        out.push((cb.clone(), cb));
        out
    }
}

/// Push one block's `(name, shape)` pairs, in reference declaration order.
///
/// Public because CodeFormer's `Fuse_sft_block` embeds a plain VQGAN `ResBlock`
/// (`fuse_convs_dict.{size}.encode_enc`), so `crates/codeformer`'s manifest spells
/// it with this function instead of a second copy of the naming convention.
pub fn block_tensors(p: &str, b: &Block, m: &mut Vec<(String, Vec<usize>)>) {
    let conv = |m: &mut Vec<(String, Vec<usize>)>, name: String, cin: u32, cout: u32, k: usize| {
        m.push((format!("{name}.weight"), vec![cout as usize, cin as usize, k, k]));
        m.push((format!("{name}.bias"), vec![cout as usize]));
    };
    let norm = |m: &mut Vec<(String, Vec<usize>)>, name: String, c: u32| {
        m.push((format!("{name}.weight"), vec![c as usize]));
        m.push((format!("{name}.bias"), vec![c as usize]));
    };
    match *b {
        Block::Conv { cin, cout } => conv(m, p.to_string(), cin, cout, 3),
        Block::Proj { cin, cout } => conv(m, p.to_string(), cin, cout, 1),
        Block::Res { cin, cout } => {
            norm(m, format!("{p}.norm1"), cin);
            conv(m, format!("{p}.conv1"), cin, cout, 3);
            norm(m, format!("{p}.norm2"), cout);
            conv(m, format!("{p}.conv2"), cout, cout, 3);
            if cin != cout {
                conv(m, format!("{p}.conv_out"), cin, cout, 1);
            }
        }
        Block::Attn { c } => {
            norm(m, format!("{p}.norm"), c);
            for leaf in ["q", "k", "v", "proj_out"] {
                conv(m, format!("{p}.{leaf}"), c, c, 1);
            }
        }
        Block::Down { c } | Block::Up { c } => conv(m, format!("{p}.conv"), c, c, 3),
        Block::Norm { c, .. } => norm(m, p.to_string(), c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block topology recorded in the step-1 golden manifest
    /// (`testdata/restore/vqgan/*/manifest.json`, key `topology`).
    #[test]
    fn schedule_matches_reference_topology() {
        let cfg = VqganConfig::codeformer();
        let enc: Vec<&str> = cfg.encoder_blocks().iter().map(Block::class).collect();
        let gen: Vec<&str> = cfg.generator_blocks().iter().map(Block::class).collect();
        assert_eq!(enc.len(), 25, "encoder block count");
        assert_eq!(gen.len(), 25, "generator block count");
        let want_enc = [
            "Conv2d", "ResBlock", "ResBlock", "Downsample", "ResBlock", "ResBlock", "Downsample",
            "ResBlock", "ResBlock", "Downsample", "ResBlock", "ResBlock", "Downsample", "ResBlock",
            "ResBlock", "Downsample", "ResBlock", "AttnBlock", "ResBlock", "AttnBlock", "ResBlock",
            "AttnBlock", "ResBlock", "GroupNorm", "Conv2d",
        ];
        let want_gen = [
            "Conv2d", "ResBlock", "AttnBlock", "ResBlock", "ResBlock", "AttnBlock", "ResBlock",
            "AttnBlock", "Upsample", "ResBlock", "ResBlock", "Upsample", "ResBlock", "ResBlock",
            "Upsample", "ResBlock", "ResBlock", "Upsample", "ResBlock", "ResBlock", "Upsample",
            "ResBlock", "ResBlock", "GroupNorm", "Conv2d",
        ];
        assert_eq!(enc, want_enc, "encoder topology");
        assert_eq!(gen, want_gen, "generator topology");
    }

    #[test]
    fn manifest_covers_both_nets_and_the_codebook() {
        let m = VqganConfig::codeformer().tensor_manifest();
        // 164 encoder + 164 generator + 1 codebook = the 329 tensors the
        // reference load reports for `vqgan_code1024.pth`.
        assert_eq!(m.len(), 329, "tensor count");
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate tensor name in manifest");
        assert!(names.contains("encoder.blocks.0.weight"));
        assert!(names.contains("encoder.blocks.17.proj_out.bias"));
        assert!(names.contains("encoder.blocks.4.conv_out.weight"));
        assert!(names.contains("generator.blocks.8.conv.weight"));
        assert!(names.contains("quantize.embedding.weight"));
    }

    /// LlamaGen `VQ_16` (`vq_model.py`): the encoder runs 2 resnets per level
    /// with attention after each at the last level and ends in `quant_conv`;
    /// the decoder opens with `post_quant_conv` and runs 3 resnets per level;
    /// both heads are GroupNorm → SiLU → conv.
    #[test]
    fn llamagen_vq16_schedule_matches_the_reference() {
        let cfg = VqganConfig::llamagen_vq16();
        let enc = cfg.encoder_blocks();
        let gen = cfg.generator_blocks();
        let enc_cls: Vec<&str> = enc.iter().map(Block::class).collect();
        let gen_cls: Vec<&str> = gen.iter().map(Block::class).collect();
        let want_enc = [
            "Conv2d", "ResBlock", "ResBlock", "Downsample", "ResBlock", "ResBlock", "Downsample",
            "ResBlock", "ResBlock", "Downsample", "ResBlock", "ResBlock", "Downsample", "ResBlock",
            "AttnBlock", "ResBlock", "AttnBlock", "ResBlock", "AttnBlock", "ResBlock", "GroupNorm",
            "Conv2d", "Conv2d",
        ];
        let want_gen = [
            "Conv2d", "Conv2d", "ResBlock", "AttnBlock", "ResBlock", "ResBlock", "AttnBlock",
            "ResBlock", "AttnBlock", "ResBlock", "AttnBlock", "Upsample", "ResBlock", "ResBlock",
            "ResBlock", "Upsample", "ResBlock", "ResBlock", "ResBlock", "Upsample", "ResBlock",
            "ResBlock", "ResBlock", "Upsample", "ResBlock", "ResBlock", "ResBlock", "GroupNorm",
            "Conv2d",
        ];
        assert_eq!(enc_cls, want_enc, "encoder topology");
        assert_eq!(gen_cls, want_gen, "generator topology");
        assert_eq!(enc[21], Block::Conv { cin: 512, cout: 256 }, "encoder conv_out -> z_channels");
        assert_eq!(enc[22], Block::Proj { cin: 256, cout: 8 }, "quant_conv");
        assert_eq!(gen[0], Block::Proj { cin: 8, cout: 256 }, "post_quant_conv");
        assert_eq!(gen[1], Block::Conv { cin: 256, cout: 512 }, "decoder conv_in");
        assert_eq!(enc[20], Block::Norm { c: 512, silu: true });
        assert_eq!(gen[27], Block::Norm { c: 128, silu: true });
        assert_eq!(cfg.downscale(), 16);
    }

    /// Janus-Pro-7B ships 344 `gen_vision_model.*` tensors: these 343 plus the
    /// `quantize.codebook_used` usage buffer. The name map is a bijection onto
    /// the LlamaGen spelling.
    #[test]
    fn llamagen_name_map_spells_the_reference_state_dict() {
        let cfg = VqganConfig::llamagen_vq16();
        let manifest = cfg.tensor_manifest();
        assert_eq!(manifest.len(), 343, "tensor count");
        let map = cfg.llamagen_tensor_names();
        let flat: Vec<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
        let mapped: Vec<&str> = map.iter().map(|(f, _)| f.as_str()).collect();
        assert_eq!(flat, mapped, "the map covers the manifest, in order");
        let lg: std::collections::HashMap<&str, &str> =
            map.iter().map(|(f, l)| (l.as_str(), f.as_str())).collect();
        assert_eq!(lg.len(), map.len(), "two tensors mapped to one LlamaGen name");
        for (want_lg, want_flat) in [
            ("encoder.conv_in.weight", "encoder.blocks.0.weight"),
            ("encoder.conv_blocks.2.res.0.nin_shortcut.bias", "encoder.blocks.7.conv_out.bias"),
            ("encoder.conv_blocks.3.downsample.conv.weight", "encoder.blocks.12.conv.weight"),
            ("encoder.conv_blocks.4.attn.1.proj_out.weight", "encoder.blocks.16.proj_out.weight"),
            ("encoder.mid.1.q.bias", "encoder.blocks.18.q.bias"),
            ("encoder.norm_out.weight", "encoder.blocks.20.weight"),
            ("quant_conv.weight", "encoder.blocks.22.weight"),
            ("post_quant_conv.bias", "generator.blocks.0.bias"),
            ("decoder.conv_blocks.0.attn.2.v.weight", "generator.blocks.10.v.weight"),
            ("decoder.conv_blocks.1.res.0.nin_shortcut.weight", "generator.blocks.12.conv_out.weight"),
            ("decoder.conv_blocks.3.upsample.conv.bias", "generator.blocks.23.conv.bias"),
            ("decoder.conv_out.weight", "generator.blocks.28.weight"),
            ("quantize.embedding.weight", "quantize.embedding.weight"),
        ] {
            assert_eq!(lg.get(want_lg), Some(&want_flat), "{want_lg}");
        }
    }
}
