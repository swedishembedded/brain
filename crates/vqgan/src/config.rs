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
    /// `Downsample(c)`: `F.pad(x,(0,1,0,1))` then `Conv2d(c,c,3,stride 2,pad 0)`
    /// at `{prefix}.conv`. Halves H and W.
    Down { c: u32 },
    /// `Upsample(c)`: nearest-2× interpolate then `Conv2d(c,c,3,1,1)` at
    /// `{prefix}.conv`. Doubles H and W.
    Up { c: u32 },
    /// The bare head `GroupNorm(32, c, eps 1e-6)`. **No activation follows it**
    /// in VQGAN (unlike the diffusers VAE head, which is norm→SiLU→conv).
    Norm { c: u32 },
}

impl Block {
    /// Channel count of this block's output.
    pub fn out_channels(&self) -> u32 {
        match *self {
            Block::Conv { cout, .. } | Block::Res { cout, .. } => cout,
            Block::Attn { c } | Block::Down { c } | Block::Up { c } | Block::Norm { c } => c,
        }
    }

    /// The reference class name, as recorded in the golden manifest's topology.
    pub fn class(&self) -> &'static str {
        match self {
            Block::Conv { .. } => "Conv2d",
            Block::Res { .. } => "ResBlock",
            Block::Attn { .. } => "AttnBlock",
            Block::Down { .. } => "Downsample",
            Block::Up { .. } => "Upsample",
            Block::Norm { .. } => "GroupNorm",
        }
    }
}

/// `VQAutoEncoder` hyperparameters.
#[derive(Clone, Debug, PartialEq)]
pub struct VqganConfig {
    pub in_channels: u32,
    pub out_channels: u32,
    /// Base width; per-level channels are `nf * ch_mult[i]`.
    pub nf: u32,
    pub ch_mult: Vec<u32>,
    /// Residual blocks per resolution level.
    pub res_blocks: u32,
    /// Spatial resolutions (at `img_size` scale) that carry an `AttnBlock`
    /// after every residual block.
    pub attn_resolutions: Vec<u32>,
    /// Resolution the module list was constructed for. Frozen into the block
    /// schedule; the runtime input may differ.
    pub img_size: u32,
    pub codebook_size: u32,
    pub emb_dim: u32,
    /// Weight of the VQ **codebook** loss (training only; unused by the
    /// forward). Named `beta` after the reference's constructor argument, whose
    /// own comment calls it the *commitment* cost — but `vqgan_arch.py:55`
    /// multiplies it into `torch.mean((z_q - z.detach())**2)`, the `z.detach()`
    /// half, which is the term that reaches the CODEBOOK. The comment and the
    /// code disagree there; `vqgan::train` follows the code.
    pub beta: f32,
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
            attn_resolutions: vec![16],
            img_size: 512,
            codebook_size: 1024,
            emb_dim: 256,
            beta: 0.25,
            norm_groups: 32,
            norm_eps: 1e-6,
        }
    }

    /// Total spatial downscale of the encoder: one `Downsample` per level
    /// except the last. `[1,2,2,4,4,8]` → 32.
    pub fn downscale(&self) -> u32 {
        1 << (self.ch_mult.len() as u32 - 1)
    }

    /// `Encoder.blocks`, index for index.
    pub fn encoder_blocks(&self) -> Vec<Block> {
        let n = self.ch_mult.len();
        let mut curr_res = self.img_size;
        let mut out = vec![Block::Conv { cin: self.in_channels, cout: self.nf }];
        // `in_ch_mult = (1,) + ch_mult`
        for i in 0..n {
            let mult_in = if i == 0 { 1 } else { self.ch_mult[i - 1] };
            let mut cin = self.nf * mult_in;
            let cout = self.nf * self.ch_mult[i];
            for _ in 0..self.res_blocks {
                out.push(Block::Res { cin, cout });
                cin = cout;
                if self.attn_resolutions.contains(&curr_res) {
                    out.push(Block::Attn { c: cin });
                }
            }
            if i != n - 1 {
                out.push(Block::Down { c: cin });
                curr_res /= 2;
            }
        }
        let mid = self.nf * self.ch_mult[n - 1];
        out.push(Block::Res { cin: mid, cout: mid });
        out.push(Block::Attn { c: mid });
        out.push(Block::Res { cin: mid, cout: mid });
        out.push(Block::Norm { c: mid });
        out.push(Block::Conv { cin: mid, cout: self.emb_dim });
        out
    }

    /// `Generator.blocks`, index for index.
    pub fn generator_blocks(&self) -> Vec<Block> {
        let n = self.ch_mult.len();
        let mut cin = self.nf * self.ch_mult[n - 1];
        let mut curr_res = self.img_size >> (n as u32 - 1);
        let mut out = vec![Block::Conv { cin: self.emb_dim, cout: cin }];
        out.push(Block::Res { cin, cout: cin });
        out.push(Block::Attn { c: cin });
        out.push(Block::Res { cin, cout: cin });
        for i in (0..n).rev() {
            let cout = self.nf * self.ch_mult[i];
            for _ in 0..self.res_blocks {
                out.push(Block::Res { cin, cout });
                cin = cout;
                if self.attn_resolutions.contains(&curr_res) {
                    out.push(Block::Attn { c: cin });
                }
            }
            if i != 0 {
                out.push(Block::Up { c: cin });
                curr_res *= 2;
            }
        }
        out.push(Block::Norm { c: cin });
        out.push(Block::Conv { cin, cout: self.out_channels });
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
}

/// Push one block's `(name, shape)` pairs, in reference declaration order.
///
/// Public because CodeFormer's `Fuse_sft_block` embeds a plain VQGAN `ResBlock`
/// (`fuse_convs_dict.{size}.encode_enc`), so `crates/restore`'s manifest spells
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
        Block::Norm { c } => norm(m, p.to_string(), c),
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
}
