// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL ViT vision encoder — GPU forward, built on the shared `model::vit`
//! block builder.
//!
//! Pipeline per image (single frame, `t=1`): pack patches host-side into
//! `[N, patch_vec]` → patch-embed matmul + bias → add the bilinearly-resampled
//! learned pos-embed → `depth`× [`vit_block_fwd`] (2-D vision RoPE, tanh-GELU, no
//! QK-norm, no LayerScale, full attention over the whole image) → final LayerNorm.
//! Output is `[N, hidden]` patch features for the PatchMerger.
//!
//! The learned pos-embed is resampled on the host (bilinear over the frozen table,
//! [`crate::vision::pos_embed_bilinear`]) and added as a positional input; the
//! patch-embed and the transformer blocks run on device and are gradient-capable.
//! (A device-side pos-embed gather for full pos-embed finetuning is a later step.)

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::vit::{vit_block_fwd, RopeTables, VitBlockWeights, VitKernelIds, VitScratch, VitShape};

use crate::config::VisionConfig;
use crate::vision::{pos_embed_bilinear, vision_position_ids, vision_rope_tables};

/// θ for the ViT's 2-D RoPE.
const VISION_ROPE_THETA: f32 = 10000.0;
/// LayerNorm eps in the ViT (matches the text side; HF vision uses 1e-6).
const VISION_EPS: f32 = 1e-6;

/// The kernels the ViT dispatches, in the order [`vision_pipelines`] lists them.
/// gelu slot carries the **tanh** GELU (Qwen3-VL vision act = gelu_pytorch_tanh).
pub fn vision_pipelines() -> &'static [(&'static str, &'static str)] {
    &[
        ("layernorm", kernels::LAYERNORM),
        ("matmul", kernels::MATMUL),
        ("matmul_rows", kernels::MATMUL_ROWS),
        ("bias_add", kernels::BIAS_ADD),
        ("gelu", kernels::GELU), // tanh
        ("scale_chan", kernels::SCALE_CHAN),
        ("add2", kernels::ADD2),
        ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
        ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
        ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
        ("ln_head", kernels::LN_HEAD),
        ("rope2d", kernels::ROPE2D),
        ("gelu_erf", kernels::GELU_ERF), // index 12: erf GELU for the PatchMerger
    ]
}

// PatchMerger kernel indices into `vision_pipelines`.
const M_LAYERNORM: usize = 0;
const M_MATMUL: usize = 1;
const M_BIAS_ADD: usize = 3;
const M_GELU_ERF: usize = 12;

fn vit_ids() -> VitKernelIds {
    VitKernelIds {
        layernorm: 0,
        matmul: 1,
        matmul_rows: 2,
        bias_add: 3,
        gelu_erf: 4, // tanh GELU wired here for Qwen3-VL
        scale_chan: 5,
        add2: 6,
        attn_scores_cross: 7,
        attn_softmax_cross: 8,
        attn_apply_cross: 9,
        ln_head: 10,
        rope2d: 11,
    }
}

/// Per-block weight leaf names (relative to `blocks.{b}.`). PyTorch layouts:
/// Linear `[out, in]`, bias `[out]`. Fused qkv `[3·hidden, hidden]`.
const BLOCK_LEAVES: &[&str] = &[
    "norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias", "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
    "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
];

/// Qwen3-VL ViT encoder over a `Gpu` preloaded with [`vision_pipelines`].
pub struct VisionEncoder<'g> {
    gpu: &'g Gpu,
    cfg: VisionConfig,
    w: HashMap<String, DeviceBuffer>,
    /// Host copy of the learned pos-embed table `[num_position_embeddings, hidden]`.
    pos_table: Vec<f32>,
}

impl<'g> VisionEncoder<'g> {
    /// Build from host weights. Required keys: `patch_embed.weight` `[hidden,
    /// patch_vec]`, `patch_embed.bias` `[hidden]`, `pos_embed` `[num_pos, hidden]`,
    /// `norm.weight`/`norm.bias` `[hidden]`, and per block `blocks.{b}.<leaf>` for
    /// every `BLOCK_LEAVES`.
    pub fn new(gpu: &'g Gpu, cfg: VisionConfig, weights: &HashMap<String, Vec<f32>>) -> VisionEncoder<'g> {
        let mut w = HashMap::new();
        for (name, data) in weights {
            if name == "pos_embed" {
                continue; // kept host-side for resampling
            }
            w.insert(name.clone(), gpu.storage_init(name, data));
        }
        let pos_table = weights["pos_embed"].clone();
        VisionEncoder { gpu, cfg, w, pos_table }
    }

    fn wb(&self, name: &str) -> &DeviceBuffer {
        self.w.get(name).unwrap_or_else(|| panic!("vision weight missing: {name}"))
    }

    /// Host-resample the learned pos-embed onto the `grid_h × grid_w` patch grid
    /// (merge-block order), returning `[N, hidden]`.
    fn pos_embeds(&self, grid_h: u32, grid_w: u32) -> Vec<f32> {
        let hidden = self.cfg.hidden as usize;
        let side = self.cfg.pos_grid();
        let (idx, wts) = pos_embed_bilinear(grid_h, grid_w, self.cfg.spatial_merge_size, side);
        let mut pos = vec![0f32; idx.len() * hidden];
        for (p, (id, wt)) in idx.iter().zip(&wts).enumerate() {
            for c in 0..hidden {
                let mut acc = 0f32;
                for k in 0..4 {
                    acc += self.pos_table[id[k] as usize * hidden + c] * wt[k];
                }
                pos[p * hidden + c] = acc;
            }
        }
        pos
    }

    /// Encode one image of `grid_h × grid_w` patches (`t = 1`). `pixels` is the
    /// host-packed `[N, patch_vec]` patch tensor in merge-block order. Returns the
    /// `[N, hidden]` patch features (post final LayerNorm).
    pub fn encode(&self, grid_h: u32, grid_w: u32, pixels: &[f32]) -> Vec<f32> {
        let g = self.gpu;
        let ids = vit_ids();
        let c = self.cfg.hidden;
        let n = grid_h * grid_w;
        let pv = self.cfg.patch_vec_dim();
        let sh = VitShape { dim: c, heads: self.cfg.num_heads, mlp: self.cfg.intermediate, eps: VISION_EPS };
        assert_eq!(pixels.len(), (n * pv) as usize, "pixels must be [N, patch_vec]");

        // Positions → 2-D vision RoPE tables (uploaded as rope2d cos/sin).
        let positions = vision_position_ids(grid_h, grid_w, self.cfg.spatial_merge_size);
        let (cos, sin) = vision_rope_tables(&positions, sh.head_dim(), VISION_ROPE_THETA);
        let cos_b = g.storage_init("vit.rope.cos", &cos);
        let sin_b = g.storage_init("vit.rope.sin", &sin);

        // Inputs: packed patches + host-resampled pos-embed.
        let pix = g.storage_init("vit.pixels", pixels);
        let pos = g.storage_init("vit.pos", &self.pos_embeds(grid_h, grid_w));
        let pe = g.storage((n * c) as u64); // patch-embed output (kept distinct from x)
        let x = g.storage((n * c) as u64);

        let scr = VitScratch::new(g, &sh, n, n, n); // one image → whole-image spans

        let mut steps: Vec<Step> = Vec::new();
        // patch-embed matmul [N,pv]·[hidden,pv]^T + bias, then x = patch_embed + pos.
        steps.push(g.step(ids.matmul, &[&pix, self.wb("patch_embed.weight"), &pe], &[n, pv, c], n * c));
        steps.push(g.step(ids.bias_add, &[&pe, self.wb("patch_embed.bias")], &[n, c], n * c));
        steps.push(g.step(ids.add2, &[&pe, &pos, &x], &[n * c], n * c));

        // Transformer blocks (in place on x); full attention over the image.
        let rope = RopeTables { cos: &cos_b, sin: &sin_b, tmod: n };
        for b in 0..self.cfg.depth {
            let p = |leaf: &str| self.wb(&format!("blocks.{b}.{leaf}"));
            let bw = VitBlockWeights {
                norm1_w: p("norm1.weight"),
                norm1_b: p("norm1.bias"),
                qkv_w: p("qkv.weight"),
                qkv_b: p("qkv.bias"),
                qk_norm: None,
                rope: Some(RopeTables { cos: rope.cos, sin: rope.sin, tmod: rope.tmod }),
                proj_w: p("proj.weight"),
                proj_b: p("proj.bias"),
                ls1: None,
                norm2_w: p("norm2.weight"),
                norm2_b: p("norm2.bias"),
                fc1_w: p("fc1.weight"),
                fc1_b: p("fc1.bias"),
                fc2_w: p("fc2.weight"),
                fc2_b: p("fc2.bias"),
                ls2: None,
            };
            vit_block_fwd(g, &ids, &sh, &bw, &x, n, &[(0, n)], n, &scr, &mut steps);
        }
        // Final LayerNorm into a fresh output buffer.
        let out = g.storage((n * c) as u64);
        steps.push(g.step(ids.layernorm, &[&x, self.wb("norm.weight"), self.wb("norm.bias"), &out], &[c, n, gpu_core::f(sh.eps)], n));

        g.submit(&[], &steps);
        g.read(&out, (n * c) as usize)
    }
}

/// Qwen3-VL PatchMerger: fold each `merge×merge` block of patch features into one
/// visual token. LayerNorm (over `in_dim` pre-shuffle for the main merger, or over
/// `in_dim·merge²` post-shuffle for the DeepStack mergers) → Linear(→ merged) →
/// GELU(erf) → Linear(→ out_dim). The 2×2 gather is a free contiguous reshape
/// because patches arrive in spatial-merge-block order. Required weight keys:
/// `ln.weight`/`ln.bias`, `fc1.weight` `[merged, merged]`/`fc1.bias`,
/// `fc2.weight` `[out_dim, merged]`/`fc2.bias`, where `merged = in_dim·merge²`.
pub struct PatchMerger<'g> {
    gpu: &'g Gpu,
    w: HashMap<String, DeviceBuffer>,
    in_dim: u32,
    merge: u32,
    out_dim: u32,
    /// `true` for DeepStack mergers (LayerNorm over the shuffled `in_dim·merge²`),
    /// `false` for the main merger (LayerNorm over `in_dim` per patch).
    postshuffle_norm: bool,
}

impl<'g> PatchMerger<'g> {
    pub fn new(
        gpu: &'g Gpu,
        weights: &HashMap<String, Vec<f32>>,
        in_dim: u32,
        merge: u32,
        out_dim: u32,
        postshuffle_norm: bool,
    ) -> PatchMerger<'g> {
        let w = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        PatchMerger { gpu, w, in_dim, merge, out_dim, postshuffle_norm }
    }

    fn wb(&self, name: &str) -> &DeviceBuffer {
        self.w.get(name).unwrap_or_else(|| panic!("merger weight missing: {name}"))
    }

    /// Merge `x` `[n, in_dim]` (patch features, merge-block order) into
    /// `[n/merge², out_dim]` visual tokens.
    pub fn merge(&self, x: &[f32], n: u32) -> Vec<f32> {
        let g = self.gpu;
        let m2 = self.merge * self.merge;
        assert!(n % m2 == 0, "n must be a multiple of merge²");
        let mrows = n / m2;
        let merged = self.in_dim * m2; // e.g. 1024·4 = 4096
        let eps = gpu_core::f(VISION_EPS);

        let inp = g.storage_init("mrg.in", x);
        let xn = g.storage((n * self.in_dim) as u64); // == mrows·merged elements
        let mut steps: Vec<Step> = Vec::new();
        if self.postshuffle_norm {
            // Reshape [n, in_dim] -> [mrows, merged] first, then LayerNorm over merged.
            steps.push(g.step(M_LAYERNORM, &[&inp, self.wb("ln.weight"), self.wb("ln.bias"), &xn], &[merged, mrows, eps], mrows));
        } else {
            // LayerNorm per patch over in_dim; the reshape to [mrows, merged] is free.
            steps.push(g.step(M_LAYERNORM, &[&inp, self.wb("ln.weight"), self.wb("ln.bias"), &xn], &[self.in_dim, n, eps], n));
        }
        let h = g.storage((mrows * merged) as u64);
        let h2 = g.storage((mrows * merged) as u64);
        steps.push(g.step(M_MATMUL, &[&xn, self.wb("fc1.weight"), &h], &[mrows, merged, merged], mrows * merged));
        steps.push(g.step(M_BIAS_ADD, &[&h, self.wb("fc1.bias")], &[mrows, merged], mrows * merged));
        steps.push(g.step(M_GELU_ERF, &[&h, &h2], &[mrows * merged], mrows * merged));
        let out = g.storage((mrows * self.out_dim) as u64);
        steps.push(g.step(M_MATMUL, &[&h2, self.wb("fc2.weight"), &out], &[mrows, merged, self.out_dim], mrows * self.out_dim));
        steps.push(g.step(M_BIAS_ADD, &[&out, self.wb("fc2.bias")], &[mrows, self.out_dim], mrows * self.out_dim));
        g.submit(&[], &steps);
        g.read(&out, (mrows * self.out_dim) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;

    fn tiny_cfg() -> VisionConfig {
        VisionConfig {
            depth: 2,
            hidden: 32,
            num_heads: 2,
            intermediate: 64,
            patch_size: 2,
            temporal_patch_size: 1,
            spatial_merge_size: 2,
            num_position_embeddings: 16, // 4×4 table
            out_hidden_size: 40,
            in_channels: 2,
            deepstack_indexes: vec![],
        }
    }

    fn rand_weights(cfg: &VisionConfig, seed: u64) -> HashMap<String, Vec<f32>> {
        let mut rng = Rng::new(seed);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let (c, pv, mlp) = (cfg.hidden as usize, cfg.patch_vec_dim() as usize, cfg.intermediate as usize);
        let mut w = HashMap::new();
        w.insert("patch_embed.weight".into(), r(c * pv));
        w.insert("patch_embed.bias".into(), r(c));
        w.insert("pos_embed".into(), r(cfg.num_position_embeddings as usize * c));
        w.insert("norm.weight".into(), vec![1.0; c]);
        w.insert("norm.bias".into(), r(c));
        for b in 0..cfg.depth {
            let dims = [c, c, 3 * c * c, 3 * c, c * c, c, c, c, mlp * c, mlp, c * mlp, c];
            for (leaf, &sz) in BLOCK_LEAVES.iter().zip(&dims) {
                let v = if leaf.ends_with("norm1.weight") || leaf.ends_with("norm2.weight") { vec![1.0; sz] } else { r(sz) };
                w.insert(format!("blocks.{b}.{leaf}"), v);
            }
        }
        w
    }

    fn merger_weights(in_dim: u32, merge: u32, out_dim: u32, postshuffle: bool, seed: u64) -> HashMap<String, Vec<f32>> {
        let mut rng = Rng::new(seed);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let merged = (in_dim * merge * merge) as usize;
        let ln_dim = if postshuffle { merged } else { in_dim as usize };
        let mut w = HashMap::new();
        w.insert("ln.weight".into(), vec![1.0; ln_dim]);
        w.insert("ln.bias".into(), r(ln_dim));
        w.insert("fc1.weight".into(), r(merged * merged));
        w.insert("fc1.bias".into(), r(merged));
        w.insert("fc2.weight".into(), r(out_dim as usize * merged));
        w.insert("fc2.bias".into(), r(out_dim as usize));
        w
    }

    #[test]
    fn patch_merger_shapes_and_norm_variants() {
        let gpu = Gpu::new_cpu(vision_pipelines());
        let (in_dim, merge, out_dim, n) = (8u32, 2u32, 10u32, 8u32);
        let mut rng = Rng::new(1);
        let x: Vec<f32> = (0..(n * in_dim) as usize).map(|_| rng.next_f32() - 0.5).collect();
        for postshuffle in [false, true] {
            let w = merger_weights(in_dim, merge, out_dim, postshuffle, 5);
            let m = PatchMerger::new(&gpu, &w, in_dim, merge, out_dim, postshuffle);
            let out = m.merge(&x, n);
            assert_eq!(out.len(), (n / (merge * merge) * out_dim) as usize); // 2×10
            assert!(out.iter().all(|v| v.is_finite()));
            assert!(out.iter().any(|&v| v.abs() > 1e-6));
        }
    }

    #[test]
    fn encode_runs_and_shape_is_right() {
        let cfg = tiny_cfg();
        let gpu = Gpu::new_cpu(vision_pipelines());
        let weights = rand_weights(&cfg, 7);
        let enc = VisionEncoder::new(&gpu, cfg.clone(), &weights);
        let (gh, gw) = (4u32, 4u32); // 16 patches, one 2×2-merge grid
        let n = (gh * gw) as usize;
        let pv = cfg.patch_vec_dim() as usize;
        let mut rng = Rng::new(99);
        let pixels: Vec<f32> = (0..n * pv).map(|_| rng.next_f32() - 0.5).collect();
        let out = enc.encode(gh, gw, &pixels);
        assert_eq!(out.len(), n * cfg.hidden as usize);
        assert!(out.iter().all(|v| v.is_finite()), "output must be finite");
        assert!(out.iter().any(|&v| v.abs() > 1e-6), "output must not be all zero");
    }
}
