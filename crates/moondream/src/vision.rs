// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream SigLIP-style ViT vision encoder, built on the shared `model::vit`
//! block builder. Pre-LN bidirectional transformer, no CLS token, no QK-norm, no
//! RoPE, no LayerScale, tanh-GELU MLP, learned absolute pos-embed, and a final
//! post-LN. Patches are host-packed to `[729, 588]` (14×14×3) and linearly
//! embedded. One crop per span; the model runs all crops as independent spans.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::vit::{vit_block_fwd, VitBlockWeights, VitKernelIds, VitScratch, VitShape};

use crate::config::VisionConfig;

const DINO_EPS: f32 = 1e-6;

/// The ViT kernels (tanh-GELU at the `gelu_erf` slot for Moondream's gelu_approx).
pub fn vision_pipelines() -> &'static [(&'static str, &'static str)] {
    &[
        ("layernorm", kernels::LAYERNORM),
        ("matmul", kernels::MATMUL),
        ("matmul_rows", kernels::MATMUL_ROWS),
        ("bias_add", kernels::BIAS_ADD),
        ("gelu", kernels::GELU), // tanh (gelu_approx)
        ("scale_chan", kernels::SCALE_CHAN),
        ("add2", kernels::ADD2),
        ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
        ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
        ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
        ("ln_head", kernels::LN_HEAD),
        ("rope2d", kernels::ROPE2D),
        ("adaptive_avgpool2d", kernels::ADAPTIVE_AVGPOOL2D), // 12: reconstruct→27×27 pool
    ]
}

/// Pipeline slot of `adaptive_avgpool2d` within [`vision_pipelines`].
pub const ADAPTIVE_AVGPOOL2D_ID: usize = 12;

fn vit_ids() -> VitKernelIds {
    VitKernelIds {
        layernorm: 0,
        matmul: 1,
        matmul_rows: 2,
        bias_add: 3,
        gelu_erf: 4, // tanh GELU wired here
        scale_chan: 5,
        add2: 6,
        attn_scores_cross: 7,
        attn_softmax_cross: 8,
        attn_apply_cross: 9,
        ln_head: 10,
        rope2d: 11,
    }
}

/// Moondream SigLIP ViT encoder over a `Gpu` preloaded with [`vision_pipelines`].
pub struct SiglipEncoder<'g> {
    gpu: &'g Gpu,
    cfg: VisionConfig,
    w: HashMap<String, DeviceBuffer>,
}

impl<'g> SiglipEncoder<'g> {
    /// Required keys: `patch_emb.weight` `[dim, patch_vec]`, `patch_emb.bias`
    /// `[dim]`, `pos_emb` `[patches, dim]`, `post_ln.weight`/`post_ln.bias`
    /// `[dim]`, and per block `blocks.{b}.<leaf>`.
    pub fn new(gpu: &'g Gpu, cfg: VisionConfig, weights: &HashMap<String, Vec<f32>>) -> SiglipEncoder<'g> {
        let w = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        SiglipEncoder { gpu, cfg, w }
    }

    fn wb(&self, name: &str) -> &DeviceBuffer {
        self.w.get(name).unwrap_or_else(|| panic!("vision weight missing: {name}"))
    }

    /// Encode `n_crops` crops of host-packed patches `[n_crops·patches, patch_vec]`
    /// (patch-major), returning `[n_crops·patches, dim]` post-LN features. Each
    /// crop attends within itself (a span).
    pub fn encode(&self, n_crops: u32, packed: &[f32]) -> Vec<f32> {
        let g = self.gpu;
        let ids = vit_ids();
        let c = self.cfg.dim;
        let ppc = self.cfg.patches_per_crop();
        let rows = n_crops * ppc;
        let pv = self.cfg.patch_vec();
        let sh = VitShape { dim: c, heads: self.cfg.n_heads, mlp: self.cfg.ff_dim, eps: DINO_EPS };
        assert_eq!(packed.len(), (rows * pv) as usize, "packed must be [rows, patch_vec]");

        let pix = g.storage_init("md.pix", packed);
        let pe = g.storage((rows * c) as u64);
        let x = g.storage((rows * c) as u64);
        // Per-crop learned pos-embed, tiled over crops.
        let pos_tiled: Vec<f32> = {
            let base = &self.wb_host();
            (0..n_crops).flat_map(|_| base.iter().copied()).collect()
        };
        let pos = g.storage_init("md.pos", &pos_tiled);
        let scr = VitScratch::new(g, &sh, rows, ppc, ppc); // spans = per-crop

        let mut steps: Vec<Step> = Vec::new();
        // patch-embed: [rows, pv] · [dim, pv]^T + bias, then + pos.
        steps.push(g.step(ids.matmul, &[&pix, self.wb("patch_emb.weight"), &pe], &[rows, pv, c], rows * c));
        steps.push(g.step(ids.bias_add, &[&pe, self.wb("patch_emb.bias")], &[rows, c], rows * c));
        steps.push(g.step(ids.add2, &[&pe, &pos, &x], &[rows * c], rows * c));

        let spans: Vec<(u32, u32)> = (0..n_crops).map(|i| (i * ppc, ppc)).collect();
        for b in 0..self.cfg.n_layers {
            let p = |leaf: &str| self.wb(&format!("blocks.{b}.{leaf}"));
            let bw = VitBlockWeights {
                norm1_w: p("ln1.weight"),
                norm1_b: p("ln1.bias"),
                qkv_w: p("attn.qkv.weight"),
                qkv_b: p("attn.qkv.bias"),
                qk_norm: None,
                rope: None,
                proj_w: p("attn.proj.weight"),
                proj_b: p("attn.proj.bias"),
                ls1: None,
                norm2_w: p("ln2.weight"),
                norm2_b: p("ln2.bias"),
                fc1_w: p("mlp.fc1.weight"),
                fc1_b: p("mlp.fc1.bias"),
                fc2_w: p("mlp.fc2.weight"),
                fc2_b: p("mlp.fc2.bias"),
                ls2: None,
            };
            vit_block_fwd(g, &ids, &sh, &bw, &x, rows, &spans, ppc, &scr, &mut steps);
        }
        // Final post-LN into a fresh buffer.
        let out = g.storage((rows * c) as u64);
        steps.push(g.step(ids.layernorm, &[&x, self.wb("post_ln.weight"), self.wb("post_ln.bias"), &out], &[c, rows, f(sh.eps)], rows));
        g.submit(&[], &steps);
        g.read(&out, (rows * c) as usize)
    }

    // The learned pos-embed host copy (one crop's worth) for tiling.
    fn wb_host(&self) -> Vec<f32> {
        self.gpu.read(self.wb("pos_emb"), (self.cfg.patches_per_crop() * self.cfg.dim) as usize)
    }
}

/// Moondream connector: a 2-layer MLP `Linear(in→inner)` → tanh-GELU →
/// `Linear(inner→out)` mapping the `[729, 2·dim]` global‖local concat to `[729,
/// dim_text]` image tokens. Reuses matmul/bias/gelu (no new kernels). Weight keys:
/// `fc1.weight` `[inner,in]`/`fc1.bias`, `fc2.weight` `[out,inner]`/`fc2.bias`.
pub struct Connector<'g> {
    gpu: &'g Gpu,
    w: HashMap<String, DeviceBuffer>,
    in_dim: u32,
    inner: u32,
    out_dim: u32,
}

impl<'g> Connector<'g> {
    pub fn new(gpu: &'g Gpu, weights: &HashMap<String, Vec<f32>>, in_dim: u32, inner: u32, out_dim: u32) -> Connector<'g> {
        let w = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        Connector { gpu, w, in_dim, inner, out_dim }
    }
    fn wb(&self, n: &str) -> &DeviceBuffer {
        self.w.get(n).unwrap_or_else(|| panic!("connector weight missing: {n}"))
    }
    /// Project `rows × in_dim` → `rows × out_dim`.
    pub fn forward(&self, rows: u32, x: &[f32]) -> Vec<f32> {
        let g = self.gpu;
        assert_eq!(x.len(), (rows * self.in_dim) as usize);
        let xb = g.storage_init("cin", x);
        let h = g.storage((rows * self.inner) as u64);
        let h2 = g.storage((rows * self.inner) as u64);
        let out = g.storage((rows * self.out_dim) as u64);
        // matmul(1) + bias(3) + gelu(4) + matmul(1) + bias(3), per vision_pipelines.
        g.submit(
            &[],
            &[
                g.step(1, &[&xb, self.wb("fc1.weight"), &h], &[rows, self.in_dim, self.inner], rows * self.inner),
                g.step(3, &[&h, self.wb("fc1.bias")], &[rows, self.inner], rows * self.inner),
                g.step(4, &[&h, &h2], &[rows * self.inner], rows * self.inner),
                g.step(1, &[&h2, self.wb("fc2.weight"), &out], &[rows, self.inner, self.out_dim], rows * self.out_dim),
                g.step(3, &[&out, self.wb("fc2.bias")], &[rows, self.out_dim], rows * self.out_dim),
            ],
        );
        g.read(&out, (rows * self.out_dim) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MoondreamConfig;
    use data::rng::Rng;

    const BLOCK_LEAVES: &[&str] = &[
        "ln1.weight", "ln1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias", "ln2.weight",
        "ln2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias",
    ];

    // A tiny SigLIP config: 4×4 grid (16 patches), dim 32, 2 heads, 2 layers.
    fn tiny() -> VisionConfig {
        VisionConfig { dim: 32, patch: 2, n_layers: 2, ff_dim: 64, n_heads: 2, crop_size: 8, max_crops: 4, overlap_margin: 1 }
    }

    #[test]
    fn siglip_encodes_crops() {
        let _ = MoondreamConfig::preview; // config linkage
        let cfg = tiny();
        let gpu = Gpu::new_cpu(vision_pipelines());
        let (c, pv, ppc) = (cfg.dim as usize, cfg.patch_vec() as usize, cfg.patches_per_crop() as usize);
        let mut rng = Rng::new(3);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("patch_emb.weight".into(), r(c * pv));
        w.insert("patch_emb.bias".into(), r(c));
        w.insert("pos_emb".into(), r(ppc * c));
        w.insert("post_ln.weight".into(), vec![1.0; c]);
        w.insert("post_ln.bias".into(), r(c));
        for b in 0..cfg.n_layers {
            let dims = [c, c, 3 * c * c, 3 * c, c * c, c, c, c, cfg.ff_dim as usize * c, cfg.ff_dim as usize, c * cfg.ff_dim as usize, c];
            for (leaf, &sz) in BLOCK_LEAVES.iter().zip(&dims) {
                let v = if leaf.ends_with("ln1.weight") || leaf.ends_with("ln2.weight") { vec![1.0; sz] } else { r(sz) };
                w.insert(format!("blocks.{b}.{leaf}"), v);
            }
        }
        let enc = SiglipEncoder::new(&gpu, cfg.clone(), &w);
        let n_crops = 2u32;
        let packed: Vec<f32> = (0..(n_crops as usize * ppc * pv)).map(|_| rng.next_f32() - 0.5).collect();
        let out = enc.encode(n_crops, &packed);
        assert_eq!(out.len(), n_crops as usize * ppc * c);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn connector_projects() {
        let gpu = Gpu::new_cpu(vision_pipelines());
        let (in_dim, inner, out_dim, rows) = (48u32, 96u32, 32u32, 9u32);
        let mut rng = Rng::new(4);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("fc1.weight".into(), r((inner * in_dim) as usize));
        w.insert("fc1.bias".into(), r(inner as usize));
        w.insert("fc2.weight".into(), r((out_dim * inner) as usize));
        w.insert("fc2.bias".into(), r(out_dim as usize));
        let conn = Connector::new(&gpu, &w, in_dim, inner, out_dim);
        let x: Vec<f32> = (0..(rows * in_dim) as usize).map(|_| rng.next_f32() - 0.5).collect();
        let out = conn.forward(rows, &x);
        assert_eq!(out.len(), (rows * out_dim) as usize);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
