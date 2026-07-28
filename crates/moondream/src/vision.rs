// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream SigLIP-style ViT vision encoder, built on the shared `model::vit`
//! block builder. Pre-LN bidirectional transformer, no CLS token, no QK-norm, no
//! RoPE, no LayerScale, tanh-GELU MLP, learned absolute pos-embed, and a final
//! post-LN. Patches are host-packed to `[729, 588]` (14×14×3) and linearly
//! embedded. One crop per span; the model runs all crops as independent spans.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::vit::{vit_block_bwd, vit_block_fwd, vit_block_fwd_cached, VitBlockCache, VitBlockGrads, VitBlockWeights, VitBwdIds, VitBwdScratch, VitKernelIds, VitScratch, VitShape};

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
        // --- backward (ViT training) ---
        ("matmul_dx", kernels::MATMUL_DX),               // 13
        ("matmul_dw", kernels::MATMUL_DW),               // 14
        ("bias_grad", kernels::BIAS_GRAD),               // 15
        ("gelu_bwd", kernels::GELU_BWD),                 // 16 (tanh gelu bwd)
        ("layernorm_dx", kernels::LAYERNORM_DX),         // 17
        ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA), // 18
        ("layernorm_dbeta", kernels::LAYERNORM_DBETA),   // 19
        ("ln_stats", kernels::LN_STATS),                 // 20
        ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS), // 21
        ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS), // 22
        ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS), // 23
        ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS), // 24
        ("axpy", kernels::AXPY),                         // 25
    ]
}

/// Pipeline slot of `adaptive_avgpool2d` within [`vision_pipelines`].
pub const ADAPTIVE_AVGPOOL2D_ID: usize = 12;

/// Backward kernel ids for [`vit_block_bwd`]. Moondream has no QK-norm/LayerScale,
/// so `scale_chan_dg`/`ln_head_*`/`region_copy` are never dispatched (point at 0);
/// the gelu backward is the TANH variant to match the forward's tanh-GELU slot.
fn vit_bwd_ids() -> VitBwdIds {
    VitBwdIds {
        layernorm_dx: 17,
        ln_dgamma: 18,
        ln_dbeta: 19,
        matmul_dx: 13,
        matmul_dw: 14,
        bias_grad: 15,
        gelu_erf_bwd: 16, // tanh gelu bwd (matches the tanh-GELU forward slot)
        scale_chan_dg: 0, // unused (no LayerScale)
        ln_head_dx: 0,    // unused (no QK-norm)
        ln_head_dgb: 0,   // unused
        attn_bwd_dscores_cross: 21,
        attn_bwd_dv_cross: 22,
        attn_bwd_dq_cross: 23,
        attn_bwd_dk_cross: 24,
        ln_stats: 20,
        region_copy: 0, // unused (no QK-norm)
        axpy: 25,
    }
}

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

    fn shape(&self) -> VitShape {
        VitShape { dim: self.cfg.dim, heads: self.cfg.n_heads, mlp: self.cfg.ff_dim, eps: DINO_EPS }
    }

    /// The block's weights (no QK-norm/RoPE/LayerScale for SigLIP).
    fn vit_bw(&self, b: u32) -> VitBlockWeights<'_> {
        VitBlockWeights {
            norm1_w: self.wb(&format!("blocks.{b}.ln1.weight")),
            norm1_b: self.wb(&format!("blocks.{b}.ln1.bias")),
            qkv_w: self.wb(&format!("blocks.{b}.attn.qkv.weight")),
            qkv_b: self.wb(&format!("blocks.{b}.attn.qkv.bias")),
            qk_norm: None,
            rope: None,
            proj_w: self.wb(&format!("blocks.{b}.attn.proj.weight")),
            proj_b: self.wb(&format!("blocks.{b}.attn.proj.bias")),
            ls1: None,
            norm2_w: self.wb(&format!("blocks.{b}.ln2.weight")),
            norm2_b: self.wb(&format!("blocks.{b}.ln2.bias")),
            fc1_w: self.wb(&format!("blocks.{b}.mlp.fc1.weight")),
            fc1_b: self.wb(&format!("blocks.{b}.mlp.fc1.bias")),
            fc2_w: self.wb(&format!("blocks.{b}.mlp.fc2.weight")),
            fc2_b: self.wb(&format!("blocks.{b}.mlp.fc2.bias")),
            ls2: None,
        }
    }

    /// Training-mode forward: like [`encode`](Self::encode) but every block runs
    /// `vit_block_fwd_cached` into its own [`VitBlockCache`] so the backward has the
    /// SSA activations. Returns the caches + the pre-post-LN output + the post-LN
    /// output. `n_crops` spans (each attends within itself).
    pub fn forward_train(&self, n_crops: u32, packed: &[f32]) -> SiglipTrain {
        let g = self.gpu;
        let ids = vit_ids();
        let kb = vit_bwd_ids();
        let (c, ppc, pv) = (self.cfg.dim, self.cfg.patches_per_crop(), self.cfg.patch_vec());
        let rows = n_crops * ppc;
        let sh = self.shape();
        assert_eq!(packed.len(), (rows * pv) as usize, "packed must be [rows, patch_vec]");

        let pix = g.storage_init("md.pix", packed);
        let pe = g.storage((rows * c) as u64);
        let pos_tiled: Vec<f32> = { let base = self.wb_host(); (0..n_crops).flat_map(|_| base.iter().copied()).collect() };
        let pos = g.storage_init("md.pos", &pos_tiled);
        let caches: Vec<VitBlockCache> = (0..self.cfg.n_layers).map(|_| VitBlockCache::new(g, &sh, rows, ppc)).collect();
        let x_last = g.storage((rows * c) as u64);
        let out = g.storage((rows * c) as u64);
        let scr_tmp = g.storage((rows * c) as u64);
        let scores = g.storage((sh.heads * ppc * ppc) as u64);
        let spans: Vec<(u32, u32)> = (0..n_crops).map(|i| (i * ppc, ppc)).collect();

        let mut steps: Vec<Step> = Vec::new();
        steps.push(g.step(ids.matmul, &[&pix, self.wb("patch_emb.weight"), &pe], &[rows, pv, c], rows * c));
        steps.push(g.step(ids.bias_add, &[&pe, self.wb("patch_emb.bias")], &[rows, c], rows * c));
        steps.push(g.step(ids.add2, &[&pe, &pos, &caches[0].x_in], &[rows * c], rows * c));
        for b in 0..self.cfg.n_layers as usize {
            let bw = self.vit_bw(b as u32);
            let x_out = if b + 1 < caches.len() { &caches[b + 1].x_in } else { &x_last };
            vit_block_fwd_cached(g, &ids, &kb, &sh, &bw, &caches[b], x_out, rows, &spans, &scr_tmp, &scores, &mut steps);
        }
        steps.push(g.step(ids.layernorm, &[&x_last, self.wb("post_ln.weight"), self.wb("post_ln.bias"), &out], &[c, rows, f(sh.eps)], rows));
        g.submit(&[], &steps);
        SiglipTrain { pix, caches, x_last, out, rows, n_crops, spans }
    }

    /// ViT backward from the post-LN output grad `d_out` (`[rows, dim]`): fill `gr`
    /// and return the input-patch grad `[rows, patch_vec]`. Chain: post-LN → blocks
    /// in reverse (`vit_block_bwd`) → patch-embed (matmul/bias) + pos-embed scatter.
    pub fn backward(&self, tr: &SiglipTrain, d_out: &DeviceBuffer, gr: &SiglipGrads) -> Vec<f32> {
        let g = self.gpu;
        let ids = vit_ids();
        let kb = vit_bwd_ids();
        let (c, ppc, pv) = (self.cfg.dim, self.cfg.patches_per_crop(), self.cfg.patch_vec());
        let rows = tr.rows;
        let sh = self.shape();
        let sb = VitBwdScratch::new(g, &sh, rows, ppc);
        let mean = g.storage(rows as u64);
        let inv = g.storage(rows as u64);
        let d_xlast = g.storage((rows * c) as u64);
        let d_x: Vec<DeviceBuffer> = (0..self.cfg.n_layers).map(|_| g.storage((rows * c) as u64)).collect();

        // post-LN backward.
        g.submit(
            &[],
            &[
                g.step(20, &[&tr.x_last, &mean, &inv], &[c, rows, f(sh.eps)], rows),
                g.step(18, &[d_out, &tr.x_last, &mean, &inv, &gr.post_ln_w], &[c, rows], c),
                g.step(19, &[d_out, &gr.post_ln_b], &[c, rows], c),
                g.step(17, &[&tr.x_last, self.wb("post_ln.weight"), d_out, &d_xlast], &[c, rows, f(sh.eps)], rows),
            ],
        );
        // Blocks in reverse.
        for b in (0..self.cfg.n_layers as usize).rev() {
            let bw = self.vit_bw(b as u32);
            let bg = &gr.blocks[b];
            let vg = VitBlockGrads {
                norm1_w: &bg.ln1_w,
                norm1_b: &bg.ln1_b,
                qkv_w: &bg.qkv_w,
                qkv_b: &bg.qkv_b,
                q_norm_w: None,
                q_norm_b: None,
                k_norm_w: None,
                k_norm_b: None,
                proj_w: &bg.proj_w,
                proj_b: &bg.proj_b,
                ls1: None,
                norm2_w: &bg.ln2_w,
                norm2_b: &bg.ln2_b,
                fc1_w: &bg.fc1_w,
                fc1_b: &bg.fc1_b,
                fc2_w: &bg.fc2_w,
                fc2_b: &bg.fc2_b,
                ls2: None,
            };
            let d_out_b = if b + 1 < d_x.len() { &d_x[b + 1] } else { &d_xlast };
            let mut s: Vec<Step> = Vec::new();
            vit_block_bwd(g, &ids, &kb, &sh, &bw, &vg, &tr.caches[b], d_out_b, &d_x[b], rows, &tr.spans, &sb, &mut s);
            g.submit(&[], &s);
        }
        // Patch-embed backward: d_x[0] is the grad of x0 = patch_emb(pix) + pos.
        let d_pix = g.storage((rows * pv) as u64);
        g.submit(
            &[],
            &[
                g.step(14, &[&d_x[0], &tr.pix, &gr.patch_emb_w], &[rows, pv, c], c * pv),
                g.step(15, &[&d_x[0], &gr.patch_emb_b], &[rows, c], c),
                g.step(13, &[&d_x[0], self.wb("patch_emb.weight"), &d_pix], &[rows, pv, c, 0], rows * pv),
            ],
        );
        // Pos-embed grad: sum the per-crop slices of d_x0 (host).
        let dx0 = g.read(&d_x[0], (rows * c) as usize);
        let ppcc = (ppc * c) as usize;
        let mut dpos = vec![0.0f32; ppcc];
        for cr in 0..tr.n_crops as usize {
            for i in 0..ppcc {
                dpos[i] += dx0[cr * ppcc + i];
            }
        }
        g.write(&gr.pos_emb, &dpos.iter().map(|&v| f(v)).collect::<Vec<u32>>());
        g.read(&d_pix, (rows * pv) as usize)
    }
}

/// Cached training-forward state for the SigLIP ViT backward.
pub struct SiglipTrain {
    pix: DeviceBuffer,
    caches: Vec<VitBlockCache>,
    x_last: DeviceBuffer,
    pub out: DeviceBuffer,
    rows: u32,
    n_crops: u32,
    spans: Vec<(u32, u32)>,
}

/// Per-block SigLIP ViT parameter grads (zeroed on build; the bwd kernels `+=`).
pub struct SiglipBlockGrads {
    ln1_w: DeviceBuffer,
    ln1_b: DeviceBuffer,
    qkv_w: DeviceBuffer,
    qkv_b: DeviceBuffer,
    proj_w: DeviceBuffer,
    proj_b: DeviceBuffer,
    ln2_w: DeviceBuffer,
    ln2_b: DeviceBuffer,
    fc1_w: DeviceBuffer,
    fc1_b: DeviceBuffer,
    fc2_w: DeviceBuffer,
    fc2_b: DeviceBuffer,
}

/// All SigLIP ViT grads (patch-embed, pos-embed, post-LN, per block).
pub struct SiglipGrads {
    pub patch_emb_w: DeviceBuffer,
    pub patch_emb_b: DeviceBuffer,
    pub pos_emb: DeviceBuffer,
    pub post_ln_w: DeviceBuffer,
    pub post_ln_b: DeviceBuffer,
    pub blocks: Vec<SiglipBlockGrads>,
}

impl SiglipGrads {
    pub fn new(g: &Gpu, cfg: &VisionConfig) -> SiglipGrads {
        let z = |n: u32| g.storage_init("md.vg", &vec![0.0f32; n as usize]);
        let (c, m, pv, ppc) = (cfg.dim, cfg.ff_dim, cfg.patch_vec(), cfg.patches_per_crop());
        SiglipGrads {
            patch_emb_w: z(c * pv),
            patch_emb_b: z(c),
            pos_emb: z(ppc * c),
            post_ln_w: z(c),
            post_ln_b: z(c),
            blocks: (0..cfg.n_layers)
                .map(|_| SiglipBlockGrads {
                    ln1_w: z(c),
                    ln1_b: z(c),
                    qkv_w: z(3 * c * c),
                    qkv_b: z(3 * c),
                    proj_w: z(c * c),
                    proj_b: z(c),
                    ln2_w: z(c),
                    ln2_b: z(c),
                    fc1_w: z(m * c),
                    fc1_b: z(m),
                    fc2_w: z(c * m),
                    fc2_b: z(c),
                })
                .collect(),
        }
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

    fn vision_weights(cfg: &VisionConfig, rng: &mut Rng) -> HashMap<String, Vec<f32>> {
        let (c, pv, ppc, m) = (cfg.dim as usize, cfg.patch_vec() as usize, cfg.patches_per_crop() as usize, cfg.ff_dim as usize);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.3).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("patch_emb.weight".into(), r(c * pv));
        w.insert("patch_emb.bias".into(), r(c));
        w.insert("pos_emb".into(), r(ppc * c));
        w.insert("post_ln.weight".into(), vec![1.0; c]);
        w.insert("post_ln.bias".into(), r(c));
        for b in 0..cfg.n_layers {
            let dims = [c, c, 3 * c * c, 3 * c, c * c, c, c, c, m * c, m, c * m, c];
            for (leaf, &sz) in BLOCK_LEAVES.iter().zip(&dims) {
                let v = if leaf.ends_with("ln1.weight") || leaf.ends_with("ln2.weight") { vec![1.0; sz] } else { r(sz) };
                w.insert(format!("blocks.{b}.{leaf}"), v);
            }
        }
        w
    }

    #[test]
    fn siglip_vit_backward_matches_finite_diff() {
        // Gradcheck the SigLIP ViT training fwd/bwd (first in-tree consumer of the
        // shared vit_block_fwd_cached/vit_block_bwd): input-patch grad exercises the
        // whole tower; patch_emb/pos_emb/block-fc1 grads cover the rest.
        let gpu = Gpu::new_cpu(vision_pipelines());
        let cfg = VisionConfig { dim: 16, patch: 2, n_layers: 2, ff_dim: 32, n_heads: 2, crop_size: 6, max_crops: 4, overlap_margin: 1 };
        let (c, pv, ppc) = (cfg.dim as usize, cfg.patch_vec() as usize, cfg.patches_per_crop() as usize);
        let mut rng = Rng::new(5);
        let w = vision_weights(&cfg, &mut rng);
        let n_crops = 1u32;
        let rows = (n_crops * cfg.patches_per_crop()) as usize;
        let packed: Vec<f32> = (0..rows * pv).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        let n = rows * c;

        let enc = SiglipEncoder::new(&gpu, cfg.clone(), &w);
        let tr = enc.forward_train(n_crops, &packed);
        let d_out = gpu.storage_init("dout", &vec![1.0f32; n]);
        let gr = SiglipGrads::new(&gpu, &cfg);
        let d_pix = enc.backward(&tr, &d_out, &gr);
        let g_pe = gpu.read(&gr.patch_emb_w, c * pv);
        let g_fc1 = gpu.read(&gr.blocks[0].fc1_w, (cfg.ff_dim * cfg.dim) as usize);
        let g_pos = gpu.read(&gr.pos_emb, ppc * c);

        let loss = |wm: &HashMap<String, Vec<f32>>, pk: &[f32]| -> f32 {
            let e = SiglipEncoder::new(&gpu, cfg.clone(), wm);
            gpu.read(&e.forward_train(n_crops, pk).out, n).iter().sum::<f32>()
        };
        let eps = 1e-3f32;
        let ok = |a: f32, num: f32| (a - num).abs() <= 4e-3 + 8e-2 * num.abs();

        for &i in &[0usize, 5, 11, 20, 50, 90] {
            let (mut pp, mut pm) = (packed.clone(), packed.clone());
            pp[i] += eps;
            pm[i] -= eps;
            let num = (loss(&w, &pp) - loss(&w, &pm)) / (2.0 * eps);
            assert!(ok(d_pix[i], num), "d_pix[{i}]: analytic {} vs numeric {}", d_pix[i], num);
        }
        for &j in &[0usize, 7, 50] {
            let (mut wp, mut wm2) = (w.clone(), w.clone());
            wp.get_mut("patch_emb.weight").unwrap()[j] += eps;
            wm2.get_mut("patch_emb.weight").unwrap()[j] -= eps;
            let num = (loss(&wp, &packed) - loss(&wm2, &packed)) / (2.0 * eps);
            assert!(ok(g_pe[j], num), "d patch_emb.w[{j}]: analytic {} vs numeric {}", g_pe[j], num);
        }
        for &j in &[0usize, 17, 40] {
            let (mut wp, mut wm2) = (w.clone(), w.clone());
            wp.get_mut("blocks.0.mlp.fc1.weight").unwrap()[j] += eps;
            wm2.get_mut("blocks.0.mlp.fc1.weight").unwrap()[j] -= eps;
            let num = (loss(&wp, &packed) - loss(&wm2, &packed)) / (2.0 * eps);
            assert!(ok(g_fc1[j], num), "d blocks.0.fc1[{j}]: analytic {} vs numeric {}", g_fc1[j], num);
        }
        for &j in &[0usize, 5, 11] {
            let (mut wp, mut wm2) = (w.clone(), w.clone());
            wp.get_mut("pos_emb").unwrap()[j] += eps;
            wm2.get_mut("pos_emb").unwrap()[j] -= eps;
            let num = (loss(&wp, &packed) - loss(&wm2, &packed)) / (2.0 * eps);
            assert!(ok(g_pos[j], num), "d pos_emb[{j}]: analytic {} vs numeric {}", g_pos[j], num);
        }
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
