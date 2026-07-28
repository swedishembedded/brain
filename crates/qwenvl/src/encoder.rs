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

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::vit::{vit_block_bwd, vit_block_fwd, vit_block_fwd_cached, RopeTables, VitBlockCache, VitBlockGrads, VitBlockWeights, VitBwdIds, VitBwdScratch, VitKernelIds, VitScratch, VitShape};

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
        ("region_copy", kernels::REGION_COPY), // index 13: DeepStack tap snapshots
        // --- backward (ViT training) ---
        ("matmul_dx", kernels::MATMUL_DX),               // 14
        ("matmul_dw", kernels::MATMUL_DW),               // 15
        ("bias_grad", kernels::BIAS_GRAD),               // 16
        ("gelu_bwd", kernels::GELU_BWD),                 // 17 (tanh gelu bwd — block MLP)
        ("layernorm_dx", kernels::LAYERNORM_DX),         // 18
        ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA), // 19
        ("layernorm_dbeta", kernels::LAYERNORM_DBETA),   // 20
        ("ln_stats", kernels::LN_STATS),                 // 21
        ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS), // 22
        ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS), // 23
        ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS), // 24
        ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS), // 25
        ("axpy", kernels::AXPY),                         // 26
    ]
}

/// Backward kernel ids for [`vit_block_bwd`]. Qwen3-VL ViT has no QK-norm/LayerScale
/// (those ids are never dispatched → 0); the block MLP is tanh-GELU, so gelu_erf_bwd
/// points at the TANH gelu_bwd (matching the tanh-GELU forward slot).
fn vit_bwd_ids() -> VitBwdIds {
    VitBwdIds {
        layernorm_dx: 18,
        ln_dgamma: 19,
        ln_dbeta: 20,
        matmul_dx: 14,
        matmul_dw: 15,
        bias_grad: 16,
        gelu_erf_bwd: 17,
        scale_chan_dg: 0,
        ln_head_dx: 0,
        ln_head_dgb: 0,
        attn_bwd_dscores_cross: 22,
        attn_bwd_dv_cross: 23,
        attn_bwd_dq_cross: 24,
        attn_bwd_dk_cross: 25,
        ln_stats: 21,
        region_copy: 13,
        axpy: 26,
    }
}

/// Snapshot a `[rows, dim]` buffer via region_copy (whole-buffer contiguous copy).
const V_REGION_COPY: usize = 13;

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
    /// and per block `blocks.{b}.<leaf>` for every `BLOCK_LEAVES`. (No post-block
    /// norm — the PatchMerger's LayerNorm is the final norm, matching HF.)
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
        self.encode_with_taps(grid_h, grid_w, pixels, &[]).0
    }

    /// Like [`Self::encode`], but also snapshot the output of each block whose
    /// index is in `taps` (DeepStack tap points, e.g. `[5, 11, 17]`). Returns the
    /// final `[N, hidden]` features and one `[N, hidden]` snapshot per tap (in the
    /// order given), for the DeepStack mergers.
    pub fn encode_with_taps(&self, grid_h: u32, grid_w: u32, pixels: &[f32], taps: &[u32]) -> (Vec<f32>, Vec<Vec<f32>>) {
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
        let tap_bufs: Vec<DeviceBuffer> = taps.iter().map(|_| g.storage((n * c) as u64)).collect();

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
            // Snapshot this block's output for DeepStack before the next block
            // overwrites x (region_copy: whole-buffer contiguous copy).
            if let Some(i) = taps.iter().position(|&t| t == b) {
                steps.push(g.step(V_REGION_COPY, &[&x, &tap_bufs[i]], &[n, c, c, 0], n * c));
            }
        }
        // No post-block norm: HF applies the PatchMerger (which carries its own
        // LayerNorm) directly to the last block's output.
        g.submit(&[], &steps);
        let features = g.read(&x, (n * c) as usize);
        let tap_feats = tap_bufs.iter().map(|tb| g.read(tb, (n * c) as usize)).collect();
        (features, tap_feats)
    }

    fn shape(&self) -> VitShape {
        VitShape { dim: self.cfg.hidden, heads: self.cfg.num_heads, mlp: self.cfg.intermediate, eps: VISION_EPS }
    }

    /// Block weights with 2-D vision RoPE (no QK-norm/LayerScale).
    fn vit_bw<'a>(&'a self, b: u32, cos: &'a DeviceBuffer, sin: &'a DeviceBuffer, n: u32) -> VitBlockWeights<'a> {
        VitBlockWeights {
            norm1_w: self.wb(&format!("blocks.{b}.norm1.weight")),
            norm1_b: self.wb(&format!("blocks.{b}.norm1.bias")),
            qkv_w: self.wb(&format!("blocks.{b}.qkv.weight")),
            qkv_b: self.wb(&format!("blocks.{b}.qkv.bias")),
            qk_norm: None,
            rope: Some(RopeTables { cos, sin, tmod: n }),
            proj_w: self.wb(&format!("blocks.{b}.proj.weight")),
            proj_b: self.wb(&format!("blocks.{b}.proj.bias")),
            ls1: None,
            norm2_w: self.wb(&format!("blocks.{b}.norm2.weight")),
            norm2_b: self.wb(&format!("blocks.{b}.norm2.bias")),
            fc1_w: self.wb(&format!("blocks.{b}.fc1.weight")),
            fc1_b: self.wb(&format!("blocks.{b}.fc1.bias")),
            fc2_w: self.wb(&format!("blocks.{b}.fc2.weight")),
            fc2_b: self.wb(&format!("blocks.{b}.fc2.bias")),
            ls2: None,
        }
    }

    /// Training-mode forward (per-block [`VitBlockCache`] for the backward). No
    /// post-LN — the last block's output is the feature map (matching `encode`).
    pub fn forward_train(&self, grid_h: u32, grid_w: u32, pixels: &[f32]) -> QwenVitTrain {
        let g = self.gpu;
        let ids = vit_ids();
        let kb = vit_bwd_ids();
        let (c, pv) = (self.cfg.hidden, self.cfg.patch_vec_dim());
        let n = grid_h * grid_w;
        let sh = self.shape();
        assert_eq!(pixels.len(), (n * pv) as usize, "pixels must be [N, patch_vec]");

        let positions = vision_position_ids(grid_h, grid_w, self.cfg.spatial_merge_size);
        let (cos, sin) = vision_rope_tables(&positions, sh.head_dim(), VISION_ROPE_THETA);
        let cos_b = g.storage_init("vit.rope.cos", &cos);
        let sin_b = g.storage_init("vit.rope.sin", &sin);
        let pix = g.storage_init("vit.pixels", pixels);
        let pos = g.storage_init("vit.pos", &self.pos_embeds(grid_h, grid_w));
        let pe = g.storage((n * c) as u64);
        let caches: Vec<VitBlockCache> = (0..self.cfg.depth).map(|_| VitBlockCache::new(g, &sh, n, n)).collect();
        let x_last = g.storage((n * c) as u64);
        let scr_tmp = g.storage((n * c) as u64);
        let scores = g.storage((sh.heads * n * n) as u64);

        let mut steps: Vec<Step> = Vec::new();
        steps.push(g.step(ids.matmul, &[&pix, self.wb("patch_embed.weight"), &pe], &[n, pv, c], n * c));
        steps.push(g.step(ids.bias_add, &[&pe, self.wb("patch_embed.bias")], &[n, c], n * c));
        steps.push(g.step(ids.add2, &[&pe, &pos, &caches[0].x_in], &[n * c], n * c));
        for b in 0..self.cfg.depth as usize {
            let bw = self.vit_bw(b as u32, &cos_b, &sin_b, n);
            let x_out = if b + 1 < caches.len() { &caches[b + 1].x_in } else { &x_last };
            vit_block_fwd_cached(g, &ids, &kb, &sh, &bw, &caches[b], x_out, n, &[(0, n)], &scr_tmp, &scores, &mut steps);
        }
        g.submit(&[], &steps);
        QwenVitTrain { pix, caches, x_last, cos_b, sin_b, grid_h, grid_w, n }
    }

    /// ViT backward from the feature-map grad `d_out` (`[N, hidden]`): fill `gr` and
    /// return the input-patch grad `[N, patch_vec]`. No post-LN; blocks in reverse
    /// (`vit_block_bwd`, 2-D RoPE) → patch-embed + pos-embed (bilinear-transpose).
    pub fn backward(&self, tr: &QwenVitTrain, d_out: &DeviceBuffer, gr: &QwenVitGrads) -> Vec<f32> {
        let g = self.gpu;
        let ids = vit_ids();
        let kb = vit_bwd_ids();
        let (c, pv) = (self.cfg.hidden, self.cfg.patch_vec_dim());
        let n = tr.n;
        let sh = self.shape();
        let sb = VitBwdScratch::new(g, &sh, n, n);
        let d_x: Vec<DeviceBuffer> = (0..self.cfg.depth).map(|_| g.storage((n * c) as u64)).collect();

        for b in (0..self.cfg.depth as usize).rev() {
            let bw = self.vit_bw(b as u32, &tr.cos_b, &tr.sin_b, n);
            let bg = &gr.blocks[b];
            let vg = VitBlockGrads {
                norm1_w: &bg.norm1_w,
                norm1_b: &bg.norm1_b,
                qkv_w: &bg.qkv_w,
                qkv_b: &bg.qkv_b,
                q_norm_w: None,
                q_norm_b: None,
                k_norm_w: None,
                k_norm_b: None,
                proj_w: &bg.proj_w,
                proj_b: &bg.proj_b,
                ls1: None,
                norm2_w: &bg.norm2_w,
                norm2_b: &bg.norm2_b,
                fc1_w: &bg.fc1_w,
                fc1_b: &bg.fc1_b,
                fc2_w: &bg.fc2_w,
                fc2_b: &bg.fc2_b,
                ls2: None,
            };
            let d_out_b = if b + 1 < d_x.len() { &d_x[b + 1] } else { d_out };
            let mut s: Vec<Step> = Vec::new();
            vit_block_bwd(g, &ids, &kb, &sh, &bw, &vg, &tr.caches[b], d_out_b, &d_x[b], n, &[(0, n)], &sb, &mut s);
            g.submit(&[], &s);
        }
        // Patch-embed backward: d_x[0] is the grad of x0 = patch_embed(pix) + pos.
        let d_pix = g.storage((n * pv) as u64);
        g.submit(
            &[],
            &[
                g.step(15, &[&d_x[0], &tr.pix, &gr.patch_embed_w], &[n, pv, c], c * pv),
                g.step(16, &[&d_x[0], &gr.patch_embed_b], &[n, c], c),
                g.step(14, &[&d_x[0], self.wb("patch_embed.weight"), &d_pix], &[n, pv, c, 0], n * pv),
            ],
        );
        // Pos-embed grad: bilinear transpose of d_x0 back onto the raw pos table.
        let dx0 = g.read(&d_x[0], (n * c) as usize);
        let hidden = c as usize;
        let side = self.cfg.pos_grid();
        let (idx, wts) = pos_embed_bilinear(tr.grid_h, tr.grid_w, self.cfg.spatial_merge_size, side);
        let mut dpos = vec![0.0f32; self.pos_table.len()];
        for (p, (id, wt)) in idx.iter().zip(&wts).enumerate() {
            for ch in 0..hidden {
                let d = dx0[p * hidden + ch];
                for k in 0..4 {
                    dpos[id[k] as usize * hidden + ch] += d * wt[k];
                }
            }
        }
        g.write(&gr.pos_embed, &dpos.iter().map(|&v| f(v)).collect::<Vec<u32>>());
        g.read(&d_pix, (n * pv) as usize)
    }
}

/// Cached training-forward state for the Qwen3-VL ViT backward.
pub struct QwenVitTrain {
    pix: DeviceBuffer,
    caches: Vec<VitBlockCache>,
    pub x_last: DeviceBuffer,
    cos_b: DeviceBuffer,
    sin_b: DeviceBuffer,
    grid_h: u32,
    grid_w: u32,
    n: u32,
}

/// Per-block Qwen3-VL ViT parameter grads (zeroed on build).
pub struct QwenVitBlockGrads {
    norm1_w: DeviceBuffer,
    norm1_b: DeviceBuffer,
    qkv_w: DeviceBuffer,
    qkv_b: DeviceBuffer,
    proj_w: DeviceBuffer,
    proj_b: DeviceBuffer,
    norm2_w: DeviceBuffer,
    norm2_b: DeviceBuffer,
    fc1_w: DeviceBuffer,
    fc1_b: DeviceBuffer,
    fc2_w: DeviceBuffer,
    fc2_b: DeviceBuffer,
}

/// All Qwen3-VL ViT grads (patch-embed, pos-embed table, per block). No post-LN.
pub struct QwenVitGrads {
    pub patch_embed_w: DeviceBuffer,
    pub patch_embed_b: DeviceBuffer,
    pub pos_embed: DeviceBuffer,
    pub blocks: Vec<QwenVitBlockGrads>,
}

impl QwenVitGrads {
    pub fn new(g: &Gpu, cfg: &VisionConfig, num_pos: u32) -> QwenVitGrads {
        let z = |k: u32| g.storage_init("vit.g", &vec![0.0f32; k as usize]);
        let (c, m, pv) = (cfg.hidden, cfg.intermediate, cfg.patch_vec_dim());
        QwenVitGrads {
            patch_embed_w: z(c * pv),
            patch_embed_b: z(c),
            pos_embed: z(num_pos * c),
            blocks: (0..cfg.depth)
                .map(|_| QwenVitBlockGrads {
                    norm1_w: z(c),
                    norm1_b: z(c),
                    qkv_w: z(3 * c * c),
                    qkv_b: z(3 * c),
                    proj_w: z(c * c),
                    proj_b: z(c),
                    norm2_w: z(c),
                    norm2_b: z(c),
                    fc1_w: z(m * c),
                    fc1_b: z(m),
                    fc2_w: z(c * m),
                    fc2_b: z(c),
                })
                .collect(),
        }
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
    fn qwen_vit_backward_matches_finite_diff() {
        // Gradcheck the Qwen3-VL ViT training fwd/bwd (2-D vision RoPE): input-patch
        // grad exercises the tower; patch_embed/pos_embed(bilinear-transpose)/block
        // fc1 grads cover the rest.
        let cfg = tiny_cfg();
        let gpu = Gpu::new_cpu(vision_pipelines());
        let w = rand_weights(&cfg, 7);
        // Small grid keeps the summed loss (hence the central-diff roundoff floor) low.
        let (gh, gw) = (2u32, 2u32);
        let (c, pv) = (cfg.hidden as usize, cfg.patch_vec_dim() as usize);
        let n = (gh * gw) as usize;
        let mut rng = Rng::new(3);
        let pixels: Vec<f32> = (0..n * pv).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        let nn = n * c;

        let enc = VisionEncoder::new(&gpu, cfg.clone(), &w);
        let tr = enc.forward_train(gh, gw, &pixels);
        let d_out = gpu.storage_init("dout", &vec![1.0f32; nn]);
        let gr = QwenVitGrads::new(&gpu, &cfg, cfg.num_position_embeddings);
        let d_pix = enc.backward(&tr, &d_out, &gr);
        let g_pe = gpu.read(&gr.patch_embed_w, c * pv);
        let g_fc1 = gpu.read(&gr.blocks[0].fc1_w, (cfg.intermediate * cfg.hidden) as usize);
        let g_pos = gpu.read(&gr.pos_embed, (cfg.num_position_embeddings * cfg.hidden) as usize);

        let loss = |wm: &HashMap<String, Vec<f32>>, px: &[f32]| -> f32 {
            let e = VisionEncoder::new(&gpu, cfg.clone(), wm);
            gpu.read(&e.forward_train(gh, gw, px).x_last, nn).iter().sum::<f32>()
        };
        let eps = 1e-3f32;
        let ok = |a: f32, num: f32| (a - num).abs() <= 4e-3 + 8e-2 * num.abs();

        for &i in &[0usize, 5, 11, 20, 25, 30] {
            let (mut pp, mut pm) = (pixels.clone(), pixels.clone());
            pp[i] += eps;
            pm[i] -= eps;
            let num = (loss(&w, &pp) - loss(&w, &pm)) / (2.0 * eps);
            assert!(ok(d_pix[i], num), "d_pix[{i}]: analytic {} vs numeric {}", d_pix[i], num);
        }
        for (key, grad, idxs) in [
            ("patch_embed.weight", &g_pe, [0usize, 7, 200]),
            ("blocks.0.fc1.weight", &g_fc1, [0, 17, 40]),
            ("pos_embed", &g_pos, [0, 5, 11]),
        ] {
            for &j in &idxs {
                let (mut wp, mut wm2) = (w.clone(), w.clone());
                wp.get_mut(key).unwrap()[j] += eps;
                wm2.get_mut(key).unwrap()[j] -= eps;
                let num = (loss(&wp, &pixels) - loss(&wm2, &pixels)) / (2.0 * eps);
                assert!(ok(grad[j], num), "d {key}[{j}]: analytic {} vs numeric {}", grad[j], num);
            }
        }
    }

    #[test]
    fn encode_with_taps_captures_intermediate_blocks() {
        let cfg = tiny_cfg(); // depth 2
        let gpu = Gpu::new_cpu(vision_pipelines());
        let weights = rand_weights(&cfg, 7);
        let enc = VisionEncoder::new(&gpu, cfg.clone(), &weights);
        let (gh, gw) = (4u32, 4u32);
        let n = (gh * gw) as usize;
        let pv = cfg.patch_vec_dim() as usize;
        let mut rng = Rng::new(99);
        let pixels: Vec<f32> = (0..n * pv).map(|_| rng.next_f32() - 0.5).collect();

        // Tap block 0 (its output must differ from the final post-norm features).
        let (features, taps) = enc.encode_with_taps(gh, gw, &pixels, &[0]);
        assert_eq!(taps.len(), 1);
        assert_eq!(taps[0].len(), n * cfg.hidden as usize);
        assert!(taps[0].iter().all(|v| v.is_finite()) && taps[0].iter().any(|&v| v.abs() > 1e-6));
        assert!(taps[0].iter().zip(&features).any(|(a, b)| (a - b).abs() > 1e-4), "tap must differ from final features");

        // A DeepStack merger (postshuffle norm) consumes the tap → visual tokens.
        let (in_dim, merge, out_dim) = (cfg.hidden, cfg.spatial_merge_size, cfg.out_hidden_size);
        let mw = merger_weights(in_dim, merge, out_dim, true, 8);
        let ds = PatchMerger::new(&gpu, &mw, in_dim, merge, out_dim, true);
        let embeds = ds.merge(&taps[0], gh * gw);
        assert_eq!(embeds.len(), (gh * gw / (merge * merge) * out_dim) as usize);
        assert!(embeds.iter().all(|v| v.is_finite()));
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
