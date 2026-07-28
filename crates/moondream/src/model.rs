// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream 3 composite: SigLIP ViT → connector → spliced MoE decoder.
//!
//! Mirrors `qwenvl::Qwen3Vl`/`fastvlm::FastVlm`: the vision tower runs on its own
//! `Gpu`, image tokens cross host-side into the decoder's positional prefix. The
//! vision→connector→decoder pieces are rebuilt per forward from stored weights
//! (like the other composites). NB: this wires the single-crop path — the global‖
//! local overlap-multi-crop concat that widens the connector input to `2·dim`
//! (and the tau temperature + backward/gradcheck) are follow-ups.

use std::collections::HashMap;

use gpu_core::Gpu;

use crate::config::MoondreamConfig;
use crate::decoder::{MoeFfn, MoondreamBlock, MoondreamDecoder};
use crate::vision::{vision_pipelines, Connector, SiglipEncoder};

/// An assembled Moondream 3 (forward path). Image tokens occupy rows `[1, 1+n_img)`.
pub struct MoondreamModel {
    vgpu: Gpu,
    dgpu: Gpu,
    cfg: MoondreamConfig,
    vweights: HashMap<String, Vec<f32>>,
    conn_weights: HashMap<String, Vec<f32>>,
    dweights: HashMap<String, Vec<f32>>,
    conn_in: u32,
    seq_len: u32,
}

impl MoondreamModel {
    pub fn new(
        cfg: MoondreamConfig,
        vweights: HashMap<String, Vec<f32>>,
        conn_weights: HashMap<String, Vec<f32>>,
        dweights: HashMap<String, Vec<f32>>,
        conn_in: u32,
        seq_len: u32,
    ) -> MoondreamModel {
        MoondreamModel {
            vgpu: Gpu::new_cpu(vision_pipelines()),
            dgpu: Gpu::new_cpu(crate::decoder::pipelines()),
            cfg,
            vweights,
            conn_weights,
            dweights,
            conn_in,
            seq_len,
        }
    }

    /// Build the block stack for one forward (dense 0..3, MoE 4..23) from the
    /// prefixed decoder weights.
    fn build_blocks<'g>(&self, gpu: &'g Gpu, t: u32) -> Vec<MoondreamBlock<'g>> {
        let c = &self.cfg;
        (0..c.n_layers)
            .map(|l| {
                let bw = strip_prefix(&self.dweights, &format!("blocks.{l}."));
                let blk = MoondreamBlock::new(gpu, &bw, t, c.dim, c.n_heads, c.head_dim, c.ff_dim, c.prefix_attn, c.rot_dim, c.rope_theta);
                if c.is_moe_layer(l) {
                    let mw = strip_prefix(&self.dweights, &format!("blocks.{l}.moe."));
                    blk.with_moe(MoeFfn::new(gpu, &mw, t, c.dim, c.moe.inner_dim, c.moe.num_experts, c.moe.top_k))
                } else {
                    blk
                }
            })
            .collect()
    }

    /// End-to-end forward: encode one crop, project, splice, decode → loss.
    /// `tokens`/`targets` length `seq_len`; `packed` is `[patches, patch_vec]`.
    pub fn forward(&self, tokens: &[u32], targets: &[u32], packed: &[f32]) -> f32 {
        let enc = SiglipEncoder::new(&self.vgpu, self.cfg.vision.clone(), &self.vweights);
        let feats = enc.encode(1, packed);
        let ppc = self.cfg.vision.patches_per_crop();
        let conn = Connector::new(&self.vgpu, &self.conn_weights, self.conn_in, self.cfg.proj_inner, self.cfg.proj_out);
        let img_embeds = conn.forward(ppc, &feats);

        let blocks = self.build_blocks(&self.dgpu, self.seq_len);
        let dec = MoondreamDecoder::new(&self.dgpu, &self.dweights, blocks, self.seq_len, self.cfg.dim, self.cfg.vocab, ppc);
        dec.forward(tokens, targets, &img_embeds)
    }

    /// Faithful overlap multi-crop forward: encode the global crop and `h·w` local
    /// crops, reconstruct + adaptive-pool the locals and channel-concat with the
    /// global into the `[729, 2·dim]` connector input, project, splice, decode →
    /// loss. Requires `conn_in == 2·vision.dim`. `global_packed` is one crop's
    /// `[ppc, patch_vec]`; `locals_packed` is `[h·w·ppc, patch_vec]` (tile order).
    pub fn forward_multicrop(&self, tokens: &[u32], targets: &[u32], global_packed: &[f32], locals_packed: &[f32], h_tiles: u32, w_tiles: u32) -> f32 {
        let (dim, grid, margin) = (self.cfg.vision.dim, self.cfg.vision.grid(), self.cfg.vision.overlap_margin);
        assert_eq!(self.conn_in, 2 * dim, "multi-crop connector input must be 2·vision.dim");
        let ppc = self.cfg.vision.patches_per_crop();
        let n_local = h_tiles * w_tiles;

        let enc = SiglipEncoder::new(&self.vgpu, self.cfg.vision.clone(), &self.vweights);
        let global = enc.encode(1, global_packed);
        let locals = enc.encode(n_local, locals_packed);
        let concat = crate::preprocess::build_connector_input(&self.vgpu, &global, &locals, h_tiles, w_tiles, grid, dim, margin);

        let conn = Connector::new(&self.vgpu, &self.conn_weights, self.conn_in, self.cfg.proj_inner, self.cfg.proj_out);
        let img_embeds = conn.forward(ppc, &concat);

        let blocks = self.build_blocks(&self.dgpu, self.seq_len);
        let dec = MoondreamDecoder::new(&self.dgpu, &self.dweights, blocks, self.seq_len, self.cfg.dim, self.cfg.vocab, ppc);
        dec.forward(tokens, targets, &img_embeds)
    }
}

/// Extract the entries of `w` whose key starts with `prefix`, with the prefix
/// stripped (and dropping any deeper `moe.` sub-keys for the block-level map).
fn strip_prefix(w: &HashMap<String, Vec<f32>>, prefix: &str) -> HashMap<String, Vec<f32>> {
    w.iter()
        .filter_map(|(k, v)| k.strip_prefix(prefix).map(|s| (s.to_string(), v.clone())))
        .filter(|(k, _)| !k.starts_with("moe.")) // block map excludes the MoE sub-tree
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{MoeConfig, VisionConfig};
    use data::rng::Rng;
    use crate::decoder::IGNORE;

    #[test]
    fn end_to_end_image_to_loss() {
        // Tiny everything: ViT 4×4-patch dim 32 → connector 32→48→24 → decoder d24.
        let vision = VisionConfig { dim: 32, patch: 2, n_layers: 2, ff_dim: 64, n_heads: 2, crop_size: 8, max_crops: 4, overlap_margin: 1 };
        let cfg = MoondreamConfig {
            dim: 24,
            ff_dim: 48,
            n_layers: 2,
            vocab: 23,
            n_heads: 3,
            head_dim: 8,
            prefix_attn: 17, // 1 bos + 16 image
            rot_dim: 4,
            rope_theta: 1.5e6,
            proj_inner: 48,
            proj_out: 24,
            vision: vision.clone(),
            moe: MoeConfig { num_experts: 3, start_layer: 1, top_k: 2, inner_dim: 8 },
        };
        let ppc = vision.patches_per_crop(); // 16
        let (c, pv) = (vision.dim as usize, vision.patch_vec() as usize);
        let mut rng = Rng::new(2);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();

        // Vision weights.
        let mut vw = HashMap::new();
        vw.insert("patch_emb.weight".into(), r(c * pv));
        vw.insert("patch_emb.bias".into(), r(c));
        vw.insert("pos_emb".into(), r(ppc as usize * c));
        vw.insert("post_ln.weight".into(), vec![1.0; c]);
        vw.insert("post_ln.bias".into(), r(c));
        for b in 0..vision.n_layers {
            for (leaf, sz) in [
                ("ln1.weight", c), ("ln1.bias", c), ("attn.qkv.weight", 3 * c * c), ("attn.qkv.bias", 3 * c),
                ("attn.proj.weight", c * c), ("attn.proj.bias", c), ("ln2.weight", c), ("ln2.bias", c),
                ("mlp.fc1.weight", vision.ff_dim as usize * c), ("mlp.fc1.bias", vision.ff_dim as usize),
                ("mlp.fc2.weight", c * vision.ff_dim as usize), ("mlp.fc2.bias", c),
            ] {
                let v = if leaf.ends_with("ln1.weight") || leaf.ends_with("ln2.weight") { vec![1.0; sz] } else { r(sz) };
                vw.insert(format!("blocks.{b}.{leaf}"), v);
            }
        }
        // Connector 32→48→24.
        let mut cw = HashMap::new();
        cw.insert("fc1.weight".into(), r((cfg.proj_inner * vision.dim) as usize)); // [inner, in=dim]
        cw.insert("fc1.bias".into(), r(cfg.proj_inner as usize));
        cw.insert("fc2.weight".into(), r((cfg.proj_out * cfg.proj_inner) as usize));
        cw.insert("fc2.bias".into(), r(cfg.proj_out as usize));
        // Decoder weights.
        let (d, ff, vocab) = (cfg.dim, cfg.ff_dim, cfg.vocab);
        let mut dw = HashMap::new();
        dw.insert("tok.weight".into(), r((vocab * d) as usize));
        dw.insert("post_ln.weight".into(), vec![1.0; d as usize]);
        dw.insert("post_ln.bias".into(), r(d as usize));
        dw.insert("lm_head.weight".into(), r((vocab * d) as usize));
        dw.insert("lm_head.bias".into(), r(vocab as usize));
        for l in 0..cfg.n_layers {
            for (leaf, sz) in [
                ("ln.weight", d as usize), ("ln.bias", d as usize), ("attn.qkv.weight", (3 * d * d) as usize),
                ("attn.proj.weight", (d * d) as usize), ("attn.proj.bias", d as usize),
                ("mlp.fc1.weight", (ff * d) as usize), ("mlp.fc1.bias", ff as usize),
                ("mlp.fc2.weight", (d * ff) as usize), ("mlp.fc2.bias", d as usize),
            ] {
                let v = if leaf.ends_with("ln.weight") { vec![1.0; sz] } else { r(sz) };
                dw.insert(format!("blocks.{l}.{leaf}"), v);
            }
            if cfg.is_moe_layer(l) {
                let (inner, e) = (cfg.moe.inner_dim, cfg.moe.num_experts);
                dw.insert(format!("blocks.{l}.moe.router.weight"), r((e * d) as usize));
                for ei in 0..e {
                    dw.insert(format!("blocks.{l}.moe.experts.{ei}.w_h.weight"), r((inner * d) as usize));
                    dw.insert(format!("blocks.{l}.moe.experts.{ei}.w_g.weight"), r((inner * d) as usize));
                    dw.insert(format!("blocks.{l}.moe.experts.{ei}.w_down.weight"), r((d * inner) as usize));
                }
            }
        }

        let seq = 1 + ppc + 3; // bos + image + 3 text = 20
        let model = MoondreamModel::new(cfg, vw, cw, dw, vision.dim, seq);
        let mut tokens = vec![0u32]; // bos
        tokens.extend(std::iter::repeat(5u32).take(ppc as usize)); // image placeholders
        tokens.extend([7u32, 9, 11]);
        let mut targets = tokens[1..].to_vec();
        targets.push(13);
        for tg in targets.iter_mut().take(1 + ppc as usize) {
            *tg = IGNORE; // bos + image rows not supervised
        }
        let packed: Vec<f32> = r((ppc * vision.patch_vec()) as usize);
        let loss = model.forward(&tokens, &targets, &packed);
        assert!(loss.is_finite() && loss > 0.0, "moondream end-to-end loss must be finite+positive, got {loss}");
    }

    #[test]
    fn multicrop_forward_with_tau_is_finite() {
        // Faithful path: global + 2×2 local crops → [ppc,2·dim] concat → connector.
        // conn_in = 2·vision.dim; tau enabled on the decoder blocks.
        let vision = VisionConfig { dim: 16, patch: 2, n_layers: 2, ff_dim: 32, n_heads: 2, crop_size: 8, max_crops: 4, overlap_margin: 1 };
        let cfg = MoondreamConfig {
            dim: 24, ff_dim: 48, n_layers: 2, vocab: 23, n_heads: 3, head_dim: 8, prefix_attn: 17,
            rot_dim: 4, rope_theta: 1.5e6, proj_inner: 48, proj_out: 24, vision: vision.clone(),
            moe: MoeConfig { num_experts: 3, start_layer: 1, top_k: 2, inner_dim: 8 },
        };
        let ppc = vision.patches_per_crop();
        let pv = vision.patch_vec() as usize;
        let mut rng = Rng::new(4);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();

        let mut vw = HashMap::new();
        vw.insert("patch_emb.weight".into(), r(vision.dim as usize * pv));
        vw.insert("patch_emb.bias".into(), r(vision.dim as usize));
        vw.insert("pos_emb".into(), r(ppc as usize * vision.dim as usize));
        vw.insert("post_ln.weight".into(), vec![1.0; vision.dim as usize]);
        vw.insert("post_ln.bias".into(), r(vision.dim as usize));
        let c = vision.dim as usize;
        for b in 0..vision.n_layers {
            for (leaf, sz) in [
                ("ln1.weight", c), ("ln1.bias", c), ("attn.qkv.weight", 3 * c * c), ("attn.qkv.bias", 3 * c),
                ("attn.proj.weight", c * c), ("attn.proj.bias", c), ("ln2.weight", c), ("ln2.bias", c),
                ("mlp.fc1.weight", vision.ff_dim as usize * c), ("mlp.fc1.bias", vision.ff_dim as usize),
                ("mlp.fc2.weight", c * vision.ff_dim as usize), ("mlp.fc2.bias", c),
            ] {
                let v = if leaf.ends_with("ln1.weight") || leaf.ends_with("ln2.weight") { vec![1.0; sz] } else { r(sz) };
                vw.insert(format!("blocks.{b}.{leaf}"), v);
            }
        }
        // Connector in = 2·dim.
        let conn_in = 2 * vision.dim;
        let mut cw = HashMap::new();
        cw.insert("fc1.weight".into(), r((cfg.proj_inner * conn_in) as usize));
        cw.insert("fc1.bias".into(), r(cfg.proj_inner as usize));
        cw.insert("fc2.weight".into(), r((cfg.proj_out * cfg.proj_inner) as usize));
        cw.insert("fc2.bias".into(), r(cfg.proj_out as usize));
        // Decoder weights incl. tau.
        let (d, ff, vocab) = (cfg.dim, cfg.ff_dim, cfg.vocab);
        let mut dw = HashMap::new();
        dw.insert("tok.weight".into(), r((vocab * d) as usize));
        dw.insert("post_ln.weight".into(), vec![1.0; d as usize]);
        dw.insert("post_ln.bias".into(), r(d as usize));
        dw.insert("lm_head.weight".into(), r((vocab * d) as usize));
        dw.insert("lm_head.bias".into(), r(vocab as usize));
        for l in 0..cfg.n_layers {
            for (leaf, sz) in [
                ("ln.weight", d as usize), ("ln.bias", d as usize), ("attn.qkv.weight", (3 * d * d) as usize),
                ("attn.proj.weight", (d * d) as usize), ("attn.proj.bias", d as usize),
                ("attn.tau.wq", (cfg.n_heads * 3 * d) as usize), ("attn.tau.wv", (cfg.n_heads * 3 * d) as usize), ("attn.tau.alpha", cfg.n_heads as usize),
                ("mlp.fc1.weight", (ff * d) as usize), ("mlp.fc1.bias", ff as usize),
                ("mlp.fc2.weight", (d * ff) as usize), ("mlp.fc2.bias", d as usize),
            ] {
                let v = if leaf.ends_with("ln.weight") { vec![1.0; sz] } else { r(sz) };
                dw.insert(format!("blocks.{l}.{leaf}"), v);
            }
            if cfg.is_moe_layer(l) {
                let (inner, e) = (cfg.moe.inner_dim, cfg.moe.num_experts);
                dw.insert(format!("blocks.{l}.moe.router.weight"), r((e * d) as usize));
                for ei in 0..e {
                    dw.insert(format!("blocks.{l}.moe.experts.{ei}.w_h.weight"), r((inner * d) as usize));
                    dw.insert(format!("blocks.{l}.moe.experts.{ei}.w_g.weight"), r((inner * d) as usize));
                    dw.insert(format!("blocks.{l}.moe.experts.{ei}.w_down.weight"), r((d * inner) as usize));
                }
            }
        }

        let seq = 1 + ppc + 3;
        let model = MoondreamModel::new(cfg, vw, cw, dw, conn_in, seq);
        let mut tokens = vec![0u32];
        tokens.extend(std::iter::repeat(5u32).take(ppc as usize));
        tokens.extend([7u32, 9, 11]);
        let mut targets = tokens[1..].to_vec();
        targets.push(13);
        for tg in targets.iter_mut().take(1 + ppc as usize) {
            *tg = IGNORE;
        }
        let (ht, wt) = (2u32, 2u32);
        let global: Vec<f32> = r((ppc * vision.patch_vec()) as usize);
        let locals: Vec<f32> = r((ht * wt * ppc * vision.patch_vec()) as usize);
        let loss = model.forward_multicrop(&tokens, &targets, &global, &locals, ht, wt);
        assert!(loss.is_finite() && loss > 0.0, "multi-crop loss must be finite+positive, got {loss}");
    }
}
