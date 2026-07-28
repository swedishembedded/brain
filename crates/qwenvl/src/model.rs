// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL composite: ViT encoder → PatchMerger → spliced M-RoPE Qwen decoder.
//!
//! Ties the pieces together for an end-to-end forward: the vision encoder produces
//! patch features, the main PatchMerger folds them into visual tokens at the
//! decoder width, and the decoder (with the image-embedding splice + interleaved
//! M-RoPE enabled) consumes them at the image-placeholder positions. The vision
//! side runs on its own `Gpu`; visual tokens cross to the decoder's `Gpu`
//! host-side via `write_img_embeds` (a fused single-device path is a later step,
//! as is DeepStack and the vision backward for full-tower finetune).

use std::collections::HashMap;

use gpu_core::Gpu;
use qwen::{Qwen, QwenConfig};

use crate::config::VisionConfig;
use crate::encoder::{vision_pipelines, PatchMerger, VisionEncoder};
use crate::mrope::{get_rope_index, mrope_tables};

/// An assembled Qwen3-VL model (forward path). Image tokens occupy a contiguous
/// run of `image_token_id` in the text stream starting at `image_row0`.
pub struct Qwen3Vl {
    vgpu: Gpu,
    vcfg: VisionConfig,
    vweights: HashMap<String, Vec<f32>>,
    merger_weights: HashMap<String, Vec<f32>>,
    decoder: Qwen,
    merge: u32,
    image_token_id: u32,
    mrope_section: [u32; 3],
    image_row0: u32,
}

impl Qwen3Vl {
    /// Assemble from a vision config, a decoder config (its `d_model` must equal
    /// the merger output width), pre-uploaded host weights, and the image
    /// placement. `enable_mm_splice`/`enable_mrope` are wired on the decoder here.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vcfg: VisionConfig,
        dcfg: QwenConfig,
        vweights: HashMap<String, Vec<f32>>,
        merger_weights: HashMap<String, Vec<f32>>,
        dweights: &HashMap<String, Vec<f32>>,
        seq_len: u32,
        image_token_id: u32,
        image_row0: u32,
        n_visual: u32,
        mrope_section: [u32; 3],
    ) -> Qwen3Vl {
        let merge = vcfg.spatial_merge_size;
        let mut decoder = Qwen::new(dcfg, 1, seq_len, dweights);
        decoder.enable_mm_splice(image_row0, n_visual);
        decoder.enable_mrope();
        Qwen3Vl {
            vgpu: Gpu::new_cpu(vision_pipelines()),
            vcfg,
            vweights,
            merger_weights,
            decoder,
            merge,
            image_token_id,
            mrope_section,
            image_row0,
        }
    }

    /// End-to-end forward for one image + text stream; returns the decoder's scalar
    /// loss. `pixels` is the host-packed `[grid_h·grid_w, patch_vec]` patch tensor;
    /// `tokens`/`targets` are the full text stream (image placeholders carry IGNORE
    /// targets). Panics if the visual-token count disagrees with the placement.
    pub fn forward(&self, tokens: &[u32], targets: &[u32], grid: (u32, u32), pixels: &[f32]) -> f32 {
        let (gh, gw) = grid;
        let n = gh * gw;
        let m2 = self.merge * self.merge;
        let n_visual = n / m2;
        let d_model = self.decoder.cfg.d_model;

        // Vision tower → visual tokens at the decoder width.
        let enc = VisionEncoder::new(&self.vgpu, self.vcfg.clone(), &self.vweights);
        let feats = enc.encode(gh, gw, pixels);
        let merger = PatchMerger::new(&self.vgpu, &self.merger_weights, self.vcfg.hidden, self.merge, d_model, false);
        let visual = merger.merge(&feats, n);
        assert_eq!(visual.len(), (n_visual * d_model) as usize);

        // M-RoPE tables from the 3-axis position ids for this stream.
        let grids_llm = [(1, gh / self.merge, gw / self.merge)];
        let positions = get_rope_index(tokens, self.image_token_id, &grids_llm);
        let (cos, sin) = mrope_tables(&positions, self.mrope_section, self.decoder.cfg.head_dim, self.decoder.cfg.rope_theta);

        // Splice + decode.
        self.decoder.write_mrope_tables(&cos, &sin);
        self.decoder.write_img_embeds(&visual);
        self.decoder.set_batch(tokens, targets);
        let _ = self.image_row0; // (placement already baked into enable_mm_splice)
        self.decoder.forward()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;

    const IMG: u32 = 7;

    fn rand_map(mut rng: Rng, specs: &[(&str, usize, bool)]) -> HashMap<String, Vec<f32>> {
        let mut m = HashMap::new();
        for &(name, n, ones) in specs {
            let v = if ones { vec![1.0; n] } else { (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect() };
            m.insert(name.to_string(), v);
        }
        m
    }

    #[test]
    fn end_to_end_forward_is_finite() {
        // Tiny dims with everything aligned: vision hidden 32, merge 2 →
        // merged 128; decoder d_model 40 = merger out; head_dim 8 → mrope [2,1,1].
        let vcfg = VisionConfig {
            depth: 2,
            hidden: 32,
            num_heads: 2,
            intermediate: 64,
            patch_size: 2,
            temporal_patch_size: 1,
            spatial_merge_size: 2,
            num_position_embeddings: 16,
            out_hidden_size: 40,
            in_channels: 2,
            deepstack_indexes: vec![],
        };
        let dcfg = QwenConfig {
            vocab: 23,
            block_size: 16,
            n_layers: 2,
            d_model: 40,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8,
            d_ff: 64,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            tie_embeddings: true,
            lora: None,
        };

        // Vision + merger weights.
        let (c, pv, mlp) = (vcfg.hidden as usize, vcfg.patch_vec_dim() as usize, vcfg.intermediate as usize);
        let mut vspecs: Vec<(&str, usize, bool)> = vec![
            ("patch_embed.weight", c * pv, false),
            ("patch_embed.bias", c, false),
            ("pos_embed", vcfg.num_position_embeddings as usize * c, false),
            ("norm.weight", c, true),
            ("norm.bias", c, false),
        ];
        let block_leaf_dims: Vec<(String, usize, bool)> = (0..vcfg.depth)
            .flat_map(|b| {
                [
                    (format!("blocks.{b}.norm1.weight"), c, true),
                    (format!("blocks.{b}.norm1.bias"), c, false),
                    (format!("blocks.{b}.qkv.weight"), 3 * c * c, false),
                    (format!("blocks.{b}.qkv.bias"), 3 * c, false),
                    (format!("blocks.{b}.proj.weight"), c * c, false),
                    (format!("blocks.{b}.proj.bias"), c, false),
                    (format!("blocks.{b}.norm2.weight"), c, true),
                    (format!("blocks.{b}.norm2.bias"), c, false),
                    (format!("blocks.{b}.fc1.weight"), mlp * c, false),
                    (format!("blocks.{b}.fc1.bias"), mlp, false),
                    (format!("blocks.{b}.fc2.weight"), c * mlp, false),
                    (format!("blocks.{b}.fc2.bias"), c, false),
                ]
            })
            .collect();
        for (n, s, o) in &block_leaf_dims {
            vspecs.push((n.as_str(), *s, *o));
        }
        let vweights = rand_map(Rng::new(1), &vspecs);

        let merged = (c * 4) as usize; // in_dim·merge²
        let mweights = rand_map(
            Rng::new(2),
            &[
                ("ln.weight", c, true),
                ("ln.bias", c, false),
                ("fc1.weight", merged * merged, false),
                ("fc1.bias", merged, false),
                ("fc2.weight", 40 * merged, false),
                ("fc2.bias", 40, false),
            ],
        );

        let dweights = qwen::init_weights(&dcfg, 3);

        // Stream: 2 text, 4 image (2×2 grid merged), 1 text. IGNORE at image rows.
        let tokens: Vec<u32> = vec![1, 2, IMG, IMG, IMG, IMG, 3];
        let mut targets = vec![2u32, 3, 0, 0, 0, 0, 5];
        for t in targets.iter_mut().take(6).skip(2) {
            *t = qwen::IGNORE;
        }

        let model = Qwen3Vl::new(vcfg.clone(), dcfg, vweights, mweights, &dweights, tokens.len() as u32, IMG, 2, 4, [2, 1, 1]);

        let pv_total = (16 * vcfg.patch_vec_dim()) as usize;
        let mut rng = Rng::new(4);
        let pixels: Vec<f32> = (0..pv_total).map(|_| rng.next_f32() - 0.5).collect();

        let loss = model.forward(&tokens, &targets, (4, 4), &pixels);
        assert!(loss.is_finite(), "end-to-end loss must be finite, got {loss}");
        assert!(loss > 0.0, "cross-entropy loss should be positive");
    }
}
