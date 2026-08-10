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
use qwen3::{Qwen, QwenConfig};

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
    /// One postshuffle-norm merger weight set per DeepStack tap (empty = no DeepStack).
    ds_merger_weights: Vec<HashMap<String, Vec<f32>>>,
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
        ds_merger_weights: Vec<HashMap<String, Vec<f32>>>,
        dweights: &HashMap<String, Vec<f32>>,
        seq_len: u32,
        image_token_id: u32,
        image_row0: u32,
        n_visual: u32,
        mrope_section: [u32; 3],
    ) -> Qwen3Vl {
        assert_eq!(ds_merger_weights.len(), vcfg.deepstack_indexes.len(), "one merger per DeepStack tap");
        let merge = vcfg.spatial_merge_size;
        let mut decoder = Qwen::new(dcfg, 1, seq_len, dweights);
        decoder.enable_mm_splice(image_row0, n_visual);
        decoder.enable_mrope();
        if !ds_merger_weights.is_empty() {
            decoder.enable_deepstack(image_row0, n_visual, ds_merger_weights.len() as u32);
        }
        Qwen3Vl {
            vgpu: Gpu::new_cpu(vision_pipelines()),
            vcfg,
            vweights,
            merger_weights,
            ds_merger_weights,
            decoder,
            merge,
            image_token_id,
            mrope_section,
            image_row0,
        }
    }

    /// Assemble from already-loaded HF tensors (name → f32). Partitions them via
    /// [`crate::import`] and constructs the model for a fixed image placement.
    #[allow(clippy::too_many_arguments)]
    pub fn from_tensors(
        tensors: Vec<checkpoint::safetensors::StTensor>,
        vcfg: VisionConfig,
        dcfg: QwenConfig,
        seq_len: u32,
        image_token_id: u32,
        image_row0: u32,
        n_visual: u32,
        mrope_section: [u32; 3],
    ) -> Qwen3Vl {
        let map: HashMap<String, Vec<f32>> = tensors.into_iter().map(|t| (t.name, t.data)).collect();
        let w = crate::import::partition(map, vcfg.deepstack_indexes.len());
        Qwen3Vl::new(
            vcfg,
            dcfg,
            w.vision,
            w.main_merger,
            w.deepstack,
            &w.decoder,
            seq_len,
            image_token_id,
            image_row0,
            n_visual,
            mrope_section,
        )
    }

    /// Load a Hugging Face Qwen3-VL checkpoint directory (`config.json` +
    /// `model.safetensors[.index.json]`, bf16 → f32) and assemble the model for a
    /// fixed image placement. Note the released 4B checkpoint is ~16 GB in f32.
    #[allow(clippy::too_many_arguments)]
    pub fn from_hf(
        dir: &str,
        vcfg: VisionConfig,
        dcfg: QwenConfig,
        seq_len: u32,
        image_token_id: u32,
        image_row0: u32,
        n_visual: u32,
        mrope_section: [u32; 3],
    ) -> Result<Qwen3Vl, String> {
        let tensors = checkpoint::safetensors::read_model_dir(std::path::Path::new(dir))?;
        Ok(Self::from_tensors(tensors, vcfg, dcfg, seq_len, image_token_id, image_row0, n_visual, mrope_section))
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

        // Vision tower → visual tokens at the decoder width (+ DeepStack taps).
        let enc = VisionEncoder::new(&self.vgpu, self.vcfg.clone(), &self.vweights);
        let (feats, tap_feats) = enc.encode_with_taps(gh, gw, pixels, &self.vcfg.deepstack_indexes);
        let merger = PatchMerger::new(&self.vgpu, &self.merger_weights, self.vcfg.hidden, self.merge, d_model, false);
        let visual = merger.merge(&feats, n);
        assert_eq!(visual.len(), (n_visual * d_model) as usize);

        // DeepStack: each tap → its own postshuffle merger → decoder level buffer.
        for (level, (tap, mw)) in tap_feats.iter().zip(&self.ds_merger_weights).enumerate() {
            let ds = PatchMerger::new(&self.vgpu, mw, self.vcfg.hidden, self.merge, d_model, true);
            self.decoder.write_deepstack(level, &ds.merge(tap, n));
        }

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

    /// Greedy KV-cache generation for one image + text prompt: real
    /// `qwen3::Qwen` `step`/`step_embed` machinery (via this session's new
    /// `step_mrope`/`step_embed_mrope`, Phase 7a), not [`Self::forward`]'s
    /// training-loss-shaped batched path -- the gap `docs/models/omni/
    /// status.md`'s own note names ("`Qwen3Vl::forward()` returns `f32`...
    /// there is no sampling loop").
    ///
    /// Prefill splices the image at its token run the same way
    /// [`Self::forward`] does (image-placeholder token ids → step_embed_mrope
    /// with the matching merged visual row; every other token → step_mrope),
    /// so the KV cache never knows the difference (mirrors `qwen3::Qwen::
    /// prefill`'s own doc). Decode continues the position sequence past the
    /// prompt as plain text (T=H=W, +1 per token — the same "media block
    /// then plain text" case `qwenvl::mrope::get_rope_index_multi` documents).
    ///
    /// Validation-tier: greedy argmax only (no temperature/top-k/top-p),
    /// matching every other validation-tier `generate` in this repo (e.g.
    /// `omni::generate::generate_greedy`) — a real sampling policy is a
    /// separate, later concern. Returns the generated token ids (prompt not
    /// included), stopping early at any id in `eos_ids`.
    ///
    /// **DeepStack IS applied here**: `qwen3::Qwen::decode_steps`'s
    /// `deepstack_row` parameter adds each level's per-row residual
    /// contribution during the incremental step that embeds that row (was
    /// missing before this session — `qwen3::Qwen::enable_deepstack`'s
    /// `SPLICE_ADD` used to be wired ONLY into the batched `forward_steps()`
    /// graph, now also threaded into incremental decode via `decode_steps`'s
    /// `deepstack_row` parameter; see also
    /// `crates/qwen3/tests/deepstack_decode_parity.rs`).
    pub fn generate(&self, tokens: &[u32], grid: (u32, u32), pixels: &[f32], max_new: u32, eos_ids: &[u32]) -> Vec<u32> {
        let (gh, gw) = grid;
        let n = gh * gw;
        let m2 = self.merge * self.merge;
        let n_visual = n / m2;
        let d_model = self.decoder.cfg.d_model as usize;

        // Vision tower -> visual tokens (+ DeepStack taps), same as forward().
        let enc = VisionEncoder::new(&self.vgpu, self.vcfg.clone(), &self.vweights);
        let (feats, tap_feats) = enc.encode_with_taps(gh, gw, pixels, &self.vcfg.deepstack_indexes);
        let merger = PatchMerger::new(&self.vgpu, &self.merger_weights, self.vcfg.hidden, self.merge, d_model as u32, false);
        let visual = merger.merge(&feats, n);
        assert_eq!(visual.len(), (n_visual as usize) * d_model);
        for (level, (tap, mw)) in tap_feats.iter().zip(&self.ds_merger_weights).enumerate() {
            let ds = PatchMerger::new(&self.vgpu, mw, self.vcfg.hidden, self.merge, d_model as u32, true);
            self.decoder.write_deepstack(level, &ds.merge(tap, n));
        }

        // M-RoPE positions for the KNOWN prompt (whole-sequence, once).
        let grids_llm = [(1, gh / self.merge, gw / self.merge)];
        let prompt_positions = get_rope_index(tokens, self.image_token_id, &grids_llm);
        assert_eq!(prompt_positions.len(), tokens.len());

        // Prefill: image rows via step_embed_mrope, text rows via
        // step_mrope, each with its own 1-row M-RoPE table (mrope_tables
        // called per-position -- the plan's own recommended shape, "a
        // single-element positions slice").
        self.decoder.reset_cache();
        let mut visual_row = 0usize;
        let mut hidden = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            let (cos, sin) = mrope_tables(&prompt_positions[i..=i], self.mrope_section, self.decoder.cfg.head_dim, self.decoder.cfg.rope_theta);
            hidden = if tok == self.image_token_id {
                let row = &visual[visual_row * d_model..(visual_row + 1) * d_model];
                let ds_row = Some(visual_row as u32);
                visual_row += 1;
                self.decoder.step_embed_mrope(row, &cos, &sin, ds_row)
            } else {
                self.decoder.step_mrope(tok, &cos, &sin)
            };
        }
        assert_eq!(visual_row, n_visual as usize, "image token count in the prompt must match n_visual");

        // Decode: greedy argmax, continuing the position sequence past the
        // prompt as plain text.
        let head = self.decoder.read_weight(self.decoder.cfg.head_weight());
        let vocab = self.decoder.cfg.vocab as usize;
        let mut next_pos = prompt_positions.last().map(|p| p[0] + 1).unwrap_or(0);
        let mut out = Vec::with_capacity(max_new as usize);
        for _ in 0..max_new {
            let next = argmax_tied_head(&head, &hidden, vocab, d_model);
            if eos_ids.contains(&next) {
                break;
            }
            out.push(next);
            let (cos, sin) = mrope_tables(&[[next_pos; 3]], self.mrope_section, self.decoder.cfg.head_dim, self.decoder.cfg.rope_theta);
            hidden = self.decoder.step_mrope(next, &cos, &sin);
            next_pos += 1;
        }
        out
    }
}

/// `argmax_i(head[i] . hidden)` -- the host-side tied/untied head application
/// every KV-cache decode path in this engine uses (`qwen3::sample`'s own doc:
/// "The tied/untied head is applied on the host to the final-norm hidden
/// state"), inlined rather than imported since `qwen3::sample`'s equivalent
/// (`sample_logits`) is private and bundled with temperature/top-k/top-p
/// machinery this validation-tier greedy path does not need.
fn argmax_tied_head(head: &[f32], hidden: &[f32], vocab: usize, d_model: usize) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for row in 0..vocab {
        let wr = &head[row * d_model..(row + 1) * d_model];
        let v: f32 = wr.iter().zip(hidden).map(|(a, b)| a * b).sum();
        if v > best_v {
            best_v = v;
            best = row;
        }
    }
    best as u32
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
            deepstack_indexes: vec![0, 1], // tap both blocks → decoder layers 0,1
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
            max_position_embeddings: 16,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        };

        // Vision + merger weights.
        let (c, pv, mlp) = (vcfg.hidden as usize, vcfg.patch_vec_dim() as usize, vcfg.intermediate as usize);
        let mut vspecs: Vec<(&str, usize, bool)> = vec![
            ("patch_embed.weight", c * pv, false),
            ("patch_embed.bias", c, false),
            ("pos_embed", vcfg.num_position_embeddings as usize * c, false),
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

        let merged = c * 4; // in_dim·merge²
        // Main merger: LayerNorm over in_dim (postshuffle_norm=false).
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
        // DeepStack mergers (one per tap): LayerNorm over merged (postshuffle_norm=true).
        let ds_mweights: Vec<HashMap<String, Vec<f32>>> = (0..2u64)
            .map(|i| {
                rand_map(
                    Rng::new(20 + i),
                    &[
                        ("ln.weight", merged, true),
                        ("ln.bias", merged, false),
                        ("fc1.weight", merged * merged, false),
                        ("fc1.bias", merged, false),
                        ("fc2.weight", 40 * merged, false),
                        ("fc2.bias", 40, false),
                    ],
                )
            })
            .collect();

        let dweights = qwen3::init_weights(&dcfg, 3);

        // Stream: 2 text, 4 image (2×2 grid merged), 1 text. IGNORE at image rows.
        let tokens: Vec<u32> = vec![1, 2, IMG, IMG, IMG, IMG, 3];
        let mut targets = vec![2u32, 3, 0, 0, 0, 0, 5];
        for t in targets.iter_mut().take(6).skip(2) {
            *t = qwen3::IGNORE;
        }

        let model =
            Qwen3Vl::new(vcfg.clone(), dcfg, vweights, mweights, ds_mweights, &dweights, tokens.len() as u32, IMG, 2, 4, [2, 1, 1]);

        let pv_total = (16 * vcfg.patch_vec_dim()) as usize;
        let mut rng = Rng::new(4);
        let pixels: Vec<f32> = (0..pv_total).map(|_| rng.next_f32() - 0.5).collect();

        let loss = model.forward(&tokens, &targets, (4, 4), &pixels);
        assert!(loss.is_finite(), "end-to-end loss must be finite, got {loss}");
        assert!(loss > 0.0, "cross-entropy loss should be positive");
    }

    /// Same tiny synthetic shape as [`end_to_end_forward_is_finite`], but
    /// exercising [`Qwen3Vl::generate`] (Phase 7b) instead of the training-
    /// loss-shaped [`Qwen3Vl::forward`]. Not a numerical-parity test (there is
    /// no independent oracle for "qwenvl KV-cache generation with random
    /// weights") -- proves the real plumbing this session added (vision
    /// encode -> image-row splice via `step_embed_mrope` -> text prefill via
    /// `step_mrope` -> greedy decode) runs end to end, stays within vocab,
    /// is deterministic (greedy + no RNG), and that `eos_ids` actually stops
    /// generation early rather than running the full `max_new` budget.
    #[test]
    fn generate_is_deterministic_and_respects_eos() {
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
            deepstack_indexes: vec![0, 1],
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
            max_position_embeddings: 16,
            tie_embeddings: true,
            qk_norm: true,
            attn_bias: false,
            lora: None,
        };

        let (c, pv, mlp) = (vcfg.hidden as usize, vcfg.patch_vec_dim() as usize, vcfg.intermediate as usize);
        let mut vspecs: Vec<(&str, usize, bool)> = vec![
            ("patch_embed.weight", c * pv, false),
            ("patch_embed.bias", c, false),
            ("pos_embed", vcfg.num_position_embeddings as usize * c, false),
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
        let vweights = rand_map(Rng::new(11), &vspecs);

        let merged = c * 4;
        let mweights = rand_map(
            Rng::new(12),
            &[
                ("ln.weight", c, true),
                ("ln.bias", c, false),
                ("fc1.weight", merged * merged, false),
                ("fc1.bias", merged, false),
                ("fc2.weight", 40 * merged, false),
                ("fc2.bias", 40, false),
            ],
        );
        // Matches vcfg.deepstack_indexes' length (0 -- see its own comment
        // above): Qwen3Vl::new asserts the two agree.
        let ds_mweights: Vec<HashMap<String, Vec<f32>>> = (0..vcfg.deepstack_indexes.len() as u64)
            .map(|i| {
                rand_map(
                    Rng::new(30 + i),
                    &[
                        ("ln.weight", merged, true),
                        ("ln.bias", merged, false),
                        ("fc1.weight", merged * merged, false),
                        ("fc1.bias", merged, false),
                        ("fc2.weight", 40 * merged, false),
                        ("fc2.bias", 40, false),
                    ],
                )
            })
            .collect();

        let dweights = qwen3::init_weights(&dcfg, 13);

        // Prompt: 2 text, 4 image (2×2 grid merged), 1 text -- room left in
        // block_size (16) for generated tokens beyond the 7-token prompt.
        let tokens: Vec<u32> = vec![1, 2, IMG, IMG, IMG, IMG, 3];
        let seq_len = 16u32; // >= prompt len + max_new, so decode never exceeds the KV cache

        let model = Qwen3Vl::new(vcfg.clone(), dcfg, vweights, mweights, ds_mweights, &dweights, seq_len, IMG, 2, 4, [2, 1, 1]);

        let pv_total = (16 * vcfg.patch_vec_dim()) as usize;
        let mut rng = Rng::new(14);
        let pixels: Vec<f32> = (0..pv_total).map(|_| rng.next_f32() - 0.5).collect();

        let max_new = 5u32;
        let out1 = model.generate(&tokens, (4, 4), &pixels, max_new, &[]);
        assert!(!out1.is_empty(), "generate produced no tokens");
        assert!(out1.len() as u32 <= max_new, "generate exceeded max_new");
        for &t in &out1 {
            assert!((t as usize) < 23, "generated token {t} outside vocab 23");
        }

        // Greedy + no RNG: a second call from a fresh model instance (same
        // weights, same everything) must reproduce the SAME sequence.
        let model2 = Qwen3Vl::new(
            vcfg.clone(),
            QwenConfig {
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
                max_position_embeddings: 16,
                tie_embeddings: true,
                qk_norm: true,
                attn_bias: false,
                lora: None,
            },
            rand_map(Rng::new(11), &vspecs),
            rand_map(
                Rng::new(12),
                &[
                    ("ln.weight", c, true),
                    ("ln.bias", c, false),
                    ("fc1.weight", merged * merged, false),
                    ("fc1.bias", merged, false),
                    ("fc2.weight", 40 * merged, false),
                    ("fc2.bias", 40, false),
                ],
            ),
            (0..vcfg.deepstack_indexes.len() as u64)
                .map(|i| {
                    rand_map(
                        Rng::new(30 + i),
                        &[
                            ("ln.weight", merged, true),
                            ("ln.bias", merged, false),
                            ("fc1.weight", merged * merged, false),
                            ("fc1.bias", merged, false),
                            ("fc2.weight", 40 * merged, false),
                            ("fc2.bias", 40, false),
                        ],
                    )
                })
                .collect(),
            &qwen3::init_weights(
                &QwenConfig {
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
                    max_position_embeddings: 16,
                    tie_embeddings: true,
                    qk_norm: true,
                    attn_bias: false,
                    lora: None,
                },
                13,
            ),
            seq_len,
            IMG,
            2,
            4,
            [2, 1, 1],
        );
        let out2 = model2.generate(&tokens, (4, 4), &pixels, max_new, &[]);
        assert_eq!(out1, out2, "greedy generation must be deterministic across independently-constructed identical models");

        // eos_ids actually stops generation early: the first token out1[0]
        // treated as an immediate stop id must yield an empty sequence.
        let out3 = model.generate(&tokens, (4, 4), &pixels, max_new, &[out1[0]]);
        assert!(out3.is_empty(), "an eos id matching the very first generated token must stop before emitting it, got {out3:?}");
    }
}
