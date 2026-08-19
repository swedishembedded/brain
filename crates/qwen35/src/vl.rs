// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.8-27B composite: `qwen3vl`'s ViT encoder + PatchMerger, reused
//! AS-IS, spliced into this crate's dense hybrid decoder via the M-RoPE +
//! embedding-splice seam `crate::model::Qwen35` exposes (M9).
//!
//! Mirrors `qwen35moe::vl::Qwen35Vl` almost exactly (same field shape, same
//! `forward` contract) - the difference is what it wraps, not how it's
//! wired: this model's vision config is numerically IDENTICAL to
//! `qwen3vl::VisionConfig::qwen3_omni()` except `out_hidden_size` (this
//! decoder's `d_model`, 5120 at real scale) and `deepstack_indexes` (EMPTY, a
//! deliberate vision-tower reuse decision - a config-level reuse, not a fork
//! of `crates/qwenvl`), so there is no DeepStack here at all (no tap-feature
//! encode, no per-level merger, no `write_deepstack` call).
//!
//! Scope: single image, prefill/training-loss-shaped forward only (matching
//! `qwen3vl::model::Qwen3Vl::forward`'s own scope) - no video (multi-frame),
//! and no incremental (KV-cache) decode-time image splice yet (this crate has
//! no KV-cache decode path at all today - M9's own scope is the batched
//! prefill splice, matching `crate::model::Qwen35::forward`'s existing shape).
//!
//! Vision runs on its own CPU-backed `Gpu` (matching `qwen35moe::vl::Qwen35Vl`'s
//! own choice); visual tokens cross to the decoder's `Gpu` host-side via
//! `Qwen35::write_img_embeds` (a fused single-device path, and a vision-tower
//! backward for full-tower finetune, are later steps - matching
//! `qwen35moe::vl::Qwen35Vl`'s own documented scope for the same gaps).

use std::collections::HashMap;

use gpu_core::Gpu;

use qwen3vl::config::VisionConfig;
use qwen3vl::encoder::{vision_pipelines, PatchMerger, VisionEncoder};
use qwen3vl::mrope::{get_rope_index, mrope_tables};

use crate::config::Qwen35Config;
use crate::model::{pipelines, Qwen35};

/// An assembled Qwen3.8-27B vision-language model (forward path only). Image
/// tokens occupy a contiguous run of `image_token_id` in the text stream
/// starting at `image_row0`.
pub struct Qwen35Vl {
    vgpu: Gpu,
    vcfg: VisionConfig,
    vweights: HashMap<String, Vec<f32>>,
    merger_weights: HashMap<String, Vec<f32>>,
    decoder: Qwen35,
    merge: u32,
    image_token_id: u32,
    mrope_section: [u32; 3],
}

impl Qwen35Vl {
    /// Assemble from a vision config, a decoder config (its `d_model` must
    /// equal the merger output width), pre-uploaded host weights, and the
    /// image placement. Decoder batch is fixed at 1, matching
    /// `qwen3vl::model::Qwen3Vl::new`'s own choice.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vcfg: VisionConfig,
        dcfg: Qwen35Config,
        vweights: HashMap<String, Vec<f32>>,
        merger_weights: HashMap<String, Vec<f32>>,
        dweights: &HashMap<String, Vec<f32>>,
        seq_len: u32,
        image_token_id: u32,
        image_row0: u32,
        n_visual: u32,
    ) -> Qwen35Vl {
        assert!(vcfg.deepstack_indexes.is_empty(), "this model has no DeepStack -- see this module's own doc");
        let merge = vcfg.spatial_merge_size;
        let mrope_section = dcfg.mrope_section;
        let mut decoder = Qwen35::new_on(Gpu::new(pipelines()), dcfg, 1, seq_len, dweights);
        decoder.enable_mm_splice(image_row0, n_visual);
        Qwen35Vl { vgpu: Gpu::new_cpu(vision_pipelines()), vcfg, vweights, merger_weights, decoder, merge, image_token_id, mrope_section }
    }

    /// End-to-end forward for one image + text stream; returns the decoder's
    /// scalar loss. `pixels` is the host-packed `[grid_h*grid_w, patch_vec]`
    /// patch tensor; `tokens`/`targets` are the full text stream (image
    /// placeholders carry IGNORE targets). Panics if the visual-token count
    /// disagrees with the placement `enable_mm_splice` was built with.
    pub fn forward(&self, tokens: &[u32], targets: &[u32], grid: (u32, u32), pixels: &[f32]) -> f32 {
        let (gh, gw) = grid;
        let n = gh * gw;
        let m2 = self.merge * self.merge;
        let n_visual = n / m2;
        let d_model = self.decoder.cfg.d_model;

        // Vision tower -> merger -> visual tokens at the decoder width. No
        // DeepStack taps (empty `deepstack_indexes`, asserted in `new`).
        let enc = VisionEncoder::new(&self.vgpu, self.vcfg.clone(), &self.vweights);
        let feats = enc.encode(gh, gw, pixels);
        let merger = PatchMerger::new(&self.vgpu, &self.merger_weights, self.vcfg.hidden, self.merge, d_model, false);
        let visual = merger.merge(&feats, n);
        assert_eq!(visual.len(), (n_visual * d_model) as usize);

        // M-RoPE tables from the REAL 3-axis position ids for this stream
        // (the image-token run's grid, in merged/LLM units).
        let grids_llm = [(1, gh / self.merge, gw / self.merge)];
        let positions = get_rope_index(tokens, self.image_token_id, &grids_llm);
        let (cos, sin) = mrope_tables(&positions, self.mrope_section, self.decoder.cfg.rotary_dim(), self.decoder.cfg.rope_theta);

        // Splice + decode: `Qwen35::forward` reads back whatever `set_batch`/
        // `write_img_embeds`/`write_mrope_tables` most recently wrote (no
        // args of its own - matching its plain-text-decoder contract).
        self.decoder.write_mrope_tables(&cos, &sin);
        self.decoder.write_img_embeds(&visual);
        self.decoder.set_batch(tokens, targets);
        self.decoder.forward()
    }
}
