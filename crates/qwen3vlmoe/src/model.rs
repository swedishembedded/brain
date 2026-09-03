// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL-30B-A3B composite: `qwen3vl`'s ViT encoder + PatchMerger +
//! DeepStack, reused AS-IS, spliced onto `qwen3omnimoe::thinker`'s MoE text
//! decoder (GQA + QK-norm + M-RoPE + top-k sparse MoE FFN, no shared expert)
//! via a host-side embedding-gather + row splice, the same seam
//! `qwen3omnimoe::generate`'s own real prefill uses (`EmbedTable::row` +
//! a per-token gather - see that module's doc).
//!
//! **Why `qwen3omnimoe::thinker`, not `qwen35moe`'s decoder**: `qwen35moe::vl`
//! is this repo's closest PATTERN precedent for "splice `qwen3vl`'s vision
//! tower onto a different decoder" (its own module doc says so), but its
//! decoder is a hybrid Gated-DeltaNet/GQA stack - the real architecture of a
//! DIFFERENT checkpoint family (Qwen3.5-35B-A3B), not Qwen3-VL-30B-A3B. The
//! REAL `Qwen/Qwen3-VL-30B-A3B-Instruct` `text_config` (fetched and quoted in
//! `crate::config`'s doc) is a plain GQA + QK-norm + RoPE decoder with a
//! top-k-of-128 sparse MoE FFN and no shared expert on every layer -
//! byte-for-byte the same SHAPE `qwen3omnimoe::config::MoeTextConfig::
//! thinker_defaults` already models for Qwen3-Omni's Thinker (confirmed, not
//! assumed - see `crate::config`'s doc, point 2). So this module reuses
//! `qwen3omnimoe::thinker`'s existing, already-gradient-adjacent forward
//! functions (`layer_fwd`/`final_norm`/`lm_head_fwd`) directly, per this
//! workspace's "one implementation" rule, rather than writing a second copy
//! of the exact same GQA+MoE math under a new name.
//!
//! **DeepStack**: `qwen3omnimoe::thinker` has no DeepStack splice of its own
//! (Qwen3-Omni's real served path does not use one - see `qwen3omnimoe::mm`'s
//! own doc, "not needed for a plain (non-DeepStack) splice path"). This
//! module adds it back the way `qwen3::Qwen`'s own DeepStack add does: one
//! extra `splice_add` kernel dispatch (`kernels::SPLICE_ADD`, the exact WGSL
//! kernel `qwen3::model`'s own DeepStack residual add already uses - reused,
//! not reimplemented) after each of the first `deepstack_indexes.len()`
//! decoder layers, accumulating that tap's merged vision features into the
//! image-token rows of the layer's own output - level `i` after decoder
//! layer `i`, the same convention `qwen3::Qwen::enable_deepstack`'s own doc
//! states. [`decoder_pipelines`] is `qwen3omnimoe::thinker::thinker_pipelines`'s
//! table with exactly that one kernel appended, so every hard-coded kernel
//! index `qwen3omnimoe::thinker`'s functions dispatch by stays valid.
//!
//! **Scope - this is SHAPE, not real-weight parity.** No real
//! `Qwen3-VL-30B-A3B` checkpoint (safetensors or GGUF) was available to
//! import against in this environment (see `crate::import`'s doc for exactly
//! what that blocks). [`Qwen3VlMoe::forward`] is exercised ONLY against
//! synthetic tiny configs with random weights in this module's own tests -
//! proving the plumbing (vision tower -> merger -> DeepStack -> MoE decoder
//! -> M-RoPE position math) is wired correctly end to end, finite output,
//! nothing more. This is exactly the bar `qwen3vl::model`'s own
//! `end_to_end_forward_is_finite` holds itself to on synthetic configs, and
//! must never be read as "works on the real checkpoint" - that claim is not
//! made anywhere in this crate.

use std::collections::HashMap;
use std::sync::OnceLock;

use gpu_core::{DeviceBuffer, Gpu};

use qwen3omnimoe::config::MoeTextConfig;
use qwen3omnimoe::thinker::{final_norm, layer_fwd, lm_head_fwd, thinker_pipelines, ThinkerLayerWeights};
use qwen3vl::encoder::{vision_pipelines, PatchMerger, VisionEncoder};
use qwen3vl::mrope::{get_rope_index, mrope_tables};

use crate::config::Qwen3VlMoeConfig;

/// [`qwen3omnimoe::thinker::thinker_pipelines`]'s kernel table plus ONE
/// appended entry (`splice_add`, this module's DeepStack add - see the module
/// doc). Appending rather than inserting preserves every hard-coded index
/// `qwen3omnimoe::thinker`'s functions dispatch by (`kernel_ids`/`moe_ids`/
/// `attn_ids` are all indices into THIS list's leading, unchanged prefix).
/// Built once and leaked (`gpu_core::testgpu::dev` and `Gpu::new_like` both
/// want a `'static` slice, the same reason every OTHER pipeline table in this
/// engine is a `const`/`static` - this one just cannot be, since it is
/// computed from another crate's table at run time).
pub fn decoder_pipelines() -> &'static [(&'static str, &'static str)] {
    static TABLE: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut v = thinker_pipelines().to_vec();
        v.push(("splice_add", kernels::SPLICE_ADD));
        v
    })
}

/// [`decoder_pipelines`]'s `splice_add` slot, resolved by name so it cannot
/// silently drift out of step with the table it is appended to.
fn splice_add_idx() -> usize {
    decoder_pipelines().iter().position(|(n, _)| *n == "splice_add").expect("decoder_pipelines always appends splice_add")
}

/// One decoder layer's weights, already uploaded - the owned counterpart of
/// [`qwen3omnimoe::thinker::ThinkerLayerWeights`]'s borrowed view, keyed the
/// same way `qwen3omnimoe`'s own `thinker_decode.rs` test builds a tiny
/// layer (this crate has no real-checkpoint loader yet, see `crate::import`).
pub struct DecoderLayer {
    pub ln1: DeviceBuffer,
    pub wq: DeviceBuffer,
    pub wk: DeviceBuffer,
    pub wv: DeviceBuffer,
    pub wo: DeviceBuffer,
    pub q_norm: DeviceBuffer,
    pub k_norm: DeviceBuffer,
    pub ln2: DeviceBuffer,
    pub router: DeviceBuffer,
    /// Expert `e`'s `(gate.weight, up.weight, down.weight)`, indexed
    /// `0..n_experts`.
    pub experts: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>,
}

impl DecoderLayer {
    fn as_weights(&self) -> ThinkerLayerWeights<'_> {
        ThinkerLayerWeights {
            ln1: &self.ln1,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            wo: &self.wo,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            ln2: &self.ln2,
            router: &self.router,
            experts: &self.experts,
        }
    }
}

/// An assembled Qwen3-VL-30B-A3B model (forward path only; see this module's
/// doc for scope). Image tokens occupy a contiguous run of `image_token_id`
/// in the text stream (found by scanning, unlike `qwen3vl::model::Qwen3Vl`'s
/// baked-in `image_row0` - this composite is stateless per call rather than
/// KV-cache-resident, so there is nothing to bake it into ahead of time).
pub struct Qwen3VlMoe {
    vgpu: Gpu,
    gpu: Gpu,
    cfg: Qwen3VlMoeConfig,
    encoder: VisionEncoder,
    merger: PatchMerger,
    /// One postshuffle-norm merger per DeepStack tap.
    ds_mergers: Vec<PatchMerger>,
    layers: Vec<DecoderLayer>,
    final_norm_w: DeviceBuffer,
    lm_head_w: DeviceBuffer,
    /// `[vocab, hidden]`, host-resident - this composite has no KV-cache
    /// decode path yet (see this module's doc), so a per-token gather at
    /// prefill is the whole story, matching `qwen3omnimoe::generate`'s own
    /// `EmbedTable::row` convention.
    embed_table: Vec<f32>,
    merge: u32,
    image_token_id: u32,
    splice_add: usize,
}

impl Qwen3VlMoe {
    /// Assemble from a config, host vision/merger weight maps (uploaded
    /// internally by `VisionEncoder`/`PatchMerger`, the same as
    /// `qwen3vl::model::Qwen3Vl::new`), and already-uploaded decoder layer
    /// buffers + top-level norm/head/embedding - see this module's doc for
    /// why the decoder side takes buffers rather than a host map (no
    /// real-checkpoint loader exists yet for this architecture).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: Gpu,
        cfg: Qwen3VlMoeConfig,
        vweights: HashMap<String, Vec<f32>>,
        merger_weights: HashMap<String, Vec<f32>>,
        ds_merger_weights: Vec<HashMap<String, Vec<f32>>>,
        layers: Vec<DecoderLayer>,
        final_norm_w: DeviceBuffer,
        lm_head_w: DeviceBuffer,
        embed_table: Vec<f32>,
    ) -> Qwen3VlMoe {
        assert_eq!(ds_merger_weights.len(), cfg.vision.deepstack_indexes.len(), "one merger per DeepStack tap");
        assert_eq!(layers.len(), cfg.text.n_layers as usize, "one weight set per decoder layer");
        assert_eq!(cfg.vision.out_hidden_size, cfg.text.hidden, "vision tower output width must match decoder hidden size");
        assert_eq!(embed_table.len(), (cfg.text.vocab * cfg.text.hidden) as usize, "embed_table must be [vocab, hidden]");

        let merge = cfg.vision.spatial_merge_size;
        let d_model = cfg.text.hidden;
        // The vision tower runs on a second kernel set on the SAME physical
        // device as the decoder - `new_like`, not a second device, matching
        // `qwen3vl::model::Qwen3Vl::new`'s own placement choice and its
        // documented reason (the two halves are strictly sequential).
        let vgpu = gpu.new_like(vision_pipelines());
        let encoder = VisionEncoder::new(&vgpu, cfg.vision.clone(), &vweights);
        let merger = PatchMerger::new(&vgpu, &merger_weights, cfg.vision.hidden, merge, d_model, false);
        let ds_mergers =
            ds_merger_weights.iter().map(|mw| PatchMerger::new(&vgpu, mw, cfg.vision.hidden, merge, d_model, true)).collect();
        let image_token_id = cfg.image_token_id;
        Qwen3VlMoe { vgpu, gpu, cfg, encoder, merger, ds_mergers, layers, final_norm_w, lm_head_w, embed_table, merge, image_token_id, splice_add: splice_add_idx() }
    }

    /// `qwen3omnimoe::config::MoeTextConfig`'s copy for the config
    /// `qwen3omnimoe::thinker`'s functions actually read, exposed for a
    /// caller building its own layer weights against this model's shape.
    pub fn text_config(&self) -> &MoeTextConfig {
        &self.cfg.text
    }

    /// End-to-end forward for one image + text stream: vision tower -> main
    /// merger -> per-layer DeepStack splice -> `n_layers` MoE decoder layers
    /// -> final norm -> LM head. Returns the full `[n, vocab]` logits, row-
    /// major (`n = tokens.len()`). Panics if `tokens` carries no
    /// `image_token_id` run, or if that run's length disagrees with the
    /// image's merged visual-token count - see this module's doc for scope
    /// (synthetic-config wiring proof, not real-weight parity).
    pub fn forward(&self, tokens: &[u32], grid: (u32, u32), pixels: &[f32]) -> Vec<f32> {
        let (gh, gw) = grid;
        let n_patches = gh * gw;
        let m2 = self.merge * self.merge;
        let n_visual = n_patches / m2;
        let d = self.cfg.text.hidden;
        let n = tokens.len() as u32;

        // Vision tower -> visual tokens (+ DeepStack taps), exactly
        // `qwen3vl::model::Qwen3Vl::forward`'s own sequence.
        let (feats, tap_feats) = self.encoder.encode_with_taps(&self.vgpu, gh, gw, pixels, &self.cfg.vision.deepstack_indexes);
        let visual = self.merger.merge(&self.vgpu, &feats, n_patches);
        assert_eq!(visual.len(), (n_visual * d) as usize);
        let deepstack: Vec<Vec<f32>> =
            tap_feats.iter().zip(&self.ds_mergers).map(|(tap, ds)| ds.merge(&self.vgpu, tap, n_patches)).collect();

        // Host-side embedding gather + visual splice at the image-token rows
        // - the same convention `qwen3omnimoe::generate`'s real prefill uses
        // (one host buffer assembled row by row, then ONE upload) rather
        // than `qwen3vl::model::Qwen3Vl`'s on-device `write_img_embeds`,
        // since `qwen3omnimoe::thinker::decode` takes an already-assembled
        // sequence and has no splice seam of its own (see that module's doc).
        let image_row0 = tokens.iter().position(|&t| t == self.image_token_id).expect("prompt must contain the image_token_id run") as u32;
        let mut x_host = vec![0f32; (n * d) as usize];
        let mut visual_row = 0usize;
        for (i, &tok) in tokens.iter().enumerate() {
            let dst = &mut x_host[i * d as usize..(i + 1) * d as usize];
            if tok == self.image_token_id {
                dst.copy_from_slice(&visual[visual_row * d as usize..(visual_row + 1) * d as usize]);
                visual_row += 1;
            } else {
                dst.copy_from_slice(&self.embed_table[tok as usize * d as usize..(tok as usize + 1) * d as usize]);
            }
        }
        assert_eq!(visual_row, n_visual as usize, "image token run length must match the merged visual token count");

        // M-RoPE tables from the real 3-axis position ids for this stream.
        let grids_llm = [(1, gh / self.merge, gw / self.merge)];
        let positions = get_rope_index(tokens, self.image_token_id, &grids_llm);
        let section: [u32; 3] = [self.cfg.text.mrope_section[0], self.cfg.text.mrope_section[1], self.cfg.text.mrope_section[2]];
        let (cos_tab, sin_tab) = mrope_tables(&positions, section, self.cfg.text.head_dim, self.cfg.text.rope_theta);

        let mut h = self.gpu.storage_init("x", &x_host);
        let cos = self.gpu.storage_init("cos", &cos_tab);
        let sin = self.gpu.storage_init("sin", &sin_tab);

        // Decoder: N `layer_fwd` calls chained residual-to-residual (the same
        // composition `qwen3omnimoe::thinker::decode` itself performs), with
        // DeepStack tap `l` added into layer `l`'s own output at the image
        // rows, for `l` in `0..deepstack.len()` - see the module doc.
        for (l, layer) in self.layers.iter().enumerate() {
            let (out, ..) = layer_fwd(&self.gpu, &self.cfg.text, &layer.as_weights(), &h, &cos, &sin, n, None, None);
            if let Some(tap) = deepstack.get(l) {
                let tap_buf = self.gpu.storage_init("deepstack_tap", tap);
                self.gpu.submit(&[], &[self.gpu.step(self.splice_add, &[&tap_buf, &out], &[n_visual * d, image_row0 * d], n_visual * d)]);
            }
            h = out;
        }

        let hidden = final_norm(&self.gpu, &self.cfg.text, &self.final_norm_w, &h, n);
        let logits = lm_head_fwd(&self.gpu, &self.lm_head_w, &hidden, n, d, self.cfg.text.vocab);
        self.gpu.read(&logits, (n * self.cfg.text.vocab) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;
    use qwen3vl::config::VisionConfig;

    const IMG: u32 = 9;

    fn tiny_vision_config() -> VisionConfig {
        VisionConfig {
            depth: 2,
            hidden: 16,
            num_heads: 2,
            intermediate: 32,
            patch_size: 2,
            temporal_patch_size: 1,
            spatial_merge_size: 2,
            num_position_embeddings: 16,
            out_hidden_size: 12, // == tiny_text_config().hidden
            in_channels: 2,
            deepstack_indexes: vec![0, 1], // tap both ViT blocks -> decoder layers 0,1
            tokens_per_second: 2,
        }
    }

    fn tiny_text_config() -> MoeTextConfig {
        MoeTextConfig {
            n_layers: 2,
            hidden: 12,
            n_heads: 3,
            n_kv_heads: 1,
            head_dim: 8,
            moe_intermediate: 6,
            shared_expert_intermediate: 0,
            n_experts: 4,
            top_k: 2,
            norm_topk_prob: true,
            use_qk_norm: true,
            vocab: 23,
            rope_theta: 1.0e6,
            rms_norm_eps: 1e-6,
            mrope_section: vec![2, 1, 1], // sums to head_dim/2 = 4
            max_position_embeddings: 32,
        }
    }

    fn rand_map(rng: &mut Lcg, specs: &[(&str, usize, bool)]) -> HashMap<String, Vec<f32>> {
        let mut m = HashMap::new();
        for &(name, n, ones) in specs {
            let v = if ones { vec![1.0; n] } else { rng.vec_scaled(n, 0.3) };
            m.insert(name.to_string(), v);
        }
        m
    }

    fn tiny_vision_weights(rng: &mut Lcg, vcfg: &VisionConfig) -> HashMap<String, Vec<f32>> {
        let (c, pv, mlp) = (vcfg.hidden as usize, vcfg.patch_vec_dim() as usize, vcfg.intermediate as usize);
        let mut specs: Vec<(String, usize, bool)> = vec![
            ("patch_embed.weight".into(), c * pv, false),
            ("patch_embed.bias".into(), c, false),
            ("pos_embed".into(), vcfg.num_position_embeddings as usize * c, false),
        ];
        for b in 0..vcfg.depth {
            specs.extend([
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
            ]);
        }
        let refs: Vec<(&str, usize, bool)> = specs.iter().map(|(n, s, o)| (n.as_str(), *s, *o)).collect();
        rand_map(rng, &refs)
    }

    fn tiny_merger_weights(rng: &mut Lcg, vcfg: &VisionConfig, d_model: u32, postshuffle: bool) -> HashMap<String, Vec<f32>> {
        let merged = (vcfg.hidden * vcfg.merge_unit()) as usize;
        let ln_dim = if postshuffle { merged } else { vcfg.hidden as usize };
        rand_map(
            rng,
            &[
                ("ln.weight", ln_dim, true),
                ("ln.bias", ln_dim, false),
                ("fc1.weight", merged * merged, false),
                ("fc1.bias", merged, false),
                ("fc2.weight", d_model as usize * merged, false),
                ("fc2.bias", d_model as usize, false),
            ],
        )
    }

    fn tiny_layer(gpu: &Gpu, rng: &mut Lcg, cfg: &MoeTextConfig) -> DecoderLayer {
        let (d, hd, nh, nkv, ff) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.moe_intermediate);
        let (hq, hkv) = (nh * hd, nkv * hd);
        let init = |rng: &mut Lcg, n: usize| gpu.storage_init("w", &rng.vec_scaled(n, 0.3));
        DecoderLayer {
            ln1: init(rng, d as usize),
            wq: init(rng, (hq * d) as usize),
            wk: init(rng, (hkv * d) as usize),
            wv: init(rng, (hkv * d) as usize),
            wo: init(rng, (d * hq) as usize),
            q_norm: init(rng, hd as usize),
            k_norm: init(rng, hd as usize),
            ln2: init(rng, d as usize),
            router: init(rng, (cfg.n_experts * d) as usize),
            experts: (0..cfg.n_experts).map(|_| (init(rng, (ff * d) as usize), init(rng, (ff * d) as usize), init(rng, (d * ff) as usize))).collect(),
        }
    }

    #[test]
    fn end_to_end_forward_is_finite() {
        let vcfg = tiny_vision_config();
        let tcfg = tiny_text_config();
        let cfg = Qwen3VlMoeConfig { vision: vcfg.clone(), text: tcfg.clone(), image_token_id: IMG, video_token_id: IMG + 1, vision_start_token_id: 100, vision_end_token_id: 101 };

        let mut rng = Lcg::new(7);
        let vweights = tiny_vision_weights(&mut rng, &vcfg);
        let mweights = tiny_merger_weights(&mut rng, &vcfg, tcfg.hidden, false);
        let ds_mweights: Vec<HashMap<String, Vec<f32>>> = (0..vcfg.deepstack_indexes.len()).map(|_| tiny_merger_weights(&mut rng, &vcfg, tcfg.hidden, true)).collect();

        let gpu = gpu_core::testgpu::dev(decoder_pipelines());
        let layers: Vec<DecoderLayer> = (0..tcfg.n_layers).map(|_| tiny_layer(&gpu, &mut rng, &tcfg)).collect();
        let final_norm_w = gpu.storage_init("final_norm", &rng.vec_scaled(tcfg.hidden as usize, 0.3));
        let lm_head_w = gpu.storage_init("lm_head", &rng.vec_scaled((tcfg.vocab * tcfg.hidden) as usize, 0.3));
        let embed_table = rng.vec_scaled((tcfg.vocab * tcfg.hidden) as usize, 0.3);

        let model = Qwen3VlMoe::new(gpu, cfg, vweights, mweights, ds_mweights, layers, final_norm_w, lm_head_w, embed_table);

        // Stream: 2 text, 4 image (2x2 grid merged), 1 text.
        let tokens: Vec<u32> = vec![1, 2, IMG, IMG, IMG, IMG, 3];
        let pv_total = (16 * vcfg.patch_vec_dim()) as usize; // 4x4 patch grid
        let pixels = rng.vec_scaled(pv_total, 0.5);

        let logits = model.forward(&tokens, (4, 4), &pixels);
        assert_eq!(logits.len(), tokens.len() * tcfg.vocab as usize);
        assert!(logits.iter().all(|v| v.is_finite()), "forward produced a non-finite logit");
    }

    #[test]
    #[should_panic(expected = "image_token_id run")]
    fn forward_panics_without_an_image_token_run() {
        let vcfg = tiny_vision_config();
        let tcfg = tiny_text_config();
        let cfg = Qwen3VlMoeConfig { vision: vcfg.clone(), text: tcfg.clone(), image_token_id: IMG, video_token_id: IMG + 1, vision_start_token_id: 100, vision_end_token_id: 101 };

        let mut rng = Lcg::new(11);
        let vweights = tiny_vision_weights(&mut rng, &vcfg);
        let mweights = tiny_merger_weights(&mut rng, &vcfg, tcfg.hidden, false);
        let ds_mweights: Vec<HashMap<String, Vec<f32>>> = (0..vcfg.deepstack_indexes.len()).map(|_| tiny_merger_weights(&mut rng, &vcfg, tcfg.hidden, true)).collect();
        let gpu = gpu_core::testgpu::dev(decoder_pipelines());
        let layers: Vec<DecoderLayer> = (0..tcfg.n_layers).map(|_| tiny_layer(&gpu, &mut rng, &tcfg)).collect();
        let final_norm_w = gpu.storage_init("final_norm", &rng.vec_scaled(tcfg.hidden as usize, 0.3));
        let lm_head_w = gpu.storage_init("lm_head", &rng.vec_scaled((tcfg.vocab * tcfg.hidden) as usize, 0.3));
        let embed_table = rng.vec_scaled((tcfg.vocab * tcfg.hidden) as usize, 0.3);
        let model = Qwen3VlMoe::new(gpu, cfg, vweights, mweights, ds_mweights, layers, final_norm_w, lm_head_w, embed_table);

        let tokens: Vec<u32> = vec![1, 2, 3]; // no IMG token anywhere
        let pv_total = (16 * vcfg.patch_vec_dim()) as usize;
        let pixels = rng.vec_scaled(pv_total, 0.5);
        model.forward(&tokens, (4, 4), &pixels);
    }
}
