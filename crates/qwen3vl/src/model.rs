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
use std::time::Instant;

use gpu_core::Gpu;
use qwen3::{Dtype, Qwen, QwenConfig, Shard};

use data::rng::Rng;

use crate::config::VisionConfig;
use crate::encoder::{vision_pipelines, PatchMerger, VisionEncoder};
use crate::mrope::{get_rope_index, mrope_tables};

/// The decoding policy for [`Qwen3Vl::generate_timed`]'s per-token pick.
/// `temperature <= 0.0` is greedy argmax; otherwise `top_k`/`top_p` gate a
/// temperature-scaled softmax draw - identical contract to
/// `qwen3::sample::sample_logits` (`crate::sample::sample_logits` is this
/// crate's own copy of that algorithm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl SampleParams {
    /// Deterministic argmax decoding - this module's original (and still
    /// default) behaviour, and what every caller that does not care about
    /// sampling should pass.
    pub fn greedy() -> SampleParams {
        SampleParams { temperature: 0.0, top_k: 0, top_p: 1.0 }
    }
}

impl Default for SampleParams {
    fn default() -> Self {
        SampleParams::greedy()
    }
}

/// An assembled Qwen3-VL model (forward path). Image tokens occupy a contiguous
/// run of `image_token_id` in the text stream starting at `image_row0`.
pub struct Qwen3Vl {
    vgpu: Gpu,
    vcfg: VisionConfig,
    /// The tower, built ONCE. It used to be assembled inside every forward
    /// from host `Vec<f32>` weights, which re-uploaded the ViT and all four
    /// mergers - about 1.7 GB - per image, and held that much host RAM for the
    /// life of the model on top of the device copy. The cost is a fixed one
    /// per image, so it dominated a small image entirely.
    encoder: VisionEncoder,
    merger: PatchMerger,
    /// One postshuffle-norm merger per DeepStack tap (empty = no DeepStack).
    ds_mergers: Vec<PatchMerger>,
    decoder: Qwen,
    merge: u32,
    image_token_id: u32,
    mrope_section: [u32; 3],
}


/// How this composite's inner decoder is built.
///
/// Replaces a bare `decode_only: bool`, because there were always two
/// independent facts to state and a bool could only carry one: which GRAPH the
/// decoder gets, and which storage TIER its per-layer linears land on. Picking
/// the graph wrong is a silent correctness hazard rather than an OOM (see
/// [`Qwen3Vl::new`]), and the tier is lossy, so neither should be inferable
/// from the other or default silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderBuild {
    /// `Qwen::new`, the full BATCHED TRAINING constructor - every parameter
    /// trainable (weight+grad+adam_m+adam_v) plus the quadratic attention and
    /// `seq_len*vocab` logits buffers. For a caller that runs
    /// `forward()`/`backward()`. Always fp32: a quantized weight has no
    /// gradient.
    Batched,
    /// `Qwen::new_shard_dt_decode`, the incremental KV-cache decode graph, at
    /// the given storage tier for the 7 per-layer linears.
    ///
    /// [`Dtype::I8`] is **lossy** and must be an explicit, opt-in request that
    /// never arrives by defaulting. The tier a caller ASKS for is not
    /// necessarily the one it gets - `Weight::upload` promotes a request the
    /// device cannot serve back to fp32 - so read `Qwen::linear_dtype` (which
    /// [`Qwen3Vl::linear_dtype`] forwards) for what actually landed, never
    /// this request.
    Decode(Dtype),
}

/// Wall-clock attribution of one [`Qwen3Vl::generate_timed`] call, by the
/// stage boundaries an optimisation actually acts on.
///
/// The five numbers sum to the call. Four of the stages end in a device drain,
/// so their clocks are honest on their own. The exception is stated rather
/// than papered over: [`Self::decode_s`] and [`Self::head_s`] are the two
/// halves of one PIPELINED loop - the step submits without reading anything,
/// and the head's readback on the next iteration is what drains it - so
/// `head_s` carries the preceding step's execution and `decode_s` is closer to
/// the host-side encode cost. Forcing the split with a fence per token was
/// measured, and it cost about as much per token as the step itself; a
/// truthful pair of numbers with a note beats a precise pair that is slower to
/// produce. Read `decode_s + head_s` as the generation loop's real cost.
///
/// Kept as data rather than printed, so the caller (a bench, a served run's
/// log) decides what to do with it and nothing is emitted on the ordinary
/// path.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageTimes {
    /// ViT tower: patch embed, `depth` blocks, DeepStack tap snapshots.
    pub vision_s: f64,
    /// The main PatchMerger plus one DeepStack merger per tap.
    pub merge_s: f64,
    /// Prompt prefill (text tokens + spliced visual rows) into the KV cache.
    pub prefill_s: f64,
    /// The generation loop's decode steps - submit-only, see this struct's doc.
    pub decode_s: f64,
    /// The device LM head, its `[vocab]` readback and the argmax - and, since
    /// that readback is the loop's only fence, the preceding step's execution.
    pub head_s: f64,
    /// Visual tokens the image produced (post-merge rows).
    pub visual_tokens: u32,
    /// Prompt length in tokens, image placeholders included.
    pub prompt_tokens: u32,
    /// Tokens actually generated (may be under `max_new` on an EOS).
    pub new_tokens: u32,
}

impl StageTimes {
    /// Total wall time of the call the stages were measured over.
    pub fn total_s(&self) -> f64 {
        self.vision_s + self.merge_s + self.prefill_s + self.decode_s + self.head_s
    }
}

impl Qwen3Vl {
    /// Assemble from a vision config, a decoder config (its `d_model` must equal
    /// the merger output width), pre-uploaded host weights, and the image
    /// placement. `enable_mm_splice`/`enable_mrope` are wired on the decoder here.
    /// `build` selects the inner decoder's construction path:
    ///
    /// - [`DecoderBuild::Decode`] - the KV-cache decode graph, for
    ///   [`Qwen3Vl::generate`]'s
    ///   incremental KV-cache decode path (`decode_steps`/`generate_cb`),
    ///   which never calls `forward()`/`backward()` (see this crate's
    ///   `caps.rs` module doc). `Qwen::new` (the alternative, below) is the
    ///   FULL BATCHED TRAINING constructor: every param Trainable
    ///   (weight+grad+adam_m+adam_v = 4x the checkpoint's weight bytes
    ///   resident on the GPU) plus quadratic `[heads,seq_len,seq_len]`
    ///   attention-score and `seq_len*vocab` logits/d_logits buffers sized
    ///   for a batched forward a decode-only model never runs. At the real
    ///   4B checkpoint's scale (~16 GB fp32 weights) that is >60 GB of
    ///   optimizer state alone - genuine VRAM exhaustion on any 24 GB card.
    ///   `from_tensors_decode` allocates frozen weights only (1x, not 4x)
    ///   and `[heads,ctx]`-linear KV-cache scratch instead of the quadratic
    ///   batched shape - exactly what a greedy-decode-only model needs.
    /// - [`DecoderBuild::Batched`] - `Qwen::new`, the batched training constructor, for a
    ///   caller that runs `forward()`/`backward()` (gradcheck-style tests,
    ///   a future training path). A `decode_only` decoder never allocates
    ///   the batched `fwd_steps`/`bwd_steps` graph's buffers at all, so
    ///   calling `forward()` on one overruns whatever smaller buffer
    ///   happens to sit where the batched write lands - a real, silent
    ///   correctness hazard if this is ever picked wrong, not just an OOM.
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
        build: DecoderBuild,
    ) -> Qwen3Vl {
        assert_eq!(ds_merger_weights.len(), vcfg.deepstack_indexes.len(), "one merger per DeepStack tap");
        let merge = vcfg.spatial_merge_size;
        let n_layers = dcfg.n_layers as usize;
        let mut decoder = match build {
            DecoderBuild::Decode(dt) => Qwen::new_shard_dt_decode(dcfg, seq_len, dweights, Shard::whole(n_layers), dt),
            DecoderBuild::Batched => Qwen::new(dcfg, 1, seq_len, dweights),
        };
        decoder.enable_mm_splice(image_row0, n_visual);
        decoder.enable_mrope();
        if !ds_merger_weights.is_empty() {
            decoder.enable_deepstack(image_row0, n_visual, ds_merger_weights.len() as u32);
        }
        // The vision tower runs on the SAME physical device as the decoder it
        // feeds. `new_like` is a second KERNEL SET on the decoder's own card,
        // not a second device: the two halves are strictly sequential (the
        // merged visual rows ARE the decoder's prefill input), so there is
        // nothing to gain by separating them, and a hard-coded CPU handle here
        // put the whole 24-block tower plus all four PatchMergers on the CPU
        // JIT no matter where the caller placed the model - which is the
        // larger half of a caption's arithmetic, so it decided the run's cost.
        // On a CPU-only build `new_like` still lands on the CPU backend, so
        // placement follows the decoder rather than being asserted here.
        let vgpu = decoder.gpu().new_like(vision_pipelines());
        let d_model = decoder.cfg.d_model;
        let encoder = VisionEncoder::new(&vgpu, vcfg.clone(), &vweights);
        let merger = PatchMerger::new(&vgpu, &merger_weights, vcfg.hidden, merge, d_model, false);
        let ds_mergers =
            ds_merger_weights.iter().map(|mw| PatchMerger::new(&vgpu, mw, vcfg.hidden, merge, d_model, true)).collect();
        Qwen3Vl { vgpu, vcfg, encoder, merger, ds_mergers, decoder, merge, image_token_id, mrope_section }
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
        dt: Dtype,
    ) -> Qwen3Vl {
        let map: HashMap<String, Vec<f32>> = tensors.into_iter().map(|t| (t.name, t.data)).collect();
        let w = crate::import::partition(map, vcfg.deepstack_indexes.len());
        Qwen3Vl::from_imported(w, vcfg, dcfg, seq_len, image_token_id, image_row0, n_visual, mrope_section, dt)
    }

    /// Assemble from the four already-partitioned weight sets.
    ///
    /// The seam every source format meets at: `from_tensors` reaches it after
    /// partitioning HF safetensors names, `crate::gguf_import` reaches it after
    /// reading a two-file GGUF checkpoint, and neither format gets its own copy
    /// of the construction below. Placement is NOT decided here -- whatever
    /// device policy `Qwen3Vl::new` applies is the one both formats get, so a
    /// GGUF checkpoint can never end up somewhere a safetensors one would not.
    #[allow(clippy::too_many_arguments)]
    pub fn from_imported(
        w: crate::import::ImportedWeights,
        vcfg: VisionConfig,
        dcfg: QwenConfig,
        seq_len: u32,
        image_token_id: u32,
        image_row0: u32,
        n_visual: u32,
        mrope_section: [u32; 3],
        dt: Dtype,
    ) -> Qwen3Vl {
        // This is the real-checkpoint load path (`brain qwen3vl generate`) -
        // always the decode graph, see `Qwen3Vl::new`'s doc. `dt` is the
        // caller's explicit tier request for the decoder's per-layer linears;
        // the vision tower stays fp32 either way (it is a small fraction of
        // the weights and none of the per-token bandwidth).
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
            DecoderBuild::Decode(dt),
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
        dt: Dtype,
    ) -> Result<Qwen3Vl, String> {
        let tensors = checkpoint::safetensors::read_model_dir(std::path::Path::new(dir))?;
        Ok(Self::from_tensors(tensors, vcfg, dcfg, seq_len, image_token_id, image_row0, n_visual, mrope_section, dt))
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
        let (feats, tap_feats) = self.encoder.encode_with_taps(&self.vgpu, gh, gw, pixels, &self.vcfg.deepstack_indexes);
        let visual = self.merger.merge(&self.vgpu, &feats, n);
        assert_eq!(visual.len(), (n_visual * d_model) as usize);

        // DeepStack: each tap → its own postshuffle merger → decoder level buffer.
        for (level, (tap, ds)) in tap_feats.iter().zip(&self.ds_mergers).enumerate() {
            self.decoder.write_deepstack(level, &ds.merge(&self.vgpu, tap, n));
        }

        // M-RoPE tables from the 3-axis position ids for this stream.
        let grids_llm = [(1, gh / self.merge, gw / self.merge)];
        let positions = get_rope_index(tokens, self.image_token_id, &grids_llm);
        let (cos, sin) = mrope_tables(&positions, self.mrope_section, self.decoder.cfg.head_dim, self.decoder.cfg.rope_theta);

        // Splice + decode.
        self.decoder.write_mrope_tables(&cos, &sin);
        self.decoder.write_img_embeds(&visual);
        self.decoder.set_batch(tokens, targets);
        self.decoder.forward()
    }

    /// Greedy KV-cache generation for one image + text prompt: real
    /// `qwen3::Qwen` `step`/`step_embed` machinery (via this session's new
    /// `step_mrope`/`step_embed_mrope`, Phase 7a), not [`Self::forward`]'s
    /// training-loss-shaped batched path -- the gap noted as
    /// ("`Qwen3Vl::forward()` returns `f32`...
    /// there is no sampling loop").
    ///
    /// Prefill splices the image at its token run the same way
    /// [`Self::forward`] does (image-placeholder token ids → step_embed_mrope
    /// with the matching merged visual row; every other token → step_mrope),
    /// so the KV cache never knows the difference (mirrors `qwen3::Qwen::
    /// prefill`'s own doc). Decode continues the position sequence past the
    /// prompt as plain text (T=H=W, +1 per token — the same "media block
    /// then plain text" case `qwen3vl::mrope::get_rope_index_multi` documents).
    ///
    /// `sample` selects the decoding policy: [`SampleParams::greedy`] is
    /// argmax (deterministic, matching this function's original behaviour);
    /// a real `temperature`/`top_k`/`top_p` request samples via
    /// `crate::sample::sample_logits` (`qwen3::sample`'s algorithm, this
    /// crate's own small copy per this repo's per-model sampling-tail
    /// convention). Returns the generated token ids (prompt not included),
    /// stopping early at any id in `eos_ids`.
    ///
    /// **DeepStack IS applied here**: `qwen3::Qwen::decode_steps`'s
    /// `deepstack_row` parameter adds each level's per-row residual
    /// contribution during the incremental step that embeds that row (was
    /// missing before this session — `qwen3::Qwen::enable_deepstack`'s
    /// `SPLICE_ADD` used to be wired ONLY into the batched `forward_steps()`
    /// graph, now also threaded into incremental decode via `decode_steps`'s
    /// `deepstack_row` parameter; see also
    /// `crates/qwen3/tests/deepstack_decode_parity.rs`).
    #[allow(clippy::too_many_arguments)]
    pub fn generate(&self, tokens: &[u32], grid: (u32, u32), pixels: &[f32], max_new: u32, eos_ids: &[u32], sample: SampleParams, rng: &mut Rng) -> Vec<u32> {
        self.generate_cb(tokens, grid, pixels, max_new, eos_ids, sample, rng, |_| {})
    }

    /// [`Self::generate`] with a per-token callback — the seam the served
    /// caps path uses to emit REAL streaming deltas (its ActionSpec declares
    /// `.streaming()`, which used to be satisfied by exactly two Progress
    /// emissions around the whole decode; audit F11).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_cb(
        &self,
        tokens: &[u32],
        grid: (u32, u32),
        pixels: &[f32],
        max_new: u32,
        eos_ids: &[u32],
        sample: SampleParams,
        rng: &mut Rng,
        on_token: impl FnMut(u32),
    ) -> Vec<u32> {
        self.generate_timed(tokens, grid, pixels, max_new, eos_ids, sample, rng, on_token).0
    }

    /// The storage tier the decoder's per-layer linears ACTUALLY landed on,
    /// read off the resident weights rather than off the request. A device
    /// that cannot serve the asked-for tier is promoted back to fp32 by
    /// `Weight::upload`, and a caller reporting what it asked for would then
    /// claim a lossy run that never happened - or hide one that did.
    pub fn linear_dtype(&self) -> Option<Dtype> {
        self.decoder.linear_dtype()
    }

    /// The device this model runs on - both halves of it, since the vision
    /// tower is a second kernel set on the decoder's own card. For placement
    /// reporting and for the roofline a profile is graded against.
    pub fn gpu(&self) -> &Gpu {
        self.decoder.gpu()
    }

    /// [`Self::generate_cb`] with per-stage wall-clock attribution.
    ///
    /// The measuring seam this crate's profiler drives: the stage boundaries
    /// are inside this function, so a bench cannot get them by wrapping the
    /// public call, and two parallel implementations of the generate loop (one
    /// timed, one not) would be free to drift about which stage owns what.
    /// Timing is a few `Instant::now()` calls around work measured in seconds,
    /// so the untimed entry point above is exactly this one.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_timed(
        &self,
        tokens: &[u32],
        grid: (u32, u32),
        pixels: &[f32],
        max_new: u32,
        eos_ids: &[u32],
        sample: SampleParams,
        rng: &mut Rng,
        mut on_token: impl FnMut(u32),
    ) -> (Vec<u32>, StageTimes) {
        let mut st = StageTimes { prompt_tokens: tokens.len() as u32, ..StageTimes::default() };
        let (gh, gw) = grid;
        let n = gh * gw;
        let m2 = self.merge * self.merge;
        let n_visual = n / m2;
        let d_model = self.decoder.cfg.d_model as usize;

        // Vision tower -> visual tokens (+ DeepStack taps), same as forward().
        let t_vision = Instant::now();
        let (feats, tap_feats) = self.encoder.encode_with_taps(&self.vgpu, gh, gw, pixels, &self.vcfg.deepstack_indexes);
        st.vision_s = t_vision.elapsed().as_secs_f64();

        let t_merge = Instant::now();
        let visual = self.merger.merge(&self.vgpu, &feats, n);
        assert_eq!(visual.len(), (n_visual as usize) * d_model);
        for (level, (tap, ds)) in tap_feats.iter().zip(&self.ds_mergers).enumerate() {
            self.decoder.write_deepstack(level, &ds.merge(&self.vgpu, tap, n));
        }
        st.merge_s = t_merge.elapsed().as_secs_f64();
        st.visual_tokens = n_visual;

        // M-RoPE positions for the KNOWN prompt (whole-sequence, once).
        let grids_llm = [(1, gh / self.merge, gw / self.merge)];
        let prompt_positions = get_rope_index(tokens, self.image_token_id, &grids_llm);
        assert_eq!(prompt_positions.len(), tokens.len());

        // Prefill: image rows via step_embed_mrope, text rows via
        // step_mrope, each with its own 1-row M-RoPE table (mrope_tables
        // called per-position -- the plan's own recommended shape, "a
        // single-element positions slice").
        let t_prefill = Instant::now();
        self.decoder.reset_cache();
        let mut visual_row = 0usize;
        for (i, &tok) in tokens.iter().enumerate() {
            let (cos, sin) = mrope_tables(&prompt_positions[i..=i], self.mrope_section, self.decoder.cfg.head_dim, self.decoder.cfg.rope_theta);
            // Nothing reads a prompt position's hidden state -- only the LAST
            // one matters, and the device head below takes that from
            // `xn_final` where the step left it. `prefill_mrope` therefore
            // submits without the `[d_model]` readback its `step_*` siblings
            // promise, which at a VL prompt (mostly image rows) is one
            // submit+fence+map round trip per token removed.
            let input = if tok == self.image_token_id {
                let row = &visual[visual_row * d_model..(visual_row + 1) * d_model];
                let ds_row = Some(visual_row as u32);
                visual_row += 1;
                self.decoder.prefill_mrope(qwen3::model::PrefillInput::Embed(row), &cos, &sin, ds_row);
                continue;
            } else {
                qwen3::model::PrefillInput::Token(tok)
            };
            self.decoder.prefill_mrope(input, &cos, &sin, None);
        }
        assert_eq!(visual_row, n_visual as usize, "image token count in the prompt must match n_visual");
        // Prefill no longer reads anything back, so without this the host
        // clock would stop while the tape was still queued and the first head
        // dispatch below would inherit the whole backlog - a stage split that
        // charges the wrong stage is worse than no split at all. The fence is
        // free in wall-clock terms: the head's own readback would block on the
        // same work one statement later.
        self.decoder.gpu().poll_wait();
        st.prefill_s = t_prefill.elapsed().as_secs_f64();

        // Decode: sample (or, at SampleParams::greedy(), argmax) continuing
        // the position sequence past the prompt as plain text. The head is applied ON the device the weights
        // already sit on (`Qwen::decode_logits`, vocab-tiled so each binding
        // stays inside the limit) and only a `[vocab]` row crosses back. The
        // host path this replaces read the WHOLE tied head table once per
        // caption -- 1.5 GB at the 4B config -- and then swept it, scalar and
        // single-threaded, for every generated token.
        let mut next_pos = prompt_positions.last().map(|p| p[0] + 1).unwrap_or(0);
        let mut out = Vec::with_capacity(max_new as usize);
        for _ in 0..max_new {
            let t_head = Instant::now();
            let next = crate::sample::sample_logits(&self.decoder.decode_logits(), sample.temperature, sample.top_k, sample.top_p, rng);
            st.head_s += t_head.elapsed().as_secs_f64();
            if eos_ids.contains(&next) {
                break;
            }
            out.push(next);
            on_token(next);
            let t_step = Instant::now();
            let (cos, sin) = mrope_tables(&[[next_pos; 3]], self.mrope_section, self.decoder.cfg.head_dim, self.decoder.cfg.rope_theta);
            self.decoder.prefill_mrope(qwen3::model::PrefillInput::Token(next), &cos, &sin, None);
            st.decode_s += t_step.elapsed().as_secs_f64();
            next_pos += 1;
        }
        st.new_tokens = out.len() as u32;
        (out, st)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            Qwen3Vl::new(vcfg.clone(), dcfg, vweights, mweights, ds_mweights, &dweights, tokens.len() as u32, IMG, 2, 4, [2, 1, 1], DecoderBuild::Batched);

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

        let model = Qwen3Vl::new(vcfg.clone(), dcfg, vweights, mweights, ds_mweights, &dweights, seq_len, IMG, 2, 4, [2, 1, 1], DecoderBuild::Decode(Dtype::F32));

        let pv_total = (16 * vcfg.patch_vec_dim()) as usize;
        let mut rng = Rng::new(14);
        let pixels: Vec<f32> = (0..pv_total).map(|_| rng.next_f32() - 0.5).collect();

        let max_new = 5u32;
        let mut gen_rng = Rng::new(99);
        let out1 = model.generate(&tokens, (4, 4), &pixels, max_new, &[], SampleParams::greedy(), &mut gen_rng);
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
            DecoderBuild::Decode(Dtype::F32),
        );
        let mut gen_rng2 = Rng::new(1);
        let out2 = model2.generate(&tokens, (4, 4), &pixels, max_new, &[], SampleParams::greedy(), &mut gen_rng2);
        assert_eq!(out1, out2, "greedy generation must be deterministic across independently-constructed identical models, regardless of the RNG state");

        // eos_ids actually stops generation early: the first token out1[0]
        // treated as an immediate stop id must yield an empty sequence.
        let mut gen_rng3 = Rng::new(2);
        let out3 = model.generate(&tokens, (4, 4), &pixels, max_new, &[out1[0]], SampleParams::greedy(), &mut gen_rng3);
        assert!(out3.is_empty(), "an eos id matching the very first generated token must stop before emitting it, got {out3:?}");

        // A real sampling request (temperature > 0) must actually consult the
        // RNG: two different seeds decoding the same prompt at temperature=1
        // must not always agree, or "sampling" would be greedy in disguise.
        let sample = SampleParams { temperature: 1.0, top_k: 0, top_p: 1.0 };
        let mut disagreed = false;
        for seed in 0..16u64 {
            let mut rng_a = Rng::new(seed);
            let mut rng_b = Rng::new(seed + 1000);
            let a = model.generate(&tokens, (4, 4), &pixels, max_new, &[], sample, &mut rng_a);
            let b = model.generate(&tokens, (4, 4), &pixels, max_new, &[], sample, &mut rng_b);
            if a != b {
                disagreed = true;
                break;
            }
        }
        assert!(disagreed, "temperature=1.0 sampling must vary across RNG seeds on at least one of 16 trials");
    }
}
