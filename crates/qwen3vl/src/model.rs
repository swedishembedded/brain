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
use crate::mrope::{get_rope_index, get_rope_index_multi, mrope_tables};

/// One image's host-packed input to [`Qwen3Vl::generate`]/[`Qwen3Vl::generate_cb`]/
/// [`Qwen3Vl::generate_timed`]: its pre-merge patch grid `(grid_h, grid_w)` and its
/// `[grid_h·grid_w, patch_vec]` packed pixel tensor, in the same units and layout
/// [`VisionEncoder::encode_with_taps`] takes for one image. A request supplies one
/// of these per image, in the order its vision-start/`[IMG]*`/vision-end run
/// appears in `tokens` (see `crate::caps`'s module doc for the served request
/// shape this backs).
#[derive(Clone, Copy)]
pub struct ImageInput<'a> {
    pub grid: (u32, u32),
    pub pixels: &'a [f32],
}

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

    /// Assemble a TRAINING build (`DecoderBuild::Batched`, decoder config
    /// carrying `Some(LoraCfg)`) from already-loaded HF tensors.
    ///
    /// `ParamStore::new_with_roles_src` needs every trainable/frozen tensor
    /// its role list names to be present in the source it reads from, and a
    /// base checkpoint never contains `.lora_a`/`.lora_b` (nothing trained
    /// them yet). So this mirrors `qwen3::finetune::finetune`'s own
    /// init-then-overwrite: seed the WHOLE decoder param set fresh via
    /// [`qwen3::init_weights`] (which gives every LoRA adapter its standard
    /// `A ~ small random, B = 0` init - `B = 0` is what makes the freshly
    /// LoRA-ed model start identical to the frozen base), then overwrite
    /// every base tensor the checkpoint actually has with the checkpoint's
    /// own value. The vision tower + mergers need no such merge - they carry
    /// no LoRA and are frozen, so they load directly from the checkpoint.
    ///
    /// Panics if `dcfg.lora` is `None`: this constructor exists for LoRA
    /// fine-tuning specifically (see [`crate::finetune`]), not general
    /// training - a full-parameter batched build has no missing tensors to
    /// merge and should use [`Self::new`] with [`DecoderBuild::Batched`] directly.
    #[allow(clippy::too_many_arguments)]
    pub fn from_tensors_train(
        tensors: Vec<checkpoint::safetensors::StTensor>,
        vcfg: VisionConfig,
        dcfg: QwenConfig,
        seq_len: u32,
        image_token_id: u32,
        image_row0: u32,
        n_visual: u32,
        mrope_section: [u32; 3],
        seed: u64,
    ) -> Qwen3Vl {
        assert!(dcfg.lora.is_some(), "from_tensors_train: dcfg.lora must be Some(..) - this constructor is for LoRA fine-tuning");
        let map: HashMap<String, Vec<f32>> = tensors.into_iter().map(|t| (t.name, t.data)).collect();
        let w = crate::import::partition(map, vcfg.deepstack_indexes.len());
        let mut dweights = qwen3::init_weights(&dcfg, seed);
        for (k, v) in w.decoder {
            dweights.insert(k, v);
        }
        Qwen3Vl::new(vcfg, dcfg, w.vision, w.main_merger, w.deepstack, &dweights, seq_len, image_token_id, image_row0, n_visual, mrope_section, DecoderBuild::Batched)
    }

    /// [`Self::from_tensors_train`] reading straight from an HF checkpoint
    /// directory (`config.json` + `model.safetensors[.index.json]`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_hf_train(
        dir: &str,
        vcfg: VisionConfig,
        dcfg: QwenConfig,
        seq_len: u32,
        image_token_id: u32,
        image_row0: u32,
        n_visual: u32,
        mrope_section: [u32; 3],
        seed: u64,
    ) -> Result<Qwen3Vl, String> {
        let tensors = checkpoint::safetensors::read_model_dir(std::path::Path::new(dir))?;
        Ok(Self::from_tensors_train(tensors, vcfg, dcfg, seq_len, image_token_id, image_row0, n_visual, mrope_section, seed))
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

    /// Zero the decoder's gradient buffers before a training step. The vision
    /// tower + mergers are never trainable in this composite (see
    /// [`DecoderBuild::Batched`]'s doc), so there is nothing else here to zero.
    pub fn zero_grads(&self) {
        self.decoder.zero_grads();
    }

    /// Backprop through the decoder from the loss [`Self::forward`] returned.
    /// Reaches only whatever the decoder's own training mode marks trainable
    /// (LoRA: `.lora_a`/`.lora_b` only) - the vision tower + mergers were
    /// built as plain frozen weights ([`Self::new`]) and have no gradient
    /// buffers, even though the loss depends on their output through the
    /// spliced image embeddings.
    pub fn backward(&self) {
        self.decoder.backward();
    }

    /// One AdamW step over the decoder's trainable parameters.
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, clip: Option<f32>, extra_scale: f32) {
        self.decoder.adamw_step(t, lr, wd, clip, extra_scale);
    }

    /// Names of the decoder's trainable parameters (LoRA: `.lora_a`/`.lora_b`
    /// only) - for inspection / finite-difference checks, never the hot
    /// training path.
    pub fn decoder_param_names(&self) -> Vec<String> {
        use ::model::Model;
        self.decoder.param_names()
    }

    /// Read back one decoder weight tensor by its `qwen3::Qwen` param-store
    /// name (e.g. `"blocks.0.attn.wq.weight.lora_a"`).
    pub fn read_decoder_weight(&self, name: &str) -> Vec<f32> {
        self.decoder.read_weight(name)
    }

    /// Overwrite one decoder weight tensor in place - finite-difference checks only.
    pub fn write_decoder_weight(&self, name: &str, data: &[f32]) {
        self.decoder.write_weight(name, data);
    }

    /// Read back one decoder parameter's gradient (post-[`Self::backward`]).
    pub fn read_decoder_grad(&self, name: &str) -> Vec<f32> {
        self.decoder.read_grad(name)
    }

    /// Write only the decoder's `.lora_a`/`.lora_b` tensors - never the
    /// frozen base, and never the vision tower (never trainable in this
    /// composite) - to `path`. Panics if this model's decoder was not built
    /// with a `LoraCfg`. See `qwen3::lora::save_adapter`.
    pub fn save_lora_adapter(&self, path: &str, card_id: &str, base_id: &str, dataset_id: Option<&str>) -> std::io::Result<()> {
        qwen3::lora::save_adapter(path, &self.decoder, card_id, base_id, dataset_id)
    }

    /// Greedy KV-cache generation for zero-or-more images + a text prompt:
    /// real `qwen3::Qwen` `step`/`step_embed` machinery (via this session's
    /// new `step_mrope`/`step_embed_mrope`, Phase 7a), not [`Self::forward`]'s
    /// training-loss-shaped batched path -- the gap noted as
    /// ("`Qwen3Vl::forward()` returns `f32`...
    /// there is no sampling loop").
    ///
    /// `images` supplies one [`ImageInput`] per image, in the order its
    /// vision-start/`[IMG]*`/vision-end run appears in `tokens` -- a single
    /// image is `&[image]`, matching every caller before multi-image support
    /// existed byte-for-byte (see `crates/qwen3vl/src/caps.rs`'s regression
    /// test). Prefill splices each image at its own token run the same way
    /// [`Self::forward`] does (image-placeholder token ids → step_embed_mrope
    /// with the matching merged visual row, walking a SINGLE row counter
    /// across every image in the request in order; every other token →
    /// step_mrope), so the KV cache never knows the difference, and never
    /// knows how many images contributed the rows it is walking (mirrors
    /// `qwen3::Qwen::prefill`'s own doc). Decode continues the position
    /// sequence past the prompt as plain text (T=H=W, +1 per token -- the same
    /// "media block then plain text" case `qwen3vl::mrope::get_rope_index_multi`
    /// documents).
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
    pub fn generate(&self, tokens: &[u32], images: &[ImageInput<'_>], max_new: u32, eos_ids: &[u32], sample: SampleParams, rng: &mut Rng) -> Vec<u32> {
        self.generate_cb(tokens, images, max_new, eos_ids, sample, rng, |_| {})
    }

    /// [`Self::generate`] with a per-token callback — the seam the served
    /// caps path uses to emit REAL streaming deltas (its ActionSpec declares
    /// `.streaming()`, which used to be satisfied by exactly two Progress
    /// emissions around the whole decode; audit F11).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_cb(
        &self,
        tokens: &[u32],
        images: &[ImageInput<'_>],
        max_new: u32,
        eos_ids: &[u32],
        sample: SampleParams,
        rng: &mut Rng,
        on_token: impl FnMut(u32),
    ) -> Vec<u32> {
        self.generate_timed(tokens, images, max_new, eos_ids, sample, rng, on_token).0
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
        images: &[ImageInput<'_>],
        max_new: u32,
        eos_ids: &[u32],
        sample: SampleParams,
        rng: &mut Rng,
        on_token: impl FnMut(u32),
    ) -> (Vec<u32>, StageTimes) {
        let mut st = StageTimes { prompt_tokens: tokens.len() as u32, ..StageTimes::default() };
        let d_model = self.decoder.cfg.d_model as usize;

        // Vision tower -> visual tokens (+ DeepStack taps), once per image (the
        // tower has no batch axis -- see `crate::caps`'s module doc). Each
        // image's merged rows and each DeepStack level's merged rows are
        // concatenated in image order into ONE flat buffer per level, because
        // that is the shape the decoder's splice/DeepStack buffers were sized
        // for: `qwen3::Qwen::enable_deepstack`'s `n_rows` is a construction-time
        // CAPACITY that must bound the SUM of every image's visual tokens in one
        // request (not any single image's own count), since `decode_steps`
        // addresses that buffer by a `deepstack_row` that walks the WHOLE
        // request's image rows, not a per-image index.
        let mut visual: Vec<f32> = Vec::new();
        let mut ds_levels: Vec<Vec<f32>> = vec![Vec::new(); self.ds_mergers.len()];
        let mut grids_llm: Vec<(u32, u32, u32)> = Vec::with_capacity(images.len());
        for img in images {
            let (gh, gw) = img.grid;
            let n = gh * gw;
            let t_vision = Instant::now();
            let (feats, tap_feats) = self.encoder.encode_with_taps(&self.vgpu, gh, gw, img.pixels, &self.vcfg.deepstack_indexes);
            st.vision_s += t_vision.elapsed().as_secs_f64();

            let t_merge = Instant::now();
            visual.extend(self.merger.merge(&self.vgpu, &feats, n));
            for (level, (tap, ds)) in tap_feats.iter().zip(&self.ds_mergers).enumerate() {
                ds_levels[level].extend(ds.merge(&self.vgpu, tap, n));
            }
            st.merge_s += t_merge.elapsed().as_secs_f64();

            grids_llm.push((1, gh / self.merge, gw / self.merge));
        }
        let n_visual = (visual.len() / d_model) as u32;
        assert_eq!(visual.len(), (n_visual as usize) * d_model);
        for (level, data) in ds_levels.iter().enumerate() {
            self.decoder.write_deepstack(level, data);
        }
        st.visual_tokens = n_visual;

        // M-RoPE positions for the KNOWN prompt (whole-sequence, once):
        // `get_rope_index_multi` with ONE placeholder kind (image) whose
        // `grids_llm` lists every image's merged `(t,h,w)` extent in the
        // order it appears in `tokens` -- the same multi-occurrence walk
        // `mrope::rope_index_two_adjacent_images` already gates, generalized
        // here from "always exactly one grid" to N.
        let prompt_positions = get_rope_index_multi(tokens, &[(self.image_token_id, &grids_llm)]);
        assert_eq!(prompt_positions.len(), tokens.len());

        let (out, prefill_s, decode_s, head_s) =
            self.splice_prefill_and_decode(tokens, &prompt_positions, self.image_token_id, &visual, max_new, eos_ids, sample, rng, on_token);
        st.prefill_s = prefill_s;
        st.decode_s = decode_s;
        st.head_s = head_s;
        st.new_tokens = out.len() as u32;
        (out, st)
    }

    /// [`Self::generate_timed`] generalized to a MULTI-FRAME video clip, with
    /// the T axis driven by each frame group's REAL timestamp
    /// ([`crate::mrope::get_rope_index_video`]) instead of `get_rope_index`'s
    /// `t = 1` image case. See that function's doc for exactly what is
    /// verified vs assumed about the real-timestamp formula.
    ///
    /// `grid` is the RAW per-frame patch grid `(gh, gw)` - identical for
    /// EVERY frame group, since Qwen3-VL's own video preprocessor picks one
    /// smart-resize target for the whole clip, not one per frame. `pixels`
    /// is [`crate::preprocess::pack_patches_temporal`]'s `[n_frames·gh·gw,
    /// patch_vec]` output, grid_t-major (group `g`'s rows are
    /// `pixels[g·gh·gw·pv .. (g+1)·gh·gw·pv]`). `frame_timestamps_s` carries
    /// one REAL timestamp (seconds) per merged temporal GROUP (i.e.
    /// `frame_timestamps_s.len() == n_frames`, not the raw, pre-temporal-pack
    /// frame count) and `video_token_id` is the placeholder id the prompt
    /// spliced these rows under (`Qwen3VlConfig::video_token_id` - kept as a
    /// call parameter rather than a constructor field so this method needs
    /// no change to any existing `Qwen3Vl` constructor call site).
    ///
    /// **The vision tower has no native multi-frame attention path** - its
    /// own `encode_with_taps` doc says "one image -> one whole-image span",
    /// full self-attention scoped to exactly `gh·gw` patches. Rather than
    /// change that (a real, unverified architectural claim about how
    /// Qwen3-VL's own ViT actually treats a video clip), this method encodes
    /// EACH FRAME GROUP SEPARATELY through the exact same, already-tested
    /// single-frame call [`Self::generate_timed`] makes, then concatenates
    /// the per-group merged visual rows (and per-level DeepStack taps)
    /// T-major - matching `get_rope_index_video`'s own T-major meshgrid
    /// order. No cross-frame attention inside the ViT is modeled; only the
    /// M-RoPE splice into the decoder sees the whole clip at once. This is a
    /// deliberately bounded scope, not a claim that Qwen3-VL's real video
    /// ViT windowing is ported.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_video_timed(
        &self,
        tokens: &[u32],
        grid: (u32, u32),
        n_frames: u32,
        pixels: &[f32],
        video_token_id: u32,
        frame_timestamps_s: &[f32],
        tokens_per_second: f32,
        max_new: u32,
        eos_ids: &[u32],
        sample: SampleParams,
        rng: &mut Rng,
        on_token: impl FnMut(u32),
    ) -> (Vec<u32>, StageTimes) {
        let mut st = StageTimes { prompt_tokens: tokens.len() as u32, ..StageTimes::default() };
        let (gh, gw) = grid;
        assert!(n_frames > 0, "a video needs at least one frame group");
        assert_eq!(frame_timestamps_s.len(), n_frames as usize, "one real timestamp per frame group");
        let n_per_frame = gh * gw;
        let pv = self.vcfg.patch_vec_dim() as usize;
        assert_eq!(
            pixels.len(),
            n_frames as usize * n_per_frame as usize * pv,
            "pixels must be pack_patches_temporal's [n_frames*gh*gw, patch_vec]"
        );
        let m2 = self.merge * self.merge;
        let n_visual_per_frame = n_per_frame / m2;
        let d_model = self.decoder.cfg.d_model as usize;

        // Vision tower -> visual tokens (+ DeepStack taps), one frame group
        // at a time (see this method's own doc on why). Vision and merge are
        // interleaved per group here, unlike generate_timed's split stages,
        // so both fold into `vision_s`.
        let t_vision = Instant::now();
        let n_levels = self.vcfg.deepstack_indexes.len();
        let mut visual = Vec::with_capacity(n_frames as usize * n_visual_per_frame as usize * d_model);
        let mut ds_levels: Vec<Vec<f32>> = vec![Vec::new(); n_levels];
        let group_len = n_per_frame as usize * pv;
        for g in 0..n_frames as usize {
            let group_pixels = &pixels[g * group_len..(g + 1) * group_len];
            let (feats, tap_feats) = self.encoder.encode_with_taps(&self.vgpu, gh, gw, group_pixels, &self.vcfg.deepstack_indexes);
            let group_visual = self.merger.merge(&self.vgpu, &feats, n_per_frame);
            assert_eq!(group_visual.len(), (n_visual_per_frame as usize) * d_model);
            visual.extend_from_slice(&group_visual);
            for (level, (tap, ds)) in tap_feats.iter().zip(&self.ds_mergers).enumerate() {
                ds_levels[level].extend(ds.merge(&self.vgpu, tap, n_per_frame));
            }
        }
        for (level, data) in ds_levels.iter().enumerate() {
            self.decoder.write_deepstack(level, data);
        }
        st.vision_s = t_vision.elapsed().as_secs_f64();
        st.visual_tokens = n_frames * n_visual_per_frame;

        let grid_llm = (n_frames, gh / self.merge, gw / self.merge);
        let prompt_positions = crate::mrope::get_rope_index_video(tokens, video_token_id, grid_llm, frame_timestamps_s, tokens_per_second);
        assert_eq!(prompt_positions.len(), tokens.len());

        let (out, prefill_s, decode_s, head_s) =
            self.splice_prefill_and_decode(tokens, &prompt_positions, video_token_id, &visual, max_new, eos_ids, sample, rng, on_token);
        st.prefill_s = prefill_s;
        st.decode_s = decode_s;
        st.head_s = head_s;
        st.new_tokens = out.len() as u32;
        (out, st)
    }

    /// [`Self::generate_video_timed`] without the per-stage timing.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_video_cb(
        &self,
        tokens: &[u32],
        grid: (u32, u32),
        n_frames: u32,
        pixels: &[f32],
        video_token_id: u32,
        frame_timestamps_s: &[f32],
        tokens_per_second: f32,
        max_new: u32,
        eos_ids: &[u32],
        sample: SampleParams,
        rng: &mut Rng,
        on_token: impl FnMut(u32),
    ) -> Vec<u32> {
        self.generate_video_timed(tokens, grid, n_frames, pixels, video_token_id, frame_timestamps_s, tokens_per_second, max_new, eos_ids, sample, rng, on_token)
            .0
    }

    /// Splice already-computed real M-RoPE `prompt_positions` (one `[t,h,w]`
    /// per token) into the decoder's KV cache - text tokens verbatim,
    /// `placeholder_token_id` rows replaced row-by-row from `visual` (in
    /// order) - then run greedy KV-cache decode. Shared by
    /// [`Self::generate_timed`] (single image) and
    /// [`Self::generate_video_timed`] (multi-frame video): both differ ONLY
    /// in how `prompt_positions`/`visual`/the placeholder id are produced,
    /// never in how they are consumed, so this is the ONE place that
    /// consumption happens (AGENTS.md: never duplicate an implementation -
    /// this loop used to exist only inside `generate_timed`, and the video
    /// path would otherwise have been a byte-identical second copy of it).
    #[allow(clippy::too_many_arguments)]
    fn splice_prefill_and_decode(
        &self,
        tokens: &[u32],
        prompt_positions: &[[u32; 3]],
        placeholder_token_id: u32,
        visual: &[f32],
        max_new: u32,
        eos_ids: &[u32],
        sample: SampleParams,
        rng: &mut Rng,
        mut on_token: impl FnMut(u32),
    ) -> (Vec<u32>, f64, f64, f64) {
        let d_model = self.decoder.cfg.d_model as usize;
        let n_visual = if d_model == 0 { 0 } else { visual.len() / d_model };

        // Prefill: placeholder rows via step_embed_mrope, text rows via
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
            // promise, which at a VL prompt (mostly placeholder rows) is one
            // submit+fence+map round trip per token removed.
            if tok == placeholder_token_id {
                let row = &visual[visual_row * d_model..(visual_row + 1) * d_model];
                let ds_row = Some(visual_row as u32);
                visual_row += 1;
                self.decoder.prefill_mrope(qwen3::model::PrefillInput::Embed(row), &cos, &sin, ds_row);
                continue;
            }
            self.decoder.prefill_mrope(qwen3::model::PrefillInput::Token(tok), &cos, &sin, None);
        }
        assert_eq!(visual_row, n_visual, "placeholder token count in the prompt must match the visual row count");
        // Prefill no longer reads anything back, so without this the host
        // clock would stop while the tape was still queued and the first head
        // dispatch below would inherit the whole backlog - a stage split that
        // charges the wrong stage is worse than no split at all. The fence is
        // free in wall-clock terms: the head's own readback would block on the
        // same work one statement later.
        self.decoder.gpu().poll_wait();
        let prefill_s = t_prefill.elapsed().as_secs_f64();

        // Decode: sample (or, at SampleParams::greedy(), argmax) continuing
        // the position sequence past the prompt as plain text. The head is applied ON the device the weights
        // already sit on (`Qwen::decode_logits`, vocab-tiled so each binding
        // stays inside the limit) and only a `[vocab]` row crosses back. The
        // host path this replaces read the WHOLE tied head table once per
        // caption -- 1.5 GB at the 4B config -- and then swept it, scalar and
        // single-threaded, for every generated token.
        let mut next_pos = prompt_positions.last().map(|p| p[0] + 1).unwrap_or(0);
        let mut out = Vec::with_capacity(max_new as usize);
        let (mut decode_s, mut head_s) = (0.0, 0.0);
        for _ in 0..max_new {
            let t_head = Instant::now();
            let next = crate::sample::sample_logits(&self.decoder.decode_logits(), sample.temperature, sample.top_k, sample.top_p, rng);
            head_s += t_head.elapsed().as_secs_f64();
            if eos_ids.contains(&next) {
                break;
            }
            out.push(next);
            on_token(next);
            let t_step = Instant::now();
            let (cos, sin) = mrope_tables(&[[next_pos; 3]], self.mrope_section, self.decoder.cfg.head_dim, self.decoder.cfg.rope_theta);
            self.decoder.prefill_mrope(qwen3::model::PrefillInput::Token(next), &cos, &sin, None);
            decode_s += t_step.elapsed().as_secs_f64();
            next_pos += 1;
        }
        (out, prefill_s, decode_s, head_s)
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
            tokens_per_second: 2,
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
            tokens_per_second: 2,
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
        let images = [ImageInput { grid: (4, 4), pixels: &pixels }];
        let mut gen_rng = Rng::new(99);
        let out1 = model.generate(&tokens, &images, max_new, &[], SampleParams::greedy(), &mut gen_rng);
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
        let out2 = model2.generate(&tokens, &images, max_new, &[], SampleParams::greedy(), &mut gen_rng2);
        assert_eq!(out1, out2, "greedy generation must be deterministic across independently-constructed identical models, regardless of the RNG state");

        // eos_ids actually stops generation early: the first token out1[0]
        // treated as an immediate stop id must yield an empty sequence.
        let mut gen_rng3 = Rng::new(2);
        let out3 = model.generate(&tokens, &images, max_new, &[out1[0]], SampleParams::greedy(), &mut gen_rng3);
        assert!(out3.is_empty(), "an eos id matching the very first generated token must stop before emitting it, got {out3:?}");

        // A real sampling request (temperature > 0) must actually consult the
        // RNG: two different seeds decoding the same prompt at temperature=1
        // must not always agree, or "sampling" would be greedy in disguise.
        let sample = SampleParams { temperature: 1.0, top_k: 0, top_p: 1.0 };
        let mut disagreed = false;
        for seed in 0..16u64 {
            let mut rng_a = Rng::new(seed);
            let mut rng_b = Rng::new(seed + 1000);
            let a = model.generate(&tokens, &images, max_new, &[], sample, &mut rng_a);
            let b = model.generate(&tokens, &images, max_new, &[], sample, &mut rng_b);
            if a != b {
                disagreed = true;
                break;
            }
        }
        assert!(disagreed, "temperature=1.0 sampling must vary across RNG seeds on at least one of 16 trials");

        // Regression: this exact tiny-config single-image scenario produced
        // [3, 3, 3, 3, 3] before multi-image support existed (captured from
        // this same test prior to `Qwen3Vl::generate`'s signature changing
        // from `(grid, pixels)` to `images: &[ImageInput]`). A single-image
        // request (`&[image]`) must still reach that exact sequence --
        // multi-image wiring must be additive, not a behavior change for the
        // request shape every existing caller already used.
        assert_eq!(out1, vec![3, 3, 3, 3, 3], "single-image generate output changed -- multi-image wiring must be byte-for-byte for one image");
    }

    /// Two images in one request: proves (a) it runs end to end without
    /// panic/assert with a real multi-image token stream and a combined
    /// DeepStack buffer wider than either image alone needs, and (b) the
    /// SECOND image actually influences the generated sequence (not merely
    /// decorative token-count padding) -- changing only its pixels, with the
    /// first image and the text prompt held fixed, must change the output.
    #[test]
    fn two_image_generate_runs_and_second_image_changes_output() {
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
            tokens_per_second: 2,
        };
        let dcfg = QwenConfig {
            vocab: 23,
            block_size: 24,
            n_layers: 2,
            d_model: 40,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8,
            d_ff: 64,
            rope_theta: 1.0e6,
            rms_eps: 1e-6,
            max_position_embeddings: 24,
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
        let vweights = rand_map(Rng::new(21), &vspecs);

        let merged = c * 4;
        let mweights = rand_map(
            Rng::new(22),
            &[
                ("ln.weight", c, true),
                ("ln.bias", c, false),
                ("fc1.weight", merged * merged, false),
                ("fc1.bias", merged, false),
                ("fc2.weight", 40 * merged, false),
                ("fc2.bias", 40, false),
            ],
        );
        let ds_mweights: Vec<HashMap<String, Vec<f32>>> = (0..vcfg.deepstack_indexes.len() as u64)
            .map(|i| {
                rand_map(
                    Rng::new(40 + i),
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

        let dweights = qwen3::init_weights(&dcfg, 23);

        // Prompt: 2 text, then TWO back-to-back 2x2-merged (4-token) image
        // runs (8 image tokens total, capacity 8 -- exactly the SUM two
        // images need, wider than either alone). No trailing text token: an
        // untrained tiny transformer's near-identity residual stream tends to
        // echo whatever token immediately precedes the generated position
        // (attention barely perturbs it at this scale/init), so ending the
        // prompt on a real vocab id would let that echo dominate the argmax
        // regardless of image content and make this test measure the echo,
        // not the image -- ending on an image row (a continuous embedding,
        // not any one vocab id's own row) removes that confound.
        let tokens: Vec<u32> = vec![1, 2, IMG, IMG, IMG, IMG, IMG, IMG, IMG, IMG];
        let seq_len = 24u32;
        let max_new = 5u32;

        let build_model = || {
            Qwen3Vl::new(
                vcfg.clone(),
                dcfg.clone(),
                vweights.clone(),
                mweights.clone(),
                ds_mweights.clone(),
                &dweights,
                seq_len,
                IMG,
                2,
                8, // capacity = sum of both images' 4 visual tokens each
                [2, 1, 1],
                DecoderBuild::Decode(Dtype::F32),
            )
        };

        let pv_total = (16 * vcfg.patch_vec_dim()) as usize;
        let image0: Vec<f32> = { let mut r = Rng::new(24); (0..pv_total).map(|_| r.next_f32() - 0.5).collect() };
        let image1_a: Vec<f32> = { let mut r = Rng::new(25); (0..pv_total).map(|_| r.next_f32() - 0.5).collect() };
        let image1_b: Vec<f32> = { let mut r = Rng::new(26); (0..pv_total).map(|_| r.next_f32() - 0.5).collect() };

        let model = build_model();
        let images_a = [ImageInput { grid: (4, 4), pixels: &image0 }, ImageInput { grid: (4, 4), pixels: &image1_a }];
        let mut rng_a = Rng::new(50);
        let out_a = model.generate(&tokens, &images_a, max_new, &[], SampleParams::greedy(), &mut rng_a);
        assert!(!out_a.is_empty(), "two-image generate produced no tokens");
        assert!(out_a.len() as u32 <= max_new);
        for &t in &out_a {
            assert!((t as usize) < 23, "generated token {t} outside vocab 23");
        }

        // Same first image, DIFFERENT second image, same model weights and
        // prompt: the sequence must change if the second image's rows are
        // really reaching the decoder (not dropped, zeroed, or only the
        // first image's rows ever read).
        let model2 = build_model();
        let images_b = [ImageInput { grid: (4, 4), pixels: &image0 }, ImageInput { grid: (4, 4), pixels: &image1_b }];
        let mut rng_b = Rng::new(51);
        let out_b = model2.generate(&tokens, &images_b, max_new, &[], SampleParams::greedy(), &mut rng_b);
        assert_ne!(out_a, out_b, "changing only the SECOND image did not change the generated sequence -- it is not actually influencing output");
    }

    /// Same tiny synthetic shape as [`generate_is_deterministic_and_respects_eos`],
    /// but a 2-frame VIDEO clip through [`Qwen3Vl::generate_video_cb`] instead
    /// of a single image: proves the real plumbing this change added (per-group
    /// vision encode -> concatenated splice via
    /// `crate::mrope::get_rope_index_video`'s real-timestamp T axis -> greedy
    /// decode) runs end to end, stays within vocab, and is deterministic.
    #[test]
    fn generate_video_is_deterministic_and_runs_end_to_end() {
        const VIDEO: u32 = 8;
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
            tokens_per_second: 2,
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
        let mut vspecs: Vec<(&str, usize, bool)> =
            vec![("patch_embed.weight", c * pv, false), ("patch_embed.bias", c, false), ("pos_embed", vcfg.num_position_embeddings as usize * c, false)];
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
        let vweights = rand_map(Rng::new(41), &vspecs);

        let merged = c * 4;
        let mweights = rand_map(
            Rng::new(42),
            &[
                ("ln.weight", c, true),
                ("ln.bias", c, false),
                ("fc1.weight", merged * merged, false),
                ("fc1.bias", merged, false),
                ("fc2.weight", 40 * merged, false),
                ("fc2.bias", 40, false),
            ],
        );
        let ds_mweights: Vec<HashMap<String, Vec<f32>>> = (0..vcfg.deepstack_indexes.len() as u64)
            .map(|i| {
                rand_map(
                    Rng::new(60 + i),
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
        let dweights = qwen3::init_weights(&dcfg, 43);

        // 2 text, 2 video placeholders (2 frame groups x a 2x2 raw grid,
        // merge=2 -> 1 visual token per group), 1 text.
        let tokens: Vec<u32> = vec![1, 2, VIDEO, VIDEO, 3];
        let seq_len = 16u32;
        let model = Qwen3Vl::new(vcfg.clone(), dcfg, vweights, mweights, ds_mweights, &dweights, seq_len, IMG, 2, 2, [2, 1, 1], DecoderBuild::Decode(Dtype::F32));

        let n_frames = 2u32;
        let pv_total = (n_frames * 4 * vcfg.patch_vec_dim()) as usize; // n_frames * (gh*gw) * patch_vec
        let mut rng = Rng::new(44);
        let pixels: Vec<f32> = (0..pv_total).map(|_| rng.next_f32() - 0.5).collect();
        // Two real, NON-uniformly-spaced timestamps -- exactly the case a
        // constant-fps assumption cannot reproduce.
        let frame_timestamps_s = [0.0f32, 2.0];

        let max_new = 5u32;
        let mut gen_rng1 = Rng::new(99);
        let out1 = model.generate_video_cb(&tokens, (2, 2), n_frames, &pixels, VIDEO, &frame_timestamps_s, 1.0, max_new, &[], SampleParams::greedy(), &mut gen_rng1, |_| {});
        assert!(!out1.is_empty(), "generate_video_cb produced no tokens");
        assert!(out1.len() as u32 <= max_new, "generate_video_cb exceeded max_new");
        for &t in &out1 {
            assert!((t as usize) < 23, "generated token {t} outside vocab 23");
        }

        // Greedy + no RNG: a second call must reproduce the same sequence.
        let mut gen_rng2 = Rng::new(100);
        let out2 = model.generate_video_cb(&tokens, (2, 2), n_frames, &pixels, VIDEO, &frame_timestamps_s, 1.0, max_new, &[], SampleParams::greedy(), &mut gen_rng2, |_| {});
        assert_eq!(out1, out2, "greedy video generation must be deterministic");

        // Different real timestamp spacing changes the ACTUAL positions this
        // path computes -- the property `crate::mrope::get_rope_index_video`'s
        // own unit tests pin exactly (this is an integration smoke, not a
        // second place to re-prove that formula: with tiny untrained random
        // weights an argmax output is not reliably sensitive to a position
        // perturbation, so asserting inequality here would be a flaky
        // over-claim, not a real spec requirement).
        let want_positions = crate::mrope::get_rope_index_video(&tokens, VIDEO, (2, 1, 1), &frame_timestamps_s, 1.0);
        let other_positions = crate::mrope::get_rope_index_video(&tokens, VIDEO, (2, 1, 1), &[0.0, 9.0], 1.0);
        assert_ne!(want_positions, other_positions, "sanity: the two timestamp spacings used above must actually produce different positions");
    }

    /// Finite-difference check on the decoder's LoRA delta parameters
    /// (`.lora_a`/`.lora_b`) through the FULL composite forward (vision tower
    /// -> merger -> M-RoPE splice -> LoRA decoder), not just the bare
    /// decoder `gradcheck::check_qwen_lora` already covers in isolation. This
    /// test's job is the COMPOSITION added by [`crate::finetune`]: that
    /// wiring `QwenConfig::lora` through [`Qwen3Vl::new`] and driving
    /// [`Qwen3Vl::forward`]/[`Qwen3Vl::backward`] still backprops correctly
    /// into the adapters once the vision splice and M-RoPE tables are also
    /// in the graph.
    ///
    /// Elementwise central differences (`gradcheck::elementwise_check`'s own
    /// recipe, hand-rolled here rather than pulling in the `gradcheck` crate:
    /// `Qwen3Vl::forward` takes extra arguments `gradcheck::CheckModel::loss`
    /// does not, so the fixed batch is closed over instead) on a HANDFUL of
    /// entries of one targeted projection's `lora_a`, not every entry - the
    /// composition either backprops correctly for every entry or for none,
    /// so an exhaustive sweep would only cost more wall time for the same
    /// answer. **Honest scope**: this is the smallest new trainable surface,
    /// per-parameter; a whole-model `gradcheck` entry point for this
    /// composite (a `CheckModel` impl over `Qwen3Vl` itself) is NOT added
    /// here and remains open work.
    #[test]
    fn lora_delta_gradient_matches_finite_difference() {
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
            deepstack_indexes: vec![], // no DeepStack needed for this check
            tokens_per_second: 2,
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
            lora: Some(qwen3::LoraCfg::attn(2, 4.0)),
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
        let vweights = rand_map(Rng::new(41), &vspecs);

        let merged = c * 4;
        let mweights = rand_map(
            Rng::new(42),
            &[
                ("ln.weight", c, true),
                ("ln.bias", c, false),
                ("fc1.weight", merged * merged, false),
                ("fc1.bias", merged, false),
                ("fc2.weight", 40 * merged, false),
                ("fc2.bias", 40, false),
            ],
        );

        let dweights = qwen3::init_weights(&dcfg, 43);

        let tokens: Vec<u32> = vec![1, 2, IMG, IMG, IMG, IMG, 3];
        let mut targets = vec![2u32, 3, 0, 0, 0, 0, 5];
        for t in targets.iter_mut().take(6).skip(2) {
            *t = qwen3::IGNORE;
        }

        let model =
            Qwen3Vl::new(vcfg.clone(), dcfg, vweights, mweights, vec![], &dweights, tokens.len() as u32, IMG, 2, 4, [2, 1, 1], DecoderBuild::Batched);

        let pv_total = (16 * vcfg.patch_vec_dim()) as usize;
        let mut rng = Rng::new(44);
        let pixels: Vec<f32> = (0..pv_total).map(|_| rng.next_f32() - 0.5).collect();

        // Move B off its zero init first - an FD check at B=0 can't tell a
        // correct dB from a wrong one, both start at exactly zero. Same
        // recipe as `gradcheck::check_qwen_lora`.
        for step in 1..=5u32 {
            model.zero_grads();
            model.forward(&tokens, &targets, (4, 4), &pixels);
            model.backward();
            model.adamw_step(step, 5e-2, 0.0, Some(1.0), 1.0);
        }

        let loss = |m: &Qwen3Vl| m.forward(&tokens, &targets, (4, 4), &pixels);
        let (eps, name) = (5e-3f32, "blocks.0.attn.wq.weight.lora_a");
        model.zero_grads();
        let _ = loss(&model);
        model.backward();
        let w0 = model.read_decoder_weight(name);
        let g = model.read_decoder_grad(name);
        assert_eq!(g.len(), w0.len(), "{name}: grad/weight size mismatch");
        assert!(!w0.is_empty());

        let mut w = w0.clone();
        let n_check = w0.len().min(6);
        for i in 0..n_check {
            w[i] = w0[i] + eps;
            model.write_decoder_weight(name, &w);
            let lp = loss(&model);
            w[i] = w0[i] - eps;
            model.write_decoder_weight(name, &w);
            let lm = loss(&model);
            w[i] = w0[i];

            let numeric = (lp - lm) / (2.0 * eps);
            let analytic = g[i];
            let abs_err = (analytic - numeric).abs();
            let denom = analytic.abs().max(numeric.abs()).max(1e-3);
            let rel_err = abs_err / denom;
            assert!(rel_err < 5e-2, "{name}[{i}]: analytic {analytic} vs numeric {numeric} (rel_err {rel_err})");
        }
        model.write_decoder_weight(name, &w0);
    }
}
