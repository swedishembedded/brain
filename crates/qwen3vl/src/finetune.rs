// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA fine-tuning over a captioned-image dataset: turn a folder of images +
//! captions (`data::imageset`, the format `brain label` writes) into a trained
//! LoRA adapter for the decoder.
//!
//! Reuses, rather than reinvents, three pieces that already exist elsewhere in
//! this workspace:
//!  - the decoder's own LoRA machinery ([`Qwen3Vl::new`] with
//!    [`crate::model::DecoderBuild::Batched`] and `cfg.text.lora = Some(..)`
//!    builds a `qwen3::Qwen` whose `.lora_a`/`.lora_b` are the only trainable
//!    parameters - the exact mechanism `qwen3::finetune::Mode::Lora` drives);
//!  - `data::imageset::load_dir` for the captioned-image folder format (never a
//!    hand-rolled scanner - see that module's own doc);
//!  - `qwen3::lora::save_adapter` (via [`Qwen3Vl::save_lora_adapter`]) for the
//!    adapter-only checkpoint.
//!
//! **Only the decoder is trainable.** The vision tower + PatchMerger(s) stay
//! frozen, exactly as [`crate::model::DecoderBuild::Batched`]'s own doc
//! describes this composite - `Qwen3Vl::new` never gives them gradient
//! buffers. This is a real, honest scope limit (a caption LoRA that could also
//! adapt the vision tower would need `DecoderBuild` extended with a trainable
//! vision path, which does not exist yet), not an oversight: it is exactly
//! the surface `qwen3::finetune::Mode::Lora` targets on the text-only model,
//! carried over unchanged.
//!
//! **Training images share ONE fixed size for the whole run**, unlike
//! `caps.rs`'s per-request `smart_resize`. The BATCHED decoder graph
//! ([`DecoderBuild::Batched`](crate::model::DecoderBuild::Batched)) is built
//! once at a fixed `seq_len` and a fixed image-token placement (`image_row0`/
//! `n_visual`), so every sample in a run must produce the same visual-token
//! count - `data::imageset::load_dir`'s own center-crop-then-resize-to-`size`
//! already guarantees a fixed image geometry, so this is just carrying that
//! fact through to the patch grid rather than a new restriction.

use std::path::Path;
use std::time::Instant;

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use qwen3::{IGNORE, LoraCfg};
use serde_json::Value;

use crate::config::Qwen3VlConfig;
use crate::model::Qwen3Vl;
use crate::preprocess::{normalize_unit, pack_patches};

/// The default LoRA target set: the four attention projections plus the three
/// MLP projections - the same set `qwen3::finetune::Mode::Lora` uses.
pub fn lora_targets() -> Vec<String> {
    ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect()
}

/// A fixed generic captioning instruction. `data::imageset::CaptionFile` only
/// carries `filename -> caption text`, no separate per-sample instruction, so
/// every training example is framed as the same captioning task and the
/// caption text is the target completion - the (image, instruction,
/// target-text) triple the dataset format actually supports.
const INSTRUCTION: &str = "Describe the image.";

/// LoRA fine-tuning hyperparameters.
#[derive(Clone, Debug)]
pub struct TrainOpts {
    pub rank: u32,
    pub alpha: f32,
    pub steps: u32,
    pub lr: f32,
    pub min_lr: f32,
    pub warmup: u32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    /// Training image square size, px. Must be a multiple of
    /// `patch_size * spatial_merge_size` (32 for the 4B config).
    pub size: u32,
    /// Fixed per-sample token budget (prefix + caption + eos, padded).
    pub seq_len: u32,
    pub seed: u64,
    /// Wall-clock checkpoint cadence, seconds; 0 disables periodic saves
    /// (only the final one runs). Same convention as `model::FitOpts`.
    pub checkpoint_secs: u64,
}

impl Default for TrainOpts {
    fn default() -> Self {
        TrainOpts {
            rank: 8,
            alpha: 16.0,
            steps: 200,
            lr: 1e-4,
            min_lr: 1e-5,
            warmup: 10,
            weight_decay: 0.0,
            grad_clip: 1.0,
            size: 224,
            seq_len: 256,
            seed: 0,
            checkpoint_secs: 300,
        }
    }
}

/// Fine-tune a LoRA adapter for `base_dir`'s decoder on the captioned-image
/// folder `data_dir`, writing the adapter checkpoint to `save_path`. Returns
/// `(initial_loss, final_loss)`.
pub fn run(
    base_dir: &str,
    data_dir: &Path,
    opts: &TrainOpts,
    save_path: &str,
    cancel: &capability::CancelToken,
    mut progress: impl FnMut(u32, u32, String),
) -> Result<(f32, f32), String> {
    // 1. Base config + tokenizer.
    let cfg_path = format!("{base_dir}/config.json");
    let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| format!("qwen3vl finetune: cannot read {cfg_path}: {e}"))?;
    let cfg_json: Value = serde_json::from_str(&cfg_text).map_err(|e| format!("qwen3vl finetune: cannot parse {cfg_path}: {e}"))?;
    let mut cfg = Qwen3VlConfig::from_hf(&cfg_json);
    let tok = QwenBpe::from_dir(base_dir).map_err(|e| format!("qwen3vl finetune: tokenizer: {e}"))?;

    // 2. Dataset.
    let factor = cfg.vision.patch_size * cfg.vision.spatial_merge_size;
    if !opts.size.is_multiple_of(factor) {
        return Err(format!("qwen3vl finetune: size {} must be a multiple of patch_size*spatial_merge_size ({factor})", opts.size));
    }
    let samples = data::imageset::load_dir(data_dir, opts.size, |w| progress(0, opts.steps, format!("dataset: {w}")))?;
    progress(0, opts.steps, format!("loaded {} images from {}", samples.len(), data_dir.display()));

    // 3. Enable LoRA on the decoder config.
    cfg.text.lora = Some(LoraCfg { rank: opts.rank, alpha: opts.alpha, targets: lora_targets() });

    // 4. Fixed prompt template -> image placement, constant for every sample
    // (see this module's own doc on the fixed-size-per-run constraint).
    let gh = opts.size / cfg.vision.patch_size;
    let gw = gh;
    let n_visual = gh * gw / (cfg.vision.spatial_merge_size * cfg.vision.spatial_merge_size);
    let prefix = build_prefix(&tok, &cfg, n_visual);
    let image_row0 = prefix.iter().position(|&t| t == cfg.image_token_id).expect("build_prefix always inserts the image run") as u32;

    // 5. Per-sample (tokens, targets, packed patches); a caption too long for
    // `seq_len` is skipped (named, not silently truncated/corrupted).
    let pad_id = tok.encode("<|im_end|>").first().copied().unwrap_or(0);
    let mut examples: Vec<(Vec<u32>, Vec<u32>, Vec<f32>)> = Vec::new();
    for s in &samples {
        let mut chw = imaging::pixels::hwc_to_chw(&s.hwc, 3, s.size as usize, s.size as usize);
        normalize_unit(&mut chw);
        let pixels = pack_patches(&chw, cfg.vision.in_channels, opts.size, opts.size, cfg.vision.patch_size, cfg.vision.spatial_merge_size, cfg.vision.temporal_patch_size);
        match build_sample(&prefix, &tok, &s.prompt, opts.seq_len, pad_id) {
            Some((tokens, targets)) => examples.push((tokens, targets, pixels)),
            None => progress(0, opts.steps, format!("skipping {}: caption too long for seq_len {}", s.path.display(), opts.seq_len)),
        }
    }
    if examples.is_empty() {
        return Err(format!("qwen3vl finetune: no usable samples - every caption exceeded seq_len {}", opts.seq_len));
    }

    // 6. Build the trainable model.
    let model = Qwen3Vl::from_hf_train(base_dir, cfg.vision.clone(), cfg.text.clone(), opts.seq_len, cfg.image_token_id, image_row0, n_visual, cfg.mrope_section, opts.seed)?;

    // 7. Train.
    let fit = model::FitOpts { steps: opts.steps, lr: opts.lr, min_lr: opts.min_lr, warmup: opts.warmup, decay_iters: opts.steps.max(1), weight_decay: opts.weight_decay, grad_clip: opts.grad_clip, ..Default::default() };
    let mut rng = data::rng::Rng::new(opts.seed ^ 0xA5A5_5A5A);

    let initial = {
        let mut sum = 0.0f32;
        let n = examples.len().min(3);
        for (tokens, targets, pixels) in examples.iter().take(n) {
            sum += model.forward(tokens, targets, (gh, gw), pixels);
        }
        sum / n as f32
    };

    let mut last = initial;
    let mut last_save = Instant::now();
    for step in 0..opts.steps {
        if cancel.is_cancelled() {
            return Err("cancelled".to_string());
        }
        let lr = model::cosine_lr(step, &fit);
        let idx = rng.gen_range_inclusive(0, examples.len() as i64 - 1) as usize;
        let (tokens, targets, pixels) = &examples[idx];

        model.zero_grads();
        let loss = model.forward(tokens, targets, (gh, gw), pixels);
        model.backward();
        let clip = (opts.grad_clip > 0.0).then_some(opts.grad_clip);
        model.adamw_step(step + 1, lr, opts.weight_decay, clip, 1.0);
        last = loss;
        progress(step + 1, opts.steps, format!("step {} loss {:.4}", step + 1, loss));

        if opts.checkpoint_secs > 0 && last_save.elapsed().as_secs() >= opts.checkpoint_secs {
            model
                .save_lora_adapter(save_path, "qwen3vl-lora", base_dir, Some(data_dir.to_string_lossy().as_ref()))
                .map_err(|e| format!("qwen3vl finetune: checkpoint save: {e}"))?;
            last_save = Instant::now();
        }
    }
    model
        .save_lora_adapter(save_path, "qwen3vl-lora", base_dir, Some(data_dir.to_string_lossy().as_ref()))
        .map_err(|e| format!("qwen3vl finetune: save: {e}"))?;
    Ok((initial, last))
}

/// The fixed prompt prefix every training sample shares: `<|im_start|>user\n`,
/// `<|vision_start|>`, `n_visual` image placeholders, `<|vision_end|>`, the
/// fixed instruction, then `<|im_end|>\n<|im_start|>assistant\n` - the same
/// chat shape `caps.rs::Prepared::build` assembles for serving, so a trained
/// adapter matches what `generate` actually feeds the decoder.
fn build_prefix(tok: &impl Tokenizer, cfg: &Qwen3VlConfig, n_visual: u32) -> Vec<u32> {
    let mut tokens = tok.encode("<|im_start|>user\n");
    tokens.push(cfg.vision_start_token_id);
    tokens.extend(std::iter::repeat_n(cfg.image_token_id, n_visual as usize));
    tokens.push(cfg.vision_end_token_id);
    tokens.extend(tok.encode(&format!("{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n")));
    tokens
}

/// Build one training example's `(tokens, targets)`, both `seq_len` long.
/// `targets[i]` is the token to predict AT position `i` (i.e. `tokens[i+1]`)
/// for every position inside the caption+eos response, `IGNORE` everywhere
/// else (the prompt/image prefix and any padding). Returns `None` when
/// `prefix.len() + response.len() > seq_len` - the caller skips and warns
/// rather than silently truncating a caption.
fn build_sample(prefix: &[u32], tok: &impl Tokenizer, caption: &str, seq_len: u32, pad_id: u32) -> Option<(Vec<u32>, Vec<u32>)> {
    let mut response = tok.encode(caption);
    response.extend(tok.encode("<|im_end|>"));
    if prefix.len() + response.len() > seq_len as usize {
        return None;
    }
    let mut tokens = prefix.to_vec();
    tokens.extend_from_slice(&response);
    tokens.resize(seq_len as usize, pad_id);

    let mut targets = vec![IGNORE; seq_len as usize];
    for (k, &tid) in response.iter().enumerate() {
        targets[prefix.len() + k - 1] = tid;
    }
    Some((tokens, targets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VisionConfig;

    fn tiny_vl_cfg() -> Qwen3VlConfig {
        Qwen3VlConfig {
            vision: VisionConfig {
                depth: 1,
                hidden: 16,
                num_heads: 2,
                intermediate: 32,
                patch_size: 2,
                temporal_patch_size: 1,
                spatial_merge_size: 2,
                num_position_embeddings: 64,
                out_hidden_size: 16,
                in_channels: 2,
                deepstack_indexes: vec![],
                tokens_per_second: 2,
            },
            text: qwen3::QwenConfig {
                vocab: 64,
                block_size: 64,
                n_layers: 1,
                d_model: 16,
                n_heads: 2,
                n_kv_heads: 2,
                head_dim: 8,
                d_ff: 32,
                rope_theta: 1.0e6,
                rms_eps: 1e-6,
                max_position_embeddings: 64,
                tie_embeddings: true,
                qk_norm: true,
                attn_bias: false,
                lora: None,
            },
            mrope_section: [4, 2, 2],
            image_token_id: 5,
            video_token_id: 6,
            vision_start_token_id: 3,
            vision_end_token_id: 4,
        }
    }

    /// A stub tokenizer good enough for `build_prefix`/`build_sample`'s own
    /// unit tests: `encode` just maps each byte to `byte as u32 + 10` (keeps
    /// every id well clear of the special ids 0..9 the tiny config above
    /// uses), `decode` is never exercised here.
    struct ByteTok;
    impl Tokenizer for ByteTok {
        fn encode(&self, s: &str) -> Vec<u32> {
            s.bytes().map(|b| b as u32 + 10).collect()
        }
        fn decode(&self, ids: &[u32]) -> String {
            String::from_utf8(ids.iter().map(|&i| (i - 10) as u8).collect()).unwrap_or_default()
        }
        fn vocab_size(&self) -> usize {
            256 + 10
        }
    }

    #[test]
    fn build_sample_targets_only_the_response_and_pads_with_ignore() {
        let prefix = vec![100u32, 101, 102]; // stand-in prefix, image run already inside
        let tok = ByteTok;
        let seq_len = 20u32; // prefix(3) + "hi"(2) + "<|im_end|>"(10) = 15, plus padding room
        let (tokens, targets) = build_sample(&prefix, &tok, "hi", seq_len, 99).unwrap();
        assert_eq!(tokens.len(), seq_len as usize);
        assert_eq!(targets.len(), seq_len as usize);
        // prefix unchanged
        assert_eq!(&tokens[0..3], &prefix[..]);
        // targets IGNORE strictly BEFORE the prefix's last position - the
        // last prefix token is the one that predicts the first response
        // token, so it legitimately carries a real target, not IGNORE.
        assert!(targets[0..2].iter().all(|&t| t == IGNORE));
        // response = encode("hi") ++ encode("<|im_end|>"), target[i] = tokens[i+1]
        // for the response span.
        let resp_len = tok.encode("hi").len() + tok.encode("<|im_end|>").len();
        for i in 0..resp_len {
            assert_eq!(targets[3 + i - 1], tokens[3 + i], "target at response position {i} must equal the next token");
        }
        // padding beyond the response is IGNORE, filled with pad_id.
        let end = 3 + resp_len;
        assert!(targets[end..].iter().all(|&t| t == IGNORE), "padding must never be a training target");
        assert!(tokens[end..].iter().all(|&t| t == 99), "padding must use pad_id");
    }

    #[test]
    fn build_sample_skips_a_caption_too_long_for_seq_len() {
        let prefix = vec![1u32; 5];
        let tok = ByteTok;
        assert!(build_sample(&prefix, &tok, "a very very long caption that will not fit", 8, 0).is_none());
    }

    #[test]
    fn build_prefix_places_exactly_n_visual_image_tokens() {
        let cfg = tiny_vl_cfg();
        let tok = ByteTok;
        let n_visual = 6u32;
        let prefix = build_prefix(&tok, &cfg, n_visual);
        let n = prefix.iter().filter(|&&t| t == cfg.image_token_id).count();
        assert_eq!(n as u32, n_visual);
        let row0 = prefix.iter().position(|&t| t == cfg.image_token_id).unwrap();
        // every image token is contiguous, matching the mm-splice contract.
        for t in &prefix[row0..row0 + n_visual as usize] {
            assert_eq!(*t, cfg.image_token_id);
        }
    }

    /// The training-convergence smoke lives in `crate::train_smoke` (mirrors
    /// `fastvlm::train_smoke`'s own naming/shape) rather than here - this
    /// file's tests cover the data-prep helpers `run` is built from.
    #[test]
    fn lora_targets_are_the_seven_attn_and_mlp_projections() {
        let t = lora_targets();
        assert_eq!(t, ["wq", "wk", "wv", "wo", "gate", "up", "down"]);
    }
}
