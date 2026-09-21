// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`EncoderFineTuner`]: full-encoder contrastive fine-tuning of the
//! LFM2.5-Encoder backbone, DRIVING `lfm2::model::Lfm`'s seeded backward
//! pass (`Lfm::seed_buf`/`backward_seeded`, gradient-checked directly in
//! `crates/gradcheck/src/lfm2_seeded.rs`) with the same symmetric InfoNCE
//! objective [`crate::EmbeddingTrainer`] trains a FROZEN-backbone projection
//! head with - see `crate::embed_train::info_nce_core`, shared by both.
//!
//! Unlike `EmbeddingTrainer`, which caches embeddings once and trains only a
//! small head on top, this type re-runs the encoder's own forward AND
//! backward every step - it is training the checkpoint's own weights
//! directly, so there is no frozen cache to keep the work tractable with.
//! Qwen3 has no seeded backward pass built (only LFM2 does), so this type is
//! LFM2-specific rather than a second `Backend` arm on a shared type the way
//! `EmbeddingPipeline` itself dispatches - there is no second architecture
//! yet for it to share a dispatch with.
//!
//! ## The exact-length constraint
//!
//! LFM2's bidirectional attention has no padding mask (see
//! `lfm2::caps::EncoderAction`'s own doc), so every row [`EncoderFineTuner`]
//! builds its graph for must be a real, fully-populated sequence - there is
//! no way to leave some rows shorter than others the way a causal decoder's
//! padding-plus-mask can. This type resolves that by fixing one `seq_len` at
//! construction (like [`crate::EmbeddingPipelineBuilder::capacity`]) and
//! building ONE `[2*batch_size, seq_len]` graph up front: every
//! `EncoderFineTuner::step` call tokenizes each anchor/positive text,
//! truncates it to exactly `seq_len` tokens, and refuses (rather than pads)
//! any text that tokenizes shorter. A caller training on real corpora picks
//! `seq_len` to fit its shortest examples, or chunks longer ones itself.
//!
//! ## Mean-pool's backward
//!
//! The forward pooling this type mirrors is the same one
//! `crate::embedding`'s own LFM2 backend runs (mean over every token row,
//! then L2-normalize) - so a fine-tuned checkpoint's zero-shot behavior via
//! `EmbeddingPipeline` is exactly what training here optimizes. Mean pool's
//! adjoint is "divide by `n` and broadcast to every row": each sequence's
//! `dL/d(pooled)/seq_len` is written identically into all `seq_len` rows of
//! [`lfm2::model::Lfm::seed_buf`] that sequence occupies - the same
//! broadcast `crates/decide/src/head.rs`'s own doc describes for its
//! (different) pooling head.

use crate::embed_train::{info_nce_core, normalize_bwd};
use crate::{Error, Result};

/// A live, trainable LFM2.5-Encoder plus its own optimizer state, contrastively
/// fine-tuned end to end. See this module's own doc for the exact-length
/// constraint every training batch must satisfy.
pub struct EncoderFineTuner {
    model: lfm2::model::Lfm,
    tok: data::qwen_tokenizer::QwenBpe,
    seq_len: u32,
    batch_size: usize,
    d_model: usize,
    temperature: f32,
    step: u32,
}

impl std::fmt::Debug for EncoderFineTuner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderFineTuner").field("seq_len", &self.seq_len).field("batch_size", &self.batch_size).field("step", &self.step).finish()
    }
}

impl EncoderFineTuner {
    /// Load a trainable LFM2.5-Encoder from a local checkpoint and build its
    /// graph at the FIXED `[2*batch_size, seq_len]` shape every
    /// [`EncoderFineTuner::step`] call reuses - one anchor row plus one
    /// positive row per pair, every step. `weights`/`tokenizer` are local
    /// paths (this type trains a checkpoint already on disk, not a resolved
    /// hub id - a caller resolving one first is `EmbeddingPipeline`'s job,
    /// not this type's).
    pub fn open(weights: impl AsRef<str>, tokenizer: impl AsRef<str>, batch_size: usize, seq_len: u32) -> Result<EncoderFineTuner> {
        assert!(batch_size > 0, "batch_size must be at least 1");
        assert!(seq_len > 0, "seq_len must be at least 1");
        let tok = data::qwen_tokenizer::QwenBpe::from_file(tokenizer.as_ref()).map_err(Error::Backend)?;
        let model = lfm2::model::Lfm::load_train(weights.as_ref(), (2 * batch_size) as u32, seq_len);
        let d_model = model.cfg.d_model as usize;
        Ok(EncoderFineTuner { model, tok, seq_len, batch_size, d_model, temperature: 0.05, step: 0 })
    }

    /// See [`crate::EmbeddingTrainer::temperature`] - the same knob, the same
    /// default.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// One training step: tokenize the whole batch into ONE forward graph,
    /// mean-pool and L2-normalize each row's own sequence, the symmetric
    /// InfoNCE loss and gradient over those pooled vectors
    /// (`crate::embed_train::info_nce_core`), scatter that gradient back
    /// across every row of the sequence it came from, `backward_seeded`, one
    /// AdamW step (`wd = 0.0`, gradient-norm clipped to `1.0` - the same
    /// choice every other small trainer in this workspace's engine crates
    /// makes for a from-checkpoint fine-tune). Returns the batch's mean loss.
    ///
    /// `anchors.len()` and `positives.len()` must both equal this tuner's
    /// `batch_size`. Every text must tokenize to AT LEAST `seq_len` tokens
    /// (longer is truncated, the same truncate-not-refuse rule
    /// `EmbeddingOptions::max_tokens` uses elsewhere in this SDK); shorter is
    /// refused rather than padded - see this module's own doc for why.
    pub fn step(&mut self, anchors: &[&str], positives: &[&str], lr: f32) -> Result<f32> {
        assert_eq!(anchors.len(), self.batch_size, "expected {} anchors, got {}", self.batch_size, anchors.len());
        assert_eq!(positives.len(), self.batch_size, "expected {} positives, got {}", self.batch_size, positives.len());

        let seq_len = self.seq_len as usize;
        let mut x = Vec::with_capacity(2 * self.batch_size * seq_len);
        for text in anchors.iter().chain(positives.iter()) {
            x.extend(self.tokenize_exact(text)?);
        }
        let y = vec![lfm2::model::IGNORE; x.len()];
        self.model.set_batch(&x, &y);
        self.model.forward();

        let d = self.d_model;
        let hidden = self.model.read_hidden();
        let rows: Vec<&[f32]> = hidden.chunks_exact(d).collect();
        assert_eq!(rows.len(), 2 * self.batch_size * seq_len, "read_hidden: unexpected row count");

        // Mean-pool each sequence's own `seq_len` rows, keeping both the raw
        // (unnormalized) pool and its norm - `normalize_bwd` needs the norm,
        // not just the unit vector, same reason `crate::embed_train::project_fwd`
        // keeps both.
        let pool = |seq: usize| -> (Vec<f32>, f64) {
            let start = seq * seq_len;
            let mut z = vec![0.0f64; d];
            for row in &rows[start..start + seq_len] {
                for (m, &v) in z.iter_mut().zip(*row) {
                    *m += v as f64;
                }
            }
            for v in &mut z {
                *v /= seq_len as f64;
            }
            let norm = z.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            let u: Vec<f32> = z.iter().map(|v| (v / norm) as f32).collect();
            (u, norm)
        };

        let (mut ua, mut norm_a) = (Vec::with_capacity(self.batch_size), Vec::with_capacity(self.batch_size));
        for i in 0..self.batch_size {
            let (u, n) = pool(i);
            ua.push(u);
            norm_a.push(n);
        }
        let (mut up, mut norm_p) = (Vec::with_capacity(self.batch_size), Vec::with_capacity(self.batch_size));
        for i in 0..self.batch_size {
            let (u, n) = pool(self.batch_size + i);
            up.push(u);
            norm_p.push(n);
        }

        let ua_ref: Vec<&[f32]> = ua.iter().map(|v| v.as_slice()).collect();
        let up_ref: Vec<&[f32]> = up.iter().map(|v| v.as_slice()).collect();
        let (loss, d_ua, d_up) = info_nce_core(d, &ua_ref, &up_ref, self.temperature);

        // Scatter dL/d(pooled_raw)/seq_len across every row of the sequence
        // it pooled from - see this module's own doc for why this IS mean
        // pool's adjoint.
        let mut seed = vec![0.0f32; rows.len() * d];
        let scatter = |seed: &mut [f32], seq: usize, u: &[f32], norm: f64, d_u: &[f32]| {
            let d_z = normalize_bwd(u, norm, d_u);
            let d_z: Vec<f32> = d_z.iter().map(|v| v / seq_len as f32).collect();
            let start = seq * seq_len * d;
            for r in 0..seq_len {
                seed[start + r * d..start + (r + 1) * d].copy_from_slice(&d_z);
            }
        };
        for i in 0..self.batch_size {
            scatter(&mut seed, i, &ua[i], norm_a[i], &d_ua[i]);
        }
        for i in 0..self.batch_size {
            scatter(&mut seed, self.batch_size + i, &up[i], norm_p[i], &d_up[i]);
        }

        self.model.zero_grads();
        self.model.gpu.write_f32(self.model.seed_buf(), &seed);
        self.model.backward_seeded();
        self.step += 1;
        self.model.adamw_step(self.step, lr, 0.0, Some(1.0), 1.0);

        Ok(loss)
    }

    /// Tokenize `text` (this checkpoint's own template prefix plus
    /// `text`'s own tokens), truncated to exactly this tuner's `seq_len` -
    /// or a typed error naming both lengths if it is too short to truncate
    /// at all. See this module's own doc for why a short text is refused
    /// rather than padded.
    fn tokenize_exact(&self, text: &str) -> Result<Vec<u32>> {
        use data::tokenizer::Tokenizer;
        let mut ids: Vec<u32> = self.tok.template_prefix().to_vec();
        ids.extend(self.tok.encode(text));
        if ids.len() < self.seq_len as usize {
            return Err(Error::Backend(format!(
                "encoder fine-tune: {text:?} tokenizes to {} tokens, short of this tuner's fixed seq_len ({}) -- LFM2 has no padding mask, so every batch row must be an exact-length sequence; supply longer text or rebuild with a smaller seq_len",
                ids.len(),
                self.seq_len
            )));
        }
        ids.truncate(self.seq_len as usize);
        Ok(ids)
    }

    /// Save the current (fine-tuned) weights to `path`, in the same
    /// carded-safetensors format [`crate::EmbeddingPipeline`] and
    /// `lfm2::caps` read - so a caller can point a fresh pipeline straight
    /// at the result.
    pub fn save(&self, path: &str) {
        self.model.save(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_fixture(dir: &std::path::Path, block: u32) -> (std::path::PathBuf, std::path::PathBuf) {
        std::fs::create_dir_all(dir).unwrap();
        let cfg = lfm2::LfmConfig { vocab: 256, block_size: block, ..lfm2::LfmConfig::tiny() };
        let init = lfm2::init::init_weights(&cfg, 11);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone()))
            .collect();
        let weights = dir.join("lfm2-finetune.safetensors");
        let card = checkpoint::st::ModelCard::new("brain/lfm2", lfm2::spec::CARD_FAMILY);
        checkpoint::save_carded(weights.to_str().unwrap(), cfg.to_json(), &tensors, &card);

        let mut vocab = serde_json::Map::new();
        for (i, c) in data::bpe::bytes_to_unicode().iter().enumerate() {
            vocab.insert(c.to_string(), serde_json::json!(i));
        }
        let tok_path = dir.join("tokenizer.json");
        std::fs::write(&tok_path, serde_json::json!({"model": {"vocab": vocab, "merges": []}}).to_string()).unwrap();

        (weights, tok_path)
    }

    /// Training must reduce the batch's own loss over its own pairs - the
    /// one end-to-end contract `step` owes a caller, the same discipline
    /// `crate::embed_train`'s own `training_reduces_loss_on_its_own_batch`
    /// applies to the frozen-backbone head. Byte-level tokenization makes
    /// text length exactly controllable, so every anchor/positive here is
    /// deliberately the same byte length as `seq_len`.
    #[test]
    fn training_reduces_loss_on_its_own_batch() {
        let dir = std::env::temp_dir().join(format!("brain-sdk-encoder-finetune-{}", std::process::id()));
        let (weights, tok) = write_fixture(&dir, 16);
        let mut tuner = EncoderFineTuner::open(weights.to_str().unwrap(), tok.to_str().unwrap(), 3, 8).unwrap().temperature(0.2);

        let anchors = ["aardvark", "beekeeper", "chocolate"];
        let positives = ["aardvarks!", "beekeepers", "chocolates"];

        let first = tuner.step(&anchors, &positives, 5e-2).unwrap();
        let mut last = first;
        for _ in 0..19 {
            last = tuner.step(&anchors, &positives, 5e-2).unwrap();
        }

        std::fs::remove_dir_all(&dir).ok();
        assert!(last < first, "loss did not drop: first {first} last {last}");
    }

    /// A text shorter than `seq_len` is refused with a message naming both
    /// lengths, never silently padded - see this module's own doc for why
    /// padding a bidirectional encoder is unsound.
    #[test]
    fn a_short_text_is_refused_not_padded() {
        let dir = std::env::temp_dir().join(format!("brain-sdk-encoder-finetune-short-{}", std::process::id()));
        let (weights, tok) = write_fixture(&dir, 16);
        let mut tuner = EncoderFineTuner::open(weights.to_str().unwrap(), tok.to_str().unwrap(), 1, 8).unwrap();

        let err = tuner.step(&["hi"], &["also-short-but-8"], 5e-2).unwrap_err();
        std::fs::remove_dir_all(&dir).ok();
        match err {
            Error::Backend(msg) => assert!(msg.contains("seq_len"), "got: {msg}"),
            other => panic!("expected Error::Backend naming seq_len, got {other:?}"),
        }
    }
}
