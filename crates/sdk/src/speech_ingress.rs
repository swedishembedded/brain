// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speech interfaces to language models whose
// existing behaviour is kept, for its clients. If your team needs expertise
// in teaching a trained language model to listen without retraining it, you
// can procure our services by sending an email to info@swedishembedded.com.

//! [`SpeechIngress`]: teach a language model to listen without changing it.
//!
//! A recognizer's audio encoder turns speech into rows of features
//! ([`crate::AudioFeatures`]); a small projector maps each row into the language
//! model's embedding space; the rows are spliced into the model's residual
//! stream where a user's words would be, and the model answers as it would
//! have answered the words. Only the projector learns. The model, and any
//! adapter on it (a persona's, say), stays exactly as it was: its learning
//! rate is zero, so what it does with text is untouched.
//!
//! Every example has the same shape: a fixed prefix (the chat template up to
//! the user's turn), exactly `rows` audio rows, a fixed suffix (the end of the
//! turn and the start of the reply), and the reply to learn. Clips are padded
//! with silence to the audio window before encoding, so a short clip fills the
//! rows with the encoder's own features of silence, the same at training and at
//! use.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use model::projector::{ProjectorConfig, TrainableProjector};
use qwen3::finetune::{build_trainer, lora_start, LoraStart, Trained};
use qwen3::{Dtype, IGNORE};

use crate::{Error, Result};

/// The LoRA rank given to the (never trained) adapter slot of a decoder that
/// has no persona adapter.
const FALLBACK_RANK: u32 = 8;

/// What to build.
#[derive(Clone, Debug)]
pub struct SpeechIngressOptions {
    /// The language model's checkpoint directory (a brain store path).
    pub base: PathBuf,
    /// An adapter on it that stays on, frozen; `None` for the plain model.
    pub adapter: Option<PathBuf>,
    /// Width of one feature row the projector reads.
    pub input_dim: usize,
    /// Rows of audio in every example.
    pub rows: usize,
    /// Tokens in one training row; the longest example must fit.
    pub block: u32,
    /// Seed of the projector's initial weights.
    pub seed: u64,
    /// Hold the frozen model at bf16 (half the memory of f32).
    pub bf16: bool,
}

/// One training or scoring example, ready for the model.
#[derive(Clone, Debug)]
pub struct SpeechExample {
    features: Vec<f32>,
    tokens: Vec<u32>,
    targets: Vec<u32>,
    row0: u32,
}

impl SpeechExample {
    /// Tokens the model reads (the reply supervised from where it starts).
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the example reads no tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// The optimiser settings of one step.
#[derive(Clone, Copy, Debug)]
pub struct IngressHyper {
    /// The projector's learning rate.
    pub projector_lr: f32,
    /// The model's (or its adapter's): `0.0` keeps it exactly as it is.
    pub model_lr: f32,
    /// Weight decay.
    pub weight_decay: f32,
    /// Global gradient-norm clip; `0` disables.
    pub grad_clip: f32,
}

/// A language model that listens through a trained projector.
pub struct SpeechIngress {
    decoder: Trained,
    projector: TrainableProjector,
    tokenizer: QwenBpe,
    rows: usize,
    input_dim: usize,
    block: u32,
    d_model: usize,
    splice_at: Option<u32>,
}

fn backend(what: impl std::fmt::Display) -> Error {
    Error::Backend(what.to_string())
}

impl SpeechIngress {
    /// Build the model (with its adapter, if any, frozen under a zero learning
    /// rate) and a projector with fresh weights.
    pub fn new(opts: &SpeechIngressOptions) -> Result<SpeechIngress> {
        let base = opts.base.to_str().ok_or_else(|| backend(format!("{} is not UTF-8", opts.base.display())))?;
        let (rank, alpha, start) = match &opts.adapter {
            Some(path) => {
                let path_str = path.to_str().ok_or_else(|| backend(format!("{} is not UTF-8", path.display())))?;
                let card = checkpoint::st::read_card(path_str).map_err(|e| backend(format!("{}: {e}", path.display())))?;
                let adapter = card.and_then(|c| c.adapter).ok_or_else(|| backend(format!("{}: not an adapter file", path.display())))?;
                let rank = adapter.rank.ok_or_else(|| backend(format!("{}: the adapter card names no rank", path.display())))?;
                (rank, adapter.alpha.unwrap_or(2.0 * rank as f32), LoraStart::Continue(path_str))
            }
            None => (FALLBACK_RANK, 2.0 * FALLBACK_RANK as f32, LoraStart::Fresh),
        };
        let (cfg, init) = lora_start(base, rank, alpha, opts.seed, &start).map_err(backend)?;
        let d_model = cfg.d_model as usize;
        let fit = model::FitOpts { batch_size: 1, block_size: opts.block, ..model::FitOpts::default() };
        let dtype = if opts.bf16 { Dtype::BF16 } else { Dtype::F32 };
        let mut decoder = build_trainer(cfg, &fit, &init, dtype).map_err(backend)?;
        decoder.enable_mm_splices(&[(0, opts.rows as u32)]);
        let projector_cfg = ProjectorConfig::from_type("mlp2x_gelu", 2, opts.input_dim as u32, d_model as u32).map_err(backend)?;
        let projector = TrainableProjector::new(projector_cfg, initial_weights(projector_cfg, opts.seed), opts.rows).map_err(backend)?;
        let tokenizer = QwenBpe::from_dir(base).map_err(backend)?;
        Ok(SpeechIngress { decoder, projector, tokenizer, rows: opts.rows, input_dim: opts.input_dim, block: opts.block, d_model, splice_at: None })
    }

    /// The model's chat template up to where the user's audio goes.
    #[must_use]
    pub fn prefix(system: &str) -> String {
        format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n")
    }

    /// The chat template from the end of the audio to where the reply starts:
    /// the user's instruction (empty for none), the end of the turn, and the
    /// assistant's empty reasoning block, which is how a non-thinking model
    /// is trained and asked.
    #[must_use]
    pub fn suffix(instruction: &str) -> String {
        format!("{instruction}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    /// An example: `features` (`rows * input_dim` values) spoken between
    /// `prefix` and `suffix`, to be answered with `reply`.
    pub fn example(&mut self, features: &[f32], prefix: &str, suffix: &str, reply: &str) -> Result<SpeechExample> {
        if features.len() != self.rows * self.input_dim {
            return Err(backend(format!("{} feature values, expected {} rows of {}", features.len(), self.rows, self.input_dim)));
        }
        let before = self.tokenizer.encode(prefix);
        let after = self.tokenizer.encode(suffix);
        let answer = self.tokenizer.encode(&format!("{reply}<|im_end|>"));
        let mut sequence = before.clone();
        sequence.extend(std::iter::repeat_n(0u32, self.rows));
        sequence.extend(&after);
        let supervised_from = sequence.len();
        sequence.extend(&answer);
        let n = sequence.len() - 1;
        if n > self.block as usize {
            return Err(backend(format!("an example of {n} tokens does not fit the {}-token block", self.block)));
        }
        let tokens = sequence[..n].to_vec();
        let targets = (0..n).map(|i| if i + 1 >= supervised_from { sequence[i + 1] } else { IGNORE }).collect();
        Ok(SpeechExample { features: features.to_vec(), tokens, targets, row0: before.len() as u32 })
    }

    fn load(&mut self, ex: &SpeechExample) {
        if self.splice_at != Some(ex.row0) {
            self.decoder.enable_mm_splices(&[(ex.row0, self.rows as u32)]);
            self.splice_at = Some(ex.row0);
        }
        let rows = self.projector.forward(&[ex.features.clone()]);
        self.decoder.write_img_embeds(&rows);
        let (mut x, mut y) = (ex.tokens.clone(), ex.targets.clone());
        x.resize(self.block as usize, 0);
        y.resize(self.block as usize, IGNORE);
        self.decoder.set_batch(&x, &y);
    }

    /// The mean cross-entropy of the reply of `ex` given its audio, with the
    /// current weights, over the reply's tokens.
    pub fn loss(&mut self, ex: &SpeechExample) -> f32 {
        self.load(ex);
        self.decoder.forward()
    }

    fn accumulate(&mut self, ex: &SpeechExample) -> f32 {
        self.load(ex);
        let loss = self.decoder.forward();
        self.decoder.backward();
        let d_rows = self.decoder.read_d_img_embeds();
        // The projector's activations hold the last forward: run it again for
        // this example before its gradient.
        self.projector.forward(&[ex.features.clone()]);
        self.projector.backward(&d_rows);
        loss
    }

    /// One optimiser step (1-based `t`) on the mean gradient of `batch`;
    /// returns the mean loss.
    pub fn step(&mut self, batch: &[&SpeechExample], t: u32, h: &IngressHyper) -> f32 {
        assert!(!batch.is_empty(), "a step needs at least one example");
        self.decoder.zero_grads();
        self.projector.zero_grads();
        let mean = 1.0 / batch.len() as f32;
        let loss = batch.iter().map(|ex| self.accumulate(ex)).sum::<f32>() * mean;
        self.projector.step_scaled(t, h.projector_lr, h.weight_decay, h.grad_clip, mean);
        self.decoder.adamw_step(t, h.model_lr, h.weight_decay, model::Adam::default(), (h.grad_clip > 0.0).then_some(h.grad_clip), mean);
        self.decoder.poll_wait();
        loss
    }

    /// Write the projector to `path` (a safetensors file).
    pub fn save_projector(&self, path: &Path) -> Result<()> {
        let weights = self.projector.weights();
        let mut names: Vec<&String> = weights.keys().collect();
        names.sort();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = names.into_iter().map(|n| (n.clone(), self.projector.shape(n), weights[n].clone())).collect();
        let meta = serde_json::json!({ "input_dim": self.input_dim, "rows": self.rows, "d_model": self.d_model });
        let path_str = path.to_str().ok_or_else(|| backend(format!("{} is not UTF-8", path.display())))?;
        checkpoint::st::save_safetensors(path_str, &tensors, &meta, None).map_err(|e| backend(format!("{}: {e}", path.display())))
    }

    /// Replace the projector's weights with those in `path`.
    pub fn load_projector(&mut self, path: &Path) -> Result<()> {
        let path_str = path.to_str().ok_or_else(|| backend(format!("{} is not UTF-8", path.display())))?;
        let st = checkpoint::st::load_safetensors(path_str).map_err(|e| backend(format!("{}: {e}", path.display())))?;
        let weights: HashMap<String, Vec<f32>> = st.tensors.into_iter().collect();
        self.projector.set_weights(&weights);
        Ok(())
    }

    /// Rows of audio in every example.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// A projector's initial weights: uniform in `+-1/sqrt(fan_in)`, biases zero,
/// from a splitmix64 stream seeded by `seed`.
fn initial_weights(cfg: ProjectorConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    };
    cfg.param_list()
        .into_iter()
        .map(|(name, len)| {
            let weights = if name.ends_with(".bias") {
                vec![0.0; len]
            } else {
                let fan_in = if name.starts_with("in") { cfg.input_dim } else { cfg.n_embed } as f32;
                let bound = 1.0 / fan_in.sqrt();
                (0..len).map(|_| (next() * 2.0 - 1.0) * bound).collect()
            };
            (name, weights)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_weights_are_bounded_by_fan_in_and_deterministic() {
        let cfg = ProjectorConfig::from_type("mlp2x_gelu", 2, 16, 8).unwrap();
        let a = initial_weights(cfg, 3);
        assert_eq!(a, initial_weights(cfg, 3));
        assert_ne!(a, initial_weights(cfg, 4));
        assert!(a["in.weight"].iter().all(|w| w.abs() <= 0.25), "bound is 1/sqrt(16)");
        assert!(a["layers.1.weight"].iter().all(|w| w.abs() <= 1.0 / 8f32.sqrt()));
        assert!(a["in.bias"].iter().all(|b| *b == 0.0));
        assert_eq!(a["in.weight"].len(), 8 * 16);
    }

    #[test]
    fn the_chat_template_pieces_frame_a_turn_and_close_the_empty_reasoning_block() {
        assert_eq!(SpeechIngress::prefix("You are X."), "<|im_start|>system\nYou are X.<|im_end|>\n<|im_start|>user\n");
        assert_eq!(SpeechIngress::suffix(""), "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert!(SpeechIngress::suffix("Transcribe.").starts_with("Transcribe.<|im_end|>"));
    }
}
