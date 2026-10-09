// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`TtsPolicy`]: the synthesizer as a policy that can be trained from scores
//! of its own takes.
//!
//! A caller draws a group of takes of a sentence ([`TtsPolicy::rollout`]),
//! scores them however it likes (a listener, a judge, word error rate) and hands
//! back each take with its advantage ([`TtsPolicy::update`]); the policy moves
//! toward the takes that scored above their group, held near the original
//! voice by a KL term to a frozen copy. The first codec codebook is the policy;
//! the code predictor and the codec stay frozen. Qwen3-TTS on a device only.

use std::path::PathBuf;

use capability::CancelToken;

use super::{Audio, Backend, TtsOptions, TtsPipeline, SAMPLE_RATE};
use crate::{Error, Result};

/// One sampled take: the audio, and what it was generated from.
pub struct Take {
    rollout: qwen3tts::Rollout,
    audio: Audio,
}

impl Take {
    /// The take as audio.
    #[must_use]
    pub fn audio(&self) -> &Audio {
        &self.audio
    }

    /// Codec frames in the take.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.rollout.codes.len() / 16
    }
}

/// How a policy is built.
#[derive(Clone, Copy, Debug)]
pub struct PolicyOptions {
    /// LoRA rank.
    pub rank: u32,
    /// LoRA alpha.
    pub alpha: f32,
    /// The most rows (prompt plus frames) of one take that can be trained on.
    pub block: u32,
    /// Seed of the adapter's initial values.
    pub seed: u64,
}

impl Default for PolicyOptions {
    fn default() -> Self {
        Self { rank: 8, alpha: 16.0, block: 320, seed: 1 }
    }
}

/// One policy-gradient step.
#[derive(Clone, Copy, Debug)]
pub struct PolicyUpdate {
    /// Learning rate.
    pub lr: f32,
    /// Weight of the KL term to the original policy.
    pub kl_beta: f32,
    /// Global gradient-norm clip.
    pub clip: f32,
}

impl Default for PolicyUpdate {
    fn default() -> Self {
        Self { lr: 1e-4, kl_beta: 0.02, clip: 1.0 }
    }
}

/// What an update saw, before the step.
#[derive(Clone, Copy, Debug)]
pub struct PolicyStats {
    /// The objective.
    pub loss: f32,
    /// Mean log-probability of the sampled tokens.
    pub mean_logprob: f32,
    /// Mean k3 KL estimate to the original policy.
    pub mean_kl: f32,
    /// How many sampled tokens the step was over.
    pub tokens: usize,
}

/// The trainable synthesizer. See the module documentation.
pub struct TtsPolicy {
    engine: qwen3tts::engine::ResidentEngine,
    reference: qwen3tts::engine::ResidentEngine,
    trainer: qwen3tts::train::TalkerTrainer,
}

impl std::fmt::Debug for TtsPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsPolicy").field("arch", &"qwen3tts").finish()
    }
}

fn backend(e: impl std::fmt::Display) -> Error {
    Error::Backend(e.to_string())
}

impl TtsPipeline {
    /// The trainable form of this pipeline. Qwen3-TTS on a device only.
    pub fn policy(&self, options: PolicyOptions) -> Result<TtsPolicy> {
        let Backend::Qwen3Tts(paths) = &self.backend else {
            return Err(Error::MissingArgument("cosyvoice: has no trainable form".to_string()));
        };
        let device = qwen3tts::engine::Placement::Device;
        let engine = qwen3tts::engine::ResidentEngine::load_with(paths, device).map_err(backend)?;
        let reference = qwen3tts::engine::ResidentEngine::load_with(paths, device).map_err(backend)?;
        let trainer = qwen3tts::train::TalkerTrainer::load(&paths.talker, options.block, options.rank, options.alpha, options.seed).map_err(backend)?;
        Ok(TtsPolicy { engine, reference, trainer })
    }
}

impl TtsPolicy {
    /// Sample one take of `text` under the current policy. `opts` carries the
    /// seed, the sampling knobs and an optional pinned voice.
    pub fn rollout(&mut self, text: &str, opts: &TtsOptions) -> Result<Take> {
        let gen_opts = opts.to_gen_opts();
        let voice = match &opts.voice {
            Some((reference, words)) => Some((reference.to_str().ok_or_else(|| backend(format!("{} is not UTF-8", reference.display())))?.to_string(), words.clone())),
            None => None,
        };
        let never = CancelToken::default();
        let rollout = self
            .engine
            .rollout(text, &opts.lang, voice.as_ref().map(|(r, w)| (r.as_str(), w.as_str())), &gen_opts, &never)
            .map_err(backend)?;
        let samples = self.engine.decode(&rollout.codes, &mut |_, _| {}).map_err(backend)?;
        Ok(Take { rollout, audio: Audio::new(samples, SAMPLE_RATE) })
    }

    /// Take one step over `group`: each take with the advantage it earned (its
    /// score less its group's mean, over the group's spread). The updated policy
    /// is what the next [`Self::rollout`] draws from.
    pub fn update(&mut self, group: &[(&Take, f32)], opts: &PolicyUpdate) -> Result<PolicyStats> {
        let mut rows = Vec::with_capacity(group.len());
        let mut references = Vec::with_capacity(group.len());
        for (take, _) in group {
            let r = self.reference.policy_rows(&take.rollout).map_err(backend)?;
            if r.len() > self.trainer_block() {
                return Err(backend(format!("a take of {} rows does not fit the trainer's block", r.len())));
            }
            let logits = self.reference.score(&r).map_err(backend)?;
            references.push(qwen3tts::train::logprobs(&logits, &r));
            rows.push(r);
        }
        let samples: Vec<qwen3tts::train::Sample<'_>> = group
            .iter()
            .zip(&rows)
            .zip(references)
            .map(|(((_, advantage), rows), reference_logprobs)| qwen3tts::train::Sample { rows, advantage: *advantage, reference_logprobs })
            .collect();
        let stats = self.trainer.update(&samples, &qwen3tts::train::UpdateOpts { lr: opts.lr, kl_beta: opts.kl_beta, clip: Some(opts.clip), weight_decay: 0.0 });
        self.engine.set_talker_weights(&self.trainer.folded_weights()).map_err(backend)?;
        Ok(PolicyStats { loss: stats.loss, mean_logprob: stats.mean_logprob, mean_kl: stats.mean_kl, tokens: stats.tokens })
    }

    fn trainer_block(&self) -> usize {
        self.trainer.block()
    }

    /// Write the trained adapter to `path`.
    pub fn save(&self, path: impl Into<PathBuf>) -> Result<()> {
        self.trainer.save_adapter(&path.into()).map_err(backend)
    }
}
