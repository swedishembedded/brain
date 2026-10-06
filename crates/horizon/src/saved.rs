// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A trained model as it is kept on disk: a directory holding the weights
//! (with the configuration in their header) and the fitted vocabulary. The
//! one place that reads, writes and predicts from that directory - the SDK's
//! `TimelineModel` and the serving capability ([`crate::caps`]) both go
//! through it.

use std::path::Path;

use crate::encode::{encode, Encoded};
use crate::survival::Curves;
use crate::timeline::Subject;
use crate::train::predict_log_hazards;
use crate::vocab::Vocab;
use crate::Horizon;

/// The weights file inside a saved model's directory.
pub const WEIGHTS_FILE: &str = "model.safetensors";
/// The vocabulary file inside a saved model's directory.
pub const VOCAB_FILE: &str = "vocab.json";
/// Subjects per device batch when a saved model is loaded to predict.
pub const PREDICT_BATCH: u32 = 256;

/// A model with the vocabulary it was trained under.
pub struct Saved {
    /// The model.
    pub model: Horizon,
    /// Its vocabulary.
    pub vocab: Vocab,
}

fn utf8(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("{}: not a UTF-8 path", path.display()))
}

impl Saved {
    /// Load the model [`Saved::save`] wrote into `dir`.
    pub fn load(dir: &Path) -> Result<Saved, String> {
        let weights = dir.join(WEIGHTS_FILE);
        if !weights.is_file() {
            return Err(format!(
                "{}: no saved timeline model (missing {WEIGHTS_FILE})",
                dir.display()
            ));
        }
        let vocab_path = dir.join(VOCAB_FILE);
        let text = std::fs::read_to_string(&vocab_path)
            .map_err(|e| format!("{}: {e}", vocab_path.display()))?;
        let vocab = Vocab::from_json(&text)?;
        let model = Horizon::load(utf8(&weights)?, PREDICT_BATCH)?;
        Ok(Saved { model, vocab })
    }

    /// Write the weights and the vocabulary into `dir`, creating it.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        self.model.save(utf8(&dir.join(WEIGHTS_FILE))?);
        let vocab = serde_json::to_string(&self.vocab).map_err(|e| format!("vocab: {e}"))?;
        let path = dir.join(VOCAB_FILE);
        std::fs::write(&path, vocab).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Each subject validated and encoded for this model.
    pub fn encode(&self, subjects: &[Subject]) -> Result<Vec<Encoded>, String> {
        subjects
            .iter()
            .map(|s| {
                s.validate()
                    .map(|_| encode(s, &self.vocab, &self.model.cfg))
            })
            .collect()
    }

    /// One set of outcome curves per subject, in order.
    pub fn predict(&self, subjects: &[Subject]) -> Result<Vec<Curves>, String> {
        let enc = self.encode(subjects)?;
        let absorbing: Vec<bool> = self
            .vocab
            .codes
            .iter()
            .map(|c| self.vocab.absorbing.contains(c))
            .collect();
        let (knots, columns) = (&self.model.cfg.knots, self.model.cfg.n_codes as usize);
        Ok(predict_log_hazards(&self.model, &enc)
            .into_iter()
            .map(|lh| Curves::outcomes(&lh, knots, columns, &absorbing))
            .collect())
    }

    /// One set of next-event curves per subject, in order: for each code of
    /// the vocabulary's next-event group, the probability that it is the
    /// first of the group to happen by a time. Empty when the model was
    /// trained without the group.
    pub fn predict_next_events(&self, subjects: &[Subject]) -> Result<Vec<Curves>, String> {
        let group = self.vocab.next_events.len();
        if group == 0 {
            return Ok(Vec::new());
        }
        let enc = self.encode(subjects)?;
        let (knots, outcomes) = (&self.model.cfg.knots, self.vocab.codes.len());
        Ok(predict_log_hazards(&self.model, &enc)
            .into_iter()
            .map(|lh| Curves::first_events(&lh, knots, outcomes, group))
            .collect())
    }

    /// The longest horizon the model predicts to (its last knot).
    pub fn horizon(&self) -> f64 {
        self.model.cfg.knots.last().map_or(0.0, |&k| f64::from(k))
    }
}

/// Subjects from `timeline-v1` text: one per non-blank line, each validated;
/// an error names the line.
pub fn parse_jsonl(text: &str) -> Result<Vec<Subject>, String> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(n, l)| Subject::from_json_line(l).map_err(|e| format!("line {}: {e}", n + 1)))
        .collect()
}
