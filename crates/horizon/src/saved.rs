// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A trained model as it is kept on disk: a directory holding the weights
//! (with the configuration in their header), the fitted vocabulary and, when
//! they were made, the [`Calibration`]. The one place that reads, writes and
//! predicts from that directory - the SDK's `TimelineModel` and the serving
//! capability ([`crate::caps`]) both go through it.
//!
//! The calibration is optional and a directory without it loads as
//! uncalibrated. When present it must have been fitted for exactly these
//! weights (it records their SHA-256): loading refuses a mismatch.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::calibration::{self, Calibration};
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
    /// Its calibration, if one was fitted ([`Saved::calibrate`]).
    pub calibration: Option<Arc<Calibration>>,
}

/// The SHA-256 of a file, as lowercase hex.
fn file_digest(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// `path`'s text, or `None` when the file does not exist.
fn read_optional(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(t) => Ok(Some(t)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

fn utf8(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("{}: not a UTF-8 path", path.display()))
}

impl Saved {
    /// A model and vocabulary with no calibration.
    pub fn new(model: Horizon, vocab: Vocab) -> Saved {
        Saved {
            model,
            vocab,
            calibration: None,
        }
    }

    /// The SHA-256 (hex) of the weights file [`Saved::save`] writes for this
    /// model: what a calibration is bound to. The model lives on the device,
    /// so its weights are written to a scratch file to be hashed.
    pub fn weights_digest(&self) -> Result<String, String> {
        static SCRATCH: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "horizon-digest-{}-{}.safetensors",
            std::process::id(),
            SCRATCH.fetch_add(1, Ordering::Relaxed)
        ));
        self.model.save(utf8(&path)?);
        let digest = file_digest(&path);
        std::fs::remove_file(&path).ok();
        digest
    }

    /// Fit a calibration on `validation` (never test) for every outcome code
    /// at every horizon of `horizons`, replacing any earlier one. See
    /// [`Calibration::fit`].
    pub fn calibrate(
        &mut self,
        validation: &[Subject],
        horizons: &[f64],
        min_events: usize,
    ) -> Result<(), String> {
        let digest = self.weights_digest()?;
        self.calibration = Some(Arc::new(Calibration::fit(
            self, digest, validation, horizons, min_events,
        )?));
        Ok(())
    }

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
        let calibration = read_optional(&dir.join(calibration::FILE))?
            .map(|text| Calibration::from_json(&text).map(Arc::new))
            .transpose()?;
        if let Some(c) = &calibration {
            let have = file_digest(&weights)?;
            if c.weights_sha256 != have {
                return Err(format!(
                    "{}: {} was fitted for weights {} but {WEIGHTS_FILE} is {have}: refusing to serve \
                     probabilities calibrated for a different model (calibrate again or remove the file)",
                    dir.display(),
                    calibration::FILE,
                    c.weights_sha256
                ));
            }
        }
        let model = Horizon::load(utf8(&weights)?, PREDICT_BATCH)?;
        Ok(Saved {
            model,
            vocab,
            calibration,
        })
    }

    /// Write the weights, the vocabulary and the calibration (if any) into
    /// `dir`, creating it; a directory this replaces loses a calibration
    /// this model does not have. A calibration of different weights is an
    /// error, not a file.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let weights = dir.join(WEIGHTS_FILE);
        self.model.save(utf8(&weights)?);
        let vocab = serde_json::to_string(&self.vocab).map_err(|e| format!("vocab: {e}"))?;
        let path = dir.join(VOCAB_FILE);
        std::fs::write(&path, vocab).map_err(|e| format!("{}: {e}", path.display()))?;
        let calibration = dir.join(calibration::FILE);
        match &self.calibration {
            Some(c) => {
                let have = file_digest(&weights)?;
                if c.weights_sha256 != have {
                    return Err(format!(
                        "{}: the calibration was fitted for weights {} but these are {have}",
                        dir.display(),
                        c.weights_sha256
                    ));
                }
                std::fs::write(&calibration, c.to_json()?)
                    .map_err(|e| format!("{}: {e}", calibration.display()))
            }
            None => match std::fs::remove_file(&calibration) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    Err(format!("{}: {e}", calibration.display()))
                }
                _ => Ok(()),
            },
        }
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
