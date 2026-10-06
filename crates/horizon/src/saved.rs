// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A trained model as it is kept on disk: a directory holding the weights
//! (with the configuration in their header), the fitted vocabulary and, when
//! they were made, the [`Calibration`] and the training [`Support`]. The one place that reads, writes and
//! predicts from that directory - the SDK's `TimelineModel` and the serving
//! capability ([`crate::caps`]) both go through it.
//!
//! The calibration is optional and a directory without it loads as
//! uncalibrated, and one without the support loads with support unknown.
//! When present, each must have been fitted for exactly these weights (they
//! record their SHA-256): loading refuses a mismatch.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::calibration::{self, Calibration};
use crate::encode::{encode, Encoded};
use crate::support::{self, AssessOptions, Assessment, Support};
use crate::survival::Curves;
use crate::timeline::Subject;
use crate::train::{predict_hazards_and_states, predict_log_hazards, predict_states};
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
    /// What it was trained on, if recorded ([`Saved::fit_support`]); `None`
    /// for a model saved before support was kept.
    pub support: Option<Support>,
    /// The weights digest, computed once: the weights of a `Saved` are fixed
    /// (replace the whole value to change them).
    digest: OnceLock<String>,
}

/// Which model produced an answer: enough to tell, long after, whether two
/// answers came from the same weights, configuration and code.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelIdentity {
    /// SHA-256 (hex) of the weights file [`Saved::save`] writes.
    pub weights_sha256: String,
    /// SHA-256 (hex) of the model configuration's JSON.
    pub config_sha256: String,
    /// The brain version that produced the answer.
    pub brain_version: String,
}

/// One subject's curves and standing against the training support, from one
/// forward pass.
pub struct Scored {
    /// The outcome curves.
    pub curves: Curves,
    /// Whether the subject is inside what the model was trained on.
    pub assessment: Assessment,
}

/// The SHA-256 of a file, as lowercase hex.
fn file_digest(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(hex(&Sha256::digest(&bytes)))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
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
            support: None,
            digest: OnceLock::new(),
        }
    }

    /// Which model this is: its weights digest ([`Saved::weights_digest`]),
    /// the digest of its configuration and this build's version.
    pub fn identity(&self) -> Result<ModelIdentity, String> {
        let config = serde_json::to_vec(&self.model.cfg).map_err(|e| format!("config: {e}"))?;
        Ok(ModelIdentity {
            weights_sha256: self.weights_digest()?,
            config_sha256: hex(&Sha256::digest(&config)),
            brain_version: env!("CARGO_PKG_VERSION").to_string(),
        })
    }

    /// The SHA-256 (hex) of the weights file [`Saved::save`] writes for this
    /// model: what a calibration is bound to. The model lives on the device,
    /// so its weights are written to a scratch file to be hashed.
    pub fn weights_digest(&self) -> Result<String, String> {
        if let Some(d) = self.digest.get() {
            return Ok(d.clone());
        }
        static SCRATCH: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "horizon-digest-{}-{}.safetensors",
            std::process::id(),
            SCRATCH.fetch_add(1, Ordering::Relaxed)
        ));
        self.model.save(utf8(&path)?);
        let digest = file_digest(&path);
        std::fs::remove_file(&path).ok();
        let digest = digest?;
        Ok(self.digest.get_or_init(|| digest).clone())
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

    /// Record what the model was trained on: the support of `train`, the
    /// training subjects, which must be the ones the weights were fitted on.
    pub fn fit_support(&mut self, train: &[Subject]) -> Result<(), String> {
        let digest = self.weights_digest()?;
        self.support = Some(Support::fit(self, digest, train)?);
        Ok(())
    }

    /// Each subject against the training support. With no support recorded
    /// every assessment is [`Assessment::unknown`].
    pub fn assess(&self, subjects: &[Subject], opts: &AssessOptions) -> Result<Vec<Assessment>, String> {
        let enc = self.encode(subjects)?;
        let Some(support) = &self.support else {
            return Ok(subjects
                .iter()
                .map(|s| self.advised(Assessment::unknown(), s))
                .collect());
        };
        let states = predict_states(&self.model, &enc);
        Ok(subjects
            .iter()
            .zip(&states)
            .map(|(s, z)| self.advised(support.assess(s, Some(z), opts), s))
            .collect())
    }

    /// One set of outcome curves and one assessment per subject, in order,
    /// from a single forward pass.
    pub fn score(&self, subjects: &[Subject], opts: &AssessOptions) -> Result<Vec<Scored>, String> {
        let enc = self.encode(subjects)?;
        let absorbing = self.absorbing();
        let (knots, columns) = (&self.model.cfg.knots, self.model.cfg.n_codes as usize);
        Ok(subjects
            .iter()
            .zip(predict_hazards_and_states(&self.model, &enc))
            .map(|(s, (lh, z))| Scored {
                curves: Curves::outcomes(&lh, knots, columns, &absorbing),
                assessment: self.advised(
                    self.support
                        .as_ref()
                        .map_or_else(Assessment::unknown, |sp| sp.assess(s, Some(&z), opts)),
                    s,
                ),
            })
            .collect())
    }

    /// `assessment` with the input advisories of `subject` that depend on the
    /// vocabulary rather than on the training support.
    fn advised(&self, mut assessment: Assessment, subject: &Subject) -> Assessment {
        assessment.advisories = support::unit_advisories(&self.vocab, subject);
        assessment
    }

    fn absorbing(&self) -> Vec<bool> {
        self.vocab
            .codes
            .iter()
            .map(|c| self.vocab.absorbing.contains(c))
            .collect()
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
        let support = read_optional(&dir.join(support::FILE))?
            .map(|text| Support::from_json(&text))
            .transpose()?;
        if calibration.is_some() || support.is_some() {
            let have = file_digest(&weights)?;
            let bound = [
                (calibration::FILE, calibration.as_ref().map(|c| &c.weights_sha256)),
                (support::FILE, support.as_ref().map(|s| &s.weights_sha256)),
            ];
            for (file, digest) in bound {
                if let Some(d) = digest.filter(|d| **d != have) {
                    return Err(format!(
                        "{}: {file} was fitted for weights {d} but {WEIGHTS_FILE} is {have}: refusing to \
                         serve a calibration or support recorded for a different model (refit it or \
                         remove the file)",
                        dir.display()
                    ));
                }
            }
        }
        let model = Horizon::load(utf8(&weights)?, PREDICT_BATCH)?;
        Ok(Saved {
            model,
            vocab,
            calibration,
            support,
            digest: OnceLock::new(),
        })
    }

    /// Write the weights, the vocabulary and the calibration and support (if
    /// any) into `dir`, creating it; a directory this replaces loses a
    /// calibration or support this model does not have. A record of
    /// different weights is an error, not a file.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let weights = dir.join(WEIGHTS_FILE);
        self.model.save(utf8(&weights)?);
        let vocab = serde_json::to_string(&self.vocab).map_err(|e| format!("vocab: {e}"))?;
        let path = dir.join(VOCAB_FILE);
        std::fs::write(&path, vocab).map_err(|e| format!("{}: {e}", path.display()))?;
        let have = file_digest(&weights)?;
        let optional = [
            (calibration::FILE, self.calibration.as_ref().map(|c| (&c.weights_sha256, c.to_json()))),
            (support::FILE, self.support.as_ref().map(|s| (&s.weights_sha256, s.to_json()))),
        ];
        for (file, content) in optional {
            let path = dir.join(file);
            match content {
                Some((digest, json)) => {
                    if *digest != have {
                        return Err(format!(
                            "{}: {file} was fitted for weights {digest} but these are {have}",
                            dir.display()
                        ));
                    }
                    std::fs::write(&path, json?).map_err(|e| format!("{}: {e}", path.display()))?;
                }
                // This model has none: a stale one from a replaced directory
                // would be a record of a different model.
                None => match std::fs::remove_file(&path) {
                    Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                        return Err(format!("{}: {e}", path.display()))
                    }
                    _ => {}
                },
            }
        }
        Ok(())
    }

    /// Each subject validated and encoded for this model.
    pub fn encode(&self, subjects: &[Subject]) -> Result<Vec<Encoded>, String> {
        subjects
            .iter()
            .map(|s| {
                s.validate()?;
                self.vocab.check_units(s)?;
                Ok(encode(s, &self.vocab, &self.model.cfg))
            })
            .collect()
    }

    /// One set of outcome curves per subject, in order.
    pub fn predict(&self, subjects: &[Subject]) -> Result<Vec<Curves>, String> {
        let enc = self.encode(subjects)?;
        let absorbing = self.absorbing();
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
