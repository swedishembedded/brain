// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Ensembles of timeline models: several models trained apart whose mean is
//! the prediction and whose disagreement is the uncertainty about it.
//!
//! Swedish Embedded AB implements risk models that say how sure they are, for
//! its clients. If your team needs expertise in uncertainty for time-to-event
//! predictions you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Two kinds, both first-class trained objects ([`Ensemble::train`]):
//!
//! - [`Kind::Seeded`]: the same training subjects, a different seed per member
//!   (initial weights, batches, masks);
//! - [`Kind::Bootstrap`]: each member trains on the subjects resampled with
//!   replacement BY GROUP (a subject without a `group_id` is its own group), so
//!   households or sites stay together, and the member's seed differs too.
//!
//! The vocabulary is fitted once on the full training set and shared, so the
//! members predict over the same codes, variables and knots. Early stopping
//! uses the same held-out subjects for every member.
//!
//! On disk an ensemble is one directory: `ensemble.json` (the kind, and per
//! member its seed, its bootstrap draws and the SHA-256 of its weights) and
//! `members/0`, `members/1`, ... each a model directory [`Saved::save`] writes.
//! Loading verifies every digest and that the members share a vocabulary.
//! Dropout is not part of horizon's architecture, so there is no Monte-Carlo
//! dropout and none is offered.

use std::path::Path;

use data::rng::Rng;
use serde::{Deserialize, Serialize};

use crate::fit::{fit_vocab, train_with_vocab, Hooks, Report, StepProgress, TrainError, TrainSpec};
use crate::forecast::{forecast, ForecastRequest, RiskForecast};
use crate::history::PatientHistory;
use crate::saved::{file_digest, write_dir_atomically, Saved, Scored, WEIGHTS_FILE};
use crate::support::AssessOptions;
use crate::timeline::Subject;

/// The manifest file at the top of an ensemble directory.
pub const MANIFEST_FILE: &str = "ensemble.json";
/// The directory holding the members' model directories.
pub const MEMBERS_DIR: &str = "members";
const FORMAT: &str = "horizon-ensemble/1";

/// How an ensemble's members differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Same subjects, a different seed per member.
    Seeded,
    /// Subjects resampled with replacement by group, and a different seed.
    Bootstrap,
}

impl Kind {
    /// The name on the command line and in the manifest.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Seeded => "seeded",
            Kind::Bootstrap => "bootstrap",
        }
    }

    /// The kind called `name`.
    pub fn parse(name: &str) -> Option<Kind> {
        match name {
            "seeded" => Some(Kind::Seeded),
            "bootstrap" => Some(Kind::Bootstrap),
            _ => None,
        }
    }
}

/// What the manifest records of one member.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberRecord {
    /// The member's directory below the ensemble's, `members/<index>`.
    pub dir: String,
    /// The seed the member was trained with.
    pub seed: u64,
    /// For a bootstrap member: the group drawn at each of the resample's
    /// positions (an index into the training set's groups in order of first
    /// appearance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_draws: Option<Vec<u32>>,
    /// SHA-256 (hex) of the member's weights file.
    pub weights_sha256: String,
}

/// `ensemble.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// The format tag.
    pub format: String,
    /// How the members differ.
    pub kind: Kind,
    /// Groups in the training set a bootstrap drew from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub groups: Option<u32>,
    /// The members, in order.
    pub members: Vec<MemberRecord>,
}

/// Several models trained apart, with the record of how.
pub struct Ensemble {
    manifest: Manifest,
    members: Vec<Saved>,
}

/// The groups of `subjects` in order of first appearance, each the indices of
/// its subjects. A subject without a group is a group of one.
fn groups_of(subjects: &[Subject]) -> Vec<Vec<usize>> {
    let mut index: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, s) in subjects.iter().enumerate() {
        match s.group_id.as_deref() {
            Some(g) => {
                let slot = *index.entry(g).or_insert_with(|| {
                    groups.push(Vec::new());
                    groups.len() - 1
                });
                groups[slot].push(i);
            }
            None => groups.push(vec![i]),
        }
    }
    groups
}

/// `groups` drawn with replacement, as many as there are, from `seed`.
fn draw_groups(groups: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..groups).map(|_| (rng.next_u64() % groups as u64) as u32).collect()
}

impl Ensemble {
    /// Train `members` models (at least two) on `train`, early-stopping each on
    /// `held_out`. Member `i` uses the seed `spec`'s seed plus `i`, so the
    /// first member of a seeded ensemble is the model the spec alone would
    /// train. `progress` hears of every member's evaluation intervals
    /// (member index first); `cancelled` is polled after every optimiser
    /// step, and a cancelled run returns [`TrainError::Cancelled`] with no
    /// ensemble.
    pub fn train(
        train: &[Subject],
        held_out: &[Subject],
        spec: &TrainSpec,
        members: usize,
        kind: Kind,
        progress: &mut dyn FnMut(usize, &StepProgress),
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(Ensemble, Vec<Report>), TrainError> {
        if members < 2 {
            return Err(TrainError::Failed(format!(
                "an ensemble needs at least 2 members, got {members}"
            )));
        }
        if train.is_empty() || held_out.is_empty() {
            return Err(TrainError::Failed("training and held-out subjects are both required".into()));
        }
        let vocab = fit_vocab(train, spec)?;
        let groups = groups_of(train);
        let mut trained = Vec::with_capacity(members);
        let mut records = Vec::with_capacity(members);
        let mut reports = Vec::with_capacity(members);
        for i in 0..members {
            let seed = spec.run_seed().wrapping_add(i as u64);
            let member_spec = spec.clone().seed(seed);
            let (draws, resampled);
            let subjects: &[Subject] = match kind {
                Kind::Seeded => {
                    draws = None;
                    train
                }
                Kind::Bootstrap => {
                    let picks = draw_groups(groups.len(), seed ^ 0xB007_57A9_0000_0001);
                    resampled = picks
                        .iter()
                        .flat_map(|&g| groups[g as usize].iter().map(|&s| train[s].clone()))
                        .collect::<Vec<_>>();
                    draws = Some(picks);
                    &resampled
                }
            };
            let mut report_progress = |p: &StepProgress| progress(i, p);
            let mut hooks = Hooks { progress: Some(&mut report_progress), cancelled: Some(cancelled) };
            let (saved, report) =
                train_with_vocab(subjects, held_out, &member_spec, vocab.clone(), &mut hooks)?;
            records.push(MemberRecord {
                dir: format!("{MEMBERS_DIR}/{i}"),
                seed,
                bootstrap_draws: draws,
                weights_sha256: saved.weights_digest()?,
            });
            trained.push(saved);
            reports.push(report);
        }
        let manifest = Manifest {
            format: FORMAT.into(),
            kind,
            groups: (kind == Kind::Bootstrap).then_some(groups.len() as u32),
            members: records,
        };
        Ok((Ensemble { manifest, members: trained }, reports))
    }

    /// What the manifest records.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Whether `dir` holds an ensemble (rather than one model).
    pub fn is_dir(dir: &Path) -> bool {
        dir.join(MANIFEST_FILE).is_file()
    }

    /// Write the ensemble into a new directory `dir`, atomically (see
    /// [`write_dir_atomically`]; an existing `dir` is refused unless
    /// `replace`).
    pub fn save(&self, dir: &Path, replace: bool) -> Result<(), String> {
        if replace && dir.exists() && !Ensemble::is_dir(dir) {
            return Err(format!("{}: exists and is not an ensemble: not replaced", dir.display()));
        }
        write_dir_atomically(dir, replace, |staging| {
            for (record, member) in self.manifest.members.iter().zip(&self.members) {
                member.save(&staging.join(&record.dir))?;
            }
            let json = serde_json::to_string_pretty(&self.manifest).map_err(|e| format!("{MANIFEST_FILE}: {e}"))?;
            let path = staging.join(MANIFEST_FILE);
            std::fs::write(&path, json).map_err(|e| format!("{}: {e}", path.display()))
        })
    }

    /// Load the ensemble [`Ensemble::save`] wrote. Refuses a manifest of
    /// another format, a member directory other than `members/<index>`, a
    /// member whose weights do not match the recorded digest and members that
    /// do not share a vocabulary.
    pub fn load(dir: &Path) -> Result<Ensemble, String> {
        let path = dir.join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let manifest: Manifest = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        if manifest.format != FORMAT {
            return Err(format!("{}: format {:?}, expected {FORMAT:?}", path.display(), manifest.format));
        }
        if manifest.members.len() < 2 {
            return Err(format!("{}: an ensemble has at least 2 members", path.display()));
        }
        let mut members = Vec::with_capacity(manifest.members.len());
        for (i, record) in manifest.members.iter().enumerate() {
            // The directory is derived, not trusted: a manifest cannot point
            // the loader outside the ensemble.
            if record.dir != format!("{MEMBERS_DIR}/{i}") {
                return Err(format!("{}: member {i} is at {:?}, expected \"{MEMBERS_DIR}/{i}\"", path.display(), record.dir));
            }
            let member_dir = dir.join(&record.dir);
            let have = file_digest(&member_dir.join(WEIGHTS_FILE))?;
            if have != record.weights_sha256 {
                return Err(format!(
                    "{}: member {i} weights are {have} but the manifest records {}: refusing a member that is not the one that was trained",
                    dir.display(),
                    record.weights_sha256
                ));
            }
            members.push(Saved::load(&member_dir)?);
        }
        let shared = serde_json::to_string(&members[0].vocab).map_err(|e| e.to_string())?;
        for (i, m) in members.iter().enumerate().skip(1) {
            if serde_json::to_string(&m.vocab).map_err(|e| e.to_string())? != shared {
                return Err(format!("{}: member {i} has a different vocabulary than member 0", dir.display()));
            }
        }
        Ok(Ensemble { manifest, members })
    }

    /// Each member's scores of `subjects`: one list per member, in order.
    pub fn score(&self, subjects: &[Subject], opts: &AssessOptions) -> Result<Vec<Vec<Scored>>, String> {
        self.members.iter().map(|m| m.score(subjects, opts)).collect()
    }

    /// The forecast of each history: the members' forecasts combined by
    /// [`RiskForecast::ensemble`] (the mean, the member range, abstention if
    /// any member abstains).
    pub fn forecast(&self, histories: &[PatientHistory], req: &ForecastRequest) -> Result<Vec<RiskForecast>, String> {
        let per_member: Vec<Vec<RiskForecast>> = self
            .members
            .iter()
            .map(|m| forecast(m, histories, req))
            .collect::<Result<_, _>>()?;
        (0..histories.len())
            .map(|h| {
                let parts: Vec<RiskForecast> = per_member.iter().map(|f| f[h].clone()).collect();
                RiskForecast::ensemble(&parts)
            })
            .collect()
    }
}

/// Something that answers with the mean of one or more models: a [`Saved`], an
/// [`Ensemble`] or a [`Loaded`] directory.
pub trait Members {
    /// The models behind the predictions: one, or every member.
    fn members(&self) -> &[Saved];
}

impl Members for Saved {
    fn members(&self) -> &[Saved] {
        std::slice::from_ref(self)
    }
}

impl Members for Ensemble {
    fn members(&self) -> &[Saved] {
        &self.members
    }
}

impl Members for Loaded {
    fn members(&self) -> &[Saved] {
        match self {
            Loaded::Single(s) => std::slice::from_ref(s),
            Loaded::Ensemble(e) => &e.members,
        }
    }
}

/// What a model directory holds: one model or an ensemble.
#[allow(clippy::large_enum_variant)] // one value per served directory; boxing buys nothing
pub enum Loaded {
    /// One model.
    Single(Saved),
    /// Several, whose mean is the prediction.
    Ensemble(Ensemble),
}

impl Loaded {
    /// Load the model or ensemble in `dir`.
    pub fn load(dir: &Path) -> Result<Loaded, String> {
        if Ensemble::is_dir(dir) {
            Ensemble::load(dir).map(Loaded::Ensemble)
        } else {
            Saved::load(dir).map(Loaded::Single)
        }
    }

    /// Whether `dir` holds something [`Loaded::load`] can read.
    pub fn is_dir(dir: &Path) -> bool {
        Ensemble::is_dir(dir) || dir.join(WEIGHTS_FILE).is_file()
    }

    /// The first model: the one whose vocabulary, knots and horizon every
    /// member shares.
    pub fn lead(&self) -> &Saved {
        &Members::members(self)[0]
    }

    /// The forecast of each history (see [`Ensemble::forecast`]).
    pub fn forecast(&self, histories: &[PatientHistory], req: &ForecastRequest) -> Result<Vec<RiskForecast>, String> {
        match self {
            Loaded::Single(s) => forecast(s, histories, req),
            Loaded::Ensemble(e) => e.forecast(histories, req),
        }
    }
}
