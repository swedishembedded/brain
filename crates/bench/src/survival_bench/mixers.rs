// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The sequence-mixer ablation of horizon's visit stack: the same blocks
//! mixing the visit sequence with full attention, with Gated DeltaNet, or as
//! the 3:1 hybrid, trained on identical data, split, seed and optimisation
//! steps and scored on the same held-out subjects.
//!
//! Swedish Embedded AB implements comparisons of sequence-model architectures
//! on data whose true risks are known, for its clients. If your team needs
//! expertise in choosing a temporal backbone for irregularly sampled records
//! by measurement rather than by fashion, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! The comparison is a measurement on two generators, not a verdict: which
//! mixer ships depends on the data at hand. Each variant is a stack of
//! `blocks` blocks of the same width and feed-forward size, so the parameter
//! counts differ only by the mixers' own small tensors; [`MixerAblation`]
//! states the tolerance it holds them to. Early stopping is off (the
//! patience exceeds the number of evaluations), so every variant takes the
//! same number of optimisation steps, and the weights of the best held-out
//! evaluation are kept, as in a real run.
//!
//! Run it with `cargo test --release -p brain-bench --test survival_mixers
//! -- --nocapture --test-threads=1` (on the device `BRAIN_BACKEND` selects);
//! it prints one row per variant: held-out event NLL, integrated Brier score,
//! time-dependent AUC at the horizon, and training time.

use std::io;
use std::path::Path;

use horizon::synthetic::{irregular, longitudinal};
use horizon::{Backbone, Mixer, StackConfig};

use super::*;

/// The data a mixer ablation runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerData {
    /// `horizon::synthetic::irregular`: the visit TIMES carry the information
    /// (a Poisson process whose rate depends on the frailty), the values are
    /// noise.
    Irregular,
    /// `horizon::synthetic::longitudinal`: a hidden state seen through noisy
    /// partial measurements at two to four visits.
    Longitudinal,
}

impl MixerData {
    /// Visit slots: every pre-entry visit of the longitudinal generator (two
    /// to four), the most recent twelve of the irregular process's Poisson
    /// visits (a handful on average, rarely more than twelve).
    fn visits(self) -> u32 {
        match self {
            MixerData::Irregular => 12,
            MixerData::Longitudinal => 4,
        }
    }
}

/// One variant's scores on the held-out test subjects.
#[derive(Clone, Debug)]
pub struct MixerRow {
    /// `attention`, `gated_delta_net` or `hybrid`.
    pub variant: &'static str,
    /// Trainable parameters.
    pub parameters: usize,
    /// Optimiser steps run.
    pub steps: u32,
    /// Weighted mean event NLL on the test subjects.
    pub nll: f32,
    /// Integrated Brier score (IPCW, over the grid up to the horizon).
    pub ibs: f64,
    /// Time-dependent AUC at the horizon.
    pub auc: f64,
    /// Wall-clock seconds of the optimisation loop, evaluations included.
    pub train_secs: f64,
}

/// The ablation's budget and the stated tolerance on parameter counts.
#[derive(Clone, Debug)]
pub struct MixerAblation {
    /// Training / early-stopping / scoring subjects.
    pub sizes: (usize, usize, usize),
    /// Optimiser steps, the same for every variant.
    pub steps: u32,
    /// Blocks of every stack (a multiple of the hybrid period, 4).
    pub blocks: u32,
    /// The largest allowed ratio of the biggest variant's parameter count to
    /// the smallest's, minus one.
    pub tolerance: f64,
}

impl Default for MixerAblation {
    fn default() -> Self {
        MixerAblation { sizes: (8_000, 2_000, 3_000), steps: 1_500, blocks: 4, tolerance: 0.01 }
    }
}

/// The variants compared, in the order they are reported.
const VARIANTS: [(&str, Mixer); 3] = [
    ("attention", Mixer::Attention),
    ("gated_delta_net", Mixer::GatedDeltaNet),
    ("hybrid", Mixer::Hybrid),
];

impl MixerAblation {
    /// Write the dataset (the same `timeline-v1` splits for every variant).
    pub fn prepare(&self, data: MixerData, dir: &Path, seed: u64) -> io::Result<()> {
        match data {
            MixerData::Irregular => write_dataset(dir, self.sizes, seed, irregular::population),
            MixerData::Longitudinal => write_dataset(dir, self.sizes, seed, longitudinal::population),
        }
    }

    /// Train and score every variant on the dataset under `dir`. Fails when
    /// the parameter counts differ by more than [`Self::tolerance`].
    pub fn run(&self, data: MixerData, dir: &Path, seed: u64) -> io::Result<Vec<MixerRow>> {
        match data {
            MixerData::Irregular => {
                let d: Dataset<irregular::Truth> = read_dataset(dir)?;
                let spec = base_spec(data, irregular::CODE, 6, vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]);
                self.compare(&d, &spec, &[irregular::CODE], Grid::new(4.0, 8), seed, |t, i, h| t[i].oracle_cif(h))
            }
            MixerData::Longitudinal => {
                let d: Dataset<longitudinal::Truth> = read_dataset(dir)?;
                let spec = base_spec(data, longitudinal::CODE, 16, vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0]);
                self.compare(&d, &spec, &[longitudinal::CODE], Grid::new(4.0, 8), seed, |t, i, h| t[i].oracle_cif(h))
            }
        }
    }

    fn compare<T>(
        &self,
        data: &Dataset<T>,
        base: &ModelSpec,
        codes: &'static [&'static str],
        grid: Grid,
        seed: u64,
        oracle_cif: impl Fn(&[T], usize, f64) -> f64,
    ) -> io::Result<Vec<MixerRow>> {
        let eval = Evaluation { test: &data.test, train: &data.train, codes, grid };
        let oracle = eval.tabulate(|i, _, t| oracle_cif(&data.truth, i, t));
        let mut rows = Vec::new();
        for (variant, mixer) in VARIANTS {
            let spec = ModelSpec {
                backbone: Backbone::Stack(StackConfig::new(mixer, self.blocks)),
                steps: self.steps,
                // More evaluations never come: every variant takes every step.
                patience: u32::MAX,
                ..base.clone()
            };
            let fitted = Fitted::fit(&spec, &data.train, &data.held_out, seed)?;
            let model = eval.model_table(&fitted.curves(&data.test));
            let scored = eval.compare(&model, &oracle, &[]);
            let field = |name: &str| scored.get(name).expect("compare reports it") as f64;
            rows.push(MixerRow {
                variant,
                parameters: fitted.model.ps.params.iter().map(|(n, _)| fitted.model.ps.numel(n)).sum(),
                steps: fitted.steps,
                nll: fitted.nll(&data.test),
                ibs: field("ibs"),
                auc: field("auc"),
                train_secs: fitted.train_secs,
            });
        }
        let (min, max) = rows.iter().fold((usize::MAX, 0), |(lo, hi), r| (lo.min(r.parameters), hi.max(r.parameters)));
        let spread = max as f64 / min as f64 - 1.0;
        if spread > self.tolerance {
            return Err(io::Error::other(format!(
                "the variants' parameter counts differ by {:.2}%, over the stated {:.2}%: {:?}",
                100.0 * spread,
                100.0 * self.tolerance,
                rows.iter().map(|r| (r.variant, r.parameters)).collect::<Vec<_>>()
            )));
        }
        Ok(rows)
    }
}

/// The model shared by every variant on `data`: width, feed-forward size and
/// heads follow the other survival benchmarks.
fn base_spec(data: MixerData, code: &str, max_tokens: u32, knots: Vec<f32>) -> ModelSpec {
    ModelSpec {
        codes: vec![code.into()],
        absorbing: vec![true],
        knots,
        max_tokens,
        d_model: 32,
        forecasts: 0,
        visits: data.visits(),
        backbone: Backbone::State,
        steps: 0,
        patience: DEFAULT_PATIENCE,
    }
}
