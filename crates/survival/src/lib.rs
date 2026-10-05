// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Survival estimation and evaluation arithmetic.
//!
//! Swedish Embedded AB implements validation of risk models whose outcomes
//! arrive years later, censored and competing, for its clients. If your team
//! needs expertise in evaluating time-to-event predictions honestly, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! Everything takes [`Obs`] - an observed time, the cause of the event (or
//! none: censored) and a sampling weight - and works in f64. A LEAF like
//! `promote` and `rlcd`: it depends on no other brain crate, so a model crate,
//! an application or a sample can evaluate predictions without pulling in a
//! model.
//!
//! - [`estimate`]: weighted Kaplan-Meier, the censoring distribution, and
//!   Aalen-Johansen cumulative incidence for one cause among several.
//! - [`concordance`]: Harrell's C and Uno's truncated, IPCW-weighted C, with
//!   competing causes, in `O(n log n)`.
//! - [`brier`]: the IPCW Brier score at a horizon and its integral.
//! - [`compare`]: two models on the same data - the corrected resampled
//!   t-test across folds and a cluster bootstrap over units.
//! - [`calibration`]: D-calibration over predicted survival, and calibration
//!   at a horizon (observed-over-expected, IPCW logistic recalibration
//!   intercept and slope, a risk-group table).
//!
//! Weights are sampling weights: a subject of integer weight `w` contributes
//! exactly as `w` copies of it would (the tests hold every metric to that).
//! Concordance is not a proper scoring rule; report it beside a Brier score
//! and calibration, never alone.

pub mod brier;
pub mod calibration;
pub mod compare;
pub mod concordance;
pub mod estimate;
mod special;

/// One subject's observed outcome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obs {
    /// Time of the event or of censoring.
    pub time: f64,
    /// The cause that happened at `time`, or `None` if censored.
    pub cause: Option<usize>,
    /// Sampling weight (1 when unweighted).
    pub weight: f64,
}

impl Obs {
    /// An event of `cause` at `time`, weight 1.
    pub fn event(time: f64, cause: usize) -> Obs {
        Obs {
            time,
            cause: Some(cause),
            weight: 1.0,
        }
    }
    /// Censored at `time`, weight 1.
    pub fn censored(time: f64) -> Obs {
        Obs {
            time,
            cause: None,
            weight: 1.0,
        }
    }
    /// The same observation with weight `w`.
    pub fn weighted(self, w: f64) -> Obs {
        Obs { weight: w, ..self }
    }
}
