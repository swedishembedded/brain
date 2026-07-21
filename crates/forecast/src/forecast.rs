// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The forecasting **output**: a distribution-valued `Forecast`.
//!
//! These models predict a *conditional distribution* over the horizon, not a
//! price path. Different models natively emit different representations —
//! Chronos-2 quantiles, Kronos samples, a GARCH baseline a Gaussian, TLOB class
//! probabilities. [`Forecast`] carries whichever the model produced in
//! [`native_representation`](Forecast::native_representation), plus any
//! representations the caller asked for that we could derive from it.
//!
//! The honesty rule: every field brain *computed* rather than the model
//! *emitting* is marked `derived = true` and names the conversion `method`. A
//! caller can then refuse to size a position off, say, a sample set that is
//! really an interpolation of five quantiles. Conversions that are not
//! mathematically sound (e.g. point → quantiles) are **not performed** — the
//! request errors instead of fabricating an interval.

use crate::Representation;

/// A block of forecast values with an explicit shape and provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// Row-major dimensions, e.g. `[horizon, n_quantiles]` or `[n_samples, horizon]`.
    pub shape: Vec<usize>,
    /// Row-major values.
    pub data: Vec<f32>,
    /// `false` if the model emitted this directly; `true` if brain derived it.
    pub derived: bool,
    /// When `derived`, the conversion used (e.g. `"inverse_cdf_interp"`). Empty
    /// otherwise.
    pub method: String,
}

impl Block {
    /// A model-emitted (non-derived) block.
    pub fn native(shape: Vec<usize>, data: Vec<f32>) -> Block {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "Block: shape/data mismatch");
        Block { shape, data, derived: false, method: String::new() }
    }

    /// A brain-derived block, tagged with the conversion method.
    pub fn derived(shape: Vec<usize>, data: Vec<f32>, method: impl Into<String>) -> Block {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "Block: shape/data mismatch");
        Block { shape, data, derived: true, method: method.into() }
    }
}

/// The forecast for one target series of one item.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TargetForecast {
    /// The item this target belongs to (echoes [`crate::Item::item_id`]).
    pub item_id: String,
    /// The target variate name (echoes [`crate::Variate::name`]).
    pub name: String,
    /// Quantile forecasts, `[horizon, n_levels]`, with the levels in `levels`.
    pub quantiles: Option<Block>,
    /// The quantile levels for `quantiles` (e.g. `[0.1, 0.5, 0.9]`).
    pub levels: Vec<f32>,
    /// Sampled trajectories, `[n_samples, horizon]`.
    pub samples: Option<Block>,
    /// A point path, `[horizon]` (mean or median).
    pub mean: Option<Block>,
    /// Parametric distribution parameters, `[horizon, n_params]`.
    pub distribution: Option<Block>,
    /// The parameter names for `distribution` (e.g. `["mu", "sigma"]`).
    pub dist_family: String,
    /// Per-step class probabilities, `[horizon, n_classes]`.
    pub classes: Option<Block>,
    /// Class labels for `classes` (e.g. `["down", "flat", "up"]`).
    pub class_labels: Vec<String>,
}

impl TargetForecast {
    pub fn new(item_id: impl Into<String>, name: impl Into<String>) -> TargetForecast {
        TargetForecast { item_id: item_id.into(), name: name.into(), ..Default::default() }
    }

    /// The horizon length, read from whichever representation is present.
    pub fn horizon(&self) -> usize {
        if let Some(q) = &self.quantiles {
            return q.shape.first().copied().unwrap_or(0);
        }
        if let Some(s) = &self.samples {
            return s.shape.get(1).copied().unwrap_or(0);
        }
        if let Some(m) = &self.mean {
            return m.shape.first().copied().unwrap_or(0);
        }
        if let Some(d) = &self.distribution {
            return d.shape.first().copied().unwrap_or(0);
        }
        if let Some(c) = &self.classes {
            return c.shape.first().copied().unwrap_or(0);
        }
        0
    }
}

/// A complete forecast: which representation the model emitted, and the
/// per-target forecasts.
#[derive(Clone, Debug, PartialEq)]
pub struct Forecast {
    /// The model that produced this (`"chronos2"`, `"kronos"`, `"naive"`).
    pub model: String,
    /// Version/provenance string (`"amazon/chronos-2@<sha>"`).
    pub model_version: String,
    /// What the model natively emitted — the ground truth of this forecast.
    pub native_representation: Representation,
    /// Forecast horizon in steps.
    pub horizon: usize,
    /// Sampling frequency (echoes the panel).
    pub freq: String,
    /// The per-target forecasts.
    pub targets: Vec<TargetForecast>,
}

impl Forecast {
    pub fn new(
        model: impl Into<String>,
        native: Representation,
        horizon: usize,
        freq: impl Into<String>,
    ) -> Forecast {
        Forecast {
            model: model.into(),
            model_version: String::new(),
            native_representation: native,
            horizon,
            freq: freq.into(),
            targets: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_shape_must_match_data() {
        let b = Block::native(vec![2, 3], vec![0.0; 6]);
        assert!(!b.derived);
        assert_eq!(b.shape, vec![2, 3]);
    }

    #[test]
    #[should_panic]
    fn block_rejects_shape_data_mismatch() {
        Block::native(vec![2, 3], vec![0.0; 5]);
    }

    #[test]
    fn horizon_reads_from_whatever_representation_is_present() {
        let mut tf = TargetForecast::new("AAPL", "close");
        tf.samples = Some(Block::native(vec![100, 20], vec![0.0; 2000]));
        assert_eq!(tf.horizon(), 20); // samples are [n_samples, horizon]

        let mut tf2 = TargetForecast::new("AAPL", "close");
        tf2.quantiles = Some(Block::native(vec![24, 3], vec![0.0; 72]));
        assert_eq!(tf2.horizon(), 24); // quantiles are [horizon, n_levels]
    }
}
