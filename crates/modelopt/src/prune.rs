// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for shrinking trained models to
// the compute a deployment has, by removing what they need least, for its
// clients. If your team needs expertise in structured pruning and model
// compression then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Structured pruning: remove whole channels of the dimensions a model
//! declares prunable ([`Model::prunable_dims`]), ranked by the estimated loss
//! increase of removing each, so the pruned model runs smaller dense products
//! rather than masked ones.
//!
//! Removing a channel sets every weight it owns to zero, so the Taylor
//! estimate of [`crate::sensitivity`] applies with the codec error replaced by
//! the weight itself: `importance[c] = 1/2 * sum_j F_j * w_j^2` over the
//! channel's slices. A model with a channel's slices zeroed computes exactly
//! what the model without the channel computes, which is how
//! [`measure_pruned`] measures a plan before anything is exported, and
//! [`pruned_weights`] writes the narrower model itself.

use std::collections::HashMap;

use model::prunable::{Axis, PrunableDim};
use model::{Batch, Model, ModelConfig};

use crate::sensitivity::fisher;
use crate::Progress;

/// Which channels of each dimension stay, all narrowed to one width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrunePlan {
    /// The config key the dimensions share ([`PrunableDim::config_key`]).
    pub config_key: &'static str,
    pub width: usize,
    /// Per dimension, in [`Model::prunable_dims`] order: the kept channels, ascending.
    pub keep: Vec<Vec<usize>>,
}

/// The `[rows, cols]` of a slice of a dimension `width` wide holding `len` elements.
fn slice_shape(len: usize, width: usize, axis: Axis) -> Result<(usize, usize), String> {
    if width == 0 || !len.is_multiple_of(width) {
        return Err(format!("{len} elements are not whole lines of a {width}-wide dimension"));
    }
    Ok(match axis {
        Axis::Rows => (width, len / width),
        Axis::Cols => (len / width, width),
    })
}

/// The elements of channel `c` in a `[rows, cols]` slice along `axis`.
fn channel_elements(rows: usize, cols: usize, axis: Axis, c: usize) -> Box<dyn Iterator<Item = usize>> {
    match axis {
        Axis::Rows => Box::new(c * cols..(c + 1) * cols),
        Axis::Cols => Box::new((0..rows).map(move |r| r * cols + c)),
    }
}

/// The estimated loss increase of removing each channel of each dimension,
/// from the curvature of the loss on `calibration` (one forward and backward
/// per batch).
pub fn channel_importance<M: Model>(m: &M, dims: &[PrunableDim], calibration: &[Batch], on_progress: &mut dyn FnMut(Progress)) -> Result<Vec<Vec<f64>>, String> {
    let names: Vec<(&str, usize)> = dims.iter().flat_map(|d| &d.slices).map(|(name, _)| (name.as_str(), m.read_weight(name).len())).collect();
    let fisher = fisher(m, &names, calibration, on_progress)?;
    let scale = 0.5 / calibration.len() as f64;
    let mut fisher = fisher.into_iter();
    dims.iter()
        .map(|dim| {
            let mut importance = vec![0f64; dim.width];
            for ((name, axis), f) in dim.slices.iter().zip(fisher.by_ref()) {
                let w = m.read_weight(name);
                let (rows, cols) = slice_shape(w.len(), dim.width, *axis).map_err(|e| format!("{name}: {e}"))?;
                for (c, score) in importance.iter_mut().enumerate() {
                    *score += channel_elements(rows, cols, *axis, c).map(|j| f[j] as f64 * (w[j] as f64).powi(2)).sum::<f64>() * scale;
                }
            }
            Ok(importance)
        })
        .collect()
}

/// Keep the `width` most important channels of every dimension. All the
/// dimensions must share one config key, so the pruned config has one width.
pub fn plan_prune(dims: &[PrunableDim], importance: &[Vec<f64>], width: usize) -> Result<PrunePlan, String> {
    let config_key = dims.first().ok_or("the model declares no prunable dimension")?.config_key;
    if let Some(other) = dims.iter().find(|d| d.config_key != config_key) {
        return Err(format!("{} is sized by {}, the others by {config_key}; prune one config key at a time", other.name, other.config_key));
    }
    if let Some(narrow) = dims.iter().find(|d| width == 0 || width > d.width) {
        return Err(format!("cannot keep {width} channels of {}, which has {}", narrow.name, narrow.width));
    }
    let keep = importance
        .iter()
        .map(|scores| {
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            order.truncate(width);
            order.sort_unstable();
            order
        })
        .collect();
    Ok(PrunePlan { config_key, width, keep })
}

/// The held-out loss of `m` with every channel `plan` drops zeroed, which is
/// what the pruned model computes; the weights are written back before this
/// returns.
pub fn measure_pruned<M: Model>(m: &M, dims: &[PrunableDim], plan: &PrunePlan, held_out: &[Batch]) -> Result<f64, String> {
    let mut saved: Vec<(String, Vec<f32>)> = Vec::new();
    let result = (|| {
        for (dim, keep) in dims.iter().zip(&plan.keep) {
            for (name, axis) in &dim.slices {
                let original = m.read_weight(name);
                let (rows, cols) = slice_shape(original.len(), dim.width, *axis).map_err(|e| format!("{name}: {e}"))?;
                let mut zeroed = original.clone();
                for c in (0..dim.width).filter(|c| keep.binary_search(c).is_err()) {
                    for j in channel_elements(rows, cols, *axis, c) {
                        zeroed[j] = 0.0;
                    }
                }
                m.write_weight(name, &zeroed);
                saved.push((name.clone(), original));
            }
        }
        Ok(crate::held_out_loss(m, held_out))
    })();
    for (name, w) in saved.into_iter().rev() {
        m.write_weight(&name, &w);
    }
    result
}

/// A pruned model, ready for [`Model::new`]: its config and every parameter.
pub struct PrunedModel<C> {
    pub config: C,
    pub weights: HashMap<String, Vec<f32>>,
}

/// The pruned model: its config, with `plan.config_key` set to `plan.width`,
/// and every parameter of `m`, each prunable slice keeping only its plan's
/// channels. Build it with [`Model::new`] and save it as any checkpoint.
pub fn pruned_weights<M: Model>(m: &M, dims: &[PrunableDim], plan: &PrunePlan) -> Result<PrunedModel<M::Config>, String> {
    let mut json = m.config_json();
    json.get_mut(plan.config_key).ok_or_else(|| format!("the model's config has no {} to narrow", plan.config_key))?;
    json[plan.config_key] = serde_json::json!(plan.width);
    let config = M::Config::from_json(&json);
    let mut slices: HashMap<&str, (usize, Axis, &[usize])> = HashMap::new();
    for (dim, keep) in dims.iter().zip(&plan.keep) {
        for (name, axis) in &dim.slices {
            slices.insert(name.as_str(), (dim.width, *axis, keep.as_slice()));
        }
    }
    let mut weights = HashMap::new();
    for name in m.param_names() {
        let w = m.read_weight(&name);
        let kept = match slices.get(name.as_str()) {
            None => w,
            Some(&(width, axis, keep)) => {
                let (rows, cols) = slice_shape(w.len(), width, axis).map_err(|e| format!("{name}: {e}"))?;
                match axis {
                    Axis::Rows => keep.iter().flat_map(|&r| w[r * cols..(r + 1) * cols].iter().copied()).collect(),
                    Axis::Cols => (0..rows).flat_map(|r| keep.iter().map(move |&c| (r, c))).map(|(r, c)| w[r * cols + c]).collect(),
                }
            }
        };
        weights.insert(name, kept);
    }
    Ok(PrunedModel { config, weights })
}
