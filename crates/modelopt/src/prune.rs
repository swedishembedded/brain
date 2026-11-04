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

/// How narrow a pruned dimension becomes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PruneWidth {
    /// This share of each dimension's channels, rounded to the nearest channel.
    Fraction(f64),
    Channels(usize),
}

impl PruneWidth {
    /// The channels a dimension `width` wide keeps.
    pub fn channels(self, width: usize) -> Result<usize, String> {
        match self {
            PruneWidth::Channels(n) => Ok(n),
            PruneWidth::Fraction(f) if f > 0.0 && f <= 1.0 => Ok(((width as f64 * f).round() as usize).max(1)),
            PruneWidth::Fraction(f) => Err(format!("a kept fraction must be in (0, 1], got {f}")),
        }
    }
}

/// Which channels of each dimension stay, and the width each config key
/// narrows to (dimensions sharing a key share it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrunePlan {
    /// Per config key: its width before and after, in first-seen order.
    pub widths: Vec<(&'static str, usize, usize)>,
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

/// Keep the most important channels of every dimension, `width` of each.
/// Dimensions with one config key must be one width, so the pruned config
/// has one value per key.
pub fn plan_prune(dims: &[PrunableDim], importance: &[Vec<f64>], width: PruneWidth) -> Result<PrunePlan, String> {
    if dims.is_empty() {
        return Err("the model declares no prunable dimension".into());
    }
    let mut widths: Vec<(&'static str, usize, usize)> = Vec::new();
    for d in dims {
        let kept = width.channels(d.width)?;
        if kept == 0 || kept > d.width {
            return Err(format!("cannot keep {kept} channels of {}, which has {}", d.name, d.width));
        }
        match widths.iter().find(|(key, _, _)| *key == d.config_key) {
            Some(&(_, before, _)) if before != d.width => return Err(format!("{} is {} wide, other dimensions sized by {} are {before}", d.name, d.width, d.config_key)),
            Some(_) => {}
            None => widths.push((d.config_key, d.width, kept)),
        }
    }
    let keep = dims
        .iter()
        .zip(importance)
        .map(|(d, scores)| {
            let kept = widths.iter().find(|(key, _, _)| *key == d.config_key).map_or(0, |w| w.2);
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            order.truncate(kept);
            order.sort_unstable();
            order
        })
        .collect();
    Ok(PrunePlan { widths, keep })
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

/// The pruned model: its config, with each of `plan.widths`' keys set,
/// and every parameter of `m`, each prunable slice keeping only its plan's
/// channels. Build it with [`Model::new`] and save it as any checkpoint.
pub fn pruned_weights<M: Model>(m: &M, dims: &[PrunableDim], plan: &PrunePlan) -> Result<PrunedModel<M::Config>, String> {
    let mut json = m.config_json();
    for &(key, _, after) in &plan.widths {
        json.get_mut(key).ok_or_else(|| format!("the model's config has no {key} to narrow"))?;
        json[key] = serde_json::json!(after);
    }
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
