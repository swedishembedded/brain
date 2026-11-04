// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How much each site's loss would rise at each tier, estimated from the
//! curvature of the loss on calibration data.

use gpu_core::select::Dtype;
use model::ops::Weight;
use model::{Batch, Model};

use crate::Site;

/// One candidate tier for one site: its physical bytes
/// ([`Weight::bytes_for`]: packed words plus scales, summed over the site's
/// tensors) and the estimated loss increase of storing the site at it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Choice {
    pub tier: Dtype,
    pub bytes: u64,
    /// The measured seconds of the site's product at this tier, summed over
    /// its tensors; 0 when the search times nothing.
    pub seconds: f64,
    pub error: f64,
}

/// A site and its candidate tiers, cheapest first.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteChoices {
    pub site: Site,
    pub choices: Vec<Choice>,
}

/// The estimated loss increase of every site at every tier in `tiers` all of
/// its tensors pack ([`Weight::packs`]), by the second-order Taylor expansion
/// of the loss around the trained weights:
///
/// `error[i, q] = 1/2 * sum_j F_j * (w_j - Q_q(w_j))^2`
///
/// with `Q_q` the deployed codec ([`Weight::round_trip`]) and `F` the
/// diagonal empirical Fisher, the mean over `calibration` of each batch's
/// squared gradient, standing in for the Hessian diagonal. The first-order
/// term is dropped: at a trained minimum the mean gradient is near zero,
/// while the rounding error is not. Sites are assumed independent, which is
/// why this is an estimate that picks candidates and never a measurement that
/// accepts one ([`crate::measure_policy`]).
///
/// Runs one forward and backward per calibration batch and holds one f32 per
/// site element on the host. A loss or gradient that is not finite is an error
/// naming where, never a silently zero sensitivity.
pub fn sensitivity<M: Model>(m: &M, sites: &[Site], tiers: &[Dtype], calibration: &[Batch], on_progress: &mut dyn FnMut(crate::Progress)) -> Result<Vec<SiteChoices>, String> {
    let tensors: Vec<(&str, usize)> = sites.iter().flat_map(|s| &s.tensors).map(|t| (t.name.as_str(), t.n * t.k)).collect();
    let fisher = fisher(m, &tensors, calibration, on_progress)?;
    let scale = 0.5 / calibration.len() as f64;
    // The estimated error of one tensor at every tier, `None` where a tier does not pack it: one read of the weight.
    let errors = |t: &crate::Tensor, f: &[f32]| -> Vec<Option<f64>> {
        let w = m.read_weight(&t.name);
        tiers
            .iter()
            .map(|&tier| {
                Weight::packs(tier, t.n, t.k).then(|| {
                    let q = Weight::round_trip(&w, t.n, t.k, tier);
                    f.iter().zip(w.iter().zip(&q)).map(|(&fj, (&wj, &qj))| fj as f64 * ((wj - qj) as f64).powi(2)).sum::<f64>() * scale
                })
            })
            .collect()
    };
    let mut fisher = fisher.into_iter();
    let mut out = Vec::with_capacity(sites.len());
    for (i, site) in sites.iter().enumerate() {
        let per_tensor: Vec<Vec<Option<f64>>> = site.tensors.iter().zip(fisher.by_ref()).map(|(t, f)| errors(t, &f)).collect();
        let mut choices: Vec<Choice> = tiers
            .iter()
            .enumerate()
            .filter_map(|(q, &tier)| {
                let error = per_tensor.iter().map(|e| e[q]).sum::<Option<f64>>()?;
                let bytes = site.tensors.iter().map(|t| Weight::bytes_for(tier, t.n, t.k)).sum();
                Some(Choice { tier, bytes, seconds: 0.0, error })
            })
            .collect();
        choices.sort_by(|a, b| a.bytes.cmp(&b.bytes).then(a.error.total_cmp(&b.error)));
        out.push(SiteChoices { site: site.clone(), choices });
        on_progress(crate::Progress::Estimated { site: i + 1, of: sites.len() });
    }
    Ok(out)
}

/// The summed squared gradient of each named tensor (`(name, elements)`) over
/// the calibration batches, one forward and backward per batch: the diagonal
/// empirical Fisher times the batch count. A loss or gradient that is not
/// finite is an error naming where. Gradients are zeroed afterwards.
pub(crate) fn fisher<M: Model>(m: &M, tensors: &[(&str, usize)], calibration: &[Batch], on_progress: &mut dyn FnMut(crate::Progress)) -> Result<Vec<Vec<f32>>, String> {
    if calibration.is_empty() {
        return Err("the curvature estimate needs at least one calibration batch".into());
    }
    let mut fisher: Vec<Vec<f32>> = tensors.iter().map(|&(_, len)| vec![0f32; len]).collect();
    for (i, &batch) in calibration.iter().enumerate() {
        m.zero_grads();
        m.set_batch(batch);
        let loss = m.forward();
        if !loss.is_finite() {
            return Err(format!("a calibration batch has a non-finite loss ({loss})"));
        }
        m.backward();
        for (&(name, _), f) in tensors.iter().zip(&mut fisher) {
            let grad = m.read_grad(name);
            if grad.len() != f.len() {
                return Err(format!("{name}: gradient has {} elements, the tensor {}", grad.len(), f.len()));
            }
            for (fj, gj) in f.iter_mut().zip(&grad) {
                if !gj.is_finite() {
                    return Err(format!("{name}: non-finite gradient on calibration data"));
                }
                *fj += gj * gj;
            }
        }
        on_progress(crate::Progress::Calibrated { batch: i + 1, of: calibration.len() });
    }
    m.zero_grads();
    Ok(fisher)
}
