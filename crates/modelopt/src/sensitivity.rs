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
pub fn sensitivity<M: Model>(m: &M, sites: &[Site], tiers: &[Dtype], calibration: &[Batch]) -> Result<Vec<SiteChoices>, String> {
    if calibration.is_empty() {
        return Err("sensitivity needs at least one calibration batch".into());
    }
    let tensors: Vec<&crate::Tensor> = sites.iter().flat_map(|s| &s.tensors).collect();
    let mut fisher: Vec<Vec<f32>> = tensors.iter().map(|t| vec![0f32; t.n * t.k]).collect();
    for &batch in calibration {
        m.zero_grads();
        m.set_batch(batch);
        let loss = m.forward();
        if !loss.is_finite() {
            return Err(format!("a calibration batch has a non-finite loss ({loss})"));
        }
        m.backward();
        for (t, f) in tensors.iter().zip(&mut fisher) {
            let grad = m.read_grad(&t.name);
            if grad.len() != f.len() {
                return Err(format!("{}: gradient has {} elements, the tensor {}", t.name, grad.len(), f.len()));
            }
            for (fj, gj) in f.iter_mut().zip(&grad) {
                if !gj.is_finite() {
                    return Err(format!("{}: non-finite gradient on calibration data", t.name));
                }
                *fj += gj * gj;
            }
        }
    }
    m.zero_grads();
    let scale = 0.5 / calibration.len() as f64;
    // error[tensor][tier], for the tiers the tensor packs.
    let per_tensor: Vec<Vec<Option<f64>>> = tensors
        .iter()
        .zip(&fisher)
        .map(|(t, f)| {
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
        })
        .collect();
    let mut next = 0;
    Ok(sites
        .iter()
        .map(|site| {
            let rows = &per_tensor[next..next + site.tensors.len()];
            next += site.tensors.len();
            let mut choices: Vec<Choice> = tiers
                .iter()
                .enumerate()
                .filter_map(|(q, &tier)| {
                    let error = rows.iter().map(|r| r[q]).sum::<Option<f64>>()?;
                    let bytes = site.tensors.iter().map(|t| Weight::bytes_for(tier, t.n, t.k)).sum();
                    Some(Choice { tier, bytes, error })
                })
                .collect();
            choices.sort_by(|a, b| a.bytes.cmp(&b.bytes).then(a.error.total_cmp(&b.error)));
            SiteChoices { site: site.clone(), choices }
        })
        .collect())
}
