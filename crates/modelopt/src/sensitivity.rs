// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How much each site's loss would rise at each tier, estimated from the
//! curvature of the loss on calibration data.

use gpu_core::select::Dtype;
use model::ops::Weight;
use model::{Batch, Model};

use crate::Site;

/// One candidate tier for one site: its physical bytes
/// ([`Weight::bytes_for`]: packed words plus scales) and the estimated loss
/// increase of storing the site at it.
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

/// The estimated loss increase of every site at every tier in `tiers` its
/// shape packs ([`Weight::packs`]), by the second-order Taylor expansion of
/// the loss around the trained weights:
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
/// site element on the host. A gradient that is not finite is an error naming
/// the site, never a silently zero sensitivity.
pub fn sensitivity<M: Model>(m: &M, sites: &[Site], tiers: &[Dtype], calibration: &[Batch]) -> Result<Vec<SiteChoices>, String> {
    if calibration.is_empty() {
        return Err("sensitivity needs at least one calibration batch".into());
    }
    let mut fisher: Vec<Vec<f32>> = sites.iter().map(|s| vec![0f32; s.numel()]).collect();
    for &batch in calibration {
        m.zero_grads();
        m.set_batch(batch);
        let loss = m.forward();
        if !loss.is_finite() {
            return Err(format!("a calibration batch has a non-finite loss ({loss})"));
        }
        m.backward();
        for (site, f) in sites.iter().zip(&mut fisher) {
            let grad = m.read_grad(&site.name);
            if grad.len() != f.len() {
                return Err(format!("{}: gradient has {} elements, the site {}", site.name, grad.len(), f.len()));
            }
            for (fj, gj) in f.iter_mut().zip(&grad) {
                if !gj.is_finite() {
                    return Err(format!("{}: non-finite gradient on calibration data", site.name));
                }
                *fj += gj * gj;
            }
        }
    }
    m.zero_grads();
    let scale = 0.5 / calibration.len() as f64;
    Ok(sites
        .iter()
        .zip(&fisher)
        .map(|(site, f)| {
            let w = m.read_weight(&site.name);
            let mut choices: Vec<Choice> = tiers
                .iter()
                .filter(|&&tier| Weight::packs(tier, site.n, site.k))
                .map(|&tier| {
                    let q = Weight::round_trip(&w, site.n, site.k, tier);
                    let error = f.iter().zip(w.iter().zip(&q)).map(|(&fj, (&wj, &qj))| fj as f64 * ((wj - qj) as f64).powi(2)).sum::<f64>() * scale;
                    Choice { tier, bytes: Weight::bytes_for(tier, site.n, site.k), error }
                })
                .collect();
            choices.sort_by(|a, b| a.bytes.cmp(&b.bytes).then(a.error.total_cmp(&b.error)));
            SiteChoices { site: site.clone(), choices }
        })
        .collect())
}
