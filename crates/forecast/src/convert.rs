// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Representation conversion — the honesty layer.
//!
//! A caller asks for a set of [`Representation`]s; the model emits exactly one.
//! This module derives the requested ones from the native one *where it is
//! mathematically sound*, tagging every derived block with its `method`, and
//! **errors** rather than fabricate an unsupported conversion.
//!
//! Soundness table (native → requested):
//! - `samples`      → quantiles (empirical), mean (average), functionals — exact.
//! - `distribution` → quantiles, mean (closed form) — exact.
//! - `quantiles`    → mean (interpolate the median / average — approximate), and
//!   samples (inverse-CDF interpolation — **lossy**).
//! - `point`        → nothing. A point forecast asked for quantiles is an error,
//!   not a zero-width interval.
//! - `classes`      → nothing numeric; class probabilities are their own thing.

use crate::{Block, ForecastError, Representation, TargetForecast};

/// Ensure `tf` carries every representation in `want`, deriving from its native
/// `native` representation where sound. Returns an error for an impossible
/// conversion (the caller decides whether to surface or ignore it).
pub fn ensure_representations(
    tf: &mut TargetForecast,
    native: Representation,
    want: &[Representation],
    levels: &[f32],
    num_samples: usize,
    seed: u64,
) -> Result<(), ForecastError> {
    for &r in want {
        if has(tf, r) {
            continue;
        }
        match (native, r) {
            (Representation::Samples, Representation::Quantiles) => {
                let (q, shape) = samples_to_quantiles(tf, levels)?;
                tf.levels = levels.to_vec();
                tf.quantiles = Some(Block::derived(shape, q, "empirical_quantiles"));
            }
            (Representation::Samples, Representation::Point) => {
                tf.mean = Some(samples_to_mean(tf)?);
            }
            (Representation::Quantiles, Representation::Point) => {
                tf.mean = Some(quantiles_to_mean(tf)?);
            }
            (Representation::Quantiles, Representation::Samples) => {
                tf.samples = Some(quantiles_to_samples(tf, num_samples, seed)?);
            }
            // Distribution decoding is model-family specific (Gaussian, StudentT,
            // knots); the model supplies a decoder. The generic path only
            // handles Gaussian `[mu, sigma]`, wired in convert_gaussian.
            (Representation::Distribution, Representation::Quantiles) => {
                let (q, shape) = gaussian_to_quantiles(tf, levels)?;
                tf.levels = levels.to_vec();
                tf.quantiles = Some(Block::derived(shape, q, "gaussian_ppf"));
            }
            (Representation::Distribution, Representation::Point) => {
                tf.mean = Some(gaussian_to_mean(tf)?);
            }
            (Representation::Distribution, Representation::Samples) => {
                tf.samples = Some(gaussian_to_samples(tf, num_samples, seed)?);
            }
            (from, to) if from == to => {}
            (from, to) => {
                return Err(ForecastError::unsupported(format!(
                    "cannot derive {} from a {} forecast",
                    to.as_str(),
                    from.as_str()
                )));
            }
        }
    }
    Ok(())
}

fn has(tf: &TargetForecast, r: Representation) -> bool {
    match r {
        Representation::Quantiles => tf.quantiles.is_some(),
        Representation::Samples => tf.samples.is_some(),
        Representation::Point => tf.mean.is_some(),
        Representation::Distribution => tf.distribution.is_some(),
        Representation::Classes => tf.classes.is_some(),
    }
}

/// Empirical quantiles per horizon step from a `[n_samples, horizon]` sample
/// block. Linear interpolation between order statistics (numpy default).
fn samples_to_quantiles(
    tf: &TargetForecast,
    levels: &[f32],
) -> Result<(Vec<f32>, Vec<usize>), ForecastError> {
    let s = tf.samples.as_ref().ok_or_else(|| ForecastError::internal("no samples"))?;
    let (n, h) = (s.shape[0], s.shape[1]);
    let mut out = vec![0.0f32; h * levels.len()];
    let mut col = vec![0.0f32; n];
    for t in 0..h {
        for (i, c) in col.iter_mut().enumerate() {
            *c = s.data[i * h + t];
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (j, &q) in levels.iter().enumerate() {
            out[t * levels.len() + j] = interp_quantile(&col, q);
        }
    }
    Ok((out, vec![h, levels.len()]))
}

/// Type-7 (numpy default) quantile of a sorted slice.
fn interp_quantile(sorted: &[f32], q: f32) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f32;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

fn samples_to_mean(tf: &TargetForecast) -> Result<Block, ForecastError> {
    let s = tf.samples.as_ref().ok_or_else(|| ForecastError::internal("no samples"))?;
    let (n, h) = (s.shape[0], s.shape[1]);
    let mut out = vec![0.0f32; h];
    for (t, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for i in 0..n {
            acc += s.data[i * h + t];
        }
        *o = acc / n as f32;
    }
    Ok(Block::derived(vec![h], out, "sample_mean"))
}

/// Approximate mean from quantiles: take the median level if present, else the
/// average of the supplied quantiles. Flagged approximate.
fn quantiles_to_mean(tf: &TargetForecast) -> Result<Block, ForecastError> {
    let q = tf.quantiles.as_ref().ok_or_else(|| ForecastError::internal("no quantiles"))?;
    let (h, ql) = (q.shape[0], q.shape[1]);
    // nearest level to 0.5
    let mid = tf
        .levels
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (a.1 - 0.5).abs().partial_cmp(&(b.1 - 0.5).abs()).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i);
    let mut out = vec![0.0f32; h];
    for (t, o) in out.iter_mut().enumerate() {
        *o = match mid {
            Some(j) => q.data[t * ql + j],
            None => q.data[t * ql..t * ql + ql].iter().sum::<f32>() / ql as f32,
        };
    }
    Ok(Block::derived(vec![h], out, "quantile_median_approx"))
}

/// Samples from quantiles via inverse-CDF (piecewise-linear) interpolation.
/// **Lossy** — the tails beyond the outermost quantiles are flat. Deterministic
/// given `seed`.
fn quantiles_to_samples(
    tf: &TargetForecast,
    n: usize,
    seed: u64,
) -> Result<Block, ForecastError> {
    let q = tf.quantiles.as_ref().ok_or_else(|| ForecastError::internal("no quantiles"))?;
    if tf.levels.len() < 2 {
        return Err(ForecastError::unsupported(
            "need >= 2 quantile levels to interpolate samples",
        ));
    }
    let (h, ql) = (q.shape[0], q.shape[1]);
    let mut rng = SplitMix64::new(seed);
    let mut out = vec![0.0f32; n * h];
    for i in 0..n {
        for t in 0..h {
            let u = rng.next_f32();
            let row = &q.data[t * ql..t * ql + ql];
            out[i * h + t] = inverse_cdf(&tf.levels, row, u);
        }
    }
    Ok(Block::derived(vec![n, h], out, "inverse_cdf_interp"))
}

/// Piecewise-linear inverse CDF: given quantile `levels` and their `values` at
/// one step, return the value at cumulative probability `u`. Flat outside the
/// level range.
fn inverse_cdf(levels: &[f32], values: &[f32], u: f32) -> f32 {
    if u <= levels[0] {
        return values[0];
    }
    if u >= levels[levels.len() - 1] {
        return values[values.len() - 1];
    }
    for k in 1..levels.len() {
        if u <= levels[k] {
            let span = levels[k] - levels[k - 1];
            let frac = if span > 0.0 { (u - levels[k - 1]) / span } else { 0.0 };
            return values[k - 1] + (values[k] - values[k - 1]) * frac;
        }
    }
    values[values.len() - 1]
}

// -- Gaussian parametric decode (the one generic distribution family) ---------

fn gaussian_params(tf: &TargetForecast) -> Result<(usize, &Block), ForecastError> {
    let d = tf.distribution.as_ref().ok_or_else(|| ForecastError::internal("no distribution"))?;
    if tf.dist_family != "gaussian" || d.shape[1] != 2 {
        return Err(ForecastError::unsupported(format!(
            "generic distribution decode supports only gaussian [mu,sigma], got {}",
            tf.dist_family
        )));
    }
    Ok((d.shape[0], d))
}

fn gaussian_to_quantiles(
    tf: &TargetForecast,
    levels: &[f32],
) -> Result<(Vec<f32>, Vec<usize>), ForecastError> {
    let (h, d) = gaussian_params(tf)?;
    let mut out = vec![0.0f32; h * levels.len()];
    for t in 0..h {
        let mu = d.data[t * 2];
        let sigma = d.data[t * 2 + 1];
        for (j, &q) in levels.iter().enumerate() {
            out[t * levels.len() + j] = mu + sigma * norm_ppf(q);
        }
    }
    Ok((out, vec![h, levels.len()]))
}

fn gaussian_to_mean(tf: &TargetForecast) -> Result<Block, ForecastError> {
    let (h, d) = gaussian_params(tf)?;
    let out: Vec<f32> = (0..h).map(|t| d.data[t * 2]).collect();
    Ok(Block::derived(vec![h], out, "gaussian_mean"))
}

/// Draw `n` trajectories from a per-step Gaussian `[mu, sigma]`. Deterministic
/// given `seed`. Uses the standard-normal inverse CDF on uniform draws.
fn gaussian_to_samples(
    tf: &TargetForecast,
    n: usize,
    seed: u64,
) -> Result<Block, ForecastError> {
    let (h, d) = gaussian_params(tf)?;
    let mut rng = SplitMix64::new(seed);
    let mut out = vec![0.0f32; n * h];
    for i in 0..n {
        for t in 0..h {
            let (mu, sigma) = (d.data[t * 2], d.data[t * 2 + 1]);
            out[i * h + t] = mu + sigma * norm_ppf(rng.next_f32());
        }
    }
    Ok(Block::derived(vec![n, h], out, "gaussian_sampling"))
}

/// Standard-normal inverse CDF (Acklam's rational approximation, ~1e-9 abs
/// error). Dependency-free, adequate for fp32.
pub fn norm_ppf(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6) as f64;
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    let x = if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    x as f32
}

/// SplitMix64 — a tiny deterministic PRNG for reproducible sample derivation.
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        // 24-bit mantissa -> [0,1)
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Forecast;

    fn tf_with_samples() -> TargetForecast {
        // 4 samples x horizon 1: [1, 2, 3, 4]
        let mut tf = TargetForecast::new("X", "close");
        tf.samples = Some(Block::native(vec![4, 1], vec![1.0, 2.0, 3.0, 4.0]));
        tf
    }

    #[test]
    fn samples_to_quantiles_are_empirical_and_flagged_derived() {
        let mut tf = tf_with_samples();
        ensure_representations(
            &mut tf,
            Representation::Samples,
            &[Representation::Quantiles],
            &[0.0, 0.5, 1.0],
            0,
            0,
        )
        .unwrap();
        let q = tf.quantiles.as_ref().unwrap();
        assert!(q.derived && q.method == "empirical_quantiles");
        // type-7 on [1,2,3,4]: q0=1, q0.5=2.5, q1=4
        assert!((q.data[0] - 1.0).abs() < 1e-6);
        assert!((q.data[1] - 2.5).abs() < 1e-6);
        assert!((q.data[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn samples_to_mean_averages() {
        let mut tf = tf_with_samples();
        ensure_representations(
            &mut tf,
            Representation::Samples,
            &[Representation::Point],
            &[],
            0,
            0,
        )
        .unwrap();
        let m = tf.mean.as_ref().unwrap();
        assert!(m.derived && (m.data[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn point_to_quantiles_is_an_error_not_a_fabricated_interval() {
        let mut tf = TargetForecast::new("X", "close");
        tf.mean = Some(Block::native(vec![1], vec![5.0]));
        let err = ensure_representations(
            &mut tf,
            Representation::Point,
            &[Representation::Quantiles],
            &[0.1, 0.9],
            0,
            0,
        )
        .unwrap_err();
        assert_eq!(err.code, "unsupported_capability");
        assert!(tf.quantiles.is_none(), "must not fabricate an interval");
    }

    #[test]
    fn quantiles_to_samples_is_lossy_and_flagged() {
        let mut tf = TargetForecast::new("X", "close");
        // horizon 1, levels [0.1, 0.5, 0.9] -> values [0, 10, 20]
        tf.quantiles = Some(Block::native(vec![1, 3], vec![0.0, 10.0, 20.0]));
        tf.levels = vec![0.1, 0.5, 0.9];
        ensure_representations(
            &mut tf,
            Representation::Quantiles,
            &[Representation::Samples],
            &[0.1, 0.5, 0.9],
            1000,
            42,
        )
        .unwrap();
        let s = tf.samples.as_ref().unwrap();
        assert!(s.derived && s.method == "inverse_cdf_interp");
        assert_eq!(s.shape, vec![1000, 1]);
        // sample mean should sit near the median value (10) by symmetry
        let mean: f32 = s.data.iter().sum::<f32>() / s.data.len() as f32;
        assert!((mean - 10.0).abs() < 1.5, "mean {mean}");
        // determinism: same seed -> same draws
        let mut tf2 = TargetForecast::new("X", "close");
        tf2.quantiles = Some(Block::native(vec![1, 3], vec![0.0, 10.0, 20.0]));
        tf2.levels = vec![0.1, 0.5, 0.9];
        ensure_representations(
            &mut tf2,
            Representation::Quantiles,
            &[Representation::Samples],
            &[0.1, 0.5, 0.9],
            1000,
            42,
        )
        .unwrap();
        assert_eq!(tf.samples, tf2.samples);
    }

    #[test]
    fn gaussian_distribution_decodes_to_quantiles_exactly() {
        let mut tf = TargetForecast::new("X", "close");
        // horizon 1, mu=0 sigma=1
        tf.distribution = Some(Block::native(vec![1, 2], vec![0.0, 1.0]));
        tf.dist_family = "gaussian".into();
        ensure_representations(
            &mut tf,
            Representation::Distribution,
            &[Representation::Quantiles],
            &[0.5, 0.975],
            0,
            0,
        )
        .unwrap();
        let q = tf.quantiles.as_ref().unwrap();
        assert!((q.data[0] - 0.0).abs() < 1e-4); // median
        assert!((q.data[1] - 1.959_964).abs() < 1e-2); // the 0.975 quantile ~ 1.96
        assert_eq!(q.method, "gaussian_ppf");
    }

    #[test]
    fn norm_ppf_hits_known_points() {
        assert!(norm_ppf(0.5).abs() < 1e-4);
        assert!((norm_ppf(0.975) - 1.959_964).abs() < 1e-3);
        assert!((norm_ppf(0.025) + 1.959_964).abs() < 1e-3);
    }

    #[test]
    fn already_present_representation_is_left_untouched() {
        let mut tf = tf_with_samples();
        // ask for samples when samples is native+present: no-op, stays native
        ensure_representations(
            &mut tf,
            Representation::Samples,
            &[Representation::Samples],
            &[],
            0,
            0,
        )
        .unwrap();
        assert!(!tf.samples.as_ref().unwrap().derived);
        let _ = Forecast::new("m", Representation::Samples, 1, "1d");
    }
}
