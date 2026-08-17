// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Statistical forecasting baselines — the controls every foundation model must
//! beat before its skill is believed.
//!
//! Each is a [`ForecastModel`] that forecasts every target variate
//! independently as a per-step Gaussian `N(mu_h, sigma_h^2)` (native
//! representation [`Representation::Distribution`]), from which the honesty
//! layer derives quantiles / samples / point on request. They are deliberately
//! univariate and covariate-blind — that is the whole point of a baseline.
//!
//! - [`RandomWalk`] — last value flat, `sigma_h = sigma_1 * sqrt(h)`. The
//!   canonical control: on a true random walk *nothing should beat this*.
//! - [`SeasonalNaive`] — repeat the last season.
//! - [`Drift`] — extrapolate the mean per-step change (a line).
//! - [`Arima`] — AR(p) on the `d`-times-differenced series by OLS, then
//!   re-integrate. `(p, d, 0)` — the MA term is future work (documented).
//! - [`Garch11`] — random-walk mean with a GARCH(1,1) conditional-variance
//!   forecast on the returns; the classical volatility baseline.

use crate::util;
use forecast::{
    convert, Block, Capabilities, CovariateSupport, Forecast, ForecastError, ForecastModel,
    ForecastSpec, Panel, Representation, TargetForecast,
};

/// Assemble a Gaussian [`Forecast`] from a per-target `(mus, sigmas)` closure.
fn gaussian_forecast(
    name: &str,
    panel: &Panel,
    spec: &ForecastSpec,
    per_target: impl Fn(&[f32], usize) -> (Vec<f32>, Vec<f32>),
) -> Result<Forecast, ForecastError> {
    let mut fc = Forecast::new(name, Representation::Distribution, spec.horizon, &panel.freq);
    for it in &panel.items {
        for tgt in it.targets() {
            let (mus, sigmas) = per_target(&tgt.data, spec.horizon);
            let mut data = vec![0.0f32; spec.horizon * 2];
            for h in 0..spec.horizon {
                data[h * 2] = *mus.get(h).unwrap_or(&0.0);
                // a strictly positive floor keeps the distribution non-degenerate
                data[h * 2 + 1] = sigmas.get(h).copied().unwrap_or(0.0).max(1e-9);
            }
            let mut tf = TargetForecast::new(&it.item_id, &tgt.name);
            tf.distribution = Some(Block::native(vec![spec.horizon, 2], data));
            tf.dist_family = "gaussian".into();
            convert::ensure_representations(
                &mut tf,
                Representation::Distribution,
                &spec.representations,
                &spec.quantile_levels,
                spec.num_samples,
                spec.seed,
            )?;
            fc.targets.push(tf);
        }
    }
    Ok(fc)
}

/// The baseline capability template: unbounded context/horizon, univariate,
/// covariate-blind, Gaussian distribution native.
fn caps(name: &str) -> Capabilities {
    Capabilities {
        name: name.to_string(),
        max_context: usize::MAX,
        max_horizon: None,
        native_representation: Representation::Distribution,
        covariates: CovariateSupport::None,
        supports_known_future: false,
        multivariate: false,
        arbitrary_quantile_levels: true,
        stochastic: false,
        requires_variates: vec![],
    }
}

/// Baselines forecast targets only; covariates are legitimately N/A, so they
/// override the default validate (which would reject a covariate-bearing panel)
/// to just require at least one target with usable history.
fn validate_targets(panel: &Panel) -> Result<(), ForecastError> {
    if panel.items.is_empty() {
        return Err(ForecastError::bad_request("empty panel"));
    }
    let has_target = panel.items.iter().any(|it| it.targets().next().is_some());
    if !has_target {
        return Err(ForecastError::bad_request("no target variates to forecast"));
    }
    Ok(())
}

// ---- random walk / naive ---------------------------------------------------

/// Last-value ("naive") forecast with random-walk uncertainty.
pub struct RandomWalk;

impl ForecastModel for RandomWalk {
    fn capabilities(&self) -> Capabilities {
        caps("naive")
    }
    fn validate(&self, panel: &Panel, _spec: &ForecastSpec) -> Result<(), ForecastError> {
        validate_targets(panel)
    }
    fn forecast(&self, panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError> {
        self.validate(panel, spec)?;
        gaussian_forecast("naive", panel, spec, |c, h| {
            let last = c.last().copied().unwrap_or(0.0);
            let s1 = util::diff_std(c);
            let mus = vec![last; h];
            let sigmas = (0..h).map(|k| s1 * ((k + 1) as f32).sqrt()).collect();
            (mus, sigmas)
        })
    }
}

// ---- seasonal naive --------------------------------------------------------

/// Repeat the last full season; uncertainty from the seasonal differences.
pub struct SeasonalNaive {
    /// Season length in steps (e.g. 5 or 7 for daily bars, 24 for hourly).
    pub season: usize,
}

impl ForecastModel for SeasonalNaive {
    fn capabilities(&self) -> Capabilities {
        caps("seasonal_naive")
    }
    fn validate(&self, panel: &Panel, _spec: &ForecastSpec) -> Result<(), ForecastError> {
        validate_targets(panel)
    }
    fn forecast(&self, panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError> {
        self.validate(panel, spec)?;
        let m = self.season.max(1);
        gaussian_forecast("seasonal_naive", panel, spec, |c, h| {
            let n = c.len();
            let last = c.last().copied().unwrap_or(0.0);
            // seasonal innovation scale: std of y[t]-y[t-m]
            let sdiff: Vec<f32> =
                (m..n).map(|t| c[t] - c[t - m]).collect();
            let s1 = util::std(&sdiff);
            let mut mus = vec![0.0f32; h];
            let mut sigmas = vec![0.0f32; h];
            for k in 0..h {
                mus[k] = if n >= m { c[n - m + (k % m)] } else { last };
                // variance grows once per completed season cycle
                let cycles = (k / m + 1) as f32;
                sigmas[k] = s1 * cycles.sqrt();
            }
            (mus, sigmas)
        })
    }
}

// ---- drift -----------------------------------------------------------------

/// Extrapolate the average per-step change (a straight line through first and
/// last points); uncertainty from residuals about the drift line.
pub struct Drift;

impl ForecastModel for Drift {
    fn capabilities(&self) -> Capabilities {
        caps("drift")
    }
    fn validate(&self, panel: &Panel, _spec: &ForecastSpec) -> Result<(), ForecastError> {
        validate_targets(panel)
    }
    fn forecast(&self, panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError> {
        self.validate(panel, spec)?;
        gaussian_forecast("drift", panel, spec, |c, h| {
            let n = c.len();
            let last = c.last().copied().unwrap_or(0.0);
            let slope = if n >= 2 { (c[n - 1] - c[0]) / (n - 1) as f32 } else { 0.0 };
            // residual scale about the drift line, per step
            let s1 = if n >= 2 {
                let resid: Vec<f32> = (1..n).map(|t| (c[t] - c[t - 1]) - slope).collect();
                util::std(&resid)
            } else {
                0.0
            };
            let mus = (0..h).map(|k| last + (k + 1) as f32 * slope).collect();
            let sigmas = (0..h).map(|k| s1 * ((k + 1) as f32).sqrt()).collect();
            (mus, sigmas)
        })
    }
}

// ---- ARIMA(p, d, 0) --------------------------------------------------------

/// AR(p) on the `d`-times-differenced series (fit by OLS), re-integrated. The MA
/// term is not modelled — this is `(p, d, 0)`, adequate as a classical control.
pub struct Arima {
    /// Autoregressive order.
    pub p: usize,
    /// Differencing order.
    pub d: usize,
}

impl Arima {
    /// Fit AR(p) coefficients `[c0, a1..ap]` on `z` by ordinary least squares.
    fn fit_ar(z: &[f32], p: usize) -> Option<Vec<f32>> {
        let n = z.len();
        if n <= p + 1 || p == 0 {
            return None;
        }
        // design: rows t=p..n, columns [1, z[t-1], .., z[t-p]]
        let k = p + 1;
        let rows = n - p;
        let mut xtx = vec![0.0f32; k * k];
        let mut xty = vec![0.0f32; k];
        for t in p..n {
            let mut row = vec![1.0f32; k];
            for j in 1..k {
                row[j] = z[t - j];
            }
            let y = z[t];
            for a in 0..k {
                xty[a] += row[a] * y;
                for b in 0..k {
                    xtx[a * k + b] += row[a] * row[b];
                }
            }
        }
        let _ = rows;
        util::solve(&xtx, &xty, k)
    }
}

impl ForecastModel for Arima {
    fn capabilities(&self) -> Capabilities {
        caps("arima")
    }
    fn validate(&self, panel: &Panel, _spec: &ForecastSpec) -> Result<(), ForecastError> {
        validate_targets(panel)
    }
    fn forecast(&self, panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError> {
        self.validate(panel, spec)?;
        let (p, d) = (self.p.max(1), self.d);
        gaussian_forecast("arima", panel, spec, move |c, h| {
            let z = util::diff_n(c, d);
            let last = c.last().copied().unwrap_or(0.0);
            let coef = Self::fit_ar(&z, p);
            // one-step residual scale on the differenced series
            let s1 = util::std(&z).max(util::diff_std(c));
            let (mus, sigmas) = match coef {
                Some(coef) => {
                    // recursively forecast the differenced series
                    let mut hist: Vec<f32> = z.clone();
                    let mut zf = Vec::with_capacity(h);
                    for _ in 0..h {
                        let mut yhat = coef[0];
                        for (j, cj) in coef.iter().enumerate().skip(1) {
                            let idx = hist.len() as isize - j as isize;
                            let v = if idx >= 0 { hist[idx as usize] } else { 0.0 };
                            yhat += cj * v;
                        }
                        zf.push(yhat);
                        hist.push(yhat);
                    }
                    // re-integrate d times (only d in {0,1} exercised; d=1 is a cumsum)
                    let mus = integrate(&zf, c, d, last);
                    let sigmas = (0..h).map(|k| s1 * ((k + 1) as f32).sqrt()).collect();
                    (mus, sigmas)
                }
                None => {
                    // fall back to random walk if the fit is degenerate
                    let mus = vec![last; h];
                    let sigmas = (0..h).map(|k| s1 * ((k + 1) as f32).sqrt()).collect();
                    (mus, sigmas)
                }
            };
            (mus, sigmas)
        })
    }
}

/// Re-integrate a `d`-differenced forecast back to the original level. Only
/// `d <= 1` is fully supported (d=1 is a cumulative sum anchored at `last`); for
/// `d = 0` the forecast is the level itself.
fn integrate(zf: &[f32], _ctx: &[f32], d: usize, last: f32) -> Vec<f32> {
    match d {
        0 => zf.to_vec(),
        _ => {
            // d>=1: undo one difference by cumulative sum from `last`; higher d
            // is approximated by repeated cumsum (adequate for a baseline).
            let mut level = last;
            let mut out = Vec::with_capacity(zf.len());
            for &dz in zf {
                level += dz;
                out.push(level);
            }
            out
        }
    }
}

// ---- GARCH(1,1) volatility -------------------------------------------------

/// Random-walk mean with a GARCH(1,1) conditional-variance forecast on the
/// returns. Parameters are estimated by variance targeting plus a small
/// `(alpha, beta)` grid search maximising the Gaussian log-likelihood — cheap
/// and deterministic, not full MLE. The forecast interval widens with the
/// GARCH volatility path rather than a fixed `sqrt(h)`.
pub struct Garch11;

impl Garch11 {
    /// Estimate `(omega, alpha, beta)` on returns `r` by variance targeting and a
    /// coarse grid on persistence. Returns the unconditional variance too.
    fn fit(r: &[f32]) -> (f32, f32, f32, f32) {
        let var = {
            let m = util::mean(r);
            r.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / (r.len().max(1) as f32)
        };
        if r.len() < 8 || var <= 0.0 {
            return (var.max(1e-12), 0.0, 0.0, var.max(1e-12));
        }
        let mut best = (f32::INFINITY, 0.05f32, 0.90f32);
        // grid over (alpha, beta) with alpha+beta < 1 (stationarity)
        let alphas = [0.02f32, 0.05, 0.10, 0.15, 0.20];
        let betas = [0.70f32, 0.80, 0.85, 0.90, 0.95];
        for &a in &alphas {
            for &b in &betas {
                if a + b >= 0.999 {
                    continue;
                }
                let omega = (1.0 - a - b) * var; // variance targeting
                let nll = Self::nll(r, omega, a, b, var);
                if nll < best.0 {
                    best = (nll, a, b);
                }
            }
        }
        let (_, a, b) = best;
        ((1.0 - a - b) * var, a, b, var)
    }

    /// Gaussian negative log-likelihood of the GARCH(1,1) recursion on `r`.
    fn nll(r: &[f32], omega: f32, alpha: f32, beta: f32, var0: f32) -> f32 {
        let mut s2 = var0.max(1e-12);
        let mut acc = 0.0f32;
        for &e in r {
            // 0.5*(ln s2 + e^2/s2)
            acc += 0.5 * (s2.ln() + (e * e) / s2);
            s2 = omega + alpha * e * e + beta * s2;
            if s2 < 1e-12 {
                s2 = 1e-12;
            }
        }
        acc
    }
}

impl ForecastModel for Garch11 {
    fn capabilities(&self) -> Capabilities {
        caps("garch")
    }
    fn validate(&self, panel: &Panel, _spec: &ForecastSpec) -> Result<(), ForecastError> {
        validate_targets(panel)
    }
    fn forecast(&self, panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError> {
        self.validate(panel, spec)?;
        gaussian_forecast("garch", panel, spec, |c, h| {
            let last = c.last().copied().unwrap_or(0.0);
            let r = util::diff(c);
            let (omega, alpha, beta, uncond) = Garch11::fit(&r);
            // current conditional variance from the recursion over the returns
            let mut s2 = uncond.max(1e-12);
            for &e in &r {
                s2 = omega + alpha * e * e + beta * s2;
            }
            // multi-step variance forecast + cumulative variance of the sum of
            // returns (price random walk): Var(price_{t+h}) = sum_{k<=h} E[s2_{t+k}]
            let mut cum = 0.0f32;
            let mut fwd = s2;
            let mut sigmas = Vec::with_capacity(h);
            for _ in 0..h {
                cum += fwd;
                sigmas.push(cum.max(1e-18).sqrt());
                // E[s2_{t+k+1}] = omega + (alpha+beta) * E[s2_{t+k}]
                fwd = omega + (alpha + beta) * fwd;
            }
            let mus = vec![last; h];
            (mus, sigmas)
        })
    }
}

/// The default baseline set used in comparisons. Seasonal period defaults to 1
/// (plain naive) — callers with a known period construct [`SeasonalNaive`]
/// directly.
pub fn default_set() -> Vec<Box<dyn ForecastModel>> {
    vec![
        Box::new(RandomWalk),
        Box::new(Drift),
        Box::new(Arima { p: 2, d: 1 }),
        Box::new(Garch11),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use forecast::{Item, Variate};

    fn panel_of(series: Vec<f32>) -> Panel {
        Panel::single("1d", "X", vec![Variate::target("y", series)])
    }

    fn spec(h: usize) -> ForecastSpec {
        ForecastSpec {
            horizon: h,
            representations: vec![Representation::Quantiles, Representation::Point],
            quantile_levels: vec![0.1, 0.5, 0.9],
            num_samples: 0,
            seed: 0,
        }
    }

    fn median_path(fc: &Forecast) -> Vec<f32> {
        // levels [0.1,0.5,0.9] -> median is column index 1
        let q = fc.targets[0].quantiles.as_ref().unwrap();
        let h = q.shape[0];
        (0..h).map(|t| q.data[t * 3 + 1]).collect()
    }

    #[test]
    fn naive_forecasts_last_value_flat() {
        let fc = RandomWalk.forecast(&panel_of(vec![1.0, 2.0, 3.0, 7.0]), &spec(3)).unwrap();
        let m = fc.targets[0].mean.as_ref().unwrap();
        assert_eq!(m.data, vec![7.0, 7.0, 7.0]);
        // quantile median equals the last value too
        for v in median_path(&fc) {
            assert!((v - 7.0).abs() < 1e-4);
        }
    }

    #[test]
    fn naive_interval_widens_with_horizon() {
        let fc = RandomWalk.forecast(&panel_of(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]), &spec(4)).unwrap();
        let q = fc.targets[0].quantiles.as_ref().unwrap();
        // width of the 10-90 interval must be non-decreasing in the horizon
        let width = |t: usize| q.data[t * 3 + 2] - q.data[t * 3];
        assert!(width(3) > width(0), "interval should widen: {} vs {}", width(0), width(3));
    }

    #[test]
    fn seasonal_naive_repeats_the_season() {
        // period 2: ...,5,9 -> forecast 5,9,5,9
        let p = panel_of(vec![5.0, 9.0, 5.0, 9.0, 5.0, 9.0]);
        let fc = SeasonalNaive { season: 2 }.forecast(&p, &spec(4)).unwrap();
        let med = median_path(&fc);
        assert!((med[0] - 5.0).abs() < 1e-3);
        assert!((med[1] - 9.0).abs() < 1e-3);
        assert!((med[2] - 5.0).abs() < 1e-3);
        assert!((med[3] - 9.0).abs() < 1e-3);
    }

    #[test]
    fn drift_extrapolates_a_line_exactly() {
        // exact line slope 2 -> next points 12,14,16, sigma ~ 0
        let fc = Drift.forecast(&panel_of(vec![2.0, 4.0, 6.0, 8.0, 10.0]), &spec(3)).unwrap();
        let med = median_path(&fc);
        assert!((med[0] - 12.0).abs() < 1e-3, "{med:?}");
        assert!((med[1] - 14.0).abs() < 1e-3);
        assert!((med[2] - 16.0).abs() < 1e-3);
    }

    #[test]
    fn arima_ar1_forecast_decays_toward_mean() {
        // AR(1) around mean 0 with phi ~ 0.6: x_t = 0.6 x_{t-1} + noise (use a
        // clean decaying sequence so OLS recovers a positive phi < 1)
        let mut s = vec![1.0f32];
        for _ in 0..40 {
            let last = *s.last().unwrap();
            s.push(0.6 * last);
        }
        // d=0 AR(2) fit; forecast should keep decaying toward 0
        let fc = Arima { p: 2, d: 0 }.forecast(&panel_of(s), &spec(3)).unwrap();
        let med = median_path(&fc);
        assert!(med[0] > 0.0 && med[0] < 0.3, "decay step 0: {med:?}");
        assert!(med[1] < med[0] + 1e-3, "monotone decay: {med:?}");
    }

    #[test]
    fn garch_produces_widening_nonnegative_vol() {
        // alternating returns -> nonzero vol; sigma must grow and stay finite
        let mut s = vec![100.0f32];
        for i in 0..60 {
            let r = if i % 2 == 0 { 1.0 } else { -1.0 };
            s.push(s.last().unwrap() + r);
        }
        let fc = Garch11.forecast(&panel_of(s), &spec(5)).unwrap();
        let q = fc.targets[0].quantiles.as_ref().unwrap();
        let sig = |t: usize| q.data[t * 3 + 2] - q.data[t * 3 + 1]; // 0.9 - median
        assert!(sig(0) > 0.0, "vol must be positive");
        assert!(sig(4) >= sig(0), "cumulative vol must not shrink");
        assert!(sig(4).is_finite());
    }

    #[test]
    fn baselines_reject_empty_and_targetless_panels() {
        let empty = Panel { freq: "1d".into(), start: None, items: vec![] };
        assert!(RandomWalk.forecast(&empty, &spec(1)).is_err());
        let no_target = Panel {
            freq: "1d".into(),
            start: None,
            items: vec![Item::new(
                "X",
                vec![Variate {
                    name: "vol".into(),
                    role: forecast::Role::PastCovariate,
                    kind: forecast::Kind::Continuous,
                    data: vec![1.0; 5],
                    future: None,
                    observed: None,
                    cardinality: None,
                }],
            )],
        };
        assert!(RandomWalk.forecast(&no_target, &spec(1)).is_err());
    }

    #[test]
    fn baselines_tolerate_covariates_by_ignoring_them() {
        // unlike the default validate, a baseline accepts a covariate-bearing
        // panel and just forecasts the target.
        let p = Panel::single(
            "1d",
            "X",
            vec![
                Variate::target("close", vec![1.0, 2.0, 3.0]),
                Variate {
                    name: "vol".into(),
                    role: forecast::Role::PastCovariate,
                    kind: forecast::Kind::Continuous,
                    data: vec![10.0, 11.0, 12.0],
                    future: None,
                    observed: None,
                    cardinality: None,
                },
            ],
        );
        let fc = RandomWalk.forecast(&p, &spec(2)).unwrap();
        assert_eq!(fc.targets.len(), 1);
        assert_eq!(fc.targets[0].name, "close");
    }
}
