// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forecasting scenarios as first-class [`Benchmark`]s.
//!
//! Each `fcbench` [`Scenario`] (a data-generating process with known structure)
//! becomes a benchmark named `forecast_<scenario>`, so the forecasting battery
//! flows through the same `bench eval` / `bench compare` / axes / advisor
//! machinery as the LM and algorithmic benchmarks (plan §4.1). They all map to
//! the `forecasting` capability axis.
//!
//! Like [`crate::mad_compress`], the objective is **not** a causal next-token
//! decoder, so these benchmarks IGNORE the supplied `lm`: a forecast benchmark's
//! score reflects *how much learnable structure the scenario has*, not a decoder
//! architecture. They are therefore [`informational`](Benchmark::informational)
//! (reported, never gating the arch suite). The authoritative pass/fail path for
//! forecasting — including the hard negative-control gate — is the dedicated
//! `brain forecast compare` harness.
//!
//! ## Headline score
//! A 0..1 **skill score** = `clamp(1 − MAE_best / MAE_naive, 0, 1)`: how much the
//! best *structured* forecast beats the dumb last-value (naive) forecast, averaged
//! over the windows. The structured candidates are the closed-form oracle (when
//! the scenario has one), a seasonal-naive repeat, and a linear-drift line — no
//! model fitting, so the score is a clean "how much exploitable structure is
//! here?" probe. On the **random-walk** negative control nothing beats last-value,
//! so skill stays ≈ 0 (the property the harness's dedicated gate enforces hard).

use std::path::Path;

use crate::{Benchmark, DecoderLm, Metrics};
use fcbench::scenarios::{self, Scenario};

/// Mean absolute error of a point forecast against the realized future.
fn mae(pred: &[f32], actual: &[f32]) -> f32 {
    let n = pred.len().min(actual.len());
    if n == 0 {
        return f32::NAN;
    }
    pred.iter().zip(actual).map(|(p, a)| (p - a).abs()).sum::<f32>() / n as f32
}

/// One forecasting scenario wrapped as a [`Benchmark`].
pub struct ForecastBench {
    scenario: Box<dyn Scenario>,
    name: String,
    desc: String,
    /// Number of seeded windows averaged for the score.
    pub windows: usize,
}

impl ForecastBench {
    /// Wrap a scenario; the benchmark id is `forecast_<scenario name>`.
    pub fn new(scenario: Box<dyn Scenario>) -> ForecastBench {
        let control = if scenario.is_negative_control() { ", negative control" } else { "" };
        let name = format!("forecast_{}", scenario.name());
        let desc = format!("forecasting {} process [{}{}]", scenario.name(), scenario.axis(), control);
        ForecastBench { scenario, name, desc, windows: 64 }
    }

    /// The naive (last-value) forecast — the baseline skill is measured against.
    fn naive_forecast(&self, context: &[f32]) -> Vec<f32> {
        vec![context.last().copied().unwrap_or(0.0); self.scenario.horizon()]
    }

    /// Structured candidate forecasts to beat the naive one: the closed-form
    /// oracle (if the scenario exposes one), a seasonal-naive repeat, and a
    /// linear-drift extrapolation. The best (lowest MAE) is what scores.
    fn structured_forecasts(&self, context: &[f32]) -> Vec<Vec<f32>> {
        let h = self.scenario.horizon();
        let n = context.len();
        let last = context.last().copied().unwrap_or(0.0);
        let mut out: Vec<Vec<f32>> = Vec::new();

        if let Some(oracle) = self.scenario.oracle(context) {
            if oracle.len() == h {
                out.push(oracle);
            }
        }
        // seasonal-naive: repeat the last full season.
        let season = self.scenario.season();
        if season > 1 && n >= season {
            out.push((0..h).map(|i| context[n - season + (i % season)]).collect());
        }
        // linear drift: last value + in-sample average slope.
        if n >= 2 {
            let slope = (last - context[0]) / (n as f32 - 1.0);
            out.push((1..=h).map(|k| last + slope * k as f32).collect());
        }
        out
    }
}

/// Flat f32-LE layout: `windows` records of `context_len + horizon` values each.
const DATASET: &str = "windows.f32";

impl Benchmark for ForecastBench {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.desc
    }

    fn prepare(&self, dir: &Path, seed: u64) -> std::io::Result<()> {
        let mut buf: Vec<u8> = Vec::new();
        for i in 0..self.windows {
            let w = self.scenario.generate(seed.wrapping_add(i as u64));
            for &x in w.context.iter().chain(w.future.iter()) {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        std::fs::write(dir.join(DATASET), buf)
    }

    /// Non-LM objective: the decoder `lm` is ignored (documented above). Score the
    /// best structured forecast's skill vs naive, averaged over the windows.
    fn evaluate_with(&self, _lm: &dyn DecoderLm, dir: &Path, _seed: u64) -> std::io::Result<Metrics> {
        let ctx = self.scenario.context_len();
        let h = self.scenario.horizon();
        let stride = ctx + h;

        let bytes = std::fs::read(dir.join(DATASET))?;
        let vals: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let (mut naive_sum, mut best_sum, mut count) = (0.0f32, 0.0f32, 0usize);
        for rec in vals.chunks_exact(stride) {
            let (context, future) = rec.split_at(ctx);
            let en = mae(&self.naive_forecast(context), future);
            let eb = self
                .structured_forecasts(context)
                .iter()
                .map(|p| mae(p, future))
                .fold(en, f32::min); // never worse than naive itself
            if en.is_finite() && eb.is_finite() {
                naive_sum += en;
                best_sum += eb;
                count += 1;
            }
        }
        let (mae_naive, mae_best) = if count > 0 {
            (naive_sum / count as f32, best_sum / count as f32)
        } else {
            (f32::NAN, f32::NAN)
        };
        // Fraction of naive error removed by the best structured forecast.
        let skill = if mae_naive > 1e-9 { (1.0 - mae_best / mae_naive).clamp(0.0, 1.0) } else { 0.0 };
        Ok(Metrics::new(skill)
            .with("mae_naive", mae_naive)
            .with("mae_best", mae_best)
            .with("windows", count as f32))
    }

    fn threshold(&self) -> f32 {
        // Structured scenarios should show real skill; the negative control must
        // not (its bar is 0 — informational, so this is a reference line, not a
        // gate; the hard gate lives in `brain forecast compare`).
        if self.scenario.is_negative_control() {
            0.0
        } else {
            0.10
        }
    }

    fn report_fields(&self) -> Vec<&str> {
        vec!["mae_naive", "mae_best", "windows"]
    }

    fn informational(&self) -> bool {
        true
    }
}

/// The forecasting battery as benchmarks, one per `fcbench` scenario.
pub fn forecast_benchmarks() -> Vec<Box<dyn Benchmark>> {
    scenarios::default_battery()
        .into_iter()
        .map(|s| Box::new(ForecastBench::new(s)) as Box<dyn Benchmark>)
        .collect()
}

/// Build one forecast benchmark by its `forecast_<scenario>` name at a chosen
/// window budget. Used by capscale to construct the forecasting probe.
pub fn build(name: &str, windows: usize) -> Option<Box<dyn Benchmark>> {
    scenarios::default_battery()
        .into_iter()
        .find(|s| format!("forecast_{}", s.name()) == name)
        .map(|s| {
            let mut b = ForecastBench::new(s);
            b.windows = windows;
            Box::new(b) as Box<dyn Benchmark>
        })
}

/// The forecasting battery with a slashed window count for the smoke registry.
pub fn forecast_benchmarks_smoke() -> Vec<Box<dyn Benchmark>> {
    scenarios::default_battery()
        .into_iter()
        .map(|s| {
            let mut b = ForecastBench::new(s);
            b.windows = 8;
            Box::new(b) as Box<dyn Benchmark>
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score_of(bench: &dyn Benchmark, seed: u64) -> Metrics {
        // Unique dir per call: parallel tests can score the same scenario, and a
        // shared path would let one test's cleanup wipe another's dataset mid-read.
        use std::sync::atomic::{AtomicU64, Ordering};
        static NONCE: AtomicU64 = AtomicU64::new(0);
        let uniq = NONCE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join(format!("fcbench_test_{}_{}_{}", bench.name(), std::process::id(), uniq));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        bench.prepare(&dir, seed).unwrap();
        let m = bench.evaluate(&dir, seed).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        m
    }

    #[test]
    fn every_forecast_benchmark_is_named_and_informational() {
        for b in forecast_benchmarks() {
            assert!(b.name().starts_with("forecast_"), "name: {}", b.name());
            assert!(b.informational(), "forecast benchmarks ignore the arch → informational");
        }
    }

    #[test]
    fn structured_scenarios_show_skill_negative_control_does_not() {
        let benches = forecast_benchmarks();
        let get = |n: &str| benches.iter().find(|b| b.name() == n).expect("scenario present");

        // A scenario with obvious structure (seasonal + trend): the seasonal-naive
        // repeat removes a clear chunk of the naive error → real skill.
        let trend = score_of(get("forecast_seasonal_trend").as_ref(), 42);
        assert!(trend.score > 0.15, "seasonal_trend should show clear skill: {trend:?}");

        // The random-walk negative control has no exploitable structure → ~0 skill.
        let rw = score_of(get("forecast_random_walk").as_ref(), 42);
        assert!(rw.score < 0.1, "random walk must show ~no skill (negative control): {rw:?}");
        // And skill(trend) must dominate skill(random walk) by a clear margin.
        assert!(trend.score > rw.score + 0.1, "structured must beat control: {trend:?} vs {rw:?}");
    }

    #[test]
    fn score_is_deterministic_in_seed() {
        let b = &forecast_benchmarks()[0];
        assert_eq!(score_of(b.as_ref(), 7).score, score_of(b.as_ref(), 7).score);
    }
}
