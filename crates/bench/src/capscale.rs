// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Per-capability predictive scaling** — sweep a single architecture across a
//! grid of model SIZES and, for *each capability axis*, fit how that axis's score
//! improves with parameter count `N`, then extrapolate the predicted score at
//! larger `N`.
//!
//! This is the predictive half of the eval harness. The plain
//! [`scaling`](crate::scaling) sweep answers "how does *training loss* on one task
//! fall as the model grows?"; this module instead asks, *per capability*, "how
//! does the benchmark *score* on this axis improve as the model grows, and where
//! does it plateau?" — so when a new architecture is registered you can predict
//! how each of its capabilities will respond to more capacity *before* paying for
//! the bigger run.
//!
//! ## What it does
//! For one architecture and a small SIZE grid (≥3 points, increasing params via
//! [`ScaledGpt`]/[`Size`](crate::arch::Size)):
//! 1. pick **one representative benchmark per axis** (the cheapest informative one,
//!    documented in [`representative_bench`]),
//! 2. at each size, train + score that benchmark (a smoke-like reduced budget),
//! 3. per axis, fit a **saturating trend** `score(N) ≈ ceil − A·N^(−β)` (scores
//!    rise toward a ceiling, so we fit the *gap to a ceiling* as a power law),
//! 4. record the slope/exponent **β**, the fit **R²**, an extrapolated predicted
//!    score at **2×** and **4×** the largest `N`, and a coarse **verdict**
//!    (improving / saturating / flat).
//!
//! The output [`CapScaleReport`] is serialized to
//! `results/scale-<arch>-<seed>.json` and consumed by the [`advisor`](crate::advisor).
//!
//! ## The sweep DIMENSION (`Knob`) — wiring experts in later
//! Today the only swept dimension is [`Knob::Size`] (the GPT family's
//! depth/width). A future Mixture-of-Experts `DecoderLm` will want the *same*
//! machinery applied to a **number-of-experts** axis: train at experts ∈ {2,4,8},
//! fit score(experts). The sweep is written against a generic [`Knob`] so that
//! slots in without re-plumbing the fit/extrapolation/advisor — see the
//! `// TODO(experts)` markers in [`grid_for`] and [`Knob`]. We do NOT implement
//! MoE scoring here (no MoE arch is registered yet).
//!
//! ## Budget
//! 3 sizes × 6 axes (one benchmark each) at a smoke step budget. On the CPU
//! (Cranelift) backend this targets a few minutes total; see [`CapScaleConfig`].

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::arch::{self, ScaledGpt, Size};
use crate::axes::axes;
use crate::scaling::ols;
use crate::Benchmark;

/// The dimension being swept. `Size` is wired today (GPT depth/width). `Experts`
/// is reserved for a future Mixture-of-Experts `DecoderLm`; the fit + advisor are
/// dimension-agnostic, so activating it is registering the arch + filling in
/// [`grid_for`]'s `// TODO(experts)` branch — no other change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Knob {
    /// Model size: depth / width / heads (the GPT family). The independent axis is
    /// the resulting parameter count `N`.
    Size,
    // TODO(experts): a Mixture-of-Experts arch sweeps the number of experts here.
    // The independent axis is still "params N" (experts multiply the FFN param
    // count), so `fit_saturating` / the advisor need no change — only `grid_for`
    // grows an `Experts` branch that returns per-expert-count `ScaledGpt`-like
    // factories once a MoE `DecoderLm` exists.
    #[allow(dead_code)]
    Experts,
}

impl Knob {
    /// Human label used in artifacts / CLI output.
    pub fn label(&self) -> &'static str {
        match self {
            Knob::Size => "size",
            Knob::Experts => "experts",
        }
    }
}

/// Configuration for a per-capability scaling sweep.
#[derive(Clone, Debug)]
pub struct CapScaleConfig {
    /// Which dimension to sweep (today: [`Knob::Size`]).
    pub knob: Knob,
    /// Training steps per (size, benchmark) point — a smoke-like budget so the
    /// whole 3×6 grid finishes in a few minutes on the CPU backend.
    pub steps: u32,
    /// Synthetic corpus size per benchmark (kept small for speed).
    pub n_sequences: usize,
    /// Eval-set size per benchmark (kept small for speed).
    pub eval_sequences: usize,
    pub seed: u64,
}

impl Default for CapScaleConfig {
    /// A smoke-like budget: ~60 steps over a small corpus. The grid is 3 sizes ×
    /// 6 axes = 18 short training runs; this targets a few minutes total on the
    /// CPU (Cranelift) backend. Scores are coarse (low budget) — the *shape* of
    /// the per-axis curve and its extrapolation are the deliverable, not an
    /// absolute leaderboard number.
    fn default() -> Self {
        CapScaleConfig {
            knob: Knob::Size,
            steps: 60,
            n_sequences: 800,
            eval_sequences: 60,
            seed: 1337,
        }
    }
}

/// The SIZE grid for the sweep, in increasing-capacity order. Three points so the
/// fit has a real trend; width grows 32→64→96 with matched depth/heads (head_dim
/// 16 throughout: 32/2, 64/4, 96/6) so params rise clearly. Mirrors the
/// `scaling::Sweep` default grid for consistency.
pub const SIZE_GRID: &[(u32, u32, u32)] = &[(1, 32, 2), (2, 64, 4), (3, 96, 6)];

/// Build the list of `(label, Size)` points for the requested knob.
fn grid_for(knob: Knob) -> Vec<(String, Size)> {
    match knob {
        Knob::Size => SIZE_GRID
            .iter()
            .map(|&(l, d, h)| (format!("L{l}xD{d}xH{h}"), Size::fixed(l, d, h)))
            .collect(),
        // TODO(experts): return one point per expert count once a MoE arch is
        // registered, e.g. experts ∈ {2,4,8} each mapped to a MoE `DecoderLm`
        // factory. Until then this dimension is unreachable (no MoE arch).
        Knob::Experts => Vec::new(),
    }
}

/// The representative benchmark for each capability axis — the *cheapest
/// informative* one, so the 3×6 grid stays within a few-minute budget.
///
/// Rationale per axis (one benchmark each, smallest sequence/corpus that still
/// responds to capacity):
/// - **recall** → `mqar`: the canonical recall probe; cheap sequences and a
///   recall score that climbs clearly with capacity.
/// - **copying** → `mad_selective_copy`: a copy/route task that is cheaper than
///   `toolcall` (no structured tool grammar) yet still capacity-sensitive.
/// - **memory** → `mad_memorize`: the only memory benchmark.
/// - **state_tracking** → `parity`: cheaper than `dyck` (binary state, shorter
///   sequences) and a clean accuracy curve.
/// - **compression** → `mad_compress`: the only compression benchmark (note: it
///   trains its own autoencoder and ignores `lm`, so its curve reflects budget,
///   not the swept arch — documented; still reported for completeness).
/// - **arithmetic** → `mod_add`: the only arithmetic benchmark (informational;
///   grokking, so its curve is high-variance at a smoke budget — reported as a
///   diagnostic).
/// - **forecasting** → `forecast_seasonal_trend`: the forecasting probe ignores
///   the swept decoder entirely (its skill is a property of the scenario, not the
///   arch), so its curve is flat — informational, reported for completeness.
pub fn representative_bench(axis: &str) -> Option<&'static str> {
    match axis {
        "recall" => Some("mqar"),
        "copying" => Some("mad_selective_copy"),
        "memory" => Some("mad_memorize"),
        "state_tracking" => Some("parity"),
        "compression" => Some("mad_compress"),
        "arithmetic" => Some("mod_add"),
        "forecasting" => Some("forecast_seasonal_trend"),
        _ => None,
    }
}

/// Construct the representative benchmark for an axis at the given smoke budget.
/// Returns the boxed [`Benchmark`] plus whether it is informational.
fn build_bench(name: &str, cfg: &CapScaleConfig) -> Option<(Box<dyn Benchmark>, bool)> {
    use crate::*;
    let steps = cfg.steps;
    let n = cfg.n_sequences;
    let e = cfg.eval_sequences;
    let b: Box<dyn Benchmark> = match name {
        "mqar" => Box::new(mqar::Mqar { steps, n_sequences: n, eval_sequences: e, ..Default::default() }),
        "mad_selective_copy" => Box::new(mad_selective_copy::MadSelectiveCopy {
            steps,
            n_sequences: n,
            eval_sequences: e,
            ..Default::default()
        }),
        "mad_memorize" => Box::new(mad_memorize::MadMemorize { steps, n_sequences: n, eval_sequences: e, ..Default::default() }),
        "parity" => Box::new(parity::Parity { steps, n_sequences: n, eval_sequences: e, ..Default::default() }),
        "mad_compress" => Box::new(mad_compress::MadCompress { steps, n_sequences: n, eval_sequences: e, ..Default::default() }),
        "mod_add" => Box::new(mod_add::ModAdd { steps, ..Default::default() }),
        // Forecasting probes ignore the decoder; build by scenario name.
        n if n.starts_with("forecast_") => {
            let b = crate::forecast_bench::build(n, e.max(8))?;
            let informational = b.informational();
            return Some((b, informational));
        }
        _ => return None,
    };
    let informational = b.informational();
    Some((b, informational))
}

/// A fitted saturating trend `score(N) ≈ ceil − A·N^(−β)`.
///
/// Scores rise toward a ceiling as capacity grows, so we fit the **gap to a
/// ceiling** as a power law in `N`: `log(ceil − score) = log A − β·log N`. A
/// larger **β** ⇒ the score closes its gap to the ceiling faster with size. The
/// ceiling is grid-searched just above the largest observed score.
#[derive(Clone, Debug)]
pub struct SaturatingFit {
    /// The fitted ceiling the score saturates toward.
    pub ceil: f64,
    /// Coefficient `A` (initial gap scale).
    pub a: f64,
    /// Exponent `β` (how fast the gap to the ceiling closes with size).
    pub beta: f64,
    /// Log-space coefficient of determination (1.0 = perfect power-law gap).
    pub r2: f64,
}

impl SaturatingFit {
    /// Predicted score at parameter count `n`.
    pub fn predict(&self, n: f64) -> f64 {
        (self.ceil - self.a * n.powf(-self.beta)).clamp(0.0, 1.0)
    }
}

/// One axis's scaling result: the per-size points, the fitted trend, predictions,
/// and a coarse verdict.
#[derive(Clone, Debug)]
pub struct AxisScaling {
    pub axis: String,
    /// Benchmark used as this axis's probe.
    pub bench: String,
    /// `true` if the probe is informational (diagnostic; curve is noisy).
    pub informational: bool,
    /// Per-size parameter counts (the independent axis `N`), increasing.
    pub params: Vec<u64>,
    /// Per-size scores at those `N` (same order).
    pub scores: Vec<f32>,
    /// Size labels (e.g. `"L1xD32xH2"`), same order.
    pub labels: Vec<String>,
    /// Fitted saturating trend.
    pub fit: SaturatingFit,
    /// Predicted score at 2× the largest `N`.
    pub pred_2x: f64,
    /// Predicted score at 4× the largest `N`.
    pub pred_4x: f64,
    /// Local slope: score gain per doubling of `N` over the measured grid
    /// (Δscore / Δlog2 N). This is the "how much does this axis respond to the
    /// knob" lever the advisor ranks by.
    pub slope_per_doubling: f64,
    /// Coarse verdict from slope + headroom.
    pub verdict: Verdict,
}

/// How an axis responds to the swept knob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Score is still climbing meaningfully with size — more capacity helps.
    Improving,
    /// Score is near its ceiling — little further gain from size.
    Saturating,
    /// Score barely moves with size — this axis is architecture-bound, not
    /// capacity-bound (changing the *mechanism* will help, not more params).
    Flat,
}

impl Verdict {
    pub fn label(&self) -> &'static str {
        match self {
            Verdict::Improving => "improving",
            Verdict::Saturating => "saturating",
            Verdict::Flat => "flat",
        }
    }
}

/// The full per-capability scaling report for one architecture.
#[derive(Clone, Debug)]
pub struct CapScaleReport {
    pub arch: String,
    pub knob: Knob,
    pub seed: u64,
    pub commit: String,
    /// The largest `N` in the grid (the base for 2×/4× extrapolation).
    pub max_params: u64,
    pub axes: Vec<AxisScaling>,
}

impl CapScaleReport {
    pub fn to_json(&self) -> Value {
        let axes: Vec<Value> = self
            .axes
            .iter()
            .map(|a| {
                json!({
                    "axis": a.axis,
                    "bench": a.bench,
                    "informational": a.informational,
                    "params": a.params,
                    "scores": a.scores,
                    "labels": a.labels,
                    "fit": {
                        "ceil": a.fit.ceil,
                        "a": a.fit.a,
                        "beta": a.fit.beta,
                        "r2": a.fit.r2,
                    },
                    "slope_per_doubling": a.slope_per_doubling,
                    "predicted_2x": a.pred_2x,
                    "predicted_4x": a.pred_4x,
                    "verdict": a.verdict.label(),
                })
            })
            .collect();
        json!({
            "arch": self.arch,
            "knob": self.knob.label(),
            "seed": self.seed,
            "commit": self.commit,
            "max_params": self.max_params,
            "axes": axes,
        })
    }
}

/// Default artifact path for a capscale run: `results/scale-<arch>-<seed>.json`.
pub fn default_out_path(arch: &str, seed: u64) -> std::path::PathBuf {
    std::path::PathBuf::from("results").join(format!("scale-{arch}-{seed}.json"))
}

/// Write the report's JSON artifact to `path` (creating parent dirs).
pub fn write_artifact(report: &CapScaleReport, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let pretty = serde_json::to_string_pretty(&report.to_json()).map_err(std::io::Error::other)?;
    std::fs::write(path, pretty)
}

/// Run the per-capability scaling sweep for `arch_name` across the size grid.
///
/// For each axis, trains+scores its representative benchmark at every size, fits
/// the saturating trend, and extrapolates. Only the GPT family (which honors a
/// [`ScaledGpt`] size override) is supported today; the `arch` arg selects the
/// *seed* identity recorded in the artifact, while the actual swept decoders are
/// `ScaledGpt` at each grid size.
pub fn run(arch_name: &str, cfg: &CapScaleConfig) -> std::io::Result<CapScaleReport> {
    // Validate the architecture exists (so an unknown name errors early like eval).
    arch::get_arch(arch_name).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "unknown architecture '{arch_name}'; known: {}",
                arch::arch_names().join(", ")
            ),
        )
    })?;

    let grid = grid_for(cfg.knob);
    if grid.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("knob '{}' has no usable grid yet (need ≥2 points)", cfg.knob.label()),
        ));
    }

    let mut axis_results = Vec::new();
    let mut max_params = 0u64;

    for ax in axes() {
        let Some(bench_name) = representative_bench(ax) else { continue };
        let Some((bench, informational)) = build_bench(bench_name, cfg) else { continue };

        let mut params = Vec::new();
        let mut scores = Vec::new();
        let mut labels = Vec::new();

        for (label, size) in &grid {
            let lm: Box<dyn crate::DecoderLm> = match cfg.knob {
                Knob::Size => Box::new(ScaledGpt(*size)),
                // TODO(experts): build the MoE `DecoderLm` at this expert count.
                Knob::Experts => unreachable!("experts knob has no grid yet"),
            };
            let score = score_at(bench.as_ref(), lm.as_ref(), cfg.seed)?;
            // Param count at this size, using the same basis the eval artifact uses
            // so curves are comparable to the eval `param_count`.
            let n = size_param_count(size);
            params.push(n);
            scores.push(score);
            labels.push(label.clone());
            max_params = max_params.max(n);
        }

        let fit = fit_saturating(&params, &scores);
        let nmax = *params.last().unwrap() as f64;
        let pred_2x = fit.predict(2.0 * nmax);
        let pred_4x = fit.predict(4.0 * nmax);
        let slope = slope_per_doubling(&params, &scores);
        let verdict = verdict_of(&scores, slope, fit.ceil);

        axis_results.push(AxisScaling {
            axis: ax.to_string(),
            bench: bench_name.to_string(),
            informational,
            params,
            scores,
            labels,
            fit,
            pred_2x,
            pred_4x,
            slope_per_doubling: slope,
            verdict,
        });
    }

    Ok(CapScaleReport {
        arch: arch_name.to_string(),
        knob: cfg.knob,
        seed: cfg.seed,
        commit: git_commit(),
        max_params,
        axes: axis_results,
    })
}

/// Param count for a fixed [`Size`] at the eval artifact's representative basis
/// (`vocab=64, block_size=32`), so capscale `N` lines up with eval `param_count`.
fn size_param_count(size: &Size) -> u64 {
    use gpt2::GptConfig;
    let d_model = size.d_model.unwrap_or(64);
    let cfg = GptConfig {
        vocab: 64,
        block_size: 32,
        n_layers: size.n_layers.unwrap_or(2),
        d_model,
        n_heads: size.n_heads.unwrap_or(4),
        d_ff: d_model * 4,
    };
    cfg.param_list().iter().map(|(_, n)| *n as u64).sum()
}

/// Train + score one benchmark with one decoder in a scratch dir (mirrors
/// `eval::run_one` but takes a prebuilt `lm`).
fn score_at(b: &dyn Benchmark, lm: &dyn crate::DecoderLm, seed: u64) -> std::io::Result<f32> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static UNIQ: AtomicU64 = AtomicU64::new(0);
    let uniq = UNIQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "brain_capscale_{}_{}_{}",
        b.name(),
        std::process::id(),
        uniq
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    b.prepare(&dir, seed)?;
    let metrics = b.evaluate_with(lm, &dir, seed)?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(metrics.score)
}

/// Fit `score(N) ≈ ceil − A·N^(−β)` by grid-searching the ceiling just above the
/// largest observed score and, for each candidate, OLS-fitting
/// `log(ceil − score) = log A − β·log N`. Keeps the ceiling with the best R².
///
/// Mirrors [`scaling::fit_power_law`](crate::scaling::fit_power_law) but fits the
/// *gap to a ceiling* (scores rise, losses fall). With non-monotone / flat points
/// the power fit degrades to β≈0 (the advisor reads that as "flat").
pub fn fit_saturating(params: &[u64], scores: &[f32]) -> SaturatingFit {
    assert!(params.len() >= 2 && params.len() == scores.len(), "need ≥2 aligned points");
    let xs: Vec<f64> = params.iter().map(|&n| (n as f64).ln()).collect();
    let ys_score: Vec<f64> = scores.iter().map(|&s| s as f64).collect();
    let max_score = ys_score.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Candidate ceilings strictly above the largest score (so log(ceil−score) is
    // finite), from just-above up to 1.0 (scores are in [0,1]). Also try a small
    // headroom band above max so a still-climbing curve fits.
    let mut best: Option<SaturatingFit> = None;
    let steps = 64;
    let lo = (max_score + 1e-3).min(1.0);
    let hi = 1.0_f64.max(max_score + 1e-3);
    for i in 0..steps {
        let ceil = if (hi - lo).abs() < 1e-9 {
            lo + 1e-3
        } else {
            lo + (hi - lo) * (i as f64 / (steps - 1) as f64)
        };
        let ys: Vec<f64> = ys_score.iter().map(|&s| (ceil - s).max(1e-9).ln()).collect();
        let (slope, intercept, r2) = ols(&xs, &ys);
        let beta = -slope; // log(ceil−score) = logA − β·logN ⇒ slope = −β
        if beta < 0.0 {
            continue; // score must rise with size for a valid saturating law
        }
        let fit = SaturatingFit { ceil, a: intercept.exp(), beta, r2 };
        if best.as_ref().map(|b| fit.r2 > b.r2).unwrap_or(true) {
            best = Some(fit);
        }
    }

    best.unwrap_or_else(|| {
        // Degenerate (flat / non-monotone): a flat fit at the mean.
        let mean = ys_score.iter().sum::<f64>() / ys_score.len() as f64;
        SaturatingFit { ceil: (mean + 1e-3).min(1.0), a: 0.0, beta: 0.0, r2: 0.0 }
    })
}

/// Local slope = score gain per doubling of `N`: `(last − first) / log2(Nlast /
/// Nfirst)`. A direct, fit-free "responsiveness to the knob" measure the advisor
/// ranks by (a flat axis ⇒ ≈0).
pub fn slope_per_doubling(params: &[u64], scores: &[f32]) -> f64 {
    if params.len() < 2 {
        return 0.0;
    }
    let n0 = *params.first().unwrap() as f64;
    let n1 = *params.last().unwrap() as f64;
    let doublings = (n1 / n0).log2();
    if doublings.abs() < 1e-9 {
        return 0.0;
    }
    let s0 = *scores.first().unwrap() as f64;
    let s1 = *scores.last().unwrap() as f64;
    (s1 - s0) / doublings
}

/// Coarse verdict from the measured slope + how close the top score is to the
/// fitted ceiling:
/// - near the ceiling (gap < 0.05) ⇒ **saturating**,
/// - else slope per doubling ≥ 0.03 ⇒ **improving**,
/// - else ⇒ **flat** (architecture-bound, not capacity-bound).
fn verdict_of(scores: &[f32], slope_per_doubling: f64, ceil: f64) -> Verdict {
    let top = *scores.last().unwrap() as f64;
    if (ceil - top) < 0.05 && top > 0.6 {
        Verdict::Saturating
    } else if slope_per_doubling >= 0.03 {
        Verdict::Improving
    } else {
        Verdict::Flat
    }
}

/// Short git commit hash (`git rev-parse --short HEAD`), or `"unknown"`.
fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Print the per-axis curves + fitted trend + predictions for one report.
pub fn print_report(report: &CapScaleReport) {
    println!(
        "\nper-capability scaling sweep — arch: {}  knob: {}  seed: {}  commit {}",
        report.arch,
        report.knob.label(),
        report.seed,
        report.commit,
    );
    println!(
        "grid (params N): {}\n",
        report
            .axes
            .first()
            .map(|a| a
                .params
                .iter()
                .zip(&a.labels)
                .map(|(n, l)| format!("{l}={n}"))
                .collect::<Vec<_>>()
                .join("  "))
            .unwrap_or_default()
    );

    let header = format!(
        "{:<14} {:<18} {:>22} {:>7} {:>6} {:>8} {:>8} {:>11}",
        "axis", "bench", "scores@sizes", "slope", "beta", "pred@2x", "pred@4x", "verdict"
    );
    println!("{header}");
    println!("{}", "-".repeat(header.len()));
    for a in &report.axes {
        let scores = a
            .scores
            .iter()
            .map(|s| format!("{s:.3}"))
            .collect::<Vec<_>>()
            .join("/");
        let info = if a.informational { "*" } else { "" };
        println!(
            "{:<14} {:<18} {:>22} {:>7.3} {:>6.2} {:>8.3} {:>8.3} {:>11}",
            a.axis,
            format!("{}{}", a.bench, info),
            scores,
            a.slope_per_doubling,
            a.fit.beta,
            a.pred_2x,
            a.pred_4x,
            a.verdict.label(),
        );
    }
    println!(
        "\nslope = Δscore per doubling of N; beta = saturating-fit exponent (gap→ceiling);\n\
         pred@2x/@4x = extrapolated score at 2×/4× the largest N ({} params).  (* = informational)\n",
        report.max_params,
    );
}

/// Load a capscale artifact written by [`write_artifact`] (the subset the advisor
/// needs): per-axis slope, verdict, and the 2×/4× predictions.
pub struct LoadedCapScale {
    pub arch: String,
    pub knob: String,
    pub max_params: u64,
    /// axis -> (slope_per_doubling, beta, pred_2x, pred_4x, verdict).
    pub by_axis: BTreeMap<String, LoadedAxis>,
}

/// One axis's loaded capscale fields.
#[derive(Clone, Debug)]
pub struct LoadedAxis {
    pub bench: String,
    pub slope_per_doubling: f64,
    pub beta: f64,
    pub pred_2x: f64,
    pub pred_4x: f64,
    pub verdict: String,
    pub top_score: f64,
}

/// Load a capscale artifact. Errors only on missing file / bad JSON.
pub fn load_artifact(path: &Path) -> std::io::Result<LoadedCapScale> {
    let text = std::fs::read_to_string(path)?;
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let arch = v["arch"].as_str().unwrap_or("?").to_string();
    let knob = v["knob"].as_str().unwrap_or("size").to_string();
    let max_params = v["max_params"].as_u64().unwrap_or(0);
    let mut by_axis = BTreeMap::new();
    if let Some(arr) = v["axes"].as_array() {
        for a in arr {
            let Some(axis) = a["axis"].as_str() else { continue };
            let top_score = a["scores"]
                .as_array()
                .and_then(|s| s.last())
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0);
            by_axis.insert(
                axis.to_string(),
                LoadedAxis {
                    bench: a["bench"].as_str().unwrap_or("?").to_string(),
                    slope_per_doubling: a["slope_per_doubling"].as_f64().unwrap_or(0.0),
                    beta: a["fit"]["beta"].as_f64().unwrap_or(0.0),
                    pred_2x: a["predicted_2x"].as_f64().unwrap_or(0.0),
                    pred_4x: a["predicted_4x"].as_f64().unwrap_or(0.0),
                    verdict: a["verdict"].as_str().unwrap_or("flat").to_string(),
                    top_score,
                },
            );
        }
    }
    Ok(LoadedCapScale { arch, knob, max_params, by_axis })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_a_known_saturating_curve() {
        // score(N) = 1.0 − 5·N^(−0.2): rises toward 1.0; recover β.
        let params: Vec<u64> = vec![1_000, 10_000, 100_000, 1_000_000];
        let scores: Vec<f32> =
            params.iter().map(|&n| (1.0 - 5.0 * (n as f64).powf(-0.2)) as f32).collect();
        let fit = fit_saturating(&params, &scores);
        assert!((fit.beta - 0.2).abs() < 0.06, "beta {} != 0.2", fit.beta);
        assert!(fit.r2 > 0.95, "r2 {} too low", fit.r2);
        // Extrapolation rises above the largest observed score.
        let nmax = *params.last().unwrap() as f64;
        assert!(fit.predict(4.0 * nmax) >= *scores.last().unwrap() as f64);
    }

    #[test]
    fn flat_curve_has_near_zero_slope() {
        let params = vec![1_000u64, 10_000, 100_000];
        let scores = vec![0.30f32, 0.31, 0.29];
        let slope = slope_per_doubling(&params, &scores);
        assert!(slope.abs() < 0.02, "flat slope should be ~0, got {slope}");
        let fit = fit_saturating(&params, &scores);
        assert_eq!(verdict_of(&scores, slope, fit.ceil), Verdict::Flat);
    }

    #[test]
    fn rising_curve_is_improving() {
        let params = vec![1_000u64, 10_000, 100_000];
        let scores = vec![0.20f32, 0.40, 0.55];
        let slope = slope_per_doubling(&params, &scores);
        let fit = fit_saturating(&params, &scores);
        assert_eq!(verdict_of(&scores, slope, fit.ceil), Verdict::Improving);
    }

    #[test]
    fn saturated_curve_near_ceiling() {
        let scores = vec![0.95f32, 0.98, 0.99];
        // ceiling just above top, gap < 0.05.
        assert_eq!(verdict_of(&scores, 0.02, 1.0), Verdict::Saturating);
    }

    #[test]
    fn every_axis_has_a_representative_bench() {
        for ax in axes() {
            assert!(representative_bench(ax).is_some(), "axis '{ax}' has no probe");
        }
    }

    #[test]
    fn size_grid_params_increase() {
        let mut prev = 0u64;
        for &(l, d, h) in SIZE_GRID {
            let n = size_param_count(&Size::fixed(l, d, h));
            assert!(n > prev, "grid params not increasing: {n} <= {prev}");
            prev = n;
        }
    }

    #[test]
    fn knob_label_roundtrip() {
        assert_eq!(Knob::Size.label(), "size");
        assert_eq!(Knob::Experts.label(), "experts");
    }
}
