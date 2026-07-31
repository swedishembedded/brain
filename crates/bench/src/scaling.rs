// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Multi-scale **scaling-law sweep** — train the same task at several model
//! sizes, then fit a power law to (parameter count → final loss).
//!
//! This is a *separate entry point* from the [`Benchmark`](crate::Benchmark)
//! registry: a single benchmark answers "does architecture X learn task T?"; the
//! scaling sweep instead asks "how does loss on task T improve as we grow the
//! model?". Concretely it
//!
//! 1. synthesizes one fixed task (reusing an existing benchmark's dataset — the
//!    [`Mqar`](crate::mqar::Mqar) recall task, which improves clearly with model
//!    capacity),
//! 2. trains a [`gpt::Gpt`] (via the architecture-agnostic
//!    [`model::train::fit`]) at a grid of increasing sizes (n_layers / d_model),
//! 3. records per size: **parameter count**, a **training-FLOPs proxy**
//!    (`≈ 6 · params · tokens`, the Kaplan/Chinchilla compute estimate), and the
//!    **final training loss**, and
//! 4. fits a Chinchilla-style power law `L(N) ≈ E + A · N^(−α)` over the points,
//!    reporting the fitted exponent **α**, the fit quality **R²**, and the
//!    per-size table.
//!
//! ## Why this is the foundation for predictive scaling
//! The later per-capability predictive-scaling / eval-harness work needs exactly
//! this machinery — a reproducible "train a grid of sizes, fit `L(N)`" loop whose
//! output is a single extrapolatable exponent. Here the grid is tiny (≈3 sizes, a
//! few hundred steps each) so it runs in minutes on the CPU backend; the same
//! [`run`] / [`fit_power_law`] code generalizes to larger grids, more tasks, and
//! per-capability loss slices (just widen [`Sweep::sizes`] and swap the dataset).
//!
//! ## Running
//! - `brain bench scaling` (CLI subcommand) or `make bench/scaling`.
//! - [`run`] returns the structured [`SweepResult`]; [`SweepResult::print`]
//!   renders the size table + fitted α / R².

use std::path::Path;

use crate::model::{DecoderLm, GptDecoder, TrainConfig};
use crate::mqar::Mqar;
use crate::Benchmark;
use gpt::GptConfig;

/// One (n_layers, d_model) point in the sweep, with its measured outcome.
#[derive(Clone, Debug)]
pub struct SizePoint {
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    /// Trainable parameter count (`N` in the power law).
    pub params: u64,
    /// Training-FLOPs proxy `≈ 6 · params · tokens` (Kaplan/Chinchilla).
    pub flops: f64,
    /// Final training cross-entropy (nats) — the loss we fit against.
    pub final_loss: f32,
}

/// A fitted power law `L(N) ≈ E + A · N^(−α)`.
#[derive(Clone, Debug)]
pub struct PowerLaw {
    /// Irreducible-loss floor `E`.
    pub e: f64,
    /// Coefficient `A`.
    pub a: f64,
    /// Exponent `α` (larger ⇒ loss falls faster with size).
    pub alpha: f64,
    /// Coefficient of determination of the log-space fit (1.0 = perfect).
    pub r2: f64,
}

impl PowerLaw {
    /// Predicted loss at parameter count `n`.
    pub fn predict(&self, n: f64) -> f64 {
        self.e + self.a * n.powf(-self.alpha)
    }
}

/// The full sweep outcome: the per-size table plus the fitted law.
#[derive(Clone, Debug)]
pub struct SweepResult {
    pub points: Vec<SizePoint>,
    pub law: PowerLaw,
    /// Tokens seen per training run (`steps · batch · block`) — the `tokens`
    /// factor in the FLOPs proxy. Shared across sizes (same task + budget).
    pub tokens_per_run: u64,
}

/// Configuration for a scaling sweep: the size grid + the per-run training
/// budget. Defaults are calibrated to run in a few minutes on the CPU backend.
#[derive(Clone, Debug)]
pub struct Sweep {
    /// `(n_layers, d_model, n_heads)` grid, in increasing-capacity order.
    pub sizes: Vec<(u32, u32, u32)>,
    /// Training steps per size.
    pub steps: u32,
    pub batch_size: u32,
    pub lr: f32,
    pub seed: u64,
}

impl Default for Sweep {
    /// Three increasing sizes on the MQAR recall task, ~400 steps each. The grid
    /// grows both depth and width so the parameter count rises clearly across
    /// points; head count divides each d_model evenly (32/2, 64/4, 96/6 ⇒
    /// head_dim 16). Sized for ≤ ~5 min total on the CPU (Cranelift) backend.
    fn default() -> Self {
        Sweep {
            sizes: vec![(1, 32, 2), (2, 64, 4), (3, 96, 6)],
            steps: 400,
            batch_size: 32,
            lr: 3e-3,
            seed: 1337,
        }
    }
}

impl Sweep {
    /// Parameter count for one size on the given task vocab/block (uses GPT's own
    /// `param_list`, so it matches what is actually allocated and trained).
    fn param_count(&self, vocab: u32, block: u32, n_layers: u32, d_model: u32) -> u64 {
        let cfg = GptConfig {
            vocab,
            block_size: block,
            n_layers,
            d_model,
            n_heads: 1, // irrelevant to param count
            d_ff: d_model * 4,
        };
        cfg.param_list().iter().map(|(_, n)| *n as u64).sum()
    }
}

/// Run the scaling sweep: synthesize the (fixed) MQAR task once, train a GPT at
/// each size, and fit the power law. `dir` is a scratch directory for the
/// dataset + checkpoints (created if absent).
pub fn run(sweep: &Sweep, dir: &Path) -> std::io::Result<SweepResult> {
    std::fs::create_dir_all(dir)?;

    // Fixed task that improves clearly with capacity: multi-query associative
    // recall. Reuse the benchmark's own dataset synthesis so the sweep trains on
    // exactly the reference task (no bespoke generator to drift from it).
    //
    // We deliberately use a *harder* MQAR config than the benchmark default: more
    // bindings/queries over a larger content vocab so the answer-region loss is
    // genuinely capacity-bound. The default (2 pairs / 2 queries) is so easy that
    // even the smallest model nearly solves it, leaving the *loss* flat/noisy
    // across sizes (recall still rises, but the headline here is the fitted loss
    // curve) — the harder config makes bigger models reach a clearly lower loss.
    let task = scaling_task();
    task.prepare(dir, sweep.seed)?;
    let block = task_block_size(&task);
    let vocab = task_vocab(&task);

    let tokens_per_run = sweep.steps as u64 * sweep.batch_size as u64 * block as u64;

    let mut points = Vec::with_capacity(sweep.sizes.len());
    for &(n_layers, d_model, n_heads) in &sweep.sizes {
        let cfg = TrainConfig {
            steps: sweep.steps,
            batch_size: sweep.batch_size,
            lr: sweep.lr,
            n_layers,
            d_model,
            n_heads,
            mask_before: Some('='), // SEP — same answer-masking recipe as MQAR
            mask_per_line: true,
            align_to_lines: true,
            seed: sweep.seed,
        };
        let out = dir.join(format!("scaling_l{n_layers}_d{d_model}.safetensors"));
        // Fresh checkpoint per size (fit resumes if present; remove to retrain).
        let _ = std::fs::remove_file(&out);
        let (_init, final_loss) = GptDecoder.train_decoder(dir, block, &cfg, &out)?;
        let _ = std::fs::remove_file(&out);

        let params = sweep.param_count(vocab, block, n_layers, d_model);
        let flops = 6.0 * params as f64 * tokens_per_run as f64;
        points.push(SizePoint { n_layers, d_model, n_heads, params, flops, final_loss });
    }

    let law = fit_power_law(&points);
    Ok(SweepResult { points, law, tokens_per_run })
}

/// The fixed task the sweep trains on: a harder MQAR than the benchmark default
/// (more bindings/queries over a bigger content vocab) so the answer-region loss
/// is capacity-bound and falls with model size. Kept here (not the registry
/// default) so widening it never perturbs the `mqar` benchmark's calibration.
fn scaling_task() -> Mqar {
    Mqar {
        vocab_content: 32, // 16 keys + 16 values → chance recall 1/16
        n_pairs: 6,        // more bindings to disambiguate (the difficulty knob)
        n_queries: 4,
        n_sequences: 8000,
        ..Mqar::default()
    }
}

/// MQAR's sequence length is its training/scoring block size; recompute it from
/// the prepared dataset's meta so we stay in lockstep with the benchmark.
fn task_block_size(task: &Mqar) -> u32 {
    // seq_len = 2*n_pairs + 1 (SEP) + 2*n_queries + 1 (NL); mirrors Mqar::seq_len.
    (2 * task.n_pairs + 1 + 2 * task.n_queries + 1) as u32
}

fn task_vocab(task: &Mqar) -> u32 {
    // CONTENT0 (=2) + vocab_content; mirrors Mqar::vocab.
    2 + task.vocab_content as u32
}

/// Fit `L(N) ≈ E + A · N^(−α)` to the measured points.
///
/// With only a handful of points the three-parameter fit is unstable if all
/// three are optimized jointly, so we fix the irreducible floor `E` by a coarse
/// grid search and, for each candidate `E`, fit the remaining two parameters by
/// **ordinary least squares in log–log space**: `log(L − E) = log A − α·log N`.
/// We keep the `E` whose linear fit has the best R². The reported `r2` is that
/// best log-space R² (1.0 = the points lie exactly on a power law).
pub fn fit_power_law(points: &[SizePoint]) -> PowerLaw {
    assert!(points.len() >= 2, "need ≥2 sizes to fit a scaling law");
    let xs: Vec<f64> = points.iter().map(|p| (p.params as f64).ln()).collect();
    let ys_loss: Vec<f64> = points.iter().map(|p| p.final_loss as f64).collect();
    let min_loss = ys_loss.iter().cloned().fold(f64::INFINITY, f64::min);

    // Candidate floors E in [0, min_loss): below every observed loss so log(L−E)
    // is finite. Grid from 0 up to just under the smallest loss.
    let mut best: Option<PowerLaw> = None;
    let steps = 64;
    for i in 0..steps {
        let e = min_loss * (i as f64 / steps as f64) * 0.999;
        let ys: Vec<f64> = ys_loss.iter().map(|&l| (l - e).max(1e-9).ln()).collect();
        let (slope, intercept, r2) = ols(&xs, &ys);
        let alpha = -slope; // log(L−E) = logA − α·logN ⇒ slope = −α
        if alpha <= 0.0 {
            continue; // loss must decrease with size for a valid law
        }
        let law = PowerLaw { e, a: intercept.exp(), alpha, r2 };
        if best.as_ref().map(|b| law.r2 > b.r2).unwrap_or(true) {
            best = Some(law);
        }
    }

    // Fallback (e.g. non-monotone points): plain log–log fit with E = 0.
    best.unwrap_or_else(|| {
        let ys: Vec<f64> = ys_loss.iter().map(|&l| l.max(1e-9).ln()).collect();
        let (slope, intercept, r2) = ols(&xs, &ys);
        PowerLaw { e: 0.0, a: intercept.exp(), alpha: -slope, r2 }
    })
}

/// Ordinary least squares `y ≈ slope·x + intercept`; returns `(slope, intercept,
/// r2)`. `r2` is the coefficient of determination (clamped to ≥0). Shared with
/// [`capscale`](crate::capscale)'s saturating-trend fit.
pub(crate) fn ols(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        sxx += (x - mean_x) * (x - mean_x);
        sxy += (x - mean_x) * (y - mean_y);
        syy += (y - mean_y) * (y - mean_y);
    }
    let slope = if sxx.abs() < 1e-30 { 0.0 } else { sxy / sxx };
    let intercept = mean_y - slope * mean_x;
    // R² = 1 − SS_res/SS_tot.
    let mut ss_res = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        let pred = slope * x + intercept;
        ss_res += (y - pred) * (y - pred);
    }
    let r2 = if syy.abs() < 1e-30 { 1.0 } else { (1.0 - ss_res / syy).max(0.0) };
    (slope, intercept, r2)
}

impl SweepResult {
    /// Render the per-size table followed by the fitted law (α + R²).
    pub fn print(&self) {
        println!(
            "\nscaling-law sweep — task: mqar (multi-query associative recall)\n\
             tokens/run ≈ {} (steps · batch · block); FLOPs proxy ≈ 6 · params · tokens\n",
            self.tokens_per_run
        );
        let header = format!(
            "{:<14} {:>10} {:>14} {:>12}",
            "size", "params", "flops", "final_loss"
        );
        println!("{header}");
        println!("{}", "-".repeat(header.len()));
        for p in &self.points {
            println!(
                "{:<14} {:>10} {:>14.3e} {:>12.4}",
                format!("L{}xD{}", p.n_layers, p.d_model),
                p.params,
                p.flops,
                p.final_loss,
            );
        }
        println!(
            "\nfitted power law  L(N) ≈ E + A·N^(−α)\n  \
             α (exponent) = {:.4}\n  \
             E (floor)    = {:.4}\n  \
             A (coeff)    = {:.4e}\n  \
             R² (fit)     = {:.4}",
            self.law.alpha, self.law.e, self.law.a, self.law.r2
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(params: u64, loss: f32) -> SizePoint {
        SizePoint { n_layers: 1, d_model: 1, n_heads: 1, params, flops: 0.0, final_loss: loss }
    }

    #[test]
    fn fits_a_known_power_law() {
        // Synthesize L(N) = 0.5 + 10 · N^(−0.3) exactly and recover α.
        let truth_alpha = 0.3f64;
        let truth_e = 0.5f64;
        let truth_a = 10.0f64;
        let points: Vec<SizePoint> = [1_000u64, 10_000, 100_000, 1_000_000]
            .iter()
            .map(|&n| {
                let l = truth_e + truth_a * (n as f64).powf(-truth_alpha);
                pt(n, l as f32)
            })
            .collect();
        let law = fit_power_law(&points);
        assert!((law.alpha - truth_alpha).abs() < 0.05, "alpha {} != {truth_alpha}", law.alpha);
        assert!(law.r2 > 0.99, "r2 {} too low", law.r2);
    }

    #[test]
    fn default_sweep_grid_is_increasing() {
        let s = Sweep::default();
        let mut prev = 0u64;
        for &(l, d, _) in &s.sizes {
            let n = s.param_count(18, 8, l, d);
            assert!(n > prev, "params not strictly increasing: {n} <= {prev}");
            prev = n;
        }
    }

    #[test]
    fn power_law_predict_matches_definition() {
        let law = PowerLaw { e: 0.5, a: 10.0, alpha: 0.3, r2: 1.0 };
        let n = 50_000.0;
        assert!((law.predict(n) - (0.5 + 10.0 * n.powf(-0.3))).abs() < 1e-9);
    }
}
