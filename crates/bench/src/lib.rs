// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! brain's architecture-evaluation **benchmark suite** — a reusable,
//! model-agnostic layer for asking "does this architecture actually learn task
//! X?" the same way across benchmarks.
//!
//! A [`Benchmark`] owns its **dataset** (how to synthesize/write it) and its
//! **scoring** (what a good model looks like on it). The harness owns *running*:
//! every benchmark trains a model on its generated data and returns [`Metrics`]
//! whose headline `score` is checked against the benchmark's `threshold`.
//!
//! Benchmarks are model-agnostic at the **data / metric** level: a benchmark
//! never names "GPT" in its dataset or its scoring. Training + scoring go through
//! the [`model::DecoderLm`] trait (an architecture that can be trained as a
//! causal next-token decoder and queried for per-position logits). The default
//! architecture is [`model::GptDecoder`]; dropping in a MoE/PID decoder is a new
//! `DecoderLm` impl, with no benchmark changes — that is what makes the suite an
//! *architecture* evaluation rather than a GPT-only one.
//!
//! ## Adding a benchmark
//! 1. Add a module under `crates/bench/src/` implementing [`Benchmark`].
//! 2. Register it in [`registry`].
//! 3. (optional) Add a `make bench/<name>` shortcut — the generic
//!    `bench/%` rule already runs any registered name.
//!
//! ## Running
//! - [`run_all`] runs every registered benchmark and prints one comparison table.
//! - [`run_one`] runs a single benchmark by name.

use std::path::Path;

pub mod advisor;
pub mod arch;
pub mod axes;
pub mod capscale;
pub mod eval;
pub mod mad_compress;
pub mod mad_fuzzy_recall;
pub mod mad_memorize;
pub mod mad_noisy_recall;
pub mod mad_recall;
pub mod mad_selective_copy;
pub mod metrics;
pub mod mod_add;
pub mod model;
pub mod moe_decoder;
pub mod mqar;
pub mod dyck;
pub mod parity;
pub mod scaling;
pub mod toolcall;

pub use metrics::Metrics;
pub use model::{DecoderLm, GptDecoder, Scorer, TrainConfig};
pub use moe_decoder::MoeDecoder;

pub use arch::{arch_registry, get_arch, Arch};
pub use axes::{axes, axis_of};

/// A benchmark: owns its dataset and its scoring, model-agnostic at this level.
///
/// Implementors are cheap value types holding their config. The harness calls
/// [`prepare`](Benchmark::prepare) once to write the dataset, then
/// [`evaluate`](Benchmark::evaluate) to train + score a model on it.
pub trait Benchmark {
    /// Stable identifier used on the CLI, in `make bench/<name>`, and as the
    /// table row label. Lowercase, no spaces.
    fn name(&self) -> &str;

    /// One-line human description for the table header / help.
    fn description(&self) -> &str;

    /// Generate and write this benchmark's dataset under `dir` (created if
    /// absent), deterministically from `seed`. After this returns, `dir` holds
    /// whatever [`evaluate`](Benchmark::evaluate) needs (e.g. brain's
    /// `train.bin`/`val.bin`/`meta.json` token-dataset layout).
    fn prepare(&self, dir: &Path, seed: u64) -> std::io::Result<()>;

    /// Train a model on the dataset in `dir` and return its [`Metrics`]. The
    /// headline `Metrics::score` is what [`threshold`](Benchmark::threshold)
    /// gates.
    ///
    /// This defaults to scoring the **GPT baseline** ([`model::GptDecoder`]). It
    /// is the entry point the single-arch runner (`run_all`/`run_one`) uses, so
    /// existing behavior is unchanged. To score a *different* architecture, the
    /// eval harness calls [`evaluate_with`](Benchmark::evaluate_with) directly.
    fn evaluate(&self, dir: &Path, seed: u64) -> std::io::Result<Metrics> {
        self.evaluate_with(&GptDecoder, dir, seed)
    }

    /// Train + score this benchmark with a **specific architecture** (any
    /// [`DecoderLm`]) and return its [`Metrics`]. This is the architecture-agnostic
    /// core: [`evaluate`](Benchmark::evaluate) is just this with the GPT baseline,
    /// and the architecture-eval harness ([`arch`]) drives the whole battery
    /// through here with a registered [`arch::Arch`]'s decoder.
    ///
    /// Benchmarks whose objective is *not* a causal next-token decoder (e.g. the
    /// [`mad_compress`] autoencoder) ignore `lm` and train their own model; that
    /// is documented per benchmark.
    fn evaluate_with(
        &self,
        lm: &dyn DecoderLm,
        dir: &Path,
        seed: u64,
    ) -> std::io::Result<Metrics>;

    /// Pass/fail bar for `Metrics::score` (higher is better unless a benchmark
    /// documents otherwise). Calibrated against measured CPU-backend runs.
    fn threshold(&self) -> f32;

    /// Names of the extra [`Metrics`] fields worth printing in the table, in
    /// column order. The headline `score` is always shown first.
    fn report_fields(&self) -> Vec<&str> {
        Vec::new()
    }

    /// If `true`, this is a *diagnostic* benchmark: its score is reported (and
    /// compared against `threshold` as a reference line) but it does **not**
    /// gate the suite — failing it never makes `brain bench` exit non-zero.
    /// Use for tasks whose single-run result is inherently high-variance or
    /// budget-bound (e.g. a grokking/generalization probe), where a hard
    /// pass/fail bar would be flaky rather than meaningful.
    fn informational(&self) -> bool {
        false
    }
}

/// One benchmark's outcome: its metrics and whether it cleared its threshold.
pub struct Outcome {
    pub name: String,
    pub metrics: Metrics,
    pub threshold: f32,
    pub passed: bool,
    /// Diagnostic-only benchmark (see [`Benchmark::informational`]): reported
    /// but not counted toward the suite's pass/fail.
    pub informational: bool,
}

/// All registered benchmarks. Sibling agents add new ones here (MAD, formal
/// languages, scaling sweeps, …) by pushing another boxed [`Benchmark`].
pub fn registry() -> Vec<Box<dyn Benchmark>> {
    vec![
        Box::new(mqar::Mqar::default()),
        Box::new(toolcall::Toolcall::default()),
        Box::new(mad_recall::MadRecall::default()),
        Box::new(mad_fuzzy_recall::MadFuzzyRecall::default()),
        Box::new(mad_noisy_recall::MadNoisyRecall::default()),
        Box::new(mad_selective_copy::MadSelectiveCopy::default()),
        Box::new(mad_memorize::MadMemorize::default()),
        // Formal-language / algorithmic state-tracking benchmarks.
        Box::new(parity::Parity::default()),
        Box::new(mod_add::ModAdd::default()),
        Box::new(dyck::Dyck::default()),
        // Non-LM objective: a bottleneck autoencoder with an MSE Regression head
        // (ADR §6 / PR-10) — sequence -> single compressed `z` -> reconstruction.
        Box::new(mad_compress::MadCompress::default()),
    ]
}

/// A **smoke** registry: the same benchmarks as [`registry`] but with their step
/// budgets (and corpus / eval sizes) slashed so the whole battery runs in a
/// couple of minutes on the CPU backend. Scores from a smoke run are *not*
/// meaningful as architecture quality — it exists purely so the eval harness +
/// artifact path can be exercised end-to-end in a fast integration test (see
/// `tests/eval.rs`). Selected via `brain bench eval --smoke`.
pub fn registry_smoke() -> Vec<Box<dyn Benchmark>> {
    // Tiny but still > chance-shaped: a handful of steps over a small corpus.
    const STEPS: u32 = 30;
    const SEQS: usize = 600;
    const EVALS: usize = 40;
    vec![
        Box::new(mqar::Mqar { steps: STEPS, n_sequences: SEQS, eval_sequences: EVALS, ..Default::default() }),
        Box::new(toolcall::Toolcall { steps: STEPS, n_sequences: SEQS, eval_sequences: EVALS, ..Default::default() }),
        Box::new(mad_recall::MadRecall { steps: STEPS, n_sequences: SEQS, eval_sequences: EVALS, ..Default::default() }),
        Box::new(mad_fuzzy_recall::MadFuzzyRecall { steps: STEPS, n_sequences: SEQS, eval_sequences: EVALS, ..Default::default() }),
        Box::new(mad_noisy_recall::MadNoisyRecall { steps: STEPS, n_sequences: SEQS, eval_sequences: EVALS, ..Default::default() }),
        Box::new(mad_selective_copy::MadSelectiveCopy { steps: STEPS, n_sequences: SEQS, eval_sequences: EVALS, ..Default::default() }),
        Box::new(mad_memorize::MadMemorize { steps: STEPS, n_sequences: SEQS, eval_sequences: EVALS, ..Default::default() }),
        Box::new(parity::Parity { steps: STEPS, n_sequences: SEQS, eval_sequences: EVALS, ..Default::default() }),
        Box::new(mod_add::ModAdd { steps: STEPS, ..Default::default() }),
        Box::new(dyck::Dyck { steps: STEPS, n_sequences: SEQS, eval_sequences: EVALS, ..Default::default() }),
        Box::new(mad_compress::MadCompress { steps: STEPS, n_sequences: SEQS, eval_sequences: EVALS, ..Default::default() }),
    ]
}

/// Look up a benchmark by [`Benchmark::name`].
pub fn get(name: &str) -> Option<Box<dyn Benchmark>> {
    registry().into_iter().find(|b| b.name() == name)
}

/// Prepare + evaluate one benchmark, returning its [`Outcome`].
fn run(bench: &dyn Benchmark, seed: u64) -> std::io::Result<Outcome> {
    let dir = std::env::temp_dir().join(format!("brain_bench_{}_{}", bench.name(), std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    bench.prepare(&dir, seed)?;
    let metrics = bench.evaluate(&dir, seed)?;
    let _ = std::fs::remove_dir_all(&dir);
    let threshold = bench.threshold();
    Ok(Outcome {
        name: bench.name().to_string(),
        metrics: metrics.clone(),
        threshold,
        passed: metrics.score >= threshold,
        informational: bench.informational(),
    })
}

/// Run a single benchmark by name, printing a one-row table. Returns whether it
/// passed (`Err` if the name is unknown).
pub fn run_one(name: &str, seed: u64) -> std::io::Result<bool> {
    let Some(bench) = get(name) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("unknown benchmark '{name}'; known: {}", known_names().join(", ")),
        ));
    };
    let outcome = run(bench.as_ref(), seed)?;
    let ok = outcome.passed || outcome.informational;
    print_table(std::slice::from_ref(&outcome), &[bench.as_ref()]);
    Ok(ok)
}

/// Run every registered benchmark, printing one comparison table. Returns
/// whether **all** passed.
pub fn run_all(seed: u64) -> std::io::Result<bool> {
    let benches = registry();
    let mut outcomes = Vec::new();
    for b in &benches {
        outcomes.push(run(b.as_ref(), seed)?);
    }
    let refs: Vec<&dyn Benchmark> = benches.iter().map(|b| b.as_ref()).collect();
    print_table(&outcomes, &refs);
    // Informational (diagnostic) benchmarks are reported but never gate the suite.
    Ok(outcomes.iter().all(|o| o.informational || o.passed))
}

/// The names of all registered benchmarks.
pub fn known_names() -> Vec<String> {
    registry().iter().map(|b| b.name().to_string()).collect()
}

/// Print the `benchmark | metric(s) | threshold | pass/fail` comparison table.
fn print_table(outcomes: &[Outcome], benches: &[&dyn Benchmark]) {
    // Union of report fields across the shown benchmarks, in first-seen order.
    let mut field_cols: Vec<String> = Vec::new();
    for b in benches {
        for f in b.report_fields() {
            if !field_cols.iter().any(|c| c == f) {
                field_cols.push(f.to_string());
            }
        }
    }

    let mut header = format!("{:<14} {:>10}", "benchmark", "score");
    for f in &field_cols {
        header.push_str(&format!(" {f:>12}"));
    }
    header.push_str(&format!(" {:>10} {:>6}", "threshold", "result"));
    println!("\n{header}");
    println!("{}", "-".repeat(header.len()));

    for o in outcomes {
        let mut row = format!("{:<14} {:>10.4}", o.name, o.metrics.score);
        for f in &field_cols {
            match o.metrics.get(f) {
                Some(v) => row.push_str(&format!(" {v:>12.4}")),
                None => row.push_str(&format!(" {:>12}", "-")),
            }
        }
        let result = if o.informational {
            "INFO"
        } else if o.passed {
            "PASS"
        } else {
            "FAIL"
        };
        row.push_str(&format!(" {:>10.4} {:>6}", o.threshold, result));
        println!("{row}");
    }
    println!();
}
