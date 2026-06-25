// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The **architecture-eval harness** — run the whole benchmark battery against
//! one registered architecture, aggregate per capability axis, and write a
//! structured, comparable results artifact.
//!
//! This is the turn-key core: `brain bench eval --arch <name>` builds the named
//! [`Arch`](crate::Arch)'s [`DecoderLm`](crate::DecoderLm), runs *every*
//! registered [`Benchmark`](crate::Benchmark) through
//! [`evaluate_with`](crate::Benchmark::evaluate_with), then
//! 1. computes a per-axis aggregate (mean of that axis's benchmark scores),
//! 2. summarizes overall gating pass-rate (informational benchmarks excluded),
//! 3. writes `results/<arch>-<seed>.json` — a stable, diffable artifact —, and
//! 4. prints the familiar comparison table plus the per-axis summary.
//!
//! [`compare`] then loads ≥2 of those artifacts and prints a side-by-side
//! leaderboard so a new architecture is diffed against priors at a glance.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::axes::{axes, axis_of};
use crate::{arch, Benchmark};

/// One benchmark's result within an eval run.
pub struct BenchResult {
    pub name: String,
    pub axis: String,
    pub score: f32,
    pub threshold: f32,
    pub passed: bool,
    pub informational: bool,
    pub metrics: crate::Metrics,
}

/// A full eval run: every benchmark scored against one architecture, plus the
/// per-axis aggregates and gating summary. Serializes to the results artifact.
pub struct EvalReport {
    pub arch: String,
    pub size_label: String,
    pub param_count: u64,
    pub param_count_basis: String,
    pub commit: String,
    pub seed: u64,
    pub smoke: bool,
    pub timestamp: String,
    pub benchmarks: Vec<BenchResult>,
    /// axis -> mean score (only axes with ≥1 benchmark present).
    pub axis_scores: BTreeMap<String, f32>,
    /// Gating benchmarks (non-informational) that passed / total.
    pub gating_passed: usize,
    pub gating_total: usize,
}

impl EvalReport {
    /// Serialize to the artifact JSON object.
    pub fn to_json(&self) -> Value {
        let benches: Vec<Value> = self
            .benchmarks
            .iter()
            .map(|b| {
                json!({
                    "name": b.name,
                    "axis": b.axis,
                    "score": b.score,
                    "threshold": b.threshold,
                    "passed": b.passed,
                    "informational": b.informational,
                    "metrics": b.metrics.to_json(),
                })
            })
            .collect();
        let axis_obj: serde_json::Map<String, Value> =
            self.axis_scores.iter().map(|(k, v)| (k.clone(), json!(*v))).collect();
        json!({
            "arch": self.arch,
            "size": self.size_label,
            "param_count": self.param_count,
            "param_count_basis": self.param_count_basis,
            "commit": self.commit,
            "seed": self.seed,
            "smoke": self.smoke,
            "timestamp": self.timestamp,
            "benchmarks": benches,
            "axis_scores": axis_obj,
            "gating": {
                "passed": self.gating_passed,
                "total": self.gating_total,
                "pass_rate": if self.gating_total == 0 { 0.0 }
                    else { self.gating_passed as f64 / self.gating_total as f64 },
            },
        })
    }
}

/// Short git commit hash via `git rev-parse --short HEAD`, or `"unknown"`.
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

/// An ISO-8601-ish UTC-seconds timestamp from [`std::time::SystemTime`] (no
/// external chrono dep): `"1970-01-01T00:00:00Z+<unix_secs>"` shape is avoided —
/// we emit the unix epoch seconds plus a readable suffix.
fn timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Convert unix seconds to a UTC civil date-time (proleptic Gregorian).
    iso_utc(secs)
}

/// Format unix epoch seconds as `YYYY-MM-DDTHH:MM:SSZ` (UTC), no external deps.
fn iso_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Representative `(vocab, block_size)` for the artifact's headline param count.
/// Param count is dataset-dependent (the token embedding scales with vocab); we
/// report a single number at a representative shape and record the basis so it is
/// unambiguous. Most benchmarks here use a small synthetic vocab and a short
/// block, so this is a faithful order-of-magnitude.
const PARAM_VOCAB: u32 = 64;
const PARAM_BLOCK: u32 = 32;

/// Run the full battery for architecture `arch_name` and return the report.
/// `smoke` selects the reduced-budget [`registry_smoke`](crate::registry_smoke)
/// (fast, scores not meaningful) instead of the calibrated [`registry`](crate::registry).
pub fn run(arch_name: &str, seed: u64, smoke: bool) -> io::Result<EvalReport> {
    let a = arch::get_arch(arch_name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "unknown architecture '{arch_name}'; known: {}",
                arch::arch_names().join(", ")
            ),
        )
    })?;
    let lm = (a.factory)();

    let benches = if smoke { crate::registry_smoke() } else { crate::registry() };
    let mut results = Vec::new();
    for b in &benches {
        let r = run_one(b.as_ref(), lm.as_ref(), seed)?;
        results.push(r);
    }

    // Per-axis aggregate = mean of that axis's benchmark scores.
    let mut axis_scores: BTreeMap<String, f32> = BTreeMap::new();
    for ax in axes() {
        let scores: Vec<f32> =
            results.iter().filter(|r| r.axis == ax).map(|r| r.score).collect();
        if !scores.is_empty() {
            let mean = scores.iter().sum::<f32>() / scores.len() as f32;
            axis_scores.insert(ax.to_string(), mean);
        }
    }

    let gating_total = results.iter().filter(|r| !r.informational).count();
    let gating_passed = results.iter().filter(|r| !r.informational && r.passed).count();

    Ok(EvalReport {
        arch: arch_name.to_string(),
        size_label: a.size.label(),
        param_count: a.param_count(PARAM_VOCAB, PARAM_BLOCK),
        param_count_basis: format!("vocab={PARAM_VOCAB},block_size={PARAM_BLOCK}"),
        commit: git_commit(),
        seed,
        smoke,
        timestamp: timestamp(),
        benchmarks: results,
        axis_scores,
        gating_passed,
        gating_total,
    })
}

/// Prepare + evaluate one benchmark against `lm` in a scratch dir.
fn run_one(b: &dyn Benchmark, lm: &dyn crate::DecoderLm, seed: u64) -> io::Result<BenchResult> {
    // A process-unique counter keeps concurrent eval runs (e.g. parallel tests
    // scoring the same benchmark in the same process) from sharing a scratch dir
    // and clobbering each other's `val.bin`.
    use std::sync::atomic::{AtomicU64, Ordering};
    static UNIQ: AtomicU64 = AtomicU64::new(0);
    let uniq = UNIQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "brain_eval_{}_{}_{}",
        b.name(),
        std::process::id(),
        uniq
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    b.prepare(&dir, seed)?;
    let metrics = b.evaluate_with(lm, &dir, seed)?;
    let _ = std::fs::remove_dir_all(&dir);
    let threshold = b.threshold();
    Ok(BenchResult {
        name: b.name().to_string(),
        axis: axis_of(b.name()).to_string(),
        score: metrics.score,
        threshold,
        passed: metrics.score >= threshold,
        informational: b.informational(),
        metrics,
    })
}

/// Default artifact path for an eval run: `results/<arch>-<seed>.json`.
pub fn default_out_path(arch: &str, seed: u64) -> PathBuf {
    PathBuf::from("results").join(format!("{arch}-{seed}.json"))
}

/// Write the report's JSON artifact to `path` (creating parent dirs).
pub fn write_artifact(report: &EvalReport, path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let pretty =
        serde_json::to_string_pretty(&report.to_json()).map_err(io::Error::other)?;
    std::fs::write(path, pretty)
}

/// Print the per-benchmark comparison table + the per-axis summary for one run.
pub fn print_report(report: &EvalReport) {
    println!(
        "\narchitecture: {}  ({})  params≈{}  [{}]  commit {}  seed {}{}",
        report.arch,
        report.size_label,
        report.param_count,
        report.param_count_basis,
        report.commit,
        report.seed,
        if report.smoke { "  SMOKE" } else { "" },
    );

    let header = format!(
        "{:<16} {:<14} {:>10} {:>10} {:>6}",
        "benchmark", "axis", "score", "threshold", "result"
    );
    println!("\n{header}");
    println!("{}", "-".repeat(header.len()));
    for b in &report.benchmarks {
        let result = if b.informational {
            "INFO"
        } else if b.passed {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "{:<16} {:<14} {:>10.4} {:>10.4} {:>6}",
            b.name, b.axis, b.score, b.threshold, result
        );
    }

    println!("\nper-axis capability scores");
    println!("{}", "-".repeat(34));
    for ax in axes() {
        match report.axis_scores.get(ax) {
            Some(v) => println!("{ax:<20} {v:>10.4}"),
            None => println!("{ax:<20} {:>10}", "-"),
        }
    }
    println!(
        "\ngating: {}/{} passed ({:.1}%)\n",
        report.gating_passed,
        report.gating_total,
        if report.gating_total == 0 {
            0.0
        } else {
            100.0 * report.gating_passed as f64 / report.gating_total as f64
        },
    );
}

// ---------------------------------------------------------------------------
// compare
// ---------------------------------------------------------------------------

/// A loaded results artifact, the subset `compare` needs.
pub struct LoadedReport {
    pub arch: String,
    pub seed: u64,
    pub param_count: u64,
    pub commit: String,
    pub gating_pass_rate: f64,
    pub axis_scores: BTreeMap<String, f32>,
    /// benchmark name -> score (for the per-benchmark rows).
    pub bench_scores: BTreeMap<String, f32>,
    /// Display label (`arch@commit` or `arch-seed` if commits collide-ish).
    pub label: String,
}

/// Load a results artifact written by [`write_artifact`].
pub fn load_artifact(path: &Path) -> io::Result<LoadedReport> {
    let text = std::fs::read_to_string(path)?;
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let arch = v["arch"].as_str().unwrap_or("?").to_string();
    let seed = v["seed"].as_u64().unwrap_or(0);
    let param_count = v["param_count"].as_u64().unwrap_or(0);
    let commit = v["commit"].as_str().unwrap_or("unknown").to_string();
    let gating_pass_rate = v["gating"]["pass_rate"].as_f64().unwrap_or(0.0);

    let mut axis_scores = BTreeMap::new();
    if let Some(obj) = v["axis_scores"].as_object() {
        for (k, val) in obj {
            if let Some(x) = val.as_f64() {
                axis_scores.insert(k.clone(), x as f32);
            }
        }
    }
    let mut bench_scores = BTreeMap::new();
    if let Some(arr) = v["benchmarks"].as_array() {
        for b in arr {
            if let (Some(n), Some(s)) = (b["name"].as_str(), b["score"].as_f64()) {
                bench_scores.insert(n.to_string(), s as f32);
            }
        }
    }
    let label = format!("{arch}@{commit}");
    Ok(LoadedReport {
        arch,
        seed,
        param_count,
        commit,
        gating_pass_rate,
        axis_scores,
        bench_scores,
        label,
    })
}

/// Load ≥2 artifacts and print a side-by-side leaderboard: one column per
/// architecture, rows = overall pass-rate, per-axis scores, then per-benchmark
/// scores. So a new architecture is diffed against every prior at a glance.
pub fn compare(paths: &[PathBuf]) -> io::Result<()> {
    if paths.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "compare needs at least 2 results artifacts",
        ));
    }
    let reports: Vec<LoadedReport> =
        paths.iter().map(|p| load_artifact(p)).collect::<io::Result<_>>()?;

    // Column headers (labels), and a width that fits.
    let labels: Vec<String> = reports.iter().map(|r| r.label.clone()).collect();
    let col_w = labels.iter().map(|l| l.len()).max().unwrap_or(8).max(8);
    let row_w = 22usize;

    let mut header = format!("{:<row_w$}", "metric");
    for l in &labels {
        header.push_str(&format!(" {l:>col_w$}"));
    }
    println!("\nleaderboard ({} architectures)\n", reports.len());
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    // seed / params / commit context rows.
    print_str_row("seed", &reports.iter().map(|r| r.seed.to_string()).collect::<Vec<_>>(), row_w, col_w);
    print_str_row(
        "params",
        &reports.iter().map(|r| r.param_count.to_string()).collect::<Vec<_>>(),
        row_w,
        col_w,
    );
    print_str_row(
        "commit",
        &reports.iter().map(|r| r.commit.clone()).collect::<Vec<_>>(),
        row_w,
        col_w,
    );
    println!("{}", "-".repeat(header.len()));

    // overall gating pass-rate.
    print_f32_row(
        "gating pass-rate",
        &reports.iter().map(|r| r.gating_pass_rate as f32).collect::<Vec<_>>(),
        row_w,
        col_w,
    );
    println!("{}", "-".repeat(header.len()));

    // per-axis scores.
    for ax in axes() {
        let vals: Vec<Option<f32>> =
            reports.iter().map(|r| r.axis_scores.get(ax).copied()).collect();
        print_opt_row(&format!("axis:{ax}"), &vals, row_w, col_w);
    }
    println!("{}", "-".repeat(header.len()));

    // per-benchmark scores (union of names across reports, sorted).
    let mut names: Vec<String> = Vec::new();
    for r in &reports {
        for n in r.bench_scores.keys() {
            if !names.contains(n) {
                names.push(n.clone());
            }
        }
    }
    names.sort();
    for n in &names {
        let vals: Vec<Option<f32>> =
            reports.iter().map(|r| r.bench_scores.get(n).copied()).collect();
        print_opt_row(n, &vals, row_w, col_w);
    }
    println!();
    Ok(())
}

fn print_str_row(label: &str, vals: &[String], row_w: usize, col_w: usize) {
    let mut row = format!("{label:<row_w$}");
    for v in vals {
        row.push_str(&format!(" {v:>col_w$}"));
    }
    println!("{row}");
}

fn print_f32_row(label: &str, vals: &[f32], row_w: usize, col_w: usize) {
    let mut row = format!("{label:<row_w$}");
    for v in vals {
        row.push_str(&format!(" {v:>col_w$.4}"));
    }
    println!("{row}");
}

fn print_opt_row(label: &str, vals: &[Option<f32>], row_w: usize, col_w: usize) {
    let mut row = format!("{label:<row_w$}");
    for v in vals {
        match v {
            Some(x) => row.push_str(&format!(" {x:>col_w$.4}")),
            None => row.push_str(&format!(" {:>col_w$}", "-")),
        }
    }
    println!("{row}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_utc_known_epoch() {
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
        // 2021-01-01T00:00:00Z = 1609459200
        assert_eq!(iso_utc(1_609_459_200), "2021-01-01T00:00:00Z");
    }

    #[test]
    fn default_out_path_shape() {
        let p = default_out_path("gpt", 1234);
        assert_eq!(p, PathBuf::from("results/gpt-1234.json"));
    }

    #[test]
    fn metrics_json_roundtrip_in_report() {
        let m = crate::Metrics::new(0.5).with("chance", 0.1);
        let j = m.to_json();
        let back = crate::Metrics::from_json(&j);
        assert!((back.score - 0.5).abs() < 1e-6);
        assert!((back.get("chance").unwrap() - 0.1).abs() < 1e-6);
    }
}
