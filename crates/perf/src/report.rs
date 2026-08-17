// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rendering results, and comparing them across models and hardware.
//!
//! The comparison rules exist because the easiest way to produce a wrong
//! conclusion is to compare two runs that were not comparable:
//!
//! * results whose `artifact_unit` differs are **never ranked** — tokens/s and
//!   frames/s are not the same axis;
//! * `valid: false` results are excluded entirely;
//! * a result whose correctness gate **never ran** is warned about by name.
//!   "checked and passed" and "never checked" are different facts, and a
//!   comparison that renders them identically hands an optimisation a green
//!   light nothing verified;
//! * every environment/target/workload axis that differs between runs prints a
//!   **warning line**, so an accidental apples-to-oranges comparison is loud
//!   rather than silent;
//! * a run on a software rasteriser is labelled as such wherever it appears.

use serde_json::Value;

/// One loaded artifact, reduced to what the leaderboard needs.
pub struct Row {
    pub path: String,
    pub scenario: String,
    pub model: String,
    pub unit: String,
    pub label: String,
    pub valid: bool,
    pub invalid_reason: Option<String>,
    pub smoke: bool,
    /// Whether a correctness gate ran at all (`correctness.passed` non-null).
    /// A `false` here is NOT a failed gate - a failed gate sets `valid: false`
    /// and is excluded outright. It means nothing verified that these numbers
    /// came from the right computation, which is a different fact from
    /// "verified", and one the reader has to be told.
    pub fidelity_checked: bool,
    pub software_gpu: bool,
    pub output_per_s: Option<f64>,
    pub goodput_per_s: Option<f64>,
    pub ttfa_p99: Option<f64>,
    pub ial_p99: Option<f64>,
    pub axes: Vec<(String, String)>,
}

fn f(v: &Value) -> Option<f64> {
    v.as_f64()
}

pub fn load(path: &str) -> Result<Row, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    if v["schema"].as_str() != Some(crate::schema::SCHEMA) {
        return Err(format!("{path}: not a {} artifact", crate::schema::SCHEMA));
    }
    let env = &v["env"];
    let perf = &v["performance"];
    let mut axes = Vec::new();
    for k in ["device", "backend", "adapter", "build"] {
        axes.push((k.to_string(), env[k].as_str().unwrap_or("-").to_string()));
    }
    axes.push(("cpu_cores".into(), env["cpu"]["cores"].to_string()));
    for k in ["quant", "artifact_unit"] {
        axes.push((k.to_string(), v["target"][k].as_str().unwrap_or("-").to_string()));
    }
    for k in ["name", "arrival", "concurrency", "num_requests"] {
        axes.push((format!("workload.{k}"), v["workload"][k].to_string()));
    }

    let software_gpu = env["adapter_is_software"].as_bool().unwrap_or(false);
    let device = env["device"].as_str().unwrap_or("?");
    let adapter = env["adapter"].as_str().unwrap_or("");
    let short = adapter.split_whitespace().next().unwrap_or("");
    let label = match (short.is_empty(), software_gpu) {
        (true, _) => format!("{device}/{}c", env["cpu"]["cores"]),
        (false, true) => format!("{device}/{short}(sw)"),
        (false, false) => format!("{device}/{short}"),
    };

    Ok(Row {
        path: path.to_string(),
        scenario: v["scenario"].as_str().unwrap_or("?").to_string(),
        model: v["target"]["model"].as_str().unwrap_or("?").to_string(),
        unit: v["target"]["artifact_unit"].as_str().unwrap_or("?").to_string(),
        label,
        valid: v["valid"].as_bool().unwrap_or(true),
        invalid_reason: v["invalid_reason"].as_str().map(|s| s.to_string()),
        smoke: v["smoke"].as_bool().unwrap_or(false),
        fidelity_checked: !v["correctness"]["passed"].is_null(),
        software_gpu,
        output_per_s: f(&perf["output_artifacts_per_s"]),
        goodput_per_s: f(&perf["goodput_per_s"]),
        ttfa_p99: f(&perf["ttfa_ms"]["p99"]),
        ial_p99: f(&perf["ial_ms"]["p99"]),
        axes,
    })
}

fn cell(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.1}"),
        None => "—".to_string(),
    }
}

/// Render the leaderboard for a set of artifacts.
pub fn compare(paths: &[String]) -> String {
    let mut out = String::new();
    let mut rows = Vec::new();
    for p in paths {
        match load(p) {
            Ok(r) => rows.push(r),
            Err(e) => out.push_str(&format!("skipped: {e}\n")),
        }
    }
    if rows.is_empty() {
        out.push_str("no comparable artifacts\n");
        return out;
    }

    // Invalid results are reported, then excluded.
    let (valid, invalid): (Vec<_>, Vec<_>) = rows.into_iter().partition(|r| r.valid);
    for r in &invalid {
        out.push_str(&format!(
            "EXCLUDED (correctness gate failed): {} — {}\n",
            r.path,
            r.invalid_reason.clone().unwrap_or_else(|| "no reason recorded".into())
        ));
    }
    if valid.is_empty() {
        out.push_str("no valid artifacts left to compare\n");
        return out;
    }

    // Never rank across artifact units.
    let mut units: Vec<&str> = valid.iter().map(|r| r.unit.as_str()).collect();
    units.sort_unstable();
    units.dedup();
    if units.len() > 1 {
        out.push_str(&format!(
            "REFUSING to rank: artifacts use different units ({}). \
             Rates in different units are not comparable — compare within a unit.\n",
            units.join(", ")
        ));
        return out;
    }
    let unit = valid[0].unit.clone();

    // Warn on every axis that differs.
    let mut warned = Vec::new();
    if let Some(first) = valid.first() {
        for (i, (name, val)) in first.axes.iter().enumerate() {
            let differs = valid.iter().any(|r| r.axes.get(i).map(|(_, v)| v != val).unwrap_or(true));
            if differs {
                warned.push(name.clone());
            }
        }
    }
    if !warned.is_empty() {
        out.push_str(&format!(
            "WARNING: these axes differ across the compared runs — {}.\n\
             Only compare deliberately along ONE axis (same model across hardware, \
             or same hardware across configs).\n",
            warned.join(", ")
        ));
    }
    if valid.iter().any(|r| r.smoke) && valid.iter().any(|r| !r.smoke) {
        out.push_str("WARNING: mixing --smoke and full runs; smoke numbers are not comparable.\n");
    }
    // "Never checked" and "checked and passed" render identically in the table
    // below, so say the difference out loud: an unverified number ranks here on
    // the assumption that its computation was still right, and nothing tested
    // that assumption.
    let unverified: Vec<&str> = valid.iter().filter(|r| !r.fidelity_checked).map(|r| r.path.as_str()).collect();
    if !unverified.is_empty() {
        out.push_str(&format!(
            "WARNING: {} of {} compared runs never ran a correctness gate ({}).\n\
             UNVERIFIED is not a passing gate: nothing here says those numbers came from the \
             right computation.\n",
            unverified.len(),
            valid.len(),
            unverified.join(", ")
        ));
    }
    if valid.iter().any(|r| r.software_gpu) {
        out.push_str("NOTE: (sw) marks a software rasteriser — not a hardware GPU result.\n");
    }

    // Rank by goodput (the primary metric), falling back to output rate.
    let mut ranked = valid;
    ranked.sort_by(|a, b| {
        let ka = a.goodput_per_s.or(a.output_per_s).unwrap_or(f64::NEG_INFINITY);
        let kb = b.goodput_per_s.or(b.output_per_s).unwrap_or(f64::NEG_INFINITY);
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });

    out.push_str(&format!(
        "\n{:<22} {:<10} {:<18} {:>12} {:>12} {:>11} {:>10}\n",
        "artifact", "model", "hardware", "out/s", "goodput/s", "ttfa p99", "ial p99"
    ));
    out.push_str(&format!("{:-<100}\n", ""));
    for r in &ranked {
        let name = std::path::Path::new(&r.path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| r.path.clone());
        let name: String = name.chars().take(22).collect();
        out.push_str(&format!(
            "{:<22} {:<10} {:<18} {:>12} {:>12} {:>11} {:>10}\n",
            name,
            r.model.chars().take(10).collect::<String>(),
            r.label.chars().take(18).collect::<String>(),
            cell(r.output_per_s),
            cell(r.goodput_per_s),
            cell(r.ttfa_p99),
            cell(r.ial_p99),
        ));
    }
    out.push_str(&format!(
        "\nrates are {unit}s/s. goodput = output meeting the workload SLO; it, not out/s, is the comparison metric.\n"
    ));
    out
}

/// One-run summary table.
pub fn render(artifact: &crate::schema::Artifact) -> String {
    let j = artifact.to_json();
    let p = &j["performance"];
    let unit = &artifact.target.artifact_unit;
    let mut s = String::new();
    s.push_str(&format!(
        "\n{} — {} on {}\n",
        artifact.scenario,
        artifact.target.model,
        artifact.env.label()
    ));
    if artifact.env.is_software_gpu() {
        s.push_str("  ! software rasteriser: this is NOT a hardware GPU result\n");
    }
    if !artifact.valid {
        s.push_str(&format!(
            "  ! INVALID: {}\n",
            artifact.invalid_reason.clone().unwrap_or_default()
        ));
    }
    if j["correctness"]["passed"].is_null() {
        s.push_str("  ! correctness gate did not run — result is unverified\n");
    }
    s.push_str(&format!("  {:<26} {}\n", "workload", j["workload"]["name"].as_str().unwrap_or("-")));
    s.push_str(&format!("  {:<26} {}\n", "arrival", j["workload"]["arrival"].as_str().unwrap_or("-")));
    let show = |k: &str, label: &str, suffix: &str| -> String {
        match p[k].as_f64() {
            Some(v) => format!("  {label:<26} {v:.2}{suffix}\n"),
            None => format!("  {label:<26} —\n"),
        }
    };
    s.push_str(&show("requests_per_s", "requests/s", ""));
    s.push_str(&show("output_artifacts_per_s", &format!("output {unit}s/s"), ""));
    s.push_str(&show("goodput_per_s", "goodput/s", ""));
    s.push_str(&show("slo_attainment", "slo attainment", ""));
    for (k, label) in [("ttfa_ms", "ttfa ms"), ("ial_ms", "ial ms"), ("e2e_ms", "e2e ms")] {
        let d = &p[k];
        if d["p50"].is_null() {
            continue;
        }
        s.push_str(&format!(
            "  {:<26} p50 {:.1}  p95 {:.1}  p99 {:.1}  p99.9 {:.1}\n",
            label,
            d["p50"].as_f64().unwrap_or(0.0),
            d["p95"].as_f64().unwrap_or(0.0),
            d["p99"].as_f64().unwrap_or(0.0),
            d["p999"].as_f64().unwrap_or(0.0),
        ));
    }
    if artifact.best_of_n > 1 {
        s.push_str(&format!(
            "  {:<26} best of {} (spread {})\n",
            "repeats",
            artifact.best_of_n,
            artifact.spread_pct.map(|v| format!("{v:.1}%")).unwrap_or_else(|| "—".into())
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write(dir: &std::path::Path, name: &str, v: Value) -> String {
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        p.to_string_lossy().to_string()
    }

    /// SPEC: `compare` must distinguish "checked and passed" from "never
    /// checked". Both used to render identically - a clean, comparable,
    /// apparently-verified row - which is exactly how three of the four perf
    /// targets handed an optimisation a green light nothing had verified.
    #[test]
    fn names_the_runs_whose_correctness_gate_never_ran() {
        let d = tmp("unchecked");
        let checked = write(&d, "checked.json", artifact("token", 100.0, true, false, "wgpu"));
        let mut v = artifact("token", 300.0, true, false, "wgpu");
        v["correctness"]["passed"] = Value::Null;
        let never = write(&d, "never.json", v);

        let out = compare(&[checked.clone(), never.clone()]);
        assert!(out.contains("never ran a correctness gate"), "the gap must be loud:\n{out}");
        assert!(out.contains("never.json"), "it must name WHICH run:\n{out}");
        assert!(!out.contains("checked.json"), "a verified run must not be accused:\n{out}");

        // A comparison where everything was checked stays quiet.
        let quiet = compare(&[checked]);
        assert!(!quiet.contains("never ran a correctness gate"), "{quiet}");
        let _ = std::fs::remove_dir_all(&d);
    }

    fn artifact(unit: &str, out_per_s: f64, valid: bool, software: bool, backend: &str) -> Value {
        json!({
            "schema": crate::schema::SCHEMA,
            "scenario": "serve",
            "valid": valid,
            "invalid_reason": if valid { Value::Null } else { Value::from("greedy mismatch") },
            "smoke": false,
            "env": {
                "device": "gpu", "backend": backend,
                "adapter": if software { "llvmpipe (Cpu, Vulkan)" } else { "NVIDIA (DiscreteGpu, Vulkan)" },
                "adapter_is_software": software,
                "cpu": { "cores": 48 }, "build": "release"
            },
            "target": { "model": "qwen", "artifact_unit": unit, "quant": "fp32" },
            "workload": { "name": "chat", "arrival": "closed_loop", "concurrency": 8, "num_requests": 32 },
            "performance": {
                "output_artifacts_per_s": out_per_s,
                "goodput_per_s": out_per_s * 0.9,
                "ttfa_ms": { "p99": 120.0 },
                "ial_ms": { "p99": 12.0 }
            },
            "correctness": { "passed": true }
        })
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("perf-report-{}-{}", tag, std::process::id()))
    }

    #[test]
    fn ranks_by_goodput_descending() {
        let d = tmp("rank");
        let a = write(&d, "a.json", artifact("token", 100.0, true, false, "wgpu"));
        let b = write(&d, "b.json", artifact("token", 300.0, true, false, "wgpu"));
        let out = compare(&[a, b]);
        // Compare row order, not raw substring position: the header and footer
        // contain plenty of stray letters.
        let order: Vec<&str> = out
            .lines()
            .filter_map(|l| l.split_whitespace().next())
            .filter(|w| *w == "a" || *w == "b")
            .collect();
        assert_eq!(order, vec!["b", "a"], "higher goodput must rank first:\n{out}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn refuses_to_rank_across_artifact_units() {
        let d = tmp("units");
        let a = write(&d, "a.json", artifact("token", 100.0, true, false, "wgpu"));
        let b = write(&d, "b.json", artifact("frame", 300.0, true, false, "wgpu"));
        let out = compare(&[a, b]);
        assert!(out.contains("REFUSING to rank"), "must refuse tokens/s vs frames/s:\n{out}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn excludes_results_that_failed_the_correctness_gate() {
        let d = tmp("invalid");
        let a = write(&d, "a.json", artifact("token", 100.0, true, false, "wgpu"));
        let b = write(&d, "b.json", artifact("token", 9999.0, false, false, "wgpu"));
        let out = compare(&[a, b]);
        assert!(out.contains("EXCLUDED"), "invalid run must be excluded:\n{out}");
        assert!(!out.contains("9999"), "an invalid run must never win the table:\n{out}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn warns_when_a_comparison_axis_differs() {
        let d = tmp("axes");
        let a = write(&d, "a.json", artifact("token", 100.0, true, false, "wgpu"));
        let b = write(&d, "b.json", artifact("token", 300.0, true, false, "cpu"));
        let out = compare(&[a, b]);
        assert!(out.contains("WARNING"), "differing backend must warn:\n{out}");
        assert!(out.contains("backend"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn marks_software_rasteriser_runs() {
        let d = tmp("sw");
        let a = write(&d, "a.json", artifact("token", 100.0, true, true, "wgpu"));
        let out = compare(&[a]);
        assert!(out.contains("(sw)"), "software adapter must be visible in the table:\n{out}");
        assert!(out.contains("software rasteriser"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn rejects_foreign_files() {
        let d = tmp("foreign");
        let a = write(&d, "a.json", json!({ "schema": "something/else" }));
        assert!(load(&a).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
