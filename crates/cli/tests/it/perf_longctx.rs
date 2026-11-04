// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain perf run longctx` end to end, once per model family, on the smallest
//! real checkpoint each has: the plumbing from `--target` through the engine
//! adapter, the scenario and the artifact. Numbers are never asserted - only
//! that what is reported is present, finite and honestly labelled.
//!
//! Swedish Embedded AB implements long-context inference serving and the
//! measurement that shows what a given GPU can really sustain for its clients.
//! If your team needs expertise in sizing and tuning LLM serving on a specific
//! accelerator then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Each test skips by name where this machine cannot run it: no GPU (the
//! scenario is GPU-only), or the checkpoint is not on disk.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("brain");
    path
}

/// Run a `--smoke` longctx on `target` and return the artifact.
fn smoke(target: &str, tag: &str, extra: &[&str]) -> serde_json::Value {
    let out = std::env::temp_dir().join(format!("brain-longctx-{tag}-{}.json", std::process::id()));
    let run = Command::new(bin())
        .args(["perf", "run", "longctx", "--target", target, "--smoke", "--ladder", "1,2", "--prefill", "128"])
        .args(extra)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run brain perf");
    assert!(run.status.success(), "brain perf run longctx failed:\n{}", String::from_utf8_lossy(&run.stderr));
    let text = std::fs::read_to_string(&out).expect("the artifact was written");
    let _ = std::fs::remove_file(&out);
    serde_json::from_str(&text).expect("the artifact is JSON")
}

fn assert_artifact_is_honest(a: &serde_json::Value) {
    assert_eq!(a["schema"], "brain.perf/1");
    assert_eq!(a["scenario"], "longctx");
    assert_eq!(a["smoke"], true);
    assert_eq!(a["workload"]["context"], "synthetic");
    assert_eq!(a["workload"]["host_offload"], false);
    assert!(a["env"]["backend"].is_string());

    let rows = a["curve"].as_array().expect("a sweep curve");
    assert!(!rows.is_empty(), "batch 1 must have been measured");
    assert_eq!(rows[0]["batch"], 1);
    for r in rows {
        for k in ["ms_per_step_median", "tok_per_s_total", "tok_per_s_per_stream"] {
            let v = r[k].as_f64().unwrap_or_else(|| panic!("{k} must be a number: {r}"));
            assert!(v.is_finite() && v > 0.0, "{k} = {v}");
        }
        assert!(r["planned_bytes"].as_u64().unwrap() > 0);
    }
    assert!(a["performance"]["single_stream_decode"]["tok_per_s"].as_f64().unwrap() > 0.0);
    assert!(a["performance"]["max_sustained_throughput"]["batch"].is_number());
    let prefill = a["performance"]["prefill"]["rows"].as_array().expect("a prefill table");
    assert_eq!(prefill.len(), 1);
    assert!(prefill[0]["tok_per_s"].as_f64().unwrap() > 0.0);
    assert!(a["performance"]["prefill"]["error"].is_null());
}

#[test]
fn qwen3_paged_engine_runs_longctx() {
    if gpu_core::devices::gpus().is_empty() {
        brain_testutil::skip_unavailable("perf longctx qwen: no GPU (the scenario is GPU-only)");
        return;
    }
    let Some(dir) = brain_testutil::model_dir("Qwen/Qwen3-0.6B").filter(|d| std::path::Path::new(d).join("config.json").is_file()) else {
        brain_testutil::skip_unavailable("perf longctx qwen: Qwen/Qwen3-0.6B is not in the model store");
        return;
    };
    let a = smoke(&format!("qwen:{dir}"), "qwen", &[]);
    assert_artifact_is_honest(&a);
    assert_eq!(a["target"]["config"]["kv_precision"], "int8");
}

#[test]
fn qwen35_gguf_resident_runs_longctx() {
    if gpu_core::devices::gpus().is_empty() {
        brain_testutil::skip_unavailable("perf longctx qwen35: no GPU (the scenario is GPU-only)");
        return;
    }
    // No smaller checkpoint exists for this family and loading the real one is
    // minutes of work, so it is opt-in: point BRAIN_QWEN35_GGUF at it.
    let Ok(_gguf) = std::env::var("BRAIN_QWEN35_GGUF") else {
        brain_testutil::skip_unavailable("perf longctx qwen35: BRAIN_QWEN35_GGUF is not set (no small checkpoint for this family)");
        return;
    };
    let a = smoke("qwen35-gguf", "qwen35", &[]);
    assert_artifact_is_honest(&a);
    assert_eq!(a["target"]["config"]["kv_precision"], "fp32");
}
