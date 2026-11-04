// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fitting open-weight language
// models into the memory a deployment has, by measurement, for its clients. If
// your team needs expertise in mixed-precision model compression then you can
// procure our services by sending an email to info@swedishembedded.com.

#![cfg(all(feature = "study", feature = "text"))]

//! `brain::ModelOptimizer` end to end on a synthetic chat model.
//!
//! * A search writes its report: a plan within the budget whose policy parses
//!   as `--weight-format` and never splits a fused projection, measured on the
//!   deployed kernels against the original and every uniform format that fits,
//!   with `accepted` decided by those measurements alone.
//! * The calibration and held-out records are disjoint and cover the dataset.
//! * Asked to, it distills a recovery adapter over the base held at the plan
//!   and measures the plan without and with it on the same build.
//! * Every planned format is one the architecture serves per linear. A default
//!   candidate the device would widen (int8 on the CPU backend) is left out and
//!   reported; one named by the caller that cannot be measured on its deployed
//!   kernels or served, and a run missing its dataset or out directory, are
//!   refused by name.

use std::path::{Path, PathBuf};

use brain::ModelOptimizer;

mod chat_fixture;
use chat_fixture::{chat_model_dir, chat_model_dir_of, cpu};

fn write_dataset(dir: &Path, records: usize) -> PathBuf {
    let words = ["cab", "bad", "fed", "deaf", "ace", "face", "bag", "gab", "dab", "bead", "fade", "cafe"];
    let lines: Vec<String> = (0..records)
        .map(|i| {
            serde_json::json!({
                "messages": [
                    {"role": "user", "content": words[i % words.len()], "train": false},
                    {"role": "assistant", "content": words[(i * 5 + 3) % words.len()], "train": true}
                ],
                "tools": []
            })
            .to_string()
        })
        .collect();
    let path = dir.join("domain.jsonl");
    std::fs::write(&path, lines.join("\n")).unwrap();
    path
}

#[test]
fn a_search_reports_a_plan_measured_on_the_deployed_kernels() {
    let model = chat_model_dir_of("optimize", qwen3::QwenConfig::tiny_i8());
    let dataset = write_dataset(&model, 12);
    let out = model.join("run");
    let outcome = ModelOptimizer::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(&dataset).out_dir(&out).run().expect("the search runs");

    assert_eq!(outcome.calibration_records + outcome.held_out_records, 12, "{outcome:?}");
    assert_eq!(outcome.held_out_records, 3, "a quarter is held out");
    let report = &outcome.report;
    assert!(report.plan_bytes <= report.budget_bytes, "{report:?}");
    let policy = model::ops::TierPolicy::parse(&report.policy).expect("the policy is a --weight-format value");
    for l in 0..2 {
        let qkv: Vec<_> = ["wq", "wk", "wv"].iter().map(|p| policy.want(&format!("blocks.{l}.attn.{p}.weight"))).collect();
        let gate_up: Vec<_> = ["gate", "up"].iter().map(|p| policy.want(&format!("blocks.{l}.mlp.{p}.weight"))).collect();
        assert!(qkv.windows(2).all(|w| w[0] == w[1]) && gate_up[0] == gate_up[1], "a fused projection takes one format: {}", report.policy);
    }

    let served = [gpu_core::select::Dtype::F32, gpu_core::select::Dtype::BF16, gpu_core::select::Dtype::I8];
    assert!(policy.dtypes().all(|t| served.contains(&t)), "every planned format is one qwen3 serves per linear: {}", report.policy);
    let labels: Vec<&str> = outcome.deployed.iter().map(|d| d.label.as_str()).collect();
    assert_eq!(&labels[..2], ["plan", "original"]);
    for b in &report.baselines {
        assert!(labels.contains(&b.tier), "uniform {} fits the budget, so it is measured on its kernels: {labels:?}", b.tier);
    }
    assert!(!report.baselines.is_empty(), "uniform int8 fits a budget of its own size");
    let plan = outcome.deployed[0].loss;
    let beats = outcome.deployed.iter().filter(|d| report.baselines.iter().any(|b| b.tier == d.label)).all(|d| plan < d.loss);
    assert_eq!(outcome.accepted, beats, "accepted is decided by the deployed measurements: {:?}", outcome.deployed);
    assert_eq!(outcome.measured_on, "serving engine", "qwen3 is measured through the engine brain serve runs");
    assert!(report.plan_seconds.is_some_and(|s| s > 0.0) && report.f32_seconds.is_some(), "every plan is timed on the device: {report:?}");
    assert!(report.baselines.iter().all(|b| b.seconds.is_some()), "{:?}", report.baselines);
    assert!(outcome.recovery.is_none(), "no recovery unless asked");
    assert!(outcome.deployed.iter().all(|d| d.loss.is_finite()), "{:?}", outcome.deployed);

    eprintln!("policy {}, deployed {:?}, simulated plan {} baselines {:?}, excluded {:?}", report.policy, outcome.deployed, report.plan_loss, report.baselines, outcome.excluded_tiers);
    let written: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&outcome.report_path).expect("the report is written")).expect("the report is JSON");
    assert_eq!(written["report"]["policy"], report.policy.as_str());
    assert_eq!(written["accepted"], outcome.accepted);
}

#[test]
fn a_candidate_that_cannot_be_measured_is_refused_before_anything_loads() {
    let model = chat_model_dir_of("optimize-refuse", qwen3::QwenConfig::tiny_i8());
    let dataset = write_dataset(&model, 4);
    let base = model.join("model.safetensors");
    let err = ModelOptimizer::from_pretrained(base.to_str().unwrap()).dataset(&dataset).out_dir(model.join("run")).tiers("int8,q4").run().expect_err("q4 has no frozen-base build to measure");
    assert!(err.to_string().contains("q4"), "{err}");
    let err = ModelOptimizer::from_pretrained(base.to_str().unwrap()).dataset(&dataset).out_dir(model.join("run")).tiers("int8,nf4").run().expect_err("qwen3 does not serve nf4 per linear");
    assert!(err.to_string().contains("nf4") && err.to_string().contains("serve"), "{err}");

    let err = ModelOptimizer::from_pretrained(base.to_str().unwrap()).dataset(&dataset).out_dir(model.join("run")).tiers("int8").device(cpu()).run().expect_err("the CPU backend holds int8 as f32");
    assert!(err.to_string().contains("i8") && err.to_string().contains("f32"), "{err}");

    let plain = chat_model_dir("optimize-args");
    let base = plain.join("model.safetensors");
    let err = ModelOptimizer::from_pretrained(base.to_str().unwrap()).out_dir(plain.join("run")).run().expect_err("no dataset");
    assert!(err.to_string().contains("dataset"), "{err}");
    let err = ModelOptimizer::from_pretrained(base.to_str().unwrap()).dataset(&dataset).run().expect_err("no out directory");
    assert!(err.to_string().contains("out directory"), "{err}");
}

#[test]
fn a_default_candidate_the_device_would_widen_is_left_out_and_reported() {
    let model = chat_model_dir_of("optimize-cpu", qwen3::QwenConfig::tiny_i8());
    let dataset = write_dataset(&model, 8);
    let outcome = ModelOptimizer::from_pretrained(model.join("model.safetensors").to_str().unwrap()).dataset(&dataset).out_dir(model.join("run")).device(cpu()).budget_like("bf16").run().expect("the search runs on the CPU backend");
    assert!(outcome.excluded_tiers.iter().any(|e| e.tier == "i8" && e.held_as == "f32"), "{:?}", outcome.excluded_tiers);
    let policy = model::ops::TierPolicy::parse(&outcome.report.policy).expect("a policy");
    assert!(policy.dtypes().all(|t| t != gpu_core::select::Dtype::I8), "an excluded tier is never planned: {}", outcome.report.policy);
}

#[test]
fn recovery_distills_an_adapter_over_the_base_held_at_the_plan() {
    let model = chat_model_dir_of("optimize-recover", qwen3::QwenConfig::tiny_i8());
    let dataset = write_dataset(&model, 12);
    let out = model.join("run");
    let mut steps = 0;
    let outcome = ModelOptimizer::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .dataset(&dataset)
        .out_dir(&out)
        .recover(6)
        .recover_lr(1e-3)
        .recover_rank(4)
        .run_with(|p| {
            if let brain::OptimizerProgress::Recovering { step, .. } = p {
                steps = *step;
            }
        })
        .expect("the search and the recovery run");
    let rec = outcome.recovery.expect("a recovery was asked for");
    assert_eq!((rec.steps, rec.rank, steps), (6, 4, 6), "every step ran and was reported");
    assert!(rec.adapter.is_file(), "the adapter is written: {}", rec.adapter.display());
    assert!(rec.plan_loss.is_finite() && rec.recovered_loss.is_finite(), "{rec:?}");
    assert_ne!(rec.plan_loss, rec.recovered_loss, "the trained adapter changes what the base at the plan computes");
    let written: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&outcome.report_path).unwrap()).unwrap();
    assert_eq!(written["recovery"]["steps"], 6);
}

#[test]
fn a_latency_limit_holds_the_plan_to_a_uniform_format_s_speed() {
    // Uniform int8 meets both limits (its bytes are within bf16's, its time is the limit), so a plan exists on
    // any device, whichever format that device runs fastest at these small shapes.
    let model = chat_model_dir_of("optimize-latency", qwen3::QwenConfig::tiny_i8());
    let dataset = write_dataset(&model, 8);
    let outcome = ModelOptimizer::from_pretrained(model.join("model.safetensors").to_str().unwrap())
        .dataset(&dataset)
        .out_dir(model.join("run"))
        .budget_like("bf16")
        .max_latency_like("int8")
        .run()
        .expect("the search runs");
    let r = &outcome.report;
    let limit = r.latency_limit_seconds.expect("a limit like uniform int8");
    assert!(r.plan_seconds.is_some_and(|s| s <= limit * (1.0 + 1e-9)), "plan {:?} s over its limit {limit} s", r.plan_seconds);
    assert!(r.baselines.iter().all(|b| b.seconds.is_some_and(|s| s <= limit * (1.0 + 1e-9))), "{:?}", r.baselines);
}
