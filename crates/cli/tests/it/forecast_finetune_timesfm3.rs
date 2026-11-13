// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end test of TimesFM-3's actions through the architecture CLI,
//! resolving the checkpoint the way every `brain <model> <action>` does:
//! `brain timesfm3 finetune --weights <ckpt> --data <csv-dir>` and
//! `brain timesfm3 predict --weights <ckpt> --in csv=<ohlcv.csv>`.
//!
//! Runs the actual compiled `brain` binary against a checkpoint this test
//! writes itself at `Timesfm3Config::tiny()` scale, so it needs no
//! multi-hundred-MB fixture on disk and can run in ordinary CI. What it proves
//! is wiring: the CLI, the universe loader, `timesfm3::finetune` and the
//! licence stamping are connected, not merely individually correct.

use std::path::PathBuf;
use std::process::Command;

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn bin() -> PathBuf {
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.push("brain");
    p
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-cli-tfm3-finetune-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `n` valid daily bars with a trend and a seasonal component - the same CSV
/// shape the universe loader's own tests use, with a close that is actually
/// forecastable so the gate has something to measure.
fn csv(n: usize, phase: f32) -> String {
    let mut s = String::from("Date,open,high,low,close,volume\n");
    for i in 0..n {
        let (month, day) = (1 + i / 28, 1 + i % 28);
        let t = i as f32;
        let c = 100.0 + 0.05 * t + 3.0 * (t * 0.3 + phase).sin();
        s.push_str(&format!("2024-{month:02}-{day:02},{c:.2},{:.2},{:.2},{c:.2},1000\n", c + 2.0, c - 2.0));
    }
    s
}

/// Write a base checkpoint in the brain container layout `load_base` reads.
fn write_base(dir: &std::path::Path) -> String {
    let cfg = timesfm3::Timesfm3Config::tiny();
    let w = timesfm3::train::init_weights(&cfg, 7);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
        cfg.param_list().into_iter().map(|(n, shape)| (n.clone(), shape.iter().map(|&x| x as u64).collect(), w[&n].clone())).collect();
    let path = dir.join("base.safetensors");
    let p = path.to_string_lossy().to_string();
    checkpoint::save(&p, cfg.to_json(), &tensors);
    p
}

/// The action runs TimesFM-3 to a real verdict, and says so with the gate line.
///
/// The licence notice is asserted too, and asserted to be the first line of
/// the verdict (it is also streamed before any work, but a terminal shows only
/// the latest progress line): an operator who spent hours producing an artifact
/// they may not redistribute should be told so where the result is.
#[test]
fn finetune_timesfm3_runs_to_a_gate_verdict() {
    if skip() {
        brain_testutil::skip_unavailable("timesfm3 finetune: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let dir = tmp("run");
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    for (i, t) in ["AAA", "BBB", "CCC"].iter().enumerate() {
        std::fs::write(data.join(format!("{t}.csv")), csv(200, i as f32)).unwrap();
    }
    let base = write_base(&dir);
    let out = dir.join("ft.safetensors");

    let o = Command::new(bin())
        .args([
            "timesfm3",
            "finetune",
            "--weights",
            &base,
            "--data",
            &data.to_string_lossy(),
            "--save",
            &out.to_string_lossy(),
            "--context",
            "8",
            "--horizon",
            "4",
            "--epochs",
            "1",
            "--batch",
            "2",
            "--lr",
            "1e-3",
            "--seed",
            "11",
        ])
        .output()
        .expect("run brain");
    let stdout = String::from_utf8_lossy(&o.stdout);
    let stderr = String::from_utf8_lossy(&o.stderr);
    let all = format!("{stdout}{stderr}");

    assert!(o.status.success(), "exit {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}", o.status.code());
    let verdict = stdout.split("verdict:").nth(1).unwrap_or("").trim_start_matches([' ', '"']);
    assert!(verdict.starts_with("timesfm3: the TimesFM-3 3.0 weights are timesfm-non-commercial-license-v1.0"), "the licence notice opens the verdict\n{stdout}");
    assert!(all.contains("finetune: timesfm3"), "the run must say which model it is training\n{all}");
    assert!(all.contains("gate (INCLUDED names, held-out future):"), "the gate line must be reported\n{all}");
    assert!(all.contains("PROMOTE") || all.contains("KEEP BASE"), "the gate must reach a verdict\n{all}");
    assert!(all.contains("11"), "the seed is echoed\n{all}");

    // Whichever way the gate went, a WRITTEN checkpoint carries the upstream
    // licence and the base it derives from.
    if out.exists() {
        let card = checkpoint::st::read_card(&out.to_string_lossy()).unwrap().expect("a promoted checkpoint carries a card");
        assert_eq!(card.license.as_deref(), Some("timesfm-non-commercial-license-v1.0"));
        assert_eq!(card.variant_of.as_deref(), Some("google/timesfm-3.0-pytorch"));
        assert!(checkpoint::license::redistributable(card.license.as_deref()).is_err(), "the written artifact must not be publishable");
    } else {
        assert!(all.contains("not promoted"), "no checkpoint was written, so the run must say it was not promoted\n{all}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A horizon past one forecast patch is refused by name, rather than
/// panicking somewhere in the stitching arithmetic. `tiny()`'s
/// `stitch_extract_len` is 8.
#[test]
fn a_horizon_past_one_forecast_patch_is_refused() {
    if skip() {
        brain_testutil::skip_unavailable("timesfm3 finetune: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let dir = tmp("horizon");
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    for (i, t) in ["AAA", "BBB"].iter().enumerate() {
        std::fs::write(data.join(format!("{t}.csv")), csv(120, i as f32)).unwrap();
    }
    let base = write_base(&dir);

    let o = Command::new(bin())
        .args(["timesfm3", "finetune", "--weights", &base, "--data", &data.to_string_lossy(), "--context", "8", "--horizon", "9"])
        .output()
        .expect("run brain");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "stderr:\n{stderr}");
    assert!(stderr.contains("stitch_extract_len"), "the refusal must name the limit\nstderr:\n{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A mistyped checkpoint path is an exit code and a sentence, not a panic: a
/// checkpoint the caller names must be one the store can classify.
#[test]
fn a_missing_checkpoint_is_an_error_not_a_panic() {
    let dir = tmp("missing");
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    for (i, t) in ["AAA", "BBB"].iter().enumerate() {
        std::fs::write(data.join(format!("{t}.csv")), csv(60, i as f32)).unwrap();
    }
    let o = Command::new(bin())
        .args(["timesfm3", "finetune", "--weights", &dir.join("nope.safetensors").to_string_lossy(), "--data", &data.to_string_lossy(), "--context", "8", "--horizon", "4"])
        .output()
        .expect("run brain");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "stderr:\n{stderr}");
    assert!(!stderr.contains("panicked"), "a missing checkpoint must not panic\nstderr:\n{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The old verb is gone, and the two models' fine-tune parameters are listed by
/// their own `--help`.
#[test]
fn the_forecast_verb_is_gone_and_each_model_lists_its_finetune_parameters() {
    let gone = Command::new(bin()).args(["forecast", "finetune"]).output().expect("run brain");
    assert!(!gone.status.success(), "`brain forecast` no longer exists");
    for model in ["kronos", "timesfm3"] {
        let h = Command::new(bin()).args([model, "finetune", "--help"]).output().expect("run brain");
        let text = String::from_utf8_lossy(&h.stdout);
        for flag in ["--data", "--save", "--holdout_data", "--skip_invalid", "--seed", "--activations"] {
            assert!(text.contains(flag), "{model} finetune --help lists {flag}\n{text}");
        }
    }
}

/// `predict` scores the model against the baselines on bars it was not shown,
/// through the CLI, and echoes the seed it ran with.
#[test]
fn predict_scores_timesfm3_against_the_baselines_on_held_out_bars() {
    if skip() {
        brain_testutil::skip_unavailable("timesfm3 predict: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let dir = tmp("predict");
    let base = write_base(&dir);
    let csv_path = dir.join("series.csv");
    std::fs::write(&csv_path, csv(120, 0.0)).unwrap();
    let report = dir.join("report.txt");

    let o = Command::new(bin())
        .args(["timesfm3", "predict", "--weights", &base, "--in"])
        .arg(format!("csv={}", csv_path.display()))
        .args(["--horizon", "4", "--context", "16", "--origins", "2", "--season", "7", "--seed", "5", "--out"])
        .arg(format!("report={}", report.display()))
        .output()
        .expect("run brain");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "exit {:?}\nstderr:\n{stderr}", o.status.code());
    let text = std::fs::read_to_string(&report).expect("the report output was written");
    for row in ["persistence (last close)", "drift (context mean return)", "seasonal naive (7 bars)", "vs persistence:"] {
        assert!(text.contains(row), "the report has `{row}`\n{text}");
    }
    assert!(text.contains("2 rolling origins"), "{text}");
    let _ = std::fs::remove_dir_all(&dir);
}
