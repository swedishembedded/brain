// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end test of `brain forecast finetune --timesfm3 <weights> --data
//! <csv-dir>`: the verb used to be Kronos-only, and `--timesfm3` was an
//! unrecognised flag that fell through to the Kronos usage block and exited 2.
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
/// shape `forecast_cli`'s own loader tests use, with a close that is actually
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

/// The verb runs TimesFM-3 to a real verdict, and says so with the same gate
/// line the Kronos path prints.
///
/// The licence notice is asserted too, and asserted to come out BEFORE any
/// work: an operator about to spend hours producing an artifact they may not
/// redistribute should learn that in the first line.
#[test]
fn finetune_timesfm3_runs_to_a_gate_verdict() {
    if skip() {
        brain_testutil::skip_unavailable("forecast finetune --timesfm3: MOE_SKIP_GPU_TESTS set");
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
            "forecast",
            "finetune",
            "--timesfm3",
            &base,
            "--data",
            &data.to_string_lossy(),
            "--out",
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
        ])
        .output()
        .expect("run brain");
    let stdout = String::from_utf8_lossy(&o.stdout);
    let stderr = String::from_utf8_lossy(&o.stderr);

    assert!(o.status.success(), "exit {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}", o.status.code());
    assert!(stderr.contains("timesfm-non-commercial-license-v1.0"), "the licence notice must be printed at run start\nstderr:\n{stderr}");
    assert!(stderr.contains("finetune: timesfm3"), "the run must say which model it is training\nstderr:\n{stderr}");
    assert!(stdout.contains("gate (INCLUDED names, held-out future):"), "the gate line must match the Kronos path's format\nstdout:\n{stdout}");
    assert!(stdout.contains("PROMOTE") || stdout.contains("KEEP BASE"), "the gate must reach a verdict\nstdout:\n{stdout}");

    // Whichever way the gate went, a WRITTEN checkpoint carries the upstream
    // licence and the base it derives from.
    if out.exists() {
        let card = checkpoint::st::read_card(&out.to_string_lossy()).unwrap().expect("a promoted checkpoint carries a card");
        assert_eq!(card.license.as_deref(), Some("timesfm-non-commercial-license-v1.0"));
        assert_eq!(card.variant_of.as_deref(), Some("google/timesfm-3.0-pytorch"));
        assert!(checkpoint::license::redistributable(card.license.as_deref()).is_err(), "the written artifact must not be publishable");
    } else {
        assert!(stdout.contains("not promoted"), "no checkpoint was written, so the run must say it was not promoted\nstdout:\n{stdout}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A horizon past one forecast patch is refused by name with exit 2, rather
/// than panicking somewhere in the stitching arithmetic. `tiny()`'s
/// `stitch_extract_len` is 8.
#[test]
fn a_horizon_past_one_forecast_patch_is_refused() {
    if skip() {
        brain_testutil::skip_unavailable("forecast finetune --timesfm3: MOE_SKIP_GPU_TESTS set");
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
        .args(["forecast", "finetune", "--timesfm3", &base, "--data", &data.to_string_lossy(), "--context", "8", "--horizon", "9"])
        .output()
        .expect("run brain");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(2), "stderr:\n{stderr}");
    assert!(stderr.contains("stitch_extract_len"), "the refusal must name the limit\nstderr:\n{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A mistyped checkpoint path is an exit code and a sentence, not a panic.
/// `checkpoint::load` aborts the process on an unreadable file, so the guard
/// has to sit in front of it.
#[test]
fn a_missing_checkpoint_is_an_error_not_a_panic() {
    let dir = tmp("missing");
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    for (i, t) in ["AAA", "BBB"].iter().enumerate() {
        std::fs::write(data.join(format!("{t}.csv")), csv(60, i as f32)).unwrap();
    }
    let o = Command::new(bin())
        .args(["forecast", "finetune", "--timesfm3", &dir.join("nope.safetensors").to_string_lossy(), "--data", &data.to_string_lossy(), "--context", "8", "--horizon", "4"])
        .output()
        .expect("run brain");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(1), "stderr:\n{stderr}");
    assert!(!stderr.contains("panicked"), "a missing checkpoint must not panic\nstderr:\n{stderr}");
    assert!(stderr.contains("load timesfm3 from"), "stderr:\n{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The Kronos path is untouched: a command line with neither model flag still
/// prints the Kronos usage block and exits 2, and naming both is refused.
#[test]
fn the_kronos_path_and_the_both_flags_refusal_are_unchanged() {
    let bare = Command::new(bin()).args(["forecast", "finetune"]).output().expect("run brain");
    assert_eq!(bare.status.code(), Some(2));
    let e = String::from_utf8_lossy(&bare.stderr);
    assert!(e.contains("--kronos-tokenizer"), "a bare finetune still prints the Kronos usage\nstderr:\n{e}");

    let both = Command::new(bin())
        .args(["forecast", "finetune", "--timesfm3", "w", "--kronos-decoder", "d", "--data", "csv"])
        .output()
        .expect("run brain");
    assert_eq!(both.status.code(), Some(2));
    let e = String::from_utf8_lossy(&both.stderr);
    assert!(e.contains("select different models"), "stderr:\n{e}");
}
