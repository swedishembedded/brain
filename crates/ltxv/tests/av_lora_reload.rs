// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A LoRA fine-tune must change what the model actually outputs after a
//! save + reload cycle **in a genuinely separate OS process**, not merely a
//! fresh struct still sharing this process's memory - lesson #23's own
//! third requirement: "(2) save, then load in a fresh process/struct ...
//! (3) assert the reloaded logits differ from the base by a real margin".
//! `crates/qwen3/tests/lora_roundtrip.rs` satisfies (2) with a fresh
//! struct in the SAME process; this test goes one step further and
//! satisfies it with a fresh PROCESS, closing the whole round trip: no
//! static, no thread-local, no accidental shared state between "trained"
//! and "reloaded" can paper over a save/load defect here, because they
//! never share an address space.
//!
//! This file holds exactly ONE test, deliberately - so re-invoking the
//! compiled test binary bare (no filter) runs exactly this one test again,
//! in the child role, rather than the whole crate's suite.
//!
//! ## The self-respawn trick
//!
//! The parent trains a real AV LoRA adapter (several Adam steps, so `B ≠
//! 0`), saves it to disk, and evaluates the LIVE trained model. It then
//! re-executes `std::env::current_exe()` - the SAME compiled test binary -
//! as a CHILD process with `--nocapture` (so the child's `println!` reaches
//! the real stdout pipe the parent captures, bypassing libtest's default
//! output capture) and an env var telling it to take the "child" branch
//! instead of re-running the training. The child rebuilds the BASE weights
//! deterministically from a shared seed (never touching the parent's
//! `base`/`adapter` in memory - the whole point), loads the adapter from
//! disk, evaluates the same fixed input, and prints one marked line the
//! parent parses back out.

use ltxv::av_lora::{load_adapter, save_adapter, LoraAdapter, LoraCfg};
use ltxv::av_modelgrad::{forward, grads, init_model, make_av_flow_batch, AvCfg, AvModelWeights};

const CHILD_ENV: &str = "LTXV_AV_RELOAD_CHILD";
const PATH_ENV: &str = "LTXV_AV_RELOAD_PATH";
const SEED_ENV: &str = "LTXV_AV_RELOAD_SEED";
const MARKER: &str = "RELOAD_OUTPUT:";

/// One deterministic, fixed forward - both streams' predicted velocity,
/// concatenated - the "model output" this test compares across processes.
fn eval_model(cfg: &AvCfg, w: &AvModelWeights<f32>) -> Vec<f32> {
    let v_latent: Vec<f32> = (0..cfg.tv * cfg.v_in_channels).map(|i| (i % 13) as f32 / 13.0 - 0.5).collect();
    let a_latent: Vec<f32> = (0..cfg.ta * cfg.a_in_channels).map(|i| (i % 11) as f32 / 11.0 - 0.5).collect();
    let v_ctx: Vec<f32> = (0..cfg.v_context_len * cfg.vdim).map(|i| (i % 7) as f32 / 7.0 - 0.5).collect();
    let a_ctx: Vec<f32> = (0..cfg.a_context_len * cfg.adim).map(|i| (i % 5) as f32 / 5.0 - 0.5).collect();
    let v_timesteps = vec![0.5f64; cfg.tv];
    let a_timesteps = vec![0.5f64; cfg.ta];
    let v_keyframes_mask = vec![1.0f64; cfg.tv];
    let v_positions = cfg.simple_positions_v();
    let a_positions = cfg.simple_positions_a();
    let (v_rope, a_rope, v_cross, a_cross) = cfg.rope_tables_f32(&v_positions, &a_positions);
    let (v_pred, a_pred, _) = forward(
        cfg, w, &v_latent, &v_timesteps, &v_keyframes_mask, &v_ctx, &a_latent, &a_timesteps, &a_ctx, 0.5, 0.5, &v_rope.cos, &v_rope.sin, &a_rope.cos, &a_rope.sin, &v_cross.cos, &v_cross.sin,
        &a_cross.cos, &a_cross.sin,
    );
    [v_pred, a_pred].concat()
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "mean_abs_diff: length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

/// The child role: rebuild the base deterministically, load the adapter
/// from disk, print the marked eval line, exit. Never touches the parent's
/// in-memory `base`/`adapter` - only the seed (an integer) and the saved
/// file cross the process boundary.
fn run_as_child() {
    let cfg = AvCfg::tiny();
    let seed: u64 = std::env::var(SEED_ENV).expect("seed env").parse().expect("seed is a u64");
    let path = std::env::var(PATH_ENV).expect("path env");
    let base = init_model::<f32>(&cfg, seed);
    let adapter = load_adapter(&path, &cfg).expect("load AV adapter in child process");
    let out = eval_model(&cfg, &adapter.apply(&base));
    let joined: Vec<String> = out.iter().map(|v| format!("{v:.9e}")).collect();
    println!("{MARKER}{}", joined.join(","));
}

#[test]
fn av_lora_reload_survives_a_fresh_process() {
    if std::env::var(CHILD_ENV).is_ok() {
        run_as_child();
        return;
    }

    let cfg = AvCfg::tiny();
    let seed = 0xF00D_1234_5678_9ABCu64;
    let base = init_model::<f32>(&cfg, seed);

    // Real optimiser steps, so B != 0 (a fresh LoRA init is B=0, which would
    // make the reload margin zero regardless of whether loading actually
    // works - lesson #23's own first requirement, and lesson #4's "a
    // degenerate setup hides a whole bug class" in this specific shape).
    let mut adapter = LoraAdapter::new(&cfg, LoraCfg::new(4));
    let mut rng = data::rng::Rng::new(99);
    for _ in 0..15 {
        let v_x0: Vec<f32> = (0..cfg.tv * cfg.v_in_channels).map(|_| rng.next_gaussian() as f32).collect();
        let a_x0: Vec<f32> = (0..cfg.ta * cfg.a_in_channels).map(|_| rng.next_gaussian() as f32).collect();
        let v_ctx: Vec<f32> = (0..cfg.v_context_len * cfg.vdim).map(|_| rng.next_gaussian() as f32).collect();
        let a_ctx: Vec<f32> = (0..cfg.a_context_len * cfg.adim).map(|_| rng.next_gaussian() as f32).collect();
        let v_noise: Vec<f32> = (0..v_x0.len()).map(|_| rng.next_gaussian() as f32).collect();
        let a_noise: Vec<f32> = (0..a_x0.len()).map(|_| rng.next_gaussian() as f32).collect();
        let b = make_av_flow_batch(&cfg, &v_x0, &a_x0, &v_ctx, &a_ctx, 0.5, 0.5, &v_noise, &a_noise);
        let (_l, g) = grads(&cfg, &adapter.apply(&base), &b);
        adapter.step(&g, 5e-3);
    }

    let dir = std::env::temp_dir().join(format!("ltxv-av-lora-reload-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let path = dir.join("adapter.brain");
    let path_str = path.to_str().expect("utf-8 path").to_string();
    save_adapter(&path_str, &adapter);

    let logits_live = eval_model(&cfg, &adapter.apply(&base));
    let logits_base = eval_model(&cfg, &base);
    let live_vs_base_before_reload = mean_abs_diff(&logits_live, &logits_base);
    assert!(live_vs_base_before_reload > 1e-4, "training did not move the live AV model's output away from base: {live_vs_base_before_reload}");

    let exe = std::env::current_exe().expect("current_exe");
    let out = std::process::Command::new(&exe).arg("--nocapture").env(CHILD_ENV, "1").env(PATH_ENV, &path_str).env(SEED_ENV, seed.to_string()).output().expect("spawn fresh-process child");
    assert!(out.status.success(), "child process failed (status {:?}):\nstdout:\n{}\nstderr:\n{}", out.status, String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().find(|l| l.starts_with(MARKER)).unwrap_or_else(|| panic!("child produced no {MARKER} line; stdout was:\n{stdout}"));
    let logits_reloaded: Vec<f32> = line[MARKER.len()..].split(',').map(|s| s.parse().expect("f32")).collect();
    assert_eq!(logits_reloaded.len(), logits_live.len(), "reloaded output has the wrong length");

    let reload_err = mean_abs_diff(&logits_live, &logits_reloaded);
    let base_diff = mean_abs_diff(&logits_live, &logits_base);
    println!("av lora reload (fresh process): live-vs-reloaded={reload_err:.3e}  live-vs-base={base_diff:.3e}");

    assert!(reload_err < 1e-5, "a fresh-process reload of the AV adapter does not reproduce the live trained model: mean abs diff {reload_err:.3e}");
    assert!(base_diff > 1e-4, "training did not move the AV model away from base by a real margin: {base_diff:.3e}");

    let _ = std::fs::remove_dir_all(&dir);
}
