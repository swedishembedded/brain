// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training-graph tests: (1) anti-drift — the trainer's forward must equal
//! the inference graph's forward on identical weights; (2) finite-difference
//! gradient check of the composed backward; (3) tiny overfit — loss drops.
//! All on the CPU backend with the tiny fixture-shaped config, random
//! weights generated in-test (no fixtures needed).

use std::collections::HashMap;
use wm_diamond::train::{trainable_list, DiamondTrainer};
use wm_diamond::{DiamondConfig, DiamondUNet, Tensors};

fn h2_cfg() -> DiamondConfig {
    // channels 16 (attn heads=2) but still 1 GN group (16<32); to force
    // multi-group set channels to 64 -> heavy. Use 16 for heads, and a
    // separate 64-ch check for groups.
    DiamondConfig {
        img_channels: 3, num_steps_conditioning: 2, cond_channels: 32,
        depths: vec![1, 1], channels: vec![16, 16], attn_depths: vec![false, true],
        num_actions: 4, h: 8, w: 8, sigma_data: 0.5, sigma_offset_noise: 0.3,
    }
}


fn g2_cfg() -> DiamondConfig {
    // channels 64 -> GroupNorm num_groups = 2 (the multi-group path) AND
    // attention heads = 8. The real-config bug lives here.
    DiamondConfig {
        img_channels: 3, num_steps_conditioning: 2, cond_channels: 64,
        depths: vec![1, 1], channels: vec![64, 64], attn_depths: vec![false, true],
        num_actions: 4, h: 8, w: 8, sigma_data: 0.5, sigma_offset_noise: 0.3,
    }
}

fn tiny_cfg() -> DiamondConfig {
    DiamondConfig {
        img_channels: 3,
        num_steps_conditioning: 2,
        cond_channels: 16,
        depths: vec![1, 1],
        channels: vec![8, 8],
        attn_depths: vec![false, true],
        num_actions: 4,
        h: 8,
        w: 8,
        sigma_data: 0.5,
        sigma_offset_noise: 0.3,
    }
}

/// Deterministic pseudo-random tensors (SplitMix64-ish).
fn randn(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn random_tensors(cfg: &DiamondConfig, seed: u64) -> Tensors {
    random_tensors_scaled(cfg, seed, 0.25)
}
fn random_tensors_scaled(cfg: &DiamondConfig, seed: u64, scale: f32) -> Tensors {
    let mut t: Tensors = HashMap::new();
    for (i, (name, shape)) in cfg.param_list().into_iter().enumerate() {
        let n: usize = shape.iter().product();
        // conv weights fan-in-scaled so deep stacks don't blow up the loss.
        let sc = if name.contains(".weight") && shape.len() == 4 {
            scale / (shape[1] as f32 * shape[2] as f32 * shape[3] as f32).sqrt()
        } else { scale };
        t.insert(name, (shape, randn(seed + i as u64 * 7919, n, sc)));
    }
    t
}

struct Fixture {
    obs: Vec<f32>,
    clean: Vec<f32>,
    noise: Vec<f32>,
    sigma: f32,
    actions: Vec<u32>,
}

fn fixture(cfg: &DiamondConfig) -> Fixture {
    let n_px = (cfg.img_channels * cfg.h * cfg.w) as usize;
    let nsc = cfg.num_steps_conditioning as usize;
    Fixture {
        obs: randn(11, n_px * nsc, 0.8),
        clean: randn(22, n_px, 0.8),
        noise: randn(33, n_px, 1.0),
        sigma: 0.7,
        actions: vec![1, 3],
    }
}

#[test]
fn train_forward_matches_inference_graph() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let cfg = tiny_cfg();
    let tensors = random_tensors(&cfg, 5);
    let tr = DiamondTrainer::from_tensors(cfg.clone(), &tensors, Some("cpu"));
    let fx = fixture(&cfg);
    tr.set_transition(&fx.obs, &fx.clean, &fx.noise, fx.sigma, &fx.actions);
    let _ = tr.forward_loss();
    let f_train = tr.read_output();

    // Same math through the INFERENCE graph.
    let unet = DiamondUNet::new(cfg.clone(), &tensors, Some("cpu"));
    let cs = wm_diamond::cond::conditioners(fx.sigma, cfg.sigma_data, cfg.sigma_offset_noise);
    let s_eff =
        (fx.sigma * fx.sigma + cfg.sigma_offset_noise * cfg.sigma_offset_noise).sqrt();
    let noisy: Vec<f32> = fx.clean.iter().zip(&fx.noise).map(|(c, n)| c + s_eff * n).collect();
    let x_scaled: Vec<f32> = noisy.iter().map(|v| v * cs.c_in).collect();
    let obs_rescaled: Vec<f32> = fx.obs.iter().map(|v| v / cfg.sigma_data).collect();
    unet.set_context(&obs_rescaled);
    let f_ref = unet.forward(&x_scaled, cs.c_noise, &fx.actions);

    let max = f_train.iter().zip(&f_ref).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max < 1e-5, "train vs inference forward diverged: {max}");
}

#[test]
fn train_gradcheck_finite_differences() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let cfg = tiny_cfg();
    let tensors = random_tensors(&cfg, 9);
    let tr = DiamondTrainer::from_tensors(cfg.clone(), &tensors, Some("cpu"));
    let fx = fixture(&cfg);
    tr.set_transition(&fx.obs, &fx.clean, &fx.noise, fx.sigma, &fx.actions);

    tr.zero_grads();
    let _ = tr.forward_loss();
    tr.backward();

    // Directional FD vs analytic <grad, v> on several parameters, matching
    // crates/gradcheck tolerances (eps 5e-3, atol 4e-3, rtol 8e-2).
    let eps = 5e-3f32;
    let names = ["unet.d_blocks.0.resblocks.0.conv1.weight", "conv_in.weight", "unet.d_blocks.1.resblocks.0.conv1.weight",
        "unet.mid_blocks.resblocks.0.attn.qkv_proj.weight",
        "unet.u_blocks.1.resblocks.1.conv2.weight", "conv_out.weight", "conv_out.bias"];
    for (pi, name) in names.iter().enumerate() {
        let w0 = tr.read_weight(name);
        let g = tr.read_grad(name);
        let dir = randn(1000 + pi as u64, w0.len(), 1.0);
        let analytic: f32 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
        let wp: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w + eps * d).collect();
        tr.write_weight(name, &wp);
        let lp = tr.forward_loss();
        let wm: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w - eps * d).collect();
        tr.write_weight(name, &wm);
        let lm = tr.forward_loss();
        tr.write_weight(name, &w0);
        let numeric = (lp - lm) / (2.0 * eps);
        let tol = 4e-3 + 8e-2 * analytic.abs().max(numeric.abs());
        assert!(
            (analytic - numeric).abs() < tol,
            "{name}: analytic {analytic} vs numeric {numeric} (tol {tol})"
        );
    }
}

#[test]
fn train_tiny_overfit_reduces_loss() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let cfg = tiny_cfg();
    let tensors = random_tensors(&cfg, 17);
    let tr = DiamondTrainer::from_tensors(cfg.clone(), &tensors, Some("cpu"));
    let fx = fixture(&cfg);
    tr.set_transition(&fx.obs, &fx.clean, &fx.noise, fx.sigma, &fx.actions);
    let l0 = tr.forward_loss();
    for t in 1..=100 {
        tr.zero_grads();
        let _ = tr.forward_loss();
        tr.backward();
        tr.adamw_step(t, 3e-3, 0.0, Some(1.0));
    }
    let l1 = tr.forward_loss();
    assert!(
        l1 < 0.5 * l0,
        "overfit failed to reduce loss: {l0} -> {l1}"
    );
    assert!(l1.is_finite());
}

/// Repeatability probe: identical transition, zero+forward+backward twice —
/// every parameter grad must be BIT-IDENTICAL. Any drift = a backward step
/// reading state that persists across steps (the t=166 NaN divergence).
#[test]
fn train_backward_is_repeatable() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let cfg = tiny_cfg();
    let tensors = random_tensors(&cfg, 5);
    let tr = DiamondTrainer::from_tensors(cfg.clone(), &tensors, Some("cpu"));
    let fx = fixture(&cfg);
    tr.set_transition(&fx.obs, &fx.clean, &fx.noise, fx.sigma, &fx.actions);

    let names: Vec<String> = trainable_list(&cfg).into_iter().map(|(n, _)| n).collect();
    let run = |tr: &DiamondTrainer| -> Vec<(String, Vec<f32>)> {
        tr.zero_grads();
        let _ = tr.forward_loss();
        tr.backward();
        names.iter().map(|n| (n.clone(), tr.read_grad(n))).collect()
    };
    let g1 = run(&tr);
    let g2 = run(&tr);
    for ((n1, a), (_, b)) in g1.iter().zip(&g2) {
        let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(max == 0.0, "grad for {n1} drifted between identical steps: max abs {max}");
    }
}

/// FD gradcheck on the REAL imported Breakout weights (config-dependent
/// backward bugs — multi-group GN, 8-head attention, 4 levels — would hide
/// from the tiny config). Ignored by default: needs out/diamond-breakout.weights.
#[test]
#[ignore]
fn train_gradcheck_real_config() {
    let (cfg, tensors) =
        wm_diamond::import::load("../../out/diamond-breakout.weights").expect("import first");
    let tr = DiamondTrainer::from_tensors(cfg.clone(), &tensors, Some("cpu"));
    let n_px = (cfg.img_channels * cfg.h * cfg.w) as usize;
    let nsc = cfg.num_steps_conditioning as usize;
    let fx = Fixture {
        obs: randn(11, n_px * nsc, 0.8),
        clean: randn(22, n_px, 0.8),
        noise: randn(33, n_px, 1.0),
        sigma: 0.7,
        actions: vec![1, 3, 0, 2],
    };
    tr.set_transition(&fx.obs, &fx.clean, &fx.noise, fx.sigma, &fx.actions);
    tr.zero_grads();
    let _ = tr.forward_loss();
    tr.backward();
    let eps = 1e-3f32;
    for (pi, name) in ["conv_out.weight", "unet.mid_blocks.resblocks.0.attn.qkv_proj.weight", "conv_in.weight"].iter().enumerate() {
        let w0 = tr.read_weight(name);
        let g = tr.read_grad(name);
        let dir = randn(2000 + pi as u64, w0.len(), 1.0);
        let analytic: f32 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
        let wp: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w + eps * d).collect();
        tr.write_weight(name, &wp);
        let lp = tr.forward_loss();
        let wm: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w - eps * d).collect();
        tr.write_weight(name, &wm);
        let lm = tr.forward_loss();
        tr.write_weight(name, &w0);
        let numeric = (lp - lm) / (2.0 * eps);
        eprintln!("{name}: analytic {analytic:.6e} vs FD {numeric:.6e}");
        let tol = 4e-3 + 8e-2 * analytic.abs().max(numeric.abs());
        assert!((analytic - numeric).abs() < tol, "{name} mismatch");
    }
}

fn gradcheck_cfg(cfg: DiamondConfig, seed: u64, names: &[&str]) {
    let tensors = random_tensors(&cfg, seed);
    let tr = DiamondTrainer::from_tensors(cfg.clone(), &tensors, Some("cpu"));
    let fx = fixture(&cfg);
    tr.set_transition(&fx.obs, &fx.clean, &fx.noise, fx.sigma, &fx.actions);
    tr.zero_grads();
    let _ = tr.forward_loss();
    tr.backward();
    let eps = 5e-3f32;
    for (pi, name) in names.iter().enumerate() {
        let w0 = tr.read_weight(name);
        let g = tr.read_grad(name);
        let dir = randn(3000 + pi as u64, w0.len(), 1.0);
        let analytic: f32 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
        let wp: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w + eps * d).collect();
        tr.write_weight(name, &wp); let lp = tr.forward_loss();
        let wm: Vec<f32> = w0.iter().zip(&dir).map(|(w, d)| w - eps * d).collect();
        tr.write_weight(name, &wm); let lm = tr.forward_loss();
        tr.write_weight(name, &w0);
        let numeric = (lp - lm) / (2.0 * eps);
        eprintln!("{name}: analytic {analytic:.5e} FD {numeric:.5e}");
        let tol = 4e-3 + 8e-2 * analytic.abs().max(numeric.abs());
        assert!((analytic - numeric).abs() < tol, "{name}: {analytic} vs {numeric}");
    }
}

#[test]
fn train_gradcheck_multihead() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    gradcheck_cfg(h2_cfg(), 41, &["unet.d_blocks.0.resblocks.0.conv1.weight", "unet.mid_blocks.resblocks.0.attn.qkv_proj.weight",
        "unet.mid_blocks.resblocks.0.attn.out_proj.weight", "conv_out.weight"]);
}

#[test]
fn train_gradcheck_multigroup() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    gradcheck_cfg(g2_cfg(), 43, &["unet.d_blocks.0.resblocks.0.conv1.weight",
        "unet.mid_blocks.resblocks.0.attn.qkv_proj.weight", "conv_in.weight"]);
}

fn depth2_cfg() -> DiamondConfig {
    DiamondConfig { img_channels: 3, num_steps_conditioning: 2, cond_channels: 16,
        depths: vec![2, 2], channels: vec![8, 8], attn_depths: vec![false, false],
        num_actions: 4, h: 8, w: 8, sigma_data: 0.5, sigma_offset_noise: 0.3 }
}

#[test]
fn train_forward_matches_inference_depth2() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let cfg = depth2_cfg();
    let tensors = random_tensors(&cfg, 5);
    let tr = DiamondTrainer::from_tensors(cfg.clone(), &tensors, Some("cpu"));
    let fx = fixture(&cfg);
    tr.set_transition(&fx.obs, &fx.clean, &fx.noise, fx.sigma, &fx.actions);
    let _ = tr.forward_loss();
    let f_train = tr.read_output();
    let unet = DiamondUNet::new(cfg.clone(), &tensors, Some("cpu"));
    let cs = wm_diamond::cond::conditioners(fx.sigma, cfg.sigma_data, cfg.sigma_offset_noise);
    let s_eff = (fx.sigma*fx.sigma + cfg.sigma_offset_noise*cfg.sigma_offset_noise).sqrt();
    let noisy: Vec<f32> = fx.clean.iter().zip(&fx.noise).map(|(c,n)| c + s_eff*n).collect();
    let x_scaled: Vec<f32> = noisy.iter().map(|v| v*cs.c_in).collect();
    let obs_rescaled: Vec<f32> = fx.obs.iter().map(|v| v/cfg.sigma_data).collect();
    unet.set_context(&obs_rescaled);
    let f_ref = unet.forward(&x_scaled, cs.c_noise, &fx.actions);
    let max = f_train.iter().zip(&f_ref).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert!(max < 1e-5, "train vs inference forward diverged at depth 2: {max}");
}
