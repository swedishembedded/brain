// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P1 milestone A+B validation for the differentiable Kronos decoder
//! ([`kronos::train::KronosTrain`]): (1) a directional finite-difference gradient
//! check on every trainable parameter (blocks, s1 head, AND the hierarchical +
//! temporal embeddings), and (2) a from-scratch learning test that the tape
//! actually drives loss down (memorizes a fixed batch). Runs on whatever backend
//! `BRAIN_DEVICE` selects, so CI exercises CPU and GPU with the same test.
//!
//! Gated off by `MOE_SKIP_GPU_TESTS`.

use kronos::config::KronosConfig;
use kronos::train::{param_list_c, KronosTrain, LoraCfg, CAL};
use std::collections::HashMap;

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn sym(&mut self, a: f32) -> f32 {
        let u = (self.u64() >> 40) as f32 / (1u64 << 24) as f32;
        (u * 2.0 - 1.0) * a
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.u64() % n as u64) as u32
    }
}

/// Reference-named init (uses `embedding.fusion_proj.weight` [d,2d]; `new` splits
/// it into fusion_l/fusion_r internally).
fn init_weights(cfg: &KronosConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut r = Rng(seed);
    let d = cfg.d_model;
    let mut names: Vec<(String, usize)> = param_list_c(cfg)
        .into_iter()
        .filter(|(n, _)| n != "embedding.fusion_l" && n != "embedding.fusion_r")
        .collect();
    names.push(("embedding.fusion_proj.weight".into(), d * 2 * d));
    let mut w = HashMap::new();
    for (name, numel) in names {
        let is_norm = name.ends_with("norm.weight") || name.ends_with("norm1.weight") || name.ends_with("norm2.weight");
        let data: Vec<f32> = (0..numel)
            .map(|_| if is_norm { 1.0 + r.sym(0.02) } else { r.sym(0.06) })
            .collect();
        w.insert(name, data);
    }
    w
}

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

struct Batch {
    s1: Vec<u32>,
    s2: Vec<u32>,
    cal: [Vec<u32>; 5],
    sampled_s1: Vec<u32>,
    targets: Vec<u32>,
    s2_targets: Vec<u32>,
}
impl Batch {
    fn refs(&self) -> [&[u32]; 5] {
        [&self.cal[0], &self.cal[1], &self.cal[2], &self.cal[3], &self.cal[4]]
    }
    fn set(&self, m: &KronosTrain) {
        m.set_batch(&self.s1, &self.s2, &self.refs(), &self.sampled_s1, &self.targets, &self.s2_targets);
    }
}

fn fixture(seed: u64) -> (KronosTrain, Batch) {
    let cfg = KronosConfig::tiny();
    let t = 8u32;
    let m = KronosTrain::new(cfg.clone(), t, &init_weights(&cfg, seed));
    let mut r = Rng(seed ^ 0xDEAD_BEEF);
    let s1: Vec<u32> = (0..t).map(|_| r.below(cfg.s1_vocab() as u32)).collect();
    let s2: Vec<u32> = (0..t).map(|_| r.below(cfg.s2_vocab() as u32)).collect();
    let cal: [Vec<u32>; 5] = std::array::from_fn(|c| {
        let card = CAL[c].1 as u32;
        (0..t).map(|_| r.below(card)).collect()
    });
    // sampled_s1 is a FIXED detached input (held constant across FD forwards).
    let sampled_s1: Vec<u32> = (0..t).map(|_| r.below(cfg.s1_vocab() as u32)).collect();
    let targets: Vec<u32> = (0..t).map(|_| r.below(cfg.s1_vocab() as u32)).collect();
    let s2_targets: Vec<u32> = (0..t).map(|_| r.below(cfg.s2_vocab() as u32)).collect();
    (m, Batch { s1, s2, cal, sampled_s1, targets, s2_targets })
}

#[test]
fn milestone_c_gradcheck_every_trainable_param() {
    if skip() {
        return;
    }
    let (m, b) = fixture(1234);
    b.set(&m);
    m.zero_grads();
    let _l0 = m.forward();
    m.backward();
    m.poll_wait();

    let eps = 5e-3f32;
    let (atol, rtol) = (4e-3f32, 8e-2f32);
    let n_dirs = 4usize;
    let mut rng = Rng(999);
    let mut checked = 0usize;

    for name in m.param_names() {
        let w0 = m.read_weight(&name);
        let g = m.read_grad(&name);
        let mut best: Option<(f32, f32)> = None;
        for _ in 0..n_dirs {
            let v: Vec<f32> = (0..w0.len()).map(|_| if rng.u64() & 1 == 0 { 1.0 } else { -1.0 }).collect();
            let analytic: f32 = g.iter().zip(&v).map(|(a, b)| a * b).sum();
            let wp: Vec<f32> = w0.iter().zip(&v).map(|(a, b)| a + eps * b).collect();
            m.write_weight(&name, &wp);
            let lp = m.forward();
            let wm: Vec<f32> = w0.iter().zip(&v).map(|(a, b)| a - eps * b).collect();
            m.write_weight(&name, &wm);
            let lm = m.forward();
            m.write_weight(&name, &w0);
            let numeric = (lp - lm) / (2.0 * eps);
            if best.map(|(_, nn)| numeric.abs() > nn.abs()).unwrap_or(true) {
                best = Some((analytic, numeric));
            }
        }
        let (a, n) = best.unwrap();
        let err = (a - n).abs();
        let tol = atol + rtol * a.abs().max(n.abs());
        assert!(err <= tol, "gradcheck FAIL {name}: analytic {a:.5} vs numeric {n:.5} (err {err:.5} > tol {tol:.5})");
        checked += 1;
    }
    eprintln!("gradcheck OK: {checked} trainable params (blocks + dual head + embeddings + dep cross-attn)");
    assert!(checked >= 20);
}

#[test]
fn milestone_c_learns_a_fixed_batch_from_scratch() {
    if skip() {
        return;
    }
    let (m, b) = fixture(42);
    b.set(&m);
    let l0 = m.forward();
    let mut last = l0;
    for step in 1..=400u32 {
        m.zero_grads();
        let _ = m.forward();
        m.backward();
        m.adamw_step(step, 1e-2, 0.0, Some(3.0));
        if step % 100 == 0 {
            last = m.forward();
            eprintln!("step {step}: loss {last:.4}");
        }
    }
    m.poll_wait();
    eprintln!("learning: initial {l0:.4} -> final {last:.4}");
    assert!(last < 0.4 * l0, "loss did not collapse: {l0:.4} -> {last:.4}");
    assert!(last < 0.5, "final loss {last:.4} too high to call it memorized");
}

#[test]
fn milestone_d_lora_gradcheck_and_learns() {
    if skip() {
        return;
    }
    // frozen full-precision base + trainable rank-4 adapters on the block attention.
    let cfg = KronosConfig::tiny();
    let t = 8u32;
    let m = KronosTrain::with_lora(cfg.clone(), t, &init_weights(&cfg, 7), Some(LoraCfg::attn(4, 8.0)));
    let mut r = Rng(7 ^ 0xDEAD_BEEF);
    let s1: Vec<u32> = (0..t).map(|_| r.below(cfg.s1_vocab() as u32)).collect();
    let s2: Vec<u32> = (0..t).map(|_| r.below(cfg.s2_vocab() as u32)).collect();
    let cal: [Vec<u32>; 5] = std::array::from_fn(|c| (0..t).map(|_| r.below(CAL[c].1 as u32)).collect());
    let samp: Vec<u32> = (0..t).map(|_| r.below(cfg.s1_vocab() as u32)).collect();
    let tg1: Vec<u32> = (0..t).map(|_| r.below(cfg.s1_vocab() as u32)).collect();
    let tg2: Vec<u32> = (0..t).map(|_| r.below(cfg.s2_vocab() as u32)).collect();
    let calr: [&[u32]; 5] = [&cal[0], &cal[1], &cal[2], &cal[3], &cal[4]];
    m.set_batch(&s1, &s2, &calr, &samp, &tg1, &tg2);

    // gradcheck the adapters only (base is frozen → param_names are the lora tensors)
    m.zero_grads();
    let _ = m.forward();
    m.backward();
    m.poll_wait();
    let (eps, atol, rtol) = (5e-3f32, 4e-3f32, 8e-2f32);
    let mut rng = Rng(321);
    let names = m.param_names();
    assert!(names.iter().all(|n| n.ends_with(".lora_a") || n.ends_with(".lora_b")), "only adapters should train under LoRA");
    for name in &names {
        let w0 = m.read_weight(name);
        let g = m.read_grad(name);
        let mut best: Option<(f32, f32)> = None;
        for _ in 0..4 {
            let v: Vec<f32> = (0..w0.len()).map(|_| if rng.u64() & 1 == 0 { 1.0 } else { -1.0 }).collect();
            let a: f32 = g.iter().zip(&v).map(|(x, y)| x * y).sum();
            let wp: Vec<f32> = w0.iter().zip(&v).map(|(x, y)| x + eps * y).collect();
            m.write_weight(name, &wp);
            let lp = m.forward();
            let wm: Vec<f32> = w0.iter().zip(&v).map(|(x, y)| x - eps * y).collect();
            m.write_weight(name, &wm);
            let lm = m.forward();
            m.write_weight(name, &w0);
            let num = (lp - lm) / (2.0 * eps);
            if best.map(|(_, nn)| num.abs() > nn.abs()).unwrap_or(true) {
                best = Some((a, num));
            }
        }
        let (a, num) = best.unwrap();
        let err = (a - num).abs();
        let tol = atol + rtol * a.abs().max(num.abs());
        assert!(err <= tol, "LoRA gradcheck FAIL {name}: {a:.5} vs {num:.5} (err {err:.5} > tol {tol:.5})");
    }
    eprintln!("LoRA gradcheck OK: {} adapter tensors", names.len());

    // adapters (B=0 init → starts == base) must be able to reduce the loss.
    let l0 = m.forward();
    for step in 1..=300u32 {
        m.zero_grads();
        let _ = m.forward();
        m.backward();
        m.adamw_step(step, 5e-3, 0.0, Some(3.0));
    }
    let last = m.forward();
    eprintln!("LoRA learning: {l0:.4} -> {last:.4} ({} adapters)", names.len());
    assert!(last < l0, "LoRA adapters did not reduce the loss: {l0:.4} -> {last:.4}");
}
