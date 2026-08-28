// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Learnability tests: the real hybrid GDN/GQA engine must actually
//! *learn* tiny tasks, not just pass the gradient check. Mirrors
//! `glmdsa/tests/convergence.rs`. Gated by `MOE_SKIP_GPU_TESTS` (these need a
//! working backend - CPU JIT or GPU).

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn train_batch(m: &Qwen35, x: &[u32], y: &[u32], steps: u32, lr: f32) {
    m.set_batch(x, y);
    for step in 1..=steps {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, lr, 0.0, Some(1.0), 1.0);
        m.poll_wait();
    }
}

/// Overfitting a single fixed batch (full, all-parameter finetune) must drive
/// the loss well down - the end-to-end check that GDN + GQA + the dense MLP
/// all train together (forward *and* backward wired right).
#[test]
fn qwen35_full_finetune_overfits_fixed_batch() {
    if skip() {
        return;
    }
    let cfg = Qwen35Config::tiny();
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 11);
    let m = Qwen35::new_train_on(Gpu::new(pipelines()), cfg.clone(), 2, t, &init);
    let x: Vec<u32> = (0..2 * t).map(|i| (i * 7) % cfg.vocab).collect();
    let y: Vec<u32> = (0..2 * t).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    m.set_batch(&x, &y);
    let before = m.forward();
    train_batch(&m, &x, &y, 60, 1e-2);
    let after = m.forward();
    assert!(after < before * 0.3, "full finetune did not reduce loss enough: {before} -> {after}");
}

/// Copy every trained parameter out of `m` into a fresh **inference-shaped**
/// (`b == 1`) instance, so the single-sequence decode path
/// (`Qwen35::step`, via `qwen35::sample::generate_kv`) can be run against
/// what training produced. `logits_all`/`step` both assert `b == 1`, and the
/// training instances here are built with `b == 2`, so this transfer - the
/// same `param_names`/`read_weight` pair `Qwen35::save` itself uses - is what
/// makes a behavioural check on a trained model possible without going
/// through a checkpoint file.
fn decode_copy(m: &Qwen35, cfg: &Qwen35Config, t: u32) -> Qwen35 {
    let w: HashMap<String, Vec<f32>> = m.param_names().iter().map(|n| (n.clone(), m.read_weight(n))).collect();
    Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), 1, t, &w)
}

/// A short cyclic sequence is exactly memorisable; the model should reach a
/// low loss floor **and then actually decode the cycle**. Mirrors
/// `glmdsa::glm_memorizes_cyclic_sequence`.
///
/// The loss threshold alone is a weak gate: an average cross-entropy under
/// 0.20 is satisfiable while individual next-token predictions at the
/// higher-loss positions are still wrong. On an exactly-periodic sequence the
/// right answer is unambiguous, so the second assertion below asks the
/// question directly - greedily decode from a prefix and require the
/// continuation to BE the cycle, token for token. That cannot be satisfied by
/// a model that merely got the average down.
#[test]
fn qwen35_memorizes_cyclic_sequence() {
    if skip() {
        return;
    }
    let vocab = 7u32;
    let cfg = Qwen35Config { vocab, ..Qwen35Config::tiny() };
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 3);
    let m = Qwen35::new_train_on(Gpu::new(pipelines()), cfg.clone(), 2, t, &init);
    // Two overlapping windows over the cycle 0,1,2,...,6,0,1,... (predict next).
    let half = (t / 2) as usize;
    let cyc: Vec<u32> = (0..(t as usize + half + 1)).map(|i| (i as u32) % vocab).collect();
    let x: Vec<u32> = [&cyc[0..t as usize], &cyc[half..half + t as usize]].concat();
    let y: Vec<u32> = [&cyc[1..t as usize + 1], &cyc[half + 1..half + t as usize + 1]].concat();
    train_batch(&m, &x, &y, 300, 1e-2);
    let loss = m.forward();
    assert!(loss < 0.20, "cyclic memorization loss too high: {loss}");

    // Greedy decode from the cycle's own first three tokens. The prefix
    // starts at absolute position 0, exactly as the first training window
    // did, so RoPE/M-RoPE positions line up with what was learned.
    let g = decode_copy(&m, &cfg, t);
    let prefix: Vec<u32> = (0..3u32).collect();
    let n_new = 4usize;
    let want: Vec<u32> = (prefix.len() as u32..prefix.len() as u32 + n_new as u32).map(|i| i % vocab).collect();
    let mut rng = Rng::new(0);
    // temperature 0 => greedy argmax; no eos ids, so it always runs to length.
    let got = qwen35::sample::generate_kv(&g, &prefix, n_new, 0.0, 0, 1.0, &[], &mut rng);
    assert_eq!(got, want, "greedy continuation of the memorised cycle is wrong (prefix {prefix:?}, loss {loss})");
}

/// LoRA (adapters only, base frozen) must also drive a fixed batch's loss
/// down - the added `.lora_a`/`.lora_b` machinery doesn't break optimisation.
#[test]
fn qwen35_lora_overfits_fixed_batch() {
    if skip() {
        return;
    }
    let cfg = Qwen35Config { lora: Some(lora_cfg(4, 8.0)), ..Qwen35Config::tiny() };
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 13);
    let m = Qwen35::new_train_on(Gpu::new(pipelines()), cfg.clone(), 2, t, &init);
    let x: Vec<u32> = (0..2 * t).map(|i| (i * 7) % cfg.vocab).collect();
    let y: Vec<u32> = (0..2 * t).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    m.set_batch(&x, &y);
    let before = m.forward();
    train_batch(&m, &x, &y, 250, 3e-2);
    let after = m.forward();
    assert!(after < before * 0.8, "LoRA finetune did not learn: {before} -> {after}");
}
