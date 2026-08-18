// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P0 smoke test for `qwen35::model::Qwen35`'s forward assembly - not a
//! parity test (see `golden_parity.rs` for that): proof the wiring composes
//! at `Qwen35Config::tiny()` (both layer types, a real multi-chunk GDN
//! recurrence) and produces finite, deterministic logits on both the CPU
//! JIT and the default GPU backend.

use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{gdn_chunk_size, Qwen35, PIPELINES};

fn init_weights(cfg: &Qwen35Config, seed: u64) -> std::collections::HashMap<String, Vec<f32>> {
    qwen35::init::init_weights(cfg, seed)
}

fn run_smoke(gpu: Gpu) {
    let cfg = Qwen35Config::tiny();
    let b = 1;
    let t = cfg.block_size;
    let init = init_weights(&cfg, 7);

    let m = Qwen35::new_on(gpu, cfg.clone(), b, t, &init);

    let tokens: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let logits = m.logits_all(&tokens);

    assert_eq!(logits.len(), (t * cfg.vocab) as usize);
    assert!(logits.iter().all(|v| v.is_finite()), "every logit must be finite (no NaN/Inf)");

    let logits2 = m.logits_all(&tokens);
    assert_eq!(logits, logits2, "forward must be deterministic across repeated calls");
}

#[test]
fn tiny_config_chunk_is_smaller_than_t_and_divides_it() {
    let t = Qwen35Config::tiny().block_size;
    let chunk = gdn_chunk_size(t);
    assert_eq!(t % chunk, 0, "chunk must divide t");
    assert!(chunk < t, "chunk must be smaller than t to exercise multiple chunks (got chunk={chunk}, t={t})");
    assert!(t / chunk >= 2, "must have at least 2 chunks");
}

#[test]
fn forward_is_finite_and_deterministic_cpu() {
    run_smoke(Gpu::new_cpu(PIPELINES));
}

#[test]
fn forward_is_finite_and_deterministic_default_backend() {
    run_smoke(Gpu::new(PIPELINES));
}
