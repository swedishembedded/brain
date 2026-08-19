// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P0 smoke test for `qwen35moe::model::Qwen35`'s forward assembly: not a
//! parity test (no `torch`/`transformers` in this environment — see
//! `model.rs`'s module doc for the honest scope note), but proof the wiring
//! composes: `Qwen35Config::tiny()` exercises both layer types (Gated
//! DeltaNet at layers 0-2/4-6, GQA at layers 3/7 — `full_attention_interval
//! =4`, `n_layers=8`) and a small-but-nontrivial MoE (`n_experts=6,
//! top_k=2`), a real (>1) GDN chunk count (`tiny()`'s `block_size=24` and
//! `gdn_chunk_size(24)=8` -> 3 chunks, exercising the cross-chunk recurrence
//! loop, not just a single degenerate chunk), and produces finite,
//! deterministic logits on both the CPU JIT and the default GPU backend
//! (a barrier-crossing kernel can silently misbehave
//! on exactly one backend).

use gpu_core::Gpu;
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::{gdn_chunk_size, Qwen35, pipelines};

fn init_weights(cfg: &Qwen35Config, seed: u64) -> std::collections::HashMap<String, Vec<f32>> {
    qwen35moe::init::init_weights(cfg, seed)
}

#[test]
fn tiny_config_chunk_is_smaller_than_t_and_divides_it() {
    // Sanity check on the chunk-size heuristic itself, independent of any
    // device: tiny()'s block_size=24 must land on a REAL multi-chunk split
    // (chunk < t), not silently collapse to one giant chunk covering the
    // whole sequence (which would never exercise gdn_chunk_fwd's cross-chunk
    // state-carry loop at all).
    let t = Qwen35Config::tiny().block_size;
    let chunk = gdn_chunk_size(t);
    assert_eq!(t % chunk, 0, "chunk must divide t");
    assert!(chunk < t, "chunk must be smaller than t to exercise multiple chunks (got chunk={chunk}, t={t})");
    assert!(t / chunk >= 2, "must have at least 2 chunks");
}

/// Runs one forward pass at `Qwen35Config::tiny()` and asserts every logit is
/// finite. Shared by the CPU and GPU variants below.
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

    // Determinism: a second call at the same weights/tokens must reproduce
    // the exact same logits (no uninitialised-scratch nondeterminism from the
    // fresh-per-layer-call buffer allocation this model uses).
    let logits2 = m.logits_all(&tokens);
    assert_eq!(logits, logits2, "forward must be deterministic across repeated calls");
}

#[test]
fn forward_is_finite_and_deterministic_cpu() {
    let gpu = Gpu::new_cpu(pipelines());
    run_smoke(gpu);
}

/// `Gpu::new` honours `BRAIN_DEVICE` when set and defaults to the wgpu
/// backend otherwise (see `gpu_core`'s own doc) -- run this test under both
/// `BRAIN_DEVICE=cpu` and unset (the default GPU backend) to cover both,
/// since a barrier-crossing kernel can silently misbehave on exactly one
/// backend.
#[test]
fn forward_is_finite_and_deterministic_default_backend() {
    let gpu = Gpu::new(pipelines());
    run_smoke(gpu);
}

/// Exercises `Model::init_weights`'s determinism directly (same seed ->
/// bit-identical init), independent of any device.
#[test]
fn init_weights_deterministic_for_fixed_seed() {
    let cfg = Qwen35Config::tiny();
    let a = init_weights(&cfg, 11);
    let b = init_weights(&cfg, 11);
    for (name, va) in &a {
        assert_eq!(va, &b[name], "weight {name} must be identical across two calls at the same seed");
    }
}

/// Every A_log value must be finite (log of a positive number in (0,16]) --
/// the one init value with a numerically interesting failure mode (log(0) =
/// -inf) if the floor in `init.rs` were ever dropped.
#[test]
fn a_log_init_is_finite() {
    let cfg = Qwen35Config::tiny();
    let w = init_weights(&cfg, 3);
    let mut checked = 0;
    for (name, v) in &w {
        if name.ends_with(".A_log") {
            assert!(v.iter().all(|x| x.is_finite()), "{name} must be finite");
            checked += 1;
        }
    }
    assert!(checked > 0, "tiny() must have at least one linear-attention layer with an A_log tensor");
}
