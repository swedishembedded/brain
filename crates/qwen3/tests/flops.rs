// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLOP/OPS accounting through the dispatch seam: the OFFLINE number (walking
//! the recorded step lists, no execution) must agree EXACTLY with the ONLINE
//! counters accumulated at `Gpu::submit`, coverage must be total, and the int8
//! path must report integer OPS where the fp32 path reports FLOPs. CPU backend
//! (deterministic; runs on CI without a GPU).

use gpu_core::{set_default_backend, Backend};
use qwen3::{init_weights, Qwen, QwenConfig, Shard};

#[test]
fn offline_matches_online_and_covers_everything() {
    set_default_backend(Backend::Cpu);
    let cfg = QwenConfig::tiny();
    let (b, t) = (1u32, cfg.block_size);
    let init = init_weights(&cfg, 3);
    let m = Qwen::new(cfg.clone(), b, t, &init);

    let off_f = m.cost_fwd();
    let off_b = m.cost_bwd();
    assert!(off_f.steps > 0 && off_b.steps > 0);
    assert_eq!(off_f.covered, off_f.steps, "forward uncovered: {:?}", off_f.uncovered);
    assert_eq!(off_b.covered, off_b.steps, "backward uncovered: {:?}", off_b.uncovered);
    assert!(off_f.total.flops > 0);
    assert_eq!(off_f.total.int_ops, 0, "fp32 model must report zero integer OPS");

    let x: Vec<u32> = (0..(b * t) as usize).map(|i| i as u32 % cfg.vocab).collect();
    m.set_batch(&x, &x);
    m.gpu().reset_ops_counters();
    m.forward();
    let online = m.gpu().ops_counters();
    assert_eq!(online.steps, off_f.steps);
    assert_eq!(online.total, off_f.total, "online forward != offline forward");

    m.backward();
    let mut expect = off_f.clone();
    expect.merge(&off_b);
    let online = m.gpu().ops_counters();
    assert_eq!(online.steps, expect.steps);
    assert_eq!(online.total, expect.total, "online fwd+bwd != offline fwd+bwd");
}

/// The quantized model's linears must show up as `int_ops` (what actually
/// runs: DP4A int8 MACs), NOT as fp32 flops — offline, without executing.
#[test]
fn i8_model_reports_int_ops() {
    set_default_backend(Backend::Cpu);
    let cfg = QwenConfig::tiny();
    let init = init_weights(&cfg, 3);
    let m8 = Qwen::new_shard_i8(cfg.clone(), 1, cfg.block_size, &init, Shard::whole(cfg.n_layers as usize));

    let off = m8.cost_fwd();
    assert_eq!(off.covered, off.steps, "i8 forward uncovered: {:?}", off.uncovered);
    assert!(off.total.int_ops > 0, "int8 linears must count integer OPS");
    assert!(
        off.by_kernel.keys().any(|k| k.starts_with("matmul_i8")),
        "expected a matmul_i8 kernel in the i8 forward, got: {:?}",
        off.by_kernel.keys().collect::<Vec<_>>()
    );
    // The int8 MAC volume equals the fp32 linears it replaced (same shapes,
    // 2·m·K·n each): it must not exceed the fp32 build's GEMM flops (which
    // additionally include the lm_head).
    let t = cfg.block_size;
    let mfp = Qwen::new(cfg, 1, t, &init);
    let fp = mfp.cost_fwd();
    let fp_linear_flops: u64 = fp
        .by_kernel
        .iter()
        .filter(|(k, _)| k.as_str() == "matmul" || k.as_str() == "matmul_reg3")
        .map(|(_, v)| v.cost.flops)
        .sum();
    assert!(fp_linear_flops >= off.total.int_ops, "i8 int_ops exceed the fp32 linear volume");
}
