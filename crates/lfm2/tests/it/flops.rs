// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLOP/OPS accounting through the dispatch seam, LFM encoder edition: the
//! OFFLINE number (walking the recorded step lists, no execution) must agree
//! EXACTLY with the ONLINE counters accumulated at `Gpu::submit`, and coverage
//! must be total — this model exercises the conv mixer + bidirectional
//! attention formulas the decoder LMs don't. CPU backend (deterministic).

use gpu_core::{set_default_backend, Backend};
use lfm2::config::LfmConfig;
use lfm2::init::init_weights;
use lfm2::model::Lfm;

#[test]
fn offline_matches_online_and_covers_everything() {
    set_default_backend(Backend::Cpu);
    let cfg = LfmConfig::tiny();
    let (b, t) = (1u32, cfg.block_size);
    let init = init_weights(&cfg, 5);
    let m = Lfm::new_train(cfg.clone(), b, t, &init);

    let off_f = m.cost_fwd();
    let off_b = m.cost_bwd();
    assert!(off_f.steps > 0 && off_b.steps > 0);
    assert_eq!(off_f.covered, off_f.steps, "forward uncovered: {:?}", off_f.uncovered);
    assert_eq!(off_b.covered, off_b.steps, "backward uncovered: {:?}", off_b.uncovered);
    assert!(off_f.total.flops > 0);
    assert_eq!(off_f.total.int_ops, 0, "fp32 model must report zero integer OPS");
    // The conv mixer must be visible in the breakdown.
    assert!(off_f.by_kernel.contains_key("conv1d"), "got: {:?}", off_f.by_kernel.keys().collect::<Vec<_>>());

    let x: Vec<u32> = (0..(b * t) as usize).map(|i| i as u32 % cfg.vocab).collect();
    m.set_batch(&x, &x);
    m.gpu.reset_ops_counters();
    m.forward();
    let online = m.gpu.ops_counters();
    assert_eq!(online.steps, off_f.steps);
    assert_eq!(online.total, off_f.total, "online forward != offline forward");

    m.backward();
    let mut expect = off_f.clone();
    expect.merge(&off_b);
    let online = m.gpu.ops_counters();
    assert_eq!(online.steps, expect.steps);
    assert_eq!(online.total, expect.total, "online fwd+bwd != offline fwd+bwd");
}
