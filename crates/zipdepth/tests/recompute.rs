// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of vision models for its clients. If your team needs
// expertise in training dense-prediction networks within a fixed device budget, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Activation recompute for the depth network's normed conv units, held to the contract
//! `model::saved::check_recompute` states: a network that keeps only each unit's output and rebuilds its post-conv and
//! post-BatchNorm intermediates from the unit's input before its backward computes the loss and every gradient of the
//! network that keeps them, bit for bit, at a lower step peak. ZipDepth is a net run by a free training loop with no
//! `model::Model` impl for that function to take, so this test runs the same two clauses on the net itself. One test
//! in its own binary, so nothing else allocates while the device bytes are compared.

use gpu_core::Gpu;
use model::saved::SavedPolicy;
use paramstore::ParamStore;
use vision::Ctx;
use zipdepth::{ZipConfig, ZipDepth};

const BATCH: u32 = 2;

/// A step's output sum, every parameter's gradient bits and the live device bytes it peaked at.
type Measured = (f32, Vec<(String, Vec<u32>)>, Option<u64>);

fn config() -> ZipConfig {
    ZipConfig { dims: [8, 16, 32, 64], depths: [2, 2, 2, 2], dec_ch: 16, half_dec_ch: 8, input: 32, ..ZipConfig::base() }
}

/// One training step of the net under `policy`.
fn measured(policy: SavedPolicy) -> Measured {
    let cfg = config();
    let gpu: Gpu = gpu_core::testgpu::dev(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, zipdepth::net::ids());
    let m = ZipDepth::build(&ctx, cfg.clone(), BATCH, true);
    m.set_saved_policy(policy).unwrap();
    let init = zipdepth::init_weights(&cfg, 21);
    let ps = ParamStore::new(&gpu, m.param_list(), &init);
    let x: Vec<f32> = (0..m.in_shape.numel()).map(|i| ((i * 37 % 101) as f32) / 100.0).collect();
    let d: Vec<f32> = (0..m.out_shape.numel()).map(|i| ((i * 13 % 17) as f32 - 8.0) / 64.0).collect();
    let (xb, db) = (gpu.storage_init("x", &x), gpu.storage_init("d", &d));
    gpu.reset_memory_peak();
    ps.zero_grads(&gpu);
    m.forward(&ctx, &ps, &xb);
    m.backward(&ctx, &ps, &xb, &db);
    gpu.poll_wait();
    let peak = gpu.memory_usage().map(|u| u.peak_live);
    let out: f32 = gpu.read(m.out(), m.out_shape.numel() as usize).iter().sum();
    let mut names: Vec<String> = m.param_list().into_iter().map(|(n, _)| n).collect();
    names.sort();
    let grads = names.into_iter().filter(|n| ps.grad.contains_key(n)).map(|n| (n.clone(), ps.read_grad(&gpu, &n).into_iter().map(f32::to_bits).collect())).collect();
    (out, grads, peak)
}

#[test]
fn zipdepth_recomputes_its_units_exactly() {
    let (want, want_grads, keep_peak) = measured(SavedPolicy::Keep);
    let (got, got_grads, recompute_peak) = measured(SavedPolicy::Recompute);
    if let (Some(keep), Some(recompute)) = (keep_peak, recompute_peak) {
        assert!(recompute < keep, "the recomputing step peaks at {recompute} live device bytes, keeping every unit's intermediates {keep}");
    }
    assert_eq!(got.to_bits(), want.to_bits(), "recomputing gives output {got}, keeping {want}");
    assert!(want_grads.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient to compare");
    assert_eq!(want_grads.len(), got_grads.len());
    for ((n, a), (_, b)) in want_grads.iter().zip(&got_grads) {
        assert!(a == b, "{n}: gradient differs recomputing");
    }

    // Narrow saved formats do not apply, and a recomputing net does not go back to keeping.
    let ctx_gpu = gpu_core::testgpu::dev(zipdepth::net::PIPELINES);
    let ctx = Ctx::new(&ctx_gpu, zipdepth::net::ids());
    let m = ZipDepth::build(&ctx, config(), BATCH, true);
    assert!(m.set_saved_policy(SavedPolicy::Quantized(model::saved::SavedFormat::Bf16)).unwrap_err().contains("zipdepth"));
    m.set_saved_policy(SavedPolicy::Recompute).unwrap();
    assert!(m.set_saved_policy(SavedPolicy::Keep).unwrap_err().contains("zipdepth"));
}
