// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The cooperative decoder kernels (any number of workgroup barriers, registers instead of serial folds) against the
//! one-barrier originals they replace on an accelerator: same inputs, same operator, results equal up to the order of
//! the reductions. On the CPU backend the originals are the only kernels and the comparison is skipped.
//!
//! Swedish Embedded AB implements solutions for latency-bound GPU inference for its clients. If your team needs
//! expertise in making batch-1 decode run at the speed of the card then you can procure our services by sending an
//! email to info@swedishembedded.com.

use data::rng::Lcg;
use mirage::device::Device;

fn noise(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Lcg::new(seed); // seed: filler compared between two paths
    (0..n).map(|_| rng.signed() * scale).collect()
}

/// The device, or `None` when there is none or it runs only the one-barrier kernels.
fn cooperative_device(test: &str) -> Option<Device> {
    crate::common::cap_threads();
    match Device::open() {
        Ok(d) if d.gpu().caps().workgroup_reductions => Some(d),
        Ok(_) => {
            brain_testutil::skip_unavailable(&format!("{test}: the device has no multi-barrier workgroups"));
            None
        }
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("{test}: no device: {e}"));
            None
        }
    }
}

/// Run `build` with the cooperative kernels off and on; each returns the buffers to compare.
fn both(dev: &mut Device, build: impl Fn(&Device) -> Vec<Vec<f32>>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    dev.set_cooperative(false);
    let plain = build(dev);
    dev.set_cooperative(true);
    (plain, build(dev))
}

/// Largest absolute difference relative to the largest magnitude of the reference.
fn assert_agree(what: &str, plain: &[Vec<f32>], coop: &[Vec<f32>]) {
    for (i, (a, b)) in plain.iter().zip(coop).enumerate() {
        let peak = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(worst <= 2e-5 * peak, "{what}: output {i} differs by {worst:e} against a peak of {peak:e}");
    }
}

#[test]
fn the_cooperative_residual_norm_matches_the_one_barrier_kernel() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(mut dev) = cooperative_device("rms_axpy_norm") else { return };
    for (rows, d) in [(1usize, 512usize), (5, 512), (3, 256), (2, 48)] {
        let (x, w1, w2, y0) = (noise(rows * d, 1, 2.0), noise(d, 2, 0.3), noise(d, 3, 0.3), noise(rows * d, 4, 1.5));
        let (plain, coop) = both(&mut dev, |dev| {
            let g = dev.gpu();
            let (xb, w1b, w2b, yb) = (g.storage_init("x", &x), g.storage_init("w1", &w1), g.storage_init("w2", &w2), g.storage_init("y", &y0));
            let mut s = Vec::new();
            let h = dev.zc_norm_axpy_norm(&mut s, 0.7, &xb, &w1b, &w2b, rows as u32, d as u32, &yb);
            g.submit(&[], &s);
            vec![g.read(&yb, rows * d), g.read(&h, rows * d)]
        });
        assert_agree(&format!("rms_axpy_norm {rows}x{d}"), &plain, &coop);
    }
}
