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

#[test]
fn the_cooperative_row_norm_matches_the_one_barrier_kernel() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(mut dev) = cooperative_device("rmsnorm_rows") else { return };
    // Lane norm (2048), model width (512), head width (48), a width past the vector loads' reach and an odd one.
    for (rows, d) in [(1usize, 2048usize), (4, 512), (8, 48), (2, 4096), (3, 50)] {
        let (x, w) = (noise(rows * d, 5, 3.0), noise(d, 6, 0.4));
        let (plain, coop) = both(&mut dev, |dev| {
            let g = dev.gpu();
            let (xb, wb, out) = (g.storage_init("x", &x), g.storage_init("w", &w), g.storage((rows * d) as u64));
            let mut s = Vec::new();
            dev.zc_norm(&mut s, &xb, &wb, rows as u32, d as u32, &out);
            g.submit(&[], &s);
            vec![g.read(&out, rows * d)]
        });
        assert_agree(&format!("rmsnorm_rows {rows}x{d}"), &plain, &coop);
    }
}

/// The Kronecker factors, the pre and post scales and the permutation of one rotation.
type KronFactors = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>);

fn kron_factors(fa: usize, fb: usize, seed: u64) -> KronFactors {
    let n = fa * fb;
    let mut perm: Vec<u32> = (0..n as u32).collect();
    let mut rng = Lcg::new(seed); // seed: a shuffled gather permutation compared between two paths
    for i in (1..n).rev() {
        perm.swap(i, (rng.signed().abs() * 7919.0) as usize % (i + 1));
    }
    (noise(fa * fa, seed + 1, 0.4), noise(fb * fb, seed + 2, 0.4), noise(n, seed + 3, 1.0), noise(n, seed + 4, 1.0), perm)
}

#[test]
fn the_cooperative_kronecker_rows_match_the_one_barrier_kernel() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(mut dev) = cooperative_device("kron_rows") else { return };
    for (fa, fb, rows) in [(16usize, 32usize, 1usize), (32, 16, 3), (8, 8, 5), (4, 32, 2)] {
        let n = fa * fb;
        let (a, b, pre, post, perm) = kron_factors(fa, fb, 10);
        let (x, out0) = (noise(rows * n, 11, 1.0), noise(rows * n, 12, 1.0));
        for acc in [false, true] {
            let (plain, coop) = both(&mut dev, |dev| {
                let k = mirage::device::kron::Kron::upload(dev, (fa, fb), &mirage::device::kron::KronParams { a: &a, b: &b, pre: Some(&pre), perm: Some(&perm), post: Some(&post) });
                let g = dev.gpu();
                let (xb, out) = (g.storage_init("x", &x), g.storage_init("out", &out0));
                let mut s = Vec::new();
                dev.kron(&mut s, &k, &xb, rows as u32, &out, acc);
                g.submit(&[], &s);
                vec![g.read(&out, rows * n)]
            });
            assert_agree(&format!("kron_rows {fa}x{fb} rows {rows} acc {acc}"), &plain, &coop);
        }
    }
}

#[test]
fn the_cooperative_kronecker_mlp_stage_matches_the_one_barrier_kernel() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(mut dev) = cooperative_device("kron_hada") else { return };
    for (fa, fb, rows, rank) in [(16usize, 32usize, 1usize, 4usize), (32, 16, 3, 8), (8, 8, 2, 1)] {
        let n = fa * fb;
        let (a, b, pre, post, perm) = kron_factors(fa, fb, 20);
        let (x, lg, hp) = (noise(rows * n, 21, 1.0), noise(rows * rank, 22, 2.0), noise((rank + 2) * n, 23, 0.5));
        let (plain, coop) = both(&mut dev, |dev| {
            let k = mirage::device::kron::Kron::upload(dev, (fa, fb), &mirage::device::kron::KronParams { a: &a, b: &b, pre: Some(&pre), perm: Some(&perm), post: Some(&post) });
            let g = dev.gpu();
            let (xb, lgb, hpb, out) = (g.storage_init("x", &x), g.storage_init("lg", &lg), g.storage_init("hp", &hp), g.storage((rows * n) as u64));
            let mut s = Vec::new();
            dev.kron_hada(&mut s, &k, &xb, rows as u32, (&lgb, &hpb), rank as u32, &out);
            g.submit(&[], &s);
            vec![g.read(&out, rows * n)]
        });
        assert_agree(&format!("kron_hada {fa}x{fb} rows {rows} rank {rank}"), &plain, &coop);
    }
}

#[test]
fn the_single_workgroup_attention_matches_the_split_kernels_and_the_gate() {
    use mirage::device::prims::AttnShape;
    let _g = gpu_core::testgpu::exclusive();
    let Some(mut dev) = cooperative_device("attention_gated") else { return };
    // (batch, heads, kv heads, hd, vd, nk, visible): the self step at position 150 (visible from the position word),
    // the cross step over a padded memory, a batch, and a head width that is not a multiple of 16.
    for (batch, heads, kvh, hd, vd, nk, visible) in [(1usize, 8usize, 2usize, 48usize, 48usize, 620usize, u32::MAX), (1, 8, 8, 48, 48, 700, 700), (3, 8, 2, 48, 48, 330, u32::MAX), (2, 4, 4, 20, 36, 70, 70), (1, 2, 1, 12, 44, 9, 9)] {
        let (q, k, v) = (noise(batch * heads * hd, 31, 1.5), noise(batch * nk * kvh * hd, 32, 1.5), noise(batch * nk * kvh * vd, 33, 1.0));
        // Padded memories: the last quarter of every sequence's keys is masked.
        let mask: Vec<f32> = (0..batch * nk).map(|i| if i % nk < nk * 3 / 4 { 1.0 } else { 0.0 }).collect();
        let stride = heads * vd + 17;
        let gate = noise(batch * stride, 34, 2.0);
        let (plain, coop) = both(&mut dev, |dev| {
            let g = dev.gpu();
            let pos = g.storage(1);
            g.write(&pos, &[150]);
            let bufs = [g.storage_init("q", &q), g.storage_init("k", &k), g.storage_init("v", &v), g.storage_init("m", &mask)];
            let (gb, out) = (g.storage_init("g", &gate), g.storage((batch * heads * vd) as u64));
            let shape = AttnShape { batch: batch as u32, nq: 1, heads: heads as u32, kvh: kvh as u32, hd: hd as u32, vd: vd as u32, nk: nk as u32, visible };
            let mut s = Vec::new();
            dev.attention_gated(&mut s, shape, [&bufs[0], &bufs[1], &bufs[2], &bufs[3], &pos], (&gb, stride as u32, 5), &out);
            g.submit(&[], &s);
            vec![g.read(&out, batch * heads * vd)]
        });
        assert_agree(&format!("attention {batch}x{heads} nk {nk} visible {visible}"), &plain, &coop);
    }
}

#[test]
fn the_cooperative_lane_front_matches_maps_mix_and_copy() {
    use mirage::device::prims::MhcShape;
    let _g = gpu_core::testgpu::exclusive();
    let Some(mut dev) = cooperative_device("mhc_front") else { return };
    for (rows, d, lane_id) in [(1usize, 512usize, 0u32), (3, 512, 3), (2, 256, 1)] {
        let n = 4usize;
        let stride = 2 * n + n * n;
        let (lanes, phi) = (noise(rows * n * d, 41, 1.5), noise(rows * stride, 42, 2.0));
        let (bp, bq, br) = (noise(n, 43, 0.5), noise(n, 44, 0.5), noise(n * n, 45, 0.5));
        let (plain, coop) = both(&mut dev, |dev| {
            let g = dev.gpu();
            let (lb, pb) = (g.storage_init("l", &lanes), g.storage_init("p", &phi));
            let (b0, b1, b2) = (g.storage_init("b0", &bp), g.storage_init("b1", &bq), g.storage_init("b2", &br));
            let (maps, u, y) = (g.storage((rows * stride) as u64), g.storage((rows * d) as u64), g.storage((rows * d) as u64));
            let shape = MhcShape { rows: rows as u32, n: n as u32, lane_id, iters: 20 };
            let mut s = Vec::new();
            dev.mhc_front(&mut s, shape, [0.3, 0.2, 0.5], &lb, &pb, [&b0, &b1, &b2], d as u32, (&maps, &u, &y));
            g.submit(&[], &s);
            vec![g.read(&maps, rows * stride), g.read(&u, rows * d), g.read(&y, rows * d)]
        });
        assert_agree(&format!("mhc_front rows {rows} d {d}"), &plain, &coop);
    }
}

#[test]
fn the_single_dispatch_hadamard_mlp_matches_the_three_rotations() {
    use mirage::decoder::hadamard::HadamardMlp;
    let _g = gpu_core::testgpu::exclusive();
    let Some(mut dev) = cooperative_device("hada_mlp") else { return };
    let Some(dir) = crate::common::released_dir() else { return };
    let ck = mirage::import::load_dir(&dir, &mirage::import::ImportOptions::default()).unwrap();
    let weights = mirage::vision::weights::Weights::new(&ck);
    let dec = mirage::decoder::Decoder::new(&ck.config, &weights);
    let mlp = HadamardMlp::new(&ck.config, &weights, 3);
    let n = ck.config.hadamard_width();
    let resident = mirage::device::hadamard::DeviceHadamard::upload(&dev, &mlp, dec.permutations());
    for rows in [1usize, 4] {
        let (h, y0) = (noise(rows * n, 51, 1.0), noise(rows * n, 52, 1.0));
        let (plain, coop) = both(&mut dev, |dev| {
            let g = dev.gpu();
            let (hb, yb) = (g.storage_init("h", &h), g.storage_init("y", &y0));
            let mut s = Vec::new();
            resident.encode_add(dev, &mut s, &hb, rows as u32, &yb);
            g.submit(&[], &s);
            vec![g.read(&yb, rows * n)]
        });
        assert_agree(&format!("hadamard mlp rows {rows}"), &plain, &coop);
    }
}

#[test]
fn the_cooperative_banned_argmax_matches_add_then_argmax() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(mut dev) = cooperative_device("argmax_banned") else { return };
    for (m, n) in [(1usize, 8299usize), (3, 8299), (2, 100), (1, 2048)] {
        let mut logits = noise(m * n, 61, 5.0);
        // Ties: the same maximum at two positions of row 0, the lower index must win.
        logits[7] = 9.0;
        logits[n - 3] = 9.0;
        let mut ban = vec![0f32; m * n];
        ban[n - 3] = -1.0e30;
        ban[13] = -1.0e30;
        let (plain, coop) = both(&mut dev, |dev| {
            let g = dev.gpu();
            let (lb, bb, out) = (g.storage_init("l", &logits), g.storage_init("b", &ban), g.storage(m as u64));
            let mut s = Vec::new();
            dev.argmax_banned(&mut s, &lb, &bb, &out, m as u32, n as u32);
            g.submit(&[], &s);
            vec![g.read(&out, m)]
        });
        assert_eq!(plain, coop, "argmax_banned {m}x{n}");
    }
}
