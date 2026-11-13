// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a rotated-codebook (`CQ2`/`CQ3`/`CQ4`) weight is multiplied in place from its stored indices and norms - the
//! product equals the host product of the dequantised weight to f32 rounding for every row count, every index width and a
//! width that is not a whole number of groups; the tile decode equals the host dequantiser; each product kernel agrees
//! on its own (the cooperative GEMV where the device has workgroup barriers, the row-blocked kernel everywhere).

use data::rng::Lcg;
use gpu_core::{Dispatch, Gpu};
use model::cq::{a8_quantize_group, padded, Cq, CqActivations, GROUP};
use model::ops::{cq_kernels, training_kernel_list, Ops, Weight};

const BOOKS: [&[f32]; 3] = [&[-1.51, -0.45, 0.45, 1.51], &[-1.75, -1.05, -0.5, -0.16, 0.16, 0.5, 1.05, 1.75], &[-2.4, -1.84, -1.44, -1.1, -0.8, -0.52, -0.25, 0.0, 0.25, 0.52, 0.8, 1.1, 1.44, 1.84, 2.4, 2.9]];

fn ops() -> Ops {
    let list: Vec<(&str, &str)> = training_kernel_list().iter().chain(cq_kernels()).copied().collect();
    Ops::new(Gpu::new(Box::leak(list.into_boxed_slice()))).expect("Ops::new")
}

fn book(bits: u32) -> &'static [f32] {
    BOOKS[bits as usize - 2]
}

/// `y[i, r] = sum_c x[i, c] w[r, c]` in f64 with the root of the summed squared terms beside it.
fn host_product(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<(f64, f64)> {
    let mut y = Vec::with_capacity(m * n);
    for i in 0..m {
        for r in 0..n {
            let terms: Vec<f64> = (0..k).map(|c| f64::from(x[i * k + c]) * f64::from(w[r * k + c])).collect();
            y.push((terms.iter().sum(), terms.iter().map(|t| t * t).sum::<f64>().sqrt()));
        }
    }
    y
}

fn assert_close(label: &str, got: &[f32], want: &[(f64, f64)]) {
    assert_within(label, got, want, 2e-5);
}

fn assert_within(label: &str, got: &[f32], want: &[(f64, f64)], bound: f64) {
    let worst = got.iter().zip(want).map(|(&g, &(w, scale))| (f64::from(g) - w).abs() / (1e-6 + scale)).fold(0f64, f64::max);
    // f32 summation of up to a few hundred terms: the error is a small multiple of the unit roundoff of the term scale.
    eprintln!("{label}: worst error {worst:e} of the term scale");
    assert!(worst < bound, "{label}: worst error {worst:e} of the term scale (bound {bound:e})");
}

#[test]
fn products_equal_the_dequantised_matmul_for_every_width_and_row_count() {
    let ops = ops();
    let g = ops.gpu();
    for bits in [2u32, 3, 4] {
        // 200 is not a whole number of groups: the rotation pads the activation, the weight pads with quantised zeros.
        let (n, k) = (37usize, 200usize);
        let mut rng = Lcg::new(0xC0 + u64::from(bits));
        let cq = Cq::quantize(&rng.vec_scaled(n * k, 1.0), n, k, bits, book(bits)).expect("quantises");
        let wd = cq.dequantize();
        let weight = Weight::cq(&ops, "cq", &cq);
        assert!(!weight.is_dense_f32() && weight.reads_f32_activations());
        assert_eq!(weight.resident_bytes() as usize, cq.bytes(), "{bits}-bit resident bytes");
        for m in [1usize, 2, 7, 8, 9, 33, 100] {
            let x = rng.vec_scaled(m * k, 1.0);
            let xb = g.storage_init("x", &x);
            let y = g.storage((m * n) as u64);
            let mut steps = Vec::new();
            let act = ops.act_f32(&xb, 0, m as u32, k as u32);
            ops.matmul(&mut steps, &weight, &act, &y, 0);
            g.submit(&[], &steps);
            // m = 100 is the dense-GEMM-over-the-rotated-weight path on a device with barriers; it holds the same bound.
            assert_close(&format!("cq{bits} m={m} Ops::matmul"), &g.read(&y, m * n), &host_product(&x, m, k, &wd, n));
        }
    }
}

#[test]
fn the_tile_decode_equals_the_host_dequantiser() {
    let ops = ops();
    let g = ops.gpu();
    for bits in [2u32, 3, 4] {
        let (n, k) = (9usize, 200usize);
        let mut rng = Lcg::new(0xD0 + u64::from(bits));
        let cq = Cq::quantize(&rng.vec_scaled(n * k, 1.0), n, k, bits, book(bits)).expect("quantises");
        let weight = Weight::cq(&ops, "cq", &cq);
        let out = g.storage((n * k) as u64);
        let mut steps = Vec::new();
        ops.decode_tile(&mut steps, &weight, 0, n as u32, &out);
        g.submit(&[], &steps);
        let got = g.read(&out, n * k);
        let want = cq.dequantize();
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "cq{bits}: tile decode differs from the host dequantiser by {worst:e}");
    }
}

#[test]
fn each_product_kernel_agrees_with_the_host_on_its_own() {
    let ops = ops();
    let g = ops.gpu();
    let barriers = g.caps().workgroup_reductions;
    eprintln!("cq kernels on {} (workgroup barriers: {barriers})", g.kind());
    for bits in [2u32, 3, 4] {
        let (n, k) = (21usize, 256usize);
        let mut rng = Lcg::new(0xE0 + u64::from(bits));
        let cq = Cq::quantize(&rng.vec_scaled(n * k, 1.0), n, k, bits, book(bits)).expect("quantises");
        let wd = cq.dequantize();
        let weight = Weight::cq(&ops, "cq", &cq);
        let Weight::Cq { w, s, book: levels, .. } = &weight else { unreachable!() };
        for m in [1usize, 3, 8, 9, 33] {
            let x = rng.vec_scaled(m * k, 1.0);
            let xr = g.storage_init("xr", &Cq::rotate(&x, m, k));
            let want = host_product(&x, m, k, &wd, n);
            let params = [m as u32, padded(k) as u32, n as u32, 0];
            let mut variants = vec![(kernels::template::cq_gemm_kernel_name(bits), Dispatch::Threads(n as u32 * (m as u32).div_ceil(kernels::template::CQ_GEMM_ROWS)))];
            if barriers {
                variants.push((kernels::template::cq_gemv_kernel_name(bits, m as u32), Dispatch::Workgroups((n as u32).div_ceil(kernels::template::CQ_GEMV_COLS) * (m as u32).div_ceil(8))));
            }
            for (name, grid) in variants {
                let y = g.storage((m * n) as u64);
                let index = g.kernel_index(&name).unwrap_or_else(|| panic!("{name} is registered by cq_kernels"));
                g.submit(&[], &[g.dispatch(index, &[&xr, w, s, levels, &y], &params, grid)]);
                assert_close(&format!("{name} m={m}"), &g.read(&y, m * n), &want);
            }
        }
    }
    assert_eq!(GROUP, 128, "the kernels hard-code the 128-lane group");
}

#[test]
fn rotation_kernels_match_the_host_transform() {
    let ops = ops();
    let g = ops.gpu();
    let (m, k) = (5usize, 200usize);
    let x = Lcg::new(77).vec_scaled(m * k, 1.0);
    let xb = g.storage_init("x", &x);
    let want = Cq::rotate(&x, m, k);
    let params = [m as u32, k as u32, padded(k) as u32, 0];
    let groups = (m * padded(k) / GROUP) as u32;
    let mut variants = vec![("cq_rotate", Dispatch::Threads(groups))];
    if g.caps().workgroup_reductions {
        variants.push(("cq_rotate_wg", Dispatch::Workgroups(groups)));
    }
    for (name, grid) in variants {
        let out = g.storage((m * padded(k)) as u64);
        g.submit(&[], &[g.dispatch(g.kernel_index(name).expect("registered"), &[&xb, &out], &params, grid)]);
        let worst = g.read(&out, m * padded(k)).iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "{name} differs from the host rotation by {worst:e}");
    }
}

/// The int8 lanes (four to a word, lane 0 in the low byte) and group scales the A8 rotation kernels write.
fn read_a8_rows(g: &Gpu, q: &gpu_core::DeviceBuffer, scales: &gpu_core::DeviceBuffer, m: usize, kpad: usize) -> (Vec<i8>, Vec<f32>) {
    let lanes = g.read(q, m * kpad / 4).iter().flat_map(|w| w.to_bits().to_le_bytes()).map(|b| b as i8).collect();
    (lanes, g.read(scales, m * kpad / GROUP))
}

#[test]
fn the_int8_rotation_kernels_equal_the_host_quantiser_bit_for_bit() {
    let ops = ops();
    let g = ops.gpu();
    let (m, k) = (5usize, 200usize);
    let kpad = padded(k);
    // One row of exact zeros (scale 1) and one with a single outlier (every other lane rounds to 0 or +-1).
    let mut x = Lcg::new(78).vec_scaled(m * k, 1.0);
    x[..k].fill(0.0);
    x[k + 3] = 400.0;
    let xb = g.storage_init("x", &x);
    let rotated = Cq::rotate(&x, m, k);
    let want: Vec<(Vec<i8>, f32)> = rotated.as_chunks::<GROUP>().0.iter().map(|c| a8_quantize_group(c.as_slice())).collect();
    let params = [m as u32, k as u32, kpad as u32, 0];
    let groups = (m * kpad / GROUP) as u32;
    let mut variants = vec![("cq_rotate_a8", Dispatch::Threads(groups))];
    if g.caps().workgroup_reductions {
        variants.push(("cq_rotate_a8_wg", Dispatch::Workgroups(groups)));
    }
    for (name, grid) in variants {
        let (q, s) = (g.storage((m * kpad / 4) as u64), g.storage(groups as u64));
        g.submit(&[], &[g.dispatch(g.kernel_index(name).expect("registered"), &[&xb, &q, &s], &params, grid)]);
        let (lanes, scales) = read_a8_rows(g, &q, &s, m, kpad);
        for (i, (bytes, scale)) in want.iter().enumerate() {
            assert_eq!(&lanes[i * GROUP..(i + 1) * GROUP], &bytes[..], "{name}: int8 lanes of group {i}");
            assert_eq!(scales[i].to_bits(), scale.to_bits(), "{name}: scale of group {i}");
        }
    }
}

#[test]
fn the_int8_product_kernels_equal_the_host_integer_emulation() {
    let ops = ops();
    let g = ops.gpu();
    let barriers = g.caps().workgroup_reductions;
    eprintln!("cq a8 kernels on {} (workgroup barriers: {barriers})", g.kind());
    // 700 is six groups and a partial one (two rounds of the cooperative kernel's four group slots).
    for (bits, k) in [(2u32, 256usize), (3, 256), (4, 256), (2, 700), (4, 700)] {
        let n = 21usize;
        let mut rng = Lcg::new(0xF0 + u64::from(bits) + k as u64);
        let cq = Cq::quantize(&rng.vec_scaled(n * k, 1.0), n, k, bits, book(bits)).expect("quantises");
        let weight = Weight::cq(&ops, "cq", &cq);
        let Weight::Cq { w, s, book: levels, .. } = &weight else { unreachable!() };
        for m in [1usize, 3, 8, 9, 33] {
            let x = rng.vec_scaled(m * k, 1.0);
            let want = cq.matmul_a8(&x, m);
            let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let (kpad, groups) = (padded(k), padded(k) / GROUP);
            let rows: Vec<(Vec<i8>, f32)> = Cq::rotate(&x, m, k).as_chunks::<GROUP>().0.iter().map(|c| a8_quantize_group(c.as_slice())).collect();
            let packed: Vec<f32> = rows.iter().flat_map(|(q, _)| q.as_chunks::<4>().0.iter().map(|c| f32::from_bits(u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])))).collect();
            let (q, a) = (g.storage_init("q", &packed), g.storage_init("a", &rows.iter().map(|r| r.1).collect::<Vec<_>>()));
            assert_eq!(packed.len(), m * kpad / 4);
            assert_eq!(rows.len(), m * groups);
            let params = [m as u32, kpad as u32, n as u32, 0];
            let mut variants = vec![(kernels::template::cq_a8_gemm_kernel_name(bits), Dispatch::Threads(n as u32 * (m as u32).div_ceil(kernels::template::CQ_GEMM_ROWS)), true)];
            if barriers {
                variants.push((kernels::template::cq_a8_gemv_kernel_name(bits, m as u32), Dispatch::Workgroups((n as u32).div_ceil(kernels::template::CQ_A8_GEMV_COLS) * (m as u32).div_ceil(8)), false));
            }
            for (name, grid, same_order) in variants {
                let y = g.storage((m * n) as u64);
                let index = g.kernel_index(&name).unwrap_or_else(|| panic!("{name} is registered by cq_kernels"));
                g.submit(&[], &[g.dispatch(index, &[&q, &a, w, s, levels, &y], &params, grid)]);
                let got = g.read(&y, m * n);
                let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale;
                eprintln!("{name} k={k} m={m}: worst difference from the host emulation {worst:e} of the largest output");
                // The integer parts are exact and the per-group float terms identical. The barrier-free kernel adds them in the
                // host's order (bit for bit on the CPU backend; a GPU compiler may contract a multiply-add and move the last
                // bit); the cooperative kernel adds a row of more than four groups in four interleaved partial sums.
                let bound = if same_order && g.kind() == "cpu" { 0.0 } else { 4e-6 };
                assert!(worst <= bound, "{name} k={k} m={m}: {worst:e} of the largest output (bound {bound:e})");
            }
        }
    }
}

#[test]
fn the_int8_mode_is_opt_in_and_runs_through_ops_matmul() {
    let mut ops = ops();
    assert_eq!(ops.cq_activations(), CqActivations::F32, "f32 activations unless asked");
    let (n, k, bits) = (37usize, 200usize, 2u32);
    let mut rng = Lcg::new(0x1A8);
    let cq = Cq::quantize(&rng.vec_scaled(n * k, 1.0), n, k, bits, book(bits)).expect("quantises");
    let weight = Weight::cq(&ops, "cq", &cq);
    let run = |ops: &Ops, x: &[f32], m: usize| {
        let g = ops.gpu();
        let (xb, y) = (g.storage_init("x", x), g.storage((m * n) as u64));
        let mut steps = Vec::new();
        ops.matmul(&mut steps, &weight, &ops.act_f32(&xb, 0, m as u32, k as u32), &y, 0);
        g.submit(&[], &steps);
        g.read(&y, m * n)
    };
    for m in [1usize, 4, 9, 100] {
        let x = rng.vec_scaled(m * k, 1.0);
        let f32_mode = run(&ops, &x, m);
        ops.set_cq_activations(CqActivations::A8);
        let a8 = run(&ops, &x, m);
        ops.set_cq_activations(CqActivations::F32);
        assert_eq!(run(&ops, &x, m), f32_mode, "switching back restores the f32 product");
        let want = cq.matmul_a8(&x, m);
        let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let worst = a8.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale;
        assert!(worst <= 4e-6, "m={m}: Ops::matmul in A8 mode differs from the host emulation by {worst:e}");
        assert_ne!(a8, f32_mode, "m={m}: the A8 product is a different (coarser) computation");
    }
}

/// Crossover between the cooperative product and the row-blocked one over the row count, on the shapes of Mirage's
/// decoder projections. Run by hand on an accelerator: `cargo test ... -- --ignored cq_crossover`.
#[test]
#[ignore = "benchmark: run by hand on a GPU"]
fn cq_crossover() {
    let ops = ops();
    let g = ops.gpu();
    if !g.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("cq crossover: needs workgroup barriers (a GPU)");
        return;
    }
    let (n, k, bits) = (2048usize, 512usize, 2u32);
    let mut rng = Lcg::new(31);
    let cq = Cq::quantize(&rng.vec_scaled(n * k, 1.0), n, k, bits, book(bits)).expect("quantises");
    let weight = Weight::cq(&ops, "cq", &cq);
    let Weight::Cq { w, s, book: levels, .. } = &weight else { unreachable!() };
    eprintln!("m | cooperative us | row-blocked us   (n={n} k={k} cq{bits}, kernel time per product over 50 replays)");
    for m in [1usize, 4, 8, 16, 32, 64, 128, 256, 512] {
        let xr = g.storage_init("xr", &Cq::rotate(&Lcg::new(3).vec_scaled(m * k, 1.0), m, k));
        let y = g.storage((m * n) as u64);
        let params = [m as u32, k as u32, n as u32, 0];
        let time = |name: String, grid: Dispatch| -> f64 {
            let index = g.kernel_index(&name).expect("registered");
            let steps: Vec<_> = (0..50).map(|_| g.dispatch(index, &[&xr, w, s, levels, &y], &params, grid)).collect();
            g.submit(&[], &steps);
            let _ = g.read(&y, 1);
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                g.submit(&[], &steps);
                let _ = g.read(&y, 1);
                best = best.min(t.elapsed().as_secs_f64() / 50.0);
            }
            best * 1e6
        };
        let coop = time(kernels::template::cq_gemv_kernel_name(bits, m as u32), Dispatch::Workgroups((n as u32).div_ceil(kernels::template::CQ_GEMV_COLS) * (m as u32).div_ceil(8)));
        let blocked = time(kernels::template::cq_gemm_kernel_name(bits), Dispatch::Threads(n as u32 * (m as u32).div_ceil(kernels::template::CQ_GEMM_ROWS)));
        eprintln!("{m:4} | {coop:10.1} | {blocked:10.1}");
    }
}
