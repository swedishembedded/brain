// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of the two shared ops a bilinear-position, axial-RoPE vision tower needs:
//!
//! * `rope2d_pairs`: rotates the adjacent channel pairs `(2d, 2d + 1)` in place on a fused buffer; its backward is the
//!   same kernel with the sign flipped, held to a finite difference of the forward;
//! * `rows_gather_weighted` and `rows_gather_weighted_bwd` with the host plan of `model::postable`: a learned table
//!   resampled by tap weights, the table gradient held to a finite difference (f64 oracle), and the bilinear plan for
//!   image grids equal to the reference helper's indices and weights (`tools/goldens/inkling2_vision_dump.py tiny`).

use data::rng::{random_seed, Lcg};
use model::hostmath::{gather_weighted_rows, rope_pairs_rows};
use model::postable::{bilinear_axis_taps, bilinear_grid_plan, DeviceTaps, PosTableKernels};

const PIPES: &[(&str, &str)] = &[
    ("rope2d_pairs", kernels::ROPE2D_PAIRS),
    ("rows_gather_weighted", kernels::ROWS_GATHER_WEIGHTED),
    ("rows_gather_weighted_bwd", kernels::ROWS_GATHER_WEIGHTED_BWD),
];

fn rotate_f64(x: &[f64], cos: &[f64], sin: &[f64], rows: usize, heads: usize, half: usize, stride: usize, off: usize, tmod: usize) -> Vec<f64> {
    let mut y = x.to_vec();
    for row in 0..rows {
        for h in 0..heads {
            for d in 0..half {
                let t = (row % tmod) * half + d;
                let base = row * stride + off + h * 2 * half + 2 * d;
                y[base] = x[base] * cos[t] - x[base + 1] * sin[t];
                y[base + 1] = x[base + 1] * cos[t] + x[base] * sin[t];
            }
        }
    }
    y
}

#[test]
fn rope2d_pairs_matches_the_reference_and_its_backward_is_the_finite_difference() {
    let g = gpu_core::testgpu::dev(PIPES);
    let kernel = g.kernel_index("rope2d_pairs").expect("rope2d_pairs registered");
    let mut rng = Lcg::new(random_seed());
    // A fused [rows, 3*dim] qkv buffer with the k region rotated; tmod < rows so the table repeats.
    let (rows, heads, half, tmod) = (7usize, 3usize, 4usize, 3usize);
    let dim = heads * 2 * half;
    let stride = 3 * dim;
    let angle = rng.vec_scaled(tmod * half, 3.0);
    let (cos, sin): (Vec<f32>, Vec<f32>) = angle.iter().map(|a| (a.cos(), a.sin())).unzip();
    let x = rng.vec(rows * stride);
    let dy = rng.vec(rows * stride);
    let run = |buf: &[f32], sign: f32| {
        let b = g.storage_init("buf", buf);
        let (c, s) = (g.storage_init("cos", &cos), g.storage_init("sin", &sin));
        let params = [rows as u32, heads as u32, half as u32, stride as u32, dim as u32, tmod as u32, sign.to_bits()];
        g.submit(&[], &[g.step(kernel, &[&b, &c, &s], &params, (rows * heads * half) as u32)]);
        g.read(&b, buf.len())
    };
    let fwd = run(&x, 1.0);
    let mut want = x.clone();
    rope_pairs_rows(&mut want, rows, heads, half, stride, dim, &cos, &sin, tmod, 1.0);
    for (i, (a, b)) in fwd.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-6, "forward[{i}] {a} vs host reference {b}");
    }
    // Only the k region moved: q and v columns are untouched.
    for row in 0..rows {
        for col in (0..dim).chain(2 * dim..stride) {
            assert_eq!(fwd[row * stride + col].to_bits(), x[row * stride + col].to_bits(), "row {row} col {col} outside the region changed");
        }
    }
    // Backward (sign -1) of dy is the gradient of sum(dy * rope(x)): compare with central differences in f64.
    let dx = run(&dy, -1.0);
    let (c64, s64): (Vec<f64>, Vec<f64>) = (cos.iter().map(|&v| f64::from(v)).collect(), sin.iter().map(|&v| f64::from(v)).collect());
    let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let loss = |x: &[f64]| rotate_f64(x, &c64, &s64, rows, heads, half, stride, dim, tmod).iter().zip(&dy).map(|(y, &d)| y * f64::from(d)).sum::<f64>();
    let h = 1e-5;
    for i in (0..x.len()).step_by(5) {
        let (mut hi, mut lo) = (x64.clone(), x64.clone());
        hi[i] += h;
        lo[i] -= h;
        let fd = (loss(&hi) - loss(&lo)) / (2.0 * h);
        assert!((f64::from(dx[i]) - fd).abs() < 1e-5, "dx[{i}] {} vs finite difference {fd}", dx[i]);
    }
}

#[test]
fn weighted_gather_forward_and_table_gradient_match_reference_and_finite_differences() {
    let g = gpu_core::testgpu::dev(PIPES);
    let mut rng = Lcg::new(random_seed());
    let (grid, side, width, merge) = (&[(4u32, 6u32), (2, 2), (6, 2)][..], 5u32, 7u32, 2u32);
    let plan = bilinear_grid_plan(grid, side, merge, false).unwrap();
    let rows = plan.rows();
    let table = rng.vec((side * side * width) as usize);
    let dout = rng.vec((rows * width) as usize);
    let k = PosTableKernels::by_name(&g).unwrap();
    let taps = DeviceTaps::upload(&g, &plan, side * side).unwrap();
    let (tb, db) = (g.storage_init("table", &table), g.storage_init("dout", &dout));
    let out = g.storage((rows * width) as u64);
    let dtable = g.storage_init("dtable", &vec![0.5f32; table.len()]);
    g.submit(&[], &[taps.gather_step(&g, &k, &tb, &out, width), taps.table_grad_step(&g, &k, &db, &dtable, width).unwrap()]);
    let got = g.read(&out, (rows * width) as usize);
    let want = gather_weighted_rows(&plan.idx, &plan.weight, 4, &table, width as usize);
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-6, "gather[{i}] {a} vs host reference {b}");
    }
    // The gradient accumulates: the buffer started at 0.5, and the rest is d(sum(dout * out))/d(table) by central differences.
    let grad = g.read(&dtable, table.len());
    let oracle = |t: &[f64]| -> f64 {
        let mut s = 0.0;
        for r in 0..rows as usize {
            for c in 0..width as usize {
                let o: f64 = (0..4).map(|j| f64::from(plan.weight[r * 4 + j]) * t[plan.idx[r * 4 + j] as usize * width as usize + c]).sum();
                s += o * f64::from(dout[r * width as usize + c]);
            }
        }
        s
    };
    let t64: Vec<f64> = table.iter().map(|&v| f64::from(v)).collect();
    let h = 1e-3;
    for i in 0..table.len() {
        let (mut hi, mut lo) = (t64.clone(), t64.clone());
        hi[i] += h;
        lo[i] -= h;
        let fd = (oracle(&hi) - oracle(&lo)) / (2.0 * h);
        assert!((f64::from(grad[i]) - 0.5 - fd).abs() < 1e-4, "dtable[{i}] {} - 0.5 vs finite difference {fd}", grad[i]);
    }
}

#[test]
fn bilinear_plan_has_the_reference_shape_and_rejects_malformed_grids() {
    // Partition of unity, and the identity when the grid is the table (half-pixel centres land on the rows).
    let plan = bilinear_grid_plan(&[(8, 8)], 8, 1, false).unwrap();
    for (r, w) in plan.weight.chunks(4).enumerate() {
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6, "patch {r} weights do not sum to one");
        assert_eq!(plan.idx[r * 4], r as u32, "patch {r} reads its own table row");
        assert_eq!(w[0], 1.0, "patch {r}");
    }
    // The edge clamps: a coordinate left of the first centre reads row 0 twice.
    let (taps, w) = bilinear_axis_taps(0, 100, 8, false);
    assert_eq!(taps, [0, 0]);
    assert!((w[0] + w[1] - 1.0).abs() < 1e-6);
    // Malformed grids are refused by name.
    for (grids, merge) in [(&[(3u32, 4u32)][..], 2u32), (&[(0, 2)][..], 2), (&[(2, 2)][..], 0)] {
        assert!(bilinear_grid_plan(grids, 8, merge, false).is_err(), "{grids:?} merge {merge}");
    }
    assert!(bilinear_grid_plan(&[(2, 2)], 0, 1, false).is_err());
    let mut bad = bilinear_grid_plan(&[(2, 2)], 4, 1, false).unwrap();
    bad.idx[3] = 99;
    assert!(bad.transpose(16).unwrap_err().contains("table row 99"));
}

#[test]
fn bilinear_plan_equals_the_reference_interpolation_taps_and_weights() {
    let dir = brain_testutil::testdata_path("golden/inkling2/vision/tiny");
    let Some(src) = brain_testutil::golden::Source::open(&dir, "tools/goldens/inkling2_vision_dump.py tiny") else { return };
    if !src.require(&[("posemb_grid_size", 8)]) {
        return;
    }
    let golden = brain_testutil::parity::load(&dir.join("golden.safetensors"));
    // The three images' grids, counted in language-model tokens; the tower's patch grid is merge times larger.
    let merge = 2u32;
    let grids: Vec<(u32, u32)> = golden["image_grid_thw"].data.chunks(3).map(|g| (g[1] as u32 * merge, g[2] as u32 * merge)).collect();
    let plan = bilinear_grid_plan(&grids, 8, merge, false).unwrap();
    let want_idx = &golden["interp_indices"].data;
    let want_w = &golden["interp_weights"].data;
    assert_eq!(plan.idx.len(), want_idx.len());
    assert!(plan.idx.iter().zip(want_idx).all(|(&a, &b)| a as f32 == b), "tap indices differ from the reference");
    let worst = plan.weight.iter().zip(want_w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(worst < 1e-6, "tap weights differ from the reference by {worst}");
}
