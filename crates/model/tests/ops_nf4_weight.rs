// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-lean fine-tuning of large language
// models on a single GPU for its clients. If your team needs expertise in
// training adapters over a 4-bit frozen base then you can procure our
// services by sending an email to info@swedishembedded.com.

//! `Weight::NF4`, the float-read 4-bit frozen-weight tier (QLoRA).
//!
//! Spec: a NF4 weight uploaded through `Weight::upload` is read by `Ops::matmul`
//! (`y = x * W^T`) and `Ops::matmul_dx` (`dX = dY * W`) as `NF4_LUT[code] *
//! group_scale` against UNQUANTISED f32 activations with f32 accumulation. The
//! oracle is therefore exact weight dequantisation (`dequantize_weight_lut4`)
//! followed by an ordinary f64 matmul: the only difference left is f32
//! accumulation order. Shapes are non-square, `k` is not a multiple of the
//! 128-wide tile or the 8-wide contraction chunk, and `n` is large in one case.
//!
//! The host tests run on the CPU backend (`BRAIN_DEVICE=cpu`), where the naive
//! kernels execute; the register-tiled variants need workgroup barriers and are
//! covered by the `--ignored` device test, which also measures the cost of the
//! on-the-fly dequantisation against the bf16 tier at real widths.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::lut4::{dequantize_weight_lut4, quantize_weight_lut4, NF4_LUT};
use model::ops::{training_kernel_list, Ops, TierPolicy, Weight};

fn host_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0f64; m * n];
    for r in 0..m {
        for j in 0..n {
            out[r * n + j] = (0..k).map(|i| x[r * k + i] as f64 * w[j * k + i] as f64).sum();
        }
    }
    out
}

fn host_matmul_dx(dy: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0f64; m * k];
    for r in 0..m {
        for c in 0..k {
            out[r * k + c] = (0..n).map(|j| dy[r * n + j] as f64 * w[j * k + c] as f64).sum();
        }
    }
    out
}

fn assert_close(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f64, |a, &v| a.max(v.abs())).max(1e-6);
    let worst = got.iter().zip(want).map(|(&g, &w)| (g as f64 - w).abs()).fold(0f64, f64::max);
    assert!(worst <= 1e-4 * scale, "{what}: worst abs error {worst:e} exceeds 1e-4 of the output scale {scale:e}");
}

fn ops() -> Ops {
    Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new")
}

fn forward(ops: &Ops, w: &Weight, x: &DeviceBuffer, xr0: u32, m: u32) -> Vec<f32> {
    let g = ops.gpu();
    let y = g.storage(m as u64 * w.n() as u64);
    let mut steps = Vec::new();
    let act = ops.act_f32(x, xr0, m, w.k());
    ops.matmul(&mut steps, w, &act, &y, 0);
    g.submit(&[], &steps);
    g.read(&y, (m * w.n()) as usize)
}

fn backward(ops: &Ops, w: &Weight, dy: &DeviceBuffer, m: u32, seed_dx: Option<&[f32]>) -> Vec<f32> {
    let g = ops.gpu();
    let len = (m * w.k()) as usize;
    let dx = match seed_dx {
        Some(init) => g.storage_init("dx", init),
        None => g.storage(len as u64),
    };
    let mut steps = Vec::new();
    ops.matmul_dx(&mut steps, w, dy, m, &dx, seed_dx.is_some());
    g.submit(&[], &steps);
    g.read(&dx, len)
}

/// (m, n, k): non-square; k a multiple of the quantisation group (32) but of
/// neither the 128 tile nor, for 160 and 224, a power of two; n large in one.
const SHAPES: [(usize, usize, usize); 4] = [(5, 70, 96), (37, 33, 160), (3, 4100, 64), (130, 131, 224)];

#[test]
fn nf4_forward_and_input_gradient_match_the_dequantised_f32_oracle() {
    let ops = ops();
    let g = ops.gpu();
    for (m, n, k) in SHAPES {
        let mut rng = Lcg::random();
        let raw = rng.vec_scaled(n * k, 0.5);
        let x_h = rng.vec_scaled(m * k, 1.0);
        let dy_h = rng.vec_scaled(m * n, 1.0);

        let (packed, scales) = quantize_weight_lut4(&raw, n, k, &NF4_LUT);
        let deq = dequantize_weight_lut4(&packed, &scales, n, k, &NF4_LUT);

        let w = Weight::upload(&ops, &raw, n, k, Dtype::NF4);
        assert!(matches!(w, Weight::NF4 { .. }), "{m}x{n}x{k}: NF4 must not be promoted away on this device");
        assert_eq!((w.dtype(), w.n(), w.k()), (Dtype::NF4, n as u32, k as u32));

        let x = g.storage_init("x", &x_h);
        assert_close(&forward(&ops, &w, &x, 0, m as u32), &host_matmul(&x_h, &deq, m, k, n), &format!("forward {m}x{n}x{k}"));

        let dy = g.storage_init("dy", &dy_h);
        let want_dx = host_matmul_dx(&dy_h, &deq, m, k, n);
        assert_close(&backward(&ops, &w, &dy, m as u32, None), &want_dx, &format!("dx {m}x{n}x{k}"));

        // accumulate = true adds to what dX already holds.
        let prior = rng.vec_scaled(m * k, 1.0);
        let want_acc: Vec<f64> = want_dx.iter().zip(&prior).map(|(d, p)| d + *p as f64).collect();
        assert_close(&backward(&ops, &w, &dy, m as u32, Some(&prior)), &want_acc, &format!("dx accumulate {m}x{n}x{k}"));
    }
}

/// A sliced activation (rows `[64, 64+m)` of a taller buffer) must give the
/// rows a zero-offset dispatch over the same data gives: a wrong row offset
/// reads other rows and cannot reproduce this.
#[test]
fn nf4_forward_reads_the_activation_rows_it_was_given() {
    let ops = ops();
    let g = ops.gpu();
    let (m, n, k) = (7usize, 40usize, 96usize);
    let mut rng = Lcg::random();
    let w = Weight::upload(&ops, &rng.vec_scaled(n * k, 0.5), n, k, Dtype::NF4);
    let tall = rng.vec_scaled((64 + m) * k, 1.0);
    let want = forward(&ops, &w, &g.storage_init("rows", &tall[64 * k..]), 0, m as u32);
    let got = forward(&ops, &w, &g.storage_init("tall", &tall), 64, m as u32);
    assert_eq!(got, want);
}

/// d(loss)/d(x) through a frozen NF4 linear, by central finite differences of
/// the device forward: `loss = 0.5 * |x W^T|^2`, so `dY = y` and `dX = matmul_dx(y)`.
#[test]
fn nf4_input_gradient_agrees_with_finite_differences_of_the_forward() {
    let ops = ops();
    let g = ops.gpu();
    let (m, n, k) = (3usize, 20usize, 64usize);
    let mut rng = Lcg::random();
    let w = Weight::upload(&ops, &rng.vec_scaled(n * k, 0.5), n, k, Dtype::NF4);
    let x_h = rng.vec_scaled(m * k, 1.0);

    let loss = |x: &[f32]| -> f64 { forward(&ops, &w, &g.storage_init("x", x), 0, m as u32).iter().map(|&y| 0.5 * (y as f64).powi(2)).sum() };
    let y = forward(&ops, &w, &g.storage_init("x", &x_h), 0, m as u32);
    let dx = backward(&ops, &w, &g.storage_init("dy", &y), m as u32, None);

    let eps = 0.05f32;
    let scale = dx.iter().fold(0f32, |a, v| a.max(v.abs()));
    for i in (0..m * k).step_by(7) {
        let (mut hi, mut lo) = (x_h.clone(), x_h.clone());
        hi[i] += eps;
        lo[i] -= eps;
        let fd = (loss(&hi) - loss(&lo)) / (2.0 * eps as f64);
        assert!((fd - dx[i] as f64).abs() <= 1e-2 * scale as f64, "element {i}: finite difference {fd} vs analytic {}", dx[i]);
    }
}

#[test]
fn nf4_is_a_spellable_tier() {
    let p = TierPolicy::parse("nf4,in_proj_a.weight=f32").expect("nf4 parses");
    assert_eq!(p.want("layers.0.mlp.up.weight"), Dtype::NF4);
    assert_eq!(p.want("layers.0.in_proj_a.weight"), Dtype::F32);
    assert_eq!(p.describe(), "nf4,in_proj_a.weight=f32");
}

#[test]
fn the_training_kernel_list_registers_the_frozen_base_kernels() {
    let names: Vec<&str> = training_kernel_list().iter().map(|(n, _)| *n).collect();
    for want in ["matmul#w=nf4", "matmul_reg3#w=nf4", "matmul_dx#w=nf4", "matmul_dx_reg#w=nf4", "matmul_dx_reg#w=bf16"] {
        assert!(names.contains(&want), "training_kernel_list() lacks {want}");
    }
}

/// Real-width device run (`gpu cargo test ... -- --ignored --nocapture`, with
/// `BRAIN_DEVICE=gpu BRAIN_BACKEND=cuda`): the tiled NF4 and bf16 kernels against
/// the f32 tier of the same (dequantised / bf16-rounded) weight, and the cost of
/// decoding on the fly against the bf16 tier at m = 4096 rows.
#[test]
#[ignore = "needs a GPU: run under `gpu` with BRAIN_DEVICE=gpu BRAIN_BACKEND=cuda"]
fn nf4_tiled_kernels_match_the_f32_tier_and_report_their_cost_at_real_widths() {
    let ops = ops();
    let g = ops.gpu();
    assert!(ops.caps().workgroup_reductions, "this test needs a device that runs the register-tiled kernels, not the CPU JIT");
    for (n, k) in [(17408usize, 5120usize), (5120, 17408)] {
        let mut rng = Lcg::random();
        let raw = rng.vec_scaled(n * k, 0.05);
        let (packed, scales) = quantize_weight_lut4(&raw, n, k, &NF4_LUT);
        let deq = dequantize_weight_lut4(&packed, &scales, n, k, &NF4_LUT);
        let nf4 = Weight::upload(&ops, &raw, n, k, Dtype::NF4);
        let f32w = Weight::upload(&ops, &deq, n, k, Dtype::F32);
        let bf16 = Weight::upload(&ops, &raw, n, k, Dtype::BF16);
        // The bf16 tier's weight values: the high half of each f32 (`pack_bf16`).
        let bf16_rounded: Vec<f32> = raw.iter().map(|&v| f32::from_bits((model::half::f32_to_bf16(v) as u32) << 16)).collect();
        let bf16_as_f32 = Weight::upload(&ops, &bf16_rounded, n, k, Dtype::F32);

        let m = 512usize;
        let x = g.storage_init("x", &rng.vec_scaled(m * k, 1.0));
        let dy = g.storage_init("dy", &rng.vec_scaled(m * n, 1.0));
        let rel = |a: &[f32], b: &[f32]| {
            let (num, den) = a.iter().zip(b).fold((0f64, 0f64), |(nu, de), (&p, &q)| (nu + (p as f64 - q as f64).powi(2), de + (q as f64).powi(2)));
            (num / den).sqrt()
        };
        let fwd = rel(&forward(&ops, &nf4, &x, 0, m as u32), &forward(&ops, &f32w, &x, 0, m as u32));
        let bwd = rel(&backward(&ops, &nf4, &dy, m as u32, None), &backward(&ops, &f32w, &dy, m as u32, None));
        eprintln!("n={n} k={k} m={m}: nf4 vs f32-tier relative L2 forward {fwd:.2e} dx {bwd:.2e}");
        assert!(fwd < 5e-3 && bwd < 5e-3, "tiled NF4 kernels disagree with the f32 tier: forward {fwd:e} dx {bwd:e}");
        let (bf_fwd, bf_bwd) = (
            rel(&forward(&ops, &bf16, &x, 0, m as u32), &forward(&ops, &bf16_as_f32, &x, 0, m as u32)),
            rel(&backward(&ops, &bf16, &dy, m as u32, None), &backward(&ops, &bf16_as_f32, &dy, m as u32, None)),
        );
        eprintln!("n={n} k={k} m={m}: bf16 vs f32-tier relative L2 forward {bf_fwd:.2e} dx {bf_bwd:.2e}");
        assert!(bf_fwd < 5e-3 && bf_bwd < 5e-3, "tiled bf16 kernels disagree with the f32 tier: forward {bf_fwd:e} dx {bf_bwd:e}");

        let m = 4096usize;
        let x = g.storage_init("x", &rng.vec_scaled(m * k, 1.0));
        let dy = g.storage_init("dy", &rng.vec_scaled(m * n, 1.0));
        let (y, dx) = (g.storage((m * n) as u64), g.storage((m * k) as u64));
        for (label, w) in [("nf4 ", &nf4), ("bf16", &bf16)] {
            let (mut fs, mut bs) = (Vec::new(), Vec::new());
            let act = ops.act_f32(&x, 0, m as u32, k as u32);
            ops.matmul(&mut fs, w, &act, &y, 0);
            ops.matmul_dx(&mut bs, w, &dy, m as u32, &dx, false);
            let time = |steps: &[gpu_core::Step]| {
                g.submit(&[], steps);
                g.read(&y, 1);
                let t = std::time::Instant::now();
                for _ in 0..5 {
                    g.submit(&[], steps);
                }
                g.read(&y, 1);
                t.elapsed().as_secs_f64() * 1e3 / 5.0
            };
            eprintln!("n={n} k={k} m={m} {label}: matmul {:.2} ms, matmul_dx {:.2} ms", time(&fs), time(&bs));
        }
    }
}
