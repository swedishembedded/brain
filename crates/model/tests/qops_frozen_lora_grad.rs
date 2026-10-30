// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA and full fine-tuning
// over quantized checkpoints for its clients. If your team needs expertise in
// verifying gradients through frozen low-precision weights against a decoded
// reference, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The quantized-execution contract of `model::ops`, gated for every `Weight` tier.
//!
//! Spec: a frozen base linear of ANY storage tier supports the three products training needs through
//! the same `Ops` entry points - forward `Z = H W^T` (`matmul_rows`), input gradient `dH = dZ W`
//! (`matmul_dx`) and weight gradient `dW = dZ^T H` into a chosen gradient dtype (`weight_grad`) - and a
//! frozen-base LoRA step built from them (`Y1 = X W1^T + s B A x`, `Z = Y1 W2^T`, loss gradient `dZ`,
//! `dH = dZ W2`, then `dA`, `dB` of the adapter on layer 1) gives the gradients an f64 reference over
//! the DECODED fp32 weights gives. The oracle is therefore exact decoding of the stored bytes (each tier
//! decoded here by its own independent host code) followed by f64 arithmetic: what remains is the
//! arithmetic of the device path (fp32 accumulation on the portable kernels, TF32/bf16 tensor-core
//! products where the device has them), so the tolerances are chosen from the path the device took and
//! reported with every number.
//!
//! The same results must come out whether the quantized weight is decoded in one tile or in many
//! one-row-scale tiles (`set_tile_budget_bytes`): the scratch is bounded by the budget, not by the
//! tensor.

use data::rng::Lcg;
use gpu_core::provider::library::{GemmPath, GemmPrecision};
use gpu_core::Gpu;
use model::ops::{training_kernel_list, GradDtype, Ops, Weight};

mod tier_kit;
use tier_kit::{bf16_round, build_case, Case};

const M: usize = 16;
const K: usize = 256;
const H: usize = 256;
const O: usize = 64;
const RANK: usize = 4;
const LORA_SCALE: f32 = 0.5;

/// The exact fp32 reference tier: the oracle tolerances of the first tests are stated against it.
fn ops() -> Ops {
    let mut ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    ops.set_gemm_precision(GemmPrecision::Fp32Reference);
    ops
}


const TIERS: [&str; 13] = ["f32", "bf16", "nf4", "i8", "fp8_e4m3", "fp8_e5m2", "q4k", "q5k", "q6k", "nvfp4", "mxfp4", "mxfp8_e4m3", "mxfp8_e5m2"];

fn matmul_nt(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    // a [m,k] @ b[n,k]^T -> [m,n]
    let mut out = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|t| a[i * k + t] * b[j * k + t]).sum();
        }
    }
    out
}

fn matmul_nn(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    // a [m,k] @ b[k,n] -> [m,n]
    let mut out = vec![0f64; m * n];
    for i in 0..m {
        for t in 0..k {
            let av = a[i * k + t];
            for j in 0..n {
                out[i * n + j] += av * b[t * n + j];
            }
        }
    }
    out
}

fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0f64; a.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// Agreement of a device result with the f64 reference: the four numbers the gate states.
struct Agreement {
    max_abs: f64,
    rel_l2: f64,
    cosine: f64,
    nonfinite: usize,
}

fn agreement(got: &[f32], want: &[f64]) -> Agreement {
    assert_eq!(got.len(), want.len(), "length");
    let nonfinite = got.iter().filter(|v| !v.is_finite()).count();
    let (mut dot, mut gg, mut ww, mut dd, mut max_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let g = if g.is_finite() { g as f64 } else { 0.0 };
        dot += g * w;
        gg += g * g;
        ww += w * w;
        dd += (g - w) * (g - w);
        max_abs = max_abs.max((g - w).abs());
    }
    Agreement { max_abs, rel_l2: (dd / ww.max(1e-30)).sqrt(), cosine: dot / (gg.sqrt() * ww.sqrt()).max(1e-30), nonfinite }
}

/// Format- and path-specific tolerances: `(rel_l2, min_cosine, max_abs relative to the reference's peak)`.
fn tolerance(case: &Case, path: &GemmPath, bf16_grad: bool) -> (f64, f64, f64) {
    let tensor_cores = matches!(path, GemmPath::Bf16TensorCore);
    match (tensor_cores, case.bf16_path, bf16_grad) {
        (_, _, true) => (6e-3, 0.9999, 3e-2),
        (false, _, _) => (2e-5, 0.999_999, 2e-4),
        (true, true, _) => (6e-3, 0.9999, 5e-2),
        (true, false, _) => (3e-3, 0.99999, 2e-2),
    }
}

fn check(what: &str, case: &Case, path: &GemmPath, bf16_grad: bool, got: &[f32], want: &[f64], report: &mut Vec<String>) {
    let a = agreement(got, want);
    let (rel_tol, cos_tol, abs_tol) = tolerance(case, path, bf16_grad);
    let peak = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
    report.push(format!(
        "{:<9} {:<18} max_abs {:.3e} ({:.2e} of peak) rel_l2 {:.3e} cos {:.8} nonfinite {}  (tol rel_l2 {rel_tol:.0e} cos {cos_tol} abs {abs_tol:.0e})",
        case.label,
        what,
        a.max_abs,
        a.max_abs / peak,
        a.rel_l2,
        a.cosine,
        a.nonfinite
    ));
    assert_eq!(a.nonfinite, 0, "{} {what}: non-finite values", case.label);
    assert!(a.rel_l2 <= rel_tol, "{} {what}: rel_l2 {:.3e} > {rel_tol:e}", case.label, a.rel_l2);
    assert!(a.cosine >= cos_tol, "{} {what}: cosine {:.8} < {cos_tol}", case.label, a.cosine);
    assert!(a.max_abs <= abs_tol * peak, "{} {what}: max_abs {:.3e} > {abs_tol:e} of peak {peak:.3e}", case.label, a.max_abs);
}

fn unpack_bf16(words: &[f32], count: usize) -> Vec<f32> {
    (0..count).map(|i| f32::from_bits((words[i / 2].to_bits() >> ((i % 2) * 16)) << 16)).collect()
}

/// One frozen-base LoRA step through `ops` for one tier, compared with the f64 reference.
fn run_case(ops: &Ops, label: &'static str, report: &mut Vec<String>) {
    let g = ops.gpu();
    let mut rng = Lcg::random();
    let w1 = build_case(ops, label, H, K, &mut rng);
    let w2 = build_case(ops, label, O, H, &mut rng);
    let path = ops.gemm_path().clone();

    let x_h = rng.vec_scaled(M * K, 1.0);
    let a_h = rng.vec_scaled(RANK * K, 0.3);
    let b_h = rng.vec_scaled(H * RANK, 0.3);
    let t_h = rng.vec_scaled(M * O, 1.0);

    // f64 reference over the decoded weights.
    let (x, a, b, t) = (f64s(&x_h), f64s(&a_h), f64s(&b_h), f64s(&t_h));
    let (d1, d2) = (f64s(&w1.decoded), f64s(&w2.decoded));
    let xa = matmul_nt(&x, &a, M, K, RANK); // [M, RANK]
    let lora = matmul_nt(&xa, &b, M, RANK, H); // [M, H]
    let base = matmul_nt(&x, &d1, M, K, H);
    let hid: Vec<f64> = base.iter().zip(&lora).map(|(p, q)| p + LORA_SCALE as f64 * q).collect();
    let z = matmul_nt(&hid, &d2, M, H, O);
    let dz: Vec<f64> = z.iter().zip(&t).map(|(p, q)| p - q).collect();
    let dh = matmul_nn(&dz, &d2, M, O, H); // dz [M,O] @ W2 [O,H]
    let dw2 = matmul_nn(&transpose(&dz, M, O), &hid, O, M, H); // [O, H]
    let adapter_grads = |dh: &[f64]| {
        let db = matmul_nn(&transpose(dh, M, H), &xa, H, M, RANK).iter().map(|v| v * LORA_SCALE as f64).collect::<Vec<_>>();
        let dhb = matmul_nn(dh, &b, M, H, RANK); // [M, RANK]
        let da = matmul_nn(&transpose(&dhb, M, RANK), &x, RANK, M, K).iter().map(|v| v * LORA_SCALE as f64).collect::<Vec<_>>();
        (da, db)
    };
    let (da_ref, db_ref) = adapter_grads(&dh);

    // Device: each stage fed the reference's input, so the stages are compared in isolation.
    let run = |w: &Weight, input: &[f32], rows: usize, out_cols: usize| -> Vec<f32> {
        let xin = g.storage_init("in", input);
        let y = g.storage((rows * out_cols) as u64);
        let mut steps = Vec::new();
        let act = ops.act_f32(&xin, 0, rows as u32, w.k());
        ops.matmul_rows(&mut steps, w, (0, w.n()), &act, &y, 0);
        g.submit(&[], &steps);
        g.read(&y, rows * out_cols)
    };
    let y1 = run(&w1.weight, &x_h, M, H);
    check("forward W1", &w1, &path, false, &y1, &base, report);
    let hid_h: Vec<f32> = hid.iter().map(|&v| v as f32).collect();
    let z_dev = run(&w2.weight, &hid_h, M, O);
    check("forward W2", &w2, &path, false, &z_dev, &z, report);

    let dz_h: Vec<f32> = dz.iter().map(|&v| v as f32).collect();
    let dz_buf = g.storage_init("dz", &dz_h);
    let dh_buf = g.storage((M * H) as u64);
    let mut steps = Vec::new();
    ops.matmul_dx(&mut steps, &w2.weight, &dz_buf, M as u32, &dh_buf, false);
    g.submit(&[], &steps);
    let dh_dev = g.read(&dh_buf, M * H);
    check("input grad dH", &w2, &path, false, &dh_dev, &dh, report);

    // Accumulating input gradient: adds to what dX holds.
    let prior = rng.vec_scaled(M * H, 1.0);
    let acc_buf = g.storage_init("acc", &prior);
    let mut steps = Vec::new();
    ops.matmul_dx(&mut steps, &w2.weight, &dz_buf, M as u32, &acc_buf, true);
    g.submit(&[], &steps);
    let want_acc: Vec<f64> = dh.iter().zip(&prior).map(|(d, p)| d + *p as f64).collect();
    check("input grad accum", &w2, &path, false, &g.read(&acc_buf, M * H), &want_acc, report);

    // The adapter gradients of layer 1 from the device dH.
    let dh_dev64 = f64s(&dh_dev);
    let (da_dev, db_dev) = adapter_grads(&dh_dev64);
    let as_f32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
    check("LoRA dA", &w2, &path, false, &as_f32(&da_dev), &da_ref, report);
    check("LoRA dB", &w2, &path, false, &as_f32(&db_dev), &db_ref, report);

    // Weight gradient of W2 (full fine-tuning), fp32 and bf16, accumulated twice.
    let hid_buf = g.storage_init("hid", &hid_h);
    let dw_f32 = g.storage((O * H) as u64);
    let mut steps = Vec::new();
    ops.weight_grad(&mut steps, &w2.weight, &hid_buf, &dz_buf, M as u32, &dw_f32, GradDtype::F32);
    g.submit(&[], &steps);
    check("dW fp32", &w2, &path, false, &g.read(&dw_f32, O * H), &dw2, report);

    let dw_bf16 = g.storage((O * H / 2) as u64);
    for _ in 0..2 {
        let mut steps = Vec::new();
        ops.weight_grad(&mut steps, &w2.weight, &hid_buf, &dz_buf, M as u32, &dw_bf16, GradDtype::Bf16);
        g.submit(&[], &steps);
    }
    let twice: Vec<f64> = dw2.iter().map(|v| 2.0 * v).collect();
    check("dW bf16 (x2 acc)", &w2, &path, true, &unpack_bf16(&g.read(&dw_bf16, O * H / 2), O * H), &twice, report);
}

#[test]
fn every_weight_tier_gives_the_lora_gradients_of_the_decoded_reference() {
    let mut report = Vec::new();
    for label in TIERS {
        run_case(&ops(), label, &mut report);
    }
    println!("{}", report.join("\n"));
}

/// A tile of rows decodes to exactly the host decode of those rows (f32), and, where the device decodes to
/// bf16 for the tensor cores, to that value rounded to bf16 (round to nearest even), for every tier and for
/// a window that does not start at the first row.
#[test]
fn decoded_tiles_equal_the_host_decode_of_their_rows() {
    let (first, count) = (3usize, 9usize);
    for label in TIERS {
        let mut ops = ops();
        ops.set_gemm_precision(GemmPrecision::Auto);
        let case = build_case(&ops, label, 24, K, &mut Lcg::random());
        let g = ops.gpu();
        let want = &case.decoded[first * K..(first + count) * K];
        if label != "f32" {
            let out = g.storage((count * K) as u64);
            let mut steps = Vec::new();
            ops.decode_tile(&mut steps, &case.weight, first as u32, count as u32, &out);
            g.submit(&[], &steps);
            let got = g.read(&out, count * K);
            assert!(got.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()), "{label}: f32 tile differs from the host decode");
        }
        let out16 = g.storage((count * K / 2) as u64);
        let mut steps = Vec::new();
        if label != "f32" && ops.decode_tile_bf16(&mut steps, &case.weight, first as u32, count as u32, &out16) {
            g.submit(&[], &steps);
            let got = unpack_bf16(&g.read(&out16, count * K / 2), count * K);
            assert!(got.iter().zip(want).all(|(a, b)| a.to_bits() == bf16_round(*b).to_bits()), "{label}: bf16 tile differs from the rounded host decode");
        }
    }
}

/// The default tier of a trained model: tensor-core products wherever the device has them (TF32 for the decoded
/// tiles, bf16 for bf16/NF4), the fp32 kernels elsewhere. The tolerances follow the path the device took.
#[test]
fn the_tensor_core_tier_stays_within_its_stated_tolerances() {
    let mut report = Vec::new();
    for label in TIERS {
        let mut ops = ops();
        ops.set_gemm_precision(GemmPrecision::Auto);
        report.push(format!("{label}: gemm path {:?}", ops.gemm_path()));
        run_case(&ops, label, &mut report);
    }
    println!("{}", report.join("\n"));
}

/// Inside a `decoded_weights` scope a quantized weight is decoded for the first product that reads it and
/// shared by the later ones (forward, rebuild and input gradient of a training layer); the values do not change.
#[test]
fn a_weight_is_decoded_once_per_scope_and_the_products_are_unchanged() {
    let mut ops = ops();
    ops.set_gemm_precision(GemmPrecision::Auto);
    for label in ["nf4", "i8", "fp8_e4m3"] {
        let mut rng = Lcg::random();
        let case = build_case(&ops, label, H, K, &mut rng);
        let g = ops.gpu();
        let x = g.storage_init("x", &rng.vec_scaled(M * K, 1.0));
        let product = |steps: &mut Vec<_>| {
            let y = g.storage((M * H) as u64);
            ops.matmul_rows(steps, &case.weight, (0, H as u32), &ops.act_f32(&x, 0, M as u32, K as u32), &y, 0);
            y
        };
        let mut steps = Vec::new();
        let plain = product(&mut steps);
        let plain_steps = steps.len();
        g.submit(&[], &steps);
        let want = g.read(&plain, M * H);

        let _scope = ops.decoded_weights();
        let mut scoped = Vec::new();
        let first = product(&mut scoped);
        let after_first = scoped.len();
        let second = product(&mut scoped);
        let second_steps = scoped.len() - after_first;
        g.submit(&[], &scoped);
        assert_eq!(g.read(&first, M * H), want, "{label}: the scoped product differs from the unscoped one");
        assert_eq!(g.read(&second, M * H), want, "{label}: the shared decode gives a different product");
        if after_first == plain_steps {
            // The device decodes this tier to f32 (no bf16 tile for the library product): nothing is shared.
            brain_testutil::skip_unavailable(&format!("{label}: no bf16 tile decode on this device"));
            continue;
        }
        assert!(second_steps < after_first, "{label}: the second product decoded the weight again ({second_steps} steps after {after_first})");
    }
}

/// Decoding a weight in many tiles (a budget of three rows of `k` f32) gives the results of one tile.
#[test]
fn a_small_tile_budget_changes_nothing_but_the_scratch() {
    let mut report = Vec::new();
    for label in ["nf4", "i8", "fp8_e4m3", "q4k", "q6k"] {
        let mut ops = ops();
        ops.set_gemm_precision(GemmPrecision::Auto);
        ops.set_tile_budget_bytes(3 * 4 * K as u64);
        run_case(&ops, label, &mut report);
    }
    println!("{}", report.join("\n"));
}

/// The row norms of a weight and its product with a small dense matrix, taken through the bounded tile
/// decode, equal what the decoded fp32 weight gives - for every tier and whether the weight is decoded in
/// one tile or in single rows. DoRA's magnitude normalisation reads exactly these two quantities of its
/// frozen base.
#[test]
fn row_norms_and_projections_through_the_tile_decode_equal_the_decoded_weight_of_every_tier() {
    const N: usize = 24;
    const R: usize = 3;
    for budget in [u64::MAX / 8, 3 * 4 * K as u64] {
        for label in TIERS {
            let mut ops = ops();
            ops.set_tile_budget_bytes(budget);
            let mut rng = Lcg::random();
            let case = build_case(&ops, label, N, K, &mut rng);
            let g = ops.gpu();
            let x = rng.vec_scaled(R * K, 1.0);
            let (xb, norms, proj) = (g.storage_init("x", &x), g.storage((N) as u64), g.storage((N * R) as u64));
            let mut steps = Vec::new();
            ops.row_sq_norms(&mut steps, &case.weight, &norms);
            ops.project_rows(&mut steps, &case.weight, &xb, R as u32, &proj);
            g.submit(&[], &steps);

            let w = f64s(&case.decoded);
            let want_norms: Vec<f64> = (0..N).map(|n| w[n * K..(n + 1) * K].iter().map(|v| v * v).sum()).collect();
            let want_proj: Vec<f64> = (0..N * R).map(|i| (0..K).map(|t| w[(i / R) * K + t] * x[(i % R) * K + t] as f64).sum()).collect();
            for (what, got, want) in [("row norms", g.read(&norms, N), want_norms), ("projection", g.read(&proj, N * R), want_proj)] {
                let a = agreement(&got, &want);
                assert!(a.nonfinite == 0 && a.rel_l2 < 1e-5, "{label} at a {budget} byte tile: {what} rel_l2 {} max_abs {}", a.rel_l2, a.max_abs);
            }
        }
    }
}
