// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P1.glue — tests for the nine elementwise/layout glue WGSL kernels.
//!
//! Written FROM the spec (hand-computed references, required properties,
//! edge cases, required tests, determinism) — NEVER from the implementation.
//!
//! These tests do NOT build any model: they drive the WGSL kernels directly
//! via `gpu_core`, exactly like `mse_fd.rs`. Gating runs are
//! `BRAIN_DEVICE=cpu` (wm-locked-make.sh sets this; `Gpu::new` then selects
//! the CPU backend, so no MOE_SKIP_GPU_TESTS skip is needed here — nothing
//! in this file ever touches a physical GPU under gating).
//!
//! Tolerances are the spec's and are never loosened: 1e-6 for arithmetic
//! hand references, bitwise (`f32::to_bits`) for pure copies / roundtrips /
//! determinism, exact f32 `==` for the edm_mix skip-identity, and the global
//! gradcheck tolerances (h=5e-3, atol=4e-3, rtol=8e-2) for the FD entry.

use data::rng::Lcg;
use gpu_core::{f, DeviceBuffer, Gpu};

// Kernel order passed to Gpu::new; indices below reference these.
static KERNELS: &[(&str, &str)] = &[
    ("mul", kernels::MUL),                 // 0
    ("scale_row", kernels::SCALE_ROW),     // 1
    ("edm_mix", kernels::EDM_MIX),         // 2
    ("mse_value_w", kernels::MSE_VALUE_W), // 3
    ("mse_grad_w", kernels::MSE_GRAD_W),   // 4
    ("pad2d", kernels::PAD2D),             // 5
    ("crop2d", kernels::CROP2D),           // 6
    ("nchw_nlc", kernels::NCHW_NLC),       // 7
    ("nlc_nchw", kernels::NLC_NCHW),       // 8
];
const K_MUL: usize = 0;
const K_SCALE_ROW: usize = 1;
const K_EDM_MIX: usize = 2;
const K_MSE_VALUE_W: usize = 3;
const K_MSE_GRAD_W: usize = 4;
const K_PAD2D: usize = 5;
const K_CROP2D: usize = 6;
const K_NCHW_NLC: usize = 7;
const K_NLC_NCHW: usize = 8;

/// Generic single-step dispatch: inputs (in §8 buffer order), fresh output
/// buffer last, one submit, read back. Every call allocates FRESH buffers.
fn run(
    gpu: &Gpu,
    kind: usize,
    inputs: &[&[f32]],
    params: &[u32],
    out_len: usize,
    threads: u32,
) -> Vec<f32> {
    let in_bufs: Vec<DeviceBuffer> = inputs
        .iter()
        .enumerate()
        .map(|(i, d)| gpu.storage_init(&format!("in{i}"), d))
        .collect();
    let out = gpu.storage(out_len as u64);
    let mut bufs: Vec<&DeviceBuffer> = in_bufs.iter().collect();
    bufs.push(&out);
    let st = gpu.step(kind, &bufs, params, threads);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&out, out_len)
}

// ---- §8 dispatch-table wrappers ---------------------------------------------

/// `mul`: y[i] = a[i]*b[i]; params [n]; n threads.
fn k_mul(gpu: &Gpu, a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    run(gpu, K_MUL, &[a, b], &[n as u32], n, n as u32)
}

/// `scale_row`: y[i] = s[i/m]*x[i]; params [total, m]; total threads.
fn k_scale_row(gpu: &Gpu, x: &[f32], s: &[f32], m: usize) -> Vec<f32> {
    let total = x.len();
    assert_eq!(total, s.len() * m);
    run(gpu, K_SCALE_ROW, &[x, s], &[total as u32, m as u32], total, total as u32)
}

/// `edm_mix`: y[i] = ab[2n]*x[i] + ab[2n+1]*f[i], n = i/m; params [total, m].
fn k_edm_mix(gpu: &Gpu, x: &[f32], fbuf: &[f32], ab: &[f32], m: usize) -> Vec<f32> {
    let total = x.len();
    assert_eq!(total, fbuf.len());
    assert_eq!(ab.len() * m, 2 * total);
    run(gpu, K_EDM_MIX, &[x, fbuf, ab], &[total as u32, m as u32], total, total as u32)
}

/// `mse_value_w`: out[n] = w[n]*Σ_m d²/M; params [n, m]; n (SAMPLE) threads.
fn k_mse_value_w(gpu: &Gpu, pred: &[f32], tgt: &[f32], w: &[f32], m: usize) -> Vec<f32> {
    let n = w.len();
    assert_eq!(pred.len(), n * m);
    assert_eq!(tgt.len(), n * m);
    run(gpu, K_MSE_VALUE_W, &[pred, tgt, w], &[n as u32, m as u32], n, n as u32)
}

/// `mse_grad_w`: dpred[i] = w[i/m]*2*(pred-tgt)/M*scale; params [total, m, f(scale)].
fn k_mse_grad_w(gpu: &Gpu, pred: &[f32], tgt: &[f32], w: &[f32], m: usize, scale: f32) -> Vec<f32> {
    let total = pred.len();
    assert_eq!(total, w.len() * m);
    run(
        gpu,
        K_MSE_GRAD_W,
        &[pred, tgt, w],
        &[total as u32, m as u32, f(scale)],
        total,
        total as u32,
    )
}

/// `pad2d`: x is [NC, h, w] (unpadded); output [NC, h+t+b, w+l+r].
/// Params [total_out, h, w, l, r, t, b]; total_out threads.
fn k_pad2d(gpu: &Gpu, x: &[f32], h: usize, w: usize, o: (usize, usize, usize, usize)) -> Vec<f32> {
    let (l, r, t, b) = o;
    assert_eq!(x.len() % (h * w), 0);
    let nc = x.len() / (h * w);
    let total = nc * (h + t + b) * (w + l + r);
    run(
        gpu,
        K_PAD2D,
        &[x],
        &[total as u32, h as u32, w as u32, l as u32, r as u32, t as u32, b as u32],
        total,
        total as u32,
    )
}

/// `crop2d`: x is the PADDED tensor [NC, h+t+b, w+l+r]; output [NC, h, w].
/// Identical Params layout to pad2d up to total.
fn k_crop2d(gpu: &Gpu, x: &[f32], h: usize, w: usize, o: (usize, usize, usize, usize)) -> Vec<f32> {
    let (l, r, t, b) = o;
    let (hp, wp) = (h + t + b, w + l + r);
    assert_eq!(x.len() % (hp * wp), 0);
    let nc = x.len() / (hp * wp);
    let total = nc * h * w;
    run(
        gpu,
        K_CROP2D,
        &[x],
        &[total as u32, h as u32, w as u32, l as u32, r as u32, t as u32, b as u32],
        total,
        total as u32,
    )
}

/// `nchw_nlc`: NCHW [N,C,hw] -> NLC [N,hw,C]; params [total, c, hw].
fn k_nchw_nlc(gpu: &Gpu, x: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let total = x.len();
    assert_eq!(total % (c * hw), 0);
    run(gpu, K_NCHW_NLC, &[x], &[total as u32, c as u32, hw as u32], total, total as u32)
}

/// `nlc_nchw`: NLC [N,hw,C] -> NCHW [N,C,hw]; params [total, c, hw].
fn k_nlc_nchw(gpu: &Gpu, x: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let total = x.len();
    assert_eq!(total % (c * hw), 0);
    run(gpu, K_NLC_NCHW, &[x], &[total as u32, c as u32, hw as u32], total, total as u32)
}

// ---- assertion helpers -------------------------------------------------------

fn assert_close(got: &[f32], want: &[f32], atol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length {} != {}", got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= atol,
            "{what}[{i}]: got {g}, want {w} (atol {atol})"
        );
    }
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length {} != {}", got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}[{i}]: got {g} ({:#010x}), want {w} ({:#010x}) — must be BITWISE equal",
            g.to_bits(),
            w.to_bits()
        );
    }
}

/// Host f32 dot in ascending index order (spec §6.1 summation-order contract).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ---- §9.1 mul -----------------------------------------------------------------

#[test]
fn glue_mul_matches_hand_reference() {
    // Spec §5.1.
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let a = [1.5f32, -2.0, 0.25, 3.0];
    let b = [4.0f32, 0.5, -8.0, -1.0];

    let y = k_mul(&gpu, &a, &b);
    assert_close(&y, &[6.0, -1.0, -2.0, -3.0], 1e-6, "mul y");

    // Backward COMPOSES from the same kernel: da = mul(dy,b), db = mul(dy,a).
    let dy = [1.0f32, 2.0, 3.0, 4.0];
    let da = k_mul(&gpu, &dy, &b);
    assert_close(&da, &[4.0, 1.0, -24.0, -4.0], 1e-6, "mul da");
    let db = k_mul(&gpu, &dy, &a);
    assert_close(&db, &[1.5, -4.0, 0.75, 12.0], 1e-6, "mul db");
}

// ---- §9.2 scale_row ------------------------------------------------------------

#[test]
fn glue_scale_row_matches_hand_reference() {
    // Spec §5.2: N=2, M=3.
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let s = [2.0f32, -0.5];
    let x = [1.0f32, 2.0, 3.0, 4.0, 6.0, -2.0];

    let y = k_scale_row(&gpu, &x, &s, 3);
    assert_close(&y, &[2.0, 4.0, 6.0, -2.0, -3.0, 1.0], 1e-6, "scale_row y");

    // Backward composition (§3.2): dx = scale_row(dy, s); no ds by design.
    let dy = [1.0f32, 1.0, 0.0, 2.0, -2.0, 4.0];
    let dx = k_scale_row(&gpu, &dy, &s, 3);
    assert_close(&dx, &[2.0, 2.0, 0.0, -1.0, 1.0, -2.0], 1e-6, "scale_row dx");
}

// ---- §9.3 edm_mix ---------------------------------------------------------------

#[test]
fn glue_edm_mix_matches_hand_reference() {
    // Spec §5.3: N=2, M=2, ab packed [a0,b0,a1,b1].
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let ab = [2.0f32, 0.5, -1.0, 3.0];
    let x = [1.0f32, -2.0, 3.0, 0.5];
    let fv = [4.0f32, 8.0, -2.0, 1.0];

    let y = k_edm_mix(&gpu, &x, &fv, &ab, 2);
    assert_close(&y, &[4.0, 0.0, -9.0, 2.5], 1e-6, "edm_mix y");
}

// ---- §9.4 edm_mix skip-identity (§6.3) --------------------------------------------

#[test]
fn glue_edm_mix_identity_skip() {
    // Spec §6.3 / §3.3: a=1, b=0 => y[i] == x[i] EXACT f32 equality
    // (`==`, NOT bitwise: x = -0.0 yields +0.0).
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, m) = (2usize, 6usize);
    let mut st = Lcg::new(0x1D_5EEDu64);
    let x = st.vec(n * m);
    let fv = st.vec(n * m);
    let ab = [1.0f32, 0.0, 1.0, 0.0]; // a_vec = 1.0, b_vec = 0.0 packed

    let y = k_edm_mix(&gpu, &x, &fv, &ab, m);
    assert_eq!(y.len(), x.len());
    for (i, (&got, &want)) in y.iter().zip(x.iter()).enumerate() {
        assert!(
            got == want,
            "edm_mix identity y[{i}]: got {got}, want exactly {want} (f32 ==)"
        );
    }
}

// ---- §9.5 mse_value_w -------------------------------------------------------------

#[test]
fn glue_mse_value_w_matches_hand_reference() {
    // Spec §5.4: N=2, M=4; out = [0.75, 4.5]; host plain sum = 5.25.
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let pred = [1.0f32, 2.0, 3.0, 0.0, 0.0, -1.0, 2.0, 2.0];
    let tgt = [0.0f32, 2.0, 5.0, -1.0, 1.0, -1.0, 0.0, 4.0];
    let w = [0.5f32, 2.0];

    let out = k_mse_value_w(&gpu, &pred, &tgt, &w, 4);
    assert_close(&out, &[0.75, 4.5], 1e-6, "mse_value_w out");

    let host_sum: f32 = out.iter().sum();
    assert!(
        (host_sum - 5.25).abs() <= 1e-6,
        "mse_value_w host sum: got {host_sum}, want 5.25"
    );
}

// ---- §9.6 mse_grad_w ---------------------------------------------------------------

#[test]
fn glue_mse_grad_w_matches_hand_reference() {
    // Spec §5.5: same inputs as §5.4, scale = 0.5 (deliberately not 1.0 so
    // the test proves `scale` is read from params via bitcast<f32>).
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let pred = [1.0f32, 2.0, 3.0, 0.0, 0.0, -1.0, 2.0, 2.0];
    let tgt = [0.0f32, 2.0, 5.0, -1.0, 1.0, -1.0, 0.0, 4.0];
    let w = [0.5f32, 2.0];

    let dpred = k_mse_grad_w(&gpu, &pred, &tgt, &w, 4, 0.5);
    let want = [
        0.125f32, 0.0, -0.25, 0.125, // row0: w=0.5, factor 0.125
        -0.5, 0.0, 1.0, -1.0, // row1: w=2.0, factor 0.5
    ];
    assert_close(&dpred, &want, 1e-6, "mse_grad_w dpred");
}

// ---- §9.7 FD gradcheck entry: mse_grad_w vs mse_value_w (§6.5) --------------------------

/// Scalar loss L(pred) = scale * Σ_n out[n] via mse_value_w + host plain sum.
fn loss_w(gpu: &Gpu, pred: &[f32], tgt: &[f32], w: &[f32], m: usize, scale: f32) -> f32 {
    scale * k_mse_value_w(gpu, pred, tgt, w, m).iter().sum::<f32>()
}

#[test]
fn glue_fd_mse_grad_w() {
    // Spec §9.7: N=3, M=5, LCG pred/tgt in [-1,1], w in [0.25, 2],
    // scale = 0.7. Global gradcheck tolerances (playbook §3): h = 5e-3,
    // atol = 4e-3, rtol = 8e-2 — NEVER loosened.
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, m) = (3usize, 5usize);
    let total = n * m;
    let scale = 0.7f32;
    let mut st = Lcg::new(0xF00D_5EEDu64);
    let pred = st.vec(total);
    let tgt = st.vec(total);
    // w in [0.25, 2): positive, bounded away from 0.
    let w: Vec<f32> = (0..n).map(|_| 1.125 + 0.875 * st.signed()).collect();

    // GUARD: a zero-stub forward must NOT let the FD check pass trivially.
    let l0 = loss_w(&gpu, &pred, &tgt, &w, m, scale);
    assert!(
        l0.abs() > 1e-3,
        "FD guard: unperturbed loss |{l0}| <= 1e-3 — degenerate problem or zero-stub forward"
    );

    let analytic = k_mse_grad_w(&gpu, &pred, &tgt, &w, m, scale);
    assert_eq!(analytic.len(), total);

    let h = 5e-3f32;
    let (atol, rtol) = (4e-3f32, 8e-2f32);
    for dir in 0..2 {
        // LCG direction over ALL 15 entries.
        let v = st.vec(total);
        let a = dot(&analytic, &v);

        let plus: Vec<f32> = pred.iter().zip(v.iter()).map(|(&p, &vi)| p + h * vi).collect();
        let minus: Vec<f32> = pred.iter().zip(v.iter()).map(|(&p, &vi)| p - h * vi).collect();
        let lp = loss_w(&gpu, &plus, &tgt, &w, m, scale);
        let lm = loss_w(&gpu, &minus, &tgt, &w, m, scale);
        let num = (lp - lm) / (2.0 * h);

        let tol = atol + rtol * a.abs().max(num.abs());
        println!("glue FD dir {dir}: analytic {a:.6e}, numeric {num:.6e}, tol {tol:.3e}");
        assert!(
            (a - num).abs() <= tol,
            "mse_grad_w FD dir {dir}: |{a} - {num}| = {} > {tol}",
            (a - num).abs()
        );
    }
}

// ---- §9.8 pad2d -----------------------------------------------------------------------

#[test]
fn glue_pad2d_matches_hand_reference() {
    // Spec §5.6: NC=1, h=2, w=2, (l,r,t,b) = (1,0,0,1) => 3x3 output.
    // Everything is either literal 0.0 or a copied input => bitwise.
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let y = k_pad2d(&gpu, &x, 2, 2, (1, 0, 0, 1));
    let want = [0.0f32, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0];
    assert_bits(&y, &want, "pad2d y");
}

// ---- §9.9 crop2d ----------------------------------------------------------------------

#[test]
fn glue_crop2d_matches_hand_reference() {
    // Spec §5.7: same offsets (1,0,0,1); input is the padded 3x3.
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let xp = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let y = k_crop2d(&gpu, &xp, 2, 2, (1, 0, 0, 1));
    assert_bits(&y, &[2.0, 3.0, 5.0, 6.0], "crop2d y");

    // Adjointness hand-check: <pad2d(x), x'> == <x, crop2d(x')> == 47.
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let px = k_pad2d(&gpu, &x, 2, 2, (1, 0, 0, 1));
    let lhs = dot(&px, &xp);
    let rhs = dot(&x, &y);
    assert!(lhs == 47.0, "adjoint hand-check lhs: got {lhs}, want 47");
    assert!(rhs == 47.0, "adjoint hand-check rhs: got {rhs}, want 47");
}

// ---- §9.10 pad/crop adjointness + identities (§6.1, §6.4) --------------------------------

#[test]
fn glue_pad_crop_adjoint() {
    // Spec §6.1: NC=2, h=3, w=2; offsets {(1,2,0,1), (0,0,0,0), (2,0,3,0)};
    // random tiny tensors (LCG, [-1,1]); adjoint gap <= 1e-5 with host f32
    // dot in ascending index order on both sides.
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (nc, h, w) = (2usize, 3usize, 2usize);
    let mut st = Lcg::new(0xADD_017);

    for &(l, r, t, b) in &[(1usize, 2usize, 0usize, 1usize), (0, 0, 0, 0), (2, 0, 3, 0)] {
        let (hp, wp) = (h + t + b, w + l + r);
        let x = st.vec(nc * h * w); // unpadded shape
        let y = st.vec(nc * hp * wp); // padded shape

        let px = k_pad2d(&gpu, &x, h, w, (l, r, t, b));
        let cy = k_crop2d(&gpu, &y, h, w, (l, r, t, b));
        assert_eq!(px.len(), y.len());
        assert_eq!(cy.len(), x.len());

        let gap = (dot(&px, &y) - dot(&x, &cy)).abs();
        assert!(
            gap <= 1e-5,
            "pad/crop adjoint gap {gap} > 1e-5 for offsets ({l},{r},{t},{b})"
        );

        if (l, r, t, b) == (0, 0, 0, 0) {
            // Zero offsets: both kernels are bitwise identity copies (§7).
            assert_bits(&px, &x, "pad2d zero-offset identity");
            assert_bits(&cy, &y, "crop2d zero-offset identity");
        }
    }

    // §6.4: crop2d(pad2d(x)) bitwise == x for asymmetric (1,2,0,1).
    let x = st.vec(nc * h * w);
    let px = k_pad2d(&gpu, &x, h, w, (1, 2, 0, 1));
    let back = k_crop2d(&gpu, &px, h, w, (1, 2, 0, 1));
    assert_bits(&back, &x, "crop2d(pad2d(x))");
}

// ---- §9.11 nchw_nlc --------------------------------------------------------------------

#[test]
fn glue_nchw_nlc_matches_hand_reference() {
    // Spec §5.8: N=1, C=2, hw=4.
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // NCHW: c0 | c1
    let nlc = k_nchw_nlc(&gpu, &x, 2, 4);
    assert_bits(&nlc, &[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], "nchw_nlc");

    // Roundtrip bitwise.
    let back = k_nlc_nchw(&gpu, &nlc, 2, 4);
    assert_bits(&back, &x, "nlc_nchw(nchw_nlc(x))");

    // Adjointness hand-check: both sides == 1900.
    let y = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]; // NLC
    let ty = k_nlc_nchw(&gpu, &y, 2, 4);
    assert_bits(&ty, &[10.0, 30.0, 50.0, 70.0, 20.0, 40.0, 60.0, 80.0], "nlc_nchw(y)");
    let lhs = dot(&nlc, &y);
    let rhs = dot(&x, &ty);
    assert!(lhs == 1900.0, "nchw/nlc adjoint lhs: got {lhs}, want 1900");
    assert!(rhs == 1900.0, "nchw/nlc adjoint rhs: got {rhs}, want 1900");
}

// ---- §9.12 NCHW<->NLC roundtrip + adjointness on random data (§6.2) ------------------------

#[test]
fn glue_nlc_roundtrip_and_adjoint() {
    // Spec §6.2: random tiny N=2, C=3, hw=4.
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, c, hw) = (2usize, 3usize, 4usize);
    let total = n * c * hw;
    let mut st = Lcg::new(0x00DD_BA11_u64);
    let x = st.vec(total); // NCHW
    let y = st.vec(total); // NLC

    // Roundtrip BITWISE (permutations are pure copies).
    let fwd = k_nchw_nlc(&gpu, &x, c, hw);
    let back = k_nlc_nchw(&gpu, &fwd, c, hw);
    assert_bits(&back, &x, "nlc_nchw(nchw_nlc(x)) random");

    // Adjointness: <nchw_nlc(x), y> == <x, nlc_nchw(y)> within 1e-5.
    let ty = k_nlc_nchw(&gpu, &y, c, hw);
    let gap = (dot(&fwd, &y) - dot(&x, &ty)).abs();
    assert!(gap <= 1e-5, "nchw/nlc adjoint gap {gap} > 1e-5");
}

// ---- §9.13 determinism: fixed seed, twice-run, bitwise -------------------------------------

#[test]
fn glue_deterministic_bitwise() {
    // Spec §9.13 / §10: run every one of the nine kernels twice on identical
    // LCG-seeded inputs with FRESH output buffers; ALL outputs bitwise equal.
    let gpu = gpu_core::testgpu::dev(KERNELS);
    const SEED: u64 = 0xDE7E_121Eu64;

    // Regenerate identical inputs from the same fixed seed for each pass.
    let inputs = |st: &mut Lcg| {
        let mul_a = st.vec(7);
        let mul_b = st.vec(7);
        let sr_x = st.vec(10); // N=2, M=5
        let sr_s = st.vec(2);
        let em_x = st.vec(6); // N=2, M=3
        let em_f = st.vec(6);
        let em_ab = st.vec(4);
        let ms_pred = st.vec(12); // N=3, M=4
        let ms_tgt = st.vec(12);
        let ms_w: Vec<f32> = (0..3).map(|_| 1.125 + 0.875 * st.signed()).collect();
        let pad_x = st.vec(2 * 2 * 3); // NC=2, h=2, w=3
        let crop_x = st.vec(2 * 5 * 4); // padded: hp=5, wp=4 for (1,0,2,1)
        let perm_x = st.vec(2 * 3 * 4); // N=2, C=3, hw=4
        (mul_a, mul_b, sr_x, sr_s, em_x, em_f, em_ab, ms_pred, ms_tgt, ms_w, pad_x, crop_x, perm_x)
    };

    let pass = || {
        let mut st = Lcg::new(SEED);
        let (mul_a, mul_b, sr_x, sr_s, em_x, em_f, em_ab, ms_pred, ms_tgt, ms_w, pad_x, crop_x, perm_x) =
            inputs(&mut st);
        vec![
            k_mul(&gpu, &mul_a, &mul_b),
            k_scale_row(&gpu, &sr_x, &sr_s, 5),
            k_edm_mix(&gpu, &em_x, &em_f, &em_ab, 3),
            k_mse_value_w(&gpu, &ms_pred, &ms_tgt, &ms_w, 4),
            k_mse_grad_w(&gpu, &ms_pred, &ms_tgt, &ms_w, 4, 0.7),
            k_pad2d(&gpu, &pad_x, 2, 3, (1, 0, 2, 1)),
            k_crop2d(&gpu, &crop_x, 2, 3, (1, 0, 2, 1)),
            k_nchw_nlc(&gpu, &perm_x, 3, 4),
            k_nlc_nchw(&gpu, &perm_x, 3, 4),
        ]
    };

    let first = pass();
    let second = pass();
    let names = [
        "mul", "scale_row", "edm_mix", "mse_value_w", "mse_grad_w", "pad2d", "crop2d",
        "nchw_nlc", "nlc_nchw",
    ];
    for ((a, b), name) in first.iter().zip(second.iter()).zip(names.iter()) {
        assert_bits(a, b, &format!("determinism {name}"));
    }
}

// ---- edge paths (spec §7) -------------------------------------------------------------------

#[test]
fn glue_zero_weight_sample_contributes_zero() {
    // Spec §7: w[n] = 0 => the sample contributes EXACTLY 0.0 to both the
    // loss partial sum and the gradient (finite inputs by contract).
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (m, scale) = (3usize, 0.7f32);
    let mut st = Lcg::new(0x00C0_FFEE); // fixed seed
    let pred = st.vec(2 * m);
    let tgt = st.vec(2 * m);
    let w = [0.0f32, 1.5];

    let out = k_mse_value_w(&gpu, &pred, &tgt, &w, m);
    assert!(out[0] == 0.0, "mse_value_w w=0 sample: got {}, want exactly 0.0", out[0]);
    // Non-zero-weight row must actually contribute (guards a zero stub).
    let d1: f32 = (0..m).map(|j| (pred[m + j] - tgt[m + j]).powi(2)).sum::<f32>() / m as f32;
    assert!(
        (out[1] - 1.5 * d1).abs() <= 1e-6,
        "mse_value_w w=1.5 sample: got {}, want {}",
        out[1],
        1.5 * d1
    );

    let dpred = k_mse_grad_w(&gpu, &pred, &tgt, &w, m, scale);
    for (i, &g) in dpred[..m].iter().enumerate() {
        assert!(g == 0.0, "mse_grad_w w=0 row [{i}]: got {g}, want exactly 0.0");
    }
    for (j, &g) in dpred[m..].iter().enumerate() {
        let want = 1.5 * 2.0 * (pred[m + j] - tgt[m + j]) / m as f32 * scale;
        assert!(
            (g - want).abs() <= 1e-6,
            "mse_grad_w w=1.5 row [{j}]: got {g}, want {want}"
        );
    }
}

#[test]
fn glue_m1_per_element_rows() {
    // Spec §7: M = 1 => n = i, per-element weights; must work unchanged.
    let gpu = gpu_core::testgpu::dev(KERNELS);

    // scale_row with M=1 degenerates to elementwise product s[i]*x[i].
    let x = [1.0f32, -2.0, 0.5, 4.0];
    let s = [3.0f32, 0.5, -2.0, 0.25];
    let y = k_scale_row(&gpu, &x, &s, 1);
    assert_close(&y, &[3.0, -1.0, -1.0, 1.0], 1e-6, "scale_row M=1");

    // edm_mix with M=1: y[i] = a[i]*x[i] + b[i]*f[i].
    let ab = [2.0f32, 1.0, -1.0, 0.5]; // a=[2,-1], b=[1,0.5]
    let ex = [1.0f32, 3.0];
    let ef = [4.0f32, 2.0];
    let ey = k_edm_mix(&gpu, &ex, &ef, &ab, 1);
    assert_close(&ey, &[6.0, -2.0], 1e-6, "edm_mix M=1"); // [2*1+1*4, -1*3+0.5*2]

    // mse pair with M=1: out[n] = w[n]*d_n^2, dpred[n] = w[n]*2*d_n*scale.
    let pred = [1.0f32, -0.5, 2.0];
    let tgt = [0.5f32, -0.5, 3.0];
    let w = [2.0f32, 4.0, 0.5];
    let out = k_mse_value_w(&gpu, &pred, &tgt, &w, 1);
    assert_close(&out, &[0.5, 0.0, 0.5], 1e-6, "mse_value_w M=1");
    let dpred = k_mse_grad_w(&gpu, &pred, &tgt, &w, 1, 0.5);
    assert_close(&dpred, &[1.0, 0.0, -0.5], 1e-6, "mse_grad_w M=1");
}
