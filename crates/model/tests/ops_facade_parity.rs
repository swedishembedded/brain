// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::ops::{Ops, Weight}` façade parity (B3).
//!
//! For each weight tier the façade supports (`F32`, `I8`, `Q4`) and a
//! representative `m` sweep spanning the decode regime, the GEMV/tile
//! crossover, and the register-tiled regime, `Ops::matmul` must reproduce
//! EXACTLY what today's hand-written dispatch already produces on the SAME
//! input - driven by hand with `model::dispatch::mm_rows_off`/`mm8_rows_off`
//! plus `model::int8::quantize_weight`/`model::int4::quantize_weight_q4`, the
//! same helpers `crates/flux1`/`crates/flux2`/`crates/qwen3` call today. Not
//! "close" - bit for bit, since every kernel variant these dispatch to is
//! already documented (and measured) as bit-identical to the naive reference.
//!
//! Also exercises a non-trivial row offset for both `I8` and `Q4` weights -
//! `model::int8`'s own module doc names this exact bug class: a wrong packed
//! offset divisor is silently-wrong arithmetic, not a crash, so a tolerance
//! check alone could pass on a subtly wrong result. The offset test instead
//! asserts the offset dispatch is IDENTICAL to a zero-offset dispatch over the
//! identical slice of data - any divisor mistake (e.g. dividing by `Q4`'s own
//! `per_word()` = 8 instead of the activation's real int8 `per_word()` = 4)
//! reads the wrong byte range and cannot coincidentally reproduce that.
//!
//! This is also `Weight::Q4`'s first real MODEL-FACING dispatch exercise
//! outside `crates/model/tests/matmul_q4_gemm.rs` (which only exercises the
//! raw kernel, not a `Weight`/`Ops`-shaped call) - a real milestone: Q4
//! weight dispatch driven through a model-facing call for the first time.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::block::GemmVariants;
use model::dispatch::{mm4_rows_off, mm8_rows_off, mm_rows_off, I8Scratch};
use model::int4::quantize_weight_q4;
use model::int8::quantize_weight;
use model::ops::{Ops, Weight};

/// The full façade kernel set, including the three bf16-storage variants
/// (B4) and three f16-storage variants (B5) - `Ops::new` requires all of them
/// even for tests that only exercise `F32`/`I8`/`Q4` here
/// (`crates/model/tests/bf16_roundtrip.rs`/`f16_roundtrip.rs` are what
/// actually exercise those tiers' own numerics). `gpu_core::testgpu::dev`
/// keys its device pool by the kernel slice's pointer identity, so this
/// leaks the `Vec` once (via `OnceLock`) rather than reallocating a fresh one
/// per call - the same tradeoff `model::ops::tests::kernel_list` makes.
fn kernel_list() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let bf16_matmul = kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", Dtype::BF16).unwrap();
        let bf16_gemv =
            kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::BF16).unwrap();
        let bf16_reg3 =
            kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::BF16).unwrap();
        let f16_matmul = kernels::template::dtype_variant("matmul", kernels::MATMUL, "w", Dtype::F16).unwrap();
        let f16_gemv =
            kernels::template::dtype_variant("matmul_gemv", kernels::MATMUL_GEMV, "w", Dtype::F16).unwrap();
        let f16_reg3 =
            kernels::template::dtype_variant("matmul_reg3", kernels::MATMUL_REG3, "w", Dtype::F16).unwrap();
        vec![
            ("matmul", kernels::MATMUL),
            ("matmul_gemv", kernels::MATMUL_GEMV),
            ("matmul_reg2", kernels::MATMUL_REG2),
            ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
            ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
            ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
            ("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV),
            ("max_abs_row", kernels::MAX_ABS_ROW),
            ("quant_pack", kernels::QUANT_PACK),
            bf16_matmul,
            bf16_gemv,
            bf16_reg3,
            f16_matmul,
            f16_gemv,
            f16_reg3,
        ]
    })
}

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn host_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[r * k + i] * w[j * k + i];
            }
            out[r * n + j] = acc;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    d / (na.sqrt() * nb.sqrt())
}

/// `Ops::matmul` for an `F32` weight must be bit-identical to
/// `dispatch::mm_rows_off` driven by hand on the same `x`/weight buffers.
fn check_f32(m: usize, n: usize, k: usize) {
    let gpu = gpu_core::testgpu::dev(kernel_list());
    let ops = Ops::new(gpu).expect("Ops::new");
    let g = ops.gpu();

    let mut rng = Lcg::new(0xF32_0000 ^ m as u64);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let w_h = rng.vec_scaled(n * k, 1.0);
    let x = g.storage_init("x", &x_h);
    let weight = Weight::upload(&ops, &w_h, n, k, Dtype::F32);
    assert_eq!(weight.dtype(), Dtype::F32);

    let mut got_steps = Vec::new();
    let act = ops.act(&mut got_steps, &x, 0, m as u32, k as u32);
    let out_got = g.storage((m * n) as u64);
    ops.matmul(&mut got_steps, &weight, &act, &out_got, 0);
    g.submit(&[], &got_steps);
    let got = g.read(&out_got, m * n);

    // Oracle: today's `dispatch::mm_rows_off`, hand-driven on the SAME `x`
    // and an independently-uploaded copy of the SAME weight bytes.
    let wbuf = g.storage_init("w_oracle", &w_h);
    let gemv = idx(g, "matmul_gemv");
    let tiled = idx(g, "matmul_reg2");
    let out_want = g.storage((m * n) as u64);
    let want_step =
        mm_rows_off(g, GemmVariants::Fast { gemv: Some(gemv), tiled }, &x, &wbuf, &out_want, 0, 0, m as u32, k as u32, n as u32);
    g.submit(&[], &[want_step]);
    let want = g.read(&out_want, m * n);

    assert_eq!(got, want, "F32 tier m={m} n={n} k={k}: facade output != dispatch.rs oracle (bit-for-bit)");
}

/// `Ops::matmul` for an `I8` weight must be bit-identical to
/// `model::int8::quantize_weight` + `dispatch::I8Scratch` + `mm8_rows_off`
/// driven by hand on the same `x`/weight buffers.
fn check_i8(m: usize, n: usize, k: usize) {
    let gpu = gpu_core::testgpu::dev(kernel_list());
    let ops = Ops::new(gpu).expect("Ops::new");
    let g = ops.gpu();

    let mut rng = Lcg::new(0x18_0000 ^ m as u64);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let w_h = rng.vec_scaled(n * k, 1.0);
    let x = g.storage_init("x", &x_h);
    let weight = Weight::upload(&ops, &w_h, n, k, Dtype::I8);
    assert_eq!(weight.dtype(), Dtype::I8, "this device must support int8_dot for this test to exercise the I8 tier");

    let mut got_steps = Vec::new();
    let act = ops.act(&mut got_steps, &x, 0, m as u32, k as u32);
    let out_got = g.storage((m * n) as u64);
    ops.matmul(&mut got_steps, &weight, &act, &out_got, 0);
    g.submit(&[], &got_steps);
    let got = g.read(&out_got, m * n);

    // Oracle: today's quantize+dispatch path, hand-driven.
    let (max_abs_row, quant_pack) = (idx(g, "max_abs_row"), idx(g, "quant_pack"));
    let gemv = idx(g, "matmul_i8_gemv");
    let dyn_ = idx(g, "matmul_i8_dyn");
    let scr = I8Scratch::new(g, m as u64, m as u64, &[k as u32]);
    let mut want_steps = Vec::new();
    scr.quant_rows(g, [max_abs_row, quant_pack], &mut want_steps, &x, 0, m as u32, k as u32);
    let (wq, sw) = quantize_weight(&w_h, n, k);
    let wqb = g.storage(wq.len() as u64);
    g.write(&wqb, &wq);
    let swb = g.storage_init("sw", &sw);
    let out_want = g.storage((m * n) as u64);
    want_steps.push(mm8_rows_off(
        g,
        GemmVariants::Fast { gemv: Some(gemv), tiled: dyn_ },
        &scr,
        &wqb,
        &swb,
        &out_want,
        0,
        0,
        m as u32,
        k as u32,
        n as u32,
    ));
    g.submit(&[], &want_steps);
    let want = g.read(&out_want, m * n);

    assert_eq!(got, want, "I8 tier m={m} n={n} k={k}: facade output != dispatch.rs oracle (bit-for-bit)");
}

/// `Ops::matmul` for a `Q4` weight must be bit-identical to
/// `model::int4::quantize_weight_q4` + `dispatch::I8Scratch` (the SAME int8
/// activation quantizer - q4 is W4A8) + `mm8_rows_off` driven by hand.
fn check_q4(m: usize, n: usize, k: usize) {
    let gpu = gpu_core::testgpu::dev(kernel_list());
    let ops = Ops::new(gpu).expect("Ops::new");
    let g = ops.gpu();

    let mut rng = Lcg::new(0x04_0000 ^ m as u64);
    let x_h = rng.vec_scaled(m * k, 1.0);
    let w_h = rng.vec_scaled(n * k, 1.0);
    let x = g.storage_init("x", &x_h);
    let weight = Weight::upload(&ops, &w_h, n, k, Dtype::Q4);
    assert_eq!(weight.dtype(), Dtype::Q4, "this device must support int8_dot for this test to exercise the Q4 tier");

    let mut got_steps = Vec::new();
    let act = ops.act(&mut got_steps, &x, 0, m as u32, k as u32);
    let out_got = g.storage((m * n) as u64);
    ops.matmul(&mut got_steps, &weight, &act, &out_got, 0);
    g.submit(&[], &got_steps);
    let got = g.read(&out_got, m * n);

    let (max_abs_row, quant_pack) = (idx(g, "max_abs_row"), idx(g, "quant_pack"));
    let gemv = idx(g, "matmul_q4_gemv");
    let dyn_ = idx(g, "matmul_q4_dyn");
    let scr = I8Scratch::new(g, m as u64, m as u64, &[k as u32]);
    let mut want_steps = Vec::new();
    scr.quant_rows(g, [max_abs_row, quant_pack], &mut want_steps, &x, 0, m as u32, k as u32);
    let (wq, sw) = quantize_weight_q4(&w_h, n, k);
    let wqb = g.storage(wq.len() as u64);
    g.write(&wqb, &wq);
    let swb = g.storage_init("sw", &sw);
    let out_want = g.storage((m * n) as u64);
    want_steps.push(mm4_rows_off(
        g,
        GemmVariants::Fast { gemv: Some(gemv), tiled: dyn_ },
        &scr,
        &wqb,
        &swb,
        &out_want,
        0,
        0,
        m as u32,
        k as u32,
        n as u32,
    ));
    g.submit(&[], &want_steps);
    let want = g.read(&out_want, m * n);

    assert_eq!(got, want, "Q4 tier m={m} n={n} k={k}: facade output != dispatch.rs oracle (bit-for-bit)");
}

/// `m ∈ {1, 8, 64, 512}` covers: the decode regime's GEMV kernels (1, 8 - at
/// and below `I8_GEMV_MAX_ROWS`/near `DECODE_REGIME_MAX_ROWS`), and the
/// register-tiled / packed-int8 regime clear of the decode cutoff (64, 512).
/// `n=64, k=128`: `k` a multiple of 64 keeps every row offset naturally
/// 256B-aligned, and both are cheap (no real checkpoint needed).
#[test]
fn matmul_matches_dispatch_rs_bit_identically_across_tiers_and_m() {
    let (n, k) = (64usize, 128usize);
    for &m in &[1usize, 8, 64, 512] {
        check_f32(m, n, k);
        check_i8(m, n, k);
        check_q4(m, n, k);
    }
}

/// The offset-arithmetic gate: a non-trivial row offset (not row 0) for both
/// `I8` and `Q4` weights. `xr0=64` clears `quant_rows_steps`'s own 64-row
/// (256B) storage-binding alignment requirement.
#[test]
fn matmul_row_offset_is_correct_for_i8_and_q4() {
    let (n, k) = (64usize, 128usize);
    let (xr0, m) = (64u32, 8usize);
    let total_rows = xr0 as usize + m;

    for &tier in &[Dtype::I8, Dtype::Q4] {
        let gpu = gpu_core::testgpu::dev(kernel_list());
        let ops = Ops::new(gpu).expect("Ops::new");
        let g = ops.gpu();

        let seed = 0xA11_0000u64 ^ (if tier == Dtype::I8 { 8 } else { 4 });
        let mut rng = Lcg::new(seed);
        let x_h = rng.vec_scaled(total_rows * k, 1.0);
        let w_h = rng.vec_scaled(n * k, 1.0);
        let x_full: DeviceBuffer = g.storage_init("x_full", &x_h);
        let weight = Weight::upload(&ops, &w_h, n, k, tier);
        assert_eq!(weight.dtype(), tier, "this device must support int8_dot for this test to mean anything for {tier:?}");

        // The facade call under test: quantize+matmul at row offset `xr0`
        // within the larger `x_full` buffer.
        let mut off_steps = Vec::new();
        let act_off = ops.act(&mut off_steps, &x_full, xr0, m as u32, k as u32);
        let out_off = g.storage((m * n) as u64);
        ops.matmul(&mut off_steps, &weight, &act_off, &out_off, 0);
        g.submit(&[], &off_steps);
        let got_off = g.read(&out_off, m * n);

        // Reference: the SAME facade call, but on a FRESH buffer holding
        // ONLY the sliced rows at offset 0 -- identical data, zero offset.
        // If `Ops::matmul`'s internal offset divisor were wrong (e.g. `Q4`'s
        // own `per_word()`=8 instead of the activation's real int8
        // `per_word()`=4), this reads the WRONG byte range at `xr0` and
        // cannot coincidentally reproduce this zero-offset dispatch on the
        // identical slice.
        let x_slice = &x_h[xr0 as usize * k..(xr0 as usize + m) * k];
        let x0 = g.storage_init("x0", x_slice);
        let mut zero_steps = Vec::new();
        let act0 = ops.act(&mut zero_steps, &x0, 0, m as u32, k as u32);
        let out_zero = g.storage((m * n) as u64);
        ops.matmul(&mut zero_steps, &weight, &act0, &out_zero, 0);
        g.submit(&[], &zero_steps);
        let want_exact = g.read(&out_zero, m * n);

        assert_eq!(got_off, want_exact, "{tier:?} row offset xr0={xr0} m={m}: offset dispatch != zero-offset dispatch on the identical slice -- offset arithmetic is wrong");

        // Belt-and-suspenders sanity check against an independent fp32 host
        // oracle over the same slice (real quantization rounding, so a
        // similarity bound rather than exact equality -- this test's own
        // job is the OFFSET, the tier x m sweep above already re-proves
        // quantization accuracy bit-for-bit against dispatch.rs).
        let want_fp32 = host_matmul(x_slice, &w_h, m, k, n);
        let cos = cosine(&got_off, &want_fp32);
        assert!(cos >= 0.98, "{tier:?} row offset xr0={xr0} m={m}: cosine {cos:.6} vs fp32 host oracle too low");
    }
}
