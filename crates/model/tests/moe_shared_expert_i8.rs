// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::moe::shared_expert_fwd_i8` vs. the fp32 `shared_expert_fwd`
//! (`crates/model/tests/moe_shared_expert.rs` already proved that one exact
//! against a host oracle, so that is the reference here — same shape as
//! `moe_sparse_i8_parity.rs`'s own "fp32 sparse path is the oracle for the
//! int8 one" discipline).
//!
//! Tolerance follows `model::int8::quantize_weight`'s own round-trip test:
//! per-element quantization error is bounded by `~scale/2`, propagated
//! through three chained linears (gate/up/down) plus one activation-side
//! quantization of the SwiGLU output — the same error budget
//! `moe_sparse_i8_parity.rs` already measured and gated at 0.02 for the
//! routed-expert case. A relative-L2 bound is the honest way to state that
//! for a whole tensor rather than picking an absolute epsilon that only
//! holds at one magnitude.

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::int8::{quant_rows_steps, quantize_weight, QuantRows};
use model::moe::{shared_expert_fwd, shared_expert_fwd_i8, Lin8, SharedExpertIds, SharedExpertIds8, SharedExpertScratch, SharedExpertScratch8};

/// Upload a packed-int8 `[n, k/4]` `u32` weight buffer -- `storage_init` is
/// f32-only, so a packed weight goes through the raw `storage`+`write` pair
/// instead, matching `moe_sparse_i8_parity.rs`'s own precedent.
fn upload_u32(g: &Gpu, data: &[u32]) -> DeviceBuffer {
    let b = g.storage(data.len() as u64);
    g.write(&b, data);
    b
}

const PIPES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("silu_mul", kernels::SILU_MUL),
    ("sigmoid", kernels::SIGMOID),
    ("scale_row", kernels::SCALE_ROW),
    ("add2", kernels::ADD2),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&a, &b) in got.iter().zip(want) {
        num += ((a - b) as f64).powi(2);
        den += (b as f64).powi(2);
    }
    (num / den.max(1e-12)).sqrt()
}

/// `shared_gate_w: Some` — Qwen3-Omni's Talker shape (the one this test
/// exists to validate before real weights, verified standalone against a
/// host-computed oracle first, the same precedent already established for
/// this component's fp32 primitive).
#[test]
fn shared_expert_i8_matches_fp32_within_quant_tolerance() {
    let g = gpu_core::testgpu::dev(PIPES);
    let ids = SharedExpertIds { matmul: idx(&g, "matmul"), silu_mul: idx(&g, "silu_mul"), sigmoid: idx(&g, "sigmoid"), scale_row: idx(&g, "scale_row"), add2: idx(&g, "add2") };
    let ids8 = SharedExpertIds8 {
        matmul_i8: idx(&g, "matmul_i8_dyn"),
        matmul: idx(&g, "matmul"),
        silu_mul: idx(&g, "silu_mul"),
        sigmoid: idx(&g, "sigmoid"),
        scale_row: idx(&g, "scale_row"),
        add2: idx(&g, "add2"),
        quant: [idx(&g, "max_abs_row"), idx(&g, "quant_pack")],
    };

    // d_model and shared_ff must be multiples of 4 (int8 packing).
    // `d` and `ff` are each some linear's contraction width, so both must be
    // whole multiples of `model::int8::GROUP` (32).
    let (rows, d, ff) = (6usize, 64usize, 32usize);
    let mut rng = Lcg::new(0x5EED);
    let x_h = rng.vec_scaled(rows * d, 1.0);
    let gw_h = rng.vec_scaled(ff * d, 0.5);
    let uw_h = rng.vec_scaled(ff * d, 0.5);
    let dw_h = rng.vec_scaled(d * ff, 0.5);
    let sgw_h = rng.vec_scaled(d, 0.5);
    let acc_h = rng.vec_scaled(rows * d, 1.0);

    // --- fp32 oracle ---
    let x = g.storage_init("x", &x_h);
    let gw = g.storage_init("gw", &gw_h);
    let uw = g.storage_init("uw", &uw_h);
    let dw = g.storage_init("dw", &dw_h);
    let sgw = g.storage_init("sgw", &sgw_h);
    let acc = g.storage_init("acc", &acc_h);
    let scratch = SharedExpertScratch {
        gate_pre: &g.storage((rows * ff) as u64),
        up: &g.storage((rows * ff) as u64),
        h: &g.storage((rows * ff) as u64),
        mlp_out: &g.storage((rows * d) as u64),
        gate_logits: &g.storage(rows as u64),
        gate_scalar: &g.storage(rows as u64),
        scaled: &g.storage((rows * d) as u64),
    };
    let out = g.storage((rows * d) as u64);
    let steps = shared_expert_fwd(&g, &ids, rows as u32, d as u32, ff as u32, &x, &gw, &uw, &dw, Some(&sgw), &scratch, &acc, &out);
    g.submit(&[], &steps);
    let fp32 = g.read(&out, rows * d).to_vec();
    assert!(fp32.iter().any(|&v| v.abs() > 1e-9), "fp32 reference is all-zero — the test shape is degenerate");

    // --- int8 path: quantize gate/up/down weights, quantize x once, dispatch ---
    let (gwq, gws) = quantize_weight(&gw_h, ff, d);
    let (uwq, uws) = quantize_weight(&uw_h, ff, d);
    let (dwq, dws) = quantize_weight(&dw_h, d, ff);
    let gw8 = Lin8 { wq: &upload_u32(&g, &gwq), sw: &g.storage_init("gws", &gws) };
    let uw8 = Lin8 { wq: &upload_u32(&g, &uwq), sw: &g.storage_init("uws", &uws) };
    let dw8 = Lin8 { wq: &upload_u32(&g, &dwq), sw: &g.storage_init("dws", &dws) };

    let xq = g.storage((rows * d / 4) as u64);
    let sx = g.storage(rows as u64);
    let mut steps8 = quant_rows_steps(&g, QuantRows { kernels: ids8.quant, x: &x, sx: &sx, xq: &xq }, 0, rows as u32, d as u32).to_vec();

    let scratch8 = SharedExpertScratch8 {
        gate_pre: &g.storage((rows * ff) as u64),
        up: &g.storage((rows * ff) as u64),
        h: &g.storage((rows * ff) as u64),
        hq: &g.storage((rows * ff / 4) as u64),
        sh: &g.storage(rows as u64),
        mlp_out: &g.storage((rows * d) as u64),
        gate_logits: &g.storage(rows as u64),
        gate_scalar: &g.storage(rows as u64),
        scaled: &g.storage((rows * d) as u64),
    };
    let out8 = g.storage((rows * d) as u64);
    steps8.extend(shared_expert_fwd_i8(&g, &ids8, rows as u32, d as u32, ff as u32, &xq, &sx, &x, gw8, uw8, dw8, Some(&sgw), &scratch8, &acc, &out8));
    g.submit(&[], &steps8);
    let i8out = g.read(&out8, rows * d).to_vec();

    let err = rel_l2(&i8out, &fp32);
    eprintln!("shared_expert_fwd_i8 (gated) rel_l2: {err:.6}");
    // Measured 0.0014 on this shape/seed; 0.02 leaves headroom for RNG/shape
    // drift without hiding an actual quantization regression, matching
    // moe_sparse_i8_parity.rs's own threshold for the routed-expert case
    // (measured 0.0084 there).
    assert!(err < 0.02, "shared_expert_fwd_i8 vs fp32 rel_l2 {err} exceeds tolerance");
}

/// `shared_gate_w: None` — the unweighted architecture (matches
/// `shared_expert_fwd`'s own second test) — no current caller needs this
/// int8 shape yet, but the code path exists and must not silently diverge.
#[test]
fn shared_expert_i8_unweighted_matches_fp32_within_quant_tolerance() {
    let g = gpu_core::testgpu::dev(PIPES);
    let ids = SharedExpertIds { matmul: idx(&g, "matmul"), silu_mul: idx(&g, "silu_mul"), sigmoid: idx(&g, "sigmoid"), scale_row: idx(&g, "scale_row"), add2: idx(&g, "add2") };
    let ids8 = SharedExpertIds8 {
        matmul_i8: idx(&g, "matmul_i8_dyn"),
        matmul: idx(&g, "matmul"),
        silu_mul: idx(&g, "silu_mul"),
        sigmoid: idx(&g, "sigmoid"),
        scale_row: idx(&g, "scale_row"),
        add2: idx(&g, "add2"),
        quant: [idx(&g, "max_abs_row"), idx(&g, "quant_pack")],
    };

    // `d` and `ff` are each some linear's contraction width, so both must be
    // whole multiples of `model::int8::GROUP` (32).
    let (rows, d, ff) = (6usize, 64usize, 32usize);
    let mut rng = Lcg::new(0x1234);
    let x_h = rng.vec_scaled(rows * d, 1.0);
    let gw_h = rng.vec_scaled(ff * d, 0.5);
    let uw_h = rng.vec_scaled(ff * d, 0.5);
    let dw_h = rng.vec_scaled(d * ff, 0.5);
    let acc_h = rng.vec_scaled(rows * d, 1.0);

    let x = g.storage_init("x", &x_h);
    let gw = g.storage_init("gw", &gw_h);
    let uw = g.storage_init("uw", &uw_h);
    let dw = g.storage_init("dw", &dw_h);
    let acc = g.storage_init("acc", &acc_h);
    let scratch = SharedExpertScratch {
        gate_pre: &g.storage((rows * ff) as u64),
        up: &g.storage((rows * ff) as u64),
        h: &g.storage((rows * ff) as u64),
        mlp_out: &g.storage((rows * d) as u64),
        gate_logits: &g.storage(rows as u64),
        gate_scalar: &g.storage(rows as u64),
        scaled: &g.storage((rows * d) as u64),
    };
    let out = g.storage((rows * d) as u64);
    let steps = shared_expert_fwd(&g, &ids, rows as u32, d as u32, ff as u32, &x, &gw, &uw, &dw, None, &scratch, &acc, &out);
    g.submit(&[], &steps);
    let fp32 = g.read(&out, rows * d).to_vec();
    assert!(fp32.iter().any(|&v| v.abs() > 1e-9), "fp32 reference is all-zero — the test shape is degenerate");

    let (gwq, gws) = quantize_weight(&gw_h, ff, d);
    let (uwq, uws) = quantize_weight(&uw_h, ff, d);
    let (dwq, dws) = quantize_weight(&dw_h, d, ff);
    let gw8 = Lin8 { wq: &upload_u32(&g, &gwq), sw: &g.storage_init("gws", &gws) };
    let uw8 = Lin8 { wq: &upload_u32(&g, &uwq), sw: &g.storage_init("uws", &uws) };
    let dw8 = Lin8 { wq: &upload_u32(&g, &dwq), sw: &g.storage_init("dws", &dws) };

    let xq = g.storage((rows * d / 4) as u64);
    let sx = g.storage(rows as u64);
    let mut steps8 = quant_rows_steps(&g, QuantRows { kernels: ids8.quant, x: &x, sx: &sx, xq: &xq }, 0, rows as u32, d as u32).to_vec();

    let scratch8 = SharedExpertScratch8 {
        gate_pre: &g.storage((rows * ff) as u64),
        up: &g.storage((rows * ff) as u64),
        h: &g.storage((rows * ff) as u64),
        hq: &g.storage((rows * ff / 4) as u64),
        sh: &g.storage(rows as u64),
        mlp_out: &g.storage((rows * d) as u64),
        gate_logits: &g.storage(rows as u64),
        gate_scalar: &g.storage(rows as u64),
        scaled: &g.storage((rows * d) as u64),
    };
    let out8 = g.storage((rows * d) as u64);
    steps8.extend(shared_expert_fwd_i8(&g, &ids8, rows as u32, d as u32, ff as u32, &xq, &sx, &x, gw8, uw8, dw8, None, &scratch8, &acc, &out8));
    g.submit(&[], &steps8);
    let i8out = g.read(&out8, rows * d).to_vec();

    let err = rel_l2(&i8out, &fp32);
    eprintln!("shared_expert_fwd_i8 (unweighted) rel_l2: {err:.6}");
    // Measured 0.0038 on this shape/seed; see the gated test above for the
    // tolerance rationale.
    assert!(err < 0.02, "shared_expert_fwd_i8 (unweighted) vs fp32 rel_l2 {err} exceeds tolerance");
}
