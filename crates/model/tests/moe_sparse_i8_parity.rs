// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::moe::expert_fwd_i8` vs. the fp32 sparse path it quantizes
//! (`crates/model/tests/moe_sparse_parity.rs` already proved the fp32 path
//! exact against glm's dense oracle, so that is the reference here).
//!
//! Tolerance follows `model::int8::quantize_weight`'s own round-trip test:
//! per-element quantization error is bounded by `~scale/2`, propagated through
//! three chained linears (gate/up/down) plus one activation-side quantization
//! of the SwiGLU output. A relative-L2 bound is the honest way to state that
//! for a whole tensor rather than picking an absolute epsilon that only holds
//! at one magnitude.

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::int8::{quant_rows_steps, quantize_weight, QuantRows};
use model::moe::{expert_fwd, expert_fwd_i8, router_fwd, ExpertScratch, ExpertScratch8, Lin8, MoeIds, MoeIds8, MoeShape};

const PIPES: &[(&str, &str)] = &[
    ("router_gate", kernels::ROUTER_GATE),
    ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
    ("moe_linear_gated_i8", kernels::MOE_LINEAR_GATED_I8),
    ("silu_mul", kernels::SILU_MUL),
    ("scale_add", kernels::SCALE_ADD),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

#[test]
fn int8_matches_fp32_sparse_within_quant_tolerance() {
    let g = gpu_core::testgpu::dev(PIPES);
    let ids = MoeIds {
        router_gate: idx(&g, "router_gate"),
        linear_gated: idx(&g, "moe_linear_gated"),
        silu_mul: idx(&g, "silu_mul"),
        scale_add: idx(&g, "scale_add"),
    };
    let ids8 = MoeIds8 {
        linear_gated_i8: idx(&g, "moe_linear_gated_i8"),
        silu_mul: idx(&g, "silu_mul"),
        scale_add: idx(&g, "scale_add"),
        quant: [idx(&g, "max_abs_row"), idx(&g, "quant_pack")],
    };

    // d_model and moe_ff must be multiples of 32: each is the CONTRACTION
    // width of some expert linear, and `model::int8::quantize_weight` scales
    // per 32-element group of it (`model::int8::GROUP`).
    let shape = MoeShape { rows: 6, d_model: 64, moe_ff: 32, n_experts: 8, top_k: 2 };
    let (m, d, ff, e) = (shape.rows, shape.d_model, shape.moe_ff, shape.n_experts);

    let mut rng = Lcg::new(4242);
    let logits = g.storage_init("logits", &rng.vec_scaled((m * e) as usize, 2.0));
    let x_host = rng.vec_scaled((m * d) as usize, 1.0);
    let x = g.storage_init("x", &x_host);

    let gate_w_host: Vec<Vec<f32>> = (0..e).map(|_| rng.vec_scaled((ff * d) as usize, 0.5)).collect();
    let up_w_host: Vec<Vec<f32>> = (0..e).map(|_| rng.vec_scaled((ff * d) as usize, 0.5)).collect();
    let down_w_host: Vec<Vec<f32>> = (0..e).map(|_| rng.vec_scaled((d * ff) as usize, 0.5)).collect();

    let gate_w: Vec<DeviceBuffer> = gate_w_host.iter().enumerate().map(|(i, w)| g.storage_init(&format!("gw{i}"), w)).collect();
    let up_w: Vec<DeviceBuffer> = up_w_host.iter().enumerate().map(|(i, w)| g.storage_init(&format!("uw{i}"), w)).collect();
    let down_w: Vec<DeviceBuffer> = down_w_host.iter().enumerate().map(|(i, w)| g.storage_init(&format!("dw{i}"), w)).collect();

    // Quantized weights, packed once (import-time, in a real model).
    let gate_w8: Vec<(Vec<u32>, Vec<f32>)> = gate_w_host.iter().map(|w| quantize_weight(w, ff as usize, d as usize)).collect();
    let up_w8: Vec<(Vec<u32>, Vec<f32>)> = up_w_host.iter().map(|w| quantize_weight(w, ff as usize, d as usize)).collect();
    let down_w8: Vec<(Vec<u32>, Vec<f32>)> = down_w_host.iter().map(|w| quantize_weight(w, d as usize, ff as usize)).collect();
    let upload8 = |v: &[(Vec<u32>, Vec<f32>)], prefix: &str| -> Vec<(DeviceBuffer, DeviceBuffer)> {
        v.iter()
            .enumerate()
            .map(|(i, (packed, scale))| {
                let wq = g.storage(packed.len() as u64);
                g.write(&wq, packed);
                (wq, g.storage_init(&format!("{prefix}sw{i}"), scale))
            })
            .collect()
    };
    let gate_w8_dev = upload8(&gate_w8, "gate");
    let up_w8_dev = upload8(&up_w8, "up");
    let down_w8_dev = upload8(&down_w8, "down");

    let gate = g.storage((m * e) as u64);
    g.submit(&[], &[router_fwd(&g, &ids, &shape, &logits, &gate, true, 1.0)]);

    // fp32 sparse reference.
    let scratch = ExpertScratch {
        gate_pre: &g.storage((m * ff) as u64),
        up: &g.storage((m * ff) as u64),
        h: &g.storage((m * ff) as u64),
        expert_out: &g.storage((m * d) as u64),
    };
    let acc_fp32 = g.storage((m * d) as u64);
    for ei in 0..e {
        let steps = expert_fwd(&g, &ids, &shape, &x, &gate, &gate_w[ei as usize], &up_w[ei as usize], &down_w[ei as usize], &scratch, &acc_fp32, ei, ei != 0);
        g.submit(&[], &steps);
    }

    // int8 sparse path: quantize x once (shared across every expert), then loop.
    let xq = g.storage((m * d / 4) as u64);
    let sx = g.storage(m as u64);
    g.submit(&[], &quant_rows_steps(&g, QuantRows { kernels: ids8.quant, x: &x, sx: &sx, xq: &xq, xgs: None }, 0, m, d));

    let scratch8 = ExpertScratch8 {
        gate_pre: &g.storage((m * ff) as u64),
        up: &g.storage((m * ff) as u64),
        h: &g.storage((m * ff) as u64),
        hq: &g.storage((m * ff / 4) as u64),
        sh: &g.storage(m as u64),
        expert_out: &g.storage((m * d) as u64),
    };
    let acc_i8 = g.storage((m * d) as u64);
    for ei in 0..e {
        let gw = Lin8 { wq: &gate_w8_dev[ei as usize].0, sw: &gate_w8_dev[ei as usize].1 };
        let uw = Lin8 { wq: &up_w8_dev[ei as usize].0, sw: &up_w8_dev[ei as usize].1 };
        let dw = Lin8 { wq: &down_w8_dev[ei as usize].0, sw: &down_w8_dev[ei as usize].1 };
        let steps = expert_fwd_i8(&g, &ids8, &shape, &xq, &sx, &gate, gw, uw, dw, &scratch8, &acc_i8, ei, ei != 0);
        g.submit(&[], &steps);
    }

    g.poll_wait();
    let fp32 = g.read(&acc_fp32, (m * d) as usize);
    let i8out = g.read(&acc_i8, (m * d) as usize);

    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in i8out.iter().zip(fp32.iter()) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    let rel_l2 = (num / den.max(1e-12)).sqrt();
    // Measured 0.0084 on this shape/seed; 0.02 leaves headroom for RNG/shape
    // drift without hiding an actual quantization regression.
    assert!(rel_l2 < 0.02, "int8 sparse diverged from fp32 sparse: rel_l2={rel_l2:.4} i8[..4]={:?} fp32[..4]={:?}", &i8out[..4], &fp32[..4]);
    assert!(fp32.iter().any(|&v| v.abs() > 1e-9), "fp32 reference is all-zero - the test shape routes nothing");
}
