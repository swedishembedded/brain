// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::moe::shared_expert_fwd` vs. a host-computed oracle, on a tiny
//! synthetic shape. Validates the always-active (non-gated) dense SwiGLU +
//! sigmoid-gated combine `Qwen3OmniMoeTalkerTextSparseMoeBlock.forward` uses
//! for its shared expert: `acc + sigmoid(x @ shared_gate_w^T) * swiglu(x)`.

use data::rng::Lcg;
use gpu_core::Gpu;
use model::moe::{shared_expert_fwd, SharedExpertIds, SharedExpertScratch};

const PIPES: &[(&str, &str)] = &[("matmul", kernels::MATMUL), ("silu_mul", kernels::SILU_MUL), ("sigmoid", kernels::SIGMOID), ("scale_row", kernels::SCALE_ROW), ("add2", kernels::ADD2)];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn host_matmul(x: &[f32], w: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * n];
    for r in 0..rows {
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

fn host_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn shared_expert_matches_host_oracle() {
    let g = gpu_core::testgpu::dev(PIPES);
    let ids = SharedExpertIds { matmul: idx(&g, "matmul"), silu_mul: idx(&g, "silu_mul"), sigmoid: idx(&g, "sigmoid"), scale_row: idx(&g, "scale_row"), add2: idx(&g, "add2") };

    let (rows, d, ff) = (5usize, 6usize, 4usize);
    let mut rng = Lcg::new(9001);
    let x_h = rng.vec_scaled(rows * d, 1.0);
    let gw_h = rng.vec_scaled(ff * d, 0.5);
    let uw_h = rng.vec_scaled(ff * d, 0.5);
    let dw_h = rng.vec_scaled(d * ff, 0.5);
    let sgw_h = rng.vec_scaled(d, 0.5); // [1, d]
    let acc_h = rng.vec_scaled(rows * d, 1.0);

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
    let steps = shared_expert_fwd(&g, &ids, rows as u32, d as u32, ff as u32, &x, &gw, &uw, &dw, &sgw, &scratch, &acc, &out);
    g.submit(&[], &steps);
    let got = g.read(&out, rows * d);

    // Host oracle: identical formula, computed independently.
    let gate_pre = host_matmul(&x_h, &gw_h, rows, d, ff);
    let up = host_matmul(&x_h, &uw_h, rows, d, ff);
    let h: Vec<f32> = gate_pre.iter().zip(&up).map(|(&a, &b)| (a / (1.0 + (-a).exp())) * b).collect();
    let mlp_out = host_matmul(&h, &dw_h, rows, ff, d);
    let gate_logits = host_matmul(&x_h, &sgw_h, rows, d, 1);
    let mut want = vec![0f32; rows * d];
    for r in 0..rows {
        let s = host_sigmoid(gate_logits[r]);
        for c in 0..d {
            want[r * d + c] = acc_h[r * d + c] + s * mlp_out[r * d + c];
        }
    }

    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-4, "index {i}: got {g} want {w}");
    }
}
