// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::moe`'s gated-linear sparse dispatch vs. `crates/glm`'s proven-exact
//! dense-eval-then-mask oracle, on a tiny synthetic MoE.
//!
//! `router_gate.wgsl`'s own doc comment states running every expert densely
//! and masking by the gate is numerically identical to true sparse top-k
//! dispatch (no capacity dropping). That is the oracle here: build the dense
//! path with the SAME kernels `crates/glm`'s `Mlp::Moe` arm uses
//! (`matmul` -> `silu_mul` -> `matmul` -> `scale_add`, looped over every
//! expert), and assert `model::moe`'s sparse path (`moe_linear_gated`,
//! skipping non-routed rows) produces the identical accumulator.

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::moe::{expert_fwd, router_fwd, ExpertScratch, MoeIds, MoeShape};

const PIPES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
    ("silu_mul", kernels::SILU_MUL),
    ("scale_add", kernels::SCALE_ADD),
    ("router_gate", kernels::ROUTER_GATE),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// The dense oracle: `crates/glm`'s exact per-expert loop shape, using the
/// plain (ungated) `matmul` kernel and relying on `scale_add`'s gate multiply
/// to discard non-selected rows.
#[allow(clippy::too_many_arguments)]
fn dense_expert_step(
    g: &Gpu,
    matmul: usize,
    silu_mul: usize,
    scale_add: usize,
    x: &DeviceBuffer,
    gate: &DeviceBuffer,
    gate_w: &DeviceBuffer,
    up_w: &DeviceBuffer,
    down_w: &DeviceBuffer,
    scratch: &ExpertScratch,
    acc: &DeviceBuffer,
    m: u32,
    d: u32,
    ff: u32,
    e: u32,
    e_idx: u32,
    accumulate: bool,
) {
    g.submit(
        &[],
        &[
            g.step(matmul, &[x, gate_w, scratch.gate_pre], &[m, d, ff], m * ff),
            g.step(matmul, &[x, up_w, scratch.up], &[m, d, ff], m * ff),
            g.step(silu_mul, &[scratch.gate_pre, scratch.up, scratch.h], &[m * ff], m * ff),
            g.step(matmul, &[scratch.h, down_w, scratch.expert_out], &[m, ff, d], m * d),
            g.step(scale_add, &[gate, scratch.expert_out, acc], &[m, d, e, e_idx, accumulate as u32], m * d),
        ],
    );
}

#[test]
fn sparse_matches_dense_oracle() {
    let g = gpu_core::testgpu::dev(PIPES);
    let ids = MoeIds {
        router_gate: idx(&g, "router_gate"),
        linear_gated: idx(&g, "moe_linear_gated"),
        silu_mul: idx(&g, "silu_mul"),
        scale_add: idx(&g, "scale_add"),
    };
    let matmul = idx(&g, "matmul");
    let silu_mul = idx(&g, "silu_mul");
    let scale_add = idx(&g, "scale_add");

    // Small but not degenerate: enough rows/experts that most experts get a
    // real mix of routed and non-routed rows.
    let shape = MoeShape { rows: 6, d_model: 8, moe_ff: 12, n_experts: 8, top_k: 2 };
    let (m, d, ff, e) = (shape.rows, shape.d_model, shape.moe_ff, shape.n_experts);

    let mut rng = Lcg::new(1337);
    let logits = g.storage_init("logits", &rng.vec_scaled((m * e) as usize, 2.0));
    let x = g.storage_init("x", &rng.vec_scaled((m * d) as usize, 1.0));
    let gate_w: Vec<DeviceBuffer> =
        (0..e).map(|i| g.storage_init(&format!("gate_w{i}"), &rng.vec_scaled((ff * d) as usize, 0.5))).collect();
    let up_w: Vec<DeviceBuffer> =
        (0..e).map(|i| g.storage_init(&format!("up_w{i}"), &rng.vec_scaled((ff * d) as usize, 0.5))).collect();
    let down_w: Vec<DeviceBuffer> =
        (0..e).map(|i| g.storage_init(&format!("down_w{i}"), &rng.vec_scaled((d * ff) as usize, 0.5))).collect();

    let gate = g.storage((m * e) as u64);
    g.submit(&[], &[router_fwd(&g, &ids, &shape, &logits, &gate)]);

    let scratch_gate_pre = g.storage((m * ff) as u64);
    let scratch_up = g.storage((m * ff) as u64);
    let scratch_h = g.storage((m * ff) as u64);
    let scratch_out = g.storage((m * d) as u64);
    let scratch = ExpertScratch { gate_pre: &scratch_gate_pre, up: &scratch_up, h: &scratch_h, expert_out: &scratch_out };

    // Sparse path (the code under test).
    let acc_sparse = g.storage((m * d) as u64);
    for ei in 0..e {
        let steps = expert_fwd(&g, &ids, &shape, &x, &gate, &gate_w[ei as usize], &up_w[ei as usize], &down_w[ei as usize], &scratch, &acc_sparse, ei, ei != 0);
        g.submit(&[], &steps);
    }

    // Dense oracle (the same math glm's Mlp::Moe arm runs).
    let acc_dense = g.storage((m * d) as u64);
    for ei in 0..e {
        dense_expert_step(&g, matmul, silu_mul, scale_add, &x, &gate, &gate_w[ei as usize], &up_w[ei as usize], &down_w[ei as usize], &scratch, &acc_dense, m, d, ff, e, ei, ei != 0);
    }

    g.poll_wait();
    let sparse = g.read(&acc_sparse, (m * d) as usize);
    let dense = g.read(&acc_dense, (m * d) as usize);

    let mut max_abs_diff = 0.0f32;
    for (a, b) in sparse.iter().zip(dense.iter()) {
        max_abs_diff = max_abs_diff.max((a - b).abs());
    }
    assert!(
        max_abs_diff < 1e-5,
        "sparse dispatch diverged from the dense oracle: max_abs_diff={max_abs_diff} sparse[..4]={:?} dense[..4]={:?}",
        &sparse[..4.min(sparse.len())],
        &dense[..4.min(dense.len())],
    );

    // A meaningful test, not a vacuous one: every row must actually have been
    // routed somewhere (router_gate always keeps exactly top_k > 0 experts
    // per row), so the accumulator cannot be all zero.
    assert!(dense.iter().any(|&v| v.abs() > 1e-9), "oracle output is all-zero — the test shape routes nothing");
}
