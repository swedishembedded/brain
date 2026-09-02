// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::moe::expert_fwd_compact` (row-compacted, tiled-GEMM dispatch) vs.
//! `crates/glm`'s proven-exact dense-eval-then-mask oracle, on a tiny
//! synthetic MoE - the same oracle `moe_sparse_parity.rs` already validates
//! the naive sparse path against, reused here for the compacted path.
//!
//! Also checks it agrees with `expert_fwd` (the already-gradcheck-validated
//! naive sparse path) directly, and that `CompactExpertScratch`'s capacity
//! guard panic is real (mutation-verify: deliberately undersize the scratch
//! and confirm it actually panics, not just "would panic in theory").
//!
//! `expert_fwd_compact_layer` (M5.4: the layer-batched entry point that
//! replaces `expert_fwd_compact`'s one-`Gpu::submit`-per-expert dispatch
//! shape with one `Vec<Step>` for the whole layer) gets the same bit-for-bit
//! and mutation-verify treatment below, plus a submit-count gate.

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::moe::{
    expert_fwd, expert_fwd_compact, expert_fwd_compact_layer, router_fwd, CompactExpertFwdIds, CompactExpertScratch, ExpertScratch, MoeIds,
    MoeShape,
};

const PIPES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("moe_linear_gated", kernels::MOE_LINEAR_GATED),
    ("silu_mul", kernels::SILU_MUL),
    ("scale_add", kernels::SCALE_ADD),
    ("router_gate", kernels::ROUTER_GATE),
    ("embed", kernels::EMBED),
    ("moe_scatter_scaled_add", kernels::MOE_SCATTER_SCALED_ADD),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

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

struct Setup {
    g: Gpu,
    moe_ids: MoeIds,
    compact_ids: CompactExpertFwdIds,
    matmul: usize,
    silu_mul: usize,
    scale_add: usize,
    shape: MoeShape,
    x: DeviceBuffer,
    gate: DeviceBuffer,
    host_gate: Vec<f32>,
    gate_w: Vec<DeviceBuffer>,
    up_w: Vec<DeviceBuffer>,
    down_w: Vec<DeviceBuffer>,
}

fn build(shape: MoeShape, seed: u64) -> Setup {
    let g = gpu_core::testgpu::dev(PIPES);
    let moe_ids = MoeIds {
        router_gate: idx(&g, "router_gate"),
        linear_gated: idx(&g, "moe_linear_gated"),
        silu_mul: idx(&g, "silu_mul"),
        scale_add: idx(&g, "scale_add"),
    };
    let compact_ids = CompactExpertFwdIds {
        gather: idx(&g, "embed"),
        gemm_naive: idx(&g, "matmul"),
        gemm_tiled: idx(&g, "matmul_reg3"),
        silu_mul: idx(&g, "silu_mul"),
        scatter: idx(&g, "moe_scatter_scaled_add"),
    };
    let matmul = idx(&g, "matmul");
    let silu_mul = idx(&g, "silu_mul");
    let scale_add = idx(&g, "scale_add");

    let (m, d, ff, e) = (shape.rows, shape.d_model, shape.moe_ff, shape.n_experts);
    let mut rng = Lcg::new(seed);
    let logits = g.storage_init("logits", &rng.vec_scaled((m * e) as usize, 2.0));
    let x = g.storage_init("x", &rng.vec_scaled((m * d) as usize, 1.0));
    let gate_w: Vec<DeviceBuffer> =
        (0..e).map(|i| g.storage_init(&format!("gate_w{i}"), &rng.vec_scaled((ff * d) as usize, 0.5))).collect();
    let up_w: Vec<DeviceBuffer> =
        (0..e).map(|i| g.storage_init(&format!("up_w{i}"), &rng.vec_scaled((ff * d) as usize, 0.5))).collect();
    let down_w: Vec<DeviceBuffer> =
        (0..e).map(|i| g.storage_init(&format!("down_w{i}"), &rng.vec_scaled((d * ff) as usize, 0.5))).collect();

    let gate = g.storage((m * e) as u64);
    g.submit(&[], &[router_fwd(&g, &moe_ids, &shape, &logits, &gate, true, 1.0)]);
    g.poll_wait();
    let host_gate = g.read(&gate, (m * e) as usize);

    Setup { g, moe_ids, compact_ids, matmul, silu_mul, scale_add, shape, x, gate, host_gate, gate_w, up_w, down_w }
}

/// The compacted path must reproduce the dense-eval-then-mask oracle exactly
/// (within float tolerance) - the same claim `moe_sparse_parity.rs` already
/// proves for the naive sparse path, checked here for the compacted one.
#[test]
fn compact_matches_dense_oracle() {
    // rows=20 experts=6 top_k=2: enough rows/expert on average (~6-7) to
    // exercise BOTH pick_gemm branches across the 6 experts (some land under
    // the naive threshold, some over) without hand-picking which.
    let s = build(MoeShape { rows: 20, d_model: 8, moe_ff: 12, n_experts: 6, top_k: 2 }, 4242);
    let (m, d, ff, e) = (s.shape.rows, s.shape.d_model, s.shape.moe_ff, s.shape.n_experts);

    let scratch_gate_pre = s.g.storage((m * ff) as u64);
    let scratch_up = s.g.storage((m * ff) as u64);
    let scratch_h = s.g.storage((m * ff) as u64);
    let scratch_out = s.g.storage((m * d) as u64);
    let dense_scratch = ExpertScratch { gate_pre: &scratch_gate_pre, up: &scratch_up, h: &scratch_h, expert_out: &scratch_out };

    let acc_dense = s.g.storage((m * d) as u64);
    for ei in 0..e {
        dense_expert_step(
            &s.g, s.matmul, s.silu_mul, s.scale_add, &s.x, &s.gate,
            &s.gate_w[ei as usize], &s.up_w[ei as usize], &s.down_w[ei as usize],
            &dense_scratch, &acc_dense, m, d, ff, e, ei, ei != 0,
        );
    }

    let compact_scratch = CompactExpertScratch::new(&s.g, &s.shape, m);
    let acc_compact = s.g.storage((m * d) as u64);
    let mut total_routed = 0usize;
    for ei in 0..e {
        total_routed += expert_fwd_compact(
            &s.g, &s.compact_ids, &s.shape, &s.host_gate, &s.x, &s.gate,
            &s.gate_w[ei as usize], &s.up_w[ei as usize], &s.down_w[ei as usize],
            &compact_scratch, &acc_compact, ei, ei != 0,
        );
    }
    // top_k=2 over 20 rows routes exactly 40 (row, expert) pairs total.
    assert_eq!(total_routed, (m * s.shape.top_k) as usize, "routing accounting mismatch");

    s.g.poll_wait();
    let dense = s.g.read(&acc_dense, (m * d) as usize);
    let compact = s.g.read(&acc_compact, (m * d) as usize);

    let mut max_abs_diff = 0.0f32;
    for (a, b) in compact.iter().zip(dense.iter()) {
        max_abs_diff = max_abs_diff.max((a - b).abs());
    }
    assert!(
        max_abs_diff < 1e-5,
        "compacted dispatch diverged from the dense oracle: max_abs_diff={max_abs_diff} \
         compact[..4]={:?} dense[..4]={:?}",
        &compact[..4.min(compact.len())],
        &dense[..4.min(dense.len())],
    );
    assert!(dense.iter().any(|&v| v.abs() > 1e-9), "oracle output is all-zero - the test shape routes nothing");
}

/// The compacted path and the existing naive sparse path ([`expert_fwd`])
/// must agree directly, not just both happen to match the dense oracle
/// independently.
#[test]
fn compact_matches_naive_sparse() {
    let s = build(MoeShape { rows: 17, d_model: 6, moe_ff: 10, n_experts: 5, top_k: 2 }, 999);
    let (m, d, ff, e) = (s.shape.rows, s.shape.d_model, s.shape.moe_ff, s.shape.n_experts);

    let scratch_gate_pre = s.g.storage((m * ff) as u64);
    let scratch_up = s.g.storage((m * ff) as u64);
    let scratch_h = s.g.storage((m * ff) as u64);
    let scratch_out = s.g.storage((m * d) as u64);
    let sparse_scratch = ExpertScratch { gate_pre: &scratch_gate_pre, up: &scratch_up, h: &scratch_h, expert_out: &scratch_out };

    let acc_sparse = s.g.storage((m * d) as u64);
    for ei in 0..e {
        let steps = expert_fwd(
            &s.g, &s.moe_ids, &s.shape, &s.x, &s.gate,
            &s.gate_w[ei as usize], &s.up_w[ei as usize], &s.down_w[ei as usize],
            &sparse_scratch, &acc_sparse, ei, ei != 0,
        );
        s.g.submit(&[], &steps);
    }

    let compact_scratch = CompactExpertScratch::new(&s.g, &s.shape, m);
    let acc_compact = s.g.storage((m * d) as u64);
    for ei in 0..e {
        expert_fwd_compact(
            &s.g, &s.compact_ids, &s.shape, &s.host_gate, &s.x, &s.gate,
            &s.gate_w[ei as usize], &s.up_w[ei as usize], &s.down_w[ei as usize],
            &compact_scratch, &acc_compact, ei, ei != 0,
        );
    }

    s.g.poll_wait();
    let sparse = s.g.read(&acc_sparse, (m * d) as usize);
    let compact = s.g.read(&acc_compact, (m * d) as usize);
    let mut max_abs_diff = 0.0f32;
    for (a, b) in compact.iter().zip(sparse.iter()) {
        max_abs_diff = max_abs_diff.max((a - b).abs());
    }
    assert!(max_abs_diff < 1e-5, "compact vs naive-sparse diverged: max_abs_diff={max_abs_diff}");
}

/// A layer where AT LEAST ONE expert is routed zero rows exercises the
/// `count == 0` early-return path - real at low `rows`/high `n_experts`, and
/// the one branch the two parity tests above (chosen to route densely) never
/// hit.
#[test]
fn an_unrouted_expert_still_honours_the_accumulate_contract() {
    let s = build(MoeShape { rows: 4, d_model: 6, moe_ff: 8, n_experts: 16, top_k: 1 }, 7);
    let (m, d, e) = (s.shape.rows, s.shape.d_model, s.shape.n_experts);
    // At least one of 16 experts must be unrouted when only 4 rows x top_k=1
    // select experts (pigeonhole: 4 draws cannot cover 16 experts).
    let routed_experts: std::collections::HashSet<u32> =
        (0..m).filter_map(|r| (0..e).find(|&ei| s.host_gate[(r * e + ei) as usize] > 0.0)).collect();
    assert!(routed_experts.len() < e as usize, "expected at least one unrouted expert in this shape");

    let scratch = CompactExpertScratch::new(&s.g, &s.shape, m);
    let acc = s.g.storage((m * d) as u64);
    for ei in 0..e {
        expert_fwd_compact(
            &s.g, &s.compact_ids, &s.shape, &s.host_gate, &s.x, &s.gate,
            &s.gate_w[ei as usize], &s.up_w[ei as usize], &s.down_w[ei as usize],
            &scratch, &acc, ei, ei != 0,
        );
    }
    s.g.poll_wait();
    let out = s.g.read(&acc, (m * d) as usize);
    // Every row got routed to exactly top_k=1 real expert, so the final
    // accumulator must be finite and non-trivial, not left as whatever the
    // FIRST (possibly unrouted) expert's `accumulate=false` branch produced.
    assert!(out.iter().any(|&v| v.abs() > 1e-9), "accumulator is all-zero -- the unrouted-first-expert case broke accumulation");
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output: {out:?}");
}

/// Mutation-verify: [`CompactExpertScratch`]'s capacity guard must be a real,
/// load-bearing panic - not dead code that happens to never trigger.
#[test]
#[should_panic(expected = "exceeding scratch capacity")]
fn undersized_scratch_panics_loudly_rather_than_truncating() {
    let s = build(MoeShape { rows: 20, d_model: 8, moe_ff: 12, n_experts: 2, top_k: 2 }, 55);
    // top_k == n_experts: EVERY row routes to EVERY expert, so each expert
    // sees exactly `rows` routed rows -- capacity 1 must panic on expert 0.
    let scratch = CompactExpertScratch::new(&s.g, &s.shape, 1);
    let acc = s.g.storage((s.shape.rows * s.shape.d_model) as u64);
    expert_fwd_compact(
        &s.g, &s.compact_ids, &s.shape, &s.host_gate, &s.x, &s.gate,
        &s.gate_w[0], &s.up_w[0], &s.down_w[0],
        &scratch, &acc, 0, false,
    );
}

fn expert_weights(s: &Setup) -> Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)> {
    (0..s.shape.n_experts as usize).map(|ei| (s.gate_w[ei].clone(), s.up_w[ei].clone(), s.down_w[ei].clone())).collect()
}

/// M5.4's correctness half: `expert_fwd_compact_layer` must reproduce
/// `expert_fwd_compact`'s per-expert-loop output bit-for-bit (same
/// arithmetic, same expert iteration order, same IEEE-754 summation order -
/// only the host/submit granularity differs).
#[test]
fn compact_layer_matches_per_expert_compact_bit_for_bit() {
    let s = build(MoeShape { rows: 17, d_model: 6, moe_ff: 10, n_experts: 5, top_k: 2 }, 999);
    let (m, d, e) = (s.shape.rows, s.shape.d_model, s.shape.n_experts);

    // Today's dispatch shape: one `expert_fwd_compact` call (host scan +
    // index upload + `Gpu::submit`) per expert.
    let scratch_old = CompactExpertScratch::new(&s.g, &s.shape, m);
    let acc_old = s.g.storage((m * d) as u64);
    for ei in 0..e {
        expert_fwd_compact(
            &s.g, &s.compact_ids, &s.shape, &s.host_gate, &s.x, &s.gate,
            &s.gate_w[ei as usize], &s.up_w[ei as usize], &s.down_w[ei as usize],
            &scratch_old, &acc_old, ei, ei != 0,
        );
    }

    // The fix: one `Vec<Step>` for the whole layer (plus the caller's own
    // explicit pre-zero of `acc`, matching
    // `crates/glmdsa::model::Glm::forward_compact`'s real convention).
    let scratch_new = CompactExpertScratch::new(&s.g, &s.shape, m);
    let acc_new = s.g.storage((m * d) as u64);
    let weights = expert_weights(&s);
    s.g.submit(&[&acc_new], &[]);
    let (steps, counts) =
        expert_fwd_compact_layer(&s.g, &s.compact_ids, &s.shape, &s.host_gate, &s.x, &s.gate, &weights, &scratch_new, &acc_new);
    s.g.submit(&[], &steps);

    assert_eq!(counts.iter().sum::<usize>(), (m * s.shape.top_k) as usize, "routing accounting mismatch");

    s.g.poll_wait();
    let old = s.g.read(&acc_old, (m * d) as usize);
    let new = s.g.read(&acc_new, (m * d) as usize);
    assert_eq!(old, new, "expert_fwd_compact_layer diverged from the per-expert path bit-for-bit");
}

/// M5.4's actual fix: a whole layer's row-compacted MoE forward must cost a
/// CONSTANT, small number of `Gpu::submit` calls regardless of `n_experts` -
/// not one call per expert (`crates/model/src/moe.rs`'s own module comment
/// names this exact defect: ~6100 submits/forward at GLM-5.2 scale). Swept
/// across two very different `n_experts` so a result that happened to hold
/// only by coincidence at one scale would show up as growth at the other.
#[test]
fn compact_layer_submit_count_does_not_scale_with_expert_count() {
    for &n_experts in &[4u32, 32u32] {
        let s = build(MoeShape { rows: 12, d_model: 6, moe_ff: 8, n_experts, top_k: 2 }, 1000 + n_experts as u64);
        let (m, d) = (s.shape.rows, s.shape.d_model);
        let scratch = CompactExpertScratch::new(&s.g, &s.shape, m);
        let acc = s.g.storage((m * d) as u64);
        let weights = expert_weights(&s);

        let before = s.g.stats().expect("this backend reports device stats");
        s.g.submit(&[&acc], &[]);
        let (steps, _counts) = expert_fwd_compact_layer(&s.g, &s.compact_ids, &s.shape, &s.host_gate, &s.x, &s.gate, &weights, &scratch, &acc);
        s.g.submit(&[], &steps);
        s.g.poll_wait();
        let after = s.g.stats().unwrap();

        let submits = after.submits - before.submits;
        assert!(
            submits <= 3,
            "expert_fwd_compact_layer at n_experts={n_experts} cost {submits} Gpu::submit calls \
             for one layer -- expected a small constant, independent of expert count"
        );
    }
}

/// The layer path's `count == 0` skip (an expert with no routed rows this
/// call contributes NO step at all, unlike the per-expert function's
/// dedicated hardware-clear fallback) must still leave `acc` correctly
/// accumulated once every OTHER expert's dispatch runs.
#[test]
fn compact_layer_handles_an_unrouted_expert() {
    let s = build(MoeShape { rows: 4, d_model: 6, moe_ff: 8, n_experts: 16, top_k: 1 }, 7);
    let (m, d, e) = (s.shape.rows, s.shape.d_model, s.shape.n_experts);
    let routed_experts: std::collections::HashSet<u32> =
        (0..m).filter_map(|r| (0..e).find(|&ei| s.host_gate[(r * e + ei) as usize] > 0.0)).collect();
    assert!(routed_experts.len() < e as usize, "expected at least one unrouted expert in this shape");

    let scratch = CompactExpertScratch::new(&s.g, &s.shape, m);
    let acc = s.g.storage((m * d) as u64);
    let weights = expert_weights(&s);
    s.g.submit(&[&acc], &[]);
    let (steps, counts) = expert_fwd_compact_layer(&s.g, &s.compact_ids, &s.shape, &s.host_gate, &s.x, &s.gate, &weights, &scratch, &acc);
    s.g.submit(&[], &steps);
    s.g.poll_wait();
    let out = s.g.read(&acc, (m * d) as usize);
    assert!(out.iter().any(|&v| v.abs() > 1e-9), "accumulator is all-zero -- the unrouted-expert case broke accumulation");
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output: {out:?}");
    assert_eq!(counts.iter().sum::<usize>(), (m * s.shape.top_k) as usize, "routing accounting mismatch");
}

/// Mutation-verify: the layer path's per-expert capacity guard must be a
/// real, load-bearing panic too, not dropped in translation from the
/// per-expert function.
#[test]
#[should_panic(expected = "exceeding scratch capacity")]
fn compact_layer_undersized_scratch_panics_loudly() {
    let s = build(MoeShape { rows: 20, d_model: 8, moe_ff: 12, n_experts: 2, top_k: 2 }, 55);
    let scratch = CompactExpertScratch::new(&s.g, &s.shape, 1);
    let acc = s.g.storage((s.shape.rows * s.shape.d_model) as u64);
    let weights = expert_weights(&s);
    let _ = expert_fwd_compact_layer(&s.g, &s.compact_ids, &s.shape, &s.host_gate, &s.x, &s.gate, &weights, &scratch, &acc);
}
