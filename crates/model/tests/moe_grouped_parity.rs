// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::moe::expert_fwd_grouped` (device-side row-permuted grouped GEMM,
//! zero host readback - M5.10) vs. a dense-eval-then-mask oracle built with
//! `model::block::pick_gemm`, the SAME selector `crates/glm`'s real dense
//! arm and [`model::moe::expert_fwd_compact_layer`] both already use.
//!
//! `matmul_reg3_grouped.wgsl`'s K-accumulation loop is copied verbatim from
//! `matmul_reg3.wgsl`, so THAT half of the claim - a grouped GEMM over
//! disjoint row ranges reassociates no output row's own K-reduction - holds
//! by construction and is what these tests exercise at `d_model`/`moe_ff`
//! `>= 128` (`backend_api::select::GEMM_TILE_MIN_COLS`) so the oracle's own
//! `pick_gemm` call ALSO selects `matmul_reg3` rather than the naive
//! reference kernel (whose different accumulation order would make the
//! comparison merely close, not exact, the same way `moe_compact_parity
//! .rs`'s generic tests already are at their own naive-reference shapes).
//!
//! Measured, NOT assumed to match this module's own doc's stronger claim:
//! the FULL pipeline still shows an occasional few-ULP difference (~2e-6
//! absolute, at value magnitudes ~1-20) against the dense oracle, traced to
//! `moe_group_combine.wgsl` - a NEW kernel, not textually identical to
//! `scale_add.wgsl`'s per-expert-dispatch accumulate chain, so a software
//! renderer's legal floating-point contraction (fusing adjacent
//! multiply+add into one rounded FMA where `scale_add`'s memory-round-
//! tripped, one-term-per-dispatch shape gives the optimizer no equivalent
//! opportunity) is free to differ between the two even though both encode
//! the identical mathematical sum in the identical term order. This is the
//! SAME category of "mathematically equivalent, not textually identical
//! kernel" tolerance `moe_compact_parity.rs` already accepts for its own
//! naive-vs-tiled GEMM comparisons, not a routing/indexing defect:
//! bit-exactness here is scoped to the GEMM's own K-reduction (verbatim
//! copied from `matmul_reg3.wgsl`), not to the combine.
//!
//! Also pins the whole builder's dispatch count as a submit-free `Vec<Step>`
//! (no `Gpu::submit`/`Gpu::read` anywhere inside it) and exercises the
//! capacity guard (mutation-verify).

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use model::block::pick_gemm;
use model::moe::{expert_fwd_grouped, router_fwd, GroupedExpertFwdIds, GroupedExpertScratch, MoeIds, MoeShape};

const PIPES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_reg3_grouped", kernels::MATMUL_REG3_GROUPED),
    ("silu_mul", kernels::SILU_MUL),
    ("scale_add", kernels::SCALE_ADD),
    ("router_gate", kernels::ROUTER_GATE),
    ("router_topk_compact", kernels::ROUTER_TOPK_COMPACT),
    ("moe_group_counts", kernels::MOE_GROUP_COUNTS),
    ("moe_group_perm_emit", kernels::MOE_GROUP_PERM_EMIT),
    ("moe_group_combine", kernels::MOE_GROUP_COMBINE),
    ("scan_block", kernels::SCAN_BLOCK),
    ("scan_add", kernels::SCAN_ADD),
    ("embed", kernels::EMBED),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

struct ExpertScratchDense<'a> {
    gate_pre: &'a DeviceBuffer,
    up: &'a DeviceBuffer,
    h: &'a DeviceBuffer,
    expert_out: &'a DeviceBuffer,
}

/// The dense-eval-then-mask oracle, routed through [`pick_gemm`] exactly
/// like `crates/glm`'s real dense MoE arm - NOT a hardcoded naive `matmul`
/// (which is what would make this comparison merely close, not exact; see
/// this file's own header doc).
#[allow(clippy::too_many_arguments)]
fn dense_expert_step(
    g: &Gpu,
    matmul: usize,
    matmul_reg3: usize,
    silu_mul: usize,
    scale_add: usize,
    x: &DeviceBuffer,
    gate: &DeviceBuffer,
    gate_w: &DeviceBuffer,
    up_w: &DeviceBuffer,
    down_w: &DeviceBuffer,
    scratch: &ExpertScratchDense,
    acc: &DeviceBuffer,
    m: u32,
    d: u32,
    ff: u32,
    e: u32,
    e_idx: u32,
    accumulate: bool,
) {
    let lin = |x_in: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, k: u32, n: u32| {
        let (kid, threads) = pick_gemm(m as usize, n as usize, matmul, matmul_reg3, false);
        g.step(kid, &[x_in, w, out], &[m, k, n], threads)
    };
    g.submit(
        &[],
        &[
            lin(x, gate_w, scratch.gate_pre, d, ff),
            lin(x, up_w, scratch.up, d, ff),
            g.step(silu_mul, &[scratch.gate_pre, scratch.up, scratch.h], &[m * ff], m * ff),
            lin(scratch.h, down_w, scratch.expert_out, ff, d),
            g.step(scale_add, &[gate, scratch.expert_out, acc], &[m, d, e, e_idx, accumulate as u32], m * d),
        ],
    );
}

struct Setup {
    g: Gpu,
    grouped_ids: GroupedExpertFwdIds,
    matmul: usize,
    matmul_reg3: usize,
    silu_mul: usize,
    scale_add: usize,
    shape: MoeShape,
    x: DeviceBuffer,
    gate: DeviceBuffer,
    gate_w: Vec<DeviceBuffer>,
    up_w: Vec<DeviceBuffer>,
    down_w: Vec<DeviceBuffer>,
    /// Every expert's gate/up/down weight matrix concatenated back to back,
    /// same source floats as `gate_w`/`up_w`/`down_w` above (so the dense
    /// oracle and the grouped path read IDENTICAL numeric weights, not just
    /// statistically similar ones) - what `expert_fwd_grouped` needs.
    gate_w_all: DeviceBuffer,
    up_w_all: DeviceBuffer,
    down_w_all: DeviceBuffer,
}

fn build(shape: MoeShape, seed: u64) -> Setup {
    let g = gpu_core::testgpu::dev(PIPES);
    let moe_ids = MoeIds {
        router_gate: idx(&g, "router_gate"),
        linear_gated: idx(&g, "router_gate"), // unused by this test, any valid index
        silu_mul: idx(&g, "silu_mul"),
        scale_add: idx(&g, "scale_add"),
    };
    let grouped_ids = GroupedExpertFwdIds {
        router_topk_compact: idx(&g, "router_topk_compact"),
        group_counts: idx(&g, "moe_group_counts"),
        scan_block: idx(&g, "scan_block"),
        scan_add: idx(&g, "scan_add"),
        perm_emit: idx(&g, "moe_group_perm_emit"),
        gather: idx(&g, "embed"),
        gemm_grouped: idx(&g, "matmul_reg3_grouped"),
        silu_mul: idx(&g, "silu_mul"),
        combine: idx(&g, "moe_group_combine"),
    };
    let matmul = idx(&g, "matmul");
    let matmul_reg3 = idx(&g, "matmul_reg3");
    let silu_mul = idx(&g, "silu_mul");
    let scale_add = idx(&g, "scale_add");

    let (m, d, ff, e) = (shape.rows, shape.d_model, shape.moe_ff, shape.n_experts);
    let mut rng = Lcg::new(seed);
    let logits = g.storage_init("logits", &rng.vec_scaled((m * e) as usize, 2.0));
    let x = g.storage_init("x", &rng.vec_scaled((m * d) as usize, 1.0));

    let gate_w_flat = rng.vec_scaled((e * ff * d) as usize, 0.5);
    let up_w_flat = rng.vec_scaled((e * ff * d) as usize, 0.5);
    let down_w_flat = rng.vec_scaled((e * d * ff) as usize, 0.5);
    let gate_w: Vec<DeviceBuffer> = (0..e as usize)
        .map(|i| g.storage_init(&format!("gate_w{i}"), &gate_w_flat[i * (ff * d) as usize..(i + 1) * (ff * d) as usize]))
        .collect();
    let up_w: Vec<DeviceBuffer> = (0..e as usize)
        .map(|i| g.storage_init(&format!("up_w{i}"), &up_w_flat[i * (ff * d) as usize..(i + 1) * (ff * d) as usize]))
        .collect();
    let down_w: Vec<DeviceBuffer> = (0..e as usize)
        .map(|i| g.storage_init(&format!("down_w{i}"), &down_w_flat[i * (d * ff) as usize..(i + 1) * (d * ff) as usize]))
        .collect();
    let gate_w_all = g.storage_init("gate_w_all", &gate_w_flat);
    let up_w_all = g.storage_init("up_w_all", &up_w_flat);
    let down_w_all = g.storage_init("down_w_all", &down_w_flat);

    let gate = g.storage((m * e) as u64);
    g.submit(&[], &[router_fwd(&g, &moe_ids, &shape, &logits, &gate, true, 1.0)]);

    Setup { g, grouped_ids, matmul, matmul_reg3, silu_mul, scale_add, shape, x, gate, gate_w, up_w, down_w, gate_w_all, up_w_all, down_w_all }
}

/// This file's own header doc explains why this is a float tolerance, not
/// `== 0.0`: `moe_group_combine.wgsl` is a new kernel, not textually
/// identical to `scale_add.wgsl`'s per-expert accumulate chain, so a few-ULP
/// contraction difference is legitimate. `1e-4` is ~50x the measured
/// ~2e-6-at-magnitude-~20 differences (headroom, not a fudge), while still
/// being nowhere near what a real routing/indexing bug would produce - a
/// misrouted row diverges by a whole output value, not a handful of ULP.
const TOLERANCE: f32 = 1e-4;

/// Runs both the dense oracle and [`expert_fwd_grouped`] over `s`, asserting
/// the two accumulators agree within [`TOLERANCE`].
fn assert_grouped_matches_dense_exactly(s: &Setup) {
    let (m, d, ff, e) = (s.shape.rows, s.shape.d_model, s.shape.moe_ff, s.shape.n_experts);

    let scratch_gate_pre = s.g.storage((m * ff) as u64);
    let scratch_up = s.g.storage((m * ff) as u64);
    let scratch_h = s.g.storage((m * ff) as u64);
    let scratch_out = s.g.storage((m * d) as u64);
    let dense_scratch = ExpertScratchDense { gate_pre: &scratch_gate_pre, up: &scratch_up, h: &scratch_h, expert_out: &scratch_out };

    let acc_dense = s.g.storage((m * d) as u64);
    for ei in 0..e {
        dense_expert_step(
            &s.g, s.matmul, s.matmul_reg3, s.silu_mul, s.scale_add, &s.x, &s.gate,
            &s.gate_w[ei as usize], &s.up_w[ei as usize], &s.down_w[ei as usize],
            &dense_scratch, &acc_dense, m, d, ff, e, ei, ei != 0,
        );
    }

    let scratch = GroupedExpertScratch::new(&s.g, &s.shape);
    let acc_grouped = s.g.storage((m * d) as u64);
    let steps = expert_fwd_grouped(
        &s.g, &s.grouped_ids, &s.shape, &s.x, &s.gate,
        &s.gate_w_all, &s.up_w_all, &s.down_w_all, &scratch, &acc_grouped,
    );
    s.g.submit(&[], &steps);

    s.g.poll_wait();
    let dense = s.g.read(&acc_dense, (m * d) as usize);
    let grouped = s.g.read(&acc_grouped, (m * d) as usize);

    let mut max_abs_diff = 0.0f32;
    for (a, b) in grouped.iter().zip(dense.iter()) {
        max_abs_diff = max_abs_diff.max((a - b).abs());
    }
    assert!(
        max_abs_diff < TOLERANCE,
        "grouped GEMM diverged from the dense oracle: max_abs_diff={max_abs_diff} \
         grouped[..4]={:?} dense[..4]={:?}",
        &grouped[..4.min(grouped.len())],
        &dense[..4.min(dense.len())],
    );
    assert!(dense.iter().any(|&v| v.abs() > 1e-9), "oracle output is all-zero - the test shape routes nothing");
}

/// Small `rows`, but `d_model`/`moe_ff` >= `GEMM_TILE_MIN_COLS` (128) so the
/// oracle's own `pick_gemm` still selects `matmul_reg3` - exercises
/// `matmul_reg3_grouped`'s ragged-tail bound checks (every expert's routed
/// rows land far short of one full 128-row tile).
#[test]
fn grouped_matches_dense_oracle_ragged_tail() {
    let s = build(MoeShape { rows: 20, d_model: 128, moe_ff: 128, n_experts: 6, top_k: 2 }, 4242);
    assert_grouped_matches_dense_exactly(&s);
}

/// `rows=512, n_experts=4`: ~256 routed rows/expert on average, clearing
/// MORE than one full 128-row GEMM tile per expert group - exercises
/// `matmul_reg3_grouped`'s tile-to-expert-group search across multiple
/// tiles per group, not just the single-tile/ragged-tail case above.
#[test]
fn grouped_matches_dense_oracle_multi_tile_scale() {
    let s = build(MoeShape { rows: 512, d_model: 128, moe_ff: 128, n_experts: 4, top_k: 2 }, 777);
    assert_grouped_matches_dense_exactly(&s);
}

/// The whole builder must record its steps and return without ever calling
/// `Gpu::submit`/`Gpu::read` itself - the entire point of M5.10 over M5.4's
/// per-expert-`submit` compacted path. Measured via `Gpu::stats().submits`:
/// building the steps must not move that counter at all, only the caller's
/// OWN single `submit` at the end may.
#[test]
fn expert_fwd_grouped_never_submits_internally() {
    let s = build(MoeShape { rows: 12, d_model: 6, moe_ff: 8, n_experts: 5, top_k: 2 }, 1000);
    let scratch = GroupedExpertScratch::new(&s.g, &s.shape);
    let acc = s.g.storage((s.shape.rows * s.shape.d_model) as u64);

    s.g.poll_wait();
    let before = s.g.stats().expect("this backend reports device stats");
    let steps = expert_fwd_grouped(
        &s.g, &s.grouped_ids, &s.shape, &s.x, &s.gate,
        &s.gate_w_all, &s.up_w_all, &s.down_w_all, &scratch, &acc,
    );
    let after_build = s.g.stats().unwrap();
    assert_eq!(after_build.submits, before.submits, "expert_fwd_grouped submitted internally - it must return a plain Vec<Step>");

    s.g.submit(&[], &steps);
    s.g.poll_wait();
    let out = s.g.read(&acc, (s.shape.rows * s.shape.d_model) as usize);
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output: {out:?}");
}

/// Mutation-verify: [`GroupedExpertScratch`]'s capacity guard must be a
/// real, load-bearing panic - not dead code that happens to never trigger.
#[test]
#[should_panic(expected = "exceeding GroupedExpertScratch capacity")]
fn undersized_grouped_scratch_panics_loudly() {
    let s = build(MoeShape { rows: 20, d_model: 8, moe_ff: 12, n_experts: 2, top_k: 2 }, 55);
    // top_k == n_experts, so every row routes to every expert: cap=1 must
    // panic against rows*top_k=40.
    let small_shape = MoeShape { rows: 1, d_model: 8, moe_ff: 12, n_experts: 2, top_k: 1 };
    let scratch = GroupedExpertScratch::new(&s.g, &small_shape);
    let acc = s.g.storage((s.shape.rows * s.shape.d_model) as u64);
    let _ = expert_fwd_grouped(
        &s.g, &s.grouped_ids, &s.shape, &s.x, &s.gate,
        &s.gate_w_all, &s.up_w_all, &s.down_w_all, &scratch, &acc,
    );
}
