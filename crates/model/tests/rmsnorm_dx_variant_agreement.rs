// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::block::rmsnorm_bwd` must compute the same input gradient (`dx`)
//! whichever kernel it selects, at every shape a real model's backward tape
//! dispatches one at.
//!
//! Companion to `rmsnorm_variant_agreement.rs`'s forward gate, for the
//! backward half: `rmsnorm_dx.wgsl` had NO cooperative sibling at all before
//! this milestone (unlike the forward `rmsnorm`/`rmsnorm_rows` pair) - one
//! thread walked the whole row TWICE, once per independent reduction
//! (`sum(x^2)` and `sum(dy*w*x)`). `rmsnorm_dx_rows` folds both partials in
//! ONE pass across 64 threads, in a different order than the per-element
//! kernel's single-threaded loop, so the two agree to floating-point
//! rounding, not to the bit - hence a numerical gate against a HOST
//! reference (`hostmath::rmsnorm_dx_rows`), never the two device kernels
//! compared to each other.
//!
//! Swedish Embedded AB implements validated GPU kernel selection for its
//! clients. If your team needs expertise in numerically-gated kernel
//! optimization then you can procure our services by sending an email to
//! info@swedishembedded.com.

use model::block::{self, assert_rmsnorm_dx_variant_agrees, KernelIds};

const PIPELINES: &[(&str, &str)] =
    &[("rmsnorm_dx", kernels::RMSNORM_DX), ("rmsnorm_dx_rows", kernels::RMSNORM_DX_ROWS)];

fn ids(rmsnorm_dx_rows: usize) -> KernelIds {
    KernelIds {
        rmsnorm: block::UNREGISTERED,
        rms_inv: block::UNREGISTERED,
        rmsnorm_dx: 0,
        rmsnorm_dw: block::UNREGISTERED,
        rope: block::UNREGISTERED,
        rope_bwd: block::UNREGISTERED,
        gqa_scores: block::UNREGISTERED,
        gqa_apply: block::UNREGISTERED,
        attn_softmax: block::UNREGISTERED,
        gqa_dscores: block::UNREGISTERED,
        gqa_dv: block::UNREGISTERED,
        gqa_dq: block::UNREGISTERED,
        gqa_dk: block::UNREGISTERED,
        silu_mul: block::UNREGISTERED,
        silu_da: block::UNREGISTERED,
        silu_db: block::UNREGISTERED,
        rmsnorm_rows: block::UNREGISTERED,
        rmsnorm_dx_rows,
    }
}

/// The shapes real backward tapes dispatch one at: a decode-shaped single
/// row, the narrow per-head QK-norm gradient row counts, and a wide
/// prefill/training row block - same family as `rmsnorm_variant_agreement`'s
/// own shape list, plus one row count that does NOT divide 64 evenly (37) to
/// exercise the cooperative kernel's tail handling.
const SHAPES: &[(u32, u32, &str)] = &[
    (1, 5120, "residual dx, decode step (rows = 1)"),
    (1, 2048, "residual dx, decode step, narrower model"),
    (37, 896, "row count not a multiple of 64 (tail handling)"),
    (16, 256, "gqa_mixer q_norm dx at decode (rows = n_heads)"),
    (512, 1024, "residual dx, prefill/training width"),
];

#[test]
fn the_shared_rmsnorm_dx_builder_matches_the_host_reference_at_every_builder_shape() {
    let gpu = gpu_core::testgpu::dev(PIPELINES);

    let (picked, _) = block::rms_variant(&gpu, 0, Some(1), 1, 5120);
    println!("device selects {} for a one-row RMSNorm dx", PIPELINES[picked].0);

    // Both arms of the seam on the same inputs: registered (the device's own
    // `select` policy then decides) and UNREGISTERED (always the per-element
    // reference).
    assert_rmsnorm_dx_variant_agrees(&gpu, &ids(1), SHAPES);
    assert_rmsnorm_dx_variant_agrees(&gpu, &ids(block::UNREGISTERED), SHAPES);
}

/// The CPU JIT cannot run `rmsnorm_dx_rows`'s workgroup barrier, so a model
/// that registers the slot on a device reporting `workgroup_reductions:
/// false` must still land on the per-element reference and still be
/// correct, the same reference-kernel-with-the-cooperative-slot-registered
/// case `rmsnorm_variant_agreement.rs` pins for the forward half.
#[test]
fn the_reference_kernel_still_computes_dx_when_the_cooperative_slot_is_registered_on_the_cpu_jit() {
    let gpu = gpu_core::Gpu::new_cpu(PIPELINES);
    assert_rmsnorm_dx_variant_agrees(&gpu, &ids(1), &SHAPES[..3]);
}
