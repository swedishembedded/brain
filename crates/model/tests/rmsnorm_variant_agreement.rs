// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::block::rmsnorm_fwd` must compute the same normalization whichever
//! kernel it selects, at every shape the SHARED builders dispatch one at.
//!
//! This gate exists because the selection is not a bit-identical swap and was
//! adopted for speed: `rmsnorm_rows` folds 64 partial sums in a different
//! order than `rmsnorm`'s single-threaded loop, so the two agree to
//! floating-point rounding, not to the bit. Every model that registers the
//! coalesced slot inherits the swap inside `gqa_attn_qkv`, `gqa_mixer_fwd` and
//! `gdn_mixer_fwd` without editing a single one of its own call sites, so what
//! those builders compute has to be pinned HERE, once, rather than
//! rediscovered per model.
//!
//! The comparison itself is `block::assert_rmsnorm_variant_agrees` - the same
//! helper every adopting model calls with its own shapes - against a HOST
//! reference. Comparing the two device kernels to each other would pass if
//! both were wrong the same way.
//!
//! Swedish Embedded AB implements validated GPU kernel selection for its
//! clients. If your team needs expertise in numerically-gated kernel
//! optimization then you can procure our services by sending an email to
//! info@swedishembedded.com.

use model::block::{self, assert_rmsnorm_variant_agrees, KernelIds};

const PIPELINES: &[(&str, &str)] = &[("rmsnorm", kernels::RMSNORM), ("rmsnorm_rows", kernels::RMSNORM_ROWS)];

fn ids(rmsnorm_rows: usize) -> KernelIds {
    KernelIds {
        rmsnorm: 0,
        rms_inv: block::UNREGISTERED,
        rmsnorm_dx: block::UNREGISTERED,
        rmsnorm_dx_rows: block::UNREGISTERED,
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
        rmsnorm_rows,
    }
}

/// The shapes the shared builders really dispatch, named by where they come
/// from rather than by number: a one-row residual norm (`rows = 1` decode
/// step, the case the per-element kernel is worst at), the narrow per-head
/// QK-norms whose row counts are a head count rather than a token count, and a
/// wide prefill/encoder row block.
const SHAPES: &[(u32, u32, &str)] = &[
    (1, 5120, "residual norm, decode step (rows = 1)"),
    (1, 2048, "residual norm, decode step, narrower model"),
    (16, 256, "gqa_mixer q_norm at decode (rows = n_heads)"),
    (2, 256, "gqa_mixer k_norm at decode (rows = n_kv_heads)"),
    (32, 128, "gdn_mixer gated norm at decode (rows = n_value_heads)"),
    (512, 1024, "residual norm, prefill/encoder width"),
];

#[test]
fn the_shared_rmsnorm_builder_matches_the_host_reference_at_every_builder_shape() {
    let gpu = gpu_core::testgpu::dev(PIPELINES);

    // Say out loud which arm the device picked. On a device that cannot run a
    // workgroup reduction both arms ARE the reference kernel and the
    // comparison is a tautology - a legitimate outcome of the selection
    // policy, but it must not be mistaken for coverage of the coalesced
    // kernel.
    let (picked, _) = block::rms_variant(&gpu, 0, Some(1), 1, 5120);
    println!("device selects {} for a one-row RMSNorm", PIPELINES[picked].0);

    // Both arms of the seam on the same inputs: registered (the device's own
    // `select` policy then decides) and UNREGISTERED (always the per-element
    // reference). A model adopting the coalesced kernel must land inside
    // tolerance of the reference AND of the tape it had before.
    assert_rmsnorm_variant_agrees(&gpu, &ids(1), 1e-6, SHAPES);
    assert_rmsnorm_variant_agrees(&gpu, &ids(block::UNREGISTERED), 1e-6, SHAPES);
}

/// The CPU JIT cannot run the workgroup barrier, so a registered cooperative
/// slot still lands on the per-element reference there - which must be
/// correct on its own.
#[test]
fn the_reference_kernel_normalizes_on_the_cpu_jit() {
    let gpu = gpu_core::Gpu::new_cpu(PIPELINES);
    assert_rmsnorm_variant_agrees(&gpu, &ids(block::UNREGISTERED), 1e-6, &SHAPES[..3]);
}

/// Both variants normalize at the CALLER's epsilon.
///
/// `rmsnorm.wgsl` used to add a compiled-in 1e-6 whatever the model asked
/// for, so every Llama-family checkpoint (1e-5), GLM, Kronos and the Mimi
/// codec were normalized with an epsilon they never declared. The inputs are
/// ~1e-3 in magnitude, so `mean(x^2)` (~5e-7) is swamped by the requested
/// 1e-2: a kernel still adding 1e-6 misses by ~100x, far outside tolerance,
/// where the builder-shape gate's O(1) inputs could not tell the two apart.
#[test]
fn every_variant_normalizes_at_the_callers_epsilon() {
    let (rows, dim, eps) = (5usize, 96usize, 1e-2f32);
    let x: Vec<f32> = (0..rows * dim).map(|i| 1e-3 * (i as f32 * 0.7 + 0.1).sin()).collect();
    let w: Vec<f32> = (0..dim).map(|i| 0.5 + 0.25 * (i as f32 * 0.31).cos()).collect();
    let want = model::hostmath::rmsnorm_rows(&x, &w, rows, dim, eps);
    for (gpu, device) in [(gpu_core::testgpu::dev(PIPELINES), "device"), (gpu_core::Gpu::new_cpu(PIPELINES), "cpu jit")] {
        for (coop, arm) in [(1, "registered"), (block::UNREGISTERED, "reference")] {
            let xb = gpu.storage_init("x", &x);
            let wb = gpu.storage_init("w", &w);
            let ob = gpu.storage((rows * dim) as u64);
            gpu.submit(&[], &[block::rmsnorm_fwd(&gpu, &ids(coop), &xb, &wb, &ob, dim as u32, rows as u32, eps)]);
            let got = gpu.read(&ob, rows * dim);
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let err = got.iter().zip(&want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
            assert!(err < 1e-5, "{device}/{arm}: relative error {err:e} at eps {eps}");
        }
    }
}
