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
    assert_rmsnorm_variant_agrees(&gpu, &ids(1), SHAPES);
    assert_rmsnorm_variant_agrees(&gpu, &ids(block::UNREGISTERED), SHAPES);
}

/// The reference kernel's `Params` struct declares TWO fields; `rmsnorm_fwd`
/// now writes three so both variants can share one uniform layout. That is
/// fine on a GPU backend (a uniform buffer may be larger than the struct bound
/// to it) but `backend-cpu` has no binding or uniform-size check at dispatch
/// at all, so "it compiles and the GPU is happy" is not evidence for the JIT.
/// The CPU device also cannot run the workgroup barrier, so this is exactly
/// the reference-kernel-with-a-third-param case.
#[test]
fn the_reference_kernel_still_normalizes_when_handed_the_three_field_uniform_on_the_cpu_jit() {
    let gpu = gpu_core::Gpu::new_cpu(PIPELINES);
    assert_rmsnorm_variant_agrees(&gpu, &ids(block::UNREGISTERED), &SHAPES[..3]);
}
