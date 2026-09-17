// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The optimizer must get its COOPERATIVE grad-norm reduction.
//!
//! `optim::Optim` picks between two gradient-norm implementations by looking
//! for `gradnorm_part`/`clip_coef_wg` **by name** on the device, and silently
//! falls back to `gradnorm_sq` when either is absent. That fallback is one
//! single-threaded dispatch per tensor: `gradnorm_sq.wgsl` returns from every
//! invocation except `gid == 0`, which then loops serially over the whole
//! buffer.
//!
//! For this model that fallback is not a small tax. The token embedding table
//! is `vocab x d_model` - 11.7M floats on the released MiniLM tier - and one
//! thread walking it dominated the entire training step: **1163 ms of a 1195 ms
//! step, 97%**, against 30 ms for the encoder's forward and backward passes
//! combined. Registering the pair took the same step to 37 ms.
//!
//! Nothing about that failure is visible: the arithmetic is identical either
//! way, so the model trains correctly and merely takes 30x as long. Hence a
//! test rather than a comment.

use decide::kern::PIPELINES;

#[test]
fn the_cooperative_gradnorm_pair_is_registered() {
    for name in ["gradnorm_part", "clip_coef_wg"] {
        assert!(
            PIPELINES.iter().any(|(n, _)| *n == name),
            "{name} is missing from decide::kern::PIPELINES, so optim falls back to the \
             single-threaded gradnorm_sq reduction over every parameter tensor"
        );
    }
}

/// Resolving by name is only half of it - the names have to reach the device.
#[test]
fn the_device_built_from_pipelines_can_resolve_them() {
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    for name in ["gradnorm_part", "clip_coef_wg", "gradnorm_sq", "clip_coef", "adamw"] {
        assert!(gpu.kernel_index(name).is_some(), "{name} did not resolve on the device");
    }
}
