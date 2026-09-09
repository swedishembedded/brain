// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: [`EvaVision`] must resolve its kernels **by name** against the device
//! it is handed, not by raw position in [`clip::model::VISION_PIPELINES`].
//!
//! Every real caller shares ONE device between several towers, so the vision
//! kernels do not start at pipeline index 0:
//!
//! * `clip::caps::Session` builds `TEXT_PIPELINES ++ VISION_PIPELINES ++
//!   imaging::PIPELINES` and hands that one device to both the text tower and
//!   the EVA tower (plus `embed_image`'s device-side resize).
//! * `pulid::caps::Bundle` needs the EVA tower AND `imaging`'s resize on one
//!   handle for the same reason.
//!
//! Indexing positionally against such a device silently binds the WRONG
//! pipeline: with `TEXT_PIPELINES` first, EVA's `V_BIAS_ADD` (3) lands on
//! `layernorm`, whose bind group layout has 5 bindings against the 3 a
//! `bias_add` dispatch supplies - a `Device::create_bind_group` validation
//! failure, and for the index pairs that happen to agree on arity (`V_CONV2D`
//! (0) -> `embed`) silently wrong numbers instead.
//!
//! The test is a same-device A/B: identical config, identical weights,
//! identical pixels, once on the bare `VISION_PIPELINES` and once on the
//! shared union. Both must dispatch the same kernels, so both must return
//! bit-identical activations.

use std::collections::HashMap;

use clip::config::EvaVisionConfig;
use clip::model::{EvaVision, TEXT_PIPELINES, VISION_PIPELINES};

/// A tower small enough to build in milliseconds but structurally complete:
/// two blocks (so residuals and the block loop run), a 2x2 patch grid, and a
/// `width` that is a multiple of 64 floats - the stem's `step_sliced` row
/// offsets are multiples of `width`, and a storage binding offset must satisfy
/// the adapter's 256-byte `min_storage_buffer_offset_alignment`.
fn tiny() -> EvaVisionConfig {
    EvaVisionConfig {
        image_size: 8,
        patch: 4,
        width: 64,
        layers: 2,
        heads: 4,
        mlp_hidden: 64,
        embed_dim: 64,
        eps: 1e-6,
        pt_seq_len: 2,
        rope_theta: 10000.0,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    }
}

/// Deterministic, small-magnitude weights for every tensor the manifest names -
/// the values are irrelevant, only that both towers get the SAME ones.
fn weights(cfg: &EvaVisionConfig) -> HashMap<String, Vec<f32>> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    cfg.tensor_manifest()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n).map(|_| next() * 0.1).collect();
            (name, v)
        })
        .collect()
}

#[test]
fn eva_vision_resolves_its_kernels_by_name_on_a_shared_device() {
    let cfg = tiny();
    let w = weights(&cfg);
    let px: Vec<f32> = (0..(3 * cfg.image_size * cfg.image_size) as usize)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();

    let dev = gpu_core::testgpu::dev(VISION_PIPELINES);

    let bare = EvaVision::new_on(dev.share(), cfg.clone(), 1, &w);
    bare.set_pixels(&px);
    bare.forward();
    let want_cls = bare.read_cls_embed_l2norm();
    let want_last = bare.read_x(cfg.layers as usize);
    drop(bare);

    // Exactly the list `clip::caps::Session` builds, in that order.
    let union: Vec<(&str, &str)> = TEXT_PIPELINES
        .iter()
        .chain(VISION_PIPELINES.iter())
        .chain(imaging::PIPELINES.iter())
        .copied()
        .collect();
    let shared = EvaVision::new_on(dev.new_like(&union), cfg.clone(), 1, &w);
    shared.set_pixels(&px);
    shared.forward();

    assert_eq!(
        shared.read_x(cfg.layers as usize),
        want_last,
        "EVA tower output differs on a shared device: kernels were resolved by position, not by name"
    );
    assert_eq!(
        shared.read_cls_embed_l2norm(),
        want_cls,
        "EVA CLS embedding differs on a shared device: kernels were resolved by position, not by name"
    );
}
