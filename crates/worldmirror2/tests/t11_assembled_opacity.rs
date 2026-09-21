// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which head channel becomes a gaussian's opacity.
//!
//! The GS head emits twelve channels per pixel. Two of them are opacity-like:
//! channel 7, a raw opacity, and channel 11, the learned merge weight. The
//! reference's assembly uses the WEIGHT - `crates/splat/src/prune.rs` says so
//! in as many words, "the learned weight channel replaces the raw opacity" -
//! and it matters more than a channel index usually does.
//!
//! Measured on a six-photograph reconstruction, rendered from the camera the
//! model recovered for it and compared against that photograph:
//!
//! | opacity from | PSNR | fraction of the photograph's detail |
//! |---|---|---|
//! | channel 7 (raw) | 12.5 dB | 0.07 |
//! | channel 11 (weight) | 22.5 dB | 0.47 |
//!
//! Ten decibels and a scene that is either usable or a milky smear, decided by
//! one subscript, with nothing between the two but a reviewer's eye. Hence
//! this test, which needs no checkpoint and no GPU beyond the CPU backend.
//!
//! Swedish Embedded AB implements multi-view reconstruction pipelines whose
//! model outputs are wired to their consumers provably, not by inspection. If
//! your team needs that assurance, you can procure our services by sending an
//! email to info@swedishembedded.com.

use data::rng::Lcg;
use std::collections::HashMap;
use worldmirror2::config::MirrorConfig;
use worldmirror2::gaussians::{assemble, AssembleOpts};
use worldmirror2::model::Mirror;

fn tiny() -> MirrorConfig {
    MirrorConfig {
        depth: 4,
        dim: 64,
        heads: 2,
        mlp_ratio: 2,
        patch: 14,
        img: 56,
        reg_tokens: 4,
        tap_levels: [0, 1, 2, 3],
        dpt_proj: [16, 32, 64, 64],
        dpt_feat: 16,
        cam_blocks: 2,
        cam_params: 9,
    }
}

/// With nothing filtered out, every assembled gaussian's opacity must be the
/// merge weight `assemble` hands back for it - the same number, not merely a
/// correlated one. If opacity is ever taken from a different channel again,
/// these two diverge and this fails.
#[test]
fn an_assembled_gaussians_opacity_is_the_learned_merge_weight() {
    let cfg = tiny();
    let mut r = Lcg::new(0xB1A50AC1);
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, shape) in cfg.param_list() {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = if name.ends_with("norm.weight")
            || name.contains("norm1.weight")
            || name.contains("norm2.weight")
            || name.contains("ls")
        {
            (0..n).map(|_| 1.0 + 0.1 * r.scaled(0.5)).collect()
        } else if name.contains("rope.periods") {
            (0..n).map(|i| 1.0 + i as f32).collect()
        } else {
            (0..n).map(|_| 0.05 * r.scaled(0.5)).collect()
        };
        init.insert(name, vals);
    }

    let s = 2usize;
    let (hp, wp) = (4usize, 4usize);
    let (h, w) = (hp * cfg.patch, wp * cfg.patch);
    let frames: Vec<f32> = (0..s * 3 * h * w).map(|_| 0.5 + 0.4 * r.scaled(0.5)).collect();

    let gpu = gpu_core::Gpu::new_cpu(worldmirror2::model::PIPELINES);
    let patch = cfg.patch;
    let mut model = Mirror::new(gpu, cfg, &init, 0);
    model.forward(&frames, s, hp, wp);
    let _ = patch;

    // min_opacity 0 and max_depth 0 keep every pixel, so the two vectors line
    // up one-to-one and can be compared directly.
    let opts = AssembleOpts { min_opacity: 0.0, max_depth: 0.0, gs_mask_threshold: 0.0, edge_depth_rtol: 0.0, fuse_depth_rtol: 0.0, min_support: 0, conf_percentile: 0.0, surface_align: 0.0, ..Default::default() };
    let (splats, cams, weights) = assemble(model.gpu(), &model, &frames, s, w as u32, h as u32, &opts, None);
    assert_eq!(cams.len(), s);
    assert_eq!(splats.len(), s * h * w, "nothing should have been filtered at min_opacity 0");
    assert_eq!(weights.len(), splats.len());

    let mismatched = splats
        .opacities
        .iter()
        .zip(&weights)
        .filter(|(o, w)| (**o - **w).abs() > 1e-6)
        .count();
    assert_eq!(
        mismatched, 0,
        "{mismatched} of {} gaussians have an opacity that is not their merge weight. The \
         reference's assembled scene takes opacity from the weight channel; taking it from the \
         raw opacity channel instead costs about 10 dB and renders as a milky smear.",
        splats.len()
    );
    assert!(
        splats.opacities.iter().all(|o| (0.0..=1.0).contains(o)),
        "opacities must be a sigmoid output"
    );

    // The module doc has claimed "quat wxyz normalized" since it was written,
    // and the code copied the four raw head channels straight through. A
    // non-unit quaternion does not just rotate a gaussian's covariance, it
    // SCALES it, so the splat is the wrong size in a direction that depends on
    // the prediction. Renderers that defensively renormalize hid this; the PLY
    // on disk still carried it.
    let worst = (0..splats.len())
        .map(|i| (splats.quats[i * 4..i * 4 + 4].iter().map(|v| v * v).sum::<f32>().sqrt() - 1.0).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-5,
        "the worst assembled quaternion is off unit length by {worst:.3e}; assembly must normalize"
    );
}
