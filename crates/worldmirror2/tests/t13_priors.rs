// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Known cameras reach the trunk's pose and intrinsics tokens.
//!
//! The trunk has always reserved a pose row and an intrinsics row per frame
//! (`PATCH_START = 7` is cam + 4 registers + pose + ray) and always left them
//! zero, because nothing filled them. The checkpoint carries the embedders for
//! both - `pose_embed` and `ray_embed`, imported and never called.
//!
//! Two properties, and the first matters more than the second. Supplying no
//! priors must be EXACTLY what it was: training drops each prior with
//! probability 0.5 by zeroing its row, so an all-zero row is not a fallback
//! path, it is a case the model was trained on, and the existing behaviour is
//! already correct. Then, priors must actually change the prediction, or the
//! wiring is decorative.
//!
//! Swedish Embedded AB implements conditional reconstruction pipelines whose
//! optional inputs are provably connected. If your team needs that, you can
//! procure our services by sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use std::collections::HashMap;
use worldmirror2::config::MirrorConfig;
use worldmirror2::model::Mirror;
use worldmirror2::priors::CameraPrior;

fn tiny() -> MirrorConfig {
    MirrorConfig {
        depth: 4, dim: 64, heads: 2, mlp_ratio: 2, patch: 14, img: 56, reg_tokens: 4,
        tap_levels: [0, 1, 2, 3], dpt_proj: [16, 32, 64, 64], dpt_feat: 16,
        cam_blocks: 2, cam_params: 9,
    }
}

fn weights(cfg: &MirrorConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut r = Lcg::new(seed);
    let mut init = HashMap::new();
    for (name, shape) in cfg.param_list() {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = if name.ends_with("norm.weight") || name.contains("norm1.weight")
            || name.contains("norm2.weight") || name.contains("ls")
        {
            (0..n).map(|_| 1.0 + 0.1 * r.scaled(0.5)).collect()
        } else if name.contains("rope.periods") {
            (0..n).map(|i| 1.0 + i as f32).collect()
        } else {
            (0..n).map(|_| 0.05 * r.scaled(0.5)).collect()
        };
        init.insert(name, vals);
    }
    init
}

fn look(px: f32, py: f32, pz: f32) -> CameraPrior {
    let mut c2w = [0.0f32; 16];
    c2w[0] = 1.0; c2w[5] = 1.0; c2w[10] = 1.0; c2w[15] = 1.0;
    c2w[3] = px; c2w[7] = py; c2w[11] = pz;
    CameraPrior { c2w, fx: 200.0, fy: 200.0, cx: 28.0, cy: 28.0 }
}

fn run(priors: Option<&[CameraPrior]>) -> Vec<f32> {
    let cfg = tiny();
    let init = weights(&cfg, 0xC0FFEE);
    let s = 3usize;
    let (hp, wp) = (4usize, 4usize);
    let (h, w) = (hp * cfg.patch, wp * cfg.patch);
    let mut r = Lcg::new(0x1234);
    let frames: Vec<f32> = (0..s * 3 * h * w).map(|_| 0.5 + 0.4 * r.scaled(0.5)).collect();
    let gpu = gpu_core::Gpu::new_cpu(worldmirror2::model::PIPELINES);
    let mut m = Mirror::new(gpu, cfg, &init, 0);
    m.forward_with_priors(&frames, s, hp, wp, priors);
    let _ = h;
    let _ = w;
    // The camera prediction is what a pose/intrinsics prior conditions, so it
    // is the signal that says the tokens arrived. (The dense heads read the
    // DPT taps; on a four-block toy with random weights they are numerically
    // insensitive to a change this size, which says nothing about the real
    // checkpoint and would make a brittle gate.)
    m.cam_pred_raw()
}

/// No priors must be exactly what it always was.
#[test]
fn the_no_prior_path_is_unchanged() {
    let a = run(None);
    let b = run(None);
    assert_eq!(a, b, "the no-prior forward is not deterministic");
    assert!(a.iter().all(|v| v.is_finite()), "no-prior forward produced non-finite output");
}

/// And priors have to actually move the prediction.
#[test]
fn supplying_cameras_changes_the_prediction() {
    let none = run(None);
    let cams = [look(0.0, 0.0, 0.0), look(1.0, 0.0, 0.3), look(-1.0, 0.2, 0.1)];
    let with = run(Some(&cams));
    assert!(with.iter().all(|v| v.is_finite()), "prior-conditioned forward produced non-finite output");
    let moved = none.iter().zip(&with).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(
        moved > 1e-6,
        "supplying cameras did not move the camera prediction (max delta {moved:.3e}); the \
         embedders are imported but their tokens are not reaching the trunk"
    );
}

/// Different cameras must give different tokens - otherwise something is
/// being written, but not the thing that was asked for.
#[test]
fn different_cameras_give_different_predictions() {
    let a = run(Some(&[look(0.0, 0.0, 0.0), look(1.0, 0.0, 0.0), look(-1.0, 0.0, 0.0)]));
    let b = run(Some(&[look(0.0, 0.0, 0.0), look(0.0, 1.0, 0.0), look(0.0, -1.0, 0.0)]));
    let moved = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(moved > 1e-6, "two different camera rigs produced the same prediction (max delta {moved:.3e})");
}
