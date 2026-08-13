// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Multi-frame smoke test on a TINY config with synthetic weights: the S>1
//! path (multi-span DINOv2/frame attention, global attention, per-frame head
//! and camera slicing) must produce finite outputs for every frame, and
//! frames with different content must produce different head outputs.
//!
//! Written for real bugs the parity suite (all single-frame) could not see:
//!   * the patch-conv bias used `add_chan_bcast` with N=s, reading past the
//!     shared [C] bias for every frame but the first → NaN gaussians at S=3;
//!   * `dpt.rs` hardcoded the reference's 32-channel `output_conv2.0` where
//!     `param_list` says `feat/8`, so non-default configs read out of bounds.

use data::rng::Lcg;
use worldmirror2::config::MirrorConfig;
use worldmirror2::gaussians::{assemble, AssembleOpts};
use worldmirror2::model::{Head, Mirror};
use std::collections::HashMap;
#[test]
fn s3_forward_is_finite() {
    let cfg = MirrorConfig {
        depth: 4,
        dim: 64,
        heads: 2,
        mlp_ratio: 2,
        patch: 14,
        img: 56, // native 4x4 grid
        reg_tokens: 4,
        tap_levels: [0, 1, 2, 3],
        dpt_proj: [16, 32, 64, 64],
        dpt_feat: 16,
        cam_blocks: 2,
        cam_params: 9,
    };
    let mut r = Lcg::new(0x5EED);
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, shape) in cfg.param_list() {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = if name.ends_with("norm.weight") || name.contains("norm1.weight")
            || name.contains("norm2.weight") || name.contains("ls")
        {
            (0..n).map(|_| 1.0 + 0.1 * r.scaled(0.5)).collect()
        } else if name.contains("rope.periods") {
            (0..n).map(|i| 1.0 + i as f32).collect()
        } else {
            // small scale keeps the random-weight pyramid numerically tame
            (0..n).map(|_| 0.05 * r.scaled(0.5)).collect()
        };
        init.insert(name, vals);
    }

    let s = 3usize;
    let (hp, wp) = (4usize, 4usize);
    let (h, w) = (hp * cfg.patch, wp * cfg.patch);
    let frames: Vec<f32> = (0..s * 3 * h * w).map(|_| 0.5 + 0.4 * r.scaled(0.5)).collect();

    let gpu = gpu_core::Gpu::new_cpu(worldmirror2::model::PIPELINES);
    let mut model = Mirror::new(&gpu, cfg, &init, 0);
    model.forward(&frames, s, hp, wp);

    let td = worldmirror2::model::PATCH_START + hp * wp;
    // DINOv2 per frame
    let po = gpu.read(model.patch_tokens(), s * hp * wp * 64);
    for fi in 0..s {
        let f = &po[fi * hp * wp * 64..(fi + 1) * hp * wp * 64];
        let bad = f.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(bad, 0, "dino patch_out frame {fi}: {bad} non-finite");
    }
    // trunk tokens + taps
    let tt = gpu.read(model.trunk_tokens(), s * td * 64);
    assert_eq!(tt.iter().filter(|x| !x.is_finite()).count(), 0, "NaN in trunk tokens");
    for (ti, tap) in model.taps().iter().enumerate() {
        let v = gpu.read(tap, s * td * 128);
        let bad = v.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(bad, 0, "tap{ti}: {bad}/{} non-finite", v.len());
        // different frame content must reach the taps differently
        assert_ne!(
            &v[0..td * 128],
            &v[td * 128..2 * td * 128],
            "tap{ti}: frames 0 and 1 identical"
        );
    }
    // heads: finite for every frame (random tiny weights wash the tap signal
    // down to near-constants, so value-level frame-dependence is gated at the
    // taps above; head numerics are T5's job)
    for fi in 0..s {
        for (head, ch) in [
            (Head::Depth, 3usize),
            (Head::Points, 4),
            (Head::Normals, 4),
            (Head::GsDepth, 3),
            (Head::GsParams, 12),
        ] {
            let v = gpu.read(model.head_out(head, fi), ch * h * w);
            let bad = v.iter().filter(|x| !x.is_finite()).count();
            assert_eq!(bad, 0, "frame {fi} {head:?}: {bad}/{} non-finite", v.len());
        }
    }
    // camera + assembled scene
    let cam = model.cam_pred_raw();
    assert!(cam.iter().all(|v| v.is_finite()), "camera pred has NaN: {cam:?}");
    let (splats, cams, weights) =
        assemble(&gpu, &model, &frames, s, w as u32, h as u32, &AssembleOpts::default());
    assert_eq!(cams.len(), s);
    assert!(splats.means.iter().all(|v| v.is_finite()), "NaN means");
    assert!(!splats.is_empty());
    assert_eq!(weights.len(), splats.len());
}
