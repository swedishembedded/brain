// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T9 - the `reconstruct` capability action against the direct
//! `Mirror::forward` + `gaussians::assemble` call, at the same TINY
//! synthetic-weight `MirrorConfig`
//! `t8_multiframe_tiny.rs` uses (device-free, no real checkpoint).
//!
//! Two things this file proves, neither of which a source read can:
//!
//! 1. **caps == direct, bit-for-bit.** "Identical inputs" means identical PLY
//!    bytes on both sides, following `splat::caps`'s own precedent
//!    (`render_caps_matches_the_direct_renderer_path`'s doc): `scales`,
//!    `opacities` and `colors` round-trip through an activation/inverse-
//!    activation pair (`exp`/`ln`, `sigmoid`/`logit`) that PLY's own format
//!    requires, which is not guaranteed bit-exact for arbitrary floats even
//!    though it is exact for `means`/`quats` (copied verbatim). So the
//!    "direct" side is ALSO round-tripped through `splat::ply::serialize` +
//!    `parse` before comparing - the same deterministic function applied to
//!    the same numbers on both sides, which is what actually gets `k = 0`
//!    rather than a tolerance table.
//! 2. **One resident instance suffices across a shape change.** `Mirror` is
//!    shape-adaptive (rebuilds its own per-shape buffers on demand - see
//!    `model.rs`'s doc), so `resident_worldmirror2.rs` keys its ONE resident
//!    instance on checkpoint identity alone. Driving `reconstruct` at shape A,
//!    then B, then A again, all on ONE `WorldMirror2Provider` (one `Session`,
//!    one `Mirror`), and diffing every result against an independently-built
//!    single-shape run is what actually proves that suffices - not just that
//!    `Mirror::forward`'s rebuild-on-shape-change branch exists.

use std::collections::HashMap;

use capability::blob::video_blob;
use capability::{Blob, Invocation, Media, Outcome, Provider as _};
use data::rng::Lcg;
use serde_json::json;
use splat::types::{Camera, Splats};
use worldmirror2::caps::{Session, WorldMirror2Provider};
use worldmirror2::config::MirrorConfig;
use worldmirror2::gaussians::{assemble, AssembleOpts};
use worldmirror2::model::Mirror;

/// The exact tiny config `t8_multiframe_tiny.rs::s3_forward_is_finite` uses -
/// small enough to build+run on the CPU JIT backend in milliseconds.
fn tiny_cfg() -> MirrorConfig {
    MirrorConfig {
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
    }
}

/// Synthetic weights for `cfg`, deterministic in `seed` - same generation
/// shape `t8_multiframe_tiny.rs` uses (small values, norm/gamma rows near 1).
fn synthetic_init(cfg: &MirrorConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut r = Lcg::new(seed);
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, shape) in cfg.param_list() {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = if name.ends_with("norm.weight") || name.contains("norm1.weight") || name.contains("norm2.weight") || name.contains("ls") {
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

/// `s` synthetic RGB frames (interleaved HWC, `[0,1]`-ish), deterministic in
/// `seed` - the video-blob wire shape.
fn synthetic_frames(s: usize, hp: usize, wp: usize, patch: usize, seed: u64) -> Vec<(Vec<f32>, u32, u32)> {
    let (h, w) = (hp * patch, wp * patch);
    let mut r = Lcg::new(seed);
    (0..s)
        .map(|_| {
            // 8-bit exact, as a decoded photograph is: the action reads the
            // clip back as 8-bit RGB before it resizes to the model's grid.
            let hwc: Vec<f32> = (0..h * w * 3).map(|_| ((0.5 + 0.4 * r.scaled(0.5)) * 255.0).round() / 255.0).collect();
            (hwc, w as u32, h as u32)
        })
        .collect()
}

/// Count of f32 words whose `to_bits()` disagree - the same idiom
/// `splat::caps`'s own tests use.
fn differing_bits(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "differing_bits: length mismatch ({} vs {})", a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// Total differing bits and total floats compared, across every `Splats`
/// array (means, quats, scales, opacities, colors).
fn diff_splats(a: &Splats, b: &Splats) -> (usize, usize) {
    assert_eq!(a.len(), b.len(), "diff_splats: gaussian count mismatch ({} vs {})", a.len(), b.len());
    let bits = differing_bits(&a.means, &b.means)
        + differing_bits(&a.quats, &b.quats)
        + differing_bits(&a.scales, &b.scales)
        + differing_bits(&a.opacities, &b.opacities)
        + differing_bits(&a.colors, &b.colors);
    let n = a.means.len() + a.quats.len() + a.scales.len() + a.opacities.len() + a.colors.len();
    (bits, n)
}

/// The direct path: a fresh `Mirror` on its own `Gpu`, `forward` + `assemble` -
/// at request-time frames.
fn run_direct(cfg: MirrorConfig, init: &HashMap<String, Vec<f32>>, frames: &[(Vec<f32>, u32, u32)], s: usize, hp: usize, wp: usize) -> (Splats, Vec<Camera>) {
    let (w, h) = ((wp * cfg.patch) as u32, (hp * cfg.patch) as u32);
    let gpu = gpu_core::Gpu::new_cpu(worldmirror2::model::PIPELINES);
    let mut model = Mirror::new(gpu, cfg, init, 0);
    let chw: Vec<f32> = frames.iter().flat_map(|(hwc, fw, fh)| imaging::pixels::hwc_to_chw(hwc, 3, *fh as usize, *fw as usize)).collect();
    model.forward(&chw, s, hp, wp);
    let (splats, cams, _weights) = assemble(model.gpu(), &model, &chw, s, w, h, &AssembleOpts::default(), None);
    (splats, cams)
}

/// The direct path's `Splats`, round-tripped through the identical PLY
/// encode/decode the `reconstruct` action's wire format uses - see the module
/// doc for why this (not the raw pre-serialize `Splats`) is the correct
/// bit-exact comparison baseline.
fn direct_scene_via_ply(splats: &Splats) -> Splats {
    splat::ply::parse(&splat::ply::serialize(splats).unwrap()).unwrap()
}


/// The request that leaves the scene exactly as `assemble` returns it: no
/// duplicate fusion, no scale rejection, no re-framing.
fn raw_scene_request() -> Invocation {
    Invocation::new().set("prune", json!(0.0)).set("max_scale_quantile", json!(0.0)).set("keep_camera_frame", json!(true))
}

fn try_caps(provider: &WorldMirror2Provider, request: Invocation, frames: &[(Vec<f32>, u32, u32)]) -> Result<Outcome, String> {
    let action = provider.action("reconstruct").expect("reconstruct action");
    let video = video_blob(frames).expect("video_blob");
    let inv = action.spec().validate(request.blob("images", video)).expect("valid invocation");
    action.run(&inv, &mut |_| {})
}

/// Run `reconstruct` on `provider` (already seeded with a hot `Session`)
/// against `frames`, leaving the scene raw.
fn run_caps(provider: &WorldMirror2Provider, frames: &[(Vec<f32>, u32, u32)]) -> Outcome {
    try_caps(provider, raw_scene_request(), frames).expect("reconstruct")
}

fn scene_from_outcome(out: &Outcome) -> Splats {
    let bytes = &out.blobs.get("scene").expect("scene blob").bytes;
    splat::ply::parse(bytes).expect("valid PLY")
}

/// (1): the `reconstruct` action reproduces `Mirror::forward` +
/// `gaussians::assemble` called directly, bit-for-bit, on identical inputs -
/// and the `cameras` blob matches the directly-decoded cameras exactly.
#[test]
fn caps_matches_direct_bit_for_bit() {
    let cfg = tiny_cfg();
    let init = synthetic_init(&cfg, 0x5EED);
    let (s, hp, wp) = (1usize, 4usize, 4usize); // native 4x4 grid, single frame
    let frames = synthetic_frames(s, hp, wp, cfg.patch, 0xF00D);

    let (direct_splats, direct_cams) = run_direct(cfg.clone(), &init, &frames, s, hp, wp);
    let direct_scene = direct_scene_via_ply(&direct_splats);

    let gpu = gpu_core::Gpu::new_cpu(worldmirror2::model::PIPELINES);
    let model = Mirror::new(gpu, cfg, &init, 0);
    let provider = WorldMirror2Provider::with_session(Session::new("t9-single", model));
    let out = run_caps(&provider, &frames);
    let caps_scene = scene_from_outcome(&out);

    let (bits, n) = diff_splats(&direct_scene, &caps_scene);
    println!("caps-vs-direct: {bits} differing bits over {n} floats");
    assert_eq!(bits, 0, "the reconstruct action must reproduce the direct forward+assemble path exactly");
    assert!(!caps_scene.is_empty(), "a tiny random-weight forward produced zero gaussians");

    let served = splat::types::cameras_from_json(std::str::from_utf8(&out.blobs["cameras"].bytes).unwrap()).expect("cameras blob is the cameras.json shape");
    assert_eq!(served.len(), direct_cams.len());
    for (got, want) in served.iter().zip(&direct_cams) {
        assert_eq!(differing_bits(&got.c2w, &want.c2w), 0, "served pose must match the directly-decoded pose exactly");
        assert_eq!((got.fx, got.fy, got.cx, got.cy, got.width, got.height), (want.fx, want.fy, want.cx, want.cy, want.width, want.height));
    }
}

/// (2): ONE resident `Session`/`Mirror`, driven at shape A, then B, then A
/// again, must reproduce each shape's own INDEPENDENTLY-built single-shape
/// run bit-for-bit - proving `Mirror`'s internal rebuild-on-shape-change is
/// what lets `resident_worldmirror2.rs` key on checkpoint identity alone
/// (see that module's doc), not merely that the rebuild branch exists.
#[test]
fn shape_cycle_one_instance_matches_independent_single_shape_runs() {
    let cfg = tiny_cfg();
    let init = synthetic_init(&cfg, 0x5EED);
    // Shape A: native 4x4 grid, 1 frame. Shape B: non-native 3x5 grid (forces
    // the pos-embed bicubic-interpolation path - see model.rs::build), 2
    // frames - both a grid AND a frame-count change from A.
    let shapes: [(usize, usize, usize, u64); 2] = [(1, 4, 4, 0xAAAA), (2, 3, 5, 0xBBBB)];

    let reference: Vec<Splats> = shapes
        .iter()
        .map(|&(s, hp, wp, seed)| {
            let frames = synthetic_frames(s, hp, wp, cfg.patch, seed);
            let (splats, _cams) = run_direct(cfg.clone(), &init, &frames, s, hp, wp);
            direct_scene_via_ply(&splats)
        })
        .collect();

    let gpu = gpu_core::Gpu::new_cpu(worldmirror2::model::PIPELINES);
    let model = Mirror::new(gpu, cfg.clone(), &init, 0);
    let provider = WorldMirror2Provider::with_session(Session::new("t9-cycle", model));

    let mut total_bits = 0usize;
    for &shape_idx in &[0usize, 1, 0] {
        let (s, hp, wp, seed) = shapes[shape_idx];
        let frames = synthetic_frames(s, hp, wp, cfg.patch, seed);
        let out = run_caps(&provider, &frames);
        let got = scene_from_outcome(&out);
        let (bits, _n) = diff_splats(&reference[shape_idx], &got);
        total_bits += bits;
    }
    println!("shape-cycle: {total_bits} differing bits");
    assert_eq!(total_bits, 0, "one instance cycling shapes must match each shape's own independent single-shape run exactly");
}

/// The pass resizes whatever it is handed to the model's patch grid, so a
/// frame size that is not a multiple of it is a request like any other, and
/// the scene is built on the resized grid.
#[test]
fn a_frame_size_off_the_patch_grid_is_resized_not_rejected() {
    let cfg = tiny_cfg();
    let init = synthetic_init(&cfg, 0x5EED);
    let model = Mirror::new(gpu_core::Gpu::new_cpu(worldmirror2::model::PIPELINES), cfg, &init, 0);
    let provider = WorldMirror2Provider::with_session(Session::new("t9-off-grid", model));
    // 60x44 is no multiple of 14: the grid it lands on is 56x42 (4x3 patches).
    let frames = vec![(vec![0.5f32; 60 * 44 * 3], 60u32, 44u32)];
    let out = try_caps(&provider, raw_scene_request(), &frames).expect("an off-grid size is resized");
    assert_eq!((out.outputs["width"].as_u64(), out.outputs["height"].as_u64()), (Some(56), Some(42)));
}

/// `maps` and `frames_out` return one clip frame per selected view at the
/// model's resolution, and nothing is returned that was not asked for.
#[test]
fn maps_and_conformed_frames_are_returned_only_on_request() {
    let cfg = tiny_cfg();
    let init = synthetic_init(&cfg, 0x5EED);
    let model = Mirror::new(gpu_core::Gpu::new_cpu(worldmirror2::model::PIPELINES), cfg.clone(), &init, 0);
    let provider = WorldMirror2Provider::with_session(Session::new("t9-maps", model));
    let frames = synthetic_frames(2, 4, 4, cfg.patch, 0xCAFE);

    let plain = run_caps(&provider, &frames);
    assert!(["depth", "normals", "frames"].iter().all(|k| !plain.blobs.contains_key(*k)));

    let asked = try_caps(&provider, raw_scene_request().set("maps", json!(true)).set("frames_out", json!(true)), &frames).unwrap();
    for name in ["depth", "normals", "frames"] {
        let b = &asked.blobs[name];
        assert_eq!(b.media, Media::Video, "{name}");
        assert_eq!((b.meta["frames"].as_u64(), b.meta["w"].as_u64(), b.meta["h"].as_u64()), (Some(2), Some(56), Some(56)), "{name}");
    }
    // the frames the model saw are the frames it was given, at its own grid
    let seen = capability::blob::decode_video(&Invocation::new().blob("v", asked.blobs["frames"].clone()), "v").unwrap();
    assert_eq!(seen[0].0, frames[0].0, "a frame already on the grid comes back unchanged");
}

/// Known cameras must be one per selected frame; the run is refused before
/// any weight work when they are not.
#[test]
fn a_pose_prior_that_does_not_match_the_frames_is_refused() {
    let cfg = tiny_cfg();
    let init = synthetic_init(&cfg, 0x5EED);
    let model = Mirror::new(gpu_core::Gpu::new_cpu(worldmirror2::model::PIPELINES), cfg.clone(), &init, 0);
    let provider = WorldMirror2Provider::with_session(Session::new("t9-poses", model));
    let frames = synthetic_frames(2, 4, 4, cfg.patch, 0xCAFE);
    let action = provider.action("reconstruct").unwrap();
    let one_camera = json!([{ "c2w": vec![0.0f32; 16] }]).to_string();
    let video = video_blob(&frames).unwrap();
    let inv = action.spec().validate(raw_scene_request().blob("images", video).blob("poses", Blob::new(Media::Text, one_camera.into_bytes()))).unwrap();
    let err = action.run(&inv, &mut |_| {}).unwrap_err();
    assert!(err.contains("per frame"), "{err}");
}

/// A cancellation requested before the pass starts is honoured, not ignored.
#[test]
fn a_cancelled_request_does_not_run() {
    let cfg = tiny_cfg();
    let init = synthetic_init(&cfg, 0x5EED);
    let model = Mirror::new(gpu_core::Gpu::new_cpu(worldmirror2::model::PIPELINES), cfg.clone(), &init, 0);
    let provider = WorldMirror2Provider::with_session(Session::new("t9-cancel", model));
    let frames = synthetic_frames(1, 4, 4, cfg.patch, 0xCAFE);
    let action = provider.action("reconstruct").unwrap();
    let mut inv = action.spec().validate(raw_scene_request().blob("images", video_blob(&frames).unwrap())).unwrap();
    inv.cancel = capability::CancelToken::armed();
    inv.cancel.cancel();
    assert_eq!(action.run(&inv, &mut |_| {}).unwrap_err(), "cancelled");
}
