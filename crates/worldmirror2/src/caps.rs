// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WorldMirror-2 multi-view 3D reconstruction behind the generalized
//! [`capability`] interface - what makes `brain caps` list this model and
//! `brain do worldmirror2 reconstruct …`, the D-Bus `Run` method and the
//! event API work with no WorldMirror-2-specific plumbing in the CLI or the
//! transports.
//!
//! Swedish Embedded AB implements discoverable, schedulable on-device model
//! serving for teams who need one uniform interface across a fleet of very
//! different architectures. If your team needs expertise in capability-driven
//! model serving then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! One action, `reconstruct`: N unposed RGB frames in (`images`, a
//! [`Media::Video`] blob - every frame shares one `(w,h)`, the same
//! constraint `crates/cli/src/mirror_cli.rs::load_frames`'s own
//! `assert!(... "mixed image sizes")` enforces today for the file-loading CLI
//! path), a Gaussian-splat scene out. This is a single feed-forward pass -
//! unlike `splat::caps::fit`, there is no iteration count to expose.
//!
//! `weights` carries [`ParamSpec::host_env`] (`BRAIN_WORLDMIRROR2_WEIGHTS`):
//! a checkpoint path is a fact about the machine that runs the action, never
//! a per-request parameter a remote caller could answer - see
//! [`manifest_resident`].
//!
//! # `Session`: one hot model, keyed on checkpoint identity
//!
//! [`Session`] holds one built [`Mirror`] across calls, rebuilding only when
//! the requested `weights` path changes - the direct-call analogue of
//! `glmdsa::caps::Hot` / `sam2::caps::Session`. `Mirror` itself is already
//! shape-ADAPTIVE (see `model.rs`'s own doc): it rebuilds its per-shape
//! buffers internally whenever `(frames, hp, wp)` changes, keeping the
//! ~5GB `ParamStore` resident across a shape change. So `Session` needs no
//! shape bookkeeping of its own - reusing one `Mirror` instance across an
//! arbitrary sequence of request shapes is exactly what its own `forward`
//! already does, which is also why `crates/cli/src/resident_worldmirror2.rs`
//! can serve every shape from ONE resident instance keyed on the checkpoint
//! alone (see that module's doc for the full argument, and its one caveat:
//! only one shape's buffers are cached at a time).
//!
//! [`load`] takes an already-built [`Gpu`] rather than building one itself
//! (the `sam2::caps::load` shape): a direct call builds one on the ambient
//! device, the resident adapter builds one on the scheduler-assigned device -
//! one import/construction path serves both.

use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use gpu_core::Gpu;
use serde_json::json;
use splat::types::Camera;

use crate::config::MirrorConfig;
use crate::gaussians::{assemble, frame_maps, AssembleOpts};
use crate::model::Mirror;

/// The model id used on the CLI (`brain do worldmirror2 …`), over D-Bus and
/// in the residency manifest.
pub const MODEL: &str = "brain/worldmirror2";

pub fn reconstruct_spec() -> ActionSpec {
    ActionSpec::new(
        "reconstruct",
        "multi-view 3D reconstruction: N unposed RGB frames in, a Gaussian-splat scene + per-frame cameras out (single feed-forward pass: DINOv2 + alternating-attention trunk + DPT/camera/Gaussian heads)",
    )
    .param(ParamSpec::new("weights", ParamType::Str, "path to a brain-format WorldMirror-2 checkpoint (.safetensors)").required().host_env("BRAIN_WORLDMIRROR2_WEIGHTS"))
    .param(ParamSpec::new("min_opacity", ParamType::Float, "drop gaussians below this opacity").default(json!(0.01)).min(0.0).max(1.0))
    .param(ParamSpec::new("max_depth", ParamType::Float, "clip gaussians past this depth, in scene units (0 = off)").default(json!(0.0)).min(0.0))
    .param(ParamSpec::new("prune_voxel", ParamType::Float, "voxel-merge duplicate gaussians at this edge length, in scene units (0 = off)").default(json!(0.0)).min(0.0))
    .param(ParamSpec::new("maps", ParamType::Bool, "also return a per-frame depth-map video (min-max normalized, replicated to RGB)").default(json!(false)))
    .input(BlobSpec::new(
        "images",
        Media::Video,
        "N frames, one per view, in the standard video wire convention (capability::blob::decode_video) - every frame shares one (w,h), which must be a multiple of the model's 14px patch grid",
    ).required())
    .output(BlobSpec::new("scene", Media::Bytes, "the reconstructed scene: Inria-layout binary PLY (splat::ply::serialize)"))
    .output(BlobSpec::new("maps", Media::Video, "per-frame depth map - present only when 'maps' is set"))
}

/// The full, static capability manifest - safe to build with no checkpoint on
/// disk (building it costs nothing; loading only happens inside `run`).
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "WorldMirror-2: single-pass multi-view 3D reconstruction (DINOv2 + alternating-attention trunk + DPT/camera/Gaussian heads) into a Gaussian-splat scene.",
        vec![reconstruct_spec()],
    )
}

/// The manifest for the RESIDENT/scheduled service (D-Bus, executor, HTTP):
/// the checkpoint is service-side configuration (`BRAIN_WORLDMIRROR2_WEIGHTS`),
/// so the action carries only request parameters. One line, following
/// `glmdsa::caps::manifest_resident`'s precedent: `weights` carries
/// [`ParamSpec::host_env`], and [`Manifest::for_serving`] projects every such
/// param out for every model at once.
pub fn manifest_resident() -> Manifest {
    manifest().for_serving()
}

/// One camera as the `cameras` output's JSON shape - the same fields
/// `mirror_cli.rs::write_cameras_json` and `splat::caps::fit`'s `views` param
/// use, so a client that already speaks one of those shapes speaks this one.
fn camera_json(c: &Camera) -> serde_json::Value {
    json!({"c2w": c.c2w.to_vec(), "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy, "width": c.width, "height": c.height})
}

/// Interleaved HWC f32 (the video-blob wire layout) to concatenated CHW - the
/// layout [`Mirror::forward`] expects. `pub` so
/// `tests/t9_caps_matches_direct.rs`'s own "direct" comparison path builds
/// its `Mirror::forward` input through the identical conversion this action
/// uses, rather than a second hand-written copy that could silently drift.
pub fn hwc_to_chw(hwc: &[f32], w: u32, h: u32) -> Vec<f32> {
    let (w, h) = (w as usize, h as usize);
    let mut chw = vec![0.0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                chw[c * w * h + y * w + x] = hwc[(y * w + x) * 3 + c];
            }
        }
    }
    chw
}

/// One frame's depth map, min-max normalized over ITS OWN pixels and
/// replicated to RGB - the same visualization
/// `mirror_cli.rs::write_maps` writes to a PPM, in `[0,1]` f32 rather than
/// `u8`. The `maps` output carries depth alone (one blob slot): a caller
/// that wants normals/confidence/mask too still has the direct CLI's
/// `--maps` PPMs.
fn depth_map_frame(gpu: &Gpu, model: &Mirror, fi: usize, w: u32, h: u32) -> Vec<f32> {
    let m = frame_maps(gpu, model, fi, w, h);
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &d in &m.depth {
        lo = lo.min(d);
        hi = hi.max(d);
    }
    let span = (hi - lo).max(1e-9);
    let mut rgb = Vec::with_capacity(m.depth.len() * 3);
    for &d in &m.depth {
        let v = ((d - lo) / span).clamp(0.0, 1.0);
        rgb.extend_from_slice(&[v, v, v]);
    }
    rgb
}

// ===================== Session: the hot model =====================

/// A built [`Mirror`] plus the checkpoint path that fixes it - see the module
/// doc for why one `Session` suffices across every request shape.
pub struct Session {
    weights: String,
    model: Mirror,
}

impl Session {
    /// Build directly from an already-constructed [`Mirror`] - what [`load`]
    /// (the production path, real checkpoint on disk) and this module's own
    /// tests use (a TINY synthetic-weight `Mirror`, so a caps-level test
    /// never needs the real ~5GB checkpoint).
    pub fn new(weights: impl Into<String>, model: Mirror) -> Session {
        Session { weights: weights.into(), model }
    }
}

/// Import `weights` under the default [`MirrorConfig`] and build a [`Mirror`]
/// on `gpu` (built by the caller - the ambient device for a direct call, the
/// scheduler-assigned device for `resident_worldmirror2.rs::activate` -
/// see the module doc).
pub fn load(weights: &str, gpu: Gpu) -> Result<Session, String> {
    let cfg = MirrorConfig::default();
    let init = crate::import::load_weights(weights, &cfg)?;
    let model = Mirror::new(gpu, cfg, &init, 0);
    Ok(Session::new(weights, model))
}

/// Run one `reconstruct` invocation against an ALREADY-BUILT `session` -
/// no `weights` handling here, deliberately: this is the one implementation
/// shared by [`reconstruct`] below (the direct/`brain do` path, which first
/// resolves-or-rebuilds `session` from the request's `weights`) and
/// `resident_worldmirror2.rs`'s `Instance::run` (which already IS the
/// session `activate` built once, on the scheduler-assigned device, and
/// whose served invocation carries no `weights` param at all - see
/// [`manifest_resident`]). Matches `sam2::caps::Session::segment`'s shape:
/// the resident calls straight into the model-owning session, never through
/// the `Action`/`Provider` layer that only the direct path needs.
pub fn run_reconstruct(session: &mut Session, inv: &Invocation) -> ActionResult {
    let min_opacity = inv.get_f64("min_opacity").unwrap_or(0.01) as f32;
    let max_depth = inv.get_f64("max_depth").unwrap_or(0.0) as f32;
    let prune_voxel = inv.get_f64("prune_voxel").unwrap_or(0.0) as f32;
    let want_maps = inv.get_bool("maps").unwrap_or(false);

    let frames = capability::blob::decode_video(inv, "images")?;
    let (_, w, h) = *frames.first().ok_or("worldmirror2 reconstruct: 'images' has no frames")?;

    // Every frame in a video blob structurally shares this one (w,h) - see
    // `capability::blob::decode_video`'s doc - but the model additionally
    // needs it to land on a whole number of patches. `mirror_cli.rs`'s file-
    // loading path guarantees this by resizing; a served request supplies
    // pixels directly, so THIS is the one geometry check standing between a
    // bad request and a panic deep inside `Mirror::forward`'s own
    // `assert_eq!` (mirror_cli.rs's own "mixed image sizes" panic point,
    // moved here as a clean `Err` - a one-shot CLI can afford to abort, a
    // resident process serving other concurrent requests cannot).
    let patch = session.model.cfg.patch;
    if w == 0 || h == 0 || !(w as usize).is_multiple_of(patch) || !(h as usize).is_multiple_of(patch) {
        return Err(format!(
            "worldmirror2 reconstruct: image size {w}x{h} is not a multiple of the {patch}px patch grid"
        ));
    }
    let (hp, wp) = (h as usize / patch, w as usize / patch);
    let s = frames.len();
    let mut frames_chw = Vec::with_capacity(s * 3 * h as usize * w as usize);
    for (hwc, fw, fh) in &frames {
        debug_assert_eq!((*fw, *fh), (w, h), "decode_video guarantees one shared (w,h)");
        frames_chw.extend(hwc_to_chw(hwc, w, h));
    }

    let model = &mut session.model;
    model.forward(&frames_chw, s, hp, wp);
    let opts = AssembleOpts { min_opacity, max_depth };
    let (mut splats, cams, gweights) = assemble(model.gpu(), model, &frames_chw, s, w, h, &opts);
    if prune_voxel > 0.0 {
        splats = splat::prune::voxel_merge(&splats, &gweights, prune_voxel, 0);
    }
    let ply_bytes = splat::ply::serialize(&splats)?;
    let cameras: Vec<serde_json::Value> = cams.iter().map(camera_json).collect();

    let mut out = Outcome::new().set("cameras", json!(cameras)).blob("scene", Blob::new(Media::Bytes, ply_bytes));
    if want_maps {
        let map_frames: Vec<(Vec<f32>, u32, u32)> =
            (0..s).map(|fi| (depth_map_frame(model.gpu(), model, fi, w, h), w, h)).collect();
        out = out.blob("maps", capability::blob::video_blob(&map_frames)?);
    }
    Ok(out)
}

/// Run one `reconstruct` invocation (already validated against
/// [`reconstruct_spec`]), rebuilding `hot` only when `weights` changed, then
/// deferring to [`run_reconstruct`] for the actual work.
fn reconstruct(inv: &Invocation, hot: &Mutex<Option<Session>>) -> ActionResult {
    let weights = inv.get_str("weights").unwrap_or_default();
    if weights.is_empty() {
        return Err("worldmirror2 reconstruct: 'weights' is required (path to a brain-format WorldMirror-2 checkpoint)".into());
    }
    let mut guard = hot.lock().map_err(|_| "worldmirror2 reconstruct: session lock poisoned")?;
    if guard.as_ref().map(|s| s.weights != weights).unwrap_or(true) {
        *guard = None; // free the old ~5GB ParamStore before allocating the new one
        *guard = Some(load(&weights, Gpu::new(crate::model::PIPELINES))?);
    }
    run_reconstruct(guard.as_mut().expect("built above"), inv)
}

// ===================== the provider =====================

/// The executable WorldMirror-2 model behind the manifest. Construction is
/// free - the checkpoint loads lazily on the first `reconstruct` call and
/// stays resident (see [`Session`]), so `brain caps` costs nothing and a
/// repeated `brain do worldmirror2 reconstruct` pays the load once.
#[derive(Default)]
pub struct WorldMirror2Provider {
    hot: Arc<Mutex<Option<Session>>>,
}

impl WorldMirror2Provider {
    pub fn new() -> WorldMirror2Provider {
        WorldMirror2Provider::default()
    }

    /// Seed the hot cache with an already-built [`Session`] - what
    /// `resident_worldmirror2.rs::activate` uses to hand the scheduler-built,
    /// device-pinned model straight in, and what this module's own tests use
    /// to drive `reconstruct` against a TINY synthetic-weight model without
    /// touching a real checkpoint.
    pub fn with_session(session: Session) -> WorldMirror2Provider {
        WorldMirror2Provider { hot: Arc::new(Mutex::new(Some(session))) }
    }
}

impl Provider for WorldMirror2Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "reconstruct" => Some(Arc::new(ReconstructAction { hot: self.hot.clone() }) as Arc<dyn Action>),
            _ => None,
        }
    }
}

struct ReconstructAction {
    hot: Arc<Mutex<Option<Session>>>,
}

impl Action for ReconstructAction {
    fn spec(&self) -> ActionSpec {
        reconstruct_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        reconstruct(inv, &self.hot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_declares_one_action_named_reconstruct() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 1);
        assert_eq!(m.actions[0].name, "reconstruct");
        assert!(!m.actions[0].streaming, "a single feed-forward pass is not a streaming action");
    }

    /// The resident surface advertises no `weights`: the executor supplies
    /// the checkpoint from `BRAIN_WORLDMIRROR2_WEIGHTS`, so a caller cannot
    /// set it and must not be told otherwise. Every other parameter survives.
    #[test]
    fn the_resident_manifest_drops_weights_and_keeps_the_rest() {
        let direct = &manifest().actions[0];
        let resident = &manifest_resident().actions[0];
        assert!(direct.params.iter().any(|p| p.name == "weights"));
        assert!(!resident.params.iter().any(|p| p.name == "weights"));
        let kept: Vec<&str> = resident.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(kept, vec!["min_opacity", "max_depth", "prune_voxel", "maps"]);
    }

    /// Construction must not touch the filesystem - `brain caps` builds every
    /// provider in the catalog, so a provider that loads weights eagerly
    /// makes listing capabilities cost a checkpoint read per model.
    #[test]
    fn a_missing_checkpoint_is_a_named_error_not_a_panic() {
        let p = WorldMirror2Provider::new();
        let action = p.action("reconstruct").expect("reconstruct exists");
        let err = action.run(&Invocation::default(), &mut |_| {}).unwrap_err();
        assert!(err.contains("weights"), "{err}");
    }

    #[test]
    fn an_unknown_action_is_none_and_reconstruct_resolves() {
        let p = WorldMirror2Provider::new();
        assert!(p.action("reconstruct").is_some());
        assert!(p.action("infer").is_none());
    }
}
