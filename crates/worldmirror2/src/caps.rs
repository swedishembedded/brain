// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WorldMirror-2 multi-view 3D reconstruction behind the generalized
//! [`capability`] interface - what makes `brain caps` list this model and
//! `brain worldmirror2 <action> ...`, the D-Bus `Run` method and the event API
//! work with no WorldMirror-2-specific plumbing in the transports.
//!
//! Swedish Embedded AB implements discoverable, schedulable on-device model
//! serving for teams who need one uniform interface across a fleet of very
//! different architectures. If your team needs expertise in capability-driven
//! model serving then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Two actions:
//!
//! * **`reconstruct`** - N unposed RGB frames in (`images`, a clip: a
//!   directory of frames or a video), a Gaussian-splat scene and the camera of
//!   every frame out, in one feed-forward pass. A long capture is narrowed to
//!   the frames the model sees (`stride`, `max_frames`, `fps`), a turntable's
//!   background is masked away (`mask`), known cameras condition the trunk
//!   (`poses`), and every scene-assembly choice the reference makes is a
//!   parameter. `maps` and `frames_out` add the per-frame depth and normal
//!   maps and the frames exactly as the model saw them.
//! * **`import`** - convert the reference checkpoint once into brain's own
//!   `.safetensors` container (config embedded, layout verified). It runs on
//!   files on the serving machine, so both of its parameters are
//!   [`ParamSpec::host_resolved`].
//!
//! # `Session`: one hot model, keyed on checkpoint identity
//!
//! [`Session`] holds one built [`Mirror`] across calls, rebuilding only when
//! the checkpoint changes. `Mirror` itself is shape-ADAPTIVE (see `model.rs`):
//! it rebuilds its per-shape buffers whenever `(frames, hp, wp)` changes while
//! the ~5GB `ParamStore` stays resident. So `Session` needs no shape
//! bookkeeping, which is also why the resident adapter serves every shape
//! from ONE instance keyed on the checkpoint alone (only one shape's buffers
//! are cached at a time).
//!
//! [`load`] takes an already-built [`Gpu`]: a direct call builds one on the
//! ambient device, the resident adapter builds one on the scheduler-assigned
//! device - one import/construction path serves both.

use std::sync::Arc;

use capability::bounds::{FPS, SIDE_PX, UNIT, VIDEO_FRAMES};
use capability::{
    hot::Hot, Action, ActionResult, ActionSpec, Assembly, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use gpu_core::Gpu;
use serde_json::json;

use crate::config::MirrorConfig;
use crate::frames::{self, FrameSel};
use crate::gaussians::{frame_maps, AssembleOpts};
use crate::model::Mirror;
use crate::scene::{self, SceneOpts};

/// The model id used by `brain do`, the generic runner, over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "brain/worldmirror2";

/// Ceiling on a distance in scene units (a clip depth, a voxel edge). The
/// scene is in the first camera's units, which are the model's own, so this is
/// only a sanity ceiling well past any scene.
const MAX_SCENE_UNITS: f64 = 1_000_000.0;
/// Ceiling on `surface_align`, a thickness ratio between a gaussian's axes.
const MAX_ALIGN_RATIO: f64 = 64.0;
/// Ceiling on `conf_percentile`, a percentage of pixels.
const PERCENT: f64 = 100.0;

pub fn reconstruct_spec() -> ActionSpec {
    let source = |values: &[&str]| ParamType::Enum(values.iter().map(|v| v.to_string()).collect());
    ActionSpec::new(
        "reconstruct",
        "multi-view 3D reconstruction: N unposed RGB frames in, a Gaussian-splat scene + per-frame cameras out (single feed-forward pass: DINOv2 + alternating-attention trunk + DPT/camera/Gaussian heads)",
    ).cancellable()
    .param(ParamSpec::new("stride", ParamType::Int, "use every Nth frame of the capture").default(json!(1)).min(1.0).max(VIDEO_FRAMES))
    .param(ParamSpec::new("max_frames", ParamType::Int, "use at most this many frames, spread ACROSS the capture rather than taking a prefix (0 = a clip with a frame rate of its own gets 48, a set of stills is used whole)").default(json!(0)).min(0.0).max(VIDEO_FRAMES))
    .param(ParamSpec::new("fps", ParamType::Float, "resample a clip to this many frames per second before selecting (0 = the clip's own rate; needs a clip that states its frame rate)").default(json!(0.0)).min(0.0).max(FPS))
    // perf-number: derived from the two input sizes, (952/518)^2, not a measurement
    .param(ParamSpec::new("target_size", ParamType::Int, "longest-edge cap, in pixels, the frames are resized to before the model (floored to the 14px patch grid; any aspect ratio); the reference's inference default, about 3.4x the samples of the native 518").default(json!(952)).min(28.0).max(SIDE_PX))
    .param(ParamSpec::new("min_opacity", ParamType::Float, "drop gaussians below this opacity").default(json!(0.01)).min(0.0).max(UNIT))
    .param(ParamSpec::new("max_depth", ParamType::Float, "clip gaussians past this depth, in scene units (0 = off)").default(json!(0.0)).min(0.0).max(MAX_SCENE_UNITS))
    .param(ParamSpec::new("prune", ParamType::Float, "voxel-merge duplicate gaussians at this edge length, in scene units (0 = off); collapses the near-duplicate surfaces overlapping views predict at slightly different depths").default(json!(0.002)).min(0.0).max(MAX_SCENE_UNITS))
    .param(ParamSpec::new("gs_mask_threshold", ParamType::Float, "keep a pixel only where the Gaussian head's own validity mask is above this (the reference thresholds at 0.5; 0 = off)").default(json!(0.5)).min(0.0).max(UNIT))
    .param(ParamSpec::new("edge_depth_threshold", ParamType::Float, "relative depth step between neighbouring pixels above which a pixel is a silhouette edge and is dropped, since its depth blends two surfaces (reference 0.03; 0 = off)").default(json!(0.03)).min(0.0).max(UNIT))
    .param(ParamSpec::new("fuse_depth", ParamType::Float, "reconcile the per-view depth maps against each other before they become geometry, rejecting pairs that differ by more than this fraction as occlusions (0 = off)").default(json!(0.05)).min(0.0).max(UNIT))
    .param(ParamSpec::new("min_support", ParamType::Int, "keep a pixel only where at least this many OTHER frames agree a surface is there (0 = keep everything)").default(json!(0)).min(0.0).max(VIDEO_FRAMES))
    .param(ParamSpec::new("conf_percentile", ParamType::Float, "drop this percentage of pixels, the least confident first, by the depth head's own confidence, globally across frames (0 = keep everything)").default(json!(0.0)).min(0.0).max(PERCENT))
    .param(ParamSpec::new("surface_align", ParamType::Float, "lay each gaussian flat against its surface at this thickness ratio between its axes (0 = keep the model's own orientation)").default(json!(4.0)).min(0.0).max(MAX_ALIGN_RATIO))
    .param(ParamSpec::new("position_from", source(&["gsdepth", "points", "blend"]), "which head places the gaussians: the Gaussian branch's depth, the pointmap head, or a confidence-weighted blend of the two").default(json!("gsdepth")))
    .param(ParamSpec::new("normals_from", source(&["geometry", "head"]), "where the surface normal behind surface_align comes from: the depth map's own geometry, or the normals head").default(json!("geometry")))
    .param(ParamSpec::new("max_scale_quantile", ParamType::Float, "drop the gaussians whose scale is above this quantile of the scene's (0 or 1 = off)").default(json!(0.98)).min(0.0).max(UNIT))
    .param(ParamSpec::new("keep_camera_frame", ParamType::Bool, "return the scene in frame 0's own frame instead of re-framing it upright (vertical = the axis the cameras sweep about, origin = where they look); needs 3 or more frames to re-frame at all").default(json!(false)))
    .param(ParamSpec::new("maps", ParamType::Bool, "also return the per-frame depth and normal maps as the 'depth' and 'normals' clips").default(json!(false)))
    .param(ParamSpec::new("frames_out", ParamType::Bool, "also return the frames exactly as the model saw them, at the resolution it saw them, as the 'frames' clip - what a later fit must be scored against, since the target size picks its own grid").default(json!(false)))
    .input(BlobSpec::new(
        "images",
        Media::Video,
        "the capture, N views of one rigid scene: a directory of frames or a video; frames are resized to the model's grid, so any size works as long as every frame shares one",
    ).required().clip())
    .input(BlobSpec::new("mask", Media::Mask, "keep/drop mask applied to every frame (white keeps), nearest-neighbour resized to the frame: for TURNTABLE captures, where a rotating object in front of a still background contradicts the rigid-scene assumption; mask the background away"))
    .input(BlobSpec::new("poses", Media::Text, "known cameras, one per selected frame, as JSON in the shape this action's 'cameras' output has: conditions the trunk and becomes the back-projection cameras; WorldMirror-2 is an any-prior model"))
    .output(BlobSpec::new("scene", Media::Bytes, "the reconstructed scene: Inria-layout binary PLY (splat::ply::serialize)"))
    .output(BlobSpec::new("cameras", Media::Text, "the camera of every frame, as JSON in the shape splat's 'fit' takes as 'views' - in the returned scene's frame"))
    .output(BlobSpec::new("depth", Media::Video, "per-frame depth map, min-max normalized over its own pixels and replicated to RGB - present only when 'maps' is set"))
    .output(BlobSpec::new("normals", Media::Video, "per-frame normal map, unit normals mapped to RGB - present only when 'maps' is set"))
    .output(BlobSpec::new("frames", Media::Video, "the frames as the model saw them - present only when 'frames_out' is set"))
}

pub fn import_spec() -> ActionSpec {
    ActionSpec::new(
        "import",
        "convert the reference WorldMirror-2 checkpoint once into brain's own .safetensors container: config embedded, every tensor shape-verified against the model, streamed one tensor at a time",
    ).cancellable()
    .from_scratch()
    .param(ParamSpec::new("source", ParamType::Str, "the reference model.safetensors, or a Hugging Face directory holding it, on the serving machine").required().host_resolved())
    .param(ParamSpec::new("dest", ParamType::Str, "where to write the converted .safetensors, on the serving machine").required().host_resolved())
}

/// The full, static capability manifest - safe to build with no checkpoint on
/// disk (building it costs nothing; loading only happens inside `run`).
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "WorldMirror-2: single-pass multi-view 3D reconstruction (DINOv2 + alternating-attention trunk + DPT/camera/Gaussian heads) into a Gaussian-splat scene.",
        vec![reconstruct_spec(), import_spec()],
    )
}

/// The manifest the residency scheduler serves: reconstruction only.
/// Converting a checkpoint is a one-off operation on files of the host, not a
/// request a scheduled device instance answers.
pub fn manifest_resident() -> Manifest {
    let mut m = manifest();
    m.actions.retain(|a| a.name == "reconstruct");
    m
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
/// scheduler-assigned device for the resident adapter).
pub fn load(weights: &str, gpu: Gpu) -> Result<Session, String> {
    let cfg = MirrorConfig::default();
    let init = crate::import::load_weights(weights, &cfg)?;
    let model = Mirror::new(gpu, cfg, &init, 0);
    Ok(Session::new(weights, model))
}

// ===================== reconstruct =====================

/// Everything a `reconstruct` request sets, read once from the validated
/// invocation.
struct Request {
    sel: FrameSel,
    target: usize,
    scene: SceneOpts,
    maps: bool,
    frames_out: bool,
}

impl Request {
    fn read(inv: &Invocation) -> Result<Request, String> {
        let f = |name: &str, default: f64| inv.get_f64(name).unwrap_or(default) as f32;
        let n = |name: &str, default: i64| inv.get_i64(name).unwrap_or(default).max(0) as usize;
        let choice = |name: &str, default: &str| inv.get_str(name).unwrap_or_else(|| default.to_string());
        let defaults = AssembleOpts::default();
        let assemble = AssembleOpts {
            min_opacity: f("min_opacity", defaults.min_opacity as f64),
            max_depth: f("max_depth", 0.0),
            gs_mask_threshold: f("gs_mask_threshold", defaults.gs_mask_threshold as f64),
            edge_depth_rtol: f("edge_depth_threshold", defaults.edge_depth_rtol as f64),
            fuse_depth_rtol: f("fuse_depth", defaults.fuse_depth_rtol as f64),
            min_support: n("min_support", 0).min(u16::MAX as usize) as u16,
            conf_percentile: f("conf_percentile", 0.0),
            surface_align: f("surface_align", defaults.surface_align as f64),
            position_from: scene::position_source(&choice("position_from", "gsdepth"))?,
            normals_from: scene::normal_source(&choice("normals_from", "geometry"))?,
        };
        let scene_defaults = SceneOpts::default();
        Ok(Request {
            sel: FrameSel { stride: n("stride", 1), max: n("max_frames", 0), fps: inv.get_f64("fps").unwrap_or(0.0) },
            target: n("target_size", 952),
            scene: SceneOpts {
                assemble,
                prune_voxel: f("prune", scene_defaults.prune_voxel as f64),
                max_scale_quantile: f("max_scale_quantile", scene_defaults.max_scale_quantile as f64),
                keep_camera_frame: inv.get_bool("keep_camera_frame").unwrap_or(false),
            },
            maps: inv.get_bool("maps").unwrap_or(false),
            frames_out: inv.get_bool("frames_out").unwrap_or(false),
        })
    }
}

/// The frames the model will see: decoded from the `images` clip, narrowed by
/// the request's selection, masked.
fn gather(inv: &Invocation, sel: &FrameSel) -> Result<Vec<imaging::Rgb8>, String> {
    let clip = capability::blob::decode_video(inv, "images")?;
    let source_fps = inv.get_blob("images").and_then(|b| b.meta.get("fps")).and_then(|v| v.as_f64());
    let keep = sel.indices(clip.len(), source_fps).map_err(|e| format!("worldmirror2 reconstruct: {e}"))?;
    let mask = match inv.get_blob("mask") {
        Some(_) => {
            let (hwc, w, h, c) = capability::blob::decode_hwc(inv, "mask")?;
            Some(frames::rgb8_from_unit_hwc(&hwc, w, h, c)?)
        }
        None => None,
    };
    keep.into_iter()
        .map(|i| {
            let (hwc, w, h) = &clip[i];
            let mut img = frames::rgb8_from_unit_hwc(hwc, *w, *h, 3)?;
            if let Some(m) = &mask {
                frames::apply_mask(&mut img, m);
            }
            Ok(img)
        })
        .collect()
}

/// One frame of a clip in the wire convention `video_blob` takes: interleaved
/// HWC `[0,1]` RGB with its width and height.
type ClipFrame = (Vec<f32>, u32, u32);

/// One frame's depth and normal maps as RGB clip frames. Depth is min-max
/// normalized over ITS OWN pixels and replicated to RGB; normals are unit
/// vectors mapped from `[-1,1]` to `[0,1]`.
fn map_frames(model: &Mirror, fi: usize, w: u32, h: u32) -> (ClipFrame, ClipFrame) {
    let m = frame_maps(model.gpu(), model, fi, w, h);
    let hw = (w * h) as usize;
    let (lo, hi) = m.depth.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &d| (lo.min(d), hi.max(d)));
    let span = (hi - lo).max(1e-9);
    let depth = m.depth.iter().flat_map(|&d| [((d - lo) / span).clamp(0.0, 1.0); 3]).collect();
    let normals = (0..hw).flat_map(|i| (0..3).map(move |c| (c, i))).map(|(c, i)| (m.normals[c * hw + i] * 0.5 + 0.5).clamp(0.0, 1.0)).collect();
    ((depth, w, h), (normals, w, h))
}

/// Run one `reconstruct` invocation against an ALREADY-BUILT `session` - no
/// `weights` handling here, deliberately: this is the one implementation shared
/// by [`WorldMirror2Provider`]'s action (which resolves-or-rebuilds the session
/// from its checkpoint) and the resident adapter (whose instance IS the session
/// `activate` built, on the scheduler-assigned device). Matches
/// `sam2::caps::Session::segment`'s shape.
///
/// The pass itself cannot be interrupted, so cancellation is honoured between
/// its stages: before the trunk runs, before assembly, and between map frames.
pub fn run_reconstruct(session: &mut Session, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    inv.cancel.check()?;
    let req = Request::read(inv).map_err(|e| format!("worldmirror2 reconstruct: {e}"))?;
    let model = &mut session.model;
    let cfg = model.cfg.clone();

    progress(Progress::step(0, 3, "preparing frames"));
    let images = gather(inv, &req.sel)?;
    let conformed = frames::conform(&images, &cfg, req.target).map_err(|e| format!("worldmirror2 reconstruct: {e}"))?;
    drop(images);
    let (s, size) = (conformed.frames, conformed.size(&cfg));
    let (w, h) = size;

    // What a shape actually costs in ONE binding. Global attention is
    // query-chunked, so this is not the token count squared.
    let limit = model.gpu().max_storage_binding_bytes();
    let need = crate::model::largest_binding_bytes(&cfg, s, conformed.hp, conformed.wp);
    if limit > 0 && need > limit {
        return Err(format!(
            "worldmirror2 reconstruct: {s} frame(s) at {w}x{h} need {:.2} GiB in one storage binding against this device's {:.2} GiB limit; lower target_size or use fewer frames (max_frames, stride)",
            need as f64 / (1u64 << 30) as f64,
            limit as f64 / (1u64 << 30) as f64
        ));
    }
    let priors = match inv.get_blob("poses") {
        Some(b) => {
            let text = std::str::from_utf8(&b.bytes).map_err(|_| "worldmirror2 reconstruct: 'poses' is not UTF-8 JSON".to_string())?;
            Some(crate::priors::parse_cameras(text, s).map_err(|e| format!("worldmirror2 reconstruct: 'poses': {e}"))?)
        }
        None => None,
    };

    inv.cancel.check()?;
    progress(Progress::step(1, 3, format!("running WorldMirror-2 on {s} frame(s) at {w}x{h}")));
    let known = scene::forward(model, &conformed.chw, s, size, priors.as_deref());

    inv.cancel.check()?;
    progress(Progress::step(2, 3, "assembling the scene"));
    let built = scene::build(model, &conformed.chw, s, size, known.as_deref(), &req.scene);
    let ply = splat::ply::serialize(&built.splats)?;

    let mut out = Outcome::new()
        .set("gaussians", json!(built.splats.len()))
        .set("frames", json!(s))
        .set("width", json!(w))
        .set("height", json!(h))
        .blob("scene", Blob::new(Media::Bytes, ply))
        .blob("cameras", Blob::new(Media::Text, splat::types::cameras_to_json(&built.cameras).into_bytes()));
    if req.maps {
        let (mut depth, mut normals) = (Vec::with_capacity(s), Vec::with_capacity(s));
        for fi in 0..s {
            inv.cancel.check()?;
            let (d, n) = map_frames(model, fi, w, h);
            depth.push(d);
            normals.push(n);
        }
        out = out.blob("depth", capability::blob::video_blob(&depth)?).blob("normals", capability::blob::video_blob(&normals)?);
    }
    if req.frames_out {
        let (hw, planes) = ((w * h) as usize, 3);
        let seen: Vec<ClipFrame> = (0..s)
            .map(|f| (imaging::pixels::chw_to_hwc(&conformed.chw[f * planes * hw..(f + 1) * planes * hw], planes, h as usize, w as usize), w, h))
            .collect();
        out = out.blob("frames", capability::blob::video_blob(&seen)?);
    }
    progress(Progress::step(3, 3, format!("{} gaussians", built.splats.len())));
    Ok(out)
}

// ===================== import =====================

fn run_import(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    inv.cancel.check()?;
    let (source, dest) = (inv.get_str("source").unwrap_or_default(), inv.get_str("dest").unwrap_or_default());
    if source.is_empty() || dest.is_empty() {
        return Err("worldmirror2 import: 'source' and 'dest' are both required".to_string());
    }
    let cfg = MirrorConfig::default();
    progress(Progress::step(0, 1, format!("importing {source} ({} tensors expected)", cfg.param_list().len())));
    let tensors = crate::import::convert(&source, &dest, &cfg).map_err(|e| format!("worldmirror2 import: {e}"))?;
    progress(Progress::step(1, 1, format!("wrote {dest}")));
    Ok(Outcome::new().set("tensors", json!(tensors)).set("dest", json!(dest)))
}

// ===================== the provider =====================

/// The executable WorldMirror-2 model behind the manifest. Construction is
/// free - the checkpoint loads lazily on the first `reconstruct` call and
/// stays resident (see [`Session`]), so `brain caps` costs nothing and repeated
/// calls pay the load once.
#[derive(Default)]
pub struct WorldMirror2Provider {
    weights: Option<String>,
    hot: Arc<Hot<String, Session>>,
}

impl WorldMirror2Provider {
    /// A provider over the checkpoint at `weights`.
    pub fn new(weights: impl Into<String>) -> WorldMirror2Provider {
        WorldMirror2Provider { weights: Some(weights.into()), hot: Arc::default() }
    }

    /// A provider with no checkpoint: `import` works, `reconstruct` says what
    /// is missing. What a run that has not resolved weights gets.
    pub fn without_weights() -> WorldMirror2Provider {
        WorldMirror2Provider::default()
    }

    /// Seed the hot cache with an already-built [`Session`] - what this
    /// module's own tests use to drive `reconstruct` against a TINY
    /// synthetic-weight model without touching a real checkpoint.
    pub fn with_session(session: Session) -> WorldMirror2Provider {
        WorldMirror2Provider { weights: Some(session.weights.clone()), hot: Arc::new(Hot::with(session.weights.clone(), session)) }
    }
}

/// The provider over a resolved [`Assembly`] (its `weights` role). An assembly
/// with no weights still builds one, so `import` can create the checkpoint
/// every other action needs.
pub fn provider(assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(match assembly.role_path("weights") {
        Ok(weights) => WorldMirror2Provider::new(weights),
        Err(_) => WorldMirror2Provider::without_weights(),
    }))
}

impl Provider for WorldMirror2Provider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "reconstruct" => Some(Arc::new(ReconstructAction { weights: self.weights.clone(), hot: self.hot.clone() }) as Arc<dyn Action>),
            "import" => Some(Arc::new(ImportAction)),
            _ => None,
        }
    }
}

struct ReconstructAction {
    weights: Option<String>,
    hot: Arc<Hot<String, Session>>,
}

impl Action for ReconstructAction {
    fn spec(&self) -> ActionSpec {
        reconstruct_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let weights = self.weights.clone().ok_or("worldmirror2 reconstruct: no checkpoint: put one in the model store, or name it with --weights (`import` converts the reference release)")?;
        self.hot.try_get_mut(weights.clone(), || load(&weights, Gpu::new(crate::model::PIPELINES)), |s| run_reconstruct(s, inv, progress))
    }
}

struct ImportAction;

impl Action for ImportAction {
    fn spec(&self) -> ActionSpec {
        import_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        run_import(inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_declares_reconstruct_and_import() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["reconstruct", "import"]);
        assert!(!m.actions[0].streaming, "a single feed-forward pass is not a streaming action");
    }

    /// The scheduler serves reconstruction only; converting a checkpoint is
    /// not a device job.
    #[test]
    fn the_resident_manifest_serves_reconstruction_alone() {
        let names: Vec<String> = manifest_resident().actions.into_iter().map(|a| a.name).collect();
        assert_eq!(names, vec!["reconstruct"]);
    }

    /// Every number a caller sets has a ceiling, and the paths of `import`
    /// are host-resolved, so a remote caller can never name a file.
    #[test]
    fn every_number_is_bounded_and_every_path_is_host_resolved() {
        for a in manifest().actions {
            for p in &a.params {
                if matches!(p.ty, ParamType::Int | ParamType::Float) {
                    assert!(p.max.is_some(), "{}.{} has no max", a.name, p.name);
                }
                if p.ty == ParamType::Str {
                    assert!(p.host_resolved, "{}.{} is a string a remote caller could point at a file", a.name, p.name);
                }
            }
        }
        assert!(manifest_resident().actions[0].params.iter().all(|p| p.ty != ParamType::Str));
    }

    /// Every flag of the reference command line is a parameter, input or
    /// output of the action, so nothing the old verb could do is unreachable.
    #[test]
    fn reconstruct_carries_every_knob_of_the_reference_command_line() {
        let spec = reconstruct_spec();
        let params: Vec<&str> = spec.params.iter().map(|p| p.name.as_str()).collect();
        for knob in [
            "stride", "max_frames", "fps", "target_size", "min_opacity", "max_depth", "prune", "gs_mask_threshold", "edge_depth_threshold", "fuse_depth",
            "min_support", "conf_percentile", "surface_align", "position_from", "normals_from", "max_scale_quantile", "keep_camera_frame", "maps", "frames_out",
        ] {
            assert!(params.contains(&knob), "missing param {knob}");
        }
        let inputs: Vec<&str> = spec.inputs.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(inputs, vec!["images", "mask", "poses"]);
        let outputs: Vec<&str> = spec.outputs.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(outputs, vec!["scene", "cameras", "depth", "normals", "frames"]);
        assert!(spec.is_clip_input("images"), "a video must be read whole, at its own rate, not sampled as prompt context");
    }

    /// Construction must not touch the filesystem - `brain caps` builds every
    /// provider in the catalog, so a provider that loads weights eagerly
    /// makes listing capabilities cost a checkpoint read per model.
    #[test]
    fn a_missing_checkpoint_is_a_named_error_not_a_panic() {
        let p = WorldMirror2Provider::without_weights();
        let action = p.action("reconstruct").expect("reconstruct exists");
        let err = action.run(&Invocation::default(), &mut |_| {}).unwrap_err();
        assert!(err.contains("checkpoint"), "{err}");
    }

    #[test]
    fn import_converts_with_no_checkpoint_and_rejects_a_missing_source() {
        let p = WorldMirror2Provider::without_weights();
        let action = p.action("import").expect("import exists");
        let inv = action.spec().validate(Invocation::new().set("source", json!("/nonexistent/model.safetensors")).set("dest", json!("/nonexistent/out.safetensors"))).unwrap();
        let err = action.run(&inv, &mut |_| {}).unwrap_err();
        assert!(err.contains("worldmirror2 import"), "{err}");
    }

    #[test]
    fn an_unknown_action_is_none() {
        let p = WorldMirror2Provider::without_weights();
        assert!(p.action("reconstruct").is_some() && p.action("import").is_some());
        assert!(p.action("infer").is_none());
    }
}
