// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The served `fit` action: optimize a Gaussian-splat scene against posed
//! photographs.
//!
//! Every knob of [`FitCfg`] a reconstruction is steered by is a param, with
//! the library's own default: how far geometry may travel, which low-pass the
//! scene is fitted for, how much view-dependent colour it may carry, density
//! control, the photometric camera model, pose refinement, the surface
//! regularizers and the bounds on a gaussian's size and shape.
//!
//! The target views come from the request (`views` + a `video` blob, which is
//! what a remote caller can send) or from the serving machine (`cameras` +
//! `images`, a cameras document and a folder of photographs next to the
//! action, which is what a command line has).
//!
//! Swedish Embedded AB implements reconstruction pipelines that fit a scene
//! to a customer's own photographs for its clients. If your team needs
//! expertise in differentiable rendering and scene optimization then you can
//! procure our services by sending an email to info@swedishembedded.com.

use capability::bounds::{SIDE_PX, TRAIN_STEPS, UNIT, VIDEO_FRAMES};
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress};
use gpu_core::Gpu;
use serde_json::json;

use crate::caps::{MAX_GAUSSIANS, MAX_RATIO, MAX_WEIGHT};
use crate::opt::{fit_bundle, CameraRefine, Densify, FitCfg, TargetView};
use crate::types::{cameras_from_json, cameras_to_json, Camera};
use crate::{Kernels, PIPELINES};

/// The pixel losses `fit` can minimise, by the name its `loss` param takes.
const LOSSES: [&str; 2] = ["mse", "l1-ssim"];
/// The density controls, by the name `densify_strategy` takes.
const STRATEGIES: [&str; 3] = ["heuristic", "mcmc", "hybrid"];

fn names(of: &[&str]) -> ParamType {
    ParamType::Enum(of.iter().map(|s| s.to_string()).collect())
}

pub fn fit_spec() -> ActionSpec {
    let d = FitCfg::default();
    ActionSpec::new("fit", "optimize a Gaussian-splat scene against N posed target views (Adam on log-scale / logit gaussian parameters, ray-evaluated rasterizer backward)")
        .training()
        .streaming()
        .param(ParamSpec::new(
            "views",
            ParamType::Str,
            "camera array as JSON, one entry per target-view frame in order: [{\"c2w\":[16 floats],\"fx\":...,\"fy\":...,\"cx\":...,\"cy\":...,\"width\":...,\"height\":...}, ...] - the same shape `mirror_cli.rs`'s cameras.json uses; a lens (\"model\": \"opencv\"|\"fisheye\"|\"equirect\" with \"k\"/\"p\"/\"s\" coefficients) and a rolling-shutter \"shutter\" twist are optional. Give this or 'cameras'",
        ))
        .param(ParamSpec::new("cameras", ParamType::Str, "path of a cameras JSON file on the serving machine (the array 'views' takes), one camera per photograph in order; give this or 'views'").host_resolved())
        .param(ParamSpec::new("images", ParamType::Str, "folder of photographs on the serving machine (read in name order), or a comma-separated list of image file paths, one per camera; give this or the 'video' input").host_resolved())
        .param(ParamSpec::new("iters", ParamType::Int, "optimization steps").max(TRAIN_STEPS).default(json!(200)).min(1.0))
        // No ceiling: `scripts/gates/unbounded-params.allow` still lists this param as an
        // unbounded rate, and the served-bounds test holds the two in step.
        .param(ParamSpec::new("lr_position", ParamType::Float, "starting position step, in units of a scene normalized to about one across; decays a hundredfold over the fit").default(json!(d.lr_position)).min(0.0))
        .param(ParamSpec::new("position_radii", ParamType::Float, "for a scene that is already metric (a feed-forward reconstruction's): the travel budget of a gaussian over the whole run, in its own radii, so it settles onto its surface and does not leave it; replaces 'lr_position'. 0 = unbounded, which a fit from a sparse point cloud needs").default(json!(0.0)).min(0.0).max(MAX_RATIO))
        .param(ParamSpec::new("eps2d", ParamType::Float, "a pixel's footprint variance, pixels squared: part of the forward model, not a display setting - render the result with this same value or the optimizer's compensation for it shows up as blur (rendered higher) or aliasing (lower)").default(json!(d.eps2d)).min(0.0).max(MAX_WEIGHT))
        .param(ParamSpec::new("inria_lowpass", ParamType::Bool, "fit for an Inria-convention viewer (uncompensated dilation at 0.3) instead of the Mip filter; a scene fitted under one low-pass and rendered under another comes out wrong in both directions").default(json!(false)))
        .param(ParamSpec::new("mip_scale", ParamType::Float, "Mip-Splatting's 3D smoothing filter: every gaussian is low-passed to this many samples of the finest camera that saw it; 0 disables").default(json!(d.mip_scale)).min(0.0).max(MAX_WEIGHT))
        .param(ParamSpec::new("sh_degree", ParamType::Int, "view-dependent colour, spherical-harmonic degree 0 to 3; every extra degree is more freedom to explain a view away without moving geometry, which cuts both ways when the views are few").default(json!(d.sh_degree)).min(0.0).max(3.0))
        .param(ParamSpec::new("pose_lr", ParamType::Float, "refine the cameras alongside the scene at this step (radians and scene-normalized units); a pipeline that predicts its own poses hands this step cameras that are wrong. 0 = hold them. The refined cameras come back as the 'cameras' output").default(json!(0.0)).min(0.0).max(UNIT))
        .param(ParamSpec::new("densify", ParamType::Int, "run density control every N steps, adding gaussians where the loss is still pulling; 0 = off, because a feed-forward scene is already dense and growing it can push the backward past the device's gradient-record ceiling").default(json!(0)).min(0.0).max(TRAIN_STEPS))
        .param(ParamSpec::new("densify_frac", ParamType::Float, "fraction of gaussians refined per density round when there is no budget").default(json!(d.densify_frac)).min(0.0).max(UNIT))
        .param(ParamSpec::new("max_gaussians", ParamType::Int, "refuse to grow past this many gaussians; 0 = unbounded. With the 'mcmc' strategy it is the budget that is spent").default(json!(0)).min(0.0).max(MAX_GAUSSIANS))
        .param(ParamSpec::new("densify_strategy", names(&STRATEGIES), "which density control: the heuristic clone/split/prune, 3DGS-MCMC (spends 'max_gaussians' as a budget and recycles transparent gaussians instead of deleting them), or both").default(json!("heuristic")))
        .param(ParamSpec::new("loss", names(&LOSSES), "the photometric objective: mean squared error, or the L1 + D-SSIM objective 3DGS is defined with").default(json!("mse")))
        .param(ParamSpec::new("camera_model", ParamType::Bool, "fit a photometric camera model (exposure, white balance, vignetting, response) with the scene, so a view shot a stop brighter is explained by its camera rather than by brighter gaussians").default(json!(false)))
        .param(ParamSpec::new("batch", ParamType::Int, "views per step; 0 = all of them").default(json!(0)).min(0.0).max(VIDEO_FRAMES))
        .param(ParamSpec::new("distortion", ParamType::Float, "weight of the 2DGS depth distortion regularizer; 0 = off").default(json!(0.0)).min(0.0).max(MAX_WEIGHT))
        .param(ParamSpec::new("normal_consistency", ParamType::Float, "weight of normal consistency against the rendered range map's own normals; 0 = off").default(json!(0.0)).min(0.0).max(MAX_WEIGHT))
        .param(ParamSpec::new("geometry_after", ParamType::Float, "fraction of the fit after which the surface regularizers switch on; applied before the scene has any shape they pull gaussians onto whatever surface happens to be rendered first").default(json!(0.0)).min(0.0).max(UNIT))
        .param(ParamSpec::new("max_scale_pixels", ParamType::Float, "how large a gaussian may get, in pixels of the view that samples it best: wider than a few pixels it cannot carry detail the cameras resolved, only blur it; 0 = unbounded").default(json!(d.max_scale_pixels)).min(0.0).max(SIDE_PX))
        .param(ParamSpec::new("max_growth", ParamType::Float, "how many times its starting size a gaussian may grow; 0 = unbounded").default(json!(d.max_growth)).min(0.0).max(MAX_RATIO))
        .param(ParamSpec::new("max_needle", ParamType::Float, "largest ratio of a gaussian's longest axis to its middle one (a needle is always wrong); 1 or less = unconstrained").default(json!(d.max_needle)).min(0.0).max(MAX_RATIO))
        .param(ParamSpec::new("max_flat", ParamType::Float, "largest ratio of a gaussian's middle axis to its shortest (a disc is a legitimate surface element, a blade seen edge on is a needle)").default(json!(d.max_flat)).min(0.0).max(MAX_RATIO))
        .input(BlobSpec::new("scene", Media::Bytes, "the initial scene to optimize: Inria-layout binary PLY").required())
        .input(BlobSpec::new("video", Media::Video, "N target views: interleaved-HWC f32 RGB frames, one per camera, same convention as every other video input (capability::blob::decode_video); give this or 'images'"))
        .output(BlobSpec::new("scene", Media::Bytes, "the optimized scene: Inria-layout binary PLY"))
        .output(BlobSpec::new("cameras", Media::Text, "the refined cameras as JSON, in the shape 'views' takes (only when 'pose_lr' is set): a scene fitted against moved cameras only means anything read back with those cameras"))
}

/// The cameras the request names: the `views` JSON, or the document at the
/// `cameras` path.
fn request_cameras(inv: &Invocation) -> Result<Vec<Camera>, String> {
    let raw = match (inv.get_str("views"), inv.get_str("cameras")) {
        (Some(json), None) => json,
        (None, Some(path)) => std::fs::read_to_string(&path).map_err(|e| format!("splat fit: cannot read cameras {path}: {e}"))?,
        (Some(_), Some(_)) => return Err("splat fit: give the cameras as 'views' or as 'cameras', not both".to_string()),
        (None, None) => return Err("splat fit: missing the cameras: give 'views' (JSON) or 'cameras' (a cameras document on the serving machine)".to_string()),
    };
    cameras_from_json(&raw).map_err(|e| format!("splat fit: cameras: {e}"))
}

/// The target views of `cams`: the `video` blob's frames, or the photographs
/// at `images`.
fn request_targets(inv: &Invocation, cams: Vec<Camera>) -> Result<Vec<TargetView>, String> {
    match (inv.get_blob("video").is_some(), inv.get_str("images")) {
        (true, None) => {
            let frames = capability::blob::decode_video(inv, "video")?;
            if cams.len() != frames.len() {
                return Err(format!("splat fit: {} cameras but {} frames in 'video'", cams.len(), frames.len()));
            }
            cams.into_iter().zip(frames).enumerate().map(|(i, (cam, (rgb, w, h)))| crate::tools::target_view(cam, rgb, w, h, &format!("view {i}'s frame"))).collect::<Result<_, _>>().map_err(|e| format!("splat fit: {e}"))
        }
        (false, Some(spec)) => {
            let paths = crate::tools::collect_images(&spec).map_err(|e| format!("splat fit: {e}"))?;
            crate::tools::load_targets(&paths, &cams).map_err(|e| format!("splat fit: {e}"))
        }
        (true, Some(_)) => Err("splat fit: give the photographs as the 'video' input or as 'images', not both".to_string()),
        (false, None) => Err("splat fit: missing the photographs: give the 'video' input or 'images' (a folder on the serving machine)".to_string()),
    }
}

/// The optimizer settings the request asks for.
fn fit_cfg(inv: &Invocation) -> FitCfg {
    let d = FitCfg::default();
    let f = |name: &str, default: f32| inv.get_f64(name).map_or(default, |v| v as f32);
    let n = |name: &str, default: usize| inv.get_i64(name).map_or(default, |v| v.max(0) as usize);
    let iters = n("iters", d.iters).max(1);
    let position_radii = f("position_radii", 0.0);
    let inria = inv.get_bool("inria_lowpass").unwrap_or(false);
    let pose_lr = f("pose_lr", 0.0);
    FitCfg {
        iters,
        relative_position: position_radii > 0.0,
        // one radius over the whole fit is a rate of 4.65 / iters (see `FitCfg::lr_position`)
        lr_position: if position_radii > 0.0 { position_radii * 4.65 / iters as f32 } else { f("lr_position", d.lr_position) },
        eps2d: if inria { 0.3 } else { f("eps2d", d.eps2d) },
        antialiased: !inria,
        mip_scale: f("mip_scale", d.mip_scale),
        densify_every: n("densify", 0),
        densify_frac: f("densify_frac", d.densify_frac),
        max_gaussians: n("max_gaussians", 0),
        strategy: match inv.get_str("densify_strategy").as_deref() {
            Some("mcmc") => Densify::Mcmc,
            Some("hybrid") => Densify::Hybrid,
            _ => Densify::Heuristic,
        },
        sh_degree: n("sh_degree", d.sh_degree as usize) as u32,
        camera: CameraRefine { pose_after: if pose_lr > 0.0 { 0.0 } else { 1.0 }, pose_lr, ..Default::default() },
        max_scale_pixels: f("max_scale_pixels", d.max_scale_pixels),
        max_growth: f("max_growth", d.max_growth),
        max_needle: f("max_needle", d.max_needle),
        max_flat: f("max_flat", d.max_flat),
        loss: match inv.get_str("loss").as_deref() {
            Some("l1-ssim") => crate::loss::PixelLoss::gaussian_splatting(),
            _ => crate::loss::PixelLoss::Mse,
        },
        isp: inv.get_bool("camera_model").unwrap_or(false).then(crate::isp::IspCfg::default),
        batch: n("batch", 0),
        distortion_weight: f("distortion", 0.0),
        normal_consistency_weight: f("normal_consistency", 0.0),
        geometry_after: f("geometry_after", 0.0),
        log_every: 0,
        ..d
    }
}

/// Run one `fit` invocation (already validated against [`fit_spec`]).
///
/// Cancellation: `on_step` reports [`Progress::step`] then checks
/// `inv.cancel` - the same "poll every step, `Err("cancelled")` if it fired"
/// contract `wan`/`flux2`'s training actions use, adapted to `opt::fit`'s
/// `on_step` hook rather than a cancel token threaded through the library
/// call itself (splat has no `capability` dependency below this module).
pub fn fit(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    if inv.cancel.is_cancelled() {
        return Err("cancelled".to_string());
    }
    let scene_blob = inv.get_blob("scene").ok_or("splat fit: missing required input 'scene'")?;
    let init = crate::ply::parse(&scene_blob.bytes)?;
    let targets = request_targets(inv, request_cameras(inv)?)?;
    let cfg = fit_cfg(inv);
    let iters = cfg.iters;

    let gpu = Gpu::new(PIPELINES);
    let ks = Kernels::at(0);
    let cancel = inv.cancel.clone();
    let mut on_step = |it: usize, mse: f32| -> bool {
        progress(Progress::step((it + 1) as u32, iters as u32, format!("mse {mse}")));
        !cancel.is_cancelled()
    };
    let (scene, refined, mse) = fit_bundle(&gpu, ks, &init, &targets, &cfg, &mut on_step);
    if inv.cancel.is_cancelled() {
        return Err("cancelled".to_string());
    }
    let bytes = crate::ply::serialize(&scene)?;
    let mut out = Outcome::new()
        .set("mse", json!(mse))
        .set("gaussians", json!(scene.len()))
        .set("eps2d", json!(cfg.eps2d))
        .set("antialiased", json!(cfg.antialiased))
        .blob("scene", Blob::new(Media::Bytes, bytes));
    if cfg.camera.pose_lr > 0.0 {
        out = out.blob("cameras", Blob::new(Media::Text, cameras_to_json(&refined).into_bytes()));
    }
    Ok(out)
}
