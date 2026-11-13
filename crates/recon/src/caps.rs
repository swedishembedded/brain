// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Splat's served actions with `sfm` and `train` beside them: photographs
//! to cameras and a sparse cloud, or all the way to a finished splat scene.
//!
//! The two live here rather than in `splat::caps` because they run structure
//! from motion and multi-view stereo ([`crate::photogrammetry`]), which
//! depend on `splat` themselves. [`provider`] wraps splat's own provider, so
//! `brain/splat` is one model with every action, served and on the command
//! line alike.
//!
//! Swedish Embedded AB implements photogrammetry pipelines that turn a
//! customer's photographs into scenes it can ship for its clients. If your
//! team needs expertise in structure from motion and Gaussian splatting then
//! you can procure our services by sending an email to info@swedishembedded.com.

use std::sync::Arc;

use capability::bounds::{SIDE_PX, TRAIN_STEPS, UNIT};
use capability::{Action, ActionResult, ActionSpec, Assembly, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use gpu_core::Gpu;
use crate::photogrammetry::{self, PhotoCfg};
use serde_json::json;
use sfm::incremental::SfmCfg;

/// Ceiling on a gaussian budget: 2^28, past any scene a device holds.
const MAX_GAUSSIANS: f64 = 268_435_456.0;
/// Ceiling on a focal-length guess, as a multiple of the longer side.
const MAX_FOCAL: f64 = 20.0;
/// Most 2x2 box downsamplings of a photograph a caller may ask for.
const MAX_HALVINGS: f64 = 16.0;

/// Splat's provider with `sfm` and `train` beside its own actions.
pub struct Photogrammetry {
    splat: Arc<dyn Provider>,
}

impl Photogrammetry {
    pub fn new(splat: Arc<dyn Provider>) -> Photogrammetry {
        Photogrammetry { splat }
    }
}

/// Splat's served provider with `sfm` and `train` added: what the catalog
/// serves as `brain/splat`.
pub fn provider(assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(Photogrammetry::new(splat::caps::provider(assembly)?)))
}

/// `brain/splat`'s manifest: splat's own actions, `sfm` and `train`.
pub fn manifest() -> Manifest {
    let mut m = splat::caps::manifest();
    m.actions.push(sfm_spec());
    m.actions.push(train_spec());
    m
}

impl Provider for Photogrammetry {
    fn manifest(&self) -> Manifest {
        manifest()
    }

    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "sfm" => Some(Arc::new(Sfm) as Arc<dyn Action>),
            "train" => Some(Arc::new(Train) as Arc<dyn Action>),
            _ => self.splat.action(name),
        }
    }
}

fn images_param() -> ParamSpec {
    ParamSpec::new("images", ParamType::Str, "folder of photographs on the serving machine (read in name order), or a comma-separated list of image file paths").required().host_resolved()
}

fn focal_guess_param() -> ParamSpec {
    ParamSpec::new("focal_guess", ParamType::Float, "focal length as a multiple of the longer side, for a photograph with no lens metadata; 0.8 is about a 26 mm-equivalent lens, what phone and compact main cameras ship")
        .default(json!(SfmCfg::default().focal_guess))
        .min(0.05)
        .max(MAX_FOCAL)
}

fn cameras_output(what: &str) -> BlobSpec {
    BlobSpec::new("cameras", Media::Text, what)
}

pub fn sfm_spec() -> ActionSpec {
    ActionSpec::new("sfm", "structure from motion alone: the cameras and the sparse point cloud of a set of photographs, for inspection or for a fit")
        .param(images_param())
        .param(ParamSpec::new("halvings", ParamType::Int, "the training resolution of the cameras, as exact 2x2 box-downsamplings of the photographs' own: structure from motion runs on the full photographs, the fit on their own pixels through their own lens, this many halvings below").default(json!(0)).min(0.0).max(MAX_HALVINGS))
        .param(focal_guess_param())
        .param(ParamSpec::new("init_opacity", ParamType::Float, "what the sparse points start as: 3DGS's 0.1, so the fit decides what is solid - fat blobs of high opacity occlude what is behind them from the first step and starve it of gradient").default(json!(0.1)).min(0.0).max(UNIT))
        .output(BlobSpec::new("scene", Media::Bytes, "the sparse cloud as a scene: Inria-layout binary PLY"))
        .output(cameras_output("the registered photographs' cameras as JSON, in the shape the 'fit' action's 'views' takes"))
}

pub fn train_spec() -> ActionSpec {
    let d = PhotoCfg::default();
    ActionSpec::new("train", "photographs to a finished splat scene: structure from motion, multi-view stereo priors and a dense start, and the full reconstruction objective, rendered along each pixel's ray through its own lens")
        .training()
        .streaming()
        .param(images_param())
        .param(ParamSpec::new("max_width", ParamType::Int, "widest training image, px: the photographs are halved exactly until they fit; the fit itself climbs a pyramid up to it").default(json!(d.max_width)).min(1.0).max(SIDE_PX))
        .param(ParamSpec::new("iters", ParamType::Int, "optimizer steps; unset = decided from the number of views (every view revisited about five hundred times, within 3000 and 30000 steps)").min(1.0).max(TRAIN_STEPS))
        .param(ParamSpec::new("max_gaussians", ParamType::Int, "gaussian budget; unset = decided from the capture (a dense start gets a quarter more than it began with, a sparse one one gaussian per eight training pixels)").min(1.0).max(MAX_GAUSSIANS))
        .param(focal_guess_param())
        .param(ParamSpec::new("sparse", ParamType::Bool, "start from the structure-from-motion points alone, without multi-view stereo").default(json!(false)))
        .param(ParamSpec::new("camera_model", ParamType::Bool, "fit the photometric camera (exposure, white balance, vignetting, colour matrix, response) with the scene").default(json!(d.camera_model.is_some())))
        .param(ParamSpec::new("environment", ParamType::Int, "fit sky and distant scenery as radiance by direction, as a spherical-harmonic expansion of this degree, instead of gaussians hanging behind the scene; unset = off").min(0.0).max(splat::env::MAX_DEGREE as f64))
        .param(ParamSpec::new("transients", ParamType::Bool, "the photographs may hold things that are not in every one (people, traffic): stop supervising the regions the scene cannot explain").default(json!(false)))
        .output(BlobSpec::new("scene", Media::Bytes, "the finished scene: Inria-layout binary PLY, for a standard splat viewer"))
        .output(cameras_output("every placed photograph's refined camera, at the training resolution, as JSON"))
}

/// The photographs `images` names, or why there are none.
fn photograph_paths(inv: &Invocation, action: &str) -> Result<Vec<String>, String> {
    let spec = inv.get_str("images").ok_or_else(|| format!("splat {action}: missing required param 'images'"))?;
    let paths = splat::tools::collect_images(&spec).map_err(|e| format!("splat {action}: {e}"))?;
    if paths.is_empty() {
        return Err(format!("splat {action}: no photographs in {spec}"));
    }
    Ok(paths)
}

fn not_placed(paths: &[String], placed: &[usize]) -> Vec<String> {
    paths.iter().enumerate().filter(|(i, _)| !placed.contains(i)).map(|(_, p)| p.clone()).collect()
}

struct Sfm;

impl Action for Sfm {
    fn spec(&self) -> ActionSpec {
        sfm_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if inv.cancel.is_cancelled() {
            return Err("cancelled".to_string());
        }
        let paths = photograph_paths(inv, "sfm")?;
        let photos: Vec<imaging::Rgb8> = paths.iter().map(|p| imaging::load(p).map_err(|e| format!("splat sfm: {p}: {e}"))).collect::<Result<_, _>>()?;
        progress(Progress::step(0, 1, format!("structure from motion on {} photographs", photos.len())));
        let defaults = SfmCfg::default();
        let cfg = SfmCfg { focal_guess: inv.get_f64("focal_guess").unwrap_or(defaults.focal_guess), verbose: true, ..defaults };
        let halvings = inv.get_i64("halvings").unwrap_or(0).max(0) as u32;
        let opacity = inv.get_f64("init_opacity").unwrap_or(0.1) as f32;
        let set = photogrammetry::training_set(&photos, halvings, opacity, &cfg).map_err(|e| format!("splat sfm: {e}"))?;
        let cameras: Vec<splat::types::Camera> = set.targets.iter().map(|t| t.cam).collect();
        let lens = sfm::lens::describe(&set.sfm.intrinsics[0]);
        Ok(Outcome::new()
            .set("photographs", json!(photos.len()))
            .set("registered", json!(set.targets.len()))
            .set("points", json!(set.init.len()))
            .set("reprojection_rms_px", json!(set.sfm.rms_px))
            .set("lens", json!(lens))
            .set("not_registered", json!(not_placed(&paths, &set.source)))
            .blob("scene", Blob::new(Media::Bytes, splat::ply::serialize(&set.init)?))
            .blob("cameras", Blob::new(Media::Text, splat::types::cameras_to_json(&cameras).into_bytes())))
    }
}

struct Train;

impl Action for Train {
    fn spec(&self) -> ActionSpec {
        train_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if inv.cancel.is_cancelled() {
            return Err("cancelled".to_string());
        }
        let paths = photograph_paths(inv, "train")?;
        // as recorded: GPS places the scene, EXIF exposure starts the camera
        // model's, a linear or deep photograph keeps its precision
        let photos: Vec<imaging::Photo> = paths.iter().map(|p| imaging::load_photo(p).map_err(|e| format!("splat train: {p}: {e}"))).collect::<Result<_, _>>()?;
        let defaults = PhotoCfg::default();
        let iterations = inv.get_i64("iters").map(|v| v.max(1) as usize);
        let cfg = PhotoCfg {
            sfm: SfmCfg { focal_guess: inv.get_f64("focal_guess").unwrap_or(defaults.sfm.focal_guess), ..defaults.sfm },
            max_width: inv.get_i64("max_width").map_or(defaults.max_width, |v| v.max(1) as u32),
            dense: if inv.get_bool("sparse").unwrap_or(false) { None } else { defaults.dense },
            iterations,
            max_gaussians: inv.get_i64("max_gaussians").map(|v| v.max(1) as usize),
            camera_model: inv.get_bool("camera_model").unwrap_or(true).then(splat::isp::IspCfg::default),
            environment: inv.get_i64("environment").map(|v| v.max(0) as u32),
            transients: inv.get_bool("transients").unwrap_or(false),
        };
        // The real step count is known once the placed photographs are; this is the
        // estimate the progress bar scales to, never an upper bound on the fit.
        let expected = iterations.unwrap_or_else(|| photogrammetry::auto_iterations(photos.len())) as u32;
        let every = (expected / 100).max(1);
        let mut log = Vec::new();
        let cancel = inv.cancel.clone();
        let gpu = Gpu::new(&photogrammetry::pipelines());
        let res = photogrammetry::reconstruct_photos(&gpu, &photos, &cfg, &mut |line| log.push(line.to_string()), &mut |it, loss| {
            if (it as u32).is_multiple_of(every) {
                progress(Progress::step(it as u32 + 1, expected.max(it as u32 + 1), format!("loss {loss:.5}")));
            }
            !cancel.is_cancelled()
        })
        .map_err(|e| format!("splat train: {e}"))?;
        if inv.cancel.is_cancelled() {
            return Err("cancelled".to_string());
        }
        Ok(Outcome::new()
            .set("photographs", json!(photos.len()))
            .set("placed", json!(res.source.len()))
            .set("gaussians", json!(res.scene.len()))
            .set("loss", json!(res.loss))
            .set("iterations", json!(res.iterations))
            .set("eps2d", json!(res.render.eps2d))
            .set("not_placed", json!(not_placed(&paths, &res.source)))
            .set("log", json!(log))
            .blob("scene", Blob::new(Media::Bytes, splat::ply::serialize(&res.export())?))
            .blob("cameras", Blob::new(Media::Text, splat::types::cameras_to_json(&res.cameras).into_bytes())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both actions are declared with a ceiling on every number, and the
    /// photographs they read are named host-side.
    #[test]
    fn the_photogrammetry_actions_are_bounded_and_read_host_photographs() {
        for spec in [sfm_spec(), train_spec()] {
            for p in spec.params.iter().filter(|p| matches!(p.ty, ParamType::Int | ParamType::Float)) {
                assert!(p.max.is_some(), "{}:{} has no max", spec.name, p.name);
            }
            assert!(spec.params.iter().find(|p| p.name == "images").unwrap().host_resolved);
        }
        assert!(train_spec().streaming);
    }

    /// The manifest the command line shows holds splat's own actions and
    /// these two, and each resolves to a runnable action.
    #[test]
    fn the_provider_offers_splats_actions_and_its_own() {
        let provider = provider(&Assembly { id: String::new(), arch: String::new(), variant: None, roles: Default::default(), provenance: Vec::new() }).unwrap();
        for a in manifest().actions {
            assert!(provider.action(&a.name).is_some(), "{} is declared but not provided", a.name);
        }
        assert!(provider.action("nope").is_none());
    }

    #[test]
    fn a_folder_without_photographs_is_refused() {
        let d = std::env::temp_dir().join(format!("splat-empty-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let inv = Invocation::new().set("images", json!(d.to_str().unwrap()));
        let e = photograph_paths(&inv, "sfm").unwrap_err();
        assert!(e.contains("no photographs"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
