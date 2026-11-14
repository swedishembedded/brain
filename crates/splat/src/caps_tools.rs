// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The served scene tools: `info`, `orient`, `merge` and `prune`.
//!
//! Each is a request-shaped wrapper over [`crate::tools`]: a scene arrives as
//! PLY bytes and cameras as a JSON blob, the result leaves the same way, and
//! the numbers a person wants to read (counts, bounds, the residual of every
//! link of a chain) are scalar outputs. `merge` chains scenes that live on
//! the serving machine, so it names them by path rather than carrying a
//! variable number of blobs.
//!
//! Swedish Embedded AB implements 3D scene processing pipelines that clean,
//! align and combine captures for its clients. If your team needs expertise
//! in point-cloud and Gaussian-scene post-processing then you can procure our
//! services by sending an email to info@swedishembedded.com.

use capability::bounds::{SIDE_PX, TRAIN_COUNT, UNIT};
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType};
use serde_json::json;

use crate::caps::{MAX_GAUSSIANS, MAX_WEIGHT};
use crate::tools::{self, CameraSet, PruneCfg};
use crate::types::{cameras_from_json, Splats};

pub fn info_spec() -> ActionSpec {
    ActionSpec::new("info", "what a Gaussian-splat scene is: gaussian count, bounds, mean opacity and spherical-harmonic degree").cancellable()
        .input(BlobSpec::new("scene", Media::Bytes, "the scene: Inria-layout binary PLY").required())
}

pub fn orient_spec() -> ActionSpec {
    ActionSpec::new("orient", "rigidly re-frame a scene so it opens the right way up: the plane the cameras orbit in is the scene's vertical, and their rays meet at its centre").cancellable()
        .param(ParamSpec::new("y_up", ParamType::Bool, "make +Y up for a viewer that expects it; brain's own world is y-down, so by default up is -Y").default(json!(false)))
        .input(BlobSpec::new("scene", Media::Bytes, "the scene: Inria-layout binary PLY").required())
        .input(BlobSpec::new("cameras", Media::Text, "the cameras the scene was reconstructed with, as JSON in the shape the 'fit' action's 'views' takes; at least 3, orbiting the subject").required())
        .output(BlobSpec::new("scene", Media::Bytes, "the re-framed scene: Inria-layout binary PLY"))
        .output(BlobSpec::new("cameras", Media::Text, "the cameras moved by the same transform, so a later fit still lines up"))
}

pub fn merge_spec() -> ActionSpec {
    ActionSpec::new("merge", "chain chunks of one long capture into a single scene in the first chunk's world, aligning each onto the one before it by the frames they share").cancellable()
        .param(ParamSpec::new("scenes", ParamType::Str, "comma-separated paths of the scene files on the serving machine, in capture order").required().host_resolved())
        .param(ParamSpec::new("cameras", ParamType::Str, "comma-separated paths of the cameras JSON files on the serving machine, one per scene in the same order").required().host_resolved())
        .param(ParamSpec::new("overlap", ParamType::Int, "how many trailing frames of each chunk lead the next one; at least 2, because a single shared frame fixes no scale").required().min(2.0).max(TRAIN_COUNT))
        .param(ParamSpec::new("prune", ParamType::Float, "fuse the doubled surface the overlap leaves on a voxel grid of this size in world units, a confident gaussian outvoting a faint duplicate; 0 = off").default(json!(0.0)).min(0.0).max(MAX_WEIGHT))
        .output(BlobSpec::new("scene", Media::Bytes, "the merged scene: Inria-layout binary PLY"))
        .output(BlobSpec::new("cameras", Media::Text, "every chunk's cameras in the merged world, as JSON"))
}

pub fn prune_spec() -> ActionSpec {
    ActionSpec::new("prune", "fuse duplicate gaussians and drop the faint ones: a pixel-aligned reconstruction counts one surface once per view that saw it").cancellable()
        .param(ParamSpec::new("voxel", ParamType::Float, "fuse on a grid of this cell size in world units; 0 = use 'voxel_pixels'").default(json!(0.0)).min(0.0).max(MAX_WEIGHT))
        .param(ParamSpec::new("voxel_pixels", ParamType::Float, "fuse at the distance the cameras can resolve, in pixels at the median sampling rate - two gaussians closer than the finest detail any camera saw cannot be told apart by any training view; needs 'cameras'; 0 = off; ignored when 'voxel' is set").default(json!(2.0)).min(0.0).max(SIDE_PX))
        .param(ParamSpec::new("min_opacity", ParamType::Float, "drop gaussians fainter than this").default(json!(0.0)).min(0.0).max(UNIT))
        .param(ParamSpec::new("recalibrate_opacity", ParamType::Bool, "re-solve opacity for the anti-alias filter (a reconstruction predicts it with none in mind, so rendering it through one that compensates energy dims the scene); needs 'cameras'").default(json!(false)))
        .param(ParamSpec::new("max_gaussians", ParamType::Int, "keep at most this many fused gaussians, the heaviest first; 0 = all").default(json!(0)).min(0.0).max(MAX_GAUSSIANS))
        .param(ParamSpec::new("opacity_gain", ParamType::Float, "multiply every opacity, for asking what a scene would look like if its gaussians were as confident as they should be").default(json!(1.0)).min(0.0).max(MAX_WEIGHT))
        .param(ParamSpec::new("range", ParamType::Str, "keep only a contiguous slice 'START:COUNT' of the gaussians, for asking what one frame's gaussians look like when nothing else is there to cover for them"))
        .input(BlobSpec::new("scene", Media::Bytes, "the scene: Inria-layout binary PLY").required())
        .input(BlobSpec::new("cameras", Media::Text, "the cameras the scene was reconstructed with, as JSON in the shape the 'fit' action's 'views' takes; sets the scale of 'voxel_pixels'"))
        .output(BlobSpec::new("scene", Media::Bytes, "the thinned scene: Inria-layout binary PLY"))
}

/// The `scene` input, parsed.
fn scene_input(inv: &Invocation, action: &str) -> Result<Splats, String> {
    let blob = inv.get_blob("scene").ok_or_else(|| format!("splat {action}: missing required input 'scene'"))?;
    crate::ply::parse(&blob.bytes).map_err(|e| format!("splat {action}: 'scene': {e}"))
}

/// The text of the `cameras` input, when there is one.
fn cameras_text(inv: &Invocation, action: &str) -> Result<Option<String>, String> {
    let Some(blob) = inv.get_blob("cameras") else { return Ok(None) };
    String::from_utf8(blob.bytes.clone()).map(Some).map_err(|_| format!("splat {action}: 'cameras' is not UTF-8 JSON"))
}

fn scene_blob(scene: &Splats) -> Result<Blob, String> {
    Ok(Blob::new(Media::Bytes, crate::ply::serialize(scene)?))
}

fn cameras_blob(cameras: &CameraSet) -> Blob {
    Blob::new(Media::Text, cameras.to_json().into_bytes())
}

pub(crate) fn info(inv: &Invocation) -> ActionResult {
    let i = tools::info(&scene_input(inv, "info")?);
    Ok(Outcome::new()
        .set("gaussians", json!(i.gaussians))
        .set("min", json!(i.min))
        .set("max", json!(i.max))
        .set("mean_opacity", json!(i.mean_opacity))
        .set("sh_degree", json!(i.sh_degree)))
}

pub(crate) fn orient(inv: &Invocation) -> ActionResult {
    let scene = scene_input(inv, "orient")?;
    let text = cameras_text(inv, "orient")?.ok_or("splat orient: missing required input 'cameras'")?;
    let cameras = CameraSet::parse(&text).map_err(|e| format!("splat orient: 'cameras': {e}"))?;
    let (oriented, moved) = tools::orient(&scene, &cameras, inv.get_bool("y_up").unwrap_or(false)).map_err(|e| format!("splat {e}"))?;
    Ok(Outcome::new().set("gaussians", json!(oriented.len())).blob("scene", scene_blob(&oriented)?).blob("cameras", cameras_blob(&moved)))
}

/// A comma-separated list of paths, without the empty entries.
fn path_list(inv: &Invocation, name: &str) -> Result<Vec<String>, String> {
    let raw = inv.get_str(name).ok_or_else(|| format!("splat merge: missing required param '{name}'"))?;
    Ok(raw.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect())
}

pub(crate) fn merge(inv: &Invocation) -> ActionResult {
    let (scenes, cameras) = (path_list(inv, "scenes")?, path_list(inv, "cameras")?);
    if scenes.len() < 2 || cameras.len() != scenes.len() {
        return Err(format!("splat merge: give N scenes and N cameras files, comma separated, N at least 2 (got {} and {})", scenes.len(), cameras.len()));
    }
    let chunks: Vec<(Splats, CameraSet)> = scenes
        .iter()
        .zip(&cameras)
        .map(|(scene, cams)| {
            inv.cancel.check()?;
            let s = crate::ply::read(scene).map_err(|e| format!("splat merge: cannot read {scene}: {e}"))?;
            let text = std::fs::read_to_string(cams).map_err(|e| format!("splat merge: cannot read {cams}: {e}"))?;
            Ok((s, CameraSet::parse(&text).map_err(|e| format!("splat merge: {cams}: {e}"))?))
        })
        .collect::<Result<_, String>>()?;
    let overlap = inv.get_i64("overlap").ok_or("splat merge: missing required param 'overlap'")?.max(0) as usize;
    let merged = tools::merge(&chunks, overlap, inv.get_f64("prune").unwrap_or(0.0) as f32).map_err(|e| e.replacen("merge", "splat merge", 1))?;
    let links: Vec<_> = merged
        .links
        .iter()
        .map(|l| json!({"chunk": l.chunk, "scale": l.scale, "residual": l.residual, "residual_fraction": l.residual_fraction}))
        .collect();
    Ok(Outcome::new()
        .set("chunks", json!(chunks.len()))
        .set("gaussians", json!(merged.scene.len()))
        .set("links", json!(links))
        .blob("scene", scene_blob(&merged.scene)?)
        .blob("cameras", cameras_blob(&merged.cameras)))
}

pub(crate) fn prune(inv: &Invocation) -> ActionResult {
    let scene = scene_input(inv, "prune")?;
    let cameras = match cameras_text(inv, "prune")? {
        Some(text) => Some(cameras_from_json(&text).map_err(|e| format!("splat prune: 'cameras': {e}"))?),
        None => None,
    };
    let range = inv.get_str("range").filter(|r| !r.trim().is_empty()).map(|r| tools::parse_range(&r)).transpose().map_err(|e| format!("splat prune: {e}"))?;
    let cfg = PruneCfg {
        voxel: inv.get_f64("voxel").unwrap_or(0.0) as f32,
        voxel_pixels: inv.get_f64("voxel_pixels").unwrap_or(2.0) as f32,
        min_opacity: inv.get_f64("min_opacity").unwrap_or(0.0) as f32,
        recalibrate_opacity: inv.get_bool("recalibrate_opacity").unwrap_or(false),
        max_gaussians: inv.get_i64("max_gaussians").unwrap_or(0).max(0) as usize,
        opacity_gain: inv.get_f64("opacity_gain").unwrap_or(1.0) as f32,
        range,
    };
    let p = tools::prune(&scene, cameras.as_deref(), &cfg)?;
    Ok(Outcome::new()
        .set("gaussians_in", json!(p.before))
        .set("gaussians_after_opacity", json!(p.after_opacity))
        .set("gaussians_out", json!(p.scene.len()))
        .set("voxel", json!(p.voxel))
        .blob("scene", scene_blob(&p.scene)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::SplatProvider;
    use capability::Provider;

    fn scene(n: usize) -> Splats {
        let mut s = Splats::default();
        for i in 0..n {
            s.means.extend_from_slice(&[i as f32 * 0.01, 0.0, 1.0]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[0.05; 3]);
            s.opacities.push(0.1 + 0.8 * (i % 2) as f32);
            s.colors.extend_from_slice(&[0.5; 3]);
        }
        s
    }

    fn run(action: &str, inv: Invocation) -> ActionResult {
        let act = SplatProvider::new().action(action).unwrap();
        act.run(&act.spec().validate(inv)?, &mut |_| {})
    }

    fn with_scene(n: usize) -> Invocation {
        Invocation::new().blob("scene", Blob::new(Media::Bytes, crate::ply::serialize(&scene(n)).unwrap()))
    }

    #[test]
    fn info_reports_the_scene() {
        let out = run("info", with_scene(10)).unwrap();
        assert_eq!(out.outputs["gaussians"], json!(10));
        assert_eq!(out.outputs["sh_degree"], json!(0));
        assert!((out.outputs["mean_opacity"].as_f64().unwrap() - 0.5).abs() < 1e-5);
    }

    /// The thinned scene comes back as a scene blob with the stage counts,
    /// and a pixel voxel without cameras is refused rather than skipped.
    #[test]
    fn prune_returns_the_thinned_scene_and_its_counts() {
        let out = run("prune", with_scene(10).set("voxel_pixels", json!(0.0)).set("min_opacity", json!(0.5))).unwrap();
        assert_eq!(out.outputs["gaussians_in"], json!(10));
        assert_eq!(out.outputs["gaussians_out"], json!(5));
        assert_eq!(crate::ply::parse(&out.blobs["scene"].bytes).unwrap().len(), 5);
        let e = run("prune", with_scene(10)).unwrap_err();
        assert!(e.contains("needs the cameras"), "{e}");
    }

    #[test]
    fn orient_needs_its_cameras() {
        let e = run("orient", with_scene(4)).unwrap_err();
        assert!(e.contains("cameras"), "{e}");
    }

    /// `merge` names its inputs by path, so a lone scene or a mismatched
    /// pair is refused before any file is read.
    #[test]
    fn merge_refuses_a_lone_scene_and_a_mismatched_count() {
        let inv = |scenes: &str, cams: &str| Invocation::new().set("scenes", json!(scenes)).set("cameras", json!(cams)).set("overlap", json!(2));
        assert!(run("merge", inv("a.ply", "a.json")).unwrap_err().contains("N at least 2"));
        assert!(run("merge", inv("a.ply,b.ply", "a.json")).unwrap_err().contains("N scenes and N cameras"));
    }

    /// Every scene tool is declared with its bounds, and the ones that name
    /// serving-machine paths declare them host-resolved.
    #[test]
    fn the_scene_tools_are_declared_with_bounds_and_host_paths() {
        for spec in [info_spec(), orient_spec(), merge_spec(), prune_spec()] {
            for p in spec.params.iter().filter(|p| matches!(p.ty, ParamType::Int | ParamType::Float)) {
                assert!(p.max.is_some(), "{}:{} has no max", spec.name, p.name);
            }
        }
        assert!(merge_spec().params.iter().filter(|p| p.ty == ParamType::Str).all(|p| p.host_resolved));
    }
}
