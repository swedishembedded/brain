// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! From a model forward pass to a finished scene: the geometry-source choices,
//! duplicate fusion, outlier rejection, and re-framing the result upright.
//!
//! [`gaussians::assemble`] turns the heads' output into gaussians under the
//! reference's own filters; this module is what a caller does with them next.
//! Both the served action and any direct caller go through [`build`], so the
//! two cannot drift.
//!
//! Swedish Embedded AB implements feed-forward 3D reconstruction pipelines
//! whose output opens the right way up and without ghost surfaces for its
//! clients. If your team needs expertise in turning raw reconstructions into
//! viewable scenes then you can procure our services by sending an email to
//! info@swedishembedded.com.

use splat::types::{Camera, Splats};

use crate::gaussians::{assemble, AssembleOpts, NormalSource, PositionSource};
use crate::model::Mirror;
use crate::priors::CameraPrior;

/// `gsdepth`, `points` or `blend`: the model predicts the scene's geometry
/// through more than one head, and which one the scene is built on is a
/// question worth being able to ask.
pub fn position_source(name: &str) -> Result<PositionSource, String> {
    match name {
        "gsdepth" => Ok(PositionSource::GsDepth),
        "points" => Ok(PositionSource::Points),
        "blend" => Ok(PositionSource::Blend),
        other => Err(format!("position source '{other}' (gsdepth|points|blend)")),
    }
}

/// `geometry` or `head`: where the surface normal behind surface alignment
/// comes from.
pub fn normal_source(name: &str) -> Result<NormalSource, String> {
    match name {
        "geometry" => Ok(NormalSource::Geometry),
        "head" => Ok(NormalSource::Head),
        other => Err(format!("normal source '{other}' (geometry|head)")),
    }
}

/// What happens to the assembled gaussians before they are returned.
pub struct SceneOpts {
    pub assemble: AssembleOpts,
    /// Voxel-merge duplicates at this edge length, in scene units; 0 = off.
    ///
    /// Fusion collapses the near-duplicate surfaces that overlapping views
    /// each predict at slightly different depths. Without it the scene is
    /// those duplicates stacked, which a viewer sees as translucent
    /// superposition.
    pub prune_voxel: f32,
    /// Drop the gaussians whose scale is above this quantile; 0 or 1 = off.
    pub max_scale_quantile: f32,
    /// Keep the scene in frame 0's own frame instead of re-framing it upright.
    ///
    /// The model anchors the world to the FIRST frame (its c2w comes back as
    /// the identity), so a scene written there opens tipped by however the
    /// camera happened to be held, with the subject off to one side of a
    /// viewer's default view. That angle is an artifact of which photograph
    /// came first, not information about the scene.
    pub keep_camera_frame: bool,
}

impl Default for SceneOpts {
    fn default() -> SceneOpts {
        SceneOpts { assemble: AssembleOpts::default(), prune_voxel: 0.002, max_scale_quantile: 0.98, keep_camera_frame: false }
    }
}

/// A finished scene and the camera each frame was taken from.
pub struct Scene {
    pub splats: Splats,
    pub cameras: Vec<Camera>,
}

/// Run the trunk over `frames_chw` (`s` frames of `w`x`h`), conditioned on
/// `priors` when the caller knows the cameras. Returns those cameras as the
/// back-projection cameras `assemble` should use: known cameras are not just
/// a hint to the trunk, they are what the geometry is built through.
pub fn forward(model: &mut Mirror, frames_chw: &[f32], s: usize, (w, h): (u32, u32), priors: Option<&[CameraPrior]>) -> Option<Vec<Camera>> {
    let (hp, wp) = (h as usize / model.cfg.patch, w as usize / model.cfg.patch);
    model.forward_with_priors(frames_chw, s, hp, wp, priors);
    priors.map(|ps| ps.iter().map(|p| Camera::pinhole(p.c2w, p.fx, p.fy, p.cx, p.cy, w, h)).collect())
}

/// Assemble the forward pass the model last ran into a finished scene.
pub fn build(model: &Mirror, frames_chw: &[f32], s: usize, (w, h): (u32, u32), known: Option<&[Camera]>, opts: &SceneOpts) -> Scene {
    let (mut splats, cameras, weights) = assemble(model.gpu(), model, frames_chw, s, w, h, &opts.assemble, known);
    if opts.prune_voxel > 0.0 {
        splats = splat::prune::voxel_merge(&splats, &weights, opts.prune_voxel, 0);
    }
    if opts.max_scale_quantile > 0.0 && opts.max_scale_quantile < 1.0 {
        splats = splat::prune::drop_largest_scales(&splats, opts.max_scale_quantile);
    }
    if !opts.keep_camera_frame && cameras.len() >= 3 {
        let (splats, cameras) = upright(&splats, &cameras);
        return Scene { splats, cameras };
    }
    Scene { splats, cameras }
}

/// Re-express a scene and its cameras in the frame the cameras describe: the
/// vertical is the axis they sweep about, the origin is where they all look.
pub fn upright(splats: &Splats, cams: &[Camera]) -> (Splats, Vec<Camera>) {
    let mats: Vec<[f64; 16]> = cams.iter().map(|c| std::array::from_fn(|i| c.c2w[i] as f64)).collect();
    let (r, centre) = splat::orient::frame_from_cameras(&mats, -1.0);
    let moved = cams
        .iter()
        .zip(&mats)
        .map(|(c, m)| {
            let t = splat::orient::transform_c2w(m, &r, &centre);
            Camera { c2w: std::array::from_fn(|i| t[i] as f32), ..*c }
        })
        .collect();
    (splat::orient::apply(splats, &r, &centre), moved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_sources_parse_and_refuse_unknowns() {
        assert!(matches!(position_source("blend"), Ok(PositionSource::Blend)));
        assert!(matches!(normal_source("head"), Ok(NormalSource::Head)));
        assert!(position_source("depth").unwrap_err().contains("gsdepth|points|blend"));
        assert!(normal_source("mesh").unwrap_err().contains("geometry|head"));
    }

    /// Cameras on a tilted orbit around a subject: re-framing is a rigid
    /// motion that puts the point they all look at at the origin and leaves
    /// every camera-to-camera distance alone.
    #[test]
    fn upright_is_rigid_and_centres_the_subject_the_cameras_look_at() {
        let subject = [1.0f32, 2.0, 3.0];
        let cams: Vec<Camera> = (0..6)
            .map(|k| {
                let a = k as f32 * std::f32::consts::TAU / 6.0;
                let eye = [subject[0] + 4.0 * a.cos(), subject[1] + 1.0, subject[2] + 4.0 * a.sin()];
                Camera::look_at(eye, subject, [0.0, -1.0, 0.0], 60.0, 64, 48)
            })
            .collect();
        let splats = Splats { means: subject.to_vec(), quats: vec![1.0, 0.0, 0.0, 0.0], scales: vec![0.1; 3], opacities: vec![0.5], colors: vec![0.5; 3], sh_rest: None };
        let (moved, cams2) = upright(&splats, &cams);
        let centre = &moved.means[..3];
        assert!(centre.iter().all(|v| v.abs() < 1e-3), "the subject should land on the origin, got {centre:?}");
        let pos = |c: &Camera| [c.c2w[3], c.c2w[7], c.c2w[11]];
        let dist = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        for (i, j) in [(0, 1), (0, 3), (2, 5)] {
            assert!((dist(pos(&cams[i]), pos(&cams[j])) - dist(pos(&cams2[i]), pos(&cams2[j]))).abs() < 1e-3, "cameras {i},{j} moved relative to each other");
        }
        assert_eq!(cams2[0].fx, cams[0].fx, "intrinsics are untouched");
    }
}
