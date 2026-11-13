// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Scene tools: inspecting, re-framing, chaining and thinning a Gaussian
//! scene, and turning photographs into the target views a fit needs.
//!
//! Everything here is plain host code over [`Splats`] and camera files; the
//! served actions in [`crate::caps`] and [`crate::caps_tools`] are thin
//! wrappers that decode their inputs, call these and encode the result, so a
//! program embedding the crate and a remote caller run the same logic.
//!
//! Swedish Embedded AB implements 3D reconstruction pipelines that turn
//! photographs into scenes a customer can ship for its clients. If your team
//! needs expertise in Gaussian splatting and photogrammetry then you can
//! procure our services by sending an email to info@swedishembedded.com.

use serde_json::{json, Value};

use crate::opt::TargetView;
use crate::types::{Camera, Splats};

/// What a scene is, in a few numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneInfo {
    pub gaussians: usize,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub mean_opacity: f32,
    /// Spherical-harmonic degree of the view-dependent colour (0 = flat).
    pub sh_degree: u32,
}

/// Summarize `s`.
pub fn info(s: &Splats) -> SceneInfo {
    let (min, max) = s.bounds();
    SceneInfo {
        gaussians: s.len(),
        min,
        max,
        mean_opacity: s.opacities.iter().sum::<f32>() / s.len().max(1) as f32,
        sh_degree: s.sh_rest.as_ref().map_or(0, |(degree, _)| *degree),
    }
}

// ===================== cameras files =====================

/// A `cameras.json` document kept as written, so a tool that moves the
/// cameras rewrites only each `c2w` and leaves intrinsics, lens and shutter
/// as they were.
#[derive(Clone, Debug)]
pub struct CameraSet {
    entries: Vec<Value>,
}

impl CameraSet {
    /// Parse a JSON array of cameras, each with a 16-number `c2w`.
    pub fn parse(raw: &str) -> Result<CameraSet, String> {
        let v: Value = serde_json::from_str(raw).map_err(|e| format!("cameras must be a JSON array: {e}"))?;
        let entries = v.as_array().ok_or("cameras must be a JSON array")?.clone();
        let set = CameraSet { entries };
        set.c2w()?;
        Ok(set)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every camera's camera-to-world matrix, row-major.
    pub fn c2w(&self) -> Result<Vec<[f64; 16]>, String> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let m: Vec<f64> = c["c2w"]
                    .as_array()
                    .ok_or_else(|| format!("camera {i} has no 'c2w'"))?
                    .iter()
                    .map(|x| x.as_f64().ok_or_else(|| format!("camera {i}: 'c2w' holds a non-number")))
                    .collect::<Result<_, _>>()?;
                m.try_into().map_err(|m: Vec<f64>| format!("camera {i}: 'c2w' has {} entries, expected 16", m.len()))
            })
            .collect()
    }

    /// These cameras with their poses replaced by `c2w` (one per camera).
    pub fn with_c2w(&self, c2w: &[[f64; 16]]) -> CameraSet {
        assert_eq!(c2w.len(), self.entries.len(), "one pose per camera");
        let entries = self
            .entries
            .iter()
            .zip(c2w)
            .map(|(c, m)| {
                let mut c = c.clone();
                c["c2w"] = json!(m.to_vec());
                c
            })
            .collect();
        CameraSet { entries }
    }

    /// `parts` laid end to end.
    pub fn concat(parts: Vec<CameraSet>) -> CameraSet {
        CameraSet { entries: parts.into_iter().flat_map(|p| p.entries).collect() }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.entries).expect("plain numbers serialize")
    }
}

// ===================== orient =====================

/// Rigidly re-frame a scene so it opens the right way up.
///
/// A feed-forward reconstruction's world frame is the FIRST camera's frame -
/// `c2w[0]` comes back as the identity - so "up" in the file is whichever way
/// the camera happened to be held, and every viewer shows the scene tipped by
/// that angle.
///
/// The recovered cameras say which way is really up. They orbit the subject,
/// so the normal of the plane they lie in IS the scene's vertical, and their
/// rays intersect at its centre. Both come out of the cameras alone, with no
/// assumption about the subject.
///
/// Rigid, so it changes nothing measurable: means rotate, quaternions
/// compose, scales and opacities are untouched, and the cameras are moved by
/// the same transform so a later fit still lines up. brain's world is y-down,
/// so "up" is -Y; `y_up` flips that for a viewer that expects +Y.
pub fn orient(scene: &Splats, cameras: &CameraSet, y_up: bool) -> Result<(Splats, CameraSet), String> {
    let mats = cameras.c2w()?;
    if mats.len() < 3 {
        return Err(format!("orient needs at least 3 cameras to find the orbit plane, got {}", mats.len()));
    }
    let up_sign = if y_up { 1.0 } else { -1.0 };
    let (rot, centre) = crate::orient::frame_from_cameras(&mats, up_sign);
    let moved: Vec<[f64; 16]> = mats.iter().map(|m| crate::orient::transform_c2w(m, &rot, &centre)).collect();
    Ok((crate::orient::apply(scene, &rot, &centre), cameras.with_c2w(&moved)))
}

// ===================== merge =====================

/// How one chunk of a chain landed on the one before it.
#[derive(Clone, Copy, Debug)]
pub struct Link {
    pub chunk: usize,
    /// The similarity's scale: how much larger this chunk's world was.
    pub scale: f64,
    /// How far the shared cameras land from where they should, worst case.
    pub residual: f64,
    /// `residual` as a fraction of the overlap's own span: the number that
    /// says whether the overlap was big enough.
    pub residual_fraction: f64,
}

/// A chain of chunks as one scene in the first chunk's world.
pub struct Merged {
    pub scene: Splats,
    pub cameras: CameraSet,
    pub links: Vec<Link>,
}

/// Chain chunks of one long capture into a single scene.
///
/// Chunk i+1 is aligned onto chunk i using the `overlap` frames they share
/// (the LAST `overlap` of one are the FIRST `overlap` of the next), and the
/// resulting similarity is composed along the chain so everything lands in
/// the first chunk's world. Composing means drift accumulates - each link's
/// error is carried by every chunk after it - so the per-link residual is
/// reported rather than hidden. `voxel > 0` fuses the doubled surface the
/// overlap leaves, a confident gaussian outvoting a faint duplicate of
/// itself.
pub fn merge(chunks: &[(Splats, CameraSet)], overlap: usize, voxel: f32) -> Result<Merged, String> {
    if chunks.len() < 2 {
        return Err(format!("merge needs at least 2 chunks, got {}", chunks.len()));
    }
    if overlap < 2 {
        return Err("merge: overlap must be at least 2 shared frames; a single shared frame fixes no scale".to_string());
    }
    let poses: Vec<Vec<[f64; 16]>> = chunks.iter().map(|(_, c)| c.c2w()).collect::<Result<_, _>>()?;
    let mut world = crate::align::Sim3::default();
    let mut parts = Vec::with_capacity(chunks.len());
    let mut cameras = Vec::with_capacity(chunks.len());
    let mut links = Vec::new();
    for (i, (scene, set)) in chunks.iter().enumerate() {
        if i > 0 {
            let (prev, cur) = (&poses[i - 1], &poses[i]);
            if prev.len() < overlap || cur.len() < overlap {
                return Err(format!("merge: chunk {i} has fewer cameras than the stated overlap"));
            }
            let tail = &prev[prev.len() - overlap..];
            let head = &cur[..overlap];
            let step = crate::align::sim3_from_cameras(tail, head).ok_or_else(|| format!("merge: chunk {i} does not overlap the one before it usefully"))?;
            let residual = crate::align::camera_residual(tail, head, &step);
            links.push(Link { chunk: i, scale: step.s, residual, residual_fraction: residual / eye_span(tail) });
            world = world.after(&step);
        }
        let moved: Vec<[f64; 16]> = poses[i].iter().map(|m| crate::align::transform_c2w_sim3(m, &world)).collect();
        cameras.push(set.with_c2w(&moved));
        parts.push(crate::align::apply_sim3(scene, &world));
    }
    let mut scene = crate::align::concat(&parts);
    if voxel > 0.0 {
        let weights = scene.opacities.clone();
        scene = crate::prune::voxel_merge(&scene, &weights, voxel, 0);
    }
    Ok(Merged { scene, cameras: CameraSet::concat(cameras), links })
}

/// The largest distance of any camera centre in `poses` from their mean.
fn eye_span(poses: &[[f64; 16]]) -> f64 {
    let eyes: Vec<[f64; 3]> = poses.iter().map(|m| [m[3], m[7], m[11]]).collect();
    let n = eyes.len() as f64;
    let mean = [0, 1, 2].map(|k| eyes.iter().map(|e| e[k]).sum::<f64>() / n);
    eyes.iter().map(|e| (0..3).map(|k| (e[k] - mean[k]).powi(2)).sum::<f64>().sqrt()).fold(0.0, f64::max).max(1e-9)
}

// ===================== prune =====================

/// How [`prune`] thins a scene.
#[derive(Clone, Debug)]
pub struct PruneCfg {
    /// Fuse gaussians on a grid of this world-unit cell; 0 = off, and
    /// `voxel_pixels` decides instead.
    pub voxel: f32,
    /// Fuse at the distance the cameras can resolve, in pixels at the median
    /// sampling rate (needs cameras); 0 = off. Ignored when `voxel` is set.
    pub voxel_pixels: f32,
    /// Drop gaussians fainter than this.
    pub min_opacity: f32,
    /// A reconstruction predicts opacity with no anti-alias filter in mind,
    /// so rendering it through one that compensates energy dims the scene;
    /// this re-solves opacity for that filter (needs cameras).
    pub recalibrate_opacity: bool,
    /// Keep at most this many fused gaussians, the heaviest first; 0 = all.
    pub max_gaussians: usize,
    /// Multiplier on every opacity.
    pub opacity_gain: f32,
    /// A contiguous slice `(start, count)` of the scene's gaussians, for
    /// asking what ONE frame's gaussians look like with nothing else there to
    /// cover for them.
    pub range: Option<(usize, usize)>,
}

/// What [`prune`] did, stage by stage.
pub struct Pruned {
    pub scene: Splats,
    pub before: usize,
    pub after_opacity: usize,
    /// The grid cell the fusion used, in world units; 0 when it did not run.
    pub voxel: f32,
}

/// Parse `START:COUNT` (a [`PruneCfg::range`]).
pub fn parse_range(s: &str) -> Result<(usize, usize), String> {
    let (start, count) = s.split_once(':').ok_or_else(|| format!("range '{s}' wants START:COUNT"))?;
    let num = |t: &str| t.trim().parse::<usize>().map_err(|_| format!("range '{s}' wants START:COUNT, whole numbers"));
    Ok((num(start)?, num(count)?))
}

/// Fuse duplicate gaussians and drop the faint ones.
///
/// A pixel-aligned reconstruction emits one gaussian per source pixel per
/// view, so a surface seen by twelve cameras is represented twelve times
/// over. That is not detail, it is the same detail counted repeatedly, and it
/// buys the optimizer enough freedom to memorise its training views.
///
/// A cell in world units means nothing without knowing how big the scene is.
/// Two gaussians closer together than the finest detail any camera resolved
/// cannot be told apart by any training view, so `voxel_pixels` states the
/// distance to fuse at in pixels and `cameras` convert it.
pub fn prune(scene: &Splats, cameras: Option<&[Camera]>, cfg: &PruneCfg) -> Result<Pruned, String> {
    let (lo, hi) = match cfg.range {
        Some((start, count)) => (start.min(scene.len()), start.saturating_add(count).min(scene.len())),
        None => (0, scene.len()),
    };
    let mut kept = Splats::default();
    for i in (lo..hi).filter(|&i| scene.opacities[i] >= cfg.min_opacity) {
        kept.means.extend_from_slice(&scene.means[i * 3..i * 3 + 3]);
        kept.quats.extend_from_slice(&scene.quats[i * 4..i * 4 + 4]);
        kept.scales.extend_from_slice(&scene.scales[i * 3..i * 3 + 3]);
        kept.opacities.push(scene.opacities[i]);
        kept.colors.extend_from_slice(&scene.colors[i * 3..i * 3 + 3]);
    }
    if cfg.recalibrate_opacity {
        let cams = cameras.ok_or("prune: recalibrate_opacity needs the cameras the scene was reconstructed from")?;
        kept = crate::mip::recalibrate_opacity(&kept, cams, crate::types::RenderOpts::default().eps2d);
    }
    if cfg.opacity_gain != 1.0 {
        for v in kept.opacities.iter_mut() {
            *v = (*v * cfg.opacity_gain).clamp(1e-4, 1.0 - 1e-4);
        }
    }
    let after_opacity = kept.len();
    let mut voxel = cfg.voxel;
    if voxel <= 0.0 && cfg.voxel_pixels > 0.0 {
        let cams = cameras.ok_or("prune: voxel_pixels needs the cameras (it is the cameras that set the scale)")?;
        let mut sigma: Vec<f32> = crate::mip::smoothing_sigma(&kept, cams, cfg.voxel_pixels, &|_, _| true).into_iter().filter(|&v| v > 0.0).collect();
        if sigma.is_empty() {
            return Err("prune: no gaussian is visible from any of those cameras".to_string());
        }
        sigma.sort_by(f32::total_cmp);
        voxel = sigma[sigma.len() / 2];
    }
    if voxel > 0.0 {
        let weights = kept.opacities.clone();
        kept = crate::prune::voxel_merge(&kept, &weights, voxel, cfg.max_gaussians);
    }
    Ok(Pruned { scene: kept, before: scene.len(), after_opacity, voxel })
}

// ===================== photographs as target views =====================

/// Image extensions a folder of photographs may hold.
const IMAGE_EXTS: [&str; 7] = ["ppm", "png", "jpg", "jpeg", "bmp", "tif", "tiff"];

/// The photographs `spec` names, in the order a capture's own frames come:
/// every image of a folder by name, or a comma-separated list of files as
/// given.
pub fn collect_images(spec: &str) -> Result<Vec<String>, String> {
    let p = std::path::Path::new(spec);
    if !p.is_dir() {
        return Ok(spec.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect());
    }
    let mut found: Vec<String> = std::fs::read_dir(p)
        .map_err(|e| format!("{spec}: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str())))
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    found.sort();
    Ok(found)
}

/// The target view a camera and its photograph make, `rgb` being `w x h`
/// interleaved RGB in `[0, 1]`.
///
/// A target view is defined by its camera, and the camera says how big it
/// is, so a photograph of another size is resampled to it - but only within
/// a tolerance: a camera a feed-forward model recovered is written for the
/// model's own grid, whose short side is rounded to its patch size, so it is
/// about 1% off the photograph's aspect by construction. Anything further
/// apart is a real mismatch (portrait against landscape, the wrong folder),
/// and stretching it would have the fit chase the distortion.
pub fn target_view(cam: Camera, rgb: Vec<f32>, w: u32, h: u32, label: &str) -> Result<TargetView, String> {
    if (w, h) == (cam.width, cam.height) {
        return Ok(TargetView::new(cam, rgb));
    }
    let skew = (w as f32 / h as f32) / (cam.width as f32 / cam.height as f32);
    if !(0.95..1.05).contains(&skew) {
        return Err(format!(
            "{label} is {w}x{h} but its camera is {}x{}, a {:.0}% different aspect ratio; resampling would stretch the target and the fit would chase the distortion - crop the photographs to the camera's aspect first",
            cam.width,
            cam.height,
            (skew - 1.0).abs() * 100.0
        ));
    }
    Ok(TargetView::new(cam, imaging::host::resize_bilinear_hwc(&rgb, 3, w, h, cam.width, cam.height)))
}

/// Load the photographs `paths` as the target views of `cameras`, in order.
pub fn load_targets(paths: &[String], cameras: &[Camera]) -> Result<Vec<TargetView>, String> {
    if paths.len() != cameras.len() {
        return Err(format!("{} photographs but {} cameras: there must be one camera per photograph, in the same order", paths.len(), cameras.len()));
    }
    paths
        .iter()
        .zip(cameras)
        .map(|(path, cam)| {
            let img = imaging::load(path).map_err(|e| format!("{path}: {e}"))?;
            target_view(*cam, img.to_hwc_unit(), img.w, img.h, path)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam_json(eye: [f64; 3]) -> Value {
        json!({"c2w": [1.0, 0.0, 0.0, eye[0], 0.0, 1.0, 0.0, eye[1], 0.0, 0.0, 1.0, eye[2], 0.0, 0.0, 0.0, 1.0], "fx": 10.0, "fy": 10.0, "cx": 4.0, "cy": 4.0, "width": 8, "height": 8})
    }

    /// Cameras `from..to` of an orbit of `total` evenly spaced ones.
    fn orbit_part(from: usize, to: usize, total: usize, radius: f64) -> CameraSet {
        let v: Vec<Value> = (from..to)
            .map(|i| {
                let a = i as f64 / total as f64 * std::f64::consts::TAU;
                cam_json([radius * a.cos(), 0.0, radius * a.sin()])
            })
            .collect();
        CameraSet::parse(&serde_json::to_string(&v).unwrap()).unwrap()
    }

    fn orbit(n: usize, radius: f64) -> CameraSet {
        orbit_part(0, n, n, radius)
    }

    fn scene(n: usize) -> Splats {
        let mut s = Splats::default();
        for i in 0..n {
            let f = i as f32;
            s.means.extend_from_slice(&[f * 0.01, 0.0, 1.0]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[0.05; 3]);
            s.opacities.push(0.1 + 0.8 * (i % 2) as f32);
            s.colors.extend_from_slice(&[0.5; 3]);
        }
        s
    }

    /// A camera file moved by a tool keeps everything but the pose.
    #[test]
    fn moving_cameras_keeps_their_intrinsics() {
        let set = orbit(4, 1.0);
        let mut pose = set.c2w().unwrap();
        pose[0][3] = 9.0;
        let moved = set.with_c2w(&pose);
        assert_eq!(moved.c2w().unwrap()[0][3], 9.0);
        let back: Vec<Value> = serde_json::from_str(&moved.to_json()).unwrap();
        assert_eq!(back[0]["fx"], json!(10.0));
        assert_eq!(back.len(), 4);
    }

    #[test]
    fn a_malformed_camera_file_is_refused_with_its_index() {
        assert!(CameraSet::parse("{}").err().unwrap().contains("array"));
        let e = CameraSet::parse(r#"[{"c2w": [1, 2]}]"#).err().unwrap();
        assert!(e.contains("camera 0") && e.contains("16"), "{e}");
    }

    #[test]
    fn orient_needs_an_orbit_and_is_rigid() {
        let s = scene(8);
        assert!(orient(&s, &orbit(2, 1.0), false).err().unwrap().contains("at least 3"));
        let (o, cams) = orient(&s, &orbit(8, 2.0), false).unwrap();
        assert_eq!(o.len(), s.len());
        assert_eq!(o.opacities, s.opacities);
        assert_eq!(cams.len(), 8);
        // rigid: the spread between two gaussians is preserved
        let d = |a: &Splats| ((a.means[0] - a.means[21]).powi(2) + (a.means[1] - a.means[22]).powi(2) + (a.means[2] - a.means[23]).powi(2)).sqrt();
        assert!((d(&o) - d(&s)).abs() < 1e-5);
    }

    #[test]
    fn merge_refuses_a_single_shared_frame_and_a_lone_chunk() {
        let chunk = (scene(4), orbit(6, 1.0));
        assert!(merge(std::slice::from_ref(&chunk), 2, 0.0).err().unwrap().contains("at least 2 chunks"));
        assert!(merge(&[chunk.clone(), chunk.clone()], 1, 0.0).err().unwrap().contains("at least 2"));
        assert!(merge(&[chunk.clone(), chunk], 9, 0.0).err().unwrap().contains("fewer cameras"));
    }

    /// Two chunks of one orbit that share four frames, in the same world,
    /// chain at unit scale with the shared cameras landing on each other.
    #[test]
    fn chunks_sharing_frames_in_one_world_link_at_unit_scale() {
        let (a, b) = (orbit_part(0, 8, 12, 2.0), orbit_part(4, 12, 12, 2.0));
        let m = merge(&[(scene(4), a), (scene(4), b)], 4, 0.0).unwrap();
        assert_eq!(m.scene.len(), 8);
        assert_eq!(m.cameras.len(), 16);
        let link = m.links[0];
        assert!((link.scale - 1.0).abs() < 1e-6 && link.residual < 1e-6, "{link:?}");
    }

    #[test]
    fn prune_drops_the_faint_and_slices_a_range() {
        let s = scene(10);
        let cfg = PruneCfg { voxel: 0.0, voxel_pixels: 0.0, min_opacity: 0.5, recalibrate_opacity: false, max_gaussians: 0, opacity_gain: 1.0, range: None };
        let p = prune(&s, None, &cfg).unwrap();
        assert_eq!((p.before, p.after_opacity, p.scene.len()), (10, 5, 5));
        let sliced = prune(&s, None, &PruneCfg { min_opacity: 0.0, range: Some((2, 3)), ..cfg }).unwrap();
        assert_eq!(sliced.scene.len(), 3);
    }

    #[test]
    fn prune_in_pixels_or_recalibration_needs_cameras() {
        let s = scene(4);
        let cfg = PruneCfg { voxel: 0.0, voxel_pixels: 2.0, min_opacity: 0.0, recalibrate_opacity: false, max_gaussians: 0, opacity_gain: 1.0, range: None };
        assert!(prune(&s, None, &cfg).err().unwrap().contains("needs the cameras"));
        let e = prune(&s, None, &PruneCfg { voxel_pixels: 0.0, recalibrate_opacity: true, ..cfg.clone() }).err().unwrap();
        assert!(e.contains("recalibrate_opacity"), "{e}");
        // a world-unit voxel needs none, and fuses the two gaussians sharing a cell
        let fused = prune(&s, None, &PruneCfg { voxel: 1.0, ..cfg }).unwrap();
        assert_eq!(fused.scene.len(), 1);
    }

    #[test]
    fn a_range_is_start_colon_count() {
        assert_eq!(parse_range("4:9").unwrap(), (4, 9));
        assert!(parse_range("4").is_err() && parse_range("a:b").is_err());
    }

    /// A photograph is resampled to its camera within the aspect tolerance
    /// and refused beyond it.
    #[test]
    fn a_target_view_follows_its_camera_within_the_aspect_tolerance() {
        let cam = Camera::pinhole([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.], 8.0, 8.0, 4.0, 4.0, 8, 8);
        let t = target_view(cam, vec![0.5; 16 * 16 * 3], 16, 16, "a.png").unwrap();
        assert_eq!(t.rgb.len(), 8 * 8 * 3);
        let e = target_view(cam, vec![0.5; 16 * 8 * 3], 16, 8, "b.png").err().unwrap();
        assert!(e.contains("b.png") && e.contains("aspect"), "{e}");
    }

    /// A folder is read in name order and only its images; a list is taken as given.
    #[test]
    fn collect_images_reads_a_folder_in_order_and_a_list_as_given() {
        let d = std::env::temp_dir().join(format!("splat-collect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for n in ["b.PNG", "a.jpeg", "c.ppm", "cameras.json"] {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        let got = collect_images(d.to_str().unwrap()).unwrap();
        let names: Vec<&str> = got.iter().map(|p| p.rsplit('/').next().unwrap()).collect();
        assert_eq!(names, ["a.jpeg", "b.PNG", "c.ppm"]);
        assert_eq!(collect_images("x.png, y.png").unwrap(), ["x.png", "y.png"]);
        let _ = std::fs::remove_dir_all(&d);
    }
}
