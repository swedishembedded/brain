// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The PatchMatch pyramid: every view's range and normal map, coarse to fine.
//!
//! Schedule (Xu & Tao, CVPR 2019, §3.3, simplified to one geometric sweep per
//! level):
//!
//! 1. every photograph becomes an exact box-halved gray + colour pyramid on
//!    the device (`mvs_prepare`);
//! 2. at the COARSEST level every view starts from random planes - range
//!    uniform in inverse range over its [`crate::select::range_bounds`] - and
//!    runs [`StereoCfg::photo_iters`] photometric red-black iterations
//!    (`mvs_pm.wgsl`);
//! 3. then, level by level down to the finest, every view runs
//!    [`StereoCfg::geom_iters`] iterations with the geometric-consistency term
//!    against its sources' CURRENT maps (views are visited in order, so a
//!    later view already sees the earlier ones' refined maps), and hands its
//!    planes to the next finer level (`mvs_upsample.wgsl`);
//! 4. at the finest level every view is filtered for multi-view consistency
//!    (`mvs_filter.wgsl`) and read back as a [`DepthMap`].
//!
//! The finest level is the photograph halved [`StereoCfg::halving`] times.
//! Per reference view, its sources' gray images are gathered into one stack,
//! packed for bilinear sampling (`mvs_quad` into sub-ranges), and so are
//! their current range maps (`region_copy`), so a PatchMatch dispatch binds a
//! fixed five buffers however many views the capture has.
//!
//! Swedish Embedded AB implements GPU multi-view stereo for its clients. If
//! your team needs dense geometry from photographs at speed, you can procure
//! our services by sending an email to info@swedishembedded.com.

use camera::Intrinsics;
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use sfm::camera::Pose;
use sfm::linalg::{mm, mv, sub, transpose, M3, V3};
use splat::types::Camera;

use crate::depth::DepthMap;
use crate::select::{range_bounds, select_sources, SelectCfg, Track};
use crate::{pose_of, Kernels, MvsError, Timings};

/// Most source views one reference is matched against (the size of the
/// kernels' per-view arrays).
pub const MAX_SOURCES: usize = 16;

/// Words of one `MvsCam` record (`wgsl/lib/mvs.wgsl`).
pub const CAM_WORDS: usize = 16 + camera::DEVICE_LENS_WORDS;

/// Planes of the PatchMatch state: range, normal (3), cost.
const STATE_PLANES: u64 = 5;

/// Planes `mvs_prepare` writes: gray, R, G, B.
const IMAGE_PLANES: u64 = 4;

/// A photograph and the camera that took it.
#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    /// Pose and full-resolution calibration. The shutter is taken to be
    /// global: `Camera::shutter` is not modelled.
    pub cam: Camera,
    /// Interleaved 8-bit RGB, `cam.width x cam.height`.
    pub rgb: &'a [u8],
}

/// When a pixel's final hypothesis is kept (`mvs_filter.wgsl`).
#[derive(Clone, Debug, PartialEq)]
pub struct FilterCfg {
    /// Fewest source views that must agree.
    pub min_views: u32,
    /// Agreeing views at which the view part of the confidence saturates.
    pub conf_views: u32,
    /// Largest forward-backward reprojection error, pixels of the map.
    pub max_reproj_px: f32,
    /// Largest relative range disagreement.
    pub max_rel_range: f32,
    /// Smallest triangulation angle an agreeing view needs, degrees.
    pub min_angle_deg: f32,
    /// Largest angle between the normal and the direction to the camera,
    /// degrees.
    pub max_incidence_deg: f32,
    /// Largest PatchMatch cost (1 - NCC, plus the geometric term).
    pub max_cost: f32,
}

impl Default for FilterCfg {
    fn default() -> Self {
        FilterCfg {
            min_views: 2,
            conf_views: 4,
            max_reproj_px: 1.0,
            max_rel_range: 0.01,
            min_angle_deg: 2.0,
            max_incidence_deg: 85.0,
            max_cost: 1.0,
        }
    }
}

/// Multi-view stereo settings. The defaults are for photographs; every
/// pixel-denominated value is in pixels of the level it applies at.
#[derive(Clone, Debug, PartialEq)]
pub struct StereoCfg {
    pub select: SelectCfg,
    /// The finest level is the photograph halved this many times.
    pub halving: u32,
    /// Pyramid levels, the finest included (3: 1/4, 1/2, 1 of the finest).
    pub levels: u32,
    /// Photometric red-black iterations at the coarsest level.
    pub photo_iters: u32,
    /// Iterations with the geometric-consistency term, at every level.
    pub geom_iters: u32,
    /// Patch half-size and sample spacing, pixels (5, 2: 6x6 samples over an
    /// 11x11 window).
    pub window_radius: u32,
    pub window_step: u32,
    /// Bilateral weights: gray difference (gray in [0, 1]) and distance
    /// (pixels) at one standard deviation.
    pub sigma_color: f32,
    pub sigma_spatial: f32,
    /// Weighted gray variance (gray in [0, 1]) below which a patch carries no
    /// photometric evidence and costs the maximum. Low enough that faint
    /// texture - wood grain under a light finish - is still matched; a patch
    /// with none at all is never kept (see `mvs_filter.wgsl`).
    pub min_var: f32,
    /// View selection: a hypothesis matches a view well below `tau_good`,
    /// badly above `tau_bad`; a trusted view weighs exp(-c^2 / 2 sel_sigma^2).
    pub tau_good: f32,
    pub tau_bad: f32,
    pub sel_sigma: f32,
    /// Geometric consistency: weight of the reprojection error and its
    /// truncation, pixels.
    pub geo_weight: f32,
    pub geo_max_px: f32,
    /// Refinement perturbation at the first iteration - relative range and
    /// normal (radians) - halved every iteration.
    pub perturb_range: f32,
    pub perturb_normal: f32,
    pub filter: FilterCfg,
    pub seed: u32,
}

impl Default for StereoCfg {
    fn default() -> Self {
        StereoCfg {
            select: SelectCfg::default(),
            halving: 1,
            levels: 3,
            photo_iters: 6,
            geom_iters: 3,
            window_radius: 5,
            window_step: 2,
            sigma_color: 0.03,
            sigma_spatial: 5.0,
            min_var: 2e-6,
            tau_good: 0.8,
            tau_bad: 1.2,
            sel_sigma: 0.3,
            geo_weight: 0.2,
            geo_max_px: 3.0,
            perturb_range: 0.1,
            perturb_normal: 0.5,
            filter: FilterCfg::default(),
            seed: 1,
        }
    }
}

impl StereoCfg {
    fn validate(&self) -> Result<(), MvsError> {
        let bad = |m: &str| Err(MvsError::Config(m.to_string()));
        if self.select.max_sources == 0 || self.select.max_sources > MAX_SOURCES {
            return bad(&format!("select.max_sources must be 1..={MAX_SOURCES}"));
        }
        if self.levels == 0 {
            return bad("levels must be at least 1");
        }
        if self.window_step == 0 || self.window_radius == 0 {
            return bad("window_radius and window_step must be positive");
        }
        // `v > 0.0` is false for NaN too
        if ![self.sigma_color, self.sigma_spatial, self.sel_sigma].iter().all(|v| *v > 0.0) {
            return bad("sigma_color, sigma_spatial and sel_sigma must be positive");
        }
        if self.filter.max_cost.is_nan() || self.filter.max_cost <= 0.0 {
            return bad("filter.max_cost must be positive");
        }
        Ok(())
    }
}

/// Every view's filtered geometry at the finest level.
#[derive(Clone, Debug)]
pub struct Stereo {
    /// The cameras at the maps' resolution (the photographs' scaled by
    /// exactly `2^-halving`).
    pub cams: Vec<Camera>,
    /// One per view; empty (no measurement) for a view with no sources.
    pub depth: Vec<DepthMap>,
    /// Interleaved RGB in [0, 1] at the maps' resolution - the box-halved
    /// photographs the maps were matched on.
    pub rgb: Vec<Vec<f32>>,
    /// Each view's source views.
    pub sources: Vec<Vec<usize>>,
    pub timings: Timings,
}

/// A view's geometry at one pyramid level.
#[derive(Clone, Copy, Debug)]
struct Level {
    w: u32,
    h: u32,
    /// Floats per plane: `w * h` rounded up to 64, so a plane starts on a
    /// 256-byte boundary and can be bound as a sub-range.
    plane: u32,
    k: Intrinsics,
}

impl Level {
    fn of(cam: &Camera, halving: u32) -> Level {
        let k = cam.intrinsics();
        let s = 0.5f64.powi(halving as i32);
        let (w, h) = (k.width >> halving, k.height >> halving);
        let k = Intrinsics { fx: k.fx * s, fy: k.fy * s, cx: k.cx * s, cy: k.cy * s, width: w, height: h, ..k };
        Level { w, h, plane: (w * h).div_ceil(64) * 64, k }
    }

    fn pixels(&self) -> u32 {
        self.w * self.h
    }
}

/// `X_to = R X_from + t` between two poses.
pub(crate) fn relative(from: &Pose, to: &Pose) -> (M3, V3) {
    let r = mm(&to.r, &transpose(&from.r));
    (r, sub(to.t, mv(&r, from.t)))
}

/// One `MvsCam` record.
pub(crate) fn cam_words(r: &M3, t: V3, k: &Intrinsics) -> [u32; CAM_WORDS] {
    let mut w = [0u32; CAM_WORDS];
    for row in 0..3 {
        for c in 0..3 {
            w[4 * row + c] = f(r[3 * row + c] as f32);
        }
        w[4 * row + 3] = f(t[row] as f32);
    }
    w[12] = k.width;
    w[13] = k.height;
    w[16..].copy_from_slice(&k.device_lens());
    w
}

/// The `MvsCam` record of `view` relative to `reference`: the rigid transform
/// from `reference`'s camera frame into `view`'s, and `view`'s lens and
/// size - what a kernel testing points of `reference` against `view`'s
/// measurements (`lib/mvs.wgsl`'s `mvs_seen`, `mvs_measured`) is handed.
pub fn cam_record(reference: &splat::types::Camera, view: &splat::types::Camera) -> [u32; CAM_WORDS] {
    let (r, t) = relative(&pose_of(reference), &pose_of(view));
    cam_words(&r, t, &view.intrinsics())
}

/// A `MvsCam` record carrying camera-to-world: rotation rows of `c2w`, the
/// centre as translation.
pub(crate) fn c2w_words(pose: &Pose, k: &Intrinsics) -> [u32; CAM_WORDS] {
    cam_words(&transpose(&pose.r), pose.centre(), k)
}

/// Per-pass PatchMatch inputs that vary.
struct Pass {
    color: u32,
    pass_id: u32,
    flags: u32,
    perturb: (f32, f32),
}

const FLAG_GEO: u32 = 1;
const FLAG_SCORE_ONLY: u32 = 2;

/// Everything one reference view's dispatches share, with the per-pass words
/// left to [`RefCtx::pm_params`].
struct RefCtx {
    lvl: Level,
    /// The level's stack plane: every view's buffers at this level, and every
    /// slot of a stack, are this many floats apart.
    plane: u32,
    bounds: (f32, f32),
    srcs: Vec<[u32; CAM_WORDS]>,
}

impl RefCtx {
    fn new(v: usize, lvls: &[Level], plane: u32, bounds: (f32, f32), sources: &[usize], poses: &[Pose]) -> RefCtx {
        let srcs = sources
            .iter()
            .map(|&s| {
                let (r, t) = relative(&poses[v], &poses[s]);
                cam_words(&r, t, &lvls[s].k)
            })
            .collect();
        RefCtx { lvl: lvls[v], plane, bounds, srcs }
    }

    fn pm_params(&self, cfg: &StereoCfg, pass: &Pass) -> Vec<u32> {
        let mut p = vec![
            self.lvl.w,
            self.lvl.h,
            self.plane,
            self.srcs.len() as u32,
            pass.color,
            pass.pass_id,
            cfg.seed,
            pass.flags,
            f(self.bounds.0),
            f(self.bounds.1),
            f(cfg.sigma_color),
            f(cfg.sigma_spatial),
            cfg.window_radius,
            cfg.window_step,
            0,
            0,
            f(cfg.tau_good),
            f(cfg.tau_bad),
            f(cfg.sel_sigma),
            f(cfg.min_var),
            f(cfg.geo_weight),
            f(cfg.geo_max_px),
            f(pass.perturb.0),
            f(pass.perturb.1),
        ];
        p.extend_from_slice(&self.lvl.k.device_lens());
        self.push_sources(&mut p);
        p
    }

    fn filter_params(&self, c: &FilterCfg) -> Vec<u32> {
        let mut p = vec![
            self.lvl.w,
            self.lvl.h,
            self.plane,
            self.srcs.len() as u32,
            c.min_views,
            c.conf_views,
            0,
            0,
            f(c.max_reproj_px),
            f(c.max_rel_range),
            f(c.min_angle_deg.to_radians()),
            f(c.max_incidence_deg.to_radians().cos()),
            f(c.max_cost),
            0,
            0,
            0,
        ];
        p.extend_from_slice(&self.lvl.k.device_lens());
        self.push_sources(&mut p);
        p
    }

    fn push_sources(&self, p: &mut Vec<u32>) {
        for s in &self.srcs {
            p.extend_from_slice(s);
        }
        p.resize(p.len() + (MAX_SOURCES - self.srcs.len()) * CAM_WORDS, 0);
    }
}

/// The unit-ray planes of a level (`mvs_rays`).
pub(crate) fn rays_step(gpu: &Gpu, ks: &Kernels, k: &Intrinsics, plane: u32, out: &DeviceBuffer) -> Step {
    let mut p = vec![k.width, k.height, plane, 0];
    p.extend_from_slice(&k.device_lens());
    gpu.step(ks.rays, &[out], &p, k.width * k.height)
}

/// Copy the first `n` floats of `src` to `dst` at word `at`.
fn copy_step(gpu: &Gpu, ks: &Kernels, src: &DeviceBuffer, n: u32, dst: &DeviceBuffer, at: u64) -> Step {
    gpu.step_sliced(ks.region_copy, &[src, dst], &[(0, 0), (at, n as u64)], &[1, n, n, 0], n)
}

/// The perturbation of iteration `t` of a level's schedule.
fn perturb(cfg: &StereoCfg, t: u32) -> (f32, f32) {
    let s = 0.5f32.powi(t as i32);
    ((cfg.perturb_range * s).max(0.002), (cfg.perturb_normal * s).max(0.02))
}

/// Run multi-view stereo over `views`, with sources and range bounds read off
/// `tracks` (see [`crate::select`]).
pub fn depth_maps(gpu: &Gpu, ks: &Kernels, views: &[View], tracks: &[Track], cfg: &StereoCfg) -> Result<Stereo, MvsError> {
    cfg.validate()?;
    for (i, v) in views.iter().enumerate() {
        let expected = v.cam.width as usize * v.cam.height as usize * 3;
        if v.rgb.len() != expected {
            return Err(MvsError::ImageSize { view: i, expected, got: v.rgb.len() });
        }
    }
    let cams: Vec<Camera> = views.iter().map(|v| v.cam).collect();
    let sources = select_sources(&cams, tracks, &cfg.select);
    let bounds = range_bounds(&cams, tracks, &cfg.select);
    let poses: Vec<Pose> = cams.iter().map(pose_of).collect();
    let active: Vec<usize> = (0..views.len()).filter(|&v| !sources[v].is_empty() && bounds[v].is_some()).collect();
    let mut timings = Timings::default();
    let finest = cfg.halving;
    let coarsest = cfg.halving + cfg.levels - 1;
    let max_src = sources.iter().map(Vec::len).max().unwrap_or(0).max(1) as u64;

    // 1. image pyramids
    let t = std::time::Instant::now();
    let mut images: Vec<Vec<DeviceBuffer>> = Vec::with_capacity(views.len());
    for v in views {
        let words: Vec<u32> = v.rgb.chunks(4).map(|c| c.iter().enumerate().fold(0u32, |w, (i, b)| w | (*b as u32) << (8 * i))).collect();
        let bytes = gpu.storage(words.len() as u64);
        gpu.write(&bytes, &words);
        let mut levels = Vec::new();
        for h in finest..=coarsest {
            let lv = Level::of(&v.cam, h);
            let out = gpu.storage(IMAGE_PLANES * lv.plane as u64);
            let p = [v.cam.width, v.cam.height, lv.w, lv.h, 1 << h, lv.plane, 0, 0];
            gpu.submit(&[], &[gpu.step(ks.prepare, &[&bytes, &out], &p, lv.pixels())]);
            levels.push(out);
        }
        images.push(levels);
    }
    gpu.poll_wait();
    timings.push("prepare images", t);

    let mut pass_id = 0u32;
    let mut states: Vec<Option<DeviceBuffer>> = vec![None; views.len()];
    for h in (finest..=coarsest).rev() {
        let li = (h - finest) as usize;
        let lvls: Vec<Level> = cams.iter().map(|c| Level::of(c, h)).collect();
        let plane = lvls.iter().map(|l| l.plane).max().unwrap_or(64) as u64;
        let limit = gpu.max_storage_binding_bytes();
        let stack_bytes = max_src * plane * 8;
        if stack_bytes > limit {
            return Err(MvsError::TooLarge { bytes: stack_bytes, limit });
        }
        let quad_stack = gpu.storage(2 * max_src * plane);
        let depth_stack = gpu.storage(max_src * plane);
        let rays = gpu.storage(3 * plane);
        // `active` views all have bounds
        let ctx = |v: usize| RefCtx::new(v, &lvls, plane as u32, bounds[v].unwrap_or_default(), &sources[v], &poses);
        // the ray map of reference v, its sources' packed gray images and
        // (for the geometric term) their current ranges
        let gather = |v: usize, with_depth: bool, states: &[Option<DeviceBuffer>]| -> Vec<Step> {
            let mut steps = vec![rays_step(gpu, ks, &lvls[v].k, plane as u32, &rays)];
            for (k, &s) in sources[v].iter().enumerate() {
                let slot = (2 * k as u64 * plane, 2 * plane);
                let p = [lvls[s].w, lvls[s].h, 0, 0];
                steps.push(gpu.step_sliced(ks.quad, &[&images[s][li], &quad_stack], &[(0, 0), slot], &p, lvls[s].pixels()));
                if let (true, Some(st)) = (with_depth, &states[s]) {
                    steps.push(copy_step(gpu, ks, st, lvls[s].pixels(), &depth_stack, k as u64 * plane));
                }
            }
            steps
        };
        let pm = |v: usize, c: &RefCtx, state: &DeviceBuffer, pass: &Pass| -> Step {
            let p = c.pm_params(cfg, pass);
            gpu.step(ks.pm, &[&images[v][li], &rays, &quad_stack, &depth_stack, state], &p, c.lvl.w.div_ceil(2) * c.lvl.h)
        };
        let iterate = |v: usize, state: &DeviceBuffer, iters: u32, t0: u32, flags: u32, pass_id: &mut u32| -> Vec<Step> {
            let c = ctx(v);
            let mut steps = Vec::new();
            for it in 0..iters {
                for color in 0..2 {
                    *pass_id += 1;
                    steps.push(pm(v, &c, state, &Pass { color, pass_id: *pass_id, flags, perturb: perturb(cfg, t0 + it) }));
                }
            }
            steps
        };

        let t = std::time::Instant::now();
        let first = h == coarsest;
        for &v in &active {
            let state = gpu.storage(STATE_PLANES * plane);
            let mut steps = gather(v, false, &states);
            let mut clears: Vec<&DeviceBuffer> = Vec::new();
            if first {
                // a cleared state holds no hypothesis, so the scoring pass
                // draws a random one for every pixel
                clears.push(&state);
            } else {
                let parent = states[v].as_ref().expect("the coarser level ran this view");
                let (pl, cl) = (Level::of(&cams[v], h + 1), &lvls[v]);
                let mut p = vec![cl.w, cl.h, plane as u32, pl.w, pl.h, prev_plane(&cams, h + 1), 0, 0];
                p.extend_from_slice(&cl.k.device_lens());
                steps.push(gpu.step(ks.upsample, &[parent, &images[v][li + 1], &images[v][li], &rays, &state], &p, cl.pixels()));
            }
            steps.extend(iterate(v, &state, 1, 0, FLAG_SCORE_ONLY, &mut pass_id));
            if first {
                steps.extend(iterate(v, &state, cfg.photo_iters, 0, 0, &mut pass_id));
            }
            gpu.submit(&clears, &steps);
            states[v] = Some(state);
        }
        gpu.poll_wait();
        timings.push(format!("level 1/{} {}", 1u32 << h, if first { "photometric" } else { "upsample" }), t);

        let t = std::time::Instant::now();
        let t0 = if first { cfg.photo_iters } else { 2 };
        for &v in &active {
            let mut steps = gather(v, true, &states);
            let state = states[v].as_ref().expect("initialised above");
            steps.extend(iterate(v, state, cfg.geom_iters, t0, FLAG_GEO, &mut pass_id));
            // cleared first: a source with no map must read as "no range"
            gpu.submit(&[&depth_stack], &steps);
        }
        gpu.poll_wait();
        timings.push(format!("level 1/{} geometric", 1u32 << h), t);
    }

    // 4. filter and read back
    let t = std::time::Instant::now();
    let lvls: Vec<Level> = cams.iter().map(|c| Level::of(c, finest)).collect();
    let plane = lvls.iter().map(|l| l.plane).max().unwrap_or(64) as u64;
    let depth_stack = gpu.storage(max_src * plane);
    let rays = gpu.storage(3 * plane);
    let filtered = gpu.storage(STATE_PLANES * plane);
    let mut depth = Vec::with_capacity(views.len());
    let mut rgb = Vec::with_capacity(views.len());
    for v in 0..views.len() {
        let lv = lvls[v];
        let n = lv.pixels() as usize;
        let img = gpu.read(&images[v][0], (IMAGE_PLANES * lv.plane as u64) as usize);
        let pl = lv.plane as usize;
        rgb.push((0..n).flat_map(|i| [img[pl + i], img[2 * pl + i], img[3 * pl + i]]).collect());
        let Some(state) = states[v].as_ref() else {
            depth.push(DepthMap::empty(lv.w, lv.h));
            continue;
        };
        let c = RefCtx::new(v, &lvls, plane as u32, (0.0, 0.0), &sources[v], &poses);
        let mut steps = vec![rays_step(gpu, ks, &lv.k, plane as u32, &rays)];
        for (k, &s) in sources[v].iter().enumerate() {
            if let Some(st) = &states[s] {
                steps.push(copy_step(gpu, ks, st, lvls[s].pixels(), &depth_stack, k as u64 * plane));
            }
        }
        steps.push(gpu.step(ks.filter, &[state, &rays, &depth_stack, &filtered], &c.filter_params(&cfg.filter), lv.pixels()));
        gpu.submit(&[&depth_stack], &steps);
        let out = gpu.read(&filtered, (STATE_PLANES * plane) as usize);
        let p = plane as usize;
        depth.push(DepthMap {
            width: lv.w,
            height: lv.h,
            range: out[..n].to_vec(),
            normal: (0..n).flat_map(|i| [out[p + i], out[2 * p + i], out[3 * p + i]]).collect(),
            conf: out[4 * p..4 * p + n].to_vec(),
        });
    }
    timings.push("filter + read back", t);
    let cams = cams.iter().zip(&lvls).map(|(c, l)| Camera::with_intrinsics(c.c2w, &l.k)).collect();
    Ok(Stereo { cams, depth, rgb, sources, timings })
}

/// The plane size a level's per-view buffers were allocated with.
fn prev_plane(cams: &[Camera], h: u32) -> u32 {
    cams.iter().map(|c| Level::of(c, h).plane).max().unwrap_or(64)
}
