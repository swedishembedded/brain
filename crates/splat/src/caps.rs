// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 3D Gaussian Splatting behind the generalized [`capability`] interface -
//! what makes `brain caps splat` / `brain do splat render …`, the D-Bus `Run`
//! method and the event API work with no splat-specific plumbing in the CLI
//! or the transports.
//!
//! Two actions:
//!
//! * **`render`** - ONE-SHOT: a scene (Inria PLY bytes) plus a camera pose in,
//!   one image out. The renderer is sized for a fixed `(gaussian count,
//!   width, height)` at construction (`Renderer::new`), so [`RenderSession`]
//!   caches the built renderer keyed on a cheap digest of `(scene bytes,
//!   width, height)` - the same shape `vqgan::caps::VqAction` caches its
//!   built graph on `(weights, size)`, adapted to a model whose "weights" are
//!   per-request bytes rather than a checkpoint path.
//! * **`fit`** - `.streaming()`: an initial scene plus N posed target views in,
//!   the optimized scene out. Wraps [`crate::opt::fit`]'s new `on_step` hook
//!   (added alongside this module - see that function's doc) to report
//!   [`Progress::step`] and honour [`Invocation::cancel`], the same
//!   cancellation contract `wan::caps`/`flux2::caps`'s training actions use.
//!
//! `view` (the interactive SDL fly-through, `crates/cli/src/splat_cli.rs::view`)
//! is deliberately OUT OF SCOPE here: it is a human-in-the-loop WASD/mouse loop
//! with no request/response shape a capability call can express, and is not
//! served.

use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use gpu_core::Gpu;
use serde_json::json;

use crate::opt::{fit as opt_fit, FitCfg, TargetView};
use crate::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use crate::types::{auto_camera_from_bounds, Camera, Mode, RenderOpts, Splats};
use crate::{Kernels, PIPELINES};

/// The model id used on the CLI (`brain do splat …`), over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "brain/splat";

/// Per-side cap on a requested render size: `Renderer::new` sizes GPU buffers
/// (`width*height*4` f32 pixels, isect scratch, …) directly off `width`/
/// `height` with no bound of its own - an unchecked huge request would OOM
/// deep inside GPU allocation rather than fail cleanly. `8192` covers every
/// real use (an 8K frame) with room to spare.
pub const MAX_SIDE: u32 = 8192;

/// Reject a size `Renderer::new` would over-allocate for, BEFORE any GPU
/// buffer is created - mirrors `vqgan::caps::check_size`'s role.
pub fn check_size(width: u32, height: u32) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err(format!("splat render: width/height must both be > 0 (got {width}x{height})"));
    }
    if width > MAX_SIDE || height > MAX_SIDE {
        return Err(format!("splat render: {width}x{height} exceeds the {MAX_SIDE}px per-side cap"));
    }
    Ok(())
}

fn vec3_param(name: &str, help: &str) -> ParamSpec {
    ParamSpec::new(name, ParamType::Str, help)
}

pub fn render_spec() -> ActionSpec {
    ActionSpec::new("render", "one-shot render of a Gaussian-splat scene from a posed camera (tiled rasterizer)")
        .param(ParamSpec::new("width", ParamType::Int, "output width, px").default(json!(960)).min(1.0).max(MAX_SIDE as f64).step(1.0))
        .param(ParamSpec::new("height", ParamType::Int, "output height, px").default(json!(720)).min(1.0).max(MAX_SIDE as f64).step(1.0))
        .param(ParamSpec::new("fov", ParamType::Float, "vertical field of view, degrees").default(json!(60.0)).min(1.0).max(179.0))
        .param(vec3_param("eye", "camera position 'x,y,z' - with 'target', an explicit pose; omit both to auto-frame the scene"))
        .param(vec3_param("target", "camera look-at point 'x,y,z' (pairs with 'eye')"))
        .param(vec3_param("up", "camera up vector 'x,y,z'").default(json!("0,-1,0")))
        .param(ParamSpec::new("depth", ParamType::Bool, "render alpha-weighted expected depth (replicated to RGB) instead of color").default(json!(false)))
        .param(ParamSpec::new("antialiased", ParamType::Bool, "multiply the AA blur compensation into opacity (gsplat 'antialiased' mode; Inria-trained PLYs expect false)").default(json!(false)))
        .param(vec3_param("bg", "background color 'r,g,b' in [0,1]").default(json!("0,0,0")))
        .input(BlobSpec::new("scene", Media::Bytes, "the scene: Inria-layout binary PLY").required())
        .output(BlobSpec::new("image", Media::Image, "the rendered image, RGB f32 (raw expected-depth values when 'depth' is set, not normalized)"))
}

pub fn fit_spec() -> ActionSpec {
    ActionSpec::new("fit", "optimize a Gaussian-splat scene against N posed target views (AdamW on gaussian parameters, rasterizer backward)")
        .streaming()
        .param(ParamSpec::new(
            "views",
            ParamType::Str,
            "camera array as JSON, one entry per target-view frame in order: [{\"c2w\":[16 floats],\"fx\":...,\"fy\":...,\"cx\":...,\"cy\":...,\"width\":...,\"height\":...}, ...] - the same shape `mirror_cli.rs`'s cameras.json uses",
        ).required())
        .param(ParamSpec::new("iters", ParamType::Int, "optimization steps").default(json!(200)).min(1.0))
        .param(ParamSpec::new("lr", ParamType::Float, "AdamW learning rate").default(json!(5e-3)))
        .param(ParamSpec::new("min_scale", ParamType::Float, "projected-gradient clamp floor for linear scales").default(json!(1e-4)))
        .input(BlobSpec::new("scene", Media::Bytes, "the initial scene to optimize: Inria-layout binary PLY").required())
        .input(BlobSpec::new("video", Media::Video, "N target views: interleaved-HWC f32 RGB frames, one per camera in 'views', same convention as every other video input (capability::blob::decode_video)").required())
        .output(BlobSpec::new("scene", Media::Bytes, "the optimized scene: Inria-layout binary PLY"))
}

/// The full, static capability manifest - safe to build with no scene loaded.
/// `view` is deliberately absent - see the module doc.
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "3D Gaussian Splatting: render a posed view of a scene, or fit a scene to posed target images.", vec![render_spec(), fit_spec()])
}

// ===================== render: a cached, scene-sized session =====================

/// A built renderer for one `(scene, width, height)` - the GPU buffers
/// `Renderer::new` allocates are sized for exactly this gaussian count and
/// resolution, so a session is invalidated (and rebuilt) whenever any of the
/// three changes.
struct RenderSession {
    gpu: Gpu,
    renderer: Renderer,
    gs: GpuSplats,
    /// The scene's own bounds - enough to auto-frame a camera without keeping
    /// the whole (potentially large) `Splats` around.
    bounds: ([f32; 3], [f32; 3]),
}

/// A cheap FNV-1a over the scene bytes plus the render resolution - the cache
/// key. Two requests with the same scene bytes and size share one built
/// renderer; anything else (a different scene, or the same scene at a
/// different resolution) needs its own - exactly the values `Renderer::new`
/// bakes into its GPU buffer sizes.
fn scene_key(bytes: &[u8], width: u32, height: u32) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x100_0000_01b3);
    };
    for &byte in bytes {
        mix(byte as u64);
    }
    mix(width as u64);
    mix(height as u64);
    h
}

fn build_render_session(scene: &Splats, width: u32, height: u32) -> RenderSession {
    let gpu = Gpu::new(PIPELINES);
    let ks = Kernels::at(0);
    let renderer = Renderer::new(&gpu, ks, scene.len(), width, height, 0);
    let gs = GpuSplats::upload(&gpu, scene);
    RenderSession { gpu, renderer, gs, bounds: scene.bounds() }
}

/// Parse a `"x,y,z"` CSV param into `[f32; 3]`. `None` when the param is
/// absent/empty (the caller decides what that means - an optional override
/// vs. a defaulted vector).
fn get_vec3(inv: &Invocation, name: &str) -> Result<Option<[f32; 3]>, String> {
    let Some(s) = inv.get_str(name).filter(|s| !s.trim().is_empty()) else { return Ok(None) };
    let v: Result<Vec<f32>, String> = s.split(',').map(|t| t.trim().parse::<f32>().map_err(|_| format!("splat: '{name}' must be 'x,y,z' floats, got '{s}'"))).collect();
    let v = v?;
    if v.len() != 3 {
        return Err(format!("splat: '{name}' wants 'x,y,z', got {} number(s)", v.len()));
    }
    Ok(Some([v[0], v[1], v[2]]))
}

fn get_vec3_or(inv: &Invocation, name: &str, default: [f32; 3]) -> Result<[f32; 3], String> {
    Ok(get_vec3(inv, name)?.unwrap_or(default))
}

/// Build the camera `render` uses: an explicit `eye`+`target` pose, or (with
/// neither set) an auto-framed one over the scene's own bounds. `eye` xor
/// `target` alone is rejected - a half-specified pose is a caller error, not
/// something to silently complete.
fn camera_from_params(inv: &Invocation, bounds: ([f32; 3], [f32; 3]), width: u32, height: u32, fov: f32) -> Result<Camera, String> {
    let up = get_vec3_or(inv, "up", [0.0, -1.0, 0.0])?;
    match (get_vec3(inv, "eye")?, get_vec3(inv, "target")?) {
        (Some(eye), Some(target)) => Ok(Camera::look_at(eye, target, up, fov, width, height)),
        (None, None) => Ok(auto_camera_from_bounds(bounds, width, height, fov)),
        _ => Err("splat render: 'eye' and 'target' must be given together".to_string()),
    }
}

/// Run one `render` invocation (already validated against [`render_spec`]),
/// reusing `hot` when `(scene, width, height)` matches the last call.
fn render(inv: &Invocation, hot: &Mutex<Option<(u64, RenderSession)>>) -> ActionResult {
    let width = inv.get_i64("width").unwrap_or(960).max(0) as u32;
    let height = inv.get_i64("height").unwrap_or(720).max(0) as u32;
    check_size(width, height)?;
    let fov = inv.get_f64("fov").unwrap_or(60.0) as f32;
    let scene_blob = inv.get_blob("scene").ok_or("splat render: missing required input 'scene'")?;
    let key = scene_key(&scene_blob.bytes, width, height);

    let mut guard = hot.lock().map_err(|_| "splat render: session lock poisoned")?;
    if !matches!(&*guard, Some((k, _)) if *k == key) {
        let scene = crate::ply::parse(&scene_blob.bytes)?;
        *guard = None; // free the old build before allocating the new one
        *guard = Some((key, build_render_session(&scene, width, height)));
    }
    let sess = &mut guard.as_mut().expect("built above").1;

    let cam = camera_from_params(inv, sess.bounds, width, height, fov)?;
    let opts = RenderOpts {
        bg: get_vec3_or(inv, "bg", [0.0; 3])?,
        mode: if inv.get_bool("depth").unwrap_or(false) { Mode::Depth } else { Mode::Color },
        antialiased: inv.get_bool("antialiased").unwrap_or(false),
        ..Default::default()
    };
    sess.renderer.render(&sess.gpu, &sess.gs, &cam, &opts);
    let rgba = sess.renderer.read_rgba(&sess.gpu, width, height);
    let rgb = rgba_to_rgb(&rgba);
    Ok(Outcome::new().set("width", json!(width)).set("height", json!(height)).blob("image", capability::blob::image_blob(&rgb, width, height, 3)))
}

// ===================== fit: no caching - a fresh optimization every call =====================

/// One camera entry of the `views` JSON param - the same shape
/// `mirror_cli.rs::write_cameras_json` writes.
fn parse_views(raw: &str) -> Result<Vec<Camera>, String> {
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| format!("splat fit: 'views' must be a JSON array: {e}"))?;
    let arr = v.as_array().ok_or("splat fit: 'views' must be a JSON array")?;
    arr.iter()
        .enumerate()
        .map(|(i, c)| {
            let c2w: Vec<f32> = c["c2w"]
                .as_array()
                .ok_or_else(|| format!("splat fit: views[{i}] missing 'c2w'"))?
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect();
            let c2w: [f32; 16] = c2w.try_into().map_err(|v: Vec<f32>| format!("splat fit: views[{i}] 'c2w' has {} entries, expected 16", v.len()))?;
            let get_f = |k: &str| c[k].as_f64().ok_or_else(|| format!("splat fit: views[{i}] missing '{k}'"));
            Ok(Camera {
                c2w,
                fx: get_f("fx")? as f32,
                fy: get_f("fy")? as f32,
                cx: get_f("cx")? as f32,
                cy: get_f("cy")? as f32,
                width: c["width"].as_u64().ok_or_else(|| format!("splat fit: views[{i}] missing 'width'"))? as u32,
                height: c["height"].as_u64().ok_or_else(|| format!("splat fit: views[{i}] missing 'height'"))? as u32,
            })
        })
        .collect()
}

/// Run one `fit` invocation (already validated against [`fit_spec`]).
///
/// Cancellation: `on_step` reports [`Progress::step`] then checks
/// `inv.cancel` - the same "poll every step, `Err("cancelled")` if it fired"
/// contract `wan`/`flux2`'s training actions use, adapted to `opt::fit`'s
/// `on_step` hook rather than a cancel token threaded through the library
/// call itself (splat has no `capability` dependency below this module).
fn fit(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    if inv.cancel.is_cancelled() {
        return Err("cancelled".to_string());
    }
    let scene_blob = inv.get_blob("scene").ok_or("splat fit: missing required input 'scene'")?;
    let init = crate::ply::parse(&scene_blob.bytes)?;
    let frames = capability::blob::decode_video(inv, "video")?;
    let views_raw = inv.get_str("views").ok_or("splat fit: missing required param 'views'")?;
    let cams = parse_views(&views_raw)?;
    if cams.len() != frames.len() {
        return Err(format!("splat fit: {} cameras in 'views' but {} frames in 'video'", cams.len(), frames.len()));
    }
    let targets: Vec<TargetView> = cams
        .into_iter()
        .zip(frames)
        .enumerate()
        .map(|(i, (cam, (rgb, w, h)))| {
            if (w, h) != (cam.width, cam.height) {
                return Err(format!("splat fit: view {i} is {}x{} but its frame is {w}x{h}", cam.width, cam.height));
            }
            Ok(TargetView { cam, rgb })
        })
        .collect::<Result<_, _>>()?;

    let iters = inv.get_i64("iters").unwrap_or(200).max(1) as usize;
    let cfg = FitCfg {
        iters,
        lr: inv.get_f64("lr").unwrap_or(5e-3) as f32,
        min_scale: inv.get_f64("min_scale").unwrap_or(1e-4) as f32,
        log_every: 0,
    };

    let gpu = Gpu::new(PIPELINES);
    let ks = Kernels::at(0);
    let cancel = inv.cancel.clone();
    let mut on_step = |it: usize, mse: f32| -> bool {
        progress(Progress::step((it + 1) as u32, iters as u32, format!("mse {mse}")));
        !cancel.is_cancelled()
    };
    let (scene, mse) = opt_fit(&gpu, ks, &init, &targets, &cfg, &mut on_step);
    if inv.cancel.is_cancelled() {
        return Err("cancelled".to_string());
    }
    let bytes = crate::ply::serialize(&scene)?;
    Ok(Outcome::new().set("mse", json!(mse)).blob("scene", Blob::new(Media::Bytes, bytes)))
}

// ===================== the provider =====================

/// The executable splat model behind the manifest. Construction is free: no
/// weights to load - every request supplies its own scene.
pub struct SplatProvider {
    /// `render`'s built-renderer cache - see [`RenderSession`].
    hot: Arc<Mutex<Option<(u64, RenderSession)>>>,
}

impl SplatProvider {
    pub fn new() -> SplatProvider {
        SplatProvider { hot: Arc::new(Mutex::new(None)) }
    }
}

impl Default for SplatProvider {
    fn default() -> Self {
        SplatProvider::new()
    }
}

impl Provider for SplatProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "render" => Some(Arc::new(RenderAction { hot: self.hot.clone() }) as Arc<dyn Action>),
            "fit" => Some(Arc::new(FitAction) as Arc<dyn Action>),
            _ => None,
        }
    }
}

struct RenderAction {
    hot: Arc<Mutex<Option<(u64, RenderSession)>>>,
}

impl Action for RenderAction {
    fn spec(&self) -> ActionSpec {
        render_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        render(inv, &self.hot)
    }
}

struct FitAction;

impl Action for FitAction {
    fn spec(&self) -> ActionSpec {
        fit_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        fit(inv, progress)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::{blob::video_blob, CancelToken};
    use std::sync::atomic::{AtomicU32, Ordering};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as f32 / (1u64 << 31) as f32
        }
    }

    fn scene(n: usize, seed: u64) -> Splats {
        let mut r = Lcg(seed);
        let mut s = Splats::default();
        for _ in 0..n {
            s.means.extend_from_slice(&[(r.next() - 0.5) * 2.0, (r.next() - 0.5) * 2.0, 3.0 + r.next() * 2.0]);
            s.quats.extend_from_slice(&[0.5 + r.next(), r.next() - 0.5, r.next() - 0.5, r.next() - 0.5]);
            s.scales.extend_from_slice(&[0.1 + r.next() * 0.15, 0.1 + r.next() * 0.15, 0.1 + r.next() * 0.15]);
            s.opacities.push(0.35 + 0.5 * r.next());
            s.colors.extend_from_slice(&[r.next(), r.next(), r.next()]);
        }
        s
    }

    fn camera_json(c: &Camera) -> serde_json::Value {
        json!({"c2w": c.c2w.to_vec(), "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy, "width": c.width, "height": c.height})
    }

    /// Bits, not values - `differing_bits` names the idiom already used by
    /// `crates/ltxv/tests/block_pipeline.rs` and `crates/vae/tests/blocks3d_norm.rs`:
    /// count of f32 words whose `to_bits()` disagree.
    fn differing_bits(a: &[f32], b: &[f32]) -> usize {
        assert_eq!(a.len(), b.len(), "differing_bits: length mismatch ({} vs {})", a.len(), b.len());
        a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
    }

    /// Per-field bit/tolerance diff between two same-shaped scenes - shared by
    /// the determinism probe (step 0) and the fit-caps-vs-library check
    /// (same measurement decides both).
    struct SceneDiff {
        bits: usize,
        max_means: f32,
        max_scales: f32,
        max_quats: f32,
        max_colors: f32,
        max_opacities: f32,
    }

    fn diff_scene(a: &Splats, b: &Splats) -> SceneDiff {
        assert_eq!(a.len(), b.len(), "diff_scene: gaussian count mismatch");
        let dmax = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        SceneDiff {
            bits: differing_bits(&a.means, &b.means)
                + differing_bits(&a.scales, &b.scales)
                + differing_bits(&a.quats, &b.quats)
                + differing_bits(&a.colors, &b.colors)
                + differing_bits(&a.opacities, &b.opacities),
            max_means: dmax(&a.means, &b.means),
            max_scales: dmax(&a.scales, &b.scales),
            max_quats: dmax(&a.quats, &b.quats),
            max_colors: dmax(&a.colors, &b.colors),
            max_opacities: dmax(&a.opacities, &b.opacities),
        }
    }

    /// Assert `d` is within item 3's PLY-roundtrip tolerance table
    /// (means/quats 1e-6, scales 1e-4, colors/opacities 1e-5).
    fn assert_within_tolerance(d: &SceneDiff) {
        assert!(d.max_means < 1e-6, "means: max deviation {:e}", d.max_means);
        assert!(d.max_quats < 1e-6, "quats: max deviation {:e}", d.max_quats);
        assert!(d.max_scales < 1e-4, "scales: max deviation {:e}", d.max_scales);
        assert!(d.max_colors < 1e-5, "colors: max deviation {:e}", d.max_colors);
        assert!(d.max_opacities < 1e-5, "opacities: max deviation {:e}", d.max_opacities);
    }

    /// A small deterministic fit problem: a "truth" scene rendered from a few
    /// cameras to build target views, and a perturbed `init` to optimize back
    /// toward it - the same shape `s5_bwd_fit.rs::fit_recovers_perturbed_scene`
    /// uses, sized down for a fast capability-level test.
    fn fixture_fit_inputs() -> (Gpu, Kernels, Splats, Vec<TargetView>, FitCfg) {
        let gpu = Gpu::new(PIPELINES);
        let ks = Kernels::at(0);
        let truth = scene(8, 0x60a1);
        let cams = [
            Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, 24, 24),
            Camera::look_at([1.0, -0.3, 0.3], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, 24, 24),
        ];
        let mut ren = Renderer::new(&gpu, ks, truth.len(), 24, 24, 0);
        let gst = GpuSplats::upload(&gpu, &truth);
        let opts = RenderOpts::default();
        let targets: Vec<TargetView> = cams
            .iter()
            .map(|c| {
                ren.render(&gpu, &gst, c, &opts);
                let img = ren.read_rgba(&gpu, c.width, c.height);
                TargetView { cam: *c, rgb: rgba_to_rgb(&img) }
            })
            .collect();

        let mut init = truth.clone();
        let mut r = Lcg(0xd00d);
        for v in init.colors.iter_mut() {
            *v = (*v + (r.next() - 0.5) * 0.4).clamp(0.0, 1.0);
        }
        for v in init.means.iter_mut() {
            *v += (r.next() - 0.5) * 0.08;
        }
        let cfg = FitCfg { iters: 40, lr: 5e-3, min_scale: 1e-4, log_every: 0 };
        (gpu, ks, init, targets, cfg)
    }

    fn scene_mse(gpu: &Gpu, ks: Kernels, s: &Splats, targets: &[TargetView]) -> f64 {
        let (mw, mh) = targets.iter().fold((0u32, 0u32), |(mw, mh), t| (mw.max(t.cam.width), mh.max(t.cam.height)));
        let mut ren = Renderer::new(gpu, ks, s.len(), mw, mh, 0);
        let gs = GpuSplats::upload(gpu, s);
        let opts = RenderOpts::default();
        let mut acc = 0.0f64;
        for t in targets {
            ren.render(gpu, &gs, &t.cam, &opts);
            let img = ren.read_rgba(gpu, t.cam.width, t.cam.height);
            let px = (t.cam.width * t.cam.height) as usize;
            let mut l = 0.0f64;
            for i in 0..px {
                for c in 0..3 {
                    let d = img[i * 4 + c] - t.rgb[i * 3 + c];
                    l += (d * d) as f64;
                }
            }
            acc += l / (px as f64 * 3.0);
        }
        acc / targets.len() as f64
    }

    fn fit_invocation(init: &Splats, targets: &[TargetView], cfg: &FitCfg) -> Invocation {
        let scene_bytes = crate::ply::serialize(init).unwrap();
        let frames: Vec<(Vec<f32>, u32, u32)> = targets.iter().map(|t| (t.rgb.clone(), t.cam.width, t.cam.height)).collect();
        let video = video_blob(&frames).unwrap();
        let views_json = serde_json::to_string(&targets.iter().map(|t| camera_json(&t.cam)).collect::<Vec<_>>()).unwrap();
        Invocation::new()
            .set("views", json!(views_json))
            .set("iters", json!(cfg.iters as i64))
            .set("lr", json!(cfg.lr as f64))
            .set("min_scale", json!(cfg.min_scale as f64))
            .blob("scene", Blob::new(Media::Bytes, scene_bytes))
            .blob("video", video)
    }

    #[test]
    fn manifest_declares_render_and_fit_but_not_view() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 2);
        assert!(m.actions.iter().any(|a| a.name == "render" && !a.streaming));
        let f = m.actions.iter().find(|a| a.name == "fit").expect("fit");
        assert!(f.streaming);
        assert!(!m.actions.iter().any(|a| a.name == "view"), "the interactive viewer must never be advertised as a servable action");
    }

    /// SUB-BLOCKER (item 4, step 0): is `opt::fit` run-to-run bit-deterministic?
    /// Runs it twice on IDENTICAL inputs and diffs the results bitwise,
    /// printing the answer either way - this is what selects, below, whether
    /// `fit caps-vs-library` is asserted at bit-identity or at item 3's
    /// tolerance table.
    #[test]
    fn determinism_probe_fit_twice_on_identical_inputs() {
        let (gpu, ks, init, targets, cfg) = fixture_fit_inputs();
        let (s1, mse1) = opt_fit(&gpu, ks, &init, &targets, &cfg, &mut |_, _| true);
        let (s2, mse2) = opt_fit(&gpu, ks, &init, &targets, &cfg, &mut |_, _| true);
        let d = diff_scene(&s1, &s2);
        let mse_bits = (mse1.to_bits() != mse2.to_bits()) as usize;
        if d.bits == 0 && mse_bits == 0 {
            println!("determinism probe: 0 differing bits (mse {mse1} both runs)");
        } else {
            println!(
                "determinism probe: NOT bit-identical - max|dmeans|={:e} max|dscales|={:e} max|dquats|={:e} max|dcolors|={:e} max|dopacities|={:e} mse {mse1} vs {mse2}",
                d.max_means, d.max_scales, d.max_quats, d.max_colors, d.max_opacities
            );
        }
    }

    /// `render` vs the direct renderer path on identical fixture+camera+size
    /// inputs must be bit-identical: same GPU pipeline, same inputs - any
    /// difference is a wrapper bug, not something to tolerate.
    ///
    /// "Identical inputs" means identical PLY BYTES on both sides: the wire
    /// format `render` accepts is PLY, so the direct comparison must also go
    /// through `ply::parse` on the same bytes rather than the pre-serialize
    /// `Splats` - otherwise item 3's own (already-verified, sub-ULP) PLY
    /// round-trip quantization shows up here as a false "wrapper bug": a
    /// gaussian's alpha-compositing order is sensitive enough to a sub-ULP
    /// position/scale nudge that many PIXELS can flip from a change this
    /// small, despite every input value differing only in its last bit.
    #[test]
    fn render_caps_matches_the_direct_renderer_path() {
        let scene_bytes = crate::ply::serialize(&scene(6, 0xC0FFEE)).unwrap();
        let s = crate::ply::parse(&scene_bytes).unwrap();
        let (width, height) = (32u32, 24u32);
        let fov = 60.0f32;
        let cam = auto_camera_from_bounds(s.bounds(), width, height, fov);

        let gpu = Gpu::new(PIPELINES);
        let ks = Kernels::at(0);
        let mut renderer = Renderer::new(&gpu, ks, s.len(), width, height, 0);
        let gs = GpuSplats::upload(&gpu, &s);
        renderer.render(&gpu, &gs, &cam, &RenderOpts::default());
        let direct_rgb = rgba_to_rgb(&renderer.read_rgba(&gpu, width, height));

        let provider = SplatProvider::new();
        let act = provider.action("render").expect("render action");
        let inv = act
            .spec()
            .validate(
                Invocation::new()
                    .set("width", json!(width))
                    .set("height", json!(height))
                    .set("fov", json!(fov as f64))
                    .blob("scene", Blob::new(Media::Bytes, scene_bytes)),
            )
            .unwrap();
        let out = act.run(&inv, &mut |_| {}).expect("render");
        let img = out.blobs.get("image").expect("image blob");
        let caps_rgb: Vec<f32> = img.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();

        let k = differing_bits(&direct_rgb, &caps_rgb);
        println!("render caps-vs-direct: {k} differing bits of {}", direct_rgb.len());
        assert_eq!(k, 0, "the render action must reproduce the direct renderer path exactly");
    }

    /// `fit` vs calling `opt::fit` directly with a no-op `on_step`, at the
    /// determinism probe's measured strictness: bit-identity, or item 3's
    /// tolerance table. Also prints `mse {start} -> {end}` unconditionally and
    /// asserts real optimization happened (finite, strictly improved).
    #[test]
    fn fit_caps_matches_the_library_call() {
        let (gpu, ks, init, targets, cfg) = fixture_fit_inputs();
        let start = scene_mse(&gpu, ks, &init, &targets);

        let (lib_scene, lib_mse) = opt_fit(&gpu, ks, &init, &targets, &cfg, &mut |_, _| true);

        let provider = SplatProvider::new();
        let act = provider.action("fit").expect("fit action");
        let inv = act.spec().validate(fit_invocation(&init, &targets, &cfg)).unwrap();
        let out = act.run(&inv, &mut |_| {}).expect("fit");
        let caps_mse = out.outputs["mse"].as_f64().expect("mse output") as f32;
        let caps_scene = crate::ply::parse(&out.blobs.get("scene").expect("scene blob").bytes).unwrap();

        let d = diff_scene(&lib_scene, &caps_scene);
        if d.bits == 0 {
            println!("fit caps-vs-library: 0 differing bits");
        } else {
            println!(
                "fit caps-vs-library: max|dmeans|={:e} max|dscales|={:e} max|dquats|={:e} max|dcolors|={:e} max|dopacities|={:e}",
                d.max_means, d.max_scales, d.max_quats, d.max_colors, d.max_opacities
            );
            assert_within_tolerance(&d);
        }
        assert!((lib_mse as f64) < start, "the direct library call did not improve either: start {start:.6} end {lib_mse:.6}");
        println!("mse {start} -> {caps_mse}");
        assert!(caps_mse.is_finite() && (caps_mse as f64) < start, "fit did not improve: start {start:.6} end {caps_mse:.6}");
    }

    /// Cancelling from inside the progress callback at iteration 3 must abort
    /// the loop within a handful of steps and surface `Err("cancelled")`.
    #[test]
    fn fit_cancellation_aborts_early() {
        let (_gpu, _ks, init, targets, mut cfg) = fixture_fit_inputs();
        cfg.iters = 50; // plenty of headroom past the iteration-3 cancel point
        let provider = SplatProvider::new();
        let act = provider.action("fit").expect("fit action");
        let mut inv = act.spec().validate(fit_invocation(&init, &targets, &cfg)).unwrap();
        let cancel = CancelToken::armed();
        inv.cancel = cancel.clone();

        let last_step = Arc::new(AtomicU32::new(0));
        let last_step2 = last_step.clone();
        let result = act.run(&inv, &mut move |p: Progress| {
            last_step2.store(p.step, Ordering::SeqCst);
            if p.step == 3 {
                cancel.cancel();
            }
        });

        let err = result.expect_err("a mid-run cancellation must fail the action");
        assert_eq!(err, "cancelled");
        let k = last_step.load(Ordering::SeqCst);
        println!("cancelled after {k} steps of {}", cfg.iters);
        assert!(k < cfg.iters as u32, "cancellation did not actually cut the run short");
        assert!(k <= 4, "cancellation took too long to take effect ({k} steps)");
    }

    /// An oversized render request must be a clean `Err`, never a panic deep
    /// inside GPU buffer allocation.
    #[test]
    fn an_oversized_render_request_is_rejected_cleanly() {
        let s = scene(2, 1);
        let provider = SplatProvider::new();
        let act = provider.action("render").expect("render action");
        let inv = act
            .spec()
            .validate(
                Invocation::new()
                    .set("width", json!(1_000_000))
                    .set("height", json!(1_000_000))
                    .blob("scene", Blob::new(Media::Bytes, crate::ply::serialize(&s).unwrap())),
            )
            .unwrap();
        let err = act.run(&inv, &mut |_| {}).unwrap_err();
        assert!(err.contains("exceeds"), "expected a size-cap error, got: {err}");
    }
}
