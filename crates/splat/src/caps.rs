// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 3D Gaussian Splatting behind the generalized [`capability`] interface -
//! what makes `brain caps splat` / `brain do splat render …`, the D-Bus `Run`
//! method and the event API work with no splat-specific plumbing in the CLI
//! or the transports.
//!
//! The actions, every knob a param with its own default and bounds:
//!
//! * **`render`** - ONE-SHOT: a scene (Inria PLY bytes) plus a camera in,
//!   one image out. The renderer is sized for a fixed `(gaussian count,
//!   width, height, instance cap)` at construction (`Renderer::new`), so
//!   [`RenderSession`] caches the built renderer keyed on a cheap digest of
//!   those and the scene bytes - the same shape `vqgan::caps::VqAction`
//!   caches its built graph on `(weights, size)`, adapted to a model whose
//!   "weights" are per-request bytes rather than a checkpoint path.
//! * **`fit`** ([`crate::caps_fit`]) - `.streaming()`: an initial scene plus N
//!   posed target views in, the optimized scene out, reporting
//!   [`Progress::step`] and honouring [`Invocation::cancel`].
//! * **`info`**, **`orient`**, **`merge`**, **`prune`**
//!   ([`crate::caps_tools`]) - inspecting, re-framing, chaining and thinning
//!   scenes, over [`crate::tools`].
//!
//! The interactive fly-through is an application, not a capability (it has no
//! request/response shape), so it is not served; `samples/models/splat`
//! carries it.
//!
//! Swedish Embedded AB implements 3D scene capture and rendering services for
//! its clients. If your team needs expertise in Gaussian splatting and
//! real-time neural rendering then you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::sync::Arc;

use capability::bounds::TRAIN_COUNT;
use capability::{hot::Hot, Action, ActionResult, ActionSpec, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use gpu_core::Gpu;
use serde_json::json;

use crate::renderer::{rgba_to_rgb, sorted_by_depth, GpuSplats, Renderer};
use crate::types::{auto_camera_from_bounds, cameras_from_json, Camera, Mode, RenderOpts, Splats};
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

/// Ceiling on a gaussian count a caller sets (a budget, an instance cap):
/// 2^28, about 4 GB of positions alone, past any scene a device holds.
pub(crate) const MAX_GAUSSIANS: f64 = 268_435_456.0;
/// Ceiling on a ratio or a budget in radii: a bound looser than this is
/// "unbounded", which every such param spells as 0.
pub(crate) const MAX_RATIO: f64 = 1_000.0;
/// Ceiling on a loss weight or a filter size in pixels squared.
pub(crate) const MAX_WEIGHT: f64 = 1_000.0;

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

pub(crate) fn vec3_param(name: &str, help: &str) -> ParamSpec {
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
        .param(ParamSpec::new("view", ParamType::Int, "with the 'cameras' input: which of its cameras to render, to put next to its photograph; the camera brings its own size and lens, so 'width', 'height', 'fov', 'eye' and 'target' are not used").default(json!(0)).min(0.0).max(TRAIN_COUNT))
        .param(ParamSpec::new("depth", ParamType::Bool, "render alpha-weighted expected depth (replicated to RGB) instead of color").default(json!(false)))
        .param(ParamSpec::new("normalize_depth", ParamType::Bool, "with 'depth': rescale the covered pixels to [0,1] so the image is viewable (the raw range comes back as 'depth_min' and 'depth_max')").default(json!(true)))
        .param(ParamSpec::new("antialiased", ParamType::Bool, "put back the energy the low-pass spreads out, by scaling opacity (Mip-Splatting's 2D Mip filter); Inria-trained PLYs were fitted without it and expect false at eps2d 0.3").default(json!(RenderOpts::default().antialiased)))
        .param(ParamSpec::new("inria_lowpass", ParamType::Bool, "render for an Inria-convention scene: the uncompensated dilation, whatever 'antialiased' says. The low-pass a scene was fitted under is part of it; rendering under a different one un-does the compensation the optimizer folded in").default(json!(false)))
        .param(ParamSpec::new("eps2d", ParamType::Float, "the 2D anti-alias dilation, in pixels squared, added to every splat's screen-space covariance so splats smaller than a pixel do not alias as the camera moves; it is a blur, and the reference 0.3 costs most of the high-frequency content of a scene whose splats are about a pixel across").default(json!(RenderOpts::default().eps2d)).min(0.0).max(MAX_WEIGHT))
        .param(vec3_param("bg", "background color 'r,g,b' in [0,1]").default(json!("0,0,0")))
        .param(ParamSpec::new("naive", ParamType::Bool, "composite every gaussian in depth order instead of through the tiled pipeline - the reference the tiled path is checked against; slow, and without view-dependent colour").default(json!(false)))
        .param(ParamSpec::new("isect_cap", ParamType::Int, "capacity of the tile-instance buffer; 0 = the renderer's own (8 per gaussian). A scene whose gaussians have grown (anything a fit has optimized) covers more tiles each and can exceed it, which silently drops the depth-latest splats and shows up as bright flares where an occluder went missing - the 'clamped' output says when it did").default(json!(0)).min(0.0).max(MAX_GAUSSIANS))
        .param(ParamSpec::new("bench", ParamType::Int, "after the first frame, re-render the same frame this many times and report the steady-state time per frame as 'ms'; 0 = off").default(json!(0)).min(0.0).max(10_000.0))
        .input(BlobSpec::new("scene", Media::Bytes, "the scene: Inria-layout binary PLY").required())
        .input(BlobSpec::new("cameras", Media::Text, "cameras as JSON, in the shape the 'fit' action's 'views' takes; 'view' selects one"))
        .output(BlobSpec::new("image", Media::Image, "the rendered image, RGB f32 (with 'depth' and without 'normalize_depth', the raw expected-depth values)"))
}

/// The full, static capability manifest - safe to build with no scene loaded.
/// `view` (the interactive viewer) is deliberately absent - see the module doc.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "3D Gaussian Splatting: render a posed view of a scene, fit a scene to posed target images, and inspect, re-frame, chain and thin scenes.",
        vec![render_spec(), crate::caps_fit::fit_spec(), crate::caps_tools::info_spec(), crate::caps_tools::orient_spec(), crate::caps_tools::merge_spec(), crate::caps_tools::prune_spec()],
    )
}

// ===================== render: a cached, scene-sized session =====================

/// A built renderer for one `(scene, width, height, isect_cap)` - the GPU
/// buffers `Renderer::new` allocates are sized for exactly this gaussian
/// count, resolution and instance capacity, so a session is invalidated (and
/// rebuilt) whenever any of them changes.
struct RenderSession {
    gpu: Gpu,
    renderer: Renderer,
    gs: GpuSplats,
    /// The host scene: the naive path re-sorts it per camera and the tiled
    /// path shades its view-dependent colour per camera.
    scene: Splats,
}

/// A cheap FNV-1a over the scene bytes plus the values `Renderer::new` bakes
/// into its GPU buffer sizes - the cache key. Two requests with the same key
/// share one built renderer.
fn scene_key(bytes: &[u8], width: u32, height: u32, isect_cap: usize) -> u64 {
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
    mix(isect_cap as u64);
    h
}

fn build_render_session(scene: Splats, width: u32, height: u32, isect_cap: usize) -> RenderSession {
    let gpu = Gpu::new(PIPELINES);
    let ks = Kernels::at(0);
    let renderer = Renderer::new(&gpu, ks, scene.len(), width, height, isect_cap);
    let gs = GpuSplats::upload(&gpu, &scene);
    RenderSession { gpu, renderer, gs, scene }
}

/// Parse a `"x,y,z"` CSV param into `[f32; 3]`. `None` when the param is
/// absent/empty (the caller decides what that means - an optional override
/// vs. a defaulted vector).
pub(crate) fn get_vec3(inv: &Invocation, name: &str) -> Result<Option<[f32; 3]>, String> {
    let Some(s) = inv.get_str(name).filter(|s| !s.trim().is_empty()) else { return Ok(None) };
    let v: Result<Vec<f32>, String> = s.split(',').map(|t| t.trim().parse::<f32>().map_err(|_| format!("splat: '{name}' must be 'x,y,z' floats, got '{s}'"))).collect();
    let v = v?;
    if v.len() != 3 {
        return Err(format!("splat: '{name}' wants 'x,y,z', got {} number(s)", v.len()));
    }
    Ok(Some([v[0], v[1], v[2]]))
}

pub(crate) fn get_vec3_or(inv: &Invocation, name: &str, default: [f32; 3]) -> Result<[f32; 3], String> {
    Ok(get_vec3(inv, name)?.unwrap_or(default))
}

/// The camera `render` uses when the request brings none of its own: an
/// explicit `eye`+`target` pose, or (with neither set) an auto-framed one
/// over the scene's own bounds. `eye` xor `target` alone is rejected - a
/// half-specified pose is a caller error, not something to silently complete.
fn camera_from_params(inv: &Invocation, bounds: ([f32; 3], [f32; 3]), width: u32, height: u32, fov: f32) -> Result<Camera, String> {
    let up = get_vec3_or(inv, "up", [0.0, -1.0, 0.0])?;
    match (get_vec3(inv, "eye")?, get_vec3(inv, "target")?) {
        (Some(eye), Some(target)) => Ok(Camera::look_at(eye, target, up, fov, width, height)),
        (None, None) => Ok(auto_camera_from_bounds(bounds, width, height, fov)),
        _ => Err("splat render: 'eye' and 'target' must be given together".to_string()),
    }
}

/// The camera the `cameras` input's `view`th entry describes.
fn camera_from_input(inv: &Invocation) -> Result<Option<Camera>, String> {
    let Some(blob) = inv.get_blob("cameras") else { return Ok(None) };
    let text = std::str::from_utf8(&blob.bytes).map_err(|_| "splat render: 'cameras' is not UTF-8 JSON".to_string())?;
    let cams = cameras_from_json(text).map_err(|e| format!("splat render: 'cameras': {e}"))?;
    let view = inv.get_i64("view").unwrap_or(0).max(0) as usize;
    cams.get(view).copied().map(Some).ok_or_else(|| format!("splat render: view {view}, but 'cameras' holds {} cameras", cams.len()))
}

/// A depth image's covered pixels (alpha above zero) rescaled to `[0,1]` in
/// all three channels, with the `(min, max)` range they had; an empty or flat
/// image maps through `[0,1]` unchanged.
fn normalize_depth(rgba: &[f32]) -> (Vec<f32>, f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for px in rgba.as_chunks::<4>().0.iter().filter(|px| px[3] > 1e-6) {
        lo = lo.min(px[0]);
        hi = hi.max(px[0]);
    }
    if !lo.is_finite() || hi <= lo {
        (lo, hi) = (0.0, 1.0);
    }
    let unit = |v: f32| ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
    (rgba.as_chunks::<4>().0.iter().flat_map(|px| [unit(px[0]), unit(px[1]), unit(px[2])]).collect(), lo, hi)
}

/// Run one `render` invocation (already validated against [`render_spec`]),
/// reusing `hot` when `(scene, width, height, isect_cap)` matches the last
/// call.
fn render(inv: &Invocation, hot: &Hot<u64, RenderSession>) -> ActionResult {
    let file_camera = camera_from_input(inv)?;
    // a camera from the 'cameras' input brings its own size
    let (width, height) = match &file_camera {
        Some(cam) => (cam.width, cam.height),
        None => (inv.get_i64("width").unwrap_or(960).max(0) as u32, inv.get_i64("height").unwrap_or(720).max(0) as u32),
    };
    check_size(width, height)?;
    let fov = inv.get_f64("fov").unwrap_or(60.0) as f32;
    let isect_cap = inv.get_i64("isect_cap").unwrap_or(0).max(0) as usize;
    let scene_blob = inv.get_blob("scene").ok_or("splat render: missing required input 'scene'")?;
    let key = scene_key(&scene_blob.bytes, width, height, isect_cap);

    hot.try_get_mut(
        key,
        || Ok(build_render_session(crate::ply::parse(&scene_blob.bytes)?, width, height, isect_cap)),
        |sess| {
            let cam = match file_camera {
                Some(cam) => cam,
                None => camera_from_params(inv, sess.scene.bounds(), width, height, fov)?,
            };
            let depth = inv.get_bool("depth").unwrap_or(false);
            let inria = inv.get_bool("inria_lowpass").unwrap_or(false);
            let opts = RenderOpts {
                bg: get_vec3_or(inv, "bg", [0.0; 3])?,
                mode: if depth { Mode::Depth } else { Mode::Color },
                antialiased: inv.get_bool("antialiased").unwrap_or(RenderOpts::default().antialiased) && !inria,
                eps2d: inv.get_f64("eps2d").map_or(RenderOpts::default().eps2d, |v| v as f32),
                ..Default::default()
            };
            let started = std::time::Instant::now();
            let mut out = Outcome::new().set("width", json!(width)).set("height", json!(height)).set("gaussians", json!(sess.scene.len()));
            let rgba = if inv.get_bool("naive").unwrap_or(false) {
                let sorted = sorted_by_depth(&sess.scene, &cam);
                let gs = GpuSplats::upload(&sess.gpu, &sorted);
                out = out.set("how", json!("naive"));
                sess.renderer.render_naive_gpu(&sess.gpu, &gs, &cam, &opts)
            } else {
                // View-dependent colour, shaded for this camera: a fitted
                // scene's harmonics are part of how it looks.
                if let Some(c) = crate::sh::shade(&sess.scene, cam.eye()) {
                    sess.gpu.write_f32(&sess.gs.colors, &c);
                }
                let stats = sess.renderer.render(&sess.gpu, &sess.gs, &cam, &opts);
                out = out.set("how", json!("tiled")).set("isects", json!(stats.n_isects)).set("clamped", json!(stats.clamped));
                if stats.clamped {
                    out = out.set("warning", json!(format!("the tile-instance buffer overflowed and the depth-latest splats were dropped from this frame - raise 'isect_cap' (try {})", (sess.scene.len() * 16).max(1 << 21))));
                }
                sess.renderer.read_rgba(&sess.gpu, width, height)
            };
            let mut ms = started.elapsed().as_secs_f32() * 1000.0;
            let bench = inv.get_i64("bench").unwrap_or(0).max(0) as u32;
            if bench > 0 && !inv.get_bool("naive").unwrap_or(false) {
                // steady state: the same frame again, buffers warm
                let timed = std::time::Instant::now();
                for _ in 0..bench {
                    sess.renderer.render(&sess.gpu, &sess.gs, &cam, &opts);
                    let _ = sess.renderer.read_rgb24(&sess.gpu, width, height);
                }
                ms = timed.elapsed().as_secs_f32() * 1000.0 / bench as f32;
            }
            let rgb = if depth && inv.get_bool("normalize_depth").unwrap_or(true) {
                let (rgb, lo, hi) = normalize_depth(&rgba);
                out = out.set("depth_min", json!(lo)).set("depth_max", json!(hi));
                rgb
            } else {
                rgba_to_rgb(&rgba)
            };
            Ok(out.set("ms", json!(ms)).blob("image", capability::blob::image_blob(&rgb, width, height, 3)))
        },
    )
}

// ===================== the provider =====================

/// The provider over a resolved [`capability::Assembly`]: splat reads no
/// weights, so the assembly is not consulted - every request supplies its own
/// scene.
pub fn provider(_assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(SplatProvider::new()))
}

/// The executable splat model behind the manifest. Construction is free: no
/// weights to load - every request supplies its own scene.
pub struct SplatProvider {
    /// `render`'s built-renderer cache - see [`RenderSession`].
    hot: Arc<Hot<u64, RenderSession>>,
}

impl SplatProvider {
    pub fn new() -> SplatProvider {
        SplatProvider { hot: Arc::new(Hot::new()) }
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
        let tool = |spec: fn() -> ActionSpec, run: ToolRun| Some(Arc::new(ToolAction { spec, run }) as Arc<dyn Action>);
        match name {
            "render" => Some(Arc::new(RenderAction { hot: self.hot.clone() }) as Arc<dyn Action>),
            "fit" => tool(crate::caps_fit::fit_spec, crate::caps_fit::fit),
            "info" => tool(crate::caps_tools::info_spec, |inv, _| crate::caps_tools::info(inv)),
            "orient" => tool(crate::caps_tools::orient_spec, |inv, _| crate::caps_tools::orient(inv)),
            "merge" => tool(crate::caps_tools::merge_spec, |inv, _| crate::caps_tools::merge(inv)),
            "prune" => tool(crate::caps_tools::prune_spec, |inv, _| crate::caps_tools::prune(inv)),
            _ => None,
        }
    }
}

struct RenderAction {
    hot: Arc<Hot<u64, RenderSession>>,
}

impl Action for RenderAction {
    fn spec(&self) -> ActionSpec {
        render_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        render(inv, &self.hot)
    }
}

/// How a stateless action runs: no cache to share, just the invocation.
type ToolRun = fn(&Invocation, &mut dyn FnMut(Progress)) -> ActionResult;

/// An action with no state of its own - `fit` and the scene tools.
struct ToolAction {
    spec: fn() -> ActionSpec,
    run: ToolRun,
}

impl Action for ToolAction {
    fn spec(&self) -> ActionSpec {
        (self.spec)()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        (self.run)(inv, progress)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use crate::opt::{fit as opt_fit, FitCfg, TargetView};
    use capability::{blob::video_blob, Blob, CancelToken};
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
                TargetView::new(*c, rgba_to_rgb(&img))
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
        // the action's own defaults: fitted for the Mip filter unless `inria_lowpass`
        let cfg = FitCfg { iters: 40, lr_position: 5e-3, log_every: 0, antialiased: true, ..Default::default() };
        (gpu, ks, init, targets, cfg)
    }

    fn scene_mse(gpu: &Gpu, ks: Kernels, s: &Splats, targets: &[TargetView]) -> f64 {
        let (mw, mh) = targets.iter().fold((0u32, 0u32), |(mw, mh), t| (mw.max(t.cam.width), mh.max(t.cam.height)));
        let mut ren = Renderer::new(gpu, ks, s.len(), mw, mh, 0);
        let gs = GpuSplats::upload(gpu, s);
        // the fit's own forward model
        let opts = RenderOpts { ray: true, ..Default::default() };
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
            .set("lr_position", json!(cfg.lr_position as f64))
            .blob("scene", Blob::new(Media::Bytes, scene_bytes))
            .blob("video", video)
    }

    #[test]
    fn manifest_declares_every_scene_action_but_not_view() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["render", "fit", "info", "orient", "merge", "prune"]);
        assert!(m.actions.iter().any(|a| a.name == "render" && !a.streaming));
        let f = m.actions.iter().find(|a| a.name == "fit").expect("fit");
        assert!(f.streaming);
        assert!(!m.actions.iter().any(|a| a.name == "view"), "the interactive viewer must never be advertised as a servable action");
        let p = provider(&capability::Assembly { id: String::new(), arch: String::new(), variant: None, roles: Default::default(), provenance: Vec::new() }).unwrap();
        for a in &m.actions {
            assert!(p.action(&a.name).is_some(), "{} is declared but not provided", a.name);
        }
    }

    /// Every number a caller can set is bounded (but the one rate the
    /// unbounded-params list still carries), and every param that names a
    /// place on the serving machine says so.
    #[test]
    fn every_number_is_bounded_and_every_host_path_is_declared() {
        for a in manifest().actions {
            for p in &a.params {
                if matches!(p.ty, ParamType::Int | ParamType::Float) && !(a.name == "fit" && p.name == "lr_position") {
                    assert!(p.max.is_some(), "{}:{} has no max", a.name, p.name);
                }
            }
        }
        let fit = fit_spec_for_test();
        for name in ["cameras", "images"] {
            assert!(fit.params.iter().find(|p| p.name == name).unwrap().host_resolved, "{name}");
        }
    }

    fn fit_spec_for_test() -> ActionSpec {
        crate::caps_fit::fit_spec()
    }

    /// A render of a camera from the `cameras` input takes that camera's own
    /// size, and a view past the end is refused.
    #[test]
    fn render_takes_its_camera_and_size_from_the_cameras_input() {
        let s = scene(4, 7);
        let cam = Camera::look_at([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -1.0, 0.0], 60.0, 20, 12);
        let cams = serde_json::to_string(&vec![camera_json(&cam)]).unwrap();
        let provider = SplatProvider::new();
        let act = provider.action("render").unwrap();
        let request = |view: i64| {
            Invocation::new()
                .set("view", json!(view))
                .blob("scene", Blob::new(Media::Bytes, crate::ply::serialize(&s).unwrap()))
                .blob("cameras", Blob::new(Media::Text, cams.clone().into_bytes()))
        };
        let out = act.run(&act.spec().validate(request(0)).unwrap(), &mut |_| {}).unwrap();
        assert_eq!((out.outputs["width"].clone(), out.outputs["height"].clone()), (json!(20), json!(12)));
        let e = act.run(&act.spec().validate(request(3)).unwrap(), &mut |_| {}).unwrap_err();
        assert!(e.contains("holds 1 cameras"), "{e}");
    }

    /// The naive composite and the tiled pipeline are two evaluations of one
    /// image: they agree to within float noise on a small scene, and the
    /// depth view comes back normalized with its raw range reported.
    #[test]
    fn naive_and_tiled_agree_and_depth_is_normalized() {
        let bytes = crate::ply::serialize(&scene(6, 0xBEEF)).unwrap();
        let provider = SplatProvider::new();
        let act = provider.action("render").unwrap();
        let render = |extra: &[(&str, serde_json::Value)]| {
            let mut inv = Invocation::new().set("width", json!(24)).set("height", json!(24)).blob("scene", Blob::new(Media::Bytes, bytes.clone()));
            for (k, v) in extra {
                inv = inv.set(k, v.clone());
            }
            act.run(&act.spec().validate(inv).unwrap(), &mut |_| {}).unwrap()
        };
        let pixels = |o: &capability::Outcome| -> Vec<f32> { o.blobs["image"].bytes.as_chunks::<4>().0.iter().map(|q| f32::from_le_bytes(*q)).collect() };
        let (naive, tiled) = (render(&[("naive", json!(true))]), render(&[]));
        assert_eq!(naive.outputs["how"], json!("naive"));
        let worst = pixels(&naive).iter().zip(pixels(&tiled)).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-2, "naive and tiled differ by {worst}");
        let depth = render(&[("depth", json!(true))]);
        assert!(pixels(&depth).iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(depth.outputs["depth_max"].as_f64().unwrap() > depth.outputs["depth_min"].as_f64().unwrap());
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
        let caps_rgb: Vec<f32> = img.bytes.as_chunks::<4>().0.iter().map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();

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
        let request = || {
            Invocation::new()
                .set("width", json!(1_000_000))
                .set("height", json!(1_000_000))
                .blob("scene", Blob::new(Media::Bytes, crate::ply::serialize(&s).unwrap()))
        };
        // The spec's own maximum refuses it first...
        let err = act.spec().validate(request()).unwrap_err();
        assert!(err.contains("at most"), "expected the spec's maximum to refuse it, got: {err}");
        // ...and the action refuses it again for a caller that skips
        // validation, rather than failing inside GPU buffer allocation.
        let err = act.run(&request(), &mut |_| {}).unwrap_err();
        assert!(err.contains("exceeds"), "expected a size-cap error, got: {err}");
    }
}
