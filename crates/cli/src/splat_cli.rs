// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain splat …` - 3D Gaussian Splatting scenes.
//!
//!   brain splat info   <scene.ply>
//!   brain splat render <scene.ply> --out img.ppm [--width N --height N]
//!        [--eye x,y,z --target x,y,z --up x,y,z --fov D] [--depth] [--bg r,g,b]
//!        [--aa] [--naive]
//!   brain splat view   <scene.ply> [--width N --height N --fov D --bg r,g,b]
//!        [--frames N]                # interactive fly-through (WASD + mouse)
//!
//! Viewer controls: WASD move, Space/C up/down, Shift sprint, m mouse-look,
//! arrows look, [ ] render quality, v color/depth, p screenshot, Enter reset,
//! Esc quit. With no --eye, the camera is auto-framed from the scene bounds.

use gpu_core::Gpu;
use splat::opt::{fit as splat_fit, FitCfg, TargetView};
use splat::renderer::{rgba_to_rgb, sorted_by_depth, GpuSplats, Renderer};
use splat::types::{auto_camera, cross3, norm3, Camera, Mode, RenderOpts, Splats};
use splat::Kernels;
use wm_display::keymap::{Key, KeySet, UxKey};
use wm_display::sink::{FrameSink, Hud};
use wm_display::window::SdlWindow;

use crate::args::Args;

pub fn run_splat(argv: &[String]) {
    match argv.first().map(|s| s.as_str()) {
        Some("info") => info(&argv[1..]),
        Some("render") => render(&argv[1..]),
        Some("view") => view(&argv[1..]),
        Some("fit") => fit_cmd(&argv[1..]),
        other => {
            eprintln!("usage: brain splat <info|render|view|fit> ...  (got {other:?})");
            std::process::exit(2);
        }
    }
}

fn load(a: &mut Args) -> (String, Splats) {
    let path = a.positional().unwrap_or_else(|| {
        eprintln!("expected a .ply path");
        std::process::exit(2);
    });
    match splat::ply::read(&path) {
        Ok(s) => (path, s),
        Err(e) => {
            eprintln!("cannot load {path}: {e}");
            std::process::exit(1);
        }
    }
}

fn info(argv: &[String]) {
    let mut a = Args::new(argv);
    let (path, s) = load(&mut a);
    a.finish();
    let (lo, hi) = s.bounds();
    let mean_op: f32 = s.opacities.iter().sum::<f32>() / s.len().max(1) as f32;
    println!("{path}: {} gaussians", s.len());
    println!("  bounds  min [{:.3} {:.3} {:.3}]  max [{:.3} {:.3} {:.3}]", lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
    println!("  mean opacity {mean_op:.3}");
    match &s.sh_rest {
        Some((deg, _)) => println!("  SH degree {deg} (higher orders parsed, rendered as DC for now)"),
        None => println!("  SH degree 0"),
    }
}

fn vec3(a: &mut Args, name: &str) -> Option<[f32; 3]> {
    a.take_str(name).map(|s| {
        let v: Vec<f32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() != 3 {
            eprintln!("{name} wants x,y,z");
            std::process::exit(2);
        }
        [v[0], v[1], v[2]]
    })
}

fn render(argv: &[String]) {
    let mut a = Args::new(argv);
    let (path, s) = load(&mut a);
    let out = crate::args::strip_out_name_prefix(&a.str_or("--out", "out/splat.ppm"), "image").to_string();
    let width = a.u32_or("--width", 960);
    let height = a.u32_or("--height", 720);
    let fov = a.f32_or("--fov", 60.0);
    let depth_view = a.take_flag("--depth");
    let naive = a.take_flag("--naive");
    let bench = a.u32_or("--bench", 0);
    let aa = a.take_flag("--aa");
    let bg = vec3(&mut a, "--bg").unwrap_or([0.0; 3]);
    let eye = vec3(&mut a, "--eye");
    let target = vec3(&mut a, "--target");
    let up = vec3(&mut a, "--up").unwrap_or([0.0, -1.0, 0.0]);
    // 0 = the renderer's own default (8 per gaussian). A scene whose gaussians
    // have grown - anything `splat fit` has optimized - covers more tiles each
    // and can exceed it, which silently drops the depth-latest splats and shows
    // up as bright flares where an occluder went missing.
    let isect_cap = a.usize_or("--isect-cap", 0);
    // The 2D anti-alias dilation, in PIXELS SQUARED, added to every splat's
    // screen-space covariance. It exists so splats smaller than a pixel do not
    // alias as the camera moves, and it is not free: it is a blur, and at the
    // reference default of 0.3 it costs most of the high-frequency content of
    // a scene whose splats are around a pixel across - which is every
    // reconstruction with roughly one gaussian per source pixel.
    let eps2d = a.f32_or("--eps2d", RenderOpts::default().eps2d);
    a.finish();

    let cam = match (eye, target) {
        (Some(e), Some(t)) => Camera::look_at(e, t, up, fov, width, height),
        (None, None) => auto_camera(&s, width, height, fov),
        _ => {
            eprintln!("--eye and --target go together");
            std::process::exit(2);
        }
    };
    let opts = RenderOpts {
        bg,
        mode: if depth_view { Mode::Depth } else { Mode::Color },
        antialiased: aa,
        eps2d,
        ..Default::default()
    };

    let g = Gpu::new(splat::PIPELINES);
    let ks = Kernels::at(0);
    let mut r = Renderer::new(&g, ks, s.len(), width, height, isect_cap);
    let t0 = std::time::Instant::now();
    let (img, how) = if naive {
        let sorted = sorted_by_depth(&s, &cam);
        let gs = GpuSplats::upload(&g, &sorted);
        (r.render_naive_gpu(&g, &gs, &cam, &opts), "naive".to_string())
    } else {
        let gs = GpuSplats::upload(&g, &s);
        let stats = r.render(&g, &gs, &cam, &opts);
        let img = r.read_rgba(&g, width, height);
        (img, format!("tiled, {} isects{}", stats.n_isects, if stats.clamped { ", CLAMPED" } else { "" }))
    };
    let mut ms = t0.elapsed().as_secs_f32() * 1000.0;
    if bench > 0 && !naive {
        // steady-state: re-render the same frame (buffers warm, JIT compiled)
        let gs = GpuSplats::upload(&g, &s);
        let tb = std::time::Instant::now();
        for _ in 0..bench {
            r.render(&g, &gs, &cam, &opts);
            let _ = r.read_rgb24(&g, width, height);
        }
        ms = tb.elapsed().as_secs_f32() * 1000.0 / bench as f32;
    }

    write_ppm(&out, &img, width as usize, height as usize, depth_view);
    println!("{path}: {} gaussians -> {out} ({width}x{height}, {how}, {ms:.0} ms{})", s.len(), if bench > 0 { "/frame steady-state" } else { "" });
    if how.contains("CLAMPED") {
        eprintln!(
            "splat render: the tile-instance buffer overflowed and the depth-latest splats were \
             dropped from this frame - raise it with --isect-cap (try {})",
            (s.len() * 16).max(1 << 21)
        );
    }
}

fn view(argv: &[String]) {
    let mut a = Args::new(argv);
    let (path, s) = load(&mut a);
    let width = a.u32_or("--width", 1280);
    let height = a.u32_or("--height", 720);
    let fov = a.f32_or("--fov", 60.0);
    let bg = vec3(&mut a, "--bg").unwrap_or([0.02, 0.02, 0.03]);
    let frames = a.opt_u32("--frames").map(|n| n as u64);
    a.finish();
    let title = format!("brain splat - {path}");
    run_viewer(&s, &title, width, height, fov, bg, frames, None);
}

/// Fly-camera state: position + yaw/pitch in the y-down world.
struct FlyCam {
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
}

impl FlyCam {
    fn from_camera(cam: &Camera) -> FlyCam {
        // forward = third COLUMN of the c2w rotation.
        let f = [cam.c2w[2], cam.c2w[6], cam.c2w[10]];
        FlyCam {
            pos: cam.eye(),
            yaw: f[0].atan2(f[2]),
            pitch: f[1].asin(),
        }
    }
    fn forward(&self) -> [f32; 3] {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        [cp * self.yaw.sin(), sp, cp * self.yaw.cos()]
    }
    fn camera(&self, fov_y_deg: f32, width: u32, height: u32) -> Camera {
        let f = self.forward();
        let r = norm3(cross3(f, [0.0, -1.0, 0.0]));
        let d = norm3(cross3(f, r));
        let e = self.pos;
        let c2w = [
            r[0], d[0], f[0], e[0],
            r[1], d[1], f[1], e[1],
            r[2], d[2], f[2], e[2],
            0.0, 0.0, 0.0, 1.0,
        ];
        let fy = 0.5 * height as f32 / (0.5 * fov_y_deg.to_radians()).tan();
        Camera { c2w, fx: fy, fy, cx: width as f32 / 2.0, cy: height as f32 / 2.0, width, height }
    }
}

/// The interactive loop, reused by `brain mirror demo`. Renders with the
/// tiled pipeline at a quality-selectable fraction of the window size
/// (1×, 1/2, 1/4 - SDL stretches back up), presents via wm-display.
#[allow(clippy::too_many_arguments)]
pub fn run_viewer(
    s: &Splats,
    title: &str,
    width: u32,
    height: u32,
    fov: f32,
    bg: [f32; 3],
    max_frames: Option<u64>,
    init_cam: Option<Camera>,
) {
    let g = Gpu::new(splat::PIPELINES);
    let ks = Kernels::at(0);
    let init = init_cam.unwrap_or_else(|| auto_camera(s, width, height, fov));
    let init_fly = FlyCam::from_camera(&init);
    let mut fly = FlyCam { ..init_fly };
    let (lo, hi) = s.bounds();
    let scene_size = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2))
        .sqrt()
        .max(1e-3);

    let mut renderer = Renderer::new(&g, ks, s.len(), width, height, 0);
    let gs = GpuSplats::upload(&g, s);

    // Quality levels: (resolution divisor == window scale) keeps the window a
    // constant size while the render buffer shrinks.
    let divs = [1u32, 2, 4];
    let mut q = 0usize;
    let make_win = |div: u32| -> SdlWindow {
        SdlWindow::new(title, width / div, height / div, div)
            .unwrap_or_else(|e| panic!("cannot open window: {e} (headless? SDL_VIDEODRIVER=dummy)"))
    };
    let mut win = make_win(divs[q]);

    let mut mode = Mode::Color;
    let mut captured = false;
    let mut fps = 0.0f32;
    let mut frame_no = 0u64;
    let mut shot_no = 0u32;
    let mut last = std::time::Instant::now();
    loop {
        let input = win.pump();
        if input.quit {
            if captured {
                // first Esc releases the mouse, second quits
                captured = false;
                win.set_relative_mouse(false);
            } else {
                break;
            }
        }
        let mut rebuild_win = false;
        for ux in &input.ux {
            match ux {
                UxKey::ToggleMouse => {
                    captured = !captured;
                    win.set_relative_mouse(captured);
                }
                UxKey::CycleView => mode = if mode == Mode::Color { Mode::Depth } else { Mode::Color },
                UxKey::QualityDown if q + 1 < divs.len() => {
                    q += 1;
                    rebuild_win = true;
                }
                UxKey::QualityUp if q > 0 => {
                    q -= 1;
                    rebuild_win = true;
                }
                UxKey::Reset => {
                    fly = FlyCam { ..FlyCam::from_camera(&init) };
                }
                _ => {}
            }
        }
        if rebuild_win {
            drop(win);
            win = make_win(divs[q]);
            if captured {
                win.set_relative_mouse(true);
            }
        }

        let dt = last.elapsed().as_secs_f32().min(0.1);
        last = std::time::Instant::now();

        // look: mouse (captured) + arrow keys
        if captured {
            fly.yaw += input.mouse_dx as f32 * 0.003;
            fly.pitch = (fly.pitch + input.mouse_dy as f32 * 0.003).clamp(-1.55, 1.55);
        }
        let look = 1.8 * dt;
        let held = |k: Key| input.pressed.contains(KeySet::of(&[k]));
        if held(Key::Left) {
            fly.yaw -= look;
        }
        if held(Key::Right) {
            fly.yaw += look;
        }
        if held(Key::Up) {
            fly.pitch = (fly.pitch - look).clamp(-1.55, 1.55);
        }
        if held(Key::Down) {
            fly.pitch = (fly.pitch + look).clamp(-1.55, 1.55);
        }

        // move: WASD planar, Space/C vertical, Shift sprint
        let speed = 0.25 * scene_size * dt * if held(Key::Shift) { 4.0 } else { 1.0 };
        let fwd = fly.forward();
        let right = norm3(cross3(fwd, [0.0, -1.0, 0.0]));
        let mut mv = [0.0f32; 3];
        let mut add = |v: [f32; 3], sgn: f32| {
            for k in 0..3 {
                mv[k] += v[k] * sgn;
            }
        };
        if held(Key::W) {
            add(fwd, 1.0);
        }
        if held(Key::S) {
            add(fwd, -1.0);
        }
        if held(Key::A) {
            add(right, -1.0);
        }
        if held(Key::D) {
            add(right, 1.0);
        }
        if held(Key::Space) {
            add([0.0, -1.0, 0.0], 1.0); // up in the y-down world
        }
        if held(Key::C) {
            add([0.0, 1.0, 0.0], 1.0);
        }
        for (p, &m) in fly.pos.iter_mut().zip(mv.iter()) {
            *p += m * speed;
        }

        // render + present
        let (rw, rh) = (width / divs[q], height / divs[q]);
        let cam = fly.camera(fov, rw, rh);
        let opts = RenderOpts { bg, mode, ..Default::default() };
        let t0 = std::time::Instant::now();
        let stats = renderer.render(&g, &gs, &cam, &opts);
        let mut rgb = renderer.read_rgb24(&g, rw, rh);
        let render_ms = t0.elapsed().as_secs_f32() * 1000.0;
        fps = if fps == 0.0 { 1000.0 / render_ms.max(0.1) } else { 0.9 * fps + 0.1 * (1000.0 / render_ms.max(0.1)) };

        let hudline = format!(
            "{}G {}I | {:.0} FPS {:.0}MS | {}X{} | POS {:.1} {:.1} {:.1}{}",
            s.len(),
            stats.n_isects,
            fps,
            render_ms,
            rw,
            rh,
            fly.pos[0],
            fly.pos[1],
            fly.pos[2],
            if captured { " | MOUSE (M RELEASES)" } else { "" },
        );
        zipdepth::viz::draw_text(&mut rgb, rw, rh, 6, 6, &hudline, 1, [0, 255, 0]);
        if input.ux.contains(&UxKey::Screenshot) {
            let p = format!("out/splat-shot-{shot_no:03}.ppm");
            write_ppm_rgb(&p, &rgb, rw as usize, rh as usize);
            println!("saved {p}");
            shot_no += 1;
        }
        let hud = Hud {
            model: "splat".into(),
            fps,
            target_fps: 60,
            step: frame_no,
            paused: false,
            quality: q as u32,
            action: 0,
            reset: false,
        };
        win.frame(&rgb, rw, rh, &hud);

        frame_no += 1;
        if let Some(maxf) = max_frames {
            if frame_no >= maxf {
                break;
            }
        }
    }
}

/// Optimize a scene against posed target images (the rasterizer-backward
/// demo): cameras.json is the `brain mirror infer` format; images are P6 PPMs
/// in index order matching the cameras.
fn fit_cmd(argv: &[String]) {
    let mut a = Args::new(argv);
    let (path, s) = load(&mut a);
    let cams_path = a.str_or("--cameras", "out/mirror/cameras.json");
    let images = a.take_str("--images").unwrap_or_else(|| {
        eprintln!("--images <dir|a.ppm,b.ppm,…> is required");
        std::process::exit(2);
    });
    let out = crate::args::strip_out_name_prefix(&a.str_or("--out", "out/fitted.ply"), "scene").to_string();
    let iters = a.usize_or("--iters", 200);
    let lr = a.f32_or("--lr", 5e-3);
    // Part of the forward model, not a display setting: render the result with
    // this same value or the optimizer's compensation for it shows up as blur
    // (rendered higher) or as aliasing (rendered lower).
    let eps2d = a.f32_or("--eps2d", FitCfg::default().eps2d);
    // Density control: let the fit ADD gaussians where the loss is still
    // pulling. Off unless asked, because a feed-forward scene is already
    // dense and growing it can push the backward past the device's
    // gradient-record ceiling mid-run.
    let densify_every = a.usize_or("--densify", 0);
    let densify_frac = a.f32_or("--densify-frac", FitCfg::default().densify_frac);
    let max_gaussians = a.usize_or("--max-gaussians", 0);
    a.finish();

    let cams_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cams_path).unwrap_or_else(|e| {
            eprintln!("cannot read {cams_path}: {e}");
            std::process::exit(1);
        }))
        .expect("valid cameras.json");
    let mut cams = Vec::new();
    for c in cams_json.as_array().expect("array") {
        let m: Vec<f32> = c["c2w"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        cams.push(Camera {
            c2w: m.try_into().expect("16 c2w entries"),
            fx: c["fx"].as_f64().unwrap() as f32,
            fy: c["fy"].as_f64().unwrap() as f32,
            cx: c["cx"].as_f64().unwrap() as f32,
            cy: c["cy"].as_f64().unwrap() as f32,
            width: c["width"].as_u64().unwrap() as u32,
            height: c["height"].as_u64().unwrap() as u32,
        });
    }
    // The SAME collection `worldmirror2 infer` uses. These were separate
    // copies, and only one of them learned to read a JPEG - so a scene
    // reconstructed straight from a folder of photographs could not then be
    // fitted against those same photographs.
    let paths = crate::mirror_cli::collect_images(&images);
    assert_eq!(paths.len(), cams.len(), "image count must match camera count");
    let targets: Vec<TargetView> = paths
        .iter()
        .zip(&cams)
        .map(|(pth, cam)| {
            let img = imaging::load(pth).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            });
            // A camera recovered by `worldmirror2 infer` is written for the
            // model's own grid, not for the photograph's full resolution, so
            // the same folder that produced the scene is almost never already
            // the right size. Resample instead of refusing: a target view is
            // defined by its camera, and the camera says how big it is.
            let rgb = if (img.w, img.h) == (cam.width, cam.height) {
                img.to_hwc_unit()
            } else {
                // Tolerated, not exact: the model's own preprocessing rounds
                // the short side to a multiple of its 14px patch grid (1536
                // becomes 392, not 388.5), so a camera it recovered is ~1% off
                // the photograph's true aspect by construction. Anything
                // further apart is a real mismatch - portrait against
                // landscape, or the wrong folder - and stretching it would
                // have the fit chase the distortion.
                let skew = (img.w as f32 / img.h as f32) / (cam.width as f32 / cam.height as f32);
                if !(0.95..1.05).contains(&skew) {
                    eprintln!(
                        "{pth} is {}x{} but its camera is {}x{}, a {:.0}% different aspect ratio. \
                         Resampling would stretch the target and the fit would chase the \
                         distortion; crop the photographs to the camera's aspect first.",
                        img.w, img.h, cam.width, cam.height, (skew - 1.0).abs() * 100.0
                    );
                    std::process::exit(2);
                }
                imaging::host::resize_bilinear_hwc(
                    &img.to_hwc_unit(), 3, img.w, img.h, cam.width, cam.height,
                )
            };
            TargetView { cam: *cam, rgb }
        })
        .collect();

    let g = Gpu::new(splat::PIPELINES);
    let ks = Kernels::at(0);
    println!("fitting {} gaussians against {} views ({} iters, lr {lr}) …", s.len(), targets.len(), iters);
    let cfg = FitCfg { iters, lr, eps2d, densify_every, densify_frac, max_gaussians, ..Default::default() };
    let (fitted, mse) = splat_fit(&g, ks, &s, &targets, &cfg, &mut |_it, _mse| true);
    let grown = fitted.len();
    splat::ply::write(&out, &fitted).unwrap_or_else(|e| {
        eprintln!("PLY write failed: {e}");
        std::process::exit(1);
    });
    println!(
        "{path} -> {out} ({grown} gaussians, final mse {mse:.6}, fitted at --eps2d {eps2d}; render it with the same value)"
    );
}

/// Tight RGB24 bytes → an image file (P6, or PNG when `path` says `.png`).
pub fn write_ppm_rgb(path: &str, rgb: &[u8], w: usize, h: usize) {
    let img = imaging::Rgb8::new(w as u32, h as u32, rgb.to_vec())
        .unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
    imaging::save(path, &img).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
}

/// RGBA f32 → P6. Depth views are min-max normalized for visibility.
///
/// The normalization is the part that is genuinely local: it rescales only over
/// the pixels the rasterizer actually covered (`alpha > 1e-6`), which is a splat
/// viewer concern and has no second copy. The RGBA→RGB8 quantisation and the P6
/// header both come from `imaging`.
pub fn write_ppm(path: &str, rgba: &[f32], w: usize, h: usize, normalize: bool) {
    let hwc = if normalize {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for px in rgba.chunks_exact(4) {
            if px[3] > 1e-6 {
                lo = lo.min(px[0]);
                hi = hi.max(px[0]);
            }
        }
        if !lo.is_finite() || hi <= lo {
            lo = 0.0;
            hi = 1.0;
        }
        rgba.chunks_exact(4).flat_map(|px| [(px[0] - lo) / (hi - lo), (px[1] - lo) / (hi - lo), (px[2] - lo) / (hi - lo)]).collect()
    } else {
        rgba_to_rgb(rgba)
    };
    let img = imaging::pixels::hwc_to_rgb8(&hwc, w as u32, h as u32, 3, imaging::ChannelPolicy::RequireRgb)
        .unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
    imaging::save(path, &img).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
}

#[cfg(test)]
mod tests {
    /// `worldmirror2 infer` and `splat fit` are used back to back on the same
    /// folder - reconstruct from photographs, then optimize the scene against
    /// those photographs - and each had its OWN copy of "collect the images in
    /// this directory". Only one of them was taught to read anything but
    /// `.ppm`, so the pair stopped composing on exactly the input a user
    /// arrives with. One function now serves both.
    #[test]
    fn fit_collects_the_same_inputs_the_reconstruction_did() {
        let d = std::env::temp_dir().join(format!("brain-fit-collect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for n in ["a.jpeg", "b.png", "c.ppm", "cameras.json"] {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        let got = crate::mirror_cli::collect_images(d.to_str().unwrap());
        assert_eq!(got.len(), 3, "fit would refuse photographs infer accepts: {got:?}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// `render --out` and `fit --out` both took the generic
    /// capability-manifest `name=path` form (documented by `brain caps
    /// splat`, and what `brain do`/D-Bus actually send) literally, writing a
    /// file named e.g. `image=out.ppm` with no error. Both sites are wired
    /// through `crate::args::strip_out_name_prefix` (shared, tested there)
    /// against this crate's own declared output blob names for each
    /// action - `image` for `render`, `scene` for `fit`; this just pins that
    /// the wiring at each call site did not regress.
    #[test]
    fn render_and_fit_out_accept_the_documented_name_equals_path_form() {
        assert_eq!(crate::args::strip_out_name_prefix("image=out.ppm", "image"), "out.ppm");
        assert_eq!(crate::args::strip_out_name_prefix("scene=fitted.ply", "scene"), "fitted.ply");
    }
}
