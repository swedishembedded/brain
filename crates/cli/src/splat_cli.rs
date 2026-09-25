// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain splat …` - 3D Gaussian Splatting scenes.
//!
//!   brain splat info   <scene.ply>
//!   brain splat render <scene.ply> --out img.ppm [--width N --height N]
//!        [--eye x,y,z --target x,y,z --up x,y,z --fov D] [--depth] [--bg r,g,b]
//!        [--aa | --inria-lowpass] [--naive]
//!   brain splat fit    <scene.ply> --cameras <cams.json> --images <dir> --out F
//!        [--iters N --lr R] [--inria-lowpass]  # fit for an Inria-convention
//!        viewer (uncompensated dilation at 0.3) instead of the Mip filter
//!        [--position-budget R --scale-budget R --rotation-budget R]
//!        # how far geometry may move, in units of a gaussian's own radius;
//!        # set these when the scene is already metric, 0 (default) = unbounded
//!   brain splat merge  <a.ply,b.ply,...> --cameras <a.json,b.json,...>
//!        --overlap K --out merged.ply [--cameras-out J] [--prune VOXEL]
//!        # K = how many trailing frames of each chunk lead the next one
//!   brain splat view   <scene.ply> [--width N --height N --fov D --bg r,g,b]
//!        [--frames N]                # interactive fly-through (WASD + mouse)
//!
//! Viewer controls: WASD move, Space/C up/down, Shift sprint, m mouse-look,
//! arrows look, [ ] render quality, v color/depth, p screenshot, Enter reset,
//! Esc quit. With no --eye, the camera is auto-framed from the scene bounds.

use gpu_core::Gpu;
use splat::opt::{Densify, FitCfg, TargetView};
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
        Some("orient") => orient_cmd(&argv[1..]),
        Some("merge") => merge_cmd(&argv[1..]),
        Some("prune") => prune_cmd(&argv[1..]),
        Some("sfm") => sfm_cmd(&argv[1..]),
        Some("train") => train_cmd(&argv[1..]),
        other => {
            eprintln!("usage: brain splat <info|render|view|fit|orient|merge|prune|sfm|train> ...  (got {other:?})");
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
        Some((deg, _)) => println!("  SH degree {deg} (view-dependent colour, shaded by render and view)"),
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
    // The low-pass a scene was FITTED under is part of it; rendering under a
    // different one un-does the compensation the optimizer folded in.
    let inria_render = a.take_flag("--inria-lowpass");
    let aa = a.take_flag("--aa") || (!inria_render && RenderOpts::default().antialiased);
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
    let eps2d =
        a.f32_or("--eps2d", if inria_render { 0.3 } else { RenderOpts::default().eps2d });
    // Render exactly one of the cameras a scene was trained or fitted with,
    // to put next to its photograph.
    let cams_file = a.take_str("--cameras");
    let view = a.usize_or("--view", 0);
    a.finish();

    let cam = match (eye, target, cams_file) {
        (_, _, Some(f)) => {
            let cams = read_cameras(&f);
            *cams.get(view).unwrap_or_else(|| {
                eprintln!("--view {view}: {f} holds {} cameras", cams.len());
                std::process::exit(2);
            })
        }
        (Some(e), Some(t), None) => Camera::look_at(e, t, up, fov, width, height),
        (None, None, None) => auto_camera(&s, width, height, fov),
        _ => {
            eprintln!("--eye and --target go together");
            std::process::exit(2);
        }
    };
    // a camera from a file brings its own size
    let (width, height) = (cam.width, cam.height);
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
        // View-dependent colour, shaded for this camera: a fitted scene's
        // harmonics are part of how it looks.
        let eye = [cam.c2w[3], cam.c2w[7], cam.c2w[11]];
        let gs = GpuSplats::upload(&g, &s);
        if let Some(c) = splat::sh::shade(&s, eye) {
            g.write_f32(&gs.colors, &c);
        }
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
        if let Some(c) = splat::sh::shade(s, [cam.c2w[3], cam.c2w[7], cam.c2w[11]]) {
            g.write_f32(&gs.colors, &c);
        }
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
/// Rigidly re-frame a scene so it opens the right way up.
///
/// A feed-forward reconstruction's world frame is the FIRST camera's frame -
/// `c2w[0]` comes back as the identity - so "up" in the file is whichever way
/// the camera happened to be held, and every viewer shows the scene tipped by
/// that angle. Measured on a real capture: 63 degrees.
///
/// The recovered cameras say which way is really up. They orbit the subject,
/// so the normal of the plane they lie in IS the scene's vertical, and their
/// rays intersect at its centre. Both come out of the cameras alone, with no
/// assumption about the subject.
///
/// Rigid, so it changes nothing measurable: means rotate, quaternions compose,
/// scales and opacities are untouched, and `cameras.json` is rewritten by the
/// same transform so a later `fit` still lines up.
fn orient_cmd(argv: &[String]) {
    let mut a = Args::new(argv);
    let (path, s) = load(&mut a);
    let cams_path = a.str_or("--cameras", "out/mirror/cameras.json");
    let out = crate::args::strip_out_name_prefix(&a.str_or("--out", "out/oriented.ply"), "scene").to_string();
    let cams_out = a.str_or("--cameras-out", &format!("{out}.cameras.json"));
    // brain's world is y-down, so "up" is -Y; a viewer expecting y-up gets
    // there with one flip rather than a different scene on disk.
    let up_sign = if a.take_flag("--y-up") { 1.0f64 } else { -1.0 };
    a.finish();

    let text = std::fs::read_to_string(&cams_path).unwrap_or_else(|e| {
        eprintln!("cannot read {cams_path}: {e}");
        std::process::exit(1);
    });
    let mut j: serde_json::Value = serde_json::from_str(&text).expect("valid cameras.json");
    let arr = j.as_array_mut().expect("array of cameras");
    let mats: Vec<[f64; 16]> = arr
        .iter()
        .map(|c| {
            let v: Vec<f64> = c["c2w"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            v.try_into().unwrap()
        })
        .collect();
    if mats.len() < 3 {
        eprintln!("orient: need at least 3 cameras to find the orbit plane, got {}", mats.len());
        std::process::exit(2);
    }
    let (rot, centre) = splat::orient::frame_from_cameras(&mats, up_sign);
    let oriented = splat::orient::apply(&s, &rot, &centre);
    splat::ply::write(&out, &oriented).unwrap_or_else(|e| {
        eprintln!("PLY write failed: {e}");
        std::process::exit(1);
    });
    for (c, m) in arr.iter_mut().zip(&mats) {
        let t = splat::orient::transform_c2w(m, &rot, &centre);
        c["c2w"] = serde_json::json!(t.to_vec());
    }
    std::fs::write(&cams_out, serde_json::to_string_pretty(&j).unwrap()).unwrap_or_else(|e| {
        eprintln!("cannot write {cams_out}: {e}");
        std::process::exit(1);
    });
    println!("{path} -> {out} ({} gaussians, re-framed; cameras -> {cams_out})", oriented.len());
}

/// Fuse duplicate gaussians and drop the faint ones.
///
/// A pixel-aligned reconstruction emits one gaussian per source pixel per
/// view, so a surface seen by twelve cameras is represented twelve times over.
/// That is not detail, it is the same detail counted repeatedly, and it buys
/// the optimizer enough freedom to memorise its training views.
fn prune_cmd(argv: &[String]) {
    let mut a = Args::new(argv);
    let (path, s) = load(&mut a);
    let mut voxel = a.f32_or("--voxel", 0.0);
    // A voxel in world units means nothing without knowing how big the scene
    // is. Two gaussians closer together than the finest detail any camera
    // resolved cannot be told apart by any training view, so THAT is the
    // distance to fuse at, and it is stated in pixels.
    let voxel_px = a.f32_or("--voxel-pixels", if voxel > 0.0 { 0.0 } else { 2.0 });
    let cams_path = a.take_str("--cameras");
    let min_op = a.f32_or("--min-opacity", 0.0);
    // A reconstruction predicts opacity with no anti-alias filter in mind, so
    // rendering it through one that compensates energy dims the whole scene.
    let a_recal = a.take_flag("--recalibrate-opacity");
    let max_points = a.usize_or("--max-gaussians", 0);
    // Multiply every opacity, for asking what a scene would look like if its
    // gaussians were as confident as they should be.
    let gain = a.f32_or("--opacity-gain", 1.0);
    // A contiguous slice, for asking what ONE frame's gaussians look like when
    // nothing else in the scene is there to cover for them.
    let range = a.take_str("--range");
    let out = crate::args::strip_out_name_prefix(&a.str_or("--out", "out/pruned.ply"), "scene").to_string();
    a.finish();

    let (lo, hi) = match &range {
        Some(r) => {
            let (a0, b0) = r.split_once(':').expect("--range START:COUNT");
            let st: usize = a0.parse().expect("range start");
            let ct: usize = b0.parse().expect("range count");
            (st.min(s.len()), (st + ct).min(s.len()))
        }
        None => (0, s.len()),
    };
    let kept: Vec<usize> = (lo..hi).filter(|&i| s.opacities[i] >= min_op).collect();
    let mut t = Splats::default();
    for i in kept {
        t.means.extend_from_slice(&s.means[i * 3..i * 3 + 3]);
        t.quats.extend_from_slice(&s.quats[i * 4..i * 4 + 4]);
        t.scales.extend_from_slice(&s.scales[i * 3..i * 3 + 3]);
        t.opacities.push(s.opacities[i]);
        t.colors.extend_from_slice(&s.colors[i * 3..i * 3 + 3]);
    }
    if let (true, Some(cp)) = (a_recal, cams_path.as_ref()) {
        t = splat::mip::recalibrate_opacity(&t, &read_cameras(cp), splat::types::RenderOpts::default().eps2d);
    }
    if gain != 1.0 {
        for v in t.opacities.iter_mut() {
            *v = (*v * gain).clamp(1e-4, 1.0 - 1e-4);
        }
    }
    let after_op = t.len();
    if voxel_px > 0.0 {
        let cams = read_cameras(&cams_path.clone().unwrap_or_else(|| {
            eprintln!("prune: --voxel-pixels needs --cameras (it is the cameras that set the scale)");
            std::process::exit(2);
        }));
        let mut sig: Vec<f32> =
            splat::mip::smoothing_sigma(&t, &cams, voxel_px).into_iter().filter(|&v| v > 0.0).collect();
        if sig.is_empty() {
            eprintln!("prune: no gaussian is visible from any of those cameras");
            std::process::exit(2);
        }
        sig.sort_by(f32::total_cmp);
        voxel = sig[sig.len() / 2];
        println!("  {voxel_px} px at the median sampling rate is {voxel:.5} in world units");
    }
    if voxel > 0.0 {
        let w = t.opacities.clone();
        t = splat::prune::voxel_merge(&t, &w, voxel, max_points);
    }
    splat::ply::write(&out, &t).unwrap_or_else(|e| {
        eprintln!("PLY write failed: {e}");
        std::process::exit(1);
    });
    println!(
        "{path} -> {out}: {} -> {after_op} (opacity) -> {} gaussians",
        s.len(),
        t.len()
    );
}

/// Write cameras in the `cameras.json` format [`read_cameras`] reads.
pub fn write_cameras(path: &str, cams: &[Camera]) {
    std::fs::write(path, splat::types::cameras_to_json(cams)).unwrap_or_else(|e| {
        eprintln!("cannot write {path}: {e}");
        std::process::exit(1);
    });
}

/// Photographs, decoded, and structure from motion run on them: the shared
/// front half of `sfm` and `train`.
fn photogrammetry(a: &mut Args) -> (Vec<String>, recon::photogrammetry::TrainingSet) {
    let images = a.take_str("--images").unwrap_or_else(|| {
        eprintln!("--images <dir|a.jpg,b.jpg,...> is required");
        std::process::exit(2);
    });
    // The training resolution: SfM runs on the full photographs, the fit on
    // pinhole-resampled copies this wide.
    let width = a.u32_or("--width", 1024);
    let defaults = sfm::incremental::SfmCfg::default();
    let focal = a.f32_or("--focal-guess", defaults.focal_guess as f32);
    // What the sparse points start as. 3DGS starts at 0.1 so the fit decides
    // what is solid, but every pixel then composites many layers deep and the
    // backward's cost is proportional to that depth: on a 16-photo capture at
    // 768x576, 0.5 ran 2.9x faster per iteration than 0.1 AND was lower in
    // loss at iteration 100 (0.223 against 0.232).
    let opacity = a.f32_or("--init-opacity", 0.5);
    let paths = crate::mirror_cli::collect_images(&images);
    let photos: Vec<imaging::Rgb8> = paths
        .iter()
        .map(|p| {
            imaging::load(p).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            })
        })
        .collect();
    println!("structure from motion on {} photographs ...", photos.len());
    let cfg = sfm::incremental::SfmCfg { focal_guess: focal as f64, verbose: true, ..defaults };
    let set = recon::photogrammetry::training_set(&photos, width, opacity, &cfg).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let k = set.sfm.intrinsics;
    println!(
        "registered {}/{} photographs, {} points, reprojection rms {:.2} px, focal {:.1} px, k1 {:+.4}, k2 {:+.4}",
        set.targets.len(),
        photos.len(),
        set.sfm.points.len(),
        set.sfm.rms_px,
        k.f,
        k.k1,
        k.k2
    );
    for (i, p) in paths.iter().enumerate() {
        if !set.source.contains(&i) {
            println!("  not registered: {p}");
        }
    }
    let registered = set.source.iter().map(|&i| paths[i].clone()).collect();
    (registered, set)
}

/// Structure from motion alone: cameras and the sparse cloud, for inspection
/// or for `fit`.
fn sfm_cmd(argv: &[String]) {
    let mut a = Args::new(argv);
    let out = a.str_or("--out", "out/sfm.ply");
    let cams_out = a.str_or("--cameras-out", &format!("{out}.cameras.json"));
    let (paths, set) = photogrammetry(&mut a);
    a.finish();
    let cams: Vec<Camera> = set.targets.iter().map(|t| t.cam).collect();
    splat::ply::write(&out, &set.init).unwrap_or_else(|e| {
        eprintln!("PLY write failed: {e}");
        std::process::exit(1);
    });
    write_cameras(&cams_out, &cams);
    println!("{} points -> {out}, {} cameras -> {cams_out} (for images: {})", set.init.len(), cams.len(), paths.join(","));
}

/// Photographs to a finished splat scene: structure from motion, the sparse
/// cloud as the starting scene, and the full reconstruction objective
/// ([`FitCfg::from_sparse_points`]).
fn train_cmd(argv: &[String]) {
    let mut a = Args::new(argv);
    let out = crate::args::strip_out_name_prefix(&a.str_or("--out", "out/trained.ply"), "scene").to_string();
    let iters = a.usize_or("--iters", 3000);
    let budget = a.usize_or("--max-gaussians", 500_000);
    let lr = a.f32_or("--lr", 5e-3);
    let coarse = a.f32_or("--coarse", FitCfg::from_sparse_points(iters, budget, 0).coarse);
    let strategy = match a.str_or("--densify-strategy", "hybrid").as_str() {
        "heuristic" => Densify::Heuristic,
        "mcmc" => Densify::Mcmc,
        "hybrid" => Densify::Hybrid,
        other => {
            eprintln!("--densify-strategy must be `heuristic`, `mcmc` or `hybrid`, got `{other}`");
            std::process::exit(2);
        }
    };
    let cams_out = a.take_str("--cameras-out");
    // Per-photo exposure and white balance and the lens's vignetting, for a
    // capture that needs them; see `FitCfg::from_sparse_points` for why the
    // preset leaves them off.
    let camera_model = a.take_flag("--camera-model");
    let (_, set) = photogrammetry(&mut a);
    a.finish();
    // Land the scene upright: structure from motion leaves it in its first
    // camera's frame, however that camera was held.
    let cams: Vec<Camera> = set.targets.iter().map(|t| t.cam).collect();
    let (init, cams) = splat::orient::upright(&set.init, &cams);
    let targets: Vec<TargetView> = set
        .targets
        .into_iter()
        .zip(&cams)
        .map(|(mut t, c)| {
            t.cam = *c;
            t
        })
        .collect();
    let preset = FitCfg::from_sparse_points(iters, budget, targets.len());
    let isp = if camera_model { Some(splat::isp::IspCfg::default()) } else { preset.isp };
    let cfg = FitCfg { lr, coarse, strategy, isp, ..preset };
    let g = Gpu::new(splat::PIPELINES);
    println!(
        "training {} gaussians against {} views at {}x{} ({iters} iters, budget {budget}, SH degree {}, camera model {}) ...",
        init.len(),
        targets.len(),
        targets[0].cam.width,
        targets[0].cam.height,
        cfg.sh_degree,
        if cfg.isp.is_some() { "on" } else { "off" }
    );
    let res = splat::opt::fit_full(&g, Kernels::at(0), &init, &targets, &cfg, &mut |_, _| true);
    splat::ply::write(&out, &res.scene).unwrap_or_else(|e| {
        eprintln!("PLY write failed: {e}");
        std::process::exit(1);
    });
    let path = cams_out.unwrap_or_else(|| format!("{out}.cameras.json"));
    write_cameras(&path, &res.cams);
    println!(
        "{} gaussians -> {out} (final loss {:.6}); cameras -> {path}. Rendered with the Mip filter \
         (`brain splat render --aa --eps2d {}`).",
        res.scene.len(),
        res.loss,
        cfg.eps2d
    );
}

pub fn read_cameras(path: &str) -> Vec<Camera> {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    splat::types::cameras_from_json(&raw).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    })
}

fn read_cams(path: &str) -> Vec<[f64; 16]> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let j: serde_json::Value = serde_json::from_str(&text).expect("valid cameras.json");
    j.as_array()
        .expect("array of cameras")
        .iter()
        .map(|c| {
            let v: Vec<f64> = c["c2w"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            v.try_into().unwrap()
        })
        .collect()
}

/// Chain chunks of one long capture into a single scene.
///
/// Chunk i+1 is aligned onto chunk i using the frames they share, and the
/// resulting similarity is composed along the chain so everything lands in the
/// first chunk's world. Composing means drift accumulates - each link's error
/// is carried by every chunk after it - so the per-link residual is printed
/// rather than hidden; it is the number that says whether the overlap was big
/// enough.
fn merge_cmd(argv: &[String]) {
    let mut a = Args::new(argv);
    let plys: Vec<String> =
        a.positional().unwrap_or_default().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let cam_paths: Vec<String> =
        a.str_or("--cameras", "").split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let overlap = a.usize_or("--overlap", 0);
    let out = crate::args::strip_out_name_prefix(&a.str_or("--out", "out/merged.ply"), "scene").to_string();
    let cams_out = a.str_or("--cameras-out", &format!("{out}.cameras.json"));
    let prune = a.f32_or("--prune", 0.0);
    a.finish();

    if plys.len() < 2 || cam_paths.len() != plys.len() {
        eprintln!("merge: give N scenes and N cameras.json, comma separated (got {} and {})", plys.len(), cam_paths.len());
        std::process::exit(2);
    }
    if overlap < 2 {
        eprintln!("merge: --overlap must be at least 2 shared frames; a single shared frame fixes no scale");
        std::process::exit(2);
    }

    let cams: Vec<Vec<[f64; 16]>> = cam_paths.iter().map(|p| read_cams(p)).collect();
    let mut world = splat::align::Sim3::default();
    let mut parts: Vec<Splats> = Vec::new();
    let mut all: Vec<[f64; 16]> = Vec::new();
    for (i, path) in plys.iter().enumerate() {
        let s = splat::ply::read(path).unwrap_or_else(|e| {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(1);
        });
        if i > 0 {
            let prev = &cams[i - 1];
            let cur = &cams[i];
            if prev.len() < overlap || cur.len() < overlap {
                eprintln!("merge: chunk {i} has fewer cameras than the stated overlap");
                std::process::exit(2);
            }
            // the LAST `overlap` frames of the previous chunk are the FIRST
            // `overlap` frames of this one
            let tail: Vec<[f64; 16]> = prev[prev.len() - overlap..].to_vec();
            let head: Vec<[f64; 16]> = cur[..overlap].to_vec();
            let step = splat::align::sim3_from_cameras(&tail, &head).unwrap_or_else(|| {
                eprintln!("merge: chunk {i} does not overlap the one before it usefully");
                std::process::exit(2);
            });
            let res = splat::align::camera_residual(&tail, &head, &step);
            let span = {
                let e: Vec<[f64; 3]> = tail.iter().map(|m| [m[3], m[7], m[11]]).collect();
                let c = [
                    e.iter().map(|p| p[0]).sum::<f64>() / e.len() as f64,
                    e.iter().map(|p| p[1]).sum::<f64>() / e.len() as f64,
                    e.iter().map(|p| p[2]).sum::<f64>() / e.len() as f64,
                ];
                e.iter()
                    .map(|p| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt())
                    .fold(0.0f64, f64::max)
                    .max(1e-9)
            };
            println!(
                "  chunk {i}: scale x{:.4}, shared cameras land {:.4} off ({:.1}% of the overlap's own span)",
                step.s, res, 100.0 * res / span
            );
            world = world.after(&step);
        }
        for m in &cams[i] {
            all.push(splat::align::transform_c2w_sim3(m, &world));
        }
        parts.push(splat::align::apply_sim3(&s, &world));
    }

    let mut merged = splat::align::concat(&parts);
    if prune > 0.0 {
        // Overlapping chunks cover the same surface twice, which a viewer
        // sees as stacked translucency. Fuse by opacity: a confident gaussian
        // should outvote a faint duplicate of itself.
        let w = merged.opacities.clone();
        merged = splat::prune::voxel_merge(&merged, &w, prune, 0);
    }
    splat::ply::write(&out, &merged).unwrap_or_else(|e| {
        eprintln!("PLY write failed: {e}");
        std::process::exit(1);
    });
    let js: Vec<serde_json::Value> = all.iter().map(|m| serde_json::json!({ "c2w": m.to_vec() })).collect();
    std::fs::write(&cams_out, serde_json::to_string_pretty(&js).unwrap()).ok();
    println!("{} chunks -> {out} ({} gaussians; cameras -> {cams_out})", plys.len(), merged.len());
}

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
    // Inria-convention viewers render the uncompensated dilation, and a scene
    // fitted under one low-pass and rendered under another comes out wrong in
    // both directions. Fit for the renderer that will show it.
    let inria = a.take_flag("--inria-lowpass");
    let mip_scale = a.f32_or("--mip-scale", FitCfg::default().mip_scale);
    // View-dependent colour. Every extra degree is more freedom to explain a
    // view away without moving geometry, which cuts both ways when the views
    // are few, so it is a dial rather than a default.
    let sh_degree = a.u32_or("--sh-degree", FitCfg::default().sh_degree);
    // Refine the cameras alongside the scene. A pipeline that predicts its own
    // poses hands this step cameras that are wrong, and a pose error is a
    // position error for every pixel of that frame.
    let pose_lr = a.f32_or("--pose-lr", FitCfg::default().pose_lr);
    let cams_out = a.take_str("--cameras-out");
    // Density control: let the fit ADD gaussians where the loss is still
    // pulling. Off unless asked, because a feed-forward scene is already
    // dense and growing it can push the backward past the device's
    // gradient-record ceiling mid-run.
    let densify_every = a.usize_or("--densify", 0);
    let densify_frac = a.f32_or("--densify-frac", FitCfg::default().densify_frac);
    let max_gaussians = a.usize_or("--max-gaussians", 0);
    // Which density control. The heuristic is the default so that `--densify`
    // keeps meaning what it meant; `mcmc` is 3DGS-MCMC, which spends
    // `--max-gaussians` as a budget and recycles transparent gaussians
    // instead of deleting them.
    let strategy = match a.str_or("--densify-strategy", "heuristic").as_str() {
        "heuristic" => Densify::Heuristic,
        "mcmc" => Densify::Mcmc,
        "hybrid" => Densify::Hybrid,
        other => {
            eprintln!("--densify-strategy must be `heuristic`, `mcmc` or `hybrid`, got `{other}`");
            std::process::exit(2);
        }
    };
    let loss = match a.str_or("--loss", "mse").as_str() {
        "mse" => splat::loss::PixelLoss::Mse,
        "l1-ssim" => splat::loss::PixelLoss::gaussian_splatting(),
        other => {
            eprintln!("--loss must be `mse` or `l1-ssim`, got `{other}`");
            std::process::exit(2);
        }
    };
    // Fit a photometric camera model (exposure, white balance, vignetting,
    // response) with the scene, so a view shot a stop brighter is explained by
    // its camera rather than by brighter gaussians.
    let isp = a.take_flag("--camera-model").then(splat::isp::IspCfg::default);
    let batch = a.usize_or("--batch", 0);
    let distortion_weight = a.f32_or("--distortion", 0.0);
    let normal_consistency_weight = a.f32_or("--normal-consistency", 0.0);
    let geometry_after = a.f32_or("--geometry-after", 0.0);
    a.finish();

    let cams = read_cameras(&cams_path);
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
            TargetView::new(*cam, rgb)
        })
        .collect();

    let g = Gpu::new(splat::PIPELINES);
    let ks = Kernels::at(0);
    println!("fitting {} gaussians against {} views ({} iters, lr {lr}) …", s.len(), targets.len(), iters);
    // How large a fitted gaussian may get, in pixels of the view that samples
    // it best. A gaussian wider than a couple of pixels cannot carry detail
    // the cameras resolved; it can only blur it, and from a grazing angle a
    // scene of them is fog. The two shape bounds cap the axis RATIOS - see
    // `splat::opt::clamp_axes` - which is a separate question from size.
    let max_scale_pixels = a.f32_or("--max-scale-pixels", 16.0);
    let max_growth = a.f32_or("--max-growth", 2.0);
    let max_needle = a.f32_or("--max-needle", 2.0);
    let max_flat = a.f32_or("--max-flat", 4.0);
    // Radius-relative budgets for the three geometry groups: how far a
    // gaussian may travel, resize or turn over the WHOLE run, measured in
    // units of a median gaussian's own radius. `p_geo` packs a position in
    // world units, a linear scale and a unit quaternion into one buffer, and
    // Adam's step is ~lr regardless of gradient, so one rate cannot serve all
    // three - at a rate small enough to be a rounding error on the scene
    // diagonal it is still a sixth of a small gaussian's own radius.
    //
    // Set these when the scene is ALREADY metric, as a feed-forward
    // reconstruction's is: then a gaussian should settle onto its surface and
    // not leave it, and `--lr` becomes the appearance rate (the budget divides
    // by it, so raising it cannot move a gaussian further than it already
    // could). They are off by default because a fit from a sparse point cloud
    // needs the opposite - there the geometry HAS to migrate a long way.
    let position_budget = a.f32_or("--position-budget", FitCfg::default().position_budget);
    let scale_budget = a.f32_or("--scale-budget", FitCfg::default().scale_budget);
    let rotation_budget = a.f32_or("--rotation-budget", FitCfg::default().rotation_budget);
    let cfg = FitCfg {
        iters,
        lr,
        eps2d: if inria { 0.3 } else { eps2d },
        antialiased: !inria,
        mip_scale,
        densify_every,
        densify_frac,
        max_gaussians,
        strategy,
        sh_degree,
        pose_lr,
        max_scale_pixels,
        max_growth,
        max_needle,
        max_flat,
        position_budget,
        scale_budget,
        rotation_budget,
        loss,
        isp,
        batch,
        distortion_weight,
        normal_consistency_weight,
        geometry_after,
        ..Default::default()
    };
    let (fitted, refined, mse) =
        splat::opt::fit_bundle(&g, ks, &s, &targets, &cfg, &mut |_it, _mse| true);
    if pose_lr > 0.0 {
        // The refined poses ARE part of the result: a scene fitted against
        // moved cameras only means anything when read back with those cameras.
        let path = cams_out.clone().unwrap_or_else(|| format!("{out}.cameras.json"));
        write_cameras(&path, &refined);
        println!("refined cameras -> {path}");
    }
    let grown = fitted.len();
    splat::ply::write(&out, &fitted).unwrap_or_else(|e| {
        eprintln!("PLY write failed: {e}");
        std::process::exit(1);
    });
    println!(
        "{path} -> {out} ({grown} gaussians, final loss {mse:.6}, fitted at --eps2d {} with \
         compensation {}; render it the same way)",
        cfg.eps2d,
        cfg.antialiased
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
