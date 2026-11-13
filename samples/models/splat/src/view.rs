// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `view`: an interactive fly-through of a scene, in a window.
//!
//! WASD move, Space/C up and down, Shift sprints, `m` captures the mouse for
//! look (the arrow keys look too), `[` and `]` trade render resolution for
//! speed, `v` toggles colour and depth, `p` saves a screenshot, Enter returns
//! to the start, Esc releases the mouse and then quits.
//!
//! It is an application, not a capability: it is a human-in-the-loop frame
//! loop with no request and response to serve. It takes the flags of
//! `render` that describe how a frame looks (`--width`, `--height`, `--fov`,
//! `--bg`, `--eye`/`--target`/`--up` or `--cameras`/`--view` for where it
//! starts, `--depth` to start in the depth view, `--antialiased`,
//! `--inria-lowpass`, `--eps2d`, `--isect-cap`) and, of its own, `--frames N`
//! to quit after N frames and `--shots DIR` for where screenshots go.
//! Without a start pose the camera is framed from the scene's bounds.
//!
//! Swedish Embedded AB implements real-time 3D scene viewers on its clients'
//! own hardware. If your team needs expertise in interactive rendering of
//! captured scenes then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::{Path, PathBuf};

use gpu_core::Gpu;
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{auto_camera, cross3, norm3, Camera, Mode, RenderOpts, Splats};
use splat::Kernels;
use wm_display::keymap::{Key, KeySet, UxKey};
use wm_display::sink::{FrameSink, Hud};
use wm_display::window::SdlWindow;

/// Pitch is kept this far (radians) from straight up or down, where the
/// camera's right vector degenerates.
const PITCH_LIMIT: f32 = 1.55;
/// Radians of look per pixel of mouse motion.
const MOUSE_LOOK: f32 = 0.003;
/// Radians per second the arrow keys look.
const KEY_LOOK: f32 = 1.8;
/// A fraction of the scene's extent per second a held movement key covers.
const MOVE_SPEED: f32 = 0.25;
/// What holding Shift multiplies the speed by.
const SPRINT: f32 = 4.0;
/// A frame longer than this counts as this long, so a stall does not throw
/// the camera across the scene.
const MAX_STEP_SECONDS: f32 = 0.1;

/// Fly-camera state: position + yaw/pitch in the y-down world.
#[derive(Clone, Copy)]
struct FlyCam {
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
}

impl FlyCam {
    fn from_camera(cam: &Camera) -> FlyCam {
        // forward = third COLUMN of the c2w rotation.
        let f = [cam.c2w[2], cam.c2w[6], cam.c2w[10]];
        FlyCam { pos: cam.eye(), yaw: f[0].atan2(f[2]), pitch: f[1].asin() }
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
        let c2w = [r[0], d[0], f[0], e[0], r[1], d[1], f[1], e[1], r[2], d[2], f[2], e[2], 0.0, 0.0, 0.0, 1.0];
        let fy = 0.5 * height as f32 / (0.5 * fov_y_deg.to_radians()).tan();
        Camera::pinhole(c2w, fy, fy, width as f32 / 2.0, height as f32 / 2.0, width, height)
    }
}

/// Everything a viewing session is given.
pub struct Viewer<'a> {
    pub scene: &'a Splats,
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub fov: f32,
    /// Background, antialiasing and dilation; its `mode` is the starting view.
    pub opts: RenderOpts,
    /// Tile-instance buffer capacity; 0 = the renderer's own.
    pub isect_cap: usize,
    pub start: Camera,
    /// Quit after this many frames; `None` = until Esc.
    pub max_frames: Option<u64>,
    pub shots_dir: PathBuf,
}

/// Render resolutions as divisors of the window size; the window stays one
/// size while the render buffer shrinks and SDL stretches it back up.
const DIVISORS: [u32; 3] = [1, 2, 4];

impl Viewer<'_> {
    pub fn run(&self) -> Result<(), String> {
        let gpu = Gpu::new(splat::PIPELINES);
        let mut renderer = Renderer::new(&gpu, Kernels::at(0), self.scene.len(), self.width, self.height, self.isect_cap);
        let splats = GpuSplats::upload(&gpu, self.scene);
        let (lo, hi) = self.scene.bounds();
        let scene_size = (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f32>().sqrt().max(1e-3);

        let open = |div: u32| SdlWindow::new(&self.title, self.width / div, self.height / div, div).map_err(|e| format!("cannot open a window: {e} (headless? try SDL_VIDEODRIVER=dummy)"));
        let mut quality = 0usize;
        let mut win = open(DIVISORS[quality])?;
        let mut fly = FlyCam::from_camera(&self.start);
        let mut mode = self.opts.mode;
        let mut captured = false;
        let mut fps = 0.0f32;
        let mut frame_no = 0u64;
        let mut shot_no = 0u32;
        let mut last = std::time::Instant::now();
        loop {
            let input = win.pump();
            if input.quit {
                if !captured {
                    break;
                }
                // the first Esc releases the mouse, the second quits
                captured = false;
                win.set_relative_mouse(false);
            }
            let mut reopen = false;
            for ux in &input.ux {
                match ux {
                    UxKey::ToggleMouse => {
                        captured = !captured;
                        win.set_relative_mouse(captured);
                    }
                    UxKey::CycleView => mode = if mode == Mode::Color { Mode::Depth } else { Mode::Color },
                    UxKey::QualityDown if quality + 1 < DIVISORS.len() => {
                        quality += 1;
                        reopen = true;
                    }
                    UxKey::QualityUp if quality > 0 => {
                        quality -= 1;
                        reopen = true;
                    }
                    UxKey::Reset => fly = FlyCam::from_camera(&self.start),
                    _ => {}
                }
            }
            if reopen {
                drop(win);
                win = open(DIVISORS[quality])?;
                if captured {
                    win.set_relative_mouse(true);
                }
            }

            let dt = last.elapsed().as_secs_f32().min(MAX_STEP_SECONDS);
            last = std::time::Instant::now();
            let held = |k: Key| input.pressed.contains(KeySet::of(&[k]));

            // look: mouse (captured) and the arrow keys
            if captured {
                fly.yaw += input.mouse_dx as f32 * MOUSE_LOOK;
                fly.pitch = (fly.pitch + input.mouse_dy as f32 * MOUSE_LOOK).clamp(-PITCH_LIMIT, PITCH_LIMIT);
            }
            let look = KEY_LOOK * dt;
            if held(Key::Left) {
                fly.yaw -= look;
            }
            if held(Key::Right) {
                fly.yaw += look;
            }
            if held(Key::Up) {
                fly.pitch = (fly.pitch - look).clamp(-PITCH_LIMIT, PITCH_LIMIT);
            }
            if held(Key::Down) {
                fly.pitch = (fly.pitch + look).clamp(-PITCH_LIMIT, PITCH_LIMIT);
            }

            // move: WASD planar, Space/C vertical (up is -Y in this world), Shift sprints
            let fwd = fly.forward();
            let right = norm3(cross3(fwd, [0.0, -1.0, 0.0]));
            let mut step = [0.0f32; 3];
            for (key, dir, sign) in [(Key::W, fwd, 1.0), (Key::S, fwd, -1.0), (Key::D, right, 1.0), (Key::A, right, -1.0), (Key::Space, [0.0, -1.0, 0.0], 1.0), (Key::C, [0.0, 1.0, 0.0], 1.0)] {
                if held(key) {
                    step.iter_mut().zip(dir).for_each(|(s, d)| *s += d * sign);
                }
            }
            let speed = MOVE_SPEED * scene_size * dt * if held(Key::Shift) { SPRINT } else { 1.0 };
            fly.pos.iter_mut().zip(step).for_each(|(p, s)| *p += s * speed);

            // render and present
            let (rw, rh) = (self.width / DIVISORS[quality], self.height / DIVISORS[quality]);
            let cam = fly.camera(self.fov, rw, rh);
            let opts = RenderOpts { mode, ..self.opts };
            let started = std::time::Instant::now();
            if let Some(colors) = splat::sh::shade(self.scene, cam.eye()) {
                gpu.write_f32(&splats.colors, &colors);
            }
            let stats = renderer.render(&gpu, &splats, &cam, &opts);
            let mut rgb = renderer.read_rgb24(&gpu, rw, rh);
            let render_ms = started.elapsed().as_secs_f32() * 1000.0;
            let instant_fps = 1000.0 / render_ms.max(0.1);
            fps = if fps == 0.0 { instant_fps } else { 0.9 * fps + 0.1 * instant_fps };

            let hud_line = format!(
                "{}G {}I | {:.0} FPS {:.0}MS | {}X{} | POS {:.1} {:.1} {:.1}{}",
                self.scene.len(),
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
            imaging::viz::draw_text(&mut rgb, rw, rh, 6, 6, &hud_line, 1, [0, 255, 0]);
            if input.ux.contains(&UxKey::Screenshot) {
                let path = self.shots_dir.join(format!("splat-shot-{shot_no:03}.png"));
                save_screenshot(&path, rgb.clone(), rw, rh)?;
                println!("saved {}", path.display());
                shot_no += 1;
            }
            let hud = Hud { model: "splat".into(), fps, target_fps: 60, step: frame_no, paused: false, quality: quality as u32, action: 0, reset: false };
            win.frame(&rgb, rw, rh, &hud);

            frame_no += 1;
            if self.max_frames.is_some_and(|max| frame_no >= max) {
                return Ok(());
            }
        }
        Ok(())
    }
}

fn save_screenshot(path: &Path, rgb: Vec<u8>, w: u32, h: u32) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let image = imaging::Rgb8::new(w, h, rgb).map_err(|e| format!("screenshot: {e}"))?;
    imaging::save(path, &image).map_err(|e| format!("{}: {e}", path.display()))
}

/// The vertical field of view a pinhole camera of height `height` and focal
/// length `fy` (pixels) has, in degrees.
pub fn fov_of(fy: f32, height: u32) -> f32 {
    2.0 * (0.5 * height as f32 / fy).atan().to_degrees()
}

/// The camera a view starts from: the `start` pose when the request names
/// one, else the scene framed from its bounds.
pub fn start_camera(scene: &Splats, start: Option<Camera>, width: u32, height: u32, fov: f32) -> Camera {
    start.unwrap_or_else(|| auto_camera(scene, width, height, fov))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fly camera built from a camera and turned back into one looks the
    /// same way from the same place: the viewer starts exactly where it was
    /// told to.
    #[test]
    fn a_fly_camera_keeps_the_pose_it_started_from() {
        let cam = Camera::look_at([1.0, -2.0, 3.0], [0.5, 0.0, 6.0], [0.0, -1.0, 0.0], 60.0, 64, 48);
        let back = FlyCam::from_camera(&cam).camera(60.0, 64, 48);
        for (a, b) in cam.c2w.iter().zip(back.c2w) {
            assert!((a - b).abs() < 1e-4, "{:?} vs {:?}", cam.c2w, back.c2w);
        }
    }

    #[test]
    fn the_field_of_view_round_trips_through_the_focal_length() {
        let cam = FlyCam { pos: [0.0; 3], yaw: 0.0, pitch: 0.0 }.camera(75.0, 100, 80);
        assert!((fov_of(cam.fy, 80) - 75.0).abs() < 1e-3);
    }
}
