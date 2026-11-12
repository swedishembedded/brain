// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The window: a winit event loop presenting the renderer to a wgpu surface.
//!
//! The glass a frame is rendered through is the part of the physical display
//! the window's client area covers, found from the monitor's pixel size and
//! the window's position on it - so dragging the window across the screen
//! looks through a different part of the same fixed scene, which is how a
//! real window behaves. Where the window system does not report a position
//! (Wayland), the window is taken to be centred.

use std::sync::Arc;
use std::time::Instant;

use headpose::Display;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Window, WindowId};

use crate::eye::{self, Eye, Source};
use crate::renderer::Renderer;
use crate::tracking::{self, Tracking};
use crate::{hud, scene, view_through, Args};

/// How far ahead of "now" a frame is seen: one refresh at 60 Hz, the time a
/// frame rendered now waits for its scan-out.
const PRESENT_LEAD_S: f64 = 1.0 / 60.0;

fn now_s() -> f64 {
    capture::clock::monotonic_ns() as f64 * 1e-9
}

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    adapter_name: String,
}

struct App {
    args: Args,
    display: Display,
    gfx: Option<Gfx>,
    error: Option<String>,
    start: Instant,
    /// Cursor position as fractions of the window, `(0, 0)` top-left.
    cursor: [f64; 2],
    manual_z: f64,
    nudge: [f64; 2],
    frozen: Option<Eye>,
    grid: bool,
    diagnostics: bool,
    shots: u32,
    frame_times: Vec<f64>,
    presented: u64,
    tracking: Option<Tracking>,
    /// The last calibration message, shown in the HUD.
    note: String,
}

pub fn run(args: Args) -> Result<(), String> {
    let display = Display::flat(args.display_w_m, args.display_h_m).map_err(|e| e.to_string())?;
    let tracking = match args.source {
        Source::Webcam => {
            eprintln!("fishtank: opening {:?} and loading the face tracker", args.camera);
            let t = Tracking::start(tracking::Options {
                camera: args.camera.clone(),
                size: args.camera_size,
                landmarks: args.landmarks,
                device: args.tracker_device.clone(),
                profile: args.profile.clone(),
                save_to: args.profile_path.clone(),
            })
            .map_err(|e| format!("{e}\n(run with --pose-source manual to use the mouse instead)"))?;
            Some(t)
        }
        _ => None,
    };
    let event_loop = EventLoop::new().map_err(|e| format!("no window system: {e} (use --headless --output PATH to render without one)"))?;
    let mut app = App {
        args,
        display,
        gfx: None,
        error: None,
        start: Instant::now(),
        cursor: [0.5, 0.5],
        manual_z: eye::REST[2],
        nudge: [0.0; 2],
        frozen: None,
        grid: true,
        diagnostics: true,
        shots: 0,
        frame_times: Vec::new(),
        presented: 0,
        tracking,
        note: String::new(),
    };
    event_loop.run_app(&mut app).map_err(|e| format!("event loop: {e}"))?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

impl App {
    fn init(&mut self, el: &ActiveEventLoop) -> Result<Gfx, String> {
        let (w, h) = self.args.size;
        let mut attrs = Window::default_attributes().with_title("brain fishtank").with_inner_size(PhysicalSize::new(w, h));
        if self.args.fullscreen {
            attrs = attrs.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }
        let window = Arc::new(el.create_window(attrs).map_err(|e| format!("opening a window: {e}"))?);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(Box::new(el.owned_display_handle())));
        let surface = instance.create_surface(window.clone()).map_err(|e| format!("creating the window surface: {e}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .map_err(|e| format!("no GPU adapter can present to this window: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("fishtank"), ..Default::default() })).map_err(|e| format!("opening the GPU: {e}"))?;
        let size = window.inner_size();
        let mut config = surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)).ok_or("the GPU cannot present to this window")?;
        let caps = surface.get_capabilities(&adapter);
        if let Some(f) = caps.formats.iter().find(|f| f.is_srgb()) {
            config.format = *f;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);
        let tank = scene::Tank::for_display(self.args.display_w_m, self.args.display_h_m);
        let renderer = Renderer::new(device, queue, config.format, config.width, config.height, tank);
        let info = adapter.get_info();
        eprintln!("fishtank: {} ({:?}), {}x{} {:?}", info.name, info.backend, config.width, config.height, config.format);
        Ok(Gfx { window, surface, config, renderer, adapter_name: info.name })
    }

    /// The part of the display the window covers.
    fn glass(&self, g: &Gfx) -> Display {
        let size = g.window.inner_size();
        let Some(mon) = g.window.current_monitor() else { return self.display };
        let ms = mon.size();
        let (x, y) = match g.window.inner_position() {
            Ok(p) => (p.x - mon.position().x, p.y - mon.position().y),
            Err(_) => ((ms.width as i32 - size.width as i32) / 2, (ms.height as i32 - size.height as i32) / 2),
        };
        self.display.window((ms.width, ms.height), x, y, size.width, size.height).unwrap_or(self.display)
    }

    fn eye(&self, glass: &Display, t: f64) -> Eye {
        if let Some(e) = self.frozen {
            return e;
        }
        let e = match self.args.source {
            Source::Fixed(e) => e,
            Source::Synthetic => eye::synthetic(t),
            Source::Webcam => self.tracking.as_ref().map(|tr| tr.viewer(now_s() + PRESENT_LEAD_S).cyclopean()).unwrap_or(eye::REST),
            Source::Manual => {
                let p = glass.point(self.cursor[0], 1.0 - self.cursor[1]);
                eye::manual([p[0], p[1]], self.manual_z)
            }
        };
        [e[0] + self.nudge[0], e[1] + self.nudge[1], e[2]]
    }

    fn hud_lines(&self, g: &Gfx, eye: Eye, glass: &Display) -> Vec<String> {
        let fps = if self.frame_times.len() > 1 { (self.frame_times.len() - 1) as f64 / (self.frame_times.last().unwrap() - self.frame_times[0]) } else { 0.0 };
        let mut lines = vec![
            format!("FISHTANK  {}{}", self.args.source.label(), if self.frozen.is_some() { "  [FROZEN]" } else { "" }),
            format!("eye   x {:+6.1}  y {:+6.1}  z {:6.1} cm", eye[0] * 100.0, eye[1] * 100.0, eye[2] * 100.0),
            format!("glass {:.1} x {:.1} cm   render {:5.1} fps", glass.width() * 100.0, glass.height() * 100.0, fps),
            format!("gpu   {}", g.adapter_name),
        ];
        if let Some(tr) = &self.tracking {
            let v = tr.viewer(now_s() + PRESENT_LEAD_S);
            let s = tr.stats.lock().unwrap().clone();
            lines.push(format!("track {:?}  conf {:.2}  fit {:.1} px  age {:3.0} ms  lead {:2.0} ms", v.status, s.confidence, s.fit_rmse_px, v.sample_age_s.min(9.999) * 1e3, v.lead_s * 1e3));
            lines.push(format!("cam   {}  infer {:4.1} ms  frames {}  dropped {}  detects {}", s.camera, s.inference_ms, s.frames, s.dropped, s.detections));
            if !s.last_problem.is_empty() {
                lines.push(format!("      {}", s.last_problem.chars().take(70).collect::<String>()));
            }
            lines.push("keys  C recentre  Z set distance  Esc G D Space P F".to_string());
        } else {
            lines.push("keys  Esc G D Space P F  arrows  PgUp/PgDn".to_string());
        }
        if !self.note.is_empty() {
            lines.push(self.note.chars().take(76).collect());
        }
        lines
    }

    fn frame(&mut self) -> Result<(), String> {
        let Some(g) = self.gfx.as_ref() else { return Ok(()) };
        let t = self.start.elapsed().as_secs_f64();
        self.frame_times.push(t);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
        let glass = self.glass(g);
        let eye = self.eye(&glass, t);
        let hud = self.diagnostics.then(|| hud::render(&self.hud_lines(g, eye, &glass)));
        let v = view_through(&glass, eye, t as f32, self.grid, hud.as_ref().map(|(p, w, h)| (p.as_slice(), *w, *h)))?;
        let g = self.gfx.as_mut().expect("checked above");
        let tex = match g.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                g.surface.configure(&g.renderer.device, &g.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            other => return Err(format!("the window surface failed: {other:?}")),
        };
        let target = tex.texture.create_view(&wgpu::TextureViewDescriptor::default());
        g.renderer.render(&target, &v);
        g.window.pre_present_notify();
        tex.present();
        self.presented += 1;
        Ok(())
    }

    /// Render the current view offscreen, with the HUD when it is on, and
    /// save it to `path` (or the next `fishtank-NNN.png`).
    fn screenshot(&mut self, path: Option<&std::path::Path>) -> Result<String, String> {
        let Some(g) = self.gfx.as_ref() else { return Err("no window yet".into()) };
        let t = self.start.elapsed().as_secs_f64();
        let glass = self.glass(g);
        let eye = self.eye(&glass, t);
        let hud = self.diagnostics.then(|| hud::render(&self.hud_lines(g, eye, &glass)));
        let v = view_through(&glass, eye, t as f32, self.grid, hud.as_ref().map(|(p, w, h)| (p.as_slice(), *w, *h)))?;
        let g = self.gfx.as_mut().expect("checked above");
        let img = g.renderer.render_to_rgb(&v)?;
        let path = match path {
            Some(p) => p.display().to_string(),
            None => {
                self.shots += 1;
                format!("fishtank-{:03}.png", self.shots)
            }
        };
        imaging::codec::save_png(&path, &img)?;
        Ok(path)
    }

    fn key(&mut self, el: &ActiveEventLoop, k: &Key) {
        let step = 0.01;
        match k {
            Key::Named(NamedKey::Escape) => el.exit(),
            Key::Named(NamedKey::ArrowLeft) => self.nudge[0] -= step,
            Key::Named(NamedKey::ArrowRight) => self.nudge[0] += step,
            Key::Named(NamedKey::ArrowUp) => self.nudge[1] += step,
            Key::Named(NamedKey::ArrowDown) => self.nudge[1] -= step,
            Key::Named(NamedKey::PageUp) => self.manual_z = (self.manual_z - 0.02).max(0.15),
            Key::Named(NamedKey::PageDown) => self.manual_z = (self.manual_z + 0.02).min(1.5),
            Key::Named(NamedKey::Space) => {
                self.frozen = match self.frozen {
                    Some(_) => None,
                    None => self.gfx.as_ref().map(|g| {
                        let glass = self.glass(g);
                        self.eye(&glass, self.start.elapsed().as_secs_f64())
                    }),
                }
            }
            Key::Character(c) => match c.as_str() {
                "g" | "G" => self.grid = !self.grid,
                "c" | "C" => {
                    if let Some(tr) = &self.tracking {
                        self.note = tr.recentre(now_s()).unwrap_or_else(|e| format!("recentre: {e}"));
                        eprintln!("fishtank: {}", self.note);
                    }
                }
                "z" | "Z" => {
                    if let Some(tr) = &self.tracking {
                        self.note = tr.set_distance(now_s(), self.args.calibration_distance_m).unwrap_or_else(|e| format!("set distance: {e}"));
                        eprintln!("fishtank: {}", self.note);
                    }
                }
                "d" | "D" => self.diagnostics = !self.diagnostics,
                "f" | "F" => {
                    if let Some(g) = self.gfx.as_ref() {
                        let full = g.window.fullscreen().is_some();
                        g.window.set_fullscreen(if full { None } else { Some(Fullscreen::Borderless(None)) });
                    }
                }
                "p" | "P" => match self.screenshot(None) {
                    Ok(p) => eprintln!("fishtank: saved {p}"),
                    Err(e) => eprintln!("fishtank: screenshot failed: {e}"),
                },
                _ => {}
            },
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        match self.init(el) {
            Ok(g) => self.gfx = Some(g),
            Err(e) => {
                self.error = Some(e);
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gfx.as_mut() {
                    if size.width > 0 && size.height > 0 {
                        g.config.width = size.width;
                        g.config.height = size.height;
                        g.surface.configure(&g.renderer.device, &g.config);
                        g.renderer.resize(size.width, size.height);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(g) = self.gfx.as_ref() {
                    let s = g.window.inner_size();
                    self.cursor = [position.x / s.width.max(1) as f64, position.y / s.height.max(1) as f64];
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                self.manual_z = (self.manual_z - 0.02 * d).clamp(0.15, 1.5);
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => self.key(el, &event.logical_key),
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.frame() {
                    self.error = Some(e);
                    el.exit();
                }
                if self.args.frames.is_some_and(|n| self.presented >= n) {
                    let fps = if self.frame_times.len() > 1 { (self.frame_times.len() - 1) as f64 / (self.frame_times.last().unwrap() - self.frame_times[0]) } else { 0.0 };
                    eprintln!("fishtank: presented {} frames, {fps:.1} fps over the last {}", self.presented, self.frame_times.len());
                    if let Some(out) = self.args.output.clone() {
                        match self.screenshot(Some(&out)) {
                            Ok(p) => eprintln!("fishtank: saved the last frame to {p}"),
                            Err(e) => eprintln!("fishtank: saving the last frame failed: {e}"),
                        }
                    }
                    if let Some(tr) = &self.tracking {
                        let v = tr.viewer(now_s() + PRESENT_LEAD_S);
                        let s = tr.stats.lock().unwrap().clone();
                        let e = v.cyclopean();
                        eprintln!(
                            "fishtank: tracking {:?}, eye ({:.1}, {:.1}, {:.1}) cm, conf {:.2}, fit {:.1} px, {} frames ({} dropped, {} detections), inference {:.1} ms{}",
                            v.status, e[0] * 100.0, e[1] * 100.0, e[2] * 100.0, s.confidence, s.fit_rmse_px, s.frames, s.dropped, s.detections, s.inference_ms,
                            if s.last_problem.is_empty() { String::new() } else { format!(", last problem: {}", s.last_problem) }
                        );
                    }
                    el.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(g) = self.gfx.as_ref() {
            g.window.request_redraw();
        }
    }
}
