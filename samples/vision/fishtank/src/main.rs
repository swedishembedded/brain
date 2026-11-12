// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! sample: vision/fishtank - head-coupled perspective ("fish tank VR").
//!
//! The laptop screen becomes a window onto a small scene that stays fixed in
//! the room: a neon tank behind the glass, floating crystal, pillars and
//! rings at several depths, orbs in front of the glass and stars beyond.
//! Each frame is rendered through the off-axis projection from the eye
//! (`headpose::off_axis`), so moving your head shows the parallax a real
//! window would.
//!
//! Swedish Embedded AB implements interactive, camera-driven 3D for products
//! that react to the person using them. If your team needs expertise in
//! real-time rendering and perception on ordinary hardware, you can procure
//! our services by sending an email to info@swedishembedded.com.

mod app;
mod eye;
mod hud;
mod renderer;
mod scene;
mod tracking;

use std::path::PathBuf;
use std::process::ExitCode;

use headpose::calibration::Profile;
use headpose::{off_axis, Display};

const USAGE: &str = "\
sample-vision-fishtank - a scene that stays put behind (and in front of) your screen as you move

USAGE:
    sample-vision-fishtank [--pose-source webcam|manual|synthetic] [--eye X,Y,Z] [DISPLAY] [CAMERA] [--fullscreen]
    sample-vision-fishtank --headless --output PATH [--eye X,Y,Z | --pose-source synthetic] [--size WxH] [--time S]

EYE:
    --pose-source webcam   track your head with the webcam (default; needs fetch-data.sh once)
    --pose-source manual   the mouse steers the eye, the wheel moves it nearer/further
    --pose-source synthetic  a scripted orbit around a centred viewer
    --eye X,Y,Z            a fixed eye in centimetres from the screen centre (+X right, +Y up, +Z toward you)

DISPLAY (what the glass physically is):
    --display-cm W,H       visible screen area in centimetres (default 34.4,19.4: a 15.6\" 16:9 panel)
    --profile PATH         a calibration profile (JSON, headpose::calibration::Profile); wins over the
                           display and camera flags, and is where C and Z save to
                           (default ~/.config/brain/fishtank-profile.json, loaded when it exists)

CAMERA (webcam):
    --camera DEV           V4L2 device (default /dev/video0; must offer YUYV)
    --camera-image PATH    instead of a camera, replay a still photo at 30 Hz, panned sideways
                           (a test of the whole tracking chain; labelled as a replay)
    --camera-size WxH      requested capture size (default 640x480)
    --camera-hfov DEG      the webcam's horizontal field of view (default 60)
    --camera-gap-cm CM     lens centre above the top edge of the visible screen (default 0.8)
    --ipd-mm MM            your interpupillary distance (default 63)
    --landmarks fast|best  OpenSeeFace landmark network (default fast)
    --tracker-device DEV   where the face networks run: cpu (default; leaves the GPU to the
                           renderer), gpu, gpu<i>
    --calibration-distance-cm CM  the true eye-to-screen distance the Z key calibrates to (default 50)

HEADLESS:
    --headless             render offscreen, no window
    --output PATH          the PNG to write
    --size WxH             output size in pixels (default 1280x720)
    --time S               scene time of the frame, seconds (default 4)
    --hud                  draw the diagnostics overlay into the PNG too

BOTH:
    --frames N             stop after N frames (headless: render N frames 1/60 s apart, keep the last;
                           window: with --output PATH, save the last frame, HUD included)

KEYS (window):
    Esc quit   G grid   D diagnostics   Space freeze the eye   P screenshot   F fullscreen
    arrows nudge the eye   PageUp/PageDown nearer/further
    C  (webcam) sit centred in front of the screen and press: recentre, saved to the profile
    Z  (webcam) sit at --calibration-distance-cm from the screen and press: fix depth, saved
";

pub struct Args {
    pub source: eye::Source,
    pub display_w_m: f64,
    pub display_h_m: f64,
    pub fullscreen: bool,
    pub headless: bool,
    pub output: Option<PathBuf>,
    pub size: (u32, u32),
    pub time: f64,
    pub frames: Option<u64>,
    pub hud: bool,
    pub camera: tracking::Camera,
    pub camera_size: (u32, u32),
    pub landmarks: openseeface::config::NetConfig,
    pub tracker_device: String,
    pub calibration_distance_m: f64,
    /// The calibration the webcam mode starts from.
    pub profile: Profile,
    /// Where C and Z save the corrected profile.
    pub profile_path: PathBuf,
}

/// Where `C` and `Z` save, and where a later run finds, the calibration.
fn default_profile_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("brain").join("fishtank-profile.json")
}

fn parse_wxh(flag: &str, v: &str) -> Result<(u32, u32), String> {
    let (w, h) = v.split_once('x').ok_or_else(|| format!("{flag} {v:?}: expected WxH"))?;
    let parse = |s: &str| s.parse::<u32>().ok().filter(|n| (16..=8192).contains(n));
    Ok((parse(w).ok_or(format!("{flag}: width must be 16..8192"))?, parse(h).ok_or(format!("{flag}: height must be 16..8192"))?))
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        source: eye::Source::Webcam,
        display_w_m: 0.344,
        display_h_m: 0.194,
        fullscreen: false,
        headless: false,
        output: None,
        size: (1280, 720),
        time: 4.0,
        frames: None,
        hud: false,
        camera: tracking::Camera::Device("/dev/video0".into()),
        camera_size: (640, 480),
        landmarks: openseeface::config::LANDMARKS_FAST,
        tracker_device: "cpu".into(),
        calibration_distance_m: 0.5,
        profile: Profile {
            display_width_m: 0.344,
            display_height_m: 0.194,
            camera_gap_m: 0.008,
            camera_pitch_deg: 0.0,
            camera_offset_x_m: 0.0,
            camera: headpose::calibration::CameraProfile::Fov { width: 640, height: 480, hfov_deg: 60.0 },
            ipd_m: 0.063,
        },
        profile_path: default_profile_path(),
    };
    let mut profile = None;
    let (mut hfov, mut gap, mut ipd) = (60.0, 0.008, 0.063);
    let mut headless_source_given = false;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match flag.as_str() {
            "--pose-source" => {
                a.source = eye::Source::parse(&value("--pose-source")?)?;
                headless_source_given = true;
            }
            "--eye" => {
                a.source = eye::Source::Fixed(eye::parse_cm(&value("--eye")?)?);
                headless_source_given = true;
            }
            "--camera" => a.camera = tracking::Camera::Device(value("--camera")?),
            "--camera-image" => a.camera = tracking::Camera::Image { path: PathBuf::from(value("--camera-image")?), pan_px: 60 },
            "--camera-size" => a.camera_size = parse_wxh("--camera-size", &value("--camera-size")?)?,
            "--camera-hfov" => hfov = value("--camera-hfov")?.parse().map_err(|e| format!("--camera-hfov: {e}"))?,
            "--camera-gap-cm" => gap = value("--camera-gap-cm")?.parse::<f64>().map_err(|e| format!("--camera-gap-cm: {e}"))? / 100.0,
            "--ipd-mm" => ipd = value("--ipd-mm")?.parse::<f64>().map_err(|e| format!("--ipd-mm: {e}"))? / 1000.0,
            "--landmarks" => {
                a.landmarks = match value("--landmarks")?.as_str() {
                    "fast" => openseeface::config::LANDMARKS_FAST,
                    "best" => openseeface::config::LANDMARKS_BEST,
                    other => return Err(format!("--landmarks must be fast or best, got {other:?}")),
                }
            }
            "--tracker-device" => {
                a.tracker_device = value("--tracker-device")?;
                let d = a.tracker_device.as_str();
                if !(d == "cpu" || d == "gpu" || d.strip_prefix("gpu").is_some_and(|i| i.parse::<u32>().is_ok())) {
                    return Err(format!("--tracker-device must be cpu, gpu or gpu<i>, got {d:?}"));
                }
            }
            "--calibration-distance-cm" => {
                a.calibration_distance_m = value("--calibration-distance-cm")?.parse::<f64>().map_err(|e| format!("--calibration-distance-cm: {e}"))? / 100.0
            }
            "--display-cm" => {
                let v = value("--display-cm")?;
                let (w, h) = v.split_once(',').ok_or_else(|| format!("--display-cm {v:?}: expected W,H"))?;
                a.display_w_m = w.trim().parse::<f64>().map_err(|e| format!("--display-cm: {e}"))? / 100.0;
                a.display_h_m = h.trim().parse::<f64>().map_err(|e| format!("--display-cm: {e}"))? / 100.0;
            }
            "--profile" => profile = Some(PathBuf::from(value("--profile")?)),
            "--fullscreen" => a.fullscreen = true,
            "--headless" => a.headless = true,
            "--output" => a.output = Some(PathBuf::from(value("--output")?)),
            "--size" => a.size = parse_wxh("--size", &value("--size")?)?,
            "--time" => a.time = value("--time")?.parse().map_err(|e| format!("--time: {e}"))?,
            "--frames" => a.frames = Some(value("--frames")?.parse().ok().filter(|n| *n > 0).ok_or("--frames must be a positive count")?),
            "--hud" => a.hud = true,
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown flag {other:?}")),
        }
    }
    let (cw, ch) = a.camera_size;
    a.profile = Profile {
        display_width_m: a.display_w_m,
        display_height_m: a.display_h_m,
        camera_gap_m: gap,
        camera: headpose::calibration::CameraProfile::Fov { width: cw, height: ch, hfov_deg: hfov },
        ipd_m: ipd,
        ..a.profile
    };
    let profile = profile.or_else(|| a.profile_path.exists().then(|| a.profile_path.clone()));
    if let Some(p) = profile {
        eprintln!("fishtank: calibration from {}", p.display());
        let text = std::fs::read_to_string(&p).map_err(|e| format!("--profile {}: {e}", p.display()))?;
        a.profile = serde_json::from_str(&text).map_err(|e| format!("--profile {}: {e}", p.display()))?;
        a.display_w_m = a.profile.display_width_m;
        a.display_h_m = a.profile.display_height_m;
        a.profile_path = p;
    }
    a.profile.calibration().map_err(|e| format!("calibration: {e}"))?;
    Display::flat(a.display_w_m, a.display_h_m).map_err(|e| format!("--display-cm: {e}"))?;
    if a.headless && a.output.is_none() {
        return Err("--headless needs --output PATH".into());
    }
    if a.headless {
        match a.source {
            eye::Source::Webcam if headless_source_given => return Err("--headless renders a fixed or synthetic eye; webcam tracking needs the window".into()),
            eye::Source::Webcam | eye::Source::Manual => a.source = eye::Source::Fixed(eye::REST),
            _ => {}
        }
    }
    Ok(a)
}

/// The clip planes, metres from the eye along the display normal: near
/// enough for an orb a few centimetres in front of the glass, far enough
/// for the tank's back and the sky behind it.
const NEAR: f64 = 0.01;
const FAR: f64 = 50.0;

/// The view through `glass` from `eye`: the whole contract with the renderer.
pub fn view_through<'a>(glass: &Display, eye: eye::Eye, time_s: f32, grid: bool, hud: Option<(&'a [u8], u32, u32)>) -> Result<renderer::View<'a>, String> {
    let p = off_axis(glass, eye, NEAR, FAR).map_err(|e| e.to_string())?;
    Ok(renderer::View {
        view_proj: p.view_projection(),
        inv_view_proj: p.inverse_view_projection(),
        eye: [eye[0] as f32, eye[1] as f32, eye[2] as f32],
        time_s,
        grid,
        hud,
    })
}

fn headless(a: &Args) -> Result<(), String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .map_err(|e| format!("no GPU adapter: {e}"))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("fishtank"), ..Default::default() })).map_err(|e| format!("opening the GPU: {e}"))?;
    let info = adapter.get_info();
    eprintln!("fishtank: {} ({:?})", info.name, info.backend);
    let (w, h) = a.size;
    // the output's aspect decides the glass: its width is the display's,
    // its height follows the pixels, so pixels stay physically square
    let glass = Display::flat(a.display_w_m, a.display_w_m * h as f64 / w as f64).map_err(|e| e.to_string())?;
    let tank = scene::Tank::for_display(a.display_w_m, a.display_h_m);
    let mut r = renderer::Renderer::new(device, queue, wgpu::TextureFormat::Rgba8UnormSrgb, w, h, tank);
    let frames = a.frames.unwrap_or(1);
    let mut last = None;
    for k in 0..frames {
        let t = a.time + (k as f64 - (frames - 1) as f64) / 60.0;
        let eye = match a.source {
            eye::Source::Fixed(e) => e,
            _ => eye::synthetic(t),
        };
        let lines = [
            format!("FISHTANK  {} (headless)", a.source.label()),
            format!("eye   x {:+6.1}  y {:+6.1}  z {:6.1} cm", eye[0] * 100.0, eye[1] * 100.0, eye[2] * 100.0),
            format!("glass {:.1} x {:.1} cm   t {:.2} s", glass.width() * 100.0, glass.height() * 100.0, t),
            format!("gpu   {}", info.name),
        ];
        let hud = a.hud.then(|| hud::render(&lines));
        let v = view_through(&glass, eye, t as f32, true, hud.as_ref().map(|(p, w, h)| (p.as_slice(), *w, *h)))?;
        last = Some((r.render_to_rgb(&v)?, eye));
    }
    let (img, eye) = last.expect("at least one frame");
    let out = a.output.as_ref().expect("checked by parse");
    imaging::codec::save_png(out, &img)?;
    eprintln!("fishtank: {} from eye ({:.1}, {:.1}, {:.1}) cm -> {}", a.source.label(), eye[0] * 100.0, eye[1] * 100.0, eye[2] * 100.0, out.display());
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) if e.is_empty() => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let r = if args.headless { headless(&args) } else { app::run(args) };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fishtank: {e}");
            ExitCode::FAILURE
        }
    }
}
