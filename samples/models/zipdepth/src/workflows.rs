// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `show` and `camera`: the depth of one image, or of a live webcam, in a
//! [`DepthWindow`].
//!
//! Swedish Embedded AB implements camera pipelines that run depth models in
//! real time on its clients' own hardware. If your team needs expertise in
//! embedded vision then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::time::{Duration, Instant};

use gpu_core::Gpu;
use imaging::viz::Bounds;

use crate::depth;
use crate::flags::{size, start_view, take_count, take_flag, take_value};
use crate::window::DepthWindow;

/// What both windows read off their command line.
struct Common {
    npu: bool,
    scale: u32,
    frames: Option<u64>,
    parsed: invoke::args::Parsed,
    weights: String,
}

fn common(call: &invoke::WorkflowCall, rest: &mut Vec<String>, needs_image: bool, default_scale: u32) -> Result<Common, String> {
    let npu = take_flag(rest, "--npu");
    let scale = take_count(rest, "--scale")?.map_or(default_scale, |s| s.max(1) as u32);
    let frames = take_count(rest, "--frames")?;
    let mut spec = zipdepth::caps::infer_spec();
    if !needs_image {
        spec.inputs.clear();
    }
    // `bench` times one-shot inference; a window reports its own frame rate.
    spec.params.retain(|p| p.name != "bench");
    let parsed = invoke::args::check(call.program, &spec, rest).map_err(|_| "see the flag error above".to_string())?;
    let weights = call.assembly.role_path("weights")?;
    Ok(Common { npu, scale, frames, parsed, weights })
}

fn window(c: &Common, title: &str) -> Result<DepthWindow, String> {
    let inv = &c.parsed.invocation;
    let colormap = imaging::viz::parse_colormap(&inv.get_str("colormap").unwrap_or_else(|| "turbo".into()))?;
    let stripes = inv.get_i64("stripes").unwrap_or(5).max(1) as u32;
    Ok(DepthWindow::new(title, c.scale, start_view(inv)?, colormap, stripes, c.frames))
}

pub fn show(call: &invoke::WorkflowCall) -> Result<(), String> {
    let mut rest = call.args.to_vec();
    let c = common(call, &mut rest, true, 2)?;
    let image = c.parsed.inputs.iter().find(|(n, _)| n == "image").map(|(_, p)| p.clone()).ok_or("give the image: --image <img>")?;
    let img = imaging::load(&image)?;
    let (w, h) = (img.w, img.h);
    let hwc: Vec<f32> = img.px.iter().map(|&b| b as f32 / 255.0).collect();
    let cfg = depth::config(&c.weights, c.parsed.invocation.get_i64("input"))?;
    let started = Instant::now();
    let depth = if c.npu {
        depth::npu(&c.weights, &cfg, w, h)?.predict(&hwc, w, h)?
    } else {
        let gpu = Gpu::new(zipdepth::net::PIPELINES);
        depth::engine(&gpu, &c.weights, &cfg)?.predict(&hwc, w, h)
    };
    let ms = started.elapsed().as_secs_f32() * 1000.0;
    eprintln!("depth: {w}x{h}, inference {ms:.1} ms ({})", if c.npu { "npu" } else { "engine" });
    let bounds = Bounds::from_percentiles(&depth, 0.02, 0.98);
    let mut win = window(&c, &format!("zipdepth - {image}"))?;
    loop {
        win.draw(&img.px, &depth, w, h, bounds, &format!("{ms:.0}MS"), 0.0)?;
        if !win.keys() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(16));
    }
}

#[cfg(not(target_os = "linux"))]
pub fn camera(_call: &invoke::WorkflowCall) -> Result<(), String> {
    Err("the live camera reads V4L2, which is Linux only".into())
}

#[cfg(target_os = "linux")]
pub fn camera(call: &invoke::WorkflowCall) -> Result<(), String> {
    use capture::{Device, Frame, FrameSlot};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let mut rest = call.args.to_vec();
    let device_path = take_value(&mut rest, "--device-path")?.unwrap_or_else(|| "/dev/video0".into());
    let (want_w, want_h) = take_value(&mut rest, "--res")?.map_or(Ok((640, 480)), |v| size(&v, "--res"))?;
    let c = common(call, &mut rest, false, 1)?;
    let cfg = depth::config(&c.weights, c.parsed.invocation.get_i64("input"))?;

    // The driver reports the size it accepted; a camera that offers only
    // MJPEG is refused (frames are converted from YUYV).
    let mut device = Device::open(&device_path, want_w, want_h, 4).map_err(|e| format!("opening {device_path}: {e} (the camera must offer YUYV)"))?;
    let (w, h) = (device.width, device.height);
    eprintln!("camera {device_path}: {w}x{h} YUYV");

    // The capture thread overwrites one slot with the latest frame, so the
    // loop below always draws the newest and never builds a backlog.
    let slot = FrameSlot::new();
    let producer = slot.clone();
    let running = Arc::new(AtomicBool::new(true));
    let capturing = running.clone();
    let capture_thread = std::thread::spawn(move || {
        let mut seq = 0u64;
        while capturing.load(Ordering::Relaxed) {
            match device.next_frame(imaging::color::yuyv_to_rgb) {
                Ok((rgb, stamp)) => {
                    seq += 1;
                    producer.push(Frame { rgb, w, h, seq, stamp });
                }
                Err(e) => {
                    eprintln!("capture error: {e}");
                    break;
                }
            }
        }
    });
    let result = (|| {
        let gpu = Gpu::new(zipdepth::net::PIPELINES);
        let mut npu = if c.npu { Some(depth::npu(&c.weights, &cfg, w, h)?) } else { None };
        let engine = if c.npu { None } else { Some(depth::engine(&gpu, &c.weights, &cfg)?) };
        let backend = if c.npu { "NPU" } else { "ENGINE" };
        let mut win = window(&c, "zipdepth camera")?;
        let mut bounds: Option<Bounds> = None;
        let (mut fps, mut last) = (0.0f32, Instant::now());
        // The engine is pipelined: the frame whose depth is computing is kept
        // so it is drawn against its own depth one frame later.
        let mut in_flight: Option<Frame> = None;
        while win.keys() {
            let Some(frame) = slot.take() else {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            };
            let hwc: Vec<f32> = frame.rgb.iter().map(|&b| b as f32 / 255.0).collect();
            let started = Instant::now();
            let (frame, depth) = match (&mut npu, &engine) {
                (Some(session), _) => {
                    let d = session.predict(&hwc, frame.w, frame.h)?;
                    (frame, d)
                }
                (None, Some(p)) => {
                    let done = p.in_flight().then(|| p.finish());
                    p.begin(&hwc, frame.w, frame.h);
                    match (in_flight.replace(frame), done) {
                        (Some(previous), Some(d)) => (previous, d),
                        _ => continue, // the pipeline is filling
                    }
                }
                (None, None) => unreachable!("one backend is built"),
            };
            let infer_ms = started.elapsed().as_secs_f32() * 1000.0;
            // Smooth the window the colors are drawn against, or a scene
            // breathes as one bright pixel comes and goes.
            let target = Bounds::from_percentiles(&depth, 0.02, 0.98);
            let b = bounds.map_or(target, |b| b.ema(target, 0.1));
            bounds = Some(b);
            let now = Instant::now();
            fps = 0.9 * fps + 0.1 / now.duration_since(last).as_secs_f32().max(1e-3);
            last = now;
            let status = format!("{backend} {fps:.0}FPS {infer_ms:.0}MS DROP:{}", slot.stats().dropped);
            win.draw(&frame.rgb, &depth, frame.w, frame.h, b, &status, fps)?;
        }
        Ok(())
    })();
    running.store(false, Ordering::Relaxed);
    let _ = capture_thread.join();
    result
}
