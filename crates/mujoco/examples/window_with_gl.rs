// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A headless GL context and an SDL window in one process: the minimal
//! reproducer for a window that shows nothing.
//!
//! Offscreen rendering needs an OpenGL context that belongs to no window, and
//! a realtime display needs a window that belongs to no OpenGL context. Both
//! live in the same process here, and the order they are built in, and which
//! of them holds the thread's current context, decide whether anything is ever
//! drawn. Nothing about that is visible from inside either one: every call
//! returns success and the window stays black.
//!
//! So this is the control. The window, the renderer and the blit are identical
//! in both runs; the only difference is whether a GL context exists.
//!
//! ```text
//! cargo run -p brain-mujoco --example window_with_gl -- --no-gl   # control
//! cargo run -p brain-mujoco --example window_with_gl -- --gl      # bare context
//! cargo run -p brain-mujoco --example window_with_gl -- --model body.xml
//! ```
//!
//! A picture in both is the expected outcome. A picture in the first and a
//! black window in the second localises the fault to the coexistence of the
//! two graphics stacks and rules out the scene, the blit and the display
//! server, none of which differ between the runs.
//!
//! Swedish Embedded AB builds offscreen-rendering and realtime display
//! pipelines that have to share one machine with a compute workload. If your
//! team needs OpenGL, EGL and a window toolkit to coexist on an embedded or
//! edge-AI target, you can procure our services by emailing
//! info@swedishembedded.com.

use wm_display::sink::{FrameSink, Hud};
use wm_display::window::SdlWindow;

fn pattern(w: u32, h: u32, phase: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for (i, px) in rgb.chunks_exact_mut(3).enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        let on = ((x + phase * 8) / 40 + y / 40).is_multiple_of(2);
        px[0] = if on { 255 } else { 0 };
        px[1] = if on { 0 } else { 255 };
        px[2] = 128;
    }
    rgb
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let with_gl = !args.iter().any(|a| a == "--no-gl");
    let seconds: u64 = args.iter().find_map(|a| a.strip_prefix("--seconds=")?.parse().ok()).unwrap_or(5);
    // Size is a variable, not a constant: the display path is not
    // size-independent and testing one size proves one size.
    let w: u32 = args.iter().find_map(|a| a.strip_prefix("--width=")?.parse().ok()).unwrap_or(640);
    let h: u32 = args.iter().find_map(|a| a.strip_prefix("--height=")?.parse().ok()).unwrap_or(480);

    let model_path = args.iter().find_map(|a| a.strip_prefix("--model=").map(String::from));

    // Held to the end of main: dropping the renderer would tear the GL
    // context's resources down and the window would then be tested without
    // the thing under test.
    let _renderer = match (&model_path, with_gl) {
        (Some(path), _) => {
            // The full offscreen renderer: an EGL context AND MuJoCo's own GL
            // objects on top of it (`mjr_makeContext`), which is a great deal
            // more GL state than a bare context.
            let mj = mujoco::MuJoCo::load().unwrap_or_else(|e| { eprintln!("window_with_gl: {e}"); std::process::exit(1) });
            let model = mujoco::Model::from_xml(&mj, path).unwrap_or_else(|e| { eprintln!("window_with_gl: {e}"); std::process::exit(1) });
            let r = mujoco::Renderer::for_model(&mj, &model, w, h).unwrap_or_else(|e| { eprintln!("window_with_gl: {e}"); std::process::exit(1) });
            eprintln!("window_with_gl: FULL MuJoCo renderer created BEFORE the window");
            Some(r)
        }
        (None, true) => {
            match mujoco::EglContext::get() {
                Ok(_) => eprintln!("window_with_gl: bare GL context created BEFORE the window"),
                Err(e) => {
                    eprintln!("window_with_gl: no GL context available ({e}); this run proves nothing");
                    std::process::exit(1);
                }
            }
            None
        }
        (None, false) => {
            eprintln!("window_with_gl: no GL context (control)");
            None
        }
    };

    let mut win = match SdlWindow::new("brain window with gl", w, h, 1) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("window_with_gl: {e}");
            std::process::exit(1);
        }
    };
    let hud = Hud { model: "window_with_gl".to_string(), target_fps: 30, ..Default::default() };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    let mut phase = 0u32;
    while std::time::Instant::now() < deadline {
        if win.pump().quit {
            break;
        }
        win.frame(&pattern(w, h, phase), w, h, &hud);
        phase += 1;
        std::thread::sleep(std::time::Duration::from_millis(33));
    }
    eprintln!(
        "window_with_gl: presented {} frames ({})",
        win.presented(),
        match (&model_path, with_gl) {
            (Some(p), _) => format!("full MuJoCo renderer, {p}"),
            (None, true) => "bare GL context".to_string(),
            (None, false) => "no GL context".to_string(),
        }
    );
}
