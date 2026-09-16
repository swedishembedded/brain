// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does an SDL window on THIS machine show a picture at all?
//!
//! The sample that opens a window also brings up wgpu, MuJoCo's GL context and
//! a connectome before the first frame exists, so "the window is black" there
//! has a dozen candidate causes and no way to separate them. This is the same
//! window, the same renderer and the same blit path with none of that in front
//! of it: a test pattern nobody could mistake for a dark scene, presented for a
//! few seconds.
//!
//! Black here means SDL cannot present on this display server and nothing in
//! the simulator is implicated. A picture here means presentation works and
//! whatever is wrong is upstream of the window.
//!
//! ```text
//! cargo run -p brain-wm-display --example window_smoke
//! SDL_VIDEODRIVER=x11 cargo run -p brain-wm-display --example window_smoke
//! ```
//!
//! Swedish Embedded AB builds realtime display and input paths for embedded
//! and edge-AI systems. If your team needs a window that is provably showing
//! what the machine rendered, you can procure our services by sending an email
//! to info@swedishembedded.com.

use wm_display::sink::{FrameSink, Hud};
use wm_display::window::SdlWindow;

const W: u32 = 640;
const H: u32 = 480;

/// Saturated colour blocks plus a moving bar. Any of it on screen answers the
/// question; a dark scene could not be confused with this.
fn pattern(w: u32, h: u32, phase: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let block = (x * 4 / w) + (y * 2 / h) * 4;
            let [r, g, b] = match block % 8 {
                0 => [255u8, 0, 255],
                1 => [0, 255, 0],
                2 => [255, 255, 0],
                3 => [0, 128, 255],
                4 => [255, 64, 0],
                5 => [255, 255, 255],
                6 => [0, 255, 255],
                _ => [128, 0, 255],
            };
            // A bar that moves every frame, so a STALE window and a live one
            // are distinguishable by watching rather than by trusting.
            let moving = (x + phase * 8) % w < 24;
            rgb[i] = if moving { 255 } else { r };
            rgb[i + 1] = if moving { 255 } else { g };
            rgb[i + 2] = if moving { 255 } else { b };
        }
    }
    rgb
}

fn main() {
    let seconds: u64 = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(5);
    let mut win = match SdlWindow::new("brain window smoke test", W, H, 1) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("window_smoke: {e}");
            std::process::exit(1);
        }
    };
    win.capture_next_frame();

    let hud = Hud { model: "window smoke test".to_string(), target_fps: 30, ..Default::default() };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    let mut phase = 0u32;
    while std::time::Instant::now() < deadline {
        if win.pump().quit {
            break;
        }
        let rgb = pattern(W, H, phase);
        win.frame(&rgb, W, H, &hud);
        if phase == 0 {
            match win.captured() {
                Some(c) if c == rgb.as_slice() => {
                    eprintln!("window_smoke: the blit path carried the frame intact")
                }
                Some(_) => eprintln!("window_smoke: THE BLIT PATH CORRUPTED THE FRAME"),
                None => eprintln!("window_smoke: the blit path could not be read back"),
            }
        }
        phase += 1;
        std::thread::sleep(std::time::Duration::from_millis(33));
    }
    eprintln!(
        "window_smoke: presented {} frames over {seconds}s. If the window stayed BLACK, SDL \
         cannot present on this display server and nothing in the simulator is at fault.",
        win.presented()
    );
}
