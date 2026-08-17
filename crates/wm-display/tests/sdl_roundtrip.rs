// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDL pixel-path smoke test: present a known RGB frame through the real SDL
//! texture path (dummy video driver) and read it back — catches pixel-format
//! or pitch bugs that headless sinks can never see (this exact class of bug
//! shipped once: a wrong SDL_PIXELFORMAT_RGB24 constant garbled the window
//! while every headless golden test stayed green).

use wm_display::sink::{FrameSink, Hud};
use wm_display::window::SdlWindow;

#[test]
fn sdl_texture_roundtrip_is_pixel_faithful() {
    std::env::set_var("SDL_VIDEODRIVER", "dummy");
    let (w, h) = (8u32, 8u32);
    let mut win = match SdlWindow::new("roundtrip", w, h, 1) {
        Ok(win) => win,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("SDL unavailable ({e})"));
            return;
        }
    };
    // Distinct per-pixel pattern exercising all three channels.
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
        rgb[i * 3] = (i * 7 % 256) as u8;
        rgb[i * 3 + 1] = (i * 13 % 256) as u8;
        rgb[i * 3 + 2] = (255 - i * 5 % 256) as u8;
    }
    win.frame(&rgb, w, h, &Hud::default());
    let back = win.read_back(w, h).expect("read back");
    assert_eq!(back, rgb, "SDL texture path altered pixels (format/pitch bug)");
}
