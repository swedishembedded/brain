// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

use depth::viz::{colorize, draw_text, Bounds, Colormap};

#[test]
fn hud_is_legible_on_a_colored_depth_background() {
    // A turbo-colored gradient background (like the depth half of the demo).
    let (w, h) = (520u32, 80u32);
    let depth: Vec<f32> = (0..(w * h)).map(|i| (i % w) as f32 / w as f32).collect();
    let mut rgb = colorize(&depth, Bounds { lo: 0.0, hi: 1.0 }, Colormap::Turbo);
    draw_text(&mut rgb, w, h, 6, 6, "ZIPDEPTH NPU 28FPS 34MS DROP:1", 2, [0, 255, 0]);
    if let Ok(out) = std::env::var("FONT_OUT") {
        imaging::save_ppm(&out, &imaging::Rgb8::new(w, h, rgb.clone()).unwrap()).unwrap();
    }
    let green = rgb.chunks(3).filter(|p| p[1] > 200 && p[0] < 50 && p[2] < 50).count();
    assert!(green > 100, "glyphs must be drawn, got {green} green px");
}
