// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The diagnostics overlay: a few lines of the workspace's bitmap font on a
//! translucent panel, rasterized on the CPU and uploaded as a small texture
//! (an upload, never a readback), composited by `post.wgsl`.

use imaging::viz::draw_text;

const SCALE: u32 = 2;
const LINE: u32 = 9 * SCALE + 4;
const PAD: u32 = 8;

/// Rasterize `lines` into RGBA8 (sRGB) with the panel's own alpha.
pub fn render(lines: &[String]) -> (Vec<u8>, u32, u32) {
    let cols = lines.iter().map(|l| l.len()).max().unwrap_or(0) as u32;
    let w = cols * 6 * SCALE + 2 * PAD + 2 * SCALE;
    let h = lines.len() as u32 * LINE + 2 * PAD;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for (i, l) in lines.iter().enumerate() {
        draw_text(&mut rgb, w, h, PAD, PAD + i as u32 * LINE, l, SCALE, [190, 235, 255]);
    }
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for p in rgb.as_chunks::<3>().0 {
        let lit = p.iter().any(|&c| c > 60);
        rgba.extend_from_slice(&[p[0], p[1], p[2], if lit { 255 } else { 150 }]);
    }
    (rgba, w, h)
}
