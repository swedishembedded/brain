// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YUYV 4:2:2 -> interleaved RGB8.
//!
//! YUYV (a.k.a. YUY2) packs two pixels in four bytes: `Y0 U Y1 V`, so the two
//! pixels share one chroma pair. We force this format at negotiation (MJPEG would
//! need a JPEG decoder brain does not have), so this is the only conversion the
//! demo runs. The BT.601 full-range coefficients match what UVC webcams emit.

/// Convert a `YUYV` buffer (`w*h*2` bytes) to interleaved `RGB8` (`w*h*3` bytes).
///
/// `w` must be even (YUYV pairs pixels horizontally); an odd width is a caller
/// bug, not something to paper over, so it panics.
pub fn yuyv_to_rgb(yuyv: &[u8], w: u32, h: u32) -> Vec<u8> {
    assert_eq!(w % 2, 0, "YUYV width must be even (pixels are paired), got {w}");
    assert_eq!(yuyv.len(), (w * h * 2) as usize, "YUYV buffer must be w*h*2 bytes");
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut si = 0usize;
    let mut di = 0usize;
    let npairs = (w * h / 2) as usize;
    for _ in 0..npairs {
        let y0 = yuyv[si] as f32;
        let u = yuyv[si + 1] as f32 - 128.0;
        let y1 = yuyv[si + 2] as f32;
        let v = yuyv[si + 3] as f32 - 128.0;
        si += 4;
        // BT.601. Same u/v for both pixels (4:2:2 shares chroma).
        let r_off = 1.402 * v;
        let g_off = -0.344136 * u - 0.714136 * v;
        let b_off = 1.772 * u;
        for y in [y0, y1] {
            rgb[di] = clamp8(y + r_off);
            rgb[di + 1] = clamp8(y + g_off);
            rgb[di + 2] = clamp8(y + b_off);
            di += 3;
        }
    }
    rgb
}

fn clamp8(v: f32) -> u8 {
    v.clamp(0.0, 255.0).round() as u8
}
