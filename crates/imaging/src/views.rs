// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements document-OCR pipelines that read pages the way
// their models were trained to. If your team needs expertise in reproducing a
// vision model's reference preprocessing exactly then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The views a document-OCR model is fed, prepared the way its reference code
//! prepares them (Pillow on 8-bit RGB), byte for byte:
//!
//! * [`pad_square`] is `ImageOps.pad(image, (side, side), BICUBIC, color)`: an
//!   aspect-preserving antialiased bicubic fit, centred on a flat-colour square
//!   (the *global view*);
//! * [`mosaic_tiles`] is the *local view*: the image resized (BICUBIC) to a
//!   `cols x rows` mosaic of `tile`-pixel squares and cut into them row-major;
//! * [`to_chw_half`] is `ToTensor` + `Normalize(0.5, 0.5)`: planar f32 in `[-1, 1]`.
//!
//! The tile grid itself is a policy, not geometry: see [`crate::tiling::internvl_grid`].

use crate::host::resize_bicubic_pil;
use crate::pixels::Rgb8;

/// The local tile grid `(cols, rows)` the DeepSeek-OCR family cuts an image into, or
/// `None` when the image fits one tile (both sides at most `tile`) and is read as a
/// single view. `max_tiles` is 9 for DeepSeek-OCR (640 px tiles) and 6 for
/// DeepSeek-OCR-2 (768 px tiles); the grid is [`crate::tiling::internvl_grid`] with at least two tiles.
pub fn tile_grid(w: u32, h: u32, tile: u32, max_tiles: u32) -> Option<(u32, u32)> {
    if w <= tile && h <= tile {
        return None;
    }
    Some(crate::tiling::internvl_grid(w, h, tile, 2, max_tiles))
}

/// Python's `round()`: nearest integer, ties to even (not `f64::round`'s ties away from zero).
fn py_round(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r as i64 % 2 != 0 {
        r as i64 - x.signum() as i64
    } else {
        r as i64
    }
}

/// The extent `ImageOps.contain` fits a `w x h` image to inside a `side x side` square.
pub fn contain_extent(w: u32, h: u32, side: u32) -> (u32, u32) {
    assert!(w > 0 && h > 0 && side > 0, "contain_extent: {w}x{h} into {side}");
    let im_ratio = w as f64 / h as f64;
    if im_ratio == 1.0 {
        return (side, side);
    }
    if im_ratio > 1.0 {
        (side, py_round(h as f64 / w as f64 * side as f64).max(1) as u32)
    } else {
        (py_round(w as f64 / h as f64 * side as f64).max(1) as u32, side)
    }
}

/// `ImageOps.pad(img, (side, side), color=(fill, fill, fill))` with Pillow's default
/// bicubic and `centering = (0.5, 0.5)`.
pub fn pad_square(img: &Rgb8, side: u32, fill: u8) -> Rgb8 {
    let (fw, fh) = contain_extent(img.w, img.h, side);
    let resized = if (fw, fh) == (img.w, img.h) { img.clone() } else { resize_bicubic_pil(img, fw as usize, fh as usize) };
    if (fw, fh) == (side, side) {
        return resized;
    }
    let mut out = Rgb8 { w: side, h: side, px: vec![fill; side as usize * side as usize * 3] };
    // Pillow offsets only the axis that is short, by `round(gap * 0.5)`.
    let (x0, y0) = if fw != side { (py_round((side - fw) as f64 * 0.5) as u32, 0) } else { (0, py_round((side - fh) as f64 * 0.5) as u32) };
    for y in 0..fh as usize {
        let dst = ((y0 as usize + y) * side as usize + x0 as usize) * 3;
        out.px[dst..dst + fw as usize * 3].copy_from_slice(&resized.px[y * fw as usize * 3..][..fw as usize * 3]);
    }
    out
}

/// The `cols x rows` mosaic of `tile`-pixel squares, row-major (`index = row * cols + col`).
pub fn mosaic_tiles(img: &Rgb8, tile: u32, cols: u32, rows: u32) -> Vec<Rgb8> {
    assert!(tile > 0 && cols > 0 && rows > 0, "mosaic_tiles: {cols}x{rows} tiles of {tile}");
    let (mw, mh) = (tile * cols, tile * rows);
    let resized = if (img.w, img.h) == (mw, mh) { img.clone() } else { resize_bicubic_pil(img, mw as usize, mh as usize) };
    let t = tile as usize;
    let mut out = Vec::with_capacity((cols * rows) as usize);
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            let mut px = Vec::with_capacity(t * t * 3);
            for y in 0..t {
                let at = ((r * t + y) * mw as usize + c * t) * 3;
                px.extend_from_slice(&resized.px[at..at + t * 3]);
            }
            out.push(Rgb8 { w: tile, h: tile, px });
        }
    }
    out
}

/// Planar `[3, h, w]` f32 in `[-1, 1]`: `(x / 255 - 0.5) / 0.5`.
pub fn to_chw_half(img: &Rgb8) -> Vec<f32> {
    let (w, h) = (img.w as usize, img.h as usize);
    let mut out = vec![0f32; 3 * w * h];
    // One row of one plane per task; rows are independent.
    backend_cpu::par::chunks_mut(&mut out, w, |r, row| {
        let (c, y) = (r / h, r % h);
        for (x, v) in row.iter_mut().enumerate() {
            *v = (img.px[(y * w + x) * 3 + c] as f32 / 255.0 - 0.5) / 0.5;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic test image: a smooth field plus LCG noise, so a resampler
    /// that is wrong anywhere changes a byte.
    fn image(w: u32, h: u32) -> Rgb8 {
        let mut s = 12345u32;
        let px = (0..w * h * 3)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let (x, y) = ((i / 3) % w, (i / 3) / w);
                ((x * 255 / w + y * 255 / h + (i % 3) * 40) / 2 + (s >> 28)) as u8
            })
            .collect();
        Rgb8 { w, h, px }
    }

    /// FNV-1a 64 of the bytes; `tools/goldens/ocr_views_pil.py` computes the
    /// same function over Pillow's output.
    fn fnv(b: &[u8]) -> u64 {
        b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
    }

    #[test]
    fn small_images_are_one_view_and_larger_ones_get_a_grid_of_at_least_two_tiles() {
        assert_eq!(tile_grid(640, 500, 640, 9), None);
        assert_eq!(tile_grid(641, 100, 640, 9).map(|(c, r)| c * r >= 2), Some(true));
        // A portrait A4 scan at 150 dpi: 2 columns x 3 rows of tiles.
        assert_eq!(tile_grid(1240, 1754, 640, 9), Some((2, 3)));
        assert_eq!(tile_grid(2480, 3508, 768, 6), Some((2, 3)), "the tile cap bounds the grid");
    }

    #[test]
    fn python_round_ties_go_to_even() {
        assert_eq!([0.5, 1.5, 2.5, 3.5, -0.5, 2.4, 2.6].map(py_round), [0, 2, 2, 4, 0, 2, 3]);
    }

    /// Pillow's `contain` extents, including the tie cases.
    #[test]
    fn contain_extents_follow_pillow() {
        assert_eq!(contain_extent(2480, 1753, 1024), (1024, 724));
        assert_eq!(contain_extent(100, 300, 640), (213, 640));
        assert_eq!(contain_extent(500, 500, 64), (64, 64));
        assert_eq!(contain_extent(3, 1, 4), (4, 1), "1/3 * 4 = 1.33 rounds to 1");
    }

    /// Pillow's `ImageOps.pad` and a Pillow mosaic, byte for byte (hashes from
    /// `tools/goldens/ocr_views_pil.py`, Pillow 13).
    #[test]
    fn views_match_pillow_byte_for_byte() {
        let landscape = image(173, 101);
        assert_eq!(fnv(&pad_square(&landscape, 64, 127).px), PAD_LANDSCAPE, "pad 173x101 -> 64");
        // Pillow resamples vertically first when that is cheaper, and the order changes a pixel by up to
        // four levels (rounding to 8 bits between the passes); the portrait's geometry is what is pinned.
        let portrait = pad_square(&image(61, 150), 64, 127);
        let content: Vec<usize> = (0..64).filter(|&x| (0..64).any(|y| portrait.px[(y * 64 + x) * 3..][..3] != [127, 127, 127])).collect();
        assert_eq!((content[0], *content.last().unwrap()), (19, 44), "pad 61x150 -> 64 puts 26 columns at x = 19");
        let tiles = mosaic_tiles(&image(150, 90), 32, 3, 2);
        assert_eq!(tiles.len(), 6);
        let h: Vec<u64> = tiles.iter().map(|t| fnv(&t.px)).collect();
        assert_eq!(h, MOSAIC, "3x2 mosaic of 32 px tiles from 150x90");
    }

    #[test]
    fn normalization_is_the_half_affine() {
        let img = Rgb8 { w: 1, h: 2, px: vec![0, 255, 51, 255, 0, 255] };
        let v = to_chw_half(&img);
        // Planar: the R plane (pixels 0 and 1) first, then G, then B.
        assert_eq!(v[..2], [-1.0, 1.0]);
        assert_eq!(v[2..4], [1.0, -1.0], "G plane: 255 then 0");
        assert!((v[4] - (51.0 / 255.0 - 0.5) / 0.5).abs() < 1e-7);
        assert_eq!(v[5], 1.0);
    }

    const PAD_LANDSCAPE: u64 = 4845578648140569513;
    const MOSAIC: [u64; 6] = [4476174800052333252, 4506216825006873800, 12229689418379448210, 8901965889726546321, 13922864024796162883, 2566294066096634774];
}
