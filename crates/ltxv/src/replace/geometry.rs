// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Silhouette geometry for `replace`: boxes around the actor, the canvas an
//! edit is painted on, carrying a figure along the actor's track, and the
//! grow/shrink/feather/blend of masks on the device.

use imaging::{AlignCorners, Ctx, Filter, Rgb8, Shape};

pub(super) fn count(m: &[bool]) -> usize {
    m.iter().filter(|&&v| v).count()
}

/// An axis-aligned pixel box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Box2 {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

pub(super) fn bounding_box(m: &[bool], w: usize, h: usize) -> Option<Box2> {
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for (i, _) in m.iter().enumerate().filter(|(_, &v)| v) {
        let (x, y) = (i % w, i / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
    }
    // Lazy: an empty mask leaves y0 at `h` and y1 at 0, and `then_some`
    // would still evaluate the subtraction that overflows.
    (x1 > x0).then(|| Box2 { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
}

/// The part of a still that carries who the character is rather than where
/// they are: the top of the actor's silhouette (head, hair and the shoulders
/// of their clothes), tightened to the mask in that band and with a tenth of
/// its size as margin, inside the frame. Every guide still is painted with
/// this beside the character sheet, so the face, hair and clothes are the
/// frame's own from the first still to the last. The band stays near the head
/// ON PURPOSE: a crop that reaches down the body carries the first frame's
/// pose with it, and the guide then takes that pose instead of its source
/// frame's, which moves its figure off the actor. An actor who spans
/// the whole frame gives the whole frame, which is still an appearance
/// reference.
pub(super) fn appearance_box(actor: &[bool], w: usize, h: usize) -> Option<Box2> {
    let b = bounding_box(actor, w, h)?;
    let band_end = (b.y + ((b.h as f32 * 0.35).round() as usize).max(1)).min(h);
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in b.y..band_end {
        for (dx, &on) in actor[y * w + b.x..y * w + b.x + b.w].iter().enumerate() {
            if on {
                let x = b.x + dx;
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            }
        }
    }
    if x1 <= x0 {
        return None;
    }
    let margin = (x1 - x0).max(y1 - y0) / 10;
    let (x0, y0) = (x0.saturating_sub(margin), y0.saturating_sub(margin));
    let (x1, y1) = ((x1 + margin).min(w), (y1 + margin).min(h));
    Some(Box2 { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
}

/// How many rows of body the bottom edge of the frame cuts off `actor`
/// (boxed by `b`): a standing person's silhouette is about `ratio` times as
/// tall as its median row width, so what that height exceeds the visible
/// one by is below the frame - at most one more actor height, and nothing
/// when the actor does not reach the edge or stands whole on it.
pub(super) fn body_below_frame(actor: &[bool], b: Box2, w: usize, h: usize, ratio: f32) -> usize {
    if b.y + b.h < h {
        return 0;
    }
    let mut widths: Vec<usize> = (b.y..b.y + b.h).map(|y| count(&actor[y * w..(y + 1) * w])).collect();
    widths.sort_unstable();
    let expected = (ratio * widths[widths.len() / 2] as f32).round() as usize;
    expected.saturating_sub(b.h).min(b.h)
}

/// The canvas an edit works on: the actor's box with an eighth of its height
/// of margin above and below, at least 3:4 wide so a portrait figure has
/// room, inside the frame sideways. An actor the bottom edge cuts off gets
/// `below` more rows of canvas BELOW the frame ([`body_below_frame`]), room
/// for the rest of a body at the actor's own scale; those rows are only
/// ever repainted.
pub(super) fn canvas_around(actor: Box2, below: usize, w: usize, h: usize) -> Box2 {
    let margin = actor.h / 8;
    let y = actor.y.saturating_sub(margin);
    let cut_off = below > 0;
    let ch = if cut_off { h - y + below } else { (actor.h + 2 * margin).min(h) };
    let y = if cut_off { y } else { y.min(h - ch) };
    let cw = (actor.w + actor.w / 5).max((ch * 3).div_ceil(4)).min(w);
    Box2 { x: (actor.x + actor.w / 2).saturating_sub(cw / 2).min(w - cw), y, w: cw, h: ch }
}

/// The edit's render size: the canvas's aspect with the long side at
/// `long_side`, both sides multiples of 16 (FLUX.2's patch grid).
pub(super) fn edit_size(cw: usize, ch: usize, long_side: u32) -> (u32, u32) {
    let long = cw.max(ch) as f32;
    let side = |s: usize| ((s as f32 * long_side as f32 / long / 16.0).round().max(1.0) * 16.0) as u32;
    (side(cw), side(ch))
}

/// The pixels of `b`; rows below the `h`-row frame are mid grey.
pub(super) fn cut(px: &[u8], w: usize, h: usize, b: Box2) -> Vec<u8> {
    (b.y..b.y + b.h).flat_map(|y| if y < h { px[(y * w + b.x) * 3..(y * w + b.x + b.w) * 3].to_vec() } else { vec![128; b.w * 3] }).collect()
}

pub(super) fn paste(px: &mut [u8], w: usize, b: Box2, tile: &[u8]) {
    for (row, y) in (b.y..b.y + b.h).enumerate() {
        px[(y * w + b.x) * 3..(y * w + b.x + b.w) * 3].copy_from_slice(&tile[row * b.w * 3..(row + 1) * b.w * 3]);
    }
}

/// Ten clicks spread over a silhouette: two per row at five heights, at three tenths and
/// seven tenths of the way across the row, so head, torso and legs are all asked about.
pub(super) fn spread_points(m: &[bool], w: usize, h: usize) -> Vec<(f32, f32)> {
    let rows: Vec<usize> = (0..h).filter(|&y| m[y * w..(y + 1) * w].contains(&true)).collect();
    if rows.is_empty() {
        return Vec::new();
    }
    let mut points = Vec::new();
    for q in [0.15, 0.35, 0.55, 0.75, 0.9] {
        let y = rows[((rows.len() - 1) as f32 * q).round() as usize];
        let row = &m[y * w..(y + 1) * w];
        let (x0, x1) = (row.iter().position(|&v| v).unwrap_or(0), row.iter().rposition(|&v| v).unwrap_or(0));
        for f in [0.3, 0.7] {
            points.push((x0 as f32 + f * (x1 - x0) as f32, y as f32));
        }
    }
    points
}

/// Carry `figure`, drawn where the actor stood as `from`, to where the actor
/// stands as `to`: shifted with the actor's centroid and scaled about it by
/// the square root of the actor's change in area. An actor not visible on
/// either frame carries nothing.
pub(super) fn carry(figure: &[bool], from: &[bool], to: &[bool], w: usize, h: usize) -> Vec<bool> {
    let (Some((fx, fy, fa)), Some((tx, ty, ta))) = (centroid(from, w), centroid(to, w)) else {
        return vec![false; w * h];
    };
    let s = (ta / fa).sqrt();
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            // Inverse map: where on `figure` this pixel comes from.
            let (sx, sy) = (fx + (x as f32 - tx) / s, fy + (y as f32 - ty) / s);
            let (ix, iy) = (sx.round() as isize, sy.round() as isize);
            if (0..w as isize).contains(&ix) && (0..h as isize).contains(&iy) {
                out[y * w + x] = figure[iy as usize * w + ix as usize];
            }
        }
    }
    out
}

pub(super) fn centroid(m: &[bool], w: usize) -> Option<(f32, f32, f32)> {
    let (mut sx, mut sy, mut n) = (0f64, 0f64, 0f64);
    for (i, _) in m.iter().enumerate().filter(|(_, &v)| v) {
        sx += (i % w) as f64;
        sy += (i / w) as f64;
        n += 1.0;
    }
    (n > 0.0).then(|| ((sx / n) as f32, (sy / n) as f32, n as f32))
}

pub(super) fn to_plane(ctx: &Ctx, m: &[bool]) -> gpu_core::DeviceBuffer {
    ctx.upload("ltxv.replace.mask", &m.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect::<Vec<f32>>())
}

pub(super) fn shrink(ctx: &Ctx, m: &[bool], w: usize, h: usize, radius: u32) -> Vec<bool> {
    let s = Shape::new(1, 1, h as u32, w as u32);
    let shrunk = imaging::mask::erode(ctx, &to_plane(ctx, m), s, radius);
    ctx.download(&shrunk, s.numel()).iter().map(|&v| v > 0.5).collect()
}

pub(super) fn grow(ctx: &Ctx, m: &[bool], w: usize, h: usize, radius: u32) -> Vec<bool> {
    let s = Shape::new(1, 1, h as u32, w as u32);
    let grown = imaging::mask::dilate(ctx, &to_plane(ctx, m), s, radius);
    ctx.download(&grown, s.numel()).iter().map(|&v| v > 0.5).collect()
}

/// A `[0, 1]` alpha: `m` with its edge ramped over `radius` pixels.
pub(super) fn feather(ctx: &Ctx, m: &[bool], w: usize, h: usize, radius: u32) -> Vec<f32> {
    let s = Shape::new(1, 1, h as u32, w as u32);
    let soft = imaging::mask::feather(ctx, &to_plane(ctx, m), s, radius);
    ctx.download(&soft, s.numel())
}

/// Resample interleaved RGB8 `px` (`w x h`) to `tw x th`.
pub(super) fn resample(ctx: &Ctx, px: &[u8], w: usize, h: usize, tw: usize, th: usize, filter: Filter) -> Vec<u8> {
    let unit: Vec<f32> = px.iter().map(|&v| v as f32 / 255.0).collect();
    let chw = imaging::pixels::hwc_to_chw(&unit, 3, h, w);
    let (out, os) = ctx.resize(&ctx.upload("ltxv.replace.canvas", &chw), Shape::new(1, 3, h as u32, w as u32), th as u32, tw as u32, filter, AlignCorners::HalfPixel);
    interleave(&ctx.download(&out, os.numel()), tw * th)
}

/// Planar `[3, hw]` values in `[0, 1]` to interleaved RGB8.
pub(super) fn interleave(plane: &[f32], hw: usize) -> Vec<u8> {
    (0..hw).flat_map(|i| (0..3).map(move |c| i + c * hw)).map(|j| (plane[j] * 255.0).round().clamp(0.0, 255.0) as u8).collect()
}

/// `img` at `tw x th`: area-averaged when it shrinks, so the downscale does
/// not alias, bilinear when it grows (a canvas taller than the edit's render).
pub(super) fn fit(ctx: &Ctx, img: &Rgb8, tw: usize, th: usize) -> Vec<u8> {
    let (w, h) = (img.w as usize, img.h as usize);
    if tw > w || th > h {
        return resample(ctx, &img.px, w, h, tw, th, Filter::Bilinear);
    }
    let s = Shape::new(1, 3, img.h, img.w);
    let chw = imaging::pixels::hwc_to_chw(&img.to_hwc_unit(), 3, h, w);
    let (small, ss) = imaging::mask::downsample(ctx, &ctx.upload("ltxv.replace.edit", &chw), s, th as u32, tw as u32);
    interleave(&ctx.download(&small, ss.numel()), tw * th)
}

/// `alpha * over + (1 - alpha) * under`, per pixel.
pub(super) fn blend(over: &[u8], under: &[u8], alpha: &[f32]) -> Vec<u8> {
    over.as_chunks::<3>()
        .0
        .iter()
        .zip(under.as_chunks::<3>().0)
        .zip(alpha)
        .flat_map(|((o, u), &a)| (0..3).map(move |c| (a * o[c] as f32 + (1.0 - a) * u[c] as f32).round() as u8))
        .collect()
}
