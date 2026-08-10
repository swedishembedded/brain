// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream overlap multi-crop bookkeeping and the feature-space
//! reconstruct → adaptive-pool → global‖local channel-concat that forms the
//! connector's `[729, 2·dim]` input. Ports `select_tiling`/`reconstruct_from_crops`
//! (`image_crops.py`) faithfully — reconstruct runs in **patch units**
//! (`patch_size=1`), stitching the ViT's `[n_local, 27, 27, dim]` local feature
//! maps and trimming the 4-patch overlap on interior edges. The pixel-space
//! `overlap_crop_image` (which needs a JPEG/PNG decoder brain still lacks) is a
//! follow-up; this covers everything downstream of the ViT.

use gpu_core::Gpu;

use crate::vision::ADAPTIVE_AVGPOOL2D_ID;

/// Pick `(h_tiles, w_tiles)` with `h·w ≤ max_crops` best matching the image aspect
/// ratio. Faithful port of `image_crops.py::select_tiling` — inputs are the
/// margin-subtracted pixel dims and the usable `crop_window_size`.
pub fn select_tiling(height: u32, width: u32, crop_size: u32, max_crops: u32) -> (u32, u32) {
    if height <= crop_size || width <= crop_size {
        return (1, 1);
    }
    let (h, w, cs, mc) = (height as f64, width as f64, crop_size as f64, max_crops as f64);
    let min_h = (h / cs).ceil();
    let min_w = (w / cs).ceil();
    if min_h * min_w > mc {
        let ratio = (mc / (min_h * min_w)).sqrt();
        return ((min_h * ratio).floor().max(1.0) as u32, (min_w * ratio).floor().max(1.0) as u32);
    }
    let mut h_tiles = (mc * h / w).sqrt().floor().max(min_h);
    let mut w_tiles = (mc * w / h).sqrt().floor().max(min_w);
    if h_tiles * w_tiles > mc {
        if w_tiles > h_tiles {
            w_tiles = (mc / h_tiles).floor();
        } else {
            h_tiles = (mc / w_tiles).floor();
        }
    }
    (h_tiles.max(1.0) as u32, w_tiles.max(1.0) as u32)
}

/// Stitch local feature maps `[n_local, grid, grid, dim]` (row-major, tile order
/// `(tile_y, tile_x)`) into a single channel-first `[dim, out_h, out_w]` map,
/// trimming `margin` patches on each interior edge. Returns `(flat, out_h, out_w)`.
///
/// Mirrors `reconstruct_from_crops` with `patch_size=1`: `out = (grid-2·margin)·tile
/// + 2·margin` per axis; a tile keeps its left/top margin only in the first
/// column/row and its right/bottom margin only in the last.
pub fn reconstruct_from_crops(locals: &[f32], h_tiles: u32, w_tiles: u32, grid: u32, dim: u32, margin: u32) -> (Vec<f32>, u32, u32) {
    let (g, m) = (grid as i64, margin as i64);
    let out_h = ((grid - 2 * margin) * h_tiles + 2 * margin) as usize;
    let out_w = ((grid - 2 * margin) * w_tiles + 2 * margin) as usize;
    let dim = dim as usize;
    assert_eq!(locals.len(), (h_tiles * w_tiles) as usize * grid as usize * grid as usize * dim, "locals must be [n_local, grid, grid, dim]");
    let mut out = vec![0.0f32; dim * out_h * out_w];
    let plane = out_h * out_w;
    let step = (grid - 2 * margin) as i64;
    for ty in 0..h_tiles as i64 {
        for tx in 0..w_tiles as i64 {
            let tile = (ty * w_tiles as i64 + tx) as usize;
            let base = tile * grid as usize * grid as usize * dim;
            let y_start = if ty == 0 { 0 } else { m };
            let y_end = if ty == h_tiles as i64 - 1 { g } else { g - m };
            let x_start = if tx == 0 { 0 } else { m };
            let x_end = if tx == w_tiles as i64 - 1 { g } else { g - m };
            let (out_y0, out_x0) = (ty * step, tx * step);
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let src = base + (y as usize * grid as usize + x as usize) * dim;
                    let oy = (out_y0 + y) as usize;
                    let ox = (out_x0 + x) as usize;
                    let dst = oy * out_w + ox;
                    for c in 0..dim {
                        out[c * plane + dst] = locals[src + c];
                    }
                }
            }
        }
    }
    (out, out_h as u32, out_w as u32)
}

/// Form the connector input `[grid², 2·dim]`: reconstruct the local feature maps,
/// adaptive-avg-pool the `[dim, H', W']` map to `[dim, grid, grid]` on device, then
/// channel-concat the (patch-major) global features with the pooled locals.
///
/// `global`/`locals` are ViT post-LN features (`[grid², dim]` and
/// `[n_local, grid², dim]`). Returns `[grid², 2·dim]` ready for [`Connector`].
///
/// [`Connector`]: crate::vision::Connector
pub fn build_connector_input(gpu: &Gpu, global: &[f32], locals: &[f32], h_tiles: u32, w_tiles: u32, grid: u32, dim: u32, margin: u32) -> Vec<f32> {
    let ppc = (grid * grid) as usize;
    let d = dim as usize;
    assert_eq!(global.len(), ppc * d, "global must be [grid², dim]");
    // Reconstruct locals → channel-first [dim, H', W'].
    let (recon, oh, ow) = reconstruct_from_crops(locals, h_tiles, w_tiles, grid, dim, margin);
    // Device adaptive pool [1, dim, H', W'] → [1, dim, grid, grid] = [dim, ppc].
    let xb = gpu.storage_init("md.recon", &recon);
    let yb = gpu.storage((dim as u64) * (grid as u64) * (grid as u64));
    gpu.submit(&[], &[gpu.step(ADAPTIVE_AVGPOOL2D_ID, &[&xb, &yb], &[1, dim, oh, ow, grid, grid], dim * grid * grid)]);
    let pooled = gpu.read(&yb, d * ppc); // [dim, ppc], channel-first
    // Channel-concat: out[p, 0:dim]=global[p], out[p, dim:2dim]=pooled[:, p].
    let mut out = vec![0.0f32; ppc * 2 * d];
    for p in 0..ppc {
        out[p * 2 * d..p * 2 * d + d].copy_from_slice(&global[p * d..p * d + d]);
        for c in 0..d {
            out[p * 2 * d + d + c] = pooled[c * ppc + p];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::vision_pipelines;

    #[test]
    fn select_tiling_matches_reference() {
        // height/width ≤ crop → single tile.
        assert_eq!(select_tiling(300, 300, 378, 12), (1, 1));
        // Square large image → balanced tiling under budget.
        let (h, w) = select_tiling(1000, 1000, 266, 12);
        assert!(h * w <= 12 && h >= 1 && w >= 1);
        assert_eq!((h, w), (3, 3)); // floor(sqrt(12)) each, then min_h/min_w=ceil(1000/266)=4 lifts→ capped
        // Wide image → more horizontal tiles.
        let (h2, w2) = select_tiling(400, 1600, 266, 12);
        assert!(w2 >= h2 && h2 * w2 <= 12);
    }

    #[test]
    fn reconstruct_dims_and_trim() {
        // 2×2 tiling, grid 4, margin 1, dim 2 → out = (4-2)·2+2 = 6 per axis.
        let (ht, wt, grid, dim, m) = (2u32, 2u32, 4u32, 2u32, 1u32);
        let n = (ht * wt * grid * grid * dim) as usize;
        let locals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let (out, oh, ow) = reconstruct_from_crops(&locals, ht, wt, grid, dim, m);
        assert_eq!((oh, ow), (6, 6));
        assert_eq!(out.len(), (dim * oh * ow) as usize);
        // Top-left tile's (0,0) patch, channel 0, lands at output (0,0).
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn connector_input_shape_and_global_copy() {
        let gpu = gpu_core::testgpu::dev(vision_pipelines());
        let (ht, wt, grid, dim, m) = (2u32, 2u32, 4u32, 3u32, 1u32);
        let ppc = (grid * grid) as usize;
        let d = dim as usize;
        let global: Vec<f32> = (0..ppc * d).map(|i| i as f32 * 0.01).collect();
        let n_local = (ht * wt) as usize;
        let locals: Vec<f32> = (0..n_local * ppc * d).map(|i| (i % 7) as f32 * 0.1).collect();
        let out = build_connector_input(&gpu, &global, &locals, ht, wt, grid, dim, m);
        assert_eq!(out.len(), ppc * 2 * d);
        // Global half is copied verbatim into the first `dim` channels of each patch.
        for p in 0..ppc {
            for c in 0..d {
                assert_eq!(out[p * 2 * d + c], global[p * d + c]);
            }
        }
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
