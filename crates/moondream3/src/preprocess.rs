// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream overlap multi-crop bookkeeping and the feature-space
//! reconstruct → adaptive-pool → global‖local channel-concat that forms the
//! connector's `[729, 2·dim]` input. Ports `select_tiling`/`reconstruct_from_crops`
//! (`image_crops.py`) faithfully - reconstruct runs in **patch units**
//! (`patch_size=1`), stitching the ViT's `[n_local, 27, 27, dim]` local feature
//! maps and trimming the 4-patch overlap on interior edges.
//!
//! The pixel-space half ([`overlap_crop_image`]) is here too. An older note
//! said it was blocked on "a JPEG/PNG decoder brain still lacks"; that is no
//! longer true (`crates/imaging` has codecs, and a served request hands over
//! already-decoded HWC pixels anyway), so what remained was the geometry.

use gpu_core::Gpu;

use crate::config::VisionConfig;
use crate::vision::ADAPTIVE_AVGPOOL2D_ID;

/// Pick `(h_tiles, w_tiles)` with `h·w ≤ max_crops` best matching the image
/// aspect ratio. Faithful port of `image_crops.py::select_tiling` - inputs are
/// the margin-subtracted pixel dims and the usable `crop_window_size`.
///
/// **The definition now lives in `imaging::tiling`**, beside the other models'
/// named crop policies (InternVL/DeepSeek-OCR's discrete `internvl_grid`), per
/// that crate's own stated convention: the policies sit side by side, named per
/// reference model, and are never unified. This is a re-export, not a second
/// copy.
pub use imaging::tiling::moondream_select_tiling as select_tiling;

/// Stitch local feature maps `[n_local, grid, grid, dim]` (row-major, tile order
/// `(tile_y, tile_x)`) into a single channel-first `[dim, out_h, out_w]` map,
/// trimming `margin` patches on each interior edge. Returns `(flat, out_h, out_w)`.
///
/// Mirrors `reconstruct_from_crops` with `patch_size=1`:
/// `out = (grid-2·margin)·tile + 2·margin` per axis; a tile keeps its left/top
/// margin only in the first column/row and its right/bottom margin only in the
/// last.
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

    /// The policy itself is gated in `imaging::tiling` (where it now lives);
    /// this pins that Moondream still reaches the SAME function through its own
    /// name, so the re-export cannot silently point somewhere else.
    #[test]
    fn select_tiling_reexports_the_moondream_policy() {
        assert_eq!(select_tiling(300, 300, 378, 12), imaging::tiling::moondream_select_tiling(300, 300, 378, 12));
        assert_eq!(select_tiling(1000, 1000, 266, 12), (3, 3));
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


/// Where the local crops sit in the resized image.
///
/// # The geometry is DERIVED from [`reconstruct_from_crops`], not guessed
///
/// No dumped reference for the pixel-space crop exists in this workspace, so
/// this is not checked against one. What it IS checked against is the
/// feature-space stitch above, which was ported faithfully and states the same
/// layout in patch units: `out = (grid - 2·margin)·tiles + 2·margin`. Reading
/// that as the inverse of the crop layout gives a stride of `grid - 2·margin`
/// patches between crop origins, a crop of `grid` patches, and a resized image
/// of exactly `stride·tiles + 2·margin` patches - which is what this computes,
/// in pixels, by multiplying through by `patch`.
///
/// [`crop_plan_round_trips_the_feature_space_geometry`] pins that correspondence
/// so the two halves cannot drift apart.
///
/// [`crop_plan_round_trips_the_feature_space_geometry`]: self::tests
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropPlan {
    pub h_tiles: u32,
    pub w_tiles: u32,
    /// The size the source image is resized to before cropping.
    pub resized_h: u32,
    pub resized_w: u32,
    /// Crop side in pixels (`crop_size`), and the distance between crop origins.
    pub crop_px: u32,
    pub stride_px: u32,
}

/// Choose the tiling and the resize target for an image of `(img_h, img_w)`.
pub fn plan_crops(cfg: &VisionConfig, img_h: u32, img_w: u32) -> CropPlan {
    let (grid, patch, margin) = (cfg.grid(), cfg.patch, cfg.overlap_margin);
    let stride_patches = grid.saturating_sub(2 * margin).max(1);
    let stride_px = stride_patches * patch;
    let margin_px = margin * patch;
    // `select_tiling` takes the MARGIN-SUBTRACTED dims against the usable crop
    // window, matching the reference's own call.
    let (h_tiles, w_tiles) = select_tiling(
        img_h.saturating_sub(2 * margin_px).max(1),
        img_w.saturating_sub(2 * margin_px).max(1),
        stride_px,
        cfg.max_crops,
    );
    CropPlan {
        h_tiles,
        w_tiles,
        resized_h: stride_px * h_tiles + 2 * margin_px,
        resized_w: stride_px * w_tiles + 2 * margin_px,
        crop_px: cfg.crop_size,
        stride_px,
    }
}

/// Flatten one `crop_px`-square HWC crop into the `[patches_per_crop,
/// patch_vec]` patch-major layout [`crate::vision::SiglipEncoder::encode`] takes.
///
/// Patch `(py, px)` occupies row `py·grid + px`, and within a row the values run
/// `(y, x, channel)` over the patch - the same order `patch_emb.weight`'s
/// `[dim, 3·patch²]` columns are in.
pub fn patchify_crop(crop_hwc: &[f32], side: u32, patch: u32) -> Vec<f32> {
    let grid = side / patch;
    let (side, patch) = (side as usize, patch as usize);
    let mut out = vec![0.0f32; (grid * grid) as usize * 3 * patch * patch];
    let pv = 3 * patch * patch;
    for py in 0..grid as usize {
        for px in 0..grid as usize {
            let row = (py * grid as usize + px) * pv;
            for y in 0..patch {
                for x in 0..patch {
                    let src = (((py * patch + y) * side) + (px * patch + x)) * 3;
                    let dst = row + (y * patch + x) * 3;
                    out[dst..dst + 3].copy_from_slice(&crop_hwc[src..src + 3]);
                }
            }
        }
    }
    out
}

/// Pixel-space overlap multi-crop: an HWC image in, the global crop and the
/// `h·w` local crops out, both already patch-packed for the ViT.
///
/// Returns `(global_packed, locals_packed, plan)`. `global_packed` is one
/// crop's `[ppc, patch_vec]` (the whole image resized to `crop_size`);
/// `locals_packed` is `[h·w·ppc, patch_vec]` in `(tile_y, tile_x)` order, which
/// is the order [`reconstruct_from_crops`] stitches them back in.
///
/// The resize is the shared `imaging::host::resize_bilinear_hwc`, not a local
/// sampler - `crates/imaging` exists because five copies of that loop did not
/// agree.
pub fn overlap_crop_image(hwc: &[f32], img_w: u32, img_h: u32, cfg: &VisionConfig) -> (Vec<f32>, Vec<f32>, CropPlan) {
    assert_eq!(hwc.len(), (img_w * img_h * 3) as usize, "overlap_crop_image: expected [h, w, 3]");
    let plan = plan_crops(cfg, img_h, img_w);
    let side = cfg.crop_size;

    // The global view is the whole image at one crop's resolution.
    let global = patchify_crop(&imaging::host::resize_bilinear_hwc(hwc, 3, img_w, img_h, side, side), side, cfg.patch);

    // Locals are cut from the image resized so the crop lattice lands exactly.
    let big = imaging::host::resize_bilinear_hwc(hwc, 3, img_w, img_h, plan.resized_w, plan.resized_h);
    let mut locals = Vec::with_capacity((plan.h_tiles * plan.w_tiles) as usize * global.len());
    for ty in 0..plan.h_tiles {
        for tx in 0..plan.w_tiles {
            let (y0, x0) = (ty * plan.stride_px, tx * plan.stride_px);
            let mut crop = vec![0.0f32; (side * side * 3) as usize];
            for y in 0..side {
                let src = (((y0 + y) * plan.resized_w) + x0) as usize * 3;
                let dst = (y * side) as usize * 3;
                crop[dst..dst + (side * 3) as usize].copy_from_slice(&big[src..src + (side * 3) as usize]);
            }
            locals.extend(patchify_crop(&crop, side, cfg.patch));
        }
    }
    (global, locals, plan)
}

#[cfg(test)]
mod crop_tests {
    use super::*;

    fn preview_vision() -> VisionConfig {
        crate::config::MoondreamConfig::preview().vision
    }

    /// THE INVARIANT THAT TIES THE TWO HALVES TOGETHER: the resized image must
    /// be exactly the crop lattice, in the same units the feature-space stitch
    /// uses. `reconstruct_from_crops` produces
    /// `(grid - 2·margin)·tiles + 2·margin` PATCHES; the plan must produce that
    /// many pixels. If these drift, the crops and the stitch describe different
    /// images and nothing errors - the shapes still line up.
    #[test]
    fn crop_plan_round_trips_the_feature_space_geometry() {
        let v = preview_vision();
        for (h, w) in [(378, 378), (800, 600), (1024, 768), (2000, 500), (100, 100)] {
            let p = plan_crops(&v, h, w);
            let out_h_patches = (v.grid() - 2 * v.overlap_margin) * p.h_tiles + 2 * v.overlap_margin;
            let out_w_patches = (v.grid() - 2 * v.overlap_margin) * p.w_tiles + 2 * v.overlap_margin;
            assert_eq!(p.resized_h, out_h_patches * v.patch, "{h}x{w}: resized height is not the stitch's own height");
            assert_eq!(p.resized_w, out_w_patches * v.patch, "{h}x{w}: resized width is not the stitch's own width");
            assert!(p.h_tiles * p.w_tiles <= v.max_crops, "{h}x{w}: {} tiles exceeds max_crops", p.h_tiles * p.w_tiles);
        }
    }

    /// Every local crop must lie wholly inside the resized image - the last one
    /// is the tight case, and an off-by-one stride would slice past the end.
    #[test]
    fn the_last_crop_ends_exactly_at_the_image_edge() {
        let v = preview_vision();
        for (h, w) in [(800, 600), (1024, 768), (2000, 500)] {
            let p = plan_crops(&v, h, w);
            assert_eq!((p.h_tiles - 1) * p.stride_px + p.crop_px, p.resized_h, "{h}x{w}: vertical lattice does not close");
            assert_eq!((p.w_tiles - 1) * p.stride_px + p.crop_px, p.resized_w, "{h}x{w}: horizontal lattice does not close");
        }
    }

    /// Shapes out of `overlap_crop_image` are exactly what `SiglipEncoder::encode`
    /// takes, and the tile count matches the plan.
    #[test]
    fn cropping_produces_the_encoders_own_packed_layout() {
        // A tiny vision config so the test is fast; same code path.
        let v = VisionConfig { dim: 8, patch: 2, n_layers: 1, ff_dim: 16, n_heads: 2, crop_size: 8, max_crops: 4, overlap_margin: 1 };
        let (w, h) = (13u32, 9u32);
        let img: Vec<f32> = (0..(w * h * 3) as usize).map(|i| (i % 17) as f32 / 17.0).collect();
        let (global, locals, plan) = overlap_crop_image(&img, w, h, &v);
        let (ppc, pv) = (v.patches_per_crop() as usize, v.patch_vec() as usize);
        assert_eq!(global.len(), ppc * pv, "the global crop must be one crop's worth");
        assert_eq!(locals.len(), (plan.h_tiles * plan.w_tiles) as usize * ppc * pv);
        assert!(global.iter().chain(&locals).all(|v| v.is_finite()));
    }

    /// `patchify` must lay a patch out as `(y, x, channel)` within its row, and
    /// put patch `(py, px)` at row `py·grid + px`. A transposed variant has the
    /// same length and produces a plausible image embedding.
    #[test]
    fn patchify_is_patch_major_with_yxc_inside_a_patch() {
        // 4x4 image, patch 2 -> 2x2 grid of 2x2 patches. Channel 0 carries the
        // pixel index so each position is identifiable.
        let side = 4u32;
        let mut img = vec![0.0f32; (side * side * 3) as usize];
        for i in 0..(side * side) as usize {
            img[i * 3] = i as f32;
        }
        let out = patchify_crop(&img, side, 2);
        let pv = 3 * 2 * 2;
        // Patch (0,0) covers pixels 0,1,4,5 in (y,x) order.
        assert_eq!([out[0], out[3], out[6], out[9]], [0.0, 1.0, 4.0, 5.0]);
        // Patch (0,1) is the NEXT row of the packed output and covers 2,3,6,7.
        assert_eq!([out[pv], out[pv + 3], out[pv + 6], out[pv + 9]], [2.0, 3.0, 6.0, 7.0]);
        // Patch (1,0) is row 2 and covers 8,9,12,13.
        assert_eq!([out[2 * pv], out[2 * pv + 3]], [8.0, 9.0]);
    }
}
