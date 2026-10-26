// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resolution-flexible vision encoders for
// clients who read documents at the size their pages really are. If your team
// needs expertise in running a ViT at a patch grid other than the one it was
// trained on then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Running the tower at a patch grid other than its checkpoint's.
//!
//! DeepSeek-OCR's tiles are 640x640 (v1) or 768x768 (v2) crops fed to a tower
//! whose absolute position table is 64x64. The reference resamples that table to
//! the tile's grid on every forward (`get_abs_pos_sam`: bicubic, **antialiased**,
//! `align_corners = False`). The relative-position tables need nothing here: the
//! graph resamples them itself (`SamViTConfig::rel_pos_table_rows`).
//!
//! [`GridSource`] applies that resample once, at load: it is a
//! [`checkpoint::TensorSource`] over the native checkpoint that serves the
//! tower's other tensors untouched and `vision.sam.pos_embed` at the new grid,
//! so a tower for another grid is a second `SamEncoder` built from the same
//! weights with `grid_h`/`grid_w` changed - no per-forward resample, no second
//! copy of the weights on disk.

use checkpoint::TensorSource;

/// The brain-side name of the absolute position table, `[grid_h, grid_w, d_model]`.
pub const POS_EMBED: &str = "vision.sam.pos_embed";

/// `src` is `[from_h, from_w, c]` row-major; the result is `[to_h, to_w, c]`:
/// torch's `F.interpolate(mode="bicubic", antialias=True, align_corners=False)`
/// per channel (`imaging::host::resize_aa_planar`, the one implementation of it).
pub fn resample_pos_embed(src: &[f32], from: (usize, usize), to: (usize, usize), c: usize) -> Vec<f32> {
    assert_eq!(src.len(), from.0 * from.1 * c, "position table is not {}x{}x{c}", from.0, from.1);
    if from == to {
        return src.to_vec();
    }
    let planar = imaging::pixels::hwc_to_chw(src, c, from.0, from.1);
    let resized = imaging::host::resize_aa_planar(&planar, c, from.0, from.1, to.0, to.1, imaging::host::AaFilter::Bicubic);
    imaging::pixels::chw_to_hwc(&resized, c, to.0, to.1)
}

/// The checkpoint's tower tensors, with the absolute position table resampled
/// from its native `from` grid to `to`. Every other tensor is served as is.
pub struct GridSource<'a> {
    inner: &'a dyn TensorSource,
    from: (usize, usize),
    to: (usize, usize),
    d_model: usize,
}

impl<'a> GridSource<'a> {
    pub fn new(inner: &'a dyn TensorSource, from: (u32, u32), to: (u32, u32), d_model: u32) -> GridSource<'a> {
        GridSource { inner, from: (from.0 as usize, from.1 as usize), to: (to.0 as usize, to.1 as usize), d_model: d_model as usize }
    }
}

impl TensorSource for GridSource<'_> {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        self.with_tensor_chunks(name, usize::MAX, &mut |_, d| f(d))
    }
    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        if name == POS_EMBED {
            return None;
        }
        self.inner.raw_words(name)
    }
    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        if name != POS_EMBED {
            return self.inner.with_tensor_chunks(name, max_elems, f);
        }
        // The table is a few MB and the resample needs all of it at once.
        self.inner.with_tensor_chunks(name, usize::MAX, &mut |_, src| f(0, &resample_pos_embed(src, self.from, self.to, self.d_model)))
    }
    fn numel(&self, name: &str) -> Option<usize> {
        if name == POS_EMBED {
            return Some(self.to.0 * self.to.1 * self.d_model);
        }
        self.inner.numel(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Resampling to the native grid is the identity, and a constant table stays
    /// constant at any grid: the weights are normalised.
    #[test]
    fn identity_and_constants_are_preserved() {
        let src: Vec<f32> = (0..4 * 4 * 3).map(|i| (i as f32 * 0.37).sin()).collect();
        assert_eq!(resample_pos_embed(&src, (4, 4), (4, 4), 3), src);
        let flat = vec![0.75f32; 8 * 8 * 2];
        for to in [(5, 5), (3, 6), (12, 12)] {
            for v in resample_pos_embed(&flat, (8, 8), to, 2) {
                assert!((v - 0.75).abs() < 1e-6, "{to:?}: {v}");
            }
        }
    }

    /// A linear ramp is reproduced (bicubic interpolation is exact on degree-1
    /// data away from the borders), which pins the kernel's centre and scale: a
    /// half-pixel shift or a wrong stretch would bend it.
    #[test]
    fn a_ramp_stays_a_ramp() {
        let (n, m) = (64usize, 40usize);
        let src: Vec<f32> = (0..n * n).map(|i| (i % n) as f32).collect();
        let out = resample_pos_embed(&src, (n, n), (m, m), 1);
        let scale = n as f32 / m as f32;
        for x in 4..m - 4 {
            let want = (x as f32 + 0.5) * scale - 0.5;
            assert!((out[(m / 2) * m + x] - want).abs() < 2e-2, "x={x}: {} vs {want}", out[(m / 2) * m + x]);
        }
    }

    /// The source serves the resampled table under its own name and every other
    /// tensor untouched; the declared element count follows the new grid.
    #[test]
    fn the_source_resamples_only_the_position_table() {
        let mut m: HashMap<String, Vec<f32>> = HashMap::new();
        m.insert(POS_EMBED.into(), vec![1.0; 4 * 4 * 2]);
        m.insert("vision.sam.patch_embed.bias".into(), vec![3.0, 4.0]);
        let g = GridSource::new(&m, (4, 4), (2, 2), 2);
        assert_eq!(g.numel(POS_EMBED), Some(8));
        let mut got = Vec::new();
        assert!(g.with_tensor_chunks(POS_EMBED, usize::MAX, &mut |_, d| got = d.to_vec()));
        assert_eq!(got.len(), 8);
        let mut bias = Vec::new();
        assert!(g.with_tensor_chunks("vision.sam.patch_embed.bias", usize::MAX, &mut |_, d| bias = d.to_vec()));
        assert_eq!(bias, vec![3.0, 4.0]);
    }
}
