// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Vision-language embedding-splice step-builders — the shared seam every VLM
//! composes. After the decoder's text token-embedding gather fills the residual
//! stream `res[0]`, the projected image-token embeddings (produced by a vision
//! encoder + connector) are written over the placeholder rows; the backward
//! routes those rows' gradient to the connector output and zeros them in the
//! residual grad so the token-embedding backward (`emb_bwd`) never scatters them
//! into the placeholder token's `tok.weight` row.
//!
//! These are pure dispatch assembly (like [`crate::block`]): no ParamStore, no
//! buffer ownership. The two kernels are `splice` / `splice_bwd`
//! (`crates/kernels/wgsl/splice{,_bwd}.wgsl`); a model supplies their pipeline
//! indices from its own PIPELINES list. Image tokens are contiguous per image,
//! so a run of `n = n_img_rows * d_model` values starting at `base = row0 *
//! d_model` is spliced with one call — emit one call per image run.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Forward splice: `res[base .. base+n] = img[0 .. n]`.
///
/// `img` is the compact `[n_img_rows, d_model]` connector output; `res` is the
/// decoder residual stream `[seq, d_model]`. `base = row0 * d_model`,
/// `n = n_img_rows * d_model`.
pub fn splice_fwd(g: &Gpu, splice: usize, img: &DeviceBuffer, res: &DeviceBuffer, base: u32, n: u32) -> Step {
    // splice.wgsl bindings: (1) src=img, (2) dst=res; params (n, base).
    g.step(splice, &[img, res], &[n, base], n)
}

/// Backward splice: `d_img[0 .. n] = d_res[base .. base+n]`, then zero
/// `d_res[base .. base+n]`.
///
/// `d_res` is the residual-stream grad (read-write; the spliced region is zeroed
/// in place); `d_img` receives the compact image-embedding grad that flows on
/// into the connector backward.
pub fn splice_bwd(g: &Gpu, splice_bwd: usize, d_res: &DeviceBuffer, d_img: &DeviceBuffer, base: u32, n: u32) -> Step {
    // splice_bwd.wgsl bindings: (1) d_dst=d_res, (2) d_src=d_img; params (n, base).
    g.step(splice_bwd, &[d_res, d_img], &[n, base], n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::Gpu;

    #[test]
    fn splice_fwd_overwrites_rows_bwd_extracts_and_zeros() {
        // seq=4, d_model=3. Splice 2 image rows into residual rows [1,2]
        // (base = row0·d = 3, n = 2·d = 6). CPU backend so it runs headless.
        let gpu = Gpu::new_cpu(&[("splice", kernels::SPLICE), ("splice_bwd", kernels::SPLICE_BWD)]);
        let (base, n) = (3u32, 6u32);

        // Forward: rows 1,2 overwritten by the image embeds; rows 0,3 untouched.
        let res = gpu.storage_init("res", &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]);
        let img = gpu.storage_init("img", &[100., 101., 102., 200., 201., 202.]);
        gpu.submit(&[], &[splice_fwd(&gpu, 0, &img, &res, base, n)]);
        assert_eq!(gpu.read(&res, 12), vec![0., 1., 2., 100., 101., 102., 200., 201., 202., 9., 10., 11.]);

        // Backward: the spliced region's grad is moved into d_img and zeroed in
        // d_res (so emb_bwd never sees it); rows 0,3 keep their grad.
        let d_res = gpu.storage_init("d_res", &[0.5, 0.5, 0.5, 1., 2., 3., 4., 5., 6., 0.5, 0.5, 0.5]);
        let d_img = gpu.storage(6);
        gpu.submit(&[], &[splice_bwd(&gpu, 1, &d_res, &d_img, base, n)]);
        assert_eq!(gpu.read(&d_img, 6), vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(gpu.read(&d_res, 12), vec![0.5, 0.5, 0.5, 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.5]);
    }
}
