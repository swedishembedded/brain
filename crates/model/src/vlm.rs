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
//! indices from its own PIPELINES list. Image tokens are contiguous per image
//! (or per view of a sliced image), so a run of `n = n_img_rows * d_model`
//! values starting at `base = row0 * d_model` is spliced with one call;
//! [`SpliceRuns`] owns the list of runs and their buffers.

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

/// The image-token runs of one example and the device buffers they are
/// spliced from: each `(row0, n_rows)` run of the residual stream is
/// overwritten by its own image's rows on the forward, and its gradient is
/// moved out (and zeroed in the residual grad) on the backward.
///
/// One owner for the bookkeeping every decoder with a splice needs (run
/// validation, per-run buffers, the concatenated host view, the step lists),
/// so a decoder only says where in its graph the steps go. A prompt with a
/// sliced image has many runs - one per view - which is why runs are a list.
pub struct SpliceRuns {
    runs: Vec<(u32, u32)>,
    img: Vec<DeviceBuffer>,
    d_img: Vec<DeviceBuffer>,
    d_model: u32,
}

impl SpliceRuns {
    /// No splice: every method is a no-op and the step lists are empty.
    pub fn none() -> SpliceRuns {
        SpliceRuns { runs: Vec::new(), img: Vec::new(), d_img: Vec::new(), d_model: 0 }
    }

    /// Allocate one input and one gradient buffer per run. The runs are given
    /// in the order their rows are concatenated in [`Self::write`] and
    /// [`Self::read_grad`]; they must be non-empty and must not overlap.
    pub fn new(g: &Gpu, runs: &[(u32, u32)], d_model: u32) -> SpliceRuns {
        assert!(!runs.is_empty(), "SpliceRuns: at least one image run");
        assert!(runs.iter().all(|&(_, n)| n > 0), "SpliceRuns: an empty image run in {runs:?}");
        let mut sorted = runs.to_vec();
        sorted.sort_unstable();
        assert!(sorted.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0), "SpliceRuns: the image runs {runs:?} overlap");
        let buffers = || runs.iter().map(|&(_, n)| g.storage((n * d_model) as u64)).collect::<Vec<_>>();
        SpliceRuns { img: buffers(), d_img: buffers(), runs: runs.to_vec(), d_model }
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// The `(row0, n_rows)` runs, in concatenation order.
    pub fn runs(&self) -> &[(u32, u32)] {
        &self.runs
    }

    /// Total spliced elements (`sum(n_rows) * d_model`).
    pub fn numel(&self) -> usize {
        self.runs.iter().map(|&(_, n)| (n * self.d_model) as usize).sum()
    }

    /// Write every run's image rows, concatenated in run order.
    pub fn write(&self, g: &Gpu, data: &[f32]) {
        assert_eq!(data.len(), self.numel(), "SpliceRuns::write: the rows of every run, concatenated");
        let mut at = 0;
        for (&(_, n), buf) in self.runs.iter().zip(&self.img) {
            let len = (n * self.d_model) as usize;
            g.write_f32(buf, &data[at..at + len]);
            at += len;
        }
    }

    /// Every run's image-row gradient after a backward, concatenated in run
    /// order.
    pub fn read_grad(&self, g: &Gpu) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.numel());
        for (&(_, n), buf) in self.runs.iter().zip(&self.d_img) {
            out.extend(g.read(buf, (n * self.d_model) as usize));
        }
        out
    }

    /// The input buffer of run `i`, for a producer on the same device that
    /// writes it with a step instead of through the host.
    pub fn img_buffer(&self, i: usize) -> &DeviceBuffer {
        &self.img[i]
    }

    /// The gradient buffer of run `i`.
    pub fn grad_buffer(&self, i: usize) -> &DeviceBuffer {
        &self.d_img[i]
    }

    /// Forward steps overwriting `res`'s run rows, for a residual stream whose
    /// row 0 is absolute row `row_offset` (0 for a whole-sequence forward, the
    /// chunk's first position for a chunked one). Runs outside
    /// `[row_offset, row_offset + rows)` are skipped; a run crossing the
    /// window's edge is an error, since its rows would land half outside.
    pub fn fwd(&self, g: &Gpu, splice: usize, res: &DeviceBuffer, row_offset: u32, rows: u32) -> Vec<Step> {
        let d = self.d_model;
        self.runs
            .iter()
            .zip(&self.img)
            .filter(|(&(row0, n), _)| row0 + n > row_offset && row0 < row_offset + rows)
            .map(|(&(row0, n), img)| {
                assert!(
                    row0 >= row_offset && row0 + n <= row_offset + rows,
                    "SpliceRuns: run ({row0}, {n}) crosses the window [{row_offset}, {})",
                    row_offset + rows
                );
                splice_fwd(g, splice, img, res, (row0 - row_offset) * d, n * d)
            })
            .collect()
    }

    /// Backward steps moving each run's rows of `d_res` (a whole-sequence
    /// residual grad) into its gradient buffer and zeroing them there.
    pub fn bwd(&self, g: &Gpu, splice_bwd_pipeline: usize, d_res: &DeviceBuffer) -> Vec<Step> {
        let d = self.d_model;
        self.runs.iter().zip(&self.d_img).map(|(&(row0, n), d_img)| splice_bwd(g, splice_bwd_pipeline, d_res, d_img, row0 * d, n * d)).collect()
    }
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

    #[test]
    fn splice_runs_overwrite_each_run_and_return_their_gradients_in_run_order() {
        // seq=6, d_model=2; runs given out of row order: rows [4,6) then [1,2).
        let gpu = Gpu::new_cpu(&[("splice", kernels::SPLICE), ("splice_bwd", kernels::SPLICE_BWD)]);
        let runs = SpliceRuns::new(&gpu, &[(4, 2), (1, 1)], 2);
        assert_eq!(runs.numel(), 6);
        runs.write(&gpu, &[40., 41., 50., 51., 10., 11.]);
        let res = gpu.storage_init("res", &[0.; 12]);
        gpu.submit(&[], &runs.fwd(&gpu, 0, &res, 0, 6));
        assert_eq!(gpu.read(&res, 12), vec![0., 0., 10., 11., 0., 0., 0., 0., 40., 41., 50., 51.]);

        // A chunk holding rows [3,6) only splices the run inside it, at
        // chunk-relative rows.
        let chunk = gpu.storage_init("chunk", &[0.; 6]);
        gpu.submit(&[], &runs.fwd(&gpu, 0, &chunk, 3, 3));
        assert_eq!(gpu.read(&chunk, 6), vec![0., 0., 40., 41., 50., 51.]);

        let d_res = gpu.storage_init("d_res", &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
        gpu.submit(&[], &runs.bwd(&gpu, 1, &d_res));
        assert_eq!(runs.read_grad(&gpu), vec![9., 10., 11., 12., 3., 4.]);
        assert_eq!(gpu.read(&d_res, 12), vec![1., 2., 0., 0., 5., 6., 7., 8., 0., 0., 0., 0.]);
    }

    #[test]
    #[should_panic(expected = "overlap")]
    fn overlapping_runs_are_refused() {
        let gpu = Gpu::new_cpu(&[("splice", kernels::SPLICE), ("splice_bwd", kernels::SPLICE_BWD)]);
        SpliceRuns::new(&gpu, &[(0, 3), (2, 2)], 4);
    }
}
