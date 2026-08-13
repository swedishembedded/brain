// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Assembling the **real interleaved image block**: the projector's
//! `[projector_rows, d_model]` output plus the two learned vectors
//! (`vision.image_newline`, `vision.view_separator`), gathered into the
//! `[n_rows, d_model]` slab the decoder actually splices.
//!
//! [`crate::rows`] says which decoder row carries which vector; this module is
//! what puts a vector there. The whole thing is **one row-table gather**, so it
//! adds no kernel:
//!
//! ```text
//!   table  = [ projector_out (P rows) | image_newline | view_separator ]   [P+2, d]
//!   idx[r] = i  for Src::Projector(i),  P for Src::Newline,  P+1 for Src::Separator
//!   block  = table[idx]                                                    [n_rows, d]
//! ```
//!
//! The table is assembled with three `splice` copies (the same
//! `model::vlm::splice_fwd` seam the decoder's own image splice uses - a
//! compact source into a destination at an element offset) and gathered with
//! `embed`, exactly the ABI `model::vit::gather_rows` documents.
//!
//! ## The adjoint, and why it is two different kernels
//!
//! The index map is **not** a permutation: 16 rows of a real 273-row block all
//! read `image_newline`. So the adjoint splits by row kind:
//!
//! * **Projector rows** - each projector row is used **exactly once**
//!   ([`RowGather::new`] asserts it), so restricted to them the map is a
//!   bijection and its adjoint is again a *gather*: `d_proj[i] = d_block[r(i)]`
//!   over the inverse index. One `embed` dispatch, `P*d` threads, no inner
//!   loop, no clear, no read-modify-write.
//! * **The two learned vectors** - a broadcast parameter's gradient is the SUM
//!   over every row that read it (16 rows for the newline, 1 for the
//!   separator), which is exactly what `emb_bwd` computes and **accumulates**.
//!   That is the same shape as `model::vit::RelPosAxis::build_bwd`'s table
//!   adjoint, and it carries the same contract: the destination is a parameter
//!   gradient, so the caller's `zero_grads` clears it, not this module.
//!
//! Using `row_scatter` for the projector half instead would work too, but
//! `emb_bwd`'s `[vocab, d]`-threaded scan costs `P*d*n_rows` comparisons where
//! the inverse gather costs `P*d` reads, so the gather is what ships.
//!
//! ## Scope
//!
//! One image, one contiguous decoder run, any [`Src`] sequence - the multi-view
//! (Base/Gundam) layouts [`crate::rows::row_plan`] can already produce work
//! here unchanged, because nothing below knows about tiles. What still limits
//! the composite to the global view is `deepseekv2`'s single-run splice, not
//! this file.

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::vlm::splice_fwd;

use crate::rows::Src;

/// The three kernels this module dispatches, in the order
/// [`RowGatherIds::STANDALONE`] indexes them. A model that already registers
/// them elsewhere in its own pipeline list passes its own indices instead.
pub const LAYOUT_PIPELINES: &[(&str, &str)] =
    &[("splice", kernels::SPLICE), ("embed", kernels::EMBED), ("emb_bwd", kernels::EMB_BWD)];

/// Pipeline indices of [`LAYOUT_PIPELINES`] inside the owner's kernel set.
#[derive(Clone, Copy, Debug)]
pub struct RowGatherIds {
    /// `splice` - table assembly (compact source, offset destination).
    pub splice: usize,
    /// `embed` - the row gather, forward AND the projector half of the adjoint.
    pub embed: usize,
    /// `emb_bwd` - the accumulating scatter onto the two shared vectors.
    pub emb_bwd: usize,
}

impl RowGatherIds {
    /// Indices when [`LAYOUT_PIPELINES`] is the WHOLE kernel set, in order -
    /// what a test registers; `DeepEncoder` appends them to its own list and
    /// passes the offset indices instead.
    pub const STANDALONE: RowGatherIds = RowGatherIds { splice: 0, embed: 1, emb_bwd: 2 };
}

/// Buffers and index vectors for one image's row layout.
///
/// Built once per `(layout, d_model)` and reused across forwards - the index
/// vectors are pure host index math and never change, so re-uploading them per
/// step would be the only cost this whole path has.
pub struct RowGather {
    /// `[P+2, d]` - projector output, then the newline row, then the separator.
    table: DeviceBuffer,
    /// `[n_rows, d]` - the assembled block, the decoder's splice input.
    block: DeviceBuffer,
    /// `[n_rows, d]` - the block's gradient, the decoder's splice output.
    d_block: DeviceBuffer,
    /// `[P, d]` - the projector's gradient, `DeepEncoder::backward`'s input.
    d_proj: DeviceBuffer,
    /// `u32[n_rows]` - layout row -> table row.
    idx: DeviceBuffer,
    /// `u32[P]` - projector row -> the ONE layout row that reads it.
    inv_proj: DeviceBuffer,
    /// `u32[n_rows]` - `0` on a newline row, `1` (out of a 1-row table) else.
    idx_newline: DeviceBuffer,
    /// `u32[n_rows]` - `0` on the separator row, `1` else.
    idx_separator: DeviceBuffer,
    n: u32,
    p: u32,
    d: u32,
    newline_rows: u32,
    separator_rows: u32,
}

impl RowGather {
    /// Build the layout's device state.
    ///
    /// `rows` is [`crate::rows::RowPlan::rows`] (or any [`Src`] sequence);
    /// `projector_rows` is what the encoder actually produces
    /// (`DeepseekOcrConfig::image_tokens()` for a single view). Every
    /// `Src::Projector(i)` must satisfy `i < projector_rows` and must appear
    /// **exactly once** - that bijection is what makes the adjoint a gather
    /// rather than a scatter, and a layout that dropped or duplicated a token
    /// row would otherwise silently lose or double its gradient.
    pub fn new(g: &Gpu, rows: &[Src], projector_rows: u32, d_model: u32) -> RowGather {
        assert!(!rows.is_empty(), "an image block must have at least one row");
        assert!(projector_rows > 0 && d_model > 0, "projector_rows {projector_rows} and d_model {d_model} must be positive");
        let (n, p, d) = (rows.len() as u32, projector_rows, d_model);

        let mut idx = Vec::with_capacity(rows.len());
        let mut inv = vec![u32::MAX; p as usize];
        let (mut newline_rows, mut separator_rows) = (0u32, 0u32);
        let mut idx_newline = Vec::with_capacity(rows.len());
        let mut idx_separator = Vec::with_capacity(rows.len());
        for (r, src) in rows.iter().enumerate() {
            // `1` is deliberately OUT of the 1-row destination `emb_bwd` scans,
            // so a row of the other kind contributes to neither vector.
            let (nl, sep) = (matches!(src, Src::Newline), matches!(src, Src::Separator));
            idx_newline.push(if nl { 0 } else { 1 });
            idx_separator.push(if sep { 0 } else { 1 });
            newline_rows += u32::from(nl);
            separator_rows += u32::from(sep);
            idx.push(match *src {
                Src::Projector(i) => {
                    assert!(i < p, "layout row {r} names projector row {i}, but the encoder produces {p}");
                    assert_eq!(inv[i as usize], u32::MAX, "projector row {i} is used by more than one layout row");
                    inv[i as usize] = r as u32;
                    i
                }
                Src::Newline => p,
                Src::Separator => p + 1,
            });
        }
        if let Some(i) = inv.iter().position(|v| *v == u32::MAX) {
            panic!("projector row {i} is used by no layout row: its gradient would be silently dropped");
        }

        RowGather {
            table: g.storage((p as u64 + 2) * d as u64),
            block: g.storage(n as u64 * d as u64),
            d_block: g.storage(n as u64 * d as u64),
            d_proj: g.storage(p as u64 * d as u64),
            idx: index_buffer(g, "rowgather.idx", &idx),
            inv_proj: index_buffer(g, "rowgather.inv_proj", &inv),
            idx_newline: index_buffer(g, "rowgather.idx_newline", &idx_newline),
            idx_separator: index_buffer(g, "rowgather.idx_separator", &idx_separator),
            n,
            p,
            d,
            newline_rows,
            separator_rows,
        }
    }

    /// Rows of the assembled block (the decoder run's width).
    pub fn rows(&self) -> u32 {
        self.n
    }
    /// Projector rows the encoder must supply.
    pub fn projector_rows(&self) -> u32 {
        self.p
    }
    /// `n_rows * d_model` - the element count of [`Self::block`].
    pub fn block_len(&self) -> usize {
        (self.n * self.d) as usize
    }
    /// `projector_rows * d_model`.
    pub fn proj_len(&self) -> usize {
        (self.p * self.d) as usize
    }
    /// How many rows read `image_newline` / `view_separator` - the number of
    /// terms each of those two gradients sums.
    pub fn shared_row_counts(&self) -> (u32, u32) {
        (self.newline_rows, self.separator_rows)
    }
    /// `[n_rows, d_model]` - the assembled block.
    pub fn block(&self) -> &DeviceBuffer {
        &self.block
    }
    /// `[n_rows, d_model]` - where the caller writes the block's gradient.
    pub fn d_block(&self) -> &DeviceBuffer {
        &self.d_block
    }
    /// `[projector_rows, d_model]` - the de-interleaved projector gradient.
    pub fn d_proj(&self) -> &DeviceBuffer {
        &self.d_proj
    }

    /// Assemble the table and gather the block.
    ///
    /// `proj_out` is `[projector_rows, d_model]`; `newline` and `separator` are
    /// the two `[d_model]` learned vectors (weight buffers, not host copies -
    /// they never leave the device on this path).
    pub fn build_fwd(
        &self,
        g: &Gpu,
        ids: RowGatherIds,
        proj_out: &DeviceBuffer,
        newline: &DeviceBuffer,
        separator: &DeviceBuffer,
        steps: &mut Vec<Step>,
    ) {
        let (p, d) = (self.p, self.d);
        steps.push(splice_fwd(g, ids.splice, proj_out, &self.table, 0, p * d));
        steps.push(splice_fwd(g, ids.splice, newline, &self.table, p * d, d));
        steps.push(splice_fwd(g, ids.splice, separator, &self.table, (p + 1) * d, d));
        // `embed` ABI, as `model::vit::gather_rows` documents it:
        // `dst[r, :] = src[idx[r], :]`, Params `[d_model, seq_len]`.
        steps.push(g.step(ids.embed, &[&self.idx, &self.table, &self.block], &[d, self.n], self.n * d));
    }

    /// Adjoint of [`Self::build_fwd`], reading [`Self::d_block`].
    ///
    /// `d_newline` / `d_separator` are the two vectors' **parameter gradient**
    /// buffers and are ACCUMULATED onto (`emb_bwd` adds), so the caller clears
    /// them with the rest of its gradients exactly once per step.
    /// [`Self::d_proj`] is fully overwritten and needs no clear.
    pub fn build_bwd(
        &self,
        g: &Gpu,
        ids: RowGatherIds,
        d_newline: &DeviceBuffer,
        d_separator: &DeviceBuffer,
        steps: &mut Vec<Step>,
    ) {
        let (n, p, d) = (self.n, self.p, self.d);
        // Projector rows: the restricted map is a bijection, so its adjoint is
        // the inverse GATHER -- one read per output element, no accumulation.
        steps.push(g.step(ids.embed, &[&self.inv_proj, &self.d_block, &self.d_proj], &[d, p], p * d));
        // The two shared vectors: sum over every row that read them.
        // `emb_bwd` Params `[n_rows, d_model, vocab]`, threads `vocab*d_model`.
        steps.push(g.step(ids.emb_bwd, &[&self.idx_newline, &self.d_block, d_newline], &[n, d, 1], d));
        steps.push(g.step(ids.emb_bwd, &[&self.idx_separator, &self.d_block, d_separator], &[n, d, 1], d));
    }
}

/// Upload a row-index vector as the `u32` buffer `embed`/`emb_bwd` bind - the
/// same thing `model::vit::row_index_buffer` does for the ViT permutations.
fn index_buffer(g: &Gpu, label: &str, idx: &[u32]) -> DeviceBuffer {
    let b = g.buffer(label, idx.len() as u64 * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST);
    g.write(&b, idx);
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rows::{row_plan, ViewGrid};
    use data::rng::Lcg;

    /// A deliberately small layout that still has every row kind: `g = 3` over
    /// the global view is 3 token rows of 3, each followed by a newline, then
    /// the separator -- 13 rows over 9 projector rows.
    const G: u32 = 3;
    const D: u32 = 5;

    struct Fixture {
        gpu: Gpu,
        rg: RowGather,
        proj: DeviceBuffer,
        newline: DeviceBuffer,
        separator: DeviceBuffer,
        d_newline: DeviceBuffer,
        d_separator: DeviceBuffer,
        rows: Vec<Src>,
    }

    impl Fixture {
        fn new() -> Fixture {
            let gpu = gpu_core::testgpu::dev(LAYOUT_PIPELINES);
            let rows = row_plan(G, ViewGrid::global_only()).rows;
            let p = G * G;
            let rg = RowGather::new(&gpu, &rows, p, D);
            Fixture {
                proj: gpu.storage((p * D) as u64),
                newline: gpu.storage(D as u64),
                separator: gpu.storage(D as u64),
                d_newline: gpu.storage(D as u64),
                d_separator: gpu.storage(D as u64),
                gpu,
                rg,
                rows,
            }
        }

        /// Run the forward and return the assembled `[n_rows, d]` block.
        fn forward(&self) -> Vec<f32> {
            let mut steps = Vec::new();
            self.rg.build_fwd(&self.gpu, RowGatherIds::STANDALONE, &self.proj, &self.newline, &self.separator, &mut steps);
            self.gpu.submit(&[], &steps);
            self.gpu.read(self.rg.block(), self.rg.block_len())
        }

        /// Run the backward on `d_block`, clearing both shared gradients first,
        /// and return `(d_proj, d_newline, d_separator)`.
        fn backward(&self, d_block: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            self.gpu.write_f32(self.rg.d_block(), d_block);
            let mut steps = Vec::new();
            self.rg.build_bwd(&self.gpu, RowGatherIds::STANDALONE, &self.d_newline, &self.d_separator, &mut steps);
            self.gpu.submit(&[&self.d_newline, &self.d_separator], &steps);
            (
                self.gpu.read(self.rg.d_proj(), self.rg.proj_len()),
                self.gpu.read(&self.d_newline, D as usize),
                self.gpu.read(&self.d_separator, D as usize),
            )
        }
    }

    fn ramp(n: usize, base: f32) -> Vec<f32> {
        (0..n).map(|i| base + i as f32).collect()
    }

    /// Every assembled row must be **exactly** its source row -- this is a copy,
    /// not a computed value, so the assertion is equality and not a tolerance.
    /// Checked row by row against the plan rather than in aggregate: a swapped
    /// newline and separator, or an off-by-one in the projector index, still
    /// produces a block of the right shape with the right multiset of rows.
    #[test]
    fn every_assembled_row_is_exactly_its_source_row() {
        let f = Fixture::new();
        let p = (G * G) as usize;
        let proj = ramp(p * D as usize, 1.0);
        let newline: Vec<f32> = (0..D).map(|c| -100.0 - c as f32).collect();
        let separator: Vec<f32> = (0..D).map(|c| 1000.0 + c as f32).collect();
        f.gpu.write_f32(&f.proj, &proj);
        f.gpu.write_f32(&f.newline, &newline);
        f.gpu.write_f32(&f.separator, &separator);

        let block = f.forward();
        assert_eq!(block.len(), f.rg.block_len());
        assert_eq!(f.rg.rows(), G * (G + 1) + 1);
        assert_eq!(f.rg.shared_row_counts(), (G, 1));
        let d = D as usize;
        for (r, src) in f.rows.iter().enumerate() {
            let got = &block[r * d..(r + 1) * d];
            let want: &[f32] = match *src {
                Src::Projector(i) => &proj[i as usize * d..(i as usize + 1) * d],
                Src::Newline => &newline,
                Src::Separator => &separator,
            };
            assert_eq!(got, want, "row {r} ({src:?}) is not its source row");
        }
    }

    /// The adjoint: projector rows land back where they came from, and each
    /// shared vector's gradient is the SUM over every row that read it -- 3
    /// newline rows here, 1 separator row. Exact arithmetic, so exact equality.
    #[test]
    fn the_adjoint_scatters_projector_rows_and_sums_the_shared_vectors() {
        let f = Fixture::new();
        let d = D as usize;
        let d_block = ramp(f.rg.block_len(), 0.5);
        let (d_proj, d_nl, d_sep) = f.backward(&d_block);

        let mut want_nl = vec![0f32; d];
        let mut want_sep = vec![0f32; d];
        for (r, src) in f.rows.iter().enumerate() {
            let row = &d_block[r * d..(r + 1) * d];
            match *src {
                Src::Projector(i) => assert_eq!(&d_proj[i as usize * d..(i as usize + 1) * d], row, "projector row {i}"),
                Src::Newline => want_nl.iter_mut().zip(row).for_each(|(a, b)| *a += *b),
                Src::Separator => want_sep.iter_mut().zip(row).for_each(|(a, b)| *a += *b),
            }
        }
        assert_eq!(d_nl, want_nl, "image_newline's gradient is not the sum over its {G} rows");
        assert_eq!(d_sep, want_sep, "view_separator's gradient is not its row");
        // Not a tautology: a gradient that summed only the LAST newline row
        // would equal that row, so assert the sum really is bigger than one term.
        let last_nl = f.rows.iter().rposition(|s| *s == Src::Newline).expect("a newline row");
        assert_ne!(d_nl, d_block[last_nl * d..(last_nl + 1) * d].to_vec(), "only one newline row was accumulated");
    }

    /// The two shared gradients ACCUMULATE across backwards without an
    /// intervening clear -- the contract `build_bwd` documents, and the reason
    /// the caller's `zero_grads` (not this module) owns clearing them.
    #[test]
    fn the_shared_gradients_accumulate_until_the_caller_clears_them() {
        let f = Fixture::new();
        let d_block = ramp(f.rg.block_len(), 0.25);
        let (_, once, _) = f.backward(&d_block);
        // Second pass with NO clear in the submit.
        f.gpu.write_f32(f.rg.d_block(), &d_block);
        let mut steps = Vec::new();
        f.rg.build_bwd(&f.gpu, RowGatherIds::STANDALONE, &f.d_newline, &f.d_separator, &mut steps);
        f.gpu.submit(&[], &steps);
        let twice = f.gpu.read(&f.d_newline, D as usize);
        for (a, b) in twice.iter().zip(&once) {
            assert_eq!(*a, 2.0 * *b, "a second backward did not accumulate");
        }
    }

    /// Directional finite-difference check of the backward against the forward,
    /// over all three inputs at once -- the house gradcheck shape
    /// (`gradcheck::directional_check`), specialised here because the fixture is
    /// three leaf buffers and a linear objective rather than a `CheckModel`.
    ///
    /// `L = <r, block>` for a fixed random `r`, so `d(block) = r` seeds the
    /// backward directly and the numeric derivative along `v` is
    /// `(L(x + eps*v) - L(x - eps*v)) / (2*eps)`.
    #[test]
    fn the_backward_matches_a_finite_difference_of_the_forward() {
        let f = Fixture::new();
        let mut rng = Lcg::new(7);
        let mut rand = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.signed()).collect() };
        let (pl, bl, dl) = (f.rg.proj_len(), f.rg.block_len(), D as usize);
        let x0 = [rand(pl), rand(dl), rand(dl)];
        let bufs = [&f.proj, &f.newline, &f.separator];
        let names = ["projector_out", "image_newline", "view_separator"];
        let dir: Vec<Vec<f32>> = [pl, dl, dl].iter().map(|n| rand(*n)).collect();
        let r = rand(bl);

        let loss = |x: &[Vec<f32>]| -> f64 {
            for (b, v) in bufs.iter().zip(x) {
                f.gpu.write_f32(b, v);
            }
            let block = f.forward();
            block.iter().zip(&r).map(|(a, b)| *a as f64 * *b as f64).sum()
        };
        // Analytic: one backward seeded with d(block) = r.
        loss(&x0);
        let (d_proj, d_nl, d_sep) = f.backward(&r);
        let grads = [d_proj, d_nl, d_sep];

        let eps = 1e-3f32;
        for (k, name) in names.iter().enumerate() {
            let analytic: f64 = grads[k].iter().zip(&dir[k]).map(|(g, v)| *g as f64 * *v as f64).sum();
            let mut xp = x0.clone();
            let mut xm = x0.clone();
            for i in 0..x0[k].len() {
                xp[k][i] = x0[k][i] + eps * dir[k][i];
                xm[k][i] = x0[k][i] - eps * dir[k][i];
            }
            let numeric = (loss(&xp) - loss(&xm)) / (2.0 * eps as f64);
            let rel = (analytic - numeric).abs() / analytic.abs().max(numeric.abs()).max(1e-3);
            println!("  {name:<14} analytic {analytic:+.6e}  numeric {numeric:+.6e}  rel {rel:.3e}");
            assert!(rel < 1e-3, "{name}: analytic {analytic} vs numeric {numeric} (rel {rel})");
            assert!(analytic.abs() > 1e-3, "{name}: a ~zero directional derivative proves nothing");
        }
    }

    /// A layout that drops or duplicates a projector row is refused at
    /// construction: silently losing a token row's gradient is the failure this
    /// whole index map can have and nothing downstream would see.
    #[test]
    fn a_layout_that_does_not_use_every_projector_row_exactly_once_is_refused() {
        let gpu = gpu_core::testgpu::dev(LAYOUT_PIPELINES);
        // `AssertUnwindSafe`: `Gpu` owns a boxed backend and is not
        // `RefUnwindSafe`, but nothing here observes it after the panic.
        let refused = |rows: &[Src]| -> String {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = RowGather::new(&gpu, rows, 2, D);
            }))
            .expect_err("this layout should have been refused");
            e.downcast_ref::<String>().cloned().unwrap_or_default()
        };
        let msg = refused(&[Src::Projector(0), Src::Newline, Src::Projector(0)]);
        assert!(msg.contains("more than one layout row"), "got {msg:?}");
        let msg = refused(&[Src::Projector(0), Src::Separator]);
        assert!(msg.contains("used by no layout row"), "got {msg:?}");
        let msg = refused(&[Src::Projector(5)]);
        assert!(msg.contains("but the encoder produces 2"), "got {msg:?}");
    }
}
