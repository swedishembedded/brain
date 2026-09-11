// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reference-image conditioning: the **one** construction of the joint token
//! sequence a FLUX.2 DiT evaluation runs on, shared by generation
//! ([`crate::pipeline`]) and by training ([`crate::modelgrad`],
//! [`crate::finetune`]).
//!
//! FLUX.2 conditions on a reference photograph by VAE-encoding it and
//! **concatenating** its latent tokens onto the image half of the joint
//! sequence, with a distinct t-axis RoPE id (`10·(i+1)` for reference `i`) that
//! separates them from the generated image's own axis. Nothing about that is
//! implied by the weights: it is a convention, held in two places if it is
//! written twice.
//!
//! It *was* written twice. Generation built reference tokens and their position
//! ids; training built neither - `make_flow_batch` passed an empty reference
//! list to `position_ids`, and `Cfg` had no slot for a reference at all - so an
//! adapter trained on a `--ref` workflow was fitted against an input
//! distribution the deployed path never presents. The two halves of that bug
//! (the token layout and the latent packing) are the two halves of this module,
//! and both callers go through them:
//!
//! * [`JointLayout`] - how many tokens of which kind, in which order, with
//!   which position ids and therefore which RoPE tables. [`crate::position_ids`]
//!   is this type's [`JointLayout::ids`].
//! * [`pack_tokens`] - a VAE latent mean to packed DiT tokens `[lh·lw, cin]`,
//!   the transform `Pipeline::encode_image` and `finetune::encode_samples` each
//!   used to spell out for themselves.
//!
//! The layout is also where an evaluation smaller than the canvas gets its
//! reference rows. A reference encoded at the canvas's own token grid is the
//! picture being edited, not a photograph of something else, so a window of the
//! canvas is conditioned on the matching window of IT
//! ([`JointLayout::register_aligned_refs`], [`RefGrid`]) - the whole reference
//! on every window is the same "no signal for which part of the scene this is"
//! failure window-local position ids would be.
//!
//! Swedish Embedded AB implements train/inference-consistent conditioning for
//! diffusion models for its clients. If your team needs expertise in keeping a
//! training pipeline and its deployed sampler provably on the same input
//! distribution, you can procure our services by sending an email to
//! info@swedishembedded.com.

/// One reference image's rows in ONE evaluation of the joint sequence: which
/// slice of that reference the evaluation carries, and where the slice sits on
/// the reference itself.
///
/// A whole-canvas forward carries every reference whole. So does a window of a
/// canvas - **except** for a reference that is spatially *registered* to the
/// canvas ([`RefGrid::registered`]), which is cropped to the window, because
/// for such a reference the token at `(y, x)` is a photograph of the very
/// canvas cell `(y, x)` the window is painting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefGrid {
    /// The reference's own latent grid `(h, w)` in tokens - the WHOLE
    /// photograph, whatever slice of it this evaluation carries. The token
    /// buffer [`JointLayout::joint_tokens`] is handed is always this big, so a
    /// reference is encoded once per request and never per window.
    pub full: (usize, usize),
    /// Top-left token of the slice this evaluation carries, on the reference's
    /// own grid. `(0, 0)` for a whole reference.
    pub origin: (usize, usize),
    /// Extent of the slice, in tokens; `full` for a whole reference.
    pub lh: usize,
    pub lw: usize,
    /// True when this reference's own token grid IS the canvas grid, token for
    /// token, so reference token `(y, x)` and generated token `(y, x)` describe
    /// the same place in the same framing - the `--ref`/`--strength` editing
    /// case, where the reference is the picture being edited rather than a
    /// separate photograph being described.
    ///
    /// It is what [`JointLayout::window`] reads: a registered reference is
    /// cropped to the window, an unregistered one is repeated whole.
    pub registered: bool,
}

impl RefGrid {
    /// The whole of an `h x w` reference, unregistered - the plain conditioning
    /// reference every window of a tiled canvas carries in full.
    pub fn whole(h: usize, w: usize) -> RefGrid {
        RefGrid { full: (h, w), origin: (0, 0), lh: h, lw: w, registered: false }
    }

    /// Tokens this slice contributes to the joint sequence.
    pub fn n(&self) -> usize {
        self.lh * self.lw
    }

    /// Tokens the whole reference holds - the size of the encoded buffer this
    /// slice is gathered out of.
    pub fn n_full(&self) -> usize {
        self.full.0 * self.full.1
    }
}

/// `lh x lw` tokens starting at `(y0, x0)` gathered raster-major out of a grid
/// that is `stride` tokens wide, each token `cin` floats.
///
/// The one gather in this module: a window of the canvas latent and a crop of a
/// registered reference are the same operation on different buffers, and a
/// second spelling of it is a tile that quietly carries somebody else's pixels.
fn gather(src: &[f32], stride: usize, (y0, x0): (usize, usize), lh: usize, lw: usize, cin: usize, out: &mut Vec<f32>) {
    for y in 0..lh {
        let row = ((y0 + y) * stride + x0) * cin;
        out.extend_from_slice(&src[row..row + lw * cin]);
    }
}

/// The joint token layout of ONE FLUX.2 DiT evaluation: `txt_len` text rows,
/// then the generated image's `lh×lw` latent tokens, then each reference
/// image's own latent grid, in the order the references were supplied.
///
/// This ordering is not a detail: the head reads the FIRST `n_gen` image rows
/// (`Flux2Model::forward_batch`'s `n_pred`), so a layout that put references
/// first would predict a velocity for the photograph instead of for the image
/// being generated.
///
/// An evaluation need not cover the whole canvas. [`JointLayout::window`]
/// returns the layout of one tile of a larger canvas: same text rows, the same
/// type and therefore the same [`JointLayout::ids`] and [`JointLayout::rope`]
/// construction - `lh×lw` shrinks to the window, [`JointLayout::origin`]
/// records where on the canvas it sits, and a reference that is registered to
/// the canvas ([`JointLayout::register_aligned_refs`]) narrows to the same
/// region while an unregistered one comes across whole. That is what lets tiled
/// generation and reference conditioning compose instead of being two parallel
/// spellings of the same convention.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JointLayout {
    /// Text conditioning rows (the model's fixed text window).
    pub txt_len: usize,
    /// The generated image's latent grid **for this evaluation**, in tokens
    /// (pixels / 16 per axis). The whole canvas for an untiled run and for all
    /// training; one window of it under [`JointLayout::window`].
    pub lh: usize,
    pub lw: usize,
    /// Where this evaluation's grid sits on the canvas it belongs to, as the
    /// `(y, x)` latent token of its top-left corner. `(0, 0)` for a
    /// whole-canvas evaluation.
    ///
    /// This is a RoPE property, not bookkeeping: the image rows are given the
    /// ids they hold on the CANVAS, so a window at `(32, 0)` gets exactly what
    /// a single full-canvas forward would have assigned that region. Ids local
    /// to the window would put every window's content at the canvas origin and
    /// each would compose its own independent scene.
    pub origin: (usize, usize),
    /// Each reference image's contribution, in sequence order. Empty is the
    /// plain caption-only / text-to-image layout.
    pub refs: Vec<RefGrid>,
}

impl JointLayout {
    /// A layout with no reference images - text-to-image, and caption-only
    /// training.
    pub fn unpaired(txt_len: usize, lh: usize, lw: usize) -> JointLayout {
        JointLayout { txt_len, lh, lw, origin: (0, 0), refs: Vec::new() }
    }

    /// A layout conditioned on `refs`, each a latent grid `(h, w)` in tokens,
    /// none of them registered to the canvas - every window carries every one
    /// of them whole. [`JointLayout::register_aligned_refs`] is what decides
    /// otherwise, and only a caller that knows the canvas is in a position to.
    pub fn with_refs(txt_len: usize, lh: usize, lw: usize, refs: Vec<(usize, usize)>) -> JointLayout {
        JointLayout { txt_len, lh, lw, origin: (0, 0), refs: refs.into_iter().map(|(h, w)| RefGrid::whole(h, w)).collect() }
    }

    /// This canvas layout with every reference whose own token grid IS the
    /// canvas grid marked [`RefGrid::registered`], so [`JointLayout::window`]
    /// crops those to the window instead of repeating them whole.
    ///
    /// Grid equality is the whole criterion, and it is not a heuristic: the
    /// reference ids `(10·(i+1), h, w, 0)` put reference token `(h, w)` at the
    /// same spatial RoPE phase as generated token `(h, w)`, so a reference at
    /// the canvas's own token grid is *already* registered to it by
    /// construction - cropping only stops repeating the parts of it a window is
    /// not painting. A reference at any other grid (an unrelated photograph, or
    /// the first one under a `ref_resolution_scale` below 1) has no such
    /// correspondence and is left whole.
    ///
    /// Whole-canvas evaluations are untouched by this: their single window is
    /// the canvas, so every crop is the whole reference again.
    pub fn register_aligned_refs(mut self) -> JointLayout {
        let canvas = (self.lh, self.lw);
        for r in &mut self.refs {
            r.registered = r.full == canvas;
        }
        self
    }

    /// The layout of one `th×tw` **window** of this canvas, whose top-left
    /// latent token is the canvas token `(y0, x0)`.
    ///
    /// The text rows come across untouched: they describe the CONDITIONING,
    /// which is a property of what the image is of and not of which part of it
    /// is being predicted.
    ///
    /// So does an *unregistered* reference, for the same reason - a photograph
    /// of something else is what every window of the canvas is painting from.
    /// A **registered** reference is not in that position: it is the canvas, at
    /// the canvas's own framing, so the part of it that belongs to this window
    /// is this window's region of it and nothing else. Handing every window the
    /// whole registered reference is what made a tiled declutter render the
    /// same sofa in every tile - each window was told "this is the room" with no
    /// signal for which corner of the room it was responsible for, and each
    /// painted a plausible whole room. The crop keeps its canvas-absolute ids
    /// (`RefGrid::origin` rides along), so the tokens a window sees carry
    /// exactly the positions one full-canvas forward would have given them.
    ///
    /// The offsets are canvas-absolute, so windowing a window is not a
    /// composition of offsets and is not what this is for; take windows of the
    /// canvas layout.
    pub fn window(&self, y0: usize, x0: usize, th: usize, tw: usize) -> JointLayout {
        debug_assert!(y0 + th <= self.lh && x0 + tw <= self.lw, "window must lie inside the canvas");
        let refs = self
            .refs
            .iter()
            .map(|r| if r.registered { RefGrid { origin: (y0, x0), lh: th, lw: tw, ..*r } } else { *r })
            .collect();
        JointLayout { txt_len: self.txt_len, lh: th, lw: tw, origin: (y0, x0), refs }
    }

    /// Tokens of the image being generated - the rows the head predicts a
    /// velocity for, and the only rows the flow-matching loss is defined on.
    pub fn n_gen(&self) -> usize {
        self.lh * self.lw
    }

    /// Conditioning tokens the references contribute **to this evaluation** -
    /// the crops a window carries, not the photographs they came from.
    pub fn n_ref(&self) -> usize {
        self.refs.iter().map(RefGrid::n).sum()
    }

    /// Tokens the references hold in FULL - the size of the encoded buffer
    /// [`JointLayout::joint_tokens`] gathers from, which is the whole
    /// photograph however little of it this evaluation carries.
    pub fn n_ref_full(&self) -> usize {
        self.refs.iter().map(RefGrid::n_full).sum()
    }

    /// Rows in the image half of the joint sequence: generated **plus**
    /// reference. Attention, the image embedder and the RoPE tables are sized
    /// in these; the head is not.
    pub fn n_img(&self) -> usize {
        self.n_gen() + self.n_ref()
    }

    /// Rows in the whole joint sequence.
    pub fn n(&self) -> usize {
        self.txt_len + self.n_img()
    }

    /// The 4-axis position ids, text rows first.
    ///
    /// Text tokens: `(0,0,0,l)`; generated image: `(0, y₀+h, x₀+w, 0)`
    /// raster-major, where `(y₀, x₀)` is [`JointLayout::origin`] - the position
    /// the token holds on the canvas, which for a whole-canvas layout is just
    /// `(0, h, w, 0)`; reference `i`: `(10·(i+1), h, w, 0)`. The t-axis offset
    /// is what keeps a reference token from colliding with the generated token
    /// at the same spatial position.
    ///
    /// A reference's `(h, w)` is its position on the REFERENCE - which for a
    /// registered reference cropped to a window ([`JointLayout::window`]) means
    /// `(ry₀+h, rx₀+w)` from [`RefGrid::origin`], the same canvas-absolute rule
    /// the image rows follow. A crop is then bit-for-bit the rows one
    /// full-canvas forward would have put at those positions, ids and all.
    pub fn ids(&self) -> Vec<u32> {
        let (y0, x0) = self.origin;
        let mut ids = Vec::with_capacity(self.n() * 4);
        for l in 0..self.txt_len {
            ids.extend([0, 0, 0, l as u32]);
        }
        for h in 0..self.lh {
            for w in 0..self.lw {
                ids.extend([0, (y0 + h) as u32, (x0 + w) as u32, 0]);
            }
        }
        for (i, r) in self.refs.iter().enumerate() {
            let t = 10 * (i as u32 + 1);
            let (ry0, rx0) = r.origin;
            for h in 0..r.lh {
                for w in 0..r.lw {
                    ids.extend([t, (ry0 + h) as u32, (rx0 + w) as u32, 0]);
                }
            }
        }
        ids
    }

    /// The interleaved-RoPE cos/sin tables for this layout, `[n · head_dim/2]`
    /// each.
    pub fn rope(&self, axes_dim: [usize; 4], theta: f64) -> dit::rope::RopeTables {
        let rc = dit::rope::RopeConfig {
            axes_dims: axes_dim.iter().map(|&a| a as u32).collect(),
            axes_lens: vec![4096, 4096, 4096, 4096],
            theta,
        };
        dit::rope::tables_for_ids(&rc, &self.ids(), 4)
    }

    /// The image half of the joint sequence: the generated image's tokens
    /// followed by every reference's, `[n_img · cin]`.
    ///
    /// `gen` is `[n_gen · cin]` (the noised latent under training, the current
    /// latent under sampling) and `refs` is the concatenation of every
    /// reference's **whole** tokens in layout order, `[n_ref_full · cin]` -
    /// exactly what repeated [`pack_tokens`] calls produce, encoded once per
    /// request.
    ///
    /// Which slice of each reference reaches the sequence is this layout's
    /// business, not the caller's: a registered reference is gathered at
    /// [`RefGrid::origin`] for this window, an unregistered one is copied
    /// whole, and a whole-canvas layout copies all of them whole. The caller
    /// therefore hands over the same buffer for every window and cannot hand
    /// one window a reference the next one is not looking at.
    pub fn joint_tokens(&self, gen: &[f32], refs: &[f32], cin: usize) -> Vec<f32> {
        assert_eq!(gen.len(), self.n_gen() * cin, "generated tokens");
        assert_eq!(refs.len(), self.n_ref_full() * cin, "reference tokens");
        let mut out = Vec::with_capacity(self.n_img() * cin);
        out.extend_from_slice(gen);
        let mut base = 0usize;
        for r in &self.refs {
            let whole = &refs[base * cin..(base + r.n_full()) * cin];
            gather(whole, r.full.1, r.origin, r.lh, r.lw, cin, &mut out);
            base += r.n_full();
        }
        out
    }

    /// This layout's generated rows gathered out of a whole-canvas latent
    /// `[canvas_lh · canvas_lw · cin]` that is `canvas_lw` tokens wide.
    ///
    /// Raster-major within the window, which is the order [`JointLayout::ids`]
    /// hands out this window's ids in - the gather and the ids are written next
    /// to each other because a window whose rows are collected in one order and
    /// positioned in another is not a crash, it is a tile that quietly renders
    /// somebody else's part of the picture. A whole-canvas layout gathers the
    /// canvas unchanged.
    pub fn window_tokens(&self, canvas: &[f32], canvas_lw: usize, cin: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_gen() * cin);
        gather(canvas, canvas_lw, self.origin, self.lh, self.lw, cin, &mut out);
        out
    }
}

/// A VAE latent **mean** at an `lh8×lw8` 8x-downsampled grid to packed DiT
/// tokens `[(lh8/2)·(lw8/2), cin]`, row-major - the token order
/// [`JointLayout::ids`] assigns position ids in.
///
/// One implementation for generation and training. It is two steps that must
/// agree exactly and are easy to spell differently: `vae::latent::pack` folds
/// the 2×2 pixel-unshuffle and the eval-BatchNorm affine into `[cin, lh, lw]`,
/// and this transposes that to token-major. A transposed second copy is not a
/// crash, it is an adapter trained on a channel-shuffled image.
pub fn pack_tokens(
    mean: &[f32],
    lh8: usize,
    lw8: usize,
    bn_mean: &[f32],
    bn_var: &[f32],
    eps: f32,
    cin: usize,
) -> Vec<f32> {
    // The latent is `cin/4` channels before the 2x2 unshuffle packs it to `cin`.
    let packed = vae::latent::pack(mean, cin / 4, lh8, lw8, bn_mean, bn_var, eps);
    let (lh, lw) = (lh8 / 2, lw8 / 2);
    let mut tokens = vec![0.0f32; lh * lw * cin];
    for c in 0..cin {
        for y in 0..lh {
            for x in 0..lw {
                tokens[(y * lw + x) * cin + c] = packed[(c * lh + y) * lw + x];
            }
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The t-axis offset is the whole point of a reference id: without it a
    /// reference token and the generated token at the same (h, w) would carry
    /// identical position ids and therefore identical RoPE phase, and the
    /// model could not tell the photograph from the canvas.
    #[test]
    fn reference_ids_carry_the_t_axis_offset() {
        let l = JointLayout::with_refs(2, 2, 2, vec![(2, 2), (1, 1)]);
        let ids = l.ids();
        assert_eq!(ids.len(), l.n() * 4);
        let row = |i: usize| &ids[i * 4..(i + 1) * 4];
        assert_eq!(row(0), [0, 0, 0, 0]); // text
        assert_eq!(row(1), [0, 0, 0, 1]);
        assert_eq!(row(2), [0, 0, 0, 0]); // generated (0,0)
        assert_eq!(row(5), [0, 1, 1, 0]); // generated (1,1)
        assert_eq!(row(6), [10, 0, 0, 0]); // reference 0 (0,0)
        assert_eq!(row(9), [10, 1, 1, 0]);
        assert_eq!(row(10), [20, 0, 0, 0]); // reference 1
    }

    /// Counting is the sizing contract every buffer in the trainer and the
    /// denoiser is allocated from; the head's `n_pred` is `n_gen`, never
    /// `n_img`.
    #[test]
    fn the_counts_separate_generated_from_conditioning_rows() {
        let l = JointLayout::with_refs(3, 4, 5, vec![(2, 2), (1, 3)]);
        assert_eq!(l.n_gen(), 20);
        assert_eq!(l.n_ref(), 7);
        assert_eq!(l.n_img(), 27);
        assert_eq!(l.n(), 30);
        let u = JointLayout::unpaired(3, 4, 5);
        assert_eq!((u.n_gen(), u.n_ref(), u.n_img(), u.n()), (20, 0, 20, 23));
    }

    /// A window of a canvas crops the reference that IS the canvas and repeats
    /// the one that is not, and the crop carries the ids and the token values
    /// the whole-canvas evaluation gave that same region. Cropping is a slice
    /// of the already-encoded reference, so the buffer handed in is the whole
    /// photograph either way.
    #[test]
    fn a_window_crops_a_registered_reference_and_repeats_an_unregistered_one() {
        let cin = 2;
        let canvas = JointLayout::with_refs(1, 4, 4, vec![(4, 4), (2, 2)]).register_aligned_refs();
        assert_eq!([canvas.refs[0].registered, canvas.refs[1].registered], [true, false]);
        // Token `(y, x)` of reference 0 is `[y, x]`; reference 1 is negative.
        let refs: Vec<f32> = (0..4)
            .flat_map(|y| (0..4).flat_map(move |x| [y as f32, x as f32]))
            .chain((0..2).flat_map(|y| (0..2).flat_map(move |x| [-(y as f32), -(x as f32)])))
            .collect();
        assert_eq!(refs.len(), canvas.n_ref_full() * cin);

        let win = canvas.window(2, 1, 2, 3);
        assert_eq!((win.n_gen(), win.n_ref()), (6, 6 + 4));
        let gen = vec![0.0f32; win.n_gen() * cin];
        let joint = win.joint_tokens(&gen, &refs, cin);
        let ids = win.ids();
        assert_eq!(joint.len(), win.n_img() * cin);
        assert_eq!(ids.len(), win.n() * 4);

        // The registered reference's crop: its own region, at canvas-absolute
        // ids - byte for byte what the whole-canvas layout carries there.
        let whole = canvas.joint_tokens(&vec![0.0f32; canvas.n_gen() * cin], &refs, cin);
        let (wids, tail) = (canvas.ids(), canvas.n_gen());
        for y in 0..2 {
            for x in 0..3 {
                let mine = win.n_gen() + y * 3 + x;
                let theirs = tail + (2 + y) * 4 + (1 + x);
                assert_eq!(&joint[mine * cin..][..cin], &whole[theirs * cin..][..cin], "crop ({y},{x})");
                assert_eq!(
                    &ids[(win.txt_len + mine) * 4..][..4],
                    &wids[(canvas.txt_len + theirs) * 4..][..4],
                    "crop id ({y},{x})"
                );
            }
        }
        // The unregistered reference is repeated whole, ids and all.
        let (m, t) = (win.n_gen() + 6, tail + 16);
        assert_eq!(&joint[m * cin..], &whole[t * cin..], "unregistered reference was cropped");
        assert_eq!(&ids[(win.txt_len + m) * 4..], &wids[(canvas.txt_len + t) * 4..]);
    }
}
